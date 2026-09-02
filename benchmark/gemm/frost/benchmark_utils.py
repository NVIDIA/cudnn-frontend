# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the gemm/frost benchmark scripts.

Each script builds its own `spec_map` (label -> (geometry cfg, cta_group))
from the registry funnel and its own graph / data, but the way they
select configs, rotate buffers, time launches and read an nsys report is the
same everywhere. That part lives here.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable

import torch

from cudnn.gemm.frost.tile_config import by_name as _by_name

# Config selection

LABEL_RE = re.compile(r"^(CONFIG_sm\d+_\d+x\d+x\d+_\d+x\d+x\d+_cluster\d+x\d+)_([12])ctamma$")


def spec_for(label, spec_map):
    """(geometry cfg, cta_group) for a --configs label, or None.

    The sweep map comes from the registry funnel over CATALOG; a label naming a
    geometry outside it (e.g. a num_mma_m > 1 tile, which `by_name` synthesizes) is
    still runnable, so parse it rather than calling it unsweepable."""
    spec = spec_map.get(label)
    if spec is not None:
        return spec
    m = LABEL_RE.match(label)
    if m is None:
        return None
    try:
        cfg = _by_name(m.group(1))
    except (KeyError, NotImplementedError):
        return None
    return cfg, int(m.group(2))


def select_configs(arg, spec_map):
    """--configs value -> concrete label list. A token carrying a glob
    (`CONFIG_sm107_*`) is expanded against the sweep map; anything else is kept
    verbatim so `spec_for` can still synthesize an off-catalog geometry."""
    if not arg:
        return list(spec_map)
    out = []
    for tok in (t.strip() for t in arg.split(",")):
        if not tok:
            continue
        if any(ch in tok for ch in "*?["):
            hits = [n for n in spec_map if fnmatch.fnmatchcase(n, tok)]
            if not hits:
                sys.exit(f"--configs pattern {tok!r} matched no config")
            out += hits
        else:
            out.append(tok)
    return list(dict.fromkeys(out))


# Argument surface

CONFIGS_HELP = "comma-separated config names or globs, e.g. 'CONFIG_sm107_*' (default: sweep all)"

ROTATE_HELP = (
    "allocate N independent copies of every tensor and rotate the timed launches "
    "across them so a kernel doesn't re-read the previous launch's data from a hot "
    "L2 (inflates small-shape TFLOPS). Warmup uses a separate dedicated buffer. "
    "Default 'auto': size the pool to exceed the L2 (large shapes -> 2, small -> "
    "scaled up), capped at 4 GB. Pass an integer to override; 1 disables."
)

STREAM_HELP = "print each config's result as soon as it finishes (the last 'running' line points at a hang)."


def add_sweep_args(parser, *, nsys: bool = True, warmup: int = 10, iters: int = 20):
    """The knobs every sweep script exposes. `nsys=False` for the fused benches,
    whose baseline is several kernels so a median-kernel-time mode is meaningless."""
    parser.add_argument("--warmup", type=int, default=warmup)
    # CLAUDE.md: keep iters <= 20 — more just lengthens the run / holds the GPU.
    parser.add_argument("--iters", type=int, default=iters)
    parser.add_argument("--configs", default=None, help=CONFIGS_HELP)
    parser.add_argument(
        "--timing",
        choices=("delayed", "events", "nsys") if nsys else ("delayed", "events"),
        default="delayed",
        help="delayed (default) hides host-launch overhead behind a _sleep; events is raw wall clock"
        + ("; nsys is per-kernel median from a profile" if nsys else ""),
    )
    parser.add_argument("--stream", action="store_true", help=STREAM_HELP)
    parser.add_argument("--rotate-buffers", default="auto", metavar="N", help=ROTATE_HELP)
    if nsys:
        parser.add_argument("--_nsys-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


# Buffer rotation (defeat a hot L2 on small shapes)

L2_BYTES = 126 * 1024 * 1024  # B200 class
_AUTO_POOL_BUDGET_BYTES = 4 * 1024 * 1024 * 1024
_AUTO_NBUF_CAP = 1024


def set_bytes(tensors) -> int:
    """GMEM footprint of one tensor set (packed FP4 counts as 1 byte per pair)."""
    return sum(t.numel() * t.element_size() for t in tensors)


def auto_nbuf(per_set_bytes: int, *, free_divisor: int = 2) -> int:
    """Smallest buffer count whose pool exceeds L2 (1.5x margin), clamped to a
    memory budget. Large shapes -> 2; small shapes -> scaled up until L2-cold."""
    nbuf = max(2, -(-int(1.5 * L2_BYTES) // per_set_bytes))
    budget = _AUTO_POOL_BUDGET_BYTES
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        budget = min(budget, free // free_divisor)
    return max(1, min(nbuf, max(1, budget // per_set_bytes), _AUTO_NBUF_CAP))


def resolve_nbuf(spec, per_set_bytes: int, *, free_divisor: int = 2) -> int:
    """Resolve --rotate-buffers: 'auto' -> shape-sized count, else the integer
    (1 = rotation disabled)."""
    if str(spec).strip().lower() == "auto":
        return auto_nbuf(per_set_bytes, free_divisor=free_divisor)
    return max(1, int(spec))


def rotating(fn_of_set: Callable, pool: list) -> Callable:
    """Wrap a `set -> None` callable into `i -> pool[i % len(pool)]`."""
    n = len(pool)
    return lambda i: fn_of_set(pool[i % n])


def report_pool(nbuf: int, per_set_bytes: int) -> None:
    """The one-line pool banner, plus the warning when the pool still fits in L2."""
    if nbuf <= 1:
        print("  [rotate-buffers: 1 (disabled) — small shapes read a hot L2 and over-report TFLOPS]")
        return
    footprint = per_set_bytes * nbuf
    print(f"  [rotate-buffers: {nbuf} copies/tensor, {footprint / 1024 / 1024:.0f} MB pool — defeats hot-L2 on small shapes]")
    if footprint < L2_BYTES:
        print(
            f"  [WARNING: pool ({footprint / 1024 / 1024:.0f} MB) < L2 (~{L2_BYTES / 1024 / 1024:.0f} MB) — "
            f"it fits in cache, so launches stay warm after one rotation. Bump --rotate-buffers.]"
        )


# Timing


def time_ms_events(timed_fn: Callable, warmup_fn: Callable, *, warmup: int, iters: int) -> float:
    """Wall-clock CUDA Event timing around a python loop. Inflated by
    Python+TVM-FFI dispatch overhead (~50us/call). `timed_fn(i)` rotates
    buffers; `warmup_fn()` uses a separate dedicated buffer."""
    for _ in range(warmup):
        warmup_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        timed_fn(i)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def time_ms_delayed(timed_fn: Callable, warmup_fn: Callable, *, warmup: int, iters: int) -> float:
    """Kernel-only timing: queue a long `_sleep` first so the host enqueues
    every launch behind it -> kernels run back-to-back with no host gaps.
    `timed_fn(i)` rotates buffers; `warmup_fn()` uses a separate buffer.

    The post-sleep warmup ramps SM clocks (DVFS) back up before timing —
    without it the first fast-config measurement inflates ~2x."""
    for _ in range(warmup):
        warmup_fn()
    torch.cuda.synchronize()
    # Delay must outlast iters x ~50us host overhead. B200 ~1.7 GHz; floor 1e8.
    torch.cuda._sleep(max(int(1e8), int((iters * 0.05 + 20.0) * 1.7e6)))
    for _ in range(max(5, warmup)):
        warmup_fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        timed_fn(i)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def time_ms(timed_fn: Callable, warmup_fn: Callable | None = None, *, warmup: int, iters: int, timing: str) -> float:
    """Dispatch on --timing. `timed_fn(i)` takes the rotation index (pass
    `lambda _i: ...` when there is nothing to rotate); `warmup_fn()` defaults to
    `timed_fn(0)`."""
    wf = warmup_fn if warmup_fn is not None else (lambda: timed_fn(0))
    fn = time_ms_delayed if timing == "delayed" else time_ms_events
    return fn(timed_fn, wf, warmup=warmup, iters=iters)


# nsys mode


def kernel_match_token(cfg, cta_group: int) -> str:
    """The substring the demangled symbol carries. `compiler` names the kernel
    `<template_file_stem>_<geometry_name>`, so it reads `..._1ctamma_128x256x128_...`.
    The --configs LABEL is not usable here at all: it spells the pipeline before
    the geometry and the cta_group after it, so it is never a substring."""
    return f"{cta_group}ctamma_{cfg.geometry_name}"


def parse_nsys_stats(text: str) -> dict[str, float]:
    """Parse `nsys stats --report cuda_gpu_kern_sum --format csv` into
    {kernel_name: median_ms}. CSV, not the console table: the column format
    truncates the symbol at 100 chars with an ellipsis, which cuts the geometry
    off the longer template names."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("Time (%),") and ln.rstrip().endswith(",Name"):
            header_i = i
            break
    else:
        sys.exit("could not find the cuda_gpu_kern_sum CSV header in nsys stats output:\n  " + "\n  ".join(lines[:60]))

    cols = next(csv.reader([lines[header_i]]))
    med = next((i for i, c in enumerate(cols) if re.fullmatch(r"Med \(\w+\)", c)), None)
    if med is None or "Name" not in cols:
        sys.exit(f"cuda_gpu_kern_sum CSV header has no Med (<unit>) / Name column: {lines[header_i]!r}")
    med_i, name_i = med, cols.index("Name")
    unit = re.fullmatch(r"Med \((\w+)\)", cols[med_i]).group(1)
    per_ms = {"ns": 1e6, "us": 1e3, "ms": 1.0, "s": 1e-3}.get(unit)
    if per_ms is None:
        sys.exit(f"unsupported time unit {unit!r} in the cuda_gpu_kern_sum CSV header: {lines[header_i]!r}")
    out: dict[str, float] = {}
    for row in csv.reader(io.StringIO("\n".join(lines[header_i + 1 :]))):
        if len(row) <= max(med_i, name_i):
            continue
        try:
            out[row[name_i]] = float(row[med_i].replace(",", "")) / per_ms
        except ValueError:
            continue
    return out


FROST_SYMBOL_MARKER = "cudnn_frost_"


def find_cublas_time(kern_times: dict[str, float]) -> tuple[str, float] | None:
    """The reference GEMM in the report: cuBLAS's `nvjet_*` if present, else the
    heaviest kernel that is neither one of ours nor a torch helper. The
    block-scale reference goes through `F.scaled_mm`, which dispatches to a
    `cutlass3x_sm100_bstensorop_*` kernel rather than nvjet."""
    cands = [(k, v) for k, v in kern_times.items() if k.startswith("nvjet_")]
    if not cands:
        cands = [(k, v) for k, v in kern_times.items() if FROST_SYMBOL_MARKER not in k and "at::" not in k]
    if not cands:
        return None
    return max(cands, key=lambda x: x[1])


def nsys_run_and_parse(script: str, inner_args: list[str], *, tag: str) -> dict[str, float]:
    """Re-exec `script` under nsys with `--_nsys-worker` + `inner_args`, then parse
    `cuda_gpu_kern_sum` for the median kernel time (ms). Returns {kernel_name: ms}."""
    nsys = "/usr/local/bin/nsys" if os.path.exists("/usr/local/bin/nsys") else shutil.which("nsys")
    if nsys is None:
        sys.exit("nsys not found — install nsight-systems or use the default events mode.")

    workdir = os.path.join(os.environ.get("TMPDIR", tempfile.gettempdir()), f"{tag}_nsys_{os.getpid()}")
    os.makedirs(workdir, exist_ok=True)
    report_prefix = os.path.join(workdir, "report")

    # Redirect nsys's default /tmp/nvidia path (root-owned on some hosts) via env.
    nsys_env = os.environ.copy()
    nsys_env.setdefault("TMPDIR", os.environ.get("TMPDIR", tempfile.gettempdir()))

    inner = [sys.executable, "-u", os.path.abspath(script), "--_nsys-worker"] + inner_args
    profile_cmd = [
        nsys,
        "profile",
        "-o",
        report_prefix,
        "--force-overwrite=true",
        "--cuda-um-cpu-page-faults=false",
        "--cuda-um-gpu-page-faults=false",
        "--trace=cuda",
    ] + inner
    print(f"  + {' '.join(profile_cmd)}\n")
    proc = subprocess.run(profile_cmd, capture_output=True, text=True, env=nsys_env)
    if proc.returncode != 0:
        print("nsys stdout:\n" + proc.stdout)
        print("nsys stderr:\n" + proc.stderr, file=sys.stderr)
        sys.exit(f"nsys profile exited {proc.returncode}")

    stats_cmd = [nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--force-export=true", report_prefix + ".nsys-rep"]
    proc = subprocess.run(stats_cmd, capture_output=True, text=True, env=nsys_env)
    if proc.returncode != 0:
        print("nsys stats stdout:\n" + proc.stdout)
        print("nsys stats stderr:\n" + proc.stderr, file=sys.stderr)
        sys.exit(f"nsys stats exited {proc.returncode}")
    return parse_nsys_stats(proc.stdout)


# Block-scale / MoE data helpers


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def to_blocked(x: torch.Tensor) -> torch.Tensor:
    """(rows, cols) scale tensor -> the flat F8_128x4 reordered blob, rows padded
    to 128 and cols to 4."""
    rows, cols = x.shape
    nrb, ncb = ceil_div(rows, 128), ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def rand_e8m0(shape, dev):
    """Random E8M0 scale bytes centred on 2^0 (biased exponent 127). The graph
    declares the SF as FP8_E8M0, so the blob must carry that dtype."""
    return torch.randint(125, 129, shape, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)


def group_offsets(S: int, E: int) -> torch.Tensor:
    """Even split, remainder absorbed by the last group: group g starts at
    g * (S // E). This is the `first_token_offset` tensor."""
    return torch.arange(E, dtype=torch.int32, device="cuda") * (S // E)


def even_offsets(S: int, E: int) -> list[int]:
    """Routed-group start offsets for ``S`` tokens spread as evenly as possible
    over ``E`` experts — the first ``S % E`` groups take one extra token.
    ``S=10, E=3`` -> group sizes 4, 3, 3 -> ``[0, 4, 7]``."""
    if E < 1:
        raise ValueError(f"expert count must be >= 1, got {E}")
    base, rem = divmod(S, E)
    offsets, start = [], 0
    for i in range(E):
        offsets.append(start)
        start += base + (1 if i < rem else 0)
    return offsets
