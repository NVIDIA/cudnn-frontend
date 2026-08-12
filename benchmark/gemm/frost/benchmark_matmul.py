# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark every CATALOG config on a single matmul shape vs cuBLAS.

`--shape` is `B,M,N,K` (B independent same-shape GEMMs; B=1 = plain matmul).
Timing modes: delayed (default) / nsys / events. `--rotate-buffers` defeats
hot-L2 inflation on small shapes.

    python benchmark/gemm/frost/benchmark_matmul.py --shape 1,8192,8192,8192
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable

import cudnn  # noqa: F401
import cudnn.gemm.frost  # noqa: F401
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.fusion_ir import FusionChain as _FC, MatmulSpec as _MS, OutputSpec as _OS
from cudnn.gemm.frost.kernel_registry import candidates as _candidates
from cudnn.gemm.frost.tile_config import by_name as _by_name


def _build_spec_map():
    """Legacy label -> (geometry cfg, cta_group, scheduler) for every sweepable
    matmul strategy, via the registry funnel. Labels reconstruct the old
    CONFIG_..._Nctamma[_static] form so --configs still accepts them."""
    chain = _FC(
        matmul=_MS(
            M=4096,
            N=4096,
            K=4096,
            a_major="k",
            b_major="k",
            a_dtype="bf16",
            b_dtype="bf16",
            accum_dtype="fp32",
        ),
        output_specs=[_OS(source_ref=-1, dtype="bf16")],
    )
    m = {}
    for t, cfg in _candidates(chain):
        label = f"{cfg.name}_{t.cta_group}ctamma" + ("_static" if t.static_sched else "")
        m[label] = (cfg, t.cta_group, t.scheduler)
    return m


_SPEC_MAP = _build_spec_map()

_LABEL_RE = re.compile(r"^(CONFIG_sm\d+_\d+x\d+x\d+_\d+x\d+x\d+_cluster\d+x\d+)_([12])ctamma(_static)?$")


def _spec_for(name):
    """(geometry cfg, cta_group, scheduler) for a --configs label, or None.

    The sweep set comes from the registry funnel over CATALOG; a label naming a
    geometry outside it (e.g. a num_mma_m > 1 tile, which `by_name` synthesizes) is
    still runnable, so parse it rather than reporting UNKNOWN_CONFIG."""
    spec = _SPEC_MAP.get(name)
    if spec is not None:
        return spec
    m = _LABEL_RE.match(name)
    if m is None:
        return None
    try:
        cfg = _by_name(m.group(1))
    except (KeyError, NotImplementedError):
        return None
    return cfg, int(m.group(2)), "static" if m.group(3) else "clc"


def _vp(handles, a, b, c):
    """Variant-pack dict {cuDNN tensor: buffer} keyed by the graph's tensors."""
    A, B, C = handles
    return {A: a, B: b, C: c}


def _build_plan(g, cfg, name):
    """JIT-compile the recorded graph with a forced tile config."""
    _, cta_group, scheduler = _spec_for(name)
    return jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group, scheduler=scheduler)


# ---------------------------------------------------------------------------
# Graph + data setup
# ---------------------------------------------------------------------------


def _graph_matmul(batch: int, M: int, N: int, K: int):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1])
    Bt = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=Bt, name="mm")
    C.set_output(True)
    return g, (A, Bt, C)


def _mkdata(batch: int, M: int, N: int, K: int):
    torch.manual_seed(0)
    a = torch.empty(batch, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(batch, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(batch, M, N, dtype=torch.bfloat16, device="cuda")
    return a, b, c


# ---------------------------------------------------------------------------
# Buffer rotation — defeat the hot-L2 artifact on small shapes
# ---------------------------------------------------------------------------
# Rotating launches across a pool of independent tensor copies (launch i uses
# pool[i % N]) forces DRAM reads once the pool exceeds L2 — otherwise a small
# matmul re-reads hot-L2 inputs and reports inflated TFLOPS.


# B200 L2 is ~126 MB; a pool smaller than this stays fully cached — warn to bump.
_L2_BYTES_B200 = 126 * 1024 * 1024


def _mkdata_pool(batch: int, M: int, N: int, K: int, nbuf: int):
    """`nbuf` independent (a, b, c) triples at distinct GMEM addresses (nbuf<=1
    returns the single base triple)."""
    a, b, c = _mkdata(batch, M, N, K)
    pool = [(a, b, c)]
    # Distinct allocations (clone → fresh GMEM); contents don't matter for timing.
    for _ in range(max(0, nbuf - 1)):
        pool.append((a.clone(), b.clone(), c.clone()))
    return pool


def _per_set_bytes(batch: int, M: int, N: int, K: int) -> int:
    # BF16 = 2 bytes/elem; a:(batch,M,K) b:(batch,N,K) c:(batch,M,N).
    return 2 * batch * (M * K + N * K + M * N)


def _pool_footprint_bytes(batch: int, M: int, N: int, K: int, nbuf: int) -> int:
    return _per_set_bytes(batch, M, N, K) * nbuf


# Cap the auto-sized pool so a large shape doesn't allocate needless copies.
_AUTO_POOL_BUDGET_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
_AUTO_NBUF_CAP = 1024


def _auto_nbuf(batch: int, M: int, N: int, K: int) -> int:
    """Smallest buffer count whose pool exceeds L2 (1.5× margin), clamped to a
    memory budget. Large shapes → 2; small shapes → scaled up until L2-cold."""
    per_set = _per_set_bytes(batch, M, N, K)
    target = int(1.5 * _L2_BYTES_B200)
    nbuf = max(2, -(-target // per_set))  # ceil-div

    # Cap at min(4 GB, half of currently-free GMEM).
    budget = _AUTO_POOL_BUDGET_BYTES
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        budget = min(budget, free // 2)
    max_by_budget = max(1, budget // per_set)

    return max(1, min(nbuf, max_by_budget, _AUTO_NBUF_CAP))


def _resolve_nbuf(spec: str, batch: int, M: int, N: int, K: int) -> int:
    """Resolve --rotate-buffers: 'auto' → shape-sized count, else the integer
    (1 = rotation disabled)."""
    if spec.strip().lower() == "auto":
        return _auto_nbuf(batch, M, N, K)
    return max(1, int(spec))


def _rotating(fn_of_buf: Callable, pool: list) -> Callable:
    """Wrap `(a,b,c) -> None` into `i -> None` selecting pool[i % len(pool)]."""
    n = len(pool)
    return lambda i: fn_of_buf(pool[i % n])


# ---------------------------------------------------------------------------
# Timing (events mode)
# ---------------------------------------------------------------------------


def _time_ms_events(
    timed_fn: Callable,
    warmup_fn: Callable,
    *,
    warmup: int,
    iters: int,
) -> float:
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


def _time_ms_delayed(
    timed_fn: Callable,
    warmup_fn: Callable,
    *,
    warmup: int,
    iters: int,
) -> float:
    """Kernel-only timing: queue a long `_sleep` first so the host enqueues
    every launch behind it → kernels run back-to-back with no host gaps.
    `timed_fn(i)` rotates buffers; `warmup_fn()` uses a separate buffer.

    The post-sleep warmup ramps SM clocks (DVFS) back up before timing —
    without it the first fast-config measurement inflates ~2×."""
    for _ in range(warmup):
        warmup_fn()
    torch.cuda.synchronize()

    # Delay must outlast iters × ~50us host overhead. B200 ~1.7 GHz; floor 1e8.
    delay_cycles = max(
        int(1e8),
        int((iters * 0.05 + 20.0) * 1.7e6),
    )
    torch.cuda._sleep(delay_cycles)

    # Post-sleep warmup (behind the sleep) ramps clocks before timing.
    post_warmup = max(5, warmup)
    for _ in range(post_warmup):
        warmup_fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        timed_fn(i)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# nsys mode
# ---------------------------------------------------------------------------


def _nsys_run_and_parse(
    shape: str,
    configs: list[str],
    warmup: int,
    iters: int,
    nbuf: int,
) -> dict[str, float]:
    """Re-exec self under nsys, parse `cuda_gpu_kern_sum` for median kernel
    time (ms) per config. Returns {config_name_or_'cuBLAS': median_ms}."""
    nsys = "/usr/local/bin/nsys" if os.path.exists("/usr/local/bin/nsys") else shutil.which("nsys")
    if nsys is None:
        sys.exit("nsys not found — install nsight-systems or use the default events mode.")

    workdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"benchmark_matmul_nsys_{os.getpid()}")
    os.makedirs(workdir, exist_ok=True)
    report_prefix = os.path.join(workdir, "report")

    # Redirect nsys's default /tmp/nvidia path (root-owned on this host) via env.
    nsys_env = os.environ.copy()
    nsys_env.setdefault("TMPDIR", os.environ.get("TMPDIR", tempfile.gettempdir()))

    # Inner command — same script with --_nsys-worker.
    inner = [
        sys.executable,
        "-u",
        os.path.abspath(__file__),
        "--_nsys-worker",
        "--shape",
        shape,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--rotate-buffers",
        str(nbuf),
    ]
    if configs:
        inner += ["--configs", ",".join(configs)]

    # Step 1: profile (record only; --stats stdout is unreliable across versions).
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

    # Step 2: extract the kernel-summary table.
    stats_cmd = [
        nsys,
        "stats",
        "--report",
        "cuda_gpu_kern_sum",
        "--force-export=true",
        report_prefix + ".nsys-rep",
    ]
    proc = subprocess.run(stats_cmd, capture_output=True, text=True, env=nsys_env)
    if proc.returncode != 0:
        print("nsys stats stdout:\n" + proc.stdout)
        print("nsys stats stderr:\n" + proc.stderr, file=sys.stderr)
        sys.exit(f"nsys stats exited {proc.returncode}")

    return _parse_nsys_stats(proc.stdout)


def _parse_nsys_stats(text: str) -> dict[str, float]:
    """Parse `nsys stats --report cuda_gpu_kern_sum`. Returns {kernel_name:
    median_ms}. Columns: 8 numeric (Time% Total Instances Avg Med Min Max
    StdDev), then Name. Numbers may carry commas but no internal spaces, so
    whitespace tokenization is reliable."""
    lines = text.splitlines()
    header_i = None
    for i, ln in enumerate(lines):
        if "Med (" in ln and "Name" in ln and ("ns)" in ln or "us)" in ln or "ms)" in ln):
            header_i = i
            break
    if header_i is None:
        sys.exit("could not find kernel-summary header in nsys stats output:\n  " + "\n  ".join(lines[:60]))

    m_unit = re.search(r"Med \((\w+)\)", lines[header_i])
    unit = m_unit.group(1) if m_unit else "ns"
    unit_div = {"ns": 1e6, "us": 1e3, "ms": 1.0, "s": 1e-3}.get(unit, 1e6)

    # Med is numeric col 4 (0-indexed); the name is everything after col 8.
    NUM_NUMERIC_COLS = 8
    MED_COL = 4

    result: dict[str, float] = {}
    in_data = False
    for j in range(header_i + 1, len(lines)):
        row = lines[j]
        stripped = row.strip()
        if not stripped:
            if in_data:
                break
            continue
        if set(stripped) <= set("- "):
            in_data = True
            continue
        if not in_data:
            continue
        if stripped.startswith("**") or stripped.startswith("##"):
            break

        toks = stripped.split()
        if len(toks) <= NUM_NUMERIC_COLS:
            continue
        try:
            med = float(toks[MED_COL].replace(",", ""))
        except ValueError:
            continue
        name = " ".join(toks[NUM_NUMERIC_COLS:]).rstrip()
        if not name:
            continue
        result[name] = med / unit_div
    return result


def _match_kernel_name(kern_name: str, config_name: str) -> bool:
    """Match a config against nsys's demangled symbol by substring."""
    return config_name in kern_name


def _find_cublas_time(kern_times: dict[str, float]) -> tuple[str, float] | None:
    """The cuBLAS 'nvjet_*' kernel with the longest median."""
    cands = [(k, v) for k, v in kern_times.items() if k.startswith("nvjet_")]
    if not cands:
        return None
    return max(cands, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Worker mode: just run the kernels under nsys profile, no Python timing.
# ---------------------------------------------------------------------------


def _nsys_worker(
    shape: str,
    configs: list[str],
    warmup: int,
    iters: int,
    nbuf: int,
) -> None:
    """Inner mode re-exec'd under nsys: run each config (and cuBLAS) for
    warmup+iters launches, no timing — nsys captures it. Timed iters rotate
    across the pool; warmup uses a dedicated buffer."""
    B, M, N, K = (int(x) for x in shape.split(","))
    wa, wb, wc = _mkdata(B, M, N, K)  # dedicated warmup buffer
    pool = _mkdata_pool(B, M, N, K, nbuf)  # rotation pool for timed iters

    print(f"[worker] shape={B}x{M}x{N}x{K}, configs={len(configs)}, " f"warmup={warmup}, iters={iters}, rotate_buffers={nbuf}")

    # 1. cuBLAS.
    for _ in range(warmup):
        torch.matmul(wa, wb.transpose(-1, -2), out=wc)
    for i in range(iters):
        a, b, c = pool[i % nbuf]
        torch.matmul(a, b.transpose(-1, -2), out=c)
    torch.cuda.synchronize()

    # 2. each GEMM config.
    config_names = configs or list(_SPEC_MAP)
    for name in config_names:
        spec = _spec_for(name)
        if spec is None:
            continue
        cfg = spec[0]
        try:
            g, h = _graph_matmul(B, M, N, K)
            plan = _build_plan(g, cfg, name)
            for _ in range(warmup):
                plan(_vp(h, wa, wb, wc))
            for i in range(iters):
                a, b, c = pool[i % nbuf]
                plan(_vp(h, a, b, c))
            torch.cuda.synchronize()
            print(f"[worker] OK   {name}")
        except Exception as e:
            print(f"[worker] FAIL {name}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        default="1,4096,4096,4096",
        help="B,M,N,K (default 1,4096,4096,4096; B = batch / number of " "independent same-shape GEMMs)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    # CLAUDE.md: keep iters <= 20 (more doesn't sharpen the measurement).
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--configs",
        default=None,
        help="comma-separated config names to test (default: every CATALOG entry)",
    )
    parser.add_argument(
        "--timing",
        choices=("delayed", "events", "nsys"),
        default="delayed",
        help="delayed (default): events-around-loop, with a long delay kernel "
        "queued first to hide host-launch overhead — matches nsys to "
        "<2%%. events: plain events around a loop (has ~50us/call "
        "Python overhead — inflates sub-ms kernels). nsys: ground "
        "truth via nsys profile (read GPU-side kernel timestamps).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="print each config's result line as soon as it finishes (and a "
        "'running …' line before measurement starts). Useful when a "
        "config hangs — the last 'running' line points at the culprit. "
        "events/delayed modes only; no effect under --timing nsys.",
    )
    parser.add_argument(
        "--rotate-buffers",
        default="auto",
        metavar="N",
        help="allocate N independent copies of every tensor and rotate the "
        "timed launches across them (launch i uses copy i%%N) so a kernel "
        "doesn't re-read the previous launch's data from a hot L2 — the "
        "main source of inflated TFLOPS on small shapes. Warmup uses a "
        "separate dedicated buffer (never rotated). Default 'auto': size "
        "the pool to exceed the ~126 MB B200 L2 for this shape (large "
        "shapes → 2 copies, small shapes → scaled up), capped at 4 GB. "
        "Pass an integer to override; 1 disables rotation.",
    )
    parser.add_argument(
        "--_nsys-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be B,M,N,K (four values; use B=1 for a plain matmul)")
    B, M, N, K = parts
    nbuf = _resolve_nbuf(args.rotate_buffers, B, M, N, K)

    if getattr(args, "_nsys_worker"):
        configs = [c.strip() for c in args.configs.split(",")] if args.configs else []
        _nsys_worker(args.shape, configs, args.warmup, args.iters, nbuf)
        return 0

    flops = 2 * B * M * N * K
    config_names = [c.strip() for c in args.configs.split(",")] if args.configs else list(_SPEC_MAP)

    print(f"\n=== matmul B={B} {M}x{N}x{K}  (~{flops / 1e9:.1f} GFLOP) — BF16 ===")

    if nbuf > 1:
        footprint = _pool_footprint_bytes(B, M, N, K, nbuf)
        print(f"  [rotate-buffers: {nbuf} copies/tensor, " f"{footprint / 1024 / 1024:.0f} MB pool — defeats hot-L2 on small shapes]")
        if footprint < _L2_BYTES_B200:
            print(
                f"  [WARNING: pool ({footprint / 1024 / 1024:.0f} MB) < B200 L2 "
                f"(~{_L2_BYTES_B200 / 1024 / 1024:.0f} MB) — it fits in cache, so "
                f"launches stay warm after one rotation. Bump --rotate-buffers.]"
            )
    else:
        print("  [rotate-buffers: disabled (1) — small-shape TFLOPS may be hot-L2-inflated]")

    rows: list[tuple[str, float, float, str]] = []  # (name, tflops, ms, note)
    t0 = time.time()

    def _fmt_row(name: str, tflops: float, ms: float, note: str, ref_tflops: float) -> str:
        if note:
            return f"  {name:50s} {'':8s}   {'':7s}   {note}"
        ratio = tflops / ref_tflops if ref_tflops > 0 else 0.0
        return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {ratio:>9.2f}×"

    if args.timing == "nsys":
        print("  [timing: nsys median kernel duration]\n")
        kern_times = _nsys_run_and_parse(args.shape, config_names, args.warmup, args.iters, nbuf)

        cublas_hit = _find_cublas_time(kern_times)
        if cublas_hit:
            cublas_name, cublas_ms = cublas_hit
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            print(f"  cuBLAS kernel: {cublas_name}")
        else:
            cublas_tflops, cublas_ms = float("nan"), float("nan")
            print("  cuBLAS kernel: not detected in nsys output")

        for name in config_names:
            spec = _spec_for(name)
            cfg = spec[0] if spec else None
            if cfg is None:
                rows.append((name, 0.0, float("inf"), "UNKNOWN_CONFIG"))
                continue
            matches = [(k, v) for k, v in kern_times.items() if _match_kernel_name(k, name)]
            if not matches:
                rows.append((name, 0.0, float("inf"), "NO_KERNEL_IN_NSYS"))
                continue
            # Multiple specializations share a config name → pick the heaviest.
            _, ms = max(matches, key=lambda x: x[1])
            rows.append((name, flops / (ms * 1e-3) / 1e12, ms, ""))
    else:
        timer = _time_ms_delayed if args.timing == "delayed" else _time_ms_events
        if args.timing == "delayed":
            print("  [timing: events bracketed around delayed back-to-back " "launches — host overhead hidden behind a CUDA _sleep]\n")
        else:
            print(
                "  [timing: torch.cuda.Event wall-clock around python loop — "
                "includes ~50us/call Python+TVM-FFI dispatch overhead; use "
                "--timing delayed or --timing nsys for kernel-only timing]\n"
            )
        wa, wb, wc = _mkdata(B, M, N, K)  # dedicated warmup buffer
        pool = _mkdata_pool(B, M, N, K, nbuf)  # rotation pool for timed iters
        if args.stream:
            print("  ▶ running cuBLAS reference ...", flush=True)
        cublas_ms = timer(
            _rotating(lambda t: torch.matmul(t[0], t[1].transpose(-1, -2), out=t[2]), pool),
            lambda: torch.matmul(wa, wb.transpose(-1, -2), out=wc),
            warmup=args.warmup,
            iters=args.iters,
        )
        cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
        if args.stream:
            print(
                _fmt_row("cuBLAS (reference)", cublas_tflops, cublas_ms, "", cublas_tflops),
                flush=True,
            )

        # An async device fault sticky-poisons the CUDA context for the rest of
        # the process (every later launch returns LAUNCH_FAILED). After the
        # first such error, short-circuit the remaining configs as CTX_DEAD.
        ctx_dead = False
        for name in config_names:
            spec = _spec_for(name)
            cfg = spec[0] if spec else None
            if cfg is None:
                row = (name, 0.0, float("inf"), "UNKNOWN_CONFIG")
            elif ctx_dead:
                row = (name, 0.0, float("inf"), "skipped (CUDA context dead)")
            else:
                if args.stream:
                    print(f"  ▶ running {name} ...", flush=True)
                try:
                    g, h = _graph_matmul(B, M, N, K)
                    plan = _build_plan(g, cfg, name)
                    ms = timer(
                        _rotating(
                            lambda t, _plan=plan, _h=h: _plan(_vp(_h, t[0], t[1], t[2])),
                            pool,
                        ),
                        lambda _plan=plan, _h=h: _plan(_vp(_h, wa, wb, wc)),
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    row = (name, flops / (ms * 1e-3) / 1e12, ms, "")
                except Exception as e:
                    msg = str(e).splitlines()[0][:50] if str(e) else type(e).__name__
                    row = (name, 0.0, float("inf"), f"ERR {msg}")
                    # Context-poisoning errors are unrecoverable — stop trying.
                    if any(
                        s in str(e)
                        for s in (
                            "illegal memory access",
                            "unspecified launch failure",
                            "CUDA_ERROR_LAUNCH_FAILED",
                        )
                    ):
                        ctx_dead = True
            rows.append(row)
            if args.stream:
                print(_fmt_row(*row, cublas_tflops), flush=True)

    rows.sort(key=lambda r: -r[1])
    print("=" * 88)
    print(f"  {'config':50s} {'TFLOPS':>8s}   {'ms':>7s}   {'vs cuBLAS':>10s}")
    print("=" * 88)
    for name, tflops, ms, note in rows:
        print(_fmt_row(name, tflops, ms, note, cublas_tflops))
    print("=" * 88)
    if cublas_tflops > 0:
        print(f"  {'cuBLAS (reference)':50s} {cublas_tflops:8.2f}   {cublas_ms:7.3f}   {'1.00×':>10s}")
    else:
        print("  cuBLAS reference: n/a")

    ok = [r for r in rows if not r[3]]
    if ok and cublas_tflops > 0:
        best_name, best_tflops, _best_ms, _ = ok[0]
        print(f"\nbest GEMM: {best_name}" f" — {best_tflops:.2f} TFLOPS" f" ({best_tflops / cublas_tflops:.2f}× cuBLAS)")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
