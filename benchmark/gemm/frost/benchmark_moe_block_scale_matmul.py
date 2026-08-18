# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark MoE grouped block-scale matmul (FP4/FP8 + per-block SF) configs vs a
cuBLAS BF16 batched-GEMM reference. `--shape` is G,M,N,K (even split, S=G*M).

The reference is NOT a block-scaled GEMM — it's dense BF16 over the G groups — so
the "vs cuBLAS" ratio is FP4/FP8-MoE throughput relative to equivalent BF16 cuBLAS.
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
from cudnn.gemm.frost.tile_config import by_name as _by_name
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import candidates as _candidates
from cudnn.gemm.frost.tile_config import TileConfig


def _vp_moe_bs(handles, token, weight, sfa, sfb, fto, output):
    """MoE block-scale single-GEMM variant-pack dict keyed by the graph's tensors."""
    TOK, W, SFA, SFB, FTO, OUT = handles
    return {TOK: token, W: weight, SFA: sfa, SFB: sfb, FTO: fto, OUT: output}


def _build_plan(g, cfg, name):
    """JIT-compile the recorded graph with a forced tile config -> compiled kernel."""
    return jit_from_cudnn_graph(g, config=cfg, cta_group=_spec_for(name)[1], scheduler=_spec_for(name)[2])


# combo : (is_fp4, block_size, a_dtype, sf_dtype)
_COMBOS = {
    "nvfp4": (True, 16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (True, 32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (False, 32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


# Graph + data setup.


def _graph_moe_bs(S: int, N: int, K: int, E: int, combo: str):
    is_fp4, block_size, a_dt, sf_dt = _COMBOS[combo]
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
    SFA = g.tensor(
        name="SFA",
        dim=[1, S, sf_k],
        stride=[S * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[E, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[E, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size])
    w_d = g.block_scale_dequantize(input=w, descale=SFB, block_size=[block_size, 1])
    out = g.moe_grouped_matmul(
        tok_d,
        w_d,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
    )
    out.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    return g, (tok, w, SFA, SFB, fto, out)


def _build_spec_map():
    """Label -> (geometry cfg, cta_group, scheduler) for every MoE-block-scale
    strategy. Enumerated from an nvfp4 graph (template set is combo-independent)."""
    chain = analyze(_graph_moe_bs(512, 256, 512, 2, "nvfp4")[0])
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


def _offsets(S: int, E: int) -> torch.Tensor:
    """Even split: group g owns rows [g*group_m, (g+1)*group_m)."""
    group_m = S // E
    return torch.arange(E, dtype=torch.int32, device="cuda") * group_m


def _ceil_div(a, b):
    return (a + b - 1) // b


def _to_blocked(x):
    """(rows, cols) scale tensor -> flat F8_128x4 blocked layout (padded to 128 rows / 4 cols)."""
    rows, cols = x.shape
    nrb, ncb = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _rand_e8m0(shape, dev):
    return torch.randint(125, 129, shape, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)


def _mkdata(S: int, N: int, K: int, E: int, combo: str):
    """MoE-block-scale runtime tensors: (token, weight, sfa, sfb, output).

    token/weight are packed FP4 or FP8. SFA is reordered + padded to 128 rows PER
    GROUP then concatenated (even split); SFB is per-expert; output is BF16."""
    dev = "cuda"
    torch.manual_seed(0)
    is_fp4, block_size, _, _ = _COMBOS[combo]
    sf_k = K // block_size
    group_m = S // E

    if is_fp4:
        tok = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
        w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
    else:
        tok = (torch.randn(1, S, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w = (torch.randn(E, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((S, sf_k), dev)
        sfb_log = _rand_e8m0((E, N, sf_k), dev)

    # SFA padded to 128 rows PER GROUP (group sizes need not be 128-aligned); SFB per-expert.
    sfa = torch.cat([_to_blocked(sfa_log[g * group_m : (g + 1) * group_m]) for g in range(E)]).view(1, -1, 1)
    sfb = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).reshape(E, sf_k, N)
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    return tok, w, sfa, sfb, out


def _mkdata_bf16(S: int, N: int, K: int, E: int):
    """BF16 reference operands (token, weight, output) for the batched-GEMM ref."""
    dev = "cuda"
    tok = torch.empty(1, S, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device=dev)
    w = torch.empty(E, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device=dev)
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    return tok, w, out


# Buffer rotation: rotate timed launches across N independent tensor sets so a
# kernel never re-reads its inputs from a hot L2 (inflates small-shape TFLOPS).

_L2_BYTES_B200 = 126 * 1024 * 1024
_AUTO_POOL_BUDGET_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
_AUTO_NBUF_CAP = 1024


def _set_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _mkdata_pool(S: int, N: int, K: int, E: int, combo: str, nbuf: int):
    base = _mkdata(S, N, K, E, combo)
    pool = [base]
    for _ in range(max(0, nbuf - 1)):
        pool.append(tuple(t.clone() for t in base))
    return pool


def _mkdata_bf16_pool(S: int, N: int, K: int, E: int, nbuf: int):
    base = _mkdata_bf16(S, N, K, E)
    pool = [base]
    for _ in range(max(0, nbuf - 1)):
        pool.append(tuple(t.clone() for t in base))
    return pool


def _per_set_bytes(S: int, N: int, K: int, E: int, combo: str) -> int:
    return _set_bytes(_mkdata(S, N, K, E, combo))


def _pool_footprint_bytes(S, N, K, E, combo, nbuf) -> int:
    return _per_set_bytes(S, N, K, E, combo) * nbuf


def _auto_nbuf(S: int, N: int, K: int, E: int, combo: str) -> int:
    per_set = _per_set_bytes(S, N, K, E, combo)
    target = int(1.5 * _L2_BYTES_B200)
    nbuf = max(2, -(-target // per_set))
    budget = _AUTO_POOL_BUDGET_BYTES
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        budget = min(budget, free // 4)
    max_by_budget = max(1, budget // per_set)
    return max(1, min(nbuf, max_by_budget, _AUTO_NBUF_CAP))


def _resolve_nbuf(spec: str, S, N, K, E, combo: str) -> int:
    if spec.strip().lower() == "auto":
        return _auto_nbuf(S, N, K, E, combo)
    return max(1, int(spec))


def _rotating(fn_of_buf: Callable, pool: list) -> Callable:
    n = len(pool)
    return lambda i: fn_of_buf(pool[i % n])


# cuBLAS reference — BF16 batched GEMM over the E equal-sized groups.


def _cublas_launch(buf, S: int, N: int, K: int, E: int) -> None:
    tok, w, out = buf
    group_m = S // E
    tok_g = tok.view(E, group_m, K)
    out_g = out.view(E, group_m, N)
    torch.matmul(tok_g, w.transpose(-1, -2), out=out_g)


# Timing (events / delayed).


def _time_ms_events(timed_fn, warmup_fn, *, warmup, iters) -> float:
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


def _time_ms_delayed(timed_fn, warmup_fn, *, warmup, iters) -> float:
    """Kernel-only timing: hide host-launch overhead behind a delay kernel
    (queue a long _sleep first so launches enqueue behind it, then run back-to-back)."""
    for _ in range(warmup):
        warmup_fn()
    torch.cuda.synchronize()
    delay_cycles = max(int(1e8), int((iters * 0.05 + 20.0) * 1.7e6))
    torch.cuda._sleep(delay_cycles)
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


# nsys mode.


def _nsys_run_and_parse(shape, combo, configs, warmup, iters, nbuf, no_baseline=False) -> dict[str, float]:
    nsys = "/usr/local/bin/nsys" if os.path.exists("/usr/local/bin/nsys") else shutil.which("nsys")
    if nsys is None:
        sys.exit("nsys not found — install nsight-systems or use the default events mode.")
    workdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"bench_moebs_nsys_{os.getpid()}")
    os.makedirs(workdir, exist_ok=True)
    report_prefix = os.path.join(workdir, "report")
    nsys_env = os.environ.copy()
    nsys_env.setdefault("TMPDIR", os.environ.get("TMPDIR", tempfile.gettempdir()))
    inner = [
        sys.executable,
        "-u",
        os.path.abspath(__file__),
        "--_nsys-worker",
        "--shape",
        shape,
        "--combo",
        combo,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--rotate-buffers",
        str(nbuf),
    ]
    if configs:
        inner += ["--configs", ",".join(configs)]
    if no_baseline:
        inner += ["--no-baseline"]
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
    """Parse `nsys stats --report cuda_gpu_kern_sum` -> {kernel_name: median_ms}."""
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
    NUM_NUMERIC_COLS = 8
    MED_COL = 4
    result: dict[str, float] = {}
    in_data = False
    for j in range(header_i + 1, len(lines)):
        stripped = lines[j].strip()
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
        if name:
            result[name] = med / unit_div
    return result


def _kernel_match_token(cfg: TileConfig, cta_group: int) -> str:
    """`<G>ctamma_<geometry_name>` — the substring the nsys-demangled kernel symbol carries."""
    return f"{cta_group}ctamma_{cfg.geometry_name}"


def _find_cublas_time(kern_times: dict[str, float]) -> tuple[str, float] | None:
    cands = [(k, v) for k, v in kern_times.items() if k.startswith("nvjet_")]
    if not cands:
        return None
    return max(cands, key=lambda x: x[1])


# Worker mode (re-exec'd under nsys).


def _nsys_worker(shape, combo, configs, warmup, iters, nbuf, no_baseline=False) -> None:
    G, M, N, K = (int(x) for x in shape.split(","))
    S, E = G * M, G
    offsets = _offsets(S, E)
    print(f"[worker] shape G={G} M={M} N={N} K={K} (S={S}) combo={combo}, " f"configs={len(configs)}, warmup={warmup}, iters={iters}, rotate={nbuf}")

    # 1. BF16 batched-GEMM reference.
    if not no_baseline:
        wbf = _mkdata_bf16(S, N, K, E)
        bf_pool = _mkdata_bf16_pool(S, N, K, E, nbuf)
        for _ in range(warmup):
            _cublas_launch(wbf, S, N, K, E)
        for i in range(iters):
            _cublas_launch(bf_pool[i % nbuf], S, N, K, E)
        torch.cuda.synchronize()
        del wbf, bf_pool
        torch.cuda.empty_cache()
    wset = _mkdata(S, N, K, E, combo)
    pool = _mkdata_pool(S, N, K, E, combo, nbuf)

    # 2. each MoE-block-scale config.
    for name in configs or list(_SPEC_MAP):
        spec = _spec_for(name)
        cfg = spec[0] if spec else None
        if cfg is None:
            continue
        try:
            g, h = _graph_moe_bs(S, N, K, E, combo)
            plan = _build_plan(g, cfg, name)
            for _ in range(warmup):
                plan(_vp_moe_bs(h, wset[0], wset[1], wset[2], wset[3], offsets, wset[4]))
            for i in range(iters):
                t = pool[i % nbuf]
                plan(_vp_moe_bs(h, t[0], t[1], t[2], t[3], offsets, t[4]))
            torch.cuda.synchronize()
            print(f"[worker] OK   {name}")
        except Exception as e:
            print(f"[worker] FAIL {name}: {type(e).__name__}: {e}")


# Main.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        default="8,512,4096,4096",
        help="G,M,N,K (groups × per-group-M × N × K; default " "8,512,4096,4096 → S=G*M=4096 tokens)",
    )
    parser.add_argument("--combo", choices=tuple(_COMBOS), default="nvfp4")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)  # CLAUDE.md: keep <= 20
    parser.add_argument(
        "--configs",
        default=None,
        help="comma-separated config names (default: every MoE-block-scale entry)",
    )
    parser.add_argument("--timing", choices=("delayed", "events", "nsys"), default="delayed")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="print each config's result as soon as it finishes (events/delayed)",
    )
    parser.add_argument(
        "--rotate-buffers",
        default="auto",
        metavar="N",
        help="independent tensor copies to rotate timed launches across " "(default 'auto' sizes the pool past L2; 1 disables).",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the cuBLAS BF16 reference; its BF16 tensor set is never " "allocated, which is the larger footprint on big shapes",
    )
    parser.add_argument("--_nsys-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be G,M,N,K (four values: groups × per-group-M × N × K)")
    G, M, N, K = parts
    S, E = G * M, G  # total tokens, num groups (== experts)
    combo = args.combo
    nbuf = _resolve_nbuf(args.rotate_buffers, S, N, K, E, combo)

    if getattr(args, "_nsys_worker"):
        configs = [c.strip() for c in args.configs.split(",")] if args.configs else []
        _nsys_worker(args.shape, combo, configs, args.warmup, args.iters, nbuf, args.no_baseline)
        return 0

    flops = 2 * S * N * K
    config_names = [c.strip() for c in args.configs.split(",")] if args.configs else list(_SPEC_MAP)

    print(f"\n=== moe_block_scale_matmul G={G} M={M} N={N} K={K}  " f"(S={S} tokens, ~{flops / 1e9:.1f} GFLOP) — {combo} ===")

    if nbuf > 1:
        footprint = _pool_footprint_bytes(S, N, K, E, combo, nbuf)
        print(f"  [rotate-buffers: {nbuf} copies/tensor, {footprint / 1024 / 1024:.0f} MB pool]")
        if footprint < _L2_BYTES_B200:
            print(f"  [WARNING: pool ({footprint / 1024 / 1024:.0f} MB) < B200 L2 " f"(~{_L2_BYTES_B200 / 1024 / 1024:.0f} MB) — bump --rotate-buffers.]")
    else:
        print("  [rotate-buffers: disabled (1) — small-shape TFLOPS may be hot-L2-inflated]")

    rows: list[tuple[str, float, float, str]] = []
    t0 = time.time()

    def _fmt_row(name, tflops, ms, note, ref_tflops) -> str:
        if note:
            return f"  {name:50s} {'':8s}   {'':7s}   {note}"
        if ref_tflops <= 0:
            return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {'':>10s}"
        return f"  {name:50s} {tflops:8.2f}   {ms:7.3f}   {tflops / ref_tflops:>9.2f}×"

    if args.timing == "nsys":
        print("  [timing: nsys median kernel duration]\n")
        kern_times = _nsys_run_and_parse(args.shape, combo, config_names, args.warmup, args.iters, nbuf, args.no_baseline)
        cublas_tflops, cublas_ms = 0.0, float("nan")
        cublas_hit = None if args.no_baseline else _find_cublas_time(kern_times)
        if cublas_hit:
            cublas_name, cublas_ms = cublas_hit
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            print(f"  cuBLAS (BF16) kernel: {cublas_name}")
        elif not args.no_baseline:
            cublas_tflops, cublas_ms = float("nan"), float("nan")
            print("  cuBLAS kernel: not detected in nsys output")
        for name in config_names:
            spec = _spec_for(name)
            cfg = spec[0] if spec else None
            if cfg is None:
                rows.append((name, 0.0, float("inf"), "UNKNOWN_CONFIG"))
                continue
            tok = _kernel_match_token(cfg, _spec_for(name)[1])
            matches = [(k, v) for k, v in kern_times.items() if tok in k]
            if not matches:
                rows.append((name, 0.0, float("inf"), "NO_KERNEL_IN_NSYS"))
                continue
            _, ms = max(matches, key=lambda x: x[1])
            rows.append((name, flops / (ms * 1e-3) / 1e12, ms, ""))
    else:
        timer = _time_ms_delayed if args.timing == "delayed" else _time_ms_events
        if args.timing == "delayed":
            print("  [timing: events around delayed back-to-back launches]\n")
        else:
            print("  [timing: torch.cuda.Event wall-clock (incl ~50us/call host overhead)]\n")
        offsets = _offsets(S, E)
        cublas_tflops, cublas_ms = 0.0, float("nan")
        if not args.no_baseline:
            wbf = _mkdata_bf16(S, N, K, E)
            bf_pool = _mkdata_bf16_pool(S, N, K, E, nbuf)
            if args.stream:
                print("  ▶ running cuBLAS (BF16) reference ...", flush=True)
            cublas_ms = timer(
                _rotating(lambda t: _cublas_launch(t, S, N, K, E), bf_pool),
                lambda: _cublas_launch(wbf, S, N, K, E),
                warmup=args.warmup,
                iters=args.iters,
            )
            cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12
            if args.stream:
                print(
                    _fmt_row(
                        "cuBLAS BF16 (reference)",
                        cublas_tflops,
                        cublas_ms,
                        "",
                        cublas_tflops,
                    ),
                    flush=True,
                )
            # Freed before the block-scale pool is built: the reference is done
            # with, and on large shapes its BF16 set is the bigger allocation.
            del wbf, bf_pool
            torch.cuda.empty_cache()
        wset = _mkdata(S, N, K, E, combo)
        pool = _mkdata_pool(S, N, K, E, combo, nbuf)

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
                    g, h = _graph_moe_bs(S, N, K, E, combo)
                    plan = _build_plan(g, cfg, name)
                    ms = timer(
                        _rotating(
                            lambda t, _plan=plan, _h=h: _plan(_vp_moe_bs(_h, t[0], t[1], t[2], t[3], offsets, t[4])),
                            pool,
                        ),
                        lambda _plan=plan, _h=h: _plan(_vp_moe_bs(_h, wset[0], wset[1], wset[2], wset[3], offsets, wset[4])),
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    row = (name, flops / (ms * 1e-3) / 1e12, ms, "")
                except Exception as e:
                    msg = str(e).splitlines()[0][:50] if str(e) else type(e).__name__
                    row = (name, 0.0, float("inf"), f"ERR {msg}")
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
    print(f"  {'config':50s} {'TFLOPS':>8s}   {'ms':>7s}   {'vs BF16':>10s}")
    print("=" * 88)
    for name, tflops, ms, note in rows:
        print(_fmt_row(name, tflops, ms, note, cublas_tflops))
    print("=" * 88)
    if cublas_tflops > 0:
        print(f"  {'cuBLAS BF16 (batched, reference)':50s} {cublas_tflops:8.2f}   {cublas_ms:7.3f}   {'1.00×':>10s}")
    else:
        print("  cuBLAS reference: n/a")

    ok = [r for r in rows if not r[3]]
    if ok:
        best_name, best_tflops, best_ms, _ = ok[0]
        ratio = f" ({best_tflops / cublas_tflops:.2f}× BF16 cuBLAS)" if cublas_tflops > 0 else ""
        print(f"\nbest GEMM: {best_name} — {best_tflops:.2f} TFLOPS{ratio}")
    print(f"total: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
