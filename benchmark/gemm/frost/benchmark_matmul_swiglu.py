# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the fused multi-GEMM SwiGLU block silu(a@b0)*(a@b1)*scale vs an
unfused 2×cuBLAS + pointwise baseline. --shape is B,M,N,K.

    python benchmark/gemm/frost/benchmark_matmul_swiglu.py --shape 1,4096,11008,4096
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Callable

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import torch

from types import SimpleNamespace

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.tile_config import by_name as _by_name
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import candidates as _registry_candidates


def _build_plan(g, cfg, cta_group, sched):
    """JIT-compile the recorded graph with a forced tile config."""
    return jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group, scheduler=sched)


def _vp_mg(handles, gemm_pairs, outs, *aux):
    """Multi-GEMM variant-pack dict (dedup pairs by tensor identity → distinct
    A/B slots, + outputs + aux)."""
    bd = handles
    a_seen, b_seen = [], []
    for ag, bg in gemm_pairs:
        if not any(ag is x for x in a_seen):
            a_seen.append(ag)
        if not any(bg is x for x in b_seen):
            b_seen.append(bg)
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {}
    vp.update({t: buf for t, buf in zip(bd.a_operands, a_seen)})
    vp.update({t: buf for t, buf in zip(bd.b_operands, b_seen)})
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


_CUDNN_DT = {"bf16": cudnn.data_type.BFLOAT16, "fp16": cudnn.data_type.HALF}
_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}


# ---------------------------------------------------------------------------
# Graph + data
# ---------------------------------------------------------------------------


def _graph_swiglu(B: int, M: int, N: int, K: int, in_dt: str, out_dt: str):
    """Build the DualMatmulSiluMulDequant graph: silu(A@B0) * (A@B1) * scale."""
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DT[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="aTensor", dim=[B, M, K], stride=[M * K, K, 1])
    B0 = g.tensor(name="b0Tensor", dim=[B, K, N], stride=[K * N, 1, K])
    B1 = g.tensor(name="b1Tensor", dim=[B, K, N], stride=[K * N, 1, K])
    scale = g.tensor(
        name="scaleFactor",
        dim=[1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    C0 = g.matmul(A=A, B=B0, name="mm0")
    C1 = g.matmul(A=A, B=B1, name="mm1")
    S0 = g.swish(input=C0, name="silu0")
    MU = g.mul(a=S0, b=C1, name="mul0")
    DQ = g.mul(a=MU, b=scale, name="dequant0")
    DQ.set_output(True).set_data_type(_CUDNN_DT[out_dt])
    return g, SimpleNamespace(a_operands=[A], b_operands=[B0, B1], outputs=[DQ], aux=[scale])


def _mkdata(B, M, N, K, in_dt, out_dt):
    td_in, td_out = _TORCH_DT[in_dt], _TORCH_DT[out_dt]
    a = torch.randn(B, M, K, device="cuda", dtype=td_in) * 0.4
    b0 = torch.randn(B, N, K, device="cuda", dtype=td_in) * 0.4
    b1 = torch.randn(B, N, K, device="cuda", dtype=td_in) * 0.4
    out = torch.empty(B, M, N, device="cuda", dtype=td_out)
    scale = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)
    return a, b0, b1, out, scale


def _reference(a, b0, b1, scale, out_dt):
    """Correctness reference: 2 GEMMs + elementwise chain (einsum 'bmk,bnk->bmn'
    matches the (B,N,K) operands)."""
    c0 = torch.einsum("bmk,bnk->bmn", a.float(), b0.float())
    c1 = torch.einsum("bmk,bnk->bmn", a.float(), b1.float())
    return (torch.nn.functional.silu(c0) * c1 * scale.flatten()[0]).to(_TORCH_DT[out_dt])


def _unfused_launch(a, b0, b1, scale, out):
    """Baseline: 2 cuBLAS GEMMs + pointwise, matching the fused kernel's math."""
    c0 = torch.matmul(a, b0.transpose(-1, -2))
    c1 = torch.matmul(a, b1.transpose(-1, -2))
    out.copy_((torch.nn.functional.silu(c0.float()) * c1.float() * scale.flatten()[0]).to(out.dtype))


# Timing (delayed / events) — same pattern as benchmark_matmul.py. delayed hides
# host-launch overhead behind a CUDA _sleep so kernels run back-to-back.


def _time_ms(timed_fn: Callable, *, warmup: int, iters: int, delayed: bool) -> float:
    for _ in range(warmup):
        timed_fn()
    torch.cuda.synchronize()
    if delayed:
        delay_cycles = max(int(1e8), int((iters * 0.05 + 20.0) * 1.7e6))
        torch.cuda._sleep(delay_cycles)
        for _ in range(max(5, warmup)):
            timed_fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        timed_fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _build_spec_map():
    """Legacy label -> (geometry cfg, cta_group, scheduler) for every multi-GEMM-
    capable sm100 matmul strategy. Multi-GEMM TMEM fits num_gemms accumulators
    only for cta_tile_n<=256 (2*256<=512); cta_tile_m=128."""
    chain = analyze(_graph_swiglu(1, 256, 256, 256, "bf16", "bf16")[0])
    m = {}
    for t, cfg in _registry_candidates(chain):
        if cfg.pipeline != "sm100" or cfg.cta_tile_n > 256 or cfg.cgrp_size_n != 1 or cfg.cta_tile_m != 128:
            continue
        label = f"{cfg.name}_{t.cta_group}ctamma" + ("_static" if t.static_sched else "")
        m[label] = (cfg, t.cta_group, t.scheduler)
    return m


_SPEC_MAP = _build_spec_map()

_LABEL_RE = re.compile(r"^(CONFIG_sm\d+_\d+x\d+x\d+_\d+x\d+x\d+_cluster\d+x\d+)_([12])ctamma(_static)?$")


def _spec_for(name):
    """(geometry cfg, cta_group, scheduler) for a --configs label, or None.

    The sweep set comes from the registry funnel over CATALOG; a label naming a
    geometry outside it (e.g. a num_mma_m > 1 tile, which `by_name` synthesizes) is
    still runnable, so parse it rather than reporting it unsweepable."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="1,4096,4096,4096", help="B,M,N,K")
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)  # CLAUDE.md: <= 20
    p.add_argument(
        "--configs",
        default=None,
        help="comma-separated CONFIG_..._Nctamma[_static] labels " "(same form as benchmark_matmul.py; default: sweep all)",
    )
    p.add_argument("--timing", choices=("delayed", "events"), default="delayed")
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--atol", type=float, default=2e-1)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be B,M,N,K (four values; B=1 = a single SwiGLU block)")
    B, M, N, K = parts
    in_dt = out_dt = args.dtype
    delayed = args.timing == "delayed"

    # 2 GEMMs, each 2*B*M*N*K flops.
    flops = 2 * (2 * B * M * N * K)
    print(f"\n=== SwiGLU dual-matmul  B={B} {M}x{N}x{K}  " f"(~{flops / 1e9:.1f} GFLOP, 2 GEMMs) — {in_dt} in / {out_dt} out ===")
    print(f"  [timing: {args.timing}, warmup={args.warmup}, iters={args.iters}]\n")

    a, b0, b1, out, scale = _mkdata(B, M, N, K, in_dt, out_dt)
    ref = _reference(a, b0, b1, scale, out_dt)

    # --- baseline: unfused 2×cuBLAS + pointwise ---
    out_bl = torch.empty_like(out)
    bl_ms = _time_ms(
        lambda: _unfused_launch(a, b0, b1, scale, out_bl),
        warmup=args.warmup,
        iters=args.iters,
        delayed=delayed,
    )
    bl_tflops = flops / (bl_ms * 1e-3) / 1e12
    print(f"  {'unfused 2xcuBLAS + pointwise':52s} {bl_tflops:8.2f} TFLOP/s  " f"{bl_ms:8.3f} ms   {'1.00×':>8s}")

    # --- candidate (config, cta_group, scheduler) strategies ---
    config_names = [c.strip() for c in args.configs.split(",")] if args.configs else list(_SPEC_MAP)

    best = None
    for label in config_names:
        spec = _spec_for(label)
        if spec is None:
            print(f"  {label:62s} UNKNOWN (not a sweepable swiglu strategy)")
            continue
        cfg, cta_group, sched = spec
        try:
            g, h = _graph_swiglu(B, M, N, K, in_dt, out_dt)
            plan = _build_plan(g, cfg, cta_group, sched)
        except (NotImplementedError, ValueError):
            continue  # geometry/strategy can't run this shape/dtype — skip
        try:
            plan(_vp_mg(h, [(a, b0), (a, b1)], out, scale))
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {label:62s} LAUNCH FAIL: {type(e).__name__}: {str(e)[:30]}")
            continue
        err = (out.float() - ref.float()).abs().max().item()
        ok = torch.allclose(out.float(), ref.float(), rtol=args.rtol, atol=args.atol)
        ms = _time_ms(
            lambda: plan(_vp_mg(h, [(a, b0), (a, b1)], out, scale)),
            warmup=args.warmup,
            iters=args.iters,
            delayed=delayed,
        )
        tflops = flops / (ms * 1e-3) / 1e12
        ratio = bl_ms / ms if ms > 0 else 0.0
        flag = "" if ok else f"  !! maxerr={err:.3g}"
        print(f"  {label:62s} {tflops:8.2f} TFLOP/s  {ms:8.3f} ms   " f"{ratio:>7.2f}×{flag}")
        if ok and (best is None or ms < best[1]):
            best = (label, ms, tflops, ratio)

    if best is None:
        print("\n  no strategy produced a correct result for this shape.")
        return 1
    print(f"\n  best: {best[0]}  {best[2]:.2f} TFLOP/s  {best[1]:.3f} ms  " f"{best[3]:.2f}× vs unfused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
