# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the fused multi-GEMM SwiGLU block silu(a@b0)*(a@b1)*scale vs an
unfused 2×cuBLAS + pointwise baseline. --shape is B,M,N,K.

    python benchmark/gemm/frost/benchmark_matmul_swiglu.py --shape 1,4096,11008,4096
"""

from __future__ import annotations

import argparse
import sys

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import torch

from types import SimpleNamespace

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import candidates as _registry_candidates

from benchmark_utils import add_sweep_args, report_pool, resolve_nbuf, rotating, select_configs, set_bytes, spec_for, time_ms


def _build_plan(g, cfg, cta_group):
    """JIT-compile the recorded graph with a forced tile config."""
    return jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group)


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


def _mkdata_pool(B, M, N, K, in_dt, out_dt, nbuf):
    """`nbuf` independent (a, b0, b1, out, scale) sets at distinct GMEM addresses."""
    base = _mkdata(B, M, N, K, in_dt, out_dt)
    pool = [base]
    for _ in range(max(0, nbuf - 1)):
        pool.append(tuple(t.clone() for t in base))
    return pool


def _unpack(s):
    """A pooled set is (a, b0, b1, out, scale); _unfused_launch wants scale before out."""
    a, b0, b1, out, scale = s
    return a, b0, b1, scale, out


def _gemm_args(s):
    """A pooled set -> _vp_mg's (gemm_pairs, outs, *aux)."""
    a, b0, b1, out, scale = s
    return [(a, b0), (a, b1)], out, scale


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


def _build_spec_map():
    """Legacy label -> (geometry cfg, cta_group) for every multi-GEMM-
    capable sm100 matmul strategy. Multi-GEMM TMEM fits num_gemms accumulators
    only for cta_tile_n<=256 (2*256<=512); cta_tile_m=128."""
    chain = analyze(_graph_swiglu(1, 256, 256, 256, "bf16", "bf16")[0])
    m = {}
    for t, cfg in _registry_candidates(chain):
        if cfg.pipeline != "sm100" or cfg.cta_tile_n > 256 or cfg.cgrp_size_n != 1 or cfg.mma_inst_m != 128:
            continue
        label = f"{cfg.name}_{t.cta_group}ctamma"
        m[label] = (cfg, t.cta_group)
    return m


_SPEC_MAP = _build_spec_map()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="1,4096,4096,4096", help="B,M,N,K")
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    add_sweep_args(p, nsys=False)
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

    # 2 GEMMs, each 2*B*M*N*K flops.
    flops = 2 * (2 * B * M * N * K)
    print(f"\n=== SwiGLU dual-matmul  B={B} {M}x{N}x{K}  " f"(~{flops / 1e9:.1f} GFLOP, 2 GEMMs) — {in_dt} in / {out_dt} out ===")
    print(f"  [timing: {args.timing}, warmup={args.warmup}, iters={args.iters}]")

    wa, wb0, wb1, w_out, wscale = _mkdata(B, M, N, K, in_dt, out_dt)
    per_set = set_bytes((wa, wb0, wb1, w_out, wscale))
    nbuf = resolve_nbuf(args.rotate_buffers, per_set)
    report_pool(nbuf, per_set)
    print()
    pool = _mkdata_pool(B, M, N, K, in_dt, out_dt, nbuf)
    ref = _reference(wa, wb0, wb1, wscale, out_dt)

    # --- baseline: unfused 2×cuBLAS + pointwise ---
    out_bl = torch.empty_like(w_out)
    if args.stream:
        print("  ▶ running unfused baseline ...", flush=True)
    bl_ms = time_ms(
        rotating(lambda s: _unfused_launch(*_unpack(s)), pool),
        lambda: _unfused_launch(wa, wb0, wb1, wscale, out_bl),
        warmup=args.warmup,
        iters=args.iters,
        timing=args.timing,
    )
    bl_tflops = flops / (bl_ms * 1e-3) / 1e12
    print(f"  {'unfused 2xcuBLAS + pointwise':52s} {bl_tflops:8.2f} TFLOP/s  " f"{bl_ms:8.3f} ms   {'1.00×':>8s}")

    # --- candidate (config, cta_group) strategies ---
    config_names = select_configs(args.configs, _SPEC_MAP)

    best = None
    for label in config_names:
        spec = spec_for(label, _SPEC_MAP)
        if spec is None:
            print(f"  {label:62s} UNKNOWN (not a sweepable swiglu strategy)")
            continue
        cfg, cta_group = spec
        if args.stream:
            print(f"  ▶ running {label} ...", flush=True)
        try:
            g, h = _graph_swiglu(B, M, N, K, in_dt, out_dt)
            plan = _build_plan(g, cfg, cta_group)
        except (NotImplementedError, ValueError):
            continue  # geometry/strategy can't run this shape/dtype — skip
        try:
            plan(_vp_mg(h, [(wa, wb0), (wa, wb1)], w_out, wscale))
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {label:62s} LAUNCH FAIL: {type(e).__name__}: {str(e)[:30]}")
            continue
        err = (w_out.float() - ref.float()).abs().max().item()
        ok = torch.allclose(w_out.float(), ref.float(), rtol=args.rtol, atol=args.atol)
        ms = time_ms(
            rotating(lambda s, _plan=plan, _h=h: _plan(_vp_mg(_h, *_gemm_args(s))), pool),
            lambda _plan=plan, _h=h: _plan(_vp_mg(_h, [(wa, wb0), (wa, wb1)], w_out, wscale)),
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
        )
        tflops = flops / (ms * 1e-3) / 1e12
        ratio = bl_ms / ms if ms > 0 else 0.0
        flag = "" if ok else f"  !! maxerr={err:.3g}"
        print(f"  {label:62s} {tflops:8.2f} TFLOP/s  {ms:8.3f} ms   " f"{ratio:>7.2f}×{flag}", flush=True)
        if ok and (best is None or ms < best[1]):
            best = (label, ms, tflops, ratio)

    if best is None:
        print("\n  no strategy produced a correct result for this shape.")
        return 1
    print(f"\n  best: {best[0]}  {best[2]:.2f} TFLOP/s  {best[1]:.3f} ms  " f"{best[3]:.2f}× vs unfused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
