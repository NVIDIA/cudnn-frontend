# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the fused dual MoE grouped-matmul SwiGLU block vs an unfused
2×cuBLAS-batched-GEMM + pointwise baseline. ``--shape`` is ``G,M,N,K`` (even
split, S=G*M).

    python .../benchmark_moe_grouped_matmul_swiglu.py --shape 8,512,4096,4096
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

from benchmark_utils import add_sweep_args, group_offsets, report_pool, resolve_nbuf, rotating, select_configs, set_bytes, spec_for, time_ms


def _build_plan(g, cfg, cta_group):
    """JIT-compile the graph with a forced tile config → callable kernel."""
    return jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group)


def _vp_moe_mg(handles, gemm_pairs, fto, outs, *aux):
    """MoE multi-GEMM variant-pack dict: dedup (token, weight) pairs → distinct
    A/B slots, + first_token_offset + outputs + aux."""
    bd = handles
    a_seen, b_seen = [], []
    for ag, bg in gemm_pairs:
        if not any(ag is x for x in a_seen):
            a_seen.append(ag)
        if not any(bg is x for x in b_seen):
            b_seen.append(bg)
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {bd.first_token_offset: fto}
    vp.update({t: buf for t, buf in zip(bd.a_operands, a_seen)})
    vp.update({t: buf for t, buf in zip(bd.b_operands, b_seen)})
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


# Graph + data


def _graph_swiglu(S: int, N: int, K: int, E: int):
    """Dual MoE grouped-matmul SwiGLU: silu(moe(tok, w0)) * moe(tok, w1) * scale.
    token is shared by both grouped matmuls (loaded once → two accumulators)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(
        name="token",
        dim=[1, S, K],
        stride=[S * K, K, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    w0 = g.tensor(
        name="weight0",
        dim=[E, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.BFLOAT16,
    )
    w1 = g.tensor(
        name="weight1",
        dim=[E, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.BFLOAT16,
    )
    scale = g.tensor(
        name="scaleFactor",
        dim=[1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    # fto MUST be the SAME tensor for both matmuls (shared routed-group layout).
    fto = g.tensor(
        name="first_token_offset",
        dim=[E, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    c0 = g.moe_grouped_matmul(
        tok,
        w0,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe0",
    )
    c1 = g.moe_grouped_matmul(
        tok,
        w1,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe1",
    )
    s0 = g.swish(input=c0, name="silu0")
    mu = g.mul(a=s0, b=c1, name="mul0")
    dq = g.mul(a=mu, b=scale, name="dequant0")
    dq.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g, SimpleNamespace(
        first_token_offset=fto,
        a_operands=[tok],
        b_operands=[w0, w1],
        outputs=[dq],
        aux=[scale],
    )


def _mkdata(S, N, K, E):
    torch.manual_seed(0)
    tok = torch.randn(1, S, K, device="cuda", dtype=torch.bfloat16) * 0.4
    w0 = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.4
    w1 = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.4
    out = torch.empty(1, S, N, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)
    return tok, w0, w1, out, scale


def _mkdata_pool(S, N, K, E, nbuf):
    """`nbuf` independent (tok, w0, w1, out, scale) sets at distinct GMEM addresses."""
    base = _mkdata(S, N, K, E)
    return [base] + [tuple(t.clone() for t in base) for _ in range(max(0, nbuf - 1))]


def _reference(tok, w0, w1, scale, S, N, K, E):
    """Correctness reference: even-split grouped GEMMs + the elementwise chain."""
    group_m = S // E
    tok_g = tok.view(E, group_m, K).float()
    c0 = torch.bmm(tok_g, w0.transpose(-1, -2).float())
    c1 = torch.bmm(tok_g, w1.transpose(-1, -2).float())
    out = torch.nn.functional.silu(c0) * c1 * scale.flatten()[0]
    return out.reshape(S, N).to(torch.bfloat16)


def _unfused_launch(dset, S, N, K, E):
    """Unfused baseline: 2 cuBLAS batched GEMMs + pointwise."""
    tok, w0, w1, out, scale = dset
    group_m = S // E
    tok_g = tok.view(E, group_m, K)
    c0 = torch.bmm(tok_g, w0.transpose(-1, -2))
    c1 = torch.bmm(tok_g, w1.transpose(-1, -2))
    res = torch.nn.functional.silu(c0.float()) * c1.float() * scale.flatten()[0]
    out.view(E, group_m, N).copy_(res.to(out.dtype))


def _fused_launch(plan, handles, dset, fto):
    tok, w0, w1, out, scale = dset
    plan(_vp_moe_mg(handles, [(tok, w0), (tok, w1)], fto, out, scale))


# Config candidates — MoE templates only (1ctamma cluster1x1, 2ctamma cluster2x1)


def _build_spec_map():
    """Label -> (cfg, cta_group) for multi-GEMM-capable MoE strategies.
    Dual-GEMM TMEM fits two accumulators only for cta_tile_n<=256 (2*256<=512);
    cta_tile_m=128."""
    chain = analyze(_graph_swiglu(2048, 256, 256, 9)[0])
    m = {}
    for t, cfg in _registry_candidates(chain):
        if cfg.pipeline != "sm100" or cfg.cta_tile_n > 256 or cfg.mma_inst_m != 128:
            continue
        label = f"{cfg.name}_{t.cta_group}ctamma"
        m[label] = (cfg, t.cta_group)
    return m


_SPEC_MAP = _build_spec_map()

# Main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="8,512,4096,4096", help="G,M,N,K (even split)")
    add_sweep_args(p, nsys=False)
    p.add_argument("--rtol", type=float, default=5e-2)
    p.add_argument("--atol", type=float, default=2e-1)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be G,M,N,K (four values; even split, S=G*M)")
    G, M, N, K = parts
    E = G
    S = G * M

    flops = 2 * (2 * S * N * K)  # 2 grouped GEMMs, each 2*S*N*K
    print(f"\n=== MoE dual grouped-matmul SwiGLU  G={G} groups × M={M}  " f"(S={S}) {N}x{K}  (~{flops / 1e9:.1f} GFLOP, 2 GEMMs) ===")
    print(f"  [timing: {args.timing}, warmup={args.warmup}, iters={args.iters}]")

    wset = _mkdata(S, N, K, E)  # dedicated warmup set; also what gets verified
    tok, w0, w1, out, scale = wset
    per_set = set_bytes(wset)
    nbuf = resolve_nbuf(args.rotate_buffers, per_set)
    report_pool(nbuf, per_set)
    print()
    pool = _mkdata_pool(S, N, K, E, nbuf)

    offsets = group_offsets(S, E)
    ref = _reference(tok, w0, w1, scale, S, N, K, E)

    # baseline: unfused 2×cuBLAS batched GEMM + pointwise
    if args.stream:
        print("  ▶ running unfused baseline ...", flush=True)
    bl_ms = time_ms(
        rotating(lambda s: _unfused_launch(s, S, N, K, E), pool),
        lambda: _unfused_launch(wset, S, N, K, E),
        warmup=args.warmup,
        iters=args.iters,
        timing=args.timing,
    )
    bl_tflops = flops / (bl_ms * 1e-3) / 1e12
    print(f"  {'unfused 2xcuBLAS batched + pointwise':54s} {bl_tflops:8.2f} TFLOP/s  " f"{bl_ms:8.3f} ms   {'1.00×':>8s}")

    config_names = select_configs(args.configs, _SPEC_MAP)

    best = None
    for label in config_names:
        spec = spec_for(label, _SPEC_MAP)
        if spec is None:
            print(f"  {label:64s} UNKNOWN (not a sweepable MoE swiglu strategy)")
            continue
        cfg, cta_group = spec
        if args.stream:
            print(f"  ▶ running {label} ...", flush=True)
        try:
            g, h = _graph_swiglu(S, N, K, E)
            plan = _build_plan(g, cfg, cta_group)
        except (NotImplementedError, ValueError):
            continue
        try:
            _fused_launch(plan, h, wset, offsets)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {label:64s} LAUNCH FAIL: {type(e).__name__}: {str(e)[:30]}")
            continue
        err = (out.float() - ref.float()).abs().max().item()
        ok = torch.allclose(out.float(), ref.float(), rtol=args.rtol, atol=args.atol)
        ms = time_ms(
            rotating(lambda s, _plan=plan, _h=h: _fused_launch(_plan, _h, s, offsets), pool),
            lambda _plan=plan, _h=h: _fused_launch(_plan, _h, wset, offsets),
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
        )
        tflops = flops / (ms * 1e-3) / 1e12
        ratio = bl_ms / ms if ms > 0 else 0.0
        flag = "" if ok else f"  !! maxerr={err:.3g}"
        print(f"  {label:64s} {tflops:8.2f} TFLOP/s  {ms:8.3f} ms   " f"{ratio:>7.2f}×{flag}", flush=True)
        if ok and (best is None or ms < best[1]):
            best = (label, ms, tflops, ratio)

    if best is None:
        print("\n  no strategy produced a correct result for this shape.")
        return 1
    print(f"\n  best: {best[0]}  {best[2]:.2f} TFLOP/s  {best[1]:.3f} ms  " f"{best[3]:.2f}× vs unfused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
