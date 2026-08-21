# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark fused dual MoE grouped block-scale matmul + SwiGLU vs an unfused
2×cuBLAS-batched-BF16-GEMM + pointwise baseline. Shared token+SFA are loaded
once, feeding both grouped matmuls. --combo picks nvfp4 / mxfp4 / mxfp8.
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

from benchmark_utils import (
    add_sweep_args,
    group_offsets,
    rand_e8m0,
    report_pool,
    resolve_nbuf,
    rotating,
    select_configs,
    set_bytes,
    spec_for,
    time_ms,
    to_blocked,
)


def _build_plan(g, cfg, cta_group):
    """JIT-compile the recorded graph with a forced tile config."""
    return jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group)


def _vp_moe_bs_mg(handles, gemm_pairs, fto, outs, *aux):
    """MoE block-scale multi-GEMM variant-pack dict; each pair is
    ((token, sfa), (weight, sfb)), deduped by packed-data identity."""
    bd = handles
    a_seen, b_seen, sfa_seen, sfb_seen = [], [], [], []
    for (ag, sfag), (bg, sfbg) in gemm_pairs:
        if not any(ag is x for x in a_seen):
            a_seen.append(ag)
            sfa_seen.append(sfag)
        if not any(bg is x for x in b_seen):
            b_seen.append(bg)
            sfb_seen.append(sfbg)
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    vp = {bd.first_token_offset: fto}
    vp.update({t: buf for t, buf in zip(bd.a_operands, a_seen)})
    vp.update({t: buf for t, buf in zip(bd.b_operands, b_seen)})
    vp.update({t: buf for t, buf in zip(bd.sfa_operands, sfa_seen)})
    vp.update({t: buf for t, buf in zip(bd.sfb_operands, sfb_seen)})
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


# combo -> (block_size, data dtype, SF dtype)
_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


def _graph_swiglu(S, N, K, E, combo):
    block_size, a_dt, sf_dt = _COMBOS[combo]
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
    w0 = g.tensor(name="weight0", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
    w1 = g.tensor(name="weight1", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
    SFA = g.tensor(
        name="SFA",
        dim=[1, S, sf_k],
        stride=[S * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB0 = g.tensor(
        name="SFB0",
        dim=[E, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB1 = g.tensor(
        name="SFB1",
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
    sf = g.tensor(
        name="scaleFactor",
        dim=[1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size])
    w0_d = g.block_scale_dequantize(input=w0, descale=SFB0, block_size=[block_size, 1])
    w1_d = g.block_scale_dequantize(input=w1, descale=SFB1, block_size=[block_size, 1])
    c0 = g.moe_grouped_matmul(
        tok_d,
        w0_d,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe0",
    )
    c1 = g.moe_grouped_matmul(
        tok_d,
        w1_d,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe1",
    )
    s0 = g.swish(input=c0, name="silu0")
    mu = g.mul(a=s0, b=c1, name="mul0")
    dq = g.mul(a=mu, b=sf, name="dequant0")
    dq.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g, SimpleNamespace(
        first_token_offset=fto,
        a_operands=[tok],
        b_operands=[w0, w1],
        sfa_operands=[SFA],
        sfb_operands=[SFB0, SFB1],
        outputs=[dq],
        aux=[sf],
    )


def _mkdata(S, N, K, E, combo):
    """Packed FP4/FP8 token+weights + F8_128x4-blocked SFs (even split)."""
    dev = "cuda"
    torch.manual_seed(0)
    block_size, a_dt, _ = _COMBOS[combo]
    is_fp4 = a_dt == cudnn.data_type.FP4_E2M1
    sf_k = K // block_size
    group_m = S // E
    if is_fp4:
        tok = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
        w0 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
        w1 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
    else:
        tok = (torch.randn(1, S, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w0 = (torch.randn(E, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w1 = (torch.randn(E, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb0_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb1_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = rand_e8m0((S, sf_k), dev)
        sfb0_log = rand_e8m0((E, N, sf_k), dev)
        sfb1_log = rand_e8m0((E, N, sf_k), dev)
    sfa = torch.cat([to_blocked(sfa_log[g * group_m : (g + 1) * group_m]) for g in range(E)]).view(1, -1, 1)
    sfb0 = torch.cat([to_blocked(sfb0_log[e]) for e in range(E)]).reshape(E, sf_k, N)
    sfb1 = torch.cat([to_blocked(sfb1_log[e]) for e in range(E)]).reshape(E, sf_k, N)
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)
    return tok, w0, w1, sfa, sfb0, sfb1, out, scale


def _mkdata_pool(S, N, K, E, combo, nbuf):
    """`nbuf` independent token/weight/SF sets at distinct GMEM addresses."""
    base = _mkdata(S, N, K, E, combo)
    pool = [base]
    for _ in range(max(0, nbuf - 1)):
        pool.append(tuple(t.clone() for t in base))
    return pool


def _gemm_pairs(s):
    """((token, SFA), (weight_g, SFB_g)) per GEMM — token+SFA shared."""
    tok, w0, w1, sfa, sfb0, sfb1 = s[:6]
    return [((tok, sfa), (w0, sfb0)), ((tok, sfa), (w1, sfb1))]


def _fused_launch(plan, handles, s, offsets):
    plan(_vp_moe_bs_mg(handles, _gemm_pairs(s), offsets, s[6], s[7]))


def _mkdata_bf16(S, N, K, E):
    dev = "cuda"
    tok = torch.empty(1, S, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device=dev)
    w0 = torch.empty(E, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device=dev)
    w1 = torch.empty(E, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device=dev)
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    return tok, w0, w1, out


def _mkdata_bf16_pool(S, N, K, E, nbuf):
    base = _mkdata_bf16(S, N, K, E)
    pool = [base]
    for _ in range(max(0, nbuf - 1)):
        pool.append(tuple(t.clone() for t in base))
    return pool


def _unfused_launch(tok, w0, w1, out, S, N, K, E):
    """Unfused baseline: 2 cuBLAS batched BF16 GEMMs + pointwise SwiGLU."""
    group_m = S // E
    tok_g = tok.view(E, group_m, K)
    c0 = torch.bmm(tok_g, w0.transpose(-1, -2))
    c1 = torch.bmm(tok_g, w1.transpose(-1, -2))
    res = torch.nn.functional.silu(c0.float()) * c1.float() * 0.5
    out.view(E, group_m, N).copy_(res.to(out.dtype))


def _build_spec_map():
    """Label -> (geometry cfg, cta_group) for multi-GEMM MoE
    block-scale strategies. Dual TMEM fits two accs + SF only at cta_tile_n<=128."""
    chain = analyze(_graph_swiglu(1024, 256, 512, 2, "nvfp4")[0])
    m = {}
    for t, cfg in _registry_candidates(chain):
        if cfg.pipeline not in ("sm100", "sm107") or cfg.cta_tile_n > 128 or cfg.mma_inst_m != 128:
            continue
        label = f"{cfg.name}_{t.cta_group}ctamma"
        m[label] = (cfg, t.cta_group)
    return m


_SPEC_MAP = _build_spec_map()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="8,512,4096,4096", help="G,M,N,K (even split)")
    p.add_argument("--combo", choices=tuple(_COMBOS), default="nvfp4")
    add_sweep_args(p, nsys=False)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be G,M,N,K (four values; even split, S=G*M)")
    G, M, N, K = parts
    E, S = G, G * M
    combo = args.combo

    config_names = select_configs(args.configs, _SPEC_MAP)
    per_set = set_bytes(_mkdata(S, N, K, E, combo))
    nbuf = resolve_nbuf(args.rotate_buffers, per_set)

    flops = 2 * (2 * S * N * K)  # 2 grouped GEMMs
    print(f"\n=== MoE dual block-scale ({combo}) grouped-matmul SwiGLU  " f"G={G} × M={M} (S={S}) {N}x{K}  (~{flops / 1e9:.1f} GFLOP, 2 GEMMs) ===")
    print(f"  [timing: {args.timing}, warmup={args.warmup}, iters={args.iters}]")
    report_pool(nbuf, per_set)
    print()

    offsets = group_offsets(S, E)

    # --- baseline: unfused 2×cuBLAS batched BF16 GEMM + pointwise ---
    wbf = _mkdata_bf16(S, N, K, E)
    bf_pool = _mkdata_bf16_pool(S, N, K, E, nbuf)
    if args.stream:
        print("  ▶ running unfused 2xcuBLAS batched bf16 baseline ...", flush=True)
    bl_ms = time_ms(
        rotating(lambda t: _unfused_launch(t[0], t[1], t[2], t[3], S, N, K, E), bf_pool),
        lambda: _unfused_launch(wbf[0], wbf[1], wbf[2], wbf[3], S, N, K, E),
        warmup=args.warmup,
        iters=args.iters,
        timing=args.timing,
    )
    bl_tflops = flops / (bl_ms * 1e-3) / 1e12
    print(f"  {'unfused 2xcuBLAS batched bf16 + pointwise':56s} {bl_tflops:8.2f} TFLOP/s  " f"{bl_ms:8.3f} ms   {'1.00×':>8s}")
    # Freed before the block-scale pool: on large shapes the BF16 sets dominate.
    del wbf, bf_pool
    torch.cuda.empty_cache()

    wset = _mkdata(S, N, K, E, combo)  # dedicated warmup set, never rotated
    pool = _mkdata_pool(S, N, K, E, combo, nbuf)

    best = None
    for label in config_names:
        spec = spec_for(label, _SPEC_MAP)
        if spec is None:
            print(f"  {label:66s} UNKNOWN (not a sweepable MoE block-scale swiglu strategy)")
            continue
        cfg, cta_group = spec
        if args.stream:
            print(f"  ▶ running {label} ...", flush=True)
        try:
            g, h = _graph_swiglu(S, N, K, E, combo)
            plan = _build_plan(g, cfg, cta_group)
        except (NotImplementedError, ValueError):
            continue
        try:
            _fused_launch(plan, h, wset, offsets)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {label:66s} LAUNCH FAIL: {type(e).__name__}: {str(e)[:30]}")
            continue
        ms = time_ms(
            rotating(lambda s, _plan=plan, _h=h: _fused_launch(_plan, _h, s, offsets), pool),
            lambda _plan=plan, _h=h: _fused_launch(_plan, _h, wset, offsets),
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
        )
        tflops = flops / (ms * 1e-3) / 1e12
        ratio = bl_ms / ms if ms > 0 else 0.0
        print(f"  {label:66s} {tflops:8.2f} TFLOP/s  {ms:8.3f} ms   {ratio:>7.2f}×", flush=True)
        if best is None or ms < best[1]:
            best = (label, ms, tflops, ratio)

    if best is None:
        print("\n  no strategy ran for this shape.")
        return 1
    print(f"\n  best: {best[0]}  {best[2]:.2f} TFLOP/s  {best[1]:.3f} ms  " f"{best[3]:.2f}× vs unfused bf16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
