# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark fused dual MoE grouped block-scale matmul + SwiGLU vs an unfused
2×cuBLAS-batched-BF16-GEMM + pointwise baseline. Shared token+SFA are loaded
once, feeding both grouped matmuls. --combo picks nvfp4 / mxfp4 / mxfp8.
"""

from __future__ import annotations

import argparse
import re
import sys
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


def _ceil_div(a, b):
    return (a + b - 1) // b


def _to_blocked(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    nrb, ncb = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _rand_e8m0(shape, dev):
    return torch.randint(125, 129, shape, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)


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


def _offsets(S, E):
    group_m = S // E
    return torch.arange(E, dtype=torch.int32, device="cuda") * group_m


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
        sfa_log = _rand_e8m0((S, sf_k), dev)
        sfb0_log = _rand_e8m0((E, N, sf_k), dev)
        sfb1_log = _rand_e8m0((E, N, sf_k), dev)
    sfa = torch.cat([_to_blocked(sfa_log[g * group_m : (g + 1) * group_m]) for g in range(E)]).view(1, -1, 1)
    sfb0 = torch.cat([_to_blocked(sfb0_log[e]) for e in range(E)]).reshape(E, sf_k, N)
    sfb1 = torch.cat([_to_blocked(sfb1_log[e]) for e in range(E)]).reshape(E, sf_k, N)
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)
    return tok, w0, w1, sfa, sfb0, sfb1, out, scale


def _mkdata_bf16(S, N, K, E):
    dev = "cuda"
    tok = torch.empty(1, S, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device=dev)
    w0 = torch.empty(E, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device=dev)
    w1 = torch.empty(E, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device=dev)
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    return tok, w0, w1, out


def _unfused_launch(tok, w0, w1, out, S, N, K, E):
    """Unfused baseline: 2 cuBLAS batched BF16 GEMMs + pointwise SwiGLU."""
    group_m = S // E
    tok_g = tok.view(E, group_m, K)
    c0 = torch.bmm(tok_g, w0.transpose(-1, -2))
    c1 = torch.bmm(tok_g, w1.transpose(-1, -2))
    res = torch.nn.functional.silu(c0.float()) * c1.float() * 0.5
    out.view(E, group_m, N).copy_(res.to(out.dtype))


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
    """Label -> (geometry cfg, cta_group, scheduler) for multi-GEMM MoE
    block-scale strategies. Dual TMEM fits two accs + SF only at cta_tile_n<=128."""
    chain = analyze(_graph_swiglu(1024, 256, 512, 2, "nvfp4")[0])
    m = {}
    for t, cfg in _registry_candidates(chain):
        if cfg.pipeline != "sm100" or cfg.cta_tile_n > 128 or cfg.cta_tile_m != 128:
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="8,512,4096,4096", help="G,M,N,K (even split)")
    p.add_argument("--combo", choices=tuple(_COMBOS), default="nvfp4")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)  # CLAUDE.md: <= 20
    p.add_argument(
        "--configs",
        default=None,
        help="comma-separated CONFIG_..._Nctamma labels (default: sweep all)",
    )
    p.add_argument("--timing", choices=("delayed", "events"), default="delayed")
    p.add_argument(
        "--stream",
        action="store_true",
        help="accepted for CLI parity; results already print inline",
    )
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
    delayed = args.timing == "delayed"

    flops = 2 * (2 * S * N * K)  # 2 grouped GEMMs
    print(f"\n=== MoE dual block-scale ({combo}) grouped-matmul SwiGLU  " f"G={G} × M={M} (S={S}) {N}x{K}  (~{flops / 1e9:.1f} GFLOP, 2 GEMMs) ===")
    print(f"  [timing: {args.timing}, warmup={args.warmup}, iters={args.iters}]\n")

    tok, w0, w1, sfa, sfb0, sfb1, out, scale = _mkdata(S, N, K, E, combo)
    offsets = _offsets(S, E)

    # --- baseline: unfused 2×cuBLAS batched BF16 GEMM + pointwise ---
    btok, bw0, bw1, bout = _mkdata_bf16(S, N, K, E)
    bl_ms = _time_ms(
        lambda: _unfused_launch(btok, bw0, bw1, bout, S, N, K, E),
        warmup=args.warmup,
        iters=args.iters,
        delayed=delayed,
    )
    bl_tflops = flops / (bl_ms * 1e-3) / 1e12
    print(f"  {'unfused 2xcuBLAS batched bf16 + pointwise':56s} {bl_tflops:8.2f} TFLOP/s  " f"{bl_ms:8.3f} ms   {'1.00×':>8s}")

    config_names = [c.strip() for c in args.configs.split(",")] if args.configs else list(_SPEC_MAP)

    best = None
    for label in config_names:
        spec = _spec_for(label)
        if spec is None:
            print(f"  {label:66s} UNKNOWN (not a sweepable MoE block-scale swiglu strategy)")
            continue
        cfg, cta_group, sched = spec
        try:
            g, h = _graph_swiglu(S, N, K, E, combo)
            plan = _build_plan(g, cfg, cta_group, sched)
        except (NotImplementedError, ValueError):
            continue
        gemm = [((tok, sfa), (w0, sfb0)), ((tok, sfa), (w1, sfb1))]
        try:
            plan(_vp_moe_bs_mg(h, gemm, offsets, out, scale))
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {label:66s} LAUNCH FAIL: {type(e).__name__}: {str(e)[:30]}")
            continue
        ms = _time_ms(
            lambda: plan(_vp_moe_bs_mg(h, gemm, offsets, out, scale)),
            warmup=args.warmup,
            iters=args.iters,
            delayed=delayed,
        )
        tflops = flops / (ms * 1e-3) / 1e12
        ratio = bl_ms / ms if ms > 0 else 0.0
        print(f"  {label:66s} {tflops:8.2f} TFLOP/s  {ms:8.3f} ms   {ratio:>7.2f}×")
        if best is None or ms < best[1]:
            best = (label, ms, tflops, ratio)

    if best is None:
        print("\n  no strategy ran for this shape.")
        return 1
    print(f"\n  best: {best[0]}  {best[2]:.2f} TFLOP/s  {best[1]:.3f} ms  " f"{best[3]:.2f}× vs unfused bf16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
