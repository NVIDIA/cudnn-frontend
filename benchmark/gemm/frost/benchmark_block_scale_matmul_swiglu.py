# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the fused nvfp4 block-scale dual-matmul SwiGLU vs an unfused baseline.

The shared block_scale_dequantize(a) is matched into BOTH GEMMs → one distinct A
operand (+ SFA), loaded once. Runs on 1-CTA-MMA templates; cta_n capped at 128
(2×cta_n + SF must fit 512 TMEM cols). --iters <= 20 (CLAUDE.md).
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


def _vp_bs_mg(handles, gemm_pairs, outs, *aux):
    """Block-scale multi-GEMM variant-pack dict. Pairs ((a, sfa), (b, sfb)) dedup
    by packed-data identity into distinct A/B slots (SF travels with its data)."""
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
    vp = {}
    vp.update({t: buf for t, buf in zip(bd.a_operands, a_seen)})
    vp.update({t: buf for t, buf in zip(bd.b_operands, b_seen)})
    vp.update({t: buf for t, buf in zip(bd.sfa_operands, sfa_seen)})
    vp.update({t: buf for t, buf in zip(bd.sfb_operands, sfb_seen)})
    vp.update({o: buf for o, buf in zip(bd.outputs, outs)})
    vp.update({x: buf for x, buf in zip(bd.aux, aux)})
    return vp


# nvfp4 packing/reorder helpers (matches test/python/gemm/frost/test_block_scale.py).
_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _to_blocked(x):
    rows, cols = x.shape
    nrb, ncb = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _unpack_fp4(u8, lut):
    lo = lut[(u8 & 0xF).long()]
    hi = lut[(u8 >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


# --- Graph + data (nvfp4) ---
def _graph(B, M, N, K, bs=16):
    sf_k = K // bs
    fp4, e4m3 = cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3
    rk = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[B, M, K], stride=[M * K, K, 1], data_type=fp4)
    SFA = g.tensor(name="SFA", dim=[B, M, sf_k], stride=[M * sf_k, sf_k, 1], data_type=e4m3, **rk)
    B0 = g.tensor(name="B0", dim=[B, K, N], stride=[K * N, 1, K], data_type=fp4)
    SFB0 = g.tensor(name="SFB0", dim=[B, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=e4m3, **rk)
    B1 = g.tensor(name="B1", dim=[B, K, N], stride=[K * N, 1, K], data_type=fp4)
    SFB1 = g.tensor(name="SFB1", dim=[B, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=e4m3, **rk)
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, bs])
    B0d = g.block_scale_dequantize(input=B0, descale=SFB0, block_size=[bs, 1])
    B1d = g.block_scale_dequantize(input=B1, descale=SFB1, block_size=[bs, 1])
    C0 = g.matmul(A=Ad, B=B0d, name="mm0")
    C1 = g.matmul(A=Ad, B=B1d, name="mm1")
    Y = g.mul(a=g.swish(input=C0), b=C1)
    Y.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return g, SimpleNamespace(
        a_operands=[A],
        b_operands=[B0, B1],
        sfa_operands=[SFA],
        sfb_operands=[SFB0, SFB1],
        outputs=[Y],
        aux=[],
    )


def _mkdata(B, M, N, K, bs=16):
    dev = "cuda"
    sf_k = K // bs
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    def _mk(rows):
        u8 = torch.randint(0, 256, (B, rows, K // 2), dtype=torch.uint8, device=dev)
        return u8.view(torch.float4_e2m1fn_x2), _unpack_fp4(u8, lut).view(B, rows, K)

    a_rt, a_deq = _mk(M)
    b0_rt, b0_deq = _mk(N)
    b1_rt, b1_deq = _mk(N)

    def _sf(rows):
        log = torch.randint(1, 4, (B, rows, sf_k), device=dev).to(torch.float8_e4m3fn)
        # _to_blocked pads to whole 128-row x 4-SF-K atoms — view the PADDED dims.
        blk = torch.stack([_to_blocked(log[b]) for b in range(B)]).view(B, _ceil_div(rows, 128) * 128, _ceil_div(sf_k, 4) * 4)
        return log, blk

    sfa_log, sfa_b = _sf(M)
    sfb0_log, sfb0_b = _sf(N)
    sfb1_log, sfb1_b = _sf(N)

    a_s = a_deq * sfa_log.float().repeat_interleave(bs, 2)
    b0_s = b0_deq * sfb0_log.float().repeat_interleave(bs, 2)
    b1_s = b1_deq * sfb1_log.float().repeat_interleave(bs, 2)
    pairs = [((a_rt, sfa_b), (b0_rt, sfb0_b)), ((a_rt, sfa_b), (b1_rt, sfb1_b))]
    return pairs, (a_s, b0_s, b1_s)


def _reference(a_s, b0_s, b1_s):
    c0 = torch.einsum("bmk,bnk->bmn", a_s, b0_s)
    c1 = torch.einsum("bmk,bnk->bmn", a_s, b1_s)
    return torch.nn.functional.silu(c0) * c1


def _unfused_launch(a_s, b0_s, b1_s, out):
    """Baseline: dequantized fp16 operands → 2 cuBLAS GEMMs + pointwise."""
    a, b0, b1 = a_s.half(), b0_s.half(), b1_s.half()
    c0 = torch.matmul(a, b0.transpose(-1, -2))
    c1 = torch.matmul(a, b1.transpose(-1, -2))
    out.copy_(torch.nn.functional.silu(c0.float()) * c1.float())


# --- Timing (delayed / events) ---
def _time_ms(timed_fn: Callable, *, warmup: int, iters: int, delayed: bool) -> float:
    for _ in range(warmup):
        timed_fn()
    torch.cuda.synchronize()
    if delayed:
        torch.cuda._sleep(max(int(1e8), int((iters * 0.05 + 20.0) * 1.7e6)))
        for _ in range(max(5, warmup)):
            timed_fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        timed_fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _build_spec_map():
    """Label CONFIG_..._Nctamma[_static] -> (cfg, cta_group, scheduler) for every
    multi-GEMM-capable sm100 block-scale strategy. Pins cta_tile_n=128 (SF 128x4
    swizzle + dual-GEMM TMEM: 256 overflows, 64/32 break the swizzle)."""
    chain = analyze(_graph(1, 256, 128, 512)[0])
    m = {}
    for t, cfg in _registry_candidates(chain):
        if cfg.pipeline != "sm100" or cfg.cta_tile_m != 128 or cfg.cta_tile_n != 128:
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
    p.add_argument("--shape", default="1,4096,4096,4096", help="B,M,N,K")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)  # CLAUDE.md: <= 20
    p.add_argument("--configs", default=None)
    p.add_argument("--timing", choices=("delayed", "events"), default="delayed")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be B,M,N,K")
    B, M, N, K = parts
    delayed = args.timing == "delayed"
    flops = 2 * (2 * B * M * N * K)
    print(f"\n=== block-scale SwiGLU dual-matmul (nvfp4)  B={B} {M}x{N}x{K}  " f"(~{flops / 1e9:.1f} GFLOP, 2 GEMMs) ===")
    print(f"  [timing: {args.timing}, warmup={args.warmup}, iters={args.iters}]\n")

    pairs, (a_s, b0_s, b1_s) = _mkdata(B, M, N, K)
    ref = _reference(a_s, b0_s, b1_s)
    out = torch.zeros(B, M, N, dtype=torch.float32, device="cuda")

    out_bl = torch.empty_like(out)
    bl_ms = _time_ms(
        lambda: _unfused_launch(a_s, b0_s, b1_s, out_bl),
        warmup=args.warmup,
        iters=args.iters,
        delayed=delayed,
    )
    print(f"  {'unfused dequant+2xcuBLAS+pointwise':52s} " f"{flops / (bl_ms * 1e-3) / 1e12:8.2f} TFLOP/s  {bl_ms:8.3f} ms   {'1.00×':>8s}")

    config_names = [c.strip() for c in args.configs.split(",")] if args.configs else list(_SPEC_MAP)

    best = None
    for name in config_names:
        spec = _spec_for(name)
        if spec is None:
            print(f"  {name:62s} UNKNOWN (not a sweepable block-scale strategy)")
            continue
        cfg, cta_group, sched = spec
        try:
            g, h = _graph(B, M, N, K)
            plan = _build_plan(g, cfg, cta_group, sched)
        except (NotImplementedError, ValueError) as e:
            print(f"  {name:62s} SKIP: {str(e)[:42]}")
            continue
        try:
            plan(_vp_bs_mg(h, pairs, out))
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {name:62s} LAUNCH FAIL: {type(e).__name__}: {str(e)[:30]}")
            continue
        ok = torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-1)  # swish: fast approx
        err = (out.float() - ref.float()).abs().max().item()
        ms = _time_ms(
            lambda: plan(_vp_bs_mg(h, pairs, out)),
            warmup=args.warmup,
            iters=args.iters,
            delayed=delayed,
        )
        ratio = bl_ms / ms if ms > 0 else 0.0
        flag = "" if ok else f"  !! maxerr={err:.3g}"
        print(f"  {name:62s} {flops / (ms * 1e-3) / 1e12:8.2f} TFLOP/s  " f"{ms:8.3f} ms   {ratio:>7.2f}×{flag}")
        if ok and (best is None or ms < best[1]):
            best = (name, ms, flops / (ms * 1e-3) / 1e12, ratio)

    if best is None:
        print("\n  no strategy produced a correct result for this shape.")
        return 1
    print(f"\n  best: {best[0]}  {best[2]:.2f} TFLOP/s  {best[1]:.3f} ms  " f"{best[3]:.2f}× vs unfused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
