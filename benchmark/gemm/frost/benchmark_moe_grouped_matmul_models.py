# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the fused dual MoE grouped-matmul GLU block on the production
model shapes, sweeping the FROST tile configs.

Shapes / routed-group offsets and the per-model epilogue come from
``all_tests/model_benchmark.json`` (the cudnnTest abstract graphs):

    swiglu          silu(gate) * up                                (GLM 5.2)
    swiglu_clamped  silu(min(gate, hi)) * clamp(up, lo, hi)        (Deepseek V4)
    swiglu_oai      gate' * sigmoid(a*gate') * (clamp(up) + 1)     (Minimax M3)
    situglu         b*tanh(gate/b)*sigmoid(gate) * lb*tanh(up/lb)  (Kimi K3)

A model fixes only ``N``, ``K`` and the epilogue; the workload — token count ``-S``
and expert count ``-E`` — is yours to pick and applies to every selected model.
Routed groups split those tokens as evenly as possible, the first ``S % E`` groups
taking one extra token: ``S=10, E=3`` -> sizes 4, 3, 3 -> offsets 0, 4, 7.

``--dtype`` picks the ...BF16 or the ...MXFP8 flavour of the same graphs: mxfp8
loads token/weights as E4M3 + per-32-block E8M0 scales (F8_128x4 reordered) and
block_scale_dequantize's them into the grouped matmuls; the epilogue and the
BF16 output are identical.

    python .../benchmark_moe_grouped_matmul_models.py --models glm52 -S 16384 -E 16
    python .../benchmark_moe_grouped_matmul_models.py --models glm52 -S 4096 -E 8 --dtype mxfp8
    python .../benchmark_moe_grouped_matmul_models.py -S 8192 -E 8 --configs CONFIG_..._2ctamma
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import candidates as _registry_candidates

from benchmark_utils import (
    add_sweep_args,
    ceil_div,
    even_offsets,
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

# Model cases

MODELS: dict[str, dict] = {
    "dsv4_pro": dict(
        label="Deepseek V4 pro",
        variant="swiglu_clamped",
        N=3072,
        K=7168,
    ),
    "dsv4_flash": dict(
        label="Deepseek V4 flash",
        variant="swiglu_clamped",
        N=2048,
        K=4096,
    ),
    "glm52": dict(
        label="GLM 5.2",
        variant="swiglu",
        N=2048,
        K=6144,
    ),
    "kimi_k3": dict(
        label="Kimi K3",
        variant="situglu",
        N=3072,
        K=3584,
    ),
    "minimax_m3": dict(
        label="Minimax M3",
        variant="swiglu_oai",
        N=3072,
        K=6144,
    ),
}

VARIANTS = ("swiglu", "swiglu_clamped", "swiglu_oai", "situglu")
DTYPES = ("bf16", "mxfp8")

_MXFP8_BLOCK = 32

# Epilogue scalar constants, per model_benchmark.json.
_CLAMP_LIMIT = 10.0
_OAI_LIMIT = 7.0
_OAI_ALPHA = 1.702
_OAI_BIAS = 1.0
_SITU_BETA = 4.0
_SITU_LINEAR_BETA = 25.0


# Graph


def _sf_rows(offsets: list[int], S: int) -> int:
    """SFA height: every routed group is padded to 128 rows in the F8_128x4 blob,
    so it is Σ ceil(group_m/128)*128 — equal to S only when every group size is a
    multiple of 128 (Kimi K3's 585/586-token groups are not: 14*640 = 8960)."""
    return sum(ceil_div(hi - lo, 128) * 128 for lo, hi in _group_ranges(offsets, S))


def _operands(g, S: int, N: int, K: int, E: int, dtype: str, offsets: list[int]):
    """Token + two weights, either plain BF16 or MXFP8 (E4M3 data + per-32-block
    E8M0 scales, F8_128x4 reordered) dequantized into the matmuls. Returns the
    matmul-facing tensors plus the packed-data / SF tensors for the variant pack."""
    if dtype == "bf16":
        tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
        w0 = g.tensor(name="weight0", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
        w1 = g.tensor(name="weight1", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
        return (tok, w0, w1), [tok], [w0, w1], [], []

    sf_k = K // _MXFP8_BLOCK
    sf_m = _sf_rows(offsets, S)
    fp8, e8m0 = cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0
    reorder = cudnn.tensor_reordering.F8_128x4
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=fp8)
    w0 = g.tensor(name="weight0", dim=[E, K, N], stride=[K * N, 1, K], data_type=fp8)
    w1 = g.tensor(name="weight1", dim=[E, K, N], stride=[K * N, 1, K], data_type=fp8)
    sfa = g.tensor(name="SFA", dim=[1, sf_m, sf_k], stride=[sf_m * sf_k, sf_k, 1], data_type=e8m0, reordering_type=reorder)
    sfb0 = g.tensor(name="SFB0", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=e8m0, reordering_type=reorder)
    sfb1 = g.tensor(name="SFB1", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=e8m0, reordering_type=reorder)
    tok_d = g.block_scale_dequantize(input=tok, descale=sfa, block_size=[1, _MXFP8_BLOCK])
    w0_d = g.block_scale_dequantize(input=w0, descale=sfb0, block_size=[_MXFP8_BLOCK, 1])
    w1_d = g.block_scale_dequantize(input=w1, descale=sfb1, block_size=[_MXFP8_BLOCK, 1])
    return (tok_d, w0_d, w1_d), [tok], [w0, w1], [sfa], [sfb0, sfb1]


def _graph(S: int, N: int, K: int, E: int, variant: str, dtype: str = "bf16", offsets: list[int] | None = None):
    """Dual MoE grouped matmul (token shared → loaded once, two accumulators)
    feeding the model's GLU epilogue. ``offsets`` sizes the SFA blob under mxfp8
    (defaults to an even split, which is what the config sweep probes with)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    offsets = offsets if offsets is not None else even_offsets(S, E)
    (a, b0, b1), a_ops, b_ops, sfa_ops, sfb_ops = _operands(g, S, N, K, E, dtype, offsets)
    # fto MUST be the SAME tensor for both matmuls (shared routed-group layout).
    fto = g.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)

    consts: list[tuple] = []

    def const(name: str, value: float):
        t = g.tensor(name=name, dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
        consts.append((t, value))
        return t

    moe = dict(mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT)
    gate = g.moe_grouped_matmul(a, b0, fto, name="mm0", **moe)
    up = g.moe_grouped_matmul(a, b1, fto, name="mm1", **moe)

    if variant == "swiglu":
        y = g.mul(a=g.swish(input=gate, name="silu0"), b=up, name="mul0")
    elif variant == "swiglu_clamped":
        hi = const("highLimit", _CLAMP_LIMIT)
        lo = const("lowLimit", -_CLAMP_LIMIT)
        gate_c = g.min(input0=gate, input1=hi, name="cap0")
        up_c = g.min(input0=up, input1=hi, name="cap1")
        up_cl = g.max(input0=up_c, input1=lo, name="clamp0")
        y = g.mul(a=g.swish(input=gate_c, name="silu0"), b=up_cl, name="mul0")
    elif variant == "swiglu_oai":
        hi = const("highLimit", _OAI_LIMIT)
        lo = const("lowLimit", -_OAI_LIMIT)
        alpha = const("scale", _OAI_ALPHA)
        bias = const("bias", _OAI_BIAS)
        gate_c = g.min(input0=gate, input1=hi, name="cap0")
        up_c = g.min(input0=up, input1=hi, name="cap1")
        up_cl = g.max(input0=up_c, input1=lo, name="clamp0")
        gate_s = g.mul(a=gate_c, b=alpha, name="scale0")
        gate_sig = g.sigmoid(input=gate_s, name="sigmoid0")
        gate_silu = g.mul(a=gate_c, b=gate_sig, name="mul0")
        up_b = g.add(a=up_cl, b=bias, name="bias0")
        y = g.mul(a=gate_silu, b=up_b, name="mul1")
    elif variant == "situglu":
        beta = const("situBeta", _SITU_BETA)
        beta_inv = const("situBetaInverse", 1.0 / _SITU_BETA)
        lbeta = const("situLinearBeta", _SITU_LINEAR_BETA)
        lbeta_inv = const("situLinearBetaInverse", 1.0 / _SITU_LINEAR_BETA)
        gate_s = g.mul(a=gate, b=beta_inv, name="mul0")
        gate_t = g.tanh(input=gate_s, name="tanh0")
        gate_sig = g.sigmoid(input=gate, name="sigmoid0")
        gate_p = g.mul(a=gate_t, b=gate_sig, name="mul1")
        gate_situ = g.mul(a=gate_p, b=beta, name="mul2")
        up_s = g.mul(a=up, b=lbeta_inv, name="mul3")
        up_t = g.tanh(input=up_s, name="tanh1")
        up_r = g.mul(a=up_t, b=lbeta, name="mul4")
        y = g.mul(a=gate_situ, b=up_r, name="mul5")
    else:
        raise ValueError(f"unknown variant {variant!r}")

    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g, SimpleNamespace(
        first_token_offset=fto,
        a_operands=a_ops,
        b_operands=b_ops,
        sfa_operands=sfa_ops,
        sfb_operands=sfb_ops,
        outputs=[y],
        aux=[t for t, _ in consts],
        aux_values=[v for _, v in consts],
    )


def _vp(handles, d, fto, out, aux_bufs):
    """MoE multi-GEMM variant-pack dict: distinct A/B operands (token shared →
    one A slot) + first_token_offset + output + scalar auxes. Under mxfp8 each
    operand's scale-factor tensor rides along in the same dict."""
    vp = {handles.first_token_offset: fto, handles.a_operands[0]: d.tok, handles.outputs[0]: out}
    vp.update({t: b for t, b in zip(handles.b_operands, (d.w0, d.w1))})
    vp.update({t: b for t, b in zip(handles.sfa_operands, (d.sfa,))})
    vp.update({t: b for t, b in zip(handles.sfb_operands, (d.sfb0, d.sfb1))})
    vp.update({t: b for t, b in zip(handles.aux, aux_bufs)})
    return vp


# Data + reference


def _dequant_bf16(data: torch.Tensor, sf_log: torch.Tensor) -> torch.Tensor:
    """E4M3 x per-32-block E8M0 -> BF16, exactly (3-bit mantissa, power-of-two
    scale), so the BF16 reference sees the kernel's true operand values.
    Chunked over rows to bound the fp32 temporary."""
    rows = data.reshape(-1, data.shape[-1])
    scales = sf_log.reshape(-1, sf_log.shape[-1])
    deq = torch.empty(rows.shape, dtype=torch.bfloat16, device=data.device)
    for i in range(0, rows.shape[0], 4096):
        j = min(i + 4096, rows.shape[0])
        deq[i:j] = (rows[i:j].float() * scales[i:j].float().repeat_interleave(_MXFP8_BLOCK, 1)).to(torch.bfloat16)
    return deq.view(data.shape)


def _mkdata(S: int, N: int, K: int, E: int, offsets: list[int], dtype: str, need_ref: bool = True):
    """Runtime buffers. ``*_ref`` are the BF16 operand values the kernel actually
    multiplies (identical tensors under bf16, dequantized ones under mxfp8) and
    drive both the reference and the unfused cuBLAS baseline. Under mxfp8 they are
    a second, wider copy of every operand, so ``need_ref=False`` skips them when
    neither consumer is running."""
    torch.manual_seed(0)
    dev = "cuda"
    out = torch.empty(1, S, N, device=dev, dtype=torch.bfloat16)
    if dtype == "bf16":
        tok = torch.randn(1, S, K, device=dev, dtype=torch.bfloat16) * 0.4
        w0 = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) * 0.4
        w1 = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) * 0.4
        return SimpleNamespace(tok=tok, w0=w0, w1=w1, sfa=None, sfb0=None, sfb1=None, tok_ref=tok, w0_ref=w0, w1_ref=w1, out=out)

    sf_k = K // _MXFP8_BLOCK
    tok = (torch.randn(1, S, K, device=dev) * 0.4).to(torch.float8_e4m3fn)
    w0 = (torch.randn(E, N, K, device=dev) * 0.4).to(torch.float8_e4m3fn)
    w1 = (torch.randn(E, N, K, device=dev) * 0.4).to(torch.float8_e4m3fn)
    sfa_log = rand_e8m0((S, sf_k), dev)
    sfb0_log = rand_e8m0((E, N, sf_k), dev)
    sfb1_log = rand_e8m0((E, N, sf_k), dev)
    # SFA is blocked PER routed group (each padded to 128 rows) then concatenated —
    # the kernel walks the same ceil(group_m/128) SF-block prefix. SFB is per expert.
    sfa = torch.cat([to_blocked(sfa_log[lo:hi]) for lo, hi in _group_ranges(offsets, S)]).view(1, -1, 1)
    sfb0 = torch.cat([to_blocked(sfb0_log[e]) for e in range(E)]).view(E, sf_k, N)
    sfb1 = torch.cat([to_blocked(sfb1_log[e]) for e in range(E)]).view(E, sf_k, N)
    return SimpleNamespace(
        tok=tok,
        w0=w0,
        w1=w1,
        sfa=sfa,
        sfb0=sfb0,
        sfb1=sfb1,
        tok_ref=_dequant_bf16(tok, sfa_log.view(1, S, sf_k)) if need_ref else None,
        w0_ref=_dequant_bf16(w0, sfb0_log) if need_ref else None,
        w1_ref=_dequant_bf16(w1, sfb1_log) if need_ref else None,
        out=out,
    )


def _mkdata_pool(S: int, N: int, K: int, E: int, offsets: list[int], dtype: str, nbuf: int):
    """``nbuf`` independent operand sets at distinct GMEM addresses. They feed only
    the timed launches, so they skip the widened reference copies."""
    return [_mkdata(S, N, K, E, offsets, dtype, need_ref=False) for _ in range(nbuf)]


def _epilogue_ref(gate: torch.Tensor, up: torch.Tensor, variant: str) -> torch.Tensor:
    """fp32 closed form of each model_benchmark.json epilogue."""
    if variant == "swiglu":
        return torch.nn.functional.silu(gate) * up
    if variant == "swiglu_clamped":
        gate_c = gate.clamp(max=_CLAMP_LIMIT)
        up_cl = up.clamp(max=_CLAMP_LIMIT).clamp(min=-_CLAMP_LIMIT)
        return torch.nn.functional.silu(gate_c) * up_cl
    if variant == "swiglu_oai":
        gate_c = gate.clamp(max=_OAI_LIMIT)
        up_cl = up.clamp(max=_OAI_LIMIT).clamp(min=-_OAI_LIMIT)
        return gate_c * torch.sigmoid(_OAI_ALPHA * gate_c) * (up_cl + _OAI_BIAS)
    if variant == "situglu":
        gate_situ = _SITU_BETA * torch.tanh(gate / _SITU_BETA) * torch.sigmoid(gate)
        up_r = _SITU_LINEAR_BETA * torch.tanh(up / _SITU_LINEAR_BETA)
        return gate_situ * up_r
    raise ValueError(f"unknown variant {variant!r}")


def _group_ranges(offsets: list[int], S: int) -> list[tuple[int, int]]:
    return [(offsets[g], offsets[g + 1] if g + 1 < len(offsets) else S) for g in range(len(offsets))]


def _reference(tok, w0, w1, offsets, S, N, K, E, variant) -> torch.Tensor:
    """Per-routed-group grouped GEMM + the model epilogue, group g -> expert g%E."""
    ref = torch.empty(S, N, device="cuda", dtype=torch.bfloat16)
    tok2 = tok.view(S, K)
    for g, (lo, hi) in enumerate(_group_ranges(offsets, S)):
        if hi <= lo:
            continue
        e = g % E
        rows = tok2[lo:hi]
        gate = (rows @ w0[e].transpose(0, 1)).float()
        up = (rows @ w1[e].transpose(0, 1)).float()
        ref[lo:hi] = _epilogue_ref(gate, up, variant).to(torch.bfloat16)
    return ref


def _unfused_launch(tok, w0, w1, out, offsets, S, N, K, E, variant) -> None:
    """Unfused baseline: 2 cuBLAS GEMMs per routed group + pointwise."""
    tok2 = tok.view(S, K)
    out2 = out.view(S, N)
    for g, (lo, hi) in enumerate(_group_ranges(offsets, S)):
        if hi <= lo:
            continue
        e = g % E
        rows = tok2[lo:hi]
        gate = (rows @ w0[e].transpose(0, 1)).float()
        up = (rows @ w1[e].transpose(0, 1)).float()
        out2[lo:hi] = _epilogue_ref(gate, up, variant).to(out.dtype)


# Config candidates — MoE templates only, dual-GEMM TMEM fits two accumulators
# only for cta_tile_n <= 256 (2*256 <= 512), and only <= 128 under mxfp8 where the
# per-operand SF shares those TMEM columns; cta_tile_m=128.


def _build_spec_map(variant: str, dtype: str) -> dict[str, tuple]:
    chain = analyze(_graph(2048, 256, 256, 9, variant, dtype)[0])
    n_cap = 128 if dtype == "mxfp8" else 256
    m = {}
    for t, cfg in _registry_candidates(chain):
        if cfg.pipeline != "sm100" or cfg.cta_tile_n > n_cap or cfg.mma_inst_m != 128:
            continue
        label = f"{cfg.name}_{t.cta_group}ctamma"
        m[label] = (cfg, t.cta_group)
    return m


# Main


def _run_model(key: str, spec: dict, args) -> tuple | None:
    N, K = spec["N"], spec["K"]
    S, E = args.tokens, args.experts
    offsets = even_offsets(S, E)
    variant = spec["variant"]

    flops = 2 * (2 * S * N * K)  # 2 grouped GEMMs, each 2*S*N*K
    group_sizes = [hi - lo for lo, hi in _group_ranges(offsets, S)]
    print(f"\n=== {spec['label']}  [{key}]  {variant}  {args.dtype} ===", flush=True)
    print(f"  E={E} S={S} N={N} K={K}  groups={min(group_sizes)}..{max(group_sizes)} tokens  " f"(~{flops / 1e9:.1f} GFLOP, 2 GEMMs)", flush=True)
    print(f"  [timing: {args.timing}, warmup={args.warmup}, iters={args.iters}]", flush=True)

    d = _mkdata(S, N, K, E, offsets, args.dtype, need_ref=not (args.no_verify and args.no_baseline))
    out = d.out
    fto = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    per_set = set_bytes([t for t in (d.tok, d.w0, d.w1, d.sfa, d.sfb0, d.sfb1, d.out) if t is not None])
    nbuf = resolve_nbuf(args.rotate_buffers, per_set)
    report_pool(nbuf, per_set)
    pool = _mkdata_pool(S, N, K, E, offsets, args.dtype, nbuf)

    ref = None if args.no_verify else _reference(d.tok_ref, d.w0_ref, d.w1_ref, offsets, S, N, K, E, variant)

    bl_ms = None
    if not args.no_baseline:
        out_bl = torch.empty_like(out)
        # An mxfp8 pool set holds no widened copy, so the bf16 baseline stays on the verify set.
        bl_sets = pool if args.dtype == "bf16" else [d]
        if args.stream:
            print("  ▶ running unfused per-group cuBLAS baseline ...", flush=True)
        bl_ms = time_ms(
            rotating(lambda s: _unfused_launch(s.tok_ref, s.w0_ref, s.w1_ref, out_bl, offsets, S, N, K, E, variant), bl_sets),
            lambda: _unfused_launch(d.tok_ref, d.w0_ref, d.w1_ref, out_bl, offsets, S, N, K, E, variant),
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
        )
        bl_label = "unfused per-group cuBLAS bf16 + pointwise" + ("" if bl_sets is pool else " [no rotation]")
        print(f"  {bl_label:64s} {flops / (bl_ms * 1e-3) / 1e12:8.2f} TFLOP/s  " f"{bl_ms:8.3f} ms", flush=True)

    spec_map = _build_spec_map(variant, args.dtype)
    labels = select_configs(args.configs, spec_map)
    print(f"  sweeping {len(labels)} configs (each JITs once, ~15-25 s)", flush=True)

    best = None
    for label in labels:
        sel = spec_for(label, spec_map)
        if sel is None:
            print(f"  {label:64s} UNKNOWN (not a sweepable MoE dual-GEMM strategy)", flush=True)
            continue
        cfg, cta_group = sel
        if args.stream:
            print(f"  ▶ running {label} ...", flush=True)
        try:
            g, h = _graph(S, N, K, E, variant, args.dtype, offsets)
            plan = jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group)
        except (NotImplementedError, ValueError) as e:
            print(f"  {label:64s} SKIP: {type(e).__name__}: {str(e)[:40]}", flush=True)
            continue
        aux_bufs = [torch.tensor([[[v]]], device="cuda", dtype=torch.float32) for v in h.aux_values]
        vp = _vp(h, d, fto, out, aux_bufs)
        try:
            plan(vp)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {label:64s} LAUNCH FAIL: {type(e).__name__}: {str(e)[:40]}", flush=True)
            continue
        flag = ""
        ok = True
        if ref is not None:
            got = out.view(S, N).float()
            ok = torch.allclose(got, ref.float(), rtol=args.rtol, atol=args.atol)
            if not ok:
                flag = f"  !! maxerr={(got - ref.float()).abs().max().item():.3g}"
        vps = [_vp(h, s, fto, s.out, aux_bufs) for s in pool]
        ms = time_ms(rotating(plan, vps), lambda _plan=plan, _vp=vp: _plan(_vp), warmup=args.warmup, iters=args.iters, timing=args.timing)
        tflops = flops / (ms * 1e-3) / 1e12
        ratio = f"{bl_ms / ms:>7.2f}x" if bl_ms else " " * 8
        print(f"  {label:64s} {tflops:8.2f} TFLOP/s  {ms:8.3f} ms  {ratio}{flag}", flush=True)
        if ok and (best is None or ms < best[1]):
            best = (label, ms, tflops)

    if best is None:
        print("  no strategy produced a correct result for this model.", flush=True)
        return None
    print(f"  best: {best[0]}  {best[2]:.2f} TFLOP/s  {best[1]:.3f} ms", flush=True)
    return (key, spec["label"], variant, *best)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="all", help=f"comma-separated: all | {' | '.join(MODELS)}")
    p.add_argument("--dtype", choices=DTYPES, default="bf16", help="operand precision: bf16, or mxfp8 (e4m3 + per-32-block e8m0)")
    p.add_argument("-S", "--tokens", type=int, default=None, help="token count (required)")
    p.add_argument("-E", "--experts", type=int, default=None, help="expert count (required)")
    add_sweep_args(p, nsys=False)
    p.add_argument("--no-verify", action="store_true", help="skip the torch reference check")
    p.add_argument("--no-baseline", action="store_true", help="skip the unfused per-group cuBLAS baseline")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--rtol", type=float, default=5e-2)
    p.add_argument("--atol", type=float, default=2e-1)
    args = p.parse_args()

    if args.list_models:
        for k, s in MODELS.items():
            print(f"{k:12s} {s['label']:20s} {s['variant']:16s} N={s['N']:5d} K={s['K']:5d}")
        return 0

    if args.tokens is None or args.experts is None:
        sys.exit("-S/--tokens and -E/--experts are required (a model fixes only N, K and the epilogue)")
    if args.tokens < 1:
        sys.exit(f"--tokens must be >= 1, got {args.tokens}")
    if args.experts < 1:
        sys.exit(f"--experts must be >= 1, got {args.experts}")

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    keys = list(MODELS) if args.models.strip().lower() == "all" else [k.strip() for k in args.models.split(",")]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {', '.join(unknown)} (see --list-models)")

    summary = []
    for k in keys:
        res = _run_model(k, MODELS[k], args)
        if res is not None:
            summary.append(res)
        torch.cuda.empty_cache()

    if len(summary) > 1:
        print("\n=== summary (best config per model) ===")
        for key, label, variant, cfg_label, ms, tflops in summary:
            print(f"  {label:20s} {variant:16s} {tflops:8.2f} TFLOP/s  {ms:8.3f} ms  {cfg_label}")
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
