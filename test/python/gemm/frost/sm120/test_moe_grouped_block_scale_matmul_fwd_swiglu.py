# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused dual MoE grouped block-scale matmul + SwiGLU:
out = silu(moe(tok_d, w0_d)) * moe(tok_d, w1_d) * scale.

Two block-scaled grouped matmuls share the token (A)+SFA and one
first_token_offset; weights (B)+SFB distinct. Multi-GEMM MoE block-scale path,
covering nvfp4 / mxfp4 / mxfp8, checked vs a torch dequant + group-loop reference.
"""

from __future__ import annotations

import cudnn
import pytest
import torch

from gemm_test_utils import (
    requires_sm120,
    Plan as _plan,
    E2M1 as _E2M1,
    to_blocked as _to_blocked,
    unpack_fp4 as _unpack_fp4,
    rand_e8m0 as _rand_e8m0,
    block_quant_ref as _block_quant_ref,
    with_static_segmented_capacity as _with_static_segmented_capacity,
)

from cudnn.gemm.frost.dtypes import DTYPE_FROM_CUDNN as _DTYPE_FROM_CUDNN
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0


def _vp_moe_bs_mg(compiled, gemm_pairs, fto, outs, *aux):
    """MoE block-scale multi-GEMM variant-pack dict. Each pair is
    ((token, sfa), (weight, sfb)); dedup by packed-data identity into distinct
    A/B slots, + first_token_offset + outputs + aux."""
    from gemm_test_utils import graph_binding

    bd = graph_binding(compiled)
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


# cta_tile_n=128: dual block-scale TMEM fits two accs + SF only at n<=128.
# (config, cta_group): 2-CTA cluster2x1 (reference) + 1-CTA cluster1x1, on both
# the sm120 pipeline and the sm120 one (same geometry, 64-byte-K MMA).
_GEOMETRIES = [
    ("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 2),
    ("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 1),
    ("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps8x1", 2),
    ("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps8x1", 1),
    pytest.param("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4", 2, marks=requires_sm120),
    pytest.param("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2", 1, marks=requires_sm120),
]


# combo -> (block_size, data dtype, SF dtype).
_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


def _build_graph(
    E,
    S,
    N,
    K,
    num_groups,
    combo,
    offset_dt=cudnn.data_type.INT32,
    reduction_mode=None,
    reduction_dims=None,
    quant_block=None,
    weight_major="k",
):
    block_size, a_dt, sf_dt = _COMBOS[combo]
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
    w_stride = [K * N, 1, K] if weight_major == "k" else [K * N, N, 1]
    w0 = g.tensor(name="weight0", dim=[E, K, N], stride=w_stride, data_type=a_dt)
    w1 = g.tensor(name="weight1", dim=[E, K, N], stride=w_stride, data_type=a_dt)
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
        dim=[num_groups + 1, 1, 1],
        stride=[1, 1, 1],
        data_type=offset_dt,
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
    c0silu = g.swish(input=c0, name="silu0")
    mul = g.mul(a=c0silu, b=c1, name="mul0")
    dq = g.mul(a=mul, b=sf, name="dequant0")
    if quant_block is not None:
        Q, QS = g.block_scale_quantize(input=dq, block_size=quant_block, name="q")
        Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
        QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    else:
        dq.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    if reduction_mode is not None:
        assert reduction_dims is not None
        R = g.reduction(input=dq, mode=reduction_mode, name="red")
        R.set_dim(list(reduction_dims)).set_stride([reduction_dims[1] * reduction_dims[2], reduction_dims[2], 1])
        R.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return g


# --------------------------------------------------------------------------- #
# Analyzer (no GPU)
# --------------------------------------------------------------------------- #


def test_analyzer_detects_dual_moe_grouped_block_scale_matmul_fwd() -> None:
    chain = analyze(_build_graph(2, 1024, 256, 512, 4, "nvfp4"))
    assert chain.has_moe and chain.has_block_scale and chain.is_multi_gemm
    assert chain.num_gemms == 2
    assert chain.num_a_operands == 1 and chain.num_b_operands == 2
    assert (chain.block_scale.sf_dtype, chain.block_scale.block_size) == ("fp8_e4m3", 16)
    assert chain.moe.num_experts == 2
    assert [o.op for o in chain.ops] == ["swish", "mul", "mul"]
    assert len(chain.outputs) == 1 and chain.outputs[0].source == "op_2"


def test_one_packed_token_carries_one_dequant() -> None:
    """The MoE builder dedups operands by the PACKED-DATA tensor exactly like the
    dense one, so two dequantizes of one token tensor would collapse onto the
    first and run both grouped GEMMs with its SFA. Rejected, not silently dropped
    (the second SFA would not even be a graph input for the variant pack)."""
    E, S, N, K, bs = 2, 1024, 256, 512, 16
    sf_k = K // bs
    rk = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)
    a_dt, sf_dt = cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
    fto = g.tensor(name="fto", dim=[E + 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    sfa = [g.tensor(name=f"SFA{i}", dim=[1, S, sf_k], stride=[S * sf_k, sf_k, 1], data_type=sf_dt, **rk) for i in range(2)]
    outs = []
    for i in range(2):
        w = g.tensor(name=f"w{i}", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
        sfb = g.tensor(name=f"SFB{i}", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, **rk)
        outs.append(
            g.moe_grouped_matmul(
                g.block_scale_dequantize(input=tok, descale=sfa[i], block_size=[1, bs]),
                g.block_scale_dequantize(input=w, descale=sfb, block_size=[bs, 1]),
                fto,
                mode=cudnn.moe_grouped_matmul_mode.NONE,
                compute_data_type=cudnn.data_type.FLOAT,
                name=f"moe{i}",
            )
        )
    Y = g.mul(a=g.swish(input=outs[0]), b=outs[1])
    Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    with pytest.raises(ValueError, match="A operand .* dequantized more than once"):
        analyze(g)


def test_analyzer_detects_dual_moe_grouped_block_scale_matmul_fwd_quant_epilogue() -> None:
    chain = analyze(_build_graph(2, 1024, 256, 512, 4, "nvfp4", quant_block=32))
    assert chain.has_moe and chain.has_block_scale and chain.is_multi_gemm
    assert len(chain.quants) == 1
    assert chain.quants[0].block_size == 32
    assert chain.quants[0].scale_dtype == "fp8_e8m0"
    assert chain.quants[0].source_ref == 2  # the dequant0 mul feeds the quant
    assert chain.output_dtype == "fp8_e4m3"
    assert [o.op for o in chain.ops] == ["swish", "mul", "mul"]


def test_analyzer_detects_dual_moe_grouped_block_scale_matmul_fwd_reduction() -> None:
    chain = analyze(
        _build_graph(
            2,
            1024,
            256,
            512,
            4,
            "nvfp4",
            reduction_mode=cudnn.reduction_mode.ADD,
            reduction_dims=(1, 1, 1),
        )
    )
    assert chain.has_moe and chain.has_block_scale and chain.is_multi_gemm
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "add"
    assert [o.source for o in chain.outputs] == ["op_2", "reduction_0"]


# --------------------------------------------------------------------------- #
# End-to-end correctness (GPU)
# --------------------------------------------------------------------------- #


def _mk_operand(combo, batch, rows, K, dev, lut):
    """Return (runtime packed tensor, dequantized float (.., rows, K))."""
    is_fp4 = combo in ("nvfp4", "mxfp4")
    if is_fp4:
        u8 = torch.randint(0, 256, (batch, rows, K // 2), dtype=torch.uint8, device=dev)
        return u8.view(torch.float4_e2m1fn_x2), _unpack_fp4(u8, lut).view(batch, rows, K)
    rt = (torch.randn(batch, rows, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    return rt, rt.float().view(batch, rows, K)


def _mk_sf(combo, shape, dev):
    if combo == "nvfp4":
        return torch.randint(1, 4, shape, device=dev).to(torch.float8_e4m3fn)
    return _rand_e8m0(shape, dev)


@requires_sm120
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
def test_dual_moe_grouped_block_scale_matmul_fwd_swiglu(combo, cfg_name, cta_group) -> None:
    """Spec case: S=1024, N=256, K=512, E=2, 4 groups (offsets [0,256,384,512,1024])."""
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 1024, 256, 512
    offsets_list = [0, 256, 384, 512, S]
    num_groups = len(offsets_list) - 1
    block_size = _COMBOS[combo][0]
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    tok_rt, tok_deq = _mk_operand(combo, 1, S, K, dev, lut)
    w0_rt, w0_deq = _mk_operand(combo, E, N, K, dev, lut)
    w1_rt, w1_deq = _mk_operand(combo, E, N, K, dev, lut)
    tok_deq = tok_deq.view(S, K)
    sfa_log = _mk_sf(combo, (S, sf_k), dev)
    sfb0_log = _mk_sf(combo, (E, N, sf_k), dev)
    sfb1_log = _mk_sf(combo, (E, N, sf_k), dev)
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)

    cfg = by_name(cfg_name)
    compiled = _plan(_build_graph(E, S, N, K, num_groups, combo), config=cfg, cta_group=cta_group)
    _blk, _, _sf_dt = _COMBOS[combo]
    _bs = compiled.chain.block_scale
    assert (_bs.sf_dtype, _bs.block_size) == (_DTYPE_FROM_CUDNN[_sf_dt], _blk)

    # SFA padded to 128 rows PER GROUP, then concatenated; SFB per-expert.
    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1]]) for gi in range(num_groups)]
    sfa_blk = _with_static_segmented_capacity(torch.cat(sfa_parts), S, num_groups, sf_k)
    sfb0_blk = torch.cat([_to_blocked(sfb0_log[e]) for e in range(E)]).view(E, sf_k, N)
    sfb1_blk = torch.cat([_to_blocked(sfb1_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)

    compiled(
        _vp_moe_bs_mg(
            compiled,
            [
                ((tok_rt, sfa_blk), (w0_rt, sfb0_blk)),
                ((tok_rt, sfa_blk), (w1_rt, sfb1_blk)),
            ],
            offsets,
            output,
            scale,
        )
    )
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w0_s = w0_deq * sfb0_log.float().repeat_interleave(block_size, 2)
    w1_s = w1_deq * sfb1_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1]
        if b == e:
            continue
        ex = gi % E
        c0 = tok_s[b:e] @ w0_s[ex].T
        c1 = tok_s[b:e] @ w1_s[ex].T
        ref[b:e] = torch.nn.functional.silu(c0) * c1 * 0.5
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=5e-2, rtol=5e-2)


@requires_sm120
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
def test_dual_moe_grouped_block_scale_matmul_fwd_swiglu_quant_epilogue(combo, cfg_name, cta_group) -> None:
    """Terminal block_scale_quantize on the dual-MoE SwiGLU chain: the fused
    result is re-quantized to FP8 E4M3 + per-32-block E8M0 scale (two outputs)."""
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 1024, 256, 512
    offsets_list = [0, 256, 384, 512, S]
    num_groups = len(offsets_list) - 1
    block_size = _COMBOS[combo][0]
    qblock = 32
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    tok_rt, tok_deq = _mk_operand(combo, 1, S, K, dev, lut)
    w0_rt, w0_deq = _mk_operand(combo, E, N, K, dev, lut)
    w1_rt, w1_deq = _mk_operand(combo, E, N, K, dev, lut)
    tok_deq = tok_deq.view(S, K)
    sfa_log = _mk_sf(combo, (S, sf_k), dev)
    sfb0_log = _mk_sf(combo, (E, N, sf_k), dev)
    sfb1_log = _mk_sf(combo, (E, N, sf_k), dev)
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)

    cfg = by_name(cfg_name)
    compiled = _plan(
        _build_graph(E, S, N, K, num_groups, combo, quant_block=qblock),
        config=cfg,
        cta_group=cta_group,
    )
    assert compiled.chain.quants

    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1]]) for gi in range(num_groups)]
    sfa_blk = _with_static_segmented_capacity(torch.cat(sfa_parts), S, num_groups, sf_k)
    sfb0_blk = torch.cat([_to_blocked(sfb0_log[e]) for e in range(E)]).view(E, sf_k, N)
    sfb1_blk = torch.cat([_to_blocked(sfb1_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    q = torch.zeros(1, S, N, dtype=torch.float8_e4m3fn, device=dev)
    q_scale = torch.zeros(1, S, N // qblock, dtype=torch.float8_e8m0fnu, device=dev)

    compiled(
        _vp_moe_bs_mg(
            compiled,
            [
                ((tok_rt, sfa_blk), (w0_rt, sfb0_blk)),
                ((tok_rt, sfa_blk), (w1_rt, sfb1_blk)),
            ],
            offsets,
            [q, q_scale],
            scale,
        )
    )
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w0_s = w0_deq * sfb0_log.float().repeat_interleave(block_size, 2)
    w1_s = w1_deq * sfb1_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1]
        if b == e:
            continue
        ex = gi % E
        c0 = tok_s[b:e] @ w0_s[ex].T
        c1 = tok_s[b:e] @ w1_s[ex].T
        ref[b:e] = torch.nn.functional.silu(c0) * c1 * 0.5
    q_ref, scale_ref = _block_quant_ref(ref, qblock, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    # The kernel's swish uses fast __expf → pre-quant values sit within ~1e-3
    # rel of torch; allow one E4M3 mantissa step where that crosses a boundary.
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=2**-8, rtol=1 / 8)


@requires_sm120
def test_dual_moe_grouped_block_scale_matmul_fwd_swiglu_reduction_scalar() -> None:
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 512, 128, 512
    combo = "nvfp4"
    offsets_list = [0, 100, 300, S, S]
    num_groups = len(offsets_list) - 1
    block_size = _COMBOS[combo][0]
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    tok_rt, tok_deq = _mk_operand(combo, 1, S, K, dev, lut)
    w0_rt, w0_deq = _mk_operand(combo, E, N, K, dev, lut)
    w1_rt, w1_deq = _mk_operand(combo, E, N, K, dev, lut)
    tok_deq = tok_deq.view(S, K)
    sfa_log = _mk_sf(combo, (S, sf_k), dev)
    sfb0_log = _mk_sf(combo, (E, N, sf_k), dev)
    sfb1_log = _mk_sf(combo, (E, N, sf_k), dev)
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)

    cfg = by_name(_GEOMETRIES[1][0])
    compiled = _plan(
        _build_graph(
            E,
            S,
            N,
            K,
            num_groups,
            combo,
            reduction_mode=cudnn.reduction_mode.ADD,
            reduction_dims=(1, 1, 1),
        ),
        config=cfg,
        cta_group=_GEOMETRIES[1][1],
    )

    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1]]) for gi in range(num_groups)]
    sfa_blk = _with_static_segmented_capacity(torch.cat(sfa_parts), S, num_groups, sf_k)
    sfb0_blk = torch.cat([_to_blocked(sfb0_log[e]) for e in range(E)]).view(E, sf_k, N)
    sfb1_blk = torch.cat([_to_blocked(sfb1_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    red = torch.empty(1, 1, 1, dtype=torch.float32, device=dev)

    compiled(
        _vp_moe_bs_mg(
            compiled,
            [
                ((tok_rt, sfa_blk), (w0_rt, sfb0_blk)),
                ((tok_rt, sfa_blk), (w1_rt, sfb1_blk)),
            ],
            offsets,
            [output, red],
            scale,
        )
    )
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w0_s = w0_deq * sfb0_log.float().repeat_interleave(block_size, 2)
    w1_s = w1_deq * sfb1_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1]
        if b == e:
            continue
        ex = gi % E
        c0 = tok_s[b:e] @ w0_s[ex].T
        c1 = tok_s[b:e] @ w1_s[ex].T
        ref[b:e] = torch.nn.functional.silu(c0) * c1 * 0.5
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        red,
        ref.view(1, S, N).sum(dim=(1, 2), keepdim=True),
        atol=1e-1,
        rtol=1e-2,
    )


def _build_two_token_gather_graph(E, S, N, K, num_groups, rows, combo):
    """silu(moe(tok0, w0)) * moe(tok1, w1) * scale over GATHERed rows: two
    distinct tokens (each with its own linear SFA) and two weights, so the
    kernel holds 2 A + 2 B packed tiles and four SF rings per stage."""
    block_size, a_dt, sf_dt = _COMBOS[combo]
    sf_k = K // block_size
    pitch = -(-sf_k // 16) * 16
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    fto = g.tensor(name="first_token_offset", dim=[num_groups + 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    sf = g.tensor(name="scaleFactor", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    token_index = g.tensor(name="token_index", dim=[1, rows, 1], stride=[rows, 1, 1], data_type=cudnn.data_type.INT32)
    outs = []
    for i in range(2):
        tok = g.tensor(name=f"token{i}", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
        sfa = g.tensor(
            name=f"SFA{i}",
            dim=[1, S, sf_k],
            stride=[S * pitch, pitch, 1],
            data_type=sf_dt,
            reordering_type=cudnn.tensor_reordering.NONE,
        )
        w = g.tensor(name=f"weight{i}", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
        sfb = g.tensor(
            name=f"SFB{i}",
            dim=[E, sf_k, N],
            stride=[sf_k * N, 1, sf_k],
            data_type=sf_dt,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        outs.append(
            g.moe_grouped_matmul(
                g.block_scale_dequantize(input=tok, descale=sfa, block_size=[1, block_size]),
                g.block_scale_dequantize(input=w, descale=sfb, block_size=[block_size, 1]),
                fto,
                mode=cudnn.moe_grouped_matmul_mode.GATHER,
                compute_data_type=cudnn.data_type.FLOAT,
                name=f"moe{i}",
                token_index=token_index,
            )
        )
    y = g.mul(a=g.mul(a=g.swish(input=outs[0], name="silu0"), b=outs[1], name="mul0"), b=sf, name="dequant0")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    return g


@requires_sm120
@pytest.mark.parametrize(
    "cfg_name,swap_ab",
    [
        ("CONFIG_sm120_32x64x128_16x16x32_cluster1x1_warps2x4", False),
        # The same operands on the swap-AB template (weights are A, tokens B).
        ("CONFIG_sm120_64x32x128_16x16x32_cluster1x1_warps4x2", True),
    ],
)
def test_dual_moe_grouped_block_scale_two_token_gather_fits_smem(cfg_name, swap_ab) -> None:
    """2 A + 2 B mxfp8 operands, GATHER, on a 32-row token tile: three stages
    of four 512-B SF rings. With the rings 1024-B aligned each one padded 512 B
    and the CTA asked for 101904 B of SMEM on a 101376-B part, so the launch
    failed although every gate accepted the plan."""
    from dataclasses import replace

    dev = "cuda"
    torch.manual_seed(1)
    combo = "mxfp8"
    E, S, N, K, rows = 2, 137, 128, 512, 259
    starts = [0, 3, 3, 131, 258, rows, rows]
    num_groups = len(starts) - 1
    block_size, _, _ = _COMBOS[combo]
    sf_k = K // block_size
    pitch = -(-sf_k // 16) * 16
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    compiled = _plan(
        _build_two_token_gather_graph(E, S, N, K, num_groups, rows, combo),
        config=replace(by_name(cfg_name), swap_ab=swap_ab),
    )
    assert (compiled.chain.num_a_operands, compiled.chain.num_b_operands) == (2, 2)
    toks = [_mk_operand(combo, 1, S, K, dev, lut) for _ in range(2)]
    ws = [_mk_operand(combo, E, N, K, dev, lut) for _ in range(2)]
    sfa_logs = [_mk_sf(combo, (S, sf_k), dev) for _ in range(2)]
    sfb_logs = [_mk_sf(combo, (E, N, sf_k), dev) for _ in range(2)]
    sfa_bufs = []
    for s in sfa_logs:
        buf = torch.zeros((1, S, pitch), dtype=torch.uint8, device=dev)[:, :, :sf_k].view(s.dtype)
        buf.copy_(s.view(1, S, sf_k))
        sfa_bufs.append(buf)
    sfb_bufs = [torch.cat([_to_blocked(s[e]) for e in range(E)]).view(E, sf_k, N) for s in sfb_logs]
    index = (torch.arange(rows, device=dev, dtype=torch.int32) * 37 % S).view(1, rows, 1)
    offsets = torch.tensor(starts, dtype=torch.int32, device=dev)
    output = torch.zeros(1, rows, N, dtype=torch.bfloat16, device=dev)
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)

    from gemm_test_utils import graph_binding

    vp = _vp_moe_bs_mg(
        compiled,
        [((toks[i][0], sfa_bufs[i]), (ws[i][0], sfb_bufs[i])) for i in range(2)],
        offsets,
        output,
        scale,
    )
    vp[graph_binding(compiled).token_index] = index
    compiled(vp)
    torch.cuda.synchronize()

    a = [toks[i][1].view(S, K) * sfa_logs[i].float().repeat_interleave(block_size, 1) for i in range(2)]
    b = [ws[i][1] * sfb_logs[i].float().repeat_interleave(block_size, 2) for i in range(2)]
    ref = torch.zeros((rows, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        lo, hi = starts[gi], starts[gi + 1]
        if lo == hi:
            continue
        sel = index.flatten()[lo:hi].long()
        ref[lo:hi] = torch.nn.functional.silu(a[0][sel] @ b[0][gi % E].T) * (a[1][sel] @ b[1][gi % E].T) * 0.5
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=5e-2, rtol=5e-2)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
def test_dual_moe_grouped_block_scale_matmul_fwd_swiglu_n_major_weights(swap_ab) -> None:
    """mxfp8 SwiGLU with two N-major weights: each weight is read through the
    transposing ldmatrix.trans.b8 path from its own SMEM ring (two N-major B
    operands on the token-by-weight kernel, two M-major A operands after the
    swap), over ragged groups with an empty one."""
    from dataclasses import replace

    dev = "cuda"
    torch.manual_seed(0)
    combo = "mxfp8"
    E, S, N, K = 2, 1024, 256, 512
    offsets_list = [0, 1, 1, 304, 401, 513, S]
    num_groups = len(offsets_list) - 1
    block_size = _COMBOS[combo][0]
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    tok_rt, tok_deq = _mk_operand(combo, 1, S, K, dev, lut)
    w0_rt, w0_deq = _mk_operand(combo, E, N, K, dev, lut)
    w1_rt, w1_deq = _mk_operand(combo, E, N, K, dev, lut)
    # The (E, N, K) logical weights stored N-contiguous, as the graph declares.
    w0_rt = w0_rt.transpose(1, 2).contiguous().transpose(1, 2)
    w1_rt = w1_rt.transpose(1, 2).contiguous().transpose(1, 2)
    tok_deq = tok_deq.view(S, K)
    sfa_log = _mk_sf(combo, (S, sf_k), dev)
    sfb0_log = _mk_sf(combo, (E, N, sf_k), dev)
    sfb1_log = _mk_sf(combo, (E, N, sf_k), dev)
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)

    compiled = _plan(
        _build_graph(E, S, N, K, num_groups, combo, weight_major="n"),
        config=replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab),
    )

    # torch.cat has no CUDA E8M0 kernel once an (empty) group's slice is in the
    # list: join the scale blocks as bytes.
    def _cat(parts):
        return torch.cat([p.view(torch.uint8) for p in parts]).view(parts[0].dtype)

    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1]]) for gi in range(num_groups)]
    sfa_blk = _with_static_segmented_capacity(_cat(sfa_parts), S, num_groups, sf_k)
    sfb0_blk = _cat([_to_blocked(sfb0_log[e]) for e in range(E)]).view(E, sf_k, N)
    sfb1_blk = _cat([_to_blocked(sfb1_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)

    compiled(
        _vp_moe_bs_mg(
            compiled,
            [
                ((tok_rt, sfa_blk), (w0_rt, sfb0_blk)),
                ((tok_rt, sfa_blk), (w1_rt, sfb1_blk)),
            ],
            offsets,
            output,
            scale,
        )
    )
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w0_s = w0_deq * sfb0_log.float().repeat_interleave(block_size, 2)
    w1_s = w1_deq * sfb1_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1]
        if b == e:
            continue
        ex = gi % E
        ref[b:e] = torch.nn.functional.silu(tok_s[b:e] @ w0_s[ex].T) * (tok_s[b:e] @ w1_s[ex].T) * 0.5
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=5e-2, rtol=5e-2)


@requires_sm120
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
def test_dual_moe_block_scale_auto_pick_keeps_two_ab_stages(combo) -> None:
    """The automatic dual block-scale pick keeps a two-stage AB ring:
    select_config's 128x128 holds one stage of the token, both weights and their
    scale-factor boxes next to the epilogue staging, which serializes every K
    tile's load and MMA (the 128x64 re-cut ran up to 1.41x faster)."""
    from cudnn.gemm.frost import compiler as C
    from cudnn.gemm.frost.kernel_registry import select_template

    chain = analyze(_build_graph(8, 8192, 2048, 4096, 8, combo))
    cfg = C.plan_config(chain)
    assert cfg.pipeline == "sm120" and cfg.cta_tile_m == 128, cfg.name
    assert select_template(chain, cfg).multi_gemm_ab_stages(chain, cfg)[0] >= 2, cfg.name


@requires_sm120
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
def test_triple_moe_block_scale_auto_pick(combo) -> None:
    """Three parallel block-scaled GEMMs cap the N tile at 64. select_config used
    to apply the tcgen05 rule (whole 128-wide SF blocks) and raise, so the graph
    had no automatic plan on sm120, whose warp MMA takes any CTA tile that
    divides the 128-row block. The pick now keeps a two-stage ring and computes
    silu(A @ B0) * (A @ B1) * scale + A @ B2 over ragged groups with an empty one."""
    from cudnn.gemm.frost import compiler as C
    from cudnn.gemm.frost.kernel_registry import select_template

    dt = cudnn.data_type
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 1024, 256, 512
    offsets_list = [0, 1, 1, 304, 401, 513, S]
    num_groups = len(offsets_list) - 1
    block_size, a_dt, sf_dt = _COMBOS[combo]
    sf_k = K // block_size

    g = cudnn.pygraph(io_data_type=dt.BFLOAT16, intermediate_data_type=dt.FLOAT, compute_data_type=dt.FLOAT)
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
    sfa = g.tensor(name="SFA", dim=[1, S, sf_k], stride=[S * sf_k, sf_k, 1], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
    fto = g.tensor(name="first_token_offset", dim=[num_groups + 1, 1, 1], stride=[1, 1, 1], data_type=dt.INT32)
    sf = g.tensor(name="scaleFactor", dim=[1, 1, 1], stride=[1, 1, 1], data_type=dt.FLOAT)
    tok_d = g.block_scale_dequantize(input=tok, descale=sfa, block_size=[1, block_size])
    cs = []
    for i in range(3):
        w = g.tensor(name=f"weight{i}", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
        sfb = g.tensor(name=f"SFB{i}", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
        w_d = g.block_scale_dequantize(input=w, descale=sfb, block_size=[block_size, 1])
        cs.append(g.moe_grouped_matmul(tok_d, w_d, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=dt.FLOAT))
    y = g.add(a=g.mul(a=g.mul(a=g.swish(input=cs[0]), b=cs[1]), b=sf), b=cs[2])
    y.set_output(True).set_data_type(dt.BFLOAT16)

    chain = analyze(g)
    assert chain.num_gemms == 3
    cfg = C.plan_config(chain)
    assert cfg.pipeline == "sm120" and cfg.cta_tile_n < 128, cfg.name
    assert select_template(chain, cfg).multi_gemm_ab_stages(chain, cfg)[0] >= 2, cfg.name
    compiled = _plan(g, config=cfg)

    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    tok_rt, tok_deq = _mk_operand(combo, 1, S, K, dev, lut)
    tok_deq = tok_deq.view(S, K)
    ws = [_mk_operand(combo, E, N, K, dev, lut) for _ in range(3)]
    sfa_log = _mk_sf(combo, (S, sf_k), dev)
    sfb_logs = [_mk_sf(combo, (E, N, sf_k), dev) for _ in range(3)]
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)

    # torch.cat has no CUDA E8M0 kernel once an (empty) group's slice is in the
    # list: join the scale blocks as bytes.
    def _cat(parts):
        return torch.cat([p.view(torch.uint8) for p in parts]).view(parts[0].dtype)

    sfa_blk = _with_static_segmented_capacity(
        _cat([_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1]]) for gi in range(num_groups)]), S, num_groups, sf_k
    )
    sfb_blks = [_cat([_to_blocked(s[e]) for e in range(E)]).view(E, sf_k, N) for s in sfb_logs]
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)
    compiled(_vp_moe_bs_mg(compiled, [((tok_rt, sfa_blk), (w_rt, sfb)) for (w_rt, _), sfb in zip(ws, sfb_blks)], offsets, output, scale))
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w_s = [w_deq * s.float().repeat_interleave(block_size, 2) for (_, w_deq), s in zip(ws, sfb_logs)]
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b, e = offsets_list[gi], offsets_list[gi + 1]
        if b == e:
            continue
        a, ex = tok_s[b:e], gi % E
        ref[b:e] = torch.nn.functional.silu(a @ w_s[0][ex].T) * (a @ w_s[1][ex].T) * 0.5 + a @ w_s[2][ex].T
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=5e-2, rtol=5e-2)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
def test_dual_moe_block_scale_launch_resets_its_scheduler_counter_sm120(swap_ab) -> None:
    """The block-scale twin of the dense dual-MoE counter test: the compiled host
    zeroes the tile-scheduler counter itself, stream-ordered (the counter-init kernel
    in front of the PDL main kernel), on the multi-GEMM launch path as on the single
    one. A reused workspace whose counter slot holds a stale claim (every byte 0x7f)
    would otherwise end every CTA's first claim past the tile space and leave the
    output unwritten."""
    dev = "cuda"
    torch.manual_seed(0)
    E, N, K, combo = 8, 256, 128, "nvfp4"
    group_sizes = [64, 0, 200, 128, 100, 12, 196, 68]
    S = sum(group_sizes)
    offsets_list = [sum(group_sizes[:g]) for g in range(E + 1)]
    block_size = _COMBOS[combo][0]
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    tok_rt, tok_deq = _mk_operand(combo, 1, S, K, dev, lut)
    tok_deq = tok_deq.view(S, K)
    (w0_rt, w0_deq), (w1_rt, w1_deq) = _mk_operand(combo, E, N, K, dev, lut), _mk_operand(combo, E, N, K, dev, lut)
    sfa_log, sfb0_log, sfb1_log = _mk_sf(combo, (S, sf_k), dev), _mk_sf(combo, (E, N, sf_k), dev), _mk_sf(combo, (E, N, sf_k), dev)
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)
    try:
        compiled = _plan(
            _build_graph(E, S, N, K, E, combo), config=by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), cta_group=1, swap_ab=swap_ab
        )
    except NotImplementedError as exc:  # the frost conftest would report an arch-tree decline as a skip
        pytest.fail(f"the sm120 tree declined a dual block-scale MoE plan it serves: {exc}")
    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1]]).view(torch.uint8) for gi in range(E)]
    sfa_blk = _with_static_segmented_capacity(torch.cat(sfa_parts).view(sfa_log.dtype), S, E, sf_k)
    sfb0_blk = torch.cat([_to_blocked(sfb0_log[e]) for e in range(E)]).view(E, sf_k, N)
    sfb1_blk = torch.cat([_to_blocked(sfb1_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.empty(1, S, N, dtype=torch.bfloat16, device=dev)
    vp = _vp_moe_bs_mg(compiled, [((tok_rt, sfa_blk), (w0_rt, sfb0_blk)), ((tok_rt, sfa_blk), (w1_rt, sfb1_blk))], offsets, output, scale)
    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w0_s = w0_deq * sfb0_log.float().repeat_interleave(block_size, 2)
    w1_s = w1_deq * sfb1_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(E):
        b, e = offsets_list[gi], offsets_list[gi + 1]
        if b < e:
            ref[b:e] = torch.nn.functional.silu(tok_s[b:e] @ w0_s[gi].T) * (tok_s[b:e] @ w1_s[gi].T) * 0.5
    # Every byte 0x7f: the counter reads as a claim far beyond the tile space.
    workspace = torch.full((compiled.workspace_bytes,), 0x7F, dtype=torch.uint8, device=dev)
    for _ in range(2):  # the second launch finds the first one's final count
        output.fill_(float("nan"))
        compiled(vp, workspace=workspace)
        torch.cuda.synchronize()
        torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=5e-2, rtol=5e-2)
