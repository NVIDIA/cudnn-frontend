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
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import pytest
import torch

from gemm_test_utils import (
    requires_sm100,
    requires_sm107,
    Plan as _plan,
    E2M1 as _E2M1,
    to_blocked as _to_blocked,
    unpack_fp4 as _unpack_fp4,
    rand_e8m0 as _rand_e8m0,
    block_quant_ref as _block_quant_ref,
)

from cudnn.gemm.frost.dtypes import DTYPE_FROM_CUDNN as _DTYPE_FROM_CUDNN
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0


def _vp_moe_bs_mg(compiled, gemm_pairs, fto, outs, *aux):
    """MoE block-scale multi-GEMM variant-pack dict. Each pair is
    ((token, sfa), (weight, sfb)); dedup by packed-data identity into distinct
    A/B slots, + first_token_offset + outputs + aux."""
    bd = compiled.binding
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
# the sm100 pipeline and the sm107 one (same geometry, 64-byte-K MMA).
_GEOMETRIES = [
    ("CONFIG_sm100_128x128x128_128x128x32_cluster2x1", 2),
    ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1", 1),
    pytest.param("CONFIG_sm107_128x128x128_128x128x64_cluster2x1", 2, marks=requires_sm107),
    pytest.param("CONFIG_sm107_128x128x128_128x128x64_cluster1x1", 1, marks=requires_sm107),
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
):
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
        dim=[num_groups, 1, 1],
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


@requires_sm100
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
def test_dual_moe_grouped_block_scale_matmul_fwd_swiglu(combo, cfg_name, cta_group) -> None:
    """Spec case: S=1024, N=256, K=512, E=2, 4 groups (offsets [0,256,384,512])."""
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 1024, 256, 512
    offsets_list = [0, 256, 384, 512]
    num_groups = len(offsets_list)
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
    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1] if gi + 1 < num_groups else S]) for gi in range(num_groups)]
    sfa_blk = torch.cat(sfa_parts).view(1, -1, 1)
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
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        ex = gi % E
        c0 = tok_s[b:e] @ w0_s[ex].T
        c1 = tok_s[b:e] @ w1_s[ex].T
        ref[b:e] = torch.nn.functional.silu(c0) * c1 * 0.5
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=5e-2, rtol=5e-2)


@requires_sm100
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
def test_dual_moe_grouped_block_scale_matmul_fwd_swiglu_quant_epilogue(combo, cfg_name, cta_group) -> None:
    """Terminal block_scale_quantize on the dual-MoE SwiGLU chain: the fused
    result is re-quantized to FP8 E4M3 + per-32-block E8M0 scale (two outputs)."""
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 1024, 256, 512
    offsets_list = [0, 256, 384, 512]
    num_groups = len(offsets_list)
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

    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1] if gi + 1 < num_groups else S]) for gi in range(num_groups)]
    sfa_blk = torch.cat(sfa_parts).view(1, -1, 1)
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
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
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


@requires_sm100
def test_dual_moe_grouped_block_scale_matmul_fwd_swiglu_reduction_scalar() -> None:
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 512, 128, 512
    combo = "nvfp4"
    offsets_list = [0, 100, 300]
    num_groups = len(offsets_list)
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

    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1] if gi + 1 < num_groups else S]) for gi in range(num_groups)]
    sfa_blk = torch.cat(sfa_parts).view(1, -1, 1)
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
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
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
