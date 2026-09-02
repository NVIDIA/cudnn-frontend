# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE grouped block-scale matmul forward (NVFP4, mode=NONE): analyzer detection
(dequant + moe folded → both `moe` and `block_scale` set) + end-to-end vs a torch
dequant + group-loop reference. Covers the BxE > E case."""

from __future__ import annotations

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import pytest
import torch

from gemm_test_utils import (
    requires_sm100,
    requires_sm107,
    Plan as _plan,
    vp_bs as _vp_bs,
    E2M1 as _E2M1,
    ceil_div as _ceil_div,
    to_blocked as _to_blocked,
    unpack_fp4 as _unpack_fp4,
    rand_e8m0 as _rand_e8m0,
    block_quant_ref as _block_quant_ref,
    reduction_ref as _reduction_ref,
    reduction_dims as _reduction_dims,
    assert_block_scale_reduction_close as _assert_block_scale_reduction_close,
)

from cudnn.gemm.frost.dtypes import DTYPE_FROM_CUDNN as _DTYPE_FROM_CUDNN
from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0


_CFG = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1"
_CFG_1CTA = "CONFIG_sm100_128x256x128_128x256x32_cluster1x1"


def _block_quant_q_atol(scale_dtype) -> float:
    # Non-pow2 E4M3 scales use the kernel's approximate reciprocal → up to one
    # smallest E4M3 output step off the torch reference.
    return 1.0 / 512.0 if scale_dtype is torch.float8_e4m3fn else 0.0


# combo -> (block_size, data dtype, SF dtype).
_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}

_QUANT_CASES = [
    (
        "nvfp4_2cta_e4m3_out_e8m0_scale",
        "nvfp4",
        _CFG,
        2,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        512,
        256,
        [0, 100, 300],
    ),
    (
        "nvfp4_1cta_e4m3_out_e8m0_scale",
        "nvfp4",
        _CFG_1CTA,
        1,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        512,
        256,
        [0, 100, 300],
    ),
    (
        "mxfp8_1cta_e4m3_out_e8m0_scale",
        "mxfp8",
        _CFG_1CTA,
        1,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        512,
        256,
        [0, 100, 300],
    ),
    (
        "nvfp4_1cta_e5m2_out_e8m0_scale",
        "nvfp4",
        _CFG_1CTA,
        1,
        cudnn.data_type.FP8_E5M2,
        torch.float8_e5m2,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        512,
        256,
        [0, 100, 300],
    ),
    (
        "nvfp4_1cta_e4m3_out_e4m3_scale",
        "nvfp4",
        _CFG_1CTA,
        1,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        False,
        512,
        256,
        [0, 100, 300],
    ),
    (
        "nvfp4_1cta_e4m3_out_e8m0_scale_f8_128x4",
        "nvfp4",
        _CFG_1CTA,
        1,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        True,
        300,
        256,
        [0, 100, 220],
    ),
]


def _quant_scale_shape(S: int, N: int, reorder: bool) -> tuple[int, int, int]:
    if reorder:
        return (1, _ceil_div(S, 128) * 128, _ceil_div(N // 32, 4) * 4)
    return (1, S, N // 32)


def _build_graph(
    E,
    S,
    N,
    K,
    num_groups,
    combo="nvfp4",
    offset_dt=cudnn.data_type.INT32,
    quant=False,
    output_dt=cudnn.data_type.BFLOAT16,
    reduction_mode=None,
    reduction_dims=None,
    reduction_stride=None,
    reduction_dt=cudnn.data_type.FLOAT,
    reduction_compute_dt=None,
    quant_out_dt=cudnn.data_type.FP8_E4M3,
    quant_scale_dt=cudnn.data_type.FP8_E8M0,
    quant_scale_reorder=False,
    quant_scale_dim=None,
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
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K] if weight_major == "k" else [K * N, N, 1], data_type=a_dt)
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
        dim=[num_groups, 1, 1],
        stride=[1, 1, 1],
        data_type=offset_dt,
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
    if reduction_mode is not None:
        red_kwargs = {}
        if reduction_compute_dt is not None:
            red_kwargs["compute_data_type"] = reduction_compute_dt
        R = g.reduction(input=out, mode=reduction_mode, name="red", **red_kwargs)
        assert reduction_dims is not None
        stride = reduction_stride
        if stride is None:
            stride = (
                reduction_dims[1] * reduction_dims[2],
                reduction_dims[2],
                1,
            )
        R.set_dim(list(reduction_dims)).set_stride(list(stride))
        R.set_output(True).set_data_type(reduction_dt)
    if quant:
        q, q_scale = g.block_scale_quantize(input=out, block_size=32, name="q")
        q.set_data_type(quant_out_dt).set_output(True)
        if quant_scale_dim is not None:
            q_scale.set_dim(list(quant_scale_dim)).set_stride([quant_scale_dim[1] * quant_scale_dim[2], quant_scale_dim[2], 1])
        q_scale.set_data_type(quant_scale_dt).set_output(True)
        if quant_scale_reorder:
            q_scale.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
        return g
    out.set_data_type(output_dt).set_output(True)
    return g


# --------------------------------------------------------------------------- #
# Analyzer (no GPU needed)
# --------------------------------------------------------------------------- #


def test_analyzer_detects_moe_grouped_block_scale_matmul_fwd() -> None:
    E, S, N, K = 2, 1024, 256, 512
    chain = analyze(_build_graph(E, S, N, K, num_groups=4))
    assert chain.has_moe and chain.has_block_scale
    assert chain.moe.num_experts == E
    assert chain.moe.mode == "none"
    assert (chain.block_scale.sf_dtype, chain.block_scale.block_size) == ("fp8_e4m3", 16)
    assert chain.matmul.a_dtype == "fp4_e2m1"
    assert chain.matmul.b_dtype == "fp4_e2m1"
    assert (chain.matmul.M, chain.matmul.N, chain.matmul.K) == (S, N, K)
    assert chain.output_dtype == "bf16"


def test_analyzer_offset_dtype_int64() -> None:
    chain = analyze(_build_graph(2, 1024, 256, 512, num_groups=4, offset_dt=cudnn.data_type.INT64))
    assert chain.moe.offset_dtype == "int64"
    assert chain.has_moe and chain.has_block_scale


def test_analyzer_detects_moe_grouped_block_scale_matmul_fwd_reduction() -> None:
    chain = analyze(
        _build_graph(
            2,
            1024,
            256,
            512,
            num_groups=4,
            reduction_mode=cudnn.reduction_mode.ADD,
            reduction_dims=(1, 1, 1),
        )
    )
    assert chain.has_moe and chain.has_block_scale
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "add"
    assert [o.source for o in chain.outputs] == ["matmul", "reduction_0"]


# --------------------------------------------------------------------------- #
# End-to-end (GPU)
# --------------------------------------------------------------------------- #


def _run_e2e(
    E,
    S,
    N,
    K,
    offsets_list,
    combo="nvfp4",
    offset_dt=cudnn.data_type.INT32,
    offset_torch_dt=torch.int32,
    config_name=_CFG,
    cta_group=2,
    quant=False,
    reduction_mode=None,
    reduction_dims=None,
    reduction_stride=None,
    reduction_dt=cudnn.data_type.FLOAT,
    reduction_torch_dt=torch.float32,
    reduction_compute_dt=None,
    quant_out_dt=cudnn.data_type.FP8_E4M3,
    quant_out_torch_dt=torch.float8_e4m3fn,
    quant_scale_dt=cudnn.data_type.FP8_E8M0,
    quant_scale_torch_dt=torch.float8_e8m0fnu,
    quant_scale_reorder=False,
    weight_major="k",
):
    dev = "cuda"
    torch.manual_seed(0)
    block_size = _COMBOS[combo][0]
    is_fp4 = combo in ("nvfp4", "mxfp4")
    sf_k = K // block_size
    num_groups = len(offsets_list)

    if is_fp4:
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        tok_u8 = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev)
        w_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        tok_rt = tok_u8.view(torch.float4_e2m1fn_x2)
        w_rt = w_u8.view(torch.float4_e2m1fn_x2)
        tok_deq = _unpack_fp4(tok_u8, lut).view(S, K)
        w_deq = _unpack_fp4(w_u8, lut).view(E, N, K)
    else:  # mxfp8: FP8 E4M3 data, 1 byte/elem (not packed)
        tok_rt = (torch.randn(1, S, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w_rt = (torch.randn(E, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        tok_deq = tok_rt.float().view(S, K)
        w_deq = w_rt.float().view(E, N, K)
    if weight_major == "n":
        w_rt = w_rt.transpose(1, 2).contiguous().transpose(1, 2)
    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:  # mxfp4 / mxfp8 use an E8M0 (power-of-2) scale
        sfa_log = _rand_e8m0((S, sf_k), dev)
        sfb_log = _rand_e8m0((E, N, sf_k), dev)

    cfg = by_name(config_name)
    quant_scale_shape = _quant_scale_shape(S, N, quant_scale_reorder)
    compiled = _plan(
        _build_graph(
            E,
            S,
            N,
            K,
            num_groups,
            combo,
            offset_dt,
            quant=quant,
            output_dt=(cudnn.data_type.FLOAT if reduction_mode is not None else cudnn.data_type.BFLOAT16),
            reduction_mode=reduction_mode,
            reduction_dims=reduction_dims,
            reduction_stride=reduction_stride,
            reduction_dt=reduction_dt,
            reduction_compute_dt=reduction_compute_dt,
            quant_out_dt=quant_out_dt,
            quant_scale_dt=quant_scale_dt,
            quant_scale_reorder=quant_scale_reorder,
            quant_scale_dim=quant_scale_shape if quant_scale_reorder else None,
            weight_major=weight_major,
        ),
        config=cfg,
        cta_group=cta_group,
    )
    _blk, _, _sf_dt = _COMBOS[combo]
    _bs = compiled.chain.block_scale
    assert (_bs.sf_dtype, _bs.block_size) == (_DTYPE_FROM_CUDNN[_sf_dt], _blk)

    # SFA reordered + padded to 128 rows PER GROUP, then concatenated (for
    # 128-aligned groups this equals a single global _to_blocked). SFB per-expert.
    sfa_parts = []
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        sfa_parts.append(_to_blocked(sfa_log[b:e]))
    sfa_blk = torch.cat(sfa_parts).view(1, -1, 1)
    sfb_blk = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=offset_torch_dt, device=dev)
    if quant:
        q = torch.empty(1, S, N, dtype=quant_out_torch_dt, device=dev)
        if quant_scale_reorder:
            q_scale = torch.zeros(*quant_scale_shape, dtype=quant_scale_torch_dt, device=dev)
        else:
            q_scale = torch.empty(*quant_scale_shape, dtype=quant_scale_torch_dt, device=dev)
        output = [q, q_scale]
    elif reduction_mode is not None:
        term = torch.zeros(1, S, N, dtype=torch.float32, device=dev)
        if reduction_stride is None:
            red = torch.empty(*reduction_dims, dtype=reduction_torch_dt, device=dev)
        else:
            red = torch.empty_strided(
                reduction_dims,
                reduction_stride,
                dtype=reduction_torch_dt,
                device=dev,
            )
        output = [term, red]
    else:
        output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)

    compiled(_vp_bs(compiled, tok_rt, w_rt, output, sfa_blk, sfb_blk, fto=offsets))
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w_s = w_deq * sfb_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        ref[b:e] = tok_s[b:e] @ w_s[gi % E].T
    # nvfp4 (integer operands) is tight; mx paths carry fp16 rounding.
    tol = (1e-1, 1e-2) if combo == "nvfp4" else (2e-1, 2e-2)
    if quant:
        q_ref, scale_ref = _block_quant_ref(ref, 32, quant_out_torch_dt, quant_scale_torch_dt)
        if quant_scale_reorder:
            scale_ref = _to_blocked(scale_ref[0]).view_as(q_scale)
        torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(
            q.float(),
            q_ref.float(),
            atol=_block_quant_q_atol(quant_scale_torch_dt),
            rtol=0,
        )
    elif reduction_mode is not None:
        term, red = output
        torch.testing.assert_close(term[0], ref, atol=2e-1, rtol=2e-2)
        ref_dims = _reduction_dims(tuple(reduction_dims), (1, S, N))
        _assert_block_scale_reduction_close(
            red,
            _reduction_ref(term, reduction_mode, ref_dims).to(reduction_torch_dt),
            reduction_mode,
        )
    else:
        torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=tol[0], rtol=tol[1])


def _run_nonpacked_e2e(combo, config_name, cta_group, mode):
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 512, 256, 512
    offsets_list = [0, 100, 300]
    block_size = _COMBOS[combo][0]
    is_fp4 = combo in ("nvfp4", "mxfp4")
    sf_k = K // block_size

    if is_fp4:
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        if mode == "zero_stride":
            tok_base = torch.randint(0, 256, (K // 2,), dtype=torch.uint8, device=dev)
            w_base = torch.randint(0, 256, (K // 2,), dtype=torch.uint8, device=dev)
            tok_u8 = torch.as_strided(tok_base, (1, S, K // 2), (0, 0, 1))
            w_u8 = torch.as_strided(w_base, (E, N, K // 2), (0, 0, 1))
        else:
            pad = 16
            tok_store = torch.randint(0, 256, (1, S, K // 2 + pad), dtype=torch.uint8, device=dev)
            w_store = torch.randint(0, 256, (E, N, K // 2 + pad), dtype=torch.uint8, device=dev)
            tok_u8 = tok_store[:, :, : K // 2]
            w_u8 = w_store[:, :, : K // 2]
        tok_rt = tok_u8.view(torch.float4_e2m1fn_x2)
        w_rt = w_u8.view(torch.float4_e2m1fn_x2)
        tok_deq = _unpack_fp4(tok_u8, lut).view(S, K)
        w_deq = _unpack_fp4(w_u8, lut).view(E, N, K)
    elif mode == "zero_stride":
        tok_base = (torch.randn(K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w_base = (torch.randn(K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        tok_rt = torch.as_strided(tok_base, (1, S, K), (0, 0, 1))
        w_rt = torch.as_strided(w_base, (E, N, K), (0, 0, 1))
        tok_deq = tok_rt.float().view(S, K)
        w_deq = w_rt.float().view(E, N, K)
    else:
        pad = 16
        tok_store = (torch.randn(1, S, K + pad, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w_store = (torch.randn(E, N, K + pad, device=dev) * 0.5).to(torch.float8_e4m3fn)
        tok_rt = tok_store[:, :, :K]
        w_rt = w_store[:, :, :K]
        tok_deq = tok_rt.float().view(S, K)
        w_deq = w_rt.float().view(E, N, K)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((S, sf_k), dev)
        sfb_log = _rand_e8m0((E, N, sf_k), dev)

    cfg = by_name(config_name)
    compiled = _plan(
        _build_graph(E, S, N, K, len(offsets_list), combo),
        config=cfg,
        cta_group=cta_group,
    )

    sfa_blk = torch.cat(
        [_to_blocked(sfa_log[offsets_list[gi] : (offsets_list[gi + 1] if gi + 1 < len(offsets_list) else S)]) for gi in range(len(offsets_list))]
    ).view(1, -1, 1)
    sfb_blk = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output_store = torch.zeros(1, S, N + 16, dtype=torch.bfloat16, device=dev)
    output = output_store[:, :, :N]

    assert not tok_rt.is_contiguous() or not w_rt.is_contiguous()
    assert not output.is_contiguous()

    compiled(_vp_bs(compiled, tok_rt, w_rt, output, sfa_blk, sfb_blk, fto=offsets))
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w_s = w_deq * sfb_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(len(offsets_list)):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < len(offsets_list) else S
        if b != e:
            ref[b:e] = tok_s[b:e] @ w_s[gi % E].T
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize(
    "offsets_list",
    [
        [0, 512],  # 1 group / 1 expert, full S
        [0, 512, 768, 896],  # 4 groups over E=2 (BxE > E)
        [0, 256, 384, 512],  # 4 groups, last extends to S
    ],
)
@requires_sm100
def test_e2e_nvfp4_groups(offsets_list) -> None:
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=offsets_list)


@pytest.mark.parametrize(
    "cfg_name,cta_group",
    [
        ("CONFIG_sm100_256x128x128_128x128x32_cluster1x1", 1),
        ("CONFIG_sm100_256x128x128_128x128x32_cluster2x1", 2),
    ],
)
@requires_sm100
def test_e2e_split_m_tile(cfg_name, cta_group) -> None:
    """CTA tile spanning two MMA instructions along M. The SF words are one per
    128-row block, so M block mi reads SF word block mi; the per-routed-group A
    descriptor patch and the `row < group_end` guard are untouched."""
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512],
        combo="nvfp4",
        config_name=cfg_name,
        cta_group=cta_group,
    )


@pytest.mark.parametrize("combo", ["mxfp4", "mxfp8"])
@requires_sm100
def test_e2e_mx_combos(combo) -> None:
    # mxfp4 (FP4 + E8M0, block32) / mxfp8 (FP8 E4M3 + E8M0, block32), K-major.
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=[0, 256, 384, 512], combo=combo)


@pytest.mark.parametrize("cfg_name,cta_group", [(_CFG, 2), (_CFG_1CTA, 1)])
@requires_sm100
def test_e2e_mxfp8_n_major_weight(cfg_name, cta_group) -> None:
    # mxfp8 is the only MoE block-scale combo that allows an N-major weight
    # (fp4 sub-byte packing is rejected as non-K-major at JIT time).
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512],
        combo="mxfp8",
        config_name=cfg_name,
        cta_group=cta_group,
        weight_major="n",
    )


@requires_sm100
def test_e2e_fp4_rejects_n_major_weight() -> None:
    g = _build_graph(2, 512, 256, 512, num_groups=2, combo="nvfp4", weight_major="n")
    with pytest.raises(ValueError, match="must be K-major"):
        _plan(g, config=by_name(_CFG), cta_group=2)


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@requires_sm100
def test_e2e_1ctamma(combo) -> None:
    # 1-CTA MMA path (cluster1x1), all three block-scale combos. BxE>E groups.
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 256, 384],
        combo=combo,
        config_name=_CFG_1CTA,
        cta_group=1,
    )


@pytest.mark.parametrize(
    "case_name,combo,config_name,cta_group,out_dt,out_torch_dt,scale_dt," "scale_torch_dt,scale_reorder,S,N,offsets_list",
    _QUANT_CASES,
    ids=[case[0] for case in _QUANT_CASES],
)
@requires_sm100
def test_e2e_block_quant_epilogue(
    case_name,
    combo,
    config_name,
    cta_group,
    out_dt,
    out_torch_dt,
    scale_dt,
    scale_torch_dt,
    scale_reorder,
    S,
    N,
    offsets_list,
) -> None:
    _run_e2e(
        E=2,
        S=S,
        N=N,
        K=512,
        offsets_list=offsets_list,
        combo=combo,
        config_name=config_name,
        cta_group=cta_group,
        quant=True,
        quant_out_dt=out_dt,
        quant_out_torch_dt=out_torch_dt,
        quant_scale_dt=scale_dt,
        quant_scale_torch_dt=scale_torch_dt,
        quant_scale_reorder=scale_reorder,
    )


@requires_sm100
@pytest.mark.parametrize("cfg_name,cta_group", [(_CFG, 2), (_CFG_1CTA, 1)])
@pytest.mark.parametrize(
    "mode",
    [
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.AMAX,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ],
)
def test_e2e_reduction_epilogue(mode, cfg_name, cta_group) -> None:
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 100, 300],
        config_name=cfg_name,
        cta_group=cta_group,
        reduction_mode=mode,
        reduction_dims=[1, 1, 1],
    )


@requires_sm100
@pytest.mark.parametrize(
    "mode,red_dims,red_stride",
    [
        (cudnn.reduction_mode.ADD, [1, 512, 1], [0, 2, 1]),
        (cudnn.reduction_mode.AMAX, [1, 1, 256], [0, 0, 2]),
    ],
)
def test_e2e_reduction_epilogue_strided_output(mode, red_dims, red_stride) -> None:
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 100, 300],
        config_name=_CFG,
        cta_group=2,
        reduction_mode=mode,
        reduction_dims=red_dims,
        reduction_stride=red_stride,
    )


def test_moe_grouped_block_scale_matmul_fwd_reduction_rejects_int32() -> None:
    g = _build_graph(
        2,
        512,
        256,
        512,
        num_groups=3,
        reduction_mode=cudnn.reduction_mode.ADD,
        reduction_dims=(1, 1, 1),
        reduction_dt=cudnn.data_type.INT32,
        reduction_compute_dt=cudnn.data_type.INT32,
    )
    cfg = by_name(_CFG_1CTA)
    with pytest.raises(
        NotImplementedError,
        match="MoE block-scale reduction supports only fp32 compute/output",
    ):
        jit_from_cudnn_graph(g, config=cfg, cta_group=1)


# Group boundaries NOT multiples of 128: SFA is padded to 128 rows PER GROUP, so
# the kernel must track each group's start SF-block (scheduler cumsum), not
# group_begin//128 — else silent miscompute.
@pytest.mark.parametrize(
    "cta_group,config_name",
    [
        (2, _CFG),
        (1, _CFG_1CTA),
    ],
)
@requires_sm100
def test_e2e_unaligned_groups(cta_group, config_name) -> None:
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 100, 300],
        combo="nvfp4",
        config_name=config_name,
        cta_group=cta_group,
    )


@requires_sm100
def test_e2e_nvfp4_offset_int64() -> None:
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512],
        offset_dt=cudnn.data_type.INT64,
        offset_torch_dt=torch.int64,
    )


@requires_sm100
def test_e2e_nvfp4_empty_group() -> None:
    # An empty routed group (begin == end) must be skipped cleanly.
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=[0, 256, 256, 512])


@pytest.mark.parametrize(
    "combo,config_name,cta_group,mode",
    [
        ("nvfp4", _CFG, 2, "padded"),
        ("nvfp4", _CFG_1CTA, 1, "padded"),
        ("mxfp8", _CFG_1CTA, 1, "zero_stride"),
    ],
)
@requires_sm100
def test_e2e_nonpacked_tensors(combo, config_name, cta_group, mode) -> None:
    _run_nonpacked_e2e(combo, config_name, cta_group, mode)


@requires_sm100
@pytest.mark.parametrize("S,N", [(64, 4096), (4096, 64), (4096, 4096)])
def test_auto_config_is_accepted_by_the_registry(S, N):
    """Same invariant as the dense block-scale case: the grouped path shares the
    BlockScaleSpec machinery, so its 128-multiple tile constraint applies too and
    ``select_config`` must not pick a geometry the registry rejects."""
    from cudnn.gemm.frost.kernel_registry import candidates, preferred_pipeline
    from cudnn.gemm.frost.tile_config import as_pipeline, select_config

    chain = analyze(_build_graph(8, S, N, 512, 8))
    assert chain.has_block_scale and chain.has_moe
    cfg, _cta_group = select_config(chain.matmul.M, chain.matmul.N, chain.num_gemms, block_scale=chain.has_block_scale)
    cfg = as_pipeline(cfg, preferred_pipeline(chain))  # the config build_gemm_plan actually builds
    accepted = {c.name for _t, c in candidates(chain)}
    assert accepted, "the registry accepts no geometry at all for this chain"
    assert cfg.name in accepted, f"select_config picked {cfg.name!r}, which the registry rejects for this graph"


# ---------------------------------------------------------------------------
# sm107 pipeline (the sm100 grouped pipeline on the 64-byte-K block-scale MMA)
# ---------------------------------------------------------------------------

_SM107_CFG = "CONFIG_sm107_128x256x128_128x256x64_cluster2x1"
_SM107_CFG_1CTA = "CONFIG_sm107_128x256x128_128x256x64_cluster1x1"


def test_sm107_template_selection_and_arch_gate(monkeypatch) -> None:
    from cudnn.gemm.frost import compiler as C
    from cudnn.gemm.frost.kernel_registry import TEMPLATES, select_template

    monkeypatch.setattr(C, "_current_arch", lambda: 107)
    chain = analyze(_build_graph(2, 512, 256, 512, num_groups=2))
    for cta_group, cfg_name, want in (
        (1, _SM107_CFG_1CTA, "sm107_moe_grouped_block_scale_matmul_fwd_1ctamma.py"),
        (2, _SM107_CFG, "sm107_moe_grouped_block_scale_matmul_fwd_2ctamma.py"),
    ):
        cfg = by_name(cfg_name)
        tmpl = select_template(chain, cfg, cta_group=cta_group)
        assert tmpl.file == want
        assert tmpl.accepts(chain, cfg) is None
    # An sm100 config still pairs with the sm100 grouped templates on the same GPU.
    assert select_template(chain, by_name(_CFG), cta_group=2).file == "sm100_moe_grouped_block_scale_matmul_fwd_2ctamma.py"
    # ... and the sm107 templates are gated off older Blackwell.
    monkeypatch.setattr(C, "_current_arch", lambda: 100)
    tmpl = next(t for t in TEMPLATES if t.file == "sm107_moe_grouped_block_scale_matmul_fwd_1ctamma.py")
    assert "107 <= SM < 110" in tmpl.accepts(chain, by_name(_SM107_CFG_1CTA))


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("cfg_name,cta_group", [(_SM107_CFG, 2), (_SM107_CFG_1CTA, 1)])
@requires_sm107
def test_e2e_sm107(combo, cfg_name, cta_group) -> None:
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512],
        combo=combo,
        config_name=cfg_name,
        cta_group=cta_group,
    )


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("cta_group", [1, 2])
@pytest.mark.parametrize("cta_m,cta_n", [(128, 256), (256, 128), (256, 256)])
@requires_sm107
def test_e2e_sm107_multi_mma_m(combo, cta_group, cta_m, cta_n) -> None:
    # The grouped pipeline with the CTA tile split along M. The per-group-padded
    # SF blob is unchanged; what moves is the TMEM side, where SFA is indexed per
    # M block and SFB is walked across N blocks (they only differ once a block
    # count exceeds one, i.e. at cta_m/cta_n = 256).
    cluster = "cluster1x1" if cta_group == 1 else "cluster2x1"
    name = f"CONFIG_sm107_{cta_m}x{cta_n}x128_128x{cta_n}x64_{cluster}"
    _run_e2e(E=4, S=512, N=256, K=256, offsets_list=[0, 128, 256, 384], combo=combo, config_name=name, cta_group=cta_group)


@pytest.mark.parametrize("cfg_name,cta_group", [(_SM107_CFG, 2), (_SM107_CFG_1CTA, 1)])
@requires_sm107
def test_e2e_sm107_unaligned_groups(cfg_name, cta_group) -> None:
    # Group offsets that are not 128-aligned — the per-group-padded SF blob
    # layout is the sm100 one, so the 64-byte-K MMA must not disturb it.
    _run_e2e(E=2, S=512, N=256, K=512, offsets_list=[0, 100, 300], config_name=cfg_name, cta_group=cta_group)
