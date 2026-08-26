# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single MoE grouped matmul + pointwise epilogue (rides the multi-MoE machinery
with num_gemms == 1): op DAG, per-row aux, per-group aux, fan-in, grouped amax,
quant terminal."""

from __future__ import annotations

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import pytest
import torch

from gemm_test_utils import requires_sm100
from test_matmul import _fp4_quant_ref, _unpack_e2m1, _col_quant_reference

import inspect

from cudnn.gemm.frost import compiler
from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.epilogue_codegen import generate
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.tile_config import by_name

pytestmark = [pytest.mark.L0, requires_sm100]

_GEOMETRIES = [
    ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", 1),
    ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2),
]

_E, _S, _N, _K, _G = 4, 512, 256, 256, 4
_OFFSETS = (0, 128, 130, 384)


def _graph(io=cudnn.data_type.BFLOAT16):
    g = cudnn.pygraph(
        io_data_type=io,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, _S, _K], stride=[_S * _K, _K, 1], data_type=cudnn.data_type.BFLOAT16)
    w = g.tensor(name="weight", dim=[_E, _K, _N], stride=[_K * _N, 1, _K], data_type=cudnn.data_type.BFLOAT16)
    fto = g.tensor(name="fto", dim=[_G, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    c = g.moe_grouped_matmul(
        tok,
        w,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe0",
    )
    return g, c, fto


def _data():
    torch.manual_seed(0)
    token = torch.randn(1, _S, _K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(_E, _N, _K, device="cuda", dtype=torch.bfloat16) * 0.05
    offs = torch.tensor(_OFFSETS, dtype=torch.int32, device="cuda")
    return token, weight, offs


def _ref_c(token, weight):
    c = torch.zeros(_S, _N, device="cuda", dtype=torch.float32)
    starts = list(_OFFSETS)
    for gi in range(_G):
        b, e = starts[gi], (starts[gi + 1] if gi + 1 < _G else _S)
        if b < e:
            c[b:e] = token[0, b:e].float() @ weight[gi % _E].float().T
    return c


@pytest.mark.parametrize("config_name,cta_group", _GEOMETRIES, ids=lambda v: str(v))
def test_single_moe_relu_scale_epilogue(config_name, cta_group):
    g, c, _fto = _graph()
    sc = g.tensor(name="scale", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    r = g.relu(input=c, name="r")
    y = g.mul(a=r, b=sc, name="m")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    compiled = jit_from_cudnn_graph(g, by_name(config_name), cta_group=cta_group)
    assert compiled.chain.num_gemms == 1 and [op.op for op in compiled.chain.ops] == ["relu", "mul"]

    token, weight, offs = _data()
    scale = torch.full((1, 1, 1), 0.5, device="cuda", dtype=torch.float32)
    out = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    bd = compiled.binding
    compiled({bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out, bd.aux[0]: scale})
    torch.cuda.synchronize()

    ref = (torch.relu(_ref_c(token, weight)) * 0.5).to(torch.bfloat16).float()
    torch.testing.assert_close(out[0].float(), ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("config_name,cta_group", _GEOMETRIES, ids=lambda v: str(v))
def test_single_moe_srelu_prob_epilogue(config_name, cta_group):
    g, c, _fto = _graph()
    p = g.tensor(name="prob", dim=[1, _S, 1], stride=[_S, 1, 1], data_type=cudnn.data_type.FLOAT)
    r = g.relu(input=c, name="r")
    sq = g.mul(a=r, b=r, name="sq")
    y = g.mul(a=sq, b=p, name="gate")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    compiled = jit_from_cudnn_graph(g, by_name(config_name), cta_group=cta_group)

    token, weight, offs = _data()
    prob = torch.rand(1, _S, 1, device="cuda", dtype=torch.float32)
    out = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    bd = compiled.binding
    compiled({bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out, bd.aux[0]: prob})
    torch.cuda.synchronize()

    ref = (torch.relu(_ref_c(token, weight)) ** 2 * prob[0]).to(torch.bfloat16).float()
    torch.testing.assert_close(out[0].float(), ref, atol=8e-2, rtol=8e-2)


def _group_ranges(offsets):
    starts = list(offsets)
    return [(starts[gi], starts[gi + 1] if gi + 1 < len(starts) else _S) for gi in range(len(starts))]


@pytest.mark.parametrize("config_name,cta_group", _GEOMETRIES, ids=lambda v: str(v))
def test_single_moe_per_group_alpha(config_name, cta_group):
    g, c, _fto = _graph()
    alpha = g.tensor(name="alpha", dim=[_G, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    y = g.mul(a=c, b=alpha, name="scale")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    compiled = jit_from_cudnn_graph(g, by_name(config_name), cta_group=cta_group)
    (aux_ref,) = compiled.chain.aux_tensors
    assert aux_ref.grouped_by_moe and aux_ref.bcast_mode == "scalar"

    token, weight, offs = _data()
    alpha_t = torch.rand(_G, 1, 1, device="cuda", dtype=torch.float32) + 0.5
    out = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    bd = compiled.binding
    compiled({bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out, bd.aux[0]: alpha_t})
    torch.cuda.synchronize()

    ref = _ref_c(token, weight)
    for gi, (b, e) in enumerate(_group_ranges(_OFFSETS)):
        ref[b:e] *= alpha_t[gi, 0, 0]
    torch.testing.assert_close(out[0].float(), ref.to(torch.bfloat16).float(), atol=5e-2, rtol=5e-2)


def test_single_moe_per_group_bias_empty_group():
    offsets = (0, 128, 128, 384)
    g, c, _fto = _graph()
    bias = g.tensor(name="bias", dim=[_G, 1, _N], stride=[_N, _N, 1], data_type=cudnn.data_type.FLOAT)
    y = g.add(a=c, b=bias, name="b")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)
    (aux_ref,) = compiled.chain.aux_tensors
    assert aux_ref.grouped_by_moe and aux_ref.bcast_mode == "per_col"

    token, weight, _ = _data()
    offs = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    bias_t = torch.randn(_G, 1, _N, device="cuda", dtype=torch.float32)
    out = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    bd = compiled.binding
    compiled({bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out, bd.aux[0]: bias_t})
    torch.cuda.synchronize()

    ref = torch.zeros(_S, _N, device="cuda", dtype=torch.float32)
    for gi, (b, e) in enumerate(_group_ranges(offsets)):
        if b < e:
            ref[b:e] = token[0, b:e].float() @ weight[gi % _E].float().T + bias_t[gi, 0]
    torch.testing.assert_close(out[0].float(), ref.to(torch.bfloat16).float(), atol=5e-2, rtol=5e-2)


def test_dual_moe_swiglu_per_group_alpha():
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, _S, _K], stride=[_S * _K, _K, 1], data_type=cudnn.data_type.BFLOAT16)
    w0 = g.tensor(name="weight0", dim=[_E, _K, _N], stride=[_K * _N, 1, _K], data_type=cudnn.data_type.BFLOAT16)
    w1 = g.tensor(name="weight1", dim=[_E, _K, _N], stride=[_K * _N, 1, _K], data_type=cudnn.data_type.BFLOAT16)
    fto = g.tensor(name="fto", dim=[_G, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    c0 = g.moe_grouped_matmul(tok, w0, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe0")
    c1 = g.moe_grouped_matmul(tok, w1, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe1")
    alpha = g.tensor(name="alpha", dim=[_G, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    sw = g.swish(input=c0, name="sw")
    prod = g.mul(a=sw, b=c1, name="prod")
    y = g.mul(a=prod, b=alpha, name="scale")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x128x128_128x128x32_cluster2x1"), cta_group=2)
    assert compiled.chain.num_gemms == 2
    (aux_ref,) = compiled.chain.aux_tensors
    assert aux_ref.grouped_by_moe and aux_ref.bcast_mode == "scalar"

    token, weight0, offs = _data()
    weight1 = torch.randn(_E, _N, _K, device="cuda", dtype=torch.bfloat16) * 0.05
    alpha_t = torch.rand(_G, 1, 1, device="cuda", dtype=torch.float32) + 0.5
    out = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    bd = compiled.binding
    compiled(
        {
            bd.a_operands[0]: token,
            bd.b_operands[0]: weight0,
            bd.b_operands[1]: weight1,
            bd.first_token_offset: offs,
            bd.outputs[0]: out,
            bd.aux[0]: alpha_t,
        }
    )
    torch.cuda.synchronize()

    ref = torch.zeros(_S, _N, device="cuda", dtype=torch.float32)
    for gi, (b, e) in enumerate(_group_ranges(_OFFSETS)):
        if b < e:
            r0 = token[0, b:e].float() @ weight0[gi % _E].float().T
            r1 = token[0, b:e].float() @ weight1[gi % _E].float().T
            ref[b:e] = r0 * torch.sigmoid(r0) * r1 * alpha_t[gi, 0, 0]
    torch.testing.assert_close(out[0].float(), ref.to(torch.bfloat16).float(), atol=8e-2, rtol=8e-2)


def test_single_moe_per_group_aux_bad_leading_dim():
    g, c, _fto = _graph()
    alpha = g.tensor(name="alpha", dim=[_G - 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    y = g.mul(a=c, b=alpha, name="scale")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    with pytest.raises(ValueError, match="must be 1 or"):
        jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster1x1"), cta_group=1)


def test_single_moe_per_group_aux_rejects_m_axis():
    g, c, _fto = _graph()
    alpha = g.tensor(name="alpha", dim=[_G, _S, 1], stride=[_S, 1, 1], data_type=cudnn.data_type.FLOAT)
    y = g.mul(a=c, b=alpha, name="scale")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    with pytest.raises(ValueError, match="per-group aux"):
        jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster1x1"), cta_group=1)


def test_single_moe_per_group_aux_runtime_shape_mismatch():
    g, c, _fto = _graph()
    alpha = g.tensor(name="alpha", dim=[_G, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    y = g.mul(a=c, b=alpha, name="scale")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster1x1"), cta_group=1)

    token, weight, offs = _data()
    bad_alpha = torch.rand(_G - 1, 1, 1, device="cuda", dtype=torch.float32)
    out = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    bd = compiled.binding
    with pytest.raises(ValueError, match="per-group aux"):
        compiled({bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out, bd.aux[0]: bad_alpha})


def test_single_moe_grouped_amax_only_no_dense_output():
    g, c, fto = _graph()
    sw = g.swish(input=c, name="sw")
    amax = g.reduction(input=sw, mode=cudnn.reduction_mode.AMAX, name="amax", group_offset=fto)
    amax.set_dim([_G, 1, 1]).set_stride([1, 1, 1])
    amax.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)
    assert not compiled.chain.output_specs
    assert [o.source for o in compiled.chain.outputs] == ["reduction_0"]

    token, weight, offs = _data()
    out_am = torch.zeros(_G, 1, 1, device="cuda", dtype=torch.float32)
    bd = compiled.binding
    compiled({bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out_am})
    torch.cuda.synchronize()

    csw = _ref_c(token, weight)
    csw = csw * torch.sigmoid(csw)
    am_ref = torch.stack([(csw[b:e].abs().amax() if b < e else torch.tensor(0.0, device="cuda")) for b, e in _group_ranges(_OFFSETS)])
    torch.testing.assert_close(out_am.flatten(), am_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bs,scale_dt", [(32, "e8m0"), (16, "e4m3")], ids=["mxfp4", "nvfp4"])
def test_single_moe_fp4_quant_grouped_amax(bs, scale_dt):
    g, c, fto = _graph()
    sw = g.swish(input=c, name="sw")
    am = g.reduction(input=sw, mode=cudnn.reduction_mode.AMAX, name="amax", group_offset=fto)
    am.set_dim([_G, 1, 1]).set_stride([1, 1, 1])
    am.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    q, qs = g.block_scale_quantize(input=sw, block_size=bs, axis=-1, name="q4")
    q.set_data_type(cudnn.data_type.FP4_E2M1).set_output(True)
    qs.set_dim([1, _S, _N // bs]).set_stride([_S * (_N // bs), _N // bs, 1])
    scale_cudnn = cudnn.data_type.FP8_E8M0 if scale_dt == "e8m0" else cudnn.data_type.FP8_E4M3
    scale_torch = torch.float8_e8m0fnu if scale_dt == "e8m0" else torch.float8_e4m3fn
    qs.set_data_type(scale_cudnn).set_output(True)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)

    token, weight, offs = _data()
    out_q = torch.zeros(1, _S, _N // 2, dtype=torch.uint8, device="cuda")
    out_am = torch.zeros(_G, 1, 1, device="cuda", dtype=torch.float32)
    out_sf = torch.zeros(1, _S, _N // bs, dtype=scale_torch, device="cuda")
    bd = compiled.binding
    compiled(
        {bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out_q, bd.outputs[1]: out_am, bd.outputs[2]: out_sf}
    )
    torch.cuda.synchronize()

    csw = _ref_c(token, weight)
    csw = csw * torch.sigmoid(csw)
    q_ref, sf_ref = _fp4_quant_ref(csw.unsqueeze(0), bs, scale_torch, -1)
    if scale_dt == "e8m0":
        torch.testing.assert_close(out_sf.float(), sf_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(_unpack_e2m1(out_q, _N), q_ref, atol=0, rtol=0)
    else:
        torch.testing.assert_close(out_sf.float(), sf_ref.float(), atol=0, rtol=0.07)
        deq = _unpack_e2m1(out_q, _N) * out_sf.float().repeat_interleave(bs, dim=2)
        err = (deq - csw.unsqueeze(0)).abs()
        tol = 0.34 * csw.unsqueeze(0).abs() + 0.05 * csw.abs().max()
        assert (err <= tol).float().mean().item() > 0.999
    am_ref = torch.stack([(csw[b:e].abs().amax() if b < e else torch.tensor(0.0, device="cuda")) for b, e in _group_ranges(_OFFSETS)])
    torch.testing.assert_close(out_am.flatten(), am_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bs", [32, 16], ids=["block32", "block16"])
def test_single_moe_col_quant_aligned_groups(bs):
    """MoE col (M-axis) quant under the alignment CONTRACT: every fto value is
    a multiple of block_size, so no column block spans a group boundary."""
    # group row counts are blocksize*4 multiples (bs=32 -> 128-multiples,
    # bs=16 -> 64-multiples), aligning with the F8_128x4 4-block atom quads.
    offsets = (0, 128, 256, 384) if bs == 32 else (0, 128, 192, 384)
    g, c, _fto = _graph()
    sw = g.swish(input=c, name="sw")
    q, qs = g.block_scale_quantize(input=sw, block_size=bs, axis=1, name="qc")
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    qs.set_dim([1, _S // bs, _N]).set_stride([_S // bs * _N, _N, 1])
    qs.set_data_type(cudnn.data_type.FP8_E8M0).set_output(True)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)

    token, weight, _ = _data()
    offs = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    out_q = torch.zeros(1, _S, _N, dtype=torch.float8_e4m3fn, device="cuda")
    out_sf = torch.zeros(1, _S // bs, _N, dtype=torch.float8_e8m0fnu, device="cuda")
    bd = compiled.binding
    compiled({bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out_q, bd.outputs[1]: out_sf})
    torch.cuda.synchronize()

    csw = torch.zeros(_S, _N, device="cuda", dtype=torch.float32)
    for gi, (b, e) in enumerate(_group_ranges(offsets)):
        if b < e:
            csw[b:e] = token[0, b:e].float() @ weight[gi % _E].float().T
    csw = csw * torch.sigmoid(csw)
    q_ref, s_ref = _col_quant_reference(csw.unsqueeze(0), bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(out_sf.float(), s_ref.float(), atol=0, rtol=0)
    # bf16 randn accumulation-order noise can flip isolated fp8 RN boundaries;
    # require near-total bit-exactness and 1-ulp closeness everywhere.
    mismatch = (out_q.float() != q_ref.float()).float().mean().item()
    assert mismatch < 1e-4, mismatch
    deq = out_q.float() * out_sf.float().repeat_interleave(bs, dim=1)
    err = (deq - csw.unsqueeze(0)).abs()
    tol = 0.08 * csw.unsqueeze(0).abs() + 0.02 * csw.abs().max()
    assert (err <= tol).float().mean().item() > 0.999


def _discrete_col_sf_gather_idx(offsets, bs):
    """Flat element indices of every group's segmented col-SF bytes (the
    cutedsl discrete_col_sfd layout): same transposed F8_128x4 atom, but each
    group is its own compact table at its atom-quad base, N-atom stride = the
    group's OWN M-atom count. Row order matches the logical (M//bs, N) scale."""
    n_atoms = (_N + 127) // 128
    rows = []
    for b, e in _group_ranges(offsets):
        if b >= e:
            continue
        base = (b // (4 * bs)) * n_atoms * 512
        mcb = ((e - b) // bs + 3) // 4
        for mb in range((e - b) // bs):
            rows.append([base + ((n // 128) * mcb + mb // 4) * 512 + (n % 32) * 16 + ((n % 128) // 32) * 4 + (mb % 4) for n in range(_N)])
    return torch.tensor(rows, device="cuda", dtype=torch.long)


@pytest.mark.parametrize("bs", [32, 16])
def test_single_moe_col_quant_grouped_segmented(bs):
    """Grouped (per-group segmented) col quant: passing the MoE fto as the
    quant node's group_offset selects the cutedsl discrete_col_sfd layout.
    CONTRACT: every fto value is a multiple of 4*block_size (atom quads)."""
    offsets = (0, 128, 256, 384) if bs == 32 else (0, 128, 192, 384)
    g, c, fto = _graph()
    sw = g.swish(input=c, name="sw")
    q, qs = g.block_scale_quantize(input=sw, block_size=bs, axis=1, group_offset=fto, name="qc")
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    n_pad = (_N + 127) // 128 * 128
    mcb4 = (_S // bs + 3) // 4 * 4
    qs.set_dim([1, n_pad, mcb4]).set_stride([n_pad * mcb4, mcb4, 1])
    qs.set_data_type(cudnn.data_type.FP8_E8M0).set_output(True)
    qs.set_reordering_type(cudnn.tensor_reordering.F8_128x4)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)
    assert compiled.chain.quants[0].grouped_by_moe

    token, weight, _ = _data()
    offs = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    out_q = torch.zeros(1, _S, _N, dtype=torch.float8_e4m3fn, device="cuda")
    out_sf = torch.zeros(1, n_pad, mcb4, dtype=torch.float8_e8m0fnu, device="cuda")
    bd = compiled.binding
    compiled({bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out_q, bd.outputs[1]: out_sf})
    torch.cuda.synchronize()

    csw = torch.zeros(_S, _N, device="cuda", dtype=torch.float32)
    for gi, (b, e) in enumerate(_group_ranges(offsets)):
        if b < e:
            csw[b:e] = token[0, b:e].float() @ weight[gi % _E].float().T
    csw = csw * torch.sigmoid(csw)
    q_ref, s_ref = _col_quant_reference(csw.unsqueeze(0), bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    got_s = out_sf.view(-1)[_discrete_col_sf_gather_idx(offsets, bs)]
    torch.testing.assert_close(got_s.float(), s_ref[0].float(), atol=0, rtol=0)
    mismatch = (out_q.float() != q_ref.float()).float().mean().item()
    assert mismatch < 1e-4, mismatch


def test_single_moe_col_quant_group_offset_rejections():
    # Row quant with group_offset: rejected (row SF is already per-group contiguous).
    g, c, fto = _graph()
    sw = g.swish(input=c, name="sw")
    q, qs = g.block_scale_quantize(input=sw, block_size=32, axis=-1, group_offset=fto, name="q")
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    qs.set_data_type(cudnn.data_type.FP8_E8M0).set_output(True)
    with pytest.raises(ValueError, match="supports only the M axis"):
        jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)

    # group_offset that is not the MoE fto: rejected.
    g, c, fto = _graph()
    other = g.tensor(name="other", dim=[_G, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    sw = g.swish(input=c, name="sw")
    q, qs = g.block_scale_quantize(input=sw, block_size=32, axis=1, group_offset=other, name="q")
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    qs.set_data_type(cudnn.data_type.FP8_E8M0).set_output(True)
    with pytest.raises(ValueError, match="must be the MoE"):
        jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)

    # Grouped col quant without F8_128x4 reordering: rejected.
    g, c, fto = _graph()
    sw = g.swish(input=c, name="sw")
    q, qs = g.block_scale_quantize(input=sw, block_size=32, axis=1, group_offset=fto, name="q")
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    qs.set_dim([1, _S // 32, _N]).set_stride([_S // 32 * _N, _N, 1])
    qs.set_data_type(cudnn.data_type.FP8_E8M0).set_output(True)
    with pytest.raises(ValueError, match="requires F8_128x4"):
        jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)


def test_single_moe_swish_quant_grouped_amax():
    g, c, fto = _graph()
    sw = g.swish(input=c, name="sw")
    amax = g.reduction(input=sw, mode=cudnn.reduction_mode.AMAX, name="amax", group_offset=fto)
    amax.set_dim([_G, 1, 1]).set_stride([1, 1, 1])
    amax.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    q, qs = g.block_scale_quantize(input=sw, block_size=32, axis=-1, name="quant")
    q.set_data_type(cudnn.data_type.FP8_E4M3).set_output(True)
    qs.set_data_type(cudnn.data_type.FP8_E8M0).set_output(True)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)

    token, weight, offs = _data()
    out_q = torch.empty(1, _S, _N, device="cuda", dtype=torch.float8_e4m3fn)
    out_sf = torch.empty(1, _S, _N // 32, device="cuda", dtype=torch.float8_e8m0fnu)
    out_am = torch.zeros(_G, 1, 1, device="cuda", dtype=torch.float32)
    bd = compiled.binding
    compiled(
        {
            bd.a_operands[0]: token,
            bd.b_operands[0]: weight,
            bd.first_token_offset: offs,
            bd.outputs[0]: out_q,
            bd.outputs[1]: out_am,
            bd.outputs[2]: out_sf,
        }
    )
    torch.cuda.synchronize()

    csw = _ref_c(token, weight)
    csw = csw * torch.sigmoid(csw)
    deq = out_q[0].float() * out_sf[0].float().repeat_interleave(32, dim=1)
    valid = csw.abs() > 1e-3
    assert ((deq - csw).abs() / csw.abs().clamp(min=1e-3))[valid].max().item() < 0.15

    starts = list(_OFFSETS)
    am_ref = torch.stack(
        [
            (
                csw[starts[gi] : (starts[gi + 1] if gi + 1 < _G else _S)].abs().amax()
                if starts[gi] < (starts[gi + 1] if gi + 1 < _G else _S)
                else torch.tensor(0.0, device="cuda")
            )
            for gi in range(_G)
        ]
    )
    torch.testing.assert_close(out_am.flatten(), am_ref, atol=1e-3, rtol=1e-3)


def test_single_moe_srelu_full_cutedsl_mirror():
    """The full cutedsl srelu-family semantics in one graph:
    D = dual_quant(srelu(alpha[e] * (tok @ w[e].T) + bias[e]) * prob[m])
    with per-expert amax — per-group scalar alpha, per-group per-col bias,
    srelu = relu^2 (fan-in mul), per-token prob, dual row+col fp8 quant with
    F8_128x4 scales, grouped amax reduction."""
    from test_matmul import _block_quant_reference, _f8_row_scale_addr, _f8_col_scale_addr

    bs = 32
    offsets = (0, 128, 256, 384)  # col-quant MoE CONTRACT: fto % bs == 0
    g, c, fto = _graph()
    alpha = g.tensor(name="alpha", dim=[_G, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    bias = g.tensor(name="bias", dim=[_G, 1, _N], stride=[_N, _N, 1], data_type=cudnn.data_type.FLOAT)
    prob = g.tensor(name="prob", dim=[1, _S, 1], stride=[_S, 1, 1], data_type=cudnn.data_type.FLOAT)
    v = g.mul(a=c, b=alpha, name="alpha_scale")
    v = g.add(a=v, b=bias, name="bias_add")
    r = g.relu(input=v, name="r")
    sq = g.mul(a=r, b=r, name="sq")
    s = g.mul(a=sq, b=prob, name="prob_gate")
    amax = g.reduction(input=s, mode=cudnn.reduction_mode.AMAX, name="amax", group_offset=fto)
    amax.set_dim([_G, 1, 1]).set_stride([1, 1, 1])
    amax.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    qr, qsr = g.block_scale_quantize(input=s, block_size=bs, axis=-1, name="qrow")
    qr.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    qsr.set_dim([1, _S, (_N // bs + 3) // 4 * 4]).set_stride([_S * ((_N // bs + 3) // 4 * 4), (_N // bs + 3) // 4 * 4, 1])
    qsr.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    qsr.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
    qc, qsc = g.block_scale_quantize(input=s, block_size=bs, axis=1, name="qcol")
    qc.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    qsc.set_dim([1, (_N + 127) // 128 * 128, (_S // bs + 3) // 4 * 4]).set_stride(
        [((_N + 127) // 128 * 128) * ((_S // bs + 3) // 4 * 4), (_S // bs + 3) // 4 * 4, 1]
    )
    qsc.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    qsc.set_reordering_type(cudnn.tensor_reordering.F8_128x4)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)
    assert [op.op for op in compiled.chain.ops] == ["mul", "add", "relu", "mul", "mul"]
    assert len(compiled.chain.quants) == 2 and len(compiled.chain.reductions) == 1

    token, weight, _ = _data()
    offs = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    alpha_t = torch.rand(_G, 1, 1, device="cuda", dtype=torch.float32) + 0.5
    bias_t = torch.randn(_G, 1, _N, device="cuda", dtype=torch.float32)
    prob_t = torch.rand(1, _S, 1, device="cuda", dtype=torch.float32)
    q_row = torch.zeros(1, _S, _N, dtype=torch.float8_e4m3fn, device="cuda")
    q_col = torch.zeros(1, _S, _N, dtype=torch.float8_e4m3fn, device="cuda")
    am = torch.zeros(_G, 1, 1, dtype=torch.float32, device="cuda")
    qs_row = torch.zeros(1, _S, (_N // bs + 3) // 4 * 4, dtype=torch.float8_e8m0fnu, device="cuda")
    qs_col = torch.zeros(1, (_N + 127) // 128 * 128, (_S // bs + 3) // 4 * 4, dtype=torch.float8_e8m0fnu, device="cuda")
    bd = compiled.binding
    compiled(
        {
            bd.a_operands[0]: token,
            bd.b_operands[0]: weight,
            bd.first_token_offset: offs,
            bd.outputs[0]: q_row,
            bd.outputs[1]: q_col,
            bd.outputs[2]: am,
            bd.outputs[3]: qs_row,
            bd.outputs[4]: qs_col,
            bd.aux[0]: alpha_t,
            bd.aux[1]: bias_t,
            bd.aux[2]: prob_t,
        }
    )
    torch.cuda.synchronize()

    ref = torch.zeros(_S, _N, device="cuda", dtype=torch.float32)
    for gi, (b, e) in enumerate(_group_ranges(offsets)):
        if b < e:
            cg = token[0, b:e].float() @ weight[gi % _E].float().T
            ref[b:e] = alpha_t[gi, 0, 0] * cg + bias_t[gi, 0]
    ref = torch.relu(ref) ** 2 * prob_t[0]

    qr_ref, sr_ref = _block_quant_reference(ref.unsqueeze(0), bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    qc_ref, sc_ref = _col_quant_reference(ref.unsqueeze(0), bs, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    got_sr = qs_row.view(1, -1)[:, _f8_row_scale_addr(_S, _N, bs)]
    torch.testing.assert_close(got_sr.float(), sr_ref.float(), atol=0, rtol=0)
    got_sc = qs_col.view(1, -1)[:, _f8_col_scale_addr(_S, _N, bs)]
    torch.testing.assert_close(got_sc.float(), sc_ref.permute(0, 2, 1).float(), atol=0, rtol=0)
    assert (q_row.float() != qr_ref.float()).float().mean().item() < 1e-3
    assert (q_col.float() != qc_ref.float()).float().mean().item() < 1e-3
    am_ref = torch.stack([ref[b:e].abs().amax() if b < e else torch.tensor(0.0, device="cuda") for b, e in _group_ranges(offsets)])
    torch.testing.assert_close(am.flatten(), am_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("config_name,cta_group", _GEOMETRIES, ids=lambda v: str(v))
def test_single_moe_dsrelu_backward_mirror(config_name, cta_group):
    """cutedsl dsrelu-family backward on aux-rooted op chains:
    dX = alpha[e] * acc * relu(C) * 2 * prob[m]; d_srelu = relu(C)^2 (pure-aux
    dense output); dprob[m] = sum_n(relu(C)^2 * alpha * acc)."""
    g, acc, fto = _graph()
    C = g.tensor(name="saved_c", dim=[1, _S, _N], stride=[_S * _N, _N, 1], data_type=cudnn.data_type.FLOAT)
    alpha = g.tensor(name="alpha", dim=[_G, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    prob = g.tensor(name="prob", dim=[1, _S, 1], stride=[_S, 1, 1], data_type=cudnn.data_type.FLOAT)
    two = g.tensor(name="two", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    rc = g.relu(input=C, name="rC")
    ga = g.mul(a=acc, b=alpha, name="ga")
    gg = g.mul(a=ga, b=prob, name="gg")
    dx = g.mul(a=g.mul(a=gg, b=rc, name="dx0"), b=two, name="dx")
    dx.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    rsq = g.mul(a=rc, b=rc, name="rsq")
    rsq.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    p2 = g.mul(a=rsq, b=ga, name="p2")
    dprob = g.reduction(input=p2, mode=cudnn.reduction_mode.ADD, name="dprob")
    dprob.set_dim([1, _S, 1]).set_stride([_S, 1, 1])
    dprob.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    compiled = jit_from_cudnn_graph(g, by_name(config_name), cta_group=cta_group)
    assert compiled.chain.ops[0].op == "aux_load" and compiled.chain.ops[0].aux == "saved_c"

    token, weight, offs = _data()
    c_t = torch.randn(1, _S, _N, device="cuda", dtype=torch.float32)
    alpha_t = torch.rand(_G, 1, 1, device="cuda", dtype=torch.float32) + 0.5
    prob_t = torch.rand(1, _S, 1, device="cuda", dtype=torch.float32)
    two_t = torch.full((1, 1, 1), 2.0, device="cuda", dtype=torch.float32)
    out_dx = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    out_dsr = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    out_dp = torch.zeros(1, _S, 1, device="cuda", dtype=torch.float32)
    bd = compiled.binding
    compiled(
        {
            bd.a_operands[0]: token,
            bd.b_operands[0]: weight,
            bd.first_token_offset: offs,
            bd.outputs[0]: out_dx,
            bd.outputs[1]: out_dsr,
            bd.outputs[2]: out_dp,
            bd.aux[0]: c_t,
            bd.aux[1]: alpha_t,
            bd.aux[2]: prob_t,
            bd.aux[3]: two_t,
        }
    )
    torch.cuda.synchronize()

    acc_ref = _ref_c(token, weight)
    ga_ref = acc_ref.clone()
    for gi, (b, e) in enumerate(_group_ranges(_OFFSETS)):
        ga_ref[b:e] *= alpha_t[gi, 0, 0]
    rc_ref = torch.relu(c_t[0])
    dx_ref = ga_ref * prob_t[0] * rc_ref * 2.0
    dsr_ref = rc_ref**2
    dp_ref = (dsr_ref * ga_ref).sum(dim=1, keepdim=True)
    torch.testing.assert_close(out_dx[0].float(), dx_ref.to(torch.bfloat16).float(), atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(out_dsr[0].float(), dsr_ref.to(torch.bfloat16).float(), atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(out_dp[0], dp_ref, atol=5e-1, rtol=5e-2)


def test_single_moe_dswiglu_backward():
    """cutedsl dswiglu backward under the split layout: single upstream-grad
    GEMM + y1/y2 saved pre-activations as aux roots ->
    dy1 = gg * y2 * sig * (1 + y1*(1-sig)); dy2 = gg * y1 * sig;
    dprob = sum_n(y2 * silu(y1) * ga); dbias1/dbias2 = grouped col sums."""
    g, acc, fto = _graph()
    y1 = g.tensor(name="y1", dim=[1, _S, _N], stride=[_S * _N, _N, 1], data_type=cudnn.data_type.FLOAT)
    y2 = g.tensor(name="y2", dim=[1, _S, _N], stride=[_S * _N, _N, 1], data_type=cudnn.data_type.FLOAT)
    alpha = g.tensor(name="alpha", dim=[_G, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    prob = g.tensor(name="prob", dim=[1, _S, 1], stride=[_S, 1, 1], data_type=cudnn.data_type.FLOAT)
    one = g.tensor(name="one", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    ga = g.mul(a=acc, b=alpha, name="ga")
    gg = g.mul(a=ga, b=prob, name="gg")
    sig = g.sigmoid(input=y1, name="sig")
    silu = g.mul(a=sig, b=y1, name="silu")
    dy2 = g.mul(a=gg, b=silu, name="dy2")
    dy2.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    oms = g.sub(a=one, b=sig, name="oms")
    t1 = g.add(a=g.mul(a=oms, b=y1, name="t"), b=one, name="t1")
    ds = g.mul(a=sig, b=t1, name="ds")
    dy1 = g.mul(a=g.mul(a=gg, b=ds, name="d1a"), b=y2, name="dy1")
    dy1.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    p2 = g.mul(a=g.mul(a=silu, b=y2, name="p1"), b=ga, name="p2")
    dprob = g.reduction(input=p2, mode=cudnn.reduction_mode.ADD, name="dprob")
    dprob.set_dim([1, _S, 1]).set_stride([_S, 1, 1])
    dprob.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    db1 = g.reduction(input=dy1, mode=cudnn.reduction_mode.ADD, name="dbias1", group_offset=fto)
    db1.set_dim([_G, 1, _N]).set_stride([_N, _N, 1])
    db1.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    db2 = g.reduction(input=dy2, mode=cudnn.reduction_mode.ADD, name="dbias2", group_offset=fto)
    db2.set_dim([_G, 1, _N]).set_stride([_N, _N, 1])
    db2.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)
    assert sum(1 for op in compiled.chain.ops if op.op == "aux_load") == 1
    assert len(compiled.chain.reductions) == 3

    token, weight, offs = _data()
    y1_t = torch.randn(1, _S, _N, device="cuda", dtype=torch.float32)
    y2_t = torch.randn(1, _S, _N, device="cuda", dtype=torch.float32)
    alpha_t = torch.rand(_G, 1, 1, device="cuda", dtype=torch.float32) + 0.5
    prob_t = torch.rand(1, _S, 1, device="cuda", dtype=torch.float32)
    one_t = torch.ones(1, 1, 1, device="cuda", dtype=torch.float32)
    out_dy2 = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    out_dy1 = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    out_dp = torch.zeros(1, _S, 1, device="cuda", dtype=torch.float32)
    out_db1 = torch.zeros(_G, 1, _N, device="cuda", dtype=torch.float32)
    out_db2 = torch.zeros(_G, 1, _N, device="cuda", dtype=torch.float32)
    bd = compiled.binding
    compiled(
        {
            bd.a_operands[0]: token,
            bd.b_operands[0]: weight,
            bd.first_token_offset: offs,
            bd.outputs[0]: out_dy2,
            bd.outputs[1]: out_dy1,
            bd.outputs[2]: out_dp,
            bd.outputs[3]: out_db1,
            bd.outputs[4]: out_db2,
            bd.aux[0]: alpha_t,
            bd.aux[1]: prob_t,
            bd.aux[2]: y1_t,
            bd.aux[3]: one_t,
            bd.aux[4]: y2_t,
        }
    )
    torch.cuda.synchronize()

    acc_ref = _ref_c(token, weight)
    ga_ref = acc_ref.clone()
    for gi, (b, e) in enumerate(_group_ranges(_OFFSETS)):
        ga_ref[b:e] *= alpha_t[gi, 0, 0]
    gg_ref = ga_ref * prob_t[0]
    sig_ref = torch.sigmoid(y1_t[0])
    silu_ref = y1_t[0] * sig_ref
    dy2_ref = gg_ref * silu_ref
    dy1_ref = gg_ref * sig_ref * (1.0 + y1_t[0] * (1.0 - sig_ref)) * y2_t[0]
    dp_ref = (silu_ref * y2_t[0] * ga_ref).sum(dim=1, keepdim=True)
    torch.testing.assert_close(out_dy2[0].float(), dy2_ref.to(torch.bfloat16).float(), atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(out_dy1[0].float(), dy1_ref.to(torch.bfloat16).float(), atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(out_dp[0], dp_ref, atol=5e-1, rtol=5e-2)
    db1_ref = torch.zeros(_G, 1, _N, device="cuda", dtype=torch.float32)
    db2_ref = torch.zeros(_G, 1, _N, device="cuda", dtype=torch.float32)
    for gi, (b, e) in enumerate(_group_ranges(_OFFSETS)):
        if b < e:
            db1_ref[gi, 0] = dy1_ref.to(torch.bfloat16).float()[b:e].sum(dim=0)
            db2_ref[gi, 0] = dy2_ref.to(torch.bfloat16).float()[b:e].sum(dim=0)
    torch.testing.assert_close(out_db1, db1_ref, atol=5e-1, rtol=5e-2)
    torch.testing.assert_close(out_db2, db2_ref, atol=5e-1, rtol=5e-2)


def test_single_moe_dgeglu_backward():
    """cutedsl dglu act_func="dgeglu" under the split layout, clamp masks via
    cmp ops; reference = torch.autograd through the verified GeGLU forward."""
    cmax_v, cmin_v, alpha_g, off_v = 2.0, -2.0, 1.702, 1.0
    g, acc, fto = _graph()
    y1 = g.tensor(name="y1", dim=[1, _S, _N], stride=[_S * _N, _N, 1], data_type=cudnn.data_type.FLOAT)
    y2 = g.tensor(name="y2", dim=[1, _S, _N], stride=[_S * _N, _N, 1], data_type=cudnn.data_type.FLOAT)
    alpha = g.tensor(name="alpha", dim=[_G, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    prob = g.tensor(name="prob", dim=[1, _S, 1], stride=[_S, 1, 1], data_type=cudnn.data_type.FLOAT)
    cmax = g.tensor(name="cmax", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    cmin = g.tensor(name="cmin", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    ag = g.tensor(name="ag", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    off = g.tensor(name="off", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    one = g.tensor(name="one", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)

    ga = g.mul(a=acc, b=alpha, name="ga")
    gg = g.mul(a=ga, b=prob, name="gg")
    y1c = g.min(input0=y1, input1=cmax, name="y1c")
    u = g.mul(a=y1c, b=ag, name="u")
    sig = g.sigmoid(input=u, name="sig")
    silu_u = g.mul(a=sig, b=u, name="silu_u")
    m2 = g.mul(a=g.cmp_ge(input=y2, comparison=cmin, name="m2a"), b=g.cmp_le(input=y2, comparison=cmax, name="m2b"), name="m2")
    dy2 = g.mul(a=g.mul(a=gg, b=silu_u, name="dy2t"), b=m2, name="dy2")
    dy2.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    y2c = g.max(input0=g.min(input0=y2, input1=cmax, name="y2c1"), input1=cmin, name="y2c")
    w = g.add(a=y2c, b=off, name="w")
    t1 = g.add(a=g.mul(a=u, b=g.sub(a=one, b=sig, name="oms"), name="t"), b=one, name="t1")
    dsu = g.mul(a=sig, b=t1, name="dsu")
    m1 = g.cmp_le(input=y1, comparison=cmax, name="m1")
    dy1 = g.mul(a=g.mul(a=g.mul(a=g.mul(a=gg, b=dsu, name="d1a"), b=w, name="d1b"), b=ag, name="d1c"), b=m1, name="dy1")
    dy1.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)

    token, weight, offs = _data()
    y1_t = torch.randn(1, _S, _N, device="cuda", dtype=torch.float32) * 3.0
    y2_t = torch.randn(1, _S, _N, device="cuda", dtype=torch.float32) * 3.0
    alpha_t = torch.rand(_G, 1, 1, device="cuda", dtype=torch.float32) + 0.5
    prob_t = torch.rand(1, _S, 1, device="cuda", dtype=torch.float32)
    scalars = {"cmax": cmax_v, "cmin": cmin_v, "ag": alpha_g, "off": off_v, "one": 1.0}
    out_dy2 = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    out_dy1 = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    bd = compiled.binding
    name_to_buf = {
        "alpha": alpha_t,
        "prob": prob_t,
        "y1": y1_t,
        "y2": y2_t,
        **{k: torch.full((1, 1, 1), v, device="cuda", dtype=torch.float32) for k, v in scalars.items()},
    }
    vp = {bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out_dy2, bd.outputs[1]: out_dy1}
    vp.update({t: name_to_buf[ref.name] for t, ref in zip(bd.aux, compiled.chain.aux_tensors)})
    compiled(vp)
    torch.cuda.synchronize()

    acc_ref = _ref_c(token, weight)
    gg_ref = acc_ref.clone()
    for gi, (b, e) in enumerate(_group_ranges(_OFFSETS)):
        gg_ref[b:e] *= alpha_t[gi, 0, 0]
    gg_ref = gg_ref * prob_t[0]
    y1_a = y1_t[0].detach().requires_grad_(True)
    y2_a = y2_t[0].detach().requires_grad_(True)
    fwd = (torch.clamp(y2_a, cmin_v, cmax_v) + off_v) * torch.nn.functional.silu(alpha_g * torch.clamp(y1_a, max=cmax_v))
    dy1_ref, dy2_ref = torch.autograd.grad(fwd, (y1_a, y2_a), grad_outputs=gg_ref)
    bad1 = (~torch.isclose(out_dy1[0].float(), dy1_ref.to(torch.bfloat16).float(), atol=8e-2, rtol=8e-2)).float().mean().item()
    bad2 = (~torch.isclose(out_dy2[0].float(), dy2_ref.to(torch.bfloat16).float(), atol=8e-2, rtol=8e-2)).float().mean().item()
    assert bad1 < 2e-3, bad1
    assert bad2 < 2e-3, bad2


def test_single_moe_grouped_avg_reduction():
    """Grouped AVG: the per-group divisor is the runtime group row count."""
    g, c, fto = _graph()
    sw = g.swish(input=c, name="sw")
    sw.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    red = g.reduction(input=sw, mode=cudnn.reduction_mode.AVG, name="avg", group_offset=fto)
    red.set_dim([_G, 1, _N]).set_stride([_N, _N, 1])
    red.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    compiled = jit_from_cudnn_graph(g, by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1"), cta_group=2)

    token, weight, offs = _data()
    out = torch.empty(1, _S, _N, device="cuda", dtype=torch.bfloat16)
    r = torch.zeros(_G, 1, _N, device="cuda", dtype=torch.float32)
    bd = compiled.binding
    compiled({bd.a_operands[0]: token, bd.b_operands[0]: weight, bd.first_token_offset: offs, bd.outputs[0]: out, bd.outputs[1]: r})
    torch.cuda.synchronize()

    csw = _ref_c(token, weight)
    csw = (csw * torch.sigmoid(csw)).to(torch.bfloat16).float()
    ref = torch.zeros(_G, 1, _N, device="cuda", dtype=torch.float32)
    for gi, (b, e) in enumerate(_group_ranges(_OFFSETS)):
        if b < e:
            ref[gi, 0] = csw[b:e].mean(dim=0)
    torch.testing.assert_close(r, ref, atol=5e-2, rtol=2e-2)


def test_moe_tma_arm_bounds_extra_outputs_by_the_routed_group() -> None:
    """The TMA arm carries no row bound of its own, so an extra dense output's
    store has to reapply the one its STG sibling uses. On MoE that is the routed
    group's end, NOT the problem M — `row < M` writes rows belonging to the next
    group and the result is wrong, not merely unclipped."""
    g, c, _ = _graph()
    r = g.relu(input=c, name="r")
    r.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    g.gelu_approx_tanh(input=r, name="ge").set_output(True)
    chain = analyze(g)
    assert chain.has_moe and len(chain.output_specs) == 2

    tma = generate(chain, tma_slots=frozenset({0})).epilogue
    stg = generate(chain, tma_slots=frozenset()).epilogue

    assert "row < group_end" in tma
    assert "row < M" not in tma
    # The STG arm is enclosed by the template's own guard and must stay bare.
    assert "row < group_end" not in stg and "row < M" not in stg


def test_per_group_per_col_aux_group_stride_need_not_divide_the_chunk() -> None:
    """A per-group per-col aux loads at `group_idx * stride[0] + col_j`, and the
    group stride used to have to be a whole number of epilogue chunks. It does
    not: `ALIGN_AUX_<name>` is `min(the aux's OWN layout alignment, the chunk)`,
    and `tensor_alignment` already folds the group stride in — so the alignment
    promise degrades on its own. The old check predated that and only bit once
    the TMA arm widened the chunk from the STG width to `epi_n`."""
    from cudnn.gemm.frost.compiler import _use_tma_store_epi
    from cudnn.gemm.frost.dtypes import tensor_alignment

    # 248 * 4 bytes = 992 -> a 32-byte alignment, and 248 % 32 == 24.
    assert tensor_alignment((_G, 1, 248), (248, 248, 1), 4) == 32
    assert 248 % 32 != 0

    g, c, _ = _graph()
    bias = g.tensor(name="bias", dim=[_G, 1, _N], stride=[_N, _N, 1], data_type=cudnn.data_type.FLOAT)
    g.add(a=c, b=bias, name="bi").set_output(True)
    chain = analyze(g)
    aux = chain.aux_tensors[0]
    assert aux.grouped_by_moe and aux.bcast_mode == "per_col"

    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")
    # `_N` is 256 here, so assert on the rule's shape rather than on this N:
    # nothing in the gate may consult a per-group aux's stride.
    assert _use_tma_store_epi(chain, cfg, 1) is True
    src = inspect.getsource(compiler._output_store_mode)
    for probe in ("grouped_by_moe", "bcast_mode", "aux_tensors"):
        assert probe not in src, f"{probe}: an aux load's alignment is a pointwise concern, not a store rule"
