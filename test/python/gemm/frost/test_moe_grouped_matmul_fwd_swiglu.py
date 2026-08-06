# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused dual MoE grouped matmul + SwiGLU: two grouped matmuls sharing token (A)
and first_token_offset feed one pointwise epilogue DAG (multi-GEMM, 1/2ctamma)."""

from __future__ import annotations

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import pytest
import torch

from gemm_test_utils import (
    requires_sm100,
    Plan as _plan,
    vp_mg as _vp_mg,
    FULL_EXPERT_REDUCE_OFFSETS as _FULL_EXPERT_REDUCE_OFFSETS,
)

from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0


# (config name, cta_group): 1-CTA cluster1x1 + 2-CTA cluster2x1 (reference design)
_GEOMETRIES = [
    ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1", 1),
    ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1", 2),
]


def _build_graph(
    E,
    S,
    N,
    K,
    num_groups,
    out_dt=cudnn.data_type.BFLOAT16,
    reduction_mode=None,
    reduction_dims=None,
):
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
    fto = g.tensor(
        name="first_token_offset",
        dim=[num_groups, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    sf = g.tensor(
        name="scaleFactor",
        dim=[1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
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
    c0silu = g.swish(input=c0, name="silu0")
    mul = g.mul(a=c0silu, b=c1, name="mul0")
    dq = g.mul(a=mul, b=sf, name="dequant0")
    dq.set_data_type(out_dt).set_output(True)
    if reduction_mode is not None:
        assert reduction_dims is not None
        R = g.reduction(input=dq, mode=reduction_mode, name="red")
        R.set_dim(list(reduction_dims)).set_stride([reduction_dims[1] * reduction_dims[2], reduction_dims[2], 1])
        R.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return g


def _ref_f32(token, w0, w1, offsets, scale, S, N, num_experts, num_groups):
    out = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    starts = offsets.tolist()
    for gi in range(num_groups):
        b = starts[gi]
        e = starts[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        ex = gi % num_experts
        c0 = token[0, b:e].float() @ w0[ex].float().T
        c1 = token[0, b:e].float() @ w1[ex].float().T
        out[b:e] = torch.nn.functional.silu(c0) * c1 * scale.flatten()[0]
    return out


def _ref(token, w0, w1, offsets, scale, S, N, num_experts, num_groups):
    out = _ref_f32(token, w0, w1, offsets, scale, S, N, num_experts, num_groups)
    return out.to(torch.bfloat16)


# --- Analyzer (no GPU needed) ---


def test_analyzer_detects_dual_moe_grouped_matmul_fwd() -> None:
    chain = analyze(_build_graph(9, 2000, 248, 520, 36))
    assert chain.has_moe and chain.is_multi_gemm
    assert chain.num_gemms == 2
    assert chain.num_a_operands == 1 and chain.num_b_operands == 2
    assert chain.gemm_operands == [(0, 0), (0, 1)]
    assert chain.moe.num_experts == 9
    assert (chain.matmul.M, chain.matmul.N, chain.matmul.K) == (2000, 248, 520)
    assert [o.op for o in chain.ops] == ["swish", "mul", "mul"]
    assert len(chain.outputs) == 1 and chain.outputs[0].source == "op_2"


def test_analyzer_detects_dual_moe_grouped_matmul_fwd_reduction() -> None:
    chain = analyze(
        _build_graph(
            9,
            2000,
            248,
            520,
            36,
            reduction_mode=cudnn.reduction_mode.ADD,
            reduction_dims=(1, 1, 1),
        )
    )
    assert chain.has_moe and chain.is_multi_gemm
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "add"
    assert [o.source for o in chain.outputs] == ["op_2", "reduction_0"]


# --- End-to-end correctness (GPU) ---


@requires_sm100
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
def test_dual_moe_grouped_matmul_fwd_swiglu_exact_case(cfg_name, cta_group) -> None:
    """Spec case: S=2000, N=248, K=520, E=9, 36 routed groups (BxE > E)."""
    S, N, K, E = 2000, 248, 520, 9
    offset_values = _FULL_EXPERT_REDUCE_OFFSETS
    num_groups = len(offset_values)
    cfg = by_name(cfg_name)
    compiled = _plan(_build_graph(E, S, N, K, num_groups), config=cfg, cta_group=cta_group)

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    w0 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device="cuda")
    out = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(offset_values, dtype=torch.int32, device="cuda")

    compiled(_vp_mg(compiled, [(token, w0), (token, w1)], out, scale, fto=offsets))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out[0],
        _ref(token, w0, w1, offsets, scale, S, N, E, num_groups),
        atol=5e-2,
        rtol=5e-2,
    )


@requires_sm100
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
@pytest.mark.parametrize(
    "group_sizes",
    [
        [64, 0, 200, 128, 100, 12, 196, 68],  # uneven + one empty group
        [96, 96, 96, 96, 96, 96, 96, 96],
    ],
)
def test_dual_moe_grouped_matmul_fwd_swiglu_groups(group_sizes, cfg_name, cta_group) -> None:
    E, N, K = 8, 256, 128
    S = sum(group_sizes)
    num_groups = E
    cfg = by_name(cfg_name)
    compiled = _plan(_build_graph(E, S, N, K, num_groups), config=cfg, cta_group=cta_group)

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    w0 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device="cuda")
    out = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    starts, cur = [], 0
    for gs in group_sizes:
        starts.append(cur)
        cur += gs
    offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")

    compiled(_vp_mg(compiled, [(token, w0), (token, w1)], out, scale, fto=offsets))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out[0],
        _ref(token, w0, w1, offsets, scale, S, N, E, num_groups),
        atol=2e-1,
        rtol=5e-2,
    )


@requires_sm100
def test_dual_moe_grouped_matmul_fwd_swiglu_reduction_scalar() -> None:
    E, N, K = 4, 128, 128
    group_sizes = [64, 0, 120, 72]
    S = sum(group_sizes)
    num_groups = E
    cfg = by_name(_GEOMETRIES[0][0])
    compiled = _plan(
        _build_graph(
            E,
            S,
            N,
            K,
            num_groups,
            reduction_mode=cudnn.reduction_mode.ADD,
            reduction_dims=(1, 1, 1),
        ),
        config=cfg,
        cta_group=_GEOMETRIES[0][1],
    )

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    w0 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([[[0.5]]], dtype=torch.float32, device="cuda")
    out = torch.empty(1, S, N, dtype=torch.bfloat16, device="cuda")
    red = torch.empty(1, 1, 1, dtype=torch.float32, device="cuda")
    starts, cur = [], 0
    for gs in group_sizes:
        starts.append(cur)
        cur += gs
    offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")

    compiled(_vp_mg(compiled, [(token, w0), (token, w1)], [out, red], scale, fto=offsets))
    torch.cuda.synchronize()

    ref = _ref_f32(token, w0, w1, offsets, scale, S, N, E, num_groups)
    torch.testing.assert_close(out[0], ref.to(torch.bfloat16), atol=2e-1, rtol=5e-2)
    torch.testing.assert_close(
        red,
        ref.view(1, S, N).sum(dim=(1, 2), keepdim=True),
        atol=1e-1,
        rtol=1e-2,
    )


# --- GeGLU (the cutedsl glu family's act_func="geglu", composed on the DAG) --


def _build_geglu_graph(E, S, N, K, num_groups):
    """GeGLU per the cutedsl glu kernel, as a pointwise DAG on dual MoE GEMMs:
    out = (clamp(up, cmin, cmax) + linear_offset)
          * silu(geglu_alpha * clamp(gate, max=cmax))
    with per-group per-col bias folded into each GEMM branch."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
    w0 = g.tensor(name="weight0", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    w1 = g.tensor(name="weight1", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    fto = g.tensor(name="first_token_offset", dim=[num_groups, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    bias0 = g.tensor(name="bias0", dim=[num_groups, 1, N], stride=[N, N, 1], data_type=cudnn.data_type.FLOAT)
    bias1 = g.tensor(name="bias1", dim=[num_groups, 1, N], stride=[N, N, 1], data_type=cudnn.data_type.FLOAT)
    cmax = g.tensor(name="cmax", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    cmin = g.tensor(name="cmin", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    alpha = g.tensor(name="geglu_alpha", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    loff = g.tensor(name="linear_offset", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    gate = g.moe_grouped_matmul(tok, w0, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe_gate")
    up = g.moe_grouped_matmul(tok, w1, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe_up")
    gate_b = g.add(a=gate, b=bias0, name="bias_gate")
    up_b = g.add(a=up, b=bias1, name="bias_up")
    gate_c = g.min(input0=gate_b, input1=cmax, name="clamp_gate")  # one-sided clamp
    gate_a = g.mul(a=gate_c, b=alpha, name="alpha_gate")
    s = g.swish(input=gate_a, name="silu")
    up_hi = g.min(input0=up_b, input1=cmax, name="clamp_up_hi")
    up_c = g.max(input0=up_hi, input1=cmin, name="clamp_up_lo")
    u = g.add(a=up_c, b=loff, name="offset_up")
    y = g.mul(a=s, b=u, name="geglu")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    return g


def _geglu_ref(token, w0, w1, bias0, bias1, offsets, S, N, E, num_groups, cmax=7.0, cmin=-7.0, alpha=1.702, loff=1.0):
    out = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    starts = offsets.tolist()
    for gi in range(num_groups):
        b = starts[gi]
        e = starts[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        ex = gi % E
        gate = token[0, b:e].float() @ w0[ex].float().T + bias0[gi, 0]
        up = token[0, b:e].float() @ w1[ex].float().T + bias1[gi, 0]
        s = torch.nn.functional.silu(alpha * torch.clamp(gate, max=cmax))
        out[b:e] = s * (torch.clamp(up, cmin, cmax) + loff)
    return out


def test_analyzer_detects_dual_moe_geglu() -> None:
    chain = analyze(_build_geglu_graph(9, 2000, 248, 520, 36))
    assert chain.has_moe and chain.is_multi_gemm and chain.num_gemms == 2
    assert [o.op for o in chain.ops] == ["add", "add", "min", "mul", "swish", "min", "max", "add", "mul"]
    assert [a.name for a in chain.aux_tensors] == ["bias0", "bias1", "cmax", "geglu_alpha", "cmin", "linear_offset"]
    assert [a.bcast_mode for a in chain.aux_tensors] == ["per_col", "per_col", "scalar", "scalar", "scalar", "scalar"]
    assert [a.grouped_by_moe for a in chain.aux_tensors] == [True, True, False, False, False, False]


@requires_sm100
@pytest.mark.parametrize("cfg_name,cta_group", _GEOMETRIES)
def test_dual_moe_geglu(cfg_name, cta_group) -> None:
    S, N, K, E = 2000, 248, 520, 9
    offset_values = _FULL_EXPERT_REDUCE_OFFSETS
    num_groups = len(offset_values)
    compiled = _plan(_build_geglu_graph(E, S, N, K, num_groups), config=by_name(cfg_name), cta_group=cta_group)

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    # std(gate/up) ~ sqrt(K)*0.3 ~ 6.8 => the +/-7 clamps are exercised on
    # roughly half the elements without fully saturating the test.
    w0 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.3
    w1 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.3
    bias0 = torch.randn(num_groups, 1, N, dtype=torch.float32, device="cuda")
    bias1 = torch.randn(num_groups, 1, N, dtype=torch.float32, device="cuda")
    scalars = [torch.full((1, 1, 1), v, dtype=torch.float32, device="cuda") for v in (7.0, -7.0, 1.702, 1.0)]
    cmax_t, cmin_t, alpha_t, loff_t = scalars
    out = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(offset_values, dtype=torch.int32, device="cuda")

    compiled(_vp_mg(compiled, [(token, w0), (token, w1)], out, bias0, bias1, cmax_t, alpha_t, cmin_t, loff_t, fto=offsets))
    torch.cuda.synchronize()

    ref = _geglu_ref(token, w0, w1, bias0, bias1, offsets, S, N, E, num_groups)
    torch.testing.assert_close(out[0].float(), ref.to(torch.bfloat16).float(), atol=8e-2, rtol=8e-2)


@requires_sm100
def test_dual_moe_distinct_tokens_mixed_strides() -> None:
    """Two DISTINCT token (A) operands with DIFFERENT row strides: tokA compact,
    tokB a padded view (stride_m = K + pad). Exercises per-operand a_stride in
    the host descriptor build AND the per-group kernel descriptor patch."""
    S, N, K, E = 512, 128, 256, 4
    pad = 64
    offset_values = (0, 128, 200, 384)
    num_groups = len(offset_values)
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tokA = g.tensor(name="tokA", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
    tokB = g.tensor(name="tokB", dim=[1, S, K], stride=[S * (K + pad), K + pad, 1], data_type=cudnn.data_type.BFLOAT16)
    w0 = g.tensor(name="w0", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    w1 = g.tensor(name="w1", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    fto = g.tensor(name="fto", dim=[num_groups, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    c0 = g.moe_grouped_matmul(tokA, w0, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe0")
    c1 = g.moe_grouped_matmul(tokB, w1, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe1")
    y = g.mul(a=g.swish(input=c0, name="silu"), b=c1, name="mul")
    y.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    compiled = _plan(g, config=by_name("CONFIG_sm100_128x128x128_128x128x32_cluster2x1"), cta_group=2)
    assert compiled.chain.num_a_operands == 2

    torch.manual_seed(0)
    tokA_t = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    tokB_pad = torch.randn(1, S, K + pad, dtype=torch.bfloat16, device="cuda")
    tokB_t = tokB_pad[:, :, :K]
    assert tokB_t.stride(1) == K + pad
    w0_t = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    w1_t = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    out = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(offset_values, dtype=torch.int32, device="cuda")

    compiled(_vp_mg(compiled, [(tokA_t, w0_t), (tokB_t, w1_t)], out, fto=offsets))
    torch.cuda.synchronize()

    ref = torch.zeros(S, N, dtype=torch.float32, device="cuda")
    starts = list(offset_values)
    for gi in range(num_groups):
        b, e = starts[gi], (starts[gi + 1] if gi + 1 < num_groups else S)
        if b < e:
            ca = tokA_t[0, b:e].float() @ w0_t[gi % E].float().T
            cb = tokB_t[0, b:e].float() @ w1_t[gi % E].float().T
            ref[b:e] = torch.nn.functional.silu(ca) * cb
    torch.testing.assert_close(out[0], ref.to(torch.bfloat16), atol=5e-2, rtol=5e-2)
