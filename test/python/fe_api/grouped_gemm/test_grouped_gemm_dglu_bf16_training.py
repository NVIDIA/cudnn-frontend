# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialized BF16 dactivation boundary; diagnostic shapes, no timing."""

import pytest
import torch

from fe_api.grouped_gemm._workspace import ws

import cudnn
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
from fe_api.test_grouped_gemm_dglu_bf16_utils import make_grouped_gemm_dglu_bf16_problem

pytestmark = pytest.mark.L0


def problem(discrete):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10 or cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("requires Blackwell and CuTe DSL>=4.7.0")
    p = make_grouped_gemm_dglu_bf16_problem(discrete=discrete, alpha_values=(1.0, 1.0), beta_values=(1.0, 1.0))
    p["a"].zero_()
    p["a"][:, :2] = 1
    p["b_reference"].zero_()
    p["b_reference"][:, :, 0] = 1
    p["b_reference"][:, :, 1] = 2**-8
    pair = p["c"].view(512, 4, 2, 32)
    pair[:, :, 0] = 1
    pair[:, :, 1] = 1.5
    p["prob"].fill_(1)
    p["d"] = torch.empty_like(p["c"])
    p["decl"] = dict(num_experts=2, b_shape=(128, 128), b_dtype=torch.bfloat16) if discrete else dict(sample_b=p["b"])
    p["bind"] = dict(b_ptrs=p["b_ptrs"]) if discrete else dict(b_tensor=p["b"])
    return p


def reference(p, rounded):
    # Exact two-term products avoid making a TF32 GEMM our rounding oracle.
    da = p["a"][:, 0, 0].float() + p["a"][:, 1, 0].float() * 2**-8
    if rounded:
        da = da.bfloat16().float()
    gate = p["c"][:, 0, 0].float()
    up = p["c"][:, 32, 0].float()
    sig = gate.sigmoid()
    dg = da * p["prob"].view(512) * up * sig * (1 + gate * (1 - sig))
    du = da * p["prob"].view(512) * gate * sig
    d = torch.empty_like(p["c"])
    pair = d.view(512, 4, 2, 32)
    pair[:, :, 0] = dg[:, None, None]
    pair[:, :, 1] = du[:, None, None]
    return d, (da * gate * sig * up * 128).view(512, 1, 1)


def check(p, d, rounded):
    ref_d, ref_dp = reference(p, rounded)
    torch.testing.assert_close(d, ref_d, rtol=0, atol=0)
    torch.testing.assert_close(p["dprob"], ref_dp, rtol=2e-6, atol=0)
    wrong_d, wrong_dp = reference(p, not rounded)
    for actual, wrong in ((d, wrong_d), (p["dprob"], wrong_dp)):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(actual, wrong, rtol=2e-6, atol=0)


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("vector_f32", [False, True])
@pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"])
def test_dglu_bf16_dactivation_rounding(discrete, vector_f32, act_func):
    p = problem(discrete)
    plan = cudnn.GroupedGemmDgluSm100(
        p["a"],
        p["c"],
        p["d"],
        None,
        None,
        p["offsets"],
        p["alpha"],
        p["beta"],
        p["prob"],
        p["dprob"],
        **p["decl"],
        act_func=act_func,
        vector_f32=vector_f32,
        geglu_alpha=1.0,
        glu_clamp_max=10.0,
        glu_clamp_min=-10.0,
        linear_offset=0.0,
        round_dgrad_to_input_dtype=True,
    )
    plan.compile()
    workspace = ws(plan)

    def run():
        p["dprob"].zero_()  # Required atomic accumulator reset, part of the graph.
        plan.execute(p["a"], p["c"], p["d"], None, None, p["offsets"], p["alpha"], p["beta"], p["prob"], p["dprob"], **p["bind"], workspace=workspace)

    run()  # Prepare pointer metadata before capture in discrete mode.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    for scale in (1.0, -1.0):
        p["a"][:, :2] = scale
        p["prob"][::17] = 0
        p["d"].fill_(float("nan"))
        p["dprob"].fill_(float("nan"))
        graph.replay()
        check(p, p["d"], True)


def test_dglu_wrapper_rounding_compile_identity():
    p = problem(False)
    # Repeated identical operands exercise both wrapper memo and compiled-plan
    # cache, including the omitted flag's legacy FP32-accumulator default.
    for rounded in (None, True, False, True, None):
        p["dprob"].fill_(float("nan"))
        p["dprob"].zero_()
        out = cudnn.grouped_gemm_dglu_wrapper_sm100(
            p["a"],
            p["c"],
            None,
            p["offsets"],
            p["alpha"],
            p["beta"],
            p["prob"],
            p["dprob"],
            b_tensor=p["b"],
            act_func="dgeglu",
            geglu_alpha=1.0,
            glu_clamp_max=10.0,
            glu_clamp_min=-10.0,
            linear_offset=0.0,
            **({} if rounded is None else dict(round_dgrad_to_input_dtype=rounded)),
        )
        check(p, out["d_row_tensor"], bool(rounded))


@pytest.mark.parametrize("wrapper", [False, True])
def test_dglu_rounding_rejects_blockscaled(wrapper):
    p = problem(False)
    a, b = p["a"].to(torch.float8_e4m3fn), p["b"].to(torch.float8_e4m3fn)
    # FP8 operands select the block-scaled backend. Decline this unsupported
    # arithmetic option before compiling or binding scale-factor storage.
    with pytest.raises(ValueError, match="round_dgrad_to_input_dtype is supported only by the BF16 kernel"):
        if wrapper:
            cudnn.grouped_gemm_dglu_wrapper_sm100(
                a,
                p["c"],
                None,
                p["offsets"],
                p["alpha"],
                p["beta"],
                p["prob"],
                p["dprob"],
                b_tensor=b,
                round_dgrad_to_input_dtype=True,
            )
        else:
            cudnn.GroupedGemmDgluSm100(
                a,
                p["c"],
                p["d"],
                None,
                None,
                p["offsets"],
                p["alpha"],
                p["beta"],
                p["prob"],
                p["dprob"],
                sample_b=b,
                round_dgrad_to_input_dtype=True,
            ).check_support()
