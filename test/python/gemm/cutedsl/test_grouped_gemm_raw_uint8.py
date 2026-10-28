# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw uint8 FP8 operands with dtype metadata give the same results as typed operands."""

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_glu_canonical import assert_outputs_equal, natural_inputs
from gemm.cutedsl.test_grouped_gemm_wrapper_memo import glu_block_scaled_call, mxfp8_inputs

pytestmark = pytest.mark.L0
FP8 = dict(a_dtype=torch.float8_e4m3fn, b_dtype=torch.float8_e4m3fn)


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")


def raw(operands, names=("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor")):
    result = dict(operands)
    for name in names:
        result[name] = operands[name].view(torch.uint8)
    return result


@pytest.mark.parametrize("flat", [False, True])
def test_glu_raw_uint8_matches_typed(flat):
    from cudnn.gemm.cutedsl.grouped.glu import api

    natural = natural_inputs(mxfp8_inputs([512] * 4), flat)
    typed = glu_block_scaled_call(natural, use_dynamic_sched=True)
    for _ in range(2):
        assert_outputs_equal(glu_block_scaled_call(raw(natural), use_dynamic_sched=True, **FP8), typed, 2048)
    raw_apis = [op for op in api._cache_of_GroupedGemmGluSm100Objects.values() if getattr(op._implementation, "raw_dtypes", None)]
    assert raw_apis and set(raw_apis[-1]._implementation.raw_dtypes) == {"a", "b", "sfa", "sfb"}


def test_glu_raw_uint8_rejects_non_fp8_dtype():
    natural = natural_inputs(mxfp8_inputs([512] * 4))
    with pytest.raises(ValueError, match="raw uint8 data must hold"):
        glu_block_scaled_call(raw(natural), a_dtype=torch.bfloat16, b_dtype=torch.float8_e4m3fn)


def raw_wrapper(monkeypatch, name, operands, **dtypes):
    """Route calls to ``cudnn.<name>`` through raw uint8 views of ``operands`` plus the dtype metadata."""
    import cudnn

    original = getattr(cudnn, name)

    def call(**kwargs):
        for key in operands:
            kwargs[key] = kwargs[key].view(torch.uint8)
        return original(**kwargs, **dtypes)

    monkeypatch.setattr(cudnn, name, call)


def assert_same_outputs(actual, expected, names):
    for name in names:
        left, right = actual[name], expected[name]
        assert left.shape == right.shape and left.dtype == right.dtype, name
        assert torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)), name


@pytest.mark.parametrize("b_major", ["k", "n"])
@pytest.mark.parametrize("d_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_quant_raw_uint8_matches_typed(b_major, d_dtype):
    from cudnn import grouped_gemm_quant_wrapper_sm100

    inputs = mxfp8_inputs([512] * 4)
    natural = natural_inputs(inputs)
    b = natural["b_tensor"] if b_major == "k" else inputs["b_tensor"].permute(2, 1, 0).contiguous()

    def call(a, sfa, b, sfb, **dtypes):
        return grouped_gemm_quant_wrapper_sm100(
            a_tensor=a,
            sfa_tensor=sfa,
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            b_tensor=b,
            sfb_tensor=sfb,
            b_major=b_major,
            norm_const_tensor=inputs["norm_const_tensor"],
            d_dtype=d_dtype,
            sf_vec_size=32,
            use_dynamic_sched=True,
            **dtypes,
        )

    typed = call(natural["a_tensor"], natural["sfa_tensor"], b, natural["sfb_tensor"])
    names = ["d_tensor", "amax_tensor"] if d_dtype == torch.bfloat16 else ["d_tensor", "d_col_tensor", "sfd_row_tensor", "sfd_col_tensor"]
    for _ in range(2):
        out = call(*(t.view(torch.uint8) for t in (natural["a_tensor"], natural["sfa_tensor"], b, natural["sfb_tensor"])), **FP8)
        assert_same_outputs(out, typed, names)


@pytest.mark.parametrize("b_major", ["k", "n"])
@pytest.mark.parametrize("flat_sf", [False, True], ids=["physical_sf", "flat_sf"])
def test_dglu_raw_uint8_matches_typed(request, monkeypatch, b_major, flat_sf):
    from gemm.cutedsl.test_grouped_gemm_backward_canonical import run_dglu
    from gemm.cutedsl.test_grouped_gemm_canonical_layouts import dswiglu_case

    cfg, inputs = dswiglu_case(request)
    typed = run_dglu(cfg, inputs, canonical=True, b_major=b_major, flat_sf=flat_sf, deterministic=True)
    raw_wrapper(monkeypatch, "grouped_gemm_dglu_wrapper_sm100", ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor"), **FP8)
    for _ in range(2):
        out = run_dglu(cfg, inputs, canonical=True, b_major=b_major, flat_sf=flat_sf, deterministic=True)
        assert_same_outputs(out, typed, ["d_row_tensor", "d_col_tensor", "sfd_row_tensor", "sfd_col_tensor", "dprob_tensor"])


def test_wgrad_raw_uint8_matches_typed(monkeypatch):
    from gemm.cutedsl.test_grouped_gemm_backward_canonical import wgrad_call, wgrad_inputs

    inputs = wgrad_inputs([256, 384])
    sfa, sfb = inputs["sfa_tensor"].reshape(-1), inputs["sfb_tensor"].reshape(-1)
    typed = wgrad_call(inputs, sfa, sfb)
    raw_wrapper(monkeypatch, "grouped_gemm_wgrad_wrapper_sm100", ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor"), **FP8)
    for _ in range(2):
        assert_same_outputs(wgrad_call(inputs, sfa, sfb), typed, ["wgrad_tensor"])
