# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_wrapper_memo import mxfp8_inputs, glu_block_scaled_call, raw_bytes

pytestmark = pytest.mark.L0
SF_PHYSICAL = (5, 2, 4, 0, 1, 3)


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")


def natural_inputs(inputs, flat=True):
    result = dict(inputs)
    result["a_tensor"] = inputs["a_tensor"].squeeze(-1)
    result["b_tensor"] = inputs["b_tensor"].permute(2, 0, 1)
    result["prob_tensor"] = inputs["prob_tensor"].view(-1)
    for name in ("sfa_tensor", "sfb_tensor"):
        result[name] = inputs[name].permute(SF_PHYSICAL)
        if flat:
            result[name] = result[name].view(-1)
    return result


def assert_outputs_match(natural, legacy, valid_m):
    for name in ("c_tensor", "d_tensor", "d_col_tensor", "sfd_row_tensor", "sfd_col_tensor", "amax_tensor"):
        left, right = natural[name], legacy[name]
        if left is None:
            assert right is None
            continue
        if name.startswith("sfd"):
            right = right.permute(SF_PHYSICAL)
            if name == "sfd_row_tensor":
                left, right = left[:, : valid_m // 128], right[:, : valid_m // 128]
            else:
                left, right = left[:, :, : valid_m // 128], right[:, :, : valid_m // 128]
        elif name != "amax_tensor":
            left, right = left[:valid_m], right[:valid_m]
        assert torch.equal(raw_bytes(left), raw_bytes(right)), name


@pytest.mark.parametrize("dynamic", ["0", "1"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "geglu", "situglu"])
def test_glu_canonical_matches_legacy(monkeypatch, dynamic, flat, activation):
    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", dynamic)
    inputs = mxfp8_inputs([512] * 4)
    natural = natural_inputs(inputs, flat)
    options = dict(act_func=activation, use_dynamic_sched=True)
    legacy = glu_block_scaled_call(inputs, **options)
    cold = glu_block_scaled_call(natural, **options)
    warm = glu_block_scaled_call(natural, **options)
    assert cold["c_tensor"].shape == (2048, 512)
    assert cold["d_tensor"].shape == (2048, 256)
    assert cold["sfd_row_tensor"].is_contiguous()
    assert_outputs_match(cold, legacy, 2048)
    assert_outputs_match(warm, legacy, 2048)


def test_glu_canonical_memo_changed_data_and_routing():
    from cudnn.gemm.cutedsl.grouped.glu.api import _glu_wrapper_memo

    inputs = mxfp8_inputs([256] * 4)
    glu_block_scaled_call(natural_inputs(inputs))
    changed = mxfp8_inputs([512, 256, 512, 256])
    warm = glu_block_scaled_call(natural_inputs(changed))
    _glu_wrapper_memo.clear()
    cold = glu_block_scaled_call(natural_inputs(changed))
    legacy = glu_block_scaled_call(changed)
    assert_outputs_match(warm, legacy, changed["valid_m"])
    assert_outputs_match(cold, legacy, changed["valid_m"])


@pytest.mark.parametrize("operand", ["a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "prob_tensor"])
def test_glu_canonical_rejects_bad_layout_after_warmup(operand):
    inputs = natural_inputs(mxfp8_inputs([512] * 4))
    glu_block_scaled_call(inputs)
    bad = dict(inputs)
    if operand == "a_tensor":
        bad[operand] = inputs[operand].t().contiguous().t()
    elif operand == "b_tensor":
        bad[operand] = inputs[operand].transpose(0, 1).contiguous().transpose(0, 1)
    elif operand == "sfa_tensor":
        bad[operand] = inputs[operand][:-16]
    else:
        bad[operand] = (
            inputs[operand].view(torch.uint8).repeat_interleave(2)[::2].view(inputs[operand].dtype)
            if operand == "sfb_tensor"
            else inputs[operand].repeat_interleave(2)[::2]
        )
    with pytest.raises((ValueError, RuntimeError)):
        glu_block_scaled_call(bad)


def test_glu_canonical_graph_changed_inputs():
    inputs = mxfp8_inputs([512] * 4)
    natural = natural_inputs(inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        glu_block_scaled_call(natural)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = glu_block_scaled_call(natural)
    inputs["alpha_tensor"].mul_(0.5)
    inputs["prob_tensor"].mul_(0.75)
    legacy = glu_block_scaled_call(inputs)
    graph.replay()
    torch.cuda.synchronize()
    assert_outputs_match(result, legacy, 2048)


@pytest.mark.parametrize("natural_operand", ["a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "prob_tensor"])
def test_glu_mixed_layouts(natural_operand):
    inputs = mxfp8_inputs([512] * 4)
    mixed = dict(inputs)
    mixed[natural_operand] = natural_inputs(inputs)[natural_operand]
    reference = glu_block_scaled_call(inputs)
    result = glu_block_scaled_call(mixed)
    if natural_operand != "a_tensor":
        for name in ("sfd_row_tensor", "sfd_col_tensor"):
            result[name] = result[name].permute(SF_PHYSICAL)
    assert_outputs_match(result, reference, 2048)


@pytest.mark.parametrize("flat", [False, True])
def test_glu_discrete_canonical(flat):
    from cudnn import grouped_gemm_glu_wrapper_sm100

    inputs = mxfp8_inputs([512] * 4)
    natural = natural_inputs(inputs, flat)
    b = natural["b_tensor"]
    sfb = natural_inputs(inputs, False)["sfb_tensor"]
    b_ptrs = torch.tensor([b[i].data_ptr() for i in range(4)], device="cuda", dtype=torch.int64)
    sfb_ptrs = torch.tensor([sfb[i].data_ptr() for i in range(4)], device="cuda", dtype=torch.int64)

    def call(operands):
        return grouped_gemm_glu_wrapper_sm100(
            a_tensor=operands["a_tensor"],
            sfa_tensor=operands["sfa_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=operands["prob_tensor"],
            norm_const_tensor=inputs["norm_const_tensor"],
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            n=512,
            b_dtype=b.dtype,
            d_dtype=torch.float8_e4m3fn,
            sf_vec_size=32,
        )

    assert_outputs_match(call(natural), call(inputs), 2048)


@pytest.mark.parametrize("dynamic", ["0", "1"])
@pytest.mark.parametrize("flat", [False, True])
def test_glu_canonical_reuses_compilation_for_new_m(monkeypatch, dynamic, flat):
    from gemm.cutedsl.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors
    from cudnn.gemm.cutedsl.grouped.glu import api

    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", dynamic)
    monkeypatch.setattr(api, "_glu_wrapper_memo", {})
    monkeypatch.setattr(api, "_cache_of_GroupedGemmGluSm100Objects", {})
    for m in (1024, 2048, 1024):
        inputs = allocate_grouped_gemm_input_tensors(
            n=512,
            k=512,
            l=4,
            group_m_list=[m // 4] * 4,
            ab_dtype=torch.float8_e4m3fn,
            sf_dtype=torch.float8_e8m0fnu,
            sf_vec_size=32,
            m_aligned=256,
        )
        result = glu_block_scaled_call(natural_inputs(inputs, flat))
        assert result["d_tensor"].shape == (m, 256)
        assert len(api._cache_of_GroupedGemmGluSm100Objects) == 1
        reference = glu_block_scaled_call(inputs)
        assert_outputs_match(result, reference, m)
        legacy_keys = [key for key, value in api._cache_of_GroupedGemmGluSm100Objects.items() if not value._implementation.canonical_a]
        for key in legacy_keys:
            del api._cache_of_GroupedGemmGluSm100Objects[key]


def test_glu_canonical_warm_call_has_no_tensor_views(monkeypatch):
    inputs = natural_inputs(mxfp8_inputs([512] * 4))
    glu_block_scaled_call(inputs)

    def unexpected_view(*args, **kwargs):
        raise AssertionError("canonical launch must not construct host tensor views")

    with monkeypatch.context() as patch:
        for method in ("view", "reshape", "permute", "unsqueeze", "as_strided"):
            patch.setattr(torch.Tensor, method, unexpected_view)
        result = glu_block_scaled_call(inputs)
    assert result["d_tensor"].shape == (2048, 256)


@pytest.mark.parametrize("ab_dtype,sf_vec_size", [(torch.float8_e4m3fn, 32), (torch.float4_e2m1fn_x2, 16)])
def test_glu_canonical_bf16_output(ab_dtype, sf_vec_size):
    from gemm.cutedsl.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors

    inputs = allocate_grouped_gemm_input_tensors(
        n=512,
        k=512,
        l=4,
        group_m_list=[512] * 4,
        ab_dtype=ab_dtype,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=sf_vec_size,
        m_aligned=256,
    )
    natural = natural_inputs(inputs)
    reference = glu_block_scaled_call(inputs, d_dtype=torch.bfloat16, sf_vec_size=sf_vec_size)
    result = glu_block_scaled_call(natural, d_dtype=torch.bfloat16, sf_vec_size=sf_vec_size)
    for name in ("c_tensor", "d_tensor", "amax_tensor"):
        assert torch.equal(raw_bytes(result[name]), raw_bytes(reference[name])), name


def test_glu_canonical_rejects_padded_rows_after_warmup():
    inputs = natural_inputs(mxfp8_inputs([512] * 4))
    glu_block_scaled_call(inputs)
    padded = torch.empty((2048, 528), dtype=inputs["a_tensor"].dtype, device="cuda")[:, :512]
    inputs["a_tensor"] = padded
    with pytest.raises(ValueError, match="contiguous"):
        glu_block_scaled_call(inputs)
