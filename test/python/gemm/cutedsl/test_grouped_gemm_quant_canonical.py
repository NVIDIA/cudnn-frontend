# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_wrapper_memo import mxfp8_inputs, raw_bytes
from gemm.cutedsl.test_grouped_gemm_glu_canonical import natural_inputs, SF_PHYSICAL

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")


def quant_call(inputs, **overrides):
    from cudnn import grouped_gemm_quant_wrapper_sm100

    kwargs = dict(
        a_tensor=inputs["a_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        b_tensor=inputs["b_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        norm_const_tensor=inputs["norm_const_tensor"],
        prob_tensor=inputs["prob_tensor"],
        d_dtype=torch.bfloat16,
        sf_vec_size=32,
    )
    kwargs.update(overrides)
    return grouped_gemm_quant_wrapper_sm100(**kwargs)


def assert_quant_equal(actual, expected, valid_m):
    for name, left in actual.items():
        right = expected[name]
        if left is None:
            assert right is None
            continue
        if name.startswith("sfd"):
            if left.is_contiguous():
                left = left.permute(3, 4, 1, 5, 2, 0)
            if right.is_contiguous():
                right = right.permute(3, 4, 1, 5, 2, 0)
            if name == "sfd_row_tensor":
                left, right = left[:, :, : valid_m // 128], right[:, :, : valid_m // 128]
            else:
                left, right = left[:, :, :, :, : valid_m // 128], right[:, :, :, :, : valid_m // 128]
        elif name != "amax_tensor":
            left, right = left[:valid_m], right[:valid_m]
        assert torch.equal(raw_bytes(left), raw_bytes(right)), name


@pytest.mark.parametrize("dynamic", ["0", "1"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("d_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_quant_canonical_matches_legacy(monkeypatch, dynamic, flat, d_dtype):
    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", dynamic)
    inputs = mxfp8_inputs([512] * 4)
    reference = quant_call(inputs, d_dtype=d_dtype)
    natural = natural_inputs(inputs, flat)
    for _ in range(2):
        result = quant_call(natural, d_dtype=d_dtype)
        assert result["d_tensor"].shape == (2048, 512)
        assert_quant_equal(result, reference, 2048)


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("first_amax", [False, True])
def test_quant_optional_amax_and_preallocated_output(canonical, first_amax):
    inputs = mxfp8_inputs([512] * 4)
    if canonical:
        inputs = natural_inputs(inputs)
    reference = quant_call(inputs)
    for flag in (first_amax, not first_amax, first_amax):
        output = torch.empty_like(reference["d_tensor"])
        result = quant_call(inputs, generate_amax=flag, d_tensor=output)
        assert result["d_tensor"] is output
        assert (result["amax_tensor"] is not None) == flag
        expected = dict(reference.items())
        if not flag:
            expected["amax_tensor"] = None
        assert_quant_equal(result, expected, 2048)


def test_quant_memo_uses_current_data_and_routing():
    from cudnn.gemm.cutedsl.grouped.quant.api import _quant_wrapper_memo

    inputs = natural_inputs(mxfp8_inputs([256] * 4))
    quant_call(inputs, generate_amax=False)
    changed = natural_inputs(mxfp8_inputs([512, 256, 512, 256]))
    row_scale = torch.rand(2048, device="cuda")
    bias = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16).t()
    quant_call(changed, row_scale_tensor=row_scale, bias_tensor=bias, generate_amax=False)
    count = len(_quant_wrapper_memo)
    changed["alpha_tensor"].mul_(0.5)
    changed["prob_tensor"].mul_(0.75)
    bias.add_(1)
    row_scale.mul_(0.25)
    warm = quant_call(changed, row_scale_tensor=row_scale, bias_tensor=bias, generate_amax=False)
    assert len(_quant_wrapper_memo) == count
    _quant_wrapper_memo.clear()
    cold = quant_call(changed, row_scale_tensor=row_scale, bias_tensor=bias, generate_amax=False)
    assert_quant_equal(warm, cold, changed["valid_m"])


@pytest.mark.parametrize("operand", ["a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "prob_tensor", "d_tensor"])
def test_quant_rejects_invalid_layout_after_warmup(operand):
    inputs = natural_inputs(mxfp8_inputs([512] * 4))
    output = quant_call(inputs)["d_tensor"]
    if operand == "d_tensor":
        bad = output.t().contiguous().t()
    elif operand in ("sfa_tensor", "sfb_tensor"):
        bad = inputs[operand][:-16]
    elif operand == "a_tensor":
        bad = inputs[operand].t().contiguous().t()
    elif operand == "b_tensor":
        bad = inputs[operand].transpose(0, 1).contiguous().transpose(0, 1)
    else:
        bad = inputs[operand].repeat_interleave(2)[::2]
    with pytest.raises((ValueError, RuntimeError)):
        quant_call(inputs, **{operand: bad})


@pytest.mark.parametrize("dynamic", ["0", "1"])
def test_quant_canonical_dynamic_m_reuses_compilation(monkeypatch, dynamic):
    from cudnn.gemm.cutedsl.grouped.quant import api

    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", dynamic)
    monkeypatch.setattr(api, "_quant_wrapper_memo", {})
    monkeypatch.setattr(api, "_cache_of_GroupedGemmQuantSm100Objects", {})
    inputs = natural_inputs(mxfp8_inputs([512] * 4))
    quant_call(inputs, generate_amax=False)
    smaller = dict(inputs)
    smaller["a_tensor"] = inputs["a_tensor"][:1024]
    smaller["sfa_tensor"] = inputs["sfa_tensor"][: inputs["sfa_tensor"].numel() // 2]
    smaller["prob_tensor"] = inputs["prob_tensor"][:1024]
    smaller["padded_offsets_tensor"] = torch.arange(256, 1025, 256, device="cuda", dtype=torch.int32)
    result = quant_call(smaller, generate_amax=False)
    assert len(api._cache_of_GroupedGemmQuantSm100Objects) == 1
    assert len(api._quant_wrapper_memo) == 2
    api._quant_wrapper_memo.clear()
    api._cache_of_GroupedGemmQuantSm100Objects.clear()
    cold = quant_call(smaller, generate_amax=False)
    assert_quant_equal(result, cold, 1024)


@pytest.mark.parametrize("generate_amax", [False, True])
def test_quant_discrete_canonical(generate_amax):
    inputs = mxfp8_inputs([512] * 4)
    natural = natural_inputs(inputs, False)
    b, sfb = natural["b_tensor"], natural["sfb_tensor"]
    options = dict(
        b_tensor=None,
        sfb_tensor=None,
        b_ptrs=torch.tensor([b[i].data_ptr() for i in range(4)], device="cuda", dtype=torch.int64),
        sfb_ptrs=torch.tensor([sfb[i].data_ptr() for i in range(4)], device="cuda", dtype=torch.int64),
        n=512,
        b_dtype=b.dtype,
        generate_amax=generate_amax,
    )
    reference = quant_call(inputs, **options)
    for _ in range(2):
        assert_quant_equal(quant_call(natural, **options), reference, 2048)


@pytest.mark.parametrize("generate_amax", [False, True])
def test_quant_graph_current_stream(generate_amax):
    from cuda.bindings import driver as cuda

    inputs = natural_inputs(mxfp8_inputs([512] * 4))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    options = dict(generate_amax=generate_amax, current_stream=cuda.CUstream(stream.cuda_stream))
    with torch.cuda.stream(stream):
        quant_call(inputs, **options)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = quant_call(inputs, **options)
    inputs["alpha_tensor"].mul_(0.5)
    reference = quant_call(inputs, generate_amax=generate_amax)
    graph.replay()
    torch.cuda.synchronize()
    assert_quant_equal(result, reference, 2048)


@pytest.mark.parametrize("operand", ["a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "prob_tensor"])
def test_quant_mixed_layouts(operand):
    inputs = mxfp8_inputs([512] * 4)
    mixed = dict(inputs)
    mixed[operand] = natural_inputs(inputs)[operand]
    options = dict(d_dtype=torch.float8_e4m3fn)
    assert_quant_equal(quant_call(mixed, **options), quant_call(inputs, **options), 2048)


def test_quant_canonical_nvfp4():
    from gemm.cutedsl.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors

    inputs = allocate_grouped_gemm_input_tensors(
        n=512,
        k=512,
        l=4,
        group_m_list=[512] * 4,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        m_aligned=256,
    )
    options = dict(sf_vec_size=16, generate_amax=False)
    assert_quant_equal(quant_call(natural_inputs(inputs), **options), quant_call(inputs, **options), 2048)


def test_quant_memo_environment_controls(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.quant import api

    monkeypatch.setattr(api, "_quant_wrapper_memo", {})
    monkeypatch.setattr(api, "_cache_of_GroupedGemmQuantSm100Objects", {})
    inputs = natural_inputs(mxfp8_inputs([512] * 4))
    for dynamic, margin in (("1", "0"), ("0", "0"), ("1", "8"), ("1", "0")):
        monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", dynamic)
        monkeypatch.setenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", margin)
        quant_call(inputs, generate_amax=False)
        memo = next(value for key, value in api._quant_wrapper_memo.items() if key[-2:] == (margin, dynamic != "0"))
        assert memo[0].num_cluster_overlap_margin == int(margin)
        assert memo[0]._use_full_dynamic_mnkl == (dynamic != "0")
    assert len(api._quant_wrapper_memo) == len(api._cache_of_GroupedGemmQuantSm100Objects) == 3


def test_quant_canonical_warm_call_creates_no_views():
    inputs = natural_inputs(mxfp8_inputs([512] * 4))
    quant_call(inputs, generate_amax=False)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        quant_call(inputs, generate_amax=False)
    names = {event.key for event in profile.key_averages()}
    assert not names.intersection({"aten::view", "aten::permute", "aten::as_strided", "aten::unsqueeze", "aten::reshape"})
