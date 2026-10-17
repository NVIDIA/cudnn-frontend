# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_wrapper_memo import mxfp8_inputs, raw_bytes
from gemm.cutedsl.test_grouped_gemm_glu_canonical import natural_inputs

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")


def prepared_kwargs(kind, groups=(512,) * 4):
    from cuda.bindings import driver as cuda

    inputs = natural_inputs(mxfp8_inputs(list(groups), tensor_m=sum(groups)))
    result = {name: inputs[name] for name in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "alpha_tensor", "prob_tensor", "norm_const_tensor")}
    result.update(
        padded_offsets=inputs["padded_offsets_tensor"],
        sf_vec_size=32,
        use_dynamic_sched=True,
        current_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    if kind == "glu":
        result.update(d_dtype=torch.float8_e4m3fn, c_dtype=torch.bfloat16, act_func="swiglu")
    else:
        result.update(d_dtype=torch.bfloat16, generate_amax=False)
    return result


def fresh_arguments(kwargs):
    result = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in kwargs.items()}
    result["alpha_tensor"].mul_(0.5)
    result["prob_tensor"].mul_(0.75)
    result["padded_offsets"].copy_(torch.tensor([768, 1024, 1536, 2048], dtype=torch.int32, device="cuda"))
    return result


def wrapper_call(kind, kwargs):
    from cudnn import grouped_gemm_glu_wrapper_sm100, grouped_gemm_quant_wrapper_sm100

    wrapper = grouped_gemm_glu_wrapper_sm100 if kind == "glu" else grouped_gemm_quant_wrapper_sm100
    return wrapper(**kwargs)


def assert_exact_outputs(actual, expected):
    assert actual.keys() == expected.keys()
    for name, tensor in actual.items():
        if tensor is None:
            assert expected[name] is None, name
        else:
            assert torch.equal(raw_bytes(tensor), raw_bytes(expected[name])), name


@pytest.mark.parametrize("kind", ["glu", "quant"])
@pytest.mark.parametrize("check", [False, True])
def test_prepared_uses_fresh_values_and_routing(kind, check):
    from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm

    initial = prepared_kwargs(kind)
    plan = prepare_grouped_gemm(kind, **initial)
    current = fresh_arguments(initial)
    initial_result = plan.run(check=check, **initial)
    result = plan.run(check=check, **current)
    assert_exact_outputs(result, wrapper_call(kind, current))
    assert not torch.equal(raw_bytes(initial_result["d_tensor"]), raw_bytes(result["d_tensor"]))


@pytest.mark.parametrize("kind", ["glu", "quant"])
@pytest.mark.parametrize("check", [False, True])
def test_prepared_serves_new_routed_rows(kind, check):
    from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm

    plan = prepare_grouped_gemm(kind, reuse_row_outputs=True, **prepared_kwargs(kind))
    for groups in ((256, 768, 512, 1024), (512,) * 4, (256,) * 4):
        current = prepared_kwargs(kind, groups)
        assert_exact_outputs(plan.run(check=check, **current), wrapper_call(kind, current))
    assert prepare_grouped_gemm(kind, **prepared_kwargs(kind, (256,) * 4)).api is plan.api


def test_prepared_quant_preserves_supplied_output_identity():
    from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm

    kwargs = prepared_kwargs("quant")
    provided = torch.empty((2048, 512), dtype=torch.bfloat16, device="cuda")
    kwargs["d_tensor"] = provided
    plan = prepare_grouped_gemm("quant", **kwargs)
    reference_kwargs = dict(kwargs, d_tensor=None)
    expected = wrapper_call("quant", reference_kwargs)
    result = plan.run(**kwargs)
    assert result["d_tensor"] is provided
    assert_exact_outputs(result, expected)
    supplied_again = torch.empty_like(provided)
    kwargs.pop("d_tensor")
    result = plan.run(outputs={"d_tensor": supplied_again}, **kwargs)
    assert result["d_tensor"] is supplied_again
    assert_exact_outputs(result, expected)


def test_prepared_glu_only_reuses_consumed_row_outputs():
    from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm

    kwargs = prepared_kwargs("glu")
    plan = prepare_grouped_gemm("glu", reuse_row_outputs=True, **kwargs)
    first = plan.run(**kwargs)
    assert_exact_outputs(first, wrapper_call("glu", kwargs))
    retained = {name: first[name].clone() for name in ("c_tensor", "d_col_tensor", "sfd_col_tensor")}
    current = fresh_arguments(kwargs)
    second = plan.run(**current)
    assert_exact_outputs(second, wrapper_call("glu", current))
    for name in ("d_tensor", "sfd_row_tensor"):
        assert first[name].data_ptr() == second[name].data_ptr(), name
    for name, saved in retained.items():
        assert first[name].data_ptr() != second[name].data_ptr(), name
        assert torch.equal(raw_bytes(first[name]), raw_bytes(saved)), name


@pytest.mark.parametrize("kind", ["glu", "quant"])
def test_prepared_does_not_retain_sample_operands(kind):
    from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm

    kwargs = prepared_kwargs(kind)
    refs = [weakref.ref(value) for value in kwargs.values() if isinstance(value, torch.Tensor)]
    current = fresh_arguments(kwargs)
    plan = prepare_grouped_gemm(kind, **kwargs)
    del kwargs
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert_exact_outputs(plan.run(**current), wrapper_call(kind, current))


@pytest.mark.parametrize("operand", ["a_tensor", "sfa_tensor", "prob_tensor"])
def test_prepared_checked_call_rejects_metadata_changes(operand):
    from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm

    kwargs = prepared_kwargs("quant")
    plan = prepare_grouped_gemm("quant", **kwargs)
    bad = dict(kwargs)
    if operand == "a_tensor":
        bad[operand] = bad[operand].t().contiguous().t()
    elif operand == "sfa_tensor":
        bad[operand] = bad[operand][:-16]
    else:
        bad[operand] = bad[operand].to(torch.bfloat16)
    with pytest.raises(ValueError, match=operand):
        plan.run(**bad)
    assert_exact_outputs(plan.run(**kwargs), wrapper_call("quant", kwargs))


def test_prepared_rejects_scalar_and_output_changes():
    from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm

    kwargs = prepared_kwargs("quant")
    plan = prepare_grouped_gemm("quant", **kwargs)
    with pytest.raises(ValueError, match="sf_vec_size"):
        plan.run(**dict(kwargs, sf_vec_size=16))
    bad_output = torch.empty((2048, 512), dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="output"):
        plan.run(outputs={"d_tensor": bad_output}, **kwargs)


def test_prepared_rejects_another_stream():
    from cuda.bindings import driver as cuda
    from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm

    kwargs = prepared_kwargs("quant")
    plan = prepare_grouped_gemm("quant", **kwargs)
    stream = torch.cuda.Stream()
    with pytest.raises(ValueError, match="stream"):
        plan.run(**dict(kwargs, current_stream=cuda.CUstream(stream.cuda_stream)))


@pytest.mark.parametrize("kind", ["glu", "quant"])
def test_prepared_external_scheduler_counter_reset(kind):
    from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm

    kwargs = prepared_kwargs(kind)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    plan = prepare_grouped_gemm(kind, **dict(kwargs, scheduler_counter_tensor=counter))
    for current in (kwargs, fresh_arguments(kwargs)):
        expected = wrapper_call(kind, current)
        counter.zero_()
        result = plan.run(**dict(current, scheduler_counter_tensor=counter))
        assert_exact_outputs(result, expected)
        assert counter.item() > 0


def test_prepared_accepts_default_stream_argument():
    from cudnn.gemm.cutedsl.grouped.prepared import prepare_grouped_gemm

    kwargs = prepared_kwargs("quant")
    kwargs["current_stream"] = None
    plan = prepare_grouped_gemm("quant", **kwargs)
    assert_exact_outputs(plan.run(**kwargs), wrapper_call("quant", kwargs))
