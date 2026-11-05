# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped GEMM wrappers' repeated calls run by launch plans (cudnn.launch_plan)."""

import inspect

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_bf16_utils import make_grouped_gemm_bf16_problem
from gemm.cutedsl.test_grouped_gemm_dglu_bf16_utils import make_grouped_gemm_dglu_bf16_problem
from gemm.cutedsl.test_grouped_gemm_glu_bf16_utils import make_grouped_gemm_glu_bf16_problem
from gemm.cutedsl.test_grouped_gemm_prepared import prepared_kwargs
from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import make_grouped_gemm_wgrad_bf16_problem
from gemm.cutedsl.test_grouped_gemm_wrapper_memo import dglu_block_scaled_inputs, raw_bytes, wgrad_inputs

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")


def gemm_bf16():
    p = make_grouped_gemm_bf16_problem()
    kwargs = dict(a_tensor=p["a"].squeeze(-1), padded_offsets=p["offsets"], alpha_tensor=p["alpha"], b_tensor=p["b_storage"], prob_tensor=p["prob"].view(-1))
    return "", dict(kwargs, c_dtype=torch.bfloat16, d_dtype=torch.bfloat16, generate_c=True)


def glu_bf16():
    p = make_grouped_gemm_glu_bf16_problem()
    kwargs = dict(a_tensor=p["a"].squeeze(-1), sfa_tensor=None, padded_offsets=p["offsets"], alpha_tensor=p["alpha"], b_tensor=p["b_reference"])
    return "glu_", dict(kwargs, prob_tensor=p["prob"].view(-1), c_dtype=torch.bfloat16, d_dtype=torch.bfloat16, generate_c=True)


def dglu_bf16():
    p = make_grouped_gemm_dglu_bf16_problem()
    kwargs = dict(a_tensor=p["a"].squeeze(-1), c_tensor=p["c"].squeeze(-1), sfa_tensor=None, padded_offsets=p["offsets"], alpha_tensor=p["alpha"])
    kwargs.update(beta_tensor=p["beta"], prob_tensor=p["prob"].view(-1), dprob_tensor=torch.zeros_like(p["dprob"]).view(-1), b_tensor=p["b_reference"])
    return "dglu_", dict(kwargs, d_dtype=torch.bfloat16)


def wgrad_bf16(accumulate=False):
    p = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(256, 256))
    kwargs = dict(a_tensor=p["a"], b_tensor=p["b"], sfa_tensor=None, sfb_tensor=None, offsets_tensor=p["offsets"], acc_dtype=torch.float32)
    kwargs.update(wgrad_dtype=p["output_dtype"], mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1))
    if accumulate:
        kwargs.update(wgrad_tensor=torch.randn_like(p["output"]), accumulate_on_output=True)
    return "wgrad_", kwargs


def dglu_mxfp8():
    inputs = dglu_block_scaled_inputs([512] * 4)
    names = ("a_tensor", "c_tensor", "sfa_tensor", "alpha_tensor", "beta_tensor", "prob_tensor", "dprob_tensor", "b_tensor", "sfb_tensor", "norm_const_tensor")
    kwargs = {name: inputs[name] for name in names}
    return "dglu_", dict(kwargs, padded_offsets=inputs["padded_offsets_tensor"], d_dtype=torch.float8_e4m3fn, sf_vec_size=32)


def wgrad_mxfp8():
    inputs = wgrad_inputs([256, 384])
    kwargs = {name: inputs[name] for name in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "offsets_tensor")}
    return "wgrad_", dict(kwargs, wgrad_dtype=torch.bfloat16, mma_tiler_mn=(256, 256), cluster_shape_mn=(2, 1), sf_vec_size=32)


CASES = {
    "gemm_bf16": gemm_bf16,
    "glu_bf16": glu_bf16,
    "dglu_bf16": dglu_bf16,
    "wgrad_bf16": wgrad_bf16,
    "wgrad_bf16_accumulate": lambda: wgrad_bf16(accumulate=True),
    "glu_mxfp8": lambda: ("glu_", prepared_kwargs("glu")),
    "quant_mxfp8": lambda: ("quant_", prepared_kwargs("quant")),
    "quant_mxfp8_amax": lambda: ("quant_", dict(prepared_kwargs("quant"), generate_amax=True)),  # amax: filled with -inf on the launch stream
    "dglu_mxfp8": dglu_mxfp8,
    "wgrad_mxfp8": wgrad_mxfp8,
}
ACCUMULATED = ("dprob_tensor", "wgrad_tensor")  # caller tensors a call accumulates into: restored before each call


def planned_wrapper(kind):
    """A fresh planned wrapper, so plans from other tests do not answer, and the unplanned function."""
    import cudnn
    from cudnn.launch_plan import planned

    plain = inspect.unwrap(getattr(cudnn, f"grouped_gemm_{kind}wrapper_sm100"))
    return planned(plain), plain


def call(fn, kwargs, initial):
    """``fn(**kwargs)`` from the initial accumulators, with its outputs copied once it finished."""
    for name, value in initial.items():
        kwargs[name].copy_(value)
    outputs = fn(**kwargs)
    torch.cuda.synchronize()
    return {name: None if t is None else t.clone() for name, t in outputs.items()}


def assert_same_outputs(actual, expected):
    assert actual.keys() == expected.keys()
    for name, tensor in actual.items():
        if tensor is None:
            assert expected[name] is None, name
        elif name in ("dprob_tensor", "dbias_tensor"):  # atomic float accumulation: the order differs between launches
            torch.testing.assert_close(tensor, expected[name], rtol=1e-4, atol=1e-4)
        else:
            assert torch.equal(raw_bytes(tensor), raw_bytes(expected[name])), name


def the_plan(run):
    (plan,) = run.plans.values()
    assert type(plan).__name__ == "_Plan"
    return plan


@pytest.mark.parametrize("case", list(CASES))
def test_repeated_call_is_planned_and_matches_the_wrapper(case):
    kind, kwargs = CASES[case]()
    run, plain = planned_wrapper(kind)
    initial = {name: kwargs[name].clone() for name in ACCUMULATED if name in kwargs}
    for _ in range(3):  # compile, record, run the plan
        planned_outputs = call(run, kwargs, initial)
    the_plan(run)
    assert_same_outputs(planned_outputs, call(plain, kwargs, initial))
    fresh = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in kwargs.items()}  # new pointers
    assert_same_outputs(call(run, fresh, initial), call(plain, fresh, initial))


def test_a_less_aligned_call_runs_the_wrapper():
    from cudnn import launch_plan

    run, plain = planned_wrapper("glu_")
    kwargs = prepared_kwargs("glu")
    for _ in range(3):
        run(**kwargs)
    plan = the_plan(run)
    tensor = kwargs["a_tensor"]
    nbytes = tensor.numel() * tensor.element_size()
    buffer = torch.empty(nbytes + 16, dtype=torch.uint8, device="cuda")
    shifted = dict(kwargs, a_tensor=buffer[8:][:nbytes].view(tensor.dtype).view(tensor.shape))
    shifted["a_tensor"].copy_(tensor)
    assert plan((), shifted) is launch_plan._UNPLANNED
    try:
        result, error = call(run, shifted, {}), None
    except Exception as e:  # noqa: BLE001 -- whatever the wrapper does with it
        result, error = None, e
    if error is None:
        assert_same_outputs(result, call(plain, shifted, {}))
    else:
        with pytest.raises(type(error)):
            plain(**shifted)


@pytest.mark.parametrize("case", ["quant_mxfp8_amax", "wgrad_bf16"])
def test_outputs_are_allocated_on_the_launch_stream(case):
    """A launch stream other than torch's current one: the plan allocates its outputs there, on each call's stream."""
    from cuda.bindings import driver as cuda

    kind, kwargs = CASES[case]()
    run, plain = planned_wrapper(kind)
    for _ in range(3):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        call(run, dict(kwargs, current_stream=cuda.CUstream(side.cuda_stream)), {})
    assert any(on for *_, on in the_plan(run).outputs)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    planned_outputs = call(run, dict(kwargs, current_stream=cuda.CUstream(side.cuda_stream)), {})
    assert_same_outputs(planned_outputs, call(plain, dict(kwargs, current_stream=cuda.CUstream(side.cuda_stream)), {}))


def test_plans_can_be_turned_off(monkeypatch):
    from cudnn import launch_plan

    monkeypatch.setattr(launch_plan, "ENABLED", False)
    plain = planned_wrapper("glu_")[1]
    assert launch_plan.planned(plain) is plain
