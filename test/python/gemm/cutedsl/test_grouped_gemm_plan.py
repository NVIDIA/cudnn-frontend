# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped GEMM wrapper calls planned once and executed in C++ (cudnn.gemm.cutedsl.grouped.plan)."""

import inspect

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_prepared import assert_exact_outputs, fresh_arguments, prepared_kwargs

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")


def assert_cpp_plans(run, count):
    """``count`` plans, each executed by the C++ op."""
    from cudnn import _compiled_module

    assert len(run.plans) == count
    assert all(isinstance(getattr(plan, "op", None), _compiled_module._GroupedGemm) for plan in run.plans.values())


def planned_wrapper(kind):
    """A fresh planned wrapper, so plans from other tests do not answer."""
    import cudnn
    from cudnn.gemm.cutedsl.grouped.plan import planned

    return planned(inspect.unwrap(getattr(cudnn, f"grouped_gemm_{kind}_wrapper_sm100")))


@pytest.mark.parametrize("kind", ["glu", "quant"])
def test_plan_matches_the_wrapper(kind):
    run = planned_wrapper(kind)
    plain = inspect.unwrap(run)
    kwargs = prepared_kwargs(kind)
    for _ in range(3):  # derive and compile, plan, run the plan
        assert_exact_outputs(run(**kwargs), plain(**kwargs))
    assert_cpp_plans(run, 1)
    current = fresh_arguments(kwargs)  # new values and pointers, same signature
    assert_exact_outputs(run(**current), plain(**current))


@pytest.mark.parametrize("kind", ["glu", "quant"])
def test_each_routed_row_count_has_its_plan(kind):
    run = planned_wrapper(kind)
    plain = inspect.unwrap(run)
    for groups in ((512,) * 4, (256, 768, 512, 1024), (256,) * 4):
        kwargs = prepared_kwargs(kind, groups)
        for _ in range(3):
            assert_exact_outputs(run(**kwargs), plain(**kwargs))
    assert_cpp_plans(run, 3)


def test_a_less_aligned_call_runs_the_wrapper():
    run = planned_wrapper("glu")
    plain = inspect.unwrap(run)
    kwargs = prepared_kwargs("glu")
    for _ in range(3):
        run(**kwargs)
    (plan,) = run.plans.values()
    name, alignment = "a_tensor", 16
    tensor = kwargs[name]
    nbytes = tensor.numel() * tensor.element_size()
    buffer = torch.empty(nbytes + alignment, dtype=torch.uint8, device="cuda")
    shifted = dict(kwargs)
    shifted[name] = buffer[alignment // 2 :][:nbytes].view(tensor.dtype).view(tensor.shape)
    shifted[name].copy_(tensor)
    assert plan(shifted) is None
    try:
        result, error = run(**shifted), None
    except Exception as e:  # noqa: BLE001 -- whatever the wrapper does with it
        result, error = None, e
    if error is None:
        assert_exact_outputs(result, plain(**shifted))
    else:
        with pytest.raises(type(error)):
            plain(**shifted)


def test_plans_can_be_turned_off(monkeypatch):
    from cudnn.gemm.cutedsl.grouped import plan

    monkeypatch.setattr(plan, "ENABLED", False)
    wrapper = inspect.unwrap(planned_wrapper("glu"))
    assert plan.planned(wrapper) is wrapper


def test_cpp_op_runs_the_registered_kernel():
    """What a C++ caller does: create GroupedGemm for its own operands, by name, and execute it."""
    import json

    from cudnn import _compiled_module
    from cudnn.gemm.cutedsl.grouped.plan import _data_type, _option

    run = planned_wrapper("glu")
    plain = inspect.unwrap(run)
    kwargs = prepared_kwargs("glu")
    for _ in range(2):  # derive and compile, then plan: registers the kernel
        run(**kwargs)
    (plan,) = run.plans.values()
    current = fresh_arguments(kwargs)
    outputs, _ = plan.allocate(lambda on_stream: True)
    operands = [(name, current[name]) for name in plan.inputs] + [(name, outputs[name]) for name, *_ in plan.outputs]
    described = [(name, (t.data_ptr(), _data_type(t.dtype), tuple(t.shape), t.stride())) for name, t in operands]
    options = json.dumps({k: _option(v) for k, v in current.items() if k != "current_stream" and not isinstance(v, torch.Tensor)}, sort_keys=True)
    op = _compiled_module._GroupedGemm("glu", described, options)
    assert op.execute([t.data_ptr() for _, t in operands], 0, torch.cuda.current_stream().cuda_stream)
    expected = plain(**current)
    assert_exact_outputs({name: outputs[name] for name in outputs}, {name: expected[name] for name in outputs})
