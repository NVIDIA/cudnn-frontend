# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json

import pytest
import torch


@pytest.fixture
def mhc_api():
    pytest.importorskip("cuda.tile", minversion="1.5.0")
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mHC projection backward requires SM100")
    from cudnn import MhcProjectionBackward

    return MhcProjectionBackward


def _descriptors():
    from cudnn.api_base import TensorDesc

    specs = {
        "x": ((4096, 20480), (20480, 1), torch.bfloat16),
        "weight": ((24, 20480), (20480, 1), torch.float32),
        "grad_proj": ((4096, 32), (32, 1), torch.float32),
        "grad_r": ((4096, 1), (1, 1), torch.float32),
        "r": ((4096, 1), (1, 1), torch.float32),
        "dx": ((4096, 20480), (20480, 1), torch.bfloat16),
        "dweight": ((24, 20480), (20480, 1), torch.float32),
    }
    return {
        name: TensorDesc(dtype, shape, stride, (1, 0), torch.device("cuda", torch.cuda.current_device()), name=name)
        for name, (shape, stride, dtype) in specs.items()
    }


@pytest.mark.L0
def test_mhc_projection_descriptors_and_precision(mhc_api):
    descriptors = _descriptors()
    plan = mhc_api(**descriptors, allow_tf32=True, backend="frost")
    assert plan.check_support()
    assert plan.scratch_workspace_bytes() == 15_728_640
    with pytest.raises(NotImplementedError, match="allow_tf32"):
        mhc_api(**descriptors).check_support()
    with pytest.raises(ValueError, match="backend='frost'"):
        mhc_api(**descriptors, allow_tf32=True, backend="other").check_support()
    for name in descriptors:
        invalid = dict(descriptors)
        invalid[name] = replace(invalid[name], dtype=torch.float16)
        with pytest.raises(NotImplementedError, match=name):
            mhc_api(**invalid, allow_tf32=True).check_support()
    for name in ("x", "weight", "grad_proj", "dx", "dweight"):
        invalid = dict(descriptors)
        invalid[name] = replace(invalid[name], stride=(1, invalid[name].shape[0]))
        with pytest.raises(NotImplementedError, match=name):
            mhc_api(**invalid, allow_tf32=True).check_support()


def _inputs():
    torch.manual_seed(4103793)
    x = torch.randn((4096, 20480), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((24, 20480), device="cuda", dtype=torch.float32) * 0.01
    gp = torch.randn((4096, 32), device="cuda", dtype=torch.float32)
    # Ignored physical padding is deliberately nonzero.
    gp[:, 24:] = 37
    gr = torch.randn((4096, 1), device="cuda", dtype=torch.float32)
    r = torch.rand((4096, 1), device="cuda", dtype=torch.float32) + 0.5
    return dict(x=x, weight=weight, grad_proj=gp, grad_r=gr, r=r)


def _reference(inputs):
    x, w = inputs["x"].double(), inputs["weight"].double()
    p = inputs["grad_proj"][:, :24].double()
    return dict(dx=p @ w + inputs["grad_r"].double() / (inputs["r"].double() * 20480) * x, dweight=p.T @ x)


def _check(actual, reference):
    for name in ("dx", "dweight"):
        value = actual[name].double()
        ref = reference[name]
        difference = value - ref
        assert torch.isfinite(value).all()
        assert (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(ref).clamp_min(1e-30)).item() <= 0.003
        assert (difference.abs().max() / ref.abs().max().clamp_min(1e-30)).item() <= 0.01


@pytest.mark.L1
def test_mhc_projection_prepared_graph_and_route(mhc_api, monkeypatch, tmp_path):
    inputs = _inputs()
    outputs = dict(dx=torch.empty_like(inputs["x"]), dweight=torch.empty_like(inputs["weight"]))
    plan = mhc_api(**inputs, **outputs, allow_tf32=True)
    with pytest.raises(RuntimeError, match="compile"):
        plan.execute(**inputs, **outputs, workspace=None)
    plan.compile()
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
    reference = _reference(inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Explicit launch on a stream other than the current stream.
    old_sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        plan.execute(**inputs, **outputs, workspace=workspace, current_stream=stream)
    finally:
        torch.cuda.set_sync_debug_mode(old_sync_mode)
    stream.synchronize()
    _check(outputs, reference)

    # Autograd workers may have no current CUDA context. An explicit raw
    # stream must still work, and the caller's context must be restored.
    def worker():
        from cuda.bindings import driver

        status, before = driver.cuCtxGetCurrent()
        assert int(status) == 0
        plan.execute(**inputs, **outputs, workspace=workspace, current_stream=stream.cuda_stream)
        status, after = driver.cuCtxGetCurrent()
        assert int(status) == 0 and int(before) == int(after)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(worker).result(timeout=30)
    stream.synchronize()
    _check(outputs, reference)

    import cudnn.gemm.mhc_projection_bwd._compile as compiler

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared execute allocated storage or compiled again")

    with monkeypatch.context() as patch:
        patch.setattr(compiler, "_binaries", forbidden)
        for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like"):
            patch.setattr(torch, name, forbidden)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            plan.execute(**inputs, **outputs, workspace=workspace)
            torch.cuda.synchronize()
    trace = tmp_path / "mhc_projection.trace.json"
    prof.export_chrome_trace(str(trace))
    kernels = [e["name"] for e in json.loads(trace.read_text())["traceEvents"] if e.get("cat") == "kernel"]
    assert len(kernels) == 2
    assert sum("cudnn_mhc_projection_partial" in name for name in kernels) == 1
    assert sum("cudnn_mhc_projection_reduce" in name for name in kernels) == 1
    original = {name: value.clone() for name, value in outputs.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.execute(**inputs, **outputs, workspace=workspace, current_stream=stream)
    for value in outputs.values():
        value.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(outputs[name], value) for name, value in original.items())
    inputs["x"].mul_(0.5)
    inputs["grad_proj"].mul_(0.625)
    changed = _reference(inputs)
    for value in outputs.values():
        value.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _check(outputs, changed)
    assert all(not torch.equal(outputs[name], value) for name, value in original.items())
    with pytest.raises(ValueError, match="overlap"):
        plan.execute(**dict(inputs, r=inputs["grad_r"]), **outputs, workspace=workspace)
    with pytest.raises(ValueError, match="workspace"):
        plan.execute(**inputs, **outputs, workspace=workspace[:-1])
    with pytest.raises(ValueError, match="compiled descriptor"):
        plan.execute(**dict(inputs, grad_proj=inputs["grad_proj"][:, :24]), **outputs, workspace=workspace)


@pytest.mark.L1
def test_mhc_projection_wrapper(mhc_api):
    from cudnn import mhc_projection_backward

    inputs = _inputs()
    reference = _reference(inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    result = mhc_projection_backward(**inputs, allow_tf32=True, backend="frost", current_stream=stream)
    stream.synchronize()
    assert list(result.keys()) == ["dx", "dweight"]
    dx, dw = result
    assert dx is result["dx"] and dw is result["dweight"]
    _check(result, reference)
