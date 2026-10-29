# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared architecture-entry checks for native dense data strides."""

from copy import copy
from dataclasses import replace
from itertools import permutations

import pytest
import torch


@pytest.mark.parametrize("family", ["half", "fp8", "mxfp8"])
@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("order", list(permutations(range(3))))
def check_native_dense_permutations(family, d, dv, order, monkeypatch):
    from sdpa.frost.native_binding_utils import compact_forward_kernel_count, forbid_python_forward_binding, intercept_native_forward
    from test_sdpa_bwd_staged_sm80 import _assert_kernel_count

    cc = torch.cuda.get_device_capability()
    if cc[0] not in (10, 12):
        pytest.skip("native dense layouts require SM100/SM107/SM120")
    if cc[0] == 12 and family != "mxfp8" and (d != dv or d > 384):
        pytest.skip("outside the SM120 head envelope")
    if family == "mxfp8":
        from native_forward_mxfp8_cases import _case, _check, _execute

        api, buffers, storage = _case(d, dv)
        check = lambda: _check(buffers, thd=False)
    elif cc[0] == 12:
        from native_forward_sm120_cases import _case, _check, _execute

        api, buffers, storage = _case(d, fp8=family == "fp8")
        check = lambda: _check(buffers, storage)
    elif family == "fp8":
        from native_forward_fp8_cases import _case, _check, _execute

        api, buffers, storage = _case(d, dv)
        check = lambda: _check(buffers, storage)
    else:
        from native_forward_half_cases import _case, _check, _execute

        api, buffers, storage = _case(d, dv)
        check = lambda: _check(buffers)

    allocations = {}
    api = copy(api)
    api._compiled_kernel = None
    api._is_supported = False
    for i, role in enumerate(("q", "k", "v", "o")):
        tensor = buffers[role]
        alignment = 16 // tensor.element_size()
        gap = (i + 1) * alignment
        strides = [0, 0, 0, 1]
        span = tensor.shape[-1] + gap
        physical = order[i % 3 :] + order[: i % 3]
        for axis in reversed(physical):
            strides[axis] = span
            span = tensor.shape[axis] * span + gap
        raw = torch.full((span + 2 * alignment,), 11 if role == "o" else float("nan"), device=tensor.device, dtype=tensor.dtype)
        view = raw.as_strided(tensor.shape, strides, alignment)
        view.copy_(tensor)
        padding = torch.ones(raw.shape, device=raw.device, dtype=torch.bool)
        padding.as_strided(view.shape, view.stride(), alignment).fill_(False)
        allocations[role] = (raw, padding)
        buffers[role], storage[role] = view, raw
        desc = getattr(api, role + "_desc")
        setattr(api, role + "_desc", replace(desc, stride=tuple(strides), stride_order=tuple(sorted(range(4), key=lambda axis: strides[axis]))))
    assert api.check_support()
    required = api.scratch_workspace_bytes()
    api.compile()
    assert api._dense_spec.native is not None and not hasattr(api, "_staged_spec")
    expected_kernels = compact_forward_kernel_count(api, buffers, _execute)
    forbid_python_forward_binding(monkeypatch)
    pointers = []

    def inspect(frame):
        for role in ("q", "k", "v", "o"):
            assert frame[role + "_ptr"] == buffers[role].data_ptr()
        pointers.append(frame["q_ptr"])

    intercept_native_forward(monkeypatch, api._dense_spec, inspect)
    workspace = torch.empty(required, dtype=torch.uint8, device=buffers["q"].device)
    for rebound in (False, True):
        if rebound:
            for role, (raw, padding) in allocations.items():
                fresh = raw.clone()
                old = buffers[role]
                buffers[role] = fresh.as_strided(old.shape, old.stride(), old.storage_offset())
                allocations[role], storage[role] = (fresh, padding), fresh
        _execute(api, buffers, workspace)
        check()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with torch.cuda.graph(graph):
                _execute(api, buffers, workspace)
            _assert_kernel_count(graph, expected_kernels)
            buffers["q"].copy_(buffers["q"].float().mul_(0.5).to(buffers["q"].dtype))
            buffers["o"].fill_(float("nan"))
            graph.replay()
            check()
            for role, (raw, padding) in allocations.items():
                values = raw.float()[padding]
                if role == "o":
                    assert torch.all(values == 11)
                else:
                    assert torch.isnan(values).all()
        finally:
            graph.reset()
    assert len(set(pointers)) == 2


@pytest.mark.parametrize("kind", ["bfloat16", "fp8", "mx"])
@pytest.mark.parametrize("order", list(permutations(range(3))))
def check_native_gate_permutations(kind, order, monkeypatch, cudnn_handle):
    import cudnn
    from sdpa.frost.native_binding_utils import forbid_python_forward_binding
    from native_gate_cases import _case

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("fused output gates require SM107")
    shape = (2, 4, 128, 256)
    strides, span = [0, 0, 0, 1], 264
    for axis in reversed(order):
        strides[axis] = span
        span = shape[axis] * span + 8
    raw = torch.full((span + 16,), 77, device="cuda", dtype=torch.bfloat16)
    gate = raw.as_strided(shape, strides, 8)
    gate.fill_(1)
    padding = torch.ones(raw.shape, device="cuda", dtype=torch.bool)
    padding.as_strided(shape, strides, 8).fill_(False)
    graph, bindings, workspace, buffers, tensors = _case(kind, gate)
    assert graph._compiled_plans[graph._plan_index]._prepared.spec.native is not None
    forbid_python_forward_binding(monkeypatch)

    def check():
        if kind in ("fp8", "mx"):
            from test_sdpa_prepared_quantized_gate import _helper

            _helper(kind == "mx")._check(buffers, thd=False)
        else:
            q = buffers["q"].double()
            k = buffers["k"].double().repeat_interleave(2, 1)
            v = buffers["v"].double().repeat_interleave(2, 1)
            scores = q @ k.transpose(-1, -2) / 16
            expected = (scores.softmax(-1) @ v) * buffers["gate"].double().sigmoid()
            torch.testing.assert_close(buffers["o"].double(), expected, atol=0.005, rtol=0.01)
            torch.testing.assert_close(buffers["lse"].double(), scores.logsumexp(-1), atol=0.005, rtol=0.01)

    stream = torch.cuda.Stream()
    previous_stream = cudnn.get_stream(cudnn_handle)
    try:
        for rebound in (False, True):
            if rebound:
                raw = raw.clone()
                gate = raw.as_strided(shape, strides, 8)
                bindings[tensors["gate"]] = buffers["gate"] = gate
            stream.wait_stream(torch.cuda.current_stream())
            cudnn.set_stream(cudnn_handle, stream.cuda_stream)
            captured = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.stream(stream):
                    graph.execute(bindings, workspace, handle=cudnn_handle)
                torch.cuda.current_stream().wait_stream(stream)
                check()
                ungated = {name: buffers[name].clone() for name in ("lse", "amax_o") if name in buffers}
                with torch.cuda.graph(captured, stream=stream):
                    graph.execute(bindings, workspace, handle=cudnn_handle)
                for value in (-1e4, 1e4):
                    gate.fill_(value)
                    captured.replay()
                    check()
                    for name, expected in ungated.items():
                        torch.testing.assert_close(buffers[name], expected, atol=0, rtol=0)
                    assert torch.all(raw[padding] == 77)
            finally:
                captured.reset()
    finally:
        cudnn.set_stream(cudnn_handle, previous_stream)
