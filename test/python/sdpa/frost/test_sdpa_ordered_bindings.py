# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect shared ordered-binding smoke cases in the FROST SDPA CI target."""

import cudnn
import pytest

import core.graph.test_execute_overload as ordered_cases
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.fixture
def frost_attention_case(cudnn_handle):
    original_stream = cudnn.get_stream(cudnn_handle)
    try:
        graph, tensors = ordered_cases._graph(cudnn_handle, "frost")
        yield graph, tensors, cudnn_handle, "frost"
    finally:
        cudnn.set_stream(cudnn_handle, original_stream)


def test_frost_ordered_matches_mapping_with_fresh_buffers_and_uid_order(frost_attention_case):
    ordered_cases.test_ordered_matches_mapping_with_fresh_buffers_and_uid_order(frost_attention_case)


def test_frost_ordered_overrides_follow_mutable_values(frost_attention_case):
    ordered_cases.test_ordered_overrides_follow_mutable_values(frost_attention_case)


@pytest.mark.parametrize("ordered", [False, True])
def test_python_workspace_device_is_checked_before_launch(frost_attention_case, monkeypatch, ordered):
    from types import SimpleNamespace

    import torch

    graph, tensors, handle, _ = frost_attention_case
    buffers = ordered_cases._buffers(seed=31)
    workspace = ordered_cases._workspace(graph, handle)
    assert graph.get_workspace_size() > 0
    values, uids = ordered_cases._ordered(tensors, buffers)

    class PythonBuffer:
        def __init__(self, tensor):
            self.tensor = tensor
            self.shape, self.dtype, self.device = tensor.shape, tensor.dtype, tensor.device

        def data_ptr(self):
            return self.tensor.data_ptr()

        def stride(self):
            return self.tensor.stride()

        def element_size(self):
            return self.tensor.element_size()

    producer = PythonBuffer(workspace)

    def call():
        if ordered:
            graph.execute(values, producer, handle=handle, tensor_uids=uids)
        else:
            graph.execute(ordered_cases._mapping(tensors, buffers), producer, handle=handle)

    call()
    torch.cuda.synchronize()
    ordered_cases._assert_result(buffers)
    plan = graph._compiled_plans[graph._plan_index]
    # Record a would-be launch instead of executing an invalid device pointer.
    # This keeps the RED control safe even on a single-GPU CI worker.
    launches = []
    with monkeypatch.context() as patch:
        patch.setattr(plan._prepared, "execute", lambda *args: launches.append(args))
        # An absent declaration retains the raw-pointer caller contract.
        del producer.device
        call()
        assert len(launches) == 1
        launches.clear()
        for device in (
            "cpu",
            torch.device("cpu"),
            torch.device("cuda", workspace.device.index + 1),
            SimpleNamespace(type="cuda"),
            SimpleNamespace(index=workspace.device.index),
            None,
        ):
            producer.device = device
            with pytest.raises(ValueError, match="workspace must be on CUDA device"):
                call()
            assert not launches
    producer.device = workspace.device
    buffers["o"].fill_(float("nan"))
    buffers["lse"].fill_(float("nan"))
    call()
    torch.cuda.synchronize()
    ordered_cases._assert_result(buffers)


@pytest.mark.parametrize("layout", ["dense", "thd"])
def test_frost_torch_wrapper_ordered_bindings_rebind_and_replay(layout, monkeypatch):
    from sdpa.torch.test_ordered_bindings import test_torch_ordered_bindings_rebind_and_replay

    test_torch_ordered_bindings_rebind_and_replay("frost", layout, monkeypatch)
