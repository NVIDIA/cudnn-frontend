# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sm90 KDA marshal never hands a default-stream sentinel to ``torch.cuda.ExternalStream``,
and never allocates: conversions land in the staging the plan carved from the workspace.

On torch <= 2.12 and some 2.13 nightlies ``ExternalStream(0)`` is a fresh non-blocking
pool stream, so the dtype-conversion copies and the staged-output write-back issued in it
raced the kernel launched on ``CUstream(0)`` (qa sm90 nightly reds, load-dependent).
"""

import pytest
import torch

import cudnn  # noqa: F401 -- import-order requirement, see test/python/conftest.py
from cudnn._compiled_module import make_operand_buffer
from cudnn.frost.buffers import DTYPES
from cudnn.linear_attention.hopper import marshal

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

_NAMES = {torch.float32: "float32", torch.bfloat16: "bfloat16", torch.int64: "int64", torch.int32: "int32"}


def _view(t: torch.Tensor):
    code, bits = DTYPES[_NAMES[t.dtype]]
    return make_operand_buffer(int(t.data_ptr()), list(t.shape), code, bits, t.device.index)


@pytest.fixture
def no_external_stream(monkeypatch):
    def refuse(handle, device=None):
        raise AssertionError(f"torch.cuda.ExternalStream({handle}) built for a default-stream handle")

    monkeypatch.setattr(torch.cuda, "ExternalStream", refuse)


@pytest.mark.parametrize("handle", [0, 1, 2, "default"], ids=["null", "legacy", "per_thread", "torch_default"])
def test_default_stream_handles_run_on_torch_default_stream(no_external_stream, handle):
    default = torch.cuda.default_stream()
    if handle == "default":
        handle = default.cuda_stream
    with marshal.stream_ctx(handle):
        assert torch.cuda.current_stream().cuda_stream == default.cuda_stream


def test_side_stream_handle_selects_that_stream():
    side = torch.cuda.Stream()
    with marshal.stream_ctx(side.cuda_stream):
        assert torch.cuda.current_stream().cuda_stream == side.cuda_stream
    with torch.cuda.stream(side), marshal.stream_ctx(side.cuda_stream):
        assert torch.cuda.current_stream() == side
    assert torch.cuda.current_stream() == torch.cuda.default_stream()


def test_resolve_inputs_converts_into_staging_on_the_default_stream(no_external_stream):
    g = torch.randn(3, 8, device="cuda").to(torch.bfloat16)
    cu = torch.tensor([0, 5, 8], device="cuda", dtype=torch.int64)
    g_stage = torch.full((3, 8), 9.0, device="cuda", dtype=torch.float32)
    cu_stage = torch.full((3,), 9, device="cuda", dtype=torch.int32)
    staging = {"g": _view(g_stage), "cu_seqlens": _view(cu_stage)}
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]  # the counter: memory_allocated() also moves on unrelated frees
    addr = marshal.resolve_inputs(["g", "cu_seqlens"], [_view(g), _view(cu)], "test", 0, want={"g": "float32", "cu_seqlens": "int32"}, staging=staging)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations, "a conversion allocated instead of using its staging carve"
    assert [addr["g"], addr["cu_seqlens"]] == [g_stage.data_ptr(), cu_stage.data_ptr()]
    assert torch.equal(g_stage, g.float())
    assert torch.equal(cu_stage, cu.to(torch.int32))


def test_resolve_inputs_without_staging_raises_naming_the_port():
    """A port the graph declared native has no staging; a mismatched buffer is refused, not adapted."""
    cu = torch.tensor([0, 5, 8], device="cuda", dtype=torch.int64)
    with pytest.raises(NotImplementedError, match="'cu_seqlens'.*declared it packed and int32.*reserved no staging"):
        marshal.resolve_inputs(["cu_seqlens"], [_view(cu)], "test", 0, want={"cu_seqlens": "int32"})
    wide = torch.zeros(4, 6, 8, device="cuda", dtype=torch.bfloat16)
    padded = wide[:, :3]
    with pytest.raises(NotImplementedError, match="'q'.*declared it packed, so the plan reserved no staging"):
        marshal.resolve_inputs(["q"], [padded], "test", 0)


def test_staged_output_writes_back_on_the_default_stream(no_external_stream):
    final_state = torch.zeros(2, 4, 16, 16, device="cuda", dtype=torch.bfloat16)
    stage = torch.empty(2, 4, 16, 16, device="cuda", dtype=torch.float32)
    ptr, staged, destination = marshal.stage_output(_view(final_state), "float32", 0, _view(stage), port="final_state")
    assert staged is not None and ptr == staged.data_ptr() == stage.data_ptr() and staged.dtype == torch.float32
    staged.copy_(torch.randn_like(staged))
    marshal.write_back(staged, destination, 0)
    torch.cuda.synchronize()
    assert torch.equal(final_state, stage.to(torch.bfloat16))
    with pytest.raises(NotImplementedError, match="'final_state'.*reserved no float32 staging"):
        marshal.stage_output(_view(final_state), "float32", 0, None, port="final_state")
