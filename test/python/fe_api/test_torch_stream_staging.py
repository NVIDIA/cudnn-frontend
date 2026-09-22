# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R1 staging: a copy of a caller input made on an explicit launch stream must keep the
original's block from being reused on the caller's stream until the copy has run."""

import pytest
import torch

from cudnn._torch_stream import contiguous_on_stream, record_streams

_ROWS, _COLS = 1024, 2048  # 8 MiB fp32: one large-pool block, handed back whole to the next same-size request


def _delayed_copy_then_release(stage):
    """Queue a long kernel on a side stream, stage ``base.t()`` through ``stage(view, side)``, release
    ``base``, and fill the released block through a same-size allocation on the current stream.
    Returns ``(staged, expected, reused)`` after a full synchronize. With the original recorded, the
    allocator normally defers the block's reuse until the copy has run (``reused`` False), or waits for
    it before handing the block out; either way the copy is intact."""
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # no other cached block of this size: the released one is the only candidate for reuse
    base = torch.arange(_ROWS * _COLS, dtype=torch.float32, device="cuda").view(_ROWS, _COLS)
    expected = base.t().clone()
    torch.cuda.synchronize()
    released_ptr = base.data_ptr()
    with torch.cuda.stream(side):
        torch.cuda._sleep(1_000_000_000)  # the staging copy queues behind this
    staged = stage(base.t(), side)
    assert staged.is_contiguous() and staged.data_ptr() != released_ptr
    del base  # the caller releases its reference while the copy is still pending
    poison = torch.empty(_ROWS, _COLS, dtype=torch.float32, device="cuda")
    reused = poison.data_ptr() == released_ptr
    poison.fill_(-1.0)
    torch.cuda.synchronize()
    return staged, expected, reused


@pytest.mark.L0
def test_contiguous_on_stream_keeps_the_released_original_alive_until_the_copy_runs():
    staged, expected, _ = _delayed_copy_then_release(lambda view, side: contiguous_on_stream(view, side.cuda_stream, view.device))
    assert torch.equal(staged, expected)


@pytest.mark.L0
def test_record_streams_then_copy_keeps_the_released_original_alive():
    def stage(view, side):
        record_streams((view,), side, view.device)
        with torch.cuda.stream(side):
            return view.to(torch.float32).contiguous()

    staged, expected, _ = _delayed_copy_then_release(stage)
    assert torch.equal(staged, expected)


@pytest.mark.L0
def test_bare_contiguous_under_a_side_stream_context_reads_the_reused_block():
    """Control for the detector above: the same sequence without the recording observes the poison,
    so a green run of the two tests above is evidence, not luck."""

    def stage(view, side):
        with torch.cuda.stream(side):
            return view.contiguous()

    staged, expected, reused = _delayed_copy_then_release(stage)
    if not reused:
        pytest.skip("the caching allocator did not hand the released block to the next same-size allocation")
    assert not torch.equal(staged, expected)


@pytest.mark.L0
def test_staging_helpers_pass_through_and_noop_on_the_current_stream():
    t = torch.empty(4, 8, device="cuda")
    assert contiguous_on_stream(None, None) is None
    assert contiguous_on_stream(t, None) is t
    assert contiguous_on_stream(t, torch.cuda.current_stream().cuda_stream, t.device) is t
    record_streams((None, t), None)  # no stream: nothing to record
    record_streams((t,), torch.cuda.current_stream(), t.device)  # current stream: the allocation stream orders reuse
    copy = contiguous_on_stream(t.t(), None)
    assert copy.is_contiguous() and copy.shape == (8, 4)
