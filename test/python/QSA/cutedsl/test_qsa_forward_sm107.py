# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native 64-key forward: packed masks, saved state, bounded storage and replay."""

import pytest
import torch
from cuda.bindings import driver as cuda

from cudnn.ops import qsa_backward, qsa_forward, qsa_indexed_forward
from QSA.cutedsl.test_qsa import make_inputs, reference, assert_close, supported_gpu

pytestmark = pytest.mark.L0


@pytest.fixture
def rubin(supported_gpu):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("The 64-key forward requires SM107")


def native_forward(inputs):
    import cutlass
    from cudnn.QSA.api import _compact
    from cudnn.QSA.kernels import fprop_n64_sm107

    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    ids, counts = _compact(inputs["sel_blocks"], inputs["lo"], inputs["hi"], inputs["block_first"], inputs["block_last"], 107, stream)
    outputs = (torch.empty_like(q), torch.empty_like(q, dtype=torch.float32), torch.empty((q.shape[1], q.shape[0]), device=q.device))
    kernel = fprop_n64_sm107.compile_kernel(cutlass.BFloat16, q.shape[1], k.shape[1], 107)
    fprop_n64_sm107.launch(
        kernel, q, k, v, *outputs, ids, counts, inputs["lo"], inputs["hi"], inputs["block_first"], inputs["block_last"], inputs["scale"], stream
    )
    return outputs


@pytest.mark.parametrize("heads", [(12, 1), (24, 2)])
@pytest.mark.parametrize("lengths", [(1,), (3, 5, 9), (61, 68), (127, 129, 5)])
@pytest.mark.parametrize("scale", [1 / 16, 0.0, -0.125])
def test_native_forward_and_saved_state(rubin, heads, lengths, scale):
    inputs = make_inputs(lengths, *heads, empty=True)
    inputs["scale"] = scale
    out, saved, lse = native_forward(inputs)
    for actual, expected in zip((out, saved, lse), reference(inputs)):
        assert_close(actual, expected)
    grad = inputs["q"].clone()
    grads = qsa_backward(grad_out=grad, out_fp32=saved, lse=lse, **inputs)
    for actual, expected in zip(grads, reference(inputs, grad, saved, lse)):
        assert_close(actual, expected)


def test_native_forward_masks_nonfinite_kv(rubin):
    inputs = make_inputs((3, 65), 12, 1, all_selected=True)
    expected = native_forward(inputs)
    torch.cuda.synchronize()
    inputs["k"][3:] = torch.nan
    inputs["v"][3:] = torch.inf
    actual = native_forward(inputs)
    for got, ref in zip(actual[:2], expected[:2]):
        assert_close(got[:3], ref[:3])
    assert_close(actual[2][:, :3], expected[2][:, :3])


@pytest.mark.parametrize("lengths", [(32768,), (16385, 16386)])
def test_bitmap_long_public_dispatch(rubin, lengths):
    """Exercise the public long-row launcher with aligned and packed bitmaps."""
    total = sum(lengths)
    torch.manual_seed(92)
    q = torch.randn((total, 12, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total, 1, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    first, last, origins, bases = [], [], [], []
    origin, base = 0, 0
    for length in lengths:
        starts = list(range(origin, origin + length, 4))
        first.extend(starts)
        last.extend(min(start + 3, origin + length - 1) for start in starts)
        origins.extend([origin] * length)
        bases.extend([base] * length)
        origin += length
        base += len(starts)
    first, last, origins, bases = [torch.tensor(x, device="cuda", dtype=torch.int32) for x in (first, last, origins, bases)]
    rows = torch.arange(total, device="cuda", dtype=torch.int32)
    sel = torch.zeros((total, base), device="cuda", dtype=torch.uint8)
    current = bases + (rows - origins) // 4
    for delta in range(4):
        sel[rows.long(), (current - delta).clamp_min(0).long()] = 255
    # Empty rows and a row above the canonical top-512 cap exercise general bitmap semantics.
    sel[::7] = 0
    sel[-1] = 1
    outputs = qsa_forward(q, k, v, sel, origins, rows, bases, origins, first, last, 1 / 16)
    for row in sorted(
        {
            0,
            1,
            7,
            63,
            64,
            lengths[0] - 1,
            total - 2,
            total - 1,
            min(lengths[0], total - 1),
        }
    ):
        tokens = torch.arange(total, device="cuda")
        blocks = bases[row] + (tokens - origins[row]) // 4
        valid = (tokens >= origins[row]) & (tokens <= row)
        valid &= sel[row, blocks.clamp(0, base - 1).long()] != 0
        keys = tokens[valid]
        if keys.numel() == 0:
            assert torch.count_nonzero(outputs["out"][row]) == 0
            assert torch.count_nonzero(outputs["out_fp32"][row]) == 0
            assert torch.isneginf(outputs["lse"][:, row]).all()
            continue
        scores = (q[row].float() @ k[keys, 0].float().T) / 16
        expected = scores.softmax(-1) @ v[keys, 0].float()
        assert_close(outputs["out"][row], expected.to(q.dtype))
        assert_close(outputs["out_fp32"][row], expected)
        assert_close(outputs["lse"][:, row], scores.logsumexp(-1))


def test_indexed_long_packed_workspace_and_replay(rubin, monkeypatch):
    from cudnn.QSA.indexed.api import _plan

    total, boundary, width = 32771, 16385, 20
    torch.manual_seed(91)
    q = torch.randn((total, 12, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total, 1, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    rows = torch.arange(total, device="cuda", dtype=torch.int32)
    origins = torch.where(rows < boundary, 0, boundary).to(torch.int32)
    # Two padding holes and non-monotonic groups; each sequence has an unaligned tail.
    relative = (rows - origins) // 4
    groups = relative[:, None] + torch.tensor([-1, 1, -2, 0, 1], device="cuda", dtype=torch.int32)
    tokens = origins[:, None, None] + groups[:, :, None] * 4 + torch.arange(4, device="cuda", dtype=torch.int32)
    indices = torch.where((groups[:, :, None] >= 0) & (tokens <= rows[:, None, None]), tokens, -1).reshape(total, width)
    original = indices.clone()
    outputs = qsa_indexed_forward(q, k, v, indices)
    plan = _plan(total, width, q.device, "fprop")
    # The scratch contract scales with selected groups, not the sequence squared.
    assert plan.scratch_workspace_bytes() <= total * (width + 24) + 6 * 256
    workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)
    selected_rows = torch.tensor([0, 1, 63, 64, boundary - 1, boundary, boundary + 1, total - 1], device="cuda")

    def check():
        for row in selected_rows.tolist():
            keys = indices[row]
            keys = keys[keys >= 0].long()
            scores = (q[row].float()[:, None, :] * k[keys, 0].float()[None, :, :]).sum(-1) / 16
            expected = (scores.softmax(-1)[:, :, None] * v[keys, 0].float()[None, :, :]).sum(1)
            assert_close(outputs["out_fp32"][row], expected)
            assert_close(outputs["lse"][:, row], scores.logsumexp(-1))

    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()

    def forbidden(*args, **kwargs):
        raise AssertionError("execute allocated, converted, or synchronized")

    try:
        with torch.cuda.graph(graph, stream=stream):
            with monkeypatch.context() as patch:
                for name in ("empty", "empty_like", "zeros", "zeros_like", "arange"):
                    patch.setattr(torch, name, forbidden)
                patch.setattr(torch.cuda, "synchronize", forbidden)
                for name in ("cpu", "to", "item", "tolist", "numpy"):
                    patch.setattr(torch.Tensor, name, forbidden)
                plan.execute(q, k, v, indices, tuple(outputs), 1 / 16, workspace, current_stream=stream.cuda_stream)
        indices.fill_(-1)
        for tensor in outputs:
            tensor.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(outputs["out"]) == 0
        assert torch.count_nonzero(outputs["out_fp32"]) == 0
        assert torch.isneginf(outputs["lse"]).all()
        indices.copy_(original)
        q.mul_(-1)
        graph.replay()
        torch.cuda.synchronize()
        check()
    finally:
        graph.reset()


@pytest.mark.parametrize("lengths,next_lengths", [((3, 5, 9), (65, 68)), ((63, 66), (129, 131))])
def test_native_forward_reuses_dynamic_artifact(rubin, monkeypatch, lengths, next_lengths):
    import cutlass.cute as cute

    native_forward(make_inputs(lengths, 12, 1, empty=True))

    def forbidden(*args, **kwargs):
        raise AssertionError("a changed token count recompiled the forward kernel")

    inputs = make_inputs(next_lengths, 12, 1, empty=True)
    with monkeypatch.context() as patch:
        patch.setattr(cute, "compile", forbidden)
        actual = native_forward(inputs)
    for got, expected in zip(actual, reference(inputs)):
        assert_close(got, expected)
