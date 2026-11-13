# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Zero-weight split payloads cannot affect the merged output or statistics."""

import math

import pytest
import torch

from cudnn.frost.compiled_cache import positional_entry
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot


@pytest.mark.parametrize("form", ["grid", "walk"])
@pytest.mark.parametrize("kind,splits", [("local", 16), ("chain", 32), ("round", 129)])
@pytest.mark.parametrize("payload", [float("nan"), float("inf"), -float("inf")], ids=["nan", "posinf", "neginf"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
def test_zero_weight_partial_and_chain_are_excluded(form, kind, splits, payload, dtype):
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    op = torch.ones((splits, 1, 1, 128), device="cuda")
    lp = torch.zeros((splits, 1, 1), device="cuda")
    ignored = 1 if kind == "local" else 16 if kind == "chain" else splits - 1
    lp[:ignored] = -2048  # Weight is exactly zero even in the FP64 reference.
    op[:ignored] = payload
    o = torch.empty((1, 1, 1, 128), dtype=dtype, device="cuda")
    lse = torch.empty((1, 1, 1), device="cuda")
    code = comb.DECODE_GRID_ON if form == "grid" else comb.DECODE_GRID_LEAN_WALK
    fn = positional_entry(comb.compile_ptr(dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=True, decode_grid=code))
    fn(op.data_ptr(), lp.data_ptr(), o.data_ptr(), lse.data_ptr(), (1, 1, 1, 128), splits, o.stride(), lse.stride(), torch.cuda.current_stream().cuda_stream)
    torch.testing.assert_close(o, torch.ones_like(o), atol=0, rtol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(splits - ignored)), atol=2e-6, rtol=2e-6)


def _fixture(splits):
    # Exact eighths avoid making the determinism input depend on GPU Philox layout.
    op = (torch.arange(splits * 3 * 2 * 2 * 128).reshape(splits, 3, 2, 2, 128) % 17).float().cuda() / 8
    lp = torch.zeros((splits, 3, 2, 2), device="cuda")
    ignored = 16 if splits == 32 else 128
    op[:ignored] = float("nan")
    lp[:ignored] = -2048
    # The second query of head 1 is entirely empty, independently in each batch.
    op[:, :, 1, 1] = float("nan")
    lp[:, :, 1, 1] = -float("inf")
    return op, lp, ignored


def _output(op, dtype):
    _, b, sq, h, d = op.shape
    return {
        "o": torch.full((b, sq, h, d), float("nan"), dtype=dtype, device="cuda"),
        "lse": torch.full((b, h, sq), float("nan"), device="cuda"),
    }


def _invoke(fn, op, lp, out):
    splits, b, sq, h, d = op.shape
    fn(
        op.data_ptr(),
        lp.data_ptr(),
        out["o"].data_ptr(),
        out["lse"].data_ptr(),
        (b, h, sq, d),
        splits,
        out["o"].stride(),
        out["lse"].stride(),
        torch.cuda.current_stream().cuda_stream,
    )
    return out


def _reference(op, lp, out):
    scores = lp.double()
    maximum = scores.amax(dim=0)
    weights = torch.exp(scores - torch.where(torch.isfinite(maximum), maximum, 0))
    weights_o = weights.permute(0, 1, 3, 2).unsqueeze(-1)
    safe_partial = torch.where(weights_o > 0, op.double(), 0)
    denominator = weights.sum(dim=0)
    expected = (safe_partial * weights_o).sum(dim=0) / denominator.permute(0, 2, 1).unsqueeze(-1).clamp_min(1)
    # This fixture's positive weights are one and its live counts are powers of two.
    torch.testing.assert_close(out["o"], expected.to(out["o"].dtype), atol=0, rtol=0)
    torch.testing.assert_close(out["lse"].double(), maximum + denominator.log(), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("form", ["grid", "walk"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("splits", [32, 129], ids=["chain32", "round129"])
def test_zero_weight_replay_and_batch_invariance(form, dtype, splits):
    """Forward combine only: no backward API, producer, or dispatch-policy claim."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    code = comb.DECODE_GRID_ON if form == "grid" else comb.DECODE_GRID_LEAN_WALK
    tag = "f16" if dtype == torch.float16 else "bf16"
    fn = positional_entry(comb.compile_ptr(dtype_o=tag, has_lse=True, decode_grid=code))
    op, lp, ignored = _fixture(splits)
    saved = {"op": op.clone(), "lp": lp.clone()}
    target_op, target_lp = op[:, 1:2].contiguous(), lp[:, 1:2].contiguous()

    def run_target():
        out = _invoke(fn, target_op, target_lp, _output(target_op, dtype))
        _reference(target_op, target_lp, out)
        return out

    expected = assert_replays(run_target)
    # Reorder, duplicate, remove and change neighboring sequences without changing target data.
    for indices in ([0, 1, 2], [2, 1, 0], [1, 2, 1, 0]):
        packed_op, packed_lp = op[:, indices].contiguous(), lp[:, indices].contiguous()
        for change_neighbors in (False, True):
            if change_neighbors:
                for batch, source in enumerate(indices):
                    if source != 1:
                        packed_op[:, batch].add_(0.25)
            out = _invoke(fn, packed_op, packed_lp, _output(packed_op, dtype))
            _reference(packed_op, packed_lp, out)
            for batch, source in enumerate(indices):
                if source == 1:
                    assert_bitwise({name: value[batch : batch + 1] for name, value in out.items()}, expected, f"{indices}, {change_neighbors}")

    # More blocks than SMs exercise multiple scheduling waves with the same row arithmetic.
    count = torch.cuda.get_device_properties(0).multi_processor_count + 3
    large_op = target_op.repeat(1, count, 1, 1, 1)
    large_lp = target_lp.repeat(1, count, 1, 1)
    large_out = _invoke(fn, large_op, large_lp, _output(large_op, dtype))
    _reference(large_op, large_lp, large_out)
    assert_bitwise(large_out, {name: value.repeat(count, *([1] * (value.ndim - 1))) for name, value in expected.items()}, "multiple waves")

    out = _output(op, dtype)
    _invoke(fn, op, lp, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _invoke(fn, op, lp, out)
    changed = saved["op"].clone()
    changed[ignored:].add_(0.25)
    for current in (changed, saved["op"]):
        op.copy_(current)
        for value in out.values():
            value.fill_(float("nan"))
        graph.replay()
        _reference(op, lp, out)
        snapshot(out)
    assert_bitwise({"op": op, "lp": lp}, saved, "inputs restored and unmodified")
    assert_bitwise({name: value[1:2] for name, value in out.items()}, expected, "restored capture")
    assert_bitwise({"op": target_op, "lp": target_lp}, {"op": saved["op"][:, 1:2], "lp": saved["lp"][:, 1:2]}, "target inputs unmodified")
    if splits == 129:
        assert_golden(f"sdpa-combine-zero-weight-{form}-{tag}", {"op": target_op, "lp": target_lp}, expected)
