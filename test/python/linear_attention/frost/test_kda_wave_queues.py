# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent-head queues preserve every task when prioritizing long pieces."""

from itertools import accumulate

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest
import torch

from cudnn.linear_attention.frost.kernel.kda_chain_backward_f16 import head_wave_items
from linear_attention.test_la import frost_la_unavailable

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@cute.jit
def _emit_wave(divisor: cutlass.Constexpr[int], phase: cutlass.Int32, heads: cutlass.Int32, cu, rows, source_count, dest, count, series, backward, stream):
    head_wave_items(divisor, 16, phase, heads, cu, rows, source_count, dest, count, series, backward).launch(
        grid=(1, 1, 1), block=(256, 1, 1), stream=stream, use_pdl=True
    )


@pytest.mark.parametrize("divisor", [2, 4])
def test_wave_queue_identity_and_partition(divisor):
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    heads = 8
    compiled = None
    # Empty identities, both sides of the 512-token threshold, several CTA
    # scan rounds, partial rounds, and homogeneous buckets.
    for lengths in ([0], [0, 511, 512, 513] * 19, [64] * 257 + [4097] * 3, [17] * 65, [512] * 65):
        offsets = [0, *accumulate(lengths)]
        n = len(lengths) * heads // divisor
        cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        rows = torch.arange(len(lengths) + 1, dtype=torch.int32, device="cuda") * heads
        original_rows = rows.clone()
        source_count = rows[-1:]
        storage = torch.full((n + 8, 10), -12345, dtype=torch.int32, device="cuda")
        dest = storage[:n]
        count = torch.empty(1, dtype=torch.int32, device="cuda")
        series = torch.empty(2, dtype=torch.int32, device="cuda")
        backward = torch.empty_like(series)
        tensors = (cu, rows, source_count, dest, count, series, backward)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if compiled is None:
            placeholders = [from_dlpack(t).mark_layout_dynamic() for t in tensors]
            compiled = cute.compile(_emit_wave, divisor, cutlass.Int32(0), cutlass.Int32(heads), *placeholders, stream)
        for phase in range(divisor):
            dest.fill_(-12345)
            count.fill_(-1)
            series.fill_(-1)
            backward.fill_(-1)
            compiled(cutlass.Int32(phase), cutlass.Int32(heads), *tensors, stream)
            actual = dest.cpu().tolist()
            expected = []
            for slot, length in enumerate(lengths):
                chunks = (length + 15) // 16
                for head in range(phase, heads, divisor):
                    expected.append([slot, head, 0, chunks, 0, chunks, offsets[slot], offsets[slot + 1], slot, slot])
            assert sorted(actual) == sorted(expected), "queue dropped, duplicated or modified a task"
            buckets = [int(row[7] - row[6] >= 512) for row in actual]
            assert buckets == sorted(buckets, reverse=True), "long tasks must precede short tasks"
            assert count.item() == n
            assert torch.equal(rows, original_rows), "source row/count table was overwritten"
            assert (series == 0).all() and (backward == 0).all()
            assert (storage[n:] == -12345).all(), "queue wrote beyond capacity"
