# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the indexed benchmark oracle against dense PyTorch autograd on CPU."""

import pytest
import torch

from benchmark.qsa import reference

pytestmark = pytest.mark.L0


def test_indexed_reference_packed_tail_padding_and_empty_rows():
    generator = torch.Generator().manual_seed(91)
    total, heads, dim = 9, 12, 8
    q = torch.randn(total, heads, dim, generator=generator).bfloat16()
    k = torch.randn(total, 1, dim, generator=generator).bfloat16()
    v = torch.randn(total, 1, dim, generator=generator).bfloat16()
    grad = torch.randn(q.shape, generator=generator).bfloat16()
    indices = torch.full((total, 12), -1, dtype=torch.int32)
    # The second document starts at unaligned token 3. Keep gaps and empty rows.
    for row in range(total):
        origin = 0 if row < 3 else 3
        if row not in (1, 6):
            visible = torch.arange(origin, row + 1)
            indices[row, -len(visible) :] = visible
    origins = torch.tensor([0] * 3 + [3] * 6, dtype=torch.int32)
    args = dict(
        q=q,
        k=k,
        v=v,
        sel_blocks=None,
        lo=origins,
        hi=torch.arange(total),
        blk_base=None,
        tok_base=origins,
        block_first=None,
        block_last=None,
        scale=dim**-0.5,
        block_size=4,
        selection_indices=indices,
    )
    actual_fwd = reference.forward_reference(**args)
    qf, kf, vf = [x.float().detach().requires_grad_() for x in (q, k, v)]
    mask = torch.zeros(total, total, dtype=torch.bool)
    for row in range(total):
        mask[row, indices[row][indices[row] >= 0].long()] = True
    live = mask.any(-1)
    scores = torch.einsum("thd,shd->hts", qf, kf.expand(-1, heads, -1)) * args["scale"]
    scores = scores.masked_fill(~mask[None], -torch.inf)
    expected_lse = scores.logsumexp(-1)
    probabilities = torch.where(live[None, :, None], scores, 0).softmax(-1).masked_fill(~mask[None], 0)
    expected_out = torch.einsum("hts,shd->thd", probabilities, vf.expand(-1, heads, -1))
    torch.testing.assert_close(actual_fwd[1], expected_out)
    torch.testing.assert_close(actual_fwd[2], expected_lse)
    expected_out.backward(grad.float())
    actual_bwd = reference.backward_reference(**args, grad_out=grad, out_fp32=actual_fwd[1], lse=actual_fwd[2])
    for actual, expected in zip(actual_bwd, (qf.grad, kf.grad, vf.grad)):
        torch.testing.assert_close(actual, expected.bfloat16(), atol=0.004, rtol=0.01)
