# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdpa.fp16_ref import compute_ref_backward


@pytest.mark.L0
@pytest.mark.parametrize("score", [-256.0, -128.0, 128.0, 256.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_backward_reference_large_equal_scores(score, dtype):
    """Equal logits have exact P=1/2, even beside a half rounding boundary."""
    q = torch.tensor([[[[score]]]], device="cuda")
    k = torch.ones((1, 1, 2, 1), device="cuda")
    step = torch.finfo(dtype).eps
    v = torch.tensor([[[[1.0 + step], [-1.0]]]], device="cuda")
    o = torch.full_like(q, step / 2)
    dq, dk, dv, _, _ = compute_ref_backward(q, k, v, o, torch.ones_like(o), torch_type=dtype)

    # dS=(V-O)/2 lands exactly halfway between adjacent values. The reference
    # deliberately rounds dS to the input type before its gradient GEMMs.
    ds = ((v.double() - o.double()) / 2).to(dtype).double()
    expected_dk = (ds * score).to(dtype).float()
    torch.testing.assert_close(dq, torch.zeros_like(dq), rtol=0, atol=0)
    torch.testing.assert_close(dk, expected_dk, rtol=0, atol=0)
    torch.testing.assert_close(dv, torch.full_like(dv, 0.5), rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.parametrize("score", [-256.0, -128.0, 128.0, 256.0])
def test_backward_reference_large_equal_scores_with_sink(score):
    """The sink and two equal KV logits each have probability exactly 1/3."""
    q = torch.tensor([[[[score]]]], device="cuda")
    k = torch.ones((1, 1, 2, 1), device="cuda")
    v = torch.ones_like(k)
    o = torch.full_like(q, 2 / 3)
    sink = torch.full_like(q, score)
    _, _, _, _, dsink = compute_ref_backward(q, k, v, o, torch.ones_like(o), sink_token=sink)
    expected = (-o.double() / 3).float()
    torch.testing.assert_close(dsink, expected, rtol=1e-7, atol=0)
