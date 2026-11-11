# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real SSA backward ownership and numerical contracts on SM103.

TopK exceeds the physical key extent, so the FP32 reference independently uses
every allowed key. It does not read SSA's selected IDs or mutable workspaces.
"""

import pytest

import torch

from cudnn.simplified_sparse_attention.cutedsl.backend import RowMetadata, sparse_attention

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="The packaged CuTe kernels require SM100 or SM103",
    ),
]


def _reference(inputs, starts, ends, valid, frozen_upstream):
    q, k, v = (tensor.detach().float().requires_grad_() for tensor in inputs[:3])
    key_ids = torch.arange(k.size(0), device=k.device)
    allowed = (key_ids[None, :] >= starts[:, None]) & (key_ids[None, :] < ends[:, None])
    # Empty padding rows receive a safe dummy key before softmax; their final
    # output is zero. This avoids NaN gradients from all-minus-infinity logits.
    safe_allowed = (valid[:, None] & allowed) | (~valid[:, None] & (key_ids[None, :] == 0))
    logits = torch.einsum("qhd,kd->hqk", q, k[:, 0]) / 16.0
    probabilities = logits.masked_fill(~safe_allowed[None, :, :], -torch.inf).softmax(-1)
    output = torch.einsum("hqk,kd->qhd", probabilities, v[:, 0])
    output = torch.where(valid[:, None, None], output, 0.0)
    gradients = torch.autograd.grad(output, (q, k, v), frozen_upstream.float())
    return output.detach(), gradients


def _assert_relative_l2(name, actual, expected):
    assert torch.isfinite(actual).all().item(), f"{name}: nonfinite result"
    assert torch.isfinite(expected).all().item(), f"{name}: nonfinite reference"
    reference_norm = expected.float().norm().item()
    assert reference_norm > 0, f"{name}: test fixture produced a zero reference"
    error = (actual.detach().float() - expected.float()).norm().item() / reference_norm
    assert error <= 2**-7, f"{name}: relative L2 error {error} exceeds {2 ** -7}"
    return error


@pytest.mark.parametrize("upstream_layout", ["contiguous", "strided"])
def test_packed_backward_preserves_reused_upstream_and_matches_fp32(upstream_layout, record_property):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(602103)
    tq, tk, heads, dim = 128, 256, 16, 256
    owner_shape = (tq + 2, heads, dim) if upstream_layout == "contiguous" else (tq, heads, 2 * dim)
    # Nonzero signed values on every row expose in-place masking; extra storage
    # detects writes outside the gradient view as well.
    owner = (2 * torch.randint(0, 2, owner_shape, device=device, generator=generator) - 1).to(torch.bfloat16)
    upstream = owner[1:-1] if upstream_layout == "contiguous" else owner[:, :, ::2]
    assert upstream.is_contiguous() == (upstream_layout == "contiguous")
    owner_before = owner.clone()
    frozen_upstream = upstream.detach().clone()
    rows = torch.arange(tq, device=device)
    first_valid = rows % 3 != 0
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # Reuse the exact caller tensor with changed operands and complementary
        # masks: rows masked on the first call contribute on the second call.
        for call, valid in enumerate((first_valid, ~first_valid)):
            shapes = ((tq, heads, dim), (tk, 1, dim), (tk, 1, dim), (tq, 128), (tk, 128))
            inputs = tuple((0.2 * torch.randn(shape, device=device, generator=generator)).to(torch.bfloat16).requires_grad_() for shape in shapes)
            starts = torch.where(rows % 2 == 0, 0, 64).to(torch.int32)
            ends = torch.where(valid, tk, starts).to(torch.int32)
            reference_output, reference_grads = _reference(inputs, starts, ends, valid, frozen_upstream)
            layout = RowMetadata(
                key_starts=starts,
                key_ends=ends,
                query_valid=valid,
                loss_denominator=valid.sum().float().reshape(1),
                triangular_scores=False,
                mask_invalid_rows=True,
            )
            output, loss = sparse_attention(*inputs, layout, 512, loss_coeff=0.0)
            gradients = torch.autograd.grad((output, loss), inputs, (upstream, torch.ones_like(loss)))
            torch.cuda.synchronize()

            assert torch.equal(owner, owner_before), "SSA backward changed caller gradient storage"
            assert torch.equal(upstream, frozen_upstream), "SSA backward changed incoming gradient"
            assert torch.count_nonzero(output[~valid]).item() == 0
            assert torch.count_nonzero(gradients[0][~valid]).item() == 0
            assert torch.isfinite(loss).item() and loss.item() == 0.0
            for name, gradient in zip(("q_idx", "k_idx"), gradients[3:]):
                assert torch.isfinite(gradient).all().item(), f"{name}: nonfinite gradient"
                assert torch.count_nonzero(gradient).item() == 0, f"{name}: loss_coeff=0 gradient"
            for name, actual, expected in zip(
                ("output", "dq", "dk", "dv"),
                (output, *gradients[:3]),
                (reference_output, *reference_grads),
            ):
                error = _assert_relative_l2(name, actual, expected)
                record_property(f"call{call}_{name}_relative_l2", error)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
