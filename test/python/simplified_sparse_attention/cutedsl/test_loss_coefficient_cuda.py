# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare runtime auxiliary coefficients with a PyTorch objective on SM103."""

import pytest

import torch

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="The packaged CuTe kernels require SM100 or SM103",
    ),
]


@pytest.mark.parametrize("spmm_impl", ["tile", "gather"])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("per_token_loss", [False, True])
def test_runtime_loss_coefficient_matches_reference(spmm_impl, topk, per_token_loss):
    from cudnn.simplified_sparse_attention.cutedsl.kernels import indexer_aux_grad_precompute_tcgen05 as aux

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(812)
    tq, tk, dim = 64, 4096, 128
    q = torch.randn((tq, dim), device=device, dtype=torch.bfloat16, generator=generator)
    k = torch.randn((tk, dim), device=device, dtype=torch.bfloat16, generator=generator)
    # Production indices are unique and sorted by key position. Include rows
    # with partial support and empty rows to exercise masking as well.
    indices = torch.rand((tq, tk), device=device, generator=generator).topk(topk).indices
    indices = indices.sort(dim=1).values.to(torch.int32).contiguous()
    lengths = torch.full((tq,), topk, device=device, dtype=torch.int32)
    lengths[::7] = topk // 3
    lengths[0] = 0
    valid = torch.arange(topk, device=device)[None, :] < lengths[:, None]

    q_ref = q.float().requires_grad_()
    k_ref = k.float().requires_grad_()
    scores = (q_ref @ k_ref.T * aux.SCORE_SCALE).gather(1, indices.long())
    predict = scores.masked_fill(~valid, -1e9).softmax(dim=-1).masked_fill(~valid, 0.0)
    teacher_logits = torch.randn((tq, topk), device=device, generator=generator)
    teacher = teacher_logits.masked_fill(~valid, -1e9).softmax(dim=-1).masked_fill(~valid, 0.0)
    unit_loss = (teacher * ((teacher + aux.EPS).log() - (predict + aux.EPS).log())).sum()
    ref_dq, ref_dk = torch.autograd.grad(unit_loss, (q_ref, k_ref))

    denominator_value = 1.0 if per_token_loss else float(tq)
    denominator = torch.tensor([denominator_value], device=device)
    loss = torch.empty((), device=device)
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    compiled_keys = None
    for coefficient in (0.0, 0.03, 0.1, 0.2):
        # Changing the coefficient must reuse the compiled executable. Starting
        # with zero also catches accidentally baking the first value into it.
        aux.run(
            q,
            k,
            indices,
            lengths,
            predict.detach(),
            teacher,
            loss,
            dq,
            dk,
            denominator,
            loss_coeff=coefficient,
            rectangular=True,
            spmm_impl=spmm_impl,
        )
        current_keys = set(aux._compiled)
        if compiled_keys is None:
            compiled_keys = current_keys
        else:
            assert current_keys == compiled_keys
        scale = coefficient / denominator_value
        torch.testing.assert_close(loss, unit_loss.detach() * scale, rtol=2e-5, atol=1e-7)
        for actual, reference in ((dq, ref_dq), (dk, ref_dk)):
            expected = (reference * scale).to(torch.bfloat16)
            if coefficient == 0.0:
                assert torch.count_nonzero(actual).item() == 0
            else:
                # BF16 storage quantizes the accumulated result. Check both
                # relative RMS error and an elementwise bound, including tails.
                error = (actual.float() - expected.float()).norm() / expected.float().norm()
                assert error.item() < 0.01
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0.02,
                    atol=0.005 * expected.float().abs().max().item(),
                )

    # Existing callers omitting the new keyword must preserve lambda=0.1.
    aux.run(
        q,
        k,
        indices,
        lengths,
        predict.detach(),
        teacher,
        loss,
        dq,
        dk,
        denominator,
        rectangular=True,
        spmm_impl=spmm_impl,
    )
    torch.testing.assert_close(loss, unit_loss.detach() * (0.1 / denominator_value), rtol=2e-5, atol=1e-7)
