# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent summary factor warps must retain their ring parity between work items."""

import pytest
import torch

from cudnn.linear_attention import kimi_delta_attention

pytestmark = [pytest.mark.L1, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


@pytest.mark.parametrize("tokens", [16224, 16384])
def test_summary_multiple_work_items(tokens):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires Blackwell FROST KDA")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    available, version = cutedsl_state()
    if not available or cutedsl_too_old(version):
        pytest.skip("requires a FROST-compatible CuTe DSL")
    heads, dim = 128, 128
    if torch.cuda.get_device_properties(0).multi_processor_count >= 2 * heads:
        pytest.skip("requires more summary work items than SMs")
    # BI splits these lengths into two pieces of 507 / 512 chunks. At least
    # one CTA must continue to another work item after its first piece.
    x = torch.zeros((tokens, heads, dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    beta = torch.ones((tokens, heads), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    a_log = torch.zeros(heads, device="cuda", requires_grad=True)
    dt_bias = torch.full((heads, dim), -5.5, device="cuda", requires_grad=True)
    output = kimi_delta_attention(
        q=x,
        k=x,
        v=x,
        g=x,
        beta=beta,
        cu_seqlens=cu,
        a_log=a_log,
        dt_bias=dt_bias,
        safe_gate=True,
        gate_lower_bound=-5.0,
        batch_invariant=True,
        initial_state=None,
        output_final_state=False,
        plan_name="kda_frost",
    )[0]
    grads = torch.autograd.grad(output, (x, beta, a_log, dt_bias), torch.ones_like(output))
    torch.cuda.synchronize()
    for value in (output, *grads):
        assert torch.equal(value, torch.zeros_like(value))
