# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public pre-routed SwiGLU MoE operation tests."""

import pytest
import torch

from cudnn.gemm import swiglu_moe


def _cc():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _relative_l2(actual, expected):
    return (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-9)


def _reference(x, Wg, Wu, Wd, starts):
    S = x.shape[1]
    output = torch.empty_like(x)
    for expert, begin in enumerate(starts):
        end = starts[expert + 1] if expert + 1 < len(starts) else S
        if begin == end:
            continue
        token = x[:, begin:end]
        gate = token @ Wg[expert].transpose(0, 1)
        up = token @ Wu[expert].transpose(0, 1)
        h = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
        output[:, begin:end] = h @ Wd[expert].transpose(0, 1)
    return output


@pytest.mark.L0
def test_swiglu_moe_is_public():
    import cudnn.gemm as gemm
    import cudnn.gemm.ops as ops

    assert gemm.swiglu_moe is swiglu_moe
    assert ops.swiglu_moe is swiglu_moe


@pytest.mark.L0
@pytest.mark.skipif(
    not (torch.cuda.is_available() and 100 <= _cc() < 120),
    reason="FROST SwiGLU MoE requires SM100-SM119",
)
def test_swiglu_moe_forward_and_all_gradients_with_empty_expert():
    E, H, I = 4, 128, 256
    group_sizes = [64, 0, 96, 32]
    starts, total = [], 0
    for size in group_sizes:
        starts.append(total)
        total += size

    torch.manual_seed(7)
    base = (
        torch.randn(1, total, H, device="cuda", dtype=torch.bfloat16) * 0.2,
        torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.02,
    )
    actual_inputs = tuple(t.detach().clone().requires_grad_(True) for t in base)
    reference_inputs = tuple(t.detach().clone().requires_grad_(True) for t in base)
    offsets = torch.tensor(starts, device="cuda", dtype=torch.int32)

    actual = swiglu_moe(*actual_inputs, offsets)
    expected = _reference(*reference_inputs, starts)
    dout = torch.randn_like(actual)
    actual.backward(dout)
    expected.backward(dout)

    assert _relative_l2(actual, expected) < 1e-2
    for name, got, ref in zip(("dx", "dWg", "dWu", "dWd"), actual_inputs, reference_inputs):
        error = _relative_l2(got.grad, ref.grad)
        assert error < 1e-2, f"{name} relative L2 error {error.item():.3e}"


@pytest.mark.L0
def test_swiglu_moe_validates_public_layout():
    x = torch.empty(1, 8, 16, dtype=torch.bfloat16)
    Wg = torch.empty(2, 32, 16, dtype=torch.bfloat16)
    Wu = torch.empty_like(Wg)
    Wd = torch.empty(2, 16, 32, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        swiglu_moe(x, Wg, Wu, Wd, offsets)
