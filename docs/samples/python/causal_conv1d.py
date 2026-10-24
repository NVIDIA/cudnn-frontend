# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify Causal Conv1d forward, backward, and torch.compile integration."""

import torch
import torch.nn.functional as F

from cudnn.ops.causal_conv1d import causal_conv1d


def reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    kernel_size = weight.shape[1]
    y = F.conv1d(
        F.pad(x, (kernel_size - 1, 0)),
        weight.unsqueeze(1),
        bias,
        groups=x.shape[1],
    )
    return F.silu(y) if activation == "silu" else y


def main() -> None:
    torch.manual_seed(7)
    batch, channels, sequence, kernel_size = 2, 128, 256, 4
    x = torch.randn(
        batch,
        channels,
        sequence,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(
        channels,
        kernel_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    bias = torch.randn(
        channels,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    actual = causal_conv1d(x, weight, bias, activation="silu")

    reference_inputs = tuple(
        tensor.detach().float().requires_grad_() for tensor in (x, weight, bias)
    )
    expected = reference(*reference_inputs, "silu")
    torch.testing.assert_close(actual.float(), expected, rtol=3e-2, atol=3e-2)

    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, weight, bias), grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad.float())
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got.float(), want, rtol=5e-2, atol=5e-2)

    compiled = torch.compile(causal_conv1d, fullgraph=True)
    compiled_result = compiled(x.detach(), weight.detach(), bias.detach(), "identity")
    compiled_reference = reference(
        x.detach().float(), weight.detach().float(), bias.detach().float(), "identity"
    )
    torch.testing.assert_close(compiled_result.float(), compiled_reference, rtol=3e-2, atol=3e-2)
    print("causal conv1d ok: eager, backward, compile")


if __name__ == "__main__":
    main()
