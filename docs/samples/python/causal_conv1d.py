"""Verify Causal Conv1d forward, backward, and torch.compile integration."""

import torch
import torch.nn.functional as F

import cudnn


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

    actual = cudnn.ops.causal_conv1d(x, weight, bias, activation="silu")
    expected = reference(x, weight, bias, "silu")
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, weight, bias), grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (x, weight, bias), grad)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, rtol=5e-2, atol=5e-2)

    compiled = torch.compile(cudnn.ops.causal_conv1d, fullgraph=True)
    compiled_result = compiled(x.detach(), weight.detach(), bias.detach(), "identity")
    compiled_reference = reference(x.detach(), weight.detach(), bias.detach(), "identity")
    torch.testing.assert_close(compiled_result, compiled_reference, rtol=3e-2, atol=3e-2)
    print("causal conv1d ok: eager, backward, compile")


if __name__ == "__main__":
    main()
