# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""B200 correctness smoke for packed causal-conv autograd backward."""

from pathlib import Path

import cudnn
import torch
import torch.nn.functional as F

SOURCE_CUDNN = Path(__file__).resolve().parents[1] / "python" / "cudnn"
if str(SOURCE_CUDNN) not in cudnn.__path__:
    cudnn.__path__.insert(0, str(SOURCE_CUDNN))

from cudnn.causal_conv1d_bulk_sm100.autograd import (
    CausalConv1dBulkAutogradPrototype,
)


def dense_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    tokens = x.shape[1]
    z = F.conv1d(
        x.transpose(1, 2),
        weight.unsqueeze(1),
        padding=3,
        groups=x.shape[2],
    )[
        ..., :tokens
    ].transpose(1, 2)
    return F.silu(z)


def packed_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: tuple[int, ...],
) -> torch.Tensor:
    return torch.cat(
        [dense_reference(x[:, start:end], weight) for start, end in zip(offsets[:-1], offsets[1:], strict=True)],
        dim=1,
    )


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    difference = actual_float - expected_float
    return {
        "max_abs": float(difference.abs().max()),
        "rel_l2": float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected_float)),
    }


def main() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    shape = (1, 257, 256)
    offsets = (0, 1, 5, 70, 257)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    weight = (torch.randn((shape[2], 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25).requires_grad_()
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25

    x_ref = x.detach().float().requires_grad_()
    weight_ref = weight.detach().float().requires_grad_()
    expected = packed_reference(x_ref, weight_ref, offsets)
    expected.backward(dy.float())

    operation = CausalConv1dBulkAutogradPrototype(x, weight, cu_seqlens)
    actual = operation(x, weight, cu_seqlens)
    actual.backward(dy)
    torch.cuda.synchronize()

    results = {
        "output": metrics(actual, expected),
        "dx": metrics(x.grad, x_ref.grad),
        "dw": metrics(weight.grad, weight_ref.grad),
    }
    torch.testing.assert_close(actual.float(), expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(x.grad.float(), x_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(weight.grad.float(), weight_ref.grad, atol=1e-1, rtol=5e-2)

    backend = operation.backward_backend
    alias_rejected = False
    try:
        backend.execute(
            x.detach(),
            weight.detach(),
            dy,
            x.detach(),
            torch.empty_like(weight, dtype=torch.float32),
            cu_seqlens=cu_seqlens,
            packed_tile_map=torch.empty(
                backend.packed_tile_map_numel,
                device="cuda",
                dtype=torch.int32,
            ),
        )
    except ValueError as error:
        alias_rejected = "must not overlap" in str(error)
    if not alias_rejected:
        raise AssertionError("packed backward accepted an X/dX storage alias")
    print(
        {
            "device": torch.cuda.get_device_name(),
            "shape": shape,
            "offsets": offsets,
            "schedule": operation.backward_backend.schedule,
            "tile_map_bytes": operation.backward_backend.packed_tile_map_bytes,
            "alias_rejected": alias_rejected,
            "results": results,
        }
    )


if __name__ == "__main__":
    main()
