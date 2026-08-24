# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared reference helpers for MoE EP backward tests."""

from __future__ import annotations

import torch

from moe_ep.moe_ep_forward_support import _reference_args
from moe_ep.moe_ep_reference import MoeEpReference

__all__ = [
    "_assert_backward_matches",
    "_dense_wgrads_from_operands",
    "_expected_backward",
    "_grad_output",
    "_reference_backward",
]


_BACKWARD_CLOSE_KWARGS = (
    {"rtol": 0.15, "atol": 0.125},  # grad_activation is BF16-rounded.
    {"rtol": 0.15, "atol": 0.125},  # router-weight gradient.
)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _unpack_wgrad_scale_part(
    packed: torch.Tensor,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Invert grouped-wgrad's 128x4 scale-atom swizzle."""

    padded_rows = _round_up(rows, 128)
    padded_columns = _round_up(columns, 4)
    row_atoms = padded_rows // 128
    column_atoms = padded_columns // 4
    atom_count = row_atoms * column_atoms
    expected = padded_rows * padded_columns
    if packed.numel() != expected:
        raise ValueError(
            f"packed scale part has {packed.numel()} bytes, expected {expected}"
        )
    blocked = (
        packed.reshape(atom_count, 32, 4, 4)
        .transpose(1, 2)
        .reshape(row_atoms, column_atoms, 128, 4)
        .permute(0, 2, 1, 3)
        .reshape(padded_rows, padded_columns)
    )
    return blocked[:rows, :columns].view(torch.float8_e8m0fnu).float()


def _dequantize_wgrad_operand(
    data: torch.Tensor,
    scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    k_dim: int,
) -> torch.Tensor:
    """Decode one public grouped-wgrad operand without launching a GEMM."""

    if data.ndim != 2 or k_dim not in (0, 1):
        raise ValueError("wgrad operand must be rank 2 with k_dim 0 or 1")
    non_k = int(data.shape[1 - k_dim])
    padded_non_k = _round_up(non_k, 128)
    flat_scales = scales.view(torch.uint8).reshape(-1)
    output = torch.empty(data.shape, dtype=torch.float32, device=data.device)
    ends = [int(value) for value in expert_offsets.detach().cpu().tolist()]
    previous = 0
    scale_byte_offset = 0
    for end in ends:
        extent = end - previous
        scale_columns = _round_up(extent // 32, 4)
        scale_byte_count = padded_non_k * scale_columns
        part = flat_scales.narrow(
            0,
            scale_byte_offset,
            scale_byte_count,
        )
        logical_scale = _unpack_wgrad_scale_part(
            part,
            non_k,
            extent // 32,
        )
        if k_dim == 1:
            expanded_scale = logical_scale.repeat_interleave(32, dim=1)
            output[:, previous:end] = (
                data[:, previous:end].float() * expanded_scale
            )
        else:
            expanded_scale = logical_scale.repeat_interleave(
                32,
                dim=1,
            ).transpose(0, 1)
            output[previous:end, :] = (
                data[previous:end, :].float() * expanded_scale
            )
        previous = end
        scale_byte_offset += scale_byte_count
    if previous != data.shape[k_dim]:
        raise ValueError("expert offsets do not cover the operand K dimension")
    if scale_byte_offset != flat_scales.numel():
        raise ValueError("expert offsets do not cover the scale tensor")
    return output


def _dense_wgrads_from_operands(operands):
    """Reference grouped matmuls over the exported operand ABI."""

    fc1_a = _dequantize_wgrad_operand(
        operands.fc1_a,
        operands.fc1_sfa,
        operands.expert_offsets,
        k_dim=1,
    )
    fc1_b = _dequantize_wgrad_operand(
        operands.fc1_b,
        operands.fc1_sfb,
        operands.expert_offsets,
        k_dim=0,
    )
    fc2_a = _dequantize_wgrad_operand(
        operands.fc2_a,
        operands.fc2_sfa,
        operands.expert_offsets,
        k_dim=1,
    )
    fc2_b = _dequantize_wgrad_operand(
        operands.fc2_b,
        operands.fc2_sfb,
        operands.expert_offsets,
        k_dim=0,
    )
    fc1_parts = []
    fc2_parts = []
    previous = 0
    for end_value in operands.expert_offsets.detach().cpu().tolist():
        end = int(end_value)
        fc1_parts.append(
            fc1_a[:, previous:end] @ fc1_b[previous:end, :]
        )
        fc2_parts.append(
            fc2_a[:, previous:end] @ fc2_b[previous:end, :]
        )
        previous = end
    return torch.stack(fc1_parts), torch.stack(fc2_parts)


def _reference_backward(config) -> MoeEpReference:
    options = dict(config)
    options.pop("tuning", None)
    options.pop("sf_padding_size", None)
    options["intermediate_format"] = "mxfp8"
    options["backward_operand_format"] = "mxfp8"
    return MoeEpReference(**options)


def _grad_output(
    device: torch.device,
    token_count: int,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(
            token_count,
            128,
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / 8
    )


def _expected_backward(reference, grad_output, args, stash):
    return reference.backward(
        grad_output,
        *_reference_args(args)[1:],
        *stash,
    )


def _assert_backward_matches(actual, expected, topk_idx) -> None:
    assert len(actual) == len(expected) == 2
    for name, gradient, reference, close_kwargs in zip(
        ("grad_activation", "grad_topk_weights"),
        actual,
        expected,
        _BACKWARD_CLOSE_KWARGS,
    ):
        assert gradient.shape == reference.shape
        assert gradient.dtype == torch.float32
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(
            gradient,
            reference,
            msg=lambda default, name=name: (
                f"{name} does not match the backward reference\n{default}"
            ),
            **close_kwargs,
        )

    dropped = topk_idx == -1
    assert actual[1][dropped].eq(0).all()
