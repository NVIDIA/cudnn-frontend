# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Materialize caller-owned grouped-wgrad operands from backward auxiliaries."""

from __future__ import annotations

import torch

from ..._contracts import ValidatedBackwardRequest
from ..._types import MoeEpWgradOperands
from ._backward_launch import Mxfp8DgluResult
from ._wgrad_layout import (
    assemble_dfc2_atom_scales,
    assemble_discrete_col_requant_scales,
    deinterleave_gate_up_columns,
    pool_data_as_wgrad_a,
    pool_data_as_wgrad_b,
)


def export_wgrad_operands(
    request: ValidatedBackwardRequest,
    dglu: Mxfp8DgluResult,
) -> MoeEpWgradOperands:
    """Convert physical Rubin pools to the grouped-wgrad Tensor2D ABI."""

    if request.config.backward_wgrad_mode != "operands":
        raise ValueError("wgrad operands can only be exported in operands mode")
    stash = request.wgrad_forward_stash
    if stash is None:
        raise ValueError("wgrad operand export requires a forward stash")

    padded_ends = tuple(
        int(value) for value in stash.expert_offsets.detach().cpu().tolist()
    )
    valid_counts = tuple(
        int(value)
        for value in stash.valid_route_counts.detach().cpu().tolist()
    )
    padded_routes = padded_ends[-1] if padded_ends else 0
    config = request.config

    _validate_aux_shapes(dglu, padded_routes, config)

    # dC is emitted in the kernel's 32-column gate/up strip order. Upstream
    # exports route-weighted h and unweighted dY, whose product is the same
    # dW2 contract previously represented as unweighted h and weighted dY.
    # Every conversion allocates fresh storage, so no returned tensor aliases
    # the reusable execution plan.
    dc_pool = deinterleave_gate_up_columns(
        dglu.fc1_col_output[:padded_routes],
        config.intermediate_size,
    )
    fc1_b = pool_data_as_wgrad_b(dc_pool, padded_routes)
    fc1_sfb = assemble_dfc2_atom_scales(
        dglu.fc1_col_output_sf,
        valid_counts,
        padded_ends,
        2 * config.intermediate_size,
        config.sf_padding_size,
        deinterleave_gate_up=config.intermediate_size,
    )

    fc2_a = pool_data_as_wgrad_a(
        dglu.fc1_recompute,
        padded_routes,
    )
    fc2_sfa = assemble_dfc2_atom_scales(
        dglu.fc1_recompute_sf,
        valid_counts,
        padded_ends,
        config.intermediate_size,
        config.sf_padding_size,
    )
    fc2_b = pool_data_as_wgrad_b(
        dglu.grad_y2,
        padded_routes,
    )
    fc2_sfb = assemble_discrete_col_requant_scales(
        dglu.grad_y2_sf,
        valid_counts,
        padded_ends,
        config.hidden_size,
        config.sf_padding_size,
    )

    operands = MoeEpWgradOperands(
        fc1_a=stash.fc1_a,
        fc1_sfa=stash.fc1_sfa,
        fc1_b=fc1_b,
        fc1_sfb=fc1_sfb,
        fc2_a=fc2_a,
        fc2_sfa=fc2_sfa,
        fc2_b=fc2_b,
        fc2_sfb=fc2_sfb,
        expert_offsets=stash.expert_offsets,
        valid_route_counts=stash.valid_route_counts,
        route_metadata=stash.route_metadata,
    )
    _validate_grouped_wgrad_abi(operands, config, padded_routes)
    return operands


def _validate_aux_shapes(
    dglu: Mxfp8DgluResult,
    padded_routes: int,
    config,
) -> None:
    expected_columns = (
        ("fc1_recompute", dglu.fc1_recompute, config.intermediate_size),
        (
            "fc1_col_output",
            dglu.fc1_col_output,
            2 * config.intermediate_size,
        ),
        (
            "grad_y2",
            dglu.grad_y2,
            config.hidden_size,
        ),
    )
    for name, tensor, columns in expected_columns:
        if tensor.ndim != 2 or tensor.shape[1] != columns:
            raise RuntimeError(
                f"{name} must have shape (pool_capacity, {columns}), "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] < padded_routes or not tensor.is_contiguous():
            raise RuntimeError(
                f"{name} does not contain a contiguous {padded_routes}-row "
                "pool prefix"
            )
        if tensor.dtype is not torch.float8_e4m3fn:
            raise TypeError(f"{name} must have dtype torch.float8_e4m3fn")
        _require_alignment(name, tensor, 16)

    for name, tensor in (
        ("fc1_recompute_sf", dglu.fc1_recompute_sf),
        ("fc1_col_output_sf", dglu.fc1_col_output_sf),
    ):
        if tensor.ndim != 2 or tensor.dtype is not torch.float8_e8m0fnu:
            raise TypeError(f"{name} must be a rank-2 E8M0 tensor")
        _require_alignment(name, tensor, 16)
    if dglu.grad_y2_sf.ndim != 1 or dglu.grad_y2_sf.dtype is not torch.uint8:
        raise TypeError("grad_y2_sf must be a rank-1 uint8 tensor")
    _require_alignment("grad_y2_sf", dglu.grad_y2_sf, 16)


def _validate_grouped_wgrad_abi(
    operands: MoeEpWgradOperands,
    config,
    padded_routes: int,
) -> None:
    rounded_hidden = _round_up(config.hidden_size, 128)
    rounded_intermediate = _round_up(config.intermediate_size, 128)
    rounded_gate_up = _round_up(2 * config.intermediate_size, 128)
    sf_columns = _round_up(padded_routes // 32, 4)
    expected = (
        ("fc1_a", operands.fc1_a, (config.hidden_size, padded_routes)),
        ("fc1_sfa", operands.fc1_sfa, (rounded_hidden, sf_columns)),
        (
            "fc1_b",
            operands.fc1_b,
            (padded_routes, 2 * config.intermediate_size),
        ),
        ("fc1_sfb", operands.fc1_sfb, (rounded_gate_up, sf_columns)),
        (
            "fc2_a",
            operands.fc2_a,
            (config.intermediate_size, padded_routes),
        ),
        (
            "fc2_sfa",
            operands.fc2_sfa,
            (rounded_intermediate, sf_columns),
        ),
        ("fc2_b", operands.fc2_b, (padded_routes, config.hidden_size)),
        ("fc2_sfb", operands.fc2_sfb, (rounded_hidden, sf_columns)),
    )
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise RuntimeError(
                f"grouped-wgrad {name} shape must be {shape}, "
                f"got {tuple(tensor.shape)}"
            )
        _require_alignment(name, tensor, 16)

    for name in ("fc1_a", "fc2_a"):
        tensor = getattr(operands, name)
        expected_stride = (padded_routes, 1)
        if padded_routes and tensor.stride() != expected_stride:
            raise RuntimeError(
                f"grouped-wgrad {name} strides must be "
                f"{expected_stride}, got {tensor.stride()}"
            )
    for name in ("fc1_b", "fc2_b"):
        tensor = getattr(operands, name)
        expected_stride = (1, padded_routes)
        if padded_routes and tensor.stride() != expected_stride:
            raise RuntimeError(
                f"grouped-wgrad {name} strides must be "
                f"{expected_stride}, got {tensor.stride()}"
            )
    for name in ("fc1_sfa", "fc1_sfb", "fc2_sfa", "fc2_sfb"):
        if not getattr(operands, name).is_contiguous():
            raise RuntimeError(f"grouped-wgrad {name} must be contiguous")
    _require_alignment("expert_offsets", operands.expert_offsets, 4)


def _require_alignment(
    name: str,
    tensor: torch.Tensor,
    alignment: int,
) -> None:
    if tensor.data_ptr() % alignment:
        raise RuntimeError(
            f"{name} address is not {alignment}-byte aligned"
        )


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


__all__ = ["export_wgrad_operands"]
