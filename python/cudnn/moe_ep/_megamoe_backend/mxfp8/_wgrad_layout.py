# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""MXFP8 pool-to-grouped-wgrad data and scale layout conversion."""

from __future__ import annotations

import torch

from .._workspace import _align_up

_SF_VEC_SIZE = 32
_SF_ATOM_ROWS = 128
_SF_ATOM_COLUMNS = 4
_SF_ATOM_BYTES = _SF_ATOM_ROWS * _SF_ATOM_COLUMNS
_GATE_UP_INTERLEAVE = 32
_NEUTRAL_E8M0 = 127


def cumulative_padded_offsets(
    valid_counts: tuple[int, ...],
    padding: int,
    device: torch.device,
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Return cumulative padded expert ends as host values and device Int32."""

    ends = []
    total = 0
    for count in valid_counts:
        if count < 0:
            raise ValueError("expert route counts must be non-negative")
        total += _align_up(count, padding)
        ends.append(total)
    return tuple(ends), torch.tensor(
        ends,
        dtype=torch.int32,
        device=device,
    )


def pool_data_as_wgrad_a(
    pool_data: torch.Tensor,
    padded_routes: int,
) -> torch.Tensor:
    """Copy pool ``(K,M)`` bytes into contiguous logical wgrad A ``(M,K)``."""

    _validate_pool_prefix(pool_data, padded_routes)
    return pool_data[:padded_routes].transpose(0, 1).contiguous()


def pool_data_as_wgrad_b(
    pool_data: torch.Tensor,
    padded_routes: int,
) -> torch.Tensor:
    """Copy pool ``(K,N)`` bytes into a K-major logical wgrad B ``(K,N)``."""

    _validate_pool_prefix(pool_data, padded_routes)
    return (
        pool_data[:padded_routes]
        .transpose(0, 1)
        .contiguous()
        .transpose(0, 1)
    )


def deinterleave_gate_up_columns(
    tensor: torch.Tensor,
    intermediate: int,
) -> torch.Tensor:
    """Convert 32-column ``gate,up`` strips to logical ``gate || up``."""

    expected = 2 * intermediate
    if tensor.ndim != 2 or tensor.shape[1] != expected:
        raise ValueError(
            f"gate/up tensor must have shape (rows, {expected}), "
            f"got {tuple(tensor.shape)}"
        )
    if intermediate % _GATE_UP_INTERLEAVE:
        raise ValueError(
            "intermediate size must be divisible by "
            f"{_GATE_UP_INTERLEAVE}"
        )
    pairs = intermediate // _GATE_UP_INTERLEAVE
    blocks = tensor.reshape(
        tensor.shape[0],
        pairs,
        2,
        _GATE_UP_INTERLEAVE,
    )
    gate = blocks[:, :, 0, :].reshape(tensor.shape[0], intermediate)
    up = blocks[:, :, 1, :].reshape(tensor.shape[0], intermediate)
    return torch.cat((gate, up), dim=1)


def assemble_discrete_col_requant_scales(
    packed_scales: torch.Tensor,
    valid_counts: tuple[int, ...],
    padded_ends: tuple[int, ...],
    non_k_size: int,
    sf_padding: int,
) -> torch.Tensor:
    """Assemble upstream col-requant SF atoms for grouped wgrad.

    Since upstream revision 71d5fc1, each expert already emits
    ``(non-K/128, K/128, 512)`` atoms, which is grouped-wgrad order.
    """

    return _assemble_atom_scales(
        packed_scales,
        valid_counts,
        padded_ends,
        non_k_size,
        sf_padding,
        source_hidden_major=True,
        source_name="col-requant",
    )


def assemble_dfc2_atom_scales(
    packed_scales: torch.Tensor,
    valid_counts: tuple[int, ...],
    padded_ends: tuple[int, ...],
    non_k_size: int,
    sf_padding: int,
    *,
    deinterleave_gate_up: int | None = None,
) -> torch.Tensor:
    """Reorder dFC2 epilogue atoms from token-major to grouped-wgrad order."""

    return _assemble_atom_scales(
        packed_scales,
        valid_counts,
        padded_ends,
        non_k_size,
        sf_padding,
        source_hidden_major=False,
        source_name="dFC2",
        deinterleave_gate_up=deinterleave_gate_up,
    )


def _assemble_atom_scales(
    packed_scales: torch.Tensor,
    valid_counts: tuple[int, ...],
    padded_ends: tuple[int, ...],
    non_k_size: int,
    sf_padding: int,
    *,
    source_hidden_major: bool,
    source_name: str,
    deinterleave_gate_up: int | None = None,
) -> torch.Tensor:
    """Expand compact per-expert SF atoms to the data-padded K extent."""

    if len(valid_counts) != len(padded_ends):
        raise ValueError("expert count and padded offset lengths must match")
    _validate_padded_ends(padded_ends)
    if sf_padding % _SF_ATOM_ROWS:
        raise ValueError("scale padding must be divisible by 128")
    padded_non_k = _align_up(non_k_size, _SF_ATOM_ROWS)
    non_k_atoms = padded_non_k // _SF_ATOM_ROWS
    flat_u8 = packed_scales.view(torch.uint8).reshape(-1)
    expert_parts = []
    previous_end = 0
    source_byte_offset = 0
    for count, end in zip(valid_counts, padded_ends):
        target_extent = end - previous_end
        if count < 0 or count > target_extent:
            raise ValueError("valid expert routes exceed their padded extent")
        if target_extent % _SF_ATOM_ROWS:
            raise ValueError(
                "data-padded expert extents must be multiples of 128"
            )
        source_extent = _align_up(count, sf_padding)
        source_token_atoms = source_extent // _SF_ATOM_ROWS
        target_token_atoms = target_extent // _SF_ATOM_ROWS
        if source_token_atoms > target_token_atoms:
            raise ValueError("scale-padded extent exceeds data-padded extent")
        source_byte_count = (
            source_token_atoms * non_k_atoms * _SF_ATOM_BYTES
        )
        if source_byte_offset + source_byte_count > flat_u8.numel():
            raise ValueError(
                f"{source_name} scale output is smaller than its layout"
            )
        target_raw = torch.full(
            (padded_non_k, target_token_atoms * _SF_ATOM_COLUMNS),
            _NEUTRAL_E8M0,
            dtype=torch.uint8,
            device=flat_u8.device,
        )
        if source_byte_count:
            source = flat_u8.narrow(
                0,
                source_byte_offset,
                source_byte_count,
            )
            if source_hidden_major:
                source = source.reshape(
                    non_k_atoms,
                    source_token_atoms,
                    _SF_ATOM_BYTES,
                )
            else:
                source = (
                    source.reshape(
                        source_token_atoms,
                        non_k_atoms,
                        _SF_ATOM_BYTES,
                    )
                    .permute(1, 0, 2)
                    .contiguous()
                )
            source_raw = _from_blocked_bytes(
                source.reshape(-1),
                padded_non_k,
                source_token_atoms * _SF_ATOM_COLUMNS,
            )
            target_raw[:, : source_raw.shape[1]].copy_(source_raw)
        if deinterleave_gate_up is not None:
            target_raw = (
                deinterleave_gate_up_columns(
                    target_raw.transpose(0, 1),
                    deinterleave_gate_up,
                )
                .transpose(0, 1)
                .contiguous()
            )
        expert_parts.append(_to_blocked_bytes(target_raw))
        source_byte_offset += source_byte_count
        previous_end = end

    total_routes = padded_ends[-1] if padded_ends else 0
    scale_columns = _align_up(total_routes // _SF_VEC_SIZE, 4)
    if expert_parts:
        assembled_u8 = torch.cat(expert_parts)
    else:
        assembled_u8 = flat_u8.new_empty((0,))
    expected = padded_non_k * scale_columns
    if assembled_u8.numel() != expected:
        raise RuntimeError(
            f"assembled {source_name} scale size mismatch: "
            f"{assembled_u8.numel()} != {expected}"
        )
    return assembled_u8.reshape(padded_non_k, scale_columns).view(
        torch.float8_e8m0fnu
    )


def assemble_plain_col_scales(
    col_scales: torch.Tensor,
    valid_counts: tuple[int, ...],
    padded_ends: tuple[int, ...],
    non_k_size: int,
    sf_padding: int,
    *,
    deinterleave_gate_up: int | None = None,
) -> torch.Tensor:
    """Assemble plain ``(K/32,N)`` col scales into grouped-wgrad SF atoms."""

    if len(valid_counts) != len(padded_ends):
        raise ValueError("expert count and padded offset lengths must match")
    _validate_padded_ends(padded_ends)
    if col_scales.ndim != 2 or col_scales.shape[1] != non_k_size:
        raise ValueError(
            "plain column scales must have shape "
            f"(rows, {non_k_size}), got {tuple(col_scales.shape)}"
        )
    if sf_padding % _SF_VEC_SIZE:
        raise ValueError("scale padding must be divisible by 32")

    source_u8 = col_scales.view(torch.uint8)
    expert_parts = []
    previous_end = 0
    sf_row = 0
    for count, end in zip(valid_counts, padded_ends):
        padded_extent = end - previous_end
        if padded_extent % _SF_VEC_SIZE:
            raise ValueError("padded expert extents must be divisible by 32")
        valid_sf_rows = (count + _SF_VEC_SIZE - 1) // _SF_VEC_SIZE
        padded_sf_rows = padded_extent // _SF_VEC_SIZE
        if count < 0 or count > padded_extent:
            raise ValueError("valid expert routes exceed their padded extent")
        if sf_row + valid_sf_rows > source_u8.shape[0]:
            raise ValueError("plain column scale output is too short")

        raw = torch.full(
            (non_k_size, padded_sf_rows),
            _NEUTRAL_E8M0,
            dtype=torch.uint8,
            device=col_scales.device,
        )
        if valid_sf_rows:
            source = source_u8[
                sf_row : sf_row + valid_sf_rows,
                :,
            ]
            if deinterleave_gate_up is not None:
                source = deinterleave_gate_up_columns(
                    source,
                    deinterleave_gate_up,
                )
            raw[:, :valid_sf_rows].copy_(source.transpose(0, 1))
        expert_parts.append(_to_blocked_bytes(raw))
        sf_row += _align_up(count, sf_padding) // _SF_VEC_SIZE
        previous_end = end

    padded_non_k = _align_up(non_k_size, _SF_ATOM_ROWS)
    total_routes = padded_ends[-1] if padded_ends else 0
    scale_columns = _align_up(total_routes // _SF_VEC_SIZE, 4)
    if expert_parts:
        assembled_u8 = torch.cat(expert_parts)
    else:
        assembled_u8 = source_u8.new_empty((0,))
    expected = padded_non_k * scale_columns
    if assembled_u8.numel() != expected:
        raise RuntimeError(
            "assembled plain column scale size mismatch: "
            f"{assembled_u8.numel()} != {expected}"
        )
    return assembled_u8.reshape(padded_non_k, scale_columns).view(
        torch.float8_e8m0fnu
    )


def _to_blocked_bytes(raw_scale: torch.Tensor) -> torch.Tensor:
    rows, columns = raw_scale.shape
    if rows == 0 or columns == 0:
        return raw_scale.new_empty((0,), dtype=torch.uint8)
    padded_rows = _align_up(rows, _SF_ATOM_ROWS)
    padded_columns = _align_up(columns, _SF_ATOM_COLUMNS)
    padded = torch.full(
        (padded_rows, padded_columns),
        _NEUTRAL_E8M0,
        dtype=torch.uint8,
        device=raw_scale.device,
    )
    padded[:rows, :columns].copy_(raw_scale)
    blocks = padded.view(
        padded_rows // _SF_ATOM_ROWS,
        _SF_ATOM_ROWS,
        padded_columns // _SF_ATOM_COLUMNS,
        _SF_ATOM_COLUMNS,
    ).permute(0, 2, 1, 3)
    return (
        blocks.reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
    )


def _from_blocked_bytes(
    packed_scale: torch.Tensor,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Invert the grouped-wgrad 128x4 scale-atom swizzle."""

    padded_rows = _align_up(rows, _SF_ATOM_ROWS)
    padded_columns = _align_up(columns, _SF_ATOM_COLUMNS)
    expected = padded_rows * padded_columns
    flat = packed_scale.view(torch.uint8).reshape(-1)
    if flat.numel() != expected:
        raise ValueError(
            f"blocked scale has {flat.numel()} bytes, expected {expected}"
        )
    if expected == 0:
        return flat.new_empty((rows, columns))
    row_atoms = padded_rows // _SF_ATOM_ROWS
    column_atoms = padded_columns // _SF_ATOM_COLUMNS
    raw = (
        flat.reshape(row_atoms * column_atoms, 32, 4, 4)
        .transpose(1, 2)
        .reshape(row_atoms, column_atoms, _SF_ATOM_ROWS, _SF_ATOM_COLUMNS)
        .permute(0, 2, 1, 3)
        .reshape(padded_rows, padded_columns)
    )
    return raw[:rows, :columns]


def _validate_pool_prefix(
    pool_data: torch.Tensor,
    padded_routes: int,
) -> None:
    if pool_data.ndim != 2 or not pool_data.is_contiguous():
        raise ValueError("pool data must be a contiguous rank-2 tensor")
    if padded_routes < 0 or padded_routes > pool_data.shape[0]:
        raise ValueError(
            f"padded route count {padded_routes} exceeds pool capacity "
            f"{pool_data.shape[0]}"
        )


def _validate_padded_ends(padded_ends: tuple[int, ...]) -> None:
    previous = 0
    for end in padded_ends:
        if end < previous:
            raise ValueError("expert offsets must be non-decreasing")
        previous = end


__all__ = [
    "assemble_dfc2_atom_scales",
    "assemble_discrete_col_requant_scales",
    "assemble_plain_col_scales",
    "cumulative_padded_offsets",
    "deinterleave_gate_up_columns",
    "pool_data_as_wgrad_a",
    "pool_data_as_wgrad_b",
]
