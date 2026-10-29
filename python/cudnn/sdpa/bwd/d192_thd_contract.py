# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declaration-only workspace and pointer contracts for packed d192 backward."""

from cudnn.sdpa.bwd.d128_contract import WORKSPACE_ALIGNMENT, WORKSPACE_SLACK, validate_accumulation_scale

ABI_VERSION = 1
IMPLEMENTATION = "accumulation_thd"
ORIGIN_PORTS = ("q", "k", "v", "o", "do", "dq", "dk", "dv", "stats")
RAGGED_ROLES = tuple("ragged_" + port for port in ORIGIN_PORTS)
Q_PORTS = frozenset({"q", "o", "do", "dq", "stats"})
KV_PORTS = frozenset({"k", "v", "dk", "dv"})
MAPS_PER_SEQUENCE = 7


def _align(value):
    return (value + WORKSPACE_ALIGNMENT - 1) // WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT


def workspace_offsets(batch, heads, t_q):
    """Byte offsets shared with the traced host; arithmetic also accepts Int64."""
    rounded = (t_q + 127) // 128 * 128
    delta = 4 * rounded * heads * 192
    meta = delta + 4 * rounded * heads
    origins = meta + _align(4 * (4 * batch + 4))
    maps = origins + _align(8 * len(ORIGIN_PORTS) * batch)
    total = maps + MAPS_PER_SEQUENCE * batch * 128
    return rounded, delta, meta, origins, maps, total


def workspace_layout(batch, heads, t_q):
    """Packed FP32 dQ/Delta, metadata, INT64 origins and per-sequence maps."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, t_q)):
        raise ValueError("d192 THD workspace dimensions must be positive integers")
    rounded, delta, meta, origins, maps, total = workspace_offsets(batch, heads, t_q)
    if total + WORKSPACE_SLACK > 2**63 - 1:
        raise ValueError("d192 THD workspace exceeds the signed pointer range")
    regions = (
        (0, (1, rounded, heads, 192), (rounded * heads * 192, heads * 192, 192, 1)),
        (delta, (1, heads, rounded), (heads * rounded, rounded, 1)),
        (meta, (4 * batch + 4,), (1,)),
        (origins, (len(ORIGIN_PORTS), batch), (batch, 1)),
        (maps, (MAPS_PER_SEQUENCE * batch * 16,), (1,)),
    )
    return regions, total + WORKSPACE_SLACK


def validate_stats_span(heads, row_capacity):
    """Match the full FP32 [T,H] or [1,H,pitch] prepared carrier span.

    Positive dimensions and packing are validated by the callers. HN pitch
    remains Int64; only the existing signed-pointer storage bound applies.
    """
    if heads * row_capacity * 4 > 2**63 - 1:
        raise ValueError("packed FP32 Stats span exceeds the signed pointer range")


def validate_launch(pointers, spans, workspace, workspace_bytes, scale):
    """Validate all pointer relationships before setup can write any scratch."""
    validate_accumulation_scale(scale)
    if len(pointers) != 20 or len(spans) != 20:
        raise ValueError("d192 THD requires nine tensors, two lengths and nine optional offset slots")
    if workspace <= 0 or workspace % 16 or workspace + workspace_bytes > 2**63 - 1:
        raise ValueError("d192 THD requires aligned caller workspace within the signed pointer range")
    extents = []
    for pointer, span in zip(pointers, spans):
        if span == 0:
            if pointer is not None:
                raise ValueError("d192 THD received an operand absent from this specialization")
            extents.append(None)
            continue
        if pointer is None or pointer <= 0 or span <= 0 or pointer + span > 2**63 - 1:
            raise ValueError("d192 THD operand exceeds the signed pointer range")
        if pointer < workspace + workspace_bytes and workspace < pointer + span:
            raise ValueError("d192 THD caller workspace overlaps an operand")
        extents.append((pointer, pointer + span))
    for i in (6, 7, 8):
        lo, hi = extents[i]
        for j, extent in enumerate(extents):
            if i != j and extent is not None and lo < extent[1] and extent[0] < hi:
                raise ValueError("d192 THD gradient output overlaps another operand or metadata")


def validate_bounds(batch, heads, max_sq, max_skv, t_q, t_kv, t_q_phys, t_kv_phys):
    """Declaration bounds for the compact prefixes, 3D launch and TMA pitches."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, max_sq, max_skv, t_q, t_kv, t_q_phys, t_kv_phys)):
        raise ValueError("d192 THD requires positive integer dimensions and capacities")
    if max(max_sq, max_skv) > 2**31 - 128:
        raise ValueError("d192 THD sequence envelope exceeds safe Int32 tile rounding")
    if batch > 65535 or heads > 65535 or batch * heads * ((max_skv + 127) // 128) > 2**31 - 1:
        raise ValueError("d192 THD launch envelope exceeds the grid or Int32 work-id range")
    if max(t_q, t_kv) > 2**31 - 1:
        raise ValueError("d192 THD compact packed capacity exceeds the Int32 prefix ABI")
    if max(t_q_phys, t_kv_phys) > 2**32 - 1:
        raise ValueError("d192 THD physical token capacity exceeds the tensor-map dimension range")
    if max(t_q_phys, t_kv_phys) * heads * 192 * 2 >= 2**40:
        raise ValueError("d192 THD physical BSHD byte stride exceeds the tensor-map stride range")
    if ((t_q + 127) // 128 * 128) * heads * 192 * 4 >= 2**40:
        raise ValueError("d192 THD FP32 dQ byte stride exceeds the tensor-map stride range")
