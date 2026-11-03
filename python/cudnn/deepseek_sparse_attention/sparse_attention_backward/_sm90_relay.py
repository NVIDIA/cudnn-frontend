# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared public constants and validation for the SM90 deterministic relay."""

import operator

from packaging.version import InvalidVersion, Version

SM90_DETERMINISTIC_RELAY_CLUSTER_SIZES = (4, 8, 16, 32, 64)
SM90_DEFAULT_DETERMINISTIC_RELAY_CLUSTER_SIZE = 16
SM90_SIGNED_INT32_ELEMENT_LIMIT = 1 << 31


def _round_up(value: int, multiple: int) -> int:
    return (int(value) + multiple - 1) // multiple * multiple


def sm90_deterministic_relay_layout(total_s_q, total_s_kv, head_dim, num_heads, relay_cluster_size):
    """Return (planes, physical plane elements, mask rows, waves, head tiles)."""
    relay_cluster_size = normalize_sm90_relay_cluster_size(relay_cluster_size)
    seqlen_k_rounded = _round_up(total_s_kv, 64)
    head_dim_rounded = _round_up(head_dim, 32)
    num_waves = _round_up(total_s_q, 128) // 128
    num_head_tiles = _round_up(num_heads, 64) // 64
    num_dkv_planes = 128 // relay_cluster_size
    mask_word_bits = 64 if relay_cluster_size == 64 else 32
    mask_rows_per_word = mask_word_bits // relay_cluster_size
    words_per_phase = seqlen_k_rounded // mask_rows_per_word
    mask_storage_elems = num_waves * num_head_tiles * words_per_phase * (mask_word_bits // 32)
    mask_rows = _round_up(mask_storage_elems, head_dim_rounded) // head_dim_rounded
    physical_plane_elems = (seqlen_k_rounded + mask_rows) * head_dim_rounded
    return num_dkv_planes, physical_plane_elems, mask_rows, num_waves, num_head_tiles


def sm90_deterministic_relay_accumulator_elements(total_s_q, total_s_kv, head_dim, num_heads, relay_cluster_size) -> int:
    """Return the flat FP32 relay-workspace span addressed by SM90 kernels."""
    num_dkv_planes, physical_plane_elems, *_ = sm90_deterministic_relay_layout(total_s_q, total_s_kv, head_dim, num_heads, relay_cluster_size)
    return num_dkv_planes * physical_plane_elems


def sm90_deterministic_relay_indexing_error(accumulator_elements):
    """Return an error when dynamic tensor offsets can exceed signed Int32."""
    if accumulator_elements <= SM90_SIGNED_INT32_ELEMENT_LIMIT:
        return None
    return f"deterministic SM90 DSA backward dKV accumulator has {accumulator_elements} elements, " "which exceeds the signed Int32 element offset range"


def sm90_deterministic_relay_requirement_error():
    """Return the public CuTeDSL version error for the SM90 relay, if any."""
    from cudnn.frost.buffers import cutedsl_state

    installed, version = cutedsl_state()
    requirement = "deterministic SM90 DSA backward requires nvidia-cutlass-dsl >= 4.8.0"
    if not installed:
        return requirement
    if version is None or version[0] != "nvidia-cutlass-dsl":
        return None
    try:
        too_old = Version(version[1]).release < (4, 8, 0)
    except InvalidVersion:
        return None
    return f"{requirement}; found {version[1]}" if too_old else None


def normalize_sm90_relay_cluster_size(value) -> int:
    """Return an exact integer relay width and reject lossy coercions."""
    try:
        cluster_size = operator.index(value)
    except TypeError as error:
        raise ValueError("SM90 deterministic_relay_cluster_size must be an integer " f"in {SM90_DETERMINISTIC_RELAY_CLUSTER_SIZES}, got {value!r}") from error
    if cluster_size not in SM90_DETERMINISTIC_RELAY_CLUSTER_SIZES:
        raise ValueError("SM90 deterministic_relay_cluster_size must be 4, 8, 16, 32, or 64 " f"(32, 16, 8, 4, or 2 FP32 planes), got {cluster_size}")
    return cluster_size
