# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Version-bounded exclusions for measured backend backward correctness defects."""

RAGGED_TILE_MAPPING_SINCE = 92400
RAGGED_TILE_MAPPING_FIXED_IN = 92501


def backend_engine_guard(graph, facts):
    """Keep the affected engine off a graph; preserve other backend candidates."""
    import cudnn

    if facts is None or facts.invalid is not None or not facts.is_backward:
        return {}
    if facts.device_cc != (9, 0) or not facts.thd or facts.is_fp8 or facts.is_mxfp8:
        return {}
    if facts.dtype not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
        return {}
    if not (192 < facts.d_qk <= 256 and 64 < facts.d_v <= 128):
        return {}
    if not RAGGED_TILE_MAPPING_SINCE <= cudnn.backend_version() < RAGGED_TILE_MAPPING_FIXED_IN:
        return {}
    # The old SM90 ragged scheduler omits half of the KV tiles when Q/K use
    # the 256-wide implementation and V uses its 128-wide implementation.
    # Dq <= 192, Dv <= 64, and dense/padded layouts use unaffected mappings.
    # Actual 9.24.0.43 / 9.25.0.15 reproduce wrong gradients; the same inputs
    # pass on 9.25.1.1 / 9.27.0.42. No hidden layout conversion is attempted.
    return {
        6: f"cuDNN {cudnn.backend_version_string()} engine 6 returns incorrect gradients for SM90 THD SDPA backward "
        "with 192 < d_qk <= 256 and 64 < d_v <= 128; the ragged tile mapping is fixed in cuDNN 9.25.1"
    }
