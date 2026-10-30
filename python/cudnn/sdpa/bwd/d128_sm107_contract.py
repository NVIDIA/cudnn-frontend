# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declaration bounds for the native SM107 dense D128 accumulation chain."""

import numbers
import re

from .config_sm107_d128_accum import CFG
from .d128_contract import validate_accumulation_scale, validate_dense_bounds


def validate_scale(scale):
    """A host scalar only: never coerce a device tensor during preflight."""
    if isinstance(scale, bool) or not isinstance(scale, numbers.Real):
        raise ValueError("native SM107 D128 attn_scale must be a host real scalar")
    validate_accumulation_scale(scale)


def validate_dsl_version(version):
    """Check this row's qualified DSL floor independently of target support."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version[1]) if version and version[0] == "nvidia-cutlass-dsl" else None
    if match is None or tuple(map(int, match.groups())) < (4, 8, 1):
        raise ValueError(f"native SM107 D128 backward requires nvidia-cutlass-dsl >= 4.8.1; found {version}")


def validate_declaration(cc, version, batch, heads, sq, skv, scale):
    """Validate plan facts without importing a compiler or touching device data.

    This route owns dense MHA only; it never falls back to SM100 recompute,
    grouped or packed bodies. Single-query/key qualification remains separate
    (2026-10-29): the pinned graph gradient/replay test must cover those cases
    before removing their decline.
    """
    if cc != CFG.ARCH_CC:
        raise ValueError(f"native D128 backward requires SM107; got {cc}")
    validate_dsl_version(version)
    validate_dense_bounds(batch, heads, sq, skv)
    validate_scale(scale)
    if sq == 1 or skv == 1:
        raise ValueError("native SM107 D128 backward requires S_q and S_kv > 1; single-query/key qualification is pending")
    q_tiles, k_tiles = (sq + CFG.N - 1) // CFG.N, (skv + CFG.M - 1) // CFG.M
    q_rounded = q_tiles * CFG.N
    if batch > 65535 or heads > 65535 or batch * heads * k_tiles > 2**31 - 1:
        raise ValueError("native SM107 D128 launch envelope exceeds the grid or Int32 work-id range")
    if batch * heads * q_rounded > 2**31 - 1:
        raise ValueError("native SM107 D128 rounded Stats index exceeds the Int32 range")
    # This also bounds each FP32 dQ tensor-map byte stride strictly below 2**40:
    # one row is CFG.D * CFG.BPE_ACC == 512 bytes.
    heads_per_l2 = CFG.SOURCE_SCHED_L2_BYTES // (sq * (CFG.D + CFG.D) * CFG.BPE)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    if swizzle * k_tiles > 2**31 - 1:
        raise ValueError("native SM107 D128 scheduler divisor exceeds the Int32 range")
