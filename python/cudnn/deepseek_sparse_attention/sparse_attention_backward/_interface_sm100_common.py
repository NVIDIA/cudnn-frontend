# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared workspace contracts for the SM100 DSA backward interfaces."""

import math
from typing import Optional, Tuple

import cutlass

from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
from .dsa_bwd_sm100_deterministic import FlashAttentionDSABackwardSm100Deterministic, deterministic_kernel_class

_BLACKWELL_CAPABILITIES = ((10, 0), (10, 3))
_WORKSPACE_ALIGNMENT = 128


def _align_workspace_bytes(num_bytes: int) -> int:
    """Round a workspace segment size up to the shared alignment boundary."""
    return -(-int(num_bytes) // _WORKSPACE_ALIGNMENT) * _WORKSPACE_ALIGNMENT


def _workspace_shapes_sm100(
    total_s_q: int,
    total_s_kv: int,
    head_dim: int,
    num_heads: int,
    deterministic: bool,
    backend: Optional[str] = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return byte-shaped LSE/OdO and dKV scratch layouts for one launch.

    Generic deterministic M64 launches use the adaptive shard specialization.
    H96/H128 D576 two-CTA kernels use their compact 96-plane workspace. An
    unspecified backend retains the conservative 128-plane capacity.
    """
    acc_dtype = cutlass.Float32
    workspace_lse_odo_shape = FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(
        total_s_q,
        head_dim,
        num_heads,
        1,
        acc_dtype,
    )
    if deterministic:
        if backend == "generic_m64":
            dkv_kernel_cls = deterministic_kernel_class(total_s_q)
        elif backend in {"h96_d576_2cta_m64", "h128_d576_2cta_m64"}:
            from .dsa_bwd_sm100_h128_d576_2cta_deterministic import TwoCTADeterministicReduction

            dkv_kernel_cls = TwoCTADeterministicReduction
        else:
            dkv_kernel_cls = FlashAttentionDSABackwardSm100Deterministic
    else:
        dkv_kernel_cls = FlashAttentionDSABackwardSm100
    workspace_dkv_shape = dkv_kernel_cls._get_workspace_size_dKV(
        total_s_kv,
        head_dim,
        1,
        acc_dtype,
    )
    return workspace_lse_odo_shape, workspace_dkv_shape


def flash_attn_bwd_sm100_workspace_size(
    total_s_q: int,
    total_s_kv: int,
    head_dim: int,
    num_heads: int,
    deterministic: bool = False,
    *,
    backend: Optional[str] = None,
) -> int:
    """Return the reusable caller scratch required by one SM100 launch.

    The generic deterministic M64 backend selects a 32-, 64-, or 128-plane
    dKV layout from ``total_s_q``. H96/H128 D576 two-CTA backends use a
    compact 96-plane layout; unspecified backends use the conservative fixed
    128-plane layout. The result includes alignment padding and is the minimum
    byte count for a contiguous ``torch.uint8`` workspace.
    """
    workspace_lse_odo_shape, workspace_dkv_shape = _workspace_shapes_sm100(
        total_s_q,
        total_s_kv,
        head_dim,
        num_heads,
        deterministic,
        backend,
    )
    return _align_workspace_bytes(math.prod(workspace_lse_odo_shape)) + _align_workspace_bytes(math.prod(workspace_dkv_shape))
