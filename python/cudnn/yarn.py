# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""YARN RoPE frequency computation.

Pure host-side implementation following the YARN paper (Peng et al., 2023, arXiv:2309.00071)
and the Megatron-Core reference at
megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py.

YARN extends RoPE's effective context window by:
  1. Per-dim frequency blending: low-freq dims get scaled by 1/factor (interpolation),
     high-freq dims pass through (extrapolation), with a linear ramp between
     correction dims defined by beta_fast/beta_slow.
  2. Attention temperature (mscale): 0.1*log(factor)+1, multiplied into cos/sin.
     This is mathematically equivalent to multiplying SDPA logits by mscale**2.

This module produces the modified ``freqs`` tensor that the existing
``graph.rope()`` op consumes (raw angles, layout ``[max_seq_len, 1, 1, head_dim]``)
and the ``mscale`` scalar to fold into ``output_scale`` on each RoPE node.

Workflow (host-side, called once at model init)::

    from cudnn.yarn import compute_yarn_freqs

    freqs, mscale = compute_yarn_freqs(
        max_seq_len=131072, head_dim=64, base=10000.0,
        scaling_factor=40.0, original_max_position=4096,
        beta_fast=32, beta_slow=1,
    )
    # Per-graph build:
    q_rot = graph.rope(q, freqs, output_scale=mscale)
    k_rot = graph.rope(k, freqs, output_scale=mscale)
    o = graph.sdpa(q=q_rot, k=k_rot, v=v, attn_scale=1.0/sqrt(d))
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position: int) -> float:
    """Inverse of the canonical RoPE rotations-per-period formula.

    Returns the (fractional) dim index whose period in radians equals
    ``max_position / num_rotations``.
    """
    return (dim * math.log(max_position / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(base))


def _yarn_find_correction_range(beta_fast: float, beta_slow: float, dim: int, base: float, max_position: int) -> Tuple[int, int]:
    """Compute [low, high] dim indices over which to ramp from interpolation -> extrapolation.

    beta_fast (alpha in paper, default 32) -> high end of ramp (above which dims extrapolate)
    beta_slow (beta in paper, default 1)   -> low end of ramp  (below which dims interpolate)
    """
    low = math.floor(_yarn_find_correction_dim(beta_fast, dim, base, max_position))
    high = math.ceil(_yarn_find_correction_dim(beta_slow, dim, base, max_position))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: int, high: int, num_dims: int, *, device, dtype) -> torch.Tensor:
    """Build a [num_dims] linear ramp clamped to [0, 1], rising from low to high."""
    if low == high:
        high = high + 0.001  # avoid division by zero
    idx = torch.arange(num_dims, device=device, dtype=dtype)
    return torch.clamp((idx - low) / (high - low), 0.0, 1.0)


def yarn_get_mscale(scaling_factor: float, mscale_factor: float = 1.0) -> float:
    """Per-paper attention temperature: ``0.1 * mscale_factor * log(s) + 1``.

    Returns 1.0 if ``scaling_factor <= 1`` (YARN inactive).
    DeepSeek-V3 uses ``mscale_factor=1.0`` and ``mscale_all_dim_factor=1.0``,
    yielding final mscale = 0.1*log(s)+1 (e.g. ~1.369 for s=40).
    """
    if scaling_factor <= 1.0:
        return 1.0
    return 0.1 * mscale_factor * math.log(scaling_factor) + 1.0


def compute_yarn_inv_freq(
    head_dim: int,
    *,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_position: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute the YARN-blended inverse frequency table.

    Returns a 1D tensor of length ``head_dim // 2`` with the inv_freq values
    used by RoPE: ``freqs[s, i] = position[s] * inv_freq[i]``.

    For ``scaling_factor <= 1`` this reduces to the standard RoPE inv_freq
    (no blending applied).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"YARN/RoPE require even head_dim, got {head_dim}")

    half = head_dim // 2
    arange_even = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq_extra = 1.0 / (base ** (arange_even / head_dim))

    if scaling_factor <= 1.0:
        return inv_freq_extra

    inv_freq_inter = inv_freq_extra / scaling_factor
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, head_dim, base, original_max_position)
    # mask=1 on extrapolation dims (high freqs), 0 on interpolation dims (low freqs)
    mask = 1.0 - _yarn_linear_ramp_mask(low, high, half, device=device, dtype=dtype)
    return inv_freq_inter * (1.0 - mask) + inv_freq_extra * mask


def compute_yarn_freqs(
    max_seq_len: int,
    head_dim: int,
    *,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_position: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale_factor: float = 1.0,
    mscale_all_dim_factor: float = 0.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, float]:
    """Compute the YARN ``freqs`` tensor and corresponding ``mscale``.

    Returns
    -------
    freqs : torch.Tensor
        Shape ``[max_seq_len, 1, 1, head_dim]``, layout matching the existing
        cuDNN RoPE op contract. The first ``head_dim/2`` columns hold raw
        angle values ``position * inv_freq``; the last ``head_dim/2``
        columns are zero-padded (kernel only reads first ``head_dim/2``).
    mscale : float
        Attention-temperature scalar to pass as ``output_scale`` on each
        RoPE node. For ``scaling_factor <= 1`` this is 1.0 (no-op).

    Notes
    -----
    The DeepSeek-V2/V3 effective mscale uses two factors:
      ``mscale = mscale_get(s, mscale_factor) / mscale_get(s, mscale_all_dim_factor)``
    With both factors equal (DSv3 default 1.0/1.0), this collapses to 1.0.
    With ``mscale_all_dim_factor=0`` (default here), only the numerator applies.
    """
    inv_freq = compute_yarn_inv_freq(
        head_dim,
        base=base,
        scaling_factor=scaling_factor,
        original_max_position=original_max_position,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        device=device,
        dtype=dtype,
    )
    half = head_dim // 2
    pos = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.outer(pos, inv_freq)  # [max_seq_len, half]

    freqs = torch.zeros(max_seq_len, 1, 1, head_dim, device=device, dtype=dtype)
    freqs[:, 0, 0, :half] = angles

    mscale_num = yarn_get_mscale(scaling_factor, mscale_factor)
    mscale_den = yarn_get_mscale(scaling_factor, mscale_all_dim_factor) if mscale_all_dim_factor != 0.0 else 1.0
    mscale = mscale_num / mscale_den
    return freqs, mscale


__all__ = [
    "compute_yarn_freqs",
    "compute_yarn_inv_freq",
    "yarn_get_mscale",
]
