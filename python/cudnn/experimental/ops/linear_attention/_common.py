"""
Shared utilities for the linear-attention torch custom ops.

All reference math runs in fp32 for stability; casts back to the original
dtype happen at the op boundary.

Public tensor layout:

    q, k:  [B, T, H,  K]
    v:     [B, T, H,  V]
    g/alpha, beta, b, w: see per-variant docstrings.

Internally the reference math operates on the "batch-major" ``[B, H, T, *]``
layout by permuting ``[B, T, H, *]``. The helpers below keep that conversion
in one place.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def bthd_to_bhtd(x: torch.Tensor) -> torch.Tensor:
    """[B, T, H, D] -> [B, H, T, D]."""
    return x.transpose(1, 2).contiguous()


def bhtd_to_bthd(x: torch.Tensor) -> torch.Tensor:
    """[B, H, T, D] -> [B, T, H, D]."""
    return x.transpose(1, 2).contiguous()


def bth_to_bht(x: torch.Tensor) -> torch.Tensor:
    """[B, T, H] -> [B, H, T]."""
    return x.transpose(1, 2).contiguous()


def bht_to_bth(x: torch.Tensor) -> torch.Tensor:
    """[B, H, T] -> [B, T, H]."""
    return x.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Chunk-factor builders
# ---------------------------------------------------------------------------


def chunk_factors_scalar(alpha_c: torch.Tensor):
    """Per-token *scalar* decay factors (used by GDN).

    Args:
        alpha_c: ``(..., T_chunks, B_chunk)`` per-chunk scalar alphas in ``(0, 1]``.

    Returns:
        Lambda: ``(..., T_chunks, B_chunk)``        — ``prod_{r=i+1}^{B-1} a_r``
        Gamma:  ``(..., T_chunks, B_chunk)``        — ``prod_{r=0}^{i}     a_r``
        L:      ``(..., T_chunks, B_chunk, B_chunk)`` — within-chunk causal decay
        g:      ``(..., T_chunks)``                  — full-chunk decay
    """
    B = alpha_c.shape[-1]
    log_a = torch.log(alpha_c)
    cum = torch.cumsum(log_a, dim=-1)
    Gamma = torch.exp(cum)
    g = Gamma[..., -1]
    rev = log_a.flip(-1).cumsum(-1).flip(-1)
    rev_padded = torch.nn.functional.pad(rev, (0, 1))
    Lambda = torch.exp(rev_padded[..., 1:])
    log_L = cum.unsqueeze(-1) - cum.unsqueeze(-2)
    L = torch.exp(log_L)
    i = torch.arange(B, device=alpha_c.device).view(B, 1)
    j = torch.arange(B, device=alpha_c.device).view(1, B)
    L = torch.where(i >= j, L, torch.zeros_like(L))
    return Lambda, Gamma, L, g


def chunk_factors_channelwise(alpha_c: torch.Tensor):
    """Per-channel decay factors (used by KDA and GDN-2).

    Args:
        alpha_c: ``(..., T_chunks, B_chunk, C)`` per-chunk per-channel alphas.

    Returns:
        Lambda: ``(..., T_chunks, B_chunk, C)``
        Gamma:  ``(..., T_chunks, B_chunk, C)``
        g:      ``(..., T_chunks, C)``
    """
    log_a = torch.log(alpha_c)
    cum = torch.cumsum(log_a, dim=-2)
    Gamma = torch.exp(cum)
    g = Gamma[..., -1, :]
    Lambda = torch.exp(cum[..., -1:, :] - cum)
    return Lambda, Gamma, g


# ---------------------------------------------------------------------------
# Padding / chunking
# ---------------------------------------------------------------------------


def pad_to_multiple(t: torch.Tensor, multiple: int, dim: int, value: float = 0.0) -> Tuple[torch.Tensor, int]:
    """Pad ``t`` along ``dim`` to the next multiple of ``multiple``."""
    n = t.shape[dim]
    pad = (-n) % multiple
    if pad == 0:
        return t, 0
    # torch.nn.functional.pad operates on the last dims; convert to that form.
    pad_spec = [0, 0] * (t.ndim - 1 - dim) + [0, pad]
    return torch.nn.functional.pad(t, pad_spec, value=value), pad


# ---------------------------------------------------------------------------
# Optional initial-state handling
# ---------------------------------------------------------------------------


def compute_dtype_for(input_dtype: torch.dtype) -> torch.dtype:
    """Compute dtype to use inside the reference: fp64 if input is fp64, else fp32."""
    return torch.float64 if input_dtype == torch.float64 else torch.float32


def maybe_zero_state(initial_state: Optional[torch.Tensor], B: int, H: int, K: int, V: int, dtype, device) -> torch.Tensor:
    """Return a zero state ``[B, H, K, V]`` if ``initial_state`` is None."""
    if initial_state is None:
        return torch.zeros(B, H, K, V, dtype=dtype, device=device)
    if initial_state.shape != (B, H, K, V):
        raise ValueError(
            f"initial_state shape {tuple(initial_state.shape)} does not match expected (B, H, K, V) = ({B}, {H}, {K}, {V})"
        )
    return initial_state.to(dtype=dtype, device=device)
