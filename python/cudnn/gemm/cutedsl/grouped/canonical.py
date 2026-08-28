# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers for canonical (natural row-major) grouped GEMM tensor layouts.

The contiguous grouped GEMM kernels historically required callers to pre-permute
every operand into the kernel-facing form: A/C/D as (m, x, 1) with a trailing unit
L mode, B as (n, k, l) k-major strided views, prob as (m, 1, 1), and the block
scale factors as 6-D MMA-tiled (32, 4, mn//128, 4, rest_k, l) strided views of a
dense buffer. These helpers additively accept the natural buffers instead --
A (sum_m, k) row-major, B (l, n, k) row-major, prob (sum_m,), and flat/dense
C-contiguous scale-factor buffers -- and normalize them to the kernel-facing form
with zero-copy views. Kernel-facing inputs pass through unchanged, so existing
callers are unaffected.

The scale-factor kernels rebuild the MMA-tiled SF layouts from the A/B/D shapes on
device and consume only the SF base pointers, so a canonical (C-contiguous) SF
buffer is compiled as a flat 1-D tensor: no MMA-permuted view is ever materialized.
"""

from __future__ import annotations

import cutlass.cute as cute

_cache_of_alpha_ones = {}


def default_alpha_ones(l: int, device):
    """Cached all-ones per-group scale for callers that don't scale per group."""
    import torch

    key = (l, str(device))
    alpha = _cache_of_alpha_ones.get(key)
    if alpha is None:
        alpha = torch.ones(l, dtype=torch.float32, device=device)
        _cache_of_alpha_ones[key] = alpha
    return alpha


def unsqueeze_l_dim(tensor):
    """Canonical (m, x) row-major -> kernel-facing (m, x, 1); 3-D passes through."""
    if tensor is not None and tensor.ndim == 2:
        return tensor.unsqueeze(-1)
    return tensor


def is_canonical_b(tensor) -> bool:
    """True for a canonical (l, n, k) row-major weight tensor.

    The kernel-facing forms keep a stride-1 k mode at dim 1 (k-major (n, k, l)) or a
    stride-1 n mode at dim 0 (n-major), so a stride-1 innermost dim 2 identifies the
    canonical form.
    """
    if tensor is None or tensor.ndim != 3:
        return False
    stride = tensor.stride()
    return stride[2] == 1 and stride[1] != 1 and stride[0] != 1


def to_kernel_b(tensor):
    """Canonical (l, n, k) row-major -> kernel-facing (n, k, l); other forms pass through."""
    if is_canonical_b(tensor):
        return tensor.permute(1, 2, 0)
    return tensor


def to_kernel_prob(tensor):
    """Canonical (m,) -> kernel-facing (m, 1, 1); other ranks pass through."""
    if tensor is not None and tensor.ndim == 1:
        return tensor.view(-1, 1, 1)
    return tensor


def is_flat_sf(tensor) -> bool:
    """True when a scale-factor tensor is a dense C-contiguous buffer (canonical form).

    The legacy MMA-tiled 6-D views are non-contiguous except for degenerate unit-dim
    cases where both interpretations address identical memory.
    """
    return tensor is not None and tensor.is_contiguous()


def to_kernel_sf(tensor, flat: bool):
    """Flatten a canonical scale-factor buffer to 1-D; legacy MMA views pass through."""
    if tensor is None or not flat:
        return tensor
    return tensor if tensor.ndim == 1 else tensor.view(-1)


def make_flat_sf_fake(api, desc):
    """Fake cute tensor for a flat (1-D, dynamic-length) scale-factor buffer.

    The kernels rebuild the SF layout from the GEMM operand shapes and read only the
    base pointer, so the compiled signature needs nothing beyond dtype and a dynamic
    length (always a multiple of the 32x4x4 = 512-element SF atom).
    """
    if desc is None:
        return None
    return api._make_fake_cute_tensor(
        dtype=desc.dtype,
        shape=(cute.sym_int(divisibility=512),),
        stride=(1,),
    )
