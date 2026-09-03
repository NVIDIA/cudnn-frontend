# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical (natural row-major) layouts for the contiguous grouped GEMM kernels.

The kernels historically required callers to pre-permute every operand into the
kernel-facing form: A/C/D as (m, x, 1) with a trailing unit L mode, B as (n, k, l)
k-major strided views, prob as (m, 1, 1), and the block scale factors as 6-D
MMA-tiled (32, 4, mn//128, 4, rest_k, l) strided views of a dense buffer. The
canonical forms are the natural buffers instead: A (sum_m, k) row-major, B (l, n, k)
row-major, prob (sum_m,), and dense C-contiguous scale-factor buffers of any shape.

Canonical operands compile at their own rank and bind straight to the kernel with no
per-call host work: the host-side APIs derive the canonical fakes from the
kernel-facing ones (sharing the symbolic dims), and the jit launcher normalizes the
cute tensors back to the kernel-facing views at trace time. The kernels rebuild the
MMA-tiled SF layouts from the A/B/D shapes on device and read only the SF base
pointers, so a canonical SF buffer compiles as a rank-matched fully-dynamic tensor.
Kernel-facing inputs pass through every helper unchanged.
"""

from __future__ import annotations

import math

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


# Host side: classify sample tensors and derive the kernel-facing views for the descriptors.


def normalize_mx(tensor):
    """(is_canonical, kernel-facing view) for an (m, x) / (m, x, 1) operand."""
    if tensor is not None and tensor.ndim == 2:
        return True, tensor.unsqueeze(-1)
    return False, tensor


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


def normalize_b(tensor):
    """(is_canonical, kernel-facing (n, k, l) view) for a weight tensor."""
    if is_canonical_b(tensor):
        return True, tensor.permute(1, 2, 0)
    return False, tensor


def normalize_prob(tensor):
    """(is_canonical, kernel-facing (m, 1, 1) view) for a per-row tensor."""
    if tensor is not None and tensor.ndim == 1:
        return True, tensor.view(-1, 1, 1)
    return False, tensor


def is_flat_sf(tensor) -> bool:
    """True when a scale-factor tensor is a dense C-contiguous buffer (canonical form).

    The legacy MMA-tiled 6-D views are non-contiguous except for degenerate unit-dim
    cases where both interpretations address identical memory.
    """
    return tensor is not None and tensor.is_contiguous()


def check_sf_shape(api, desc, flat: bool, mma_shape, name: str):
    """Check an SF descriptor against its MMA-tiled shape, or its element count when flat."""
    if desc is None:
        return
    if flat:
        api._check_tensor_shape((math.prod(desc.shape),), (math.prod(mma_shape),), name)
    else:
        api._check_tensor_shape(desc, mma_shape, name)


# Compile time: canonical-rank fakes for the compiled signature.


def make_flat_sf_fake(api, desc):
    """Rank-matched fully-dynamic fake for a dense scale-factor buffer.

    The kernels read only the SF base pointer, so the signature needs nothing beyond
    dtype and rank; any dense buffer of that rank binds without a host-side view.
    """
    if desc is None:
        return None
    rank = len(desc.shape)
    return api._make_fake_cute_tensor(
        dtype=desc.dtype,
        shape=tuple(cute.sym_int() for _ in range(rank)),
        stride=tuple(cute.sym_int() for _ in range(rank)),
    )


def refake(fake, shape, stride):
    return cute.runtime.make_fake_tensor(dtype=fake.element_type, shape=shape, stride=stride, assumed_align=16)


def canonical_mx_fake(fake, canonical: bool):
    """Kernel-facing (m, x, 1) fake -> canonical (m, x) fake sharing its symbolic dims."""
    if fake is None or not canonical:
        return fake
    return refake(fake, fake.shape[:2], fake.stride[:2])


def canonical_b_fake(fake, canonical: bool):
    """Kernel-facing (n, k, l) fake -> canonical (l, n, k) fake sharing its symbolic dims."""
    if fake is None or not canonical:
        return fake
    n, k, l = fake.shape
    sn, sk, sl = fake.stride
    return refake(fake, (l, n, k), (sl, sn, sk))


def canonical_prob_fake(fake, canonical: bool):
    """Kernel-facing (m, 1, 1) fake -> canonical (m,) fake sharing its symbolic dim."""
    if fake is None or not canonical:
        return fake
    return refake(fake, fake.shape[:1], fake.stride[:1])


# Trace time, inside the jit launcher: canonical cute tensors -> kernel-facing views.


def kernel_facing_mx(tensor):
    """Canonical (m, x) -> (m, x, 1) with the dense unit-L stride; rank 3 passes through."""
    if tensor is None or cute.rank(tensor) != 2:
        return tensor
    m, x = tensor.shape
    sm, sx = tensor.stride
    return cute.make_tensor(tensor.iterator, cute.make_layout((m, x, 1), stride=(sm, sx, m * x)))


def kernel_facing_b(tensor):
    """Canonical (l, n, k) row-major -> k-major (n, k, l); kernel-facing forms pass through."""
    if tensor is None:
        return tensor
    innermost = tensor.stride[2]
    if cute.is_static(innermost) and innermost == 1:
        return cute.make_tensor(tensor.iterator, cute.select(tensor.layout, [1, 2, 0]))
    return tensor


def kernel_facing_prob(tensor):
    """Canonical (m,) -> (m, 1, 1); rank 3 passes through."""
    if tensor is None or cute.rank(tensor) != 1:
        return tensor
    m = tensor.shape[0]
    return cute.make_tensor(tensor.iterator, cute.make_layout((m, 1, 1), stride=(1, m, m)))
