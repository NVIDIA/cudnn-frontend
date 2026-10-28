# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical (natural row-major) layouts for the contiguous grouped GEMM kernels.

The kernels historically required callers to pre-permute every operand into the
kernel-facing form: A/C/D as (m, x, 1) with a trailing unit L mode, B as (n, k, l)
k-major strided views, prob as (m, 1, 1), and the block scale factors as 6-D
MMA-tiled (32, 4, mn//128, 4, rest_k, l) strided views of a dense buffer. The
canonical forms are the natural buffers instead: A (sum_m, k) row-major, B (l, n, k)
row-major, prob (sum_m,), and dense C-contiguous scale-factor buffers of any shape.
Scale-factor bytes must already be packed in the required MMA-tiled physical order;
flattening does not convert ordinary logical row-major scales.

Canonical operands compile at their own rank and bind straight to the kernel with no
per-call host work: the host-side APIs derive the canonical fakes from the
kernel-facing ones (sharing the symbolic dims), and the jit launcher normalizes the
cute tensors back to the kernel-facing views at trace time. The kernels rebuild the
MMA-tiled SF layouts from the A/B/D shapes on device and read only the SF base
pointers, so a canonical SF buffer compiles as a rank-matched fully-dynamic tensor.
Kernel-facing inputs pass through every helper unchanged.
"""

from __future__ import annotations

from dataclasses import replace
import math

import cutlass
import cutlass.cute as cute

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
    stride = tensor.stride() if callable(tensor.stride) else tensor.stride
    return stride[2] == 1 and stride[1] != 1 and stride[0] != 1


def normalize_b(tensor, n_major=False):
    """(is_canonical, kernel-facing (n, k, l) view) for a weight tensor.

    A canonical n-major weight is (l, k, n) row-major, e.g. a column-wise weight buffer.
    """
    if is_canonical_b(tensor):
        return True, tensor.permute(2, 1, 0) if n_major else tensor.permute(1, 2, 0)
    return False, tensor


def b_nkl(tensor, b_major="k"):
    """(n, k, l) of a weight tensor in kernel-facing or canonical layout."""
    if not is_canonical_b(tensor):
        return tuple(tensor.shape)
    l, x, y = tensor.shape
    return (y, x, l) if b_major == "n" else (x, y, l)


def normalize_prob(tensor):
    """(is_canonical, kernel-facing (m, 1, 1) view) for a per-row tensor."""
    if tensor is not None and tensor.ndim == 1:
        return True, tensor.view(-1, 1, 1)
    return False, tensor


def live_kernel_view(tensor, normalize, canonical: bool, name: str):
    """Kernel-facing view of an execute operand, which must use the layout the kernel was compiled for."""
    is_canonical, view = normalize(tensor)
    if is_canonical != canonical:
        raise ValueError(f"{name} layout mismatch: the kernel was compiled for the {'canonical' if canonical else 'legacy'} layout")
    return view


def operand_signature(tensor):
    """Hashable (shape, stride, dtype, device) of an execute operand, or None; data pointers excluded."""
    if tensor is None:
        return None
    from cudnn.tensor_adapter import get_device, get_shape, get_strides, is_torch_tensor

    if is_torch_tensor(tensor):
        return (tensor.shape, tensor.stride(), tensor.dtype, tensor.device)
    return (get_shape(tensor), get_strides(tensor), tensor.dtype, get_device(tensor))


def validate_once(cache: dict, check, operands) -> None:
    """``check(*operands)`` once per distinct operand metadata; the outcome depends on nothing else."""
    signature = tuple(operand_signature(tensor) for tensor in operands)
    if signature not in cache:
        check(*operands)
        if len(cache) >= 1024:
            cache.clear()
        cache[signature] = True


def is_flat_sf(tensor) -> bool:
    """True when a scale-factor tensor is a dense C-contiguous buffer (canonical form).

    The legacy MMA-tiled 6-D views are non-contiguous except for degenerate unit-dim
    cases where both interpretations address identical memory.
    """
    return tensor is not None and tensor.is_contiguous()


def check_canonical_contiguous(a, b, prob):
    for name, tensor, canonical in (
        ("a_tensor", a, a.ndim == 2),
        ("b_tensor", b, is_canonical_b(b)),
        ("prob_tensor", prob, prob is not None and prob.ndim == 1),
    ):
        if canonical and not tensor.is_contiguous():
            raise ValueError(f"Canonical {name} must be contiguous")


def check_packed_sf(sfa, sfb, a_rows, b_rows, experts, rest_k):
    for name, tensor, rows, groups in (("sfa_tensor", sfa, a_rows, 1), ("sfb_tensor", sfb, b_rows, experts)):
        if tensor is None:
            continue
        if is_flat_sf(tensor):
            if tensor.numel() != 512 * ((rows + 127) // 128) * rest_k * groups:
                raise ValueError(f"{name} must contain the complete MMA-packed scale buffer")
        elif tensor.ndim != 6:
            raise ValueError(f"{name} must be a contiguous packed buffer or a legacy 6-D MMA view")


def layout_desc(api, tensor, name):
    if tensor is None:
        return None
    return replace(api._make_tensor_desc(tensor, name=name, canonical=True), dtype=tensor.dtype)


def check_sf_shape(api, desc, flat: bool, mma_shape, name: str):
    """Check an SF descriptor against its MMA-tiled shape, or its element count when flat."""
    if desc is None:
        return
    if flat:
        api._check_tensor_shape((math.prod(desc.shape),), (math.prod(mma_shape),), name)
    else:
        api._check_tensor_shape(desc, mma_shape, name)


def raw_uint8_dtype(tensor, dtype=None, sf_vec_size=None):
    """Logical dtype of a raw uint8 operand, or None when ``tensor`` is typed.

    A data operand needs ``dtype`` (FP8 E4M3 or E5M2), because bare uint8 data already
    means packed FP4. A scale-factor operand (``sf_vec_size`` given) follows its block
    size: 32 -> E8M0, 16 -> E4M3.
    """
    import torch

    if tensor is None or tensor.dtype != torch.uint8:
        return None
    if sf_vec_size is not None:
        return torch.float8_e8m0fnu if sf_vec_size == 32 else torch.float8_e4m3fn
    if dtype not in (None, torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"raw uint8 data must hold float8_e4m3fn or float8_e5m2, got {dtype}")
    return dtype


def raw_desc(api, name, desc, dtype):
    """``desc`` relabeled to ``dtype`` for every check; the kernel relabels the raw uint8 operand at trace time."""
    from cudnn.datatypes import _convert_to_cutlass_data_type

    import torch

    if desc is None or dtype is None:
        return desc
    logical = api.raw_dtypes[name] = _convert_to_cutlass_data_type(dtype)
    return replace(desc, dtype=dtype if isinstance(desc.dtype, torch.dtype) else logical)


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


def refake(fake, shape, stride, dtype=None):
    return cute.runtime.make_fake_tensor(dtype=dtype or fake.element_type, shape=shape, stride=stride, assumed_align=16)


def raw_fake(api, name, fake):
    """The fake over raw uint8 storage when ``name`` is a raw operand, so the call takes the uint8 tensor as is."""
    if fake is None or name not in api.raw_dtypes:
        return fake
    return refake(fake, fake.shape, fake.stride, cutlass.Uint8)


def canonical_mx_fake(fake, canonical: bool):
    """Kernel-facing (m, x, 1) fake -> canonical (m, x) fake sharing its symbolic dims."""
    if fake is None or not canonical:
        return fake
    return refake(fake, fake.shape[:2], fake.stride[:2])


def canonical_b_fake(fake, canonical: bool, n_major=False):
    """Kernel-facing (n, k, l) fake -> canonical (l, n, k), or (l, k, n) when n-major."""
    if fake is None or not canonical:
        return fake
    n, k, l = fake.shape
    sn, sk, sl = fake.stride
    if n_major:
        return refake(fake, (l, k, n), (sl, sk, sn))
    return refake(fake, (l, n, k), (sl, sn, sk))


def canonical_prob_fake(fake, canonical: bool):
    """Kernel-facing (m, 1, 1) fake -> canonical (m,) fake sharing its symbolic dim."""
    if fake is None or not canonical:
        return fake
    return refake(fake, fake.shape[:1], fake.stride[:1])


# Trace time, inside the jit launcher: canonical cute tensors -> kernel-facing views.


def kernel_facing_dtype(kernel, name, tensor):
    """Raw uint8 operand -> its logical dtype (same width, no data change); typed operands pass through."""
    dtype = getattr(kernel, "raw_dtypes", {}).get(name)
    if tensor is None or dtype is None:
        return tensor
    return cute.make_tensor(cute.recast_ptr(tensor.iterator, dtype=dtype), tensor.layout)


def kernel_facing_mx(tensor):
    """Canonical (m, x) -> (m, x, 1) with the dense unit-L stride; rank 3 passes through."""
    if tensor is None or cute.rank(tensor) != 2:
        return tensor
    m, x = tensor.shape
    sm, sx = tensor.stride
    return cute.make_tensor(tensor.iterator, cute.make_layout((m, x, 1), stride=(sm, sx, m * x)))


def kernel_facing_b(tensor, n_major=False):
    """Canonical (l, n, k) -> k-major (n, k, l), or (l, k, n) -> n-major (n, k, l); kernel-facing forms pass through."""
    if tensor is None:
        return tensor
    innermost = tensor.stride[2]
    if cute.is_static(innermost) and innermost == 1:
        return cute.make_tensor(tensor.iterator, cute.select(tensor.layout, [2, 1, 0] if n_major else [1, 2, 0]))
    return tensor


def kernel_facing_wgrad_sf(tensor, rows, tokens, sf_vec_size):
    """Flat wgrad SF buffer -> (round_up(rows, 128), round_up(ceil(tokens / sf_vec_size), 4)); rank 2 passes through."""
    if tensor is None or cute.rank(tensor) != 1:
        return tensor
    cols = (tokens + sf_vec_size * 4 - 1) // (sf_vec_size * 4) * 4
    return cute.make_tensor(tensor.iterator, cute.make_layout(((rows + 127) // 128 * 128, cols), stride=(cols, 1)))


def kernel_facing_fp4(tensor):
    """Packed FP4 carried as uint8 -> 4-bit elements with twice the innermost extent; other element types pass through.

    The JAX bridge presents uint8 arrays as Int8.
    """
    if tensor is None or tensor.element_type not in (cutlass.Uint8, cutlass.Int8):
        return tensor
    return cute.make_tensor(cute.recast_ptr(tensor.iterator, dtype=cutlass.Float4E2M1FN), cute.recast_layout(4, 8, tensor.layout))


def kernel_facing_prob(tensor):
    """Canonical (m,) -> (m, 1, 1); rank 3 passes through."""
    if tensor is None or cute.rank(tensor) != 1:
        return tensor
    m = tensor.shape[0]
    return cute.make_tensor(tensor.iterator, cute.make_layout((m, 1, 1), stride=(1, m, m)))
