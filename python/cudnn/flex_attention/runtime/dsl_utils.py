# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.

import contextlib
from functools import lru_cache
import re
from typing import Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensor

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import NumericMeta, dsl_user_op
from cutlass.cute.runtime import from_dlpack

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


def _cute_dsl_version() -> tuple[int, int, int]:
    """Return the installed CUTLASS DSL version as a three-integer tuple."""
    version = getattr(cutlass, "__version__", None)
    try:
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    except TypeError as exc:
        raise RuntimeError(f"Cannot parse CUTLASS DSL version {version!r}") from exc
    if match is None:
        raise RuntimeError(f"Cannot parse CUTLASS DSL version {version!r}")
    return tuple(int(part) for part in match.groups())


_CUTE_DSL_VERSION = _cute_dsl_version()


def _cute_dsl_bulk_copy_self_elects() -> bool:
    """Return whether cute.copy elects a lane for bulk-async copies."""
    return (4, 6, 0) <= _CUTE_DSL_VERSION < (4, 6, 2)


_BULK_COPY_SELF_ELECTS = _cute_dsl_bulk_copy_self_elects()


def bulk_copy_elect_one():
    """Select a lane only when the installed DSL does not do so internally.

    CUTLASS DSL 4.6.0 and 4.6.1 add an internal warp-collective election to
    ``cute.copy`` for bulk-async atoms. Nesting that copy inside
    ``cute.arch.elect_one()`` leaves one lane at the inner collective and
    deadlocks the warp. Earlier versions and 4.6.2 or newer require the outer
    guard.
    """
    if _BULK_COPY_SELF_ELECTS:
        return contextlib.nullcontext()
    return cute.arch.elect_one()


@dsl_user_op
def bulk_copy(atom, src, dst, *, loc=None, ip=None, **kwargs):
    """Issue a bulk-async ``cute.copy`` with version-correct lane election."""
    with bulk_copy_elect_one():
        cute.copy(atom, src, dst, loc=loc, ip=ip, **kwargs)


def struct_scalar_ptr(field):
    """Return a pointer for a scalar shared-storage field across DSL versions.

    CUTLASS DSL 4.6 wraps scalar fields and exposes their address through
    ``.ptr``. Older releases return the pointer directly.
    """
    return field.ptr if hasattr(field, "ptr") else field


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


@lru_cache
def get_device_capacity(device: torch.device = None) -> Tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _has_aligned_pointer(tensor: torch.Tensor, align_bytes: int) -> bool:
    address = tensor.storage_offset() * tensor.element_size() if isinstance(tensor, FakeTensor) else tensor.data_ptr()
    return address % align_bytes == 0


def _is_aligned_layout(tensor: torch.Tensor, align_bytes: int) -> bool:
    if tensor.stride(-1) != 1 or not _has_aligned_pointer(tensor, align_bytes):
        return False
    stride_alignment = max(1, align_bytes // tensor.element_size())
    return all(stride == 0 or stride % stride_alignment == 0 for stride in tensor.stride()[:-1])


def maybe_contiguous(
    tensor: torch.Tensor | None,
    align_bytes: int = 16,
) -> torch.Tensor | None:
    """Canonicalize a tensor to the pointer and stride alignment kernel ABI."""
    if tensor is None:
        return None
    if tensor.is_contiguous():
        return tensor if _has_aligned_pointer(tensor, align_bytes) else tensor.clone(memory_format=torch.contiguous_format)
    if not _has_aligned_pointer(tensor, align_bytes):
        return tensor.clone(memory_format=torch.contiguous_format)
    return tensor if _is_aligned_layout(tensor, align_bytes) else tensor.contiguous()


def assume_strides_aligned(t):
    """Assume all strides except the last are divisible by 128 bits.

    Python int strides (e.g., stride=0 from GQA expand) are kept as-is
    since they're static and don't need alignment assumptions.
    """
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    """Rebuild a tensor with 128-bit aligned stride assumptions. Passes through None."""
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))


def as_bshkrd_tensor(
    tensor: cute.Tensor,
    h_k: cutlass.Int32,
    h_r: cutlass.Int32,
    varlen: bool,
) -> cute.Tensor:
    """Normalize (B,S,H,D)/(S,H,D) to a (B,S,H_k,H_r,D) view."""
    if cutlass.const_expr(cute.rank(tensor.layout) == 5):
        if cutlass.const_expr(varlen):
            return cute.make_tensor(
                tensor.iterator,
                cute.make_layout(
                    tensor.shape,
                    stride=(
                        0,
                        tensor.stride[1],
                        tensor.stride[2],
                        tensor.stride[3],
                        tensor.stride[4],
                    ),
                ),
            )
        return tensor
    if cutlass.const_expr(cute.rank(tensor.layout) == 4):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[0], tensor.shape[1], h_k, h_r, tensor.shape[3]),
                stride=(
                    tensor.stride[0],
                    tensor.stride[1],
                    tensor.stride[2] * h_r,
                    tensor.stride[2],
                    tensor.stride[3],
                ),
            ),
        )
    assert cutlass.const_expr(cute.rank(tensor.layout) == 3), "Expected rank-3 varlen tensor"
    assert cutlass.const_expr(varlen), "Rank-3 input is only valid for varlen"
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (1, tensor.shape[0], h_k, h_r, tensor.shape[2]),
            stride=(
                0,
                tensor.stride[0],
                tensor.stride[1] * h_r,
                tensor.stride[1],
                tensor.stride[2],
            ),
        ),
    )


def to_cute_tensor(
    t,
    assumed_align=16,
    leading_dim=-1,
    fully_dynamic=False,
    enable_tvm_ffi=True,
):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def get_broadcast_dims(tensor: torch.Tensor) -> Tuple[bool, ...]:
    """Return tuple of bools indicating which dims have stride=0 (broadcast).

    This is useful for compile keys since CuTe's mark_layout_dynamic() keeps
    stride=0 as static, meaning kernels compiled with different broadcast
    patterns are not interchangeable.
    """
    return tuple(s == 0 for s in tensor.stride())
