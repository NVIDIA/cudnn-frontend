# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public FE-OSS API for the SM10x HSTU attention kernels."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import logging
import math
from typing import Optional, Tuple

from cuda.bindings import driver as cuda
import torch

from cudnn.api_base import APIBase, TupleDict

from . import _interface

_logger = logging.getLogger(__name__)
_API_CACHE_CAPACITY = 128
_FWD_CACHE = OrderedDict()
_BWD_CACHE = OrderedDict()


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _API_CACHE_CAPACITY:
        cache.popitem(last=False)


def _tensor_signature(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _has_non_overlapping_strides(tensor: torch.Tensor) -> bool:
    """Conservatively validate non-overlap from host-visible tensor metadata."""
    dimensions = sorted((int(stride), int(size)) for size, stride in zip(tensor.shape, tensor.stride()) if size > 1)
    covered_span = 1
    for stride, size in dimensions:
        if stride < covered_span:
            return False
        covered_span += (size - 1) * stride
    return True


def _require_16_byte_alignment(tensor: torch.Tensor, name: str) -> None:
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(f"{name} storage must be 16-byte aligned")


def _storage_span(tensor: torch.Tensor) -> Tuple[int, int]:
    start = tensor.data_ptr()
    last_element_offset = sum((int(size) - 1) * int(stride) for size, stride in zip(tensor.shape, tensor.stride()) if size > 0)
    return start, start + (last_element_offset + 1) * tensor.element_size()


def _are_disjoint_fused_thd_slices(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Recognize disjoint Q/K/V slices from one fused packed THD allocation."""
    if (
        a.ndim != 3
        or b.ndim != 3
        or a.shape != b.shape
        or a.stride() != b.stride()
        or a.dtype != b.dtype
        or a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    ):
        return False
    _, heads, head_dim = a.shape
    token_stride, head_stride, dim_stride = map(int, a.stride())
    token_chunk = int(heads) * int(head_dim)
    if dim_stride != 1 or head_stride != head_dim or token_stride < 3 * token_chunk:
        return False

    a_begin = int(a.storage_offset()) % token_stride
    b_begin = int(b.storage_offset()) % token_stride
    a_end = a_begin + token_chunk
    b_end = b_begin + token_chunk
    if a_end > token_stride or b_end > token_stride:
        return False
    return a_end <= b_begin or b_end <= a_begin


def _require_disjoint_writes(
    writes: Tuple[Tuple[str, torch.Tensor], ...],
    reads: Tuple[Tuple[str, Optional[torch.Tensor]], ...],
) -> None:
    comparisons = tuple(reads) + tuple(writes)
    for write_name, write_tensor in writes:
        write_start, write_end = _storage_span(write_tensor)
        for other_name, other_tensor in comparisons:
            if other_tensor is write_tensor and other_name == write_name:
                continue
            if other_tensor is None or other_tensor.device != write_tensor.device:
                continue
            other_start, other_end = _storage_span(other_tensor)
            spans_overlap = write_start < other_end and other_start < write_end
            if spans_overlap and not _are_disjoint_fused_thd_slices(write_tensor, other_tensor):
                raise ValueError(f"{write_name} storage must not overlap {other_name} storage")


def _validate_cu_seqlens_metadata(
    cu_seqlens: torch.Tensor,
    name: str,
) -> int:
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got rank {cu_seqlens.ndim}")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {cu_seqlens.dtype}")
    if not cu_seqlens.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if cu_seqlens.numel() < 2:
        raise ValueError(f"{name} must contain at least two entries")
    _require_16_byte_alignment(cu_seqlens, name)
    return int(cu_seqlens.numel() - 1)


def _resolve_max_seqlen(
    requested: Optional[int],
    total_tokens: int,
    name: str,
) -> int:
    value = int(total_tokens) if requested is None else int(requested)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _require_same_cuda_device(
    reference: torch.Tensor,
    tensors: Tuple[Tuple[str, Optional[torch.Tensor]], ...],
) -> None:
    if not reference.is_cuda:
        raise ValueError("q_tensor must be a CUDA tensor")
    for name, tensor in tensors:
        if tensor is None:
            continue
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {tensor.device}")


def _stream_context(
    stream: Optional[cuda.CUstream | torch.cuda.Stream],
    device: torch.device,
):
    if stream is None:
        return nullcontext()
    return torch.cuda.stream(_as_torch_stream(stream, device))


def _as_torch_stream(
    stream: cuda.CUstream | torch.cuda.Stream,
    device: torch.device,
) -> torch.cuda.Stream:
    if isinstance(stream, torch.cuda.Stream):
        if stream.device != device:
            raise ValueError(f"stream must be on {device}, got {stream.device}")
        return stream
    if int(stream) == 0:
        return torch.cuda.default_stream(device)
    return torch.cuda.ExternalStream(int(stream), device=device)


def _record_streams(
    tensors: Tuple[Optional[torch.Tensor], ...],
    stream: Optional[cuda.CUstream | torch.cuda.Stream],
    device: torch.device,
) -> None:
    """Keep raw-pointer operands alive until an explicit-stream launch completes."""
    if stream is None:
        return
    consumer = _as_torch_stream(stream, device)
    for tensor in tensors:
        if tensor is not None and tensor.is_cuda:
            tensor.record_stream(consumer)


def _empty_grad_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor, memory_format=torch.preserve_format)


def _validate_page_metadata(
    page_ids: torch.Tensor,
    page_indptrs: torch.Tensor,
    batch_size: int,
) -> None:
    if page_ids.ndim != 1 or page_ids.dtype != torch.int32:
        raise ValueError("page_ids_tensor must be a rank-1 torch.int32 tensor")
    if not page_ids.is_contiguous():
        raise ValueError("page_ids_tensor must be contiguous")
    _require_16_byte_alignment(page_ids, "page_ids_tensor")
    if page_indptrs.ndim != 1 or page_indptrs.dtype != torch.int32 or page_indptrs.shape[0] != batch_size + 1:
        raise ValueError(f"page_indptrs_tensor must be torch.int32 with shape " f"({batch_size + 1},)")
    if not page_indptrs.is_contiguous():
        raise ValueError("page_indptrs_tensor must be contiguous")
    _require_16_byte_alignment(page_indptrs, "page_indptrs_tensor")


class _HSTUBase(APIBase):
    """Validation shared by the paired HSTU forward/backward APIs."""

    def _init_common(
        self,
        *,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_cu_seqlens_q: torch.Tensor,
        sample_cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        window_size: Tuple[int, int],
        alpha: float,
        scaling_seqlen: Optional[float],
        sample_func: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self._warn_experimental_api()

        self._sample_q = sample_q
        self._sample_k = sample_k
        self._sample_v = sample_v
        self._sample_cu_seqlens_q = sample_cu_seqlens_q
        self._sample_cu_seqlens_k = sample_cu_seqlens_k
        self._sample_func = sample_func

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.cu_seqlens_q_desc = self._make_tensor_desc(sample_cu_seqlens_q, name="cu_seqlens_q")
        self.cu_seqlens_k_desc = self._make_tensor_desc(sample_cu_seqlens_k, name="cu_seqlens_k")
        self.func_desc = self._make_tensor_desc(sample_func, name="func")

        self.max_seqlen_q = None if max_seqlen_q is None else int(max_seqlen_q)
        self.max_seqlen_k = None if max_seqlen_k is None else int(max_seqlen_k)
        if len(window_size) != 2:
            raise ValueError(f"window_size must contain two integers, got {window_size}")
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.alpha = float(alpha)
        self.scaling_seqlen = None if scaling_seqlen is None else float(scaling_seqlen)

        self.batch_size = None
        self.head_dim = None
        self.num_heads = None
        self.is_causal = None
        self.is_local = None

    def _check_common(self, supported_head_dims: Tuple[int, ...]) -> None:
        q = self._sample_q
        k = self._sample_k
        v = self._sample_v
        cu_q = self._sample_cu_seqlens_q
        cu_k = self._sample_cu_seqlens_k

        _require_same_cuda_device(
            q,
            (
                ("k_tensor", k),
                ("v_tensor", v),
                ("cu_seqlens_q_tensor", cu_q),
                ("cu_seqlens_k_tensor", cu_k),
                ("func_tensor", self._sample_func),
            ),
        )

        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.ndim != 3:
                raise ValueError(f"{name}_tensor must use packed THD rank-3 layout, got " f"shape {tuple(tensor.shape)}")
            if tensor.shape[0] <= 0 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
                raise ValueError(f"{name}_tensor dimensions must be positive")
            if not _has_non_overlapping_strides(tensor):
                raise ValueError(f"{name}_tensor must have non-overlapping strides")
            _require_16_byte_alignment(tensor, f"{name}_tensor")

        total_q, q_heads, q_dim = q.shape
        total_k, k_heads, k_dim = k.shape
        if v.shape != k.shape:
            raise ValueError(f"v_tensor must have the same shape as k_tensor; got " f"{tuple(v.shape)} and {tuple(k.shape)}")
        if q_heads != k_heads:
            raise ValueError("HSTU SM100 currently supports MHA only; q, k, and v must " f"have the same head count, got {q_heads} and {k_heads}")
        if q_dim != k_dim:
            raise ValueError(f"q, k, and v head dimensions must match, got {q_dim} and {k_dim}")
        if q_dim not in supported_head_dims:
            raise ValueError(f"head_dim must be one of {supported_head_dims}, got {q_dim}")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"q_tensor must have dtype torch.float16 or torch.bfloat16, got {q.dtype}")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q_tensor, k_tensor, and v_tensor must have the same dtype")

        q_batch_size = _validate_cu_seqlens_metadata(cu_q, "cu_seqlens_q_tensor")
        k_batch_size = _validate_cu_seqlens_metadata(cu_k, "cu_seqlens_k_tensor")
        if q_batch_size != k_batch_size:
            raise ValueError("cu_seqlens_q_tensor and cu_seqlens_k_tensor must describe the " "same batch size")
        self.batch_size = q_batch_size
        self.max_seqlen_q = _resolve_max_seqlen(self.max_seqlen_q, total_q, "max_seqlen_q")
        self.max_seqlen_k = _resolve_max_seqlen(self.max_seqlen_k, total_k, "max_seqlen_k")
        if self.max_seqlen_q > self.max_seqlen_k:
            raise ValueError("max_seqlen_q must be <= max_seqlen_k for the HSTU SM100 kernel")

        if not math.isfinite(self.alpha):
            raise ValueError(f"alpha must be finite, got {self.alpha}")
        if self.scaling_seqlen is None:
            self.scaling_seqlen = float(self.max_seqlen_q)
        if not math.isfinite(self.scaling_seqlen) or self.scaling_seqlen <= 0.0:
            raise ValueError(f"scaling_seqlen must be positive and finite, got {self.scaling_seqlen}")

        window_left, window_right = self.window_size
        if window_left < -1 or window_left > self.max_seqlen_k:
            raise ValueError(f"window_size[0] must be -1 or in [0, {self.max_seqlen_k}], " f"got {window_left}")
        if window_right < -1 or window_right > self.max_seqlen_k:
            raise ValueError(f"window_size[1] must be -1 or in [0, {self.max_seqlen_k}], " f"got {window_right}")
        normalized_left = self.max_seqlen_k if window_left < 0 else window_left
        normalized_right = self.max_seqlen_k if window_right < 0 else window_right
        self.is_causal = normalized_left == self.max_seqlen_k and normalized_right == 0
        self.is_local = (normalized_left < self.max_seqlen_k or normalized_right < self.max_seqlen_k) and not self.is_causal

        if self._sample_func is not None:
            func = self._sample_func
            if func.ndim != 3:
                raise ValueError("func_tensor must have shape (1, n_func, total_q + 256)")
            if func.dtype != torch.int32:
                raise ValueError(f"func_tensor must have dtype torch.int32, got {func.dtype}")
            _require_16_byte_alignment(func, "func_tensor")
            if func.shape[0] != 1 or func.shape[1] <= 0 or func.shape[1] % 2 == 0:
                raise ValueError("func_tensor must have shape (1, n_func, total_q + 256) " "with n_func positive and odd")
            if func.shape[2] < total_q + 256:
                raise ValueError(f"func_tensor's last dimension must be at least total_q + " f"256 ({total_q + 256}), got {func.shape[2]}")
            if self.is_causal or self.is_local:
                raise ValueError("arbitrary func masking cannot be combined with causal or local " "masking")

        major, minor = torch.cuda.get_device_capability(q.device)
        if major != 10:
            raise RuntimeError("HSTU SM100 requires an SM10x GPU; found " f"SM{major}{minor} on {q.device}")

        self.head_dim = int(q_dim)
        self.num_heads = int(q_heads)

    def _check_runtime_common(
        self,
        *,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        cu_seqlens_q_tensor: torch.Tensor,
        cu_seqlens_k_tensor: torch.Tensor,
        func_tensor: Optional[torch.Tensor],
    ) -> None:
        for name, tensor, desc in (
            ("q_tensor", q_tensor, self.q_desc),
            ("k_tensor", k_tensor, self.k_desc),
            ("v_tensor", v_tensor, self.v_desc),
            ("cu_seqlens_q_tensor", cu_seqlens_q_tensor, self.cu_seqlens_q_desc),
            ("cu_seqlens_k_tensor", cu_seqlens_k_tensor, self.cu_seqlens_k_desc),
        ):
            if tuple(tensor.shape) != tuple(desc.shape):
                raise ValueError(f"{name} shape must match the compiled shape {desc.shape}, " f"got {tuple(tensor.shape)}")
            if tensor.dtype != desc.dtype or tensor.device != desc.device or tuple(tensor.stride()) != tuple(desc.stride):
                raise ValueError(f"{name} dtype/device/stride must match the compiled descriptor")
            if not _has_non_overlapping_strides(tensor):
                raise ValueError(f"{name} must have non-overlapping strides")
            _require_16_byte_alignment(tensor, name)

        _require_same_cuda_device(
            q_tensor,
            (
                ("k_tensor", k_tensor),
                ("v_tensor", v_tensor),
                ("cu_seqlens_q_tensor", cu_seqlens_q_tensor),
                ("cu_seqlens_k_tensor", cu_seqlens_k_tensor),
                ("func_tensor", func_tensor),
            ),
        )
        _validate_cu_seqlens_metadata(cu_seqlens_q_tensor, "cu_seqlens_q_tensor")
        _validate_cu_seqlens_metadata(cu_seqlens_k_tensor, "cu_seqlens_k_tensor")

        if (func_tensor is None) != (self.func_desc is None):
            raise ValueError("func_tensor presence must match the compiled configuration")
        if func_tensor is not None:
            if tuple(func_tensor.shape) != tuple(self.func_desc.shape):
                raise ValueError("func_tensor shape changed after compilation")
            if func_tensor.dtype != torch.int32:
                raise ValueError("func_tensor must have dtype torch.int32")
            if tuple(func_tensor.stride()) != tuple(self.func_desc.stride):
                raise ValueError("func_tensor stride changed after compilation")
            _require_16_byte_alignment(func_tensor, "func_tensor")

    def _release_compile_samples(self) -> None:
        self._sample_q = None
        self._sample_k = None
        self._sample_v = None
        self._sample_cu_seqlens_q = None
        self._sample_cu_seqlens_k = None
        self._sample_func = None


class HSTUFwdSm100(_HSTUBase):
    """Explicit compile/execute API for packed HSTU forward on SM10x."""

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_cu_seqlens_q: torch.Tensor,
        sample_cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        window_size: Tuple[int, int] = (-1, -1),
        alpha: float = 1.0,
        scaling_seqlen: Optional[float] = None,
        sample_func: Optional[torch.Tensor] = None,
        sample_paged_kv: Optional[torch.Tensor] = None,
        sample_page_ids: Optional[torch.Tensor] = None,
        sample_page_indptrs: Optional[torch.Tensor] = None,
    ) -> None:
        self._init_common(
            sample_q=sample_q,
            sample_k=sample_k,
            sample_v=sample_v,
            sample_cu_seqlens_q=sample_cu_seqlens_q,
            sample_cu_seqlens_k=sample_cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            window_size=window_size,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
            sample_func=sample_func,
        )
        self._sample_o = sample_o
        self._sample_paged_kv = sample_paged_kv
        self._sample_page_ids = sample_page_ids
        self._sample_page_indptrs = sample_page_indptrs
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.paged_kv_desc = self._make_tensor_desc(sample_paged_kv, name="paged_kv")
        self.page_ids_desc = self._make_tensor_desc(sample_page_ids, name="page_ids")
        self.page_indptrs_desc = self._make_tensor_desc(sample_page_indptrs, name="page_indptrs")

    def check_support(self) -> bool:
        if self._is_supported:
            return True
        self._check_common((64, 128, 256))
        q = self._sample_q
        o = self._sample_o
        _require_same_cuda_device(
            q,
            (
                ("o_tensor", o),
                ("paged_kv_tensor", self._sample_paged_kv),
                ("page_ids_tensor", self._sample_page_ids),
                ("page_indptrs_tensor", self._sample_page_indptrs),
            ),
        )
        if o.shape != q.shape or o.dtype != q.dtype:
            raise ValueError("o_tensor must have the same shape and dtype as q_tensor")
        if not o.is_contiguous():
            raise ValueError("o_tensor must be contiguous")
        _require_16_byte_alignment(o, "o_tensor")

        paged = self._sample_paged_kv
        page_ids = self._sample_page_ids
        page_indptrs = self._sample_page_indptrs
        if paged is None:
            if page_ids is not None or page_indptrs is not None:
                raise ValueError("page_ids_tensor and page_indptrs_tensor require paged_kv_tensor")
        else:
            if not self.is_causal or self.is_local or self._sample_func is not None:
                raise ValueError("paged KV requires causal attention and cannot be combined " "with local or arbitrary masking")
            if page_ids is None or page_indptrs is None:
                raise ValueError("paged KV requires page_ids_tensor and page_indptrs_tensor")
            expected_tail = (2, 128, self.num_heads, self.head_dim)
            if paged.ndim != 5 or paged.shape[0] <= 0 or tuple(paged.shape[1:]) != expected_tail:
                raise ValueError(
                    "paged_kv_tensor must have shape "
                    f"(num_pages, 2, 128, {self.num_heads}, {self.head_dim}) "
                    "with num_pages > 0, "
                    f"got {tuple(paged.shape)}"
                )
            if paged.dtype != q.dtype:
                raise ValueError("paged_kv_tensor dtype must match q_tensor")
            if not paged.is_contiguous():
                raise ValueError("paged_kv_tensor must be contiguous")
            _require_16_byte_alignment(paged, "paged_kv_tensor")
            _validate_page_metadata(
                page_ids,
                page_indptrs,
                self.batch_size,
            )

        _require_disjoint_writes(
            (("o_tensor", o),),
            (
                ("q_tensor", q),
                ("k_tensor", self._sample_k),
                ("v_tensor", self._sample_v),
                ("cu_seqlens_q_tensor", self._sample_cu_seqlens_q),
                ("cu_seqlens_k_tensor", self._sample_cu_seqlens_k),
                ("func_tensor", self._sample_func),
                ("paged_kv_tensor", paged),
                ("page_ids_tensor", page_ids),
                ("page_indptrs_tensor", page_indptrs),
            ),
        )

        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        _interface.hstu_varlen_fwd_100(
            self._sample_q,
            self._sample_k,
            self._sample_v,
            self._sample_cu_seqlens_q,
            self._sample_cu_seqlens_k,
            self.max_seqlen_q,
            self.max_seqlen_k,
            self.window_size[0],
            self.window_size[1],
            self.alpha,
            self._sample_func,
            self._sample_paged_kv,
            self._sample_page_ids,
            self._sample_page_indptrs,
            self.scaling_seqlen,
            out=self._sample_o,
            _compile_only=True,
        )
        self._compiled_kernel = _interface.hstu_varlen_fwd_100
        self._release_compile_samples()
        self._sample_o = None
        self._sample_paged_kv = None
        self._sample_page_ids = None
        self._sample_page_indptrs = None

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        cu_seqlens_q_tensor: torch.Tensor,
        cu_seqlens_k_tensor: torch.Tensor,
        func_tensor: Optional[torch.Tensor] = None,
        paged_kv_tensor: Optional[torch.Tensor] = None,
        page_ids_tensor: Optional[torch.Tensor] = None,
        page_indptrs_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        if self._compiled_kernel is None:
            raise RuntimeError("HSTUFwdSm100 kernel is not compiled")
        self._check_runtime_common(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            cu_seqlens_q_tensor=cu_seqlens_q_tensor,
            cu_seqlens_k_tensor=cu_seqlens_k_tensor,
            func_tensor=func_tensor,
        )
        if tuple(o_tensor.shape) != tuple(self.o_desc.shape):
            raise ValueError("o_tensor shape changed after compilation")
        if o_tensor.dtype != self.o_desc.dtype or o_tensor.device != self.o_desc.device:
            raise ValueError("o_tensor dtype/device changed after compilation")
        if not o_tensor.is_contiguous():
            raise ValueError("o_tensor must be contiguous")
        _require_16_byte_alignment(o_tensor, "o_tensor")
        for name, tensor, desc in (
            ("paged_kv_tensor", paged_kv_tensor, self.paged_kv_desc),
            ("page_ids_tensor", page_ids_tensor, self.page_ids_desc),
            ("page_indptrs_tensor", page_indptrs_tensor, self.page_indptrs_desc),
        ):
            if (tensor is None) != (desc is None):
                raise ValueError(f"{name} presence changed after compilation")
            if tensor is not None and (
                tuple(tensor.shape) != tuple(desc.shape)
                or tensor.dtype != desc.dtype
                or tensor.device != desc.device
                or tuple(tensor.stride()) != tuple(desc.stride)
            ):
                raise ValueError(f"{name} specification changed after compilation")
        if paged_kv_tensor is not None:
            if not paged_kv_tensor.is_contiguous():
                raise ValueError("paged_kv_tensor must be contiguous")
            _require_16_byte_alignment(paged_kv_tensor, "paged_kv_tensor")
            _validate_page_metadata(
                page_ids_tensor,
                page_indptrs_tensor,
                self.batch_size,
            )
        _require_disjoint_writes(
            (("o_tensor", o_tensor),),
            (
                ("q_tensor", q_tensor),
                ("k_tensor", k_tensor),
                ("v_tensor", v_tensor),
                ("cu_seqlens_q_tensor", cu_seqlens_q_tensor),
                ("cu_seqlens_k_tensor", cu_seqlens_k_tensor),
                ("func_tensor", func_tensor),
                ("paged_kv_tensor", paged_kv_tensor),
                ("page_ids_tensor", page_ids_tensor),
                ("page_indptrs_tensor", page_indptrs_tensor),
            ),
        )

        with _stream_context(current_stream, q_tensor.device):
            _interface.hstu_varlen_fwd_100(
                q_tensor,
                k_tensor,
                v_tensor,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                self.max_seqlen_q,
                self.max_seqlen_k,
                self.window_size[0],
                self.window_size[1],
                self.alpha,
                func_tensor,
                paged_kv_tensor,
                page_ids_tensor,
                page_indptrs_tensor,
                self.scaling_seqlen,
                out=o_tensor,
            )
        _record_streams(
            (
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                func_tensor,
                paged_kv_tensor,
                page_ids_tensor,
                page_indptrs_tensor,
            ),
            current_stream,
            q_tensor.device,
        )


class HSTUBwdSm100(_HSTUBase):
    """Explicit compile/execute API for packed HSTU backward on SM10x."""

    def __init__(
        self,
        sample_do: torch.Tensor,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_dq: torch.Tensor,
        sample_dk: torch.Tensor,
        sample_dv: torch.Tensor,
        sample_cu_seqlens_q: torch.Tensor,
        sample_cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        window_size: Tuple[int, int] = (-1, -1),
        alpha: float = 1.0,
        scaling_seqlen: Optional[float] = None,
        sample_func: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> None:
        self._init_common(
            sample_q=sample_q,
            sample_k=sample_k,
            sample_v=sample_v,
            sample_cu_seqlens_q=sample_cu_seqlens_q,
            sample_cu_seqlens_k=sample_cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            window_size=window_size,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
            sample_func=sample_func,
        )
        self._sample_do = sample_do
        self._sample_dq = sample_dq
        self._sample_dk = sample_dk
        self._sample_dv = sample_dv
        self.do_desc = self._make_tensor_desc(sample_do, name="do")
        self.dq_desc = self._make_tensor_desc(sample_dq, name="dq")
        self.dk_desc = self._make_tensor_desc(sample_dk, name="dk")
        self.dv_desc = self._make_tensor_desc(sample_dv, name="dv")
        self.deterministic = bool(deterministic)

    def check_support(self) -> bool:
        if self._is_supported:
            return True
        self._check_common((64, 128, 256))
        q = self._sample_q
        k = self._sample_k
        v = self._sample_v
        _require_same_cuda_device(
            q,
            (
                ("do_tensor", self._sample_do),
                ("dq_tensor", self._sample_dq),
                ("dk_tensor", self._sample_dk),
                ("dv_tensor", self._sample_dv),
            ),
        )
        for name, tensor, expected in (
            ("do_tensor", self._sample_do, q),
            ("dq_tensor", self._sample_dq, q),
            ("dk_tensor", self._sample_dk, k),
            ("dv_tensor", self._sample_dv, v),
        ):
            if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
                raise ValueError(f"{name} must have shape {tuple(expected.shape)} and dtype " f"{expected.dtype}")
            if not _has_non_overlapping_strides(tensor):
                raise ValueError(f"{name} must have non-overlapping strides")
            _require_16_byte_alignment(tensor, name)
        if self.deterministic:
            raise NotImplementedError("deterministic HSTU backward is not supported by HSTU SM100")
        _require_disjoint_writes(
            (
                ("dq_tensor", self._sample_dq),
                ("dk_tensor", self._sample_dk),
                ("dv_tensor", self._sample_dv),
            ),
            (
                ("do_tensor", self._sample_do),
                ("q_tensor", q),
                ("k_tensor", k),
                ("v_tensor", v),
                ("cu_seqlens_q_tensor", self._sample_cu_seqlens_q),
                ("cu_seqlens_k_tensor", self._sample_cu_seqlens_k),
                ("func_tensor", self._sample_func),
            ),
        )
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        _interface.hstu_varlen_bwd_100(
            self._sample_do,
            self._sample_q,
            self._sample_k,
            self._sample_v,
            self._sample_cu_seqlens_q,
            self._sample_cu_seqlens_k,
            self.max_seqlen_q,
            self.max_seqlen_k,
            self._sample_dq,
            self._sample_dk,
            self._sample_dv,
            self.window_size[0],
            self.window_size[1],
            self.alpha,
            self._sample_func,
            self.deterministic,
            self.scaling_seqlen,
            _compile_only=True,
        )
        self._compiled_kernel = _interface.hstu_varlen_bwd_100
        self._release_compile_samples()
        self._sample_do = None
        self._sample_dq = None
        self._sample_dk = None
        self._sample_dv = None

    def execute(
        self,
        do_tensor: torch.Tensor,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        cu_seqlens_q_tensor: torch.Tensor,
        cu_seqlens_k_tensor: torch.Tensor,
        func_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    ) -> None:
        if self._compiled_kernel is None:
            raise RuntimeError("HSTUBwdSm100 kernel is not compiled")
        self._check_runtime_common(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            cu_seqlens_q_tensor=cu_seqlens_q_tensor,
            cu_seqlens_k_tensor=cu_seqlens_k_tensor,
            func_tensor=func_tensor,
        )
        for name, tensor, desc in (
            ("do_tensor", do_tensor, self.do_desc),
            ("dq_tensor", dq_tensor, self.dq_desc),
            ("dk_tensor", dk_tensor, self.dk_desc),
            ("dv_tensor", dv_tensor, self.dv_desc),
        ):
            if (
                tuple(tensor.shape) != tuple(desc.shape)
                or tensor.dtype != desc.dtype
                or tensor.device != desc.device
                or tuple(tensor.stride()) != tuple(desc.stride)
            ):
                raise ValueError(f"{name} specification changed after compilation")
            if not _has_non_overlapping_strides(tensor):
                raise ValueError(f"{name} must have non-overlapping strides")
            _require_16_byte_alignment(tensor, name)

        _require_disjoint_writes(
            (
                ("dq_tensor", dq_tensor),
                ("dk_tensor", dk_tensor),
                ("dv_tensor", dv_tensor),
            ),
            (
                ("do_tensor", do_tensor),
                ("q_tensor", q_tensor),
                ("k_tensor", k_tensor),
                ("v_tensor", v_tensor),
                ("cu_seqlens_q_tensor", cu_seqlens_q_tensor),
                ("cu_seqlens_k_tensor", cu_seqlens_k_tensor),
                ("func_tensor", func_tensor),
            ),
        )

        with _stream_context(current_stream, q_tensor.device):
            _interface.hstu_varlen_bwd_100(
                do_tensor,
                q_tensor,
                k_tensor,
                v_tensor,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                self.max_seqlen_q,
                self.max_seqlen_k,
                dq_tensor,
                dk_tensor,
                dv_tensor,
                self.window_size[0],
                self.window_size[1],
                self.alpha,
                func_tensor,
                False,
                self.scaling_seqlen,
            )
        _record_streams(
            (
                do_tensor,
                q_tensor,
                k_tensor,
                v_tensor,
                dq_tensor,
                dk_tensor,
                dv_tensor,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                func_tensor,
            ),
            current_stream,
            q_tensor.device,
        )


def hstu_attention_forward(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    cu_seqlens_q_tensor: torch.Tensor,
    cu_seqlens_k_tensor: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_size: Tuple[int, int] = (-1, -1),
    alpha: float = 1.0,
    scaling_seqlen: Optional[float] = None,
    func_tensor: Optional[torch.Tensor] = None,
    paged_kv_tensor: Optional[torch.Tensor] = None,
    page_ids_tensor: Optional[torch.Tensor] = None,
    page_indptrs_tensor: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
) -> TupleDict:
    """Allocate and compute packed HSTU forward on an SM10x GPU."""
    _validate_cu_seqlens_metadata(cu_seqlens_q_tensor, "cu_seqlens_q_tensor")
    _validate_cu_seqlens_metadata(cu_seqlens_k_tensor, "cu_seqlens_k_tensor")
    resolved_max_q = _resolve_max_seqlen(max_seqlen_q, q_tensor.shape[0], "max_seqlen_q")
    resolved_max_k = _resolve_max_seqlen(max_seqlen_k, k_tensor.shape[0], "max_seqlen_k")
    resolved_scaling = float(resolved_max_q if scaling_seqlen is None else scaling_seqlen)
    with torch.cuda.device(q_tensor.device), _stream_context(stream, q_tensor.device):
        o_tensor = torch.empty(
            q_tensor.shape,
            dtype=q_tensor.dtype,
            device=q_tensor.device,
        )
    cache_key = (
        _tensor_signature(q_tensor),
        _tensor_signature(k_tensor),
        _tensor_signature(v_tensor),
        _tensor_signature(cu_seqlens_q_tensor),
        _tensor_signature(cu_seqlens_k_tensor),
        _tensor_signature(func_tensor),
        _tensor_signature(paged_kv_tensor),
        _tensor_signature(page_ids_tensor),
        _tensor_signature(page_indptrs_tensor),
        resolved_max_q,
        resolved_max_k,
        tuple(window_size),
        float(alpha),
        resolved_scaling,
    )
    api = _cache_get(_FWD_CACHE, cache_key)
    if api is None:
        api = HSTUFwdSm100(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_cu_seqlens_q=cu_seqlens_q_tensor,
            sample_cu_seqlens_k=cu_seqlens_k_tensor,
            max_seqlen_q=resolved_max_q,
            max_seqlen_k=resolved_max_k,
            window_size=window_size,
            alpha=alpha,
            scaling_seqlen=resolved_scaling,
            sample_func=func_tensor,
            sample_paged_kv=paged_kv_tensor,
            sample_page_ids=page_ids_tensor,
            sample_page_indptrs=page_indptrs_tensor,
        )
        api.check_support()
        api.compile()
        _cache_put(_FWD_CACHE, cache_key, api)
    api.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        cu_seqlens_q_tensor=cu_seqlens_q_tensor,
        cu_seqlens_k_tensor=cu_seqlens_k_tensor,
        func_tensor=func_tensor,
        paged_kv_tensor=paged_kv_tensor,
        page_ids_tensor=page_ids_tensor,
        page_indptrs_tensor=page_indptrs_tensor,
        current_stream=stream,
    )
    return TupleDict(o_tensor=o_tensor)


def hstu_attention_backward(
    do_tensor: torch.Tensor,
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    cu_seqlens_q_tensor: torch.Tensor,
    cu_seqlens_k_tensor: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_size: Tuple[int, int] = (-1, -1),
    alpha: float = 1.0,
    scaling_seqlen: Optional[float] = None,
    func_tensor: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    dq_tensor: Optional[torch.Tensor] = None,
    dk_tensor: Optional[torch.Tensor] = None,
    dv_tensor: Optional[torch.Tensor] = None,
) -> TupleDict:
    """Compute packed HSTU backward on an SM10x GPU.

    Any gradient output tensor that is not provided is allocated by this
    function. Caller-provided gradient outputs are overwritten and returned.
    """
    _validate_cu_seqlens_metadata(cu_seqlens_q_tensor, "cu_seqlens_q_tensor")
    _validate_cu_seqlens_metadata(cu_seqlens_k_tensor, "cu_seqlens_k_tensor")
    resolved_max_q = _resolve_max_seqlen(max_seqlen_q, q_tensor.shape[0], "max_seqlen_q")
    resolved_max_k = _resolve_max_seqlen(max_seqlen_k, k_tensor.shape[0], "max_seqlen_k")
    resolved_scaling = float(resolved_max_q if scaling_seqlen is None else scaling_seqlen)
    with torch.cuda.device(q_tensor.device), _stream_context(stream, q_tensor.device):
        if dq_tensor is None:
            dq_tensor = _empty_grad_like(q_tensor)
        if dk_tensor is None:
            dk_tensor = _empty_grad_like(k_tensor)
        if dv_tensor is None:
            dv_tensor = _empty_grad_like(v_tensor)
    cache_key = (
        _tensor_signature(do_tensor),
        _tensor_signature(q_tensor),
        _tensor_signature(k_tensor),
        _tensor_signature(v_tensor),
        _tensor_signature(dq_tensor),
        _tensor_signature(dk_tensor),
        _tensor_signature(dv_tensor),
        _tensor_signature(cu_seqlens_q_tensor),
        _tensor_signature(cu_seqlens_k_tensor),
        _tensor_signature(func_tensor),
        resolved_max_q,
        resolved_max_k,
        tuple(window_size),
        float(alpha),
        resolved_scaling,
        bool(deterministic),
    )
    api = _cache_get(_BWD_CACHE, cache_key)
    if api is None:
        api = HSTUBwdSm100(
            sample_do=do_tensor,
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_dq=dq_tensor,
            sample_dk=dk_tensor,
            sample_dv=dv_tensor,
            sample_cu_seqlens_q=cu_seqlens_q_tensor,
            sample_cu_seqlens_k=cu_seqlens_k_tensor,
            max_seqlen_q=resolved_max_q,
            max_seqlen_k=resolved_max_k,
            window_size=window_size,
            alpha=alpha,
            scaling_seqlen=resolved_scaling,
            sample_func=func_tensor,
            deterministic=deterministic,
        )
        api.check_support()
        api.compile()
        _cache_put(_BWD_CACHE, cache_key, api)
    api.execute(
        do_tensor=do_tensor,
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        dq_tensor=dq_tensor,
        dk_tensor=dk_tensor,
        dv_tensor=dv_tensor,
        cu_seqlens_q_tensor=cu_seqlens_q_tensor,
        cu_seqlens_k_tensor=cu_seqlens_k_tensor,
        func_tensor=func_tensor,
        current_stream=stream,
    )
    return TupleDict(
        dq_tensor=dq_tensor,
        dk_tensor=dk_tensor,
        dv_tensor=dv_tensor,
    )


__all__ = [
    "HSTUFwdSm100",
    "HSTUBwdSm100",
    "hstu_attention_forward",
    "hstu_attention_backward",
]
