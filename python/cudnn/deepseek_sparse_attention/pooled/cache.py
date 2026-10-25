# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared writes for pooled DSA index and tail caches."""

from functools import lru_cache

import cuda.bindings.driver as cuda
import torch

from cudnn.api_base import APIBase, TupleDict
from .indices import _require_dsl

_NAMES = ("keys", "gates", "ape", "cache", "tail", "pool_slots", "tail_slots", "positions", "cu_seqlens")


@lru_cache(maxsize=128)
def _compile_cache_update(device, capability, pool_size, page_size, cache_dtype, mode, integer_dtypes):
    _require_dsl()
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
    from ._cache_sm100 import PoolCacheKernel

    def matrix(dtype, width):
        return make_fake_tensor(dtype, (cute.sym_int(64), width), (cute.sym_int(64), 1), assumed_align=4)

    def vector(dtype):
        if dtype is None:
            return None
        return make_fake_tensor(cutlass.Int64 if dtype == torch.int64 else cutlass.Int32, (cute.sym_int(64),), (cute.sym_int(64),), assumed_align=4)

    width = page_size * {"bf16": 256, "fp8": 132, "fp4": 68}[cache_dtype]
    ape = make_fake_tensor(cutlass.Float32, (pool_size, 128), (cute.sym_int(64), 1), assumed_align=4)
    tail = make_fake_tensor(
        cutlass.BFloat16,
        (cute.sym_int(64), 2, pool_size, 128),
        (cute.sym_int(64), cute.sym_int(64), 128, 1),
        assumed_align=4,
    )
    with torch.cuda.device(device):
        return cute.compile(
            PoolCacheKernel(pool_size, page_size, cache_dtype, mode),
            matrix(cutlass.BFloat16, 128),
            matrix(cutlass.BFloat16, 128),
            ape,
            matrix(cutlass.Uint8, width),
            tail,
            *(vector(t) for t in integer_dtypes),
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options=compile_options(),
        )


class PoolCacheUpdate(APIBase):
    """Pool raw K/gates directly into paged BF16, FP8, or MXFP4 storage.

    Prefill batches start on pool boundaries. Decode uses actual packed
    queries and cu_seqlens; no padded-token scatter is required. Each
    request owns distinct tail slots and completed-pool cache slots.
    """

    def __init__(
        self,
        keys,
        gates,
        ape,
        cache,
        tail,
        pool_slots,
        tail_slots,
        positions=None,
        cu_seqlens=None,
        *,
        pool_size=4,
        page_size=64,
        cache_dtype="fp8",
        mode="prefill",
    ):
        super().__init__()
        self.pool_size, self.page_size = int(pool_size), int(page_size)
        self.cache_dtype, self.mode = cache_dtype, mode
        tensors = (keys, gates, ape, cache, tail, pool_slots, tail_slots, positions, cu_seqlens)
        self._descs = tuple(self._make_tensor_desc(t) for t in tensors)
        self._device = torch.device(keys.device)
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._integer_dtypes = tuple(t.dtype if t is not None else None for t in tensors[5:])

    def _validate(self, tensors, *, descriptors=False):
        if not descriptors and any(t is not None and not isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("execute() requires torch tensors")
        keys, gates, ape, cache, tail, pool_slots, tail_slots, positions, cu_seqlens = tensors
        ranks = (2, 2, 2, 2, 4, 1, 1)
        if any(t is None or t.ndim != rank for t, rank in zip(tensors, ranks)):
            raise ValueError("expected K/gate/APE/cache matrices, a 4D tail, and slot vectors")
        n = keys.shape[0]
        if n >= 2**31:
            raise ValueError("packed token count must fit int32 cu_seqlens")
        width = self.page_size * {"bf16": 256, "fp8": 132, "fp4": 68}[self.cache_dtype]
        expected = ((n, 128), (n, 128), (self.pool_size, 128), (cache.shape[0], width), (tail.shape[0], 2, self.pool_size, 128), (n,), (n,))
        dtypes = (torch.bfloat16, torch.bfloat16, torch.float32, torch.uint8, torch.bfloat16, *self._integer_dtypes[:2])
        for name, tensor, shape, dtype in zip(_NAMES, tensors, expected, dtypes):
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
        if any(t not in (torch.int32, torch.int64) for t in self._integer_dtypes[:2]):
            raise ValueError("pool_slots and tail_slots must be int32 or int64")
        if self.mode == "decode":
            if positions is None or cu_seqlens is None:
                raise ValueError("decode requires positions and cu_seqlens")
            if positions.shape != (n,) or positions.dtype != self._integer_dtypes[2] or positions.dtype not in (torch.int32, torch.int64):
                raise ValueError("positions must be an integer vector matching keys and the plan dtype")
            if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2 or cu_seqlens.dtype != torch.int32:
                raise ValueError("cu_seqlens must be an int32 vector with at least two entries")
            if cu_seqlens.shape[0] > 4 * (2**31 - 1):
                raise ValueError("request count exceeds the CUDA launch grid limit")
        elif positions is not None or cu_seqlens is not None:
            raise ValueError("prefill takes neither positions nor cu_seqlens")
        for name, tensor in zip(_NAMES, tensors):
            if tensor is None:
                continue
            if tensor.device != self._device or tensor.device.type != "cuda":
                raise ValueError(f"{name} must be on the planned CUDA device")
            strides = tensor.stride if descriptors else tensor.stride()
            if any(s <= 0 for s in strides):
                raise ValueError(f"{name} requires positive strides")
            if tensor.ndim == 2 and (strides[1] != 1 or strides[0] < tensor.shape[1]):
                raise ValueError(f"{name} requires non-overlapping contiguous rows")
            if name == "cache" and strides[0] % 4:
                raise ValueError("cache page stride must be a multiple of four bytes")
            if name == "tail" and (strides[3] != 1 or strides[2] != 128 or strides[1] < self.pool_size * 128 or strides[0] < 2 * strides[1]):
                raise ValueError("tail requires contiguous token vectors and non-overlapping K/gate blocks")
            if not descriptors and tensor.data_ptr() % 4:
                raise ValueError(f"{name} must be four-byte aligned")
        if not descriptors:
            spans = {}
            for name, tensor in zip(_NAMES, tensors):
                if tensor is not None and tensor.numel():
                    span = (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))) * tensor.element_size()
                    spans[name] = (tensor.data_ptr(), tensor.data_ptr() + span)
                if tensor is not None and tensor.requires_grad:
                    raise ValueError("PoolCacheUpdate has no autograd contract; detach inputs explicitly")
            for output in ("cache", "tail"):
                if output in spans:
                    low, high = spans[output]
                    for name, (other_low, other_high) in spans.items():
                        if name != output and low < other_high and other_low < high:
                            raise ValueError(f"{output} must not overlap {name}")

    def check_support(self):
        _require_dsl()
        if self.pool_size not in (4, 16) or self.page_size < 1 or self.mode not in ("prefill", "decode"):
            raise ValueError("expected pool_size 4 or 16, positive page_size, and prefill/decode mode")
        if self.cache_dtype not in ("bf16", "fp8", "fp4"):
            raise ValueError("cache_dtype must be bf16, fp8, or fp4 (MXFP4)")
        self._validate(self._descs, descriptors=True)
        self._capability = torch.cuda.get_device_capability(self._device)
        if self._capability not in ((10, 0), (10, 3)):
            raise NotImplementedError("Pooled DSA requires SM100 or SM103 Blackwell")
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        self._compiled_kernel = _compile_cache_update(
            self._device.index, self._capability, self.pool_size, self.page_size, self.cache_dtype, self.mode, self._integer_dtypes
        )

    def scratch_workspace_bytes(self):
        return 0

    def execute(self, keys, gates, ape, cache, tail, pool_slots, tail_slots, positions=None, cu_seqlens=None, current_stream=None):
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before execute()")
        tensors = (keys, gates, ape, cache, tail, pool_slots, tail_slots, positions, cu_seqlens)
        self._validate(tensors)
        if keys.shape[0]:
            with torch.cuda.device(self._device):
                stream = cuda.CUstream(int(current_stream)) if current_stream is not None else self._get_default_stream(None)
                self._compiled_kernel(*tensors, stream)
        return TupleDict(cache=cache, tail=tail)


def pool_cache_update_wrapper(
    keys,
    gates,
    ape,
    cache,
    tail,
    pool_slots,
    tail_slots,
    positions=None,
    cu_seqlens=None,
    *,
    pool_size=4,
    page_size=64,
    cache_dtype="fp8",
    mode="prefill",
    current_stream=None,
):
    """Update caller-owned caches; compile once before graph capture."""
    args = (keys, gates, ape, cache, tail, pool_slots, tail_slots, positions, cu_seqlens)
    plan = PoolCacheUpdate(*args, pool_size=pool_size, page_size=page_size, cache_dtype=cache_dtype, mode=mode)
    plan.compile()
    return plan.execute(*args, current_stream=current_stream)
