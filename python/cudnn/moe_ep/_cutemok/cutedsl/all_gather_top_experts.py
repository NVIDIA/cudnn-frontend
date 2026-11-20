# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""All-gather of the routes: a rank copies its ``[T, topk]`` routes into its slice of the ``[EP, T, topk]`` buffer.

One single-thread CTA per ``chunk_bytes`` chunk: a bulk copy of the chunk into shared memory, then a bulk
store to the raw address ``multicast_ptr + rank * T * topk * 4 + chunk offset``.  The kernel waits for no
other rank: the cross-rank barrier (``barrier_all``) stays with the caller.

The caller owns the destination buffers and their symmetric-memory handle until that barrier and the
consumers finish; ``record_stream`` is not a replacement for keeping the handle alive.  Extents, addresses,
the rank and the chunk size are runtime.
"""

import threading

CHUNK_BYTES = 2048  # Public default, not a compilation parameter.
from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, check_device

_CODEGEN_EPOCH = "all-gather-top-experts-runtime-o2-v2"
from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack
from ._host_common import persistent_executor
from ._host_common import current_cuda_context_key as _current_cuda_context_key
from ._host_common import ValidatedCalls, current_stream_handle, tensor_layout_signature, typed_key
from ._host_common import current_device_index as _current_device_index, device_properties as _device_properties


@dsl_user_op
def _load_chunk(dst_smem, src_addr: Int64, mbar, num_bytes: Int32, *, loc=None, ip=None):
    # Raw-address CTA-scope bulk copy, without a cache-policy operand.
    llvm.inline_asm(
        None,
        [dst_smem.toint(loc=loc, ip=ip).ir_value(), src_addr.ir_value(), num_bytes.ir_value(), mbar.toint(loc=loc, ip=ip).ir_value()],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_chunk(dst_addr: Int64, src_smem, num_bytes: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [dst_addr.ir_value(), src_smem.toint(loc=loc, ip=ip).ir_value(), num_bytes.ir_value()],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


class _RouteCopy:
    @cute.jit
    def __call__(self, top_experts: cute.Tensor, multicast_addr: Int64, numel: Int32, rank: Int32, chunk_bytes: Int32, stream: cuda.CUstream):
        self.kernel(top_experts, multicast_addr, numel, rank, chunk_bytes).launch(
            grid=(Int64(numel) * Int64(4) // Int64(chunk_bytes), 1, 1), block=(1, 1, 1), smem=chunk_bytes + 1024, stream=stream
        )

    @cute.kernel
    def kernel(self, top_experts: cute.Tensor, multicast_addr: Int64, numel: Int32, rank: Int32, chunk_bytes: Int32):
        # The first 1024 B hold the mbarrier, so the payload stays 1024 B aligned.
        base = cute.arch.get_dyn_smem(Uint8, alignment=1024)
        mbar = cute.recast_ptr(base, dtype=Int64)
        arena = (base + Int64(1024)).align(min_align=1024)
        offset_bytes = Int64(cute.arch.block_idx()[0]) * Int64(chunk_bytes)
        source = top_experts.iterator.toint() + offset_bytes
        destination = multicast_addr + Int64(rank) * Int64(numel) * Int64(4) + offset_bytes
        cute.arch.mbarrier_init(mbar, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.mbarrier_arrive_and_expect_tx(mbar, chunk_bytes)
        _load_chunk(arena, source, mbar, chunk_bytes)
        cute.arch.mbarrier_wait(mbar, 0)
        cute.arch.fence_proxy("async.shared", space="cta")
        _store_chunk(destination, arena, chunk_bytes)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=False)


_KERNEL = _RouteCopy()
_EXECUTOR_CACHE = {}
_EXECUTOR_CACHE_DEVICES = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()
_VALIDATED_CALLS = ValidatedCalls()


def _validate_inputs(top_experts, output_buffer, multicast_ptr, rank, chunk_bytes):
    import torch

    # Error types: ValueError throughout, except TypeError for the buffer's dtype and for the
    # multicast pointer.
    for tensor in (top_experts, output_buffer):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise ValueError("top_experts and output_buffer must be CUDA tensors")
    if top_experts.dtype != torch.int32:
        raise ValueError("top_experts must be CUDA int32")
    if output_buffer.dtype != torch.int32:
        raise TypeError("all_gather_top_experts_buffer must have dtype torch.int32")
    for tensor in (top_experts, output_buffer):
        if not tensor.is_contiguous() or tensor.device != top_experts.device:
            raise ValueError("tensors must be contiguous on the same CUDA device")
        if tensor.data_ptr() % 16:
            raise ValueError("tensor addresses must be 16-byte aligned")
    if top_experts.ndim != 2 or any(int(d) <= 0 for d in top_experts.shape):
        raise ValueError(f"top_experts must be [T,topk] with positive dimensions; got {list(top_experts.shape)}")
    num_tokens, topk = map(int, top_experts.shape)
    if output_buffer.ndim != 3 or tuple(output_buffer.shape[1:]) != (num_tokens, topk):
        raise ValueError(f"output_buffer must have shape [EP,T,topk]; got {list(output_buffer.shape)} for T={num_tokens}, topk={topk}")
    world_size = int(output_buffer.shape[0])
    if not 0 < world_size < 2**31:
        raise ValueError(f"EP must be a positive int32; got {world_size}")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError(f"rank must be an integer in [0,EP) for EP={world_size}")
    if type(chunk_bytes) is not int or not 0 < chunk_bytes < 2**31 - 1024 or chunk_bytes % 16:
        raise ValueError(f"chunk_bytes must be positive, 16-byte aligned and int32-safe; got {chunk_bytes!r}")
    numel = num_tokens * topk
    rank_bytes = numel * 4
    if numel >= 2**31 or rank_bytes % chunk_bytes:
        raise ValueError("rank elements must fit int32 and rank bytes must divide into whole chunks")
    if world_size * rank_bytes >= 2**63:
        raise ValueError("output byte extent must fit int64")
    if type(multicast_ptr) is not int or not 0 < multicast_ptr < 2**63 or multicast_ptr % 16:
        raise TypeError("multicast_ptr must be a positive, 16-byte-aligned int64 address")
    source, output = top_experts.data_ptr(), output_buffer.data_ptr()
    if source < output + world_size * rank_bytes and output < source + rank_bytes:
        raise ValueError("input and output_buffer must not overlap")
    device = top_experts.device
    index = device.index if device.index is not None else _current_device_index()
    if _current_device_index() != index:
        raise RuntimeError("the input CUDA device must be current")
    if chunk_bytes + 1024 > _device_properties(device).shared_memory_per_block_optin:
        raise ValueError("chunk_bytes exceeds dynamic shared-memory capacity")
    return device, index, numel


def all_gather_top_experts(top_experts, output_buffer, multicast_ptr, rank, chunk_bytes=CHUNK_BYTES):
    """Copy this rank's routes (``top_experts``, int32 ``[T, topk]``) into slice ``rank`` of the
    ``[EP, T, topk]`` buffer addressed by ``multicast_ptr``, in ``chunk_bytes`` chunks (a multiple of 16
    that divides ``T * topk * 4``).  Pointer ownership and the cross-rank barrier stay with the caller.

    With an ordinary ``output_buffer.data_ptr()`` as ``multicast_ptr`` only the local buffer is written.
    """
    import torch

    # One host path, as the other CuTe DSL ops: the full check only on a new call signature; the
    # address checks (alignment, overlap) on every call.
    try:
        signature = (
            tensor_layout_signature((top_experts, output_buffer)),
            typed_key(multicast_ptr, rank, chunk_bytes),
            _current_device_index(),
            _current_cuda_context_key(),
        )
        validated = _VALIDATED_CALLS.get(signature)
    except (AttributeError, TypeError):  # not tensors / unhashable: the full check reports it
        signature = validated = None
    if validated is None:
        device, index, numel = _validate_inputs(top_experts, output_buffer, multicast_ptr, rank, chunk_bytes)
        validated = (device, index, numel, int(output_buffer.shape[0]))
        if signature is not None:
            _VALIDATED_CALLS.put(signature, validated)
    else:
        device, index, numel, world_size = validated
        source, output = top_experts.data_ptr(), output_buffer.data_ptr()
        if source % 16 or output % 16:
            raise ValueError("tensor addresses must be 16-byte aligned")
        if source < output + world_size * numel * 4 and output < source + numel * 4:
            raise ValueError("input and output_buffer must not overlap")
    device, index, numel = validated[:3]
    # Contiguous singleton axes may have arbitrary strides. Rebind metadata only.
    top_experts = top_experts.as_strided(top_experts.shape, (int(top_experts.shape[1]), 1))
    stream = current_stream_handle(index)
    # int32 rank-2 contiguous tensor ABI; extents, addresses and scalars are runtime.
    key = (_CODEGEN_EPOCH, index, _current_cuda_context_key(), "i32-rank2-contiguous", _COMPILE_OPTIONS)
    executor = _EXECUTOR_CACHE.get(key)
    if executor is None:
        executor = _compile_executor(top_experts, multicast_ptr, numel, rank, chunk_bytes, stream, key, device, index)
    executor(top_experts, multicast_ptr, numel, rank, chunk_bytes, stream)
    # No record_stream: like PyTorch's own ops (see fwd_epilogue_bf16).


def _compile_executor(top_experts, multicast_ptr, numel, rank, chunk_bytes, stream, key, device, index):
    import torch

    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(key)
        if executor is None:
            check_device(device, "route copy")
            arguments = (
                from_dlpack(top_experts, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
                Int64(multicast_ptr),
                Int32(numel),
                Int32(rank),
                Int32(chunk_bytes),
                stream,
            )
            executor = persistent_executor(("all_gather_top_experts", key[0], *key[3:]), lambda: cute.compile(_KERNEL, *arguments, options=_COMPILE_OPTIONS))
            _EXECUTOR_CACHE[key] = executor
            _EXECUTOR_CACHE_DEVICES.add(index)
    return executor


def compile_all_gather_top_experts(top_experts, multicast_ptr) -> None:
    """The plan step: compile the kernel without launching.  ``top_experts`` only gives the tensor's rank and
    dtype (int32 ``[T, topk]``, 16-byte aligned), so a slice of the gathered buffer stands in for the caller's."""
    device = top_experts.device
    index = device.index if device.index is not None else _current_device_index()
    key = (_CODEGEN_EPOCH, index, _current_cuda_context_key(), "i32-rank2-contiguous", _COMPILE_OPTIONS)
    _compile_executor(top_experts, multicast_ptr, int(top_experts.numel()), 0, CHUNK_BYTES, current_stream_handle(index), key, device, index)


def clear_all_gather_top_experts_executor_cache(*, synchronize=True):
    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
        _VALIDATED_CALLS.clear()
