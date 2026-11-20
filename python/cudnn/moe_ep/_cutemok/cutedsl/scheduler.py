# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Schedule builder of one rank: one host executor, three device launches, no host fills.

The entry allocates its outputs; the all-gather of the routes is not part of it.  The three kernels
(scheduler_count, scheduler_pad, scheduler_schedule) run in order on the current stream; each writes every
entry it owns, so the host allocates with ``torch.empty`` only.  The ordering between them is the stream
order of the launches: count writes the per-column and per-column-block counts and the -1 fill of peer_rank,
pad turns them into per-(expert, peer) counts, padded rows and the smallest peer count, schedule places the
routed rows and stores the padded total.  No kernel needs a grid-wide barrier because no phase reads what
another block of the same launch writes.

Allocations per call: peer_rank, peer_token_idx, one 16-byte-sectioned ``outputs`` tensor (total | padded |
valid, returned as views) and one ``scratch`` tensor of the intermediate counts (E * EP * 1057 int32, about
2 MiB for 64 experts x EP 8), which no output references, so it is freed when the call returns.

Limits (ValueError): E = num_local_experts and EP = topk_all.shape[0] must satisfy EP * T * topk < 2**31 and
E * EP * 1057 < 2**31; the count kernel's histogram 128 * E bytes must fit the device's opt-in shared memory
per block (E <= 1816 at 232448 B); the schedule kernel's runtime tables 256 * EP + 4 * E bytes must fit the
34496 B left of its static 110 KiB block (EP <= 134 at E = 8; EP = 64 leaves E <= 4528).

T, topk, E, EP, the capacity and the rank are runtime; the kernels' constants are tile constants.
"""

import threading

from ._host_common import persistent_executor
from ._host_common import (
    ValidatedCalls,
    current_stream_handle,
    current_cuda_context_key as _current_cuda_context_key,
    row_major_view as _row_major_view,
    tensor_layout_signature,
    typed_key,
)
from ._host_common import current_device_index as _current_device_index, device_properties as _device_properties
from .scheduler_count import _KERNEL as _COUNT_KERNEL, CHUNKS, THREADS as _COLUMNS_TOTAL
from .scheduler_pad import _KERNEL as _PAD_KERNEL
from .scheduler_schedule import (
    _KERNEL as _SCHEDULE_KERNEL,
    _FIXED_BYTES as _SCHEDULE_FIXED_BYTES,
    _SMEM_BYTES as _SCHEDULE_SMEM_BYTES,
    schedule_shared_memory_bytes,
)
from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, check_device
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

_CODEGEN_EPOCH = "scheduler-local-runtime-o2-v10-outputs"
_ALIGN = 4  # int32 elements per 16 bytes: every scratch section starts 16-byte aligned (the consumers' requirement)


def _aligned(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


#: (experts, world) -> scheduler's output section offsets and sizes, and its scratch size.
_SECTIONS: dict = {}


def output_sections(experts: int):
    """Element offsets of the int32 ``outputs`` sections (total, padded, valid) and its size.  The same arithmetic
    runs in the executor (``_Scheduler.__call__``) on the runtime values."""
    total = 0
    padded = total + _ALIGN
    valid = padded + _aligned(experts)
    return total, padded, valid, valid + _aligned(experts)


def scratch_sections(experts: int, world: int):
    """Element offsets of the int32 intermediate ``scratch`` sections (min_counts, counts, chunk_counts, column_counts)
    and its size."""
    min_counts = 0
    counts = min_counts + _aligned(experts)
    chunk_counts = counts + _aligned(experts * world)
    column_counts = chunk_counts + experts * world * CHUNKS
    return min_counts, counts, chunk_counts, column_counts, column_counts + experts * world * _COLUMNS_TOTAL


class _Scheduler:
    def __init__(self):
        self.count = _COUNT_KERNEL
        self.pad = _PAD_KERNEL
        self.schedule = _SCHEDULE_KERNEL

    @cute.jit
    def __call__(
        self,
        topk: cute.Tensor,
        peer_rank: cute.Tensor,
        peer_token: cute.Tensor,
        outputs: cute.Tensor,
        scratch: cute.Tensor,
        num_tokens: Int32,
        world: Int32,
        top_k: Int32,
        experts: Int32,
        capacity: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        # Sections (see output_sections / scratch_sections): outputs = [total | padded | valid],
        # scratch = [min_counts | counts | chunk_counts | column_counts]; every section starts 16-byte aligned.
        aligned_experts = (Int64(experts) + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)
        bins = Int64(experts) * Int64(world)
        counts_offset = aligned_experts
        chunk_offset = counts_offset + (bins + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)
        column_offset = chunk_offset + bins * Int64(CHUNKS)
        total = cute.make_tensor(outputs.iterator, cute.make_layout(Int64(1)))
        padded = cute.make_tensor(outputs.iterator + Int64(_ALIGN), cute.make_layout(Int64(experts)))
        valid = cute.make_tensor(outputs.iterator + Int64(_ALIGN) + aligned_experts, cute.make_layout(Int64(experts)))
        min_counts = cute.make_tensor(scratch.iterator, cute.make_layout(Int64(experts)))
        counts = cute.make_tensor(scratch.iterator + counts_offset, cute.make_layout(bins))
        chunk_counts = cute.make_tensor(scratch.iterator + chunk_offset, cute.make_layout(bins * Int64(CHUNKS)))
        column_counts = cute.make_tensor(scratch.iterator + column_offset, cute.make_layout(bins * Int64(_COLUMNS_TOTAL)))
        self.count(topk, chunk_counts, column_counts, peer_rank, num_tokens, world, top_k, experts, capacity, rank, stream)
        self.pad(chunk_counts, counts, min_counts, padded, valid, experts, world, stream)
        self.schedule(
            topk,
            counts,
            min_counts,
            chunk_counts,
            column_counts,
            padded,
            total,
            peer_rank,
            peer_token,
            num_tokens,
            world,
            top_k,
            experts,
            capacity,
            rank,
            stream,
        )


_KERNEL = _Scheduler()
_EXECUTOR_CACHE = {}
_EXECUTOR_CACHE_DEVICES = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()
_VALIDATED_CALLS = ValidatedCalls()


def _validate_inputs(topk_all, num_local_experts, schedule_capacity, rank):
    import torch

    if type(num_local_experts) is not int or not 0 < num_local_experts < 2**31:
        raise ValueError("num_local_experts must be a positive int32")
    if not isinstance(topk_all, torch.Tensor) or not topk_all.is_cuda:
        raise ValueError("topk_all must be a CUDA tensor")
    shape, dtype, device = topk_all.shape, topk_all.dtype, topk_all.device
    if dtype != torch.int32 or not topk_all.is_contiguous():
        raise ValueError("topk_all must be contiguous int32")
    if len(shape) != 3 or any(int(d) <= 0 for d in shape):
        raise ValueError("topk_all must have positive shape [EP,T,topk]")
    world, num_tokens, top_k = map(int, shape)
    if type(rank) is not int or not 0 <= rank < world:
        raise ValueError("rank must be an integer in [0,EP)")
    if world * num_tokens * top_k >= 2**31 or scratch_sections(num_local_experts, world)[-1] >= 2**31:
        raise ValueError(
            f"routing elements EP*T*topk = {world * num_tokens * top_k} and E*EP*1057 = {num_local_experts * world * 1057} " "must fit positive int32"
        )
    if topk_all.data_ptr() % 4:
        raise ValueError("int32 input must be 4-byte aligned")
    if type(schedule_capacity) is not int or not 0 <= schedule_capacity < 2**31:
        raise ValueError("schedule_capacity must be a nonnegative int32")
    budget = _device_properties(device).shared_memory_per_block_optin
    if 4 * num_local_experts * 32 > budget or _SCHEDULE_SMEM_BYTES > budget:
        raise ValueError(
            f"scheduler shared memory: the count histogram needs {4 * num_local_experts * 32} B (E <= {budget // 128} "
            f"local experts) and the schedule block {_SCHEDULE_SMEM_BYTES} B; the device allows {budget} B per block"
        )
    if schedule_shared_memory_bytes(num_local_experts, world) > _SCHEDULE_SMEM_BYTES:
        raise ValueError(
            f"scheduler shared memory: the schedule kernel's tables need {schedule_shared_memory_bytes(num_local_experts, world)} B "
            f"for E = {num_local_experts}, EP = {world} (256 * EP + 4 * E <= {_SCHEDULE_SMEM_BYTES - _SCHEDULE_FIXED_BYTES} B) "
            f"of its {_SCHEDULE_SMEM_BYTES} B block"
        )
    current_device = _current_device_index()
    device_index = device.index if device.index is not None else current_device
    if current_device != device_index:
        raise RuntimeError("the input CUDA device must be current")
    return device_index, num_tokens, world, top_k


def scheduler(topk_all, num_local_experts, schedule_capacity, rank, out=None):
    """Return rank/token/total/padded/valid; token padding and tail are undefined.

    peer_rank is -1 on every unrouted entry.  ``valid`` (int32 [E]) is each local expert's unpadded routed
    row count, which the BF16 forward needs (its routed GEMMs run with the valid rows of each 256-row block
    as the instruction's N).  total / padded / valid are 16-byte aligned views of one small ``outputs``
    allocation (they share its storage); the intermediate counts live in a separate scratch that is released
    when this call returns.

    ``out`` (the execute step of a plan): ``(peer_rank, peer_token, outputs, scratch)``, the caller's int32
    buffers of ``schedule_buffer_lengths`` (``scratch`` may be longer), 16-byte aligned; nothing is allocated
    and the returned tensors are ``peer_rank``, ``peer_token`` and views of ``outputs``.

    Capacity must also hold the actual padded routing total; the schedule device kernel traps if that
    data-dependent bound fails.  Nonnegative counts and int32-safe rounded sums remain device-data
    preconditions.  One host path, as every CuTe DSL op: the full input check and executor key only on a new
    call signature; torch tensors and plain scalars go straight to the compiled TVM FFI function.
    """
    import torch

    try:
        signature = (
            tensor_layout_signature((topk_all,)),
            typed_key(num_local_experts, schedule_capacity, rank),
            _current_device_index(),
            _current_cuda_context_key(),
        )
        validated = _VALIDATED_CALLS.get(signature)  # hashes the signature
    except (AttributeError, TypeError):  # not a tensor / unhashable: the full check reports it
        signature = validated = None
    if validated is None:
        device_index, num_tokens, world, top_k = _validate_inputs(topk_all, num_local_experts, schedule_capacity, rank)
        key = (_CODEGEN_EPOCH, device_index, _current_cuda_context_key(), _COMPILE_OPTIONS)
        validated = (device_index, num_tokens, world, top_k, key)
        if signature is not None:
            _VALIDATED_CALLS.put(signature, validated)
    elif topk_all.data_ptr() % 4:
        # The signature holds no address: the alignment check is repeated on every call.
        raise ValueError("int32 input must be 4-byte aligned")
    device_index, num_tokens, world, top_k, key = validated
    device = topk_all.device
    sections = _SECTIONS.get((num_local_experts, world))
    if sections is None:
        _, padded_offset, valid_offset, output_size = output_sections(num_local_experts)
        sections = _SECTIONS[(num_local_experts, world)] = (padded_offset, valid_offset, output_size, scratch_sections(num_local_experts, world)[-1])
    padded_offset, valid_offset, output_size, scratch_size = sections
    if out is None:
        # Separate allocations: peer_rank and peer_token_idx never share a buffer.
        peer_rank = torch.empty((schedule_capacity,), device=device, dtype=torch.int32)
        peer_token = torch.empty((schedule_capacity,), device=device, dtype=torch.int32)
        outputs = torch.empty((output_size,), device=device, dtype=torch.int32)
        scratch = torch.empty((scratch_size,), device=device, dtype=torch.int32)
    else:
        peer_rank, peer_token, outputs, scratch = out
        for name, tensor, length in (
            ("peer_rank", peer_rank, schedule_capacity),
            ("peer_token", peer_token, schedule_capacity),
            ("outputs", outputs, output_size),
            ("scratch", scratch, scratch_size),
        ):
            if (
                type(tensor) is not torch.Tensor
                or tensor.dtype != torch.int32
                or tensor.device != device
                or tensor.ndim != 1
                or tensor.data_ptr() % 16
                or not tensor.is_contiguous()
                or (tensor.numel() < length if name == "scratch" else tensor.numel() != length)
            ):
                raise ValueError(f"out.{name} must be a 16-byte aligned int32 [{length}] tensor on {device}")
    topk_all = _row_major_view(topk_all)
    handle = current_stream_handle(device_index)
    executor = _EXECUTOR_CACHE.get(key)
    if executor is None:
        executor = _compile_executor(
            topk_all, peer_rank, peer_token, outputs, scratch, num_tokens, world, top_k, num_local_experts, schedule_capacity, rank, key, handle
        )
    executor(topk_all, peer_rank, peer_token, outputs, scratch, num_tokens, world, top_k, num_local_experts, schedule_capacity, rank, handle)
    # No record_stream: like PyTorch's own ops (see fwd_epilogue_bf16).
    # as_strided rather than slicing: the same views at a lower host cost.
    return (
        peer_rank,
        peer_token,
        outputs.as_strided((1,), (1,), 0),
        outputs.as_strided((num_local_experts,), (1,), padded_offset),
        outputs.as_strided((num_local_experts,), (1,), valid_offset),
    )


def schedule_buffer_lengths(num_local_experts: int, world: int, schedule_capacity: int):
    """Lengths of the int32 buffers ``scheduler(out=...)`` takes: peer_rank, peer_token, outputs, scratch."""
    return (schedule_capacity, schedule_capacity, output_sections(num_local_experts)[-1], scratch_sections(num_local_experts, world)[-1])


def compile_scheduler(topk_all, scratch) -> None:
    """The plan step: compile the three scheduler kernels without launching.  The arguments only give the
    tensors' ranks and dtypes: the gathered routes ``[EP, T, topk]`` and any int32 vector for the other four."""
    device = topk_all.device
    index = device.index if device.index is not None else _current_device_index()
    key = (_CODEGEN_EPOCH, index, _current_cuda_context_key(), _COMPILE_OPTIONS)
    _compile_executor(
        _row_major_view(topk_all),
        scratch,
        scratch,
        scratch,
        scratch,
        int(topk_all.shape[1]),
        int(topk_all.shape[0]),
        int(topk_all.shape[2]),
        1,
        256,
        0,
        key,
        current_stream_handle(index),
    )


def _compile_executor(topk_all, peer_rank, peer_token, outputs, scratch, num_tokens, world, top_k, experts, capacity, rank, key, handle):
    import torch

    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(key)
        if executor is None:
            device = topk_all.device
            check_device(device, "CuTe DSL scheduler")
            arguments = (
                from_dlpack(topk_all, assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=2),
                *(
                    from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=0)
                    for tensor in (peer_rank, peer_token, outputs, scratch)
                ),
                Int32(num_tokens),
                Int32(world),
                Int32(top_k),
                Int32(experts),
                Int32(capacity),
                Int32(rank),
                handle,
            )
            executor = persistent_executor(("scheduler", key[0], *key[3:]), lambda: cute.compile(_KERNEL, *arguments, options=_COMPILE_OPTIONS))
            _EXECUTOR_CACHE[key] = executor
            _EXECUTOR_CACHE_DEVICES.add(key[1])
    return executor


def clear_scheduler_executor_cache(*, synchronize=True):
    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
        _VALIDATED_CALLS.clear()
