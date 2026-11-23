# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Standalone, in-place zeroing of the routed BF16 weight gradients of selected experts (SM100 / SM103).

``zero_empty_routed_wgrads`` zeroes ``gate`` / ``up`` ``[E, I, H]`` and ``down`` ``[E, H, I]`` for every
expert without rows and, with ``zero_split_experts``, for every expert whose rows cross a macrobatch
boundary; every other element is untouched.  The grid has one row of CTAs per expert, striding over the
expert's elements.  The sizes, the macrobatch size and the split switch are runtime values.

The caller guarantees nonnegative int32 counts whose prefix sum fits int32.  ``cudnn.moe_ep._cutemok.functional`` does not
call this op.
"""

import threading


from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, check_device

GRID_X = 128
THREADS = 256
_CODEGEN_EPOCH = "zero-routed-wgrads-bf16-runtime-o2-native-host-v8"

from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Boolean, Int32, Int64, Uint32
from cutlass.cute.runtime import from_dlpack
from ._host_common import persistent_executor
from ._host_common import current_cuda_context_key as _current_cuda_context_key
from ._host_common import (
    ValidatedCalls,
    current_stream_handle,
    row_major_view as _row_major_view,
    tensor_layout_signature,
    typed_key,
)
from ._host_common import current_device_index as _current_device_index


@cute.struct
class _SharedStorage:
    zero_this_expert: Int32


class _ZeroRoutedWgrads:
    @cute.jit
    def __call__(
        self,
        gate: cute.Tensor,
        up: cute.Tensor,
        down: cute.Tensor,
        counts: cute.Tensor,
        experts: Int32,
        elements: Int64,
        macro: Int32,
        split: Boolean,
        stream: cuda.CUstream,
    ):
        flat_layout = cute.make_layout(Int64(experts) * elements)
        self.kernel(
            cute.make_tensor(gate.iterator, flat_layout),
            cute.make_tensor(up.iterator, flat_layout),
            cute.make_tensor(down.iterator, flat_layout),
            counts,
            elements,
            macro,
            split,
        ).launch(
            grid=(GRID_X, experts, 1),
            block=(THREADS, 1, 1),
            smem=_SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gate: cute.Tensor,
        up: cute.Tensor,
        down: cute.Tensor,
        counts: cute.Tensor,
        elements: Int64,
        macro: Int32,
        split: Boolean,
    ):
        thread = cute.arch.thread_idx()[0]
        block, expert, _ = cute.arch.block_idx()
        # Fixed one-flag storage, independent of E/H/I and the runtime branch.
        storage = utils.SmemAllocator().allocate(_SharedStorage)
        flag = cute.make_tensor(storage.zero_this_expert.ptr, cute.make_layout(1))
        zero_this_expert = Boolean(False)
        if split:
            if thread == Int32(0):
                rows = counts[expert]
                begin = Int32(0)
                for index in cutlass.range(expert, unroll=16):
                    begin += counts[index]
                flag[0] = Int32(0)
                if rows == Int32(0):
                    flag[0] = Int32(1)
                elif Uint32(begin) // Uint32(macro) != Uint32(begin + rows - Int32(1)) // Uint32(macro):
                    flag[0] = Int32(1)
            cute.arch.sync_threads()
            zero_this_expert = flag[0] != Int32(0)
        else:
            zero_this_expert = counts[expert] == Int32(0)
        if zero_this_expert:
            expert_offset = Int64(expert) * Int64(elements)
            start = Int64(block) * Int64(THREADS) + Int64(thread)
            base = start
            while base < Int64(elements):
                offset = expert_offset + base
                gate[offset] = BFloat16(0.0)
                up[offset] = BFloat16(0.0)
                down[offset] = BFloat16(0.0)
                base += Int64(GRID_X * THREADS)


_KERNEL = _ZeroRoutedWgrads()
_EXECUTOR_CACHE = {}
_EXECUTOR_CACHE_DEVICES = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()


def _validate_inputs(gate, up, down, counts, macro, split):
    if type(macro) is not int or not 0 < macro < 2**31:
        raise ValueError(f"macrobatch_size must be a positive int32; got {macro!r}")
    if type(split) is not bool:
        raise TypeError("zero_split_experts must be bool")
    device = None
    for tensor in (gate, up, down, counts):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise ValueError("all arguments must be CUDA tensors")
        tensor_device = tensor.device
        if device is None:
            device = tensor_device
        if tensor_device != device:
            raise ValueError("all tensors must be on the same CUDA device")
        if not tensor.is_contiguous():
            raise ValueError("all tensors must be contiguous on the same CUDA device")
    gate_shape = gate.shape
    if gate.ndim != 3:
        raise ValueError(f"gate must have shape [E, I, H]; got {list(gate_shape)}")
    experts, intermediate, hidden = map(int, gate_shape)
    elements = intermediate * hidden
    # CUDA grid.y is limited to 65535; byte extents and elements use int64.
    if not 0 < experts <= 65535:
        raise ValueError(f"E must be in [1,65535] (CUDA grid.y limit); got {experts}")
    if intermediate <= 0 or hidden <= 0 or experts * elements * 2 >= 2**63:
        raise ValueError("gradient dimensions must be positive with int64-safe byte extent; got " f"[E, I, H]={list(gate_shape)}")
    if tuple(up.shape) != tuple(gate_shape) or tuple(down.shape) != (
        experts,
        hidden,
        intermediate,
    ):
        raise ValueError("up must match gate; down must have shape [E, H, I]; got gate " f"{list(gate_shape)}, up {list(up.shape)}, down {list(down.shape)}")
    for tensor in (gate, up, down):
        if tensor.dtype != torch.bfloat16 or tensor.data_ptr() % 16:
            raise ValueError("gradients must be 16-byte-aligned BF16")
    if counts.dtype != torch.int32 or tuple(counts.shape) != (experts,):
        raise ValueError(f"counts must be int32[E]; got shape {list(counts.shape)} for E={experts}")
    if counts.data_ptr() % 4:
        raise ValueError("counts must be 4-byte aligned")
    current_device = _current_device_index()
    index = device.index if device.index is not None else current_device
    if current_device != index:
        raise RuntimeError("the gradient CUDA device must be current")
    return device, index, experts, elements, gate_shape


_VALIDATED_CALLS = ValidatedCalls()


def zero_empty_routed_wgrads(gate, up, down, counts, *, macrobatch_size, zero_split_experts=False):
    """Zero selected experts in place; all unselected elements are untouched.

    Selected: experts with a zero count and, with ``zero_split_experts``, experts whose rows
    cross a ``macrobatch_size`` boundary.  Counts must be nonnegative with an int32-safe prefix
    sum (device-data precondition, not checked with a host synchronization).  No output is
    allocated.  The full input check and the executor key run only on a new call signature;
    torch tensors and plain scalars go straight to the compiled TVM FFI function.
    """
    try:
        signature = (
            tensor_layout_signature((gate, up, down, counts)),
            typed_key(macrobatch_size, zero_split_experts),
            _current_device_index(),
            _current_cuda_context_key(),
        )
        validated = _VALIDATED_CALLS.get(signature)  # hashes the signature
    except (AttributeError, TypeError):  # not tensors / unhashable: the full check reports it
        signature = validated = None
    if validated is None:
        device, device_index, experts, elements, _ = _validate_inputs(gate, up, down, counts, macrobatch_size, zero_split_experts)
        key = (_CODEGEN_EPOCH, device_index, _current_cuda_context_key(), "bf16-rank3-contiguous-i32-counts", _COMPILE_OPTIONS)
        validated = (device, device_index, experts, elements, key)
        if signature is not None:
            _VALIDATED_CALLS.put(signature, validated)
    else:
        # The signature holds no address: the alignment checks are repeated on every call.
        if any(t.data_ptr() % 16 for t in (gate, up, down)):
            raise ValueError("gradients must be 16-byte-aligned BF16")
        if counts.data_ptr() % 4:
            raise ValueError("counts must be 4-byte aligned")
    device, device_index, experts, elements, key = validated
    gate, up, down, counts = map(_row_major_view, (gate, up, down, counts))
    cuda_stream = current_stream_handle(device_index)
    executor = _EXECUTOR_CACHE.get(key)
    if executor is None:
        executor = _compile_zero_executor(
            gate,
            up,
            down,
            counts,
            macrobatch_size,
            zero_split_experts,
            key,
            cuda_stream,
        )
    # Every call, including cache hits, supplies the current runtime scalars.
    executor(gate, up, down, counts, experts, elements, macrobatch_size, zero_split_experts, cuda_stream)
    # No record_stream: like PyTorch's own ops (see fwd_epilogue_bf16).


def _compile_zero_executor(gate, up, down, counts, macro, split, key, cuda_stream):
    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(key)
        if executor is None:
            device, device_index, experts, elements, _ = _validate_inputs(gate, up, down, counts, macro, split)
            check_device(device, "zero routed Wgrads")
            gate, up, down, counts = map(_row_major_view, (gate, up, down, counts))
            arguments = (
                from_dlpack(gate, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=2),
                from_dlpack(up, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=2),
                from_dlpack(down, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=2),
                from_dlpack(counts, assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=0),
                Int32(experts),
                Int64(elements),
                Int32(macro),
                Boolean(split),
                cuda_stream,
            )
            executor = persistent_executor(("zero_empty_routed_wgrads", key[0], *key[3:]), lambda: cute.compile(_KERNEL, *arguments, options=_COMPILE_OPTIONS))
            _EXECUTOR_CACHE[key] = executor
            _EXECUTOR_CACHE_DEVICES.add(device_index)
    return executor


def clear_zero_empty_routed_wgrads_executor_cache(*, synchronize=True):
    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device)
        _EXECUTOR_CACHE.clear()
        _VALIDATED_CALLS.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
