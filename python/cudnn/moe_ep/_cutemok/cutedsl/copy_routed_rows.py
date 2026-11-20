# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Stages a rank's inputs in its symmetric buffers, copying only the routed rows.

Copies the rows of each source that have at least one routed slot (top_experts >= 0) into the
matching buffer and leaves the other rows untouched (peers read only routed rows, through the
buffer pointers; the shared expert reads the caller's tensors), and copies the full router weights.
The backward's variant also writes zero into the unused slots (top_experts < 0) of its router-weight
gradient buffer: the peers write only used slots of that buffer in the call that follows.

One warp per (token, 1024-column segment): the warp votes whether the token has a route, lane
``l`` moves the 16-byte vectors at columns ``segment * 1024 + v * 256 + 8 * l`` (v = 0..3) of each
of the one or two source / destination pairs, and the token's first segment also copies its
``topk`` router weights (and zeroes the unused gradient slots).  Token count, hidden size and top-k
are runtime values; the number of pairs and the presence of router weights and of the gradient buffer
are compile time.  Sources and destinations are 16-byte
aligned contiguous BF16 ``[T, H]`` with ``H % 8 == 0``.
"""

import threading

from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, TARGET_ARCH, TARGET_CAPABILITY

THREADS = 128
WARPS = THREADS // 32
VEC = 8  # BF16 elements per 16-byte access
VECS_PER_LANE = 4
SEGMENT = 32 * VEC * VECS_PER_LANE  # 1024 columns per warp task
MAX_PAIRS = 2
_CODEGEN_EPOCH = "copy-routed-rows-bf16-v2"

from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import from_dlpack

from ._host_common import current_cuda_context_key as _current_cuda_context_key
from ._host_common import current_stream_handle, persistent_executor
from ._host_common import current_device_index as _current_device_index, device_capability as _device_capability


@cute.jit
def _warp_sum(value: Int32) -> Int32:
    """Sum over the 32 lanes (inclusive shuffle-up scan, then lane 31's total)."""
    lane = cute.arch.lane_idx()
    for shift in cutlass.range_constexpr(5):
        offset = Int32(1 << shift)
        previous = cute.arch.shuffle_sync_up(value, offset, mask=0xFFFFFFFF, mask_and_clamp=0)
        if lane >= offset:
            value = value + previous
    return cute.arch.shuffle_sync(value, 31)


@cute.jit
def _vector(tensor: cute.Tensor, token: Int32, hidden: Int32, column: Int32):
    """The 16-byte vector at (token, column) of a contiguous BF16 [T, H] tensor: the host checked a 16-byte
    aligned base and H % 8 == 0, so every such vector is 16-byte aligned and moves as one access."""
    pointer = (tensor.iterator + (Int64(token) * Int64(hidden) + Int64(column))).align(min_align=16)
    return cute.make_tensor(pointer, cute.make_layout(VEC))


class _CopyRoutedRows:
    def __init__(self, num_pairs: int, has_weights: bool, zeroes_unrouted: bool = False):
        self.num_pairs = num_pairs
        self.has_weights = has_weights
        self.zeroes_unrouted = zeroes_unrouted

    @cute.jit
    def __call__(
        self,
        top_experts: cute.Tensor,
        src0: cute.Tensor,
        dst0: cute.Tensor,
        src1: cute.Tensor,
        dst1: cute.Tensor,
        weights: cute.Tensor,
        weights_dst: cute.Tensor,
        unrouted_dst: cute.Tensor,
        num_tokens: Int32,
        hidden: Int32,
        topk: Int32,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(top_experts, src0, dst0, src1, dst1, weights, weights_dst, unrouted_dst, num_tokens, hidden, topk).launch(
            grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        top_experts: cute.Tensor,
        src0: cute.Tensor,
        dst0: cute.Tensor,
        src1: cute.Tensor,
        dst1: cute.Tensor,
        weights: cute.Tensor,
        weights_dst: cute.Tensor,
        unrouted_dst: cute.Tensor,
        num_tokens: Int32,
        hidden: Int32,
        topk: Int32,
    ):
        thread = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        lane = thread % Int32(32)
        warp = thread // Int32(32)
        segments = (hidden + Int32(SEGMENT - 1)) // Int32(SEGMENT)
        task = block * Int32(WARPS) + warp
        token = task // segments
        if token < num_tokens:
            col = (task - token * segments) * Int32(SEGMENT) + lane * Int32(VEC)
            # The token's routes, 32 slots per vote.
            routed = Int32(0)
            first_slot = Int32(0)
            while first_slot < topk:
                slot = first_slot + lane
                flag = Int32(0)
                if slot < topk:
                    if top_experts[token, slot] >= Int32(0):
                        flag = Int32(1)
                routed = routed + _warp_sum(flag)
                first_slot = first_slot + Int32(32)
            if cutlass.const_expr(self.has_weights):
                if col < Int32(32 * VEC):  # the token's first segment
                    slot = lane
                    while slot < topk:
                        weights_dst[token, slot] = weights[token, slot]
                        slot = slot + Int32(32)
            if cutlass.const_expr(self.zeroes_unrouted):
                if col < Int32(32 * VEC):  # the token's first segment: its unused gradient slots
                    slot = lane
                    while slot < topk:
                        if top_experts[token, slot] < Int32(0):
                            unrouted_dst[token, slot] = Float32(0.0)
                        slot = slot + Int32(32)
            if routed != Int32(0):
                for v in cutlass.range_constexpr(VECS_PER_LANE):
                    column = col + Int32(v * 32 * VEC)
                    if column < hidden:
                        r0 = cute.make_rmem_tensor((VEC,), BFloat16)
                        cute.autovec_copy(_vector(src0, token, hidden, column), r0)
                        if cutlass.const_expr(self.num_pairs == 2):
                            r1 = cute.make_rmem_tensor((VEC,), BFloat16)
                            cute.autovec_copy(_vector(src1, token, hidden, column), r1)
                            cute.autovec_copy(r1, _vector(dst1, token, hidden, column))
                        cute.autovec_copy(r0, _vector(dst0, token, hidden, column))


_EXECUTORS: dict = {}
_LOCK = threading.Lock()


def _check(t, name, shape, dtype, device):
    if not isinstance(t, torch.Tensor) or not t.is_cuda or t.device != device or t.dtype != dtype or not t.is_contiguous() or tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} must be contiguous CUDA {dtype} {list(shape)} on the input device")


#: (destination ids, weight-destination id, device) -> (destination refs, their addresses, num_tokens, hidden,
#: topk, executor): the workspace buffers a full check accepted, so a repeated call checks only its sources.
_FAST: dict = {}


def _fresh(tensor, dtype, shape, device) -> bool:
    return type(tensor) is torch.Tensor and tensor.dtype == dtype and tensor.device == device and tensor.shape == shape and tensor.is_contiguous()


def _executor(operands, num_pairs, has_weights, zeroes_unrouted, num_tokens, hidden, topk, index, stream):
    """The compiled variant (one or two pairs, with or without the router weights, with or without the
    gradient buffer whose unused slots it zeroes), compiled on first use."""
    key = (_CODEGEN_EPOCH, num_pairs, has_weights, zeroes_unrouted, index, _current_cuda_context_key(), _COMPILE_OPTIONS)
    executor = _EXECUTORS.get(key)
    if executor is None:
        with _LOCK:
            executor = _EXECUTORS.get(key)
            if executor is None:
                aligns = (4, 16, 16, 16, 16, 4, 4, 4)
                arguments = tuple(from_dlpack(t, assumed_align=a, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1) for t, a in zip(operands, aligns))
                kernel = _CopyRoutedRows(num_pairs, has_weights, zeroes_unrouted)
                executor = persistent_executor(
                    ("copy_routed_rows", _CODEGEN_EPOCH, num_pairs, has_weights, zeroes_unrouted, _COMPILE_OPTIONS),
                    lambda: cute.compile(kernel, *arguments, Int32(num_tokens), Int32(hidden), Int32(topk), Int32(1), stream, options=_COMPILE_OPTIONS),
                )
                _EXECUTORS[key] = executor
    return executor


def compile_copy_routed_rows(top_experts, destination, weights_destination, num_pairs: int = 1) -> None:
    """The plan step: compile the variant with router weights and one pair (the forward's staging copy) or two
    with the gradient slots zeroed (the backward's) without launching.  The arguments only give the kernel's
    tensor ranks and dtypes -- int32 [T, topk], BF16 [T, H], FP32 [T, topk] -- so the workspace buffers stand in
    for the caller's tensors."""
    index = destination.device.index
    index = _current_device_index() if index is None else index
    operands = (top_experts, destination, destination, destination, destination, weights_destination, weights_destination, weights_destination)
    _executor(
        operands,
        num_pairs,
        True,
        num_pairs == 2,
        int(top_experts.shape[0]),
        int(destination.shape[1]),
        int(top_experts.shape[1]),
        index,
        current_stream_handle(index),
    )


def copy_routed_rows(top_experts, sources, destinations, weights=None, weights_destination=None, zero_unrouted=None) -> None:
    """Copy one or two source -> destination BF16 [T, H] pairs, rows with at least one route
    (``top_experts[t, k] >= 0``) only; and, when given, the full FP32 [T, topk] router weights.
    ``zero_unrouted`` (FP32 [T, topk], optional): its unused slots (``top_experts[t, k] < 0``) are set to
    zero and its other slots are left untouched."""
    fast = None
    if type(top_experts) is torch.Tensor:
        fast = _FAST.get((tuple(map(id, destinations)), id(weights_destination), id(zero_unrouted), top_experts.device))
    if fast is not None:
        refs, pointers, num_tokens, hidden, topk, executor, index = fast
        device = top_experts.device
        owned = (*destinations, weights_destination, zero_unrouted)
        if (
            len(owned) == len(refs)
            and all(ref() is buffer for ref, buffer in zip(refs, owned))
            and all(buffer is None or buffer.data_ptr() == pointer for buffer, pointer in zip(owned, pointers))
            and len(sources) == len(destinations)
            and _fresh(top_experts, torch.int32, (num_tokens, topk), device)
            and all(_fresh(source, torch.bfloat16, (num_tokens, hidden), device) and not source.data_ptr() % 16 for source in sources)
            and (weights is None) == (weights_destination is None)
            and (weights is None or _fresh(weights, torch.float32, (num_tokens, topk), device))
        ):
            num_pairs = len(sources)
            src1 = sources[1] if num_pairs == 2 else sources[0]
            dst1 = destinations[1] if num_pairs == 2 else destinations[0]
            w = weights if weights is not None else top_experts.new_zeros((1, 1), dtype=torch.float32)
            wd = weights_destination if weights is not None else w
            zd = zero_unrouted if zero_unrouted is not None else wd
            tasks = num_tokens * ((hidden + SEGMENT - 1) // SEGMENT)
            executor(
                top_experts,
                sources[0],
                destinations[0],
                src1,
                dst1,
                w,
                wd,
                zd,
                num_tokens,
                hidden,
                topk,
                (tasks + WARPS - 1) // WARPS,
                current_stream_handle(index),
            )
            return
    if not isinstance(top_experts, torch.Tensor) or top_experts.dim() != 2:
        raise ValueError("top_experts must be 2-D")
    sources, destinations = list(sources), list(destinations)
    if not sources or len(sources) > MAX_PAIRS or len(sources) != len(destinations):
        raise ValueError("copy_routed_rows takes one or two source/destination pairs")
    if (weights is None) != (weights_destination is None):
        raise ValueError("router weights and their destination must be given together")
    num_tokens, topk = map(int, top_experts.shape)
    hidden = int(sources[0].shape[1]) if sources[0].dim() == 2 else -1
    if hidden <= 0 or hidden % VEC:
        raise ValueError("hidden size must be a positive multiple of 8")
    device = top_experts.device
    _check(top_experts, "top_experts", (num_tokens, topk), torch.int32, device)
    for index, tensor in enumerate((*sources, *destinations)):
        _check(tensor, "source" if index < len(sources) else "destination", (num_tokens, hidden), torch.bfloat16, device)
        if tensor.data_ptr() % 16:
            raise ValueError("sources and destinations must be 16-byte aligned")
    if weights is not None:
        _check(weights, "router_weights", (num_tokens, topk), torch.float32, device)
        _check(weights_destination, "router_weight_buffer", (num_tokens, topk), torch.float32, device)
    if zero_unrouted is not None:
        _check(zero_unrouted, "zero_unrouted", (num_tokens, topk), torch.float32, device)
    tasks = num_tokens * ((hidden + SEGMENT - 1) // SEGMENT)
    if tasks == 0:
        return
    if _device_capability(device) != TARGET_CAPABILITY:
        raise NotImplementedError(f"copy_routed_rows requires {TARGET_ARCH}")
    num_pairs, has_weights = len(sources), weights is not None
    src1 = sources[1] if num_pairs == 2 else sources[0]
    dst1 = destinations[1] if num_pairs == 2 else destinations[0]
    w = weights if has_weights else top_experts.new_zeros((1, 1), dtype=torch.float32)
    wd = weights_destination if has_weights else w
    zeroes_unrouted = zero_unrouted is not None
    zd = zero_unrouted if zeroes_unrouted else wd
    index = device.index if device.index is not None else _current_device_index()
    stream = current_stream_handle(index)
    operands = (top_experts, sources[0], destinations[0], src1, dst1, w, wd, zd)
    executor = _executor(operands, num_pairs, has_weights, zeroes_unrouted, num_tokens, hidden, topk, index, stream)
    blocks = (tasks + WARPS - 1) // WARPS
    executor(*operands, num_tokens, hidden, topk, blocks, stream)
    import weakref

    if len(_FAST) >= 64:
        _FAST.clear()
    owned = (*destinations, weights_destination, zero_unrouted)
    _FAST[(tuple(map(id, destinations)), id(weights_destination), id(zero_unrouted), device)] = (
        tuple((lambda: None) if buffer is None else weakref.ref(buffer) for buffer in owned),
        tuple(None if buffer is None else buffer.data_ptr() for buffer in owned),
        num_tokens,
        hidden,
        topk,
        executor,
        index,
    )


def clear_copy_routed_rows_executor_cache() -> None:
    with _LOCK:
        _EXECUTORS.clear()
        _FAST.clear()
