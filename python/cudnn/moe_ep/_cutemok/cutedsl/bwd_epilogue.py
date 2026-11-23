# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""CuTe DSL BF16 backward epilogue: combines the shared and the runtime-topk routed gradients of each token.

One 256-thread CTA owns one ``(token, 1024-column)`` tile. Eight warps split
runtime-topk public-TMA copies into disjoint dynamic shared-memory vectors.
Elements accumulate in strict shared, routed 0 through topk-1 order in FP32
before one BF16 conversion and TMA store. TMA masks runtime H tails.
The copies are issued through the warp-collective TMA APIs.
"""

import threading


from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, check_device

TILE_COLUMNS = 1024
THREADS_PER_CTA = 256
WARPS_PER_CTA = THREADS_PER_CTA // 32
BF16_BYTES = 2
TILE_BYTES = TILE_COLUMNS * BF16_BYTES
TMA_ALIGNMENT_BYTES = 1024
CLUSTER_SHAPE = (1, 1, 1)

_CODEGEN_EPOCH = "bwd-epilogue-bf16-runtime-v6-o2-masked-sum"


def _block_coordinates_cpu(block_index: int, hidden_size: int) -> tuple[int, int]:
    """Reference mapping used by the CPU-only topology test."""

    return divmod(block_index, (hidden_size + TILE_COLUMNS - 1) // TILE_COLUMNS)


def _routed_byte_offset_cpu(
    token_index: int,
    route_index: int,
    column_block: int,
    hidden_size: int,
    topk: int,
) -> int:
    """Reference byte offset, including the target-scale 64-bit case."""

    routed_row = token_index * topk + route_index
    return routed_row * hidden_size * BF16_BYTES + column_block * TILE_BYTES


def _shared_memory_bytes(topk):
    # [0,1024): barrier; then topk+1 aligned 2048B vector slots.
    return TMA_ALIGNMENT_BYTES + (topk + 1) * TILE_BYTES + topk * 4


from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from ._host_common import persistent_executor
from ._host_common import current_cuda_context_key as _current_cuda_context_key
from ._host_common import ValidatedCalls, current_stream_handle, tensor_layout_signature
from ._host_common import current_device_index as _current_device_index, device_properties as _device_properties


@cute.jit
def _issue_tma_load(
    atom: cute.CopyAtom,
    tensor: cute.Tensor,
    row_tile: Int32,
    column_tile: Int32,
    smem_tile: cute.Tensor,
    input_arrived: cute.Pointer,
) -> None:
    global_tile = cute.local_tile(
        tensor,
        (1, TILE_COLUMNS),
        (row_tile, column_tile),
    )
    smem_flat = cute.group_modes(smem_tile, 0, cute.rank(smem_tile))
    global_flat = cute.group_modes(global_tile, 0, cute.rank(global_tile))
    smem_part, global_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        smem_flat,
        global_flat,
    )
    cute.copy(
        atom,
        global_part,
        smem_part,
        tma_bar_ptr=input_arrived,
    )


@cute.jit
def _issue_routed_tma_load(
    atom: cute.CopyAtom,
    tensor: cute.Tensor,
    token_index: Int32,
    column_tile: Int32,
    arena: cute.Pointer,
    smem_layout: cute.Layout,
    input_arrived: cute.Pointer,
    route_index: Int32,
    topk: Int32,
) -> None:
    routed_smem_tile = cute.make_tensor(
        cute.recast_ptr(
            (arena + Int64(route_index + 1) * Int64(TILE_BYTES)).align(min_align=TMA_ALIGNMENT_BYTES),
            dtype=BFloat16,
        ),
        smem_layout,
    )
    # TMA row coordinates are signed32; byte strides remain signed64.
    routed_row = Int32(Int64(token_index) * Int64(topk) + Int64(route_index))
    _issue_tma_load(
        atom,
        tensor,
        routed_row,
        column_tile,
        routed_smem_tile,
        input_arrived,
    )


@cute.jit
def _issue_tma_store(
    atom: cute.CopyAtom,
    tensor: cute.Tensor,
    row_tile: Int32,
    column_tile: Int32,
    smem_tile: cute.Tensor,
) -> None:
    global_tile = cute.local_tile(
        tensor,
        (1, TILE_COLUMNS),
        (row_tile, column_tile),
    )
    smem_flat = cute.group_modes(smem_tile, 0, cute.rank(smem_tile))
    global_flat = cute.group_modes(global_tile, 0, cute.rank(global_tile))
    smem_part, global_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        smem_flat,
        global_flat,
    )
    cute.copy(atom, smem_part, global_part)


@cute.jit
def _aligned_activation(tensor: cute.Tensor) -> cute.Tensor:
    # H%256 and contiguous BF16 inputs guarantee 16B row alignment.
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(tensor.shape, stride=(cute.assume(tensor.stride[0], divby=8), 1)),
    )


class _BwdEpilogue:
    """``has_shared=False``: a routed-only layer: the accumulator starts at zero and only the
    routed rows are loaded; vector 0 still stages the output tile and the shared argument is a
    placeholder never read."""

    def __init__(self, has_shared: bool = True) -> None:
        self.has_shared = bool(has_shared)

    @cute.jit
    def __call__(
        self,
        d_x_shared: cute.Tensor,
        d_x_routed_buffer: cute.Tensor,
        top_experts: cute.Tensor,
        d_x: cute.Tensor,
        num_tokens: Int32,
        hidden_size: Int32,
        topk: Int32,
        stream: cuda.CUstream,
    ) -> None:
        d_x_shared = _aligned_activation(d_x_shared)
        d_x_routed_buffer = _aligned_activation(d_x_routed_buffer)
        d_x = _aligned_activation(d_x)
        column_blocks = (hidden_size + TILE_COLUMNS - 1) // TILE_COLUMNS
        smem_layout = cute.make_layout(
            (1, TILE_COLUMNS),
            stride=(TILE_COLUMNS, 1),
        )
        tma_tiler = (1, TILE_COLUMNS)
        load_op = cpasync.CopyBulkTensorTileG2SOp()
        store_op = cpasync.CopyBulkTensorTileS2GOp()
        shared_load_atom, shared_tma = cpasync.make_tiled_tma_atom(
            load_op,
            d_x_shared,
            smem_layout,
            tma_tiler,
        )
        routed_load_atom, routed_tma = cpasync.make_tiled_tma_atom(
            load_op,
            d_x_routed_buffer,
            smem_layout,
            tma_tiler,
        )
        output_store_atom, output_tma = cpasync.make_tiled_tma_atom(
            store_op,
            d_x,
            smem_layout,
            tma_tiler,
        )
        self.kernel(
            shared_load_atom,
            shared_tma,
            routed_load_atom,
            routed_tma,
            output_store_atom,
            output_tma,
            top_experts,
            smem_layout,
            hidden_size,
            topk,
        ).launch(
            grid=(num_tokens * column_blocks, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=TMA_ALIGNMENT_BYTES + (topk + 1) * TILE_BYTES + topk * 4,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        shared_load_atom: cute.CopyAtom,
        d_x_shared: cute.Tensor,
        routed_load_atom: cute.CopyAtom,
        d_x_routed_buffer: cute.Tensor,
        output_store_atom: cute.CopyAtom,
        d_x: cute.Tensor,
        top_experts: cute.Tensor,
        smem_layout: cute.Layout,
        hidden_size: Int32,
        topk: Int32,
    ) -> None:
        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block = cute.arch.make_warp_uniform(cute.arch.block_idx()[0])
        column_blocks = (hidden_size + Int32(TILE_COLUMNS - 1)) // Int32(TILE_COLUMNS)
        token_index = block // column_blocks
        column_block = block - token_index * column_blocks

        # The barrier lives in a reserved prefix of the dynamic shared memory.
        base = cute.arch.get_dyn_smem(Uint8, alignment=TMA_ALIGNMENT_BYTES)
        input_arrived = cute.recast_ptr(base, dtype=Int64)
        arena = base + Int64(TMA_ALIGNMENT_BYTES)
        num_input_vectors = topk + Int32(1)
        token_vectors = cute.make_tensor(
            cute.recast_ptr(arena, dtype=BFloat16),
            cute.make_layout(
                (num_input_vectors, TILE_COLUMNS),
                stride=(TILE_COLUMNS, 1),
            ),
        )
        shared_smem_tile = cute.make_tensor(
            cute.recast_ptr(arena, dtype=BFloat16),
            smem_layout,
        )
        routes = cute.make_tensor(
            cute.recast_ptr(arena + Int64(num_input_vectors) * Int64(TILE_BYTES), dtype=Int32),
            cute.make_layout((topk,), stride=(1,)),
        )

        if thread == Int32(0):
            # Every route slot's row is loaded, used or not (see the routes below), so the bytes the
            # barrier expects are known without reading the routes: the loads are issued at once.
            cute.arch.mbarrier_init(input_arrived, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(
                input_arrived,
                (topk + Int32(int(self.has_shared))) * Int32(TILE_BYTES),
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # Warp-uniform contiguous partition; every vector has one issuer.
        copies_per_warp = num_input_vectors // Int32(WARPS_PER_CTA)
        extra = num_input_vectors % Int32(WARPS_PER_CTA)
        begin = warp * copies_per_warp + cutlass.min(warp, extra)
        end = begin + copies_per_warp + Int32(warp < extra)
        for vector_index in cutlass.range(begin, end):
            if vector_index == Int32(0):
                if cutlass.const_expr(self.has_shared):
                    _issue_tma_load(
                        shared_load_atom,
                        d_x_shared,
                        token_index,
                        column_block,
                        shared_smem_tile,
                        input_arrived,
                    )
            else:
                _issue_routed_tma_load(
                    routed_load_atom,
                    d_x_routed_buffer,
                    token_index,
                    column_block,
                    arena,
                    smem_layout,
                    input_arrived,
                    vector_index - Int32(1),
                    topk,
                )

        # A token may route to fewer than topk experts; the schedule marks the empty slots -1 and the
        # backward never writes their d_x_routed rows, so those hold whatever the symmetric buffer held
        # before (it is allocated, not zeroed).  There is no weight to zero here, so the sum skips the
        # slot.  The routes are read here, while the loads are in flight, instead of ahead of them: an
        # empty slot's row is loaded like any other (the memory is there) and left out of the sum.
        for route_index in cutlass.range(thread, topk, THREADS_PER_CTA):
            routes[route_index] = top_experts[Int64(token_index), Int64(route_index)]
        cute.arch.mbarrier_wait(input_arrived, 0)
        cute.arch.sync_threads()  # the routes, written by the first topk threads, are read by all
        # Four columns per thread: the 1024-column tile over 256 threads.
        column0 = thread
        column1 = thread + Int32(THREADS_PER_CTA)
        column2 = thread + Int32(2 * THREADS_PER_CTA)
        column3 = thread + Int32(3 * THREADS_PER_CTA)
        accumulator0 = Float32(0.0)
        accumulator1 = Float32(0.0)
        accumulator2 = Float32(0.0)
        accumulator3 = Float32(0.0)
        if cutlass.const_expr(self.has_shared):
            accumulator0 = Float32(token_vectors[0, column0])
            accumulator1 = Float32(token_vectors[0, column1])
            accumulator2 = Float32(token_vectors[0, column2])
            accumulator3 = Float32(token_vectors[0, column3])
        for route_index in cutlass.range(topk):
            if routes[route_index] >= Int32(0):
                accumulator0 = accumulator0 + Float32(token_vectors[route_index + 1, column0])
                accumulator1 = accumulator1 + Float32(token_vectors[route_index + 1, column1])
                accumulator2 = accumulator2 + Float32(token_vectors[route_index + 1, column2])
                accumulator3 = accumulator3 + Float32(token_vectors[route_index + 1, column3])
        token_vectors[0, column0] = accumulator0.to(BFloat16)
        token_vectors[0, column1] = accumulator1.to(BFloat16)
        token_vectors[0, column2] = accumulator2.to(BFloat16)
        token_vectors[0, column3] = accumulator3.to(BFloat16)

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        if warp == Int32(0):
            _issue_tma_store(
                output_store_atom,
                d_x,
                token_index,
                column_block,
                shared_smem_tile,
            )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=False)


# has_shared -> the compile-time variant.
_EPILOGUES = {True: _BwdEpilogue(), False: _BwdEpilogue(has_shared=False)}
_EXECUTOR_CACHE: dict[tuple[object, ...], object] = {}
_EXECUTOR_CACHE_DEVICES: set[int] = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()


def _validate_inputs(d_x_shared, d_x_routed_buffer, top_experts) -> tuple[object, int, int, int, int]:
    import torch

    if d_x_shared is not None and not isinstance(d_x_shared, torch.Tensor):
        raise TypeError("d_x_shared must be a torch.Tensor or None")
    if not isinstance(d_x_routed_buffer, torch.Tensor):
        raise TypeError("d_x_routed_buffer must be a torch.Tensor")
    if not isinstance(top_experts, torch.Tensor):
        raise TypeError("top_experts must be a torch.Tensor")
    if not d_x_routed_buffer.is_cuda:
        raise ValueError("d_x_routed_buffer must be a CUDA tensor")
    device = d_x_routed_buffer.device
    if d_x_routed_buffer.ndim != 2 or top_experts.ndim != 2:
        raise ValueError("d_x_routed_buffer must have shape (T * topk, H) and top_experts (T, topk)")
    # From the routed side: a routed-only layer has no d_x_shared.
    num_tokens, hidden_size = int(top_experts.shape[0]), int(d_x_routed_buffer.shape[1])
    if d_x_shared is not None:
        if not d_x_shared.is_cuda or d_x_shared.device != device:
            raise ValueError("both inputs must be on the same CUDA device")
        if d_x_shared.dtype != torch.bfloat16:
            raise NotImplementedError("CuTe DSL backward epilogue requires BF16 inputs")
        if tuple(d_x_shared.shape) != (num_tokens, hidden_size):
            raise ValueError(f"d_x_shared must have shape (T, H) = {(num_tokens, hidden_size)}; got {list(d_x_shared.shape)}")
        if not d_x_shared.is_contiguous():
            raise ValueError("d_x_shared must be contiguous")
    if hidden_size <= 0 or hidden_size % 128 != 0:
        raise ValueError(f"hidden_size must be positive and divisible by 128; got {hidden_size}")
    if num_tokens < 512 or num_tokens % 256 != 0:
        raise NotImplementedError("CuTe DSL backward epilogue requires T>=512 with T divisible by 256")
    if d_x_routed_buffer.device != device:
        raise ValueError("both inputs must be on the same CUDA device")
    if d_x_routed_buffer.dtype != torch.bfloat16:
        raise NotImplementedError("CuTe DSL backward epilogue requires BF16 inputs")
    if d_x_routed_buffer.ndim != 2 or d_x_routed_buffer.shape[1] != hidden_size or d_x_routed_buffer.shape[0] % num_tokens != 0:
        raise ValueError("d_x_routed_buffer must have shape (T * topk, H); got " f"{list(d_x_routed_buffer.shape)} for T={num_tokens}, H={hidden_size}")
    topk = int(d_x_routed_buffer.shape[0]) // num_tokens
    if not 0 < topk <= 255:
        raise ValueError(f"topk must be in [1, 255]; got {topk}")
    if num_tokens * topk >= 2**31 or hidden_size + TILE_COLUMNS - 1 >= 2**31:
        raise ValueError("epilogue dimensions exceed signed32 TMA coordinates")
    if num_tokens * ((hidden_size + TILE_COLUMNS - 1) // TILE_COLUMNS) >= 2**31:
        raise ValueError("epilogue grid exceeds signed32 CUDA grid size")
    if not d_x_routed_buffer.is_contiguous():
        raise ValueError("d_x_routed_buffer must be contiguous")
    if (d_x_shared is not None and d_x_shared.data_ptr() % 16) or d_x_routed_buffer.data_ptr() % 16:
        raise ValueError("CuTe DSL backward epilogue inputs must be 16-byte aligned")
    device_index = device.index if device.index is not None else _current_device_index()
    if _current_device_index() != device_index:
        raise RuntimeError("the input CUDA device must be current")
    if _shared_memory_bytes(topk) > _device_properties(device).shared_memory_per_block_optin:
        raise ValueError("topk requires more dynamic shared memory than the device supports")
    if (
        not top_experts.is_cuda
        or top_experts.device != device
        or top_experts.dtype != torch.int32
        or not top_experts.is_contiguous()
        or tuple(top_experts.shape) != (num_tokens, topk)
    ):
        raise ValueError(f"top_experts must be a contiguous int32 ({num_tokens}, {topk}) tensor on {device}")
    if top_experts.data_ptr() % 16:
        raise ValueError("top_experts must be 16-byte aligned")
    return device, device_index, num_tokens, hidden_size, topk


def _make_arguments(d_x_shared, d_x_routed_buffer, top_experts, d_x, num_tokens, hidden_size, topk, stream):
    # Extents/row strides stay dynamic (64-bit); only unit column stride static.
    return (
        from_dlpack(d_x_shared, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
        from_dlpack(d_x_routed_buffer, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
        from_dlpack(top_experts, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
        from_dlpack(d_x, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1),
        Int32(num_tokens),
        Int32(hidden_size),
        Int32(topk),
        cuda.CUstream(stream.cuda_stream),
    )


_VALIDATED_CALLS = ValidatedCalls()


def _compile_bwd_epilogue_executor(
    d_x_shared,
    d_x_routed_buffer,
    top_experts,
    d_x,
    num_tokens,
    hidden_size,
    topk,
    cache_key,
    has_shared,
):
    import torch

    device = d_x_routed_buffer.device
    stream = torch.cuda.current_stream(device)
    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(cache_key)
        if executor is None:
            check_device(device, "CuTe DSL backward epilogue")
            arguments = _make_arguments(
                d_x_shared,
                d_x_routed_buffer,
                top_experts,
                d_x,
                num_tokens,
                hidden_size,
                topk,
                stream,
            )
            executor = persistent_executor(
                ("bwd_epilogue", cache_key[0], *cache_key[3:]), lambda: cute.compile(_EPILOGUES[has_shared], *arguments, options=_COMPILE_OPTIONS)
            )
            _EXECUTOR_CACHE[cache_key] = executor
            _EXECUTOR_CACHE_DEVICES.add(device.index)
    return executor


def bwd_epilogue_bf16(d_x_shared, d_x_routed_buffer, top_experts, out=None):
    """Combine shared and runtime-topk routed BF16 gradients.

    ``top_experts`` marks a token's unused route slots with -1.  Their d_x_routed rows were never
    written, so they are skipped; unlike the forward there is no weight that could zero them.

    ``d_x_shared`` None: a routed-only layer, the routed sum alone.

    ``out``: the caller's output buffer, contiguous BF16 ``[T, H]``, 16-byte aligned, not an input (the execute
    step of a plan); None allocates the output.

    One host path, as every CuTe DSL op (see fwd_epilogue_bf16): full input check and
    executor key only on a new call signature; torch tensors and plain scalars go straight
    to the compiled TVM FFI function.
    """

    import torch

    has_shared = d_x_shared is not None
    inputs = (d_x_shared, d_x_routed_buffer, top_experts) if has_shared else (d_x_routed_buffer, top_experts)
    try:
        signature = (
            tensor_layout_signature(inputs),
            has_shared,
            _current_device_index(),
            _current_cuda_context_key(),
        )
        validated = _VALIDATED_CALLS.get(signature)  # hashes the signature
    except (AttributeError, TypeError):  # not tensors / unhashable: the full check reports it
        signature = validated = None
    if validated is None:
        device, device_index, num_tokens, hidden_size, topk = _validate_inputs(d_x_shared, d_x_routed_buffer, top_experts)
        validated = (device, num_tokens, hidden_size, topk, (_CODEGEN_EPOCH, device_index, _current_cuda_context_key()))
        if signature is not None:
            _VALIDATED_CALLS.put(signature, validated)
    elif any(t.data_ptr() % 16 for t in inputs):
        # Addresses are not part of the signature: keep the alignment check per call.
        raise ValueError("CuTe DSL backward epilogue inputs must be 16-byte aligned")
    device, num_tokens, hidden_size, topk, cache_key = validated
    cache_key = cache_key + (("has_shared", has_shared),)
    if out is None:
        d_x = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    else:
        d_x = out
        if (
            type(d_x) is not torch.Tensor
            or d_x.dtype != torch.bfloat16
            or d_x.device != device
            or tuple(d_x.shape) != (num_tokens, hidden_size)
            or not d_x.is_contiguous()
            or d_x.data_ptr() % 16
        ):
            raise ValueError("out must be a contiguous 16-byte aligned CUDA BF16 [T, H] tensor on the input device")
        if any(d_x.data_ptr() == tensor.data_ptr() for tensor in inputs):
            raise ValueError("out must not be one of the inputs")
    if not has_shared:
        # The absent shared gradient's argument: the routed buffer's metadata, never loaded.
        d_x_shared = d_x_routed_buffer
    executor = _EXECUTOR_CACHE.get(cache_key)
    if executor is None:
        executor = _compile_bwd_epilogue_executor(
            d_x_shared,
            d_x_routed_buffer,
            top_experts,
            d_x,
            num_tokens,
            hidden_size,
            topk,
            cache_key,
            has_shared,
        )
    executor(d_x_shared, d_x_routed_buffer, top_experts, d_x, num_tokens, hidden_size, topk, current_stream_handle(device.index))
    # No record_stream (see fwd_epilogue_bf16).
    return d_x


def compile_bwd_epilogue(d_x_shared, d_x_routed_buffer, top_experts) -> None:
    """The plan step: compile one variant (shared gradient present or not) without launching.  The arguments
    are checked as a call's are and only give the kernel's tensor ranks and dtypes, so workspace buffers of a
    call's shapes stand in for the caller's tensors; ``d_x_shared`` None is the routed-only variant."""
    has_shared = d_x_shared is not None
    device, device_index, num_tokens, hidden_size, topk = _validate_inputs(d_x_shared, d_x_routed_buffer, top_experts)
    cache_key = (_CODEGEN_EPOCH, device_index, _current_cuda_context_key(), ("has_shared", has_shared))
    stand_in = d_x_shared if has_shared else d_x_routed_buffer  # the shared gradient's and the output's argument
    _compile_bwd_epilogue_executor(stand_in, d_x_routed_buffer, top_experts, stand_in, num_tokens, hidden_size, topk, cache_key, has_shared)


def clear_bwd_epilogue_executor_cache(*, synchronize: bool = True) -> None:
    """Release compiled executors after their launches are quiescent."""

    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device_index in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device_index)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
        _VALIDATED_CALLS.clear()


__all__ = [
    "bwd_epilogue_bf16",
    "compile_bwd_epilogue",
    "clear_bwd_epilogue_executor_cache",
]
