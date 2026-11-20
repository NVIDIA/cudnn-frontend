# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""BF16 -> MXFP8 quantization kernel on SM100 / SM103 (the implementation of ``cudnn.moe_ep._cutemok.ops.mxfp8_quantize``).

One CTA per 128 x 128 tile: the e4m3 payload plus one e8m0 scale per 32 elements, in the normal
and/or the transposed layout.  The quantization itself is ``_mxfp8_quant.quantize_tile``, shared
with the megakernels' quantization points, so this entry doubles as their byte-level test.

The tile moves between global and shared memory with plain loads and stores.  Rows, columns and
the expert count are runtime; the requested layout pair is compile time (one executor per pair).
"""

import threading


from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, check_device

THREADS = 128  # one tile row per thread
_CODEGEN_EPOCH = "mxfp8-quantize-tile-v2"

from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Int32

from ._host_common import persistent_executor
from ._host_common import (
    ValidatedCalls,
    current_cuda_context_key as _current_cuda_context_key,
    current_stream_handle,
    row_major_view as _row_major_view,
    tensor_layout_signature,
    typed_key,
)
from ._host_common import current_device_index as _current_device_index
from ._mxfp8_quant import (
    QUANT_TILE,
    SCALE_TILE_ROWS,
    SCALE_TILE_WORDS,
    quantize_tile,
)
from .persistent_mxfp8_fwd import dynamic_layout_tensor

# Shared memory: the BF16 tile, then one e4m3 tile with its scale tile per requested layout (the
# transposed one first).  No output overlaps the BF16 tile: quantize_tile's threads store their rows
# while the others still load theirs.
_FP8_TILE_BYTES = QUANT_TILE * QUANT_TILE
_BF16_TILE_BYTES = 2 * _FP8_TILE_BYTES
_SCALE_TILE_BYTES = SCALE_TILE_ROWS * SCALE_TILE_WORDS * 4
_LAYOUT_BYTES = _FP8_TILE_BYTES + _SCALE_TILE_BYTES


class _Mxfp8Quantize:
    """One compiled variant per requested layout pair."""

    def __init__(self, return_normal: bool, return_transposed: bool) -> None:
        self.return_normal = return_normal
        self.return_transposed = return_transposed
        # Byte offset of the normal layout's tiles: after the transposed ones when both are requested.
        self.normal_offset = _BF16_TILE_BYTES + (_LAYOUT_BYTES if return_transposed else 0)
        tile_bytes = _BF16_TILE_BYTES + (int(return_normal) + int(return_transposed)) * _LAYOUT_BYTES

        @cute.struct
        class SharedStorage:
            tile: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, tile_bytes], 1024]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        payload_t: cute.Tensor,
        scales_t: cute.Tensor,
        row_blocks: Int32,
        col_blocks: Int32,
        experts: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x, payload, scales, payload_t, scales_t, row_blocks, col_blocks).launch(
            grid=(col_blocks, row_blocks, experts),
            block=(THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        payload_t: cute.Tensor,
        scales_t: cute.Tensor,
        row_blocks: Int32,
        col_blocks: Int32,
    ):
        thread = cute.arch.thread_idx()[0]
        col_block, row_block, expert = cute.arch.block_idx()
        row_base = row_block * Int32(QUANT_TILE)
        col_base = col_block * Int32(QUANT_TILE)

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        base = storage.tile.data_ptr()
        tile = cute.make_tensor(
            cute.recast_ptr(base, dtype=BFloat16),
            cute.make_layout((QUANT_TILE, QUANT_TILE), stride=(QUANT_TILE, 1)),
        )
        # Column per thread, so consecutive threads read consecutive elements.
        for row in cutlass.range(Int32(QUANT_TILE), unroll=1):
            tile[row, thread] = x[expert, row_base + row, col_base + thread]
        cute.arch.sync_threads()

        quantize_tile(
            base,
            base + self.normal_offset,
            base + self.normal_offset + _FP8_TILE_BYTES,
            base + _BF16_TILE_BYTES,
            base + _BF16_TILE_BYTES + _FP8_TILE_BYTES,
            thread,
            QUANT_TILE,
            self.return_normal,
            self.return_transposed,
        )
        cute.arch.sync_threads()

        # Thread t writes word t % 32 of 32 consecutive rows: one tile row is 32 words.
        payload_words = cute.make_tensor(
            cute.recast_ptr(base + self.normal_offset, dtype=Int32),
            cute.make_layout((QUANT_TILE, QUANT_TILE // 4), stride=(QUANT_TILE // 4, 1)),
        )
        payload_t_words = cute.make_tensor(
            cute.recast_ptr(base + _BF16_TILE_BYTES, dtype=Int32),
            cute.make_layout((QUANT_TILE, QUANT_TILE // 4), stride=(QUANT_TILE // 4, 1)),
        )
        scale_words = cute.make_tensor(
            cute.recast_ptr(base + self.normal_offset + _FP8_TILE_BYTES, dtype=Int32),
            cute.make_layout((SCALE_TILE_ROWS, SCALE_TILE_WORDS), stride=(SCALE_TILE_WORDS, 1)),
        )
        scale_t_words = cute.make_tensor(
            cute.recast_ptr(base + _BF16_TILE_BYTES + _FP8_TILE_BYTES, dtype=Int32),
            cute.make_layout((SCALE_TILE_ROWS, SCALE_TILE_WORDS), stride=(SCALE_TILE_WORDS, 1)),
        )
        word = thread % Int32(QUANT_TILE // 4)
        scale_row = thread // Int32(SCALE_TILE_WORDS)
        scale_slot = thread % Int32(SCALE_TILE_WORDS)
        if cutlass.const_expr(self.return_normal):
            for step in cutlass.range(Int32(QUANT_TILE // 4), unroll=1):
                row = step * Int32(THREADS // (QUANT_TILE // 4)) + thread // Int32(QUANT_TILE // 4)
                payload[expert, row_base + row, col_base // Int32(4) + word] = payload_words[row, word]
            scales[expert * row_blocks + row_block, col_block, scale_row, scale_slot] = scale_words[scale_row, scale_slot]
        if cutlass.const_expr(self.return_transposed):
            for step in cutlass.range(Int32(QUANT_TILE // 4), unroll=1):
                row = step * Int32(THREADS // (QUANT_TILE // 4)) + thread // Int32(QUANT_TILE // 4)
                payload_t[expert, col_base + row, row_base // Int32(4) + word] = payload_t_words[row, word]
            scales_t[expert * col_blocks + col_block, row_block, scale_row, scale_slot] = scale_t_words[scale_row, scale_slot]


_EXECUTOR_CACHE = {}
_EXECUTOR_CACHE_LOCK = threading.Lock()
_EXECUTOR_CACHE_DEVICES = set()
_VALIDATED_CALLS = ValidatedCalls()


def _validate(x_bf16, return_normal, return_transposed):
    if not isinstance(x_bf16, torch.Tensor):
        raise TypeError("x_bf16 must be a torch.Tensor")
    if x_bf16.ndim not in (2, 3):
        raise ValueError("x_bf16 must have shape (M, N) or (E, M, N)")
    if any(extent <= 0 for extent in x_bf16.shape):
        raise ValueError("x_bf16 dimensions must be positive")
    if x_bf16.shape[-2] % QUANT_TILE or x_bf16.shape[-1] % QUANT_TILE:
        raise ValueError("x_bf16 M and N dimensions must be divisible by 128")
    if type(return_normal) is not bool or type(return_transposed) is not bool:
        raise TypeError("return_normal and return_transposed must be booleans")
    if not return_normal and not return_transposed:
        raise ValueError("at least one quantized layout must be requested")
    if not x_bf16.is_cuda or x_bf16.dtype is not torch.bfloat16 or not x_bf16.is_contiguous():
        raise ValueError("x_bf16 must be a contiguous bfloat16 CUDA tensor")
    device = x_bf16.device
    device_index = device.index if device.index is not None else _current_device_index()
    if _current_device_index() != device_index:
        raise RuntimeError("the input CUDA device must be current")
    check_device(device, "the CuTe DSL MXFP8 quantizer")
    return device_index


def _make_arguments(source, payload, scales, payload_t, scales_t, row_blocks, col_blocks, experts, stream):
    """The CuTe tensors and scalars ``cute.compile`` traces (the call passes the torch views)."""
    return (
        dynamic_layout_tensor(source),
        dynamic_layout_tensor(payload),
        dynamic_layout_tensor(scales),
        dynamic_layout_tensor(payload_t),
        dynamic_layout_tensor(scales_t),
        Int32(row_blocks),
        Int32(col_blocks),
        Int32(experts),
        stream,
    )


def mxfp8_quantize(
    x_bf16: torch.Tensor,
    return_normal: bool,
    return_transposed: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Quantize a BF16 matrix to MXFP8 in the normal and/or transposed layout.

    ``x_bf16`` is a contiguous bfloat16 ``(M, N)`` or ``(E, M, N)`` tensor with M and N divisible by 128.
    Returns ``(x_fp8, x_sc, x_fp8_t, x_sc_t)``, None for a layout that was not requested:
        x_fp8:   float8_e4m3fn (M, N) or (E, M, N)
        x_sc:    uint8 (E * M // 128, N // 128, 32, 16)
        x_fp8_t: float8_e4m3fn (N, M) or (E, N, M)
        x_sc_t:  uint8 (E * N // 128, M // 128, 32, 16)

    One host path, as the other CuTe DSL ops: the full input check when the call signature
    (layout, flags, device, context) is new, the alignment check every call; the executor per
    layout pair is cached; the call passes the torch views and plain scalars to the compiled
    TVM FFI function on the current stream, without ``record_stream``.
    """
    try:
        signature = (
            tensor_layout_signature((x_bf16,)),
            typed_key(return_normal, return_transposed),
            _current_device_index(),
            _current_cuda_context_key(),
        )
        device_index = _VALIDATED_CALLS.get(signature)
    except (AttributeError, TypeError):  # not a tensor / unhashable: the full check reports it
        signature = device_index = None
    if device_index is None:
        device_index = _validate(x_bf16, return_normal, return_transposed)
        if signature is not None:
            _VALIDATED_CALLS.put(signature, device_index)
    if x_bf16.data_ptr() % 16:
        raise ValueError("x_bf16 must be 16-byte aligned")
    device = x_bf16.device

    experts = int(x_bf16.shape[0]) if x_bf16.ndim == 3 else 1
    rows = int(x_bf16.shape[-2])
    columns = int(x_bf16.shape[-1])
    row_blocks = rows // QUANT_TILE
    col_blocks = columns // QUANT_TILE

    x_fp8 = x_sc = x_fp8_t = x_sc_t = None
    if return_normal:
        x_fp8 = torch.empty(x_bf16.shape, dtype=torch.float8_e4m3fn, device=device)
        x_sc = torch.empty((experts * row_blocks, col_blocks, SCALE_TILE_ROWS, 16), dtype=torch.uint8, device=device)
    if return_transposed:
        transposed_shape = (experts, columns, rows) if x_bf16.ndim == 3 else (columns, rows)
        x_fp8_t = torch.empty(transposed_shape, dtype=torch.float8_e4m3fn, device=device)
        x_sc_t = torch.empty((experts * col_blocks, row_blocks, SCALE_TILE_ROWS, 16), dtype=torch.uint8, device=device)

    # The kernel writes int32 words: a 32-element payload block is eight aligned
    # words, and a tile row's four scale bytes are one packed word.
    source = _row_major_view(x_bf16).reshape(experts, rows, columns)
    payload = x_fp8.view(torch.int32).reshape(experts, rows, columns // 4) if return_normal else None
    scales = x_sc.view(torch.int32) if return_normal else None
    payload_t = x_fp8_t.view(torch.int32).reshape(experts, columns, rows // 4) if return_transposed else None
    scales_t = x_sc_t.view(torch.int32) if return_transposed else None
    # Unrequested layouts are compile-time dead; pass the live pair as stand-ins.
    if payload is None:
        payload, scales = payload_t, scales_t
    if payload_t is None:
        payload_t, scales_t = payload, scales

    stream = current_stream_handle(device_index)
    key = (
        _CODEGEN_EPOCH,
        device_index,
        _current_cuda_context_key(),
        (return_normal, return_transposed),
        _COMPILE_OPTIONS,
    )
    executor = _EXECUTOR_CACHE.get(key)
    if executor is None:
        with _EXECUTOR_CACHE_LOCK:
            executor = _EXECUTOR_CACHE.get(key)
            if executor is None:
                executor = persistent_executor(
                    ("mxfp8_quantize", key[0], *key[3:]),
                    lambda: cute.compile(
                        _Mxfp8Quantize(return_normal, return_transposed),
                        *_make_arguments(source, payload, scales, payload_t, scales_t, row_blocks, col_blocks, experts, stream),
                        options=_COMPILE_OPTIONS,
                    ),
                )
                _EXECUTOR_CACHE[key] = executor
                _EXECUTOR_CACHE_DEVICES.add(device_index)
    # The TVM FFI executor takes the torch views and plain scalars; no record_stream (the inputs
    # are used on the current stream, like PyTorch's own ops).
    executor(source, payload, scales, payload_t, scales_t, row_blocks, col_blocks, experts, stream)
    return x_fp8, x_sc, x_fp8_t, x_sc_t


def clear_mxfp8_quantize_executor_cache(*, synchronize: bool = True) -> None:
    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            for device in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
        _VALIDATED_CALLS.clear()


__all__ = ["clear_mxfp8_quantize_executor_cache", "mxfp8_quantize"]
