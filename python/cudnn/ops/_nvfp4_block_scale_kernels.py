# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTeDSL kernels for explicit NVFP4 block-scale conversion.

The kernels are deliberately independent of FROST GEMM.  Quantized data and
scale factors are materialized once and can be consumed by any operation that
implements the same public E2M1 + E4M3/F8_128x4 tensor contract.
"""

from __future__ import annotations

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

BLOCK_SIZE = 16
MAX_GROUPS_PER_ROW = 1024
ROWS_PER_CTA = 1
MAX_GROUPS_FOR_128_THREADS = 384


def _threads_per_row(k: int) -> int:
    return 128 if k // BLOCK_SIZE <= MAX_GROUPS_FOR_128_THREADS else 256


def _validate_k(k: int) -> None:
    if k <= 0 or k % 64:
        raise ValueError(f"NVFP4 block-scale conversion requires K divisible by 64; got {k}")
    groups_per_row = k // BLOCK_SIZE
    if groups_per_row > MAX_GROUPS_PER_ROW:
        raise ValueError("NVFP4 block-scale conversion supports at most " f"{MAX_GROUPS_PER_ROW} groups per row; got {groups_per_row}")


def _scale_storage_index(row, group, groups_per_row: int):
    """F8_128x4 byte offset for logical ``[row, group]``."""

    scale_block = (row >> 7) * cutlass.Int64(groups_per_row // 4) + (group >> 2)
    return scale_block * cutlass.Int64(512) + (row & 31) * cutlass.Int64(16) + ((row & 127) >> 5) * cutlass.Int64(4) + (group & 3)


class _Nvfp4BlockScaleQuantizeKernel:
    """Each thread quantizes one contiguous 16-element block."""

    def __init__(self, *, k: int):
        _validate_k(k)
        self.k = k
        self.groups_per_row = k // BLOCK_SIZE
        self.threads_per_row = _threads_per_row(k)
        self.group_passes = (self.groups_per_row + self.threads_per_row - 1) // self.threads_per_row

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        encode_scale: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
    ):
        tid = cute.arch.thread_idx()[0]
        group_lane = tid & (self.threads_per_row - 1)
        row = cutlass.Int64(cute.arch.block_idx()[0])

        encode = encode_scale.iterator.raw_ptr().load()
        packed_ptr = cute.make_ptr(
            cutlass.Int64,
            packed.iterator.toint(),
            mem_space=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        scale_ptr = cute.make_ptr(
            cutlass.Float8E4M3FN,
            scales.iterator.toint(),
            mem_space=cute.AddressSpace.gmem,
            assumed_align=16,
        )

        for group_pass in cutlass.range_constexpr(self.group_passes):
            group = group_lane + group_pass * self.threads_per_row
            if group < self.groups_per_row:
                group_i64 = cutlass.Int64(group)
                src_offset = row * cutlass.Int64(self.k) + group_i64 * cutlass.Int64(BLOCK_SIZE)
                # The row and group strides preserve 16-byte alignment.  Re-state
                # it after the dynamic offset so autovec_copy emits vector loads.
                source_ptr = cute.make_ptr(
                    cutlass.BFloat16,
                    (x.iterator + src_offset).llvm_ptr,
                    mem_space=cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                source = cute.make_tensor(source_ptr, cute.make_layout((BLOCK_SIZE,)))
                loaded = cute.make_rmem_tensor((BLOCK_SIZE,), cutlass.BFloat16)
                cute.autovec_copy(source, loaded)
                values = cute.make_rmem_tensor((BLOCK_SIZE,), cutlass.Float32)
                values.store(loaded.load().to(cutlass.Float32) * encode)

                vector = values.load().to_vector()
                absolute = cute.math.abs(vector)
                amax = absolute[0]
                for i in cutlass.range_constexpr(1, BLOCK_SIZE):
                    amax = cute.math.max(amax, absolute[i])

                # Store the rounded E4M3 scale, then normalize by exactly that
                # stored value.  This matches the NVFP4 serving recipe used by
                # the existing FROST terminal quantizer.
                scale_f32 = amax * cute.arch.rcp_approx(cutlass.Float32(6.0))
                scale_e4m3 = scale_f32.to(cutlass.Float8E4M3FN)
                widened_scale = scale_e4m3.to(cutlass.Float32)
                inverse = cute.math.min(
                    cute.arch.rcp_approx(widened_scale),
                    cutlass.Float32(3.402823466e38),
                )
                for i in cutlass.range_constexpr(0, BLOCK_SIZE, 2):
                    values[i], values[i + 1] = cute.arch.mul_packed_f32x2(
                        (vector[i], vector[i + 1]),
                        (inverse, inverse),
                        rnd="rn",
                        ftz=False,
                    )

                packed_e2m1 = cute.make_rmem_tensor((BLOCK_SIZE,), cutlass.Float4E2M1FN)
                packed_e2m1.store(values.load().to(cutlass.Float4E2M1FN))
                packed_word = cute.recast_tensor(packed_e2m1, cutlass.Int64)
                (packed_ptr + row * cutlass.Int64(self.groups_per_row) + group_i64).store(packed_word[0])

                scale_idx = _scale_storage_index(row, group_i64, self.groups_per_row)
                (scale_ptr + scale_idx).store(scale_e4m3)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        encode_scale: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(x, encode_scale, packed, scales).launch(
            grid=(x.shape[1] // ROWS_PER_CTA, 1, 1),
            block=(self.threads_per_row * ROWS_PER_CTA, 1, 1),
            stream=stream,
        )


class _Nvfp4BlockScaleDequantizeKernel:
    """Each thread expands one packed E2M1 block to sixteen BF16 values."""

    def __init__(self, *, k: int):
        _validate_k(k)
        self.k = k
        self.groups_per_row = k // BLOCK_SIZE
        self.threads_per_row = _threads_per_row(k)
        self.group_passes = (self.groups_per_row + self.threads_per_row - 1) // self.threads_per_row

    @cute.kernel
    def kernel(
        self,
        packed: cute.Tensor,
        scales: cute.Tensor,
        decode_scale: cute.Tensor,
        output: cute.Tensor,
    ):
        tid = cute.arch.thread_idx()[0]
        group_lane = tid & (self.threads_per_row - 1)
        row = cutlass.Int64(cute.arch.block_idx()[0])

        scale_ptr = cute.make_ptr(
            cutlass.Float8E4M3FN,
            scales.iterator.toint(),
            mem_space=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        decode = decode_scale.iterator.raw_ptr().load()

        for group_pass in cutlass.range_constexpr(self.group_passes):
            group = group_lane + group_pass * self.threads_per_row
            if group < self.groups_per_row:
                group_i64 = cutlass.Int64(group)
                packed_offset = row * cutlass.Int64(self.k // 2) + group_i64 * cutlass.Int64(BLOCK_SIZE // 2)
                # A 16-value group occupies eight bytes.  Recast an eight-byte
                # register tensor so the DSL expands the layout to 16 E2M1
                # elements; recasting a scalar i64 keeps a scalar layout.
                packed_source_ptr = cute.make_ptr(
                    cutlass.Int8,
                    (packed.iterator + packed_offset).llvm_ptr,
                    mem_space=cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                packed_source = cute.make_tensor(packed_source_ptr, cute.make_layout((BLOCK_SIZE // 2,)))
                packed_bytes = cute.make_rmem_tensor((BLOCK_SIZE // 2,), cutlass.Int8)
                cute.autovec_copy(packed_source, packed_bytes)
                packed_e2m1 = cute.recast_tensor(packed_bytes, cutlass.Float4E2M1FN)
                values = cute.make_rmem_tensor((BLOCK_SIZE,), cutlass.Float32)
                values.store(packed_e2m1.load().to(cutlass.Float32))

                scale_idx = _scale_storage_index(row, group_i64, self.groups_per_row)
                factor = (scale_ptr + scale_idx).load().to(cutlass.Float32) * decode
                vector = values.load().to_vector()
                for i in cutlass.range_constexpr(0, BLOCK_SIZE, 2):
                    values[i], values[i + 1] = cute.arch.mul_packed_f32x2(
                        (vector[i], vector[i + 1]),
                        (factor, factor),
                        rnd="rn",
                        ftz=False,
                    )

                output_values = cute.make_rmem_tensor((BLOCK_SIZE,), cutlass.BFloat16)
                output_values.store(values.load().to(cutlass.BFloat16))
                dst_offset = row * cutlass.Int64(self.k) + group_i64 * cutlass.Int64(BLOCK_SIZE)
                destination_ptr = cute.make_ptr(
                    cutlass.BFloat16,
                    (output.iterator + dst_offset).llvm_ptr,
                    mem_space=cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                destination = cute.make_tensor(destination_ptr, cute.make_layout((BLOCK_SIZE,)))
                cute.autovec_copy(output_values, destination)

    @cute.jit
    def __call__(
        self,
        packed: cute.Tensor,
        scales: cute.Tensor,
        decode_scale: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(packed, scales, decode_scale, output).launch(
            grid=(output.shape[1] // ROWS_PER_CTA, 1, 1),
            block=(self.threads_per_row * ROWS_PER_CTA, 1, 1),
            stream=stream,
        )


def compile_nvfp4_block_scale_quantize(*, k: int):
    """Compile one quantizer with symbolic M and plan-time K."""

    kernel = _Nvfp4BlockScaleQuantizeKernel(k=k)
    sym_m = cute.sym_int64(divisibility=128)
    fake_x = make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, sym_m, k),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_encode = make_fake_compact_tensor(
        cutlass.Float32,
        (1, 1, 1),
        stride_order=(2, 1, 0),
        assumed_align=4,
    )
    fake_packed = make_fake_compact_tensor(
        cutlass.Int8,
        (1, sym_m, k // 2),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_scales = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (1, sym_m, k // BLOCK_SIZE),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    return cute.compile(
        kernel,
        fake_x,
        fake_encode,
        fake_packed,
        fake_scales,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )


def compile_nvfp4_block_scale_dequantize(*, k: int):
    """Compile one dequantizer with symbolic M and plan-time K."""

    kernel = _Nvfp4BlockScaleDequantizeKernel(k=k)
    sym_m = cute.sym_int64(divisibility=128)
    fake_packed = make_fake_compact_tensor(
        cutlass.Int8,
        (1, sym_m, k // 2),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_scales = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (1, sym_m, k // BLOCK_SIZE),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_decode = make_fake_compact_tensor(
        cutlass.Float32,
        (1, 1, 1),
        stride_order=(2, 1, 0),
        assumed_align=4,
    )
    fake_output = make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, sym_m, k),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    return cute.compile(
        kernel,
        fake_packed,
        fake_scales,
        fake_decode,
        fake_output,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
