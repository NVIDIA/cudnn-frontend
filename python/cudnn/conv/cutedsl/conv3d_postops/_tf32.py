# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Device algorithm source: pkuwangh/mlsys@72c2384a5c325198d6600906d5835d1ef6faabac
# mlsys/gpu/kernels/wan_vae/mega_wan_vae/kernels/_precision.py

"""Explicit nearest/even operand rounding for TF32 convolution."""

import cutlass
from cuda.bindings import driver
from cutlass import cute


@cute.jit
def round_tf32(value):
    bits = value.bitcast(cutlass.Int32)
    rounded = (bits + 4095 + ((bits >> 13) & 1)) & -8192
    rounded = cutlass.vector.where((bits & 0x7F800000) != 0x7F800000, rounded, bits)
    return rounded.bitcast(cutlass.Float32)


@cute.kernel
def _round_kernel(x: cute.Tensor, output: cute.Tensor, numel: cutlass.Int64):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    offset = (cutlass.Int64(block) * 256 + tid) * 4
    if offset < numel:
        value = (x.iterator.raw_ptr() + offset).load(count=4, alignment=16)
        (output.iterator.raw_ptr() + offset).store(round_tf32(value), alignment=16)


class RoundInputLaunch:
    """Launch explicit input rounding into the caller's FP32 workspace."""

    @cute.jit
    def __call__(self, x: cute.Tensor, output: cute.Tensor, numel: cutlass.Int64, stream: driver.CUstream):
        _round_kernel(x, output, numel).launch(grid=((numel + 1023) // 1024, 1, 1), block=(256, 1, 1), stream=stream)


_round_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
