# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combine N=128 state-tile outputs and optionally apply SiLU before BF16 rounding."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


class Mamba2GateForward:
    def __init__(self, elements, threads=128):
        self.elements = elements
        self.threads = threads

    @cute.jit
    def __call__(self, raw, z, out, saved, stream: cuda.CUstream):
        self.kernel(raw, z, out, saved).launch(
            grid=((self.elements + self.threads * 8 - 1) // (self.threads * 8), 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, raw, z, out, saved):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        idx = (block * self.threads + tid) * 8
        if idx < self.elements:
            row, col = idx // 64, idx % 64
            offset = row * 128 + col
            values = (raw.iterator.raw_ptr() + offset).load(count=8, alignment=16)
            values += (raw.iterator.raw_ptr() + offset + 64).load(count=8, alignment=16)
            if cutlass.const_expr(z is not None):
                gates = (z.iterator.raw_ptr() + idx).load(count=8, alignment=16).to(cutlass.Float32)
            outputs = []
            for j in cutlass.range_constexpr(8):
                activation = cutlass.Float32(1.0)
                if cutlass.const_expr(z is not None):
                    activation = gates[j] / (1.0 + cute.math.exp(-gates[j], fastmath=True))
                outputs.append((values[j] * activation).to(cutlass.BFloat16))
            (out.iterator.raw_ptr() + idx).store(cutlass.Vector.from_elements(tuple(outputs), cutlass.BFloat16), alignment=16)
            if cutlass.const_expr(saved is not None):
                (saved.iterator.raw_ptr() + idx).store(values.to(cutlass.BFloat16), alignment=16)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
