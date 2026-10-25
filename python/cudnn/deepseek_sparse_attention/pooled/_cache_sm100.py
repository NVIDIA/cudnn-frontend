# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct gated pooling, Hadamard rotation, quantization, and paged writes."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8


class PoolCacheKernel:
    def __init__(self, pool_size, page_size, cache_dtype, mode):
        self.pool_size = pool_size
        self.page_size = page_size
        self.cache_dtype = cache_dtype
        self.mode = mode
        self.warps = 4

    @cute.jit
    def _power_two_scale(self, ratio):
        value = cute.make_rmem_tensor((1,), Float32)
        value[0] = ratio
        bits = cute.recast_tensor(value, Int32)[0]
        exponent = ((bits >> 23) & 255) - 127
        if (bits & 0x7FFFFF) != 0:
            exponent += 1
        return cute.math.exp2(Float32(exponent), fastmath=True), exponent

    @cute.jit
    def _compress(self, keys, gates, ape, tail, row, tail_block):
        lane = cute.arch.lane_idx()
        result = cute.make_rmem_tensor((4,), Float32)
        scores = cute.make_rmem_tensor((self.pool_size,), Float32)
        for j in cutlass.range_constexpr(4):
            dim = lane + j * 32
            largest = Float32(-float("inf"))
            for i in cutlass.range_constexpr(self.pool_size):
                if cutlass.const_expr(self.mode == "decode" and i != self.pool_size - 1):
                    score = Float32(tail[tail_block, 1, i, dim])
                else:
                    source = row if cutlass.const_expr(self.mode == "decode") else row - self.pool_size + 1 + i
                    score = Float32(gates[source, dim])
                score += ape[i, dim]
                scores[i] = score
                largest = cute.arch.fmax(largest, score)
            numerator, denominator = Float32(0), Float32(0)
            for i in cutlass.range_constexpr(self.pool_size):
                probability = cute.math.exp2((scores[i] - largest) * 1.4426950408889634, fastmath=True)
                if cutlass.const_expr(self.mode == "decode" and i != self.pool_size - 1):
                    value = Float32(tail[tail_block, 0, i, dim])
                else:
                    source = row if cutlass.const_expr(self.mode == "decode") else row - self.pool_size + 1 + i
                    value = Float32(keys[source, dim])
                numerator += probability * value
                denominator += probability
            result[j] = (numerator / denominator).to(BFloat16).to(Float32)
        return result

    @cute.jit
    def _rotate_and_store(self, values, cache, slot):
        lane = cute.arch.lane_idx()
        for stage in cutlass.range_constexpr(5):
            for j in cutlass.range_constexpr(4):
                partner = cute.arch.shuffle_sync_bfly(values[j], 1 << stage)
                if (lane & (1 << stage)) == 0:
                    values[j] = values[j] + partner
                else:
                    values[j] = partner - values[j]
        for i in cutlass.range_constexpr(2):
            a, b = values[2 * i], values[2 * i + 1]
            values[2 * i], values[2 * i + 1] = a + b, a - b
        for i in cutlass.range_constexpr(2):
            a, b = values[i], values[i + 2]
            values[i], values[i + 2] = a + b, a - b
        for j in cutlass.range_constexpr(4):
            values[j] = (values[j] * 0.08838834764831845).to(BFloat16).to(Float32)

        page, offset = slot // self.page_size, slot % self.page_size
        base = cache.iterator + page * cache.stride[0]
        if cutlass.const_expr(self.cache_dtype == "bf16"):
            output = cute.make_tensor(cute.recast_ptr(base, dtype=BFloat16), cute.make_layout((self.page_size * 128,)))
            for j in cutlass.range_constexpr(4):
                output[offset * 128 + lane + j * 32] = values[j].to(BFloat16)
        elif cutlass.const_expr(self.cache_dtype == "fp8"):
            largest = Float32(1e-4)
            for j in cutlass.range_constexpr(4):
                largest = cute.arch.fmax(largest, Float32(cute.math.absf(values[j])))
            largest = cute.arch.warp_reduction_max(largest)
            scale, exponent = self._power_two_scale(largest * (1.0 / 448.0))
            output = cute.make_tensor(cute.recast_ptr(base, dtype=cutlass.Float8E4M3FN), cute.make_layout((self.page_size * 128,)))
            scale_out = cute.make_tensor(cute.recast_ptr(base + self.page_size * 128, dtype=Float32), cute.make_layout((self.page_size,)))
            quantized = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
            quantized.store((values.load() / scale).to(cutlass.Float8E4M3FN))
            for j in cutlass.range_constexpr(4):
                output[offset * 128 + lane + j * 32] = quantized[j]
            if lane == 0:
                scale_out[offset] = scale
        else:
            output = cute.make_tensor(base, cute.make_layout((self.page_size * 68,)))
            value_bits = cute.recast_tensor(values, Int32)
            for j in cutlass.range_constexpr(4):
                largest = cute.arch.warp_reduction_max(Float32(cute.math.absf(values[j])))
                largest = cute.arch.fmax(largest, Float32(6 * 2**-126))
                scale, exponent = self._power_two_scale(largest * (1.0 / 6.0))
                magnitude = Float32(cute.math.absf(values[j] / scale))
                code = Int32(0)
                if magnitude > 0.25:
                    code = Int32(1)
                if magnitude >= 0.75:
                    code = Int32(2)
                if magnitude > 1.25:
                    code = Int32(3)
                if magnitude >= 1.75:
                    code = Int32(4)
                if magnitude > 2.5:
                    code = Int32(5)
                if magnitude >= 3.5:
                    code = Int32(6)
                if magnitude > 5.0:
                    code = Int32(7)
                code |= (value_bits[j] >> 31) & 8
                partner = cute.arch.shuffle_sync_bfly(code, 1)
                if lane % 2 == 0:
                    output[offset * 64 + j * 16 + lane // 2] = Uint8(code | (partner << 4))
                if lane == 0:
                    output[self.page_size * 64 + offset * 4 + j] = Uint8(exponent + 127)

    @cute.jit
    def __call__(self, keys, gates, ape, cache, tail, pool_slots, tail_slots, positions, cu_seqlens, stream: cuda.CUstream):
        count = cute.size(keys.shape[0]) if cutlass.const_expr(self.mode == "prefill") else cute.size(cu_seqlens.shape[0]) - 1
        self.kernel(keys, gates, ape, cache, tail, pool_slots, tail_slots, positions, cu_seqlens).launch(
            grid=(cute.ceil_div(count, self.warps), 1, 1),
            block=(self.warps * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, keys, gates, ape, cache, tail, pool_slots, tail_slots, positions, cu_seqlens):
        task = cute.arch.block_idx()[0] * self.warps + cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        count = cute.size(keys.shape[0])
        if cutlass.const_expr(self.mode == "prefill"):
            if task < count:
                slot = Int64(pool_slots[task])
                if task >= self.pool_size - 1 and slot >= 0 and slot < cute.size(cache.shape[0]) * self.page_size:
                    values = self._compress(keys, gates, ape, tail, task, Int64(0))
                    self._rotate_and_store(values, cache, slot)
                tail_slot = Int64(tail_slots[task])
                if tail_slot >= 0 and tail_slot < cute.size(tail.shape[0]) * self.pool_size:
                    block = tail_slot // self.pool_size
                    ahead = Int64(-1)
                    if task + self.pool_size < count:
                        ahead = Int64(tail_slots[task + self.pool_size])
                    if ahead < 0 or ahead // self.pool_size != block:
                        for j in cutlass.range_constexpr(4):
                            dim = lane + j * 32
                            tail[block, 0, tail_slot % self.pool_size, dim] = keys[task, dim]
                            tail[block, 1, tail_slot % self.pool_size, dim] = gates[task, dim]
        else:
            if task < cute.size(cu_seqlens.shape[0]) - 1:
                start, end = Int32(cu_seqlens[task]), Int32(cu_seqlens[task + 1])
                for row in range(start, end):
                    tail_slot = Int64(tail_slots[row])
                    position = Int64(positions[row])
                    if position >= 0 and tail_slot >= 0 and tail_slot < cute.size(tail.shape[0]) * self.pool_size:
                        block = tail_slot // self.pool_size
                        slot = Int64(pool_slots[row])
                        if position % self.pool_size == self.pool_size - 1 and slot >= 0 and slot < cute.size(cache.shape[0]) * self.page_size:
                            values = self._compress(keys, gates, ape, tail, row, block)
                            self._rotate_and_store(values, cache, slot)
                        for j in cutlass.range_constexpr(4):
                            dim = lane + j * 32
                            tail[block, 0, tail_slot % self.pool_size, dim] = keys[row, dim]
                            tail[block, 1, tail_slot % self.pool_size, dim] = gates[row, dim]

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
