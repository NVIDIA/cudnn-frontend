# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local two-pair improvement with shape-independent shared-memory tiles."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cudnn._cutlass_compat import SmemAllocator
from cutlass import Int32
from ..utils.kernel_utils import warp_prefix_sum


class LocalPairRefineBatchWide:
    def __init__(self, width, iterations, word_capacity):
        assert width in (16, 32, 64)
        assert word_capacity > 0
        self.width, self.iterations = width, iterations
        self.word_capacity = word_capacity
        self.threads = 512 if width == 64 else (256 if width == 32 else 128)

    @cute.jit
    def __call__(self, bits: cute.Tensor, duplicates: cute.Tensor, partners: cute.Tensor, owners: cute.Tensor, common: cute.Tensor, stream: cuda.CUstream):
        self.refine(bits, duplicates, partners, owners, common).launch(
            grid=(cute.ceil_div(bits.shape[2], self.width), bits.shape[1], bits.shape[0]), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def refine(self, bits: cute.Tensor, duplicates: cute.Tensor, partners: cute.Tensor, owners: cute.Tensor, common: cute.Tensor):
        # Keep the historical cost limit, evaluated for every invocation.
        # Above it, exact-eight matching remains untouched.
        if bits.shape[3] <= 1024:
            tid, _, _ = cute.arch.thread_idx()
            group, h, b = cute.arch.block_idx()
            lane = tid % 32
            start = group * self.width
            real = cute.min(self.width, bits.shape[2] - start)
            pairs = cute.ceil_div(real, 2)
            smem = SmemAllocator()
            data = smem.allocate_tensor(Int32, cute.make_layout((self.width, self.word_capacity), stride=(1, self.width)), byte_alignment=16)
            weights = smem.allocate_tensor(Int32, cute.make_layout((self.width, self.width), stride=(self.width, 1)), byte_alignment=16)
            aa = smem.allocate_tensor(Int32, cute.make_layout(self.width // 2), byte_alignment=16)
            bb = smem.allocate_tensor(Int32, cute.make_layout(self.width // 2), byte_alignment=16)
            duplicate = Int32(0)
            for i in cutlass.range_constexpr(self.width // 2):
                if start // 2 + i < duplicates.shape[2]:
                    duplicate |= duplicates[b, h, start // 2 + i]
            if tid < pairs:
                a = owners[b, h, start // 2 + tid] - start
                second = partners[b, h, start + a] - start
                if second < 0:
                    second = real
                if duplicate != 0:
                    a, second = tid * 2, tid * 2 + 1
                aa[tid], bb[tid] = a, second
            accum = cute.make_rmem_tensor((self.width * self.width // self.threads,), Int32)
            accum.fill(0)
            col = tid % self.width
            first_row = tid // self.width
            for word_start in cutlass.range(0, bits.shape[3], self.word_capacity):
                # Uniform CTA branch: finish all readers before reusing the tile.
                if word_start != 0:
                    cute.arch.sync_threads()
                tile_words = cute.min(self.word_capacity, bits.shape[3] - word_start)
                for index in cutlass.range(tid, self.width * tile_words, self.threads):
                    row = index % self.width
                    word = index // self.width
                    value = Int32(0)
                    if row < real:
                        value = bits[b, h, start + row, word_start + word]
                    data[row, word] = value
                cute.arch.sync_threads()
                for word in cutlass.range(tile_words):
                    column_bits = data[col, word]
                    for part in cutlass.range_constexpr(self.width * self.width // self.threads):
                        row = first_row + part * (self.threads // self.width)
                        accum[part] += Int32(cute.arch.popc(data[row, word] & column_bits))
            for part in cutlass.range_constexpr(self.width * self.width // self.threads):
                weights[first_row + part * (self.threads // self.width), col] = accum[part]
            cute.arch.sync_threads()
            if tid < 32:
                if duplicate == 0:
                    iteration = Int32(0)
                    active = Int32(1)
                    while iteration < self.iterations and active != 0:
                        gain, code, mate = Int32(0), Int32(2147483647), Int32(0)
                        if lane < pairs:
                            for j in cutlass.range(pairs):
                                if lane != j:
                                    a, d, c, e = aa[lane], bb[lane], aa[j], bb[j]
                                    old = weights[a, d] + weights[c, e]
                                    first = weights[a, c] + weights[d, e] - old
                                    second = weights[a, e] + weights[d, c] - old
                                    candidate = cute.max(first, second)
                                    candidate_code = (cute.min(lane, j) * pairs + cute.max(lane, j)) * 2 + Int32(second > first)
                                    if candidate > gain or (candidate == gain and candidate_code < code):
                                        gain, code, mate = candidate, candidate_code, j
                        other_mate = cute.arch.shuffle_sync(mate, mate)
                        selected = lane < pairs and gain > 0 and lane < mate and other_mate == lane
                        active = Int32(cute.arch.vote_ballot_sync(selected) != 0)
                        iteration += 1
                        # All reads finish before any selected disjoint pair writes.
                        cute.arch.sync_warp()
                        if selected:
                            a, c, d, e = aa[lane], aa[mate], bb[lane], bb[mate]
                            x, y, z, t = a, c, d, e
                            if code % 2 != 0:
                                x, y, z, t = a, e, d, c
                            aa[lane], bb[lane] = cute.min(x, y), cute.max(x, y)
                            aa[mate], bb[mate] = cute.min(z, t), cute.max(z, t)
                        cute.arch.sync_warp()
                score = Int32(0)
                if lane < pairs:
                    a, c = aa[lane], bb[lane]
                    owners[b, h, start // 2 + lane] = start + a
                    partners[b, h, start + a] = Int32(-1) if duplicate != 0 or c >= real else start + c
                    if c < real:
                        partners[b, h, start + c] = Int32(-1) if duplicate != 0 else start + a
                    if duplicate == 0:
                        score = weights[a, c]
                score = cute.arch.shuffle_sync(warp_prefix_sum(score, lane), 31)
                if lane == 0:
                    common[b, h, group] = score
