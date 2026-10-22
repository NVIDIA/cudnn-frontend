# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match KV neighborhoods in groups of eight and partition common/residual edges.

Duplicate edges disable pairing for their eight-KV group, preserving multiset
semantics through the regular path. All metadata is rebuilt on every call.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from ..utils.kernel_utils import warp_prefix_sum


class PairedKvNeighborhoods:
    @cute.kernel
    def clear(self, bits: cute.Tensor, duplicates: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        i = block * 256 + tid
        flat_bits = cute.make_tensor(bits.iterator, cute.make_layout(cute.size(bits)))
        flat_duplicates = cute.make_tensor(duplicates.iterator, cute.make_layout(cute.size(duplicates)))
        if i < cute.size(bits):
            flat_bits[i] = Int32(0)
        if i < cute.size(duplicates):
            flat_duplicates[i] = Int32(0)

    @cute.kernel
    def scatter(
        self,
        indices: cute.Tensor,
        counts: cute.Tensor,
        bits: cute.Tensor,
        duplicates: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        q, h, b = cute.arch.block_idx()
        mask = Int32(1) << (q % 32)
        count = counts[b, h, q]
        for edge in cutlass.range(tid, count, 128):
            k = indices[b, h, q, edge]
            if k >= 0 and k < bits.shape[2]:
                ptr = bits.iterator + cute.crd2idx((b, h, k, q // 32), bits.layout)
                previous = Int32(cute.arch.atomic_or(ptr.llvm_ptr, mask, sem="relaxed", scope="gpu"))
                if (previous & mask) != 0:
                    dup_ptr = duplicates.iterator + cute.crd2idx((b, h, k // 2), duplicates.layout)
                    cute.arch.atomic_or(dup_ptr.llvm_ptr, Int32(1), sem="relaxed", scope="gpu")


def _matchings(vertices):
    if not vertices:
        yield ()
    else:
        first = vertices[0]
        for other in vertices[1:]:
            for rest in _matchings(tuple(x for x in vertices if x not in (first, other))):
                yield ((first, other),) + rest


class LocalPairMatch8:
    def __init__(self):
        edges = tuple((i, j) for i in range(8) for j in range(i + 1, 8))
        patterns = tuple(_matchings(tuple(range(8))))
        assert len(patterns) == 105
        self.edges = edges
        self.pair_ids = tuple(tuple(edges.index(edge) for edge in pattern) for pattern in patterns)
        self.larger_nodes = tuple(tuple(j for i, j in pattern) for pattern in patterns)
        encodings = []
        for pattern in patterns:
            partner = [0] * 8
            for i, j in pattern:
                partner[i], partner[j] = j, i
            encodings.append(sum(p << (3 * i) for i, p in enumerate(partner)))
        self.encodings = tuple(encodings)

    @cute.jit
    def __call__(
        self,
        bits: cute.Tensor,
        duplicates: cute.Tensor,
        partners: cute.Tensor,
        owners: cute.Tensor,
        common: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.match(bits, duplicates, partners, owners, common).launch(
            grid=(cute.ceil_div(bits.shape[2], 8), bits.shape[1], bits.shape[0]), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def match(
        self,
        bits: cute.Tensor,
        duplicates: cute.Tensor,
        partners: cute.Tensor,
        owners: cute.Tensor,
        common: cute.Tensor,
    ):
        lane, _, _ = cute.arch.thread_idx()
        group, head, batch = cute.arch.block_idx()
        start = group * 8
        weights = cute.make_rmem_tensor((28,), Int32)
        weights.fill(0)
        values = cute.make_rmem_tensor((8,), Int32)
        for word in cutlass.range(lane, bits.shape[3], 32):
            for i in cutlass.range_constexpr(8):
                values[i] = Int32(0)
                if start + i < bits.shape[2]:
                    values[i] = bits[batch, head, start + i, word]
            for e in cutlass.range_constexpr(28):
                weights[e] += Int32(cute.arch.popc(values[self.edges[e][0]] & values[self.edges[e][1]]))
        for e in cutlass.range_constexpr(28):
            weights[e] = cute.arch.shuffle_sync(warp_prefix_sum(weights[e], lane), 31)
        best_score, best_tie, encoding = Int32(-1), Int32(0), Int32(0)
        for p in cutlass.range_constexpr(105):
            if lane == p % 32:
                score, real_pairs = Int32(0), Int32(0)
                for j in cutlass.range_constexpr(4):
                    score += weights[self.pair_ids[p][j]]
                    real_pairs += Int32(start + self.larger_nodes[p][j] < bits.shape[2])
                # Prefer real-real pairs over multiple real-dummy ties at tails.
                tie = real_pairs * 4096 + (104 - p) * 32 + lane
                if score > best_score or (score == best_score and tie > best_tie):
                    best_score, best_tie, encoding = score, tie, Int32(self.encodings[p])
        for step in cutlass.range_constexpr(5):
            other_score = cute.arch.shuffle_sync(best_score, lane ^ (1 << step))
            other_tie = cute.arch.shuffle_sync(best_tie, lane ^ (1 << step))
            if other_score > best_score or (other_score == best_score and other_tie > best_tie):
                best_score, best_tie = other_score, other_tie
        encoding = cute.arch.shuffle_sync(encoding, best_tie & 31)
        duplicate = Int32(0)
        for i in cutlass.range_constexpr(4):
            if start // 2 + i < duplicates.shape[2]:
                duplicate |= duplicates[batch, head, start // 2 + i]
        if duplicate != 0:
            encoding = Int32(self.encodings[0])
        relative_partner = (encoding >> (3 * (lane % 8))) & 7
        partner = start + relative_partner
        valid = lane < 8 and start + lane < bits.shape[2]
        owner = valid and relative_partner > lane
        rank = warp_prefix_sum(Int32(owner), lane)
        if valid:
            partners[batch, head, start + lane] = Int32(-1) if duplicate != 0 or partner >= bits.shape[2] else partner
        if owner:
            owners[batch, head, start // 2 + rank - 1] = start + lane
        if lane == 0:
            common[batch, head, group] = Int32(0) if duplicate != 0 else best_score


class LocalPairPartition8:
    @cute.jit
    def __call__(
        self,
        bits: cute.Tensor,
        partners: cute.Tensor,
        offsets: cute.Tensor,
        indices: cute.Tensor,
        residuals: cute.Tensor,
        bucket_size: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.partition(bits, partners, offsets, indices, residuals, bucket_size).launch(
            grid=(cute.ceil_div(bits.shape[2], 2) * offsets.shape[2], bits.shape[1], bits.shape[0]),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def partition(
        self,
        bits: cute.Tensor,
        partners: cute.Tensor,
        offsets: cute.Tensor,
        indices: cute.Tensor,
        residuals: cute.Tensor,
        bucket_size: cutlass.Constexpr,
    ):
        lane, _, _ = cute.arch.thread_idx()
        task, h, b = cute.arch.block_idx()
        tasks_per_group = cute.ceil_div(bits.shape[2], 2)
        pair_slot = task % tasks_per_group
        group = task // tasks_per_group
        first_k = partners[b, h, bits.shape[2] + pair_slot]
        start = (first_k // 8) * 8
        partner0 = partners[b, h, start]
        partner1 = Int32(-1)
        if start + 1 < bits.shape[2]:
            partner1 = partners[b, h, start + 1]
        fallback = partner0 < 0 and partner1 < 0
        if first_k < bits.shape[2]:
            second_k = partners[b, h, first_k]
            if fallback:
                second_k = first_k + 1
            has_second = second_k >= 0 and second_k < bits.shape[2]
            begin0 = offsets[b, h, group, first_k]
            end0 = offsets[b, h, group, first_k + 1]
            begin1, end1 = Int32(0), Int32(0)
            valid = has_second and not fallback
            if has_second:
                begin1 = offsets[b, h, group, second_k]
                end1 = offsets[b, h, group, second_k + 1]
            common_count = Int32(0)
            q_begin = group * bucket_size
            q_end = q_begin + bucket_size
            word_begin = q_begin // 32
            word_end = cute.min(cute.ceil_div(q_end, 32), bits.shape[3])
            if valid:
                for word in cutlass.range(word_begin + lane, word_end, 32):
                    common = bits[b, h, first_k, word] & bits[b, h, second_k, word]
                    if word * 32 < q_begin:
                        common = common & (Int32(-1) << (q_begin % 32))
                    if word * 32 + 32 > q_end and q_end % 32 != 0:
                        common = common & ((Int32(1) << (q_end % 32)) - 1)
                    common_count += Int32(cute.arch.popc(common))
                common_count = cute.arch.shuffle_sync(warp_prefix_sum(common_count, lane), 31)
            residual0 = end0 - begin0 - common_count
            residual1 = end1 - begin1 - common_count
            if lane == 0:
                residuals[b, h, group, first_k] = residual0
                if has_second:
                    residuals[b, h, group, second_k] = residual1
            if common_count > 0:
                running0 = begin0
                running1 = begin1
                running_common = begin0 + residual0
                for tile in cutlass.range(cute.ceil_div(word_end - word_begin, 32)):
                    word = word_begin + tile * 32 + lane
                    first = Int32(0)
                    second = Int32(0)
                    if word < word_end:
                        first = bits[b, h, first_k, word]
                        second = bits[b, h, second_k, word]
                    keep = Int32(-1)
                    if word * 32 < q_begin:
                        keep = keep & (Int32(-1) << (q_begin % 32))
                    if word * 32 + 32 > q_end and q_end % 32 != 0:
                        keep = keep & ((Int32(1) << (q_end % 32)) - 1)
                    for category in cutlass.range_constexpr(3):
                        if cutlass.const_expr(category == 0):
                            value = first & ~second & keep
                            running = running0
                        elif cutlass.const_expr(category == 1):
                            value = second & ~first & keep
                            running = running1
                        else:
                            value = first & second & keep
                            running = running_common
                        count = Int32(cute.arch.popc(value))
                        inclusive = warp_prefix_sum(count, lane)
                        position = running + inclusive - count
                        running += cute.arch.shuffle_sync(inclusive, 31)
                        while value != 0:
                            low = value & -value
                            bit = Int32(31) - Int32(cute.arch.clz(low))
                            indices[b, h, position] = word * 32 + bit
                            position += 1
                            value = value & (value - 1)
                        if cutlass.const_expr(category == 0):
                            running0 = running
                        elif cutlass.const_expr(category == 1):
                            running1 = running
                        else:
                            running_common = running
