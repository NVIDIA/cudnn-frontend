# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hopper sparse-prefill forward for DeepSeek latent attention.

One CTA owns one query and one 64-head tile.  Two compute warpgroups process a
pair of 64-token sparse KV tiles at a time while a third warpgroup gathers the
next reusable KV halves.  The compute warpgroups exchange their probabilities:
WG0 owns output columns [0, 256), WG1 owns [256, 512), and both accumulate both
tiles.  The latent value is ``KV[..., :512]``.

The implementation intentionally keeps sequence and physical top-k extents
dynamic.  Only Dqk, element dtype, optional operands, and the optional indexer
prefix change generated code.
"""

import math
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.nvgpu import cpasync
from cutlass.experimental import primitives as prims

from cudnn.block_sparse_attention.csrc.utils.sm90_barriers import named_barrier_arrive, named_barrier_sync
from cudnn.deepseek_sparse_attention.utils import copy as copy_utils
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b
from cudnn.frost.tile_dsl.wgmma import wgmma_m64nNk16_f32
from cudnn.sdpa.fwd.kernels.sm90._common_hopper import owned_row, quad_reduce, rescale_output, store_fragment, wgmma_descriptor


class SparseAttentionForwardSm90:
    """Fused SM90 sparse attention over 64 heads and paired 64-row KV tiles."""

    HEAD_TILE = 64
    TOPK_TILE = 64
    VALUE_DIM = 512
    VALUE_HALF = 256
    THREADS_PER_WARPGROUP = 128
    THREADS = 3 * THREADS_PER_WARPGROUP
    VECTOR_ELEMS = 8  # one 16-byte cp.async for FP16/BF16
    QK_SLOTS = 32
    O_SLOTS = 128
    MAX_INIT = -1.0e30
    # CUTLASS TMA::CacheHintSm90::EVICT_FIRST.  Q is consumed once per CTA;
    # keep it from displacing the sparse KV working set, as in FlashMLA.
    L2_EVICT_FIRST = 0x12F0000000000000

    # KV ready/free hand-offs are transactional mbarriers, matching the Hopper
    # C++ kernel: cp.async completion advances READY without a producer-side
    # wait, and the last compute reader advances FREE.  Named barriers remain
    # only for the two compute warpgroups' register/shared exchanges.
    MB_READY_0L = 0
    MB_READY_0R = 1
    MB_READY_1L = 2
    MB_READY_1R = 3
    MB_FREE_0L = 4
    MB_FREE_0R = 5
    MB_FREE_1L = 6
    MB_FREE_1R = 7
    MB_VALID = 8
    MB_Q = 9
    NUM_MBAR = 10
    BAR_MAX_0 = 1
    BAR_MAX_1 = 2
    BAR_P_0 = 3
    BAR_P_1 = 4
    BAR_FINAL = 5
    BAR_EPILOGUE_0 = 6
    BAR_EPILOGUE_1 = 7
    BAR_INDEXER_REUSE = 8
    BARRIER_PARTICIPANTS = 2 * THREADS_PER_WARPGROUP

    def __init__(
        self,
        head_dim: int,
        *,
        dtype: type[cutlass.Numeric],
        has_attn_sink: bool,
        has_topk_length: bool,
        indexer_topk: int = 0,
    ):
        """Specialize only code-shaping dtype, dimension, and optional operands."""
        if head_dim not in (512, 576):
            raise ValueError(f"SM90 DSA forward supports Dqk 512 or 576, got {head_dim}")
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError(f"SM90 DSA forward supports FP16 or BF16, got {dtype}")
        if indexer_topk not in (0, 512, 1024, 2048):
            raise ValueError(f"unsupported indexer_topk={indexer_topk}")
        self.head_dim = head_dim
        self.dtype = dtype
        self.has_attn_sink = has_attn_sink
        self.has_topk_length = has_topk_length
        self.indexer_topk = indexer_topk
        self.num_slabs = head_dim // 64
        self.qkv_elems = self.HEAD_TILE * head_dim
        self.p_elems = self.HEAD_TILE * self.TOPK_TILE

    @cute.jit
    def _issue_kv_gather_half(
        self,
        kv,
        sKV,
        token_offsets,
        validities,
        ready_mbar,
        l2_policy,
        local_tid,
        slab_begin: cutlass.Constexpr,
        slab_end: cutlass.Constexpr,
    ):
        """Issue one KV half and bind every lane's cp.async group to READY."""
        group_lane = local_tid % 8
        group = local_tid // 8
        for local_row in cutlass.range_constexpr(4):
            row = local_row * 16 + group
            live = validities[local_row] != cutlass.Uint8(0)
            for slab in cutlass.range_constexpr(slab_begin, slab_end):
                col = slab * 64 + group_lane * self.VECTOR_ELEMS
                src_off = token_offsets[local_row] + cutlass.Int64(col)
                dst = slab * self.HEAD_TILE * 64 + row * 64 + swizzle_xor_128b(row, group_lane * self.VECTOR_ELEMS)
                copy_bytes = cutlass.Int32(arith.select(live.ir_value(), cutlass.Int32(16).ir_value(), cutlass.Int32(0).ir_value()))
                inline_ptx(
                    "cp.async.cg.shared.global.L2::cache_hint.L2::256B [$0], [$1], 16, $2, $3;",
                    write_only_types=[],
                    read_only_args=[sKV.subview(dst).data_ptr().toint(), (kv.iterator + src_off).toint(), copy_bytes, l2_policy],
                )
        prims.cp_async_mbarrier_arrive(ready_mbar, noinc=True)

    @cute.jit
    def _issue_qk_range(self, accum, q_desc, k_desc, k_begin: cutlass.Constexpr, k_end: cutlass.Constexpr, clear: cutlass.Constexpr):
        """Issue one contiguous QK K-range and leave its WGMMA group in flight."""
        for k in cutlass.range_constexpr(k_begin, k_end):
            offset = (k // 4) * 512 + (k % 4) * 2
            accum = wgmma_m64nNk16_f32(
                accum,
                q_desc + offset,
                k_desc + offset,
                self.dtype,
                accumulate=not (cutlass.const_expr(clear) and cutlass.const_expr(k == k_begin)),
            )
        prims.wgmma_commit_group_sync_aligned()
        return accum

    @cute.jit
    def _online_softmax(self, scores, sValid, mi, li, alpha, floor_mi0, floor_mi1, scale_log2):
        """Update owned rows and replace score slots with FP32 probabilities."""
        lane_quad = cute.arch.thread_idx()[0] % 4
        neg_inf = -cutlass.Float32.inf
        for row_half in cutlass.range_constexpr(2):
            pairs = []
            chains = []
            for k_frag in cutlass.range_constexpr(self.QK_SLOTS // 4):
                score_off = k_frag * 4 + row_half * 2
                col0 = k_frag * 8 + lane_quad * 2
                col1 = col0 + 1
                valid0 = sValid[col0] != cutlass.Uint8(0)
                valid1 = sValid[col1] != cutlass.Uint8(0)
                s0 = scores[score_off] if valid0 else neg_inf
                s1 = scores[score_off + 1] if valid1 else neg_inf
                pairs.append((s0, s1))
                if cutlass.const_expr(k_frag < 2):
                    chains += [s0, s1]
                else:
                    chain = 2 * (k_frag % 2)
                    chains[chain] = cute.arch.fmax(chains[chain], s0)
                    chains[chain + 1] = cute.arch.fmax(chains[chain + 1], s1)

            old_mi = mi[row_half]
            current = cute.arch.fmax(cute.arch.fmax(chains[0], chains[1]), cute.arch.fmax(chains[2], chains[3])) * scale_log2
            floor_mi = floor_mi0 if cutlass.const_expr(row_half == 0) else floor_mi1
            current = cute.arch.fmax(current, floor_mi)
            current = quad_reduce(current, "max")
            old_scale = cute.math.exp2(old_mi - current, fastmath=True)
            alpha[row_half] = old_scale
            mi[row_half] = current

            total = li[row_half] * old_scale
            for k_frag in cutlass.range_constexpr(self.QK_SLOTS // 4):
                s0, s1 = pairs[k_frag]
                # The mask already replaced invalid scores with -inf, so the
                # unconditional exp2 below produces exact zero.  A finite
                # MAX_INIT keeps an all-invalid row away from inf - inf, as in
                # FlashMLA's Hopper implementation.
                p0 = cute.math.exp2(cute.math.fma(s0, scale_log2, -current), fastmath=True)
                p1 = cute.math.exp2(cute.math.fma(s1, scale_log2, -current), fastmath=True)
                total = total + p0 + p1
                scores[k_frag * 4 + row_half * 2] = p0
                scores[k_frag * 4 + row_half * 2 + 1] = p1
            li[row_half] = total

    @cute.jit
    def _pack_probabilities(self, scores, factor0, factor1):
        """Pack one QK fragment after applying its two row-half factors."""
        words = []
        for k_frag in cutlass.range_constexpr(self.QK_SLOTS // 4):
            offset = k_frag * 4
            words.append(fp32_to_fp16(scores[offset] * factor0, scores[offset + 1] * factor0, dtype=self.dtype))
            words.append(fp32_to_fp16(scores[offset + 2] * factor1, scores[offset + 3] * factor1, dtype=self.dtype))
        return cutlass.Vector.from_elements(words, cutlass.Int32)

    @cute.jit
    def _issue_pv(self, output, p, v_desc):
        """Issue one PV group into O without waiting for its completion."""
        accum = output.load(0, self.O_SLOTS)
        prims.wgmma_fence_aligned()
        for k in cutlass.range_constexpr(4):
            a = p[4 * k : 4 * k + 4] if cutlass.const_expr(isinstance(p, cutlass.Vector)) else p + 2 * k
            accum = wgmma_m64nNk16_f32(accum, a, v_desc + 128 * k, self.dtype, transpose_b=True, accumulate=True)
        prims.wgmma_commit_group_sync_aligned()
        # This is a register-to-register assignment. Callers must not read O
        # until a matching wait_group has retired this asynchronous result.
        output.store(accum)

    @cute.jit
    def _stage_output_tile(self, sO, output, inv0, inv1, local_tid, half: cutlass.Constexpr, tile: cutlass.Constexpr):
        """Normalize and stage one 64x64 O tile with stmatrix.x4."""
        values = output.load(tile * 32, 32)
        words = []
        for word in cutlass.range_constexpr(16):
            slot = 2 * word
            scale = inv0 if cutlass.const_expr(word % 2 == 0) else inv1
            words.append(fp32_to_fp16(values[slot] * scale, values[slot + 1] * scale, dtype=self.dtype))
        packed = cutlass.Vector.from_elements(words, cutlass.Int32)
        slab = half * (self.VALUE_HALF // 64) + tile
        store_fragment(packed, sO.subview(slab * self.HEAD_TILE * 64), local_tid)

    @cute.kernel
    def kernel(
        self,
        q,
        q_tma,
        kv,
        indices,
        out,
        out_tma,
        max_logits,
        lse,
        lse_indexer,
        attn_sink,
        topk_length,
        softmax_scale,
        tma_atom_q,
        tma_atom_o,
        q_smem_layout,
        o_smem_layout,
    ):
        """Run one query/head-tile CTA with a gather and two compute warpgroups."""
        sQ = cutlass.Array(self.dtype, self.qkv_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        sKV = cutlass.Array(self.dtype, 2 * self.qkv_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        # D576 reuses buffer 0's non-value RoPE slab for P0, exactly after QK
        # has consumed it.  D512 has no dead slab and needs two explicit P tiles.
        sPStorage = cutlass.Array(self.dtype, self.p_elems if self.head_dim == 576 else 2 * self.p_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        sValid = cutlass.Array(cutlass.Uint8, 2 * self.TOPK_TILE, alignment=16, space=cutlass.AddressSpace.smem)
        sMax = cutlass.Array(cutlass.Float32, self.HEAD_TILE, alignment=128, space=cutlass.AddressSpace.smem)
        sPartial = cutlass.Array(cutlass.Float32, 2 * self.HEAD_TILE, alignment=128, space=cutlass.AddressSpace.smem)
        sFinalMax = cutlass.Array(cutlass.Float32, self.HEAD_TILE, alignment=128, space=cutlass.AddressSpace.smem)
        sFinalLse = cutlass.Array(cutlass.Float32, self.HEAD_TILE, alignment=128, space=cutlass.AddressSpace.smem)
        sFinalIndexerLse = cutlass.Array(cutlass.Float32, self.HEAD_TILE, alignment=128, space=cutlass.AddressSpace.smem)
        sMbar = cutlass.Array(cutlass.Int64, self.NUM_MBAR, alignment=8, space=cutlass.AddressSpace.smem)
        mbar = sMbar.data_ptr()
        q_mbar = cute.make_ptr(
            cutlass.Int64,
            (mbar + self.MB_Q).toint(),
            cutlass.AddressSpace.smem,
            assumed_align=8,
        )

        sKV0 = sKV.subview(0)
        sKV1 = sKV.subview(self.qkv_elems)
        sQTma = cute.make_tensor(sQ.data_ptr(), q_smem_layout)
        if cutlass.const_expr(self.head_dim == 576):
            sP0 = sKV0.subview(self.HEAD_TILE * self.VALUE_DIM)
            sP1 = sPStorage
        else:
            sP0 = sPStorage.subview(self.p_elems)
            sP1 = sPStorage

        tid = cute.arch.thread_idx()[0]
        warpgroup = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4
        local_tid = tid % self.THREADS_PER_WARPGROUP
        head_tiles = cute.ceil_div(q.shape[1], self.HEAD_TILE)
        linear_block = cute.arch.block_idx()[0]
        query_idx = linear_block // head_tiles
        head_base = (linear_block - query_idx * head_tiles) * self.HEAD_TILE

        effective_topk = indices.shape[1]
        if cutlass.const_expr(self.has_topk_length):
            effective_topk = topk_length[query_idx]
        effective_topk = cute.math.max(cutlass.Int32(0), cute.math.min(cutlass.Int32(indices.shape[1]), cutlass.Int32(effective_topk)))
        # A zero-row KV carrier has no address from which even a predicated
        # cp.async may be formed. Treat every sparse slot as invalid in-kernel
        # so prepared execute keeps its single-launch contract.
        if kv.shape[0] == 0:
            effective_topk = cutlass.Int32(0)
        tile_count = cute.ceil_div(effective_topk, self.TOPK_TILE)
        pair_count = (tile_count + cutlass.Int32(1)) // cutlass.Int32(2)

        if tid == 0:
            for bar in cutlass.range_constexpr(self.NUM_MBAR - 2):
                cute.arch.mbarrier_init(mbar + bar, self.THREADS_PER_WARPGROUP)
            # One lane per eight-thread gather group publishes both tiles'
            # validity bits after all four halves have been issued.
            cute.arch.mbarrier_init(mbar + self.MB_VALID, 16)
            cute.arch.mbarrier_init(q_mbar, 1)
            cute.arch.mbarrier_init_fence()
        if warpgroup == 2:
            prims.setmaxregister(72, prims.SetMaxRegisterAction.DECREASE)
        prims.barrier_cta_sync()

        # Match the reference's one elected TMA Q load.  The producer starts
        # gathering KV concurrently while both compute warpgroups wait on Q.
        if warpgroup == 0:
            if local_tid < cutlass.Int32(32):
                cpasync.prefetch_descriptor(tma_atom_q)
                gQ = cute.local_tile(
                    q_tma,
                    (64, self.HEAD_TILE, self.num_slabs),
                    (0, head_base // self.HEAD_TILE, 0, query_idx),
                )
                load_q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_q,
                    0,
                    cute.make_layout(1),
                    gQ,
                    sQTma,
                    single_stage=True,
                )
                if prims.elect_sync():
                    cute.arch.mbarrier_arrive_and_expect_tx(q_mbar, self.qkv_elems * (self.dtype.width // 8))
                load_q(tma_bar_ptr=q_mbar, cache_policy=cutlass.Int64(self.L2_EVICT_FIRST))

        if warpgroup == 2:
            group_lane = local_tid % 8
            group = local_tid // 8
            free_phase = cutlass.Int32(1)
            l2_policy = inline_ptx(
                "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
                write_only_types=[cutlass.Int64],
                read_only_args=[],
            )
            pair = cutlass.Int32(0)
            while pair < pair_count:
                tile0 = pair * cutlass.Int32(2)
                token_offsets0 = cutlass.Array(cutlass.Int64, 4, alignment=8)
                token_offsets1 = cutlass.Array(cutlass.Int64, 4, alignment=8)
                valid0 = cutlass.Array(cutlass.Uint8, 4, alignment=4)
                valid1 = cutlass.Array(cutlass.Uint8, 4, alignment=4)
                for local_row in cutlass.range_constexpr(4):
                    row = local_row * 16 + group
                    slot0 = tile0 * self.TOPK_TILE + row
                    slot1 = (tile0 + cutlass.Int32(1)) * self.TOPK_TILE + row
                    active0 = slot0 < effective_topk
                    active1 = slot1 < effective_topk
                    token0 = cutlass.Int32(-1)
                    token1 = cutlass.Int32(-1)
                    if active0:
                        token0 = indices[query_idx, slot0]
                    if active1:
                        token1 = indices[query_idx, slot1]
                    live0 = active0 & (token0 >= cutlass.Int32(0)) & (token0 < kv.shape[0])
                    live1 = active1 & (token1 >= cutlass.Int32(0)) & (token1 < kv.shape[0])
                    safe0 = token0 if live0 else cutlass.Int32(0)
                    safe1 = token1 if live1 else cutlass.Int32(0)
                    token_offsets0[local_row] = cutlass.Int64(safe0) * cutlass.Int64(kv.stride[0])
                    token_offsets1[local_row] = cutlass.Int64(safe1) * cutlass.Int64(kv.stride[0])
                    valid0[local_row] = cutlass.Uint8(1) if live0 else cutlass.Uint8(0)
                    valid1[local_row] = cutlass.Uint8(1) if live1 else cutlass.Uint8(0)

                # The load order matches the two compute warpgroups' first QK
                # halves.  A released half may be refilled while its peer half
                # is still serving the remote PV of the preceding pair.
                cute.arch.mbarrier_wait(mbar + self.MB_FREE_0L, free_phase)
                self._issue_kv_gather_half(kv, sKV0, token_offsets0, valid0, mbar + self.MB_READY_0L, l2_policy, local_tid, 0, 4)

                cute.arch.mbarrier_wait(mbar + self.MB_FREE_1R, free_phase)
                self._issue_kv_gather_half(kv, sKV1, token_offsets1, valid1, mbar + self.MB_READY_1R, l2_policy, local_tid, 4, self.num_slabs)

                cute.arch.mbarrier_wait(mbar + self.MB_FREE_0R, free_phase)
                self._issue_kv_gather_half(kv, sKV0, token_offsets0, valid0, mbar + self.MB_READY_0R, l2_policy, local_tid, 4, self.num_slabs)

                cute.arch.mbarrier_wait(mbar + self.MB_FREE_1L, free_phase)
                self._issue_kv_gather_half(kv, sKV1, token_offsets1, valid1, mbar + self.MB_READY_1L, l2_policy, local_tid, 0, 4)

                # All four free credits above have retired the preceding
                # pair's QK/PV consumers. In particular, WG0 and WG1 have both
                # finished reading the previous validity generation before
                # these rows are overwritten.
                if group_lane == 0:
                    for local_row in cutlass.range_constexpr(4):
                        row = local_row * 16 + group
                        sValid[row] = valid0[local_row]
                        sValid[self.TOPK_TILE + row] = valid1[local_row]
                    cute.arch.mbarrier_arrive(mbar + self.MB_VALID)
                free_phase = free_phase ^ cutlass.Int32(1)
                pair = pair + cutlass.Int32(1)

        if warpgroup != 2:
            prims.setmaxregister(216, prims.SetMaxRegisterAction.INCREASE)
            cute.arch.mbarrier_wait(q_mbar, cutlass.Int32(0))
            output = cutlass.Array(cutlass.Float32, self.O_SLOTS, alignment=16)
            output.store((0.0,) * self.O_SLOTS)
            scale_log2 = softmax_scale * cutlass.Float32(math.log2(math.e))
            q_desc = wgmma_descriptor(sQ)
            scores = cutlass.Array(cutlass.Float32, self.QK_SLOTS, alignment=16)
            mi = cutlass.Array(cutlass.Float32, 2, alignment=8)
            li = cutlass.Array(cutlass.Float32, 2, alignment=8)
            alpha = cutlass.Array(cutlass.Float32, 2, alignment=8)
            indexer_mi = cutlass.Array(cutlass.Float32, 2, alignment=8)
            indexer_li = cutlass.Array(cutlass.Float32, 2, alignment=8)
            mi.store((cutlass.Float32(self.MAX_INIT), cutlass.Float32(self.MAX_INIT)))
            li.store((0.0, 0.0))
            indexer_mi.store((cutlass.Float32(self.MAX_INIT), cutlass.Float32(self.MAX_INIT)))
            indexer_li.store((0.0, 0.0))
            one = cutlass.Vector.from_elements((cutlass.Float32(1.0), cutlass.Float32(1.0)), cutlass.Float32)
            ready_phase = cutlass.Int32(0)
            pair = cutlass.Int32(0)
            while pair < pair_count:
                accum = cutlass.Vector.from_elements((cutlass.Float32(0.0),) * self.QK_SLOTS, cutlass.Float32)
                prims.wgmma_fence_aligned()
                if warpgroup == 0:
                    cute.arch.mbarrier_wait(mbar + self.MB_READY_0L, ready_phase)
                    accum = self._issue_qk_range(accum, q_desc, wgmma_descriptor(sKV0), 0, 16, True)
                    # Keep the preceding pair's remote PV in flight across the
                    # next QK-left group.  wait(1) retires only that older PV;
                    # QK-left remains live while its V half becomes reusable.
                    if pair > cutlass.Int32(0):
                        prims.wgmma_wait_group_sync_aligned(1)
                        cute.arch.mbarrier_arrive(mbar + self.MB_FREE_1L)
                    cute.arch.mbarrier_wait(mbar + self.MB_READY_0R, ready_phase)
                    accum = self._issue_qk_range(accum, q_desc, wgmma_descriptor(sKV0), 16, self.head_dim // 16, False)
                else:
                    cute.arch.mbarrier_wait(mbar + self.MB_READY_1R, ready_phase)
                    accum = self._issue_qk_range(accum, q_desc, wgmma_descriptor(sKV1), 16, self.head_dim // 16, True)
                    cute.arch.mbarrier_wait(mbar + self.MB_READY_1L, ready_phase)
                    accum = self._issue_qk_range(accum, q_desc, wgmma_descriptor(sKV1), 0, 16, False)
                prims.wgmma_wait_group_sync_aligned(0)
                scores.store(accum)
                cute.arch.mbarrier_wait(mbar + self.MB_VALID, ready_phase)

                if warpgroup == 0:
                    self._online_softmax(scores, sValid, mi, li, alpha, mi[0], mi[1], scale_log2)
                    rescale_output(output, alpha.load(0, 2))
                    # The quad leader is about to publish the next sMax
                    # generation.  Match FlashMLA's __syncwarp(): all lanes
                    # must finish consuming the current register results before
                    # the leader makes that shared generation visible to WG1.
                    cute.arch.sync_warp()
                    if local_tid % 4 == 0:
                        for row_half in cutlass.range_constexpr(2):
                            sMax[owned_row(local_tid, row_half)] = mi[row_half]
                    named_barrier_arrive(barrier_id=self.BAR_MAX_0, number_of_threads=self.BARRIER_PARTICIPANTS)

                    p_local = self._pack_probabilities(scores, one[0], one[1])
                    self._issue_pv(output, p_local, wgmma_descriptor(sKV0, mn_major=True))
                    prims.wgmma_wait_group_sync_aligned(0)
                    cute.arch.mbarrier_arrive(mbar + self.MB_FREE_0L)

                    named_barrier_sync(barrier_id=self.BAR_MAX_1, number_of_threads=self.BARRIER_PARTICIPANTS)
                    correction = cutlass.Array(cutlass.Float32, 2, alignment=8)
                    for row_half in cutlass.range_constexpr(2):
                        new_mi = sMax[owned_row(local_tid, row_half)]
                        correction[row_half] = cute.math.exp2(mi[row_half] - new_mi, fastmath=True)
                        mi[row_half] = new_mi
                        li[row_half] = li[row_half] * correction[row_half]
                    p_remote = self._pack_probabilities(scores, correction[0], correction[1])
                    store_fragment(p_remote, sP0, local_tid)
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    named_barrier_arrive(barrier_id=self.BAR_P_0, number_of_threads=self.BARRIER_PARTICIPANTS)

                    named_barrier_sync(barrier_id=self.BAR_P_1, number_of_threads=self.BARRIER_PARTICIPANTS)
                    # Match the C++ pipeline: defer O rescaling until WG1 has
                    # published P1, immediately before the remote PV issue.
                    # This overlaps the correction/P0 publication with WG1's
                    # local and remote WGMMA work instead of extending the
                    # pre-publication critical path.
                    rescale_output(output, correction.load(0, 2))
                    self._issue_pv(output, wgmma_descriptor(sP1), wgmma_descriptor(sKV1, mn_major=True))
                else:
                    named_barrier_sync(barrier_id=self.BAR_MAX_0, number_of_threads=self.BARRIER_PARTICIPANTS)
                    floor_mi = cutlass.Array(cutlass.Float32, 2, alignment=8)
                    for row_half in cutlass.range_constexpr(2):
                        floor_mi[row_half] = sMax[owned_row(local_tid, row_half)]
                    self._online_softmax(scores, sValid.subview(self.TOPK_TILE), mi, li, alpha, floor_mi[0], floor_mi[1], scale_log2)
                    rescale_output(output, alpha.load(0, 2))
                    # Every lane in a quad loaded its old floor from sMax.
                    # Keep those reads ordered before the leader overwrites the
                    # row for WG0, exactly like the C++ __syncwarp().
                    cute.arch.sync_warp()
                    if local_tid % 4 == 0:
                        for row_half in cutlass.range_constexpr(2):
                            sMax[owned_row(local_tid, row_half)] = mi[row_half]
                    named_barrier_arrive(barrier_id=self.BAR_MAX_1, number_of_threads=self.BARRIER_PARTICIPANTS)

                    p_local = self._pack_probabilities(scores, one[0], one[1])
                    self._issue_pv(
                        output,
                        p_local,
                        wgmma_descriptor(sKV1.subview(self.HEAD_TILE * self.VALUE_HALF), mn_major=True),
                    )
                    store_fragment(p_local, sP1, local_tid)

                    named_barrier_sync(barrier_id=self.BAR_P_0, number_of_threads=self.BARRIER_PARTICIPANTS)
                    self._issue_pv(
                        output,
                        wgmma_descriptor(sP0),
                        wgmma_descriptor(sKV0.subview(self.HEAD_TILE * self.VALUE_HALF), mn_major=True),
                    )
                    # Publish P1 only after the remote PV is committed.  WG0
                    # then wakes while both of WG1's PV groups are in flight,
                    # exactly as in FlashMLA's Hopper pipeline.
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    named_barrier_arrive(barrier_id=self.BAR_P_1, number_of_threads=self.BARRIER_PARTICIPANTS)
                    # Local and remote PV are consecutive groups on the same
                    # accumulator.  wait(1) makes V1 reusable while V0 remains
                    # protected until the following wait(0).
                    prims.wgmma_wait_group_sync_aligned(1)
                    cute.arch.mbarrier_arrive(mbar + self.MB_FREE_1R)
                    prims.wgmma_wait_group_sync_aligned(0)
                    cute.arch.mbarrier_arrive(mbar + self.MB_FREE_0R)

                if cutlass.const_expr(self.indexer_topk != 0):
                    if (pair + cutlass.Int32(1)) * cutlass.Int32(2 * self.TOPK_TILE) == cutlass.Int32(self.indexer_topk):
                        indexer_mi.store(mi.load(0, 2))
                        indexer_li.store(li.load(0, 2))
                ready_phase = ready_phase ^ cutlass.Int32(1)
                pair = pair + cutlass.Int32(1)

            # WG0 deliberately carries the final remote PV beyond the loop;
            # retire it before reading O for the epilogue and release V1.
            if warpgroup == 0:
                if pair_count > cutlass.Int32(0):
                    prims.wgmma_wait_group_sync_aligned(0)
                    cute.arch.mbarrier_arrive(mbar + self.MB_FREE_1L)

            partial = cutlass.Array(cutlass.Float32, 2, alignment=8)
            indexer_partial = cutlass.Array(cutlass.Float32, 2, alignment=8)
            for row_half in cutlass.range_constexpr(2):
                row = owned_row(local_tid, row_half)
                partial[row_half] = quad_reduce(li[row_half], "sum")
                if local_tid % 4 == 0:
                    sPartial[warpgroup * self.HEAD_TILE + row] = partial[row_half]
                if cutlass.const_expr(self.indexer_topk != 0):
                    indexer_partial[row_half] = quad_reduce(indexer_li[row_half], "sum")
            named_barrier_sync(barrier_id=self.BAR_FINAL, number_of_threads=self.BARRIER_PARTICIPANTS)

            inv = cutlass.Array(cutlass.Float32, 2, alignment=8)
            totals = cutlass.Array(cutlass.Float32, 2, alignment=8)
            prefix_mis = cutlass.Array(cutlass.Float32, 2, alignment=8)
            prefix_totals = cutlass.Array(cutlass.Float32, 2, alignment=8)
            for row_half in cutlass.range_constexpr(2):
                row = owned_row(local_tid, row_half)
                total = partial[row_half] + sPartial[(cutlass.Int32(1) - warpgroup) * self.HEAD_TILE + row]
                row_mi = mi[row_half]
                row_inv = cutlass.Float32(0.0)
                if total > cutlass.Float32(0.0):
                    denominator = total
                    if cutlass.const_expr(self.has_attn_sink):
                        head = head_base + row
                        sink = attn_sink[head] if head < q.shape[1] else -cutlass.Float32.inf
                        denominator = denominator + cute.math.exp2(sink * cutlass.Float32(math.log2(math.e)) - row_mi, fastmath=True)
                    row_inv = cute.math.rcp(denominator, approx=True, ftz=True)
                inv[row_half] = row_inv
                totals[row_half] = total

                if cutlass.const_expr(self.indexer_topk != 0):
                    prefix_mi = indexer_mi[row_half]
                    prefix_mis[row_half] = prefix_mi

            if cutlass.const_expr(self.indexer_topk != 0):
                # FlashMLA reuses its sL slab for the indexer-prefix reduction.
                # First fence every peer read of the full-attention partials,
                # then publish and consume the prefix generation through the
                # same slab.  This keeps the dynamic shared-memory footprint
                # byte-for-byte aligned with the C++ plan.
                named_barrier_sync(barrier_id=self.BAR_INDEXER_REUSE, number_of_threads=self.BARRIER_PARTICIPANTS)
                if local_tid % 4 == 0:
                    for row_half in cutlass.range_constexpr(2):
                        row = owned_row(local_tid, row_half)
                        sPartial[warpgroup * self.HEAD_TILE + row] = indexer_partial[row_half]
                named_barrier_sync(barrier_id=self.BAR_FINAL, number_of_threads=self.BARRIER_PARTICIPANTS)
                for row_half in cutlass.range_constexpr(2):
                    row = owned_row(local_tid, row_half)
                    prefix_total = indexer_partial[row_half] + sPartial[(cutlass.Int32(1) - warpgroup) * self.HEAD_TILE + row]
                    prefix_totals[row_half] = prefix_total

            for o_tile in cutlass.range_constexpr(self.VALUE_HALF // 64):
                if warpgroup == 0:
                    self._stage_output_tile(sQ, output, inv[0], inv[1], local_tid, half=0, tile=o_tile)
                else:
                    self._stage_output_tile(sQ, output, inv[0], inv[1], local_tid, half=1, tile=o_tile)
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                if warpgroup == 0:
                    named_barrier_sync(barrier_id=self.BAR_EPILOGUE_0, number_of_threads=self.THREADS_PER_WARPGROUP)
                else:
                    named_barrier_sync(barrier_id=self.BAR_EPILOGUE_1, number_of_threads=self.THREADS_PER_WARPGROUP)

                if local_tid < cutlass.Int32(32):
                    cpasync.prefetch_descriptor(tma_atom_o)
                    slab = warpgroup * cutlass.Int32(self.VALUE_HALF // 64) + cutlass.Int32(o_tile)
                    sOTileTma = cute.make_tensor(
                        sQ.subview(slab * self.HEAD_TILE * 64).data_ptr(),
                        o_smem_layout,
                    )
                    gO = cute.local_tile(
                        out_tma,
                        (64, self.HEAD_TILE, 1),
                        (0, head_base // self.HEAD_TILE, slab, query_idx),
                    )
                    store_o, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_o,
                        0,
                        cute.make_layout(1),
                        sOTileTma,
                        gO,
                        single_stage=True,
                    )
                    store_o()
            if local_tid < cutlass.Int32(32):
                cute.arch.cp_async_bulk_commit_group()

            if warpgroup == 1:
                # FlashMLA deliberately keeps log/LSE off the O-store critical
                # path: launch all O TMA stores first, then prepare the compact
                # statistics slabs for their bulk S2G copies.
                ln2 = cutlass.Float32(math.log(2.0))
                if local_tid % 4 == 0:
                    for row_half in cutlass.range_constexpr(2):
                        row = owned_row(local_tid, row_half)
                        total = totals[row_half]
                        row_max = cutlass.Float32(-cutlass.Float32.inf)
                        row_lse = cutlass.Float32(cutlass.Float32.inf)
                        if total > cutlass.Float32(0.0):
                            row_max = mi[row_half] * ln2
                            row_lse = row_max + cute.math.log(total, fastmath=True)
                        sFinalMax[row] = row_max
                        sFinalLse[row] = row_lse
                        if cutlass.const_expr(self.indexer_topk != 0):
                            prefix_lse = cutlass.Float32(cutlass.Float32.inf)
                            if prefix_totals[row_half] > cutlass.Float32(0.0):
                                prefix_lse = prefix_mis[row_half] * ln2 + cute.math.log(prefix_totals[row_half], fastmath=True)
                            sFinalIndexerLse[row] = prefix_lse
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                named_barrier_sync(barrier_id=self.BAR_EPILOGUE_1, number_of_threads=self.THREADS_PER_WARPGROUP)

                full_head_tile = head_base + cutlass.Int32(self.HEAD_TILE) <= q.shape[1]
                if full_head_tile:
                    if local_tid == cutlass.Int32(0):
                        stats_offset = cutlass.Int64(query_idx) * cutlass.Int64(max_logits.stride[0]) + cutlass.Int64(head_base)
                        inline_ptx(
                            "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], 256;",
                            write_only_types=[],
                            read_only_args=[(max_logits.iterator + stats_offset).toint(), sFinalMax.data_ptr().toint()],
                        )
                        inline_ptx(
                            "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], 256;",
                            write_only_types=[],
                            read_only_args=[(lse.iterator + stats_offset).toint(), sFinalLse.data_ptr().toint()],
                        )
                        if cutlass.const_expr(self.indexer_topk != 0):
                            inline_ptx(
                                "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], 256;",
                                write_only_types=[],
                                read_only_args=[(lse_indexer.iterator + stats_offset).toint(), sFinalIndexerLse.data_ptr().toint()],
                            )
                        cute.arch.cp_async_bulk_commit_group()
                else:
                    # The compiled plan is symbolic in H.  Partial head tiles
                    # cannot issue a fixed 256-byte bulk store past the caller's
                    # allocation, so retain the predicated scalar tail path.
                    if local_tid % 4 == 0:
                        for row_half in cutlass.range_constexpr(2):
                            row = owned_row(local_tid, row_half)
                            head = head_base + row
                            if head < q.shape[1]:
                                max_logits[query_idx, head] = sFinalMax[row]
                                lse[query_idx, head] = sFinalLse[row]
                                if cutlass.const_expr(self.indexer_topk != 0):
                                    lse_indexer[query_idx, head] = sFinalIndexerLse[row]

            # Every O-store issuer committed a bulk S2G group; WG1 lane 0 may
            # also have committed the full-tile statistics group.  Keep the
            # waits at the common epilogue tail so WG1 can overlap statistics
            # preparation with the O transfers, but do not let any issuer
            # exit while a transfer can still read its shared-memory source.
            if local_tid < cutlass.Int32(32):
                cute.arch.cp_async_bulk_wait_group(0, read=True)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        kv: cute.Tensor,
        indices: cute.Tensor,
        out: cute.Tensor,
        max_logits: cute.Tensor,
        lse: cute.Tensor,
        lse_indexer: Optional[cute.Tensor],
        attn_sink: Optional[cute.Tensor],
        topk_length: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Build TMA descriptors and launch the symbolic-shape SM90 kernel."""
        if cutlass.const_expr(q.element_type != self.dtype or kv.element_type != self.dtype or out.element_type != self.dtype):
            raise TypeError("q, kv, and out must match the configured element dtype")
        if cutlass.const_expr((attn_sink is not None) != self.has_attn_sink):
            raise ValueError("attn_sink presence must match the compiled SM90 plan")
        if cutlass.const_expr((topk_length is not None) != self.has_topk_length):
            raise ValueError("topk_length presence must match the compiled SM90 plan")
        if cutlass.const_expr((lse_indexer is not None) != (self.indexer_topk != 0)):
            raise ValueError("lse_indexer presence must match indexer_topk")
        q_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (64, self.HEAD_TILE, self.num_slabs),
                stride=(1, 64, self.HEAD_TILE * 64),
            ),
        )
        o_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (64, self.HEAD_TILE, 1),
                stride=(1, 64, self.HEAD_TILE * 64),
            ),
        )
        q_tma = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (64, q.shape[1], self.num_slabs, q.shape[0]),
                stride=(1, q.stride[1], 64, q.stride[0]),
            ),
        )
        tma_atom_q, q_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            q_tma,
            q_smem_layout,
            (64, self.HEAD_TILE, self.num_slabs),
        )
        out_tma = cute.make_tensor(
            out.iterator,
            cute.make_layout(
                (64, out.shape[1], self.VALUE_DIM // 64, out.shape[0]),
                stride=(1, out.stride[1], 64, out.stride[0]),
            ),
        )
        tma_atom_o, out_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            out_tma,
            o_smem_layout,
            (64, self.HEAD_TILE, 1),
        )
        head_tiles = cute.ceil_div(q.shape[1], self.HEAD_TILE)
        self.kernel(
            q,
            q_tma,
            kv,
            indices,
            out,
            out_tma,
            max_logits,
            lse,
            lse_indexer,
            attn_sink,
            topk_length,
            softmax_scale,
            tma_atom_q,
            tma_atom_o,
            q_smem_layout,
            o_smem_layout,
        ).launch(
            grid=(q.shape[0] * head_tiles, 1, 1),
            block=(self.THREADS, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
