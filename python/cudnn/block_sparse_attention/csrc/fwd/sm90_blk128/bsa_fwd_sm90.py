# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Hopper Q128/KV128 attention with shared KV and warp specialization."""

from cudnn.block_sparse_attention.csrc.utils.sm90_barriers import named_barrier_sync, named_barrier_arrive

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as hopper
from cudnn.block_sparse_attention.csrc.fwd.sm90_blk64.bsa_fwd_sm90 import (
    gemm_zero_acc,
    mask,
    layout_acc_mn,
    make_acc_into_op,
    reduce_max,
    rescale_o_for_next_acc,
    get_final_ratio_and_lse_empty_safe,
)


class BlockSparseAttnForwardSm90Blk128:
    def __init__(
        self, head_dim, value_dim, gqa_ratio=1, variable_count=False, block_sizes_mode=0, num_splits=1, stages=2, scale_nonpositive=False, has_kv_tail=False
    ):
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.gqa_ratio = gqa_ratio
        self.variable_count = variable_count
        self.has_kv_tail = has_kv_tail
        self.block_sizes_mode = block_sizes_mode
        self.num_splits = num_splits
        self.stages = stages
        self.scale_nonpositive = scale_nonpositive

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        indices: cute.Tensor,
        counts: cute.Tensor,
        sizes: cute.Tensor,
        scale: cutlass.Float32,
        fixed_count: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        dtype = q.element_type
        self.dtype = dtype
        self.out_dtype = o.element_type
        q_layout = utils.LayoutEnum.ROW_MAJOR
        v_layout = utils.LayoutEnum.COL_MAJOR
        qa = cute.nvgpu.warpgroup.make_smem_layout_atom(hopper.get_smem_layout_atom(q_layout, dtype, self.head_dim), dtype)
        va = cute.nvgpu.warpgroup.make_smem_layout_atom(hopper.get_smem_layout_atom(v_layout, dtype, self.value_dim), dtype)
        oa = cute.nvgpu.warpgroup.make_smem_layout_atom(hopper.get_smem_layout_atom(q_layout, o.element_type, self.value_dim), o.element_type)
        sq_layout = cute.tile_to_shape(qa, (128, self.head_dim), order=(0, 1))
        sk_layout = cute.tile_to_shape(qa, (128, self.head_dim, self.stages), order=(0, 1, 2))
        sv_layout = cute.tile_to_shape(va, (self.value_dim, 128, self.stages), order=(1, 0, 2))
        so_layout = cute.tile_to_shape(oa, (128, self.value_dim), order=(0, 1))
        # Q and the final output have disjoint lifetimes, including split FP32 output.
        q_bytes = max(cute.cosize(sq_layout) * dtype.width // 8, cute.cosize(so_layout) * o.element_type.width // 8)

        @cute.struct
        class Storage:
            q_bar: cute.struct.MemRange[cutlass.Int64, 1]
            k_bar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            v_bar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            q: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, q_bytes], 128]
            k: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sk_layout)], 128]
            v: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sv_layout)], 128]

        qk = hopper.make_trivial_tiled_mma(
            dtype, dtype, cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K, cutlass.Float32, (2, 1, 1), tiler_mn=(64, 128)
        )
        pv = hopper.make_trivial_tiled_mma(
            dtype,
            dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32,
            (2, 1, 1),
            tiler_mn=(64, self.value_dim),
            a_source=cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        aq, tq = cute.nvgpu.cpasync.make_tiled_tma_atom(load, q, sq_layout, (128, self.head_dim))
        ak, tk = cute.nvgpu.cpasync.make_tiled_tma_atom(load, k, cute.select(sk_layout, mode=[0, 1]), (128, self.head_dim))
        av, tv = cute.nvgpu.cpasync.make_tiled_tma_atom(load, v, cute.select(sv_layout, mode=[0, 1]), (self.value_dim, 128))
        ao, to = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), o, so_layout, (128, self.value_dim))
        self.kernel(
            tq,
            tk,
            tv,
            to,
            lse,
            indices,
            counts,
            sizes,
            aq,
            ak,
            av,
            ao,
            qk,
            pv,
            sq_layout,
            sk_layout,
            sv_layout,
            so_layout,
            Storage,
            scale * 1.4426950408889634,
            fixed_count,
        ).launch(
            grid=(cute.ceil_div(q.shape[0], 128), self.num_splits, q.shape[2] * q.shape[3]),
            block=(384, 1, 1),
            smem=Storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        indices: cute.Tensor,
        counts: cute.Tensor,
        sizes: cute.Tensor,
        aq: cute.CopyAtom,
        ak: cute.CopyAtom,
        av: cute.CopyAtom,
        ao: cute.CopyAtom,
        qk: cute.TiledMma,
        pv: cute.TiledMma,
        qlayout: cute.ComposedLayout,
        klayout: cute.ComposedLayout,
        vlayout: cute.ComposedLayout,
        olayout: cute.ComposedLayout,
        Storage: cutlass.Constexpr,
        scale: cutlass.Float32,
        fixed_count: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        qb, split, bh = cute.arch.block_idx()
        h = bh % q.shape[2]
        b = bh // q.shape[2]
        kh = h // self.gqa_ratio
        oh = split * q.shape[2] + h
        smem = utils.SmemAllocator().allocate(Storage)
        sq = cute.make_tensor(cute.recast_ptr(smem.q.data_ptr(), qlayout.inner, self.dtype), qlayout.outer)
        sk = smem.k.get_tensor(klayout.outer, swizzle=klayout.inner)
        sv = smem.v.get_tensor(vlayout.outer, swizzle=vlayout.inner)
        qbarr = pipeline.MbarrierArray(
            smem.q_bar.data_ptr(), num_stages=1, agent=(pipeline.PipelineOp.TmaLoad, pipeline.CooperativeGroup(pipeline.Agent.Thread))
        )
        kpipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=smem.k_bar.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
            tx_count=128 * self.head_dim * self.dtype.width // 8,
        )
        vpipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=smem.v_bar.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
            tx_count=128 * self.value_dim * self.dtype.width // 8,
        )
        gq = cute.local_tile(q[None, None, h, b], (128, self.head_dim), (qb, 0))
        gk = cute.local_tile(k[None, None, kh, b], (128, self.head_dim), (None, 0))
        gv = cute.local_tile(v[None, None, kh, b], (self.value_dim, 128), (0, None))
        part = (0, cute.make_layout(1))
        tq_s, tq_g = cute.nvgpu.cpasync.tma_partition(aq, *part, cute.group_modes(sq, 0, 2), cute.group_modes(gq, 0, 2))
        tk_s, tk_g = cute.nvgpu.cpasync.tma_partition(ak, *part, cute.group_modes(sk, 0, 2), cute.group_modes(gk, 0, 2))
        tv_s, tv_g = cute.nvgpu.cpasync.tma_partition(av, *part, cute.group_modes(sv, 0, 2), cute.group_modes(gv, 0, 2))
        if cutlass.const_expr(self.variable_count):
            count = counts[qb, h, b]
        elif cutlass.const_expr(self.stages == 1 and self.num_splits == 1):
            # A positive fixed count can use one stage without splitting only
            # when it is one; avoid emitting an unreachable steady-state loop.
            count = cutlass.Int32(1)
        else:
            count = fixed_count
        if cutlass.const_expr(self.num_splits == 1):
            begin, end = 0, count
        else:
            begin = count * split // self.num_splits
            end = count * (split + 1) // self.num_splits
        # The public fixed-count contract requires count > 0. Keep this fact
        # visible when count is a runtime scalar, without specializing its value.
        has_blocks = True if cutlass.const_expr(not self.variable_count and self.num_splits == 1) else begin < end
        if warp < 4:
            cute.arch.setmaxregister_decrease(24)
            if warp == 0:
                cute.nvgpu.cpasync.prefetch_descriptor(aq)
                cute.nvgpu.cpasync.prefetch_descriptor(ak)
                cute.nvgpu.cpasync.prefetch_descriptor(av)
                qbarr.arrive_and_expect_tx(0, 128 * self.head_dim * self.dtype.width // 8)
                cute.copy(aq, tq_g, tq_s, tma_bar_ptr=qbarr.get_barrier(0))
                state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                if begin < end:
                    # K leads V by one list position, matching QK[i+1] / PV[i].
                    previous_block = indices[begin, qb, h, b]
                    kpipe.producer_acquire(state)
                    cute.copy(ak, tk_g[None, previous_block], tk_s[None, state.index], tma_bar_ptr=kpipe.producer_get_barrier(state))
                    for slot in cutlass.range(begin + 1, end, unroll=1):
                        vstate = state.clone()
                        state.advance()
                        block = indices[slot, qb, h, b]
                        kpipe.producer_acquire(state)
                        cute.copy(ak, tk_g[None, block], tk_s[None, state.index], tma_bar_ptr=kpipe.producer_get_barrier(state))
                        vpipe.producer_acquire(vstate)
                        cute.copy(av, tv_g[None, previous_block], tv_s[None, vstate.index], tma_bar_ptr=vpipe.producer_get_barrier(vstate))
                        previous_block = block
                    vpipe.producer_acquire(state)
                    cute.copy(av, tv_g[None, previous_block], tv_s[None, state.index], tma_bar_ptr=vpipe.producer_get_barrier(state))
                    state.advance()
                # producer_tail advances its state through a full ring.
                kpipe.producer_tail(state.clone())
                vpipe.producer_tail(state)
        else:
            cute.arch.setmaxregister_increase(240)
            mtid = tid - 128
            qm = qk.get_slice(mtid)
            pm = pv.get_slice(mtid)
            ra = qk.make_fragment_A(qm.partition_A(sq))
            rb = qk.make_fragment_B(qm.partition_B(sk))
            rv = pv.make_fragment_B(pm.partition_B(sv))
            scores = cute.make_rmem_tensor(qm.partition_shape_C((128, 128)), cutlass.Float32)
            out = cute.make_rmem_tensor(pm.partition_shape_C((128, self.value_dim)), cutlass.Float32)
            coords = qm.partition_C(cute.make_identity_tensor((128, 128)))
            stat_layout = cute.make_layout(cute.size(layout_acc_mn(pv, out.layout), mode=[0]))
            mx = cute.make_rmem_tensor(stat_layout, cutlass.Float32)
            den = cute.make_rmem_tensor(stat_layout, cutlass.Float32)
            mx.fill(-cutlass.Float32.inf)
            den.fill(0.0)
            out.fill(0.0)
            qbarr.wait(0, 0)
            kstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            vstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            softmax_scale = cutlass.Float32(1.0) if cutlass.const_expr(self.scale_nonpositive) else scale
            if has_blocks:
                # Prologue: P[0] is ready before the first overlapped QK/PV pair.
                kpipe.consumer_wait(kstate)
                cute.nvgpu.warpgroup.fence()
                gemm_zero_acc(qk, ra, rb[None, None, None, kstate.index], scores)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                kpipe.consumer_release(kstate)
                kstate.advance()
                self.softmax(qk, scores, coords, mx, den, indices, sizes, begin, qb, h, b, k.shape[0], scale, softmax_scale, is_first=True)
                rp = make_acc_into_op(scores, pv.tv_layout_A, self.dtype)

                # The second compute WG seeds the first WG's scheduler token.
                # IDs 2/3 are independent of the shared-output barrier (ID 1).
                wg = cute.arch.make_warp_uniform((tid - 128) // 128)
                if wg == 1:
                    named_barrier_arrive(barrier_id=2, number_of_threads=256)
                for slot in cutlass.range(begin + 1, end, unroll=1):
                    # WG0 publishes K/V readiness with its scheduler arrive.
                    # WG1 waits on that token before using either shared tile.
                    if wg == 0:
                        kpipe.consumer_wait(kstate)
                    named_barrier_sync(barrier_id=2 + wg, number_of_threads=256)
                    cute.nvgpu.warpgroup.fence()
                    gemm_zero_acc(qk, ra, rb[None, None, None, kstate.index], scores)
                    cute.nvgpu.warpgroup.commit_group()
                    if wg == 0:
                        vpipe.consumer_wait(vstate)
                    pv.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.gemm(pv, out, rp, rv[None, None, None, vstate.index], out)
                    cute.nvgpu.warpgroup.commit_group()
                    # Hand the tensor-core issue slot to the other compute WG.
                    named_barrier_arrive(barrier_id=3 - wg, number_of_threads=256)
                    # QK is complete; PV may still use rp, V and out during softmax.
                    cute.nvgpu.warpgroup.wait_group(1)
                    kpipe.consumer_release(kstate)
                    kstate.advance()
                    ratio = self.softmax(qk, scores, coords, mx, den, indices, sizes, slot, qb, h, b, k.shape[0], scale, softmax_scale)
                    cute.nvgpu.warpgroup.wait_group(0)
                    vpipe.consumer_release(vstate)
                    vstate.advance()
                    # Do not overwrite P or rescale O until the preceding PV retires.
                    cute.make_tensor(rp.iterator, scores.layout).store(scores.load().to(self.dtype))
                    rescale_o_for_next_acc(pv, out, ratio)

                # Drain the last P/V pair; an empty list never acquires either ring.
                vpipe.consumer_wait(vstate)
                cute.nvgpu.warpgroup.fence()
                pv.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.gemm(pv, out, rp, rv[None, None, None, vstate.index], out)
                cute.nvgpu.warpgroup.commit_group()
            # Final statistics do not read O and can overlap the last PV.
            ratio, logsum = get_final_ratio_and_lse_empty_safe(mx, den, softmax_scale)
            if has_blocks:
                cute.nvgpu.warpgroup.wait_group(0)
                vpipe.consumer_release(vstate)
            rescale_o_for_next_acc(pv, out, ratio)
            cmn = cute.make_tensor(coords.iterator, layout_acc_mn(qk, coords.layout))
            if cute.arch.lane_idx() % 4 == 0:
                for r in cutlass.range_constexpr(cute.size(logsum)):
                    row = qb * 128 + cmn[r, 0][0]
                    if row < q.shape[0]:
                        lse[row, oh, b] = logsum[r]
            # Both compute WGs must finish reading Q before any warp reuses it for O.
            named_barrier_sync(barrier_id=1, number_of_threads=256)
            so = cute.make_tensor(cute.recast_ptr(smem.q.data_ptr(), olayout.inner, self.out_dtype), olayout.outer)
            go = cute.local_tile(o[None, None, oh, b], (128, self.value_dim), (qb, 0))
            out_cvt = cute.make_rmem_tensor_like(out, self.out_dtype)
            out_cvt.store(out.load().to(self.out_dtype))
            if cutlass.const_expr(self.out_dtype.width == 16):
                store_o = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4), self.out_dtype)
            else:
                # Split-KV partials are FP32, which stmatrix.b16 cannot represent.
                store_o = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.out_dtype, num_bits_per_copy=32)
            copy_o = cute.make_tiled_copy_C(store_o, pv)
            cute.copy(copy_o, copy_o.retile(out_cvt), copy_o.get_slice(mtid).partition_D(so))
            cute.arch.fence_proxy("async.shared", space="cta")
            named_barrier_sync(barrier_id=1, number_of_threads=256)
            if warp == 4:
                os, og = cute.nvgpu.cpasync.tma_partition(ao, *part, cute.group_modes(so, 0, 2), cute.group_modes(go, 0, 2))
                cute.copy(ao, os, og)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def softmax(self, qk, scores, coords, mx, den, indices, sizes, slot, qb, h, b, seqlen_k, scale, softmax_scale, is_first: cutlass.Constexpr = False):
        if cutlass.const_expr(self.scale_nonpositive):
            scores.store(scores.load() * scale)
        if cutlass.const_expr(self.block_sizes_mode != 0 or self.has_kv_tail):
            block = indices[slot, qb, h, b]
            valid = cutlass.min(128, seqlen_k - block * 128)
            if cutlass.const_expr(self.block_sizes_mode == 1):
                valid = cutlass.min(valid, sizes[block])
            elif cutlass.const_expr(self.block_sizes_mode == 2):
                valid = cutlass.min(valid, sizes[b, block])
            elif cutlass.const_expr(self.block_sizes_mode == 3):
                valid = cutlass.min(valid, sizes[b, h, block])
            mask(qk, scores, coords, valid)
        scores_mn = cute.make_tensor(scores.iterator, layout_acc_mn(qk, scores.layout))
        previous_max = cute.make_rmem_tensor_like(mx)
        previous_max.store(mx.load())
        reduce_max(scores_mn, mx)
        ratio = cute.make_rmem_tensor_like(mx)
        for m in cutlass.range_constexpr(cute.size(mx)):
            current_max = mx[m]
            if cutlass.const_expr(self.block_sizes_mode != 0 or self.has_kv_tail):
                # Masked blocks may contain no valid tokens, even in the prologue.
                current_max = 0.0 if current_max == -cutlass.Float32.inf else current_max
            if cutlass.const_expr(is_first):
                ratio[m] = 1.0
            else:
                ratio[m] = cute.math.exp2((previous_max[m] - current_max) * softmax_scale, fastmath=True)
                den[m] *= ratio[m]
            scaled_max = current_max * softmax_scale
            for n in cutlass.range_constexpr(cute.size(scores_mn, mode=[1])):
                scores_mn[m, n] = cute.math.exp2(cute.math.fma(scores_mn[m, n], softmax_scale, -scaled_max), fastmath=True)
            for n in cutlass.range_constexpr(cute.size(scores_mn, mode=[1])):
                if cutlass.const_expr(is_first and n == 0):
                    den[m] = scores_mn[m, n]
                else:
                    den[m] += scores_mn[m, n]
        return ratio
