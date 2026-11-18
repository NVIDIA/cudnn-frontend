# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 KPage forward for blk64 metadata, with paired Q blocks.

One CTA owns Q blocks 2p (compute WG0) and 2p + 1 (compute WG1) and walks the union of their KV lists as packed
16-row pages (``kpage._build.build_pair_walk_tables_blk64``), ordered first-only, common, second-only, so a K/V tile
of common blocks is loaded once for both. A KV block with v valid rows costs 16 * ceil(v / 16) rows. A producer warp
gathers each 128-row K/V tile with one TMA box per page and 64-column plane; otherwise the schedule is the SM90 blk128
forward's warp-specialized ping-pong. WG0 computes tiles [0, ranges[2]) and WG1 tiles [ranges[3], ranges[4]); outside
its range a WG keeps the pipeline and scheduler handshakes but issues no MMA. Per-WG packed keep words mask rows that
are padding or belong only to the other Q block.
"""

from cudnn.block_sparse_attention.csrc.utils.sm90_barriers import named_barrier_sync, named_barrier_arrive

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as hopper
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator
from cudnn.block_sparse_attention.csrc.fwd.sm90_blk64.bsa_fwd_sm90 import (
    gemm_zero_acc,
    layout_acc_mn,
    make_acc_into_op,
    reduce_max,
    rescale_o_for_next_acc,
    get_final_ratio_and_lse_empty_safe,
)
from cudnn.block_sparse_attention.csrc.utils.kpage_utils import PAGE


@cute.jit
def mask_packed(qk_tiled_mma: cute.TiledMma, tSrS: cute.Tensor, tScS: cute.Tensor, keep: cute.Tensor):
    """Keep score column c iff bit c % 32 of keep word c // 32 is set."""
    tSrS_mn = cute.make_tensor(tSrS.iterator, layout_acc_mn(qk_tiled_mma, tSrS.layout))
    tScS_mn = cute.make_tensor(tScS.iterator, layout_acc_mn(qk_tiled_mma, tScS.layout))
    for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
        col = tScS_mn[0, n][1]
        word = keep[0]
        for w in cutlass.range_constexpr(1, cute.size(keep)):
            word = keep[w] if col >= 32 * w else word
        keep_col = ((cutlass.Uint32(word) >> cutlass.Uint32(col % 32)) & cutlass.Uint32(1)) != cutlass.Uint32(0)
        for m in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[0])):
            tSrS_mn[m, n] = tSrS_mn[m, n] if keep_col else -cutlass.Float32.inf


class BlockSparseAttnForwardSm90KPage:
    def __init__(self, head_dim: int = 128, nonpositive_scale: bool = False, stages: int = 2):
        assert head_dim == 128
        self.head_dim = head_dim
        self.value_dim = head_dim
        # A nonpositive scale multiplies the scores before the masks (a masked -inf times it is not -inf).
        self.nonpositive_scale = nonpositive_scale
        self.stages = stages

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # (seqlen, head_dim, heads, batch)
        k: cute.Tensor,
        v: cute.Tensor,  # (value_dim, seqlen, heads, batch)
        o: cute.Tensor,
        lse: cute.Tensor,  # (seqlen, heads, batch)
        page_table: cute.Tensor,  # [T, 8]
        keep: cute.Tensor,  # [T, 2, 4]
        ranges: cute.Tensor,  # [H, P, 5]: tile offset, tiles, WG0 hi, WG1 lo, WG1 hi
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        q = cute.make_tensor(q.iterator, cute.make_layout((q.shape[0], self.head_dim, q.shape[2], q.shape[3]), stride=q.stride))
        k = cute.make_tensor(k.iterator, cute.make_layout((k.shape[0], self.head_dim, k.shape[2], k.shape[3]), stride=k.stride))
        v = cute.make_tensor(v.iterator, cute.make_layout((self.value_dim, v.shape[1], v.shape[2], v.shape[3]), stride=v.stride))
        o = cute.make_tensor(o.iterator, cute.make_layout((o.shape[0], self.value_dim, o.shape[2], o.shape[3]), stride=o.stride))
        dtype = q.element_type
        self.dtype = dtype
        self.out_dtype = o.element_type
        q_layout = LayoutEnum.ROW_MAJOR
        v_layout = LayoutEnum.COL_MAJOR
        qa = cute.nvgpu.warpgroup.make_smem_layout_atom(hopper.get_smem_layout_atom(q_layout, dtype, self.head_dim), dtype)
        va = cute.nvgpu.warpgroup.make_smem_layout_atom(hopper.get_smem_layout_atom(v_layout, dtype, self.value_dim), dtype)
        oa = cute.nvgpu.warpgroup.make_smem_layout_atom(hopper.get_smem_layout_atom(q_layout, o.element_type, self.value_dim), o.element_type)
        sq_layout = cute.tile_to_shape(qa, (128, self.head_dim), order=(0, 1))
        sk_layout = cute.tile_to_shape(qa, (128, self.head_dim, self.stages), order=(0, 1, 2))
        sv_layout = cute.tile_to_shape(va, (self.value_dim, 128, self.stages), order=(1, 0, 2))
        so_layout = cute.tile_to_shape(oa, (128, self.value_dim), order=(0, 1))
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
        page_k = cute.make_composed_layout(sk_layout.inner, 0, cute.make_layout((PAGE, 64), stride=(64, 1)))
        page_v = cute.make_composed_layout(sv_layout.inner, 0, cute.make_layout((64, PAGE), stride=(1, 64)))
        ak, tk = cute.nvgpu.cpasync.make_tiled_tma_atom(load, k, page_k, (PAGE, 64))
        av, tv = cute.nvgpu.cpasync.make_tiled_tma_atom(load, v, page_v, (64, PAGE))
        ao, to = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), o, so_layout, (128, self.value_dim))
        self.kernel(
            tq,
            tk,
            tv,
            to,
            lse,
            page_table,
            keep,
            ranges,
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
        ).launch(
            grid=(cute.ceil_div(q.shape[0], 128), q.shape[2], q.shape[3]),
            block=(384, 1, 1),
            smem=Storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def load_tile(self, atom, src2d, stage_base, pages: cute.Tensor, barrier, k_major: cutlass.Constexpr[bool]):
        layout = cute.make_layout((64, PAGE), stride=(1, 64))
        if cutlass.const_expr(k_major):
            layout = cute.make_layout((PAGE, 64), stride=(64, 1))
        for i in cutlass.range_constexpr(8):
            page = pages[i]
            for plane in cutlass.range_constexpr(2):
                dest = cute.make_tensor(stage_base + plane * 8192 + i * PAGE * 64, layout)
                if cutlass.const_expr(k_major):
                    tile = cute.local_tile(cute.domain_offset((page * PAGE, plane * 64), src2d), (PAGE, 64), (0, 0))
                else:
                    tile = cute.local_tile(cute.domain_offset((plane * 64, page * PAGE), src2d), (64, PAGE), (0, 0))
                dst, src = cute.nvgpu.cpasync.tma_partition(atom, 0, cute.make_layout(1), cute.group_modes(dest, 0, 2), cute.group_modes(tile, 0, 2))
                cute.copy(atom, src, dst, tma_bar_ptr=barrier)

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        page_table: cute.Tensor,
        keep: cute.Tensor,
        ranges: cute.Tensor,
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
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        qb, h, b = cute.arch.block_idx()
        smem = SmemAllocator().allocate(Storage)
        sq = cute.make_tensor(cute.recast_ptr(smem.q.data_ptr(), qlayout.inner, self.dtype), qlayout.outer)
        sk = smem.k.get_tensor(klayout.outer, swizzle=klayout.inner)
        sv = smem.v.get_tensor(vlayout.outer, swizzle=vlayout.inner)
        stage_elems = 128 * self.head_dim
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
        part = (0, cute.make_layout(1))
        tq_s, tq_g = cute.nvgpu.cpasync.tma_partition(aq, *part, cute.group_modes(sq, 0, 2), cute.group_modes(gq, 0, 2))
        k2d = k[None, None, h, b]
        v2d = v[None, None, h, b]
        tile_off = ranges[h, qb, 0]
        tiles = ranges[h, qb, 1]
        if warp < 4:
            cute.arch.setmaxregister_decrease(24)
            if warp == 0:
                cute.nvgpu.cpasync.prefetch_descriptor(aq)
                cute.nvgpu.cpasync.prefetch_descriptor(ak)
                cute.nvgpu.cpasync.prefetch_descriptor(av)
                qbarr.arrive_and_expect_tx(0, 128 * self.head_dim * self.dtype.width // 8)
                cute.copy(aq, tq_g, tq_s, tma_bar_ptr=qbarr.get_barrier(0))
                state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                kpages = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Int32)
                vpages = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Int32)
                if tiles > 0:
                    # K leads V by one tile, matching QK[i+1] / PV[i].
                    for i in cutlass.range_constexpr(8):
                        kpages[i] = page_table[tile_off, i]
                    kpipe.producer_acquire(state)
                    self.load_tile(ak, k2d, sk.iterator + state.index * stage_elems, kpages, kpipe.producer_get_barrier(state), True)
                    for t in cutlass.range(1, tiles, unroll=1):
                        vstate = state.clone()
                        state.advance()
                        cute.autovec_copy(kpages, vpages)
                        for i in cutlass.range_constexpr(8):
                            kpages[i] = page_table[tile_off + t, i]
                        kpipe.producer_acquire(state)
                        self.load_tile(ak, k2d, sk.iterator + state.index * stage_elems, kpages, kpipe.producer_get_barrier(state), True)
                        vpipe.producer_acquire(vstate)
                        self.load_tile(av, v2d, sv.iterator + vstate.index * stage_elems, vpages, vpipe.producer_get_barrier(vstate), False)
                    vpipe.producer_acquire(state)
                    self.load_tile(av, v2d, sv.iterator + state.index * stage_elems, kpages, vpipe.producer_get_barrier(state), False)
                    state.advance()
                kpipe.producer_tail(state.clone())
                vpipe.producer_tail(state)
        else:
            cute.arch.setmaxregister_increase(240)
            mtid = tid - 128
            wg = cute.arch.make_warp_uniform(mtid // 128)
            lo = cutlass.Int32(0)
            hi = ranges[h, qb, 2]
            if wg == 1:
                lo = ranges[h, qb, 3]
                hi = ranges[h, qb, 4]
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
            rp = make_acc_into_op(scores, pv.tv_layout_A, self.dtype)
            kw = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
            ratio = cute.make_rmem_tensor_like(mx)
            pv.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            qbarr.wait(0, 0)
            kstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            vstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            if wg == 1:
                named_barrier_arrive(barrier_id=2, number_of_threads=256)
            # Slot t carries QK[t] and PV[t-1]; slot `tiles` only drains PV. Every slot runs the pipeline and scheduler
            # handshakes in both WGs, but the MMAs stay out of per-slot branches (a wgmma under a runtime condition makes
            # ptxas serialize it): the WG's range [lo, hi) runs as a QK-only slot, an unconditional steady loop and a
            # PV-only slot.
            if hi > lo:
                for t in cutlass.range(0, lo, unroll=1):
                    self.slot_begin(t, tiles, wg, kpipe, vpipe, kstate, vstate)
                    named_barrier_arrive(barrier_id=3 - wg, number_of_threads=256)
                    if t < tiles:
                        kpipe.consumer_release(kstate)
                        kstate.advance()
                    if t > 0:
                        vpipe.consumer_release(vstate)
                        vstate.advance()
                # slot lo: QK only
                self.read_keep(keep, tile_off + lo, wg, kw)
                self.slot_begin(lo, tiles, wg, kpipe, vpipe, kstate, vstate)
                cute.nvgpu.warpgroup.fence()
                gemm_zero_acc(qk, ra, rb[None, None, None, kstate.index], scores)
                cute.nvgpu.warpgroup.commit_group()
                named_barrier_arrive(barrier_id=3 - wg, number_of_threads=256)
                cute.nvgpu.warpgroup.wait_group(0)
                kpipe.consumer_release(kstate)
                kstate.advance()
                self.softmax(qk, scores, coords, mx, den, kw, scale, ratio)
                if lo > 0:
                    vpipe.consumer_release(vstate)
                    vstate.advance()
                cute.make_tensor(rp.iterator, scores.layout).store(scores.load().to(self.dtype))
                # steady state: QK[t] and PV[t-1]
                for t in cutlass.range(lo + 1, hi, unroll=1):
                    self.read_keep(keep, tile_off + t, wg, kw)
                    if wg == 0:
                        kpipe.consumer_wait(kstate)
                        vpipe.consumer_wait(vstate)
                    named_barrier_sync(barrier_id=2 + wg, number_of_threads=256)
                    cute.nvgpu.warpgroup.fence()
                    gemm_zero_acc(qk, ra, rb[None, None, None, kstate.index], scores)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.gemm(pv, out, rp, rv[None, None, None, vstate.index], out)
                    cute.nvgpu.warpgroup.commit_group()
                    named_barrier_arrive(barrier_id=3 - wg, number_of_threads=256)
                    cute.nvgpu.warpgroup.wait_group(1)
                    kpipe.consumer_release(kstate)
                    kstate.advance()
                    self.softmax(qk, scores, coords, mx, den, kw, scale, ratio)
                    cute.nvgpu.warpgroup.wait_group(0)
                    vpipe.consumer_release(vstate)
                    vstate.advance()
                    cute.make_tensor(rp.iterator, scores.layout).store(scores.load().to(self.dtype))
                    rescale_o_for_next_acc(pv, out, ratio)
                # slot hi: PV[hi - 1] only
                self.slot_begin(hi, tiles, wg, kpipe, vpipe, kstate, vstate)
                cute.nvgpu.warpgroup.fence()
                cute.gemm(pv, out, rp, rv[None, None, None, vstate.index], out)
                cute.nvgpu.warpgroup.commit_group()
                named_barrier_arrive(barrier_id=3 - wg, number_of_threads=256)
                if hi < tiles:
                    kpipe.consumer_release(kstate)
                    kstate.advance()
                cute.nvgpu.warpgroup.wait_group(0)
                vpipe.consumer_release(vstate)
                vstate.advance()
                for t in cutlass.range(hi + 1, tiles + 1, unroll=1):
                    self.slot_begin(t, tiles, wg, kpipe, vpipe, kstate, vstate)
                    named_barrier_arrive(barrier_id=3 - wg, number_of_threads=256)
                    if t < tiles:
                        kpipe.consumer_release(kstate)
                        kstate.advance()
                    if t > 0:
                        vpipe.consumer_release(vstate)
                        vstate.advance()
            else:
                for t in cutlass.range(0, tiles + 1, unroll=1):
                    self.slot_begin(t, tiles, wg, kpipe, vpipe, kstate, vstate)
                    named_barrier_arrive(barrier_id=3 - wg, number_of_threads=256)
                    if t < tiles:
                        kpipe.consumer_release(kstate)
                        kstate.advance()
                    if t > 0:
                        vpipe.consumer_release(vstate)
                        vstate.advance()
            final_ratio, logsum = get_final_ratio_and_lse_empty_safe(mx, den, cutlass.Float32(1.0) if cutlass.const_expr(self.nonpositive_scale) else scale)
            rescale_o_for_next_acc(pv, out, final_ratio)
            cmn = cute.make_tensor(coords.iterator, layout_acc_mn(qk, coords.layout))
            if cute.arch.lane_idx() % 4 == 0:
                for r in cutlass.range_constexpr(cute.size(logsum)):
                    row = qb * 128 + cmn[r, 0][0]
                    if row < q.shape[0]:
                        lse[row, h, b] = logsum[r]
            named_barrier_sync(barrier_id=1, number_of_threads=256)
            so = cute.make_tensor(cute.recast_ptr(smem.q.data_ptr(), olayout.inner, self.out_dtype), olayout.outer)
            go = cute.local_tile(o[None, None, h, b], (128, self.value_dim), (qb, 0))
            out_cvt = cute.make_rmem_tensor_like(out, self.out_dtype)
            out_cvt.store(out.load().to(self.out_dtype))
            copy_o = cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4), self.out_dtype), pv)
            cute.copy(copy_o, copy_o.retile(out_cvt), copy_o.get_slice(mtid).partition_D(so))
            cute.arch.fence_proxy("async.shared", space="cta")
            named_barrier_sync(barrier_id=1, number_of_threads=256)
            if warp == 4:
                os, og = cute.nvgpu.cpasync.tma_partition(ao, *part, cute.group_modes(so, 0, 2), cute.group_modes(go, 0, 2))
                cute.copy(ao, os, og)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def read_keep(self, keep, row, wg, kw):
        for i in cutlass.range_constexpr(4):
            kw[i] = keep[row, wg, i]

    @cute.jit
    def slot_begin(self, t, tiles, wg, kpipe, vpipe, kstate, vstate):
        """WG0 waits for slot t's K (t < tiles) and V[t - 1] (t > 0) before handing the slot token on.

        Pipeline states are only read here: their advance() stays in the caller's loop body, where the DSL sees the
        loop-carried update.
        """
        if wg == 0:
            if t < tiles:
                kpipe.consumer_wait(kstate)
            if t > 0:
                vpipe.consumer_wait(vstate)
        named_barrier_sync(barrier_id=2 + wg, number_of_threads=256)

    @cute.jit
    def softmax(self, qk, scores, coords, mx, den, kw, scale, ratio):
        """Online softmax of one tile; the first computed tile sees mx = -inf and gets ratio 0 (den and O are 0)."""
        if cutlass.const_expr(self.nonpositive_scale):
            scores.store(scores.load() * scale)
            scale = cutlass.Float32(1.0)
        full = (kw[0] & kw[1] & kw[2] & kw[3]) == cutlass.Int32(-1)
        if not full:
            mask_packed(qk, scores, coords, kw)
        scores_mn = cute.make_tensor(scores.iterator, layout_acc_mn(qk, scores.layout))
        previous_max = cute.make_rmem_tensor_like(mx)
        previous_max.store(mx.load())
        reduce_max(scores_mn, mx)
        for m in cutlass.range_constexpr(cute.size(mx)):
            current_max = mx[m]
            current_max = 0.0 if current_max == -cutlass.Float32.inf else current_max
            ratio[m] = cute.math.exp2((previous_max[m] - current_max) * scale, fastmath=True)
            den[m] *= ratio[m]
            scaled_max = current_max * scale
            for n in cutlass.range_constexpr(cute.size(scores_mn, mode=[1])):
                scores_mn[m, n] = cute.math.exp2(cute.math.fma(scores_mn[m, n], scale, -scaled_max), fastmath=True)
            for n in cutlass.range_constexpr(cute.size(scores_mn, mode=[1])):
                den[m] += scores_mn[m, n]
