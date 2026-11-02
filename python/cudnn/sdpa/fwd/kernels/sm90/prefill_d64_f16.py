# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native dense D64 Hopper attention, with one TMA and one compute warpgroup.

QK and PV are both m64n64k64 (four SS WGMMA k16 instructions). P occupies
one 64x64 SW128 shared tile. The compute group owns all 64 output columns;
there is no D512 padding, second PV group, P ring or cross-group row-factor
handoff. The established Hopper map builder and sign-aware online softmax
are inherited; all synchronization and MMA geometry are local to this body.

Only exact dense D64 FP16/BF16 MHA/unpacked GQA is admitted. Optional FP32
Stats, top-left/bottom-right causal masks and the three existing single-CTA
schedulers retain the common prepared-pointer contract. Hopper execution
qualification is required; cross-compilation alone is insufficient.
"""

from types import SimpleNamespace
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims

from cudnn.frost.compiled_cache import template_key
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16, MASK_CAUSAL, MASK_NONE, MASK_PADDED, MASK_SWA, SCHED_LPT_L2, SCHED_NATURAL
from cudnn.frost.tile_dsl.handles import SmemTile
from cudnn.frost.tile_dsl.mask import compute_kv_loop_bounds
from cudnn.frost.tile_dsl.scheduler import lpt_l2_tile_coords, lpt_tile_coords
from cudnn.frost.tile_dsl.tma import tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.frost.tile_dsl.wgmma import wgmma_m64nNk16_f32
from cudnn.sdpa.fwd.config_sm90 import SCALE_POSITIVE, TemplateParams, native_d64_mismatch, validate_params
from cudnn.sdpa.fwd.kernels.sm90._common_hopper import (
    FLT_MAX_F64,
    LOG2_E,
    SCHED_L2_BUDGET_BYTES,
    RolePipelineStates,
    normalize_row,
    owned_row,
    quad_reduce,
    rescale_output,
    store_fragment,
    wgmma_descriptor,
)
from cudnn.sdpa.fwd.kernels.sm90.prefill_d512_f16 import SM90FusedMultiHeadAttentionForward as _HopperForward
from cudnn.sdpa.fwd.kernels.sm90.prepared_host import host as _host, compile_host

PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
validate_params(PARAMS)
why = native_d64_mismatch(
    thd=PARAMS.thd_varlen,
    sink=PARAMS.has_sink,
    pack_gqa=PARAMS.pack_gqa,
    q_lengths=PARAMS.seq_q_lens_present,
    kv_lengths=PARAMS.seq_kv_lens_present,
    window_left=PARAMS.window_left,
    window_right=PARAMS.window_right,
)
if why is not None:
    raise ValueError(why)

PREPARED_FIXED_BATCH = True
PREPARED_WORKSPACE_ALIGNMENT = 128
PREPARED_DENSE_FLEX = True
PREPARED_FIXED_SHAPE = True
PREPARED_KV_TAIL_NATIVE = True
STORAGE_DTYPE = {DTYPE_FP16: cutlass.Float16, DTYPE_BF16: cutlass.BFloat16}[PARAMS.dtype_qkv]


class Bars(NamedTuple):
    """TMA FULL has one elected producer; EMPTY has four consumer warps."""

    mb_tma_q_full: object
    mb_tma_q_empty: object
    mb_tma_k_full: object
    mb_tma_k_empty: object
    mb_tma_v_full: object
    mb_tma_v_empty: object


class SM90FusedMultiHeadAttentionForward(_HopperForward):
    """A 64x64x64 tile; all four compute warps execute each WGMMA together."""

    HEAD_TILE = 64
    PV_HALF = 64  # Full output width; the inherited geometry helper sizes this fragment.
    P_STAGES = 1
    REG_BUDGETS = (24, 200)

    @cute.jit
    def compute_one_kv_tile(
        self, basic_params, mma_params, softmax_params, s_regs, kv_step, pipelines, active_masks: cutlass.Constexpr, is_first_kv_tile: cutlass.Constexpr
    ):
        kv_state, p_state, alpha_state = pipelines
        bars, work, tid = basic_params.bars, basic_params.work, basic_params.local_tid
        bars.mb_tma_k_full.wait(kv_state.phase)
        c = cutlass.Vector.from_elements((cutlass.Float32(0.0),) * 32, cutlass.Float32)
        prims.wgmma_fence_aligned()
        for k in cutlass.range_constexpr(4):
            c = wgmma_m64nNk16_f32(c, mma_params.q_desc + 2 * k, mma_params.k_desc + 2 * k, self.storage_dtype, accumulate=k != 0)
        prims.wgmma_commit_group_sync_aligned()
        prims.wgmma_wait_group_sync_aligned(0)
        s_regs.store(c)
        prims.bar_warp_sync(cute.arch.FULL_MASK)
        if prims.elect_sync():
            bars.mb_tma_k_empty.arrive()
        kv_seq_idx = (work.kv_right - 1 - kv_step) * self.KV_TILE
        self.online_softmax(basic_params, softmax_params, s_regs, kv_seq_idx, active_masks=active_masks, is_first_kv_tile=is_first_kv_tile)
        words = cutlass.Vector.from_elements([s_regs[2 * word] for word in range(self.qk_row_slots)], cutlass.Float32).bitcast(cutlass.Uint32)
        store_fragment(words, basic_params.sP_xfer_raw, tid)
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        # Publish every warp's P writes before SS PV, then protect P reuse until
        # all four warps have completed the previous asynchronous operand reads.
        prims.barrier_cta_sync(4, thread_count=cute.arch.THREADS_PER_WARPGROUP)
        if cutlass.const_expr(not is_first_kv_tile):
            rescale_output(mma_params.o_regs, softmax_params.alpha.load(0, 2))
        bars.mb_tma_v_full.wait(kv_state.phase)
        p_desc = wgmma_descriptor(basic_params.sP_xfer_raw)
        c = mma_params.o_regs.load(0, 32)
        prims.wgmma_fence_aligned()
        for k in cutlass.range_constexpr(4):
            c = wgmma_m64nNk16_f32(
                c, p_desc + 2 * k, mma_params.v_desc + 128 * k, self.storage_dtype, transpose_b=True, accumulate=not is_first_kv_tile or k != 0
            )
        prims.wgmma_commit_group_sync_aligned()
        prims.wgmma_wait_group_sync_aligned(0)
        mma_params.o_regs.store(c)
        prims.barrier_cta_sync(4, thread_count=cute.arch.THREADS_PER_WARPGROUP)
        if prims.elect_sync():
            bars.mb_tma_v_empty.arrive()
        return RolePipelineStates(advance(kv_state, 1), p_state, alpha_state)

    @cute.jit
    def epilogue(self, bars, work, tma_o_desc, sO, output, row_sum_inv, tid, half: cutlass.Constexpr):
        rescale_output(output, row_sum_inv)
        if work.iterations > 0:
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                bars.mb_tma_q_empty.arrive()
        # Every V read has completed before its aliased O tile is overwritten.
        prims.barrier_cta_sync(1, thread_count=cute.arch.THREADS_PER_WARPGROUP)
        words = output.load(0, 32).to(self.out_dtype).bitcast(cutlass.Uint32)
        store_fragment(words, sO.base, tid)
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        prims.barrier_cta_sync(1, thread_count=cute.arch.THREADS_PER_WARPGROUP)
        if tid < cute.arch.WARP_SIZE:
            if prims.elect_sync():
                tma_store_tile(sO, self.tma_slice(tma_o_desc, None, work.head, work.q_base, work.batch, self.tma_orders[3]))
                tma_store_commit()
                tma_store_wait(0)
        prims.barrier_cta_sync(1, thread_count=cute.arch.THREADS_PER_WARPGROUP)

    @cute.jit
    def _run_unit(
        self,
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        tma_o_desc,
        lse,
        sinks,
        seq_q_lens,
        seq_kv_lens,
        scale,
        sP_xfer_raw,
        sQ,
        sK,
        sV,
        sO,
        bars,
        wg,
        tile,
        head,
        batch,
    ):
        q_base = tile * self.Q_TILE
        q_len, kv_len = (cutlass.Int32(self.s_q), cutlass.Int32(self.s_kv))
        bounds = compute_kv_loop_bounds(
            q_base, q_len, kv_len, 0, self.mask_flags, self.KV_TILE, self.Q_TILE, bottom_right=self.kv_loop_bottom_right, window_right=0
        )
        iterations = cute.math.max(cutlass.Int32(0), bounds.right)
        work = SimpleNamespace(
            batch=batch,
            head=head,
            kv_head=head // self.qh_per_kh,
            q_base=q_base,
            q_len=q_len,
            kv_len=kv_len,
            q_offset=cutlass.Int32(0),
            kv_offset=cutlass.Int32(0),
            lse_offset=cutlass.Int32(0),
            q_output_limit=q_len,
            iterations=iterations,
            k_desc_ptr=None,
            v_desc_ptr=None,
            o_desc_ptr=None,
            kv_right=iterations,
        )
        if wg == 0:
            prims.setmaxregister(self.REG_BUDGETS[0], prims.SetMaxRegisterAction.DECREASE)
            self.tmaldg_warp_group(sQ, sK, sV, bars, tma_q_desc, tma_k_desc, tma_v_desc, work)
        else:
            prims.setmaxregister(self.REG_BUDGETS[1], prims.SetMaxRegisterAction.INCREASE)
            local_tid = cute.arch.thread_idx()[0] % cute.arch.THREADS_PER_WARPGROUP
            if work.q_base < work.q_output_limit:
                qk = cutlass.Array(cutlass.Float32, 32, alignment=16)
                output = cutlass.Array(cutlass.Float32, self.o_fragment_slots, alignment=16)
                output.store((0.0,) * self.o_fragment_slots)
                q_desc, k_desc = (wgmma_descriptor(sQ.base), wgmma_descriptor(sK.base))
                v_desc = wgmma_descriptor(sV.base, mn_major=True)
                row_sum_inv = cutlass.Array(cutlass.Float32, 2, alignment=8)
                row_sum_inv.store((0.0, 0.0))
                anchor = cutlass.Array(cutlass.Float32, 2, alignment=16)
                sentinel = cutlass.Float32(self.anchor_sentinel)
                anchor.store((sentinel, sentinel))
                partial_sum = cutlass.Array(cutlass.Float32, 2, alignment=16)
                partial_sum.store((0.0, 0.0))
                alpha = cutlass.Array(cutlass.Float32, 2, alignment=8)
                scale_log2 = scale * cutlass.Float32(LOG2_E)
                scale_log2_wide = cutlass.Float64(scale) * cutlass.Float64(LOG2_E)
                wide_uniform, wide_floor, wide_limit = (cutlass.Boolean(False), cutlass.Float32(0.0), cutlass.Float32(0.0))
                if cutlass.const_expr(self.scale_mode == SCALE_POSITIVE):
                    magnitude = cutlass.Float64(cute.math.abs(scale_log2))
                    wide_uniform = magnitude > cutlass.Float64(FLT_MAX_F64)
                    limit = cutlass.Float32(cutlass.Float64(FLT_MAX_F64) / magnitude)
                    if cutlass.Float64(limit) * magnitude > cutlass.Float64(FLT_MAX_F64):
                        limit = (limit.bitcast(cutlass.Uint32) - cutlass.Uint32(1)).bitcast(cutlass.Float32)
                    wide_floor, wide_limit = (cute.arch.fmin(limit, cutlass.Float32(2.0**126)), limit)
                softmax_params = SimpleNamespace(
                    anchor=anchor,
                    partial_sum=partial_sum,
                    alpha=alpha,
                    scale=scale,
                    scale_log2=scale_log2,
                    scale_log2_wide=scale_log2_wide,
                    wide_uniform=wide_uniform,
                    wide_floor=wide_floor,
                    wide_limit=wide_limit,
                )
                row_lse = cutlass.Array(cutlass.Float32, 2, alignment=8)
                row_lse.store((-cutlass.Float32.inf, -cutlass.Float32.inf))
                row_sum_inv_values, row_lse_values = (row_sum_inv.load(0, 2), row_lse.load(0, 2))
                if work.iterations > 0:
                    basic_params = SimpleNamespace(sP_xfer_raw=sP_xfer_raw, bars=bars, local_tid=local_tid, work=work)
                    mma_params = SimpleNamespace(o_regs=output, q_desc=q_desc, k_desc=k_desc, v_desc=v_desc)
                    pipelines = RolePipelineStates(PipelineState.start(phase=0), PipelineState.start(phase=1), PipelineState.start(phase=1))
                    bars.mb_tma_q_full.wait(cutlass.Int32(0))
                    pipelines = self.compute_one_kv_tile(
                        basic_params,
                        mma_params,
                        softmax_params,
                        qk,
                        cutlass.Int32(0),
                        pipelines,
                        active_masks=self.mask_flags | MASK_PADDED,
                        is_first_kv_tile=True,
                    )
                    right = cute.math.max(cutlass.Int32(0), work.kv_right)
                    left = right - work.iterations
                    diagonal = work.kv_len - work.q_len if cutlass.const_expr(self.bottom_right) else cutlass.Int32(0)
                    hi = cute.math.min(right, work.kv_len // self.KV_TILE)
                    if cutlass.const_expr(self.mask_flags & MASK_CAUSAL):
                        full_columns = cute.math.max(cutlass.Int32(0), work.q_base + diagonal + self.window_right + 1)
                        hi = cute.math.min(hi, full_columns // self.KV_TILE)
                    hi = cute.math.max(left, hi)
                    lo = left
                    if cutlass.const_expr(self.mask_flags & MASK_SWA):
                        first_column = cute.math.max(cutlass.Int32(0), work.q_base + self.tokens_per_tile - 1 + diagonal - self.window_left)
                        lo = cute.math.max(lo, (first_column + self.KV_TILE - 1) // self.KV_TILE)
                    lo = cute.math.min(lo, hi)
                    if cutlass.const_expr(not self.q_rows_full):
                        if work.q_base + self.tokens_per_tile > work.q_len:
                            lo = hi
                    masked_end = cute.math.min(work.iterations, cute.math.max(cutlass.Int32(1), right - hi))
                    unmasked_end = cute.math.min(work.iterations, cute.math.max(masked_end, right - lo))
                    for kv_step in cutlass.range(1, masked_end, unroll=1):
                        pipelines = self.compute_one_kv_tile(
                            basic_params, mma_params, softmax_params, qk, kv_step, pipelines, active_masks=self.mask_flags | MASK_PADDED, is_first_kv_tile=False
                        )
                    for kv_step in cutlass.range(masked_end, unmasked_end, unroll=1):
                        pipelines = self.compute_one_kv_tile(
                            basic_params, mma_params, softmax_params, qk, kv_step, pipelines, active_masks=MASK_NONE, is_first_kv_tile=False
                        )
                    for kv_step in cutlass.range(unmasked_end, work.iterations, unroll=1):
                        pipelines = self.compute_one_kv_tile(
                            basic_params, mma_params, softmax_params, qk, kv_step, pipelines, active_masks=self.mask_flags | MASK_PADDED, is_first_kv_tile=False
                        )
                    for row_half in cutlass.range_constexpr(2):
                        total = quad_reduce(softmax_params.partial_sum[row_half], "sum")
                        inv, value = normalize_row(softmax_params.anchor[row_half], softmax_params.scale, softmax_params.scale_log2, total, self.scale_mode)
                        row_sum_inv[row_half], row_lse[row_half] = (inv, value)
                    row_sum_inv_values, row_lse_values = (row_sum_inv.load(0, 2), row_lse.load(0, 2))
                if cutlass.const_expr(lse is not None):
                    if local_tid % 4 == 0:
                        for row_half in cutlass.range_constexpr(2):
                            row = owned_row(local_tid, row_half)
                            token, subhead = (row // self.heads_per_tile + work.q_base, row % self.heads_per_tile)
                            live = token < work.q_output_limit
                            if live:
                                q_head = work.head * self.heads_per_tile + subhead
                                origin = cutlass.Int64(work.batch) * cutlass.Int64(lse.stride[0])
                                lse_out = row_lse_values[row_half]
                                if cutlass.const_expr(self.stats_log2):
                                    lse_out = lse_out * cutlass.Float32(LOG2_E)
                                lse.iterator[
                                    origin + cutlass.Int64(q_head) * cutlass.Int64(lse.stride[1]) + cutlass.Int64(token) * cutlass.Int64(lse.stride[2])
                                ] = lse_out
                self.epilogue(bars, work, tma_o_desc, sO, output, row_sum_inv_values, local_tid, half=0)

    @cute.kernel
    def kernel(
        self,
        tma_q_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_o_desc: cutlass.GridConstant[cuda.TensorMap],
        lse,
        sinks,
        seq_q_lens,
        seq_kv_lens,
        scale: cutlass.Float32,
    ):
        sQ_raw = cutlass.Array(self.storage_dtype, 4096, alignment=1024, space=cutlass.AddressSpace.smem)
        sK_raw = cutlass.Array(self.storage_dtype, 4096, alignment=1024, space=cutlass.AddressSpace.smem)
        sVO_raw = cutlass.Array(self.storage_dtype, 4096, alignment=1024, space=cutlass.AddressSpace.smem)
        sP_raw = cutlass.Array(self.storage_dtype, 4096, alignment=1024, space=cutlass.AddressSpace.smem)
        bars = Bars(
            mb_tma_q_full=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem), stages=1, init_count=1, producer=Producer.TMA_LOAD
            ),
            mb_tma_q_empty=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem), stages=1, init_count=4, producer=Producer.THREAD
            ),
            mb_tma_k_full=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem), stages=1, init_count=1, producer=Producer.TMA_LOAD
            ),
            mb_tma_k_empty=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem), stages=1, init_count=4, producer=Producer.THREAD
            ),
            mb_tma_v_full=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem), stages=1, init_count=1, producer=Producer.TMA_LOAD
            ),
            mb_tma_v_empty=MBarrier(
                cutlass.Array(cutlass.Int64, 1, alignment=16, space=cutlass.AddressSpace.smem), stages=1, init_count=4, producer=Producer.THREAD
            ),
        )

        def tile(base):
            return SmemTile(
                base=base,
                elems_per_stage=4096,
                leading_byte_offset=0,
                stride_byte_offset=0,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                tma_loads_per_tile=1,
                tma_granu_elems=64,
                tma_subtile_stride_elems=4096,
            )

        sQ, sK, sV, sO = tile(sQ_raw), tile(sK_raw), tile(sVO_raw), tile(sVO_raw)
        if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 0:
            if prims.elect_sync():
                bars.mb_tma_q_full.init()
                bars.mb_tma_q_empty.init()
                bars.mb_tma_k_full.init()
                bars.mb_tma_k_empty.init()
                bars.mb_tma_v_full.init()
                bars.mb_tma_v_empty.init()
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync()
        tile_id, head, batch = cute.arch.block_idx()
        if cutlass.const_expr(self.sched_policy == SCHED_LPT_L2):
            row_bytes = 2 * self.HEAD_TILE * self.storage_dtype.bytes
            groups = SCHED_L2_BUDGET_BYTES // max(1, self.s_kv * row_bytes)
            l2_bytes = (1 << groups.bit_length() - 1) * self.s_kv * row_bytes if groups > 1 else 0
            tile_id, head, batch = lpt_l2_tile_coords(
                tile_id,
                cutlass.Int32(self.heads),
                cutlass.Int32(self.b),
                cutlass.Int32(self.q_tiles),
                self.heads // self.h_kv,
                cutlass.Int32(self.s_kv),
                row_bytes,
                l2_bytes,
            )
        elif cutlass.const_expr(self.sched_policy != SCHED_NATURAL):
            tile_id, head, batch = lpt_tile_coords(tile_id, cutlass.Int32(self.heads), cutlass.Int32(self.b), cutlass.Int32(self.q_tiles))
        wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // cute.arch.WARPS_PER_WARPGROUP
        self._run_unit(
            tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc, lse, sinks, seq_q_lens, seq_kv_lens, scale, sP_raw, sQ, sK, sV, sO, bars, wg, tile_id, head, batch
        )

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def compile(b, h_q, h_kv, s_q, s_kv, q_stride, k_stride, v_stride, o_stride, lse_stride, *, target="sm_90a", d_qk=64, d_v=64):
    """Compile one exact dense D64 pointer artifact without allocating device storage."""
    cache_key = template_key(globals(), locals(), "compile")
    if (d_qk, d_v) != (64, 64):
        raise ValueError("native SM90 D64 requires exact D_QK=D_V=64")
    why = native_d64_mismatch(bottom_right=PARAMS.causal and PARAMS.bottom_right is not False, s_q=s_q, s_kv=s_kv)
    if why is not None:
        raise ValueError(why)
    kernel = SM90FusedMultiHeadAttentionForward(
        (b, h_q, h_kv, s_q, s_kv),
        in_dtype=STORAGE_DTYPE,
        out_dtype=STORAGE_DTYPE,
        is_causal=PARAMS.causal,
        sched_policy=PARAMS.sched_policy,
        bottom_right=PARAMS.causal if PARAMS.bottom_right is None else PARAMS.bottom_right,
        qh_per_kh=PARAMS.qh_per_kh,
        scale_mode=PARAMS.scale_mode,
        stats_log2=PARAMS.stats_log2,
    )

    def order(stride, shape):
        return (3, *sorted(range(3), key=lambda mode: (stride[mode], shape[mode], mode)))

    kernel.tma_orders = tuple(
        order(stride, shape)
        for stride, shape in ((q_stride, (b, h_q, s_q)), (k_stride, (b, h_kv, s_kv)), (v_stride, (b, h_kv, s_kv)), (o_stride, (b, h_q, s_q)))
    )
    return compile_host(kernel, STORAGE_DTYPE, 64, 64, PARAMS.has_lse, False, cache_key, target)
