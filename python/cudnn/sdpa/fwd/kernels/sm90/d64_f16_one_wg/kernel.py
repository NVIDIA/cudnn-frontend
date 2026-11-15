# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Methods copied from prefill_d512_f16.py retain its MIT terms; the D64-specific
# implementation remains Apache-2.0. See LICENSING.md and THIRD_PARTY_LICENSES.txt.
"""Native dense D64 Hopper attention, with one TMA and one compute warpgroup.

QK and PV are both m64n64k64 (four SS WGMMA k16 instructions). P occupies
one 64x64 SW128 shared tile. The compute group owns all 64 output columns;
there is no D512 padding, second PV group, P ring or cross-group row-factor
handoff. The established Hopper map builder and sign-aware online softmax
are local; all synchronization and MMA geometry are local to this body.

Only exact dense D64 FP16/BF16 MHA/unpacked GQA is admitted. Optional FP32
Stats, top-left/bottom-right causal masks and the three existing single-CTA
schedulers retain the common prepared-pointer contract. Hopper execution
qualification is required; cross-compilation alone is insufficient.
"""

from types import SimpleNamespace
from typing import NamedTuple, Optional, Type
import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16, MASK_CAUSAL, MASK_NONE, MASK_PADDED, MASK_SWA
from cudnn.frost.tile_dsl.handles import GmemTileTmaSlice, SmemTile
from cudnn.frost.tile_dsl.mask import compute_kv_loop_bounds
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16, opaque_f32_zero
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat
from cudnn.frost.tile_dsl.scheduler import SCHED_LPT_L2, SCHED_NATURAL, lpt_l2_tile_coords, lpt_tile_coords
from cudnn.frost.tile_dsl.thd import TENSOR_MAP_QWORDS, THD_MAPS_OFF
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.sdpa.fwd.config_sm100 import pack_gqa_group_size
from cudnn.sdpa.fwd.config_sm90 import SCALE_POSITIVE, SCALE_ZERO, thd_meta_words, thd_origin_maps, thd_origins_off
from cudnn.sdpa.fwd.kernels.sm90.prepared_host import host as _host, compile_host
from cudnn.sdpa.fwd.kernels.sm90.thd_origins import build_thd_origin_setup_kernel
from cudnn.sdpa.fwd.kernels.thd_helpers import THD_SETUP_THREADS, build_thd_meta_o_descs_kernel
from cudnn.sdpa.fwd.kernels.sm90._common_hopper import (
    FLT_MAX_F64,
    LOG2_E,
    SCHED_L2_BUDGET_BYTES,
    RolePipelineStates,
    normalize_row,
    owned_row,
    quad_reduce,
    rescale_output,
    score_prefix_mask,
    store_fragment,
    wgmma_descriptor,
)
from cudnn.frost.compiled_cache import template_key
from cudnn.frost.tile_dsl.wgmma import wgmma_m64nNk16_f32
from cudnn.sdpa.fwd.kernels.sm90.d64_f16_one_wg.contract import TEMPLATE_NAME
from cudnn.sdpa.fwd.kernels.sm90.d64_f16_one_wg.config import TemplateParams, from_shared, native_d64_mismatch, validate_params

PARAMS: TemplateParams = from_shared(globals().get("FROST_TEMPLATE_PARAMS", TemplateParams()))
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


class SM90FusedMultiHeadAttentionForward:
    """A 64x64x64 tile; all four compute warps execute each WGMMA together."""

    Q_TILE = 64
    KV_TILE = 64
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
        block_table=None,
        block_table_v=None,
    ):
        if cutlass.const_expr(block_table is not None or block_table_v is not None):
            raise ValueError("native SM90 D64 paging uses the D128 body")
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

    def __init__(
        self,
        shape: tuple[int, int, int, int, int],
        in_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        out_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        is_causal: bool = False,
        sched_policy: int = SCHED_NATURAL,
        bottom_right: bool = False,
        window_size_left: int | None = None,
        window_size_right: int | None = None,
        seq_q_lens_present: bool = False,
        seq_kv_lens_present: bool = False,
        has_sink: bool = False,
        thd_varlen: bool = False,
        pack_gqa: bool = False,
        qh_per_kh: int = 1,
        scale_mode: int = SCALE_POSITIVE,
        stats_log2: bool = False,
        thd_origins: tuple | None = None,
        split_kv: int = 1,
        page_size: int = 0,
    ):
        """Initialize the SM90 D512 FMHA prefill kernel configuration.

        :param page_size: Dense-Q paged KV specialization; zero means ordinary
            dense or THD KV. Page sizes divide 64 or are multiples of 64, with
            an eight-row minimum for complete SW128 atoms.
        :param shape: The declared problem ``(b, h_q, h_kv, s_q, s_kv)``, fixed at plan
            time. ``b``, ``h_q`` and ``s_q`` size the grid, and ``s_q``/``s_kv`` are the
            dense lengths that clamp device lengths. Under THD, ``b`` is the sequence
            count and ``s_q``/``s_kv`` are unread: ``thd_max_sq`` sizes the grid at
            launch.
        :param in_dtype: Q/K/V element type (Float16 or BFloat16). It types the SMEM
            tiles and sets ``swizzle_chunk_elems``, the SW128 slab width in columns.
        :param out_dtype: O element type. Must match ``in_dtype``: V and O alias one
            SMEM region.
        :param is_causal: Apply the causal upper bound to QK (``MASK_CAUSAL``).
        :param sched_policy: CTA order over the (Q tile, head, batch) tiles.
            ``SCHED_NATURAL`` launches the 3-D grid; ``SCHED_LPT`` and ``SCHED_LPT_L2``
            launch one axis that ``kernel`` unflattens.
        :param bottom_right: Anchor the diagonal at ``delta = kv_len - q_len`` instead
            of 0.
        :param window_size_left: Sliding-window offset ``W``: key ``k`` is visible from
            row ``q`` when ``k >= q + delta - W`` (``MASK_SWA``). ``None`` leaves the
            left side unbounded.
        :param window_size_right: Right-band offset ``R``: ``k`` is visible when
            ``k <= q + delta + R``. A right bound implies the causal bound;
            ``config_sm90.validate_params`` takes ``R >= 1`` only without ``causal``,
            and spells plain causal as ``causal`` alone.
        :param seq_q_lens_present: Read ``(B,)`` device Q lengths, clamped to ``s_q``.
            Dense only.
        :param seq_kv_lens_present: Read ``(B,)`` device KV lengths, clamped to ``s_kv``.
            THD sets it.
        :param has_sink: Fold the per-Q-head FP32 sink logit from ``sinks`` into the
            softmax denominator, as a virtual column with no V row. A live row with no
            visible key then stores the sink logit as its LSE. ``False`` binds no
            ``sinks`` operand.
        :param thd_varlen: THD (ragged / packed) mode. Q/K/V/O and LSE are packed
            batch-1 views, and ``__call__`` first launches the shared THD setup kernel.
        :param pack_gqa: Pack ``gcd(Q_TILE, qh_per_kh)`` query heads per token.
            Each work head owns one subgroup, and adjacent subgroups may share
            a KV head (e.g. two 64-head subgroups for a 128:1 MQA group).
        :param qh_per_kh: The GQA ratio, refused unless ``h_q == h_kv * qh_per_kh``. Q
            head ``h`` reads KV head ``h // qh_per_kh``. Under ``pack_gqa`` it must
            share a factor with ``Q_TILE``.
        :param scale_mode: The sign of the softmax scale: ``SCALE_POSITIVE``,
            ``SCALE_ZERO`` or ``SCALE_NEGATIVE``. Each sign compiles its own probability
            path, and a negative scale anchors each row at its minimum raw score. The
            runtime ``scale`` must carry this sign.
        :param stats_log2: Store Stats in base 2. The store multiplies each natural-log
            row LSE by ``log2(e)`` after normalization; ``-inf`` stays ``-inf``, and
            ``lse`` None removes it.
        :param thd_origins: THD only: ``config_sm90.TemplateParams.thd_origins``, the
            ports whose bound ragged offsets give each sequence's origin in the caller's
            buffer. ``None`` keeps the packed layout.
        :param split_kv: Dense-only KV partitions. The producer writes sink-free
            partials; the shared combine folds sinks and the final Stats log base.
        """

        _, h_q, h_kv, _, _ = shape
        if out_dtype != in_dtype:
            raise ValueError("prefill_d512_f16_sm90: out_dtype must match in_dtype; V and O alias one SMEM region")
        if h_q != h_kv * qh_per_kh:
            raise ValueError(f"prefill_d512_f16_sm90: KV-head addressing needs H_q == H_kv * qh_per_kh; got {h_q}, {h_kv}, {qh_per_kh}")
        self.b, self.h_q, self.h_kv, self.s_q, self.s_kv = shape
        self.storage_dtype = in_dtype
        self.out_dtype = out_dtype
        self.sched_policy = sched_policy
        self.bottom_right = bottom_right
        # Band model: a right bound, plain or widened causal, sets MASK_CAUSAL, and a
        # left bound sets MASK_SWA. An offset whose flag is clear stays 0 and folds
        # away.
        self.window_left = window_size_left or 0
        self.window_right = window_size_right or 0
        self.mask_flags = (MASK_CAUSAL if is_causal or window_size_right is not None else MASK_NONE) | (MASK_SWA if window_size_left is not None else MASK_NONE)
        self.banded = self.mask_flags != MASK_NONE
        # Without a band, compute_kv_loop_bounds never reads the diagonal, so this flag
        # is inert there and True is the value it keeps.
        self.kv_loop_bottom_right = self.bottom_right if self.banded else True
        self.has_sink = has_sink
        self.fold_sink = has_sink and split_kv == 1
        self.split_kv = split_kv
        self.thd_varlen = thd_varlen
        self.page_size = page_size
        self.kv_tma_rows = min(page_size, self.KV_TILE) if page_size else self.KV_TILE
        self.thd_origins = thd_origins if thd_varlen else None
        # Dense graphs clamp per-sequence device lengths to the declared envelope; THD
        # reads its lengths from the setup launch's metadata instead.
        self.dense_q_lengths = not thd_varlen and seq_q_lens_present
        self.dense_kv_lengths = not thd_varlen and seq_kv_lens_present

        self.pack_gqa = pack_gqa
        self.qh_per_kh = qh_per_kh
        # Split a GQA group into equal subgroups that divide the 64-row tile.
        self.heads_per_tile = pack_gqa_group_size(qh_per_kh, self.Q_TILE, partial=True) if pack_gqa else 1
        self.scale_mode = scale_mode
        # The anchor is the raw-score extremum that maximizes the scaled score: the
        # maximum for a non-negative scale, the minimum for a negative one, with the
        # sentinel as its identity.
        self.anchor_sentinel = -cutlass.Float32.inf if scale_mode >= SCALE_ZERO else cutlass.Float32.inf
        self.stats_log2 = stats_log2 and split_kv == 1

        # Named-barrier ids. The init publish takes barrier_cta_sync's default id 0, so
        # the named ones start at 1. Id 1 is the V/O alias rendezvous of the 256 PV
        # threads. Ids 2 and 3 order each O half's four staging warps before its TMA
        # store.
        self.bar_vo = 1
        self.bar_o_half_base = 2
        # Ids 4 and 5 order each compute warpgroup's V-tail zeroing before its PV.
        self.bar_v_tail_base = 4

        self.threads_per_cta = len(self.REG_BUDGETS) * cute.arch.THREADS_PER_WARPGROUP

        self._setup_attributes()

    def _setup_attributes(self):
        """Compute derived tile, MMA, SMEM, and TMA constants."""

        # Tiling; a THD launch sizes its Q-tile axis from thd_max_sq instead.
        self.tokens_per_tile = self.Q_TILE // self.heads_per_tile
        self.q_tiles = (self.s_q + self.tokens_per_tile - 1) // self.tokens_per_tile
        self.heads = self.h_q // self.heads_per_tile
        # Every launched Q row is real only on a dense graph whose declared Q length is
        # whole tiles; online_softmax's row-validity test and _run_unit's partial-tile
        # guard fold away.
        self.q_rows_full = not self.thd_varlen and not self.dense_q_lengths and self.s_q % self.tokens_per_tile == 0
        # Q_TILE == KV_TILE, so one Q_TILE x HEAD_TILE count sizes the Q, K and V/O
        # regions.
        self.qkv_tile_elems = self.Q_TILE * self.HEAD_TILE
        self.vo_half_elements = self.Q_TILE * self.PV_HALF
        self.p_xfer_elems = self.Q_TILE * self.KV_TILE

        # MMA, per 128-thread warpgroup: a thread's QK C slots per row, and its O-half C
        # slots.
        self.qk_row_slots = self.p_xfer_elems // cute.arch.THREADS_PER_WARPGROUP // 2
        self.o_fragment_slots = self.vo_half_elements // cute.arch.THREADS_PER_WARPGROUP
        # SMEM: kernel() allocates every region at a fixed size, 222,408 B of SM90's
        # 232,448 B. Row r's FP32 factor sits at slot 2*r of the alpha and inverse
        # arrays; the odd slots are unused. WG1 writes each slot, WG2 reads it for its O
        # half.
        self.row_factor_elems = 2 * self.Q_TILE

        # TMA: every region moves in SW128 slabs of 128-byte rows, so one Q_TILE x
        # HEAD_TILE tile is tma_swizzle_chunks boxes of swizzle_chunk_elems columns,
        # slab_elements apart in SMEM. One geometry serves all three load ports.
        # __call__ builds the maps themselves, over the caller's strides.
        def get_swizzle(head_tile: int):
            head_bytes = head_tile * self.storage_dtype.bytes
            # SW128 only: wgmma_descriptor hard-codes the 8-row x 128-byte atom
            # (SBO=1024, LBO=16/8192) with no mode to select, so a 64- or 32-byte
            # swizzle would load correctly and then be misread.
            if head_bytes % 128:
                raise ValueError(f"prefill_d512_f16_sm90: a {head_bytes} B head tile is not whole SW128 slabs, which every WGMMA descriptor assumes")
            swizzle, span = cuda.TensorMapSwizzle.s128b, 128
            return swizzle, head_bytes // span, head_tile // (head_bytes // span)

        self.tma_swizzle, self.tma_swizzle_chunks, self.swizzle_chunk_elems = get_swizzle(self.HEAD_TILE)
        self.slab_elements = self.Q_TILE * self.swizzle_chunk_elems

    def tma_slice(self, tma_desc, desc_ptr, head_idx, seq_coord, batch_idx, tma_order, d_coord=None):
        """Return one TMA box at view coordinates ``(batch, head, token, d)``.

        The coordinates are listed in the map's own axis order. A dense box reads the
        grid-constant ``tma_desc`` with ``desc_ptr`` None. Under THD a runtime
        ``desc_ptr`` overrides it: O reads its own sequence's map, K and V the one
        clamped map every sequence shares (their own sequence's maps under caller-buffer
        origins), and Q keeps the grid-constant map. THD maps
        drop the batch axis. A PackGQA Q/O box spans ``heads_per_tile`` heads from
        ``head_idx``. ``tma_order`` is the port's ``self.tma_orders`` entry, the view
        modes ``__call__`` listed innermost first.
        """
        view = ((head_idx, seq_coord) if self.thd_varlen else (batch_idx, head_idx, seq_coord)) + (cutlass.Int32(0) if d_coord is None else d_coord,)
        return GmemTileTmaSlice(tma_desc, tuple(view[mode] for mode in tma_order), desc_ptr=desc_ptr)

    @cute.jit
    def load_q_tile_tma(
        self,
        sQ: SmemTile,
        tma_q_desc: cutlass.GridConstant[cuda.TensorMap],
        mbar: MBarrier,
        work,
    ) -> None:
        """Launch the TMA loads for the resident Q tile into its eight SW128 slabs.

        The producer warp calls this after the caller's EMPTY wait; the consumer's FULL
        wait completes the transfer. A PackGQA Q box spans heads_per_tile heads from the
        tile's first query head and lands head-fast, in the (D, H, S, B) map order.

        :param sQ: The destination SmemTile, tma_loads_per_tile slabs.
        :param tma_q_desc: Q's grid-constant map, dense and THD alike. Q needs no
            runtime map: a tile overhanging a sequence end lands rows the O store's
            clipped map discards.
        :param mbar: Q's FULL barrier.
        :param work: The decoded work tile, read for its Q coordinates and THD token
            origin.
        """
        # Keep this construction above the arrive: work.head * heads_per_tile is
        # evaluated here, not after the Q FULL arrive.
        q_coord = work.q_base + work.q_offset if cutlass.const_expr(self.thd_varlen) else work.q_base
        source = self.tma_slice(tma_q_desc, None, work.head * self.heads_per_tile, q_coord, work.batch, self.tma_orders[0])
        # One elected arrival expects every copy's box bytes; zero-filled out-of-bounds
        # elements still count.
        mbar.arrive(n_bytes=tma_q_desc.global_tx_bytes() * sQ.tma_loads_per_tile, pred=prims.elect_sync())
        tma_load_tile(sQ, source, mbar.smem_ptr)

    @cute.jit
    def load_one_kv_tile(
        self,
        s_dst: SmemTile,
        tma_desc: cutlass.GridConstant[cuda.TensorMap],
        mbar: MBarrier,
        work,
        kv_seq_idx: cutlass.Int32,
        is_v: cutlass.Constexpr[bool],
    ) -> None:
        """Launch the TMA loads for one whole K or V tile into its eight SW128 slabs.

        The producer warp calls this after the caller's EMPTY wait; the consumer's FULL
        wait completes the transfer. The map's D extent bounds the box, so there is no
        envelope parameter.

        :param s_dst: The destination SmemTile, sK or sV, tma_loads_per_tile slabs.
        :param tma_desc: The port's grid-constant map.
        :param mbar: The port's FULL barrier.
        :param work: The decoded work tile, read for its KV head, batch, THD token
            origin and, under THD, the clamped map every sequence shares on this port
            (this sequence's own map under caller-buffer origins).
        :param kv_seq_idx: The first token of the box, from the caller's reverse KV
            walk.
        :param is_v: Select the V port over the K one: its THD map pointer and map axis
            order.
        """
        # is_v picks the port at trace time, so no branch is staged. This construction
        # carries no arithmetic, so it has no ordering constraint against the arrive.
        if cutlass.const_expr(self.page_size > 0):
            table = work.block_table_v if is_v else work.block_table
            parts = self.KV_TILE // self.kv_tma_rows
            mbar.arrive(n_bytes=tma_desc.global_tx_bytes() * s_dst.tma_loads_per_tile * parts, pred=prims.elect_sync())
            for part in cutlass.range_constexpr(parts):
                token = kv_seq_idx + part * self.kv_tma_rows
                # Dead subpages still issue zero-filling TMA boxes, preserving the
                # barrier byte count without reading unused page-table entries.
                page = cutlass.Int32(-1)
                if token < work.kv_len:
                    page = table[work.batch, token // self.page_size]
                source = self.tma_slice(tma_desc, None, work.kv_head, token % self.page_size, page, self.tma_orders[2 if is_v else 1])
                chunk = SimpleNamespace(
                    base=s_dst.base.subview(part * self.kv_tma_rows * self.swizzle_chunk_elems),
                    tma_loads_per_tile=s_dst.tma_loads_per_tile,
                    tma_granu_elems=s_dst.tma_granu_elems,
                    tma_subtile_stride_elems=s_dst.tma_subtile_stride_elems,
                )
                tma_load_tile(chunk, source, mbar.smem_ptr)
        else:
            desc_ptr = work.v_desc_ptr if is_v else work.k_desc_ptr
            kv_coord = kv_seq_idx + work.kv_offset if cutlass.const_expr(self.thd_varlen) else kv_seq_idx
            source = self.tma_slice(tma_desc, desc_ptr, work.kv_head, kv_coord, work.batch, self.tma_orders[2 if is_v else 1])
            mbar.arrive(n_bytes=tma_desc.global_tx_bytes() * s_dst.tma_loads_per_tile, pred=prims.elect_sync())
            # Caller-origin maps were acquired once at the top of tmaldg_warp_group.
            tma_load_tile(s_dst, source, mbar.smem_ptr, acquire=self.thd_origins is None)

    @cute.jit
    def tmaldg_warp_group(self, sQ, sK, sV, bars, tma_q_desc, tma_k_desc, tma_v_desc, work):
        """Load resident Q once, then the reverse K/V tiles both PV halves consume.

        Warp 0 of producer warpgroup WG0 owns the waits, the reverse walk and the drain.
        WG1's epilogue releases Q. K and V are single-stage and refilled together once
        per KV step, each after its own EMPTY wait. The closing waits on the last
        generation of every EMPTY barrier leave no TMA payload in flight at exit.
        """
        tid = cute.arch.thread_idx()[0]
        if (work.q_base < work.q_len) & (work.iterations > 0):
            if tid < cute.arch.WARP_SIZE:
                # EMPTY waits start at phase 1, which passes at once, so the first trip
                # needs no branch. One state serves K and V, which refill together.
                q_empty_state = PipelineState.start(phase=1)
                kv_empty_state = PipelineState.start(phase=1)
                if cutlass.const_expr(self.thd_origins is not None):
                    # This sequence's own K/V maps: acquire each once for this warp and
                    # start their fetch before the Q load, so the K/V loads below need no
                    # per-tile acquire (with shared maps every CTA hits one cached map).
                    for desc_ptr in (work.k_desc_ptr, work.v_desc_ptr):
                        prims.fence_proxy_acquire(prims.MemScope.GPU, desc_ptr, 128, from_proxy=prims.Proxy.GENERIC, to_proxy=prims.Proxy.TENSORMAP)
                        prims.prefetch_tensormap(desc_ptr)
                bars.mb_tma_q_empty.wait(q_empty_state.phase)
                q_empty_state = advance(q_empty_state, 1)
                self.load_q_tile_tma(sQ, tma_q_desc, bars.mb_tma_q_full, work)
                for kv_step in cutlass.range(work.iterations, unroll=1):
                    kv_seq_idx = (work.kv_right - 1 - kv_step) * self.KV_TILE
                    bars.mb_tma_k_empty.wait(kv_empty_state.phase)
                    self.load_one_kv_tile(sK, tma_k_desc, bars.mb_tma_k_full, work, kv_seq_idx, is_v=False)
                    bars.mb_tma_v_empty.wait(kv_empty_state.phase)
                    self.load_one_kv_tile(sV, tma_v_desc, bars.mb_tma_v_full, work, kv_seq_idx, is_v=True)
                    kv_empty_state = advance(kv_empty_state, 1)
                bars.mb_tma_q_empty.wait(q_empty_state.phase)
                bars.mb_tma_k_empty.wait(kv_empty_state.phase)
                bars.mb_tma_v_empty.wait(kv_empty_state.phase)

    @cute.jit
    def online_softmax(
        self,
        basic_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        s_regs: cutlass.Array,
        kv_seq_idx: cutlass.Int32,
        active_masks: cutlass.Constexpr,
        is_first_kv_tile: cutlass.Constexpr,
    ):
        """Update two rows and pack P for both PV consumers into consumed score slots.

        ``basic_params`` supplies the fixed work/lane references; ``softmax_params`` the
        mutable row state, this tile's alpha pair and the launch-uniform scales and
        guards. ``s_regs`` arrives with completed QK scores. ``compute_one_kv_tile``
        publishes the returned P words and alpha to WG2.

        Per row half r, ``anchor[r]`` is the exact sign-aware raw-score extremum, the
        sentinel while the row has no valid score, and ``partial_sum[r]`` this lane's
        FP32 share relative to ``exp(scale * anchor[r])``. After the first tile each
        step also yields ``alpha[r]``, which corrects both O halves.

        C slot 4*k_frag+2*row_half+{0,1} owns row 16*warp+lane//4+8*row_half and key
        kv_seq_idx+8*k_frag+2*(lane%4)+{0,1}. Four adjacent lanes share a row, so their
        anchors and alpha agree and their complementary sums merge only at finalization.
        ``MASK_NONE`` requires every score in the tile to be valid.

        :return: ``s_regs``, with this tile's P packed into the consumed score slots.
        """
        work, local_tid = basic_params.work, basic_params.local_tid
        row_anchor, row_sum, alpha = softmax_params.anchor, softmax_params.partial_sum, softmax_params.alpha
        scale_log2, scale_log2_wide = softmax_params.scale_log2, softmax_params.scale_log2_wide
        extremum, quad_op = (cute.arch.fmax, "max") if cutlass.const_expr(self.scale_mode >= SCALE_ZERO) else (cute.arch.fmin, "min")
        sentinel, zero = cutlass.Float32(self.anchor_sentinel), cutlass.Float32(0.0)
        # A masked-out P stays a runtime value. On a dense unmasked graph the first
        # tile's padding mask folds, and a literal 0.0 would reach fp32_to_fp16's inline
        # PTX as an immediate, which ICEs libNVVM (pointwise.opaque_f32_zero).
        p_zero = opaque_f32_zero() if cutlass.const_expr(active_masks != MASK_NONE) else zero

        for row_half in cutlass.range_constexpr(2):
            s_reg_idx_lo = row_half * 2
            s_reg_idx_hi = row_half * 2 + 1

            # Resolve this row's validity once; each bit names one explicit score slot.
            row_live = cutlass.Boolean(True)
            column_mask = cutlass.Uint32(0xFFFF)
            if cutlass.const_expr(active_masks != MASK_NONE):
                q_token = work.q_base + owned_row(local_tid, row_half) // self.heads_per_tile
                if cutlass.const_expr(not self.q_rows_full):
                    row_live = q_token < work.q_len
                diagonal = q_token + work.kv_len - work.q_len if cutlass.const_expr(self.bottom_right) else q_token
                column_base = kv_seq_idx + 2 * (local_tid % 4)
                if cutlass.const_expr(active_masks & MASK_PADDED):
                    column_mask = score_prefix_mask(work.kv_len, column_base)
                if cutlass.const_expr(active_masks & MASK_CAUSAL):
                    column_mask = column_mask & score_prefix_mask(diagonal + cutlass.Int32(self.window_right), column_base, inclusive=True)
                if cutlass.const_expr(active_masks & MASK_SWA):
                    column_mask = column_mask & ~score_prefix_mask(diagonal - cutlass.Int32(self.window_left), column_base)

            # Fold query validity into the lane's bitmap once for all score/P uses.
            column_mask = column_mask if row_live else cutlass.Uint32(0)

            # Gather masked pairs in column order, keeping each pair's validity for the
            # sum pass. Column c = 2*k_frag + j extends extremum chain c % 4; the four
            # chains are independent.
            score_pairs, validity, chains = [], [], []
            local_abs = cutlass.Float32(0.0)
            for k_frag in cutlass.range_constexpr(self.qk_row_slots // 2):
                s_off = k_frag * 4
                # A zero scale scores every valid key 0, so a live row's anchor is 0 and
                # alpha and P are exp2(+-0) = 1 from runtime EX2s. P must stay a runtime
                # value: a constant would reach fp32_to_fp16's inline PTX as an
                # immediate, which ICEs libNVVM (pointwise.opaque_f32_zero).
                s0 = zero if cutlass.const_expr(self.scale_mode == SCALE_ZERO) else s_regs[s_off + s_reg_idx_lo]
                s1 = zero if cutlass.const_expr(self.scale_mode == SCALE_ZERO) else s_regs[s_off + s_reg_idx_hi]
                valid0 = cutlass.Boolean(column_mask & cutlass.Uint32(1 << (2 * k_frag)))
                valid1 = cutlass.Boolean(column_mask & cutlass.Uint32(1 << (2 * k_frag + 1)))
                validity.append((valid0, valid1))
                if not valid0:
                    s0 = sentinel
                if not valid1:
                    s1 = sentinel
                score_pairs.append(cutlass.Vector.from_elements((s0, s1), cutlass.Float32))
                if cutlass.const_expr(k_frag < 2):
                    chains += [s0, s1]
                else:
                    chain = 2 * (k_frag % 2)
                    chains[chain], chains[chain + 1] = extremum(chains[chain], s0), extremum(chains[chain + 1], s1)
                if cutlass.const_expr(self.scale_mode == SCALE_POSITIVE):
                    # A magnitude guard, not a finiteness proof: fmax quiets NaNs.
                    if valid0:
                        local_abs = cute.arch.fmax(local_abs, cute.math.abs(s0))
                    if valid1:
                        local_abs = cute.arch.fmax(local_abs, cute.math.abs(s1))

            scores = RegTile(vec_concat(score_pairs), size=self.qk_row_slots)

            # Update the exact sign-aware anchor, then derive the old-output factor.
            # History joins the anchor before the quad tree.
            old = row_anchor[row_half]
            current = extremum(extremum(chains[0], chains[1]), extremum(chains[2], chains[3]))
            if cutlass.const_expr(not is_first_kv_tile):
                current = extremum(current, old)
            current = quad_reduce(current, quad_op)
            row_anchor[row_half] = current
            alive, old_alive = current != sentinel, old != sentinel
            safe_current, safe_old = current if alive else zero, old if old_alive else zero
            # A negative scale always takes FP64 exponents and a zero scale never does.
            # A positive scale takes them when the launch-uniform term is set, or when a
            # valid score, either anchor or their distance could overflow the FP32
            # products.
            wide = cutlass.Boolean(self.scale_mode < SCALE_ZERO)
            if cutlass.const_expr(self.scale_mode == SCALE_POSITIVE):
                bound = cute.arch.fmax(cute.arch.fmax(quad_reduce(local_abs, "max"), cute.math.abs(safe_current)), cute.math.abs(safe_old))
                wide = softmax_params.wide_uniform
                if bound > softmax_params.wide_floor:
                    wide = (
                        wide
                        | (bound > softmax_params.wide_limit)
                        | (cute.math.abs(cutlass.Float64(safe_old) - cutlass.Float64(safe_current)) > cutlass.Float64(FLT_MAX_F64))
                    )
            # Resolve the row's numerical path once, for its scores and the old anchor.
            # Keeping the branch outside the pairs avoids reconvergence around each
            # exp2, and computing the old anchor's exponent inside it keeps its FP64
            # difference off the narrow path.
            exponents, old_exponent = scores.vec, zero
            if wide:
                old_exponent = cutlass.Float32((cutlass.Float64(old) - cutlass.Float64(current)) * scale_log2_wide)
                # FP64 differences; the zero-select below discards every invalid slot's
                # exponent.
                exponent_pairs = []
                for k_frag in cutlass.range_constexpr(self.qk_row_slots // 2):
                    pair = scores[2 * k_frag : 2 * k_frag + 2]
                    s0, s1 = pair[0], pair[1]
                    x0 = cutlass.Float32((cutlass.Float64(s0) - cutlass.Float64(current)) * scale_log2_wide)
                    x1 = cutlass.Float32((cutlass.Float64(s1) - cutlass.Float64(current)) * scale_log2_wide)
                    exponent_pairs.append(cutlass.Vector.from_elements((x0, x1), cutlass.Float32))
                exponents = vec_concat(exponent_pairs)
            else:
                old_exponent = (old - safe_current) * scale_log2
                # Preserve both products and the existing exponent rounding policy.
                exponents = (scores * scale_log2 - safe_current * scale_log2).vec
            old_scale = zero
            if cutlass.const_expr(is_first_kv_tile):
                old_scale = cutlass.Float32(1.0)
            elif old_alive & alive:
                old_scale = cute.math.exp2(old_exponent, fastmath=True)
            alpha[row_half] = old_scale

            # History is the first addend, then each pair's p0 and p1; the sum stays
            # lane-partial until finalization. Moving the history product can change
            # ptxas's multiply/add contraction and the recurrent FP32 sum.
            total = zero if cutlass.const_expr(is_first_kv_tile) else row_sum[row_half] * old_scale
            probabilities = RegTile(cute.math.exp2(exponents, fastmath=True), size=self.qk_row_slots)
            for k_frag in cutlass.range_constexpr(self.qk_row_slots // 2):
                p0, p1 = probabilities[2 * k_frag], probabilities[2 * k_frag + 1]
                valid0, valid1 = validity[k_frag]
                if not valid0:
                    p0 = p_zero
                if not valid1:
                    p1 = p_zero
                total = total + p0 + p1
                # One packed word serves both RS and SS PV, in the consumed score slot
                # of p0.
                p_pack = fp32_to_fp16(p0, p1, dtype=self.storage_dtype).bitcast(cutlass.Float32)
                p_pack_off = k_frag * 4 + row_half * 2
                s_regs[p_pack_off] = p_pack
            row_sum[row_half] = total

        return s_regs

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: Optional[cute.Tensor],
        sinks: Optional[cute.Tensor],
        seq_q_lens: cute.Tensor,
        seq_kv_lens: cute.Tensor,
        scale: cutlass.Float32,
        thd_max_sq: cutlass.Int32,
        thd_q_lens: Optional[cute.Tensor],
        thd_kv_lens: Optional[cute.Tensor],
        thd_lens_form: Optional[cutlass.Int32],
        thd_ro_q: Optional[cute.Tensor],
        thd_ro_k: Optional[cute.Tensor],
        thd_ro_v: Optional[cute.Tensor],
        thd_ro_o: Optional[cute.Tensor],
        thd_ro_lse: Optional[cute.Tensor],
        stream: cuda_driver.CUstream,
        block_table: Optional[cute.Tensor] = None,
        block_table_v: Optional[cute.Tensor] = None,
    ) -> None:
        """Build the port maps, run the THD setup when packed, and launch the kernel.

        Only the compiled pointer host calls this internal view-building entry.
        The shared prepared binder validates runtime buffers before launch.

        :param q: Query tensor ``(B, H_q, S_q, D_qk)`` over the caller's strides, or the
            packed ``(1, H_q, T_q, D_qk)`` THD view whose token total is dynamic.
        :param k: Key tensor ``(B, H_kv, S_kv, D_qk)``, or packed
            ``(1, H_kv, T_kv, D_qk)``.
        :param v: Value tensor ``(B, H_kv, S_kv, D_v)``, or packed
            ``(1, H_kv, T_kv, D_v)``.
        :param o: Output tensor ``(B, H_q, S_q, D_v)``, or packed
            ``(1, H_q, T_q, D_v)``, written in place.
        :param lse: FP32 Stats, dense ``(B, H_q, S_q)`` or packed ``(1, H_q, T_q)``: the
            natural-log LSE, or base 2 under ``stats_log2``. ``None`` removes the Stats
            store at compile time.
        :param sinks: ``(H_q,)`` FP32 per-Q-head sink logits, ``None`` without
            ``has_sink``.
        :param seq_q_lens: Int32 ``(B,)`` Q lengths, or a dead null view.
        :param seq_kv_lens: Int32 ``(B,)`` KV lengths, or a dead null view. Under THD, the
            128-byte-aligned ``config_sm90.thd_meta_words(B, origins)`` scratch: metadata,
            then maps (then the token origins).
        :param scale: The host softmax scale in natural units; the kernel folds
            ``log2(e)`` itself. Its sign must match ``scale_mode``.
        :param thd_max_sq: THD only: the declared S_q envelope, which sizes the grid.
        :param thd_q_lens: THD only: the caller's ``(B,)`` lengths or ``(B+1,)`` prefix
            sums. ``None`` when dense.
        :param thd_kv_lens: THD only: the same for KV.
        :param thd_lens_form: THD only: bit 0: Q is cu, bit 1: KV is cu.
        :param thd_ro_q: Caller-origin THD only: Q's bound ``(B+1,)`` ragged offsets, or
            ``None`` for a port without them; likewise ``thd_ro_k`` .. ``thd_ro_lse``.
            Each THD view's token extent is then that port's own capacity.
        :param stream: CUDA stream of the THD setup launch and the attention launch.
        """

        # The shared prepared binder validates addresses, extents and Int64 strides.
        # Each port's compile-time axis order also drives the kernel's coordinates.
        def port_tma_desc(tensor, heads, rows, stride_order):
            box = (1, heads, rows // heads, self.swizzle_chunk_elems)
            if cutlass.const_expr(self.thd_varlen):
                # The packed token total is dynamic and a zero-extent map is invalid:
                # keep one unread token.
                tokens = cute.math.max(cutlass.Int32(tensor.shape[2]), cutlass.Int32(1))
                tensor = cute.make_tensor(tensor.iterator, cute.make_layout((tensor.shape[1], tokens, tensor.shape[3]), stride=tensor.stride[1:]))
                box = box[1:]
            return cuda.create_tensor_map_tiled_from_view(tensor, box_dims=box, stride_order=stride_order, swizzle=self.tma_swizzle)

        tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc = (
            port_tma_desc(q, self.heads_per_tile, self.Q_TILE, self.tma_orders[0]),
            port_tma_desc(k, 1, self.kv_tma_rows, self.tma_orders[1]),
            port_tma_desc(v, 1, self.kv_tma_rows, self.tma_orders[2]),
            port_tma_desc(o, self.heads_per_tile, self.Q_TILE, self.tma_orders[3]),
        )
        # The LPT policies launch the same tiles on one axis, which kernel unflattens.
        # THD is NATURAL, over ceil(thd_max_sq / tokens_per_tile) tiles.
        grid = (self.q_tiles, self.heads, self.b)
        if cutlass.const_expr(self.sched_policy != SCHED_NATURAL or self.split_kv > 1):
            grid = (self.q_tiles * self.heads * self.b, 1, self.split_kv)
        if cutlass.const_expr(self.thd_varlen):
            meta_words = thd_meta_words(self.b, self.thd_origins is not None)
            if cutlass.const_expr(tuple(seq_kv_lens.shape) != (meta_words,)):
                raise ValueError(f"prefill_d512_f16_sm90: THD seq_kv_lens must be the ({meta_words},) metadata tensor")
            q_tiles = (thd_max_sq + self.tokens_per_tile - 1) // self.tokens_per_tile
            grid = (q_tiles, cutlass.Int32(self.heads), cutlass.Int32(self.b))
        if cutlass.const_expr(self.thd_origins is not None):
            # Caller-buffer origins: metadata, the clamped per-port token origins and one
            # map per sequence for O, K and V, released to the tensor-map proxy exactly as
            # the shared setup launch does (thd_origins.py).
            capacities = (q.shape[2], k.shape[2], v.shape[2], o.shape[2], lse.shape[2] if cutlass.const_expr(lse is not None) else cutlass.Int32(0))
            token_strides = (
                cutlass.Int64(q.stride[2]),
                cutlass.Int64(k.stride[2]),
                cutlass.Int64(v.stride[2]),
                cutlass.Int64(o.stride[2]),
                cutlass.Int64(lse.stride[2]) if cutlass.const_expr(lse is not None) else cutlass.Int64(1),
            )
            build_thd_origin_setup_kernel(
                k,
                v,
                o,
                tma_o_desc,
                tma_k_desc,
                tma_v_desc,
                cute.make_tensor(
                    cute.recast_ptr(seq_kv_lens.iterator + THD_MAPS_OFF(self.b), dtype=cutlass.Int64),
                    cute.make_layout((thd_origin_maps(self.b) * TENSOR_MAP_QWORDS,), stride=(1,)),
                ),
                cute.make_tensor(
                    cute.recast_ptr(seq_kv_lens.iterator + thd_origins_off(self.b), dtype=cutlass.Int64), cute.make_layout((5, self.b), stride=(self.b, 1))
                ),
                seq_kv_lens,
                thd_q_lens,
                thd_kv_lens,
                thd_lens_form,
                cutlass.Int32(self.b),
                thd_ro_q,
                thd_ro_k,
                thd_ro_v,
                thd_ro_o,
                thd_ro_lse,
                token_strides,
                capacities,
                self.thd_origins,
            ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        elif cutlass.const_expr(self.thd_varlen):
            # The shared THD setup launch: one elected thread writes the metadata, the
            # per-sequence O maps and the two clamped K/V maps, then issues one
            # GENERIC->TENSORMAP release fence. Stream order publishes all of it before
            # the attention launch, and each TMA issuer's acquire completes the handoff.
            # O's token stride keeps its call-site width: the parameter is unannotated
            # so a >= 2**31 element stride is not truncated. The batch ranking and
            # live-unit words the same kernel writes are dead here.
            build_thd_meta_o_descs_kernel(
                o,
                tma_o_desc,
                tma_k_desc,
                tma_v_desc,
                cute.make_tensor(
                    cute.recast_ptr(seq_kv_lens.iterator + THD_MAPS_OFF(self.b), dtype=cutlass.Int64),
                    cute.make_layout(((self.b + 3) * TENSOR_MAP_QWORDS,), stride=(1,)),
                ),
                seq_kv_lens,
                thd_q_lens,
                thd_kv_lens,
                thd_lens_form,
                cutlass.Int32(self.heads),
                cutlass.Int32(self.b),
                cutlass.Int64(o.stride[2]),
                cutlass.Int32(self.tokens_per_tile),
                q_tiles * cutlass.Int32(self.heads * self.b),
            ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        # Only a has_sink specialization reads sinks; any other plan would ignore the
        # operand.
        if cutlass.const_expr(self.has_sink != (sinks is not None)):
            raise ValueError("prefill_d512_f16_sm90: sinks must be provided exactly when the kernel is configured with has_sink")
        if cutlass.const_expr(sinks is not None and tuple(sinks.shape) != (self.h_q,)):
            raise ValueError(f"prefill_d512_f16_sm90: sinks must have shape (H_q,) = ({self.h_q},); got {tuple(sinks.shape)}")
        # The declared problem sizes the grid, the KV-head addressing and the Stats
        # store, so each operand must carry it. A THD port is a batch-1 view whose token
        # total has no static extent.
        batch = 1 if self.thd_varlen else self.b
        ports = (("Q", q, self.h_q, self.s_q), ("K", k, self.h_kv, self.s_kv), ("V", v, self.h_kv, self.s_kv), ("O", o, self.h_q, self.s_q))
        if cutlass.const_expr(lse is not None):
            ports = (*ports, ("LSE", lse, self.h_q, self.s_q))
        for name, tensor, heads, tokens in ports:
            declared = (batch * (self.split_kv if name in ("O", "LSE") else 1), heads, tokens)
            if cutlass.const_expr(self.page_size > 0 and name in ("K", "V")):
                declared = (tensor.shape[0], heads, self.page_size)
            if cutlass.const_expr(any(isinstance(extent, int) and extent != size for extent, size in zip(tensor.shape, declared))):
                raise ValueError(f"prefill_d512_f16_sm90: {name} shape {tuple(tensor.shape)} disagrees with the declared (B, H, S) = {declared}")
        # Only a length tensor some role reads reaches the kernel; THD's is the metadata.
        read_q_lens = seq_q_lens if cutlass.const_expr(self.dense_q_lengths) else None
        read_kv_lens = seq_kv_lens if cutlass.const_expr(self.thd_varlen or self.dense_kv_lengths) else None
        if cutlass.const_expr(self.page_size > 0):
            self.kernel(tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc, lse, sinks, read_q_lens, read_kv_lens, scale, block_table, block_table_v).launch(
                grid=grid, block=(self.threads_per_cta, 1, 1), stream=stream
            )
        else:
            self.kernel(tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc, lse, sinks, read_q_lens, read_kv_lens, scale).launch(
                grid=grid, block=(self.threads_per_cta, 1, 1), stream=stream
            )


SM90FusedMultiHeadAttentionForward.kernel.set_name_prefix("cudnn_sdpa_fwd_sm90_d64_one_wg", remove_cutlass_symbol=True)


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
