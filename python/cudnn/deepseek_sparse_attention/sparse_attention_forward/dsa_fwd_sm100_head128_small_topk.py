# Copyright (c) 2025 DeepSeek
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100 2-CTA sparse-prefill forward kernel for H=128 and D=512.

The implementation uses a 2-SM topology: a ``(2, 1, 1)``
cluster, a CLC persistent outer loop, a four-stage 64x256 half-KV ring, and
cooperative TMEM allocation. Sparse KV uses the SM100 2-D TMA gather4 opcode
through a narrow low-level bridge while retaining compiler-owned tensor maps.

QK and PV are genuine ``cta_group::2`` tcgen05 operations. QK accumulates
FP32 scores in TMEM, softmax quantizes P to the input dtype in shared memory, and the
PV permutation accumulates O in TMEM. The public mathematical semantics
include invalid indices, dynamic topk length, sink-only normalization,
indexer LSE, and empty-row sentinels (O=0, max_logits=-inf, lse=-inf per the
generic sparse_attention_forward_wrapper contract; the internal
``lse_indexer`` prefix statistic keeps its own +inf empty sentinel).

DSL-style note (PR2 of the fwd-API roadmap): kept on ``cutlass.pipeline`` /
``cutlass.utils.blackwell_helpers`` rather than ported onto
``cudnn.frost.tile_dsl`` in this pass, for the same reason as
``dsa_fwd_sm100_head64.py`` (see that module's docstring for the full
rationale) -- this is a 2-CTA cluster, CLC-persistent, gather4-driven mainloop
whose validity-aware sparse-gather TMA usage has no equivalent in
``cudnn.frost.tile_dsl.tma`` today. Dispatch-surface uniformity with the
frost-tile_dsl-styled PR4 GQA substrate kernels is instead provided by
``_interface_sm100.py``'s ``DsaSm100TemplateParams``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import T

from ..utils import copy as copy_utils
from ._nvvm_compat import fmax_ftz_nonan
from ._tcgen05_sync import tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync
from ._tma_gather4 import tma_gather4_cta2_cta0_mbar


@cute.jit
def _initialize_q_load_mbarrier(tidx: Int32, q_load_mbar: cute.Pointer) -> None:
    """Initialize the per-CTA Q barrier without carrying ``SharedStorage`` through a dynamic branch."""

    # CuTe DSL 4.5 cannot flatten a local ``cute.struct`` object across a
    # dynamic ``if``.  The helper retains the original one-thread protocol
    # while restricting the branch state to primitive DSL values.
    if tidx == Int32(0):
        cute.arch.mbarrier_init(q_load_mbar, 1)
        cute.arch.mbarrier_init_fence()


@cute.jit
def _ldg_indices_256(
    indices: cute.Tensor,
    query_idx: Int32,
    slot: Int32,
    l2_evict_first: cutlass.Constexpr[bool],
) -> Tuple[Int32, ...]:
    """Load eight aligned indices with a 256-bit cache policy."""

    # The public wrapper pads every index row to 64 Int32 elements.  Torch's
    # allocation base, the 256-byte row stride, and the eight-element slot
    # granularity therefore make every address passed here 32-byte aligned.
    ptr = indices.iterator + cute.crd2idx((query_idx, slot), indices.layout)
    ptr = cute.make_ptr(Int32, ptr.toint(), indices.memspace, assumed_align=32)
    l2_policy = "evict_first" if const_expr(l2_evict_first) else "evict_normal"
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 8),
        [ptr.llvm_ptr],
        "{\n\t"
        ".reg .b64 v<4>;\n\t"
        f"ld.global.nc.L1::no_allocate.L2::{l2_policy}.L2::256B.v4.u64 "
        "{v0, v1, v2, v3}, [$8];\n\t"
        "mov.b64 {$0, $1}, v0;\n\t"
        "mov.b64 {$2, $3}, v1;\n\t"
        "mov.b64 {$4, $5}, v2;\n\t"
        "mov.b64 {$6, $7}, v3;\n\t"
        "}\n",
        ",".join(["=r"] * 8 + ["l"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i])) for i in range(8))


class SparseAttentionForwardSm100Head128SmallTopKPrefill:
    """Small-topk Prefill specialization for H=128, D=512."""

    H_Q = 128
    D_QK = 512
    D_V = 512
    B_TOPK = 64
    HALF_H = 64
    HALF_D = 256
    QK_K_TILE = 64
    NUM_THREADS = 512
    NUM_WARPS = 16
    WARPGROUP_SIZE = 128
    NUM_KV_STAGES = 4
    TMEM_COLUMNS = 512
    TMEM_O_OFFSET = 0
    TMEM_Q_OFFSET = 256
    TMEM_P_OFFSET = 384
    MAX_INIT_LOG2 = -1.0e30
    RESCALE_THRESHOLD_LOG2 = 6.0
    LOG2_E = math.log2(math.e)
    LN_2 = math.log(2.0)
    WG_QO = 0
    WG_KV = 1
    WG_MMA = 2
    WG_SOFTMAX = 3
    CLC_WARP = 10

    def __init__(
        self,
        d_qk: int = 512,
        indexer_topk: int = 0,
    ):
        if d_qk != self.D_QK:
            raise ValueError("head128 small-topk Prefill only instantiates D_QK=512")
        if indexer_topk not in (0, 512, 1024):
            raise ValueError("indexer_topk must be one of 0, 512, or 1024")

        self.indexer_topk = indexer_topk
        self.indexer_tile = indexer_topk // self.B_TOPK - 1 if indexer_topk else -1
        self.cluster_shape_mnk = (2, 1, 1)
        self.num_clc_stages = 1
        self.num_clc_response_bytes = 16
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.NUM_THREADS)
        # The score exchange pairs warps (0,2) and (1,3).  Independent 64-
        # thread barriers match the source and avoid coupling the two head
        # halves at every softmax synchronization point.
        self.partial_p_barrier_even = pipeline.NamedBarrier(barrier_id=2, num_threads=2 * 32)
        self.partial_p_barrier_odd = pipeline.NamedBarrier(barrier_id=4, num_threads=2 * 32)
        # The one-time li reductions reuse a slice of sPExchange that belongs
        # to the even score-exchange pair.  Synchronize the whole softmax
        # warpgroup around that reuse so the even pair cannot start the next
        # score exchange while the odd pair is still reading its li partial.
        self.softmax_wg_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=self.WARPGROUP_SIZE)
        # WG0 stages the TMEM epilogue into the aliased sQ/sO surface before
        # one elected lane launches its TMA S2G store.  Keep this ID distinct
        # from both independent softmax-pair barriers above.
        self.qo_epilogue_barrier = pipeline.NamedBarrier(barrier_id=5, num_threads=self.WARPGROUP_SIZE)

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
        softmax_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        self.element_dtype = q.element_type
        if const_expr(self.element_dtype not in (Float16, BFloat16)):
            raise TypeError("q must be FP16 or BF16")
        if const_expr(kv.element_type is not self.element_dtype or out.element_type is not self.element_dtype):
            raise TypeError("q, kv, and out must have matching dtypes")
        if const_expr(indices.element_type != Int32):
            raise TypeError("indices must be INT32")
        if const_expr(max_logits.element_type != Float32 or lse.element_type != Float32):
            raise TypeError("max_logits and lse must be FP32")
        if const_expr(lse_indexer is not None and lse_indexer.element_type != Float32):
            raise TypeError("lse_indexer must be FP32")
        if const_expr(attn_sink is not None and attn_sink.element_type != Float32):
            raise TypeError("attn_sink must be FP32")
        if const_expr(topk_length is not None and topk_length.element_type != Int32):
            raise TypeError("topk_length must be INT32")
        if const_expr(cute.rank(q.shape) != 3 or cute.rank(kv.shape) != 2 or cute.rank(indices.shape) != 2):
            raise ValueError("expected q[Tq,H,D], kv[Tkv,D], and indices[Tq,K]")
        # Shape extents are staged values in the DLPack compile path.  The
        # public interface pads indices K to a positive multiple of 64 before
        # reaching this internal kernel, so do not force that extent through
        # ``const_expr`` here.
        if const_expr(self.indexer_topk == 0 and lse_indexer is not None):
            raise ValueError("lse_indexer must be None when indexer_topk is zero")
        if const_expr(self.indexer_topk > 0 and lse_indexer is None):
            raise ValueError("lse_indexer is required when indexer_topk is nonzero")

        qk_tiler = (self.H_Q, self.B_TOPK, self.HALF_D)
        qk_op = tcgen05.MmaF16BF16Op(
            self.element_dtype,
            Float32,
            (128, 128, 16),
            tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.TMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        qk_mma = cute.make_tiled_mma(
            cute.make_mma_atom(qk_op),
            cute.make_layout((1, 1, 1)),
            (128, 64, 16),
        )

        # Use the public-API spelling of the output tiled MMA.
        pv_op = tcgen05.MmaF16BF16Op(
            self.element_dtype,
            Float32,
            (128, 256, 16),
            tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.MN,
        )
        pv_mma = cute.make_tiled_mma(
            cute.make_mma_atom(pv_op),
            cute.make_layout((1, 1, 1)),
            (128, cute.make_layout((128, 2, 2), stride=(1, 256, 128)), 16),
        )
        pv_tiler = (self.H_Q, self.D_V, self.B_TOPK)

        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.cluster_shape_mnk), (qk_mma.thr_id.shape,))
        q_layout = sm100_utils.make_smem_layout_a(
            qk_mma,
            (self.H_Q, self.B_TOPK, self.D_QK),
            self.element_dtype,
            1,
        )
        # Re-express contiguous Q as a five-dimensional TMA tensor map. The
        # (D64, H, D256, D64, Tq) coordinates preserve source
        # D order while the SMEM layout materializes physical stages
        # [0, 4, 1, 5, 2, 6, 3, 7] for the following 2-CTA S2T copy.  Both
        # CTAs issue the group-2 TMA at their respective 64-head coordinate;
        # hardware credits their combined 128 KiB transaction to CTA0.
        tq = cute.size(q.shape[0])
        q_tma_tiler = (self.QK_K_TILE, self.HALF_H, 2, self.HALF_D // self.QK_K_TILE)
        q_tma = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (self.QK_K_TILE, self.H_Q, 2, self.HALF_D // self.QK_K_TILE, tq),
                stride=(1, self.D_QK, self.HALF_D, self.QK_K_TILE, self.H_Q * self.D_QK),
            ),
        )
        q_tma_smem_layout = cute.make_composed_layout(
            q_layout.inner,
            0,
            cute.make_layout(
                q_tma_tiler,
                stride=(1, self.QK_K_TILE, self.QK_K_TILE * self.HALF_H, 2 * self.QK_K_TILE * self.HALF_H),
            ),
        )
        tma_atom_q, q_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO),
            q_tma,
            q_tma_smem_layout,
            q_tma_tiler,
        )
        q_tma_bytes = cute.size_in_bytes(self.element_dtype, q_tma_smem_layout)

        # The per-CTA 64x512 16-bit output tile uses the same physical
        # shared-memory allocation as Q. The public TMA
        # store atom owns the output tensor map; no private MLIR bridge is
        # needed for the ordinary S2G operation.
        o_tma_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (64, self.HALF_H, self.D_V // 64),
                stride=(1, 64, 64 * self.HALF_H),
            ),
        )
        if const_expr(cute.cosize(o_tma_smem_layout) != cute.cosize(q_layout)):
            raise RuntimeError("head128 sQ/sO alias layouts must have identical storage size")
        # Keep the descriptor five-dimensional. The final singleton is a
        # batch coordinate (this frontend
        # accepts an already-flattened THD tensor, so its extent is one).
        out_tma = cute.make_tensor(
            out.iterator,
            cute.make_layout(
                (64, self.H_Q, self.D_V // 64, tq, 1),
                stride=(1, self.D_V, 64, self.H_Q * self.D_V, self.H_Q * self.D_V * tq),
            ),
        )
        tma_atom_o, out_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            out_tma,
            o_tma_smem_layout,
            (64, self.HALF_H, self.D_V // 64),
        )

        # The gather descriptor is a logical (D, Tkv) tensor map. A regular
        # 1-CTA atom is sufficient to own the descriptor; the bridge
        # below selects gather4 and cta_group::2 for the actual instruction.
        kv_descriptor_tensor = cute.make_tensor(
            kv.iterator,
            cute.make_layout(
                (self.D_QK, cute.size(kv.shape[0])),
                stride=(1, self.D_QK),
            ),
        )
        kv_descriptor_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((64, 1), stride=(1, 64)),
        )
        tma_atom_kv, _ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            kv_descriptor_tensor,
            kv_descriptor_smem_layout,
            (64, 1),
        )
        kv_tma_bytes = self.B_TOPK * self.D_QK * (self.element_dtype.width // 8)
        # UTCCP consumes four paired 128-d tiles. This view groups the physical
        # 64-d Q stages according to the 2-SM TMA map.
        q_s2t_outer = cute.make_layout(((self.HALF_H, 16), 1, 8, 4), stride=((64, 1), 0, 16, 8192))
        k_layout = sm100_utils.make_smem_layout_b(qk_mma, qk_tiler, self.element_dtype, self.NUM_KV_STAGES)
        p_layout = sm100_utils.make_smem_layout_a(pv_mma, pv_tiler, self.element_dtype, 1)
        v_layout = sm100_utils.make_smem_layout_b(pv_mma, pv_tiler, self.element_dtype, self.NUM_KV_STAGES)
        if const_expr(cute.cosize(k_layout) != cute.cosize(v_layout)):
            raise RuntimeError("K-major and permuted V-major ring views must alias exactly")

        # Source-style peer exchange: each lane moves one aligned float4 and
        # every eight-lane 128B transaction touches each bank exactly once.
        p_exchange_layout = cute.make_layout(
            (4, self.B_TOPK // 8, 32, 4),
            stride=(32 * (self.B_TOPK // 2), 32 * 4, 4, 1),
        )
        row_layout = cute.make_layout((self.HALF_H,), stride=(1,))
        valid_layout = cute.make_layout((self.B_TOPK // 8, self.NUM_KV_STAGES), stride=(1, self.B_TOPK // 8))

        @cute.struct
        class SharedStorage:
            qk_mbar: cute.struct.MemRange[Int64, 2]
            p_mbar: cute.struct.MemRange[Int64, 2]
            o_mbar: cute.struct.MemRange[Int64, 2]
            tq_mbar: cute.struct.MemRange[Int64, 2]
            tout_mbar: cute.struct.MemRange[Int64, 2]
            out_scale_mbar: cute.struct.MemRange[Int64, 2]
            kv_mbar: cute.struct.MemRange[Int64, 2 * self.NUM_KV_STAGES]
            valid_mbar: cute.struct.MemRange[Int64, 2 * self.NUM_KV_STAGES]
            clc_mbar: cute.struct.MemRange[Int64, 2]
            clc_response: cute.struct.MemRange[Int32, 4]
            tmem_dealloc_mbar: Int64
            tmem_holding_buf: Int32
            sQ: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(q_layout)], 1024]
            sKV: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(k_layout)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(p_layout)], 1024]
            sPExchange: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(p_exchange_layout)], 16]
            sRowMax: cute.struct.Align[cute.struct.MemRange[Float32, self.WARPGROUP_SIZE], 16]
            sValid: cute.struct.Align[cute.struct.MemRange[Uint8, (self.B_TOPK // 8) * self.NUM_KV_STAGES], 16]
            sOutScale: cute.struct.Align[cute.struct.MemRange[Float32, self.HALF_H], 16]
            q_load_mbar: cute.struct.MemRange[Int64, 1]

        if const_expr(SharedStorage.size_in_bytes() > 232448):
            raise RuntimeError("head128 Prefill shared storage exceeds the SM100 per-block limit")
        self.shared_storage = SharedStorage
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams((2 * tq, 1, 1), self.cluster_shape_mnk)
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)

        self.kernel(
            q_tma,
            kv,
            indices,
            out_tma,
            max_logits,
            lse,
            lse_indexer,
            attn_sink,
            topk_length,
            Float32(softmax_scale),
            tma_atom_q,
            q_tma_bytes,
            tma_atom_o,
            tma_atom_kv,
            kv_tma_bytes,
            qk_mma,
            pv_mma,
            cluster_layout_vmnk,
            q_layout,
            q_tma_smem_layout,
            q_s2t_outer,
            o_tma_smem_layout,
            k_layout,
            p_layout,
            v_layout,
            p_exchange_layout,
            row_layout,
            valid_layout,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=(self.NUM_THREADS, 1, 1),
            cluster=self.cluster_shape_mnk,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q_tma: cute.Tensor,
        kv: cute.Tensor,
        indices: cute.Tensor,
        out_tma: cute.Tensor,
        max_logits: cute.Tensor,
        lse: cute.Tensor,
        lse_indexer: Optional[cute.Tensor],
        attn_sink: Optional[cute.Tensor],
        topk_length: Optional[cute.Tensor],
        softmax_scale: Float32,
        tma_atom_q: cute.CopyAtom,
        q_tma_bytes: cutlass.Constexpr[int],
        tma_atom_o: cute.CopyAtom,
        tma_atom_kv: cute.CopyAtom,
        kv_tma_bytes: cutlass.Constexpr[int],
        qk_mma: cute.TiledMma,
        pv_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        q_layout: cute.ComposedLayout,
        q_tma_smem_layout: cute.ComposedLayout,
        q_s2t_outer: cute.Layout,
        o_tma_smem_layout: cute.ComposedLayout,
        k_layout: cute.ComposedLayout,
        p_layout: cute.ComposedLayout,
        v_layout: cute.ComposedLayout,
        p_exchange_layout: cute.Layout,
        row_layout: cute.Layout,
        valid_layout: cute.Layout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        warpgroup_idx = tidx // self.WARPGROUP_SIZE
        thread_in_wg = tidx % self.WARPGROUP_SIZE
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_cluster_leader = cta_rank == Int32(0)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # Keep the pointer, rather than the local SharedStorage object, live
        # across the persistent loop for CuTe DSL 4.5 CFG flattening.
        q_load_mbar_ptr = storage.q_load_mbar.data_ptr()
        # PipelineUmmaAsync stores its single full barrier followed by its
        # single empty barrier.  Materialize the empty pointer before the
        # persistent CFG so CuTe DSL 4.5 does not need to carry the immutable
        # handle's pipeline object graph through the nested MMA loop.
        tq_empty_mbar_ptr = storage.tq_mbar.data_ptr() + 1
        o_mbar_ptr = storage.o_mbar.data_ptr()
        sKV_base_ptr = storage.sKV.data_ptr()
        kv_mbar_ptr = storage.kv_mbar.data_ptr()
        sQ_tma = storage.sQ.get_tensor(q_tma_smem_layout.outer, swizzle=q_tma_smem_layout.inner)
        sQ_s2t = storage.sQ.get_tensor(q_s2t_outer, swizzle=q_layout.inner)
        sO = storage.sQ.get_tensor(o_tma_smem_layout.outer, swizzle=o_tma_smem_layout.inner)
        sK_mma = storage.sKV.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sV_mma = storage.sKV.get_tensor(v_layout.outer, swizzle=v_layout.inner)
        sP_mma = storage.sP.get_tensor(p_layout.outer, swizzle=p_layout.inner)
        sP_slice = sP_mma[(None, None), 0, None, 0]
        sP = cute.composition(sP_slice, cute.make_layout((self.HALF_H, self.B_TOPK)))
        sPExchange = storage.sPExchange.get_tensor(p_exchange_layout)
        sPExchangeLinear = storage.sPExchange.get_tensor(cute.make_layout((4 * 32 * (self.B_TOPK // 2),), stride=(1,)))
        sRowMax = storage.sRowMax.get_tensor(cute.make_layout((self.WARPGROUP_SIZE,), stride=(1,)))
        sValid = storage.sValid.get_tensor(valid_layout)
        sValid_words = cute.make_tensor(
            cute.recast_ptr(storage.sValid.data_ptr(), dtype=Uint32),
            cute.make_layout((self.B_TOPK // 32, self.NUM_KV_STAGES), stride=(1, self.B_TOPK // 32)),
        )
        sOutScale = storage.sOutScale.get_tensor(row_layout)

        qk_producer, qk_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.WARPGROUP_SIZE),
            barrier_storage=storage.qk_mbar.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.WARPGROUP_SIZE),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            barrier_storage=storage.p_mbar.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.WARPGROUP_SIZE),
            barrier_storage=o_mbar_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        # S2T and QK are both tcgen05 producers.  PipelineUmmaAsync supplies
        # the S2T -> QK full barrier; after the final QK, the MMA warp commits
        # directly to the pipeline's empty barrier so the next S2T may reuse Q.
        tq_producer, tq_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            barrier_storage=storage.tq_mbar.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        tq_alias_reader = tq_consumer.clone()
        # Cross-query output hand-off.  The final PV commit publishes TMEM O
        # to WG0 in both CTAs; WG0 releases the slot after all TMEM loads.  A
        # local async pipeline protects the single shared normalization-scale
        # surface while allowing the next query's softmax to run independently.
        tout_producer, tout_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.WARPGROUP_SIZE),
            barrier_storage=storage.tout_mbar.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        out_scale_producer, out_scale_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.HALF_H),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.WARPGROUP_SIZE),
            barrier_storage=storage.out_scale_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        # One elected producer lane per CTA drives the ring.  CTA0 expects the
        # combined 64x512 16-bit transaction count; group-2 gathers in both CTAs
        # credit that full barrier.  PV multicasts the empty signal only after
        # it has consumed the aliased K/V stage.
        kv_producer, kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.NUM_KV_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=kv_tma_bytes,
            barrier_storage=kv_mbar_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        kv_releaser = kv_consumer.clone()
        kv_softmax_reader = kv_consumer.clone()
        # The warp-9 validity role publishes eight packed bytes per KV tile.
        # All softmax threads consume the mask before that stage can be
        # reused, independently of the much larger TMA KV transaction ring.
        valid_producer, valid_consumer = pipeline.PipelineAsync.create(
            num_stages=self.NUM_KV_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.B_TOPK // 8),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.WARPGROUP_SIZE),
            barrier_storage=storage.valid_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()

        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar.data_ptr(),
            num_stages=self.num_clc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.NUM_THREADS),
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        clc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_clc_stages)
        clc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stages)

        # Use CUTLASS's canonical 2-CTA allocator/deallocator protocol.  CUDA
        # 13.2 racecheck currently attributes the hardware allocator-result
        # write and its post-NamedBarrier read as a RAW hazard in this helper's
        # shared holding buffer; memcheck and synccheck are clean, and the same
        # protocol is used by the repository's 2-CTA SM100 SDPA kernel.
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.WG_MMA * 4,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        tmem.allocate(self.TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        qk_thr_mma = qk_mma.get_slice(cta_rank)
        pv_thr_mma = pv_mma.get_slice(cta_rank)
        tKrK = qk_thr_mma.make_fragment_B(sK_mma)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.H_Q, self.B_TOPK))
        tStS_fragment = qk_thr_mma.make_fragment_C(qk_acc_shape)
        tStS = cute.make_tensor(tmem_ptr + self.TMEM_P_OFFSET, tStS_fragment.layout)
        tPrP = pv_thr_mma.make_fragment_A(sP_mma)
        tVrV = pv_thr_mma.make_fragment_B(sV_mma)
        pv_acc_shape = pv_thr_mma.partition_shape_C((self.H_Q, self.D_V))
        tOtO_fragment = pv_thr_mma.make_fragment_C(pv_acc_shape)

        q_tmem_shape = qk_thr_mma.partition_shape_A((self.HALF_H, self.HALF_D))
        tQtQ_fragment = qk_thr_mma.make_fragment_A(q_tmem_shape)
        # TMEM offsets are expressed in FP32 columns, while the A fragment is
        # 16-bit-addressed, so the Q base must be scaled by the width ratio.
        q_tmem_ptr = cute.recast_ptr(tmem_ptr, dtype=self.element_dtype) + Int32(self.TMEM_Q_OFFSET * (Float32.width // self.element_dtype.width))
        tQtQ = cute.make_tensor(q_tmem_ptr, tQtQ_fragment.layout)
        compact_sQ = cute.filter_zeros(sQ_s2t)
        compact_tQ = cute.filter_zeros(tQtQ)
        q_s2t_atom = cute.make_copy_atom(tcgen05.Cp128x256bOp(tcgen05.CtaGroup.TWO), self.element_dtype)
        q_s2t_copy = tcgen05.make_s2t_copy(q_s2t_atom, compact_tQ)
        q_s2t_thr = q_s2t_copy.get_slice(0)
        q_s2t_src = tcgen05.get_s2t_smem_desc_tensor(q_s2t_copy, q_s2t_thr.partition_S(compact_sQ))
        q_s2t_dst = q_s2t_thr.partition_D(compact_tQ)

        if warp_idx == Int32(0):
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_o)
        if warp_idx == Int32(self.WG_KV * 4):
            cpasync.prefetch_descriptor(tma_atom_kv)
        _initialize_q_load_mbarrier(tidx, q_load_mbar_ptr)
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
        cute.arch.sync_threads()
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # Seed the persistent O-completion ring with one empty token per CTA.
        # Advance the sole MMA producer by the same logical token so its first
        # real PV waits until softmax has consumed and released the seed.
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(o_mbar_ptr)
        if warp_idx == Int32(self.WG_MMA * 4) and is_cluster_leader:
            o_producer.advance()

        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            storage.clc_response.data_ptr(),
        )
        work_tile = tile_sched.initial_work_tile_info()
        head_global_base = cta_rank * self.HALF_H

        # Keep the CTA0 MMA warp in one warp-uniform persistent loop.  Leaving
        # this role inside the all-thread loop below makes ptxas carry a
        # divergent reconvergence region across the outer backedge; although
        # PTX contains one 24-instruction MMA body, SASS then tail-duplicates it
        # into first-trip and steady-state copies.  Consuming the same CLC
        # response here preserves the 1024-thread empty-barrier arrival count.
        cluster_warp_role = cute.arch.make_warp_uniform(warp_idx + cta_rank * Int32(self.NUM_WARPS))
        if cluster_warp_role == Int32(self.WG_MMA * 4):
            while work_tile.is_valid_tile:
                query_idx = work_tile.tile_idx[0] // 2
                physical_topk = Int32(cute.size(indices.shape[1]))
                effective_topk = physical_topk
                if const_expr(topk_length is not None):
                    effective_topk = Int32(topk_length[query_idx])
                effective_topk = effective_topk if effective_topk > Int32(0) else Int32(0)
                effective_topk = effective_topk if effective_topk < physical_topk else physical_topk
                num_tiles = (effective_topk + Int32(self.B_TOPK - 1)) // Int32(self.B_TOPK)
                num_tiles = num_tiles if num_tiles > Int32(0) else Int32(1)

                tq_handle = tq_consumer.wait_and_advance()
                tile_idx = Int32(0)
                while tile_idx <= num_tiles:
                    if tile_idx < num_tiles:
                        kv_handle = kv_consumer.wait_and_advance()
                        stage = kv_handle.index
                        qk_handle = qk_producer.acquire_and_advance()
                        tcgen05_fence_after_thread_sync()
                        qk_mma.set(tcgen05.Field.ACCUMULATE, False)
                        tK = tKrK[None, None, None, stage]
                        for kphase in cutlass.range(cute.size(tK, mode=[2]), unroll_full=True):
                            cute.gemm(qk_mma, tStS, tQtQ[None, None, kphase], tK[None, None, kphase], tStS)
                            qk_mma.set(tcgen05.Field.ACCUMULATE, True)
                        qk_handle.commit()
                        if tile_idx == num_tiles - Int32(1):
                            # The final QK is the last consumer of persistent
                            # tQ.  Commit its completion directly to tq_empty;
                            # a normal async-thread release would race the UMMA.
                            with cute.arch.elect_one():
                                tcgen05.commit(tq_empty_mbar_ptr, Int32(0b11), tcgen05.CtaGroup.TWO)

                    if tile_idx > Int32(0):
                        # The releaser trails the QK consumer by one tile.  Its
                        # multicast empty signal is committed after PV, so the
                        # producer cannot overwrite a K/V stage while V is live.
                        # The first PV also acquires the cross-query O slot;
                        # QK(0) is therefore free to overlap WG0's prior output
                        # drain before O is overwritten.
                        if tile_idx == Int32(1):
                            tout_producer.acquire()
                        release_stage = kv_releaser.current_handle().index
                        p_handle = p_consumer.wait_and_advance()
                        o_handle = o_producer.acquire_and_advance()
                        tcgen05_fence_after_thread_sync()
                        tP = tPrP[None, None, None, 0]
                        pv_mma.set(tcgen05.Field.ACCUMULATE, tile_idx != Int32(1))
                        tO = cute.make_tensor(tmem_ptr + Int32(self.TMEM_O_OFFSET), tOtO_fragment.layout)
                        tV = tVrV[None, None, None, release_stage]
                        for kphase in cutlass.range(cute.size(tV, mode=[2]), unroll_full=True):
                            cute.gemm(pv_mma, tO, tP[None, None, kphase], tV[None, None, kphase], tO)
                            pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                        o_handle.commit()
                        if tile_idx == num_tiles:
                            tout_producer.commit()
                            tout_producer.advance()
                        p_handle.release()
                        kv_releaser.release()
                        kv_releaser.advance()
                    tile_idx += Int32(1)

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

        # Keep WG0's Q load and one-query-delayed O epilogue in a dedicated
        # persistent loop.  The remaining roles no longer build Q/O tensor-map
        # coordinates or cross this large branch on every CLC assignment.
        if warpgroup_idx == Int32(self.WG_QO):
            q_load_phase = Int32(0)
            previous_query_idx = Int32(-1)
            while work_tile.is_valid_tile:
                query_idx = work_tile.tile_idx[0] // 2
                gQ = cute.local_tile(
                    q_tma,
                    (self.QK_K_TILE, self.HALF_H, 2, self.HALF_D // self.QK_K_TILE),
                    (0, cta_rank, 0, 0, query_idx),
                )
                load_q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_q,
                    0,
                    cute.make_layout(1),
                    gQ,
                    sQ_tma,
                    single_stage=True,
                )
                if warp_idx == Int32(0):
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    if is_cluster_leader:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(q_load_mbar_ptr, 2 * q_tma_bytes)
                    q_load_barrier = cute.arch.map_dsmem_ptr(q_load_mbar_ptr, Int32(0))
                    load_q(tma_bar_ptr=q_load_barrier)
                    if is_cluster_leader:
                        cute.arch.mbarrier_wait(q_load_mbar_ptr, q_load_phase)
                        q_load_phase ^= Int32(1)
                        tq_handle = tq_producer.acquire_and_advance()
                        tcgen05_fence_after_thread_sync()
                        for q_tile in cutlass.range_constexpr(self.D_QK // (2 * self.QK_K_TILE)):
                            for subtile in cutlass.range_constexpr(4):
                                dst_phase = q_tile * 4 + subtile
                                cute.copy(
                                    q_s2t_copy,
                                    q_s2t_src[None, None, None, subtile, q_tile],
                                    q_s2t_dst[None, None, None, dst_phase],
                                )
                        tq_handle.commit()

                if previous_query_idx >= Int32(0):
                    scale_handle = out_scale_consumer.wait_and_advance()
                    output_scale = Float32(sOutScale[thread_in_wg % Int32(self.HALF_H)])
                    scale_handle.release()

                    tout_handle = tout_consumer.wait_and_advance()
                    tcgen05_fence_after_thread_sync()
                    first_output_ssa = self._prepare_output_first_chunk(
                        tOtO_fragment,
                        tmem_ptr,
                        output_scale,
                        thread_in_wg,
                    )
                    tq_alias_reader.wait_and_advance()
                    self._store_output(
                        sO,
                        tOtO_fragment,
                        tmem_ptr,
                        output_scale,
                        first_output_ssa,
                        thread_in_wg,
                        tout_handle,
                    )
                    cute.arch.fence_view_async_shared()
                    self.qo_epilogue_barrier.arrive_and_wait()
                    if warp_idx == Int32(0):
                        gO = cute.local_tile(
                            out_tma,
                            (64, self.HALF_H, self.D_V // 64),
                            (0, cta_rank, 0, previous_query_idx, 0),
                        )
                        store_o, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_o,
                            0,
                            cute.make_layout(1),
                            sO,
                            gO,
                            single_stage=True,
                        )
                        store_o()
                        cute.arch.cp_async_bulk_commit_group()
                else:
                    tq_alias_reader.wait_and_advance()
                previous_query_idx = query_idx

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            if previous_query_idx >= Int32(0):
                if warp_idx == Int32(0):
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.qo_epilogue_barrier.arrive_and_wait()

                scale_handle = out_scale_consumer.wait_and_advance()
                output_scale = Float32(sOutScale[thread_in_wg % Int32(self.HALF_H)])
                scale_handle.release()

                tout_handle = tout_consumer.wait_and_advance()
                tcgen05_fence_after_thread_sync()
                first_output_ssa = self._prepare_output_first_chunk(
                    tOtO_fragment,
                    tmem_ptr,
                    output_scale,
                    thread_in_wg,
                )
                self._store_output(
                    sO,
                    tOtO_fragment,
                    tmem_ptr,
                    output_scale,
                    first_output_ssa,
                    thread_in_wg,
                    tout_handle,
                )
                cute.arch.fence_view_async_shared()
                self.qo_epilogue_barrier.arrive_and_wait()
                if warp_idx == Int32(0):
                    gO = cute.local_tile(
                        out_tma,
                        (64, self.HALF_H, self.D_V // 64),
                        (0, cta_rank, 0, previous_query_idx, 0),
                    )
                    store_o, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_o,
                        0,
                        cute.make_layout(1),
                        sO,
                        gO,
                        single_stage=True,
                    )
                    store_o()
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

        # Drain CLC responses for warps with no compute role before entering the
        # all-role loop.  Once their local work tile becomes invalid they skip
        # the large CFG below naturally.  Keeping this as a tiny, separate loop
        # avoids carrying a composite role predicate around the hot loop while
        # removing all of its address and role-dispatch work from four warps.
        is_empty_cluster_warp = (
            (cluster_warp_role == Int32(11))
            | (cluster_warp_role == Int32(self.NUM_WARPS + self.WG_MMA * 4))
            | (cluster_warp_role == Int32(self.NUM_WARPS + self.CLC_WARP))
            | (cluster_warp_role == Int32(self.NUM_WARPS + self.CLC_WARP + 1))
        )
        if is_empty_cluster_warp:
            while work_tile.is_valid_tile:
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

        while work_tile.is_valid_tile:
            query_idx = work_tile.tile_idx[0] // 2
            # The current response has already been decoded into registers, so
            # let the dedicated scheduler warp issue the next CLC query before
            # starting Q/KV work.  Its one-stage slot was released at the end
            # of the preceding iteration and can overlap this entire query.
            if warp_idx == self.CLC_WARP and is_cluster_leader:
                clc_pipeline.producer_acquire(clc_producer_state)
                tile_sched.advance_to_next_work(clc_pipeline.producer_get_barrier(clc_producer_state))
                clc_producer_state.advance()

            physical_topk = Int32(cute.size(indices.shape[1]))
            effective_topk = physical_topk
            if const_expr(topk_length is not None):
                effective_topk = Int32(topk_length[query_idx])
            effective_topk = effective_topk if effective_topk > Int32(0) else Int32(0)
            effective_topk = effective_topk if effective_topk < physical_topk else physical_topk
            num_tiles = (effective_topk + Int32(self.B_TOPK - 1)) // Int32(self.B_TOPK)
            num_tiles = num_tiles if num_tiles > Int32(0) else Int32(1)

            # Four independent role loops implement the source software
            # pipeline.  WG1 may gather four KV tiles ahead; warp 8 issues
            # QK(k) before PV(k-1); WG3 consumes scores and publishes P.  The
            # only query-wide synchronization is after all roles drain.
            if warp_idx == Int32(self.WG_MMA * 4 + 1):
                if lane_idx < Int32(self.B_TOPK // 8):
                    valid_tile_idx = Int32(0)
                    while valid_tile_idx < num_tiles:
                        valid_prod_handle = valid_producer.acquire_and_advance()
                        index_slot = valid_tile_idx * Int32(self.B_TOPK) + lane_idx * Int32(8)
                        r_indices = _ldg_indices_256(indices, query_idx, index_slot, False)

                        valid_mask = Uint32(0)
                        slot_base = valid_tile_idx * Int32(self.B_TOPK) + lane_idx * Int32(8)
                        for elem in cutlass.range_constexpr(8):
                            token = Int32(r_indices[elem])
                            slot = slot_base + Int32(elem)
                            is_valid = slot < effective_topk and token >= Int32(0) and Int64(token) < Int64(kv.shape[0])
                            valid_mask = valid_mask | ((Uint32(1) << elem) if is_valid else Uint32(0))
                        sValid[lane_idx, valid_prod_handle.index] = Uint8(valid_mask)
                        cute.arch.fence_view_async_shared()
                        valid_prod_handle.commit()
                        valid_tile_idx += Int32(1)

            if warpgroup_idx == self.WG_KV:
                tile_idx = Int32(0)
                while tile_idx < num_tiles:
                    kv_handle = kv_producer.current_handle()
                    kv_prod_stage = kv_handle.index
                    logical_tile_is_full = Int32(1)
                    if const_expr(topk_length is not None):
                        logical_tile_is_full = cute.arch.make_warp_uniform(Int32((tile_idx + Int32(1)) * Int32(self.B_TOPK) <= effective_topk))
                    # Keep direct index loads and gather4 coordinates in one
                    # selected lane per producer warp.  Pipeline state advances
                    # below the elected child so its SSA tuple remains valid.
                    with cute.arch.elect_one():
                        local_warp = warp_idx - Int32(self.WG_KV * 4)
                        index_slot = tile_idx * Int32(self.B_TOPK) + local_warp * Int32(8)
                        index_lo = _ldg_indices_256(indices, query_idx, index_slot, True)
                        index_hi = _ldg_indices_256(indices, query_idx, index_slot + Int32(self.B_TOPK // 2), True)
                        empty_phase = Int32(1) ^ ((kv_handle.count // Int32(self.NUM_KV_STAGES)) & Int32(1))
                        cute.arch.mbarrier_wait(
                            kv_mbar_ptr + Int32(self.NUM_KV_STAGES) + kv_prod_stage,
                            empty_phase,
                        )
                        if is_cluster_leader and local_warp == Int32(0):
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                kv_mbar_ptr + kv_prod_stage,
                                kv_tma_bytes,
                            )

                        if const_expr(topk_length is not None):
                            if logical_tile_is_full:
                                self._load_indices_and_gather_half_kv_tma(
                                    sKV_base_ptr,
                                    index_slot,
                                    index_lo,
                                    index_hi,
                                    tma_atom_kv,
                                    kv_mbar_ptr + kv_prod_stage,
                                    effective_topk,
                                    kv_prod_stage,
                                    cta_rank,
                                    local_warp,
                                    False,
                                )
                            else:
                                self._load_indices_and_gather_half_kv_tma(
                                    sKV_base_ptr,
                                    index_slot,
                                    index_lo,
                                    index_hi,
                                    tma_atom_kv,
                                    kv_mbar_ptr + kv_prod_stage,
                                    effective_topk,
                                    kv_prod_stage,
                                    cta_rank,
                                    local_warp,
                                    True,
                                )
                        else:
                            self._load_indices_and_gather_half_kv_tma(
                                sKV_base_ptr,
                                index_slot,
                                index_lo,
                                index_hi,
                                tma_atom_kv,
                                kv_mbar_ptr + kv_prod_stage,
                                effective_topk,
                                kv_prod_stage,
                                cta_rank,
                                local_warp,
                                False,
                            )
                    kv_producer.advance()
                    tile_idx += Int32(1)

            if warpgroup_idx == self.WG_SOFTMAX:
                # Every softmax thread owns one 32-token half-row and carries
                # its online state in registers.
                # The two token halves are reduced only once in the epilogue;
                # putting mi/li/real_max through shared memory on every tile
                # creates a serialized long-scoreboard dependency chain.
                scale_log2 = softmax_scale * Float32(self.LOG2_E)
                softmax_mi = Float32(self.MAX_INIT_LOG2)
                softmax_li = Float32(0.0)
                softmax_real_max = -Float32.inf
                softmax_mi_indexer = Float32(self.MAX_INIT_LOG2)
                softmax_li_indexer = Float32(0.0)
                tile_idx = Int32(0)
                while tile_idx < num_tiles:
                    valid_handle = valid_consumer.wait_and_advance()
                    valid_word = Uint32(sValid_words[thread_in_wg // Int32(64), valid_handle.index])
                    valid_handle.release()
                    qk_handle = qk_consumer.wait_and_advance()
                    stage = kv_softmax_reader.current_handle().index

                    tcgen05_fence_after_thread_sync()
                    softmax_mi, softmax_li, softmax_real_max, scale_for_old, should_rescale, probability_ssa = self._scores_tmem_to_smem(
                        qk_thr_mma,
                        tStS,
                        sPExchange,
                        sRowMax,
                        qk_handle,
                        valid_word,
                        thread_in_wg,
                        scale_log2,
                        softmax_mi,
                        softmax_li,
                        softmax_real_max,
                    )

                    # Compute the probability fragment first, then rendezvous
                    # with PV(k-1)
                    # immediately before publishing S(k).  Keep the pipeline
                    # participants in this parent region so their mutable
                    # state remains well-defined for CuTeDSL lowering.
                    o_handle = o_consumer.wait_and_advance()
                    tcgen05_fence_after_thread_sync()
                    o_handle.release()
                    p_handle = p_producer.acquire_and_advance()

                    r_probability = cute.make_rmem_tensor((self.B_TOPK // 2,), self.element_dtype)
                    r_probability.store(probability_ssa)
                    local_softmax_warp = thread_in_wg // Int32(32)
                    softmax_lane = thread_in_wg % Int32(32)
                    softmax_head = (local_softmax_warp & Int32(1)) * Int32(32) + softmax_lane
                    r_probability_vectors = cute.flat_divide(r_probability, (8,))
                    sP_vectors = cute.flat_divide(sP[softmax_head, None], (8,))
                    if const_expr(self.indexer_topk > 0):
                        half_vector_offset = (local_softmax_warp // Int32(2)) * Int32(self.B_TOPK // 16)
                        for vector in cutlass.range_constexpr((self.B_TOPK // 2) // 8):
                            cute.autovec_copy(
                                r_probability_vectors[None, vector],
                                sP_vectors[None, half_vector_offset + Int32(vector)],
                            )
                    else:
                        if local_softmax_warp < Int32(2):
                            for vector in cutlass.range_constexpr((self.B_TOPK // 2) // 8):
                                cute.autovec_copy(r_probability_vectors[None, vector], sP_vectors[None, vector])
                        else:
                            for vector in cutlass.range_constexpr((self.B_TOPK // 2) // 8):
                                cute.autovec_copy(r_probability_vectors[None, vector], sP_vectors[None, vector + self.B_TOPK // 16])

                    if const_expr(self.indexer_topk > 0):
                        if tile_idx == Int32(self.indexer_tile):
                            # Keep both half-row snapshots in registers.  They
                            # are peer-reduced together with the final li below,
                            # removing two WG-wide barriers from the tile loop.
                            softmax_mi_indexer = softmax_mi
                            softmax_li_indexer = softmax_li

                    if tile_idx > Int32(0):
                        if should_rescale:
                            self._rescale_o_tmem(tOtO_fragment, tmem_ptr, scale_for_old, thread_in_wg)

                    cute.arch.fence_view_async_shared()
                    p_handle.commit()
                    kv_softmax_reader.advance()
                    tile_idx += Int32(1)

                # Reduce the two 32-token li partials after the last tile,
                # exactly once per query.  Paired warps have identical mi and
                # real_max, so only li (and optional indexer li) needs adding.
                # Drain the final score-exchange reads before reusing their
                # aliased surface as reduction scratch.
                self.softmax_wg_barrier.arrive_and_wait()
                sum_offset = Int32(self.WARPGROUP_SIZE)
                indexer_sum_offset = Int32(2 * self.WARPGROUP_SIZE)
                sPExchangeLinear[sum_offset + (thread_in_wg ^ Int32(64))] = softmax_li
                if const_expr(self.indexer_topk > 0):
                    sPExchangeLinear[indexer_sum_offset + (thread_in_wg ^ Int32(64))] = softmax_li_indexer
                self.softmax_wg_barrier.arrive_and_wait()
                softmax_li += Float32(sPExchangeLinear[sum_offset + thread_in_wg])
                if const_expr(self.indexer_topk > 0):
                    softmax_li_indexer += Float32(sPExchangeLinear[indexer_sum_offset + thread_in_wg])
                self.softmax_wg_barrier.arrive_and_wait()

                local_softmax_warp = thread_in_wg // Int32(32)
                if local_softmax_warp < Int32(2):
                    scale_handle = out_scale_producer.acquire_and_advance()
                    self._finalize_stats(
                        max_logits,
                        lse,
                        lse_indexer,
                        attn_sink,
                        query_idx,
                        head_global_base,
                        softmax_mi,
                        softmax_li,
                        softmax_real_max,
                        softmax_mi_indexer,
                        softmax_li_indexer,
                        sOutScale,
                        scale_handle,
                        thread_in_wg,
                    )

            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()

        if warp_idx == self.CLC_WARP and is_cluster_leader:
            clc_pipeline.producer_tail(clc_producer_state)
        if warp_idx == Int32(self.WG_MMA * 4) and is_cluster_leader:
            qk_producer.tail()
            o_producer.tail()
            tout_producer.tail()
        if warp_idx == Int32(0) and is_cluster_leader:
            tq_producer.tail()
        if warp_idx == Int32(self.WG_KV * 4):
            kv_producer.tail()
        if warp_idx == Int32(self.WG_MMA * 4 + 1) and lane_idx < Int32(self.B_TOPK // 8):
            valid_producer.tail()
        if warpgroup_idx == self.WG_SOFTMAX:
            # Drain the final query's last PV completion.  Earlier completions
            # flow into tile 0 of the following query.
            o_handle = o_consumer.wait_and_advance()
            tcgen05_fence_after_thread_sync()
            o_handle.release()
            p_producer.tail()
        if warpgroup_idx == self.WG_SOFTMAX and thread_in_wg < Int32(self.HALF_H):
            out_scale_producer.tail()

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _load_indices_and_gather_half_kv_tma(
        self,
        sKV_base_ptr,
        index_slot,
        index_lo,
        index_hi,
        tma_atom_kv,
        mbar_ptr,
        effective_topk,
        stage,
        cta_rank,
        local_warp,
        sanitize_logical_tail: cutlass.Constexpr[bool],
    ):
        """Gather one selected lane's 16 prefetched rows."""

        stage_offset = stage * Int32(self.B_TOPK * self.HALF_D)
        stage_ptr = sKV_base_ptr + cute.assume(stage_offset, divby=64)

        # Use four row4 groups per producer warp while keeping all 16 gather
        # coordinates in the selected lane's registers.
        for row_group in cutlass.range_constexpr(4):
            index_base = (row_group % 2) * 4
            if const_expr(row_group < 2):
                row0 = Int32(index_lo[index_base])
                row1 = Int32(index_lo[index_base + 1])
                row2 = Int32(index_lo[index_base + 2])
                row3 = Int32(index_lo[index_base + 3])
            else:
                row0 = Int32(index_hi[index_base])
                row1 = Int32(index_hi[index_base + 1])
                row2 = Int32(index_hi[index_base + 2])
                row3 = Int32(index_hi[index_base + 3])
            if const_expr(sanitize_logical_tail):
                # Range-invalid coordinates are already zero-filled by TMA and
                # masked by warp 9.  Redirect only logically excluded rows so
                # ignored but otherwise valid NaN rows are never fetched.
                row_slot = index_slot + Int32((row_group // 2) * (self.B_TOPK // 2) + (row_group % 2) * 4)
                row0 = row0 if row_slot < effective_topk else Int32(-1)
                row1 = row1 if row_slot + Int32(1) < effective_topk else Int32(-1)
                row2 = row2 if row_slot + Int32(2) < effective_topk else Int32(-1)
                row3 = row3 if row_slot + Int32(3) < effective_topk else Int32(-1)
            row_base = local_warp * Int32(8) + Int32((row_group // 2) * 32) + Int32((row_group % 2) * 4)
            for local_col in cutlass.range_constexpr(self.HALF_D // 64):
                smem_offset = row_base * Int32(64) + Int32(local_col * self.B_TOPK * 64)
                dst_ptr = stage_ptr + cute.assume(smem_offset, divby=64)
                tma_gather4_cta2_cta0_mbar(
                    tma_atom_kv,
                    dst_ptr,
                    mbar_ptr,
                    cta_rank * Int32(self.HALF_D) + Int32(local_col * 64),
                    row0,
                    row1,
                    row2,
                    row3,
                )

    @cute.jit
    def _scores_tmem_to_smem(
        self,
        qk_thr_mma,
        tStS,
        sPExchange,
        sRowMax,
        qk_handle,
        valid_word: Uint32,
        thread_in_wg,
        scale_log2,
        old_mi,
        old_li,
        old_real_max,
    ):
        tS_slice = tStS[(None, None), 0, 0]
        # The N=128 instruction returns two 32-value partials per thread.  Split
        # the T2R partition before loading so the merged score remains a
        # 32-value fragment instead of a monolithic 64-value fragment.
        cS = qk_thr_mma.partition_C(cute.make_identity_tensor((self.H_Q, 2 * self.B_TOPK)))
        cS_slice = cS[(None, None), 0, 0]
        load_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32)
        tiled_load = tcgen05.make_tmem_copy(load_atom, tS_slice)
        thr_load = tiled_load.get_slice(thread_in_wg)
        t_coord = thr_load.partition_D(cS_slice)
        t_tmem = thr_load.partition_S(tS_slice)
        # The outermost T2R mode is the two 32-score N subtiles.  The source
        # partition also carries the 32-lane TMEM copy mode, so its total size
        # is 32x larger than the coordinate/register partition.  Keep this
        # check here: flattening the whole source partition would silently
        # treat that copy mode as score values and recreate the 64-register
        # fragment this path is intended to avoid.
        # Keep these compile-time checks independent.  CuTe DSL 4.5 expands a
        # chained ``or`` through its BoolOp preprocessor and creates a multi-MB
        # Python code object for this helper, making source-location lookup
        # superlinear during lowering.  Separate checks preserve the invariant
        # without changing generated device code.
        if const_expr(cute.rank(t_coord.shape) != 3):
            raise RuntimeError("unexpected H128 score T2R coordinate rank")
        if const_expr(cute.size(t_coord, mode=[0]) != self.B_TOPK // 2):
            raise RuntimeError("unexpected H128 score T2R coordinate mode-0 size")
        if const_expr(cute.size(t_coord, mode=[1]) != 1):
            raise RuntimeError("unexpected H128 score T2R coordinate mode-1 size")
        if const_expr(cute.size(t_coord, mode=[2]) != 2):
            raise RuntimeError("unexpected H128 score T2R coordinate mode-2 size")
        if const_expr(cute.rank(t_tmem.shape) != 3):
            raise RuntimeError("unexpected H128 score T2R TMEM rank")
        if const_expr(cute.size(t_tmem, mode=[0]) != 32 * (self.B_TOPK // 2)):
            raise RuntimeError("unexpected H128 score T2R TMEM mode-0 size")
        if const_expr(cute.size(t_tmem, mode=[1]) != 1):
            raise RuntimeError("unexpected H128 score T2R TMEM mode-1 size")
        if const_expr(cute.size(t_tmem, mode=[2]) != 2):
            raise RuntimeError("unexpected H128 score T2R TMEM mode-2 size")
        r_left = cute.make_rmem_tensor(t_coord[None, None, 0].shape, Float32)
        r_right = cute.make_rmem_tensor(t_coord[None, None, 0].shape, Float32)

        local_warp = thread_in_wg // Int32(32)
        lane = thread_in_wg % Int32(32)

        # Indexer-enabled instances have enough tiles to amortize the dynamic
        # TMEM-half address.  Load the owned half into r_left and its peer into
        # r_right up front so the exchange below stays straight-line and does
        # not materialize a 32-value SSA select at the warp-role merge.
        if const_expr(self.indexer_topk > 0):
            owned_half = local_warp // Int32(2)
            peer_half = owned_half ^ Int32(1)
            cute.copy(tiled_load, t_tmem[None, None, owned_half], r_left)
            cute.copy(tiled_load, t_tmem[None, None, peer_half], r_right)
        else:
            # Keep tcgen05 loads out of runtime control flow on the low-
            # pressure indexer-free specialization.
            cute.copy(tiled_load, t_tmem[None, None, 0], r_left)
            cute.copy(tiled_load, t_tmem[None, None, 1], r_right)
        cute.arch.fence_view_async_tmem_load()

        # The score slot is dead once both TMEM fragments are resident in
        # registers.  Release it before mask/reduce/exp so QK(k+1) can overlap
        # the remainder of softmax(k), matching the source pipeline.
        tcgen05_fence_before_thread_sync()
        qk_handle.release()

        # The indexer-free specialization also serves one-tile requests.  Its
        # lower-pressure mask-before-add form is faster for that short path;
        # indexer-enabled kernels have at least four tiles and benefit from
        # masking only the owned half after the peer reduction below.
        if const_expr(self.indexer_topk == 0):
            for i in cutlass.range_constexpr(self.B_TOPK // 2):
                invalid = (valid_word & (Uint32(1) << i)) == Uint32(0)
                mask_left = invalid and local_warp < Int32(2)
                mask_right = invalid and local_warp >= Int32(2)
                r_left[i] = -Float32.inf if mask_left else r_left[i]
                r_right[i] = -Float32.inf if mask_right else r_right[i]

        # Exchange the peer partial as eight aligned float4 vectors.  The
        # (warp, vector, lane, element) layout avoids the lane*32 same-bank
        # pattern of the former scalar score surface.
        half_linear_layout = cute.make_layout((self.B_TOPK // 2,), stride=(1,))
        r_left_linear = cute.make_tensor(r_left.iterator, half_linear_layout)
        r_right_linear = cute.make_tensor(r_right.iterator, half_linear_layout)
        r_left_vectors = cute.flat_divide(r_left_linear, (4,))
        r_right_vectors = cute.flat_divide(r_right_linear, (4,))
        peer_warp = local_warp ^ Int32(2)
        if const_expr(self.indexer_topk > 0):
            for vector in cutlass.range_constexpr((self.B_TOPK // 2) // 4):
                cute.autovec_copy(r_right_vectors[None, vector], sPExchange[peer_warp, vector, lane, None])
        else:
            if local_warp < Int32(2):
                for vector in cutlass.range_constexpr((self.B_TOPK // 2) // 4):
                    cute.autovec_copy(r_right_vectors[None, vector], sPExchange[peer_warp, vector, lane, None])
            else:
                for vector in cutlass.range_constexpr((self.B_TOPK // 2) // 4):
                    cute.autovec_copy(r_left_vectors[None, vector], sPExchange[peer_warp, vector, lane, None])
        self._softmax_pair_sync(local_warp)

        if const_expr(self.indexer_topk > 0):
            for vector in cutlass.range_constexpr((self.B_TOPK // 2) // 4):
                cute.autovec_copy(sPExchange[local_warp, vector, lane, None], r_right_vectors[None, vector])
        else:
            if local_warp < Int32(2):
                for vector in cutlass.range_constexpr((self.B_TOPK // 2) // 4):
                    cute.autovec_copy(sPExchange[local_warp, vector, lane, None], r_right_vectors[None, vector])
            else:
                for vector in cutlass.range_constexpr((self.B_TOPK // 2) // 4):
                    cute.autovec_copy(sPExchange[local_warp, vector, lane, None], r_left_vectors[None, vector])
        for i in cutlass.range_constexpr(0, self.B_TOPK // 2, 2):
            r_left[i], r_left[i + 1] = cute.arch.add_packed_f32x2(
                (r_left[i], r_left[i + 1]),
                (r_right[i], r_right[i + 1]),
            )

        if const_expr(self.indexer_topk > 0):
            # Each warp now owns one fully reduced 32-token half.  Applying
            # the packed validity word once here is equivalent to masking its
            # local partial before the peer add (invalid gathers are
            # zero-filled), while avoiding a second 32-element select stream
            # for the unowned half.
            for i in cutlass.range_constexpr(self.B_TOPK // 2):
                invalid = (valid_word & (Uint32(1) << i)) == Uint32(0)
                r_left[i] = -Float32.inf if invalid else r_left[i]

        half_raw_max = -Float32.inf
        for i in cutlass.range_constexpr(self.B_TOPK // 2):
            if const_expr(self.indexer_topk > 0):
                half_raw_max = fmax_ftz_nonan(half_raw_max, r_left[i])
            else:
                half_raw_max = r_left[i] if r_left[i] > half_raw_max else half_raw_max
        half_max = half_raw_max * scale_log2
        sRowMax[thread_in_wg] = half_max
        self._softmax_pair_sync(local_warp)

        peer_half_max = Float32(sRowMax[thread_in_wg ^ Int32(64)])
        if const_expr(self.indexer_topk > 0):
            tile_max = fmax_ftz_nonan(peer_half_max, half_max)
        else:
            tile_max = peer_half_max if peer_half_max > half_max else half_max
        new_mi = old_mi
        scale_old = Float32(1.0)

        # Paired warps see identical merged maxima, so their warp votes make
        # the threshold decision uniform for the same 32-head group.
        should_rescale = cute.arch.vote_any_sync(tile_max - old_mi > Float32(self.RESCALE_THRESHOLD_LOG2))
        if should_rescale:
            if const_expr(self.indexer_topk > 0):
                new_mi = fmax_ftz_nonan(tile_max, old_mi)
            else:
                new_mi = tile_max if tile_max > old_mi else old_mi
            scale_old = cute.math.exp2(old_mi - new_mi, fastmath=True)

        # Keep the two probability partial sums packed so Blackwell can issue
        # one FFMA2/FADD2 for each score pair
        # instead of scalarizing both the affine transform and accumulation.
        half_sum_pair = (Float32(0.0), Float32(0.0))
        r_probability = cute.make_rmem_tensor((self.B_TOPK // 2,), self.element_dtype)
        for i in cutlass.range_constexpr(0, self.B_TOPK // 2, 2):
            probability_0, probability_1 = cute.arch.fma_packed_f32x2(
                (r_left[i], r_left[i + 1]),
                (scale_log2, scale_log2),
                (-new_mi, -new_mi),
            )
            probability_0 = cute.math.exp2(probability_0, fastmath=True)
            probability_1 = cute.math.exp2(probability_1, fastmath=True)
            half_sum_pair = cute.arch.add_packed_f32x2(half_sum_pair, (probability_0, probability_1))
            r_probability[i] = self.element_dtype(probability_0)
            r_probability[i + 1] = self.element_dtype(probability_1)

        half_sum = half_sum_pair[0] + half_sum_pair[1]

        new_li = old_li * scale_old + half_sum
        if const_expr(self.indexer_topk > 0):
            new_real_max = fmax_ftz_nonan(tile_max, old_real_max)
        else:
            new_real_max = tile_max if tile_max > old_real_max else old_real_max
        return new_mi, new_li, new_real_max, scale_old, should_rescale, r_probability.load()

    @cute.jit
    def _softmax_pair_sync(self, local_warp):
        if (local_warp & Int32(1)) == Int32(0):
            self.partial_p_barrier_even.arrive_and_wait()
        else:
            self.partial_p_barrier_odd.arrive_and_wait()

    @cute.jit
    def _rescale_o_tmem(self, tOtO_fragment, tmem_ptr, scale_for_old, thread_in_wg):
        # The preceding O completion wait carries the after-thread-sync side
        # of this protocol.  Publish this read/modify/write before the next PV
        # producer is allowed to reuse the accumulator.
        tO = cute.make_tensor(tmem_ptr + Int32(self.TMEM_O_OFFSET), tOtO_fragment.layout)
        tO_physical = cute.make_tensor(
            tO.iterator,
            cute.make_layout(
                ((self.HALF_H, 2), (128, 2)),
                stride=((65536, 4194304), (1, 128)),
            ),
        )
        load_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32)
        load_copy = tcgen05.make_tmem_copy(load_atom, tO_physical)
        load_thr = load_copy.get_slice(thread_in_wg)
        t_src = load_thr.partition_S(tO_physical)
        store_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32)), Float32)
        store_copy = tcgen05.make_tmem_copy(store_atom, tO_physical)
        store_thr = store_copy.get_slice(thread_in_wg)
        t_dst = store_thr.partition_D(tO_physical)
        r_o = cute.make_rmem_tensor(((32, 1), 1), Float32)
        for chunk in cutlass.range_constexpr(self.HALF_D // 32):
            cute.copy(load_copy, t_src[None, None, chunk], r_o)
            cute.arch.fence_view_async_tmem_load()
            for elem in cutlass.range_constexpr(32):
                r_o[elem] *= scale_for_old
            cute.copy(store_copy, r_o, t_dst[None, None, chunk])
            cute.arch.fence_view_async_tmem_store()
        tcgen05_fence_before_thread_sync()

    @cute.jit
    def _finalize_stats(
        self,
        max_logits,
        lse,
        lse_indexer,
        attn_sink,
        query_idx,
        head_global_base,
        mi,
        li,
        real_max,
        mi_indexer,
        li_indexer,
        sOutScale,
        scale_handle,
        thread_in_wg,
    ):
        head = thread_in_wg
        has_valid = real_max != -Float32.inf
        scale = Float32(0.0)
        if has_valid:
            sink = -Float32.inf
            if const_expr(attn_sink is not None):
                sink = Float32(attn_sink[head_global_base + head]) * Float32(self.LOG2_E)
            scale = Float32(1.0) / (li + cute.math.exp2(sink - mi, fastmath=True))

        # O normalization only depends on the reciprocal scale.  Publish it
        # before the logarithms and statistics stores so WG0 may start the
        # next output epilogue while WG3 finishes this query's metadata.
        sOutScale[head] = scale
        scale_handle.commit()

        if has_valid:
            max_logits[query_idx, head_global_base + head] = real_max * Float32(self.LN_2)
            lse[query_idx, head_global_base + head] = mi * Float32(self.LN_2) + cute.math.log(li, fastmath=False)
        else:
            max_logits[query_idx, head_global_base + head] = -Float32.inf
            # See dsa_fwd_sm100_head64.py's matching note: the generic
            # sparse_attention_forward_wrapper contract fixes the dead-row
            # sentinel at lse = -inf (the LSE-merge identity). lse_indexer is
            # the DSA indexer's own internal prefix statistic and keeps its
            # historical +inf empty sentinel.
            lse[query_idx, head_global_base + head] = -Float32.inf
        if const_expr(self.indexer_topk > 0 and lse_indexer is not None):
            if li_indexer > Float32(0.0):
                lse_indexer[query_idx, head_global_base + head] = mi_indexer * Float32(self.LN_2) + cute.math.log(li_indexer, fastmath=False)
            else:
                lse_indexer[query_idx, head_global_base + head] = Float32.inf

    @cute.jit
    def _prepare_output_first_chunk(
        self,
        tOtO_fragment,
        tmem_ptr,
        output_scale,
        thread_in_wg,
    ):
        """Load and convert the first O chunk before waiting for aliased sQ."""
        tO = cute.make_tensor(tmem_ptr + Int32(self.TMEM_O_OFFSET), tOtO_fragment.layout)
        tO_physical = cute.make_tensor(
            tO.iterator,
            cute.make_layout(
                ((self.HALF_H, 2), (128, 2)),
                stride=((65536, 4194304), (1, 128)),
            ),
        )
        load_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32)
        load_copy = tcgen05.make_tmem_copy(load_atom, tO_physical)
        load_thr = load_copy.get_slice(thread_in_wg)
        t_src = load_thr.partition_S(tO_physical)
        r_o = cute.make_rmem_tensor(((32, 1), 1), Float32)
        cute.copy(load_copy, t_src[None, None, 0], r_o)
        cute.arch.fence_view_async_tmem_load()
        for elem in cutlass.range_constexpr(0, 32, 2):
            r_o[elem], r_o[elem + 1] = cute.arch.mul_packed_f32x2(
                (r_o[elem], r_o[elem + 1]),
                (output_scale, output_scale),
                rnd="rn",
                ftz=False,
            )
        r_out = cute.make_rmem_tensor((32,), self.element_dtype)
        for elem in cutlass.range_constexpr(32):
            r_out[elem] = self.element_dtype(r_o[elem])
        return r_out.load()

    @cute.jit
    def _store_output(
        self,
        sO,
        tOtO_fragment,
        tmem_ptr,
        output_scale,
        first_output_ssa,
        thread_in_wg,
        tout_handle,
    ):
        # tout_consumer's wait carries the after-thread-sync side of the final
        # PV commit.  Release tOut immediately after the last TMEM load, before
        # converting/storing that register tile, so WG2 can begin the next PV
        # while WG0 finishes the SMEM/TMA epilogue.
        tO = cute.make_tensor(tmem_ptr + Int32(self.TMEM_O_OFFSET), tOtO_fragment.layout)
        tO_physical = cute.make_tensor(
            tO.iterator,
            cute.make_layout(
                ((self.HALF_H, 2), (128, 2)),
                stride=((65536, 4194304), (1, 128)),
            ),
        )
        load_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32)
        load_copy = tcgen05.make_tmem_copy(load_atom, tO_physical)
        load_thr = load_copy.get_slice(thread_in_wg)
        t_src = load_thr.partition_S(tO_physical)
        r_o = cute.make_rmem_tensor(((32, 1), 1), Float32)
        local_head = thread_in_wg % Int32(self.HALF_H)
        dim_base = (thread_in_wg // Int32(self.HALF_H)) * Int32(self.HALF_D)
        vector_layout = cute.make_layout((8,), stride=(1,))

        # The first chunk was converted while Q(n)'s S2T was still in flight;
        # publish it immediately after the caller's tQ wait.
        r_out = cute.make_rmem_tensor((32,), self.element_dtype)
        r_out.store(first_output_ssa)
        r_out_vectors = cute.flat_divide(r_out, vector_layout)
        dim_tile = dim_base // Int32(64)
        vector_base = (dim_base % Int32(64)) // Int32(8)
        s_out_vectors = cute.flat_divide(sO[None, local_head, dim_tile], vector_layout)
        for vector in cutlass.range_constexpr(4):
            cute.autovec_copy(r_out_vectors[None, vector], s_out_vectors[None, vector_base + Int32(vector)])

        for chunk in cutlass.range_constexpr(1, self.HALF_D // 32):
            cute.copy(load_copy, t_src[None, None, chunk], r_o)
            cute.arch.fence_view_async_tmem_load()
            if chunk == self.HALF_D // 32 - 1:
                tcgen05_fence_before_thread_sync()
                tout_handle.release()
            for elem in cutlass.range_constexpr(0, 32, 2):
                r_o[elem], r_o[elem + 1] = cute.arch.mul_packed_f32x2(
                    (r_o[elem], r_o[elem + 1]),
                    (output_scale, output_scale),
                    rnd="rn",
                    ftz=False,
                )
            r_out = cute.make_rmem_tensor((32,), self.element_dtype)
            for elem in cutlass.range_constexpr(32):
                r_out[elem] = self.element_dtype(r_o[elem])
            r_out_vectors = cute.flat_divide(r_out, vector_layout)
            dim = dim_base + Int32(chunk * 32)
            dim_tile = dim // Int32(64)
            vector_base = (dim % Int32(64)) // Int32(8)
            s_out_vectors = cute.flat_divide(sO[None, local_head, dim_tile], vector_layout)
            for vector in cutlass.range_constexpr(4):
                cute.autovec_copy(r_out_vectors[None, vector], s_out_vectors[None, vector_base + Int32(vector)])


__all__ = ["SparseAttentionForwardSm100Head128SmallTopKPrefill"]
