# Copyright (c) 2025 DeepSeek
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100 DeepSeek sparse-attention forward kernel for 64 query heads.

This module uses public CuTe DSL APIs except for the small gather4
bridge in ``_tma_gather4.py``.  The bridge is compile-time optional so the
audited predicated cp.async path remains available with older DSL wheels.

This is a correctness-first implementation. A CTA owns one query and its
twelve warps split the 64 heads. Q is staged once by a regular public
1-CTA TMA copy and each 64-token sparse tile is gathered with 2-D TMA gather4
(or safe, predicated 128-bit cp.async copies as a compatibility fallback).
QK and the two 256-column PV halves are real 1-SM tcgen05
MMAs; online-softmax statistics stay in registers while O persists in TMEM
across sparse tiles.  Keeping validity separate from the zero-filled KV tile
is important: an invalid KV row must produce ``-inf`` rather than a zero logit.

For both D=512 and D=576, the gather4 fast path uses three
physical NoPE/V stages so the four gather warps can maintain a two-tile lead;
the predicated cp.async fallback keeps two stages.  A NoPE/V stage is not
reusable until the corresponding PV tcgen05 commit completes, while D=576's
single tail stage is released immediately after tail QK. Direct TMEM-to-
register score consumption, byte validity flags, and a four-word warp-vote
exchange keep the ring below SM100's shared-memory limit while preserving the
Q-to-TMEM mapping.

The implementation supports D=576's K-only tail, duplicate indices, dynamic
per-query lengths, sink-only output normalization, optional indexer-prefix
LSE, and the empty-row sentinel (O=0, max_logits=-inf, LSE=+inf).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import T
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils

from ..utils import copy as copy_utils
from ._tcgen05_sync import (
    tcgen05_fence_after_thread_sync,
    tcgen05_fence_before_thread_sync,
)
from ._tcgen05_mma_ws import (
    copy_q_128xk_f16_s2t_cta1,
    mma_ws_ss_f16_cta1,
    mma_ws_ts_f16_cta1,
    tmem_load_32dp32b32x,
    tmem_store_32dp32b32x,
)
from ._nvvm_compat import fmax_ftz_nonan
from ._tma_gather4 import TMA_GATHER4_AVAILABLE, tma_gather4_cta1


@cute.jit
def _initialize_mbarriers(
    tidx: Int32,
    q_ready_mbar: cute.Pointer,
    q_tail_ready_mbar: cute.Pointer,
    q_s2t_done_mbar: cute.Pointer,
    kv_ready_mbar: cute.Pointer,
    valid_ready_mbar: cute.Pointer,
    valid_free_mbar: cute.Pointer,
    kv_tail_ready_mbar: cute.Pointer,
    kv_tail_free_mbar: cute.Pointer,
    mma_done_mbar: cute.Pointer,
    p_free_mbar: cute.Pointer,
    so_ready_mbar: cute.Pointer,
    pv_done_mbar: cute.Pointer,
    num_kv_stages: cutlass.Constexpr[int],
    num_kv_parts: cutlass.Constexpr[int],
    num_tail_stages: cutlass.Constexpr[int],
) -> None:
    """Initialize all CTA barriers without carrying ``SharedStorage`` through a dynamic branch."""

    # CuTe DSL 4.5 cannot flatten a local ``cute.struct`` object across a
    # dynamic ``if``.  Keep the exact one-thread initialization protocol in a
    # pointer-only helper so the branch state consists entirely of DSL values.
    if tidx == Int32(0):
        cute.arch.mbarrier_init(q_ready_mbar, 1)
        cute.arch.mbarrier_init(q_tail_ready_mbar, 4 * cute.arch.WARP_SIZE)
        cute.arch.mbarrier_init(q_s2t_done_mbar, 1)
        for stage_part in cutlass.range_constexpr(num_kv_stages * num_kv_parts):
            cute.arch.mbarrier_init(kv_ready_mbar + stage_part, 1)
        for stage in cutlass.range_constexpr(num_kv_stages):
            cute.arch.mbarrier_init(valid_ready_mbar + stage, 1)
            cute.arch.mbarrier_init(valid_free_mbar + stage, 4 * cute.arch.WARP_SIZE)
        for stage in cutlass.range_constexpr(num_tail_stages):
            cute.arch.mbarrier_init(kv_tail_ready_mbar + stage, 2 * cute.arch.WARP_SIZE)
        cute.arch.mbarrier_init(kv_tail_free_mbar, 1)
        cute.arch.mbarrier_init(mma_done_mbar, 1)
        cute.arch.mbarrier_init(p_free_mbar, 4 * cute.arch.WARP_SIZE)
        cute.arch.mbarrier_init(so_ready_mbar, 4 * cute.arch.WARP_SIZE)
        for stage in cutlass.range_constexpr(num_kv_stages):
            cute.arch.mbarrier_init(pv_done_mbar + stage, 1)
        cute.arch.mbarrier_init_fence()


@cute.jit
def _ldg_indices_128(indices: cute.Tensor, query_idx: Int32, slot: Int32) -> Tuple[Int32, ...]:
    """Load one aligned ``int4`` directly into the elected gather lane."""

    # The wrapper pads every row to 64 Int32 values.  Its 256-byte row stride
    # and each four-element gather group make these addresses 16-byte aligned.
    ptr = indices.iterator + cute.crd2idx((query_idx, slot), indices.layout)
    ptr = cute.make_ptr(Int32, ptr.toint(), indices.memspace, assumed_align=16)
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [ptr.llvm_ptr],
        "{\n\t" ".reg .b64 v<2>;\n\t" "ld.global.nc.v2.u64 {v0, v1}, [$4];\n\t" "mov.b64 {$0, $1}, v0;\n\t" "mov.b64 {$2, $3}, v1;\n\t" "}\n",
        ",".join(["=r"] * 4 + ["l"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i])) for i in range(4))


class SparseAttentionForwardSm100Head64:
    """Regular sparse-prefill forward specialization for H=64.

    Args:
        head_dim: Q/K head dimension; supported values are 512 and 576.
        indexer_topk: Prefix length whose LSE is returned.  Zero disables the
            prefix statistic; the instantiated values are 0/512/1024/2048.

    ``indices`` passed to :meth:`__call__` is the physical, wrapper-padded
    ``[Tq, Kp]`` tensor.  ``Kp`` may be any multiple of 64.  ``topk_length``
    is clamped to ``[0, Kp]`` inside the kernel.
    """

    NUM_HEADS = 64
    VALUE_DIM = 512
    BLOCK_TOPK = 64
    NUM_WARPS = 12
    WARP_SIZE = 32
    THREADS_PER_CTA = NUM_WARPS * WARP_SIZE
    RESCALE_THRESHOLD_LOG2 = 6.0
    COPY_BITS = 128
    COPY_ELEMS = COPY_BITS // 16

    # Q is still sourced from SMEM, but reserving its TMEM columns keeps the
    # persistent O/P placement compatible with the TS-QK fast path.
    TMEM_O_OFFSET = 0
    TMEM_O_HALF_COLUMNS = 128
    TMEM_Q_OFFSET = 256
    TMEM_P_OFFSET = 400
    TMEM_COLUMNS = 512
    TMEM_IO_CHUNK = 32

    # Initialize mi to a large finite value so that an all-invalid tile never
    # evaluates -inf - (-inf). real_mi is tracked independently and restores
    # the public empty-row sentinel in epilogue.
    MAX_INIT_VALUE = -1.0e30

    def __init__(
        self,
        head_dim: int,
        indexer_topk: int = 0,
    ):
        if head_dim not in (512, 576):
            raise ValueError(f"head_dim must be 512 or 576, got {head_dim}")
        if indexer_topk not in (0, 512, 1024, 2048):
            raise ValueError(f"indexer_topk must be one of 0/512/1024/2048, got {indexer_topk}")

        self.head_dim = head_dim
        self.indexer_topk = indexer_topk
        # This is an environment capability, not a caller-selectable kernel
        # specialization. Keep the audited cp.async fallback internally for
        # DSL builds that cannot expose the gather4 descriptor bridge.
        self.use_tma_gather4 = TMA_GATHER4_AVAILABLE
        self.indexer_tile = indexer_topk // self.BLOCK_TOPK - 1 if indexer_topk else -1
        # The gather4 fast path uses a three-stage NoPE/V ring. The
        # compatibility cp.async path keeps its proven two-stage allocation
        # and schedule.
        self.num_kv_stages = 3 if self.use_tma_gather4 else 2
        self.num_tail_stages = 1 if self.use_tma_gather4 else self.num_kv_stages
        self.num_kv_parts = 2
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.THREADS_PER_CTA)
        self.gather_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=4 * self.WARP_SIZE)
        self.softmax_sync_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=4 * self.WARP_SIZE)

        # D=576 still writes only the first 512 values.  The last 64 elements
        # participate in QK and are never read by the PV loop.
        assert self.VALUE_DIM % self.WARP_SIZE == 0
        assert head_dim % self.COPY_ELEMS == 0
        assert self.BLOCK_TOPK % self.WARP_SIZE == 0

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # [Tq, 64, D], FP16/BF16
        kv: cute.Tensor,  # [Tkv, D], FP16/BF16; first 512 values are also V
        indices: cute.Tensor,  # [Tq, Kp], INT32; Kp is padded to a multiple of 64
        out: cute.Tensor,  # [Tq, 64, 512], matches Q/KV dtype
        max_logits: cute.Tensor,  # [Tq, 64], FP32
        lse: cute.Tensor,  # [Tq, 64], FP32, excludes sink
        lse_indexer: Optional[cute.Tensor],  # [Tq, 64], FP32
        attn_sink: Optional[cute.Tensor],  # [64], FP32
        topk_length: Optional[cute.Tensor],  # [Tq], INT32
        softmax_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        """Build and launch one regular-head64 CTA per query token."""

        self.element_dtype = q.element_type
        if const_expr(self.element_dtype not in (Float16, BFloat16)):
            raise TypeError("head64 sparse attention requires FP16 or BF16 q")
        if const_expr(kv.element_type is not self.element_dtype or out.element_type is not self.element_dtype):
            raise TypeError("head64 sparse attention requires matching q, kv, and out dtypes")
        if const_expr(indices.element_type != Int32):
            raise TypeError("indices must have INT32 element type")
        if const_expr(max_logits.element_type != Float32 or lse.element_type != Float32):
            raise TypeError("max_logits and lse must have FP32 element type")
        if const_expr(lse_indexer is not None and lse_indexer.element_type != Float32):
            raise TypeError("lse_indexer must have FP32 element type")
        if const_expr(attn_sink is not None and attn_sink.element_type != Float32):
            raise TypeError("attn_sink must have FP32 element type")
        if const_expr(topk_length is not None and topk_length.element_type != Int32):
            raise TypeError("topk_length must have INT32 element type")

        # The interface validates ranks and public shapes.  These compile-time
        # checks guard accidental direct use of the internal kernel class.
        if const_expr(cute.rank(q.shape) != 3 or cute.rank(kv.shape) != 2 or cute.rank(indices.shape) != 2):
            raise ValueError("expected q[Tq,H,D], kv[Tkv,D], and indices[Tq,Kp]")
        # Shape extents are intentionally dynamic in the API's DLPack
        # descriptors, so H/D checks live in APIBase.check_support() rather
        # than being forced through const_expr here.

        # The public object below is used only for descriptor/fragment layout.
        # QK uses the one-CTA warp-specialized TS instruction (the public
        # builder currently only emits non-WS MMA).
        # N=128 is intentional: it forges two independent 64-token dot products
        # whose partials are combined by paired softmax warps.
        qk_op = tcgen05.MmaF16BF16Op(
            self.element_dtype,
            Float32,
            (self.NUM_HEADS, 2 * self.BLOCK_TOPK, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        tiled_mma_qk = cute.make_tiled_mma(
            cute.make_mma_atom(qk_op),
            cute.make_layout((1, 1, 1)),
            (self.NUM_HEADS, 2 * self.BLOCK_TOPK, 16),
        )

        # A separate SS object describes the physical Q/K staging surfaces and
        # the ordinary Q TMA transaction.  The first 512 dimensions use SW128;
        # D=576's final 64 dimensions use SW64 before they are re-viewed as
        # 128x32 for implicit dual GEMM.
        storage_mma_tiler = (self.NUM_HEADS, self.BLOCK_TOPK, self.VALUE_DIM)
        tiled_mma_qk_storage = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            storage_mma_tiler[:2],
        )
        mma_tiler_pv = (self.NUM_HEADS, self.VALUE_DIM // 2, self.BLOCK_TOPK)
        tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            mma_tiler_pv[:2],
        )
        q_nope_layout = sm100_utils.make_smem_layout_a(tiled_mma_qk_storage, storage_mma_tiler, self.element_dtype, 1)
        kv_nope_layout = sm100_utils.make_smem_layout_b(tiled_mma_qk_storage, storage_mma_tiler, self.element_dtype, 1)
        # Keep the D=576 tail byte-for-byte compatible with the
        # Layout_K_SW64 tile_to_shape. Replacing only the swizzle on the
        # public SW128 helper is not equivalent: its outer layout keeps the
        # 16-column atom decomposition, whereas SW64 has two physical 32-wide
        # planes separated by 2,048 16-bit elements.
        tail_physical_outer = cute.make_layout((64, (32, 2)), stride=(32, (1, 2048)))
        q_tail_layout = cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0, tail_physical_outer)
        kv_tail_layout = cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0, tail_physical_outer)
        kv_dual_nope_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_qk,
            (self.NUM_HEADS, 2 * self.BLOCK_TOPK, self.VALUE_DIM // 2),
            self.element_dtype,
            1,
        )
        kv_dual_tail_layout = cute.make_composed_layout(
            cute.make_swizzle(2, 4, 3),
            0,
            cute.make_layout((2 * self.BLOCK_TOPK, 32), stride=(32, 1)),
        )
        p_mma_layout = sm100_utils.make_smem_layout_a(tiled_mma_pv, mma_tiler_pv, self.element_dtype, 1)
        v_mma_layout = sm100_utils.make_smem_layout_b(tiled_mma_pv, mma_tiler_pv, self.element_dtype, 1)

        # Build a compiler-owned 2-D tensor map for gather4: logical (D, Tkv),
        # a 64x1 16-bit box, and SW128.
        # The local low-level bridge only supplies the missing gather4 opcode.
        kv_descriptor_tensor = cute.make_tensor(
            kv.iterator,
            cute.make_layout(
                (self.VALUE_DIM, cute.size(kv.shape[0])),
                stride=(1, cute.size(kv.shape[1])),
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
        # gather4 covers only the 512-d NoPE/V surface. Copy the D=576 SW64
        # tail with cp.async because four gathered 64-element boxes
        # are not laid out like one 64x64 SW64 operand tile.
        kv_tma_bytes = self.BLOCK_TOPK * self.VALUE_DIM * (self.element_dtype.width // 8)

        # Q is physically [Tq, H, D].  Put the two tiled modes first so the
        # public SM100 TMA-A helper can describe one complete [H, D] query
        # tile.  Unlike sparse KV, this copy is regular and needs no gather
        # extension.
        q_tma_global = cute.make_tensor(q.iterator, cute.select(q.layout, mode=[1, 2, 0]))
        q_nope_tma_layout = cute.select(q_nope_layout, mode=[0, 1, 2])
        tma_atom_q, q_tma = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            q_tma_global,
            q_nope_tma_layout,
            storage_mma_tiler,
            tiled_mma_qk_storage,
        )
        q_tail_global = cute.make_tensor(
            q.iterator + self.VALUE_DIM,
            cute.make_layout(
                (cute.size(q.shape[1]), 64, cute.size(q.shape[0])),
                stride=cute.select(q.layout, mode=[1, 2, 0]).stride,
            ),
        )
        q_tma_bytes = cute.size_in_bytes(self.element_dtype, q_nope_tma_layout)

        # Reuse the drained arena as an eight-tile SW128 output stage.  Keep
        # the tensor map rank three, (D, H, Tq), so every 64x64 store lowers to
        # the source-style UTMASTG.3D rather than a tiled-dimension 4-D map.
        o_chunk_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((64, self.NUM_HEADS), stride=(1, 64)),
        )
        o_full_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (64, self.NUM_HEADS, self.VALUE_DIM // 64),
                stride=(1, 64, 64 * self.NUM_HEADS),
            ),
        )
        out_tma = cute.make_tensor(out.iterator, cute.select(out.layout, mode=[2, 1, 0]))
        tma_atom_o, out_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            out_tma,
            o_chunk_smem_layout,
            (64, self.NUM_HEADS),
        )
        assert cute.cosize(o_full_smem_layout) == self.NUM_HEADS * self.VALUE_DIM

        vector_layout = cute.make_layout((self.BLOCK_TOPK,), stride=(1,))
        valid_layout = cute.make_layout((self.BLOCK_TOPK // 8,), stride=(1,))
        row_layout = cute.make_layout((self.NUM_HEADS,), stride=(1,))
        p_exchange_layout = cute.make_layout(
            (4, self.BLOCK_TOPK // 8, self.WARP_SIZE, 4),
            stride=(self.WARP_SIZE * (self.BLOCK_TOPK // 2), self.WARP_SIZE * 4, 4, 1),
        )
        q_tmem_nope_layout = cute.make_layout((2 * self.NUM_HEADS, self.VALUE_DIM // 2), stride=(131072, 1))
        q_tmem_tail_layout = cute.make_layout((2 * self.NUM_HEADS, 32), stride=(131072, 1))

        q_nope_elements = cute.cosize(q_nope_layout)
        tail_elements = cute.cosize(q_tail_layout) if self.head_dim == 576 else 0
        kv_nope_elements = cute.cosize(kv_nope_layout)
        # Fast arena: [tail?][K0/K1/K2(Q alias)][independent FP32 P exchange].
        # Fallback arena: [Q][K0/K1], including each D=576 tail.  A constexpr
        # expression keeps the nested SharedStorage extent code-read-time
        # constant; staged ``if`` assignments are not visible to that type.
        p_exchange_elements = cute.cosize(p_exchange_layout) * (Float32.width // self.element_dtype.width)
        arena_elements = (
            tail_elements + self.num_kv_stages * kv_nope_elements + p_exchange_elements
            if self.use_tma_gather4
            else q_nope_elements + tail_elements + self.num_kv_stages * (kv_nope_elements + tail_elements)
        )

        @cute.struct
        class SharedStorage:
            q_ready_mbar: cute.struct.MemRange[Int64, 1]
            q_tail_ready_mbar: cute.struct.MemRange[Int64, 1]
            q_s2t_done_mbar: cute.struct.MemRange[Int64, 1]
            kv_ready_mbar: cute.struct.MemRange[Int64, self.num_kv_stages * self.num_kv_parts]
            valid_ready_mbar: cute.struct.MemRange[Int64, self.num_kv_stages]
            valid_free_mbar: cute.struct.MemRange[Int64, self.num_kv_stages]
            kv_tail_ready_mbar: cute.struct.MemRange[Int64, self.num_tail_stages]
            kv_tail_free_mbar: cute.struct.MemRange[Int64, 1]
            mma_done_mbar: cute.struct.MemRange[Int64, 1]
            p_free_mbar: cute.struct.MemRange[Int64, 1]
            so_ready_mbar: cute.struct.MemRange[Int64, 1]
            pv_done_mbar: cute.struct.MemRange[Int64, self.num_kv_stages]
            tmem_holding_buf: Int32
            # Pack all small metadata before the first 1-KiB surface.  It fits
            # in exactly two alignment pages for every specialization, which
            # yields the source-sized 223,232/231,424-byte fast-path totals.
            sIndex: cute.struct.Align[cute.struct.MemRange[Int32, self.num_kv_stages * self.BLOCK_TOPK], 16]
            sValid: cute.struct.Align[
                cute.struct.MemRange[Uint8, self.num_kv_stages * (self.BLOCK_TOPK // 8)],
                16,
            ]
            sGroupRescale: cute.struct.Align[cute.struct.MemRange[Int32, 2], 8]
            sScale: cute.struct.Align[cute.struct.MemRange[Float32, self.NUM_HEADS], 16]
            sArena: cute.struct.Align[cute.struct.MemRange[self.element_dtype, arena_elements], 1024]
            sWeight: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(p_mma_layout)], 1024]

        # SM100 exposes up to 227 KiB of dynamic shared memory. D=576 is the
        # limiting specialization at 231,424 bytes, leaving 1,024 bytes.
        assert SharedStorage.size_in_bytes() <= 227 * 1024, f"H64 shared storage is {SharedStorage.size_in_bytes()} bytes"
        self.shared_storage = SharedStorage

        grid = (cute.size(q.shape[0]), 1, 1)
        self.kernel(
            q_tma,
            q_tail_global,
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
            tiled_mma_qk,
            tiled_mma_qk_storage,
            tiled_mma_pv,
            q_nope_layout,
            q_tail_layout,
            kv_nope_layout,
            kv_tail_layout,
            kv_dual_nope_layout,
            kv_dual_tail_layout,
            p_mma_layout,
            v_mma_layout,
            q_tmem_nope_layout,
            q_tmem_tail_layout,
            vector_layout,
            valid_layout,
            row_layout,
            p_exchange_layout,
            o_full_smem_layout,
        ).launch(
            grid=grid,
            block=(self.THREADS_PER_CTA, 1, 1),
            cluster=(1, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q_tma: cute.Tensor,
        q_tail_global: cute.Tensor,
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
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_qk_storage: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        q_nope_layout: cute.ComposedLayout,
        q_tail_layout: cute.ComposedLayout,
        kv_nope_layout: cute.ComposedLayout,
        kv_tail_layout: cute.ComposedLayout,
        kv_dual_nope_layout: cute.ComposedLayout,
        kv_dual_tail_layout: cute.ComposedLayout,
        p_mma_layout: cute.ComposedLayout,
        v_mma_layout: cute.ComposedLayout,
        q_tmem_nope_layout: cute.Layout,
        q_tmem_tail_layout: cute.Layout,
        vector_layout: cute.Layout,
        valid_layout: cute.Layout,
        row_layout: cute.Layout,
        p_exchange_layout: cute.Layout,
        o_full_smem_layout: cute.ComposedLayout,
    ):
        query_idx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # CuTe DSL 4.5 cannot carry a local SharedStorage object into a
        # dynamic control-flow region.  Materialize the fields used by those
        # regions as primitive pointer SSA values before the first branch.
        q_ready_mbar_ptr = storage.q_ready_mbar.data_ptr()
        q_tail_ready_mbar_ptr = storage.q_tail_ready_mbar.data_ptr()
        q_s2t_done_mbar_ptr = storage.q_s2t_done_mbar.data_ptr()
        kv_ready_mbar_ptr = storage.kv_ready_mbar.data_ptr()
        valid_ready_mbar_ptr = storage.valid_ready_mbar.data_ptr()
        valid_free_mbar_ptr = storage.valid_free_mbar.data_ptr()
        kv_tail_ready_mbar_ptr = storage.kv_tail_ready_mbar.data_ptr()
        kv_tail_free_mbar_ptr = storage.kv_tail_free_mbar.data_ptr()
        mma_done_mbar_ptr = storage.mma_done_mbar.data_ptr()
        p_free_mbar_ptr = storage.p_free_mbar.data_ptr()
        so_ready_mbar_ptr = storage.so_ready_mbar.data_ptr()
        pv_done_mbar_ptr = storage.pv_done_mbar.data_ptr()
        s_index_base_ptr = storage.sIndex.data_ptr()
        s_valid_base_ptr = storage.sValid.data_ptr()
        arena_base_ptr = storage.sArena.data_ptr()
        sO = cute.make_tensor(
            cute.recast_ptr(arena_base_ptr, o_full_smem_layout.inner),
            o_full_smem_layout.outer,
        )
        q_nope_elements = cute.cosize(q_nope_layout)
        kv_nope_elements = cute.cosize(kv_nope_layout)
        tail_elements = cute.cosize(kv_tail_layout) if self.head_dim == 576 else 0
        if const_expr(self.use_tma_gather4):
            # [tail0?][K0 64KiB][K1 64KiB][K2/Q 64KiB]
            sKV_tail_base_ptr = arena_base_ptr
            sKV_base_ptr = arena_base_ptr + tail_elements
            sQ_base_ptr = sKV_base_ptr + Int32(2 * kv_nope_elements)
            sQ_tail_base_ptr = storage.sWeight.data_ptr()
        else:
            # Compatibility arena retains [Q][K0][K1], including per-stage
            # D=576 tails.  This is byte-for-byte the old two-stage mapping.
            sQ_base_ptr = arena_base_ptr
            sQ_tail_base_ptr = sQ_base_ptr + q_nope_elements
            sKV_base_ptr = sQ_tail_base_ptr + tail_elements
            sKV_tail_base_ptr = sKV_base_ptr + kv_nope_elements

        sQ_nope_mma = cute.make_tensor(cute.recast_ptr(sQ_base_ptr, q_nope_layout.inner), q_nope_layout.outer)
        sQ_tail_mma = cute.make_tensor(
            cute.recast_ptr(sQ_tail_base_ptr, q_tail_layout.inner),
            q_tail_layout.outer,
        )
        sQ_tail = cute.composition(sQ_tail_mma, cute.make_layout((self.NUM_HEADS, 64)))
        sQ_dual_nope_mma = cute.make_tensor(cute.recast_ptr(sQ_base_ptr, kv_dual_nope_layout.inner), kv_dual_nope_layout.outer)
        sQ_dual_nope_slice = sQ_dual_nope_mma[(None, None), 0, (None, None), 0]
        sQ_dual_nope = cute.composition(
            sQ_dual_nope_slice,
            cute.make_layout((2 * self.NUM_HEADS, self.VALUE_DIM // 2)),
        )
        sQ_dual_tail_mma = cute.make_tensor(
            cute.recast_ptr(sQ_tail_base_ptr, kv_dual_tail_layout.inner),
            kv_dual_tail_layout.outer,
        )
        sQ_dual_tail = cute.composition(sQ_dual_tail_mma, cute.make_layout((2 * self.NUM_HEADS, 32)))
        sKV_tail_mma0 = cute.make_tensor(
            cute.recast_ptr(sKV_tail_base_ptr, kv_tail_layout.inner),
            kv_tail_layout.outer,
        )
        sKV_tail0 = cute.composition(sKV_tail_mma0, cute.make_layout((self.BLOCK_TOPK, 64)))
        sP_mma = storage.sWeight.get_tensor(p_mma_layout.outer, swizzle=p_mma_layout.inner)
        sP_slice = sP_mma[(None, None), 0, None, 0]
        sWeight = cute.composition(sP_slice, cute.make_layout((self.NUM_HEADS, self.BLOCK_TOPK)))
        # K-NoPE and V are the same 512-element latent. The SM100 K-major
        # SW128 and V MN-major operand layouts are physical aliases. Each
        # 256-column PV instruction
        # selects one contiguous half of that shared allocation.
        sValidWords = cute.make_tensor(
            cute.recast_ptr(s_valid_base_ptr, dtype=Uint32),
            cute.make_layout((self.num_kv_stages * (self.BLOCK_TOPK // self.WARP_SIZE),), stride=(1,)),
        )
        sGroupRescale = storage.sGroupRescale.get_tensor(cute.make_layout((2,), stride=(1,)))
        sScale = storage.sScale.get_tensor(row_layout)
        p_exchange_arena_offset = tail_elements + self.num_kv_stages * kv_nope_elements if self.use_tma_gather4 else 0
        p_exchange_storage_ptr = arena_base_ptr + Int32(p_exchange_arena_offset)
        # Runtime pointer arithmetic drops CuTe's inferred alignment even
        # though every arena component is 1-KiB aligned.  Restore the 16-byte
        # contract needed by the source-equivalent float4 exchange.
        p_exchange_base_ptr = cute.make_ptr(
            Float32,
            p_exchange_storage_ptr.llvm_ptr,
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        sPExchange = cute.make_tensor(p_exchange_base_ptr, p_exchange_layout)
        sPExchangeLinear = cute.make_tensor(
            p_exchange_base_ptr,
            cute.make_layout((4 * self.WARP_SIZE * (self.BLOCK_TOPK // 2),), stride=(1,)),
        )

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.element_dtype,
            num_bits_per_copy=self.COPY_BITS,
        )
        thread_copy = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_layout((8,)),
            cute.make_layout((self.COPY_ELEMS,)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            # WG0 warp 0 allocates and frees TMEM, so the epilogue does not
            # have to rendezvous with producer and MMA roles before
            # deallocation.
            allocator_warp_id=0,
            is_two_cta=False,
        )

        _initialize_mbarriers(
            tidx,
            q_ready_mbar_ptr,
            q_tail_ready_mbar_ptr,
            q_s2t_done_mbar_ptr,
            kv_ready_mbar_ptr,
            valid_ready_mbar_ptr,
            valid_free_mbar_ptr,
            kv_tail_ready_mbar_ptr,
            kv_tail_free_mbar_ptr,
            mma_done_mbar_ptr,
            p_free_mbar_ptr,
            so_ready_mbar_ptr,
            pv_done_mbar_ptr,
            self.num_kv_stages,
            self.num_kv_parts,
            self.num_tail_stages,
        )
        tmem.allocate(self.TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        q_tmem_ptr = cute.recast_ptr(tmem_ptr, dtype=self.element_dtype)
        tQ_nope = cute.make_tensor(
            q_tmem_ptr + Int32(self.TMEM_Q_OFFSET * (Float32.width // self.element_dtype.width)),
            q_tmem_nope_layout,
        )
        tQ_tail = cute.make_tensor(
            q_tmem_ptr + Int32((self.TMEM_Q_OFFSET + 128) * (Float32.width // self.element_dtype.width)),
            q_tmem_tail_layout,
        )

        # Stage the regular Q tile once with an ordinary 1-CTA TMA.  Sparse KV
        # remains on the audited predicated gather path below.
        gQ = cute.local_tile(q_tma, (self.NUM_HEADS, self.VALUE_DIM), (0, 0, query_idx))
        tAgQ = tiled_mma_qk_storage.get_slice(0).partition_A(gQ)
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_nope_mma, 0, cute.rank(sQ_nope_mma)),
            cute.group_modes(tAgQ, 0, cute.rank(tAgQ)),
        )
        if warp_idx == Int32(0):
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_o)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(q_ready_mbar_ptr, q_tma_bytes)
            cute.copy(tma_atom_q, tQgQ, tQsQ, tma_bar_ptr=q_ready_mbar_ptr)
        if const_expr(not self.use_tma_gather4):
            cute.arch.mbarrier_wait(q_ready_mbar_ptr, 0)
        if const_expr(self.head_dim == 576):
            # The public TMA-A builder decomposes this SW64 tail into 64-byte
            # destination steps, which violates UTMALDG's alignment contract.
            # Copy the small one-time 8-KiB tail with aligned cp.async chunks;
            # the following S2T path and QK descriptor remain source-exact.
            if tidx < Int32(4 * self.WARP_SIZE):
                for copy_iter in cutlass.range_constexpr((self.NUM_HEADS * 64 // self.COPY_ELEMS) // (4 * self.WARP_SIZE)):
                    linear_chunk = tidx + Int32(copy_iter * 4 * self.WARP_SIZE)
                    q_head = linear_chunk // Int32(64 // self.COPY_ELEMS)
                    q_chunk = linear_chunk % Int32(64 // self.COPY_ELEMS)
                    g_tail_row = q_tail_global[q_head, None, query_idx]
                    s_tail_row = sQ_tail[q_head, None]
                    g_tail_chunk = cute.local_tile(g_tail_row, (self.COPY_ELEMS,), (q_chunk,))
                    s_tail_chunk = cute.local_tile(s_tail_row, (self.COPY_ELEMS,), (q_chunk,))
                    cute.copy(copy_atom, g_tail_chunk, s_tail_chunk)
                if const_expr(self.use_tma_gather4):
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(q_tail_ready_mbar_ptr)
            if const_expr(not self.use_tma_gather4):
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
        if const_expr(not self.use_tma_gather4):
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()
        if warp_idx == Int32(8):
            if const_expr(self.use_tma_gather4):
                cute.arch.mbarrier_wait(q_ready_mbar_ptr, 0)
                if const_expr(self.head_dim == 576):
                    cute.arch.mbarrier_wait(q_tail_ready_mbar_ptr, 0)
                cute.arch.fence_view_async_shared()
            tcgen05_fence_after_thread_sync()
            copy_q_128xk_f16_s2t_cta1(sQ_dual_nope, tQ_nope)
            if const_expr(self.head_dim == 576):
                copy_q_128xk_f16_s2t_cta1(sQ_dual_tail, tQ_tail)
            if const_expr(self.use_tma_gather4):
                with cute.arch.elect_one():
                    tcgen05.commit(q_s2t_done_mbar_ptr)
            tcgen05_fence_before_thread_sync()
        if const_expr(not self.use_tma_gather4):
            cute.arch.sync_threads()

        if warp_idx == Int32(4):
            cpasync.prefetch_descriptor(tma_atom_kv)

        physical_topk = Int32(cute.size(indices.shape[1]))
        effective_length = physical_topk
        if const_expr(topk_length is not None):
            effective_length = Int32(topk_length[query_idx])
        effective_length = effective_length if effective_length > Int32(0) else Int32(0)
        effective_length = effective_length if effective_length < physical_topk else physical_topk
        tile_count = (effective_length + Int32(self.BLOCK_TOPK - 1)) // Int32(self.BLOCK_TOPK)
        tile_count = tile_count if tile_count > Int32(0) else Int32(1)

        # In the double-buffered schedule, WG1 (warps 4-7) is a pure gather
        # producer, warp 8 is the pure UMMA issuer, and WG0 uses the
        # two-threads-per-head softmax mapping.
        rMi = cute.make_rmem_tensor((1,), Float32)
        rLi = cute.make_rmem_tensor((1,), Float32)
        rRealMax = cute.make_rmem_tensor((1,), Float32)
        rMiIndexer = cute.make_rmem_tensor((1,), Float32)
        rLiIndexer = cute.make_rmem_tensor((1,), Float32)
        rMi.fill(Float32(self.MAX_INIT_VALUE))
        rLi.fill(Float32(0.0))
        rRealMax.fill(-Float32.inf)
        rMiIndexer.fill(Float32(self.MAX_INIT_VALUE))
        rLiIndexer.fill(Float32(0.0))

        # The gather4 warpgroup owns one dynamic producer loop. It naturally
        # establishes the two-tile lead (K0/K1/K2
        # are launched before the first stage-reuse wait) without statically
        # cloning the 32 gather4 instructions for prime and steady state.
        # The compatibility path retains its proven one-tile prime/refill flow.
        kv_stage_elements = Int32(cute.cosize(kv_nope_layout))
        if const_expr(not self.use_tma_gather4):
            kv_stage_elements = kv_stage_elements + Int32(cute.cosize(kv_tail_layout) if self.head_dim == 576 else 0)
        if const_expr(self.use_tma_gather4):
            if warp_idx >= Int32(4) and warp_idx < Int32(8):
                producer_tile = Int32(0)
                while producer_tile < tile_count:
                    producer_stage = producer_tile % Int32(self.num_kv_stages)
                    if producer_tile >= Int32(self.num_kv_stages):
                        prior_use = producer_tile // Int32(self.num_kv_stages) - Int32(1)
                        cute.arch.mbarrier_wait(
                            pv_done_mbar_ptr + producer_stage,
                            prior_use % Int32(2),
                        )
                    if producer_tile == Int32(2):
                        cute.arch.mbarrier_wait(q_s2t_done_mbar_ptr, 0)

                    producer_stage_elem_offset = producer_stage * kv_stage_elements
                    producer_kv_ptr = sKV_base_ptr + cute.assume(
                        producer_stage_elem_offset,
                        divby=self.COPY_ELEMS,
                    )
                    logical_tile_is_full = Int32(1)
                    if const_expr(topk_length is not None):
                        logical_tile_is_full = cute.arch.make_warp_uniform(Int32((producer_tile + Int32(1)) * Int32(self.BLOCK_TOPK) <= effective_length))
                    self._gather_kv_tma_direct(
                        indices,
                        query_idx,
                        producer_tile,
                        effective_length,
                        logical_tile_is_full,
                        producer_kv_ptr,
                        tma_atom_kv,
                        kv_ready_mbar_ptr + producer_stage * Int32(self.num_kv_parts),
                        kv_tma_bytes,
                        warp_idx,
                    )
                    producer_tile = producer_tile + Int32(1)

            # Warp 9 independently publishes the two packed validity words.
            # Its three-stage ring is released only after every WG0 thread has
            # consumed the corresponding half-tile mask.
            if warp_idx == Int32(9):
                valid_tile = Int32(0)
                while valid_tile < tile_count:
                    valid_stage = valid_tile % Int32(self.num_kv_stages)
                    valid_phase = (valid_tile // Int32(self.num_kv_stages)) % Int32(2)
                    cute.arch.mbarrier_wait(
                        valid_free_mbar_ptr + valid_stage,
                        valid_phase ^ Int32(1),
                    )
                    for half in cutlass.range_constexpr(self.BLOCK_TOPK // self.WARP_SIZE):
                        slot = valid_tile * Int32(self.BLOCK_TOPK) + Int32(half * self.WARP_SIZE) + lane_idx
                        token = Int32(indices[query_idx, slot])
                        valid = slot < effective_length and token >= Int32(0) and Int64(token) < Int64(kv.shape[0])
                        valid_mask = cute.arch.vote_ballot_sync(valid)
                        if lane_idx == Int32(0):
                            valid_offset = valid_stage * Int32(self.BLOCK_TOPK // self.WARP_SIZE) + Int32(half)
                            sValidWords[valid_offset] = Uint32(valid_mask)
                    cute.arch.fence_view_async_shared()
                    if lane_idx == Int32(0):
                        cute.arch.mbarrier_arrive(valid_ready_mbar_ptr + valid_stage)
                    valid_tile = valid_tile + Int32(1)
        else:
            if warp_idx >= Int32(4) and warp_idx < Int32(8):
                producer_tidx = tidx - Int32(4 * self.WARP_SIZE)
                prime_tile = Int32(0)
                prime_kv_ptr = sKV_base_ptr
                prime_tail_ptr = sKV_tail_base_ptr
                prime_sKV_nope_mma = cute.make_tensor(
                    cute.recast_ptr(prime_kv_ptr, kv_nope_layout.inner),
                    kv_nope_layout.outer,
                )
                prime_sKV_nope_slice = prime_sKV_nope_mma[(None, None), 0, (None, None), 0]
                prime_sKV_nope = cute.composition(
                    prime_sKV_nope_slice,
                    cute.make_layout((self.BLOCK_TOPK, self.VALUE_DIM)),
                )
                prime_sKV_tail_mma = cute.make_tensor(
                    cute.recast_ptr(prime_tail_ptr, kv_tail_layout.inner),
                    kv_tail_layout.outer,
                )
                prime_sKV_tail = cute.composition(
                    prime_sKV_tail_mma,
                    cute.make_layout((self.BLOCK_TOPK, 64)),
                )
                prime_sIndex = cute.make_tensor(s_index_base_ptr, vector_layout)
                prime_sValid = cute.make_tensor(s_valid_base_ptr, valid_layout)
                self._prepare_indices(
                    indices,
                    prime_sIndex,
                    prime_sValid,
                    query_idx,
                    prime_tile,
                    effective_length,
                    physical_topk,
                    kv.shape[0],
                    producer_tidx,
                )
                self.gather_sync_barrier.arrive_and_wait()
                self._gather_kv(
                    kv,
                    prime_sKV_nope,
                    prime_sKV_tail,
                    prime_sIndex,
                    prime_sValid,
                    warp_idx,
                    lane_idx,
                    copy_atom,
                    thread_copy,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
        if const_expr(self.head_dim == 576 and self.use_tma_gather4):
            if warp_idx >= Int32(10) and warp_idx < Int32(12):
                self._gather_kv_tail_cpasync(
                    kv,
                    indices,
                    sKV_tail0,
                    kv_tail_ready_mbar_ptr,
                    kv_tail_free_mbar_ptr,
                    query_idx,
                    Int32(0),
                    effective_length,
                    physical_topk,
                    warp_idx,
                    lane_idx,
                    copy_atom,
                )
        # The gather4 consumers wait their per-stage TMA/tail barriers.  The
        # compatibility cp.async path has no cross-warp completion mbarrier,
        # so it deliberately retains the original CTA-wide prime handoff.
        if const_expr(not self.use_tma_gather4):
            cute.arch.sync_threads()

        # Warp 8 owns one dynamic QK/PV issuer loop on the gather4 path.  The
        # inclusive sentinel iteration drains PV(last): iteration k first
        # issues QK(k), when it exists, and then issues PV(k-1).  This preserves
        # the existing p_free/so_ready ordering while keeping exactly one
        # static QK body and one static PV body in SASS.
        if const_expr(self.use_tma_gather4):
            if warp_idx == Int32(8):
                issuer_tile = Int32(0)
                while issuer_tile <= tile_count:
                    if issuer_tile < tile_count:
                        qk_stage = issuer_tile % Int32(self.num_kv_stages)
                        qk_stage_elem_offset = qk_stage * kv_stage_elements
                        qk_kv_ptr = sKV_base_ptr + cute.assume(
                            qk_stage_elem_offset,
                            divby=self.COPY_ELEMS,
                        )
                        qk_sKV_dual_nope = cute.make_tensor(
                            cute.recast_ptr(qk_kv_ptr, kv_dual_nope_layout.inner),
                            kv_dual_nope_layout.outer,
                        )
                        qk_sKV_dual_nope_slice = qk_sKV_dual_nope[(None, None), 0, (None, None), 0]
                        qk_sKV_dual_nope_logical = cute.composition(
                            qk_sKV_dual_nope_slice,
                            cute.make_layout((2 * self.BLOCK_TOPK, self.VALUE_DIM // 2)),
                        )
                        qk_sKV_dual_tail = cute.make_tensor(
                            cute.recast_ptr(sKV_tail_base_ptr, kv_dual_tail_layout.inner),
                            kv_dual_tail_layout.outer,
                        )
                        qk_sKV_dual_tail_logical = cute.composition(
                            qk_sKV_dual_tail,
                            cute.make_layout((2 * self.BLOCK_TOPK, 32)),
                        )
                        self._qk_mma_issue_double_buffered(
                            tiled_mma_qk,
                            qk_sKV_dual_nope_logical,
                            qk_sKV_dual_tail_logical,
                            mma_done_mbar_ptr,
                            kv_ready_mbar_ptr + qk_stage * Int32(self.num_kv_parts),
                            (issuer_tile // Int32(self.num_kv_stages)) % Int32(2),
                            kv_tail_ready_mbar_ptr,
                            issuer_tile % Int32(2),
                            kv_tail_free_mbar_ptr,
                            q_s2t_done_mbar_ptr,
                            p_free_mbar_ptr,
                            issuer_tile,
                            warp_idx,
                        )

                    if issuer_tile > Int32(0):
                        pv_tile = issuer_tile - Int32(1)
                        pv_stage = pv_tile % Int32(self.num_kv_stages)
                        pv_stage_elem_offset = pv_stage * kv_stage_elements
                        pv_kv_ptr = sKV_base_ptr + cute.assume(
                            pv_stage_elem_offset,
                            divby=self.COPY_ELEMS,
                        )
                        pv_sV_mma0 = cute.make_tensor(
                            cute.recast_ptr(pv_kv_ptr, v_mma_layout.inner),
                            v_mma_layout.outer,
                        )
                        pv_sV_mma1 = cute.make_tensor(
                            cute.recast_ptr(pv_kv_ptr + (self.VALUE_DIM // 2) * self.BLOCK_TOPK, v_mma_layout.inner),
                            v_mma_layout.outer,
                        )
                        self._pv_mma_to_tmem(
                            tiled_mma_pv,
                            sP_mma,
                            pv_sV_mma0,
                            pv_sV_mma1,
                            so_ready_mbar_ptr,
                            pv_done_mbar_ptr + pv_stage,
                            pv_tile,
                            warp_idx,
                        )
                    issuer_tile = issuer_tile + Int32(1)

        scale_log2_e = softmax_scale * Float32(math.log2(math.e))
        tile_idx = Int32(0)
        if const_expr(self.use_tma_gather4):
            if warp_idx >= Int32(4) and warp_idx < Int32(10):
                # WG1, the MMA issuer, and validity producer all finish in
                # their independent dynamic loops above.
                tile_idx = tile_count
        qk_phase = Int32(0)
        while tile_idx < tile_count:
            stage_idx = tile_idx % Int32(self.num_kv_stages)
            stage_elem_offset = stage_idx * kv_stage_elements
            stage_valid_offset = stage_idx * Int32(self.BLOCK_TOPK // 8)
            stage_kv_ptr = sKV_base_ptr + cute.assume(stage_elem_offset, divby=self.COPY_ELEMS)
            stage_tail_ptr = sKV_tail_base_ptr
            tail_stage_idx = Int32(0)
            if const_expr(not self.use_tma_gather4):
                stage_tail_ptr = sKV_tail_base_ptr + cute.assume(stage_elem_offset, divby=self.COPY_ELEMS)
                tail_stage_idx = stage_idx
            sKV_dual_nope = cute.make_tensor(
                cute.recast_ptr(stage_kv_ptr, kv_dual_nope_layout.inner),
                kv_dual_nope_layout.outer,
            )
            sKV_dual_nope_slice = sKV_dual_nope[(None, None), 0, (None, None), 0]
            sKV_dual_nope_logical = cute.composition(
                sKV_dual_nope_slice,
                cute.make_layout((2 * self.BLOCK_TOPK, self.VALUE_DIM // 2)),
            )
            sKV_dual_tail = cute.make_tensor(
                cute.recast_ptr(stage_tail_ptr, kv_dual_tail_layout.inner),
                kv_dual_tail_layout.outer,
            )
            sKV_dual_tail_logical = cute.composition(
                sKV_dual_tail,
                cute.make_layout((2 * self.BLOCK_TOPK, 32)),
            )
            sValid = cute.make_tensor(s_valid_base_ptr + stage_valid_offset, valid_layout)
            sV_mma0 = cute.make_tensor(cute.recast_ptr(stage_kv_ptr, v_mma_layout.inner), v_mma_layout.outer)
            sV_mma1 = cute.make_tensor(
                cute.recast_ptr(stage_kv_ptr + (self.VALUE_DIM // 2) * self.BLOCK_TOPK, v_mma_layout.inner),
                v_mma_layout.outer,
            )

            if const_expr(not self.use_tma_gather4):
                # The compatibility path primes tile zero here.  Gather4's
                # independent warp-8 loop above owns its complete QK chain.
                if tile_idx == Int32(0):
                    self._qk_mma_issue_double_buffered(
                        tiled_mma_qk,
                        sKV_dual_nope_logical,
                        sKV_dual_tail_logical,
                        mma_done_mbar_ptr,
                        kv_ready_mbar_ptr + stage_idx * Int32(self.num_kv_parts),
                        (tile_idx // Int32(self.num_kv_stages)) % Int32(2),
                        kv_tail_ready_mbar_ptr + tail_stage_idx,
                        tile_idx % Int32(2),
                        kv_tail_free_mbar_ptr,
                        q_s2t_done_mbar_ptr,
                        p_free_mbar_ptr,
                        tile_idx,
                        warp_idx,
                    )
            pair_mi, pair_li, pair_real_max = self._scores_consume_double_buffered(
                sPExchange,
                sPExchangeLinear,
                sWeight,
                sValid,
                mma_done_mbar_ptr,
                qk_phase,
                valid_ready_mbar_ptr + stage_idx,
                (tile_idx // Int32(self.num_kv_stages)) % Int32(2),
                valid_free_mbar_ptr + stage_idx,
                p_free_mbar_ptr,
                pv_done_mbar_ptr,
                tile_idx,
                tidx,
                warp_idx,
                scale_log2_e,
                rMi[0],
                rLi[0],
                rRealMax[0],
                sGroupRescale,
                sScale,
            )
            if warp_idx < Int32(4):
                rMi[0] = pair_mi
                rLi[0] = pair_li
                rRealMax[0] = pair_real_max
                if const_expr(self.indexer_topk > 0):
                    if tile_idx == Int32(self.indexer_tile):
                        rMiIndexer[0] = pair_mi
                        rLiIndexer[0] = pair_li
            qk_phase = qk_phase ^ Int32(1)

            # Gather4 runs in WG1's independent producer loop above.  The
            # compatibility path retains its one-tile refill here, while the
            # independent D576 tail warps and QK issuer keep their existing
            # pipelines.
            next_tile = tile_idx + Int32(1)
            if const_expr(self.num_kv_stages > 1):
                if const_expr(not self.use_tma_gather4):
                    # Keep the compatibility producer byte-for-byte equivalent
                    # to the proven two-stage schedule.  In particular, do not
                    # route its divergent warpgroup control flow through the
                    # gather4 path's two-tile prefetch SSA values.
                    if next_tile < tile_count and warp_idx >= Int32(4) and warp_idx < Int32(8):
                        next_stage = next_tile % Int32(self.num_kv_stages)
                        if next_tile >= Int32(self.num_kv_stages):
                            prior_use = next_tile // Int32(self.num_kv_stages) - Int32(1)
                            cute.arch.mbarrier_wait(
                                pv_done_mbar_ptr + next_stage,
                                prior_use % Int32(2),
                            )
                        next_stage_elem_offset = next_stage * kv_stage_elements
                        next_stage_meta_offset = next_stage * Int32(self.BLOCK_TOPK)
                        next_stage_valid_offset = next_stage * Int32(self.BLOCK_TOPK // 8)
                        next_kv_ptr = sKV_base_ptr + cute.assume(next_stage_elem_offset, divby=self.COPY_ELEMS)
                        next_tail_ptr = sKV_tail_base_ptr + cute.assume(
                            next_stage_elem_offset,
                            divby=self.COPY_ELEMS,
                        )
                        next_sKV_nope_mma = cute.make_tensor(
                            cute.recast_ptr(next_kv_ptr, kv_nope_layout.inner),
                            kv_nope_layout.outer,
                        )
                        next_sKV_nope_slice = next_sKV_nope_mma[(None, None), 0, (None, None), 0]
                        next_sKV_nope = cute.composition(
                            next_sKV_nope_slice,
                            cute.make_layout((self.BLOCK_TOPK, self.VALUE_DIM)),
                        )
                        next_sKV_tail_mma = cute.make_tensor(
                            cute.recast_ptr(next_tail_ptr, kv_tail_layout.inner),
                            kv_tail_layout.outer,
                        )
                        next_sKV_tail = cute.composition(
                            next_sKV_tail_mma,
                            cute.make_layout((self.BLOCK_TOPK, 64)),
                        )
                        next_sIndex = cute.make_tensor(
                            s_index_base_ptr + next_stage_meta_offset,
                            vector_layout,
                        )
                        next_sValid = cute.make_tensor(
                            s_valid_base_ptr + next_stage_valid_offset,
                            valid_layout,
                        )
                        producer_tidx = tidx - Int32(4 * self.WARP_SIZE)
                        self._prepare_indices(
                            indices,
                            next_sIndex,
                            next_sValid,
                            query_idx,
                            next_tile,
                            effective_length,
                            physical_topk,
                            kv.shape[0],
                            producer_tidx,
                        )
                        self.gather_sync_barrier.arrive_and_wait()
                        self._gather_kv(
                            kv,
                            next_sKV_nope,
                            next_sKV_tail,
                            next_sIndex,
                            next_sValid,
                            warp_idx,
                            lane_idx,
                            copy_atom,
                            thread_copy,
                        )
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()

                    # Compatibility copies have no transaction-completion
                    # mbarrier for warp 8.  Keep this path deliberately
                    # conservative: all roles observe K(next) before QK(next).
                    cute.arch.sync_threads()

                if const_expr(self.head_dim == 576 and self.use_tma_gather4):
                    if next_tile < tile_count and warp_idx >= Int32(10) and warp_idx < Int32(12):
                        self._gather_kv_tail_cpasync(
                            kv,
                            indices,
                            sKV_tail0,
                            kv_tail_ready_mbar_ptr,
                            kv_tail_free_mbar_ptr,
                            query_idx,
                            next_tile,
                            effective_length,
                            physical_topk,
                            warp_idx,
                            lane_idx,
                            copy_atom,
                        )

                if const_expr(not self.use_tma_gather4):
                    # Compatibility warp 8 issues QK(k+1) before joining the
                    # two PV named barriers.  Gather4 uses the independent
                    # dynamic issuer loop above.
                    if next_tile < tile_count:
                        next_stage = next_tile % Int32(self.num_kv_stages)
                        next_stage_elem_offset = next_stage * kv_stage_elements
                        next_kv_ptr = sKV_base_ptr + cute.assume(next_stage_elem_offset, divby=self.COPY_ELEMS)
                        next_tail_ptr = sKV_tail_base_ptr + cute.assume(
                            next_stage_elem_offset,
                            divby=self.COPY_ELEMS,
                        )
                        next_sKV_dual_nope = cute.make_tensor(
                            cute.recast_ptr(next_kv_ptr, kv_dual_nope_layout.inner),
                            kv_dual_nope_layout.outer,
                        )
                        next_sKV_dual_nope_slice = next_sKV_dual_nope[(None, None), 0, (None, None), 0]
                        next_sKV_dual_nope_logical = cute.composition(
                            next_sKV_dual_nope_slice,
                            cute.make_layout((2 * self.BLOCK_TOPK, self.VALUE_DIM // 2)),
                        )
                        next_sKV_dual_tail = cute.make_tensor(
                            cute.recast_ptr(next_tail_ptr, kv_dual_tail_layout.inner),
                            kv_dual_tail_layout.outer,
                        )
                        next_sKV_dual_tail_logical = cute.composition(
                            next_sKV_dual_tail,
                            cute.make_layout((2 * self.BLOCK_TOPK, 32)),
                        )
                        self._qk_mma_issue_double_buffered(
                            tiled_mma_qk,
                            next_sKV_dual_nope_logical,
                            next_sKV_dual_tail_logical,
                            mma_done_mbar_ptr,
                            kv_ready_mbar_ptr + next_stage * Int32(self.num_kv_parts),
                            (next_tile // Int32(self.num_kv_stages)) % Int32(2),
                            kv_tail_ready_mbar_ptr + next_stage,
                            next_tile % Int32(2),
                            kv_tail_free_mbar_ptr,
                            q_s2t_done_mbar_ptr,
                            p_free_mbar_ptr,
                            next_tile,
                            warp_idx,
                        )

            # O persists in TMEM across all sparse tiles.  WG0 publishes P and
            # a possible O rescale through a one-way mbarrier, then immediately
            # starts consuming the next score tile.  Warp 8 independently
            # waits this handoff before issuing the current PV.
            if warp_idx < Int32(4):
                if tile_idx > Int32(0):
                    # Warps 0 and 1 publish the rescale metadata. Synchronize
                    # all four WG0 warps before any of them consume it.
                    self.softmax_sync_barrier.arrive_and_wait()
                    self._rescale_o_tmem(sScale, sGroupRescale, tidx)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(so_ready_mbar_ptr)

            # Gather4's dynamic warp-8 loop owns PV.  The compatibility path
            # keeps its common-loop issue point and completion protocol.
            if const_expr(not self.use_tma_gather4):
                self._pv_mma_to_tmem(
                    tiled_mma_pv,
                    sP_mma,
                    sV_mma0,
                    sV_mma1,
                    so_ready_mbar_ptr,
                    pv_done_mbar_ptr + stage_idx,
                    tile_idx,
                    warp_idx,
                )

            tile_idx = tile_idx + Int32(1)

        # The final output read must observe the last PV completion.
        final_tile = tile_count - Int32(1)
        final_stage = final_tile % Int32(self.num_kv_stages)
        final_phase = (final_tile // Int32(self.num_kv_stages)) % Int32(2)
        if warp_idx < Int32(4):
            # Only the output consumer needs to observe the final PV. Producer
            # and issuer roles can retire after publishing their own pipeline
            # state before the WG0-local epilogue.
            cute.arch.mbarrier_wait(pv_done_mbar_ptr + final_stage, final_phase)
            self._epilogue_stats(
                max_logits,
                lse,
                lse_indexer,
                attn_sink,
                query_idx,
                warp_idx,
                lane_idx,
                sScale,
                sPExchangeLinear,
                rMi,
                rLi,
                rRealMax,
                rMiIndexer,
                rLiIndexer,
            )
        if warp_idx < Int32(4):
            self._epilogue_o_tma(
                out_tma,
                tma_atom_o,
                query_idx,
                sO,
                sScale,
                tidx,
                warp_idx,
            )
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _prepare_indices(
        self,
        indices: cute.Tensor,
        sIndex: cute.Tensor,
        sValid: cute.Tensor,
        query_idx: Int32,
        tile_idx: Int32,
        effective_length: Int32,
        physical_topk: Int32,
        seqlen_kv,
        tidx: Int32,
    ):
        token_idx = Int32(-1)
        slot = tile_idx * Int32(self.BLOCK_TOPK) + tidx
        if tidx < Int32(self.BLOCK_TOPK):
            # Test the physical bound before reading the padded index tensor.
            if slot < physical_topk:
                token_idx = Int32(indices[query_idx, slot])
        valid = tidx < Int32(self.BLOCK_TOPK) and slot < effective_length and token_idx >= Int32(0) and Int64(token_idx) < Int64(seqlen_kv)
        if tidx < Int32(self.BLOCK_TOPK):
            # Invalid rows must also be OOB at the TMA coordinate level.  A
            # merely masked, otherwise-valid index can still fetch NaNs that
            # would contaminate PV even when its softmax weight is zero.
            sIndex[tidx] = token_idx if valid else Int32(-1)
        # Keep eight validity bits per byte and let every softmax thread load
        # one 32-bit half-tile mask. Ballot preserves the parallel
        # two-warp index load while avoiding 32 scalar shared loads per thread.
        valid_mask = cute.arch.vote_ballot_sync(valid)
        if tidx < Int32(self.BLOCK_TOPK) and (tidx % Int32(self.WARP_SIZE)) == Int32(0):
            sValid_words = cute.make_tensor(
                cute.recast_ptr(sValid.iterator, dtype=Uint32),
                cute.make_layout((self.BLOCK_TOPK // self.WARP_SIZE,), stride=(1,)),
            )
            sValid_words[tidx // Int32(self.WARP_SIZE)] = Uint32(valid_mask)

    @cute.jit
    def _gather_kv_tma_direct(
        self,
        indices: cute.Tensor,
        query_idx: Int32,
        tile_idx: Int32,
        effective_length: Int32,
        logical_tile_is_full: Int32,
        sKV_base_ptr: cute.Pointer,
        tma_atom_kv: cute.CopyAtom,
        mbar_ptr: cute.Pointer,
        transaction_bytes: cutlass.Constexpr[int],
        warp_idx: Int32,
    ):
        """Load gather coordinates into registers and launch one sparse tile."""

        if warp_idx == Int32(4):
            with cute.arch.elect_one():
                first_part_bytes = self.BLOCK_TOPK * 256 * (self.element_dtype.width // 8)
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, first_part_bytes)
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + 1, transaction_bytes - first_part_bytes)

        if warp_idx >= Int32(4) and warp_idx < Int32(8):
            # Load one int4 immediately before its eight gather4 launches.  The
            # short live range matches the source producer and avoids retaining
            # all four coordinate groups across both KV parts.
            local_warp_idx = warp_idx - Int32(4)
            with cute.arch.elect_one():
                tile_slot = tile_idx * Int32(self.BLOCK_TOPK)
                for local_row in cutlass.range_constexpr(self.BLOCK_TOPK // (4 * 4)):
                    row_base = cute.assume(Int32(local_row * 4 * 4) + local_warp_idx * Int32(4), divby=4)
                    row_indices = _ldg_indices_128(indices, query_idx, tile_slot + row_base)
                    row0 = Int32(row_indices[0])
                    row1 = Int32(row_indices[1])
                    row2 = Int32(row_indices[2])
                    row3 = Int32(row_indices[3])
                    if logical_tile_is_full == Int32(0):
                        row_slot = tile_slot + row_base
                        row0 = row0 if row_slot < effective_length else Int32(-1)
                        row1 = row1 if row_slot + Int32(1) < effective_length else Int32(-1)
                        row2 = row2 if row_slot + Int32(2) < effective_length else Int32(-1)
                        row3 = row3 if row_slot + Int32(3) < effective_length else Int32(-1)
                    for part_idx in cutlass.range_constexpr(self.num_kv_parts):
                        col_begin = part_idx * (256 // 64)
                        col_end = (part_idx + 1) * (256 // 64)
                        for local_col in cutlass.range_constexpr(col_begin, col_end):
                            smem_offset = row_base * Int32(64) + Int32(local_col * self.BLOCK_TOPK * 64)
                            dst_ptr = sKV_base_ptr + cute.assume(smem_offset, divby=64)
                            tma_gather4_cta1(
                                tma_atom_kv,
                                dst_ptr,
                                mbar_ptr + part_idx,
                                Int32(local_col * 64),
                                row0,
                                row1,
                                row2,
                                row3,
                            )

    @cute.jit
    def _gather_kv_tail_cpasync(
        self,
        kv: cute.Tensor,
        indices: cute.Tensor,
        sKV_tail: cute.Tensor,
        mbar_ptr: cute.Pointer,
        free_mbar_ptr: cute.Pointer,
        query_idx: Int32,
        tile_idx: Int32,
        effective_length: Int32,
        physical_topk: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
        copy_atom: cute.CopyAtom,
    ):
        """Copy the D=576 SW64 tail with two dedicated warps."""

        if warp_idx >= Int32(10) and warp_idx < Int32(12):
            # The tail has a single physical stage.  Its consumer commits this
            # free barrier immediately after the two tail QK instructions, so
            # copy(k+1) need not wait for the much later PV(k) completion.
            if tile_idx > Int32(0):
                cute.arch.mbarrier_wait(free_mbar_ptr, (tile_idx - Int32(1)) % Int32(2))
            producer_tidx = (warp_idx - Int32(10)) * Int32(self.WARP_SIZE) + lane_idx
            row_group = producer_tidx // Int32(8)
            chunk_idx = producer_tidx % Int32(8)
            for row_iter in cutlass.range_constexpr(self.BLOCK_TOPK // 8):
                row = row_group + Int32(row_iter * 8)
                slot = tile_idx * Int32(self.BLOCK_TOPK) + row
                token_idx = Int32(-1)
                if slot < physical_topk:
                    token_idx = Int32(indices[query_idx, slot])
                valid = slot < effective_length and token_idx >= Int32(0) and Int64(token_idx) < Int64(kv.shape[0])
                tile_sKV_tail = sKV_tail[row, None]
                s_tail_chunk = cute.local_tile(tile_sKV_tail, (self.COPY_ELEMS,), (chunk_idx,))
                if valid:
                    g_row = kv[Int64(token_idx), None]
                    g_tail_ptr = cute.make_ptr(
                        self.element_dtype,
                        (g_row.iterator + self.VALUE_DIM + chunk_idx * Int32(self.COPY_ELEMS)).toint(),
                        mem_space=cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    g_tail_chunk = cute.make_tensor(
                        g_tail_ptr,
                        cute.make_layout((self.COPY_ELEMS,), stride=(1,)),
                    )
                    cute.copy(copy_atom, g_tail_chunk, s_tail_chunk)
                else:
                    s_tail_chunk.fill(self.element_dtype(0.0))
            cute.arch.cp_async_mbarrier_arrive_noinc(mbar_ptr)

    @cute.jit
    def _gather_kv(
        self,
        kv: cute.Tensor,
        sKV_nope: cute.Tensor,
        sKV_tail: cute.Tensor,
        sIndex: cute.Tensor,
        sValid: cute.Tensor,
        warp_idx: Int32,
        lane_idx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        # Match the proven DSA backward gather partition: four producer warps,
        # one row per warp at a time, with each row split into 8-thread x
        # 128-bit copy groups.  This partition respects the SW128 operand
        # layout consumed by tcgen05, unlike treating its physical pointer as
        # a flat row-major allocation.
        if warp_idx >= Int32(4) and warp_idx < Int32(8):
            local_warp_idx = warp_idx - Int32(4)
            async_thread_copy = thread_copy.get_slice(lane_idx % Int32(8))
            for row_iter in cutlass.range_constexpr(self.BLOCK_TOPK // 4):
                row = Int32(row_iter * 4) + local_warp_idx
                token_idx = Int32(sIndex[row])
                valid_byte = Uint32(sValid[row // Int32(8)])
                valid = (valid_byte & (Uint32(1) << (row % Int32(8)))) != Uint32(0)
                tile_sKV_nope = sKV_nope[row, (None, None)]
                tile_sKV_tail = sKV_tail[row, None]

                # The global row pointer is formed only in the valid branch;
                # this is stronger than merely predicating a previously formed
                # OOB address.  The branch is the cp.async predicate and the
                # false path explicitly zero-fills the complete SW128 row.
                if valid:
                    self._copy_kv_row(
                        kv,
                        token_idx,
                        tile_sKV_nope,
                        tile_sKV_tail,
                        lane_idx,
                        copy_atom,
                        async_thread_copy,
                    )
                else:
                    self._zero_kv_row(tile_sKV_nope, tile_sKV_tail, lane_idx)

    @cute.jit
    def _copy_kv_row(
        self,
        kv: cute.Tensor,
        token_idx: Int32,
        tile_sKV_nope: cute.Tensor,
        tile_sKV_tail: cute.Tensor,
        lane_idx: Int32,
        copy_atom: cute.CopyAtom,
        async_thread_copy: cute.TiledCopy,
    ):
        # Promote before applying the potentially large row stride.  Keeping
        # the token as Int32 here can wrap the address calculation for large
        # Tkv even though the range check itself used Int64.
        g_row = kv[Int64(token_idx), None]
        tile_gKV = cute.composition(g_row, cute.make_layout(tile_sKV_nope.shape))
        for half in cutlass.range_constexpr(2):
            group_idx = Int32(half * 4) + lane_idx // Int32(8)
            t_g = async_thread_copy.partition_S(tile_gKV[None, group_idx])
            t_s = async_thread_copy.partition_D(tile_sKV_nope[None, group_idx])
            cute.copy(copy_atom, t_g, t_s)

        if const_expr(self.head_dim == 576):
            tile_gKV_tail = cute.make_tensor(
                g_row.iterator + self.VALUE_DIM,
                cute.make_layout(tile_sKV_tail.shape),
            )
            t_g = async_thread_copy.partition_S(tile_gKV_tail)
            t_s = async_thread_copy.partition_D(tile_sKV_tail)
            if lane_idx < Int32(8):
                cute.copy(copy_atom, t_g, t_s)

    @cute.jit
    def _zero_kv_row(self, tile_sKV_nope: cute.Tensor, tile_sKV_tail: cute.Tensor, lane_idx: Int32):
        for half in cutlass.range_constexpr(2):
            group_idx = Int32(half * 4) + lane_idx // Int32(8)
            chunked = cute.flat_divide(tile_sKV_nope[None, group_idx], (self.COPY_ELEMS,))
            chunked[None, lane_idx % Int32(8)].fill(self.element_dtype(0.0))

        if const_expr(self.head_dim == 576):
            chunked = cute.flat_divide(tile_sKV_tail, (self.COPY_ELEMS,))
            if lane_idx < Int32(8):
                chunked[None, lane_idx].fill(self.element_dtype(0.0))

    @cute.jit
    def _qk_mma_issue_double_buffered(
        self,
        tiled_mma_qk: cute.TiledMma,
        sKV_dual_nope: cute.Tensor,
        sKV_dual_tail: cute.Tensor,
        mma_done_mbar,
        kv_ready_mbar,
        kv_phase: Int32,
        kv_tail_ready_mbar,
        kv_tail_phase: Int32,
        kv_tail_free_mbar,
        q_s2t_done_mbar,
        p_free_mbar,
        tile_idx: Int32,
        warp_idx: Int32,
    ):
        """Issue one QK tile after the prior score readers release TMEM P."""

        if warp_idx == Int32(8):
            # The score accumulator has one physical TMEM stage.  Tile k may
            # overwrite it only after all 128 WG0 threads have completed their
            # synchronous Ld32x32 reads for tile k-1.
            if tile_idx > Int32(0):
                cute.arch.mbarrier_wait(p_free_mbar, (tile_idx - Int32(1)) % Int32(2))

            # On gather4, S2T completion both orders Q before QK0 and publishes
            # the aliased Q-NoPE stage to the K2 producer.  The fallback keeps
            # its original CTA-synchronized same-warp instruction ordering.
            if const_expr(self.use_tma_gather4):
                if tile_idx == Int32(0):
                    cute.arch.mbarrier_wait(q_s2t_done_mbar, 0)
                    tcgen05_fence_after_thread_sync()

            # The D=576 producer dedicates warps 10/11 to the SW64 tail. Their
            # 64 cp.async arrivals have an independent barrier,
            # so the two tail MMAs can run before either NoPE gather half is
            # complete and initialize P without joining a 128-thread wait.
            if const_expr(self.head_dim == 576):
                if const_expr(self.use_tma_gather4):
                    cute.arch.mbarrier_wait(kv_tail_ready_mbar, kv_tail_phase)
                    cute.arch.fence_view_async_shared()
                    tcgen05_fence_after_thread_sync()
                else:
                    tcgen05_fence_after_thread_sync()
                for k_block in cutlass.range_constexpr(2):
                    sK_tile = cute.local_tile(
                        sKV_dual_tail,
                        (2 * self.BLOCK_TOPK, 16),
                        (0, k_block),
                    )
                    mma_ws_ts_f16_cta1(
                        tiled_mma_qk.op,
                        Int32(self.TMEM_P_OFFSET),
                        Int32(self.TMEM_Q_OFFSET + 128 + k_block * 8),
                        sK_tile,
                        k_block != 0,
                    )
                if const_expr(self.use_tma_gather4):
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_tail_free_mbar)

            # The physical 64x512 K tile is viewed as 128x256.  Each original
            # 256-d TMA half becomes one K128 implicit-dual phase and consumes
            # the matching 64-column Q TMEM segment.  Wait immediately before
            # each half so the first eight MMAs overlap the second gather4
            # half (plus D=576's cp.async tail).
            for part_idx in cutlass.range_constexpr(2):
                if const_expr(self.use_tma_gather4):
                    cute.arch.mbarrier_wait(kv_ready_mbar + part_idx, kv_phase)
                    cute.arch.fence_view_async_shared()
                    tcgen05_fence_after_thread_sync()
                elif const_expr(self.head_dim == 512 and part_idx == 0):
                    tcgen05_fence_after_thread_sync()
                for k_block in cutlass.range_constexpr(8):
                    sK_tile = cute.local_tile(
                        sKV_dual_nope,
                        (2 * self.BLOCK_TOPK, 16),
                        (0, part_idx * 8 + k_block),
                    )
                    mma_ws_ts_f16_cta1(
                        tiled_mma_qk.op,
                        Int32(self.TMEM_P_OFFSET),
                        Int32(self.TMEM_Q_OFFSET + part_idx * 64 + k_block * 8),
                        sK_tile,
                        self.head_dim == 576 or part_idx != 0 or k_block != 0,
                    )
            with cute.arch.elect_one():
                tcgen05.commit(mma_done_mbar)

    @cute.jit
    def _scores_consume_double_buffered(
        self,
        sPExchange: cute.Tensor,
        sPExchangeLinear: cute.Tensor,
        sWeight: cute.Tensor,
        sValid: cute.Tensor,
        mma_done_mbar,
        mma_phase: Int32,
        valid_ready_mbar,
        valid_phase: Int32,
        valid_free_mbar,
        p_free_mbar,
        pv_done_mbar,
        tile_idx: Int32,
        tidx: Int32,
        warp_idx: Int32,
        scale_log2_e: Float32,
        old_mi: Float32,
        old_li: Float32,
        old_real_max: Float32,
        sGroupRescale: cute.Tensor,
        sScale: cute.Tensor,
    ):
        """Consume one QK result and materialize its online-softmax P tile."""

        new_mi = old_mi
        new_li = old_li
        new_real_max = old_real_max

        if warp_idx < Int32(4):
            cute.arch.mbarrier_wait(mma_done_mbar, mma_phase)
            if const_expr(self.use_tma_gather4):
                cute.arch.mbarrier_wait(valid_ready_mbar, valid_phase)
            tcgen05_fence_after_thread_sync()
            local_warp = tidx // Int32(self.WARP_SIZE)
            lane_idx = tidx % Int32(self.WARP_SIZE)
            head_idx = (local_warp & Int32(1)) * Int32(self.WARP_SIZE) + lane_idx
            # The public M64N128 object models a non-WS accumulator fragment,
            # while mma.ws uses tmem_frg_ws_1sm. Load the two physical
            # 32-column pieces directly with tmem_ld_32dp32bNx.
            if const_expr(self.indexer_topk > 0):
                # Put the warp-owned score half in r_left and the peer partial
                # in r_right up front.  This keeps the exchange below
                # straight-line instead of merging two 32-value SSA fragments
                # at the local-warp role join.
                owned_half = local_warp // Int32(2)
                peer_half = owned_half ^ Int32(1)
                left_values = tmem_load_32dp32b32x(Int32(self.TMEM_P_OFFSET) + owned_half * Int32(self.BLOCK_TOPK // 2))
                right_values = tmem_load_32dp32b32x(Int32(self.TMEM_P_OFFSET) + peer_half * Int32(self.BLOCK_TOPK // 2))
            else:
                left_values = tmem_load_32dp32b32x(Int32(self.TMEM_P_OFFSET))
                right_values = tmem_load_32dp32b32x(Int32(self.TMEM_P_OFFSET + 32))
            r_left = cute.make_rmem_tensor((self.BLOCK_TOPK // 2,), Float32)
            r_right = cute.make_rmem_tensor((self.BLOCK_TOPK // 2,), Float32)
            for elem in cutlass.range_constexpr(self.BLOCK_TOPK // 2):
                r_left[elem] = left_values[elem]
                r_right[elem] = right_values[elem]
            cute.arch.fence_view_async_tmem_load()
            tcgen05_fence_before_thread_sync()
            # Release the single score-TMEM stage as soon as its values are in
            # registers.  Warp 8 can now issue QK(k+1) while WG0 performs the
            # warp exchange, softmax, and P stores for tile k.
            cute.arch.mbarrier_arrive(p_free_mbar)

            # Each N=128 result contains two dot-product partials.  Paired
            # warps exchange the non-owned half as aligned float4 vectors and
            # add it in registers, matching retrieve_mask_and_reduce_p().
            sValid_words_ptr = cute.make_ptr(
                Uint32,
                sValid.iterator.llvm_ptr,
                cute.AddressSpace.smem,
                assumed_align=4,
            )
            sValid_words = cute.make_tensor(
                sValid_words_ptr,
                cute.make_layout((self.BLOCK_TOPK // self.WARP_SIZE,), stride=(1,)),
            )
            if const_expr(self.indexer_topk > 0):
                valid_word = Uint32(sValid_words[owned_half])
            else:
                valid_word = Uint32(sValid_words[0 if local_warp < Int32(2) else 1])
            if const_expr(self.use_tma_gather4):
                cute.arch.mbarrier_arrive(valid_free_mbar)
            for elem in cutlass.range_constexpr(self.BLOCK_TOPK // 2):
                invalid = (valid_word & (Uint32(1) << elem)) == Uint32(0)
                if const_expr(self.indexer_topk > 0):
                    r_left[elem] = -Float32.inf if invalid else r_left[elem]
                else:
                    mask_left = invalid and local_warp < Int32(2)
                    mask_right = invalid and local_warp >= Int32(2)
                    r_left[elem] = -Float32.inf if mask_left else r_left[elem]
                    r_right[elem] = -Float32.inf if mask_right else r_right[elem]
            half_layout = cute.make_layout((self.BLOCK_TOPK // 2,), stride=(1,))
            r_left_linear = cute.make_tensor(r_left.iterator, half_layout)
            r_right_linear = cute.make_tensor(r_right.iterator, half_layout)
            r_left_vectors = cute.flat_divide(r_left_linear, (4,))
            r_right_vectors = cute.flat_divide(r_right_linear, (4,))
            peer_warp = local_warp ^ Int32(2)
            if const_expr(self.indexer_topk > 0):
                for vector in cutlass.range_constexpr((self.BLOCK_TOPK // 2) // 4):
                    cute.autovec_copy(
                        r_right_vectors[None, vector],
                        sPExchange[peer_warp, vector, lane_idx, None],
                    )
            else:
                if local_warp < Int32(2):
                    for vector in cutlass.range_constexpr((self.BLOCK_TOPK // 2) // 4):
                        cute.autovec_copy(
                            r_right_vectors[None, vector],
                            sPExchange[peer_warp, vector, lane_idx, None],
                        )
                else:
                    for vector in cutlass.range_constexpr((self.BLOCK_TOPK // 2) // 4):
                        cute.autovec_copy(
                            r_left_vectors[None, vector],
                            sPExchange[peer_warp, vector, lane_idx, None],
                        )
            self.softmax_sync_barrier.arrive_and_wait()
            if const_expr(self.indexer_topk > 0):
                for vector in cutlass.range_constexpr((self.BLOCK_TOPK // 2) // 4):
                    cute.autovec_copy(
                        sPExchange[local_warp, vector, lane_idx, None],
                        r_right_vectors[None, vector],
                    )
            else:
                if local_warp < Int32(2):
                    for vector in cutlass.range_constexpr((self.BLOCK_TOPK // 2) // 4):
                        cute.autovec_copy(
                            sPExchange[local_warp, vector, lane_idx, None],
                            r_right_vectors[None, vector],
                        )
                else:
                    for vector in cutlass.range_constexpr((self.BLOCK_TOPK // 2) // 4):
                        cute.autovec_copy(
                            sPExchange[local_warp, vector, lane_idx, None],
                            r_left_vectors[None, vector],
                        )
            self.softmax_sync_barrier.arrive_and_wait()
            for elem in cutlass.range_constexpr(0, self.BLOCK_TOPK // 2, 2):
                r_left[elem], r_left[elem + 1] = cute.arch.add_packed_f32x2(
                    (r_left[elem], r_left[elem + 1]),
                    (r_right[elem], r_right[elem + 1]),
                )

            half_max = -Float32.inf
            for elem in cutlass.range_constexpr(self.BLOCK_TOPK // 2):
                half_max = fmax_ftz_nonan(half_max, r_left[elem])
            half_max *= scale_log2_e
            sPExchangeLinear[tidx] = half_max
            self.softmax_sync_barrier.arrive_and_wait()
            peer_half_max = Float32(sPExchangeLinear[tidx ^ Int32(64)])
            # Drain every peer read before the next tile overwrites the aliased
            # score-exchange surface.
            self.softmax_sync_barrier.arrive_and_wait()
            tile_max = peer_half_max if peer_half_max > half_max else half_max
            new_real_max = old_real_max if old_real_max > tile_max else tile_max
            should_rescale = cute.arch.vote_any_sync(tile_max - old_mi > Float32(self.RESCALE_THRESHOLD_LOG2))
            scale_for_old = Float32(1.0)
            if should_rescale:
                new_mi = tile_max if tile_max > old_mi else old_mi
                scale_for_old = cute.math.exp2(old_mi - new_mi, fastmath=True)

            half_sum_pair = (Float32(0.0), Float32(0.0))
            r_probability = cute.make_rmem_tensor((self.BLOCK_TOPK // 2,), self.element_dtype)
            for elem in cutlass.range_constexpr(0, self.BLOCK_TOPK // 2, 2):
                probability_0, probability_1 = cute.arch.fma_packed_f32x2(
                    (r_left[elem], r_left[elem + 1]),
                    (scale_log2_e, scale_log2_e),
                    (-new_mi, -new_mi),
                )
                probability_0 = cute.math.exp2(probability_0, fastmath=True)
                probability_1 = cute.math.exp2(probability_1, fastmath=True)
                half_sum_pair = cute.arch.add_packed_f32x2(half_sum_pair, (probability_0, probability_1))
                r_probability[elem] = self.element_dtype(probability_0)
                r_probability[elem + 1] = self.element_dtype(probability_1)

            half_sum = half_sum_pair[0] + half_sum_pair[1]

            # Li stays half-row partial until epilogue. Mi and the real maximum
            # are identical in paired warps after every tile.
            new_li = old_li * scale_for_old + half_sum
            if local_warp < Int32(2):
                sScale[head_idx] = scale_for_old
                if lane_idx == Int32(0):
                    sGroupRescale[local_warp] = Int32(1) if should_rescale else Int32(0)

            # Keep exp and 16-bit conversion overlapped with PV(k-1).  Only the
            # first store that overwrites the single physical P stage needs to
            # wait for that asynchronous reader to finish.
            if tile_idx > Int32(0):
                prior_tile = tile_idx - Int32(1)
                prior_stage = prior_tile % Int32(self.num_kv_stages)
                prior_phase = (prior_tile // Int32(self.num_kv_stages)) % Int32(2)
                cute.arch.mbarrier_wait(pv_done_mbar + prior_stage, prior_phase)

            # Four aligned 16-bit x8 stores per thread reproduce the source's
            # uint128 P writes.  Keep the existing K_SW128 P operand layout.
            r_probability_vectors = cute.flat_divide(r_probability, (8,))
            s_weight_vectors = cute.flat_divide(sWeight[head_idx, None], (8,))
            if local_warp < Int32(2):
                for vector in cutlass.range_constexpr((self.BLOCK_TOPK // 2) // 8):
                    cute.autovec_copy(
                        r_probability_vectors[None, vector],
                        s_weight_vectors[None, vector],
                    )
            else:
                for vector in cutlass.range_constexpr((self.BLOCK_TOPK // 2) // 8):
                    cute.autovec_copy(
                        r_probability_vectors[None, vector],
                        s_weight_vectors[None, vector + self.BLOCK_TOPK // 16],
                    )
            # ``so_ready_mbar`` is initialized with 128 arrivals and every WG0
            # thread executes a generic-proxy fence before arriving below.  Its
            # completion therefore publishes the whole P tile to warp 8; an
            # additional 128-thread named barrier here would only serialize
            # WG0.

        return new_mi, new_li, new_real_max

    @cute.jit
    def _pv_mma_to_tmem(
        self,
        tiled_mma_pv: cute.TiledMma,
        sP_mma: cute.Tensor,
        sV_mma0: cute.Tensor,
        sV_mma1: cute.Tensor,
        so_ready_mbar,
        pv_done_mbar,
        tile_idx: Int32,
        warp_idx: Int32,
    ):
        """Accumulate both 64x256 P@V halves into persistent TMEM O."""

        if warp_idx == Int32(8):
            # WG0's 128 arrivals publish both the generic-proxy P stores and a
            # possible TMEM O rescale without forcing either role to execute
            # in lockstep.
            cute.arch.mbarrier_wait(so_ready_mbar, tile_idx % Int32(2))
            tcgen05_fence_after_thread_sync()
            sP_slice = sP_mma[(None, None), 0, None, 0]
            sP_logical = cute.composition(
                sP_slice,
                cute.make_layout((self.NUM_HEADS, self.BLOCK_TOPK)),
            )
            for value_half in cutlass.range_constexpr(2):
                sV_mma = sV_mma0 if const_expr(value_half == 0) else sV_mma1
                sV_slice = sV_mma[((None, None), None), 0, None, 0]
                sV_logical = cute.composition(
                    sV_slice,
                    cute.make_layout((self.VALUE_DIM // 2, self.BLOCK_TOPK)),
                )
                # The WS M64N256 C fragment occupies 128 physical TMEM rows
                # by 128 columns.  Its two N256 logical halves therefore sit
                # at columns 0 and 128.  The public non-WS C fragment encodes
                # the second half with a row bit (+0x100000), which is invalid
                # for mma.ws and traps as a misaligned TMEM address.
                tO_half_addr = Int32(self.TMEM_O_OFFSET + value_half * self.TMEM_O_HALF_COLUMNS)
                # Each half owns disjoint TMEM rows, so both halves clear on
                # tile zero and independently accumulate on later tiles.
                for k_block in cutlass.range_constexpr(self.BLOCK_TOPK // 16):
                    sP_tile = cute.local_tile(
                        sP_logical,
                        (self.NUM_HEADS, 16),
                        (0, k_block),
                    )
                    sV_tile = cute.local_tile(
                        sV_logical,
                        (self.VALUE_DIM // 2, 16),
                        (0, k_block),
                    )
                    mma_ws_ss_f16_cta1(
                        tiled_mma_pv.op,
                        tO_half_addr,
                        sP_tile,
                        sV_tile,
                        tile_idx != Int32(0) or k_block != 0,
                    )
            with cute.arch.elect_one():
                tcgen05.commit(pv_done_mbar)

    @cute.jit
    def _rescale_o_tmem(
        self,
        sScale: cute.Tensor,
        group_rescale: cute.Tensor,
        tidx: Int32,
    ):
        """Apply online-softmax correction using the actual 1-CTA MMA layout."""

        tcgen05_fence_after_thread_sync()
        # mma.ws uses a 128-data-path C fragment. Address each 32-column slice
        # directly; the public non-WS C fragment encodes
        # a different row interleave and cannot safely describe these loads.
        if group_rescale[0] != Int32(0) or group_rescale[1] != Int32(0):
            head_idx = tidx % Int32(self.NUM_HEADS)
            for chunk in cutlass.range_constexpr((self.VALUE_DIM // 2) // self.TMEM_IO_CHUNK):
                values = tmem_load_32dp32b32x(Int32(self.TMEM_O_OFFSET + chunk * self.TMEM_IO_CHUNK))
                cute.arch.fence_view_async_tmem_load()
                r_o = cute.make_rmem_tensor((self.TMEM_IO_CHUNK,), Float32)
                for elem in cutlass.range_constexpr(self.TMEM_IO_CHUNK):
                    r_o[elem] = values[elem] * Float32(sScale[head_idx])
                tmem_store_32dp32b32x(Int32(self.TMEM_O_OFFSET + chunk * self.TMEM_IO_CHUNK), r_o)
                cute.arch.fence_view_async_tmem_store()
        tcgen05_fence_before_thread_sync()

    @cute.jit
    def _epilogue_stats(
        self,
        max_logits: cute.Tensor,
        lse: cute.Tensor,
        lse_indexer: Optional[cute.Tensor],
        attn_sink: Optional[cute.Tensor],
        query_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
        sScale: cute.Tensor,
        sPExchangeLinear: cute.Tensor,
        rMi: cute.Tensor,
        rLi: cute.Tensor,
        rRealMax: cute.Tensor,
        rMiIndexer: cute.Tensor,
        rLiIndexer: cute.Tensor,
    ):
        ln2 = Float32(math.log(2.0))
        log2_e = Float32(math.log2(math.e))
        if warp_idx < Int32(4):
            sPExchangeLinear[warp_idx * Int32(self.WARP_SIZE) + lane_idx] = rLi[0]
            if const_expr(self.indexer_topk > 0 and lse_indexer is not None):
                sPExchangeLinear[Int32(4 * self.WARP_SIZE) + warp_idx * Int32(self.WARP_SIZE) + lane_idx] = rLiIndexer[0]
        self.softmax_sync_barrier.arrive_and_wait()
        # Warps 0/1 are the unique row owners; warps 2/3 contributed the
        # other 32-token half and published their partial Li above.
        if warp_idx < Int32(2):
            head_idx = warp_idx * Int32(self.WARP_SIZE) + lane_idx
            mi = rMi[0]
            peer_tidx = (warp_idx ^ Int32(2)) * Int32(self.WARP_SIZE) + lane_idx
            li = rLi[0] + Float32(sPExchangeLinear[peer_tidx])
            real_max = rRealMax[0]
            has_valid = real_max != -Float32.inf
            if has_valid:
                max_logits[query_idx, head_idx] = real_max * ln2
                lse[query_idx, head_idx] = mi * ln2 + cute.math.log(li, fastmath=True)
            else:
                max_logits[query_idx, head_idx] = -Float32.inf
                lse[query_idx, head_idx] = Float32.inf

            if const_expr(self.indexer_topk > 0 and lse_indexer is not None):
                indexer_li = rLiIndexer[0] + Float32(sPExchangeLinear[Int32(4 * self.WARP_SIZE) + peer_tidx])
                if indexer_li > Float32(0.0):
                    lse_indexer[query_idx, head_idx] = rMiIndexer[0] * ln2 + cute.math.log(indexer_li, fastmath=True)
                else:
                    lse_indexer[query_idx, head_idx] = Float32.inf

            output_scale = Float32(0.0)
            if has_valid:
                sink_log2 = -Float32.inf
                if const_expr(attn_sink is not None):
                    sink_log2 = Float32(attn_sink[head_idx]) * log2_e
                denom = li + cute.math.exp2(sink_log2 - mi, fastmath=True)
                output_scale = Float32(1.0) / denom
            sScale[head_idx] = output_scale
        self.softmax_sync_barrier.arrive_and_wait()

    @cute.jit
    def _epilogue_o_tma(
        self,
        out_tma: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        query_idx: Int32,
        sO: cute.Tensor,
        sScale: cute.Tensor,
        tidx: Int32,
        warp_idx: Int32,
    ):
        """Stage persistent O in the drained arena and emit eight TMA stores."""

        tcgen05_fence_after_thread_sync()
        head_idx = tidx % Int32(self.NUM_HEADS)
        row_half = tidx // Int32(self.NUM_HEADS)
        vector_layout = cute.make_layout((8,), stride=(1,))
        for value_half in cutlass.range_constexpr(2):
            for chunk64 in cutlass.range_constexpr(2):
                dim_tile = Int32(value_half * 4) + row_half * Int32(2) + Int32(chunk64)
                s_out_vectors = cute.flat_divide(sO[None, head_idx, dim_tile], vector_layout)
                for chunk32 in cutlass.range_constexpr(2):
                    tmem_col = value_half * self.TMEM_O_HALF_COLUMNS + chunk64 * 64 + chunk32 * self.TMEM_IO_CHUNK
                    values = tmem_load_32dp32b32x(Int32(self.TMEM_O_OFFSET + tmem_col))
                    cute.arch.fence_view_async_tmem_load()
                    r_out = cute.make_rmem_tensor((self.TMEM_IO_CHUNK,), self.element_dtype)
                    for elem in cutlass.range_constexpr(self.TMEM_IO_CHUNK):
                        r_out[elem] = self.element_dtype(values[elem] * Float32(sScale[head_idx]))
                    r_out_vectors = cute.flat_divide(r_out, vector_layout)
                    for vector in cutlass.range_constexpr(self.TMEM_IO_CHUNK // 8):
                        cute.autovec_copy(
                            r_out_vectors[None, vector],
                            s_out_vectors[
                                None,
                                Int32(chunk32 * (self.TMEM_IO_CHUNK // 8) + vector),
                            ],
                        )

                # Each round materializes one 64x64 chunk per row half.  WG0's
                # existing 128-thread barrier publishes both chunks before
                # warps 0/1 independently launch their corresponding stores.
                cute.arch.fence_view_async_shared()
                self.softmax_sync_barrier.arrive_and_wait()
                if warp_idx == Int32(0):
                    chunk_idx = value_half * 4 + chunk64
                    gO = cute.local_tile(
                        out_tma,
                        (64, self.NUM_HEADS),
                        (chunk_idx, 0, query_idx),
                    )
                    store_o, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_o,
                        0,
                        cute.make_layout(1),
                        sO[None, None, chunk_idx],
                        gO,
                        single_stage=True,
                    )
                    store_o()
                if warp_idx == Int32(1):
                    chunk_idx = value_half * 4 + 2 + chunk64
                    gO = cute.local_tile(
                        out_tma,
                        (64, self.NUM_HEADS),
                        (chunk_idx, 0, query_idx),
                    )
                    store_o, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_o,
                        0,
                        cute.make_layout(1),
                        sO[None, None, chunk_idx],
                        gO,
                        single_stage=True,
                    )
                    store_o()
        tcgen05_fence_before_thread_sync()
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)


__all__ = ["SparseAttentionForwardSm100Head64"]
