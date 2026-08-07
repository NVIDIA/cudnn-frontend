# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
A fused multi-head attention (FMHA) FP16/BF16 kernel for the NVIDIA Blackwell
SM120 family (SM120 and SM121) using TMA K/V loads.

This example demonstrates an implementation of fused multi-head attention using
SM80-era ``mma.sync.aligned.m16n8k16`` tensor cores and TMA loads for K/V tiles.
The implementation fuses the Q*K^T matrix multiplication, softmax normalization,
and softmax(Q*K^T)*V into a single kernel, avoiding intermediate data movement
through global memory.

The kernel implements key optimizations including:
- A dedicated load warp that streams K/V tiles into shared memory using TMA and
  mbarrier completion
- Q fragments loaded directly from global memory into registers to reduce shared
  memory footprint
- Online softmax fused into the main loop, with per-lane registers and intra-warp
  threadquad shuffles for max/sum reductions across the 4 lanes that share a Q-row
- Shared-memory epilogue staging that aliases the K/V buffer after compute warps
  finish reading the last tile
- MHA, GQA, and MQA head mapping
- Top-left or bottom-right causal masks, sliding windows, and per-batch lengths
- Per-row natural-log LSE (softmax stats) output
- Optional per-Q-head attention-sink logits folded into the softmax denominator
- THD (ragged / packed variable-length) batches via cu_seqlens token offsets,
  with no per-sequence descriptors: Q/O use direct addressing and K/V bias a
  single TMA descriptor's seq coordinate

Constraints:
* Supported input dtypes: Float16 and BFloat16, output dtype must match input dtype
* Head dimension must be a multiple of 16 between 16 and 256, inclusive
* Q heads must be divisible by the number of K/V heads
* Q/K/V/O use compact BSHD storage
* Supported CTA Q/KV tiles are 128 or 64
* K/V pipeline and output staging storage must fit within the target SM120 SMEM
  capacity
"""

from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute

from cutlass.experimental import primitives as prims
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16
from cudnn.frost.tile_dsl.mma import ptx_mma_m16n8k16_f32
from cudnn.frost.tile_dsl.swizzle import swizzle_xor
from cudnn.sdpa.fwd.config_sm120 import (
    SEQ_KV_TILES as _SEQ_KV_TILES,
    SEQ_Q_TILES as _SEQ_Q_TILES,
    SUPPORTED_HEAD_TILES as _SUPPORTED_HEAD_TILES,
    TemplateParams,
    validate_params,
)

# The FROST loader injects one immutable specialization before executing this
# module. A direct import uses the dense FP16 defaults.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
validate_params(PARAMS)

STORAGE_DTYPE = {DTYPE_FP16: cutlass.Float16, DTYPE_BF16: cutlass.BFloat16}[PARAMS.dtype_qkv]

# ---------------------------------------------------------------------------
# PTX and layout helpers.
# ---------------------------------------------------------------------------


@cute.jit
def nvvm_threadquad_reduction_max(val: cutlass.Float32) -> cutlass.Float32:
    """Butterfly thread-quad (4 lanes) reduction max via shfl.sync.bfly."""
    val = cute.arch.fmax(
        val,
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=val,
            offset=2,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        ),
    )
    val = cute.arch.fmax(
        val,
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=val,
            offset=1,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        ),
    )
    return val


@cute.jit
def nvvm_threadquad_reduction_sum(val: cutlass.Float32) -> cutlass.Float32:
    """Butterfly thread-quad (4 lanes) reduction sum via shfl.sync.bfly."""
    val = val + prims.shfl_sync(
        thread_mask=0xFFFFFFFF,
        val=val,
        offset=2,
        mask_and_clamp=0x1F,
        kind=prims.Shfl.BFLY,
    )
    val = val + prims.shfl_sync(
        thread_mask=0xFFFFFFFF,
        val=val,
        offset=1,
        mask_and_clamp=0x1F,
        kind=prims.Shfl.BFLY,
    )
    return val


@cute.jit
def pack_to_i32(
    src: tuple,
    dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> cutlass.Int32:
    """Pack four 8-bit or two 16-bit values into one 32-bit register."""
    vals = cutlass.Vector.from_elements(src, dtype)
    return vals.bitcast(cutlass.Int32)[0]


def ceil_div(a: int, b: int) -> int:
    """Return the ceiling division of a by b."""
    return (a + b - 1) // b


fmul2 = partial(prims.mul_packed_f32x2, ftz=False, rnd=prims.FPRoundingMode.RN)
fma2 = partial(prims.fma_packed_f32x2, ftz=False, rnd=prims.FPRoundingMode.RN)


# ---------------------------------------------------------------------------
# Main kernel class
# ---------------------------------------------------------------------------


class SM120FusedMultiHeadAttentionForward:
    """Configure and launch the SM120/SM121 FP16/BF16 FMHA prefill kernel."""

    SEQ_Q_TILES = _SEQ_Q_TILES
    SEQ_KV_TILES = _SEQ_KV_TILES
    SUPPORTED_HEAD_TILES = _SUPPORTED_HEAD_TILES
    MMA_TILER = (16, 8, 16)  # mma.sync.aligned.m16n8k16

    @staticmethod
    def is_layout_supported(
        shape: tuple[int, ...],
        stride: tuple[int, ...],
    ) -> bool:
        """Return whether a BSHD tensor uses compact storage."""

        if len(shape) != 4 or len(stride) != 4:
            return False
        _, sequence, heads, head_dim = shape
        return stride == (
            sequence * heads * head_dim,
            heads * head_dim,
            head_dim,
            1,
        )

    def __init__(
        self,
        in_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        out_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        is_causal: bool = False,
        causal_bottom_right: bool = False,
        window_size_left: int | None = None,
        seq_q_lens_present: bool = False,
        seq_kv_lens_present: bool = False,
        has_sink: bool = False,
        thd_varlen: bool = False,
        thd_batch: int = 1,
        thd_max_sq: int = 0,
        head_tile_qk: int = 128,
        head_tile_v: int = 128,
        kv_tile: int = SEQ_KV_TILES[0],
        q_tile: int = SEQ_Q_TILES[0],
    ):
        """Initialize the FMHA prefill kernel configuration.

        :param in_dtype: Q/K/V element type (Float16 or BFloat16).
        :param out_dtype: O element type. Must match ``in_dtype``.
        :param is_causal: Apply an upper causal bound to QK.
        :param causal_bottom_right: Shift the causal diagonal by ``Skv - Sq``.
        :param window_size_left: Inclusive left-window offset, or ``None``.
        :param seq_q_lens_present: Read per-batch query lengths at runtime.
        :param seq_kv_lens_present: Read per-batch key/value lengths at runtime.
        :param has_sink: Fold the per-Q-head sink logit from the ``sinks``
            tensor into the softmax denominator; when ``False`` the ``sinks``
            argument is an unused dummy.
        :param thd_varlen: THD (ragged) mode — Q/K/V/O and LSE are fully
            packed batch-1 views, ``seq_kv_lens`` is the
            ``[seq_kv(B) | cu_q(B+1) | cu_k(B+1)]`` metadata tensor, and the
            grid covers ``ceil(thd_max_sq / q_tile)`` tiles per sequence.
        :param thd_batch: THD only: the real sequence count B.
        :param thd_max_sq: THD only: the longest sequence's Q length.
        :param head_tile_qk: Q/K head dimension (the QK^T contraction width).
            Must be a multiple of 16 between 16 and 256, inclusive.
        :param head_tile_v: V/O head dimension (the P@V output width). Same
            constraint as ``head_tile_qk``.
        :param q_tile: Query sequence tile size.
        :param kv_tile: Key/value sequence tile size.
        """

        if out_dtype != in_dtype:
            raise ValueError("out_dtype must match in_dtype")
        if thd_varlen and (thd_batch < 1 or thd_max_sq < 1):
            raise ValueError("thd_varlen requires thd_batch >= 1 and thd_max_sq >= 1")
        self.in_dtype = in_dtype
        self.out_dtype = in_dtype
        self.is_causal = is_causal
        self.causal_bottom_right = causal_bottom_right
        self.window_size_left = window_size_left
        self.seq_q_lens_present = seq_q_lens_present
        self.seq_kv_lens_present = seq_kv_lens_present
        self.has_sink = has_sink
        self.thd_varlen = thd_varlen
        self.thd_batch = thd_batch
        self.thd_max_sq = thd_max_sq

        self.head_tile_qk = head_tile_qk
        self.head_tile_v = head_tile_v
        self.q_tile = q_tile
        self.kv_tile = kv_tile

        # Warp roles
        if self.q_tile == 128:
            self.compute_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7)
            self.load_warp_id = 8
            self.empty_warp_ids = (9, 10, 11)
            self.num_warps = 12
        else:
            self.compute_warp_ids = (0, 1, 2, 3)
            self.load_warp_id = 4
            self.empty_warp_ids = (5, 6, 7)
            self.num_warps = 8
        self.num_compute_warps = len(self.compute_warp_ids)
        self.num_load_warps = 1

        self.bar_compute_sync = 1
        self.bar_k_consumed = 2
        self.bar_v_consumed = 3

        self.threads_per_cta = cute.arch.WARP_SIZE * self.num_warps
        self.threads_load = cute.arch.WARP_SIZE * self.num_load_warps
        self.threads_compute = cute.arch.WARP_SIZE * self.num_compute_warps
        self.threads_kv_pipeline = self.threads_compute + self.threads_load

        self._setup_attributes()

    def _setup_attributes(self):
        """Compute derived tile, MMA, and TMA constants from the configuration."""

        # Tiling
        self.k_tile_elems = self.kv_tile * self.head_tile_qk
        self.v_tile_elems = self.kv_tile * self.head_tile_v
        self.o_tile_elems = self.q_tile * self.head_tile_v

        # MMA
        self.qk_k_frags = self.kv_tile // self.MMA_TILER[1]
        self.qk_d_frags = self.head_tile_qk // self.MMA_TILER[2]
        self.pv_v_frags = self.kv_tile // self.MMA_TILER[2]
        self.pv_d_frags = self.head_tile_v // self.MMA_TILER[1]

        # TMA
        def get_swizzle(head_tile: int):
            head_bytes = head_tile * self.in_dtype.bytes
            for swizzle, span in (
                (cuda.TensorMapSwizzle.s128b, 128),
                (cuda.TensorMapSwizzle.s64b, 64),
                (cuda.TensorMapSwizzle.s32b, 32),
            ):
                if head_bytes % span == 0:
                    return swizzle, head_bytes // span, head_tile // (head_bytes // span)
            raise ValueError(f"Unsupported TMA inner dimension: {head_bytes} B")

        self.k_tma_swizzle, self.k_tma_swizzle_chunks, self.k_swizzle_chunk_elems = get_swizzle(self.head_tile_qk)
        self.v_tma_swizzle, self.v_tma_swizzle_chunks, self.v_swizzle_chunk_elems = get_swizzle(self.head_tile_v)

    @cute.jit
    def load_one_kv_tile(
        self,
        s_dst: cutlass.Array,
        tma_desc: cutlass.GridConstant[cuda.TensorMap],
        mbar: cutlass.Array,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        seq_coord: cutlass.Int32,
    ) -> None:
        """Launch one TMA load for a complete K/V tile into swizzled SMEM.

        The tensor map exposes compact ``(B, S, H, D)`` storage through a
        logical ``(B, H, I, S, C)`` view, where ``D = I * C``. Its TMA-order
        dimensions are ``(C, S, I, H, B)``, so one rank-5 copy covers every
        head chunk and uses coordinates ``(c, seq, i, head, batch)``.

        :param s_dst: Swizzled SMEM destination tile.
        :param tma_desc: K or V tensor map descriptor.
        :param mbar: TMA completion mbarrier for this stream.
        :param batch_idx: Batch index.
        :param head_idx: Attention head index.
        :param seq_coord: Starting sequence row for the K/V tile.
        """
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(mbar, tma_desc.global_tx_bytes())
            prims.cp_async_bulk_tensor_shared_cta_global(
                s_dst,
                tma_desc.get_ptr(),
                (0, seq_coord, 0, head_idx, batch_idx),
                mbar,
            )

    @cute.jit
    def load_q_tile(
        self,
        basic_params: SimpleNamespace,
    ) -> cutlass.Array:
        """Load the warp-owned Q tile directly from GMEM into MMA A registers.

        :param basic_params: Per-CTA tensor metadata, lane mapping, and Q base
            offsets.
        :return: Packed Q fragments arranged for ``mma.sync`` A operands.
        """
        q_regs = cutlass.Array(
            cutlass.Int32,
            self.qk_d_frags * 4,
            alignment=16,
        )

        # First row and column owned by this lane in each MMA A fragment.
        row0 = basic_params.lane // 4
        col0 = (basic_params.lane % 4) * 2

        row0_in_cta = basic_params.q_warp_row0 + row0
        col0_in_cta = col0
        q_regs_offset = 0
        for _ in cutlass.range_constexpr(self.qk_d_frags):
            mma_offsets_in_cta = (
                (row0_in_cta, col0_in_cta),
                (row0_in_cta + 8, col0_in_cta),
                (row0_in_cta, col0_in_cta + 8),
                (row0_in_cta + 8, col0_in_cta + 8),
            )
            for i in cutlass.range_constexpr(4):
                row_in_cta, col_in_cta = mma_offsets_in_cta[i]
                cur_q_seq_idx = basic_params.q_seq_idx + row_in_cta
                q_packed = cutlass.Int32(0)
                if cur_q_seq_idx < basic_params.seqlen_q and col_in_cta < basic_params.head_dim_qk:
                    q_pair = (basic_params.q_ptr + basic_params.q_head_off + cur_q_seq_idx * basic_params.q_seq_stride + col_in_cta).load(count=2, alignment=4)
                    q_packed = q_pair.bitcast(cutlass.Int32)[0]
                q_regs[q_regs_offset + i] = q_packed

            col0_in_cta += self.MMA_TILER[2]
            q_regs_offset += 4

        return q_regs

    @cute.jit
    def mma_qk(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        q_regs: cutlass.Array,
    ):
        """Compute ``S = Q @ K.T``.

        Q fragments are supplied in registers by ``load_q_tile``. K fragments
        are read from the TMA-populated ``sK`` tile with ``ldmatrix``.

        :param basic_params: Per-CTA tensor metadata and lane mapping.
        :param mma_params: Shared K tile and local O accumulator state.
        :param q_regs: Register-resident packed Q fragments.
        :return: Register-resident QK score fragments.
        """
        s_regs = cutlass.Array(
            cutlass.Float32,
            self.qk_k_frags * 4,
            alignment=16,
        )
        for i in cutlass.range_constexpr(self.qk_k_frags * 4):
            s_regs[i] = cutlass.Float32(0.0)

        k_row_in_frag_pair = basic_params.lane_div16 * 8 + basic_params.lane_mod8  # which half of k-frag pair  # which row in half
        k_col_in_frag_pair = (basic_params.lane_div8 % 2) * 8  # which half of d-frag

        def load_k_frag_pair(k_frag_pair: cutlass.Constexpr[int], d_frag: cutlass.Constexpr[int]):
            k_row_in_cta = k_frag_pair * 16 + k_row_in_frag_pair
            k_col_in_cta = d_frag * 16 + k_col_in_frag_pair
            k_chunk = k_col_in_cta // self.k_swizzle_chunk_elems
            k_col_in_chunk = k_col_in_cta % self.k_swizzle_chunk_elems
            k_physical_row = k_chunk * self.kv_tile + k_row_in_cta
            k_smem_ptr = (
                mma_params.sK.data_ptr()
                + k_physical_row * self.k_swizzle_chunk_elems
                + swizzle_xor(
                    k_physical_row,
                    k_col_in_chunk,
                    self.k_swizzle_chunk_elems,
                    self.in_dtype.bytes,
                )
            )
            return prims.ldmatrix(k_smem_ptr, 4, prims.MMALayout.ROW)

        for k_frag_pair in cutlass.range_constexpr(self.qk_k_frags // 2):
            for d_frag in cutlass.range_constexpr(self.qk_d_frags):
                k_vec = load_k_frag_pair(k_frag_pair, d_frag)
                q_off = d_frag * 4
                s_off = (k_frag_pair * 2) * 4
                s_regs[s_off:4] = ptx_mma_m16n8k16_f32(
                    q_regs[q_off + 0],
                    q_regs[q_off + 1],
                    q_regs[q_off + 2],
                    q_regs[q_off + 3],
                    k_vec[0],
                    k_vec[1],
                    s_regs[s_off + 0],
                    s_regs[s_off + 1],
                    s_regs[s_off + 2],
                    s_regs[s_off + 3],
                    self.in_dtype,
                )
                s_regs[s_off + 4 : 4] = ptx_mma_m16n8k16_f32(
                    q_regs[q_off + 0],
                    q_regs[q_off + 1],
                    q_regs[q_off + 2],
                    q_regs[q_off + 3],
                    k_vec[2],
                    k_vec[3],
                    s_regs[s_off + 4],
                    s_regs[s_off + 5],
                    s_regs[s_off + 6],
                    s_regs[s_off + 7],
                    self.in_dtype,
                )

        return s_regs

    @cute.jit
    def online_softmax(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        s_regs: cutlass.Array,
        kv_seq_idx: cutlass.Int32,
        in_mask_steps: cutlass.Constexpr[bool],
        is_first_kv_tile: cutlass.Constexpr[bool],
    ):
        """Online softmax and stage packed P in registers for the PV MMA.

        :param basic_params: Per-CTA tensor metadata and lane mapping.
        :param mma_params: Local output accumulator state to rescale.
        :param softmax_params: Row max/sum state and log2 softmax scale.
        :param s_regs: Register-resident QK score fragments from ``mma_qk``.
        :param kv_seq_idx: Absolute K/V row offset for this tile.
        :param in_mask_steps: Whether this iteration needs causal or tail predicates.
        :param is_first_kv_tile: Whether this is the first processed K/V tile
            for the current Q tile.
        :return: ``s_regs`` with packed P fragments staged in consumed score slots.
        """
        lane = basic_params.lane
        o_regs = mma_params.o_regs
        row_max = softmax_params.row_max
        row_sum = softmax_params.row_sum
        softmax_scale_log2 = softmax_params.softmax_scale_log2

        # Each lane owns four S registers split across two Q rows after Q@K^T.
        for row_half in cutlass.range_constexpr(2):
            s_reg_idx_lo = row_half * 2
            s_reg_idx_hi = row_half * 2 + 1

            q_row_in_cta = basic_params.q_warp_row0 + (lane // 4) + row_half * 8

            # Resolve mask bounds for this query row. ``valid_cols`` is an
            # exclusive upper bound; ``first_valid_col`` is inclusive.
            q_position = basic_params.q_seq_idx + q_row_in_cta
            diagonal_offset = cutlass.Int32(0)
            if cutlass.const_expr(self.causal_bottom_right):
                diagonal_offset = basic_params.seqlen_k - basic_params.seqlen_q
            diagonal_position = q_position + diagonal_offset

            valid_cols = basic_params.seqlen_k
            if cutlass.const_expr(self.is_causal):
                valid_cols = cute.math.max(
                    cutlass.Int32(0),
                    cute.math.min(diagonal_position + 1, basic_params.seqlen_k),
                )

            first_valid_col = cutlass.Int32(0)
            if cutlass.const_expr(self.window_size_left is not None):
                first_valid_col = cute.math.max(
                    cutlass.Int32(0),
                    diagonal_position - self.window_size_left,
                )

            # Reduce max across this lane's S values for the current Q row.
            cur_max = cutlass.Float32(-cutlass.Float32.inf)
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                s0 = s_regs[s_off + s_reg_idx_lo]
                s1 = s_regs[s_off + s_reg_idx_hi]
                if cutlass.const_expr(in_mask_steps):
                    k_col0 = kv_seq_idx + k_frag * 8 + 2 * (lane % 4)
                    k_col1 = k_col0 + 1
                    valid0 = k_col0 >= first_valid_col and k_col0 < valid_cols
                    valid1 = k_col1 >= first_valid_col and k_col1 < valid_cols
                    if not valid0:
                        s0 = -cutlass.Float32.inf
                    if not valid1:
                        s1 = -cutlass.Float32.inf
                s_regs[s_off + s_reg_idx_lo] = s0
                s_regs[s_off + s_reg_idx_hi] = s1
                cur_max = cute.arch.fmax(cur_max, cute.arch.fmax(s0, s1))

            # The four lanes that share one Q row reduce to the tile row max.
            cur_max = nvvm_threadquad_reduction_max(cur_max)

            # Update row_max and compute the old-output correction factor.
            old_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(is_first_kv_tile):
                new_max = cur_max
            else:
                row_max_prev = row_max[row_half]
                new_max = cute.arch.fmax(row_max_prev, cur_max)
                need_correct = True
                if cutlass.const_expr(in_mask_steps):
                    need_correct = new_max > -cutlass.Float32.inf
                if need_correct:
                    old_scale = cute.math.exp2(
                        (row_max_prev - new_max) * softmax_scale_log2,
                        fastmath=True,
                    )
            row_max[row_half] = new_max

            # Compute P, accumulate the per-lane partial sum, and stage P.
            exp_max = new_max
            if cutlass.const_expr(in_mask_steps):
                if exp_max == -cutlass.Float32.inf:
                    exp_max = cutlass.Float32(0.0)
            neg_exp_max_scaled = -(exp_max * softmax_scale_log2)
            tile_sum = cutlass.Float32(0.0)
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                s0 = s_regs[s_off + s_reg_idx_lo]
                s1 = s_regs[s_off + s_reg_idx_hi]
                in0, in1 = fma2(
                    (s0, s1),
                    (softmax_scale_log2, softmax_scale_log2),
                    (neg_exp_max_scaled, neg_exp_max_scaled),
                )
                p0 = cute.math.exp2(in0, fastmath=True)
                p1 = cute.math.exp2(in1, fastmath=True)
                tile_sum = tile_sum + (p0 + p1)
                p_pack = pack_to_i32(
                    (self.in_dtype(p0), self.in_dtype(p1)),
                    self.in_dtype,
                )
                p_pack_off = k_frag * 4 + row_half * 2
                s_regs[p_pack_off] = p_pack.bitcast(cutlass.Float32)

            # Reduce tile_sum across the four lanes that own one Q row.
            tile_sum = nvvm_threadquad_reduction_sum(tile_sum)

            # Correct row_sum and rescale O when row_max changes.
            if cutlass.const_expr(is_first_kv_tile):
                row_sum[row_half] = tile_sum
            else:
                row_sum[row_half] = row_sum[row_half] * old_scale + tile_sum

                for d_frag in cutlass.range_constexpr(self.pv_d_frags):
                    o_off = d_frag * 4 + row_half * 2
                    o_regs[o_off + 0], o_regs[o_off + 1] = fmul2(
                        (o_regs[o_off + 0], o_regs[o_off + 1]),
                        (old_scale, old_scale),
                    )

        return s_regs

    @cute.jit
    def mma_pv(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        p_regs: cutlass.Array,
    ) -> None:
        """Compute ``O += P @ V``.

        P fragments are already packed in registers. V fragments are streamed
        from the TMA-populated ``sV`` tile with ``ldmatrix``.

        :param basic_params: Per-CTA tensor metadata and lane mapping.
        :param mma_params: Shared V tile and local O accumulator state.
        :param p_regs: Register-resident packed P fragments from ``softmax``.
        """
        o_regs = mma_params.o_regs

        v_row_in_frag_pair = (basic_params.lane_div8 % 2) * 8 + basic_params.lane_mod8  # which half of v-frag  # which row in half
        v_col_in_frag_pair = (basic_params.lane_div8 // 2) * 8  # which half of d-frag pair

        def load_v_frag_pair(v_frag: cutlass.Constexpr[int], d_frag_pair: cutlass.Constexpr[int]):
            v_row_in_cta = v_frag * 16 + v_row_in_frag_pair
            v_col_in_cta = d_frag_pair * 16 + v_col_in_frag_pair
            v_chunk = v_col_in_cta // self.v_swizzle_chunk_elems
            v_col_in_chunk = v_col_in_cta % self.v_swizzle_chunk_elems
            v_physical_row = v_chunk * self.kv_tile + v_row_in_cta
            sV_ptr = (
                mma_params.sV.data_ptr()
                + v_physical_row * self.v_swizzle_chunk_elems
                + swizzle_xor(
                    v_physical_row,
                    v_col_in_chunk,
                    self.v_swizzle_chunk_elems,
                    self.in_dtype.bytes,
                )
            )
            return prims.ldmatrix(sV_ptr, 4, prims.MMALayout.COL)

        for v_frag in cutlass.range_constexpr(self.pv_v_frags):
            for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                v_vec = load_v_frag_pair(v_frag, d_frag_pair)
                p_pack_off = (2 * v_frag) * 4
                p0 = p_regs[p_pack_off + 0].bitcast(cutlass.Int32)
                p1 = p_regs[p_pack_off + 2].bitcast(cutlass.Int32)
                p2 = p_regs[p_pack_off + 4].bitcast(cutlass.Int32)
                p3 = p_regs[p_pack_off + 6].bitcast(cutlass.Int32)
                o_off = (d_frag_pair * 2) * 4
                o_regs[o_off:4] = ptx_mma_m16n8k16_f32(
                    p0,
                    p1,
                    p2,
                    p3,
                    v_vec[0],
                    v_vec[1],
                    o_regs[o_off + 0],
                    o_regs[o_off + 1],
                    o_regs[o_off + 2],
                    o_regs[o_off + 3],
                    self.in_dtype,
                )
                o_regs[o_off + 4 : 4] = ptx_mma_m16n8k16_f32(
                    p0,
                    p1,
                    p2,
                    p3,
                    v_vec[2],
                    v_vec[3],
                    o_regs[o_off + 4],
                    o_regs[o_off + 5],
                    o_regs[o_off + 6],
                    o_regs[o_off + 7],
                    self.in_dtype,
                )

    @cute.jit
    def compute_one_kv_tile(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        q_regs: cutlass.Array,
        num_kv_tiles: cutlass.Int32,
        kv_tile_idx: cutlass.Int32,
        in_mask_steps: cutlass.Constexpr[bool],
        is_first_kv_tile: cutlass.Constexpr[bool],
    ) -> None:
        """One compute-side iteration of the right-to-left FMHA prefill loop.

        :param basic_params: Per-CTA tensor metadata, lane mapping, and TMA mbarriers.
        :param mma_params: Shared-memory tiles and local MMA state.
        :param softmax_params: Online softmax row state.
        :param q_regs: Register-resident packed Q fragments.
        :param num_kv_tiles: Number of K/V tiles processed by this CTA.
        :param kv_tile_idx: K/V tile index processed by this iteration.
        :param in_mask_steps: Whether this tile needs causal or K-tail masking.
        :param is_first_kv_tile: Whether this tile initializes the online softmax state.
        """

        # The K/V loop walks tile indices in reverse order. The mbarrier parity
        # still follows the load iteration count: 0, 1, 0, 1, ...
        tma_phase = (num_kv_tiles - 1 - kv_tile_idx) & cutlass.Int32(1)
        while not prims.mbarrier_try_wait_parity(basic_params.k_tma_mbar, tma_phase):
            pass

        s_regs = self.mma_qk(basic_params, mma_params, q_regs)
        prims.barrier_cta_arrive(self.bar_k_consumed, self.threads_kv_pipeline)

        while not prims.mbarrier_try_wait_parity(basic_params.v_tma_mbar, tma_phase):
            pass

        p_regs = self.online_softmax(
            basic_params,
            mma_params,
            softmax_params,
            s_regs,
            kv_tile_idx * self.kv_tile,
            in_mask_steps,
            is_first_kv_tile,
        )

        self.mma_pv(basic_params, mma_params, p_regs)
        prims.barrier_cta_arrive(self.bar_v_consumed, self.threads_kv_pipeline)

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: Optional[cute.Tensor],
        sinks: Optional[cute.Tensor],
        seq_q_lens: cute.Tensor,
        seq_kv_lens: cute.Tensor,
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda.TensorMap],
        softmax_scale_log2: cutlass.Float32,
    ) -> None:
        """SM120 FMHA prefill kernel.

        :param q: Query tensor.
        :param k: Key tensor.
        :param v: Value tensor.
        :param o: Output tensor.
        :param lse: ``(B, H, Sq)`` fp32 log-sum-exp output, or ``None`` to
            compile the LSE store out (the DSL specializes on ``None``).
        :param sinks: ``(H,)`` fp32 per-Q-head sink logits; ``None`` iff the
            kernel is configured without ``has_sink``.
        :param seq_q_lens: Per-batch query lengths, or an unused dummy tensor.
        :param seq_kv_lens: Per-batch key/value lengths, or an unused dummy tensor.
        :param tma_k_desc: Tensor map descriptor for K.
        :param tma_v_desc: Tensor map descriptor for V.
        :param softmax_scale_log2: ``softmax_scale * log2(e)``, pre-folded host-side.
        """
        tidx, _, _ = cute.arch.thread_idx()
        q_tile_idx, batch_idx, head_idx = cute.arch.block_idx()
        if cutlass.const_expr(self.is_causal):
            # Causal work grows with the Q tile. Launch long tiles first to
            # avoid leaving a few expensive CTAs in the final scheduler waves.
            grid_q, _, _ = cute.arch.grid_dim()
            q_tile_idx = grid_q - q_tile_idx - 1
        q_seq_idx = q_tile_idx * self.q_tile

        lane = tidx % cute.arch.WARP_SIZE
        warp = cute.arch.warp_idx()

        seqlen_q = cutlass.Int32(q.shape[1])
        seqlen_k = cutlass.Int32(k.shape[1])
        q_row_base = cutlass.Int32(0)
        kv_row_base = cutlass.Int32(0)
        if cutlass.const_expr(self.thd_varlen):
            # THD: seq_kv_lens is the metadata tensor [seq_kv(B) | cu_q(B+1) | cu_k(B+1)].
            # Per-sequence lengths come from the prefix sums; the bases offset
            # every packed (1, T, H, D) access below.
            n_batch = (seq_kv_lens.shape[0] - 2) // 3
            meta = cutlass.make_array_view(seq_kv_lens)
            q_row_base = cutlass.Int32(meta[n_batch + batch_idx])
            seqlen_q = cutlass.Int32(meta[n_batch + batch_idx + 1]) - q_row_base
            kv_row_base = cutlass.Int32(meta[2 * n_batch + 1 + batch_idx])
            seqlen_k = cutlass.Int32(meta[2 * n_batch + 1 + batch_idx + 1]) - kv_row_base
        else:
            if cutlass.const_expr(self.seq_q_lens_present):
                seqlen_q = cute.math.max(
                    cutlass.Int32(0),
                    cute.math.min(seq_q_lens[batch_idx], cutlass.Int32(q.shape[1])),
                )
            if cutlass.const_expr(self.seq_kv_lens_present):
                seqlen_k = cute.math.max(
                    cutlass.Int32(0),
                    cute.math.min(seq_kv_lens[batch_idx], cutlass.Int32(k.shape[1])),
                )

        num_heads_q = q.shape[2]
        num_heads_kv = k.shape[2]
        head_dim_qk = q.shape[3]
        head_dim_v = v.shape[3]
        q_ptr = q.iterator.raw_ptr()
        o_ptr = o.iterator.raw_ptr()

        q_batch_stride, q_seq_stride, q_head_stride, _ = q.stride
        o_batch_stride, o_seq_stride, o_head_stride, _ = o.stride
        if cutlass.const_expr(self.thd_varlen):
            # Packed view has batch 1: the sequence's token base replaces the
            # batch stride term, and every Q/O row index below stays
            # sequence-local.
            q_head_off = q_row_base * q_seq_stride + head_idx * q_head_stride
            o_head_off = q_row_base * o_seq_stride + head_idx * o_head_stride
        else:
            q_head_off = batch_idx * q_batch_stride + head_idx * q_head_stride
            o_head_off = batch_idx * o_batch_stride + head_idx * o_head_stride
        kv_head_idx = head_idx // (num_heads_q // num_heads_kv)

        num_kv_tiles = ceil_div(seqlen_k, self.kv_tile)
        if cutlass.const_expr(self.thd_varlen):
            # The grid covers ceil(max_seq_q / q_tile) tiles per sequence; a
            # tile past this sequence's Q length has no rows to produce.
            # Zeroing its KV work makes the whole CTA drain through the
            # barriers without loads, compute, or stores.
            if q_seq_idx >= seqlen_q:
                num_kv_tiles = cutlass.Int32(0)
        if cutlass.const_expr(self.is_causal):
            causal_k_end = q_seq_idx + self.q_tile
            if cutlass.const_expr(self.causal_bottom_right):
                causal_k_end += seqlen_k - seqlen_q
            causal_k_end = cute.math.max(cutlass.Int32(0), cute.math.min(causal_k_end, seqlen_k))
            num_kv_tiles_causal = ceil_div(causal_k_end, self.kv_tile)
            num_kv_tiles = cute.math.min(num_kv_tiles, num_kv_tiles_causal)

        min_kv_tile = cutlass.Int32(0)
        if cutlass.const_expr(self.window_size_left is not None):
            first_q_position = q_seq_idx
            if cutlass.const_expr(self.causal_bottom_right):
                first_q_position += seqlen_k - seqlen_q
            first_valid_col = cute.math.max(cutlass.Int32(0), first_q_position - self.window_size_left)
            min_kv_tile = first_valid_col // self.kv_tile
        has_kv_work = num_kv_tiles > 0 and (num_kv_tiles - 1) >= min_kv_tile

        # Shared-memory layout:
        #   sK: one kv_tile x head_tile_qk K tile
        #   sV: one kv_tile x head_tile_v V tile
        # The epilogue later aliases this storage as the q_tile x head_tile_v
        # sO staging tile after compute warps finish consuming the final K/V
        # tile.
        sKV = cutlass.Array(
            k.dtype,
            max(self.k_tile_elems + self.v_tile_elems, self.o_tile_elems),
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        sK = sKV
        sV = sKV.subview(self.k_tile_elems)
        tma_mbar = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
        k_tma_mbar = tma_mbar
        v_tma_mbar = tma_mbar.subview(1)

        # Initialize the TMA completion barriers before any load or compute warp
        # can touch the K/V pipeline.
        if warp == self.load_warp_id:
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_k_desc.get_ptr())
                prims.prefetch_tensormap(tma_v_desc.get_ptr())
                prims.mbarrier_init(k_tma_mbar, 1)
                prims.mbarrier_init(v_tma_mbar, 1)
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)

        # /////////////////////////////////////////////////////////////////////////////
        #  LOAD K/V
        # /////////////////////////////////////////////////////////////////////////////
        if warp == self.load_warp_id:
            prims.setmaxregister(40, prims.SetMaxRegisterAction.DECREASE)

            # THD collapses the packed view's batch coordinate to 0; the
            # per-sequence token base rides the seq coordinate instead. Every
            # K/V load (including the first) must apply both, or batch >= 1
            # reads the wrong packed rows.
            tma_batch_idx = batch_idx
            if cutlass.const_expr(self.thd_varlen):
                tma_batch_idx = cutlass.Int32(0)

            # The attention loop walks K/V tiles right-to-left so causal and
            # tail-masked tiles are processed before fully unmasked tiles.
            kv_seq_idx = (num_kv_tiles - 1) * self.kv_tile
            if has_kv_work:
                self.load_one_kv_tile(
                    sK,
                    tma_k_desc,
                    k_tma_mbar,
                    tma_batch_idx,
                    kv_head_idx,
                    kv_row_base + kv_seq_idx,
                )
                self.load_one_kv_tile(
                    sV,
                    tma_v_desc,
                    v_tma_mbar,
                    tma_batch_idx,
                    kv_head_idx,
                    kv_row_base + kv_seq_idx,
                )

                kv_seq_idx -= self.kv_tile
                while kv_seq_idx >= min_kv_tile * self.kv_tile:
                    prims.barrier_cta_sync(
                        self.bar_k_consumed,
                        thread_count=self.threads_kv_pipeline,
                    )
                    self.load_one_kv_tile(
                        sK,
                        tma_k_desc,
                        k_tma_mbar,
                        tma_batch_idx,
                        kv_head_idx,
                        kv_row_base + kv_seq_idx,
                    )

                    prims.barrier_cta_sync(
                        self.bar_v_consumed,
                        thread_count=self.threads_kv_pipeline,
                    )
                    self.load_one_kv_tile(
                        sV,
                        tma_v_desc,
                        v_tma_mbar,
                        tma_batch_idx,
                        kv_head_idx,
                        kv_row_base + kv_seq_idx,
                    )
                    kv_seq_idx -= self.kv_tile
        # /////////////////////////////////////////////////////////////////////////////
        #  COMPUTE
        # /////////////////////////////////////////////////////////////////////////////
        elif warp < self.load_warp_id:
            prims.setmaxregister(232, prims.SetMaxRegisterAction.INCREASE)

            compute_warp_idx = warp
            q_warp_row0 = compute_warp_idx * self.MMA_TILER[0]

            lane_div8 = lane // 8
            lane_mod8 = lane % 8
            lane_div16 = lane // 16

            # Per-lane row_max and row_sum for online softmax. Each lane owns
            # two Q rows: lane//4 and lane//4 + 8 within this compute warp.
            row_max = cutlass.Array(cutlass.Float32, 2, alignment=16)
            row_sum = cutlass.Array(cutlass.Float32, 2, alignment=16)
            for i in cutlass.range_constexpr(2):
                row_max[i] = -cutlass.Float32.inf
                row_sum[i] = 0.0

            # Per-lane fp32 accumulator for O = P @ V.
            o_regs = cutlass.Array(
                cutlass.Float32,
                self.pv_d_frags * 4,
                alignment=16,
            )
            for i in cutlass.range_constexpr(self.pv_d_frags * 4):
                o_regs[i] = 0.0

            basic_params = SimpleNamespace(
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                head_dim_qk=head_dim_qk,
                q_ptr=q_ptr,
                batch_idx=batch_idx,
                head_idx=head_idx,
                q_seq_idx=q_seq_idx,
                q_head_off=q_head_off,
                q_seq_stride=q_seq_stride,
                q_warp_row0=q_warp_row0,
                lane=lane,
                lane_div8=lane_div8,
                lane_mod8=lane_mod8,
                lane_div16=lane_div16,
                tma_k_desc=tma_k_desc,
                tma_v_desc=tma_v_desc,
                k_tma_mbar=k_tma_mbar,
                v_tma_mbar=v_tma_mbar,
            )
            mma_params = SimpleNamespace(
                sK=sK,
                sV=sV,
                o_regs=o_regs,
            )
            softmax_params = SimpleNamespace(
                row_max=row_max,
                row_sum=row_sum,
                softmax_scale_log2=softmax_scale_log2,
            )

            # Load Q into registers.
            q_regs = self.load_q_tile(basic_params)

            # Main attention loop.
            mask_steps = 1
            if cutlass.const_expr(self.is_causal):
                mask_steps = ceil_div(self.q_tile, self.kv_tile)
                if cutlass.const_expr(self.causal_bottom_right):
                    # The shifted diagonal can straddle one additional KV tile.
                    mask_steps = ceil_div(self.q_tile + self.kv_tile - 1, self.kv_tile)
            left_mask_steps = 1
            if cutlass.const_expr(self.window_size_left is not None):
                left_mask_steps = ceil_div(self.q_tile + self.kv_tile - 1, self.kv_tile)

            kv_tile_idx = num_kv_tiles - 1
            # Phase 1: potentially masked iterations.
            for step in cutlass.range_constexpr(mask_steps):
                if kv_tile_idx >= min_kv_tile:
                    self.compute_one_kv_tile(
                        basic_params,
                        mma_params,
                        softmax_params,
                        q_regs,
                        num_kv_tiles,
                        kv_tile_idx,
                        in_mask_steps=True,
                        is_first_kv_tile=(step == 0),
                    )
                kv_tile_idx -= 1

            # Phase 2: remaining fully unmasked iterations.
            while kv_tile_idx > min_kv_tile + (left_mask_steps - 1):
                self.compute_one_kv_tile(
                    basic_params,
                    mma_params,
                    softmax_params,
                    q_regs,
                    num_kv_tiles,
                    kv_tile_idx,
                    in_mask_steps=False,
                    is_first_kv_tile=False,
                )
                kv_tile_idx -= 1

            # The sliding-window left edge sweeps across the Q tile and can
            # therefore cut through more than one K/V tile.
            if cutlass.const_expr(self.window_size_left is not None):
                for _ in cutlass.range_constexpr(left_mask_steps):
                    if kv_tile_idx >= min_kv_tile:
                        self.compute_one_kv_tile(
                            basic_params,
                            mma_params,
                            softmax_params,
                            q_regs,
                            num_kv_tiles,
                            kv_tile_idx,
                            in_mask_steps=True,
                            is_first_kv_tile=False,
                        )
                    kv_tile_idx -= 1
            else:
                if kv_tile_idx >= 0:
                    self.compute_one_kv_tile(
                        basic_params,
                        mma_params,
                        softmax_params,
                        q_regs,
                        num_kv_tiles,
                        kv_tile_idx,
                        in_mask_steps=False,
                        is_first_kv_tile=False,
                    )

            # Per-row O normalization factor and natural-log LSE.
            # The thread-quad reductions left row_max / row_sum replicated across
            # the four lanes that share a Q row, so every lane finalizes the two
            # rows it owns without further exchange. row_max holds the raw (unscaled)
            # score max; the scale is applied in log2 domain and converted with ln(2).
            # With has_sink, the per-head sink logit joins the softmax denominator
            # as a virtual column with no V row: it rescales O, enters the LSE,
            # and gives a row with no visible key a finite LSE (the sink alone).
            LN2 = cutlass.Float32(0.6931471805599453)
            row_sum_inv = cutlass.Array(cutlass.Float32, 2, alignment=8)
            row_lse = cutlass.Array(cutlass.Float32, 2, alignment=8)
            for row_half in cutlass.range_constexpr(2):
                row_max_nat = row_max[row_half] * softmax_scale_log2 * LN2
                if cutlass.const_expr(self.has_sink):
                    sinks_arr = cutlass.make_array_view(sinks)
                    sink_logit = cutlass.Float32(sinks_arr[head_idx])
                    new_max = cute.arch.fmax(row_max_nat, sink_logit)
                    # alpha re-normalizes the loop's accumulator and sum from
                    # row_max_nat to the sink-extended max; it is 0 for a row
                    # with no visible key, so O := 0 falls out.
                    alpha = cute.math.exp(row_max_nat - new_max, fastmath=True)
                    new_sum = row_sum[row_half] * alpha + cute.math.exp(sink_logit - new_max, fastmath=True)
                    row_sum_inv[row_half] = alpha / new_sum
                    row_lse[row_half] = new_max + cute.math.log(new_sum, fastmath=True)
                else:
                    inv = cutlass.Float32(0.0)
                    if row_sum[row_half] > 0.0:
                        inv = cute.math.rcp(row_sum[row_half], approx=True, ftz=True)
                    row_sum_inv[row_half] = inv
                    lse_val = row_max_nat + cute.math.log(
                        cute.math.max(row_sum[row_half], cutlass.Float32(1e-30)),
                        fastmath=True,
                    )
                    # Rows with no visible key write -inf / O := 0.
                    if row_sum[row_half] <= 0.0:
                        lse_val = -cutlass.Float32.inf
                    row_lse[row_half] = lse_val

            if cutlass.const_expr(lse is not None):
                if lane % 4 == 0:
                    lse_arr = cutlass.make_array_view(lse)
                    for row_half in cutlass.range_constexpr(2):
                        lse_q_idx = q_seq_idx + q_warp_row0 + (lane // 4) + row_half * 8
                        lse_out = cutlass.Float32(row_lse[row_half])
                        if cutlass.const_expr(self.thd_varlen):
                            # Packed (1, H, T) LSE: rows past this sequence's Q
                            # length belong to the NEXT sequence — never written,
                            # and there is no padded region to trim.
                            if lse_q_idx < seqlen_q:
                                lse_row = lse_arr[0, head_idx, :]
                                lse_row[q_row_base + lse_q_idx] = lse_out
                        else:
                            # Rows at/past this batch's Q length trim to -inf.
                            if lse_q_idx >= seqlen_q:
                                lse_out = -cutlass.Float32.inf
                            if lse_q_idx < q.shape[1]:
                                lse_row = lse_arr[batch_idx, head_idx, :]
                                lse_row[lse_q_idx] = lse_out

            prims.barrier_cta_sync(self.bar_compute_sync, thread_count=self.threads_compute)

            # Epilogue: normalize O, stage it through an stmatrix-friendly SMEM
            # layout, then store one contiguous 8-element vector per lane to GMEM.
            sO = sKV
            row_sum_inv_vec = cutlass.Vector.from_elements(
                (
                    row_sum_inv[0],
                    row_sum_inv[0],
                    row_sum_inv[1],
                    row_sum_inv[1],
                    row_sum_inv[0],
                    row_sum_inv[0],
                    row_sum_inv[1],
                    row_sum_inv[1],
                ),
                cutlass.Float32,
            )
            for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                o_off = (d_frag_pair * 2) * 4
                o_scaled = fmul2(o_regs[o_off:8], row_sum_inv_vec)
                o_packed = o_scaled.to(o.dtype).bitcast(cutlass.Int32)
                sO_ptr = sO.data_ptr() + (compute_warp_idx * (self.pv_d_frags // 2) + d_frag_pair) * (16 * 16) + lane * 8
                prims.stmatrix(
                    sO_ptr,
                    o_packed,
                    prims.MMALayout.ROW,
                )

            store_row = lane_mod8 + ((lane_div8) % 2) * 8
            store_col = lane_div16 * 8
            store_q_seq_idx = q_seq_idx + q_warp_row0 + store_row
            for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                store_col_in_cta = d_frag_pair * 16 + store_col
                if cutlass.const_expr(self.thd_varlen):
                    # Packed storage: rows past this sequence's Q length are
                    # the NEXT sequence's tokens — no store, and never the
                    # dense path's zero-fill.
                    if store_q_seq_idx < seqlen_q and store_col_in_cta < head_dim_v:
                        gO_ptr = o_ptr + o_head_off + store_q_seq_idx * o_seq_stride + store_col_in_cta
                        sO_ptr = sO.data_ptr() + (compute_warp_idx * (self.pv_d_frags // 2) + d_frag_pair) * (16 * 16) + lane * 8
                        gO_ptr.store(sO_ptr.load(count=8, alignment=16), alignment=16)
                else:
                    if store_q_seq_idx < q.shape[1] and store_col_in_cta < head_dim_v:
                        gO_ptr = o_ptr + o_head_off + store_q_seq_idx * o_seq_stride + store_col_in_cta
                        if store_q_seq_idx < seqlen_q:
                            sO_ptr = sO.data_ptr() + (compute_warp_idx * (self.pv_d_frags // 2) + d_frag_pair) * (16 * 16) + lane * 8
                            gO_ptr.store(sO_ptr.load(count=8, alignment=16), alignment=16)
                        else:
                            zero_vec = cutlass.Vector.from_elements(
                                (
                                    o.dtype(0.0),
                                    o.dtype(0.0),
                                    o.dtype(0.0),
                                    o.dtype(0.0),
                                    o.dtype(0.0),
                                    o.dtype(0.0),
                                    o.dtype(0.0),
                                    o.dtype(0.0),
                                ),
                                o.dtype,
                            )
                            gO_ptr.store(zero_vec, alignment=16)

        # /////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # /////////////////////////////////////////////////////////////////////////////
        else:
            prims.setmaxregister(40, prims.SetMaxRegisterAction.DECREASE)

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
        softmax_scale_log2: cutlass.Float32,
        stream: cuda_driver.CUstream,
    ) -> None:
        """Launch the SM120 cutlass FMHA kernel.

        :param q: Query tensor with shape ``(B, Sq, H, D)``.
        :param k: Key tensor with shape ``(B, Sk, H, D)``.
        :param v: Value tensor with shape ``(B, Sk, H, D)``.
        :param o: Output tensor with shape ``(B, Sq, H, D)``.
        :param lse: ``(B, H, Sq)`` fp32 log-sum-exp output, or ``None`` to
            compile the LSE store out entirely (no dummy buffer needed).
        :param sinks: ``(H,)`` fp32 per-Q-head sink logits; must be ``None``
            exactly when the kernel is configured without ``has_sink``.
        :param seq_q_lens: Per-batch query lengths, or an unused dummy tensor.
        :param seq_kv_lens: Per-batch key/value lengths, or an unused dummy tensor.
        :param softmax_scale_log2: ``softmax_scale * log2(e)``.
        :param stream: CUDA stream used for the launch.
        """
        head_dim_qk = q.shape[3]
        head_dim_v = v.shape[3]
        if cutlass.const_expr(head_dim_qk != k.shape[3] or head_dim_qk != self.head_tile_qk):
            raise ValueError("runtime Q/K head dimensions must match the kernel head_tile_qk")
        if cutlass.const_expr(head_dim_v != o.shape[3] or head_dim_v != self.head_tile_v):
            raise ValueError("runtime V/O head dimensions must match the kernel head_tile_v")
        if cutlass.const_expr(
            q.shape[0] != k.shape[0]
            or k.shape[:3] != v.shape[:3]
            or q.shape[0] != o.shape[0]
            or q.shape[1] != o.shape[1]
            or q.shape[2] != o.shape[2]
            or q.shape[2] % k.shape[2] != 0
        ):
            raise ValueError("runtime Q/K/V/O batch, sequence, or head geometry mismatch")
        for name, tensor in (("Q", q), ("K", k), ("V", v), ("O", o)):
            if cutlass.const_expr(not self.is_layout_supported(tensor.shape, tensor.stride)):
                raise ValueError(f"{name} must use compact BSHD storage")
        if cutlass.const_expr(lse is not None):
            if cutlass.const_expr(lse.shape != (q.shape[0], q.shape[2], q.shape[1])):
                raise ValueError("LSE must have shape (B, H, Sq)")
            if cutlass.const_expr(lse.stride != (q.shape[2] * q.shape[1], q.shape[1], 1)):
                raise ValueError("LSE must be compact row-major")
        if cutlass.const_expr(self.has_sink != (sinks is not None)):
            raise ValueError("sinks must be provided exactly when the kernel is configured with has_sink")
        if cutlass.const_expr(sinks is not None and sinks.shape != (q.shape[2],)):
            raise ValueError("sinks must have shape (H,)")
        if cutlass.const_expr(self.thd_varlen):
            if cutlass.const_expr(q.shape[0] != 1):
                raise ValueError("THD Q/K/V/O must be packed batch-1 views")
            if cutlass.const_expr(seq_kv_lens.shape != (3 * self.thd_batch + 2,)):
                raise ValueError("THD seq_kv_lens must be the (3*B+2,) metadata tensor")

        # Split D into I contiguous C-element chunks while preserving the
        # compact (B, S, H, D) global-memory address calculation. TMA order
        # (C, S, I, H, B) linearizes the SMEM destination as [I][kv_tile][C].
        k_tma_layout = cute.make_layout(
            (
                k.shape[0],
                k.shape[2],
                self.k_tma_swizzle_chunks,
                k.shape[1],
                self.k_swizzle_chunk_elems,
            ),
            stride=(
                k.shape[1] * k.shape[2] * head_dim_qk,
                head_dim_qk,
                self.k_swizzle_chunk_elems,
                k.shape[2] * head_dim_qk,
                1,
            ),
        )
        k_tma_box = (
            1,
            1,
            self.k_tma_swizzle_chunks,
            self.kv_tile,
            self.k_swizzle_chunk_elems,
        )
        tma_k_desc = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(k.iterator, k_tma_layout),
            box_dims=k_tma_box,
            stride_order=(4, 3, 2, 1, 0),
            swizzle=self.k_tma_swizzle,
        )
        v_tma_layout = cute.make_layout(
            (
                v.shape[0],
                v.shape[2],
                self.v_tma_swizzle_chunks,
                v.shape[1],
                self.v_swizzle_chunk_elems,
            ),
            stride=(
                v.shape[1] * v.shape[2] * head_dim_v,
                head_dim_v,
                self.v_swizzle_chunk_elems,
                v.shape[2] * head_dim_v,
                1,
            ),
        )
        v_tma_box = (
            1,
            1,
            self.v_tma_swizzle_chunks,
            self.kv_tile,
            self.v_swizzle_chunk_elems,
        )
        tma_v_desc = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(v.iterator, v_tma_layout),
            box_dims=v_tma_box,
            stride_order=(4, 3, 2, 1, 0),
            swizzle=self.v_tma_swizzle,
        )
        self.kernel(
            q,
            k,
            v,
            o,
            lse,
            sinks,
            seq_q_lens,
            seq_kv_lens,
            tma_k_desc,
            tma_v_desc,
            softmax_scale_log2,
        ).launch(
            # THD: ceil(max_seq_q / q_tile) tiles per sequence over the real
            # batch count (the packed view's batch mode is 1); tiles past a
            # shorter sequence's length drain without work.
            grid=(
                ceil_div(self.thd_max_sq, self.q_tile) if cutlass.const_expr(self.thd_varlen) else ceil_div(q.shape[1], self.q_tile),
                self.thd_batch if cutlass.const_expr(self.thd_varlen) else q.shape[0],
                q.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    compute_capability: tuple[int, int],
    b: int = 1,
    qh: int = 1,
    kh: int = 1,
    sq: int = 128,
    skv: int = 128,
    d_qk: int = 128,
    d_v: int = 128,
    max_sq: int = 0,
    has_lse: bool = True,
) -> Callable:
    """Compile and cache one architecture-specific compact BSHD shape.

    ``d_qk`` is the Q/K head dim (QK^T contraction width) and ``d_v`` the V/O
    head dim (P@V output width); they are independent, e.g. (192, 128).

    THD specializations pack the batch: ``b`` is the real sequence count,
    ``sq``/``skv`` are the packed token totals, and ``max_sq`` (the longest
    sequence's Q length) sizes the per-sequence grid.

    ``has_lse=False`` compiles the LSE store out (the kernel specializes on a
    ``None`` LSE argument) — callers that don't want stats pass no LSE buffer
    at all instead of a dummy.
    """

    kernel = SM120FusedMultiHeadAttentionForward(
        in_dtype=STORAGE_DTYPE,
        out_dtype=STORAGE_DTYPE,
        is_causal=PARAMS.is_causal,
        causal_bottom_right=PARAMS.causal_bottom_right,
        window_size_left=PARAMS.window_size_left,
        seq_q_lens_present=PARAMS.seq_q_lens_present,
        seq_kv_lens_present=PARAMS.seq_kv_lens_present,
        has_sink=PARAMS.has_sink,
        thd_varlen=PARAMS.thd_varlen,
        thd_batch=b,
        thd_max_sq=max_sq,
        head_tile_qk=d_qk,
        head_tile_v=d_v,
        q_tile=PARAMS.q_tile,
        kv_tile=PARAMS.kv_tile,
    )
    fake_batch = 1 if PARAMS.thd_varlen else b
    fake_q = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (fake_batch, sq, qh, d_qk),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_k = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (fake_batch, skv, kh, d_qk),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_v = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (fake_batch, skv, kh, d_v),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_o = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (fake_batch, sq, qh, d_v),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_lse = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (fake_batch, qh, sq),
            stride_order=(2, 1, 0),
            assumed_align=4,
        )
        if has_lse
        else None
    )
    fake_sinks = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (qh,),
            stride_order=(0,),
            assumed_align=4,
        )
        if PARAMS.has_sink
        else None
    )
    fake_seq_q_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (b,),
        stride_order=(0,),
        assumed_align=4,
    )
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (3 * b + 2,) if PARAMS.thd_varlen else (b,),  # THD: [ seq_kv(B) | cu_q(B+1) | cu_k(B+1) ]
        stride_order=(0,),
        assumed_align=4,
    )
    return cute.compile(
        kernel,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        fake_sinks,
        fake_seq_q_lens,
        fake_seq_kv_lens,
        cutlass.Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
