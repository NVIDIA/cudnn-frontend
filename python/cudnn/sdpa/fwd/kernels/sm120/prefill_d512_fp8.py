# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 SDPA for SM120/SM121 with independent head dimensions in (256, 512].

A 64-query-row, 64-key tile uses eight compute warps. Two warps share each
16-row query slab: each forms a 256-column partial QK dot, then exchanges
fp32 scores before softmax and accumulates its own 256-column output half.
Both products use native ``mma.sync.m16n8k32`` FP8 Tensor Cores.

Dense Q loads by TMA into the K/V slab, and persistent CTAs overlap the next
Q load with output stores. THD loads Q directly and claims live units from
device metadata. All Q fragments reside in registers; K/V tiles consume
64 KiB of shared memory and the partial-score exchange consumes 32 KiB.
The output epilogue reuses the exchange storage in two rounds.

The quantization contract matches the SM120 per-tensor FP8 engine: Q/K/V are
already quantized, device descales are applied in-kernel, and P receives a
fixed 2**4 cast bias whose inverse is folded into O. Reciprocal Scale_S and
Descale_S do not enter the stateless execution ABI. Output may be FP16, BF16,
E4M3, or E5M2; LSE is fp32 and Amax_O measures the scaled pre-cast output.
Each runtime head dimension is a multiple of 16; TMA zero-fills to 512.
Q/K/V/O bind declared BSHD strides natively, compact or padded in 16-byte
multiples (THD views of kv-interleaved records included); the adapter
normalizes only non-BSHD dense layouts.
"""

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute

from cutlass.experimental import primitives as prims
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_E5M2, DTYPE_FP16
from cudnn.frost.tile_dsl.scheduler import (
    SCHED_LPT_L2,
    SCHED_NATURAL,
    lpt_tile_coords,
    lpt_l2_tile_coords,
)
from cudnn.frost.tile_dsl.mma import mma_m16n8k32_f32
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp8x2, pack_fp8x2_pairs
from cudnn.frost.tile_dsl.swizzle import swizzle_xor
from cudnn.sdpa.fwd.kernels.thd_helpers import (
    build_thd_meta_kernel,
    sanitize_v_tail,
    thd_claim_next,
    thd_decode_unit,
    THD_SETUP_THREADS,
)
from cudnn.sdpa.fwd.kernels.sm120._common import (
    SCHED_L2_BUDGET_BYTES,
    ceil_div,
    nvvm_threadquad_reduction_max,
    nvvm_threadquad_reduction_sum,
)
from cudnn.sdpa.fwd.config_sm120 import (
    D512_FLAVOR,
    FP8_D512_TILE,
    SEQ_KV_TILES as _SEQ_KV_TILES,
    SEQ_Q_TILES as _SEQ_Q_TILES,
    FP8_GENERAL_HEAD_TILE_MAX,
    TemplateParams,
    pick_flavor,
    validate_params,
)

# The FROST loader injects one immutable specialization before executing this
# module. A direct import uses dense e4m3 defaults.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(dtype_qkv=DTYPE_E4M3))

# P -> fp8 cast bias (BAKED constant — NOT cuDNN's Scale_S; that pair is
# accepted and ignored). P is quantized as P * 2**P_CAST_LOG2_SCALE via the
# exp2 bias: the online softmax adopts every new row max, so P <= 1 and the
# cast peaks at 2^4 = 16, far below the e4m3 maximum of 448, while flat-row
# entries (P ~ 1/S) sit four binades above e4m3's subnormal cliff
# (quantization stays normal out to S ~ 2^13). row_sum accumulates in the
# same 2^4-scaled units and is de-scaled by the EXACT 2^-4 before the
# finalize paths (sink mix, rcp, LSE, zero-row guards run on bit-identical
# true sums); the O leg's 2^4 is cancelled by the 2^-4 folded into
# o_scale_fused.
P_CAST_LOG2_SCALE = 4.0
# 0x80808080 as a signed Int32: the sign bit of each packed E4M3/E5M2 byte.
FP8X4_SIGN_BITS = -2139062144
validate_params(
    PARAMS,
    allowed_dtypes=(DTYPE_E4M3, DTYPE_E5M2),
    allowed_o_dtypes=(DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16),
    allow_right_band=True,
)

IN_DTYPE = cutlass.Float8E4M3FN if PARAMS.dtype_qkv == DTYPE_E4M3 else cutlass.Float8E5M2
OUT_DTYPE = {
    DTYPE_E4M3: cutlass.Float8E4M3FN,
    DTYPE_E5M2: cutlass.Float8E5M2,
    DTYPE_BF16: cutlass.BFloat16,
    DTYPE_FP16: cutlass.Float16,
}[PARAMS.dtype_o]


# THD only: pull units from a device-side counter over a machine-sized grid
# instead of launching the plan-time envelope as a padded rectangle. The
# envelope scales with the DECLARED S_q, so on ragged batches most of that
# rectangle is tiles no sequence owns.
#
# This is what made the K/V consumer barriers have to balance per unit (see the
# drain in _run_unit): a CTA here runs the tile range of several units in turn.
THD_PERSISTENT = True


fmul2 = partial(prims.mul_packed_f32x2, ftz=False, rnd=prims.FPRoundingMode.RN)
fma2 = partial(prims.fma_packed_f32x2, ftz=False, rnd=prims.FPRoundingMode.RN)


# ---------------------------------------------------------------------------
# Main kernel class
# ---------------------------------------------------------------------------


class SM120FusedMultiHeadAttentionForward:
    """Configure and launch the SM120/SM121 d512 per-tensor FP8 prefill kernel."""

    SEQ_Q_TILES = _SEQ_Q_TILES
    SEQ_KV_TILES = _SEQ_KV_TILES
    SUPPORTED_HEAD_TILES = (512,)
    MMA_TILER = (16, 8, 32)  # mma.sync.aligned.m16n8k32
    # Warps per 16-row Q slab: each owns one head-dim half of Q/K and of V/O
    # (config_sm120.smem_bytes, the SMEM model the ranking uses, assumes 2).
    HEAD_SPLIT = 2

    @staticmethod
    def is_layout_supported(
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        elem_bytes: int = 1,
    ) -> bool:
        """Return whether a BSHD tensor uses storage the kernel can address.

        The head dim must be innermost-contiguous, and the head/seq strides
        must be 16-byte multiples (``16 // elem_bytes`` elements: 16 at 1 byte,
        8 for a 2-byte O) covering the dims below them (compact or padded).
        """

        if len(shape) != 4 or len(stride) != 4:
            return False
        batch, sequence, heads, head_dim = shape
        quantum = 16 // elem_bytes
        if stride[3] != 1:
            return False
        if stride[2] % quantum != 0 or stride[2] < head_dim:
            return False
        if stride[1] % quantum != 0 or stride[1] < heads * stride[2]:
            return False
        if batch != 1 and stride[0] < sequence * stride[1]:
            return False
        return True

    def __init__(
        self,
        in_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
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
        split_kv: int = 1,
        thd_batch: int = 1,
        thd_lse_head_major: bool = False,
        thd_lse_padded: bool = False,
        head_tile_qk: int = 128,
        head_tile_v: int = 128,
        kv_tile: int = SEQ_KV_TILES[0],
        q_tile: int = SEQ_Q_TILES[0],
        pack_gqa: bool = False,
        qh_per_kh: int = 1,
        stats_log2: bool = False,
    ):
        """Initialize the FMHA prefill kernel configuration.

        :param in_dtype: Q/K/V element type (Float8E4M3FN or Float8E5M2).
        :param out_dtype: O element type (Float16, BFloat16, Float8E4M3FN, or Float8E5M2).
        :param is_causal: Apply an upper causal bound to QK.
        :param bottom_right: Shift the causal diagonal by ``Skv - Sq``.
        :param window_size_left: Inclusive left-window offset, or ``None``.
        :param window_size_right: Widen the causal diagonal to the right by
            this many columns (inclusive; 0 = plain causal). Only meaningful
            with ``is_causal`` — ``compile()`` maps the band model's
            ``window_right`` to ``is_causal=(window_right is not None)`` plus
            this offset.
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
        :param thd_lse_head_major: THD only: the packed LSE is head-major
            ``(H, head_stride)`` (FlashAttention's ``softmax_lse`` layout; tokens
            contiguous within a head, ``head_stride >= T``) instead of the default
            token-major ``(T, H)``.
        :param head_tile_qk: Q/K head TILE (the QK^T contraction width): 512,
            the one tile this kernel is built for. Runtime head dims may be any
            multiple of 16 in (256, 512] — the TMA descriptors keep the actual
            extents and zero-fill the pad columns
            (the head-dim ENVELOPE).
        :param head_tile_v: V/O head TILE (the P@V output width). Same
            constraint as ``head_tile_qk``.
        :param q_tile: Query sequence tile size.
        :param kv_tile: Key/value sequence tile size.
        :param pack_gqa: Enable PackGQA: each Q tile holds ``q_tile/qh_per_kh``
            tokens x qh_per_kh query heads sharing one KV head, token-major
            (row r ↔ token ``r // G``, head ``r % G``).
        :param qh_per_kh: The graph's GQA ratio ``h_q // h_kv``; must divide
            ``q_tile`` when ``pack_gqa`` is enabled, and is validated against
            the runtime Q/K head extents at ``__call__``.
        """

        if in_dtype not in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
            raise ValueError("in_dtype must be Float8E4M3FN or Float8E5M2")
        if out_dtype not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2):
            raise ValueError("out_dtype must be Float16, BFloat16, Float8E4M3FN, or Float8E5M2")
        if qh_per_kh < 1:
            raise ValueError(f"qh_per_kh ({qh_per_kh}) must be >= 1")
        if pack_gqa and q_tile % qh_per_kh != 0:
            raise ValueError(f"qh_per_kh ({qh_per_kh}) must divide q_tile ({q_tile}) when pack_gqa is enabled")
        if pack_gqa and thd_varlen:
            raise ValueError("PackGQA is dense-only (THD keeps the unpacked path)")
        if thd_varlen and thd_batch < 1:
            raise ValueError("thd_varlen requires thd_batch >= 1")
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.is_causal = is_causal
        self.sched_policy = sched_policy
        self.bottom_right = bottom_right
        self.window_size_left = window_size_left
        # Band model: a right bound implies the causal upper limit (compile()
        # maps window_right -> is_causal), so the masking sites key off
        # is_causal and add the (compile-time) widening. 0 = plain causal.
        self.window_right = window_size_right if window_size_right is not None else 0
        # A translated diagonal (bottom-right anchoring or a right band) can
        # straddle one more KV tile than the tile-aligned top-left one.
        self.diag_shifted = bottom_right or self.window_right > 0
        self.seq_q_lens_present = seq_q_lens_present
        self.seq_kv_lens_present = seq_kv_lens_present
        self.has_sink = has_sink
        self.stats_log2 = stats_log2
        self.thd_varlen = thd_varlen
        self.thd_batch = thd_batch
        self.thd_lse_head_major = thd_lse_head_major
        self.thd_lse_padded = thd_lse_padded  # THD Stats without ragged offsets: per-batch (B, H, s_max) rows

        self.head_tile_qk = head_tile_qk
        self.head_tile_v = head_tile_v
        self.q_tile = q_tile
        self.kv_tile = kv_tile
        self.pack_gqa = pack_gqa
        self.qh_per_kh = qh_per_kh
        # Dense Q tiles load by TMA: a row block of one head, or under PackGQA a
        # G-heads x T-tokens box whose token-major landing order is the packed
        # row order. THD rows ride the packed token base and keep the per-lane
        # global loads.
        self.tma_q = not thd_varlen
        # Rows of a Q tile per token (PackGQA packs qh_per_kh heads per token).
        self.q_tile_tokens = q_tile // qh_per_kh if pack_gqa else q_tile
        # KV split: SPLIT_KV CTAs per (q_tile, batch, head), each covering a
        # contiguous slice of that tile's KV-tile range.  1 = off (folds away).
        self.split_kv = split_kv

        if (self.q_tile, self.kv_tile) != FP8_D512_TILE:
            raise ValueError(f"SM120 SDPA d512 kernel: (q_tile, kv_tile) must be {FP8_D512_TILE}; got {(self.q_tile, self.kv_tile)}")
        if (self.head_tile_qk, self.head_tile_v) != D512_FLAVOR:
            raise ValueError(f"SM120 SDPA d512 kernel: head tiles must be {D512_FLAVOR}; got {(self.head_tile_qk, self.head_tile_v)}")

        # Warp roles: every warp computes; HEAD_SPLIT warps share each 16-row Q
        # slab, one head-dim half each.
        self.compute_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7)
        self.num_warps = 8
        self.num_compute_warps = len(self.compute_warp_ids)

        self.bar_compute_sync = 1
        self.bar_k_consumed = 2
        self.bar_v_consumed = 3
        # One named barrier per Q slab (ids 4..7) for the partial-S exchange.
        self.bar_pair_base = 4

        self.threads_per_cta = cute.arch.WARP_SIZE * self.num_warps
        self.threads_compute = cute.arch.WARP_SIZE * self.num_compute_warps
        self.threads_kv_pipeline = self.threads_compute

        self._setup_attributes()

    def _setup_attributes(self):
        """Compute derived tile, MMA, and TMA constants from the configuration."""

        # Tiling
        self.k_tile_elems = self.kv_tile * self.head_tile_qk
        self.v_tile_elems = self.kv_tile * self.head_tile_v

        # MMA, per warp: its head-dim half of Q/K (QK^T) and of V/O (P@V).
        self.qk_k_frags = self.kv_tile // self.MMA_TILER[1]
        self.qk_d_frags = self.head_tile_qk // self.HEAD_SPLIT // self.MMA_TILER[2]
        self.pv_v_frags = self.kv_tile // self.MMA_TILER[2]
        self.pv_d_frags = self.head_tile_v // self.HEAD_SPLIT // self.MMA_TILER[1]
        # Element offset of a warp's K/V column half inside the swizzled sK/sV
        # tile: whole swizzle chunks, and the XOR pattern depends on row % 8 only.
        self.k_half_elems = self.k_tile_elems // self.HEAD_SPLIT
        self.v_half_elems = self.v_tile_elems // self.HEAD_SPLIT
        # SMEM: the K + V tile, the partial-S exchange ([warp][k_frag][lane][4]
        # fp32) -- which the epilogue reuses as O staging -- and the TMA
        # mbarriers. Every Q fragment lives in registers.
        # config_sm120.smem_bytes sizes the same storage for the ranking and
        # the adapter; the kernel does not keep a second copy of that model.
        self.s_smem_words = self.num_compute_warps * self.qk_k_frags * 32 * 4

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
        is_v: cutlass.Constexpr[bool],
        envelope: cutlass.Constexpr[bool],
    ) -> None:
        """Launch the TMA load(s) for a complete K/V tile into swizzled SMEM.

        Exact head dims (``envelope=False``, the common case) issue ONE rank-5
        copy whose descriptor pre-splits the head dim into swizzle-span chunks
        — the head boundary coincides with the tile so no zero-fill is needed.

        Envelope head dims (``envelope=True``, actual d < compile-time tile)
        issue ``chunks`` copies over a rank-4 descriptor that keeps the ACTUAL
        head extent as the innermost dimension, stepping the head coordinate
        by ``chunk_elems``. Head columns at or past the actual extent are
        outside that dimension, so the hardware zero-fills them (zero K columns
        add exact zero terms to every Q@K^T; zero V columns produce O columns
        the store guard clips). A single copy cannot serve this case: TMA
        bounds-checks each coordinate against its OWN dimension, so a
        chunk-dimension descriptor would fetch the next head's data instead of
        zeros past d.

        :param s_dst: Swizzled SMEM destination tile.
        :param tma_desc: K or V tensor map descriptor (rank matches
            ``envelope``).
        :param mbar: TMA completion mbarrier for this stream.
        :param batch_idx: Batch index.
        :param head_idx: Attention head index.
        :param seq_coord: Starting sequence row for the K/V tile.
        :param is_v: Selects the V-side chunk geometry over the K-side one
            (the two carry independent head tiles and swizzle spans).
        :param envelope: Actual head dim < compile-time tile (zero-padded).
        """
        chunks = self.v_tma_swizzle_chunks if is_v else self.k_tma_swizzle_chunks
        chunk_elems = self.v_swizzle_chunk_elems if is_v else self.k_swizzle_chunk_elems
        if prims.elect_sync():
            if cutlass.const_expr(envelope):
                # Every copy completes with its full box (OOB regions arrive as
                # zeros but still count), so the expected transaction total is
                # simply chunks x the per-copy box bytes.
                prims.mbarrier_arrive_expect_tx(mbar, chunks * tma_desc.global_tx_bytes())
                for i in cutlass.range_constexpr(chunks):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        s_dst.subview(i * self.kv_tile * chunk_elems),
                        tma_desc.get_ptr(),
                        (i * chunk_elems, seq_coord, head_idx, batch_idx),
                        mbar,
                    )
            else:
                # Rank-5 coordinates (c, seq, i, head, batch): one copy covers
                # every head chunk of the tile.
                prims.mbarrier_arrive_expect_tx(mbar, tma_desc.global_tx_bytes())
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_dst,
                    tma_desc.get_ptr(),
                    (0, seq_coord, 0, head_idx, batch_idx),
                    mbar,
                )

    @cute.jit
    def load_q_tile_tma(
        self,
        s_dst: cutlass.Array,
        tma_desc: cutlass.GridConstant[cuda.TensorMap],
        mbar: cutlass.Array,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        seq_coord: cutlass.Int32,
        envelope: cutlass.Constexpr[bool],
    ) -> None:
        """Launch the TMA load(s) for the whole q_tile x head_tile_qk Q tile into
        the swizzled K/V slab: same chunk geometry and envelope handling as
        ``load_one_kv_tile``, with ``q_tile`` rows per swizzle chunk."""
        chunks = self.k_tma_swizzle_chunks
        chunk_elems = self.k_swizzle_chunk_elems
        if prims.elect_sync():
            if cutlass.const_expr(envelope):
                prims.mbarrier_arrive_expect_tx(mbar, chunks * tma_desc.global_tx_bytes())
                for i in cutlass.range_constexpr(chunks):
                    # Packed descriptors place head before sequence.
                    coords = (i * chunk_elems, head_idx, seq_coord, batch_idx) if self.pack_gqa else (i * chunk_elems, seq_coord, head_idx, batch_idx)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        s_dst.subview(i * self.q_tile * chunk_elems),
                        tma_desc.get_ptr(),
                        coords,
                        mbar,
                    )
            else:
                prims.mbarrier_arrive_expect_tx(mbar, tma_desc.global_tx_bytes())
                coords = (0, head_idx, seq_coord, 0, batch_idx) if self.pack_gqa else (0, seq_coord, 0, head_idx, batch_idx)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_dst,
                    tma_desc.get_ptr(),
                    coords,
                    mbar,
                )

    @cute.jit
    def load_q_frags_from_smem(self, basic_params: SimpleNamespace, sKV: cutlass.Array) -> cutlass.Array:
        """Read this warp's 256-column FP8 Q half as packed k32 A operands."""
        q_regs = cutlass.Array(cutlass.Int32, self.qk_d_frags * 4, alignment=16)
        row_in_cta = basic_params.q_warp_row0 + basic_params.lane_mod8 + (basic_params.lane_div8 % 2) * 8
        col_in_frag = basic_params.lane_div16 * 16
        chunk_elems = self.k_swizzle_chunk_elems
        half_base = sKV.data_ptr() + basic_params.half * (self.q_tile * self.head_tile_qk // self.HEAD_SPLIT)
        for d_frag in cutlass.range_constexpr(self.qk_d_frags):
            col_in_half = d_frag * self.MMA_TILER[2] + col_in_frag
            chunk = col_in_half // chunk_elems
            col_in_chunk = col_in_half % chunk_elems
            physical_row = chunk * self.q_tile + row_in_cta
            q_ptr = half_base + physical_row * chunk_elems + swizzle_xor(physical_row, col_in_chunk, chunk_elems, self.in_dtype.bytes)
            q_regs[d_frag * 4 : 4] = prims.ldmatrix(q_ptr, 4, prims.MMALayout.ROW)
        return q_regs

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
        # m16n8k32 e4m3 A layout: a0 = A[r0, 4q..4q+3], a1 = A[r0+8, same],
        # a2/a3 = +16 in k; each reg is 4 bytes packed little-endian, loaded
        # as one aligned 4-byte GMEM access.
        row0 = basic_params.lane // 4
        col0 = (basic_params.lane % 4) * 4

        row0_in_cta = basic_params.q_warp_row0 + row0
        col0_in_cta = basic_params.q_col_half0 + col0
        q_regs_offset = 0
        for _ in cutlass.range_constexpr(self.qk_d_frags):
            mma_offsets_in_cta = (
                (row0_in_cta, col0_in_cta),
                (row0_in_cta + 8, col0_in_cta),
                (row0_in_cta, col0_in_cta + 16),
                (row0_in_cta + 8, col0_in_cta + 16),
            )
            for i in cutlass.range_constexpr(4):
                row_in_cta, col_in_cta = mma_offsets_in_cta[i]
                cur_q_seq_idx = basic_params.q_seq_idx + (row_in_cta if cutlass.const_expr(not self.pack_gqa) else row_in_cta // self.qh_per_kh)
                q_row_off = cur_q_seq_idx * basic_params.q_seq_stride
                if cutlass.const_expr(self.pack_gqa and self.qh_per_kh != 1):
                    q_row_off = q_row_off + (row_in_cta % self.qh_per_kh) * basic_params.q_head_stride
                q_packed = cutlass.Int32(0)
                if cur_q_seq_idx < basic_params.seqlen_q and col_in_cta < basic_params.head_dim_qk:
                    q_quad = (basic_params.q_ptr + basic_params.q_head_off + q_row_off + col_in_cta).load(count=4, alignment=4)
                    q_packed = q_quad.bitcast(cutlass.Int32)[0]
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

        # 8-bit K path: byte-preserving ldmatrix.m8n8.x4.b16 — each lane points
        # at one 16-byte K-segment; tile pairs cover (n8 kv rows) x (k16-half)
        # of one k32 d_frag, so k_vec[0],[1] form the m16n8k32 B fragment of
        # the first n8 block and k_vec[2],[3] the second.
        k_row_in_frag_pair = basic_params.lane_div16 * 8 + basic_params.lane_mod8  # which half of k-frag pair  # which row in half
        k_col_in_frag_pair = (basic_params.lane_div8 % 2) * 16  # which k16-half of the k32 d-frag

        def load_k_frag_pair(k_frag_pair: cutlass.Constexpr[int], d_frag: cutlass.Constexpr[int]):
            k_row_in_cta = k_frag_pair * 16 + k_row_in_frag_pair
            k_col_in_cta = d_frag * self.MMA_TILER[2] + k_col_in_frag_pair
            k_chunk = k_col_in_cta // self.k_swizzle_chunk_elems
            k_col_in_chunk = k_col_in_cta % self.k_swizzle_chunk_elems
            k_physical_row = k_chunk * self.kv_tile + k_row_in_cta
            k_smem_ptr = (
                mma_params.sK.data_ptr()
                + basic_params.k_half_off
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
                s_regs[s_off:4] = mma_m16n8k32_f32(
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
                s_regs[s_off + 4 : 4] = mma_m16n8k32_f32(
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
        :return: packed e4m3 P fragments, indexed ``[k_frag * 2 + row_half]``.
        """
        lane = basic_params.lane
        o_regs = mma_params.o_regs
        row_max = softmax_params.row_max
        row_sum = softmax_params.row_sum
        softmax_scale_log2 = softmax_params.softmax_scale_log2
        p_regs = cutlass.Array(cutlass.Uint16, self.qk_k_frags * 2)

        # Each lane owns four S registers split across two Q rows after Q@K^T.
        for row_half in cutlass.range_constexpr(2):
            s_reg_idx_lo = row_half * 2
            s_reg_idx_hi = row_half * 2 + 1

            q_row_in_cta = basic_params.q_warp_row0 + (lane // 4) + row_half * 8

            # Resolve mask bounds for this query row. ``valid_cols`` is an
            # exclusive upper bound; ``first_valid_col`` is inclusive.
            q_position = basic_params.q_seq_idx + (q_row_in_cta if cutlass.const_expr(not self.pack_gqa) else q_row_in_cta // self.qh_per_kh)
            diagonal_offset = cutlass.Int32(0)
            if cutlass.const_expr(self.bottom_right):
                diagonal_offset = basic_params.seqlen_k - basic_params.seqlen_q
            diagonal_position = q_position + diagonal_offset

            valid_cols = basic_params.seqlen_k
            if cutlass.const_expr(self.is_causal):
                valid_cols = cute.math.max(
                    cutlass.Int32(0),
                    cute.math.min(diagonal_position + 1 + self.window_right, basic_params.seqlen_k),
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
                # Keep this as inline PTX so old_scale lowers to one predicated
                # EX2 with 1.0 as the default value. The equivalent Python DSL
                # branch currently materializes extra MOV instructions in this
                # hot softmax loop and increases issue pressure on SM120.
                old_scale = cute.arch.inline_ptx(
                    (
                        "{\n"
                        "  .reg .pred p;\n"
                        "  .reg .f32 delta;\n"
                        "  sub.rn.f32 delta, $1, $2;\n"  # delta = row_max_prev - new_max
                        "  mul.rn.f32 delta, delta, $3;\n"  # delta = delta * softmax_scale_log2
                        "  setp.gt.f32 p, $2, $1;\n"  # p = new_max > row_max_prev
                        "  mov.f32 $0, 0f3f800000;\n"  # res = 1.0
                        "  @p ex2.approx.ftz.f32 $0, delta;\n"  # if p: res = exp2(delta)
                        "}"
                    ),
                    write_only_types=[cutlass.Float32],
                    read_only_args=[row_max_prev, new_max, softmax_scale_log2],
                )
            row_max[row_half] = new_max

            if cutlass.const_expr(not is_first_kv_tile):
                for d_frag in cutlass.range_constexpr(self.pv_d_frags):
                    o_off = d_frag * 4 + row_half * 2
                    if new_max > row_max_prev:
                        o_regs[o_off + 0], o_regs[o_off + 1] = fmul2(
                            (o_regs[o_off + 0], o_regs[o_off + 1]),
                            (old_scale, old_scale),
                        )

            # Compute P, accumulate the per-lane partial sum, and stage P.
            exp_max = new_max
            if cutlass.const_expr(in_mask_steps):
                if exp_max == -cutlass.Float32.inf:
                    exp_max = cutlass.Float32(0.0)
            # P-cast bias: exp2(x + P_CAST_LOG2_SCALE) = 2^4 * P (EX2 is
            # binade-shift-exact); see P_CAST_LOG2_SCALE.
            neg_exp_max_scaled = cutlass.Float32(P_CAST_LOG2_SCALE) - exp_max * softmax_scale_log2
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
                # P stays in registers at the C-fragment coordinates; mma_pv
                # redistributes it to the k32 A layout with shfl. The cast
                # carries the baked 2^P_CAST_LOG2_SCALE bias only; graph
                # Scale_S/Descale_S never reach the kernel.
                p_regs[k_frag * 2 + row_half] = fp32_to_fp8x2(p0, p1, dtype=self.in_dtype)

            # Reduce tile_sum across the four lanes that own one Q row.
            tile_sum = nvvm_threadquad_reduction_sum(tile_sum)

            # Correct row_sum (old_scale is exactly 1.0 when the max held).
            if cutlass.const_expr(is_first_kv_tile):
                row_sum[row_half] = tile_sum
            else:
                row_sum[row_half] = row_sum[row_half] * old_scale + tile_sum

        return p_regs

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
        lane = basic_params.lane

        # The QK C-fragment gives each lane columns 2*(t%4)+{0,1} while the k32
        # A-fragment wants 4 consecutive bytes; the two differ only by an
        # exchange inside each thread quad, so two shfl and one prmt replace
        # the SMEM round trip (and the 16 KB it needed).
        lane_mod4 = lane % 4
        src0 = (lane // 4) * 4 + (lane_mod4 % 2) * 2
        selector = cutlass.Int32(0x5410) if lane_mod4 < 2 else cutlass.Int32(0x7632)

        def pack_p_cols(k_frag0: cutlass.Constexpr[int], row_half: cutlass.Constexpr[int]) -> cutlass.Int32:
            pairs = pack_fp8x2_pairs(p_regs[k_frag0 * 2 + row_half], p_regs[(k_frag0 + 1) * 2 + row_half])
            lo = prims.shfl_sync(thread_mask=0xFFFFFFFF, val=pairs, offset=src0, mask_and_clamp=0x1F, kind=prims.Shfl.IDX)
            hi = prims.shfl_sync(thread_mask=0xFFFFFFFF, val=pairs, offset=src0 + 1, mask_and_clamp=0x1F, kind=prims.Shfl.IDX)
            return cute.arch.inline_ptx(
                "prmt.b32 $0, $1, $2, $3;",
                write_only_types=[cutlass.Int32],
                read_only_args=[lo, hi, selector],
            )

        # V B-fragments use the hardware 8-bit transposed load
        # ``ldmatrix.m16n16.x2.trans.b8`` (SASS LDSM.8.MT1616): every lane
        # supplies the start of smem kv-row ``v_frag*32 + lane`` at one 16-byte
        # d-chunk; one issue covers 32(kv) x 16(d) and feeds TWO MMAs with
        # register map (0, 2, 1, 3).
        def load_v_frags(v_frag: cutlass.Constexpr[int], d_frag_pair: cutlass.Constexpr[int]):
            v_row_in_cta = v_frag * self.MMA_TILER[2] + lane
            v_col_in_cta = d_frag_pair * 16
            v_chunk = v_col_in_cta // self.v_swizzle_chunk_elems
            v_col_in_chunk = v_col_in_cta % self.v_swizzle_chunk_elems
            v_physical_row = v_chunk * self.kv_tile + v_row_in_cta
            sV_ptr = (
                mma_params.sV.data_ptr()
                + basic_params.v_half_off
                + v_physical_row * self.v_swizzle_chunk_elems
                + swizzle_xor(
                    v_physical_row,
                    v_col_in_chunk,
                    self.v_swizzle_chunk_elems,
                    self.in_dtype.bytes,
                )
            )
            return prims.ldmatrix(
                sV_ptr,
                4,
                prims.MMALayout.COL,
                shape=prims.LoadShape.M16N16,
                src_format=prims.LoadSrcFormat.B8,
            )

        # V fragments load in-loop, immediately before the MMAs consuming them.
        # Issuing them one step ahead was measured at within +/-0.5% (noise
        # floor ~1%): keeping P in registers leaves little ldmatrix latency to
        # hide, so the prefetch buys nothing.
        for v_frag in cutlass.range_constexpr(self.pv_v_frags):
            # One k32 PV step consumes four QK k-fragments, paired (0,1) and (2,3).
            p_vec = (
                pack_p_cols(v_frag * 4 + 0, 0),
                pack_p_cols(v_frag * 4 + 0, 1),
                pack_p_cols(v_frag * 4 + 2, 0),
                pack_p_cols(v_frag * 4 + 2, 1),
            )
            for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                v_vec = load_v_frags(v_frag, d_frag_pair)
                o_off = (d_frag_pair * 2) * 4
                o_regs[o_off:4] = mma_m16n8k32_f32(
                    p_vec[0],
                    p_vec[1],
                    p_vec[2],
                    p_vec[3],
                    v_vec[0],
                    v_vec[2],
                    o_regs[o_off + 0],
                    o_regs[o_off + 1],
                    o_regs[o_off + 2],
                    o_regs[o_off + 3],
                    self.in_dtype,
                )
                o_regs[o_off + 4 : 4] = mma_m16n8k32_f32(
                    p_vec[0],
                    p_vec[1],
                    p_vec[2],
                    p_vec[3],
                    v_vec[1],
                    v_vec[3],
                    o_regs[o_off + 4],
                    o_regs[o_off + 5],
                    o_regs[o_off + 6],
                    o_regs[o_off + 7],
                    self.in_dtype,
                )

    @cute.jit
    def exchange_partial_s(
        self,
        basic_params: SimpleNamespace,
        s_regs: cutlass.Array,
    ) -> None:
        """Sum the two head-dim halves of ``S`` across the warps of a Q slab.

        Each warp publishes its partial scores in its own ``sX`` slots
        (``[warp][k_frag][lane][4]``, one 16-byte store per fragment), meets its
        partner on the slab's named barrier, and adds the partner's slots in
        place. The buffer is single-buffered: the caller runs this after the V
        tile's mbarrier wait, and the next V tile is issued only after every
        warp has consumed the current one, so the partner has finished reading
        this tile's partials before anyone writes the next tile's.

        :param basic_params: Lane mapping, warp-pair indices, and ``sX``.
        :param s_regs: This warp's partial QK scores; the full scores on return.
        """
        # sX is Int32 storage; the fp32 partials travel as bit patterns
        # (16-byte stores / loads with a vector bitcast).
        own_base = basic_params.sX.data_ptr() + (basic_params.compute_warp_idx * self.qk_k_frags * 32 + basic_params.lane) * 4
        partner_base = basic_params.sX.data_ptr() + (basic_params.partner_warp_idx * self.qk_k_frags * 32 + basic_params.lane) * 4
        for k_frag in cutlass.range_constexpr(self.qk_k_frags):
            s_off = k_frag * 4
            (own_base + k_frag * 32 * 4).store(
                cutlass.Vector.from_elements((s_regs[s_off + 0], s_regs[s_off + 1], s_regs[s_off + 2], s_regs[s_off + 3]), cutlass.Float32).bitcast(
                    cutlass.Int32
                ),
                alignment=16,
            )
        prims.barrier_cta_sync(self.bar_pair_base + basic_params.slab, thread_count=self.HEAD_SPLIT * cute.arch.WARP_SIZE)
        for k_frag in cutlass.range_constexpr(self.qk_k_frags):
            s_off = k_frag * 4
            partner = (partner_base + k_frag * 32 * 4).load(count=4, alignment=16).bitcast(cutlass.Float32)
            for i in cutlass.range_constexpr(4):
                s_regs[s_off + i] = s_regs[s_off + i] + partner[i]

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
        # still follows the load iteration count: 0, 1, 0, 1, ... Under a
        # persistent grid the mbarriers are armed once for the CTA and reused by
        # every unit it claims, so the count continues across units — phase_base
        # is how many tiles this CTA already loaded (0 on the one-unit paths).
        tma_phase = (basic_params.phase_base + num_kv_tiles - 1 - kv_tile_idx) & cutlass.Int32(1)
        while not prims.mbarrier_try_wait_parity(basic_params.k_tma_mbar, tma_phase):
            pass

        s_regs = self.mma_qk(basic_params, mma_params, q_regs)
        # Warp 0 waits until every warp has read sK, then issues the next K tile's
        # TMA load while the other warps run softmax.
        if basic_params.compute_warp_idx == 0:
            prims.barrier_cta_sync(self.bar_k_consumed, thread_count=self.threads_kv_pipeline)
            if kv_tile_idx - 1 >= basic_params.min_kv_tile:
                self.load_one_kv_tile(
                    mma_params.sK,
                    basic_params.tma_k_desc,
                    basic_params.k_tma_mbar,
                    basic_params.tma_batch_idx,
                    basic_params.kv_head_idx,
                    basic_params.kv_row_base + (kv_tile_idx - 1) * self.kv_tile,
                    is_v=False,
                    envelope=basic_params.k_envelope,
                )
        else:
            prims.barrier_cta_arrive(self.bar_k_consumed, self.threads_kv_pipeline)

        while not prims.mbarrier_try_wait_parity(basic_params.v_tma_mbar, tma_phase):
            pass

        # Full scores = this warp's head-dim half + the partner's. After the V
        # wait on purpose: see exchange_partial_s for why that keeps sX race-free.
        self.exchange_partial_s(basic_params, s_regs)

        p_regs = self.online_softmax(
            basic_params,
            mma_params,
            softmax_params,
            s_regs,
            kv_tile_idx * self.kv_tile,
            in_mask_steps,
            is_first_kv_tile,
        )

        if cutlass.const_expr((self.thd_varlen or self.seq_kv_lens_present) and in_mask_steps and is_first_kv_tile):
            sanitize_v_tail(
                mma_params.sV,
                basic_params.lane,
                basic_params.seqlen_k,
                kv_tile_idx * self.kv_tile,
                self.in_dtype,
                self.head_tile_v,
                self.kv_tile,
                self.v_swizzle_chunk_elems,
            )
        self.mma_pv(basic_params, mma_params, p_regs)
        if basic_params.compute_warp_idx == 0:
            prims.barrier_cta_sync(self.bar_v_consumed, thread_count=self.threads_kv_pipeline)
            if kv_tile_idx - 1 >= basic_params.min_kv_tile:
                self.load_one_kv_tile(
                    mma_params.sV,
                    basic_params.tma_v_desc,
                    basic_params.v_tma_mbar,
                    basic_params.tma_batch_idx,
                    basic_params.kv_head_idx,
                    basic_params.kv_row_base + (kv_tile_idx - 1) * self.kv_tile,
                    is_v=True,
                    envelope=basic_params.v_envelope,
                )
        else:
            prims.barrier_cta_arrive(self.bar_v_consumed, self.threads_kv_pipeline)

    @cute.jit
    def _dense_unit_coords(
        self,
        uid: cutlass.Int32,
        n_q_tiles: cutlass.Int32,
        n_batch: cutlass.Int32,
        n_qh: cutlass.Int32,
        k: cute.Tensor,
    ):
        """Decode a linear dense unit id into ``(q_tile_idx, batch_idx, head_idx,
        split_idx, o_batch_idx)``.

        LPT / LPT_L2 order the whole unit set globally (heaviest causal rows
        first). NATURAL walks Q tiles fastest, then the composite batch axis
        (``batch + split * B`` under KV split), then heads, with the causal Q-tile
        reversal so the longest diagonal rows launch first.
        """
        split_idx = cutlass.Int32(0)
        if cutlass.const_expr(self.sched_policy != SCHED_NATURAL):
            if cutlass.const_expr(self.sched_policy == SCHED_LPT_L2):
                q_tile_idx, head_idx, batch_idx = lpt_l2_tile_coords(
                    uid,
                    n_qh,
                    n_batch,
                    n_q_tiles,
                    n_qh // cutlass.Int32(k.shape[2]),
                    cutlass.Int32(k.shape[1]),
                    (self.head_tile_qk + self.head_tile_v) * self.in_dtype.width // 8,
                    SCHED_L2_BUDGET_BYTES,
                )
            else:
                q_tile_idx, head_idx, batch_idx = lpt_tile_coords(uid, n_qh, n_batch, n_q_tiles)
            o_batch_idx = batch_idx
        else:
            q_tile_idx = uid % n_q_tiles
            rest = uid // n_q_tiles
            n_by = n_batch * cutlass.Int32(self.split_kv)
            by = rest % n_by
            head_idx = rest // n_by
            batch_idx = by
            o_batch_idx = by
            if cutlass.const_expr(self.split_kv > 1):
                split_idx = by // n_batch
                batch_idx = by % n_batch
            if cutlass.const_expr(self.is_causal):
                # Diagonal-bounded work grows with the Q tile: long tiles first.
                q_tile_idx = n_q_tiles - 1 - q_tile_idx
        return q_tile_idx, batch_idx, head_idx, split_idx, o_batch_idx

    @cute.jit
    def _next_dense_unit(self, unit_idx: cutlass.Int32) -> cutlass.Int32:
        """The unit this CTA takes after ``unit_idx``: one round of G = grid-size
        units later.

        Without a sliding window unit costs vary and the scheduler order is
        heaviest-first, so the walk is boustrophedon: round k hands CTA b unit
        k*G + b on even rounds and k*G + (G-1-b) on odd ones, and each CTA
        alternates heavy and light units (a plain stride would hand CTA 0 the
        heaviest unit of every round). Windowed units cost the same, so there
        the plain stride k*G + b is used.
        """
        gdim, _, _ = cute.arch.grid_dim()
        if cutlass.const_expr(self.window_size_left is None):
            rnd = unit_idx // gdim
            pos = unit_idx - rnd * gdim
            next_unit = (rnd + cutlass.Int32(1)) * gdim + (gdim - cutlass.Int32(1) - pos)
        else:
            next_unit = unit_idx + gdim
        return next_unit

    @cute.jit
    def _dense_unit_count(self, q: cute.Tensor, n_q_tiles: cutlass.Int32):
        """``(n_batch, n_heads, n_units)`` of the dense unit space, as the launch
        computed it: the head axis is the packed group count under PackGQA and
        the full Q head count otherwise (``qh_per_kh`` is the GQA ratio either way)."""
        n_batch = cutlass.Int32(q.shape[0])
        n_qh = cutlass.Int32(q.shape[2] // self.qh_per_kh if self.pack_gqa else q.shape[2])
        return n_batch, n_qh, n_q_tiles * n_batch * cutlass.Int32(self.split_kv) * n_qh

    @cute.jit
    def _run_unit(
        self,
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
        tma_q_desc,
        softmax_scale_log2,
        o_scale_fused,
        amax_o,
        sKV,
        sK,
        sV,
        sX,
        k_tma_mbar,
        v_tma_mbar,
        q_tma_mbar,
        lane,
        warp,
        phase_base,
        split_idx,
        o_batch_idx,
        q_tile_idx,
        batch_idx,
        head_idx,
        unit_idx,
        n_q_tiles,
    ) -> cutlass.Int32:
        """Compute one (Q tile, sequence, head) unit, optionally within a KV split.

        Dense and THD persistent CTAs call this once per assigned unit. SMEM
        belongs to the CTA and is reused across units; dense units prefetch
        the next Q tile while storing the current output.
        """
        # Raw-score maxima bound P only for a nonnegative multiplier. Fold
        # its sign into Q once per unit, including the device Q/K descales.
        negate_q = softmax_scale_log2 < 0.0
        softmax_scale_log2 = cute.math.abs(softmax_scale_log2)
        q_seq_idx = q_tile_idx * (self.q_tile // self.qh_per_kh if self.pack_gqa else self.q_tile)

        seqlen_q = cutlass.Int32(q.shape[1])
        seqlen_k = cutlass.Int32(k.shape[1])
        q_row_base = cutlass.Int32(0)
        kv_row_base = cutlass.Int32(0)
        if cutlass.const_expr(self.thd_varlen):
            # THD: seq_kv_lens is the metadata tensor
            # [seq_kv(B) | cu_q(B+1) | cu_k(B+1) | remap(B) | live | ctr].
            # Per-sequence lengths come from the prefix sums; the bases offset
            # every packed (1, T, H, D) access below.
            n_batch = (seq_kv_lens.shape[0] - 4) // 4
            meta = cutlass.make_array_view(seq_kv_lens)
            # batch_idx is a real batch index here: whichever dispatch produced
            # it has already resolved the longest-first batch_remap.
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
        # ENVELOPE flags (static shapes): actual dim < compile-time tile means
        # the TMA loads must zero-fill the pad columns via per-chunk copies.
        k_envelope = head_dim_qk != self.head_tile_qk
        v_envelope = head_dim_v != self.head_tile_v
        q_ptr = q.iterator.raw_ptr()
        o_ptr = o.iterator.raw_ptr()

        q_batch_stride, q_seq_stride, q_head_stride, _ = q.stride
        o_batch_stride, o_seq_stride, o_head_stride, _ = o.stride
        q_head_base = head_idx if cutlass.const_expr(not self.pack_gqa) else head_idx * cutlass.Int32(self.qh_per_kh)
        if cutlass.const_expr(self.thd_varlen):
            # Packed view has batch 1: the sequence's token base replaces the
            # batch stride term, and every Q/O row index below stays
            # sequence-local.
            q_head_off = q_row_base * q_seq_stride + q_head_base * q_head_stride
            o_head_off = q_row_base * o_seq_stride + q_head_base * o_head_stride
        else:
            q_head_off = batch_idx * q_batch_stride + q_head_base * q_head_stride
            o_head_off = o_batch_idx * o_batch_stride + q_head_base * o_head_stride
        kv_head_idx = q_head_base // (num_heads_q // num_heads_kv)

        num_kv_tiles = ceil_div(seqlen_k, self.kv_tile)
        if cutlass.const_expr(self.thd_varlen):
            # The grid covers ceil(max_seq_q / q_tile) tiles per sequence; a
            # tile past this sequence's Q length has no rows to produce.
            # Zeroing its KV work makes the whole CTA drain through the
            # barriers without loads, compute, or stores.
            if q_seq_idx >= seqlen_q:
                num_kv_tiles = cutlass.Int32(0)
        if cutlass.const_expr(self.is_causal):
            causal_k_end = q_seq_idx + (self.q_tile // self.qh_per_kh if self.pack_gqa else self.q_tile) + self.window_right
            if cutlass.const_expr(self.bottom_right):
                causal_k_end += seqlen_k - seqlen_q
            causal_k_end = cute.math.max(cutlass.Int32(0), cute.math.min(causal_k_end, seqlen_k))
            num_kv_tiles_causal = ceil_div(causal_k_end, self.kv_tile)
            num_kv_tiles = cute.math.min(num_kv_tiles, num_kv_tiles_causal)

        min_kv_tile = cutlass.Int32(0)
        if cutlass.const_expr(self.window_size_left is not None):
            first_q_position = q_seq_idx
            if cutlass.const_expr(self.bottom_right):
                first_q_position += seqlen_k - seqlen_q
            first_valid_col = cute.math.max(cutlass.Int32(0), first_q_position - self.window_size_left)
            min_kv_tile = first_valid_col // self.kv_tile
        if cutlass.const_expr(self.split_kv > 1):
            # Cut the ALREADY-masked [min_kv_tile, num_kv_tiles) into SPLIT_KV
            # near-equal chunks; the first `rem` splits take one extra tile so
            # the slowest split (which sets the critical path) is minimal.  A
            # split past the end collapses to lo == hi, which drives has_kv_work
            # false below -- the epilogue then produces row_sum = 0, i.e.
            # O := 0 / LSE := -inf, the identity of the combine's log-sum-exp.
            _span = num_kv_tiles - min_kv_tile
            _per = _span // cutlass.Int32(self.split_kv)
            _rem = _span % cutlass.Int32(self.split_kv)
            _lo = min_kv_tile + split_idx * _per + cute.math.min(split_idx, _rem)
            _extra = cutlass.Int32(1) if split_idx < _rem else cutlass.Int32(0)
            min_kv_tile = _lo
            num_kv_tiles = _lo + _per + _extra
        has_kv_work = num_kv_tiles > 0 and (num_kv_tiles - 1) >= min_kv_tile
        # Tiles warp 0 will push through the K/V mbarriers for this unit.
        # Branchless on purpose: when has_kv_work is false the difference is
        # already <= 0, so this is exactly zero.
        tiles_loaded = cute.math.max(cutlass.Int32(0), num_kv_tiles - min_kv_tile)

        # THD collapses the packed view's batch coordinate to 0; the
        # per-sequence token base rides the seq coordinate instead. Every
        # K/V load (including the first) must apply both, or batch >= 1
        # reads the wrong packed rows.
        tma_batch_idx = batch_idx
        if cutlass.const_expr(self.thd_varlen):
            tma_batch_idx = cutlass.Int32(0)
        # /////////////////////////////////////////////////////////////////////////////
        #  COMPUTE (every warp)
        # /////////////////////////////////////////////////////////////////////////////
        compute_warp_idx = warp
        # Warp pair: slab = the 16-row Q slab, half = the head-dim half of Q/K
        # and V/O this warp owns; the partner is the slab's other half.
        slab = warp // self.HEAD_SPLIT
        half = warp % self.HEAD_SPLIT
        partner_warp_idx = slab * self.HEAD_SPLIT + (self.HEAD_SPLIT - 1 - half)
        q_warp_row0 = slab * self.MMA_TILER[0]
        q_col_half0 = half * (self.head_tile_qk // self.HEAD_SPLIT)
        v_col_half0 = half * (self.head_tile_v // self.HEAD_SPLIT)
        k_half_off = half * self.k_half_elems
        v_half_off = half * self.v_half_elems

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
            phase_base=phase_base,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            head_dim_qk=head_dim_qk,
            q_ptr=q_ptr,
            batch_idx=batch_idx,
            head_idx=head_idx,
            q_seq_idx=q_seq_idx,
            q_head_off=q_head_off,
            q_seq_stride=q_seq_stride,
            q_head_stride=q_head_stride,
            q_warp_row0=q_warp_row0,
            lane=lane,
            lane_div8=lane_div8,
            lane_mod8=lane_mod8,
            lane_div16=lane_div16,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            k_tma_mbar=k_tma_mbar,
            v_tma_mbar=v_tma_mbar,
            sX=sX,
            compute_warp_idx=compute_warp_idx,
            slab=slab,
            half=half,
            partner_warp_idx=partner_warp_idx,
            q_col_half0=q_col_half0,
            k_half_off=k_half_off,
            v_half_off=v_half_off,
            tma_batch_idx=tma_batch_idx,
            kv_head_idx=kv_head_idx,
            kv_row_base=kv_row_base,
            min_kv_tile=min_kv_tile,
            k_envelope=k_envelope,
            v_envelope=v_envelope,
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

        kv_seq_idx = cutlass.Int32(0)
        if cutlass.const_expr(self.tma_q):
            # Dense: the Q tile rides the K/V slab, free until the first K/V tile.
            # Its TMA copy is already in flight, issued by the previous unit's
            # epilogue (the kernel prologue for a CTA's first unit); every warp
            # reads its A fragments with ldmatrix, and the CTA barrier hands the
            # slab to the K/V pipeline.
            kv_seq_idx = (num_kv_tiles - 1) * self.kv_tile
            # The Q mbarrier completes once per unit: its parity is the parity of
            # the CTA's round (unit_idx // grid size).
            _gdim, _, _ = cute.arch.grid_dim()
            q_parity = (unit_idx // _gdim) & cutlass.Int32(1)
            while not prims.mbarrier_try_wait_parity(q_tma_mbar, q_parity):
                pass
            q_regs = self.load_q_frags_from_smem(basic_params, sKV)
            prims.barrier_cta_sync(self.bar_compute_sync, thread_count=self.threads_compute)

        # Warp 0 issues the first K/V tile's TMA loads; compute_one_kv_tile
        # issues the rest.
        if warp == 0 and has_kv_work:
            if cutlass.const_expr(not self.tma_q):
                kv_seq_idx = (num_kv_tiles - 1) * self.kv_tile
            self.load_one_kv_tile(sK, tma_k_desc, k_tma_mbar, tma_batch_idx, kv_head_idx, kv_row_base + kv_seq_idx, is_v=False, envelope=k_envelope)
            self.load_one_kv_tile(sV, tma_v_desc, v_tma_mbar, tma_batch_idx, kv_head_idx, kv_row_base + kv_seq_idx, is_v=True, envelope=v_envelope)

        # Load Q into registers.
        if cutlass.const_expr(not self.tma_q):
            q_regs = self.load_q_tile(basic_params)

        if negate_q:
            # Flip the sign bit of every packed Q byte (FP8X4_SIGN_BITS).
            for i in cutlass.range_constexpr(self.qk_d_frags * 4):
                q_regs[i] = q_regs[i] ^ cutlass.Int32(FP8X4_SIGN_BITS)

        # Main attention loop.
        mask_steps = 1
        if cutlass.const_expr(self.is_causal):
            mask_steps = ceil_div(self.q_tile // self.qh_per_kh if self.pack_gqa else self.q_tile, self.kv_tile)
            if cutlass.const_expr(self.diag_shifted):
                # A translated diagonal (bottom-right anchoring or a right
                # band) can straddle one additional KV tile; the frontier
                # width itself is R-independent — the band only translates
                # the diagonal.
                mask_steps = ceil_div((self.q_tile // self.qh_per_kh if self.pack_gqa else self.q_tile) + self.kv_tile - 1, self.kv_tile)
        left_mask_steps = 1
        if cutlass.const_expr(self.window_size_left is not None):
            left_mask_steps = ceil_div((self.q_tile // self.qh_per_kh if self.pack_gqa else self.q_tile) + self.kv_tile - 1, self.kv_tile)

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
            # Guard against the split's START, not 0. Without KV split
            # min_kv_tile is 0 on this branch so the two agree, but a split
            # begins partway into the KV range -- and warp 0 bounds the K/V
            # refills it issues by min_kv_tile.
            if kv_tile_idx >= min_kv_tile:
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
            row_sum[row_half] = row_sum[row_half] * cutlass.Float32(2.0**-P_CAST_LOG2_SCALE)
            row_max_nat = row_max[row_half] * softmax_scale_log2 * LN2
            if cutlass.const_expr(self.has_sink):
                sinks_arr = cutlass.make_array_view(sinks)
                _sink_head = q_head_base if cutlass.const_expr(not self.pack_gqa) else q_head_base + (q_warp_row0 + (lane // 4) + row_half * 8) % self.qh_per_kh
                sink_logit = cutlass.Float32(sinks_arr[_sink_head])
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

        for row_half in cutlass.range_constexpr(2):
            row_sum_inv[row_half] = row_sum_inv[row_half] * o_scale_fused

        if cutlass.const_expr(lse is not None):
            # Both warps of a slab hold the same stats; the half-0 warp stores them.
            if half == 0 and lane % 4 == 0:
                lse_arr = cutlass.make_array_view(lse)
                for row_half in cutlass.range_constexpr(2):
                    _lse_row_in_cta = q_warp_row0 + (lane // 4) + row_half * 8
                    lse_q_idx = q_seq_idx + (_lse_row_in_cta if cutlass.const_expr(not self.pack_gqa) else _lse_row_in_cta // self.qh_per_kh)
                    _lse_head = q_head_base if cutlass.const_expr(not self.pack_gqa) else q_head_base + _lse_row_in_cta % self.qh_per_kh
                    lse_out = cutlass.Float32(row_lse[row_half])
                    # Split partials stay natural-log; the combine owns final base conversion.
                    if cutlass.const_expr(self.stats_log2 and self.split_kv == 1):
                        lse_out = lse_out * cutlass.Float32(1.4426950408889634)
                    if cutlass.const_expr(self.thd_varlen):
                        # Packed ragged-Stats LSE, written directly in the
                        # caller's declared layout: token-major (T, H) or
                        # head-major (H, head_stride).
                        # Rows past this sequence's Q length belong to the
                        # NEXT sequence — never written, and there is no
                        # padded region to trim.
                        if lse_q_idx < seqlen_q:
                            if cutlass.const_expr(self.thd_lse_padded):
                                # per-batch padded Stats (B, H, s_max), no ragged offsets
                                lse_arr[batch_idx, _lse_head, lse_q_idx] = lse_out
                            elif cutlass.const_expr(self.thd_lse_head_major):
                                lse_row = lse_arr[_lse_head, :]
                                lse_row[q_row_base + lse_q_idx] = lse_out
                            else:
                                lse_row = lse_arr[q_row_base + lse_q_idx, :]
                                lse_row[_lse_head] = lse_out
                    else:
                        # Rows at/past this batch's Q length trim to -inf.
                        if lse_q_idx >= seqlen_q:
                            lse_out = -cutlass.Float32.inf
                        if lse_q_idx < q.shape[1]:
                            lse_arr[o_batch_idx, _lse_head, lse_q_idx] = lse_out

        prims.barrier_cta_sync(self.bar_compute_sync, thread_count=self.threads_compute)

        if cutlass.const_expr(self.tma_q):
            # Every warp is past its last K/V wait and no K/V copy is in flight, so
            # the K/V mbarriers are re-armed for the next unit: its tile parity
            # restarts at 0 and no phase count crosses the unit boundary. The
            # unit-end CTA barrier orders the re-arm before the next unit's waits.
            if compute_warp_idx == 0:
                if prims.elect_sync():
                    prims.mbarrier_inval(k_tma_mbar)
                    prims.mbarrier_inval(v_tma_mbar)
                    prims.mbarrier_init(k_tma_mbar, 1)
                    prims.mbarrier_init(v_tma_mbar, 1)
                prims.fence_mbarrier_init()
            next_unit = self._next_dense_unit(unit_idx)
            _n_batch, _n_qh, n_units = self._dense_unit_count(q, n_q_tiles)
            if compute_warp_idx == 0 and next_unit < n_units:
                next_q_tile_idx, next_batch_idx, next_head_idx, _, _ = self._dense_unit_coords(next_unit, n_q_tiles, _n_batch, _n_qh, k)
                next_q_seq_idx = next_q_tile_idx * cutlass.Int32(self.q_tile_tokens)
                next_q_head_base = next_head_idx * cutlass.Int32(self.qh_per_kh if self.pack_gqa else 1)
                self.load_q_tile_tma(sKV, tma_q_desc, q_tma_mbar, next_batch_idx, next_q_head_base, next_q_seq_idx, envelope=k_envelope)

        # Normalize and store each warp's output half. Two 4-KiB rounds per
        # warp reuse the score-exchange allocation while the next Q TMA runs.
        pairs_per_round = self.pv_d_frags // 4
        sO = sX.data_ptr() + compute_warp_idx * (pairs_per_round * 16 * 16 // 2)
        if cutlass.const_expr(self.out_dtype.bytes == 1):
            # Each warp owns a 16x256-byte region. XOR only the vector index:
            # scatter rows use distinct banks and every 16-byte drain stays contiguous.
            sO_fp8 = cutlass.Array(sO, dtype=cutlass.Uint16, shape=16 * 256 // 2)
        row_sum_inv_vec = cutlass.Vector.from_elements(
            (row_sum_inv[0], row_sum_inv[0], row_sum_inv[1], row_sum_inv[1], row_sum_inv[0], row_sum_inv[0], row_sum_inv[1], row_sum_inv[1]),
            cutlass.Float32,
        )
        row_valid = cutlass.Array(cutlass.Float32, 2, alignment=8)
        for row_half in cutlass.range_constexpr(2):
            amax_row_in_cta = q_warp_row0 + (lane // 4) + row_half * 8
            amax_q_idx = q_seq_idx + (amax_row_in_cta if cutlass.const_expr(not self.pack_gqa) else amax_row_in_cta // self.qh_per_kh)
            row_valid[row_half] = cutlass.Float32(1.0) if amax_q_idx < seqlen_q else cutlass.Float32(0.0)
        lane_amax_half = cutlass.Array(cutlass.Float32, 4, alignment=16)
        for i in cutlass.range_constexpr(4):
            lane_amax_half[i] = 0.0
        store_row = lane_mod8 + (lane_div8 % 2) * 8
        store_col = lane_div16 * 8
        store_row_in_cta = q_warp_row0 + store_row
        store_q_seq_idx = q_seq_idx + (store_row_in_cta if cutlass.const_expr(not self.pack_gqa) else store_row_in_cta // self.qh_per_kh)
        store_head_off = o_head_off
        if cutlass.const_expr(self.pack_gqa and self.qh_per_kh != 1):
            store_head_off = store_head_off + (store_row_in_cta % self.qh_per_kh) * o_head_stride
        for o_round in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(pairs_per_round):
                d_frag_pair = o_round * pairs_per_round + j
                o_off = d_frag_pair * 8
                o_scaled = fmul2(o_regs[o_off:8], row_sum_inv_vec)
                for i in cutlass.range_constexpr(8):
                    row_half = (i // 2) % 2
                    acc = row_half * 2 + i % 2
                    lane_amax_half[acc] = cute.arch.fmax(lane_amax_half[acc], cute.math.abs(o_scaled[i]))
                if cutlass.const_expr(self.out_dtype.bytes == 1):
                    for frag in cutlass.range_constexpr(2):
                        for row_half in cutlass.range_constexpr(2):
                            e = frag * 4 + row_half * 2
                            o_pair = fp32_to_fp8x2(o_scaled[e], o_scaled[e + 1], dtype=self.out_dtype)
                            local_row = (lane // 4) + row_half * 8
                            local_col = d_frag_pair * 16 + frag * 8 + (lane % 4) * 2
                            physical_col = local_col ^ ((local_row & 7) << 4)
                            sO_fp8.data_ptr((local_row * 256 + physical_col) // 2).store(o_pair, alignment=2)
                else:
                    o_packed = o_scaled.to(self.out_dtype).bitcast(cutlass.Int32)
                    sO_ptr = sO + j * (16 * 16 // 2) + lane * 4
                    prims.stmatrix(sO_ptr, o_packed, prims.MMALayout.ROW)
            if cutlass.const_expr(self.out_dtype.bytes == 2):
                prims.bar_warp_sync(0xFFFFFFFF)
                for j in cutlass.range_constexpr(pairs_per_round):
                    d_frag_pair = o_round * pairs_per_round + j
                    store_col_in_cta = v_col_half0 + d_frag_pair * 16 + store_col
                    sO_ptr = sO + j * (16 * 16 // 2) + lane * 4
                    if cutlass.const_expr(self.thd_varlen):
                        if store_q_seq_idx < seqlen_q and store_col_in_cta < head_dim_v:
                            gO_ptr = o_ptr + store_head_off + store_q_seq_idx * o_seq_stride + store_col_in_cta
                            gO_ptr.store(sO_ptr.load(count=4, alignment=16).bitcast(self.out_dtype), alignment=16)
                    elif store_q_seq_idx < q.shape[1] and store_col_in_cta < head_dim_v:
                        gO_ptr = o_ptr + store_head_off + store_q_seq_idx * o_seq_stride + store_col_in_cta
                        if store_q_seq_idx < seqlen_q:
                            gO_ptr.store(sO_ptr.load(count=4, alignment=16).bitcast(self.out_dtype), alignment=16)
                        else:
                            zero_vec = cutlass.Vector.from_elements(tuple(self.out_dtype(0.0) for _ in range(8)), self.out_dtype)
                            gO_ptr.store(zero_vec, alignment=16)
                prims.bar_warp_sync(0xFFFFFFFF)
        if cutlass.const_expr(self.out_dtype.bytes == 1):
            prims.bar_warp_sync(0xFFFFFFFF)
            for drain_iter in cutlass.range_constexpr(8):
                vector_idx = drain_iter * 32 + lane
                local_row = vector_idx // 16
                local_col = (vector_idx % 16) * 16
                physical_col = local_col ^ ((local_row & 7) << 4)
                row_in_cta = q_warp_row0 + local_row
                q_store_idx = q_seq_idx + (row_in_cta if cutlass.const_expr(not self.pack_gqa) else row_in_cta // self.qh_per_kh)
                head_store_off = o_head_off
                if cutlass.const_expr(self.pack_gqa and self.qh_per_kh != 1):
                    head_store_off = head_store_off + (row_in_cta % self.qh_per_kh) * o_head_stride
                col_store_idx = v_col_half0 + local_col
                staged_ptr = sO + (local_row * 256 + physical_col) // 4
                if cutlass.const_expr(self.thd_varlen):
                    if q_store_idx < seqlen_q and col_store_idx < head_dim_v:
                        gO_ptr = o_ptr + head_store_off + q_store_idx * o_seq_stride + col_store_idx
                        gO_ptr.store(staged_ptr.load(count=4, alignment=16).bitcast(self.out_dtype), alignment=16)
                elif q_store_idx < q.shape[1] and col_store_idx < head_dim_v:
                    gO_ptr = o_ptr + head_store_off + q_store_idx * o_seq_stride + col_store_idx
                    if q_store_idx < seqlen_q:
                        gO_ptr.store(staged_ptr.load(count=4, alignment=16).bitcast(self.out_dtype), alignment=16)
                    else:
                        zeros = cutlass.Vector.from_elements((cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)), cutlass.Int32)
                        gO_ptr.store(zeros.bitcast(self.out_dtype), alignment=16)
        if cutlass.const_expr(self.split_kv == 1):
            lane_amax_o = cute.arch.fmax(
                cute.arch.fmax(lane_amax_half[0], lane_amax_half[1]) * row_valid[0],
                cute.arch.fmax(lane_amax_half[2], lane_amax_half[3]) * row_valid[1],
            )
            prims.atomicrmw(prims.AtomicOp.MAX, cutlass.make_array_view(amax_o), lane_amax_o.bitcast(cutlass.Int32))
        prims.barrier_cta_sync(self.bar_compute_sync, thread_count=self.threads_compute)

        return tiles_loaded

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
        tma_q_desc: cutlass.GridConstant[cuda.TensorMap],
        softmax_scale_log2: cutlass.Float32,
        n_q_tiles: cutlass.Int32,
        amax_o: cute.Tensor,
        o_scale_fused: cutlass.Float32,
        descale_q_t: cute.Tensor,
        descale_k_t: cute.Tensor,
        descale_v_t: cute.Tensor,
        scale_o_t: cute.Tensor,
    ) -> None:
        """SM120 FMHA prefill kernel.

        :param q: Query tensor.
        :param k: Key tensor.
        :param v: Value tensor.
        :param o: Output tensor.
        :param lse: fp32 log-sum-exp output — ``(B, H, Sq)`` dense; packed
            token-major ``(T, H)`` or head-major ``(H, head_stride)`` (per
            ``thd_lse_head_major``) under ``thd_varlen``; or ``None`` to
            compile the LSE store out (the DSL specializes on ``None``).
        :param sinks: ``(H,)`` fp32 per-Q-head sink logits; ``None`` iff the
            kernel is configured without ``has_sink``.
        :param seq_q_lens: Per-batch query lengths, or an unused dummy tensor.
        :param seq_kv_lens: Per-batch key/value lengths, or an unused dummy tensor.
        :param tma_k_desc: Tensor map descriptor for K.
        :param tma_v_desc: Tensor map descriptor for V.
        :param tma_q_desc: Tensor map descriptor for Q (dense path; the K
            descriptor stands in when Q loads per lane).
        :param softmax_scale_log2: ``softmax_scale * log2(e)``, pre-folded host-side.
        :param n_q_tiles: Q tiles per (batch, head): the fastest axis of the dense
            unit decode.
        :param amax_o: ``(1,)`` int32 view of the fp32 Amax_O slot; every CTA
            atomicMax'es the bit pattern of its pre-cast ``max|O|`` into it
            (skipped under KV split, where the combine owns the amax).
        :param o_scale_fused: Host base of the O multiplier (1.0); the kernel
            folds ``descale_v * scale_o * 2^-P_CAST_LOG2_SCALE`` into it.
        :param descale_q_t: ``(1,)`` fp32 device descale of Q, folded with
            ``descale_k_t`` into the softmax scale in-kernel.
        :param descale_k_t: ``(1,)`` fp32 device descale of K.
        :param descale_v_t: ``(1,)`` fp32 device descale of V.
        :param scale_o_t: ``(1,)`` fp32 device Scale_O, applied before the O cast.
        """
        descale_q = cutlass.Float32(cutlass.make_array_view(descale_q_t)[0])
        descale_k = cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
        descale_v = cutlass.Float32(cutlass.make_array_view(descale_v_t)[0])
        scale_o = cutlass.Float32(cutlass.make_array_view(scale_o_t)[0])
        softmax_scale_log2 = softmax_scale_log2 * descale_q * descale_k
        o_scale_fused = o_scale_fused * descale_v * scale_o * cutlass.Float32(2.0**-P_CAST_LOG2_SCALE)
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % cute.arch.WARP_SIZE
        warp = cute.arch.warp_idx()

        # Shared-memory layout:
        #   sK: one kv_tile x head_tile_qk K tile
        #   sV: one kv_tile x head_tile_v V tile
        #   sX: every warp's partial-S exchange slots
        # The epilogue stages O through sX (in two rounds) while the K/V
        # storage may already be receiving the next unit's Q tile.
        sKV = cutlass.Array(
            k.dtype,
            self.k_tile_elems + self.v_tile_elems,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        sK = sKV
        sV = sKV.subview(self.k_tile_elems)
        sX = cutlass.Array(
            cutlass.Int32,
            self.s_smem_words,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )
        tma_mbar = cutlass.Array(cutlass.Int64, 3, space=cutlass.AddressSpace.smem, alignment=8)
        k_tma_mbar = tma_mbar
        v_tma_mbar = tma_mbar.subview(1)
        q_tma_mbar = tma_mbar.subview(2)

        # Initialize the TMA completion barriers before any load or compute warp
        # can touch the K/V pipeline.
        if warp == 0:
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_k_desc.get_ptr())
                prims.prefetch_tensormap(tma_v_desc.get_ptr())
                if cutlass.const_expr(self.tma_q):
                    prims.prefetch_tensormap(tma_q_desc.get_ptr())
                prims.mbarrier_init(k_tma_mbar, 1)
                prims.mbarrier_init(v_tma_mbar, 1)
                prims.mbarrier_init(q_tma_mbar, 1)
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)

        if cutlass.const_expr(self.thd_varlen and THD_PERSISTENT):
            # Persistent grid: the launch is sized to the MACHINE, not to the
            # plan-time envelope, and each CTA pulls units from a device-side
            # counter until the live total runs out. That total depends on the
            # real sequence lengths, which never reach the host (issue #552), so
            # the setup launch computes it into the metadata buffer. Launching
            # more CTAs than there is work is harmless — the extra ones fail the
            # loop test and retire.
            _meta = cutlass.make_array_view(seq_kv_lens)
            _nb = (seq_kv_lens.shape[0] - 4) // 4
            _live = cutlass.Int32(_meta[4 * _nb + 2])
            _slot = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=16)
            _bidx, _, _ = cute.arch.block_idx()
            _uid = cutlass.Int32(_bidx)
            _phase = cutlass.Int32(0)
            while _uid < _live:
                _qt, _b, _h = thd_decode_unit(
                    _meta,
                    cutlass.Int32(_nb),
                    _uid,
                    cutlass.Int32(q.shape[2]),
                    cutlass.Int32(self.q_tile),
                    self.is_causal,
                )
                _phase = _phase + self._run_unit(
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
                    tma_q_desc,
                    softmax_scale_log2,
                    o_scale_fused,
                    amax_o,
                    sKV,
                    sK,
                    sV,
                    sX,
                    k_tma_mbar,
                    v_tma_mbar,
                    q_tma_mbar,
                    lane,
                    warp,
                    _phase,
                    cutlass.Int32(0),
                    _b,
                    _qt,
                    _b,
                    _h,
                    cutlass.Int32(0),
                    n_q_tiles,
                )
                _uid = thd_claim_next(seq_kv_lens, cutlass.Int32(4 * _nb + 3), _slot, cutlass.Int32(tidx))
        else:
            # Dense: a flat grid over the (Q tile, batch x split, head) units.
            # Each CTA walks rounds of G = grid-size units. Full attention uses
            # boustrophedon order (unit k*G + b, then k*G + (G-1-b), ...), which
            # balances the heaviest-first LPT order; windowed attention uses a
            # plain grid stride because units have similar costs. One CTA per
            # unit when the grid is the unit count. Under tma_q the next unit's Q
            # tile is issued from the current unit's epilogue, the CTA's first
            # from here.
            _bidx, _, _ = cute.arch.block_idx()
            _n_batch, _n_qh, _n_units = self._dense_unit_count(q, n_q_tiles)
            _uid = cutlass.Int32(_bidx)
            if cutlass.const_expr(self.tma_q):
                _q_envelope = q.shape[3] != self.head_tile_qk
                if warp == 0 and _uid < _n_units:
                    _qt, _b, _h, _, _ = self._dense_unit_coords(_uid, n_q_tiles, _n_batch, _n_qh, k)
                    _q_seq = _qt * cutlass.Int32(self.q_tile_tokens)
                    _q_head_base = _h * cutlass.Int32(self.qh_per_kh if self.pack_gqa else 1)
                    self.load_q_tile_tma(sKV, tma_q_desc, q_tma_mbar, _b, _q_head_base, _q_seq, envelope=_q_envelope)
            while _uid < _n_units:
                q_tile_idx, batch_idx, head_idx, split_idx, o_batch_idx = self._dense_unit_coords(_uid, n_q_tiles, _n_batch, _n_qh, k)
                self._run_unit(
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
                    tma_q_desc,
                    softmax_scale_log2,
                    o_scale_fused,
                    amax_o,
                    sKV,
                    sK,
                    sV,
                    sX,
                    k_tma_mbar,
                    v_tma_mbar,
                    q_tma_mbar,
                    lane,
                    warp,
                    cutlass.Int32(0),
                    split_idx,
                    o_batch_idx,
                    q_tile_idx,
                    batch_idx,
                    head_idx,
                    _uid,
                    n_q_tiles,
                )
                _uid = self._next_dense_unit(_uid)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

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
        amax_o: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        o_scale_fused: cutlass.Float32,
        descale_q_t: cute.Tensor,
        descale_k_t: cute.Tensor,
        descale_v_t: cute.Tensor,
        scale_o_t: cute.Tensor,
        thd_max_sq: cutlass.Int32,
        thd_q_lens: Optional[cute.Tensor],
        thd_kv_lens: Optional[cute.Tensor],
        thd_lens_form: Optional[cutlass.Int32],
        thd_n_ctas: cutlass.Int32,
        stream: cuda_driver.CUstream,
    ) -> None:
        """Launch the SM120 cutlass FMHA kernel.

        :param q: Query tensor with shape ``(B, Sq, H, D)``.
        :param k: Key tensor with shape ``(B, Sk, H, D)``.
        :param v: Value tensor with shape ``(B, Sk, H, D)``.
        :param o: Output tensor with shape ``(B, Sq, H, D)``.
        :param lse: fp32 log-sum-exp output — ``(B, H, Sq)`` dense; packed
            token-major ``(T, H)`` or head-major ``(H, head_stride)`` (per
            ``thd_lse_head_major``) under ``thd_varlen``; or ``None`` to
            compile the LSE store out entirely (no dummy buffer needed).
        :param sinks: ``(H,)`` fp32 per-Q-head sink logits; must be ``None``
            exactly when the kernel is configured without ``has_sink``.
        :param seq_q_lens: Per-batch query lengths, or an unused dummy tensor.
        :param seq_kv_lens: Per-batch key/value lengths, or an unused dummy tensor.
        :param amax_o: ``(1,)`` int32 view of the fp32 Amax_O slot (atomicMax target).
        :param softmax_scale_log2: ``softmax_scale * log2(e)``.
        :param o_scale_fused: Host base of the O multiplier (1.0).
        :param descale_q_t: ``(1,)`` fp32 device descales of Q, K, V and the
            device Scale_O (``descale_k_t``, ``descale_v_t``, ``scale_o_t``);
            folded in-kernel, never read on the host.
        :param thd_max_sq: THD only: the PLAN-TIME declared S_q envelope (it
            sizes the per-sequence grid without entering the compile cache
            key; every runtime length is bounded by it, and tiles past a
            sequence's real length drain without loads or stores); 0 /
            ignored when dense.
        :param thd_q_lens: THD only: the CALLER's Q length tensor — (B,)
            per-batch lengths or (B+1,) cu prefix sums — consumed by the
            setup kernel's device-side metadata build (issue #552). None
            (folded out of the ABI) when dense.
        :param thd_kv_lens: THD only: same for the KV side.
        :param thd_lens_form: THD only: runtime bitmask — bit 0: Q is cu,
            bit 1: KV is cu.
        :param thd_n_ctas: Persistent CTA count. THD: the flat machine-sized
            grid that claims live units. Dense: caps the flat unit grid so each
            CTA walks several units and prefetches the next Q tile; 0 = one CTA
            per unit.
        :param stream: CUDA stream used for the launch.
        """
        head_dim_qk = q.shape[3]
        head_dim_v = v.shape[3]
        if cutlass.const_expr(head_dim_qk != k.shape[3] or not FP8_GENERAL_HEAD_TILE_MAX < head_dim_qk <= self.head_tile_qk):
            raise ValueError("runtime Q/K head dimensions must match and be in (256, 512]")
        if cutlass.const_expr(head_dim_v != o.shape[3] or not FP8_GENERAL_HEAD_TILE_MAX < head_dim_v <= self.head_tile_v):
            raise ValueError("runtime V/O head dimensions must match and be in (256, 512]")
        if cutlass.const_expr(head_dim_qk % 16 != 0 or head_dim_v % 16 != 0):
            raise ValueError("head dimensions must be multiples of 16 (TMA 16-byte global-stride rule at 1 B/elem)")

        # THD compiles the token extents DYNAMIC (mode 1 is a symbol, not an
        # int), so only statically-known modes can be compared at trace time;
        # the adapter builds the ragged views from shared totals, so the
        # dynamic seq extents match by construction.
        def _static_neq(a, b):
            return isinstance(a, int) and isinstance(b, int) and a != b

        # Under KV split, O is the split-major PARTIAL workspace: its batch mode
        # is B*SPLIT_KV while Q/K/V keep the real batch, so O's batch is checked
        # against that multiple rather than against Q's.
        if cutlass.const_expr(
            _static_neq(q.shape[0], k.shape[0])
            or any(_static_neq(a, b) for a, b in zip(k.shape[:3], v.shape[:3]))
            or _static_neq(q.shape[0] * self.split_kv, o.shape[0])
            or _static_neq(q.shape[1], o.shape[1])
            or _static_neq(q.shape[2], o.shape[2])
            or q.shape[2] % k.shape[2] != 0
            or (isinstance(q.shape[2], int) and isinstance(k.shape[2], int) and q.shape[2] != k.shape[2] * self.qh_per_kh)
        ):
            raise ValueError("runtime Q/K/V/O batch, sequence, or head geometry mismatch")
        for name, tensor, dtype in (("Q", q, self.in_dtype), ("K", k, self.in_dtype), ("V", v, self.in_dtype), ("O", o, self.out_dtype)):
            if cutlass.const_expr(not self.is_layout_supported(tensor.shape, tensor.stride, dtype.width // 8)):
                raise ValueError(
                    f"{name} layout is not supported: BSHD with the head dim innermost-contiguous "
                    f"and non-overlapping seq/head strides that are multiples of 16 bytes "
                    f"(compact or padded); got shape {tuple(tensor.shape)} stride {tuple(tensor.stride)}"
                )
        if cutlass.const_expr(lse is not None):
            if cutlass.const_expr(self.thd_varlen):
                # The packed token total (q.shape[1]) is DYNAMIC under THD, so
                # only the static modes are trace-checkable. The adapter builds
                # the token-major view from that total; head_stride >= T is the
                # caller's contract for head-major storage.
                if cutlass.const_expr(self.thd_lse_padded):
                    if cutlass.const_expr(len(lse.shape) != 3):
                        raise ValueError("padded THD LSE must be rank-3 (B, H, s_max)")
                elif cutlass.const_expr(self.thd_lse_head_major):
                    if cutlass.const_expr(lse.shape[0] != q.shape[2]):
                        raise ValueError("head-major THD LSE must have shape (H, head_stride) with head_stride >= T")
                    if cutlass.const_expr(lse.stride != (lse.shape[1], 1)):
                        raise ValueError("head-major THD LSE must be compact row-major")
                else:
                    if cutlass.const_expr(lse.shape[1] != q.shape[2]):
                        raise ValueError("THD LSE must have shape (T, H)")
                    if cutlass.const_expr(lse.stride != (q.shape[2], 1)):
                        raise ValueError("THD LSE must be compact token-major")
            else:
                # Under KV split the LSE is the split-major partial workspace,
                # batch mode B*SPLIT_KV (same as O).
                if cutlass.const_expr(lse.shape != (q.shape[0] * self.split_kv, q.shape[2], q.shape[1])):
                    raise ValueError("LSE must have shape (B * split_kv, H, Sq)")
        if cutlass.const_expr(self.has_sink != (sinks is not None)):
            raise ValueError("sinks must be provided exactly when the kernel is configured with has_sink")
        if cutlass.const_expr(sinks is not None and sinks.shape != (q.shape[2],)):
            raise ValueError("sinks must have shape (H,)")
        if cutlass.const_expr(self.thd_varlen):
            if cutlass.const_expr(q.shape[0] != 1):
                raise ValueError("THD Q/K/V/O must be packed batch-1 views")
            if cutlass.const_expr(seq_kv_lens.shape != (4 * self.thd_batch + 4,)):
                raise ValueError("THD seq_kv_lens must be the (4*B+4,) metadata tensor")

        # Split D into I contiguous C-element chunks while preserving the
        # per-tensor TMA descriptor over the declared (B, S, H, D) strides.
        #
        # Exact head dim (the common case): a rank-5 view pre-splits D into
        # swizzle-span chunks as a descriptor dimension — dims (B, H, I, S, C)
        # with D = I * C, TMA order (C, S, I, H, B) — so ONE copy covers the
        # whole tile (the fast path; ~1-2% speedup upon exact-dim workloads).
        #
        # Envelope head dim (actual d < compile-time tile): the head boundary
        # must be a single descriptor dimension for TMA's per-dimension bounds
        # check to zero-fill past it, so a rank-4 view keeps the ACTUAL extent
        # innermost — TMA order (D, S, H, B) — and load_one_kv_tile steps the
        # head coordinate per swizzle-span chunk.
        # K/V TMA layouts read the TENSORS' strides (batch/seq/head), not
        # packed recomputations: a THD view may declare a wider token stride
        # (e.g. a K/V slice of a kv-interleaved [T, 2, H, D] buffer), and TMA
        # encodes it directly (interior strides must be 16-byte multiples;
        # check_support declines declarations TMA cannot express).
        def kv_tma_desc(t, head_dim, head_tile, swizzle, swizzle_chunks, swizzle_chunk_elems, rows=None):
            # Q reuses K's geometry with q_tile rows per box; K/V take kv_tile rows and every chunk.
            rows = self.kv_tile if rows is None else rows
            if cutlass.const_expr(head_dim == head_tile):
                layout = cute.make_layout(
                    (t.shape[0], t.shape[2], swizzle_chunks, t.shape[1], swizzle_chunk_elems),
                    stride=(t.stride[0], t.stride[2], swizzle_chunk_elems, t.stride[1], 1),
                )
                box = (1, 1, swizzle_chunks, rows, swizzle_chunk_elems)
                stride_order = (4, 3, 2, 1, 0)
            else:
                layout = cute.make_layout(
                    (t.shape[0], t.shape[2], t.shape[1], head_dim),
                    stride=(t.stride[0], t.stride[2], t.stride[1], 1),
                )
                box = (1, 1, rows, swizzle_chunk_elems)
                stride_order = (3, 2, 1, 0)
            return cuda.create_tensor_map_tiled_from_view(
                cute.make_tensor(t.iterator, layout),
                box_dims=box,
                stride_order=stride_order,
                swizzle=swizzle,
            )

        tma_k_desc = kv_tma_desc(k, head_dim_qk, self.head_tile_qk, self.k_tma_swizzle, self.k_tma_swizzle_chunks, self.k_swizzle_chunk_elems)
        tma_v_desc = kv_tma_desc(v, head_dim_v, self.head_tile_v, self.v_tma_swizzle, self.v_tma_swizzle_chunks, self.v_swizzle_chunk_elems)
        # Dense Q shares K's head tile and swizzle geometry, with all chunks in
        # one box. Unpacked: q_tile rows of one head. PackGQA:
        # a G-heads x T-tokens box whose dims are ordered (C, H, S, I, B) so it
        # lands token-major (row = token * G + head), the kernel's packed row
        # order. The THD path passes K's descriptor unused.
        if cutlass.const_expr(self.tma_q and self.pack_gqa):
            _q_chunks = self.k_tma_swizzle_chunks
            _q_chunk_elems = self.k_swizzle_chunk_elems
            if cutlass.const_expr(head_dim_qk == self.head_tile_qk):
                _q_layout = cute.make_layout(
                    (q.shape[0], _q_chunks, q.shape[1], q.shape[2], _q_chunk_elems),
                    stride=(q.stride[0], _q_chunk_elems, q.stride[1], q.stride[2], 1),
                )
                _q_box = (1, _q_chunks, self.q_tile_tokens, self.qh_per_kh, _q_chunk_elems)
                _q_order = (4, 3, 2, 1, 0)
            else:
                _q_layout = cute.make_layout(
                    (q.shape[0], q.shape[1], q.shape[2], head_dim_qk),
                    stride=(q.stride[0], q.stride[1], q.stride[2], 1),
                )
                _q_box = (1, self.q_tile_tokens, self.qh_per_kh, _q_chunk_elems)
                _q_order = (3, 2, 1, 0)
            tma_q_desc = cuda.create_tensor_map_tiled_from_view(
                cute.make_tensor(q.iterator, _q_layout),
                box_dims=_q_box,
                stride_order=_q_order,
                swizzle=self.k_tma_swizzle,
            )
        elif cutlass.const_expr(self.tma_q):
            tma_q_desc = kv_tma_desc(
                q, head_dim_qk, self.head_tile_qk, self.k_tma_swizzle, self.k_tma_swizzle_chunks, self.k_swizzle_chunk_elems, rows=self.q_tile
            )
        else:
            tma_q_desc = tma_k_desc
        if cutlass.const_expr(self.thd_varlen):
            # Build the [kv|cu_q|cu_k|remap|live|ctr] metadata buffer DEVICE-side
            # from the caller's length tensors (no host cumsum, no H2D — issue
            # #552); the main kernel launched after it on this stream reads it.
            build_thd_meta_kernel(
                seq_kv_lens,
                thd_q_lens,
                thd_kv_lens,
                thd_lens_form,
                cutlass.Int32(self.thd_batch),
                cutlass.Int32(q.shape[2]),
                cutlass.Int32(self.q_tile),
                thd_n_ctas,
            ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        # Grid geometry. THD: ceil(max_seq_q / q_tile) tiles per sequence over the
        # REAL batch count (the packed view's batch mode is 1); tiles past a shorter
        # sequence's length drain without work. NOTE thd_max_sq is a __call__
        # argument in this base, not a member.
        # PackGQA: S_q*G packed rows per packed head, H_q/G packed heads on the
        # head axis (THD is always unpacked).
        n_q_tiles = (
            ceil_div(thd_max_sq, self.q_tile)
            if cutlass.const_expr(self.thd_varlen)
            else ceil_div((q.shape[1] * self.qh_per_kh if self.pack_gqa else q.shape[1]), self.q_tile)
        )
        n_batch = self.thd_batch if cutlass.const_expr(self.thd_varlen) else q.shape[0]
        n_head = q.shape[2] // self.qh_per_kh if self.pack_gqa else q.shape[2]
        # Dense: a flat grid over the units, capped at thd_n_ctas — the persistent
        # CTA count the adapter sizes to the machine (0 = one CTA per unit). The
        # kernel decodes linear unit ids with the same n_q_tiles; KV split rides
        # the composite batch axis (config_sm120 allows split_kv > 1 only under
        # NATURAL).
        n_units = n_q_tiles * n_batch * self.split_kv * n_head
        n_ctas = cutlass.Int32(thd_n_ctas)
        if n_ctas <= 0 or n_ctas > n_units:
            n_ctas = cutlass.Int32(n_units)
        grid = (n_ctas, cutlass.Int32(1), cutlass.Int32(1))
        # Persistent THD: a flat, machine-sized grid -- the unit a CTA works on
        # comes from the claim counter, not from its block index.
        _persistent = self.thd_varlen and THD_PERSISTENT
        if cutlass.const_expr(_persistent):
            grid = (thd_n_ctas, cutlass.Int32(1), cutlass.Int32(1))
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
            tma_q_desc,
            softmax_scale_log2,
            cutlass.Int32(n_q_tiles),
            amax_o,
            o_scale_fused,
            descale_q_t,
            descale_k_t,
            descale_v_t,
            scale_o_t,
        ).launch(
            grid=grid,
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
    has_lse: bool = True,
    lse_head_major: bool = False,
    lse_head_stride: int = 0,
    lse_padded_rows: int = 0,
    q_stride: Optional[tuple[int, int, int, int]] = None,
    k_stride: Optional[tuple[int, int, int, int]] = None,
    v_stride: Optional[tuple[int, int, int, int]] = None,
    o_stride: Optional[tuple[int, int, int, int]] = None,
    lse_stride: Optional[tuple[int, int, int]] = None,
) -> Callable:
    """Compile and cache one architecture-specific BSHD shape.

    ``d_qk`` is the Q/K head dim (QK^T contraction width) and ``d_v`` the V/O
    head dim (P@V output width).

    THD specializations pack the batch: ``b`` is the real sequence count and
    ``sq``/``skv`` are IGNORED — the packed token totals are runtime values
    (they change every step under continuous batching), so the token extents
    compile DYNAMIC (``cute.sym_int``) and the cache key stays plan-time-only;
    callers must not pass them. ``max_sq`` (the longest sequence's Q length,
    which sizes the per-sequence grid) is likewise a RUNTIME ``__call__``
    argument, not a compile parameter. ``q_stride``..``o_stride`` carry the
    caller's declared BSHD element strides (None = compact); THD strides carry
    a ZERO batch stride (the real view's batch stride is ``t * token_stride``,
    a runtime value; the fake rebuilds it symbolically — batch extent 1 never
    steps).

    ``has_lse=False`` compiles the LSE store out (the kernel specializes on a
    ``None`` LSE argument) — callers that don't want stats pass no LSE buffer
    at all instead of a dummy. Dense ``lse_stride`` carries the caller's
    declared ``(B, H, Sq)`` element strides into the compiled tensor. THD LSE
    is token-major ``(T, H)`` by default;
    ``lse_head_major=True`` switches to head-major ``(H, lse_head_stride)``
    (FlashAttention's ``softmax_lse`` layout), where ``lse_head_stride`` is the
    caller-declared head-row stride (``>= T``, a shape — part of the cache key).
    """

    _cache_key = _template_key(globals(), locals(), "compile")
    if pick_flavor(d_qk, d_v, fp8=True) != D512_FLAVOR:
        raise ValueError(f"SM120 SDPA d512 kernel: head dimensions must both be in (256, 512]; got ({d_qk}, {d_v})")
    kernel = SM120FusedMultiHeadAttentionForward(
        in_dtype=IN_DTYPE,
        out_dtype=OUT_DTYPE,
        is_causal=PARAMS.window_right is not None,
        sched_policy=PARAMS.sched_policy,
        bottom_right=PARAMS.bottom_right,
        window_size_left=PARAMS.window_left,
        window_size_right=PARAMS.window_right,
        seq_q_lens_present=PARAMS.seq_q_lens_present,
        seq_kv_lens_present=PARAMS.seq_kv_lens_present,
        has_sink=PARAMS.has_sink,
        stats_log2=PARAMS.stats_log2,
        thd_varlen=PARAMS.thd_varlen,
        thd_batch=b,
        thd_lse_head_major=lse_head_major,
        thd_lse_padded=bool(lse_padded_rows),
        head_tile_qk=D512_FLAVOR[0],
        head_tile_v=D512_FLAVOR[1],
        q_tile=PARAMS.q_tile,
        kv_tile=PARAMS.kv_tile,
        split_kv=PARAMS.split_kv,
        pack_gqa=PARAMS.pack_gqa,
        qh_per_kh=qh // kh,
    )
    if PARAMS.split_kv > 1 and not has_lse:
        raise ValueError("SM120 SDPA: split_kv > 1 requires an LSE output (the per-split LSE drives the combine)")
    if has_lse and lse_stride is not None and ((PARAMS.thd_varlen and not lse_padded_rows) or PARAMS.split_kv > 1):
        raise ValueError("dense LSE strides are not valid for THD or split-KV workspaces")
    fake_batch = 1 if PARAMS.thd_varlen else b
    if PARAMS.thd_varlen:
        # Dynamic packed token totals: one symbol per ragged group (Q/O and
        # the LSE share t_q; K/V share t_kv), so a new total re-binds the same
        # compiled artifact instead of minting a new one (issue #552).
        sq = cute.sym_int(divisibility=1)
        skv = cute.sym_int(divisibility=1)
    # KV split: O and LSE are the PARTIAL workspaces, stacked split-major on the
    # batch axis (B*SPLIT_KV).  Q/K/V keep the real batch.
    o_fake_batch = fake_batch * PARAMS.split_kv
    lse_fake_batch = fake_batch * PARAMS.split_kv

    def _fake_bshd(dtype, shape, stride):
        if stride is None:
            return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)
        if PARAMS.thd_varlen:
            # Batch stride = tokens * token_stride (`_thd_view`'s envelope),
            # a runtime value: rebuild it from the dynamic token extent.
            return cute.runtime.make_fake_tensor(dtype, shape, (shape[1] * stride[1], stride[1], stride[2], stride[3]), assumed_align=16)
        return cute.runtime.make_fake_tensor(dtype, shape, tuple(stride), assumed_align=16)

    fake_q = _fake_bshd(IN_DTYPE, (fake_batch, sq, qh, d_qk), q_stride)
    fake_k = _fake_bshd(IN_DTYPE, (fake_batch, skv, kh, d_qk), k_stride)
    fake_v = _fake_bshd(IN_DTYPE, (fake_batch, skv, kh, d_v), v_stride)
    fake_o = _fake_bshd(OUT_DTYPE, (o_fake_batch, sq, qh, d_v), o_stride)
    if PARAMS.thd_varlen:
        fake_lse_shape = (b, qh, lse_padded_rows) if lse_padded_rows else ((qh, lse_head_stride) if lse_head_major else (sq, qh))
    else:
        fake_lse_shape = (lse_fake_batch, qh, sq)
    if not has_lse:
        # No Stats output: the LSE argument is None-specialized and the store
        # is compiled out entirely — no dummy buffer exists at any level.
        fake_lse = None
    else:
        fake_lse = (
            cute.runtime.make_fake_tensor(cutlass.Float32, fake_lse_shape, lse_stride, assumed_align=4)
            if lse_stride is not None
            else cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                fake_lse_shape,
                stride_order=(1, 0) if (PARAMS.thd_varlen and not lse_padded_rows) else (2, 1, 0),
                assumed_align=4,
            )
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
        (4 * b + 4,) if PARAMS.thd_varlen else (b,),  # THD: [ seq_kv(B) | cu_q(B+1) | cu_k(B+1) | remap(B) | live | ctr ]
        stride_order=(0,),
        assumed_align=4,
    )
    # THD: the caller's Q/KV length tensors, consumed by the setup kernel's
    # device-side metadata build. DYNAMIC extents — (B,) per-batch lengths and
    # (B+1,) cu prefix sums bind the same artifact; the form rides the runtime
    # thd_lens_form bitmask, so no compile key grows (Rule 4).
    if PARAMS.thd_varlen:
        fake_thd_q_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_lens_form = cutlass.Int32(0)
    else:
        fake_thd_q_lens = None
        fake_thd_kv_lens = None
        fake_thd_lens_form = None
    fake_amax_o = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), stride_order=(0,), assumed_align=4)

    def _fake_scale():
        return cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4)

    return _compile_cached(
        kernel,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        fake_sinks,
        fake_seq_q_lens,
        fake_seq_kv_lens,
        fake_amax_o,
        cutlass.Float32(1.0),
        cutlass.Float32(1.0),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        cutlass.Int32(0),  # thd_max_sq: plan-time envelope grid extent (THD)
        fake_thd_q_lens,
        fake_thd_kv_lens,
        fake_thd_lens_form,
        cutlass.Int32(0),  # thd_n_ctas: persistent THD grid extent (runtime)
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd",
    )
