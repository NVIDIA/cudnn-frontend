# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A fused multi-head attention (FMHA) per-tensor FP8 (e4m3 / e5m2) kernel for
the NVIDIA Blackwell GeForce (SM120 / SM121).

Same architecture as the f16 kernel — dedicated TMA load warp, GMEM-direct Q,
fp32 online softmax in registers, right-to-left masked KV walk — with the MMA
lowered to ``mma.sync.aligned.m16n8k32.row.col.f32.{e4m3|e5m2}.{...}.f32``:

- Q/K/V arrive PRE-QUANTIZED e4m3/e5m2: the kernel never does elementwise
  math on them, so bytes flow TMA -> ldmatrix -> MMA as bit patterns; the
  element dtype only selects the MMA tag and the P-cast cvt tag.
  ``descale_q * descale_k`` folds IN-KERNEL from 1-element device scale
  tensors into ``softmax_scale_log2`` (Rule 3 — no host readback) and
  ``descale_v * scale_o`` into the ``o_scale_fused`` scalar (cuDNN SDPA_FP8
  node convention; see the SM100 fp8 adapter).
- K B-fragments: classic byte-preserving ``ldmatrix.m8n8.x4.b16`` (sm_120a has
  no non-transposed 8-bit ldmatrix; the b16 form gathers the right bytes when
  each lane points at one 16-byte K-segment).
- V B-fragments: hardware 8-bit transposed ``ldmatrix.m16n16.x2.trans.b8``
  (SASS ``LDSM.8.MT1616``) — one issue covers a 32(kv) x 16(d-bytes) tile and
  feeds two MMAs, register map (0, 2, 1, 3).
- P: fp32 softmax output packed to the INPUT fp8 dtype with
  ``cvt.rn.satfinite.{e4m3x2|e5m2x2}.f32`` and kept in REGISTERS — the k32 C->A
  fragment-column mismatch is an exchange inside each thread quad, so two
  ``shfl.sync`` and one ``prmt`` in ``mma_pv`` replace the SMEM round trip (and
  the 16 KB it needed). ``Scale_S`` multiplies P immediately before the cast;
  the softmax denominator reads the UNSCALED fp32 P.
- O may be FP16 / BF16 / E4M3 / E5M2: 2-byte O rides the stmatrix staging
  epilogue (the sKV/sO SMEM alias is sized in BYTES because KV and O element
  sizes differ); fp8 O takes a direct GMEM quantizing store.
- ``Amax_O`` (max ``|o_scaled|`` pre-cast) is produced via bitcast-int32
  atomic max on a 1-element Int32 buffer the host pre-zeros on the launch
  stream; the host divides it by ``scale_o`` afterwards (SM100 fp8
  convention). There is no Amax_S output — nothing consumes it, and graphs
  that declare one are declined at the engine row.

Constraints:
- Input dtype: e4m3 or e5m2 (the MMA tag and the P quantization target
  follow it); output dtype FP16 / BF16 / E4M3 / E5M2 (fp8 O via a direct
  quantizing store, ``o_scale_fused`` applied before the cast)
- Head TILES are multiples of 32 between 32 and 256, inclusive; actual head
  dims may be smaller multiples of 16 (TMA 16-byte global-stride rule at
  1 byte/elem) — TMA zero-fills the pad columns (the head-dim ENVELOPE)
- Q heads must be divisible by the number of K/V heads
- Q/K/V/O use compact BSHD storage (THD packs them to ``(1, T, H, D)``)
- Supported CTA Q/KV tiles are 128 or 64
- Optional per-Q-head attention-sink logits folded into the softmax
  denominator: the sink is a virtual column with no V row — it rescales O
  and enters the LSE.
- THD (ragged) is supported with token-major or head-major Stats
"""

from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute

from cutlass.experimental import primitives as prims
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_E5M2, DTYPE_FP16
from cudnn.frost.tile_dsl.mma import mma_m16n8k32_f32
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp8x2, pack_fp8x2_pairs
from cudnn.frost.tile_dsl.scheduler import (
    SCHED_LPT_L2,
    SCHED_NATURAL,
    lpt_tile_coords,
    lpt_l2_tile_coords,
)
from cudnn.frost.tile_dsl.swizzle import swizzle_xor
from cudnn.sdpa.fwd.kernels.thd_sm100 import build_thd_meta_kernel as _build_thd_meta_kernel
from cudnn.sdpa.fwd.config_sm120 import (
    FP8_HEAD_TILE_GRANULE,
    SEQ_KV_TILES as _SEQ_KV_TILES,
    SEQ_Q_TILES as _SEQ_Q_TILES,
    SUPPORTED_HEAD_TILES_FP8 as _SUPPORTED_HEAD_TILES_FP8,
    TemplateParams,
    validate_params,
)

# The FROST loader injects one immutable specialization before executing this
# module. A direct import uses dense e4m3 defaults.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(dtype_qkv=DTYPE_E4M3))

# P -> fp8 cast bias (BAKED constant — NOT cuDNN's Scale_S; that pair is
# accepted and ignored). P is quantized as P * 2**P_CAST_LOG2_SCALE via the
# exp2 bias: the lazy-rescale skip bounds P by 2**rescale_threshold (4.0 for
# fp8), so the cast peaks at 2^(4+4) = 256 < 448 (e4m3 max) — no saturation —
# while flat-row entries (P ~ 1/S) sit four binades above e4m3's subnormal
# cliff (quantization stays normal out to S ~ 2^13). row_sum accumulates in
# the same 2^4-scaled units and is de-scaled by the EXACT 2^-4 before the
# finalize paths (sink mix, rcp, LSE, zero-row guards run on bit-identical
# true sums); the O leg's 2^4 is cancelled by the 2^-4 folded into
# o_scale_fused. Invariant: rescale_threshold + P_CAST_LOG2_SCALE <= log2(448).
P_CAST_LOG2_SCALE = 4.0
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


# L2 working-set budget used by the LPT_L2 group sizing.
_SCHED_L2_BUDGET_BYTES = 50 * 1024 * 1024


def ceil_div(a: int, b: int) -> int:
    """Return the ceiling division of a by b."""
    return (a + b - 1) // b


def round_up_head_tile(d: int) -> int:
    """Smallest supported head TILE covering an actual head dim ``d``."""
    return ceil_div(d, FP8_HEAD_TILE_GRANULE) * FP8_HEAD_TILE_GRANULE


fmul2 = partial(prims.mul_packed_f32x2, ftz=False, rnd=prims.FPRoundingMode.RN)
fma2 = partial(prims.fma_packed_f32x2, ftz=False, rnd=prims.FPRoundingMode.RN)


# ---------------------------------------------------------------------------
# Main kernel class
# ---------------------------------------------------------------------------


class SM120FusedMultiHeadAttentionForward:
    """Configure and launch the SM120/SM121 per-tensor FP8 FMHA prefill kernel."""

    SEQ_Q_TILES = _SEQ_Q_TILES
    SEQ_KV_TILES = _SEQ_KV_TILES
    SUPPORTED_HEAD_TILES = _SUPPORTED_HEAD_TILES_FP8
    MMA_TILER = (16, 8, 32)  # mma.sync.aligned.m16n8k32 (e4m3)

    @staticmethod
    def is_layout_supported(
        shape: tuple[int, ...],
        stride: tuple[int, ...],
    ) -> bool:
        """Return whether a BSHD tensor uses compact storage."""

        if len(shape) != 4 or len(stride) != 4:
            return False
        _, sequence, heads, head_dim = shape
        if not isinstance(sequence, int):
            # THD: the packed token total is DYNAMIC; the batch stride is the
            # matching symbol (batch extent 1 never steps), so only the
            # static inner strides are trace-checkable.
            return stride[1:] == (heads * head_dim, head_dim, 1)
        return stride == (
            sequence * heads * head_dim,
            heads * head_dim,
            head_dim,
            1,
        )

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
        thd_lse_head_major: bool = False,
        thd_batch: int = 1,
        head_tile_qk: int = 128,
        head_tile_v: int = 128,
        kv_tile: int = SEQ_KV_TILES[0],
        q_tile: int = SEQ_Q_TILES[0],
        pack_gqa: bool = False,
        qh_per_kh: int = 1,
    ):
        """Initialize the FMHA prefill kernel configuration.

        :param in_dtype: Q/K/V element type, Float8E4M3FN or Float8E5M2.
            Selects the MMA operand tag and the P-quantization target (the
            SDPA_FP8 contract quantizes P at the input dtype); TMA/ldmatrix/
            swizzle are byte-oriented and do not care.
        :param out_dtype: O element type: Float16, BFloat16, Float8E4M3FN,
            or Float8E5M2. 2-byte types ride the stmatrix staging epilogue;
            fp8 takes the direct quantizing-store one.
        :param is_causal: Apply an upper causal bound to QK.
        :param bottom_right: Shift the causal diagonal by ``Skv - Sq``.
        :param window_size_left: Inclusive left-window offset, or ``None``.
        :param window_size_right: Widen the causal diagonal to the right by
            R columns (keep ``j <= diag + R``, inclusive), or ``None``. The
            loader maps ``window_right`` to ``is_causal=(window_right is not
            None)`` plus this widening.
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
        :param head_tile_qk: Q/K head TILE (the QK^T contraction width). Must
            be a multiple of 32 between 32 and 256, inclusive. The ACTUAL head
            dim may be any multiple of 16 up to the tile — the TMA descriptors
            keep the actual extents and zero-fill the pad columns.
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
        for tile_name, tile in (("head_tile_qk", head_tile_qk), ("head_tile_v", head_tile_v)):
            if tile not in _SUPPORTED_HEAD_TILES_FP8:
                raise ValueError(
                    f"{tile_name} must be a multiple of {FP8_HEAD_TILE_GRANULE} between "
                    f"{_SUPPORTED_HEAD_TILES_FP8[0]} and {_SUPPORTED_HEAD_TILES_FP8[-1]}, got {tile}"
                )
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.is_causal = is_causal
        self.sched_policy = sched_policy
        self.bottom_right = bottom_right
        self.window_size_left = window_size_left
        # The band only TRANSLATES the diagonal (frontier width is
        # R-independent); masking sites add the widening, mask_steps keys off
        # diag_shifted. 0 = plain causal.
        self.window_right = window_size_right if window_size_right is not None else 0
        self.diag_shifted = bottom_right or self.window_right > 0
        self.seq_q_lens_present = seq_q_lens_present
        self.seq_kv_lens_present = seq_kv_lens_present
        self.has_sink = has_sink
        self.thd_varlen = thd_varlen
        self.thd_lse_head_major = thd_lse_head_major
        self.thd_batch = thd_batch

        self.head_tile_qk = head_tile_qk
        self.head_tile_v = head_tile_v
        self.q_tile = q_tile
        self.kv_tile = kv_tile
        self.pack_gqa = pack_gqa
        self.qh_per_kh = qh_per_kh

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
        col0_in_cta = col0
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
                # redistributes it to the k32 A layout with shfl. P is cast
                # UNSCALED (descale_s/scale_s are unsupported on this cell,
                # like SM100).
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
    def sanitize_v_tail(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        kv_seq_idx: cutlass.Int32,
    ) -> None:
        """Zero ``sV`` rows at/past this tile's valid KV extent before P @ V.

        The K/V TMA descriptors span the bound buffers' CAPACITY (under THD
        the packed totals are device values, so the views bind whole-buffer
        extents), so rows between the valid KV length and the tile end can
        carry UNINITIALIZED storage — including fp8 NaN bit patterns. The
        S-side mask overwrites those columns with -inf (a select, NaN-safe),
        but P @ V still multiplies P = 0 against the NaN V row and
        ``0 * NaN = NaN`` poisons the whole accumulator column-free. Reached
        only from THD specializations' first masked step (see the call
        site): dense descriptors carry the declared S_kv, so their overhang
        loads zero-fill in hardware — a dense PADDED graph's pad rows are
        user memory and deliberately NOT sanitized here (whether the
        contract requires tolerating NaN bit patterns there is an open
        question for the sibling kernels too).

        Every compute warp redundantly zeroes the full overhang (idempotent
        zero stores race benignly), so a warp-level sync is enough for each
        warp's own ``ldmatrix`` lanes to observe the zeros.
        """
        segs_per_row = self.head_tile_v // 16  # 16-byte segments per V row
        row_lo = cute.math.max(cutlass.Int32(0), basic_params.seqlen_k - kv_seq_idx)
        for r_it in cutlass.range_constexpr(self.kv_tile // 32):
            row = cutlass.Int32(r_it * 32) + basic_params.lane
            for seg in cutlass.range_constexpr(segs_per_row):
                col = seg * 16
                phys_row = (col // self.v_swizzle_chunk_elems) * self.kv_tile + row
                sv_ptr = (
                    mma_params.sV.data_ptr()
                    + phys_row * self.v_swizzle_chunk_elems
                    + swizzle_xor(
                        phys_row,
                        col % self.v_swizzle_chunk_elems,
                        self.v_swizzle_chunk_elems,
                        self.in_dtype.bytes,
                    )
                )
                zero16 = cutlass.Vector.from_elements(
                    tuple(self.in_dtype(0.0) for _ in range(16)),
                    self.in_dtype,
                )
                if row >= row_lo:
                    sv_ptr.store(zero16, alignment=16)
        prims.bar_warp_sync(cute.arch.FULL_MASK)

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

        if cutlass.const_expr(self.thd_varlen and in_mask_steps and is_first_kv_tile):
            self.sanitize_v_tail(basic_params, mma_params, kv_tile_idx * self.kv_tile)
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
        amax_o: cute.Tensor,
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda.TensorMap],
        softmax_scale_log2: cutlass.Float32,
        o_scale_fused: cutlass.Float32,
        n_q_tiles: cutlass.Int32,
        # 1-element fp32 DEVICE scales (Rule 3): loaded and folded in-kernel —
        # the scalar args arrive as attn_scale*log2(e) / 1.0 and no host
        # readback exists anywhere. descale_s/scale_s are NOT taken: P is
        # cast unscaled (unsupported knobs on this cell, like SM100).
        descale_q_t: cute.Tensor,
        descale_k_t: cute.Tensor,
        descale_v_t: cute.Tensor,
        scale_o_t: cute.Tensor,
    ) -> None:
        """SM120 per-tensor FP8 FMHA prefill kernel.

        :param q: Query tensor (fp8).
        :param k: Key tensor (fp8).
        :param v: Value tensor (fp8).
        :param o: Output tensor (the configured ``out_dtype``).
        :param lse: fp32 log-sum-exp output — ``(B, H, Sq)`` dense, or packed
            token-major ``(T, H)`` / head-major ``(H, head_stride)`` under
            ``thd_varlen``; or ``None`` to compile the LSE store out (the DSL
            specializes on ``None``).
        :param sinks: ``(H,)`` fp32 per-Q-head sink logits; ``None`` iff the
            kernel is configured without ``has_sink``.
        :param seq_q_lens: Per-batch query lengths, or an unused dummy tensor.
        :param seq_kv_lens: Per-batch key/value lengths, or an unused dummy tensor.
        :param amax_o: 1-element Int32 buffer, host pre-zeroed; receives
            bitcast-fp32 atomic max of ``|o_scaled|`` pre-cast. The host
            divides by ``scale_o`` afterwards. Pass a dummy when unused.
        :param tma_k_desc: Tensor map descriptor for K.
        :param tma_v_desc: Tensor map descriptor for V.
        :param softmax_scale_log2: ``softmax_scale * log2(e)`` — descale_q *
            descale_k folds in IN-KERNEL from the device scale tensors.
        :param o_scale_fused: base ``1.0`` — descale_v * scale_o folds in
            IN-KERNEL from the device scale tensors. descale_s/scale_s are
            unsupported on this cell (P is cast unscaled, like SM100).
        :param descale_q_t: 1-element fp32 DEVICE tensors (with descale_k_t /
            descale_v_t / scale_o_t): loaded and folded in-kernel — Rule 3,
            no host readback.
        """
        tidx, _, _ = cute.arch.thread_idx()
        q_tile_idx, batch_idx, head_idx = cute.arch.block_idx()
        if cutlass.const_expr(self.sched_policy != SCHED_NATURAL):
            _n_qh = cutlass.Int32((q.shape[2] // self.qh_per_kh if self.pack_gqa else q.shape[2]))
            _n_batch = cutlass.Int32(self.thd_batch if cutlass.const_expr(self.thd_varlen) else q.shape[0])
            # Host-computed (see __call__): the grid is sized from the same
            # value, so the decode cannot disagree with the launch geometry.
            _q_tiles = n_q_tiles
            if cutlass.const_expr(self.sched_policy == SCHED_LPT_L2):
                q_tile_idx, head_idx, batch_idx = lpt_l2_tile_coords(
                    q_tile_idx,
                    _n_qh,
                    _n_batch,
                    _q_tiles,
                    _n_qh // cutlass.Int32(k.shape[2]),
                    cutlass.Int32(k.shape[1]),
                    (self.head_tile_qk + self.head_tile_v) * self.in_dtype.width // 8,
                    _SCHED_L2_BUDGET_BYTES,
                )
            else:
                q_tile_idx, head_idx, batch_idx = lpt_tile_coords(q_tile_idx, _n_qh, _n_batch, _q_tiles)
        elif cutlass.const_expr(self.is_causal):
            # Causal work grows with the Q tile. Launch long tiles first to
            # avoid leaving a few expensive CTAs in the final scheduler waves.
            grid_q, _, _ = cute.arch.grid_dim()
            q_tile_idx = grid_q - q_tile_idx - 1
        q_seq_idx = q_tile_idx * (self.q_tile // self.qh_per_kh if self.pack_gqa else self.q_tile)

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
        # Envelope: actual head dims narrower than the compile-time tiles —
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
            o_head_off = batch_idx * o_batch_stride + q_head_base * o_head_stride
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
        has_kv_work = num_kv_tiles > 0 and (num_kv_tiles - 1) >= min_kv_tile

        # Shared-memory layout (BYTE-sized: KV elements are 1 byte, O staging
        # elements are 2 — an element-counted max() would under-allocate the
        # sO alias):
        #   sK: one kv_tile x head_tile_qk K tile (e4m3 bytes)
        #   sV: one kv_tile x head_tile_v V tile (e4m3 bytes)
        # The epilogue later aliases sKV as the q_tile x head_tile_v sO
        # staging tile after compute warps finish consuming the final K/V tile.
        k_tile_bytes = self.k_tile_elems * self.in_dtype.bytes
        v_tile_bytes = self.v_tile_elems * self.in_dtype.bytes
        o_stage_bytes = self.o_tile_elems * self.out_dtype.bytes
        sKV = cutlass.Array(
            k.dtype,
            max(k_tile_bytes + v_tile_bytes, o_stage_bytes),
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        sK = sKV
        sV = sKV.subview(k_tile_bytes)
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
                    is_v=False,
                    envelope=k_envelope,
                )
                self.load_one_kv_tile(
                    sV,
                    tma_v_desc,
                    v_tma_mbar,
                    tma_batch_idx,
                    kv_head_idx,
                    kv_row_base + kv_seq_idx,
                    is_v=True,
                    envelope=v_envelope,
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
                        is_v=False,
                        envelope=k_envelope,
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
                        is_v=True,
                        envelope=v_envelope,
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
            )
            mma_params = SimpleNamespace(
                sK=sK,
                sV=sV,
                o_regs=o_regs,
            )
            # Device-scale fold (Rule 3): 1-element loads, same address across
            # the CTA -> L2 broadcast; folded exactly like the old host path.
            _dsc_q = cutlass.Float32(cutlass.make_array_view(descale_q_t)[0])
            _dsc_k = cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
            _dsc_v = cutlass.Float32(cutlass.make_array_view(descale_v_t)[0])
            _scl_o = cutlass.Float32(cutlass.make_array_view(scale_o_t)[0])
            softmax_scale_log2 = softmax_scale_log2 * _dsc_q * _dsc_k
            # The trailing 2^-P_CAST_LOG2_SCALE cancels the P-cast bias the O
            # accumulator picked up through BMM2 (row_sum is de-scaled
            # separately at finalize).
            o_scale_fused = o_scale_fused * _dsc_v * _scl_o * cutlass.Float32(2.0**-P_CAST_LOG2_SCALE)
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
                mask_steps = ceil_div(self.q_tile // self.qh_per_kh if self.pack_gqa else self.q_tile, self.kv_tile)
                if cutlass.const_expr(self.diag_shifted):
                    # A translated diagonal (bottom-right anchoring or a right
                    # band) can straddle one additional KV tile; the frontier
                    # width itself is R-independent -- the band only
                    # translates the diagonal.
                    mask_steps = ceil_div((self.q_tile // self.qh_per_kh if self.pack_gqa else self.q_tile) + self.kv_tile - 1, self.kv_tile)
            elif cutlass.const_expr(not self.seq_kv_lens_present and not self.thd_varlen and k.shape[1] % self.kv_tile == 0):
                mask_steps = 0
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
                # row_sum carries the P-cast 2^4 — take it out with the EXACT
                # 2^-4 so every finalize path below sees the true sum.
                row_sum[row_half] = row_sum[row_half] * cutlass.Float32(2.0**-P_CAST_LOG2_SCALE)
                row_max_nat = row_max[row_half] * softmax_scale_log2 * LN2
                if cutlass.const_expr(self.has_sink):
                    sinks_arr = cutlass.make_array_view(sinks)
                    _sink_head = (
                        q_head_base if cutlass.const_expr(not self.pack_gqa) else q_head_base + (q_warp_row0 + (lane // 4) + row_half * 8) % self.qh_per_kh
                    )
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
                if lane % 4 == 0:
                    lse_arr = cutlass.make_array_view(lse)
                    for row_half in cutlass.range_constexpr(2):
                        _lse_row_in_cta = q_warp_row0 + (lane // 4) + row_half * 8
                        lse_q_idx = q_seq_idx + (_lse_row_in_cta if cutlass.const_expr(not self.pack_gqa) else _lse_row_in_cta // self.qh_per_kh)
                        _lse_head = q_head_base if cutlass.const_expr(not self.pack_gqa) else q_head_base + _lse_row_in_cta % self.qh_per_kh
                        lse_out = cutlass.Float32(row_lse[row_half])
                        if cutlass.const_expr(self.thd_varlen):
                            # Packed ragged-Stats LSE, written directly in the
                            # caller's declared layout: token-major (T, H) or
                            # head-major (H, head_stride). Rows past this
                            # sequence's Q length belong to the NEXT sequence —
                            # never written, and there is no padded region to
                            # trim.
                            if lse_q_idx < seqlen_q:
                                if cutlass.const_expr(self.thd_lse_head_major):
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
                                lse_arr[batch_idx, _lse_head, lse_q_idx] = lse_out

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
            # sO is a BYTE array (it aliases 1-byte KV storage); each
            # (warp, d_frag_pair) block stages 16 x 16 Float16 = 512 bytes,
            # one 16-byte stmatrix row per lane. Row validity for Amax_O
            # mirrors the store-time trim: rows at/past seqlen_q store zeros
            # and must not contribute.
            o_block_bytes = 16 * 16 * self.out_dtype.bytes
            row_valid = cutlass.Array(cutlass.Float32, 2, alignment=8)
            for row_half in cutlass.range_constexpr(2):
                _amax_row_in_cta = q_warp_row0 + (lane // 4) + row_half * 8
                amax_q_idx = q_seq_idx + (_amax_row_in_cta if cutlass.const_expr(not self.pack_gqa) else _amax_row_in_cta // self.qh_per_kh)
                row_valid[row_half] = cutlass.Float32(1.0) if amax_q_idx < seqlen_q else cutlass.Float32(0.0)
            lane_amax_half = cutlass.Array(cutlass.Float32, 4, alignment=16)
            for i in cutlass.range_constexpr(4):
                lane_amax_half[i] = 0.0
            for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                o_off = (d_frag_pair * 2) * 4
                o_scaled = fmul2(o_regs[o_off:8], row_sum_inv_vec)
                for i in cutlass.range_constexpr(8):
                    # row_sum_inv_vec order: halves alternate 0,0,1,1,0,0,1,1.
                    half = (i // 2) % 2
                    acc = half * 2 + (i % 2)
                    lane_amax_half[acc] = cute.arch.fmax(lane_amax_half[acc], cute.math.abs(o_scaled[i]))
                if cutlass.const_expr(self.out_dtype.bytes == 1):
                    for frag in cutlass.range_constexpr(2):
                        for row_half in cutlass.range_constexpr(2):
                            e = frag * 4 + row_half * 2
                            o_pair = fp32_to_fp8x2(o_scaled[e], o_scaled[e + 1], dtype=self.out_dtype)
                            _pair_row_in_cta = q_warp_row0 + (lane // 4) + row_half * 8
                            pair_q_seq_idx = q_seq_idx + (_pair_row_in_cta if cutlass.const_expr(not self.pack_gqa) else _pair_row_in_cta // self.qh_per_kh)
                            pair_head_off = o_head_off
                            if cutlass.const_expr(self.pack_gqa and self.qh_per_kh != 1):
                                pair_head_off = pair_head_off + (_pair_row_in_cta % self.qh_per_kh) * o_head_stride
                            pair_col_in_cta = d_frag_pair * 16 + frag * 8 + (lane % 4) * 2
                            if cutlass.const_expr(self.thd_varlen):
                                if pair_q_seq_idx < seqlen_q and pair_col_in_cta < head_dim_v:
                                    gO_pair = o_ptr + pair_head_off + pair_q_seq_idx * o_seq_stride + pair_col_in_cta
                                    gO_pair.store(cutlass.Vector.from_elements((o_pair,), cutlass.Uint16).bitcast(self.out_dtype), alignment=2)
                            else:
                                if pair_q_seq_idx < q.shape[1] and pair_col_in_cta < head_dim_v:
                                    if pair_q_seq_idx >= seqlen_q:
                                        o_pair = cutlass.Uint16(0)
                                    gO_pair = o_ptr + pair_head_off + pair_q_seq_idx * o_seq_stride + pair_col_in_cta
                                    gO_pair.store(cutlass.Vector.from_elements((o_pair,), cutlass.Uint16).bitcast(self.out_dtype), alignment=2)
                else:
                    o_packed = o_scaled.to(self.out_dtype).bitcast(cutlass.Int32)
                    sO_ptr = sO.data_ptr() + (compute_warp_idx * (self.pv_d_frags // 2) + d_frag_pair) * o_block_bytes + lane * 16
                    prims.stmatrix(
                        sO_ptr,
                        o_packed,
                        prims.MMALayout.ROW,
                    )
            lane_amax_o = cute.arch.fmax(
                cute.arch.fmax(lane_amax_half[0], lane_amax_half[1]) * row_valid[0],
                cute.arch.fmax(lane_amax_half[2], lane_amax_half[3]) * row_valid[1],
            )
            amax_o_arr = cutlass.make_array_view(amax_o)
            prims.atomicrmw(
                prims.AtomicOp.MAX,
                amax_o_arr,
                lane_amax_o.bitcast(cutlass.Int32),
            )

            if cutlass.const_expr(self.out_dtype.bytes != 1):
                store_row = lane_mod8 + ((lane_div8) % 2) * 8
                store_col = lane_div16 * 8
                _store_row_in_cta = q_warp_row0 + store_row
                store_q_seq_idx = q_seq_idx + (_store_row_in_cta if cutlass.const_expr(not self.pack_gqa) else _store_row_in_cta // self.qh_per_kh)
                store_head_off = o_head_off
                if cutlass.const_expr(self.pack_gqa and self.qh_per_kh != 1):
                    store_head_off = store_head_off + (_store_row_in_cta % self.qh_per_kh) * o_head_stride
                for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                    store_col_in_cta = d_frag_pair * 16 + store_col
                    if cutlass.const_expr(self.thd_varlen):
                        # Packed storage: rows past this sequence's Q length are
                        # the NEXT sequence's tokens — no store, and never the
                        # dense path's zero-fill.
                        if store_q_seq_idx < seqlen_q and store_col_in_cta < head_dim_v:
                            gO_ptr = o_ptr + store_head_off + store_q_seq_idx * o_seq_stride + store_col_in_cta
                            sO_ptr = sO.data_ptr() + (compute_warp_idx * (self.pv_d_frags // 2) + d_frag_pair) * o_block_bytes + lane * 16
                            gO_ptr.store(sO_ptr.load(count=16, alignment=16).bitcast(self.out_dtype), alignment=16)
                    else:
                        if store_q_seq_idx < q.shape[1] and store_col_in_cta < head_dim_v:
                            gO_ptr = o_ptr + store_head_off + store_q_seq_idx * o_seq_stride + store_col_in_cta
                            if store_q_seq_idx < seqlen_q:
                                sO_ptr = sO.data_ptr() + (compute_warp_idx * (self.pv_d_frags // 2) + d_frag_pair) * o_block_bytes + lane * 16
                                gO_ptr.store(sO_ptr.load(count=16, alignment=16).bitcast(self.out_dtype), alignment=16)
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
        stream: cuda_driver.CUstream,
    ) -> None:
        """Launch the SM120 per-tensor FP8 FMHA kernel.

        :param q: Query tensor with shape ``(B, Sq, H, D)`` (fp8).
        :param k: Key tensor with shape ``(B, Sk, H, D)`` (fp8).
        :param v: Value tensor with shape ``(B, Sk, H, D)`` (fp8).
        :param o: Output tensor with shape ``(B, Sq, H, D)`` (``out_dtype``).
        :param lse: ``(B, H, Sq)`` fp32 log-sum-exp output, or ``None`` to
            compile the LSE store out entirely (no dummy buffer needed).
        :param sinks: ``(H,)`` fp32 per-Q-head sink logits; must be ``None``
            exactly when the kernel is configured without ``has_sink``.
        :param seq_q_lens: Per-batch query lengths, or an unused dummy tensor.
        :param seq_kv_lens: Per-batch key/value lengths, or an unused dummy tensor.
        :param amax_o: 1-element Int32 amax buffer (pre-zeroed; dummy ok).
        :param softmax_scale_log2: ``softmax_scale * log2(e)`` (bases only —
            the kernel folds the device scale tensors in).
        :param o_scale_fused: base ``1.0`` (see above). descale_s/scale_s are
            unsupported on this cell (P is cast unscaled, like SM100).
        :param descale_q_t: 1-element fp32 DEVICE scale tensors (with
            descale_k_t / descale_v_t / scale_o_t) — Rule 3, no host readback.
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
        :param stream: CUDA stream used for the launch.
        """
        head_dim_qk = q.shape[3]
        head_dim_v = v.shape[3]
        if cutlass.const_expr(head_dim_qk != k.shape[3] or round_up_head_tile(head_dim_qk) != self.head_tile_qk):
            raise ValueError("runtime Q/K head dimensions must round up to the kernel head_tile_qk")
        if cutlass.const_expr(head_dim_v != o.shape[3] or round_up_head_tile(head_dim_v) != self.head_tile_v):
            raise ValueError("runtime V/O head dimensions must round up to the kernel head_tile_v")
        if cutlass.const_expr(head_dim_qk % 16 != 0 or head_dim_v % 16 != 0):
            raise ValueError("head dims must be multiples of 16 (TMA 16-byte global-stride rule at 1 byte/elem)")

        # THD compiles the token extents DYNAMIC (mode 1 is a symbol, not an
        # int), so only statically-known modes can be compared at trace time;
        # the adapter builds the ragged views from shared totals, so the
        # dynamic seq extents match by construction.
        def _static_neq(a, b):
            return isinstance(a, int) and isinstance(b, int) and a != b

        if cutlass.const_expr(
            _static_neq(q.shape[0], k.shape[0])
            or any(_static_neq(a, b) for a, b in zip(k.shape[:3], v.shape[:3]))
            or _static_neq(q.shape[0], o.shape[0])
            or _static_neq(q.shape[1], o.shape[1])
            or _static_neq(q.shape[2], o.shape[2])
            or q.shape[2] % k.shape[2] != 0
            or (isinstance(q.shape[2], int) and isinstance(k.shape[2], int) and q.shape[2] != k.shape[2] * self.qh_per_kh)
        ):
            raise ValueError("runtime Q/K/V/O batch, sequence, or head geometry mismatch")
        for name, tensor in (("Q", q), ("K", k), ("V", v), ("O", o)):
            if cutlass.const_expr(not self.is_layout_supported(tensor.shape, tensor.stride)):
                raise ValueError(f"{name} must use compact BSHD storage")
        if cutlass.const_expr(lse is not None):
            if cutlass.const_expr(self.thd_varlen):
                if cutlass.const_expr(self.thd_lse_head_major):
                    # Packed head-major (H, head_stride): the head stride is
                    # the caller's token capacity and may exceed the packed
                    # total, so only the head extent is pinned.
                    if cutlass.const_expr(len(lse.shape) != 2 or lse.shape[0] != q.shape[2]):
                        raise ValueError("head-major THD LSE must have shape (H, head_stride)")
                    # head_stride >= T is validated by the adapter at execute:
                    # the packed total (q.shape[1]) is DYNAMIC under THD.
                    if cutlass.const_expr(lse.stride != (lse.shape[1], 1)):
                        raise ValueError("THD LSE must be head-major with unit token stride")
                else:
                    # Token-major (T, H): T is the DYNAMIC packed total (the
                    # adapter binds it to the same symbol as Q's), so only the
                    # static head extent and strides are trace-checkable.
                    if cutlass.const_expr(len(lse.shape) != 2 or lse.shape[1] != q.shape[2]):
                        raise ValueError("THD LSE must have shape (T, H)")
                    if cutlass.const_expr(lse.stride != (q.shape[2], 1)):
                        raise ValueError("THD LSE must be compact token-major")
            else:
                if cutlass.const_expr(lse.shape != (q.shape[0], q.shape[2], q.shape[1])):
                    raise ValueError("LSE must have shape (B, H, Sq)")
        if cutlass.const_expr(self.has_sink != (sinks is not None)):
            raise ValueError("sinks must be provided exactly when the kernel is configured with has_sink")
        if cutlass.const_expr(sinks is not None and sinks.shape != (q.shape[2],)):
            raise ValueError("sinks must have shape (H,)")
        if cutlass.const_expr(self.thd_varlen):
            if cutlass.const_expr(q.shape[0] != 1):
                raise ValueError("THD Q/K/V/O must be packed batch-1 views")
            if cutlass.const_expr(seq_kv_lens.shape != (3 * self.thd_batch + 2,)):
                raise ValueError("THD seq_kv_lens must be the (3*B+2,) metadata tensor")

        # Exact head dims: split D into I contiguous C-element chunks while
        # preserving the compact (B, S, H, D) global-memory address
        # calculation; TMA order (C, S, I, H, B) linearizes the SMEM
        # destination as [I][kv_tile][C] in ONE rank-5 copy. Envelope head
        # dims: a rank-4 descriptor keeps the ACTUAL head extent innermost so
        # per-chunk copies zero-fill columns past it (see load_one_kv_tile).
        def kv_tma_desc(t, head_dim, swizzle, swizzle_chunks, swizzle_chunk_elems, envelope):
            if cutlass.const_expr(not envelope):
                layout = cute.make_layout(
                    (t.shape[0], t.shape[2], swizzle_chunks, t.shape[1], swizzle_chunk_elems),
                    stride=(t.shape[1] * t.shape[2] * head_dim, head_dim, swizzle_chunk_elems, t.shape[2] * head_dim, 1),
                )
                box = (1, 1, swizzle_chunks, self.kv_tile, swizzle_chunk_elems)
                stride_order = (4, 3, 2, 1, 0)
            else:
                layout = cute.make_layout(
                    (t.shape[0], t.shape[2], t.shape[1], head_dim),
                    stride=(t.shape[1] * t.shape[2] * head_dim, head_dim, t.shape[2] * head_dim, 1),
                )
                box = (1, 1, self.kv_tile, swizzle_chunk_elems)
                stride_order = (3, 2, 1, 0)
            return cuda.create_tensor_map_tiled_from_view(
                cute.make_tensor(t.iterator, layout),
                box_dims=box,
                stride_order=stride_order,
                swizzle=swizzle,
            )

        tma_k_desc = kv_tma_desc(k, head_dim_qk, self.k_tma_swizzle, self.k_tma_swizzle_chunks, self.k_swizzle_chunk_elems, head_dim_qk != self.head_tile_qk)
        tma_v_desc = kv_tma_desc(v, head_dim_v, self.v_tma_swizzle, self.v_tma_swizzle_chunks, self.v_swizzle_chunk_elems, head_dim_v != self.head_tile_v)
        if cutlass.const_expr(self.thd_varlen):
            # Build the [kv|cu_q|cu_k] metadata buffer DEVICE-side from the
            # caller's length tensors (no host cumsum, no H2D — issue #552);
            # the main kernel launched after it on this stream reads it.
            _build_thd_meta_kernel(
                seq_kv_lens,
                thd_q_lens,
                thd_kv_lens,
                thd_lens_form,
                cutlass.Int32(self.thd_batch),
            ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
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
        # LPT / LPT_L2 flatten the 3-D grid so the decode can order the whole tile
        # set globally (heaviest causal rows first); NATURAL keeps the zero-overhead
        # 3-D grid. The kernel receives n_q_tiles so its decode uses the same value.
        if cutlass.const_expr(self.sched_policy != SCHED_NATURAL):
            grid = (n_q_tiles * n_batch * n_head, 1, 1)
        else:
            grid = (n_q_tiles, n_batch, n_head)
        self.kernel(
            q,
            k,
            v,
            o,
            lse,
            sinks,
            seq_q_lens,
            seq_kv_lens,
            amax_o,
            tma_k_desc,
            tma_v_desc,
            softmax_scale_log2,
            o_scale_fused,
            cutlass.Int32(n_q_tiles),
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
    lse_stride: Optional[tuple[int, int, int]] = None,
) -> Callable:
    """Compile and cache one architecture-specific compact BSHD shape.

    ``d_qk`` is the Q/K head dim (QK^T contraction width) and ``d_v`` the V/O
    head dim (P@V output width); they are independent, e.g. (192, 128).

    THD specializations pack the batch: ``b`` is the real sequence count and
    ``sq``/``skv`` are IGNORED — the packed token totals are runtime values
    (they change every step under continuous batching), so the token extents
    compile DYNAMIC (``cute.sym_int``) and the cache key stays plan-time-only;
    callers must not pass them. ``max_sq`` (the longest sequence's Q length,
    which sizes the per-sequence grid) is likewise a RUNTIME ``__call__``
    argument, not a compile parameter.

    ``has_lse=False`` compiles the LSE store out (the kernel specializes on a
    ``None`` LSE argument) — callers that don't want stats pass no LSE buffer
    at all instead of a dummy. Dense ``lse_stride`` carries the caller's
    declared ``(B, H, Sq)`` element strides into the compiled tensor.

    THD LSE is token-major ``(T, H)`` by default; ``lse_head_major=True``
    switches to head-major ``(H, lse_head_stride)`` (FlashAttention's
    ``softmax_lse`` layout), where ``lse_head_stride`` is the caller-declared
    head-row stride (``>= T``, a shape — part of the cache key).
    """

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
        thd_varlen=PARAMS.thd_varlen,
        thd_lse_head_major=lse_head_major,
        thd_batch=b,
        head_tile_qk=round_up_head_tile(d_qk),
        head_tile_v=round_up_head_tile(d_v),
        q_tile=PARAMS.q_tile,
        kv_tile=PARAMS.kv_tile,
        pack_gqa=PARAMS.pack_gqa,
        qh_per_kh=qh // kh,
    )
    if has_lse and lse_stride is not None and PARAMS.thd_varlen:
        raise ValueError("dense LSE strides are not valid for THD")
    fake_batch = 1 if PARAMS.thd_varlen else b
    if PARAMS.thd_varlen:
        # Dynamic packed token totals: one symbol per ragged group (Q/O share
        # t_q; K/V share t_kv), so a new total re-binds the same compiled
        # artifact instead of minting a new one (issue #552).
        sq = cute.sym_int(divisibility=1)
        skv = cute.sym_int(divisibility=1)
    fake_q = cute.runtime.make_fake_compact_tensor(
        IN_DTYPE,
        (fake_batch, sq, qh, d_qk),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_k = cute.runtime.make_fake_compact_tensor(
        IN_DTYPE,
        (fake_batch, skv, kh, d_qk),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_v = cute.runtime.make_fake_compact_tensor(
        IN_DTYPE,
        (fake_batch, skv, kh, d_v),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_o = cute.runtime.make_fake_compact_tensor(
        OUT_DTYPE,
        (fake_batch, sq, qh, d_v),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_lse_shape = ((qh, lse_head_stride) if lse_head_major else (sq, qh)) if PARAMS.thd_varlen else (fake_batch, qh, sq)
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
                stride_order=(1, 0) if PARAMS.thd_varlen else (2, 1, 0),
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
        (3 * b + 2,) if PARAMS.thd_varlen else (b,),  # THD: [ seq_kv(B) | cu_q(B+1) | cu_k(B+1) ]
        stride_order=(0,),
        assumed_align=4,
    )
    # Amax buffers are Int32 at the ABI (bitcast-fp32 atomic max targets);
    # the adapter passes torch fp32 buffers as .view(torch.int32).
    fake_amax_o = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        stride_order=(0,),
        assumed_align=4,
    )
    # THD: the caller's Q/KV length tensors, consumed by the setup kernel's
    # device-side metadata build. DYNAMIC extents — (B,) per-batch lengths and
    # (B+1,) cu prefix sums bind the same artifact; the form rides the runtime
    # thd_lens_form bitmask, so no compile key grows.
    if PARAMS.thd_varlen:
        fake_thd_q_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_lens_form = cutlass.Int32(0)
    else:
        fake_thd_q_lens = None
        fake_thd_kv_lens = None
        fake_thd_lens_form = None

    def _fake_scale():
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (1,),
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
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
