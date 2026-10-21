# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
FP16/BF16 SDPA forward DECODE tile for the NVIDIA Blackwell SM120 family
(SM120 and SM121): ``mma.sync.aligned.m16n8k16`` tensor cores fed by a
multi-stage TMA K/V ring.

The prefill templates (``sm120/prefill_f16.py`` and its D=256 sibling) give
every compute warp 16 query rows of its own and stream one K and one V tile
through a single shared-memory buffer. On a decode-shaped launch -- the
``S_q x G`` packed rows of one KV head are a handful -- most of those warps
have no live row yet still run every MMA and ``ldmatrix``, and the single
buffer exposes the DRAM latency of each tile.

This tile holds a unit's packed rows in ``q_tile`` (16 or 32) rows and puts the
warps on the KEY axis instead:

- one producer warp streams ``kv_stage``-key K/V stages through a TMA ring
  (one full and one empty mbarrier per slot), so several stages of every CTA
  are in flight while its consumers compute;
- four consumer warps form ``q_tile // 16`` M-groups x ``kv_groups`` key
  groups. Warp ``(m, g)`` owns rows ``16m .. 16m + 15`` and every stage ``s``
  with ``s % kv_groups == g``, with its own online-softmax state and fp32 O
  accumulator. Group ``g`` owns ``slots_per_group`` ring slots of its own:
  every slot then has one waiter consuming its uses in order, which is what an
  mbarrier parity wait needs (it cannot tell a phase from the one two later);
- after the loop the key groups of an M-group merge their (max, sum, O)
  through the drained ring, and the group's first warp runs the epilogue (sink
  fold, LSE, normalized O store straight from its registers).

PackGQA packs ``q_tile // G`` tokens x ``G`` query heads of one KV head per
tile with no divisibility rule: rows past ``tokens x G`` are dead (zero Q,
never stored), so G = 6 at q_tile 16 runs two tokens in 12 live rows. KV split
cuts a unit's stage range into contiguous chunks (grid y = batch + split * B)
whose partial O / LSE the shared ``sm100/split_combine`` pass reduces; a split
writes sink-free partials and the combine folds the sink once.

Constraints:
- FP16 / BF16 Q/K/V/O, O at the input dtype
- head dims multiples of 8 up to 256; the kernel compiles at head tiles rounded
  up to 16 and TMA zero-fills the pad columns (the head-dim ENVELOPE)
- dense BSHD storage with the head dim innermost (no THD), SCHED_NATURAL
- top-left / bottom-right causal, right-band widening, left sliding window,
  per-batch lengths, per-Q-head sinks, natural-log or log2 LSE
"""

from cudnn.frost.compiled_cache import template_key as _template_key
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute

from cutlass.experimental import primitives as prims
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16
from cudnn.frost.tile_dsl.mma import mma_m16n8k16_f32
from cudnn.frost.tile_dsl.swizzle import swizzle_xor
from cudnn.sdpa.fwd.kernels.sm120._common import (
    ceil_div,
    nvvm_threadquad_reduction_max,
    nvvm_threadquad_reduction_sum,
    pack_to_i32,
)
from cudnn.sdpa.fwd.config_sm120 import (
    DECODE_CONSUMER_WARPS,
    DECODE_Q_TILES,
    DECODE_REGION_ALIGN,
    HEAD_TILE_GRANULE,
    GENERAL_HEAD_TILE_MAX,
    TemplateParams,
    decode_kv_groups,
    decode_kv_stage,
    decode_merge_bytes,
    decode_region_bytes,
    decode_slots_per_group,
    decode_stage_bytes,
    head_tile,
    register_budgets,
    validate_params,
)

from cudnn.sdpa.fwd.kernels.sm120.prepared_host import host as _host, compile_host as _compile_prepared_host  # noqa: F401  (_host: the prepared binder's entry)

# The FROST loader injects one immutable specialization before executing this
# module. A direct import uses the dense FP16 defaults on the 16-row tile.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(q_tile=16, kv_tile=16))
validate_params(PARAMS, decode_tile=True)

# The decode tile masks every stage against the true KV length (the K-tail
# columns read TMA zero-fill), independently of padding flags.
PREPARED_KV_TAIL_NATIVE = True

STORAGE_DTYPE = {DTYPE_FP16: cutlass.Float16, DTYPE_BF16: cutlass.BFloat16}[PARAMS.dtype_qkv]


def round_up_head_tile(d: int) -> int:
    """Round a head dim up to the head-tile granule — the ENVELOPE head tile."""
    return head_tile(d, HEAD_TILE_GRANULE)


fmul2 = partial(prims.mul_packed_f32x2, ftz=False, rnd=prims.FPRoundingMode.RN)
fma2 = partial(prims.fma_packed_f32x2, ftz=False, rnd=prims.FPRoundingMode.RN)


class SM120DecodeForward:
    """Configure and launch the SM120/SM121 FP16/BF16 SDPA decode tile."""

    MMA_TILER = (16, 8, 16)  # mma.sync.aligned.m16n8k16

    def __init__(
        self,
        in_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        is_causal: bool = False,
        bottom_right: bool = False,
        window_size_left: int | None = None,
        window_size_right: int | None = None,
        seq_q_lens_present: bool = False,
        seq_kv_lens_present: bool = False,
        has_sink: bool = False,
        split_kv: int = 1,
        head_tile_qk: int = 128,
        head_tile_v: int = 128,
        q_tile: int = DECODE_Q_TILES[0],
        pack_gqa: bool = False,
        qh_per_kh: int = 1,
        stats_log2: bool = False,
        negate_scores: bool = False,
    ):
        """Initialize the decode tile configuration.

        :param in_dtype: Q/K/V/O element type (Float16 or BFloat16).
        :param is_causal: Apply an upper causal bound to QK.
        :param bottom_right: Shift the causal diagonal by ``Skv - Sq``.
        :param window_size_left: Inclusive left-window offset, or ``None``.
        :param window_size_right: Widen the causal diagonal right by this many
            columns (0 = plain causal); only meaningful with ``is_causal``.
        :param seq_q_lens_present: Read per-batch query lengths at runtime.
        :param seq_kv_lens_present: Read per-batch key/value lengths at runtime.
        :param has_sink: Per-Q-head sink logits from the ``sinks`` tensor: folded
            into the softmax denominator unsplit, left to the combine split.
        :param split_kv: KV chunks per unit (1 = off); the partial O / LSE slot
            of split ``s`` of batch ``b`` is batch ``s * B + b``.
        :param head_tile_qk: Q/K head TILE, a multiple of 16 up to 256.
        :param head_tile_v: V/O head TILE, a multiple of 16 up to 256.
        :param q_tile: Packed Q rows per CTA, 16 or 32.
        :param pack_gqa: Pack ``q_tile // qh_per_kh`` tokens x ``qh_per_kh``
            query heads sharing one KV head per tile, token-major (row ``r`` is
            token ``r // G``, head ``r % G``).
        :param qh_per_kh: The graph's GQA ratio ``h_q // h_kv``.
        :param stats_log2: Write the (unsplit) LSE scaled by log2(e).
        :param negate_scores: attn_scale < 0: negate Q as it loads and run at
            |attn_scale| (#1435).
        """
        if q_tile not in DECODE_Q_TILES:
            raise ValueError(f"q_tile ({q_tile}) must be one of {DECODE_Q_TILES}")
        if qh_per_kh < 1:
            raise ValueError(f"qh_per_kh ({qh_per_kh}) must be >= 1")
        if pack_gqa and qh_per_kh > q_tile:
            raise ValueError(f"PackGQA needs qh_per_kh <= q_tile; got qh_per_kh ({qh_per_kh}) > q_tile ({q_tile})")
        for name, tile in (("head_tile_qk", head_tile_qk), ("head_tile_v", head_tile_v)):
            if tile % HEAD_TILE_GRANULE != 0 or not 0 < tile <= GENERAL_HEAD_TILE_MAX:
                raise ValueError(f"{name} ({tile}) must be a multiple of {HEAD_TILE_GRANULE} up to {GENERAL_HEAD_TILE_MAX}")
        self.in_dtype = in_dtype
        self.out_dtype = in_dtype
        self.is_causal = is_causal
        self.bottom_right = bottom_right
        self.window_size_left = window_size_left
        self.window_right = window_size_right if window_size_right is not None else 0
        self.seq_q_lens_present = seq_q_lens_present
        self.seq_kv_lens_present = seq_kv_lens_present
        self.has_sink = has_sink
        self.split_kv = split_kv
        self.stats_log2 = stats_log2
        self.negate_scores = negate_scores
        # The prepared host builds THD views and LSE layouts from these; the
        # decode tile is dense-only.
        self.thd_varlen = False
        self.thd_lse_padded = False
        self.thd_lse_head_major = False

        self.head_tile_qk = head_tile_qk
        self.head_tile_v = head_tile_v
        self.q_tile = q_tile
        self.pack_gqa = pack_gqa
        self.qh_per_kh = qh_per_kh
        # Rows of a packed tile past tokens_per_tile x G are dead.
        self.group = qh_per_kh if pack_gqa else 1
        self.tokens_per_tile = q_tile // self.group
        self.live_rows = self.tokens_per_tile * self.group

        self.m_groups = q_tile // 16
        self.kv_groups = decode_kv_groups(q_tile)
        self.kv_stage = decode_kv_stage(head_tile_qk, head_tile_v)
        # Every key group owns its slots: stage t belongs to group t % kv_groups
        # and to that group's slot (t // kv_groups) % slots_per_group, so each
        # slot's one waiter consumes its uses in order.
        self.slots_per_group = decode_slots_per_group(head_tile_qk, head_tile_v, q_tile)
        if self.slots_per_group == 0:
            raise ValueError(f"the decode tile cannot ring K/V stages at head tiles ({head_tile_qk}, {head_tile_v}) and q_tile {q_tile}")
        self.num_stages = self.kv_groups * self.slots_per_group

        # Warp roles: consumers fill the first warpgroup; the producer and three
        # idle warps fill the second, so setmaxregister moves whole warpgroups.
        self.compute_warp_ids = tuple(range(DECODE_CONSUMER_WARPS))
        self.load_warp_id = DECODE_CONSUMER_WARPS
        self.num_warps = 8
        self.load_regs, self.compute_regs = register_budgets(64)
        self.threads_per_cta = cute.arch.WARP_SIZE * self.num_warps
        self.threads_compute = cute.arch.WARP_SIZE * DECODE_CONSUMER_WARPS
        self.bar_consumers = 1

        self._setup_attributes()

    def _setup_attributes(self):
        """Derive MMA fragment counts, the stage layout, and TMA geometry."""
        self.qk_k_frags = self.kv_stage // self.MMA_TILER[1]
        self.qk_d_frags = self.head_tile_qk // self.MMA_TILER[2]
        self.pv_v_frags = self.kv_stage // self.MMA_TILER[2]
        self.pv_d_frags = self.head_tile_v // self.MMA_TILER[1]

        # One stage = a K region then a V region, each starting on the period
        # of the widest swizzle (config_sm120.decode_stage_bytes).
        stage_bytes = decode_stage_bytes(self.head_tile_qk, self.head_tile_v)
        self.stage_elems = stage_bytes // 2
        self.k_region_elems = decode_region_bytes(self.kv_stage, self.head_tile_qk) // 2
        self.ring_elems = self.num_stages * self.stage_elems
        # Merge staging, per spilling warp: the O accumulator plus (max, sum) of
        # the lane's two rows, one fp32 word per lane per slot (lane-minor).
        self.merge_words = self.pv_d_frags * 4 + 4
        if decode_merge_bytes(self.head_tile_v, self.q_tile) > stage_bytes * self.num_stages:
            raise ValueError("the decode merge staging does not fit the ring")

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
    def load_kv_region(
        self,
        s_dst: cute.Pointer,
        tma_desc: cutlass.GridConstant[cuda.TensorMap],
        mbar: cute.Pointer,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        seq_coord: cutlass.Int32,
        is_v: cutlass.Constexpr[bool],
        envelope: cutlass.Constexpr[bool],
    ) -> None:
        """Issue the TMA copies of one stage's K or V region (elected lane).

        Exact head dims issue ONE rank-5 copy whose descriptor pre-splits the
        head dim into swizzle-span chunks. Envelope head dims (actual d < tile)
        issue one copy per chunk over a rank-4 descriptor that keeps the ACTUAL
        head extent innermost: TMA bounds-checks each descriptor dimension on
        its own, so it zero-fills the pad columns only when the head boundary is
        one dimension, not split across the chunk dimensions. The caller arms
        the stage's full barrier with both regions' bytes first.
        """
        chunks = self.v_tma_swizzle_chunks if is_v else self.k_tma_swizzle_chunks
        chunk_elems = self.v_swizzle_chunk_elems if is_v else self.k_swizzle_chunk_elems
        if cutlass.const_expr(envelope):
            for i in cutlass.range_constexpr(chunks):
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_dst + i * self.kv_stage * chunk_elems,
                    tma_desc.get_ptr(),
                    (i * chunk_elems, seq_coord, head_idx, batch_idx),
                    mbar,
                )
        else:
            prims.cp_async_bulk_tensor_shared_cta_global(
                s_dst,
                tma_desc.get_ptr(),
                (0, seq_coord, 0, head_idx, batch_idx),
                mbar,
            )

    @cute.jit
    def load_q_tile(self, bp: SimpleNamespace) -> cutlass.Array:
        """Load the warp's 16 packed Q rows from GMEM into MMA A registers.

        Dead rows (past ``tokens_per_tile x G``), rows past the batch's query
        length, and pad columns load zero.
        """
        q_regs = cutlass.Array(cutlass.Int32, self.qk_d_frags * 4, alignment=16)
        row0 = bp.lane // 4
        col0 = (bp.lane % 4) * 2
        q_off = 0
        for d_frag in cutlass.range_constexpr(self.qk_d_frags):
            offsets = (
                (row0, 0),
                (row0 + 8, 0),
                (row0, 8),
                (row0 + 8, 8),
            )
            for i in cutlass.range_constexpr(4):
                row_in_cta = bp.q_warp_row0 + offsets[i][0]
                col = d_frag * self.MMA_TILER[2] + col0 + offsets[i][1]
                token = bp.q_seq_idx + row_in_cta // self.group
                row_off = token * bp.q_seq_stride
                if cutlass.const_expr(self.group != 1):
                    row_off = row_off + (row_in_cta % self.group) * bp.q_head_stride
                q_packed = cutlass.Int32(0)
                if row_in_cta < self.live_rows and token < bp.seqlen_q and col < bp.head_dim_qk:
                    q_pair = (bp.q_ptr + bp.q_head_off + row_off + col).load(count=2, alignment=4)
                    q_packed = q_pair.bitcast(cutlass.Int32)[0]
                q_regs[q_off + i] = q_packed
            q_off += 4
        if cutlass.const_expr(self.negate_scores):
            for i in cutlass.range_constexpr(self.qk_d_frags * 4):
                q_regs[i] = q_regs[i] ^ cutlass.Int32(-2147450880)  # 0x80008000: every element's sign bit
        return q_regs

    @cute.jit
    def mma_qk(self, bp: SimpleNamespace, sK: cute.Pointer, q_regs: cutlass.Array) -> cutlass.Array:
        """``S = Q @ K.T`` for one stage: 16 rows x ``kv_stage`` keys."""
        s_regs = cutlass.Array(cutlass.Float32, self.qk_k_frags * 4, alignment=16)
        for i in cutlass.range_constexpr(self.qk_k_frags * 4):
            s_regs[i] = cutlass.Float32(0.0)
        k_row_in_frag_pair = bp.lane_div16 * 8 + bp.lane_mod8
        k_col_in_frag_pair = (bp.lane_div8 % 2) * 8
        for k_frag_pair in cutlass.range_constexpr(self.qk_k_frags // 2):
            for d_frag in cutlass.range_constexpr(self.qk_d_frags):
                k_row = k_frag_pair * 16 + k_row_in_frag_pair
                k_col = d_frag * 16 + k_col_in_frag_pair
                k_chunk = k_col // self.k_swizzle_chunk_elems
                k_col_in_chunk = k_col % self.k_swizzle_chunk_elems
                k_physical_row = k_chunk * self.kv_stage + k_row
                k_ptr = (
                    sK
                    + k_physical_row * self.k_swizzle_chunk_elems
                    + swizzle_xor(k_physical_row, k_col_in_chunk, self.k_swizzle_chunk_elems, self.in_dtype.bytes)
                )
                k_vec = prims.ldmatrix(k_ptr, 4, prims.MMALayout.ROW)
                q_off = d_frag * 4
                s_off = (k_frag_pair * 2) * 4
                s_regs[s_off:4] = mma_m16n8k16_f32(
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
                s_regs[s_off + 4 : 4] = mma_m16n8k16_f32(
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
    def online_softmax(self, bp: SimpleNamespace, st: SimpleNamespace, s_regs: cutlass.Array, kv_seq_idx: cutlass.Int32) -> cutlass.Array:
        """Mask one stage's scores, update the running (max, sum), rescale O,
        and stage packed P in the consumed score slots.

        Every stage is masked: the per-row bounds (causal / band, left window,
        KV length) cost a compare per score, small next to the stage's MMAs, and
        no stage can then read the tail columns' zero-fill as live keys.
        """
        lane = bp.lane
        o_regs = st.o_regs
        row_max = st.row_max
        row_sum = st.row_sum
        scale_log2 = st.softmax_scale_log2
        for row_half in cutlass.range_constexpr(2):
            lo = row_half * 2
            hi = row_half * 2 + 1
            row_in_cta = bp.q_warp_row0 + (lane // 4) + row_half * 8
            diagonal_position = bp.q_seq_idx + row_in_cta // self.group + bp.diagonal_offset
            valid_cols = bp.seqlen_k
            if cutlass.const_expr(self.is_causal):
                valid_cols = cute.math.max(cutlass.Int32(0), cute.math.min(diagonal_position + 1 + self.window_right, bp.seqlen_k))
            first_valid_col = cutlass.Int32(0)
            if cutlass.const_expr(self.window_size_left is not None):
                first_valid_col = cute.math.max(cutlass.Int32(0), diagonal_position - self.window_size_left)

            cur_max = cutlass.Float32(-cutlass.Float32.inf)
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                s0 = s_regs[s_off + lo]
                s1 = s_regs[s_off + hi]
                k_col0 = kv_seq_idx + k_frag * 8 + 2 * (lane % 4)
                k_col1 = k_col0 + 1
                if not (k_col0 >= first_valid_col and k_col0 < valid_cols):
                    s0 = -cutlass.Float32.inf
                if not (k_col1 >= first_valid_col and k_col1 < valid_cols):
                    s1 = -cutlass.Float32.inf
                s_regs[s_off + lo] = s0
                s_regs[s_off + hi] = s1
                cur_max = cute.arch.fmax(cur_max, cute.arch.fmax(s0, s1))
            cur_max = nvvm_threadquad_reduction_max(cur_max)

            row_max_prev = row_max[row_half]
            new_max = cute.arch.fmax(row_max_prev, cur_max)
            # A row with no visible key so far keeps max = -inf: its correction
            # is 1 (nothing to rescale) and its exponent base 0, so the masked
            # -inf scores give exact zeros instead of exp2(-inf - -inf) = NaN.
            old_scale = cutlass.Float32(1.0)
            exp_max = cutlass.Float32(0.0)
            if new_max > -cutlass.Float32.inf:
                old_scale = cute.math.exp2((row_max_prev - new_max) * scale_log2, fastmath=True)
                exp_max = new_max
            row_max[row_half] = new_max
            neg_exp_max_scaled = -(exp_max * scale_log2)
            tile_sum = cutlass.Float32(0.0)
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                in0, in1 = fma2(
                    (s_regs[s_off + lo], s_regs[s_off + hi]),
                    (scale_log2, scale_log2),
                    (neg_exp_max_scaled, neg_exp_max_scaled),
                )
                p0 = cute.math.exp2(in0, fastmath=True)
                p1 = cute.math.exp2(in1, fastmath=True)
                tile_sum = tile_sum + (p0 + p1)
                p_pack = pack_to_i32((self.in_dtype(p0), self.in_dtype(p1)), self.in_dtype)
                s_regs[k_frag * 4 + row_half * 2] = p_pack.bitcast(cutlass.Float32)
            tile_sum = nvvm_threadquad_reduction_sum(tile_sum)
            row_sum[row_half] = row_sum[row_half] * old_scale + tile_sum
            for d_frag in cutlass.range_constexpr(self.pv_d_frags):
                o_off = d_frag * 4 + row_half * 2
                o_regs[o_off + 0], o_regs[o_off + 1] = fmul2((o_regs[o_off + 0], o_regs[o_off + 1]), (old_scale, old_scale))
        return s_regs

    @cute.jit
    def v_keep_mask(self, key: cutlass.Int32, seqlen_k: cutlass.Int32) -> cutlass.Int32:
        """The AND mask of a V register holding keys ``key`` (low half) and
        ``key + 1`` (high half): it keeps the halves of keys below ``seqlen_k``."""
        keep = cutlass.Int32(-1)
        if key + 1 >= seqlen_k:
            keep = cutlass.Int32(0xFFFF)
        if key >= seqlen_k:
            keep = cutlass.Int32(0)
        return keep

    @cute.jit
    def mma_pv(self, bp: SimpleNamespace, sV: cute.Pointer, o_regs: cutlass.Array, p_regs: cutlass.Array, kv_seq_idx: cutlass.Int32) -> None:
        """``O += P @ V`` for one stage; V fragments via transposing ``ldmatrix``.

        With per-batch KV lengths, the keys past a batch's length are memory the
        TMA loads (it zero-fills only past the tensor): their P is 0, but
        ``0 x NaN = NaN``, so their V halves are zeroed before the MMA. The mask
        runs on every stage, not only the one that reaches past the length:
        branching each stage between a masked and an unmasked body costs more
        than the four ANDs per V fragment.
        """
        v_row_in_frag_pair = (bp.lane_div8 % 2) * 8 + bp.lane_mod8
        v_col_in_frag_pair = (bp.lane_div8 // 2) * 8
        for v_frag in cutlass.range_constexpr(self.pv_v_frags):
            p_off = (2 * v_frag) * 4
            p0 = p_regs[p_off + 0].bitcast(cutlass.Int32)
            p1 = p_regs[p_off + 2].bitcast(cutlass.Int32)
            p2 = p_regs[p_off + 4].bitcast(cutlass.Int32)
            p3 = p_regs[p_off + 6].bitcast(cutlass.Int32)
            if cutlass.const_expr(self.seq_kv_lens_present):
                # The B fragment's keys, as online_softmax numbers P's: the first register of each
                # pair holds keys key, key + 1; the second, key + 8, key + 9.
                key = kv_seq_idx + v_frag * 16 + 2 * (bp.lane % 4)
                keep_lo = self.v_keep_mask(key, bp.seqlen_k)
                keep_hi = self.v_keep_mask(key + 8, bp.seqlen_k)
            for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                v_row = v_frag * 16 + v_row_in_frag_pair
                v_col = d_frag_pair * 16 + v_col_in_frag_pair
                v_chunk = v_col // self.v_swizzle_chunk_elems
                v_col_in_chunk = v_col % self.v_swizzle_chunk_elems
                v_physical_row = v_chunk * self.kv_stage + v_row
                v_ptr = (
                    sV
                    + v_physical_row * self.v_swizzle_chunk_elems
                    + swizzle_xor(v_physical_row, v_col_in_chunk, self.v_swizzle_chunk_elems, self.in_dtype.bytes)
                )
                v_vec = prims.ldmatrix(v_ptr, 4, prims.MMALayout.COL)
                v0, v1, v2, v3 = v_vec[0], v_vec[1], v_vec[2], v_vec[3]
                if cutlass.const_expr(self.seq_kv_lens_present):
                    v0, v1, v2, v3 = v0 & keep_lo, v1 & keep_hi, v2 & keep_lo, v3 & keep_hi
                o_off = (d_frag_pair * 2) * 4
                o_regs[o_off:4] = mma_m16n8k16_f32(
                    p0,
                    p1,
                    p2,
                    p3,
                    v0,
                    v1,
                    o_regs[o_off + 0],
                    o_regs[o_off + 1],
                    o_regs[o_off + 2],
                    o_regs[o_off + 3],
                    self.in_dtype,
                )
                o_regs[o_off + 4 : 4] = mma_m16n8k16_f32(
                    p0,
                    p1,
                    p2,
                    p3,
                    v2,
                    v3,
                    o_regs[o_off + 4],
                    o_regs[o_off + 5],
                    o_regs[o_off + 6],
                    o_regs[o_off + 7],
                    self.in_dtype,
                )

    @cute.jit
    def spill_state(self, base: cute.Pointer, lane: cutlass.Int32, st: SimpleNamespace) -> None:
        """Write the lane's (max, sum, O) to its merge slot, 16 bytes per access
        (four fp32 words travel as eight half words: the ring is half-typed)."""
        words = self.merge_words
        for w in cutlass.range_constexpr(words // 4):
            vals = []
            for j in cutlass.range_constexpr(4):
                idx = w * 4 + j
                if cutlass.const_expr(idx < self.pv_d_frags * 4):
                    vals.append(st.o_regs[idx])
                elif cutlass.const_expr(idx - self.pv_d_frags * 4 < 2):
                    vals.append(st.row_max[idx - self.pv_d_frags * 4])
                else:
                    vals.append(st.row_sum[idx - self.pv_d_frags * 4 - 2])
            (base + (w * 32 + lane) * 8).store(cutlass.Vector.from_elements(tuple(vals), cutlass.Float32).bitcast(self.in_dtype), alignment=16)

    @cute.jit
    def merge_state(self, base: cute.Pointer, lane: cutlass.Int32, st: SimpleNamespace) -> None:
        """Fold another key group's spilled (max, sum, O) into the lane's own."""
        tail = (base + ((self.merge_words // 4 - 1) * 32 + lane) * 8).load(count=8, alignment=16).bitcast(cutlass.Float32)
        scale_log2 = st.softmax_scale_log2
        own_scale = cutlass.Array(cutlass.Float32, 2, alignment=8)
        other_scale = cutlass.Array(cutlass.Float32, 2, alignment=8)
        for row_half in cutlass.range_constexpr(2):
            m_own = st.row_max[row_half]
            m_other = tail[row_half]
            m_new = cute.arch.fmax(m_own, m_other)
            a = cutlass.Float32(0.0)
            b = cutlass.Float32(0.0)
            # A side that saw no visible key (max = -inf) contributes nothing; a
            # row dead on both sides keeps max = -inf and sums / O of zero.
            if m_own > -cutlass.Float32.inf:
                a = cute.math.exp2((m_own - m_new) * scale_log2, fastmath=True)
            if m_other > -cutlass.Float32.inf:
                b = cute.math.exp2((m_other - m_new) * scale_log2, fastmath=True)
            st.row_max[row_half] = m_new
            st.row_sum[row_half] = st.row_sum[row_half] * a + tail[2 + row_half] * b
            own_scale[row_half] = a
            other_scale[row_half] = b
        for w in cutlass.range_constexpr(self.pv_d_frags):
            vals = (base + (w * 32 + lane) * 8).load(count=8, alignment=16).bitcast(cutlass.Float32)
            for j in cutlass.range_constexpr(4):
                idx = w * 4 + j
                row_half = j // 2
                st.o_regs[idx] = st.o_regs[idx] * own_scale[row_half] + vals[j] * other_scale[row_half]

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
        """SM120 SDPA decode tile: one CTA per (Q tile, batch x split, packed head).

        :param q: Query tensor ``(B, Sq, Hq, D)``.
        :param k: Key tensor ``(B, Skv, Hkv, D)``.
        :param v: Value tensor ``(B, Skv, Hkv, Dv)``.
        :param o: Output ``(B * split_kv, Sq, Hq, Dv)`` (the split-major partial slab when split).
        :param lse: fp32 LSE ``(B * split_kv, Hq, Sq)``, or ``None`` to compile the store out.
        :param sinks: ``(Hq,)`` fp32 sink logits, ``None`` without ``has_sink``.
        :param seq_q_lens: Per-batch query lengths, or an unused dummy tensor.
        :param seq_kv_lens: Per-batch key/value lengths, or an unused dummy tensor.
        :param tma_k_desc: Tensor map descriptor for K.
        :param tma_v_desc: Tensor map descriptor for V.
        :param softmax_scale_log2: ``softmax_scale * log2(e)``.
        """
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % cute.arch.WARP_SIZE
        warp = cute.arch.warp_idx()

        ring = cutlass.Array(k.dtype, self.ring_elems, space=cutlass.AddressSpace.smem, alignment=DECODE_REGION_ALIGN)
        full_bar = cutlass.Array(cutlass.Int64, self.num_stages, space=cutlass.AddressSpace.smem, alignment=8)
        empty_bar = cutlass.Array(cutlass.Int64, self.num_stages, space=cutlass.AddressSpace.smem, alignment=8)
        ring_ptr = ring.data_ptr()
        full_ptr = full_bar.data_ptr()
        empty_ptr = empty_bar.data_ptr()

        if warp == self.load_warp_id:
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_k_desc.get_ptr())
                prims.prefetch_tensormap(tma_v_desc.get_ptr())
                for s in cutlass.range_constexpr(self.num_stages):
                    prims.mbarrier_init(full_ptr + s, 1)
                    # Every lane of every consumer warp of the stage's key group
                    # arrives, each after its own last ldmatrix of the stage.
                    prims.mbarrier_init(empty_ptr + s, self.m_groups * cute.arch.WARP_SIZE)
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)

        q_tile_idx, batch_split_idx, head_idx = cute.arch.block_idx()
        batch_idx = batch_split_idx
        split_idx = cutlass.Int32(0)
        o_batch_idx = batch_split_idx
        if cutlass.const_expr(self.split_kv > 1):
            n_batch = cutlass.Int32(q.shape[0])
            split_idx = batch_split_idx // n_batch
            batch_idx = batch_split_idx % n_batch

        seqlen_q = cutlass.Int32(q.shape[1])
        seqlen_k = cutlass.Int32(k.shape[1])
        if cutlass.const_expr(self.seq_q_lens_present):
            seqlen_q = cute.math.max(cutlass.Int32(0), cute.math.min(seq_q_lens[batch_idx], cutlass.Int32(q.shape[1])))
        if cutlass.const_expr(self.seq_kv_lens_present):
            seqlen_k = cute.math.max(cutlass.Int32(0), cute.math.min(seq_kv_lens[batch_idx], cutlass.Int32(k.shape[1])))

        num_heads_q = q.shape[2]
        num_heads_kv = k.shape[2]
        head_dim_qk = q.shape[3]
        head_dim_v = v.shape[3]
        k_envelope = head_dim_qk != self.head_tile_qk
        v_envelope = head_dim_v != self.head_tile_v
        q_batch_stride, q_seq_stride, q_head_stride, _ = q.stride
        o_batch_stride, o_seq_stride, o_head_stride, _ = o.stride
        q_head_base = head_idx * cutlass.Int32(self.group)
        kv_head_idx = q_head_base // (num_heads_q // num_heads_kv)
        q_seq_idx = q_tile_idx * cutlass.Int32(self.tokens_per_tile)

        # The tile's key range: the union of its live rows' visible columns.
        diagonal_offset = cutlass.Int32(0)
        if cutlass.const_expr(self.bottom_right):
            diagonal_offset = seqlen_k - seqlen_q
        last_token = cute.math.min(q_seq_idx + cutlass.Int32(self.tokens_per_tile), seqlen_q) - 1
        kv_end = seqlen_k
        if cutlass.const_expr(self.is_causal):
            kv_end = cute.math.max(cutlass.Int32(0), cute.math.min(last_token + diagonal_offset + 1 + self.window_right, seqlen_k))
        kv_begin = cutlass.Int32(0)
        if cutlass.const_expr(self.window_size_left is not None):
            kv_begin = cute.math.max(cutlass.Int32(0), q_seq_idx + diagonal_offset - self.window_size_left)
        stage_lo = kv_begin // self.kv_stage
        stage_hi = ceil_div(kv_end, self.kv_stage)
        if last_token < q_seq_idx:
            stage_hi = stage_lo
        if cutlass.const_expr(self.split_kv > 1):
            # Contiguous chunks of the stage range; the first `rem` splits take
            # one extra stage. An empty chunk leaves its rows' sums at zero, so
            # the epilogue writes O := 0 / LSE := -inf, the combine's identity.
            span = cute.math.max(cutlass.Int32(0), stage_hi - stage_lo)
            per = span // cutlass.Int32(self.split_kv)
            rem = span % cutlass.Int32(self.split_kv)
            lo = stage_lo + split_idx * per + cute.math.min(split_idx, rem)
            extra = cutlass.Int32(1) if split_idx < rem else cutlass.Int32(0)
            stage_lo = lo
            stage_hi = lo + per + extra
        n_stages = cute.math.max(cutlass.Int32(0), stage_hi - stage_lo)

        # /////////////////////////////////////////////////////////////////////
        #  PRODUCER: stream the unit's stages through the ring
        # /////////////////////////////////////////////////////////////////////
        if warp == self.load_warp_id:
            prims.setmaxregister(self.load_regs, prims.SetMaxRegisterAction.DECREASE)
            k_tx = tma_k_desc.global_tx_bytes() * (self.k_tma_swizzle_chunks if k_envelope else 1)
            v_tx = tma_v_desc.global_tx_bytes() * (self.v_tma_swizzle_chunks if v_envelope else 1)
            t = cutlass.Int32(0)
            while t < n_stages:
                use = t // cutlass.Int32(self.kv_groups)
                slot = (t % cutlass.Int32(self.kv_groups)) * cutlass.Int32(self.slots_per_group) + use % cutlass.Int32(self.slots_per_group)
                if use >= cutlass.Int32(self.slots_per_group):
                    # The slot's previous stage must be released by its group.
                    reuse_phase = ((use // cutlass.Int32(self.slots_per_group)) - 1) & cutlass.Int32(1)
                    while not prims.mbarrier_try_wait_parity(empty_ptr + slot, reuse_phase):
                        pass
                seq_coord = (stage_lo + t) * cutlass.Int32(self.kv_stage)
                s_k = ring_ptr + slot * cutlass.Int32(self.stage_elems)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(full_ptr + slot, k_tx + v_tx)
                    self.load_kv_region(s_k, tma_k_desc, full_ptr + slot, batch_idx, kv_head_idx, seq_coord, is_v=False, envelope=k_envelope)
                    self.load_kv_region(
                        s_k + cutlass.Int32(self.k_region_elems), tma_v_desc, full_ptr + slot, batch_idx, kv_head_idx, seq_coord, is_v=True, envelope=v_envelope
                    )
                t += 1

        # /////////////////////////////////////////////////////////////////////
        #  CONSUMERS: one M-group x one key group per warp
        # /////////////////////////////////////////////////////////////////////
        elif warp < self.load_warp_id:
            prims.setmaxregister(self.compute_regs, prims.SetMaxRegisterAction.INCREASE)
            m_group = warp % self.m_groups
            k_group = warp // self.m_groups
            q_warp_row0 = m_group * self.MMA_TILER[0]

            bp = SimpleNamespace(
                lane=lane,
                lane_div8=lane // 8,
                lane_mod8=lane % 8,
                lane_div16=lane // 16,
                q_warp_row0=q_warp_row0,
                q_seq_idx=q_seq_idx,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                diagonal_offset=diagonal_offset,
                head_dim_qk=head_dim_qk,
                q_ptr=q.iterator.raw_ptr(),
                q_head_off=batch_idx * q_batch_stride + q_head_base * q_head_stride,
                q_seq_stride=q_seq_stride,
                q_head_stride=q_head_stride,
            )
            row_max = cutlass.Array(cutlass.Float32, 2, alignment=8)
            row_sum = cutlass.Array(cutlass.Float32, 2, alignment=8)
            for i in cutlass.range_constexpr(2):
                row_max[i] = -cutlass.Float32.inf
                row_sum[i] = cutlass.Float32(0.0)
            o_regs = cutlass.Array(cutlass.Float32, self.pv_d_frags * 4, alignment=16)
            for i in cutlass.range_constexpr(self.pv_d_frags * 4):
                o_regs[i] = cutlass.Float32(0.0)
            st = SimpleNamespace(row_max=row_max, row_sum=row_sum, o_regs=o_regs, softmax_scale_log2=softmax_scale_log2)

            q_regs = self.load_q_tile(bp)

            t = cutlass.Int32(k_group)
            use = cutlass.Int32(0)
            while t < n_stages:
                slot = cutlass.Int32(k_group * self.slots_per_group) + use % cutlass.Int32(self.slots_per_group)
                phase = (use // cutlass.Int32(self.slots_per_group)) & cutlass.Int32(1)
                while not prims.mbarrier_try_wait_parity(full_ptr + slot, phase):
                    pass
                s_k = ring_ptr + slot * cutlass.Int32(self.stage_elems)
                s_regs = self.mma_qk(bp, s_k, q_regs)
                kv_seq_idx = (stage_lo + t) * cutlass.Int32(self.kv_stage)
                p_regs = self.online_softmax(bp, st, s_regs, kv_seq_idx)
                self.mma_pv(bp, s_k + cutlass.Int32(self.k_region_elems), o_regs, p_regs, kv_seq_idx)
                prims.mbarrier_arrive(empty_ptr + slot)
                t += cutlass.Int32(self.kv_groups)
                use += 1

            # Key-group merge through the drained ring: every stage was consumed
            # (so every TMA write landed) once all consumers pass this barrier.
            if cutlass.const_expr(self.kv_groups > 1):
                prims.barrier_cta_sync(self.bar_consumers, thread_count=self.threads_compute)
                if k_group > 0:
                    slot_idx = (k_group - 1) * self.m_groups + m_group
                    self.spill_state(ring_ptr + slot_idx * (self.merge_words * 32 * 2), lane, st)
                prims.barrier_cta_sync(self.bar_consumers, thread_count=self.threads_compute)
                if k_group == 0:
                    for g in cutlass.range_constexpr(1, self.kv_groups):
                        slot_idx = (g - 1) * self.m_groups + m_group
                        self.merge_state(ring_ptr + slot_idx * (self.merge_words * 32 * 2), lane, st)

            # /////////////////////////////////////////////////////////////////
            #  EPILOGUE (each M-group's first key group)
            # /////////////////////////////////////////////////////////////////
            if k_group == 0:
                # row_max holds the raw score max; the scale applies in the log2
                # domain and converts with ln(2). Unsplit, the per-head sink
                # logit joins the denominator as a virtual column with no V row;
                # a split leaves it to the combine.
                LN2 = cutlass.Float32(0.6931471805599453)
                row_sum_inv = cutlass.Array(cutlass.Float32, 2, alignment=8)
                row_lse = cutlass.Array(cutlass.Float32, 2, alignment=8)
                for row_half in cutlass.range_constexpr(2):
                    row_in_cta = q_warp_row0 + (lane // 4) + row_half * 8
                    row_max_nat = row_max[row_half] * softmax_scale_log2 * LN2
                    if cutlass.const_expr(self.has_sink and self.split_kv == 1):
                        sinks_arr = cutlass.make_array_view(sinks)
                        sink_logit = cutlass.Float32(sinks_arr[q_head_base + row_in_cta % self.group])
                        new_max = cute.arch.fmax(row_max_nat, sink_logit)
                        alpha = cute.math.exp(row_max_nat - new_max, fastmath=True)
                        new_sum = row_sum[row_half] * alpha + cute.math.exp(sink_logit - new_max, fastmath=True)
                        row_sum_inv[row_half] = alpha / new_sum
                        row_lse[row_half] = new_max + cute.math.log(new_sum, fastmath=True)
                    else:
                        inv = cutlass.Float32(0.0)
                        if row_sum[row_half] > 0.0:
                            inv = cute.math.rcp(row_sum[row_half], approx=True, ftz=True)
                        row_sum_inv[row_half] = inv
                        lse_val = row_max_nat + cute.math.log(cute.math.max(row_sum[row_half], cutlass.Float32(1e-30)), fastmath=True)
                        if row_sum[row_half] <= 0.0:
                            lse_val = -cutlass.Float32.inf
                        row_lse[row_half] = lse_val

                o_ptr = o.iterator.raw_ptr()
                o_base = o_batch_idx * o_batch_stride + q_head_base * o_head_stride
                for row_half in cutlass.range_constexpr(2):
                    row_in_cta = q_warp_row0 + (lane // 4) + row_half * 8
                    token = q_seq_idx + row_in_cta // self.group
                    head_in_group = row_in_cta % self.group
                    row_inv = cutlass.Float32(row_sum_inv[row_half])
                    if row_in_cta < self.live_rows and token < q.shape[1]:
                        live = token < seqlen_q
                        if cutlass.const_expr(lse is not None):
                            if lane % 4 == 0:
                                lse_out = cutlass.Float32(row_lse[row_half])
                                # Split partials stay natural-log; the combine owns the final base.
                                if cutlass.const_expr(self.stats_log2 and self.split_kv == 1):
                                    lse_out = lse_out * cutlass.Float32(1.4426950408889634)
                                if not live:
                                    lse_out = -cutlass.Float32.inf
                                lse_arr = cutlass.make_array_view(lse)
                                lse_arr[o_batch_idx, q_head_base + head_in_group, token] = lse_out
                        row_ptr = o_ptr + o_base + head_in_group * o_head_stride + token * o_seq_stride
                        if not live:
                            row_inv = cutlass.Float32(0.0)
                        for d_frag in cutlass.range_constexpr(self.pv_d_frags):
                            col = d_frag * 8 + (lane % 4) * 2
                            if col < head_dim_v:
                                o_off = d_frag * 4 + row_half * 2
                                o0, o1 = fmul2((o_regs[o_off + 0], o_regs[o_off + 1]), (row_inv, row_inv))
                                (row_ptr + col).store(cutlass.Vector.from_elements((o.dtype(o0), o.dtype(o1)), o.dtype), alignment=4)

        # The idle warps of the producer's warpgroup only hand back registers.
        else:
            prims.setmaxregister(self.load_regs, prims.SetMaxRegisterAction.DECREASE)

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
        softmax_scale_log2: cutlass.Float32,
        thd_max_sq: cutlass.Int32,
        thd_q_lens: Optional[cute.Tensor],
        thd_kv_lens: Optional[cute.Tensor],
        thd_lens_form: Optional[cutlass.Int32],
        thd_n_ctas: cutlass.Int32,
        stream: cuda_driver.CUstream,
        prepared: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Launch the decode tile; same ABI as the SM120 prefill templates (the
        THD arguments are unused: the decode tile is dense-only).

        :param q: Query tensor ``(B, Sq, Hq, D)``.
        :param k: Key tensor ``(B, Skv, Hkv, D)``.
        :param v: Value tensor ``(B, Skv, Hkv, Dv)``.
        :param o: Output ``(B * split_kv, Sq, Hq, Dv)``.
        :param lse: fp32 LSE ``(B * split_kv, Hq, Sq)``, or ``None``.
        :param sinks: ``(Hq,)`` fp32 sink logits, ``None`` without ``has_sink``.
        :param seq_q_lens: Per-batch query lengths, or an unused dummy tensor.
        :param seq_kv_lens: Per-batch key/value lengths, or an unused dummy tensor.
        :param softmax_scale_log2: ``softmax_scale * log2(e)``.
        :param stream: CUDA stream used for the launch.
        """
        head_dim_qk = q.shape[3]
        head_dim_v = v.shape[3]
        if cutlass.const_expr(head_dim_qk != k.shape[3] or round_up_head_tile(head_dim_qk) != self.head_tile_qk):
            raise ValueError("runtime Q/K head dimensions must round up (by the head-tile granule) to the kernel head_tile_qk")
        if cutlass.const_expr(head_dim_v != o.shape[3] or round_up_head_tile(head_dim_v) != self.head_tile_v):
            raise ValueError("runtime V/O head dimensions must round up (by the head-tile granule) to the kernel head_tile_v")
        if cutlass.const_expr(head_dim_qk % 8 != 0 or head_dim_v % 8 != 0):
            raise ValueError("head dimensions must be multiples of 8 (TMA 16-byte global-stride rule at 2 B/elem)")
        if cutlass.const_expr(self.has_sink != (sinks is not None)):
            raise ValueError("sinks must be provided exactly when the kernel is configured with has_sink")

        # K/V TMA descriptors over the tensors' own strides, not packed
        # recomputations, so a view with a wider token stride (K or V sliced
        # from an interleaved buffer) loads as declared: rank-5 with the head
        # dim pre-split into swizzle-span chunks when exact, rank-4 keeping the
        # actual head extent innermost (zero-filled pad columns) in the envelope.
        def kv_tma_desc(t, head_dim, head_tile, swizzle, swizzle_chunks, swizzle_chunk_elems):
            if cutlass.const_expr(head_dim == head_tile):
                layout = cute.make_layout(
                    (t.shape[0], t.shape[2], swizzle_chunks, t.shape[1], swizzle_chunk_elems),
                    stride=(t.stride[0], t.stride[2], swizzle_chunk_elems, t.stride[1], 1),
                )
                box = (1, 1, swizzle_chunks, self.kv_stage, swizzle_chunk_elems)
                stride_order = (4, 3, 2, 1, 0)
            else:
                layout = cute.make_layout(
                    (t.shape[0], t.shape[2], t.shape[1], head_dim),
                    stride=(t.stride[0], t.stride[2], t.stride[1], 1),
                )
                box = (1, 1, self.kv_stage, swizzle_chunk_elems)
                stride_order = (3, 2, 1, 0)
            return cuda.create_tensor_map_tiled_from_view(
                cute.make_tensor(t.iterator, layout),
                box_dims=box,
                stride_order=stride_order,
                swizzle=swizzle,
            )

        tma_k_desc = kv_tma_desc(k, head_dim_qk, self.head_tile_qk, self.k_tma_swizzle, self.k_tma_swizzle_chunks, self.k_swizzle_chunk_elems)
        tma_v_desc = kv_tma_desc(v, head_dim_v, self.head_tile_v, self.v_tma_swizzle, self.v_tma_swizzle_chunks, self.v_swizzle_chunk_elems)
        n_q_tiles = ceil_div(q.shape[1], self.tokens_per_tile)
        n_head = q.shape[2] // self.group
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
            grid=(n_q_tiles, q.shape[0] * self.split_kv, n_head),
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
    sq: int = 1,
    skv: int = 1,
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
    prepared: bool = True,
    persistent_ctas: int = 0,
) -> Callable:
    """Compile the dense pointer host of the decode tile; geometry binds at execution."""

    _cache_key = _template_key(globals(), locals(), "compile")
    if not prepared or any(st is not None for st in (q_stride, k_stride, v_stride, o_stride)):
        raise ValueError("the SM120 decode tile binds runtime strides through the prepared pointer entry only")
    if PARAMS.thd_varlen or lse_head_major or lse_padded_rows:
        raise ValueError("the SM120 decode tile is dense-only")
    kernel = SM120DecodeForward(
        in_dtype=STORAGE_DTYPE,
        is_causal=PARAMS.window_right is not None,
        bottom_right=PARAMS.bottom_right,
        window_size_left=PARAMS.window_left,
        window_size_right=PARAMS.window_right,
        seq_q_lens_present=PARAMS.seq_q_lens_present,
        seq_kv_lens_present=PARAMS.seq_kv_lens_present,
        has_sink=PARAMS.has_sink,
        split_kv=PARAMS.split_kv,
        head_tile_qk=round_up_head_tile(d_qk),
        head_tile_v=round_up_head_tile(d_v),
        q_tile=PARAMS.q_tile,
        pack_gqa=PARAMS.pack_gqa,
        qh_per_kh=qh // kh,
        stats_log2=PARAMS.stats_log2,
        negate_scores=PARAMS.negate_scores,
    )
    if kernel.kv_stage != PARAMS.kv_tile:
        raise ValueError(
            f"the decode tile's kv_tile is its stage width {kernel.kv_stage} at head tiles ({kernel.head_tile_qk}, {kernel.head_tile_v}); got {PARAMS.kv_tile}"
        )
    return _compile_prepared_host(kernel, STORAGE_DTYPE, qh, kh, d_qk, d_v, has_lse, persistent_ctas, _cache_key)
