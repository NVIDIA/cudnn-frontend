# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Per-tensor FP8 dense or paged decode with swap-AB MMA and a D128 physical tile.

The D64 flavor uses TMA zero fill for the physical tile's extra columns;
output stores retain the runtime logical widths. Short MTP queries span
token units sized by the live softmax columns, not padded MMA columns.

Keys occupy the 128-lane MMA M axis; at most eight live packed query rows
occupy a 16-column N tile. Four softmax warps process only the live half.
K and V alternate through five 16-KiB shared-memory slots, allowing the
TMA producer to overlap future loads with the current MMA/softmax work.
The remaining two warps issue MMA and TMA operations.

P uses one 32-byte swizzled row per key: eight FP8 probabilities, eight
explicit zeros in the same 16-byte store, and an upper half zeroed before
MMA starts. Probabilities are multiplied by 16 before FP8 conversion;
the epilogue divides the value scale by 16. Runtime Q/K descales are
applied before the online maximum, including negative scale values.

The existing FP8 prepared host supplies pointers and strides. Output
AMAX is computed before output scaling. Split-KV writes FP32 partials
and natural-log Stats for the existing combine, which applies the final
output scale. Empty and padded rows follow the public SDPA contract.
THD, MXFP8, block-scaled O and other query widths retain their existing
lowerings. This forward-only template adds no public API or knobs.
"""

from cudnn.frost.compiled_cache import template_key as _template_key
from cudnn.sdpa.fwd.kernels.sm100.d128_fp8_swap_ab import contract as _implementation_contract

TEMPLATE_NAME = _implementation_contract.TEMPLATE_NAME
from typing import Callable, NamedTuple, Optional, Tuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from cudnn.sdpa.fwd.config_fp8_decode_common import TemplateParamsDecode
from cudnn.sdpa.fwd.kernels.sm100.d128_fp8_swap_ab.config import make_cfg_d128_fp8_decode

# The loader injects this immutable, plan-time configuration before execution.
PARAMS: TemplateParamsDecode = globals().get(
    "FROST_TEMPLATE_PARAMS",
    TemplateParamsDecode(dtype_qkv=0, dtype_o=0, cta_mma=1, decode_q_tile=16, decode_live_columns=8, paged_kv=True, page_size=16, seq_kv_lens_present=True),
)
CFG, _TMA = make_cfg_d128_fp8_decode(PARAMS)
FP8 = True

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance, launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.consume import consume_then_arrive, release_zero
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.tma import tma_load_tile
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.mask import MASK_CAUSAL, MASK_PADDED, MASK_SWA, compute_kv_loop_bounds
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16, fp32_to_fp8_pack
from cudnn.sdpa.fwd.kernels._common_blackwell import row_max_for_exp2
from cudnn.sdpa.fwd.kernels._quantized import _initialize_split_amax

STORAGE_DTYPE = cutlass.Float8E4M3FN if CFG.DTYPE_QKV == 0 else cutlass.Float8E5M2
MMA_KIND = nvvm.Tcgen05MMAKind.F8F6F4
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_1
OUT_STORAGE_DTYPE = (cutlass.Float8E4M3FN, cutlass.Float8E5M2, cutlass.BFloat16, cutlass.Float16)[CFG.DTYPE_O]

N_Q = CFG.N_Q
TILE_N = CFG.TILE_N
TILE_K = CFG.TILE_K
TILE_O = CFG.TILE_O
BPE = CFG.BPE
STAGES = CFG.STAGES_KV
SPLIT_KV = CFG.SPLIT_KV

# === PackGQA === row r of the Q tile is token r // G, head r % G (the prefill
# tile's convention).  The Q TMA box covers Q_BOX_TOKENS tokens x G heads; a
# group that does not divide the live softmax width leaves tail rows, zero-filled
# once at kernel start (their S^T / O^T columns are computed and never stored).
HEADS_PER_TILE = CFG.QH_PER_KH if CFG.PACK_GQA else 1
LIVE_Q_COLUMNS = getattr(PARAMS, "decode_live_columns", 0) or N_Q
# The sixteen-column MMA has only eight computed softmax columns. Counting
# its padded columns as tokens would skip every second token unit under G8.
Q_BOX_TOKENS = LIVE_Q_COLUMNS // HEADS_PER_TILE
Q_BOX_ROWS = Q_BOX_TOKENS * HEADS_PER_TILE
if Q_BOX_TOKENS < 1:
    raise ValueError(f"decode_d128_fp8: the packed head group ({HEADS_PER_TILE}) does not fit the {LIVE_Q_COLUMNS} live Q columns")
# === Token units === a unit covers Q_BOX_TOKENS consecutive Q tokens of its head group; a graph with more tokens
# than one Q box holds is cut into ceil(S_q / Q_BOX_TOKENS) TOKEN UNITS per (head group, batch, split), each a CTA
# that streams the unit's KV range once (MTP: S_q = 4 at 24/2 = two units of two tokens on the 32-column tile).  The
# host entry derives the unit count from the runtime S_q and widens the grid's x axis by it; the kernel decodes
# (head group, token unit) from blockIdx.x and offsets every token coordinate (the Q box, the causal / window mask,
# the O / LSE rows, the padded-Q trim) by the unit's first token.  Declared here for the binders (prepared.py,
# the direct test driver): a module without the flag covers S_q x HEADS_PER_TILE <= N_Q only.
Q_TOKEN_UNITS: bool = True

# === Paged KV (same contract as prefill_d256_f16.py) ===
PAGED_KV = bool(CFG.PAGED_KV)
PAGE_SIZE = CFG.PAGE_SIZE if PAGED_KV else 0
KV_BOX_ROWS = min(PAGE_SIZE, TILE_N) if PAGED_KV else TILE_N
KV_BOXES = TILE_N // KV_BOX_ROWS

# TMA granule: one 128 B swizzle atom of the head dim (128 FP8 elements).
GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
QK_ITERS = _TMA.QK_ITERS
VO_ITERS = _TMA.VO_ITERS
if _TMA.VO_GRANU_ELEMS != GRANU_ELEMS:
    raise ValueError("decode_d128_fp8: Q/K and V/O must share the 128 B swizzle granule")

qBufferElems = N_Q * TILE_K
kvBufferElems = TILE_N * TILE_K
# P^T uses a 32-byte row, including its zero-padded MMA columns.
pBufferWords = TILE_N * CFG.P_SWZ_BYTES // 4

Q_TX_BYTES = Q_BOX_ROWS * TILE_K * BPE
KV_TX_BYTES = kvBufferElems * BPE

_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
_SWZ_BITS = {128: 3, 64: 2, 32: 1}
SMEM_LAYOUT_QK = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_P = _SWZ_ENUM[CFG.P_SWZ_BYTES]
# The generic-proxy P^T stores apply the same XOR pattern the UMMA descriptor
# decodes (Swizzle<B, 4, 3>: 16 B chunk index ^= 128 B row-group index).
_P_SMEM_SWIZZLE = cutlass.Swizzle(_SWZ_BITS[CFG.P_SWZ_BYTES], 4, 3)
P_ROW_WORDS = CFG.P_SWZ_BYTES // 4

# K-major operands (K as A of BMM1, Q^T as B of BMM1): SBO = 8 rows x 128 B.
STRIDE_BYTE_OFFSET_QK = 8 * CFG.K_SWZ_BYTES
# V^T as the MN-major A of BMM2: 128 d columns occupy one 128-byte atom.
# LBO uses TILE_N rows x 128 B; SBO = 8 key rows.
LEADING_BYTE_OFFSET_VT = TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_VT = 8 * CFG.V_SWZ_BYTES
# P^T as the MN-major B of BMM2: one padded 32-byte atom, SBO = 8 key rows.
STRIDE_BYTE_OFFSET_P = 8 * CFG.P_SWZ_BYTES

# TMEM: two S^T slots followed by the O^T region, N_Q FP32 columns per block.
S_ACC_OFF = (0, N_Q)
O_OFF = (2 * N_Q, 3 * N_Q)
TMEM_COLS = 32
while TMEM_COLS < 4 * N_Q:
    TMEM_COLS *= 2
O_BLOCKS = TILE_O // 128

# Four softmax warps read the 128 TMEM lanes (warp_id % 4). They process
# eight live query columns within the 16-column MMA tile; the unused P
# columns are zero-padded.
COLS = getattr(PARAMS, "decode_live_columns", 0) or 16
COL_GROUPS = N_Q // 16
if COLS not in (8, 16) or (COLS == 8 and N_Q != 16):
    raise ValueError("swap-AB decode: a reduced softmax width needs a single 16-column MMA group")
# This group stores eight FP8 values and explicitly pads the 16-byte atom.
P_GROUP_WORDS = COLS * BPE // 4
P_CHUNKS_PER_GROUP = (COLS * BPE + 15) // 16
SOFTMAX_WARPS = CFG.SOFTMAX_WARPS
SOFTMAX_LANES = SOFTMAX_WARPS * 32
if N_Q % COLS != 0 or SOFTMAX_WARPS != 4 * COL_GROUPS:
    raise ValueError(f"decode_d128_fp8: N_Q={N_Q} needs {4 * COL_GROUPS} softmax warps (4 per 16 columns); config says {SOFTMAX_WARPS}")
MMA_WARP_ID = CFG.MMA_WARP_ID
TMALDG_WARP_ID = CFG.TMALDG_WARP_ID
# Named barriers: 1 = TMEM base hand-off (softmax warps + MMA warp), 2 = the
# softmax warps' own cross-warp reductions (all groups, they run in lockstep).
_BAR_TMEM = 1
_BAR_SOFTMAX = 2
# Cross-warp reduction scratch.  The per-KV-tile column MAXIMA meet in ONE 16-word Int32 slot per group: each of the
# group's four warps publishes its 16 butterfly maxima with red.shared.max on order-preserving keys (_max_key), and
# after the tile's named barrier every lane reads the 16 group maxima back -- no fold, nothing retained across the
# exchange.  MAX_RED_SLOTS slots rotate per tile and the group's first warp re-arms the slot two tiles ahead, so the
# one barrier per tile also orders every reuse.  The epilogue's column SUMS keep the per-warp fp32 form ([softmax
# warp][COLS], each warp's partial read back by its group).  (Holding the 16 maxima in registers and folding 64
# loaded values was what put the 18-warp N_Q = 64 tile over ptxas's 96-register cap.)
MAX_RED_SLOTS = 3
MAX_RED_WORDS = MAX_RED_SLOTS * COL_GROUPS * COLS  # <= SOFTMAX_LANES by construction (48 per group against 128 lanes), so one lane arms one word
RED_WORDS = SOFTMAX_WARPS * COLS
INT32_MIN = -2147483648  # below every max key: the slot's armed value

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


class DecodeBars(NamedTuple):
    mb_q_full: object
    mb_kv_full: object
    mb_kv_empty: object
    mb_s_full: object
    mb_s_empty: object
    mb_p_full: object
    mb_bmm2_done: object
    mb_tmem_dealloc: object


def make_decode_bars() -> DecodeBars:
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return DecodeBars(
        mb_q_full=MBarrier(_alloc(1), stages=1, init_count=1, producer=Producer.TMA_LOAD),
        mb_kv_full=MBarrier(_alloc(STAGES), stages=STAGES, init_count=1, producer=Producer.TMA_LOAD),
        mb_kv_empty=MBarrier(_alloc(STAGES), stages=STAGES, init_count=1, producer=Producer.MMA_COMMIT),
        mb_s_full=MBarrier(_alloc(2), stages=2, init_count=1, producer=Producer.MMA_COMMIT),
        mb_s_empty=MBarrier(_alloc(2), stages=2, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_p_full=MBarrier(_alloc(2), stages=2, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_bmm2_done=MBarrier(_alloc(2), stages=2, init_count=1, producer=Producer.MMA_COMMIT),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
    )


@cute.jit
def _select_i32(cond, a, b):
    return cutlass.Int32(arith.select(cond.ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()))


@cute.jit
def _select_f32(cond, a, b):
    return cutlass.Float32(arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))


@cute.jit
def _max_key(x):
    """Order-preserving Int32 key of a non-NaN fp32 (the softmax maxima: finite or -inf): the bit pattern, with the
    magnitude bits inverted for a negative value, so signed-integer order is the float order (-inf -> 0x807FFFFF, the
    smallest key a column can hold, above INT32_MIN).  Its own inverse on the keys it produces (_key_max)."""
    b = x.bitcast(cutlass.Int32)
    return _select_i32(b < cutlass.Int32(0), b ^ cutlass.Int32(0x7FFFFFFF), b)


@cute.jit
def _key_max(k):
    """The fp32 a _max_key key encodes."""
    return _select_i32(k < cutlass.Int32(0), k ^ cutlass.Int32(0x7FFFFFFF), k).bitcast(cutlass.Float32)


@cute.jit
def _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv, tok0):
    """This unit's KV-tile range ``[lo, hi)`` plus the per-batch lengths.

    Every role computes it identically (pure function of the same inputs), so
    the barrier handshakes stay in lockstep.  The masks narrow the range the
    way the prefill tile does (compute_kv_loop_bounds over the unit's token
    span, which starts at ``tok0`` = the token unit's first Q token), the dense
    padded-Q trim collapses a dead batch to an empty range, and the split
    chunking mirrors _common_blackwell's _split_chunk.
    """
    eff_seqlen_kv = seqlen_kv
    if cutlass.const_expr(CFG.SEQ_KV_LENS_PRESENT == 1):
        eff_seqlen_kv = cutlass.Int32(cutlass.make_array_view(seq_kv_lens_tensor)[batch_idx])
    q_len_b = seqlen_q
    if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT == 1):
        _sq_arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
        q_len_b = cutlass.Int32(_sq_arr[batch_idx])
    eff_seqlen_q = seqlen_q
    if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT == 1 and CFG.BOTTOM_RIGHT == 1):
        eff_seqlen_q = cute.math.max(cutlass.Int32(0), cute.math.min(q_len_b, seqlen_q))
    bounds = compute_kv_loop_bounds(
        tok0,
        eff_seqlen_q,
        eff_seqlen_kv,
        CFG.WINDOW_LEFT,
        CFG.MASK_FLAGS,
        TILE_N,
        Q_BOX_TOKENS,
        bottom_right=bool(CFG.BOTTOM_RIGHT),
        window_right=int(CFG.WINDOW_RIGHT),
    )
    left = bounds.left
    right = cute.math.max(bounds.right, left)
    if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT == 1):
        right = _select_i32(q_len_b <= cutlass.Int32(0), left, right)
    n_tiles = right - left
    per = n_tiles // cutlass.Int32(SPLIT_KV)
    rem = n_tiles % cutlass.Int32(SPLIT_KV)
    lo = left + split_idx * per + cute.math.min(split_idx, rem)
    hi = lo + per + _select_i32(split_idx < rem, 1, 0)
    return lo, hi, eff_seqlen_kv, eff_seqlen_q, q_len_b


@cute.jit
def _paged_load_tile(
    smem_tile,
    tma,
    block_table_tensor,
    batch_idx,
    kv_head_idx,
    n_pages_b,
    kv_tile,
    mbar,
    box_rows: cutlass.Constexpr[int],
    n_boxes: cutlass.Constexpr[int],
    granu_elems: cutlass.Constexpr[int],
):
    """One paged K or V tile as ``n_boxes`` row boxes through the block table
    (same contract as prefill_d256_f16._paged_load_tile: boxes past the batch's
    live pages take page -1 = TMA-OOB zeros, bytes still credited)."""
    bt = cutlass.make_array_view(block_table_tensor)
    last_live = cute.math.max(n_pages_b - cutlass.Int32(1), cutlass.Int32(0))
    for j in cutlass.range_constexpr(n_boxes):
        g = kv_tile * cutlass.Int32(TILE_N) + cutlass.Int32(j * box_rows)
        slot = g // cutlass.Int32(PAGE_SIZE)
        row_in_page = g % cutlass.Int32(PAGE_SIZE)
        page_live = cutlass.Int32(bt[batch_idx, cute.math.min(slot, last_live)])
        in_range = slot < n_pages_b
        page = _select_i32(in_range, page_live, cutlass.Int32(-1))
        tma_load_tile(
            smem_tile.shifted(j * box_rows * granu_elems),
            tma(cutlass.Int32(0), kv_head_idx, row_in_page, page),
            mbar,
            cta_group=1,
        )


@cute.jit
def _load_kv(smem_tile, tma, block_table_tensor, batch_idx, kv_head_idx, n_pages_b, kv_tile, mbar):
    """One K or V tile (TILE_N keys x TILE_K) into ``smem_tile``: dense TMA box or
    the paged block-table walk."""
    if cutlass.const_expr(PAGED_KV):
        _paged_load_tile(smem_tile, tma, block_table_tensor, batch_idx, kv_head_idx, n_pages_b, kv_tile, mbar, KV_BOX_ROWS, KV_BOXES, GRANU_ELEMS)
    else:
        tma_load_tile(smem_tile, tma(cutlass.Int32(0), kv_head_idx, kv_tile * cutlass.Int32(TILE_N), batch_idx), mbar, cta_group=1)


@cute.jit
def _kv_slot(seq, n_tiles):
    """Ring slot and full-barrier parity of the ``seq``-th load of the unit's
    K(0), K(1), V(0), K(2), V(1), ... sequence (V(n-1) is the 2n-1-th load: no
    K(n) precedes it)."""
    slot = seq % cutlass.Int32(STAGES)
    phase = (seq // cutlass.Int32(STAGES)) & cutlass.Int32(1)
    return slot, phase


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    sQ,
    sKV,
    bars,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    head_idx,
    batch_idx,
    split_idx,
    tok0,
    block_table_tensor=None,
    block_table_v_tensor=None,
    paged_hnd: cutlass.Constexpr[bool] = False,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(PAGED_KV and paged_hnd):
        # HND page pools: descriptor dims are (D, row, H_kv, page); every load
        # site keeps the (d, head, row, page) vocabulary, the swap lives here.
        _tk, _tv = GmemTileTma(tma_k_desc), GmemTileTma(tma_v_desc)
        tma_k = lambda d, h, r, p: _tk(d, r, h, p)  # noqa: E731
        tma_v = lambda d, h, r, p: _tv(d, r, h, p)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)

    if cutlass.const_expr(use_pdl):
        # Programmatic dependent launch: every global read of this role -- the lengths and the table (a serving stack's own
        # earlier kernels write them), the Q box (the slab a projection / norm-RoPE launch wrote), the K / V pages (the
        # write-through filled them) -- sits BELOW this wait; the descriptor prefetches above it are the prologue it overlaps.
        wait_on_dependent_grids()
        # The prologue is behind this CTA: release the dependent grid so it stages its own while the mainloop runs (its wait
        # still blocks on this grid's completion and memory flush, wherever this trigger sits).
        launch_dependent_grids()

    lo, hi, eff_seqlen_kv, _eq, _ql = _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv, tok0)
    n_tiles = hi - lo
    q_head_base = head_idx * cutlass.Int32(HEADS_PER_TILE)
    kv_head_idx = head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // cutlass.Int32(CFG.QH_PER_KH)
    n_pages_b = cutlass.Int32(0)
    if cutlass.const_expr(PAGED_KV):
        n_pages_b = (eff_seqlen_kv + cutlass.Int32(PAGE_SIZE - 1)) // cutlass.Int32(PAGE_SIZE)

    if n_tiles > cutlass.Int32(0):
        bars.mb_q_full.arrive(n_bytes=Q_TX_BYTES, pred=nvvm.elect_sync())
        tma_load_tile(sQ[0], tma_q(cutlass.Int32(0), q_head_base, tok0, batch_idx), bars.mb_q_full.smem_ptr, cta_group=1)

        # Ring order K(0), K(1), V(0), K(2), V(1), ...: K runs one tile ahead of V, so
        # K(t+2) is issued when V(t-1)'s slot frees (end of iteration t-1) and lands
        # a whole iteration before BMM1(t+2) needs it, while V(t+1) is issued when
        # K(t+1)'s slot frees (start of iteration t).  The MMA warp consumes in the
        # same order, so one 3-slot ring serves both operands (see _kv_slot).
        kv_state = PipelineState.start(phase=1)
        bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase)
        bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES, pred=nvvm.elect_sync())
        _load_kv(sKV[kv_state.idx], tma_k, block_table_tensor, batch_idx, kv_head_idx, n_pages_b, lo, bars.mb_kv_full[kv_state.idx].smem_ptr)
        kv_state = advance(kv_state, STAGES)
        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            kv_tile = lo + i
            if i + cutlass.Int32(1) < n_tiles:
                bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase)
                bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES, pred=nvvm.elect_sync())
                _load_kv(
                    sKV[kv_state.idx],
                    tma_k,
                    block_table_tensor,
                    batch_idx,
                    kv_head_idx,
                    n_pages_b,
                    kv_tile + cutlass.Int32(1),
                    bars.mb_kv_full[kv_state.idx].smem_ptr,
                )
                kv_state = advance(kv_state, STAGES)
            bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase)
            bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES, pred=nvvm.elect_sync())
            _load_kv(sKV[kv_state.idx], tma_v, block_table_v_tensor, batch_idx, kv_head_idx, n_pages_b, kv_tile, bars.mb_kv_full[kv_state.idx].smem_ptr)
            kv_state = advance(kv_state, STAGES)


@cute.jit
def _mma_warp_group(
    sQ,
    sK,
    sVt,
    sP,
    tmem_ptr_i32,
    bars,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    batch_idx,
    split_idx,
    tok0,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    tmem_alloc(tmem_ptr_i32, TMEM_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(_BAR_TMEM, 32 * (SOFTMAX_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=N_Q,
        m_dim=TILE_N,
        k_dim=0,
        a_negate=int(PARAMS.negate_scores),
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=N_Q,
        m_dim=128,
        a_major=1,
        b_major=1,
        k_dim=0,
    )
    # BMM1: S^T = K . Q^T -- K [128 keys x TILE_K] K-major, Q^T [N_Q x TILE_K] K-major.
    bmm1_desc = MmaDesc(
        M=TILE_N,
        N=N_Q,
        K=TILE_K,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=CFG.TILE_K_HW,
        btranspose=False,
        cta_group=1,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    # BMM2: O^T = V^T . P^T -- V^T [128 d x 128 keys] MN-major (d contiguous),
    # P^T [N_Q x 128 keys] MN-major (the N_Q values of one key contiguous).
    bmm2_desc = MmaDesc(
        M=128,
        N=N_Q,
        K=TILE_N,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=CFG.TILE_K_HW,
        atranspose=True,
        btranspose=True,
        cta_group=1,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    if cutlass.const_expr(use_pdl):
        # PDL: the TMEM allocation and the descriptor builds above are this role's prologue; its first global read (the lengths) is below.
        wait_on_dependent_grids()
    lo, hi, _ek, _eq, _ql = _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv, tok0)
    n_tiles = hi - lo

    # Barrier bookkeeping is arithmetic in the tile index (no loop-carried
    # pipeline state inside a data-dependent branch):
    #   ring load seq (K(0), K(1), V(0), K(2), V(1), ...): slot seq % STAGES, phase (seq // STAGES) & 1;
    #   S slot t % 2: the softmax releases it once per tile that used it, so
    #   BMM1(t) waits completion #(t // 2) of s_empty[t % 2], i.e. parity
    #   ((t // 2) - 1) & 1 -- which is 1 for t < 2 and passes a fresh barrier;
    #   p_full[t % 2] / bmm2_done[t % 2]: parity (t // 2) & 1.
    if n_tiles > cutlass.Int32(0):
        bars.mb_q_full.wait(cutlass.Int32(0))
        desc_Q = sQ[0].desc()

        # Prologue: BMM1(0).
        slot0, phase0 = _kv_slot(cutlass.Int32(0), n_tiles)
        bars.mb_kv_full[slot0].wait(phase0)
        bars.mb_s_empty[0].wait(cutlass.Int32(1))
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        mma_ss(bmm1_desc, sK[slot0].desc(), desc_Q, tmem_raw.subview(cutlass.Int32(S_ACC_OFF[0])))
        elect_p = nvvm.elect_sync()
        bars.mb_s_full[0].arrive(cta_group=1, pred=elect_p)
        bars.mb_kv_empty[slot0].arrive(cta_group=1, pred=elect_p)

        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            # BMM1(i + 1) ahead of BMM2(i): the softmax of tile i overlaps the next
            # score tile, and the K slot is released as early as possible.
            if i + cutlass.Int32(1) < n_tiles:
                t_next = i + cutlass.Int32(1)
                slot_k, phase_k = _kv_slot(cutlass.Int32(2) * t_next - cutlass.Int32(1), n_tiles)
                par_next = t_next & cutlass.Int32(1)
                phase_s_empty = ((t_next // cutlass.Int32(2)) - cutlass.Int32(1)) & cutlass.Int32(1)
                bars.mb_kv_full[slot_k].wait(phase_k)
                bars.mb_s_empty[par_next].wait(phase_s_empty)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
                s_off_next = _select_i32(par_next == cutlass.Int32(0), S_ACC_OFF[0], S_ACC_OFF[1])
                mma_ss(bmm1_desc, sK[slot_k].desc(), desc_Q, tmem_raw.subview(s_off_next))
                elect_p1 = nvvm.elect_sync()
                bars.mb_s_full[par_next].arrive(cta_group=1, pred=elect_p1)
                bars.mb_kv_empty[slot_k].arrive(cta_group=1, pred=elect_p1)

            seq_v = _select_i32(i + cutlass.Int32(1) < n_tiles, cutlass.Int32(2) * i + cutlass.Int32(2), cutlass.Int32(2) * n_tiles - cutlass.Int32(1))
            slot_v, phase_v = _kv_slot(seq_v, n_tiles)
            par_cur = i & cutlass.Int32(1)
            phase_p = (i // cutlass.Int32(2)) & cutlass.Int32(1)
            bars.mb_kv_full[slot_v].wait(phase_v)
            bars.mb_p_full[par_cur].wait(phase_p)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            desc_P = sP[par_cur].desc()
            accum = cutlass.Boolean(i > cutlass.Int32(0))
            for blk in cutlass.range_constexpr(O_BLOCKS):
                mma_ss(
                    bmm2_desc,
                    sVt[slot_v].shifted(blk * 2 * TILE_N * GRANU_ELEMS).desc(),
                    desc_P,
                    tmem_raw.subview(cutlass.Int32(O_OFF[blk])),
                    accumulate=accum,
                )
            elect_p2 = nvvm.elect_sync()
            bars.mb_bmm2_done[par_cur].arrive(cta_group=1, pred=elect_p2)
            bars.mb_kv_empty[slot_v].arrive(cta_group=1, pred=elect_p2)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, TMEM_COLS, CTA_GROUP_KIND)


@cute.jit
def _warp_reduce_max(x):
    # ftz=True (the FMNMX intrinsic, as tile_dsl's row_max_reduction and the sparse sibling spell it): the bare
    # cute.math.max lowers to compare + select -- FSETP.NAN + FSETP.GTU + two FSEL per max, every predicate register busy
    # -- which tipped the 18-warp (N_Q = 64) tile's KV loop over ptxas's 96-register cap; the inline-asm max.f32 form
    # pins its operands around each asm block and spilled more.  Softmax maxima are finite or -inf, so only a score below
    # the fp32 normal range (|s| < 2^-126) could read differently (flushed to 0).
    for off in cutlass.range_constexpr(5):
        x = cute.math.max(x, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=nvvm.Shfl.BFLY)), ftz=True)
    return x


@cute.jit
def _warp_reduce_sum(x):
    for off in cutlass.range_constexpr(5):
        x = x + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=nvvm.Shfl.BFLY))
    return x


@cute.jit
def _softmax_warp_group(
    tmem_ptr_i32,
    bars,
    red_smem,
    mred_smem,
    sP_raw,
    o_tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    n_batch,
    head_idx,
    batch_idx,
    split_idx,
    tok0,
    scale_log2: cutlass.Float32,
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    scale_o_t: Optional[cute.Tensor],
    amax_o_tensor: Optional[cute.Tensor],
    rz,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    nvvm.barrier_cta_sync(barrier_id=_BAR_TMEM, thread_count=32 * (SOFTMAX_WARPS + 1))
    tidx = cute.arch.thread_idx()[0]
    sm_warp = tidx // cutlass.Int32(32)  # 0 .. SOFTMAX_WARPS - 1
    if cutlass.const_expr(COL_GROUPS == 1):
        lane = tidx  # key row of S^T / P^T, d row of O^T
        col0 = cutlass.Int32(0)
    else:
        lane = tidx % cutlass.Int32(128)
        # Column group (warp-uniform): the first S^T / O^T column this thread owns.
        col0 = cute.arch.make_warp_uniform(tidx // cutlass.Int32(128)) * cutlass.Int32(COLS)
    G = cutlass.Int32(HEADS_PER_TILE)
    tmem_base = tmem_ptr_i32.load()

    if cutlass.const_expr(use_pdl):
        # PDL: the TMEM handshake above is this role's prologue; its first global read (the lengths) is below.
        wait_on_dependent_grids()
    scale_log2 = scale_log2 * cutlass.Float32(cutlass.make_array_view(descale_q_t)[0]) * cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
    value_scale = cutlass.Float32(cutlass.make_array_view(descale_v_t)[0]) * cutlass.Float32(0.0625)
    output_scale = cutlass.Float32(1.0)
    if cutlass.const_expr(scale_o_t is not None):
        output_scale = cutlass.Float32(cutlass.make_array_view(scale_o_t)[0])
    amax_local = cutlass.Float32(0.0)
    lo, hi, eff_seqlen_kv, eff_seqlen_q, q_len_b = _unit_bounds(seq_kv_lens_tensor, seq_q_lens_addr, batch_idx, split_idx, seqlen_q, seqlen_kv, tok0)
    n_tiles = hi - lo
    causal_diag = cutlass.Int32(0)
    if cutlass.const_expr(CFG.BOTTOM_RIGHT):
        causal_diag = eff_seqlen_kv - eff_seqlen_q

    NEG_INF = cutlass.Float32(float("-inf"))
    ZERO = cutlass.Float32(0.0)
    ONE = cutlass.Float32(1.0)
    # Running column state: max in the scaled log2 domain (-inf = no live key
    # yet), this lane's partial sum (the lane sum is reduced once, at the end).
    m_vec = cutlass.Vector.from_elements(tuple(NEG_INF for _ in range(COLS)), cutlass.Float32)
    l_vec = cutlass.Vector.from_elements(tuple(ZERO for _ in range(COLS)), cutlass.Float32)

    red_ptr = Pointer(red_smem.data_ptr(), dtype=cutlass.Float32)
    mred_ptr = Pointer(mred_smem.data_ptr(), dtype=cutlass.Int32)
    # The column each thread publishes into its group's max slot (lanes 16..31 repeat 0..15: a duplicate max is idempotent),
    # the group's slot index, and the group's first warp (it re-arms the slot two tiles ahead).
    key_col = (tidx % cutlass.Int32(32)) % cutlass.Int32(COLS)
    grp = col0 // cutlass.Int32(COLS)
    group_lead = (sm_warp % cutlass.Int32(4)) == cutlass.Int32(0)

    for i in cutlass.range(0, n_tiles, 1, unroll=1):
        kv_tile = lo + i
        par = i & cutlass.Int32(1)
        phase_i = (i // cutlass.Int32(2)) & cutlass.Int32(1)
        s_off = _select_i32(par == cutlass.Int32(0), S_ACC_OFF[0], S_ACC_OFF[1])

        bars.mb_s_full[par].wait(phase_i)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        s_raw = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + s_off + col0, cutlass.Float32), num=COLS)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        # Every score word contributes to the release address before the slot is reused.
        consume_then_arrive(bars.mb_s_empty[par], s_raw, rz)

        key_abs = kv_tile * cutlass.Int32(TILE_N) + lane
        # Per-column (= per Q row) mask in the transposed orientation: the key
        # index = the thread's lane, the Q token = the unit's first token + the column's row // G.
        s_cols = []
        for j in cutlass.range_constexpr(COLS):
            v = cutlass.Float32(s_raw[j]) * scale_log2
            if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                q_abs = tok0 + (col0 + cutlass.Int32(j)) // G
                masked = None
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
                    masked = key_abs >= eff_seqlen_kv
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
                    term = key_abs > (q_abs + causal_diag + cutlass.Int32(CFG.WINDOW_RIGHT))
                    masked = term if masked is None else (masked | term)
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
                    term = key_abs < (q_abs + causal_diag - cutlass.Int32(CFG.WINDOW_LEFT))
                    masked = term if masked is None else (masked | term)
                v = _select_f32(masked, NEG_INF, v)
            s_cols.append(v)

        # Column max over the tile's 128 keys: butterfly inside the warp, then the group's four warps meet in the
        # tile's 16-word slot through red.shared.max on order-preserving keys; after the barrier every lane reads the
        # 16 group maxima back (no fold, nothing retained across the exchange).  The slot rotates per tile and the
        # group's first warp re-arms the slot two tiles ahead: its last readers passed this tile's barrier, its next
        # writers wait the next one, so the one barrier per tile orders every reuse.
        mbase = ((i % cutlass.Int32(MAX_RED_SLOTS)) * cutlass.Int32(COL_GROUPS) + grp) * cutlass.Int32(COLS)
        my_key = cutlass.Int32(INT32_MIN)
        for j in cutlass.range_constexpr(COLS):
            my_key = _select_i32(key_col == cutlass.Int32(j), _max_key(_warp_reduce_max(s_cols[j])), my_key)
        nvvm.inline_ptx("red.shared.max.s32 [{$r0}], {$r1};", read_only_args=[mred_smem.subview(mbase + key_col), my_key])
        nvvm.barrier_cta_sync(barrier_id=_BAR_SOFTMAX, thread_count=SOFTMAX_LANES)
        tile_max = []
        for j in cutlass.range_constexpr(COLS):
            tile_max.append(_key_max(cutlass.Int32((mred_ptr + (mbase + cutlass.Int32(j))).load())))
        if group_lead:
            rbase = (((i + cutlass.Int32(2)) % cutlass.Int32(MAX_RED_SLOTS)) * cutlass.Int32(COL_GROUPS) + grp) * cutlass.Int32(COLS)
            (mred_ptr + (rbase + key_col)).store(cutlass.Int32(INT32_MIN))

        # Online update per column (identical in every lane: the max is the
        # tile-wide column max, so alpha is uniform and O^T's rescale is exact).
        m_new = []
        l_new = []
        alpha = []
        all_one = None
        p_cols = []
        for j in cutlass.range_constexpr(COLS):
            m_old_j = cutlass.Float32(m_vec[j])
            m_new_j = cute.math.max(m_old_j, tile_max[j], ftz=True)
            ms_old = row_max_for_exp2(m_old_j)
            ms_new = row_max_for_exp2(m_new_j)
            alpha_j = cute.math.exp2(cute.math.min(ms_old - ms_new, ZERO), fastmath=True)
            p_j = cute.math.exp2(s_cols[j] - ms_new, fastmath=True)
            l_new.append(cutlass.Float32(l_vec[j]) * alpha_j + p_j)
            m_new.append(m_new_j)
            alpha.append(alpha_j)
            p_cols.append(p_j * cutlass.Float32(16.0) if cutlass.const_expr(FP8) else p_j)
            is_one = alpha_j == ONE
            all_one = is_one if all_one is None else (all_one & is_one)
        m_vec = cutlass.Vector.from_elements(tuple(m_new), cutlass.Float32)
        l_vec = cutlass.Vector.from_elements(tuple(l_new), cutlass.Float32)

        # P^T(t) -> SMEM slot t % 2 as packed FP8 values, row = this lane's key,
        # this group's COLS-wide slice of it, swizzled the way the BMM2 B
        # descriptor decodes it.
        p_words = []
        if cutlass.const_expr(FP8):
            # The shared packer takes one full 16-byte FP8 atom.
            for _ in cutlass.range_constexpr(16 - COLS):
                p_cols.append(ZERO)
            packed = fp32_to_fp8_pack(p_cols, dtype=STORAGE_DTYPE)
            for w in cutlass.range_constexpr(P_GROUP_WORDS):
                p_words.append(packed[w])
        else:
            for w in cutlass.range_constexpr(P_GROUP_WORDS):
                p_words.append(fp32_to_fp16(p_cols[2 * w], p_cols[2 * w + 1], dtype=STORAGE_DTYPE))
        # Keep the complete 16-byte MMA operand initialized for FP8 live8.
        for _ in cutlass.range_constexpr(P_CHUNKS_PER_GROUP * 4 - P_GROUP_WORDS):
            p_words.append(cutlass.Int32(0))
        p_row_base = par * cutlass.Int32(pBufferWords) + lane * cutlass.Int32(P_ROW_WORDS) + (col0 * cutlass.Int32(BPE) // cutlass.Int32(4))
        for c in cutlass.range_constexpr(P_CHUNKS_PER_GROUP):
            chunk = cutlass.Vector.from_elements(tuple(p_words[4 * c + k] for k in range(4)), cutlass.Int32)
            p_ptr = Pointer(sP_raw.subview(p_row_base + cutlass.Int32(4 * c)).data_ptr(), dtype=cutlass.Int32)
            p_ptr.store_swizzled(chunk, _P_SMEM_SWIZZLE, alignment=16)
        nvvm.fence_proxy("async.shared", space="cta")

        # O^T(t-1) is complete once BMM2(t-1) commits; rescale this group's
        # columns by alpha before BMM2(t) accumulates on top.  Skipped when no
        # column's max moved.
        if i > cutlass.Int32(0):
            t_prev = i - cutlass.Int32(1)
            bars.mb_bmm2_done[t_prev & cutlass.Int32(1)].wait((t_prev // cutlass.Int32(2)) & cutlass.Int32(1))
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            if ~all_one:
                # Two x8 halves per d block (the register peak of the loop is here: alpha, the running max / sum and the
                # O^T values together; the x16 readback tipped the 18-warp N_Q = 64 tile over ptxas's 96-register cap).
                for blk in cutlass.range_constexpr(O_BLOCKS):
                    for h in cutlass.range_constexpr(2):
                        o_ptr = nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(O_OFF[blk] + h * (COLS // 2)) + col0, cutlass.Float32)
                        o_vals = nvvm.tcgen05_ld("32x32b", o_ptr, num=COLS // 2)
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        o_scaled = cutlass.Vector.from_elements(
                            tuple(cutlass.Float32(o_vals[jj]) * alpha[h * (COLS // 2) + jj] for jj in range(COLS // 2)), cutlass.Float32
                        )
                        nvvm.tcgen05_st("32x32b", o_ptr, o_scaled)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        # Publish P^T(t) (and the rescaled O^T) to the MMA warp: the tcgen05 stores
        # are ordered before the arrive, and the MMA fences after its wait.
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.mb_p_full[par].arrive()

    # --- epilogue -----------------------------------------------------------
    if n_tiles > cutlass.Int32(0):
        # The last BMM2 must land before O^T is read.
        t_last = n_tiles - cutlass.Int32(1)
        bars.mb_bmm2_done[t_last & cutlass.Int32(1)].wait((t_last // cutlass.Int32(2)) & cutlass.Int32(1))
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)

    # Column sums over the 128 lanes (once per unit), via the epilogue slot.
    red_epi = sm_warp * cutlass.Int32(COLS)
    for j in cutlass.range_constexpr(COLS):
        (red_ptr + (red_epi + cutlass.Int32(j))).store(_warp_reduce_sum(cutlass.Float32(l_vec[j])))
    nvvm.barrier_cta_sync(barrier_id=_BAR_SOFTMAX, thread_count=SOFTMAX_LANES)
    red_epi_read = (sm_warp - (sm_warp % cutlass.Int32(4))) * cutlass.Int32(COLS)

    LN2_F = cutlass.Float32(LN2)
    TINY = cutlass.Float32(1e-30)
    # Under a KV split the fold moves to the combine pass (split_combine), which adds the sink column exactly once; the partials stay sink-free.
    if cutlass.const_expr(CFG.HAS_SINK and SPLIT_KV == 1):
        sinks_arr = cutlass.make_array_view(sinks_tensor)
    # Split partials stack split-major on the workspace batch axis (b + s*B).
    o_batch = batch_idx + split_idx * n_batch
    # O[q, d] = O^T[d, q] / l[q]: this lane is d (and d + 128); the store
    # writes the graph's actual d_v columns only (the tile is an envelope).
    oo = cutlass.make_array_view(o_tensor)
    D_V = cutlass.const_expr(o_tensor.shape[3])
    # The epilogue runs in two 8-column halves: each half derives its columns' inverse sum / dead / live facts, then
    # reads that half of O^T back (the x8 tcgen05.ld shape) and stores it.  The 16-wide per-column lists and the x16
    # O^T readback together were what the 18-warp N_Q = 64 tile spilled at ptxas's 96-register cap; the halves keep
    # the same arithmetic and the same stores (bitwise).  Each thread stores ONE LSE (the column its key lane names, lane == j), kept
    # as a per-lane scalar with a liveness bit; the column's token / head are two integer ops on a warp-uniform row
    # index, recomputed at each store site rather than carried in lists.
    O_HALF = COLS // 2
    my_lse = ZERO
    my_lse_live = lane < cutlass.Int32(0)  # False
    for h in cutlass.range_constexpr(2):
        inv_cols = []
        dead_cols = []
        live_cols = []
        for jj in cutlass.range_constexpr(O_HALF):
            j = h * O_HALF + jj
            jg = col0 + cutlass.Int32(j)  # the packed Q row this column is
            tok_j = tok0 + jg // G
            head_j = head_idx * G + (jg % G)
            # Every warp's partial comes back from SMEM (this warp's included), so
            # the group's four warps see the same total.
            l_tot = ZERO
            for w in cutlass.range_constexpr(4):
                l_tot = l_tot + cutlass.Float32((red_ptr + (red_epi_read + cutlass.Int32(w * COLS + j))).load())
            m_raw = cutlass.Float32(m_vec[j])  # -inf while the row has no live key
            m_nat = row_max_for_exp2(m_raw) * LN2_F
            row_dead = l_tot <= ZERO
            if cutlass.const_expr(CFG.HAS_SINK and SPLIT_KV == 1):
                # The sink folds against the row's ACTUAL max (what the prefill tile
                # does with its published max), -inf on a keyless row: new_max is
                # then the sink logit itself, scale = exp(-inf) = 0 and new_sum = 1,
                # so LSE is the sink logit exactly and O := 0 through inv = 0.  The
                # 0-substitute above is for exp2 arguments only: anchoring the fold
                # on it charged a phantom logit 0 against the sink, and a finite sink
                # far below it (-100) underflowed exp() to 0 -> LSE = -inf.
                m_fold = m_raw * LN2_F
                sink_logit = cutlass.Float32(sinks_arr[head_j])
                new_max = cute.math.max(m_fold, sink_logit)
                scale = cute.math.exp(m_fold - new_max, fastmath=True)
                new_sum = l_tot * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
                lse_j = new_max + cute.math.log(new_sum, fastmath=True)
                inv_j = scale / new_sum
            else:
                lse_j = m_nat + cute.math.log(cute.math.max(l_tot, TINY), fastmath=True)
                inv_j = ONE / cute.math.max(l_tot, TINY)
                lse_j = _select_f32(row_dead, NEG_INF, lse_j)
                inv_j = _select_f32(row_dead, ZERO, inv_j)
            live_j = tok_j < seqlen_q
            if cutlass.const_expr(Q_BOX_ROWS < N_Q):
                # The zero-filled tail rows past the Q box (a head group that does not divide N_Q) decode
                # to the NEXT unit's first token (jg // G == Q_BOX_TOKENS), a live token whenever another
                # unit follows: they are never stored -- without this guard a non-last unit's tail
                # overwrote the next unit's first token (a CTA-order race: 5-14 % of that token's O
                # elements wrong, which unit's store landed last).
                live_j = live_j & (jg < cutlass.Int32(Q_BOX_ROWS))
            if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
                # Dense padded-Q trim: rows past seq_len_q[b] write O := 0 / LSE := -inf,
                # applied AFTER the sink fold (a trimmed row is dead even with a sink).
                row_trim = tok_j >= q_len_b
                lse_j = _select_f32(row_trim, NEG_INF, lse_j)
                inv_j = _select_f32(row_trim, ZERO, inv_j)
                row_dead = row_dead | row_trim
            # Direct template callers share the natural-log partial contract
            # with the adapter; only final Stats can use base 2 (SDPA Rule S4).
            if cutlass.const_expr(CFG.STATS_LOG2 and SPLIT_KV == 1):
                lse_j = lse_j * cutlass.Float32(LOG2E)
            is_mine = lane == cutlass.Int32(j)
            my_lse = _select_f32(is_mine, lse_j, my_lse)
            my_lse_live = my_lse_live | (is_mine & live_j)
            inv_cols.append(inv_j * value_scale)
            dead_cols.append(row_dead)
            live_cols.append(live_j)
        for blk in cutlass.range_constexpr(O_BLOCKS):
            if cutlass.const_expr(blk * 128 < D_V):
                d_idx = lane + cutlass.Int32(blk * 128)
                o_vals = cutlass.Vector.from_elements(tuple(ZERO for _ in range(O_HALF)), cutlass.Float32)
                if n_tiles > cutlass.Int32(0):
                    # Empty range: O^T TMEM was never written, so the load is skipped
                    # rather than multiplied by 0 (NaN * 0 is NaN).
                    o_vals = nvvm.tcgen05_ld(
                        "32x32b", nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(O_OFF[blk] + h * O_HALF) + col0, cutlass.Float32), num=O_HALF
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if d_idx < cutlass.Int32(D_V):
                    for jj in cutlass.range_constexpr(O_HALF):
                        jg = col0 + cutlass.Int32(h * O_HALF + jj)
                        val = _select_f32(dead_cols[jj], ZERO, cutlass.Float32(o_vals[jj]) * inv_cols[jj])
                        if live_cols[jj]:
                            if cutlass.const_expr(SPLIT_KV == 1 and amax_o_tensor is not None):
                                amax_local = cute.math.max(amax_local, cute.math.abs(val))
                            o_row = oo[o_batch, tok0 + jg // G, head_idx * G + (jg % G), :]
                            o_row[d_idx] = (val * output_scale).to(o_tensor.element_type)

    if cutlass.const_expr(lse_tensor is not None):
        lse_arr = cutlass.make_array_view(lse_tensor)
        if my_lse_live:
            jg = col0 + lane  # lanes 0 .. COLS-1 own one column each (my_lse_live is False past them)
            lse_arr[o_batch, head_idx * G + (jg % G), tok0 + jg // G] = my_lse

    if cutlass.const_expr(SPLIT_KV == 1 and amax_o_tensor is not None):
        # Pre-quantization AMAX; all lanes participate, only the warp leader publishes.
        amax_warp = _warp_reduce_max(amax_local)
        if tidx % cutlass.Int32(32) == cutlass.Int32(0):
            amax_ptr = Pointer(amax_o_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
            nvvm.atomicrmw(nvvm.AtomicOp.MAX, amax_ptr, amax_warp.bitcast(cutlass.Int32))
    bars.mb_tmem_dealloc.arrive()


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_batch: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    scale_o_t: Optional[cute.Tensor],
    amax_o_tensor: Optional[cute.Tensor],
    # Dense padded-Q trim: (B,)-int32 per-batch Q lengths by address; 0 (unread)
    # unless CFG.SEQ_Q_LENS_PRESENT.
    seq_q_lens_addr: cutlass.Int64 = 0,
    # Token units per (head group, batch, split): ceil(S_q / Q_BOX_TOKENS), derived by the host entry.
    n_tok_units: cutlass.Int32 = 1,
    # Paged KV: [B, max_pages] int32 page ids for K and V; None unless CFG.PAGED_KV.
    block_table_tensor: Optional[cute.Tensor] = None,
    block_table_v_tensor: Optional[cute.Tensor] = None,
    paged_hnd: cutlass.Constexpr[bool] = False,
    # Programmatic dependent launch (the launch attribute is set by the host entry under the same flag): each role waits on the
    # dependent grid right after its own prologue and before its first global read, and the TMA role releases the dependent grid.
    use_pdl: cutlass.Constexpr[bool] = False,
    rz_arg: cutlass.Int32 = 0,
) -> None:
    rz = release_zero(rz_arg)
    # The following combine owns the reduction; one producer initializes it.
    if cutlass.const_expr(SPLIT_KV > 1 and amax_o_tensor is not None):
        _initialize_split_amax(amax_o_tensor)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    # One CTA per unit: x = (KV-head group (packed) or Q head (unpacked)) x token unit, y = batch, z = split.
    # The token unit is the fast index of x, so one group's units are launched together.
    head_idx = cute.arch.block_idx()[0] // n_tok_units
    tok0 = (cute.arch.block_idx()[0] % n_tok_units) * cutlass.Int32(Q_BOX_TOKENS)
    batch_idx = cute.arch.block_idx()[1]
    split_idx = cute.arch.block_idx()[2]

    sQ_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sKV_raw = cutlass.Array(STORAGE_DTYPE, STAGES * kvBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sP_raw = cutlass.Array(cutlass.Int32, 2 * pBufferWords, alignment=1024, space=cutlass.AddressSpace.smem)
    mred_smem = cutlass.Array(cutlass.Int32, MAX_RED_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    red_smem = cutlass.Array(cutlass.Float32, RED_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    # Q^T: the K-major B operand of BMM1 (N_Q rows x TILE_K), one TMA sub-tile
    # (N_Q rows x 128 B) per 128-element FP8 head-dim granule.
    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=QK_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=N_Q * GRANU_ELEMS,
    )
    # The K/V ring, seen as the K-major A operand of BMM1 (TILE_N rows x TILE_K) ...
    sK = SmemTile(
        base=sKV_raw,
        elems_per_stage=kvBufferElems,
        stages=STAGES,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=QK_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=TILE_N * GRANU_ELEMS,
    )
    # ... and as the MN-major A operand of BMM2 (V^T: d contiguous, LBO steps
    # the 64-wide d atoms).  Same storage, different descriptor.
    sVt = SmemTile(
        base=sKV_raw,
        elems_per_stage=kvBufferElems,
        stages=STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_VT,
        stride_byte_offset=STRIDE_BYTE_OFFSET_VT,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=VO_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=TILE_N * GRANU_ELEMS,
    )
    # P^T: the MN-major B operand of BMM2, two buffers of TILE_N key rows.
    sP = SmemTile(
        base=sP_raw,
        elems_per_stage=pBufferWords,
        stages=2,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_P,
        layout=SMEM_LAYOUT_P,
    )

    bars = make_decode_bars()

    if warp_idx == 0:
        if nvvm.elect_sync():
            bars.mb_q_full.init()
            for s in cutlass.range_constexpr(STAGES):
                bars.mb_kv_full[s].init()
                bars.mb_kv_empty[s].init()
            for p in cutlass.range_constexpr(2):
                bars.mb_s_full[p].init()
                bars.mb_s_empty[p].init()
                bars.mb_p_full[p].init()
                bars.mb_bmm2_done[p].init()
            bars.mb_tmem_dealloc.init()

    if cutlass.const_expr(Q_BOX_ROWS < N_Q):
        # Zero the Q^T tail rows the TMA box does not cover (a head group that
        # does not divide N_Q), so their never-stored columns stay finite.
        if warp_idx < SOFTMAX_WARPS:
            zero4 = cutlass.Vector.from_elements(tuple(cutlass.Int32(0) for _ in range(4)), cutlass.Int32)
            tail_chunks = (N_Q - Q_BOX_ROWS) * 8  # 16 B chunks per sub-tile
            for it in cutlass.range_constexpr((QK_ITERS * tail_chunks + SOFTMAX_LANES - 1) // SOFTMAX_LANES):
                chunk = cutlass.Int32(it * SOFTMAX_LANES) + tidx
                if chunk < cutlass.Int32(QK_ITERS * tail_chunks):
                    sub = chunk // cutlass.Int32(tail_chunks)
                    within = chunk % cutlass.Int32(tail_chunks)
                    elem_off = sub * cutlass.Int32(N_Q * GRANU_ELEMS) + cutlass.Int32(Q_BOX_ROWS * GRANU_ELEMS) + within * cutlass.Int32(16 // BPE)
                    Pointer(sQ_raw.subview(elem_off).data_ptr(), dtype=cutlass.Int32).store(zero4, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")

    if warp_idx < SOFTMAX_WARPS:
        if cutlass.const_expr(FP8 or COLS < 16):
            # The declared graph uses at most eight columns. Keep the MMA
            # operand's unused columns initialized while processing only the
            # live half in softmax; each lane owns one complete key row.
            zero4 = cutlass.Vector.from_elements(tuple(cutlass.Int32(0) for _ in range(4)), cutlass.Int32)
            for slot in cutlass.range_constexpr(2):
                offset = cutlass.Int32(slot * pBufferWords + P_CHUNKS_PER_GROUP * 4) + tidx * cutlass.Int32(P_ROW_WORDS)
                Pointer(sP_raw.subview(offset).data_ptr(), dtype=cutlass.Int32).store_swizzled(zero4, _P_SMEM_SWIZZLE, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
        # Arm every max-key slot (one word per lane): the first MAX_RED_SLOTS tiles find them at the smallest key.
        if tidx < cutlass.Int32(MAX_RED_WORDS):
            (Pointer(mred_smem.data_ptr(), dtype=cutlass.Int32) + tidx).store(cutlass.Int32(INT32_MIN))

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    if warp_idx < SOFTMAX_WARPS:
        _softmax_warp_group(
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            red_smem=red_smem,
            mred_smem=mred_smem,
            sP_raw=sP_raw,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            n_batch=n_batch,
            head_idx=head_idx,
            batch_idx=batch_idx,
            split_idx=split_idx,
            tok0=tok0,
            scale_log2=scale_softmax_log2,
            descale_q_t=descale_q_t,
            descale_k_t=descale_k_t,
            descale_v_t=descale_v_t,
            scale_o_t=scale_o_t,
            amax_o_tensor=amax_o_tensor,
            rz=rz,
            use_pdl=use_pdl,
        )
    elif warp_idx == MMA_WARP_ID:
        _mma_warp_group(
            sQ=sQ,
            sK=sK,
            sVt=sVt,
            sP=sP,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            batch_idx=batch_idx,
            split_idx=split_idx,
            tok0=tok0,
            use_pdl=use_pdl,
        )
    elif warp_idx == TMALDG_WARP_ID:
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sQ=sQ,
            sKV=sK,
            bars=bars,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            head_idx=head_idx,
            batch_idx=batch_idx,
            split_idx=split_idx,
            tok0=tok0,
            block_table_tensor=block_table_tensor,
            block_table_v_tensor=block_table_v_tensor,
            paged_hnd=paged_hnd,
            use_pdl=use_pdl,
        )


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    v_tensor: cute.Tensor,
    o_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    problem_size: Tuple[int, int, int, int, int, int],
    scale_softmax_log2: cutlass.Float32,
    o_scale_fused: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    scale_o_t: Optional[cute.Tensor],
    amax_o_tensor: Optional[cute.Tensor],
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_tensor: Optional[cute.Tensor] = None,
    thd_kv_lens_tensor: Optional[cute.Tensor] = None,
    thd_lens_form: Optional[cutlass.Int32] = None,
    o_partial_f32: Optional[cute.Tensor] = None,
    block_table_tensor: Optional[cute.Tensor] = None,
    block_table_v_tensor: Optional[cute.Tensor] = None,
    paged_hnd_prepared: cutlass.Constexpr[bool] = False,
    prepared: cutlass.Constexpr[bool] = True,
    stream: _cuda_driver.CUstream = None,
):
    B, QH, KH, SQ, SKV, _ = problem_size
    paged_hnd = paged_hnd_prepared
    # Q^T box: Q_BOX_TOKENS tokens x HEADS_PER_TILE heads of one KV group, row =
    # token * G + head (the packed-row convention); rows past S_q zero-fill.
    box_q = (1, Q_BOX_TOKENS, HEADS_PER_TILE, GRANU_ELEMS)
    box_kv = (1, KV_BOX_ROWS, 1, GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)
    kv_stride_order = (3, 1, 2, 0) if paged_hnd else stride_order

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    def _create_tma_desc(tensor: cute.Tensor, box_dims, swizzle, order=stride_order):
        return tmap.create_tensor_map_tiled(
            global_address=tensor.iterator.toint(),
            dtype=tensor.element_type,
            global_dims=tuple(tensor.shape[i] for i in order),
            global_strides=tuple(cutlass.Int64(tensor.stride[i]) * tensor.element_type.width // 128 for i in order[1:]),
            box_dims=tuple(box_dims[i] for i in order),
            swizzle=swizzle,
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )

    tma_q_desc = _create_tma_desc(q_tensor, box_q, _tma_swz(CFG.Q_SWZ_BYTES))
    tma_k_desc = _create_tma_desc(k_tensor, box_kv, _tma_swz(CFG.K_SWZ_BYTES), kv_stride_order)
    tma_v_desc = _create_tma_desc(v_tensor, box_kv, _tma_swz(CFG.V_SWZ_BYTES), kv_stride_order)

    # Token units: ceil(SQ / Q_BOX_TOKENS) CTAs per (head group, batch, split), the unit the fast index of x.
    n_tok_units = (cutlass.Int32(SQ) + cutlass.Int32(Q_BOX_TOKENS - 1)) // cutlass.Int32(Q_BOX_TOKENS)
    grid_shape = ((QH // HEADS_PER_TILE) * n_tok_units, B, SPLIT_KV)

    kernel_args = (
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        o_tensor,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(B),
        scale_softmax_log2,
        descale_q_t,
        descale_k_t,
        descale_v_t,
        scale_o_t,
        amax_o_tensor,
        seq_q_lens_addr,
        n_tok_units,
    )

    _kernel(*kernel_args, block_table_tensor, block_table_v_tensor, paged_hnd, False, cutlass.Int32(0)).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        stream=stream,
    )


LSE_KINDS = ("dense",)


def compile_prepared(
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_kind: str = "dense",
    paged_hnd: bool = False,
    has_amax: bool = True,
    scale_o_in_combine: bool = False,
) -> Callable:
    cache_key = _template_key(globals(), locals(), "compile_prepared")
    from cudnn.sdpa.fwd.kernels._fp8_host import compile_host

    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError("FP8 decode dimensions exceed the compiled envelope")
    if d_qk % 16 or d_v % 16 or lse_kind != "dense":
        raise ValueError("FP8 decode requires 16-byte head strides and dense Stats")
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("FP8 decode split-KV requires partial Stats")
    return compile_host(
        _host,
        CFG,
        STORAGE_DTYPE,
        OUT_STORAGE_DTYPE,
        False,
        cache_key,
        d_qk,
        d_v,
        has_lse,
        lse_kind,
        has_amax,
        scale_o_in_combine,
        paged_hnd,
        optional_amax=True,
        amax_prescaled=True,
    )


# The adapter inspects this signature before selecting compile_prepared.
compile = compile_prepared

from cudnn.sdpa.fwd.kernels._fp8_host import host as _host_prepared
