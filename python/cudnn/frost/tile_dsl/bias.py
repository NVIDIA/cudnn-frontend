# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The additive attention bias as a RESIDENT SMEM row-block of a persistent SDPA forward.

One work item of the classic (d128-family) pipeline owns ``BLOCK_ROWS = TILES_Q x TILE_M`` query rows and a KV
range of ``n_tiles`` ``TILE_N``-column tiles.  The bias slice it needs, ``[BLOCK_ROWS x n_tiles * TILE_N]`` of
the io dtype, is loaded ONCE per item by the TMA-LDG warp (:func:`issue_bias_block_load`) into a block the
softmax warps read for the item's lifetime -- one lane = one query row, a 128-byte row slice of a 64-column
sub-tile per 16-byte-granule swizzled load (:func:`load_bias_row_chunk`) -- and add pre-scale to the raw
scores as one packed FFMA per pair (:func:`bias_fma_chunk`): ``s_raw + bias * inv_scale`` keeps the kernel's
raw-S + scale-at-the-exp2 convention (``(s_raw + bias / scale) * scale == s_raw * scale + bias``).

The block's swizzle does BOTH jobs (frost-kernels s5): the TMA descriptor writes it under the 128-byte atom (a
descriptor reads it: job D), and the softmax lanes address it one 128-byte row each (lanes read it: job L --
a per-lane stride that is a multiple of the 128-byte bank cycle is a 32-way conflict unswizzled; the 16-byte
chunk index XOR (row % 8) spreads eight consecutive rows over eight bank groups, so a warp-wide 16-byte load
costs the ideal 4 wavefronts).  ``cutlass.Swizzle(3, 4, 3)``: ``MBase + SShift = 7 = log2(128 B)``.

Geometry is derived from the kernel's Cfg (:func:`bias_block_geometry`), never spelled per call site; the
expect_tx of the item's fill is the sum of the FULL boxes (an overshooting box past S_q / S_kv is zero-filled
AND counted), so a tail tile needs no special case.

Above the resident budget the SAME block geometry becomes a STREAMED RING (``CFG.BIAS_MODE == 2``): the TMA-LDG
warp issues ONE tile per kv tile into a ``STAGES_BIAS``-deep ring (:func:`issue_bias_ring_tile`), the softmax
warps read the stage exactly as they read a resident tile (the tile index is the ring stage) and release it per
kv tile.  Same swizzle, same boxes, same lane loads; only the SMEM extent and the two mbarrier rings differ.

The ADDITIVE per-key vector (the ``kv_mask`` input in its bf16 / fp16 / fp32 form: one value per key, broadcast
over every query row, the models' finite ``-1e9`` fold) takes the SAME pre-scale FFMA (:func:`additive_vec_fma_chunk`)
from a warp-uniform GMEM load (:func:`additive_vec_granules`): nothing is staged in SMEM and no barrier row is added
-- the slice is ``n`` elements per kv tile, read as 16-byte granules every lane of the warp fetches at the same
address, so the LSU broadcasts one sector per instruction.  A granule past the vector's end is CLAMPED to the last
in-bounds one (the band / padding mask masks those columns anyway), which is why the vector's extent must be
whole granules (``S_kv % 8 == 0``) and its base / batch stride 16-byte aligned -- typed declines on the row.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from .pointwise import f16x2_to_f32, ffma2, fmax_f32, opaque_i32
from .tma import tma_load_tile

# The TMA swizzle atom the block is written and read under (bytes per swizzled row of one box).
BIAS_SWZ_BYTES = 128


class BiasBlockGeometry(NamedTuple):
    """Host-side geometry of the resident bias block, derived from ``CFG`` by :func:`bias_block_geometry`."""

    block_rows: int  # TILES_Q x TILE_M: the work item's query rows (one lane = one row)
    tile_n: int  # kv columns per tile (the K/V tile)
    bpe: int  # bytes per bias element (the io dtype's)
    swz_bytes: int  # 128: the TMA / lane swizzle atom
    boxes_per_tile: int  # TILE_N x bpe / 128: 64-column boxes per tile
    granu_elems: int  # 128 / bpe = 64 columns per box
    subtile_stride_elems: int  # block_rows x granu_elems: one box's SMEM extent (box-major inside a tile)
    tile_elems: int  # block_rows x tile_n
    tile_bytes: int  # tile_elems x bpe = the tx bytes ONE tile's boxes deliver
    n_tiles: int  # the SMEM extent in tiles: BIAS_RESIDENT_TILES (the resident block) or the streamed ring's STAGES_BIAS
    box_dims: tuple  # (1, 1, block_rows, granu_elems) in the (B, H, S_q, S_kv) order of the bias tensor
    layout_enum: int  # the kernel's SmemTile layout enum for the 128-B swizzle
    smem_swizzle: object  # cutlass.Swizzle(3, 4, 3)


def bias_block_geometry(cfg, *, swz_enum: dict, n_tiles=None) -> BiasBlockGeometry:
    """Pure host arithmetic (called at module import by the kernel that carries the block).

    Every term comes from the Cfg: a literal here would be the canonical silent-wrong-answer bug (the box count
    and the expect_tx must agree with the SMEM extent the softmax lanes index).  ``n_tiles`` = the SMEM extent in
    tiles: the resident block's ``BIAS_RESIDENT_TILES`` (the default) or the STREAMED ring's stage count -- the
    caller passes its Cfg's answer so the box count, the expect_tx and the lane offsets stay ONE arithmetic."""
    if cfg.BIAS_SWZ_BYTES != BIAS_SWZ_BYTES:
        raise ValueError(
            f"bias_block_geometry: the resident block is written and read under the {BIAS_SWZ_BYTES}-B swizzle (CFG.BIAS_SWZ_BYTES={cfg.BIAS_SWZ_BYTES})"
        )
    if cfg.BIAS_BLOCK_ROWS <= 0 or cfg.BIAS_BLOCK_ROWS > 256:
        raise ValueError(f"bias_block_geometry: BIAS_BLOCK_ROWS={cfg.BIAS_BLOCK_ROWS} must be in 1..256 (one TMA box per 64-column sub-tile)")
    granu = BIAS_SWZ_BYTES // cfg.BPE
    if cfg.TILE_N % granu:
        raise ValueError(f"bias_block_geometry: TILE_N={cfg.TILE_N} must be whole {granu}-column boxes")
    boxes = cfg.TILE_N // granu
    if boxes != cfg.BIAS_BOXES_PER_TILE:
        raise ValueError(f"bias_block_geometry: CFG.BIAS_BOXES_PER_TILE={cfg.BIAS_BOXES_PER_TILE} but TILE_N x BPE / {BIAS_SWZ_BYTES} = {boxes}")
    tile_elems = cfg.BIAS_BLOCK_ROWS * cfg.TILE_N
    smem_tiles = cfg.BIAS_RESIDENT_TILES if n_tiles is None else int(n_tiles)
    if smem_tiles < 1:
        raise ValueError(f"bias_block_geometry: the bias block needs at least one SMEM tile (got {smem_tiles})")
    return BiasBlockGeometry(
        block_rows=cfg.BIAS_BLOCK_ROWS,
        tile_n=cfg.TILE_N,
        bpe=cfg.BPE,
        swz_bytes=BIAS_SWZ_BYTES,
        boxes_per_tile=boxes,
        granu_elems=granu,
        subtile_stride_elems=cfg.BIAS_BLOCK_ROWS * granu,
        tile_elems=tile_elems,
        tile_bytes=tile_elems * cfg.BPE,
        n_tiles=smem_tiles,
        box_dims=(1, 1, cfg.BIAS_BLOCK_ROWS, granu),
        layout_enum=swz_enum[BIAS_SWZ_BYTES],
        smem_swizzle=cutlass.Swizzle(3, 4, 3),
    )


@cute.jit
def issue_bias_block_load(
    sBias, tma_bias, mb_bias_full, n_tiles, tile_bytes: cutlass.Constexpr[int], tile_n: cutlass.Constexpr[int], kv_tile0, q_row, head, batch
):
    """TMA-LDG warp, once per non-empty work item: arm ``expect_tx`` for the WHOLE block (ONE elected lane; the
    sum of ``n_tiles`` full tiles -- every box delivers its full byte count, OOB rows / columns zero-filled and
    counted) and queue the ``n_tiles`` tiles' boxes with ``cta_group=1`` so every byte lands on THIS CTA's mbar.
    Tile ``t`` of the block holds kv columns ``(kv_tile0 + t) * tile_n ..``; the softmax indexes it as
    ``kv_tile - kv_tile0``.  The caller waits ``mb_bias_empty`` (the readers' release of the previous item)
    before this and folds the call out on an empty item (zero kv tiles: no arrive, no wait on either side)."""
    mb_bias_full.arrive(n_bytes=n_tiles * cutlass.Int32(tile_bytes), pred=nvvm.elect_sync())
    for t in cutlass.range(cutlass.Int32(0), n_tiles, 1, unroll=1):
        kv_col = (kv_tile0 + t) * cutlass.Int32(tile_n)
        tma_load_tile(sBias[t], tma_bias(kv_col, q_row, head, batch), mb_bias_full.smem_ptr, cta_group=1)


@cute.jit
def issue_bias_ring_tile(sBias_stage, tma_bias, mb_full_stage, tile_bytes: cutlass.Constexpr[int], tile_n: cutlass.Constexpr[int], kv_tile, q_row, head, batch):
    """TMA-LDG warp, once per kv tile of the STREAMED ring (``CFG.BIAS_MODE == 2``): arm ``expect_tx`` for ONE tile's
    FULL boxes (ONE elected lane; an overshooting box is zero-filled and counted) and queue the tile's boxes into ring stage
    ``sBias_stage`` with ``cta_group=1`` (every byte lands on THIS CTA's ``mb_full_stage``).  The caller waits the stage's
    ``mb_bias_ring_empty`` (the softmax warps' per-tile release, pre-armed for the first ``STAGES_BIAS`` fills) BEFORE this
    and advances its ``PipelineState`` AFTER it; the softmax indexes the tile by the same stage.  One fill per kv tile on
    every kv-tile path (prologue and mainloop of both item forms) -- the lockstep the ring's P14 statement rests on."""
    mb_full_stage.arrive(n_bytes=cutlass.Int32(tile_bytes), pred=nvvm.elect_sync())
    tma_load_tile(sBias_stage, tma_bias(kv_tile * cutlass.Int32(tile_n), q_row, head, batch), mb_full_stage.smem_ptr, cta_group=1)


def bias_chunk_smem_offset(gg: BiasBlockGeometry, tile_idx, chunk: int, row) -> cutlass.Int32:
    """Trace-time helper: the LINEAR (pre-swizzle) SMEM element offset of one lane's 64-column row slice -- tile
    ``tile_idx`` (runtime, relative to the item's first kv tile), box ``chunk`` (0..boxes_per_tile-1), row ``row``
    (the lane's query row inside the block).  Box-major inside a tile, 128 bytes per row inside a box."""
    return tile_idx * cutlass.Int32(gg.tile_elems) + cutlass.Int32(chunk * gg.subtile_stride_elems) + row * cutlass.Int32(gg.granu_elems)


def load_bias_row_chunk(sBias_base, gg: BiasBlockGeometry, tile_idx, chunk: int, row, *, opaque_offset: bool = False):
    """Trace-time helper: one lane's 64-column bias row slice as the 32 packed 16-bit PAIRS it is stored as
    (a ``Vector[Int32, 32]``), through ONE swizzled 128-byte load (the full swizzle period: 8 x 16-byte granules,
    each XOR-remapped by the row).  Kept packed: :func:`bias_fma_chunk` unpacks each pair with a 2-op shift /
    convert right where it is consumed, which is cheaper than a 64-element ``.to(Float32)`` and keeps the
    pre-loaded fragment at 32 registers across the ``bmm1_done`` wait.

    ``opaque_offset``: the SAME slice through an address the compiler cannot prove equal to another load's (an identity
    ``mov.b32`` on the offset) -- a SECOND read of a chunk whose first read must stay separate.  The hardware upper-bound
    row-max form reads a non-pre-loaded chunk twice: once EARLY (ahead of the ``bmm1_done`` wait) for its packed row max,
    the 32 words discarded right after the tree, and once late for the FFMA; one shared load would make the compiler hoist
    the FFMA's 32 words next to the early tree and hold them across both TMEM chunks (measured on the sm_107a rendering:
    STL / LDL 37 / 44 against the software form's 3 / 3 at the 128-wide tile).  SMEM read bandwidth is not the binding
    term of these bodies (ncu: the 16-B ideal wavefronts per shared load); the registers are."""
    off = bias_chunk_smem_offset(gg, tile_idx, chunk, row)
    if opaque_offset:
        off = opaque_i32(off)
    vec = sBias_base.subview(off).data_ptr().load_swizzled(gg.smem_swizzle, gg.swz_bytes, count=gg.granu_elems)
    return vec.bitcast(cutlass.Int32)


def bias_fma_chunk(raw_chunk, bias_words, inv_scale, n: int, *, dtype):
    """Trace-time helper: ``s[i] = bias[i] * inv_scale + raw[i]`` for the ``n`` scores of one softmax chunk, from the
    ``n / 2`` packed bias words of :func:`load_bias_row_chunk` -- each pair unpacked by :func:`f16x2_to_f32` (bf16:
    two bit moves; fp16: two ``cvt``) and folded by ONE ``fma.rn.f32x2`` (:func:`ffma2`), the packed form the
    ALU-bound small-D softmax needs (frost-tile-dsl s10b).  ``inv_scale`` is a runtime fp32 (a kernel argument,
    never a constant: a constant float operand into inline PTX ICEs libNVVM).  Returns a ``Vector[Float32, n]``."""
    if n % 2:
        raise ValueError(f"bias_fma_chunk: n={n} must be even (packed pairs)")
    # A ``Vector[Int32, n/2]`` (the swizzled SMEM row read) or a Python list of ``n/2`` packed words (the additive vector's
    # 16-byte GMEM granules, :func:`additive_vec_granules`).
    n_words = len(bias_words) if isinstance(bias_words, (list, tuple)) else int(bias_words.shape[0])
    if n_words != n // 2:
        raise ValueError(f"bias_fma_chunk: {n_words} bias words for {n} scores; need {n // 2}")
    elems = []
    for i in range(n // 2):
        b_lo, b_hi = f16x2_to_f32(bias_words[i], dtype=dtype)
        s_lo, s_hi = ffma2(b_lo, b_hi, inv_scale, inv_scale, raw_chunk[2 * i], raw_chunk[2 * i + 1])
        elems.append(s_lo)
        elems.append(s_hi)
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


# ---------------------------------------------------------------------------
# The per-ROW bias-tile max in the PACKED domain -- the hardware upper-bound row-max form (``CFG.BIAS_MAX_FORM == 1``)
# ---------------------------------------------------------------------------


@cute.jit
def max_f16x2_word(a, b, *, dtype):
    """PTX ``max.bf16x2`` / ``max.f16x2`` on two packed 16-bit pairs (``Int32`` words): the half-wise max (sm_80+, PTX ISA 7.0).
    NaN semantics are ``max.f32``'s (a NaN half yields the other operand's half); the max of two 16-bit values is one of them, so
    the result is EXACT in the bias dtype.  Used by :func:`packed_row_max_words`; never on unpacked fp32 (that is ``fmax_f32``)."""
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"max_f16x2_word: dtype must be Float16 or BFloat16, got {dtype}")
    tag = "bf16x2" if cutlass.const_expr(dtype == cutlass.BFloat16) else "f16x2"
    return inline_ptx(f"max.{tag} $0, $1, $2;", write_only_types=[cutlass.Int32], read_only_args=[a, b])


def packed_row_max_words(words, *, dtype):
    """Trace-time helper: ONE packed pair holding the half-wise max over the ``n / 2`` packed words of one softmax chunk's bias row
    slice (:func:`load_bias_row_chunk`'s ``Vector[Int32, n/2]`` or a Python list of words) -- a balanced binary tree of
    :func:`max_f16x2_word`, ``n / 2 - 1`` ops, on the SAME words :func:`bias_fma_chunk` unpacks for the add, so the row max costs no
    unpack of its own and no fp32 tree.  The hardware upper-bound row-max form of the bias body reduces this pair across the tile's
    chunks (one more :func:`max_f16x2_word` per extra chunk) and lifts it to fp32 with :func:`packed_pair_max_f32`."""
    n = len(words) if isinstance(words, (list, tuple)) else int(words.shape[0])
    if n < 1:
        raise ValueError("packed_row_max_words: at least one packed word")
    elems = [words[i] for i in range(n)]
    while len(elems) > 1:
        nxt = [max_f16x2_word(elems[i], elems[i + 1], dtype=dtype) for i in range(0, len(elems) - 1, 2)]
        if len(elems) % 2:
            nxt.append(elems[-1])
        elems = nxt
    return elems[0]


@cute.jit
def packed_pair_max_f32(word, *, dtype):
    """The max of the two halves of one packed pair as ``Float32`` (:func:`f16x2_to_f32` -- two bit moves on bf16, two ``cvt`` on
    fp16 -- then ``max.f32``): exact (a 16-bit value is exactly representable in fp32)."""
    lo, hi = f16x2_to_f32(word, dtype=dtype)
    return fmax_f32(lo, hi)


# ---------------------------------------------------------------------------
# The ADDITIVE per-key vector (kv_mask in its bf16 / fp16 / fp32 form): a warp-uniform GMEM slice, the same pre-scale FFMA
# ---------------------------------------------------------------------------

# One 16-byte LDG granule: 8 halves or 4 floats.
ADDITIVE_GRANULE_BYTES = 16


def additive_granule_elems(dtype) -> int:
    """Elements per 16-byte granule of an additive vector in ``dtype`` (8 for the 16-bit dtypes, 4 for fp32)."""
    if dtype == cutlass.Float32:
        return ADDITIVE_GRANULE_BYTES // 4
    if dtype == cutlass.BFloat16 or dtype == cutlass.Float16:
        return ADDITIVE_GRANULE_BYTES // 2
    raise TypeError(f"additive_granule_elems: dtype must be BFloat16, Float16 or Float32, got {dtype}")


def additive_vec_granules(vec_tensor, batch_idx, kv_col, seq_kv_len, n: int, *, dtype):
    """Trace-time helper: the ``n`` consecutive values of a per-key additive vector (a ``(B, S_kv)`` tensor in ``dtype``
    whose key dim is contiguous; a batch stride of 0 broadcasts one row) starting at key ``kv_col`` of batch ``batch_idx``,
    as a Python list of ``n / G`` 16-byte granule loads (``Vector[dtype, G]``, ``G`` = :func:`additive_granule_elems`).

    Every lane of the warp loads the SAME addresses (the vector has no query-row axis), so one instruction per granule
    costs one LSU sector; the 16-byte form (``alignment=16``) is what keeps ptxas on ``LDG.E.128`` (frost-tile-dsl s11
    row 49).  ``seq_kv_len`` is the row's live key count (the padded length, <= the tensor's S_kv extent): a granule whose
    first key is at or past it is CLAMPED to the last granule that starts below it, on a granule-aligned column -- the band
    / padding mask already masks every column at or past the sequence end, so the values loaded there are never live --
    which is why the caller's contract is a tensor extent that is whole granules (``S_kv % G == 0``) and a 16-byte-aligned
    base and batch stride (the row declines anything else typed); a clamped granule then never leaves the tensor."""
    G = additive_granule_elems(dtype)
    if n % G:
        raise ValueError(f"additive_vec_granules: n={n} must be whole {G}-element granules")
    shift = G.bit_length() - 1
    row = batch_idx.to(cutlass.Int64) * cutlass.Int64(vec_tensor.stride[0])
    base = Pointer(vec_tensor.iterator.raw_ptr(), dtype=dtype)
    # the last granule-aligned column below the live length (0 when the length is 0: the body never runs on an empty item)
    last = cute.math.max(((seq_kv_len - cutlass.Int32(1)) >> cutlass.Int32(shift)) << cutlass.Int32(shift), cutlass.Int32(0))
    granules = []
    for g in range(n // G):
        col = cute.math.min(kv_col + cutlass.Int32(g * G), last)
        granules.append((base + (row + col.to(cutlass.Int64))).load(ADDITIVE_GRANULE_BYTES, count=G))
    return granules


def additive_vec_fma_chunk(raw_chunk, granules, inv_scale, n: int, *, dtype):
    """Trace-time helper: ``s[i] = vec[i] * inv_scale + raw[i]`` for the ``n`` scores of one softmax chunk from the granules
    of :func:`additive_vec_granules` -- the bf16 / fp16 form through :func:`bias_fma_chunk` (each granule's 8 halves are 4
    packed words), the fp32 form straight into ``fma.rn.f32x2`` pairs.  Returns a ``Vector[Float32, n]``."""
    G = additive_granule_elems(dtype)
    if len(granules) * G != n:
        raise ValueError(f"additive_vec_fma_chunk: {len(granules)} granules of {G} for {n} scores")
    if dtype != cutlass.Float32:
        words = []
        for vec in granules:
            packed = vec.bitcast(cutlass.Int32)
            for j in range(G // 2):
                words.append(packed[j])
        return bias_fma_chunk(raw_chunk, words, inv_scale, n, dtype=dtype)
    elems = []
    for g, vec in enumerate(granules):
        for j in range(0, G, 2):
            i = g * G + j
            s_lo, s_hi = ffma2(vec[j], vec[j + 1], inv_scale, inv_scale, raw_chunk[i], raw_chunk[i + 1])
            elems.append(s_lo)
            elems.append(s_hi)
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)
