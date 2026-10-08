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
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm

from .pointwise import f16x2_to_f32, ffma2
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
    n_tiles: int  # BIAS_RESIDENT_TILES: how many tiles the block holds (the SMEM budget's answer)
    box_dims: tuple  # (1, 1, block_rows, granu_elems) in the (B, H, S_q, S_kv) order of the bias tensor
    layout_enum: int  # the kernel's SmemTile layout enum for the 128-B swizzle
    smem_swizzle: object  # cutlass.Swizzle(3, 4, 3)


def bias_block_geometry(cfg, *, swz_enum: dict) -> BiasBlockGeometry:
    """Pure host arithmetic (called at module import by the kernel that carries the block).

    Every term comes from the Cfg: a literal here would be the canonical silent-wrong-answer bug (the box count
    and the expect_tx must agree with the SMEM extent the softmax lanes index)."""
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
        n_tiles=cfg.BIAS_RESIDENT_TILES,
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


def bias_chunk_smem_offset(gg: BiasBlockGeometry, tile_idx, chunk: int, row) -> cutlass.Int32:
    """Trace-time helper: the LINEAR (pre-swizzle) SMEM element offset of one lane's 64-column row slice -- tile
    ``tile_idx`` (runtime, relative to the item's first kv tile), box ``chunk`` (0..boxes_per_tile-1), row ``row``
    (the lane's query row inside the block).  Box-major inside a tile, 128 bytes per row inside a box."""
    return tile_idx * cutlass.Int32(gg.tile_elems) + cutlass.Int32(chunk * gg.subtile_stride_elems) + row * cutlass.Int32(gg.granu_elems)


def load_bias_row_chunk(sBias_base, gg: BiasBlockGeometry, tile_idx, chunk: int, row):
    """Trace-time helper: one lane's 64-column bias row slice as the 32 packed 16-bit PAIRS it is stored as
    (a ``Vector[Int32, 32]``), through ONE swizzled 128-byte load (the full swizzle period: 8 x 16-byte granules,
    each XOR-remapped by the row).  Kept packed: :func:`bias_fma_chunk` unpacks each pair with a 2-op shift /
    convert right where it is consumed, which is cheaper than a 64-element ``.to(Float32)`` and keeps the
    pre-loaded fragment at 32 registers across the ``bmm1_done`` wait."""
    off = bias_chunk_smem_offset(gg, tile_idx, chunk, row)
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
    if int(bias_words.shape[0]) != n // 2:
        raise ValueError(f"bias_fma_chunk: {int(bias_words.shape[0])} bias words for {n} scores; need {n // 2}")
    elems = []
    for i in range(n // 2):
        b_lo, b_hi = f16x2_to_f32(bias_words[i], dtype=dtype)
        s_lo, s_hi = ffma2(b_lo, b_hi, inv_scale, inv_scale, raw_chunk[2 * i], raw_chunk[2 * i + 1])
        elems.append(s_lo)
        elems.append(s_hi)
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)
