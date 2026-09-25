# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 block quantization pass over ``[T, H, D]``: e4m3 codes + F8_128x4 E8M0 scale factors.

The block's MXFP8 pipeline hands the production Rubin SDPA
(``sdpa/fwd/kernels/sm107/prefill_d256_mxfp8.py``) Q/K/V as e4m3 codes plus one
E8M0 scale per 32-element block, in cuDNN's F8_128x4 swizzled order.  Q and K
are quantized ROWWISE (blocks along D, the BMM1 contraction), V COLUMNWISE
(blocks along S, the BMM2 contraction).  This is that pass, forked from the
per-tensor ``kernels/quantize.py`` (same lane mapping, same ``ld_global_v4`` /
``f16x2_to_f32`` / ``fp32_to_fp8_pack`` / ``st_global_v4`` helpers, same
dynamic-token-stride fake for the slab-slice source).

**Oracle** (bit-exact is the bar): ``test/python/sdpa/mxfp8_quant.py::quantize_to_mxfp8``
after permuting the block's ``[T, H, D]`` buffers to BHSD -- it pads S to 128, the
rowwise ``_d`` triple is Q/K, the columnwise ``_s`` triple is V, and pad rows carry
SF ``0x00``.  Scale: ``e = cvt.rp.satfinite.ue8m0x2.f32(amax * fp32(1/448))``
(ONE fp32 multiply by the exact ``0x3B124925``); data: ``e4m3_rn_satfinite(x * 2^(127-e))``.
The e8m0 helpers and the fusing abs-max tree live in ``frost/tile_dsl/pointwise.py``
(``e8m0_from_amax`` / ``e8m0_pair`` / ``abs_max_tree`` on ``fmax_f32`` -- PTX ``max.f32``, which ptxas
fuses into FMNMX3 where ``cute.math.max`` gave compare + select and zero FMNMX).

SF layout contracts (PR-B plan section 2.3, quoted; every byte below is consumed
by the SDPA's TMA descriptors, so a wrong order is numerically wrong and never an error):

* F8_128x4 atom rule -- ``mxfp8_quant._swizzle_128x4``: "Atoms (128 rows x 4 cols
  = 512 B) are laid out row-major; inside an atom, scale (r, c) lives at byte
  ``(r % 32) * 16 + (r // 32) * 4 + c``."  ``swizzle_sf_rowwise`` applies it to
  ``[..., M, K//32]``; ``swizzle_sf_columnwise`` applies it to the TRANSPOSE
  ``[K, M//32]`` and returns the storage shape.  The atom-local arithmetic is
  spelled ONCE, host and device, in ``frost/tile_dsl/sf_layout.py``
  (``sf_atom_offset`` / ``sf_atom_byte``); this kernel and the FP4 quantizer
  both import it, and the host twins below add only the per-unit atom base.
* Q/K (rowwise), consumer ``_build_sf_desc``: **tile ``(b, h, s_tile)`` = ``4*D``
  (1024 at D=256) contiguous bytes at ``((b*H + h)*n_tiles + s_tile) * 4*D``**;
  inside: ``(c//4)*512 + (s%32)*16 + ((s%128)//32)*4 + c%4`` with ``c = d//32``.
* V (columnwise), consumer ``tma_v_sf_desc``: **byte ``(b, h, s_tile, d, s)`` =
  ``(d//128) * (B*KH*n_tiles*512) + ((b*KH + h)*n_tiles + s_tile)*512 +
  ((d%128)%32)*16 + ((d%128)//32)*4 + (s%128)//32``** -- D-plane-major, the plane
  stride ``v_sf_groups*512 = B*KH*ceil(S/128)*512`` GROWS with ``B*KH*S``.
* Adapter ``_reshape_sf`` binds by STORAGE order and checks only
  ``numel == b*h*n_tiles*sf_smem_size`` -- the ``torch.equal`` test against the oracle
  is the only guard on the byte order.

Grid ``(B*H, ceil(S/128))``: one CTA owns one SF unit (Q/K: one 1024-B tile = 2 atoms;
V: 2 atoms in 2 D-planes).  The SF bytes are staged in a ``4*D``-byte SMEM tile, one
``bar.sync``, then ``D/4`` lanes burst them out 16 B each.

* **Rowwise arm** (``axis="row"``): ``quantize.py``'s mapping -- a lane moves 16
  elements (two ``ld.global.v4`` of bf16 in, one ``st.global.v4`` of e4m3 out), so
  ``D/16`` lanes cover a row (16 = half a warp at D=256), and a lane PAIR
  (``shfl.bfly(1)``) shares one 32-element block.  Both lanes of the pair compute the
  scale (one ``cvt``, cheaper than a broadcast); the even lane owns the SF byte.
* **Columnwise arm** (``axis="col"``): a warp owns 32 consecutive tokens x 64 d
  (lane = 2 adjacent d), so per token the warp reads 128 B contiguous with one
  ``ld.global.b32`` and writes 64 B with ``st.global.b16`` -- 64 live fp32 + 2 amax
  per lane, one ``e8m0_pair`` cvt for both.  The 2-byte stores are the accepted v1
  cost (about a third of a ``v4`` store's efficiency); measure before changing.
* **Tail rows** (``s >= S`` in the last tile): the load is clamped to a valid row
  and the value zeroed (rowwise: the block amax is zeroed), the data store is
  skipped, and the SF byte is WRITTEN as ``0x00`` -- an unwritten byte would be
  E8M0 NaN under the SDPA's whole-tile SF TMA, and a zero block quantizes to
  ``0x00`` in both ``cvt.rp.satfinite.ue8m0x2`` and the oracle's ``e8m0_ceil``.

Traffic: ``2 B read + 1 B write + 1/32 B SF`` per element (``moved_bytes``).  Needs
sm_100+ (``cvt.rp.satfinite.ue8m0x2.f32``); the numerics tests run on Rubin, the
shape algebra and the sm_107a trace-compile run anywhere.
"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
import torch

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.device import current_device
from cudnn.frost.tile_dsl.pointwise import abs_max_tree, e8m0_from_amax, e8m0_pair, f16x2_to_f32, fmax_f32, fp32_to_fp8_pack, fp32_to_fp8x2
from cudnn.frost.tile_dsl.sf_layout import SF_ATOM_BYTES, SF_ATOM_COLS, SF_ATOM_ROWS, sf_atom_byte, sf_atom_offset
from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global_v4

from .qk_norm_rope import fake_rowmajor_dynamic_token_stride

AXIS_ROW = "row"  # Q/K: 32-element blocks along D (the BMM1 contraction)
AXIS_COL = "col"  # V:   32-element blocks along S (the BMM2 contraction)
AXES = (AXIS_ROW, AXIS_COL)

SF_BLOCK = 32  # elements per E8M0 scale
SF_TILE_ROWS = SF_ATOM_ROWS  # 128 rows of an F8_128x4 atom == the SDPA's Q / KV tile height
# SF_ATOM_BYTES / SF_ATOM_COLS / SF_ATOM_ROWS are defined in tile_dsl.sf_layout and RE-EXPORTED from here (the
# import above is load-bearing: `from quantize_mxfp8 import SF_ATOM_BYTES` callers exist in the tests).
SF_BURST_BYTES = 16  # one st.global.v4 per burst lane

ELEMS_PER_LANE = 16  # rowwise: what fp32_to_fp8_pack converts in one call (32 B of bf16 in, 16 B of e4m3 out)
SRC_BYTES_PER_LANE = ELEMS_PER_LANE * 2
DST_BYTES_PER_LANE = ELEMS_PER_LANE * 1
LOADS_PER_LANE = SRC_BYTES_PER_LANE // 16
WARP = 32
COL_D_PER_LANE = 2  # columnwise: one ld.global.b32 = two adjacent bf16 per token
COL_D_PER_UNIT = WARP * COL_D_PER_LANE  # 64 d per warp unit
COL_TOKENS_PER_UNIT = SF_BLOCK  # one 32-token block per warp unit
COL_TOKEN_BLOCKS = SF_TILE_ROWS // COL_TOKENS_PER_UNIT  # 4 per tile

DEFAULT_THREADS_PER_CTA = 256
MAX_THREADS_PER_CTA = 1024  # the hardware CTA cap; past it validation would pass and the launch would fail untyped
COMPILE_OPTIONS = "--enable-tvm-ffi"

_FAKE_STREAM = None


# ---------------------------------------------------------------------------
# Shape algebra (pure Python; the tests pin it against the torch oracle)
# ---------------------------------------------------------------------------


def n_sf_tiles(seq_len: int) -> int:
    """SF tiles per (b, h): ``ceil(S / 128)`` -- the oracle pads S to 128 the same way."""
    return (seq_len + SF_TILE_ROWS - 1) // SF_TILE_ROWS


def sf_tile_bytes(d: int) -> int:
    """SF bytes per ``(b, h, s_tile)`` unit: ``128 * D/32`` rowwise == ``D/128`` atoms of 512 B columnwise == ``4*D`` (1024 at D=256)."""
    return SF_TILE_ROWS * d // SF_BLOCK


def sf_bytes(batch: int, h: int, seq_len: int, d: int) -> int:
    """The flat SF buffer size the SDPA adapter expects: ``B*H*ceil(S/128)*sf_tile_bytes(d)``."""
    return batch * h * n_sf_tiles(seq_len) * sf_tile_bytes(d)


def lanes_per_row(d: int) -> int:
    """Rowwise: lanes cooperating on one head row, 16 elements each, capped at a warp."""
    return min(WARP, d // ELEMS_PER_LANE)


def chunks_per_lane(d: int) -> int:
    return d // (lanes_per_row(d) * ELEMS_PER_LANE)


def sf_byte_rowwise(b: int, h: int, s: int, d_idx: int, *, n_heads: int, n_tiles: int, d: int) -> int:
    """Absolute byte of scale ``(b, h, s, d_idx // 32)`` in the flat Q/K SF buffer (plan section 2.3, Q/K row)."""
    c = d_idx // SF_BLOCK
    r = s % SF_TILE_ROWS
    tile = (b * n_heads + h) * n_tiles + s // SF_TILE_ROWS
    return tile * sf_tile_bytes(d) + sf_atom_offset(r, c)


def sf_byte_columnwise(b: int, h: int, s: int, d_idx: int, *, n_heads: int, n_tiles: int, batch: int) -> int:
    """Absolute byte of scale ``(b, h, s // 32, d_idx)`` in the flat V SF buffer (plan section 2.3, V row): D-plane-major."""
    plane = d_idx // SF_TILE_ROWS
    dm = d_idx % SF_TILE_ROWS
    tile = (b * n_heads + h) * n_tiles + s // SF_TILE_ROWS
    v_sf_groups = batch * n_heads * n_tiles
    return sf_atom_byte(dm, (s % SF_TILE_ROWS) // SF_BLOCK, base=plane * (v_sf_groups * SF_ATOM_BYTES) + tile * SF_ATOM_BYTES)


def validate_shape(d: int, threads_per_cta: int, axis: str) -> None:
    """Raise ``ValueError`` on any geometry this kernel cannot address (never an assert)."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES} ('row' for Q/K, 'col' for V), got {axis!r}")
    if d <= 0 or d % SF_TILE_ROWS != 0:
        raise ValueError(f"d_head must be a positive multiple of 128 (whole F8_128x4 atoms: D/32 blocks in fours rowwise, 128 d-rows columnwise), got {d}")
    if threads_per_cta <= 0 or threads_per_cta % WARP != 0 or threads_per_cta > MAX_THREADS_PER_CTA:
        raise ValueError(f"threads_per_cta must be a positive multiple of {WARP} up to {MAX_THREADS_PER_CTA} (the CTA cap), got {threads_per_cta}")
    burst_lanes = sf_tile_bytes(d) // SF_BURST_BYTES
    if threads_per_cta < burst_lanes:
        raise ValueError(f"threads_per_cta={threads_per_cta} cannot burst the {sf_tile_bytes(d)}-byte SF tile in 16-byte lanes ({burst_lanes} needed)")
    if axis == AXIS_ROW:
        lanes = lanes_per_row(d)
        if WARP % lanes != 0:
            raise ValueError(f"d_head={d} gives {lanes} lanes/row, which must divide a warp")
        if d != lanes * ELEMS_PER_LANE * chunks_per_lane(d):
            raise ValueError(f"d_head={d} is not covered exactly by {lanes} lanes x {chunks_per_lane(d)} chunks x {ELEMS_PER_LANE} elements")
        if threads_per_cta % lanes != 0:
            raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of the {lanes} lanes per row")
        rows_per_pass = threads_per_cta // lanes
        if SF_TILE_ROWS % rows_per_pass != 0:
            raise ValueError(f"threads_per_cta={threads_per_cta} covers {rows_per_pass} rows per pass, which must divide the 128-row tile")
    else:
        units = COL_TOKEN_BLOCKS * (d // COL_D_PER_UNIT)
        warps = threads_per_cta // WARP
        if units % warps != 0:
            raise ValueError(f"columnwise d_head={d} has {units} (32-token x 64-d) units per tile, not a multiple of the {warps} warps")


def moved_bytes(t: int, h: int, d: int, src_elem_bytes: int = 2) -> int:
    """HBM traffic of one launch: the bf16/f16 read, the 1-byte e4m3 write and the 1/32-byte SF write per element."""
    return t * h * d * (src_elem_bytes + 1) + t * h * d // SF_BLOCK


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def st_global_b16(addr, value):
    """16-bit global store of a ``Uint16`` (two packed e4m3 codes)."""
    nvvm.inline_ptx("st.global.b16 [$0], $1;", read_only_args=[addr, value])


@cute.kernel
def frost_quantize_mxfp8(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token stride (slab slice or compact), head stride D
    mDst: cute.Tensor,  # [T, H, D] e4m3, own token stride (compact in the block), head stride D
    mSf: cute.Tensor,  # [B*H*ceil(S/128)*4*D] uint8, F8_128x4 order per the module docstring
    seq_len: cutlass.Int32,
    n_tiles: cutlass.Int32,  # ceil(S/128) == gridDim.y
    v_sf_groups: cutlass.Int32,  # B*H*n_tiles: the columnwise D-plane stride in atoms (unused rowwise)
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    axis_col: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
) -> None:
    tile_bytes = cutlass.const_expr(sf_tile_bytes(d))
    burst_lanes = cutlass.const_expr(tile_bytes // SF_BURST_BYTES)
    sSF = cutlass.Array(cutlass.Uint8, tile_bytes, alignment=16, space=cutlass.AddressSpace.smem)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    bh = cutlass.Int32(cute.arch.block_idx()[0])
    s_tile = cutlass.Int32(cute.arch.block_idx()[1])
    b = bh // cutlass.Int32(h)
    head = bh % cutlass.Int32(h)
    s0 = s_tile * cutlass.Int32(SF_TILE_ROWS)
    tok0 = b * seq_len  # token index of this batch's row 0
    last = seq_len - cutlass.Int32(1)  # tail rows clamp their LOADS here (never past the tensor)
    src_tok_stride = cutlass.Int64(mSrc.stride[0]) * cutlass.Int64(2)
    dst_tok_stride = cutlass.Int64(mDst.stride[0])
    src_base = mSrc.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[1]) * cutlass.Int64(2)
    dst_base = mDst.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mDst.stride[1])

    if cutlass.const_expr(not axis_col):
        # ---- ROWWISE (Q/K): 32-element blocks along D --------------------------------
        lanes = cutlass.const_expr(lanes_per_row(d))
        chunks = cutlass.const_expr(chunks_per_lane(d))
        rows_per_pass = cutlass.const_expr(threads_per_cta // lanes)
        passes = cutlass.const_expr(SF_TILE_ROWS // rows_per_pass)
        lane = tidx % cutlass.Int32(lanes)
        grp = tidx // cutlass.Int32(lanes)
        even = (lane & cutlass.Int32(1)) == cutlass.Int32(0)
        src_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(SRC_BYTES_PER_LANE)
        dst_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(DST_BYTES_PER_LANE)
        for p in cutlass.range_constexpr(passes):
            r = cutlass.Int32(p * rows_per_pass) + grp  # row within the 128-row tile
            s = s0 + r
            valid = s < seq_len
            s_ld = s if valid else last
            tok = tok0 + s_ld
            src_row = src_base + tok.to(cutlass.Int64) * src_tok_stride
            dst_row = dst_base + tok.to(cutlass.Int64) * dst_tok_stride
            for c in cutlass.range_constexpr(chunks):
                base = src_row + cutlass.Int64((c * lanes) * SRC_BYTES_PER_LANE) + src_lane_off
                vals = []
                for j in cutlass.range_constexpr(LOADS_PER_LANE):
                    for w in ld_global_v4(base + cutlass.Int64(j * 16), cutlass.Int32):
                        lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                        vals.append(lo)
                        vals.append(hi)
                # The lane pair (lane, lane^1) shares one 32-element block: bfly(1) completes the block amax.
                amax = abs_max_tree(vals)
                partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax, cutlass.Int32(1), 31, kind=nvvm.Shfl.BFLY))
                amax = fmax_f32(amax, partner)
                amax = amax if valid else cutlass.Float32(0.0)  # tail row: SF 0x00, no data store
                rcp, sf_byte = e8m0_from_amax(amax)
                if valid:
                    scaled = []
                    for i in cutlass.range_constexpr(ELEMS_PER_LANE):
                        scaled.append(vals[i] * rcp)
                    packed = fp32_to_fp8_pack(scaled, dtype=cutlass.Float8E4M3FN)
                    st_global_v4(
                        dst_row + cutlass.Int64((c * lanes) * DST_BYTES_PER_LANE) + dst_lane_off, [packed[0], packed[1], packed[2], packed[3]], cutlass.Int32
                    )
                # SF SMEM byte for block column c_idx = d//32 of row r: (c//4)*512 + (r%32)*16 + (r//32)*4 + c%4 (sf_layout).
                c_idx = (cutlass.Int32(c * lanes) + lane) // cutlass.Int32(2)
                sf_off = sf_atom_offset(r, c_idx)
                if even:
                    sSF.store(sf_byte.to(cutlass.Uint8), sf_off)
    else:
        # ---- COLUMNWISE (V): 32-token blocks along S -----------------------------------
        warps = cutlass.const_expr(threads_per_cta // WARP)
        slices = cutlass.const_expr(d // COL_D_PER_UNIT)
        units_per_warp = cutlass.const_expr(COL_TOKEN_BLOCKS * slices // warps)
        lane = tidx % cutlass.Int32(WARP)
        warp = tidx // cutlass.Int32(WARP)
        for i in cutlass.range_constexpr(units_per_warp):
            u = warp * cutlass.Int32(units_per_warp) + cutlass.Int32(i)
            tb = u // cutlass.Int32(slices)  # 32-token block within the tile
            ds = u % cutlass.Int32(slices)  # 64-d slice
            d0 = ds * cutlass.Int32(COL_D_PER_UNIT) + lane * cutlass.Int32(COL_D_PER_LANE)
            s_blk = s0 + tb * cutlass.Int32(COL_TOKENS_PER_UNIT)
            n_valid = seq_len - s_blk  # tokens t < n_valid are in range (may be <= 0 or >= 32)
            # Token addresses walk INCREMENTALLY (one 64-bit add per token, not a 64-bit multiply each);
            # a tail token's load is redirected to the batch's last row and its word zeroed.
            src_addr = src_base + (tok0 + s_blk).to(cutlass.Int64) * src_tok_stride + d0.to(cutlass.Int64) * cutlass.Int64(2)
            src_last = src_base + (tok0 + last).to(cutlass.Int64) * src_tok_stride + d0.to(cutlass.Int64) * cutlass.Int64(2)
            dst_addr = dst_base + (tok0 + s_blk).to(cutlass.Int64) * dst_tok_stride + d0.to(cutlass.Int64)
            words = []
            valids = []
            dsts = []
            for t in cutlass.range_constexpr(COL_TOKENS_PER_UNIT):
                valid = cutlass.Int32(t) < n_valid  # warp-uniform
                w = ld_global(src_addr if valid else src_last, cutlass.Int32)
                words.append(w if valid else cutlass.Int32(0))  # tail token: contributes 0 to the block amax
                valids.append(valid)
                dsts.append(dst_addr)
                src_addr = src_addr + src_tok_stride
                dst_addr = dst_addr + dst_tok_stride
            los = []
            his = []
            for t in cutlass.range_constexpr(COL_TOKENS_PER_UNIT):
                lo, hi = f16x2_to_f32(words[t], dtype=mSrc.element_type)
                los.append(lo)
                his.append(hi)
            rcp0, rcp1, sf_pair = e8m0_pair(abs_max_tree(los), abs_max_tree(his))
            for t in cutlass.range_constexpr(COL_TOKENS_PER_UNIT):
                if valids[t]:
                    st_global_b16(dsts[t], fp32_to_fp8x2(los[t] * rcp0, his[t] * rcp1))
            # SF SMEM bytes: plane p = d//128 at p*512 + ((d%128)%32)*16 + ((d%128)//32)*4 + tb (sf_layout); d0+1 sits 16 B after d0.
            plane = d0 // cutlass.Int32(SF_TILE_ROWS)
            dm = d0 % cutlass.Int32(SF_TILE_ROWS)
            sf_off = sf_atom_byte(dm, tb, base=plane * cutlass.Int32(SF_ATOM_BYTES))
            sSF.store((sf_pair & cutlass.Int32(0xFF)).to(cutlass.Uint8), sf_off)
            sSF.store(((sf_pair >> 8) & cutlass.Int32(0xFF)).to(cutlass.Uint8), sf_off + cutlass.Int32(16))

    # ---- SF burst: the whole tile leaves SMEM as 16-B lanes ---------------------------
    nvvm.barrier_cta_sync()
    if tidx < cutlass.Int32(burst_lanes):
        smem_off = tidx * cutlass.Int32(SF_BURST_BYTES)
        words4 = sSF.load(smem_off, vector_size=SF_BURST_BYTES, alignment=16).bitcast(cutlass.Int32)
        tile_idx = (bh * n_tiles + s_tile).to(cutlass.Int64)
        sf_base = mSf.iterator.toint()
        if cutlass.const_expr(not axis_col):
            # Q/K: the tile is 4*D contiguous bytes.
            gaddr = sf_base + tile_idx * cutlass.Int64(tile_bytes) + smem_off.to(cutlass.Int64)
        else:
            # V: plane p (512 B, lanes p*32..p*32+31) lands p * v_sf_groups atoms away.
            burst_plane = (tidx // cutlass.Int32(SF_ATOM_BYTES // SF_BURST_BYTES)).to(cutlass.Int64)
            burst_within = ((tidx % cutlass.Int32(SF_ATOM_BYTES // SF_BURST_BYTES)) * cutlass.Int32(SF_BURST_BYTES)).to(cutlass.Int64)
            gaddr = (
                sf_base + burst_plane * (v_sf_groups.to(cutlass.Int64) * cutlass.Int64(SF_ATOM_BYTES)) + tile_idx * cutlass.Int64(SF_ATOM_BYTES) + burst_within
            )
        st_global_v4(gaddr, [words4[0], words4[1], words4[2], words4[3]], cutlass.Int32)


@cute.jit
def quantize_mxfp8_launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    sf: cute.Tensor,
    seq_len: cutlass.Int32,
    n_tiles: cutlass.Int32,
    v_sf_groups: cutlass.Int32,
    n_bh: cutlass.Int32,
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    axis_col: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_quantize_mxfp8(src, dst, sf, seq_len, n_tiles, v_sf_groups, h, d, axis_col, threads_per_cta).launch(
        grid=(n_bh, n_tiles, 1), block=(threads_per_cta, 1, 1), stream=stream
    )


# ---------------------------------------------------------------------------
# Host API (frozen in the PR-B plan, section 2.5)
# ---------------------------------------------------------------------------

compiled_cache = {}


@dataclass(frozen=True)
class QuantizeMxfp8Recipe:
    """Build-time facts of one MXFP8 quantize launch; ``batch`` / ``seq_len`` ride in at run time."""

    dtype_in: object  # torch dtype of the source (bf16 / f16)
    h: int
    d: int
    axis: str  # "row" (Q/K: 32-blocks along D) | "col" (V: 32-blocks along S)
    compiled: object = None
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA


def compile_quantize_mxfp8(
    *, dtype_in, h: int, d: int, axis: str, threads_per_cta: int = DEFAULT_THREADS_PER_CTA, compile_options: str = COMPILE_OPTIONS
) -> QuantizeMxfp8Recipe:
    """Build from SHAPES ALONE -- no allocation, no launch.  E4M3 codes + E8M0 SF only.

    ``compile_options`` is a dev knob: ``"--enable-tvm-ffi --gpu-arch sm_107a"`` trace-compiles for
    Rubin on any box (the SASS spill check); production leaves the default.
    """
    global _FAKE_STREAM
    validate_shape(d, threads_per_cta, axis)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"quantize_mxfp8 serves bf16/f16 sources only, got {dtype_in}")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    axis_col = axis == AXIS_COL
    key = (str(dtype_in), int(h), int(d), axis, int(threads_per_cta), compile_options, current_device())
    if key not in compiled_cache:
        tok = cute.sym_int()
        # Source: a column slice of the projection slab (token stride N_qkvg) or compact;
        # destination: compact in the block but kept symbolic so one artifact serves both.
        src = fake_rowmajor_dynamic_token_stride(dtype_in, tok, h, d)
        dst = cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(tok, h, d), stride=(cute.sym_int(), d, 1), assumed_align=16)
        sf = cute.runtime.make_fake_tensor(dtype=_convert_to_cutlass_data_type(torch.uint8), shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
        compiled_cache[key] = cute.compile(
            quantize_mxfp8_launch,
            src,
            dst,
            sf,
            cutlass.Int32(0),  # seq_len     ) runtime; the zeros pin the TYPE only
            cutlass.Int32(0),  # n_tiles     )
            cutlass.Int32(0),  # v_sf_groups )
            cutlass.Int32(0),  # n_bh        )
            int(h),
            int(d),
            bool(axis_col),
            int(threads_per_cta),
            _FAKE_STREAM,
            options=compile_options,
        )
    return QuantizeMxfp8Recipe(dtype_in=dtype_in, h=int(h), d=int(d), axis=axis, compiled=compiled_cache[key], threads_per_cta=int(threads_per_cta))


def run_quantize_mxfp8(r: QuantizeMxfp8Recipe, src: torch.Tensor, dst: torch.Tensor, sf: torch.Tensor, *, batch: int, seq_len: int, stream) -> None:
    """Launch.  ``src`` ``[T, H, D]`` bf16/f16 (strided ok), ``dst`` ``[T, H, D]`` ``float8_e4m3fn`` compact,
    ``sf`` uint8 with ``numel == B*H*ceil(S/128)*4*D`` (any shape, contiguous), ``T == batch * seq_len``.

    Cheap host checks only; every one of them guards a wild write or a silent wrong
    answer at the tvm-ffi boundary, which reports neither.
    """
    if r.compiled is None:
        raise ValueError("recipe was not built by compile_quantize_mxfp8")
    if src.dtype != r.dtype_in:
        raise ValueError(f"src is {src.dtype} but this artifact was compiled for {r.dtype_in}")
    if dst.dtype != torch.float8_e4m3fn:
        raise ValueError(f"dst must be torch.float8_e4m3fn, got {dst.dtype}")
    for name, ten in (("src", src), ("dst", dst)):
        if ten.ndim != 3 or int(ten.shape[1]) != r.h or int(ten.shape[2]) != r.d:
            raise ValueError(f"{name} must be [T, H={r.h}, D={r.d}], got {tuple(ten.shape)}")
        if ten.stride(2) != 1 or ten.stride(1) != r.d:
            raise ValueError(f"{name} must have head stride D={r.d} and element stride 1 (a column slice of the slab or compact), got strides {ten.stride()}")
    if batch <= 0 or seq_len <= 0:
        raise ValueError(f"batch and seq_len must be positive, got batch={batch} seq_len={seq_len}")
    t = int(src.shape[0])
    if t != batch * seq_len or int(dst.shape[0]) != t:
        raise ValueError(f"T must equal batch*seq_len for src and dst: src T={t}, dst T={int(dst.shape[0])}, batch*seq_len={batch * seq_len}")
    if (src.stride(0) * 2) % 16 or dst.stride(0) % 16:
        raise ValueError(f"token strides must keep every row 16-byte aligned: src {src.stride(0)} elems (bf16/f16), dst {dst.stride(0)} elems (fp8)")
    for name, ten in (("src", src), ("dst", dst)):
        if not ten.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got device {ten.device}")
        if ten.data_ptr() % 16:
            raise ValueError(
                f"{name} must be 16-byte aligned (its rows move as 16-byte vectors); a slab column offset that is not a multiple of 16 bytes is not"
            )
    if sf.dtype != torch.uint8:
        raise ValueError(f"sf must be torch.uint8 (E8M0 bytes in F8_128x4 order), got {sf.dtype}")
    if not sf.is_contiguous() or not sf.is_cuda:
        raise ValueError("sf must be a contiguous CUDA tensor")
    need = sf_bytes(batch, r.h, seq_len, r.d)
    if sf.numel() != need:
        raise ValueError(f"sf must hold B*H*ceil(S/128)*{sf_tile_bytes(r.d)} = {need} bytes for batch={batch} H={r.h} S={seq_len} D={r.d}, got {sf.numel()}")
    if sf.data_ptr() % 16:
        raise ValueError("sf must be 16-byte aligned (the SF tile leaves SMEM as 16-byte bursts)")
    if not (src.device == dst.device == sf.device):
        raise ValueError(f"src, dst and sf must live on one device, got {src.device}, {dst.device}, {sf.device}")
    tiles = n_sf_tiles(seq_len)
    n_bh = batch * r.h
    r.compiled(
        src,
        dst,
        sf.view(-1),
        cutlass.Int32(seq_len),
        cutlass.Int32(tiles),
        cutlass.Int32(n_bh * tiles),
        cutlass.Int32(n_bh),
        cuda.CUstream(int(stream)),
    )


frost_quantize_mxfp8.set_name_prefix("cudnn", remove_cutlass_symbol=True)
