# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP4 block quantization of the gated O: e2m1 codes (two per byte) + the out-projection GEMM's F8_128x4 scale blob.

The block's fp4 output mode hands the FROST block-scale GEMM (``kernels/proj_gemm.py``,
``block_scale=True`` with both operands fp4) the gated attention output ``O`` ``[T, H_q, D]``
as a ``[T, K = H_q*D]`` matrix of e2m1 codes plus one scale per ``block`` elements along K,
in cuDNN's F8_128x4 order over ``(rows=T, K)`` -- the SAME blob ``proj_gemm.sf_blob_bytes``
sizes and ``_sf_view`` binds, so the out projection reads what this kernel writes with no
re-layout.  Two formats, one kernel (``const_expr`` arms):

* **NVFP4** -- ``block = 16``, scale ``e4m3``: ``s = e4m3_rn_satfinite(max(amax * fp32(1/6), 2^-9))``
  (``pointwise.e4m3_scale_from_amax``; the e4m3 min-subnormal floor keeps an all-zero block's
  scale ``0x01`` instead of ``0`` -> no infinite encode scale), codes ``e2m1_rne(x / s)`` with
  PTX ``div.rn.f32`` (``pointwise.div_rn_f32``): a reciprocal-multiply lands one fp32 ulp off on
  exact e2m1 midpoints (``0.5859375 / 0.46875 == 1.25``) and flips a whole code.
* **MXFP4** -- ``block = 32``, scale ``E8M0``: ``e = cvt.rp.satfinite.ue8m0x2.f32(amax * fp32(1/6))``
  (``pointwise.e8m0_from_amax(inv_max=opaque_fp4_max_rcp())``), codes ``e2m1_rne(x * 2^(127-e))``
  (the exact power-of-two reciprocal, ``e8m0_rcp``).  Amax ``0`` -> ``0x00`` -> codes ``0``.

Both multiply the block amax by the fp32 rounding of ``1/6`` (``0x3E2AAAAB``, e2m1's max is 6) --
ONE fp32 multiply, never ``amax / 6`` (one ulp apart on a third of all inputs, i.e. across
e4m3 / e8m0 rounding corners).  **Oracle** (bit-exact is the bar):
``test/python/fe_api/gated_attention_block/gated_block_reference.py::fp4_quantize_rowwise_2d`` +
``mx_swizzle_sf_rowwise_padded(e, block)``; the codes are compared as uint8 (torch 2.13 cannot cast
to or from ``float4_e2m1fn_x2``) and the blob byte for byte.

**Geometry** -- ``quantize_mxfp8.py``'s rowwise lane mapping, one CTA per (128-row tile, head):
grid ``(ceil(T/128), H_q)``, 256 threads; a lane moves 16 elements (two ``ld.global.v4`` of
bf16/f16 in, one ``st.global.v2.b32`` = 8 bytes of codes out), so ``D/16`` lanes cover a head row
(16 at D=256 = half a warp), 16 rows per pass, 8 passes.  NVFP4: a lane OWNS its 16-element block
(no shuffle).  MXFP4: the lane pair ``(lane, lane^1)`` shares one 32-element block, ``shfl.bfly(1)``
completes the amax, both lanes compute the scale (one ``cvt``, cheaper than a broadcast), the even
lane owns the SF byte.

**SF layout** -- the GEMM's PADDED F8_128x4 blob over ``(rows_pad = ceil(T/128)*128, sf_k_pad =
ceil((K/block)/4)*4)``: atoms of 128 rows x 4 scale columns (512 B) row-major over the
``(rows_pad/128) x (sf_k_pad/4)`` grid, inside an atom ``(r % 32) * 16 + (r // 32) * 4 + c % 4``
(``frost/tile_dsl/sf_layout.py``, shared with the MXFP8 quantizer).  ``D % (4*block) == 0`` (typed)
makes one head's ``D/block`` scale columns ``D/(4*block)`` WHOLE atoms (4 nvfp4 / 2 mxfp4 at
D=256), so a CTA's ``128 * D/block`` SF bytes are CONTIGUOUS in the blob at atom index
``row_tile * sf_k4 + head * D/(4*block)`` with ``sf_k4 = sf_k_pad / 4`` (a runtime ``Int32``, so one
artifact serves every ``T``).  The bytes are staged in SMEM with ``sf_atom_offset(r, c_local)``, one
``bar.sync``, then ``tile_bytes/16`` lanes burst them out 16 B each.  Because ``K/block`` is a
multiple of 4 there are no pad COLUMNS; the pad ROWS are the tail of the last row tile.

**Tail rows** (``row >= T`` in the last tile): the load is clamped to row ``T-1``, the data store is
skipped, and the SF byte is WRITTEN ``0x00`` -- the padded-blob contract (a stale workspace byte
would be an e4m3 / E8M0 value the GEMM's padded M rows multiply by).  A VALID all-zero row (a dead
ragged entry) is the ordinary path: NVFP4 SF ``0x01`` (the floor) with codes ``0``, MXFP4 SF ``0x00``.

Traffic: ``2 B read + 0.5 B write + 1/block B SF`` per element (``moved_bytes``).  Needs sm_100a+
(``cvt.rn.satfinite.e2m1x2.f32``, ``cvt.rp.satfinite.ue8m0x2.f32``); the numerics tests run on Rubin,
the shape algebra and the sm_107a trace-compile run anywhere.  The source is COMPACT ``[T, H, D]`` (under
quantization the block's gated O is; a strided source is a typed decline, not a silent repack).
"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
import torch

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.device import current_device
from cudnn.frost.tile_dsl.pointwise import (
    E2M1_MAX_RCP_BITS,
    abs_max_tree,
    div_rn_f32,
    e4m3_scale_from_amax,
    e8m0_from_amax,
    f16x2_to_f32,
    fmax_f32,
    fp32_to_fp4_pack,
    opaque_fp4_max_rcp,
)
from cudnn.frost.tile_dsl.sf_layout import SF_ATOM_BYTES, SF_ATOM_COLS, SF_ATOM_ROWS, sf_atom_offset
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global_v2, st_global_v4

from .proj_gemm import sf_blob_bytes, sf_padded_dims

# The two served formats: name -> (elements per scale, scale is e4m3 [else E8M0]).  Exactly the
# kernel_registry pairs (fp4_e2m1 x fp8_e4m3 at block 16; fp4_e2m1 x fp8_e8m0 at block 32); anything
# else is unspellable here.  The block's ``Fp4Format`` enum members are NAMED after these keys.
FMT_NVFP4 = "nvfp4"
FMT_MXFP4 = "mxfp4"
FORMATS = {FMT_NVFP4: (16, True), FMT_MXFP4: (32, False)}
# The torch storage dtypes the buffers are VIEWED as -- ``None`` on a torch without them (the same ``getattr``
# guard ``api.py`` / ``proj_gemm.py`` use), so the module imports everywhere and ``check_torch_fp4_dtypes``
# turns the gap into a typed decline at compile / run, never an ``AttributeError`` at import.
SF_TORCH_DTYPE_NAMES = {FMT_NVFP4: "float8_e4m3fn", FMT_MXFP4: "float8_e8m0fnu"}
SF_TORCH_DTYPES = {k: getattr(torch, v, None) for k, v in SF_TORCH_DTYPE_NAMES.items()}  # the blob's scale dtype (bytes either way)
FP4_STORAGE_DTYPE_NAME = "float4_e2m1fn_x2"
FP4_STORAGE_DTYPE = getattr(torch, FP4_STORAGE_DTYPE_NAME, None)  # two e2m1 codes per byte, low nibble = even k
FP4_MAX_RCP_BITS = E2M1_MAX_RCP_BITS  # fp32(1/6) = 0x3E2AAAAB: the ONE constant both scale rules (and the oracle) multiply the amax by

SF_TILE_ROWS = SF_ATOM_ROWS  # 128 rows per CTA == one atom row band
SF_BURST_BYTES = 16  # one st.global.v4 per burst lane
CODES_PER_BYTE = 2

ELEMS_PER_LANE = 16  # what fp32_to_fp4_pack converts in one call (32 B of bf16 in, 8 B of e2m1 out)
SRC_BYTES_PER_LANE = ELEMS_PER_LANE * 2
DST_BYTES_PER_LANE = ELEMS_PER_LANE // CODES_PER_BYTE
LOADS_PER_LANE = SRC_BYTES_PER_LANE // 16
WARP = 32

DEFAULT_THREADS_PER_CTA = 256
MAX_THREADS_PER_CTA = 1024  # the hardware CTA cap; past it validation would pass and the launch would fail untyped
COMPILE_OPTIONS = "--enable-tvm-ffi"

_FAKE_STREAM = None


# ---------------------------------------------------------------------------
# Format + shape algebra (pure Python; the tests pin it against the torch oracle)
# ---------------------------------------------------------------------------


def fp4_format(fmt) -> tuple:
    """``"nvfp4"`` / ``"mxfp4"``, or an enum member NAMED so (the block's ``Fp4Format``) -> ``(name, block, sf_e4m3)``.

    An object that also carries a ``block_size`` must agree with the table (a typed guard against a
    format enum and this kernel drifting apart)."""
    name = str(getattr(fmt, "name", fmt)).lower()
    if name not in FORMATS:
        raise ValueError(f"unknown fp4 format {fmt!r}; quantize_fp4 serves {sorted(FORMATS)} (e2m1 x e4m3/16, e2m1 x E8M0/32)")
    block, sf_e4m3 = FORMATS[name]
    declared = getattr(fmt, "block_size", None)
    if declared is not None and int(declared) != block:
        raise ValueError(f"fp4 format {name!r} declares block_size={declared} but quantize_fp4 serves it at {block}")
    return name, block, sf_e4m3


def check_torch_fp4_dtypes(fmt_name: str) -> None:
    """Typed decline when this torch lacks a storage dtype the format's buffers are viewed as (``float4_e2m1fn_x2``
    for the codes, the format's scale dtype for the blob).  Called at compile AND run so a missing dtype is a
    ``ValueError`` naming the torch version, never an untyped escape at the ``.view`` in the launcher."""
    missing = [n for n, dt in ((FP4_STORAGE_DTYPE_NAME, FP4_STORAGE_DTYPE), (SF_TORCH_DTYPE_NAMES[fmt_name], SF_TORCH_DTYPES[fmt_name])) if dt is None]
    if missing:
        raise ValueError(
            f"torch {torch.__version__} has no {' / '.join('torch.' + m for m in missing)}: quantize_fp4 ({fmt_name}) views its codes and scale "
            f"blob as those storage dtypes and needs a torch that carries them (2.13 does)"
        )


def n_row_tiles(t: int) -> int:
    """128-row tiles over ``T`` rows == ``gridDim.x`` == the blob's padded row count / 128."""
    return (t + SF_TILE_ROWS - 1) // SF_TILE_ROWS


def sf_cols(h: int, d: int, block: int) -> int:
    """Scale columns of the ``[T, K = H*D]`` matrix: ``K / block``."""
    return h * d // block


def sf_k4(h: int, d: int, block: int) -> int:
    """Atoms per 128-row band of the PADDED blob: ``sf_padded_dims(., K, block)[1] / 4`` (``ceil((K/block)/4)``)."""
    return sf_padded_dims(SF_TILE_ROWS, h * d, block)[1] // SF_ATOM_COLS


def cta_sf_bytes(d: int, block: int) -> int:
    """SF bytes one CTA (128 rows x one head) writes: ``128 * D/block`` == ``D/(4*block)`` whole 512-B atoms."""
    return SF_TILE_ROWS * d // block


def cta_sf_atoms(d: int, block: int) -> int:
    return cta_sf_bytes(d, block) // SF_ATOM_BYTES


def sf_byte(row: int, k: int, *, h: int, d: int, block: int) -> int:
    """Absolute byte of scale ``(row, k // block)`` in the padded F8_128x4 blob over ``(rows=T, K=H*D)``:
    ``(row//128 * sf_k4) * 512 + sf_atom_offset(row % 128, k // block)`` -- the host twin of the device
    arithmetic (band base + the shared atom rule), pinned against the oracle's ``mx_swizzle_sf_rowwise_padded``."""
    return (row // SF_TILE_ROWS) * sf_k4(h, d, block) * SF_ATOM_BYTES + sf_atom_offset(row % SF_TILE_ROWS, k // block)


def lanes_per_row(d: int) -> int:
    """Lanes cooperating on one head row, 16 elements each, capped at a warp."""
    return min(WARP, d // ELEMS_PER_LANE)


def chunks_per_lane(d: int) -> int:
    return d // (lanes_per_row(d) * ELEMS_PER_LANE)


def validate_shape(d: int, threads_per_cta: int, block: int) -> None:
    """Raise ``ValueError`` on any geometry this kernel cannot address (never an assert)."""
    if block not in {b for b, _ in FORMATS.values()}:
        raise ValueError(f"block must be 16 (nvfp4) or 32 (mxfp4), got {block!r}")
    atom_elems = SF_ATOM_COLS * block
    if d <= 0 or d % atom_elems != 0:
        raise ValueError(f"d_head must be a positive multiple of 4*block = {atom_elems} (whole F8_128x4 atoms per head at block {block}), got {d}")
    if threads_per_cta <= 0 or threads_per_cta % WARP != 0 or threads_per_cta > MAX_THREADS_PER_CTA:
        raise ValueError(f"threads_per_cta must be a positive multiple of {WARP} up to {MAX_THREADS_PER_CTA} (the CTA cap), got {threads_per_cta}")
    burst_lanes = cta_sf_bytes(d, block) // SF_BURST_BYTES
    if threads_per_cta < burst_lanes:
        raise ValueError(f"threads_per_cta={threads_per_cta} cannot burst the {cta_sf_bytes(d, block)}-byte SF tile in 16-byte lanes ({burst_lanes} needed)")
    lanes = lanes_per_row(d)
    if lanes < 1 or WARP % lanes != 0:
        raise ValueError(f"d_head={d} gives {lanes} lanes/row, which must divide a warp")
    if block == 32 and lanes % 2 != 0:
        raise ValueError(f"d_head={d} gives {lanes} lanes/row; the mxfp4 lane PAIR needs an even count")
    if d != lanes * ELEMS_PER_LANE * chunks_per_lane(d):
        raise ValueError(f"d_head={d} is not covered exactly by {lanes} lanes x {chunks_per_lane(d)} chunks x {ELEMS_PER_LANE} elements")
    if threads_per_cta % lanes != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of the {lanes} lanes per row")
    rows_per_pass = threads_per_cta // lanes
    if SF_TILE_ROWS % rows_per_pass != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} covers {rows_per_pass} rows per pass, which must divide the 128-row tile")


def moved_bytes(t: int, h: int, d: int, block: int, src_elem_bytes: int = 2) -> int:
    """HBM traffic of one launch: the bf16/f16 read, the half-byte code write and the padded SF blob."""
    return t * h * d * src_elem_bytes + t * h * d // CODES_PER_BYTE + sf_blob_bytes(t, h * d, block)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@cute.kernel
def frost_quantize_fp4(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16 COMPACT (token stride H*D, head stride D)
    mDst: cute.Tensor,  # [T, H*D/2] uint8: e2m1 codes, two per byte (low nibble = even k)
    mSf: cute.Tensor,  # [sf_blob_bytes(T, H*D, block)] uint8, the padded F8_128x4 blob (module docstring)
    n_rows: cutlass.Int32,  # T
    sf_k4: cutlass.Int32,  # atoms per 128-row band of the padded blob == ceil((H*D/block)/4)
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    block: cutlass.Constexpr[int],
    sf_e4m3: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
) -> None:
    tile_bytes = cutlass.const_expr(cta_sf_bytes(d, block))
    burst_lanes = cutlass.const_expr(tile_bytes // SF_BURST_BYTES)
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(chunks_per_lane(d))
    rows_per_pass = cutlass.const_expr(threads_per_cta // lanes)
    passes = cutlass.const_expr(SF_TILE_ROWS // rows_per_pass)
    lanes_per_block = cutlass.const_expr(block // ELEMS_PER_LANE)  # 1 (nvfp4) or 2 (mxfp4)
    sSF = cutlass.Array(cutlass.Uint8, tile_bytes, alignment=16, space=cutlass.AddressSpace.smem)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    row_tile = cutlass.Int32(cute.arch.block_idx()[0])
    head = cutlass.Int32(cute.arch.block_idx()[1])
    r0 = row_tile * cutlass.Int32(SF_TILE_ROWS)
    last = n_rows - cutlass.Int32(1)  # tail rows clamp their LOADS here (never past the tensor)
    src_row_stride = cutlass.Int64(h * d * 2)
    dst_row_stride = cutlass.Int64(h * d // CODES_PER_BYTE)
    src_base = mSrc.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(d * 2)
    dst_base = mDst.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(d // CODES_PER_BYTE)

    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    owns_sf = (lane % cutlass.Int32(lanes_per_block)) == cutlass.Int32(0)  # nvfp4: every lane; mxfp4: the even lane of the pair
    src_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(SRC_BYTES_PER_LANE)
    dst_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(DST_BYTES_PER_LANE)
    for p in cutlass.range_constexpr(passes):
        r = cutlass.Int32(p * rows_per_pass) + grp  # row within the 128-row tile
        row = r0 + r
        valid = row < n_rows
        row_ld = row if valid else last
        src_row = src_base + row_ld.to(cutlass.Int64) * src_row_stride
        dst_row = dst_base + row.to(cutlass.Int64) * dst_row_stride
        for c in cutlass.range_constexpr(chunks):
            vals = []
            base = src_row + cutlass.Int64((c * lanes) * SRC_BYTES_PER_LANE) + src_lane_off
            for j in cutlass.range_constexpr(LOADS_PER_LANE):
                for w in ld_global_v4(base + cutlass.Int64(j * 16), cutlass.Int32):
                    lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                    vals.append(lo)
                    vals.append(hi)
            amax = abs_max_tree(vals)
            if cutlass.const_expr(lanes_per_block == 2):
                # The lane pair (lane, lane^1) shares one 32-element block: bfly(1) completes the block amax.
                partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax, cutlass.Int32(1), 31, kind=nvvm.Shfl.BFLY))
                amax = fmax_f32(amax, partner)
            if cutlass.const_expr(sf_e4m3):
                sf, byte = e4m3_scale_from_amax(amax)
            else:
                rcp, byte = e8m0_from_amax(amax, inv_max=opaque_fp4_max_rcp())
            if valid:
                q = []
                for i in cutlass.range_constexpr(ELEMS_PER_LANE):
                    if cutlass.const_expr(sf_e4m3):
                        q.append(div_rn_f32(vals[i], sf))
                    else:
                        q.append(vals[i] * rcp)
                packed = fp32_to_fp4_pack(q)
                st_global_v2(dst_row + cutlass.Int64((c * lanes) * DST_BYTES_PER_LANE) + dst_lane_off, [packed[0], packed[1]], cutlass.Int32)
            # SF SMEM byte of this head's LOCAL scale column c_idx = elem / block (whole atoms, contiguous in the blob).
            c_idx = (cutlass.Int32(c * lanes) + lane) // cutlass.Int32(lanes_per_block)
            sf_off = sf_atom_offset(r, c_idx)
            sf_val = byte if valid else cutlass.Int32(0)  # tail row: 0x00, the padded-blob contract (nvfp4's floor would be 0x01)
            if owns_sf:
                sSF.store(sf_val.to(cutlass.Uint8), sf_off)

    # ---- SF burst: the CTA's whole (contiguous) atom run leaves SMEM as 16-B lanes ----------
    nvvm.barrier_cta_sync()
    if tidx < cutlass.Int32(burst_lanes):
        smem_off = tidx * cutlass.Int32(SF_BURST_BYTES)
        words4 = sSF.load(smem_off, vector_size=SF_BURST_BYTES, alignment=16).bitcast(cutlass.Int32)
        atom0 = (row_tile * sf_k4 + head * cutlass.Int32(cta_sf_atoms(d, block))).to(cutlass.Int64)
        gaddr = mSf.iterator.toint() + atom0 * cutlass.Int64(SF_ATOM_BYTES) + smem_off.to(cutlass.Int64)
        st_global_v4(gaddr, [words4[0], words4[1], words4[2], words4[3]], cutlass.Int32)


@cute.jit
def quantize_fp4_launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    sf: cute.Tensor,
    n_rows: cutlass.Int32,
    sf_k4: cutlass.Int32,
    n_row_tiles: cutlass.Int32,
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    block: cutlass.Constexpr[int],
    sf_e4m3: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_quantize_fp4(src, dst, sf, n_rows, sf_k4, h, d, block, sf_e4m3, threads_per_cta).launch(
        grid=(n_row_tiles, h, 1), block=(threads_per_cta, 1, 1), stream=stream
    )


# ---------------------------------------------------------------------------
# Host API
# ---------------------------------------------------------------------------

compiled_cache = {}


@dataclass(frozen=True)
class QuantizeFp4Recipe:
    """Build-time facts of one fp4 quantize launch; ``T`` rides in at run time."""

    dtype_in: object  # torch dtype of the source (bf16 / f16)
    h: int
    d: int
    fmt: str  # "nvfp4" | "mxfp4"
    compiled: object = None
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA

    @property
    def block(self) -> int:
        return FORMATS[self.fmt][0]

    @property
    def sf_e4m3(self) -> bool:
        return FORMATS[self.fmt][1]

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        return SF_TORCH_DTYPES[self.fmt]

    @property
    def k(self) -> int:
        return self.h * self.d


def compile_quantize_fp4(
    *, dtype_in, h: int, d: int, fmt, threads_per_cta: int = DEFAULT_THREADS_PER_CTA, compile_options: str = COMPILE_OPTIONS
) -> QuantizeFp4Recipe:
    """Build from SHAPES ALONE -- no allocation, no launch.

    ``fmt`` is ``"nvfp4"`` / ``"mxfp4"`` or an enum member named so.  ``compile_options`` is a dev
    knob: ``"--enable-tvm-ffi --gpu-arch sm_107a"`` trace-compiles for Rubin on any box (the SASS
    spill check); production leaves the default."""
    global _FAKE_STREAM
    name, block, sf_e4m3 = fp4_format(fmt)
    check_torch_fp4_dtypes(name)
    validate_shape(d, threads_per_cta, block)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"quantize_fp4 serves bf16/f16 sources only, got {dtype_in}")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (str(dtype_in), int(h), int(d), name, int(threads_per_cta), compile_options, current_device())
    if key not in compiled_cache:
        tok = cute.sym_int()
        u8 = _convert_to_cutlass_data_type(torch.uint8)
        # Compact source and destination (the block's gated O is compact under quantization); the row count symbolic.
        src = cute.runtime.make_fake_tensor(dtype=_convert_to_cutlass_data_type(dtype_in), shape=(tok, h, d), stride=(h * d, d, 1), assumed_align=16)
        dst = cute.runtime.make_fake_tensor(dtype=u8, shape=(tok, h * d // CODES_PER_BYTE), stride=(h * d // CODES_PER_BYTE, 1), assumed_align=16)
        sf = cute.runtime.make_fake_tensor(dtype=u8, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
        compiled_cache[key] = cute.compile(
            quantize_fp4_launch,
            src,
            dst,
            sf,
            cutlass.Int32(0),  # n_rows      ) runtime; the zeros pin the TYPE only
            cutlass.Int32(0),  # sf_k4       )
            cutlass.Int32(0),  # n_row_tiles )
            int(h),
            int(d),
            int(block),
            bool(sf_e4m3),
            int(threads_per_cta),
            _FAKE_STREAM,
            options=compile_options,
        )
    return QuantizeFp4Recipe(dtype_in=dtype_in, h=int(h), d=int(d), fmt=name, compiled=compiled_cache[key], threads_per_cta=int(threads_per_cta))


def run_quantize_fp4(r: QuantizeFp4Recipe, src: torch.Tensor, dst4: torch.Tensor, sf: torch.Tensor, *, stream) -> None:
    """Launch.  ``src`` COMPACT ``[T, H, D]`` bf16/f16; ``dst4`` ``uint8`` or ``float4_e2m1fn_x2`` with
    ``numel == T*H*D/2`` bytes, contiguous (``[T, H*D/2]`` -- any contiguous shape of that byte count binds);
    ``sf`` ``uint8`` or the format's scale dtype with ``numel == proj_gemm.sf_blob_bytes(T, H*D, block)``,
    contiguous.  Cheap host checks only; every one of them guards a wild write or a silent wrong answer at
    the tvm-ffi boundary, which reports neither."""
    if r.compiled is None:
        raise ValueError("recipe was not built by compile_quantize_fp4")
    check_torch_fp4_dtypes(r.fmt)
    if src.dtype != r.dtype_in:
        raise ValueError(f"src is {src.dtype} but this artifact was compiled for {r.dtype_in}")
    if src.ndim != 3 or int(src.shape[1]) != r.h or int(src.shape[2]) != r.d:
        raise ValueError(f"src must be [T, H={r.h}, D={r.d}], got {tuple(src.shape)}")
    if not src.is_contiguous():
        raise ValueError(f"src must be compact [T, H, D] (row = H*D contiguous elements, the GEMM's K), got strides {src.stride()}")
    t = int(src.shape[0])
    if t <= 0:
        raise ValueError(f"T must be positive, got {t}")
    if dst4.dtype not in (torch.uint8, FP4_STORAGE_DTYPE):
        raise ValueError(f"dst4 must be torch.uint8 or {FP4_STORAGE_DTYPE} (two e2m1 codes per byte), got {dst4.dtype}")
    n_code_bytes = t * r.k // CODES_PER_BYTE
    if dst4.numel() != n_code_bytes or not dst4.is_contiguous():
        raise ValueError(
            f"dst4 must be a contiguous T*H*D/2 = {n_code_bytes}-byte buffer ([T={t}, {r.k // CODES_PER_BYTE}] codes), got {tuple(dst4.shape)} {dst4.dtype}"
        )
    if sf.dtype not in (torch.uint8, r.sf_torch_dtype):
        raise ValueError(f"sf must be torch.uint8 or {r.sf_torch_dtype} (the {r.fmt} scale bytes in F8_128x4 order), got {sf.dtype}")
    need = sf_blob_bytes(t, r.k, r.block)
    if sf.numel() != need or not sf.is_contiguous():
        raise ValueError(f"sf must be the contiguous padded F8_128x4 blob of sf_blob_bytes(T={t}, K={r.k}, block={r.block}) = {need} bytes, got {sf.numel()}")
    for name, ten in (("src", src), ("dst4", dst4), ("sf", sf)):
        if not ten.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got device {ten.device}")
        if ten.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned (rows move as 16-byte vectors, the SF tile as 16-byte bursts)")
    if not (src.device == dst4.device == sf.device):
        raise ValueError(f"src, dst4 and sf must live on one device, got {src.device}, {dst4.device}, {sf.device}")
    r.compiled(
        src,
        dst4.view(torch.uint8).reshape(t, r.k // CODES_PER_BYTE),
        sf.view(torch.uint8).reshape(-1),
        cutlass.Int32(t),
        cutlass.Int32(sf_k4(r.h, r.d, r.block)),
        cutlass.Int32(n_row_tiles(t)),
        cuda.CUstream(int(stream)),
    )


frost_quantize_fp4.set_name_prefix("cudnn", remove_cutlass_symbol=True)
