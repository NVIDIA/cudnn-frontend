# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-tensor FP8 quantization pass over ``[T, H, D]``: ``dst = sat_fp8(src * scale)``.

The block's FP8 pipeline needs its SDPA operands (Q, K, V) and the out-projection's
A operand (``O_gated``) in E4M3 with a per-tensor scale.  This is that pass, as
a NAMED, MEASURED stage rather than a fusion (the fused-epilogue version is the
roadmap in ``api.py``): one streaming read of the bf16/f16 source, one write of
the fp8 destination.

* **Source may be a STRIDED column slice** of the fused projection slab (token
  stride ``N_qkvg``, head stride ``D``, element stride 1) — same dynamic
  token-stride fake as ``elementwise.py``, so no repack.  **Destination is
  COMPACT** ``[T, H, D]`` fp8 (the SDPA's zero-copy contract at 1 B/elem needs
  16-element-aligned seq/head strides, and compact ``H*D`` satisfies it).
* **``scale`` is a 1-element fp32 CUDA tensor read IN-KERNEL** (the
  ``make_array_view(t)[0]`` idiom of the FP8 SDPA kernels): a static per-tensor
  scale supplied by the caller, no host readback, so the launch stays
  CUDA-graph capturable (AGENTS.md Rule 3).  ``descale = 1/scale`` is what the
  consumer folds back.
* **Rounding**: ``cvt.rn.satfinite.e4m3x2.f32`` — round-to-nearest-even,
  saturating to +-448.  ``torch.Tensor.to(torch.float8_e4m3fn)`` saturates the
  same way on the torch pinned here, so the two agree BIT-FOR-BIT
  (``test_quantize.py`` asserts it; the reference clamps to +-448 first so it
  stays correct on a torch whose cast produces NaN on overflow).

Per-lane layout (the reason this is not literally ``elementwise.py``): a lane
moves **16 elements** — two ``ld.global.v4`` of bf16 (32 B, adjacent) in, one
``st.global.v4.b32`` of fp8 (16 B) out — because ``fp32_to_fp8_pack`` converts
exactly 16 fp32 into four packed words, and 16 B is the narrowest fully
vectorised fp8 store.  So ``LANES = D // 16`` lanes cover a row: at ``D = 256``
that is 16 lanes = HALF a warp per row, two rows per warp, the row's 512 B read
and 256 B write both contiguous across the half-warp.  Wider ``D`` adds chunks.

Traffic: ``2 B read + 1 B write`` per element (``moved_bytes``); bandwidth-bound.
**Needs sm_89+ for the fp8 ``cvt``** (Ada/Hopper/Blackwell/Rubin); an A100 has
no such instruction, so the tests skip there.

The quantized BACKWARD's gradient-quantization API is DECLARED below with its
host checks: the amax pass (``compile_amax`` / ``run_amax``), the
scale-from-amax / publish arm of this kernel (``compile_quantize(scale_src=,
n_alpha=, margin_log2=)`` / ``run_quantize(amax=, scale_out=, descale=,
alpha_consts=, alpha_outs=)``), the scalar-block init (``compile_init_scalars``
/ ``run_init_scalars``) and the "current" scale formula ``grad_scale_from_amax``.
Every default keeps today's artifact byte-identical; every non-default arm
raises ``NotImplementedError`` naming its function until the kernels land with
the quantized backward graph.
"""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.device import current_device
from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp8_pack
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global_v4

from .qk_norm_rope import fake_rowmajor_dynamic_token_stride

ELEMS_PER_LANE = 16  # what fp32_to_fp8_pack converts in one call: 32 B of bf16 in, 16 B of fp8 out
SRC_BYTES_PER_LANE = ELEMS_PER_LANE * 2
DST_BYTES_PER_LANE = ELEMS_PER_LANE * 1
LOADS_PER_LANE = SRC_BYTES_PER_LANE // 16  # two ld.global.v4 (16 B each)
FP8_E4M3_MAX = 448.0
SCALE_SOURCES = ("given", "amax")  # where the quantize kernel takes its per-tensor scale: the caller's tensor, or derived from an amax slot

DEFAULT_THREADS_PER_CTA = 128
DEFAULT_ROWS_PER_GROUP = 2
DEFAULT_CONST_HEAD_COUNT = True  # same knob, same reason as elementwise.py: a runtime H is a software divide per row

_FAKE_STREAM = None


def lanes_per_row(d: int) -> int:
    """Lanes cooperating on one head row, 16 elements each, capped at a warp."""
    return min(32, d // ELEMS_PER_LANE)


def chunks_per_lane(d: int) -> int:
    return d // (lanes_per_row(d) * ELEMS_PER_LANE)


def validate_shape(d: int, threads_per_cta: int) -> None:
    """Raise ``ValueError`` on any geometry this kernel cannot address (never an assert)."""
    if d % ELEMS_PER_LANE != 0:
        raise ValueError(f"d_head must be a multiple of {ELEMS_PER_LANE} (one fp32_to_fp8_pack per lane), got {d}")
    lanes = lanes_per_row(d)
    if 32 % lanes != 0:
        raise ValueError(f"d_head={d} gives {lanes} lanes/row, which must divide a warp")
    if d != lanes * ELEMS_PER_LANE * chunks_per_lane(d):
        raise ValueError(f"d_head={d} is not covered exactly by {lanes} lanes x {chunks_per_lane(d)} chunks x {ELEMS_PER_LANE} elements")
    if threads_per_cta % lanes != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of the {lanes} lanes per row")


def grad_scale_from_amax(amax: float, margin_log2: int = 0) -> float:
    """``2 ** (floor(log2(FP8_E4M3_MAX / amax)) - margin_log2)`` as the kernel's ``scale_src="amax"`` arm computes it; ``1.0``
    at ``amax == 0``.  Exact integer arithmetic on the fp32 bit pattern -- never an fp32 division whose rounding could move
    the floor at a boundary.  Declared; the body lands with the amax arm."""
    raise NotImplementedError("grad_scale_from_amax: declared; the body lands with the quantize kernel's scale_src='amax' arm")


def check_scalar_slot(name: str, ten, *, numel: int = 1) -> None:
    """The scalar-slot contract of these kernels' fp32 side operands (an amax target, a published scale / descale / alpha,
    the scalar block): a ``numel``-element fp32 CUDA tensor (contiguous when ``numel > 1``) at a 4-byte-aligned address --
    the kernels declare ``assumed_align=4`` for every slot pointer, so a 4-byte slot stride is legal.  Raises ``ValueError``."""
    if not isinstance(ten, torch.Tensor):
        raise ValueError(f"{name} must be a {numel}-element fp32 CUDA tensor, got {type(ten).__name__}")
    if ten.dtype != torch.float32 or ten.numel() != numel or not ten.is_cuda:
        raise ValueError(
            f"{name} must be a {numel}-element fp32 CUDA tensor (read / written in-kernel; no host readback), got {ten.dtype} x {ten.numel()} on {ten.device}"
        )
    if numel > 1 and not ten.is_contiguous():
        raise ValueError(f"{name} must be contiguous (one fp32 [{numel}] view of the scalar block), got strides {tuple(ten.stride())}")
    if ten.data_ptr() % 4:
        raise ValueError(f"{name} must sit at a 4-byte-aligned address (the kernels' assumed_align for a slot), got {ten.data_ptr():#x}")


@cute.kernel
def frost_quantize_fp8(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token/head strides
    mDst: cute.Tensor,  # [T, H, D] fp8 e4m3, own token/head strides (compact in the block)
    mScale: cute.Tensor,  # [1] fp32
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """``dst = sat_fp8(src * scale)``; ``src`` and ``dst`` never alias (dtypes differ)."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(chunks_per_lane(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h

    # The per-tensor scale, once per thread, from device memory (no host readback).
    scale = cutlass.Float32(cutlass.make_array_view(mScale)[0])

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    row0 = (cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)
    src_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(SRC_BYTES_PER_LANE)
    dst_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(DST_BYTES_PER_LANE)

    # PASS 1: every load in flight first (pure memory-level parallelism -- no
    # reduction sits between a load and the next issue).  Tail rows clamp their
    # LOADS to the last valid row and skip the store.
    rows = []
    dsts = []
    srcs = []
    for r in cutlass.range_constexpr(rows_per_group):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        token = row_r // _h
        head = row_r % _h
        src_addr = mSrc.iterator.toint() + (
            token.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[1])
        ) * cutlass.Int64(2)
        dst_addr = mDst.iterator.toint() + (token.to(cutlass.Int64) * cutlass.Int64(mDst.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mDst.stride[1]))
        row_src = []
        for c in cutlass.range_constexpr(chunks):
            base = src_addr + cutlass.Int64((c * lanes) * SRC_BYTES_PER_LANE) + src_lane_off
            vals = []
            for j in cutlass.range_constexpr(LOADS_PER_LANE):
                for w in ld_global_v4(base + cutlass.Int64(j * 16), cutlass.Int32):
                    lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                    vals.append(lo)
                    vals.append(hi)
            row_src.append(vals)
        rows.append(row)
        dsts.append(dst_addr)
        srcs.append(row_src)

    # PASS 2: scale in fp32, convert (RNE, saturating) and pack 16 -> 4 words, one v4 store.
    for r in cutlass.range_constexpr(rows_per_group):
        if rows[r] < n_rows:
            for c in cutlass.range_constexpr(chunks):
                scaled = []
                for i in cutlass.range_constexpr(ELEMS_PER_LANE):
                    scaled.append(srcs[r][c][i] * scale)
                packed = fp32_to_fp8_pack(scaled, dtype=cutlass.Float8E4M3FN)
                st_global_v4(
                    dsts[r] + cutlass.Int64((c * lanes) * DST_BYTES_PER_LANE) + dst_lane_off, [packed[0], packed[1], packed[2], packed[3]], cutlass.Int32
                )

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def quantize_fp8_launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    scale: cute.Tensor,
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    n_blocks: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_quantize_fp8(src, dst, scale, n_rows, h, h_ct, const_head_count, d, threads_per_cta, rows_per_group, use_pdl).launch(
        grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl
    )


compiled_cache = {}


class QuantizeRecipe(NamedTuple):
    """Build-time facts of one quantize launch; the token count rides in as a runtime ``Int32``."""

    compiled: object
    h: int
    d: int
    rows_per_cta: int
    dtype_in: object
    # Appended (defaults = today's artifact): where the scale comes from, how many alpha products the launch publishes,
    # the "current" recipe's power-of-two headroom.  All three are part of the artifact's ABI, so ``run_quantize`` checks
    # its operands against them BOTH ways (Rule 1).
    scale_src: str = "given"
    n_alpha: int = 0
    margin_log2: int = 0


def compile_quantize(
    *,
    dtype_in,
    h: int,
    d: int,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
    use_pdl: bool = False,
    scale_src: str = "given",
    n_alpha: int = 0,
    margin_log2: int = 0,
) -> QuantizeRecipe:
    """Build from SHAPES ALONE -- no allocation, no launch.  E4M3 output only for now.

    Appended: ``scale_src="given"`` reads the caller's 1-element scale (today's artifact, byte-identical); ``"amax"``
    derives ``scale = grad_scale_from_amax(amax, margin_log2)`` in EVERY CTA from a pre-filled amax slot and publishes it.
    ``n_alpha`` is how many ``descale * const`` products lane 0 of CTA 0 publishes (the GEMM epilogue scales).  The amax
    arm and the publish are declared here and raise ``NotImplementedError`` until they land."""
    global _FAKE_STREAM
    validate_shape(d, threads_per_cta)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"quantize serves bf16/f16 sources only, got {dtype_in}")
    if scale_src not in SCALE_SOURCES:
        raise ValueError(f"scale_src must be one of {SCALE_SOURCES}, got {scale_src!r}")
    if isinstance(n_alpha, bool) or not isinstance(n_alpha, int) or n_alpha < 0:
        raise ValueError(f"n_alpha must be a non-negative int (the alpha products the launch publishes), got {n_alpha!r}")
    if isinstance(margin_log2, bool) or not isinstance(margin_log2, int):
        raise ValueError(f"margin_log2 must be an int (the 'current' recipe's power-of-two headroom), got {margin_log2!r}")
    if (scale_src, n_alpha, margin_log2) != ("given", 0, 0):
        raise NotImplementedError(
            f"compile_quantize: the scale_src='amax' / publish arm (scale_src={scale_src!r}, n_alpha={n_alpha}, margin_log2={margin_log2}) is "
            "declared but not implemented yet; it lands with the quantized backward graph"
        )
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (str(dtype_in), h, d, int(threads_per_cta), int(rows_per_group), bool(const_head_count), bool(use_pdl), current_device())
    if key not in compiled_cache:
        tok = cute.sym_int()
        # Both operands carry a symbolic token stride: the source is a column
        # slice of the projection slab, the destination is compact -- one
        # artifact serves both (and a compact source, too).
        src = fake_rowmajor_dynamic_token_stride(dtype_in, tok, h, d)
        dst = cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(tok, h, d), stride=(cute.sym_int(), d, 1), assumed_align=16)
        scale = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4)
        compiled_cache[key] = cute.compile(
            quantize_fp8_launch,
            src,
            dst,
            scale,
            cutlass.Int32(0),  # n_rows   ) runtime; the zeros pin the TYPE only
            cutlass.Int32(h),  # h        )
            cutlass.Int32(0),  # n_blocks )
            int(h),
            bool(const_head_count),
            d,
            int(threads_per_cta),
            int(rows_per_group),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return QuantizeRecipe(
        compiled=compiled_cache[key],
        h=h,
        d=d,
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        dtype_in=dtype_in,
        scale_src=scale_src,
        n_alpha=int(n_alpha),
        margin_log2=int(margin_log2),
    )


def run_quantize(
    r: QuantizeRecipe,
    src: torch.Tensor,
    dst: torch.Tensor,
    scale: Optional[torch.Tensor],
    *,
    stream,
    amax: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    descale: Optional[torch.Tensor] = None,
    alpha_consts: tuple = (),
    alpha_outs: tuple = (),
) -> None:
    """Launch.  ``src`` ``[T, H, D]`` bf16/f16 (strided ok), ``dst`` ``[T, H, D]`` ``float8_e4m3fn``, ``scale`` 1-element fp32 CUDA.

    Cheap host checks only; every one of them guards a wild write or a silent
    wrong answer at the tvm-ffi boundary, which reports neither.

    Appended (the quantized backward's arm; every slot operand a 1-element fp32 CUDA view at a 4-byte-aligned address,
    checked BOTH ways against the recipe -- Rule 1): under ``scale_src="given"`` ``scale`` is READ and ``amax`` must be
    ``None``; under ``"amax"`` ``scale`` must be ``None`` while ``amax`` (the slot the amax pass filled) and ``scale_out``
    are REQUIRED -- every CTA derives the scale itself and never reads ``scale_out``, which CTA 0 writes concurrently.
    In both arms lane 0 of CTA 0 publishes ``scale_out[0] = scale`` and ``descale[0] = 1 / scale`` when given, and
    ``alpha_outs[i][0] = descale * alpha_consts[i][0]`` for ``i < n_alpha`` (a length other than ``n_alpha`` is a typed
    error: the ABI is fixed per artifact).  The arm itself raises ``NotImplementedError`` until the kernel lands.
    """
    if src.dtype != r.dtype_in:
        raise ValueError(f"src is {src.dtype} but this artifact was compiled for {r.dtype_in}")
    if dst.dtype != torch.float8_e4m3fn:
        raise ValueError(f"dst must be torch.float8_e4m3fn, got {dst.dtype}")
    for name, ten in (("src", src), ("dst", dst)):
        if ten.ndim != 3 or int(ten.shape[1]) != r.h or int(ten.shape[2]) != r.d:
            raise ValueError(f"{name} must be [T, H={r.h}, D={r.d}], got {tuple(ten.shape)}")
        if ten.stride(2) != 1 or ten.stride(1) != r.d:
            raise ValueError(f"{name} must have head stride D={r.d} and element stride 1 (a column slice of the slab or compact), got strides {ten.stride()}")
    if int(src.shape[0]) != int(dst.shape[0]):
        raise ValueError(f"src has T={int(src.shape[0])} rows, dst T={int(dst.shape[0])}")
    if (src.stride(0) * 2) % 16 or (dst.stride(0)) % 16:
        raise ValueError(f"token strides must keep every row 16-byte aligned: src {src.stride(0)} elems (bf16), dst {dst.stride(0)} elems (fp8)")
    if r.scale_src == "given":
        if scale is None:
            raise ValueError("this artifact reads the caller's scale (scale_src='given'); scale must be bound at execute (Rule 1: no silent fallback)")
        if scale.dtype != torch.float32 or scale.numel() != 1 or not scale.is_cuda:
            raise ValueError("scale must be a 1-element fp32 CUDA tensor (read in-kernel; no host readback)")
        if amax is not None:
            raise ValueError("this artifact reads the caller's scale (scale_src='given'); passing amax would silently ignore it (Rule 1)")
    else:
        if scale is not None:
            raise ValueError("this artifact derives its scale from amax (scale_src='amax'); passing a caller scale would silently ignore it (Rule 1)")
        if amax is None:
            raise ValueError(
                "this artifact derives its scale from amax (scale_src='amax'); amax (the slot the amax pass filled) must be bound at execute (Rule 1)"
            )
        if scale_out is None:
            raise ValueError(
                "this artifact derives its scale from amax (scale_src='amax'); scale_out (where the derived scale is published) must be bound at execute (Rule 1)"
            )
    if len(alpha_consts) != r.n_alpha or len(alpha_outs) != r.n_alpha:
        raise ValueError(
            f"this artifact publishes n_alpha={r.n_alpha} alpha products; got {len(alpha_consts)} alpha_consts and {len(alpha_outs)} alpha_outs "
            "(the ABI is fixed per artifact)"
        )
    slots = [("amax", amax), ("scale_out", scale_out), ("descale", descale)]
    slots += [(f"alpha_consts[{i}]", c) for i, c in enumerate(alpha_consts)] + [(f"alpha_outs[{i}]", a) for i, a in enumerate(alpha_outs)]
    for name, ten in slots:
        if ten is not None:
            check_scalar_slot(name, ten)
    if r.scale_src != "given" or r.n_alpha or amax is not None or scale_out is not None or descale is not None:
        raise NotImplementedError(
            "run_quantize: the scale_src='amax' / publish arm is declared but not implemented yet; it lands with the quantized backward graph"
        )
    t = int(src.shape[0])
    n_rows = t * r.h
    n_blocks = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
    r.compiled(src, dst, scale.reshape(1), cutlass.Int32(n_rows), cutlass.Int32(r.h), cutlass.Int32(n_blocks), cuda.CUstream(int(stream)))


def moved_bytes(t: int, h: int, d: int, *, src_elem_bytes: int = 2) -> int:
    """HBM traffic of one launch: the bf16/f16 read plus the 1-byte fp8 write.  The 4-byte scale is noise."""
    return t * h * d * (src_elem_bytes + 1)


# ---------------------------------------------------------------------------
# The quantized backward's gradient-quantization API -- declared with its host checks; the kernels land with the
# quantized backward graph (every entry raises NotImplementedError naming itself until then)
# ---------------------------------------------------------------------------


class AmaxRecipe(NamedTuple):
    """Build-time facts of one amax launch (``run_amax``): the quantize kernel's own lane layout over ``[T, H, D]``."""

    compiled: object
    h: int
    d: int
    rows_per_cta: int
    dtype_in: object


def compile_amax(
    *,
    dtype_in,
    h: int,
    d: int,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
) -> AmaxRecipe:
    """The amax pass ``amax = max |fp32(src)|`` over a ``[T, H, D]`` bf16 / f16 source (compact or a slab slice): one int32
    ``atomicMax`` per warp of non-negative fp32 bit patterns, which order as int32 -- order-free, bitwise the fp32 max.
    Build from SHAPES ALONE.  Declared; raises ``NotImplementedError`` until the kernel lands."""
    validate_shape(d, threads_per_cta)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"amax serves bf16/f16 sources only, got {dtype_in}")
    raise NotImplementedError("compile_amax: the amax pass is declared but not implemented yet; it lands with the quantized backward graph")


def run_amax(r: AmaxRecipe, src: torch.Tensor, amax: torch.Tensor, *, stream) -> None:
    """Launch.  ``src`` ``[T, H, D]`` bf16 / f16 under the quantize kernel's layout rules (head stride ``D``, element stride 1,
    a 16-byte-aligned token stride); ``amax`` a 1-element fp32 CUDA view PRE-ZEROED by the caller on the same stream (an
    ``atomicMax`` target: a poisoned slot is the caller's bug).  Host checks only; declared -- raises ``NotImplementedError``
    until the kernel lands."""
    if src.dtype != r.dtype_in:
        raise ValueError(f"src is {src.dtype} but this artifact was compiled for {r.dtype_in}")
    if src.ndim != 3 or int(src.shape[1]) != r.h or int(src.shape[2]) != r.d:
        raise ValueError(f"src must be [T, H={r.h}, D={r.d}], got {tuple(src.shape)}")
    if src.stride(2) != 1 or src.stride(1) != r.d:
        raise ValueError(f"src must have head stride D={r.d} and element stride 1 (a column slice of the slab or compact), got strides {src.stride()}")
    if (src.stride(0) * 2) % 16:
        raise ValueError(f"the token stride must keep every row 16-byte aligned: src {src.stride(0)} elems (bf16/f16)")
    check_scalar_slot("amax", amax)
    raise NotImplementedError("run_amax: the amax pass is declared but not implemented yet; it lands with the quantized backward graph")


class InitScalarsRecipe(NamedTuple):
    """Build-time facts of the scalar-block init launch: the number of fp32 slots it zeroes."""

    compiled: object
    n_slots: int


def compile_init_scalars(n_slots: int) -> InitScalarsRecipe:
    """ONE thread: ``slots[0:n_slots] = 0.0``, then ``descale_dp_out[0] = 1.0 / scale_dp[0]`` -- every amax slot must be zero
    before the first ``atomicMax`` of the first pass, and the reciprocal is derived on device (one fp32 division, exact for
    a power of two) so no second caller input can disagree with ``scale_dp``.  Declared; raises ``NotImplementedError``
    until the kernel lands."""
    if isinstance(n_slots, bool) or not isinstance(n_slots, int) or n_slots < 1:
        raise ValueError(f"n_slots must be a positive int (the fp32 slots of the scalar block), got {n_slots!r}")
    raise NotImplementedError("compile_init_scalars: the scalar-block init is declared but not implemented yet; it lands with the quantized backward graph")


def run_init_scalars(r: InitScalarsRecipe, slots: torch.Tensor, scale_dp: torch.Tensor, descale_dp_out: torch.Tensor, *, stream) -> None:
    """Launch.  ``slots`` the contiguous fp32 ``[n_slots]`` view of the scalar block; ``scale_dp`` the caller's 1-element fp32
    CUDA scalar; ``descale_dp_out`` a 1-element fp32 view INTO the same block (written after the zeroing, sequentially in
    one thread, so no race).  Host checks only; declared -- raises ``NotImplementedError`` until the kernel lands."""
    check_scalar_slot("slots", slots, numel=int(r.n_slots))
    check_scalar_slot("scale_dp", scale_dp)
    check_scalar_slot("descale_dp_out", descale_dp_out)
    raise NotImplementedError("run_init_scalars: the scalar-block init is declared but not implemented yet; it lands with the quantized backward graph")


frost_quantize_fp8.set_name_prefix("cudnn", remove_cutlass_symbol=True)
