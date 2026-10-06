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

**The quantized BACKWARD's gradient quantization** rides on the same kernel plus
two small ones, all over fp32 SCALAR SLOTS (1-element fp32 CUDA views, 4-byte
aligned -- the kernels declare ``assumed_align=4`` for every slot pointer, so a
packed 4-byte slot stride is legal; ``check_scalar_slot`` is the one host spelling,
and ``_check_one_cuda_device`` pins every operand of a launch to ``src``'s device):

* **the amax pass** (``compile_amax`` / ``run_amax``): ``amax = max |fp32(src)|``
  over ``[T, H, D]`` in THIS kernel's lane layout (16 elements per lane, two
  ``ld.global.v4``), a ternary abs-max tree per lane, a warp butterfly, and ONE
  int32 ``atomicMax`` per warp of the fp32 bit pattern (``tile_dsl.pointwise.
  atomic_max_f32_bits``): non-negative fp32 patterns order as int32, so the slot
  ends at the fp32 max in any arrival order -- order-free, BITWISE
  ``x.float().abs().amax()``.  The slot is PRE-ZEROED by the caller on the same
  stream (0 is the identity of the max; a poisoned slot is the caller's bug).
* **the scale-from-amax arm** (``compile_quantize(scale_src="amax", margin_log2=)``):
  EVERY CTA derives ``scale = grad_scale_from_amax(amax, margin_log2)`` from the
  slot the amax pass filled -- ``2 ** (floor(log2(448 / amax)) - margin)``, ``1.0``
  at ``amax == 0``, as exact integer arithmetic on the fp32 bit pattern (never a
  division whose rounding could move the floor) -- and never reads the published
  copy (CTA 0 writes it concurrently).  So ``amax * scale <= 448`` for the amax the
  kernel read: no element saturates.
* **the publish** (``publish=True``, implied by ``scale_src="amax"`` or
  ``n_alpha > 0``): lane 0 of CTA 0 writes ``scale_out[0] = scale``, ``descale[0] =
  1 / scale`` (one ``div.rn.f32``: exact for a power of two) and ``alpha_outs[i][0] =
  descale * alpha_consts[i][0]`` for ``i < n_alpha`` (ONE fp32 RN multiply each:
  ``np.float32(descale) * np.float32(c)`` on the host gives the same bits) -- the
  GEMM epilogue scales of the backward.  BOTH arms publish through the same code,
  so a "delayed" recipe that feeds the caller's ``scale`` replays the "current"
  run bitwise, slots included.
* **the scalar-block init** (``compile_init_scalars`` / ``run_init_scalars``): ONE
  thread zeroes the ``n_slots`` fp32 slots (every amax slot must be zero before
  the first ``atomicMax`` of the first pass) and then writes ``descale_dp_out[0] =
  1 / scale_dp[0]`` (``div.rn.f32``) -- so no second caller input can disagree
  with ``scale_dp``.  Every store is an ordered inline-PTX store: the reciprocal
  may legally target one of the slots just zeroed.

The defaults (``scale_src="given"``, ``n_alpha=0``, ``margin_log2=0``,
``publish=False``) trace today's artifact: every new operand is ``None`` and every
new arm a ``const_expr`` folded out.  The amax pass and the init launch need no
fp8 instruction and run on every CUDA device; the fp8 ``cvt`` arms decline a
pre-sm_89 device BY NAME at ``compile_*`` (AGENTS.md Rule 7).

SMEM buffer table: NONE (no SMEM).  Barrier table: NONE (no mbarrier, no named
barrier, no TMA; the amax fold is one ``atomicMax`` per warp).  Nothing here can hang.
"""

import math
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.device import compute_capability, current_device
from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import (
    abs_max_tree,
    atomic_max_f32_bits,
    div_rn_f32,
    f16x2_to_f32,
    fmax_f32,
    fp32_to_fp8_pack,
    opaque_f32_zero,
    warp_abs_max_f32_shfl,
)
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global, st_global_v4

from .qk_norm_rope import fake_rowmajor_dynamic_token_stride

ELEMS_PER_LANE = 16  # what fp32_to_fp8_pack converts in one call: 32 B of bf16 in, 16 B of fp8 out
SRC_BYTES_PER_LANE = ELEMS_PER_LANE * 2
DST_BYTES_PER_LANE = ELEMS_PER_LANE * 1
LOADS_PER_LANE = SRC_BYTES_PER_LANE // 16  # two ld.global.v4 (16 B each)
FP8_E4M3_MAX = 448.0
SCALE_SOURCES = ("given", "amax")  # where the quantize kernel takes its per-tensor scale: the caller's tensor, or derived from an amax slot
# The "current" gradient-scale formula is spelled from frexp(448) = (0.875, 9) -- 448 = 0.875 * 2**9 -- on the host
# (grad_scale_from_amax) and in-kernel (_grad_scale_bits): never a literal 9 / 0.875 / 0x600000.
_E4M3_MAX_MANT, _E4M3_MAX_EXP = math.frexp(FP8_E4M3_MAX)
# amax = m * 2**e with m in [0.5, 1); m > 0.875  <=>  the fp32 mantissa field M > (2 * 0.875 - 1) * 2**23 (= 0x600000)
_E4M3_MAX_MANT_BITS = int(round((2.0 * _E4M3_MAX_MANT - 1.0) * (1 << 23)))
_FP32_FREXP_BIAS = 126  # the biased exponent field E of a NORMAL fp32 is frexp's e + 126
_SCALE_LOG2_MIN, _SCALE_LOG2_MAX = -126, 127  # the scale is clamped to a finite NORMAL power of two
_FP8_CVT_MIN_CC = (8, 9)  # cvt.rn.satfinite.e4m3x2.f32: Ada / Hopper / Blackwell / Rubin
MAX_ALPHA = 4  # alpha products one quantize launch can publish (the ABI reserves this many slot pairs; the block needs 2)

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
    at ``amax == 0``.  Exact integer arithmetic on the exponent (``math.frexp``), never an fp32 division whose rounding could
    move the floor at a boundary: with ``amax = m * 2**e`` (``m`` in ``[0.5, 1)``) and ``448 = 0.875 * 2**9``,
    ``floor(log2(448 / amax)) = (9 - e) - [m > 0.875]``.  The exponent is clamped to ``[-126, 127]`` -- a finite NORMAL
    power of two for ANY finite amax (a subnormal amax gives ``2**127``, a huge one ``2**-126``).  Properties: ``amax * scale
    <= 448`` for finite ``amax > 0`` (so nothing saturates), ``1 / scale`` is exact.  The amax is the kernel's own
    (``>= 0`` and finite): a negative or non-finite value is out of contract and raises."""
    if isinstance(amax, bool) or not isinstance(amax, (int, float)):
        raise ValueError(f"grad_scale_from_amax: amax must be a real number (the amax slot's value), got {type(amax).__name__}")
    amax = float(amax)
    if not math.isfinite(amax) or amax < 0.0:
        raise ValueError(f"grad_scale_from_amax: amax must be finite and >= 0 (an amax pass's max |x|), got {amax!r}")
    if isinstance(margin_log2, bool) or not isinstance(margin_log2, int):
        raise ValueError(f"grad_scale_from_amax: margin_log2 must be an int, got {margin_log2!r}")
    if amax == 0.0:
        return 1.0
    m, e = math.frexp(amax)
    k = (_E4M3_MAX_EXP - e) - (1 if m > _E4M3_MAX_MANT else 0) - margin_log2
    k = max(min(k, _SCALE_LOG2_MAX), _SCALE_LOG2_MIN)
    return math.ldexp(1.0, k)


@cute.jit
def _grad_scale_bits(amax: cutlass.Float32, margin_log2: cutlass.Constexpr[int]) -> cutlass.Float32:
    """:func:`grad_scale_from_amax` on the device, bit for bit, from the fp32 pattern of ``amax``: ``E = (bits >> 23) & 0xFF``
    is frexp's ``e + 126`` for a normal value and ``M = bits & 0x7FFFFF`` its mantissa field (``m > 0.875 <=> M > 0x600000``),
    so ``k = (9 + 126 - margin) - E - [M > 0x600000]``, clamped to ``[-126, 127]``, ``0`` for a zero amax; the scale is the
    pattern ``(k + 127) << 23``.  A subnormal amax (``E == 0``) lands on the ``2**127`` clamp like the host's.  Every
    operation is integer: no rounding anywhere."""
    bits = amax.bitcast(cutlass.Int32) & cutlass.Int32(0x7FFFFFFF)
    exp_field = (bits >> 23) & cutlass.Int32(0xFF)
    mant_field = bits & cutlass.Int32(0x7FFFFF)
    k = cutlass.Int32(_E4M3_MAX_EXP + _FP32_FREXP_BIAS - margin_log2) - exp_field
    k = k - (cutlass.Int32(1) if mant_field > cutlass.Int32(_E4M3_MAX_MANT_BITS) else cutlass.Int32(0))
    k = cutlass.Int32(_SCALE_LOG2_MAX) if k > cutlass.Int32(_SCALE_LOG2_MAX) else k
    k = cutlass.Int32(_SCALE_LOG2_MIN) if k < cutlass.Int32(_SCALE_LOG2_MIN) else k
    k = cutlass.Int32(0) if bits == cutlass.Int32(0) else k
    return ((k + cutlass.Int32(127)) << 23).bitcast(cutlass.Float32)


def require_fp8_cvt(who: str) -> None:
    """The fp8 ``cvt.rn.satfinite`` needs sm_89+: decline a pre-Ada device BY NAME at compile time (AGENTS.md Rule 7), never
    through the ptxas error the trace would otherwise die with."""
    cc = compute_capability(current_device())
    if tuple(cc) < _FP8_CVT_MIN_CC:
        raise NotImplementedError(
            f"{who}: the fp8 cvt.rn.satfinite instruction needs sm_{_FP8_CVT_MIN_CC[0]}{_FP8_CVT_MIN_CC[1]}+ (Ada / Hopper / Blackwell / Rubin); "
            f"the current device is sm_{cc[0]}{cc[1]}"
        )


def _slot_value(m: cute.Tensor) -> cutlass.Float32:
    """The fp32 a 1-element slot holds, read in-kernel (no host readback)."""
    return cutlass.Float32(cutlass.make_array_view(m)[0])


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


def _check_one_cuda_device(anchor_name: str, anchor, operands) -> None:
    """Every bound operand on ONE CUDA device, the anchor's -- the device half of every block launcher's host contract
    (``run_quantize`` here; ``sigmoid_gate_bwd`` and the qk-norm / RoPE backward import it).

    The kernel reads each operand through a device pointer.  A CPU operand (a ``seq_lens`` built as ``torch.tensor(lens,
    dtype=torch.int32)`` with no ``device=``, a scalar slot) passes the dtype / rank / length checks and would be
    dereferenced as a HOST address -- an illegal-address fault at the next synchronize, sticky for the process -- and an
    operand on ANOTHER GPU, which an ``is_cuda`` check cannot tell from a local one, is the same wild access on a foreign
    device (or silent peer traffic).  Named here -- the operand, the launch device and where it actually is -- before the
    launch.  ``operands`` is ``(name, tensor)`` pairs; a ``None`` (an unbound optional) is skipped."""
    if not anchor.is_cuda:
        raise ValueError(f"{anchor_name} must be a CUDA tensor (the kernel reads every operand through a device pointer), got device {anchor.device}")
    for name, ten in operands:
        if ten is not None and ten.device != anchor.device:
            raise ValueError(f"{name} must be on {anchor.device} with {anchor_name}, got {ten.device}")


@cute.kernel
def frost_quantize_fp8(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token/head strides
    mDst: cute.Tensor,  # [T, H, D] fp8 e4m3, own token/head strides (compact in the block)
    mScale: Optional[cute.Tensor],  # [1] fp32: the caller's scale (scale_src="given"); None under "amax"
    mAmax: Optional[cute.Tensor],  # [1] fp32: the slot the amax pass filled (scale_src="amax"); None under "given"
    mScaleOut: Optional[cute.Tensor],  # [1] fp32 OUT (publish): the scale this launch applied
    mDescale: Optional[cute.Tensor],  # [1] fp32 OUT (publish): 1 / scale
    mAlphaC0: Optional[cute.Tensor],  # [1] fp32 IN: alpha constant i (bound for i < n_alpha)
    mAlphaC1: Optional[cute.Tensor],
    mAlphaC2: Optional[cute.Tensor],
    mAlphaC3: Optional[cute.Tensor],
    mAlphaO0: Optional[cute.Tensor],  # [1] fp32 OUT: alpha_i = descale * alpha constant i
    mAlphaO1: Optional[cute.Tensor],
    mAlphaO2: Optional[cute.Tensor],
    mAlphaO3: Optional[cute.Tensor],
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    n_alpha: cutlass.Constexpr[int],
    margin_log2: cutlass.Constexpr[int],
) -> None:
    """``dst = sat_fp8(src * scale)``; ``src`` and ``dst`` never alias (dtypes differ).

    ``scale`` is the caller's (``mScale``) or derived in EVERY CTA from the amax slot (``mAmax``, module docstring); with a
    publish (``mScaleOut`` bound) lane 0 of CTA 0 also writes the scale, its reciprocal and the ``n_alpha`` alpha products.
    """
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(chunks_per_lane(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h
    from_amax = cutlass.const_expr(mAmax is not None)
    publish = cutlass.const_expr(mScaleOut is not None)

    # The per-tensor scale, once per thread, from device memory (no host readback): the caller's, or the "current" recipe's
    # power of two from the amax slot -- derived HERE, never read back from the published copy CTA 0 writes concurrently.
    scale = _grad_scale_bits(_slot_value(mAmax), margin_log2) if cutlass.const_expr(from_amax) else _slot_value(mScale)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    if cutlass.const_expr(publish):
        # Lane 0 of CTA 0 publishes: the scale, descale = 1 / scale (div.rn.f32: exact for a power of two, correctly rounded
        # otherwise -- never a reciprocal-multiply) and alpha_i = descale * c_i (one RN multiply each).  The one BLOCK the
        # scalar reads below depends on nothing this launch writes (the host refuses an aliasing slot).  The alpha stores
        # are spelled HERE, in the kernel body, never in a Python helper: the DSL transforms only this function's own source,
        # so a helper's ops have no claim to the branch they are called from (python/cudnn/AGENTS.md, "CuTeDSL kernel
        # bodies"); range_constexpr's trace-time i indexes the ABI's slot tuples, so exactly the n_alpha bound pairs are touched.
        alpha_consts = (mAlphaC0, mAlphaC1, mAlphaC2, mAlphaC3)
        alpha_outs = (mAlphaO0, mAlphaO1, mAlphaO2, mAlphaO3)
        if (cutlass.Int32(cute.arch.block_idx()[0]) == cutlass.Int32(0)) & (tidx == cutlass.Int32(0)):
            descale = div_rn_f32(opaque_f32_zero() + cutlass.Float32(1.0), scale)
            mScaleOut.iterator[0] = scale
            mDescale.iterator[0] = descale
            for i in cutlass.range_constexpr(n_alpha):
                alpha_outs[i].iterator[0] = descale * _slot_value(alpha_consts[i])
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
    scale: Optional[cute.Tensor],
    amax: Optional[cute.Tensor],
    scale_out: Optional[cute.Tensor],
    descale: Optional[cute.Tensor],
    alpha_c0: Optional[cute.Tensor],
    alpha_c1: Optional[cute.Tensor],
    alpha_c2: Optional[cute.Tensor],
    alpha_c3: Optional[cute.Tensor],
    alpha_o0: Optional[cute.Tensor],
    alpha_o1: Optional[cute.Tensor],
    alpha_o2: Optional[cute.Tensor],
    alpha_o3: Optional[cute.Tensor],
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    n_blocks: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    n_alpha: cutlass.Constexpr[int],
    margin_log2: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_quantize_fp8(
        src,
        dst,
        scale,
        amax,
        scale_out,
        descale,
        alpha_c0,
        alpha_c1,
        alpha_c2,
        alpha_c3,
        alpha_o0,
        alpha_o1,
        alpha_o2,
        alpha_o3,
        n_rows,
        h,
        h_ct,
        const_head_count,
        d,
        threads_per_cta,
        rows_per_group,
        use_pdl,
        n_alpha,
        margin_log2,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


compiled_cache = {}


def _fake_slot():
    """The trace-time shape of every fp32 scalar slot: ONE element, 4-byte aligned (a packed 4-byte slot stride is legal)."""
    return cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4)


def _slot_view(ten: Optional[torch.Tensor]):
    """The ``[1]`` view a slot is bound as (a 1-element reshape never copies -- Rule 1), or ``None``."""
    return None if ten is None else ten.reshape(1)


class QuantizeRecipe(NamedTuple):
    """Build-time facts of one quantize launch; the token count rides in as a runtime ``Int32``."""

    compiled: object
    h: int
    d: int
    rows_per_cta: int
    dtype_in: object
    # Appended (defaults = today's artifact): where the scale comes from, how many alpha products the launch publishes,
    # the "current" recipe's power-of-two headroom, and whether the publish arm (scale_out / descale) is traced.  All four
    # are part of the artifact's ABI, so ``run_quantize`` checks its operands against them BOTH ways (Rule 1).
    scale_src: str = "given"
    n_alpha: int = 0
    margin_log2: int = 0
    publish: bool = False


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
    publish: bool = False,
) -> QuantizeRecipe:
    """Build from SHAPES ALONE -- no allocation, no launch.  E4M3 output only for now.

    Appended: ``scale_src="given"`` reads the caller's 1-element scale (today's artifact); ``"amax"`` derives ``scale =
    grad_scale_from_amax(amax, margin_log2)`` in EVERY CTA from a pre-filled amax slot.  ``n_alpha`` (``<= MAX_ALPHA``) is
    how many ``descale * const`` products lane 0 of CTA 0 publishes (the GEMM epilogue scales).  ``publish`` traces the
    publish arm -- ``scale_out`` and ``descale`` become REQUIRED at execute -- and is IMPLIED by ``scale_src="amax"`` or
    ``n_alpha > 0``; pass it explicitly for a "given" launch that must publish its scale and reciprocal without alpha
    products (the backward's dO quantize under the delayed recipe).  Every knob is in the cache key; the defaults share
    the forward's artifact.  The fp8 ``cvt`` needs sm_89+: a pre-Ada device is declined by name (Rule 7)."""
    global _FAKE_STREAM
    validate_shape(d, threads_per_cta)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"quantize serves bf16/f16 sources only, got {dtype_in}")
    if scale_src not in SCALE_SOURCES:
        raise ValueError(f"scale_src must be one of {SCALE_SOURCES}, got {scale_src!r}")
    if isinstance(n_alpha, bool) or not isinstance(n_alpha, int) or n_alpha < 0:
        raise ValueError(f"n_alpha must be a non-negative int (the alpha products the launch publishes), got {n_alpha!r}")
    if n_alpha > MAX_ALPHA:
        raise ValueError(f"n_alpha={n_alpha} exceeds the {MAX_ALPHA} alpha slot pairs the quantize launch's ABI reserves")
    if isinstance(margin_log2, bool) or not isinstance(margin_log2, int):
        raise ValueError(f"margin_log2 must be an int (the 'current' recipe's power-of-two headroom), got {margin_log2!r}")
    if not isinstance(publish, bool):
        raise ValueError(f"publish must be a bool (whether the launch publishes scale_out / descale), got {publish!r}")
    publish = publish or scale_src == "amax" or n_alpha > 0
    require_fp8_cvt("compile_quantize")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (
        str(dtype_in),
        h,
        d,
        int(threads_per_cta),
        int(rows_per_group),
        bool(const_head_count),
        bool(use_pdl),
        current_device(),
        scale_src,
        int(n_alpha),
        int(margin_log2),
        publish,
    )
    if key not in compiled_cache:
        tok = cute.sym_int()
        # Both operands carry a symbolic token stride: the source is a column
        # slice of the projection slab, the destination is compact -- one
        # artifact serves both (and a compact source, too).
        src = fake_rowmajor_dynamic_token_stride(dtype_in, tok, h, d)
        dst = cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(tok, h, d), stride=(cute.sym_int(), d, 1), assumed_align=16)
        # The scalar slots: each traced ONLY when its arm is on (None otherwise -- the ABI keeps the parameter, the kernel
        # folds the arm out), every one a 1-element fp32 at 4-byte alignment.
        scale = _fake_slot() if scale_src == "given" else None
        amax = _fake_slot() if scale_src == "amax" else None
        scale_out = _fake_slot() if publish else None
        descale = _fake_slot() if publish else None
        alpha_c = [_fake_slot() if i < n_alpha else None for i in range(MAX_ALPHA)]
        alpha_o = [_fake_slot() if i < n_alpha else None for i in range(MAX_ALPHA)]
        compiled_cache[key] = cute.compile(
            quantize_fp8_launch,
            src,
            dst,
            scale,
            amax,
            scale_out,
            descale,
            *alpha_c,
            *alpha_o,
            cutlass.Int32(0),  # n_rows   ) runtime; the zeros pin the TYPE only
            cutlass.Int32(h),  # h        )
            cutlass.Int32(0),  # n_blocks )
            int(h),
            bool(const_head_count),
            d,
            int(threads_per_cta),
            int(rows_per_group),
            bool(use_pdl),
            int(n_alpha),
            int(margin_log2),
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
        publish=publish,
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
    ``None``; under ``"amax"`` ``scale`` must be ``None`` while ``amax`` (the slot the amax pass filled) is REQUIRED --
    every CTA derives the scale itself and never reads ``scale_out``, which CTA 0 writes concurrently.  With the publish
    arm (``r.publish``: implied by ``"amax"`` or ``n_alpha > 0``, explicit otherwise) ``scale_out`` AND ``descale`` are
    REQUIRED and lane 0 of CTA 0 writes ``scale_out[0] = scale``, ``descale[0] = 1 / scale`` and ``alpha_outs[i][0] =
    descale * alpha_consts[i][0]`` for ``i < n_alpha`` (a length other than ``n_alpha`` is a typed error: the ABI is fixed
    per artifact); without it every one of them must be ``None``.  No published slot may alias a slot the launch READS
    (``amax``, ``scale``, an alpha constant) -- the other CTAs read it concurrently -- nor another published slot.  Every
    operand -- ``src``, ``dst``, ``scale`` and each slot -- sits on ONE CUDA device, ``src``'s (``_check_one_cuda_device``):
    the kernel reads and writes them all through raw device pointers, and ``is_cuda`` alone cannot see a slot on another GPU.
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
    if int(src.shape[0]) < 1:
        raise ValueError("src has no rows (T == 0): nothing to quantize, and a zero-size grid is a launch error, not a no-op")
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
    if r.publish:
        if scale_out is None or descale is None:
            raise ValueError(
                "this artifact was compiled WITH the publish arm (publish=True, implied by scale_src='amax' or n_alpha > 0): scale_out AND descale "
                "(1-element fp32 CUDA views) must be bound at execute (Rule 1: no silent fallback)"
            )
    elif scale_out is not None or descale is not None:
        raise ValueError(
            "this artifact was compiled WITHOUT the publish arm (publish=False); passing scale_out / descale would silently ignore them (Rule 1) -- "
            "compile with publish=True (it is implied by scale_src='amax' or n_alpha > 0, explicit for a 'given' launch that publishes)"
        )
    slots = [("amax", amax), ("scale_out", scale_out), ("descale", descale)]
    slots += [(f"alpha_consts[{i}]", c) for i, c in enumerate(alpha_consts)] + [(f"alpha_outs[{i}]", a) for i, a in enumerate(alpha_outs)]
    for name, ten in slots:
        if ten is not None:
            check_scalar_slot(name, ten)
    # ONE CUDA device, src's, for every operand -- dst, the scale source and each slot: the kernel reads and writes them all
    # through raw device pointers, and the is_cuda half of the slot contract cannot tell a slot on ANOTHER GPU from a local one.
    _check_one_cuda_device("src", src, [("dst", dst), ("scale", scale)] + slots)
    # A published slot must not be a slot this launch READS (every CTA reads amax / scale / the alpha constants while CTA 0
    # writes), nor another published slot (two writers of one word).
    reads = [("scale", scale), ("amax", amax)] + [(f"alpha_consts[{i}]", c) for i, c in enumerate(alpha_consts)]
    writes = [("scale_out", scale_out), ("descale", descale)] + [(f"alpha_outs[{i}]", a) for i, a in enumerate(alpha_outs)]
    writes = [(n, w) for n, w in writes if w is not None]
    for wn, w in writes:
        for rn, rd in reads:
            if rd is not None and w.data_ptr() == rd.data_ptr():
                raise ValueError(f"{wn} aliases {rn}: a published slot cannot be a slot the launch reads (the other CTAs read it while CTA 0 writes)")
    for i, (wn, w) in enumerate(writes):
        for vn, v in writes[i + 1 :]:
            if w.data_ptr() == v.data_ptr():
                raise ValueError(f"{wn} and {vn} are the same slot: two published values cannot share one word")
    t = int(src.shape[0])
    n_rows = t * r.h
    n_blocks = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
    alpha_c = list(alpha_consts) + [None] * (MAX_ALPHA - r.n_alpha)
    alpha_o = list(alpha_outs) + [None] * (MAX_ALPHA - r.n_alpha)
    r.compiled(
        src,
        dst,
        _slot_view(scale),
        _slot_view(amax),
        _slot_view(scale_out),
        _slot_view(descale),
        *[_slot_view(c) for c in alpha_c],
        *[_slot_view(a) for a in alpha_o],
        cutlass.Int32(n_rows),
        cutlass.Int32(r.h),
        cutlass.Int32(n_blocks),
        cuda.CUstream(int(stream)),
    )


def moved_bytes(t: int, h: int, d: int, *, src_elem_bytes: int = 2) -> int:
    """HBM traffic of one launch: the bf16/f16 read plus the 1-byte fp8 write.  The 4-byte scale is noise."""
    return t * h * d * (src_elem_bytes + 1)


def amax_moved_bytes(t: int, h: int, d: int, *, src_elem_bytes: int = 2) -> int:
    """HBM traffic of one amax pass: the bf16/f16 read only (one 4-byte atomic per warp is noise)."""
    return t * h * d * src_elem_bytes


# ---------------------------------------------------------------------------
# The amax pass: amax = max |fp32(src)| over [T, H, D], into a pre-zeroed fp32 slot
# ---------------------------------------------------------------------------


@cute.kernel
def frost_amax_abs(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token/head strides (compact or a slab column slice)
    mAmax: cute.Tensor,  # [1] fp32, PRE-ZEROED: one int32 atomicMax of the warp's max |x| bit pattern per warp
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
) -> None:
    """The quantize kernel's lane layout (16 elements per lane, two ``ld.global.v4``), a ternary abs-max tree per lane
    (``abs_max_tree``), the warp butterfly (``warp_abs_max_f32_shfl``) and ONE ``atomicMax`` per warp.  Tail rows clamp
    their loads to the last valid row (a redundant read, never a wild one) and are SELECTED out of the fold."""
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(chunks_per_lane(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    warp_lane = tidx % cutlass.Int32(32)
    row0 = (cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)
    src_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(SRC_BYTES_PER_LANE)

    # PASS 1: every load in flight first (the quantize kernel's structure).
    lives = []
    srcs = []
    for r in cutlass.range_constexpr(rows_per_group):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        token = row_r // _h
        head = row_r % _h
        src_addr = mSrc.iterator.toint() + (
            token.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[1])
        ) * cutlass.Int64(2)
        vals = []
        for c in cutlass.range_constexpr(chunks):
            base = src_addr + cutlass.Int64((c * lanes) * SRC_BYTES_PER_LANE) + src_lane_off
            for j in cutlass.range_constexpr(LOADS_PER_LANE):
                for w in ld_global_v4(base + cutlass.Int64(j * 16), cutlass.Int32):
                    lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                    vals.append(lo)
                    vals.append(hi)
        lives.append(row < n_rows)
        srcs.append(vals)

    # PASS 2: the per-lane fold (a SELECT drops a clamped tail row -- it duplicates the last valid row, so the max would not
    # change, but the fold is over the rows that exist), then the warp max and one atomic from lane 0.  Every lane of the
    # warp reaches the butterfly: nothing above diverges.
    acc = opaque_f32_zero()
    for r in cutlass.range_constexpr(rows_per_group):
        m = abs_max_tree(srcs[r])
        acc = fmax_f32(acc, m) if lives[r] else acc
    wmax = warp_abs_max_f32_shfl(acc)
    if warp_lane == cutlass.Int32(0):
        atomic_max_f32_bits(mAmax, wmax)


@cute.jit
def amax_abs_launch(
    src: cute.Tensor,
    amax: cute.Tensor,
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    n_blocks: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_amax_abs(src, amax, n_rows, h, h_ct, const_head_count, d, threads_per_cta, rows_per_group).launch(
        grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream
    )


amax_compiled_cache = {}


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
    Build from SHAPES ALONE -- no allocation, no launch.  Needs no fp8 instruction: runs on every CUDA device."""
    global _FAKE_STREAM
    validate_shape(d, threads_per_cta)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"amax serves bf16/f16 sources only, got {dtype_in}")
    if threads_per_cta % 32 != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of 32: the fold ends in a full-warp butterfly (shfl.sync over 32 lanes)")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (str(dtype_in), h, d, int(threads_per_cta), int(rows_per_group), bool(const_head_count), current_device())
    if key not in amax_compiled_cache:
        tok = cute.sym_int()
        src = fake_rowmajor_dynamic_token_stride(dtype_in, tok, h, d)
        amax_compiled_cache[key] = cute.compile(
            amax_abs_launch,
            src,
            _fake_slot(),
            cutlass.Int32(0),  # n_rows   ) runtime; the zeros pin the TYPE only
            cutlass.Int32(h),  # h        )
            cutlass.Int32(0),  # n_blocks )
            int(h),
            bool(const_head_count),
            d,
            int(threads_per_cta),
            int(rows_per_group),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return AmaxRecipe(compiled=amax_compiled_cache[key], h=h, d=d, rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group, dtype_in=dtype_in)


def run_amax(r: AmaxRecipe, src: torch.Tensor, amax: torch.Tensor, *, stream) -> None:
    """Launch.  ``src`` ``[T, H, D]`` bf16 / f16 under the quantize kernel's layout rules (head stride ``D``, element stride 1,
    a 16-byte-aligned token stride); ``amax`` a 1-element fp32 CUDA view PRE-ZEROED by the caller on the same stream (an
    ``atomicMax`` target: a poisoned slot is the caller's bug -- the launch can only RAISE it).  Host checks only."""
    if src.dtype != r.dtype_in:
        raise ValueError(f"src is {src.dtype} but this artifact was compiled for {r.dtype_in}")
    if src.ndim != 3 or int(src.shape[1]) != r.h or int(src.shape[2]) != r.d:
        raise ValueError(f"src must be [T, H={r.h}, D={r.d}], got {tuple(src.shape)}")
    if src.stride(2) != 1 or src.stride(1) != r.d:
        raise ValueError(f"src must have head stride D={r.d} and element stride 1 (a column slice of the slab or compact), got strides {src.stride()}")
    if (src.stride(0) * 2) % 16:
        raise ValueError(f"the token stride must keep every row 16-byte aligned: src {src.stride(0)} elems (bf16/f16)")
    if int(src.shape[0]) < 1:
        raise ValueError("src has no rows (T == 0): the amax of nothing is the slot's zero, and a zero-size grid is a launch error, not a no-op")
    check_scalar_slot("amax", amax)
    if not src.is_cuda or amax.device != src.device:
        raise ValueError(f"src and amax must be on one CUDA device (the kernel reads both through device pointers), got {src.device} and {amax.device}")
    t = int(src.shape[0])
    n_rows = t * r.h
    n_blocks = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
    r.compiled(src, _slot_view(amax), cutlass.Int32(n_rows), cutlass.Int32(r.h), cutlass.Int32(n_blocks), cuda.CUstream(int(stream)))


# ---------------------------------------------------------------------------
# The scalar-block init: slots[0:n_slots] = 0, then descale_dp_out[0] = 1 / scale_dp[0]
# ---------------------------------------------------------------------------


@cute.kernel
def _init_scalars(
    mSlots: cute.Tensor,  # [n_slots] fp32 contiguous: the scalar block
    mScaleDp: cute.Tensor,  # [1] fp32: the caller's scale_dP (outside the block)
    mDescaleDpOut: cute.Tensor,  # [1] fp32 OUT: 1 / scale_dP (may be one of the slots)
    n_slots: cutlass.Constexpr[int],
) -> None:
    """ONE thread, every store an ordered inline-PTX ``st.global`` (the ``_zero_amax`` idiom of the fp8 SDPA host): the
    zeroing first, then the reciprocal -- which may legally land on a slot just zeroed, because asm stores keep program
    order whatever the compiler assumes about the two pointers."""
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    if tidx == cutlass.Int32(0):
        zero = opaque_f32_zero()
        base = mSlots.iterator.toint()
        for i in cutlass.range_constexpr(n_slots):
            # a 4-B pitch = the fp32 element size: the block's slot stride (api_bwd.QUANT_SCALAR_STRIDE, pinned equal to it by
            # api_bwd._plan_bwd_workspace) -- the readers' slot views sit at that stride, so the two must move together
            st_global(base + cutlass.Int64(i * 4), zero, cutlass.Float32)
        descale = div_rn_f32(zero + cutlass.Float32(1.0), _slot_value(mScaleDp))
        st_global(mDescaleDpOut.iterator.toint(), descale, cutlass.Float32)


@cute.jit
def init_scalars_launch(slots: cute.Tensor, scale_dp: cute.Tensor, descale_dp_out: cute.Tensor, n_slots: cutlass.Constexpr[int], stream: cuda.CUstream):
    _init_scalars(slots, scale_dp, descale_dp_out, n_slots).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


init_compiled_cache = {}


class InitScalarsRecipe(NamedTuple):
    """Build-time facts of the scalar-block init launch: the number of fp32 slots it zeroes."""

    compiled: object
    n_slots: int


def compile_init_scalars(n_slots: int) -> InitScalarsRecipe:
    """ONE thread: ``slots[0:n_slots] = 0.0``, then ``descale_dp_out[0] = 1.0 / scale_dp[0]`` -- every amax slot must be zero
    before the first ``atomicMax`` of the first pass, and the reciprocal is derived on device (one ``div.rn.f32``, exact for
    a power of two) so no second caller input can disagree with ``scale_dp``.  Build from the slot count alone."""
    global _FAKE_STREAM
    if isinstance(n_slots, bool) or not isinstance(n_slots, int) or n_slots < 1:
        raise ValueError(f"n_slots must be a positive int (the fp32 slots of the scalar block), got {n_slots!r}")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (int(n_slots), current_device())
    if key not in init_compiled_cache:
        slots = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (int(n_slots),), stride_order=(0,), assumed_align=4)
        init_compiled_cache[key] = cute.compile(init_scalars_launch, slots, _fake_slot(), _fake_slot(), int(n_slots), _FAKE_STREAM, options="--enable-tvm-ffi")
    return InitScalarsRecipe(compiled=init_compiled_cache[key], n_slots=int(n_slots))


def run_init_scalars(r: InitScalarsRecipe, slots: torch.Tensor, scale_dp: torch.Tensor, descale_dp_out: torch.Tensor, *, stream) -> None:
    """Launch.  ``slots`` the contiguous fp32 ``[n_slots]`` view of the scalar block; ``scale_dp`` the caller's 1-element fp32
    CUDA scalar -- OUTSIDE the block (a slot would be zeroed before it is read: 1 / 0); ``descale_dp_out`` a 1-element fp32
    view, INTO the same block or not (written after the zeroing, sequentially in one thread, so no race).  Host checks only."""
    check_scalar_slot("slots", slots, numel=int(r.n_slots))
    check_scalar_slot("scale_dp", scale_dp)
    check_scalar_slot("descale_dp_out", descale_dp_out)
    lo, hi = slots.data_ptr(), slots.data_ptr() + 4 * int(r.n_slots)
    if lo <= scale_dp.data_ptr() < hi:
        raise ValueError("scale_dp lies inside the slot block this launch zeroes: it would be read as 0 (descale_dp = inf); pass the caller's own scalar")
    if scale_dp.device != slots.device or descale_dp_out.device != slots.device:
        raise ValueError(f"slots, scale_dp and descale_dp_out must be on one CUDA device, got {slots.device}, {scale_dp.device}, {descale_dp_out.device}")
    r.compiled(slots, _slot_view(scale_dp), _slot_view(descale_dp_out), cuda.CUstream(int(stream)))


frost_quantize_fp8.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_amax_abs.set_name_prefix("cudnn", remove_cutlass_symbol=True)
_init_scalars.set_name_prefix("cudnn", remove_cutlass_symbol=True)
