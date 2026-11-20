# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Row kernels: one thread block per row, and the reduction of one value over that block.

A row kernel launches one thread block per row.  Every thread owns a few elements of the row, reduces them to one value in
registers, and the block then needs ONE result in every thread: the row's sum of squares, its maximum magnitude, its maximum
with the position of that maximum.  :func:`block_reduce` is that step -- warp shuffles, one shared-memory word per warp, one
thread-block barrier, a second warp level -- with the things that decide a kernel's bytes and its safety stated as arguments
instead of being re-derived in every kernel.

THE ORDER OF A SUM IS PART OF THE BYTES.  fp32 addition is not associative, so two reductions of the same 640 partial sums that
pair the lanes in another order return totals that differ in the last bit on some rows, and a normalized row then differs in
some elements.  ``order`` therefore has no default for ``kind="sum"``:

* ``"ascending"``: inside each warp five butterfly steps with the offsets 1, 2, 4, 8, 16 (``v = v + shuffle(v, lane ^ offset)``);
  lane 0 of each warp stores the warp's sum to the warp's slot; the barrier; lanes ``0 .. warps - 1`` of every warp load slot
  ``lane``, the other lanes hold ``+0``; the same five steps.  Defined for every warp count 1 .. 32.  Below 32 warps the padding
  lanes add ``+0``, so a total of ``-0`` (every addend ``-0``) is returned as ``+0``; a sum of squares has no such case.
* ``"descending"``: inside each warp the offsets 16, 8, 4, 2, 1; the same publish and barrier; then EVERY lane loads slot
  ``lane mod warps`` (no padding, no zero is added) and adds over the offsets ``warps / 2 .. 1``.  ``5 + log2(warps)`` shuffles.
  Defined for a POWER-OF-TWO warp count only: for any other count no second level with this association exists, and the call is
  refused at trace time rather than given an invented one.

Every order leaves the same bits in every thread of the block.  ``"max"`` and ``"max_index"`` are exact in any order and take
``order=None``.

WHAT A CALL EMITS.  No mbarrier, no cluster operation, no loop with a run-time trip count.

    barrier                        kind                participants     issuing lanes              guard
    the thread-block barrier       named barrier 0,    32 x warps       every thread, once:        the CALLER: every thread of the
    of this reduction              the aligned form    (the launch's    SUM = 32 x warps           block reaches the call, under
                                                       thread count)    == participants            compile-time control flow only

    buffer        dtype x words                     writer                        reader                               refilled in a launch
    warp slots    fp32 x slot_words(kind, warps)    lane 0 of warp w: word w      behind the barrier: lanes            no: one store phase, one
                                                    (and warps + w for the        0 .. warps-1 of every warp,          load phase -> nothing
                                                    index), before the barrier    word = lane                          releases it

Lanes address the slots at a 4-byte stride (one word per lane in an access phase: never a multiple of the 128-byte bank cycle);
no descriptor reads them, so there is no swizzle.
Under ``"descending"`` the reader is EVERY lane, at word ``lane mod warps``.

CALLER OBLIGATIONS.

* ``lane`` and ``warp`` are the CALLER's ``thread index % 32`` and ``thread index // 32``.  Where those two values are computed
  is part of the caller's rendering (computed inside a helper, the same expression can be emitted behind a dependent-launch wait
  that the caller placed in front of the call); a kernel that keeps a compiled form byte for byte keeps its own two statements.
* The block is ``warps`` FULL warps and every thread reaches the call exactly once, never from a warp-role branch: the shuffles
  use the full-warp mask and the barrier is the aligned thread-block form.
* ONE slot array per call, from the kernel's one shared-memory allocator.  A second reduction takes its own array: two calls on
  one array would refill a buffer behind a barrier while loads of the first fill can still be pending, which is the hazard
  ``tile_dsl.consume`` exists for.
* No value is a NaN.  A ``"max"`` or ``"max_index"`` value is any other fp32 value, of either sign: the lanes past the warp count
  hold ``-inf``, the identity of a maximum, so the result is the block's maximum at every warp count (a kernel's maximum of
  magnitudes is the usual case and never sees the identity).  The warp maximum is ``redux.sync.max.f32`` where the compile
  target is known to assemble it (compute capability 10.x) and five butterfly steps of ``max.f32`` everywhere else.  On inputs
  without a NaN the two arms return the same maximum -- bit for bit, except the sign of a zero maximum that zeros of both signs
  hold, which a maximum of magnitudes never has -- and, with an index, the same index; neither is the ``.NaN`` form, and what
  they return for a NaN operand is not part of the contract.

The row helpers at the end of the module -- one 128-bit load of eight 16-bit values, the two eight-element stores, the per-row
FP8 scale -- are plain compositions of ``tile_dsl.pointwise`` and ``tile_dsl.tma`` helpers with the element-to-byte order stated
once.
"""

import cutlass
import cutlass.cute as cute

from .pointwise import (
    div_rn_f32,
    f16x2_to_f32,
    fmax_f32,
    fp32_to_fp4_pack,
    fp32_to_fp16,
    fp32_to_fp8x2,
    min_xorsign_abs_f32,
    opaque_f32_bits,
    pack_fp8x2_pairs,
    rcp_div_rn,
)
from .tma import ld_global_v4, st_global_v2, st_global_v4

WARP = 32
BUTTERFLY_STEPS = 5  # the offsets 1, 2, 4, 8, 16: the 32 lanes of a warp
SUM_ORDERS = ("ascending", "descending")
KINDS = ("sum", "max", "max_index")


def _arch_has_redux_f32(arch) -> bool:
    """Whether ``redux.sync.max.f32`` is known to assemble for the DSL architecture ``arch``: compute capability 10.x.  A POSITIVE
    list -- a target nobody named (older, newer, or of another line) takes the shuffle arm, which assembles everywhere."""
    return int(arch.major) == 10


def _target_has_redux_f32() -> bool:
    """:func:`_arch_has_redux_f32` of the target the DSL is compiling for.  Read once per :func:`block_reduce` call, in plain
    Python at trace time, and handed to the traced bodies as a compile-time argument."""
    from cutlass.cutlass_dsl import CuTeDSL

    return _arch_has_redux_f32(CuTeDSL._get_dsl().get_arch_enum())


def _check_warps(warps, who: str) -> None:
    if isinstance(warps, bool) or not isinstance(warps, int) or not 1 <= warps <= WARP:
        raise ValueError(f"{who}: warps must be an int in 1 .. {WARP} (the block's full warps), got {warps!r}")


def slot_words(kind: str, warps: int) -> int:
    """The fp32 shared-memory words ONE :func:`block_reduce` call of ``kind`` needs at ``warps`` warps: ``warps`` for ``"sum"`` and
    ``"max"``, ``2 * warps`` for ``"max_index"`` (the warp maxima, then the bits of their indices).  Plain arithmetic: usable where
    a kernel sizes its allocation and in a config module without the DSL's tracing."""
    if kind not in KINDS:
        raise ValueError(f"slot_words: kind must be one of {KINDS}, got {kind!r}")
    _check_warps(warps, "slot_words")
    return 2 * warps if kind == "max_index" else warps


def _check_slots(element_type, words, kind: str, warps: int) -> None:
    """The slot tensor of one call: fp32, and -- where its size is a compile-time fact -- at least ``slot_words(kind, warps)``
    words.  A smaller array is a shared-memory write past its end that no test of the result need notice."""
    if element_type != cutlass.Float32:
        raise ValueError(f"block_reduce: slots must be an fp32 shared-memory tensor, got element type {element_type}")
    if isinstance(words, int) and not isinstance(words, bool) and words < slot_words(kind, warps):
        raise ValueError(f"block_reduce: kind={kind!r} at {warps} warps needs {slot_words(kind, warps)} slot words, the tensor holds {words}")


@cute.jit
def _block_sum_ascending(value, lane, warp, slots, warps: cutlass.Constexpr[int]):
    for step in cutlass.range_constexpr(BUTTERFLY_STEPS):
        value = value + cute.arch.shuffle_sync_bfly(value, offset=1 << step)
    if lane == 0:
        slots[warp] = value
    cute.arch.barrier()
    total = cutlass.Float32(0)
    if lane < warps:
        total = slots[lane]
    for step in cutlass.range_constexpr(BUTTERFLY_STEPS):
        total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << step)
    return total


@cute.jit
def _block_sum_descending(value, lane, warp, slots, warps: cutlass.Constexpr[int], levels: cutlass.Constexpr[int]):
    for step in cutlass.range_constexpr(BUTTERFLY_STEPS):
        value = value + cute.arch.shuffle_sync_bfly(value, offset=(WARP // 2) >> step)
    if lane == 0:
        slots[warp] = value
    cute.arch.barrier()
    total = slots[lane & (warps - 1)]  # every lane loads: no padding lane, no zero addend
    for step in cutlass.range_constexpr(levels):
        total = total + cute.arch.shuffle_sync_bfly(total, offset=(warps // 2) >> step)
    return total


@cute.jit
def _warp_max(value, redux: cutlass.Constexpr[bool]):
    """The fp32 maximum over the 32 lanes of the warp, in every lane: one ``redux.sync.max.f32`` under ``redux``, else five
    butterfly steps of ``max.f32``."""
    if cutlass.const_expr(redux):
        value = cute.arch.warp_redux_sync(value, "fmax")
    else:
        for step in cutlass.range_constexpr(BUTTERFLY_STEPS):
            value = fmax_f32(value, cute.arch.shuffle_sync_bfly(value, offset=1 << step))
    return value


@cute.jit
def _block_max(value, lane, warp, slots, warps: cutlass.Constexpr[int], redux: cutlass.Constexpr[bool]):
    value = _warp_max(value, redux)
    if lane == 0:
        slots[warp] = value
    cute.arch.barrier()
    total = cutlass.Uint32(0xFF800000).bitcast(cutlass.Float32)  # -inf: the identity of a maximum, for values of either sign
    if lane < warps:
        total = slots[lane]
    return _warp_max(total, redux)


@cute.jit
def _block_max_index(value, index, lane, warp, slots, warps: cutlass.Constexpr[int], index_limit: cutlass.Constexpr[int], redux: cutlass.Constexpr[bool]):
    warp_top = _warp_max(value, redux)
    candidate = cutlass.Int32(index_limit)
    if value == warp_top:
        candidate = index
    warp_first = cute.arch.warp_redux_sync(candidate, "min")
    if lane == 0:
        slots[warp] = warp_top
        slots[warps + warp] = warp_first.bitcast(cutlass.Float32)  # the index travels as its 32 bits
    cute.arch.barrier()
    slot_top = cutlass.Uint32(0xFF800000).bitcast(cutlass.Float32)  # -inf: the identity of the value maximum
    slot_first = cutlass.Int32(index_limit)
    if lane < warps:
        slot_top = slots[lane]
        slot_first = slots[warps + lane].bitcast(cutlass.Int32)
    top = _warp_max(slot_top, redux)
    winner = cutlass.Int32(index_limit)
    if slot_top == top:
        winner = slot_first
    return top, cute.arch.warp_redux_sync(winner, "min")


def block_reduce(value, lane, warp, slots, warps, *, kind, order, index_limit=None):
    """Reduce one value per thread over the whole thread block; the result is returned in EVERY thread.

    ``value``        ``"sum"``: the thread's partial sum (``Float32``).  ``"max"``: the thread's maximum (``Float32``, of
                     either sign; usually a maximum of magnitudes).  ``"max_index"``: a pair ``(Float32, Int32)`` -- the
                     thread's best value and its index, an index in ``[0, index_limit)``.  No value is a NaN.
    ``lane, warp``   the caller's ``thread index % 32`` and ``thread index // 32`` (traced ``Int32``), see the module text.
    ``slots``        an fp32 shared-memory tensor of at least ``slot_words(kind, warps)`` words, used by THIS call only.
    ``warps``        a Python int, the block's full warps, 1 .. 32.
    ``kind``         keyword-only, required: ``"sum"``, ``"max"`` or ``"max_index"``.
    ``order``        keyword-only, required: one of ``SUM_ORDERS`` for a sum (it changes bytes); ``None`` for the two maxima.
    ``index_limit``  ``"max_index"`` only, required there: a Python int above every index (at most ``2**31 - 1``).

    Returns a ``Float32`` -- or ``(Float32, Int32)`` for ``"max_index"``: the block's maximum and the LOWEST index that holds
    it.  When a maximum is a zero and zeros of both signs hold it, the sign of the returned zero is not part of the contract
    (with an index: the index is the lowest of them).

    Plain Python at trace time: every refusal below is a ``ValueError`` raised before any operation is emitted -- an unknown
    ``kind`` or ``order``, an ``order`` given to a maximum, ``warps`` that is not an int in 1 .. 32, a missing or non-int
    ``index_limit``, an ``index_limit`` given to another kind, a slot tensor that is not fp32 or is statically smaller than the
    call needs.
    ``order="descending"`` with a warp count that is not a power of two is refused the same way."""
    if kind not in KINDS:
        raise ValueError(f"block_reduce: kind must be one of {KINDS}, got {kind!r}")
    _check_warps(warps, "block_reduce")
    if kind != "max_index" and index_limit is not None:
        raise ValueError(f"block_reduce: index_limit belongs to kind='max_index', got it with kind={kind!r}")
    if kind == "sum":
        if order not in SUM_ORDERS:
            raise ValueError(f"block_reduce: a sum needs order in {SUM_ORDERS} (the order changes the result's bits), got {order!r}")
        if order == "descending" and warps & (warps - 1):
            raise ValueError(f"block_reduce: order='descending' is defined for a power-of-two warp count, got {warps}")
    elif order is not None:
        raise ValueError(f"block_reduce: kind={kind!r} is exact in any order; pass order=None, got {order!r}")
    if kind == "max_index" and (isinstance(index_limit, bool) or not isinstance(index_limit, int) or not 1 <= index_limit <= 2**31 - 1):
        raise ValueError(f"block_reduce: kind='max_index' needs index_limit, an int in 1 .. 2**31 - 1 above every index, got {index_limit!r}")
    _check_slots(slots.element_type, cute.size(slots), kind, warps)

    if kind == "sum":
        if order == "descending":
            return _block_sum_descending(value, lane, warp, slots, warps, warps.bit_length() - 1)
        return _block_sum_ascending(value, lane, warp, slots, warps)
    redux = _target_has_redux_f32()
    if kind == "max":
        return _block_max(value, lane, warp, slots, warps, redux)
    best, index = value
    return _block_max_index(best, index, lane, warp, slots, warps, index_limit, redux)


# ---------------------------------------------------------------------------
# Row helpers: eight 16-bit elements per thread, as one vector.  ``addr`` is the byte address of the vector's first element
# (``(tensor.iterator + element).toint()``); the vector is 16 bytes for the 16-bit forms and 8 bytes for the E4M3 codes, and
# the caller keeps it aligned to its own size.  Element ``j`` of a list is the element at the j-th position of the vector, in
# memory order, in every helper.
# ---------------------------------------------------------------------------
ROW_VEC = 8  # elements per thread per vector
E4M3_MAX_BITS = 0x43E00000  # fp32(448): the largest E4M3 magnitude
FP8_ROW_SCALE_FLOOR_BITS = 0x36924925  # fp32(1 / (448 * 512)): the floor of a per-row scale


def _check_vec(values, who: str) -> None:
    if not isinstance(values, (list, tuple)) or len(values) != ROW_VEC:
        raise ValueError(f"{who}: values must be a list or tuple of {ROW_VEC} Float32 values, got {values!r}")


def row_load8(addr, *, dtype):
    """ONE 128-bit global load of eight 16-bit values at ``addr``, widened exactly to fp32: a list of eight ``Float32`` whose
    element ``j`` is the value at ``addr + 2 j`` bytes.

    ``dtype`` is keyword-only and REQUIRED: ``BFloat16`` (a bit move) or ``Float16`` (a convert).  It decides the decoded values
    -- the fp16 pattern of 1.0 read as bf16 is 0.0078125 -- and the conversion helpers of ``tile_dsl.pointwise`` default to
    ``Float16``, so no default here would be right for both kinds of row.  Any other ``dtype`` is a ``ValueError`` raised before
    any operation is emitted."""
    if dtype not in (cutlass.BFloat16, cutlass.Float16):
        raise ValueError(f"row_load8: dtype must be cutlass.BFloat16 or cutlass.Float16, got {dtype!r}")
    words = ld_global_v4(addr, cutlass.Int32)
    out = []
    for pair in range(ROW_VEC // 2):
        lo, hi = f16x2_to_f32(words[pair], dtype=dtype)
        out.extend((lo, hi))
    return out


def row_store8_bf16(addr, values) -> None:
    """ONE 128-bit global store of eight fp32 values rounded to bf16 (nearest-even): ``values[j]`` goes to ``addr + 2 j`` bytes."""
    _check_vec(values, "row_store8_bf16")
    words = [fp32_to_fp16(values[2 * pair], values[2 * pair + 1], dtype=cutlass.BFloat16) for pair in range(ROW_VEC // 2)]
    st_global_v4(addr, words, cutlass.Int32)


def row_store8_e4m3(addr, values, inv, *, clamp) -> None:
    """ONE 64-bit global store of the eight E4M3 codes of ``values[j] * inv``: the code of ``values[j]`` is the BYTE at
    ``addr + j``.

    Per element one fp32 multiply by ``inv`` (the reciprocal of the dequantization scale, e.g. from
    :func:`fp8_dynamic_row_scale`), then under ``clamp`` one ``min.xorsign.abs.f32`` against 448
    (:func:`~cudnn.frost.tile_dsl.pointwise.min_xorsign_abs_f32`), then the saturating nearest-even convert.  The convert
    saturates by itself, so ``clamp`` changes no code of a finite or infinite product: it is the instruction a kernel carries
    when its instruction stream follows a reference that clamps.  ``clamp`` is keyword-only, required, a Python bool."""
    _check_vec(values, "row_store8_e4m3")
    if not isinstance(clamp, bool):
        raise ValueError(f"row_store8_e4m3: clamp must be a Python bool, got {clamp!r}")
    limit = opaque_f32_bits(E4M3_MAX_BITS) if clamp else None
    pairs = []
    for pair in range(ROW_VEC // 2):
        lo = values[2 * pair] * inv
        hi = values[2 * pair + 1] * inv
        if clamp:
            lo = min_xorsign_abs_f32(lo, limit)
            hi = min_xorsign_abs_f32(hi, limit)
        pairs.append(fp32_to_fp8x2(lo, hi))
    st_global_v2(addr, [pack_fp8x2_pairs(pairs[0], pairs[1]), pack_fp8x2_pairs(pairs[2], pairs[3])], cutlass.Int32)


def fp8_dynamic_row_scale(amax):
    """``(scale, inv)`` of one row quantized to E4M3 with its own scale: ``scale = max(div.rn(amax, 448), fp32(1 / (448 * 512)))``
    and ``inv = div.rn(1, scale)`` -- two IEEE divisions and one exact maximum, so both are functions of ``amax`` alone on every
    target.  ``amax`` is the row's non-negative maximum magnitude (``block_reduce(..., kind="max")``); a zero row gives the
    floor.  The row's codes are ``e4m3(x * inv)`` (:func:`row_store8_e4m3`) and ``scale`` is what the consumer multiplies by.

    NOT :func:`~cudnn.frost.tile_dsl.pointwise.e4m3_scale_from_amax` (the block scale of NVFP4: another rule, another floor)."""
    scale = fmax_f32(div_rn_f32(amax, opaque_f32_bits(E4M3_MAX_BITS)), opaque_f32_bits(FP8_ROW_SCALE_FLOOR_BITS))
    return scale, rcp_div_rn(scale)


# ---------------------------------------------------------------------------
# Row helper of a sixteen-element-per-thread kernel: one block of a 4-bit (E2M1) code row.  ``addr`` is the byte address of
# the block's first code byte; the block is 8 bytes and the caller keeps it aligned to 8.
# ---------------------------------------------------------------------------
ROW_BLOCK = 16  # elements per thread per block: the elements one block scale of NVFP4 covers


def row_store16_e2m1(addr, values, encode) -> None:
    """ONE 64-bit global store of the sixteen E2M1 codes of ``values[j] * encode``: the code of ``values[j]`` is NIBBLE ``j`` of
    the eight bytes at ``addr`` -- the low nibble of byte ``j // 2`` for an even ``j``, the high nibble for an odd one.

    Per element one fp32 multiply by ``encode`` (the block's encode scale, e.g. from
    :func:`~cudnn.frost.tile_dsl.pointwise.nvfp4_block_scale_global`), then the saturating nearest-even convert of each pair
    (:func:`~cudnn.frost.tile_dsl.pointwise.fp32_to_fp4_pack`: ties to the even code, saturation at +-6, the sign kept on a
    zero).  The 4-bit sibling of :func:`row_store8_e4m3`: sixteen elements per thread, because one block scale covers sixteen."""
    if not isinstance(values, (list, tuple)) or len(values) != ROW_BLOCK:
        raise ValueError(f"row_store16_e2m1: values must be a list or tuple of {ROW_BLOCK} Float32 values, got {values!r}")
    words = fp32_to_fp4_pack([v * encode for v in values])
    st_global_v2(addr, [words[0], words[1]], cutlass.Int32)
