# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Register-direct GLOBAL reduce-adds (``red.global.add``): the fire-and-forget twin of ``tma.tma_reduce_add_subtile``.

ONE primitive and its sector-paired form.  A ``red`` carries no return value and nothing inside the kernel waits on it:
the data leaves the SM through the LSU global path (no SMEM staging, no bulk-async commit / wait group), and kernel
completion is the only completion a consumer can observe -- a buffer reduced this way is read by the NEXT launch in
stream order (the d128 backward's ``cast_dq`` kernel), never inside the issuing kernel.  Default semantics ``.relaxed``
at ``.gpu`` scope, like the TMA reduce.  The vector forms ``.v2 / .v4 .f32`` exist from sm_90 (PTX ISA 8.1): the
host gate is :func:`requirements.red_global_v4_requirement_error` (``RED_GLOBAL_V4_F32_MIN_CC``), imported from the
cutlass-free module BEFORE this one on a host path.

Caller contract (G10 authoring rules): element-addressed arithmetic on the caller's side, a 16-B-ALIGNED byte address here;
predicate by BRANCHING around the call (an out-of-range row is skipped by an ``if``, never by ``inline_ptx(predicate=)``);
the four values come from registers (a tcgen05.ld image, an accumulator), never a foldable constant (opaque rule).
Sector efficiency: a 32x32b TMEM lane map hands each lane 16 B of ONE row per instruction, so a warp-level ``red.v4``
touches 32 distinct 128-B lines with 16 B each = HALF of a 32-B L2 sector per lane (RULE-GAPS 396 of the 2-CTA task);
:func:`red_global_add_f32x8_sector_paired` pairs lanes ``2k / 2k+1`` through one ``shfl.sync.bfly`` per word so that
every warp-level instruction carries FULL 32-B sectors (16 per instruction instead of 32 halves); the byte count is
identical, the L2 sector-op count halves.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.experimental import primitives as nvvm

from .requirements import RED_GLOBAL_V4_F32_MIN_CC, red_global_v4_requirement_error  # noqa: F401  (re-exported for callers that hold the DSL)

#: fp32 words per ``red.global.add.v4.f32`` and per 32-B L2 sector (the paired form's unit); bytes per fp32 word.
RED_V4_WORDS = 4
SECTOR_F32_WORDS = 8
F32_BYTES = 4
SECTOR_BYTES = SECTOR_F32_WORDS * F32_BYTES


def _byte_address(addr):
    """``Int64`` byte address from an ``Int64`` or a generic / global ``Pointer`` (the ``tma.ldg_int32x4`` spelling)."""
    if hasattr(addr, "toint"):
        addr = addr.toint(cutlass.Int64)
    return cutlass.Int64(addr)


@cute.jit
def red_global_add_v4_f32(addr, values):
    """ONE ``red.global.add.v4.f32 [addr], {v0, v1, v2, v3}``: four fp32 of ``values`` (indexable, >= 4 ``Float32``) added into
    the 16-B-ALIGNED byte address ``addr`` (``Int64`` or a ``Pointer``).  Relaxed, gpu scope, no return value; the caller skips
    out-of-range rows by branching around the call."""
    nvvm.inline_ptx(
        "red.global.add.v4.f32 [$0], {$1, $2, $3, $4};",
        read_only_args=[_byte_address(addr), values[0], values[1], values[2], values[3]],
    )


@cute.jit
def _select_f32(pred, a, b):
    """``pred ? a : b`` on two ``Float32`` (one SEL; never an FMUL by a 0 / 1 mask: a NaN operand would survive it)."""
    return cutlass.Float32(arith.select(pred.ir_value(), a.ir_value(), b.ir_value()))


@cute.jit
def red_global_add_f32x8_sector_paired(addr_even_row, row_stride_bytes, parity, live_even, live_odd, values, first: cutlass.Constexpr[int]):
    """Two lane partners (``lane ^ 1``) reduce ONE 32-B sector each with two warp-level ``red.global.add.v4.f32`` that both carry
    FULL sectors.  Each lane holds 8 consecutive fp32 of its OWN row at ``values[first : first + 8]``; the even lane's row is at
    byte address ``addr_even_row`` (the column of ``values[first]``), the odd lane's row ``row_stride_bytes`` further (the SAME
    ``addr_even_row`` value on both lanes of a pair: the caller derives it from its own row address and its parity).

    instruction 0 = the EVEN row's sector: even lane -> ``[addr_even_row]`` with its low v4; odd lane -> ``[addr_even_row + 16]``
    with the even lane's HIGH v4 (fetched by ``shfl.sync.bfly 1``).  instruction 1 = the ODD row's sector: even lane ->
    ``[addr_even_row + row_stride_bytes]`` with the odd lane's LOW v4 (fetched), odd lane -> ``+ 16`` with its own high v4.
    The four shuffles run on EVERY lane of the warp (full mask) BEFORE either predicate; ``live_even`` / ``live_odd`` (the two rows'
    in-range predicates, identical on both lanes of the pair) branch around the two instructions.  ``parity`` = ``lane & 1`` (``Int32``).
    """
    is_odd = parity != cutlass.Int32(0)
    recv = []
    for j in cutlass.range_constexpr(RED_V4_WORDS):
        send = _select_f32(is_odd, values[first + j], values[first + RED_V4_WORDS + j])  # odd sends its LOW v4, even its HIGH v4
        recv.append(cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, send, cutlass.Int32(1), 31, kind=nvvm.Shfl.BFLY)))
    half = cutlass.Int64(RED_V4_WORDS * F32_BYTES) * cutlass.Int64(parity)  # the odd lane writes the sector's upper 16 B
    op_even = tuple(_select_f32(is_odd, recv[j], values[first + j]) for j in range(RED_V4_WORDS))
    op_odd = tuple(_select_f32(is_odd, values[first + RED_V4_WORDS + j], recv[j]) for j in range(RED_V4_WORDS))
    if live_even:
        red_global_add_v4_f32(addr_even_row + half, op_even)
    if live_odd:
        red_global_add_v4_f32(addr_even_row + row_stride_bytes + half, op_odd)


__all__ = [
    "F32_BYTES",
    "RED_GLOBAL_V4_F32_MIN_CC",
    "RED_V4_WORDS",
    "SECTOR_BYTES",
    "SECTOR_F32_WORDS",
    "red_global_add_f32x8_sector_paired",
    "red_global_add_v4_f32",
    "red_global_v4_requirement_error",
]
