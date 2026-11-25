# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FP8 row quantization DECODE kernel (bf16 in; E4M3 codes and one fp32 scale per row out; one launch): every row of a slab
quantized to E4M3 under its own scale, the row maximum over 448 with a floor.  Config in ``row_quantize_fp8_decode_config.py``, the
host entry in ``row_quantize/frost/row_quantize_fp8_decode.py``.  Arch-portable PTX (global loads / stores, ``max.f32``,
``div.rn.f32``, ``mul.f32``, ``min.xorsign.abs.f32``, ``cvt``, ``bar.sync``, ``griddepcontrol``) except the warp maximum, which
``tile_dsl.rowwise.block_reduce`` spells per target; the arch axis is an admission fact of the host entry, not a directory.

THREAD BLOCK: one per row; ``cfg.THREADS`` threads in ``cfg.WARPS`` FULL warps.  Thread ``t`` owns the vectors ``t + p x THREADS``
of its row, ``p = 0 .. cfg.PASSES - 1`` -- eight contiguous elements each, their eight code bytes -- where such a vector exists:
in the last pass only the first ``cfg.live_threads(p)`` threads own one.  Every thread has the same role; thread 0 additionally
stores the row's scale.

PIPELINE, in order, for every thread:

1. ``griddepcontrol.launch_dependents`` under ``cfg.PDL_RELEASE`` -- the kernel's first statement.
2. ``griddepcontrol.wait`` under ``cfg.PDL``.
3. Per owned vector ONE 128-bit global load of eight bf16 values; bf16 -> fp32 is exact.  A vector the thread does not own is
   not loaded: its eight values are ``+0``.
4. ``m = max |x_j|`` over the thread's values: a ternary tree of ``max.f32`` on magnitudes.
5. The row maximum: ``tile_dsl.rowwise.block_reduce(m, lane, warp, slots, WARPS, kind="max", order=None)`` -- the warp maximum,
   lane 0 stores it to the warp's slot, the thread-block barrier, the lanes below the warp count load slot ``lane``, the warp
   maximum again.
6. ``scale = max( div.rn(amax, 448), fp32(1 / (448 x 512)) )`` and ``inv = div.rn(1, scale)``
   (``tile_dsl.rowwise.fp8_dynamic_row_scale``).
7. Per owned vector: eight fp32 multiplies by ``inv``, the clamp to +-448 (``min.xorsign.abs.f32``), four saturating nearest-even
   pair converts, ONE 64-bit global store of the eight code bytes (``tile_dsl.rowwise.row_store8_e4m3``).
8. Thread 0: one 32-bit global store of ``scale``.

THE ROUNDING CHAIN IS THE CONTRACT.  The two divisions are the correctly rounded ones (never a reciprocal multiply or an
approximate reciprocal: either moves a code on a rounding boundary); the floor is the bit pattern ``0x36924925``; the product
``x x inv`` is rounded to fp32 before it is rounded to E4M3.  The maximum is exact in any order for a row without a NaN, so the
thread plan (``WIDTH``) changes no byte.  ``max.f32`` returns the OTHER operand when one operand is a NaN and the warp maximum of
a NaN is not specified, so on a row that holds a NaN -- and on a row that holds an infinity, whose scale is infinite -- this
kernel's bytes are unspecified; every other row is unaffected.

BATCH INVARIANCE AND REPLAY.  A thread block reads its own row and nothing of another row; the reduction never leaves the block;
there is no atomic and no cross-row scratch.  A row's bytes are a function of its values alone.  The row index enters ADDRESSES
only.

BARRIER TABLE.  No mbarrier, no cluster operation, no ``cp.async`` group, no loop with a run-time trip count.  ONE
synchronization object:

    barrier                         kind                 participants      issuing lanes            guard
    the thread-block barrier (5)    named barrier 0      cfg.THREADS       WARPS x 32, once each    none

``SUM(issuing lanes) = 32 x WARPS == THREADS``: every thread the barrier counts reaches it exactly once on every path.  The
run-time branches in front of it -- the load of a last-pass vector, lane 0's slot store inside the reduction -- rejoin before it;
the stores of step 7 and 8 sit behind it and hold no barrier.  The barrier and the slot accesses of the table below are emitted
by ``block_reduce``; the kernel reaches that call with every thread, outside any run-time branch.

SHARED-MEMORY TABLE.  ONE buffer; no tensor memory, no descriptor:

    buffer        dtype x elems    writer                          reader                               refilled in a launch
    warp maxima   fp32 x WARPS     lane 0 of warp w, slot w,       lanes 0 .. WARPS - 1 of every        no: written once, read
                                   before the barrier              warp, slot lane, behind it           once -> nothing releases it

Lanes address the slots directly at a 4-byte stride (one word per lane in an access phase); no swizzle because no descriptor
reads it.

PDL SITES.  Under ``cfg.PDL`` the launch carries the programmatic-stream-serialization attribute and every thread issues
``griddepcontrol.wait`` BEFORE every global read and every global store: the kernel reads nothing in front of it.  Under
``cfg.PDL_RELEASE`` (which needs ``cfg.PDL``) the release is the kernel's first statement, in front of its own wait: it hands out
nothing this kernel waits on.  CALLER OBLIGATION: every grid that can start before this one completes issues its own wait before
it reads this launch's outputs and before it writes any operand of this launch.  The kernel holds no tensor memory and
4 x WARPS bytes of shared memory.

OPERANDS are device ADDRESSES (``Int64``) and ``x``'s row stride in bytes: the kernel addresses flat and reads no extent, so no
shape, stride or alignment of an operand enters the compiled kernel (the host entry checks them before every launch).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import abs_max_tree
from cudnn.frost.tile_dsl.rowwise import block_reduce, fp8_dynamic_row_scale, row_load8, row_store8_e4m3, slot_words
from cudnn.frost.tile_dsl.tma import st_global

from .row_quantize_fp8_decode_config import ELEMENT_BYTES, SCALE_BYTES, VEC, WARP

FP32, INT64, BF16 = cutlass.Float32, cutlass.Int64, cutlass.BFloat16


@cute.kernel
def frost_row_quantize_fp8(
    cfg: cutlass.Constexpr,
    x: cutlass.Int64,  # device addresses: bf16 [rows, WIDTH] with the row stride below
    x_row: cutlass.Int64,  # byte stride between two rows of x
    codes: cutlass.Int64,  # E4M3 [rows, WIDTH], contiguous
    scales: cutlass.Int64,  # fp32 [rows], contiguous
):
    if cutlass.const_expr(cfg.PDL_RELEASE):
        launch_dependent_grids()  # the first statement: in front of this launch's own wait (the config pins PDL_RELEASE -> PDL)
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tid % WARP
    warp = tid // WARP
    if cutlass.const_expr(cfg.PDL):
        wait_on_dependent_grids()  # nothing is read in front of it
    slots = cutlass.memory.SmemAllocator().allocate_tensor(FP32, cute.make_layout(slot_words("max", cfg.WARPS)), byte_alignment=16)

    row64 = row.to(INT64)
    tid64 = tid.to(INT64)
    in_addr = x + row64 * x_row + tid64 * INT64(VEC * ELEMENT_BYTES)  # the thread's vector of pass 0
    out_addr = codes + row64 * INT64(cfg.WIDTH) + tid64 * INT64(VEC)

    # A vector past the row's end is neither loaded nor stored: the bytes there belong to the next row, or to nobody.
    vals = []
    for p in cutlass.range_constexpr(cfg.PASSES):
        owners = cfg.live_threads(p)
        vec = [FP32(0)] * VEC  # +0: the identity of a maximum of magnitudes
        if cutlass.const_expr(owners == cfg.THREADS):
            vec = row_load8(in_addr + INT64(p * cfg.THREADS * VEC * ELEMENT_BYTES), dtype=BF16)
        if cutlass.const_expr(owners != cfg.THREADS):
            if tid < owners:
                vec = row_load8(in_addr + INT64(p * cfg.THREADS * VEC * ELEMENT_BYTES), dtype=BF16)
        vals.append(vec)

    # byte contract: the row maximum of magnitudes (exact in any order without a NaN).  The call holds the ONE thread-block
    # barrier: all cfg.THREADS threads reach it here, outside any run-time branch.
    amax = block_reduce(abs_max_tree([value for vec in vals for value in vec]), lane, warp, slots, cfg.WARPS, kind="max", order=None)
    # byte contract: scale = max(div.rn(amax, 448), fp32(1 / (448 x 512))), inv = div.rn(1, scale); per element one fp32
    # multiply, the clamp to +-448, the saturating nearest-even convert; element j of a vector in byte j
    scale, inv = fp8_dynamic_row_scale(amax)
    for p in cutlass.range_constexpr(cfg.PASSES):
        owners = cfg.live_threads(p)
        if cutlass.const_expr(owners == cfg.THREADS):
            row_store8_e4m3(out_addr + INT64(p * cfg.THREADS * VEC), vals[p], inv, clamp=True)
        if cutlass.const_expr(owners != cfg.THREADS):
            if tid < owners:
                row_store8_e4m3(out_addr + INT64(p * cfg.THREADS * VEC), vals[p], inv, clamp=True)
    if tid == 0:
        st_global(scales + row64 * INT64(SCALE_BYTES), scale, FP32)


frost_row_quantize_fp8.set_name_prefix("cudnn", remove_cutlass_symbol=True)

#: the kernel's operand addresses, in launch order (the host entry builds a launch's arguments by these names)
OPERANDS = ("x", "x_row", "codes", "scales")


@cute.jit
def launch(
    cfg: cutlass.Constexpr, x: cutlass.Int64, x_row: cutlass.Int64, codes: cutlass.Int64, scales: cutlass.Int64, rows: cutlass.Int32, stream: cuda.CUstream
):
    """One launch: grid ``(rows, 1, 1)``, ``cfg.THREADS`` threads, the programmatic-stream-serialization attribute under
    ``cfg.PDL``.  ``rows`` and every address are runtime arguments; the config is the compile key."""
    frost_row_quantize_fp8(cfg, x, x_row, codes, scales).launch(grid=(rows, 1, 1), block=(cfg.THREADS, 1, 1), stream=stream, use_pdl=cfg.PDL)
