# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FP4 row quantization DECODE kernel (bf16 in; E2M1 codes and E4M3 block scales out; one launch): per block of sixteen elements
of a row, the block scale under a global scale and the block's sixteen 4-bit codes, in the convention of the block-scaled multiply
that consumes them.  Config in ``row_quantize_fp4_decode_config.py``, the host entry in ``row_quantize/frost/row_quantize_fp4_decode.py``.
Arch-portable PTX (global loads / stores, ``max.f32``, ``mul.f32``, ``cvt``, ``rcp.approx.ftz``, ``griddepcontrol``); the arch axis is
an admission fact of the host entry, not a directory.

THREAD BLOCK: one per row; ``cfg.THREADS = width / 16`` threads in ``cfg.WARPS`` FULL warps; thread ``t`` owns the WHOLE block ``t``
of its row -- the sixteen contiguous elements ``16 t .. 16 t + 15``, their eight code bytes and their one scale byte.  Every thread
has the same role and no thread reads another thread's value.

PIPELINE, in order, for every thread:

1. ``gs = scale[0]``: one 32-bit global load, IN FRONT of the wait (a constant of the layer: the PDL-sites paragraph).
2. ``griddepcontrol.wait`` under ``cfg.PDL``; ``griddepcontrol.launch_dependents`` under ``cfg.PDL_RELEASE``, directly behind the wait.
3. Two 128-bit global loads of the thread's sixteen bf16 values at element ``row x width + 16 t`` (a signed 32-bit element index:
   the config bounds ``rows x width``); bf16 -> fp32 is exact.
4. ``amax = max |x_j|``: a ternary tree of ``max.f32`` on magnitudes.
5. The block scale ``byte, value = e4m3( gs x (amax x fp32(1/6)) )`` -- ``amax x fp32(1/6)`` FIRST, two fp32 multiplies -- and the
   encode scale: ``0`` when ``value`` is zero, else ``rcp.approx.ftz( value x rcp.approx.ftz(gs) )``.
6. ``code_j = e2m1( x_j x encode scale )`` (nearest-even, saturating at +-6, the sign kept on a zero): sixteen multiplies, eight
   pair converts, ONE 64-bit global store of the block's eight code bytes (two codes per byte, the even element in the low nibble).
7. One byte store of the scale byte at the block's position in the 128 x 4 scale layout over (rows, blocks).

THE ROUNDING CHAIN IS THE CONTRACT.  Steps 5 and 6 are ``tile_dsl.pointwise.nvfp4_block_scale_global`` and
``tile_dsl.rowwise.row_store16_e2m1``: the factor 1/6 is the bit pattern ``0x3E2AAAAB`` inside the helper, never an approximate
reciprocal of a constant that an assembler folds; the two reciprocals of the encode scale are the hardware's approximate ones, inside
the run-time guard of the zero rule; no division appears.  The block maximum is exact in any order for a block without a NaN.  THE
MAXIMUM IS ``max.f32``, which returns the OTHER operand when one operand is a NaN, where a compare-and-select maximum keeps the NaN:
on a block that holds a NaN this kernel's bytes are unspecified and may differ from another kernel of the same convention; every
other block is unaffected.  The test module's rendering cell pins these instructions; its reference cells pin the bytes.

BATCH INVARIANCE AND REPLAY.  A thread reads its own sixteen values and the shared ``scale``, and nothing of another block or row;
there is no reduction, no atomic and no scratch.  A block's bytes are a function of its sixteen values and of ``scale`` alone.  The
row index enters ADDRESSES only -- including the position of a scale byte in the 128 x 4 layout, which moves with the row index
while its value does not.

BARRIER TABLE.  No mbarrier, no named barrier, no warp shuffle, no cluster operation, no ``cp.async`` group, no loop with a run-time
trip count:

    barrier    kind    participants    issuing lanes    guard
    (none)     --      0               0                every statement is per thread; the only run-time branch (the zero rule's
                                                        guard around the two reciprocals) joins before the sixteen multiplies

``SUM(issuing lanes) == 0 ==`` the count of every barrier: there is none.

SHARED-MEMORY TABLE.  No buffer: the launch requests 0 bytes of shared memory and no tensor memory.  Nothing is refilled inside a
launch, so nothing is released.

PDL SITES.  Under ``cfg.PDL`` the launch carries the programmatic-stream-serialization attribute and every thread issues
``griddepcontrol.wait`` AFTER the load of ``scale[0]`` and BEFORE its first read of ``x`` and before every global store.  CALLER
OBLIGATION: ``scale`` is not written by any launch that may still be running when this one starts (it is a constant of the layer).
Under ``cfg.PDL_RELEASE`` (which needs ``cfg.PDL``) the release follows this launch's OWN wait directly: a dependent's pre-wait part
can run beside this kernel, never beside this kernel's predecessor.  CALLER OBLIGATION: every grid that can start before this one
completes issues its own wait before it reads this launch's outputs and before it writes any operand of this launch.  The kernel
holds no shared and no tensor memory, so a co-resident dependent starves nothing.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import abs_max_tree, nvfp4_block_scale_global
from cudnn.frost.tile_dsl.rowwise import ROW_VEC, row_load8, row_store16_e2m1
from cudnn.frost.tile_dsl.sf_layout import sf_band_offset

from .row_quantize_fp4_decode_config import BLOCK

BF16, UINT8 = cutlass.BFloat16, cutlass.Uint8


@cute.kernel
def frost_row_quantize_fp4(cfg: cutlass.Constexpr, x: cute.Tensor, scale: cute.Tensor, codes: cute.Tensor, sf: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    # the global scale is a constant of the layer: it loads BEFORE the dependent-launch wait; the row behind it
    gs = scale[0]
    if cutlass.const_expr(cfg.PDL):
        wait_on_dependent_grids()
    if cutlass.const_expr(cfg.PDL_RELEASE):
        launch_dependent_grids()  # directly behind this launch's own wait (the config pins PDL_RELEASE -> PDL)
    elem = row * cfg.WIDTH + tid * BLOCK
    vals = row_load8((x.iterator + elem).toint(), dtype=BF16) + row_load8((x.iterator + (elem + ROW_VEC)).toint(), dtype=BF16)
    # byte contract: the block maximum of magnitudes (exact in any order without a NaN), then the block scale and the encode scale
    # of the global-scale convention: gs x (amax x fp32(1/6)), an E4M3 rounding, two approximate reciprocals behind the zero rule
    amax = abs_max_tree(vals)
    byte, encode = nvfp4_block_scale_global(amax, gs)
    # byte contract: sixteen fp32 multiplies, eight saturating nearest-even pair converts, ONE 64-bit store; element j in nibble j
    row_store16_e2m1((codes.iterator + (row * (cfg.WIDTH // 2) + tid * (BLOCK // 2))).toint(), vals, encode)
    sf[sf_band_offset(row, tid, cfg.SF_ATOMS)] = byte.to(UINT8)


frost_row_quantize_fp4.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch(cfg: cutlass.Constexpr, x, scale, codes, sf, rows: cutlass.Int32, stream: cuda.CUstream):
    """One launch: grid ``(rows, 1, 1)``, ``cfg.THREADS`` threads, no shared memory, the programmatic-stream-serialization attribute
    under ``cfg.PDL``.  ``rows`` is a runtime argument and the operand views are layout-dynamic; the config is the compile key."""
    frost_row_quantize_fp4(cfg, x, scale, codes, sf).launch(grid=(rows, 1, 1), block=(cfg.THREADS, 1, 1), stream=stream, use_pdl=cfg.PDL)
