# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""CuTe DSL MXFP8 forward megakernel: routed experts in MXFP8, the shared expert in BF16.

One persistent launch of 2-CTA clusters.  The first ``num_comm_sms / 2`` clusters are communication CTAs; the
others take compute tasks from the CLC scheduler and drain its capacity tail (``_mxfp8_fwd_base``).

Communication (``_forward_stream``, ``_run_stream``): one dynamic task stream of 32-row x 256-column blocks over
every macrobatch's dispatch and combine.

- The comm CTAs' warps 0-6 each own two 16 KiB buffers and take blocks from a global head word by atomic add
  (a group of four column blocks of one 32-row tile row per claim).
- The stream's blocks are, in order, the last macrobatch's dispatch pulls, then for each earlier macrobatch its
  combine pushes interleaved with the next-lower macrobatch's dispatch pulls (``_stream_task``).
- A pull block is one bulk copy per lane (lane = row) into the warp's buffer; its rows arrive one by one on the
  buffer's mbarrier (32 arrivals, padding rows zero-filled and arrived).  The warp alone quantizes it: the normal
  and transposed e4m3 rows and both scale layouts go to global memory by ``st.global``, with a
  ``fence.proxy.async.global`` before the batched ``red.release.gpu`` arrive (the GEMMs read them by TMA).
- A push block moves 32 local Y rows to the peers' combine buffers by one bulk store per lane.

Compute tasks.  The shared expert is BF16 with four task kinds (Gate, Up, SwiGLU, Down).  The routed experts
have two:

- the fused FC1, one task per 256 rows x 128 Hidden columns: CTA 0 loads the Gate weights and scales, CTA 1 the
  Up ones (the accumulator's subtiles 0-3 are Gate, 4-7 Up).  Its epilogue rounds both to BF16, writes the Gate /
  Up context, computes Hidden from those BF16 values and quantizes it per 32-column subtile in the normal and the
  transposed layout (``mxgemm.store_acc_fused_swiglu``);
- Down, on the quantized Hidden.

The routed GEMMs run on e4m3 operands with their own scale ring and the block-scaled mainloop.  Warps of a
compute CTA in a GEMM task: 0-3 epilogue, 4 MMA (CTA 0), 6 routed scale loads, 7 A/B loads; warp 5 of CTA 0
requests the next CLC item.

Readiness counters in global memory (``red.release.gpu`` arrivals, acquire-load polls):

- ``x_routed_ready[minibatch]``: one arrival per quantized stream block, ``ceil(rows / 32) x ceil(hidden / 256)``
  per minibatch; the routed FC1 waits for all of them.
- ``hidden_row_block_ready[256-row block]``: one arrival per CTA per fused FC1 task (per SwiGLU tile for the
  shared expert), ``2 x (I / 128)`` per block; Down waits for all of them.
- ``y_routed_ready[minibatch]``: the routed Down's arrivals.  A push waits for its own minibatch's; a pull of a
  macrobatch before the last waits for the Down of the macrobatch after it on the same ``x_routed`` rows (the
  rows are reused).
- ``y_routed_done[128-row block]``: one arrival per push block (four per 128-row block and column block).  The
  next macrobatch's routed Down waits for all of them before it overwrites ``y_routed``.
- ``gate_up_tile_ready``: the shared Gate / Up tiles, for the shared SwiGLU task.

Further invariants:

- the pipeline slots of the GEMM rings (A/B and scales) continue across tasks, and their transaction barriers
  are armed by the pipeline's producer (``producer_acquire``);
- each 512 B scale tile moves with a 1D bulk copy (its global tile is contiguous);
- named barriers: 1 the GEMM epilogue's, 2 the TMEM allocator's, 4 the D-store tail's (the table is in
  ``_persistent_mxfp8_fwd_gemm``);
- the SwiGLU multiply runs without flush-to-zero.

Gate / Up context: ``_PersistentMxfp8FwdMega`` keeps it in BF16 (``gate_up_context="bf16"``),
``_PersistentMxfp8FwdMegaE4m3Context`` as e4m3 with scale tiles (``"mxfp8"``, the default).

Runtime: token counts, both hidden sizes (tensor extents), the intermediate size, topk, the expert count, the
macrobatch / minibatch sizes, the communication SM count and the peer count (the peer tables' length).  Compile
time: the clamp, the task decode, the context precision and the variant flags (``split_weights``, ``is_situ``,
``has_shared``, ``shared_fc1_only``, ``has_tail``), which select code.

The ``shared_fc1_only`` variant launches the shared Gate, Up and SwiGLU tasks alone -- no communication cluster, no
shared Down and no routed minibatch in the grid -- for ``persistent_mxfp8_fwd.shared_fc1_mxfp8``; the task code is
the same.
"""

import enum
import inspect

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Float8E4M3FN, Float8E8M0FNU, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum

from ._mxfp8_bwd_math import glu_forward
from ._compat.pipeline import PipelineTmaUmma, PipelineUmmaAsync

from . import _mxfp8_gemm_base as gemm
from . import _persistent_mxfp8_fwd_gemm as mxgemm

from . import routed_descriptors as routed_descriptors_pkg
from . import _comm_stream
from ._comm_stream import (
    STREAM_BUFFERS,
    STREAM_COLUMNS,
    STREAM_GROUP_BLOCKS,
    STREAM_MBARRIERS,
    STREAM_PULL,
    STREAM_PUSH,
    STREAM_ROW_BYTES,
    STREAM_ROWS,
    STREAM_WARP_BYTES,
    STREAM_WARPS,
    _divide_u32,
    _red_release_add_gpu,
    _reciprocal_u32,
)
from ._fwd_device_helpers import _swizzled_mlp_tile_coord_i32
from ._arch import COMPILE_OPTIONS
from ._host_common import TAIL_WIDTH, row_major_view
from ._mxfp8_fwd_base import (
    CLUSTER_SHAPE,
    DISPATCH_ROW_CHUNK_BYTES,
    DYNAMIC_SHARED_MEMORY_BYTES,
    SWIGLU_TILE_BYTES,
    SWIGLU_TILE_M,
    SWIGLU_TILE_N,
    _ceil_div_i32,
    _counter_wait_gpu,
    _issue_swiglu_tma_load,
    _issue_swiglu_tma_store,
    _Mxfp8FwdBase,
    make_mega_args,
)
from ._mxfp8_gemm_base import _make_tma_block_load_fn
from ._mxfp8_quant import (
    BLOCK_ELEMENTS,
    E4M3_MAX_RECIPROCAL,
    QUANT_TILE,
    RECIPROCAL_BIAS,
    SCALE_FLOOR,
    _abs_max_xorsign_bf16x2,
    _amax_float,
    _ldmatrix_x4_trans,
    _load_shared_v4,
    _shuffle_xor,
    cvt_e8m0_scale_byte,
    quantize_tile,
)
from ._persistent_bf16_fwd_mega import (
    _build_expert_cu_end,
    _routed_descriptors_or_placeholder,
    _routed_expert_tile_i32,
    _routed_tables_or_placeholder,
)
from ._persistent_mxfp8_fwd_gemm import _Mxfp8Gemm
from ._tma_1d import tma_store_1d_raw
from .mxfp8_contract import (
    CLUSTER_SIZE,
    MLP_TILE_COLUMNS,
    MLP_TILE_ROWS,
    ROUTED_SWIGLU as REPLAY_SWIGLU_KIND,  # the kind the backward's replayed routed SwiGLU passes
    THREADS_PER_CTA,
)
from .persistent_bf16_fwd_contract import EXPERT_TABLE_CAP, TaskDecode
from .persistent_mxfp8_fwd import dynamic_layout_tensor

# Shared memory: the GEMM region -- the six-stage A/B ring, the scale ring and D, 230400 B
# (mxgemm.MXFP8_ARENA_BYTES); the communication CTAs use it as their arena.
QUANT_FP8_TILE_BYTES = QUANT_TILE * QUANT_TILE
QUANT_SF_TILE_BYTES = mxgemm.SF_TILE_BYTES
# The MXFP8 task table.  The shared group is Gate, Up, SwiGLU, Down; the routed group is
# the fused one: one FC1 task per 256 rows x 128 Hidden columns computes Gate and Up (CTA 0 the Gate
# weights, CTA 1 the Up weights of the same columns) and its epilogue writes the BF16 context and
# the SwiGLU-quantized Hidden, so the routed experts have no SwiGLU task and no BF16 Gate/Up ring.
SHARED_GATE, SHARED_UP, SHARED_SWIGLU, SHARED_DOWN = range(4)
ROUTED_GATE_UP, ROUTED_DOWN = range(4, 6)
FC1_TILE_COLUMNS = QUANT_TILE  # Hidden columns per fused FC1 task: one 128 x 128 quantization tile per CTA
STREAM_ROW_TILES_PER_SCALE_TILE = QUANT_TILE // STREAM_ROWS
# y_routed_done arrivals per 128-row block and column block: a push block covers STREAM_ROWS rows x STREAM_COLUMNS
# columns, so a 128-row x 256-column block sees Y_DONE_ROW_TILES of them (the routed Down waits for all).
assert STREAM_ROW_BYTES == DISPATCH_ROW_CHUNK_BYTES  # the dispatch row chunk is a stream buffer row pair
Y_DONE_ROW_TILES = MLP_TILE_ROWS // CLUSTER_SIZE // STREAM_ROWS
Y_DONE_COLUMNS = STREAM_COLUMNS
assert STREAM_BUFFERS * STREAM_ROWS * STREAM_COLUMNS * 2 == STREAM_WARP_BYTES
assert STREAM_WARPS * STREAM_WARP_BYTES <= mxgemm.MXFP8_ARENA_BYTES
assert STREAM_WARPS < THREADS_PER_CTA // 32 and STREAM_COLUMNS % QUANT_TILE == 0
assert (MLP_TILE_ROWS // CLUSTER_SIZE) % STREAM_ROWS == 0  # a 128-row block is whole push blocks
# The has_tail kernels' tail code is written for a tail of exactly TAIL_WIDTH: half a stream column block and
# whole quantization tiles.  Checked here, when the module is loaded, so another block geometry fails before
# any kernel is compiled; the host admits only H that are whole TAIL_WIDTH halves (_host_common.has_tail).
assert TAIL_WIDTH == 128 and STREAM_COLUMNS == 2 * TAIL_WIDTH and TAIL_WIDTH % QUANT_TILE == 0


@dsl_user_op
def _store_global_v4(address: Int64, words: list, *, loc=None, ip=None) -> None:
    """``st.global.v4.b32``: 16 B, a lane's e4m3 bytes of one row (the normal pass)."""
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)] + [Uint32(w).ir_value(loc=loc, ip=ip) for w in words],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_global_u8(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    """``st.global.u8``: one scale byte."""
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _e4m3x4_f32x2(a: Float32, b: Float32, c: Float32, d: Float32, reciprocal: Float32, *, loc=None, ip=None) -> Uint32:
    """``a`` .. ``d`` times ``reciprocal`` by two ``mul.rn.f32x2`` (no flush-to-zero, each lane rounded as
    ``mul.rn.f32``), ``cvt.rn.satfinite.e4m3x2.f32`` per pair, the four bytes packed ``a`` lowest."""
    return Uint32(
        llvm.inline_asm(
            cutlass.cutlass_dsl.T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d, reciprocal)],
            "{ .reg .b64 x, y, s, u, v; .reg .b32 u0, u1, v0, v1; .reg .b16 p, q; "
            "mov.b64 x, {$1, $2}; mov.b64 y, {$3, $4}; mov.b64 s, {$5, $5}; "
            "mul.rn.f32x2 u, x, s; mul.rn.f32x2 v, y, s; mov.b64 {u0, u1}, u; mov.b64 {v0, v1}, v; "
            "cvt.rn.satfinite.e4m3x2.f32 p, u1, u0; cvt.rn.satfinite.e4m3x2.f32 q, v1, v0; mov.b32 $0, {p, q}; }",
            "=r,f,f,f,f,f",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_global_v8(address: Int64, words: list, *, loc=None, ip=None) -> None:
    """``st.global.v8.b32``: 32 B, one column's 32 e4m3 bytes of the transposed layout (a whole sector)."""
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)] + [Uint32(w).ir_value(loc=loc, ip=ip) for w in words],
        "st.global.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};",
        "l,r,r,r,r,r,r,r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _encode_words(words: list, amax: Float32):
    """The block scale of ``amax`` and the e4m3 encoding of the BF16 pairs ``words`` (a multiple of two) under it:
    (scale byte, e4m3 words), the multiply packed by ``mul.rn.f32x2``."""
    scale = cute.arch.fmax(amax * Float32(E4M3_MAX_RECIPROCAL), Float32(SCALE_FLOOR))
    scale_byte = cvt_e8m0_scale_byte(scale)
    reciprocal = ((Int32(RECIPROCAL_BIAS) - scale_byte) << Int32(23)).bitcast(Float32)
    out = []
    for k in cutlass.range_constexpr(len(words) // 2):
        w0 = words[2 * k]
        w1 = words[2 * k + 1]
        out.append(
            _e4m3x4_f32x2(
                (w0 << Uint32(16)).bitcast(Float32),
                (w0 & Uint32(0xFFFF0000)).bitcast(Float32),
                (w1 << Uint32(16)).bitcast(Float32),
                (w1 & Uint32(0xFFFF0000)).bitcast(Float32),
                reciprocal,
            )
        )
    return scale_byte, out


@cute.jit
def _quantize_transposed_warp(
    source: cute.Pointer,
    tid: Int32,
    row_stride_bytes: cutlass.Constexpr[int],
    out_t_base: Int64,
    column0: Int32,
    out_t_stride: Int64,
    scale_t_base: Int64,
    unit: Int32,
) -> None:
    """The transposed pass of a 32-row x 128-column BF16 subtile in shared memory by one warp, one half (``tid``
    0-31: columns 0-63, 32-63: columns 64-127) per call: ``ldmatrix.trans`` gathers each lane two columns' 32 rows,
    the column is one scale block, and the 32 e4m3 bytes go to global memory by one 32 B store at ``out_t_base +
    (column0 + column) x out_t_stride`` (``column0``: the subtile's first column in the tensor); the scale byte of
    column ``c`` goes to ``scale_t_base + unit + 16 (c % 32) + 4 (c / 32)`` (the 32-row tile ``unit``'s byte of the
    column's scale word).

    Both addresses add the lane's column term to a term of the pulled block (``column0``, ``unit``) in 32 bits and
    widen the sum.  Written as base + Int64(lane term), the lane term is the same for every block a warp
    quantizes: CuTe DSL 4.8 then computes the four columns' eight 64-bit offsets once in front of the stream loop
    and keeps them in registers across it (4.5.2 forms them at the store), and with those registers taken the
    pass's dependent instructions are issued back to back.  Keep the lane term inside the 32-bit sum."""
    half = tid >> Int32(5)
    lane = tid & Int32(31)
    q = lane & Int32(3)
    column_a = half * Int32(64) + (q << Int32(4)) + (lane >> Int32(2))
    address = source.toint() + (lane >> Int32(3)) * Int32(row_stride_bytes) + half * Int32(128) + (lane & Int32(7)) * Int32(16)
    registers = []
    for x in cutlass.range_constexpr(BLOCK_ELEMENTS // 4):
        registers.extend(_ldmatrix_x4_trans(address + Int32(4 * x * row_stride_bytes)))
    amax_pair = registers[0]
    for k in cutlass.range_constexpr(1, len(registers)):
        amax_pair = _abs_max_xorsign_bf16x2(amax_pair, registers[k])
    amax_pair = amax_pair & Uint32(0x7FFF7FFF)
    for h in cutlass.range_constexpr(2):
        if cutlass.const_expr(h == 0):
            amax = (amax_pair << Uint32(16)).bitcast(Float32)
            values = [(r << Uint32(16)).bitcast(Float32) for r in registers]
        else:
            amax = (amax_pair & Uint32(0xFFFF0000)).bitcast(Float32)
            values = [(r & Uint32(0xFFFF0000)).bitcast(Float32) for r in registers]
        scale = cute.arch.fmax(amax * Float32(E4M3_MAX_RECIPROCAL), Float32(SCALE_FLOOR))
        scale_byte = cvt_e8m0_scale_byte(scale)
        reciprocal = ((Int32(RECIPROCAL_BIAS) - scale_byte) << Int32(23)).bitcast(Float32)
        w = [_e4m3x4_f32x2(values[4 * k], values[4 * k + 1], values[4 * k + 2], values[4 * k + 3], reciprocal) for k in range(BLOCK_ELEMENTS // 4)]
        column = column_a + Int32(8 * h)
        _store_global_u8(scale_t_base + Int64(unit + (column & Int32(31)) * Int32(16) + (column >> Int32(5)) * Int32(4)), scale_byte)
        _store_global_v8(out_t_base + Int64(column0 + column) * out_t_stride, w)


@cute.jit
def _normal_row_store_offset(row: Int32, hb: Int32, column0: Int32, row_units: Uint32) -> Int64:
    """Byte offset of row ``row`` (0-31 of a 32-row tile), half block ``hb`` of the subtile at column ``column0``,
    from the tile's first row: ``row x hidden + column0 + 16 hb`` for rows of ``row_units`` x 16 B (``hidden / 16``).
    The sum is formed in 16-byte units in 32 bits and widened; it is below 32 x hidden / 16 < 2^32 for every hidden
    the plan accepts (below 2^31), unsigned so it does not wrap."""
    return Int64(Uint32(row) * row_units + Uint32((column0 >> Int32(4)) + hb)) << Int64(4)


@cute.jit
def _quantize_normal_warp(
    source: cute.Pointer,
    lane: Int32,
    row_stride_bytes: cutlass.Constexpr[int],
    out_row0: Int64,
    out_row_stride: Int64,
    scale0: Int64,
    unit: Int32,
    column0: Int32,
    row_units: Uint32,
    lane_terms_in_32bit: cutlass.Constexpr[bool],
) -> None:
    """The normal pass of a 32-row x 128-column BF16 subtile in shared memory by one warp, half blocks per lane.
    Lane ``l`` takes half block ``hb = l % 8`` (columns ``16 hb`` .. ``16 hb + 15``, block ``hb / 2``) of rows
    ``l / 8 + 4 i`` (i = 0..7).  Its two 16 B chunks are loaded in the order ``(t ^ s)`` with ``s = bit 2 of l``: in
    an 8-lane phase (one row, ``hb`` 0-7) the 16 B bank groups are ``2 hb + (t ^ s) mod 8``, all different, at any
    row stride.  The block amax pairs lanes ``l`` and ``l ^ 1`` (one butterfly); the 16 e4m3 bytes go out by one
    16 B store (the eight lanes of a row write its 128 B), the scale byte by the even lane, at ``scale0 + 16 (s %
    32) + 4 (s / 32) + block`` with ``s = 32 unit + row`` (``unit``: the 32-row tile within the 128-row scale
    tile).

    The e4m3 store address: with ``lane_terms_in_32bit``, ``out_row0`` is the 32-row tile's first row and the lane's
    row and half-block terms are added to the subtile's column ``column0`` in 32 bits (``_normal_row_store_offset``,
    16-byte units, ``row_units`` = hidden / 16) before widening; otherwise ``out_row0`` also holds the subtile's
    column and the address is ``out_row0 + row x out_row_stride + 16 hb`` (``column0`` / ``row_units`` unused).
    Written the second way, the lane's eight 64-bit row offsets are the same for every block a warp quantizes, and
    CuTe DSL 4.8 computes them once at kernel scope and keeps 16 registers across the stream loop (4.5.2 forms the
    eight 64-bit addresses at the top of every block and keeps them until the stores)."""
    hb = lane & Int32(7)
    swap = (lane >> Int32(2)) & Int32(1)
    block = hb >> Int32(1)
    sel = Uint32(0) - Uint32(swap)
    items = 8
    rows = [(lane >> Int32(3)) + Int32(4 * i) for i in range(items)]
    first = [_load_shared_v4(source.toint() + rows[i] * Int32(row_stride_bytes) + ((hb << Int32(1)) + swap) * Int32(16)) for i in range(items)]
    second = [_load_shared_v4(source.toint() + rows[i] * Int32(row_stride_bytes) + ((hb << Int32(1)) + (swap ^ Int32(1))) * Int32(16)) for i in range(items)]
    amaxes = []
    for i in cutlass.range_constexpr(items):
        a = first[i]
        b = second[i]
        m = _abs_max_xorsign_bf16x2(_abs_max_xorsign_bf16x2(_abs_max_xorsign_bf16x2(a[0], a[1]), a[2]), a[3])
        m = _abs_max_xorsign_bf16x2(_abs_max_xorsign_bf16x2(_abs_max_xorsign_bf16x2(_abs_max_xorsign_bf16x2(m, b[0]), b[1]), b[2]), b[3])
        amaxes.append(m)
    partners = [_shuffle_xor(value, 1) for value in amaxes]
    for i in cutlass.range_constexpr(items):
        amax = _amax_float(_abs_max_xorsign_bf16x2(amaxes[i], partners[i]))
        scale_byte, e = _encode_words(first[i] + second[i], amax)
        out_words = [e[k] ^ ((e[k] ^ e[k ^ 2]) & sel) for k in range(4)]
        if cutlass.const_expr(lane_terms_in_32bit):
            _store_global_v4(out_row0 + _normal_row_store_offset(rows[i], hb, column0, row_units), out_words)
        else:
            _store_global_v4(out_row0 + Int64(rows[i]) * out_row_stride + Int64(hb * Int32(16)), out_words)
        if (hb & Int32(1)) == Int32(0):
            scale_row = (unit << Int32(5)) + rows[i]
            _store_global_u8(scale0 + Int64((scale_row & Int32(31)) * Int32(16) + (scale_row >> Int32(5)) * Int32(4) + block), scale_byte)


@cute.jit
def _fc1_half_offset(column: Int32, cta_rank: Int32) -> Int32:
    """Row offset of the view through which CTA ``cta_rank`` reaches rows ``[128 c, 128 c + 128)`` of
    its own Gate / Up operand as half ``cta_rank`` of the 256-row tile ``c // 2``:
    ``256 (c // 2) + 128 r + offset = 128 c``."""
    return Int32(FC1_TILE_COLUMNS) * ((column & Int32(1)) - cta_rank)


@cute.jit
def _stream_task(index: Int32, last_macro: Int32, last_tiles: Int32, full_tiles: Int32, reciprocal_2f: Int32) -> tuple[Int32, Int32, Int32]:
    """Stream task ``index`` -> (queue, macrobatch, 32-row tile).  Tasks ``0 .. last_tiles - 1`` are the last
    macrobatch's dispatch pulls (tile = index).  Then one segment per macrobatch ``m`` from the last down to 0
    holds ``m``'s combine pushes interleaved with macrobatch ``m - 1``'s dispatch pulls: entry ``j`` of the segment
    is the push of tile ``j / 2`` (even ``j``) or the pull of tile ``j / 2`` of ``m - 1`` (odd ``j``) while ``j < 2 x
    tiles(m)``, then the remaining pulls of ``m - 1`` (the tiles beyond ``tiles(m)``; only the last macrobatch can
    be shorter than ``full_tiles``); macrobatch 0's segment is its pushes alone.  So a macrobatch's pulls run while
    the macrobatch after it is pushed, and its pushes while the one before it is pulled.  No per-task division:
    the segment of ``index`` is ``j2 / (2 full_tiles)`` by ``_divide_u32``."""
    queue = Int32(STREAM_PULL)
    macro = last_macro
    tile = index
    if index >= last_tiles:
        j = index - last_tiles
        m = last_macro
        count = last_tiles
        if j >= last_tiles + full_tiles:
            j2 = j - last_tiles - full_tiles
            block = _divide_u32(j2, Int32(2) * full_tiles, reciprocal_2f)
            m = last_macro - Int32(1) - block
            count = full_tiles
            j = j2 - block * Int32(2) * full_tiles
        queue = Int32(STREAM_PUSH)
        macro = m
        tile = j
        if m > Int32(0):
            pull_row = j - count
            is_push = Int32(0)
            if j < Int32(2) * count:
                pull_row = j >> Int32(1)
                if (j & Int32(1)) == Int32(0):
                    is_push = Int32(1)
            if is_push == Int32(1):
                tile = j >> Int32(1)
            else:
                queue = Int32(STREAM_PULL)
                macro = m - Int32(1)
                tile = pull_row
    return queue, macro, tile


@cute.jit
def _counter_arrive_release_gpu(counter: cute.Tensor, index: Int32) -> None:
    """Arrive on a readiness counter: ``red.release.gpu`` adds one and returns nothing (a reduction, with a
    ``memory`` clobber).  Not ``_fwd_device_helpers._counter_arrive_gpu``, whose ``cute.arch.atomic_add``
    returns, and so waits for, the old value."""
    _red_release_add_gpu((counter.iterator + index).toint(), Int32(1))


@enum.unique
class _Fs(enum.IntEnum):
    """Positions in the forward stream hooks' operand list (``_forward_stream`` builds it, ``_run_stream``
    hands it to every hook)."""

    X_ROUTED_READY = 0
    Y_ROUTED_READY = 1
    Y_ROUTED_DONE = 2
    Y_ROUTED = 3
    X_PEER_TABLE = 4
    COMBINE_PEER_TABLE = 5
    X_ROUTED_BASE = 6
    X_T_BASE = 7
    X_SC_BASE = 8
    X_SC_T_BASE = 9
    LAST_MACRO = 10
    LAST_TILES = 11
    FULL_TILES = 12
    NUM_TOKENS = 13
    HIDDEN_SIZE = 14
    TOPK = 15
    MACROBATCH_SIZE = 16
    MINIBATCH_SIZE = 17
    RECIPROCAL_2F = 18
    RECIPROCAL_MINI = 19
    RECIPROCAL_TOPK = 20
    COUNT = 21


class _Mx(enum.IntEnum):
    """Positions in the kernel's ``mxfp8_atoms`` / ``mxfp8_tensors`` lists.

    Atoms and their TMA tensors share positions 0-8 (scale loads, then the e4m3 tile stores of
    Hidden and the BF16 Gate / Up context tile stores).  ``mxfp8_tensors`` goes on with the plain
    row-major ``x_routed_t`` the stream's quantization addresses directly (no atom: the stream
    stores x by ``st.global``) and the scale outputs, written with 512 B bulk copies.
    """

    SFA_X = 0
    SFA_HIDDEN = 1
    SFB_GATE = 2
    SFB_UP = 3
    SFB_DOWN = 4
    HIDDEN_FP8 = 5
    HIDDEN_T = 6
    GATE_BF16 = 7
    UP_BF16 = 8
    X_T_ROWS = 9
    X_SC = 10
    X_SC_T = 11
    HIDDEN_SC = 12
    HIDDEN_SC_T = 13


class _MxE4m3Context(enum.IntEnum):
    """``gate_up_context="mxfp8"`` (the e4m3 / scale context): positions in the kernel's
    ``mxfp8_atoms`` / ``mxfp8_tensors`` lists.

    Atoms and their TMA tensors share positions 0-8 (scale loads, then the e4m3 tile stores of
    Gate, Up and Hidden).  ``mxfp8_tensors`` goes on with the plain row-major ``x_routed_t`` the
    stream's quantization addresses directly (no atom) and the scale outputs, written with 512 B
    bulk copies.
    """

    SFA_X = 0
    SFA_HIDDEN = 1
    SFB_GATE = 2
    SFB_UP = 3
    SFB_DOWN = 4
    GATE_FP8 = 5
    UP_FP8 = 6
    HIDDEN_FP8 = 7
    HIDDEN_T = 8
    X_T_ROWS = 9
    X_SC = 10
    X_SC_T = 11
    GATE_SC = 12
    UP_SC = 13
    HIDDEN_SC = 14
    HIDDEN_SC_T = 15


# The positions the shared forward code indexes (scale loads, the plain x_routed_t and the x scales) are the same
# in both context variants; only the context's own entries differ.
assert all(_Mx[name] == _MxE4m3Context[name] for name in ("SFA_X", "SFA_HIDDEN", "SFB_GATE", "SFB_UP", "SFB_DOWN", "X_T_ROWS", "X_SC", "X_SC_T"))


@cute.struct
class _SharedStorageMxfp8:
    clc_mbarriers: cute.struct.MemRange[cutlass.Int64, 2]
    clc_response: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
    clc_drain_mbarriers: cute.struct.MemRange[cutlass.Int64, 16]
    clc_drain_responses: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 32], 16]
    ab_full: cute.struct.MemRange[cutlass.Int64, 6]
    ab_empty: cute.struct.MemRange[cutlass.Int64, 6]
    sf_full: cute.struct.MemRange[cutlass.Int64, 6]  # the scale ring's barriers
    sf_empty: cute.struct.MemRange[cutlass.Int64, 6]
    acc_full: cutlass.Int64
    acc_empty: cutlass.Int64
    swiglu_barriers: cute.struct.MemRange[cutlass.Int64, 3]
    stream_mbarriers: cute.struct.MemRange[cutlass.Int64, STREAM_MBARRIERS]  # one per stream buffer
    tmem_dealloc: cutlass.Int64
    tmem_address: cutlass.Int32
    # The per-CTA expert row-block table (TaskDecode SMEM_BINARY / SMEM_GROUP).
    expert_cu_end: cute.struct.MemRange[cutlass.Int32, EXPERT_TABLE_CAP]
    arena: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, mxgemm.MXFP8_ARENA_BYTES], 1024]


assert _SharedStorageMxfp8.size_in_bytes() <= DYNAMIC_SHARED_MEMORY_BYTES


class _PersistentMxfp8FwdMega(_Mxfp8FwdBase):
    """The MXFP8 forward megakernel with the BF16 Gate / Up context (see the module docstring).

    ``x_routed`` / ``hidden_routed`` and the routed weights are e4m3; scales are uint8
    ``(rows / 128, cols / 128, 32, 16)`` tiles.  ``self.collective`` (BF16) serves the shared expert,
    ``self.mxfp8_collective`` the routed tasks.  ``is_clamped``, ``task_decode`` and the five variant flags are
    compile time: each combination is its own compiled kernel.
    """

    # How ``_stream_quantize`` forms the store addresses of both passes (compile time, per kernel class).  True: the
    # lane's terms are added to a term of the pulled block in 32 bits and the sum widened (see
    # ``_quantize_transposed_warp`` and ``_quantize_normal_warp``); the forward's transposed pass is faster with it
    # under CuTe DSL 4.5.2 and 4.8.  False: the block's terms go into the 64-bit bases and the lane's terms are widened
    # on their own; the backward megakernel, which runs the same stream code, is faster with that and sets False.
    stream_lane_terms_in_32bit = True

    def __init__(
        self,
        is_clamped: bool,
        task_decode: TaskDecode,
        split_weights: bool = False,
        is_situ: bool = False,
        has_shared: bool = True,
        shared_fc1_only: bool = False,
        has_tail: bool = False,
    ):
        super().__init__(is_clamped)
        # False for a routed-only layer, whose shared tensors are placeholders the kernel builds
        # atoms over and never touches -- there are no shared tasks.
        self.has_shared = bool(has_shared)
        # The launch holds the shared Gate, Up and SwiGLU tasks alone (gate_shared, up_shared, hidden_shared): no
        # communication cluster, no shared Down and no routed minibatch.  It recomputes the shared activations of a
        # forward whose context was not kept (persistent_mxfp8_fwd.shared_fc1_mxfp8); the task code is the
        # forward's, so are the results.  Every routed tensor, the Down weights and y_shared are placeholders the
        # kernel builds atoms over and no task touches; the schedule is one expert without rows.
        self.shared_fc1_only = bool(shared_fc1_only)
        if self.shared_fc1_only and not self.has_shared:
            raise ValueError("shared_fc1_only needs a shared expert")
        # SiTU-GLU instead of SwiGLU for the shared and routed MLPs, with the two betas as runtime
        # arguments (situ_beta1 / situ_beta2).  Never with a clamp.
        self.is_situ = bool(is_situ)
        if self.is_situ and self.is_clamped:
            raise ValueError("SiTU cannot be combined with swiglu_limit")
        # Task -> expert decode, a compile-time attribute as in the BF16 forward (contract TaskDecode).
        self.task_decode = TaskDecode(task_decode)
        # The two hiddens are runtime tensor extents, as in the BF16 forward: the shared expert works in
        # x_shared's width, the routed chain -- the dispatch rows, x_routed and its quantized layouts, the routed
        # GEMMs' K / N, the combine rows and y_routed -- in x_routed's (equal for an ordinary MoE, different for
        # the latent MoE of functional.forward_latent_layer).  One compiled kernel serves both.
        self.mxfp8_collective = _Mxfp8Gemm()
        # Split routed weights: six operands here rather than the BF16 forward's three, because an
        # MXFP8 weight is a payload *and* a scale and both are that expert's own allocation.  The
        # table's rows are Gate, Up, Down payloads then Gate, Up, Down scales, and so are the
        # descriptor arrays.  Compile-time, because it changes the operands' rank.
        self.split_weights = bool(split_weights)
        # H or I has a 128-wide tail (size % 256 != 0): one more 256-wide tile per row block.
        # Compile time because H and I are model parameters and the flag only selects whether the tail code is
        # generated: an aligned model runs loops without predicates or tail branches.  It doubles this kernel's
        # executors (one for every aligned shape, one for every shape with a tail) as one more entry of the
        # executor key, which a model keeps for its whole run.  The values of H and I stay runtime arguments:
        # nothing is specialised on a size.
        self.has_tail = bool(has_tail)

    @cute.jit
    def _stream_pull(
        self,
        buffer: cute.Pointer,
        mbarrier: cute.Pointer,
        lane: Int32,
        column_tile: Int32,
        peer_rank: Int32,
        source_row: Int32,
        hidden_size: Int32,
        peer_buffer: Int64,
    ) -> None:
        """One row of a pull block (``_comm_stream.pull_row`` with this kernel's ``has_tail``): row ``source_row`` of
        the peer's buffer at ``peer_buffer``, or zeros for a padding row.  The backward's stream pulls dY and the
        replayed x through this as well."""
        _comm_stream.pull_row(buffer, mbarrier, lane, column_tile, peer_rank, source_row, hidden_size, peer_buffer, self.has_tail)

    @cute.jit
    def _stream_quantize(
        self,
        buffer: cute.Pointer,
        lane: Int32,
        row_unit: Int32,
        column_tile: Int32,
        hidden_size: Int32,
        macrobatch_size: Int32,
        bases: list,
        tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Quantize the STREAM_ROWS x STREAM_COLUMNS BF16 block in ``buffer`` (rows STREAM_COLUMNS x 2 B apart), 32-row
        tile ``row_unit`` of the macrobatch and column block ``column_tile``, by this warp alone: per 128-column
        subtile the transposed pass (two halves) and the normal pass, each storing its e4m3 bytes and scale bytes
        straight to the global tensors whose addresses are ``bases`` (normal e4m3, transposed e4m3, normal scales,
        transposed scales); no shared-memory staging, no block barrier.

        ``tail`` (compile time): the instantiation for the last column block of a hidden with a 128-wide tail,
        which holds TAIL_WIDTH columns: a static count of subtiles, as a full block's."""
        x_routed_base, x_t_base, x_sc_base, x_sc_t_base = bases
        scale_columns = hidden_size // Int32(QUANT_TILE)
        scale_rows_t = macrobatch_size // Int32(QUANT_TILE)
        row_tile = row_unit // Int32(STREAM_ROW_TILES_PER_SCALE_TILE)
        unit = row_unit - row_tile * Int32(STREAM_ROW_TILES_PER_SCALE_TILE)
        subtiles = STREAM_COLUMNS // QUANT_TILE
        if cutlass.const_expr(tail):
            # A 128-wide H tail: the last column block holds TAIL_WIDTH columns, one quantization tile; the
            # second one's outputs would lie past the row (normal layout) and past the buffers (transposed
            # layout, scales).
            subtiles = TAIL_WIDTH // QUANT_TILE
        for subtile in cutlass.range(subtiles, unroll=1):
            column_block = column_tile * Int32(STREAM_COLUMNS // QUANT_TILE) + subtile
            source = buffer + subtile * Int32(QUANT_TILE * 2)
            scale_t_base = x_sc_t_base + (Int64(column_block) * Int64(scale_rows_t) + Int64(row_tile)) * Int64(QUANT_SF_TILE_BYTES)
            if cutlass.const_expr(self.stream_lane_terms_in_32bit):
                out_t_base = x_t_base + Int64(row_unit * Int32(STREAM_ROWS))
                column0 = column_block * Int32(QUANT_TILE)
                unit_t = unit
                out_row0 = x_routed_base + Int64(row_unit * Int32(STREAM_ROWS)) * Int64(hidden_size)
                row_units = Uint32(hidden_size >> Int32(4))
            else:  # the block's terms in the 64-bit bases; the constant zeros leave the lane's terms on their own
                out_t_base = x_t_base + Int64(column_block * Int32(QUANT_TILE)) * Int64(macrobatch_size) + Int64(row_unit * Int32(STREAM_ROWS))
                scale_t_base = scale_t_base + Int64(unit)
                column0 = Int32(0)
                unit_t = Int32(0)
                out_row0 = x_routed_base + Int64(row_unit * Int32(STREAM_ROWS)) * Int64(hidden_size) + Int64(column_block * Int32(QUANT_TILE))
                row_units = Uint32(0)
            scale0 = x_sc_base + (Int64(row_tile) * Int64(scale_columns) + Int64(column_block)) * Int64(QUANT_SF_TILE_BYTES)
            for h in cutlass.range_constexpr(2):
                _quantize_transposed_warp(source, Int32(32 * h) + lane, STREAM_COLUMNS * 2, out_t_base, column0, Int64(macrobatch_size), scale_t_base, unit_t)
            _quantize_normal_warp(
                source, lane, STREAM_COLUMNS * 2, out_row0, Int64(hidden_size), scale0, unit, column0, row_units, self.stream_lane_terms_in_32bit
            )

    @cute.jit
    def _stream_decode(self, operands: list, task: Int32) -> tuple[Int32, Int32, Int32]:
        """``_run_stream``'s decode hook: stream task -> (queue, macrobatch, 32-row tile), ``_stream_task``."""
        return _stream_task(task, operands[_Fs.LAST_MACRO], operands[_Fs.LAST_TILES], operands[_Fs.FULL_TILES], operands[_Fs.RECIPROCAL_2F])

    @cute.jit
    def _stream_schedule_row(self, operands: list, macro: Int32, row: Int32) -> Int32:
        """``_run_stream``'s schedule-row hook: the schedule index of row ``row`` of macrobatch ``macro``."""
        return macro * Int32(operands[_Fs.MACROBATCH_SIZE]) + row

    @cute.jit
    def _stream_issue(
        self, operands: list, queue: Int32, row: Int32, column_tile: Int32, buffer: cute.Pointer, mbarrier: cute.Pointer, lane: Int32, peer: Int32, route: Int32
    ) -> None:
        """``_run_stream``'s issue hook: a block into ``buffer`` on ``mbarrier`` with the lane's schedule entries
        (``peer``, ``route`` of the block's row ``lane``) already read: a pull of the peer's x row
        ``route / topk`` (``_stream_pull``), or a push's read of the local ``y_routed`` row ``row + lane`` by one
        bulk copy per valid lane (a padding lane arrives with no bytes)."""
        hidden_size = operands[_Fs.HIDDEN_SIZE]
        if queue == Int32(STREAM_PULL):
            x_peer_table = operands[_Fs.X_PEER_TABLE]
            peer_buffer = Int64(0)
            if peer >= Int32(0):
                peer_buffer = Int64(x_peer_table[peer])
            # A padding lane's route is -1, outside _divide_u32's domain; its quotient is unused (_stream_pull
            # zero-fills on peer < 0).
            self._stream_pull(
                buffer, mbarrier, lane, column_tile, peer, _divide_u32(route, operands[_Fs.TOPK], operands[_Fs.RECIPROCAL_TOPK]), hidden_size, peer_buffer
            )
        else:
            _comm_stream.load_local_row(buffer, mbarrier, lane, operands[_Fs.Y_ROUTED].iterator.toint(), row, column_tile, peer, hidden_size, self.has_tail)

    @cute.jit
    def _stream_ready_lane0(self, operands: list, queue: Int32, macro: Int32, row: Int32) -> Int32:
        """``_run_stream``'s readiness hook, lane 0's answer whether the block waits for nothing more (the other
        lanes return 1): a pull of a macrobatch before the last waits for the routed Down of the macrobatch after
        it on the same ``x_routed`` rows (``y_routed_ready`` of that minibatch, the rows are reused), a push for
        the routed Down of its own minibatch.  The minibatch divisions use the reciprocal; the caller broadcasts
        the answer by shuffle when it needs it."""
        ready = Int32(1)
        if cute.arch.lane_idx() == Int32(0):
            y_routed_ready = operands[_Fs.Y_ROUTED_READY]
            last_macro = operands[_Fs.LAST_MACRO]
            num_tokens = operands[_Fs.NUM_TOKENS]
            hidden_size = operands[_Fs.HIDDEN_SIZE]
            macrobatch_size = operands[_Fs.MACROBATCH_SIZE]
            minibatch_size = operands[_Fs.MINIBATCH_SIZE]
            minibatch_reciprocal = operands[_Fs.RECIPROCAL_MINI]
            row_start = _divide_u32(row, minibatch_size, minibatch_reciprocal) * minibatch_size
            target = Int32(-1)
            if queue == Int32(STREAM_PULL):
                if macro < last_macro:
                    previous = macro + Int32(1)
                    previous_rows = cutlass.min(Int32(macrobatch_size), num_tokens - previous * Int32(macrobatch_size))
                    if row_start < previous_rows:
                        target = _divide_u32(previous * Int32(macrobatch_size) + row_start, minibatch_size, minibatch_reciprocal)
            else:
                target = _divide_u32(macro * Int32(macrobatch_size) + row_start, minibatch_size, minibatch_reciprocal)
            if target >= Int32(0):
                rows = cutlass.min(Int32(minibatch_size), num_tokens - target * minibatch_size)
                if cutlass.const_expr(self.has_tail):
                    # A 128-wide H tail is one more Down tile: the push must wait for it too.
                    required = _ceil_div_i32(rows, Int32(MLP_TILE_ROWS)) * _ceil_div_i32(hidden_size, Int32(MLP_TILE_COLUMNS)) * Int32(CLUSTER_SIZE)
                else:
                    required = _ceil_div_i32(rows, Int32(MLP_TILE_ROWS)) * Int32(hidden_size // MLP_TILE_COLUMNS) * Int32(CLUSTER_SIZE)
                if cute.arch.load(y_routed_ready.iterator + target, Int32, sem="acquire", scope="gpu") < required:
                    ready = Int32(0)
        return ready

    @cute.jit
    def _stream_arrival(self, operands: list, queue: Int32, macro: Int32, row: Int32) -> Int64:
        """``_run_stream``'s arrival-word hook: the counter a processed block arrives on -- ``x_routed_ready`` of
        the pull's minibatch, ``y_routed_done`` of the push's 128-row block for a macrobatch other than 0 (the next
        macrobatch's Down waits for all of a row block's arrivals before it overwrites ``y_routed``), 0 for a push
        of macrobatch 0."""
        address = Int64(0)
        if queue == Int32(STREAM_PULL):
            address = (
                operands[_Fs.X_ROUTED_READY].iterator
                + _divide_u32(self._stream_schedule_row(operands, macro, row), operands[_Fs.MINIBATCH_SIZE], operands[_Fs.RECIPROCAL_MINI])
            ).toint()
        else:
            if macro > Int32(0):
                address = (
                    operands[_Fs.Y_ROUTED_DONE].iterator + self._stream_schedule_row(operands, macro, row) // Int32(MLP_TILE_ROWS // CLUSTER_SIZE)
                ).toint()
        return address

    @cute.jit
    def _stream_drain(self, operands: list, queue: Int32) -> None:
        """``_run_stream``'s drain hook, before pending arrivals are released: nothing to complete (a pull block's
        quantized output is plain stores, fenced by the loop; a push block's buffer was read before it was counted)."""
        pass

    @cute.jit
    def _stream_process(
        self,
        operands: list,
        queue: Int32,
        macro: Int32,
        row: Int32,
        column: Int32,
        buffer: cute.Pointer,
        lane: Int32,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
    ) -> None:
        """``_run_stream``'s process hook, the block's bytes landed in ``buffer``: a pull block is quantized by the
        warp (``_stream_quantize``, direct global stores); a push block's rows go to the peers' combine buffers by
        one bulk store per valid lane, the buffer free after ``wait_group read``."""
        hidden_size = operands[_Fs.HIDDEN_SIZE]
        macrobatch_size = operands[_Fs.MACROBATCH_SIZE]
        if queue == Int32(STREAM_PULL):
            if cutlass.const_expr(self.has_tail):
                # One branch per block, the same for the whole warp: the last column block of a hidden with a
                # 128-wide tail quantizes its TAIL_WIDTH columns, every full block all STREAM_COLUMNS.
                if (column + Int32(1)) * Int32(STREAM_COLUMNS) > hidden_size:
                    self._stream_quantize(
                        buffer,
                        lane,
                        row // Int32(STREAM_ROWS),
                        column,
                        hidden_size,
                        macrobatch_size,
                        [operands[_Fs.X_ROUTED_BASE], operands[_Fs.X_T_BASE], operands[_Fs.X_SC_BASE], operands[_Fs.X_SC_T_BASE]],
                        tail=True,
                    )
                else:
                    self._stream_quantize(
                        buffer,
                        lane,
                        row // Int32(STREAM_ROWS),
                        column,
                        hidden_size,
                        macrobatch_size,
                        [operands[_Fs.X_ROUTED_BASE], operands[_Fs.X_T_BASE], operands[_Fs.X_SC_BASE], operands[_Fs.X_SC_T_BASE]],
                    )
            else:
                self._stream_quantize(
                    buffer,
                    lane,
                    row // Int32(STREAM_ROWS),
                    column,
                    hidden_size,
                    macrobatch_size,
                    [operands[_Fs.X_ROUTED_BASE], operands[_Fs.X_T_BASE], operands[_Fs.X_SC_BASE], operands[_Fs.X_SC_T_BASE]],
                )
        else:
            schedule_row = self._stream_schedule_row(operands, macro, row) + lane
            peer = schedule_peer_rank[schedule_row]
            if peer >= Int32(0):
                route = schedule_peer_token_idx[schedule_row]
                _comm_stream.store_row(
                    buffer,
                    lane,
                    Int64(operands[_Fs.COMBINE_PEER_TABLE][peer]) + Int64(route) * Int64(hidden_size) * Int64(2),
                    column,
                    hidden_size,
                    self.has_tail,
                )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    # The task stream's loop (``_comm_stream.run_stream``), with this class's hooks.
    _run_stream = _comm_stream.run_stream
    _stream_block = _comm_stream.block_shape(STREAM_ROWS, STREAM_COLUMNS, 2)  # the quantization's 32 x 256 blocks

    @cute.jit
    def _forward_stream(
        self,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        y_routed: cute.Tensor,
        arena: cute.Pointer,
        stream_mbarriers: cute.Pointer,
        num_tokens: Int32,
        hidden_size: Int32,
        topk: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        comm_cta: Int32,
        num_comm_sms: Int32,
        x_peer_table: cute.Tensor,
        combine_peer_table: cute.Tensor,
        quant_outputs: list,
    ) -> None:
        """The forward's communication as one dynamic task stream over every macrobatch (``_run_stream`` with
        the forward's hooks): the stream tasks are ``_stream_task``'s (a dispatch pull or a combine push of one
        32-row tile of one macrobatch), the head word is the extra last element of ``x_routed_ready``.  A pull
        block is quantized by the warp and arrives on ``x_routed_ready[minibatch]``; a push block's rows go to the
        peers' combine buffers and, for a macrobatch other than macrobatch 0, arrive on ``y_routed_done`` of the
        128-row block."""
        x_routed, x_t, x_sc, x_sc_t = quant_outputs
        last_macro = _ceil_div_i32(num_tokens, Int32(macrobatch_size)) - Int32(1)
        # 32-row tiles: num_tokens and the macrobatch are multiples of STREAM_ROWS (the schedule pads every expert to
        # 256 rows, the minibatch is a multiple of 256), so the floor here and the consumers' ceil agree.
        last_tiles = cutlass.min(Int32(macrobatch_size), num_tokens - last_macro * Int32(macrobatch_size)) // Int32(STREAM_ROWS)
        full_tiles = macrobatch_size // Int32(STREAM_ROWS)
        column_blocks = _ceil_div_i32(hidden_size, Int32(STREAM_COLUMNS))
        group_blocks = Int32(STREAM_GROUP_BLOCKS)
        groups = _ceil_div_i32(column_blocks, group_blocks)  # the stream is numbered per group of column blocks
        blocks = Int32(2) * (last_tiles + last_macro * full_tiles) * groups
        head = cute.make_ptr(Int32, (x_routed_ready.iterator + (x_routed_ready.shape[0] - Int32(1))).toint(), cute.AddressSpace.gmem, assumed_align=4)
        # The per-block divisors' reciprocals (one division each, here), no division per block.
        reciprocal_groups = _reciprocal_u32(groups)
        reciprocal_group = _reciprocal_u32(group_blocks)
        reciprocal_2f = _reciprocal_u32(Int32(2) * full_tiles)
        reciprocal_mini = _reciprocal_u32(minibatch_size)
        reciprocal_topk = _reciprocal_u32(topk)
        operands = [None] * _Fs.COUNT  # the hooks' operands, by position
        operands[_Fs.X_ROUTED_READY] = x_routed_ready
        operands[_Fs.Y_ROUTED_READY] = y_routed_ready
        operands[_Fs.Y_ROUTED_DONE] = y_routed_done
        operands[_Fs.Y_ROUTED] = y_routed
        operands[_Fs.X_PEER_TABLE] = x_peer_table
        operands[_Fs.COMBINE_PEER_TABLE] = combine_peer_table
        operands[_Fs.X_ROUTED_BASE] = x_routed.iterator.toint()
        operands[_Fs.X_T_BASE] = x_t.iterator.toint()
        operands[_Fs.X_SC_BASE] = x_sc.iterator.toint()
        operands[_Fs.X_SC_T_BASE] = x_sc_t.iterator.toint()
        operands[_Fs.LAST_MACRO] = last_macro
        operands[_Fs.LAST_TILES] = last_tiles
        operands[_Fs.FULL_TILES] = full_tiles
        operands[_Fs.NUM_TOKENS] = num_tokens
        operands[_Fs.HIDDEN_SIZE] = hidden_size
        operands[_Fs.TOPK] = topk
        operands[_Fs.MACROBATCH_SIZE] = macrobatch_size
        operands[_Fs.MINIBATCH_SIZE] = minibatch_size
        operands[_Fs.RECIPROCAL_2F] = reciprocal_2f
        operands[_Fs.RECIPROCAL_MINI] = reciprocal_mini
        operands[_Fs.RECIPROCAL_TOPK] = reciprocal_topk
        assert all(operand is not None for operand in operands)
        self._run_stream(
            operands,
            schedule_peer_rank,
            schedule_peer_token_idx,
            head,
            arena,
            stream_mbarriers,
            blocks,
            groups,
            reciprocal_groups,
            reciprocal_group,
            column_blocks,
            comm_cta,
            num_comm_sms,
            Int32(0),
        )

    @cute.jit
    def _load_routed_scales(
        self,
        kind: Int32,
        row: Int32,
        column: Int32,
        expert: Int32,
        cta_rank: Int32,
        macro: Int32,
        mini: Int32,
        num_tokens: Int32,
        x_routed_ready: cute.Tensor,
        hidden_row_block_ready: cute.Tensor,
        hidden_size: Int32,
        intermediate_size: Int32,
        num_local_tokens: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        k_tiles: Int32,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        atoms: list,
        tensors: list,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sf_pipeline,
        sf_producer_state,
        num_experts: Int32,
        up_row_offset: Int32,
        descriptors=None,
    ):
        """The scale-loading warp for one routed tile: the same operand-ready wait as the A/B loads,
        then per K tile the SFA and SFB loads onto the scale ring.  ``up_row_offset`` (0 for separate
        Gate and Up weights) is in rows, as for the payload: the scale's coordinates are the operand's
        rows.  ``descriptors`` (split_weights) is None in every other variant."""
        if kind == Int32(ROUTED_DOWN):
            ready_row = Int32(num_local_tokens // 256) + macro * Int32(macrobatch_size // 256) + row
            with cute.arch.elect_one():
                _counter_wait_gpu(hidden_row_block_ready, ready_row, Int32(2) * (intermediate_size // Int32(128)))
        else:
            generation = macro * Int32(macrobatch_size // minibatch_size) + mini
            mini_rows = cutlass.min(Int32(minibatch_size), num_tokens - generation * Int32(minibatch_size))
            required = _ceil_div_i32(mini_rows, Int32(STREAM_ROWS)) * _ceil_div_i32(hidden_size, Int32(STREAM_COLUMNS))
            with cute.arch.elect_one():
                _counter_wait_gpu(x_routed_ready, generation, required)
        cute.arch.sync_warp()
        if kind == Int32(ROUTED_GATE_UP):
            # Fused FC1: CTA 0's half of the scale tile comes from the Gate scales, CTA 1's from the Up
            # scales, both rows [128 c, 128 c + 128) of the 128-column tile c (see _load_gate_up).
            offset = _fc1_half_offset(column, cta_rank)
            if cta_rank == Int32(0):
                sfb, sfb_desc = self._routed_operand(tensors[_Mx.SFB_GATE], descriptors, num_experts, 3, expert)
                sf_producer_state = self._load_scale_pair(
                    atoms[_Mx.SFA_X],
                    tensors[_Mx.SFA_X],
                    atoms[_Mx.SFB_GATE],
                    mxgemm.offset_scale_rows(sfb, offset),
                    row,
                    column // Int32(2),
                    cta_rank,
                    k_tiles,
                    tiled_mma,
                    tiled_mma_sfb,
                    cluster_layout_vmnk,
                    cluster_layout_sfb_vmnk,
                    sSFA,
                    sSFB,
                    sf_pipeline,
                    sf_producer_state,
                    sfb_desc_ptr=sfb_desc,
                )
            else:
                sfb, sfb_desc = self._routed_operand(tensors[_Mx.SFB_UP], descriptors, num_experts, 4, expert)
                sf_producer_state = self._load_scale_pair(
                    atoms[_Mx.SFA_X],
                    tensors[_Mx.SFA_X],
                    atoms[_Mx.SFB_UP],
                    mxgemm.offset_scale_rows(sfb, offset + up_row_offset),
                    row,
                    column // Int32(2),
                    cta_rank,
                    k_tiles,
                    tiled_mma,
                    tiled_mma_sfb,
                    cluster_layout_vmnk,
                    cluster_layout_sfb_vmnk,
                    sSFA,
                    sSFB,
                    sf_pipeline,
                    sf_producer_state,
                    sfb_desc_ptr=sfb_desc,
                )
        else:
            sfb, sfb_desc = self._routed_operand(tensors[_Mx.SFB_DOWN], descriptors, num_experts, 5, expert)
            sf_producer_state = self._load_scale_pair(
                atoms[_Mx.SFA_HIDDEN],
                tensors[_Mx.SFA_HIDDEN],
                atoms[_Mx.SFB_DOWN],
                sfb,
                row,
                column,
                cta_rank,
                k_tiles,
                tiled_mma,
                tiled_mma_sfb,
                cluster_layout_vmnk,
                cluster_layout_sfb_vmnk,
                sSFA,
                sSFB,
                sf_pipeline,
                sf_producer_state,
                sfb_desc_ptr=sfb_desc,
            )
        return sf_producer_state

    @cute.jit
    def _routed_operand(self, tensor: cute.Tensor, descriptors, num_experts: Int32, operand: cutlass.Constexpr[int], expert: Int32):
        """One routed operand for one expert: the view to tile, and the descriptor to read it with
        (``descriptors``: the kernel's routed_descriptors, None unless split_weights).

        Grouped: the L mode is sliced and the atom's own descriptor is used.  Split: the operand
        *is* one expert's allocation, so there is no L mode and the expert is named by its
        descriptor instead.  ``operand`` indexes the address table's rows -- the three payloads,
        then their three scales.
        """
        if cutlass.const_expr(self.split_weights):
            return tensor, routed_descriptors_pkg.descriptor_argument(routed_descriptors_pkg.operand_descriptors(descriptors, num_experts, operand), expert)
        return tensor[None, None, expert], None

    @cute.jit
    def _load_scale_pair(
        self,
        sfa_atom: cute.CopyAtom,
        sfa: cute.Tensor,
        sfb_atom: cute.CopyAtom,
        sfb: cute.Tensor,
        row: Int32,
        column: Int32,
        cta_rank: Int32,
        k_tiles: Int32,
        tiled_mma,
        tiled_mma_sfb,
        cluster_layout_vmnk,
        cluster_layout_sfb_vmnk,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sf_pipeline,
        sf_producer_state,
        sfb_desc_ptr=None,
    ):
        copy_sfa, copy_sfb = mxgemm.make_sf_load_fns(
            self.mxfp8_collective,
            tiled_mma,
            tiled_mma_sfb,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            sfa_atom,
            sfa,
            sfb_atom,
            sfb,
            sSFA,
            sSFB,
            row,
            column,
            cta_rank,
            sfb_desc_ptr=sfb_desc_ptr,
        )
        return self.mxfp8_collective.load_tma(sf_pipeline, sf_producer_state, [copy_sfa, copy_sfb], k_tiles)

    @cute.jit
    def _run_comm_prefix(
        self,
        x_peer_table: cute.Tensor,
        combine_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens_tensor: cute.Tensor,
        y_routed: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        stream_mbarriers: cute.Pointer,
        arena: cute.Pointer,
        cluster_index: Int32,
        cta_rank: Int32,
        hidden_size: Int32,
        topk: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
        quant_outputs: list,
    ) -> None:
        """The communication CTAs' branch: the stream buffers' barriers, then the task stream.  ``hidden_size``
        here is the routed chain's hidden (the width of every row moved; the kernel passes its routed hidden);
        ``quant_outputs``: the plain row-major ``x_routed`` and ``x_routed_t`` and the two scale tensors the
        quantization writes."""
        thread = cute.arch.thread_idx()[0]
        if thread == Int32(0):
            for barrier in cutlass.range_constexpr(STREAM_MBARRIERS):
                cute.arch.mbarrier_init(stream_mbarriers + Int32(barrier), STREAM_ROWS)  # one arrival per pulled row
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        num_tokens = num_tokens_tensor[0]
        comm_cta = cluster_index * Int32(CLUSTER_SIZE) + cta_rank
        self._forward_stream(
            schedule_peer_rank,
            schedule_peer_token_idx,
            x_routed_ready,
            y_routed_ready,
            y_routed_done,
            y_routed,
            arena,
            stream_mbarriers,
            num_tokens,
            hidden_size,
            topk,
            macrobatch_size,
            minibatch_size,
            comm_cta,
            num_comm_sms,
            x_peer_table,
            combine_peer_table,
            quant_outputs,
        )

    @cute.jit
    def _run_swiglu(
        self,
        local: Int32,
        cta_rank: Int32,
        load_atoms,
        load_tensors,
        store_atoms,
        store_tensors,
        smem_layout,
        arena,
        barriers,
        phase_bits: Int32,
        gate_up_tile_ready,
        hidden_row_block_ready,
        intermediate_size: Int32,
        num_local_tokens: Int32,
        num_tokens: Int32,
        macro_size: Int32,
        swiglu_limit: Float32,
        situ_beta1=None,
        situ_beta2=None,
    ):
        """The shared expert's SwiGLU task (BF16): three 128x128 input pairs and one reusable Hidden
        tile per CTA.  The routed experts have no SwiGLU task: their fused FC1 epilogue computes and
        quantizes Hidden (``mxgemm.store_acc_fused_swiglu``).  ``situ_beta1`` / ``situ_beta2``: Float32
        in the is_situ variant, None otherwise."""
        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        sg = cute.make_tensor(cute.recast_ptr(arena, smem_layout.inner, dtype=BFloat16), smem_layout.outer)
        su = cute.make_tensor(cute.recast_ptr(arena + 3 * SWIGLU_TILE_BYTES, smem_layout.inner, dtype=BFloat16), smem_layout.outer)
        hidden_layout = cute.select(smem_layout, mode=[0, 1])
        sh = cute.make_tensor(cute.recast_ptr(arena + 6 * SWIGLU_TILE_BYTES, hidden_layout.inner, dtype=BFloat16), hidden_layout.outer)
        base_tile = (local * Int32(2) + cta_rank) * Int32(3)
        local_limit = Int32((num_local_tokens // 128) * (intermediate_size // 128))
        # The three loads are issued before the stage-consumption loop.
        for stage in cutlass.range_constexpr(3):
            tile = base_tile + Int32(stage)
            global_row = tile // (intermediate_size // Int32(128))
            column = tile % (intermediate_size // Int32(128))
            valid = tile < local_limit
            if valid and thread == Int32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(barriers + stage, Int32(2 * SWIGLU_TILE_BYTES))
                _counter_wait_gpu(gate_up_tile_ready, (global_row // Int32(2)) * self._column_tiles(intermediate_size) + column // Int32(2), Int32(4))
            if valid and warp == Int32(0):
                cute.arch.sync_warp()
                _issue_swiglu_tma_load(load_atoms[0], load_tensors[0], global_row, column, sg[None, None, stage], barriers + stage)
                _issue_swiglu_tma_load(load_atoms[1], load_tensors[1], global_row, column, su[None, None, stage], barriers + stage)
        for stage in cutlass.range_constexpr(3):
            phase_bits = self._swiglu_stage(
                False,
                Int32(stage),
                Int32(SHARED_SWIGLU),
                base_tile,
                Int32(0),
                local_limit,
                num_tokens,
                barriers,
                phase_bits,
                sg,
                su,
                sh,
                arena,
                thread,
                swiglu_limit,
                intermediate_size,
                macro_size,
                store_atoms,
                store_tensors,
                [],
                situ_beta1,
                situ_beta2,
            )
        if thread == Int32(0):
            cute.arch.cp_async_bulk_wait_group(0, read=False)
            for stage in cutlass.range_constexpr(3):
                tile = base_tile + Int32(stage)
                global_row = tile // (intermediate_size // Int32(128))
                if tile < local_limit:
                    _counter_arrive_release_gpu(hidden_row_block_ready, global_row // Int32(2))

        return phase_bits

    @cute.jit
    def _swiglu_stage(
        self,
        routed: cutlass.Constexpr[bool],
        stage: Int32,
        kind: Int32,
        base_tile: Int32,
        global_base: Int32,
        local_limit: Int32,
        num_tokens: Int32,
        barriers: cute.Pointer,
        phase_bits: Int32,
        sg: cute.Tensor,
        su: cute.Tensor,
        sh: cute.Tensor,
        arena: cute.Pointer,
        thread: Int32,
        swiglu_limit: Float32,
        intermediate_size: Int32,
        macro_size: Int32,
        store_atoms: list,
        store_tensors: list,
        quant_outputs: list,
        situ_beta1=None,
        situ_beta2=None,
    ) -> Int32:
        """One SwiGLU input stage: wait, compute Hidden, then quantize (routed) or store BF16 (shared).
        ``situ_beta1`` / ``situ_beta2``: Float32 in the is_situ variant, None (unread) otherwise.

        The forward's own SwiGLU tasks are the shared expert's (``routed`` False); the routed branch
        (``routed`` True, ``kind`` = ``REPLAY_SWIGLU_KIND``) is the backward's replay SwiGLU, which
        reuses this method and ``_quantize_hidden_tile``.
        """
        tile = base_tile + stage
        global_row = (global_base + tile) // (intermediate_size // Int32(128))
        column = (global_base + tile) % (intermediate_size // Int32(128))
        valid = tile < local_limit
        if kind == Int32(REPLAY_SWIGLU_KIND):
            valid = valid and global_row * Int32(128) < num_tokens
        if valid:
            cute.arch.mbarrier_wait(barriers + stage, (phase_bits >> stage) & Int32(1))
            phase_bits = phase_bits ^ (Int32(1) << stage)
            copy_128b = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
            values = cute.make_rmem_tensor((64,), BFloat16)
            r_gate = cute.make_rmem_tensor((8,), BFloat16)
            r_up = cute.make_rmem_tensor((8,), BFloat16)
            for group in cutlass.range_constexpr(8):
                row = thread // Int32(16) + Int32(group * 16)
                column_group = thread % Int32(16)
                s_gate_group = cute.coalesce(cute.local_tile(sg[None, None, stage], (1, 8), (row, column_group)))
                s_up_group = cute.coalesce(cute.local_tile(su[None, None, stage], (1, 8), (row, column_group)))
                cute.copy(copy_128b, s_gate_group, r_gate)
                cute.copy(copy_128b, s_up_group, r_up)
                for element in cutlass.range_constexpr(8):
                    gate = Float32(r_gate[element])
                    up = Float32(r_up[element])
                    if cutlass.const_expr(self.is_clamped):
                        # The clamp is a single min.f32 / max.f32 each.
                        gate = cute.arch.fmin(gate, Float32(swiglu_limit))
                        up = cute.arch.fmin(cute.arch.fmax(up, -Float32(swiglu_limit)), Float32(swiglu_limit))
                    values[group * 8 + element] = glu_forward(gate, up, self.is_situ, situ_beta1, situ_beta2).to(BFloat16)
            if cutlass.const_expr(routed):
                self._quantize_hidden_tile(
                    values, arena, stage, thread, global_row % Int32(macro_size // 128), column, intermediate_size, macro_size, quant_outputs
                )
            else:
                self._store_hidden_bf16(values, sh, thread, global_row, column, store_atoms, store_tensors)
        return phase_bits

    @cute.jit
    def _store_hidden_bf16(
        self, values: cute.Tensor, sh: cute.Tensor, thread: Int32, global_row: Int32, column: Int32, store_atoms: list, store_tensors: list
    ) -> None:
        """One shared BF16 Hidden tile (the routed one is e4m3): wait the previous store's
        read, stage it, TMA store."""
        copy_128b = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        if thread == Int32(0):
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.sync_threads()
        for group in cutlass.range_constexpr(8):
            row = thread // Int32(16) + Int32(group * 16)
            column_group = thread % Int32(16)
            s_hidden_group = cute.coalesce(cute.local_tile(sh, (1, 8), (row, column_group)))
            r_hidden_group = cute.local_tile(values, (8,), (group,))
            cute.copy(copy_128b, r_hidden_group, s_hidden_group)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        if thread == Int32(0):
            _issue_swiglu_tma_store(store_atoms[0], store_tensors[0], global_row, column, sh)
            cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def _quantize_hidden_tile(
        self,
        values: cute.Tensor,
        arena: cute.Pointer,
        stage: Int32,
        thread: Int32,
        row: Int32,
        column: Int32,
        intermediate_size: Int32,
        macro_size: Int32,
        quant_outputs: list,
    ) -> None:
        """Quantize and store one 128 x 128 routed Hidden tile (the backward's replayed routed SwiGLU).

        Hidden goes to the plain (unswizzled) 128 x 128 tile, ``quantize_tile`` writes the normal
        e4m3 tile and scales over this stage's Gate buffer and the transposed ones over its Up
        buffer (both consumed), and thread 0 stores the four.  No earlier store has to be waited
        for here: those read other stages' buffers.
        """
        hidden_fp8_atom, hidden_fp8, hidden_t_atom, hidden_t, hidden_sc, hidden_sc_t = quant_outputs
        copy_128b = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        hidden = arena + 6 * SWIGLU_TILE_BYTES
        sh_plain = cute.make_tensor(cute.recast_ptr(hidden, dtype=BFloat16), cute.make_layout((SWIGLU_TILE_M, SWIGLU_TILE_N), stride=(SWIGLU_TILE_N, 1)))
        for group in cutlass.range_constexpr(8):
            tile_row = thread // Int32(16) + Int32(group * 16)
            column_group = thread % Int32(16)
            s_hidden_group = cute.coalesce(cute.local_tile(sh_plain, (1, 8), (tile_row, column_group)))
            cute.copy(copy_128b, cute.local_tile(values, (8,), (group,)), s_hidden_group)
        cute.arch.sync_threads()
        payload = arena + stage * Int32(SWIGLU_TILE_BYTES)
        payload_t = arena + (Int32(3) + stage) * Int32(SWIGLU_TILE_BYTES)
        # Split passes: all 256 threads, the transposed pass on threads 0-127 and the normal pass on 128-255
        # side by side.  The per-block arithmetic does not depend on the split.  Neither output overlaps the
        # plain Hidden tile, so the passes need no barrier between their loads and stores.
        quantize_tile(hidden, payload, payload + QUANT_FP8_TILE_BYTES, payload_t, payload_t + QUANT_FP8_TILE_BYTES, thread, SWIGLU_TILE_N, True, True, True)
        cute.arch.sync_threads()
        if thread == Int32(0):
            # The issuing thread's proxy fence, after the block sync and before its TMA stores.
            cute.arch.fence_view_async_shared()
            tile_layout = cute.make_layout((QUANT_TILE, QUANT_TILE), stride=(QUANT_TILE, 1))
            mxgemm.store_tile(
                hidden_fp8_atom,
                hidden_fp8,
                (QUANT_TILE, QUANT_TILE),
                (row, column),
                cute.make_tensor(cute.recast_ptr(payload, dtype=Float8E4M3FN), tile_layout),
            )
            cute.arch.cp_async_bulk_commit_group()
            tma_store_1d_raw(
                hidden_sc.iterator.toint() + (Int64(row) * Int64(intermediate_size // Int32(QUANT_TILE)) + Int64(column)) * Int64(QUANT_SF_TILE_BYTES),
                payload + QUANT_FP8_TILE_BYTES,
                Int32(QUANT_SF_TILE_BYTES),
            )
            cute.arch.cp_async_bulk_commit_group()
            mxgemm.store_tile(
                hidden_t_atom, hidden_t, (QUANT_TILE, QUANT_TILE), (column, row), cute.make_tensor(cute.recast_ptr(payload_t, dtype=Float8E4M3FN), tile_layout)
            )
            cute.arch.cp_async_bulk_commit_group()
            tma_store_1d_raw(
                hidden_sc_t.iterator.toint() + (Int64(column) * Int64(macro_size // Int32(QUANT_TILE)) + Int64(row)) * Int64(QUANT_SF_TILE_BYTES),
                payload_t + QUANT_FP8_TILE_BYTES,
                Int32(QUANT_SF_TILE_BYTES),
            )
            cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def _column_tiles(self, size: Int32) -> Int32:
        """The MLP_TILE_COLUMNS-wide tiles across an H / I extent: a floor division, and in the
        has_tail variant one more tile for a 128-wide tail."""
        if cutlass.const_expr(self.has_tail):
            return _ceil_div_i32(size, Int32(MLP_TILE_COLUMNS))
        return size // Int32(MLP_TILE_COLUMNS)

    @cute.jit
    def _task_counts(
        self,
        num_tokens: Int32,
        shared_hidden_size: Int32,
        routed_hidden_size: Int32,
        intermediate_size: Int32,
        num_local_tokens: Int32,
        macro_size: Int32,
        mini_size: Int32,
    ):
        """The per-launch task-table constants (the shared task counts, the tasks per minibatch, the
        minibatches per macrobatch and those of the last macrobatch), computed once per CTA before the
        task loop.  The routed group is the fused one: FC1 tasks of 256 rows x 128 Hidden columns, then
        Down (``routed_hidden_size`` columns; the shared Down has ``shared_hidden_size``)."""
        shared_gate = (num_local_tokens // Int32(256)) * self._column_tiles(intermediate_size)
        shared_swiglu = ((num_local_tokens // Int32(128)) * (intermediate_size // Int32(128)) + Int32(5)) // Int32(6)
        shared_down = (num_local_tokens // Int32(256)) * self._column_tiles(shared_hidden_size)
        if cutlass.const_expr(self.shared_fc1_only):
            shared_down = Int32(0)  # the task table ends after the shared SwiGLU tasks
        shared_total = Int32(2) * shared_gate + shared_swiglu + shared_down
        mini_gate_up = (mini_size // Int32(256)) * (intermediate_size // Int32(FC1_TILE_COLUMNS))
        mini_down = (mini_size // Int32(256)) * self._column_tiles(routed_hidden_size)
        mini_total = mini_gate_up + mini_down
        macro_minis = macro_size // mini_size
        num_macros = _ceil_div_i32(num_tokens, macro_size)
        last_minis = _ceil_div_i32(num_tokens, mini_size) - (num_macros - Int32(1)) * macro_minis
        return (shared_gate, shared_swiglu, shared_total, mini_gate_up, mini_total, macro_minis, num_macros, last_minis)

    @cute.jit
    def _decode_task_hoisted(
        self,
        index: Int32,
        shared_gate: Int32,
        shared_swiglu: Int32,
        shared_total: Int32,
        mini_gate_up: Int32,
        mini_total: Int32,
        macro_minis: Int32,
        num_macros: Int32,
        last_minis: Int32,
    ):
        """CLC task ``index`` -> (kind, index within the kind, macrobatch, minibatch) from ``_task_counts``'
        constants: two runtime divisions per task (by the tasks per minibatch and by the minibatches per
        macrobatch).  The shared group comes first; the routed minibatches follow, the last macrobatch's
        first, then the earlier macrobatches in descending order."""
        macro = Int32(0)
        mini = Int32(0)
        kind = Int32(SHARED_GATE)
        local = index
        if index >= shared_total:
            routed = index - shared_total
            ordered_mini = routed // mini_total
            local = routed - ordered_mini * mini_total
            if ordered_mini < last_minis:
                macro = num_macros - Int32(1)
                mini = ordered_mini
            else:
                remaining = ordered_mini - last_minis
                macro_step = remaining // macro_minis
                macro = num_macros - Int32(2) - macro_step
                mini = remaining - macro_step * macro_minis
            # The routed group: fused FC1 tasks, then Down.
            kind = Int32(ROUTED_GATE_UP)
            if local >= mini_gate_up:
                kind = Int32(ROUTED_DOWN)
                local = local - mini_gate_up
        else:
            # The shared group: Gate, Up, SwiGLU, Down.
            if local >= shared_gate:
                if local < shared_gate * Int32(2):
                    kind = Int32(SHARED_UP)
                    local = local - shared_gate
                elif local < shared_gate * Int32(2) + shared_swiglu:
                    kind = Int32(SHARED_SWIGLU)
                    local = local - shared_gate * Int32(2)
                else:
                    kind = Int32(SHARED_DOWN)
                    local = local - shared_gate * Int32(2) - shared_swiglu
        return kind, local, macro, mini

    @cute.jit
    def _task_is_swiglu(self, index: Int32, shared_gate: Int32, shared_swiglu: Int32, shared_total: Int32) -> Int32:
        """Whether CLC item ``index`` is a SwiGLU (CTA-local) task: two compares decide the SwiGLU ->
        GEMM cluster sync (only the shared group has SwiGLU tasks)."""
        is_swiglu = Int32(0)
        if index < shared_total:
            if index >= shared_gate * Int32(2):
                if index < shared_gate * Int32(2) + shared_swiglu:
                    is_swiglu = Int32(1)
        return is_swiglu

    @cute.jit
    def _decode_gemm_task(
        self,
        kind: Int32,
        local: Int32,
        macro: Int32,
        mini: Int32,
        tokens_per_expert: cute.Tensor,
        expert_cu_end: cute.Tensor,
        expert_stride: Int32,
        expert_groups: Int32,
        shared_hidden_size: Int32,
        routed_hidden_size: Int32,
        intermediate_size: Int32,
        num_local_tokens: Int32,
        macro_size: Int32,
        mini_size: Int32,
    ):
        """``(row, column, expert)`` of a GEMM task.

        Shared: a swizzled tile of the local tokens.  Routed: the minibatch's row blocks that fall
        in each expert, in expert order, by the compile-time ``TaskDecode`` (the shared-memory ``cu_end``
        table searched per expert or per expert group, or a scan of ``tokens_per_expert``);
        ``row`` is macrobatch-local, ``expert`` -1 for a capacity tile.  Row blocks are 256 rows for
        every kind; the fused FC1's column blocks are 128 Hidden columns, the others 256 (the routed
        Down's over ``routed_hidden_size``, the shared Down's over ``shared_hidden_size``).
        """
        column_blocks = self._column_tiles(intermediate_size)
        if kind == Int32(ROUTED_GATE_UP):
            column_blocks = intermediate_size // Int32(FC1_TILE_COLUMNS)
        if kind == Int32(SHARED_DOWN):
            column_blocks = self._column_tiles(shared_hidden_size)
        if kind == Int32(ROUTED_DOWN):
            column_blocks = self._column_tiles(routed_hidden_size)
        row = Int32(0)
        column = Int32(0)
        expert = Int32(-1)
        if kind < Int32(ROUTED_GATE_UP):
            row, column = _swizzled_mlp_tile_coord_i32(Int32(num_local_tokens // MLP_TILE_ROWS), column_blocks, local)
            expert = Int32(0)
        else:
            first_mini_row = (macro * macro_size + mini * mini_size) // Int32(MLP_TILE_ROWS)
            valid, owner, row_block, tile_column, _ = _routed_expert_tile_i32(
                tokens_per_expert,
                expert_cu_end,
                expert_stride,
                expert_groups,
                self.task_decode,
                first_mini_row,
                mini_size // Int32(MLP_TILE_ROWS),
                column_blocks,
                local,
            )
            if valid != Int32(0):
                row = row_block - macro * (macro_size // Int32(MLP_TILE_ROWS))
                column = tile_column
                expert = owner
        return row, column, expert

    @cute.jit
    def _load_gemm(
        self,
        a_atom: cute.CopyAtom,
        a_tensor: cute.Tensor,
        b_atom: cute.CopyAtom,
        b_tensor: cute.Tensor,
        tiled_mma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        row: Int32,
        column: Int32,
        cta_rank: Int32,
        k_tiles: Int32,
        ab_pipeline,
        ab_producer_state,
        k_tile: cutlass.Constexpr[int] = gemm.K_TILE,
        b_desc_ptr=None,
    ):
        thr_mma = tiled_mma.get_slice(cta_rank)
        a_tile = cute.local_tile(a_tensor, (256, k_tile), (row, None))
        b_tile = cute.local_tile(b_tensor, (256, k_tile), (column, None))
        copy_A = _make_tma_block_load_fn(a_atom, thr_mma.partition_A(a_tile), sA)
        copy_B = _make_tma_block_load_fn(b_atom, thr_mma.partition_B(b_tile), sB, tma_desc_ptr=b_desc_ptr)
        # Producer loop: try_acquire peek, acquire (arms the transaction barrier), TMA issue, advance.
        return self.collective.load_tma(ab_pipeline, ab_producer_state, [copy_A, copy_B], k_tiles)

    @cute.jit
    def _load_gate_up(
        self,
        a_atom: cute.CopyAtom,
        a_tensor: cute.Tensor,
        b_atom: cute.CopyAtom,
        b_tensor: cute.Tensor,
        tiled_mma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        row: Int32,
        column: Int32,
        cta_rank: Int32,
        k_tiles: Int32,
        ab_pipeline,
        ab_producer_state,
        k_tile: cutlass.Constexpr[int] = gemm.K_TILE,
        b_desc_ptr=None,
    ):
        """The fused FC1's operands: A as ``_load_gemm``; B is this CTA's own weights (``b_tensor``:
        rank 0 the Gate weights, rank 1 the Up weights), rows ``[128 c, 128 c + 128)`` of the
        128-column tile ``c``.  The 2-CTA partition gives rank r half r of a 256-row tile, so the
        tile index is ``c // 2`` on a view shifted by ``_fc1_half_offset`` rows; partition, multicast
        and transaction bytes are those of the plain GEMM load."""
        thr_mma = tiled_mma.get_slice(cta_rank)
        a_tile = cute.local_tile(a_tensor, (256, k_tile), (row, None))
        b_view = cute.domain_offset((_fc1_half_offset(column, cta_rank), Int32(0)), b_tensor)
        b_tile = cute.local_tile(b_view, (256, k_tile), (column // Int32(2), None))
        copy_A = _make_tma_block_load_fn(a_atom, thr_mma.partition_A(a_tile), sA)
        copy_B = _make_tma_block_load_fn(b_atom, thr_mma.partition_B(b_tile), sB, tma_desc_ptr=b_desc_ptr)
        return self.collective.load_tma(ab_pipeline, ab_producer_state, [copy_A, copy_B], k_tiles)

    @cute.jit
    def _epilogue_gemm(
        self,
        d_atom: cute.CopyAtom,
        d_tensor: cute.Tensor,
        tiled_mma,
        epi_tile,
        sD: cute.Tensor,
        acc_ptr,
        acc_pipeline,
        acc_consumer_state,
        thread: Int32,
        cta_rank: Int32,
        row: Int32,
        column: Int32,
        kind: Int32,
        macro: Int32,
        mini: Int32,
        num_tokens: Int32,
        gate_up_tile_ready,
        y_routed_ready,
        y_routed_done,
        hidden_size: Int32,
        intermediate_size: Int32,
        macro_size: Int32,
        mini_size: Int32,
    ):
        """The BF16 epilogue of the shared GEMMs and the routed Down: eight D3 stores, then the
        task's counter (``gate_up_tile_ready`` for the shared Gate / Up, ``y_routed_ready`` for the routed Down)."""
        collective = self.collective
        fake = tiled_mma.make_fragment_C(cute.append(tiled_mma.partition_shape_C(collective.mma_tiler[:2]), 1))
        acc_tensor = cute.make_tensor(acc_ptr, fake.layout)
        t2r, acc_partition, r_acc = collective.epilog_tmem_copy_and_partition(thread, acc_tensor, epi_tile, True)
        r_d = cute.make_rmem_tensor(r_acc.shape, Float32)
        r2s, r_d_partition, s_d_partition = collective.epilog_smem_store_and_partition(t2r, collective.d_layout, BFloat16, r_d, sD, thread)
        acc_partition = cute.group_modes(acc_partition[None, None, None, None, None, 0], 3, 5)
        copy_D, _, _ = collective.epilog_gmem_copy_and_partition(
            d_atom, d_tensor, collective.cta_tile_shape_mnk[:2], epi_tile, sD, (row * Int32(2) + cta_rank, column, Int32(0), Int32(0))
        )
        acc_consumer_state = self._epilogue_bf16(
            collective,
            t2r,
            r2s,
            acc_partition,
            r_acc,
            r_d_partition,
            s_d_partition,
            copy_D,
            acc_pipeline,
            acc_consumer_state,
            thread,
            cta_rank,
            row,
            kind,
            macro,
            num_tokens,
            y_routed_done,
            hidden_size,
            macro_size,
        )
        if kind != Int32(SHARED_DOWN):
            gemm.finish_store(thread)
            if thread == Int32(0):
                if kind == Int32(ROUTED_DOWN):
                    _counter_arrive_release_gpu(y_routed_ready, macro * Int32(macro_size // mini_size) + mini)
                else:
                    _counter_arrive_release_gpu(gate_up_tile_ready, row * self._column_tiles(intermediate_size) + column)
        return acc_consumer_state

    @cute.jit
    def _epilogue_gate_up_fused(
        self,
        tiled_mma,
        epi_tile,
        acc_ptr,
        acc_pipeline,
        acc_consumer_state,
        thread: Int32,
        cta_rank: Int32,
        row: Int32,
        column: Int32,
        macro: Int32,
        hidden_row_block_ready,
        intermediate_size: Int32,
        num_local_tokens: Int32,
        macro_size: Int32,
        swiglu_limit: Float32,
        atoms: list,
        tensors: list,
        d_stages: cute.Pointer,
        situ_beta1=None,
        situ_beta2=None,
    ):
        """The fused routed FC1 epilogue (``mxgemm.store_acc_fused_swiglu``): the BF16 Gate / Up
        context, Hidden quantized in both layouts, its two scale tiles; then ``hidden_row_block_ready`` of the
        256-row block, once per CTA (``2 x (intermediate / 128)`` arrivals per block, the Down's
        expectation).  ``situ_beta1`` / ``situ_beta2``: Float32 in the is_situ variant, None otherwise."""
        collective = self.collective
        fake = tiled_mma.make_fragment_C(cute.append(tiled_mma.partition_shape_C(collective.mma_tiler[:2]), 1))
        acc_tensor = cute.make_tensor(acc_ptr, fake.layout)
        t2r, acc_partition, r_acc = collective.epilog_tmem_copy_and_partition(thread, acc_tensor, epi_tile, True)
        acc_partition = cute.group_modes(acc_partition[None, None, None, None, None, 0], 3, 5)
        acc_consumer_state = mxgemm.store_acc_fused_swiglu(
            collective,
            t2r,
            acc_partition,
            r_acc,
            atoms[_Mx.GATE_BF16],
            tensors[_Mx.GATE_BF16],
            atoms[_Mx.UP_BF16],
            tensors[_Mx.UP_BF16],
            atoms[_Mx.HIDDEN_FP8],
            tensors[_Mx.HIDDEN_FP8],
            tensors[_Mx.HIDDEN_SC].iterator,
            atoms[_Mx.HIDDEN_T],
            tensors[_Mx.HIDDEN_T],
            tensors[_Mx.HIDDEN_SC_T].iterator,
            intermediate_size // Int32(QUANT_TILE),
            macro_size // Int32(QUANT_TILE),
            d_stages,
            row * Int32(2) + cta_rank,
            column,
            self.is_clamped,
            swiglu_limit,
            acc_pipeline,
            acc_consumer_state,
            thread,
            self.is_situ,
            situ_beta1,
            situ_beta2,
        )
        gemm.finish_store(thread)
        if thread == Int32(0):
            _counter_arrive_release_gpu(hidden_row_block_ready, Int32(num_local_tokens // 256) + macro * Int32(macro_size // 256) + row)
        return acc_consumer_state

    @cute.jit
    def _epilogue_bf16(
        self,
        collective,
        t2r,
        r2s,
        acc_partition,
        r_acc,
        r_d_partition,
        s_d_partition,
        copy_D,
        acc_pipeline,
        acc_consumer_state,
        thread: Int32,
        cta_rank: Int32,
        row: Int32,
        kind: Int32,
        macro: Int32,
        num_tokens: Int32,
        y_routed_done,
        hidden_size: Int32,
        macro_size: Int32,
    ):
        """The BF16 epilogue: the accumulator to registers, TMEM released, eight D3 stores."""
        # Accumulator hand-off: wait full, load the eight N32 fragments,
        # release TMEM on the last one (per-warp elect, both CTAs), advance.
        acc_pipeline.consumer_wait(acc_consumer_state)
        registers = cute.make_rmem_tensor(cute.append(r_d_partition.shape, 8), BFloat16)
        for subtile in cutlass.range_constexpr(8):
            collective.epi_load_acc_subtile(t2r, r2s, acc_partition, r_acc, r_d_partition, (0, subtile), acc_pipeline, acc_consumer_state, 7)
            registers[None, None, None, subtile].store(r_d_partition.load().to(BFloat16))
        acc_consumer_state.advance()
        # Down cannot overwrite this macro ring before the older Y was sent.
        if kind == Int32(ROUTED_DOWN) and thread == Int32(0):
            previous_offset = (macro + Int32(1)) * Int32(macro_size)
            previous_rows = cutlass.min(Int32(macro_size), num_tokens - previous_offset)
            first_row = row * Int32(256) + cta_rank * Int32(128)
            if first_row < previous_rows:
                _counter_wait_gpu(
                    y_routed_done, (previous_offset + first_row) // Int32(128), Int32(Y_DONE_ROW_TILES) * _ceil_div_i32(hidden_size, Int32(Y_DONE_COLUMNS))
                )
        gemm.store_acc_all(collective, registers, r2s, s_d_partition, copy_D, thread)
        return acc_consumer_state

    @cute.jit
    def __call__(
        self,
        x_peer_table: cute.Tensor,
        combine_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        x_shared: cute.Tensor,
        x_routed: cute.Tensor,
        shared_gate_weights: cute.Tensor,
        routed_gate_weights: cute.Tensor,
        shared_up_weights: cute.Tensor,
        routed_up_weights: cute.Tensor,
        shared_down_weights: cute.Tensor,
        routed_down_weights: cute.Tensor,
        gate_shared: cute.Tensor,
        up_shared: cute.Tensor,
        hidden_shared: cute.Tensor,
        hidden_routed: cute.Tensor,
        y_shared: cute.Tensor,
        y_routed: cute.Tensor,
        gate_up_tile_ready: cute.Tensor,
        hidden_row_block_ready: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        # (6, num_local_experts) int64 base addresses -- Gate, Up, Down payloads then their three
        # scales -- and the 6 * num_local_experts * 128 bytes their descriptors live in.  Both are
        # one-element placeholders unless split_weights is set.
        routed_storage_tables: cute.Tensor,
        routed_descriptors: cute.Tensor,
        # Nonzero when routed_descriptors has to be built by this launch: once per set of address
        # tables (persistent_bf16_fwd.RoutedDescriptors).
        build_descriptors: Int32,
        routed_gate_weights_sc: cute.Tensor,
        routed_up_weights_sc: cute.Tensor,
        routed_down_weights_sc: cute.Tensor,
        x_routed_sc: cute.Tensor,
        x_routed_t: cute.Tensor,
        x_routed_sc_t: cute.Tensor,
        gate_routed_bf16: cute.Tensor,
        up_routed_bf16: cute.Tensor,
        hidden_routed_sc: cute.Tensor,
        hidden_routed_t: cute.Tensor,
        hidden_routed_sc_t: cute.Tensor,
        intermediate_size: Int32,
        num_local_experts: Int32,
        topk: Int32,
        num_local_tokens: Int32,
        schedule_capacity: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
        swiglu_limit: Float32,
        # The Up half's first row when Gate and Up are one combined FC1 parameter, 0 otherwise (in rows,
        # for the payload and the scale alike).  Both weight arguments then name the full tensor and its
        # full scale, and Up is read that far down from Gate.
        up_row_offset: Int32,
        # SiTU-GLU's two betas; unread unless is_situ.
        situ_beta1: Float32,
        situ_beta2: Float32,
        stream: cuda.CUstream,
    ):
        self._launch(
            x_peer_table,
            combine_peer_table,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            tokens_per_expert,
            x_shared,
            x_routed,
            shared_gate_weights,
            routed_gate_weights,
            shared_up_weights,
            routed_up_weights,
            shared_down_weights,
            routed_down_weights,
            gate_shared,
            up_shared,
            hidden_shared,
            hidden_routed,
            y_shared,
            y_routed,
            gate_up_tile_ready,
            hidden_row_block_ready,
            x_routed_ready,
            y_routed_ready,
            y_routed_done,
            intermediate_size,
            num_local_experts,
            topk,
            num_local_tokens,
            schedule_capacity,
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            swiglu_limit,
            stream,
            routed_gate_weights_sc,
            routed_up_weights_sc,
            routed_down_weights_sc,
            x_routed_sc,
            x_routed_t,
            x_routed_sc_t,
            (gate_routed_bf16, up_routed_bf16),
            hidden_routed_sc,
            hidden_routed_t,
            hidden_routed_sc_t,
            up_row_offset,
            routed_storage_tables,
            routed_descriptors,
            situ_beta1,
            situ_beta2,
            build_descriptors,
        )

    def _launch(
        self,
        x_peer_table,
        combine_peer_table,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        tokens_per_expert,
        x_shared,
        x_routed,
        shared_gate_weights,
        routed_gate_weights,
        shared_up_weights,
        routed_up_weights,
        shared_down_weights,
        routed_down_weights,
        gate_shared,
        up_shared,
        hidden_shared,
        hidden_routed,
        y_shared,
        y_routed,
        gate_up_tile_ready,
        hidden_row_block_ready,
        x_routed_ready,
        y_routed_ready,
        y_routed_done,
        intermediate_size,
        num_local_experts,
        topk,
        num_local_tokens,
        schedule_capacity,
        macrobatch_size,
        minibatch_size,
        num_comm_sms,
        swiglu_limit,
        stream,
        routed_gate_weights_sc,
        routed_up_weights_sc,
        routed_down_weights_sc,
        x_routed_sc,
        x_routed_t,
        x_routed_sc_t,
        gate_up_context,
        hidden_routed_sc,
        hidden_routed_t,
        hidden_routed_sc_t,
        up_row_offset,
        routed_storage_tables,
        routed_descriptors,
        situ_beta1,
        situ_beta2,
        build_descriptors,
    ):
        """Host half of ``__call__`` (traced inside it); the last thirteen parameters are the routed-MXFP8
        extras (scale inputs, the e4m3 / scale outputs, the Gate / Up context: ``gate_up_context`` is
        ``(gate_bf16, up_bf16)``, or ``(gate_e4m3, gate_scales, up_e4m3, up_scales)`` in the
        ``_PersistentMxfp8FwdMegaE4m3Context`` variant, the Up row offset of a combined
        FC1, and the split weights' address tables and descriptor bytes).  The shared expert
        keeps the BF16 collective, the routed A/B atoms come from the block-scaled one.
        """
        collective = self.collective
        collective.configure(x_shared, shared_gate_weights, gate_shared)
        routed = self.mxfp8_collective
        # One expert's matrix: split allocations are that already, grouped ones slice the L mode.
        routed_gate_weights_0 = routed_gate_weights
        if cutlass.const_expr(not self.split_weights):
            routed_gate_weights_0 = routed_gate_weights[None, None, 0]
        routed.configure(x_routed, routed_gate_weights_0, y_routed)
        # Both precisions share one accumulator stage, 256 TMEM columns for it and one A/B ring
        # with one transaction count (16 KiB K tiles each).
        if (
            collective.num_acc_stage != 1
            or collective.num_tmem_alloc_cols != 256
            or routed.num_acc_stage != 1
            or routed.num_tma_load_bytes != collective.num_tma_load_bytes
        ):
            raise RuntimeError("the BF16 and MXFP8 GEMM collectives must agree on the accumulator stage, " "the TMEM columns and the A/B transaction bytes")
        # The kernel reads num_local_tokens as the shared expert's rows alone: its task counts and
        # the routed counters' offset past the shared row blocks.  A routed-only layer has none.
        if cutlass.const_expr(not self.has_shared):
            num_local_tokens = Int32(0)
        # Tensor maps are rebound every call; launch geometry is runtime.  The two hiddens are the shared
        # input's and the routed ring's widths.
        shared_hidden_size = Int32(x_shared.shape[1])
        routed_hidden_size = Int32(x_routed.shape[1])
        if cutlass.const_expr(self.shared_fc1_only):
            # The routed tensors are placeholders: I is the shared Gate weight's rows.
            intermediate_size = Int32(shared_gate_weights.shape[0])
        shared_tasks = (
            2 * (num_local_tokens // 256) * ((intermediate_size + 255) // 256)
            + ((num_local_tokens // 128) * (intermediate_size // 128) + 5) // 6
            + (num_local_tokens // 256) * ((shared_hidden_size + 255) // 256)
        )
        # The routed group: fused FC1 (256 rows x 128 Hidden columns) and Down; no SwiGLU tasks.
        minibatch_tasks = (minibatch_size // 256) * (intermediate_size // FC1_TILE_COLUMNS) + (minibatch_size // 256) * ((routed_hidden_size + 255) // 256)
        comm_clusters = num_comm_sms // 2
        capacity_clusters = comm_clusters + shared_tasks + ((schedule_capacity + minibatch_size - 1) // minibatch_size) * minibatch_tasks
        if cutlass.const_expr(self.shared_fc1_only):
            # The task table ends after the shared SwiGLU tasks (as _task_counts counts it on the device) and the
            # grid is exactly those clusters: no communication cluster (the kernel's count is 0 too), no shared
            # Down, no routed minibatch.  The host passes a routed token count of zero, so the kernel's true
            # cluster count is the grid: every cluster decodes a shared Gate, Up or SwiGLU task and the CLC
            # scheduler runs dry without a capacity tail to drain.
            shared_tasks = (
                2 * (num_local_tokens // 256) * ((intermediate_size + 255) // 256) + ((num_local_tokens // 128) * (intermediate_size // 128) + 5) // 6
            )
            capacity_clusters = shared_tasks
        a_layout = cute.slice_(collective.a_smem_layout_staged, (None, None, None, 0))
        b_layout = cute.slice_(collective.b_smem_layout_staged, (None, None, None, 0))
        routed_a_layout = cute.slice_(routed.a_smem_layout_staged, (None, None, None, 0))
        routed_b_layout = cute.slice_(routed.b_smem_layout_staged, (None, None, None, 0))
        a_pairs = [
            cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(collective.cta_group),
                tensor,
                a_layout,
                collective.mma_tiler,
                collective.tiled_mma,
                collective.cluster_layout_vmnk.shape,
            )
            for tensor in (x_shared, hidden_shared)
        ] + [
            cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(routed.cta_group), tensor, routed_a_layout, routed.mma_tiler, routed.tiled_mma, routed.cluster_layout_vmnk.shape
            )
            for tensor in (x_routed, hidden_routed)
        ]
        b_pairs = [
            cute.nvgpu.make_tiled_tma_atom_B(
                cpasync.CopyBulkTensorTileG2SOp(collective.cta_group),
                tensor,
                b_layout,
                collective.mma_tiler,
                collective.tiled_mma,
                collective.cluster_layout_vmnk.shape,
            )
            for tensor in (shared_gate_weights, shared_up_weights, shared_down_weights)
        ] + [
            cute.nvgpu.make_tiled_tma_atom_B(
                cpasync.CopyBulkTensorTileG2SOp(routed.cta_group), tensor, routed_b_layout, routed.mma_tiler, routed.tiled_mma, routed.cluster_layout_vmnk.shape
            )
            for tensor in (routed_gate_weights, routed_up_weights, routed_down_weights)
        ]
        d_pairs = [
            collective._make_tma_epi_atoms_and_tensors(tensor, collective.epi_smem_layout_staged, collective.epi_tile, op_type="store")
            for tensor in (gate_shared, up_shared, y_shared, y_routed)
        ]
        swiglu_layout = utils.make_smem_layout_epi(BFloat16, LayoutEnum.ROW_MAJOR, (128, 128), 3)
        swiglu_tile_layout = cute.select(swiglu_layout, mode=[0, 1])
        # The shared expert's SwiGLU task: its BF16 Gate / Up loads and Hidden store.
        swiglu_load_pairs = [
            cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), tensor, swiglu_tile_layout, (128, 128)) for tensor in (gate_shared, up_shared)
        ]
        swiglu_store_pairs = [cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), hidden_shared, swiglu_tile_layout, (128, 128))]
        mxfp8_atoms, mxfp8_tensors = self._mxfp8_atoms(
            x_routed,
            hidden_routed,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
            routed_gate_weights_sc,
            routed_up_weights_sc,
            routed_down_weights_sc,
            x_routed_sc,
            x_routed_t,
            x_routed_sc_t,
            *gate_up_context,
            hidden_routed_sc,
            hidden_routed_t,
            hidden_routed_sc_t,
        )
        scheduler_params = utils.ClcDynamicPersistentTileSchedulerParams((capacity_clusters * 2, 1, 1), CLUSTER_SHAPE)
        # The kernel takes the split and SiTU arguments only in those variants: None is a compile-time
        # constant, not a kernel parameter, so every other variant's kernel is the one without the feature.
        descriptors = None
        kernel_situ_beta1 = None
        kernel_situ_beta2 = None
        if cutlass.const_expr(self.is_situ):
            kernel_situ_beta1 = situ_beta1
            kernel_situ_beta2 = situ_beta2
        if cutlass.const_expr(self.split_weights):
            # One descriptor array per routed operand, filled by its own launch before the mega
            # kernel: that one is persistent, so every CTA would otherwise race to write the same
            # descriptors.  Order: the three payloads as the B atoms, then the three scales as the
            # SFB atoms, which is the order of the address table's rows.
            descriptors = cute.make_ptr(
                cutlass.Uint8, routed_descriptors.iterator.toint(), cute.AddressSpace.gmem, assumed_align=routed_descriptors_pkg.BYTES_PER_DESCRIPTOR
            )
            # _launch is not rewritten by the DSL: the runtime branch has to be built explicitly.
            routed_descriptors_pkg.build_routed_descriptors_if(
                build_descriptors,
                (b_pairs[3][0], b_pairs[4][0], b_pairs[5][0], mxfp8_atoms[_Mx.SFB_GATE], mxfp8_atoms[_Mx.SFB_UP], mxfp8_atoms[_Mx.SFB_DOWN]),
                routed_storage_tables,
                descriptors,
                Int32(cute.size(routed_storage_tables.shape[1])),
                stream,
            )
        self.kernel(
            collective.tiled_mma,
            [p[0] for p in a_pairs],
            [p[1] for p in a_pairs],
            [p[0] for p in b_pairs],
            [p[1] for p in b_pairs],
            [p[0] for p in d_pairs],
            [p[1] for p in d_pairs],
            [p[0] for p in swiglu_load_pairs],
            [p[1] for p in swiglu_load_pairs],
            [p[0] for p in swiglu_store_pairs],
            [p[1] for p in swiglu_store_pairs],
            collective.a_smem_layout_staged,
            collective.b_smem_layout_staged,
            collective.epi_smem_layout_staged,
            collective.epi_tile,
            swiglu_layout,
            scheduler_params,
            collective.cluster_layout_vmnk,
            x_peer_table,
            combine_peer_table,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            tokens_per_expert,
            x_routed,
            y_routed,
            gate_up_tile_ready,
            hidden_row_block_ready,
            x_routed_ready,
            y_routed_ready,
            y_routed_done,
            shared_hidden_size,
            intermediate_size,
            num_local_experts,
            topk,
            num_local_tokens,
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            shared_tasks,
            minibatch_tasks,
            swiglu_limit,
            routed.tiled_mma,
            routed.tiled_mma_sfb,
            routed.cluster_layout_sfb_vmnk,
            routed.a_smem_layout_staged,
            routed.b_smem_layout_staged,
            routed.sfa_smem_layout_staged,
            routed.sfb_smem_layout_staged,
            mxfp8_atoms,
            mxfp8_tensors,
            up_row_offset,
            descriptors,
            kernel_situ_beta1,
            kernel_situ_beta2,
        ).launch(
            grid=(capacity_clusters * 2, 1, 1),
            block=(256, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=_SharedStorageMxfp8.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    def _mxfp8_atoms(
        self,
        x_routed,
        hidden_routed,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        routed_gate_weights_sc,
        routed_up_weights_sc,
        routed_down_weights_sc,
        x_routed_sc,
        x_routed_t,
        x_routed_sc_t,
        gate_routed_bf16,
        up_routed_bf16,
        hidden_routed_sc,
        hidden_routed_t,
        hidden_routed_sc_t,
    ):
        """The ``_Mx`` lists: scale-load atoms over the scale tiles, e4m3 and BF16 context tile-store
        atoms, then the scale outputs (tensors only)."""
        m = self.mxfp8_collective
        sfa_layout = cute.slice_(m.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_layout = cute.slice_(m.sfb_smem_layout_staged, (None, None, None, 0))
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(m.cluster_shape_mnk, m.tiled_mma.thr_id)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(m.cluster_shape_mnk, m.tiled_mma.thr_id)

        def sfa_pair(scales, operand):
            return cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                mxgemm.scale_operand(scales, operand.shape),
                sfa_layout,
                m.mma_tiler,
                m.tiled_mma,
                m.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        def sfb_pair(scales, operand):
            return cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                mxgemm.scale_operand(scales, operand.shape),
                sfb_layout,
                m.mma_tiler_sfb,
                m.tiled_mma_sfb,
                m.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        def store_pair(tensor, columns, rows=QUANT_TILE):
            return cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), tensor, cute.make_layout((rows, columns), stride=(columns, 1)), (rows, columns)
            )

        def store_pair_transposed(tensor):
            # The fused epilogue's transposed Hidden: 32 rows x 128 B under the 128 B swizzle.
            return cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), tensor, mxgemm.hidden_t_smem_layout(), (mxgemm.HIDDEN_T_TILE_ROWS, QUANT_TILE)
            )

        def store_pair_context(tensor):
            # The fused epilogue's BF16 Gate / Up context: 128 rows x 32 columns under the 64 B swizzle.
            return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), tensor, mxgemm.context_smem_layout(), (QUANT_TILE, gemm.EPILOGUE_N))

        pairs = [
            sfa_pair(x_routed_sc, x_routed),
            sfa_pair(hidden_routed_sc, hidden_routed),
            sfb_pair(routed_gate_weights_sc, routed_gate_weights),
            sfb_pair(routed_up_weights_sc, routed_up_weights),
            sfb_pair(routed_down_weights_sc, routed_down_weights),
            store_pair(hidden_routed, gemm.EPILOGUE_N),  # the fused epilogue: one 32-column subtile at a time
            store_pair_transposed(hidden_routed_t),
            store_pair_context(gate_routed_bf16),
            store_pair_context(up_routed_bf16),
        ]
        # x_routed_t plain (the stream's transposed stores address it; x_routed itself is a kernel argument).
        tensors = [p[1] for p in pairs] + [x_routed_t, x_routed_sc, x_routed_sc_t, hidden_routed_sc, hidden_routed_sc_t]
        assert len(tensors) == _Mx.HIDDEN_SC_T + 1 and tensors[_Mx.X_T_ROWS] is x_routed_t
        return [p[0] for p in pairs], tensors

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        a_atoms: list[cute.CopyAtom],
        a_tensors: list[cute.Tensor],
        b_atoms: list[cute.CopyAtom],
        b_tensors: list[cute.Tensor],
        d_atoms: list[cute.CopyAtom],
        d_tensors: list[cute.Tensor],
        swiglu_load_atoms: list[cute.CopyAtom],
        swiglu_load_tensors: list[cute.Tensor],
        swiglu_store_atoms: list[cute.CopyAtom],
        swiglu_store_tensors: list[cute.Tensor],
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        d_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        swiglu_layout: cute.ComposedLayout,
        scheduler_params: utils.ClcDynamicPersistentTileSchedulerParams,
        cluster_layout_vmnk: cute.Layout,
        x_peer_table: cute.Tensor,
        combine_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens_tensor: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        x_routed: cute.Tensor,
        y_routed: cute.Tensor,
        gate_up_tile_ready: cute.Tensor,
        hidden_row_block_ready: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        shared_hidden_size: Int32,
        intermediate_size: Int32,
        num_local_experts: Int32,
        topk: Int32,
        num_local_tokens: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
        shared_tasks: Int32,
        minibatch_tasks: Int32,
        swiglu_limit: Float32,
        # Routed MXFP8: the block-scaled MMAs and layouts, and the ``_Mx`` atom / tensor lists.
        mxfp8_tiled_mma: cute.TiledMma,
        mxfp8_tiled_mma_sfb: cute.TiledMma,
        mxfp8_cluster_layout_sfb_vmnk: cute.Layout,
        mxfp8_a_layout: cute.ComposedLayout,
        mxfp8_b_layout: cute.ComposedLayout,
        mxfp8_sfa_layout: cute.Layout,
        mxfp8_sfb_layout: cute.Layout,
        mxfp8_atoms: list,
        mxfp8_tensors: list,
        # Where Up starts in the tensors passed as the routed Up weight and scale: the rows of Gate when Gate and
        # Up are one combined FC1 parameter, 0 when they are separate weights.
        up_row_offset: Int32,
        # Variant-only arguments, None in every other variant -- hence no annotation: the DSL casts an
        # annotated argument, and None is what keeps them out of the kernel's parameters.
        # routed_descriptors (split_weights): the per-expert TMA descriptors of the routed operands, one
        # array per operand in the address table's order.  situ_beta1 / situ_beta2 (is_situ): Float32.
        routed_descriptors,
        situ_beta1,
        situ_beta2,
    ):
        thread = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_index = cute.arch.make_warp_uniform(cute.arch.cluster_idx()[0])
        num_tokens = num_tokens_tensor[0]
        # x_routed's width is the routed chain's hidden (``shared_hidden_size`` the shared expert's); both are runtime.
        routed_hidden_size = Int32(x_routed.shape[1])
        comm_clusters = num_comm_sms // 2
        if cutlass.const_expr(self.shared_fc1_only):
            comm_clusters = Int32(0)  # no communication CTAs: the grid is the shared FC1 tasks' clusters
        true_clusters = Int32(comm_clusters + shared_tasks) + (_ceil_div_i32(num_tokens, Int32(minibatch_size)) * Int32(minibatch_tasks))
        # Both CTAs of a cluster share cluster_idx and the token count (a device value that does not change
        # during the launch), so they take the same branch.  Capacity-only clusters are excluded before any
        # barrier / TMEM / cluster resource operation; the live clusters request CLC items and run the
        # eight-stage capacity drain.
        if cluster_index < true_clusters:
            storage = utils.SmemAllocator().allocate(_SharedStorageMxfp8)
            arena = storage.arena.data_ptr().align(min_align=1024)
            # The routed GEMM tasks' expert lookup: the per-CTA row-block table in shared memory, built once
            # here (TaskDecode, a compile-time variant chosen by the host from the local expert count).
            s_expert_cu_end = cute.make_tensor(storage.expert_cu_end.data_ptr(), cute.make_layout(EXPERT_TABLE_CAP))
            expert_stride, expert_groups = _build_expert_cu_end(tokens_per_expert, s_expert_cu_end, self.task_decode, thread)
            if cutlass.const_expr(self.split_weights):
                # Once per CTA, before any task: the routed descriptors are built by the launch
                # that precedes this kernel and do not change while it runs, so the loads only
                # have to name one.  Every thread that may issue a routed TMA does it here, which
                # keeps an inline-asm fence out of the mainloop.
                for operand in cutlass.range_constexpr(6):
                    routed_descriptors_pkg.acquire_descriptors(
                        routed_descriptors_pkg.operand_descriptors(routed_descriptors, num_local_experts, operand), num_local_experts
                    )
            ab_full = storage.ab_full.data_ptr()
            acc_full = storage.acc_full.ptr
            swiglu_barriers = storage.swiglu_barriers.data_ptr()
            clc_drain_responses = storage.clc_drain_responses.data_ptr()
            # The A/B operand ring (gemm.AB_STAGES stages) and the accumulator hand-off (gemm.ACC_STAGES).
            ab_pipeline = PipelineTmaUmma.create(
                num_stages=gemm.AB_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.collective.num_mcast_ctas_a + self.collective.num_mcast_ctas_b - 1),
                tx_count=self.collective.num_tma_load_bytes,
                barrier_storage=ab_full,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            acc_pipeline = PipelineUmmaAsync.create(
                num_stages=gemm.ACC_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.collective.epilog_warp_id) * CLUSTER_SIZE),
                barrier_storage=acc_full,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            # The routed MXFP8 scales' own ring: 3072 B per stage onto CTA 0 (SFA x2, SFB halves x2
            # multicast).
            sf_pipeline = PipelineTmaUmma.create(
                num_stages=gemm.AB_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=self.mxfp8_collective.num_sf_load_bytes,
                barrier_storage=storage.sf_full.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            is_leader_cta = cta_rank == Int32(0)
            if thread == Int32(0):
                for stage in cutlass.range_constexpr(3):
                    cute.arch.mbarrier_init(swiglu_barriers + stage, 1)
            # CLC owns a different response/empty pair from the GEMM operand ring.
            clc_layout = cute.make_layout((1, 2, 1, 1))
            clc_pipeline = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=storage.clc_mbarriers.data_ptr(),
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 16),
                tx_count=16,
                cta_layout_vmnk=clc_layout,
                defer_sync=True,
            )
            drain_pipeline = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=storage.clc_drain_mbarriers.data_ptr(),
                num_stages=8,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2),
                tx_count=16,
                cta_layout_vmnk=clc_layout,
                defer_sync=True,
            )
            tmem = utils.TmemAllocator(
                storage.tmem_address.ptr,
                barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=mxgemm.TMEM_ALLOC_BARRIER_ID, num_threads=256),
                allocator_warp_id=0,
                is_two_cta=True,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc.ptr,
            )
            # All 512 TMEM columns: the accumulator, then the scale slots.
            tmem.allocate(mxgemm.MXFP8_TMEM_COLUMNS)
            # The aligned barrier of wait_for_alloc() is valid here: all 256 threads of the CTA reach this
            # instruction (the enclosing condition has one value per cluster), before any warp-role branch.
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            pipeline_init_arrive(cluster_shape_mn=CLUSTER_SHAPE, is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE)
            stream_barriers = storage.stream_mbarriers.data_ptr()  # taken outside the if: a plain pointer value
            if cluster_index < Int32(comm_clusters):
                self._run_comm_prefix(
                    x_peer_table,
                    combine_peer_table,
                    schedule_peer_rank,
                    schedule_peer_token_idx,
                    num_tokens_tensor,
                    y_routed,
                    x_routed_ready,
                    y_routed_ready,
                    y_routed_done,
                    stream_barriers,
                    arena,
                    cluster_index,
                    cta_rank,
                    routed_hidden_size,
                    topk,
                    macrobatch_size,
                    minibatch_size,
                    num_comm_sms,
                    [x_routed, mxfp8_tensors[_Mx.X_T_ROWS], mxfp8_tensors[_Mx.X_SC], mxfp8_tensors[_Mx.X_SC_T]],
                )

            sA = cute.make_tensor(cute.recast_ptr(arena, a_layout.inner, dtype=BFloat16), a_layout.outer)
            sB = cute.make_tensor(cute.recast_ptr(arena + gemm.A_BYTES, b_layout.inner, dtype=BFloat16), b_layout.outer)
            d_offset = mxgemm.MXFP8_D_OFFSET
            sD = cute.make_tensor(cute.recast_ptr(arena + d_offset, d_layout.inner, dtype=BFloat16), d_layout.outer)
            # Routed MXFP8: e4m3 views of the same A/B ring and the scale ring (TMA destinations).
            sA_fp8 = cute.make_tensor(cute.recast_ptr(arena, mxfp8_a_layout.inner, dtype=Float8E4M3FN), mxfp8_a_layout.outer)
            sB_fp8 = cute.make_tensor(cute.recast_ptr(arena + gemm.A_BYTES, mxfp8_b_layout.inner, dtype=Float8E4M3FN), mxfp8_b_layout.outer)
            sSFA = cute.make_tensor(cute.recast_ptr(arena + gemm.A_BYTES + gemm.B_BYTES, dtype=Float8E8M0FNU), mxfp8_sfa_layout)
            sSFB = cute.make_tensor(cute.recast_ptr(arena + gemm.A_BYTES + gemm.B_BYTES + mxgemm.SFA_BYTES, dtype=Float8E8M0FNU), mxfp8_sfb_layout)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            fake = tiled_mma.make_fragment_C(cute.append(tiled_mma.partition_shape_C(self.collective.mma_tiler[:2]), 1))
            acc_tensor = cute.make_tensor(acc_ptr, fake.layout)[None, None, None, 0]
            scheduler = utils.ClcDynamicPersistentTileScheduler.create(
                scheduler_params, cute.arch.block_idx(), cute.arch.grid_dim(), storage.clc_response.data_ptr()
            )
            work = scheduler.initial_work_tile_info()
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            logical_cluster = work.tile_idx[0] // Int32(2)
            active = Int32(0)
            if cluster_index >= Int32(comm_clusters) and cluster_index < true_clusters:
                active = Int32(1)
            mma_op = tiled_mma
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, gemm.AB_STAGES)
            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, gemm.AB_STAGES)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, gemm.ACC_STAGES)
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, gemm.ACC_STAGES)
            # Routed tasks: e4m3 operands and K tile 128; the scale ring's states.
            routed_mma, routed_sA, routed_sB = mxfp8_tiled_mma, sA_fp8, sB_fp8
            routed_k_tile = mxgemm.FP8_K_TILE
            sf_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, gemm.AB_STAGES)
            sf_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, gemm.AB_STAGES)
            swiglu_phases = Int32(0)
            # The task-table constants, once per CTA; the decode itself then costs two runtime divisions
            # per task.
            tc_shared_gate, tc_shared_swiglu, tc_shared_total, tc_mini_gate_up, tc_mini_total, tc_macro_minis, tc_num_macros, tc_last_minis = self._task_counts(
                num_tokens, shared_hidden_size, routed_hidden_size, intermediate_size, num_local_tokens, macrobatch_size, minibatch_size
            )
            d_stages = arena + d_offset
            while active != Int32(0):
                kind, task, macro, mini = self._decode_task_hoisted(
                    logical_cluster - Int32(comm_clusters),
                    tc_shared_gate,
                    tc_shared_swiglu,
                    tc_shared_total,
                    tc_mini_gate_up,
                    tc_mini_total,
                    tc_macro_minis,
                    tc_num_macros,
                    tc_last_minis,
                )
                # Only CTA0/W5 requests the next CLC item, before current task work.
                if warp == Int32(5) and cta_rank == Int32(0):
                    clc_pipeline.producer_acquire(producer_state)
                    scheduler.advance_to_next_work(clc_pipeline.producer_get_barrier(producer_state))
                    producer_state.advance()
                if kind == Int32(SHARED_SWIGLU):
                    swiglu_phases = self._run_swiglu(
                        task,
                        cta_rank,
                        swiglu_load_atoms,
                        swiglu_load_tensors,
                        swiglu_store_atoms,
                        swiglu_store_tensors,
                        swiglu_layout,
                        arena,
                        swiglu_barriers,
                        swiglu_phases,
                        gate_up_tile_ready,
                        hidden_row_block_ready,
                        intermediate_size,
                        num_local_tokens,
                        num_tokens,
                        macrobatch_size,
                        swiglu_limit,
                        situ_beta1,
                        situ_beta2,
                    )
                else:
                    row, column, expert = self._decode_gemm_task(
                        kind,
                        task,
                        macro,
                        mini,
                        tokens_per_expert,
                        s_expert_cu_end,
                        expert_stride,
                        expert_groups,
                        shared_hidden_size,
                        routed_hidden_size,
                        intermediate_size,
                        num_local_tokens,
                        macrobatch_size,
                        minibatch_size,
                    )
                    k_tiles = shared_hidden_size // Int32(64)
                    if kind == Int32(ROUTED_GATE_UP):
                        k_tiles = routed_hidden_size // Int32(64)
                    if kind == Int32(SHARED_DOWN) or kind == Int32(ROUTED_DOWN):
                        k_tiles = intermediate_size // Int32(64)
                    if kind >= Int32(ROUTED_GATE_UP):
                        k_tiles = k_tiles // Int32(mxgemm.FP8_K_TILE // gemm.K_TILE)  # the e4m3 K tile
                    if expert >= Int32(0):
                        if warp == Int32(7):
                            if kind == Int32(SHARED_DOWN) or kind == Int32(ROUTED_DOWN):
                                ready_row = row
                                if kind == Int32(ROUTED_DOWN):
                                    ready_row = Int32(num_local_tokens // 256) + macro * Int32(macrobatch_size // 256) + row
                                with cute.arch.elect_one():
                                    _counter_wait_gpu(hidden_row_block_ready, ready_row, Int32(2) * (intermediate_size // Int32(128)))
                            elif kind >= Int32(ROUTED_GATE_UP):
                                generation = macro * Int32(macrobatch_size // minibatch_size) + mini
                                mini_rows = cutlass.min(Int32(minibatch_size), num_tokens - generation * Int32(minibatch_size))
                                # One x_routed_ready arrival per quantized stream block of the minibatch.
                                required = _ceil_div_i32(mini_rows, Int32(STREAM_ROWS)) * _ceil_div_i32(routed_hidden_size, Int32(STREAM_COLUMNS))
                                with cute.arch.elect_one():
                                    _counter_wait_gpu(x_routed_ready, generation, required)
                            cute.arch.sync_warp()
                            if kind == Int32(SHARED_GATE):
                                ab_producer_state = self._load_gemm(
                                    a_atoms[0],
                                    a_tensors[0],
                                    b_atoms[0],
                                    b_tensors[0],
                                    tiled_mma,
                                    sA,
                                    sB,
                                    row,
                                    column,
                                    cta_rank,
                                    k_tiles,
                                    ab_pipeline,
                                    ab_producer_state,
                                )
                            elif kind == Int32(SHARED_UP):
                                ab_producer_state = self._load_gemm(
                                    a_atoms[0],
                                    a_tensors[0],
                                    b_atoms[1],
                                    b_tensors[1],
                                    tiled_mma,
                                    sA,
                                    sB,
                                    row,
                                    column,
                                    cta_rank,
                                    k_tiles,
                                    ab_pipeline,
                                    ab_producer_state,
                                )
                            elif kind == Int32(SHARED_DOWN):
                                ab_producer_state = self._load_gemm(
                                    a_atoms[1],
                                    a_tensors[1],
                                    b_atoms[2],
                                    b_tensors[2],
                                    tiled_mma,
                                    sA,
                                    sB,
                                    row,
                                    column,
                                    cta_rank,
                                    k_tiles,
                                    ab_pipeline,
                                    ab_producer_state,
                                )
                            elif kind == Int32(ROUTED_GATE_UP):
                                # Fused FC1: CTA 0 loads the Gate weights, CTA 1 the Up weights.
                                if cta_rank == Int32(0):
                                    b_view, b_desc = self._routed_operand(b_tensors[3], routed_descriptors, num_local_experts, 0, expert)
                                    ab_producer_state = self._load_gate_up(
                                        a_atoms[2],
                                        a_tensors[2],
                                        b_atoms[3],
                                        b_view,
                                        routed_mma,
                                        routed_sA,
                                        routed_sB,
                                        row,
                                        column,
                                        cta_rank,
                                        k_tiles,
                                        ab_pipeline,
                                        ab_producer_state,
                                        routed_k_tile,
                                        b_desc_ptr=b_desc,
                                    )
                                else:
                                    b_view, b_desc = self._routed_operand(b_tensors[4], routed_descriptors, num_local_experts, 1, expert)
                                    # Up is up_row_offset rows into a combined parameter (0: separate weights).
                                    b_view = cute.domain_offset((up_row_offset, Int32(0)), b_view)
                                    ab_producer_state = self._load_gate_up(
                                        a_atoms[2],
                                        a_tensors[2],
                                        b_atoms[4],
                                        b_view,
                                        routed_mma,
                                        routed_sA,
                                        routed_sB,
                                        row,
                                        column,
                                        cta_rank,
                                        k_tiles,
                                        ab_pipeline,
                                        ab_producer_state,
                                        routed_k_tile,
                                        b_desc_ptr=b_desc,
                                    )
                            elif kind == Int32(ROUTED_DOWN):
                                b_view, b_desc = self._routed_operand(b_tensors[5], routed_descriptors, num_local_experts, 2, expert)
                                ab_producer_state = self._load_gemm(
                                    a_atoms[3],
                                    a_tensors[3],
                                    b_atoms[5],
                                    b_view,
                                    routed_mma,
                                    routed_sA,
                                    routed_sB,
                                    row,
                                    column,
                                    cta_rank,
                                    k_tiles,
                                    ab_pipeline,
                                    ab_producer_state,
                                    routed_k_tile,
                                    b_desc_ptr=b_desc,
                                )
                        elif warp == Int32(6):
                            # The routed scales, after the same operand-ready wait as the A/B loads.
                            if kind >= Int32(ROUTED_GATE_UP):
                                sf_producer_state = self._load_routed_scales(
                                    kind,
                                    row,
                                    column,
                                    expert,
                                    cta_rank,
                                    macro,
                                    mini,
                                    num_tokens,
                                    x_routed_ready,
                                    hidden_row_block_ready,
                                    routed_hidden_size,
                                    intermediate_size,
                                    num_local_tokens,
                                    macrobatch_size,
                                    minibatch_size,
                                    k_tiles,
                                    mxfp8_tiled_mma,
                                    mxfp8_tiled_mma_sfb,
                                    cluster_layout_vmnk,
                                    mxfp8_cluster_layout_sfb_vmnk,
                                    mxfp8_atoms,
                                    mxfp8_tensors,
                                    sSFA,
                                    sSFB,
                                    sf_pipeline,
                                    sf_producer_state,
                                    num_local_experts,
                                    up_row_offset,
                                    routed_descriptors,
                                )
                        elif warp == Int32(4) and cta_rank == Int32(0):
                            if kind >= Int32(ROUTED_GATE_UP):
                                # The block-scaled MMAs read the e4m3 rings and scale rings in
                                # ``arena`` by descriptor (issued by one lane, no atoms).
                                ab_consumer_state, sf_consumer_state, acc_producer_state = mxgemm.mma_blockscaled(
                                    ab_pipeline, sf_pipeline, acc_pipeline, ab_consumer_state, sf_consumer_state, acc_producer_state, arena, acc_ptr, k_tiles
                                )
                            else:
                                ab_consumer_state, acc_producer_state, mma_op = self.collective.mma(
                                    ab_pipeline,
                                    acc_pipeline,
                                    ab_consumer_state,
                                    acc_producer_state,
                                    mma_op,
                                    tCrA,
                                    tCrB,
                                    acc_tensor,
                                    k_tiles,
                                    is_leader_cta,
                                    cta_rank,
                                )
                        elif warp < Int32(4):
                            if kind == Int32(SHARED_GATE):
                                acc_consumer_state = self._epilogue_gemm(
                                    d_atoms[0],
                                    d_tensors[0],
                                    tiled_mma,
                                    epi_tile,
                                    sD,
                                    acc_ptr,
                                    acc_pipeline,
                                    acc_consumer_state,
                                    thread,
                                    cta_rank,
                                    row,
                                    column,
                                    kind,
                                    macro,
                                    mini,
                                    num_tokens,
                                    gate_up_tile_ready,
                                    y_routed_ready,
                                    y_routed_done,
                                    shared_hidden_size,
                                    intermediate_size,
                                    macrobatch_size,
                                    minibatch_size,
                                )
                            elif kind == Int32(SHARED_UP):
                                acc_consumer_state = self._epilogue_gemm(
                                    d_atoms[1],
                                    d_tensors[1],
                                    tiled_mma,
                                    epi_tile,
                                    sD,
                                    acc_ptr,
                                    acc_pipeline,
                                    acc_consumer_state,
                                    thread,
                                    cta_rank,
                                    row,
                                    column,
                                    kind,
                                    macro,
                                    mini,
                                    num_tokens,
                                    gate_up_tile_ready,
                                    y_routed_ready,
                                    y_routed_done,
                                    shared_hidden_size,
                                    intermediate_size,
                                    macrobatch_size,
                                    minibatch_size,
                                )
                            elif kind == Int32(SHARED_DOWN):
                                acc_consumer_state = self._epilogue_gemm(
                                    d_atoms[2],
                                    d_tensors[2],
                                    tiled_mma,
                                    epi_tile,
                                    sD,
                                    acc_ptr,
                                    acc_pipeline,
                                    acc_consumer_state,
                                    thread,
                                    cta_rank,
                                    row,
                                    column,
                                    kind,
                                    macro,
                                    mini,
                                    num_tokens,
                                    gate_up_tile_ready,
                                    y_routed_ready,
                                    y_routed_done,
                                    shared_hidden_size,
                                    intermediate_size,
                                    macrobatch_size,
                                    minibatch_size,
                                )
                            elif kind == Int32(ROUTED_GATE_UP):
                                acc_consumer_state = self._epilogue_gate_up_fused(
                                    tiled_mma,
                                    epi_tile,
                                    acc_ptr,
                                    acc_pipeline,
                                    acc_consumer_state,
                                    thread,
                                    cta_rank,
                                    row,
                                    column,
                                    macro,
                                    hidden_row_block_ready,
                                    intermediate_size,
                                    num_local_tokens,
                                    macrobatch_size,
                                    swiglu_limit,
                                    mxfp8_atoms,
                                    mxfp8_tensors,
                                    d_stages,
                                    situ_beta1,
                                    situ_beta2,
                                )
                            elif kind == Int32(ROUTED_DOWN):
                                acc_consumer_state = self._epilogue_gemm(
                                    d_atoms[3],
                                    d_tensors[3],
                                    tiled_mma,
                                    epi_tile,
                                    sD,
                                    acc_ptr,
                                    acc_pipeline,
                                    acc_consumer_state,
                                    thread,
                                    cta_rank,
                                    row,
                                    column,
                                    kind,
                                    macro,
                                    mini,
                                    num_tokens,
                                    gate_up_tile_ready,
                                    y_routed_ready,
                                    y_routed_done,
                                    routed_hidden_size,
                                    intermediate_size,
                                    macrobatch_size,
                                    minibatch_size,
                                )
                clc_pipeline.consumer_wait(consumer_state)
                # get_current_work() ends with the scheduler's own fence.proxy.async.shared::cta (CuTe DSL 4.8,
                # insert_fence, on unless create() turns it off): every lane's read of the response is ordered
                # before the next CLC write into it, which the release below admits.  ``work`` is used after the
                # release, so that fence must stay; _drain_clc_capacity_tail issues the same fence by hand.
                work = scheduler.get_current_work()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    clc_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
                next_cluster = work.tile_idx[0] // Int32(2)
                next_active = Int32(0)
                next_is_swiglu = Int32(0)
                if work.is_valid_tile:
                    if next_cluster < true_clusters:
                        next_active = Int32(1)
                        # Whether the next item is a CTA-local SwiGLU task: two compares, not a second decode.
                        next_is_swiglu = self._task_is_swiglu(next_cluster - Int32(comm_clusters), tc_shared_gate, tc_shared_swiglu, tc_shared_total)
                if kind == Int32(SHARED_SWIGLU) and next_active != Int32(0):
                    if next_is_swiglu == Int32(0):
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()
                logical_cluster = next_cluster
                active = next_active

            if cluster_index >= Int32(comm_clusters) and cluster_index < true_clusters:
                drain_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 8)
                drain_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, 8)
                lane_stage = Int32(0)
                while lane_stage < warp:
                    drain_consumer.advance()
                    drain_producer.advance()
                    lane_stage = lane_stage + Int32(1)
                response = (clc_drain_responses + warp * Int32(4)).align(min_align=16)
                drain_scheduler = utils.ClcDynamicPersistentTileScheduler.create(scheduler_params, cute.arch.block_idx(), cute.arch.grid_dim(), response)
                needs_drain = Int32(0)
                if work.is_valid_tile:
                    needs_drain = Int32(1)
                self._drain_clc_capacity_tail(needs_drain, drain_pipeline, drain_consumer, drain_producer, drain_scheduler, cta_rank)
            tmem.relinquish_alloc_permit()
            tmem.free(acc_ptr)


class _PersistentMxfp8FwdMegaE4m3Context(_PersistentMxfp8FwdMega):
    """``gate_up_context="mxfp8"``: the routed Gate / Up context kept as e4m3 with its scale tiles
    (thirteen outputs; ``mxgemm.store_acc_fused_swiglu_e4m3_context`` writes it).  Only the context's
    parameters, its ``_MxE4m3Context`` positions and the fused FC1 epilogue differ from the BF16-context
    kernel; the rest of the kernel is shared."""

    @cute.jit
    def __call__(
        self,
        x_peer_table: cute.Tensor,
        combine_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        num_tokens: cute.Tensor,
        tokens_per_expert: cute.Tensor,
        x_shared: cute.Tensor,
        x_routed: cute.Tensor,
        shared_gate_weights: cute.Tensor,
        routed_gate_weights: cute.Tensor,
        shared_up_weights: cute.Tensor,
        routed_up_weights: cute.Tensor,
        shared_down_weights: cute.Tensor,
        routed_down_weights: cute.Tensor,
        gate_shared: cute.Tensor,
        up_shared: cute.Tensor,
        hidden_shared: cute.Tensor,
        hidden_routed: cute.Tensor,
        y_shared: cute.Tensor,
        y_routed: cute.Tensor,
        gate_up_tile_ready: cute.Tensor,
        hidden_row_block_ready: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        routed_storage_tables: cute.Tensor,
        routed_descriptors: cute.Tensor,  # split weights, as in the base kernel
        build_descriptors: Int32,  # 1: this launch builds the split descriptors
        routed_gate_weights_sc: cute.Tensor,
        routed_up_weights_sc: cute.Tensor,
        routed_down_weights_sc: cute.Tensor,
        x_routed_sc: cute.Tensor,
        x_routed_t: cute.Tensor,
        x_routed_sc_t: cute.Tensor,
        gate_routed_fp8: cute.Tensor,
        gate_routed_sc: cute.Tensor,
        up_routed_fp8: cute.Tensor,
        up_routed_sc: cute.Tensor,
        hidden_routed_sc: cute.Tensor,
        hidden_routed_t: cute.Tensor,
        hidden_routed_sc_t: cute.Tensor,
        intermediate_size: Int32,
        num_local_experts: Int32,
        topk: Int32,
        num_local_tokens: Int32,
        schedule_capacity: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        num_comm_sms: Int32,
        swiglu_limit: Float32,
        up_row_offset: Int32,  # a combined FC1's Up row offset, as in the base kernel
        situ_beta1: Float32,
        situ_beta2: Float32,  # SiTU-GLU's betas, as in the base kernel
        stream: cuda.CUstream,
    ):
        self._launch(
            x_peer_table,
            combine_peer_table,
            schedule_peer_rank,
            schedule_peer_token_idx,
            num_tokens,
            tokens_per_expert,
            x_shared,
            x_routed,
            shared_gate_weights,
            routed_gate_weights,
            shared_up_weights,
            routed_up_weights,
            shared_down_weights,
            routed_down_weights,
            gate_shared,
            up_shared,
            hidden_shared,
            hidden_routed,
            y_shared,
            y_routed,
            gate_up_tile_ready,
            hidden_row_block_ready,
            x_routed_ready,
            y_routed_ready,
            y_routed_done,
            intermediate_size,
            num_local_experts,
            topk,
            num_local_tokens,
            schedule_capacity,
            macrobatch_size,
            minibatch_size,
            num_comm_sms,
            swiglu_limit,
            stream,
            routed_gate_weights_sc,
            routed_up_weights_sc,
            routed_down_weights_sc,
            x_routed_sc,
            x_routed_t,
            x_routed_sc_t,
            (gate_routed_fp8, gate_routed_sc, up_routed_fp8, up_routed_sc),
            hidden_routed_sc,
            hidden_routed_t,
            hidden_routed_sc_t,
            up_row_offset,
            routed_storage_tables,
            routed_descriptors,
            situ_beta1,
            situ_beta2,
            build_descriptors,
        )

    def _mxfp8_atoms(
        self,
        x_routed,
        hidden_routed,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        routed_gate_weights_sc,
        routed_up_weights_sc,
        routed_down_weights_sc,
        x_routed_sc,
        x_routed_t,
        x_routed_sc_t,
        gate_routed_fp8,
        gate_routed_sc,
        up_routed_fp8,
        up_routed_sc,
        hidden_routed_sc,
        hidden_routed_t,
        hidden_routed_sc_t,
    ):
        """The ``_Mx`` lists: scale-load atoms over the scale tiles, e4m3 tile-store atoms."""
        m = self.mxfp8_collective
        sfa_layout = cute.slice_(m.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_layout = cute.slice_(m.sfb_smem_layout_staged, (None, None, None, 0))
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(m.cluster_shape_mnk, m.tiled_mma.thr_id)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(m.cluster_shape_mnk, m.tiled_mma.thr_id)

        def sfa_pair(scales, operand):
            return cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                mxgemm.scale_operand(scales, operand.shape),
                sfa_layout,
                m.mma_tiler,
                m.tiled_mma,
                m.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        def sfb_pair(scales, operand):
            return cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                mxgemm.scale_operand(scales, operand.shape),
                sfb_layout,
                m.mma_tiler_sfb,
                m.tiled_mma_sfb,
                m.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        def store_pair(tensor, columns, rows=QUANT_TILE):
            return cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), tensor, cute.make_layout((rows, columns), stride=(columns, 1)), (rows, columns)
            )

        def store_pair_transposed(tensor):
            # The fused epilogue's transposed Hidden: 32 rows x 128 B under the 128 B swizzle.
            return cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), tensor, mxgemm.hidden_t_smem_layout(), (mxgemm.HIDDEN_T_TILE_ROWS, QUANT_TILE)
            )

        pairs = [
            sfa_pair(x_routed_sc, x_routed),
            sfa_pair(hidden_routed_sc, hidden_routed),
            sfb_pair(routed_gate_weights_sc, routed_gate_weights),
            sfb_pair(routed_up_weights_sc, routed_up_weights),
            sfb_pair(routed_down_weights_sc, routed_down_weights),
            store_pair(gate_routed_fp8, gemm.EPILOGUE_N),
            store_pair(up_routed_fp8, gemm.EPILOGUE_N),
            store_pair(hidden_routed, gemm.EPILOGUE_N),  # the fused epilogue: one 32-column subtile at a time
            store_pair_transposed(hidden_routed_t),
        ]
        tensors = [p[1] for p in pairs] + [  # x_routed_t plain, then the scales (see _MxE4m3Context)
            x_routed_t,
            x_routed_sc,
            x_routed_sc_t,
            gate_routed_sc,
            up_routed_sc,
            hidden_routed_sc,
            hidden_routed_sc_t,
        ]
        assert len(tensors) == _MxE4m3Context.HIDDEN_SC_T + 1 and tensors[_MxE4m3Context.X_T_ROWS] is x_routed_t
        return [p[0] for p in pairs], tensors

    @cute.jit
    def _epilogue_gate_up_fused(
        self,
        tiled_mma,
        epi_tile,
        acc_ptr,
        acc_pipeline,
        acc_consumer_state,
        thread: Int32,
        cta_rank: Int32,
        row: Int32,
        column: Int32,
        macro: Int32,
        hidden_row_block_ready,
        intermediate_size: Int32,
        num_local_tokens: Int32,
        macro_size: Int32,
        swiglu_limit: Float32,
        atoms: list,
        tensors: list,
        d_stages: cute.Pointer,
        situ_beta1=None,
        situ_beta2=None,
    ):
        """The fused routed FC1 epilogue (``mxgemm.store_acc_fused_swiglu_e4m3_context``): the Gate / Up e4m3
        context, Hidden quantized in both layouts, the four scale tiles; then ``hidden_row_block_ready`` of the
        256-row block, once per CTA (``2 x (intermediate / 128)`` arrivals per block, the Down's
        expectation).  ``situ_beta1`` / ``situ_beta2``: Float32 in the is_situ variant, None otherwise."""
        collective = self.collective
        fake = tiled_mma.make_fragment_C(cute.append(tiled_mma.partition_shape_C(collective.mma_tiler[:2]), 1))
        acc_tensor = cute.make_tensor(acc_ptr, fake.layout)
        t2r, acc_partition, r_acc = collective.epilog_tmem_copy_and_partition(thread, acc_tensor, epi_tile, True)
        acc_partition = cute.group_modes(acc_partition[None, None, None, None, None, 0], 3, 5)
        acc_consumer_state = mxgemm.store_acc_fused_swiglu_e4m3_context(
            collective,
            t2r,
            acc_partition,
            r_acc,
            atoms[_MxE4m3Context.GATE_FP8],
            tensors[_MxE4m3Context.GATE_FP8],
            tensors[_MxE4m3Context.GATE_SC].iterator,
            atoms[_MxE4m3Context.UP_FP8],
            tensors[_MxE4m3Context.UP_FP8],
            tensors[_MxE4m3Context.UP_SC].iterator,
            atoms[_MxE4m3Context.HIDDEN_FP8],
            tensors[_MxE4m3Context.HIDDEN_FP8],
            tensors[_MxE4m3Context.HIDDEN_SC].iterator,
            atoms[_MxE4m3Context.HIDDEN_T],
            tensors[_MxE4m3Context.HIDDEN_T],
            tensors[_MxE4m3Context.HIDDEN_SC_T].iterator,
            intermediate_size // Int32(QUANT_TILE),
            macro_size // Int32(QUANT_TILE),
            d_stages,
            row * Int32(2) + cta_rank,
            column,
            self.is_clamped,
            swiglu_limit,
            acc_pipeline,
            acc_consumer_state,
            thread,
            self.is_situ,
            situ_beta1,
            situ_beta2,
        )
        gemm.finish_store(thread)
        if thread == Int32(0):
            _counter_arrive_release_gpu(hidden_row_block_ready, Int32(num_local_tokens // 256) + macro * Int32(macro_size // 256) + row)
        return acc_consumer_state


class _StateWithoutRoutedRing:
    """``PersistentMxfp8State`` as the BF16 builder reads it: the fused FC1 has no BF16 Gate / Up ring, so
    ``gate_routed`` / ``up_routed`` (which ``make_mega_args`` views but this kernel never takes) stand in
    for another ring; ``make_mega_args_mxfp8`` drops the two entries again.  This relies on the BF16
    builder naming its ring arguments ``gate_routed`` / ``up_routed``; ``make_mega_args_mxfp8`` asserts
    that both were present."""

    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        if name in ("gate_routed", "up_routed"):
            return self._state.hidden_routed
        return getattr(self._state, name)


def make_mega_args_mxfp8(
    *args,
    routed_weight_scales,
    up_row_offset: int = 0,
    situ_beta=(0.0, 0.0),
    routed_storage_tables=None,
    routed_descriptors=None,
    build_descriptors: int = 1,
    launch: bool = False,
    **kwargs,
):
    """``make_mega_args`` for ``_PersistentMxfp8FwdMega``: the BF16 builder's arguments (the routed
    weights, ``state.x_routed`` and ``state.hidden_routed`` are e4m3; the BF16 Gate / Up ring entries
    dropped, the fused FC1 writes the Gate / Up context instead) with the scale tiles, the transposed
    outputs and the Gate / Up context of ``state`` (BF16, or e4m3 / scale pairs) inserted before the
    scalars, in ``__call__``'s parameter order.  One builder wrapping the BF16 one, whose conventions
    (leading dims, peer-table cache) reach this kernel unchanged.

    ``launch=False`` builds the CuTe tensors ``cute.compile`` traces; ``launch=True`` the arguments
    of a call to the compiled TVM FFI function (the torch views, plain scalars and the ``CUstream``):
    one builder, the same path for the first and every later call.
    """

    def dynamic_tensor(tensor):
        return tensor if launch else dynamic_layout_tensor(tensor)

    state = args[-1]
    # The stream's quantization writes these four by raw pointer: the normal e4m3 rows at a row stride of the routed
    # hidden (x_routed's width), the transposed rows at a stride of the macrobatch, the scale tiles as 512 B tiles.
    for name in ("x_routed", "x_routed_t", "x_routed_sc", "x_routed_sc_t"):
        assert getattr(state, name).is_contiguous(), f"the stream writes {name} as a contiguous tile"
    gate_scale, up_scale, down_scale = routed_weight_scales
    extras = dict(
        routed_gate_weights_sc=dynamic_tensor(row_major_view(gate_scale)),
        routed_up_weights_sc=dynamic_tensor(row_major_view(up_scale)),
        routed_down_weights_sc=dynamic_tensor(row_major_view(down_scale)),
        x_routed_sc=dynamic_tensor(row_major_view(state.x_routed_sc)),
        x_routed_t=dynamic_tensor(row_major_view(state.x_routed_t)),
        x_routed_sc_t=dynamic_tensor(row_major_view(state.x_routed_sc_t)),
        **{name: dynamic_tensor(row_major_view(getattr(state, name))) for name in _context_names(state)},
        hidden_routed_sc=dynamic_tensor(row_major_view(state.hidden_routed_sc)),
        hidden_routed_t=dynamic_tensor(row_major_view(state.hidden_routed_t)),
        hidden_routed_sc_t=dynamic_tensor(row_major_view(state.hidden_routed_sc_t)),
    )
    scalar = (lambda value: value) if launch else Int32
    arguments = {}
    base = make_mega_args(*args[:-1], _StateWithoutRoutedRing(state), launch=launch, **kwargs)
    assert "gate_routed" in base and "up_routed" in base, "the BF16 builder renamed its Gate / Up ring arguments"
    # The BF16 builder's hidden_size scalar is only the insertion point of the MXFP8 extras (the first scalar):
    # this kernel has no hidden_size argument, it reads the two hiddens from x_shared / x_routed as the BF16 forward.
    assert "hidden_size" in base, "the BF16 builder renamed its hidden_size scalar"
    for name, value in base.items():
        if name in ("gate_routed", "up_routed"):
            continue
        if name == "hidden_size":
            arguments.update(extras)
            continue
        if name == "intermediate_size":
            # The BF16 builder reads I off the shared Gate weight, which a routed-only layer does
            # not have; it is read off the routed Hidden here, as the state is sized.
            value = scalar(int(state.hidden_routed.shape[1]))
        arguments[name] = value
        if name == "y_routed_done":
            # The split routed weights' address table and descriptor bytes, inserted here rather
            # than carried by the BF16 builder.
            # dynamic_layout_tensor picks the unit-stride dimension itself: the experts for
            # the (operands, E) table, the only one for the descriptor bytes.
            arguments["routed_storage_tables"] = dynamic_tensor(_routed_tables_or_placeholder(routed_storage_tables, state.y_routed_done.device))
            arguments["routed_descriptors"] = dynamic_tensor(_routed_descriptors_or_placeholder(routed_descriptors, state.y_routed_done.device))
            arguments["build_descriptors"] = scalar(build_descriptors)
        if name == "swiglu_limit":
            # Immediately after the last scalar of the BF16 ABI, as __call__ declares them.
            arguments["up_row_offset"] = scalar(up_row_offset)
            arguments["situ_beta1"] = situ_beta[0] if launch else Float32(situ_beta[0])
            arguments["situ_beta2"] = situ_beta[1] if launch else Float32(situ_beta[1])
    return arguments


def _context_names(state) -> tuple[str, ...]:
    """The state's routed Gate / Up context buffers, in ``__call__``'s order: BF16 tensors, or
    e4m3 / scale pairs (``PersistentMxfp8StateE4m3Context``)."""
    if hasattr(state, "gate_routed_fp8"):
        return ("gate_routed_fp8", "gate_routed_sc", "up_routed_fp8", "up_routed_sc")
    return ("gate_routed_bf16", "up_routed_bf16")


def mega_class(gate_up_context: str = "mxfp8") -> type:
    """The kernel class of a Gate / Up context precision (``"mxfp8"``, the default, or ``"bf16"``)."""
    return {"bf16": _PersistentMxfp8FwdMega, "mxfp8": _PersistentMxfp8FwdMegaE4m3Context}[gate_up_context]


def prepare_mega_mxfp8(
    *args,
    is_clamped: bool,
    task_decode: TaskDecode,
    gate_up_context: str = "mxfp8",
    is_situ: bool = False,
    split_weights: bool = False,
    has_shared: bool = True,
    shared_fc1_only: bool = False,
    has_tail: bool = False,
    **kwargs,
):
    """Compile the (clamp mode, task decode, Gate / Up context, routed weight layout, activation, shared
    expert, shared FC1 alone, tail tiles) variant once; the executor's positional ABI is ``__call__``'s order."""

    cute_kwargs = make_mega_args_mxfp8(*args, **kwargs)
    kernel_class = mega_class(gate_up_context)
    parameters = tuple(inspect.signature(kernel_class.__call__).parameters)[1:]
    if tuple(cute_kwargs) != parameters:
        raise RuntimeError("make_mega_args_mxfp8 keys must match __call__ parameters in order")
    return cute.compile(
        kernel_class(
            is_clamped=is_clamped,
            task_decode=task_decode,
            split_weights=split_weights,
            is_situ=is_situ,
            has_shared=has_shared,
            shared_fc1_only=shared_fc1_only,
            has_tail=has_tail,
        ),
        **cute_kwargs,
        options=COMPILE_OPTIONS,
    )


__all__ = [
    "make_mega_args_mxfp8",
    "mega_class",
    "prepare_mega_mxfp8",
]
