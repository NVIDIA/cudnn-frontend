# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM100 (Blackwell) SDPA backward, d = 128 (and the d64 recompute flavour), bf16 / fp16: the whole family in one module.

Six main kernels, one per cell of the d128 backward support matrix, each with its 14-section header kept verbatim as the docstring of
its section below (``bprop_<dim>_<dtype>.py`` is the kernels-dir naming grammar; the arch is the only directory axis).  The pointer hosts
(one ``compile_host_*`` per route; the compiled-plan cache keys and the tvm-ffi symbols are ABI) live in ``prepared_d128_host.py``.

| section | main kernel | cell (adapter implementation) | config module |
|---|---|---|---|
| 1 shared | ``dot_do_o_kernel``, ``cast_dq_d128_kernel`` | the dense accumulation chain's delta / dQ-zero and dQ-cast launches | -- |
| 2 recompute | ``bprop_d128_kernel`` | dense BSHD, deterministic, one launch, no workspace (``recompute``; d64 by template) | ``config_sm100_d128`` (``CFG_RECOMPUTE``) |
| 3 accumulation | ``bprop_d128_accum_kernel`` | dense BSHD MHA, fp32 dQ workspace, 3 launches (``accumulation``) | ``config_sm100_d128_accum`` (``CFG_ACCUM``, templated) |
| 4 THD chain | ``setup_thd_d128_kernel``, ``delta_zero_thd_d128_kernel``, ``cast_dq_thd_d128_kernel`` | the packed THD chains' side launches | ``config_sm100_d128_accum`` (``CFG_ACCUM_D128``) |
| 5 THD | ``bprop_d128_accum_thd_kernel`` | packed THD MHA (``thd``) | ``config_sm100_d128_accum`` (``CFG_ACCUM_D128``) |
| 6 grouped chain | ``host_grouped_dense`` / ``host_grouped_thd`` | the grouped (GQA) chains: dense 4 launches, THD 5 launches | -- |
| 7 grouped | ``bprop_d128_accum_grouped_dense_kernel``, ``bprop_d128_accum_grouped_thd_kernel`` | GQA, one Q-head owner, native dK / dV partials (``accumulation_grouped_*``) | ``config_sm100_d128_accum`` |
| 8 pair | ``bprop_d128_accum_grouped_2cta_stail_kernel`` | GQA THD / dense twin on the cta_group::2 datapath (``accumulation_grouped_*_2cta``) | ``config_sm100_d128_accum_2cta_stail`` (``CFG``, ``BAR_LAYOUT``) |
| 9 pair chain | ``setup_thd_d128_2cta_kernel``, ``delta_zero_stats_thd_d128_2cta_kernel`` | the pair chain's side launches (the eleven-map setup, the pre-scaled stats delta) | ``config_sm100_d128_accum_2cta`` (``CFG_PAIR_BASE``) |

Every kernel function keeps its name (the PTX symbol is ``cudnn`` + the function name: the rendering records are per kernel) and its
``set_name_prefix`` call.  Each body keeps its own text.  The PAIR body (section 8, the newest and largest; the suites' source pins read
it) owns every bare name: ``CFG``, ``BAR_LAYOUT``, ``Bars``, ``MAP_*``, ``_wait``, ``_softmax_loop``, ...  Where another body defined the
same name it carries that body's suffix: ``_softmax_loop_accum`` / ``_softmax_loop_thd``, ``BarsRecompute`` / ``BarsAccum`` / ``BarsThd``,
``CFG_RECOMPUTE`` / ``CFG_ACCUM`` (templated) / ``CFG_ACCUM_D128`` (the THD and grouped chains) / ``CFG_PAIR_BASE`` (the pair chain kernels),
``MAP_*_THD`` (the THD body's seven tensor-map slots), ``host_recompute`` / ``host_accum`` / ``host_accum_thd``.  A helper whose text is
identical in every body and binds no body-specific global is defined once (``_view``, ``_commit``, ``_tmem_load``, ``_tmem_store``,
``_map_ptr``, ``LOG2E``).  The pair body precedes its chain kernels: the chain asserts against the body's constants at import.  The module
is template-loaded for the d64 recompute / accumulation flavours (``FROST_TEMPLATE_PARAMS``: a head dimension, or
``(head_dim, dq_out, ratio, split)`` for the deterministic pair's dK / dV rendering of the accumulation body), so every import is absolute.
"""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, Scope, advance, cga_arrive, cga_wait, named_barrier_fence
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mask import MASK_WORD_COLS, apply_mask_words, band_mask_words
from cudnn.frost.tile_dsl.mma import desc_opaque, mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, ffma2, fmul2, fp32_to_fp16, opaque_i32, row_reduction_pair, tmem_load_tile
from cudnn.frost.tile_dsl.scheduler import Sched, read_clc_payload, read_tile_id_arrive_on_leader, scheduler_warp_loop
from cudnn.frost.tile_dsl.thd import (
    TENSOR_MAP_QWORDS,
    THD_SETUP_THREADS,
    emit_seq_descs,
    thd_port_origin,
    write_thd_meta,
    write_thd_port_origins,
    write_thd_prefix_warp,
    write_thd_row_offsets,
)
from cudnn.frost.tile_dsl.tma import (
    bulk_copy,
    cp_async_bulk_shared_cluster_shared_cta,
    tma_load_subtiles,
    tma_load_tile,
    tma_reduce_add_subtile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
    tma_tensormap_acquire,
)
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.sdpa.bwd.config_sm100_d128 import make_cfg
from cudnn.sdpa.bwd.config_sm100_d128_accum import BAR_LAYOUT as BAR_LAYOUT_ACCUM_D128
from cudnn.sdpa.bwd.config_sm100_d128_accum import CFG as CFG_ACCUM_D128
from cudnn.sdpa.bwd.config_sm100_d128_accum import make_bar_layout, make_config
from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta import CFG as CFG_PAIR_BASE
from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta import diag_2cta_code
from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import (
    ADDED_BARRIERS_B2,
    ADDED_BARRIERS_XFA4,
    ALIASES,
    BAR_LAYOUT,
    BAR_LAYOUT_DQ576,
    BAR_LAYOUT_RING,
    BAR_LAYOUT_RING_DQ576,
    BAR_LAYOUT_RING_DQ576_B2,
    BAR_LAYOUT_XFA4,
    CFG,
    CFG_DQ576,
    CFG_RING,
    CFG_RING_DQ576,
    CFG_RING_DQ576_B2,
    DBG_CLK,
    DELETED_BARRIERS_DQ576,
    DELETED_BARRIERS_XFA4,
    SMEM_MAP_CODES,
    STATS_RING_ROWS,
    TMEM_MAP_CODES,
    UNSERVED_BARRIERS_XFA4,
    bar_layout_for_roles,
    cfg_for_geometry,
    dbg_clk_for_stats_form,
    mask_form_code,
    stats_form_code,
    warp_roles_code,
)
from cudnn.sdpa.bwd.d128_grouped_2cta_contract import DENSE_STATS_ORIGIN_SPEC, LAYOUT_FORM_CODES, STATS_KINDS
from cudnn.sdpa.bwd.kernels.bprop_chain_common import dkv_reduce_host
from cudnn.sdpa.bwd.kernels.bprop_thd_reduce import dkv_reduce_thd_origins_host

# ======================================================================================================================
# section 1: the shared chain kernels (delta / dQ zero, the dQ cast) and the row constants
# (was kernels/sm100/bprop_d128_helpers.py)
# ======================================================================================================================
"""Dense d128 delta, dQ initialization and native output conversion.

Both launches cover dQ's 128-row-rounded sequence: tail rows past S_q zero
their dQ accumulator rows and delta, and the conversion skips them.
"""

_PARAMS = globals().get("FROST_TEMPLATE_PARAMS", 128)
HEAD_DIM = int(_PARAMS) if isinstance(_PARAMS, int) else int(_PARAMS[0])
DQ_OUT = True if isinstance(_PARAMS, int) else bool(_PARAMS[1])
RATIO = 1 if isinstance(_PARAMS, int) or len(_PARAMS) < 3 else int(_PARAMS[2])
SPLIT = 1 if isinstance(_PARAMS, int) or len(_PARAMS) < 4 else int(_PARAMS[3])
if (RATIO != 1 or SPLIT != 1) and DQ_OUT:
    raise ValueError("grouped query heads and q-range splits are served by the dK / dV-only rendering")
if SPLIT not in (1, 2, 4):
    raise ValueError("the q-range split must be 1, 2 or 4")

THREADS = 128
ROWS_PER_CTA = 8
LANES_PER_ROW = 16
ELEMENTS_PER_LANE = 8
RN = cute.math.RoundingMode.NEAREST_EVEN


@cute.kernel
def dot_do_o_kernel(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    dq_accum: Optional[cute.Tensor],
    dot_factor: cutlass.Float32,
    lanes_per_row: cutlass.Constexpr[int] = LANES_PER_ROW,
    rows_per_cta: cutlass.Constexpr[int] = ROWS_PER_CTA,
    turns: Optional[cute.Tensor] = None,
):
    """``rows_per_cta`` rows per CTA, FP32 FTZ pair reduction, and dQ initialization when a dQ scratch is given."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    row = cutlass.Int64(block) * rows_per_cta + cutlass.Int64(tid // lanes_per_row)
    column = cutlass.Int64(tid % lanes_per_row) * ELEMENTS_PER_LANE
    # Tail rows past S_q read the last O / dO row (in bounds, uniform across
    # the shuffle group) and store delta = 0; their dQ rows are still zeroed.
    valid = row < cutlass.Int64(o.shape[1])
    io_row = cutlass.Int64(arith.select(valid.ir_value(), row.ir_value(), cutlass.Int64(o.shape[1] - 1).ir_value()))
    o_offset = (
        cutlass.Int64(batch) * cutlass.Int64(o.stride[0]) + io_row * cutlass.Int64(o.stride[1]) + cutlass.Int64(head) * cutlass.Int64(o.stride[2]) + column
    )
    do_offset = (
        cutlass.Int64(batch) * cutlass.Int64(do.stride[0]) + io_row * cutlass.Int64(do.stride[1]) + cutlass.Int64(head) * cutlass.Int64(do.stride[2]) + column
    )
    delta_offset = (
        cutlass.Int64(batch) * cutlass.Int64(delta.stride[0]) + cutlass.Int64(head) * cutlass.Int64(delta.stride[1]) + row * cutlass.Int64(delta.stride[2])
    )
    if cutlass.const_expr(dq_accum is not None):
        dq_offset = (
            cutlass.Int64(batch) * cutlass.Int64(dq_accum.stride[0])
            + row * cutlass.Int64(dq_accum.stride[1])
            + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2])
            + column
        )
        zero4 = cutlass.Vector.from_elements((cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0)), cutlass.Float32)
        (dq_accum.iterator.raw_ptr() + dq_offset).store(zero4, alignment=16)
        (dq_accum.iterator.raw_ptr() + dq_offset + 4).store(zero4, alignment=16)
    if cutlass.const_expr(turns is not None):
        if tid % lanes_per_row == 0 and row % 64 == 0:
            turn_base = cutlass.Int64(batch) * cutlass.Int64(turns.stride[0]) + cutlass.Int64(head) * cutlass.Int64(turns.stride[1])
            (turns.iterator.raw_ptr() + turn_base + (row // 64) * cutlass.Int64(turns.stride[2])).store(cutlass.Int32(0))
            if row == 0:
                (turns.iterator.raw_ptr() + turn_base + cutlass.Int64(turns.shape[2] - 1) * cutlass.Int64(turns.stride[2])).store(cutlass.Int32(0))
    do_values = (do.iterator.raw_ptr() + do_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
    o_values = (o.iterator.raw_ptr() + o_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
    pairs = []
    for j in cutlass.range_constexpr(ELEMENTS_PER_LANE // 2):
        high = cute.math.mul(do_values[2 * j + 1].to(cutlass.Float32), o_values[2 * j + 1].to(cutlass.Float32), rounding=RN, ftz=True)
        pairs.append(cute.math.fma(do_values[2 * j].to(cutlass.Float32), o_values[2 * j].to(cutlass.Float32), high, rounding=RN, ftz=True))
    dot = cute.math.add(pairs[0], pairs[1], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[2], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[3], rounding=RN, ftz=True)
    for step in cutlass.range_constexpr(lanes_per_row.bit_length() - 1):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=dot, offset=lanes_per_row >> (step + 1), mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        dot = cute.math.add(dot, peer, rounding=RN, ftz=True)
    dot = cute.math.mul(dot, dot_factor, rounding=RN, ftz=True)
    dot = cutlass.Float32(arith.select(valid.ir_value(), dot.ir_value(), cutlass.Float32(0.0).ir_value()))
    if tid % lanes_per_row == 0:
        (delta.iterator.raw_ptr() + delta_offset).store(dot)


dot_do_o_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def delta_zero_host(o: cute.Tensor, do: cute.Tensor, delta: cute.Tensor, dq_accum: cute.Tensor, stream: cuda_driver.CUstream):
    """One dense no-dropout launch over dQ's rounded rows (a multiple of 128)."""
    lanes = o.shape[3] // ELEMENTS_PER_LANE
    rows = THREADS // lanes
    dot_do_o_kernel(o, do, delta, dq_accum, cutlass.Float32(1.0), lanes, rows).launch(
        grid=(dq_accum.shape[1] // rows, o.shape[0], o.shape[2]), block=(THREADS, 1, 1), stream=stream
    )


@cute.jit
def delta_zero_turns_host(o: cute.Tensor, do: cute.Tensor, delta: cute.Tensor, dq_accum: cute.Tensor, turns: cute.Tensor, stream: cuda_driver.CUstream):
    """One dense launch over dQ's rounded rows that also resets the per-query-tile turn counters and the key-tile tickets."""
    lanes = o.shape[3] // ELEMENTS_PER_LANE
    rows = THREADS // lanes
    dot_do_o_kernel(o, do, delta, dq_accum, cutlass.Float32(1.0), lanes, rows, turns).launch(
        grid=(dq_accum.shape[1] // rows, o.shape[0], o.shape[2]), block=(THREADS, 1, 1), stream=stream
    )


@cute.jit
def delta_only_host(o: cute.Tensor, do: cute.Tensor, delta: cute.Tensor, stream: cuda_driver.CUstream):
    """One dense launch over delta's rounded rows (a multiple of 128), without a dQ scratch."""
    lanes = o.shape[3] // ELEMENTS_PER_LANE
    rows = THREADS // lanes
    dot_do_o_kernel(o, do, delta, None, cutlass.Float32(1.0), lanes, rows).launch(
        grid=(delta.shape[2] // rows, o.shape[0], o.shape[2]), block=(THREADS, 1, 1), stream=stream
    )


@cute.kernel
def cast_dq_d128_kernel(
    dq_accum: cute.Tensor,
    dq_out: cute.Tensor,
    io_dtype: cutlass.Constexpr,
    lanes_per_row: cutlass.Constexpr[int] = LANES_PER_ROW,
    rows_per_cta: cutlass.Constexpr[int] = ROWS_PER_CTA,
):
    """Convert eight FP32 elements per lane without rescaling."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    row = cutlass.Int64(block) * rows_per_cta + cutlass.Int64(tid // lanes_per_row)
    column = cutlass.Int64(tid % lanes_per_row) * ELEMENTS_PER_LANE
    src_offset = (
        cutlass.Int64(batch) * cutlass.Int64(dq_accum.stride[0])
        + row * cutlass.Int64(dq_accum.stride[1])
        + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2])
        + column
    )
    dst_offset = (
        cutlass.Int64(batch) * cutlass.Int64(dq_out.stride[0])
        + row * cutlass.Int64(dq_out.stride[1])
        + cutlass.Int64(head) * cutlass.Int64(dq_out.stride[2])
        + column
    )
    lo = (dq_accum.iterator.raw_ptr() + src_offset).load(count=4, alignment=16)
    hi = (dq_accum.iterator.raw_ptr() + src_offset + 4).load(count=4, alignment=16)
    packed = cutlass.Vector.from_elements(
        (
            fp32_to_fp16(lo[0], lo[1], dtype=io_dtype),
            fp32_to_fp16(lo[2], lo[3], dtype=io_dtype),
            fp32_to_fp16(hi[0], hi[1], dtype=io_dtype),
            fp32_to_fp16(hi[2], hi[3], dtype=io_dtype),
        ),
        cutlass.Int32,
    )
    if row < cutlass.Int64(dq_out.shape[1]):
        Pointer(cutlass.make_array_view(dq_out).data_ptr() + dst_offset, dtype=cutlass.Int32).store(packed, alignment=16)


cast_dq_d128_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def fold_dkv_d128_kernel(
    part: cute.Tensor,
    out: cute.Tensor,
    splits: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    lanes_per_row: cutlass.Constexpr[int] = LANES_PER_ROW,
    rows_per_cta: cutlass.Constexpr[int] = ROWS_PER_CTA,
):
    """Sum the ``splits`` fp32 partial planes of a dK / dV row in plane order and convert eight elements per lane."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    row = cutlass.Int64(block) * rows_per_cta + cutlass.Int64(tid // lanes_per_row)
    column = cutlass.Int64(tid % lanes_per_row) * ELEMENTS_PER_LANE
    acc = [cutlass.Float32(0.0) for _ in range(ELEMENTS_PER_LANE)]
    for s in cutlass.range_constexpr(splits):
        src_offset = (
            (cutlass.Int64(batch) * splits + s) * cutlass.Int64(part.stride[0])
            + row * cutlass.Int64(part.stride[1])
            + cutlass.Int64(head) * cutlass.Int64(part.stride[2])
            + column
        )
        lo = (part.iterator.raw_ptr() + src_offset).load(count=4, alignment=16)
        hi = (part.iterator.raw_ptr() + src_offset + 4).load(count=4, alignment=16)
        for j in cutlass.range_constexpr(4):
            acc[j] = cute.math.add(acc[j], lo[j], rounding=RN, ftz=True)
            acc[4 + j] = cute.math.add(acc[4 + j], hi[j], rounding=RN, ftz=True)
    packed = cutlass.Vector.from_elements(
        (
            fp32_to_fp16(acc[0], acc[1], dtype=io_dtype),
            fp32_to_fp16(acc[2], acc[3], dtype=io_dtype),
            fp32_to_fp16(acc[4], acc[5], dtype=io_dtype),
            fp32_to_fp16(acc[6], acc[7], dtype=io_dtype),
        ),
        cutlass.Int32,
    )
    dst_offset = (
        cutlass.Int64(batch) * cutlass.Int64(out.stride[0]) + row * cutlass.Int64(out.stride[1]) + cutlass.Int64(head) * cutlass.Int64(out.stride[2]) + column
    )
    if row < cutlass.Int64(out.shape[1]):
        Pointer(cutlass.make_array_view(out).data_ptr() + dst_offset, dtype=cutlass.Int32).store(packed, alignment=16)


fold_dkv_d128_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def fold_dkv_host(part: cute.Tensor, out: cute.Tensor, splits: cutlass.Constexpr[int], stream: cuda_driver.CUstream):
    """One fold launch over the output rows; output rows must be 16-byte aligned."""
    lanes = out.shape[3] // ELEMENTS_PER_LANE
    rows = THREADS // lanes
    fold_dkv_d128_kernel(part, out, splits, out.element_type, lanes, rows).launch(
        grid=((out.shape[1] + rows - 1) // rows, out.shape[0], out.shape[2]), block=(THREADS, 1, 1), stream=stream
    )


@cute.jit
def cast_dq_host(dq_accum: cute.Tensor, dq_out: cute.Tensor, io_dtype: cutlass.Constexpr, stream: cuda_driver.CUstream):
    """One native vector cast launch; output rows must be 16-byte aligned."""
    lanes = dq_out.shape[3] // ELEMENTS_PER_LANE
    rows = THREADS // lanes
    cast_dq_d128_kernel(dq_accum, dq_out, io_dtype, lanes, rows).launch(
        grid=((dq_out.shape[1] + rows - 1) // rows, dq_out.shape[0], dq_out.shape[2]), block=(THREADS, 1, 1), stream=stream
    )


# ======================================================================================================================
# section 2: the RECOMPUTE body: one launch, no workspace (d64 / d128, left window)
# (was kernels/sm100/bprop_d128_f16.py)
# ======================================================================================================================
"""Dense D64/D128 M128/N128 backward, one launch and no workspace.

D64 specializes the operand/MMA width and output stores; score/dS stay N128.
The detailed storage numbers below describe the D128 configuration.

CTA384: compute0..7 form two128-thread groups, each owns row=tid%128 and
64 columns; loader8, MMA9, stats10+11, output/TMEM control11. Input has two
paired Q/dO or K/V stages; O and stats each have one stage. KV retains64
original FP32 P values per lane through dP readiness; native copies feed
gapped MMA-TS dV/dK. Q dS remains SW128 shared MMA-SS A. K16 only.

SMEM bytes: resident0[0,32768), resident1[32768,65536), paired input0
[65536,131072), paired input1[131072,196608), auxiliary[196608,229376)
(Q prologue O then dS, or KV O), stats[229376,230400). Sixteen aligned
barrier arrays contain18 slots at230400:230656; Int32 TMEM handle230656.
Dynamic230660 plus separate driver allowance1024 gives231684 bytes.
Each128x128 operand uses TWO SW128 TMA boxes[128,64], head coordinates0/64,
half offsets0/8192. Gradient B is MN-major LBO16384,stride1024; score and
Q dS A are K-major LBO0,stride1024. Generic Q dS stores use64-byte alignment.

TMEM: S[0,128), DP[128,256), G0[256,384), G1[384,512). Packed P halves
occupy[0,32),[64,96); dS occupies[128,160),[192,224). Groupg chunkc loads
FP32 atplane+64*g+c then LOAD-waits before storing native atplane+64*g+c/2.
Groups never compact into each other's unread source. All256 lanes STORE-
wait and BEFORE_THREAD_SYNC fence before P/dS publication. Consumers use
AFTER_THREAD_SYNC after readiness/completion waits before tcgen05 operations. TS A raw Int8 steps0/8/16/24/64/72/80/88 use
existing mma_ts_step; gradient C offsets are raw before Float32 conversion.

LOCAL CTA1 barrier ledger: field(depth,producer,count; issuers -> waiters):
 resident_full(1,TMA,1; loader8 elect -> all384)
 q_stats_ready(1,THREAD,64; Q stats10+11 bare -> Q compute256,loader8)
 input_full(2,TMA,1 each; loader8 elect -> MMA9,KV stats10+11)
 input_empty(2,MMA,1 each; MMA9 elect final gradient -> loader8)
 o_full(1,TMA,1; KV loader8 elect -> KV stats10+11)
 o_empty(1,THREAD,64; KV stats10+11 bare -> KV loader8)
 stats_full(1,THREAD,64; KV stats10+11 bare -> KV compute256)
 stats_empty(1,THREAD,256; KV compute0..7 bare -> KV stats10+11)
 s_full(1,MMA,1; MMA9 elect S or Q S+DP -> compute256)
 dp_full(1,MMA,1; KV MMA9 elect DP -> KV compute256)
 p_full(1,THREAD,256; KV compute0..7 bare after STORE wait -> MMA9)
 ds_full(1,THREAD,256; compute0..7 bare after STORE wait/Q fence -> MMA9)
 q_score_empty(1,THREAD,256; Q compute0..7 bare -> Q MMA9)
 q_ds_empty(1,MMA,1; Q MMA9 elect final gradient -> Q compute256)
 final_grad(1,MMA,1; MMA9 elect ONCE iff nonempty -> compute256 iff nonempty)
 out_ready(1,THREAD,256; compute0..7 bare output/fence -> control11)
All18 init sites are warp0 AND elect, then fence_mbarrier_init and CTA384
join before use. Full starts phase0, empty waiters phase1, independent typed
states advance once per event. No synthetic arrivals or role-local CTA join.

KV schedule S0 then DP(j),DV(j),S(j+1),DK(j). Same-issuer/group MMA A-read
before later D-write protects DV->nextS and DK->nextDP WAR. Cross-warp
packed stores still wait cumulative S/DP completion (including old DV/DK),
never merely issuer order. Input release follows DK completion and includes
both stats readers transitively through stats_full -> DS -> ds_full -> DK.
O releases only after all64 stats threads finish both rows into four private
FP32 scalars per lane. Stats then wait stats_empty before publishing those
scalars; early O release cannot overwrite the still-live statistics slab. Q shared dS reuse
requires q_ds_empty completion; q_score_empty only releases S/DP reads.

Every tail wait advances: loader input_empty twice and KV o_empty once;
both KV stats warps stats_empty once; Q MMA q_score_empty once; Q compute
q_ds_empty once. Nonempty compute waits one-shot final_grad before output.
Zero-trip still waits resident+CTA join, drains initial empties, writes zero
outputs without TMEM gradient reads, completes output TMA and final CTA384
join before warp11 collective512-column deallocation.
"""

CFG_RECOMPUTE = make_cfg(HEAD_DIM)

_SWIZZLE = cutlass.Swizzle(3, 4, 3)
_LOG2E = 1.4426950408889634


def _operand_recompute(raw, offset, rows, *, transposed=False, cols=CFG_RECOMPUTE.HEAD_DIM):
    return SmemTile(
        base=raw.subview(offset),
        elems_per_stage=rows * cols,
        leading_byte_offset=rows * CFG_RECOMPUTE.SWIZZLE_BYTES if transposed else 0,
        stride_byte_offset=8 * CFG_RECOMPUTE.SWIZZLE_BYTES,
        layout=CFG_RECOMPUTE.SMEM_LAYOUT,
        tma_loads_per_tile=cols // CFG_RECOMPUTE.HEAD_GRANULE,
        tma_granu_elems=CFG_RECOMPUTE.HEAD_GRANULE,
        tma_subtile_stride_elems=rows * CFG_RECOMPUTE.HEAD_GRANULE,
        desc_version=CFG_RECOMPUTE.DESC_VERSION,
    )


def _mma_desc_recompute(dtype, *, gradient=False):
    n = CFG_RECOMPUTE.HEAD_DIM if gradient else CFG_RECOMPUTE.TILE_N
    k = CFG_RECOMPUTE.TILE_N if gradient else CFG_RECOMPUTE.HEAD_DIM
    return MmaDesc(
        M=CFG_RECOMPUTE.TILE_M,
        N=n,
        K=k,
        bpe_a=CFG_RECOMPUTE.BPE,
        bpe_b=CFG_RECOMPUTE.BPE,
        tile_k_hw=CFG_RECOMPUTE.TILE_K_HW,
        btranspose=gradient,
        cta_group=CFG_RECOMPUTE.CTA_MMA,
        idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, a_dtype=dtype, b_dtype=dtype, m_dim=CFG_RECOMPUTE.TILE_M, n_dim=n, b_major=int(gradient)),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


def _input(slab, stage, operand, *, transposed=False):
    return _operand_recompute(
        slab,
        CFG_RECOMPUTE.INPUT + stage * CFG_RECOMPUTE.INPUT_STAGE_ELEMS + operand * CFG_RECOMPUTE.INPUT_OPERAND_ELEMS,
        CFG_RECOMPUTE.TILE_N,
        transposed=transposed,
    )


@cute.jit
def _wait_service(bar, state):
    bar[state.idx].wait(state.phase)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _read_stats_row(s_o, s_do, lse, batch, head, q_base, sq, rows: cutlass.Constexpr[int], row):
    delta = cutlass.Float32(0.0)
    for chunk in cutlass.range_constexpr(CFG_RECOMPUTE.HEAD_DIM // CFG_RECOMPUTE.HEAD_GRANULE):
        off = chunk * rows * CFG_RECOMPUTE.HEAD_GRANULE + row * CFG_RECOMPUTE.HEAD_GRANULE
        o_half = s_o.base.subview(off).data_ptr().load_swizzled(_SWIZZLE, alignment=128, count=CFG_RECOMPUTE.HEAD_GRANULE)
        do_half = s_do.base.subview(off).data_ptr().load_swizzled(_SWIZZLE, alignment=128, count=CFG_RECOMPUTE.HEAD_GRANULE)
        # Preserve the SW128 load and logical element order while expanding
        # packed pairs directly, avoiding generic BF16 vector conversion.
        o_words = o_half.bitcast(cutlass.Int32)
        do_words = do_half.bitcast(cutlass.Int32)
        o = cutlass.Vector.from_elements(
            tuple(v for i in range(CFG_RECOMPUTE.HEAD_GRANULE // 2) for v in f16x2_to_f32(o_words[i], dtype=o_half.dtype)), cutlass.Float32
        )
        do = cutlass.Vector.from_elements(
            tuple(v for i in range(CFG_RECOMPUTE.HEAD_GRANULE // 2) for v in f16x2_to_f32(do_words[i], dtype=do_half.dtype)), cutlass.Float32
        )
        pair = row_reduction_pair(o * do)
        delta += pair[0] + pair[1]
    lse_val = cutlass.Float32(float("inf"))
    if q_base + row < sq:
        lse_val = lse[batch, head, q_base + row]
    return lse_val, delta


@cute.jit
def _write_stats(s_o, s_do, stats, lse, batch, head, q_base, sq, rows: cutlass.Constexpr[int]):
    stats_thread = cute.arch.thread_idx()[0] - CFG_RECOMPUTE.STATS_WARP * CFG_RECOMPUTE.WARP_LANES
    for row_group in cutlass.range_constexpr(rows // CFG_RECOMPUTE.STATS_THREADS):
        row = stats_thread + row_group * CFG_RECOMPUTE.STATS_THREADS
        lse_val, delta = _read_stats_row(s_o, s_do, lse, batch, head, q_base, sq, rows, row)
        stats.subview(row).store(lse_val)
        stats.subview(rows + row).store(delta)


@cute.jit
def _probability_words(
    stats,
    owner_base,
    stream_base,
    sq,
    skv,
    causal: cutlass.Constexpr[bool],
    q_owner: cutlass.Constexpr[bool],
    col: cutlass.Constexpr[int],
    window_left: cutlass.Constexpr[int],
):
    tid = cute.arch.thread_idx()[0] % CFG_RECOMPUTE.TILE_M
    lo = cutlass.Int32(0)
    if cutlass.const_expr(q_owner):
        query = owner_base + tid
        hi = skv
        if cutlass.const_expr(causal):
            hi = cute.math.min(hi, query + 1)
        if cutlass.const_expr(window_left > 0):
            lo = cute.math.max(lo, query - window_left + 1)
        live = (query < sq) & (stats.subview(tid).load() != cutlass.Float32(float("-inf")))
        hi = cutlass.Int32(arith.select(live.ir_value(), hi.ir_value(), stream_base.ir_value()))
        ds_lo = lo
        ds_hi = cutlass.Int32(arith.select((hi - lo > 1).ir_value(), hi.ir_value(), stream_base.ir_value()))
    else:
        key = owner_base + tid
        hi = sq
        if cutlass.const_expr(causal):
            lo = key
        if cutlass.const_expr(window_left > 0):
            hi = cute.math.min(hi, key + window_left)
        hi = cutlass.Int32(arith.select((key < skv).ir_value(), hi.ir_value(), stream_base.ir_value()))
        ds_lo = cute.math.max(lo, cutlass.Int32(1)) if cutlass.const_expr(causal) else lo
        ds_hi = cutlass.Int32(arith.select((skv > 1).ir_value(), hi.ir_value(), stream_base.ir_value()))
        if cutlass.const_expr(window_left > 0):
            # At query skv + W - 2 only the last key remains; its dS is zero.
            ds_hi = cute.math.min(ds_hi, skv + window_left - 2)
            if cutlass.const_expr(causal and window_left == 1):
                ds_hi = stream_base
    return band_mask_words(lo, hi, stream_base + col, CFG_RECOMPUTE.POINTWISE_COLS), band_mask_words(
        ds_lo, ds_hi, stream_base + col, CFG_RECOMPUTE.POINTWISE_COLS
    )


@cute.jit
def _probability_masked(
    s,
    dp,
    stats,
    owner_base,
    stream_base,
    sq,
    skv,
    scale,
    causal: cutlass.Constexpr[bool],
    q_owner: cutlass.Constexpr[bool],
    col: cutlass.Constexpr[int],
    window_left: cutlass.Constexpr[int] = 0,
):
    tid = cute.arch.thread_idx()[0] % CFG_RECOMPUTE.TILE_M
    if cutlass.const_expr(q_owner):
        lse = stats.subview(tid).load()
        delta = stats.subview(CFG_RECOMPUTE.Q_STATS_DELTA + tid).load()
    else:
        lse = cutlass.Vector.from_elements(tuple(stats.subview(col + i).load() for i in range(CFG_RECOMPUTE.POINTWISE_COLS)), cutlass.Float32)
        delta = cutlass.Vector.from_elements(
            tuple(stats.subview(CFG_RECOMPUTE.KV_STATS_DELTA + col + i).load() for i in range(CFG_RECOMPUTE.POINTWISE_COLS)), cutlass.Float32
        )
    words, ds_words = _probability_words(stats, owner_base, stream_base, sq, skv, causal, q_owner, col, window_left)
    p = cute.math.exp2((s * scale - lse) * _LOG2E, fastmath=True)
    p = apply_mask_words(p, words, mask_value=0.0, n_cols=CFG_RECOMPUTE.POINTWISE_COLS)
    # A row with exactly one legal key has softmax derivative exactly zero.
    # Select zero rather than retaining dP-versus-delta reduction roundoff;
    # P remains unchanged because that key still contributes to dV.
    ds = apply_mask_words((dp - delta) * p * scale, ds_words, mask_value=0.0, n_cols=CFG_RECOMPUTE.POINTWISE_COLS)
    return p, ds


@cute.jit
def _probability_full(s, dp, stats, scale, q_owner: cutlass.Constexpr[bool], col: cutlass.Constexpr[int]):
    tid = cute.arch.thread_idx()[0] % CFG_RECOMPUTE.TILE_M
    if cutlass.const_expr(q_owner):
        lse = stats.subview(tid).load()
        delta = stats.subview(CFG_RECOMPUTE.Q_STATS_DELTA + tid).load()
    else:
        lse = cutlass.Vector.from_elements(tuple(stats.subview(col + i).load() for i in range(CFG_RECOMPUTE.POINTWISE_COLS)), cutlass.Float32)
        delta = cutlass.Vector.from_elements(
            tuple(stats.subview(CFG_RECOMPUTE.KV_STATS_DELTA + col + i).load() for i in range(CFG_RECOMPUTE.POINTWISE_COLS)), cutlass.Float32
        )
    # Full rectangles exclude SKV1 and causal query0, so both original
    # masks are all-one. KV keeps its original nonfinite-LSE behavior.
    p = cute.math.exp2((s * scale - lse) * _LOG2E, fastmath=True)
    ds = (dp - delta) * p * scale
    return p, ds


@cute.jit
def _probability(
    s,
    dp,
    stats,
    owner_base,
    stream_base,
    sq,
    skv,
    scale,
    causal: cutlass.Constexpr[bool],
    q_owner: cutlass.Constexpr[bool],
    col: cutlass.Constexpr[int],
    window_left: cutlass.Constexpr[int] = 0,
):
    tid = cute.arch.thread_idx()[0] % CFG_RECOMPUTE.TILE_M
    if cutlass.const_expr(q_owner):
        full = (owner_base + CFG_RECOMPUTE.TILE_M <= sq) & (stream_base + CFG_RECOMPUTE.TILE_N <= skv)
        if cutlass.const_expr(causal):
            full = full & (stream_base + CFG_RECOMPUTE.TILE_N - 1 <= owner_base)
        lse = stats.subview(tid).load()
        # All32 lanes participate even on geometric tails. A sentinel in any
        # lane sends that whole warp through the original zero-masking path.
        lse_live = nvvm.vote_sync(cute.arch.FULL_MASK, lse != cutlass.Float32(float("-inf")), nvvm.VoteSync.ALL)
        full = full & lse_live
    else:
        full = (owner_base + CFG_RECOMPUTE.TILE_M <= skv) & (stream_base + CFG_RECOMPUTE.TILE_N <= sq)
        if cutlass.const_expr(causal):
            full = full & (stream_base >= owner_base + CFG_RECOMPUTE.TILE_M - 1)
    # Seed staged-control-flow result types; both branches overwrite them.
    p, ds = s, dp
    if full & (window_left == 0):
        p, ds = _probability_full(s, dp, stats, scale, q_owner, col)
    else:
        p, ds = _probability_masked(s, dp, stats, owner_base, stream_base, sq, skv, scale, causal, q_owner, col, window_left)
    return p, ds


@cute.jit
def _kv_full(owner_base, stream_base, sq, skv, causal: cutlass.Constexpr[bool], window_left: cutlass.Constexpr[int] = 0):
    full = (owner_base + CFG_RECOMPUTE.TILE_M <= skv) & (stream_base + CFG_RECOMPUTE.TILE_N <= sq)
    if cutlass.const_expr(causal):
        full = full & (stream_base >= owner_base + CFG_RECOMPUTE.TILE_M - 1)
    return full & (window_left == 0)


@cute.jit
def _kv_probability_p(
    s, stats, owner_base, stream_base, sq, skv, scale, causal: cutlass.Constexpr[bool], col: cutlass.Constexpr[int], window_left: cutlass.Constexpr[int] = 0
):
    lse = cutlass.Vector.from_elements(tuple(stats.subview(col + i).load() for i in range(CFG_RECOMPUTE.POINTWISE_COLS)), cutlass.Float32)
    p = s
    if _kv_full(owner_base, stream_base, sq, skv, causal, window_left):
        p = cute.math.exp2((s * scale - lse) * _LOG2E, fastmath=True)
    else:
        words, _ = _probability_words(stats, owner_base, stream_base, sq, skv, causal, False, col, window_left)
        p = cute.math.exp2((s * scale - lse) * _LOG2E, fastmath=True)
        p = apply_mask_words(p, words, mask_value=0.0, n_cols=CFG_RECOMPUTE.POINTWISE_COLS)
    return p


@cute.jit
def _kv_probability_ds(
    dp, p, stats, owner_base, stream_base, sq, skv, scale, causal: cutlass.Constexpr[bool], col: cutlass.Constexpr[int], window_left: cutlass.Constexpr[int] = 0
):
    delta = cutlass.Vector.from_elements(
        tuple(stats.subview(CFG_RECOMPUTE.KV_STATS_DELTA + col + i).load() for i in range(CFG_RECOMPUTE.POINTWISE_COLS)), cutlass.Float32
    )
    ds = dp
    if _kv_full(owner_base, stream_base, sq, skv, causal, window_left):
        ds = (dp - delta) * p * scale
    else:
        _, ds_words = _probability_words(stats, owner_base, stream_base, sq, skv, causal, False, col, window_left)
        ds = apply_mask_words((dp - delta) * p * scale, ds_words, mask_value=0.0, n_cols=CFG_RECOMPUTE.POINTWISE_COLS)
    return ds


@cute.jit
def _kv_p_chunk(
    score_base,
    stats,
    owner_base,
    stream_base,
    sq,
    skv,
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    col: cutlass.Constexpr[int],
    window_left: cutlass.Constexpr[int] = 0,
):
    s = tmem_load_tile(score_base + CFG_RECOMPUTE.TMEM_S + col, num_elems=CFG_RECOMPUTE.POINTWISE_COLS, ld_num=CFG_RECOMPUTE.POINTWISE_COLS).vec
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    p = _kv_probability_p(s, stats, owner_base, stream_base, sq, skv, scale, causal, col, window_left)
    packed_col = (col // CFG_RECOMPUTE.GROUP_COLS) * CFG_RECOMPUTE.GROUP_COLS + (
        col % CFG_RECOMPUTE.GROUP_COLS
    ) * CFG_RECOMPUTE.BPE // CFG_RECOMPUTE.TMEM_WORD_BYTES
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(score_base + CFG_RECOMPUTE.TMEM_S + packed_col, cutlass.Float32), p.to(dtype))
    return p  # The original FP32 vector remains live; native conversion is a copy.


@cute.jit
def _kv_ds_chunk(
    score_base,
    p,
    stats,
    owner_base,
    stream_base,
    sq,
    skv,
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    col: cutlass.Constexpr[int],
    window_left: cutlass.Constexpr[int] = 0,
):
    dp = tmem_load_tile(score_base + CFG_RECOMPUTE.TMEM_DP + col, num_elems=CFG_RECOMPUTE.POINTWISE_COLS, ld_num=CFG_RECOMPUTE.POINTWISE_COLS).vec
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    ds = _kv_probability_ds(dp, p, stats, owner_base, stream_base, sq, skv, scale, causal, col, window_left)
    packed_col = (col // CFG_RECOMPUTE.GROUP_COLS) * CFG_RECOMPUTE.GROUP_COLS + (
        col % CFG_RECOMPUTE.GROUP_COLS
    ) * CFG_RECOMPUTE.BPE // CFG_RECOMPUTE.TMEM_WORD_BYTES
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(score_base + CFG_RECOMPUTE.TMEM_DP + packed_col, cutlass.Float32), ds.to(dtype))


class BarsRecompute(NamedTuple):
    resident_full: MBarrier
    q_stats_ready: MBarrier
    input_full: MBarrier
    input_empty: MBarrier
    o_full: MBarrier
    o_empty: MBarrier
    stats_full: MBarrier
    stats_empty: MBarrier
    s_full: MBarrier
    dp_full: MBarrier
    p_full: MBarrier
    ds_full: MBarrier
    q_score_empty: MBarrier
    q_ds_empty: MBarrier
    final_grad: MBarrier
    out_ready: MBarrier


def make_bars():
    def make(stages, count, producer):
        return MBarrier(cutlass.Array(cutlass.Int64, stages, alignment=16, space=cutlass.AddressSpace.smem), stages=stages, init_count=count, producer=producer)

    return BarsRecompute(
        make(1, CFG_RECOMPUTE.ONE_LANE, Producer.TMA_LOAD),
        make(1, CFG_RECOMPUTE.STATS_ARRIVERS, Producer.THREAD),
        make(CFG_RECOMPUTE.INPUT_STAGES, CFG_RECOMPUTE.ONE_LANE, Producer.TMA_LOAD),
        make(CFG_RECOMPUTE.INPUT_STAGES, CFG_RECOMPUTE.ONE_LANE, Producer.MMA_COMMIT),
        make(1, CFG_RECOMPUTE.ONE_LANE, Producer.TMA_LOAD),
        make(1, CFG_RECOMPUTE.STATS_ARRIVERS, Producer.THREAD),
        make(1, CFG_RECOMPUTE.STATS_ARRIVERS, Producer.THREAD),
        make(1, CFG_RECOMPUTE.COMPUTE_ARRIVERS, Producer.THREAD),
        make(1, CFG_RECOMPUTE.ONE_LANE, Producer.MMA_COMMIT),
        make(1, CFG_RECOMPUTE.ONE_LANE, Producer.MMA_COMMIT),
        make(1, CFG_RECOMPUTE.COMPUTE_ARRIVERS, Producer.THREAD),
        make(1, CFG_RECOMPUTE.COMPUTE_ARRIVERS, Producer.THREAD),
        make(1, CFG_RECOMPUTE.COMPUTE_ARRIVERS, Producer.THREAD),
        make(1, CFG_RECOMPUTE.ONE_LANE, Producer.MMA_COMMIT),
        make(1, CFG_RECOMPUTE.ONE_LANE, Producer.MMA_COMMIT),
        make(1, CFG_RECOMPUTE.COMPUTE_ARRIVERS, Producer.THREAD),
    )


def _q_ds(slab):
    return _operand_recompute(slab, CFG_RECOMPUTE.AUX, CFG_RECOMPUTE.TILE_M, cols=CFG_RECOMPUTE.TILE_N)


@cute.jit
def _mma_ts_gapped(desc, plane, desc_b, dest, accumulate):
    for k in cutlass.range_constexpr(CFG_RECOMPUTE.TILE_N // CFG_RECOMPUTE.TILE_K_HW):
        # mma_ts_step adds the packed K16 stride. Shift the origin past each
        # group gap: logical group width minus its native packed width.
        mma_ts_step(
            desc,
            nvvm.make_tmem_ptr(
                plane
                + (k // (CFG_RECOMPUTE.GROUP_COLS // CFG_RECOMPUTE.TILE_K_HW))
                * (CFG_RECOMPUTE.GROUP_COLS - CFG_RECOMPUTE.GROUP_COLS * CFG_RECOMPUTE.BPE // CFG_RECOMPUTE.TMEM_WORD_BYTES),
                cutlass.Int8,
            ),
            desc_b,
            dest,
            k_idx=k,
            accumulate=accumulate if cutlass.const_expr(k == 0) else True,
        )


@cute.jit
def _issue_kv_s(slab, bars, tmem_base, input_state, full_state, dtype: cutlass.Constexpr):
    _wait_service(bars.input_full, input_state)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    mma_ss(
        _mma_desc_recompute(dtype),
        _operand_recompute(slab, CFG_RECOMPUTE.RESIDENT0, CFG_RECOMPUTE.TILE_M).desc(),
        _input(slab, input_state.idx, 0).desc(),
        nvvm.make_tmem_ptr(tmem_base + CFG_RECOMPUTE.TMEM_S, cutlass.Float32),
    )
    bars.s_full[full_state.idx].arrive(cta_group=CFG_RECOMPUTE.CTA_MMA, pred=nvvm.elect_sync())
    return advance(input_state, CFG_RECOMPUTE.INPUT_STAGES), advance(full_state, CFG_RECOMPUTE.SINGLE_STAGE)


@cute.jit
def _mma_kv(slab, bars, tmem_base, lo, hi, dtype: cutlass.Constexpr):
    input_state = PipelineState.start()
    input_empty_state = PipelineState.start()
    s_state = PipelineState.start()
    dp_state = PipelineState.start()
    p_state = PipelineState.start()
    ds_state = PipelineState.start()
    score = _mma_desc_recompute(dtype)
    grad = _mma_desc_recompute(dtype, gradient=True)
    if hi > lo:
        input_state, s_state = _issue_kv_s(slab, bars, tmem_base, input_state, s_state, dtype)
    for tile in cutlass.range(lo, hi, 1, unroll=1):
        mma_ss(
            score,
            _operand_recompute(slab, CFG_RECOMPUTE.RESIDENT1, CFG_RECOMPUTE.TILE_M).desc(),
            _input(slab, input_empty_state.idx, 1).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG_RECOMPUTE.TMEM_DP, cutlass.Float32),
        )
        bars.dp_full[dp_state.idx].arrive(cta_group=CFG_RECOMPUTE.CTA_MMA, pred=nvvm.elect_sync())
        dp_state = advance(dp_state, CFG_RECOMPUTE.SINGLE_STAGE)
        _wait_service(bars.p_full, p_state)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        p_state = advance(p_state, CFG_RECOMPUTE.SINGLE_STAGE)
        _mma_ts_gapped(
            grad,
            tmem_base + CFG_RECOMPUTE.TMEM_S,
            _input(slab, input_empty_state.idx, 1, transposed=True).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG_RECOMPUTE.TMEM_G1, cutlass.Float32),
            tile > lo,
        )
        if tile + 1 < hi:
            input_state, s_state = _issue_kv_s(slab, bars, tmem_base, input_state, s_state, dtype)
        _wait_service(bars.ds_full, ds_state)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        ds_state = advance(ds_state, CFG_RECOMPUTE.SINGLE_STAGE)
        _mma_ts_gapped(
            grad,
            tmem_base + CFG_RECOMPUTE.TMEM_DP,
            _input(slab, input_empty_state.idx, 0, transposed=True).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG_RECOMPUTE.TMEM_G0, cutlass.Float32),
            tile > lo,
        )
        bars.input_empty[input_empty_state.idx].arrive(cta_group=CFG_RECOMPUTE.CTA_MMA, pred=nvvm.elect_sync())
        input_empty_state = advance(input_empty_state, CFG_RECOMPUTE.INPUT_STAGES)
    if hi > lo:
        bars.final_grad.arrive(cta_group=CFG_RECOMPUTE.CTA_MMA, pred=nvvm.elect_sync())


@cute.jit
def _mma_q(slab, bars, tmem_base, lo, hi, dtype: cutlass.Constexpr):
    input_state = PipelineState.start()
    input_empty_state = PipelineState.start()
    empty_state = PipelineState.start(phase=1)
    s_state = PipelineState.start()
    ds_state = PipelineState.start()
    ds_empty_state = PipelineState.start()
    score = _mma_desc_recompute(dtype)
    grad = _mma_desc_recompute(dtype, gradient=True)
    for tile in cutlass.range(lo, hi, 1, unroll=1):
        _wait_service(bars.input_full, input_state)
        _wait_service(bars.q_score_empty, empty_state)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        mma_ss(
            score,
            _operand_recompute(slab, CFG_RECOMPUTE.RESIDENT0, CFG_RECOMPUTE.TILE_M).desc(),
            _input(slab, input_state.idx, 0).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG_RECOMPUTE.TMEM_S, cutlass.Float32),
        )
        mma_ss(
            score,
            _operand_recompute(slab, CFG_RECOMPUTE.RESIDENT1, CFG_RECOMPUTE.TILE_M).desc(),
            _input(slab, input_state.idx, 1).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG_RECOMPUTE.TMEM_DP, cutlass.Float32),
        )
        bars.s_full[s_state.idx].arrive(cta_group=CFG_RECOMPUTE.CTA_MMA, pred=nvvm.elect_sync())
        input_state = advance(input_state, CFG_RECOMPUTE.INPUT_STAGES)
        empty_state = advance(empty_state, CFG_RECOMPUTE.SINGLE_STAGE)
        s_state = advance(s_state, CFG_RECOMPUTE.SINGLE_STAGE)
        _wait_service(bars.ds_full, ds_state)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        mma_ss(
            grad,
            _q_ds(slab).desc(),
            _input(slab, input_empty_state.idx, 0, transposed=True).desc(),
            nvvm.make_tmem_ptr(tmem_base + CFG_RECOMPUTE.TMEM_G0, cutlass.Float32),
            accumulate=tile > lo,
        )
        bars.input_empty[input_empty_state.idx].arrive(cta_group=CFG_RECOMPUTE.CTA_MMA, pred=nvvm.elect_sync())
        bars.q_ds_empty[ds_empty_state.idx].arrive(cta_group=CFG_RECOMPUTE.CTA_MMA, pred=nvvm.elect_sync())
        ds_state = advance(ds_state, CFG_RECOMPUTE.SINGLE_STAGE)
        input_empty_state = advance(input_empty_state, CFG_RECOMPUTE.INPUT_STAGES)
        ds_empty_state = advance(ds_empty_state, CFG_RECOMPUTE.SINGLE_STAGE)
    if hi > lo:
        bars.final_grad.arrive(cta_group=CFG_RECOMPUTE.CTA_MMA, pred=nvvm.elect_sync())
    _wait_service(bars.q_score_empty, empty_state)


@cute.jit
def _stage_gradient(tmem_base, tmem_offset: cutlass.Constexpr[int], smem, live, dtype: cutlass.Constexpr, group: cutlass.Constexpr[int]):
    # The second score group still participates in every readiness barrier,
    # but D64 has no second 64-column gradient slab to read or store.
    if cutlass.const_expr(group * CFG_RECOMPUTE.GROUP_COLS < CFG_RECOMPUTE.HEAD_DIM):
        row = cute.arch.thread_idx()[0] % CFG_RECOMPUTE.TILE_M
        values = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG_RECOMPUTE.GROUP_COLS)), cutlass.Float32)
        if live:
            values = tmem_load_tile(tmem_base + tmem_offset + group * CFG_RECOMPUTE.GROUP_COLS, num_elems=CFG_RECOMPUTE.GROUP_COLS).vec
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        smem.base.subview(group * CFG_RECOMPUTE.TILE_M * CFG_RECOMPUTE.GROUP_COLS + row * CFG_RECOMPUTE.GROUP_COLS).data_ptr().store_swizzled(
            values.to(dtype), alignment=128, swizzle=_SWIZZLE
        )


@cute.jit
def _compute_kv(
    slab,
    stats,
    bars,
    tmem_base,
    owner_base,
    lo,
    hi,
    sq,
    skv,
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    group: cutlass.Constexpr[int],
    window_left: cutlass.Constexpr[int] = 0,
):
    s_state = PipelineState.start()
    dp_state = PipelineState.start()
    stats_full_state = PipelineState.start()
    stats_empty_state = PipelineState.start()
    p_state = PipelineState.start()
    ds_state = PipelineState.start()
    for tile in cutlass.range(lo, hi, 1, unroll=1):
        bars.s_full[s_state.idx].wait(s_state.phase)
        bars.stats_full[stats_full_state.idx].wait(stats_full_state.phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        p0 = _kv_p_chunk(
            tmem_base, stats, owner_base, tile * CFG_RECOMPUTE.TILE_N, sq, skv, scale, causal, dtype, group * CFG_RECOMPUTE.GROUP_COLS, window_left
        )
        p1 = _kv_p_chunk(
            tmem_base,
            stats,
            owner_base,
            tile * CFG_RECOMPUTE.TILE_N,
            sq,
            skv,
            scale,
            causal,
            dtype,
            group * CFG_RECOMPUTE.GROUP_COLS + CFG_RECOMPUTE.POINTWISE_COLS,
            window_left,
        )
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.p_full[p_state.idx].arrive()
        bars.dp_full[dp_state.idx].wait(dp_state.phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        _kv_ds_chunk(
            tmem_base, p0, stats, owner_base, tile * CFG_RECOMPUTE.TILE_N, sq, skv, scale, causal, dtype, group * CFG_RECOMPUTE.GROUP_COLS, window_left
        )
        _kv_ds_chunk(
            tmem_base,
            p1,
            stats,
            owner_base,
            tile * CFG_RECOMPUTE.TILE_N,
            sq,
            skv,
            scale,
            causal,
            dtype,
            group * CFG_RECOMPUTE.GROUP_COLS + CFG_RECOMPUTE.POINTWISE_COLS,
            window_left,
        )
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.stats_empty[stats_empty_state.idx].arrive()
        bars.ds_full[ds_state.idx].arrive()
        s_state = advance(s_state, CFG_RECOMPUTE.SINGLE_STAGE)
        dp_state = advance(dp_state, CFG_RECOMPUTE.SINGLE_STAGE)
        stats_full_state = advance(stats_full_state, CFG_RECOMPUTE.SINGLE_STAGE)
        stats_empty_state = advance(stats_empty_state, CFG_RECOMPUTE.SINGLE_STAGE)
        p_state = advance(p_state, CFG_RECOMPUTE.SINGLE_STAGE)
        ds_state = advance(ds_state, CFG_RECOMPUTE.SINGLE_STAGE)
    if hi > lo:
        bars.final_grad.wait(PipelineState.start().phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    _stage_gradient(tmem_base, CFG_RECOMPUTE.TMEM_G0, _operand_recompute(slab, CFG_RECOMPUTE.RESIDENT0, CFG_RECOMPUTE.TILE_M), hi > lo, dtype, group)
    _stage_gradient(tmem_base, CFG_RECOMPUTE.TMEM_G1, _operand_recompute(slab, CFG_RECOMPUTE.RESIDENT1, CFG_RECOMPUTE.TILE_M), hi > lo, dtype, group)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.out_ready.arrive()


@cute.jit
def _compute_q(
    slab,
    stats,
    bars,
    tmem_base,
    owner_base,
    lo,
    hi,
    sq,
    skv,
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    group: cutlass.Constexpr[int],
    window_left: cutlass.Constexpr[int] = 0,
):
    row = cute.arch.thread_idx()[0] % CFG_RECOMPUTE.TILE_M
    s_state = PipelineState.start()
    empty_state = PipelineState.start()
    ds_state = PipelineState.start()
    ds_empty_state = PipelineState.start(phase=1)
    bars.q_stats_ready.wait(PipelineState.start().phase)
    for tile in cutlass.range(lo, hi, 1, unroll=1):
        bars.s_full[s_state.idx].wait(s_state.phase)
        bars.q_ds_empty[ds_empty_state.idx].wait(ds_empty_state.phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        for chunk in cutlass.range_constexpr(CFG_RECOMPUTE.GROUP_COLS // CFG_RECOMPUTE.POINTWISE_COLS):
            col = group * CFG_RECOMPUTE.GROUP_COLS + chunk * CFG_RECOMPUTE.POINTWISE_COLS
            s = tmem_load_tile(tmem_base + CFG_RECOMPUTE.TMEM_S + col, num_elems=CFG_RECOMPUTE.POINTWISE_COLS, ld_num=CFG_RECOMPUTE.POINTWISE_COLS).vec
            dp = tmem_load_tile(tmem_base + CFG_RECOMPUTE.TMEM_DP + col, num_elems=CFG_RECOMPUTE.POINTWISE_COLS, ld_num=CFG_RECOMPUTE.POINTWISE_COLS).vec
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            p, ds = _probability(s, dp, stats, owner_base, tile * CFG_RECOMPUTE.TILE_N, sq, skv, scale, causal, True, col, window_left)
            _q_ds(slab).base.subview(
                group * CFG_RECOMPUTE.TILE_M * CFG_RECOMPUTE.GROUP_COLS + row * CFG_RECOMPUTE.GROUP_COLS + chunk * CFG_RECOMPUTE.POINTWISE_COLS
            ).data_ptr().store_swizzled(ds.to(dtype), alignment=CFG_RECOMPUTE.POINTWISE_COLS * CFG_RECOMPUTE.BPE, swizzle=_SWIZZLE)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
        bars.q_score_empty[empty_state.idx].arrive()
        bars.ds_full[ds_state.idx].arrive()
        s_state = advance(s_state, CFG_RECOMPUTE.SINGLE_STAGE)
        empty_state = advance(empty_state, CFG_RECOMPUTE.SINGLE_STAGE)
        ds_state = advance(ds_state, CFG_RECOMPUTE.SINGLE_STAGE)
        ds_empty_state = advance(ds_empty_state, CFG_RECOMPUTE.SINGLE_STAGE)
    bars.q_ds_empty[ds_empty_state.idx].wait(ds_empty_state.phase)
    if hi > lo:
        bars.final_grad.wait(PipelineState.start().phase)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    _stage_gradient(tmem_base, CFG_RECOMPUTE.TMEM_G0, _operand_recompute(slab, CFG_RECOMPUTE.RESIDENT0, CFG_RECOMPUTE.TILE_M), hi > lo, dtype, group)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.out_ready.arrive()


@cute.jit
def _tma_warp(slab, bars, stream0_desc, stream1_desc, o_desc, lo, hi, head, batch, q_owner: cutlass.Constexpr[bool]):
    empty_state = PipelineState.start(phase=1)
    full_state = PipelineState.start()
    o_empty_state = PipelineState.start(phase=1)
    o_full_state = PipelineState.start()
    if cutlass.const_expr(q_owner):
        _wait_service(bars.q_stats_ready, PipelineState.start())
    for tile in cutlass.range(lo, hi, 1, unroll=1):
        _wait_service(bars.input_empty, empty_state)
        slot = full_state.idx
        bars.input_full[slot].arrive(n_bytes=2 * CFG_RECOMPUTE.TILE_N * CFG_RECOMPUTE.HEAD_DIM * CFG_RECOMPUTE.BPE, pred=nvvm.elect_sync())
        tma_load_tile(_input(slab, slot, 0), GmemTileTma(stream0_desc)(0, head, tile * CFG_RECOMPUTE.TILE_N, batch), bars.input_full[slot].smem_ptr)
        tma_load_tile(_input(slab, slot, 1), GmemTileTma(stream1_desc)(0, head, tile * CFG_RECOMPUTE.TILE_N, batch), bars.input_full[slot].smem_ptr)
        empty_state = advance(empty_state, CFG_RECOMPUTE.INPUT_STAGES)
        full_state = advance(full_state, CFG_RECOMPUTE.INPUT_STAGES)
        if cutlass.const_expr(not q_owner):
            _wait_service(bars.o_empty, o_empty_state)
            bars.o_full[o_full_state.idx].arrive(n_bytes=CFG_RECOMPUTE.TILE_N * CFG_RECOMPUTE.HEAD_DIM * CFG_RECOMPUTE.BPE, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_recompute(slab, CFG_RECOMPUTE.AUX, CFG_RECOMPUTE.TILE_N),
                GmemTileTma(o_desc)(0, head, tile * CFG_RECOMPUTE.TILE_N, batch),
                bars.o_full[o_full_state.idx].smem_ptr,
            )
            o_empty_state = advance(o_empty_state, CFG_RECOMPUTE.SINGLE_STAGE)
            o_full_state = advance(o_full_state, CFG_RECOMPUTE.SINGLE_STAGE)
    for _ in cutlass.range_constexpr(CFG_RECOMPUTE.INPUT_STAGES):
        _wait_service(bars.input_empty, empty_state)
        empty_state = advance(empty_state, CFG_RECOMPUTE.INPUT_STAGES)
    if cutlass.const_expr(not q_owner):
        _wait_service(bars.o_empty, o_empty_state)


@cute.jit
def _stats_warps(slab, stats, bars, lse, owner_base, lo, hi, batch, head, sq, q_owner: cutlass.Constexpr[bool]):
    if cutlass.const_expr(q_owner):
        _write_stats(
            _operand_recompute(slab, CFG_RECOMPUTE.AUX, CFG_RECOMPUTE.TILE_M),
            _operand_recompute(slab, CFG_RECOMPUTE.RESIDENT1, CFG_RECOMPUTE.TILE_M),
            stats,
            lse,
            batch,
            head,
            owner_base,
            sq,
            CFG_RECOMPUTE.TILE_M,
        )
        bars.q_stats_ready.arrive()
    else:
        row0 = cute.arch.thread_idx()[0] - CFG_RECOMPUTE.STATS_WARP * CFG_RECOMPUTE.WARP_LANES
        row1 = row0 + CFG_RECOMPUTE.STATS_THREADS
        input_state = PipelineState.start()
        empty_state = PipelineState.start(phase=1)
        full_state = PipelineState.start()
        o_full_state = PipelineState.start()
        o_empty_state = PipelineState.start()
        for tile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_service(bars.input_full, input_state)
            _wait_service(bars.o_full, o_full_state)
            lse0, delta0 = _read_stats_row(
                _operand_recompute(slab, CFG_RECOMPUTE.AUX, CFG_RECOMPUTE.TILE_N),
                _input(slab, input_state.idx, 1),
                lse,
                batch,
                head,
                tile * CFG_RECOMPUTE.TILE_N,
                sq,
                CFG_RECOMPUTE.TILE_N,
                row0,
            )
            lse1, delta1 = _read_stats_row(
                _operand_recompute(slab, CFG_RECOMPUTE.AUX, CFG_RECOMPUTE.TILE_N),
                _input(slab, input_state.idx, 1),
                lse,
                batch,
                head,
                tile * CFG_RECOMPUTE.TILE_N,
                sq,
                CFG_RECOMPUTE.TILE_N,
                row1,
            )
            bars.o_empty[o_empty_state.idx].arrive()
            _wait_service(bars.stats_empty, empty_state)
            stats.subview(row0).store(lse0)
            stats.subview(CFG_RECOMPUTE.TILE_N + row0).store(delta0)
            stats.subview(row1).store(lse1)
            stats.subview(CFG_RECOMPUTE.TILE_N + row1).store(delta1)
            bars.stats_full[full_state.idx].arrive()
            input_state = advance(input_state, CFG_RECOMPUTE.INPUT_STAGES)
            empty_state = advance(empty_state, CFG_RECOMPUTE.SINGLE_STAGE)
            full_state = advance(full_state, CFG_RECOMPUTE.SINGLE_STAGE)
            o_full_state = advance(o_full_state, CFG_RECOMPUTE.SINGLE_STAGE)
            o_empty_state = advance(o_empty_state, CFG_RECOMPUTE.SINGLE_STAGE)
        _wait_service(bars.stats_empty, empty_state)


@cute.jit
def _control_warp(slab, bars, out0_desc, out1_desc, owner_base, head, batch, q_owner: cutlass.Constexpr[bool]):
    _wait_service(bars.out_ready, PipelineState.start())
    if nvvm.elect_sync():
        tma_store_tile(_operand_recompute(slab, CFG_RECOMPUTE.RESIDENT0, CFG_RECOMPUTE.TILE_M), GmemTileTma(out0_desc)(0, head, owner_base, batch))
        if cutlass.const_expr(not q_owner):
            tma_store_tile(_operand_recompute(slab, CFG_RECOMPUTE.RESIDENT1, CFG_RECOMPUTE.TILE_M), GmemTileTma(out1_desc)(0, head, owner_base, batch))
        tma_store_commit()
        tma_store_wait(0)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _owner(
    resident0_desc,
    resident1_desc,
    o_desc,
    stream0_desc,
    stream1_desc,
    out0_desc,
    out1_desc,
    lse,
    slab,
    stats,
    bars,
    tmem_base,
    owner_base,
    batch,
    head,
    sq,
    skv,
    scale,
    causal: cutlass.Constexpr[bool],
    q_owner: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window_left: cutlass.Constexpr[int] = 0,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp == CFG_RECOMPUTE.TMA_WARP:
        nbytes = (3 if cutlass.const_expr(q_owner) else 2) * CFG_RECOMPUTE.TILE_M * CFG_RECOMPUTE.HEAD_DIM * CFG_RECOMPUTE.BPE
        bars.resident_full.arrive(n_bytes=nbytes, pred=nvvm.elect_sync())
        tma_load_tile(
            _operand_recompute(slab, CFG_RECOMPUTE.RESIDENT0, CFG_RECOMPUTE.TILE_M),
            GmemTileTma(resident0_desc)(0, head, owner_base, batch),
            bars.resident_full.smem_ptr,
        )
        tma_load_tile(
            _operand_recompute(slab, CFG_RECOMPUTE.RESIDENT1, CFG_RECOMPUTE.TILE_M),
            GmemTileTma(resident1_desc)(0, head, owner_base, batch),
            bars.resident_full.smem_ptr,
        )
        if cutlass.const_expr(q_owner):
            tma_load_tile(
                _operand_recompute(slab, CFG_RECOMPUTE.AUX, CFG_RECOMPUTE.TILE_M), GmemTileTma(o_desc)(0, head, owner_base, batch), bars.resident_full.smem_ptr
            )
    bars.resident_full.wait(PipelineState.start().phase)
    nvvm.barrier_cta_sync()
    lo = cutlass.Int32(0)
    if cutlass.const_expr(q_owner):
        hi = cute.ceil_div(skv, CFG_RECOMPUTE.TILE_N)
        if cutlass.const_expr(causal):
            hi = cute.math.min(hi, cute.ceil_div(owner_base + CFG_RECOMPUTE.TILE_M, CFG_RECOMPUTE.TILE_N))
        if cutlass.const_expr(window_left > 0):
            lo = cute.math.min(hi, cute.math.max(cutlass.Int32(0), owner_base - window_left + 1) // CFG_RECOMPUTE.TILE_N)
    else:
        hi = cute.ceil_div(sq, CFG_RECOMPUTE.TILE_N)
        if cutlass.const_expr(causal):
            lo = cute.math.min(owner_base // CFG_RECOMPUTE.TILE_N, hi)
        if cutlass.const_expr(window_left > 0):
            hi = cute.math.min(hi, cute.ceil_div(owner_base + CFG_RECOMPUTE.TILE_M + window_left - 1, CFG_RECOMPUTE.TILE_N))
    if warp < CFG_RECOMPUTE.COMPUTE_WARPS:
        if warp < CFG_RECOMPUTE.COMPUTE_WARPS // 2:
            if cutlass.const_expr(q_owner):
                _compute_q(slab, stats, bars, tmem_base, owner_base, lo, hi, sq, skv, scale, causal, dtype, 0, window_left)
            else:
                _compute_kv(slab, stats, bars, tmem_base, owner_base, lo, hi, sq, skv, scale, causal, dtype, 0, window_left)
        else:
            if cutlass.const_expr(q_owner):
                _compute_q(slab, stats, bars, tmem_base, owner_base, lo, hi, sq, skv, scale, causal, dtype, 1, window_left)
            else:
                _compute_kv(slab, stats, bars, tmem_base, owner_base, lo, hi, sq, skv, scale, causal, dtype, 1, window_left)
    elif warp == CFG_RECOMPUTE.TMA_WARP:
        _tma_warp(slab, bars, stream0_desc, stream1_desc, o_desc, lo, hi, head, batch, q_owner)
    elif warp == CFG_RECOMPUTE.MMA_WARP:
        if cutlass.const_expr(q_owner):
            _mma_q(slab, bars, tmem_base, lo, hi, dtype)
        else:
            _mma_kv(slab, bars, tmem_base, lo, hi, dtype)
    elif (warp >= CFG_RECOMPUTE.STATS_WARP) & (warp < CFG_RECOMPUTE.STATS_WARP + CFG_RECOMPUTE.STATS_WARPS):
        # Each stats warp owns independent state and its stats-empty tail wait.
        # The full count includes all 64 writers, bounding each waiter's reuse.
        _stats_warps(slab, stats, bars, lse, owner_base, lo, hi, batch, head, sq, q_owner)
        if warp == CFG_RECOMPUTE.CONTROL_WARP:
            _control_warp(slab, bars, out0_desc, out1_desc, owner_base, head, batch, q_owner)


@cute.kernel
def bprop_d128_kernel(
    q128: cutlass.GridConstant[tmap.TensorMap],
    k128: cutlass.GridConstant[tmap.TensorMap],
    v128: cutlass.GridConstant[tmap.TensorMap],
    o128: cutlass.GridConstant[tmap.TensorMap],
    do128: cutlass.GridConstant[tmap.TensorMap],
    q64: cutlass.GridConstant[tmap.TensorMap],
    k64: cutlass.GridConstant[tmap.TensorMap],
    v64: cutlass.GridConstant[tmap.TensorMap],
    o64: cutlass.GridConstant[tmap.TensorMap],
    do64: cutlass.GridConstant[tmap.TensorMap],
    dq: cutlass.GridConstant[tmap.TensorMap],
    dk: cutlass.GridConstant[tmap.TensorMap],
    dv: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window_left: cutlass.Constexpr[int] = 0,
):
    flat = cute.arch.block_idx()[0]
    q_tiles = cute.ceil_div(sq, CFG_RECOMPUTE.TILE_M)
    owners = q_tiles + cute.ceil_div(skv, CFG_RECOMPUTE.TILE_M)
    if cutlass.const_expr(causal):
        # Head/batch-fastest ranks interleave KV ascending with Q descending.
        batch_heads = lse.shape[0] * lse.shape[1]
        rank = flat // batch_heads
        hb = flat % batch_heads
        head = hb % lse.shape[1]
        batch = hb // lse.shape[1]
        kv_tiles = owners - q_tiles
        common = cute.math.min(q_tiles, kv_tiles)
        owner = cutlass.Int32(0)
        if rank < 2 * common:
            j = rank // 2
            if (rank & 1) != 0:
                owner = q_tiles - 1 - j
            else:
                owner = q_tiles + j
        elif q_tiles > kv_tiles:
            owner = q_tiles - 1 - common - (rank - 2 * common)
        else:
            owner = q_tiles + common + (rank - 2 * common)
    else:
        owner = flat % owners
        head = (flat // owners) % lse.shape[1]
        batch = flat // (owners * lse.shape[1])
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    slab = cutlass.Array(dtype, CFG_RECOMPUTE.SLAB_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    stats = cutlass.Array(cutlass.Float32, CFG_RECOMPUTE.STATS_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = make_bars()
    tmem_slot = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    if warp == 0:
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    if warp == CFG_RECOMPUTE.CONTROL_WARP:
        tmem_alloc(tmem_slot, CFG_RECOMPUTE.TMEM_COLS, nvvm.CTAGroup.CTA_1)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    tmem_base = tmem_slot.load()
    if owner < q_tiles:
        _owner(
            q128,
            do128,
            o128,
            k64,
            v64,
            dq,
            dv,
            lse,
            slab,
            stats,
            bars,
            tmem_base,
            owner * CFG_RECOMPUTE.TILE_M,
            batch,
            head,
            sq,
            skv,
            scale,
            causal,
            True,
            dtype,
            window_left,
        )
    else:
        _owner(
            k128,
            v128,
            o64,
            q64,
            do64,
            dk,
            dv,
            lse,
            slab,
            stats,
            bars,
            tmem_base,
            (owner - q_tiles) * CFG_RECOMPUTE.TILE_M,
            batch,
            head,
            sq,
            skv,
            scale,
            causal,
            False,
            dtype,
            window_left,
        )
    nvvm.barrier_cta_sync()
    if warp == CFG_RECOMPUTE.CONTROL_WARP:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem_dealloc(tmem_slot, CFG_RECOMPUTE.TMEM_COLS, nvvm.CTAGroup.CTA_1)


bprop_d128_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _tensor_map_recompute(tensor, rows):
    return tmap.create_tensor_map_tiled_from_view(
        tensor, box_dims=(1, rows, 1, CFG_RECOMPUTE.HEAD_GRANULE), stride_order=(3, 2, 1, 0), swizzle=tmap.TensorMapSwizzle.s128b
    )


@cute.jit
def host_recompute(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    window_left: cutlass.Constexpr[int] = 0,
):
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("dense d128 backward supports only BF16 and FP16")
    if cutlass.const_expr(q.shape[3] not in (64, 128) or k.shape[3] != q.shape[3]):
        raise ValueError("dense d128 backward requires equal head dimensions 64 or 128")
    if cutlass.const_expr(q.shape[2] != k.shape[2]):
        raise ValueError("dense d128 backward requires equal Q and KV head counts")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0):
        raise ValueError("dense d128 backward requires positive sequence lengths")
    q_tiles = cute.ceil_div(q.shape[1], CFG_RECOMPUTE.TILE_M)
    kv_tiles = cute.ceil_div(k.shape[1], CFG_RECOMPUTE.TILE_M)
    bprop_d128_kernel(
        _tensor_map_recompute(q, CFG_RECOMPUTE.TILE_M),
        _tensor_map_recompute(k, CFG_RECOMPUTE.TILE_M),
        _tensor_map_recompute(v, CFG_RECOMPUTE.TILE_M),
        _tensor_map_recompute(o, CFG_RECOMPUTE.TILE_M),
        _tensor_map_recompute(do, CFG_RECOMPUTE.TILE_M),
        _tensor_map_recompute(q, CFG_RECOMPUTE.TILE_N),
        _tensor_map_recompute(k, CFG_RECOMPUTE.TILE_N),
        _tensor_map_recompute(v, CFG_RECOMPUTE.TILE_N),
        _tensor_map_recompute(o, CFG_RECOMPUTE.TILE_N),
        _tensor_map_recompute(do, CFG_RECOMPUTE.TILE_N),
        _tensor_map_recompute(dq, CFG_RECOMPUTE.TILE_M),
        _tensor_map_recompute(dk, CFG_RECOMPUTE.TILE_M),
        _tensor_map_recompute(dv, CFG_RECOMPUTE.TILE_M),
        lse,
        q.shape[1],
        k.shape[1],
        scale,
        causal,
        q.element_type,
        window_left,
    ).launch(grid=((q_tiles + kv_tiles) * q.shape[2] * q.shape[0], 1, 1), block=(CFG_RECOMPUTE.THREADS, 1, 1), stream=stream)


# ======================================================================================================================
# section 3: the dense MHA ACCUMULATION body (3-launch chain, fp32 dQ workspace)
# (was kernels/sm100/bprop_d128_accum_f16.py)
# ======================================================================================================================
"""Dense d128 backward with FP32 dQ accumulation and native dK/dV stores.

The five-product pipeline retains 33 live mbarriers. Native output TMA stores
are warp-uniform, dQ reductions elected; both return 32-lane credits.

Sequence lengths need not be tile multiples: the tail Q / K / V / dO boxes
zero-fill, the tail dK / dV stores and dQ reductions clip, the tail Stats rows
publish -inf and the tail key rows are masked before the exponent.
"""

CFG_ACCUM = make_config(HEAD_DIM)
BAR_LAYOUT_ACCUM = make_bar_layout(CFG_ACCUM)
LOG2E = 1.4426950408889634
_DO_STEPS = CFG_ACCUM.D // CFG_ACCUM.K_HW // CFG_ACCUM.DO_SUBTILES


class BarsAccum(NamedTuple):
    k_empty: MBarrier
    k_full: MBarrier
    v_empty: MBarrier
    v_full: MBarrier
    do_empty: MBarrier
    do_full: MBarrier
    q_empty: MBarrier
    q_full: MBarrier
    s_done: MBarrier
    dp_done: MBarrier
    dq_done: MBarrier
    dp_empty: MBarrier
    p_ready: MBarrier
    ds_ready: MBarrier
    ds_smem: MBarrier
    dq_empty: MBarrier
    dq_full: MBarrier
    dv_full: MBarrier
    dk_full: MBarrier
    dvk_empty: MBarrier
    sched_full: MBarrier
    sched_empty: MBarrier


def _view(slab, offset, dtype, count, alignment=16):
    ptr = Pointer(slab.data_ptr() + offset, dtype=dtype)
    return cutlass.Array(ptr.ir_value(), shape=count, dtype=dtype, alignment=alignment)


def _bars_accum(slab):
    kinds = {"mma": Producer.MMA_COMMIT, "tma": Producer.TMA_LOAD, "thread": Producer.THREAD}
    return BarsAccum(
        *(
            MBarrier(_view(slab, off, cutlass.Int64, slots), stages=slots, init_count=count, producer=kinds[kind], try_wait=True)
            for _, off, slots, count, kind in BAR_LAYOUT_ACCUM
        )
    )


def _operand_accum(slab, offset, dtype, *, transpose=False, fp32=False):
    box = CFG_ACCUM.DQ_BOX if fp32 else CFG_ACCUM.HEAD_BOX
    return SmemTile(
        base=_view(slab, offset, dtype, CFG_ACCUM.M * CFG_ACCUM.D, alignment=1024),
        elems_per_stage=CFG_ACCUM.M * CFG_ACCUM.D,
        leading_byte_offset=CFG_ACCUM.M * CFG_ACCUM.SWIZZLE_BYTES if transpose else 16,
        stride_byte_offset=8 * CFG_ACCUM.SWIZZLE_BYTES,
        layout=CFG_ACCUM.SWIZZLE_LAYOUT,
        tma_loads_per_tile=CFG_ACCUM.D // box,
        tma_granu_elems=box,
        tma_subtile_stride_elems=CFG_ACCUM.M * box,
        desc_version=CFG_ACCUM.DESC_VERSION,
    )


def _mma_desc_accum(dtype, *, atranspose=False, btranspose=False, N=CFG_ACCUM.N, K=CFG_ACCUM.D):
    return MmaDesc(
        M=CFG_ACCUM.M,
        N=N,
        K=K,
        bpe_a=CFG_ACCUM.BPE,
        bpe_b=CFG_ACCUM.BPE,
        tile_k_hw=CFG_ACCUM.K_HW,
        atranspose=atranspose,
        btranspose=btranspose,
        cta_group=1,
        idesc=nvvm.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=dtype, b_dtype=dtype, m_dim=CFG_ACCUM.M, n_dim=N, a_major=int(atranspose), b_major=int(btranspose)
        ),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


@cute.jit
def _wait_accum(bar, phase):
    bar.wait(phase)
    # All full-warp consumers reconverge before any elected credit/issue.
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _wait_tcgen_accum(bar, phase):
    _wait_accum(bar, phase)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _commit(bar):
    if nvvm.elect_sync():
        bar.arrive(cta_group=1)


@cute.jit
def _credit_accum(bar):
    if nvvm.elect_sync():
        bar.arrive()


@cute.jit
def _tmem_load(address, n: cutlass.Constexpr[int], dtype: cutlass.Constexpr = cutlass.Float32):
    return nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(address, dtype), num=n)


@cute.jit
def _tmem_store(address, values):
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(address, cutlass.Int32), values)


@cute.jit
def tma_reduce_subtile(smem_tile, gmem_slice, subtile: cutlass.Constexpr[int]):
    """One elected source-sized FP32 add; caller owns commit and wait/credit."""
    desc = gmem_slice.tma_desc.get_ptr()
    coords = [gmem_slice.coord_d + cutlass.Int32(subtile * smem_tile.tma_granu_elems)] + list(gmem_slice.coords[1:])
    src = smem_tile.base.subview(subtile * smem_tile.tma_subtile_stride_elems)
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_tensor_reduce(desc, src, "add", coords)


@cute.jit
def _decode_flat(tile, nheads, nk, swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    group = tile // (swizzle * nk)
    inner = tile % (swizzle * nk)
    block = cutlass.Int32(0)
    residual = cutlass.Int32(0)
    if group < batch_heads // swizzle:
        block = inner // swizzle
        residual = inner % swizzle
    else:
        block = inner // max(batch_heads % swizzle, 1)
        residual = inner % max(batch_heads % swizzle, 1)
    hb = group * swizzle + residual
    return block, hb % nheads, hb // nheads


@cute.jit
def _next_tile_accum(sched, bars, state, nheads, nk, causal: cutlass.Constexpr[bool], swizzle: cutlass.Constexpr[int], batch_heads: cutlass.Constexpr[int]):
    _wait_accum(bars.sched_full[state.idx], state.phase)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block = x
        head = yz & 0xFFFF
        batch = (yz >> 16) & 0xFFFF
    return block, head, batch, valid, advance(state, CFG_ACCUM.SCHED_STAGES)


def _tensor_map_accum(tensor, box):
    # Logical TMA coordinates (d, sequence, head, batch), source-proved order.
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, CFG_ACCUM.M, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


@cute.jit
def _bounds_accum(block, sq, causal: cutlass.Constexpr[bool], window: cutlass.Constexpr[int] = 0):
    lo = cutlass.Int32(0)
    if cutlass.const_expr(causal):
        lo = block
    hi = (sq + CFG_ACCUM.N - 1) // CFG_ACCUM.N
    if cutlass.const_expr(window > 0):
        hi = cute.math.min(hi, (block * CFG_ACCUM.M + CFG_ACCUM.M + window - 1 + CFG_ACCUM.N - 1) // CFG_ACCUM.N)
    return lo, hi


@cute.jit
def _stats_accum(slab, lse, delta, row, group, qbase, sq, batch, head, scale):
    # Every one of256 lanes participates in both named barriers. Each group
    # writes a different128-float vector, one row per lane.
    named_barrier_fence(CFG_ACCUM.SOFTMAX_BARRIER, CFG_ACCUM.SOFTMAX_LANES)
    idx = qbase + row
    valid = idx < sq
    if group == 0:
        # Tail rows past S_q read the last Stats row (in bounds) and publish
        # -inf: their zero-filled Q / dO columns then yield exp2(-inf) = 0.
        clamped = cutlass.Int32(arith.select(valid.ir_value(), idx.ir_value(), (sq - 1).ir_value()))
        val = cutlass.Float32(lse[batch, head, clamped])
        stat = -cutlass.Float32(LOG2E) * val
        if val == cutlass.Float32(float("-inf")):
            stat = cutlass.Float32(0.0)
        stat = cutlass.Float32(arith.select(valid.ir_value(), stat.ir_value(), cutlass.Float32(float("-inf")).ir_value()))
        _view(slab, CFG_ACCUM.STATS, cutlass.Float32, CFG_ACCUM.N).subview(row).store(stat)
    else:
        # delta spans the rounded S_q; dot_do_o_kernel zeroes its tail rows.
        stat = -scale * cutlass.Float32(delta[batch, head, idx])
        _view(slab, CFG_ACCUM.DELTA, cutlass.Float32, CFG_ACCUM.N).subview(row).store(stat)
    named_barrier_fence(CFG_ACCUM.SOFTMAX_BARRIER, CFG_ACCUM.SOFTMAX_LANES)


@cute.jit
def _p_values_accum(
    slab, tmem_base, row, group, block, qtile, lo, skv, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, window: cutlass.Constexpr[int] = 0
):
    address = tmem_base + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scores = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    p = []
    packed = []
    scaled = scale * cutlass.Float32(LOG2E)
    neg_inf = cutlass.Float32(float("-inf"))
    kr = block * CFG_ACCUM.M + row
    # The tail KV tile's zero-filled key rows (kr >= skv) must not exponentiate:
    # exp2(-stat) overflows for a very negative LSE, and inf times their zero K
    # would put NaN into dQ.  Only the last KV tile pays the select.
    key_tail = (block + 1) * CFG_ACCUM.M > skv
    key_valid = kr < skv
    if cutlass.const_expr(window > 0):
        window_edge = (qtile + 1) * CFG_ACCUM.N - 1 - block * CFG_ACCUM.M >= window
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG_ACCUM.STATS, cutlass.Float32, CFG_ACCUM.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            s0, s1 = scores[i], scores[i + 1]
            qc = qtile * CFG_ACCUM.N + group * 64 + i
            if cutlass.const_expr(causal):
                if qtile == lo:
                    s0 = cutlass.Float32(arith.select((kr <= qc).ir_value(), s0.ir_value(), neg_inf.ir_value()))
                    s1 = cutlass.Float32(arith.select((kr <= qc + 1).ir_value(), s1.ir_value(), neg_inf.ir_value()))
            if cutlass.const_expr(window > 0):
                if window_edge:
                    s0 = cutlass.Float32(arith.select((kr + (window - 1) >= qc).ir_value(), s0.ir_value(), neg_inf.ir_value()))
                    s1 = cutlass.Float32(arith.select((kr + (window - 1) >= qc + 1).ir_value(), s1.ir_value(), neg_inf.ir_value()))
            if key_tail:
                s0 = cutlass.Float32(arith.select(key_valid.ir_value(), s0.ir_value(), neg_inf.ir_value()))
                s1 = cutlass.Float32(arith.select(key_valid.ir_value(), s1.ir_value(), neg_inf.ir_value()))
            e0, e1 = ffma2(scaled, scaled, s0, s1, stat[pair * 2], stat[pair * 2 + 1])
            p0 = cute.math.exp2(e0, fastmath=True)
            p1 = cute.math.exp2(e1, fastmath=True)
            p.extend((p0, p1))
            packed.append(fp32_to_fp16(p0, p1, dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))
    return cutlass.Vector.from_elements(tuple(p), cutlass.Float32)


@cute.jit
def _ds_values_accum(slab, tmem_base, group, p, scale, dtype: cutlass.Constexpr):
    address = tmem_base + 128 + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    dp = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    packed = []
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG_ACCUM.DELTA, cutlass.Float32, CFG_ACCUM.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            d0, d1 = ffma2(scale, scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            d0, d1 = fmul2(d0, d1, p[i], p[i + 1])
            packed.append(fp32_to_fp16(d0, d1, dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))


@cute.jit
def _softmax_loop_accum(
    slab,
    bars,
    sched,
    tmem_base,
    lse,
    delta,
    block,
    head,
    batch,
    sq,
    skv,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG_ACCUM.M
    group = tid // CFG_ACCUM.M
    state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        if cutlass.const_expr(not causal):
            _credit_accum(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds_accum(block, sq, causal, window)
        if lo < hi:
            _stats_accum(slab, lse, delta, row, group, lo * CFG_ACCUM.N, sq, batch, head, scale)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_accum(bars.s_done, state.phase)
            p = _p_values_accum(slab, tmem_base, row, group, block, qtile, lo, skv, scale, causal, dtype, window)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.p_ready.arrive()
            _wait_tcgen_accum(bars.dp_done, state.phase)
            _ds_values_accum(slab, tmem_base, group, p, scale, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_ready.arrive()
            if qtile + 1 < hi:
                _stats_accum(slab, lse, delta, row, group, (qtile + 1) * CFG_ACCUM.N, sq, batch, head, scale)
            state = advance(state, 1)
        if cutlass.const_expr(causal):
            _credit_accum(bars.sched_empty[sched_state.idx])
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _store_words_accum(slab, byte_offset, row, words):
    for vec in cutlass.range_constexpr(8):
        off = byte_offset + row * CFG_ACCUM.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        values = cutlass.Vector.from_elements(tuple(words[vec * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _store_dq_slice_accum(slab, row, values, index: cutlass.Constexpr[int]):
    for vec in cutlass.range_constexpr(8):
        off = CFG_ACCUM.ALIAS + index * CFG_ACCUM.SLICE_BYTES + row * CFG_ACCUM.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        v = cutlass.Vector.from_elements(tuple(values[index * 32 + vec * 4 + k] for k in range(4)), cutlass.Float32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Float32).store(v, alignment=16)


@cute.jit
def _native_epilogue_accum(slab, tmem_base, bars, row, live, empty_phase, dtype: cutlass.Constexpr, dk: cutlass.Constexpr[bool]):
    plane = 256 + CFG_ACCUM.D if cutlass.const_expr(dk) else 256
    base = CFG_ACCUM.ALIAS + (CFG_ACCUM.OPERAND_BYTES if cutlass.const_expr(dk) else 0)
    last_slice = (CFG_ACCUM.OPERAND_BYTES * (2 if cutlass.const_expr(dk) else 1)) // CFG_ACCUM.SLICE_BYTES - 1
    for half in cutlass.range_constexpr(CFG_ACCUM.D // 64):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + plane + half * 64, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
        if cutlass.const_expr(half == 0):
            _wait_accum(bars.dq_empty[last_slice], empty_phase)
        _store_words_accum(slab, base + half * CFG_ACCUM.SLICE_BYTES, row, words)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if cutlass.const_expr(dk):
        bars.dk_full.arrive()
    else:
        bars.dv_full.arrive()


@cute.jit
def _epilogue_loop_accum(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    row = cute.arch.thread_idx()[0] % CFG_ACCUM.M
    ds_state = PipelineState.start()
    dq_state = PipelineState.start()
    empty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds_accum(block, sq, causal, window)
        for _ in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_accum(bars.ds_ready, ds_state.phase)
            ds0 = _tmem_load(tmem_base + 128, 32, cutlass.Int32)
            ds1 = _tmem_load(tmem_base + 192, 32, cutlass.Int32)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            _wait_accum(bars.dq_empty[0], empty_state.phase)
            _store_words_accum(slab, CFG_ACCUM.ALIAS, row, ds0)
            _wait_accum(bars.dq_empty[1], empty_state.phase)
            _store_words_accum(slab, CFG_ACCUM.ALIAS + CFG_ACCUM.SLICE_BYTES, row, ds1)
            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_smem.arrive()
            _wait_tcgen_accum(bars.dq_done, dq_state.phase)
            dq0 = _tmem_load(tmem_base + 128, 64)
            if cutlass.const_expr(CFG_ACCUM.D == 128):
                dq1 = _tmem_load(tmem_base + 192, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.dp_empty.arrive()
            if cutlass.const_expr(CFG_ACCUM.D == 128):
                vals = cutlass.Vector.from_elements(tuple(dq0[i] for i in range(64)) + tuple(dq1[i] for i in range(64)), cutlass.Float32)
            else:
                vals = cutlass.Vector.from_elements(tuple(dq0[i] for i in range(64)), cutlass.Float32)
            for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                _wait_accum(bars.dq_empty[i], empty_state.phase)
                _store_dq_slice_accum(slab, row, vals, i)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.dq_full[i].arrive()
            ds_state = advance(ds_state, 1)
            dq_state = advance(dq_state, 1)
            empty_state = advance(empty_state, 1)
        if lo >= hi:
            _wait_tcgen_accum(bars.dq_done, dq_state.phase)
            dq_state = advance(dq_state, 1)
        # Observe every terminal source-credit phase before the next CLC
        # owner bootstraps these barriers again. Native dV/dK alone wait on
        # slices 1/3, leaving slices 0/2 unobserved after the final dQ or a
        # zero-trip owner. Each completed mbarrier phase needs a successful
        # wait before an arrival in its next phase.
        for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
            _wait_accum(bars.dq_empty[i], empty_state.phase)
        _native_epilogue_accum(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, False)
        _native_epilogue_accum(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, True)
        empty_state = advance(empty_state, 1)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _ts_accum(desc, plane, bdesc, dst, accum):
    for k in cutlass.range_constexpr(CFG_ACCUM.N // CFG_ACCUM.K_HW):
        mma_ts_step(
            desc, nvvm.make_tmem_ptr(plane + (k // 4) * 32, cutlass.Int8), bdesc, dst, k_idx=k, accumulate=accum if cutlass.const_expr(k == 0) else True
        )


@cute.jit
def _mma_loop_accum(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    score = _mma_desc_accum(dtype)
    grad = _mma_desc_accum(dtype, btranspose=True, N=CFG_ACCUM.D, K=CFG_ACCUM.N)
    dqdesc = _mma_desc_accum(dtype, atranspose=True, btranspose=True, N=CFG_ACCUM.D, K=CFG_ACCUM.M)
    kd = _operand_accum(slab, CFG_ACCUM.K, dtype).desc()
    vd = _operand_accum(slab, CFG_ACCUM.V, dtype).desc()
    dod = _operand_accum(slab, CFG_ACCUM.DO, dtype).desc()
    dot = _operand_accum(slab, CFG_ACCUM.DO, dtype, transpose=True).desc()
    dst = _operand_accum(slab, CFG_ACCUM.ALIAS, dtype, transpose=True).desc()
    kt = _operand_accum(slab, CFG_ACCUM.K, dtype, transpose=True).desc()
    s = nvvm.make_tmem_ptr(tmem_base, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + 128, cutlass.Float32)
    dv = nvvm.make_tmem_ptr(tmem_base + 256, cutlass.Float32)
    dk = nvvm.make_tmem_ptr(tmem_base + 256 + CFG_ACCUM.D, cutlass.Float32)
    q_state = PipelineState.start()
    do_state = PipelineState.start()
    kv_state = PipelineState.start()
    ready_state = PipelineState.start()
    dp_empty = PipelineState.start(phase=1)
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds_accum(block, sq, causal, window)
        if lo < hi:
            _wait_tcgen_accum(bars.k_full, kv_state.phase)
            _wait_tcgen_accum(bars.q_full[q_state.idx], q_state.phase)
            qd = _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype).desc()
            mma_ss(score, kd, qd, s)
            _commit(bars.s_done)
            _wait_tcgen_accum(bars.v_full, kv_state.phase)
        _wait_tcgen_accum(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_accum(bars.dp_empty, dp_empty.phase)
            for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                _wait_tcgen_accum(bars.do_full[half], do_state.phase)
                mma_ss(score, vd, dod, dp, k_start=half * _DO_STEPS, k_count=_DO_STEPS, accumulate=half > 0)
            _commit(bars.dp_done)
            _wait_tcgen_accum(bars.p_ready, ready_state.phase)
            _ts_accum(grad, tmem_base, dot, dv, qtile > lo)
            _commit(bars.do_empty)
            q_next = advance(q_state, CFG_ACCUM.Q_STAGES)
            if qtile + 1 < hi:
                _wait_tcgen_accum(bars.q_full[q_next.idx], q_next.phase)
                qd_next = _operand_accum(slab, CFG_ACCUM.Q + q_next.idx * CFG_ACCUM.OPERAND_BYTES, dtype).desc()
                mma_ss(score, kd, qd_next, s)
                _commit(bars.s_done)
            _wait_tcgen_accum(bars.ds_ready, ready_state.phase)
            qt = _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype, transpose=True).desc()
            _ts_accum(grad, tmem_base + 128, qt, dk, qtile > lo)
            _commit(bars.q_empty[q_state.idx])
            _wait_tcgen_accum(bars.ds_smem, ready_state.phase)
            mma_ss(dqdesc, dst, kt, dp)
            _commit(bars.dq_done)
            q_state = q_next
            do_state = advance(do_state, 1)
            ready_state = advance(ready_state, 1)
            dp_empty = advance(dp_empty, 1)
        if lo < hi:
            _commit(bars.k_empty)
            _commit(bars.v_empty)
            kv_state = advance(kv_state, 1)
        else:
            _commit(bars.dq_done)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _load_loop_accum(
    slab,
    bars,
    sched,
    q_desc,
    k_desc,
    v_desc,
    do_desc,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds_accum(block, sq, causal, window)
        if lo < hi:
            _wait_accum(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand_accum(slab, CFG_ACCUM.K, dtype), GmemTileTma(k_desc)(0, block * CFG_ACCUM.M, head, batch), bars.k_full.smem_ptr)
            _wait_accum(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand_accum(slab, CFG_ACCUM.V, dtype), GmemTileTma(v_desc)(0, block * CFG_ACCUM.M, head, batch), bars.v_full.smem_ptr)
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_accum(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype),
                GmemTileTma(q_desc)(0, qtile * CFG_ACCUM.N, head, batch),
                bars.q_full[q_state.idx].smem_ptr,
            )
            _wait_accum(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                bars.do_full[half].arrive(n_bytes=CFG_ACCUM.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                tma_load_subtiles(
                    _operand_accum(slab, CFG_ACCUM.DO, dtype), GmemTileTma(do_desc)(0, qtile * CFG_ACCUM.N, head, batch), bars.do_full[half].smem_ptr, half, 1
                )
            q_state = advance(q_state, CFG_ACCUM.Q_STAGES)
            do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


# DIAGNOSTIC lever ``diag_delete_dq_reduce_add`` (``Constexpr[bool]``, default False = the shipped kernel, rendered PTX-identical):
# the store warp skips the four fp32 dQ reduce-add issues per q tile and nothing else (every dq_full wait, commit, wait_group and
# dq_empty arrive stays: the barrier table is unchanged).  Armed, the OUTPUT IS WRONG BY DESIGN (dQ stays zero) -- a timing-only
# ceiling for the L2 reduce-add term; the dense MHA entry never arms it, the grouped dense kernel threads it from its adapter record.
# The THD twin's header (section 5 of this module) carries the full statement.
@cute.jit
def _store_loop_accum(
    slab,
    bars,
    sched,
    dq_desc,
    dk_desc,
    dv_desc,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    dq_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
            bars.dq_empty[i].arrive()  # all32 lanes, one bootstrap per KV tile
        lo, hi = _bounds_accum(block, sq, causal, window)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                _wait_accum(bars.dq_full[i], dq_state.phase)
                # DIAGNOSTIC D2 (default off = the shipped code): only the reduce-add issue is deleted; commit / wait / arrives stay.
                if cutlass.const_expr(not diag_delete_dq_reduce_add):
                    tma_reduce_subtile(
                        _operand_accum(slab, CFG_ACCUM.ALIAS, cutlass.Float32, fp32=True), GmemTileTma(dq_desc)(0, qtile * CFG_ACCUM.N, head, batch), i
                    )
                tma_store_commit()
            for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                tma_store_wait(CFG_ACCUM.DQ_SLICES - 1 - i)
                bars.dq_empty[i].arrive()  # all32 after corresponding source read
            dq_state = advance(dq_state, 1)
        _wait_accum(bars.dv_full, out_state.phase)
        # Exact backend distinction: native TMA store is warp-uniform, no elect.
        tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS, dtype), GmemTileTma(dv_desc)(0, block * CFG_ACCUM.M, head, batch))
        _wait_accum(bars.dk_full, out_state.phase)
        tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS + CFG_ACCUM.OPERAND_BYTES, dtype), GmemTileTma(dk_desc)(0, block * CFG_ACCUM.M, head, batch))
        out_state = advance(out_state, 1)
        tma_store_commit()
        tma_store_wait(0)
        bars.dvk_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _scheduler_loop_accum(bars, sched):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _wait_accum(bars.sched_empty[state.idx], state.phase)
        if nvvm.elect_sync():
            bars.sched_full[state.idx].arrive(n_bytes=16)
            nvvm.clusterlaunchcontrol_try_cancel(sched.tile_id_smem.subview(state.idx * 8), bars.sched_full[state.idx].smem_ptr, multicast=1)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        _wait_accum(bars.sched_full[state.idx], state.phase)
        _, _, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
        state = advance(state, CFG_ACCUM.SCHED_STAGES)


@cute.jit
def _pair_exp(scores, slab, group, scale):
    scaled = scale * cutlass.Float32(LOG2E)
    p = []
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG_ACCUM.STATS, cutlass.Float32, CFG_ACCUM.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            p.append(cute.math.exp2(e0, fastmath=True))
            p.append(cute.math.exp2(e1, fastmath=True))
    return cutlass.Vector.from_elements(tuple(p), cutlass.Float32)


@cute.jit
def _pair_masked_scores(scores, row, group, block, qtile, skv, causal: cutlass.Constexpr[bool], window: cutlass.Constexpr[int]):
    # Every enabled select is applied; a cell the select cannot mask keeps its score, so the body is entered only on
    # a tile that can hold a masked cell.
    neg_inf = cutlass.Float32(float("-inf"))
    kr = block * CFG_ACCUM.M + row
    out = []
    for i in cutlass.range_constexpr(64):
        v = scores[i]
        qc = qtile * CFG_ACCUM.N + group * 64 + i
        if cutlass.const_expr(causal):
            v = cutlass.Float32(arith.select((kr <= qc).ir_value(), v.ir_value(), neg_inf.ir_value()))
        if cutlass.const_expr(window > 0):
            v = cutlass.Float32(arith.select((kr + (window - 1) >= qc).ir_value(), v.ir_value(), neg_inf.ir_value()))
        v = cutlass.Float32(arith.select((kr < skv).ir_value(), v.ir_value(), neg_inf.ir_value()))
        out.append(v)
    return cutlass.Vector.from_elements(tuple(out), cutlass.Float32)


@cute.jit
def _pair_p_values(
    slab, tmem_base, row, group, block, qtile, lo, skv, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, window: cutlass.Constexpr[int] = 0
):
    address = tmem_base + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scores = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    masked = (block + 1) * CFG_ACCUM.M > skv
    if cutlass.const_expr(causal):
        masked = masked | (qtile == lo)
    if cutlass.const_expr(window > 0):
        masked = masked | ((qtile + 1) * CFG_ACCUM.N - 1 - block * CFG_ACCUM.M >= window)
    p = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
    if masked:
        p = _pair_exp(_pair_masked_scores(scores, row, group, block, qtile, skv, causal, window), slab, group, scale)
    else:
        p = _pair_exp(scores, slab, group, scale)
    packed = []
    for pair in cutlass.range_constexpr(32):
        packed.append(fp32_to_fp16(p[2 * pair], p[2 * pair + 1], dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))
    return p


@cute.jit
def _pair_bounds(block, sq, causal: cutlass.Constexpr[bool], window: cutlass.Constexpr[int] = 0):
    """The KV tile and this split's q-tile range: tile ``block`` = KV tile * SPLIT + split index."""
    kv = block // SPLIT
    part = block % SPLIT
    lo, hi = _bounds_accum(kv, sq, causal, window)
    n = hi - lo
    return kv, lo + (part * n) // SPLIT, lo + ((part + 1) * n) // SPLIT


@cute.jit
def _pair_partial_epilogue(slab, tmem_base, bars, row, live, empty_state):
    """fp32 dV then dK of this split through the dq slice ring (one credit phase per plane); returns the ring state."""
    for plane in cutlass.range_constexpr(2):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG_ACCUM.D)), cutlass.Float32)
        if live:
            if cutlass.const_expr(CFG_ACCUM.D == 128):
                a = _tmem_load(tmem_base + 256 + plane * CFG_ACCUM.D, 64)
                b = _tmem_load(tmem_base + 256 + plane * CFG_ACCUM.D + 64, 64)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                vals = cutlass.Vector.from_elements(tuple(a[i] for i in range(64)) + tuple(b[i] for i in range(64)), cutlass.Float32)
            else:
                a = _tmem_load(tmem_base + 256 + plane * CFG_ACCUM.D, 64)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                vals = cutlass.Vector.from_elements(tuple(a[i] for i in range(64)), cutlass.Float32)
        for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
            _wait_accum(bars.dq_empty[i], empty_state.phase)
            _store_dq_slice_accum(slab, row, vals, i)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.dq_full[i].arrive()
        empty_state = advance(empty_state, 1)
    return empty_state


@cute.jit
def _pair_softmax_loop(
    slab,
    bars,
    sched,
    tmem_base,
    lse,
    delta,
    block,
    head,
    batch,
    sq,
    skv,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG_ACCUM.M
    group = tid // CFG_ACCUM.M
    state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        if cutlass.const_expr(not causal):
            _credit_accum(bars.sched_empty[sched_state.idx])
        kv, lo, hi = _pair_bounds(block, sq, causal, window)
        if lo < hi:
            _stats_accum(slab, lse, delta, row, group, lo * CFG_ACCUM.N, sq, batch, head * RATIO, scale)
        for g in cutlass.range(0, RATIO, 1, unroll=1):
            qh = head * RATIO + g
            for qtile in cutlass.range(lo, hi, 1, unroll=1):
                _wait_tcgen_accum(bars.s_done, state.phase)
                p = _pair_p_values(slab, tmem_base, row, group, kv, qtile, lo, skv, scale, causal, dtype, window)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                bars.p_ready.arrive()
                _wait_tcgen_accum(bars.dp_done, state.phase)
                _ds_values_accum(slab, tmem_base, group, p, scale, dtype)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                bars.ds_ready.arrive()
                if qtile + 1 < hi:
                    _stats_accum(slab, lse, delta, row, group, (qtile + 1) * CFG_ACCUM.N, sq, batch, qh, scale)
                elif g + 1 < RATIO:
                    _stats_accum(slab, lse, delta, row, group, lo * CFG_ACCUM.N, sq, batch, qh + 1, scale)
                state = advance(state, 1)
        if cutlass.const_expr(causal):
            _credit_accum(bars.sched_empty[sched_state.idx])
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _pair_epilogue_loop(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    row = cute.arch.thread_idx()[0] % CFG_ACCUM.M
    dq_state = PipelineState.start()
    empty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        kv, lo, hi = _pair_bounds(block, sq, causal, window)
        # dq_done is committed once per KV tile after its last dK MMA: it orders the readout below after the MMAs.
        _wait_tcgen_accum(bars.dq_done, dq_state.phase)
        dq_state = advance(dq_state, 1)
        if cutlass.const_expr(SPLIT > 1):
            empty_state = _pair_partial_epilogue(slab, tmem_base, bars, row, lo < hi, empty_state)
        else:
            for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                _wait_accum(bars.dq_empty[i], empty_state.phase)
            _native_epilogue_accum(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, False)
            _native_epilogue_accum(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, True)
            empty_state = advance(empty_state, 1)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _pair_mma_loop(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    score = _mma_desc_accum(dtype)
    grad = _mma_desc_accum(dtype, btranspose=True, N=CFG_ACCUM.D, K=CFG_ACCUM.N)
    kd = _operand_accum(slab, CFG_ACCUM.K, dtype).desc()
    vd = _operand_accum(slab, CFG_ACCUM.V, dtype).desc()
    dod = _operand_accum(slab, CFG_ACCUM.DO, dtype).desc()
    dot = _operand_accum(slab, CFG_ACCUM.DO, dtype, transpose=True).desc()
    s = nvvm.make_tmem_ptr(tmem_base, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + 128, cutlass.Float32)
    dv = nvvm.make_tmem_ptr(tmem_base + 256, cutlass.Float32)
    dk = nvvm.make_tmem_ptr(tmem_base + 256 + CFG_ACCUM.D, cutlass.Float32)
    q_state = PipelineState.start()
    do_state = PipelineState.start()
    kv_state = PipelineState.start()
    ready_state = PipelineState.start()
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        kv, lo, hi = _pair_bounds(block, sq, causal, window)
        if lo < hi:
            _wait_tcgen_accum(bars.k_full, kv_state.phase)
            _wait_tcgen_accum(bars.q_full[q_state.idx], q_state.phase)
            qd = _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype).desc()
            mma_ss(score, kd, qd, s)
            _commit(bars.s_done)
            _wait_tcgen_accum(bars.v_full, kv_state.phase)
        _wait_tcgen_accum(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        first = cutlass.Int32(1)
        for g in cutlass.range(0, RATIO, 1, unroll=1):
            for qtile in cutlass.range(lo, hi, 1, unroll=1):
                for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                    _wait_tcgen_accum(bars.do_full[half], do_state.phase)
                    mma_ss(score, vd, dod, dp, k_start=half * _DO_STEPS, k_count=_DO_STEPS, accumulate=half > 0)
                _commit(bars.dp_done)
                _wait_tcgen_accum(bars.p_ready, ready_state.phase)
                _ts_accum(grad, tmem_base, dot, dv, first == 0)
                _commit(bars.do_empty)
                q_next = advance(q_state, CFG_ACCUM.Q_STAGES)
                if (qtile + 1 < hi) | (g + 1 < RATIO):
                    _wait_tcgen_accum(bars.q_full[q_next.idx], q_next.phase)
                    qd_next = _operand_accum(slab, CFG_ACCUM.Q + q_next.idx * CFG_ACCUM.OPERAND_BYTES, dtype).desc()
                    mma_ss(score, kd, qd_next, s)
                    _commit(bars.s_done)
                _wait_tcgen_accum(bars.ds_ready, ready_state.phase)
                qt = _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype, transpose=True).desc()
                _ts_accum(grad, tmem_base + 128, qt, dk, first == 0)
                _commit(bars.q_empty[q_state.idx])
                first = cutlass.Int32(0)
                q_state = q_next
                do_state = advance(do_state, 1)
                ready_state = advance(ready_state, 1)
        if lo < hi:
            _commit(bars.k_empty)
            _commit(bars.v_empty)
            kv_state = advance(kv_state, 1)
        _commit(bars.dq_done)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _pair_load_loop(
    slab,
    bars,
    sched,
    q_desc,
    k_desc,
    v_desc,
    do_desc,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        kv, lo, hi = _pair_bounds(block, sq, causal, window)
        if lo < hi:
            _wait_accum(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand_accum(slab, CFG_ACCUM.K, dtype), GmemTileTma(k_desc)(0, kv * CFG_ACCUM.M, head, batch), bars.k_full.smem_ptr)
            _wait_accum(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(_operand_accum(slab, CFG_ACCUM.V, dtype), GmemTileTma(v_desc)(0, kv * CFG_ACCUM.M, head, batch), bars.v_full.smem_ptr)
            kv_state = advance(kv_state, 1)
        for g in cutlass.range(0, RATIO, 1, unroll=1):
            qh = head * RATIO + g
            for qtile in cutlass.range(lo, hi, 1, unroll=1):
                _wait_accum(bars.q_empty[q_state.idx], q_state.phase)
                bars.q_full[q_state.idx].arrive(n_bytes=CFG_ACCUM.OPERAND_BYTES, pred=nvvm.elect_sync())
                tma_load_tile(
                    _operand_accum(slab, CFG_ACCUM.Q + q_state.idx * CFG_ACCUM.OPERAND_BYTES, dtype),
                    GmemTileTma(q_desc)(0, qtile * CFG_ACCUM.N, qh, batch),
                    bars.q_full[q_state.idx].smem_ptr,
                )
                _wait_accum(bars.do_empty, do_state.phase)
                for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                    bars.do_full[half].arrive(n_bytes=CFG_ACCUM.SLICE_BYTES, pred=nvvm.elect_sync())
                for half in cutlass.range_constexpr(CFG_ACCUM.DO_SUBTILES):
                    tma_load_subtiles(
                        _operand_accum(slab, CFG_ACCUM.DO, dtype), GmemTileTma(do_desc)(0, qtile * CFG_ACCUM.N, qh, batch), bars.do_full[half].smem_ptr, half, 1
                    )
                q_state = advance(q_state, CFG_ACCUM.Q_STAGES)
                do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _pair_store_loop(
    slab,
    bars,
    sched,
    dk_desc,
    dv_desc,
    block,
    head,
    batch,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG_ACCUM.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    out_state = PipelineState.start()
    full_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
            bars.dq_empty[i].arrive()
        kv = block // SPLIT
        if cutlass.const_expr(SPLIT > 1):
            # fp32 partials of this split: plane = batch * SPLIT + split index of the partial tensors
            plane = batch * SPLIT + block % SPLIT
            for rnd in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                    _wait_accum(bars.dq_full[i], full_state.phase)
                if cutlass.const_expr(rnd == 0):
                    tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS, cutlass.Float32, fp32=True), GmemTileTma(dv_desc)(0, kv * CFG_ACCUM.M, head, plane))
                else:
                    tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS, cutlass.Float32, fp32=True), GmemTileTma(dk_desc)(0, kv * CFG_ACCUM.M, head, plane))
                tma_store_commit()
                tma_store_wait(0)
                if cutlass.const_expr(rnd == 0):
                    # the next tile's bootstrap credits the slices after the second store
                    for i in cutlass.range_constexpr(CFG_ACCUM.DQ_SLICES):
                        bars.dq_empty[i].arrive()
                full_state = advance(full_state, 1)
        else:
            _wait_accum(bars.dv_full, out_state.phase)
            tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS, dtype), GmemTileTma(dv_desc)(0, kv * CFG_ACCUM.M, head, batch))
            _wait_accum(bars.dk_full, out_state.phase)
            tma_store_tile(_operand_accum(slab, CFG_ACCUM.ALIAS + CFG_ACCUM.OPERAND_BYTES, dtype), GmemTileTma(dk_desc)(0, kv * CFG_ACCUM.M, head, batch))
            out_state = advance(out_state, 1)
            tma_store_commit()
            tma_store_wait(0)
        bars.dvk_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.kernel
def bprop_d128_accum_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    nheads: cutlass.Int32,
    batch_heads: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    swizzle: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
    window: cutlass.Constexpr[int] = 0,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    nk = (skv + CFG_ACCUM.M - 1) // CFG_ACCUM.M
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block, head, batch = x, y, z
    slab = cutlass.Array(cutlass.Uint8, CFG_ACCUM.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars_accum(slab)
    tmem_slot = _view(slab, CFG_ACCUM.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG_ACCUM.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
        # All 512 TMEM columns belong to this CTA; use the portable allocation form.
        nvvm.tcgen05_alloc(tmem_slot, cutlass.Int32(CFG_ACCUM.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    tmem_base = tmem_slot.load()
    if cutlass.const_expr(DQ_OUT):
        if warp < 8:
            _softmax_loop_accum(
                slab, bars, sched, tmem_base, lse, delta, block, head, batch, sq, skv, nk, nheads, batch_heads, swizzle, scale, causal, dtype, window
            )
        elif warp < 12:
            _epilogue_loop_accum(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
        elif warp == 12:
            _mma_loop_accum(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
        elif warp == 13:
            _load_loop_accum(
                slab, bars, sched, q_desc, k_desc, v_desc, do_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window
            )
        elif warp == 14:
            _store_loop_accum(
                slab, bars, sched, dq_desc, dk_desc, dv_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, False, window
            )
        else:
            _scheduler_loop_accum(bars, sched)
    else:
        if warp < 8:
            _pair_softmax_loop(
                slab, bars, sched, tmem_base, lse, delta, block, head, batch, sq, skv, nk, nheads, batch_heads, swizzle, scale, causal, dtype, window
            )
        elif warp < 12:
            _pair_epilogue_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
        elif warp == 12:
            _pair_mma_loop(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
        elif warp == 13:
            _pair_load_loop(slab, bars, sched, q_desc, k_desc, v_desc, do_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, window)
        elif warp == 14:
            _pair_store_loop(slab, bars, sched, dk_desc, dv_desc, block, head, batch, nk, nheads, batch_heads, swizzle, causal, dtype)
        else:
            _scheduler_loop_accum(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG_ACCUM.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host_accum(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: Optional[cute.Tensor],
    dk: cute.Tensor,
    dv: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    window: cutlass.Constexpr[int] = 0,
):
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("dense d128 accumulation requires BF16/FP16")
    if cutlass.const_expr(q.shape[3] != CFG_ACCUM.D or k.shape[3] != CFG_ACCUM.D or q.shape[2] != k.shape[2] * RATIO):
        raise ValueError("dense d128 accumulation requires dense MHA d128")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0):
        raise ValueError("dense d128 accumulation requires positive Sq/Skv")
    if cutlass.const_expr(SPLIT > 1 and (causal or window > 0)):
        raise ValueError("the q-range split serves unmasked tiles only")
    if cutlass.const_expr(SPLIT > 1 and (dk.element_type != cutlass.Float32 or dv.element_type != cutlass.Float32)):
        raise ValueError("the q-range split writes fp32 dK / dV partials")
    nk = (k.shape[1] + CFG_ACCUM.M - 1) // CFG_ACCUM.M
    heads_per_l2 = (50 * 1024 * 1024) // (q.shape[1] * (CFG_ACCUM.D + CFG_ACCUM.D) * CFG_ACCUM.BPE * RATIO)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nk * k.shape[2] * q.shape[0], 1, 1) if causal else (nk * SPLIT, k.shape[2], q.shape[0])
    out_box = CFG_ACCUM.DQ_BOX if SPLIT > 1 else CFG_ACCUM.HEAD_BOX
    bprop_d128_accum_kernel(
        _tensor_map_accum(q, CFG_ACCUM.HEAD_BOX),
        _tensor_map_accum(k, CFG_ACCUM.HEAD_BOX),
        _tensor_map_accum(v, CFG_ACCUM.HEAD_BOX),
        _tensor_map_accum(do, CFG_ACCUM.HEAD_BOX),
        _tensor_map_accum(dq, CFG_ACCUM.DQ_BOX) if DQ_OUT else _tensor_map_accum(dk, out_box),
        _tensor_map_accum(dk, out_box),
        _tensor_map_accum(dv, out_box),
        lse,
        delta,
        q.shape[1],
        k.shape[1],
        k.shape[2],
        q.shape[0] * k.shape[2],
        scale,
        causal,
        swizzle,
        q.element_type,
        window,
    ).launch(grid=grid, block=(CFG_ACCUM.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)


# ======================================================================================================================
# section 4: the THD chain kernels (setup / delta_zero / cast_dq) and the THD chain host
# (was kernels/sm100/bprop_d128_accum_thd_helpers.py)
# ======================================================================================================================
"""Native THD metadata/maps, Delta+dQ initialization, and dQ conversion.

All caller ports retain their independent ragged origins. Compact prefixes
address only the internal Delta and FP32 dQ regions. Lengths remain on device;
launch bounds and physical capacities are runtime declaration metadata.

Grouped chain, op-level field ``dkv_accum = "fp32_reduce"``: ``delta_zero_thd_d128_kernel`` also zero-fills the per-KV-head fp32 dK / dV
accumulators over each sequence's live kv rows and ``cast_dq_thd_d128_kernel`` also converts them once to the IO dtype at each output's own
ragged origin -- both through the Optional trailing tensors (``None`` = dropped from the signature: the MHA chain and the default grouped
chain render byte-identically); the grid's row extent then covers max(max_sq, max_skv) rows and z = Hq (the kv heads are the first Hkv).
"""


def _origin_row(specs, port):
    return -1 if specs[port] is None else specs[port][0]


@cute.jit
def _zero_group_vector(acc, token, head, column):
    """One lane's eight fp32 of a per-KV-head accumulator row (``acc`` = ``[1, rounded_k, Hkv, 128]`` fp32) set to exact zero (two 16-B stores)."""
    zero4 = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(4)), cutlass.Float32)
    offset = token * cutlass.Int64(acc.stride[1]) + cutlass.Int64(head) * cutlass.Int64(acc.stride[2]) + column
    (acc.iterator.raw_ptr() + offset).store(zero4, alignment=16)
    (acc.iterator.raw_ptr() + offset + 4).store(zero4, alignment=16)


@cute.jit
def _cast_group_vector(acc, out, origins, origin: cutlass.Constexpr[int], batch, start, row, head, column, io_dtype: cutlass.Constexpr):
    """One lane's eight fp32 of a per-KV-head accumulator row -> eight ``io_dtype`` values of the output's own ragged-origin row (the cast_dq
    form: the group's ONE rounding at the output)."""
    src_token = cutlass.Int64(start) + cutlass.Int64(row)
    dst_token = thd_port_origin(origins, origin, batch, start) + cutlass.Int64(row)
    src_offset = src_token * cutlass.Int64(acc.stride[1]) + cutlass.Int64(head) * cutlass.Int64(acc.stride[2]) + column
    dst_offset = dst_token * cutlass.Int64(out.stride[1]) + cutlass.Int64(head) * cutlass.Int64(out.stride[2]) + column
    lo = (acc.iterator.raw_ptr() + src_offset).load(count=4, alignment=16)
    hi = (acc.iterator.raw_ptr() + src_offset + 4).load(count=4, alignment=16)
    packed = cutlass.Vector.from_elements(
        (
            fp32_to_fp16(lo[0], lo[1], dtype=io_dtype),
            fp32_to_fp16(lo[2], lo[3], dtype=io_dtype),
            fp32_to_fp16(hi[0], hi[1], dtype=io_dtype),
            fp32_to_fp16(hi[2], hi[3], dtype=io_dtype),
        ),
        cutlass.Int32,
    )
    Pointer(cutlass.make_array_view(out).data_ptr() + dst_offset, dtype=cutlass.Int32).store(packed, alignment=16)


@cute.kernel
def setup_thd_d128_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq_accum: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    maps: cute.Tensor,
    q_lens: cute.Tensor,
    kv_lens: cute.Tensor,
    ro_q: Optional[cute.Tensor],
    ro_k: Optional[cute.Tensor],
    ro_v: Optional[cute.Tensor],
    ro_o: Optional[cute.Tensor],
    ro_do: Optional[cute.Tensor],
    ro_dq: Optional[cute.Tensor],
    ro_dk: Optional[cute.Tensor],
    ro_dv: Optional[cute.Tensor],
    ro_stats: Optional[cute.Tensor],
    origin_specs: cutlass.Constexpr,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
    partial_dkv: cutlass.Constexpr[bool] = False,
):
    """Publish prefixes/origins, then seven sequence-local map arrays.

    Prefix lengths and port origins are separate facts. The full block barrier
    publishes both to the elected, disjoint per-sequence descriptor writers.
    Every map writer releases to the tensor-map proxy; the consuming main
    kernel acquires before use. Descriptor sequence ordinal is one for this
    body's explicit (D, sequence, head, batch) coordinate order. Grouped
    partials use compact KV origins while retaining all external origin rows
    for the final fold; the default MHA specialization is unchanged.
    """
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp, lane = tid // 32, tid % 32
    data = cutlass.make_array_view(meta)
    if n_batch <= 1:
        if tid == 0:
            write_thd_meta(data, cutlass.make_array_view(q_lens), cutlass.make_array_view(kv_lens), lens_form, n_batch)
    else:
        if warp == 0:
            write_thd_prefix_warp(data, cutlass.make_array_view(q_lens), n_batch, n_batch, (lens_form & 1) != 0, lane, store_lengths=False)
        if warp == 1:
            write_thd_prefix_warp(data, cutlass.make_array_view(kv_lens), n_batch, 2 * n_batch + 1, (lens_form & 2) != 0, lane, store_lengths=True)
    offsets = (ro_q, ro_k, ro_v, ro_o, ro_do, ro_dq, ro_dk, ro_dv, ro_stats)
    for port in cutlass.range_constexpr(9):
        if cutlass.const_expr(origin_specs[port] is not None):
            spec = origin_specs[port]
            write_thd_port_origins(origins, offsets[port], spec[0], spec[1], spec[2], n_batch, tid, cutlass.Int32(THD_SETUP_THREADS))
    cute.arch.barrier()
    if nvvm.elect_sync():
        cu_k = 2 * n_batch + 1
        batch_step = cutlass.Int32(THD_SETUP_THREADS // 32)
        emit_seq_descs(
            q_desc,
            maps,
            data,
            n_batch,
            q,
            n_batch,
            cutlass.Int64(q.stride[1]),
            1,
            slot_base=MAP_Q_THD * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 0),
        )
        emit_seq_descs(
            k_desc,
            maps,
            data,
            cu_k,
            k,
            n_batch,
            cutlass.Int64(k.stride[1]),
            1,
            slot_base=MAP_K_THD * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 1),
        )
        emit_seq_descs(
            v_desc,
            maps,
            data,
            cu_k,
            v,
            n_batch,
            cutlass.Int64(v.stride[1]),
            1,
            slot_base=MAP_V_THD * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 2),
        )
        emit_seq_descs(
            do_desc,
            maps,
            data,
            n_batch,
            do,
            n_batch,
            cutlass.Int64(do.stride[1]),
            1,
            slot_base=MAP_DO_THD * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 4),
        )
        emit_seq_descs(
            dq_desc,
            maps,
            data,
            n_batch,
            dq_accum,
            n_batch,
            cutlass.Int64(dq_accum.stride[1]),
            1,
            slot_base=MAP_DQ_THD * n_batch,
            first_batch=warp,
            batch_step=batch_step,
        )
        emit_seq_descs(
            dk_desc,
            maps,
            data,
            cu_k,
            dk,
            n_batch,
            cutlass.Int64(dk.stride[1]),
            1,
            slot_base=MAP_DK_THD * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=-1 if partial_dkv else _origin_row(origin_specs, 6),
        )
        emit_seq_descs(
            dv_desc,
            maps,
            data,
            cu_k,
            dv,
            n_batch,
            cutlass.Int64(dv.stride[1]),
            1,
            slot_base=MAP_DV_THD * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=-1 if partial_dkv else _origin_row(origin_specs, 7),
        )
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


setup_thd_d128_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def delta_zero_thd_d128_kernel(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    dq_accum: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    n_batch: cutlass.Int32,
    origin_specs: cutlass.Constexpr,
    dk_acc: Optional[cute.Tensor] = None,
    dv_acc: Optional[cute.Tensor] = None,
):
    """Eight rows per CTA; all lanes participate in the unchanged FP32 dot tree.

    ``dk_acc`` / ``dv_acc`` (dkv_accum = fp32_reduce; ``None`` = dropped, the shipping rendering): the per-KV-head fp32 accumulators are
    zero-filled over the sequence's live kv rows by the blocks whose head index is a kv head (the grid covers max(max_sq, max_skv) rows)."""
    block, batch, head = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    row = cutlass.Int32(block) * ROWS_PER_CTA + cutlass.Int32(tid // LANES_PER_ROW)
    column = cutlass.Int64(tid % LANES_PER_ROW) * ELEMENTS_PER_LANE
    data = cutlass.make_array_view(meta)
    start = cutlass.Int32(data[n_batch + batch])
    length = cutlass.Int32(data[n_batch + batch + 1]) - start
    live = row < length
    do_values = cutlass.Vector.from_elements(tuple(do.element_type(0.0) for _ in range(ELEMENTS_PER_LANE)), do.element_type)
    o_values = cutlass.Vector.from_elements(tuple(o.element_type(0.0) for _ in range(ELEMENTS_PER_LANE)), o.element_type)
    if live:
        token = cutlass.Int64(start) + cutlass.Int64(row)
        dq_offset = token * cutlass.Int64(dq_accum.stride[1]) + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2]) + column
        zero4 = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(4)), cutlass.Float32)
        (dq_accum.iterator.raw_ptr() + dq_offset).store(zero4, alignment=16)
        (dq_accum.iterator.raw_ptr() + dq_offset + 4).store(zero4, alignment=16)
        o_token = thd_port_origin(origins, _origin_row(origin_specs, 3), batch, start) + cutlass.Int64(row)
        do_token = thd_port_origin(origins, _origin_row(origin_specs, 4), batch, start) + cutlass.Int64(row)
        o_offset = o_token * cutlass.Int64(o.stride[1]) + cutlass.Int64(head) * cutlass.Int64(o.stride[2]) + column
        do_offset = do_token * cutlass.Int64(do.stride[1]) + cutlass.Int64(head) * cutlass.Int64(do.stride[2]) + column
        do_values = (do.iterator.raw_ptr() + do_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
        o_values = (o.iterator.raw_ptr() + o_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
    pairs = []
    for j in cutlass.range_constexpr(ELEMENTS_PER_LANE // 2):
        high = cute.math.mul(do_values[2 * j + 1].to(cutlass.Float32), o_values[2 * j + 1].to(cutlass.Float32), rounding=RN, ftz=True)
        pairs.append(cute.math.fma(do_values[2 * j].to(cutlass.Float32), o_values[2 * j].to(cutlass.Float32), high, rounding=RN, ftz=True))
    dot = cute.math.add(pairs[0], pairs[1], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[2], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[3], rounding=RN, ftz=True)
    # XOR distances 8,4,2,1 never cross a 16-lane row. Inactive row lanes
    # carry zeros but remain active for the full-warp shuffle mask.
    for step in cutlass.range_constexpr(4):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=dot, offset=1 << (3 - step), mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        dot = cute.math.add(dot, peer, rounding=RN, ftz=True)
    dot = cute.math.mul(dot, cutlass.Float32(1.0), rounding=RN, ftz=True)
    if live and tid % LANES_PER_ROW == 0:
        token = cutlass.Int64(start) + cutlass.Int64(row)
        delta_offset = cutlass.Int64(head) * cutlass.Int64(delta.stride[1]) + token * cutlass.Int64(delta.stride[2])
        (delta.iterator.raw_ptr() + delta_offset).store(dot)
    if cutlass.const_expr(dk_acc is not None):
        # dkv_accum = fp32_reduce: an exact-zero fp32 image for the main kernel's reduce-adds; kv rows past the sequence are never written.
        # fill dk/dv_acc (token_k, head) for head < Hkv (one CTA per kv head and row block); grid x = max(q, kv)
        kv_length = cutlass.Int32(data[batch])
        kv_start = cutlass.Int32(data[2 * n_batch + 1 + batch])
        if head < dk_acc.shape[2] and row < kv_length:
            kv_token = cutlass.Int64(kv_start) + cutlass.Int64(row)
            _zero_group_vector(dk_acc, kv_token, head, column)
            _zero_group_vector(dv_acc, kv_token, head, column)


delta_zero_thd_d128_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def cast_dq_thd_d128_kernel(
    dq_accum: cute.Tensor,
    dq_out: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    n_batch: cutlass.Int32,
    dq_origin: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    dk_acc: Optional[cute.Tensor] = None,
    dv_acc: Optional[cute.Tensor] = None,
    dk_out: Optional[cute.Tensor] = None,
    dv_out: Optional[cute.Tensor] = None,
    dk_origin: cutlass.Constexpr[int] = -1,
    dv_origin: cutlass.Constexpr[int] = -1,
):
    """Cast compact FP32 dQ once into each external sequence's own native rows.

    ``dk_acc`` / ``dv_acc`` -> ``dk_out`` / ``dv_out`` (dkv_accum = fp32_reduce; ``None`` = dropped, the shipping rendering): the per-KV-head fp32
    dK / dV accumulators are converted ONCE to the IO dtype into each output's own ragged-origin rows in the same streaming pass (the blocks
    whose head index is a kv head; the grid covers max(max_sq, max_skv) rows) -- the fold launch's job without its read of 2 x Hq / Hkv partials."""
    block, batch, head = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    row = cutlass.Int32(block) * ROWS_PER_CTA + cutlass.Int32(tid // LANES_PER_ROW)
    column = cutlass.Int64(tid % LANES_PER_ROW) * ELEMENTS_PER_LANE
    data = cutlass.make_array_view(meta)
    start = cutlass.Int32(data[n_batch + batch])
    length = cutlass.Int32(data[n_batch + batch + 1]) - start
    if row < length:
        src_token = cutlass.Int64(start) + cutlass.Int64(row)
        dst_token = thd_port_origin(origins, dq_origin, batch, start) + cutlass.Int64(row)
        src_offset = src_token * cutlass.Int64(dq_accum.stride[1]) + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2]) + column
        dst_offset = dst_token * cutlass.Int64(dq_out.stride[1]) + cutlass.Int64(head) * cutlass.Int64(dq_out.stride[2]) + column
        lo = (dq_accum.iterator.raw_ptr() + src_offset).load(count=4, alignment=16)
        hi = (dq_accum.iterator.raw_ptr() + src_offset + 4).load(count=4, alignment=16)
        packed = cutlass.Vector.from_elements(
            (
                fp32_to_fp16(lo[0], lo[1], dtype=io_dtype),
                fp32_to_fp16(lo[2], lo[3], dtype=io_dtype),
                fp32_to_fp16(hi[0], hi[1], dtype=io_dtype),
                fp32_to_fp16(hi[2], hi[3], dtype=io_dtype),
            ),
            cutlass.Int32,
        )
        Pointer(cutlass.make_array_view(dq_out).data_ptr() + dst_offset, dtype=cutlass.Int32).store(packed, alignment=16)
    if cutlass.const_expr(dk_acc is not None):
        # convert 2x512 B fp32 -> 2x256 B bf16 per (token_k, kv head) at the fold's origins
        kv_length = cutlass.Int32(data[batch])
        kv_start = cutlass.Int32(data[2 * n_batch + 1 + batch])
        if head < dk_acc.shape[2] and row < kv_length:
            _cast_group_vector(dk_acc, dk_out, origins, dk_origin, batch, kv_start, row, head, column, io_dtype)
            _cast_group_vector(dv_acc, dv_out, origins, dv_origin, batch, kv_start, row, head, column, io_dtype)


cast_dq_thd_d128_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host_thd(
    q,
    k,
    v,
    o,
    do,
    lse,
    dq,
    dk,
    dv,
    dq_accum,
    delta,
    meta,
    origins,
    maps,
    q_lens,
    kv_lens,
    ro,
    origin_specs: cutlass.Constexpr,
    lens_form,
    batch: cutlass.Constexpr[int],
    max_sq,
    max_skv,
    scale,
    causal: cutlass.Constexpr[bool],
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    """Four declared native launches; no host reads, tensor copies or allocation."""
    setup_thd_d128_kernel(
        _tensor_map_thd(q, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(k, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(v, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(do, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(dq_accum, CFG_ACCUM_D128.DQ_BOX),
        _tensor_map_thd(dk, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(dv, CFG_ACCUM_D128.HEAD_BOX),
        q,
        k,
        v,
        do,
        dq_accum,
        dk,
        dv,
        meta,
        origins,
        maps,
        q_lens,
        kv_lens,
        *ro,
        origin_specs,
        lens_form,
        cutlass.Int32(batch),
    ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
    q_grid = cute.math.max((max_sq + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
    delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs).launch(
        grid=(q_grid, batch, q.shape[2]),
        block=(THREADS, 1, 1),
        stream=stream,
    )
    host_accum_thd(
        maps,
        lse,
        delta,
        meta,
        origins,
        cutlass.Int32(batch),
        q.shape[2],
        max_skv,
        _origin_row(origin_specs, 8),
        scale,
        causal,
        q.element_type,
        stream,
        diag_delete_dq_reduce_add,
    )
    cast_dq_thd_d128_kernel(dq_accum, dq, meta, origins, cutlass.Int32(batch), _origin_row(origin_specs, 5), q.element_type).launch(
        grid=(q_grid, batch, q.shape[2]),
        block=(THREADS, 1, 1),
        stream=stream,
    )


# ======================================================================================================================
# section 5: the packed THD MHA accumulation body
# (was kernels/sm100/bprop_d128_accum_thd_f16.py)
# ======================================================================================================================
"""Native packed THD MHA d128 backward with FP32 dQ accumulation.

Sequence-local tensor maps clip every input and output tail, including caller
ports with independent ragged origins. The dense five-product pipeline and all
33 mbarriers retain their lane counts. Natural CLC visits a declared envelope;
invalid owners take the complete zero-trip protocol without touching a map.

DIAGNOSTIC lever ``diag_delete_dq_reduce_add`` (``Constexpr[bool]``, default False = the shipped kernel, rendered
PTX-identical; pinned by ``test_sdpa_bwd_d128_diag_dq_reduce_add_sm100.py``): the store warp skips the four
``tma_reduce_add_subtile`` issues of the fp32 dQ slices per q tile and nothing else -- every ``dq_full`` wait,
``tma_store_commit``, ``tma_store_wait`` and ``dq_empty`` arrive stays, so the barrier table (arrive / wait sites and
their lane sums) is unchanged.  Armed, the OUTPUT IS WRONG BY DESIGN (dQ stays zero): a timing-only ceiling for the
L2 reduce-add term, never selected by a graph, never shipped armed.  Threaded from the adapter record
(``SdpaBwdDslSm100D128.diag_delete_dq_reduce_add``) through ``compile_host`` -> ``host`` -> the kernel.

Op-level field ``dkv_accum = "fp32_reduce"`` (``Constexpr[bool]`` ``dkv_fp32`` on ``_epilogue_loop`` / ``_store_loop``, default False = the
shipped kernel, rendered PTX-identical; set by the GROUPED THD kernel only, threaded from ``SdpaBwdDslSm100D128.dkv_accum`` through the grouped
chain; the MHA kernel never sets it): the epilogue stages the fp32 dV then dK accumulator tiles as the four [M x DQ_BOX] fp32 SW128 slices of
the ALIAS slab (the dQ staging image, ``_native_epilogue_fp32``) instead of the bf16 pack, and the store warp reduce-adds each slice
(``tma_reduce_add_subtile`` add.f32, the dQ slices' primitive) into the per-KV-head fp32 workspace maps carried by the MAP_DV / MAP_DK slots
(kv head = head // ratio).  Barrier rows touched: dv_full / dk_full keep their lanes and sites (the published bytes are fp32 slices); dq_empty
gains ONE release per item on both sides -- the store warp arrives dq_empty[i] after the dV slice i source read (``tma_store_wait(3 - i)``, P7)
and the epilogue's dK round waits dq_empty[i] at that phase before overwriting slice i (the dK staging's WAR edge), advancing its empty_state
once more per item; the bootstrap, the terminal observation, dvk_empty and every other row are unchanged.  A zero-trip owner stages zeros and
issues no reduce-add (the workspace is zero-filled); every commit / wait / arrive is kept.  A ROUNDING CHANGE at the chain level (fp32 group
accumulation in hardware order replaces 8 bf16 partials summed in a fixed order), an op attribute: the test module
``test_sdpa_bwd_d128_dkv_accum_sm100.py`` pins the default rendering and the fp32 census.
"""

MAP_Q_THD, MAP_K_THD, MAP_V_THD, MAP_DO_THD, MAP_DQ_THD, MAP_DK_THD, MAP_DV_THD = range(7)
MAP_COUNT_THD = 7


class BarsThd(NamedTuple):
    k_empty: MBarrier
    k_full: MBarrier
    v_empty: MBarrier
    v_full: MBarrier
    do_empty: MBarrier
    do_full: MBarrier
    q_empty: MBarrier
    q_full: MBarrier
    s_done: MBarrier
    dp_done: MBarrier
    dq_done: MBarrier
    dp_empty: MBarrier
    p_ready: MBarrier
    ds_ready: MBarrier
    ds_smem: MBarrier
    dq_empty: MBarrier
    dq_full: MBarrier
    dv_full: MBarrier
    dk_full: MBarrier
    dvk_empty: MBarrier
    sched_full: MBarrier
    sched_empty: MBarrier


def _bars_thd(slab):
    kinds = {"mma": Producer.MMA_COMMIT, "tma": Producer.TMA_LOAD, "thread": Producer.THREAD}
    return BarsThd(
        *(
            MBarrier(_view(slab, off, cutlass.Int64, slots), stages=slots, init_count=count, producer=kinds[kind], try_wait=True)
            for _, off, slots, count, kind in BAR_LAYOUT_ACCUM_D128
        )
    )


def _operand_thd(slab, offset, dtype, *, transpose=False, fp32=False):
    box = CFG_ACCUM_D128.DQ_BOX if fp32 else CFG_ACCUM_D128.HEAD_BOX
    return SmemTile(
        base=_view(slab, offset, dtype, CFG_ACCUM_D128.M * CFG_ACCUM_D128.D, alignment=1024),
        elems_per_stage=CFG_ACCUM_D128.M * CFG_ACCUM_D128.D,
        leading_byte_offset=CFG_ACCUM_D128.M * CFG_ACCUM_D128.SWIZZLE_BYTES if transpose else 16,
        stride_byte_offset=8 * CFG_ACCUM_D128.SWIZZLE_BYTES,
        layout=CFG_ACCUM_D128.SWIZZLE_LAYOUT,
        tma_loads_per_tile=CFG_ACCUM_D128.D // box,
        tma_granu_elems=box,
        tma_subtile_stride_elems=CFG_ACCUM_D128.M * box,
        desc_version=CFG_ACCUM_D128.DESC_VERSION,
    )


def _mma_desc_thd(dtype, *, atranspose=False, btranspose=False):
    return MmaDesc(
        M=CFG_ACCUM_D128.M,
        N=CFG_ACCUM_D128.N,
        K=CFG_ACCUM_D128.D,
        bpe_a=CFG_ACCUM_D128.BPE,
        bpe_b=CFG_ACCUM_D128.BPE,
        tile_k_hw=CFG_ACCUM_D128.K_HW,
        atranspose=atranspose,
        btranspose=btranspose,
        cta_group=1,
        idesc=nvvm.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=dtype,
            b_dtype=dtype,
            m_dim=CFG_ACCUM_D128.M,
            n_dim=CFG_ACCUM_D128.N,
            a_major=int(atranspose),
            b_major=int(btranspose),
        ),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


@cute.jit
def _wait_thd(bar, phase):
    bar.wait(phase)
    # All full-warp consumers reconverge before any elected credit/issue.
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _wait_tcgen_thd(bar, phase):
    _wait_thd(bar, phase)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _credit_thd(bar):
    if nvvm.elect_sync():
        bar.arrive()


@cute.jit
def _map_ptr(maps, batch, n_batch, slot: cutlass.Constexpr[int]):
    return (maps.iterator.raw_ptr() + (cutlass.Int64(slot) * n_batch + batch) * TENSOR_MAP_QWORDS).tospace(cutlass.AddressSpace.generic)


@cute.jit
def _next_tile_thd(sched, bars, state):
    _wait_thd(bars.sched_full[state.idx], state.phase)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    return x, yz & 0xFFFF, (yz >> 16) & 0xFFFF, valid, advance(state, CFG_ACCUM_D128.SCHED_STAGES)


def _tensor_map_thd(tensor, box):
    # Logical TMA coordinates (d, sequence, head, batch), source-proved order.
    return tmap.create_tensor_map_tiled_from_view(
        tensor, box_dims=(1, CFG_ACCUM_D128.M, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b
    )


@cute.jit
def _bounds_thd(block, meta, batch, n_batch, causal: cutlass.Constexpr[bool]):
    data = cutlass.make_array_view(meta)
    q_start = cutlass.Int32(data[n_batch + batch])
    seq_q = cutlass.Int32(data[n_batch + batch + 1]) - q_start
    seq_k = cutlass.Int32(data[batch])
    lo = block if cutlass.const_expr(causal) else cutlass.Int32(0)
    hi = (seq_q + CFG_ACCUM_D128.N - 1) // CFG_ACCUM_D128.N
    if block * CFG_ACCUM_D128.M >= seq_k:
        lo = cutlass.Int32(0)
        hi = cutlass.Int32(0)
    return lo, hi, seq_q, seq_k, q_start


@cute.jit
def _stats_thd(slab, lse, delta, row, group, qbase, batch, head, scale, q_start, seq_q, origins, stats_origin: cutlass.Constexpr[int]):
    # Full 256-lane named barriers also cover inactive tail rows. Invalid rows
    # never dereference a Stats/Delta address, including an all-empty carrier.
    named_barrier_fence(CFG_ACCUM_D128.SOFTMAX_BARRIER, CFG_ACCUM_D128.SOFTMAX_LANES)
    local_row = qbase + row
    if group == 0:
        stat = cutlass.Float32(0.0)
        if local_row < seq_q:
            token = thd_port_origin(origins, stats_origin, batch, q_start) + cutlass.Int64(local_row)
            if cutlass.const_expr(len(lse.shape) == 2):
                val = cutlass.Float32(lse[token, head])
            else:
                val = cutlass.Float32(lse[0, head, token])
            stat = -cutlass.Float32(LOG2E) * val
            if val == cutlass.Float32(float("-inf")):
                stat = cutlass.Float32(0.0)
        _view(slab, CFG_ACCUM_D128.STATS, cutlass.Float32, CFG_ACCUM_D128.N).subview(row).store(stat)
    else:
        stat = cutlass.Float32(0.0)
        if local_row < seq_q:
            stat = -scale * cutlass.Float32(delta[0, head, cutlass.Int64(q_start) + local_row])
        _view(slab, CFG_ACCUM_D128.DELTA, cutlass.Float32, CFG_ACCUM_D128.N).subview(row).store(stat)
    named_barrier_fence(CFG_ACCUM_D128.SOFTMAX_BARRIER, CFG_ACCUM_D128.SOFTMAX_LANES)


@cute.jit
def _p_values_thd(slab, tmem_base, row, group, block, qtile, seq_q, seq_k, scale, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr):
    address = tmem_base + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scores = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    values = []
    scaled = scale * cutlass.Float32(LOG2E)
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG_ACCUM_D128.STATS, cutlass.Float32, CFG_ACCUM_D128.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            values.extend((cute.math.exp2(e0, fastmath=True), cute.math.exp2(e1, fastmath=True)))
    # KV-major transpose of the shared mask band: Q columns >= key for causal,
    # below the actual Q length, and entirely dead for a padded KV row. Select
    # exact zero AFTER exp2 (also safe for a no-key row whose LSE is -inf).
    key = block * CFG_ACCUM_D128.M + row
    qbase = qtile * CFG_ACCUM_D128.N + group * 64
    lo = key if cutlass.const_expr(causal) else None
    hi = cutlass.Int32(arith.select((key < seq_k).ir_value(), seq_q.ir_value(), cutlass.Int32(0).ir_value()))
    words = band_mask_words(lo, hi, qbase, 64)
    p = apply_mask_words(cutlass.Vector.from_elements(tuple(values), cutlass.Float32), words, mask_value=0.0, n_cols=64)
    packed = cutlass.Vector.from_elements(tuple(fp32_to_fp16(p[2 * i], p[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
    _tmem_store(address, packed)
    return p


@cute.jit
def _ds_values_thd(slab, tmem_base, group, p, scale, dtype: cutlass.Constexpr):
    address = tmem_base + 128 + group * 64
    a = _tmem_load(address, 32)
    b = _tmem_load(address + 32, 32)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    dp = cutlass.Vector.from_elements(tuple(a[i] for i in range(32)) + tuple(b[i] for i in range(32)), cutlass.Float32)
    packed = []
    for chunk in cutlass.range_constexpr(16):
        stat = _view(slab, CFG_ACCUM_D128.DELTA, cutlass.Float32, CFG_ACCUM_D128.N).subview(group * 64 + chunk * 4).load(vector_size=4, alignment=16)
        for pair in cutlass.range_constexpr(2):
            i = chunk * 4 + pair * 2
            d0, d1 = ffma2(scale, scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
            d0, d1 = fmul2(d0, d1, p[i], p[i + 1])
            packed.append(fp32_to_fp16(d0, d1, dtype=dtype))
    _tmem_store(address, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))


@cute.jit
def _softmax_loop_thd(
    slab,
    bars,
    sched,
    tmem_base,
    lse,
    delta,
    origins,
    stats_origin: cutlass.Constexpr[int],
    block,
    head,
    batch,
    meta,
    n_batch,
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG_ACCUM_D128.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG_ACCUM_D128.M
    group = tid // CFG_ACCUM_D128.M
    state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        if cutlass.const_expr(not causal):
            _credit_thd(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        if lo < hi:
            _stats_thd(slab, lse, delta, row, group, lo * CFG_ACCUM_D128.N, batch, head, scale, q_start, seq_q, origins, stats_origin)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_thd(bars.s_done, state.phase)
            p = _p_values_thd(slab, tmem_base, row, group, block, qtile, seq_q, seq_k, scale, causal, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.p_ready.arrive()
            _wait_tcgen_thd(bars.dp_done, state.phase)
            _ds_values_thd(slab, tmem_base, group, p, scale, dtype)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_ready.arrive()
            if qtile + 1 < hi:
                _stats_thd(slab, lse, delta, row, group, (qtile + 1) * CFG_ACCUM_D128.N, batch, head, scale, q_start, seq_q, origins, stats_origin)
            state = advance(state, 1)
        if cutlass.const_expr(causal):
            _credit_thd(bars.sched_empty[sched_state.idx])
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.jit
def _store_words_thd(slab, byte_offset, row, words):
    for vec in cutlass.range_constexpr(8):
        off = byte_offset + row * CFG_ACCUM_D128.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        values = cutlass.Vector.from_elements(tuple(words[vec * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _store_dq_slice_thd(slab, row, values, index: cutlass.Constexpr[int]):
    for vec in cutlass.range_constexpr(8):
        off = CFG_ACCUM_D128.ALIAS + index * CFG_ACCUM_D128.SLICE_BYTES + row * CFG_ACCUM_D128.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        v = cutlass.Vector.from_elements(tuple(values[index * 32 + vec * 4 + k] for k in range(4)), cutlass.Float32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Float32).store(v, alignment=16)


@cute.jit
def _native_epilogue_thd(slab, tmem_base, bars, row, live, empty_phase, dtype: cutlass.Constexpr, dk: cutlass.Constexpr[bool]):
    plane = 384 if cutlass.const_expr(dk) else 256
    base = CFG_ACCUM_D128.ALIAS + (CFG_ACCUM_D128.OPERAND_BYTES if cutlass.const_expr(dk) else 0)
    last_slice = 3 if cutlass.const_expr(dk) else 1
    for half in cutlass.range_constexpr(2):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + plane + half * 64, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
        if cutlass.const_expr(half == 0):
            _wait_thd(bars.dq_empty[last_slice], empty_phase)
        _store_words_thd(slab, base + half * CFG_ACCUM_D128.SLICE_BYTES, row, words)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if cutlass.const_expr(dk):
        bars.dk_full.arrive()
    else:
        bars.dv_full.arrive()


@cute.jit
def _native_epilogue_fp32(slab, tmem_base, bars, row, live, empty_phase, dk: cutlass.Constexpr[bool]):
    """dkv_accum = fp32_reduce (the grouped THD kernel only): the lane's kv row of the fp32 dV (TMEM plane CFG_ACCUM_D128.DV_COL) or dK (CFG_ACCUM_D128.DK_COL)
    accumulator as the FOUR [M x DQ_BOX] fp32 SW128 slices of the ALIAS slab (the dQ staging image, `_store_dq_slice`), published through
    dv_full / dk_full for the store warp's reduce-adds.  The dV round's WAR (the last dQ slice reads) is the caller's terminal dq_empty waits;
    the dK round waits dq_empty[i] (the store warp's credit after its dV slice i source read) before overwriting slice i -- the SAME ring
    object, one more completion per item on both sides.  A zero-trip owner stages zeros (its reduce-adds are skipped by the store warp)."""
    # ALIAS 64 KiB = one fp32 tensor (4 dQ-slice boxes): dV round, dK round; dq_empty x4
    plane = CFG_ACCUM_D128.DK_COL if cutlass.const_expr(dk) else CFG_ACCUM_D128.DV_COL
    lo_vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG_ACCUM_D128.D_HALF)), cutlass.Float32)
    hi_vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG_ACCUM_D128.D_HALF)), cutlass.Float32)
    if live:
        lo_vals = _tmem_load(tmem_base + plane, CFG_ACCUM_D128.D_HALF)
        hi_vals = _tmem_load(tmem_base + plane + CFG_ACCUM_D128.D_HALF, CFG_ACCUM_D128.D_HALF)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    vals = cutlass.Vector.from_elements(
        tuple(lo_vals[i] for i in range(CFG_ACCUM_D128.D_HALF)) + tuple(hi_vals[i] for i in range(CFG_ACCUM_D128.D_HALF)), cutlass.Float32
    )
    for i in cutlass.range_constexpr(CFG_ACCUM_D128.DKV_ACC_SLICES):
        if cutlass.const_expr(dk):
            _wait_thd(bars.dq_empty[i], empty_phase)  # WAR: the store warp read the dV slice out of these bytes (its post-read credit)
        _store_dq_slice_thd(slab, row, vals, i)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if cutlass.const_expr(dk):
        bars.dk_full.arrive()
    else:
        bars.dv_full.arrive()


@cute.jit
def _epilogue_loop_thd(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    dkv_fp32: cutlass.Constexpr[bool] = False,
):
    nvvm.setmaxregister(CFG_ACCUM_D128.EPILOGUE_REGS, nvvm.SetMaxRegisterAction.INCREASE)
    row = cute.arch.thread_idx()[0] % CFG_ACCUM_D128.M
    ds_state = PipelineState.start()
    dq_state = PipelineState.start()
    empty_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_thd(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        for _ in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_thd(bars.ds_ready, ds_state.phase)
            ds0 = _tmem_load(tmem_base + 128, 32, cutlass.Int32)
            ds1 = _tmem_load(tmem_base + 192, 32, cutlass.Int32)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            _wait_thd(bars.dq_empty[0], empty_state.phase)
            _store_words_thd(slab, CFG_ACCUM_D128.ALIAS, row, ds0)
            _wait_thd(bars.dq_empty[1], empty_state.phase)
            _store_words_thd(slab, CFG_ACCUM_D128.ALIAS + CFG_ACCUM_D128.SLICE_BYTES, row, ds1)
            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.ds_smem.arrive()
            _wait_tcgen_thd(bars.dq_done, dq_state.phase)
            dq0 = _tmem_load(tmem_base + 128, 64)
            dq1 = _tmem_load(tmem_base + 192, 64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.dp_empty.arrive()
            vals = cutlass.Vector.from_elements(tuple(dq0[i] for i in range(64)) + tuple(dq1[i] for i in range(64)), cutlass.Float32)
            for i in cutlass.range_constexpr(4):
                _wait_thd(bars.dq_empty[i], empty_state.phase)
                _store_dq_slice_thd(slab, row, vals, i)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.dq_full[i].arrive()
            ds_state = advance(ds_state, 1)
            dq_state = advance(dq_state, 1)
            empty_state = advance(empty_state, 1)
        if lo >= hi:
            _wait_tcgen_thd(bars.dq_done, dq_state.phase)
            dq_state = advance(dq_state, 1)
        # Observe every terminal source-credit phase before the next CLC
        # owner bootstraps these barriers again. Native dV/dK alone wait on
        # slices 1/3, leaving slices 0/2 unobserved after the final dQ or a
        # zero-trip owner. Each completed mbarrier phase needs a successful
        # wait before an arrival in its next phase.
        for i in cutlass.range_constexpr(4):
            _wait_thd(bars.dq_empty[i], empty_state.phase)
        if cutlass.const_expr(dkv_fp32):
            # dkv_accum = fp32_reduce: dV then dK as fp32 slices over the ALIAS slab -- two ring rounds per item (the store warp credits dq_empty
            # after the dV slices' source reads; the dK round waits them at the advanced phase)
            _native_epilogue_fp32(slab, tmem_base, bars, row, lo < hi, empty_state.phase, False)
            empty_state = advance(empty_state, 1)
            _native_epilogue_fp32(slab, tmem_base, bars, row, lo < hi, empty_state.phase, True)
        else:
            _native_epilogue_thd(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, False)
            _native_epilogue_thd(slab, tmem_base, bars, row, lo < hi, empty_state.phase, dtype, True)
        empty_state = advance(empty_state, 1)
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.jit
def _ts_thd(desc, plane, bdesc, dst, accum):
    for k in cutlass.range_constexpr(CFG_ACCUM_D128.D // CFG_ACCUM_D128.K_HW):
        mma_ts_step(
            desc, nvvm.make_tmem_ptr(plane + (k // 4) * 32, cutlass.Int8), bdesc, dst, k_idx=k, accumulate=accum if cutlass.const_expr(k == 0) else True
        )


@cute.jit
def _mma_loop_thd(
    slab,
    bars,
    sched,
    tmem_base,
    block,
    head,
    batch,
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    score = _mma_desc_thd(dtype)
    grad = _mma_desc_thd(dtype, btranspose=True)
    dqdesc = _mma_desc_thd(dtype, atranspose=True, btranspose=True)
    kd = _operand_thd(slab, CFG_ACCUM_D128.K, dtype).desc()
    vd = _operand_thd(slab, CFG_ACCUM_D128.V, dtype).desc()
    dod = _operand_thd(slab, CFG_ACCUM_D128.DO, dtype).desc()
    dot = _operand_thd(slab, CFG_ACCUM_D128.DO, dtype, transpose=True).desc()
    dst = _operand_thd(slab, CFG_ACCUM_D128.ALIAS, dtype, transpose=True).desc()
    kt = _operand_thd(slab, CFG_ACCUM_D128.K, dtype, transpose=True).desc()
    s = nvvm.make_tmem_ptr(tmem_base, cutlass.Float32)
    dp = nvvm.make_tmem_ptr(tmem_base + 128, cutlass.Float32)
    dv = nvvm.make_tmem_ptr(tmem_base + 256, cutlass.Float32)
    dk = nvvm.make_tmem_ptr(tmem_base + 384, cutlass.Float32)
    q_state = PipelineState.start()
    do_state = PipelineState.start()
    kv_state = PipelineState.start()
    ready_state = PipelineState.start()
    dp_empty = PipelineState.start(phase=1)
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_thd(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        if lo < hi:
            _wait_tcgen_thd(bars.k_full, kv_state.phase)
            _wait_tcgen_thd(bars.q_full[q_state.idx], q_state.phase)
            qd = _operand_thd(slab, CFG_ACCUM_D128.Q + q_state.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype).desc()
            mma_ss(score, kd, qd, s)
            _commit(bars.s_done)
            _wait_tcgen_thd(bars.v_full, kv_state.phase)
        _wait_tcgen_thd(bars.dvk_empty, out_empty.phase)
        out_empty = advance(out_empty, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen_thd(bars.dp_empty, dp_empty.phase)
            _wait_tcgen_thd(bars.do_full[0], do_state.phase)
            mma_ss(score, vd, dod, dp, k_start=0, k_count=4)
            _wait_tcgen_thd(bars.do_full[1], do_state.phase)
            mma_ss(score, vd, dod, dp, k_start=4, k_count=4, accumulate=True)
            _commit(bars.dp_done)
            _wait_tcgen_thd(bars.p_ready, ready_state.phase)
            _ts_thd(grad, tmem_base, dot, dv, qtile > lo)
            _commit(bars.do_empty)
            q_next = advance(q_state, CFG_ACCUM_D128.Q_STAGES)
            if qtile + 1 < hi:
                _wait_tcgen_thd(bars.q_full[q_next.idx], q_next.phase)
                qd_next = _operand_thd(slab, CFG_ACCUM_D128.Q + q_next.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype).desc()
                mma_ss(score, kd, qd_next, s)
                _commit(bars.s_done)
            _wait_tcgen_thd(bars.ds_ready, ready_state.phase)
            qt = _operand_thd(slab, CFG_ACCUM_D128.Q + q_state.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype, transpose=True).desc()
            _ts_thd(grad, tmem_base + 128, qt, dk, qtile > lo)
            _commit(bars.q_empty[q_state.idx])
            _wait_tcgen_thd(bars.ds_smem, ready_state.phase)
            mma_ss(dqdesc, dst, kt, dp)
            _commit(bars.dq_done)
            q_state = q_next
            do_state = advance(do_state, 1)
            ready_state = advance(ready_state, 1)
            dp_empty = advance(dp_empty, 1)
        if lo < hi:
            _commit(bars.k_empty)
            _commit(bars.v_empty)
            kv_state = advance(kv_state, 1)
        else:
            _commit(bars.dq_done)
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.jit
def _load_loop_thd(
    slab,
    bars,
    sched,
    maps,
    block,
    head,
    batch,
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
):
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_thd(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        q_desc = _map_ptr(maps, batch, n_batch, MAP_Q_THD)
        k_desc = _map_ptr(maps, batch, n_batch, MAP_K_THD)
        v_desc = _map_ptr(maps, batch, n_batch, MAP_V_THD)
        do_desc = _map_ptr(maps, batch, n_batch, MAP_DO_THD)
        if lo < hi:
            tma_tensormap_acquire(q_desc)
            tma_tensormap_acquire(k_desc)
            tma_tensormap_acquire(v_desc)
            tma_tensormap_acquire(do_desc)
            _wait_thd(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.K, dtype),
                tma_slice_runtime_desc(k_desc, 0, block * CFG_ACCUM_D128.M, head, 0),
                bars.k_full.smem_ptr,
                acquire=False,
            )
            _wait_thd(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.V, dtype),
                tma_slice_runtime_desc(v_desc, 0, block * CFG_ACCUM_D128.M, head, 0),
                bars.v_full.smem_ptr,
                acquire=False,
            )
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_thd(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.Q + q_state.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype),
                tma_slice_runtime_desc(q_desc, 0, qtile * CFG_ACCUM_D128.N, head, 0),
                bars.q_full[q_state.idx].smem_ptr,
                acquire=False,
            )
            _wait_thd(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[half].arrive(n_bytes=CFG_ACCUM_D128.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    _operand_thd(slab, CFG_ACCUM_D128.DO, dtype),
                    tma_slice_runtime_desc(do_desc, 0, qtile * CFG_ACCUM_D128.N, head, 0),
                    bars.do_full[half].smem_ptr,
                    half,
                    1,
                    acquire=False,
                )
            q_state = advance(q_state, CFG_ACCUM_D128.Q_STAGES)
            do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.jit
def _store_loop_thd(
    slab,
    bars,
    sched,
    maps,
    block,
    head,
    batch,
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    dkv_fp32: cutlass.Constexpr[bool] = False,
    ratio: cutlass.Constexpr[int] = 1,
):
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    dq_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_thd(bars.sched_empty[sched_state.idx])
        for i in cutlass.range_constexpr(4):
            bars.dq_empty[i].arrive()  # all32 lanes, one bootstrap per KV tile
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        dq_desc = _map_ptr(maps, batch, n_batch, MAP_DQ_THD)
        if lo < hi:
            tma_tensormap_acquire(dq_desc)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            for i in cutlass.range_constexpr(4):
                _wait_thd(bars.dq_full[i], dq_state.phase)
                # DIAGNOSTIC D2 (default off = the shipped code): the armed rendering deletes ONLY the reduce-add issue;
                # the commit, the wait_group and both dq_* arrives stay, so the barrier table is unchanged.
                if cutlass.const_expr(not diag_delete_dq_reduce_add):
                    tma_reduce_add_subtile(
                        _operand_thd(slab, CFG_ACCUM_D128.ALIAS, cutlass.Float32, fp32=True),
                        tma_slice_runtime_desc(dq_desc, 0, qtile * CFG_ACCUM_D128.N, head, 0),
                        i,
                        acquire=False,
                    )
                tma_store_commit()
            for i in cutlass.range_constexpr(4):
                tma_store_wait(3 - i)
                bars.dq_empty[i].arrive()  # all32 after corresponding source read
            dq_state = advance(dq_state, 1)
        _wait_thd(bars.dv_full, out_state.phase)
        if cutlass.const_expr(dkv_fp32):
            # dkv_accum = fp32_reduce: four fp32 reduce-adds per tensor from the ALIAS slices into the per-KV-head workspace maps (the MAP_DV / MAP_DK
            # slots carry fp32 [M x DQ_BOX]-box maps; kv head = head // ratio); the per-slice bulk groups credit dq_empty[i] after the dV slice's
            # source read (the dK staging's WAR edge, P7), then dK.  A zero-trip owner issues nothing (the workspace is zero-filled) and never
            # acquires a map, but keeps every commit / wait / arrive.
            live = lo < hi
            dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV_THD)
            if live:
                tma_tensormap_acquire(dv_desc)
            for i in cutlass.range_constexpr(CFG_ACCUM_D128.DKV_ACC_SLICES):
                if live:
                    tma_reduce_add_subtile(
                        _operand_thd(slab, CFG_ACCUM_D128.ALIAS, cutlass.Float32, fp32=True),
                        tma_slice_runtime_desc(dv_desc, 0, block * CFG_ACCUM_D128.M, head // ratio, 0),
                        i,
                        acquire=False,
                    )
                tma_store_commit()
            for i in cutlass.range_constexpr(CFG_ACCUM_D128.DKV_ACC_SLICES):
                tma_store_wait(CFG_ACCUM_D128.DKV_ACC_SLICES - 1 - i)
                bars.dq_empty[i].arrive()  # all32 after the dV slice's source read: the epilogue's dK round may overwrite slice i
            _wait_thd(bars.dk_full, out_state.phase)
            dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK_THD)
            if live:
                tma_tensormap_acquire(dk_desc)
            for i in cutlass.range_constexpr(CFG_ACCUM_D128.DKV_ACC_SLICES):
                if live:
                    tma_reduce_add_subtile(
                        _operand_thd(slab, CFG_ACCUM_D128.ALIAS, cutlass.Float32, fp32=True),
                        tma_slice_runtime_desc(dk_desc, 0, block * CFG_ACCUM_D128.M, head // ratio, 0),
                        i,
                        acquire=False,
                    )
                tma_store_commit()
        else:
            # Invalid envelope owners still consume both native publications and
            # return every source credit, but never acquire/use an empty map.
            if block * CFG_ACCUM_D128.M < seq_k:
                dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV_THD)
                tma_store_tile(_operand_thd(slab, CFG_ACCUM_D128.ALIAS, dtype), tma_slice_runtime_desc(dv_desc, 0, block * CFG_ACCUM_D128.M, head, 0))
            _wait_thd(bars.dk_full, out_state.phase)
            if block * CFG_ACCUM_D128.M < seq_k:
                dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK_THD)
                tma_store_tile(
                    _operand_thd(slab, CFG_ACCUM_D128.ALIAS + CFG_ACCUM_D128.OPERAND_BYTES, dtype),
                    tma_slice_runtime_desc(dk_desc, 0, block * CFG_ACCUM_D128.M, head, 0),
                )
        out_state = advance(out_state, 1)
        tma_store_commit()
        tma_store_wait(0)
        bars.dvk_empty.arrive()
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.jit
def _scheduler_loop_thd(bars, sched):
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _wait_thd(bars.sched_empty[state.idx], state.phase)
        if nvvm.elect_sync():
            bars.sched_full[state.idx].arrive(n_bytes=16)
            nvvm.clusterlaunchcontrol_try_cancel(sched.tile_id_smem.subview(state.idx * 8), bars.sched_full[state.idx].smem_ptr, multicast=1)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        _wait_thd(bars.sched_full[state.idx], state.phase)
        _, _, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
        state = advance(state, CFG_ACCUM_D128.SCHED_STAGES)


@cute.kernel
def bprop_d128_accum_thd_kernel(
    maps: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    n_batch: cutlass.Int32,
    stats_origin: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    block, head, batch = x, y, z
    slab = cutlass.Array(cutlass.Uint8, CFG_ACCUM_D128.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars_thd(slab)
    tmem_slot = _view(slab, CFG_ACCUM_D128.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG_ACCUM_D128.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
        # All 512 TMEM columns belong to this CTA; use the portable allocation form.
        nvvm.tcgen05_alloc(tmem_slot, cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    tmem_base = tmem_slot.load()
    if warp < 8:
        _softmax_loop_thd(slab, bars, sched, tmem_base, lse, delta, origins, stats_origin, block, head, batch, meta, n_batch, scale, causal, dtype)
    elif warp < 12:
        _epilogue_loop_thd(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 12:
        _mma_loop_thd(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 13:
        _load_loop_thd(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 14:
        _store_loop_thd(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype, diag_delete_dq_reduce_add)
    else:
        _scheduler_loop_thd(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_thd_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host_accum_thd(
    maps,
    lse,
    delta,
    meta,
    origins,
    n_batch,
    nheads,
    max_skv,
    stats_origin: cutlass.Constexpr[int],
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    """Launch the native THD body on a host-declared KV envelope, never live totals."""
    nk = cute.math.max((max_skv + CFG_ACCUM_D128.M - 1) // CFG_ACCUM_D128.M, cutlass.Int32(1))
    bprop_d128_accum_thd_kernel(maps, lse, delta, meta, origins, n_batch, stats_origin, scale, causal, dtype, diag_delete_dq_reduce_add).launch(
        grid=(nk, nheads, n_batch), block=(CFG_ACCUM_D128.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1
    )


# ======================================================================================================================
# section 6: the grouped chain hosts (dense, THD)
# (was kernels/sm100/bprop_d128_accum_grouped_helpers.py)
# ======================================================================================================================
"""Dense/THD grouped d128 chains with native partials and a final FP32 fold."""


@cute.jit
def host_grouped_dense(
    q,
    k,
    v,
    o,
    do,
    lse,
    dq,
    dk,
    dv,
    dq_accum,
    delta,
    dk_part,
    dv_part,
    scale,
    causal: cutlass.Constexpr[bool],
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    """Four native launches; uniquely owned partials are folded in FP32."""
    delta_zero_host(o, do, delta, dq_accum, stream)
    main_dense_host(q, k, v, do, lse, delta, dq_accum, dk_part, dv_part, scale, causal, stream, diag_delete_dq_reduce_add)
    cast_dq_host(dq_accum, dq, q.element_type, stream)
    dkv_reduce_host(dk_part, dv_part, dk, dv, CFG_ACCUM_D128.D, CFG_ACCUM_D128.D, q.shape[2] // k.shape[2], q.element_type, False, stream)


@cute.jit
def host_grouped_thd(
    q,
    k,
    v,
    o,
    do,
    lse,
    dq,
    dk,
    dv,
    dq_accum,
    delta,
    dk_part,
    dv_part,
    meta,
    origins,
    maps,
    q_lens,
    kv_lens,
    ro,
    origin_specs: cutlass.Constexpr,
    lens_form,
    batch: cutlass.Constexpr[int],
    max_sq,
    max_skv,
    scale,
    causal: cutlass.Constexpr[bool],
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    dkv_fp32: cutlass.Constexpr[bool] = False,
    dk_acc=None,
    dv_acc=None,
):
    """Five native launches, with compact partials and independent final origins; FOUR under ``dkv_fp32`` (dkv_accum = fp32_reduce): the
    per-KV-head fp32 dK / dV accumulators ``dk_acc`` / ``dv_acc`` take the MAP_DK / MAP_DV slots as fp32 [M x DQ_BOX]-box maps (compact KV
    origins, the sequence-local clip), delta_zero zero-fills them over the kv rows, the main kernel reduce-adds every owner's fp32 tile,
    cast_dq converts them once to the IO dtype at each output's own origin, and the fold is not launched."""
    if cutlass.const_expr(dkv_fp32):
        dk_map, dv_map = _tensor_map_thd(dk_acc, CFG_ACCUM_D128.DQ_BOX), _tensor_map_thd(dv_acc, CFG_ACCUM_D128.DQ_BOX)
        dk_ws, dv_ws = dk_acc, dv_acc
    else:
        dk_map, dv_map = _tensor_map_thd(dk_part, CFG_ACCUM_D128.HEAD_BOX), _tensor_map_thd(dv_part, CFG_ACCUM_D128.HEAD_BOX)
        dk_ws, dv_ws = dk_part, dv_part
    setup_thd_d128_kernel(
        _tensor_map_thd(q, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(k, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(v, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(do, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_thd(dq_accum, CFG_ACCUM_D128.DQ_BOX),
        dk_map,
        dv_map,
        q,
        k,
        v,
        do,
        dq_accum,
        dk_ws,
        dv_ws,
        meta,
        origins,
        maps,
        q_lens,
        kv_lens,
        *ro,
        origin_specs,
        lens_form,
        cutlass.Int32(batch),
        partial_dkv=True,
    ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
    q_grid = cute.math.max((max_sq + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
    if cutlass.const_expr(dkv_fp32):
        # the row extent covers the q rows (Delta, dQ zero) AND the kv rows (the fp32 dK / dV zero fill / convert); z = Hq, kv heads = the first Hkv
        kv_grid = cute.math.max((max_skv + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
        rows_grid = cute.math.max(q_grid, kv_grid)
        delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs, dk_acc, dv_acc).launch(
            grid=(rows_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
        )
    else:
        delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs).launch(
            grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
        )
    main_thd_host(
        maps,
        lse,
        delta,
        meta,
        origins,
        cutlass.Int32(batch),
        q.shape[2],
        max_skv,
        _origin_row(origin_specs, 8),
        scale,
        causal,
        q.element_type,
        q.shape[2] // k.shape[2],
        stream,
        diag_delete_dq_reduce_add,
        dkv_fp32,
    )
    if cutlass.const_expr(dkv_fp32):
        cast_dq_thd_d128_kernel(
            dq_accum,
            dq,
            meta,
            origins,
            cutlass.Int32(batch),
            _origin_row(origin_specs, 5),
            q.element_type,
            dk_acc,
            dv_acc,
            dk,
            dv,
            _origin_row(origin_specs, 6),
            _origin_row(origin_specs, 7),
        ).launch(grid=(rows_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream)
    else:
        cast_dq_thd_d128_kernel(dq_accum, dq, meta, origins, cutlass.Int32(batch), _origin_row(origin_specs, 5), q.element_type).launch(
            grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
        )
        dkv_reduce_thd_origins_host(
            dk_part,
            dv_part,
            dk,
            dv,
            meta,
            origins,
            cutlass.Int32(batch),
            max_skv,
            _origin_row(origin_specs, 6),
            _origin_row(origin_specs, 7),
            CFG_ACCUM_D128.D,
            q.shape[2] // k.shape[2],
            q.element_type,
            stream,
        )


# ======================================================================================================================
# section 7: the grouped bodies (dense, THD): the MHA roles with the grouped loaders
# (was kernels/sm100/bprop_d128_accum_grouped_f16.py)
# ======================================================================================================================
"""Grouped d128 backward: one Q-head owner, native dK/dV partials.

The qualified MHA math, epilogue, store and scheduler roles are imported
unchanged. Only the loaders map Q heads to shared K/V heads. Separate fold
kernels consume the uniquely owned native partials after this main pass.

Op-level field ``dkv_accum = "fp32_reduce"`` (``Constexpr[bool]`` ``dkv_fp32`` on the THD kernel and ``main_thd_host``, default False = the
shipped chain, rendered PTX-identical): the epilogue / store roles of the packed body take their fp32 store path (the section 5
header), reduce-adding each owner's fp32 dK / dV tile into the per-KV-head workspace maps (kv head = head // ratio) instead of storing a
bf16 partial; the fold launch is then not needed.
"""


@cute.jit
def _load_dense(
    slab,
    bars,
    sched,
    q_desc,
    k_desc,
    v_desc,
    do_desc,
    block,
    head,
    batch,
    sq,
    nk,
    nheads,
    batch_heads: cutlass.Constexpr[int],
    swizzle: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
):
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_accum(bars.sched_empty[sched_state.idx])
        lo, hi = _bounds_accum(block, sq, causal)
        if lo < hi:
            _wait_accum(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_accum(slab, CFG_ACCUM_D128.K, dtype), GmemTileTma(k_desc)(0, block * CFG_ACCUM_D128.M, head // ratio, batch), bars.k_full.smem_ptr
            )
            _wait_accum(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_accum(slab, CFG_ACCUM_D128.V, dtype), GmemTileTma(v_desc)(0, block * CFG_ACCUM_D128.M, head // ratio, batch), bars.v_full.smem_ptr
            )
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_accum(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_accum(slab, CFG_ACCUM_D128.Q + q_state.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype),
                GmemTileTma(q_desc)(0, qtile * CFG_ACCUM_D128.N, head, batch),
                bars.q_full[q_state.idx].smem_ptr,
            )
            _wait_accum(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[half].arrive(n_bytes=CFG_ACCUM_D128.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    _operand_accum(slab, CFG_ACCUM_D128.DO, dtype),
                    GmemTileTma(do_desc)(0, qtile * CFG_ACCUM_D128.N, head, batch),
                    bars.do_full[half].smem_ptr,
                    half,
                    1,
                )
            q_state = advance(q_state, CFG_ACCUM_D128.Q_STAGES)
            do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = _next_tile_accum(sched, bars, sched_state, nheads, nk, causal, swizzle, batch_heads)


@cute.jit
def _load_thd(
    slab, bars, sched, maps, block, head, batch, meta, n_batch, causal: cutlass.Constexpr[bool], dtype: cutlass.Constexpr, ratio: cutlass.Constexpr[int]
):
    nvvm.setmaxregister(CFG_ACCUM_D128.SERVICE_REGS, nvvm.SetMaxRegisterAction.DECREASE)
    q_state = PipelineState.start(phase=1)
    do_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit_thd(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _bounds_thd(block, meta, batch, n_batch, causal)
        q_desc = _map_ptr(maps, batch, n_batch, MAP_Q_THD)
        k_desc = _map_ptr(maps, batch, n_batch, MAP_K_THD)
        v_desc = _map_ptr(maps, batch, n_batch, MAP_V_THD)
        do_desc = _map_ptr(maps, batch, n_batch, MAP_DO_THD)
        if lo < hi:
            tma_tensormap_acquire(q_desc)
            tma_tensormap_acquire(k_desc)
            tma_tensormap_acquire(v_desc)
            tma_tensormap_acquire(do_desc)
            _wait_thd(bars.k_empty, kv_state.phase)
            bars.k_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.K, dtype),
                tma_slice_runtime_desc(k_desc, 0, block * CFG_ACCUM_D128.M, head // ratio, 0),
                bars.k_full.smem_ptr,
                acquire=False,
            )
            _wait_thd(bars.v_empty, kv_state.phase)
            bars.v_full.arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.V, dtype),
                tma_slice_runtime_desc(v_desc, 0, block * CFG_ACCUM_D128.M, head // ratio, 0),
                bars.v_full.smem_ptr,
                acquire=False,
            )
            kv_state = advance(kv_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_thd(bars.q_empty[q_state.idx], q_state.phase)
            bars.q_full[q_state.idx].arrive(n_bytes=CFG_ACCUM_D128.OPERAND_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                _operand_thd(slab, CFG_ACCUM_D128.Q + q_state.idx * CFG_ACCUM_D128.OPERAND_BYTES, dtype),
                tma_slice_runtime_desc(q_desc, 0, qtile * CFG_ACCUM_D128.N, head, 0),
                bars.q_full[q_state.idx].smem_ptr,
                acquire=False,
            )
            _wait_thd(bars.do_empty, do_state.phase)
            for half in cutlass.range_constexpr(2):
                bars.do_full[half].arrive(n_bytes=CFG_ACCUM_D128.SLICE_BYTES, pred=nvvm.elect_sync())
            for half in cutlass.range_constexpr(2):
                tma_load_subtiles(
                    _operand_thd(slab, CFG_ACCUM_D128.DO, dtype),
                    tma_slice_runtime_desc(do_desc, 0, qtile * CFG_ACCUM_D128.N, head, 0),
                    bars.do_full[half].smem_ptr,
                    half,
                    1,
                    acquire=False,
                )
            q_state = advance(q_state, CFG_ACCUM_D128.Q_STAGES)
            do_state = advance(do_state, 1)
        block, head, batch, valid, sched_state = _next_tile_thd(sched, bars, sched_state)


@cute.kernel
def bprop_d128_accum_grouped_dense_kernel(
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    nheads: cutlass.Int32,
    batch_heads: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    swizzle: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    nk = skv // CFG_ACCUM_D128.M
    if cutlass.const_expr(causal):
        block, head, batch = _decode_flat(x, nheads, nk, swizzle, batch_heads)
    else:
        block, head, batch = (x, y, z)
    slab = cutlass.Array(cutlass.Uint8, CFG_ACCUM_D128.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars_accum(slab)
    tmem_slot = _view(slab, CFG_ACCUM_D128.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG_ACCUM_D128.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
        nvvm.tcgen05_alloc(tmem_slot, cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    tmem_base = tmem_slot.load()
    if warp < 8:
        _softmax_loop_accum(slab, bars, sched, tmem_base, lse, delta, block, head, batch, sq, skv, nk, nheads, batch_heads, swizzle, scale, causal, dtype)
    elif warp < 12:
        _epilogue_loop_accum(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == 12:
        _mma_loop_accum(slab, bars, sched, tmem_base, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype)
    elif warp == 13:
        _load_dense(slab, bars, sched, q_desc, k_desc, v_desc, do_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, ratio)
    elif warp == 14:
        _store_loop_accum(
            slab, bars, sched, dq_desc, dk_desc, dv_desc, block, head, batch, sq, nk, nheads, batch_heads, swizzle, causal, dtype, diag_delete_dq_reduce_add
        )
    else:
        _scheduler_loop_accum(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_grouped_dense_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def bprop_d128_accum_grouped_thd_kernel(
    maps: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    n_batch: cutlass.Int32,
    stats_origin: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    dkv_fp32: cutlass.Constexpr[bool] = False,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    block, head, batch = (x, y, z)
    slab = cutlass.Array(cutlass.Uint8, CFG_ACCUM_D128.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    bars = _bars_thd(slab)
    tmem_slot = _view(slab, CFG_ACCUM_D128.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, CFG_ACCUM_D128.SCHED_PAYLOAD, cutlass.Int32, 16, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    if warp == 0:
        nvvm.tcgen05_alloc(tmem_slot, cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                for stage in cutlass.range_constexpr(bars[i].stages):
                    bars[i][stage].init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    tmem_base = tmem_slot.load()
    if warp < 8:
        _softmax_loop_thd(slab, bars, sched, tmem_base, lse, delta, origins, stats_origin, block, head, batch, meta, n_batch, scale, causal, dtype)
    elif warp < 12:
        _epilogue_loop_thd(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype, dkv_fp32)
    elif warp == 12:
        _mma_loop_thd(slab, bars, sched, tmem_base, block, head, batch, meta, n_batch, causal, dtype)
    elif warp == 13:
        _load_thd(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype, ratio)
    elif warp == 14:
        _store_loop_thd(slab, bars, sched, maps, block, head, batch, meta, n_batch, causal, dtype, diag_delete_dq_reduce_add, dkv_fp32, ratio)
    else:
        _scheduler_loop_thd(bars, sched)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_slot.load(), cutlass.Int8), cutlass.Int32(CFG_ACCUM_D128.TMEM_COLS), group=nvvm.CTAGroup.CTA_1)


bprop_d128_accum_grouped_thd_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def main_dense_host(
    q,
    k,
    v,
    do,
    lse,
    delta,
    dq_accum,
    dk_part,
    dv_part,
    scale,
    causal: cutlass.Constexpr[bool],
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
):
    """Qualified dense geometry, with the partial output head axis equal to Hq."""
    if cutlass.const_expr(q.element_type not in (cutlass.BFloat16, cutlass.Float16)):
        raise ValueError("grouped d128 accumulation requires BF16/FP16")
    if cutlass.const_expr(q.shape[3] != CFG_ACCUM_D128.D or k.shape[3] != CFG_ACCUM_D128.D or q.shape[2] <= k.shape[2] or q.shape[2] % k.shape[2]):
        raise ValueError("grouped d128 accumulation requires integral Hq/Hkv > 1 and d128")
    if cutlass.const_expr(q.shape[1] <= 0 or k.shape[1] <= 0 or q.shape[1] % CFG_ACCUM_D128.N or k.shape[1] % CFG_ACCUM_D128.M):
        raise ValueError("dense grouped d128 accumulation requires positive Sq/Skv multiples of 128")
    nk = k.shape[1] // CFG_ACCUM_D128.M
    heads_per_l2 = (50 * 1024 * 1024) // (q.shape[1] * (CFG_ACCUM_D128.D + CFG_ACCUM_D128.D) * CFG_ACCUM_D128.BPE)
    swizzle = 1 << (heads_per_l2.bit_length() - 1) if heads_per_l2 > 0 else 1
    grid = (nk * q.shape[2] * q.shape[0], 1, 1) if causal else (nk, q.shape[2], q.shape[0])
    bprop_d128_accum_grouped_dense_kernel(
        _tensor_map_accum(q, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_accum(k, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_accum(v, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_accum(do, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_accum(dq_accum, CFG_ACCUM_D128.DQ_BOX),
        _tensor_map_accum(dk_part, CFG_ACCUM_D128.HEAD_BOX),
        _tensor_map_accum(dv_part, CFG_ACCUM_D128.HEAD_BOX),
        lse,
        delta,
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.shape[0] * q.shape[2],
        scale,
        causal,
        swizzle,
        q.element_type,
        q.shape[2] // k.shape[2],
        diag_delete_dq_reduce_add,
    ).launch(grid=grid, block=(CFG_ACCUM_D128.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)


@cute.jit
def main_thd_host(
    maps,
    lse,
    delta,
    meta,
    origins,
    n_batch,
    nheads,
    max_skv,
    stats_origin: cutlass.Constexpr[int],
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    stream,
    diag_delete_dq_reduce_add: cutlass.Constexpr[bool] = False,
    dkv_fp32: cutlass.Constexpr[bool] = False,
):
    """One owner per Q head and declared KV-envelope tile; no host length read."""
    nk = cute.math.max((max_skv + CFG_ACCUM_D128.M - 1) // CFG_ACCUM_D128.M, cutlass.Int32(1))
    bprop_d128_accum_grouped_thd_kernel(
        maps,
        lse,
        delta,
        meta,
        origins,
        n_batch,
        stats_origin,
        scale,
        causal,
        dtype,
        ratio,
        diag_delete_dq_reduce_add,
        dkv_fp32,
    ).launch(grid=(nk, nheads, n_batch), block=(CFG_ACCUM_D128.THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)


# ======================================================================================================================
# section 8: the cta_group::2 PAIR body (the dQ image in S^T's dead tail)
# (was kernels/sm100/bprop_d128_accum_grouped_2cta_stail_f16.py)
# ======================================================================================================================
"""SM100 SDPA backward d=128x128 bf16/fp16 -- grouped THD, cta_group::2 datapath: the dQ image in S^T's dead tail + FAv4's rotated order
(section 1: identity)

HEADER FIRST: the tables below come before the body; a reader checks the body against them instead of reconstructing the
pipeline from code.  This is the pair body of the grouped THD route (``accumulation_grouped_thd_2cta``): one cluster of two CTAs owns
a 256-row KV work item, each CTA computes its 128-row half with cta_group::2 MMAs and the dS^T halves are exchanged through DSMEM.
Config as truth: ``config_sm100_d128_accum_2cta_stail.CFG`` (every number below is derived there and asserted at import; the barrier
table ``BAR_LAYOUT`` is the audited table, pasted, never re-derived).  FAv4's flash_bwd_sm100.py was READ as the structural reference
only (NOTE [2CTA schedules], the pipelined MMA loop, the deferred dS commit, ``Q_stage = 1``); nothing copied.
TMEM MAP (PART-KEYED; the typed field ``tmem_map`` -> ``tmem_map_code``; sections 5 / 9 / 11): ``s_tail`` = this header's tables as written (the
512-column parts cc 10.0 / 10.3, byte-identical to the admitted rendering: the sm_100a / sm_103a records); ``dq_576`` = the INDEPENDENT dQ image at
[512, 576) of the 576-column Rubin carve (selectable BY NAME on cc 10.7 under warp_roles 16w; NOT a default: section 13): the rows s_read /
s_tail_empty and every site of theirs are not
traced, the alloc / dealloc constant is ``geo.TMEM_COLS`` = 576 (exclusive), the dQ column base ``geo.DQ_COL`` = 512; every other table row, slab,
fence and wait is unchanged.  Where a section below names the two rows, read "s_tail map only"; the dq_576 deltas are called out in place.
SMEM MAP (PART-KEYED; the typed field ``smem_map`` -> ``smem_map_code``; sections 5 / 8 / 11; arm R2.c of the Rubin task, 2026-11-03): ``sm100`` = the
slab chain of section 8 as written (every part's default: the 232448-B opt-in cap, byte-identical to the admitted rendering); ``rubin_b2`` = the 327 KiB
oversized carveout of cc 10.7 ON ``dq_576`` (selectable by name together with it; neither map is a cc 10.7 default until the arm's A/B/A row): sQ1 x2,
sDO1 x2 and the TWO-STAGE sDS + sDS_xchg staging ring (SMEM_BYTES 314368, every tcgen05 root below 262144), under which the epilogue stages dS(t) BEFORE
it waits dQ(t-1) (the LAGGING order: the WAR on a stage is dq_done(t-2), one tile earlier than today's dq_done(t-1) on the single stage) and the LEADER row
``dq_read_done`` carries the dQ image's reuse edge (the leader waits it before EVERY dQ issue).  Where a section below gives a single-stage number, read
"sm100 map"; the rubin_b2 deltas are called out in place.

2. WHAT IT COMPUTES (per (batch, kv head, q head); GQA: one q-head owner per pair, as today's grouped loaders)
    S^T = K Q^T * scale        P^T = exp2(S^T * log2e - LSE[q])        dP^T = V dO^T
    dS^T = P^T * (dP^T - delta[q])        dV += P^T dO        dK += dS^T Q        dQ = dS K  (fp32 tma_reduce_add into the dq_accum
    workspace, summed in a non-deterministic fp32 order across the pairs; cast_dq folds it in the NEXT launch)
    delta = rowsum(dO * O) comes from the chain's delta launch (THD: setup_thd -> delta_zero_thd -> MAIN -> cast_dq_thd
    -> the fp32 group fold); LSE in the natural-log unit of the forward's Stats.  Numerics across the selectable forms (mask_form,
    warp_roles) are the same function by construction (the same accumulation chain per GEMM, the same k-orders): dK / dV are pinned
    BITWISE between the forms by the test module; dQ (reduce-add order across pairs) by the fp64 gate, bitwise only where at most two kv
    pairs feed an element (the test module names those cells).
    mask_form (section 11): P^T / dS^T are BITWISE between "fsel" (the default: the keep-word mask selected on every tile) and
    "predicated" (the same mask body only on a tile that can hold a masked cell: a masked cell is exactly +0.0 in both, an interior tile has
    none), hence dQ (per reduce-add order) / dK / dV; "diag_mask_off" is the DIAGNOSTIC ceiling, OUTPUT WRONG BY DESIGN.

3. WORK ITEM + MMA SHAPES (cluster (2,1,1) = one cga2 PAIR; CTA rank r owns kv rows [128 r, +128) of a 256-row block)
    one item = 256 kv rows of one (sequence, kv head) x ONE q head; the pair walks the identical q tiles (128 rows)
    S^T  [256 kv, 128 q]  = K1[own 128 kv x 128 d] (A, K-major SW128, two boxes [128 x 64]) . Q1^T (B N-split:
                              Q[64 r.., +64 q x 128 d], two boxes [64 x 64])          cta_group::2, M 256, 8 k-steps
    dP^T [256 kv, 128 q]  = V[own 128 kv x 128 d] (A) . dO1^T (B N-split: dO[64 q x 128 d])             M 256
    dV   [256 kv, 128 d] += P^T (A, TS: own TMEM, packed bf16) . dO2 (B N-split along d: dO[128 q x 64 d], MN-major,
                              one box [128 x 64])                                                   M 256, k = q
    dK   [256 kv, 128 d] += dS^T (A, TS: own TMEM) . Q2 (B N-split along d: Q[128 q x 64 d], MN-major)  M 256
    dQ   [128 q, 128 d]   = dS (A, M-split: own 64 q rows): sDS = dS^T [256 kv x own 64 q] MN-major SW128 (128 B per
                              kv row; own 128 kv rows by the epilogue's 16-B STS, the peer's 128 rows by ONE
                              cp.async.bulk DSMEM copy of its XCHG) . K2 (B N-split along d: K[256 kv x 64 d],
                              MN-major, two 128-row boxes)                          cta_group::2, M 128, K 256 (16 steps)
    The D image of the M 128 pair MMA is the half-row atom: dQ row m -> lane m % 64 + 64 * (d // 64), 64 columns -- the SAME
    lane map as the first body; only its column base moves (CFG.DQ_COL = 64 = S^T's dead tail; tmem_map dq_576: CFG_DQ576.DQ_COL = 512 = past dK).
    Operand bytes per CTA per q tile: Q1 16 + Q2 16 + dO1 16 + dO2 16 KiB; per kv block K1 32 + K2 32 + V 32 KiB.
    TS k-step order: the packed P^T / dS^T columns follow the softmax quarter ownership (group g owns q quarters {g, g+2}),
    so TS k-step k (8 packed columns = 16 q) pairs with the B q block CFG.ts_k_perm(k) = [0, 1, 4, 5, 2, 3, 6, 7][k].

4. WARP MAP        role | warps | regs (setmaxnreg) | what it credits (names = the audit's table)
    softmax   | 0-7   | 144 | s_read (NEW; s_tail map only: pair-wide on the leader, after the S^T ld pair, EVERY tile), p_ready, ds_ready (pair-wide),
                             ds_tmem (local), stats consumption (ldg: the per-tile `_stats` fetch under the SOFTMAX_BARRIER pair; tma_ring: the
                             stats_full[2] wait after the s_read arrive + the stats_empty[2] bare release (256) after _ds_values, stats_state
                             st(0) +1 per tile: the LDG chain, its STS and both `bar.sync 1,256` are not traced); mask_form predicated
                             (section 11): the keep-word mask body of _p_values only on `tile_may_mask` tiles (a CTA-uniform per-tile scalar:
                             no live register added, no barrier touched)
    epilogue  | 8-11  | 144 | dp_empty (leader; MOVED: after the dS^T readout, before the recv arm), ds_smem (own half by STS), [rubin_b2: ds_smem[s] /
                             ds_recv_full[s] of stage s = sds_state.idx; dq_read_done (leader, bare 128 x 2) after EACH dQ readout trio: in-loop under
                             `qtile > lo` and at the item end under `lo < hi`]
                             ds_recv_full (expect_tx) + the XCHG ship, s_tail_empty (NEW; s_tail map only: leader, after the dQ readout), dq_full x 2
                             slots x 2 rounds per tile, dv_full, dk_full
    mma       | 12    |  80 | LEADER only: s_done, dp_done, dq_done (commit, pair multicast; dq_done in-tile for dQ(t-1) AND at the
                             tail for dQ(hi-1)), *_empty releases; the follower's MMA warp runs the quiet body (alloc, the
                             ds_pair_ready relay, dealloc)
    tma_ldg   | 13    |  80 | q1_full[1] (ONE stage), q2_full, do1_full[2], do2_full, k1_full, k2_full, v_full (expect_tx on the
                             LEADER for 2x bytes; the follower issues its half naming the leader's mbar); dv_full (32, fix A);
                             stats_form tma_ring (section 11): stats_full[2] (expect_tx STATS_TX 1024 on its OWN copy, pred=elect_sync(), +
                             ONE 1 KiB bulk copy per q tile right after Q1(t)) after its stats_empty[2] credit wait (stats_ld st(1), +1 per tile)
                             -- RESOURCE-NOTE: split 144/144/80 unchanged; loader +~6 live regs: STL/LDL 0 gate; fallback 144/136/88
    tma_stg   | 14    |  80 | dq_empty x 2 slots per ROUND (reduce-add source read drained; the closed per-round release), dvk_empty
                             (leader), k2_empty
    sched     | 15    |  80 | sched_full / sched_empty (CLC payload delivered to the pair); never credits its own ring
    SUM(role regs x warps) = (8 x 144 + 4 x 144 + 4 x 80) x 32 = 65536 == entry 128 x 16 warps x 32; every count % 8 == 0.
    The epilogue carries 32 fp32 of dQ(t-1)'s second slice across the dS(t) staging (~96 live of 144): the vs-sass STL / LDL == 0
    pin is owed on this body (the PTX census pins `.local` == 0).
    STAMPS (the typed field ``debug_clk``, section 11): ONE representative warp per role -- softmax warp 0, the epilogue LEAD warp 8, the
    single service warps 12 / 13 / 14 -- brackets each of ITS mbarrier waits with two %clock64 reads (the wait itself unmodified) and its
    elected lane accumulates per (role, barrier) into the Int32 record in the bar block's spare (config DBG_CLK); no barrier, no fence,
    no SMEM layout change; the scheduler warp's library loop is not instrumented.  Default off = no traced code.
    ALTERNATE ROLE SHAPE (typed field ``warp_roles = "16w_fa4"``; the SAME 16 warps, the SAME split):
    softmax   | 0-7   | 144 | + the dS STS: 4 own-half + 4 XCHG STS.128 per lane per tile straight from its 32 packed dS^T words (= today's LDTM
                             image word for word, the resource plan; group g writes vec 4g..4g+3 of BOTH 64-q halves), ONE fence.proxy.async,
                             ds_smem (NOW 256 = SOFTMAX_LANES, bare), the 256-lane XCHG named barrier, warp SOFTMAX_LEAD_WARP's elected lane arms
                             ds_recv_full + ships XCHG; a WAR wait on dq_war_done(t-1) (`dq_war` st(1), one wait per live tile) before the first
                             dS STS of every tile.  ds_tmem GONE.  +4..8 transient registers (the words are live for the STTM anyway); STL / LDL 0 = the gate
    reduce    | 8-11  | 144 | (the former epilogue, `_reduce_loop`): dq_done(t) -> the x64 dQ readout -> s_tail_empty -> rounds A, B (no `carried`
                             half, no item-end round) -> the item-end dV / dK native staging (dv_full 160, dk_full 128).  No dS work: ~80-90 live
    mma / tma_ldg / tma_stg / sched: unchanged (dp_empty GONE from the leader: no slot-1 wait, no state, no drain)
    The split 144/144/80 is unchanged under both shapes (pool 2048); the softmax carries +4..8 transient registers under 16w_fa4.

5. BARRIER TABLE  (the barrier audit's table; ``config_sm100_d128_accum_2cta_stail.BAR_LAYOUT`` = offsets / counts / kinds; 35 names /
   40 slots in [230400, 230960) of the 944-B bar block, re-packed, 16 B per row; under warp_roles 16w_fa4 ``BAR_LAYOUT_XFA4`` = 33 names /
   38 slots in [230400, 230928): ds_tmem and dp_empty deleted, dvk_stage_empty unserved, dq_war_done added).  L = pair LEADER
   (rank 0, the only MMA issuer), F = FOLLOWER (rank 1); "(-)" = the follower's copy is initialised with the SAME count and never armed /
   never waited; N = hi - lo q tiles of the item; x T = per CLC item.  Issuing lanes come from the guard form ONLY: bare 1 warp = 32, bare
   8 warps = 256, pred / elect = 1 per warp, pred=is_leader & elect = 1 for the pair, a tcgen05.commit = 1 arrive per TARGET CTA; x the
   CTAs that reach the site.  Rows marked NEW / MOVED / SPLIT are the arm's; every other row is the first body's AUDIT (23:10) line.
    name[st]         kind              init L / F     SUM(issuing lanes) == init        waiter (own copy unless "L only")   start  fires
    q1_full[1]       TMA_LOAD          1 / 1(-)       1 (L, pred=is_leader & elect) == 1  MMA(L) before every S^T issue     st(0)  x N
        arm: loader `q1_full[s].arrive(n_bytes=Q1_TX, pred=is_leader & elect_sync())` (s = the single slot); BOTH CTAs issue tma_load_tile(cta_group=2)
        ONE stage (Q1_STAGES 1: FAv4's Q_stage = 1 under 2-CTA funds the sDQ ring; Q1(t+1) has ~2000 clk after q1_empty(t): 4 GEMMs between S^T issues)
    q1_empty[1]      MMA_COMMIT        1 / 1          1 commit x 2 target CTAs == 1 each  loader L+F before the Q1 reload     st(1)  x N
        site: MMA(L) right after each S^T issue (prologue + lookahead), `_commit_pair(bars.q1_empty[s], elect_p)`; DRAIN x Q1_STAGES = 1
    q2_full[1]       TMA_LOAD          1 / 1(-)       1 (L) == 1  (Q2_TX)                 MMA(L) before dK(t)               st(0)  x N
    q2_empty[1]      MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x N  (after dK(t); DRAIN x1)
    do1_full[2]      TMA_LOAD          1 / 1(-) /slot 1 (L) per half slot == 1 (DO1_HALF_TX each, range_constexpr(2))  MMA(L) dP^T k-half 0 | 1  st(0)  x N
    do1_empty[1]     MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x N
            (after dP^T(t); DRAIN x1)
    do2_full[1]      TMA_LOAD          1 / 1(-)       1 (L) == 1  (DO2_TX)                MMA(L) before dV(t)               st(0)  x N
    do2_empty[1]     MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x N  (after dV(t); DRAIN x1)
    k1_full[1]       TMA_LOAD          1 / 1(-)       1 (L) == 1  (K1_TX)                 MMA(L) before S^T(lo)             st(0)  x T (live)
    k1_empty[1]      MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x T
            (block end, after the tail dQ; DRAIN x1)
    v_full[1]        TMA_LOAD          1 / 1(-)       1 (L) == 1  (V_TX)                  MMA(L) in the prologue            st(0)  x T (live)
    v_empty[1]       MMA_COMMIT        1 / 1          1 x 2 == 1 each                      loader L+F                        st(1)  x T  (block end; DRAIN x1)
    k2_full[1]       TMA_LOAD          1 / 1(-)       1 (L) == 1  (K2_TX)                 MMA(L) in the prologue            st(0)  x T (live)
    k2_empty[1]      THREAD            32 / 32        store warp bare after tma_store_wait(0) = 32 == 32   loader (own) before K2  st(1)  x T (every item)
        the MMA's read of K2 is covered by dq_done(t_last) (observed by the epilogue before dv_full) -> dV store -> drain -> this arrive;
        BACK-EDGE (fix A carried): the loader observes it right before the K2 reload (live) or once at the item top (zero-trip), then arrives
            dv_full (bare 32); the store's NEXT k2_empty arrive follows its dv_full wait (P14-B)
    s_done[1]        MMA_COMMIT        1 / 1          1 x 2 == 1 each                      softmax L+F                       st(0)  x N
    dp_done[1]       MMA_COMMIT        1 / 1          1 x 2 == 1 each                      softmax L+F                       st(0)  x N
    dq_done[1]       MMA_COMMIT        1 / 1          1 x 2 == 1 each: after dQ(t-1) [tile t slot 2], after the tail dQ(hi-1), + one EMPTY
                                                      commit per zero-trip item            epilogue L+F (dq_state)           st(0)  x N (+1 dead)
    p_ready[1]       LEADER (relaxed)  512 / 512(-)   softmax 8 warps bare x 2 CTAs = 256 x 2 = 512 == PAIR_SOFTMAX_LANES   MMA(L) before dV(t)  st(0)  x N
        site: after tcgen05_wait(STORE) + tcgen05_fence(BEFORE_THREAD_SYNC): `_arrive_leader(bars.p_ready)` (TMEM data: relaxed, P8)
    ds_ready[1]      LEADER (relaxed)  512 / 512(-)   same lanes, after the dS^T pack = 512 == 512            MMA(L) before dK(t)    st(0)  x N
    s_read[1] NEW    LEADER (relaxed)  512 / 512(-)   softmax 8 warps bare x 2 CTAs after the S^T ld pair (`_s_values`) + tcgen05_wait(LOAD)
                                                      + tcgen05_fence(BEFORE_THREAD_SYNC), BEFORE the exp body, EVERY tile (D2: no runtime
                                                      guard on a 512-lane arrive) = 256 x 2 = 512 == PAIR_SOFTMAX_LANES
                                                      waiter MMA(L) at slot 2 of EVERY tile, before the conditional dQ(t-1) (ready_state); at
                                                      t == lo dominated by the p_ready(lo) wait that follows             st(0)  x N  (in-loop)
        tmem_map dq_576: ROW DELETED (no field, no init, no arrive, no wait, no state): the S tail [64, 128) has no second tenant; the S^T(t+1)
        WAR over the softmax's S^T(t) read rests on p_ready(t) (waited at slot 3, before S^T(t+1) at slot 4) as in the first body.
    ds_tmem[1]       THREAD            256 / 256      softmax 8 warps bare, local, at the ds_ready site = 256 == 256   epilogue (own)   st(0)  x N
    dp_empty[1] MOVED LEADER (relaxed) 256 / 256(-)   epilogue 4 warps bare x 2 CTAs right after the dS^T 4 x tcgen05.ld + tcgen05_wait(LOAD)
                                                      + fence, BEFORE `ds_recv_full.arrive` (moved off the dQ readout) = 128 x 2 = 256 ==
                                                      PAIR_EPILOGUE_LANES                  MMA(L) before dP^T(t+1) (dpe) st(1)  x N; DRAIN x1 (MMA(L))
        dp_empty = dS^T readout only (arrive before the XCHG STS); dP^T(t+1) never waits dQ
    s_tail_empty[1] NEW LEADER (relaxed) 256 / 256(-) epilogue 4 warps bare x 2 CTAs right after `_tmem_load(DQ_COL, 64)` + tcgen05_wait(LOAD) +
                                                      tcgen05_fence(BEFORE_THREAD_SYNC) (the first body's dp_empty site) = 128 x 2 = 256 ==
                                                      PAIR_EPILOGUE_LANES.  waiter MMA(L) (s_tail_state, +1 per wait) at the THREE sites of
                                                      D1: the prologue S^T(lo) [always; pre-armed for the kernel's first item], the lookahead
                                                      S^T(t+1) iff qtile > lo, the tail dQ(hi-1) iff hi - 1 > lo       st(1)  x N (one per dQ readout)
                                                      ; DRAIN x1 (MMA(L), after the loop: the last item's tail readout, P15)
        s_tail_empty right after wait(LOAD): S^T(t+1) waits the readout, never the STS
        tmem_map dq_576 (the barrier skeleton R1 2026-11-03, sections 1.1-1.6): ROW DELETED -- the D1 (i) / (ii) / (iii) waits, the P15 drain, the
        leader's s_tail state and the epilogue's arrive are not traced.  The dQ image [512, 576) has ONE writer (the dQ GEMM, slot 2 of tile t+1 /
        the tail) and ONE reader (the epilogue's readout after dq_done(t)): WAW by the in-order tensor pipe, RAW by dq_done; its REUSE edge dQ(t)
        readout -> dQ(t+1) write is the epilogue's PROGRAM ORDER: the dq_done(t) wait + LDTM + tcgen05_wait(LOAD) + tcgen05_fence(BEFORE_THREAD_SYNC)
        of its tile t precede its tile t+1's own-half STS + fence_proxy + tcgen05 fence + ds_smem.arrive() (release) -> MMA(L) _wait_tcgen(ds_smem)
        (acquire + fence::after_thread_sync) -> dQ(t+1); the follower's half composes through its ds_smem -> the relay's arrive_on_peer(release_cta)
        -> ds_pair_ready(t+1).  LOAD-BEARING SITE (P14-B): the epilogue's readout trio sits BEFORE the next tile's staging in loop order -- any
        staging-before-readout reorder (a lagging readout under a deeper sDS ring; the 16w_fa4 reduce group, which declines the map by name) voids
        the carrier and needs an explicit dq_read_done row; the test module pins the order (RED-first source assertion).  N-walk: N = 1 the tail
        dQ(lo)'s previous tenant is the previous live item's dQ(hi'-1), read out before that item's end rounds and before this item's dS(lo) STS;
        N = 2 / 3 every in-tile and tail issue is gated by the dS gates of the tile after the readout; zero-trip: no write, no readout, no gate
        flip; the kernel's first item: no tenant.  Tables after the deletion: 16w x tma_ring 35 names / 42 slots, END 231984 (BAR_LAYOUT_RING_DQ576);
        16w x ldg 33 / 38, END 230928 (BAR_LAYOUT_DQ576); kinds leader 6 -> 4; every kept row identical in count, kind, waiter and cadence.
    ds_smem[1]       THREAD            128 / 128      epilogue 4 warps bare after own-half STS + fence_proxy + tcgen05 fence = 128 == 128;
                                                      NO dq_empty wait before the STS any more (the staging alias is deleted)
            MMA warp (own): L dQ issue on xchg_state (in-tile + tail), F relay (relay_state)   st(0) x N
    ds_recv_full[1]  TMA_LOAD (dsmem)  1 / 1          epilogue warp 8, pred=is_lead_warp & elect = 1 == 1; tx = the PEER's cp.async.bulk XCHG_TX
            MMA warp (own): L on xchg_state, F relay  st(0) x N
    ds_pair_ready[1] THREAD (remote)   1 / 1(-)       MMA(F) `arrive_on_peer(0, pred=elect_p, release_cta=True)` after ds_smem + ds_recv_full = 1
                                                      -> L == 1                            MMA(L) before each dQ issue (xchg_state)  st(0) x N
        SPLIT (P2): the three dS gates ride `xchg_state` (+1 per dQ ISSUE: slot 2 of tiles lo+1 .. hi-1 and the tail), ONE tile behind
            `ready_state` (p_ready / ds_ready / s_read, +1 per tile).  On ready_state they would park on completion t while the epilogue's
            dS(t) STS waits dq_done(t-1) = the dQ(t-1) issue BEHIND the wait: deterministic HANG at tile lo+1 (section 8 of the table, RED 4)
    dq_empty[2]      THREAD            32 / 32 /slot  store warp bare after each ROUND's source reads: tma_store_wait(1) -> the slot-0 arrive,
                                                      tma_store_wait(0) -> the slot-1 arrive = 32 == STORE_LANES; NO bootstrap (D3)
            epilogue (own, empty_state st(1)): all 4 warps wait BOTH slots before each round (D4 pair: closed per-round release + both-slots
            wait form; "change both or neither")       x 2N per slot (2 rounds per tile)
    dq_full[2]       THREAD            64 / 64 /slot  epilogue warp pair dhalf (warps 8-9 -> slot 0, 10-11 -> slot 1) bare after the slice STS +
                                                      fence_proxy = 2 x 32 = 64 == DQ_SLICE_LANES, per round
            store (own, dq_state 2 stages, +1 per USE): uses per tile (slot 0, slice 0), (slot 1, slice 2), (slot 0, slice 1), (slot 1, slice 3)
                                                                                                          st(0)  x 2N per slot
    dv_full[1]       THREAD            160 / 160      epilogue 4 warps bare after the dV staging STS + fence = 128 + loader bare after its
            k2_empty observation = 32 -> 160 == DV_FULL_LANES   store (own)   st(0)  x T   (fix A carried)
    dk_full[1]       THREAD            128 / 128      epilogue 4 warps bare after the dK staging STS + fence = 128 == 128    store (own)   st(0)  x T
    dvk_empty[1]     LEADER (relaxed)  64 / 64(-)     store warp bare x 2 CTAs after tma_store_wait(0) = 32 x 2 = 64 == 64   MMA(L) item top
            st(1)  x T; DRAIN x1 (MMA(L))
    sched_full[2]    TMA_LOAD (CLC)    1 / 1          scheduler(L) elect: local arrive(n_bytes=CLC_TX) + mapa(F) mbarrier_arrive_expect_tx(CLC_TX
            , CTA) = 1 per CTA == 1; completion: clusterlaunchcontrol.try_cancel(multicast=1) lands 16 B in EACH CTA's slot
            ; waiters: every consumer warp (own copy) + scheduler(L)  st(0) x T
    sched_empty[2]   THREAD (remote)   30 / 30(-)     read_tile_id_arrive_on_leader(mb, CGA_M) from 15 warps per CTA (all but the scheduler) x 2 = 30 == 30
        waiter: scheduler(L) before re-arming the slot (every phase observed in-loop)   st(0) x T
    tmem_dealloc[1]  THREAD (+peer)    2 / 2          MMA warp, each CTA, after the final CTA sync: `if elect: arrive(); arrive_on_peer(peer rank)` = 1 + 1 == 2
        waiter: the MMA warp of each CTA, wait(0) -> tcgen05_fence -> tcgen05.dealloc.cta_group::2  x 1
    stats_form tma_ring (section 11; the barrier audit's S1 SKELETON 2026-10-27 17:30 + its 17:46 reconciliation with the bulk-copy form,
    resources/s1-barrier-table.md sections 1 / 2 / 14; the resource plan s1-stats-ring.md; ``config_sm100_d128_accum_2cta_stail.BAR_LAYOUT_RING``
    = the 35 rows above RE-BASED onto CFG_RING.BARS 231424 (+1024 per row: ring stage 1 took the 1 KiB the scaffold sat on) + the two rows below,
    37 names / 44 slots, END 232016; LOCAL rows: each CTA arms / waits its OWN copy, no "(-)" copy, no pair interaction; AUDIT of the committed
    body owed; NOT built under 16w_fa4 (declines by name)):
    name[st]         kind              init L / F     SUM(issuing lanes) == init        waiter (own copy)                  start  fires
    stats_full[2] NEW TMA_LOAD / LOCAL  1 / 1 /stage   loader warp 13 `bars.stats_full[s].arrive(n_bytes=CFG.STATS_TX, pred=own_pred)`, own_pred =
                     (kind "tma", the                 elect_sync() ONCE per item (1 per warp = 1 per CTA; NEVER is_leader: a plain bulk copy lands on
                     sched_full kind)                 the ISSUING CTA) = 1 == ONE_LANE; tx = ONE `bulk_copy` of the (head, q tile) record = 1024 ==
                                                      STATS_TX (2 x 128 fp32: the LSE row, the delta row; a FULL record every tile incl. the ragged
                                                      tail: the pad rows are real 0.0 words of the padded region, no OOB clip, no partial arm)
                                                      waiter: the 8 softmax warps, `_wait` (plain: no tcgen05 fence, no proxy fence -- the copy's
                                                      complete_tx + the acquire wait publish the async-proxy write to every lane) AFTER
                                                      `_arrive_leader(bars.s_read)` and BEFORE `_p_values`; stats_state st(0), +1 per tile
                                                                                                                                st(0)  x N
        site (loader, iteration t): after Q1(t)'s tma_load_tile and BEFORE `_wait(bars.do1_empty)`: `_wait(stats_empty[s])` -> arm -> `bulk_copy(
        stage s, stats2 record (head, rowoff[b] / N + t), STATS_TX, bars.stats_full[s].smem_ptr)` (internal elect; `cp.async.bulk.shared::cluster
        .global.mbarrier::complete_tx::bytes`, no cta_group modifier, no multicast); `stats_ld = advance(stats_ld, CFG_RING.STATS_STAGES)`
    stats_empty[2] NEW THREAD / LOCAL   256 / 256 /st  the 8 softmax warps bare `bars.stats_empty[s].arrive()` right after `words = _ds_values(..)`
                                                      (the tile's LAST stats read: the delta LDS) and before the STORE wait = 8 x 32 = 256 ==
                                                      SOFTMAX_LANES (the arrive's release orders the lane's LDS; the loader's acquire wait + program
                                                      order gate the next copy into the stage: no proxy fence)
                                                      waiter: the loader, `_wait(bars.stats_empty[s], stats_ld.phase)` before the arm; stats_ld
                                                      st(1) PRE-ARMED (P5 b: wait k <-> release k - 2 = the stage's previous tenant; the kernel's
                                                      first two waits pass on the pre-arm; NO bootstrap arrive)                   st(1)  x N
        P2: ONE `stats_state` (softmax) / ONE `stats_ld` (loader) per CTA indexes BOTH the barrier and the SMEM stage base, declared BEFORE the
        persistent loop and CARRIED across items: tile lo of an item takes stage K % 2 for K = this CTA's running tile count (both states advanced
        exactly once per tile, never on a zero-trip item, NEVER reset at the item top -- a reset at an odd K puts the copy and the wait on different
        stages = the P2 hang).  P14 flips per item (both CTAs, pair-uniform lo / hi): stats_full N / N, stats_empty N / N; zero-trip 0 / 0 (no site
        outside the two tile loops); n_q == 1: 1 / 1; kernel-wide W copies == W waits, the last two releases unwaited (LOCAL THREAD: harmless).
        P14-B back-edges: stats_full[s]'s next completion needs the loader past the softmax's release of the stage, which follows the softmax's own
        wait; stats_empty[s]'s next release needs the softmax past the next copy, which needs the loader past the release -- the closed pair, no lap.
        P15 / P6: no cross-CTA arrive on either row, every issued copy consumed in-loop before the final CTA sync -> no drain anywhere.
        Named barrier CFG.SOFTMAX_BARRIER (id 1, 256): RETIRED under tma_ring (its only two sites were `_stats`' fences); id 2 XCHG unchanged.
        STAMPS: +2 words `wait:softmax:stats_full`, `wait:loader:stats_empty` appended after every ldg word (indices 82 / 83); the record =
        (TMEM_HANDLE - BAR_LAYOUT_RING_END) / 4 = 88 words (84 used), DBG_CLK_RING; the ldg record stays 96 words at 230960.
    DIAGNOSTIC PROBES vs this table: `exchange_off` moves NO row -- the epilogue's `ds_recv_full.arrive(n_bytes=..)` keeps its one elected
    lane and arms CFG.DIAG_EXCHANGE_OFF_TX = 0 bytes, so each phase completes AT ITS OWN ARM (1 == 1, nothing in flight: no tx-pending
    over-arrival window exists), the XCHG ship is not issued, and the three dS gates (xchg_state), the follower relay, the XCHG STS / proxy
    fence / CFG.XCHG_BARRIER and every other wait and arrive stay; `stats_one_chunk` and `sdq_sts_off` touch no barrier (LDS / STS bytes
    only); STAMPS `debug_clk` adds clock reads and SMEM words only.  The WAIT probes `s_tail_wait_off` / `s_read_wait_off` /
    `s_tail_and_s_read_wait_off` (section 11) move NO row either: they delete the LEADER's `_wait_tcgen` SITES of the s_tail_empty row
    (the four D1 / P15 sites above) and / or the s_read row (slot 2); every arrive (the epilogue's 128 x 2, the softmax's 256 x 2) and every
    state advance stays, so the init counts, the lane ledger and the flip counts N / N are this table's.  Both rows are LEADER (relaxed)
    thread-count rows with NO tx bytes: a phase that completes with nobody parked on it is simply the next phase's base (an mbarrier has no
    "unobserved completion" fault; only a tx-count arm on a pending phase over-arrives, P3), so the skipped waits cannot fault or hang -- they
    let S^T(t+1) / dQ(t-1) land on live TMEM: OUTPUT WRONG BY DESIGN.  A probe that skipped the three dS gates is NOT in the domain: skipping
    them removes the ONLY back-edge of the ds_recv_full arm (arrive(t+1) <- dq_done(t) <- the ds_recv_full(t) / ds_pair_ready(t) waits),
    so the arm of tile t+1 would land on a tx-pending phase = an over-arrival fault.
    dkv_accum fp32_reduce (; the typed op-level field, ``dkv_fp32`` Constexpr[bool], default False = this table
    byte-identical; the resource plan): the item-end dV / dK publications carry fp32 slices instead of bf16 halves --
    dv_full (160, the same lanes and sites) publishes ROUND 1 = dV's four slices + dK's slice 0 (five 16 KiB slots: sK2's halves, the dS slab's
    halves, XCHG), dk_full (128) publishes ROUND 2 = dK's slices 1..3 (three of those slots).  NEW ROW dvk_stage_empty[1] THREAD 32 / 32 LOCAL
    (BAR_LAYOUT @ 230944, appended after tmem_dealloc; INIT ONLY under dkv_fp32): the store warp arrives it bare (32 == init) after round 1's
    bulk group drained (`tma_store_commit` -> `tma_store_wait(0)`, P7); the epilogue's 4 warps wait it (own, stage_state st(0), +1 per item
    incl. zero-trip) between round 1's dv_full arrive and round 2's STS = the WAR edge of the five slots; x 1 per item; no drain (every phase
    observed in-loop by the epilogue before its round 2).  The dq_empty ring is NOT touched (D3 / D4 hold): its pre-armed accounting (wait k
    <-> release k-1) cannot carry an item-end credit -- the first build tried it and round 2 overwrote round 1 under the reduce reads.  Every
    other row unchanged (dvk_empty / k2_empty still after the final drain = round 2's).  PTX census: arrive sites 41 / wait sites 69 under
    fp32_reduce (40 / 68 under the default); reduce-adds 4 -> 12, bf16 tile stores 4 -> 0; the default rendering has no instruction on the row.
    Lane ledger: every row above written as SUM == init on BOTH CTAs the barrier audit, PASS 0); unresolved: none.  Non-mbarrier
    sites: named barrier 0 (3 x barrier_cta_sync: init, TMEM-base publish, final), CFG.SOFTMAX_BARRIER (256 lanes, 2 per _stats; ldg only --
    0 sites under stats_form tma_ring), CFG.XCHG_BARRIER (128 lanes, 1 per tile, P11 KEEP: the ship's operand is all four warps' XCHG stores).
    the alternate role shape (``warp_roles = "16w_fa4"``; the barrier audit;
    ``config_sm100_d128_accum_2cta_stail.BAR_LAYOUT_XFA4`` = 32 names / 37 slots / END 230912, re-packed; its rows = the 34 above MINUS:
      ds_tmem[1]   DEL  (THREAD 256: producer and consumer both vanish -- the softmax writes sDS from registers, nobody reads dS^T out of TMEM)
      dp_empty[1]  DEL  (LEADER 256: its ONLY arrive site was the epilogue's dS^T readout; dP^T(t+1) over the packed dS^T(t) [128,192) is ordered by the
                        in-order tensor pipe after dK(t) (issued first by the same thread); over dP^T(t)'s fp32 [128,256) by the softmax's ld +
                        wait(LOAD) -> STTM -> wait(STORE) -> fence -> ds_ready(t) (512) -> the leader's dK(t) wait -> dK(t) -> dP^T(t+1): both edges
                        hold in-tile, across the item boundary, around a zero-trip item and for the kernel's first dP^T (the barrier audit))
    and with these rows CHANGED (every other row: the same object, count, kind, waiter and cadence; the epilogue's rows RENAMED to the reduce group):
    name[st]         kind              init L / F     SUM(issuing lanes) == init        waiter (own copy unless "L only")   start  fires
    ds_smem[1] CHG   THREAD            256 / 256      softmax 8 warps bare after the 4 own-half + 4 XCHG STS + ONE fence.proxy.async (no tcgen05 fence:
                                                      the words come from registers) = 8 x 32 = 256 == SOFTMAX_LANES; BEFORE the XCHG bar.sync
            MMA warp (own): L dQ issue on xchg_state (in-tile + tail), F relay (relay_state)   st(0) x N   (the epilogue's 128-lane site DELETED)
    ds_recv_full[1] CHG TMA_LOAD (dsmem) 1 / 1        softmax warp SOFTMAX_LEAD_WARP (0), `arrive(n_bytes=XCHG_TX, pred=ship_pred)`, ship_pred =
                                                      (warp == 0) & elect_sync() ONCE per item = 1 == 1; AFTER the 256-lane XCHG bar.sync, IMMEDIATELY
                                                      before `_ship_xchg` (form B); tx = the PEER softmax's cp.async.bulk of ITS XCHG = 16384 == XCHG_TX
            MMA warp (own): L on xchg_state, F relay  st(0) x N.  The arm's BACK-EDGE on every path (the 08:01 rule) = the softmax's dq_done(t-1) WAR
            wait: dq_done(t-1) <- dQ(t-1) issued by the leader after ds_recv_full(t-1) [copy(t-1) landed on CTA 0] AND ds_pair_ready(t-1) <- the relay's
            ds_recv_full(t-1) wait [copy(t-1) landed on CTA 1]; item top: the tail dq_done(hi-1) of item k; after a zero-trip item: the empty commit,
            issued by the leader AFTER the previous live tail in program order.  ds_gates_off  would delete the edge on both CTAs: declined BY NAME.
    dq_done[1]       MMA_COMMIT        1 / 1          UNCHANGED from the row above (the barrier audit: the softmax waiter set of the first build is
                                                      REMOVED): producer dQ(t-1) at slot 2 / the tail dQ(hi-1) / the zero-trip empty commit; waiter the
                                                      REDUCE group `dq_state` st(0) (the readout + the zero-trip observation) ONLY   N (+1) / N (+1)
    dq_war_done[1] NEW MMA_COMMIT      1 / 1          MMA(L) `_commit_pair(bars.dq_war_done, elect_p)` INSIDE _dq_issue at the TWO dQ issue sites ONLY
                                                      (in-tile dQ(t-1), the tail dQ(hi-1)), IMMEDIATELY BEFORE the dq_done commit; NEVER at the zero-trip
                                                      empty-commit site = 1 arrive per target CTA.  Waiter: the SOFTMAX ONLY, `dq_war` st(1) PRE-ARMED,
                                                      +1 per wait, `_wait` (plain) before the first dS STS of every LIVE tile (after the STTM + wait(STORE)
                                                      + fence + ds_ready: dK(t) never waits dQ(t-1)); NO zero-trip wait, NO zero-trip advance; ONE
                                                      unconditional drain `_wait` after the persistent loop.  Flips per item: live N / N (wait k of the item
                                                      consumes flip k-1: the first consumes the previous live item's tail flip or the pre-arm); zero-trip
                                                      0 / 0 (no site executes); n_q == 1: 1 / 1; kernel: W flips, W + 1 waits (the drain consumes the last
                                                      tail; W = 0 passes on the pre-arm).  LAP BOUND: flip w <- dQ(w) <- ds_smem <- the softmax's STS after
                                                      wait w; wait w-1 returned only after flip w-2 landed -> F in {w-1, w} at EVERY wait: the pre-armed
                                                      waiter can never find a flip landed early (the first build's shared dq_done waiter raced the
                                                      leader's zero-trip EMPTY commit: a zero-trip item as a cluster's first item HUNG at 10:24, RULE-GAPS
                                                      649; consecutive zero-trips = a lap).  The arm's back-edge (section 5.3) on every path = a DATA edge
                                                      through the previous live tile's / item's dq_war_done flip = a real dQ issued behind the leader's gates.
    s_tail_empty[1] RENAMED  LEADER (relaxed) 256 / 256(-)  the REDUCE group (the same lanes) after the x64 dQ ld + wait(LOAD) + fence, BEFORE round A;
                                                      waits / D1 sites / drain unchanged
    dq_empty[2] / dq_full[2] RENAMED  THREAD           32 / 32 per slot ; 64 / 64 per slot  the REDUCE group waits BOTH slots per round (D4 pair) / arrives
                                                      dq_full[dhalf] per round; rounds A then B inside the tile (no carried half): 2N / 2N per slot
    dv_full[1] / dk_full[1] RENAMED  THREAD            160 / 160 ; 128 / 128  the REDUCE group's item-end native staging (X9) + the loader's 32 (fix A)
    named XCHG_BARRIER (id 2) CHG                      bar.sync 2, 256 = CFG.xchg_barrier_lanes("16w_fa4") = SOFTMAX_LANES, ONE per tile by all 8 softmax
                                                      warps after the 8 STS + the proxy fence + the ds_smem arrive; P11 KEEP (the ship's operand is all
                                                      8 warps' XCHG stores); the inspector REJECTED folding it into _stats' SOFTMAX_BARRIER (X3)
    s_read[1]        KEPT exactly as above (X1; the merge into a LEADER_RELEASE deferred s_read is a STAMPS-decided follow-up, the barrier audit)
    dQ ring          FORM A kept (X6): the store warp issues the UTMAREDG; the ring rows unchanged in count, kind, waiter and the D4 pair
    BAR_KIND_COUNTS_XFA4 {tma_leader 7, mma 10, thread 7, leader 5, peer 2, tma_dsmem 1, tma 1} = 33 (BAR_LAYOUT_XFA4: 33 names / 38 slots / END
      230928, dq_war_done appended after tmem_dealloc).  Lane ledger (inspector sections 2 + 15.3, PASS 0): ds_smem 256 == 256 (8 x 32 bare, every
      tile) | ds_recv_full 1 == 1 (warp 0 elect, tx 16384 from the peer) | dq_done 1 == 1 per CTA (the reduce group) | dq_war_done 1 == 1 per CTA (the
      softmax) | s_tail_empty 128 x 2 == 256 | dq_empty 32 == 32 | dq_full 64 == 64 | dv_full 128 + 32 == 160 | dk_full 128 == 128 | s_read 512 == 512 |
      ds_pair_ready 1 == 1 | XCHG id 2: 256 threads == 256 | SOFTMAX id 1: 256 == 256; DELETED ds_tmem / dp_empty: no init, arrive, wait, drain, state.
    PER-ITEM FLIPS under 16w_fa4 (both CTAs): ds_smem N / N; ds_recv_full N / N; ds_pair_ready N / N; dq_done N (+1) / N (+1) (the reduce group);
      dq_war_done N / N (the softmax; + the ONE kernel-end drain); s_read N / N; s_tail_empty N / N; dq rings 2N / 2N per slot; p_ready / ds_ready /
      s_done / dp_done / do* / q2 / q1 N / N; k1 / v / k2 / dvk_empty / k2_empty / dv_full / dk_full / sched 1 / 1; named XCHG N; SOFTMAX 2N.  Zero-trip
      item (any position: first, consecutive, after live, THD empty): dq_done 1 / 1 (the reduce group), dq_war_done 0 / 0, dvk_empty / k2_empty /
      dv_full / dk_full / sched 1 / 1, everything else 0.  N == 1: ds_smem 1 / 1, dq_done 1 / 1, dq_war_done 1 (the tail) / 1 (tile lo consumes the
      previous tail or the pre-arm; its own tail flip goes to the next live item's first wait or the drain), s_tail_empty 1 / 1 (prologue), xchg gates
      1 / 1 (tail).  Half-dead pair: CTA 1's softmax STS zeros, ships zeros, arrives everything: identical counts.
    FENCE TABLE under 16w_fa4 (the barrier audit): F1 softmax after the 8 STS: ONE fence.proxy.async (generic STS -> the bulk copy's read of XCHG +
      the UMMA's read of sDS own) -> ds_smem.arrive -> bar.sync 2,256 -> arm -> ship [replaces the epilogue's two fences + its tcgen05 fence]; F2 / F3
      the S^T ld pair / the STTMs: tcgen05_wait + tcgen05_fence(BEFORE) before s_read / p_ready / ds_ready (unchanged); F4 the dq_war wait: plain
      `_wait` (a WAR of an async read by a generic write: the commit -> mbarrier acquire IS the order; no proxy fence exists or is needed); F5 the reduce
      group's x64 ld: wait(LOAD) + fence before s_tail_empty (unchanged); F6 each round's slice STS: fence.proxy.async before dq_full (unchanged); F7
      _native_out (unchanged); F8 the ship: none after (async -> async); F9 the store warp's wait_group.read before every credit (unchanged); F10 init.
    PER-ROLE STUB ORDER under 16w_fa4 (inspector sections 9 + 15.4 = the `# BARRIER` punch lists of _softmax_loop / _reduce_loop / _mma_loop_leader
      below): softmax per live tile: ... ds_ready -> `_wait(dq_war_done, dq_war)` -> advance -> 8 STS -> ONE fence -> ds_smem -> bar.sync 2,256 -> arm ->
      ship; nothing on dq_war at a zero-trip item; the ONE drain `_wait(dq_war_done)` after the loop.  MMA(L) `_dq_issue`: gates -> dQ MMA ->
      `_commit_pair(dq_war_done)` -> `_commit_pair(dq_done)` -> xchg += 1; the zero-trip site: `_commit_pair(dq_done)` only.
    PER-ITEM FLIPS (N = hi - lo; every path flips every ring the same number of times on BOTH CTAs: lo / hi are pair-uniform):
      s_tail_empty N fires (one readout per dQ issue) / N waits [prologue 1 + lookaheads with qtile > lo (N-2 for N >= 2) + the tail iff
      N >= 2 (1)]; s_read N / N; dp_empty N / N; the dS gates N each / N (N-1 in-tile + the tail); dq_done N (+1 zero-trip) / N (+1);
      q1 N / N; dq_empty[s] / dq_full[s] 2N / 2N per slot; p_ready, ds_ready, ds_tmem, s_done, dp_done, the do* / q2 rings N / N;
      dvk_empty, k2_empty, dv_full, dk_full, sched 1 / 1 per item.  Zero-trip item: 0 everywhere but dq_done / dvk_empty / k2_empty /
      dv_full / dk_full / sched 1 / 1.
    THE s_tail_empty WALK (s_tail map; the only row whose wait sites carry runtime guards; the barrier audit:
      N == 1: S^T(lo) waits [prev tail / pre-armed]; tile lo: no dQ, no lookahead; tail dQ(lo): hi - 1 > lo FALSE -> no wait (the
        write before it in [64, 128) is S^T(lo); the softmax's read of S^T(lo) is implied by the dS gates (lo) on both CTAs); the
        epilogue's dQ(lo) readout fires 1.  1 == 1.
      N == 2: S^T(lo) [1]; tile lo nothing; tile lo+1: dQ(lo) at slot 2, no S^T(lo+2); tail dQ(lo+1) waits dQ(lo)'s readout [1]; fires 2.
      N == 3: S^T(lo) [1]; tile lo nothing; tile lo+1: dQ(lo), then S^T(lo+2) waits dQ(lo)'s readout [1]; tile lo+2: dQ(lo+1), no
        S^T(lo+3); tail dQ(lo+2) waits dQ(lo+1)'s readout [1]; fires 3.
      Table 3's "before EVERY S^T" form would park tile lo's lookahead on the readout of a dQ not yet issued = HANG on item 1 (D1).
    DELAYED OBSERVER (ABA, the barrier audit: PASS for every pair-scope object; the LOCAL dS gates' and ds_pair_ready's back-edge
      rests on the epilogue's tile-end dq_done(t) wait (the round-A placement): a lever that moves the dQ readout out of the epilogue
      re-audits ds_pair_ready / ds_smem / ds_recv_full.  Under 16w_fa4 that back-edge is the SOFTMAX's dq_war wait (the the alternate role shape block above;
      a recorded note closed for this body by the barrier audit: the reduce group's readout wait is no longer load-bearing for the gates.
    MMA ISSUE ORDER (leader warp, per q tile t; FAv4's loop rotated onto our tile body; the typed order field accepts `fa4` ONLY on this map):
      slot | GEMM              | waits (state)                                                   | commits                | why here
      pro  | S^T(lo)           | k1_full, q1_full (q1), s_tail_empty (s_tail: the previous item's tail readout) | s_done, q1_empty | hidden behind K1 / V / K2
       1   | dP^T(t)           | dp_empty(t-1) (dpe: the dS^T(t-1) readout), do1_full x 2 (do1)  | dp_done, do1_empty     | ~+1300 clk slack: ds_tmem(t-1) + ld
       2   | dQ(t-1) if t > lo | s_read(t) (ready) [every tile]; ds_smem, ds_recv_full, ds_pair_ready (xchg) | dq_done | the exchange gets dV + S^T +
           |                   |                                                                  |         | dK + dP^T of cover (~1350 clk)
       3   | dV(t)             | p_ready(t) (ready), do2_full (do2)                               | do2_empty              | unchanged
       4   | S^T(t+1) if t+1<hi| q1_full (q1); s_tail_empty iff t > lo (s_tail: dQ(t-1)'s readout) | s_done, q1_empty | the TIGHTEST edge (~+300 clk:
           |                   |                                                                  |                  | dq_done(t-1) + ld vs dV(t))
       5   | dK(t)             | ds_ready(t) (ready), q2_full (q2)                                | q2_empty               | unchanged
      tail | dQ(hi-1)          | s_tail_empty iff hi - 1 > lo; the three dS gates (hi-1) (xchg)    | dq_done; then k1_empty, v_empty | once per item
      tmem_map dq_576: the s_tail_empty and s_read terms of the waits column vanish (slot 2 = the gated issue alone); nothing else in the order moves.
      Why this order: with dQ(t) issued before the dP^T(t+1) chain, the MMA leader's wait at the dS gates and the softmax's dP wait form ONE
      lock-step loop (dQ(t) -> readout -> dp_empty -> dP^T(t+1) -> softmax -> exchange -> dQ(t+1)); here dP^T(t+1) no longer sits behind dQ(t)'s
      exchange chain and dQ(t) is issued one GEMM later with nothing queued behind it.
    PER-ROLE PHASE TABLE (ONE PipelineState per consumption cadence, carried across CLC items, advanced on EVERY path -- P2 / P14):
      role     | state (start)                 | advance                        | feeds
      MMA(L)   | q1 st(0), 1 stage             | +1 per tile (q1_next = the lookahead's slot; the skipped last lookahead pre-positions the next
               |                               |   item's prologue, N per item == N S^T issues) | q1_full
               | q2 / do1 / do2 st(0)          | +1 per tile                    | q2_full, do1_full[0..1] (ONE do1_state), do2_full
               | kv st(0)                      | +1 per live block              | k1_full, v_full, k2_full
               | ready st(0)                   | +1 per tile                    | p_ready, ds_ready, s_read (all indexed by tile t)
               | xchg st(0) NEW (the P2 split) | +1 per dQ ISSUE (slot 2 of tiles lo+1 .. hi-1, the tail) | ds_smem, ds_recv_full, ds_pair_ready
               | s_tail st(1) NEW              | +1 per s_tail_empty wait (D1's three sites); s_tail map only (None under dq_576) | s_tail_empty
               | dpe st(1) / out_empty st(1)   | +1 per tile / +1 per item      | dp_empty / dvk_empty
      MMA(F)   | relay st(0)                   | +1 per tile                    | ds_smem, ds_recv_full (own copies) -> ds_pair_ready arrive_on_peer
      softmax  | tile st(0)                    | +1 per tile                    | s_done, dp_done (arrives: s_read, p_ready, ds_ready, ds_tmem)
      epilogue | ds st(0) / dq st(0)           | +1 per tile / +1 per tile, +1 per zero-trip item | ds_tmem / dq_done
               | empty st(1)                   | +1 per ROUND (both slots waited at one phase: the audit's 2-stage "+2 per round" collapsed to
               |                               |   its one flip per round)      | dq_empty[0], dq_empty[1]
      store    | dq st(0), 2 stages            | +1 per USE (4 per tile)        | dq_full[slot]
               | out st(0)                     | +1 per item                    | dv_full, dk_full
      loader   | q1 st(1), 1 stage; q2 / do1 / do2 st(1); kv st(1); k2e st(1) | +1 per tile / per live block / per item | q1_empty .. k2_empty
      scheduler(L) | sched st(0), 2 stages     | library loop                   | sched_empty, sched_full
    P14 paths: zero-trip item (no tile: no S^T, no dQ, the empty dq_done commit; no dq_empty traffic: 0 / 0); half-dead pair (CTA 1 runs
      every tile with P = 0 exactly: reads S^T (s_read), reads its 64 dQ rows out (s_tail_empty), ships zeros, relays -- identical flip
      counts); n_q == 1 (lo + 1 == hi: the lookahead is skipped, no in-tile dQ, the tail dQ(lo) without an s_tail_empty wait; the
      prologue's S^T(lo) is the item's only S^T); the first tile of every live item: dP^T(lo) [dp_empty: the previous item's last dS^T
      readout / pre-armed]; s_read(lo); NO dQ; dV(lo); S^T(lo+1) with q1_full only; dK(lo).

    smem_map rubin_b2 (the barrier skeleton R2.c 2026-11-03 sections 2.2-2.6 + 1.7; ``BAR_LAYOUT_RING_DQ576_B2`` = the dq_576 ring table with the ring rows
    at their slots + ONE new row, DERIVED by ``layout_for_geometry`` and asserted against every pasted table: 36 names / 51 slots / 592 B in
    [313344, 313936) (the 4-slot do1_full row takes 32 B: the cumulative walk), kinds {tma_leader 7, mma 9, thread 10, leader 5, peer 2, tma_dsmem 1,
    tma 2}; AUDIT of the committed body owed).  Every row not listed = the dq_576 lines above verbatim (count per slot, kind, waiter, cadence).
    name[st]         kind              init L / F     SUM(issuing lanes) == init        waiter (state)                  start  fires
    q1_full[2]       TMA_LOAD          1 / 1(-) /slot loader `q1_full[q1.idx].arrive(n_bytes=Q1_TX, pred=arm_pred)` = 1 (L only) per slot == 1
                                                      MMA(L) q1_state / q1_next = advance(q1, Q1_STAGES = 2)                      st(0)  N / N (slots alternate)
    q1_empty[2]      MMA_COMMIT        1 / 1 /slot    `_commit_pair(q1_empty[q1.idx])` after each S^T issue = 1 per CTA == 1
                                                      loader q1_state st(1) 2 stages: wait k <-> release k-2; DRAIN x Q1_STAGES        st(1)  N / N
    do1_full[4]      TMA_LOAD          1 / 1(-) /slot loader per tile: `do1_full[2 do1.idx + half].arrive(n_bytes=DO1_HALF_TX, pred=arm_pred)` x 2 halves
                                                      = 1 per slot == 1; tx = one [64 x 64] box per CTA x 2 = DO1_HALF_TX per half slot
                                                      MMA(L) do1_state 2 stages (slots 2 idx, 2 idx + 1 at one phase; the stage descriptor built per tile)
                                                                                                                                     st(0)  N / N per slot pair
    do1_empty[2]     MMA_COMMIT        1 / 1 /slot    `_commit_pair(do1_empty[do1.idx])` after dP^T(t) = 1 per CTA == 1
                                                      loader do1_state st(1) 2 stages (wait k <-> release k-2); DRAIN x DO1_STAGES     st(1)  N / N
    ds_smem[2]       THREAD            128 / 128 /slot epilogue bare 4 warps after the own STS into stage s + fence_proxy + tcgen05 fence = 128 per slot
                                                      MMA(L) `_dq_issue` slot xchg.idx (xchg_state 2 stages, +1 per dQ issue) / MMA(F) relay_state
                                                      (2 stages, +1 per tile)                                                        st(0)  N / N
    ds_recv_full[2]  TMA_LOAD (dsmem)  1 / 1 /slot    epilogue lead warp `ds_recv_full[s].arrive(n_bytes=XCHG_TX, pred=ship_pred)` = 1 per slot == 1;
                                                      tx = the PEER's ONE bulk copy of ITS XCHG stage s into OUR sDS stage s recv half (P9: a plain
                                                      cp.async.bulk completes on the DESTINATION's mbar, never routed; exchange_off arms 0 per slot).
                                                      BACK-EDGE of arm(t) on slot t % 2 (P14-B): this epilogue's dq_done(t-2) wait of iteration t-1 --
                                                      phase t-2 of the slot completed (the bytes landed) before dQ(t-2) issued (L: its own wait; F: via
                                                      the relay -> ds_pair_ready) -> dq_done(t-2) -> the wait -> arm(t): no arm on a tx-pending phase on
                                                      any path (in-loop, the tail: no arm, the item top: after the item-end dq_done(hi-1) wait, after a
                                                      zero-trip item: no arm, no phase)
                                                      MMA(L) xchg_state / MMA(F) relay_state, slot s                                 st(0)  N / N
    ds_pair_ready[2] THREAD (remote)   1 / 1(-) /slot MMA(F) `ds_pair_ready[relay.idx].arrive_on_peer(0, pred=elect_p, release_cta=True)` after its own
                                                      ds_smem[s] + ds_recv_full[s] waits (P12) = 1 -> L per slot == 1   MMA(L) `_dq_issue` slot xchg.idx
                                                                                                                                     st(0)  N / N
    dq_read_done[1] NEW LEADER (relaxed) 256 / 256(-) epilogue 4 warps bare x 2 CTAs right after EACH dQ readout's `_tmem_load(dq_col, 64)` +
                                                      tcgen05_wait(LOAD) + tcgen05_fence(BEFORE_THREAD_SYNC): the in-loop site (dQ(t-1), iteration t,
                                                      under the item's `qtile > lo`) and the item-end site (dQ(hi-1), under `lo < hi`) -- both the
                                                      loop-structure predicate of _item_bounds (pair-uniform: the many-lane-arrive guard exception,
                                                      named here) = 128 x 2 = 256 == PAIR_EPILOGUE_LANES.
                                                      waiter MMA(L) INSIDE `_dq_issue` after the three dS gates, before EVERY dQ MMA (in-tile and
                                                      tail) on `dq_read_state` st(1) PRE-ARMED, +1 per wait: wait w <-> the readout of issue w-1 (the
                                                      kernel's first issue passes on the pre-arm); lap bound: flip w needs the readout of issue w,
                                                      which needs dq_done(w), which needs issue w, which follows wait w -> F in {w-1, w} at every wait
                                                      (P5 sound); DRAIN x1 after the loop (P15: the last live item's readout lands on a resident
                                                      leader; a leader without a live item passes on the pre-arm).  Zero-trip: 0 / 0 (no issue, no
                                                      readout: the empty dq_done commit touches no ring).  WHY: under the LAGGING order ds_smem(t) is
                                                      arrived BEFORE the readout of dQ(t-1), so the dq_576 program-order carrier (the s_tail_empty
                                                      lines above) does not exist; this row IS the dQ image's REUSE edge        st(1)  N / N (+1 drain)
    ABA / lap of the 2-slot rings (the skeleton 2.6): on every ring, flip c+2 of slot c % 2 passes through dq_done(c) -- ds_smem: the epilogue's
    STS(c+2) follows its dq_done(c) wait (iteration c+1); ds_recv_full: arm(c+2) + the peer's bytes(c+2) follow the two epilogues' dq_done(c) waits;
    ds_pair_ready: relay(c+2) <- F's STS(c+2) <- F's dq_done(c) wait -- and dq_done(c) needs the leader's observation of flip c: depth 2 is exactly
    sufficient (a 1-slot ds_pair_ready under the ring = two relays before one wait = the deterministic hang).  q1 / do1 rings: completion c+2 <- the
    loader's *_empty(c) wait <- the commit after the issue of c <- the leader's *_full(c) wait.
    STATES (P2: one PipelineState per ring AND stage count -- a 1-stage waiter cannot ride a 2-stage state): epilogue `ds_state` st(0) 1 stage
    (ds_tmem, KEPT), `sds_state` st(0) 2 stages (+1 per LIVE tile: the sDS / XCHG stage, the ds_smem / ds_recv_full slot, the ship's target stage;
    NEW), `dq_state` st(0) (+1 per dq_done wait: N-1 in-loop + 1 at the item end, +1 zero-trip = today's count), `empty_state` as today; MMA(L)
    `xchg_state` st(0) 2 stages (+1 per dQ issue), `do1_state` 2 stages (slot 2 idx + half), `q1_state` 2 stages, `dq_read_state` st(1) 1 stage
    (+1 per wait, NEW); MMA(F) `relay_state` 2 stages; loader `q1_state` st(1) 2 stages, `do1_state` st(1) 2 stages.  The three dS-ring states
    advance once per LIVE tile on both CTAs, never on a zero-trip item, never reset at the item top (tile lo of an item takes stage K % 2 for the
    CTA's running count K: a reset at an odd K would put the STS, the arm, the ship target and the dQ descriptor on different stages).
    PER-ITEM FLIPS: ds_smem[s] / ds_recv_full[s] / ds_pair_ready[s] N / N over the two slots in tile order; q1 / do1 rings N / N over the slots;
    dq_read_done N / N (+ the kernel's one drain); zero-trip: 0 / 0 on every ring above, dq_done 1 / 1, the item rows 1 / 1 (unchanged); N == 1: one
    flip on slot K % 2 of each ring, the tail dQ(lo) consumes it; half-dead pair: identical counts (CTA 1 stages and ships zeros, reads out, arrives).

6. INIT + PHASE DISCIPLINE + EXIT DRAINS  (P4 / P5 / P6 / P14 / P15; code: the kernel entry + the helpers above the role bodies)
    init: warp 0, `if elect_sync():` every stage of every Bars field (ONE lane, both gates; under stats_form tma_ring the bundle is BarsRing =
          the 35 fields + stats_full[2] / stats_empty[2], the same loop: +4 mbarrier.init; under tmem_map dq_576 BarsDQ576 / BarsRingDQ576 = 33 / 35
          fields, s_read and s_tail_empty absent: -2 mbarrier.init) -> fence_mbarrier_init -> barrier_cta_sync
          -> cga_arrive / cga_wait (before ANY cross-CTA arrive) -> MMA warp: tmem_alloc(geo.TMEM_COLS, cta_group::2, is_exclusive=geo.TMEM_IS_EXCLUSIVE)
          on BOTH CTAs (512 | 576 exclusive under dq_576; the SAME geo.TMEM_COLS at the dealloc: G03 lifecycle rule 1)
          -> barrier_cta_sync (publishes the TMEM base) -> role dispatch.  NO bootstrap arrive anywhere (D3: the store warp's dq_empty
          bootstrap of the first body left with the staging alias); every MMA-fed or pair-fed consumer starts pre-armed where the table says
          st(1): the leader's dp_empty / s_tail_empty / dvk_empty states, the epilogue's dq_empty state, the loader's *_empty states.
          16w_fa4: the SAME block over the 33 fields of BAR_LAYOUT_XFA4 (`_bars(slab, fa4_roles)` builds `BarsXFA4`: ds_tmem / dp_empty absent,
          an AttributeError if touched, never a silent dead row; dq_war_done present); pre-armed st(1) ALSO the softmax's NEW `dq_war`; the leader's
          dp_empty state is gone; dispatch warp < 8 softmax | < 12 REDUCE (`_reduce_loop`) | 12 MMA | 13 | 14 | 15; P15 drains MMA(L): dvk_empty x1,
          s_tail_empty x1 (dp_empty's gone); SOFTMAX: the ONE unconditional `dq_war_done` drain after the loop (section 15.3: the leader's last tail
          commit is issued before it leaves its loop and depends on nothing of the softmax: the drain cannot hang; a CTA with no live item passes on
          the pre-arm).  Everything else of this section (seams, zero-trip, drains) holds unchanged; the ds_own / XCHG seams now end at the
          SOFTMAX's dS STS, which waits dq_war_done(t-1) directly (the WAR) and dp_done(t) (dvk_empty's dK-staging seam).
    phase trackers: section 5's PER-ROLE PHASE TABLE (never bit math -- P2 / P14); stats_form tma_ring adds softmax `stats_state` st(0) and
      loader `stats_ld` st(1), +1 per tile each, CFG_RING.STATS_STAGES = 2 stages, carried across items (section 5's ring rows); nothing on a
      zero-trip item; no drain.
    zero-trip item (lo >= hi, PAIR-UNIFORM from _item_bounds; a padded CTA inside a live pair runs every tile with P = 0): MMA(L) one
      empty dq_done commit (live pipe, multicast); both epilogues wait it (dq_state +1); the store warps run the native zero stores,
      arrive dvk_empty (-> L) + k2_empty; the loaders wait k2_empty + arrive dv_full; softmax / MMA(F) only credit the scheduler.
    P15 drains (OUTSIDE the persistent loop, const_expr(CFG.CTA_MMA == 2), whole ring from the carried state):
      loader L+F: q1_empty x Q1_STAGES (= 1), q2_empty, do1_empty, do2_empty, k1_empty, v_empty (the leader's last multicast commits in flight)
      MMA(L): dp_empty x1, dvk_empty x1, s_tail_empty x1 (the follower's last relaxed arrives in flight: the last item's dS^T readout,
      the last dV / dK drain, the last tail readout; tmem_map dq_576: dp_empty x1, dvk_empty x1 only); tmem_dealloc: the wait is the drain.
      Observed in-loop on every phase, no drain: s_done / dp_done / dq_done / sched_full, p_ready / ds_ready / s_read (every tile),
      ds_pair_ready / ds_recv_full / ds_smem (every dQ issue, in-tile and tail), dq_empty / dq_full (every round), sched_empty.
    smem_map rubin_b2: init over BarsRingDQ576B2 (36 fields, 51 slots: +9 mbarrier.init); P15 drains loader L+F: q1_empty x Q1_STAGES = 2, q2_empty,
      do1_empty x DO1_STAGES = 2, do2_empty, k1_empty, v_empty; MMA(L): dp_empty x1, dvk_empty x1, dq_read_done x1 (NEW: the last readout's flip);
      observed in-loop, no drain: ds_smem[s] / ds_recv_full[s] / ds_pair_ready[s] per slot (every dQ issue), the rest as above.
    SMEM LIFETIME SEAMS (Rule S3: an explicit barrier chain per reuse, never an issue-order invariant; the barrier audit:
      sDQ slot s (DEDICATED): written by the epilogue warp pair s per round after dq_empty[0] + dq_empty[1] (the round's phase), read by the
        store's reduce-add of slot s, released by dq_empty[s] after `tma_store_wait(1)` (slot 0) / `(0)` (slot 1); two cycles per tile; no
        other tenant: the dS / XCHG STS never waits dq_empty.
      ds_own(rank) (the dS^T(t+1) STS) <- the dQ(t) MMA's A read (the previous tenant) -> dq_done(t), waited by the SAME epilogue at the end
        of its tile t (the dQ(t) readout) before it begins tile t+1: program order after a barrier wait.  The gate is the MMA's COMPLETION,
        never the reduce-add drain ("the exchange is not gated").
      XCHG (the dS^T(t+1) peer-half STS) <- our ship(t)'s cp.async.bulk source read: completes before the PEER's ds_recv_full(t) -> (L's XCHG)
        the follower relay -> ds_pair_ready(t) -> dQ(t) -> dq_done(t) / (F's XCHG) the leader's own ds_recv_full(t) wait -> dQ(t) -> dq_done(t)
        -> our epilogue's dq_done(t) wait at its tile-t end.  The first body's dq_empty leg is gone.
      ds_recv(rank) <- ONLY the peer's ship(t+1), which follows the peer epilogue's dq_done(t) wait; dQ(t) read the recv half before dq_done(t).
      dK native staging over ds_own(rank) + XCHG (item end) <- dS(hi-1) own half: the dQ(hi-1) MMA -> dq_done(hi-1) waited by the epilogue (its
        last tile's readout); XCHG <- the ship(hi-1) read <- dq_done(hi-1) via the chains above.  Next tenant: the dS(lo') STS of item k+1 <-
        ds_tmem(lo') <- dp_done(lo') <- dP^T(lo') issued after the MMA's item-top dvk_empty(k) wait <- both store warps' tma_store_wait(0)
        (the dK store's source read).  The first body's second gate (the dq_empty bootstrap before the dS STS) is gone: the dvk_empty chain
        carries the seam alone (ALIASES row dk_store).
      dV native staging over sK2 (item end) <- the dQ(hi-1) MMA's B read -> dq_done(hi-1) waited by the epilogue; the previous item's dV store
        drain -> dvk_empty -> MMA -> dq_done (live) / the empty commit (zero-trip); the K2 reload <- k2_empty (store after tma_store_wait(0)),
        fix A back-edge intact.
      sQ1 (1 stage) <- the S^T(t) MMA's B read -> q1_empty commit after S^T(t) -> q1_full(t+1) before S^T(t+1); window ~2000 clk.
      sQ2 / sDO1 / sDO2 / sK1 / sV: unchanged rings, roots shifted -16384.
      smem_map rubin_b2 (the skeleton 2.7; stage s = tile t % 2): ds_own stage s <- the dS(t+2) STS: WAR vs the dQ(t) A read -> dq_done(t), waited by
        this epilogue in iteration t+1 (the LAGGING order: before the STS of t+2); XCHG stage s <- the dS(t+2) peer-half STS: WAR vs ship(t)'s source
        read, NO local signal -- the PEER's ds_recv_full[s](t) -> dQ(t) (L direct / F via the relay) -> dq_done(t) -> our wait; recv half of stage s <-
        ONLY the peer's ship(t+2), which follows the peer epilogue's dq_done(t) wait (dQ(t) read it before); dK native staging over own(rank) of STAGE 0
        + XCHG stage 0 at the item end <- dq_done(hi-1) (in-order commits cover the other stage's dQ(hi-2); both ships' reads via the chains above),
        next tenant dS(lo') of item k+1 into EITHER stage <- ds_tmem(lo') <- dp_done(lo') <- dP^T(lo') after the MMA's item-top dvk_empty(k) wait <-
        both store warps' tma_store_wait(0); the dkv_fp32 slots = the stage-0 bytes (the stage-0 recv half is dead from dq_done(hi-1)); sQ1 / sDO1 stage
        s <- the loader's *_empty[s] wait (release k-2); the sDQ ring unchanged.
    THE ~150-clk CLAIM (resource-manager; why the staging alias is deleted): at the 512-clk-per-GEMM ideal, dq_done(t) lands at ~T + 3584 and
      ds_tmem(t+1) at ~T + 3300..3500, so the epilogue's dS(t+1) STS starts after its dq_done(t) wait + readout (~T + 3800), 300..500 clk after
      the exchange could start; the sDQ round-B drain (A(t)'s reads, ~800..1050 clk after round A) completes by ~T + 4850, before the
      dq_done(t+1) wait at ~T + 6144: the reduce-add drain gates nothing on the ideal timeline, ~1200 clk of margin at the measured duty.

7. THE ONE HELPER
    _item_bounds(pair, meta, batch, n_batch, causal) -> (q_lo, q_hi, seq_q, seq_k, q_start): CLUSTER-UNION bounds
    (the pair walks the identical q range; causal trim per 256-row PAIR: lo = pair * Q_TILES_PER_PAIR; CTA 1's diagonal tile is
    fully MASKED, never skipped; a dead pair (256 pair >= seq_k) is lo = hi = 0 on both CTAs).  Every role on both CTAs derives
    its loop bounds from this pure helper over the SAME inputs -- the pair-uniform `qtile > lo` / `hi - 1 > lo` guards of the
    s_tail_empty sites and the dQ issue sites read the same (lo, hi) on the leader as the softmax / epilogue do on both CTAs.
    tile_may_mask(pair, rank, qtile, seq_q, seq_k, causal) -> Boolean (the mask-form lever, section 11; plain Python over the traced scalars = the
    SAME inputs _item_bounds reads + the CTA rank; the test module runs the identical function on host ints against the per-cell mask):
    TRUE iff a cell of THIS CTA's [128 kv x 128 q] tile can be masked = ((qtile + 1) * N > seq_q) | (kv_end > seq_k) | (causal & (qtile * N <
    kv_end)) with kv_end = pair * PAIR_KV_ROWS + (rank + 1) * M (the CTA's exclusive last kv row + 1).  CTA-uniform by construction (no lane
    term), so the `if mask_tile:` branch of _p_values never diverges.  EXACT, not merely conservative: _p_values keeps column c of the lane with
    key k iff c >= k (causal) and c < (seq_q if k < seq_k else 0), monotone in c on both sides, so a masked cell exists iff the tile's FIRST
    column fails the causal edge for the CTA's largest key (qtile * N < kv_end - 1, i.e. qtile * N < kv_end as both are multiples of N = M) or
    its LAST column fails the hi edge for some lane ((qtile + 1) * N - 1 >= seq_q, or some key >= seq_k, i.e. kv_end > seq_k).  Under causal the
    pair's live tiles start at lo = 2 pair, so CTA 0 runs the mask body on tile 2 pair only and CTA 1 on 2 pair (fully masked) and 2 pair + 1
    (its diagonal): ~2 of 64.5 tiles per item at 16K; every other tile of a full-length sequence is interior.

8. SMEM TABLE  (REQUIRED; per CTA, DECLARATION order; the SMEM audit PASS: total 231424 B of the part's 232448-B opt-in cap, slack 1024 B
   (scaffold); bank check PASS 0 / 7; DESC_VERSION implied 0, the baseline's 1 inherited.  stats_form tma_ring = the CFG_RING geometry (the
   resource plan resources/s1-stats-ring.md section 1, `vs-smem predict` PASS: total 232448 == the cap, slack 0): the sSTATS + sDELTA bytes become
   ring stage 0, stage 1 takes the 1 KiB of cap slack at 230400, and the scaffold follows by derivation (+1024: BARS = SLABS_END 231424, the bar
   block [231424, 232016) 592 B + the 352-B STAMPS record [232016, 232368) + TMEM handle 16 @ 232368 + CLC payload 64 @ 232384); no other slab,
   alias, root or descriptor moves -- the ldg table below is unchanged.  RESOURCE-NOTE: pipe/tile: async +8 (stats TMA), LSU -8 (STS.32 gone),
   LDS.128 512 unchanged: ~3620)
    name[stages]   | bytes  | 1024-al | desc | swizzle + WHY                                  | lifetime / alias   | offset
    sQ1[1]         | 16384  | 16384   | yes  | SW128 K-major: MATCH the S^T B descriptor       | tile; q1 ring (ONE stage, FAv4 Q_stage 1) | 0
    sQ2[1]         | 16384  | 16384   | yes  | SW128 MN-major (64-d rows = one atom): MATCH dK B | tile; freed by dK | 16384
    sDO1[1]        | 16384  | 16384   | yes  | SW128 K-major: MATCH the dP^T B descriptor      | tile; freed by dP^T| 32768
    sDO2[1]        | 16384  | 16384   | yes  | SW128 MN-major: MATCH the dV B descriptor       | tile; freed by dV  | 49152
    sK1[1]         | 32768  | 32768   | yes  | SW128 K-major: MATCH the S^T A descriptor       | kv block           | 65536
    sK2[1]         | 32768  | 32768   | yes  | SW128 MN-major, 256 rows (two 128-row boxes):   | kv block; dV store | 98304
                   |        |         |      | MATCH the dQ B desc                             | staging at the end |
    sV[1]          | 32768  | 32768   | yes  | SW128 K-major: MATCH the dP^T A descriptor      | kv block           | 131072
    sDS[1]         | 32768  | 32768   | yes  | SW128 MN-major (q contiguous, 128 B per kv row, | own half: STS ..   | 163840
                   |        |         |      | kv-global row order): MATCH the dQ A desc        | dQ MMA; recv half: |
                   |        |         |      | (atranspose); the 16-B `_store_words` form        | the PEER's timeline|
    sDS_xchg[1]    | 16384  | 16384   | no   | SW128 image of the peer's recv half (byte copy) | STS .. bulk copy   | 196608
    sDQ[2] NEW     |  8192  | 16384   | no   | fp32 [64 q x 32 d] SW128 (lane stride 128 B:     | DEDICATED ring: epilogue | 212992
                   | x 2    |         |      | SPREAD + MATCH the reduce-add box); slot s = d   | STS per round -> store   |
                   |        |         |      | half s's warp pair                               | reduce-add -> dq_empty   |
    sSTATS (LSE)   | 512    | 512     | no   | none: 4-B lane stride                            | tile               | 229376
    sDELTA         | 512    | 512     | no   | none                                             | tile               | 229888
    sSTATS_RING[2] | 1024   | 1024    | no   | none: stage s = STATS_RING + s * 1024, the LSE row | tma_ring ONLY: stage s live from the loader's
      (tma_ring)   | x 2    |         |      | at +0, the delta row at +512 (= today's slab    | bulk copy for tile t (~slot 5 of t-2) to the
                   |        |         |      | image: the LDS.128 immediates unchanged); the    | softmax's last delta LDS of t (stats_empty);
                   |        |         |      | warp-uniform reads are broadcasts (no bank job) | next tenant t+2; aliases NOTHING | 229376 (st 0 = sSTATS+sDELTA), 230400 (st 1 = the former cap slack)
    scaffold       | 1024   |         |      | bars [230400, 230960) 560 B + 384 spare [230960, 231344) = the STAMPS record (config DBG_CLK: 96 x Int32,
                   |        |         |      | written only under `debug_clk`) + TMEM handle 16 @ 231344 + CLC payload 64 @ 231360 | 230400
                   |        |         |      | (tma_ring: bars [231424, 232016) 592 B + 352 spare = the record DBG_CLK_RING 88 x Int32 + TMEM handle @ 232368
                   |        |         |      | + CLC payload @ 232384 -> SMEM_BYTES 232448 == the cap) | 231424 (tma_ring)
                   |        |         |      | (tmem_map dq_576: two rows fewer -- bars [230400, 230928) 528 B + 416 spare = DBG_CLK_DQ576 104 x Int32 (ldg);
                   |        |         |      | bars [231424, 231984) 560 B + 384 spare = DBG_CLK_RING_DQ576 96 x Int32 (tma_ring); the slabs, the TMEM
                   |        |         |      | handle and the CLC payload do not move: SMEM is not the map's business) |
    smem_map rubin_b2 (CFG_RING_DQ576_B2 = the dq_576 ring record with Q1_STAGES / DO1_STAGES / DS_STAGES / XCHG_STAGES = 2; the resource plan's B2 map,
    the skeleton 2.1: `vs-smem predict --arch 10.7 --scaffold 1024 --desc-version 1` PASS, total 314368 of the 334848-B oversized cap, slack 20480; the
    same table FAILS at --arch 10.0 = part-keyed by construction; every offset below is the config's, asserted at import):
    name[stages]   | bytes  | offset (stage roots)      | desc | why / lifetime
    sQ1[2]         | 32768  | 0 (0, 16384)              | yes  | SW128 K-major: MATCH the S^T B desc; q1_full[2] / q1_empty[2], stage = tile count mod 2
    sQ2[1]         | 16384  | 32768                     | yes  | unchanged form
    sDO1[2]        | 32768  | 49152 (49152, 65536)      | yes  | two [64 x 64] half boxes per stage; do1_full[4] / do1_empty[2]; the stage desc built per tile
    sDO2[1]        | 16384  | 81920                     | yes  | unchanged form
    sK1[1]         | 32768  | 98304                     | yes  | unchanged form
    sK2[1]         | 32768  | 131072 (+131072, 147456)  | yes  | unchanged form; the dV staging alias at the item end
    sV[1]          | 32768  | 163840                    | yes  | unchanged form
    sDS[2]         | 65536  | 196608 (196608, 229376)   | yes  | stage s = tile t % 2 on all three roles: own half (+rank x 16384) by the epilogue's 16-B STS,
                   |        |                           |      | recv half ONLY by the PEER's ship of ITS tile on that stage; WAR on the stage = dq_done(t-2);
                   |        |                           |      | the dQ A desc = the pinned stage-0 root + (s x 32768) >> 4 per issue; the LAST tcgen05 root
                   |        |                           |      | ends exactly at 262144 (margin 0: allowed); dK staging half 0 = own(rank) of STAGE 0
    sDS_xchg[2]    | 32768  | 262144 (262144, 278528)   | no   | the ship's source per stage (past the version-0 line: a bulk-copy source, never a root); dK
                   |        |                           |      | staging half 1 = stage 0
    sDQ[2]         | 16384  | 294912 (294912, 303104)   | no   | the dedicated ring, unchanged form
    sSTATS_RING[2] | 2048   | 311296 (311296, 312320)   | no   | unchanged form
    scaffold       | 1024   | 313344                    |      | bars [313344, 313936) 592 B (36 names / 51 slots; the 32-B do1_full row) + the 352-B STAMPS
                   |        |                           |      | record DBG_CLK_RING_DQ576_B2 (88 words, 86 named: + wait|blocked:leader:dq_read_done) + TMEM
                   |        |                           |      | handle 16 @ 314288 + CLC payload 64 @ 314304 -> SMEM_BYTES 314368
    DESC_VERSION 1 inherited (every root < 262144: v0 would suffice); per-tile bytes unchanged (only waits move); the alias tenants (dK staging, dkv_fp32
    slots) sit on the STAGE-0 bytes (`dk_store` / `dkv_acc_r1_slot` unchanged by value); the DSL launcher declares the slab as dynamic SMEM (`.extern
    .shared`) sized from `geo.SMEM_BYTES` and sets the oversized mode above the opt-in cap (section 14): the first launch on cc 10.7 is the proof.
    aliases (3; config ALIASES): dK native store (2 x 16 KiB) over ds_own(rank) + XCHG at the item end (seam: dq_done(t_last) + dvk_empty);
    dV native store (2 x 16 KiB) over sK2 (seam: dq_done(t_last) + k2_empty); ds_recv(rank) = the peer's half (the peer's timeline).
    dkv_accum fp32_reduce: round 1 = dV's four fp32 [128 kv x 32 d] slices + dK's slice 0 over
    CFG.dkv_acc_r1_slot(0..4) = sK2's two halves, the dS slab's two halves (own + recv: dead from dq_done(hi-1) -- the recv bytes landed before
    it -- until the next item's dS writes, which the dvk_empty chain gates), XCHG (80 KiB); round 2 = dK's slices 1..3 over
    CFG.dkv_acc_r2_slot(0..2) = sK2's halves + the dS slab's first half (48 KiB) after dvk_stage_empty (the one new seam: a WAR edge inside
    the item end); the entry seams (dq_done(t_last), the ship(hi-1) read) and the exit seams (dvk_empty, k2_empty after the final drain) are
    unchanged; the sDQ ring is never a staging slot.
    No dQ staging alias on ds_own + XCHG: under the rotated order such an alias is issue-order sensitive (dq_done(t) lands ~150 clk before
    the dS(t+1) STS) and a drain-gated dS write would re-create the lock-step; the dedicated ring is paid for by sQ1 2 -> 1 stage (-16384 =
    +sDQ 16384); every root from sQ2 on shifts -16384; the total is unchanged.
    proxy-fence sites: after every lane-written slab the async proxy reads (sDS own half, XCHG before the bulk copy, each sDQ slot before the
    reduce-add, dV / dK staging before the TMA store) -- the audit's rows.
    16w_fa4: the SMEM table is UNCHANGED (every slab, root, alias; the softmax's STS image = the same bytes at the same offsets, 4 wavefronts
    per STS.128 = the floor); only the bar block's used span shrinks to [230400, 230928).

9. TMEM MAP (fp32 columns; one tcgen05.alloc.cta_group::2 of geo.TMEM_COLS on BOTH CTAs, the same constant deallocates; PART-KEYED by the typed field
   tmem_map: s_tail = 512 columns, every part's default; dq_576 = the 576-column carve allocated EXCLUSIVE, selectable by name on cc 10.7)
    S^T    @   0 + 128  | P^T packed bf16 in place @ 0 + 64 (softmax group g packs into 32 g .. +32); s_tail: dQ (M 128 half-row image) @ 64 + 64 =
                          S^T's dead tail (CFG.DQ_COL = S_COL + PACK_COLS); two alias edges: WRITE dQ(t-1) <- the softmax's S^T(t) read (s_read),
                          REUSE S^T(t+1) <- the epilogue's dQ(t-1) readout (s_tail_empty).  dq_576: [64, 128) is S^T's alone (no tenant, no edge, no row)
    dP^T   @ 128 + 128  | dS^T packed @ 128 + 64; [192, 256) DEAD under BOTH maps (was the first body's dQ image: a dQ slot there re-creates the
                          dp_empty lock-step the rotated order removed); dp_empty covers the dS^T readout only
    dV     @ 256 + 128  | dK @ 384 + 128   -> s_tail: used 512 / 512.  No TMEM slot is assumed zero.
    dQ     @ 512 + 64   | dq_576 ONLY (CFG_DQ576.DQ_COL = DK_COL + DK_COLS; the SAME half-row lane map: row m -> lane m % 64 + 64 (d // 64), DQ_LANE_SPLIT
                          64) -> used 576 / 576 (TMEM_USED == TMEM_COLS; alloc == dealloc == geo.TMEM_COLS = TMEM_COLS_SM107).  Writer: the dQ GEMM
                          (slot 2 of tile t+1 / the tail; `dq_acc` = tmem_base + geo.DQ_COL); reader: the epilogue's `_tmem_load(dq_col, 64)` after
                          dq_done(t); freed by: the readout's tcgen05_wait(LOAD) + tcgen05_fence(BEFORE_THREAD_SYNC) -- NO dedicated row; next writer:
                          dQ(t+1) behind ds_smem(t+1) / ds_pair_ready(t+1) (section 5: the REUSE edge rides the epilogue's program order; load-bearing).
                          The tcgen05 address word (lane << 16 | column) holds column 575: the LDTM / STTM helpers need no form change.
                          smem_map rubin_b2: the program-order carrier above is VOID (the lagging order arrives ds_smem(t) before the readout of
                          dQ(t-1)); the REUSE edge is the row dq_read_done (section 5): freed by the readout trio + `_arrive_leader(dq_read_done)`,
                          the next writer dQ(t+1) behind the leader's dq_read_done wait inside `_dq_issue` (pre-armed; one drain).
    s_tail: TMEM 512 / 512 unchanged; [192, 256) DEAD; one alloc / dealloc constant.  dq_576: the s_tail tenants at their columns + the image; SMEM unchanged.
    16w_fa4: the TMEM map is UNCHANGED; dS^T packed @ 128 + 64 KEEPS its STTM (dK's TS A operand); the epilogue's 4 x LDTM.x16 of it are GONE
    (the softmax stores the same words to SMEM); dp_empty's job (the readout) is gone with them (header section 5, the the alternate role shape block).
    Softmax ownership: group g (128 lanes = kv rows) owns fp32 q quarters [32 g, +32) and [64 + 32 g, +32), so both groups pack into their
    OWN columns (no cross-group barrier) and the packed q order is the fixed permutation CFG.ts_k_perm = [0, 1, 4, 5, 2, 3, 6, 7] (16-q
    blocks) that the dV / dK B descriptors walk via k_idx.  Quarter b of both groups ([64, 128)) is the dQ image's home on the s_tail map: without
    s_read the follower's P^T(t) q columns 64..127 would be exp2 of dQ values (g03 Rule 6's "second half" signature; RED 1 of the test module);
    under dq_576 quarter b is S^T's alone.

10. TENSOR-MAP / STRIDE CONTRACT (CFG.MAPS_PER_SEQUENCE = 9 per-sequence runtime maps, built by setup_thd_d128_2cta_kernel; unchanged)
    Q1 / dO1: 64-row maps, two [64 rows x 64 d] SW128 boxes per CTA stage (rows = the CTA's q half); Q2 / dO2: 128-row maps, one
    [128 q x 64 d] box at d = 64 * rank; K1 / V: 128-row maps, two [128 kv x 64 d] boxes (own rows); K2: two [128 kv x 64 d] boxes at
    d = 64 * rank over the PAIR's rows; dQ reduce-add: a 64-row fp32 map, [64 q x 32 d] boxes (4 per tile per CTA in two rounds; one
    reduce per PAIR tile); dV / dK: two [128 kv x 64 d] native boxes per CTA (128-row maps).  expect_tx = FULL box bytes ON THE LEADER
    for both CTAs' halves (2 x); TMA OOB zero-fills padded rows; sequence-local maps clip the tails.  ONE tensor-map acquire per
    consuming warp per work item (`acquire=False` on every per-tile load / reduce).  LOAD FORM: every cga2 load carries the one-CTA
    multicast mask (`.multicast::cluster.cta_group::2`), the validated form; the bytes complete on the pair LEADER's mbarrier.
    stats_form tma_ring: NO tensor map for the stats (a per-head row of a token-major [T, H] Stats port is not TMA-able: 4-B inner box; the padded
    per-sequence row base is not 16-B aligned for `emit_seq_descs`) -- the chain's delta launch (``delta_zero_stats_thd_d128_2cta_kernel``) writes
    the pre-scaled stat rows into the q-tile-PADDED fp32 region ``stats2`` [heads][PAD_TOTAL / N][2 kinds][N] (kind 0 = -log2(e) * LSE with the
    -inf -> 0 guard, kind 1 = -scale * delta; pad rows [S_b, ceil(S_b / N) * N) = 0.0; PAD_TOTAL = rounded_q + N * B; the contract
    d128_grouped_2cta_contract, ABI 2) and the setup kernel writes the padded row offsets ``rowoff[b] = sum_{i<b} ceil(S_i / N) * N`` into an Int32
    [B + 1] region (``tile_dsl.thd.write_thd_row_offsets`` with ``dst``: the 4B+4 metadata tensor keeps its shape); the loader's ONE plain 1 KiB
    `cp.async.bulk` per q tile reads record `stats2 + ((head * PAD_TOTAL + rowoff[b]) / N + t) * 1024` (1024-B aligned; 16 B needed): a tile never
    crosses its sequence's padded block, so there is no extent guard to rely on -- RESOURCE-NOTE: src = stats2 + ((head*PAD_TOTAL + rowoff[b])/128
    + t)*1024: 1024-B aligned (16 needed).  The nine maps and their acquires are unchanged under both forms.

11. OPTIONAL ARMS (const_expr; each states what the default rendering does)
    THD (the unit is 256 kv rows per pair) | GQA grouped loaders (head // ratio) | causal |
    stats_form (``stats_form_code`` Constexpr[int] from config_sm100_d128_accum_2cta_stail.STATS_FORM_CODES; lever S1 = the user's "LSE
    optimization", day 4): "ldg" = the previous default, selectable (the form "16w_fa4" resolves to when the field is unset; main-kernel md5
    7de23bd7 @ 4/32/4, ca665f06 @ 2/8/2 = the merged rendering's main body, byte-identical; per-kernel records): the softmax's
    per-tile `_stats` -- one LDG per lane (group 0 the LSE through the Stats port's token origin, group 1 the delta), the FMUL scale with the -inf
    -> 0 guard and the `row < seq_q -> 0` zero, one STS into sSTATS / sDELTA, inside a `bar.sync SOFTMAX_BARRIER, 256` pair (the item prologue for
    tile lo, the per-tile lookahead for t+1: 1257 clk per tile = 34 % of the pacing role at 16K, profiler-ncu 13:32) | "tma_ring" (FAv4's form:
    the LSElog2 preprocess + one bulk copy per tile; THE default since the day-4 measurement: +1.8 / +1.9 % at the step at 16K / 32K, +1.4..+1.9 %
    at every served cell, +3.3 % on the graph replay against ldg, bitwise; main-kernel md5 306ad223 @ 4/32/4, d9b419aa @ 2/8/2): the chain's delta launch
    (``delta_zero_stats_thd_d128_2cta_kernel``, grid x = ceil(max_sq / N) * N / ROWS_PER_CTA so every pad row is written) performs the SAME two fp32
    multiplies once per q row (-log2(e) * LSE with the -inf -> 0 guard, -scale * delta; `-x * y` lowers to the same negate + mul.rn.f32, the rowsum
    operand is the one the ldg form stores raw and reads back) into the padded region ``stats2`` (section 10; pad rows 0.0 = the ldg form's `stat =
    0`), the LOADER bulk-copies the (head, q tile) record one tile ahead into stage s of the two-stage SMEM ring behind stats_full[s] /
    stats_empty[s] (section 5), the softmax waits the stage after its s_read arrive and reads the SAME LDS.128 words (`_p_values` / `_ds_values`
    with the stage base), releases the stage right after its last delta read; the `_stats` chain, its two STS and both `bar.sync 1,256` are not
    traced -- RESOURCE-NOTE: named barrier 1 retired under tma_ring (its only sites were _stats'); id 2 unchanged.  BITWISE vs ldg on P / dS
    and therefore dQ (per reduce-add order) / dK / dV (the test module's twins over both Stats layouts, S % 128 != 0, len % 128 == 1, empty and
    zero-trip cells).  Geometry CFG_RING (SMEM_BYTES 232448 == the cap; the first tma_ring smoke IS the launch proof at the cap; a launch failure
    = the resource plan's fallback sDQ 2 -> 1 slot).  Census vs ldg @ 4/32/4 (host render): `bar.sync 1,256` 4 -> 0, `st.shared.b32` 4 -> 0, the
    plain `cp.async.bulk` global -> shared 0 -> 1, `mbarrier.init` 39 -> 43, arrive 40 -> 42, try_wait 68 -> 70, expect_tx 11 -> 12, `bar.warp.sync`
    67 -> 69, `ld.global` 24 -> 19 in the MAIN kernel (-6 softmax, +1 loader), `mul.f32` 37 -> 33 / `neg.f32` 1 -> 0 / `setp.eq.f32` 2 -> 0 (the
    deleted `_stats` arithmetic, present once in the delta launch), `fence.proxy.async` 15 / `tcgen05.fence::after_thread_sync` 32 / LDS.128 38 /
    mapa 17 unchanged; SASS (host estimate, ptxas 13.3): STL / LDL 0, REG 128 entry, the 144 / 144 / 80 split, `BAR.SYNC` 9 -> 5, `UBLKCP` 1 -> 2,
    LDG 24 -> 19, the softmax segment 598 -> 509 instructions, the loader segment 319 -> 359.  Declines BY NAME under warp_roles "16w_fa4" (the
    first build sites the ring in the 16w bundle: BAR_LAYOUT_RING); SERVED with debug_clk (DBG_CLK_RING: 88 words, the two ring wait words), with
    every diag_2cta probe (`stats_one_chunk` reads one chunk of the stage) and every mask_form / dkv_accum value.  Measured on GB200
    (cc 10.0, 2026-10-28): +1.4 .. +1.9 % step at 16K / 32K against "ldg" -- the default since |
    tmem_map (``tmem_map_code`` Constexpr[int] from config_sm100_d128_accum_2cta_stail.TMEM_MAP_CODES; PART-KEYED: ``resolve_tmem_map(name, cc,
    roles)`` in the prepared module (Q's device cc) and in compile_host (``sm`` -> cc); the compile key carries the NAME; the SOL card's arm R1 of
    the Rubin task, 2026-11-03): "s_tail" = THE default on EVERY part (the only map served on the 512-column parts cc 10.0 / 10.3: a NAMED
    "dq_576" is the engine's typed decline there, the 576-column carve does not exist) = the rendering of sections 5 / 9 as written,
    byte-identical (the sm_100a / sm_103a records; main-kernel md5 d9b419aa @ 2/8/2 on sm_107a too) | "dq_576" = selectable BY NAME on cc 10.7
    under warp_roles 16w, NOT the default (section 13: a NEUTRAL structural map; the record md5_d128_accum_grouped_2cta_stail_dq576_sm_107a.txt
    is the NAMED rendering; main-kernel md5 11df3211 @ 2/8/2): the dQ image at
    [512, 576) of the 576-column carve (geo = CFG_DQ576 / CFG_RING_DQ576: TMEM_COLS 576 = TMEM_COLS_SM107, TMEM_IS_EXCLUSIVE ->
    `tcgen05.alloc.exclusive.cta_group::2` on both CTAs, the SAME constant at the dealloc; DQ_COL 512), the rows s_read + s_tail_empty DELETED
    (BAR_LAYOUT_DQ576 / BAR_LAYOUT_RING_DQ576; the leader's five wait sites, its s_tail state, the P15 drain, the softmax's fence + arrive and the
    epilogue's arrive are not traced; every other arrive / commit / wait / fence unchanged), the reuse edge of section 5 carried by the epilogue's
    program order; SMEM and the workspace unchanged.  Census vs s_tail (host render @ 2/8/2, sm_107a): mbarrier.arrive 42 -> 40, try_wait 70 -> 65,
    bar.warp.sync 69 -> 64, tcgen05.fence::after_thread_sync 32 -> 27, mapa 17 -> 15 (the two `_arrive_leader` sites), mbarrier.init 43 -> 41,
    tcgen05.fence::before_thread_sync 8 -> 7 (the softmax's fence before its s_read arrive), `tcgen05.alloc.cta_group::2` 1 -> 0 and the exclusive
    form 0 -> 1; every other row equal (72 MMA, 13 commits, 12 TMA, the ship, 4 reduce-adds, 12 expect_tx, 38 LDS.128, .local 0, nanosleep 0, the
    144 / 144 / 80 split).  Declines BY NAME: "dq_576" under warp_roles "16w_fa4" (the reduce group has no program-order carrier: it would need the
    dq_read_done row named in the skeleton), the three S-tail WAIT probes under "dq_576" (their sites are the deleted rows); SERVED with every other
    probe, every mask_form, dkv_accum, stats_form and debug_clk (DBG_CLK_DQ576 / DBG_CLK_RING_DQ576: the same word map, the four S-tail words dead =
    0).  Perf-only: dK / dV BITWISE between the maps (the same accumulation chain per GEMM; the test module's twin on cc 10.7), dQ by the fp64 gate.
    Measured 2026-11-03 on a Rubin class node (cc 10.7, 212 SMs, A/B/A x3 x100 both build orders): the ceiling probe s_tail_wait_off reads inside
    noise (+0.0 / -0.3 % step at 16K / 32K) and the two maps measure NEUTRAL (band 1.00) -> s_tail stays the default on every part; dq_576 is the
    structural base of the two-stage dS staging arm (section 13) |
    smem_map = "sm100_oversized" (DIAGNOSTIC, 2026-11-03; cc 10.7 by name, never a default; code 2): the sm100 map ITSELF -- every slab base, ring
    depth, barrier row and address identical, the body renders BYTE-IDENTICAL to the sm100 rendering of the same TMEM map -- declared with
    SMEM_BYTES 262144 (CFG_RING_OVS / CFG_RING_DQ576_OVS: SMEM_PAD_BYTES 29696 of dead space above the CLC payload), so the DSL launcher takes the
    oversized carveout for the shipping map: the probe of the carveout's own cost (the SKU card's L1 shrink: tensor-map / descriptor / CLC payload
    fetches) raised by the R2.c STAMPS; output CORRECT (--validate may run it); 16w x tma_ring, both TMEM maps; the benchmarker's A/B/A against the
    part's default at 16K / 32K is its number.
    smem_map (``smem_map_code`` Constexpr[int] from config_sm100_d128_accum_2cta_stail.SMEM_MAP_CODES; PART-KEYED: ``resolve_smem_map(name, cc, roles,
    stats_form, tmem_map)`` in the prepared module (Q's device cc) and in compile_host (``sm`` -> cc); the compile key carries the NAME; the SOL card's
    arm R2.c of the Rubin task, 2026-11-03): "sm100" = THE default on every part (the 232448-B opt-in slab chain of section 8 as written, byte-identical:
    the sm_100a / sm_103a / sm_107a records, main-kernel md5 d9b419aa @ 2/8/2; dq_576 x sm100 11df3211) | "rubin_b2" = the 327 KiB oversized carveout
    of cc 10.7, selectable BY NAME together with tmem_map "dq_576" under the default role shape and stats form (the record
    md5_d128_accum_grouped_2cta_stail_dq576_b2_sm_107a.txt; main-kernel md5 a58973ea @ 2/8/2): geo = CFG_RING_DQ576_B2 (SMEM_BYTES 314368; sQ1 x2,
    sDO1 x2, sDS x2 + sDS_xchg x2; every tcgen05 root < 262144); the ring Constexprs are the geometry's stage fields (the `_slot` / `_do1_slot` forms,
    `advance(.., geo.<RING>_STAGES)`, the per-tile sDO1 descriptor, the per-issue dS descriptor advance); the epilogue's LAGGING order (section 5: dS(t)
    staged into stage sds_state.idx BEFORE the dq_done(t-1) wait; round B of dQ(t-2) carried across the staging; the item end reads dQ(hi-1) out and
    runs rounds A, B); the NEW LEADER row dq_read_done (BAR_LAYOUT_RING_DQ576_B2: 36 names / 51 slots) arrived after EACH readout trio and waited by the
    leader inside `_dq_issue` before EVERY dQ issue (pre-armed; one P15 drain); the loader's whole-ring drains at depth 2.  Census vs dq_576 x sm100
    (host render @ 2/8/2, sm_107a): mbarrier.arrive 40 -> 44 (the two dq_read_done sites + the two extra dQ rounds' dq_full), the relaxed cluster arrive
    5 -> 7, mapa 15 -> 16 (one hoisted mapa serves both dq_read_done sites), try_wait 65 -> 75 / bar.warp.sync 64 -> 74 (dq_read_done x 3 = the two issue
    sites + the drain, the item-end dq_done wait, 2 x 2 dq_empty waits of the extra rounds, the two ring drains at depth 2), tcgen05.fence::after 27 ->
    31, fence::before 7 -> 8 and tcgen05.wait::ld 8 -> 9 (the item-end readout), mbarrier.init 41 -> 50 (8 grown slots + the row), fence.proxy.async 15
    -> 17 and st.shared.v2.b64 24 -> 40 (the two extra rounds); every other row equal (72 MMA, 13 commits, 12 TMA, the ship, 4 reduce-adds, 12 expect_tx,
    38 LDS.128, .local 0, nanosleep 0, the 144 / 144 / 80 split, the exclusive alloc).  Declines BY NAME: "rubin_b2" on a 512-column part (the engine's
    typed decline: no oversized carveout), under tmem_map "s_tail" (the lagging order voids the S-tail map's carrier; the B2 table has no seat for its two
    rows), under warp_roles "16w_fa4" (no B2 table for the reduce group) and under stats_form "ldg" (the B2 table and its record are sited under tma_ring:
    one host render + record admits ldg, not done 2026-11-03); SERVED with every diag_2cta probe outside the S-tail WAIT probes (exchange_off arms 0 bytes
    per slot), every mask_form, dkv_accum (the fp32 slots on the stage-0 bytes) and debug_clk (DBG_CLK_RING_DQ576_B2: the same word map, the leader's
    dq_read_done pair live).  Perf-only: dK / dV BITWISE between the SMEM maps (the test module's twin on cc 10.7), dQ by the fp64 gate.  Measured 2026-11-03 on a Rubin
    class node (cc 10.7, 212 SMs, A/B/A x3 x100 both build orders; control dq_576 x sm100 NEUTRAL): a LOSS, -2.5..-5.0 % step at 8K / 16K / 32K (the
    stamped tile loop is 4.4 % faster; the loss sits outside the loop or in the SMEM pipe) -- kept as a selectable map, not a default (section 13) |
    warp_roles (``warp_roles_code`` Constexpr[int] from config_sm100_d128_accum_2cta_stail.WARP_ROLES_CODES: "16w" = THE default = the
    shipped rendering, byte-identical (md5 4b3d0bbb @ 4/32/4, 645f8a24 @ 2/8/2) | "16w_fa4" = an alternate role shape kept for follow-up
    work: the softmax writes sDS own + XCHG from its dS^T registers and ships, warps 8-11 = the REDUCE group (`_reduce_loop`), ds_tmem +
    dp_empty deleted and dq_war_done added (BAR_LAYOUT_XFA4, 33 rows), the softmax's dq_war_done WAR wait, XCHG_BARRIER at 256 lanes, s_read
    kept, dQ ring form A (header sections 4 / 5 / 6); measured slower than 16w on GB200 at 8K-64K.  Perf-only on dK / dV (BITWISE vs 16w: the
    same accumulation chain per GEMM, the same k-orders; nothing of the softmax's arithmetic moves), dQ by the fp64 gate (the reduce order is
    the ring's, unchanged).  Under "16w_fa4" ``debug_clk`` (no STAMPS record layout for those roles) and ``dkv_accum = "fp32_reduce"`` (the
    fp32 staging rounds live in the 16w epilogue loop) decline BY NAME at compile_host and at the trace; ``exchange_off`` is sited at the
    softmax's arm + ship (the same two guards: the arm's bytes 0, no ship); dvk_stage_empty is UNSERVED (absent from BAR_LAYOUT_XFA4 /
    BarsXFA4); mask_form, stats_one_chunk, sdq_sts_off and dq_reduce_add_off serve both role shapes) |
    mask_form (``mask_form_code`` Constexpr[int] from config_sm100_d128_accum_2cta_stail.MASK_FORM_CODES: "fsel" = THE default, the shipped
    rendering: _p_values applies the keep-word band mask (band_mask_words + apply_mask_words; CFG.MASK_WORDS_PER_LANE_PER_TILE = 2 keep-words,
    CFG.MASK_SELECTS_PER_LANE_PER_TILE = 64 selects = R2P + FSEL per cell) on EVERY q tile | "predicated" = the SAME mask body under the
    CTA-uniform runtime branch `if mask_tile:` of _p_values (one branch per q quarter = 2 per tile), mask_tile = tile_may_mask(pair, rank,
    qtile, seq_q, seq_k, causal) (section 7) computed once per tile in _softmax_loop: TRUE on the causal straddle (qtile * N < kv_end: CTA 0's
    tile 2 pair, CTA 1's tiles 2 pair and 2 pair + 1), on the last partial q tile ((qtile + 1) * N > seq_q) and on a CTA whose kv rows reach
    seq_k (kv_end > seq_k: the padded second half, a partial last kv block); an interior tile packs the exp2 values straight -- 0 selects, 0
    keep-words, 0 R2P and no `hi` select per lane (~90 fewer instructions per lane per tile) -- and EXACTLY the fsel values (the exactness
    argument of section 7; test module `tile_mask_predicate_is_exact`), so P / dS and therefore dQ (per reduce-add order) / dK / dV are BITWISE
    between the two forms; no live register added (the predicate is a per-tile scalar, the keep-words and `hi` live inside the branch); the
    dS path (_ds_values) is untouched and inherits the exact zero; measured neutral against fsel on GB200 at 16K / 32K, kept selectable |
    "diag_mask_off" = DIAGNOSTIC: the mask body on NO tile (the lever's ceiling; OUTPUT WRONG BY DESIGN on the diagonal / edge tiles and the
    dead second half; resolved like the diag_2cta probes: on the adapter INSTANCE, keyed apart in the plan cache, a probe driver's `--validate`
    refuses it).  No barrier, SMEM or TMEM change under any value (the PTX mbarrier / MMA / commit census identical to the default's = the
    test module's _EXPECTED row, nanosleep 0, .local 0; the setmaxnreg split unchanged) |
    dkv_accum (the typed OP-LEVEL field ``SdpaBwdDslSm100D128.dkv_accum`` -> ``dkv_fp32`` Constexpr[bool]: "bf16_partials" = THE default,
    the bf16 dV / dK halves staged by `_native_out` + `tma_store_tile` into the per-Q-head partials + the fold launch; "fp32_reduce" = the fp32
    tiles staged by `_stage_fp32_slice` as [128 x 32] fp32 slices over the same bytes (+ the dS slab) in two rounds (5 + 3, dvk_stage_empty
    between) and reduce-added by the store warp (`tma_reduce_add_subtile` add.f32, the dQ slices' primitive) into the per-KV-head fp32
    workspace maps the MAP_DV / MAP_DK slots carry (128-row fp32 maps, kv head = head // ratio), the fold launch gone, the one bf16 rounding in
    cast_dq; a ROUNDING CHANGE (fp32 accumulation across the q heads of a kv head in hardware order, one bf16 rounding at the output; dK / dV
    not bitwise reproducible across runs), an op attribute, never a knob; dQ untouched; declines BY NAME on dense and MHA and under 16w_fa4;
    measured neutral on the backward step at 8K-32K: the removed fold launch is paid back by the in-kernel fp32 reduce-add traffic) |
    diag_2cta (DIAGNOSTIC; ``diag_2cta_code``, the code space config_sm100_d128_accum_2cta.CFG.DIAG_2CTA_NAMES = the four probes this body
    has code for, every other name declines BY NAME before the trace: None = this body | dq_reduce_add_off (the store warp's four reduce-add
    ISSUES skipped; the per-round `tma_store_wait` returns at once and the two dq_empty arrives per round stay, so the 2-slot ring balance
    holds with dQ staying zero) | exchange_off (the SAFE exchange-family ceiling: the epilogue's `_ship_xchg` is NOT issued and
    `ds_recv_full.arrive(n_bytes=CFG.DIAG_EXCHANGE_OFF_TX = 0, pred=ship_pred)` completes each phase at its own arm; the three dS gates, the
    follower relay, the XCHG STS / fence / named barrier and EVERY other wait stay; the recv half is never written: dQ wrong; what it deletes =
    the ship + its landing latency + the peer dependence) | stats_one_chunk (`_p_values` / `_ds_values` load ONE 16-B stats / delta chunk per
    q quarter (CFG.DIAG_STATS_CHUNK) and reuse it for all CFG.DIAG_STATS_CHUNKS_PER_QUARTER chunks: LDS.128 per lane per tile
    CFG.DIAG_STATS_LDS_PER_LANE 32 -> 4, registers / exp2 / FFMA / waits unchanged; stats wrong by design) | sdq_sts_off (`_dq_round` skips the
    sDQ slice STS + its proxy fence in every round; the dq_empty waits, the dq_full arrive and the store warp's reduce-add (reading stale
    staging) stay = -256 wavefronts per tile per CTA) | s_tail_wait_off (the LEADER skips its CFG.DIAG_S_TAIL_WAIT_SITES = 4 s_tail_empty
    `_wait_tcgen` sites: the prologue S^T(lo), the lookahead S^T(t+1) iff qtile > lo, the tail dQ(hi-1) iff hi - 1 > lo, the P15 drain; the
    s_tail_state advances, the epilogue's pair-wide 256 arrive and every other wait stay -- s_tail_empty is a LEADER relaxed thread-count row
    with no tx bytes, so its phases flip unobserved without an over-arrival fault (section 5); S^T(t+1) / the tail dQ may overwrite an unread
    dQ(t-1): dQ WRONG BY DESIGN; what it measures = the ceiling of the s_tail_empty edge, i.e. what an independent dQ TMEM image buys) |
    s_read_wait_off (the LEADER skips its CFG.DIAG_S_READ_WAIT_SITES = 1 s_read `_wait_tcgen` at slot 2 of every tile; the softmax's
    512-lane arrive and ready_state stay, the same relaxed-row argument; dQ(t-1) may land under the softmax's S^T(t) read: S / dQ WRONG BY
    DESIGN; the ceiling of the s_read edge) | s_tail_and_s_read_wait_off (both guards in ONE rendering, no third code path: the ceiling of the
    whole S-tail alias); OUTPUT WRONG BY DESIGN under a probe; a probe driver's `--validate` refuses them) |
    debug_clk (STAMPS; the typed bool field -> the ``debug_clk`` Constexpr + the ``dbg_clk`` Int32 dump tensor, None = dropped from the
    signature when off: the default rendering is byte-identical; the d512 bodies' `debug_clk` pattern): every mbarrier wait of the role's
    representative warp (section 4) is bracketed by %clock64 and its delta accumulated per (role, barrier) into the record of config
    DBG_CLK (the leader also counts waits above DBG_CLK.BLOCKED_THRESH); the epilogue lead warp adds per tile: ds_tmem pass -> ship issue
    (sum, the DBG_CLK.HIST_EDGES histogram, the max), ds_tmem pass -> dq_done pass, dq_done pass -> s_tail_empty arrive; the ship -> recv
    handshake (the leading warp writes the ship clock + the CTA's tile count as a tag, the MMA warps read it after their ds_recv_full wait: the
    SAME-CTA proxy of the exchange flight, both CTAs being symmetric; a tag mismatch counts a miss); the store warp times its
    `tma_store_wait` drains; warp 0 zeroes the record in the P4 init block and dumps it after the final CTA sync to dbg_clk[(linear CTA) x
    DBG_CLK.WORDS ..] (the adapter carves the region past the chain's workspace, config dbg_clk_region_offset; a CLC-cancelled cluster never
    runs, so its records stay unwritten and the magic word marks the ones that did).  Output CORRECT (bitwise dK / dV: the test module's
    twin), timing diagnostic: a ~30-60 clk issue cost per wait in the representative lane only; declines by name under 16w_fa4).
    Test-only skew levers (house form `DEBUG_<SITE>_DELAY_US = int(globals().get("FROST_D128_2CTA_DEBUG_<SITE>_DELAY_US", 0))`, a
    `nanosleep.u32 1000` loop under const_expr, default 0 = no traced code; the test module's census pins nanosleep == 0 on the default
    rendering): LOADER (k2_empty back-edge), FOLLOWER_OWN_HALF (ds_pair_ready), LEADER_SHIP (ds_recv_full + the ship's routing),
    FOLLOWER_EPILOGUE_ITEM_END (dvk_empty 64), SOFTMAX_S_READ (the FOLLOWER softmax between its s_done wait and the S^T ld, every tile --
    isolates s_read: with the follower asleep dQ(t-1) is held by s_read ALONE; RED = the wait removed before dQ(t-1): the follower's P^T(t) q
    columns 64..127 = exp2 of dQ values -> dV / dK RED), EPILOGUE_DQ_READOUT (the FOLLOWER epilogue between its dq_done wait and the dQ ld --
    isolates s_tail_empty: with the follower asleep S^T(t+1) is held by s_tail_empty ALONE; RED = the wait removed at the lookahead: the
    follower's 64 dQ rows of tile t-1 = S^T values; at the tail: dQ rows of tile hi-2 wrong; at the prologue: dQ rows of tile hi-1 of the
    previous item wrong), SOFTMAX_DQ_WAR (the leader's `_dq_issue` between the gate waits and the dQ issue -- isolates the sDS WAR wait of
    the 16w_fa4 softmax), and the rubin_b2 ring's two instruments (the barrier skeleton R2.c 2.8): LEADER_ARM_DELAY (the LEADER epilogue right
    before its ds_recv_full[s] arm -- the follower's ship of slot s lands first: the transiently negative tx-count path; SKEW_OK expected, a
    fault = a wrong mbar / stage mapping in `_ship_xchg`) and LEADER_SLOT1_DELAY (the LEADER MMA warp at slot 1 of every tile -- relay(t) and
    the staging of t land before flip t-1 of the 2-slot ds_* rings is consumed at slot 2: the depth-2 ABA proof; SKEW_OK expected, a 1-slot
    form would hang).  Deterministic-hang topologies are SOURCE assertions in the test module (RED 3: dp_empty arrived after the dQ
    readout; RED 4: the dS gates on ready_state; RED 5: an unguarded lookahead wait; the ring release cadence: the streaming release with the both-slots wait form).

12. DEGENERATE-INPUT MATRIX  (row -> the handling LINE; absence is the bug)
    S_kv = 0 (dead pair: lo = hi = 0, stores skipped: _item_bounds / _store_loop `live_rows`) | kv rows >= seq_k in the pair's SECOND half
    (whole CTA dead: masked columns P = 0 exactly (_p_values `hi = 0`), dK / dV stores skipped by `live_rows`, dS recv half still
    exchanged = zeros, s_read / s_tail_empty still arrived by its softmax / epilogue: the pair-wide counts need both CTAs; mask_form predicated:
    tile_may_mask is TRUE on every tile of a dead CTA (kv_end > seq_k) so the mask body runs there) | q tile fully
    masked for CTA 1 at the causal diagonal (mask words, never a skip; under predicated tile_may_mask is TRUE on it: the mask body runs) |
    interior tile under mask_form predicated (tile_may_mask FALSE: (qtile + 1) * N <= seq_q, kv_end <= seq_k, and qtile * N >= kv_end under
    causal): no cell masked by construction, the exp2 values packed straight = what the fsel form selects through (bitwise) |
    n_q tiles = 1 (N == 1: no in-tile dQ, the tail dQ(lo) without an
    s_tail_empty wait, the readout fires once for the next item's prologue) | N == 2 / N == 3 (the walk of section 5) | S % 256 != 0 (the
    256-row THD unit; the sequence-local maps clip) | one pair per launch | padded Q rows (stats 0, mask word `hi = seq_q`, no store: the
    64-row dQ map clips; mask_form predicated: the last partial q tile has (qtile + 1) * N > seq_q, so its mask body runs) |
    zero-trip owner (CLC envelope) on BOTH CTAs (the zero-trip protocol of section 6: no dq_empty traffic at all) |
    a zero-trip item between two live items (the pending s_tail_empty / dp_empty completions of the previous live item's tail stay for
    the next live item's prologue: the states are never re-armed per item) | the kernel's first item (s_tail_empty / dp_empty / dvk_empty
    pre-armed st(1); dq_empty pre-armed on the epilogue side) |
    tmem_map dq_576 (the independent image's N-walk, section 5): N == 1 (the tail dQ(lo) over the previous live item's dQ(hi'-1), read out before
    its end rounds and before this item's dS(lo) STS) | N == 2 / 3 (every in-tile and tail issue behind the dS gates of the tile after the readout) |
    a zero-trip item (no dQ write, no readout, no gate flip: the next live item is the N == 1 case) | the kernel's first item (no tenant) | the
    half-dead pair (CTA 1 reads its 64 dQ rows out and arrives ds_smem every tile: identical flips) | the STAMPS record under the map (the four
    S-tail words read 0; every other word live: the dump test skips exactly `dbg_clk_dead_words`) |
    smem_map rubin_b2 (the lagging order + the 2-slot rings, section 5): N == 1 (no in-loop readout; the item end reads dQ(lo) out and arrives
    dq_read_done once; the tail issue's dq_read_done wait consumed the previous live item's last readout or the pre-arm; every ring flips once on slot
    K % 2) | N == 2 / 3 (in-loop readouts of dQ(lo..hi-2) under `qtile > lo`, round B of dQ(t-2) under `qtile - 1 > lo`, the item-end readout of
    dQ(hi-1) + rounds A, B; 2N rounds per slot) | a zero-trip item (no issue, no readout, no flip on any 2-slot ring or on dq_read_done; the next live
    item continues the running stage count) | the kernel's first item (dq_read_state pre-armed; stage 0) | a zero-trip item between two live items (the
    carried states, never re-armed) | the half-dead pair (CTA 1 stages, ships zeros, reads out and arrives dq_read_done every tile: identical flips) |
    the STAMPS record under the map (the leader's dq_read_done words live; the four S-tail words dead) |
    stats_form tma_ring: a partial last q tile (S % 128 != 0: the record's pad rows [S_b, ceil(S_b / 128) * 128) hold 0.0 written by the delta
    launch = the ldg form's `stat = 0`; `delta_zero_stats` grid covers ceil(max_sq / 128) * 128 rows so the longest sequence's pads are written) |
    len % 128 == 1 (one live row + 127 pad rows in the record) | an empty sequence (no record read: the item is dead; PAD_TOTAL = rounded_q + 128 B
    bounds every sequence's padded block, asserted in validate_config) | tile lo of the kernel's second item (stage K % 2 for the carried K, never
    stage 0) | n_q == 1 items between zero-trips (one copy / one release each; the release consumed two live tiles later, whichever item) |
    both Stats layouts (token-major [T, H] and head-major [1, H, pitch]: the delta launch reads either through the Stats port's token origin) |
    the kernel's first two tiles (the loader's stats_empty waits pass on the pre-arm; the softmax's first stats_full wait blocks until the copy lands).

13. MEASURED LEVERS  (A/B/A x 3 x 100 in one process, control twin, band max(1 %, |pair|); one GB200-class node, clock locked, the sampled
    busy clock recorded per process; in-process ratios are the figures, absolutes are clock-capped lower bounds; the record
    is retained internally)
    lever | result | verdict | stands?
    this body (the pair, warp_roles 16w, mask_form fsel, dkv_accum bf16_partials) vs the one-CTA grouped THD body at the backward STEP | 8K
    +4.9 %, 16K +7.9 %, 32K +8.5 %, 64K +8.8 %, varlen ramps +6.5 % (16K) / +8.3 % (32K); 2K -3.9 %, 4K -3.5 % (losses on the step) | the pair serves S >= 8K
    (`CTA_PAIR_MIN_SEQ`), the one-CTA body below | yes: the heuristic's threshold
    mask_form predicated vs fsel | 16K +0.4 / +0.8 %, 32K +0.6 %, 16K ramp +0.2 %, 8K inside noise; ceiling (diag_mask_off) +0.97 % | neutral
    inside the band -> fsel stays the default, predicated selectable | yes
    dkv_accum fp32_reduce vs bf16_partials (both bodies) | inside the band at 8K / 16K / 32K and the 16K ramp (-1.5..+1.8 %); the fold launch's
    saving is paid back by the in-kernel fp32 reduce-add traffic | neutral as a perf lever; kept as a numerics option (tighter dK / dV, one
    launch fewer), default bf16_partials | yes
    warp_roles 16w_fa4 vs 16w | about -5.8 % on the step at 16K and 32K | a loss beyond the band -> kept as a selectable value for follow-up
    work, the default stays 16w | yes
    diagnostic probes (exchange_off / stats_one_chunk / sdq_sts_off; output wrong by design) | ceilings only, never shipped armed | the room
    of the exchange family and of the SMEM data pipe at 16K / 32K is inside a few percent of the step | n/a
    tmem_map dq_576 vs s_tail on cc 10.7 (arm R1 of the Rubin task; the independent dQ image, two LEADER rows fewer) | its CEILING, the probe
    s_tail_wait_off on the s_tail map (the three D1 waits + the P15 drain skipped): +0.0 / +0.55 % at 16K and -0.3 / +0.27 % at 32K (step ms /
    main-kernel cycles), the combined s_tail + s_read probe -0.4 / -0.2 % (benchmarker, rubin-perf-212, A/B/A x 3 x 100 in one process, both build
    orders, 2026-11-03) -- inside the band, below the +2.0 % MAIN-cycle threshold (band max(1 %, |pair|) + 1 pt) at both cells | the map buys
    nothing on its own: NO perf claim; it is a NEUTRAL structural map kept selectable by name as the base of the two-stage sDS + XCHG staging arm
    (whose ceiling exchange_off read +5.4 / +6.7 % step at 16K / 32K; the inspector approved that arm on this map with a dq_read_done row); the
    default stays s_tail on every part and flips only with that arm's own row | yes (structural base; no default flip)
    smem_map rubin_b2 vs sm100 on cc 10.7, both ON tmem_map dq_576 (arm R2.c of the Rubin task: the two-stage sDS + XCHG staging ring, sQ1 / sDO1 x2,
    the dq_read_done row) | measured 2026-11-03 on a Rubin class node (A/B/A x 3 x 100 in one process, both build orders; the control twin dq_576 x
    sm100 NEUTRAL): a LOSS of -2.5..-5.0 % step at 8K / 16K / 32K (the stamped tile loop is 4.4 % faster under debug_clk; the loss sits outside the
    loop or in the SMEM pipe -- the follow-up read); its ceiling probe exchange_off had read +5.4 / +6.7 % step at 16K / 32K, hence the build | a loss
    beyond the +2.0 % MAIN-cycle threshold -> kept as a selectable, part-keyed, off-by-default map (SMEM_MAP_DEFAULT_BY_PART stays sm100 on every
    part; neither this map nor dq_576 is a cc 10.7 default) | yes (selectable map; no default flip)
    Earlier pair variants that lost or came out neutral (a second TMEM map with the dQ image in dP^T's dead half, a second leader issue
    order, register-direct global dQ reduces, a plain TMA load form, a two-instruction store wait, a 136/136/104 register split) are not
    fields of this body.

14. LAUNCH ABI
    positional args (maps, stats_tokens, stats_rows, meta, origins, n_batch, nheads, max_skv, stats_origin, scale, causal, dtype, ratio,
    diag_2cta_code, warp_roles_code, mask_form_code, stats_form_code, tmem_map_code, smem_map_code, dkv_fp32, debug_clk, dbg_clk [Optional: the Int32 STAMPS dump of config
    dbg_clk_records(n_batch, nheads, max_skv) x DBG_CLK(_RING).WORDS words, None = dropped unless debug_clk], stream) + cluster (CFG.CGA_M,
    CFG.CGA_N, 1); grid x = ceil(max_skv / CFG.THD_UNIT_ROWS) * CFG.CGA_M, the declared envelope; the prepared host's RUNTIME signature is the
    grouped THD chain's (every typed field is a Constexpr keyed by NAME in the compile cache; ``tmem_map_code`` joined after ``stats_form_code``, the
    ints together, 2026-11-03: the PART-KEYED TMEM map of sections 9 / 11; ``smem_map_code`` joined after ``tmem_map_code`` the same day: the PART-KEYED
    SMEM map of sections 8 / 11).  The slab the kernel declares (`cutlass.Array(.., geo.SMEM_BYTES, space=smem)` = dynamic SMEM) is 231424 / 232448 (262144
    under the DIAGNOSTIC map sm100_oversized: the same map, padded) on the
    sm100 map and 314368 under rubin_b2: the DSL launcher sizes the CTA's shared memory from it and sets the oversized mode above the 232448-B opt-in cap
    (cc 10.7's attribute; the first launch is the proof, a launcher without the mode fails it).  The two STATS SLOTS carry the stats form's operands
    -- ldg: the user's LSE tensor ([T, H] or [1, H, pitch]) and the chain's compact delta [1, H, rounded]; tma_ring: the padded row offsets
    ``rowoff`` [B + 1] Int32 and the padded pre-scaled stats region ``stats2`` [H][PAD_TOTAL / N][2][N] fp32 -- two plain `cute.Tensor` slots
    under both forms, because an `Optional` annotation or an added operand changes the ldg rendering's mangled entry name (its md5 pin) even
    when the value is None.  Workspace (d128_grouped_2cta_contract, ABI 2): under tma_ring the compact delta region is REPLACED by `stats2` +
    `rowoff` (every later offset follows); the ldg layout is the shipped one; the STAMPS region (debug_clk) follows the form's record size.

PDL SITES: none in the accum family today (CLC-scheduled persistent envelope); unchanged.

Sources: config_sm100_d128_accum_2cta_stail.py (the pasted barrier tables; BAR_LAYOUT_RING derived); the one-CTA grouped THD body (section 7)
(the chain's other launches); FAv4 flash_bwd_sm100.py NOTE [2CTA schedules] + the pipelined MMA loop (read as the structural cross-check only:
S(t+1) first, dQ before dV, Q_stage 1, sdQaccum not aliased, the deferred dS commit = our dS gates + s_read, pipeline_dQ empty before S(t+1) =
our s_tail_empty; nothing copied).
"""

# The 2-CTA per-sequence tensor-map set (header section 10); box rows per slot = CFG.MAP_BOX_ROWS, in this order (== the first body's).
MAP_Q1, MAP_K, MAP_V, MAP_DO1, MAP_DQ, MAP_DK, MAP_DV, MAP_Q2, MAP_DO2 = range(CFG.MAPS_PER_SEQUENCE)
MAP_COUNT = CFG.MAPS_PER_SEQUENCE
# STATS FORM (header section 11): the typed adapter field ``stats_form`` reaches the kernel as the ``stats_form_code`` Constexpr[int]
# (config_sm100_d128_accum_2cta_stail.STATS_FORM_CODES).  STATS_FORM_LDG = the default (the softmax's per-tile `_stats` LDG fetch under the
# SOFTMAX_BARRIER pair, today's rendering); STATS_FORM_TMA_RING = the loader-fed two-stage SMEM stats ring (stats_full / stats_empty, the
# CFG_RING geometry: SMEM_BYTES == the cap, the scaffold re-based +1024) fed by ONE 1 KiB bulk copy per q tile from the chain's padded
# pre-scaled stats region (the delta launch writes it; no tensor map).
STATS_FORM_LDG = stats_form_code("ldg")
STATS_FORM_TMA_RING = stats_form_code("tma_ring")
assert (STATS_FORM_LDG, STATS_FORM_TMA_RING) == (0, 1)
STATS_RECORD_ELEMS = CFG.STATS_RECORD_BYTES // CFG.BPE_ACC  # 256 fp32: one (head, q tile) record of the padded region = one ring stage [LSE 128 | delta 128]
assert CFG.STATS_RECORD_BYTES == CFG.STATS_STAGE_BYTES == CFG.STATS_TX == 1024 and CFG_RING.STATS_STAGES == 2 and CFG.STATS_PAD_GRAN == CFG.N

# Role warp ids (CFG.SOFTMAX_WARPS softmax, CFG.EPILOGUE_WARPS epilogue, then the four service warps).
SOFTMAX_WARP_END = CFG.SOFTMAX_WARPS
EPILOGUE_WARP_END = SOFTMAX_WARP_END + CFG.EPILOGUE_WARPS
EPILOGUE_LEAD_WARP = SOFTMAX_WARP_END  # the one warp whose elected lane arms ds_recv_full and ships XCHG
MMA_WARP = EPILOGUE_WARP_END
TMA_LOAD_WARP = MMA_WARP + 1
TMA_STORE_WARP = MMA_WARP + 2
SCHED_WARP = MMA_WARP + 3  # the role dispatch's `else` branch: the CTA's LAST warp (pinned below; the dispatch never spells the id)
if SCHED_WARP != CFG.THREADS // 32 - 1 or SCHED_WARP != EPILOGUE_WARP_END + CFG.SERVICE_WARPS - 1:
    raise ValueError(f"the scheduler warp must be the CTA's last warp (the dispatch's else branch is exactly one warp); got {SCHED_WARP}")
# Test-only skew levers (house form; default 0 = no traced code: the test module's census pins nanosleep == 0 on the default rendering; the
# test sets the module global BEFORE the module executes -- spec loader + sys.modules; never an environment variable, never a knob).
# Carried from the first body (the barrier audit AUDIT 2026-10-26): LOADER (lever 4, the k2_empty back-edge: both CTAs' loaders sleep at the
# top of every item), FOLLOWER_OWN_HALF (1, ds_pair_ready), LEADER_SHIP (2, the relay's ds_recv_full wait + the P9 routing of the ship),
# FOLLOWER_EPILOGUE_ITEM_END (3, dvk_empty's pair-wide 64).
DEBUG_LOADER_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_LOADER_DELAY_US", 0))
DEBUG_FOLLOWER_OWN_HALF_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_FOLLOWER_OWN_HALF_DELAY_US", 0))
DEBUG_LEADER_SHIP_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_LEADER_SHIP_DELAY_US", 0))
DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US", 0))
# NEW for ARM (ii) (the barrier audit 2026-10-27 section 8, RED 1 / RED 2): SOFTMAX_S_READ = the FOLLOWER softmax between its s_done wait and
# the S^T ld pair, every tile (isolates s_read: dQ(t-1) is held by s_read ALONE); EPILOGUE_DQ_READOUT = the FOLLOWER epilogue between its
# dq_done wait and the dQ ld (isolates s_tail_empty: S^T(t+1) / the tail dQ / the next prologue are held by s_tail_empty ALONE).
DEBUG_SOFTMAX_S_READ_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_SOFTMAX_S_READ_DELAY_US", 0))
DEBUG_EPILOGUE_DQ_READOUT_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_EPILOGUE_DQ_READOUT_DELAY_US", 0))
# NEW for the alternate role shape (the barrier audit x-fa4 table section 12 item 12; the the resource planRED form): SOFTMAX_DQ_WAR = the LEADER's MMA warp between
# the three dS gates and the dQ(t-1) issue (inside _dq_issue, both sites, leader-only by construction): with the issue late, both softmaxes' dS(t)
# STS are held under 16w_fa4 by their dq_done(t-1) WAR wait ALONE -- RED = the `dq_war` wait removed: the dS(t) STS lands under dQ(t-1)'s A read of
# sDS -> dQ rows wrong, dK / dV green.  Under 16w the same lever only delays the epilogue's dq_done(t-1) wait (SKEW_OK).
DEBUG_SOFTMAX_DQ_WAR_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_SOFTMAX_DQ_WAR_DELAY_US", 0))
# NEW for smem_map rubin_b2 (the barrier skeleton R2.c section 2.8; the AUDIT R2.c follow-up: the correctness instruments of the ring's two new edges):
# LEADER_ARM_DELAY = the LEADER's epilogue warps right before their ds_recv_full[s] arm (the follower's ship of slot s lands FIRST: the transiently
# negative tx-count path; a fault here = a wrong mbar / stage mapping inside _ship_xchg); LEADER_SLOT1_DELAY = the LEADER's MMA warp at slot 1 of
# every tile, before its first wait (the follower's relay(t) and both epilogues' staging of tile t land before the leader consumed flip t-1 of the
# 2-slot ds_smem / ds_recv_full / ds_pair_ready rings at slot 2: the depth-2 ABA proof; a 1-slot form would hang).  SKEW_OK is the expected PASS of both.
DEBUG_LEADER_ARM_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_LEADER_ARM_DELAY_US", 0))
DEBUG_LEADER_SLOT1_DELAY_US: int = int(globals().get("FROST_D128_2CTA_DEBUG_LEADER_SLOT1_DELAY_US", 0))
# DIAGNOSTIC probes (header section 11): the typed adapter field ``diag_2cta`` reaches the kernel as the ``diag_2cta_code`` Constexpr[int] in
# the shared code space (config_sm100_d128_accum_2cta.diag_2cta_code); every other name declines BY NAME before the trace.
#   DIAG_DQ_REDUCE_ADD_OFF (dq_reduce_add_off): _store_loop skips the four tma_reduce_add issues (commit / wait / the per-round dq_empty arrives stay)
#   DIAG_EXCHANGE_OFF      (exchange_off):     _epilogue_loop skips the XCHG ship and arms ds_recv_full with CFG.DIAG_EXCHANGE_OFF_TX = 0 bytes; EVERY wait kept
#   DIAG_STATS_ONE_CHUNK   (stats_one_chunk):     _p_values / _ds_values load ONE stats / delta chunk per quarter and reuse it (LDS.128 32 -> 4 per lane per tile)
#   DIAG_SDQ_STS_OFF       (sdq_sts_off):      _dq_round skips the sDQ slice STS + its proxy fence (the reduce-add reads stale staging; waits / arrives stay)
#   DIAG_S_TAIL_WAIT_OFF   (s_tail_wait_off):  _mma_loop_leader skips its CFG.DIAG_S_TAIL_WAIT_SITES s_tail_empty waits (D1 i / ii / iii + the P15
#                                              drain); the s_tail_state advances and the epilogue's arrives stay (a LEADER relaxed row, no tx:
#                                              the phases flip unobserved, no over-arrival).  dQ WRONG BY DESIGN (S^T(t+1) over an unread dQ)
#   DIAG_S_READ_WAIT_OFF   (s_read_wait_off):  _dq_slot skips the leader's s_read wait (slot 2, every tile); the softmax's arrive and ready_state
#                                              stay.  S / dQ WRONG BY DESIGN (dQ(t-1) under the softmax's S^T(t) read)
#   DIAG_S_TAIL_AND_S_READ_WAIT_OFF (s_tail_and_s_read_wait_off): both guards at once (no third code path)
DIAG_DQ_REDUCE_ADD_OFF = diag_2cta_code("dq_reduce_add_off")
DIAG_EXCHANGE_OFF = diag_2cta_code("exchange_off")
DIAG_STATS_ONE_CHUNK = diag_2cta_code("stats_one_chunk")
DIAG_SDQ_STS_OFF = diag_2cta_code("sdq_sts_off")
DIAG_S_TAIL_WAIT_OFF = diag_2cta_code("s_tail_wait_off")
DIAG_S_READ_WAIT_OFF = diag_2cta_code("s_read_wait_off")
DIAG_S_TAIL_AND_S_READ_WAIT_OFF = diag_2cta_code("s_tail_and_s_read_wait_off")
assert set(CFG.DIAG_2CTA_NAMES_ON_MAP) == {
    "dq_reduce_add_off",
    "exchange_off",
    "stats_one_chunk",
    "sdq_sts_off",
    "s_tail_wait_off",
    "s_read_wait_off",
    "s_tail_and_s_read_wait_off",
}, "this body renders exactly these probes"
# STAMPS (header section 11): the typed adapter field ``debug_clk`` reaches the kernel as the ``debug_clk`` Constexpr[bool] + the ``dbg_clk`` dump
# tensor (Optional: None = dropped when off); the record layout is config truth (DBG_CLK: _W["wait:<role>:<barrier>"], _W["blocked:leader:<barrier>"],
# the header and segment names; the kernel never spells a word index).
_W = DBG_CLK.WORD
# MASK FORM (header section 11, the mask-form lever): the typed adapter field ``mask_form`` reaches the kernel as the ``mask_form_code`` Constexpr[int]
# (config_sm100_d128_accum_2cta_stail.MASK_FORM_CODES).  MASK_FORM_FSEL = the default (the keep-word mask selected on every tile);
# MASK_FORM_PREDICATED = the same mask body only on `tile_may_mask` tiles; MASK_FORM_DIAG_OFF = no mask on any tile (OUTPUT WRONG BY DESIGN).
MASK_FORM_FSEL = mask_form_code("fsel")
MASK_FORM_PREDICATED = mask_form_code("predicated")
MASK_FORM_DIAG_OFF = mask_form_code("diag_mask_off")
if {mask_form_code(n) for n in CFG.MASK_FORMS} != {MASK_FORM_FSEL, MASK_FORM_PREDICATED, MASK_FORM_DIAG_OFF}:  # config = truth: the body's
    # const_expr chain renders fsel, predicated and (anything else) diag_mask_off -- a fourth code would trace as the no-mask form, silently
    raise ValueError(f"this body renders exactly the three mask forms {CFG.MASK_FORMS}; the code space grew")
if CFG.MASK_WORD_COLS != MASK_WORD_COLS:  # config = truth: the config's keep-word width is the mask helper's (both 32; the config stays cutlass-free)
    raise ValueError(f"CFG.MASK_WORD_COLS {CFG.MASK_WORD_COLS} != tile_dsl.mask.MASK_WORD_COLS {MASK_WORD_COLS}")
# ROLE SHAPE (header sections 4 / 5 / 11; the later rank 1 "the alternate role shape"): the typed adapter field ``warp_roles`` reaches the kernel as the ``warp_roles_code``
# Constexpr[int] (config_sm100_d128_accum_2cta_stail.WARP_ROLES_CODES).  WARP_ROLES_16W = the default (today's rendering, byte-identical);
# WARP_ROLES_16W_FA4 = FAv4's role shape: the softmax writes sDS own + XCHG from its dS^T registers and ships, warps 8-11 = the REDUCE group
# (_reduce_loop), ds_tmem / dp_empty deleted (BAR_LAYOUT_XFA4), the softmax's dq_done WAR wait (the barrier audit the audit record).
WARP_ROLES_16W = warp_roles_code("16w")
WARP_ROLES_16W_FA4 = warp_roles_code("16w_fa4")
SOFTMAX_LEAD_WARP = CFG.SOFTMAX_LEAD_WARP  # 16w_fa4: the softmax warp whose elected lane arms ds_recv_full and ships XCHG (inspector X4; FAv4's tidx 0)
XCHG_BARRIER_LANES_16W = CFG.xchg_barrier_lanes("16w")  # bar.sync 2, 128: the 4 epilogue warps (today's form)
XCHG_BARRIER_LANES_16W_FA4 = CFG.xchg_barrier_lanes("16w_fa4")  # bar.sync 2, 256: the 8 softmax warps (inspector X3)
assert (WARP_ROLES_16W, WARP_ROLES_16W_FA4) == (0, 1) and XCHG_BARRIER_LANES_16W == CFG.EPILOGUE_LANES and XCHG_BARRIER_LANES_16W_FA4 == CFG.SOFTMAX_LANES
# TMEM MAP (header sections 9 / 11; PART-KEYED): the typed adapter field ``tmem_map`` reaches the kernel as the ``tmem_map_code`` Constexpr[int]
# (config_sm100_d128_accum_2cta_stail.TMEM_MAP_CODES; resolved per device cc by the prepared module / compile_host).  TMEM_MAP_S_TAIL = the dQ image in
# S^T's dead tail behind the s_read / s_tail_empty rows (the 512-column parts' rendering, byte-identical); TMEM_MAP_DQ576 = the INDEPENDENT dQ image
# at [DK_COL + DK_COLS, +DQ_COLS) of the 576-column carve (selectable by name on cc 10.7): the two rows, the leader's five wait sites + its s_tail state + the
# P15 drain and the two arrives are not traced; the alloc / dealloc constant (geo.TMEM_COLS, exclusive) and the dQ column base come from the map's
# geometry record (CFG_DQ576 / CFG_RING_DQ576), never a literal.
TMEM_MAP_S_TAIL = TMEM_MAP_CODES["s_tail"]
TMEM_MAP_DQ576 = TMEM_MAP_CODES["dq_576"]
assert (TMEM_MAP_S_TAIL, TMEM_MAP_DQ576) == (0, 1) and tuple(TMEM_MAP_CODES) == CFG.TMEM_MAPS == ("s_tail", "dq_576")
# the dQ image's column base per map code = the map's record (the stats form never moves the image)
DQ_COL_BY_MAP = {TMEM_MAP_S_TAIL: CFG.DQ_COL, TMEM_MAP_DQ576: CFG_DQ576.DQ_COL}
assert CFG_RING.DQ_COL == CFG.DQ_COL and CFG_RING_DQ576.DQ_COL == CFG_DQ576.DQ_COL and CFG_DQ576.DQ_COL == CFG.DK_COL + CFG.DK_COLS


def _dq_col(tmem_map_code: int) -> int:
    """The dQ image's column base of a TMEM map code (config = truth: the map's record, header section 9); a code outside the space is a KeyError at
    the trace (the kernel entry declines it by name first)."""
    return DQ_COL_BY_MAP[tmem_map_code]


# SMEM MAP (header sections 5 / 8 / 11; PART-KEYED): the typed adapter field ``smem_map`` reaches the kernel as the ``smem_map_code`` Constexpr[int]
# (config_sm100_d128_accum_2cta_stail.SMEM_MAP_CODES; resolved per device cc by the prepared module / compile_host).  SMEM_MAP_SM100 = the slab chain
# of section 8 as written (the 232448-B opt-in cap of every part, byte-identical); SMEM_MAP_RUBIN_B2 = the 327 KiB oversized carveout of cc 10.7 ON the
# dq_576 TMEM map (CFG_RING_DQ576_B2: sQ1 x2, sDO1 x2, sDS x2 + sDS_xchg x2 = the TWO-STAGE dS / exchange staging ring, SMEM_BYTES 314368): every ring
# depth below is the geometry's stage field (Q1_STAGES / DO1_STAGES / DS_STAGES / XCHG_STAGES; one PipelineState per ring and stage count), the
# epilogue stages dS(t) BEFORE it reads dQ(t-1) out (the LAGGING order of section 5) and the leader gates every dQ issue on the row dq_read_done.
SMEM_MAP_SM100 = SMEM_MAP_CODES["sm100"]
SMEM_MAP_RUBIN_B2 = SMEM_MAP_CODES["rubin_b2"]
# SMEM_MAP_SM100_OVERSIZED = the DIAGNOSTIC map (header section 11; cc 10.7 by name, never a default): the sm100 map itself -- every slab base, ring
# depth, row and address identical, the body byte-identical -- declared with SMEM_BYTES 262144 so the DSL launcher takes the oversized carveout
SMEM_MAP_SM100_OVERSIZED = SMEM_MAP_CODES["sm100_oversized"]
assert (SMEM_MAP_SM100, SMEM_MAP_RUBIN_B2, SMEM_MAP_SM100_OVERSIZED) == (0, 1, 2) and tuple(SMEM_MAP_CODES) == CFG.SMEM_MAPS == (
    "sm100",
    "rubin_b2",
    "sm100_oversized",
)
assert CFG_RING_DQ576_B2.SMEM_MAP == "rubin_b2" and CFG_RING_DQ576_B2.DS_STAGES == CFG_RING_DQ576_B2.XCHG_STAGES == 2 and CFG.DS_STAGES == CFG.XCHG_STAGES == 1
assert CFG.SMEM_MAP == "sm100" and CFG_RING.SMEM_MAP == CFG_DQ576.SMEM_MAP == CFG_RING_DQ576.SMEM_MAP == "sm100"


def _geometry(tma_ring: bool, dq576: bool, b2: bool = False, ovs: bool = False):
    """The geometry record of a (stats form, TMEM map, SMEM map) rendering (header sections 8 / 9): the SAME five frozen records ``cfg_for_part``
    resolves; the body reads every form- / map-dependent number through it (SMEM_BYTES, BARS, TMEM_HANDLE, SCHED_PAYLOAD, every slab base and ring
    depth; TMEM_COLS, TMEM_IS_EXCLUSIVE, TMEM_MAP; SMEM_MAP)."""
    return cfg_for_geometry("tma_ring" if tma_ring else "ldg", "dq_576" if dq576 else "s_tail", "rubin_b2" if b2 else ("sm100_oversized" if ovs else "sm100"))


def _slot(bar, state, stages: int):
    """The MBarrier slot a PipelineState names: the bar itself under ONE stage (the audited rendering's spelling, byte-identical), ``bar[state.idx]``
    under a ring (config = truth: ``stages`` is the geometry's depth; the state advances with the same count)."""
    return bar[state.idx] if stages > 1 else bar


def _do1_slot(bars, state, half: int, stages: int):
    """do1_full's slot of (stage, k-half): slot ``half`` under one stage (today's spelling), ``D_HALVES x stage + half`` under the sDO1 ring (ONE
    do1_state per role indexes the stage; the two half slots of a stage complete at one phase)."""
    return bars.do1_full[half] if stages == 1 else bars.do1_full[state.idx * CFG.D_HALVES + half]


class Bars(NamedTuple):
    """One MBarrier per ``BAR_LAYOUT`` row, in ``BAR_LAYOUT`` order (header section 5 = the table and the lane ledger; 34 rows) -- the ``16w`` bundle.

    Built by :func:`_bars` on BOTH CTAs with identical counts; a LEADER-scope field's follower copy is never armed / waited.  Under
    ``warp_roles = "16w_fa4"`` the bundle is :data:`BarsXFA4` (``BAR_LAYOUT_XFA4``: ds_tmem / dp_empty ABSENT -- an AttributeError if a path
    touches them, never a silent dead row)."""

    q1_full: MBarrier
    q1_empty: MBarrier
    q2_full: MBarrier
    q2_empty: MBarrier
    do1_full: MBarrier
    do1_empty: MBarrier
    do2_full: MBarrier
    do2_empty: MBarrier
    k1_full: MBarrier
    k1_empty: MBarrier
    v_full: MBarrier
    v_empty: MBarrier
    k2_full: MBarrier
    k2_empty: MBarrier
    s_done: MBarrier
    dp_done: MBarrier
    dq_done: MBarrier
    p_ready: MBarrier
    ds_ready: MBarrier
    s_read: MBarrier
    ds_tmem: MBarrier
    dp_empty: MBarrier
    s_tail_empty: MBarrier
    ds_smem: MBarrier
    ds_recv_full: MBarrier
    ds_pair_ready: MBarrier
    dq_empty: MBarrier
    dq_full: MBarrier
    dv_full: MBarrier
    dk_full: MBarrier
    dvk_empty: MBarrier
    sched_full: MBarrier
    sched_empty: MBarrier
    tmem_dealloc: MBarrier
    dvk_stage_empty: (
        MBarrier  # fp32_reduce (dkv_accum = fp32_reduce): store warp after round 1's fp32 reduce drain -> the epilogue's round 2; init only under dkv_fp32
    )


# BAR_LAYOUT kind -> (producer, scope).  "peer" rows are driven by arrive_on_peer / read_tile_id_arrive_on_leader (never .arrive()).
_KINDS = {
    "tma_leader": (Producer.TMA_LOAD, Scope.LEADER),
    "tma": (Producer.TMA_LOAD, Scope.LOCAL),
    "tma_dsmem": (Producer.TMA_LOAD, Scope.LOCAL),
    "mma": (Producer.MMA_COMMIT, Scope.LOCAL),
    "thread": (Producer.THREAD, Scope.LOCAL),
    "leader": (Producer.LEADER, Scope.LEADER),
    "peer": (Producer.THREAD, Scope.LEADER),
}
assert tuple(Bars._fields) == tuple(row[0] for row in BAR_LAYOUT), "Bars fields must follow BAR_LAYOUT (config = truth)"


def _dvk_stage_empty_row(bundle):
    """The index of the fp32_reduce row ``dvk_stage_empty`` in a bar bundle (class or instance) -- the ONE row whose init the P4 block guards by
    ``dkv_fp32`` (no user under the default rendering); -1 where the bundle has no such row (16w_fa4: the field declines there)."""
    return bundle._fields.index("dvk_stage_empty") if "dvk_stage_empty" in bundle._fields else -1


# 34 under the s_tail tables (BarsRing appends after it); 32 under the dq_576 tables (two rows fewer before it)
DVK_STAGE_EMPTY_ROW = _dvk_stage_empty_row(Bars)
# The 16w_fa4 bundle: the audit's 32 rows + dq_war_done (BAR_LAYOUT_XFA4) in order; the two deleted names have no field, no init, no arrive, no
# wait, no state; the fp32_reduce row dvk_stage_empty is UNSERVED there (fp32_reduce declines by name under 16w_fa4) and has no field either.
BarsXFA4 = NamedTuple("BarsXFA4", [(row[0], MBarrier) for row in BAR_LAYOUT_XFA4])
# The 16w x tma_ring bundle (header sections 5 / 11; the S1 skeleton): the 35 ldg rows re-based onto CFG_RING.BARS + stats_full[2] (TMA_LOAD /
# LOCAL, ONE_LANE: the loader's one expect_tx arm + ONE bulk copy per tile) + stats_empty[2] (THREAD / LOCAL, SOFTMAX_LANES: the 8 softmax warps bare
# after _ds_values); the row guard index of dvk_stage_empty is the same (the ring rows are appended after it).  The table's STATUS text stays in
# the config (the body carries no unused alias of it: the PR's hygiene rule).
BarsRing = NamedTuple("BarsRing", [(row[0], MBarrier) for row in BAR_LAYOUT_RING])
assert tuple(BarsRing._fields) == tuple(Bars._fields) + tuple(row[0] for row in STATS_RING_ROWS) == tuple(Bars._fields) + ("stats_full", "stats_empty")
assert BarsRing._fields.index("dvk_stage_empty") == DVK_STAGE_EMPTY_ROW
assert (
    tuple(BarsXFA4._fields)
    == tuple(row[0] for row in BAR_LAYOUT_XFA4)
    == tuple(n for n in Bars._fields if n not in DELETED_BARRIERS_XFA4 and n not in UNSERVED_BARRIERS_XFA4) + ADDED_BARRIERS_XFA4
)
# The dq_576 bundles (header sections 5 / 9 / 11; PART-KEYED, cc 10.7): the 16w tables minus DELETED_BARRIERS_DQ576 (s_read, s_tail_empty), re-packed --
# no field for either name (an AttributeError if a path touches them, never a silent dead row); the stats ring's rows follow under tma_ring.
BarsDQ576 = NamedTuple("BarsDQ576", [(row[0], MBarrier) for row in BAR_LAYOUT_DQ576])
BarsRingDQ576 = NamedTuple("BarsRingDQ576", [(row[0], MBarrier) for row in BAR_LAYOUT_RING_DQ576])
assert tuple(BarsDQ576._fields) == tuple(n for n in Bars._fields if n not in DELETED_BARRIERS_DQ576) and len(BarsDQ576._fields) == len(Bars._fields) - 2
assert tuple(BarsRingDQ576._fields) == tuple(BarsDQ576._fields) + tuple(row[0] for row in STATS_RING_ROWS)
assert _dvk_stage_empty_row(BarsDQ576) == _dvk_stage_empty_row(BarsRingDQ576) == DVK_STAGE_EMPTY_ROW - len(DELETED_BARRIERS_DQ576)
assert _dvk_stage_empty_row(BarsXFA4) == -1 and _dvk_stage_empty_row(BarsRing) == DVK_STAGE_EMPTY_ROW
# The rubin_b2 bundle (header sections 5 / 8 / 11; PART-KEYED, cc 10.7 ON dq_576 x tma_ring): the dq_576 ring table with the ring rows at their slots
# (q1_full / q1_empty / ds_smem / ds_recv_full / ds_pair_ready x 2, do1_full x 4, do1_empty x 2) + dq_read_done after dp_empty (the LEADER row of the
# lagging epilogue order); a path that touches dq_read_done on another bundle is an AttributeError, never a silent dead row.
BarsRingDQ576B2 = NamedTuple("BarsRingDQ576B2", [(row[0], MBarrier) for row in BAR_LAYOUT_RING_DQ576_B2])
_after = BarsRingDQ576._fields.index("dp_empty") + 1
assert tuple(BarsRingDQ576B2._fields) == BarsRingDQ576._fields[:_after] + ADDED_BARRIERS_B2 + BarsRingDQ576._fields[_after:] and ADDED_BARRIERS_B2 == (
    "dq_read_done",
)
assert _dvk_stage_empty_row(BarsRingDQ576B2) == _dvk_stage_empty_row(BarsRingDQ576) + len(ADDED_BARRIERS_B2)
assert {row[0]: row[2] for row in BAR_LAYOUT_RING_DQ576_B2}["do1_full"] == CFG.D_HALVES * CFG_RING_DQ576_B2.DO1_STAGES == 4
assert CFG.TMEM_MAP == "s_tail" and CFG_DQ576.TMEM_MAP == "dq_576", "this body renders the two TMEM maps of the config's code space under the fa4 issue order"
MCAST_PAIR = CFG.MCAST_PAIR_MASK  # tcgen05.commit multicast to both CTAs of the pair; the leader is cluster rank 0 under cluster (2,1,1)
LEADER_RANK = 0


class Dbg(NamedTuple):
    """STAMPS context of a role (header section 11; ``None`` = the lever is off = no traced code): ``words`` = the CTA's Int32 record in the
    bar block's spare (DBG_CLK.SMEM), ``rep`` = this warp is the role's representative (softmax warp 0, the epilogue lead warp; a single
    service warp passes True), so ONE elected lane per role accumulates into the role's own words (no atomics)."""

    words: object
    rep: object


class DbgLeader(NamedTuple):
    """The leader MMA warp's STAMPS context: the same record, plus the per-barrier BLOCKED counts (waits above DBG_CLK.BLOCKED_THRESH clk)
    DBG_CLK.LEADER_BLOCKED_STRIDE words past each wait sum (``_clk_wait_add`` tells the two contexts apart at trace time)."""

    words: object
    rep: object


_BUNDLES = {
    # (fa4_roles, tma_ring, dq576, b2) -> the NamedTuple of the table that rendering declares (the kernel entry declines the undefined combinations by name)
    (False, False, False, False): Bars,
    (True, False, False, False): BarsXFA4,
    (False, True, False, False): BarsRing,
    (True, True, False, False): None,  # 16w_fa4 x tma_ring: declined by name (validate_fields_for_roles + the trace-time raise)
    (False, False, True, False): BarsDQ576,
    (False, True, True, False): BarsRingDQ576,
    (False, True, True, True): BarsRingDQ576B2,  # rubin_b2: ON dq_576 x tma_ring x 16w only (every other b2 combination declines by name)
}


def _bars(slab, fa4_roles: bool = False, tma_ring: bool = False, dq576: bool = False, b2: bool = False):
    """The inspector's table as typed barriers: one MBarrier per row of the rendering's layout (``BAR_LAYOUT`` -> :class:`Bars` under ``16w``;
    ``BAR_LAYOUT_XFA4`` -> :data:`BarsXFA4` under ``16w_fa4``; ``BAR_LAYOUT_RING`` -> :data:`BarsRing` under ``16w x tma_ring``: the rows
    re-based onto CFG_RING.BARS + the stats ring's two rows; ``BAR_LAYOUT_DQ576`` / ``BAR_LAYOUT_RING_DQ576`` -> :data:`BarsDQ576` /
    :data:`BarsRingDQ576` under ``tmem_map = dq_576``: the s_read / s_tail_empty rows deleted, the rest re-packed; ``BAR_LAYOUT_RING_DQ576_B2`` ->
    :data:`BarsRingDQ576B2` under ``smem_map = rubin_b2``: the ring rows at their slots + dq_read_done), producer and scope from the row's kind.
    The bundle's fields and the layout's names are asserted equal at the trace (config = truth)."""
    bundle = _BUNDLES.get((fa4_roles, tma_ring, dq576, b2))
    if bundle is None:
        raise ValueError(
            f"no barrier bundle for warp_roles 16w_fa4 = {fa4_roles}, stats_form tma_ring = {tma_ring}, tmem_map dq_576 = {dq576}, smem_map rubin_b2 = {b2}"
        )
    layout = bar_layout_for_roles(
        "16w_fa4" if fa4_roles else "16w", "tma_ring" if tma_ring else "ldg", "dq_576" if dq576 else "s_tail", "rubin_b2" if b2 else "sm100"
    )
    assert tuple(bundle._fields) == tuple(row[0] for row in layout), "the bundle's fields must follow the rendering's table (config = truth)"
    return bundle(
        *(
            MBarrier(
                _view(slab, off, cutlass.Int64, slots),
                stages=slots,
                init_count=count,
                producer=_KINDS[kind][0],
                scope=_KINDS[kind][1],
                try_wait=True,
            )
            for _, off, slots, count, kind, _scope in layout
        )
    )


# ---------------------------------------------------------------------------------------------------------------------
# Barrier helpers (the guard forms of header section 5 as code; each role body below names the sites that call them).
# ---------------------------------------------------------------------------------------------------------------------


@cute.jit
def _clk_add(dbg, word: cutlass.Constexpr[int], v):
    """STAMPS: the role's representative warp's elected lane adds the Int32 ``v`` to record word ``word`` (its own words: no atomics)."""
    if dbg.rep:
        if nvvm.elect_sync():
            w = dbg.words.subview(word)
            w.store(w.load() + v)


@cute.jit
def _clk_wait_add(dbg, word: cutlass.Constexpr[int], dt):
    """STAMPS: ``dt`` (Int64 clk spent inside ONE mbarrier wait) into the wait sum ``word`` (u32 wrap); on the leader's context also the
    blocked count DBG_CLK.LEADER_BLOCKED_STRIDE words further when the wait took more than DBG_CLK.BLOCKED_THRESH clk."""
    if dbg.rep:
        if nvvm.elect_sync():
            w = dbg.words.subview(word)
            w.store(w.load() + cutlass.Int32(dt))
            if cutlass.const_expr(isinstance(dbg, DbgLeader)):
                if dt > cutlass.Int64(DBG_CLK.BLOCKED_THRESH):
                    b = dbg.words.subview(word + DBG_CLK.LEADER_BLOCKED_STRIDE)
                    b.store(b.load() + cutlass.Int32(1))


@cute.jit
def _wait(bar, phase, dbg, word: cutlass.Constexpr[int]):
    """The barrier's declared try_wait form + the warp reconvergence (the baseline's form).  ``dbg`` None = exactly that (the shipped
    statements); a STAMPS context brackets the UNMODIFIED wait with two %clock64 reads and accumulates the delta into ``word`` -- no
    barrier, no fence, no change to the wait."""
    if cutlass.const_expr(dbg is None):
        bar.wait(phase)
        # All full-warp consumers reconverge before any elected credit / issue (the baseline's form).
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    else:
        t0 = cute.arch.clock64()
        bar.wait(phase)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        _clk_wait_add(dbg, word, cute.arch.clock64() - t0)


@cute.jit
def _wait_tcgen(bar, phase, dbg, word: cutlass.Constexpr[int]):
    _wait(bar, phase, dbg, word)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _wait_store(bar, phase, dbg, word: cutlass.Constexpr[int]):
    """The STORE warp's mbarrier wait: the barrier's declared untimed parity-wait form = :func:`_wait` (the warp reconverges before the issue);
    ``dbg`` / ``word`` = the STAMPS bracket of _wait."""
    _wait(bar, phase, dbg, word)


@cute.jit
def _commit_pair(bar, elect_p):
    """MMA_COMMIT from the LEADER's MMA warp: ONE lane (pred) commits, the commit lands ONE arrive on EACH CTA of the pair (P13)."""
    bar.arrive(mcast_mask=cutlass.Int32(MCAST_PAIR), cta_group=CFG.CTA_MMA, pred=elect_p)


@cute.jit
def _arrive_leader(bar):
    """LEADER (relaxed): EVERY calling lane lands one arrive on the LEADER's copy (init = lanes x CTA_MMA); for TMEM data already
    ordered by tcgen05_wait + tcgen05_fence(BEFORE_THREAD_SYNC) or a count-only credit (P8).  Call it bare from every lane."""
    bar.arrive(cta_group=CFG.CTA_MMA, leader_cta_id=cutlass.Int32(LEADER_RANK))


@cute.jit
def _credit(bar):
    """sched_empty: ONE .release.cta arrive per calling warp on the LEADER's copy (init PAIR_SCHED_CREDIT_WARPS); at the TOP of the item."""
    read_tile_id_arrive_on_leader(bar.smem_ptr, CFG.CGA_M)


@cute.jit
def _drain(bar, state, stages: cutlass.Constexpr[int], dbg, word: cutlass.Constexpr[int]):
    """P15 / whole-ring drain from the CARRIED state: unused slots pass at their pre-armed parity, used slots wait their last release."""
    for _ in cutlass.range_constexpr(stages):
        _wait(bar[state.idx], state.phase, dbg, word)
        state = advance(state, stages)
    return state


@cute.jit
def _clk_ship(dbg, d):
    """STAMPS, the epilogue lead warp per tile: ``d`` = clk from the ds_tmem pass to the ship issue -> the sum, the DBG_CLK.HIST_EDGES
    histogram (bin 0 = the plan's "< 100 clk" line) and the max; then the ship stamp (clock lo32 + the CTA's tile count as the tag)
    the MMA warps read back after their ds_recv_full wait (:func:`_clk_recv`)."""
    if dbg.rep:
        if nvvm.elect_sync():
            s = dbg.words.subview(_W["seg_ship_sum"])
            s.store(s.load() + cutlass.Int32(d))
            if d < cutlass.Int64(DBG_CLK.HIST_EDGES[0]):
                h = dbg.words.subview(_W["seg_ship_hist0"])
                h.store(h.load() + cutlass.Int32(1))
            elif d < cutlass.Int64(DBG_CLK.HIST_EDGES[1]):
                h = dbg.words.subview(_W["seg_ship_hist1"])
                h.store(h.load() + cutlass.Int32(1))
            elif d < cutlass.Int64(DBG_CLK.HIST_EDGES[2]):
                h = dbg.words.subview(_W["seg_ship_hist2"])
                h.store(h.load() + cutlass.Int32(1))
            else:
                h = dbg.words.subview(_W["seg_ship_hist3"])
                h.store(h.load() + cutlass.Int32(1))
            m = dbg.words.subview(_W["seg_ship_max"])
            if cutlass.Int32(d) > m.load():
                m.store(cutlass.Int32(d))
            dbg.words.subview(_W["ship_clk"]).store(cutlass.Int32(cute.arch.clock64()))
            dbg.words.subview(_W["ship_tag"]).store(dbg.words.subview(_W["tiles"]).load())


@cute.jit
def _clk_recv(
    dbg,
    tiles_word: cutlass.Constexpr[int],
    sum_word: cutlass.Constexpr[int],
    matched_word: cutlass.Constexpr[int],
    missed_word: cutlass.Constexpr[int],
    tag_back: cutlass.Constexpr[int],
):
    """STAMPS, an MMA warp right after its ds_recv_full wait passed: the handshake with the epilogue lead's ship stamp (header section 11).
    The stamp's tag is the leading warp's tile count when it shipped; this warp's own count minus ``tag_back`` names the tile whose copy the wait
    consumed (DBG_CLK.TAG_BACK_*).  A match accumulates pass - ship (the SAME-CTA proxy of the exchange flight: our ship vs the peer's
    landing, both CTAs symmetric) and counts; a mismatch (the leading warp has not shipped this tile yet, or is a tile ahead) counts a miss."""
    if dbg.rep:
        if nvvm.elect_sync():
            expected = dbg.words.subview(tiles_word).load() - cutlass.Int32(tag_back)
            if dbg.words.subview(_W["ship_tag"]).load() == expected:
                s = dbg.words.subview(sum_word)
                s.store(s.load() + (cutlass.Int32(cute.arch.clock64()) - dbg.words.subview(_W["ship_clk"]).load()))
                m = dbg.words.subview(matched_word)
                m.store(m.load() + cutlass.Int32(1))
            else:
                x = dbg.words.subview(missed_word)
                x.store(x.load() + cutlass.Int32(1))


@cute.jit
def _clk_init(words, n_words: cutlass.Constexpr[int]):
    """STAMPS, warp 0's elected lane inside the P4 init block (published by the CTA sync that follows): zero the record (``n_words`` = the
    geometry's record size: DBG_CLK.WORDS under ldg, DBG_CLK_RING.WORDS under tma_ring), stamp the body start (clock lo32, %globaltimer lo32;
    the dump closes both spans -> clk / ns = the SM clock the body actually ran at)."""
    for w in cutlass.range_constexpr(n_words):
        words.subview(w).store(cutlass.Int32(0))
    words.subview(_W["body_clk"]).store(cutlass.Int32(cute.arch.clock64()))
    words.subview(_W["body_ns"]).store(cutlass.Int32(cute.arch.globaltimer()))


@cute.jit
def _clk_dump(words, dbg_clk, rank, n_words: cutlass.Constexpr[int]):
    """STAMPS, warp 0's elected lane after the final CTA sync (every role returned, every record word written): close the body clock and
    its %globaltimer span, write the magic and the rank, copy the record to dbg_clk[(linear CTA) x n_words ..] (plain stores; ``n_words`` =
    the geometry's record size)."""
    c = words.subview(_W["body_clk"])
    c.store(cutlass.Int32(cute.arch.clock64()) - c.load())
    n = words.subview(_W["body_ns"])
    n.store(cutlass.Int32(cute.arch.globaltimer()) - n.load())
    words.subview(_W["magic"]).store(cutlass.Int32(DBG_CLK.MAGIC))
    words.subview(_W["rank"]).store(rank)
    x, y, z = cute.arch.block_idx()
    nx, ny, nz = cute.arch.grid_dim()
    base = (x + nx * (y + ny * z)) * cutlass.Int32(n_words)
    out = cutlass.make_array_view(dbg_clk)
    for w in cutlass.range_constexpr(n_words):
        out[base + cutlass.Int32(w)] = words.subview(w).load()


@cute.jit
def _skew_leader_ship(rank):
    """Test-only skew lever 2 (LEADER_SHIP): the LEADER holds its XCHG ship -- the half is written, fenced and the XCHG named barrier passed.  ONE site
    for both role shapes (16w: the epilogue; 16w_fa4: the softmax, right before the elected arm + ship); default 0 = no traced code."""
    if cutlass.const_expr(DEBUG_LEADER_SHIP_DELAY_US > 0):
        if rank == cutlass.Int32(LEADER_RANK):
            for _us in cutlass.range(DEBUG_LEADER_SHIP_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])


@cute.jit
def _skew_follower_own_half(rank):
    """Test-only skew lever 1 (FOLLOWER_OWN_HALF): the FOLLOWER holds its own-half dS STS (16w: between the ship and the own STS -- its XCHG is shipped;
    16w_fa4: right after the softmax's dq_war wait, before its 8 STS -- its ds_smem AND its ship are held together).  ONE site for both shapes."""
    if cutlass.const_expr(DEBUG_FOLLOWER_OWN_HALF_DELAY_US > 0):
        if rank != cutlass.Int32(LEADER_RANK):
            for _us in cutlass.range(DEBUG_FOLLOWER_OWN_HALF_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])


@cute.jit
def _skew_follower_dq_readout(rank):
    """Test-only skew lever 6 (EPILOGUE_DQ_READOUT): the FOLLOWER's warps 8-11 hold their dQ readout between the dq_done wait and the dQ ld (s_tail_empty
    alone holds S^T(t+1) / the tail dQ / the next prologue).  ONE site for both shapes (the epilogue under 16w, the reduce group under 16w_fa4)."""
    if cutlass.const_expr(DEBUG_EPILOGUE_DQ_READOUT_DELAY_US > 0):
        if rank != cutlass.Int32(LEADER_RANK):
            for _us in cutlass.range(DEBUG_EPILOGUE_DQ_READOUT_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])


@cute.jit
def _skew_follower_item_end(rank):
    """Test-only skew lever 3 (FOLLOWER_EPILOGUE_ITEM_END): the FOLLOWER's warps 8-11 hold their dV / dK TMEM readout at the item end (dvk_empty's
    pair-wide 64).  ONE site for both shapes."""
    if cutlass.const_expr(DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US > 0):
        if rank != cutlass.Int32(LEADER_RANK):
            for _us in cutlass.range(DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])


@cute.jit
def _skew_leader_arm(rank):
    """Test-only skew lever 8 (LEADER_ARM_DELAY; the rubin_b2 ring's instrument): the LEADER's epilogue warps hold their ds_recv_full[s] arm -- the
    follower's ship of slot s lands first (the transiently negative tx-count path); the phase completes at the arm.  ONE site (the 16w epilogue;
    16w_fa4 declines rubin_b2 by name; under sm100 the same order on the single row).  SKEW_OK expected; default 0 = no traced code."""
    if cutlass.const_expr(DEBUG_LEADER_ARM_DELAY_US > 0):
        if rank == cutlass.Int32(LEADER_RANK):
            for _us in cutlass.range(DEBUG_LEADER_ARM_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])


@cute.jit
def _skew_leader_slot1():
    """Test-only skew lever 9 (LEADER_SLOT1_DELAY; the rubin_b2 ring's instrument): the LEADER's MMA warp holds slot 1 of every tile before its first
    wait (leader-only by construction) -- the follower's relay(t) and both epilogues' staging of tile t land before the leader consumed flip t-1 of the
    2-slot ds_* rings at slot 2 (the depth-2 ABA proof; a 1-slot form would hang).  ONE site.  SKEW_OK expected; default 0 = no traced code."""
    if cutlass.const_expr(DEBUG_LEADER_SLOT1_DELAY_US > 0):
        for _us in cutlass.range(DEBUG_LEADER_SLOT1_DELAY_US, unroll=1):
            nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])


@cute.jit
def _ship_xchg(slab, bars, rank, ship_pred, stage, geo: cutlass.Constexpr, b2: cutlass.Constexpr[bool]):
    """Epilogue, after named barrier CFG.XCHG_BARRIER (all 4 warps' XCHG stores + fence_proxy done): ONE lane (pred=is_lead_warp & elect)
    ships OUR XCHG (dS^T[our 128 kv x the peer's 64 q]) into the PEER's recv half -- the peer's bytes at OUR ds_own offset (kv-global row
    order) -- completing XCHG_TX on the PEER's ds_recv_full (mapa BOTH the pointer and the mbar; the d512 role-split's _xfer form).
    ``b2`` (smem_map rubin_b2, header sections 5 / 8): ring stage ``stage`` (= the epilogue's sds_state.idx, a traced Int32) of OUR XCHG into stage
    ``stage`` of the PEER's sDS at its recv half (= our own-half offset of that stage), completing on the PEER's ``ds_recv_full[stage]`` (P9: a plain
    bulk copy is local to the destination CTA's mbar, never routed); under sm100 ``stage`` is None and the single slabs / slot are the audited ones."""
    peer = rank ^ cutlass.Int32(1)
    if cutlass.const_expr(b2):
        local_src = _view(slab, geo.XCHG, cutlass.Uint8, geo.XCHG_STAGES * geo.XCHG_BYTES, alignment=1024).subview(stage * cutlass.Int32(geo.XCHG_BYTES))
        peer_dst = nvvm.mapa(
            _view(slab, geo.DS, cutlass.Uint8, geo.DS_STAGES * geo.DS_BYTES, alignment=1024).subview(
                stage * cutlass.Int32(geo.DS_BYTES) + rank * cutlass.Int32(geo.DS_HALF_BYTES)
            ),
            peer,
            addrspace=7,
        )
        peer_mbar = nvvm.mapa(bars.ds_recv_full[stage].smem_ptr, peer, addrspace=7)
    else:
        local_src = _view(slab, geo.XCHG, cutlass.Uint8, geo.XCHG_BYTES, alignment=1024)
        peer_dst = nvvm.mapa(
            _view(slab, geo.DS, cutlass.Uint8, geo.DS_BYTES, alignment=1024).subview(rank * cutlass.Int32(geo.DS_HALF_BYTES)), peer, addrspace=7
        )
        peer_mbar = nvvm.mapa(bars.ds_recv_full.smem_ptr, peer, addrspace=7)
    cp_async_bulk_shared_cluster_shared_cta(peer_dst, local_src, peer_mbar, geo.XCHG_TX, pred=ship_pred)


@cute.jit
def _tmem_alloc_pair(tmem_slot, tmem_cols: cutlass.Constexpr[int], exclusive: cutlass.Constexpr[bool]):
    """MMA warp of EACH CTA (warp-collective, all 32 lanes; after the cluster sync, before any TMEM touch): one tcgen05.alloc.cta_group::2 of
    ``tmem_cols`` = the map's ``geo.TMEM_COLS`` (512 on the 512-column parts; the whole 576-column carve with ``is_exclusive`` on cc 10.7 under
    tmem_map dq_576 -- the SKU's alloc form, header section 9); the SAME constant deallocates (:func:`_tmem_dealloc_pair`)."""
    tmem_alloc(tmem_slot, tmem_cols, nvvm.CTAGroup.CTA_2, is_exclusive=exclusive)


@cute.jit
def _tmem_dealloc_pair(bars, tmem_slot, rank, tmem_cols: cutlass.Constexpr[int]):
    """MMA warp of EACH CTA after the final barrier_cta_sync (every local TMEM reader done): one elected arrive on our tmem_dealloc and one
    on the peer's (init TMEM_DEALLOC_ARRIVERS = 2), wait(0), tcgen05 fence, tcgen05.dealloc.cta_group::2 of ``tmem_cols`` = the SAME
    ``geo.TMEM_COLS`` the alloc took -- the dealloc frees the PAIR's columns, so neither CTA frees them while the other's warps may still read;
    the wait is the drain of the peer's arrive."""
    if nvvm.elect_sync():
        bars.tmem_dealloc.arrive()
        bars.tmem_dealloc.arrive_on_peer(rank ^ cutlass.Int32(1))
    bars.tmem_dealloc.wait(cutlass.Int32(0))
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    tmem_dealloc(tmem_slot, tmem_cols, nvvm.CTAGroup.CTA_2)


# ---------------------------------------------------------------------------------------------------------------------
# Descriptor declarations.  Every SmemTile is SW128; rows are one atom (128 B) wide per TMA box.
# cta_group::2 loads complete on the PAIR LEADER's mbar: leader arms 2x bytes, follower never.
# ---------------------------------------------------------------------------------------------------------------------


def _operand_k_major(slab, offset, dtype, rows):
    """A / B operand with d along K (Q1, dO1, K1, V): ``rows`` x 128 d as two [rows x HEAD_BOX] SW128 boxes."""
    return SmemTile(
        base=_view(slab, offset, dtype, rows * CFG.D, alignment=1024),
        elems_per_stage=rows * CFG.D,
        leading_byte_offset=16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=CFG.D // CFG.HEAD_BOX,
        tma_granu_elems=CFG.HEAD_BOX,
        tma_subtile_stride_elems=rows * CFG.HEAD_BOX,
        desc_version=CFG.DESC_VERSION,
    )


def _operand_mn_major(slab, offset, dtype, rows):
    """B operand with d along N (Q2, dO2: rows = 128 q; K2: rows = 256 kv): one [rows x D_HALF] SW128 box, MN-major."""
    return SmemTile(
        base=_view(slab, offset, dtype, rows * CFG.D_HALF, alignment=1024),
        elems_per_stage=rows * CFG.D_HALF,
        leading_byte_offset=rows * CFG.SWIZZLE_BYTES,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.D_HALF,
        tma_subtile_stride_elems=rows * CFG.D_HALF,
        desc_version=CFG.DESC_VERSION,
    )


def _store_half(slab, offset, dtype):
    """One native-store box [M kv x HEAD_BOX d] (a dK staging half at a rank-dependent alias offset)."""
    return SmemTile(
        base=_view(slab, offset, dtype, CFG.M * CFG.HEAD_BOX, alignment=1024),
        elems_per_stage=CFG.M * CFG.HEAD_BOX,
        leading_byte_offset=16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.HEAD_BOX,
        tma_subtile_stride_elems=CFG.M * CFG.HEAD_BOX,
        desc_version=CFG.DESC_VERSION,
    )


def _ds_operand(slab, base, dtype):
    """dQ's A operand: dS^T [PAIR_KV_ROWS kv rows x Q_HALF q] MN-major (q contiguous: one 64-q SW128 atom per CTA, 128 B per kv
    row, 8-row swizzle period) -- the layout `_store_words` writes; the MN-major k-step walks 16 kv rows = 2 KiB.  ``base`` = the geometry's sDS
    stage-0 root (geo.DS); under the two-stage ring the leader advances the pinned descriptor by (stage x DS_BYTES) >> 4 per issue (one root)."""
    return SmemTile(
        base=_view(slab, base, dtype, CFG.Q_HALF * CFG.PAIR_KV_ROWS, alignment=1024),
        elems_per_stage=CFG.Q_HALF * CFG.PAIR_KV_ROWS,
        leading_byte_offset=CFG.DS_A_LBO,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.Q_HALF,
        tma_subtile_stride_elems=CFG.Q_HALF * CFG.PAIR_KV_ROWS,
        desc_version=CFG.DESC_VERSION,
    )


def _dq_slot_tile(slab, geo, slot):
    """fp32 reduce-add staging slot ``slot`` of the DEDICATED sDQ ring ([Q_HALF x DQ_BOX] SW128; header section 8): one source box per reduce."""
    return SmemTile(
        base=_view(slab, geo.dq_slot(slot), cutlass.Float32, CFG.Q_HALF * CFG.DQ_BOX, alignment=1024),
        elems_per_stage=CFG.Q_HALF * CFG.DQ_BOX,
        leading_byte_offset=16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.DQ_BOX,
        tma_subtile_stride_elems=CFG.Q_HALF * CFG.DQ_BOX,
        desc_version=CFG.DESC_VERSION,
    )


def _dkv_acc_tile(slab, base):
    """One fp32 [M kv x DQ_BOX d] SW128 reduce-add source slice at byte ``base`` (CFG.dkv_acc_slot(rank, i); dkv_accum = fp32_reduce): the
    sDQ slot form over 128 rows."""
    # fp32 box [128 kv x 32 d] = 16 KiB SW128 = _dq_slot_tile with M rows; same STS form
    return SmemTile(
        base=_view(slab, base, cutlass.Float32, CFG.M * CFG.DQ_BOX, alignment=1024),
        elems_per_stage=CFG.M * CFG.DQ_BOX,
        leading_byte_offset=16,
        stride_byte_offset=8 * CFG.SWIZZLE_BYTES,
        layout=CFG.SWIZZLE_LAYOUT,
        tma_loads_per_tile=1,
        tma_granu_elems=CFG.DQ_BOX,
        tma_subtile_stride_elems=CFG.M * CFG.DQ_BOX,
        desc_version=CFG.DESC_VERSION,
    )


def _mma_desc_kv(dtype, *, atranspose=False, btranspose=False):
    """S^T / dP^T (SS) and dV / dK (TS) : collective M = 256 (own 128 kv rows per CTA), N = 128, K = 128."""
    return MmaDesc(
        M=CFG.COLLECTIVE_M_KV,
        N=CFG.N,
        K=CFG.D,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.K_HW,
        atranspose=atranspose,
        btranspose=btranspose,
        cta_group=CFG.CTA_MMA,
        idesc=nvvm.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=dtype,
            b_dtype=dtype,
            m_dim=CFG.COLLECTIVE_M_KV,
            n_dim=CFG.N,
            a_major=int(atranspose),
            b_major=int(btranspose),
        ),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


def _mma_desc_dq(dtype):
    """dQ = dS K (SS): collective M = 128 (64 q rows per CTA), N = 128 d, K = the pair's 256 kv rows; A and B MN-major."""
    return MmaDesc(
        M=CFG.COLLECTIVE_M_Q,
        N=CFG.D,
        K=CFG.PAIR_KV_ROWS,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.K_HW,
        atranspose=True,
        btranspose=True,
        cta_group=CFG.CTA_MMA,
        idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, a_dtype=dtype, b_dtype=dtype, m_dim=CFG.COLLECTIVE_M_Q, n_dim=CFG.D, a_major=1, b_major=1),
        kind=nvvm.Tcgen05MMAKind.F16,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Body helpers (the first body's forms onto the s_tail map; the S^T ld pair is split out of _p_values for the s_read arrive).
# ---------------------------------------------------------------------------------------------------------------------


@cute.jit
def _store_words(slab, byte_offset, row, words):
    """One SW128 atom row (32 Int32 = 128 B) of ``row`` at ``byte_offset``: today's dS^T / dV / dK staging form (16-B vectors, XOR swizzle)."""
    for vec in cutlass.range_constexpr(8):
        off = byte_offset + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        values = cutlass.Vector.from_elements(tuple(words[vec * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _store_words_quarter(slab, row_base, words, first: cutlass.Constexpr[int], xor_lane):
    """16w_fa4 (the alternate role shape, the resource plan): ONE 32-q quarter of the lane's dS^T row -- the CFG.PACK_QUARTER_COLS = 16 packed Int32 words
    ``words[first, first + 16)`` -- into the SW128 slab row at ``row_base`` (= the own-half or XCHG base + row * SWIZZLE_BYTES) as
    CFG.DS_STS_VECS_PER_QUARTER = 4 16-B vectors at vec ``vec0 + v`` of the row, ``xor_lane = (row % 8) ^ vec0`` (``vec0`` = 4 x the softmax group:
    group g owns vec 4g .. 4g+3 of the 64-q half; ``vec0 + v == vec0 | v == vec0 ^ v`` for v < 4): the same ``(row % 8) ^ vec`` XOR of
    :func:`_store_words`, so the softmax writes today's epilogue LDTM image word for word (dK / dV bitwise, dQ's A descriptor unchanged);
    conflict-free: 4 wavefronts per STS.128 = the floor (the resource plan, the audit record).  The caller passes
    ``xor_lane`` through ``opaque_i32(.., anchor=qtile)`` so the 8 address adds live inside the tile (the d256 2x2 lesson: hoisted, they spilled)."""
    for v in cutlass.range_constexpr(CFG.DS_STS_VECS_PER_QUARTER):
        off = row_base + (xor_lane ^ v) * 16
        values = cutlass.Vector.from_elements(tuple(words[first + v * 4 + k] for k in range(4)), cutlass.Int32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Int32).store(values, alignment=16)


@cute.jit
def _store_dq_slice(slab, byte_base, row, values):
    """fp32 slice row: the lane's DQ_BOX values of q row ``row`` into the SW128 [Q_HALF x DQ_BOX] slot at ``byte_base`` (one ROUND's half of
    the 64-column dQ readout; the same 16-B vector / XOR-swizzle form as _store_words)."""
    for vec in cutlass.range_constexpr(CFG.DQ_BOX // 4):
        off = byte_base + row * CFG.SWIZZLE_BYTES + ((row % 8) ^ vec) * 16
        v = cutlass.Vector.from_elements(tuple(values[vec * 4 + k] for k in range(4)), cutlass.Float32)
        Pointer(slab.data_ptr() + off, dtype=cutlass.Float32).store(v, alignment=16)


@cute.jit
def _dq_half(values, first: cutlass.Constexpr[int]):
    """Columns [first, first + DQ_BOX) of the lane's 64-column dQ readout as one vector: round A = the first DQ_BOX (first 0), round B = the
    second (first DQ_BOX), CARRIED in registers across the next tile's dS staging (header section 4: ~96 live of 144)."""
    return cutlass.Vector.from_elements(tuple(values[first + i] for i in range(CFG.DQ_BOX)), cutlass.Float32)


@cute.jit
def _dq_zero_half():
    """The carried half before the item's first readout (never stored: round B runs only after a readout)."""
    return cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG.DQ_BOX)), cutlass.Float32)


@cute.jit
def _concat32(a, b):
    """Two 16-word tcgen05.ld results -> one 32-word row (q ascending across the two quarters)."""
    return cutlass.Vector.from_elements(tuple(a[i] for i in range(CFG.PACK_QUARTER_COLS)) + tuple(b[i] for i in range(CFG.PACK_QUARTER_COLS)), cutlass.Int32)


@cute.jit
def _next_tile(sched, bars, state, dbg, word: cutlass.Constexpr[int]):
    """The pair's next CLC payload: x = the first ctaid.x of the cancelled cluster (a multiple of CGA_M) -> pair = x // CGA_M."""
    _wait(bars.sched_full[state.idx], state.phase, dbg, word)
    x, yz, valid = read_clc_payload(sched, state.idx * 8, warp_broadcast=True)
    return x // CFG.CGA_M, yz & 0xFFFF, (yz >> 16) & 0xFFFF, valid, advance(state, CFG.SCHED_STAGES)


@cute.jit
def _item_bounds(pair, meta, batch, n_batch, causal: cutlass.Constexpr[bool]):
    """Section 7: cluster-union (q_lo, q_hi, seq_q, seq_k, q_start) for the pair's 256-row kv block; pure, no SMEM.

    # BARRIER contract: both CTAs derive IDENTICAL (lo, hi) from the same payload words, so every barrier flips the same number of
    # times on both CTAs (P14) -- including the leader's pair-uniform `qtile > lo` / `hi - 1 > lo` guards on the s_tail_empty waits and
    # the dQ issue sites; a dead owner (pair * 256 >= seq_k) is lo = hi = 0 on both; a padded SECOND half is NOT a skip.
    """
    data = cutlass.make_array_view(meta)
    q_start = cutlass.Int32(data[n_batch + batch])
    seq_q = cutlass.Int32(data[n_batch + batch + 1]) - q_start
    seq_k = cutlass.Int32(data[batch])
    lo = pair * cutlass.Int32(CFG.Q_TILES_PER_PAIR) if cutlass.const_expr(causal) else cutlass.Int32(0)
    hi = (seq_q + CFG.N - 1) // CFG.N
    if pair * CFG.PAIR_KV_ROWS >= seq_k:
        lo = cutlass.Int32(0)
        hi = cutlass.Int32(0)
    return lo, hi, seq_q, seq_k, q_start


def tile_may_mask(pair, rank, qtile, seq_q, seq_k, causal: bool):
    """Section 7 (the mask-form lever): TRUE iff a cell of CTA ``rank``'s [M kv x N q] tile ``qtile`` of pair ``pair`` can be masked by _p_values' band --
    the causal straddle (qtile * N < kv_end), the last partial q tile ((qtile + 1) * N > seq_q), kv rows at or past seq_k (kv_end > seq_k) --
    with kv_end = pair * PAIR_KV_ROWS + (rank + 1) * M.  Plain Python over the traced scalars (CTA-uniform: no lane term) AND over host ints
    (the test module's exactness row runs this very function against the per-cell mask).  EXACT (section 7): FALSE means every column of the
    tile keeps for every lane, so the fsel form's selects are all pass-through there and skipping them is bitwise."""
    kv_end = pair * CFG.PAIR_KV_ROWS + (rank + 1) * CFG.M
    edge = ((qtile + 1) * CFG.N > seq_q) | (kv_end > seq_k)
    if causal:
        edge = edge | (qtile * CFG.N < kv_end)
    return edge


@cute.jit
def _stats(slab, lse, delta, row, group, qbase, batch, head, scale, q_start, seq_q, origins, stats_origin: cutlass.Constexpr[int], geo: cutlass.Constexpr):
    # Full 256-lane named barriers also cover inactive tail rows (both CTAs hold all 128 q stats). Invalid rows
    # never dereference a Stats/Delta address, including an all-empty carrier.
    named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES)
    local_row = qbase + row
    if group == 0:
        stat = cutlass.Float32(0.0)
        if local_row < seq_q:
            token = thd_port_origin(origins, stats_origin, batch, q_start) + cutlass.Int64(local_row)
            if cutlass.const_expr(len(lse.shape) == 2):
                val = cutlass.Float32(lse[token, head])
            else:
                val = cutlass.Float32(lse[0, head, token])
            stat = -cutlass.Float32(LOG2E) * val
            if val == cutlass.Float32(float("-inf")):
                stat = cutlass.Float32(0.0)
        _view(slab, geo.STATS, cutlass.Float32, CFG.N).subview(row).store(stat)
    else:
        stat = cutlass.Float32(0.0)
        if local_row < seq_q:
            stat = -scale * cutlass.Float32(delta[0, head, cutlass.Int64(q_start) + local_row])
        _view(slab, geo.DELTA, cutlass.Float32, CFG.N).subview(row).store(stat)
    named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES)


@cute.jit
def _s_values(tmem_base, group):
    """The S^T ld pair of the lane's kv row over softmax group ``group``'s two q quarters (a = quarter g, b = quarter g + 2 = the [64, 128)
    half that is also the dQ image's home), completed by tcgen05_wait(LOAD).  The caller fences and arrives s_read BEFORE the exp body:
    once these registers hold S^T(t), dQ(t-1) may land in [64, 128)."""
    q_lo = group * CFG.GROUP_QUARTER
    a = _tmem_load(tmem_base + CFG.S_COL + q_lo, CFG.GROUP_QUARTER)
    b = _tmem_load(tmem_base + CFG.S_COL + q_lo + CFG.N // 2, CFG.GROUP_QUARTER)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    return a, b


@cute.jit
def _p_values(
    slab,
    stats_base,
    tmem_base,
    group,
    key,
    qtile,
    seq_q,
    seq_k,
    scale,
    a,
    b,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    one_chunk: cutlass.Constexpr[bool],
    mask_form_code: cutlass.Constexpr[int],
    mask_tile,
):
    """S^T (registers ``a``, ``b`` from _s_values) -> P^T for the lane's kv row ``key`` over softmax group ``group``'s two q quarters; the
    packed bf16 P^T goes in place into the group's PACK_GROUP_COLS columns of [0, 64).  Returns the 64 fp32 P values in packed order.
    ``one_chunk`` (DIAGNOSTIC stats_one_chunk `stats_one_chunk`): ONE 16-B LSE chunk per quarter (CFG.DIAG_STATS_CHUNK) is loaded and reused for all
    CFG.DIAG_STATS_CHUNKS_PER_QUARTER chunks -- the same registers, exp2 and FFMA count, 7 of 8 LDS.128 gone; the stats are WRONG BY DESIGN.
    ``mask_form_code`` (header section 11): MASK_FORM_FSEL = the keep-word band mask selected on EVERY tile (today's rendering);
    MASK_FORM_PREDICATED = the SAME mask body under the CTA-uniform runtime branch ``if mask_tile:`` (``mask_tile`` = tile_may_mask of section 7,
    read under this form only: FALSE on an interior tile = no cell of this CTA's tile can be masked, so the exp2 values are packed straight);
    MASK_FORM_DIAG_OFF = no mask on any tile (the DIAGNOSTIC ceiling: OUTPUT WRONG BY DESIGN).
    ``stats_base`` = the SMEM byte base of the tile's 128 LSE stat words (-log2(e) * LSE; 0 for a padded / masked row): ``CFG.STATS`` under
    stats_form ldg (the single slab `_stats` fills), the ring stage's LSE row under tma_ring (header section 8); read by the same 16-B
    broadcast LDS form either way."""
    q_lo = group * CFG.GROUP_QUARTER
    scaled = scale * cutlass.Float32(LOG2E)
    # KV-major transpose of the shared mask band: Q columns >= key for causal, below the actual Q length, and entirely
    # dead for a padded KV row (key >= seq_k). Select exact zero AFTER exp2 (also safe for a no-key row whose LSE is -inf).
    lo = key if cutlass.const_expr(causal) else None
    if cutlass.const_expr(mask_form_code == MASK_FORM_FSEL):
        hi = cutlass.Int32(arith.select((key < seq_k).ir_value(), seq_q.ir_value(), cutlass.Int32(0).ir_value()))
    p_all = []
    packed = []
    for half in cutlass.range_constexpr(CFG.QUARTERS_PER_GROUP):
        scores = (a, b)[half]
        col0 = q_lo + half * (CFG.N // 2)
        values = []
        if cutlass.const_expr(one_chunk):  # DIAGNOSTIC stats_one_chunk: the one chunk every column of the quarter reuses
            stat_one = _view(slab, stats_base, cutlass.Float32, CFG.N).subview(col0 + CFG.DIAG_STATS_CHUNK * 4).load(vector_size=4, alignment=16)
        for chunk in cutlass.range_constexpr(CFG.DIAG_STATS_CHUNKS_PER_QUARTER):
            if cutlass.const_expr(one_chunk):
                stat = stat_one
            else:
                stat = _view(slab, stats_base, cutlass.Float32, CFG.N).subview(col0 + chunk * 4).load(vector_size=4, alignment=16)
            for pair in cutlass.range_constexpr(2):
                i = chunk * 4 + pair * 2
                e0, e1 = ffma2(scaled, scaled, scores[i], scores[i + 1], stat[pair * 2], stat[pair * 2 + 1])
                values.extend((cute.math.exp2(e0, fastmath=True), cute.math.exp2(e1, fastmath=True)))
        if cutlass.const_expr(mask_form_code == MASK_FORM_FSEL):
            words = band_mask_words(lo, hi, qtile * CFG.N + col0, CFG.GROUP_QUARTER)
            p = apply_mask_words(cutlass.Vector.from_elements(tuple(values), cutlass.Float32), words, mask_value=0.0, n_cols=CFG.GROUP_QUARTER)
        else:
            p = cutlass.Vector.from_elements(tuple(values), cutlass.Float32)
            if cutlass.const_expr(mask_form_code == MASK_FORM_PREDICATED):
                if mask_tile:  # CTA-uniform (section 7): the mask body only on a tile that can hold a masked cell; `hi` and the words live in here
                    hi_m = cutlass.Int32(arith.select((key < seq_k).ir_value(), seq_q.ir_value(), cutlass.Int32(0).ir_value()))
                    words = band_mask_words(lo, hi_m, qtile * CFG.N + col0, CFG.GROUP_QUARTER)
                    p = apply_mask_words(p, words, mask_value=0.0, n_cols=CFG.GROUP_QUARTER)
        p_all.extend(p[i] for i in range(CFG.GROUP_QUARTER))
        packed.extend(fp32_to_fp16(p[2 * i], p[2 * i + 1], dtype=dtype) for i in range(CFG.GROUP_QUARTER // 2))
    _tmem_store(tmem_base + CFG.P_COL + group * CFG.PACK_GROUP_COLS, cutlass.Vector.from_elements(tuple(packed), cutlass.Int32))
    return cutlass.Vector.from_elements(tuple(p_all), cutlass.Float32)


@cute.jit
def _ds_values(slab, delta_base, tmem_base, group, p, scale, dtype: cutlass.Constexpr, one_chunk: cutlass.Constexpr[bool]):
    """dP^T -> dS^T = P^T * (dP^T - delta) * scale for the group's two quarters; packed bf16 in place into the group's columns.  Returns the 32
    packed Int32 words (quarter g = words [0, 16), quarter g + 2 = words [16, 32), q ascending inside each): under 16w_fa4 the softmax stores
    them to sDS own + XCHG itself (:func:`_store_words_quarter`); under 16w the value is unused (the epilogue reads the same words back by LDTM).
    ``one_chunk`` (DIAGNOSTIC stats_one_chunk): ONE delta chunk per quarter reused for all eight, as in :func:`_p_values`.
    ``delta_base`` = the SMEM byte base of the tile's 128 delta stat words (-scale * delta; 0 for a padded row): ``CFG.DELTA`` under stats_form
    ldg, the ring stage's delta row under tma_ring (header section 8)."""
    q_lo = group * CFG.GROUP_QUARTER
    a = _tmem_load(tmem_base + CFG.DP_COL + q_lo, CFG.GROUP_QUARTER)
    b = _tmem_load(tmem_base + CFG.DP_COL + q_lo + CFG.N // 2, CFG.GROUP_QUARTER)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    packed = []
    for half in cutlass.range_constexpr(CFG.QUARTERS_PER_GROUP):
        dp = (a, b)[half]
        col0 = q_lo + half * (CFG.N // 2)
        if cutlass.const_expr(one_chunk):  # DIAGNOSTIC stats_one_chunk: the one chunk every column of the quarter reuses
            stat_one = _view(slab, delta_base, cutlass.Float32, CFG.N).subview(col0 + CFG.DIAG_STATS_CHUNK * 4).load(vector_size=4, alignment=16)
        for chunk in cutlass.range_constexpr(CFG.DIAG_STATS_CHUNKS_PER_QUARTER):
            if cutlass.const_expr(one_chunk):
                stat = stat_one
            else:
                stat = _view(slab, delta_base, cutlass.Float32, CFG.N).subview(col0 + chunk * 4).load(vector_size=4, alignment=16)
            for pair in cutlass.range_constexpr(2):
                i = chunk * 4 + pair * 2
                d0, d1 = ffma2(scale, scale, dp[i], dp[i + 1], stat[pair * 2], stat[pair * 2 + 1])
                d0, d1 = fmul2(d0, d1, p[half * CFG.GROUP_QUARTER + i], p[half * CFG.GROUP_QUARTER + i + 1])
                packed.append(fp32_to_fp16(d0, d1, dtype=dtype))
    words = cutlass.Vector.from_elements(tuple(packed), cutlass.Int32)
    _tmem_store(tmem_base + CFG.DS_COL + group * CFG.PACK_GROUP_COLS, words)
    return words


@cute.jit
def _native_out(slab, tmem_base, row, live, col, bases, dtype: cutlass.Constexpr):
    """dV / dK native staging: two [128 kv x 64 d] bf16 halves of the lane's kv row from TMEM columns ``col`` (zeros when the item ran
    no tile) into the two alias bases; one proxy fence + tcgen05 fence for both halves (the caller arrives)."""
    for half in cutlass.range_constexpr(CFG.D_HALVES):
        vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG.D_HALF)), cutlass.Float32)
        if live:
            vals = _tmem_load(tmem_base + col + half * CFG.D_HALF, CFG.D_HALF)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=dtype) for i in range(CFG.D_HALF // 2)), cutlass.Int32)
        _store_words(slab, bases[half], row, words)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def _stage_fp32_slice(slab, tmem_base, row, live, col, index: cutlass.Constexpr[int], base):
    """dkv_accum = fp32_reduce: fp32 slice ``index`` (DQ_BOX columns) of the accumulator plane at TMEM column ``col``, the lane's kv row -> the
    [M x DQ_BOX] fp32 SW128 slot at byte ``base`` (the `_store_dq_slice` form over 128 rows).  Zeros when the item ran no tile (the store warp
    skips a dead CTA's reduce-adds; the workspace is zero-filled).  The caller fences and arrives once per round."""
    vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(CFG.DQ_BOX)), cutlass.Float32)
    if live:
        vals = _tmem_load(tmem_base + col + index * CFG.DQ_BOX, CFG.DQ_BOX)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    _store_dq_slice(slab, base, row, vals)


@cute.jit
def _dkv_fp32_stage(slab, bars, geo: cutlass.Constexpr, tmem_base, lane, live, stage_state, dbg):
    """dkv_accum = fp32_reduce, the epilogue's item end (header section 11; the resource plan): ROUND 1 = dV's four fp32
    slices + dK's slice 0 over the five slots CFG.dkv_acc_r1_slot(0..4) (sK2's two halves, the dS slab's two halves, XCHG) -> dv_full; then the
    NEW wait dvk_stage_empty (the store warp drained round 1's five source reads) -> ROUND 2 = dK's slices 1..3 over CFG.dkv_acc_r2_slot(0..2)
    -> dk_full.  The dq_empty ring is untouched (its pre-armed wait k <-> release k-1 accounting cannot carry an item-end credit: the first
    build's attempt let round 2 overwrite round 1 under the reduce reads).  Returns the advanced stage_state (one flip per item incl. zero-trip)."""
    # fp32_reduce: R1 dV 4 + dK 0 over K2 + ALIAS (80 KiB); R2 dK 1-3 after dvk_stage_empty
    for s in cutlass.range_constexpr(CFG.DKV_ACC_SLICES):
        _stage_fp32_slice(slab, tmem_base, lane, live, CFG.DV_COL, s, geo.dkv_acc_r1_slot(s))
    _stage_fp32_slice(slab, tmem_base, lane, live, CFG.DK_COL, 0, geo.dkv_acc_r1_slot(CFG.DKV_ACC_SLICES))
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.dv_full.arrive()
    # WAR: round 1's five slots were read out by the store warp's reduce-adds.  STAMPS: no record word for this row (DBG_CLK.WORD is the
    # 82-word default-body layout), so the wait is passed dbg=None = the shipped statements, never timed; the item's other waits keep their words.
    _wait(bars.dvk_stage_empty, stage_state.phase, None, 0)
    for s in cutlass.range_constexpr(CFG.DKV_ACC_R2_SLICES):
        _stage_fp32_slice(slab, tmem_base, lane, live, CFG.DK_COL, s + 1, geo.dkv_acc_r2_slot(s))
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    bars.dk_full.arrive()
    return advance(stage_state, 1)


@cute.jit
def _dkv_fp32_reduce_rounds(slab, bars, geo: cutlass.Constexpr, maps, batch, n_batch, kv_row0, kv_head, live_rows, out_phase, dbg):
    """dkv_accum = fp32_reduce, the store warp's item end (after its dv_full wait): ROUND 1 = five reduce-adds (dV's four slices + dK's slice 0)
    into the per-KV-head fp32 workspace maps (the MAP_DV / MAP_DK slots: 128-row fp32 [M x DQ_BOX]-box maps; kv head = head // ratio) as ONE
    bulk group -> drain (P7) -> dvk_stage_empty.arrive() (bare 32: the epilogue's round 2 may overwrite the slots) -> the dk_full wait -> ROUND 2
    = three reduce-adds (dK's slices 1..3; their drain = the caller's item-end `tma_store_wait(0)` before dvk_empty / k2_empty).  A dead CTA
    (kv_row0 >= seq_k) issues nothing and acquires no map (the workspace is zero-filled); every commit / wait / arrive is kept."""
    # 8 UTMAREDG fp32 boxes per item replace 4 UTMASTG; issue under live_rows, flips not
    dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV)
    dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK)
    if live_rows:
        tma_tensormap_acquire(dv_desc)
        tma_tensormap_acquire(dk_desc)
        for i in cutlass.range_constexpr(CFG.DKV_ACC_SLICES):
            tma_reduce_add_subtile(
                _dkv_acc_tile(slab, geo.dkv_acc_r1_slot(i)), tma_slice_runtime_desc(dv_desc, i * CFG.DQ_BOX, kv_row0, kv_head, 0), 0, acquire=False
            )
        tma_reduce_add_subtile(
            _dkv_acc_tile(slab, geo.dkv_acc_r1_slot(CFG.DKV_ACC_SLICES)), tma_slice_runtime_desc(dk_desc, 0, kv_row0, kv_head, 0), 0, acquire=False
        )
    tma_store_commit()
    tma_store_wait(0)
    bars.dvk_stage_empty.arrive()  # all 32 after round 1's five source reads: the epilogue's round 2 may overwrite the slots
    _wait_store(bars.dk_full, out_phase, dbg, _W["wait:store:dk_full"])
    if live_rows:
        for i in cutlass.range_constexpr(CFG.DKV_ACC_R2_SLICES):
            tma_reduce_add_subtile(
                _dkv_acc_tile(slab, geo.dkv_acc_r2_slot(i)), tma_slice_runtime_desc(dk_desc, (i + 1) * CFG.DQ_BOX, kv_row0, kv_head, 0), 0, acquire=False
            )


@cute.jit
def _ts(desc, plane_col, bdesc, dst, accum):
    """TS MMA over the q (K) dimension: A = the packed P^T / dS^T plane at TMEM column ``plane_col`` (8 columns per k-step, PACKED order);
    the B descriptor walks the q blocks in CFG.ts_k_perm order (the quarter-interleaved pack of header section 9)."""
    for k in cutlass.range_constexpr(CFG.N // CFG.K_HW):
        qblock = CFG.ts_k_perm(k)
        bstep = bdesc + ((desc.smem_advance_B_intra * qblock) >> 4)
        mma_ts_step(
            desc,
            nvvm.make_tmem_ptr(plane_col + k * desc.tmem_advance_A, cutlass.Int8),
            bstep,
            dst,
            k_idx=0,
            accumulate=accum if cutlass.const_expr(k == 0) else True,
        )


@cute.jit
def _s_lookahead(
    slab,
    bars,
    score,
    k1d,
    s_acc,
    q1_next,
    s_tail_state,
    elect_p,
    qtile,
    lo,
    hi,
    dtype: cutlass.Constexpr,
    s_tail_off: cutlass.Constexpr[bool],
    dq576: cutlass.Constexpr[bool],
    geo: cutlass.Constexpr,
    dbg,
):
    """S^T(t + 1) into the S slot at slot 4 of tile t (after dV(t): the in-order tensor pipe reads P^T(t) first): _wait_tcgen(q1_full[next])
    -> [s_tail map, iff qtile > lo: _wait_tcgen(s_tail_empty) + advance -- dQ(t-1) was issued into [64, 128) at slot 2 of THIS tile and the epilogue
    must have read it out (D1 ii); at tile lo nothing was issued into the tail since S^T(lo), whose softmax read p_ready(lo) already covers]
    -> mma_ss -> _commit_pair(s_done), _commit_pair(q1_empty[next]).  Skipped on the item's last tile (nothing waits those flips).
    Returns the s_tail state (one flip per wait).  each S^T issue after a dQ waits s_tail_empty (dQ read out of the S tail).
    ``s_tail_off`` (DIAGNOSTIC s_tail_wait_off, header section 11): the wait is skipped, the advance stays (the phase flips unobserved; the
    epilogue's arrive is untouched) -- S^T(t+1) may land on an unread dQ(t-1): WRONG BY DESIGN.
    ``dq576`` (tmem_map dq_576, header sections 9 / 11): the S tail has no second tenant -- no s_tail_empty row, no wait, no advance; ``s_tail_state``
    is the leader's None and is returned unchanged (the S^T(t+1) WAR over the softmax's S^T(t) read rests on p_ready(t), waited at slot 3)."""
    if qtile + 1 < hi:
        _wait_tcgen(bars.q1_full[q1_next.idx], q1_next.phase, dbg, _W["wait:leader:q1_full"])
        if cutlass.const_expr(not dq576):
            # DIAGNOSTIC s_tail_wait_off: the guard holds the wait only; the advance below is unconditional (D1 ii)
            if qtile > lo:
                if cutlass.const_expr(not s_tail_off):
                    _wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])
                s_tail_state = advance(s_tail_state, 1)
        q1d_next = _operand_k_major(slab, geo.Q1 + q1_next.idx * geo.Q1_BYTES, dtype, CFG.Q_HALF).desc()  # the sQ1 ring stage (geo.Q1_STAGES)
        mma_ss(score, k1d, q1d_next, s_acc)
        _commit_pair(bars.s_done, elect_p)
        _commit_pair(bars.q1_empty[q1_next.idx], elect_p)
    return s_tail_state


@cute.jit
def _dq_issue(
    bars,
    dqdesc,
    dsd,
    k2d,
    dq_acc,
    xchg_state,
    dq_read_state,
    elect_p,
    fa4_roles: cutlass.Constexpr[bool],
    b2: cutlass.Constexpr[bool],
    geo: cutlass.Constexpr,
    dbg,
    tag_back: cutlass.Constexpr[int],
):
    """ONE dQ issue (dQ(t-1) at slot 2 of tile t, or the tail dQ(hi-1)): the three dS gates of the exchange on ``xchg_state`` (the P2 split:
    one flip per dQ ISSUE, one tile behind ready_state), the M 128 pair MMA into [DQ_COL, +64) = S^T's dead tail, the dq_done commit.
    The three waits are UNCONDITIONAL under every probe: the ds_recv_full wait (CTA 0) / the ds_pair_ready wait fed by the relay (CTA 1) is
    the ONLY back-edge of the ds_recv_full expect_tx ARM (P14-B: arm(t+1) <- dq_done(t) <- this site); the the gate-skip probe probe that skipped
    them let arm(t+1) land on a tx-pending phase and FAULTED (CUDA 719) -- it is declined by name on this map and under every role shape
    (the barrier audit; header section 5).  ``dbg`` / ``tag_back`` = STAMPS: the wait sums and the ship -> recv handshake
    (:func:`_clk_recv`).  ``fa4_roles`` (16w_fa4; the barrier audit x-fa4 table section 15): ALSO `_commit_pair(bars.dq_war_done)` IMMEDIATELY
    BEFORE the dq_done commit -- the softmax's sDS WAR signal, flipped at the two dQ issue sites ONLY (never at the zero-trip empty-commit site),
    so every flip of the softmax's pre-armed waiter depends on its own preceding dS STS (ds_smem) and can never land early.
    ``b2`` (smem_map rubin_b2; header sections 5 / 8 / 11): the three gates are the slots ``xchg_state.idx`` of the two-stage ring (``_slot``; xchg_state
    advances by ``geo.DS_STAGES``), the A descriptor is the pinned sDS stage-0 root advanced by (stage x DS_BYTES) >> 4, and the FOURTH gate is the row
    ``dq_read_done`` on ``dq_read_state`` (st(1) pre-armed, +1 per wait: the dQ image's REUSE edge -- both epilogues read the previous issue's image out
    before this issue overwrites it; the first issue passes on the pre-arm; the drain after the loop consumes the last readout).  Returns
    (xchg_state, dq_read_state) advanced; ``dq_read_state`` is None under sm100 (no row, no wait)."""
    _wait_tcgen(_slot(bars.ds_smem, xchg_state, geo.DS_STAGES), xchg_state.phase, dbg, _W["wait:leader:ds_smem"])
    _wait_tcgen(_slot(bars.ds_recv_full, xchg_state, geo.DS_STAGES), xchg_state.phase, dbg, _W["wait:leader:ds_recv_full"])
    if cutlass.const_expr(dbg is not None):
        _clk_recv(dbg, _W["tiles_leader"], _W["recv_leader_sum"], _W["recv_leader_matched"], _W["recv_leader_missed"], tag_back)
    _wait_tcgen(_slot(bars.ds_pair_ready, xchg_state, geo.DS_STAGES), xchg_state.phase, dbg, _W["wait:leader:ds_pair_ready"])
    if cutlass.const_expr(b2):
        # BARRIER dq_read_done (LEADER 256, st(1)): the previous dQ issue's image was read out by the four epilogue warps of BOTH CTAs (their LDTM +
        # tcgen05_wait(LOAD) + fence precede the relaxed arrive); the wait's after_thread_sync fence closes the tcgen05 side before the overwrite
        _wait_tcgen(bars.dq_read_done, dq_read_state.phase, dbg, _W["wait:leader:dq_read_done"])
        dq_read_state = advance(dq_read_state, 1)
        dsd_s = dsd + cutlass.Int64(xchg_state.idx) * cutlass.Int64(geo.DS_BYTES >> 4)  # the sDS stage of this issue: one pinned root + the stage advance
    else:
        dsd_s = dsd
    # skew lever 7 (SOFTMAX_DQ_WAR): the LEADER holds the dQ issue -- under 16w_fa4 both softmaxes' dS STS wait dq_war_done ALONE
    if cutlass.const_expr(DEBUG_SOFTMAX_DQ_WAR_DELAY_US > 0):
        for _us in cutlass.range(DEBUG_SOFTMAX_DQ_WAR_DELAY_US, unroll=1):
            nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])
    mma_ss(dqdesc, dsd_s, k2d, dq_acc)
    if cutlass.const_expr(fa4_roles):
        _commit_pair(bars.dq_war_done, elect_p)  # section 15: the softmax's WAR signal (the two dQ sites only), BEFORE dq_done
    _commit_pair(bars.dq_done, elect_p)
    return advance(xchg_state, geo.DS_STAGES), dq_read_state


@cute.jit
def _dq_slot(
    bars,
    dqdesc,
    dsd,
    k2d,
    dq_acc,
    xchg_state,
    dq_read_state,
    ready_state,
    elect_p,
    qtile,
    lo,
    fa4_roles: cutlass.Constexpr[bool],
    s_read_off: cutlass.Constexpr[bool],
    dq576: cutlass.Constexpr[bool],
    b2: cutlass.Constexpr[bool],
    geo: cutlass.Constexpr,
    dbg,
):
    """The in-tile dQ SLOT of tile t (slot 2: ONE helper at ONE site of the leader): [s_tail map] _wait_tcgen(s_read(t)) on ready_state EVERY tile
    (both softmaxes hold S^T(t)'s tail in registers before dQ(t-1) overwrites [DQ_COL, +DQ_COLS); P14, N / N), then iff qtile > lo the dQ(t-1) issue
    behind the three dS gates (_dq_issue: xchg_state +1 per issue).  Returns xchg_state.  The dQ image never overlaps dV's P^T read [P_COL, +64).
    ``s_read_off`` (DIAGNOSTIC s_read_wait_off, header section 11): the s_read wait is skipped; ready_state (advanced by the caller per tile) and
    the softmax's 512-lane arrive stay -- dQ(t-1) may land under the softmax's S^T(t) read: WRONG BY DESIGN.  The three dS gates of _dq_issue are
    never under a probe.  ``dq576`` (tmem_map dq_576): no s_read row (the dQ image shares no column with S^T): the slot is the gated issue alone."""
    if cutlass.const_expr(not dq576):
        # DIAGNOSTIC s_read_wait_off: the guard holds the wait only
        if cutlass.const_expr(not s_read_off):
            _wait_tcgen(bars.s_read, ready_state.phase, dbg, _W["wait:leader:s_read"])
    if qtile > lo:
        xchg_state, dq_read_state = _dq_issue(
            bars, dqdesc, dsd, k2d, dq_acc, xchg_state, dq_read_state, elect_p, fa4_roles, b2, geo, dbg, DBG_CLK.TAG_BACK_IN_TILE
        )
    return xchg_state, dq_read_state


@cute.jit
def _dq_round(slab, bars, slot_base, dhalf, qrow, values, empty_state, sts_off: cutlass.Constexpr[bool], dbg):
    """One dQ staging ROUND of the epilogue warp pair of d half ``dhalf`` (D4 pair: the store's closed per-round release + this both-slots
    wait form, change both or neither): every warp waits BOTH slots' release at the round's phase, STS the lane's DQ_BOX fp32 into slot
    dhalf, fence_proxy, publish dq_full[dhalf] (2 warps x 32 = DQ_SLICE_LANES); ONE flip of empty_state per round.
    ``sts_off`` (DIAGNOSTIC sdq_sts_off `sdq_sts_off`) skips the STS and its proxy fence (the store warp's reduce-add then reads stale staging: -256
    wavefronts, dQ WRONG BY DESIGN); the waits and the arrive stay -- the ring PROTOCOL is unchanged (header section 5).  ``dbg`` = the STAMPS bracket of the two dq_empty waits (one word per round).
    """
    for s in cutlass.range_constexpr(CFG.DQ_SLOTS):
        _wait(bars.dq_empty[s], empty_state.phase, dbg, _W["wait:epilogue:dq_empty"])
    if cutlass.const_expr(not sts_off):
        _store_dq_slice(slab, slot_base, qrow, values)
        nvvm.fence_proxy("async.shared", space="cta")
    bars.dq_full[dhalf].arrive()
    return advance(empty_state, 1)


# ---------------------------------------------------------------------------------------------------------------------
# Role bodies (section 4).  The ``# BARRIER`` punch lists are the audit's issue-order contract (the audit record
# section 7); the code follows them line by line.
# ---------------------------------------------------------------------------------------------------------------------


@cute.jit
def _softmax_loop(
    slab,
    bars,
    sched,
    tmem_base,
    lse,
    delta,
    origins,
    stats_origin: cutlass.Constexpr[int],
    pair,
    head,
    batch,
    meta,
    n_batch,
    rank,
    warp,
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_2cta_code: cutlass.Constexpr[int],
    fa4_roles: cutlass.Constexpr[bool],
    regs: cutlass.Constexpr[int],
    mask_form_code: cutlass.Constexpr[int],
    stats_form_code: cutlass.Constexpr[int],
    tmem_map_code: cutlass.Constexpr[int],
    geo: cutlass.Constexpr,
    dbg,
):
    """Warps 0-7, BOTH CTAs (lane = own kv row; group g owns q quarters {g, g+2}): S^T / dP^T -> P^T / dS^T packed in place; under
    ``fa4_roles`` (warp_roles 16w_fa4, the alternate role shape) ALSO the dS exchange: sDS own half + XCHG from the same registers, the ship, the sDS WAR wait.
    ``diag_2cta_code`` (DIAGNOSTIC): DIAG_STATS_ONE_CHUNK (stats_one_chunk) = `_p_values` / `_ds_values` load one stats / delta chunk per quarter and
    reuse it (stats WRONG BY DESIGN; no barrier touched); DIAG_EXCHANGE_OFF (exchange_off) under 16w_fa4 = this role's arm expects 0 bytes and the ship
    is not issued (the 16w site is the epilogue's).  ``mask_form_code`` (header section 11, the mask-form lever): under MASK_FORM_PREDICATED the
    per-tile CTA-uniform scalar ``mask_tile`` = tile_may_mask(pair, rank, qtile, seq_q, seq_k, causal) (section 7) gates the mask body of
    _p_values; the barrier program is identical under every form (no wait, arrive or fence moves).  ``dbg`` (STAMPS, warp 0 = the
    representative; None under 16w_fa4 -- the lever declines there): the s_done / dp_done / sched_full wait sums; None = no traced code.
    ``tmem_map_code`` (header sections 9 / 11): under TMEM_MAP_DQ576 the s_read row does not exist -- the fence + `_arrive_leader(bars.s_read)` after
    `_s_values` are not traced (the S^T(t+1) WAR over this read rests on p_ready(t), as in the first body); nothing else of this role moves.

    # BARRIER (per item, FIRST thing): _credit(bars.sched_empty[sched_state.idx])  -- before any work (every role: the top-of-item
    #   credit is what keeps the peer provably resident for the remote credits, P15)
    # BARRIER (per tile t, tile_state st(0)):
    #   _wait_tcgen(bars.s_done, tile_state.phase) -> [test-only DEBUG_SOFTMAX_S_READ_DELAY_US sleep, FOLLOWER only, default none]
    #   -> _s_values: tcgen05.ld S^T quarters a, b -> tcgen05_wait(LOAD) -> tcgen05_fence(BEFORE_THREAD_SYNC)
    #   -> _arrive_leader(bars.s_read)   [NEW; LEADER, bare, EVERY tile (D2), 256 x 2 CTAs = 512: the S^T registers are held, dQ(t-1) may land]
    #   -> _p_values (exp2 / mask / pack, tcgen05.st into [0, 64)) -> tcgen05_wait(STORE) -> tcgen05_fence(BEFORE_THREAD_SYNC)
    #   -> _arrive_leader(bars.p_ready)                 [LEADER, bare, 512]
    #   _wait_tcgen(bars.dp_done, tile_state.phase) -> dS^T pack -> tcgen05_wait(STORE) -> fence
    #   -> _arrive_leader(bars.ds_ready)  [LEADER, 512]  and  bars.ds_tmem.arrive()          [THREAD local, bare, 256]
    #   tile_state = advance(tile_state, 1)
    # Required waits: s_done, dp_done (own copies of the leader's multicast commits).  Zero-trip item: credit only (pair-uniform lo / hi).
    # Non-mbarrier (stats_form ldg): _stats -> named_barrier_fence(CFG.SOFTMAX_BARRIER, CFG.SOFTMAX_LANES) x 2 per call (both CTAs hold all 128 q stats).
    # BARRIER under stats_form tma_ring (the S1 skeleton 2026-10-27 17:30 / 17:46, resources/s1-barrier-table.md sections 1 / 7.1 / 14; stats_state =
    #   PipelineState.start(), CFG_RING.STATS_STAGES = 2 stages, declared BEFORE the persistent loop and CARRIED across items: tile lo of an item takes
    #   stage K % 2 for K = this CTA's running tile count, NEVER a per-item reset to stage 0):
    #   per tile t, after _arrive_leader(bars.s_read): _wait(bars.stats_full[stats_state.idx], stats_state.phase)   [plain _wait, NO tcgen05 fence, NO
    #     proxy fence: the bulk copy's complete_tx + the acquire wait publish the stage to the 256 lanes; st(0): wait k <-> the copy of tile k]
    #   -> _p_values reads the LSE row at CFG.stats_lse(stats_state.idx) -> ... -> _ds_values reads the delta row at CFG.stats_delta(stats_state.idx)
    #   -> bars.stats_empty[stats_state.idx].arrive()   [THREAD local, bare, 8 warps = 256 == SOFTMAX_LANES; right after the tile's LAST stats read;
    #     the arrive's release orders this lane's LDS: no fence]  -> stats_state = advance(stats_state, CFG_RING.STATS_STAGES) with tile_state's advance
    #   The two _stats call sites (the item prologue, the per-tile lookahead) and the SOFTMAX_BARRIER pair are NOT traced.  Zero-trip item: 0 / 0 (no
    #   site outside the tile loop); n_q == 1: 1 / 1; no bootstrap, no drain (LOCAL rows, every copy consumed in-loop).
    # BARRIER under fa4_roles (16w_fa4; the barrier audit x-fa4 table section 9, the softmax stub order; dq_war = PipelineState.start(phase=1)):
    #   per item, first: _credit; ship_pred = (warp == SOFTMAX_LEAD_WARP) & nvvm.elect_sync()   [ONCE per item, P16]
    #   per tile t, steps 1-3 as above up to _arrive_leader(bars.ds_ready) [STTM + wait(STORE) + fence BEFORE the WAR wait: dK(t) never waits dQ(t-1)];
    #     bars.ds_tmem.arrive() DELETED (no field);
    #   4. _wait(bars.dq_war_done, dq_war.phase); dq_war = advance(dq_war, 1)   [NEW: plain _wait, no tcgen05 fence; the sDS WAR: dQ(t-1) read own +
    #      recv, both ships read XCHG; wait k consumes flip k-1 (the previous live tile's / item's dQ): the kernel's first dS(lo) passes on the pre-arm;
    #      section 15: the barrier is dq_war_done (flipped by dQ issues ONLY), so no flip can land before its wait and no lap is possible]
    #   5. _store_words_quarter(slab, lo_base, row, words, 0, vec0) [quarter g: q half 0 -> own on rank 0 / XCHG on rank 1];
    #      _store_words_quarter(slab, hi_base, row, words, PACK_QUARTER_COLS, vec0) [quarter g+2: q half 1 -> XCHG on rank 0 / own on rank 1]
    #      -> nvvm.fence_proxy("async.shared", space="cta") [ONE] -> bars.ds_smem.arrive()   [THREAD local, bare, 8 warps = 256 == SOFTMAX_LANES]
    #   6. named_barrier_fence(CFG.XCHG_BARRIER, XCHG_BARRIER_LANES_16W_FA4) [bar.sync 2, 256; P11 KEEP: the ship reads all 8 warps' XCHG stores]
    #      -> [test-only LEADER_SHIP sleep, LEADER only] -> bars.ds_recv_full.arrive(n_bytes=CFG.XCHG_TX, pred=ship_pred) [1 == init; the PEER's
    #      copy delivers the bytes; back-edge = step 4] -> _ship_xchg(slab, bars, rank, ship_pred)
    #   7. if qtile + 1 < hi: _stats(qtile + 1); tile_state = advance(tile_state, 1)
    #   item end: NOTHING on dq_war (section 15: a zero-trip item has 0 flips / 0 waits on dq_war_done; a live item's tail flip is consumed by the
    #     next live item's tile lo' or by the drain)
    #   after the persistent loop: _wait(bars.dq_war_done, dq_war.phase)   [the ONE unconditional drain; no advance]
    """
    # s_read (512, every tile) after S^T ld + wait(LOAD): dQ(t-1) lands in S^T's tail
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.INCREASE)  # CFG.SOFTMAX_REGS
    tid = cute.arch.thread_idx()[0]
    row = tid % CFG.M
    group = tid // CFG.M
    one_chunk = cutlass.const_expr(diag_2cta_code == DIAG_STATS_ONE_CHUNK)  # DIAGNOSTIC stats_one_chunk: one LSE / delta chunk per quarter reused
    # DIAGNOSTIC exchange_off (exchange_off) RE-SITED for 16w_fa4 : this role arms ds_recv_full with 0 bytes and issues no ship; every wait stays
    exchange_off = cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF)
    xchg_tx = CFG.DIAG_EXCHANGE_OFF_TX if cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF) else CFG.XCHG_TX
    tma_ring = cutlass.const_expr(stats_form_code == STATS_FORM_TMA_RING)  # the stats form (header section 11): the ring sites below trace under it only
    dq576 = cutlass.const_expr(tmem_map_code == TMEM_MAP_DQ576)  # the TMEM map (header section 9): no s_read row under the independent dQ image
    tile_state = PipelineState.start()
    sched_state = PipelineState.start()
    if cutlass.const_expr(tma_ring):
        # RESOURCE-NOTE: stage base = STATS_RING + idx*1024; LSE @+0, delta @+512: today's LDS.128 immediates
        stats_state = PipelineState.start()  # st(0): wait k <-> the loader's copy k; ONE state for the barrier index AND the stage byte base (P2); carried
    if cutlass.const_expr(fa4_roles):
        # lo_base=rank?XCHG:DS, hi_base=rank?DS+16K:XCHG; vec 4g..4g+3; (row%8)^vec: floor
        own_base = cutlass.Int32(geo.DS) + rank * cutlass.Int32(CFG.DS_HALF_BYTES)  # geo.ds_own(rank): our 128 kv rows of sDS
        xchg_base = cutlass.Int32(geo.XCHG)  # the peer's 64 q of our rows: the ship's source
        is_leader_cta = rank == cutlass.Int32(LEADER_RANK)
        lo_base = cutlass.Int32(arith.select(is_leader_cta.ir_value(), own_base.ir_value(), xchg_base.ir_value()))  # quarter g = q half 0 (rank 0's own)
        hi_base = cutlass.Int32(arith.select(is_leader_cta.ir_value(), xchg_base.ir_value(), own_base.ir_value()))  # quarter g+2 = q half 1 (rank 1's own)
        lo_row = lo_base + row * cutlass.Int32(CFG.SWIZZLE_BYTES)  # the lane's 128-B row of the quarter-g slab (item-invariant: 1 register)
        hi_row = hi_base + row * cutlass.Int32(CFG.SWIZZLE_BYTES)  # the lane's row of the quarter-(g+2) slab (1 register)
        xor_lane = (row % 8) ^ (group * cutlass.Int32(CFG.DS_STS_VECS_PER_QUARTER))  # (row % 8) ^ vec0: group g writes vec 4g .. 4g+3 of BOTH halves
        is_lead_warp = warp == cutlass.Int32(SOFTMAX_LEAD_WARP)
        dq_war = PipelineState.start(phase=1)  # pre-armed (P5): the kernel's first dS STS has no predecessor; +1 per live-tile wait (no zero-trip site)
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        if cutlass.const_expr(fa4_roles):
            ship_pred = is_lead_warp & nvvm.elect_sync()  # ONE lane of warp SOFTMAX_LEAD_WARP per item (P16): the arm + the ship
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        key = pair * CFG.PAIR_KV_ROWS + rank * CFG.M + row
        if cutlass.const_expr(not tma_ring):  # stats_form ldg: the item prologue's stats fetch (tile lo) under the SOFTMAX_BARRIER pair
            if lo < hi:
                _stats(slab, lse, delta, row, group, lo * CFG.N, batch, head, scale, q_start, seq_q, origins, stats_origin, geo)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.s_done, tile_state.phase, dbg, _W["wait:softmax:s_done"])
            if cutlass.const_expr(DEBUG_SOFTMAX_S_READ_DELAY_US > 0):
                if rank != cutlass.Int32(LEADER_RANK):  # skew lever 5: the FOLLOWER softmax holds its S^T read (s_read alone holds dQ(t-1))
                    for _us in cutlass.range(DEBUG_SOFTMAX_S_READ_DELAY_US, unroll=1):
                        nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])
            a, b = _s_values(tmem_base, group)
            # s_tail map: the S^T registers are held, dQ(t-1) may land in the tail (s_read, 512, EVERY tile); dq_576: no row, no fence, no arrive --
            # the tail is S^T's alone and the next fence precedes p_ready
            if cutlass.const_expr(not dq576):
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                _arrive_leader(bars.s_read)
            stats_base = geo.STATS  # stats_form ldg: the single slab `_stats` filled (a trace-time constant: the LDS immediates of today's rendering)
            if cutlass.const_expr(tma_ring):
                # RESOURCE-NOTE: stats_full wait AFTER the s_read arrive, before _p_values: dQ(t-1) never waits stats
                # BARRIER: stats_full[s] st(0) = the loader's bulk copy of THIS tile's (head, q tile) record landed in stage s; plain _wait (the
                # complete_tx + acquire publish the async-proxy write to every lane: no tcgen05 fence, no proxy fence); the STAMPS word brackets it
                _wait(bars.stats_full[stats_state.idx], stats_state.phase, dbg, _W["wait:softmax:stats_full"])
                stats_base = cutlass.Int32(geo.STATS_RING) + stats_state.idx * cutlass.Int32(CFG.STATS_STAGE_BYTES)  # the stage's LSE row (delta at +512)
            mask_tile = None  # read under MASK_FORM_PREDICATED only (section 7's CTA-uniform per-tile scalar; the other forms trace no predicate)
            if cutlass.const_expr(mask_form_code == MASK_FORM_PREDICATED):
                mask_tile = tile_may_mask(pair, rank, qtile, seq_q, seq_k, causal)
            p = _p_values(slab, stats_base, tmem_base, group, key, qtile, seq_q, seq_k, scale, a, b, causal, dtype, one_chunk, mask_form_code, mask_tile)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            _arrive_leader(bars.p_ready)
            _wait_tcgen(bars.dp_done, tile_state.phase, dbg, _W["wait:softmax:dp_done"])
            words = _ds_values(slab, stats_base + CFG.STATS_DELTA_OFF, tmem_base, group, p, scale, dtype, one_chunk)
            if cutlass.const_expr(tma_ring):
                # RESOURCE-NOTE: stats_empty bare 256 right after _ds_values (last delta LDS); st(0) +2; no proxy fence
                # BARRIER: stats_empty[s] THREAD local, bare (8 warps x 32 = 256 == SOFTMAX_LANES) right after the tile's LAST stats read (the delta
                # LDS of _ds_values): the release orders this lane's loads; the loader's acquire wait + program order gate the next copy into the stage
                bars.stats_empty[stats_state.idx].arrive()
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            _arrive_leader(bars.ds_ready)
            if cutlass.const_expr(not fa4_roles):
                bars.ds_tmem.arrive()
            else:
                # STTM + ds_ready BEFORE the dq_done WAR wait: dK(t) never waits dQ(t-1)
                # WAR: wait dq_war_done (dQ issues only, never the zero-trip commit); st(1); drain after the loop
                # section 15: dQ(t-1)'s read of sDS own + recv (and, via the leader's gates, both ships' XCHG reads) done.  STAMPS declines under
                # 16w_fa4 (no record word for this row): dbg=None = the shipped statements, untimed.
                _wait(bars.dq_war_done, dq_war.phase, None, 0)
                dq_war = advance(dq_war, 1)
                _skew_follower_own_half(rank)  # skew lever 1 (16w_fa4 site): the FOLLOWER holds its 8 STS (its ds_smem AND its ship)
                # dS STS from the 32 packed words = today's LDTM image; 4 own + 4 XCHG STS.128/lane
                xor_t = opaque_i32(xor_lane, anchor=qtile)  # pins the 8 address adds inside the tile (hoisted per item they spilled 4 words at 144 regs)
                _store_words_quarter(slab, lo_row, words, 0, xor_t)
                _store_words_quarter(slab, hi_row, words, CFG.PACK_QUARTER_COLS, xor_t)
                nvvm.fence_proxy("async.shared", space="cta")
                # ds_smem 256 THREAD bare after own STS + fence.proxy; no tcgen05 fence (registers)
                bars.ds_smem.arrive()
                # XCHG_BARRIER 256 (8 softmax warps) -> warp 0 elect: arm ds_recv_full, ship XCHG
                named_barrier_fence(CFG.XCHG_BARRIER, XCHG_BARRIER_LANES_16W_FA4)
                _skew_leader_ship(rank)  # skew lever 2 (16w_fa4 site): the LEADER holds its arm + ship (the half is written, the barrier passed)
                # BARRIER (the arm's back-edge on EVERY path, section 15.2 row 3): arm(t) follows this warp's dq_war_done wait above = dQ(t-1) issued
                # after the leader's ds_recv_full(t-1) / ds_pair_ready(t-1) waits (both copies of the previous phase landed); across items the first
                # tile's wait consumes the previous LIVE item's tail flip (a real dQ behind the leader's gates); a zero-trip item adds no phase
                # (no tile, no arm, no ship) and no flip -- the edge is a DATA edge through a real dQ, never through the zero-trip empty commit.
                bars.ds_recv_full.arrive(n_bytes=xchg_tx, pred=ship_pred)  # XCHG_TX; the probe arms CFG.DIAG_EXCHANGE_OFF_TX = 0 (completes at the arm)
                if cutlass.const_expr(not exchange_off):  # DIAGNOSTIC exchange_off (re-sited): the ship only; the arm above expects 0 bytes, every wait stays
                    _ship_xchg(slab, bars, rank, ship_pred, None, geo, False)  # 16w_fa4 x rubin_b2 declines by name: the single-stage ship
            if cutlass.const_expr(not tma_ring):  # stats_form ldg: the per-tile lookahead fetch of tile t+1 under the SOFTMAX_BARRIER pair
                if qtile + 1 < hi:
                    _stats(slab, lse, delta, row, group, (qtile + 1) * CFG.N, batch, head, scale, q_start, seq_q, origins, stats_origin, geo)
            if cutlass.const_expr(tma_ring):
                stats_state = advance(stats_state, geo.STATS_STAGES)  # +1 per tile, the same count on both sides of the ring (P14)
            tile_state = advance(tile_state, 1)
        # section 15: NO zero-trip wait and NO zero-trip advance on dq_war -- dq_war_done flips at dQ issues only (0 / 0 on every zero-trip path)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:softmax:sched_full"])
    if cutlass.const_expr(fa4_roles):
        # P15 / section 15.3: the ONE unconditional drain of dq_war_done (the kernel's last tail flip, or the pre-arm when this CTA ran no live item);
        # the leader's last commit is issued before it leaves its loop and depends on nothing of this warp: the drain cannot hang.
        _wait(bars.dq_war_done, dq_war.phase, None, 0)  # STAMPS declines under 16w_fa4: no record word for this row (passed dbg=None, untimed)


@cute.jit
def _epilogue_loop(
    slab,
    bars,
    sched,
    tmem_base,
    pair,
    head,
    batch,
    meta,
    n_batch,
    rank,
    warp,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_2cta_code: cutlass.Constexpr[int],
    tmem_map_code: cutlass.Constexpr[int],
    smem_map_code: cutlass.Constexpr[int],
    geo: cutlass.Constexpr,
    regs: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    dbg,
):
    """Warps 8-11, BOTH CTAs (lane = own kv row for dS; lane = q row of d half (warp // 2) % 2 for dQ): dS^T -> sDS own half + XCHG
    (the 16-B STS form), the DSMEM ship, the dQ fp32 rounds into the DEDICATED sDQ ring, the native dV / dK stores.
    ``diag_2cta_code`` (DIAGNOSTIC, header section 11): DIAG_EXCHANGE_OFF (exchange_off) = the ship is NOT issued and the ds_recv_full arm expects
    CFG.DIAG_EXCHANGE_OFF_TX = 0 bytes (each phase completes at its own arm; every wait, the relay, the XCHG STS / fence / named barrier
    stay; the recv half is never written); DIAG_SDQ_STS_OFF (sdq_sts_off) = `_dq_round` skips the slice STS + its fence (the reduce-add reads stale
    staging).  ``dbg`` (STAMPS; the LEADING warp = the representative): the wait sums, the per-tile hand-off segments, the ship stamp.
    ``tmem_map_code`` (header sections 9 / 11): the dQ readout's column base (`_dq_col`: CFG.DQ_COL = S^T's dead tail | CFG_DQ576.DQ_COL = past dK);
    under TMEM_MAP_DQ576 the s_tail_empty row does not exist and `_arrive_leader(bars.s_tail_empty)` is not traced -- the readout's LDTM +
    tcgen05_wait(LOAD) + tcgen05_fence(BEFORE_THREAD_SYNC) STAY: they are the local half of the independent image's REUSE edge dQ(t) readout -> dQ(t+1)
    write, which this warp's NEXT tile's own-half STS + fences + `ds_smem.arrive()` (the dQ issue's gate) carry by PROGRAM ORDER.  LOAD-BEARING
    (header section 5, P14-B): the dq_done(t) wait + the readout trio sit BEFORE the next tile's staging in loop order -- a staging-before-readout
    reorder voids the carrier (the test module pins the order).
    ``smem_map_code`` (header sections 5 / 8 / 11): under SMEM_MAP_RUBIN_B2 (ON dq_576) the staging is the TWO-STAGE ring -- tile t's own STS, XCHG STS,
    ds_recv_full arm, ship and ds_smem arrive all take stage / slot ``sds_state.idx`` (``sds_state`` 2 stages, +1 per tile, carried across items, never
    reset: P2) -- and the order is the LAGGING one: dS(t) is staged BEFORE this warp waits dq_done(t-1) (the WAR on the stage is dq_done(t-2), waited one
    iteration earlier), so the readout of dQ(t-1) follows ds_smem(t) in program order and the dq_576 carrier is void; every readout therefore arrives the
    LEADER row ``dq_read_done`` (bare 128 x 2 = 256) right after its LDTM + tcgen05_wait(LOAD) + tcgen05_fence(BEFORE_THREAD_SYNC) -- the leader's gate on
    the next dQ issue (`_dq_issue`).  The dq_done waits (N - 1 in-loop under `qtile > lo` + 1 at the item end, the zero-trip observation), the 2N rounds
    per slot and the carried half (round B of dQ(t-2) after the staging of t, round A of dQ(t-1) after its readout) keep today's counts and register shape.

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx]);  ship_pred = is_lead_warp & nvvm.elect_sync();  carried = zeros
    # BARRIER (per tile t):
    #   dS(t) staging: _wait_tcgen(bars.ds_tmem, ds_state.phase) -> tcgen05.ld dS^T (4 x 16 packed cols) -> tcgen05_wait(LOAD)
    #     -> tcgen05_fence(BEFORE_THREAD_SYNC) -> _arrive_leader(bars.dp_empty)   [MOVED HERE: LEADER, bare, 128 x 2 = 256; dP^T(t+1) never waits dQ]
    #     -> bars.ds_recv_full.arrive(n_bytes=CFG.XCHG_TX, pred=ship_pred)   [1 == init; the PEER's copy delivers the bytes]
    #     -> STS XCHG (the peer's 64 q) -> fence_proxy -> named_barrier_fence(CFG.XCHG_BARRIER, CFG.EPILOGUE_LANES)   [P11 KEEP]
    #     -> [test-only DEBUG_LEADER_SHIP_DELAY_US sleep, LEADER only] -> _ship_xchg(slab, bars, rank, ship_pred)
    #     -> [test-only DEBUG_FOLLOWER_OWN_HALF_DELAY_US sleep, FOLLOWER only] -> STS own half -> fence_proxy -> tcgen05 fence
    #     -> bars.ds_smem.arrive()   [THREAD local, bare, 128].  NO dq_empty wait anywhere in this block (the staging alias is deleted).
    #   round B of dQ(t-1): if qtile > lo: _dq_round(carried) = _wait(dq_empty[0]) + _wait(dq_empty[1]) at empty_state.phase -> STS slot dhalf
    #     -> fence_proxy -> bars.dq_full[dhalf].arrive()   [THREAD local, 2 warps = 64 == DQ_SLICE_LANES] -> empty_state += 1
    #   dQ(t) readout: _wait_tcgen(bars.dq_done, dq_state.phase) -> [test-only DEBUG_EPILOGUE_DQ_READOUT_DELAY_US sleep, FOLLOWER only]
    #     -> tcgen05.ld dQ (64 cols at CFG.DQ_COL = 64, the half-row image) -> tcgen05_wait(LOAD) -> tcgen05_fence(BEFORE_THREAD_SYNC)
    #     -> _arrive_leader(bars.s_tail_empty)   [NEW: LEADER, bare, 128 x 2 = 256: S^T(t+1) / the tail dQ / the next prologue may overwrite]
    #   round A of dQ(t): _dq_round(dq[0, 32)) (the same form) -> carried = dq[32, 64);  ds_state, dq_state = advance(.., 1)
    # BARRIER (item end): if lo < hi: round B of dQ(hi-1) (_dq_round(carried))  else: _wait_tcgen(bars.dq_done, dq_state.phase) + advance
    #   -> [test-only DEBUG_FOLLOWER_EPILOGUE_ITEM_END_DELAY_US sleep, FOLLOWER only]
    #   -> native dV over K2 (CFG.dv_store(half); live=False writes zeros) -> fence_proxy + fence -> bars.dv_full.arrive()   [128 of
    #      DV_FULL_LANES 160: the loader adds 32 after its k2_empty observation of the item, fix A]
    #   -> native dK over ds_own(rank) + XCHG (CFG.dk_store(rank, half)) -> bars.dk_full.arrive()   [128]
    #   NO terminal dq_empty observation, NO item-end empty_state flip (D3: 2N rounds == 2N releases per slot per item; zero-trip 0 / 0).
    #   dkv_fp32 (dkv_accum = fp32_reduce, `_dkv_fp32_stage`): round 1 = dV slices 0-3 + dK slice 0 over CFG.dkv_acc_r1_slot(0..4) -> fence -> bars.dv_full.arrive()
    #   -> _wait(bars.dvk_stage_empty, stage_state.phase) [NEW row: the store warp's post-round-1 drain] -> round 2 = dK slices 1-3 over
    #   CFG.dkv_acc_r2_slot(0..2) -> fence -> bars.dk_full.arrive() -> stage_state += 1   [1 wait / 1 release per item incl. zero-trip; the dq_empty ring untouched].
    # No drain: every arrive of this role is local or waited by the leader in-loop (dp_empty's / s_tail_empty's last phases are the leader's drains).
    # BARRIER under smem_map rubin_b2 (ON dq_576; the barrier skeleton R2.c 2.3 / 2.5): sds_state = PipelineState.start(), geo.DS_STAGES = 2, +1 per tile.
    #   per tile t (stage s = sds_state.idx): ds_tmem wait -> dS^T LDTM -> wait(LOAD) -> fence -> _arrive_leader(dp_empty) -> bars.ds_recv_full[s].arrive(
    #   n_bytes=XCHG_TX, pred=ship_pred) [arm(t)'s back-edge: this warp's dq_done(t-2) wait of iteration t-1] -> STS XCHG stage s -> fence_proxy -> the
    #   XCHG named barrier -> _ship_xchg(stage s: our XCHG stage s -> the PEER's sDS stage s recv half, its ds_recv_full[s]) -> STS own half of stage s ->
    #   fence_proxy -> tcgen05 fence -> bars.ds_smem[s].arrive() [128]; if t - 1 > lo: round B of dQ(t-2) (carried); if t > lo: _wait_tcgen(dq_done) ->
    #   dQ(t-1) LDTM -> wait(LOAD) -> fence -> _arrive_leader(bars.dq_read_done) [LEADER, bare, 128 x 2 = 256] -> round A of dQ(t-1) -> carried -> dq_state += 1;
    #   sds_state = advance(sds_state, 2).  item end: if lo < hi: [if hi - 1 > lo: round B of dQ(hi-2)]; dq_done wait -> dQ(hi-1) readout -> dq_read_done
    #   -> rounds A, B; else the zero-trip dq_done wait; dq_state += 1.  Flips per item: dq_read_done N (0 zero-trip) = the leader's N waits; ds_smem[s] /
    #   ds_recv_full[s] N over the two slots; dq_done N (+1); 2N rounds per slot -- the counts of section 5.
    """
    # XCHG -> peer DS_RECV by one cp.async.bulk (16 KiB), complete_tx on the peer's ds_recv_full.
    # epilogue: dS(t) stage -> dQ(t-1) round B -> dQ(t) ld + round A; 32 fp32 carried
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.INCREASE)  # CFG.EPILOGUE_REGS
    lane = cute.arch.thread_idx()[0] % CFG.M  # own kv row for dS^T; TMEM lane of the dQ half-row image
    qrow = lane % CFG.Q_HALF  # dQ row of the half-row image
    dhalf = lane // CFG.Q_HALF  # d half this warp pair holds (warp-uniform) = its sDQ slot
    is_lead_warp = warp == cutlass.Int32(EPILOGUE_LEAD_WARP)
    own_base = cutlass.Int32(geo.DS) + rank * cutlass.Int32(geo.DS_HALF_BYTES)  # geo.ds_own(rank): stage 0 (the dK staging's alias half)
    peer_q = cutlass.Int32(1) - rank  # the q half the PEER needs = the XCHG half
    own_col = cutlass.Int32(CFG.DS_COL) + rank * cutlass.Int32(CFG.PACK_QUARTER_COLS)
    xchg_col = cutlass.Int32(CFG.DS_COL) + peer_q * cutlass.Int32(CFG.PACK_QUARTER_COLS)
    slot_base = cutlass.Int32(geo.DQ) + dhalf * cutlass.Int32(geo.DQ_SLICE_BYTES)  # geo.dq_slot(dhalf): the warp pair's slot of the ring
    b2 = cutlass.const_expr(smem_map_code == SMEM_MAP_RUBIN_B2)  # the SMEM map (header section 8): the two-stage ring + the lagging order + dq_read_done
    sts_off = cutlass.const_expr(diag_2cta_code == DIAG_SDQ_STS_OFF)  # DIAGNOSTIC sdq_sts_off: the sDQ slice STS + its proxy fence skipped in every round
    exchange_off = cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF)  # DIAGNOSTIC exchange_off: no ship; the arm expects 0 bytes; every wait stays
    xchg_tx = CFG.DIAG_EXCHANGE_OFF_TX if cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF) else CFG.XCHG_TX
    dq576 = cutlass.const_expr(tmem_map_code == TMEM_MAP_DQ576)  # the TMEM map (header section 9): no s_tail_empty row under the independent dQ image
    dq_col = _dq_col(tmem_map_code)  # the dQ image's column base: S^T's dead tail (s_tail) | past dK (dq_576); the lane map is the same
    ds_state = PipelineState.start()
    dq_state = PipelineState.start()
    empty_state = PipelineState.start(phase=1)  # pre-armed (P5 b): both slots of the ring are free before the first round; NO bootstrap
    if cutlass.const_expr(b2):
        sds_state = PipelineState.start()  # rubin_b2: the sDS / XCHG ring (geo.DS_STAGES = 2), +1 per LIVE tile, carried across items, never reset (P2)
    if cutlass.const_expr(dkv_fp32):
        stage_state = (
            PipelineState.start()
        )  # fp32_reduce: dvk_stage_empty, +1 per item (incl. zero-trip); defined under the field only: the default rendering is byte-identical
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        if cutlass.const_expr(dbg is not None):
            _clk_add(dbg, _W["items"], cutlass.Int32(1))
        ship_pred = is_lead_warp & nvvm.elect_sync()
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        carried = _dq_zero_half()
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.ds_tmem, ds_state.phase, dbg, _W["wait:epilogue:ds_tmem"])
            if cutlass.const_expr(dbg is not None):
                t_ds = cute.arch.clock64()  # STAMPS: the dS^T(t) readout may begin = the hand-off chain's origin
            x0 = _tmem_load(tmem_base + xchg_col, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            x1 = _tmem_load(tmem_base + xchg_col + CFG.PACK_GROUP_COLS, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            o0 = _tmem_load(tmem_base + own_col, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            o1 = _tmem_load(tmem_base + own_col + CFG.PACK_GROUP_COLS, CFG.PACK_QUARTER_COLS, cutlass.Int32)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            # dp_empty = dS^T readout only (arrive before the XCHG STS); dP^T(t+1) never waits dQ
            _arrive_leader(bars.dp_empty)
            if cutlass.const_expr(not b2):
                stage = None
                own_sts = own_base
                xchg_sts = geo.XCHG
                recv_bar = bars.ds_recv_full
                smem_bar = bars.ds_smem
            else:
                # smem_map rubin_b2 (header sections 5 / 8): tile t stages into sDS / XCHG stage s = sds_state.idx and arms / arrives that stage's slots; the
                # arm's back-edge (P14-B) is this warp's dq_done(t-2) wait of iteration t-1: phase t-2 of slot s completed (bytes landed) before dQ(t-2) issued
                stage = sds_state.idx
                own_sts = own_base + stage * cutlass.Int32(geo.DS_BYTES)
                xchg_sts = cutlass.Int32(geo.XCHG) + stage * cutlass.Int32(geo.XCHG_BYTES)
                recv_bar = bars.ds_recv_full[stage]
                smem_bar = bars.ds_smem[stage]
            _skew_leader_arm(rank)  # skew lever 8: the LEADER holds its ds_recv_full[s] arm (the follower's ship of slot s lands first)
            recv_bar.arrive(n_bytes=xchg_tx, pred=ship_pred)  # XCHG_TX; the probe arms CFG.DIAG_EXCHANGE_OFF_TX = 0 (completes at the arm)
            _store_words(slab, xchg_sts, lane, _concat32(x0, x1))
            nvvm.fence_proxy("async.shared", space="cta")
            named_barrier_fence(CFG.XCHG_BARRIER, CFG.EPILOGUE_LANES)
            _skew_leader_ship(rank)  # skew lever 2 (ONE site for both role shapes): the LEADER holds its XCHG ship (the half is written)
            # DIAGNOSTIC exchange_off (exchange_off): the ship only; the arm above expects 0 bytes, every wait and arrive stays
            if cutlass.const_expr(not exchange_off):
                _ship_xchg(slab, bars, rank, ship_pred, stage, geo, b2)
            if cutlass.const_expr(dbg is not None):
                _clk_ship(dbg, cute.arch.clock64() - t_ds)  # STAMPS: ds_tmem pass -> ship issue (+ the stamp the MMA warps read back)
            _skew_follower_own_half(rank)  # skew lever 1: the FOLLOWER holds its own half (its XCHG is shipped)
            _store_words(slab, own_sts, lane, _concat32(o0, o1))
            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            smem_bar.arrive()
            if cutlass.const_expr(not b2):
                if qtile > lo:
                    # round B of dQ(t-1): slice 2 dhalf + 1
                    empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, carried, empty_state, sts_off, dbg)
                _wait_tcgen(bars.dq_done, dq_state.phase, dbg, _W["wait:epilogue:dq_done"])
                if cutlass.const_expr(dbg is not None):
                    t_dq = cute.arch.clock64()  # STAMPS: dQ(t) landed
                    _clk_add(dbg, _W["seg_dq_done_sum"], cutlass.Int32(t_dq - t_ds))
                # skew lever 6: the FOLLOWER holds its dQ readout (s_tail: s_tail_empty alone holds S^T(t+1); dq_576: a timing probe of the carrier)
                _skew_follower_dq_readout(rank)
                dq = _tmem_load(tmem_base + dq_col, CFG.DQ_COLS)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                # s_tail map: s_tail_empty right after wait(LOAD): S^T(t+1) waits the readout, never the STS.  dq_576: no row -- the wait(LOAD) + fence above
                # are the local half of the reuse edge dQ(t) readout -> dQ(t+1) write; this warp's next-tile ds_smem.arrive() carries it (header section 5)
                if cutlass.const_expr(not dq576):
                    _arrive_leader(bars.s_tail_empty)
                if cutlass.const_expr(dbg is not None):
                    _clk_add(dbg, _W["seg_readout_sum"], cutlass.Int32(cute.arch.clock64() - t_dq))  # STAMPS: dq_done pass -> the readout's arrive
                    _clk_add(dbg, _W["tiles"], cutlass.Int32(1))
                # round A of dQ(t): slice 2 dhalf
                empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, _dq_half(dq, 0), empty_state, sts_off, dbg)
                carried = _dq_half(dq, CFG.DQ_BOX)
                ds_state = advance(ds_state, 1)
                dq_state = advance(dq_state, 1)
            else:
                # the LAGGING order (header section 5; smem_map rubin_b2): dS(t) is staged ABOVE without waiting dQ(t-1); now round B of dQ(t-2) (the
                # carried half), then dQ(t-1): dq_done -> the readout trio -> dq_read_done (LEADER 256: the dQ image's REUSE edge, the leader's gate on
                # its next issue) -> round A; the second half is carried across the next tile's staging (today's register shape).  Both runtime guards
                # are the item's loop-structure predicate (pair-uniform _item_bounds): the counts of section 5 hold on both CTAs
                if qtile - 1 > lo:
                    empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, carried, empty_state, sts_off, dbg)  # round B of dQ(t-2)
                if qtile > lo:
                    _wait_tcgen(bars.dq_done, dq_state.phase, dbg, _W["wait:epilogue:dq_done"])  # dQ(t-1): issued at slot 2 of the leader's tile t
                    if cutlass.const_expr(dbg is not None):
                        t_dq = cute.arch.clock64()  # STAMPS: dQ(t-1) landed
                        _clk_add(dbg, _W["seg_dq_done_sum"], cutlass.Int32(t_dq - t_ds))
                    _skew_follower_dq_readout(rank)  # skew lever 6: the FOLLOWER holds its dQ readout (dq_read_done alone holds the leader's next issue)
                    dq = _tmem_load(tmem_base + dq_col, CFG.DQ_COLS)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    _arrive_leader(bars.dq_read_done)  # LEADER, bare, 128 x 2 = 256: the image is read out; the leader may overwrite it with its next dQ
                    if cutlass.const_expr(dbg is not None):
                        _clk_add(dbg, _W["seg_readout_sum"], cutlass.Int32(cute.arch.clock64() - t_dq))  # STAMPS: dq_done pass -> the readout's arrive
                    # round A of dQ(t-1): slice 2 dhalf
                    empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, _dq_half(dq, 0), empty_state, sts_off, dbg)
                    carried = _dq_half(dq, CFG.DQ_BOX)
                    dq_state = advance(dq_state, 1)
                if cutlass.const_expr(dbg is not None):
                    _clk_add(dbg, _W["tiles"], cutlass.Int32(1))
                ds_state = advance(ds_state, 1)
                sds_state = advance(sds_state, geo.DS_STAGES)
        if cutlass.const_expr(not b2):
            if lo < hi:
                empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, carried, empty_state, sts_off, dbg)  # round B of dQ(hi-1)
            else:
                # the zero-trip empty commit (both epilogues observe it: dq_state +1 per item)
                _wait_tcgen(bars.dq_done, dq_state.phase, dbg, _W["wait:epilogue:dq_done"])
                dq_state = advance(dq_state, 1)
        else:
            if lo < hi:
                # the item end under the lagging order: round B of dQ(hi-2) (the carried half, N >= 2), then the tail dQ(hi-1): dq_done -> the readout trio
                # -> dq_read_done (the next item's first issue, or the leader's drain, consumes it) -> rounds A and B back to back (today's item-end shape)
                if hi - 1 > lo:
                    empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, carried, empty_state, sts_off, dbg)  # round B of dQ(hi-2)
                _wait_tcgen(bars.dq_done, dq_state.phase, dbg, _W["wait:epilogue:dq_done"])  # dQ(hi-1): the tail issue
                _skew_follower_dq_readout(rank)
                dq = _tmem_load(tmem_base + dq_col, CFG.DQ_COLS)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                _arrive_leader(bars.dq_read_done)  # LEADER, bare, 256: the item's last readout
                empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, _dq_half(dq, 0), empty_state, sts_off, dbg)  # round A of dQ(hi-1)
                empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, _dq_half(dq, CFG.DQ_BOX), empty_state, sts_off, dbg)  # round B of dQ(hi-1)
            else:
                # the zero-trip empty commit (both epilogues observe it: dq_state +1 per item); no readout, no dq_read_done flip (the leader issued no dQ)
                _wait_tcgen(bars.dq_done, dq_state.phase, dbg, _W["wait:epilogue:dq_done"])
            dq_state = advance(dq_state, 1)
        live = lo < hi
        _skew_follower_item_end(rank)  # skew lever 3: the FOLLOWER holds its dV / dK TMEM readout at the item end
        if cutlass.const_expr(dkv_fp32):
            # dkv_accum = fp32_reduce (header section 11): round 1 (dV 4 + dK 0) -> dv_full -> dvk_stage_empty wait -> round 2 (dK 1-3) -> dk_full
            stage_state = _dkv_fp32_stage(slab, bars, geo, tmem_base, lane, live, stage_state, dbg)
        else:
            _native_out(slab, tmem_base, lane, live, CFG.DV_COL, (geo.dv_store(0), geo.dv_store(1)), dtype)
            bars.dv_full.arrive()
            _native_out(slab, tmem_base, lane, live, CFG.DK_COL, (own_base, geo.XCHG), dtype)  # geo.dk_store(rank, half): the STAGE-0 own half + XCHG stage 0
            bars.dk_full.arrive()
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:epilogue:sched_full"])


@cute.jit
def _mma_loop_leader(
    slab,
    bars,
    sched,
    tmem_base,
    pair,
    head,
    batch,
    meta,
    n_batch,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_2cta_code: cutlass.Constexpr[int],
    fa4_roles: cutlass.Constexpr[bool],
    tmem_map_code: cutlass.Constexpr[int],
    smem_map_code: cutlass.Constexpr[int],
    geo: cutlass.Constexpr,
    regs: cutlass.Constexpr[int],
    dbg,
):
    """Warp 12 on the pair LEADER: the five cta_group::2 GEMMs in FAv4's rotated order (header section 5 MMA ISSUE ORDER
    False = fa4, the default: dP^T(t), dQ(t-1), dV(t), S^T(t+1), dK(t); True = fa4b: dP^T(t), dV(t), dQ(t-1), S^T(t+1), dK(t) -- the in-tile
    dQ slot one GEMM later, the same waits / guards / counts, ONE helper `_dq_slot` at two const_expr sites); every commit multicast to the
    pair.  elect_p = nvvm.elect_sync() once per item.
    ``diag_2cta_code`` (DIAGNOSTIC, header section 11): DIAG_S_TAIL_WAIT_OFF (s_tail_wait_off) = the CFG.DIAG_S_TAIL_WAIT_SITES s_tail_empty waits
    of this warp (the prologue, the lookahead in `_s_lookahead`, the tail, the P15 drain) are skipped under `s_tail_off`; DIAG_S_READ_WAIT_OFF
    (s_read_wait_off) = the s_read wait in `_dq_slot` is skipped under `s_read_off`; DIAG_S_TAIL_AND_S_READ_WAIT_OFF = both flags.  Every
    s_tail_state / ready_state advance, every commit and every follower arrive stays: both rows are LEADER relaxed thread-count rows without tx
    bytes, so the unobserved phases flip without an over-arrival fault (section 5); OUTPUT WRONG BY DESIGN (the ceiling of the S-tail alias
    edges).  No other probe reaches this warp: the dp_empty-wait probe is declined BY NAME on this map and the gate-skip probe (`ds_gates_off`)
    is RETIRED here -- the three dS gates are the ds_recv_full arm's only back-edge (_dq_issue; the barrier audit).  ``dbg`` (STAMPS; DbgLeader): every wait of
    this warp bracketed by %clock64 into its own word + the blocked count, the tile count, the ship -> recv handshake.

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])
    #   if lo < hi: _wait_tcgen(bars.k1_full, kv.phase); _wait_tcgen(bars.q1_full[q1.idx], q1.phase);
    #               _wait_tcgen(bars.s_tail_empty, s_tail.phase) [ALWAYS: the previous item's tail dQ readout / pre-armed, D1 i] -> s_tail += 1
    #               -> S^T(lo) = mma_ss(M 256) -> _commit_pair(bars.s_done, elect_p); _commit_pair(bars.q1_empty[q1.idx], elect_p)
    #               _wait_tcgen(bars.v_full, kv.phase); _wait_tcgen(bars.k2_full, kv.phase)
    #   _wait_tcgen(bars.dvk_empty, out_empty.phase); out_empty = advance(out_empty, 1)   [pre-armed st(1); EVERY item, zero-trip too]
    # BARRIER (per tile t):
    #   1. _wait_tcgen(bars.dp_empty, dpe.phase)  [st(1): the dS^T(t-1) readout] -> _wait_tcgen(bars.do1_full[0], do1.phase) -> dP^T k-half 0
    #      -> _wait_tcgen(bars.do1_full[1], do1.phase) -> dP^T k-half 1 -> _commit_pair(dp_done); _commit_pair(do1_empty)
    #   2. _dq_slot = _wait_tcgen(bars.s_read, ready.phase)  [EVERY tile: both softmaxes hold S^T(t) cols 64..127 in registers]
    #      if qtile > lo: _dq_issue = _wait_tcgen(ds_smem, xchg.phase); _wait_tcgen(ds_recv_full, xchg.phase); _wait_tcgen(ds_pair_ready, xchg.phase)
    #         -> dQ(t-1) = mma_ss(M 128, K 256: A = sDS [256 kv x own 64 q] MN-major, B = K2) into [DQ_COL, +64) -> _commit_pair(dq_done) -> xchg += 1
    #   3. _wait_tcgen(bars.p_ready, ready.phase) -> _wait_tcgen(bars.do2_full, do2.phase) -> dV TS (accumulate = qtile > lo) -> _commit_pair(do2_empty)
    #   4. q1_next = advance(q1, Q1_STAGES); _s_lookahead: if t + 1 < hi: _wait_tcgen(q1_full[q1_next.idx]); if t > lo: _wait_tcgen(s_tail_empty,
    #      s_tail.phase) -> s_tail += 1 [D1 ii: dQ(t-1)'s readout]; S^T(t+1) -> _commit_pair(s_done); _commit_pair(q1_empty[q1_next.idx])
    #   5. _wait_tcgen(bars.ds_ready, ready.phase) -> _wait_tcgen(bars.q2_full, q2.phase) -> dK TS -> _commit_pair(q2_empty)
    #   q1 = q1_next; do1, do2, q2, ready, dpe = advance(.., 1)
    # BARRIER (block end): if lo < hi: if hi - 1 > lo: _wait_tcgen(bars.s_tail_empty, s_tail.phase) -> s_tail += 1 [D1 iii: dQ(hi-2)'s readout]
    #                                  _dq_issue (the three gates on xchg, dQ(hi-1), dq_done, xchg += 1); _commit_pair(k1_empty); _commit_pair(v_empty); kv += 1
    #                      else: _commit_pair(dq_done)   [the zero-trip empty commit on a live pipe: both epilogues wait it]
    # BARRIER (after the persistent loop, P15): _wait_tcgen(bars.dp_empty, dpe.phase); _wait_tcgen(bars.dvk_empty, out_empty.phase);
    #   _wait_tcgen(bars.s_tail_empty, s_tail.phase)   [the last item's tail readout]; then the final barrier_cta_sync and _tmem_dealloc_pair.
    # BARRIER under fa4_roles (16w_fa4; x-fa4 table section 15): `_dq_issue` commits dq_war_done IMMEDIATELY BEFORE dq_done at BOTH dQ issue sites (the
    #   softmax's WAR signal); the zero-trip site commits dq_done ONLY (unchanged).
    # BARRIER under the WAIT probes (section 11; s_tail_off / s_read_off are trace-time flags from diag_2cta_code): the four s_tail_empty
    #   `_wait_tcgen` above (D1 i / ii / iii + the P15 drain) and / or the slot-2 s_read `_wait_tcgen` are NOT emitted; every `s_tail += 1`,
    #   `ready += 1`, every _commit_pair and the follower's arrives are emitted exactly as above (the flip counts of section 5 hold).
    # BARRIER under fa4_roles (16w_fa4; x-fa4 table X5 / section 6): dp_empty is DELETED -- no slot-1 wait, no dpe state, no P15 drain; dP^T(t+1)'s
    #   overwrite of the packed dS^T(t) is ordered by the in-order tensor pipe after dK(t) (issued first by this thread), of dP^T(t)'s fp32 by the
    #   softmax's ld + wait(LOAD) -> ds_ready(t) -> this warp's dK(t) wait -> dK(t) -> dP^T(t+1).  Every other line of this role is unchanged.
    # BARRIER under tmem_map dq_576 (``tmem_map_code`` == TMEM_MAP_DQ576; header sections 5 / 9 / 11; the barrier skeleton R1): s_read AND s_tail_empty
    #   are DELETED -- no s_tail state, no D1 (i) / (ii) / (iii) wait, no slot-2 s_read wait, no P15 drain; the dQ GEMM's D image is
    #   [CFG_DQ576.DQ_COL, +64) past dK; the REUSE edge dQ(t) readout -> dQ(t+1) write is the three dS gates of t+1 (ds_smem on this CTA, ds_pair_ready
    #   through the follower's ds_smem + relay), which the epilogues arrive AFTER their dQ(t) readout by program order.  Every other wait, commit and
    #   guard of this role is unchanged; the S-tail WAIT probes decline by name under the map (no site).
    # BARRIER under smem_map rubin_b2 (``smem_map_code`` == SMEM_MAP_RUBIN_B2, ON dq_576; header sections 5 / 8 / 11; the barrier skeleton R2.c): the rings'
    #   slots follow the geometry -- q1_full / q1_empty [q1.idx] (Q1_STAGES from the geometry record: the lookahead's q1_next = advance(q1, geo.Q1_STAGES)), do1_full [2 do1.idx + half] with
    #   the sDO1 stage descriptor built per tile (DO1_STAGES 2; do1 = advance(do1, 2)), the three dS gates [xchg.idx] with the dS A descriptor advanced by
    #   (xchg.idx x DS_BYTES) >> 4 (DS_STAGES 2; xchg = advance(xchg, 2)); the FOURTH gate of every dQ issue (in-tile and tail) is _wait_tcgen(
    #   bars.dq_read_done, dq_read.phase) -> dq_read += 1 (st(1) pre-armed: the kernel's first issue passes; wait w <-> the readout of issue w-1);
    #   P15 after the loop: + _wait_tcgen(bars.dq_read_done, dq_read.phase) ONCE (the last readout of the last live item; a leader without a live item
    #   passes on the pre-arm).  Zero-trip: no issue, no wait, no advance (the empty dq_done commit touches no ring).
    """
    # the issue order: dP^T(t), dQ(t-1), dV(t), S^T(t+1), dK(t); tail dQ(hi-1); xchg_state +1 per dQ issue
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.SERVICE_REGS
    # DIAGNOSTIC wait probes (header section 11): trace-time flags; False under diag_2cta_code 0 = the shipped rendering (byte-identical)
    s_tail_off = cutlass.const_expr(diag_2cta_code == DIAG_S_TAIL_WAIT_OFF or diag_2cta_code == DIAG_S_TAIL_AND_S_READ_WAIT_OFF)
    s_read_off = cutlass.const_expr(diag_2cta_code == DIAG_S_READ_WAIT_OFF or diag_2cta_code == DIAG_S_TAIL_AND_S_READ_WAIT_OFF)
    dq576 = cutlass.const_expr(tmem_map_code == TMEM_MAP_DQ576)  # the TMEM map (header section 9): the S-tail rows and their sites exist under s_tail only
    b2 = cutlass.const_expr(smem_map_code == SMEM_MAP_RUBIN_B2)  # the SMEM map (header section 8): the rings' slots, the stage descriptors, dq_read_done
    score = _mma_desc_kv(dtype)
    grad = _mma_desc_kv(dtype, btranspose=True)
    dqdesc = _mma_desc_dq(dtype)
    s_acc = nvvm.make_tmem_ptr(tmem_base + CFG.S_COL, cutlass.Float32)
    dp_acc = nvvm.make_tmem_ptr(tmem_base + CFG.DP_COL, cutlass.Float32)
    dq_acc = nvvm.make_tmem_ptr(tmem_base + _dq_col(tmem_map_code), cutlass.Float32)  # the dQ D image: S^T's dead tail (s_tail) | past dK (dq_576)
    dv_acc = nvvm.make_tmem_ptr(tmem_base + CFG.DV_COL, cutlass.Float32)
    dk_acc = nvvm.make_tmem_ptr(tmem_base + CFG.DK_COL, cutlass.Float32)
    q1_state = PipelineState.start()
    q2_state = PipelineState.start()
    do1_state = PipelineState.start()
    do2_state = PipelineState.start()
    kv_state = PipelineState.start()
    ready_state = PipelineState.start()
    xchg_state = PipelineState.start()  # the P2 split: +1 per dQ ISSUE (one tile behind ready_state); advances by geo.DS_STAGES (the ring's slots)
    if cutlass.const_expr(not dq576):
        s_tail_state = PipelineState.start(phase=1)  # pre-armed: the kernel's first S^T(lo) finds the S tail free
    else:
        s_tail_state = None  # dq_576: no S-tail row, no state; `_s_lookahead` threads the None through unchanged (no site reads it)
    if cutlass.const_expr(b2):
        dq_read_state = PipelineState.start(phase=1)  # rubin_b2: dq_read_done, pre-armed (the kernel's first dQ issue has no previous tenant); +1 per wait
    else:
        dq_read_state = None  # sm100: no row, no wait; `_dq_slot` / `_dq_issue` thread the None through unchanged
    if cutlass.const_expr(not fa4_roles):
        dp_empty_state = PipelineState.start(phase=1)
    out_empty = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        elect_p = nvvm.elect_sync()
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        # The 1-stage slabs have STATIC addresses: pin their descriptor bases per item (desc_opaque + the item anchor), or LLVM hoists
        # every `base + k_step` add to the persistent loop's preheader and the 80-register MMA warp spills (the d256 2x2 lesson).
        k1d = desc_opaque(_operand_k_major(slab, geo.K1, dtype, CFG.M).desc(), anchor=pair)
        vd = desc_opaque(_operand_k_major(slab, geo.V, dtype, CFG.M).desc(), anchor=pair)
        k2d = desc_opaque(_operand_mn_major(slab, geo.K2, dtype, CFG.PAIR_KV_ROWS).desc(), anchor=pair)
        do1d = desc_opaque(_operand_k_major(slab, geo.DO1, dtype, CFG.Q_HALF).desc(), anchor=pair)  # sDO1 stage 0, pinned per item (rubin_b2: per tile below)
        do2d = desc_opaque(_operand_mn_major(slab, geo.DO2, dtype, CFG.N).desc(), anchor=pair)
        q2d = desc_opaque(_operand_mn_major(slab, geo.Q2, dtype, CFG.N).desc(), anchor=pair)
        dsd = desc_opaque(_ds_operand(slab, geo.DS, dtype).desc(), anchor=pair)  # the sDS stage-0 root (rubin_b2: advanced per issue inside _dq_issue)
        if lo < hi:
            _wait_tcgen(bars.k1_full, kv_state.phase, dbg, _W["wait:leader:k1_full"])
            _wait_tcgen(bars.q1_full[q1_state.idx], q1_state.phase, dbg, _W["wait:leader:q1_full"])
            if cutlass.const_expr(not dq576):
                # each S^T issue waits s_tail_empty (dQ(t-1) read out of the S tail); st(1)
                # D1 (i): the previous item's tail dQ(hi_prev - 1) has been read out of [64, 128)
                # DIAGNOSTIC s_tail_wait_off: the guard holds the wait only; the advance stays
                if cutlass.const_expr(not s_tail_off):
                    _wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])
                s_tail_state = advance(s_tail_state, 1)
            q1d = _operand_k_major(slab, geo.Q1 + q1_state.idx * geo.Q1_BYTES, dtype, CFG.Q_HALF).desc()
            mma_ss(score, k1d, q1d, s_acc)
            _commit_pair(bars.s_done, elect_p)
            _commit_pair(bars.q1_empty[q1_state.idx], elect_p)
            _wait_tcgen(bars.v_full, kv_state.phase, dbg, _W["wait:leader:v_full"])
            _wait_tcgen(bars.k2_full, kv_state.phase, dbg, _W["wait:leader:k2_full"])
        _wait_tcgen(bars.dvk_empty, out_empty.phase, dbg, _W["wait:leader:dvk_empty"])
        out_empty = advance(out_empty, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            if cutlass.const_expr(dbg is not None):
                _clk_add(dbg, _W["tiles_leader"], cutlass.Int32(1))  # STAMPS: this warp's tile count (the handshake's tag base)
            # slot 1: dP^T(t) -- 16w: gated by the dS^T(t-1) READOUT (dp_empty's moved arrive), never by a dQ readout; 16w_fa4: dp_empty OUT
            _skew_leader_slot1()  # skew lever 9: the LEADER holds slot 1 of tile t (relay(t) + the staging of t land before flip t-1 is consumed)
            if cutlass.const_expr(not fa4_roles):
                _wait_tcgen(bars.dp_empty, dp_empty_state.phase, dbg, _W["wait:leader:dp_empty"])
            # dp_empty OUT: packed dS^T(t) read by dK(t) in-order; dP^T(t) fp32 by ds_ready(t)
            if cutlass.const_expr(not b2):
                do1d_t = do1d  # the pinned single stage
            else:
                do1d_t = _operand_k_major(
                    slab, geo.DO1 + do1_state.idx * geo.DO1_BYTES, dtype, CFG.Q_HALF
                ).desc()  # the sDO1 ring stage of tile t (DO1_STAGES 2)
            _wait_tcgen(_do1_slot(bars, do1_state, 0, geo.DO1_STAGES), do1_state.phase, dbg, _W["wait:leader:do1_full"])
            mma_ss(score, vd, do1d_t, dp_acc, k_start=0, k_count=CFG.K_STEPS_PER_BOX)
            _wait_tcgen(_do1_slot(bars, do1_state, 1, geo.DO1_STAGES), do1_state.phase, dbg, _W["wait:leader:do1_full"])
            mma_ss(score, vd, do1d_t, dp_acc, k_start=CFG.K_STEPS_PER_BOX, k_count=CFG.K_STEPS_PER_BOX, accumulate=True)
            _commit_pair(bars.dp_done, elect_p)
            _commit_pair(_slot(bars.do1_empty, do1_state, geo.DO1_STAGES), elect_p)
            # slot 2 (fa4): dQ(t-1) into S^T's tail -- after both softmaxes read S^T(t) (s_read, every tile; s_tail map), behind the exchange gates of t-1
            xchg_state, dq_read_state = _dq_slot(
                bars, dqdesc, dsd, k2d, dq_acc, xchg_state, dq_read_state, ready_state, elect_p, qtile, lo, fa4_roles, s_read_off, dq576, b2, geo, dbg
            )
            # slot 3: dV(t) -- reads P^T(t) at [P_COL, +PACK_COLS), disjoint from the dQ image [DQ_COL, +DQ_COLS)
            _wait_tcgen(bars.p_ready, ready_state.phase, dbg, _W["wait:leader:p_ready"])
            _wait_tcgen(bars.do2_full, do2_state.phase, dbg, _W["wait:leader:do2_full"])
            _ts(grad, tmem_base + CFG.P_COL, do2d, dv_acc, qtile > lo)
            _commit_pair(bars.do2_empty, elect_p)
            # slot 4: S^T(t+1) -- LEADS as in FAv4; waits dQ(t-1)'s readout (s_tail_empty; s_tail map) only when a dQ was issued in this tile
            q1_next = advance(q1_state, geo.Q1_STAGES)  # the sQ1 ring's depth is the GEOMETRY's (1 on sm100, 2 under rubin_b2): the loader's walk
            s_tail_state = _s_lookahead(slab, bars, score, k1d, s_acc, q1_next, s_tail_state, elect_p, qtile, lo, hi, dtype, s_tail_off, dq576, geo, dbg)
            # slot 5: dK(t)
            _wait_tcgen(bars.ds_ready, ready_state.phase, dbg, _W["wait:leader:ds_ready"])
            _wait_tcgen(bars.q2_full, q2_state.phase, dbg, _W["wait:leader:q2_full"])
            _ts(grad, tmem_base + CFG.DS_COL, q2d, dk_acc, qtile > lo)
            _commit_pair(bars.q2_empty, elect_p)
            q1_state = q1_next
            do1_state = advance(do1_state, geo.DO1_STAGES)
            do2_state = advance(do2_state, 1)
            q2_state = advance(q2_state, 1)
            ready_state = advance(ready_state, 1)
            if cutlass.const_expr(not fa4_roles):
                dp_empty_state = advance(dp_empty_state, 1)
        if lo < hi:
            if cutlass.const_expr(not dq576):
                # tail: dQ(hi-1) -- the S tail holds the UNREAD dQ(hi-2) image iff a dQ was issued in the loop (N >= 2): wait its readout (D1 iii);
                # at N == 1 the previous tenant is S^T(lo), whose softmax read is implied by the dS gates (lo) on both CTAs
                # DIAGNOSTIC s_tail_wait_off: the guard holds the wait only; the advance stays (D1 iii)
                if hi - 1 > lo:
                    if cutlass.const_expr(not s_tail_off):
                        _wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])
                    s_tail_state = advance(s_tail_state, 1)
            # dq_576: the tail dQ(hi-1)'s previous tenant (dQ(hi-2), or the previous item's last dQ at N == 1) was read out before the epilogues
            # arrived the dS gates of hi-1 (program order): the three gates inside _dq_issue carry the edge
            xchg_state, dq_read_state = _dq_issue(
                bars, dqdesc, dsd, k2d, dq_acc, xchg_state, dq_read_state, elect_p, fa4_roles, b2, geo, dbg, DBG_CLK.TAG_BACK_TAIL
            )
            _commit_pair(bars.k1_empty, elect_p)
            _commit_pair(bars.v_empty, elect_p)
            kv_state = advance(kv_state, 1)
        else:
            _commit_pair(bars.dq_done, elect_p)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:leader:sched_full"])
    # P15: the follower's last relaxed arrives (dp_empty of the last dS^T readout, dvk_empty of the last item, s_tail_empty of the last tail
    # readout) land on a resident leader.
    if cutlass.const_expr(not fa4_roles):
        _wait_tcgen(bars.dp_empty, dp_empty_state.phase, dbg, _W["wait:leader:dp_empty"])
    _wait_tcgen(bars.dvk_empty, out_empty.phase, dbg, _W["wait:leader:dvk_empty"])
    if cutlass.const_expr(b2):
        # P15 (rubin_b2): the last readout's dq_read_done (both epilogues' relaxed arrives of the last live item's dQ(hi-1); the pre-arm on a leader
        # without a live item) lands on a resident leader; the state carries W waits for W issues, so the drain consumes flip W exactly
        _wait_tcgen(bars.dq_read_done, dq_read_state.phase, dbg, _W["wait:leader:dq_read_done"])
    # dq_576: no s_tail_empty row, no drain (dp_empty's and dvk_empty's drains above stay)
    if cutlass.const_expr(not dq576):
        # DIAGNOSTIC s_tail_wait_off: the P15 drain is a wait site too -- skipped under the probe (the follower's last arrive lands on a resident
        # leader regardless: the final barrier_cta_sync + the pair-symmetric dealloc wait follow)
        if cutlass.const_expr(not s_tail_off):
            _wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])


@cute.jit
def _mma_loop_follower(
    bars, sched, pair, head, batch, meta, n_batch, causal: cutlass.Constexpr[bool], geo: cutlass.Constexpr, regs: cutlass.Constexpr[int], dbg
):
    """Warp 12 on the FOLLOWER: the quiet body + the dS relay.  Never issues an MMA, never arms expect_tx, never waits a leader copy
    (q*_full / do*_full / k*_full / v_full / p_ready / ds_ready / s_read / dp_empty / s_tail_empty / dvk_empty / ds_pair_ready are
    LEADER-only waits).

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])   [this warp is one of this CTA's 15 credit warps]
    # BARRIER (per tile t, relay st(0)): _wait(bars.ds_smem, relay.phase); _wait(bars.ds_recv_full, relay.phase)
    #   -> bars.ds_pair_ready.arrive_on_peer(cutlass.Int32(LEADER_RANK), pred=elect_p, release_cta=True)
    #      [1 lane -> the leader's copy == init 1; the cumulative .release.cta publishes our epilogue's fenced own-half stores to the
    #       leader-issued cta_group::2 dQ MMA (P8 / P12); relaxed would NOT]
    #   relay = advance(relay, 1)   [the leader consumes it one tile later (xchg_state) or at the tail: the counts are the same, N per item]
    # Zero-trip item: credit only.  No drain: the leader waits every ds_pair_ready phase in-loop.  Teardown: _tmem_dealloc_pair.
    # ``dbg`` (STAMPS): the relay's two wait sums, its tile count and the ship -> recv handshake after the ds_recv_full wait (_clk_recv).
    # smem_map rubin_b2 (header sections 5 / 8): the three rows have geo.DS_STAGES = 2 slots; relay_state advances by 2 (one flip per live tile on slot
    # t % 2, the same running count as the epilogue's sds_state and the leader's xchg_state: P2 lockstep); the relay of slot s publishes the follower's
    # fenced own-half STS of stage s to the leader's dQ MMA of that stage (P8 / P12).
    """
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.SERVICE_REGS
    relay_state = PipelineState.start()
    sched_state = PipelineState.start()
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        elect_p = nvvm.elect_sync()
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        for _ in cutlass.range(lo, hi, 1, unroll=1):
            if cutlass.const_expr(dbg is not None):
                _clk_add(dbg, _W["tiles_follower"], cutlass.Int32(1))
            _wait(_slot(bars.ds_smem, relay_state, geo.DS_STAGES), relay_state.phase, dbg, _W["wait:follower:ds_smem"])
            _wait(_slot(bars.ds_recv_full, relay_state, geo.DS_STAGES), relay_state.phase, dbg, _W["wait:follower:ds_recv_full"])
            if cutlass.const_expr(dbg is not None):
                _clk_recv(dbg, _W["tiles_follower"], _W["recv_follower_sum"], _W["recv_follower_matched"], _W["recv_follower_missed"], DBG_CLK.TAG_BACK_RELAY)
            _slot(bars.ds_pair_ready, relay_state, geo.DS_STAGES).arrive_on_peer(cutlass.Int32(LEADER_RANK), pred=elect_p, release_cta=True)
            relay_state = advance(relay_state, geo.DS_STAGES)
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:follower:sched_full"])


@cute.jit
def _reduce_loop(
    slab,
    bars,
    sched,
    tmem_base,
    pair,
    head,
    batch,
    meta,
    n_batch,
    rank,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_2cta_code: cutlass.Constexpr[int],
    regs: cutlass.Constexpr[int],
    geo: cutlass.Constexpr,
):
    """Warps 8-11 under warp_roles 16w_fa4 (the alternate role shape): the REDUCE group = FAv4's dQ readout warps on the S-tail body (lane = q row of d half
    (warp // 2) % 2 for dQ; lane = own kv row for the item-end dV / dK staging).  No dS work: the softmax writes sDS own + XCHG and ships.
    ``diag_2cta_code`` (DIAGNOSTIC): DIAG_SDQ_STS_OFF (sdq_sts_off) = the two `_dq_round` sites skip the sDQ slice STS + its proxy fence .  STAMPS (`debug_clk`) declines under 16w_fa4: every wait here is passed dbg=None (untimed).
    `dkv_accum = "fp32_reduce"` declines under 16w_fa4 too: this group stages bf16 halves only (`_native_out`); dvk_stage_empty has no field here.

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])
    # BARRIER (per tile t; dq_state st(0), empty_state st(1); the barrier audit x-fa4 table section 9, the reduce stub order):
    #   1. _wait_tcgen(bars.dq_done, dq_state.phase)   [the AUDIT's K:1144 / an earlier tree K:1277 site STAYS on warps 8-11]
    #      -> [test-only EPILOGUE_DQ_READOUT sleep, FOLLOWER only] -> dq = tcgen05.ld x64 @ CFG.DQ_COL -> tcgen05_wait(LOAD)
    #      -> tcgen05_fence(BEFORE_THREAD_SYNC) -> _arrive_leader(bars.s_tail_empty)   [LEADER, bare, 128 x 2 = 256; BEFORE round A (X8)]
    #   2. round A: _dq_round(dq[0, 32)) = _wait(dq_empty[0]) + _wait(dq_empty[1]) at empty_state.phase -> STS slot dhalf -> fence_proxy
    #      -> bars.dq_full[dhalf].arrive()   [THREAD local, 2 warps = 64 == DQ_SLICE_LANES] -> empty_state += 1
    #   3. round B: _dq_round(dq[32, 64)) -- back-to-back inside the tile: no `carried` half, no item-end round (the resource plan)
    #   4. dq_state = advance(dq_state, 1)
    # BARRIER (item end): if lo >= hi: _wait_tcgen(bars.dq_done, dq_state.phase) + advance   [the zero-trip empty commit; both CTAs' groups observe it]
    #   -> [test-only FOLLOWER_EPILOGUE_ITEM_END sleep, FOLLOWER only] -> native dV over K2 -> bars.dv_full.arrive() [128 of 160: + the loader's 32, fix A]
    #   -> native dK over ds_own(rank) + XCHG -> bars.dk_full.arrive() [128]   (X9: the reduce group keeps the staging; the store warp keeps the TMA stores)
    # Flips: dq_done N (+1 zero-trip) observed; dq_empty / dq_full 2N / 2N per slot (A then B per tile); s_tail_empty N; dv_full / dk_full 1 per item.
    # No drain: every arrive of this role is local or waited by the leader in-loop (s_tail_empty's last phase is the leader's P15 drain).
    """
    # reduce group: dq_done -> ld x64 -> s_tail_empty -> rounds A, B; no carried half
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.INCREASE)  # CFG.EPILOGUE_REGS (the same 144)
    lane = cute.arch.thread_idx()[0] % CFG.M  # own kv row for the dV / dK staging; TMEM lane of the dQ half-row image
    qrow = lane % CFG.Q_HALF  # dQ row of the half-row image
    dhalf = lane // CFG.Q_HALF  # d half this warp pair holds (warp-uniform) = its sDQ slot
    own_base = cutlass.Int32(geo.DS) + rank * cutlass.Int32(CFG.DS_HALF_BYTES)  # geo.ds_own(rank): the dK staging's first alias half
    slot_base = cutlass.Int32(geo.DQ) + dhalf * cutlass.Int32(CFG.DQ_SLICE_BYTES)  # geo.dq_slot(dhalf): the warp pair's slot of the ring
    dq_state = PipelineState.start()
    empty_state = PipelineState.start(phase=1)  # pre-armed (P5 b): both slots of the ring are free before the first round; NO bootstrap (D3)
    sched_state = PipelineState.start()
    sts_off = cutlass.const_expr(diag_2cta_code == DIAG_SDQ_STS_OFF)  # DIAGNOSTIC sdq_sts_off: the sDQ slice STS + its proxy fence skipped in every round
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        for _ in cutlass.range(lo, hi, 1, unroll=1):
            _wait_tcgen(bars.dq_done, dq_state.phase, None, 0)
            _skew_follower_dq_readout(rank)  # skew lever 6 (16w_fa4 site): the FOLLOWER holds its dQ readout (s_tail_empty alone holds S^T(t+1))
            dq = _tmem_load(tmem_base + geo.DQ_COL, CFG.DQ_COLS)  # 16w_fa4 serves the s_tail map only (tmem_map dq_576 declines by name): geo.DQ_COL = 64
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            # s_tail_empty right after wait(LOAD): S^T(t+1) waits the readout, never the STS
            _arrive_leader(bars.s_tail_empty)
            # round A of dQ(t): slice 2 dhalf; round B: slice 2 dhalf + 1 (the tma route only; the sdq_sts_off probe's flag; no stamps under 16w_fa4)
            empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, _dq_half(dq, 0), empty_state, sts_off, None)
            empty_state = _dq_round(slab, bars, slot_base, dhalf, qrow, _dq_half(dq, CFG.DQ_BOX), empty_state, sts_off, None)
            dq_state = advance(dq_state, 1)
        if lo >= hi:
            _wait_tcgen(bars.dq_done, dq_state.phase, None, 0)  # the zero-trip empty commit (both reduce groups observe it: dq_state +1 per item)
            dq_state = advance(dq_state, 1)
        live = lo < hi
        _skew_follower_item_end(rank)  # skew lever 3 (16w_fa4 site): the FOLLOWER holds its dV / dK TMEM readout at the item end
        _native_out(slab, tmem_base, lane, live, CFG.DV_COL, (geo.dv_store(0), geo.dv_store(1)), dtype)
        bars.dv_full.arrive()
        _native_out(slab, tmem_base, lane, live, CFG.DK_COL, (own_base, geo.XCHG), dtype)
        bars.dk_full.arrive()
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, None, 0)


@cute.jit
def _load_loop(
    slab,
    bars,
    sched,
    maps,
    pair,
    head,
    batch,
    meta,
    n_batch,
    rank,
    is_leader,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    regs: cutlass.Constexpr[int],
    stats_form_code: cutlass.Constexpr[int],
    geo: cutlass.Constexpr,
    stats2,
    rowoff,
    dbg,
):
    """Warp 13, BOTH CTAs: Q1 / Q2 / dO1 / dO2 per q tile, K1 / K2 / V per kv block (grouped head // ratio), each CTA its own boxes.
    Unchanged from the first body except the Q1 ring depth (Q1_STAGES = 1: one slot, st(1) adv 1, drain x 1).  ``dbg`` (STAMPS): the
    *_empty wait sums (the drains included) and sched_full.
    ``stats_form_code`` (header section 11): under STATS_FORM_TMA_RING this warp ALSO feeds the softmax's stats ring: per q tile, right after Q1(t),
    ONE 1 KiB `bulk_copy` of the (head, q tile) record of the chain's padded pre-scaled stats region ``stats2`` (fp32 [heads][PAD_TOTAL / N][2][N]:
    the LSE row then the delta row; the record's padded token base = ``rowoff[batch]``, one word per item) into ring stage s behind `stats_full[s]`
    (expect_tx CFG.STATS_TX = 1024 on THIS CTA's copy: a plain bulk copy lands on the issuing CTA) after the `stats_empty[s]` credit; ``stats2`` /
    ``rowoff`` are None (dropped) under ldg.

    # BARRIER under stats_form tma_ring (the S1 skeleton 2026-10-27 17:30 / 17:46, resources/s1-barrier-table.md sections 1 / 7.1 / 14):
    #   stats_ld = PipelineState.start(phase=1)  [PRE-ARMED, CFG_RING.STATS_STAGES = 2: wait k <-> the softmax's release of tile k-2 = the stage's
    #     previous tenant; the kernel's first two waits pass on the pre-arm; declared with the other states, CARRIED across items, never reset]
    #   per item: own_pred = nvvm.elect_sync() [beside arm_pred; BOTH CTAs arm their OWN copy]; rowoff_b = rowoff[batch] (one meta word per item)
    #   per tile t, after Q1(t)'s tma_load_tile and BEFORE _wait(bars.do1_empty):
    #     _wait(bars.stats_empty[stats_ld.idx], stats_ld.phase) -> bars.stats_full[stats_ld.idx].arrive(n_bytes=CFG.STATS_TX, pred=own_pred) [1 == ONE_LANE;
    #     1024 == the record] -> bulk_copy(stage stats_ld.idx, stats2 record (head, rowoff_b / N + t), CFG.STATS_TX, bars.stats_full[stats_ld.idx].smem_ptr)
    #     [internal elect; cp.async.bulk.shared::cluster.global complete_tx on OUR mbar: never cta_group 2, never multicast]
    #   stats_ld = advance(stats_ld, CFG_RING.STATS_STAGES) with the other advances.  Zero-trip item: nothing (the tile loop runs 0 times); no drain.

    # arm_pred = is_leader & nvvm.elect_sync()  (P9: a cga2 tensor load completes on the pair LEADER's mbar whoever issued it; the
    # leader arms BOTH CTAs' bytes = the *_TX constants; the follower NEVER arms).  tma_mask = cutlass.Int16(1) << rank (self-only)
    # under the multicast form, None under ``tma_load_plain`` (header section 10 LOAD FORM); every load: tma_load_tile / tma_load_subtiles(...,
    # bars.X_full[..].smem_ptr, cta_group=CFG.CTA_MMA, mcast_mask=tma_mask).
    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx])   [then the test-only DEBUG_LOADER_DELAY_US sleep, default none]
    #   if lo < hi: acquire the maps; _wait(bars.k1_empty, kv.phase) -> bars.k1_full.arrive(n_bytes=CFG.K1_TX, pred=arm_pred) -> K1
    #               (2 boxes, own 128 rows); _wait(bars.v_empty, kv.phase) -> v_full.arrive(CFG.V_TX, pred=arm_pred) -> V;
    #               _wait(bars.k2_empty, k2e.phase) -> bars.dv_full.arrive() [bare 32, fix A] -> k2_full.arrive(CFG.K2_TX, pred=arm_pred)
    #               -> K2 (two [128 kv x 64 d] boxes at d = 64 * rank, the PAIR's rows); kv = advance(kv, 1)
    #   else (zero-trip): _wait(bars.k2_empty, k2e.phase) -> bars.dv_full.arrive()   [the same single observation on every path, P14]
    #   k2e = advance(k2e, 1)
    # BARRIER (per tile t):
    #   _wait(bars.q1_empty[s], q1.phase) -> q1_full[s].arrive(CFG.Q1_TX, pred=arm_pred) -> Q1 (rows 64 * rank .., 2 boxes)   [s = 0: ONE stage]
    #   _wait(bars.do1_empty, do1.phase) -> for half in range_constexpr(2): do1_full[half].arrive(CFG.DO1_HALF_TX, pred=arm_pred)
    #   -> tma_load_subtiles(sDO1, .., bars.do1_full[half].smem_ptr, half, 1, cta_group=2, mcast_mask=tma_mask) x 2
    #   _wait(bars.do2_empty, do2.phase) -> do2_full.arrive(CFG.DO2_TX, pred=arm_pred) -> dO2 ([128 q x 64 d] at d = 64 * rank)
    #   _wait(bars.q2_empty, q2.phase) -> q2_full.arrive(CFG.Q2_TX, pred=arm_pred) -> Q2 ([128 q x 64 d] at d = 64 * rank)
    #   q1 = advance(q1, geo.Q1_STAGES); do1 = advance(.., geo.DO1_STAGES); do2, q2 = advance(.., 1)
    # BARRIER (after the persistent loop, P15, if const_expr(CFG.CTA_MMA == 2)): q1 = _drain(bars.q1_empty, q1, geo.Q1_STAGES);
    #   _drain(bars.q2_empty, q2, 1); _drain(bars.do1_empty, do1, 1); _drain(bars.do2_empty, do2, 1); _drain(bars.k1_empty, kv, 1);
    #   _drain(bars.v_empty, kv, 1)   [the leader's last multicast commits must land on a resident follower]
    # smem_map rubin_b2 (header sections 5 / 8): every slab base is the geometry's (geo.K1 / V / K2 / Q1 / DO1 / DO2 / Q2 / STATS_RING); the sQ1 ring has
    #   geo.Q1_STAGES = 2 (the same q1_state form, `advance(.., 2)`, drain x 2); the sDO1 ring geo.DO1_STAGES = 2: _wait(do1_empty[do1.idx]) -> the two
    #   half-slot arms do1_full[2 do1.idx + half] (DO1_HALF_TX each) -> the two half boxes into stage do1.idx -> `advance(do1, 2)`; drain x 2.
    """
    # both CTAs hold all 128 q stats (D image spans N=128): STATS/DELTA 512 B each, 1 slot.
    # Q1 single stage (FAv4 2-CTA form) funds sDQ 16 KiB; Q1(t+1) window ~2000 clk
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.SERVICE_REGS
    q1_state = PipelineState.start(phase=1)
    q2_state = PipelineState.start(phase=1)
    do1_state = PipelineState.start(phase=1)
    do2_state = PipelineState.start(phase=1)
    kv_state = PipelineState.start(phase=1)
    k2e_state = PipelineState.start(phase=1)
    sched_state = PipelineState.start()
    tma_ring = cutlass.const_expr(stats_form_code == STATS_FORM_TMA_RING)  # the stats form (header section 11)
    if cutlass.const_expr(tma_ring):
        stats_ld = PipelineState.start(phase=1)  # P5 (b): pre-armed = the ring's two free passes; +1 per tile; carried across items (P2)
        stats_src_base = Pointer(stats2.iterator.raw_ptr(), dtype=cutlass.Float32)  # the padded pre-scaled stats region (fp32, element-addressed)
        rowoff_view = cutlass.make_array_view(rowoff)  # rowoff[b] = sum_{i<b} ceil(S_i / N) * N: the sequence's padded token base (the setup kernel)
    tma_mask = cutlass.Int16(1) << rank  # the one-CTA multicast mask of every cga2 load (the bytes complete on the leader's mbar)
    d_half = rank * cutlass.Int32(CFG.D_HALF)
    q_half = rank * cutlass.Int32(CFG.Q_HALF)
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        if cutlass.const_expr(DEBUG_LOADER_DELAY_US > 0):
            for _us in cutlass.range(DEBUG_LOADER_DELAY_US, unroll=1):
                nvvm.inline_ptx("nanosleep.u32 1000;", read_only_args=[])
        arm_pred = is_leader & nvvm.elect_sync()
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        if cutlass.const_expr(tma_ring):
            # RESOURCE-NOTE: stats_full arm = pred=elect_sync() on BOTH CTAs: a plain bulk copy never routes (P9)
            own_pred = nvvm.elect_sync()  # ONE lane of this warp per item (P16): the local stats_full arm; the copy elects inside bulk_copy
            # RESOURCE-NOTE: src = stats2 + ((head*PAD_TOTAL + rowoff[b])/128 + t)*1024: 1024-B aligned (16 needed)
            # the item's record base: head stride + the sequence's padded tile base (rowoff[b] is a multiple of N); + t records per tile below
            stats_src = stats_src_base + (
                cutlass.Int64(head) * cutlass.Int64(stats2.stride[0])
                + cutlass.Int64(cutlass.Int32(rowoff_view[batch]) // cutlass.Int32(CFG.STATS_PAD_GRAN)) * cutlass.Int64(stats2.stride[1])
            )
        q1_desc = _map_ptr(maps, batch, n_batch, MAP_Q1)
        q2_desc = _map_ptr(maps, batch, n_batch, MAP_Q2)
        k_desc = _map_ptr(maps, batch, n_batch, MAP_K)
        v_desc = _map_ptr(maps, batch, n_batch, MAP_V)
        do1_desc = _map_ptr(maps, batch, n_batch, MAP_DO1)
        do2_desc = _map_ptr(maps, batch, n_batch, MAP_DO2)
        kv_head = head // ratio
        kv_row0 = pair * CFG.PAIR_KV_ROWS + rank * CFG.M
        if lo < hi:
            tma_tensormap_acquire(q1_desc)
            tma_tensormap_acquire(q2_desc)
            tma_tensormap_acquire(k_desc)
            tma_tensormap_acquire(v_desc)
            tma_tensormap_acquire(do1_desc)
            tma_tensormap_acquire(do2_desc)
            _wait(bars.k1_empty, kv_state.phase, dbg, _W["wait:loader:k1_empty"])
            bars.k1_full.arrive(n_bytes=CFG.K1_TX, pred=arm_pred)
            tma_load_tile(
                _operand_k_major(slab, geo.K1, dtype, CFG.M),
                tma_slice_runtime_desc(k_desc, 0, kv_row0, kv_head, 0),
                bars.k1_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            _wait(bars.v_empty, kv_state.phase, dbg, _W["wait:loader:v_empty"])
            bars.v_full.arrive(n_bytes=CFG.V_TX, pred=arm_pred)
            tma_load_tile(
                _operand_k_major(slab, geo.V, dtype, CFG.M),
                tma_slice_runtime_desc(v_desc, 0, kv_row0, kv_head, 0),
                bars.v_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            # Only the K2 reload is gated by the previous item's dV store drain (dV staging aliases K2); the observation is answered
            # by a bare dv_full arrive (fix A): the store's next k2_empty arrive waits dv_full, so this waiter is never lapped.
            _wait(bars.k2_empty, k2e_state.phase, dbg, _W["wait:loader:k2_empty"])
            bars.dv_full.arrive()
            bars.k2_full.arrive(n_bytes=CFG.K2_TX, pred=arm_pred)
            for h in cutlass.range_constexpr(CFG.KV_HALVES):
                tma_load_tile(
                    _operand_mn_major(slab, geo.K2 + h * CFG.K2_HALF_BYTES, dtype, CFG.M),
                    tma_slice_runtime_desc(k_desc, d_half, pair * CFG.PAIR_KV_ROWS + h * CFG.M, kv_head, 0),
                    bars.k2_full.smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mask,
                    acquire=False,
                )
            kv_state = advance(kv_state, 1)
        else:
            # zero-trip item: the same single k2_empty observation and dv_full answer (P14: every path flips every barrier once)
            _wait(bars.k2_empty, k2e_state.phase, dbg, _W["wait:loader:k2_empty"])
            bars.dv_full.arrive()
        k2e_state = advance(k2e_state, 1)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            q_row = qtile * CFG.N
            _wait(bars.q1_empty[q1_state.idx], q1_state.phase, dbg, _W["wait:loader:q1_empty"])
            bars.q1_full[q1_state.idx].arrive(n_bytes=CFG.Q1_TX, pred=arm_pred)
            tma_load_tile(
                _operand_k_major(slab, geo.Q1 + q1_state.idx * geo.Q1_BYTES, dtype, CFG.Q_HALF),
                tma_slice_runtime_desc(q1_desc, 0, q_row + q_half, head, 0),
                bars.q1_full[q1_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            if cutlass.const_expr(tma_ring):
                # RESOURCE-NOTE: stats(t) after Q1(t), before dO1(t): need order; lands ~1 period before P(t)
                # BARRIER: stats_empty[s] st(1) -> the stage's previous tenant (tile t-2) was read out by the softmax (the pre-arm passes the kernel's
                # first two tiles); then the ONE expect_tx arm of the record (1 == ONE_LANE, 1024 == STATS_TX) and the ONE bulk copy on OUR mbar
                _wait(bars.stats_empty[stats_ld.idx], stats_ld.phase, dbg, _W["wait:loader:stats_empty"])
                bars.stats_full[stats_ld.idx].arrive(n_bytes=CFG.STATS_TX, pred=own_pred)
                bulk_copy(
                    _view(
                        slab,
                        cutlass.Int32(geo.STATS_RING) + stats_ld.idx * cutlass.Int32(CFG.STATS_STAGE_BYTES),
                        cutlass.Uint8,
                        CFG.STATS_STAGE_BYTES,
                        alignment=1024,
                    ),
                    stats_src + qtile * cutlass.Int32(STATS_RECORD_ELEMS),
                    CFG.STATS_TX,
                    bars.stats_full[stats_ld.idx].smem_ptr,
                )
            _wait(_slot(bars.do1_empty, do1_state, geo.DO1_STAGES), do1_state.phase, dbg, _W["wait:loader:do1_empty"])
            # the sDO1 ring stage (rubin_b2: geo.DO1_STAGES = 2; sm100: the single slab, today's spelling)
            do1_base = (geo.DO1 + do1_state.idx * geo.DO1_BYTES) if cutlass.const_expr(geo.DO1_STAGES > 1) else geo.DO1
            for half in cutlass.range_constexpr(CFG.D_HALVES):
                _do1_slot(bars, do1_state, half, geo.DO1_STAGES).arrive(n_bytes=CFG.DO1_HALF_TX, pred=arm_pred)
            for half in cutlass.range_constexpr(CFG.D_HALVES):
                tma_load_subtiles(
                    _operand_k_major(slab, do1_base, dtype, CFG.Q_HALF),
                    tma_slice_runtime_desc(do1_desc, 0, q_row + q_half, head, 0),
                    _do1_slot(bars, do1_state, half, geo.DO1_STAGES).smem_ptr,
                    half,
                    1,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mask,
                    acquire=False,
                )
            _wait(bars.do2_empty, do2_state.phase, dbg, _W["wait:loader:do2_empty"])
            bars.do2_full.arrive(n_bytes=CFG.DO2_TX, pred=arm_pred)
            tma_load_tile(
                _operand_mn_major(slab, geo.DO2, dtype, CFG.N),
                tma_slice_runtime_desc(do2_desc, d_half, q_row, head, 0),
                bars.do2_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            _wait(bars.q2_empty, q2_state.phase, dbg, _W["wait:loader:q2_empty"])
            bars.q2_full.arrive(n_bytes=CFG.Q2_TX, pred=arm_pred)
            tma_load_tile(
                _operand_mn_major(slab, geo.Q2, dtype, CFG.N),
                tma_slice_runtime_desc(q2_desc, d_half, q_row, head, 0),
                bars.q2_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mask,
                acquire=False,
            )
            q1_state = advance(q1_state, geo.Q1_STAGES)
            do1_state = advance(do1_state, geo.DO1_STAGES)
            do2_state = advance(do2_state, 1)
            q2_state = advance(q2_state, 1)
            if cutlass.const_expr(tma_ring):
                stats_ld = advance(stats_ld, geo.STATS_STAGES)  # +1 per tile = the softmax's count (P14); nothing on a zero-trip item
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:loader:sched_full"])
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        # P15: whole-ring drains from the CARRIED states (the leader's last multicast commits must land on a resident follower).
        _drain(bars.q1_empty, q1_state, geo.Q1_STAGES, dbg, _W["wait:loader:q1_empty"])
        _drain(bars.q2_empty, q2_state, 1, dbg, _W["wait:loader:q2_empty"])
        _drain(bars.do1_empty, do1_state, geo.DO1_STAGES, dbg, _W["wait:loader:do1_empty"])
        _drain(bars.do2_empty, do2_state, 1, dbg, _W["wait:loader:do2_empty"])
        _drain(bars.k1_empty, kv_state, 1, dbg, _W["wait:loader:k1_empty"])
        _drain(bars.v_empty, kv_state, 1, dbg, _W["wait:loader:v_empty"])


@cute.jit
def _store_loop(
    slab,
    bars,
    sched,
    maps,
    pair,
    head,
    batch,
    meta,
    n_batch,
    rank,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    diag_2cta_code: cutlass.Constexpr[int],
    regs: cutlass.Constexpr[int],
    ratio: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    geo: cutlass.Constexpr,
    dbg,
):
    """Warp 14, BOTH CTAs: dQ reduce-add out of the DEDICATED sDQ ring (4 fp32 slices [64 q x 32 d] per tile in TWO rounds of two slots; one
    reduce per PAIR tile), dV / dK native stores.
    ``diag_2cta_code`` (DIAGNOSTIC, default diag_2cta_code(None) = 0 = this body): DIAG_DQ_REDUCE_ADD_OFF skips the four tma_reduce_add ISSUES per tile
    (dq_reduce_add_off); the dq_full waits, the commit / wait groups and the per-round dq_empty arrives stay, so the 2-slot ring balance holds (2N per
    slot per item) and dQ stays ZERO (output WRONG BY DESIGN; `--validate` refuses the arm).
    The six mbarrier waits of this warp (dq_full x 4 USES, dv_full, dk_full) go through :func:`_wait_store` (the declared parity-wait form).

    # BARRIER (per item, first): _credit(bars.sched_empty[sched_state.idx]); NO dq_empty bootstrap (D3: the epilogue's empty_state is pre-armed)
    # BARRIER (per tile t; q_row = qtile * N + q_half) -- the the closed per-round release:
    #   for round r in (A = 0, B = 1): for slot in (0, 1): (slot, slice) = CFG.dq_use(r * DQ_SLOTS + slot) = (0, 0), (1, 2), (0, 1), (1, 3):
    #     _wait_store(bars.dq_full[slot], dq.phase) -> tma_reduce_add_subtile(sDQ[slot] -> dQ map @ (slice * DQ_BOX, q_row)) -> tma_store_commit()
    #     -> dq = advance(dq, DQ_SLOTS)   [2 stages, +1 per USE: idx == slot by construction, asserted in the config]
    #     then tma_store_wait(1) -> bars.dq_empty[0].arrive(); tma_store_wait(0) -> bars.dq_empty[1].arrive()   [bare, 32 each, after the
    #     corresponding SOURCE READ: `cp.async.bulk.wait_group.read`]
    # BARRIER (item end): _wait_store(bars.dv_full, out.phase) -> dV store (2 halves from CFG.dv_store(half); skipped for a padded CTA);
    #   _wait_store(bars.dk_full, out.phase) -> dK store (CFG.dk_store(rank, half)); out = advance(out, 1); tma_store_commit(); tma_store_wait(0)
    #   -> _arrive_leader(bars.dvk_empty)   [LEADER, bare, 32 x 2 = 64]   -> bars.k2_empty.arrive()   [THREAD local, bare, 32; its back-edge:
    #      this arrive follows the dv_full wait, whose 160 include the loader's 32 = the loader observed the previous k2_empty phase (fix A)]
    # BARRIER (item end, dkv_fp32 = dkv_accum fp32_reduce, `_dkv_fp32_reduce_rounds`): _wait_store(bars.dv_full) -> [live_rows: acquire MAP_DV + MAP_DK;
    #   4 x tma_reduce_add_subtile of the round-1 slots CFG.dkv_acc_r1_slot(0..3) -> dV map @ (i * DQ_BOX, kv_row0, head // ratio) + 1 of slot 4 -> dK map @ (0, ..)]
    #   -> tma_store_commit() -> tma_store_wait(0) -> bars.dvk_stage_empty.arrive()   [NEW row, bare 32 == init: the round-2 WAR release] -> _wait_store(bars.dk_full)
    #   -> [live_rows: 3 x tma_reduce_add_subtile of CFG.dkv_acc_r2_slot(0..2) -> dK map @ ((i + 1) * DQ_BOX, ..)] -> out += 1; tma_store_commit(); tma_store_wait(0)
    #   -> dvk_empty / k2_empty as above.
    # No drain: dvk_empty's last phase is the leader MMA's drain; k2_empty / dq_empty are local.
    # ``dbg`` (STAMPS): the dq_full / dv_full / dk_full / sched_full wait sums + the clk inside the per-round and item-end `tma_store_wait`.
    """
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.SERVICE_REGS
    dq_state = PipelineState.start()
    out_state = PipelineState.start()
    sched_state = PipelineState.start()
    q_half = rank * cutlass.Int32(CFG.Q_HALF)
    valid = cutlass.Int32(1)
    while valid != 0:
        _credit(bars.sched_empty[sched_state.idx])
        lo, hi, seq_q, seq_k, q_start = _item_bounds(pair, meta, batch, n_batch, causal)
        dq_desc = _map_ptr(maps, batch, n_batch, MAP_DQ)
        if lo < hi:
            tma_tensormap_acquire(dq_desc)
        for qtile in cutlass.range(lo, hi, 1, unroll=1):
            q_row = qtile * CFG.N + q_half
            for r in cutlass.range_constexpr(CFG.DQ_ROUNDS):
                for s in cutlass.range_constexpr(CFG.DQ_SLOTS):
                    slot, slice_index = CFG.dq_use(r * CFG.DQ_SLOTS + s)
                    _wait_store(bars.dq_full[slot], dq_state.phase, dbg, _W["wait:store:dq_full"])
                    if cutlass.const_expr(
                        diag_2cta_code != DIAG_DQ_REDUCE_ADD_OFF
                    ):  # DIAGNOSTIC dq_reduce_add_off: the ISSUE only (commit / wait / arrives stay)
                        tma_reduce_add_subtile(
                            _dq_slot_tile(slab, geo, slot), tma_slice_runtime_desc(dq_desc, slice_index * CFG.DQ_BOX, q_row, head, 0), 0, acquire=False
                        )
                    tma_store_commit()
                    dq_state = advance(dq_state, CFG.DQ_SLOTS)
                for s in cutlass.range_constexpr(CFG.DQ_SLOTS):
                    if cutlass.const_expr(dbg is not None):
                        t_wg = cute.arch.clock64()
                    tma_store_wait(CFG.DQ_SLOTS - 1 - s)  # the slot's source read drained: the arrive below is the ring's release
                    if cutlass.const_expr(dbg is not None):
                        _clk_add(dbg, _W["store_wait_group_rounds"], cutlass.Int32(cute.arch.clock64() - t_wg))  # STAMPS: the reduce-add source drain
                    bars.dq_empty[s].arrive()  # all 32 after this slot's source read: the round's release of slot s
        # Invalid owners and the padded SECOND half of a live pair still consume both native publications and return every source
        # credit, but never acquire / use a map for rows past the sequence.
        kv_row0 = pair * CFG.PAIR_KV_ROWS + rank * CFG.M
        live_rows = kv_row0 < seq_k
        _wait_store(bars.dv_full, out_state.phase, dbg, _W["wait:store:dv_full"])
        if cutlass.const_expr(dkv_fp32):
            # dkv_accum = fp32_reduce: 5 + 3 fp32 reduce-adds into the per-KV-head workspace maps, round 1's drain -> dvk_stage_empty before the
            # dk_full wait (`_dkv_fp32_reduce_rounds`; kv head = head // ratio); the item-end commit / wait(0) below drain round 2
            _dkv_fp32_reduce_rounds(slab, bars, geo, maps, batch, n_batch, kv_row0, head // ratio, live_rows, out_state.phase, dbg)
        else:
            if live_rows:
                dv_desc = _map_ptr(maps, batch, n_batch, MAP_DV)
                tma_store_tile(_operand_k_major(slab, geo.K2, dtype, CFG.M), tma_slice_runtime_desc(dv_desc, 0, kv_row0, head, 0))
            _wait_store(bars.dk_full, out_state.phase, dbg, _W["wait:store:dk_full"])
            if live_rows:
                dk_desc = _map_ptr(maps, batch, n_batch, MAP_DK)
                for half in cutlass.range_constexpr(CFG.D_HALVES):
                    tma_store_tile(_store_half(slab, geo.dk_store(rank, half), dtype), tma_slice_runtime_desc(dk_desc, half * CFG.HEAD_BOX, kv_row0, head, 0))
        out_state = advance(out_state, 1)
        tma_store_commit()
        if cutlass.const_expr(dbg is not None):
            t_wg = cute.arch.clock64()
        tma_store_wait(0)
        if cutlass.const_expr(dbg is not None):
            _clk_add(dbg, _W["store_wait_group_item"], cutlass.Int32(cute.arch.clock64() - t_wg))  # STAMPS: the dV / dK store drain
        _arrive_leader(bars.dvk_empty)
        bars.k2_empty.arrive()
        pair, head, batch, valid, sched_state = _next_tile(sched, bars, sched_state, dbg, _W["wait:store:sched_full"])


@cute.jit
def _scheduler_loop(sched, is_leader, regs: cutlass.Constexpr[int]):
    """Warp 15: CLC try_cancel over the 256-row-unit envelope; the LEADER's scheduler publishes to the pair, the follower's has no loop.

    # LEADER only (the follower's warp 15 falls through to the final barrier_cta_sync), st(0) adv SCHED_STAGES -- the library's
    # scheduler_warp_loop IS the punch list: _wait(bars.sched_empty[st.idx], st.phase) [30 credits: 15 local + 15 remote .release.cta];
    # if elect: bars.sched_full[st.idx].arrive(n_bytes=CFG.CLC_TX) [our copy, 1 == init] + nvvm.mbarrier_arrive_expect_tx(mapa(
    # bars.sched_full[st.idx].smem_ptr, 1), CFG.CLC_TX, scope=CTA) [the follower's copy, 1 == init] + clusterlaunchcontrol_try_cancel(
    # multicast=1); fence_proxy; warp sync; _wait(bars.sched_full[st.idx], st.phase); read_clc_payload -> valid; advance.
    # Every sched_empty phase is observed in-loop (the credits of tile n gate payload n + 1; the terminal payload carries no credit):
    # no drain.  The payload decodes the PAIR's unit (x // CGA_M); rank = the low bit of block_idx_in_cluster selects each CTA's kv half.
    """
    # THD unit = 256 kv rows per pair; causal trim per pair; CTA 1's diagonal tile is masked.
    nvvm.setmaxregister(regs, nvvm.SetMaxRegisterAction.DECREASE)  # CFG.SERVICE_REGS
    scheduler_warp_loop(sched, CFG.SCHED_STAGES, is_leader, CFG.CGA_M)


@cute.kernel
def bprop_d128_accum_grouped_2cta_stail_kernel(
    maps: cute.Tensor,
    stats_tokens: cute.Tensor,
    stats_rows: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    n_batch: cutlass.Int32,
    stats_origin: cutlass.Constexpr[int],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    diag_2cta_code: cutlass.Constexpr[int],
    warp_roles_code: cutlass.Constexpr[int],
    mask_form_code: cutlass.Constexpr[int],
    stats_form_code: cutlass.Constexpr[int],
    tmem_map_code: cutlass.Constexpr[int],
    smem_map_code: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    debug_clk: cutlass.Constexpr[bool],
    dbg_clk: Optional[cute.Tensor],
):
    """Slab, scaffold views, the barrier init block (P4), the pair TMEM allocation, the role dispatch, the pair-symmetric teardown.
    ``smem_map_code`` = the SMEM MAP (header sections 5 / 8 / 11; PART-KEYED by the prepared module: SMEM_MAP_SM100 = the opt-in slab chain of every part,
    SMEM_MAP_RUBIN_B2 = the 327 KiB carveout of cc 10.7 ON dq_576 x tma_ring x 16w: the two-stage dS / exchange ring, the sQ1 / sDO1 rings, the
    dq_read_done row); selects the geometry record (the slab size the launcher declares, every slab base and ring depth), the bar bundle
    (BarsRingDQ576B2) and the STAMPS record; declines every other combination by name at the trace.
    ``tmem_map_code`` = the TMEM MAP (header sections 9 / 11; PART-KEYED by the prepared module: TMEM_MAP_S_TAIL = the 512-column parts' rendering,
    TMEM_MAP_DQ576 = the independent dQ image of the 576-column carve, selectable by name on cc 10.7): selects the geometry record (``_geometry``: TMEM_COLS /
    TMEM_IS_EXCLUSIVE at the alloc AND the dealloc, DQ_COL), the bar bundle (BarsDQ576 / BarsRingDQ576: s_read / s_tail_empty absent) and the STAMPS
    record; declines 16w_fa4 and the S-tail WAIT probes by name at the trace.
    ``stats_tokens`` / ``stats_rows`` = the two stats operand slots, read by the stats form (header sections 11 / 14): under ``ldg`` the
    user's LSE tensor and the chain's compact delta region (the softmax's `_stats` fetch); under ``tma_ring`` the padded row offsets ``rowoff``
    [batch + 1] Int32 and the padded pre-scaled stats region ``stats2`` [heads][PAD_TOTAL / N][2][N] fp32 (the loader's bulk copies).  Two plain
    ``cute.Tensor`` slots under both forms: an ``Optional`` annotation or an added operand changes the mangled entry name of the ldg rendering
    (its md5 pin) even when the value is None, so the form selects the slots' meaning, never the signature.
    ``warp_roles_code`` = the ROLE SHAPE (header sections 4 / 5 / 11; WARP_ROLES_16W = the default, today's rendering; WARP_ROLES_16W_FA4 =
    the alternate role shape: the softmax ships the exchange from its registers, warps 8-11 run ``_reduce_loop``, the bar bundle is ``BarsXFA4``), from the typed
    ``warp_roles`` field.
    ``mask_form_code`` = the softmax's mask form (header section 11: MASK_FORM_FSEL = the default; MASK_FORM_PREDICATED = the interior-tile
    mask skip; MASK_FORM_DIAG_OFF = the DIAGNOSTIC ceiling, output wrong by design), from the typed ``mask_form`` field;
    read by the softmax role only.  ``diag_2cta_code`` = the DIAGNOSTIC probe (header section 11; diag_2cta_code(None) = 0 = this body), from the typed
    ``diag_2cta`` field; ``debug_clk`` / ``dbg_clk`` = the STAMPS lever (header section 11) and its Int32 dump tensor (None = dropped from the
    signature when off: byte-identical rendering).
    ``dkv_fp32`` = the op-level ``dkv_accum`` field (header section 11: False = bf16 partials + the fold, the default rendering; True = fp32
    reduce-adds of dK / dV into the per-KV-head workspace, a ROUNDING CHANGE).
    ``stats_form_code`` = the STATS FORM (header section 11: STATS_FORM_LDG = the default, the softmax's per-tile LDG fetch of LSE / delta under
    the SOFTMAX_BARRIER pair, ``lse`` / ``delta`` present; STATS_FORM_TMA_RING = the loader-fed two-stage SMEM stats ring behind stats_full /
    stats_empty fed from ``stats2`` (the chain's padded pre-scaled stats region) at ``rowoff[batch]`` (the sequence's padded token base), ``lse`` /
    ``delta`` None = dropped, the CFG_RING geometry: SMEM_BYTES == the cap, the bar block / TMEM handle / CLC payload re-based +1024), from the
    typed ``stats_form`` field; read by the softmax and the loader; declines by name under 16w_fa4."""
    softmax_regs, epilogue_regs, service_regs = CFG.SOFTMAX_REGS, CFG.EPILOGUE_REGS, CFG.SERVICE_REGS  # config = truth: the 144 / 144 / 80 split
    fa4_roles = warp_roles_code == WARP_ROLES_16W_FA4  # the role shape (trace-time)
    tma_ring = stats_form_code == STATS_FORM_TMA_RING  # the stats form (trace-time)
    # Trace-time invariants (const_expr: a Python raise, never a staged early exit); the config's validate_fields_for_roles declines first by name.
    if cutlass.const_expr(warp_roles_code not in (WARP_ROLES_16W, WARP_ROLES_16W_FA4)):
        raise ValueError(f"warp_roles_code must be one of {(WARP_ROLES_16W, WARP_ROLES_16W_FA4)}; got {warp_roles_code!r}")
    if cutlass.const_expr(stats_form_code not in (STATS_FORM_LDG, STATS_FORM_TMA_RING)):
        raise ValueError(f"stats_form_code must be one of {(STATS_FORM_LDG, STATS_FORM_TMA_RING)}; got {stats_form_code!r}")
    if cutlass.const_expr(tma_ring and fa4_roles):  # the first build: the stats ring rows are the 16w bundle's (validate_fields_for_roles)
        raise ValueError("stats_form tma_ring is not built under warp_roles 16w_fa4 (the stats ring rows and waits are sited in the 16w bundle)")
    dq576 = tmem_map_code == TMEM_MAP_DQ576  # the TMEM map (header section 9): the independent dQ image of the 576-column carve
    if cutlass.const_expr(tmem_map_code not in (TMEM_MAP_S_TAIL, TMEM_MAP_DQ576)):
        raise ValueError(f"tmem_map_code must be one of {(TMEM_MAP_S_TAIL, TMEM_MAP_DQ576)}; got {tmem_map_code!r}")
    # the reduce group has no program-order carrier for the independent image's reuse edge (validate_fields_for_roles)
    if cutlass.const_expr(dq576 and fa4_roles):
        raise ValueError(
            "tmem_map dq_576 is not built under warp_roles 16w_fa4 (the independent dQ image's reuse edge rides the 16w epilogue's program order into ds_smem)"
        )
    if cutlass.const_expr(dq576 and diag_2cta_code in (DIAG_S_TAIL_WAIT_OFF, DIAG_S_READ_WAIT_OFF, DIAG_S_TAIL_AND_S_READ_WAIT_OFF)):
        raise ValueError("the S-tail wait probes are not defined under tmem_map dq_576 (the s_read / s_tail_empty rows they skip are deleted)")
    b2 = smem_map_code == SMEM_MAP_RUBIN_B2  # the SMEM map (header section 8): the two-stage dS / exchange staging ring of the 327 KiB carveout
    ovs = smem_map_code == SMEM_MAP_SM100_OVERSIZED  # the DIAGNOSTIC map (section 11): the sm100 map declared in the oversized carveout (byte-identical body)
    if cutlass.const_expr(smem_map_code not in (SMEM_MAP_SM100, SMEM_MAP_RUBIN_B2, SMEM_MAP_SM100_OVERSIZED)):
        raise ValueError(f"smem_map_code must be one of {(SMEM_MAP_SM100, SMEM_MAP_RUBIN_B2, SMEM_MAP_SM100_OVERSIZED)}; got {smem_map_code!r}")
    if cutlass.const_expr(ovs and fa4_roles):
        raise ValueError("smem_map sm100_oversized is not built under warp_roles 16w_fa4 (the diagnostic map renders the 16w body only)")
    if cutlass.const_expr(ovs and not tma_ring):
        raise ValueError("smem_map sm100_oversized is not built under stats_form ldg (the diagnostic map's records are the tma_ring renderings)")
    if cutlass.const_expr(b2 and not dq576):  # the lagging order needs the dq_read_done row of the independent image (validate_fields_for_roles)
        raise ValueError("smem_map rubin_b2 is not built under tmem_map s_tail (the two-stage ring's lagging epilogue order voids the S-tail map's carrier)")
    if cutlass.const_expr(b2 and fa4_roles):
        raise ValueError("smem_map rubin_b2 is not built under warp_roles 16w_fa4 (the ring and its dq_read_done row are sited in the 16w epilogue)")
    if cutlass.const_expr(b2 and not tma_ring):
        raise ValueError("smem_map rubin_b2 is not built under stats_form ldg (the B2 table and its record are sited under tma_ring)")
    # the two stats slots by form (the host hands the form's tensors; the shapes differ, so a slot mix-up is a trace-time shape error, never silent)
    if cutlass.const_expr(tma_ring):
        lse, delta = None, None
        rowoff, stats2 = stats_tokens, stats_rows
        if cutlass.const_expr(len(rowoff.shape) != 1 or len(stats2.shape) != 4):
            raise ValueError("stats_form tma_ring: stats_tokens = rowoff [batch + 1] and stats_rows = stats2 [heads][tiles][2][N] (the prepared host)")
    else:
        lse, delta = stats_tokens, stats_rows
        rowoff, stats2 = None, None
        if cutlass.const_expr(len(lse.shape) not in (2, 3) or len(delta.shape) != 3):
            raise ValueError("stats_form ldg: stats_tokens = the LSE tensor ([T, H] or [1, H, pitch]) and stats_rows = the compact delta [1, H, rounded]")
    # the geometry by (form, map) (header sections 8 / 9): the ring's second stage re-bases the scaffold; the map sets TMEM_COLS / DQ_COL
    geo = _geometry(tma_ring, dq576, b2, ovs)
    dbg_rec = dbg_clk_for_stats_form(
        "tma_ring" if tma_ring else "ldg", None, geo.TMEM_MAP, geo.SMEM_MAP
    )  # the STAMPS record by (form, maps): the spare, one word map
    if cutlass.const_expr(fa4_roles and debug_clk):  # the merge: the STAMPS record layout is the 16w roles' (no words for the fa4 waits)
        raise ValueError("debug_clk is not defined under warp_roles 16w_fa4 (the STAMPS record layout names the 16w roles' waits)")
    if cutlass.const_expr(fa4_roles and dkv_fp32):  # the merge: the fp32 dK / dV staging rounds live in the 16w epilogue loop
        raise ValueError("dkv_accum fp32_reduce is not built under warp_roles 16w_fa4 (the reduce group stages bf16 halves only)")
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    x, y, z = cute.arch.block_idx()
    rank = cute.arch.block_idx_in_cluster() & cutlass.Int32(1)  # pair rank under cluster (2,1,1): 0 = LEADER, 1 = FOLLOWER
    is_leader = rank == cutlass.Int32(LEADER_RANK)  # the loader's arm predicate and the scheduler / MMA role select
    pair, head, batch = x // CFG.CGA_M, y, z  # grid x = pairs x CGA_M: both CTAs of a cluster decode the same pair
    # 231424 (ldg) / 232448 == the opt-in cap (tma_ring) / 314368 (rubin_b2: the DSL launcher sizes the CTA's SMEM from this declaration and sets the
    # oversized mode above the opt-in cap; header section 14)
    slab = cutlass.Array(cutlass.Uint8, geo.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)
    # 35 fields (16w) / 33 (16w_fa4) / 37 (16w x tma_ring) / 33 + 35 (dq_576) / 36 (rubin_b2): the init loop below runs over the bundle it was handed
    bars = _bars(slab, fa4_roles, tma_ring, dq576, b2)
    dvk_row = _dvk_stage_empty_row(bars)  # the fp32_reduce row's index in THIS bundle (34 under the s_tail tables, 32 under dq_576's; -1 under 16w_fa4)
    tmem_slot = _view(slab, geo.TMEM_HANDLE, cutlass.Int32, 1, alignment=4)
    sched_words = _view(slab, geo.SCHED_PAYLOAD, cutlass.Int32, CFG.SCHED_PAYLOAD_BYTES // 4, alignment=16)
    sched = Sched(bars.sched_full.base_ptr, bars.sched_empty.base_ptr, sched_words, x, y, z)
    # --- STAMPS contexts (header section 11): the record view in the bar block's spare + ONE representative warp per role; None when off.
    if cutlass.const_expr(debug_clk):
        dbg_words = _view(slab, dbg_rec.SMEM, cutlass.Int32, dbg_rec.WORDS, alignment=16)
        dbg_softmax = Dbg(dbg_words, warp == cutlass.Int32(0))
        dbg_epilogue = Dbg(dbg_words, warp == cutlass.Int32(EPILOGUE_LEAD_WARP))
        dbg_single = Dbg(dbg_words, True)
        dbg_leader = DbgLeader(dbg_words, True)
    else:
        dbg_words = None
        dbg_softmax = None
        dbg_epilogue = None
        dbg_single = None
        dbg_leader = None
    # --- P4: ONE warp, ONE lane initialises every stage of every ring; fence -> CTA sync -> CLUSTER sync, all OUTSIDE the branch.
    if warp == 0:
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(len(bars)):
                # dvk_stage_empty has no user under the default rendering (16w, dkv_fp32 False; the init guard keeps the audited rendering
                # byte-identical -- an unguarded init is a follow-up); under 16w_fa4 the bundle has no such row
                if cutlass.const_expr(fa4_roles or dkv_fp32 or i != dvk_row):
                    for stage in cutlass.range_constexpr(bars[i].stages):
                        bars[i][stage].init()
            if cutlass.const_expr(debug_clk):
                _clk_init(dbg_words, dbg_rec.WORDS)  # STAMPS: the record zeroed + the body start stamped before the CTA sync publishes it
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    cga_arrive()  # required before ANY cross-CTA arrive (multicast commits, leader arrives, the remote CLC arm, the DSMEM ship)
    cga_wait()
    # --- TMEM: one tcgen05.alloc.cta_group::2 of geo.TMEM_COLS per CTA (BOTH CTAs allocate; the leader's collective MMAs write both halves): 512 on
    # the s_tail map; the whole 576-column carve, exclusive, under dq_576 (header section 9) -- the SAME geo.TMEM_COLS at _tmem_dealloc_pair.
    if warp == MMA_WARP:
        _tmem_alloc_pair(tmem_slot, geo.TMEM_COLS, geo.TMEM_IS_EXCLUSIVE)
    nvvm.barrier_cta_sync()  # publishes the TMEM base to every warp
    tmem_base = tmem_slot.load()
    # BARRIER: role dispatch: warp < 8 softmax | < 12 epilogue (16w) / REDUCE group (16w_fa4: _reduce_loop) | == MMA_WARP (leader: _mma_loop_leader,
    #   follower: _mma_loop_follower) | == TMA_LOAD_WARP | == TMA_STORE_WARP | SCHED_WARP (leader only runs the loop).
    if warp < SOFTMAX_WARP_END:
        _softmax_loop(
            slab,
            bars,
            sched,
            tmem_base,
            lse,
            delta,
            origins,
            stats_origin,
            pair,
            head,
            batch,
            meta,
            n_batch,
            rank,
            warp,
            scale,
            causal,
            dtype,
            diag_2cta_code,
            fa4_roles,
            softmax_regs,
            mask_form_code,
            stats_form_code,
            tmem_map_code,
            geo,
            dbg_softmax,
        )
    elif warp < EPILOGUE_WARP_END:
        if cutlass.const_expr(fa4_roles):
            _reduce_loop(slab, bars, sched, tmem_base, pair, head, batch, meta, n_batch, rank, causal, dtype, diag_2cta_code, epilogue_regs, geo)
        else:
            _epilogue_loop(
                slab,
                bars,
                sched,
                tmem_base,
                pair,
                head,
                batch,
                meta,
                n_batch,
                rank,
                warp,
                causal,
                dtype,
                diag_2cta_code,
                tmem_map_code,
                smem_map_code,
                geo,
                epilogue_regs,
                dkv_fp32,
                dbg_epilogue,
            )
    elif warp == MMA_WARP:
        if is_leader:
            _mma_loop_leader(
                slab,
                bars,
                sched,
                tmem_base,
                pair,
                head,
                batch,
                meta,
                n_batch,
                causal,
                dtype,
                diag_2cta_code,
                fa4_roles,
                tmem_map_code,
                smem_map_code,
                geo,
                service_regs,
                dbg_leader,
            )
        else:
            _mma_loop_follower(bars, sched, pair, head, batch, meta, n_batch, causal, geo, service_regs, dbg_single)
    elif warp == TMA_LOAD_WARP:
        _load_loop(
            slab,
            bars,
            sched,
            maps,
            pair,
            head,
            batch,
            meta,
            n_batch,
            rank,
            is_leader,
            causal,
            dtype,
            ratio,
            service_regs,
            stats_form_code,
            geo,
            stats2,
            rowoff,
            dbg_single,
        )
    elif warp == TMA_STORE_WARP:
        _store_loop(
            slab,
            bars,
            sched,
            maps,
            pair,
            head,
            batch,
            meta,
            n_batch,
            rank,
            causal,
            dtype,
            diag_2cta_code,
            service_regs,
            ratio,
            dkv_fp32,
            geo,
            dbg_single,
        )
    else:
        _scheduler_loop(sched, is_leader, service_regs)
    # BARRIER (teardown, both CTAs): the final CTA sync after every role returned, then the MMA warp's pair-symmetric dealloc.
    #   No remote arrive targets an exited CTA: the pair-symmetric dealloc wait is the last cross-CTA hand-off.
    nvvm.barrier_cta_sync()
    if cutlass.const_expr(debug_clk):
        if warp == 0:
            if nvvm.elect_sync():
                _clk_dump(dbg_words, dbg_clk, rank, dbg_rec.WORDS)  # STAMPS: every role returned and every record word is published by the CTA sync above
    if warp == MMA_WARP:
        _tmem_dealloc_pair(bars, tmem_slot, rank, geo.TMEM_COLS)


bprop_d128_accum_grouped_2cta_stail_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def main_thd_2cta_stail_host(
    maps,
    stats_tokens,
    stats_rows,
    meta,
    origins,
    n_batch,
    nheads,
    max_skv,
    stats_origin: cutlass.Constexpr[int],
    scale,
    causal: cutlass.Constexpr[bool],
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    diag_2cta_code: cutlass.Constexpr[int],
    warp_roles_code: cutlass.Constexpr[int],
    mask_form_code: cutlass.Constexpr[int],
    stats_form_code: cutlass.Constexpr[int],
    tmem_map_code: cutlass.Constexpr[int],
    smem_map_code: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    debug_clk: cutlass.Constexpr[bool],
    dbg_clk,
    stream,
):
    """One owner PAIR per Q head and declared 256-row KV-envelope unit (header section 14); no host length read.  ``stats_tokens`` /
    ``stats_rows`` = the stats form's two operands (ldg: the LSE tensor and the compact delta; tma_ring: the padded row offsets and the padded
    pre-scaled stats region -- the same two positional slots, header section 14).  ``dbg_clk`` (the Int32
    STAMPS dump, config dbg_clk_records x DBG_CLK.WORDS words) is passed to the kernel ONLY under ``debug_clk``; otherwise it is None =
    dropped from the specialised signature (the default rendering stays byte-identical).  ``warp_roles_code`` = the role shape (header
    section 11; 0 = 16w, the default).  ``tmem_map_code`` = the TMEM map (header sections 9 / 11; PART-KEYED: 0 = s_tail, every part's default;
    1 = dq_576, selectable by name on cc 10.7).  ``smem_map_code`` = the SMEM map (header sections 8 / 11; PART-KEYED: 0 = sm100, every part's
    default; 1 = rubin_b2, selectable by name on cc 10.7 ON dq_576: the 314368-B slab the launcher declares)."""
    n_pairs = cute.math.max((max_skv + CFG.THD_UNIT_ROWS - 1) // CFG.THD_UNIT_ROWS, cutlass.Int32(1))
    dbg_clk_on = dbg_clk if cutlass.const_expr(debug_clk) else None
    bprop_d128_accum_grouped_2cta_stail_kernel(
        maps,
        stats_tokens,
        stats_rows,
        meta,
        origins,
        n_batch,
        stats_origin,
        scale,
        causal,
        dtype,
        ratio,
        diag_2cta_code,
        warp_roles_code,
        mask_form_code,
        stats_form_code,
        tmem_map_code,
        smem_map_code,
        dkv_fp32,
        debug_clk,
        dbg_clk_on,
    ).launch(
        grid=(n_pairs * CFG.CGA_M, nheads, n_batch),
        block=(CFG.THREADS, 1, 1),
        cluster=(CFG.CGA_M, CFG.CGA_N, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


# ======================================================================================================================
# section 9: the 2-CTA chain kernels (setup / delta_zero_stats) and the pair chain host
# (was kernels/sm100/bprop_d128_accum_grouped_2cta_helpers.py)
# ======================================================================================================================
"""Grouped THD d128 chain on the cta_group::2 datapath: eleven per-sequence maps, the pair main kernel, the shared folds.

Five launches, as the grouped THD chain: setup (THIS module's eleven-map variant: the nine operand / output maps + the two stats rows)
-> delta_zero -> MAIN (the 2-CTA pair body) -> cast_dq -> the fp32 dK / dV fold.  Everything but the setup kernel, the delta launch's
``stats_form = "tma_ring"`` variant and the main launch is imported from the shipping modules unchanged.  FOUR launches under the op-level
field ``dkv_accum = "fp32_reduce"`` (``dkv_fp32``; the s_tail body only): the per-KV-head fp32 dK / dV accumulators take the MAP_DK / MAP_DV
slots as 128-row fp32 [M x DQ_BOX]-box maps, delta_zero zero-fills them, the main kernel reduce-adds, cast_dq converts them once at each
output's own origin, and the fold is not launched.
Under ``stats_form = "tma_ring"`` (the s_tail body's header section 11; FAv4's LSElog2 + bulk-copy form) the delta launch is THIS module's
``delta_zero_stats_thd_d128_2cta_kernel``: the same dot tree and zero fills, plus the ldg softmax's two stat multiplies performed once per q row
and stored PRE-SCALED into the chain's q-tile-PADDED ``stats2`` region (fp32 [heads][PAD_TOTAL / N][2][N]: -log2(e) * LSE with the -inf -> 0
guard, then -scale * delta; every pad row [S_b, ceil(S_b / N) * N) written as 0.0 = the ldg form's `stat = 0` for a row past the sequence), and the
setup kernel writes the padded row offsets ``rowoff[b]`` the main kernel's loader addresses its ONE 1 KiB bulk copy per q tile with; under ``ldg``
the shipping delta kernel runs, no row offsets are written and the rendering is byte-identical.  The nine tensor maps are the same under both forms.
The DENSE twin (``d128_grouped_2cta_contract.LAYOUT_FORMS`` "dense"; ``layout_code`` = LAYOUT_FORM_DENSE): the SAME five launches over the dense
compact-BSHD bytes read as the packed view (1, B*S, H, D) -- the setup kernel's dense arm writes the metadata [kv_lens | cu_q | cu_k] from the two
declared extents (``dense_sq`` / ``dense_skv``, runtime Int32, None = dropped under THD) and the Stats origin row ``org[8][b] = b * H * S_q`` (the
contiguous (B, H, S_q) Stats read as the head-major packed form with pitch S_q); no length or offset tensor exists (``q_lens`` / ``kv_lens`` None).
Every other launch is unchanged: it reads the metadata and the origins the setup kernel wrote.
"""

# The pair body's map order is the setup kernel's (the slots below are written in the body's MAP_* order); its box rows are the base geometry's.
assert (MAP_Q1, MAP_K, MAP_V, MAP_DO1, MAP_DQ, MAP_DK, MAP_DV, MAP_Q2, MAP_DO2) == tuple(range(CFG_PAIR_BASE.MAPS_PER_SEQUENCE))
assert CFG.MAP_BOX_ROWS == CFG_PAIR_BASE.MAP_BOX_ROWS and CFG.MAPS_PER_SEQUENCE == CFG_PAIR_BASE.MAPS_PER_SEQUENCE == 9
assert (STATS_FORM_LDG, STATS_FORM_TMA_RING) == (0, 1) and CFG.STATS_KINDS == STATS_KINDS == 2 and CFG.STATS_PAD_GRAN == CFG_PAIR_BASE.N
STATS_PAD_GRAN = CFG.STATS_PAD_GRAN  # 128: every sequence's stat rows padded to a whole q tile (write_thd_row_offsets gran; config = truth)
LAYOUT_FORM_THD = LAYOUT_FORM_CODES["thd"]  # the setup kernel's ``layout_code`` arms (config = truth: d128_grouped_2cta_contract.LAYOUT_FORM_CODES)
LAYOUT_FORM_DENSE = LAYOUT_FORM_CODES["dense"]
assert (LAYOUT_FORM_THD, LAYOUT_FORM_DENSE) == (0, 1) and DENSE_STATS_ORIGIN_SPEC[0] == 8  # the stats port's origin row (ORIGIN_PORTS index 8)


def _tensor_map_rows(tensor, rows, box):
    """Logical TMA coordinates (d, sequence, head, batch) with a ``rows``-row sequence box (the 2-CTA path needs 64- and 128-row boxes)."""
    return tmap.create_tensor_map_tiled_from_view(tensor, box_dims=(1, rows, 1, box), stride_order=(3, 1, 2, 0), swizzle=tmap.TensorMapSwizzle.s128b)


@cute.kernel
def setup_thd_d128_2cta_kernel(
    q1_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    do1_desc: cutlass.GridConstant[tmap.TensorMap],
    dq_desc: cutlass.GridConstant[tmap.TensorMap],
    dk_desc: cutlass.GridConstant[tmap.TensorMap],
    dv_desc: cutlass.GridConstant[tmap.TensorMap],
    q2_desc: cutlass.GridConstant[tmap.TensorMap],
    do2_desc: cutlass.GridConstant[tmap.TensorMap],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq_accum: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    maps: cute.Tensor,
    q_lens: Optional[cute.Tensor],
    kv_lens: Optional[cute.Tensor],
    ro_q: Optional[cute.Tensor],
    ro_k: Optional[cute.Tensor],
    ro_v: Optional[cute.Tensor],
    ro_o: Optional[cute.Tensor],
    ro_do: Optional[cute.Tensor],
    ro_dq: Optional[cute.Tensor],
    ro_dk: Optional[cute.Tensor],
    ro_dv: Optional[cute.Tensor],
    ro_stats: Optional[cute.Tensor],
    origin_specs: cutlass.Constexpr,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
    rowoff: Optional[cute.Tensor] = None,
    layout_code: cutlass.Constexpr[int] = 0,
    dense_sq: Optional[cutlass.Int32] = None,
    dense_skv: Optional[cutlass.Int32] = None,
):
    """Publish prefixes / origins, then the NINE sequence-local map arrays of the 2-CTA body (header section 10 of the kernel); under
    ``stats_form = "tma_ring"`` ALSO the padded row offsets ``rowoff[b] = sum_{i<b} ceil(S_i / N) * N`` (+ the total at [B]) of the chain's padded
    pre-scaled stats region (``None`` = dropped from the signature: the ldg rendering is byte-identical).

    ``layout_code`` (``d128_grouped_2cta_contract.LAYOUT_FORM_CODES``; LAYOUT_FORM_THD = the default = this rendering, byte-identical: the dense
    operands are None = dropped from the signature): LAYOUT_FORM_DENSE = the dense compact-BSHD twin -- ``q_lens`` / ``kv_lens`` and every
    ``ro_*`` are None, and the metadata is written ARITHMETICALLY from the two declared extents ``dense_sq`` / ``dense_skv`` (runtime Int32:
    kv_lens[b] = S_kv, cu_q[b] = b * S_q, cu_k[b] = b * S_kv), plus the Stats origin row ``org[8][b] = b * H * S_q`` (H = the packed Q view's head
    count): the contiguous (B, H, S_q) Stats read as the head-major packed form with pitch S_q (the contract module's docstring).  The row-offset
    pass and the nine map arrays then read the synthesized prefix exactly as the THD form reads the device-built one.

    The form of ``setup_thd_d128_kernel`` (grouped partials: compact KV origins for dK / dV, every external origin row retained
    for the final fold); the per-sequence descriptor order is ``MAP_Q1, MAP_K, MAP_V, MAP_DO1, MAP_DQ, MAP_DK, MAP_DV, MAP_Q2,
    MAP_DO2`` with the box rows of ``CFG.MAP_BOX_ROWS``.  Every map writer releases to the tensor-map proxy; the consuming
    main kernel acquires before use.
    """
    if cutlass.const_expr(layout_code not in (LAYOUT_FORM_THD, LAYOUT_FORM_DENSE)):
        raise ValueError(f"layout_code must be one of {(LAYOUT_FORM_THD, LAYOUT_FORM_DENSE)}; got {layout_code!r}")
    dense = layout_code == LAYOUT_FORM_DENSE
    if cutlass.const_expr(dense != (dense_sq is not None and dense_skv is not None and q_lens is None and kv_lens is None)):
        raise ValueError("the dense layout carries dense_sq / dense_skv and no length tensors; the THD layout carries q_lens / kv_lens and neither extent")
    if cutlass.const_expr(dense and (origin_specs[8] != DENSE_STATS_ORIGIN_SPEC or any(spec is not None for spec in origin_specs[:8]))):
        raise ValueError(f"the dense layout's origin_specs are {(None,) * 8 + (DENSE_STATS_ORIGIN_SPEC,)}; got {origin_specs!r}")
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp, lane = tid // 32, tid % 32
    data = cutlass.make_array_view(meta)
    if cutlass.const_expr(dense):
        # dense arm: the uniform metadata and the Stats origins, every thread a stride of batches (B + 1 prefix words per side)
        n_threads = cutlass.Int32(THD_SETUP_THREADS)
        cu_k0 = 2 * n_batch + 1
        heads_q = cutlass.Int32(q.shape[2])
        org = cutlass.make_array_view(origins)  # Int64 [len(ORIGIN_PORTS)][batch]: the stats row is DENSE_STATS_ORIGIN_SPEC[0] * stride[0] + b
        org_stats0 = cutlass.Int32(DENSE_STATS_ORIGIN_SPEC[0]) * cutlass.Int32(origins.stride[0])
        for b in cutlass.range(tid, n_batch + 1, n_threads, unroll=1):
            data[n_batch + b] = b * dense_sq
            data[cu_k0 + b] = b * dense_skv
            if b < n_batch:
                data[b] = dense_skv
                org[org_stats0 + b] = cutlass.Int64(b) * cutlass.Int64(heads_q) * cutlass.Int64(dense_sq)
    else:
        if n_batch <= 1:
            if tid == 0:
                write_thd_meta(data, cutlass.make_array_view(q_lens), cutlass.make_array_view(kv_lens), lens_form, n_batch)
        else:
            if warp == 0:
                write_thd_prefix_warp(data, cutlass.make_array_view(q_lens), n_batch, n_batch, (lens_form & 1) != 0, lane, store_lengths=False)
            if warp == 1:
                write_thd_prefix_warp(data, cutlass.make_array_view(kv_lens), n_batch, 2 * n_batch + 1, (lens_form & 2) != 0, lane, store_lengths=True)
        offsets = (ro_q, ro_k, ro_v, ro_o, ro_do, ro_dq, ro_dk, ro_dv, ro_stats)
        for port in cutlass.range_constexpr(9):
            if cutlass.const_expr(origin_specs[port] is not None):
                spec = origin_specs[port]
                write_thd_port_origins(origins, offsets[port], spec[0], spec[1], spec[2], n_batch, tid, cutlass.Int32(THD_SETUP_THREADS))
    cute.arch.barrier()
    if cutlass.const_expr(rowoff is not None):
        # stats_form tma_ring: the padded row offsets over the cu_q prefix (written above), ONE thread (a serial prefix over the batches; the main
        # kernel's loader reads rowoff[batch] once per item); gran = the q tile: a sequence's last tile stays inside its own padded block
        if tid == 0:
            write_thd_row_offsets(data, n_batch, cutlass.Int32(STATS_PAD_GRAN), dst=cutlass.make_array_view(rowoff), dst0=0)
    if nvvm.elect_sync():
        cu_k = 2 * n_batch + 1
        batch_step = cutlass.Int32(THD_SETUP_THREADS // 32)
        # (base map, slot, cu offset, tensor, origin row): the q ports clip at cu_q, the kv ports at cu_k; the partials keep compact origins.
        emit_seq_descs(
            q1_desc,
            maps,
            data,
            n_batch,
            q,
            n_batch,
            cutlass.Int64(q.stride[1]),
            1,
            slot_base=MAP_Q1 * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 0),
        )
        emit_seq_descs(
            k_desc,
            maps,
            data,
            cu_k,
            k,
            n_batch,
            cutlass.Int64(k.stride[1]),
            1,
            slot_base=MAP_K * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 1),
        )
        emit_seq_descs(
            v_desc,
            maps,
            data,
            cu_k,
            v,
            n_batch,
            cutlass.Int64(v.stride[1]),
            1,
            slot_base=MAP_V * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 2),
        )
        emit_seq_descs(
            do1_desc,
            maps,
            data,
            n_batch,
            do,
            n_batch,
            cutlass.Int64(do.stride[1]),
            1,
            slot_base=MAP_DO1 * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 4),
        )
        emit_seq_descs(
            dq_desc,
            maps,
            data,
            n_batch,
            dq_accum,
            n_batch,
            cutlass.Int64(dq_accum.stride[1]),
            1,
            slot_base=MAP_DQ * n_batch,
            first_batch=warp,
            batch_step=batch_step,
        )
        emit_seq_descs(
            dk_desc,
            maps,
            data,
            cu_k,
            dk,
            n_batch,
            cutlass.Int64(dk.stride[1]),
            1,
            slot_base=MAP_DK * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=-1,
        )
        emit_seq_descs(
            dv_desc,
            maps,
            data,
            cu_k,
            dv,
            n_batch,
            cutlass.Int64(dv.stride[1]),
            1,
            slot_base=MAP_DV * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=-1,
        )
        emit_seq_descs(
            q2_desc,
            maps,
            data,
            n_batch,
            q,
            n_batch,
            cutlass.Int64(q.stride[1]),
            1,
            slot_base=MAP_Q2 * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 0),
        )
        emit_seq_descs(
            do2_desc,
            maps,
            data,
            n_batch,
            do,
            n_batch,
            cutlass.Int64(do.stride[1]),
            1,
            slot_base=MAP_DO2 * n_batch,
            first_batch=warp,
            batch_step=batch_step,
            origins=origins,
            origin_row=_origin_row(origin_specs, 4),
        )
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


setup_thd_d128_2cta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def delta_zero_stats_thd_d128_2cta_kernel(
    o: cute.Tensor,
    do: cute.Tensor,
    stats2: cute.Tensor,
    rowoff: cute.Tensor,
    dq_accum: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    lse: cute.Tensor,
    scale: cutlass.Float32,
    n_batch: cutlass.Int32,
    origin_specs: cutlass.Constexpr,
    dk_acc: Optional[cute.Tensor] = None,
    dv_acc: Optional[cute.Tensor] = None,
):
    """The delta launch of ``stats_form = "tma_ring"`` (the s_tail body's header section 11; FAv4's LSElog2 preprocess form): ``delta_zero_thd_d128_kernel``'s
    eight rows per CTA, its fp32 dot tree and its zero fills (dq_accum; dk_acc / dv_acc under dkv_accum fp32_reduce), PLUS the two stat
    multiplies the ldg softmax's `_stats` performs per q row, done HERE once per row and stored PRE-SCALED into the q-tile-PADDED region ``stats2``
    (fp32 [heads][PAD_TOTAL / N][2][N]; the row's padded token = ``rowoff[batch] + row``): kind 0 = ``-log2(e) * LSE`` (0 when the LSE is -inf: a row
    with no keys; the Stats port's own token origin, either Stats layout), kind 1 = ``-scale * rowsum(dO * O)``.  The SAME fp32 operands and the SAME
    operations as `_stats` (the ldg form stores the rowsum raw and reads it back losslessly; `-x * y` lowers to the same negate + mul.rn.f32), so
    every stat word is BITWISE the ldg form's.  Every PAD row [S_b, ceil(S_b / N) * N) of the sequence is written 0.0 (= `_stats`' `stat = 0` for a
    row past the sequence); rows past the padded block are never written.  The grid covers ceil(max_sq / N) * N rows per (batch, head) so the
    longest sequence's pad rows are written too (``host_thd_2cta``: x = ceil(max_sq / N) * (N / ROWS_PER_CTA))."""
    block, batch, head = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    row = cutlass.Int32(block) * ROWS_PER_CTA + cutlass.Int32(tid // LANES_PER_ROW)
    column = cutlass.Int64(tid % LANES_PER_ROW) * ELEMENTS_PER_LANE
    data = cutlass.make_array_view(meta)
    start = cutlass.Int32(data[n_batch + batch])
    length = cutlass.Int32(data[n_batch + batch + 1]) - start
    live = row < length
    in_pad = row < ((length + cutlass.Int32(STATS_PAD_GRAN - 1)) // cutlass.Int32(STATS_PAD_GRAN)) * cutlass.Int32(
        STATS_PAD_GRAN
    )  # the sequence's padded block
    do_values = cutlass.Vector.from_elements(tuple(do.element_type(0.0) for _ in range(ELEMENTS_PER_LANE)), do.element_type)
    o_values = cutlass.Vector.from_elements(tuple(o.element_type(0.0) for _ in range(ELEMENTS_PER_LANE)), o.element_type)
    if live:
        token = cutlass.Int64(start) + cutlass.Int64(row)
        dq_offset = token * cutlass.Int64(dq_accum.stride[1]) + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2]) + column
        zero4 = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(4)), cutlass.Float32)
        (dq_accum.iterator.raw_ptr() + dq_offset).store(zero4, alignment=16)
        (dq_accum.iterator.raw_ptr() + dq_offset + 4).store(zero4, alignment=16)
        o_token = thd_port_origin(origins, _origin_row(origin_specs, 3), batch, start) + cutlass.Int64(row)
        do_token = thd_port_origin(origins, _origin_row(origin_specs, 4), batch, start) + cutlass.Int64(row)
        o_offset = o_token * cutlass.Int64(o.stride[1]) + cutlass.Int64(head) * cutlass.Int64(o.stride[2]) + column
        do_offset = do_token * cutlass.Int64(do.stride[1]) + cutlass.Int64(head) * cutlass.Int64(do.stride[2]) + column
        do_values = (do.iterator.raw_ptr() + do_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
        o_values = (o.iterator.raw_ptr() + o_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
    pairs = []
    for j in cutlass.range_constexpr(ELEMENTS_PER_LANE // 2):
        high = cute.math.mul(do_values[2 * j + 1].to(cutlass.Float32), o_values[2 * j + 1].to(cutlass.Float32), rounding=RN, ftz=True)
        pairs.append(cute.math.fma(do_values[2 * j].to(cutlass.Float32), o_values[2 * j].to(cutlass.Float32), high, rounding=RN, ftz=True))
    dot = cute.math.add(pairs[0], pairs[1], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[2], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[3], rounding=RN, ftz=True)
    # XOR distances 8,4,2,1 never cross a 16-lane row. Inactive row lanes carry zeros but remain active for the full-warp shuffle mask.
    for step in cutlass.range_constexpr(4):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=dot, offset=1 << (3 - step), mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        dot = cute.math.add(dot, peer, rounding=RN, ftz=True)
    dot = cute.math.mul(dot, cutlass.Float32(1.0), rounding=RN, ftz=True)
    if in_pad and tid % LANES_PER_ROW == 0:
        # RESOURCE-NOTE: prescale in the producer: -log2e*LSE (-inf->0), -scale*delta; 64 FMUL2/lane/tile saved
        stat_lse = cutlass.Float32(0.0)  # the pad row's words: today's `stat = 0` guard for a row past the sequence, bit for bit
        stat_delta = cutlass.Float32(0.0)
        if live:
            stat_delta = -scale * dot  # `_stats` group 1: -scale * delta, from the rowsum the ldg form stores raw and reads back
            lse_token = thd_port_origin(origins, _origin_row(origin_specs, 8), batch, start) + cutlass.Int64(row)  # the Stats port's own origin
            if cutlass.const_expr(len(lse.shape) == 2):
                val = cutlass.Float32(lse[lse_token, head])
            else:
                val = cutlass.Float32(lse[0, head, lse_token])
            stat_lse = -cutlass.Float32(LOG2E) * val  # `_stats` group 0: -log2(e) * LSE, 0 for an LSE of -inf
            if val == cutlass.Float32(float("-inf")):
                stat_lse = cutlass.Float32(0.0)
        padded = cutlass.Int32(cutlass.make_array_view(rowoff)[batch]) + row  # the row's padded token (the sequence's block is N-aligned)
        record = cutlass.Int64(head) * cutlass.Int64(stats2.stride[0]) + cutlass.Int64(padded // cutlass.Int32(STATS_PAD_GRAN)) * cutlass.Int64(
            stats2.stride[1]
        )
        word = record + cutlass.Int64(padded % cutlass.Int32(STATS_PAD_GRAN))
        (stats2.iterator.raw_ptr() + word).store(stat_lse)
        (stats2.iterator.raw_ptr() + word + cutlass.Int64(stats2.stride[2])).store(stat_delta)
    if cutlass.const_expr(dk_acc is not None):
        # dkv_accum = fp32_reduce: an exact-zero fp32 image for the main kernel's reduce-adds; kv rows past the sequence are never written.
        kv_length = cutlass.Int32(data[batch])
        kv_start = cutlass.Int32(data[2 * n_batch + 1 + batch])
        if head < dk_acc.shape[2] and row < kv_length:
            kv_token = cutlass.Int64(kv_start) + cutlass.Int64(row)
            _zero_group_vector(dk_acc, kv_token, head, column)
            _zero_group_vector(dv_acc, kv_token, head, column)


delta_zero_stats_thd_d128_2cta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host_thd_2cta(
    q,
    k,
    v,
    o,
    do,
    lse,
    dq,
    dk,
    dv,
    dq_accum,
    delta,
    dk_part,
    dv_part,
    meta,
    origins,
    maps,
    q_lens,
    kv_lens,
    ro,
    origin_specs: cutlass.Constexpr,
    lens_form,
    batch: cutlass.Constexpr[int],
    max_sq,
    max_skv,
    scale,
    causal: cutlass.Constexpr[bool],
    diag_2cta_code: cutlass.Constexpr[int],
    warp_roles_code: cutlass.Constexpr[int],
    mask_form_code: cutlass.Constexpr[int],
    stats_form_code: cutlass.Constexpr[int],
    tmem_map_code: cutlass.Constexpr[int],
    smem_map_code: cutlass.Constexpr[int],
    dkv_fp32: cutlass.Constexpr[bool],
    dk_acc,
    dv_acc,
    debug_clk: cutlass.Constexpr[bool],
    dbg_clk,
    stats2,
    rowoff,
    stream,
    layout_code: cutlass.Constexpr[int] = 0,
    dense_sq=None,
    dense_skv=None,
):
    """Five native launches: the nine-map setup, delta / dQ zero, the cta_group::2 pair MAIN, the dQ cast, the fp32 group fold (four under
    ``dkv_fp32``: the fold is gone and the cast launch converts dK / dV too).
    ``layout_code`` (``d128_grouped_2cta_contract.LAYOUT_FORM_CODES``: 0 = thd, the default; 1 = dense) with ``dense_sq`` / ``dense_skv`` (the two
    declared extents as Int32 under dense, None under THD) and ``q_lens`` / ``kv_lens`` / every ``ro`` entry None under dense: the setup kernel's
    dense arm (its docstring); the four other launches are the same objects under both layouts.
    ``mask_form_code`` = the pair body's softmax mask form (its header section 11: 0 = fsel, the default; 1 = predicated = the interior-tile
    mask skip; 2 = diag_mask_off, OUTPUT WRONG BY DESIGN), ``diag_2cta_code`` = its DIAGNOSTIC probe (section 11; 0 = none), ``warp_roles_code``
    = its ROLE SHAPE (section 11: 0 = 16w, the default rendering; 1 = 16w_fa4, the softmax ships the dS exchange from its registers and warps
    8-11 are the reduce group), ``debug_clk`` / ``dbg_clk`` = its STAMPS lever (section 11) and the Int32 dump tensor the prepared host carved
    (None when off) -- all resolved by the prepared host from the adapter's typed fields and read by the MAIN launch only.
    ``stats_form_code`` = its STATS FORM (section 11: 0 = ldg, selectable; 1 = tma_ring, the default: the setup kernel writes the padded row offsets
    ``rowoff``, the delta launch is ``delta_zero_stats_thd_d128_2cta_kernel`` writing the padded pre-scaled stats region ``stats2`` (``delta`` is None:
    the compact delta region does not exist under the form), and the MAIN launch's loader bulk-copies one (head, q tile) record per tile).
    ``tmem_map_code`` = its TMEM MAP (section 9 / 11; PART-KEYED by the prepared host from the device cc: 0 = s_tail, the 512-column parts' rendering;
    1 = dq_576, the independent dQ image of the 576-column carve, selectable by name on cc 10.7), read by the MAIN launch only (no other launch, no workspace
    region and no map depends on it).  ``smem_map_code`` = its SMEM MAP (section 8 / 11; PART-KEYED: 0 = sm100, every part's default; 1 = rubin_b2,
    selectable by name on cc 10.7 ON dq_576: the 314368-B slab of the two-stage dS / exchange staging ring), read by the MAIN launch only too."""
    tma_ring = stats_form_code == STATS_FORM_TMA_RING
    if cutlass.const_expr(stats_form_code not in (STATS_FORM_LDG, STATS_FORM_TMA_RING)):
        raise ValueError(f"stats_form_code must be one of {(STATS_FORM_LDG, STATS_FORM_TMA_RING)}; got {stats_form_code!r}")
    if cutlass.const_expr(tma_ring != (stats2 is not None) or tma_ring != (rowoff is not None) or tma_ring == (delta is not None)):
        raise ValueError("stats_form tma_ring carries stats2 + rowoff and no compact delta; the ldg form carries delta and neither ring region")
    rows = CFG_PAIR_BASE.MAP_BOX_ROWS
    if cutlass.const_expr(dkv_fp32):
        dk_map, dv_map = _tensor_map_rows(dk_acc, rows[MAP_DK], CFG_PAIR_BASE.DQ_BOX), _tensor_map_rows(dv_acc, rows[MAP_DV], CFG_PAIR_BASE.DQ_BOX)
        dk_ws, dv_ws = dk_acc, dv_acc
    else:
        dk_map, dv_map = _tensor_map_rows(dk_part, rows[MAP_DK], CFG_PAIR_BASE.HEAD_BOX), _tensor_map_rows(dv_part, rows[MAP_DV], CFG_PAIR_BASE.HEAD_BOX)
        dk_ws, dv_ws = dk_part, dv_part
    setup_thd_d128_2cta_kernel(
        _tensor_map_rows(q, rows[MAP_Q1], CFG_PAIR_BASE.HEAD_BOX),
        _tensor_map_rows(k, rows[MAP_K], CFG_PAIR_BASE.HEAD_BOX),
        _tensor_map_rows(v, rows[MAP_V], CFG_PAIR_BASE.HEAD_BOX),
        _tensor_map_rows(do, rows[MAP_DO1], CFG_PAIR_BASE.HEAD_BOX),
        _tensor_map_rows(dq_accum, rows[MAP_DQ], CFG_PAIR_BASE.DQ_BOX),
        dk_map,
        dv_map,
        _tensor_map_rows(q, rows[MAP_Q2], CFG_PAIR_BASE.HEAD_BOX),
        _tensor_map_rows(do, rows[MAP_DO2], CFG_PAIR_BASE.HEAD_BOX),
        q,
        k,
        v,
        do,
        dq_accum,
        dk_ws,
        dv_ws,
        meta,
        origins,
        maps,
        q_lens,
        kv_lens,
        *ro,
        origin_specs,
        lens_form,
        cutlass.Int32(batch),
        rowoff,
        layout_code,
        dense_sq,
        dense_skv,
    ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
    q_grid = cute.math.max((max_sq + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
    if cutlass.const_expr(tma_ring):
        # RESOURCE-NOTE: grid x = ceil(max_sq/128)*16 under tma_ring: pad rows [S_b, ceil128) written as 0.0
        q_grid = cute.math.max((max_sq + STATS_PAD_GRAN - 1) // STATS_PAD_GRAN, cutlass.Int32(1)) * (STATS_PAD_GRAN // ROWS_PER_CTA)
    if cutlass.const_expr(dkv_fp32):
        kv_grid = cute.math.max((max_skv + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
        rows_grid = cute.math.max(q_grid, kv_grid)
        if cutlass.const_expr(tma_ring):
            delta_zero_stats_thd_d128_2cta_kernel(
                o, do, stats2, rowoff, dq_accum, meta, origins, lse, scale, cutlass.Int32(batch), origin_specs, dk_acc, dv_acc
            ).launch(grid=(rows_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream)
        else:
            delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs, dk_acc, dv_acc).launch(
                grid=(rows_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
            )
    else:
        if cutlass.const_expr(tma_ring):
            delta_zero_stats_thd_d128_2cta_kernel(o, do, stats2, rowoff, dq_accum, meta, origins, lse, scale, cutlass.Int32(batch), origin_specs).launch(
                grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
            )
        else:
            delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs).launch(
                grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
            )
    main_thd_2cta_stail_host(
        maps,
        rowoff if cutlass.const_expr(tma_ring) else lse,  # the stats slots by form (the body's header section 14): the padded row offsets | the LSE
        stats2 if cutlass.const_expr(tma_ring) else delta,  # the padded pre-scaled stats region | the compact delta
        meta,
        origins,
        cutlass.Int32(batch),
        q.shape[2],
        max_skv,
        _origin_row(origin_specs, 8),
        scale,
        causal,
        q.element_type,
        q.shape[2] // k.shape[2],
        diag_2cta_code,
        warp_roles_code,
        mask_form_code,
        stats_form_code,
        tmem_map_code,
        smem_map_code,
        dkv_fp32,
        debug_clk,
        dbg_clk,
        stream,
    )
    if cutlass.const_expr(dkv_fp32):
        cast_dq_thd_d128_kernel(
            dq_accum,
            dq,
            meta,
            origins,
            cutlass.Int32(batch),
            _origin_row(origin_specs, 5),
            q.element_type,
            dk_acc,
            dv_acc,
            dk,
            dv,
            _origin_row(origin_specs, 6),
            _origin_row(origin_specs, 7),
        ).launch(grid=(rows_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream)
    else:
        cast_dq_thd_d128_kernel(dq_accum, dq, meta, origins, cutlass.Int32(batch), _origin_row(origin_specs, 5), q.element_type).launch(
            grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
        )
        dkv_reduce_thd_origins_host(
            dk_part,
            dv_part,
            dk,
            dv,
            meta,
            origins,
            cutlass.Int32(batch),
            max_skv,
            _origin_row(origin_specs, 6),
            _origin_row(origin_specs, 7),
            CFG_PAIR_BASE.D,
            q.shape[2] // k.shape[2],
            q.element_type,
            stream,
        )
