# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The quantized (per-tensor fp8) block backward's two FUSED small-kernel launches: the PROLOGUE before the first GEMM and the
EPILOGUE after the norm backward.  One ``@cute.kernel`` each, several independent jobs behind a BLOCK-RANGE dispatch (a
per-block uniform branch on ``blockIdx.x``, so no lane ever diverges on the job), every job the very ``@cute.jit`` body its
standalone kernel runs -- the same code, two launch shapes, bitwise the same bytes.

Why: at short sequences the fp8 backward is HOST-bound -- each launch costs the host tens of microseconds while the small
device kernels take ~15 us -- and nine of its launches were quantization plumbing with no dependency between them.  The
GEMMs and the SDPA row's kernels keep their own launches; everything else that is independent shares one.

**Prologue** (``frost_fp8_bwd_prologue``; before the out-projection dgrad; grid = ``1 + n_amax + n_rec + n_v8``)::

    blocks [0, 1)                     scalar init          slots[0:n] = 0, descale_dp = 1 / scale_dp, then the n_consts PLAN-TIME CONSTANTS into
                                                           their slots out of the launch's fp32 KERNEL ARGUMENTS (quantize.init_scalars_body; one thread)
    blocks [1, 1 + n_amax)            dY amax PARTIALS     partials[c] = max |dY| over CTA c's row groups (quantize.amax_partials_rows; persistent)
    blocks [.., + n_rec)              Q / K rebuild        norm + RoPE from the slab's PRE-norm bands, e4m3 OUT at the forward's static
                                                           scale_q / scale_k (qk_norm_rope_tma.qk_norm_rope_tma_body, fp8_out; persistent TMA ring)
    blocks [.., + n_v8)               v8                   sat_e4m3(V band * scale_v), V's compaction (quantize.quantize_rows)

No job reads anything another job of the same launch writes: the init zeroes the scalar block (every slot a LATER launch
publishes -- ``amax_dy`` by the dY cast, ``amax_do`` by the dO cast, ``amax_dqkvg`` by this module's epilogue, each reduced
from per-CTA partials -- starts from a known zero, so ``quant_scalars()`` never reads a stale word) and nothing in this launch
touches a slot.  The partials are written unconditionally (one per amax CTA), so there is no zeroing to order before them --
that is what lets the amax pass share the FIRST launch.  The rebuild's and v8's static ``scale_q`` / ``scale_k`` / ``scale_v`` are
KERNEL ARGUMENTS of this launch too -- the very values the init job stores into their slots -- never slot reads: the init job
writes those slots in this same launch, and no block may read what another block of the launch writes.  Every LATER launch
reads the slots (stream order).  Nothing is filled on the device at compile time (the constants' only writer is this launch).

**Epilogue** (``frost_fp8_bwd_epilogue``; after the norm backward; grid = ``n_red + n_cast``)::

    blocks [0, n_red)                 dW_norm reduce       column b of class b // d: the fixed-order sum of the partial planes
                                                           (qk_norm_rope_bwd.dw_reduce_column at cols = 1, lanes = 128 -- the same per-column
                                                           chain as the standalone (8, 128) kernel: bitwise); n_red = 2 d, or 0 without dW
    blocks [n_red, n_red + n_cast)    dqkvg cast           every block reduces the dqkvg amax from TWO partials arrays -- the gate backward's
                                                           per-CTA max |dG| (the GATE band) and the norm backward's per-CTA max over the Q / K / V
                                                           bands (quantize.cta_max_of_partials_pair: one barrier) --, derives the scale from it
                                                           (or reads the caller's under "delayed") and casts its row group(s) -- EPILOGUE_CAST_ROWS_PER_GROUP
                                                           rows per lane group, one block per row group (or a persistent grid under
                                                           EPILOGUE_CAST_PERSISTENT; both measured, see the constants); the first cast block's
                                                           thread 0 publishes amax_dqkvg / scale / descale / alpha_b7 / alpha_b8 (quantize.quantize_rows)

The dqkvg amax arrives as PARTIALS because its producers' first form -- one ``atomicMax`` per warp into one slot -- serialised
at the L2 (MEASURED on Rubin, cc 10.7, 204 SMs, SM clock locked at 2376 MHz: the gate backward 0.385 -> 1.589 ms at S = 32K
with its two folds, the standalone amax pass 4.3x slower than the partials pass over the same bytes); every amax of the
quantized backward is now a per-CTA plain store reduced by its consumer.

**Shared geometry.** One launch has one block size (128 threads: every job's own), one SMEM footprint (the prologue's is the
TMA ring's -- ``stages x tile_rows x 512 B`` input stages and the ``mb_full`` barriers; the e4m3 epilogue needs NO output
ring -- plus the 16-B warp-reduce array; the epilogue's is the reduce's ``lanes x 4 B``) and one register count (the
heaviest job's).  A light job therefore runs at the heavy job's residency: pin a fused cast's HBM GB/s against its standalone
kernel's before trusting the fusion -- a loss is fixed by giving that job more rows per block, never reported.

Barrier table: the TMA ring's ``mb_full`` (``qk_norm_rope_tma.py``: ``SUM(issuing lanes) == init`` 1 == 1, LOCAL, no cross-CTA
arrive, so no drain) in the recompute arm only; ``barrier_cta_sync`` in the prologue's amax arm (the warp-maxima combine), the
epilogue's cast arm (the partials reduce's combine, once per block before its row loop) and the reduce arm (the ``sPart``
combine) -- every thread of the block reaches them (the dispatch is block-uniform and no divergent path precedes them).
SMEM buffer table: the recompute arm's ``sIn`` (``qk_norm_rope_tma.py``'s table, unswizzled: 16 lanes span 256 contiguous
bytes), ``sRed`` fp32 ``[warps]`` in both launches (one word per warp, no lane stride), ``sPart`` fp32 ``[lanes]`` (one word per
thread: conflict-free).  Nothing here can hang that the standalone kernels could not.
"""

import math
import numbers
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.device import current_device, multiprocessor_count
from cudnn.frost.tile_dsl.handles import GmemTileTma

from .qk_norm_rope import check_norm_weights_match_recipe
from .qk_norm_rope_bwd import REDUCE_LANES, REDUCE_UNROLL, _check_fp32_plane, dw_reduce_column
from .qk_norm_rope_tma import (
    DEFAULT_REFILL_POS,
    DEFAULT_STAGES,
    TMA_GRANU_ELEMS,
    _fake as _fake_tma,
    _fake_e4m3_thd,
    _fake_thd,
    check_e4m3_out,
    qk_norm_rope_tma_body,
    tile_counts,
)
from .qk_norm_rope_tma import validate_shape as _validate_tma_shape
from .quantize import (
    AMAX_CTAS_PER_SM,
    DEFAULT_CONST_HEAD_COUNT,
    DEFAULT_ROWS_PER_GROUP,
    DEFAULT_THREADS_PER_CTA,
    MAX_ALPHA,
    MAX_INIT_CONSTS,
    SCALE_SOURCES,
    _fake_partials,
    _fake_slot,
    _grad_scale_bits,
    _slot_value,
    _slot_view,
    amax_partials_rows,
    check_partials,
    check_scalar_slot,
    cta_max_of_partials_pair,
    fake_rowmajor_dynamic_token_stride,
    init_scalars_body,
    lanes_per_row,
    publish_scale,
    quantize_rows,
    require_fp8_cvt,
    validate_shape,
)

_FAKE_STREAM = None
THREADS = DEFAULT_THREADS_PER_CTA  # every job's own block size (128)
# The epilogue's dqkvg cast: one row group per block like the standalone quantize (EPILOGUE_CAST_PERSISTENT = False) or a persistent
# grid of min(row groups, SMs x AMAX_CTAS_PER_SM) blocks that pays the partials reduce once per resident block; and the rows per lane
# group of its blocks.  MEASURED on Rubin (cc 10.7, 204 SMs, SM clock locked at 2376 MHz; the dqkvg slab at S = 32K, 1632 + 4896
# partials, CUDA events 100 x 3): rows_per_group 4 + one block per group 0.185 ms; 2 + one block per group 0.214; 2 persistent 0.203;
# 4 persistent 0.201; 8 persistent 0.195; 8 + one block per group 0.193 -- against 0.167 for the plain quantize (rows 2) and 0.179
# (rows 4).  The per-block reduce reads 26 KiB of partials from L2, so half the blocks (rows 4) halves that traffic at a 7 % cost on
# the cast itself; the persistent loop loses 19 % of the cast's memory-level parallelism.  A part re-measures both constants.
EPILOGUE_CAST_PERSISTENT = False
EPILOGUE_CAST_ROWS_PER_GROUP = 4

__all__ = [
    "EPILOGUE_CAST_PERSISTENT",
    "EPILOGUE_CAST_ROWS_PER_GROUP",
    "Fp8BwdEpilogueRecipe",
    "Fp8BwdPrologueRecipe",
    "THREADS",
    "compile_fp8_bwd_epilogue",
    "compile_fp8_bwd_prologue",
    "epilogue_grid",
    "prologue_grid",
    "run_fp8_bwd_epilogue",
    "run_fp8_bwd_prologue",
]


def _fake_stream():
    global _FAKE_STREAM
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    return _FAKE_STREAM


# ---------------------------------------------------------------------------
# PROLOGUE: init | dY amax partials | Q / K rebuild with the e4m3 epilogue | v8
# ---------------------------------------------------------------------------


@cute.kernel
def frost_fp8_bwd_prologue(
    # -- the scalar init (block 0) --
    mSlots: cute.Tensor,  # [n_slots] fp32 contiguous: the scalar block
    mScaleDp: cute.Tensor,  # [1] fp32: the caller's scale_dP
    mDescaleDpOut: cute.Tensor,  # [1] fp32 OUT: 1 / scale_dP (a slot of the block)
    const0: cutlass.Float32,  # the plan-time constants, KERNEL ARGUMENTS: const_i -> slots[const_slot0 + i] for i < n_consts
    const1: cutlass.Float32,
    const2: cutlass.Float32,
    const3: cutlass.Float32,
    const4: cutlass.Float32,
    const5: cutlass.Float32,
    const6: cutlass.Float32,
    const7: cutlass.Float32,
    const8: cutlass.Float32,
    const9: cutlass.Float32,
    const10: cutlass.Float32,
    const11: cutlass.Float32,
    const12: cutlass.Float32,
    const13: cutlass.Float32,
    const14: cutlass.Float32,
    const15: cutlass.Float32,
    # -- the dY amax partials (blocks [1, 1 + n_amax)) --
    mDy: cute.Tensor,  # [T, d_model / D, D] bf16: dY viewed in the quantize layout
    mPartials: cute.Tensor,  # [>= n_amax] fp32 OUT
    # -- the Q / K rebuild (blocks [1 + n_amax, + n_rec)) --
    mQ: cute.Tensor,  # [T, H_q, D] the slab's PRE-norm Q band (its element_type; the data moves through the descriptors)
    mWq: Optional[cute.Tensor],  # [D] norm weights; None (both) = RoPE-only
    mWk: Optional[cute.Tensor],
    mCos: cute.Tensor,  # [T, ROPE_DIM]
    mSin: cute.Tensor,
    mQ8: cute.Tensor,  # [T, H_q, D] e4m3 OUT (compact)
    mK8: cute.Tensor,  # [T, H_kv, D] e4m3 OUT
    scale_q: cutlass.Float32,  # the static scale_q -- a KERNEL ARGUMENT (the value the init job stores into its slot), never a slot read
    scale_k: cutlass.Float32,  # the static scale_k
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    # -- the v8 cast (blocks [1 + n_amax + n_rec, + n_v8)) --
    mV: cute.Tensor,  # [T, H_kv, D] bf16: the slab's V band
    mV8: cute.Tensor,  # [T, H_kv, D] e4m3 OUT (compact: V's compaction)
    scale_v: cutlass.Float32,  # the static scale_v (a kernel argument, as above)
    # -- runtime geometry --
    n_dy_rows: cutlass.Int32,
    n_dy_groups: cutlass.Int32,
    n_amax: cutlass.Int32,
    n_tokens: cutlass.Int32,
    n_q_tiles: cutlass.Int32,
    n_tiles: cutlass.Int32,
    n_rec: cutlass.Int32,
    n_v_rows: cutlass.Int32,
    eps: cutlass.Float32,
    # -- compile-time facts --
    n_slots: cutlass.Constexpr[int],
    const_slot0: cutlass.Constexpr[int],
    n_consts: cutlass.Constexpr[int],
    h_dy_ct: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    tile_rows: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    refill_pos: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
) -> None:
    """Block-range dispatch over the four jobs (module docstring); every arm is the standalone kernel's body."""
    tile_elems = cutlass.const_expr(tile_rows * d)
    io_dtype = mQ.element_type
    # SMEM, allocated ONCE here for every arm (an allocation inside an inlined jit body ships one copy per arm): the TMA
    # input ring + its barriers (the rebuild), the warp-maxima word per warp (the amax).  No output ring: the e4m3 epilogue
    # stores straight to GMEM.
    sIn_raw = cutlass.Array(io_dtype, stages * tile_elems, alignment=128, space=cutlass.AddressSpace.smem)
    mb_full = cutlass.Array(cutlass.Int64, stages, alignment=16, space=cutlass.AddressSpace.smem)
    sRed = cutlass.Array(cutlass.Float32, threads_per_cta // 32, alignment=16, space=cutlass.AddressSpace.smem)

    cta = cutlass.Int32(cute.arch.block_idx()[0])
    base_amax = cutlass.Int32(1)
    base_rec = base_amax + n_amax
    base_v8 = base_rec + n_rec
    if cta < base_amax:
        init_scalars_body(
            mSlots,
            mScaleDp,
            mDescaleDpOut,
            const0,
            const1,
            const2,
            const3,
            const4,
            const5,
            const6,
            const7,
            const8,
            const9,
            const10,
            const11,
            const12,
            const13,
            const14,
            const15,
            n_slots,
            const_slot0,
            n_consts,
        )
    else:
        if cta < base_rec:
            amax_partials_rows(
                mDy,
                mPartials,
                n_dy_rows,
                n_dy_groups,
                cutlass.Int32(h_dy_ct),
                cta - base_amax,
                n_amax,
                sRed,
                h_dy_ct,
                const_head_count,
                d,
                threads_per_cta,
                rows_per_group,
            )
        else:
            if cta < base_v8:
                tma_q = GmemTileTma(tma_q_desc)
                tma_k = GmemTileTma(tma_k_desc)
                qk_norm_rope_tma_body(
                    mQ,
                    mWq,
                    mWk,
                    mCos,
                    mSin,
                    None,  # rstd_q: the record already carries the forward's rstd
                    None,  # rstd_k
                    tma_q,
                    tma_k,
                    tma_q,  # the bf16 output descriptors: placeholders, never touched under the e4m3 epilogue
                    tma_k,
                    mQ8,
                    mK8,
                    scale_q,
                    scale_k,
                    n_tokens,
                    n_q_tiles,
                    n_tiles,
                    cta - base_rec,
                    n_rec,
                    eps,
                    d,
                    rope_dim,
                    tile_rows,
                    stages,
                    1,  # stages_o: no output ring is traced (fp8_out); the state machine still needs a depth >= 1
                    threads_per_cta,
                    h_q_ct,
                    h_kv_ct,
                    refill_pos,
                    False,  # fused_store_wait: there is no store drain
                    sIn_raw,
                    None,  # sOut_raw
                    None,  # sRstd
                    mb_full,
                )
            else:
                quantize_rows(mV, mV8, scale_v, n_v_rows, cutlass.Int32(h_kv_ct), cta - base_v8, h_kv_ct, const_head_count, d, threads_per_cta, rows_per_group)


@cute.jit
def fp8_bwd_prologue_launch(
    slots: cute.Tensor,
    scale_dp: cute.Tensor,
    descale_dp_out: cute.Tensor,
    const0: cutlass.Float32,
    const1: cutlass.Float32,
    const2: cutlass.Float32,
    const3: cutlass.Float32,
    const4: cutlass.Float32,
    const5: cutlass.Float32,
    const6: cutlass.Float32,
    const7: cutlass.Float32,
    const8: cutlass.Float32,
    const9: cutlass.Float32,
    const10: cutlass.Float32,
    const11: cutlass.Float32,
    const12: cutlass.Float32,
    const13: cutlass.Float32,
    const14: cutlass.Float32,
    const15: cutlass.Float32,
    dy: cute.Tensor,
    partials: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    w_q: Optional[cute.Tensor],
    w_k: Optional[cute.Tensor],
    cos: cute.Tensor,
    sin: cute.Tensor,
    q8: cute.Tensor,
    k8: cute.Tensor,
    scale_q: cutlass.Float32,
    scale_k: cutlass.Float32,
    v: cute.Tensor,
    v8: cute.Tensor,
    scale_v: cutlass.Float32,
    n_dy_rows: cutlass.Int32,
    n_dy_groups: cutlass.Int32,
    n_amax: cutlass.Int32,
    n_tokens: cutlass.Int32,
    n_q_tiles: cutlass.Int32,
    n_tiles: cutlass.Int32,
    n_rec: cutlass.Int32,
    n_v_rows: cutlass.Int32,
    n_blocks: cutlass.Int32,
    eps: cutlass.Float32,
    n_slots: cutlass.Constexpr[int],
    const_slot0: cutlass.Constexpr[int],
    n_consts: cutlass.Constexpr[int],
    h_dy_ct: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    tile_rows: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    refill_pos: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """The Q / K input descriptors exactly as ``qk_norm_rope_tma_launch`` builds them (box shapes innermost-LAST, unswizzled,
    the slab's token stride in the descriptor), then the one launch."""
    box_q = (1, tile_rows, TMA_GRANU_ELEMS)
    box_k = (tile_rows // h_kv_ct, h_kv_ct, TMA_GRANU_ELEMS)
    order = (2, 1, 0)
    mk = lambda t, box: tmap.create_tensor_map_tiled_from_view(  # noqa: E731
        t,
        box_dims=box,
        stride_order=order,
        swizzle=tmap.TensorMapSwizzle.none,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    frost_fp8_bwd_prologue(
        slots,
        scale_dp,
        descale_dp_out,
        const0,
        const1,
        const2,
        const3,
        const4,
        const5,
        const6,
        const7,
        const8,
        const9,
        const10,
        const11,
        const12,
        const13,
        const14,
        const15,
        dy,
        partials,
        q,
        w_q,
        w_k,
        cos,
        sin,
        q8,
        k8,
        scale_q,
        scale_k,
        mk(q, box_q),
        mk(k, box_k),
        v,
        v8,
        scale_v,
        n_dy_rows,
        n_dy_groups,
        n_amax,
        n_tokens,
        n_q_tiles,
        n_tiles,
        n_rec,
        n_v_rows,
        eps,
        n_slots,
        const_slot0,
        n_consts,
        h_dy_ct,
        h_q_ct,
        h_kv_ct,
        d,
        rope_dim,
        tile_rows,
        stages,
        refill_pos,
        threads_per_cta,
        rows_per_group,
        const_head_count,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream)


prologue_cache = {}


class Fp8BwdPrologueRecipe(NamedTuple):
    """Build-time facts of the prologue launch (every field plan-time derivable; the token count rides in as runtime ``Int32``)."""

    compiled: object
    dtype: object
    h_q: int
    h_kv: int
    h_dy: int  # d_model / d: the dY view's head count
    d: int
    rope_dim: int
    eps: float
    tile_rows: int
    stages: int
    apply_norm: bool
    n_slots: int
    rows_per_cta: int  # the quantize layout's rows per block (the amax and v8 jobs)
    n_ctas_cap: int  # SMs x 8: the persistent caps of the amax job and of the rebuild job
    threads: int
    # Appended (defaults = an init job without constants): the slot the init job's first plan-time constant lands in and how many it
    # stores from the launch's kernel arguments -- the artifact's ABI, so run_fp8_bwd_prologue checks ``consts`` against them (Rule 1).
    const_slot0: int = 0
    n_consts: int = 0


def compile_fp8_bwd_prologue(
    *,
    dtype,
    h_q: int,
    h_kv: int,
    d_model: int,
    d: int,
    rope_dim: int,
    eps: float,
    apply_norm: bool,
    n_slots: int,
    tile_rows: int,
    stages: int = DEFAULT_STAGES,
    refill_pos: int = DEFAULT_REFILL_POS,
    threads_per_cta: int = THREADS,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
    const_slot0: int = 0,
    n_consts: int = 0,
) -> Fp8BwdPrologueRecipe:
    """Build from SHAPES ALONE -- no allocation, no launch.  ``tile_rows`` is the Q / K rebuild's TMA tile (the block's
    ``_QkNormRope.resolve_tile_rows()``: it must divide ``h_q``, be a multiple of ``h_kv`` and spread over the warps --
    ``qk_norm_rope_tma.validate_shape`` declines anything else by name); ``apply_norm=False`` traces the RoPE-only rebuild.
    ``d_model % d == 0`` (the dY view's head count).  Needs the fp8 ``cvt`` (sm_89+; declined by name) and TMA (sm_90+).
    ``const_slot0`` / ``n_consts`` (appended, defaults = an init job without constants) are the init job's constant slot range,
    exactly ``compile_init_scalars``'s: the VALUES are runtime kernel arguments of the launch (``run_fp8_bwd_prologue(consts=)``),
    so one artifact serves every QuantSpec and nothing is written to the device before the launch that reads it.  Every knob is
    in the cache key."""
    if d_model % d != 0:
        raise ValueError(f"d_model={d_model} must be a multiple of d_head={d}: dY is quantized through a [T, d_model / D, D] view")
    if isinstance(n_slots, bool) or not isinstance(n_slots, int) or n_slots < 1:
        raise ValueError(f"n_slots must be a positive int (the fp32 slots of the scalar block), got {n_slots!r}")
    if isinstance(const_slot0, bool) or not isinstance(const_slot0, int) or const_slot0 < 0:
        raise ValueError(f"const_slot0 must be a non-negative int (the slot the first plan-time constant lands in), got {const_slot0!r}")
    if isinstance(n_consts, bool) or not isinstance(n_consts, int) or n_consts < 0:
        raise ValueError(f"n_consts must be a non-negative int (the plan-time constants the init job stores from its arguments), got {n_consts!r}")
    if n_consts > MAX_INIT_CONSTS:
        raise ValueError(f"n_consts={n_consts} exceeds the {MAX_INIT_CONSTS} constant arguments the scalar-init ABI reserves")
    if const_slot0 + n_consts > n_slots:
        raise ValueError(f"the {n_consts} plan-time constants at slots [{const_slot0}, {const_slot0 + n_consts}) do not fit the {n_slots}-slot block")
    validate_shape(d, threads_per_cta)
    _validate_tma_shape(d, rope_dim, h_q, h_kv, tile_rows, threads_per_cta)
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"the fp8 backward prologue serves bf16/f16 slabs only, got {dtype}")
    if threads_per_cta % 32 != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of 32 (the amax fold's warp butterfly)")
    if not apply_norm and rope_dim == 0:
        raise ValueError("apply_norm=False with rope_dim=0 is an identity copy of Q/K; there is nothing to rebuild")
    require_fp8_cvt("compile_fp8_bwd_prologue")
    device = current_device()
    h_dy = d_model // d
    key = (
        str(dtype),
        int(h_q),
        int(h_kv),
        int(h_dy),
        int(d),
        int(rope_dim),
        bool(apply_norm),
        int(n_slots),
        int(const_slot0),
        int(n_consts),
        int(tile_rows),
        int(stages),
        int(refill_pos),
        int(threads_per_cta),
        int(rows_per_group),
        bool(const_head_count),
        device,
    )
    if key not in prologue_cache:
        tok = cute.sym_int()
        slots = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (int(n_slots),), stride_order=(0,), assumed_align=4)
        dy = fake_rowmajor_dynamic_token_stride(dtype, tok, h_dy, d)
        q, k = _fake_thd(dtype, tok, h_q, d), _fake_thd(dtype, tok, h_kv, d)
        weights = [_fake_tma(dtype, (d,), (0,)) for _ in range(2)] if apply_norm else [None, None]
        tables = [_fake_tma(dtype, (tok, rope_dim if rope_dim else 1), (1, 0)) for _ in range(2)]
        v = fake_rowmajor_dynamic_token_stride(dtype, tok, h_kv, d)
        prologue_cache[key] = cute.compile(
            fp8_bwd_prologue_launch,
            slots,
            _fake_slot(),  # scale_dp
            _fake_slot(),  # descale_dp_out
            *[cutlass.Float32(0.0) for _ in range(MAX_INIT_CONSTS)],  # the constants: runtime fp32 arguments (the zeros pin the TYPE only)
            dy,
            _fake_partials(),
            q,
            k,
            *weights,
            *tables,
            _fake_e4m3_thd(tok, h_q, d),  # q8
            _fake_e4m3_thd(tok, h_kv, d),  # k8
            cutlass.Float32(1.0),  # scale_q: a runtime fp32 argument (the value pins the TYPE only)
            cutlass.Float32(1.0),  # scale_k
            v,
            _fake_e4m3_thd(tok, h_kv, d),  # v8
            cutlass.Float32(1.0),  # scale_v
            cutlass.Int32(0),  # n_dy_rows   ) runtime; the values pin the TYPE only
            cutlass.Int32(0),  # n_dy_groups )
            cutlass.Int32(1),  # n_amax      )
            cutlass.Int32(0),  # n_tokens    )
            cutlass.Int32(0),  # n_q_tiles   )
            cutlass.Int32(0),  # n_tiles     )
            cutlass.Int32(1),  # n_rec       )
            cutlass.Int32(0),  # n_v_rows    )
            cutlass.Int32(1),  # n_blocks    )
            cutlass.Float32(eps),
            int(n_slots),
            int(const_slot0),
            int(n_consts),
            int(h_dy),
            int(h_q),
            int(h_kv),
            int(d),
            int(rope_dim),
            int(tile_rows),
            int(stages),
            int(refill_pos),
            int(threads_per_cta),
            int(rows_per_group),
            bool(const_head_count),
            _fake_stream(),
            options="--enable-tvm-ffi",
        )
    return Fp8BwdPrologueRecipe(
        compiled=prologue_cache[key],
        dtype=dtype,
        h_q=int(h_q),
        h_kv=int(h_kv),
        h_dy=int(h_dy),
        d=int(d),
        rope_dim=int(rope_dim),
        eps=float(eps),
        tile_rows=int(tile_rows),
        stages=int(stages),
        apply_norm=bool(apply_norm),
        n_slots=int(n_slots),
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        n_ctas_cap=int(multiprocessor_count(device)) * AMAX_CTAS_PER_SM,
        threads=int(threads_per_cta),
        const_slot0=int(const_slot0),
        n_consts=int(n_consts),
    )


def prologue_grid(r: Fp8BwdPrologueRecipe, t: int) -> tuple:
    """``(n_amax, n_rec, n_v8)`` for ``t`` tokens -- the three job widths of the launch (its grid is ``1 + their sum``):
    the amax job's partial count ``min(ceil(t * h_dy / rows_per_cta), cap)``, the rebuild's persistent CTAs
    ``min(n_tiles, cap)``, the v8 cast's blocks ``ceil(t * h_kv / rows_per_cta)``."""
    t = int(t)
    dy_groups = (t * r.h_dy + r.rows_per_cta - 1) // r.rows_per_cta
    n_amax = max(1, min(dy_groups, r.n_ctas_cap))
    _n_q_tiles, n_tiles = tile_counts(t, r.h_q, r.h_kv, r.tile_rows)
    n_rec = max(1, min(n_tiles, r.n_ctas_cap))
    n_v8 = (t * r.h_kv + r.rows_per_cta - 1) // r.rows_per_cta
    return n_amax, n_rec, n_v8


def _check_band(name: str, ten, t: int, h: int, d: int, dtype) -> None:
    """A ``[T, H, D]`` bf16 / f16 operand of the quantize layout: head stride ``D``, unit element stride, 16-B rows."""
    if not isinstance(ten, torch.Tensor) or ten.dtype != dtype:
        got = f"{ten.dtype}" if isinstance(ten, torch.Tensor) else type(ten).__name__
        raise ValueError(f"{name} must be a {dtype} [T, H, D] tensor, got {got}")
    if ten.dim() != 3 or int(ten.shape[0]) != t or int(ten.shape[1]) != h or int(ten.shape[2]) != d:
        raise ValueError(f"{name} must be [T={t}, H={h}, D={d}], got {tuple(ten.shape)}")
    if ten.stride(2) != 1 or ten.stride(1) != d or (ten.stride(0) * 2) % 16:
        raise ValueError(f"{name} must have head stride D={d}, element stride 1 and a 16-byte-aligned token stride, got strides {ten.stride()}")
    if ten.data_ptr() % 16:
        raise ValueError(f"{name} must sit on a 16-byte-aligned base, got {ten.data_ptr():#x}")


def run_fp8_bwd_prologue(
    r: Fp8BwdPrologueRecipe,
    *,
    slots: torch.Tensor,
    scale_dp: torch.Tensor,
    descale_dp_out: torch.Tensor,
    dy: torch.Tensor,
    partials: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    w_q: Optional[torch.Tensor],
    w_k: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    q8: torch.Tensor,
    k8: torch.Tensor,
    scale_q,
    scale_k,
    v: torch.Tensor,
    v8: torch.Tensor,
    scale_v,
    stream,
    consts: tuple = (),
) -> int:
    """Launch.  ``slots`` the contiguous fp32 ``[n_slots]`` scalar block; ``scale_dp`` the caller's 1-element fp32 scalar, OUTSIDE
    the block (a slot would be zeroed before it is read); ``descale_dp_out`` a 1-element view (into the block or not, never inside
    the constants' slot range); ``dy`` ``[T, d_model / D, D]`` bf16; ``partials`` a contiguous fp32 array of at least
    ``prologue_grid(r, T)[0]`` words (every one of its first ``n_amax`` words is OVERWRITTEN); ``q`` / ``k`` the slab's PRE-norm bands
    ``[T, H, D]`` (token stride ``N``); the norm weights both ``None`` for a RoPE-only recipe; ``cos`` / ``sin`` ``[T, rope_dim]``;
    ``q8`` / ``k8`` / ``v8`` compact e4m3 ``[T, H, D]``; ``scale_q`` / ``scale_k`` / ``scale_v`` finite Python numbers -- the static
    scales reach the kernel as fp32 ARGUMENTS (the values the init job stores into their slots; a tensor is refused: a slot of the
    block this launch writes could not be read by its other blocks, and a device fill is what the argument path replaces);
    ``consts`` (appended) the artifact's ``n_consts`` plan-time constants as finite Python numbers in slot order, stored by the init
    job at ``slots[const_slot0 + i]`` -- a length other than ``n_consts`` is a typed error (the ABI is fixed per artifact).  Returns
    ``n_amax`` (the partials written).  Host checks only, no allocation."""
    t = int(q.shape[0])
    check_scalar_slot("slots", slots, numel=r.n_slots)
    check_scalar_slot("scale_dp", scale_dp)
    check_scalar_slot("descale_dp_out", descale_dp_out)
    lo, hi = slots.data_ptr(), slots.data_ptr() + 4 * r.n_slots
    if lo <= scale_dp.data_ptr() < hi:
        raise ValueError("scale_dp lies inside the slot block this launch zeroes: it would be read as 0 (descale_dp = inf); pass the caller's own scalar")
    n_consts = int(getattr(r, "n_consts", 0))
    if len(consts) != n_consts:
        raise ValueError(f"this artifact's init job stores n_consts={n_consts} plan-time constants; got {len(consts)} consts (the ABI is fixed per artifact)")
    const_values = []
    for i, cv in enumerate(consts):
        if isinstance(cv, (bool, torch.Tensor)) or not isinstance(cv, numbers.Real) or not math.isfinite(float(cv)):
            raise ValueError(
                f"consts[{i}] must be a finite Python number (a plan-time constant handed to the kernel as an argument -- never a device tensor), got {cv!r}"
            )
        const_values.append(float(cv))
    if n_consts:
        clo, chi = lo + 4 * int(r.const_slot0), lo + 4 * (int(r.const_slot0) + n_consts)
        if clo <= descale_dp_out.data_ptr() < chi:
            raise ValueError(
                f"descale_dp_out lies inside slots [{r.const_slot0}, {r.const_slot0 + n_consts}) the plan-time constants overwrite: it would hold a constant, not 1 / scale_dp"
            )
    scale_values = []
    for name, sv in (("scale_q", scale_q), ("scale_k", scale_k), ("scale_v", scale_v)):
        if isinstance(sv, (bool, torch.Tensor)) or not isinstance(sv, numbers.Real) or not math.isfinite(float(sv)):
            raise ValueError(
                f"{name} must be a finite Python number: the static scale reaches the rebuild / v8 job as a kernel ARGUMENT (the value its slot "
                f"receives from this launch's init job), never a slot or a device tensor, got {sv!r}"
            )
        scale_values.append(float(sv))
    _check_band("dy", dy, t, r.h_dy, r.d, r.dtype)
    _check_band("q", q, t, r.h_q, r.d, r.dtype)
    _check_band("k", k, t, r.h_kv, r.d, r.dtype)
    _check_band("v", v, t, r.h_kv, r.d, r.dtype)
    check_norm_weights_match_recipe(r.apply_norm, w_q, w_k)
    if r.apply_norm:
        for name, w in (("w_q_norm", w_q), ("w_k_norm", w_k)):
            if w.dtype != r.dtype or tuple(w.shape) != (r.d,) or not w.is_contiguous():
                raise ValueError(f"{name} must be a contiguous [{r.d}] {r.dtype} vector, got {w.dtype} of shape {tuple(w.shape)}")
    tab_cols = r.rope_dim if r.rope_dim else 1
    for name, tab in (("cos", cos), ("sin", sin)):
        if tab.dtype != r.dtype or tab.dim() != 2 or int(tab.shape[0]) != t or int(tab.shape[1]) != tab_cols or tab.stride(1) != 1:
            raise ValueError(f"{name} must be a [{t}, {tab_cols}] {r.dtype} table with unit column stride, got {tab.dtype} of shape {tuple(tab.shape)}")
    check_e4m3_out("q8", q8, t, r.h_q, r.d)
    check_e4m3_out("k8", k8, t, r.h_kv, r.d)
    check_e4m3_out("v8", v8, t, r.h_kv, r.d)
    n_amax, n_rec, n_v8 = prologue_grid(r, t)
    check_partials("partials", partials, n_amax)
    dev = q.device
    for name, ten in (
        ("slots", slots),
        ("scale_dp", scale_dp),
        ("descale_dp_out", descale_dp_out),
        ("dy", dy),
        ("partials", partials),
        ("k", k),
        ("w_q_norm", w_q),
        ("w_k_norm", w_k),
        ("cos", cos),
        ("sin", sin),
        ("q8", q8),
        ("k8", k8),
        ("v", v),
        ("v8", v8),
    ):
        if ten is not None and (not ten.is_cuda or ten.device != dev):
            raise ValueError(f"{name} must be on {dev} with q (the kernel reads every operand through a device pointer), got {ten.device}")
    n_dy_rows = t * r.h_dy
    n_dy_groups = (n_dy_rows + r.rows_per_cta - 1) // r.rows_per_cta
    n_q_tiles, n_tiles = tile_counts(t, r.h_q, r.h_kv, r.tile_rows)
    const_args = [cutlass.Float32(cv) for cv in const_values] + [cutlass.Float32(0.0) for _ in range(MAX_INIT_CONSTS - n_consts)]
    r.compiled(
        slots,
        _slot_view(scale_dp),
        _slot_view(descale_dp_out),
        *const_args,
        dy,
        partials,
        q,
        k,
        w_q,
        w_k,
        cos,
        sin,
        q8,
        k8,
        cutlass.Float32(scale_values[0]),
        cutlass.Float32(scale_values[1]),
        v,
        v8,
        cutlass.Float32(scale_values[2]),
        cutlass.Int32(n_dy_rows),
        cutlass.Int32(n_dy_groups),
        cutlass.Int32(n_amax),
        cutlass.Int32(t),
        cutlass.Int32(n_q_tiles),
        cutlass.Int32(n_tiles),
        cutlass.Int32(n_rec),
        cutlass.Int32(t * r.h_kv),
        cutlass.Int32(1 + n_amax + n_rec + n_v8),
        cutlass.Float32(r.eps),
        cuda.CUstream(int(stream)),
    )
    return n_amax


# ---------------------------------------------------------------------------
# EPILOGUE: dW_norm reduce | dqkvg cast
# ---------------------------------------------------------------------------


@cute.kernel
def frost_fp8_bwd_epilogue(
    # -- the dW_norm reduce (blocks [0, n_red)) --
    mPq: Optional[cute.Tensor],  # [n_q, D] fp32 partials; None = no reduce (n_red == 0)
    mPk: Optional[cute.Tensor],  # [n_k, D]
    mDWq: Optional[cute.Tensor],  # [D] fp32 OUT
    mDWk: Optional[cute.Tensor],  # [D]
    # -- the dqkvg cast (blocks [n_red, n_red + n_cast)) --
    mSrc: cute.Tensor,  # [T, N / D, D] bf16: the dqkvg slab viewed in the quantize layout
    mDst: cute.Tensor,  # [T, N / D, D] e4m3 OUT
    mScale: Optional[cute.Tensor],  # [1] fp32: the caller's scale ("delayed"); None = derived from the reduced amax
    mPartialsGate: cute.Tensor,  # [>= n_gate] fp32: the gate backward's per-CTA max |dG| partials (the GATE band)
    mPartialsBands: cute.Tensor,  # [>= n_bands] fp32: the norm backward's per-CTA partials over the Q / K / V bands
    mAmaxOut: cute.Tensor,  # [1] fp32 OUT (published): the reduced amax = max |dqkvg| -- the scalar block's amax_dqkvg
    mScaleOut: cute.Tensor,  # [1] fp32 OUT (published)
    mDescale: cute.Tensor,  # [1] fp32 OUT
    mAlphaC0: Optional[cute.Tensor],
    mAlphaC1: Optional[cute.Tensor],
    mAlphaC2: Optional[cute.Tensor],
    mAlphaC3: Optional[cute.Tensor],
    mAlphaO0: Optional[cute.Tensor],
    mAlphaO1: Optional[cute.Tensor],
    mAlphaO2: Optional[cute.Tensor],
    mAlphaO3: Optional[cute.Tensor],
    # -- runtime geometry --
    n_q: cutlass.Int32,
    n_k: cutlass.Int32,
    n_red: cutlass.Int32,  # 2 d (Q columns then K columns), or 0
    n_rows: cutlass.Int32,
    n_gate: cutlass.Int32,  # the gate partials written (the gate backward's grid)
    n_bands: cutlass.Int32,  # the band partials written (the norm backward's grid)
    n_cast: cutlass.Int32,  # the cast blocks: under `persistent` every one strides over the row groups cast_cta, cast_cta + n_cast, ...
    # -- compile-time facts --
    d: cutlass.Constexpr[int],
    unroll: cutlass.Constexpr[int],
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    n_alpha: cutlass.Constexpr[int],
    margin_log2: cutlass.Constexpr[int],
    persistent: cutlass.Constexpr[bool],
) -> None:
    """Block-range dispatch over the two jobs (module docstring)."""
    want_dw = cutlass.const_expr(mPq is not None)
    derive = cutlass.const_expr(mScale is None)
    # the reduce arm's combine array: one fp32 per thread (cols = 1, lanes = threads); the cast arm's warp-maxima array
    sPart = cutlass.Array(cutlass.Float32, threads_per_cta, alignment=16, space=cutlass.AddressSpace.smem) if cutlass.const_expr(want_dw) else None
    sRed = cutlass.Array(cutlass.Float32, threads_per_cta // 32, alignment=16, space=cutlass.AddressSpace.smem)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    cta = cutlass.Int32(cute.arch.block_idx()[0])
    if cta < n_red:
        if cutlass.const_expr(want_dw):
            is_q = cta < cutlass.Int32(d)
            col = cta if is_q else cta - cutlass.Int32(d)
            p_base = mPq.iterator.toint() if is_q else mPk.iterator.toint()
            o_base = mDWq.iterator.toint() if is_q else mDWk.iterator.toint()
            n = n_q if is_q else n_k
            dw_reduce_column(p_base, o_base, n, col, tidx, cutlass.Int32(0), d, 1, threads_per_cta, unroll, sPart)
    else:
        cast_cta = cta - n_red
        # the amax, reduced in EVERY block from the two producers' partials (never read from the published copy the first block
        # writes), and the scale derived from it -- or the caller's under "delayed", the amax still published
        amax_val = cta_max_of_partials_pair(mPartialsGate, n_gate, mPartialsBands, n_bands, sRed, threads_per_cta)
        scale = _grad_scale_bits(amax_val, margin_log2) if cutlass.const_expr(derive) else _slot_value(mScale)
        if (cast_cta == cutlass.Int32(0)) & (tidx == cutlass.Int32(0)):
            publish_scale(
                scale, amax_val, mScaleOut, mDescale, mAmaxOut, mAlphaC0, mAlphaC1, mAlphaC2, mAlphaC3, mAlphaO0, mAlphaO1, mAlphaO2, mAlphaO3, n_alpha
            )
        if cutlass.const_expr(persistent):
            # persistent over the row groups: the reduce above is paid once per block
            rows_per_cta = cutlass.const_expr((threads_per_cta // lanes_per_row(d)) * rows_per_group)
            n_groups = (n_rows + cutlass.Int32(rows_per_cta - 1)) // cutlass.Int32(rows_per_cta)
            n_iters = (n_groups - cast_cta + n_cast - cutlass.Int32(1)) // n_cast
            for it in cutlass.range(n_iters):
                quantize_rows(
                    mSrc, mDst, scale, n_rows, cutlass.Int32(h_ct), cast_cta + it * n_cast, h_ct, const_head_count, d, threads_per_cta, rows_per_group
                )
        else:
            quantize_rows(mSrc, mDst, scale, n_rows, cutlass.Int32(h_ct), cast_cta, h_ct, const_head_count, d, threads_per_cta, rows_per_group)


@cute.jit
def fp8_bwd_epilogue_launch(
    p_q: Optional[cute.Tensor],
    p_k: Optional[cute.Tensor],
    dw_q: Optional[cute.Tensor],
    dw_k: Optional[cute.Tensor],
    src: cute.Tensor,
    dst: cute.Tensor,
    scale: Optional[cute.Tensor],
    partials_gate: cute.Tensor,
    partials_bands: cute.Tensor,
    amax_out: cute.Tensor,
    scale_out: cute.Tensor,
    descale: cute.Tensor,
    alpha_c0: Optional[cute.Tensor],
    alpha_c1: Optional[cute.Tensor],
    alpha_c2: Optional[cute.Tensor],
    alpha_c3: Optional[cute.Tensor],
    alpha_o0: Optional[cute.Tensor],
    alpha_o1: Optional[cute.Tensor],
    alpha_o2: Optional[cute.Tensor],
    alpha_o3: Optional[cute.Tensor],
    n_q: cutlass.Int32,
    n_k: cutlass.Int32,
    n_red: cutlass.Int32,
    n_rows: cutlass.Int32,
    n_gate: cutlass.Int32,
    n_bands: cutlass.Int32,
    n_cast: cutlass.Int32,
    n_blocks: cutlass.Int32,
    d: cutlass.Constexpr[int],
    unroll: cutlass.Constexpr[int],
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    n_alpha: cutlass.Constexpr[int],
    margin_log2: cutlass.Constexpr[int],
    persistent: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_fp8_bwd_epilogue(
        p_q,
        p_k,
        dw_q,
        dw_k,
        src,
        dst,
        scale,
        partials_gate,
        partials_bands,
        amax_out,
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
        n_q,
        n_k,
        n_red,
        n_rows,
        n_gate,
        n_bands,
        n_cast,
        d,
        unroll,
        h_ct,
        const_head_count,
        threads_per_cta,
        rows_per_group,
        n_alpha,
        margin_log2,
        persistent,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream)


epilogue_cache = {}


class Fp8BwdEpilogueRecipe(NamedTuple):
    """Build-time facts of the epilogue launch."""

    compiled: object
    dtype: object
    h: int  # N / d: the dqkvg view's head count
    d: int
    want_dw: bool
    scale_src: str
    n_alpha: int
    margin_log2: int
    rows_per_cta: int
    threads: int
    # Appended: the cast job's persistent cap (SMs x AMAX_CTAS_PER_SM on the compiling device; ``epilogue_grid``) and whether the
    # cast strides it (EPILOGUE_CAST_PERSISTENT at compile time; False = one row group per block).
    n_ctas_cap: int = 0
    cast_persistent: bool = False


def compile_fp8_bwd_epilogue(
    *,
    dtype,
    n_cols: int,
    d: int,
    want_dw: bool,
    scale_src: str,
    n_alpha: int,
    margin_log2: int = 0,
    threads_per_cta: int = THREADS,
    rows_per_group: int = EPILOGUE_CAST_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
) -> Fp8BwdEpilogueRecipe:
    """Build from SHAPES ALONE.  ``n_cols`` is the dqkvg slab's width ``N`` (``N % d == 0``); ``want_dw`` traces the reduce arm
    (``n_red = 2 d`` blocks at ``lanes = threads_per_cta``, which must equal ``REDUCE_LANES`` so the per-column chain is the standalone
    reduce's); ``scale_src`` ``"amax"`` derives the scale from the amax the cast blocks reduce from the two producers' partials,
    ``"given"`` reads the caller's (the amax is still reduced and published); the publish arm is always on (the cast publishes
    amax_dqkvg / scale / descale / the ``n_alpha`` alphas); ``rows_per_group`` defaults to the cast's measured
    ``EPILOGUE_CAST_ROWS_PER_GROUP``.  Needs the fp8 ``cvt`` (sm_89+; declined by name)."""
    if n_cols % d != 0:
        raise ValueError(f"n_cols={n_cols} must be a multiple of d_head={d}: dqkvg is quantized through a [T, N / D, D] view")
    validate_shape(d, threads_per_cta)
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"the fp8 backward epilogue serves bf16/f16 slabs only, got {dtype}")
    if scale_src not in SCALE_SOURCES:
        raise ValueError(f"scale_src must be one of {SCALE_SOURCES}, got {scale_src!r}")
    if isinstance(n_alpha, bool) or not isinstance(n_alpha, int) or n_alpha < 0 or n_alpha > MAX_ALPHA:
        raise ValueError(f"n_alpha must be an int in [0, {MAX_ALPHA}], got {n_alpha!r}")
    if isinstance(margin_log2, bool) or not isinstance(margin_log2, int):
        raise ValueError(f"margin_log2 must be an int, got {margin_log2!r}")
    if want_dw and threads_per_cta != REDUCE_LANES:
        raise ValueError(
            f"the fused dW_norm reduce runs at lanes = threads_per_cta = {threads_per_cta}, but the standalone reduce's fixed summation order is "
            f"REDUCE_LANES = {REDUCE_LANES} residue classes; the two must agree or the sum is a different fp32 chain"
        )
    require_fp8_cvt("compile_fp8_bwd_epilogue")
    device = current_device()
    h = n_cols // d
    persistent = bool(EPILOGUE_CAST_PERSISTENT)
    key = (
        str(dtype),
        int(h),
        int(d),
        bool(want_dw),
        scale_src,
        int(n_alpha),
        int(margin_log2),
        int(threads_per_cta),
        int(rows_per_group),
        bool(const_head_count),
        device,
        persistent,
    )
    if key not in epilogue_cache:
        tok = cute.sym_int()
        planes = [_fake_tma(torch.float32, (cute.sym_int(), d), (1, 0)) for _ in range(2)] if want_dw else [None, None]
        outs32 = [_fake_tma(torch.float32, (d,), (0,)) for _ in range(2)] if want_dw else [None, None]
        src = fake_rowmajor_dynamic_token_stride(dtype, tok, h, d)
        dst = cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(tok, h, d), stride=(cute.sym_int(), d, 1), assumed_align=16)
        scale = _fake_slot() if scale_src == "given" else None
        alpha_c = [_fake_slot() if i < n_alpha else None for i in range(MAX_ALPHA)]
        alpha_o = [_fake_slot() if i < n_alpha else None for i in range(MAX_ALPHA)]
        epilogue_cache[key] = cute.compile(
            fp8_bwd_epilogue_launch,
            *planes,
            *outs32,
            src,
            dst,
            scale,
            _fake_partials(),  # the gate partials
            _fake_partials(),  # the band partials
            _fake_slot(),  # amax_out
            _fake_slot(),  # scale_out
            _fake_slot(),  # descale
            *alpha_c,
            *alpha_o,
            cutlass.Int32(0),  # n_q      ) runtime; the values pin the TYPE only
            cutlass.Int32(0),  # n_k      )
            cutlass.Int32(0),  # n_red    )
            cutlass.Int32(0),  # n_rows   )
            cutlass.Int32(1),  # n_gate   )
            cutlass.Int32(1),  # n_bands  )
            cutlass.Int32(1),  # n_cast   )
            cutlass.Int32(1),  # n_blocks )
            int(d),
            REDUCE_UNROLL,
            int(h),
            bool(const_head_count),
            int(threads_per_cta),
            int(rows_per_group),
            int(n_alpha),
            int(margin_log2),
            persistent,
            _fake_stream(),
            options="--enable-tvm-ffi",
        )
    return Fp8BwdEpilogueRecipe(
        compiled=epilogue_cache[key],
        dtype=dtype,
        h=int(h),
        d=int(d),
        want_dw=bool(want_dw),
        scale_src=scale_src,
        n_alpha=int(n_alpha),
        margin_log2=int(margin_log2),
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        threads=int(threads_per_cta),
        n_ctas_cap=int(multiprocessor_count(device)) * AMAX_CTAS_PER_SM,
        cast_persistent=persistent,
    )


def epilogue_grid(r: Fp8BwdEpilogueRecipe, t: int) -> tuple:
    """``(n_red, n_cast)`` for ``t`` tokens -- the two job widths of the launch (its grid is their sum): ``2 d`` reduce
    blocks under ``want_dw`` (0 otherwise) and the cast blocks -- ``min(ceil(t * h / rows_per_cta), cap)`` on a persistent
    recipe, the row groups otherwise (at least 1)."""
    t = int(t)
    groups = max(1, (t * r.h + r.rows_per_cta - 1) // r.rows_per_cta)
    cap = int(getattr(r, "n_ctas_cap", 0))
    n_cast = min(groups, cap) if (getattr(r, "cast_persistent", False) and cap > 0) else groups
    return (2 * r.d if r.want_dw else 0), n_cast


def run_fp8_bwd_epilogue(
    r: Fp8BwdEpilogueRecipe,
    *,
    plane_q: Optional[torch.Tensor],
    plane_k: Optional[torch.Tensor],
    dw_q: Optional[torch.Tensor],
    dw_k: Optional[torch.Tensor],
    src: torch.Tensor,
    dst: torch.Tensor,
    scale: Optional[torch.Tensor],
    partials_gate: torch.Tensor,
    n_gate: int,
    partials_bands: torch.Tensor,
    n_bands: int,
    amax_out: torch.Tensor,
    scale_out: torch.Tensor,
    descale: torch.Tensor,
    alpha_consts: tuple,
    alpha_outs: tuple,
    stream,
) -> None:
    """Launch.  Under ``want_dw`` the partial planes (``[n_q, D]`` / ``[n_k, D]`` fp32 contiguous, EXACTLY the rows the norm
    backward wrote) and the two fp32 ``[D]`` outputs are REQUIRED, refused otherwise; ``src`` ``[T, N / D, D]`` bf16 (the dqkvg
    slab), ``dst`` its e4m3 twin; ``scale`` REQUIRED under ``"given"`` and refused under ``"amax"``; ``partials_gate`` /
    ``partials_bands`` the two producers' contiguous fp32 partials arrays with ``n_gate`` / ``n_bands`` the words each wrote
    (``quantize.check_partials``), ``amax_out`` the slot the reduced amax is published to (never a word of either array);
    ``scale_out`` / ``descale`` / the ``n_alpha`` alpha pairs as the quantize launch takes them.  Host checks only."""
    if r.want_dw:
        if plane_q is None or plane_k is None or dw_q is None or dw_k is None:
            raise ValueError("this artifact was compiled WITH the dW_norm reduce (want_dw=True): plane_q, plane_k, dw_q, dw_k must all be bound (Rule 1)")
        _check_fp32_plane("plane_q", plane_q, (None, r.d))
        _check_fp32_plane("plane_k", plane_k, (None, r.d))
        _check_fp32_plane("dw_q", dw_q, (r.d,))
        _check_fp32_plane("dw_k", dw_k, (r.d,))
        n_q, n_k = int(plane_q.shape[0]), int(plane_k.shape[0])
        if n_q < 1 or n_k < 1:
            raise ValueError(f"the partial planes need at least one row each, got {n_q} / {n_k}")
    else:
        if plane_q is not None or plane_k is not None or dw_q is not None or dw_k is not None:
            raise ValueError(
                "this artifact was compiled WITHOUT the dW_norm reduce (want_dw=False); passing planes / dw outputs would silently ignore them (Rule 1)"
            )
        n_q = n_k = 0
    t = int(src.shape[0])
    _check_band("src", src, t, r.h, r.d, r.dtype)
    check_e4m3_out("dst", dst, t, r.h, r.d)
    if r.scale_src == "given":
        if scale is None:
            raise ValueError("this artifact reads the caller's scale (scale_src='given'); scale must be bound at execute (Rule 1: no silent fallback)")
    else:
        if scale is not None:
            raise ValueError(
                "this artifact derives its scale from the reduced amax (scale_src='amax'); passing a caller scale would silently ignore it (Rule 1)"
            )
    n_gate = check_partials("partials_gate", partials_gate, n_gate)
    n_bands = check_partials("partials_bands", partials_bands, n_bands)
    if len(alpha_consts) != r.n_alpha or len(alpha_outs) != r.n_alpha:
        raise ValueError(f"this artifact publishes n_alpha={r.n_alpha} alpha products; got {len(alpha_consts)} alpha_consts and {len(alpha_outs)} alpha_outs")
    slots = [("scale", scale), ("amax_out", amax_out), ("scale_out", scale_out), ("descale", descale)]
    slots += [(f"alpha_consts[{i}]", c) for i, c in enumerate(alpha_consts)] + [(f"alpha_outs[{i}]", a) for i, a in enumerate(alpha_outs)]
    for name, ten in slots:
        if ten is not None:
            check_scalar_slot(name, ten)
    reads = [("scale", scale)] + [(f"alpha_consts[{i}]", c) for i, c in enumerate(alpha_consts)]
    writes = [("amax_out", amax_out), ("scale_out", scale_out), ("descale", descale)] + [(f"alpha_outs[{i}]", a) for i, a in enumerate(alpha_outs)]
    for wn, w in writes:
        for rn, rd in reads:
            if rd is not None and w.data_ptr() == rd.data_ptr():
                raise ValueError(f"{wn} aliases {rn}: a published slot cannot be a slot the launch reads (the other blocks read it while the first writes)")
        for pn, parts, n in (("partials_gate", partials_gate, n_gate), ("partials_bands", partials_bands, n_bands)):
            if parts.data_ptr() <= w.data_ptr() < parts.data_ptr() + 4 * n:
                raise ValueError(f"{wn} lies inside {pn}: a published slot cannot be a word the other blocks reduce")
    for i, (wn, w) in enumerate(writes):
        for vn, v in writes[i + 1 :]:
            if w.data_ptr() == v.data_ptr():
                raise ValueError(f"{wn} and {vn} are the same slot: two published values cannot share one word")
    dev = src.device
    for name, ten in [("dst", dst), ("plane_q", plane_q), ("plane_k", plane_k), ("dw_q", dw_q), ("dw_k", dw_k)] + slots:
        if ten is not None and (not ten.is_cuda or ten.device != dev):
            raise ValueError(f"{name} must be on {dev} with src (the kernel reads every operand through a device pointer), got {ten.device}")
    for name, ten in (("partials_gate", partials_gate), ("partials_bands", partials_bands)):
        if ten.device != dev:
            raise ValueError(f"{name} must be on {dev} with src (the kernel reads every operand through a device pointer), got {ten.device}")
    n_rows = t * r.h
    n_red, n_cast = epilogue_grid(r, t)
    alpha_c = list(alpha_consts) + [None] * (MAX_ALPHA - r.n_alpha)
    alpha_o = list(alpha_outs) + [None] * (MAX_ALPHA - r.n_alpha)
    r.compiled(
        plane_q,
        plane_k,
        dw_q,
        dw_k,
        src,
        dst,
        _slot_view(scale),
        partials_gate,
        partials_bands,
        _slot_view(amax_out),
        _slot_view(scale_out),
        _slot_view(descale),
        *[_slot_view(c) for c in alpha_c],
        *[_slot_view(a) for a in alpha_o],
        cutlass.Int32(n_q),
        cutlass.Int32(n_k),
        cutlass.Int32(n_red),
        cutlass.Int32(n_rows),
        cutlass.Int32(n_gate),
        cutlass.Int32(n_bands),
        cutlass.Int32(n_cast),
        cutlass.Int32(n_red + n_cast),
        cuda.CUstream(int(stream)),
    )


frost_fp8_bwd_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_fp8_bwd_epilogue.set_name_prefix("cudnn", remove_cutlass_symbol=True)
