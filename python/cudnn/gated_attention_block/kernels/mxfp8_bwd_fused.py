# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 (block-scaled) block backward's two FUSED small-kernel launches: the PROLOGUE before the first GEMM and the EPILOGUE
after the norm backward -- the MXFP8 twin of ``fp8_bwd_fused.py``.  One ``@cute.kernel`` each, several independent jobs behind a
BLOCK-RANGE dispatch (a per-block uniform branch on ``blockIdx.x``, so no lane ever diverges on the job), every job the very
``@cute.jit`` body its standalone kernel runs -- the same code, two launch shapes, bitwise the same bytes.

Why: the unfused MXFP8 backward spends ~19 % of its device time (S = 2K .. 8K) in twelve standalone quantize / amax / init launches
plus a bf16 Q / K rebuild whose only readers are four of those quantizes, and the three COLUMNWISE SDPA-layout quantizes among them
are the 2-byte-store arm of the standalone kernel (a third of the rate of the rowwise arm on the same bytes).  The MX block
constraint is what kept them apart: a rowwise E8M0 block is 32 consecutive elements of one row (any warp-per-row kernel holds one),
a columnwise block is 32 consecutive TOKENS of one ``(h, d)`` -- which the rebuild's one-token tile and the row-elementwise kernels
never hold.  The MX epilogue of the rebuild (``qk_norm_rope_tma.qk_norm_rope_tma_mx_body``: ONE head x 32 TOKENS per TMA tile) and the
dual-axis quantizer (``quantize_mxfp8.quantize_mxfp8_dual_body``: one read, both quantizations) hold it; this module puts them behind
two launches.

**Prologue** (``frost_mxfp8_bwd_prologue``; before the out-projection dgrad; grid = ``1 + n_amax + n_rec + n_v8``)::

    blocks [0, 1)                     scalar init          slots[0:n] = 0, then the n_consts PLAN-TIME CONSTANTS into their slots out of the
                                                           launch's fp32 KERNEL ARGUMENTS (quantize.init_scalars_body, descale_dp=False: the MXFP8
                                                           row has no dP scalar; one thread)
    blocks [1, 1 + n_amax)            dY amax PARTIALS     partials[c] = max |dY| over CTA c's row groups (quantize.amax_partials_rows; persistent)
    blocks [.., + n_rec)              Q / K rebuild, MX    norm + RoPE from the slab's PRE-norm bands, the ROWWISE (q8 / sf_q, k8 / sf_k) AND the
                                                           COLUMNWISE (q_T8 / sf_q_T, k_T8 / sf_k_T) MXFP8 quantizes from the same 32-token tile
                                                           (qk_norm_rope_tma_mx_body; persistent TMA ring; no bf16 recompute buffer)
    blocks [.., + n_v8)               v8                   the ROWWISE MXFP8 quantize of the slab's V band (V's compaction): one block per
                                                           (b, h_kv, 128-row SF unit) (quantize_mxfp8_rowwise_body)

No job reads anything another job of the same launch writes: the init zeroes the scalar block (every slot a LATER launch publishes
-- ``amax_dy`` / ``scale_dy`` / ``descale_dy`` / the two alphas by the dY cast -- starts from a known zero) and nothing in this launch
touches a slot.  The partials are written unconditionally (one per amax CTA).  Nothing is filled on the device at compile time.

**Epilogue** (``frost_mxfp8_bwd_epilogue``; after the norm backward; 256 threads; grid = ``n_red + n_cast``)::

    blocks [0, n_red)                 dW_norm reduce       columns cols blk .. cols blk + cols - 1 of class (cols blk) // d: the fixed-order sum of
                                                           the partial planes (qk_norm_rope_bwd.dw_reduce_column at (cols, lanes) = (threads // 128,
                                                           128) = (2, 128) at the 256-thread default: the per-column chain never depends on cols, so
                                                           it is the standalone (8, 128) reduce's sum, bitwise); n_red = ceil(2 d / cols) blocks
                                                           (epilogue_reduce_blocks: d at 256 threads, 2 d at 128, d / 2 at 512), or 0 without dW
    blocks [n_red, n_red + n_cast)    dqkvg DUAL-AXIS cast one block per (h = N / D, 128-token unit): dqkvg8 [T, N] + the canonical blob over (T, N)
                                                           (need_dh: the block-scale dgrad's A) AND dqkvg_t8 [N, T] + the canonical blob over (N, T)
                                                           (need_dw_qkvg: the block-scale wgrad's A) from ONE read (quantize_mxfp8_dual_body at
                                                           sf_layout="gemm", transposed_second); a half that is not needed is folded out

**Shared geometry.** One launch has one block size (the prologue 128 threads: the rebuild body's; the epilogue 256: the dual-axis
body's, the reduce at ``(2, 128)``), one SMEM footprint (the prologue's is the rebuild's 32-token input ring + its column-maxima array
+ the ``mb_full`` barriers, plus the v8 job's 1 KiB SF tile and the amax job's warp-maxima word per warp; the epilogue's is the dual
body's 16 KiB staging tile + the scale array + two SF tiles, plus the reduce's ``sPart``) and one register count (the heaviest
job's).  A light job therefore runs at the heavy job's residency: pin a fused job's HBM GB/s against its standalone set's before
trusting the fusion -- a loss is fixed by giving that job more rows per block, never reported.

**The rebuild job's persistent cap is its RESIDENCY, not the amax job's.** The MX token-tile arm is resident 5 CTAs / SM on the norm
arm (REG 90 -> 65536 / (128 x 96)) and 6 RoPE-only (SMEM-bound) -- ``qk_norm_rope_tma.mx_rebuild_ctas_per_sm`` -- so ``n_rec`` is
capped at ``SMs x`` that (``n_rec_cap``), one grid-stride wave.  The amax job's ``SMs x 8`` cap on the same job launched 1.6 waves:
measured on Rubin cc 10.7 (212 SMs, locked clocks, CUPTI device time, 397B geometry, S = 8K) the norm arm 0.1105 -> 0.1021 ms
(+8 %) and the RoPE-only arm 0.146 -> 0.128 ms (+14 %) at the residency cap, every cap bitwise identical.  At the residency cap
the fused PROLOGUE moves its 354 MiB in 0.099 ms -- 3.7 TB/s on its own bytes -- against the eight standalone launches it replaces at
0.184 ms (+85 % on device time), whose set reads 4.3 TB/s on ITS bytes (the fused launch sits 14 % below the set's rate: the
"within 10 % of the set's rate" criterion is NOT met), and against the per-tensor fp8 PROLOGUE's 6.3 TB/s in the same loop (-41 %;
the two-pass token tile at 5-6 CTAs / SM against the shipped one-token x 16-head tile at 14 -- the rate criterion set for the
columnwise half, within 10 % of the per-tensor prologue's rate, is not met either).  The fallback shape (the rowwise MX epilogue on
the shipped tile + the bf16 store + one dual-axis launch over the bf16 buffers) is the open alternative; the workspace carve is keyed
on the prologue's arm, so it plugs in without re-opening the carve.  Rubin cc 10.7, 212 SMs, SM clock locked, CUPTI device time,
the 397B geometry at S = 8K; the byte models behind the GB/s are ``prologue_moved_bytes`` / ``prologue_standalone_set_bytes`` below
(the host-only shared-geometry pin of the test module; the rate itself is measured by the perf tooling, never asserted by a test).

Barrier table (every arrive LOCAL; no cross-CTA arrive, no drain): the TMA ring's ``mb_full`` (``qk_norm_rope_tma.py``'s MX rows:
``SUM(issuing lanes) == init`` 1 == 1) and the two per-tile ``bar.sync`` of the MX body in the recompute arm only (every thread of the
block, 128 == 128; the writers' ``fence.proxy.async`` before the second); ``barrier_cta_sync`` in the prologue's amax arm (the
warp-maxima combine) and v8 arm (before the SF burst); in the epilogue the reduce arm's ONE barrier inside ``dw_reduce_column`` and the
cast arm's two per 32-token sub-tile + one before the bursts -- every one reached by all 256 threads (the dispatch is block-uniform
and no divergent path precedes them).  SMEM buffer table: the recompute arm's ``sIn`` / ``sRedCol`` (``qk_norm_rope_tma.py``'s MX
rows), the v8 arm's ``sSF`` (the standalone's atom-local stores), ``sRed`` fp32 ``[warps]`` (one word per warp); the epilogue's
``sStage`` / ``sScale`` / ``sSFrow`` / ``sSFcol`` (``quantize_mxfp8.py``'s dual-body table) and ``sPart`` fp32 ``[256]`` (one word per
thread).  Nothing here can hang that the standalone kernels could not.
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

from .fp8_bwd_fused import _check_band
from .qk_norm_rope import check_norm_weights_match_recipe
from .qk_norm_rope_bwd import REDUCE_LANES, REDUCE_UNROLL, _check_fp32_plane, dw_reduce_column
from .qk_norm_rope_tma import (
    DEFAULT_STAGES,
    MX_TILE_TOKENS,
    TMA_GRANU_ELEMS,
    _bshd_view,
    _fake as _fake_tma,
    _fake_bshd,
    _fake_e4m3_thd,
    _fake_u8_flat,
    check_e4m3_out,
    check_mx_sf_blob,
    mx_rebuild_ctas_per_sm,
    mx_tile_counts,
    qk_norm_rope_tma_mx_body,
    validate_mx_shape,
)
from .quantize import (
    AMAX_CTAS_PER_SM,
    DEFAULT_CONST_HEAD_COUNT,
    DEFAULT_ROWS_PER_GROUP,
    DEFAULT_THREADS_PER_CTA,
    MAX_INIT_CONSTS,
    _fake_partials,
    _fake_slot,
    _slot_view,
    amax_moved_bytes,
    amax_partials_rows,
    check_partials,
    check_scalar_slot,
    fake_rowmajor_dynamic_token_stride,
    init_scalars_body,
    lanes_per_row,
    require_fp8_cvt,
    validate_shape,
)
from .quantize_mxfp8 import (
    SF_BLOCK,
    SF_LAYOUT_GEMM,
    QuantizeMxfp8DualRecipe,
    check_dual_operands,
    moved_bytes,
    n_sf_tiles,
    quantize_mxfp8_dual_body,
    quantize_mxfp8_rowwise_body,
    sf_bytes,
    sf_tile_bytes,
    validate_dual_shape,
)
from .quantize_mxfp8 import validate_shape as _validate_mx_quant_shape

_FAKE_STREAM = None
PROLOGUE_THREADS = DEFAULT_THREADS_PER_CTA  # 128: the rebuild body's block (every prologue job's own)
EPILOGUE_THREADS = 256  # the dual-axis body's block; the reduce runs at (cols, lanes) = (EPILOGUE_THREADS // REDUCE_LANES, REDUCE_LANES)


def epilogue_reduce_cols(threads_per_cta: int) -> int:
    """Columns per reduce block: the reduce arm's thread mapping is ``(cols, lanes) = (threads_per_cta // REDUCE_LANES, REDUCE_LANES)``
    (2 at the 256-thread default) -- the kernel body and the grid read the same function."""
    return int(threads_per_cta) // REDUCE_LANES


def epilogue_reduce_blocks(d: int, threads_per_cta: int) -> int:
    """``n_red``: the reduce blocks that cover the ``2 d`` columns (``d`` Q columns, then ``d`` K columns) at ``epilogue_reduce_cols``
    columns each -- ``ceil(2 d / cols)``, DERIVED from the block size: ``d`` at the default 256 threads, ``2 d`` at 128 (one column per
    block), ``d / 2`` at 512.  A grid of ``d`` blocks at every block size covered the Q half only at 128 threads and left ``dW_k_norm``
    unwritten.  Every block size the dual-axis cast accepts divides ``2 d`` exactly; a column past ``2 d`` would idle in the body's
    ``col < d`` guard anyway."""
    return -(-2 * int(d) // epilogue_reduce_cols(threads_per_cta))


__all__ = [
    "EPILOGUE_THREADS",
    "Mxfp8BwdEpilogueRecipe",
    "Mxfp8BwdPrologueRecipe",
    "PROLOGUE_THREADS",
    "compile_mxfp8_bwd_epilogue",
    "compile_mxfp8_bwd_prologue",
    "epilogue_grid",
    "epilogue_reduce_blocks",
    "epilogue_reduce_cols",
    "prologue_grid",
    "run_mxfp8_bwd_epilogue",
    "run_mxfp8_bwd_prologue",
]


def _fake_stream():
    global _FAKE_STREAM
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    return _FAKE_STREAM


# ---------------------------------------------------------------------------
# PROLOGUE: init | dY amax partials | Q / K rebuild with the MX epilogue | v8 (MX rowwise)
# ---------------------------------------------------------------------------


@cute.kernel
def frost_mxfp8_bwd_prologue(
    # -- the scalar init (block 0) --
    mSlots: cute.Tensor,  # [n_slots] fp32 contiguous: the scalar block
    mScaleDp: cute.Tensor,  # [1] fp32: bound for the init body's ABI, NEVER read (descale_dp=False: no dP scalar on the MXFP8 row)
    mDescaleDpOut: cute.Tensor,  # [1] fp32: bound, NEVER written (same)
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
    # -- the Q / K rebuild with the MX epilogue (blocks [1 + n_amax, + n_rec)) --
    mQ: cute.Tensor,  # the (B, S, H_q, D) view of the slab's PRE-norm Q band (its element_type; the data moves through the descriptors)
    mWq: Optional[cute.Tensor],  # [D] norm weights; None (both) = RoPE-only
    mWk: Optional[cute.Tensor],
    mCos: cute.Tensor,  # [T, ROPE_DIM]
    mSin: cute.Tensor,
    mQ8: cute.Tensor,  # [T, H_q, D] e4m3 OUT (compact): the rowwise payload
    mSfQ: cute.Tensor,  # the SDPA rowwise SF tiles
    mQT8: cute.Tensor,  # [T, H_q, D] e4m3 OUT: the columnwise payload
    mSfQT: cute.Tensor,  # the SDPA columnwise D-plane-major atoms
    mK8: cute.Tensor,  # the K twins over H_kv
    mSfK: cute.Tensor,
    mKT8: cute.Tensor,
    mSfKT: cute.Tensor,
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    # -- the v8 cast (blocks [1 + n_amax + n_rec, + n_v8)) --
    mV: cute.Tensor,  # [T, H_kv, D] bf16: the slab's V band
    mV8: cute.Tensor,  # [T, H_kv, D] e4m3 OUT (compact: V's compaction)
    mSfV: cute.Tensor,  # the SDPA rowwise SF tiles of v8
    # -- runtime geometry --
    n_dy_rows: cutlass.Int32,
    n_dy_groups: cutlass.Int32,
    n_amax: cutlass.Int32,
    seq_len: cutlass.Int32,
    batch: cutlass.Int32,
    n_blk: cutlass.Int32,  # ceil128(S) / 32
    n_sft: cutlass.Int32,  # ceil(S / 128)
    n_tiles_mx: cutlass.Int32,  # B * n_blk * (h_q + h_kv)
    n_rec: cutlass.Int32,
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
    stages: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
) -> None:
    """Block-range dispatch over the four jobs (module docstring); every arm is the standalone kernel's body."""
    tile_elems = cutlass.const_expr(MX_TILE_TOKENS * d)
    io_dtype = mQ.element_type
    # SMEM, allocated ONCE here for every arm (an allocation inside an inlined jit body ships one copy per arm): the TMA input ring +
    # its barriers + the per-warp column maxima (the rebuild), the warp-maxima word per warp (the amax), the SF tile (v8).
    sIn_raw = cutlass.Array(io_dtype, stages * tile_elems, alignment=128, space=cutlass.AddressSpace.smem)
    mb_full = cutlass.Array(cutlass.Int64, stages, alignment=16, space=cutlass.AddressSpace.smem)
    sRedCol = cutlass.Array(cutlass.Float32, (threads_per_cta // 32) * d, alignment=16, space=cutlass.AddressSpace.smem)
    sRed = cutlass.Array(cutlass.Float32, threads_per_cta // 32, alignment=16, space=cutlass.AddressSpace.smem)
    sSF = cutlass.Array(cutlass.Uint8, sf_tile_bytes(d), alignment=16, space=cutlass.AddressSpace.smem)

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
            False,  # descale_dp: the MXFP8 row has no dP scalar -- the two scalar tensors are bound and never touched
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
                qk_norm_rope_tma_mx_body(
                    mQ,
                    mWq,
                    mWk,
                    mCos,
                    mSin,
                    GmemTileTma(tma_q_desc),
                    GmemTileTma(tma_k_desc),
                    mQ8,
                    mSfQ,
                    mQT8,
                    mSfQT,
                    mK8,
                    mSfK,
                    mKT8,
                    mSfKT,
                    seq_len,
                    batch,
                    n_blk,
                    n_sft,
                    n_tiles_mx,
                    cta - base_rec,
                    n_rec,
                    eps,
                    d,
                    rope_dim,
                    stages,
                    threads_per_cta,
                    h_q_ct,
                    h_kv_ct,
                    sIn_raw,
                    sRedCol,
                    mb_full,
                )
            else:
                # one block per (b * h_kv + h, s_tile) unit of the V band: the standalone rowwise kernel's (blockIdx.x, blockIdx.y)
                u = cta - base_v8
                quantize_mxfp8_rowwise_body(mV, mV8, mSfV, seq_len, n_sft, u // n_sft, u % n_sft, h_kv_ct, d, threads_per_cta, sSF)


@cute.jit
def mxfp8_bwd_prologue_launch(
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
    q4: cute.Tensor,  # the (B, S, H_q, D) view of the Q band
    k4: cute.Tensor,  # the (B, S, H_kv, D) view of the K band
    w_q: Optional[cute.Tensor],
    w_k: Optional[cute.Tensor],
    cos: cute.Tensor,
    sin: cute.Tensor,
    q8: cute.Tensor,
    sf_q: cute.Tensor,
    q_T8: cute.Tensor,
    sf_q_T: cute.Tensor,
    k8: cute.Tensor,
    sf_k: cute.Tensor,
    k_T8: cute.Tensor,
    sf_k_T: cute.Tensor,
    v: cute.Tensor,
    v8: cute.Tensor,
    sf_v: cute.Tensor,
    n_dy_rows: cutlass.Int32,
    n_dy_groups: cutlass.Int32,
    n_amax: cutlass.Int32,
    seq_len: cutlass.Int32,
    batch: cutlass.Int32,
    n_blk: cutlass.Int32,
    n_sft: cutlass.Int32,
    n_tiles_mx: cutlass.Int32,
    n_rec: cutlass.Int32,
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
    stages: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """The MX arm's descriptors exactly as ``qk_norm_rope_tma_launch`` builds them under ``mx_out`` (4-D ``(B, S, H, D)`` views, box
    ``(1, 32, 1, 128)``, unswizzled, the slab's token stride and the batch stride in the descriptor), then the one launch."""
    box = (1, MX_TILE_TOKENS, 1, TMA_GRANU_ELEMS)
    order = (3, 2, 1, 0)
    mk = lambda t: tmap.create_tensor_map_tiled_from_view(  # noqa: E731
        t,
        box_dims=box,
        stride_order=order,
        swizzle=tmap.TensorMapSwizzle.none,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    frost_mxfp8_bwd_prologue(
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
        q4,
        w_q,
        w_k,
        cos,
        sin,
        q8,
        sf_q,
        q_T8,
        sf_q_T,
        k8,
        sf_k,
        k_T8,
        sf_k_T,
        mk(q4),
        mk(k4),
        v,
        v8,
        sf_v,
        n_dy_rows,
        n_dy_groups,
        n_amax,
        seq_len,
        batch,
        n_blk,
        n_sft,
        n_tiles_mx,
        n_rec,
        eps,
        n_slots,
        const_slot0,
        n_consts,
        h_dy_ct,
        h_q_ct,
        h_kv_ct,
        d,
        rope_dim,
        stages,
        threads_per_cta,
        rows_per_group,
        const_head_count,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream)


prologue_cache = {}


class Mxfp8BwdPrologueRecipe(NamedTuple):
    """Build-time facts of the prologue launch (every field plan-time derivable; ``batch`` / ``seq_len`` ride in at run time)."""

    compiled: object
    dtype: object
    h_q: int
    h_kv: int
    h_dy: int  # d_model / d: the dY view's head count
    d: int
    rope_dim: int
    eps: float
    stages: int
    apply_norm: bool
    n_slots: int
    rows_per_cta: int  # the quantize layout's rows per block (the amax job)
    n_ctas_cap: int  # SMs x 8: the persistent cap of the amax job (and of the rebuild job when n_rec_cap is 0)
    threads: int
    const_slot0: int = 0
    n_consts: int = 0
    # Appended: the rebuild job's persistent cap = SMs x its RESIDENCY (qk_norm_rope_tma.mx_rebuild_ctas_per_sm: 5 CTAs / SM on the
    # norm arm, 6 RoPE-only); 0 = fall back to n_ctas_cap (the amax job's SMs x 8, which launched a 1.6-wave grid-stride tail).
    n_rec_cap: int = 0


def compile_mxfp8_bwd_prologue(
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
    stages: int = DEFAULT_STAGES,
    threads_per_cta: int = PROLOGUE_THREADS,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
    const_slot0: int = 0,
    n_consts: int = 0,
) -> Mxfp8BwdPrologueRecipe:
    """Build from SHAPES ALONE -- no allocation, no launch.  The rebuild's MX arm tiles ONE head x 32 tokens for any head count
    (``validate_mx_shape``: d = 256, whole warps, the RoPE rules); ``apply_norm=False`` traces the RoPE-only rebuild; ``d_model % d ==
    0`` (the dY view's head count); the v8 job is the standalone rowwise quantize at this block size (``quantize_mxfp8.validate_shape``).
    Needs the fp8 ``cvt`` (sm_89+) and ``cvt.rp.satfinite.ue8m0x2`` (sm_100+; declined by name) and TMA (sm_90+).  ``const_slot0`` /
    ``n_consts`` are the init job's constant slot range, exactly ``compile_init_scalars``'s: the VALUES are runtime kernel arguments
    (``run_mxfp8_bwd_prologue(consts=)``).  Every knob is in the cache key."""
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
    validate_shape(d, threads_per_cta)  # the amax job (the quantize layout)
    validate_mx_shape(d, rope_dim, threads_per_cta)  # the rebuild's MX arm
    _validate_mx_quant_shape(d, threads_per_cta, "row")  # the v8 job
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"the MXFP8 backward prologue serves bf16/f16 slabs only, got {dtype}")
    if threads_per_cta % 32 != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of 32 (the amax fold's warp butterfly)")
    if not apply_norm and rope_dim == 0:
        raise ValueError("apply_norm=False with rope_dim=0 is an identity copy of Q/K; there is nothing to rebuild")
    if h_q < 1 or h_kv < 1:
        raise ValueError(f"h_q and h_kv must be positive, got {h_q} / {h_kv}")
    require_fp8_cvt("compile_mxfp8_bwd_prologue")
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
        int(stages),
        int(threads_per_cta),
        int(rows_per_group),
        bool(const_head_count),
        device,
    )
    if key not in prologue_cache:
        tok = cute.sym_int()
        slots = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (int(n_slots),), stride_order=(0,), assumed_align=4)
        dy = fake_rowmajor_dynamic_token_stride(dtype, tok, h_dy, d)
        q4, k4 = _fake_bshd(dtype, h_q, d), _fake_bshd(dtype, h_kv, d)
        weights = [_fake_tma(dtype, (d,), (0,)) for _ in range(2)] if apply_norm else [None, None]
        tables = [_fake_tma(dtype, (tok, rope_dim if rope_dim else 1), (1, 0)) for _ in range(2)]
        v = fake_rowmajor_dynamic_token_stride(dtype, tok, h_kv, d)
        prologue_cache[key] = cute.compile(
            mxfp8_bwd_prologue_launch,
            slots,
            _fake_slot(),  # scale_dp (bound, never read)
            _fake_slot(),  # descale_dp_out (bound, never written)
            *[cutlass.Float32(0.0) for _ in range(MAX_INIT_CONSTS)],  # the constants: runtime fp32 arguments (the zeros pin the TYPE only)
            dy,
            _fake_partials(),
            q4,
            k4,
            *weights,
            *tables,
            _fake_e4m3_thd(tok, h_q, d),  # q8
            _fake_u8_flat(),  # sf_q
            _fake_e4m3_thd(tok, h_q, d),  # q_T8
            _fake_u8_flat(),  # sf_q_T
            _fake_e4m3_thd(tok, h_kv, d),  # k8
            _fake_u8_flat(),  # sf_k
            _fake_e4m3_thd(tok, h_kv, d),  # k_T8
            _fake_u8_flat(),  # sf_k_T
            v,
            _fake_e4m3_thd(tok, h_kv, d),  # v8
            _fake_u8_flat(),  # sf_v
            cutlass.Int32(0),  # n_dy_rows   ) runtime; the values pin the TYPE only
            cutlass.Int32(0),  # n_dy_groups )
            cutlass.Int32(1),  # n_amax      )
            cutlass.Int32(0),  # seq_len     )
            cutlass.Int32(0),  # batch       )
            cutlass.Int32(0),  # n_blk       )
            cutlass.Int32(0),  # n_sft       )
            cutlass.Int32(0),  # n_tiles_mx  )
            cutlass.Int32(1),  # n_rec       )
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
            int(stages),
            int(threads_per_cta),
            int(rows_per_group),
            bool(const_head_count),
            _fake_stream(),
            options="--enable-tvm-ffi",
        )
    return Mxfp8BwdPrologueRecipe(
        compiled=prologue_cache[key],
        dtype=dtype,
        h_q=int(h_q),
        h_kv=int(h_kv),
        h_dy=int(h_dy),
        d=int(d),
        rope_dim=int(rope_dim),
        eps=float(eps),
        stages=int(stages),
        apply_norm=bool(apply_norm),
        n_slots=int(n_slots),
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        n_ctas_cap=int(multiprocessor_count(device)) * AMAX_CTAS_PER_SM,
        threads=int(threads_per_cta),
        const_slot0=int(const_slot0),
        n_consts=int(n_consts),
        n_rec_cap=int(multiprocessor_count(device)) * mx_rebuild_ctas_per_sm(apply_norm),
    )


def prologue_grid(r: Mxfp8BwdPrologueRecipe, batch: int, seq_len: int) -> tuple:
    """``(n_amax, n_rec, n_v8)`` for ``batch x seq_len`` tokens -- the three job widths of the launch (its grid is ``1 + their sum``):
    the amax job's partial count ``min(ceil(T * h_dy / rows_per_cta), n_ctas_cap)``, the rebuild's persistent CTAs ``min(n_tiles_mx,
    n_rec_cap)`` -- its cap is the MX arm's RESIDENCY (``SMs x 5 / 6``, norm / RoPE-only; ``n_rec_cap == 0`` falls back to
    ``n_ctas_cap``) so the grid-stride walk runs ONE wave -- (``mx_tile_counts``: ``B * ceil128(S)/32 * (h_q + h_kv)`` token tiles),
    the v8 job's blocks ``B * h_kv * ceil(S / 128)`` (one per 128-row SF unit of the V band)."""
    batch, seq_len = int(batch), int(seq_len)
    t = batch * seq_len
    dy_groups = (t * r.h_dy + r.rows_per_cta - 1) // r.rows_per_cta
    n_amax = max(1, min(dy_groups, r.n_ctas_cap))
    _n_blk, n_sft, n_tiles_mx = mx_tile_counts(batch, seq_len, r.h_q, r.h_kv)
    n_rec = max(1, min(n_tiles_mx, int(getattr(r, "n_rec_cap", 0)) or r.n_ctas_cap))
    n_v8 = batch * r.h_kv * n_sft
    return n_amax, n_rec, n_v8


def _mx_payload_bytes(elems: int) -> int:
    """One MXFP8 quantization's output bytes over ``elems`` elements: the e4m3 payload plus one E8M0 byte per ``SF_BLOCK`` elements."""
    return elems + elems // SF_BLOCK


def prologue_moved_bytes(r: Mxfp8BwdPrologueRecipe, batch: int, seq_len: int, *, rope_l1_miss: bool = False) -> int:
    """HBM traffic of ONE prologue launch for ``batch x seq_len`` tokens -- the SUM of its jobs' own byte models, nothing more: the init's
    slot writes (``n_slots x 4``), the dY amax's read (``quantize.amax_moved_bytes`` over the ``[T, h_dy, D]`` view), the MX rebuild's
    read of the two pre-norm bands (bf16) and of the cos / sin rows (``T x rope_dim x 4`` ONCE per token -- the heads-inner tile walk
    keeps a 32-token range's rows L1-hot across its heads; ``rope_l1_miss=True`` counts one read per (token, head), the L1-miss UPPER
    bound) plus its FOUR quantized outputs (``q8 / q_T8 / k8 / k_T8``: two payload + SF pairs per Q / K element), and the v8 job
    (``quantize_mxfp8.moved_bytes`` over the V band).  No bf16 ``recompute`` / ``recompute_k`` byte is moved: the rebuild quantizes out of
    registers.  The byte model behind the launch's GB/s in the perf tooling; ``prologue_standalone_set_bytes`` is the eight launches'."""
    t = int(batch) * int(seq_len)
    hd, hkd = r.h_q * r.d, r.h_kv * r.d
    heads = (r.h_q + r.h_kv) if rope_l1_miss else 1
    rebuild = t * (hd + hkd) * 2 + t * r.rope_dim * 4 * heads + 2 * _mx_payload_bytes(t * (hd + hkd))
    return r.n_slots * 4 + amax_moved_bytes(t, r.h_dy, r.d) + rebuild + moved_bytes(t, r.h_kv, r.d)


def prologue_standalone_set_bytes(r: Mxfp8BwdPrologueRecipe, batch: int, seq_len: int) -> int:
    """HBM traffic of the EIGHT standalone launches the prologue replaces (init, dY amax, the bf16 TMA rebuild, the four Q / K block
    quantizes, v8), each at its own byte model: the rebuild reads the pre-norm bands and cos / sin and WRITES the bf16 ``recompute`` /
    ``recompute_k`` (``T x (HD + HKD) x 4`` + cos / sin), and each of the four quantizes READS its bf16 buffer again
    (``quantize_mxfp8.moved_bytes``: 2 B in, 1 + 1/32 B out).  ``prologue_standalone_set_bytes - prologue_moved_bytes`` is exactly the bf16
    round trip the fusion deletes: ``T x (HD + HKD) x 6`` (one write, two reads of every Q / K element)."""
    t = int(batch) * int(seq_len)
    hd, hkd = r.h_q * r.d, r.h_kv * r.d
    rebuild = t * (hd + hkd) * 4 + t * r.rope_dim * 4
    quantizes = 2 * moved_bytes(t, r.h_q, r.d) + 2 * moved_bytes(t, r.h_kv, r.d)
    return r.n_slots * 4 + amax_moved_bytes(t, r.h_dy, r.d) + rebuild + quantizes + moved_bytes(t, r.h_kv, r.d)


def run_mxfp8_bwd_prologue(
    r: Mxfp8BwdPrologueRecipe,
    *,
    slots: torch.Tensor,
    dy: torch.Tensor,
    partials: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    w_q: Optional[torch.Tensor],
    w_k: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    q8: torch.Tensor,
    sf_q: torch.Tensor,
    q_T8: torch.Tensor,
    sf_q_T: torch.Tensor,
    k8: torch.Tensor,
    sf_k: torch.Tensor,
    k_T8: torch.Tensor,
    sf_k_T: torch.Tensor,
    v: torch.Tensor,
    v8: torch.Tensor,
    sf_v: torch.Tensor,
    batch: int,
    seq_len: int,
    stream,
    consts: tuple = (),
) -> int:
    """Launch.  ``slots`` the contiguous fp32 ``[n_slots]`` scalar block (zeroed, then the ``consts`` stored at ``slots[const_slot0:]``);
    ``dy`` ``[T, d_model / D, D]`` bf16; ``partials`` a contiguous fp32 array of at least ``prologue_grid(...)[0]`` words (every one of
    its first ``n_amax`` words is OVERWRITTEN); ``q`` / ``k`` / ``v`` the slab's PRE-norm Q / K bands and its V band ``[T, H, D]`` (token
    stride ``N``); the norm weights both ``None`` for a RoPE-only recipe; ``cos`` / ``sin`` ``[T, rope_dim]``; the five e4m3 payloads
    compact ``[T, H, D]``; the five SF blobs ``B*H*ceil(S/128)*4*D`` bytes each; ``T == batch * seq_len``; ``consts`` the artifact's
    ``n_consts`` plan-time constants as finite Python numbers in slot order.  Returns ``n_amax`` (the partials written).  Host checks
    only, no allocation."""
    batch, seq_len = int(batch), int(seq_len)
    t = int(q.shape[0])
    if batch < 1 or seq_len < 1 or batch * seq_len != t:
        raise ValueError(f"T must equal batch*seq_len: q has T={t}, got batch={batch} seq_len={seq_len}")
    check_scalar_slot("slots", slots, numel=r.n_slots)
    n_consts = int(r.n_consts)
    if len(consts) != n_consts:
        raise ValueError(f"this artifact's init job stores n_consts={n_consts} plan-time constants; got {len(consts)} consts (the ABI is fixed per artifact)")
    const_values = []
    for i, cv in enumerate(consts):
        if isinstance(cv, (bool, torch.Tensor)) or not isinstance(cv, numbers.Real) or not math.isfinite(float(cv)):
            raise ValueError(
                f"consts[{i}] must be a finite Python number (a plan-time constant handed to the kernel as an argument -- never a device tensor), got {cv!r}"
            )
        const_values.append(float(cv))
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
    for name, ten, h in (("q8", q8, r.h_q), ("q_T8", q_T8, r.h_q), ("k8", k8, r.h_kv), ("k_T8", k_T8, r.h_kv), ("v8", v8, r.h_kv)):
        check_e4m3_out(name, ten, t, h, r.d)
    dev = q.device
    for name, ten, h in (("sf_q", sf_q, r.h_q), ("sf_q_T", sf_q_T, r.h_q), ("sf_k", sf_k, r.h_kv), ("sf_k_T", sf_k_T, r.h_kv), ("sf_v", sf_v, r.h_kv)):
        check_mx_sf_blob(name, ten, sf_bytes(batch, h, seq_len, r.d), dev)
    n_amax, n_rec, n_v8 = prologue_grid(r, batch, seq_len)
    check_partials("partials", partials, n_amax)
    for name, ten in (
        ("slots", slots),
        ("dy", dy),
        ("partials", partials),
        ("k", k),
        ("w_q_norm", w_q),
        ("w_k_norm", w_k),
        ("cos", cos),
        ("sin", sin),
        ("q8", q8),
        ("q_T8", q_T8),
        ("k8", k8),
        ("k_T8", k_T8),
        ("v", v),
        ("v8", v8),
    ):
        if ten is not None and (not ten.is_cuda or ten.device != dev):
            raise ValueError(f"{name} must be on {dev} with q (the kernel reads every operand through a device pointer), got {ten.device}")
    n_dy_rows = t * r.h_dy
    n_dy_groups = (n_dy_rows + r.rows_per_cta - 1) // r.rows_per_cta
    n_blk, n_sft, n_tiles_mx = mx_tile_counts(batch, seq_len, r.h_q, r.h_kv)
    const_args = [cutlass.Float32(cv) for cv in const_values] + [cutlass.Float32(0.0) for _ in range(MAX_INIT_CONSTS - n_consts)]
    dummy = _slot_view(slots[:1])  # the init body's dP scalar ports: bound, never read, never written (descale_dp=False)
    r.compiled(
        slots,
        dummy,
        dummy,
        *const_args,
        dy,
        partials,
        _bshd_view(q, batch, seq_len),
        _bshd_view(k, batch, seq_len),
        w_q,
        w_k,
        cos,
        sin,
        q8,
        sf_q.view(-1),
        q_T8,
        sf_q_T.view(-1),
        k8,
        sf_k.view(-1),
        k_T8,
        sf_k_T.view(-1),
        v,
        v8,
        sf_v.view(-1),
        cutlass.Int32(n_dy_rows),
        cutlass.Int32(n_dy_groups),
        cutlass.Int32(n_amax),
        cutlass.Int32(seq_len),
        cutlass.Int32(batch),
        cutlass.Int32(n_blk),
        cutlass.Int32(n_sft),
        cutlass.Int32(n_tiles_mx),
        cutlass.Int32(n_rec),
        cutlass.Int32(1 + n_amax + n_rec + n_v8),
        cutlass.Float32(r.eps),
        cuda.CUstream(int(stream)),
    )
    return n_amax


# ---------------------------------------------------------------------------
# EPILOGUE: dW_norm reduce | dqkvg dual-axis canonical cast
# ---------------------------------------------------------------------------


@cute.kernel
def frost_mxfp8_bwd_epilogue(
    # -- the dW_norm reduce (blocks [0, n_red)) --
    mPq: Optional[cute.Tensor],  # [n_q, D] fp32 partials; None = no reduce (n_red == 0)
    mPk: Optional[cute.Tensor],  # [n_k, D]
    mDWq: Optional[cute.Tensor],  # [D] fp32 OUT
    mDWk: Optional[cute.Tensor],  # [D]
    # -- the dqkvg dual-axis cast (blocks [n_red, n_red + n_cast)) --
    mSrc: cute.Tensor,  # [T, N / D, D] bf16: the dqkvg slab viewed in the quantize layout
    mDst: Optional[cute.Tensor],  # [T, N / D, D] e4m3 OUT: the rowwise canonical payload (want_row)
    mSf: Optional[cute.Tensor],  # the canonical blob over (T, N) (want_row)
    mDstT: Optional[cute.Tensor],  # [N, T] e4m3 OUT: the transposed canonical payload (want_col)
    mSfT: Optional[cute.Tensor],  # the canonical blob over (N, T) (want_col)
    # -- runtime geometry --
    n_q: cutlass.Int32,
    n_k: cutlass.Int32,
    n_red: cutlass.Int32,  # epilogue_reduce_blocks(d, threads_per_cta): the 2 d columns at cols per block (d at 256 threads), or 0
    n_tokens: cutlass.Int32,  # T (the batch folds into the rows under the canonical layout)
    n_t_tiles: cutlass.Int32,  # ceil(T / 128)
    n_c_atoms: cutlass.Int32,  # atoms per 128-row band of the (T, N) blob
    n_c_atoms_t: cutlass.Int32,  # atoms per 128-row band of the (N, T) blob
    # -- compile-time facts --
    d: cutlass.Constexpr[int],
    unroll: cutlass.Constexpr[int],
    h_ct: cutlass.Constexpr[int],  # N / D
    threads_per_cta: cutlass.Constexpr[int],
    want_row: cutlass.Constexpr[bool],
    want_col: cutlass.Constexpr[bool],
) -> None:
    """Block-range dispatch over the two jobs (module docstring)."""
    want_dw = cutlass.const_expr(mPq is not None)
    cols = cutlass.const_expr(epilogue_reduce_cols(threads_per_cta))  # columns per reduce block: 2 at 256 threads (the grid is derived from the same)
    # the reduce arm's combine array (lanes x cols words); the dual body's staging tile, scale array and two SF tiles
    sPart = cutlass.Array(cutlass.Float32, threads_per_cta, alignment=16, space=cutlass.AddressSpace.smem) if cutlass.const_expr(want_dw) else None
    sStage = cutlass.Array(cutlass.Int32, SF_BLOCK * d // 2, alignment=16, space=cutlass.AddressSpace.smem)
    sScale = cutlass.Array(cutlass.Float32, d, alignment=16, space=cutlass.AddressSpace.smem)
    sSFrow = cutlass.Array(cutlass.Uint8, sf_tile_bytes(d), alignment=16, space=cutlass.AddressSpace.smem)
    sSFcol = cutlass.Array(cutlass.Uint8, sf_tile_bytes(d), alignment=16, space=cutlass.AddressSpace.smem)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    cta = cutlass.Int32(cute.arch.block_idx()[0])
    if cta < n_red:
        if cutlass.const_expr(want_dw):
            # thread (l, j) = (tidx % lanes, tidx // lanes) sums column cols * blk + j; the 2 d columns are d Q columns then d K columns
            l_row = tidx % cutlass.Int32(REDUCE_LANES)
            j = tidx // cutlass.Int32(REDUCE_LANES)
            col_all = cta * cutlass.Int32(cols) + j
            is_q = col_all < cutlass.Int32(d)
            col = col_all if is_q else col_all - cutlass.Int32(d)
            p_base = mPq.iterator.toint() if is_q else mPk.iterator.toint()
            o_base = mDWq.iterator.toint() if is_q else mDWk.iterator.toint()
            n = n_q if is_q else n_k
            dw_reduce_column(p_base, o_base, n, col, l_row, j, d, cols, REDUCE_LANES, unroll, sPart)
    else:
        # one block per (h, 128-token unit) of the [T, N] slab = the standalone canonical kernel's (blockIdx.x, blockIdx.y)
        u = cta - n_red
        quantize_mxfp8_dual_body(
            mSrc,
            mDst,
            mSf,
            mDstT,
            mSfT,
            n_tokens,
            n_t_tiles,
            cutlass.Int32(0),  # v_sf_groups: the SDPA columnwise plane stride, unused under the canonical layout
            n_c_atoms,
            n_c_atoms_t,
            u // n_t_tiles,
            u % n_t_tiles,
            h_ct,
            d,
            threads_per_cta,
            True,  # sf_gemm: the canonical pair
            True,  # transposed_second: the canonical columnwise blob's one form
            want_row,
            want_col,
            sStage,
            sScale,
            sSFrow,
            sSFcol,
        )


@cute.jit
def mxfp8_bwd_epilogue_launch(
    p_q: Optional[cute.Tensor],
    p_k: Optional[cute.Tensor],
    dw_q: Optional[cute.Tensor],
    dw_k: Optional[cute.Tensor],
    src: cute.Tensor,
    dst: Optional[cute.Tensor],
    sf: Optional[cute.Tensor],
    dst_t: Optional[cute.Tensor],
    sf_t: Optional[cute.Tensor],
    n_q: cutlass.Int32,
    n_k: cutlass.Int32,
    n_red: cutlass.Int32,
    n_tokens: cutlass.Int32,
    n_t_tiles: cutlass.Int32,
    n_c_atoms: cutlass.Int32,
    n_c_atoms_t: cutlass.Int32,
    n_blocks: cutlass.Int32,
    d: cutlass.Constexpr[int],
    unroll: cutlass.Constexpr[int],
    h_ct: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    want_row: cutlass.Constexpr[bool],
    want_col: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_mxfp8_bwd_epilogue(
        p_q,
        p_k,
        dw_q,
        dw_k,
        src,
        dst,
        sf,
        dst_t,
        sf_t,
        n_q,
        n_k,
        n_red,
        n_tokens,
        n_t_tiles,
        n_c_atoms,
        n_c_atoms_t,
        d,
        unroll,
        h_ct,
        threads_per_cta,
        want_row,
        want_col,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream)


epilogue_cache = {}


class Mxfp8BwdEpilogueRecipe(NamedTuple):
    """Build-time facts of the epilogue launch."""

    compiled: object
    dtype: object
    h: int  # N / d: the dqkvg view's head count
    d: int
    want_dw: bool
    want_row: bool  # need_dh: dqkvg8 + the (T, N) blob
    want_col: bool  # need_dw_qkvg: dqkvg_t8 + the (N, T) blob
    threads: int


def compile_mxfp8_bwd_epilogue(
    *,
    dtype,
    n_cols: int,
    d: int,
    want_dw: bool,
    want_row: bool,
    want_col: bool,
    threads_per_cta: int = EPILOGUE_THREADS,
) -> Mxfp8BwdEpilogueRecipe:
    """Build from SHAPES ALONE.  ``n_cols`` is the dqkvg slab's width ``N`` (``N % d == 0``); ``want_dw`` traces the reduce arm
    (``n_red = epilogue_reduce_blocks(d, threads_per_cta)`` blocks of ``threads_per_cta // REDUCE_LANES`` columns at ``lanes =
    REDUCE_LANES``: ``threads_per_cta`` must be a multiple of ``REDUCE_LANES`` so the per-column chain is the standalone reduce's);
    ``want_row`` / ``want_col`` trace the two halves of the dual-axis cast (at least one
    of the three jobs).  Needs the fp8 ``cvt`` (sm_89+) and ``cvt.rp.satfinite.ue8m0x2`` (sm_100+; declined by name)."""
    if n_cols % d != 0:
        raise ValueError(f"n_cols={n_cols} must be a multiple of d_head={d}: dqkvg is quantized through a [T, N / D, D] view")
    for name, flag in (("want_dw", want_dw), ("want_row", want_row), ("want_col", want_col)):
        if not isinstance(flag, bool):
            raise ValueError(f"{name} must be a bool, got {flag!r}")
    if not (want_dw or want_row or want_col):
        raise ValueError("compile_mxfp8_bwd_epilogue: nothing to launch (want_dw, want_row and want_col are all False)")
    validate_dual_shape(d, threads_per_cta)
    if want_dw and (threads_per_cta % REDUCE_LANES != 0 or threads_per_cta < REDUCE_LANES):
        raise ValueError(
            f"the fused dW_norm reduce runs at lanes = REDUCE_LANES = {REDUCE_LANES} residue classes (the standalone reduce's fixed summation order) and "
            f"cols = threads_per_cta / {REDUCE_LANES} columns per block: threads_per_cta={threads_per_cta} is not a multiple of it"
        )
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"the MXFP8 backward epilogue serves bf16/f16 slabs only, got {dtype}")
    require_fp8_cvt("compile_mxfp8_bwd_epilogue")
    device = current_device()
    h = n_cols // d
    key = (str(dtype), int(h), int(d), bool(want_dw), bool(want_row), bool(want_col), int(threads_per_cta), device)
    if key not in epilogue_cache:
        tok = cute.sym_int()
        planes = [_fake_tma(torch.float32, (cute.sym_int(), d), (1, 0)) for _ in range(2)] if want_dw else [None, None]
        outs32 = [_fake_tma(torch.float32, (d,), (0,)) for _ in range(2)] if want_dw else [None, None]
        src = fake_rowmajor_dynamic_token_stride(dtype, tok, h, d)
        dst = (
            cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(tok, h, d), stride=(cute.sym_int(), d, 1), assumed_align=16) if want_row else None
        )
        sf = _fake_u8_flat() if want_row else None
        dst_t = (
            cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(h * d, tok), stride=(cute.sym_int(), 1), assumed_align=16) if want_col else None
        )
        sf_t = _fake_u8_flat() if want_col else None
        epilogue_cache[key] = cute.compile(
            mxfp8_bwd_epilogue_launch,
            *planes,
            *outs32,
            src,
            dst,
            sf,
            dst_t,
            sf_t,
            cutlass.Int32(0),  # n_q         ) runtime; the values pin the TYPE only
            cutlass.Int32(0),  # n_k         )
            cutlass.Int32(0),  # n_red       )
            cutlass.Int32(0),  # n_tokens    )
            cutlass.Int32(1),  # n_t_tiles   )
            cutlass.Int32(0),  # n_c_atoms   )
            cutlass.Int32(0),  # n_c_atoms_t )
            cutlass.Int32(1),  # n_blocks    )
            int(d),
            REDUCE_UNROLL,
            int(h),
            int(threads_per_cta),
            bool(want_row),
            bool(want_col),
            _fake_stream(),
            options="--enable-tvm-ffi",
        )
    return Mxfp8BwdEpilogueRecipe(
        compiled=epilogue_cache[key],
        dtype=dtype,
        h=int(h),
        d=int(d),
        want_dw=bool(want_dw),
        want_row=bool(want_row),
        want_col=bool(want_col),
        threads=int(threads_per_cta),
    )


def epilogue_grid(r: Mxfp8BwdEpilogueRecipe, t: int) -> tuple:
    """``(n_red, n_cast)`` for ``t`` tokens -- the two job widths of the launch (its grid is their sum): the reduce blocks
    ``epilogue_reduce_blocks(d, threads)`` under ``want_dw`` (``ceil(2 d / cols)`` over the ``2 d`` columns at ``cols = threads //
    REDUCE_LANES`` per block -- ``d`` at the 256-thread default; 0 otherwise) and the cast blocks ``(N / D) * ceil(T / 128)`` (one per
    (h, 128-token unit); 0 when neither half is traced)."""
    t = int(t)
    n_cast = r.h * n_sf_tiles(t) if (r.want_row or r.want_col) else 0
    return (epilogue_reduce_blocks(r.d, r.threads) if r.want_dw else 0), n_cast


def epilogue_moved_bytes(r: Mxfp8BwdEpilogueRecipe, t: int, n_plane_rows: int) -> int:
    """HBM traffic of ONE epilogue launch for ``t`` tokens -- the SUM of its jobs' own byte models: the dW_norm reduce's read of the two
    fp32 partial planes (``n_plane_rows`` = the Q plane's rows + the K plane's rows, ``x d x 4``; 0 bytes without ``want_dw``) and the
    dual-axis cast's ONE bf16 read of the ``[T, N]`` dqkvg slab plus one payload + SF pair per traced half (``want_row``: ``dqkvg8`` + its
    blob; ``want_col``: ``dqkvg_t8`` + its blob) -- ``quantize_mxfp8.moved_bytes_dual`` when both halves are traced.  The byte model behind
    the launch's GB/s in the perf tooling; ``epilogue_standalone_set_bytes`` is the three launches'."""
    t = int(t)
    elems = t * r.h * r.d
    halves = int(bool(r.want_row)) + int(bool(r.want_col))
    cast = (elems * 2 + halves * _mx_payload_bytes(elems)) if halves else 0
    return (int(n_plane_rows) * r.d * 4 if r.want_dw else 0) + cast


def epilogue_standalone_set_bytes(r: Mxfp8BwdEpilogueRecipe, t: int, n_plane_rows: int) -> int:
    """HBM traffic of the standalone launches the epilogue replaces: the dW_norm reduce (the same planes) and one standalone quantize per
    traced half, each READING the bf16 slab again (``quantize_mxfp8.moved_bytes``) -- so ``epilogue_standalone_set_bytes -
    epilogue_moved_bytes`` is one bf16 read of the slab (``T x N x 2``) when both halves are traced, 0 with one half."""
    t = int(t)
    halves = int(bool(r.want_row)) + int(bool(r.want_col))
    return (int(n_plane_rows) * r.d * 4 if r.want_dw else 0) + halves * moved_bytes(t, r.h, r.d)


def run_mxfp8_bwd_epilogue(
    r: Mxfp8BwdEpilogueRecipe,
    *,
    plane_q: Optional[torch.Tensor],
    plane_k: Optional[torch.Tensor],
    dw_q: Optional[torch.Tensor],
    dw_k: Optional[torch.Tensor],
    src: torch.Tensor,
    dst: Optional[torch.Tensor],
    sf: Optional[torch.Tensor],
    dst_t: Optional[torch.Tensor],
    sf_t: Optional[torch.Tensor],
    stream,
) -> None:
    """Launch.  Under ``want_dw`` the partial planes (``[n_q, D]`` / ``[n_k, D]`` fp32 contiguous, EXACTLY the rows the norm backward
    wrote) and the two fp32 ``[D]`` outputs are REQUIRED, refused otherwise; ``src`` the ``[T, N / D, D]`` bf16 view of the dqkvg slab;
    under ``want_row`` ``dst`` (its compact e4m3 twin) and ``sf`` (``sf_blob_bytes(T, N)``) are REQUIRED, under ``want_col`` ``dst_t``
    (the contiguous e4m3 ``[N, T]`` matrix; ``T % 32 == 0``) and ``sf_t`` (``sf_blob_bytes(N, T)``) -- each pair refused when its half
    is not traced (Rule 1).  Host checks only."""
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
    n_c_atoms = n_c_atoms_t = 0
    if r.want_row:
        if dst is None or sf is None:
            raise ValueError("this artifact was compiled WITH the rowwise half (want_row=True): dst and sf must be bound (Rule 1: no silent fallback)")
    elif dst is not None or sf is not None:
        raise ValueError("this artifact was compiled WITHOUT the rowwise half (want_row=False); passing dst / sf would silently ignore them (Rule 1)")
    if r.want_col:
        if dst_t is None or sf_t is None:
            raise ValueError("this artifact was compiled WITH the transposed half (want_col=True): dst_t and sf_t must be bound (Rule 1: no silent fallback)")
    elif dst_t is not None or sf_t is not None:
        raise ValueError("this artifact was compiled WITHOUT the transposed half (want_col=False); passing dst_t / sf_t would silently ignore them (Rule 1)")
    if r.want_row or r.want_col:
        # the dual arm's own operand checks over the halves that are traced: a folded-out half passes None and is NOT checked -- no
        # stand-in tensor is allocated on the execute path (a full-size stand-in was ~143 MiB per folded half at the 397B geometry, S = 8K)
        probe = QuantizeMxfp8DualRecipe(dtype_in=r.dtype, h=r.h, d=r.d, threads_per_cta=r.threads, sf_layout=SF_LAYOUT_GEMM, transposed_second=True)
        _, _, n_c_atoms, n_c_atoms_t = check_dual_operands(probe, src, dst, sf, dst_t, sf_t, batch=1, seq_len=t, want_row=r.want_row, want_col=r.want_col)
    dev = src.device
    for name, ten in (("dst", dst), ("sf", sf), ("dst_t", dst_t), ("sf_t", sf_t), ("plane_q", plane_q), ("plane_k", plane_k), ("dw_q", dw_q), ("dw_k", dw_k)):
        if ten is not None and (not ten.is_cuda or ten.device != dev):
            raise ValueError(f"{name} must be on {dev} with src (the kernel reads every operand through a device pointer), got {ten.device}")
    n_red, n_cast = epilogue_grid(r, t)
    r.compiled(
        plane_q,
        plane_k,
        dw_q,
        dw_k,
        src,
        dst,
        sf.view(-1) if sf is not None else None,
        dst_t,
        sf_t.view(-1) if sf_t is not None else None,
        cutlass.Int32(n_q),
        cutlass.Int32(n_k),
        cutlass.Int32(n_red),
        cutlass.Int32(t),
        cutlass.Int32(n_sf_tiles(t)),
        cutlass.Int32(n_c_atoms),
        cutlass.Int32(n_c_atoms_t),
        cutlass.Int32(n_red + n_cast),
        cuda.CUstream(int(stream)),
    )


frost_mxfp8_bwd_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_mxfp8_bwd_epilogue.set_name_prefix("cudnn", remove_cutlass_symbol=True)
