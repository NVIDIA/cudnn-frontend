# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared tile, reduction, and softmax helpers for the SM90 D512 kernel.

``prefill_d512_f16.py`` owns launch state and scheduling; these helpers are
stateless. For local thread ``tid``, C slot ``i`` uses row
``16*(tid//32) + (tid%32)//4 + 8*((i//2)%2)`` and column
``2*(tid%4) + i%2 + 8*(i//4)``. Helpers keep their units, fragment order, and
caller-owned synchronization at the boundary where those details matter.
"""

import math
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b
from cudnn.frost.tile_dsl.wgmma import wgmma_m64nNk16_f32, wgmma_smem_desc
from cudnn.sdpa.fwd.config_sm90 import SCALE_POSITIVE, SCALE_ZERO

# The SCHED_LPT_L2 budget; api_dsl and heuristics keep their own copies.
SCHED_L2_BUDGET_BYTES = 50 * 1024 * 1024

# FP64 FLT_MAX used by the positive-scale overflow guards.
FLT_MAX_F64 = 3.4028234663852886e38
# Python-double conversion factors for natural and base-2 logarithms.
LOG2_E = math.log2(math.e)
LN2 = math.log(2.0)


class RolePipelineStates(NamedTuple):
    """The TMA, P-ring, and alpha states carried by one compute role."""

    # Destructure and rebuild these fields in dynamic loops: the DSL tracks
    # loop-carried values by local name, not by attribute writes.

    tma: object
    p: object
    alpha: object


@cute.jit
def wgmma_descriptor(region, mn_major: cutlass.Constexpr = False):
    """Build a descriptor for a 64-column SW128 region.

    K-major Q/K/P use LBO=16 B; MN-major V uses LBO=8192 B. Both use
    SBO=1024 B. Hot-loop offsets are 16-byte start-field units and must not
    carry into another descriptor field.
    """
    leading = 8192 if cutlass.const_expr(mn_major) else 16
    return wgmma_smem_desc(region, leading_byte_offset=leading, stride_byte_offset=1024)


@cute.jit
def owned_row(tid, row_half: cutlass.Constexpr):
    """Return the m64 C/RS row owned by ``tid`` in row half 0 or 1."""
    return 16 * (tid // cute.arch.WARP_SIZE) + (tid % cute.arch.WARP_SIZE) // 4 + 8 * row_half


@cute.jit
def mma_qk(mma_params, scores, *, ab_dtype: cutlass.Constexpr):
    """Compute one 64x64 QK tile with 32 K-major k16 SS steps.

    ``mma_params`` supplies 16-byte-unit Q/K descriptors; wait(0) completes
    the 32-slot C fragment before it is stored in ``scores``.
    """
    q_desc, k_desc = mma_params.q_desc, mma_params.k_desc
    c = cutlass.Vector.from_elements((cutlass.Float32(0.0),) * 32, cutlass.Float32)
    # One group keeps the k steps in flight; wait(0) completes C before the store.
    prims.wgmma_fence_aligned()
    for k in cutlass.range_constexpr(32):
        offset = (k // 4) * 512 + (k % 4) * 2
        c = wgmma_m64nNk16_f32(c, q_desc + offset, k_desc + offset, ab_dtype, accumulate=k != 0)
    prims.wgmma_commit_group_sync_aligned()
    prims.wgmma_wait_group_sync_aligned(0)
    scores.store(c)


@cute.jit
def mma_pv(mma_params, p, is_first_kv_tile: cutlass.Constexpr, *, ab_dtype: cutlass.Constexpr):
    """Compute one 64x256 O half with four k16 PV steps.

    ``p`` is 16 RS words for WG1 or a K-major P descriptor for WG2. The
    fence/commit/wait group completes C before the output store.
    """
    output, v_desc = mma_params.o_regs, mma_params.v_desc
    c = output.load(0, output.shape[0])
    prims.wgmma_fence_aligned()
    for k in cutlass.range_constexpr(4):
        a = p[4 * k : 4 * k + 4] if cutlass.const_expr(isinstance(p, cutlass.Vector)) else p + 2 * k
        c = wgmma_m64nNk16_f32(c, a, v_desc + 128 * k, ab_dtype, transpose_b=True, accumulate=not is_first_kv_tile or k != 0)
    prims.wgmma_commit_group_sync_aligned()
    prims.wgmma_wait_group_sync_aligned(0)
    output.store(c)


@cute.jit
def score_prefix_mask(bound, column_base, inclusive: cutlass.Constexpr = False):
    """Bitmap for this lane's 16 scores below a column bound (optionally inclusive)."""
    # Interleaved columns are base + [0,1,8,9,...,56,57]; clamp to 0..16 bits.
    offset = 1 if cutlass.const_expr(inclusive) else 0
    relative = cute.math.max(bound, column_base - offset) - column_base
    limit = cutlass.Uint32(cute.math.min(relative, 64 - offset) + offset)
    count = (limit >> 3) * 2 + cute.math.min(limit & 7, cutlass.Uint32(2))
    return (cutlass.Uint32(1) << count) - cutlass.Uint32(1)


@cute.jit
def quad_reduce(value: cutlass.Float32, op: cutlass.Constexpr):
    """Reduce each lane's complementary columns across its four-lane quad."""
    assert op in ("max", "min", "sum")
    # Extrema use offsets 2,1; sum keeps SM90's FP32 tree at offsets 1,2.
    for step in cutlass.range_constexpr(2):
        delta = (1 << step) if cutlass.const_expr(op == "sum") else (2 >> step)
        peer = prims.shfl_sync(thread_mask=0xFFFFFFFF, val=value, offset=delta, mask_and_clamp=0x1F, kind=prims.Shfl.BFLY)
        if cutlass.const_expr(op == "max"):
            value = cute.arch.fmax(value, peer)
        elif cutlass.const_expr(op == "min"):
            value = cute.arch.fmin(value, peer)
        else:
            value = value + peer
    return value


@cute.jit
def rescale_output(output, factors: cutlass.Vector):
    """Scale completed O by the factors for row halves 0 and 1."""
    # C slots repeat [row0, row0, row1, row1].
    scale = cutlass.Vector.from_elements((factors[0], factors[0], factors[1], factors[1]) * (output.shape[0] // 4), cutlass.Float32)
    output.store(output.load(0, output.shape[0]) * scale)


@cute.jit
def store_fragment(words: cutlass.Vector, region: cutlass.Array, tid):
    """Store one m64 fragment to SW128 slabs with ``stmatrix.x4``.

    All 128 lanes participate; four packed words per lane cover one 16x16 warp
    tile. The caller owns conversion, readiness, and proxy publication.
    """
    for block in cutlass.range_constexpr(words.shape[0] // 4):
        lane = tid % cute.arch.WARP_SIZE
        row = 16 * (tid // cute.arch.WARP_SIZE) + lane % 8 + 8 * ((lane // 8) % 2)
        column = 16 * block + 8 * (lane // 16)
        offset = (column // 64) * 4096 + row * 64 + swizzle_xor_128b(row, column % 64)
        prims.stmatrix(region.data_ptr() + offset, words[4 * block : 4 * block + 4], prims.MMALayout.ROW)


@cute.jit
def normalize_row(anchor, scale, scale_log2, total, scale_mode: cutlass.Constexpr):
    """Return inverse row sum and LSE without sink mass; empty rows return ``(0, -inf)``.

    Positive scale keeps the FP32 product when safe and widens the fallback.
    """
    inv, value = cutlass.Float32(0.0), cutlass.Float32(-cutlass.Float32.inf)
    if total > cutlass.Float32(0.0):
        inv = cute.math.rcp(total, approx=True, ftz=True)
        narrow = cutlass.Boolean(False)
        if cutlass.const_expr(scale_mode == SCALE_POSITIVE):
            product = cutlass.Float64(anchor) * cutlass.Float64(scale_log2)
            narrow = (cute.math.abs(product) <= cutlass.Float64(FLT_MAX_F64)) & (cutlass.Float64(scale_log2) <= cutlass.Float64(FLT_MAX_F64))
        if cutlass.const_expr(scale_mode == SCALE_ZERO):
            value = cute.math.log2(total, fastmath=True) * cutlass.Float32(LN2)
        elif narrow:
            value = (anchor * scale_log2 + cute.math.log2(total, fastmath=True)) * cutlass.Float32(LN2)
        else:
            value = cutlass.Float32(
                cutlass.Float64(anchor) * cutlass.Float64(scale) + cutlass.Float64(cute.math.log2(total, fastmath=True)) * cutlass.Float64(LN2)
            )
    return inv, value


@cute.jit
def normalize_sink_row(anchor, scale, total, row_sink, scale_mode: cutlass.Constexpr):
    """Return inverse mass and LSE after adding a natural-log sink logit.

    Empty rows keep inverse mass zero and return the sink value.
    """
    inv, value = cutlass.Float32(0.0), row_sink
    if total > cutlass.Float32(0.0):
        peak = cutlass.Float64(0.0) if cutlass.const_expr(scale_mode == SCALE_ZERO) else cutlass.Float64(anchor) * cutlass.Float64(scale)
        logit = cutlass.Float64(row_sink)
        top, shrink, extra = peak, cutlass.Float32(1.0), cutlass.Float32(1.0)
        if peak >= logit:
            exponent = cutlass.Float32((logit - peak) * cutlass.Float64(LOG2_E))
            extra = cute.math.exp2(cute.arch.fmax(exponent, cutlass.Float32(-1024.0)), fastmath=True)
        else:
            exponent = cutlass.Float32((peak - logit) * cutlass.Float64(LOG2_E))
            shrink = cute.math.exp2(cute.arch.fmax(exponent, cutlass.Float32(-1024.0)), fastmath=True)
            top = logit
        mass = total * shrink + extra
        inv = shrink * cute.math.rcp(mass, approx=True, ftz=True)
        value = cutlass.Float32(top + cutlass.Float64(cute.math.log2(mass, fastmath=True)) * cutlass.Float64(LN2))
    return inv, value
