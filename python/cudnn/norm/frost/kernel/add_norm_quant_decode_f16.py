# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The add + RMSNorm + quantize DECODE kernel (bf16 in; bf16, E4M3 or NVFP4 out; one launch): per row
``y = bf16( v * rsqrt(mean(v^2) + eps) * (1 + weight) )`` with ``v = x + residual`` (or ``x``), the residual updated in place, and the
codes of ``y`` in the convention of the projection that consumes them.  Config and forms in ``add_norm_quant_decode_config.py``, the
host entry in ``norm/frost/add_norm_quant_decode.py``.  Arch-portable PTX (global loads / stores, ``cvt``, ``div.rn``,
``rsqrt.approx``, ``rcp.approx.ftz``, ``shfl.sync``, ``bar.sync``, ``griddepcontrol``); the arch axis is an admission fact of the host
entry, not a directory.

THREAD BLOCK: one per row; ``cfg.THREADS = hidden / 8`` threads in ``cfg.WARPS`` FULL warps; thread ``t`` owns the eight contiguous
elements ``8 t .. 8 t + 7`` of its row.  Every thread has the same role; lane 0 of each warp additionally publishes its warp's sum,
lanes ``0 .. WARPS - 1`` of each warp read the published sums, and in the NVFP4 form the even thread of each pair of threads (one
16-element block) stores the block's scale byte.

PIPELINE, in order, for every thread:

1. ``idx = row x hidden + 8 t`` (signed 32-bit: the host entry bounds ``rows x hidden``).
2. Pre-wait loads: one 128-bit global load of the thread's eight ``weight`` values and, in a quantizing form, one 32-bit load of
   ``scale[0]``.  Both are constants of the layer (the PDL-sites paragraph).
3. ``griddepcontrol.wait`` under ``cfg.PDL``; ``griddepcontrol.launch_dependents`` under ``cfg.PDL_RELEASE``, directly behind the wait.
4. One 128-bit global load of ``x`` (and of ``residual`` under ``cfg.ADD``); bf16 -> fp32 is exact; ``v = x + residual`` is ONE fp32
   addition per element.
5. The thread's partial sum of squares: ``s = fma(v_j, v_j, s)`` for ``j = 0 .. 7`` from ``+0`` -- one FUSED multiply-add per element,
   in element order.  ``g_j = weight_j + 1`` (one fp32 addition).
6. Under ``cfg.ADD`` or ``cfg.STORE_RESIDUAL``: ``residual <- bf16(v)`` (nearest-even), one 128-bit global store.
7. Warp reduction: a butterfly over the 32 lanes with offsets 1, 2, 4, 8, 16 (``shfl.sync.bfly`` + one addition each); every lane of
   a warp then holds the same bits.
8. Lane 0 stores its warp's sum to shared memory (slot = warp index); the thread-block barrier; lanes ``0 .. WARPS - 1`` load slot
   ``lane``, the other lanes hold ``+0``; the same five-step butterfly.  Every thread of the block then holds the same row total.
9. ``inv = rsqrt.approx( div.rn(total, hidden) + eps )``: an IEEE division by the width, one addition, the hardware's approximate
   reciprocal square root.
10. ``y_j = bf16( (v_j x inv) x g_j )``: two fp32 multiplications, then nearest-even to bf16 -- THE normalized value of record; it stays
    in registers, widened back exactly, and the codes read it.  Under ``cfg.BF16_OUT`` one 128-bit global store of the eight values.
11. ``cfg.QUANT == "e4m3_static"``: ``code_j = e4m3( y_j x div.rn(1, scale) )`` (nearest-even, saturating at +-448), eight one-byte
    global stores.
12. ``cfg.QUANT == "nvfp4"``: the block maximum of ``|y|`` over the thread's eight elements and its neighbour's (one ``shfl.sync.bfly``
    of offset 1); the block scale ``byte, value = e4m3( scale x (max x fp32(1/6)) )``; the encode scale ``0`` for a zero block scale,
    else ``rcp.approx.ftz( value x rcp.approx.ftz(scale) )``; ``code_j = e2m1( y_j x encode scale )`` (nearest-even, saturating at
    +-6), two codes per byte (the even element in the low nibble), one 32-bit global store of the thread's four code bytes; the even
    thread of the pair stores the scale byte at its position in the 128 x 4 scale layout over (rows, blocks).

THE SUM TREE AND THE ROUNDING CHAIN ARE THE CONTRACT.  The reduction is spelled inline on purpose: the fused chain in element order,
the butterfly offsets ASCENDING (1, 2, 4, 8, 16) at both levels, an IEEE division by the width (never a multiplication by
``1 / hidden``) and the default ``rsqrt``.  ``tile_dsl.pointwise.lane_group_sum`` pairs lanes in the opposite order (16, 8, 4, 2, 1)
and ``rmsnorm_rstd`` multiplies by a reciprocal width: the same quantity with other last bits, hence another bf16 value on some
elements.  The fused multiply-add is explicit (``cute.math.fma``), never left to the optimizer's contraction of ``s + v * v``; the
NVFP4 factor 1/6 is the bit pattern ``0x3E2AAAAB`` inside ``nvfp4_block_scale_global``, never an approximate reciprocal of a constant
that an assembler folds.  ``y`` is rounded to bf16 BEFORE any code is formed, so the codes are those of quantizing ``y`` itself.  The
test module's rendering cell pins these instructions; its reference cells pin the bytes.

BATCH INVARIANCE AND REPLAY.  A row's thread block reads its own row of ``x`` / ``residual``, the shared ``weight`` and ``scale``, and
nothing of another row; the reduction never leaves the block; there is no atomic and no cross-row scratch.  A row's bytes are a
function of its operands and of (``hidden``, ``eps``, ``ADD``, ``QUANT``) alone.  The row index enters ADDRESSES only -- including the
position of an NVFP4 scale byte in the 128 x 4 layout, which moves with the row index while its value does not.

BARRIER TABLE.  No mbarrier, no cluster, no ``cp.async`` group, no loop with a run-time trip count.  ONE synchronization object:

    barrier                         kind                 participants      issuing lanes            guard
    the thread-block barrier (8)    named barrier 0      cfg.THREADS       WARPS x 32, once each    none (const_expr-only control flow)

``SUM(issuing lanes) = 32 x WARPS == THREADS``: every thread the barrier counts reaches it exactly once on every path (the only
run-time branch in front of it, lane 0's slot store, rejoins before it).  The config pins ``THREADS == hidden / 8`` in full warps.

SHARED-MEMORY TABLE.  ONE buffer; no tensor memory, no descriptor:

    buffer      dtype x elems    writer                          reader                               refilled in a launch
    warp sums   fp32 x WARPS     lane 0 of warp w, slot w,       lanes 0 .. WARPS-1 of every warp,    no: written once, read
                                 before the barrier              slot = lane, behind the barrier      once -> nothing releases it

Lanes address the slots directly at a 4-byte stride (one word per lane in an access phase); no swizzle because no descriptor reads it.

PDL SITES.  Under ``cfg.PDL`` the launch carries the programmatic-stream-serialization attribute and every thread issues
``griddepcontrol.wait`` AFTER the pre-wait loads of step 2 and BEFORE its first read of ``x`` / ``residual`` and before every global
store.  CALLER OBLIGATION: ``weight`` and ``scale`` are not written by any launch that may still be running when this one starts (they
are constants of the layer).  Under ``cfg.PDL_RELEASE`` (which needs ``cfg.PDL``) the release follows this launch's OWN wait directly:
a dependent's pre-wait part can run beside this kernel, never beside this kernel's predecessor.  CALLER OBLIGATION: every grid that can
start before this one completes issues its own wait before it reads this launch's outputs and before it writes any operand of this
launch.  The kernel holds no tensor memory and 4 x WARPS bytes of shared memory, so a co-resident dependent starves nothing.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory

from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_times_inv_to_e4m3, fp32_to_e2m1x2, fp32_to_fp16, nvfp4_block_scale_global, rcp_div_rn
from cudnn.frost.tile_dsl.sf_layout import sf_atom_offset
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global, st_global_v4

from .add_norm_quant_decode_config import SF_ATOM_BYTES, SF_TILE_ROWS, VEC, WARP

FP32, INT32, BF16, UINT8 = cutlass.Float32, cutlass.Int32, cutlass.BFloat16, cutlass.Uint8
PAIRS = VEC // 2  # element pairs of a thread: one 32-bit word of a 128-bit bf16 load or store, one byte of NVFP4 codes
BUTTERFLY_STEPS = 5  # offsets 1, 2, 4, 8, 16: the 32 lanes of a warp


def _bf16_words(vals):
    """The thread's eight fp32 values as the four 32-bit words of one 128-bit bf16 store (nearest-even, two values per word)."""
    return [fp32_to_fp16(vals[2 * p], vals[2 * p + 1], dtype=BF16) for p in range(PAIRS)]


@cute.kernel
def frost_add_rms_norm_quant(
    cfg: cutlass.Constexpr,
    x: cute.Tensor,
    res: Optional[cute.Tensor],  # None unless cfg.ADD or cfg.STORE_RESIDUAL
    w: cute.Tensor,
    scale: Optional[cute.Tensor],  # None unless cfg.QUANT is not None
    y: Optional[cute.Tensor],  # None unless cfg.BF16_OUT
    codes: Optional[cute.Tensor],  # None unless cfg.QUANT is not None
    sf: Optional[cute.Tensor],  # None unless cfg.QUANT == "nvfp4"
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tid % WARP
    warp = tid // WARP
    col = tid * VEC
    idx = row * cfg.HIDDEN + col
    # the norm weight and the scale are constants of the layer: they load BEFORE the dependent-launch wait; x and the residual behind it
    wv = ld_global_v4((w.iterator + col).toint(), INT32)
    gs = FP32(0)
    if cutlass.const_expr(cfg.QUANT is not None):
        gs = scale[0]
    if cutlass.const_expr(cfg.PDL):
        wait_on_dependent_grids()
    if cutlass.const_expr(cfg.PDL_RELEASE):
        launch_dependent_grids()  # directly behind this launch's own wait (the config pins PDL_RELEASE -> PDL)
    smem = cutlass.memory.SmemAllocator().allocate_tensor(FP32, cute.make_layout(cfg.WARPS), byte_alignment=16)
    vals = cute.make_rmem_tensor((VEC,), FP32)
    weights = cute.make_rmem_tensor((VEC,), FP32)

    xv = ld_global_v4((x.iterator + idx).toint(), INT32)
    sumsq = FP32(0)
    if cutlass.const_expr(cfg.ADD):
        rv = ld_global_v4((res.iterator + idx).toint(), INT32)
    for p in cutlass.range_constexpr(PAIRS):
        v0, v1 = f16x2_to_f32(xv[p], dtype=BF16)
        if cutlass.const_expr(cfg.ADD):
            r0, r1 = f16x2_to_f32(rv[p], dtype=BF16)
            v0 = v0 + r0
            v1 = v1 + r1
        vals[2 * p] = v0
        vals[2 * p + 1] = v1
        # byte contract: ONE fused multiply-add per element, in element order, from +0 -- explicit, never the optimizer's contraction
        sumsq = cute.math.fma(v0, v0, sumsq)
        sumsq = cute.math.fma(v1, v1, sumsq)
        w0, w1 = f16x2_to_f32(wv[p], dtype=BF16)
        weights[2 * p] = w0 + FP32(1)
        weights[2 * p + 1] = w1 + FP32(1)
    # ADD updates the residual in place; STORE_RESIDUAL (no add) initializes it to x
    if cutlass.const_expr(cfg.ADD or cfg.STORE_RESIDUAL):
        st_global_v4((res.iterator + idx).toint(), _bf16_words(vals), INT32)
    # byte contract: the butterfly offsets ASCEND (1, 2, 4, 8, 16); tile_dsl's lane_group_sum pairs lanes in the opposite order
    for step in cutlass.range_constexpr(BUTTERFLY_STEPS):
        sumsq = sumsq + cute.arch.shuffle_sync_bfly(sumsq, offset=1 << step)
    if lane == 0:
        smem[warp] = sumsq
    cute.arch.barrier()  # the ONE thread-block barrier: all cfg.THREADS threads, under const_expr-only control flow
    total = FP32(0)
    if lane < cfg.WARPS:
        total = smem[lane]
    for step in cutlass.range_constexpr(BUTTERFLY_STEPS):
        total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << step)
    # byte contract: an IEEE division by the width (never a multiplication by 1 / width), then the default rsqrt (rsqrt.approx.f32)
    inv_rms = cute.math.rsqrt(total / FP32(cfg.HIDDEN) + FP32(cfg.EPS))

    amax = FP32(0)
    for j in cutlass.range_constexpr(VEC):
        v = ((vals[j] * inv_rms) * weights[j]).to(BF16).to(FP32)  # the bf16 rounding of record; the codes read this value
        vals[j] = v
        amax = cutlass.max(amax, cute.math.abs(v))
    if cutlass.const_expr(cfg.BF16_OUT):
        st_global_v4((y.iterator + idx).toint(), _bf16_words(vals), INT32)
    if cutlass.const_expr(cfg.QUANT == "e4m3_static"):
        inv = rcp_div_rn(gs)  # the IEEE reciprocal of the consumer's static scale
        for j in cutlass.range_constexpr(VEC):
            codes[idx + j] = fp32_times_inv_to_e4m3(vals[j], inv)
    if cutlass.const_expr(cfg.QUANT == "nvfp4"):
        # two adjacent threads own one 16-element block; gs is the global scale
        amax = cutlass.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
        s8, out_scale = nvfp4_block_scale_global(amax, gs)  # the stored block scale and the block's encode scale
        payload = INT32(0)
        for p in cutlass.range_constexpr(PAIRS):
            byte = fp32_to_e2m1x2(vals[2 * p] * out_scale, vals[2 * p + 1] * out_scale)
            payload = payload | (byte << INT32(p * 8))
        st_global((codes.iterator + (row * (cfg.HIDDEN // 2) + tid * PAIRS)).toint(), payload, INT32)
        if tid % 2 == 0:
            block = tid // 2
            sf[(row // SF_TILE_ROWS) * (cfg.SF_K_TILES * SF_ATOM_BYTES) + sf_atom_offset(row % SF_TILE_ROWS, block)] = s8.to(UINT8)


frost_add_rms_norm_quant.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch(cfg: cutlass.Constexpr, x, res, w, scale, y, codes, sf, rows: cutlass.Int32, stream: cuda.CUstream):
    """One launch: grid ``(rows, 1, 1)``, ``cfg.THREADS`` threads, the programmatic-stream-serialization attribute under ``cfg.PDL``.
    ``rows`` is a runtime argument and the operand views are layout-dynamic; the config is the compile key."""
    frost_add_rms_norm_quant(cfg, x, res, w, scale, y, codes, sf).launch(grid=(rows, 1, 1), block=(cfg.THREADS, 1, 1), stream=stream, use_pdl=cfg.PDL)
