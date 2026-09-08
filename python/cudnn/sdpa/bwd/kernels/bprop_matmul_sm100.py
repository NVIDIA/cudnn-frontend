# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100 batched GEMM with a 2-D ``(batch, head)`` batch — SDPA-backward stage 3.

Computes dV = S^T.dO, dK = dS^T.Q and dQ = dS.K.  All three are plain batched
bf16 GEMMs; at bf16 there is no descale epilogue, so nothing is fused here.

WHY THIS EXISTS -- the one thing the generic GEMM cannot express
---------------------------------------------------------------
``gemm/frost/kernel_templates/sm100_matmul.py`` carries a single batch axis
``l`` with ONE uniform stride.  The SDPA operands are BSHD ``[B, S, H, D]``, so
the batch element is the PAIR ``(b, h)`` at offset ``b*(S*H*D) + h*D`` -- a
two-level stride that no single uniform stride can express.  Flattening it
host-side would need a copy of every operand, per chunk, against a workspace
measured in GiB.

So the operands stay exactly as the user laid them out and the TMA descriptors
become **4-D** ``[k, m, h, b]`` (``cuTensorMapEncodeTiled`` allows 5), with
``h`` and ``b`` as separate coordinates.  The CLC scheduler still rasterizes a
single flat ``l`` -- ``_decode_bh`` splits it only where a TMA coordinate is
formed, which is the whole change.

KEEP IN SYNC WITH ``gemm/frost/kernel_templates/sm100_matmul.py``
-----------------------------------------------------------------
This file is a FORK of that template, taken from its rendered dense-bf16
expansion (config ``sm100_128x256x128_128x256x32_cluster2x1_2ctamma``, no
epilogue fusion, TMA-store epilogue).  The mainloop, the CLC scheduler, the
TMEM/accumulator pipeline and the epilogue are otherwise unchanged.

**Any correctness fix or performance improvement to either file should be
applied to BOTH.**  The generic template carries the same note.  The diff
against it is deliberately narrow, so a `diff` against a fresh rendering of
that config is the intended way to review a change:
  * ``_decode_bh`` and its four call sites;
  * ``h``/``b`` in place of ``l`` in every TMA coordinate tuple;
  * 4-D descriptors and one extra stride per operand;
  * ``problem_size`` carrying ``(n_head, n_batch)`` and 4 strides per operand.

Everything else in this file is generated output; do not hand-tune it in place
without making the same change upstream.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
from cudnn.gemm.frost.kernel_templates._tile_helpers import (
    epi_subtile_spans as _epi_subtile_spans,
    l2_swizzle_tile as _l2_swizzle_tile,
    tcgen05_alloc as _tcgen05_alloc,
    tcgen05_dealloc as _tcgen05_dealloc,
    tcgen05_mma as _tcgen05_mma,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass._mlir_helpers.vector as _cvec
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda
from cutlass.cute.arch import clc as cute_clc

from cudnn.frost.tile_dsl.constants import DTYPE_FP16
from cudnn.frost.tile_dsl.thd import TENSOR_MAP_QWORDS, emit_clamped_desc, emit_seq_descs
from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, MatmulTemplateParams, validate_matmul_params

PARAMS = globals().get("FROST_TEMPLATE_PARAMS", MatmulTemplateParams())
validate_matmul_params(PARAMS)

# A/B/D io dtype.  BF16 and FP16 are both 2 B/element and both take the
# Tcgen05MMAKind.F16 path, so this is a token swap: nothing byte-sized below
# changes.  It must agree with the stage-2 template's `dtype_qkv` -- stage 3
# reads the S/dS workspace stage 2 wrote.
_IO_DTYPE = cutlass.Float16 if int(PARAMS.dtype_qkv) == DTYPE_FP16 else cutlass.BFloat16

# Tile config: CONFIG_sm100_256x256x128_128x256x32_cluster2x2_2ctamma
# (was ...cluster2x1...; swapped on request. Every constant below is lifted
# verbatim from the upstream rendering of THIS config -- do not hand-derive.)
mma_inst_shape_mnk = (256, 256, 16)
cta_group = 2
cgrp_tile_mnk = (512, 512, 64)
cta_tile_mnk = (256, 128, 64)
epi_tile_mn = (128, 64)
threads_per_cta = 256
cluster_shape_mnk = (2, 2, 1)
# Upstream carries the rendering's batch size here purely to detect a BROADCAST
# operand (== 1 means "one batch element, reuse it for all"). No stage-3 GEMM
# broadcasts, so both are simply "batched"; the real extents are runtime.
matmul_a_batch = 0
matmul_b_batch = 0
# --- operand major, per FROST_TEMPLATE_PARAMS -------------------------------
# The three stage-3 GEMMs do NOT share one operand-major combination:
#   dV = S^T.dO  and  dK = dS^T.Q :  A m-major (S/dS's kv is contiguous and is M)
#   dQ = dS.K                     :  A k-major (dS's kv is contiguous and is K)
# B is n-major in all three (D is contiguous in BSHD, and D is N).
#
# Rendering the upstream template for each combination produces bodies that are
# textually IDENTICAL -- the whole difference is the ten constants below.  So one
# fork serves all three; the loader instantiates this module once per params set
# and every use is `cutlass.const_expr`, so each instance traces specialized code.
a_is_m_major = bool(PARAMS.a_is_m_major)
# THD / varlen.  Everything below folds out of the dense rendering, which is the
# one that has to keep diffing clean against the upstream template.
#
# Which axis is ragged follows from the operand major, so it needs no parameter
# of its own:
#     dV = S^T.dO, dK = dS^T.Q  (A m-major) -- M is kv, K is q tokens
#     dQ = dS.K                 (A k-major) -- M is q tokens, K is kv
# so A's blocked-workspace row offset lands on K in the first case and on M in
# the second, and B's token offset is cu_q in the first and cu_k in the second.
_THD_MM = bool(getattr(PARAMS, "thd_varlen", False))
# This GEMM's OWN descriptor scratch: one clipped output descriptor per
# sequence, then the packed-total-clamped B operand.  Built in `_host`, which is
# the only place that knows these descriptors' box, swizzle and dim order -- for
# (n, m, h, b) operands the sequence axis is ord=1, which is NOT stage 2's.
_THD_MM_SEQ_ORD = 1
THD_MM_DESC_SLOTS = lambda b: b + 1  # noqa: E731
B_CLAMP_SLOT = lambda b: b  # noqa: E731


@cute.jit
def _thd_desc_ptr(desc_words, slot):
    """Generic-space pointer to one 128-B tensor map in the patched array."""
    return (desc_words.iterator.raw_ptr() + slot * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)


@cute.jit
def _thd_acquire_descs(desc_words, n_batch):
    """Acquire EVERY patched descriptor into the TMA proxy, once per warp.

    ``fence.proxy.tensormap::generic.acquire`` takes a size operand whose ONLY
    legal value is 128 -- one descriptor.  Acquiring the array's base therefore
    orders the FIRST slot and nothing else, so a warp that later selects slot
    ``tile_b`` (the per-sequence C descriptors) or slot ``n_batch`` (the clamped
    B descriptor) could read metadata the TMA proxy still has stale.  Loop the
    whole array instead: ``n_batch + 1`` fences, hoisted out of the persistent
    loop, because the patch launch writes these once before this kernel starts
    and never rewrites them.
    """
    for _slot in cutlass.range(n_batch + cutlass.Int32(1)):
        nvvm.fence_proxy_acquire(
            nvvm.MemScope.GPU,
            _thd_desc_ptr(desc_words, _slot),
            128,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )


b_is_n_major = bool(PARAMS.b_is_n_major)
causal_mode = int(PARAMS.causal_mode)
causal_gran = int(PARAMS.causal_gran)
causal_shift = int(PARAMS.causal_shift)
mma_a_major = 1 if a_is_m_major else 0
mma_b_major = 1 if b_is_n_major else 0
ab_stages = 4
b_collector_ok = False
multicast_a = True
multicast_b = False
# Under a cluster with cga_n > 1 the A operand's MULTICAST also follows its
# major -- a K-major A is split across the cluster's N columns (2 slices) while
# an M-major A is broadcast whole. That was invisible at cluster2x1, where both
# are 1, and it is why this table cannot be reused across cluster shapes
# unchanged. Values lifted from the two upstream renderings of this config.
_A_MCAST = {False: (2, False), True: (1, True)}  # is_m_major: (slices, empty_full_mask)
a_mcast_slices, ab_empty_full_mask = _A_MCAST[a_is_m_major]
b_mcast_slices = 1
ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B
# MN-major packs 64 elements per TMA group and walks K in 2048-byte steps;
# K-major loads one group and steps 32 bytes.  Values lifted verbatim from the
# upstream renderings of the same tile config -- do not hand-derive them.
_MAJOR_CONSTS = {
    # is_mn_major: (desc_leading_byte_offset, k_step_bytes, tma_group_elems)
    False: (16, 32, 1),
    True: (8192, 2048, 64),
}
a_smem_desc_leading_byte_offset, a_smem_k_step_bytes, a_tma_group_elems = _MAJOR_CONSTS[a_is_m_major]
b_smem_desc_leading_byte_offset, b_smem_k_step_bytes, b_tma_group_elems = _MAJOR_CONSTS[b_is_n_major]
a_smem_desc_stride_byte_offset = 1024
b_smem_desc_stride_byte_offset = 1024
a_smem_m_step_bytes = 16384
mma_size_m = 2
mma_size_n = 1
mma_size_k = 4
ab_tma_swizzle = _tma.TensorMapSwizzle.s128b

# Dtype family: A=f16->MMAf16, B=f16->MMAf16, out=f16 (K_BYTES=128).
# `_IO_DTYPE` is BF16 or FP16 per PARAMS.dtype_qkv; the MMA kind is the same.
ab_dtype = _IO_DTYPE
cd_dtype = _IO_DTYPE
mma_a_dtype = _IO_DTYPE
mma_b_dtype = _IO_DTYPE
mma_c_dtype = cutlass.Float32
acc_widen_to_fp32 = False
ab_tma_dtype = _IO_DTYPE
mma_kind = nvvm.Tcgen05MMAKind.F16
epi_n = 64
epi_row_elems = 64
tile_swizzle_n = 1
swizzle_l2_budget_bytes = 44214954
num_gemms = 1
num_a_operands = 1
num_b_operands = 1
gemm_a_idx = (0,)
gemm_b_idx = (0,)
num_tmem_alloc_cols = 512
tmem_alloc_exclusive = False
acc_stages = 1  # 512 acc cols/stage
vec_bytes_epi = int(PARAMS.vec_bytes_epi)
frost_compile_options = "--enable-tvm-ffi --gpu-arch sm_100a"
n_tma_outputs = 1
moe_aligned_offsets = False
epi_slot_widen = 1
epi_packed_lanes = False
epi_dp22 = False
epi_stage_rows = 128
epi_chunk_elems = 64
ab_stages = 4  # SMEM-D 32784B fixed + cast LOAD 0B/stage + multi-GEMM 0B/stage
# Upstream renders (2, 1, 1) here -- a mixed-CGA fallback the driver may pick
# per cluster when the preferred shape does not fit. Pinned to None in this
# fork: `_host` always sizes the grid as a multiple of the preferred cluster, so
# the fallback is unreachable by construction, and its paths carry their own
# coordinate/multicast logic that the 2-D (b, h) batch rewrite has never
# exercised.
fallback_cluster_shape_mnk = None
mixed_a_pattern_pref = 5
mixed_b_pattern_pref = 1
mixed_a_pattern_fb = 1
mixed_b_pattern_fb = 1

# Rank decomposition below uses shifts and masks instead of runtime integer
# division.  The catalog satisfies this; keep synthesized configs from silently
# taking the fast path with a non-power-of-two cluster dimension.
if any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in cluster_shape_mnk[:2]):
    raise NotImplementedError(f"{__name__}: cluster M/N dimensions must be powers of two")
if fallback_cluster_shape_mnk is not None and any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in fallback_cluster_shape_mnk[:2]):
    raise NotImplementedError(f"{__name__}: fallback cluster M/N dimensions must be powers of two")

# Keep the two launch alternatives as host constants and spell the preferred /
# fallback operations at each use site. This exposes constant masks and shift
# alternatives before backend canonicalization.
_preferred_cluster_m_shift = cluster_shape_mnk[0].bit_length() - 1
_preferred_cluster_n_shift = cluster_shape_mnk[1].bit_length() - 1
_fallback_cluster_m_shift = _preferred_cluster_m_shift if fallback_cluster_shape_mnk is None else fallback_cluster_shape_mnk[0].bit_length() - 1
_fallback_cluster_n_shift = _preferred_cluster_n_shift if fallback_cluster_shape_mnk is None else fallback_cluster_shape_mnk[1].bit_length() - 1
_CTA_GROUP = nvvm.CTAGroup.CTA_2 if cta_group == 2 else nvvm.CTAGroup.CTA_1
_cta_group_shift = cta_group.bit_length() - 1


# Scheduler ring depth.
CLC_SCHED_STAGES = 1

# Programmatic Dependent Launch (PDL, sm_90+).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path.
EPI_SMEM_STAGES = 2

# Named barrier id for the 4-warp epilogue handoff around the TMA store.
EPI_SYNC_BAR_ID = 1

# Named barrier id for the TMEM-alloc handoff.
TMEM_ALLOC_BARRIER_ID = 2


@cute.jit
def _auto_swizzle_w(m, n, k, nt_n):
    """N-super-block width for the tile rasterization, resolved per launch.

    ``tile_swizzle_n > 0`` pins it. Otherwise: the walk keeps one operand slice
    resident and re-reads the other every super-block, so block along the SHORTER
    problem side. Once that side outgrows what L2 can hold onto while C streams
    through it, keeping it is no longer free -- fall back to the widest N block the
    budget does cover.
    """
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_dtype.width) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    if cutlass.min(m, n) * row_bytes <= budget and m <= n:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


def _a_collector_op(g):
    if cutlass.const_expr(num_gemms == 1 or num_a_operands != 1 or mma_size_m != 1):
        return None
    if cutlass.const_expr(g == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(g == num_gemms - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


def _b_collector_op(mi):
    if cutlass.const_expr(not b_collector_ok or mma_size_m == 1):
        return None
    if cutlass.const_expr(mi == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(mi == mma_size_m - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


@cute.jit
def _decode_bh(l, n_head):
    """Flat CLC batch index -> ``(head, batch)``.

    The CLC scheduler rasterizes ONE batch axis, so ``l`` stays flat there and
    the tile hand-out is unchanged.  The pair is only needed where a TMA
    coordinate is formed -- see the module docstring for why the tensors are
    not flattened instead.
    """
    return l % n_head, l // n_head


@cute.jit
def _thd_group(meta_t, tile_b, n_batch, num_k_tiles):
    """Per-sequence offsets and k-tile count for one (head, sequence) group.

    Returns ``(a_k_off, a_m_off, b_k_off, num_k_tiles)``.  Dense returns zeros
    and the kernel-wide tile count, so every use folds away.

    The k count comes from the sequence's REAL length, which is safe on both
    sides of the ragged axis because stage 2 zero-fills further than this reads:
    its blocked rows run to ``ceil(S_q/128)*128`` and its columns to
    ``ceil(S_kv/128)*128``, both at least the ``ceil(len/64)`` tiles counted
    here.  Reading further would reach the NEXT sequence's live rows -- nonzero
    data summed into the wrong gradient, with nothing to make it look wrong.
    """
    if cutlass.const_expr(not _THD_MM):
        return cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), num_k_tiles
    meta = cutlass.make_array_view(meta_t)
    cu_q0 = n_batch
    cu_k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
    row0 = cutlass.Int32(4) * n_batch + cutlass.Int32(4)
    q_tok = cutlass.Int32(meta[cu_q0 + tile_b])
    k_tok = cutlass.Int32(meta[cu_k0 + tile_b])
    s_q = cutlass.Int32(meta[cu_q0 + tile_b + cutlass.Int32(1)]) - q_tok
    s_kv = cutlass.Int32(meta[cu_k0 + tile_b + cutlass.Int32(1)]) - k_tok
    row_off = cutlass.Int32(meta[row0 + tile_b])
    a_k_off = row_off if cutlass.const_expr(a_is_m_major) else cutlass.Int32(0)
    a_m_off = cutlass.Int32(0) if cutlass.const_expr(a_is_m_major) else row_off
    b_k_off = q_tok if cutlass.const_expr(a_is_m_major) else k_tok
    k_len = s_q if cutlass.const_expr(a_is_m_major) else s_kv
    nkt = (k_len + cutlass.Int32(cta_tile_mnk[2] - 1)) // cutlass.Int32(cta_tile_mnk[2])
    return a_k_off, a_m_off, b_k_off, nkt


@cute.jit
def _causal_k_range(coord_m_cgrp, num_k_tiles):
    """The K-tile range this M tile may read, under stage 2's causal skip.

    Stage 2 leaves the tiles above the diagonal UNWRITTEN, so what is outside
    this range is whatever the caller left in the workspace.

    Whether that makes the range a CORRECTNESS bound or only an optimization
    depends on the tight-trim invariant::

        cgrp_tile_mnk[0] <= causal_gran

    i.e. one cluster M tile fits inside one stage-2 write block. It HELD at the
    2x1 cluster config (256 <= 256) and does NOT hold at the 2x2 config now in
    use (512 > 256): a 512-row M tile straddles two 256-row stage-2 blocks, and
    since the bounds below are per-tile, no range can cover the tile's live rows
    without also covering its neighbour's skipped ones.

    So at 2x2 this is an OPTIMIZATION ONLY, and the caller's zero-fill is what
    makes it correct -- `api_dsl` sets `_zero_ws` whenever the trim is active,
    which is exactly the condition under which any of this runs. Do not weaken
    that zero-fill to "only when shift != 0" on the theory that the trim
    protects the aligned case; at 2x2 it does not.

    The 2x2 config is kept despite the looser trim because it measures faster
    BOTH ways at B=1 H=128 S=8192 d=512 bf16: +3.5% no_mask, +7.8% causal (the
    wider tile more than pays for the extra k-tiles it reads).

    Keyed on the CLUSTER's M tile base (``tile_m * cgrp_tile_m``), which is
    identical on both CTAs of a pair, and rounded to ``causal_gran`` -- stage
    2's own write granularity.  Rounding OUTWARD (down for the low bound, up for
    the high one) is what keeps the range covering every structurally non-zero
    tile; under the tight-trim invariant that outward rounding also keeps it a
    subset of what stage 2 wrote.

    Never reached under THD: the adapter renders the packed stage 3 with
    ``causal_mode=CAUSAL_K_NONE`` even for a causal graph, because every bound
    here is an ABSOLUTE workspace row and the blocked layout renumbers rows per
    sequence.  ``validate_matmul_params`` enforces that.  See
    ``SdpaBwdDslSm100.compile``.
    """
    # num_k_tiles is Int64 (it derives from the Int64 `k`); normalise so the
    # two bounds and the min() below share one numeric type.
    nkt = cutlass.Int32(num_k_tiles)
    if cutlass.const_expr(causal_mode == CAUSAL_K_NONE):
        return cutlass.Int32(0), nkt
    blk = cutlass.Int32(causal_gran)
    tk = cutlass.Int32(cta_tile_mnk[2])
    m0 = cutlass.Int32(coord_m_cgrp)
    shift = cutlass.Int32(causal_shift)
    # NEVER return an empty range. The mainloop would run zero iterations, but
    # the EPILOGUE still stores the accumulator -- and `scale_d` starts False, so
    # with no MMA the accumulator is uninitialised TMEM and the output row is
    # garbage. One k-tile of a ZEROED workspace contributes exactly 0, which is
    # the answer those rows want anyway (they are structurally masked out).
    # This is why the caller must zero S/dS whenever the trim is active.
    if cutlass.const_expr(causal_mode == CAUSAL_K_LO):
        # dV / dK: output row is kv, so K (= q) starts at the stage-2 block
        # holding kv -- pulled EARLIER by the shift, because S[q, kv] is
        # non-zero for q >= kv - shift.  Clamped at 0.
        lo = m0 - shift
        lo = cute.math.max(lo, cutlass.Int32(0))
        k_lo = ((lo // blk) * blk) // tk
        return cute.math.min(k_lo, nkt - cutlass.Int32(1)), nkt
    # dQ: output row is q, so K (= kv) ends after q's stage-2 block, pushed
    # LATER by the shift (kv <= q + shift).
    hi = ((m0 + cutlass.Int32(cgrp_tile_mnk[0] - 1) + shift) // blk + cutlass.Int32(1)) * blk
    hi = cute.math.max(hi, blk)
    k_hi = cute.math.min((hi + tk - cutlass.Int32(1)) // tk, nkt)
    return cutlass.Int32(0), cute.math.max(k_hi, cutlass.Int32(1))


@cute.kernel
def _bprop_matmul_bh_sm100_kernel(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_h_0: cutlass.Int64,
    out_stride_b_0: cutlass.Int64,
    n_head: cutlass.Int32,
    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],
    # Dense: 1-element dummies.  THD: the metadata buffer the setup launch
    # published, and this GEMM's own descriptor scratch (per-sequence clipped C,
    # then the packed-total-clamped B).
    meta_t: cute.Tensor,
    desc_words: cute.Tensor,
    n_batch: cutlass.Int32,
) -> None:
    tma_a_descs = [tma_a_desc_0]
    tma_b_descs = [tma_b_desc_0]
    tma_c_descs = [tma_c_desc_0]

    mma_warp_id = 4
    tma_warp_id = 5
    scheduler_warp_id = 6
    unused_warp_id = 7
    num_epilogue_warps = 4
    epi_reg_count = 232
    prod_reg_count = 24

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]
    gridx = cute.arch.grid_dim()[0]
    gridy = cute.arch.grid_dim()[1]

    # Mixed CGA: the launch carries a preferred (wide) cluster plus a smaller
    # fallback one, and the device picks per cluster — a CTA can only tell which
    # by reading the hardware cluster dims. Everything cluster-shaped below then
    # follows from those, so the two kinds share one body; only the multicast bit
    # patterns are loop-built and come in precomputed per shape.
    a_mcast_pattern = mixed_a_pattern_pref
    if cutlass.const_expr(cta_group == 2):
        b_mcast_pattern = mixed_b_pattern_pref
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        cluster_m = cluster_shape_mnk[0]
        cluster_n = cluster_shape_mnk[1]
    else:
        cdim_x, cdim_y, _cdim_z = cute.arch.block_in_cluster_dim()
        cluster_m = cdim_x
        cluster_n = cdim_y
        a_mcast_pattern = cutlass.Int32(mixed_a_pattern_pref)
        if cutlass.const_expr(cta_group == 2):
            b_mcast_pattern = cutlass.Int32(mixed_b_pattern_pref)
        # Bitwise, not `or`: both operands are runtime Booleans (this is the form
        # cutlass.cute.experimental.is_preferred_cluster uses).
        if (cdim_x != cluster_shape_mnk[0]) | (cdim_y != cluster_shape_mnk[1]):
            a_mcast_pattern = cutlass.Int32(mixed_a_pattern_fb)
            if cutlass.const_expr(cta_group == 2):
                b_mcast_pattern = cutlass.Int32(mixed_b_pattern_fb)
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    # Every catalog cluster dimension is a power of two.  Mixed-CGA makes the
    # divisor runtime-visible, so spelling rank decomposition as div/mod would
    # otherwise lower to reciprocal-based integer division in every warp.
    m_rank = cta_rank_in_cluster & (cluster_shape_mnk[0] - 1)
    n_rank = cta_rank_in_cluster >> _preferred_cluster_m_shift
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
            m_rank = cta_rank_in_cluster & (fallback_cluster_shape_mnk[0] - 1)
            n_rank = cta_rank_in_cluster >> _fallback_cluster_m_shift

    if cutlass.const_expr(cta_group == 2):
        pair_member = m_rank % cta_group
        pair_m_idx = m_rank // cta_group
        is_pair_leader = pair_member == 0
        pair_leader_rank = pair_m_idx * cta_group + n_rank * cluster_m
    else:
        pair_member = 0
        pair_m_idx = m_rank
        is_pair_leader = True
        pair_leader_rank = cta_rank_in_cluster

    is_cluster_leader_cta = cta_rank_in_cluster == 0

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        # B and C's GridConstant descriptors are DEAD under THD: B loads take the
        # packed-total-clamped slot and C stores take the per-sequence one, both
        # from the patched array.  Prefetching them would warm metadata nobody
        # reads -- and the patched slots must NOT be prefetched here in their
        # place, because that would cache their contents ahead of the
        # `fence_proxy_acquire` that makes the patch visible to the TMA proxy.
        if cutlass.const_expr(not _THD_MM):
            for _j in cutlass.range_constexpr(num_b_operands):
                nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

            for _ci in cutlass.range_constexpr(n_tma_outputs):
                nvvm.prefetch_tensormap(tma_c_descs[_ci].get_ptr())

    init_raw_m = bidx >> _preferred_cluster_m_shift
    init_raw_n = bidy >> _preferred_cluster_n_shift
    init_nt_m = gridx >> _preferred_cluster_m_shift
    init_nt_n = gridy >> _preferred_cluster_n_shift
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
            init_raw_m = bidx >> _fallback_cluster_m_shift
            init_raw_n = bidy >> _fallback_cluster_n_shift
            init_nt_m = gridx >> _fallback_cluster_m_shift
            init_nt_n = gridy >> _fallback_cluster_n_shift
    swizzle_w = _auto_swizzle_w(m, n, k, init_nt_n)
    init_tile_m, init_tile_n = _l2_swizzle_tile(
        init_raw_m,
        init_raw_n,
        init_nt_m,
        init_nt_n,
        swizzle_w,
        identity=tile_swizzle_n == 1,
    )
    init_tile_l = bidz

    if cutlass.const_expr(cta_group == 1):
        a_pattern = a_mcast_pattern
        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
            b_pattern = (1 << cluster_m) - 1
        else:
            b_pattern = (cutlass.Int32(1) << cluster_m) - 1

        if cutlass.const_expr(multicast_a):
            tma_mcast_mask_a = cutlass.Int16(a_pattern) << m_rank
        else:
            tma_mcast_mask_a = cutlass.Int16(1) << cta_rank_in_cluster
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
        else:
            tma_mcast_mask_b = cutlass.Int16(1) << cta_rank_in_cluster
    else:
        if cutlass.const_expr(multicast_a):
            tma_mcast_mask_a = cutlass.Int16(a_mcast_pattern << m_rank)
        else:
            tma_mcast_mask_a = cutlass.Int16(1 << cta_rank_in_cluster)
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16((b_mcast_pattern << pair_member) << (n_rank * cluster_m))
        else:
            tma_mcast_mask_b = cutlass.Int16(1 << cta_rank_in_cluster)

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(cta_group == 2):
        tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    # CLC scheduler SMEM — 2-stage ring.
    _clc_response_raw = cutlass.Array(cutlass.Int128, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=16)
    clc_response_ptr_base = cute.make_ptr(
        cutlass.Int128,
        _clc_response_raw.data_ptr(),
        mem_space=cute.AddressSpace.smem,
    )
    clc_full_mbar_ptr = cutlass.Array(cutlass.Int64, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    clc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    clc_full_mbar_cute_base = cute.make_ptr(
        cutlass.Int64,
        clc_full_mbar_ptr.data_ptr(),
        mem_space=cute.AddressSpace.smem,
    )

    sA_elems = cta_tile_mnk[0] * cta_tile_mnk[2]
    sB_elems = cta_tile_mnk[1] * cta_tile_mnk[2]
    smem_a_list = [
        cutlass.Array(
            ab_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            ab_dtype,
            sB_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    # One epilogue subtile = one MMA-M block x 32 cols; the M blocks reuse it.
    # The ring slot is indexed by `tidx`, so its row count is the EPILOGUE THREAD
    # count -- which is epi_tile_mn[0] only when the MMA M block is 128.
    epi_subtile_elems = epi_stage_rows * epi_row_elems * epi_slot_widen
    smem_d_ptr = cutlass.Array(
        cd_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    acc_empty_count = num_epilogue_warps * cta_group
    if cutlass.const_expr(ab_empty_full_mask):
        if cutlass.const_expr(cta_group == 1):
            ab_empty_count = cluster_size
        else:
            ab_empty_count = cluster_size // cta_group
    else:
        if cutlass.const_expr(cta_group == 1):
            ab_empty_count = cluster_m + cluster_n - 1
        else:
            ab_empty_count = (cluster_m // cta_group) + cluster_n - 1
    num_consumer_warps_per_cta = 7
    clc_empty_count = num_consumer_warps_per_cta * cluster_size
    if warp_idx == 0:
        if cutlass.const_expr(cta_group == 2):
            if elect_one:
                nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
        for i in range(acc_stages):
            if elect_one:
                nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), acc_empty_count)
        for i in range(CLC_SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(clc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(clc_empty_mbar_ptr.subview(i), clc_empty_count)
    nvvm.fence_mbarrier_init()
    if cutlass.const_expr(cta_group == 1):

        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            nvvm.barrier_cluster_arrive_relaxed()
            nvvm.barrier_cluster_wait()
        else:
            nvvm.barrier_cta_sync(0)
    else:
        nvvm.barrier_cluster_arrive_relaxed()

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    if cutlass.const_expr(cta_group == 1):
        num_tma_copy_bytes = num_a_operands * sA_bytes + num_b_operands * sB_bytes
    else:
        num_tma_copy_bytes = (num_a_operands * sA_bytes + num_b_operands * sB_bytes) * 2

    # One descriptor for every MMA instruction of the tile — the CTA tile spans
    # mma_size_m of them, all the same shape.
    idesc = cutlass.experimental.primitives.Tcgen05InstrDesc.build(
        a_dtype=mma_a_dtype,
        b_dtype=mma_b_dtype,
        c_dtype=mma_c_dtype,
        n_dim=mma_inst_shape_mnk[1],
        m_dim=mma_inst_shape_mnk[0],
        a_major=mma_a_major,
        b_major=mma_b_major,
    )

    # Per-CTA logical tile — the cluster cancels out, so these stay compile-time
    # constants even when the cluster shape is only known at runtime.
    logical_cta_tile_m = cgrp_tile_mnk[0] // cluster_shape_mnk[0]
    logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_shape_mnk[1]
    pair_n_size = logical_cta_tile_n
    # Per-CTA output rows one MMA-M block covers. A 2-CTA pair splits M, so this
    # is the per-CTA mma_inst_m — half the instruction's hardware M.
    epi_rows_per_mma_m = cta_tile_mnk[0] // mma_size_m
    # TMEM accumulator layout, per acc stage: gemm g, M block mi -> columns
    # [g*cols_per_acc_stage + mi*epi_cols_per_mma_m, +epi_cols_per_mma_m), all at
    # TMEM lane base 0. N is NOT split across instructions, so the epilogue drains
    # a whole M block as one contiguous span.
    if cutlass.const_expr(epi_dp22):
        # cluster-MMA m=128: the pair also splits N, so each CTA drains N/2.
        epi_cols_per_mma_m = pair_n_size // 2
    else:
        epi_cols_per_mma_m = pair_n_size
    cols_per_acc_stage = mma_size_m * epi_cols_per_mma_m
    acc_region_cols = num_gemms * cols_per_acc_stage
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32

    if cutlass.const_expr(cta_group == 2):
        nvvm.barrier_cluster_wait()
        nvvm.barrier_cta_sync(0)

    vsize = epi_chunk_elems

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    # The tile this cluster owns spans its OWN cluster shape; both shapes walk
    # the grid as the identity map (tile == blockIdx), so they tile the problem
    # identically and every output tile is still covered exactly once.
    cgrp_tile_m_cur = logical_cta_tile_m * cluster_m
    cgrp_tile_n_cur = logical_cta_tile_n * cluster_n
    num_k_blocks = cta_tile_mnk[2] // mma_inst_shape_mnk[2]

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        sched_iter = cutlass.Int32(0)
        clc_empty_phase = cutlass.Int32(1)
        clc_full_phase = cutlass.Int32(0)
        is_valid_sched = cutlass.Int32(1)
        while is_valid_sched != 0:
            stage = sched_iter % CLC_SCHED_STAGES
            if stage == 0 and sched_iter != 0:
                clc_empty_phase = clc_empty_phase ^ 1
                clc_full_phase = clc_full_phase ^ 1

            if is_cluster_leader_cta:
                while not nvvm.mbarrier_try_wait_parity(clc_empty_mbar_ptr.subview(stage), clc_empty_phase, time_limit=10_000_000):
                    pass

            if elect_one:
                nvvm.mbarrier_arrive_expect_tx(clc_full_mbar_ptr.subview(stage), 16)

            if is_cluster_leader_cta:
                if elect_one:
                    cute_clc.issue_clc_query(
                        clc_full_mbar_cute_base + stage,
                        clc_response_ptr_base + stage,
                        multicast=True,
                    )

            while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(stage), clc_full_phase, time_limit=10_000_000):
                pass

            _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid_sched = vld

            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)

            sched_iter += 1

        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            if is_cluster_leader_cta:
                for _ in range(CLC_SCHED_STAGES):
                    stage = sched_iter % CLC_SCHED_STAGES
                    if stage == 0 and sched_iter != 0:
                        clc_empty_phase = clc_empty_phase ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        clc_empty_mbar_ptr.subview(stage),
                        clc_empty_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    sched_iter += 1

    if warp_idx == tma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ab_empty_phase_bit = cutlass.Int32(1)
        ab_iter = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        tile_h, tile_b = _decode_bh(tile_l, n_head)
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_tma = cutlass.Int32(0)
        # B's THD descriptor is CLAMPED to the packed total `cu_*[B]`, so the
        # last k tile of the last sequence reads the caller's unwritten capacity
        # tail as TMA zeros instead of live memory.  The GridConstant descriptor
        # is built at the buffer's CAPACITY (a declared `max_total_seq_len` is a
        # maximum, while the row that must read zero moves every step), so using
        # it here would multiply A's padding zeros by whatever the caller left
        # past `cu_*[B]` -- and `0 * NaN` is NaN.
        if cutlass.const_expr(_THD_MM):
            _thd_acquire_descs(desc_words, n_batch)
        while is_valid != 0:
            coord_m_per_cta = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            if cutlass.const_expr(cta_group == 1):
                coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * cta_tile_mnk[1]
            else:
                coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]
            # Broadcast operands sit at (h, b) = (0, 0); batched ones carry the
            # decoded pair.  Same const_expr shape as upstream, one coord wider.
            if cutlass.const_expr(matmul_a_batch == 1):
                tile_h_a = cutlass.Int32(0)
                tile_b_a = cutlass.Int32(0)
            else:
                tile_h_a = tile_h
                tile_b_a = tile_b
            if cutlass.const_expr(matmul_b_batch == 1):
                tile_h_b = cutlass.Int32(0)
                tile_b_b = cutlass.Int32(0)
            else:
                tile_h_b = tile_h
                tile_b_b = tile_b

            _a_k_off, _a_m_off, _b_k_off, _nkt = _thd_group(meta_t, tile_b, n_batch, num_k_tiles)
            if cutlass.const_expr(_THD_MM):
                # Packed operands have ONE batch element; the sequence is
                # reached by the coordinate offsets above, not by this axis.
                tile_b_a = cutlass.Int32(0)
                tile_b_b = cutlass.Int32(0)
            k_begin, k_end = _causal_k_range(tile_m * cgrp_tile_m_cur, _nkt)
            for k_tile_idx in range(k_begin, k_end):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                    pass

                coord_k = k_tile_idx * cta_tile_mnk[2]
                # A rides the blocked workspace, B the packed tokens, so the two
                # ragged bases differ and the k coordinate cannot be shared.
                coord_k_a = coord_k + _a_k_off
                coord_k_b = coord_k + _b_k_off
                coord_m_a = coord_m_per_cta + _a_m_off
                if is_pair_leader:
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                for _ai in cutlass.range_constexpr(num_a_operands):
                    sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                    tma_a_desc = tma_a_descs[_ai]
                    if cutlass.const_expr(a_mcast_slices > 1):
                        _a_rows = cta_tile_mnk[0] // a_mcast_slices
                        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_stage.subview(n_rank * _a_rows * cta_tile_mnk[2]),
                                    tma_a_desc.get_ptr(),
                                    (coord_k_a, coord_m_a + n_rank * _a_rows, tile_h_a, tile_b_a),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=_CTA_GROUP,
                                )
                        else:
                            _a_per_cta = a_mcast_slices >> _preferred_cluster_n_shift
                            if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                                _a_per_cta = a_mcast_slices >> _fallback_cluster_n_shift
                            for _asl in cutlass.range(_a_per_cta):
                                _a_idx = n_rank * _a_per_cta + _asl
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(_a_idx * _a_rows * cta_tile_mnk[2]),
                                        tma_a_desc.get_ptr(),
                                        (coord_k_a, coord_m_a + _a_idx * _a_rows, tile_h_a, tile_b_a),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=_CTA_GROUP,
                                    )
                    elif cutlass.const_expr(multicast_a):
                        if n_rank == 0:
                            if cutlass.const_expr(a_is_m_major):
                                for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(m_group * a_tma_group_elems * cta_tile_mnk[2]),
                                            tma_a_desc.get_ptr(),
                                            (
                                                coord_m_a + m_group * a_tma_group_elems,
                                                coord_k_a,
                                                tile_h_a,
                                                tile_b_a,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=_CTA_GROUP,
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage,
                                        tma_a_desc.get_ptr(),
                                        (coord_k_a, coord_m_a, tile_h_a, tile_b_a),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=_CTA_GROUP,
                                    )
                    else:
                        if cutlass.const_expr(a_is_m_major):
                            for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(m_group * a_tma_group_elems * cta_tile_mnk[2]),
                                        tma_a_desc.get_ptr(),
                                        (
                                            coord_m_a + m_group * a_tma_group_elems,
                                            coord_k_a,
                                            tile_h_a,
                                            tile_b_a,
                                        ),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=_CTA_GROUP,
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_stage,
                                    tma_a_desc.get_ptr(),
                                    (coord_k_a, coord_m_a, tile_h_a, tile_b_a),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=_CTA_GROUP,
                                )

                for _bj in cutlass.range_constexpr(num_b_operands):
                    sB_stage = smem_b_list[_bj].subview(sB_elems * stage)
                    tma_b_desc = tma_b_descs[_bj]
                    # See the note above the persistent loop: THD substitutes
                    # the packed-total-clamped descriptor for the capacity-sized
                    # GridConstant one.  Dense folds back to `.get_ptr()`.
                    _b_desc_ptr = _thd_desc_ptr(desc_words, B_CLAMP_SLOT(n_batch)) if cutlass.const_expr(_THD_MM) else tma_b_desc.get_ptr()
                    if cutlass.const_expr(b_mcast_slices > 1):
                        _b_rows = cta_tile_mnk[1] // b_mcast_slices
                        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_stage.subview(pair_m_idx * _b_rows * cta_tile_mnk[2]),
                                    _b_desc_ptr,
                                    (coord_k_b, coord_n_per_cta + pair_m_idx * _b_rows, tile_h_b, tile_b_b),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=_CTA_GROUP,
                                )
                        else:
                            _b_per_cta = b_mcast_slices >> (_preferred_cluster_m_shift - _cta_group_shift)
                            if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                                _b_per_cta = b_mcast_slices >> (_fallback_cluster_m_shift - _cta_group_shift)
                            for _bsl in cutlass.range(_b_per_cta):
                                _b_idx = pair_m_idx * _b_per_cta + _bsl
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(_b_idx * _b_rows * cta_tile_mnk[2]),
                                        _b_desc_ptr,
                                        (coord_k_b, coord_n_per_cta + _b_idx * _b_rows, tile_h_b, tile_b_b),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=_CTA_GROUP,
                                    )
                    elif cutlass.const_expr(multicast_b):
                        if pair_m_idx == 0:
                            if cutlass.const_expr(b_is_n_major):
                                for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage.subview(n_group * b_tma_group_elems * cta_tile_mnk[2]),
                                            _b_desc_ptr,
                                            (
                                                coord_n_per_cta + n_group * b_tma_group_elems,
                                                coord_k_b,
                                                tile_h_b,
                                                tile_b_b,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=_CTA_GROUP,
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage,
                                        _b_desc_ptr,
                                        (coord_k_b, coord_n_per_cta, tile_h_b, tile_b_b),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=_CTA_GROUP,
                                    )
                    else:
                        if cutlass.const_expr(b_is_n_major):
                            for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(n_group * b_tma_group_elems * cta_tile_mnk[2]),
                                        _b_desc_ptr,
                                        (
                                            coord_n_per_cta + n_group * b_tma_group_elems,
                                            coord_k_b,
                                            tile_h_b,
                                            tile_b_b,
                                        ),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=_CTA_GROUP,
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_stage,
                                    _b_desc_ptr,
                                    (coord_k_b, coord_n_per_cta, tile_h_b, tile_b_b),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=_CTA_GROUP,
                                )

                ab_iter += 1

            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_tma = clc_full_phase_tma ^ 1
            while not nvvm.mbarrier_try_wait_parity(
                clc_full_mbar_ptr.subview(consumer_stage),
                clc_full_phase_tma,
                time_limit=10_000_000,
            ):
                pass
            m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid = vld
            tma_raw_m = m_idx >> _preferred_cluster_m_shift
            tma_raw_n = n_idx >> _preferred_cluster_n_shift
            tma_nt_m = gridx >> _preferred_cluster_m_shift
            tma_nt_n = gridy >> _preferred_cluster_n_shift
            if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                    tma_raw_m = m_idx >> _fallback_cluster_m_shift
                    tma_raw_n = n_idx >> _fallback_cluster_n_shift
                    tma_nt_m = gridx >> _fallback_cluster_m_shift
                    tma_nt_n = gridy >> _fallback_cluster_n_shift
            tile_m, tile_n = _l2_swizzle_tile(
                tma_raw_m,
                tma_raw_n,
                tma_nt_m,
                tma_nt_n,
                swizzle_w,
                identity=tile_swizzle_n == 1,
            )
            tile_l = l_idx
            tile_h, tile_b = _decode_bh(tile_l, n_head)
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            tile_iter += 1

        tail_stage = ab_iter % ab_stages
        tail_phase = ab_empty_phase_bit
        if tail_stage == 0 and ab_iter != 0:
            tail_phase = tail_phase ^ 1
        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            for _ in range(ab_stages):
                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
                    pass
                tail_stage = tail_stage + 1
                if tail_stage == ab_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1

    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        b_arrive_pattern = (1 << cluster_m) - 1
    else:
        b_arrive_pattern = (cutlass.Int32(1) << cluster_m) - 1
    a_part = a_mcast_pattern << m_rank
    if cutlass.const_expr(cta_group == 2):
        a_part = a_part | (a_part << 1)
    b_part = b_arrive_pattern << (n_rank * cluster_m)
    if cutlass.const_expr(ab_empty_full_mask):
        ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
    else:
        ab_empty_arrive_mask = cutlass.Int16(a_part | b_part)
    if cutlass.const_expr(cta_group == 2):
        acc_full_mcast = cutlass.Int16(3) << pair_leader_rank
    else:
        acc_full_mcast = None
    if warp_idx == mma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        _tcgen05_alloc(
            tmem_ptr_i32,
            cutlass.Int32(num_tmem_alloc_cols),
            is_exclusive=tmem_alloc_exclusive,
            group=_CTA_GROUP,
        )
        nvvm.bar_warp_sync(0xFFFFFFFF)
        nvvm.barrier_cta_arrive(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        if cutlass.const_expr(cta_group == 2):
            peer_cta_rank = cta_rank_in_cluster ^ 1

        if is_pair_leader:
            ab_full_phase_bit = cutlass.Int32(0)
            ab_iter = cutlass.Int32(0)
            acc_empty_phase_bit = cutlass.Int32(1)
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            clc_full_phase_mma = cutlass.Int32(0)
            acc_stage = cutlass.Int32(0)
            # The MMA warp tracks its own tile_m now: the causal K range depends on
            # it, and this arm must walk exactly the range the TMA warp walks.
            # THD's per-group k count needs the SEQUENCE for the same reason, so
            # the batch index is tracked here too.
            tile_m = init_tile_m
            _, tile_b_mma = _decode_bh(init_tile_l, n_head)
            # Descriptor metadata and the SMEM allocation base are invariant
            # across persistent tiles.  Only the encoded start address advances.
            desc_a_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_a_list[i],
                    leading_byte_offset=a_smem_desc_leading_byte_offset,
                    stride_byte_offset=a_smem_desc_stride_byte_offset,
                    layout=ab_smem_swizzle,
                )
                for i in range(num_a_operands)
            ]
            desc_b_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_b_list[j],
                    leading_byte_offset=b_smem_desc_leading_byte_offset,
                    stride_byte_offset=b_smem_desc_stride_byte_offset,
                    layout=ab_smem_swizzle,
                )
                for j in range(num_b_operands)
            ]
            while is_valid != 0:
                acc_stage = tile_iter % acc_stages
                if acc_stage == 0 and tile_iter != 0:
                    acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(acc_stage),
                    acc_empty_phase_bit,
                    time_limit=10_000_000,
                ):
                    pass

                acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                # One accumulator per (gemm, M block). Column arithmetic stays on
                # the encoded (row << 16) | col integer.
                tmem_addr_mmas = [
                    [
                        cutlass.inttoptr(
                            (base_row_id << 16) | (acc_base_col + g * cols_per_acc_stage + mi * epi_cols_per_mma_m),
                            6,
                            cutlass.Int32,
                        )
                        for mi in range(mma_size_m)
                    ]
                    for g in range(num_gemms)
                ]

                # Same range as the TMA producer above, or the ab ring
                # desynchronises.  scale_d starts False here, so the accumulator
                # is overwritten on the first k-block wherever the range begins.
                # Under THD that means the same PER-GROUP count too: a consumer
                # still counting the kernel-wide tiles would wait for k-blocks
                # the producer never issues.
                _, _, _, _nkt_mma = _thd_group(meta_t, tile_b_mma, n_batch, num_k_tiles)
                k_begin, k_end = _causal_k_range(tile_m * cgrp_tile_m_cur, _nkt_mma)
                scale_d = cutlass.Boolean(False)
                for k_tile_idx in range(k_begin, k_end):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_full_phase_bit = ab_full_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        ab_full_mbar_ptr.subview(stage),
                        ab_full_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass

                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        for g in cutlass.range_constexpr(num_gemms):
                            desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * stage + a_smem_k_step_bytes * k_block_idx)
                            desc_b = desc_b_roots[gemm_b_idx[g]].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                            for mi in cutlass.range_constexpr(mma_size_m):
                                # The M sub-block offset is a whole SMEM swizzle atom
                                # (mma_inst_m x cta_tile_k_bytes), so the descriptor's
                                # swizzle phase is preserved. B is shared by every M block.
                                desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                                if elect_one:
                                    _tcgen05_mma(
                                        mma_kind,
                                        _CTA_GROUP,
                                        tmem_addr_mmas[g][mi],
                                        desc_a,
                                        desc_b,
                                        idesc,
                                        scale_d,
                                        collector_op=_a_collector_op(g),
                                        b_collector_op=_b_collector_op(mi),
                                    )
                        # Every accumulator sees scale_d=False on exactly the first
                        # k_block of the tile, so the flip stays outside mi.
                        scale_d = cutlass.Boolean(True)

                    if elect_one:
                        nvvm.tcgen05_commit(
                            ab_empty_mbar_ptr.subview(stage),
                            multicast_mask=ab_empty_arrive_mask,
                            group=_CTA_GROUP,
                        )
                    ab_iter += 1

                if elect_one:
                    nvvm.tcgen05_commit(
                        acc_full_mbar_ptr.subview(acc_stage),
                        multicast_mask=acc_full_mcast,
                        group=_CTA_GROUP,
                    )

                consumer_stage = tile_iter % CLC_SCHED_STAGES
                if consumer_stage == 0 and tile_iter != 0:
                    clc_full_phase_mma = clc_full_phase_mma ^ 1
                while not nvvm.mbarrier_try_wait_parity(
                    clc_full_mbar_ptr.subview(consumer_stage),
                    clc_full_phase_mma,
                    time_limit=10_000_000,
                ):
                    pass
                _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                mma_raw_m = _m_idx >> _preferred_cluster_m_shift
                mma_raw_n = _n_idx >> _preferred_cluster_n_shift
                mma_nt_m = gridx >> _preferred_cluster_m_shift
                mma_nt_n = gridy >> _preferred_cluster_n_shift
                if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                    if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                        mma_raw_m = _m_idx >> _fallback_cluster_m_shift
                        mma_raw_n = _n_idx >> _fallback_cluster_n_shift
                        mma_nt_m = gridx >> _fallback_cluster_m_shift
                        mma_nt_n = gridy >> _fallback_cluster_n_shift
                tile_m, _tile_n_mma = _l2_swizzle_tile(mma_raw_m, mma_raw_n, mma_nt_m, mma_nt_n, swizzle_w, identity=tile_swizzle_n == 1)
                _, tile_b_mma = _decode_bh(_l_idx, n_head)
                cute.arch.fence_proxy("async.shared", space="cta")
                is_valid = vld
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                    nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                tile_iter += 1

            if cutlass.const_expr(USE_PDL):
                nvvm.griddepcontrol("launch_dependents")

            tail_stage = acc_stage
            tail_phase = acc_empty_phase_bit
            for _ in range(acc_stages):
                tail_stage = tail_stage + 1
                if tail_stage == acc_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1
                while not nvvm.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(tail_stage),
                    tail_phase,
                    time_limit=10_000_000,
                ):
                    pass
            nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
            if cutlass.const_expr(cta_group == 2):
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            _tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=_CTA_GROUP,
            )
        else:
            if cutlass.const_expr(cta_group == 2):
                tile_iter = cutlass.Int32(0)
                is_valid = cutlass.Int32(1)
                clc_full_phase_mma = cutlass.Int32(0)
                while is_valid != 0:
                    consumer_stage = tile_iter % CLC_SCHED_STAGES
                    if consumer_stage == 0 and tile_iter != 0:
                        clc_full_phase_mma = clc_full_phase_mma ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        clc_full_mbar_ptr.subview(consumer_stage),
                        clc_full_phase_mma,
                        time_limit=10_000_000,
                    ):
                        pass
                    _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    is_valid = vld
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                        nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                    tile_iter += 1
                if cutlass.const_expr(USE_PDL):
                    nvvm.griddepcontrol("launch_dependents")
                nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
                _tcgen05_dealloc(
                    alloc_ptr,
                    cutlass.Int32(num_tmem_alloc_cols),
                    is_exclusive=tmem_alloc_exclusive,
                    group=_CTA_GROUP,
                )

    if warp_idx < num_epilogue_warps:
        nvvm.setmaxregister(epi_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        nvvm.barrier_cta_sync(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16

        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")

        tile_iter = cutlass.Int32(0)
        acc_full_phase_bit = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        tile_h, tile_b = _decode_bh(tile_l, n_head)
        if cutlass.const_expr(_THD_MM):
            # Once per epilogue warp, not per store: the patch launch wrote the
            # array before this kernel started and never rewrites it.  But it is
            # one acquire PER SLOT -- the fence's size operand only accepts 128,
            # i.e. a single descriptor, so acquiring the base would order slot 0
            # alone while this warp goes on to select slot `tile_b`.
            _thd_acquire_descs(desc_words, n_batch)
        # The C-side cu_seqlens prefix inside the metadata buffer: dV/dK write
        # kv rows, dQ writes q rows -- the same choice `_thd_patch_descs_kernel`
        # makes when it bases each sequence's descriptor.
        _thd_meta = cutlass.make_array_view(meta_t) if cutlass.const_expr(_THD_MM) else None
        _thd_c_cu0 = (
            ((cutlass.Int32(2) * n_batch + cutlass.Int32(1)) if cutlass.const_expr(a_is_m_major) else n_batch)
            if cutlass.const_expr(_THD_MM)
            else cutlass.Int32(0)
        )
        # The A-side (K) prefix is the OTHER one -- `_thd_group` reduces over S_q
        # for the m-major dV/dK GEMMs and over S_kv for the k-major dQ one,
        # exactly opposite to which axis each writes.  Used only to detect a
        # zero-length reduction; see `_thd_k_len` in the loop.
        _thd_k_cu0 = (
            (n_batch if cutlass.const_expr(a_is_m_major) else (cutlass.Int32(2) * n_batch + cutlass.Int32(1)))
            if cutlass.const_expr(_THD_MM)
            else cutlass.Int32(0)
        )
        is_valid = cutlass.Int32(1)
        clc_full_phase_epi = cutlass.Int32(0)

        if cutlass.const_expr(epi_packed_lanes):
            row_id_with_warp_offset = base_row_id
        else:
            row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        if cutlass.const_expr(epi_packed_lanes):
            shape = nvvm.Tcgen05LdStShape.SHAPE_16X32BX2
            ld_half_off = 0
        else:
            shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
            ld_half_off = None
        lane = tidx % 32

        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)

        while is_valid != 0:
            coord_m_tile = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            coord_n_c = tile_n * cgrp_tile_n_cur + n_rank * pair_n_size
            # A group whose OUTPUT sequence has zero rows. Its clipped C
            # descriptor is built at extent 1 rather than 0 (a tensor map with a
            # zero extent is INVALID and traps -- see tile_dsl.thd.emit_seq_descs),
            # so the hardware clip that drops every other overshooting tile
            # cannot drop this one: skip the store instead. Read once per tile,
            # not per subtile; `tile_b` is refreshed at the bottom of the loop.
            _thd_c_len = (
                cutlass.Int32(_thd_meta[_thd_c_cu0 + tile_b + cutlass.Int32(1)]) - cutlass.Int32(_thd_meta[_thd_c_cu0 + tile_b])
                if cutlass.const_expr(_THD_MM)
                else cutlass.Int32(1)
            )
            # A group whose REDUCTION axis is empty: S_q[b] == 0 for dV/dK,
            # S_kv[b] == 0 for dQ.  `_thd_group` then returns `nkt == 0`, the
            # mainloop runs zero iterations, and `scale_d` starts False -- so no
            # MMA ever wrote the accumulator and the TMEM read below returns
            # whatever the previous tile left there.
            #
            # Unlike `_thd_c_len` this canNOT be fixed by skipping the store:
            # the OUTPUT rows exist (`_thd_c_len > 0`) and the caller expects
            # them written.  Store ZEROS instead, which is also the right
            # answer -- a sequence with no queries contributes nothing to its
            # keys' and values' gradients, and one with no keys has no gradient
            # to receive.
            #
            # A select, not a multiply by zero: the uninitialised TMEM can hold
            # any bit pattern, and `0 * NaN` is NaN (issue #624's rule).
            _thd_k_len = (
                cutlass.Int32(_thd_meta[_thd_k_cu0 + tile_b + cutlass.Int32(1)]) - cutlass.Int32(_thd_meta[_thd_k_cu0 + tile_b])
                if cutlass.const_expr(_THD_MM)
                else cutlass.Int32(1)
            )
            if cutlass.const_expr(epi_dp22):
                coord_n_c = coord_n_c + (warp_idx // 2) * epi_cols_per_mma_m

            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(acc_stage), acc_full_phase_bit, time_limit=10_000_000):
                pass

            acc_base_col = base_col_id_root + acc_stage * acc_region_cols

            for mi in cutlass.range_constexpr(mma_size_m):
                coord_m = coord_m_tile + mi * epi_rows_per_mma_m
                mi_col_base = acc_base_col + mi * epi_cols_per_mma_m
                tmem_col_addr_gemms = [(row_id_with_warp_offset << 16) | (mi_col_base + g * cols_per_acc_stage) for g in range(num_gemms)]

                if cutlass.const_expr(epi_packed_lanes):
                    row = coord_m + warp_idx * 16 + lane
                    row_active = lane < 16
                elif cutlass.const_expr(epi_dp22):
                    row = coord_m + (warp_idx % 2) * 32 + lane
                    row_active = True
                else:
                    row = coord_m + tidx
                    row_active = True

                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                    subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                    c_rmem_vecs = []
                    for g in cutlass.range_constexpr(num_gemms):
                        subtile_tmem_addr = tmem_col_addr_gemms[g] + subtile_col_offset
                        tmem = cutlass.inttoptr(subtile_tmem_addr, 6, mma_c_dtype)
                        _cv = nvvm.tcgen05_ld(shape, tmem, num=subtile_w, offset=ld_half_off)
                        # INT8 int32 accumulate → widen to fp32 (skipped for int32 output).
                        if cutlass.const_expr(acc_widen_to_fp32):
                            _accf = _cv.to(cutlass.Float32)
                            # `+ 0.0` forces a fresh fp32 register so int32->fp32 isn't folded into an invalid int32->fp8 cast.
                            _cv = _accf + cutlass.full_like(_accf, 0.0)
                        c_rmem_vecs.append(_cv)
                    c_rmem_vec = c_rmem_vecs[0]

                    if mi == mma_size_m - 1 and subtile_idx == subtile_cnt - 1:
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if elect_one:
                            if cutlass.const_expr(cta_group == 2):
                                nvvm.mbarrier_arrive(
                                    nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                    scope=nvvm.MemScope.CLUSTER,
                                    relaxed=True,
                                )
                            else:
                                nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                    col = coord_n_c + subtile_col_offset

                    vec_f32 = c_rmem_vec
                    col_j = col
                    linear_idx = tile_b * out_stride_b_0 + tile_h * out_stride_h_0 + row * out_stride_m_0 + col_j * out_stride_n_0

                    _r_mm = (vec_f32).to(cd_dtype)
                    vec_out = (_r_mm).to(cd_dtype)

                    epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                    _tsv_0 = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape=8192, dtype=cd_dtype)
                    # The branch is CTA-uniform (it reads only `tile_b`) and it
                    # wraps the store ALONE -- the fence and the named barrier
                    # below stay outside it, so no path through here can diverge
                    # on a sync.
                    if cutlass.const_expr(_THD_MM):
                        if _thd_k_len > cutlass.Int32(0):
                            _tsv_0.data_ptr(tidx * 64).store_swizzled(vec_out, alignment=128, swizzle=cutlass.Swizzle(3, 4, 3))
                        else:
                            _tsv_0.data_ptr(tidx * 64).store_swizzled(cutlass.full_like(vec_out, 0.0), alignment=128, swizzle=cutlass.Swizzle(3, 4, 3))
                    else:
                        _tsv_0.data_ptr(tidx * 64).store_swizzled(vec_out, alignment=128, swizzle=cutlass.Swizzle(3, 4, 3))
                    cute.arch.fence_view_async_shared()
                    nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
                    if warp_idx == 0:
                        if elect_one:
                            # THD stores through THIS SEQUENCE's descriptor: its
                            # GLOBAL_ADDRESS is the sequence's first output row
                            # and its GLOBAL_DIM[seq] is the sequence's length,
                            # so the last M tile -- which overshoots into the
                            # next sequence's rows with a live accumulator
                            # behind it -- is clipped by hardware.  The batch
                            # coordinate is then 0: the descriptor already
                            # carries the base, so the M coordinate stays
                            # sequence-relative.
                            if cutlass.const_expr(_THD_MM):
                                # Skipping only the STORE keeps the epilogue's
                                # pipeline intact: the commit below still runs,
                                # and an empty bulk group commits immediately.
                                if _thd_c_len > cutlass.Int32(0):
                                    nvvm.cp_async_bulk_tensor_global_shared_cta(
                                        (desc_words.iterator.raw_ptr() + tile_b * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic),
                                        _tsv_0.data_ptr(),
                                        (col, coord_m, tile_h, cutlass.Int32(0)),
                                    )
                            else:
                                nvvm.cp_async_bulk_tensor_global_shared_cta(
                                    tma_c_descs[0].get_ptr(),
                                    _tsv_0.data_ptr(),
                                    (col, coord_m, tile_h, tile_b),
                                )
                        if elect_one:
                            nvvm.cp_async_bulk_commit_group()
                        nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)
                    nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)

            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_epi = clc_full_phase_epi ^ 1
            while not nvvm.mbarrier_try_wait_parity(
                clc_full_mbar_ptr.subview(consumer_stage),
                clc_full_phase_epi,
                time_limit=10_000_000,
            ):
                pass
            m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid = vld
            epi_raw_m = m_idx >> _preferred_cluster_m_shift
            epi_raw_n = n_idx >> _preferred_cluster_n_shift
            epi_nt_m = gridx >> _preferred_cluster_m_shift
            epi_nt_n = gridy >> _preferred_cluster_n_shift
            if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                    epi_raw_m = m_idx >> _fallback_cluster_m_shift
                    epi_raw_n = n_idx >> _fallback_cluster_n_shift
                    epi_nt_m = gridx >> _fallback_cluster_m_shift
                    epi_nt_n = gridy >> _fallback_cluster_n_shift
            tile_m, tile_n = _l2_swizzle_tile(
                epi_raw_m,
                epi_raw_n,
                epi_nt_m,
                epi_nt_n,
                swizzle_w,
                identity=tile_swizzle_n == 1,
            )
            tile_l = l_idx
            tile_h, tile_b = _decode_bh(tile_l, n_head)
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)

            tile_iter += 1

        if warp_idx == 0:
            nvvm.cp_async_bulk_wait_group(0, read=True)

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)


@cute.kernel
def _thd_patch_descs_kernel(
    c_tensor: cute.Tensor,
    base_c_desc: cutlass.GridConstant[_tma.TensorMap],
    base_b_desc: cutlass.GridConstant[_tma.TensorMap],
    desc_words: cute.Tensor,
    meta_t: cute.Tensor,
    n_batch: cutlass.Int32,
    c_row_stride: cutlass.Int64,
) -> None:
    """This GEMM's own THD descriptors, patched from the published metadata.

    Built HERE and not in the shared setup launch because a descriptor's box,
    swizzle and dim ORDER are this kernel's: its operands are ``(n, m, h, b)``,
    so the sequence axis is ``ord=1`` -- stage 2's packed ``(d, head, seq,
    batch)`` operands put it at 2, and a shared builder told the wrong number
    clamps the head extent instead, silently.

    * one C descriptor per sequence, based at that sequence's first output row
      with ``GLOBAL_DIM[seq]`` set to its length, so the overshooting last M
      tile is clipped rather than writing into the next sequence;
    * one B descriptor clamped to the CURRENT packed total, so the last k tile
      reads the caller's unwritten capacity tail as exact zeros.  A declared
      ``max_total_seq_len`` cannot do this job: it is a maximum, while the row
      that must read zero is ``cu_*[B]``, which changes every step.

    One elected thread; the release fence publishes both to the TMA proxy and
    the kernel boundary orders them before the GEMM reads them.
    """
    tidx, _, _ = cute.arch.thread_idx()
    if nvvm.elect_sync() and tidx < cutlass.Int32(32):
        meta = cutlass.make_array_view(meta_t)
        cu_q0 = n_batch
        cu_k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
        # dV/dK write kv rows and read q tokens; dQ is the mirror.
        c_cu0 = cu_k0 if cutlass.const_expr(a_is_m_major) else cu_q0
        b_cu0 = cu_q0 if cutlass.const_expr(a_is_m_major) else cu_k0
        emit_seq_descs(base_c_desc, desc_words, meta, c_cu0, c_tensor, n_batch, c_row_stride, seq_ord=_THD_MM_SEQ_ORD)
        b_total = cutlass.Int32(meta[b_cu0 + n_batch])
        emit_clamped_desc(base_b_desc, desc_words, n_batch, b_total, seq_ord=_THD_MM_SEQ_ORD)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )


_thd_patch_descs_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    a_0: cute.Tensor,
    b_0: cute.Tensor,
    c_0: cute.Tensor,
    # Dense: 1-element dummies.  THD: the setup launch's metadata and this
    # GEMM's descriptor scratch.
    meta_t: cute.Tensor,
    desc_words: cute.Tensor,
    stream: _cuda.CUstream,
) -> None:
    _a_operands = [a_0]
    _b_operands = [b_0]
    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    # The 2-D batch: (n_head, n_batch) where upstream carries one flat `batch`,
    # and FOUR strides per operand (m/n, k, h, b) where upstream carries three.
    # `batch` stays as the product because the CLC scheduler and the grid still
    # rasterize one flat axis; only the TMA coordinates are 2-D.
    n_head = problem_size[3]
    n_batch = problem_size[4]
    batch = n_head * n_batch
    _stride_idx = 5
    _a_stride_sets = []
    for _ in cutlass.range_constexpr(num_a_operands):
        _a_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
                problem_size[_stride_idx + 3],
            )
        )
        _stride_idx += 4
    _b_stride_sets = []
    for _ in cutlass.range_constexpr(num_b_operands):
        _b_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
                problem_size[_stride_idx + 3],
            )
        )
        _stride_idx += 4
    out_stride_m_0 = problem_size[_stride_idx]
    out_stride_n_0 = problem_size[_stride_idx + 1]
    out_stride_h_0 = problem_size[_stride_idx + 2]
    out_stride_b_0 = problem_size[_stride_idx + 3]
    _stride_idx += 4
    # B's K extent, separate from A's.  They coincide on the dense path, but
    # under THD A's K axis is the BLOCKED workspace (rows padded per sequence)
    # while B's is the PACKED tokens -- different lengths for the same logical
    # reduction, so one shared symbol cannot describe both.
    k_b = problem_size[_stride_idx]
    # A's and C's M extents, likewise separate.  Dense passes all three equal;
    # THD does not: for dV/dK the A operand's M is the workspace's uniform kv
    # column count while C's is the PACKED output rows, and `m` itself is only
    # the grid's M -- the longest sequence, which every group's tiles cover and
    # a shorter one's spare tiles are clipped out of.
    m_a = problem_size[_stride_idx + 1]
    m_c = problem_size[_stride_idx + 2]
    _stride_idx += 3

    # A broadcast operand collapses BOTH batch extents, not just one.
    if cutlass.const_expr(matmul_a_batch == 1):
        a_h, a_b = 1, 1
    else:
        a_h, a_b = n_head, n_batch
    if cutlass.const_expr(matmul_b_batch == 1):
        b_h, b_b = 1, 1
    else:
        b_h, b_b = n_head, n_batch
    # THD: `n_batch` is the SEQUENCE count -- it sizes the grid and indexes the
    # metadata -- but the packed operands hold ONE batch element, reached by the
    # coordinate offsets instead.  Describing them as n_batch-deep builds a
    # tensor map over memory that is not there, which fails inside
    # cuTensorMapEncodeTiled as an abort rather than an exception.
    if cutlass.const_expr(_THD_MM):
        a_b = 1
        b_b = 1
    c_batch = 1 if cutlass.const_expr(_THD_MM) else n_batch

    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_h, a_stride_b = _a_stride_sets[_a_idx]
        if cutlass.const_expr(a_is_m_major):
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[m_a, k_sym, a_h, a_b],
                    global_strides=[
                        a_stride_k * ab_dtype.width // 128,
                        a_stride_h * ab_dtype.width // 128,
                        a_stride_b * ab_dtype.width // 128,
                    ],
                    box_dims=[a_tma_group_elems, cta_tile_mnk[2], 1, 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_sym, m_a, a_h, a_b],
                    global_strides=[
                        a_stride_m * ab_dtype.width // 128,
                        a_stride_h * ab_dtype.width // 128,
                        a_stride_b * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[0] // a_mcast_slices, 1, 1],
                    swizzle=ab_tma_swizzle,
                )
            )
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_h, b_stride_b = _b_stride_sets[_b_idx]
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[n, k_b, b_h, b_b],
                    global_strides=[
                        b_stride_k * ab_dtype.width // 128,
                        b_stride_h * ab_dtype.width // 128,
                        b_stride_b * ab_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1, 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_b, n, b_h, b_b],
                    global_strides=[
                        b_stride_n * ab_dtype.width // 128,
                        b_stride_h * ab_dtype.width // 128,
                        b_stride_b * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1, 1],
                    swizzle=ab_tma_swizzle,
                )
            )

    _tma_c_outputs = [c_0]
    _c0 = _tma_c_outputs[0]
    tma_c_desc_0 = _tma.create_tensor_map_tiled(
        global_address=_c0.iterator.toint(),
        dtype=cd_dtype,
        global_dims=[n, m_c, n_head, c_batch],
        global_strides=[
            # `cd_dtype.width`, not a literal 16: the A/B descriptors above already
            # derive it, and this one is now dtype-parameterized too.
            out_stride_m_0 * cd_dtype.width // 128,
            out_stride_h_0 * cd_dtype.width // 128,
            out_stride_b_0 * cd_dtype.width // 128,
        ],
        box_dims=[64, epi_tile_mn[0], 1, 1],
        swizzle=_tma.TensorMapSwizzle.s128b,
    )
    tma_c_desc_list = [tma_c_desc_0]

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cgrp_tile_m = cgrp_tile_mnk[0]
    cgrp_tile_n = cgrp_tile_mnk[1]
    num_tile_m_host = (m + cgrp_tile_m - 1) // cgrp_tile_m
    num_tile_n_host = (n + cgrp_tile_n - 1) // cgrp_tile_n
    grid_x = num_tile_m_host * cluster_m
    grid_y = num_tile_n_host * cluster_n
    grid_shape = (grid_x, grid_y, batch)
    if cutlass.const_expr(_THD_MM):
        # Ahead of the GEMM on the same stream; kernel-boundary ordering is what
        # makes the patched descriptors visible to it.
        _thd_patch_descs_kernel(
            _c0,
            tma_c_desc_0,
            tma_b_desc_list[0],
            desc_words,
            meta_t,
            n_batch,
            out_stride_m_0,
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    launch = _bprop_matmul_bh_sm100_kernel(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        tma_a_desc_list[0],
        tma_b_desc_list[0],
        out_stride_m_0,
        out_stride_n_0,
        out_stride_h_0,
        out_stride_b_0,
        n_head,
        tma_c_desc_list[0],
        meta_t,
        desc_words,
        n_batch,
    )
    # Mixed CGA: `cluster` is the preferred (wide) shape and `fallback_cluster`
    # the regular one the device groups blocks into when a preferred cluster does
    # not fit. The grid is already a multiple of the preferred shape, which the
    # driver requires.
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        launch.launch(
            grid=grid_shape,
            block=(threads_per_cta, 1, 1),
            cluster=cluster_shape_mnk,
            use_pdl=USE_PDL,
            stream=stream,
        )
    else:
        launch.launch(
            grid=grid_shape,
            block=(threads_per_cta, 1, 1),
            cluster=cluster_shape_mnk,
            fallback_cluster=fallback_cluster_shape_mnk,
            use_pdl=USE_PDL,
            stream=stream,
        )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    ab_stride_elems = 16 // (ab_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    # K tails are supported: the K loop is ceil_div and the TMA descriptor's global K
    # extent makes a partial box HW zero-filled. The only real K rule is the 16-byte
    # TMA contiguous-extent one, already gated by _tma_alignment_reject.
    sym_k = cute.sym_int64()
    # See `_host`: A's and B's K extents are the same number only on the dense
    # path, so they are separate symbols and the artifact serves both.
    sym_k_b = cute.sym_int64()
    sym_m_a = cute.sym_int64()
    sym_m_c = cute.sym_int64()
    # Two symbolic batch extents instead of one flat `sym_l`.
    sym_h = cute.sym_int64()
    sym_b = cute.sym_int64()
    if matmul_a_batch == 1:
        sym_a_h, sym_a_b = 1, 1
    else:
        sym_a_h, sym_a_b = sym_h, sym_b
    if matmul_b_batch == 1:
        sym_b_h, sym_b_b = 1, 1
    else:
        sym_b_h, sym_b_b = sym_h, sym_b

    def _make_fake_a():
        return make_fake_compact_tensor(
            mma_a_dtype,
            (sym_m_a, sym_k, sym_a_h, sym_a_b),
            stride_order=(0, 1, 2, 3) if a_is_m_major else (1, 0, 2, 3),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            mma_b_dtype,
            (sym_n, sym_k_b, sym_b_h, sym_b_b),
            stride_order=(0, 1, 2, 3) if b_is_n_major else (1, 0, 2, 3),
            assumed_align=16,
        )

    def _make_fake_c(_dt, _div, _mm):
        return make_fake_compact_tensor(
            _dt,
            (sym_m_c, sym_n // _div, sym_h, sym_b),
            stride_order=(0, 1, 2, 3) if _mm else (1, 0, 2, 3),
            assumed_align=16,
        )

    fake_c_0 = _make_fake_c(cd_dtype, 1, False)

    def _sym_operand_strides(is_mn_major: bool) -> tuple:
        # Operand is permuted to (M|N, K, H, B): the unit stride is mode 0 when
        # MN-major, mode 1 when K-major, and never reaches TMA.  Four modes now,
        # since the batch is the (h, b) pair.
        unit = 0 if is_mn_major else 1
        return tuple(cute.sym_int64() if i == unit else cute.sym_int64(divisibility=ab_stride_elems) for i in range(4))

    sym_a_strides = []
    for _ in range(num_a_operands):
        sym_a_strides.extend(_sym_operand_strides(a_is_m_major))
    sym_b_strides = []
    for _ in range(num_b_operands):
        sym_b_strides.extend(_sym_operand_strides(b_is_n_major))
    sym_out_stride_m_0 = cute.sym_int64()
    sym_out_stride_n_0 = cute.sym_int64()
    sym_out_stride_h_0 = cute.sym_int64()
    sym_out_stride_b_0 = cute.sym_int64()
    fake_a_0 = _make_fake_a()
    fake_b_0 = _make_fake_b()
    problem_size = (
        sym_m,
        sym_n,
        sym_k,
        sym_h,
        sym_b,
        *sym_a_strides,
        *sym_b_strides,
        sym_out_stride_m_0,
        sym_out_stride_n_0,
        sym_out_stride_h_0,
        sym_out_stride_b_0,
        sym_k_b,
        sym_m_a,
        sym_m_c,
    )
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    # Dense binds 1-element dummies; THD's real extents are runtime, so both are
    # symbolic and one artifact serves every sequence count.
    # Symbolic on BOTH paths: the dense caller passes whatever 1-D buffer it has
    # to hand (it is never read), and THD's extents are runtime, so one artifact
    # serves every sequence count either way.
    fake_meta = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int64(),), stride_order=(0,), assumed_align=16)
    fake_desc = make_fake_compact_tensor(cutlass.Int64, (cute.sym_int64(),), stride_order=(0,), assumed_align=16)
    return cute.compile(
        _host,
        problem_size,
        fake_a_0,
        fake_b_0,
        fake_c_0,
        fake_meta,
        fake_desc,
        stream=_fake_stream,
        options=frost_compile_options,
    )


# ---------------------------------------------------------------------------
# SDPA-backward stage 3: the three gradient GEMMs over a 2-D (batch, head) batch
# ---------------------------------------------------------------------------


def _permuted(t, mn_dim: int, k_dim: int, h_dim: int, b_dim: int):
    """Permute a caller tensor to the kernel's ``(M|N, K, H, B)`` operand order.

    A view, never a copy -- which is the point of the 4-D descriptor: the
    workspace is measured in GiB and a per-chunk normalising copy would cost
    more than the GEMM.  The strides ride to the kernel through
    ``problem_size``, so any layout the permutation can express is legal.
    """
    return t.permute(mn_dim, k_dim, h_dim, b_dim)


def matmul_bh(a, b, out, *, n_head: int, n_batch: int, stream=None, meta=None, desc_words=None, grid_m: int = None):
    """Run one ``(batch, head)``-batched GEMM.

    ``a`` / ``b`` / ``out`` are already permuted to ``(M|N, K, H, B)`` /
    ``(M, N, H, B)``.  M, N, K and every stride are runtime values, so one
    compiled artifact serves every stage-3 shape at a given dtype and layout.

    THD passes the setup launch's ``meta`` buffer and a ``desc_words`` scratch
    of ``THD_MM_DESC_SLOTS(n_batch)`` tensor maps, which this module patches
    itself ahead of the GEMM.  ``m`` is then the LONGEST sequence's extent: the
    grid covers it, and tiles past a shorter sequence's own length compute a
    garbage accumulator that its clipped C descriptor drops.
    """
    if meta is None or desc_words is None:
        # Required on both paths: dense never reads them, but the compiled ABI
        # has the slots and a None would fail at the call boundary.
        raise ValueError("matmul_bh needs `meta` and `desc_words` (dense may pass any 1-D dummies)")
    fn = compile()
    # `m` sizes the GRID.  Dense: the operands' shared M.  THD: the caller
    # passes the longest sequence, because the per-sequence extents are device
    # values and A's own M is the workspace's column count, not an output row
    # count.
    m = int(a.shape[0]) if grid_m is None else int(grid_m)
    k = int(a.shape[1])
    n = int(b.shape[0])
    problem_size = (
        m,
        n,
        k,
        n_head,
        n_batch,
        *(int(s) for s in a.stride()),
        *(int(s) for s in b.stride()),
        *(int(s) for s in out.stride()),
        int(b.shape[1]),
        int(a.shape[0]),
        int(out.shape[0]),
    )
    return fn(problem_size, a, b, out, meta, desc_words, stream=stream)
