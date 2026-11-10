# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
import os

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass._mlir.dialects import arith, llvm
from cutlass.cute.typing import Int32 as CuteInt32
from cutlass.cute.typing import Pointer as CutePointer
from cutlass.cutlass_dsl import T, dsl_user_op

from .block_scan import block_prefix_sum_kernel, fence_acq_rel_cta

# Dev-only IKET phase timing. Off by default so the baseline IR is untouched.
# Enable with `TOPK_IKET=1` AND compile with the iket option
# (`CUTE_DSL_COMPILER_OPT=iket`); then run under `run-iket ... profile`.
# When off, no iket ops are traced into the kernel IR at all.
_TOPK_IKET = os.environ.get("TOPK_IKET", "0") == "1"
if _TOPK_IKET:
    import cutlass.cute.experimental.iket as _iket


_LARGE_OCCUPANCY_MIN_BLOCKS_PER_MP = {
    (10, 0): 4,
    (10, 3): 4,
    (10, 7): 2,
    (10, 9): 2,
}
# 228 KiB SMEM / 108 KiB L1 tier in Rubin's 336 KiB unified pool.
_RUBIN_TOPK_ARCHITECTURES = ("sm_107", "sm_109")
# TODO: for tma_load=on and smem_cache_value=true,
# there're no L1 data flows. So we could increase SMEM carveout to the max SMEM size
# and make the candidate buffer larger.
_RUBIN_TOPK_SMEM_CARVEOUT_BYTES = 228 * 1024
_SMEM_ALIGNMENT_BYTES = 128
_SMEM_RUNTIME_RESERVE_BYTES = 1024

# --- async-TMA load (p1 coarse histogram + p3 coarse filter) enable knob ---
#
# Routing the input scans through async-TMA (cp.async.bulk / UTMALDG) staging is a
# tuning knob (all of fp32/fp16/bf16 are correct with it on), shared by the prefill
# and decode large-occupancy wrappers. From an on-vs-off sweep (Rubin sm_107):
#   fp32       : load-bound at large N -> WINS for N >= _TMA_MIN_NUM_COLS (-13..-15%).
#   fp16/bf16  : NOT load-bound (2-byte L1-cached re-reads) -> loses across ~all N.
# So the tuned-on set is "fp32 on Rubin with N >= _TMA_MIN_NUM_COLS". The enable also
# requires a 16-byte-aligned num_cols and non-REREAD_ALWAYS (resolved in _prepare).
# Override per call via enable_tma_load(_p3) params / --enable_tma_load {auto,on,off}.
# fp32 TMA wins at/above this bucketed context length under fixlen AND varlen.
_TMA_MIN_NUM_COLS = 131072


def tma_tuned_default(dtype, architecture, max_num_cols) -> bool:
    """Tuned async-TMA-load default: fp32 on Rubin with a large-enough context
    (max_num_cols >= _TMA_MIN_NUM_COLS), per the win-region sweep above. fp16/bf16 and
    small contexts stay on the LDG baseline."""
    return dtype == cutlass.Float32 and architecture in _RUBIN_TOPK_ARCHITECTURES and max_num_cols >= _TMA_MIN_NUM_COLS


def auto_tma_load(explicit, tuned_default: bool) -> bool:
    """Resolve a TMA-load enable flag: an explicit True/False wins; None -> the
    tuned default (the shipped auto behaviour)."""
    return tuned_default if explicit is None else bool(explicit)


def _align_smem_bytes(size: int) -> int:
    return (size + _SMEM_ALIGNMENT_BYTES - 1) & -_SMEM_ALIGNMENT_BYTES


def get_topk_architecture_config(
    compute_capability: tuple[int, int] | None = None,
) -> tuple[str, int]:
    """Return the filtered top-k resource policy for an architecture.

    Args:
        compute_capability: A ``(major, minor)`` CUDA compute capability. When
            ``None``, the current CUDA device's capability is queried.

    Returns:
        A ``(arch, min_blocks_per_mp)`` tuple, where ``arch`` is the
        ``"sm_<major><minor>"`` string and ``min_blocks_per_mp`` is the
        large-occupancy launch bound for that architecture.

    Raises:
        ValueError: If ``compute_capability`` is not a supported top-k
            architecture (not present in ``_LARGE_OCCUPANCY_MIN_BLOCKS_PER_MP``).
    """
    if compute_capability is None:
        device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(device)
    try:
        min_blocks_per_mp = _LARGE_OCCUPANCY_MIN_BLOCKS_PER_MP[compute_capability]
    except KeyError as error:
        major, minor = compute_capability
        raise ValueError(f"Unsupported top-k architecture: sm_{major}{minor}") from error
    major, minor = compute_capability
    return f"sm_{major}{minor}", min_blocks_per_mp


def get_large_occupancy_min_blocks_per_mp(
    compute_capability: tuple[int, int] | None = None,
) -> int:
    """Return the legal large-occupancy launch bound for an architecture."""
    return get_topk_architecture_config(compute_capability)[1]


# Bytes-in-flight per SM needed to saturate HBM for a read-dominated scan.
# Rubin (HBM4) needs ~128 KiB/SM vs Blackwell's ~64 KiB/SM -- 2x, from ~2.55x
# bandwidth * ~1.29x higher static latency (Little's Law). See GCA study
# "Minimum Bytes in Flight to saturate HBM bandwidth" (page 3120051050).
_RUBIN_READ_BYTES_IN_FLIGHT_TARGET = 128 * 1024


def auto_unroll_factor(
    max_num_cols: int,
    num_threads_per_cta: int,
    blocks_per_sm: int,
    vec_size: int,
    elem_bytes: int,
    target_bytes_in_flight: int = _RUBIN_READ_BYTES_IN_FLIGHT_TARGET,
) -> int:
    """Auto load-instruction unroll for the input scans: pick the factor that fills
    ``target_bytes_in_flight`` per SM, clamped to a register-safe factor and to
    the scan length actually available.

        bytes_in_flight/SM = blocks_per_sm * num_threads_per_cta * uf * bytes/load

    The public wrappers pass this so callers get the tuned default without
    hand-picking ``unroll_factor``.
    """
    bytes_per_load = vec_size * elem_bytes
    threads_per_sm = blocks_per_sm * num_threads_per_cta
    # ceil-div: smallest uf that reaches the target.
    uf_target = -(-target_bytes_in_flight // (threads_per_sm * bytes_per_load))
    # Register budget: 65536 regs/SM split across threads_per_sm sets regs/thread.
    # The cap keys on threads_per_sm (not blocks_per_sm) because that is what fixes
    # the budget: >=1024 threads/SM => <=64 regs/thread (Rubin 2x512 *or* 1x1024),
    # validated that uf>2 spills there -> cap 2. <=512 threads/SM gives >=128
    # regs/thread, room for uf=4 (validated on under-subscribed 256/512-thread
    # decode). This is the conservative choice; a 1-block/SM CTA that still runs
    # 1024 threads is as register-tight as the 2-block case, so it must also cap 2.
    uf_cap = 2 if threads_per_sm >= 1024 else 4
    # Cannot overlap more iterations than the per-CTA scan actually has.
    step_vec = num_threads_per_cta * vec_size
    big_iters_bound = max(1, max_num_cols // step_vec)
    raw = min(uf_target, uf_cap, big_iters_bound)
    # Snap down to a validated power-of-two unroll count.
    return max(a for a in (1, 2, 4, 8) if a <= raw)


@dsl_user_op
def atomic_add(dst_ptr: CutePointer, val: CuteInt32, *, loc=None, ip=None) -> CuteInt32:
    """Atomically add an Int32 value through a CuTe DSL pointer."""
    return cute.arch.atomic_add(
        ptr=dst_ptr.llvm_ptr,
        val=val,
        sem="relaxed",
        scope="sys",
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Cluster DSMEM primitives (inline PTX) — used only by the single-pass
# multi-CTA (radix-filter) path. Defined locally to avoid a circular import
# with single_pass_multi_cta_radix_topk_cluster (which imports this module).
# ---------------------------------------------------------------------------
@dsl_user_op
def _mapa_shared_cluster(smem_ptr: CutePointer, peer_rank: CuteInt32, *, loc=None, ip=None) -> CuteInt32:
    """Map a local SMEM address to a peer CTA's SMEM in cluster address space.

    PTX: mapa.shared::cluster.u32 $0, $1, $2;
    """
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def mapa_shared_cluster(smem_ptr, peer_rank):
    """Map a local SMEM address to a peer CTA's SMEM in cluster address space."""
    return _mapa_shared_cluster(smem_ptr, peer_rank)


@dsl_user_op
def _ld_shared_cluster_i32(mapped_addr: CuteInt32, *, loc=None, ip=None) -> CuteInt32:
    """Load an int32 from a cluster SMEM address.

    PTX: ld.shared::cluster.u32 $0, [$1];
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def ld_shared_cluster_i32(mapped_addr):
    """Load an int32 from a cluster SMEM address."""
    return _ld_shared_cluster_i32(mapped_addr)


"""
top-k varlen utils. could be used by prefill and decode phase.
"""


def half_as_ushort(half_val):
    """Interpret FP16 value as uint16 bit pattern"""
    return llvm.bitcast(cutlass.Uint16.mlir_type, half_val.ir_value())


def float_as_uint32(float_val):
    """Interpret FP32 value as uint32 bit pattern"""
    return llvm.bitcast(cutlass.Uint32.mlir_type, float_val.ir_value())


def ushort_as_half(u16_val):
    """Interpret uint16 bit pattern as FP16 value"""
    return cutlass.Float16(llvm.bitcast(cutlass.Float16.mlir_type, u16_val.ir_value()))


def uint32_as_float(u32_val):
    """Interpret uint32 bit pattern as FP32 value"""
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, u32_val.ir_value()))


def _is_nan(x):
    """fp32 x is NaN (unordered self-compare, one FSETP.NAN)."""
    v = cutlass.Float32(x).ir_value()
    return cutlass.Boolean(arith.cmpf(arith.CmpFPredicate.UNO, v, v))


_SSA_LOG2E = 1.4426950408889634


@cute.jit
def _ssa_block_reduce_max(val, scratch, tidx, num_warps: cutlass.Constexpr):
    """Block max over all threads (baseline indexer semantics: scratch is a
    32-slot fp32 smem array; num_warps <= 32)."""
    lane = tidx % 32
    warp_id = tidx // 32
    w = cute.arch.warp_reduction_max(val)
    if lane == 0:
        scratch[warp_id] = w
    cute.arch.barrier()
    x = -cutlass.Float32.inf
    if lane < num_warps:
        x = scratch[lane]
    return cute.arch.warp_reduction_max(x)


@cute.jit
def _ssa_block_reduce_sum(val, scratch, tidx, num_warps: cutlass.Constexpr):
    lane = tidx % 32
    warp_id = tidx // 32
    w = cute.arch.warp_reduction_sum(val)
    if lane == 0:
        scratch[warp_id] = w
    cute.arch.barrier()
    x = cutlass.Float32(0.0)
    if lane < num_warps:
        x = scratch[lane]
    return cute.arch.warp_reduction_sum(x)


class FilteredTopKKernelVarlen:
    def __init__(
        self,
        dtype: cutlass.Numeric,
        max_num_cols: int,
        top_k: int,
        num_copy_bits: int = 256,
        return_val: bool = True,
        enable_multi_cta: bool = False,
        chunk_size_per_cta: int = 16384,
        num_ctas_per_row: int = 1,
        merge_blocks: bool = False,
        overflow_policy: str = "REREAD",
        num_threads_override: int = 0,
        cache_smem_values: bool = False,
        single_pass_multi_cta: bool = False,
        architecture: str = "sm_100",
        unroll_factor: int = 4,
        enable_tma_load: bool = False,
        tma_num_stages: int = 4,
        enable_tma_load_p3: bool = False,
        tma_num_stages_p3: int = 2,
        in_kernel_sort: bool = False,
        ssa_epilogue: bool = False,
        ssa_tri_input: bool = False,
        reg_cache: bool = False,
        boundary_classify: bool = False,
        reg_select: bool = False,
        use_external_hist: bool = False,
        subtract_row_start_on_output: bool = False,
        unroll_blocks_per_sm: int | None = None,
    ):
        """
        Args:
            architecture: Target SM string (e.g. "sm_100", "sm_107", "sm_109")
                used to select architecture-specific SMEM sizing, occupancy
                launch bounds, and copy/atomic paths.
            unroll_factor: load-instruction unroll factor for the coarse/filter input scans,
                one of 1/2/4/8 (1 = the original 1-in-flight baseline). ``0`` is
                the "auto" sentinel, resolved when unroll_blocks_per_sm is given.
            unroll_blocks_per_sm: Occupancy used to resolve automatic unrolling.
            subtract_row_start_on_output: Return indices relative to each row start.
            overflow_policy: Controls behavior when threshold-bin candidates exceed the
                SMEM input buffer (filtered_topk_smem_input_size).  Only takes effect
                when max_num_cols > filtered_topk_smem_input_size; otherwise all policies
                are equivalent and no extra cost is incurred.

                "GMEM_SPILL"    -- Spill excess candidates to a pre-allocated GMEM
                                   extra_buffer.  Exact result.  Requires caller to
                                   allocate extra_buffer proportional to batch size;
                                   may OOM at large batch.
                "TRUNCATE"      -- Discard candidates that overflow SMEM.  Only
                                   retained candidates contribute to the refinement
                                   histogram, so refinement operates consistently on
                                   the stored set.  Non-exact (may output fewer than
                                   top_k indices when the threshold bin is dense).  No
                                   extra_buffer needed.  Requires
                                   top_k <= filtered_topk_smem_input_size.
                "REREAD_ALWAYS" -- Skip SMEM collection entirely in the coarse pass;
                                   always perform a second GMEM scan to collect
                                   threshold-bin candidates.  Exact result.  No
                                   extra_buffer needed; costs one extra GMEM read per
                                   row unconditionally.
                "REREAD"        -- Optimistic: attempt SMEM collection first.  If
                                   overflow is detected at runtime (s_overflow_flag),
                                   fall back to a REREAD_ALWAYS-style second GMEM scan.
                                   Exact result.  No extra_buffer needed; pays the
                                   extra GMEM read only when overflow actually occurs.
                "BOUNDED_SPILL" -- 3-tier graceful degrade keyed on the runtime
                                   candidate count: fits in SMEM (S) -> spill the
                                   overflow to a size-capped GMEM extra_buffer of
                                   host-chosen per-row capacity G -> if even G is
                                   exceeded, s_overflow_flag triggers the REREAD
                                   second GMEM scan.  Exact result.  Like GMEM_SPILL
                                   but the extra_buffer is bounded to O(G) instead
                                   of O(max_num_cols), so it cannot OOM at large
                                   batch/context.  G is set by the caller via
                                   spill_capacity (per-row candidate count) and/or
                                   spill_budget_bytes (total HBM budget; the stricter
                                   of the two wins) and is read by the kernel from
                                   the extra_buffer's own extent, so a single
                                   compiled kernel serves any G.
        """
        # use_external_hist: the p1 coarse histogram is supplied per-row by the
        # producer (SSA fused score kernel builds it in its epilogue while the
        # tile values are still in registers), so the kernel skips its own full
        # gmem scan and loads s_histogram straight from hist_gmem[bidx]. The
        # scan-replacement paths are incompatible by construction: reg_cache
        # fills its p3 replay fragments during the p1 scan, reg_select replaces
        # the whole pipeline, and p1 TMA staging has no scan to feed.
        if use_external_hist:
            assert not reg_cache, "use_external_hist: reg_cache needs the p1 scan"
            assert not reg_select, "use_external_hist is incompatible with reg_select"
            assert not enable_tma_load, "use_external_hist replaces the p1 scan; disable p1 TMA"
            assert not single_pass_multi_cta and not merge_blocks and not enable_multi_cta
        self.use_external_hist = use_external_hist
        self.dtype = dtype
        self.max_num_cols = max_num_cols
        self.top_k = top_k
        self.num_copy_bits = num_copy_bits
        self.enable_multi_cta = enable_multi_cta
        self.chunk_size_per_cta = chunk_size_per_cta
        self.num_ctas_per_row = num_ctas_per_row
        self.merge_blocks = merge_blocks
        self.overflow_policy = overflow_policy
        # Single-pass multi-CTA (radix-filter cluster) mode. Compile-time flag;
        # when False all cluster branches are const-folded away (single-CTA path
        # keeps identical SASS). Reuses chunk_size_per_cta / num_ctas_per_row for
        # chunk partitioning (num_ctas_per_row == ctas_per_group / cluster size).
        self.single_pass_multi_cta = single_pass_multi_cta
        assert unroll_factor in (0, 1, 2, 4, 8), f"unroll_factor must be one of 0 (auto), 1, 2, 4, 8; got {unroll_factor}"

        # Stage 0 prototype: async TMA (cp.async.bulk / UTMALDG) load for the
        # coarse-histogram vecscan, gated by enable_tma_load so the baseline (all
        # archs) keeps the synchronous LDG path untouched. tma_num_stages is the
        # ring depth (tunable/sweepable), sizing the bytes-in-flight vs the SMEM
        # staging carved from the candidate buffer (see _compute_* sizing). The
        # lever targets the #1 stall: global-load latency (long_scoreboard ~40%).
        assert tma_num_stages >= 1, f"tma_num_stages must be >= 1; got {tma_num_stages}"
        self.enable_tma_load = enable_tma_load
        self.tma_num_stages = tma_num_stages
        # Stage-1 prototype: async-TMA load for the p3 coarse-filter vecscan.
        # fp32 only (nbuf==2 double-buffer -> the idle candidate buffer 1 can be
        # aliased as staging with no candidate shrink). Independent switch from
        # enable_tma_load (p1). tma_num_stages_p3 is capped by buffer-1 capacity.
        assert tma_num_stages_p3 >= 1, f"tma_num_stages_p3 must be >= 1; got {tma_num_stages_p3}"
        self.enable_tma_load_p3 = enable_tma_load_p3
        self.tma_num_stages_p3 = tma_num_stages_p3
        # in_kernel_sort: bitonic-sort the K winners by ASCENDING column index
        # inside the kernel before the phase-3 store (values follow for free
        # because the writeback re-gathers them by index). Requires a
        # power-of-two top_k. Sort-order note: MR 25209's _bitonic_sort_topk
        # sorts by VALUE descending; SSA's consumers (spmm cursor,
        # sparse_attention_backward) need INDEX ascending, so that is what
        # this knob produces.
        self.in_kernel_sort = in_kernel_sort
        if in_kernel_sort:
            assert (top_k & (top_k - 1)) == 0, "in_kernel_sort requires a power-of-two top_k"
        # ssa_epilogue: replicate the SSA baseline topk_kernel's outputs
        # in-kernel: softmax over the selected scores -> output_probs, and
        # baseline padding semantics (pad = min(row+1, num_cols-1), probs 0)
        # on the trivial short-row path. Requires in_kernel_sort (one slot
        # per thread, sorted support).
        self.ssa_epilogue = ssa_epilogue
        if ssa_epilogue:
            assert in_kernel_sort, "ssa_epilogue requires in_kernel_sort"
        # ssa_tri_input: the input tensor is SSA's triangular
        # row-contiguous causal workspace (flat 1D). Row r = (bi*128+rr)
        # starts at element tri(bi)*128*128 + rr*(bi+1)*128 and is fully
        # contiguous -- same 130MB footprint as the block-compressed
        # layout, but linear rows so every scan/gather below is unchanged.
        self.ssa_tri_input = ssa_tri_input
        assert overflow_policy in (
            "GMEM_SPILL",
            "TRUNCATE",
            "REREAD_ALWAYS",
            "REREAD",
            "BOUNDED_SPILL",
        ), f"Unknown overflow_policy: {overflow_policy}"

        # Bound the shared-memory index staging to the supported range.
        if top_k <= 0 or top_k > 16384:
            raise ValueError(f"top_k must be in range [1, 16384], got {top_k}. " "Maximum supported top_k is 16384 for Blackwell architecture.")

        # s_indices only needs top_k slots; size to top_k to save SMEM. Still
        # referenced by the prefill kernel (filtered_top_k_prefill_varlen.py).
        self.filtered_topk_max_k = top_k
        # 8 bits for radix-based filter.
        self.radix = 256

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            self.num_buffer_smem_input_idx = 2
        else:
            self.num_buffer_smem_input_idx = 1

        # 65536 is the max index value for uint16.
        # SP multi-CTA reuses the same chunk partitioning as 2-pass multi-CTA.
        if cutlass.const_expr(enable_multi_cta or single_pass_multi_cta):
            self.per_row_max_num_cols = chunk_size_per_cta * num_ctas_per_row
        else:
            self.per_row_max_num_cols = self.max_num_cols

        if cutlass.const_expr(self.per_row_max_num_cols <= 65536):
            self.index_type = cutlass.Uint16
        else:
            self.index_type = cutlass.Uint32

        self.vec_size = num_copy_bits // dtype.width
        if cutlass.const_expr(dtype not in [cutlass.Float32, cute.BFloat16, cutlass.Float16]):
            raise ValueError(f"Unsupported dtype: {dtype}")

        if num_threads_override > 0:
            self.num_threads_per_cta = num_threads_override
        elif cutlass.const_expr(dtype == cutlass.Float32):
            if self.max_num_cols >= self.vec_size * 1024:
                self.num_threads_per_cta = 1024
            else:
                if cutlass.const_expr(self.max_num_cols > 2048 and self.max_num_cols < 8192):
                    self.num_threads_per_cta = 512
                else:
                    self.num_threads_per_cta = 256
        else:
            if self.max_num_cols >= 43008:
                self.num_threads_per_cta = 1024
            else:
                if cutlass.const_expr(self.max_num_cols > 4096 and self.max_num_cols < 43008):
                    self.num_threads_per_cta = 512
                else:
                    self.num_threads_per_cta = 256

        # Resolve automatic unrolling after the thread and vector sizes are known.
        self.unroll_factor = (
            auto_unroll_factor(self.max_num_cols, self.num_threads_per_cta, unroll_blocks_per_sm, self.vec_size, self.dtype.width // 8)
            if unroll_factor == 0 and unroll_blocks_per_sm is not None
            else unroll_factor
        )

        # reg_cache: keep the vec-aligned row segment register-resident across
        # p1 (coarse histogram) and p3 (filter/collect), turning the two full
        # gmem scans into one. NV fragments of vec_size per thread; the scalar
        # prologue/left edges (< vec_size elems each) still re-read gmem.
        # Requires the whole row to fit: max_num_cols <= NV * threads * vec.
        self.reg_cache = reg_cache
        if reg_cache:
            _rc_tile = self.num_threads_per_cta * self.vec_size
            self.reg_cache_nv = -(-self.max_num_cols // _rc_tile)  # ceil_div
            # NV=2 (16 data regs @512t) keeps 4 CTA/SM. NV=4 (32 regs)
            # saturates the 32-reg budget at 2048 thr/SM; MEASURED -9% at
            # 16K on GB300 (0.349 vs 0.320 two-pass) -- never auto-picked,
            # kept reachable for re-measurement on future silicon.
            assert self.reg_cache_nv <= 4, (
                f"reg_cache needs max_num_cols ({self.max_num_cols}) <= " f"4 * threads * vec ({4 * _rc_tile}); use more threads or " "disable reg_cache"
            )
            assert not (enable_tma_load or enable_tma_load_p3), "reg_cache replaces the gmem scans; TMA staging is mutually " "exclusive"
            assert not (enable_multi_cta or single_pass_multi_cta), "reg_cache supports the single-CTA-per-row path only"
        else:
            self.reg_cache_nv = 0

        # radix-based filter parameters — set before _compute_smem_input_size() so
        # ordered_type.width is available in the SMEM budget formula.
        # boundary_classify (the sglang topk_v2 collect trick): the p1 fp16
        # coarse binning is UNCHANGED (filter selectivity preserved); only the
        # p3 filter/collect classification is replaced -- the threshold bin is
        # inverted ONCE per row into two exact 32-bit ordered-key bounds, and
        # each element classifies with integer compares of its ordered key
        # instead of re-doing the fp32->fp16 convert + bit-twiddle. The p3
        # classification must agree with the p1 histogram for EVERY bit
        # pattern (NaN, +-0, +-inf, fp16-overflow values): any disagreement
        # under- or over-fills the selection. See _coarse_ord_bound.
        self.boundary_classify = boundary_classify
        if boundary_classify:
            assert dtype == cutlass.Float32, "boundary_classify is fp32-only"
        if cutlass.const_expr(dtype == cutlass.Float32):
            self.ordered_type = cute.Uint32
            self.first_refine_shift = 24
            self.num_refine_rounds = 4
        elif cutlass.const_expr(dtype in [cutlass.Float16, cute.BFloat16]):
            self.ordered_type = cute.Uint16
            self.first_refine_shift = 0
            self.num_refine_rounds = 1
        if in_kernel_sort:
            # Hybrid shuffle/smem bitonic supports any power-of-two top_k
            # (asserted above): top_k > threads sorts top_k // threads register
            # slots per thread (capped so the key set stays register-resident);
            # top_k < threads leaves the upper threads idle at the barriers.
            assert top_k >= 32, "in_kernel_sort requires top_k >= 32"
            assert top_k <= 8 * self.num_threads_per_cta, (
                "in_kernel_sort supports at most 8 elements per thread " f"(top_k={top_k}, threads={self.num_threads_per_cta})"
            )

        self.cache_smem_values = cache_smem_values
        self.architecture = architecture

        # Adaptive async-TMA staging depth. A row spans at most
        # `max_num_cols // tile_cols` full TMA tiles, so provisioning more ring
        # stages than that is pure waste -- and for a small context it makes the
        # staging ring (n_stages * tile_bytes) larger than the candidate buffer it
        # aliases, which the capacity guards below would (correctly) reject. Cap the
        # stage counts at the tile count so the ring always fits the aliased buffer
        # with no extra SMEM; a context smaller than one tile has no aligned middle
        # to TMA at all, so disable that path (the scalar edge scan covers the row).
        # Resolved here, before sizing/guards, so every downstream use (asserts,
        # _compute_rubin_smem_input_size, the caller's allocations, and the build
        # loops) reads the capped value. Large contexts (>= configured stages) are
        # unchanged. Must run after num_threads_per_cta / vec_size are set.
        _tma_tile_cols = self.num_threads_per_cta * self.vec_size
        _max_tma_tiles = self.max_num_cols // _tma_tile_cols
        if self.enable_tma_load:
            self.tma_num_stages = min(self.tma_num_stages, _max_tma_tiles)
            if self.tma_num_stages < 1:
                self.enable_tma_load = False
        if self.enable_tma_load_p3:
            self.tma_num_stages_p3 = min(self.tma_num_stages_p3, _max_tma_tiles)
            if self.tma_num_stages_p3 < 1:
                self.enable_tma_load_p3 = False

        # num_threads_per_cta must be set before _compute_smem_input_size() since
        # _compute_smem_input_size_for_occupancy() uses it to derive num_warps.
        self.filtered_topk_smem_input_size = self._compute_smem_input_size()

        # reg_select: dedicated small-N register kernel replacing the whole
        # coarse->filter->refine machine (sglang Register-path style): one
        # register-resident scan builds a 12-bit/4096-bin histogram (aliasing
        # candidate buffer 0's bytes), a block prefix-scan finds the threshold
        # bin, collect classifies straight from the register fragments (above
        # -> s_indices, threshold-bin ties -> candidate buffer 1), and an
        # exact rank-compare over the (tiny: ~row/4096-sized) tie set fills
        # the remaining slots deterministically. No refine rounds, no second
        # gmem scan, no REREAD. fp32 + single-CTA prefill only.
        self.reg_select = reg_select
        if reg_select:
            assert dtype == cutlass.Float32, "reg_select is fp32-only"
            assert not (enable_tma_load or enable_tma_load_p3), "reg_select is register-resident; TMA staging is mutually " "exclusive"
            assert not (enable_multi_cta or single_pass_multi_cta), "reg_select supports the single-CTA-per-row path only"
            assert not reg_cache, "reg_select subsumes reg_cache; enable only one"
            _rs_tile = self.num_threads_per_cta * self.vec_size
            self.reg_select_nv = -(-self.max_num_cols // _rs_tile)  # ceil_div
            assert self.reg_select_nv <= 2, f"reg_select needs max_num_cols ({self.max_num_cols}) <= " f"2 * threads * vec ({2 * _rs_tile})"
            _idx_bytes = 2 if self.index_type == cutlass.Uint16 else 4
            assert self.filtered_topk_smem_input_size * _idx_bytes >= 16384, (
                "reg_select aliases candidate buffer 0 as the 4096-bin " "Int32 histogram; buffer too small"
            )
            assert self.max_num_cols <= self.filtered_topk_smem_input_size, (
                "reg_select stores worst-case all-tie rows in candidate " f"buffer 1 ({self.filtered_topk_smem_input_size} slots)"
            )
            assert 4096 % self.num_threads_per_cta == 0
        else:
            self.reg_select_nv = 0

        # Second adaptive pass (needs S): the staging ring aliases a candidate
        # region -- p1 aliases the whole candidate buffer (nbuf * S * slot); p3 on
        # fp32 aliases the idle buffer 1 (S * slot). Cap the depth so the ring fits
        # that region; if even one stage does not fit (a tiny context whose aliased
        # buffer is smaller than a single tile), disable the path -- such a context
        # is not load-bound and its LDG / scalar-edge scan already covers the row.
        # REREAD_ALWAYS (no candidate buffer to alias) is handled by the explicit
        # guard below, not here. (self.enable_reread_always is derived later in
        # __init__, so test the policy directly.)
        if self.overflow_policy != "REREAD_ALWAYS":
            _idx_sz = 2 if self.index_type == cutlass.Uint16 else 4
            _val_sz = self.ordered_type.width // 8 if self.cache_smem_values else 0
            _slot = _idx_sz + _val_sz
            _tile_bytes = _tma_tile_cols * (self.dtype.width // 8)
            _S = self.filtered_topk_smem_input_size
            if self.enable_tma_load:
                _p1_fit = (self.num_buffer_smem_input_idx * _S * _slot) // _tile_bytes
                self.tma_num_stages = min(self.tma_num_stages, _p1_fit)
                if self.tma_num_stages < 1:
                    self.enable_tma_load = False
            if self.enable_tma_load_p3 and self.num_buffer_smem_input_idx == 2:
                _p3_fit = (_S * _slot) // _tile_bytes
                self.tma_num_stages_p3 = min(self.tma_num_stages_p3, _p3_fit)
                if self.tma_num_stages_p3 < 1:
                    self.enable_tma_load_p3 = False

        _needs_extra = self.max_num_cols > self.filtered_topk_smem_input_size
        if overflow_policy == "TRUNCATE" and self.top_k > self.filtered_topk_smem_input_size:
            raise ValueError(
                f"TRUNCATE overflow_policy requires top_k ({self.top_k}) <= "
                "filtered_topk_smem_input_size "
                f"({self.filtered_topk_smem_input_size}); use REREAD or GMEM_SPILL."
            )
        self.enable_gmem_store = (overflow_policy == "GMEM_SPILL") and _needs_extra
        self.enable_truncate = (overflow_policy == "TRUNCATE") and _needs_extra
        self.enable_reread_always = overflow_policy == "REREAD_ALWAYS"
        self.enable_reread = (overflow_policy == "REREAD") and _needs_extra
        # BOUNDED_SPILL: spill overflow candidates to a size-capped GMEM buffer
        # (host-chosen capacity G, passed as spill_capacity); when even G is
        # exceeded, fall back to the REREAD second-scan via s_overflow_flag. A
        # graceful 3-tier degrade (SMEM -> bounded GMEM -> reread) with bounded
        # extra_buffer, unlike GMEM_SPILL whose buffer is O(max_num_cols).
        self.enable_bounded_spill = (overflow_policy == "BOUNDED_SPILL") and _needs_extra

        self.return_val = return_val
        self.subtract_row_start_on_output = subtract_row_start_on_output

        if cutlass.const_expr(self.enable_tma_load):
            # The async-TMA staging ring aliases the candidate buffer(s)
            # (s_input_idx [+ s_input_val], both unused during the p1 coarse
            # histogram and contiguous in SMEM). Fail loudly if the ring cannot fit
            # inside that region -- otherwise the TMA writes would silently spill
            # past the candidate buffers into s_overflow_flag / s_last_remain /
            # s_warp_sums and corrupt them.
            if self.enable_reread_always:
                # TODO: allocate a dedicated SMEM staging ring (carved from the
                # per-CTA budget) instead of requiring a candidate buffer to alias.
                # That would lift this restriction and also support REREAD_ALWAYS,
                # at the cost of extra SMEM (candidate shrink / occupancy tradeoff).
                raise ValueError(
                    "enable_tma_load needs a candidate buffer to alias as the TMA " "staging ring, but overflow_policy=REREAD_ALWAYS allocates none."
                )
            idx_sz = 2 if self.index_type == cutlass.Uint16 else 4
            val_sz = self.ordered_type.width // 8 if self.cache_smem_values else 0
            alias_bytes = self.num_buffer_smem_input_idx * (idx_sz + val_sz) * self.filtered_topk_smem_input_size
            tile_bytes = self.num_threads_per_cta * self.vec_size * (self.dtype.width // 8)
            staging_bytes = self.tma_num_stages * tile_bytes
            if staging_bytes > alias_bytes:
                raise ValueError(
                    f"enable_tma_load: TMA staging ring ({staging_bytes} B = "
                    f"{self.tma_num_stages} stages x {tile_bytes} B) exceeds the "
                    f"aliasable candidate buffer ({alias_bytes} B; "
                    f"S={self.filtered_topk_smem_input_size}, "
                    f"cache_smem_values={self.cache_smem_values}). Reduce "
                    f"tma_num_stages or enable value caching / raise top_k."
                )

        if cutlass.const_expr(self.enable_tma_load_p3):
            # p3 coarse-filter async-TMA. fp32 (nbuf==2): alias the IDLE candidate
            # buffer 1 -> zero extra SMEM (merged layout when cache on; buffer-major
            # idx-only when cache off; slot padded to 128 B). fp16/bf16 (nbuf==1): no
            # idle buffer, so a dedicated staging is carved from the budget (S shrinks;
            # reserved in _compute_rubin_smem_input_size). Needs a candidate buffer.
            if self.enable_reread_always:
                raise ValueError("enable_tma_load_p3 needs a candidate buffer for the filter, but " "overflow_policy=REREAD_ALWAYS allocates none.")
            idx_sz = 2 if self.index_type == cutlass.Uint16 else 4
            val_sz = self.ordered_type.width // 8 if self.cache_smem_values else 0
            S = self.filtered_topk_smem_input_size
            tile_bytes = self.num_threads_per_cta * self.vec_size * (self.dtype.width // 8)
            p3_staging_bytes = self.tma_num_stages_p3 * tile_bytes
            if cutlass.const_expr(self.num_buffer_smem_input_idx == 2):
                # fp32: alias idle candidate buffer 1; check it fits.
                if self.cache_smem_values and (S * idx_sz) % 4 != 0:
                    raise ValueError(f"enable_tma_load_p3: S*idx_sz ({S}*{idx_sz}) must be 4-aligned " "for the merged candidate layout.")
                buf1_bytes = S * (idx_sz + val_sz)  # idle candidate buffer 1
                if p3_staging_bytes > buf1_bytes:
                    raise ValueError(
                        f"enable_tma_load_p3: p3 staging ({p3_staging_bytes} B) exceeds "
                        f"the idle candidate buffer 1 ({buf1_bytes} B; S={S}, "
                        f"idx_sz={idx_sz}). Reduce tma_num_stages_p3."
                    )

    def _compute_smem_input_size(self) -> int:
        return self._compute_smem_input_size_for_occupancy(target_blocks_per_sm=1)

    def _compute_rubin_smem_input_size(self, target_blocks_per_sm: int) -> int:
        """Compute candidate capacity from Rubin's selected SMEM carveout tier."""
        if self.overflow_policy == "REREAD_ALWAYS":
            return 0

        smem_capacity = cutlass.memory.get_smem_capacity_in_bytes(self.architecture)
        per_cta_budget = min(
            smem_capacity,
            _RUBIN_TOPK_SMEM_CARVEOUT_BYTES // target_blocks_per_sm - _SMEM_RUNTIME_RESERVE_BYTES,
        )

        idx_sz = 2 if self.index_type == cutlass.Uint16 else 4
        # counter, threshold, num_input, last_remain, and warp_sums each
        # occupy one 128-byte slot.
        fixed_smem = _align_smem_bytes((self.radix + 1) * 4) + 5 * _SMEM_ALIGNMENT_BYTES + _align_smem_bytes(self.top_k * idx_sz)
        if self.overflow_policy in ("GMEM_SPILL", "REREAD"):
            fixed_smem += _SMEM_ALIGNMENT_BYTES
        if self.overflow_policy == "BOUNDED_SPILL":
            # BOUNDED_SPILL allocates BOTH g_num_input and s_overflow_flag.
            fixed_smem += 2 * _SMEM_ALIGNMENT_BYTES
        if self.single_pass_multi_cta:
            fixed_smem += _align_smem_bytes((self.radix + 1) * 4)
        if self.enable_tma_load:
            # p1 staging ring aliases the candidate buffer (no extra ring bytes), but
            # its per-stage mbarriers are a dedicated allocation -> reserve them (else
            # at 2 CTAs/SM with S at capacity the SMEM over-subscribes -> launch fault).
            fixed_smem += _align_smem_bytes(2 * self.tma_num_stages * 8)
        # NOTE: the p1 async-TMA staging ring is NOT reserved here -- it aliases the
        # candidate buffer (s_input_idx), unused during the coarse histogram (p1), so
        # S is unchanged. Likewise p3 on fp32 (nbuf==2) aliases the idle candidate
        # buffer 1. But fp16/bf16 p3 (nbuf==1) has no idle buffer, so its staging is a
        # dedicated carve -- reserve it here (S shrinks accordingly; fp16 S is large
        # so no extra REREAD).
        if self.enable_tma_load_p3 and self.num_buffer_smem_input_idx == 1:
            _p3_tile_bytes = _align_smem_bytes(self.num_threads_per_cta * self.vec_size * (self.dtype.width // 8))
            fixed_smem += self.tma_num_stages_p3 * _p3_tile_bytes
            fixed_smem += _align_smem_bytes(2 * self.tma_num_stages_p3 * 8)
        elif self.enable_tma_load_p3 and self.num_buffer_smem_input_idx == 2:
            # fp32 p3: the staging ring aliases the idle candidate buffer 1 (no extra
            # bytes), but its per-stage mbarriers are a dedicated allocation -> reserve
            # them here. (The buffer-major candidate layout also pads each buffer to
            # 128 B; the capacity alignment below makes S*slot 128-aligned so that
            # padding is zero -- otherwise the actual allocation exceeds this budget by
            # nbuf*64 B and, at 2 CTAs/SM, over-subscribes SMEM into a launch fault.)
            fixed_smem += _align_smem_bytes(2 * self.tma_num_stages_p3 * 8)

        val_sz = self.ordered_type.width // 8 if self.cache_smem_values else 0
        candidate_bytes = self.num_buffer_smem_input_idx * (idx_sz + val_sz)
        candidate_capacity = (per_cta_budget - fixed_smem) // candidate_bytes
        # Fixed allocations must fit inside the per-CTA budget. A non-positive
        # capacity means the candidate buffer cannot exist at all (a zero-length
        # SMEM buffer is as invalid as a negative size), so fail loudly instead of
        # emitting a degenerate SMEM size downstream. This is a hard invariant --
        # unreachable for the supported archs (target_blocks_per_sm in {1, 2}).
        if candidate_capacity <= 0:
            raise ValueError(
                f"filtered top-k: fixed SMEM ({fixed_smem} B) exceeds the per-CTA "
                f"budget ({per_cta_budget} B); cannot size the candidate buffer "
                f"(arch={self.architecture}, top_k={self.top_k}, "
                f"target_blocks_per_sm={target_blocks_per_sm})."
            )
        # Make each candidate buffer a 128B multiple to avoid unbudgeted padding.
        if self.enable_tma_load_p3 and self.num_buffer_smem_input_idx == 2:
            # p3 uses the buffer-major layout, which pads EACH buffer's slot
            # (S * (idx_sz + val_sz)) up to 128 B independently. Align S so that slot
            # is already a 128 B multiple -> the padding is zero and the allocation
            # matches the candidate_bytes budgeted above (else 2 CTAs/SM over-subscribe
            # SMEM and the launch faults).
            slot = idx_sz + val_sz
            capacity_alignment = _SMEM_ALIGNMENT_BYTES // math.gcd(slot, _SMEM_ALIGNMENT_BYTES)
        else:
            capacity_alignment = _SMEM_ALIGNMENT_BYTES // (2 * self.num_buffer_smem_input_idx)
        candidate_capacity -= candidate_capacity % capacity_alignment
        return min(candidate_capacity, self.max_num_cols)

    def _compute_smem_input_size_for_occupancy(self, target_blocks_per_sm: int) -> int:
        """Compute candidate capacity for the requested resident CTA count.

        Rubin accounts for its selected SMEM tier and fixed allocations.
        Blackwell preserves the tuned 128 KiB candidate-only budget.
        """
        # Fail fast: both the Rubin and Blackwell paths below floor-divide the
        # per-CTA SMEM budget by target_blocks_per_sm, so a zero/negative value
        # would raise an opaque ZeroDivisionError (or size the buffer negatively)
        # deep in the sizing math instead of here. Unreachable for the supported
        # launch bounds (>= 1); kept as a hard invariant at the point of use.
        if target_blocks_per_sm < 1:
            raise ValueError(f"target_blocks_per_sm must be >= 1, got {target_blocks_per_sm}")
        if self.architecture in _RUBIN_TOPK_ARCHITECTURES:
            return self._compute_rubin_smem_input_size(target_blocks_per_sm)

        idx_sz = 2 if self.index_type == cutlass.Uint16 else 4
        if not self.cache_smem_values:
            # cache_smem_values=False: reserve ~104 KB L1 for load-instruction caching.
            INPUT_IDX_BUDGET_BASE = 128 * 1024  # 128 KB at 1 block/SM
            input_idx_budget = INPUT_IDX_BUDGET_BASE // target_blocks_per_sm
            if self.overflow_policy == "BOUNDED_SPILL":
                # reserve the g_num_input + s_overflow_flag SMEM slots (both
                # allocated only for BOUNDED_SPILL) so the candidate buffer does
                # not over-subscribe SMEM at >1 CTA/SM.
                input_idx_budget -= 2 * _SMEM_ALIGNMENT_BYTES
            max_S = input_idx_budget // (self.num_buffer_smem_input_idx * idx_sz)
        else:
            # cache_smem_values=True: same 128 KB budget as csv=False, with slot_sz
            # = idx_sz + val_sz so SMEM per block stays ~38 KB at target=4 → L1
            # unchanged. A device-budget formula that maximises S (→4864 for fp32)
            # was tried but caused 5-9% regressions on large-num_tokens single-CTA
            # configs; root cause not yet confirmed, kept in git history.
            INPUT_IDX_BUDGET_BASE = 128 * 1024  # 128 KB at 1 block/SM
            input_idx_budget = INPUT_IDX_BUDGET_BASE // target_blocks_per_sm
            if self.overflow_policy == "BOUNDED_SPILL":
                # reserve the g_num_input + s_overflow_flag SMEM slots (both
                # allocated only for BOUNDED_SPILL) so the candidate buffer does
                # not over-subscribe SMEM at >1 CTA/SM.
                input_idx_budget -= 2 * _SMEM_ALIGNMENT_BYTES
            val_sz = self.ordered_type.width // 8  # fp32→Uint32=4B, fp16/bf16→Uint16=2B
            slot_sz = idx_sz + val_sz
            max_S = input_idx_budget // (self.num_buffer_smem_input_idx * slot_sz)
        return min(max_S, self.max_num_cols)

    @cute.jit
    def _f16_key_boundary(self, key16):
        """Exact fp32 boundary B for the fp16-derived coarse key: for real
        (non-NaN) fp32 x, key(cvt_rn_f16(x)) < key16  <=>  x > B.

        key16 and key16-1 are adjacent fp16 codes (the key is a
        value-descending bijection of fp16), so any x between their values
        classifies by RNE rounding. Their midpoint is exactly
        fp32-representable; RNE sends x == midpoint to the even-mantissa
        neighbor, so B is the midpoint, shifted one fp32 ulp down when the
        key16 code's mantissa LSB is odd (midpoint rounds UP => x == m must
        already classify above). +-inf codes enter the midpoint as +-2^16
        (IEEE overflow threshold: cvt rounds to inf at 65504 + 16 = 65520).
        Only meaningful when both codes are real values and not the +-0 edge;
        _coarse_ord_bound pins the other bins (NaN-territory codes, key16 ==
        0x8000) and must be used for classification, never a float compare.
        """
        k_hi = cutlass.Int32(key16)
        k_up = k_hi - cutlass.Int32(1)  # next code = next LARGER value
        # inverse of the key transform -> fp16 bit patterns
        bits_hi = cutlass.select_(k_hi >= cutlass.Int32(0x8000), k_hi, (~k_hi) & cutlass.Int32(0x7FFF))
        bits_up = cutlass.select_(k_up >= cutlass.Int32(0x8000), k_up, (~k_up) & cutlass.Int32(0x7FFF))
        v_hi = cutlass.Float32(ushort_as_half(cutlass.Uint16(bits_hi)))
        v_up = cutlass.Float32(ushort_as_half(cutlass.Uint16(bits_up)))
        big = cutlass.Float32(65536.0)
        v_hi = cutlass.select_(v_hi > big, big, cutlass.select_(v_hi < -big, -big, v_hi))
        v_up = cutlass.select_(v_up > big, big, cutlass.select_(v_up < -big, -big, v_up))
        m = (v_hi + v_up) * cutlass.Float32(0.5)
        odd = (bits_hi & cutlass.Int32(1)) == cutlass.Int32(1)
        u = cutlass.Uint32(float_as_uint32(m))
        u_prev = cutlass.select_(
            m > cutlass.Float32(0.0), u - cutlass.Uint32(1), cutlass.select_(m < cutlass.Float32(0.0), u + cutlass.Uint32(1), cutlass.Uint32(0x80000001))
        )
        return uint32_as_float(cutlass.select_(odd, u_prev, u))

    @cute.jit
    def _bc_key(self, x):
        """boundary_classify per-element key: the fp32 ordered key of x, with
        every NaN keyed 0 (the top) because to_coarse_key bins every NaN in
        bin 0 (cvt canonical NaN 0x7FFF). Only the p3 classify pays for the
        NaN select; p1 and the refine/rescan passes are unchanged."""
        # Re-type as Uint32: select_ does not keep the unsigned type, and the
        # callers' `key < bound` must be an unsigned compare.
        return cutlass.Uint32(cutlass.select_(_is_nan(x), cutlass.Uint32(0), self.to_ordered(x)))

    @cute.jit
    def _coarse_ord_bound(self, t):
        """Smallest 32-bit ordered key whose coarse bin is >= t, so that for
        EVERY fp32 bit pattern x:  _bc_key(x) < bound(t)  <=>  to_coarse_key(x) < t.

        to_coarse_key is monotone non-decreasing in _bc_key (RNE cvt is
        monotone, the ordered key separates -0 from +0 the same way the fp16
        key does, and every NaN is bin 0 / key 0), so a single integer bound
        per threshold is exact. For 4 <= t <= 252 (t != 128) both fp16 codes
        adjacent to the bin edge are real values and the bound is the ordered
        key of _f16_key_boundary's fp32 edge. The remaining bins are pinned:
        t == 0 (nothing is above NaN), 1 <= t <= 3 (only NaN, key 0, is above:
        bins 1-2 are empty and bin 3 starts at +inf), t == 128 is the -0/+0
        edge (+0 is above, -0 is not; float compares cannot separate them),
        and t >= 253 lies past -inf (everything is above). Verified
        exhaustively over all 2^32 inputs."""
        t = cutlass.Int32(t)
        u = self.to_ordered(self._f16_key_boundary(t << cutlass.Int32(8)))
        u = cutlass.select_(t <= cutlass.Int32(3), cutlass.Uint32(1), u)
        u = cutlass.select_(t == cutlass.Int32(0), cutlass.Uint32(0), u)
        u = cutlass.select_(t == cutlass.Int32(128), cutlass.Uint32(0x80000000), u)
        u = cutlass.select_(t >= cutlass.Int32(253), cutlass.Uint32(0xFFFFFFFF), u)
        return cutlass.Uint32(u)

    @cute.jit
    def _coarse_class_boundaries(self, threshold_bin):
        """Per-row ordered-key bounds (u_hi, u_lo) for boundary_classify:
        _bc_key(x) < u_hi <=> coarse bin < threshold_bin ("above"); else
        _bc_key(x) < u_lo <=> bin == threshold_bin ("tie"). Computed once per
        row; exact for every bit pattern (see _coarse_ord_bound)."""
        _t = cutlass.Int32(threshold_bin)
        return (self._coarse_ord_bound(_t), self._coarse_ord_bound(_t + cutlass.Int32(1)))

    @cute.jit
    def _bc_bounds(self, threshold_bin, bc_float: cutlass.Constexpr):
        """Per-row bounds for _bc_above/_bc_class: fp32 bin edges when
        bc_float (only valid where _bc_float_ok holds), else exact integer
        ordered-key bounds."""
        if cutlass.const_expr(bc_float):
            _t16 = cutlass.Int32(threshold_bin) << cutlass.Int32(8)
            return (self._f16_key_boundary(_t16), self._f16_key_boundary(_t16 + cutlass.Int32(256)))
        else:
            return self._coarse_class_boundaries(threshold_bin)

    @cute.jit
    def _bc_float_ok(self, threshold_bin, bin0_count):
        """Row-uniform test for the plain float-compare classify: exact when
        the row holds no NaN (bin 0 holds exactly the NaN scores, every NaN
        converting to the canonical fp16 NaN) and both edges of the threshold
        bin are real, nonzero fp32 values (4 <= t <= 251, t not 127/128): then
        x > B(t) <=> ordered(x) < ordered(B(t)) == _coarse_ord_bound(t)."""
        t = cutlass.Int32(threshold_bin)
        ok = cutlass.Int32(0)
        if bin0_count == 0 and t >= cutlass.Int32(4) and t <= cutlass.Int32(251) and t != cutlass.Int32(127) and t != cutlass.Int32(128):
            ok = cutlass.Int32(1)
        return ok

    @cute.jit
    def _bc_above(self, x, b_hi, bc_float: cutlass.Constexpr):
        """boundary_classify: coarse bin of x < threshold bin."""
        if cutlass.const_expr(bc_float):
            return x > b_hi
        else:
            return self._bc_key(x) < b_hi

    @cute.jit
    def _bc_class(self, x, b_hi, b_lo, bc_float: cutlass.Constexpr):
        """0 = above, 1 = tie, 2 = below (boundary_classify)."""
        if cutlass.const_expr(bc_float):
            return cutlass.select_(x > b_hi, cutlass.Int32(0), cutlass.select_(x > b_lo, cutlass.Int32(1), cutlass.Int32(2)))
        else:
            k = self._bc_key(x)
            return cutlass.select_(k < b_hi, cutlass.Int32(0), cutlass.select_(k < b_lo, cutlass.Int32(1), cutlass.Int32(2)))

    @cute.jit
    def to_coarse_key(self, x):
        """Convert to coarse 8-bit key for histogram"""

        if cutlass.const_expr(self.dtype == cutlass.Float32):
            # Convert to FP16 and extract high 8 bits. cvt.rn.f16.f32 returns
            # the canonical NaN 0x7FFF for every NaN input (any sign/payload),
            # so all NaN scores land in bin 0, above +inf; _bc_key relies on it.
            h = x.to(cutlass.Float16)
            bits = half_as_ushort(h)

            key = cutlass.Uint16(0)

            # extract the sign bit
            # key = (bits & 0x8000) ? bits : ~bits & 0x7fff;
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)

            # high 8 bits
            return cute.Uint8((key >> 8) & 0xFF)
        else:
            # For half/bfloat16, extract high 8 bits directly
            bits = half_as_ushort(x)

            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            # high 8 bits
            return cute.Uint8((key >> 8) & 0xFF)

    @cute.jit
    def to_ordered(self, x):
        """Convert to ordered integer for comparison"""
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            bits = float_as_uint32(x)

            key = cutlass.Uint32(0)
            if bits & 0x80000000:
                key = cutlass.Uint32(bits)
            else:
                key = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
            return cute.Uint32(key)
        else:
            bits = half_as_ushort(x)

            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cute.Uint16(0xFFFF)) & cute.Uint16(0x7FFF)
            return cute.Uint16(key)

    @cute.jit
    def to_ordered_and_coarse(self, x):
        """Return (ordered, coarse_key) for x.
        For bf16/fp16, shares the half_as_ushort + sign-flip computation.
        For fp32, the two transforms differ (fp32->fp16 truncation vs full 32-bit
        sign-flip), so both are computed independently.
        """
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            return self.to_ordered(x), self.to_coarse_key(x)
        else:
            ordered = self.to_ordered(x)
            coarse_shift = cutlass.const_expr(self.ordered_type.width - int(math.log2(self.radix)))
            coarse = cute.Uint8((ordered >> self.ordered_type(coarse_shift)) & self.ordered_type(0xFF))
            return ordered, coarse

    @cute.jit
    def _collect_below_threshold_coarse(
        self,
        tidx,
        threshold_bin,
        s_counter,
        s_indices,
        _copy_atom,
        scan_frag,
        _aligned_base,
        vec_start,
        aligned_size,
        score,
        row_start,
        prologue_elems,
        left_start,
        left_size,
        reg_frags=None,
        bc_float: cutlass.Constexpr = False,
    ):
        """Collect all indices with coarse bin < threshold_bin from GMEM, then barrier."""
        val_one = cutlass.Int32(1)
        if cutlass.const_expr(self.boundary_classify):
            _bc_hi, _ = self._bc_bounds(threshold_bin, bc_float)
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * self.vec_size
        vec_size = self.vec_size
        # load-instruction unroll (unroll_factor) for the REREAD 2nd-scan collect, matching
        # the coarse/filter scans. unroll_factor 1 == original while.
        ic0 = tidx * cutlass.Int32(vec_size)
        if cutlass.const_expr(self.reg_cache):
            # replay the p1 register fragments instead of re-reading gmem
            for _k in cutlass.range_constexpr(self.reg_cache_nv):
                ic = ic0 + cutlass.Int32(_k * _step_vec)
                if ic < aligned_size:
                    for j in cutlass.range_constexpr(vec_size):
                        if cutlass.const_expr(self.boundary_classify):
                            _sel = self._bc_above(reg_frags[j, _k], _bc_hi, bc_float)
                        else:
                            _sel = self.to_coarse_key(reg_frags[j, _k]) < threshold_bin
                        if _sel:
                            pos = atomic_add(s_counter.iterator, val_one)
                            if pos < cutlass.Int32(self.top_k):
                                s_indices[pos] = self.index_type(vec_start + ic + cutlass.Int32(j))
        else:
            big_iters = cutlass.Int32(0)
            if aligned_size > ic0 + cutlass.Int32(vec_size - 1):
                big_iters = (aligned_size - ic0 - cutlass.Int32(vec_size)) // cutlass.Int32(_step_vec) + cutlass.Int32(1)
            # The runtime range may execute zero iterations. Give the
            # loop-carried coordinate a stable DSL type at its join.
            ic = cutlass.Int32(0)
            for _k in cutlass.range(big_iters, unroll=self.unroll_factor):
                ic = ic0 + _k * cutlass.Int32(_step_vec)
                cute.copy(
                    _copy_atom,
                    cute.make_tensor(
                        cute.make_ptr(
                            self.dtype,
                            _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                            cute.AddressSpace.gmem,
                            assumed_align=_align_bytes,
                        ),
                        cute.make_layout((vec_size,)),
                    ),
                    scan_frag,
                )
                for j in cutlass.range_constexpr(vec_size):
                    if cutlass.const_expr(self.boundary_classify):
                        _sel = self._bc_above(scan_frag[j], _bc_hi, bc_float)
                    else:
                        _sel = self.to_coarse_key(scan_frag[j]) < threshold_bin
                    if _sel:
                        pos = atomic_add(s_counter.iterator, val_one)
                        if pos < cutlass.Int32(self.top_k):
                            s_indices[pos] = self.index_type(vec_start + ic + cutlass.Int32(j))

        for j in range(tidx, prologue_elems, self.num_threads_per_cta):
            col_idx = cutlass.Int32(row_start + j)
            raw = score[col_idx]
            if cutlass.const_expr(self.boundary_classify):
                _sel = self._bc_above(raw, _bc_hi, bc_float)
            else:
                _sel = self.to_coarse_key(raw) < threshold_bin
            if _sel:
                pos = atomic_add(s_counter.iterator, val_one)
                if pos < cutlass.Int32(self.top_k):
                    s_indices[pos] = self.index_type(col_idx)

        for j in range(tidx, left_size, self.num_threads_per_cta):
            col_idx = cutlass.Int32(left_start + j)
            raw = score[col_idx]
            if cutlass.const_expr(self.boundary_classify):
                _sel = self._bc_above(raw, _bc_hi, bc_float)
            else:
                _sel = self.to_coarse_key(raw) < threshold_bin
            if _sel:
                pos = atomic_add(s_counter.iterator, val_one)
                if pos < cutlass.Int32(self.top_k):
                    s_indices[pos] = self.index_type(col_idx)

        cute.arch.barrier()

    @cute.jit
    def _collect_below_threshold_refine(
        self,
        tidx,
        threshold,
        offset,
        num_input,
        r_idx,
        s_input_idx,
        s_input_val,
        score,
        s_counter,
        s_indices,
        cur_g_num_input,
        buffer,
    ):
        """Collect all indices with refined bin < threshold from SMEM (and GMEM buffer), then barrier."""
        val_one = cutlass.Int32(1)
        for i in range(tidx, num_input, self.num_threads_per_cta):
            idx = s_input_idx[r_idx, i]
            idx = cutlass.Int32(cutlass.Uint32(idx))
            if cutlass.const_expr(self.cache_smem_values):
                bin_val = (self.ordered_type(s_input_val[r_idx, i]) >> offset) & 0xFF
            else:
                bin_val = (self.to_ordered(score[idx]) >> offset) & 0xFF
            if bin_val < threshold:
                pos = atomic_add(s_counter.iterator, val_one)
                if pos < cutlass.Int32(self.top_k):
                    s_indices[pos] = self.index_type(idx)
        if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
            for i in range(tidx, cur_g_num_input, self.num_threads_per_cta):
                idx = buffer[r_idx, i]
                bin_val = (self.to_ordered(score[idx]) >> offset) & 0xFF
                if bin_val < threshold:
                    pos = atomic_add(s_counter.iterator, val_one)
                    if pos < cutlass.Int32(self.top_k):
                        s_indices[pos] = self.index_type(idx)
        cute.arch.barrier()

    @cute.jit
    def _filter_and_histogram_per_elem_coarse(
        self,
        bin_val,
        threshold_bin,
        idx,
        raw_input,
        s_counter,
        s_indices,
        s_input_idx,
        s_input_val,
        s_num_input,
        s_histogram,
        g_num_input,
        buffer,
        s_overflow_flag,
    ):
        """Per-element if/elif handler for the coarse filter pass.

        bin_val < threshold_bin  → write to s_indices.
        bin_val == threshold_bin → store to s_input_idx (+ optional buffer) and
                                   update s_histogram for the next refinement round.
        """
        val_one = cutlass.Int32(1)
        if bin_val < threshold_bin:
            pos = atomic_add(s_counter.iterator, val_one)
            # Every selection write is bounds-checked: the counts come from a
            # histogram, and a slot index must never leave s_indices even if
            # a future change lets the classification disagree with it.
            if pos < cutlass.Int32(self.top_k):
                s_indices[pos] = idx
        elif bin_val == threshold_bin:
            if cutlass.const_expr(self.enable_gmem_store):
                # Hoist ordered before the pos < S check so s_input_val can be written inside it.
                ordered = self.to_ordered(raw_input)
                pos = atomic_add(s_num_input.iterator, val_one)
                if pos < self.filtered_topk_smem_input_size:
                    s_input_idx[0, pos] = idx
                    if cutlass.const_expr(self.cache_smem_values):
                        s_input_val[0, pos] = ordered
                else:
                    buffer_pos = atomic_add(g_num_input.iterator, val_one)
                    buffer[0, buffer_pos] = cutlass.Int32(cutlass.Uint32(idx))
                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                atomic_add(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)
            elif cutlass.const_expr(self.enable_bounded_spill):
                # Like GMEM_SPILL, but the GMEM buffer is size-capped at
                # G = its own extent (buffer.shape[1]); once even G is full,
                # set s_overflow_flag so the refine loop falls back to the
                # REREAD second scan (tier 3). 3-tier degrade
                # SMEM -> bounded GMEM -> reread, keeping extra_buffer O(G)
                # instead of GMEM_SPILL's O(max_num_cols).
                ordered = self.to_ordered(raw_input)
                pos = atomic_add(s_num_input.iterator, val_one)
                if pos < self.filtered_topk_smem_input_size:
                    s_input_idx[0, pos] = idx
                    if cutlass.const_expr(self.cache_smem_values):
                        s_input_val[0, pos] = ordered
                else:
                    buffer_pos = atomic_add(g_num_input.iterator, val_one)
                    if buffer_pos < cute.size(buffer.shape, [1]):
                        buffer[0, buffer_pos] = cutlass.Int32(cutlass.Uint32(idx))
                    else:
                        atomic_add(s_overflow_flag.iterator, val_one)
                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                atomic_add(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)
            elif cutlass.const_expr(self.enable_truncate):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    ordered = cutlass.Uint32(0)
                    sub_bin = cutlass.Uint32(0)
                else:
                    ordered = cutlass.Uint16(0)
                    sub_bin = cutlass.Int32(0)
                pos = atomic_add(s_num_input.iterator, val_one)
                if pos < self.filtered_topk_smem_input_size:
                    s_input_idx[0, pos] = idx
                    ordered = self.to_ordered(raw_input)
                    if cutlass.const_expr(self.cache_smem_values):
                        s_input_val[0, pos] = ordered
                    sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                    atomic_add(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)
            elif cutlass.const_expr(self.enable_reread):
                # Hoist ordered before the pos < S check so s_input_val can be written inside it.
                ordered = self.to_ordered(raw_input)
                pos = atomic_add(s_num_input.iterator, val_one)
                if pos < self.filtered_topk_smem_input_size:
                    s_input_idx[0, pos] = idx
                    if cutlass.const_expr(self.cache_smem_values):
                        s_input_val[0, pos] = ordered
                else:
                    # Use atomic_add (not plain store) to avoid concurrent non-atomic writes
                    # from multiple threads to the same SMEM address.  Any non-zero value
                    # means overflow; the did_overflow check uses != 0.
                    atomic_add(s_overflow_flag.iterator, val_one)
                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                atomic_add(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)
            else:
                # Hoist ordered before the pos < S check so s_input_val can be written inside it.
                ordered = self.to_ordered(raw_input)
                if cutlass.const_expr(not self.enable_reread_always):
                    pos = atomic_add(s_num_input.iterator, val_one)
                    if pos < self.filtered_topk_smem_input_size:
                        s_input_idx[0, pos] = idx
                        if cutlass.const_expr(self.cache_smem_values):
                            s_input_val[0, pos] = ordered
                sub_bin = (ordered >> self.first_refine_shift) & 0xFF
                atomic_add(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)

    @cute.jit
    def _filter_and_histogram_per_elem_refine(
        self,
        bin_val,
        threshold,
        idx_int32,
        ordered_val,
        offset,
        r_idx,
        is_last_round,
        s_counter,
        s_indices,
        s_input_idx,
        s_input_val,
        s_num_input,
        s_histogram,
        s_last_remain,
        g_num_input,
        buffer,
    ):
        """Per-element if/elif handler for refinement rounds.

        idx_int32   – Int32 column index, used for score lookup and buffer writes.
        ordered_val – pre-computed self.to_ordered(raw_input); avoids recomputing
                      it for sub_bin extraction when bin_val == threshold.

        bin_val < threshold  → write to s_indices.
        bin_val == threshold → last round: s_last_remain countdown;
                               otherwise: store to s_input_idx[r_idx^1] (+ optional
                               buffer) and update s_histogram for the next round.
        """
        val_one = cutlass.Int32(1)
        val_one_negative = cutlass.Int32(-1)
        idx = self.index_type(idx_int32)
        if bin_val < threshold:
            pos = atomic_add(s_counter.iterator, val_one)
            if pos < cutlass.Int32(self.top_k):
                s_indices[pos] = idx
        elif bin_val == threshold:
            if is_last_round:
                cur_pos = atomic_add(s_last_remain.iterator, val_one_negative)
                if cur_pos > 0 and cur_pos <= cutlass.Int32(self.top_k):
                    s_indices[self.top_k - cur_pos] = idx
            else:
                cur_pos = atomic_add(s_num_input.iterator + (r_idx ^ 1), val_one)
                if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
                    if cur_pos < self.filtered_topk_smem_input_size:
                        s_input_idx[r_idx ^ 1, cur_pos] = idx
                        if cutlass.const_expr(self.cache_smem_values):
                            s_input_val[r_idx ^ 1, cur_pos] = ordered_val
                    else:
                        buffer_pos = atomic_add(g_num_input.iterator + (r_idx ^ 1), val_one)
                        buffer[r_idx ^ 1, buffer_pos] = idx_int32
                    sub_bin = (ordered_val >> (offset - 8)) & 0xFF
                    atomic_add(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)
                else:
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        sub_bin = cutlass.Uint32(0)
                    else:
                        sub_bin = cutlass.Int32(0)
                    if cur_pos < self.filtered_topk_smem_input_size:
                        s_input_idx[r_idx ^ 1, cur_pos] = idx
                        if cutlass.const_expr(self.cache_smem_values):
                            s_input_val[r_idx ^ 1, cur_pos] = ordered_val
                        sub_bin = (ordered_val >> (offset - 8)) & 0xFF
                        atomic_add(s_histogram.iterator + cutlass.Int32(sub_bin), val_one)

    @cute.jit
    def _filter_and_histogram_coarse(
        self,
        tidx,
        threshold_bin,
        s_counter,
        s_indices,
        s_input_idx,
        s_input_val,
        s_num_input,
        s_histogram,
        g_num_input,
        buffer,
        _copy_atom,
        scan_frag,
        _aligned_base,
        vec_start,
        aligned_size,
        score,
        row_start,
        prologue_elems,
        left_start,
        left_size,
        s_overflow_flag,
        tma_atom=None,
        tma_tensor=None,
        s_tma_stage_p3=None,
        s_tma_mbar_p3=None,
        bidx=0,
        reg_frags=None,
        bc_float: cutlass.Constexpr = False,
    ):
        """Reset histogram, filter all input elements through three loops, then barrier.

        Covers vec-aligned GMEM, prologue scalar, and left scalar segments. When
        enable_tma_load_p3, the vec-aligned segment is loaded via an async-TMA ring
        (aliasing the idle candidate buffer 1) instead of synchronous LDG.
        """
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * self.vec_size
        _ik_f3_clear = self._iket_begin("f3_clear")
        cute.arch.barrier()
        for _hi in range(tidx, self.radix + 1, self.num_threads_per_cta):
            s_histogram[_hi] = 0
        cute.arch.barrier()
        self._iket_end(_ik_f3_clear)

        if cutlass.const_expr(self.boundary_classify):
            # sglang collect trick: invert the threshold bin ONCE into exact
            # ordered-key bounds; per-element classification below becomes two
            # integer compares (no per-element fp32->fp16 convert + twiddle).
            _bc_hi, _bc_lo = self._bc_bounds(threshold_bin, bc_float)

        _ik_f3_vec = self._iket_begin("f3_vecscan")
        if cutlass.const_expr(self.reg_cache):
            # replay the p1 register fragments instead of re-reading gmem
            vec_size = self.vec_size
            ic0 = tidx * cutlass.Int32(vec_size)
            for _k in cutlass.range_constexpr(self.reg_cache_nv):
                ic = ic0 + cutlass.Int32(_k * _step_vec)
                if ic < aligned_size:
                    for j in cutlass.range_constexpr(vec_size):
                        raw_input = reg_frags[j, _k]
                        if cutlass.const_expr(self.boundary_classify):
                            bin_val = self._bc_class(raw_input, _bc_hi, _bc_lo, bc_float)
                            _thr = cutlass.Int32(1)
                        else:
                            bin_val = self.to_coarse_key(raw_input)
                            _thr = threshold_bin
                        idx = self.index_type(vec_start + ic + cutlass.Int32(j))
                        self._filter_and_histogram_per_elem_coarse(
                            bin_val,
                            _thr,
                            idx,
                            raw_input,
                            s_counter,
                            s_indices,
                            s_input_idx,
                            s_input_val,
                            s_num_input,
                            s_histogram,
                            g_num_input,
                            buffer,
                            s_overflow_flag,
                        )
        elif cutlass.const_expr(self.enable_tma_load_p3):
            # Row-native async-TMA filter over the whole row (subsumes prologue/left,
            # gated off below). Staging aliases the idle candidate buffer 1.
            self._build_coarse_filter_tma(
                tidx,
                threshold_bin,
                tma_atom,
                tma_tensor,
                s_tma_stage_p3,
                s_tma_mbar_p3,
                bidx,
                row_start,
                left_start + left_size - row_start,
                score,
                s_counter,
                s_indices,
                s_input_idx,
                s_input_val,
                s_num_input,
                s_histogram,
                g_num_input,
                buffer,
                s_overflow_flag,
            )
        else:
            vec_size = self.vec_size
            # load-instruction unroll (unroll_factor): a counted range over the
            # full-vec trip count overlaps loads vs the 1-in-flight while (1 ==
            # orig); the scalar prologue/left loops below cover the remainder.
            ic0 = tidx * cutlass.Int32(vec_size)
            big_iters = cutlass.Int32(0)
            if aligned_size > ic0 + cutlass.Int32(vec_size - 1):
                big_iters = (aligned_size - ic0 - cutlass.Int32(vec_size)) // cutlass.Int32(_step_vec) + cutlass.Int32(1)
            ic = cutlass.Int32(0)
            for _k in cutlass.range(big_iters, unroll=self.unroll_factor):
                ic = ic0 + _k * cutlass.Int32(_step_vec)
                cute.copy(
                    _copy_atom,
                    cute.make_tensor(
                        cute.make_ptr(
                            self.dtype,
                            _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                            cute.AddressSpace.gmem,
                            assumed_align=_align_bytes,
                        ),
                        cute.make_layout((vec_size,)),
                    ),
                    scan_frag,
                )
                for j in cutlass.range_constexpr(vec_size):
                    raw_input = scan_frag[j]
                    if cutlass.const_expr(self.boundary_classify):
                        bin_val = self._bc_class(raw_input, _bc_hi, _bc_lo, bc_float)
                        _thr = cutlass.Int32(1)
                    else:
                        bin_val = self.to_coarse_key(raw_input)
                        _thr = threshold_bin
                    idx = self.index_type(vec_start + ic + cutlass.Int32(j))
                    self._filter_and_histogram_per_elem_coarse(
                        bin_val,
                        _thr,
                        idx,
                        raw_input,
                        s_counter,
                        s_indices,
                        s_input_idx,
                        s_input_val,
                        s_num_input,
                        s_histogram,
                        g_num_input,
                        buffer,
                        s_overflow_flag,
                    )
        self._iket_end(_ik_f3_vec)

        _ik_f3_tail = self._iket_begin("f3_scalartail")
        # The TMA filter path already scans the whole row (its own sub-tile edges);
        # skip the LDG prologue/left when enable_tma_load_p3 to avoid double-counting.
        if cutlass.const_expr(not self.enable_tma_load_p3):
            for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                col_idx = cutlass.Int32(row_start + j)
                raw = score[col_idx]
                if cutlass.const_expr(self.boundary_classify):
                    bin_val = self._bc_class(raw, _bc_hi, _bc_lo, bc_float)
                    _thr = cutlass.Int32(1)
                else:
                    bin_val = self.to_coarse_key(raw)
                    _thr = threshold_bin
                idx = self.index_type(col_idx)
                self._filter_and_histogram_per_elem_coarse(
                    bin_val,
                    _thr,
                    idx,
                    raw,
                    s_counter,
                    s_indices,
                    s_input_idx,
                    s_input_val,
                    s_num_input,
                    s_histogram,
                    g_num_input,
                    buffer,
                    s_overflow_flag,
                )

            for j in range(tidx, left_size, self.num_threads_per_cta):
                col_idx = cutlass.Int32(left_start + j)
                raw = score[col_idx]
                if cutlass.const_expr(self.boundary_classify):
                    bin_val = self._bc_class(raw, _bc_hi, _bc_lo, bc_float)
                    _thr = cutlass.Int32(1)
                else:
                    bin_val = self.to_coarse_key(raw)
                    _thr = threshold_bin
                idx = self.index_type(col_idx)
                self._filter_and_histogram_per_elem_coarse(
                    bin_val,
                    _thr,
                    idx,
                    raw,
                    s_counter,
                    s_indices,
                    s_input_idx,
                    s_input_val,
                    s_num_input,
                    s_histogram,
                    g_num_input,
                    buffer,
                    s_overflow_flag,
                )
        self._iket_end(_ik_f3_tail)

        _ik_f3_bar = self._iket_begin("f3_finalbar")
        fence_acq_rel_cta()
        cute.arch.barrier()
        self._iket_end(_ik_f3_bar)

    @cute.jit
    def _reread_always_per_elem_output(
        self,
        include_threshold,
        raw,
        col_idx,
        threshold_bin,
        T2,
        offset,
        chain_mask,
        chain_prefix,
        s_counter,
        s_indices,
        s_last_remain,
    ):
        """Per-element handler for REREAD_ALWAYS output scan.
        include_threshold is a compile-time bool.
        chain_mask is a DSL Int32 runtime value; chain_prefix is a runtime DSL
        ordered_type value. Both carry accumulated prior-round constraints.
        When chain_mask == 0 (round 0), ordered & 0 == 0 is always True.
        """
        ordered, coarse = self.to_ordered_and_coarse(raw)
        if coarse == threshold_bin:
            passes_chain = (ordered & self.ordered_type(chain_mask)) == chain_prefix
            if passes_chain:
                bin_val = (ordered >> offset) & 0xFF
                idx = self.index_type(col_idx)
                val_one = cutlass.Int32(1)
                if bin_val < T2:
                    pos = atomic_add(s_counter.iterator, val_one)
                    if pos < cutlass.Int32(self.top_k):
                        s_indices[pos] = idx
                elif cutlass.const_expr(include_threshold):
                    if bin_val == T2:
                        cur_pos = atomic_add(s_last_remain.iterator, cutlass.Int32(-1))
                        if cur_pos > 0 and cur_pos <= cutlass.Int32(self.top_k):
                            s_indices[self.top_k - cur_pos] = idx

    @cute.jit
    def _reread_always_per_elem_combined(
        self,
        raw,
        col_idx,
        threshold_bin,
        T2,
        offset,
        chain_mask,
        chain_prefix,
        s_counter,
        s_indices,
        s_histogram,
    ):
        """Per-element handler for REREAD_ALWAYS non-last-round combined scan.
        chain_mask is a DSL Int32 runtime value; chain_prefix is a runtime DSL
        ordered_type value. Both carry accumulated prior-round constraints.
        For elements passing coarse + chain filters:
          bin_val < T2  → write col_idx to s_indices (definitely top-K).
          bin_val == T2 → histogram at (ordered >> (offset - 8)) & 0xFF.
        """
        ordered, coarse = self.to_ordered_and_coarse(raw)
        if coarse == threshold_bin:
            passes_chain = (ordered & self.ordered_type(chain_mask)) == chain_prefix
            if passes_chain:
                bin_val = (ordered >> offset) & 0xFF
                val_one = cutlass.Int32(1)
                if bin_val < T2:
                    pos = atomic_add(s_counter.iterator, val_one)
                    if pos < cutlass.Int32(self.top_k):
                        s_indices[pos] = self.index_type(col_idx)
                elif bin_val == T2:
                    next_sub_bin = (ordered >> (offset - 8)) & 0xFF
                    atomic_add(s_histogram.iterator + cutlass.Int32(next_sub_bin), val_one)

    @cute.jit
    def _reread_always_gmem_output_scan(
        self,
        include_threshold,
        tidx,
        threshold_bin,
        T2,
        offset,
        chain_mask,
        chain_prefix,
        score,
        s_counter,
        s_indices,
        s_last_remain,
        _copy_atom,
        scan_frag,
        _aligned_base,
        vec_start,
        aligned_size,
        row_start,
        prologue_elems,
        left_start,
        left_size,
    ):
        """GMEM scan for REREAD_ALWAYS output phase.
        include_threshold is a compile-time bool.
        chain_mask is a DSL Int32 runtime value; chain_prefix is a runtime DSL
        ordered_type value — both carry prior-round constraints.
        Scans all three GMEM segments and writes qualifying indices to s_indices.
        Ends with cute.arch.barrier() to sync all writes before Phase 3.
        """
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * self.vec_size
        vec_size = self.vec_size

        # load-instruction unroll (unroll_factor) for the REREAD_ALWAYS output rescan.
        ic0 = tidx * cutlass.Int32(vec_size)
        big_iters = cutlass.Int32(0)
        if aligned_size > ic0 + cutlass.Int32(vec_size - 1):
            big_iters = (aligned_size - ic0 - cutlass.Int32(vec_size)) // cutlass.Int32(_step_vec) + cutlass.Int32(1)
        ic = cutlass.Int32(0)
        for _k in cutlass.range(big_iters, unroll=self.unroll_factor):
            ic = ic0 + _k * cutlass.Int32(_step_vec)
            cute.copy(
                _copy_atom,
                cute.make_tensor(
                    cute.make_ptr(
                        self.dtype,
                        _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=_align_bytes,
                    ),
                    cute.make_layout((vec_size,)),
                ),
                scan_frag,
            )
            for j in cutlass.range_constexpr(vec_size):
                col_idx = cutlass.Int32(vec_start + ic + cutlass.Int32(j))
                self._reread_always_per_elem_output(
                    include_threshold,
                    scan_frag[j],
                    col_idx,
                    threshold_bin,
                    T2,
                    offset,
                    chain_mask,
                    chain_prefix,
                    s_counter,
                    s_indices,
                    s_last_remain,
                )

        for j in range(tidx, prologue_elems, self.num_threads_per_cta):
            col_idx = cutlass.Int32(row_start + j)
            raw = score[col_idx]
            self._reread_always_per_elem_output(
                include_threshold,
                raw,
                col_idx,
                threshold_bin,
                T2,
                offset,
                chain_mask,
                chain_prefix,
                s_counter,
                s_indices,
                s_last_remain,
            )

        for j in range(tidx, left_size, self.num_threads_per_cta):
            col_idx = cutlass.Int32(left_start + j)
            raw = score[col_idx]
            self._reread_always_per_elem_output(
                include_threshold,
                raw,
                col_idx,
                threshold_bin,
                T2,
                offset,
                chain_mask,
                chain_prefix,
                s_counter,
                s_indices,
                s_last_remain,
            )

        cute.arch.barrier()

    @cute.jit
    def _reread_always_gmem_combined_scan(
        self,
        tidx,
        threshold_bin,
        T2,
        offset,
        chain_mask,
        chain_prefix,
        score,
        s_counter,
        s_indices,
        s_histogram,
        _copy_atom,
        scan_frag,
        _aligned_base,
        vec_start,
        aligned_size,
        row_start,
        prologue_elems,
        left_start,
        left_size,
    ):
        """GMEM scan for REREAD_ALWAYS non-last rounds: reset histogram, output < T2
        elements, and build histogram for the next round.
        chain_mask is a DSL Int32 runtime value; chain_prefix is a runtime DSL
        ordered_type value — both carry prior-round constraints.
        Ends with fence_acq_rel_cta() + cute.arch.barrier().
        Returns updated chain_prefix (runtime DSL value); caller updates chain_mask
        via | (cutlass.Int32(0xFF) << offset).
        """
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * self.vec_size
        vec_size = self.vec_size

        # Barrier before clearing s_histogram: ensures all threads have already
        # read s_histogram[threshold-1] to update topk_remaining in the caller
        # before any thread starts zeroing it here.
        cute.arch.barrier()
        for _hi in range(tidx, self.radix + 1, self.num_threads_per_cta):
            s_histogram[_hi] = 0
        cute.arch.barrier()

        # load-instruction unroll (unroll_factor) for the REREAD_ALWAYS combined rescan.
        ic0 = tidx * cutlass.Int32(vec_size)
        big_iters = cutlass.Int32(0)
        if aligned_size > ic0 + cutlass.Int32(vec_size - 1):
            big_iters = (aligned_size - ic0 - cutlass.Int32(vec_size)) // cutlass.Int32(_step_vec) + cutlass.Int32(1)
        ic = cutlass.Int32(0)
        for _k in cutlass.range(big_iters, unroll=self.unroll_factor):
            ic = ic0 + _k * cutlass.Int32(_step_vec)
            cute.copy(
                _copy_atom,
                cute.make_tensor(
                    cute.make_ptr(
                        self.dtype,
                        _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=_align_bytes,
                    ),
                    cute.make_layout((vec_size,)),
                ),
                scan_frag,
            )
            for j in cutlass.range_constexpr(vec_size):
                col_idx = cutlass.Int32(vec_start + ic + cutlass.Int32(j))
                self._reread_always_per_elem_combined(
                    scan_frag[j],
                    col_idx,
                    threshold_bin,
                    T2,
                    offset,
                    chain_mask,
                    chain_prefix,
                    s_counter,
                    s_indices,
                    s_histogram,
                )

        for j in range(tidx, prologue_elems, self.num_threads_per_cta):
            col_idx = cutlass.Int32(row_start + j)
            raw = score[col_idx]
            self._reread_always_per_elem_combined(
                raw,
                col_idx,
                threshold_bin,
                T2,
                offset,
                chain_mask,
                chain_prefix,
                s_counter,
                s_indices,
                s_histogram,
            )

        for j in range(tidx, left_size, self.num_threads_per_cta):
            col_idx = cutlass.Int32(left_start + j)
            raw = score[col_idx]
            self._reread_always_per_elem_combined(
                raw,
                col_idx,
                threshold_bin,
                T2,
                offset,
                chain_mask,
                chain_prefix,
                s_counter,
                s_indices,
                s_histogram,
            )

        fence_acq_rel_cta()
        cute.arch.barrier()

        # Return updated chain_prefix (runtime DSL value).
        # Caller updates chain_mask via | (cutlass.Int32(0xFF) << offset).
        return chain_prefix | self.ordered_type(self.ordered_type(T2) << self.ordered_type(offset))

    @cute.jit
    def _reread_gmem_rescan(
        self,
        topk_remaining,
        is_last_round,
        tidx,
        threshold_bin,
        threshold,
        offset,
        chain_mask,
        chain_prefix,
        score,
        s_counter,
        s_indices,
        s_last_remain,
        s_histogram,
        _copy_atom,
        scan_frag,
        _aligned_base,
        vec_start,
        aligned_size,
        row_start,
        prologue_elems,
        left_start,
        left_size,
    ):
        """GMEM re-scan phase shared by REREAD_ALWAYS and REREAD-overflow paths.

        Returns (run_next_round, chain_mask, chain_prefix).
        """
        run_next_round = True
        if topk_remaining == 0:
            self._reread_always_gmem_output_scan(
                False,
                tidx,
                threshold_bin,
                threshold,
                offset,
                chain_mask,
                chain_prefix,
                score,
                s_counter,
                s_indices,
                s_last_remain,
                _copy_atom,
                scan_frag,
                _aligned_base,
                vec_start,
                aligned_size,
                row_start,
                prologue_elems,
                left_start,
                left_size,
            )
            run_next_round = False
        else:
            if is_last_round:
                self._reread_always_gmem_output_scan(
                    True,
                    tidx,
                    threshold_bin,
                    threshold,
                    offset,
                    chain_mask,
                    chain_prefix,
                    score,
                    s_counter,
                    s_indices,
                    s_last_remain,
                    _copy_atom,
                    scan_frag,
                    _aligned_base,
                    vec_start,
                    aligned_size,
                    row_start,
                    prologue_elems,
                    left_start,
                    left_size,
                )
            else:
                chain_prefix = self._reread_always_gmem_combined_scan(
                    tidx,
                    threshold_bin,
                    threshold,
                    offset,
                    chain_mask,
                    chain_prefix,
                    score,
                    s_counter,
                    s_indices,
                    s_histogram,
                    _copy_atom,
                    scan_frag,
                    _aligned_base,
                    vec_start,
                    aligned_size,
                    row_start,
                    prologue_elems,
                    left_start,
                    left_size,
                )
                chain_mask = chain_mask | (cutlass.Int32(0xFF) << cutlass.Int32(offset))
        return run_next_round, chain_mask, chain_prefix

    @cute.jit
    def _filter_and_histogram_refine(
        self,
        tidx,
        threshold,
        offset,
        r_idx,
        is_last_round,
        num_input,
        cur_g_num_input,
        score,
        s_counter,
        s_indices,
        s_input_idx,
        s_input_val,
        s_num_input,
        s_histogram,
        s_last_remain,
        g_num_input,
        buffer,
    ):
        """Reset histogram, filter all threshold-bucket elements, then barrier.

        Covers SMEM s_input_idx loop and optional GMEM buffer loop.
        """
        cute.arch.barrier()
        for _hi in range(tidx, self.radix + 1, self.num_threads_per_cta):
            s_histogram[_hi] = 0
        cute.arch.barrier()

        for i in range(tidx, num_input, self.num_threads_per_cta):
            idx_tmp = s_input_idx[r_idx, i]
            idx_int32 = cutlass.Int32(cutlass.Uint32(idx_tmp))
            if cutlass.const_expr(self.cache_smem_values):
                ordered_val = self.ordered_type(s_input_val[r_idx, i])
            else:
                raw_input = score[idx_int32]
                ordered_val = self.to_ordered(raw_input)
            bin_val = (ordered_val >> offset) & 0xFF
            self._filter_and_histogram_per_elem_refine(
                bin_val,
                threshold,
                idx_int32,
                ordered_val,
                offset,
                r_idx,
                is_last_round,
                s_counter,
                s_indices,
                s_input_idx,
                s_input_val,
                s_num_input,
                s_histogram,
                s_last_remain,
                g_num_input,
                buffer,
            )

        if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
            cute.arch.barrier()
            for i in range(tidx, cur_g_num_input, self.num_threads_per_cta):
                idx_int32 = buffer[r_idx, i]
                raw_input = score[idx_int32]
                ordered_val = self.to_ordered(raw_input)
                bin_val = (ordered_val >> offset) & 0xFF
                self._filter_and_histogram_per_elem_refine(
                    bin_val,
                    threshold,
                    idx_int32,
                    ordered_val,
                    offset,
                    r_idx,
                    is_last_round,
                    s_counter,
                    s_indices,
                    s_input_idx,
                    s_input_val,
                    s_num_input,
                    s_histogram,
                    s_last_remain,
                    g_num_input,
                    buffer,
                )
            fence_acq_rel_cta()
        cute.arch.barrier()

    @cute.jit
    def prefix_sum_and_find_threshold_coarse(
        self,
        tidx,
        s_histogram,
        s_warp_sums,
        num_warps,
        s_threshold_bin_id,
        s_num_input,
        s_counter,
        s_last_remain,
        topk_remaining,
        g_num_input,
        s_num_input_idx=0,
    ):
        if cutlass.const_expr(self.radix <= self.num_threads_per_cta):
            previous = 0
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(val, s_warp_sums, tidx, self.radix, num_warps, barrier_id=1)
                s_histogram[tidx] = val
                # sync among self.radix threads
                cute.arch.barrier(barrier_id=1, number_of_threads=self.radix)

                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                    s_threshold_bin_id[0] = tidx
                    s_num_input[s_num_input_idx] = 0
                    if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
                        g_num_input[s_num_input_idx] = 0
                    s_counter[0] = 0
            # sync among all threads in a cta.
            cute.arch.barrier()
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = 0
            total_sum = 0
            for i in range(tidx, self.radix, self.num_threads_per_cta):
                val = s_histogram[i]
                val, total_sum = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    self.num_threads_per_cta,
                    num_warps,
                    barrier_id=2,
                    need_total_sum=True,
                )
                s_histogram[i] = val + previous_sum
                previous_sum = previous_sum + total_sum
            # sync among all threads in a cta.
            cute.arch.barrier()

            previous = 0
            run_loop = True
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                s_threshold_bin_id[0] = tidx
                s_num_input[s_num_input_idx] = 0
                if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
                    g_num_input[s_num_input_idx] = 0
                # the difference between coarse and fine-grained.
                s_counter[0] = 0
                run_loop = False

            if run_loop:
                run_next_loop = True
                for i in range(
                    tidx + self.num_threads_per_cta,
                    self.radix,
                    self.num_threads_per_cta,
                ):
                    if run_next_loop:
                        previous = s_histogram[i - 1]
                        if previous <= topk_remaining and s_histogram[i] > topk_remaining:
                            s_threshold_bin_id[0] = i
                            s_num_input[s_num_input_idx] = 0
                            if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
                                g_num_input[s_num_input_idx] = 0
                            # the difference between coarse and fine-grained.
                            s_counter[0] = 0
                            run_next_loop = False
            # sync among all threads in a cta.
            cute.arch.barrier()

    @cute.jit
    def prefix_sum_and_find_threshold_fine_grained(
        self,
        tidx,
        s_histogram,
        s_warp_sums,
        num_warps,
        s_threshold_bin_id,
        s_num_input,
        s_counter,
        s_last_remain,
        topk_remaining,
        g_num_input,
        s_num_input_idx=0,
    ):
        if cutlass.const_expr(self.radix <= self.num_threads_per_cta):
            previous = 0
            if tidx < cutlass.Int32(self.radix):
                val = s_histogram[tidx]
                val, total_sum = block_prefix_sum_kernel(val, s_warp_sums, tidx, self.radix, num_warps, barrier_id=1)
                s_histogram[tidx] = val
                # sync
                cute.arch.barrier(barrier_id=1, number_of_threads=self.radix)

                if tidx > 0:
                    previous = s_histogram[tidx - 1]
                if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                    s_threshold_bin_id[0] = tidx
                    s_num_input[s_num_input_idx] = 0
                    if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
                        g_num_input[s_num_input_idx] = 0
                    # the first difference between coarse and fine-grained.
                    s_last_remain[0] = topk_remaining - previous
            cute.arch.barrier()
        else:
            assert self.radix % self.num_threads_per_cta == 0
            previous_sum = 0
            val = 0
            total_sum = 0
            for i in range(tidx, self.radix, self.num_threads_per_cta):
                val = s_histogram[i]
                val, total_sum = block_prefix_sum_kernel(
                    val,
                    s_warp_sums,
                    tidx,
                    self.num_threads_per_cta,
                    num_warps,
                    barrier_id=2,
                    need_total_sum=True,
                )
                s_histogram[i] = val + previous_sum
                previous_sum = previous_sum + total_sum
            # sync among all threads in a cta.
            cute.arch.barrier()

            previous = 0
            run_loop = True
            if tidx > 0:
                previous = s_histogram[tidx - 1]
            if previous <= topk_remaining and s_histogram[tidx] > topk_remaining:
                s_threshold_bin_id[0] = tidx
                s_num_input[s_num_input_idx] = 0
                if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
                    g_num_input[s_num_input_idx] = 0
                # the difference between coarse and fine-grained.
                s_last_remain[0] = topk_remaining - previous
                run_loop = False
            if run_loop:
                run_next_loop = True
                for i in range(
                    tidx + self.num_threads_per_cta,
                    self.radix,
                    self.num_threads_per_cta,
                ):
                    if run_next_loop:
                        previous = s_histogram[i - 1]
                        if previous <= topk_remaining and s_histogram[i] > topk_remaining:
                            s_threshold_bin_id[0] = i
                            s_num_input[s_num_input_idx] = 0
                            if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
                                g_num_input[s_num_input_idx] = 0
                            # the difference between coarse and fine-grained.
                            s_last_remain[0] = topk_remaining - previous
                            run_next_loop = False
            # sync among all threads in a cta.
            cute.arch.barrier()

    @cute.jit
    def _cluster_reduce_histogram(self, tidx, s_histogram, s_hist_merged):
        """DSMEM histogram reduction for the single-pass multi-CTA path.

        Each thread sums bin ``my_bin`` across all peer CTAs' LOCAL
        ``s_histogram`` (including self) via cluster DSMEM and writes the total
        to the SEPARATE ``s_hist_merged`` buffer.  Never writes in-place: a peer
        may still be reading our ``s_histogram`` via DSMEM.

        The caller owns the surrounding barriers::

            cluster_arrive(); cluster_wait()           # publish local histograms
            _cluster_reduce_histogram(...)
            cute.arch.barrier()                        # s_hist_merged ready (intra-CTA)
            <prefix sum on s_hist_merged>
            cluster_arrive_relaxed(); cluster_wait()   # peers done reading before rebuild

        The two arrives differ on purpose: the publish one is non-relaxed
        ``cluster_arrive`` (release fence, so the s_histogram stores are visible
        to a peer's ld.shared::cluster; relaxed would risk stale reads — cf. GVR
        fix bc6d0e83a3), while the post-read one is relaxed (the peer reads
        drained into s_hist_merged before the intra-CTA barrier, so it is a
        liveness-only WAR barrier).

        Only bins ``0 .. radix-1`` are merged; the ``radix`` guard slot is not
        read by the prefix-sum helpers.
        """
        for my_bin in range(tidx, self.radix, self.num_threads_per_cta):
            acc = cutlass.Int32(0)
            local_ptr = s_histogram.iterator + cutlass.Int32(my_bin)
            for peer in cutlass.range_constexpr(self.num_ctas_per_row):
                remote = mapa_shared_cluster(local_ptr, cutlass.Int32(peer))
                acc = acc + ld_shared_cluster_i32(remote)
            s_hist_merged[my_bin] = acc

    @cute.jit
    def _cluster_collect(
        self,
        tidx,
        s_indices,
        s_counter,
        s_last_remain,
        s_prefix,
        cta_in_group,
        topk_remaining,
        output_indices_row,
        score,
        output_values_row,
    ):
        """Unified DSMEM prefix-scan output collection (Path A + Path B).

        ``s_prefix`` is a per-CTA scratch (reuses ``s_histogram``; needs >= 4
        int32 slots): [0]=group-1 count, [1]=group-2 count, [2]/[3]=computed
        exclusive offsets.  ``s_indices`` already holds this CTA's local
        group-1 at [0, s_counter) and group-2 at [top_k-topk_remaining, ...),
        filled by the reused Path A/B collection.

        Decode-only: indices are absolute column indices, written directly.
        """
        num_threads = self.num_threads_per_cta
        # 1. Publish this CTA's group-1 / group-2 counts.
        #    group-2 count = topk_remaining - max(0, s_last_remain[0]); s_last_remain
        #    starts at the final topk_remaining and is decremented per group-2 write.
        if tidx == 0:
            s_prefix[0] = s_counter[0]
            slr = s_last_remain[0]
            if slr < cutlass.Int32(0):
                slr = cutlass.Int32(0)
            s_prefix[1] = topk_remaining - slr
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        # 2. Exclusive prefix over peers p < cta_in_group (thread 0 computes).
        if tidx == 0:
            eo1 = cutlass.Int32(0)
            eo2 = cutlass.Int32(0)
            p0 = s_prefix.iterator + cutlass.Int32(0)
            p1 = s_prefix.iterator + cutlass.Int32(1)
            for peer in cutlass.range_constexpr(self.num_ctas_per_row):
                if cutlass.Int32(peer) < cta_in_group:
                    eo1 = eo1 + ld_shared_cluster_i32(mapa_shared_cluster(p0, cutlass.Int32(peer)))
                    eo2 = eo2 + ld_shared_cluster_i32(mapa_shared_cluster(p1, cutlass.Int32(peer)))
            s_prefix[2] = eo1
            s_prefix[3] = eo2
        cute.arch.barrier()
        # Liveness barrier before exit: relaxed (peer s_prefix reads already drained).
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        exclusive_offset_1 = s_prefix[2]
        exclusive_offset_2 = s_prefix[3]
        group1_total = self.top_k - topk_remaining
        group2_count = s_prefix[1]
        local_g1 = s_counter[0]

        # 3. group-1: s_indices[0 .. s_counter-1] -> output[exclusive_offset_1 + i]
        for i in range(tidx, local_g1, num_threads):
            idx = cutlass.Int32(cutlass.Uint32(s_indices[i]))
            pos = exclusive_offset_1 + i
            output_indices_row[pos] = idx
            if cutlass.const_expr(self.return_val):
                output_values_row[pos] = score[idx]

        # 4. group-2: s_indices[top_k-topk_remaining + i] -> output[group1_total + eo2 + i]
        #    (Path A: group2_count == 0, loop is a no-op)
        for i in range(tidx, group2_count, num_threads):
            pos = group1_total + exclusive_offset_2 + i
            if pos < self.top_k:
                src = self.top_k - topk_remaining + i
                idx = cutlass.Int32(cutlass.Uint32(s_indices[src]))
                output_indices_row[pos] = idx
                if cutlass.const_expr(self.return_val):
                    output_values_row[pos] = score[idx]

    @cute.jit
    def _bitonic_sort_indices_ascending(self, s_indices, tid):
        """In-kernel bitonic sort of s_indices[0:top_k] by ASCENDING index.

        Adapted from MR 25209's _bitonic_sort_topk (value-descending) to
        index-ascending for SSA's sorted-support contract, with a hybrid
        register/shuffle scheme generalized to any power-of-two top_k:

          * top_k == threads: one element per thread (the original scheme;
            for K=512 this is ~16 barriers vs 45 for a pure-smem bitonic).
          * top_k >  threads: E = top_k // threads register slots per thread,
            element g = e * threads + tid. Strides < 32 stay warp-local via
            shuffle_sync_bfly per slot; strides >= 32 -- cross-thread
            (st < threads) and cross-slot (st >= threads) alike -- run as
            in-place smem compare-exchanges under one uniform rule: the pair
            (g, g ^ st) is handled by whichever thread-slot holds the lower g.
          * top_k <  threads: only the first top_k threads own elements (whole
            warps, since top_k >= 32); the rest ride along on masked reads and
            gated stores so the CTA-wide barrier count stays uniform.

        All loops are cutlass.range_constexpr (fully unrolled; stage/stride
        constants are trace-time python ints), the only loop form the public
        wheel's AST preprocessor is guaranteed to fold. Direction bits use
        keep_small <=> ((g >> log2(sort_size)) ^ (g >> log2(stride))) & 1
        == 0, avoiding Boolean-Boolean equality. Indices are unique so no tie
        handling; no payload (phase-3 re-gathers values by index). Requires
        power-of-two top_k in [32, 8 * threads] (asserted in __init__);
        every thread of the CTA must call this (internal barriers).
        """
        K = self.top_k  # python ints at trace
        T = self.num_threads_per_cta
        k_log = K.bit_length() - 1
        E = max(1, K // T)  # register slots per thread
        idx_mask = cutlass.Int32((1 << self.index_type.width) - 1)
        # For K < T only the first K threads own elements; mask the read
        # index so idle threads touch a valid (ignored) slot.
        if cutlass.const_expr(K < T):
            rtid = tid & cutlass.Int32(K - 1)
        else:
            rtid = tid

        cute.arch.barrier()

        # zero-extended register copies of this thread's slots
        vs = []
        for e in cutlass.range_constexpr(E):
            raw0 = s_indices[cutlass.Int32(e * T) + rtid]
            if cutlass.const_expr(self.index_type.width < 32):
                vs.append(cutlass.Int32(raw0) & idx_mask)
            else:
                vs.append(cutlass.Int32(raw0))

        for i in cutlass.range_constexpr(k_log):
            ss_log = i + 1  # sort_size = 2 << i
            nbig = max(0, i - 4)  # strides 2**i .. 32 -> smem exchanges
            if cutlass.const_expr(nbig > 0):
                # flush register slots so the smem exchanges see current keys
                for e in cutlass.range_constexpr(E):
                    if cutlass.const_expr(K < T):
                        if tid < cutlass.Int32(K):
                            s_indices[rtid] = self.index_type(vs[e])
                    else:
                        s_indices[cutlass.Int32(e * T) + tid] = self.index_type(vs[e])
                cute.arch.barrier()
                for j in cutlass.range_constexpr(nbig):
                    st = (2 << i) >> (j + 1)
                    for e in cutlass.range_constexpr(E):
                        g = cutlass.Int32(e * T) + rtid
                        partner = g ^ cutlass.Int32(st)
                        # pair (g, g ^ st) handled once, by the lower-g holder
                        if cutlass.const_expr(K < T):
                            if tid < cutlass.Int32(K):
                                if partner > g:
                                    mine_raw = s_indices[g]
                                    other_raw = s_indices[partner]
                                    if cutlass.const_expr(self.index_type.width < 32):
                                        mine = cutlass.Int32(mine_raw) & idx_mask
                                        other = cutlass.Int32(other_raw) & idx_mask
                                    else:
                                        mine = cutlass.Int32(mine_raw)
                                        other = cutlass.Int32(other_raw)
                                    ab = (g >> cutlass.Int32(ss_log)) & cutlass.Int32(1)
                                    asc = ab == cutlass.Int32(0)
                                    swap = cutlass.select_(asc, mine > other, mine < other)
                                    if swap:
                                        s_indices[g] = other_raw
                                        s_indices[partner] = mine_raw
                        else:
                            if partner > g:
                                mine_raw = s_indices[g]
                                other_raw = s_indices[partner]
                                if cutlass.const_expr(self.index_type.width < 32):
                                    mine = cutlass.Int32(mine_raw) & idx_mask
                                    other = cutlass.Int32(other_raw) & idx_mask
                                else:
                                    mine = cutlass.Int32(mine_raw)
                                    other = cutlass.Int32(other_raw)
                                ab = (g >> cutlass.Int32(ss_log)) & cutlass.Int32(1)
                                asc = ab == cutlass.Int32(0)
                                swap = cutlass.select_(asc, mine > other, mine < other)
                                if swap:
                                    s_indices[g] = other_raw
                                    s_indices[partner] = mine_raw
                    cute.arch.barrier()
                # reload register slots after the smem strides
                for e in cutlass.range_constexpr(E):
                    raw1 = s_indices[cutlass.Int32(e * T) + rtid]
                    if cutlass.const_expr(self.index_type.width < 32):
                        vs[e] = cutlass.Int32(raw1) & idx_mask
                    else:
                        vs[e] = cutlass.Int32(raw1)
            # remaining strides min(2**i, 16) .. 1: warp-local registers
            for j in cutlass.range_constexpr(min(i + 1, 5)):
                st = min(2**i, 16) >> j
                st_log = st.bit_length() - 1
                for e in cutlass.range_constexpr(E):
                    g = cutlass.Int32(e * T) + rtid
                    other = cutlass.Int32(cute.arch.shuffle_sync_bfly(vs[e], st))
                    ab = (g >> cutlass.Int32(ss_log)) & cutlass.Int32(1)
                    lb = (g >> cutlass.Int32(st_log)) & cutlass.Int32(1)
                    keep_small = (ab ^ lb) == cutlass.Int32(0)
                    swap = cutlass.select_(keep_small, vs[e] > other, vs[e] < other)
                    vs[e] = cutlass.select_(swap, other, vs[e])

        # final store of the sorted keys
        for e in cutlass.range_constexpr(E):
            if cutlass.const_expr(K < T):
                if tid < cutlass.Int32(K):
                    s_indices[rtid] = self.index_type(vs[e])
            else:
                s_indices[cutlass.Int32(e * T) + tid] = self.index_type(vs[e])
        cute.arch.barrier()

    @cute.jit
    def _reg_select_topk(
        self,
        tidx,
        score,
        row_start,
        prologue_elems,
        vec_start,
        aligned_size,
        left_start,
        left_size,
        _copy_atom,
        _aligned_base,
        s_counter,
        s_num_input,
        s_indices,
        s_input_idx,
        s_threshold_bin_id,
        s_last_remain,
        s_warp_sums,
    ):
        """Small-N register top-k select (single scan, no refine, no REREAD).

        One register-resident pass builds a 12-bit/4096-bin histogram (Int32,
        aliasing candidate buffer 0's bytes); a two-level shuffle prefix-scan
        finds the threshold bin; collect classifies straight from the register
        fragments (above -> s_indices via atomic slot-grab, threshold-bin ties
        -> candidate buffer 1); an exact rank-compare over the tie set (~row /
        4096 elements in practice, worst-case bounded by buffer capacity
        asserted in __init__) fills the remaining slots deterministically.
        Fills s_indices[0:top_k]; the shared p5 writeback (sort / SSA
        epilogue / store) consumes it unchanged.
        """
        val_one = cutlass.Int32(1)
        vec_size = self.vec_size
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * vec_size
        T = self.num_threads_per_cta
        NV = self.reg_select_nv
        NBINS = 4096
        BPT = NBINS // T
        num_warps = T // 32

        # 4096-bin Int32 histogram aliasing candidate buffer 0's bytes
        hist = cute.make_tensor(
            cute.recast_ptr(s_input_idx.iterator, dtype=cutlass.Int32),
            cute.make_layout((NBINS,)),
        )
        # vectorized clear: 16B zero-stores, (4096*4B)/(T*16B) per thread
        hist4 = cute.tiled_divide(hist, (4,))
        _zf = cute.make_rmem_tensor((4,), cutlass.Int32)
        for _zi in cutlass.range_constexpr(4):
            _zf[_zi] = 0
        for _z in cutlass.range_constexpr((NBINS // 4) // T):
            cute.autovec_copy(_zf, hist4[None, cutlass.Int32(_z * T) + tidx])
        if tidx == 0:
            s_counter[0] = 0
            s_num_input[0] = 0
        if cutlass.const_expr(self.ssa_epilogue):
            # Same backstop as the radix path: every output slot starts as the
            # in-range SSA pad, so no slot can ever carry stale shared memory.
            _safe = left_start + left_size - cutlass.Int32(1)
            if _safe < 0:
                _safe = cutlass.Int32(0)
            for _s0 in range(tidx, self.top_k, T):
                s_indices[_s0] = self.index_type(_safe)
        cute.arch.barrier()

        # ---- pass 1: load register fragments + build histogram ----
        reg_frags = cute.make_rmem_tensor((vec_size, NV), self.dtype)
        ic0 = tidx * cutlass.Int32(vec_size)
        for _k in cutlass.range_constexpr(NV):
            ic = ic0 + cutlass.Int32(_k * _step_vec)
            if ic < aligned_size:
                cute.copy(
                    _copy_atom,
                    cute.make_tensor(
                        cute.make_ptr(
                            self.dtype,
                            _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                            cute.AddressSpace.gmem,
                            assumed_align=_align_bytes,
                        ),
                        cute.make_layout((vec_size,)),
                    ),
                    reg_frags[None, _k],
                )
                for j in cutlass.range_constexpr(vec_size):
                    b = cutlass.Int32(self._bc_key(reg_frags[j, _k]) >> cutlass.Uint32(20))
                    atomic_add(hist.iterator + b, val_one)
        for j in range(tidx, prologue_elems, T):
            col_idx = cutlass.Int32(row_start + j)
            b = cutlass.Int32(self._bc_key(score[col_idx]) >> cutlass.Uint32(20))
            atomic_add(hist.iterator + b, val_one)
        for j in range(tidx, left_size, T):
            col_idx = cutlass.Int32(left_start + j)
            b = cutlass.Int32(self._bc_key(score[col_idx]) >> cutlass.Uint32(20))
            atomic_add(hist.iterator + b, val_one)
        cute.arch.barrier()

        # ---- pass 2: two-level prefix scan over 4096 bins -> threshold ----
        loc = cutlass.Int32(0)
        for i in cutlass.range_constexpr(BPT):
            loc += hist[tidx * cutlass.Int32(BPT) + cutlass.Int32(i)]
        lane = tidx % 32
        warp = tidx // 32
        v = loc
        for d_log in cutlass.range_constexpr(5):
            d = 1 << d_log
            o = cutlass.Int32(cute.arch.shuffle_sync_up(v, d, mask_and_clamp=0))
            v += cutlass.select_(lane >= cutlass.Int32(d), o, cutlass.Int32(0))
        if lane == 31:
            s_warp_sums[warp] = v
        cute.arch.barrier()
        # every warp redundantly scans the warp sums in registers -- no
        # warp-0-only parking and one barrier fewer than a scan + write-back
        # round trip; each thread grabs its warp's exclusive base by shuffle.
        wv = cutlass.Int32(0)
        if lane < cutlass.Int32(num_warps):
            wv = s_warp_sums[lane]
        for d_log in cutlass.range_constexpr(5):
            d = 1 << d_log
            o = cutlass.Int32(cute.arch.shuffle_sync_up(wv, d, mask_and_clamp=0))
            wv += cutlass.select_(lane >= cutlass.Int32(d), o, cutlass.Int32(0))
        _src = cutlass.select_(warp > cutlass.Int32(0), warp - cutlass.Int32(1), cutlass.Int32(0))
        prev = cutlass.Int32(cute.arch.shuffle_sync(wv, _src))
        warp_base = cutlass.select_(warp > cutlass.Int32(0), prev, cutlass.Int32(0))
        excl = warp_base + v - loc  # exclusive prefix before my bin chunk
        k32 = cutlass.Int32(self.top_k)
        if excl < k32:
            if excl + loc >= k32:
                # exactly one thread's chunk contains the crossing
                run = excl
                for i in cutlass.range_constexpr(BPT):
                    b_id = tidx * cutlass.Int32(BPT) + cutlass.Int32(i)
                    nxt = run + hist[b_id]
                    if run < k32:
                        if nxt >= k32:
                            s_threshold_bin_id[0] = b_id
                            s_last_remain[0] = k32 - run
                    run = nxt
        cute.arch.barrier()
        t_bin = cutlass.Int32(s_threshold_bin_id[0])
        kprime = s_last_remain[0]
        # Classify with the SAME key the histogram used (_bc_key >> 20; NaN
        # keys 0, i.e. ranks first, as in the radix path):
        # bin < t_bin -> above, bin == t_bin -> tie. Float compares against
        # inverted bin edges disagree with the histogram for NaN (every
        # compare false) and at the -0/+0 edge, which under-fills the
        # selection; the integer key is exact for every bit pattern.
        _rs_cap = cutlass.Int32(self.filtered_topk_smem_input_size)

        # ---- pass 3: collect from register fragments (+ scalar edges) ----
        for _k in cutlass.range_constexpr(NV):
            ic = ic0 + cutlass.Int32(_k * _step_vec)
            if ic < aligned_size:
                for j in cutlass.range_constexpr(vec_size):
                    _b = cutlass.Int32(self._bc_key(reg_frags[j, _k]) >> cutlass.Uint32(20))
                    idx32 = vec_start + ic + cutlass.Int32(j)
                    if _b < t_bin:
                        pos = atomic_add(s_counter.iterator, val_one)
                        if pos < k32:
                            s_indices[pos] = self.index_type(idx32)
                    elif _b == t_bin:
                        tp = atomic_add(s_num_input.iterator, val_one)
                        if tp < _rs_cap:
                            s_input_idx[1, tp] = self.index_type(idx32)
        for j in range(tidx, prologue_elems, T):
            col_idx = cutlass.Int32(row_start + j)
            _b = cutlass.Int32(self._bc_key(score[col_idx]) >> cutlass.Uint32(20))
            if _b < t_bin:
                pos = atomic_add(s_counter.iterator, val_one)
                if pos < k32:
                    s_indices[pos] = self.index_type(col_idx)
            elif _b == t_bin:
                tp = atomic_add(s_num_input.iterator, val_one)
                if tp < _rs_cap:
                    s_input_idx[1, tp] = self.index_type(col_idx)
        for j in range(tidx, left_size, T):
            col_idx = cutlass.Int32(left_start + j)
            _b = cutlass.Int32(self._bc_key(score[col_idx]) >> cutlass.Uint32(20))
            if _b < t_bin:
                pos = atomic_add(s_counter.iterator, val_one)
                if pos < k32:
                    s_indices[pos] = self.index_type(col_idx)
            elif _b == t_bin:
                tp = atomic_add(s_num_input.iterator, val_one)
                if tp < _rs_cap:
                    s_input_idx[1, tp] = self.index_type(col_idx)
        cute.arch.barrier()

        # ---- pass 4: exact rank select over the tie set ----
        nt = s_num_input[0]
        if nt > _rs_cap:
            nt = _rs_cap
        base = s_counter[0]
        for i in range(tidx, nt, T):
            ii = cutlass.Int32(cutlass.Uint32(s_input_idx[1, i]))
            oi = self._bc_key(score[ii])
            rank = cutlass.Int32(0)
            for jj in range(nt):
                jv = cutlass.Int32(cutlass.Uint32(s_input_idx[1, jj]))
                oj = self._bc_key(score[jv])
                # smaller ordered key == larger value; ties break by index
                rank += cutlass.select_(oj < oi, val_one, cutlass.select_(oj == oi, cutlass.select_(jv < ii, val_one, cutlass.Int32(0)), cutlass.Int32(0)))
            if rank < kprime:
                _slot = base + rank
                if _slot < k32:
                    s_indices[_slot] = self.index_type(ii)
        cute.arch.barrier()

    @cute.jit
    def _phase3_writeback(
        self,
        tidx,
        row_start,
        s_indices,
        score,
        indices,
        dst,
        dst_values,
        dst_probs=None,
        s_prob_red=None,
    ):
        """Write the selected top-k from s_indices (+ values) back to GMEM output.

        Extracted verbatim from filtered_topk_kernel_per_row (no logic change) so the
        single-pass multi-CTA path can dispatch between this and a cluster collector.
        """
        # Phase 3: Output phase
        if cutlass.const_expr(self.in_kernel_sort):
            self._bitonic_sort_indices_ascending(s_indices, tidx % self.num_threads_per_cta)
        if cutlass.const_expr(self.ssa_epilogue):
            # SSA baseline predict: softmax over the K selected scores
            # (non-trivial rows have length > K, so all K slots are valid).
            # Generalized to top_k != threads: _E slots per thread when
            # top_k > threads; when top_k < threads the upper threads feed the
            # reduction identities (-inf / 0) and skip the store.
            _t = tidx % self.num_threads_per_cta
            _K = self.top_k
            _T = self.num_threads_per_cta
            _E = max(1, _K // _T)
            if cutlass.const_expr(_K < _T):
                _rt = _t & cutlass.Int32(_K - 1)
            else:
                _rt = _t
            _xs = []
            for _sl in cutlass.range_constexpr(_E):
                _raw = s_indices[cutlass.Int32(_sl * _T) + _rt]
                if cutlass.const_expr(self.index_type.width < 32):
                    _si = cutlass.Int32(_raw) & cutlass.Int32((1 << self.index_type.width) - 1)
                else:
                    _si = cutlass.Int32(_raw)
                _xs.append(cutlass.Float32(score[_si]))
            _lm = _xs[0]
            for _sl in cutlass.range_constexpr(_E - 1):
                _lm = cutlass.select_(_xs[_sl + 1] > _lm, _xs[_sl + 1], _lm)
            if cutlass.const_expr(_K < _T):
                _lm = cutlass.select_(_t < cutlass.Int32(_K), _lm, cutlass.Float32(-3.0e38))
            _m = _ssa_block_reduce_max(_lm, s_prob_red, _t, _T // 32)
            cute.arch.barrier()
            _es = []
            for _sl in cutlass.range_constexpr(_E):
                _es.append(cute.exp2((_xs[_sl] - _m) * cutlass.Float32(_SSA_LOG2E), fastmath=True))
            _ls = _es[0]
            for _sl in cutlass.range_constexpr(_E - 1):
                _ls = _ls + _es[_sl + 1]
            if cutlass.const_expr(_K < _T):
                _ls = cutlass.select_(_t < cutlass.Int32(_K), _ls, cutlass.Float32(0.0))
            _s = _ssa_block_reduce_sum(_ls, s_prob_red, _t, _T // 32)
            _r = cute.rcp(_s)
            for _sl in cutlass.range_constexpr(_E):
                if cutlass.const_expr(_K < _T):
                    if _t < cutlass.Int32(_K):
                        dst_probs[_rt] = _es[_sl] * _r
                else:
                    dst_probs[cutlass.Int32(_sl * _T) + _t] = _es[_sl] * _r
        output_vector_width = 2 if self.top_k % 2 == 0 else 1
        vecsize_out = cutlass.const_expr(
            min(
                self.top_k,
                cute.ceil_div(self.top_k, self.num_threads_per_cta),
                self.num_copy_bits // self.dtype.width,
                # Limit stores to the output vector width supported by the dtype.
                output_vector_width,
            )
        )
        assert self.top_k % vecsize_out == 0

        nvec_per_thread = cutlass.const_expr(cute.ceil_div(self.top_k, vecsize_out * self.num_threads_per_cta))
        topk_vals = cute.make_rmem_tensor((vecsize_out, nvec_per_thread), self.dtype)
        topk_indices = cute.make_rmem_tensor((vecsize_out, nvec_per_thread), cutlass.Int32)

        stride = self.num_threads_per_cta * vecsize_out
        for i in cutlass.range(nvec_per_thread, unroll_full=True):
            idx = i * stride + tidx % self.num_threads_per_cta * vecsize_out
            if idx < self.top_k:
                for v in cutlass.range(vecsize_out, unroll_full=True):
                    index_raw = s_indices[idx + v]
                    index = cutlass.Int32(cutlass.Uint32(index_raw))
                    if cutlass.const_expr(self.return_val):
                        topk_vals[v, i] = score[index]
                    if cutlass.const_expr(self.merge_blocks):
                        topk_indices[v, i] = indices[index]
                    elif cutlass.const_expr(self.subtract_row_start_on_output):
                        topk_indices[v, i] = index - cutlass.Int32(row_start)
                    else:
                        topk_indices[v, i] = index
        # [atom, rest_vec]
        mIndices_store = cute.tiled_divide(dst, (vecsize_out,))
        if cutlass.const_expr(self.return_val):
            mValues_store = cute.tiled_divide(dst_values, (vecsize_out,))
        # i represents the index of the vector in the output.
        for i in cutlass.range(cute.size(topk_vals.shape, [1]), unroll_full=True):
            col = i * self.num_threads_per_cta + tidx % self.num_threads_per_cta
            if col < self.top_k // vecsize_out:
                cute.autovec_copy(topk_indices[None, i], mIndices_store[None, col])
                if cutlass.const_expr(self.return_val):
                    cute.autovec_copy(topk_vals[None, i], mValues_store[None, col])

    def _iket_begin(self, name):
        """Open an IKET phase range (dev-only; no-op unless TOPK_IKET=1)."""
        if cutlass.const_expr(_TOPK_IKET):
            return _iket.range_start(name)
        return None

    def _iket_end(self, token):
        """Close the IKET phase range opened by _iket_begin (dev-only)."""
        if cutlass.const_expr(_TOPK_IKET):
            _iket.range_end(token)

    @cute.jit
    def _build_coarse_histogram_tma(
        self,
        tidx,
        s_histogram,
        val_one,
        tma_atom,
        tma_tensor,
        s_tma_stage,
        s_tma_mbar,
        bidx,
        row_start,
        length,
        score,
    ):
        """Row-native async-TMA (cp.async.bulk / UTMALDG) coarse-histogram build.

        Covers the WHOLE row [row_start, row_end): the tile-aligned middle
        [tma_start, tma_end) is loaded GMEM->SMEM through a tma_num_stages ring
        (hiding global-load latency, the #1 stall ~40%), and the two sub-tile edges
        [row_start, tma_start) + [tma_end, row_end) are scanned with scalar LDG
        (each < tile_cols columns; zero iterations when the row is tile-aligned).
        TMA needs only tile_cols alignment from column 0 -- it does NOT use the LDG
        vec_start/prologue/left decomposition, so the caller must skip the scalar
        prologue/left when enable_tma_load (this method already covers them).
        """
        vec_size = self.vec_size
        tile_cols = self.num_threads_per_cta * vec_size
        n_stages = self.tma_num_stages
        tile_bytes = tile_cols * (self.dtype.width // 8)
        num_warps_cta = self.num_threads_per_cta // 32
        tile_cols_i32 = cutlass.Int32(tile_cols)

        # Largest tile_cols-aligned sub-range [tma_start, tma_end) inside the row.
        row_end = row_start + length
        tma_start = ((row_start + tile_cols_i32 - cutlass.Int32(1)) // tile_cols_i32) * tile_cols_i32
        tma_end = (row_end // tile_cols_i32) * tile_cols_i32
        num_tiles = cutlass.Int32(0)
        if tma_end > tma_start:
            num_tiles = (tma_end - tma_start) // tile_cols_i32
        else:
            # No full tile fits: collapse so the head edge below scans the whole row.
            tma_start = row_end
            tma_end = row_end
        col_tile_base = tma_start // tile_cols_i32

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # PipelineTmaAsync ring: warp 0 is the sole producer (issues cp.async.bulk
        # via block_copy); all warps are consumers. The pipeline manages the
        # full/empty mbarrier counts + phases (hand-rolling them is incorrect for
        # multi-warp -- consumer needs num_warps arrivals, not 1).
        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=s_tma_mbar,
            num_stages=n_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps_cta),
            tx_count=tile_bytes,
        )
        prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, n_stages)
        cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, n_stages)

        # ((1, tile_cols), num_rows, num_col_tiles). col_tile_base is tile-aligned
        # by construction, so the tile index maps to exact columns.
        gTiled = cute.local_tile(tma_tensor, (1, tile_cols), (None, None))

        # Prologue: warp 0 fills the ring (guard each stage against short rows).
        if warp_idx == 0:
            for _i in cutlass.range_constexpr(n_stages):
                if cutlass.Int32(_i) < num_tiles:
                    pipe.producer_acquire(prod)
                    gtile = gTiled[None, None, bidx, col_tile_base + cutlass.Int32(_i)]
                    stile = s_tma_stage[prod.index, None, None]
                    cutlass.block.block_copy(
                        tma_atom,
                        cute.group_modes(gtile, 0, 2),
                        cute.group_modes(stile, 0, 2),
                        tma_bar_ptr=pipe.producer_get_barrier(prod),
                    )
                    pipe.producer_commit(prod)
                    prod.advance()

        # Main loop: all threads fold a tile into the histogram; warp 0 refills the
        # stage n_stages tiles ahead so loads stay in flight over the atomics.
        # Unroll interleaves consecutive tiles' consume+key-compute for ILP and to
        # hide the per-tile pipeline wait.
        for t in cutlass.range(num_tiles, unroll=2):
            pipe.consumer_wait(cons)
            stage = cons.index
            col0 = tidx * cutlass.Int32(vec_size)
            # Vectorized SMEM read of the thread's contiguous vec_size chunk (one
            # LDS.128x2 instead of vec_size scalar LDS -> cuts short_scoreboard).
            s_row = s_tma_stage[stage, 0, None]
            frag = cute.make_rmem_tensor((vec_size,), self.dtype)
            cute.autovec_copy(
                cute.make_tensor(s_row.iterator + col0, cute.make_layout((vec_size,))),
                frag,
            )
            for j in cutlass.range_constexpr(vec_size):
                bin_val = self.to_coarse_key(frag[j])
                atomic_add(s_histogram.iterator + cutlass.Int32(bin_val), val_one)
            pipe.consumer_release(cons)
            cons.advance()
            if warp_idx == 0:
                t_next = t + cutlass.Int32(n_stages)
                if t_next < num_tiles:
                    pipe.producer_acquire(prod)
                    gtile = gTiled[None, None, bidx, col_tile_base + t_next]
                    stile = s_tma_stage[prod.index, None, None]
                    cutlass.block.block_copy(
                        tma_atom,
                        cute.group_modes(gtile, 0, 2),
                        cute.group_modes(stile, 0, 2),
                        tma_bar_ptr=pipe.producer_get_barrier(prod),
                    )
                    pipe.producer_commit(prod)
                    prod.advance()

        # --- scalar sub-tile edges (LDG), placed AFTER the TMA issue so warp 0's
        # prefetch starts immediately instead of blocking behind these loads.
        # head [row_start, tma_start):
        head_size = tma_start - row_start
        for j in range(tidx, head_size, self.num_threads_per_cta):
            c = cutlass.Int32(row_start + j)
            bin_val = self.to_coarse_key(score[c])
            atomic_add(s_histogram.iterator + cutlass.Int32(bin_val), val_one)
        # tail [tma_end, row_end):
        tail_size = row_end - tma_end
        for j in range(tidx, tail_size, self.num_threads_per_cta):
            c = cutlass.Int32(tma_end + j)
            bin_val = self.to_coarse_key(score[c])
            atomic_add(s_histogram.iterator + cutlass.Int32(bin_val), val_one)

    @cute.jit
    def _build_coarse_filter_tma(
        self,
        tidx,
        threshold_bin,
        tma_atom,
        tma_tensor,
        s_tma_stage,
        s_tma_mbar,
        bidx,
        row_start,
        length,
        score,
        s_counter,
        s_indices,
        s_input_idx,
        s_input_val,
        s_num_input,
        s_histogram,
        g_num_input,
        buffer,
        s_overflow_flag,
    ):
        """Row-native async-TMA coarse FILTER over [row_start, row_end).

        Same ring as _build_coarse_histogram_tma but the per-element consume is the
        branchy filter (_filter_and_histogram_per_elem_coarse) instead of a histogram
        atomic. Staging aliases the idle candidate buffer 1 (fp32 nbuf==2). Covers the
        whole row; caller must skip the scalar prologue/left when enable_tma_load_p3.
        """
        vec_size = self.vec_size
        tile_cols = self.num_threads_per_cta * vec_size
        n_stages = self.tma_num_stages_p3
        tile_bytes = tile_cols * (self.dtype.width // 8)
        num_warps_cta = self.num_threads_per_cta // 32
        tile_cols_i32 = cutlass.Int32(tile_cols)

        row_end = row_start + length
        tma_start = ((row_start + tile_cols_i32 - cutlass.Int32(1)) // tile_cols_i32) * tile_cols_i32
        tma_end = (row_end // tile_cols_i32) * tile_cols_i32
        num_tiles = cutlass.Int32(0)
        if tma_end > tma_start:
            num_tiles = (tma_end - tma_start) // tile_cols_i32
        else:
            tma_start = row_end
            tma_end = row_end
        col_tile_base = tma_start // tile_cols_i32

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=s_tma_mbar,
            num_stages=n_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps_cta),
            tx_count=tile_bytes,
        )
        prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, n_stages)
        cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, n_stages)
        gTiled = cute.local_tile(tma_tensor, (1, tile_cols), (None, None))

        if warp_idx == 0:
            for _i in cutlass.range_constexpr(n_stages):
                if cutlass.Int32(_i) < num_tiles:
                    pipe.producer_acquire(prod)
                    gtile = gTiled[None, None, bidx, col_tile_base + cutlass.Int32(_i)]
                    stile = s_tma_stage[prod.index, None, None]
                    cutlass.block.block_copy(
                        tma_atom,
                        cute.group_modes(gtile, 0, 2),
                        cute.group_modes(stile, 0, 2),
                        tma_bar_ptr=pipe.producer_get_barrier(prod),
                    )
                    pipe.producer_commit(prod)
                    prod.advance()

        for t in cutlass.range(num_tiles, unroll=2):
            pipe.consumer_wait(cons)
            stage = cons.index
            col0 = tidx * cutlass.Int32(vec_size)
            base_col = tma_start + t * tile_cols_i32 + col0
            s_row = s_tma_stage[stage, 0, None]
            frag = cute.make_rmem_tensor((vec_size,), self.dtype)
            cute.autovec_copy(
                cute.make_tensor(s_row.iterator + col0, cute.make_layout((vec_size,))),
                frag,
            )
            for j in cutlass.range_constexpr(vec_size):
                raw_input = frag[j]
                bin_val = self.to_coarse_key(raw_input)
                idx = self.index_type(base_col + cutlass.Int32(j))
                self._filter_and_histogram_per_elem_coarse(
                    bin_val,
                    threshold_bin,
                    idx,
                    raw_input,
                    s_counter,
                    s_indices,
                    s_input_idx,
                    s_input_val,
                    s_num_input,
                    s_histogram,
                    g_num_input,
                    buffer,
                    s_overflow_flag,
                )
            pipe.consumer_release(cons)
            cons.advance()
            if warp_idx == 0:
                t_next = t + cutlass.Int32(n_stages)
                if t_next < num_tiles:
                    pipe.producer_acquire(prod)
                    gtile = gTiled[None, None, bidx, col_tile_base + t_next]
                    stile = s_tma_stage[prod.index, None, None]
                    cutlass.block.block_copy(
                        tma_atom,
                        cute.group_modes(gtile, 0, 2),
                        cute.group_modes(stile, 0, 2),
                        tma_bar_ptr=pipe.producer_get_barrier(prod),
                    )
                    pipe.producer_commit(prod)
                    prod.advance()

        # scalar sub-tile edges (LDG), after the TMA issue so prefetch isn't blocked.
        head_size = tma_start - row_start
        for j in range(tidx, head_size, self.num_threads_per_cta):
            c = cutlass.Int32(row_start + j)
            raw = score[c]
            self._filter_and_histogram_per_elem_coarse(
                self.to_coarse_key(raw),
                threshold_bin,
                self.index_type(c),
                raw,
                s_counter,
                s_indices,
                s_input_idx,
                s_input_val,
                s_num_input,
                s_histogram,
                g_num_input,
                buffer,
                s_overflow_flag,
            )
        tail_size = row_end - tma_end
        for j in range(tidx, tail_size, self.num_threads_per_cta):
            c = cutlass.Int32(tma_end + j)
            raw = score[c]
            self._filter_and_histogram_per_elem_coarse(
                self.to_coarse_key(raw),
                threshold_bin,
                self.index_type(c),
                raw,
                s_counter,
                s_indices,
                s_input_idx,
                s_input_val,
                s_num_input,
                s_histogram,
                g_num_input,
                buffer,
                s_overflow_flag,
            )

    @cute.jit
    def filtered_topk_kernel_per_row(
        self,
        input: cute.Tensor,
        # gmem, used for the merge blocks kernel.
        input_indices: cute.Tensor,
        extra_buffer: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        row_start: int,
        length: int,
        bidx: int,
        s_histogram,
        s_counter,
        s_threshold_bin_id,
        s_num_input,
        g_num_input,
        s_indices,
        s_input_idx,
        s_input_val,
        s_last_remain,
        num_warps,
        s_warp_sums,
        s_overflow_flag,
        need_cluster_sync=False,
        s_hist_merged=None,
        cta_in_group=0,
        tma_atom=None,
        tma_tensor=None,
        s_tma_stage=None,
        s_tma_mbar=None,
        s_tma_stage_p3=None,
        s_tma_mbar_p3=None,
        output_probs=None,
        s_prob_red=None,
        hist_gmem=None,
    ):
        """CuTe DSL implementation of TopK kernel based on radix-based filter algorithm.

        Single-pass multi-CTA (radix-filter cluster) extras — only live when
        ``self.single_pass_multi_cta`` is True (const-folded away otherwise):
          - ``need_cluster_sync`` (runtime): True for cluster cooperation
            (needed_ctas >= 2), False for the solo fast path.
          - ``s_hist_merged``: separate DSMEM merge target (radix+1 int32).
          - ``cta_in_group``: this CTA's rank within its cluster.
        """
        # # Thread and block indexing
        tidx, _, _ = cute.arch.thread_idx()

        if cutlass.const_expr(self.ssa_tri_input):
            # SSA triangular row-contiguous workspace: row bidx = bi*128+rr
            # starts at element tri(bi)*16384 + rr*(bi+1)*128 and is linear.
            _bi = cutlass.Int64(bidx) // 128
            _rr = bidx % 128
            _off = (_bi * (_bi + 1) // 2) * cutlass.Int32(16384) + _rr * ((_bi + 1) * 128)
            _addr = input.iterator.toint() + cutlass.Int64(_off) * (self.dtype.width // 8)
            score = cute.make_tensor(cute.make_ptr(self.dtype, _addr, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((self.max_num_cols,)))
        else:
            score = input[bidx, None]
        dst_probs = None
        if cutlass.const_expr(self.ssa_epilogue):
            dst_probs = output_probs[bidx, None]
        if cutlass.const_expr(self.merge_blocks):
            indices = input_indices[bidx, None]
        else:
            indices = None
        if cutlass.const_expr(self.enable_multi_cta):
            dst = output_indices
            if cutlass.const_expr(self.return_val):
                dst_values = output_values
            else:
                dst_values = None
        else:
            dst = output_indices[bidx, None]
            if cutlass.const_expr(self.return_val):
                dst_values = output_values[bidx, None]
            else:
                dst_values = None
        # Note, for multi-cta version, each ctas must have its own extra_buffer.
        buffer = None
        if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
            if cutlass.const_expr(self.single_pass_multi_cta):
                # Per-CTA spill buffer: (num_rows * ctas_per_group, ...). bidx has
                # already been set to row_id by the decode kernel.
                buffer = extra_buffer[bidx * self.num_ctas_per_row + cta_in_group, None, None]
            elif cutlass.const_expr(self.enable_multi_cta):
                grid_dim_x, grid_dim_y, _ = cute.arch.grid_dim()
                bidx_val, bidy_val, _ = cute.arch.block_idx()
                buffer_row_id = bidx_val * grid_dim_y + bidy_val
                buffer = extra_buffer[buffer_row_id, None, None]
            else:
                buffer = extra_buffer[bidx, None, None]

        # for initial scalar load part.
        row_ptr = score.iterator + row_start
        row_addr_u64 = row_ptr.toint()

        # 256/8 = 32bytes
        align_bytes = self.num_copy_bits // 8
        # fp32: 4bytes
        elem_bytes = self.dtype.width // 8

        misalign = row_addr_u64 % align_bytes
        fix_bytes = cutlass.Int64(0)
        if misalign != 0:
            fix_bytes = align_bytes - misalign

        prologue_elems = cutlass.Int32(fix_bytes // elem_bytes)

        # SP multi-CTA cluster mode: an empty chunk (chunk_start >= eff_len ->
        # length <= 0) must scan NOTHING. Otherwise the prologue/left loops
        # (bounded by alignment, not length) would read -inf padding past the
        # row end and corrupt the DSMEM-merged histogram. Clamp so the total
        # scanned == max(length, 0). Guarded under const_expr so single-CTA /
        # 2-pass codegen is unchanged (there length > top_k in this branch).
        if cutlass.const_expr(self.single_pass_multi_cta):
            _len_nonneg = length
            if _len_nonneg < 0:
                _len_nonneg = cutlass.Int32(0)
            if prologue_elems > _len_nonneg:
                prologue_elems = _len_nonneg
            remaining = _len_nonneg - prologue_elems
        else:
            remaining = length - prologue_elems
        aligned_size = (remaining // self.vec_size) * self.vec_size
        left_size = remaining - aligned_size

        vec_start = row_start + prologue_elems
        left_start = vec_start + aligned_size

        # GVR-style direct GMEM load constants (all Python ints, compile-time).
        # Loop bounds computed from runtime aligned_size so threads past the
        # actual row end execute zero iterations — no OOB waste for short rows.
        vec_size = self.vec_size
        _elem_bytes = self.dtype.width // 8
        _align_bytes = self.num_copy_bits // 8
        _step_vec = self.num_threads_per_cta * self.vec_size
        # Byte address of the aligned portion start for this row (score[vec_start]).
        _aligned_base = (score.iterator + vec_start).toint()
        _copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.dtype,
            num_bits_per_copy=self.num_copy_bits,
        )

        scan_frag = cute.make_rmem_tensor((vec_size,), self.dtype)

        # Trivial case: length <= top_k. In SP multi-CTA cluster mode this
        # per-chunk shortcut is unsafe (a CTA taking it would skip the cluster
        # barriers -> deadlock, and emit its whole chunk as the row's top-k);
        # force the full radix path so every CTA cooperates.
        take_trivial = length <= self.top_k
        if cutlass.const_expr(self.single_pass_multi_cta):
            if need_cluster_sync:
                take_trivial = False
        if take_trivial:
            for i in range(tidx, self.top_k, self.num_threads_per_cta):
                if i < length:
                    if cutlass.const_expr(self.enable_multi_cta):
                        dst[i] = i + row_start
                    elif cutlass.const_expr(self.merge_blocks):
                        dst[i] = indices[i]
                    elif cutlass.const_expr(self.subtract_row_start_on_output):
                        dst[i] = i
                    else:
                        dst[i] = i + row_start
                    if cutlass.const_expr(self.return_val):
                        # dst[i] is a local index i; its value lives at the
                        # absolute column row_start + i (row_start may be
                        # non-zero for prefill). enable_multi_cta writes the
                        # absolute index but reads the same absolute column.
                        dst_values[i] = score[i + row_start]
                else:
                    if cutlass.const_expr(self.ssa_epilogue):
                        # SSA THD padding contract: clamp(row_end - 1, 0)
                        # (global coordinate; row_end <= key_len by contract)
                        _pad = row_start + length - 1
                        if _pad < 0:
                            _pad = cutlass.Int32(0)
                        dst[i] = _pad
                    else:
                        dst[i] = -1
                    if cutlass.const_expr(self.return_val):
                        dst_values[i] = dst_values.element_type(dst_values.element_type.inf * dst_values.element_type(-1.0))
            if cutlass.const_expr(self.ssa_epilogue):
                # baseline predict: softmax over the length valid scores,
                # zeros beyond. Generalized to top_k != threads: _E slots per
                # thread, and slots >= top_k are never stored (the old
                # one-slot form wrote dst_probs[tidx] for every thread, which
                # runs past the row when threads > top_k -- latent OOB on the
                # 1024-thread launch shape, unexercised by prior workloads).
                _T = cutlass.const_expr(self.num_threads_per_cta)
                _E = cutlass.const_expr(max(1, self.top_k // _T))
                _lm = -cutlass.Float32.inf
                for _sl in cutlass.range_constexpr(_E):
                    _i = cutlass.Int32(_sl * _T) + tidx
                    if _i < length:
                        _xi = cutlass.Float32(score[row_start + _i])
                        if _xi > _lm:
                            _lm = _xi
                _m = _ssa_block_reduce_max(_lm, s_prob_red, tidx, _T // 32)
                cute.arch.barrier()
                _ls = cutlass.Float32(0.0)
                for _sl in cutlass.range_constexpr(_E):
                    _i = cutlass.Int32(_sl * _T) + tidx
                    if _i < length:
                        _ls = _ls + cute.exp2((cutlass.Float32(score[row_start + _i]) - _m) * cutlass.Float32(_SSA_LOG2E), fastmath=True)
                _s = _ssa_block_reduce_sum(_ls, s_prob_red, tidx, _T // 32)
                _inv = cute.rcp(_s)
                for _sl in cutlass.range_constexpr(_E):
                    _i = cutlass.Int32(_sl * _T) + tidx
                    if _i < cutlass.Int32(self.top_k):
                        if _i < length:
                            dst_probs[_i] = cute.exp2((cutlass.Float32(score[row_start + _i]) - _m) * cutlass.Float32(_SSA_LOG2E), fastmath=True) * _inv
                        else:
                            dst_probs[_i] = cutlass.Float32(0.0)
        else:
            if cutlass.const_expr(self.reg_select):
                # dedicated small-N register kernel: single register-resident
                # scan + 12-bit histogram + exact tie rank; replaces the whole
                # coarse -> filter -> refine pipeline, converging on the shared
                # p5 writeback below.
                self._reg_select_topk(
                    tidx,
                    score,
                    row_start,
                    prologue_elems,
                    vec_start,
                    aligned_size,
                    left_start,
                    left_size,
                    _copy_atom,
                    _aligned_base,
                    s_counter,
                    s_num_input,
                    s_indices,
                    s_input_idx,
                    s_threshold_bin_id,
                    s_last_remain,
                    s_warp_sums,
                )
            else:
                topk_remaining = self.top_k

                val_one = cutlass.Int32(1)

                _ik_hist = self._iket_begin("p1_histogram")
                _ik_h1_clear = self._iket_begin("h1_clear")
                # Stage 1: Coarse histogram.
                # Use a strided loop so every bin is cleared even when
                # num_threads_per_cta < radix (e.g. 128 < 256).
                # use_external_hist: the producer already counted this row's
                # [row_start, row_end) into hist_gmem[bidx] -- load it instead
                # of clearing + scanning (bin `radix` is kernel scratch, not
                # part of the producer contract: keep it zeroed).
                for _hi in range(tidx, self.radix + 1, self.num_threads_per_cta):
                    if cutlass.const_expr(self.use_external_hist):
                        _hv = cutlass.Int32(0)
                        if _hi < cutlass.Int32(self.radix):
                            _hv = hist_gmem[bidx, _hi]
                        s_histogram[_hi] = _hv
                    else:
                        s_histogram[_hi] = 0
                if cutlass.const_expr(self.ssa_epilogue):
                    # Full-write invariant (backstop): every output slot starts
                    # as the in-range SSA pad, clamp(row_end - 1, 0), and every
                    # selection write is bounds-checked, so the writeback can
                    # only ever emit a scanned column or the pad -- never stale
                    # shared memory -- even if the p1 histogram and the p3/p4
                    # classification were ever to disagree (they are exact for
                    # every bit pattern; see _coarse_ord_bound).
                    _safe = row_start + length - 1
                    if _safe < 0:
                        _safe = cutlass.Int32(0)
                    for _s0 in range(tidx, self.top_k, self.num_threads_per_cta):
                        s_indices[_s0] = self.index_type(_safe)
                if cutlass.const_expr(self.enable_reread):
                    if tidx == 0:
                        s_overflow_flag[0] = 0
                cute.arch.barrier()
                self._iket_end(_ik_h1_clear)

                # reg_cache: load the vec-aligned segment ONCE into per-thread
                # register fragments; p1 histograms from them and p3 replays them
                # (single gmem scan instead of two).
                reg_frags = None
                if cutlass.const_expr(self.reg_cache):
                    # (vec, NV): mode-0 contiguous so reg_frags[None, _k] is a
                    # unit-stride vec_size fragment for cute.copy
                    reg_frags = cute.make_rmem_tensor((self.vec_size, self.reg_cache_nv), self.dtype)

                _ik_h1_vec = self._iket_begin("h1_vecscan")
                if cutlass.const_expr(self.use_external_hist):
                    pass  # s_histogram preloaded from hist_gmem above
                elif cutlass.const_expr(self.reg_cache):
                    ic0 = tidx * cutlass.Int32(vec_size)
                    for _k in cutlass.range_constexpr(self.reg_cache_nv):
                        ic = ic0 + cutlass.Int32(_k * _step_vec)
                        if ic < aligned_size:
                            cute.copy(
                                _copy_atom,
                                cute.make_tensor(
                                    cute.make_ptr(
                                        self.dtype,
                                        _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                                        cute.AddressSpace.gmem,
                                        assumed_align=_align_bytes,
                                    ),
                                    cute.make_layout((vec_size,)),
                                ),
                                reg_frags[None, _k],
                            )
                            for j in cutlass.range_constexpr(vec_size):
                                bin_val = self.to_coarse_key(reg_frags[j, _k])
                                atomic_add(
                                    s_histogram.iterator + cutlass.Int32(bin_val),
                                    val_one,
                                )
                elif cutlass.const_expr(self.enable_tma_load):
                    # Stage 0: row-native async-TMA build over the WHOLE row (tile-aligned
                    # middle via TMA + scalar sub-tile edges). It subsumes the scalar
                    # prologue/left below, which are gated off for enable_tma_load.
                    self._build_coarse_histogram_tma(
                        tidx,
                        s_histogram,
                        val_one,
                        tma_atom,
                        tma_tensor,
                        s_tma_stage,
                        s_tma_mbar,
                        bidx,
                        row_start,
                        length,
                        score,
                    )
                else:
                    # 1.1 Build histogram -- load-instruction unroll (unroll_factor):
                    # counted range overlaps loads vs the 1-in-flight while (1 == orig).
                    ic0 = tidx * cutlass.Int32(vec_size)
                    big_iters = cutlass.Int32(0)
                    if aligned_size > ic0 + cutlass.Int32(vec_size - 1):
                        big_iters = (aligned_size - ic0 - cutlass.Int32(vec_size)) // cutlass.Int32(_step_vec) + cutlass.Int32(1)
                    # The runtime range may execute zero iterations. Give the
                    # loop-carried coordinate a stable DSL type at its join.
                    ic = cutlass.Int32(0)
                    for _k in cutlass.range(big_iters, unroll=self.unroll_factor):
                        ic = ic0 + _k * cutlass.Int32(_step_vec)
                        cute.copy(
                            _copy_atom,
                            cute.make_tensor(
                                cute.make_ptr(
                                    self.dtype,
                                    _aligned_base + cutlass.Int64(ic) * cutlass.Int64(_elem_bytes),
                                    cute.AddressSpace.gmem,
                                    assumed_align=_align_bytes,
                                ),
                                cute.make_layout((vec_size,)),
                            ),
                            scan_frag,
                        )
                        for j in cutlass.range_constexpr(vec_size):
                            bin_val = self.to_coarse_key(scan_frag[j])
                            atomic_add(s_histogram.iterator + cutlass.Int32(bin_val), val_one)
                self._iket_end(_ik_h1_vec)

                _ik_h1_tail = self._iket_begin("h1_scalartail")
                # The scalar prologue/left cover the LDG vec-alignment edges. The TMA
                # path already scans the whole row (its own sub-tile edges), so skip
                # these to avoid double-counting when enable_tma_load; the external
                # hist covers the whole row too.
                if cutlass.const_expr(not (self.enable_tma_load or self.use_external_hist)):
                    # for initial scalar load part.
                    for j in range(tidx, prologue_elems, self.num_threads_per_cta):
                        col_idx = cutlass.Int32(row_start + j)
                        raw = score[col_idx]
                        bin_val = self.to_coarse_key(raw)
                        atomic_add(
                            s_histogram.iterator + cutlass.Int32(bin_val),
                            val_one,
                        )

                    # for left part (left_size)
                    for j in range(tidx, left_size, self.num_threads_per_cta):
                        col_idx = cutlass.Int32(left_start + j)
                        raw = score[col_idx]
                        bin_val = self.to_coarse_key(raw)
                        atomic_add(
                            s_histogram.iterator + cutlass.Int32(bin_val),
                            val_one,
                        )
                self._iket_end(_ik_h1_tail)

                _ik_h1_bar = self._iket_begin("h1_finalbar")
                cute.arch.barrier()
                self._iket_end(_ik_h1_bar)
                self._iket_end(_ik_hist)

                _ik_prefix = self._iket_begin("p2_prefix_threshold")
                # 1.2 and 1.3  Suffix sum to find threshold and find threshold bin.
                # SP multi-CTA cluster: DSMEM-merge peer histograms into s_hist_merged
                # first, then prefix-sum the merged buffer. The prefix-sum / threshold
                # subtraction are duplicated per branch (rather than selecting the
                # buffer into a variable) because the DSL cannot phi-merge two distinct
                # tensors across a runtime `if`. threshold_bin (a shared SMEM scalar)
                # is read straight-line; only the buffer read in the -= differs.
                if cutlass.const_expr(self.single_pass_multi_cta):
                    if need_cluster_sync:
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()
                        self._cluster_reduce_histogram(tidx, s_histogram, s_hist_merged)
                        cute.arch.barrier()
                        self.prefix_sum_and_find_threshold_coarse(
                            tidx,
                            s_hist_merged,
                            s_warp_sums,
                            num_warps,
                            s_threshold_bin_id,
                            s_num_input,
                            s_counter,
                            s_last_remain,
                            topk_remaining,
                            g_num_input,
                            s_num_input_idx=0,
                        )
                        # WAR barrier: relaxed (peer reads already drained, see docstring).
                        cute.arch.cluster_arrive_relaxed()
                        cute.arch.cluster_wait()
                    else:
                        self.prefix_sum_and_find_threshold_coarse(
                            tidx,
                            s_histogram,
                            s_warp_sums,
                            num_warps,
                            s_threshold_bin_id,
                            s_num_input,
                            s_counter,
                            s_last_remain,
                            topk_remaining,
                            g_num_input,
                            s_num_input_idx=0,
                        )
                else:
                    self.prefix_sum_and_find_threshold_coarse(
                        tidx,
                        s_histogram,
                        s_warp_sums,
                        num_warps,
                        s_threshold_bin_id,
                        s_num_input,
                        s_counter,
                        s_last_remain,
                        topk_remaining,
                        g_num_input,
                        s_num_input_idx=0,
                    )

                threshold_bin = s_threshold_bin_id[0]
                if threshold_bin > 0:
                    if cutlass.const_expr(self.single_pass_multi_cta):
                        if need_cluster_sync:
                            topk_remaining -= s_hist_merged[threshold_bin - 1]
                        else:
                            topk_remaining -= s_histogram[threshold_bin - 1]
                    else:
                        topk_remaining -= s_histogram[threshold_bin - 1]

                self._iket_end(_ik_prefix)

                # 1.4 Collect indices
                # boundary_classify: rows with no NaN and a threshold bin with
                # real, nonzero fp32 edges classify with plain float compares;
                # every other row takes the exact ordered-key compares (both
                # agree with the p1 histogram bit for bit; see _bc_float_ok).
                # Row-uniform branch, so each path is a separate specialization.
                _bc_fast = cutlass.Int32(0)
                if cutlass.const_expr(self.boundary_classify and not (self.enable_multi_cta or self.single_pass_multi_cta)):
                    _bc_fast = self._bc_float_ok(threshold_bin, s_histogram[0])
                if topk_remaining == 0:
                    _ik_collect = self._iket_begin("p3_collect")
                    _ca = (
                        tidx,
                        threshold_bin,
                        s_counter,
                        s_indices,
                        _copy_atom,
                        scan_frag,
                        _aligned_base,
                        vec_start,
                        aligned_size,
                        score,
                        row_start,
                        prologue_elems,
                        left_start,
                        left_size,
                    )
                    if _bc_fast == 1:
                        self._collect_below_threshold_coarse(*_ca, reg_frags=reg_frags, bc_float=True)
                    else:
                        self._collect_below_threshold_coarse(*_ca, reg_frags=reg_frags, bc_float=False)
                    self._iket_end(_ik_collect)
                else:
                    _ik_filter = self._iket_begin("p3_filter")
                    _fa = (
                        tidx,
                        threshold_bin,
                        s_counter,
                        s_indices,
                        s_input_idx,
                        s_input_val,
                        s_num_input,
                        s_histogram,
                        g_num_input,
                        buffer,
                        _copy_atom,
                        scan_frag,
                        _aligned_base,
                        vec_start,
                        aligned_size,
                        score,
                        row_start,
                        prologue_elems,
                        left_start,
                        left_size,
                        s_overflow_flag,
                    )
                    _fk = dict(
                        tma_atom=tma_atom,
                        tma_tensor=tma_tensor,
                        s_tma_stage_p3=s_tma_stage_p3,
                        s_tma_mbar_p3=s_tma_mbar_p3,
                        bidx=bidx,
                        reg_frags=reg_frags,
                    )
                    if _bc_fast == 1:
                        self._filter_and_histogram_coarse(*_fa, **_fk, bc_float=True)
                    else:
                        self._filter_and_histogram_coarse(*_fa, **_fk, bc_float=False)
                    self._iket_end(_ik_filter)

                    _ik_refine = self._iket_begin("p4_refine")
                    # Phase 2: Refinement rounds
                    # chain_mask (DSL Int32) and chain_prefix (runtime DSL ordered_type)
                    # accumulate prior-round constraints for REREAD_ALWAYS / REREAD overflow
                    # fallback. chain_mask is Int32 so it survives DSL phi-merge across the
                    # dynamic loop.
                    chain_mask = cutlass.Int32(0)
                    chain_prefix = self.ordered_type(0)
                    # REREAD: read overflow flag once before the loop; runtime bool that
                    # selects SMEM refinement (no overflow) vs GMEM re-scan (overflow).
                    # Visibility of s_overflow_flag[0] is guaranteed by the fence_acq_rel_cta()
                    # + barrier() at the end of _filter_and_histogram_coarse above; no additional
                    # barrier is needed here.  If that function's terminal barrier is ever moved
                    # to the call site, a barrier must be inserted before this read.
                    if cutlass.const_expr(self.enable_reread or self.enable_bounded_spill):
                        did_overflow = s_overflow_flag[0] != 0
                    run_next_round = True
                    for round in range(self.num_refine_rounds):
                        if run_next_round:
                            r_idx = round % 2
                            _ik_round = self._iket_begin(f"p4_refine_r{round}")

                            # SP multi-CTA cluster: DSMEM-merge peer histograms before
                            # the per-round prefix sum (same shape as the coarse site;
                            # duplicated per branch to avoid a tensor phi-merge).
                            if cutlass.const_expr(self.single_pass_multi_cta):
                                if need_cluster_sync:
                                    cute.arch.cluster_arrive()
                                    cute.arch.cluster_wait()
                                    self._cluster_reduce_histogram(tidx, s_histogram, s_hist_merged)
                                    cute.arch.barrier()
                                    self.prefix_sum_and_find_threshold_fine_grained(
                                        tidx,
                                        s_hist_merged,
                                        s_warp_sums,
                                        num_warps,
                                        s_threshold_bin_id,
                                        s_num_input,
                                        s_counter,
                                        s_last_remain,
                                        topk_remaining,
                                        g_num_input,
                                        s_num_input_idx=r_idx ^ 1,
                                    )
                                    # WAR barrier: relaxed (peer reads already drained).
                                    cute.arch.cluster_arrive_relaxed()
                                    cute.arch.cluster_wait()
                                else:
                                    self.prefix_sum_and_find_threshold_fine_grained(
                                        tidx,
                                        s_histogram,
                                        s_warp_sums,
                                        num_warps,
                                        s_threshold_bin_id,
                                        s_num_input,
                                        s_counter,
                                        s_last_remain,
                                        topk_remaining,
                                        g_num_input,
                                        s_num_input_idx=r_idx ^ 1,
                                    )
                            else:
                                self.prefix_sum_and_find_threshold_fine_grained(
                                    tidx,
                                    s_histogram,
                                    s_warp_sums,
                                    num_warps,
                                    s_threshold_bin_id,
                                    s_num_input,
                                    s_counter,
                                    s_last_remain,
                                    topk_remaining,
                                    g_num_input,
                                    s_num_input_idx=r_idx ^ 1,
                                )
                            threshold = s_threshold_bin_id[0]
                            if threshold > 0:
                                if cutlass.const_expr(self.single_pass_multi_cta):
                                    if need_cluster_sync:
                                        topk_remaining -= s_hist_merged[threshold - 1]
                                    else:
                                        topk_remaining -= s_histogram[threshold - 1]
                                else:
                                    topk_remaining -= s_histogram[threshold - 1]
                            offset = self.first_refine_shift - round * 8
                            is_last_round = round == self.num_refine_rounds - 1

                            if cutlass.const_expr(self.enable_reread_always):
                                run_next_round, chain_mask, chain_prefix = self._reread_gmem_rescan(
                                    topk_remaining,
                                    is_last_round,
                                    tidx,
                                    threshold_bin,
                                    threshold,
                                    offset,
                                    chain_mask,
                                    chain_prefix,
                                    score,
                                    s_counter,
                                    s_indices,
                                    s_last_remain,
                                    s_histogram,
                                    _copy_atom,
                                    scan_frag,
                                    _aligned_base,
                                    vec_start,
                                    aligned_size,
                                    row_start,
                                    prologue_elems,
                                    left_start,
                                    left_size,
                                )
                            elif cutlass.const_expr(self.enable_reread):
                                if did_overflow:
                                    # Overflow fallback: REREAD_ALWAYS-style GMEM re-scan.
                                    run_next_round, chain_mask, chain_prefix = self._reread_gmem_rescan(
                                        topk_remaining,
                                        is_last_round,
                                        tidx,
                                        threshold_bin,
                                        threshold,
                                        offset,
                                        chain_mask,
                                        chain_prefix,
                                        score,
                                        s_counter,
                                        s_indices,
                                        s_last_remain,
                                        s_histogram,
                                        _copy_atom,
                                        scan_frag,
                                        _aligned_base,
                                        vec_start,
                                        aligned_size,
                                        row_start,
                                        prologue_elems,
                                        left_start,
                                        left_size,
                                    )
                                else:
                                    # No overflow: SMEM-based refinement (same as GMEM_SPILL).
                                    num_input = min(
                                        s_num_input[r_idx],
                                        self.filtered_topk_smem_input_size,
                                    )
                                    cur_g_num_input = cutlass.Int32(0)
                                    if topk_remaining == 0:
                                        self._collect_below_threshold_refine(
                                            tidx,
                                            threshold,
                                            offset,
                                            num_input,
                                            r_idx,
                                            s_input_idx,
                                            s_input_val,
                                            score,
                                            s_counter,
                                            s_indices,
                                            cur_g_num_input,
                                            None,
                                        )
                                        run_next_round = False
                                    else:
                                        self._filter_and_histogram_refine(
                                            tidx,
                                            threshold,
                                            offset,
                                            r_idx,
                                            is_last_round,
                                            num_input,
                                            cur_g_num_input,
                                            score,
                                            s_counter,
                                            s_indices,
                                            s_input_idx,
                                            s_input_val,
                                            s_num_input,
                                            s_histogram,
                                            s_last_remain,
                                            None,
                                            None,
                                        )
                            elif cutlass.const_expr(self.enable_bounded_spill):
                                # BOUNDED_SPILL: did_overflow means even the
                                # size-capped GMEM buffer filled -> REREAD-style
                                # re-scan (tier 3). Otherwise the candidates fit in
                                # SMEM + bounded GMEM -> GMEM_SPILL-style refine.
                                if did_overflow:
                                    run_next_round, chain_mask, chain_prefix = self._reread_gmem_rescan(
                                        topk_remaining,
                                        is_last_round,
                                        tidx,
                                        threshold_bin,
                                        threshold,
                                        offset,
                                        chain_mask,
                                        chain_prefix,
                                        score,
                                        s_counter,
                                        s_indices,
                                        s_last_remain,
                                        s_histogram,
                                        _copy_atom,
                                        scan_frag,
                                        _aligned_base,
                                        vec_start,
                                        aligned_size,
                                        row_start,
                                        prologue_elems,
                                        left_start,
                                        left_size,
                                    )
                                else:
                                    num_input = min(
                                        s_num_input[r_idx],
                                        self.filtered_topk_smem_input_size,
                                    )
                                    cur_g_num_input = g_num_input[r_idx]
                                    if topk_remaining == 0:
                                        self._collect_below_threshold_refine(
                                            tidx,
                                            threshold,
                                            offset,
                                            num_input,
                                            r_idx,
                                            s_input_idx,
                                            s_input_val,
                                            score,
                                            s_counter,
                                            s_indices,
                                            cur_g_num_input,
                                            buffer,
                                        )
                                        run_next_round = False
                                    else:
                                        self._filter_and_histogram_refine(
                                            tidx,
                                            threshold,
                                            offset,
                                            r_idx,
                                            is_last_round,
                                            num_input,
                                            cur_g_num_input,
                                            score,
                                            s_counter,
                                            s_indices,
                                            s_input_idx,
                                            s_input_val,
                                            s_num_input,
                                            s_histogram,
                                            s_last_remain,
                                            g_num_input,
                                            buffer,
                                        )
                            else:
                                num_input = min(s_num_input[r_idx], self.filtered_topk_smem_input_size)
                                cur_g_num_input = cutlass.Int32(0)
                                if cutlass.const_expr(self.enable_gmem_store):
                                    cur_g_num_input = g_num_input[r_idx]

                                if topk_remaining == 0:
                                    self._collect_below_threshold_refine(
                                        tidx,
                                        threshold,
                                        offset,
                                        num_input,
                                        r_idx,
                                        s_input_idx,
                                        s_input_val,
                                        score,
                                        s_counter,
                                        s_indices,
                                        cur_g_num_input,
                                        buffer,
                                    )
                                    run_next_round = False
                                else:
                                    self._filter_and_histogram_refine(
                                        tidx,
                                        threshold,
                                        offset,
                                        r_idx,
                                        is_last_round,
                                        num_input,
                                        cur_g_num_input,
                                        score,
                                        s_counter,
                                        s_indices,
                                        s_input_idx,
                                        s_input_val,
                                        s_num_input,
                                        s_histogram,
                                        s_last_remain,
                                        g_num_input,
                                        buffer,
                                    )
                            self._iket_end(_ik_round)
                    self._iket_end(_ik_refine)

            _ik_out = self._iket_begin("p5_output")
            # Phase 3: Output phase.
            # SP multi-CTA cluster: collect via DSMEM prefix scan (each CTA
            # writes only its slice). Solo / single-CTA: full-row writeback.
            if cutlass.const_expr(self.single_pass_multi_cta):
                if need_cluster_sync:
                    self._cluster_collect(
                        tidx,
                        s_indices,
                        s_counter,
                        s_last_remain,
                        s_histogram,
                        cta_in_group,
                        topk_remaining,
                        dst,
                        score,
                        dst_values,
                    )
                else:
                    self._phase3_writeback(
                        tidx,
                        row_start,
                        s_indices,
                        score,
                        indices,
                        dst,
                        dst_values,
                        dst_probs=dst_probs,
                        s_prob_red=s_prob_red,
                    )
            else:
                self._phase3_writeback(
                    tidx,
                    row_start,
                    s_indices,
                    score,
                    indices,
                    dst,
                    dst_values,
                    dst_probs=dst_probs,
                    s_prob_red=s_prob_red,
                )
            self._iket_end(_ik_out)


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    pad_to_vec_size: bool = False,
    vec_size: int = 8,
) -> torch.Tensor:
    """Create random logits tensor for testing.

    Args:
        row_starts: Tensor of shape (num_rows,) indicating the start position of each row
        row_ends: Tensor of shape (num_rows,) indicating the end position (exclusive) of each row
        dtype: Data type for the logits tensor
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (num_rows, max_row_length) with random values and -inf padding
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    num_rows = row_starts.shape[0]
    max_len = int(row_ends.max().item())
    if pad_to_vec_size:
        max_len = (max_len + vec_size - 1) // vec_size * vec_size

    # Generate random logits
    logits = torch.randn(num_rows, max_len, dtype=dtype, device="cuda")

    # Vectorized masking: set positions outside [row_start, row_end) to -inf
    col_indices = torch.arange(max_len, device="cuda").unsqueeze(0)  # (1, max_len)
    mask_lo = col_indices < row_starts.unsqueeze(1)  # positions before row_start
    mask_hi = col_indices >= row_ends.unsqueeze(1)  # positions at or after row_end
    mask = mask_lo | mask_hi  # positions outside valid range
    logits[mask] = float("-inf")

    return logits


def run_reference_top_k(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    index_topk: int,
) -> torch.Tensor:
    """Return local top-k indices for each ``[row_start, row_end)`` span."""
    num_selected = min(index_topk, logits.shape[1])
    absolute_indices = logits.topk(num_selected, dim=-1).indices
    valid = (absolute_indices >= row_starts[:, None]) & (absolute_indices < row_ends[:, None])
    local_indices = absolute_indices - row_starts[:, None]
    return local_indices.masked_fill(~valid, -1)


def compare_top_k_results(
    logits: torch.Tensor,
    cuda_indices: torch.Tensor,
    torch_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare results from CUDA top_k_per_row with torch.topk.
    Handles different shapes and -1 placeholders in cuda_indices.

    Args:
        logits: Input logits tensor [num_rows, vocab_size]
        cuda_indices: CUDA implementation output [num_rows, cuda_k], may contain -1
        torch_indices: PyTorch reference output [num_rows, torch_k], may contain -1
        row_starts: Start positions for each row [num_rows]
        row_ends: End positions for each row [num_rows]
        top_k: Target top-k value
        tolerance: Tolerance for floating point comparison

    Returns:
        True if results match within tolerance, False otherwise
    """
    num_rows = cuda_indices.shape[0]

    # Calculate valid lengths for each row (vectorized)
    row_lengths = row_ends - row_starts

    # For each row, compare only the valid indices (non -1)
    for row_idx in range(num_rows):
        row_len = row_lengths[row_idx].item()
        expected_valid = min(row_len, top_k)

        # Get valid indices from both implementations (filter out -1)
        cuda_row = cuda_indices[row_idx]
        torch_row = torch_indices[row_idx]

        # Filter out -1 (invalid) indices
        cuda_valid_mask = cuda_row != -1
        torch_valid_mask = torch_row != -1

        cuda_valid = cuda_row[cuda_valid_mask]
        torch_valid = torch_row[torch_valid_mask]

        # Check if the number of valid indices matches
        if cuda_valid.shape[0] != torch_valid.shape[0]:
            print(f"Row {row_idx}: Different number of valid indices - " f"CUDA: {cuda_valid.shape[0]}, PyTorch: {torch_valid.shape[0]}")
            return False

        if cuda_valid.shape[0] != expected_valid:
            print(f"Row {row_idx}: Expected {expected_valid} valid indices, got {cuda_valid.shape[0]}")
            return False

        # If no valid indices, continue
        if cuda_valid.shape[0] == 0:
            continue

        # Gather the corresponding logit values
        row_start = row_starts[row_idx].item()
        logits_row = logits[row_idx]

        # Adjust indices to absolute positions (add row_start offset)
        cuda_abs_indices = cuda_valid + row_start
        torch_abs_indices = torch_valid + row_start

        # Get logit values for the selected indices
        cuda_values = logits_row[cuda_abs_indices]
        torch_values = logits_row[torch_abs_indices]

        # Sort both value arrays in descending order
        cuda_values_sorted, _ = torch.sort(cuda_values, descending=True)
        torch_values_sorted, _ = torch.sort(torch_values, descending=True)

        # Compare sorted values
        if not torch.allclose(cuda_values_sorted, torch_values_sorted, rtol=tolerance, atol=tolerance):
            # Additional debug: check if sets are identical
            cuda_set = set(cuda_valid.cpu().tolist())
            torch_set = set(torch_valid.cpu().tolist())
            print(f"row_idx: {row_idx}, row_len: {row_len}, expected_valid: {expected_valid}")
            print(f"cuda_values_sorted: {cuda_values_sorted}")
            print(f"torch_values_sorted: {torch_values_sorted}")
            if cuda_set != torch_set:
                print("  Different indices selected:")
                print(f"    Only in CUDA: {cuda_set - torch_set}")
                print(f"    Only in Torch: {torch_set - cuda_set}")

            return False

    return True
