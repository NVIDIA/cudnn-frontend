# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""BF16 FC2 two-half direct-store kernel, derived from paired SwiGLU.

Credit: NVIDIA Frost; Kernel Factory optimizer; Yanqin Zhai's
NVIDIA/cudnn-frontend PR1090 weight-M/token-N orientation; NVIDIA CUTLASS
example113 layout; canonical rank5 pairing guidance and TRT-LLM gated-row
interleaving motivation. No TRT-LLM kernel body is reused.

The graph engine enforces R<=8, full output channels divisible by128,
K divisible by64, and positive 16-byte-aligned weight strides. The retained generated scaffold is intentionally not refactored to retain parity with the independently audited kernel.

Original template: sm100 MoE grouped matmul fwd: grouped persistent scheduler + per-group A TMA
descriptor replacement.

Serves both MMA modes; ``cta_group`` is an injected tile constant.

  cta_group=1  single-CTA MMA — every CTA runs its own MMA on its own
               SMEM/TMEM, no leader-follower pair.
  cta_group=2  2-CTA MMA cluster pair — the leader CTA issues the MMA while the
               follower consumes the scheduler ring only.

Blocks that genuinely differ between the two MMA modes are expressed with
``cutlass.const_expr(cta_group == N)`` or, where an arm is not statement-shaped,
with ``@@CTA{1,2}_ONLY@@`` marker regions that the renderer strips.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
from cudnn.gemm.frost.sm100.kernel_templates._tile_helpers import (
    reset_moe_sched_counter as _reset_moe_sched_counter,
    copy_tensormap_to_workspace as _copy_tensormap_to_workspace,
    epi_subtile_spans as _epi_subtile_spans,
    fence_tensormap_acquire as _fence_tensormap_acquire,
    fence_tensormap_release as _fence_tensormap_release,
    moe_group_at as _moe_group_at,
    moe_swizzle_tile as _moe_swizzle_tile,
    moe_load_sched_word as _moe_load_sched_word,
    replace_tensormap_global_address as _replace_tensormap_global_address,
    replace_tensormap_global_dim_0 as _replace_tensormap_global_dim_0,
    replace_tensormap_global_dim_1 as _replace_tensormap_global_dim_1,
    tcgen05_alloc as _tcgen05_alloc,
    tcgen05_dealloc as _tcgen05_dealloc,
    tcgen05_mma as _tcgen05_mma,
    TENSOR_MAP_QWORDS,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass
from cudnn.frost.compiled_cache import compile_cached as _compile_cached
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda

# A TMA tensormap is 128 bytes = 16 int64 qwords.
moe_static_sched = False  # Public SCHED_POLICY: 0 dynamic (default), 1 static.
moe_absolute_a = False  # Set before injection so the rendered eligibility gate wins.
moe_wide_mma = False  # Shared-A pair packed into one doubled-N instruction.

# Tile config: CONFIG_sm100_64x64x128_64x64x32_cluster1x1_1ctamma
mma_inst_shape_mnk = (128, 8, 16)
mma_k_dim = 0
cta_group = 1
cgrp_tile_mnk = (8, 64, 64)
cta_tile_mnk = (8, 64, 64)
epi_tile_mn = (8, 64)
threads_per_cta = 256
cluster_shape_mnk = (1, 1, 1)
matmul_a_batch = 1
matmul_b_batch = 1
a_is_m_major = False
b_is_n_major = False
mma_a_major = 0
mma_b_major = 0
ab_stages = 13
b_collector_ok = False
multicast_a = False
multicast_b = False
a_mcast_slices = 1
a_tma_box_m = 8
b_mcast_slices = 1
ab_empty_full_mask = False
ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B
a_smem_desc_leading_byte_offset = 16
a_smem_desc_stride_byte_offset = 1024
a_smem_k_step_bytes = 32
a_smem_m_step_bytes = 8192
a_tma_group_elems = 1
b_smem_desc_leading_byte_offset = 16
b_smem_desc_stride_byte_offset = 1024
b_smem_k_step_bytes = 32
b_tma_group_elems = 1
mma_size_m = 1
mma_size_n = 1
mma_size_k = 4
ab_tma_swizzle = _tma.TensorMapSwizzle.s128b

# Dtype family: A=bf16->MMAbf16, B=bf16->MMAbf16, out=bf16 (K_BYTES=128)
ab_dtype = cutlass.BFloat16
cd_dtype = cutlass.BFloat16
epi_store_dtype = cutlass.BFloat16
mma_a_dtype = cutlass.BFloat16
mma_b_dtype = cutlass.BFloat16
mma_c_dtype = cutlass.Float32
acc_widen_to_fp32 = False
ab_tma_dtype = cutlass.BFloat16
mma_kind = nvvm.Tcgen05MMAKind.F16
epi_n = 8
epi_row_elems = 64
tile_swizzle_n = 0
swizzle_l2_budget_bytes = FROST_TEMPLATE_PARAMS.l2_budget_bytes
num_gemms = 1
num_a_operands = 1
num_b_operands = 1
gemm_a_idx = (0,)
gemm_b_idx = (0,)
num_tmem_alloc_cols = 512
tmem_alloc_exclusive = False
acc_stages = 2  # two independent M128xN8 accumulator tiles
grid_num_clusters = FROST_TEMPLATE_PARAMS.grid_ctas
offset_cutlass_dtype = cutlass.Int32
vec_bytes_epi = 32
split_k_slices = 1
frost_compile_options = "--enable-tvm-ffi --gpu-arch sm_100a"
n_tma_outputs = 1
moe_aligned_offsets = False
moe_token_alignment = 1
epi_slot_widen = 1
epi_packed_lanes = True
epi_dp22 = False
epi_stage_rows = 8
epi_chunk_elems = 8
ab_stages = 12  # 12 * (M128xK64 weights + N8xK64 tokens) = 208,896B
fallback_cluster_shape_mnk = None
mixed_a_pattern_pref = 1
mixed_b_pattern_pref = 1
mixed_a_pattern_fb = 1
mixed_b_pattern_fb = 1
moe_static_sched = True
moe_absolute_a = True
moe_wide_mma = True
moe_blocked_b = False

# Tensormap workspace slots per CTA: the A operands, plus the output descriptor
# when the TMA-store epilogue re-dimensions it per routed group.
moe_desc_slots = num_a_operands + n_tma_outputs
# The rank-5 weight TMA assembles gate and up into one physical MMA-M operand.
b_stage_multiplier = 1
mma_issue_gemms = 1 if moe_wide_mma else num_gemms
_CTA_GROUP = nvvm.CTAGroup.CTA_2 if cta_group == 2 else nvvm.CTAGroup.CTA_1


# Scheduler ring depth.
SCHED_STAGES = 2
SCHED_BCAST_STAGES = 2

# Number of i32 slots per scheduler ring stage.
SCHED_SLOT_WORDS = 8

# Programmatic Dependent Launch (PDL, sm_90+).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path.
# Four-deep TMA-store ring keeps up to three completed subtiles in flight.
EPI_SMEM_STAGES = 4

# Named barrier id for the 4-warp epilogue handoff around the TMA store.
EPI_SYNC_BAR_ID = 1

# Named barrier id for the TMEM-alloc handoff.
TMEM_ALLOC_BARRIER_ID = 2


@cute.jit
def _moe_auto_swizzle_w(group_rows, n, k, nt_n):
    """N-super-block width for one routed group, resolved per group.

    Same rule as the dense path, but the "M side" is THIS group's token slice, not the
    whole token tensor: block along the shorter of (group tokens, expert weight), capped
    by what L2 can hold onto. A group spanning one m-tile makes both orders identical.
    """
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_dtype.width) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    rows = cutlass.Int64(group_rows)
    if cutlass.min(rows, n) * row_bytes <= budget and rows <= n:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


def _a_collector_op(g):
    if cutlass.const_expr(moe_wide_mma or num_gemms == 1 or num_a_operands != 1 or mma_size_m != 1):
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


@cute.kernel
def frost_sm100_moe_fc2_pair_m128n8k16_sched_static_s12_early_pdl(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    num_experts: cutlass.Int32,
    num_groups: cutlass.Int32,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_1: cutlass.GridConstant[_tma.TensorMap],
    mA_0: cute.Tensor,
    mC_0: cute.Tensor,
    a_stride_m_0: cutlass.Int64,
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_l_0: cutlass.Int64,
    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],
) -> None:
    tma_a_descs = [tma_a_desc_0]
    tma_b_descs = [tma_b_desc_0]
    mA_list = [mA_0]
    a_stride_m_list = [a_stride_m_0]
    tma_c_descs = [tma_c_desc_0]
    tma_c_m_major = (False,)

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
    # Allow the dependent FC2 prologue to occupy otherwise idle SMs early.
    # This does not release data dependencies: FC2 retains griddepcontrol.wait,
    # which waits for this complete grid and its global-memory operations.
    # NVIDIA PTX ISA, griddepcontrol: repeated triggers in a CTA are harmless.
    if cutlass.const_expr(USE_PDL):
        if tidx == 0:
            nvvm.griddepcontrol("launch_dependents")
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    gridx = cute.arch.grid_dim()[0]

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    m_rank = cta_rank_in_cluster % cluster_m
    n_rank = cta_rank_in_cluster // cluster_m
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

    sched_counter_ptr = cute.make_ptr(
        cutlass.Int32,
        (a_tma_workspace.iterator.raw_ptr() + grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
        mem_space=cute.AddressSpace.generic,
    )

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

    a_pattern = 0
    for n_idx in cutlass.range_constexpr(cluster_n):
        a_pattern = a_pattern | (1 << (n_idx * cluster_m))
    if cutlass.const_expr(cta_group == 1):
        b_pattern = (1 << cluster_m) - 1
    else:
        b_pattern = 0
        for pm_idx in cutlass.range_constexpr(cluster_m // 2):
            b_pattern = b_pattern | (1 << (pm_idx * 2))

    if cutlass.const_expr(multicast_a):
        if cutlass.const_expr(cta_group == 1):
            tma_mcast_mask_a = cutlass.Int16(a_pattern) << m_rank
        else:
            tma_mcast_mask_a = cutlass.Int16(a_pattern << m_rank)
    else:
        if cutlass.const_expr(cta_group == 1):
            tma_mcast_mask_a = cutlass.Int16(1) << cta_rank_in_cluster
        else:
            tma_mcast_mask_a = cutlass.Int16(1 << cta_rank_in_cluster)
    if cutlass.const_expr(cta_group == 1):
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
        else:
            tma_mcast_mask_b = cutlass.Int16(1) << cta_rank_in_cluster

        a_part_arrive = cutlass.Int16(a_pattern) << m_rank
        b_part_arrive = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
        if cutlass.const_expr(ab_empty_full_mask):
            ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
        else:
            ab_empty_arrive_mask = a_part_arrive | b_part_arrive
    else:
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16((b_pattern << pair_member) << (n_rank * cluster_m))
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

    sched_storage = cutlass.Array(
        cutlass.Int32,
        SCHED_STAGES * SCHED_SLOT_WORDS,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    sched_full_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sched_empty_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    if cutlass.const_expr(not moe_static_sched):
        sched_bcast_slot = cutlass.Array(cutlass.Int32, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=16)
        sched_bcast_full_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
        sched_bcast_empty_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=8)

    # Physical MMA operands are transposed relative to the logical output:
    # A is 128 gate/up weight rows, B is 8 routed token rows.
    sA_elems = mma_inst_shape_mnk[0] * cta_tile_mnk[2]
    sB_elems = mma_inst_shape_mnk[1] * cta_tile_mnk[2]
    smem_a_list = [
        cutlass.Array(
            ab_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = []
    if cutlass.const_expr(moe_wide_mma):
        smem_b_all = cutlass.Array(
            ab_dtype,
            sB_elems * num_b_operands * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        smem_b_list = [smem_b_all.subview(sB_elems * j) for j in range(num_b_operands)]
    else:
        smem_b_list = [
            cutlass.Array(
                ab_dtype,
                sB_elems * ab_stages,
                space=cutlass.AddressSpace.smem,
                alignment=1024,
            )
            for _ in range(num_b_operands)
        ]

    tma_a_desc_smem_list = []
    if cutlass.const_expr(not (moe_aligned_offsets or moe_absolute_a)):
        tma_a_desc_smem_list = [
            cutlass.Array(
                cutlass.Int64,
                TENSOR_MAP_QWORDS,
                space=cutlass.AddressSpace.smem,
                alignment=128,
            )
            for _ in range(num_a_operands)
        ]

    # One block per MMA mode, every count defined in both: the 2-CTA pair
    # releases per PAIR (so ab_empty counts pairs and acc_empty counts both
    # CTAs' epilogue warps), the 1-CTA one per CTA.
    if cutlass.const_expr(cta_group == 2):
        ab_empty_count = cluster_size // cta_group if cutlass.const_expr(ab_empty_full_mask) else (cluster_m // cta_group) + cluster_n - 1
        acc_empty_count = num_epilogue_warps * 2
    else:
        ab_empty_count = cluster_size if cutlass.const_expr(ab_empty_full_mask) else cluster_m + cluster_n - 1
        acc_empty_count = num_epilogue_warps
    sched_empty_count = 1 + 1 + num_epilogue_warps
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
        for i in range(SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(sched_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(sched_empty_mbar_ptr.subview(i), sched_empty_count)
        if cutlass.const_expr(not moe_static_sched):
            for i in range(SCHED_BCAST_STAGES):
                if elect_one:
                    nvvm.mbarrier_init(sched_bcast_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(sched_bcast_empty_mbar_ptr.subview(i), cluster_size)
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

    idesc = cutlass.experimental.primitives.Tcgen05InstrDesc.build(
        a_dtype=mma_a_dtype,
        b_dtype=mma_b_dtype,
        c_dtype=mma_c_dtype,
        n_dim=mma_inst_shape_mnk[1],
        m_dim=mma_inst_shape_mnk[0],
        k_dim=mma_k_dim,
        a_major=mma_a_major,
        b_major=mma_b_major,
    )

    if cutlass.const_expr(cta_group == 1):
        # TMEM accumulator layout, per acc stage: gemm g, M block mi, N block ni
        # -> columns [g*cols_per_acc_stage + mi*epi_cols_per_mma_m + ni*mma_inst_n, +N),
        # all at TMEM lane base 0.
        epi_cols_per_mma_m = mma_inst_shape_mnk[1]
    else:
        pair_n_size = cgrp_tile_mnk[1] // cluster_n
        # Per-CTA output rows one MMA-M block covers (the pair splits M).
        epi_rows_per_mma_m = mma_inst_shape_mnk[0] // mma_size_m
        if cutlass.const_expr(epi_rows_per_mma_m == 64):
            # cluster-MMA m=128: the pair also splits N, so each CTA drains N/2.
            epi_cols_per_mma_m = pair_n_size // 2
        else:
            epi_cols_per_mma_m = pair_n_size
        # N is NOT a sub-block axis (the CTA tile is never split along N).
    cols_per_acc_stage = mma_size_m * epi_cols_per_mma_m
    acc_region_cols = num_gemms * cols_per_acc_stage
    if cutlass.const_expr(cta_group == 1):
        epi_rows_per_mma_m = cta_tile_mnk[0] // mma_size_m
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32
    if cutlass.const_expr(cta_group == 2):

        nvvm.barrier_cluster_wait()
        nvvm.barrier_cta_sync(0)

    pass

    vsize = epi_chunk_elems

    M = m
    N = n
    clusters_along_n = cute.ceil_div(cutlass.Int32(N), cgrp_tile_mnk[1])
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    num_k_blocks = cta_tile_mnk[2] // mma_inst_shape_mnk[2]
    first_token_arr = cutlass.make_array_view(first_token_offset)

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        # PDL launch completion alone does not make predecessor writes visible.
        # The scheduler consumes the reset counter and live offsets itself.
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0
        lane = cute.arch.lane_idx()
        gemm_s = cutlass.Int32(M)
        sched_stage = cutlass.Int32(0)
        sched_empty_phase = cutlass.Int32(1)
        bcast_stage = cutlass.Int32(0)
        bcast_full_phase = cutlass.Int32(0)
        bcast_empty_phase = cutlass.Int32(1)
        last_bcast_stage = cutlass.Int32(0)
        last_bcast_empty_done_phase = cutlass.Int32(0)
        linear_idx = cutlass.Int32(0)
        if cutlass.const_expr(moe_static_sched):
            linear_idx = cutlass.Int32(bidx // cluster_m)
        start_linear_idx = cutlass.Int32(0)
        total_tiles = cutlass.Int32(0)
        scan_base = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        group_begin = cutlass.Int32(0)
        group_end = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)

        while is_tile_valid != 0:
            if cutlass.const_expr(not moe_static_sched):
                # Dynamic tile assignment: the cluster leader claims the next GLOBAL
                # tile index and broadcasts it, so the clusters live at any instant
                # sit in one contiguous window of tile space and share L2. Static
                # striding lets them drift apart and share nothing.
                if cta_rank_in_cluster == 0:
                    while not nvvm.mbarrier_try_wait_parity(
                        sched_bcast_empty_mbar_ptr.subview(bcast_stage),
                        bcast_empty_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    claimed = cutlass.Int32(0)
                    if lane == 0:
                        claimed = nvvm.atomicrmw(
                            "add",
                            sched_counter_ptr,
                            cutlass.Int32(1),
                            mem_order="relaxed",
                            syncscope="gpu",
                        )
                    claimed = nvvm.shfl_sync(full_warp_mask, claimed, 0, shfl_idx_clamp, nvvm.Shfl.IDX)
                    if lane < cluster_size:
                        (nvvm.mapa(sched_bcast_slot.subview(bcast_stage), lane)).store(claimed)
                        nvvm.mbarrier_arrive(nvvm.mapa(sched_bcast_full_mbar_ptr.subview(bcast_stage), lane))
                while not nvvm.mbarrier_try_wait_parity(
                    sched_bcast_full_mbar_ptr.subview(bcast_stage),
                    bcast_full_phase,
                    time_limit=10_000_000,
                ):
                    pass
                linear_idx = _moe_load_sched_word((sched_bcast_slot.subview(bcast_stage)))
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if lane == 0:
                    nvvm.mbarrier_arrive(nvvm.mapa(sched_bcast_empty_mbar_ptr.subview(bcast_stage), 0))
                if cutlass.const_expr(cluster_size > 1):
                    # The final invalid broadcast has no next reuse of this stage to
                    # wait for its cluster-wide acknowledgements, so retain its exact
                    # stage and completion parity for the scheduler-warp drain below.
                    last_bcast_stage = bcast_stage
                    last_bcast_empty_done_phase = bcast_empty_phase ^ 1
                bcast_stage += 1
                if bcast_stage == SCHED_BCAST_STAGES:
                    bcast_stage = cutlass.Int32(0)
                    bcast_full_phase = bcast_full_phase ^ 1
                    bcast_empty_phase = bcast_empty_phase ^ 1
            if linear_idx >= start_linear_idx + total_tiles:
                is_search_live = cutlass.Int32(1)
                while is_search_live != 0:
                    visit_idx = scan_base + lane
                    my_group = _moe_group_at(visit_idx, num_groups, num_experts)
                    my_begin = cutlass.Int32(0)
                    my_end = cutlass.Int32(0)
                    my_tiles = cutlass.Int32(0)
                    if visit_idx < num_groups:
                        if my_group != 0:
                            my_begin = cutlass.Int32(first_token_arr[my_group])
                        if my_group + 1 < num_groups:
                            my_end = cutlass.Int32(first_token_arr[my_group + 1])
                        else:
                            my_end = gemm_s
                        my_tiles = cute.ceil_div(my_end - my_begin, cgrp_tile_mnk[0]) * clusters_along_n
                    prefix_tiles = my_tiles
                    for delta in (1, 2, 4, 8, 16):
                        prefix_delta = nvvm.shfl_sync(
                            full_warp_mask,
                            prefix_tiles,
                            delta,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        if lane >= delta:
                            prefix_tiles += prefix_delta
                    my_start = start_linear_idx + prefix_tiles - my_tiles
                    thread_succeed = nvvm.vote_sync(
                        full_warp_mask,
                        linear_idx < my_start + my_tiles,
                        nvvm.VoteSync.BALLOT,
                    )
                    if thread_succeed != 0:
                        winning_lane = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(thread_succeed)).to(cutlass.Int32)
                        scan_base = nvvm.shfl_sync(full_warp_mask, visit_idx, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_idx = nvvm.shfl_sync(full_warp_mask, my_group, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_begin = nvvm.shfl_sync(full_warp_mask, my_begin, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_end = nvvm.shfl_sync(full_warp_mask, my_end, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        start_linear_idx = nvvm.shfl_sync(full_warp_mask, my_start, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        total_tiles = nvvm.shfl_sync(full_warp_mask, my_tiles, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        is_search_live = cutlass.Int32(0)
                    else:
                        start_linear_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            my_start + my_tiles,
                            31,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        scan_base += 32
                        if scan_base >= num_groups:
                            is_tile_valid = cutlass.Int32(0)
                            is_search_live = cutlass.Int32(0)

            coord_expert = cutlass.Int32(0)
            cluster_tile_m = cutlass.Int32(0)
            coord_n = cutlass.Int32(0)
            if is_tile_valid != 0:
                local_linear_idx = linear_idx - start_linear_idx
                group_nt_m = total_tiles // clusters_along_n
                cluster_tile_m, coord_n = _moe_swizzle_tile(
                    local_linear_idx,
                    group_nt_m,
                    clusters_along_n,
                    _moe_auto_swizzle_w(group_nt_m * cgrp_tile_mnk[0], N, k, clusters_along_n),
                )
                coord_expert = group_idx % num_experts

            while not nvvm.mbarrier_try_wait_parity(
                sched_empty_mbar_ptr.subview(sched_stage),
                sched_empty_phase,
                time_limit=10_000_000,
            ):
                pass
            if lane == 0:
                slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
                (slot.subview(0)).store(coord_expert)
                (slot.subview(1)).store(cluster_tile_m)
                (slot.subview(2)).store(coord_n)
                (slot.subview(3)).store(is_tile_valid)
                (slot.subview(4)).store(group_begin)
                (slot.subview(5)).store(group_end)
                (slot.subview(7)).store(group_idx)
                nvvm.mbarrier_arrive(sched_full_mbar_ptr.subview(sched_stage))

            if cutlass.const_expr(moe_static_sched):
                linear_idx += grid_num_clusters
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_empty_phase = sched_empty_phase ^ 1

        # Every multi-CTA cluster emits one invalid scheduler record before
        # leaving the loop. Keep the leader CTA's DSM alive until every
        # scheduler warp has consumed that final broadcast and acknowledged the
        # leader slot. A singleton cluster has no remote DSM lifetime to drain.
        if cutlass.const_expr(not moe_static_sched and cluster_size > 1):
            if cta_rank_in_cluster == 0:
                # Each acknowledgement flips the saved pre-wrap empty phase, so
                # pre-wrap ``bcast_empty_phase ^ 1`` is the completed parity.
                while not nvvm.mbarrier_try_wait_parity(
                    sched_bcast_empty_mbar_ptr.subview(last_bcast_stage),
                    last_bcast_empty_done_phase,
                    time_limit=10_000_000,
                ):
                    pass

    if warp_idx == tma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ab_empty_phase_bit = cutlass.Int32(1)
        ab_iter = cutlass.Int32(0)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        if cutlass.const_expr(cta_group == 2):
            logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_n

        lane = tidx % 32
        block_linear = bidx + bidy * gridx
        cta_desc_base_list = [a_tma_workspace.iterator.raw_ptr() + (block_linear * moe_desc_slots + _ai) * TENSOR_MAP_QWORDS for _ai in range(num_a_operands)]
        a_desc_tma_ptr_list = [
            cute.make_ptr(
                cutlass.Int64,
                cta_desc_base_list[_ai].toint(),
                mem_space=cute.AddressSpace.generic,
            )
            for _ai in range(num_a_operands)
        ]
        previous_group_begin = cutlass.Int32(-1)
        if cutlass.const_expr(moe_aligned_offsets or moe_absolute_a):
            a_desc_load_list = [tma_a_descs[_ai].get_ptr() for _ai in range(num_a_operands)]
        else:
            a_desc_load_list = a_desc_tma_ptr_list
        if cutlass.const_expr(not (moe_aligned_offsets or moe_absolute_a)):
            if elect_one:
                for _ai in cutlass.range_constexpr(num_a_operands):
                    _copy_tensormap_to_workspace(tma_a_descs[_ai].get_ptr(), tma_a_desc_smem_list[_ai])
        nvvm.bar_warp_sync(0xFFFFFFFF)

        while is_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr.subview(sched_stage),
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            coord_expert = _moe_load_sched_word((slot.subview(0)))
            tile_m = _moe_load_sched_word((slot.subview(1)))
            tile_n = _moe_load_sched_word((slot.subview(2)))
            is_valid = _moe_load_sched_word((slot.subview(3)))
            group_begin = _moe_load_sched_word((slot.subview(4)))
            group_end = _moe_load_sched_word((slot.subview(5)))
            # Converge consumers before releasing their scheduler slot.
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_group = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                if cutlass.const_expr(moe_aligned_offsets or moe_absolute_a):
                    coord_m_desc = group_begin + coord_m_group
                else:
                    coord_m_desc = coord_m_group
                if cutlass.const_expr(cta_group == 1):
                    coord_n_per_cta = tile_n * cgrp_tile_mnk[1] + n_rank * cta_tile_mnk[1]
                else:
                    coord_n_per_cta = tile_n * cgrp_tile_mnk[1] + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]

                if cutlass.const_expr(not (moe_aligned_offsets or moe_absolute_a)):
                    if group_begin != previous_group_begin:
                        previous_group_begin = group_begin
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            _fence_tensormap_acquire(a_desc_tma_ptr_list[_ai])
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            if elect_one:
                                row_base = mA_list[_ai].iterator.raw_ptr() + group_begin * a_stride_m_list[_ai]
                                _replace_tensormap_global_address(tma_a_desc_smem_list[_ai], row_base.toint())
                                _replace_tensormap_global_dim_1(tma_a_desc_smem_list[_ai], group_end - group_begin)
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                            if lane < TENSOR_MAP_QWORDS:
                                (cta_desc_base_list[_ai] + lane).store((tma_a_desc_smem_list[_ai].subview(lane)).load())
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                            _fence_tensormap_release()

                for k_tile_idx in range(num_k_tiles):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        ab_empty_mbar_ptr.subview(stage),
                        ab_empty_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass

                    coord_k = k_tile_idx * cta_tile_mnk[2]
                    if cutlass.const_expr(cta_group == 1):
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)

                    if cutlass.const_expr(cta_group == 2):
                        if is_pair_leader:
                            if elect_one:
                                nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                    a_issue = (not multicast_a) or (n_rank == 0)
                    if cutlass.const_expr(a_mcast_slices > 1):
                        a_data_issue = True
                        _a_off = n_rank * (cta_tile_mnk[0] // a_mcast_slices)
                    else:
                        a_data_issue = a_issue
                        _a_off = 0
                    if a_data_issue:
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            sA_stage = smem_b_list[_ai].subview(sB_elems * stage)
                            for _am in cutlass.range_constexpr(cta_tile_mnk[0] // a_mcast_slices // a_tma_box_m):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(_a_off * cta_tile_mnk[2] + _am * a_tma_box_m * cta_tile_mnk[2]),
                                        a_desc_load_list[_ai],
                                        (coord_k, coord_m_desc + _a_off + _am * a_tma_box_m, cutlass.Int32(0)),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=_CTA_GROUP,
                                    )
                    b_issue = (not multicast_b) or (pair_m_idx == 0)
                    if cutlass.const_expr(b_mcast_slices > 1):
                        b_data_issue = True
                        _b_off = pair_m_idx * (cta_tile_mnk[1] // b_mcast_slices)
                    else:
                        b_data_issue = b_issue
                        _b_off = 0
                    if b_data_issue:
                        for _bj in cutlass.range_constexpr(num_b_operands):
                            sB_stage = smem_a_list[_bj].subview(sA_elems * stage)
                            if cutlass.const_expr(b_is_n_major):
                                for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage.subview(n_group * b_tma_group_elems * cta_tile_mnk[2]),
                                            tma_b_descs[_bj].get_ptr(),
                                            (
                                                coord_n_per_cta + n_group * b_tma_group_elems,
                                                coord_k,
                                                coord_expert,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=_CTA_GROUP,
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(_b_off * cta_tile_mnk[2]),
                                        tma_b_descs[_bj].get_ptr(),
                                        (
                                            coord_k,
                                            cutlass.Int32(0),
                                            cutlass.Int32(0),
                                            (coord_n_per_cta + _b_off) // 8,
                                            coord_expert,
                                        ),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=_CTA_GROUP,
                                    )
                    ab_iter += 1

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

    if cutlass.const_expr(cta_group == 2):
        pair_mask = cutlass.Int16(3) << pair_leader_rank
        a_arrive_pattern = 0
        for n_idx in cutlass.range_constexpr(cluster_n):
            a_arrive_pattern = a_arrive_pattern | (1 << (n_idx * cluster_m))
        b_arrive_pattern = 0
        for m_idx in cutlass.range_constexpr(cluster_m):
            b_arrive_pattern = b_arrive_pattern | (1 << m_idx)
        a_part = a_arrive_pattern << m_rank
        a_part = a_part | (a_part << 1)
        b_part = b_arrive_pattern << (n_rank * cluster_m)
        if cutlass.const_expr(ab_empty_full_mask):
            ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
        else:
            ab_empty_arrive_mask = cutlass.Int16(a_part | b_part)
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
        if cutlass.const_expr(cta_group == 1):
            ab_full_phase_bit = cutlass.Int32(0)
            ab_iter = cutlass.Int32(0)
            acc_empty_phase_bit = cutlass.Int32(1)
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            sched_stage = cutlass.Int32(0)
            sched_full_phase = cutlass.Int32(0)
            acc_stage = cutlass.Int32(0)
            # Per-group TMA replacement changes the GMEM source, not these invariant
            # MMA-side SMEM descriptor roots.
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
                while not nvvm.mbarrier_try_wait_parity(
                    sched_full_mbar_ptr.subview(sched_stage),
                    sched_full_phase,
                    time_limit=10_000_000,
                ):
                    pass
                is_valid = _moe_load_sched_word((sched_storage.subview(sched_stage * SCHED_SLOT_WORDS).subview(3)))
                # Converge consumers before releasing their scheduler slot.
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
                sched_stage += 1
                if sched_stage == SCHED_STAGES:
                    sched_stage = cutlass.Int32(0)
                    sched_full_phase = sched_full_phase ^ 1

                if is_valid != 0:
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

                    scale_d = cutlass.Boolean(False)
                    for k_tile_idx in range(num_k_tiles):
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
                            for g in cutlass.range_constexpr(mma_issue_gemms):
                                desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * stage + a_smem_k_step_bytes * k_block_idx)
                                desc_b_k = desc_b_roots[gemm_b_idx[g]].advance_start_address(
                                    sB_bytes * stage * b_stage_multiplier + b_smem_k_step_bytes * k_block_idx
                                )
                                for mi in cutlass.range_constexpr(mma_size_m):
                                    # The M sub-block offset is a whole SMEM swizzle atom,
                                    # so the descriptor's swizzle phase is preserved.
                                    desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                                    if elect_one:
                                        _tcgen05_mma(
                                            mma_kind,
                                            _CTA_GROUP,
                                            tmem_addr_mmas[g][mi],
                                            desc_a,
                                            desc_b_k,
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
                            group=_CTA_GROUP,
                        )
                    tile_iter += 1

            if cutlass.const_expr(USE_PDL):
                nvvm.griddepcontrol("launch_dependents")

            if tile_iter != 0:
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
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            _tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=_CTA_GROUP,
            )
        else:
            peer_cta_rank = cta_rank_in_cluster ^ 1
            if is_pair_leader:
                ab_full_phase_bit = cutlass.Int32(0)
                ab_iter = cutlass.Int32(0)
                acc_empty_phase_bit = cutlass.Int32(1)
                tile_iter = cutlass.Int32(0)
                is_valid = cutlass.Int32(1)
                sched_stage = cutlass.Int32(0)
                sched_full_phase = cutlass.Int32(0)
                acc_stage = cutlass.Int32(0)
                # Per-group TMA replacement changes the GMEM source, not these
                # invariant MMA-side SMEM descriptor roots.
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
                    while not nvvm.mbarrier_try_wait_parity(
                        sched_full_mbar_ptr.subview(sched_stage),
                        sched_full_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    is_valid = _moe_load_sched_word((sched_storage.subview(sched_stage * SCHED_SLOT_WORDS).subview(3)))
                    # Converge consumers before releasing their scheduler slot.
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
                    sched_stage += 1
                    if sched_stage == SCHED_STAGES:
                        sched_stage = cutlass.Int32(0)
                        sched_full_phase = sched_full_phase ^ 1

                    if is_valid != 0:
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
                        # One accumulator per (gemm, M block). Column arithmetic stays on the
                        # encoded (row << 16) | col integer.
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

                        scale_d = cutlass.Boolean(False)
                        for k_tile_idx in range(num_k_tiles):
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
                                for g in cutlass.range_constexpr(mma_issue_gemms):
                                    desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * stage + a_smem_k_step_bytes * k_block_idx)
                                    desc_b = desc_b_roots[gemm_b_idx[g]].advance_start_address(
                                        sB_bytes * stage * b_stage_multiplier + b_smem_k_step_bytes * k_block_idx
                                    )
                                    for mi in cutlass.range_constexpr(mma_size_m):
                                        # The M sub-block offset is a whole SMEM swizzle atom, so the
                                        # descriptor's swizzle phase is preserved. B is shared.
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
                                # k_block of the tile, so the flip stays outside mi/ni.
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
                                multicast_mask=pair_mask,
                                group=_CTA_GROUP,
                            )
                        tile_iter += 1

                if cutlass.const_expr(USE_PDL):
                    nvvm.griddepcontrol("launch_dependents")

                if tile_iter != 0:
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
                is_valid = cutlass.Int32(1)
                sched_stage = cutlass.Int32(0)
                sched_full_phase = cutlass.Int32(0)
                while is_valid != 0:
                    while not nvvm.mbarrier_try_wait_parity(
                        sched_full_mbar_ptr.subview(sched_stage),
                        sched_full_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    is_valid = _moe_load_sched_word((sched_storage.subview(sched_stage * SCHED_SLOT_WORDS).subview(3)))
                    # Converge consumers before releasing their scheduler slot.
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
                    sched_stage += 1
                    if sched_stage == SCHED_STAGES:
                        sched_stage = cutlass.Int32(0)
                        sched_full_phase = sched_full_phase ^ 1

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
        is_valid = cutlass.Int32(1)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)

        # M128 occupies all 32 data paths in each TMEM subpartition.  Each
        # epilogue warp drains one 32-row band; gate/up partners are lane^8.
        # @@EPILOGUE_SETUP:BEGIN@@
        row_id_with_warp_offset = base_row_id + warp_idx * 32
        shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@

        tile_l = cutlass.Int32(0)

        while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(sched_stage), sched_full_phase, time_limit=10_000_000):
            pass
        _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
        tile_m = _moe_load_sched_word((_slot.subview(1)))
        tile_n = _moe_load_sched_word((_slot.subview(2)))
        is_valid = _moe_load_sched_word((_slot.subview(3)))
        group_begin = _moe_load_sched_word((_slot.subview(4)))
        group_end = _moe_load_sched_word((_slot.subview(5)))
        group_idx = _moe_load_sched_word((_slot.subview(7)))
        # Converge consumers before releasing their scheduler slot.
        nvvm.bar_warp_sync(0xFFFFFFFF)
        sched_stage = cute.arch.make_warp_uniform(sched_stage)
        if elect_one:
            nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
        sched_stage += 1
        if sched_stage == SCHED_STAGES:
            sched_stage = cutlass.Int32(0)
            sched_full_phase = sched_full_phase ^ 1

        while is_valid != 0:
            coord_m_tile = group_begin + tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
            # @@EPILOGUE_DRAIN:BEGIN@@
            coord_n_c = tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group)
            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1
            while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(acc_stage), acc_full_phase_bit, time_limit=10_000_000):
                pass
            acc_base_col = base_col_id_root + acc_stage * acc_region_cols
            tmem = cutlass.inttoptr((row_id_with_warp_offset << 16) | acc_base_col, 6, mma_c_dtype)
            acc_vec = nvvm.tcgen05_ld(shape, tmem, num=8)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if elect_one:
                nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

            # Retain the proven rank-5 two-half weight map, but store both
            # projections directly. This FC2 experiment performs one ordinary
            # matmul: the two halves are output channels, not gate/up inputs.
            channel = coord_n_c + warp_idx * 16 + (lane & 7) + (lane // 16) * 8
            full_channel = channel + ((lane >> 3) & 1) * N
            for token_idx in cutlass.range_constexpr(8):
                value = acc_vec[token_idx].to(cutlass.BFloat16)
                row = coord_m_tile + token_idx
                linear_idx = tile_l * out_stride_l_0 + row * out_stride_m_0 + full_channel * out_stride_n_0
                if row < group_end:
                    (mC_0.iterator.raw_ptr() + linear_idx).store(value, alignment=2)

            # @@EPILOGUE_DRAIN:END@@
            tile_iter += 1

            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr.subview(sched_stage),
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            tile_m = _moe_load_sched_word((_slot.subview(1)))
            tile_n = _moe_load_sched_word((_slot.subview(2)))
            is_valid = _moe_load_sched_word((_slot.subview(3)))
            group_begin = _moe_load_sched_word((_slot.subview(4)))
            group_end = _moe_load_sched_word((_slot.subview(5)))
            group_idx = _moe_load_sched_word((_slot.subview(7)))
            # Converge consumers before releasing their scheduler slot.
            nvvm.bar_warp_sync(0xFFFFFFFF)
            sched_stage = cute.arch.make_warp_uniform(sched_stage)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)

    # DSM broadcasts can outlive one CTA's final tile. Keep every peer's
    # shared storage alive until the whole cluster has finished its accesses.
    if cutlass.const_expr(cluster_size > 1):
        nvvm.barrier_cluster_arrive_relaxed()
        nvvm.barrier_cluster_wait()


frost_sm100_moe_fc2_pair_m128n8k16_sched_static_s12_early_pdl.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    a_0: cute.Tensor,
    b_0: cute.Tensor,
    b_1: cute.Tensor,
    c_0: cute.Tensor,
    stream: _cuda.CUstream,
) -> None:
    _a_operands = [a_0]
    _b_operands = [b_0]

    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    num_experts = problem_size[3]
    num_groups = problem_size[4]
    _stride_idx = 5
    _a_stride_sets = []
    for _ in cutlass.range_constexpr(num_a_operands):
        _a_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
            )
        )
        _stride_idx += 3
    _b_stride_sets = []
    for _ in cutlass.range_constexpr(num_b_operands):
        _b_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
            )
        )
        _stride_idx += 3

    out_stride_m_0 = problem_size[_stride_idx]
    out_stride_n_0 = problem_size[_stride_idx + 1]
    out_stride_l_0 = problem_size[_stride_idx + 2]
    _stride_idx += 3

    _tma_c_outputs = [c_0]
    _c0 = _tma_c_outputs[0]
    tma_c_desc_0 = _tma.create_tensor_map_tiled(
        global_address=_c0.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[2 * n, m],
        global_strides=[
            out_stride_m_0 * 16 // 128,
        ],
        box_dims=[32, epi_tile_mn[0]],
        swizzle=_tma.TensorMapSwizzle.s64b,
    )
    tma_c_desc_list = [tma_c_desc_0]

    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_l = _a_stride_sets[_a_idx]
        tma_a_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_a_op.iterator.toint(),
                dtype=ab_tma_dtype,
                global_dims=[k_sym, m, 1],
                global_strides=[
                    a_stride_m * ab_dtype.width // 128,
                    a_stride_l * ab_dtype.width // 128,
                ],
                box_dims=[cta_tile_mnk[2], a_tma_box_m, 1],
                swizzle=ab_tma_swizzle,
            )
        )
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[n, k_sym, num_experts],
                    global_strides=[
                        b_stride_k * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    # Canonical [gate,up] rows are exposed as
                    # (K, i8, projection, q=N/8, expert).  One box gathers
                    # 64 output channels from both projections into M=128.
                    global_dims=[k_sym, 8, 2, n // 8, num_experts],
                    global_strides=[
                        b_stride_n * ab_dtype.width // 128,
                        n * b_stride_n * ab_dtype.width // 128,
                        8 * b_stride_n * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], 8, 2, cta_tile_mnk[1] // 8, 1],
                    swizzle=ab_tma_swizzle,
                )
            )

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_shape = (grid_num_clusters * cluster_m, cluster_n, 1)
    if cutlass.const_expr(not moe_static_sched):
        counter_qword = grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS
        _reset_moe_sched_counter(a_tma_workspace, cutlass.Int32(counter_qword)).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
    frost_sm100_moe_fc2_pair_m128n8k16_sched_static_s12_early_pdl(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        cutlass.Int32(num_experts),
        cutlass.Int32(num_groups),
        first_token_offset,
        a_tma_workspace,
        tma_a_desc_list[0],
        tma_b_desc_list[0],
        tma_b_desc_list[0],
        a_0,
        c_0,
        _a_stride_sets[0][0],
        out_stride_m_0,
        out_stride_n_0,
        out_stride_l_0,
        tma_c_desc_list[0],
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_shape_mnk,
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
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()

    def _make_fake_a():
        return make_fake_compact_tensor(
            mma_a_dtype,
            (sym_m, sym_k, 1),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return cute.runtime.make_fake_tensor(
            mma_b_dtype,
            (sym_n * 2, sym_k, sym_e),
            stride=(cute.sym_int64(divisibility=ab_stride_elems), 1, cute.sym_int64(divisibility=ab_stride_elems)),
            assumed_align=16,
        )

    fake_first_token_offset = make_fake_compact_tensor(
        offset_cutlass_dtype,
        (sym_g,),
        stride_order=(0,),
        assumed_align=offset_cutlass_dtype.width // 8,
    )
    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_ctas = grid_num_clusters * cluster_m * cluster_n
    fake_a_tma_workspace = make_fake_compact_tensor(
        cutlass.Int64,
        (grid_ctas * moe_desc_slots * 16 + 16,),
        stride_order=(0,),
        assumed_align=128,
    )

    def _sym_operand_strides(is_mn_major: bool) -> tuple:
        # Operand is permuted to (M|N, K, L): the unit stride is mode 0 when MN-major, mode 1 when K-major, and never reaches TMA.
        unit = 0 if is_mn_major else 1
        return tuple(cute.sym_int64() if i == unit else cute.sym_int64(divisibility=ab_stride_elems) for i in range(3))

    sym_a_strides = []
    for _ in range(num_a_operands):
        sym_a_strides.extend(_sym_operand_strides(a_is_m_major))
    sym_b_strides = []
    for _ in range(num_b_operands):
        sym_b_strides.extend(_sym_operand_strides(b_is_n_major))
    sym_out_stride_m_0 = cute.sym_int64()
    sym_out_stride_n_0 = cute.sym_int64()
    sym_out_stride_l_0 = cute.sym_int64()
    fake_a_0 = _make_fake_a()
    fake_b_0 = _make_fake_b()
    fake_b_1 = _make_fake_b()

    def _make_fake_c(_dt, _div, _mm):
        return make_fake_compact_tensor(
            _dt,
            (sym_m, 2 * sym_n // _div, 1),
            stride_order=(0, 1, 2) if _mm else (1, 0, 2),
            assumed_align=16,
        )

    fake_c_0 = _make_fake_c(cutlass.BFloat16, 1, False)
    problem_size = (
        sym_m,
        sym_n,
        sym_k,
        sym_e,
        sym_g,
        *sym_a_strides,
        *sym_b_strides,
        sym_out_stride_m_0,
        sym_out_stride_n_0,
        sym_out_stride_l_0,
    )
    pass
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return _compile_cached(
        _host,
        problem_size,
        fake_first_token_offset,
        fake_a_tma_workspace,
        fake_a_0,
        fake_b_0,
        fake_b_1,
        fake_c_0,
        stream=_fake_stream,
        options=frost_compile_options,
        # persistent object across processes (cudnn.frost.compiled_cache); the digest of THIS source is the key
        cache_key=globals().get("FROST_SOURCE_DIGEST"),
        symbol="frost_gemm",
    )
