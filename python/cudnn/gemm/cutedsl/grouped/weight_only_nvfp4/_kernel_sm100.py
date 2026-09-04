# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production SM100 kernels for Lightning weight-only NVFP4 grouped GEMMs.

The two schedules are selected from controlled B200 measurements:

* FC1 uses one decoded M64 weight fragment for three adjacent N128 token slices.
* FC2 uses one N128 token tile for three adjacent M64 output-channel slices.

Both entrypoints consume graph-native physical storage, reinterpret it with
CuTe pointer/layout operations inside the JIT, and preserve the exact private
seven-argument ABI.  They allocate no adapter workspace and launch one kernel.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.experimental.primitives as nvvm
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils import blackwell_helpers as sm100_utils
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ._common import (
    CTA_TILE_K,
    CTA_TILE_M,
    CTA_TILE_N,
    FC1_GRID_X,
    FC1_K,
    FC1_NOUT,
    FC2_K,
    FC2_M192_GRID_X,
    FC2_NOUT,
    FUSED_ACCUMULATOR_STAGES,
    FUSED_MMA_INSTRUCTION_MNK,
    FUSED_TILE_MNK,
    GROUPED_MAX_STATIC_SCAN_GROUPS,
    GROUPED_SCHEDULER_BARRIER_ID,
    GROUPED_WORK_TILE_WORDS,
    M64_A_TMEM_COLS,
    M192_TMEM_ALLOCATION_COLS,
    M_SLICES,
    NVFP4_GROUP_SIZE,
    SCALE_BYTES_PER_ROW_TILE,
    SUPERTILE_M,
    TOKEN_N384,
    TOKEN_N384_B_STAGES,
    TOKEN_N384_TMEM_ALLOCATION_COLS,
    TOKEN_N384_WORK_TILE_WORDS,
    TOKEN_SLICES,
    TRANSFORM_THREADS,
    _fill_one_m64_a_fragment,
    _M192SharedStorage,
    _make_b_copy_arguments,
    _TokenN384SharedStorage,
)


@cute.jit
def _store_one_m64_accumulator_ragged(
    tiled_mma: cute.TiledMma,
    t_ct_acc: cute.Tensor,
    logical_output: cute.Tensor,
    factor_f32: cutlass.Float32,
    relu_square: cutlass.Constexpr,
    store_n_extent: cutlass.Int32,
) -> None:
    """Apply the checkpoint epilogue and predicate a ragged N128 prefix."""

    tidx = cute.arch.thread_idx()[0]
    thr_mma = tiled_mma.get_slice(0)
    t_c_g_c = transform_partitioned_tensor_layout(thr_mma.partition_C(logical_output))
    c_identity = cute.make_identity_tensor((CTA_TILE_M, CTA_TILE_N))
    t_c_c = transform_partitioned_tensor_layout(thr_mma.partition_C(c_identity))
    t_ct_acc = transform_partitioned_tensor_layout(t_ct_acc)
    epilogue_tiler = sm100_utils.compute_epilogue_tile_shape(
        FUSED_TILE_MNK,
        False,
        utils.LayoutEnum.COL_MAJOR,
        cutlass.BFloat16,
    )
    t_ct_acc_epi = cute.flat_divide(t_ct_acc, epilogue_tiler)
    t_c_g_c_epi = cute.flat_divide(t_c_g_c, epilogue_tiler)
    t_c_c_epi = cute.flat_divide(t_c_c, epilogue_tiler)
    tmem_atom = sm100_utils.get_tmem_load_op(
        FUSED_TILE_MNK,
        utils.LayoutEnum.COL_MAJOR,
        cutlass.BFloat16,
        cutlass.Float32,
        epilogue_tiler,
        False,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, t_ct_acc_epi[(None, None, 0, 0)])
    tmem_thread_copy = tmem_tiled_copy.get_slice(tidx)
    t_t_acc = tmem_thread_copy.partition_S(t_ct_acc_epi)
    t_g_c = tmem_thread_copy.partition_D(t_c_g_c_epi)
    t_g_coord = tmem_thread_copy.partition_D(t_c_c_epi)
    t_t_acc = cute.group_modes(t_t_acc, 3, cute.rank(t_t_acc))
    t_g_c = cute.group_modes(t_g_c, 3, cute.rank(t_g_c))
    t_g_coord = cute.group_modes(t_g_coord, 3, cute.rank(t_g_coord))
    compact_value_layout = cute.make_layout(
        (((2, 2, 8), 1), 1, 1),
        stride=(((1, 2, 4), 0), 0, 0),
    )
    if cutlass.const_expr(t_g_coord[(None, None, None, 0)].shape != compact_value_layout.shape):
        raise ValueError("grouped compact epilogue requires the canonical M64 T2R coordinate layout")
    if cutlass.const_expr(cute.cosize(compact_value_layout) != 32):
        raise ValueError("grouped compact epilogue must materialize exactly thirty-two values per subtile")

    # Fail closed on the exact fixed copy layouts produced after the standard
    # SM100 MMA-partition transform.  This keeps the per-thread fragment at
    # thirty-two physical values and prevents a future layout change from
    # silently reintroducing zero-stride logical repetitions into rmem.
    expected_tmem_compact_stride = (((1, 65536), 0), 0, 0)
    output_m_stride, output_n_stride = logical_output.stride
    expected_gmem_compact_stride = (
        ((output_n_stride, 8 * output_m_stride, 8 * output_n_stride), 0),
        0,
        0,
    )
    expected_tmem_grouped_shape = (((64, 16), 1), 1, 1, (1, 2))
    expected_tmem_grouped_stride = (((1, 65536), 0), 0, 0, (0, 64))
    expected_gmem_grouped_shape = (((2, 2, 8), 1), 1, 1, (1, 2))
    expected_gmem_grouped_stride = (
        ((output_n_stride, 8 * output_m_stride, 8 * output_n_stride), 0),
        0,
        0,
        (0, 64 * output_n_stride),
    )
    if cutlass.const_expr(t_t_acc.shape != expected_tmem_grouped_shape or t_t_acc.stride != expected_tmem_grouped_stride):
        raise ValueError("grouped compact epilogue requires exactly two canonical TMEM subtiles")
    if cutlass.const_expr(t_g_c.shape != expected_gmem_grouped_shape or t_g_c.stride != expected_gmem_grouped_stride):
        raise ValueError("grouped compact epilogue requires exactly two canonical global subtiles")
    if cutlass.const_expr(t_t_acc[(None, None, None, 0)].stride != expected_tmem_compact_stride):
        raise ValueError("grouped compact epilogue produced an unexpected canonical TMEM layout")
    if cutlass.const_expr(t_g_c[(None, None, None, 0)].stride != expected_gmem_compact_stride):
        raise ValueError("grouped compact epilogue produced an unexpected canonical global layout")
    r_acc = cute.make_rmem_tensor(compact_value_layout, cutlass.Float32)
    r_out = cute.make_rmem_tensor(compact_value_layout, cutlass.BFloat16)

    for subtile in cutlass.range_constexpr(cute.size(t_t_acc, mode=[3])):
        t_t_acc_compact = t_t_acc[(None, None, None, subtile)]
        t_g_c_compact = t_g_c[(None, None, None, subtile)]
        t_g_coord_compact = t_g_coord[(None, None, None, subtile)]
        cute.copy(tmem_tiled_copy, t_t_acc_compact, r_acc)
        acc_values = r_acc.load()
        factor_values = cute.full_like(acc_values, factor_f32)
        checkpoint = (acc_values * factor_values).to(cutlass.BFloat16)
        if cutlass.const_expr(relu_square):
            checkpoint_f32 = checkpoint.to(cutlass.Float32)
            # cuDNN's RELU_FWD descriptor uses CUDNN_PROPAGATE_NAN.  Select
            # zero only for ordered non-positive values so an unordered NaN
            # remains on the data path and is still NaN after self-square.
            relu_values = cute.where(
                checkpoint_f32 <= cutlass.Float32(0.0),
                cute.zeros_like(checkpoint_f32),
                checkpoint_f32,
            )
            checkpoint = (relu_values * relu_values).to(cutlass.BFloat16)
        r_out.store(checkpoint)
        if store_n_extent == cutlass.Int32(CTA_TILE_N):
            cute.autovec_copy(r_out, t_g_c_compact)
        else:
            r_out_flat = cute.group_modes(r_out, 0, cute.rank(r_out))
            t_g_c_flat = cute.group_modes(t_g_c_compact, 0, cute.rank(t_g_c_compact))
            t_g_coord_flat = cute.group_modes(t_g_coord_compact, 0, cute.rank(t_g_coord_compact))
            for elem in cutlass.range_constexpr(cute.size(r_out_flat)):
                local_coord = t_g_coord_flat[elem]
                local_token = cutlass.Int32(cute.get(local_coord, mode=[1]))
                if local_token < store_n_extent:
                    t_g_c_flat[elem] = r_out_flat[elem]


@cute.kernel
def _weight_only_nvfp4_grouped_m192_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_b: cute.CopyAtom,
    tma_routed_tokens: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    packed_weight: cute.Tensor,
    weight_scale: cute.Tensor,
    routed_tokens: cute.Tensor,
    first_token_offset: cute.Tensor,
    physical_output: cute.Tensor,
    factor: cute.Tensor,
    b_smem_layout: cute.ComposedLayout,
    num_acc_tmem_cols: cutlass.Constexpr,
    num_groups: cutlass.Constexpr,
    num_experts: cutlass.Constexpr,
    total_rows: cutlass.Int32,
) -> None:
    """Schedule one ragged record, then compute its exact M192 FC2 supertile."""

    tidx = cute.arch.thread_idx()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    storage = smem.allocate(_M192SharedStorage)

    if warp_idx == 0:
        lane = cute.arch.lane_idx()
        work_y = cutlass.Int32(cute.arch.block_idx()[1])
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0

        found = cutlass.Int32(0)
        group_idx = cutlass.Int32(-1)
        group_begin = cutlass.Int32(0)
        group_end = cutlass.Int32(0)
        group_tile_start_y = cutlass.Int32(0)
        chunk_tile_start_y = cutlass.Int32(0)

        for chunk_idx in cutlass.range_constexpr((num_groups + 31) // 32):
            if found == cutlass.Int32(0):
                visit_idx = cutlass.Int32(chunk_idx * 32) + lane
                my_begin = cutlass.Int32(0)
                my_end = cutlass.Int32(0)
                my_tiles = cutlass.Int32(0)
                if visit_idx < cutlass.Int32(num_groups):
                    my_begin = cutlass.Int32(first_token_offset[visit_idx])
                    my_end = total_rows
                    if visit_idx + cutlass.Int32(1) < cutlass.Int32(num_groups):
                        my_end = cutlass.Int32(first_token_offset[visit_idx + cutlass.Int32(1)])
                    my_tiles = (my_end - my_begin + cutlass.Int32(CTA_TILE_N - 1)) // cutlass.Int32(CTA_TILE_N)

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

                my_start = chunk_tile_start_y + prefix_tiles - my_tiles
                candidates = nvvm.vote_sync(
                    full_warp_mask,
                    work_y < my_start + my_tiles and visit_idx < cutlass.Int32(num_groups),
                    nvvm.VoteSync.BALLOT,
                )
                if candidates != 0:
                    winning_lane = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(cutlass.Uint32(candidates))).to(cutlass.Int32)
                    group_idx = nvvm.shfl_sync(
                        full_warp_mask,
                        visit_idx,
                        winning_lane,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    group_begin = nvvm.shfl_sync(
                        full_warp_mask,
                        my_begin,
                        winning_lane,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    group_end = nvvm.shfl_sync(
                        full_warp_mask,
                        my_end,
                        winning_lane,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    group_tile_start_y = nvvm.shfl_sync(
                        full_warp_mask,
                        my_start,
                        winning_lane,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    found = cutlass.Int32(1)
                else:
                    chunk_tile_end_y = my_start + my_tiles
                    chunk_tile_start_y = nvvm.shfl_sync(
                        full_warp_mask,
                        chunk_tile_end_y,
                        31,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )

        if tidx == 0:
            s_work = cute.make_tensor(
                storage.grouped_work_words.data_ptr(),
                cute.make_layout((GROUPED_WORK_TILE_WORDS,), stride=(1,)),
            )
            for word in cutlass.range_constexpr(GROUPED_WORK_TILE_WORDS):
                s_work[word] = cutlass.Int32(0)
            s_work[0] = cutlass.Int32(-1)
            s_work[1] = cutlass.Int32(-1)
            if found != cutlass.Int32(0):
                supertile_m = cutlass.Int32(cute.arch.block_idx()[0])
                tile_n_idx = work_y - group_tile_start_y
                token_begin = group_begin + tile_n_idx * cutlass.Int32(CTA_TILE_N)
                s_work[0] = group_idx
                s_work[1] = group_idx % cutlass.Int32(num_experts)
                s_work[2] = supertile_m
                s_work[3] = tile_n_idx
                s_work[4] = group_begin
                s_work[5] = group_end
                s_work[6] = token_begin
                s_work[7] = cutlass.Int32(SUPERTILE_M)
                s_work[8] = cutlass.min(cutlass.Int32(CTA_TILE_N), group_end - token_begin)
                s_work[9] = cutlass.Int32(packed_weight.shape[1] // CTA_TILE_K)

    schedule_ready = pipeline.NamedBarrier(
        barrier_id=GROUPED_SCHEDULER_BARRIER_ID,
        num_threads=TRANSFORM_THREADS,
    )
    schedule_ready.arrive_and_wait()
    s_work = cute.make_tensor(
        storage.grouped_work_words.data_ptr(),
        cute.make_layout((GROUPED_WORK_TILE_WORDS,), stride=(1,)),
    )
    grouped_work = cute.make_rmem_tensor((GROUPED_WORK_TILE_WORDS,), cutlass.Int32)
    for word in cutlass.range_constexpr(GROUPED_WORK_TILE_WORDS):
        grouped_work[word] = s_work[word]

    if grouped_work[9] == cutlass.Int32(0):
        cute.arch.nvvm.exit()

    expert_idx = grouped_work[1]
    supertile_m = grouped_work[2]
    token_base = grouped_work[6]
    store_n_extent = grouped_work[8]
    output_tile_m_0 = supertile_m * cutlass.Int32(M_SLICES)
    output_tile_m_1 = output_tile_m_0 + cutlass.Int32(1)
    output_tile_m_2 = output_tile_m_0 + cutlass.Int32(2)

    s_scale = smem.allocate_tensor(
        element_type=cutlass.Float8E4M3FN,
        layout=cute.make_layout(
            (SUPERTILE_M, SCALE_BYTES_PER_ROW_TILE),
            stride=(SCALE_BYTES_PER_ROW_TILE, 1),
        ),
        byte_alignment=16,
    )
    s_b = smem.allocate_tensor(
        element_type=cutlass.BFloat16,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    thr_mma = tiled_mma.get_slice(0)
    tma_token_source = cute.domain_offset((token_base, 0, 0), tma_routed_tokens)
    g_b = cute.local_tile(
        tma_token_source,
        cute.slice_(FUSED_TILE_MNK, (0, None, None)),
        (None, None, None),
    )
    t_c_g_b = thr_mma.partition_B(g_b)
    t_b_s_b, t_b_g_b = cpasync.tma_partition(
        tma_atom_b,
        cutlass.Int32(0),
        cute.make_layout(cute.size(cluster_layout_vmnk, mode=[1])),
        cute.group_modes(s_b, 0, 3),
        cute.group_modes(t_c_g_b, 0, 3),
    )
    t_cr_b = tiled_mma.make_fragment_B(s_b)

    a_identity = cute.make_identity_tensor((CTA_TILE_M, CTA_TILE_K))
    partitioned_a_identity = thr_mma.partition_A(a_identity)
    acc_shape = tiled_mma.partition_shape_C(FUSED_TILE_MNK[:2])
    acc_template = tiled_mma.make_fragment_C(acc_shape)
    a_layout = tiled_mma.make_fragment_A(partitioned_a_identity.layout).layout

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=TRANSFORM_THREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(M192_TMEM_ALLOCATION_COLS)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    t_ct_acc_0 = cute.make_tensor(tmem_ptr, acc_template.layout)
    t_ct_acc_1 = cute.make_tensor(tmem_ptr + num_acc_tmem_cols, acc_template.layout)
    t_ct_acc_2 = cute.make_tensor(tmem_ptr + 2 * num_acc_tmem_cols, acc_template.layout)
    tmem_a_0_ptr = cute.recast_ptr(tmem_ptr + 3 * num_acc_tmem_cols, dtype=cutlass.BFloat16)
    tmem_a_1_ptr = cute.recast_ptr(tmem_ptr + 3 * num_acc_tmem_cols + M64_A_TMEM_COLS, dtype=cutlass.BFloat16)
    tmem_a_2_ptr = cute.recast_ptr(tmem_ptr + 3 * num_acc_tmem_cols + 2 * M64_A_TMEM_COLS, dtype=cutlass.BFloat16)
    t_cr_a_0 = cute.make_tensor(tmem_a_0_ptr, a_layout)
    t_cr_a_1 = cute.make_tensor(tmem_a_1_ptr, a_layout)
    t_cr_a_2 = cute.make_tensor(tmem_a_2_ptr, a_layout)

    store_atom = cute.make_copy_atom(
        tcgen05.St16x256bOp(tcgen05.Repetition(1), tcgen05.Unpack.NONE),
        cutlass.BFloat16,
    )
    tiled_store_0 = tcgen05.make_tmem_copy(store_atom, t_cr_a_0)
    tiled_store_1 = tcgen05.make_tmem_copy(store_atom, t_cr_a_1)
    tiled_store_2 = tcgen05.make_tmem_copy(store_atom, t_cr_a_2)
    thread_store_0 = tiled_store_0.get_slice(tidx)
    thread_store_1 = tiled_store_1.get_slice(tidx)
    thread_store_2 = tiled_store_2.get_slice(tidx)
    t_s_a_coords_0 = thread_store_0.partition_S(partitioned_a_identity)
    t_s_a_coords_1 = thread_store_1.partition_S(partitioned_a_identity)
    t_s_a_coords_2 = thread_store_2.partition_S(partitioned_a_identity)
    t_d_a_0 = thread_store_0.partition_D(t_cr_a_0)
    t_d_a_1 = thread_store_1.partition_D(t_cr_a_1)
    t_d_a_2 = thread_store_2.partition_D(t_cr_a_2)
    r_a_0 = cute.make_rmem_tensor(t_s_a_coords_0.shape, cutlass.BFloat16)
    r_a_1 = cute.make_rmem_tensor(t_s_a_coords_1.shape, cutlass.BFloat16)
    r_a_2 = cute.make_rmem_tensor(t_s_a_coords_2.shape, cutlass.BFloat16)

    operand_pipeline = pipeline.PipelineAsyncUmma.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, TRANSFORM_THREADS),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        barrier_storage=storage.operand_mbar_ptr.data_ptr(),
    )
    b_pipeline = pipeline.PipelineTmaUmma.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=CTA_TILE_N * CTA_TILE_K * cutlass.BFloat16.width // 8,
        barrier_storage=storage.b_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
    )
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        num_stages=FUSED_ACCUMULATOR_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, TRANSFORM_THREADS),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    )
    operand_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
    operand_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
    b_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
    b_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
    acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, FUSED_ACCUMULATOR_STAGES)
    acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, FUSED_ACCUMULATOR_STAGES)

    if warp_idx == 0:
        acc_pipeline.producer_acquire(acc_producer_state)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

    scale_u32_per_row = weight_scale.shape[1] // SCALE_BYTES_PER_ROW_TILE
    scale_expert_row_base = expert_idx * weight_scale.shape[0]
    g_scale_u32 = cute.recast_ptr(weight_scale.iterator, dtype=cutlass.Uint32)
    s_scale_u32 = cute.recast_ptr(s_scale.iterator, dtype=cutlass.Uint32)
    num_k_tiles = routed_tokens.shape[1] // CTA_TILE_K
    supertile_row_base = supertile_m * cutlass.Int32(SUPERTILE_M)
    for current_tile_k in cutlass.range(num_k_tiles, unroll=1):
        operand_pipeline.producer_acquire(operand_producer_state)
        if warp_idx == 0:
            b_pipeline.producer_acquire(b_producer_state)
            cute.copy(
                tma_atom_b,
                t_b_g_b[(None, 0, current_tile_k, 0)],
                t_b_s_b[(None, b_producer_state.index)],
                tma_bar_ptr=b_pipeline.producer_get_barrier(b_producer_state),
            )

        global_row_01 = supertile_row_base + tidx
        scale_u32_idx_01 = (scale_expert_row_base + global_row_01) * scale_u32_per_row + current_tile_k
        (s_scale_u32 + tidx).store((g_scale_u32 + scale_u32_idx_01).load())
        if tidx < cutlass.Int32(CTA_TILE_M):
            global_row_2 = supertile_row_base + cutlass.Int32(2 * CTA_TILE_M) + tidx
            scale_u32_idx_2 = (scale_expert_row_base + global_row_2) * scale_u32_per_row + current_tile_k
            (s_scale_u32 + cutlass.Int32(2 * CTA_TILE_M) + tidx).store((g_scale_u32 + scale_u32_idx_2).load())
        transform_inputs_ready = pipeline.NamedBarrier(barrier_id=1, num_threads=TRANSFORM_THREADS)
        transform_inputs_ready.arrive_and_wait()

        _fill_one_m64_a_fragment(
            packed_weight,
            s_scale,
            t_s_a_coords_0,
            r_a_0,
            output_tile_m_0,
            cutlass.Int32(0),
            current_tile_k,
            expert_idx,
        )
        _fill_one_m64_a_fragment(
            packed_weight,
            s_scale,
            t_s_a_coords_1,
            r_a_1,
            output_tile_m_1,
            cutlass.Int32(CTA_TILE_M),
            current_tile_k,
            expert_idx,
        )
        _fill_one_m64_a_fragment(
            packed_weight,
            s_scale,
            t_s_a_coords_2,
            r_a_2,
            output_tile_m_2,
            cutlass.Int32(2 * CTA_TILE_M),
            current_tile_k,
            expert_idx,
        )
        cute.copy(thread_store_0, r_a_0, t_d_a_0)
        cute.copy(thread_store_1, r_a_1, t_d_a_1)
        cute.copy(thread_store_2, r_a_2, t_d_a_2)
        transform_store_done = pipeline.NamedBarrier(barrier_id=2, num_threads=TRANSFORM_THREADS)
        transform_store_done.arrive_and_wait()
        cute.arch.fence_view_async_tmem_store()
        operand_pipeline.producer_commit(operand_producer_state)
        operand_producer_state.advance()

        if warp_idx == 0:
            operand_pipeline.consumer_wait(operand_consumer_state)
            b_pipeline.consumer_wait(b_consumer_state)
            for k_block in cutlass.range_constexpr(cute.size(t_cr_a_0, mode=[2])):
                cute.gemm(
                    tiled_mma,
                    t_ct_acc_0,
                    t_cr_a_0[(None, None, k_block)],
                    t_cr_b[(None, None, k_block, 0)],
                    t_ct_acc_0,
                )
                cute.gemm(
                    tiled_mma,
                    t_ct_acc_1,
                    t_cr_a_1[(None, None, k_block)],
                    t_cr_b[(None, None, k_block, 0)],
                    t_ct_acc_1,
                )
                cute.gemm(
                    tiled_mma,
                    t_ct_acc_2,
                    t_cr_a_2[(None, None, k_block)],
                    t_cr_b[(None, None, k_block, 0)],
                    t_ct_acc_2,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            operand_pipeline.consumer_release(operand_consumer_state)
            operand_consumer_state.advance()
            b_pipeline.consumer_release(b_consumer_state)
            b_producer_state.advance()
            b_consumer_state.advance()

    if warp_idx == 0:
        acc_pipeline.producer_commit(acc_producer_state)
    tmem.relinquish_alloc_permit()
    acc_pipeline.consumer_wait(acc_consumer_state)

    factor_f32 = factor[expert_idx].to(cutlass.Float32)
    output_0 = cute.domain_offset((cutlass.Int64(token_base), output_tile_m_0 * cutlass.Int32(CTA_TILE_M)), physical_output)
    output_1 = cute.domain_offset((cutlass.Int64(token_base), output_tile_m_1 * cutlass.Int32(CTA_TILE_M)), physical_output)
    output_2 = cute.domain_offset((cutlass.Int64(token_base), output_tile_m_2 * cutlass.Int32(CTA_TILE_M)), physical_output)
    logical_output_0 = cute.make_tensor(
        output_0.iterator,
        cute.make_layout((CTA_TILE_M, CTA_TILE_N), stride=(1, physical_output.shape[1])),
    )
    logical_output_1 = cute.make_tensor(
        output_1.iterator,
        cute.make_layout((CTA_TILE_M, CTA_TILE_N), stride=(1, physical_output.shape[1])),
    )
    logical_output_2 = cute.make_tensor(
        output_2.iterator,
        cute.make_layout((CTA_TILE_M, CTA_TILE_N), stride=(1, physical_output.shape[1])),
    )
    _store_one_m64_accumulator_ragged(tiled_mma, t_ct_acc_0, logical_output_0, factor_f32, False, store_n_extent)
    _store_one_m64_accumulator_ragged(tiled_mma, t_ct_acc_1, logical_output_1, factor_f32, False, store_n_extent)
    _store_one_m64_accumulator_ragged(tiled_mma, t_ct_acc_2, logical_output_2, factor_f32, False, store_n_extent)
    acc_pipeline.consumer_release(acc_consumer_state)

    pipeline.sync(barrier_id=3)
    tmem.free(tmem_ptr)


@cute.jit
def _launch_weight_only_nvfp4_grouped_m192(
    packed_weight: cute.Tensor,
    weight_scale: cute.Tensor,
    routed_tokens: cute.Tensor,
    first_token_offset: cute.Tensor,
    physical_output: cute.Tensor,
    factor: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """Launch the starts-only exact-FC2 grouped M192 kernel."""

    if cutlass.const_expr(
        cute.rank(packed_weight) != 3 or packed_weight.shape[:2] != (FC2_NOUT, FC2_K) or packed_weight.stride != (FC2_K, 1, FC2_NOUT * FC2_K)
    ):
        raise ValueError("grouped M192 packed weight must be exact Lightning FC2 logical [2688,1856,E]")
    if cutlass.const_expr(
        cute.rank(weight_scale) != 3
        or weight_scale.shape != (FC2_NOUT, FC2_K // NVFP4_GROUP_SIZE, packed_weight.shape[2])
        or weight_scale.stride
        != (
            FC2_K // NVFP4_GROUP_SIZE,
            1,
            FC2_NOUT * (FC2_K // NVFP4_GROUP_SIZE),
        )
    ):
        raise ValueError("grouped M192 scale must use checkpoint-native [E,2688,116] storage")
    if cutlass.const_expr(cute.rank(routed_tokens) != 2 or routed_tokens.shape[1] != FC2_K or routed_tokens.stride != (FC2_K, 1)):
        raise ValueError("grouped M192 tokens must be contiguous row-major BF16 [S,1856]")
    if cutlass.const_expr(
        cute.rank(first_token_offset) != 1
        or first_token_offset.element_type != cutlass.Int32
        or first_token_offset.stride != (1,)
        or first_token_offset.shape[0] < 1
        or first_token_offset.shape[0] != packed_weight.shape[2]
        or first_token_offset.shape[0] > GROUPED_MAX_STATIC_SCAN_GROUPS
    ):
        raise TypeError("grouped M192 first_token_offset must be contiguous starts-only INT32 [G], E=G in [1,128]")
    if cutlass.const_expr(cute.rank(physical_output) != 2 or physical_output.shape[1] != FC2_NOUT or physical_output.stride != (FC2_NOUT, 1)):
        raise ValueError("grouped M192 output must be contiguous row-major BF16 [S,2688]")
    if cutlass.const_expr(
        cute.rank(factor) != 3 or factor.shape != (packed_weight.shape[2], 1, 1) or factor.stride != (1, 1, 1) or factor.element_type != cutlass.Float32
    ):
        raise TypeError("grouped M192 factor must be contiguous per-expert FP32 [E,1,1]")
    if cutlass.const_expr(packed_weight.element_type != cutlass.Float4E2M1FN):
        raise TypeError("packed weight must be logical E2M1")
    if cutlass.const_expr(weight_scale.element_type != cutlass.Float8E4M3FN):
        raise TypeError("weight scale must be E4M3")
    if cutlass.const_expr(routed_tokens.element_type != cutlass.BFloat16):
        raise TypeError("routed tokens must be BF16")
    if cutlass.const_expr(physical_output.element_type != cutlass.BFloat16):
        raise TypeError("physical output must be BF16")

    num_experts = packed_weight.shape[2]
    num_groups = first_token_offset.shape[0]
    factor_by_expert = cute.make_tensor(factor.iterator, cute.make_layout((num_experts,), stride=(1,)))
    mma_op = tcgen05.MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        FUSED_MMA_INSTRUCTION_MNK,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.TMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, FUSED_TILE_MNK, cutlass.BFloat16, 1)
    acc_shape = tiled_mma.partition_shape_C(FUSED_TILE_MNK[:2])
    fake_accumulator = tiled_mma.make_fragment_C(acc_shape)
    num_acc_tmem_cols = utils.get_num_tmem_alloc_cols(fake_accumulator, arch="sm_100")
    if cutlass.const_expr(num_acc_tmem_cols != 128):
        raise ValueError("grouped M192 TMEM map requires exactly 128 columns per M64 accumulator")
    if cutlass.const_expr(3 * num_acc_tmem_cols + 3 * M64_A_TMEM_COLS > M192_TMEM_ALLOCATION_COLS):
        raise ValueError("grouped M192 accumulator and decoded-A fragments exceed the 512-column TMEM allocation")
    tma_atom_b, tma_routed_tokens, cluster_layout_vmnk = _make_b_copy_arguments(
        tiled_mma,
        routed_tokens,
        b_smem_layout,
    )
    grid_y = cute.ceil_div(routed_tokens.shape[0], CTA_TILE_N) + num_groups - 1
    _weight_only_nvfp4_grouped_m192_kernel(
        tiled_mma,
        tma_atom_b,
        tma_routed_tokens,
        cluster_layout_vmnk,
        packed_weight,
        weight_scale,
        routed_tokens,
        first_token_offset,
        physical_output,
        factor_by_expert,
        b_smem_layout,
        num_acc_tmem_cols,
        num_groups,
        num_experts,
        cutlass.Int32(routed_tokens.shape[0]),
    ).launch(
        grid=(FC2_M192_GRID_X, grid_y, 1),
        block=(TRANSFORM_THREADS, 1, 1),
        cluster=(1, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


@cute.kernel
def _weight_only_nvfp4_grouped_token_n384_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_b: cute.CopyAtom,
    tma_routed_tokens: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    packed_weight: cute.Tensor,
    weight_scale: cute.Tensor,
    routed_tokens: cute.Tensor,
    first_token_offset: cute.Tensor,
    physical_output: cute.Tensor,
    factor: cute.Tensor,
    b_smem_layout: cute.ComposedLayout,
    num_acc_tmem_cols: cutlass.Constexpr,
    num_groups: cutlass.Constexpr,
    num_experts: cutlass.Constexpr,
    total_rows: cutlass.Int32,
    relu_square: cutlass.Constexpr,
) -> None:
    """Schedule one expert-local N384 record and execute its present slices."""

    tidx = cute.arch.thread_idx()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    storage = smem.allocate(_TokenN384SharedStorage)

    if warp_idx == 0:
        lane = cute.arch.lane_idx()
        work_y = cutlass.Int32(cute.arch.block_idx()[1])
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0

        found = cutlass.Int32(0)
        group_idx = cutlass.Int32(-1)
        group_begin = cutlass.Int32(0)
        group_end = cutlass.Int32(0)
        group_tile_start_y = cutlass.Int32(0)
        chunk_tile_start_y = cutlass.Int32(0)

        for chunk_idx in cutlass.range_constexpr((num_groups + 31) // 32):
            if found == cutlass.Int32(0):
                visit_idx = cutlass.Int32(chunk_idx * 32) + lane
                my_begin = cutlass.Int32(0)
                my_end = cutlass.Int32(0)
                my_tiles = cutlass.Int32(0)
                if visit_idx < cutlass.Int32(num_groups):
                    my_begin = cutlass.Int32(first_token_offset[visit_idx])
                    my_end = total_rows
                    if visit_idx + cutlass.Int32(1) < cutlass.Int32(num_groups):
                        my_end = cutlass.Int32(first_token_offset[visit_idx + cutlass.Int32(1)])
                    my_tiles = (my_end - my_begin + cutlass.Int32(TOKEN_N384 - 1)) // cutlass.Int32(TOKEN_N384)

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

                my_start = chunk_tile_start_y + prefix_tiles - my_tiles
                candidates = nvvm.vote_sync(
                    full_warp_mask,
                    work_y < my_start + my_tiles and visit_idx < cutlass.Int32(num_groups),
                    nvvm.VoteSync.BALLOT,
                )
                if candidates != 0:
                    winning_lane = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(cutlass.Uint32(candidates))).to(cutlass.Int32)
                    group_idx = nvvm.shfl_sync(
                        full_warp_mask,
                        visit_idx,
                        winning_lane,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    group_begin = nvvm.shfl_sync(
                        full_warp_mask,
                        my_begin,
                        winning_lane,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    group_end = nvvm.shfl_sync(
                        full_warp_mask,
                        my_end,
                        winning_lane,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    group_tile_start_y = nvvm.shfl_sync(
                        full_warp_mask,
                        my_start,
                        winning_lane,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    found = cutlass.Int32(1)
                else:
                    chunk_tile_end_y = my_start + my_tiles
                    chunk_tile_start_y = nvvm.shfl_sync(
                        full_warp_mask,
                        chunk_tile_end_y,
                        31,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )

        if tidx == 0:
            s_work = cute.make_tensor(
                storage.grouped_work_words.data_ptr(),
                cute.make_layout((TOKEN_N384_WORK_TILE_WORDS,), stride=(1,)),
            )
            for word in cutlass.range_constexpr(TOKEN_N384_WORK_TILE_WORDS):
                s_work[word] = cutlass.Int32(0)
            s_work[0] = cutlass.Int32(-1)
            s_work[1] = cutlass.Int32(-1)
            if found != cutlass.Int32(0):
                output_tile_m = cutlass.Int32(cute.arch.block_idx()[0])
                tile_n384_idx = work_y - group_tile_start_y
                token_begin = group_begin + tile_n384_idx * cutlass.Int32(TOKEN_N384)
                first_extent = cutlass.min(cutlass.Int32(CTA_TILE_N), group_end - token_begin)
                second_remaining = group_end - (token_begin + cutlass.Int32(CTA_TILE_N))
                second_extent = cutlass.Int32(0)
                if second_remaining > cutlass.Int32(0):
                    second_extent = cutlass.min(cutlass.Int32(CTA_TILE_N), second_remaining)
                third_remaining = group_end - (token_begin + cutlass.Int32(2 * CTA_TILE_N))
                third_extent = cutlass.Int32(0)
                if third_remaining > cutlass.Int32(0):
                    third_extent = cutlass.min(cutlass.Int32(CTA_TILE_N), third_remaining)
                s_work[0] = group_idx
                s_work[1] = group_idx % cutlass.Int32(num_experts)
                s_work[2] = output_tile_m
                s_work[3] = tile_n384_idx
                s_work[4] = group_begin
                s_work[5] = group_end
                s_work[6] = token_begin
                s_work[7] = first_extent
                s_work[8] = second_extent
                s_work[9] = third_extent
                s_work[10] = cutlass.Int32(packed_weight.shape[1] // CTA_TILE_K)

    schedule_ready = pipeline.NamedBarrier(
        barrier_id=GROUPED_SCHEDULER_BARRIER_ID,
        num_threads=TRANSFORM_THREADS,
    )
    schedule_ready.arrive_and_wait()
    s_work = cute.make_tensor(
        storage.grouped_work_words.data_ptr(),
        cute.make_layout((TOKEN_N384_WORK_TILE_WORDS,), stride=(1,)),
    )
    grouped_work = cute.make_rmem_tensor((TOKEN_N384_WORK_TILE_WORDS,), cutlass.Int32)
    for word in cutlass.range_constexpr(TOKEN_N384_WORK_TILE_WORDS):
        grouped_work[word] = s_work[word]

    # Static upper-bound padding CTAs leave before any pipeline is created or
    # TMEM is acquired.  The eleven-word record makes this exit CTA-uniform.
    if grouped_work[10] == cutlass.Int32(0):
        cute.arch.nvvm.exit()

    expert_idx = grouped_work[1]
    output_tile_m = grouped_work[2]
    token_base = grouped_work[6]
    first_store_n_extent = grouped_work[7]
    second_store_n_extent = grouped_work[8]
    third_store_n_extent = grouped_work[9]
    has_second_token_slice = second_store_n_extent > cutlass.Int32(0)
    has_third_token_slice = third_store_n_extent > cutlass.Int32(0)

    s_scale = smem.allocate_tensor(
        element_type=cutlass.Float8E4M3FN,
        layout=cute.make_layout(
            (CTA_TILE_M, SCALE_BYTES_PER_ROW_TILE),
            stride=(SCALE_BYTES_PER_ROW_TILE, 1),
        ),
        byte_alignment=16,
    )
    s_b = smem.allocate_tensor(
        element_type=cutlass.BFloat16,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    thr_mma = tiled_mma.get_slice(0)
    tma_token_source = cute.domain_offset((token_base, 0, 0), tma_routed_tokens)
    g_b = cute.local_tile(
        tma_token_source,
        cute.slice_(FUSED_TILE_MNK, (0, None, None)),
        (None, None, None),
    )
    t_c_g_b = thr_mma.partition_B(g_b)
    t_b_s_b, t_b_g_b = cpasync.tma_partition(
        tma_atom_b,
        cutlass.Int32(0),
        cute.make_layout(cute.size(cluster_layout_vmnk, mode=[1])),
        cute.group_modes(s_b, 0, 3),
        cute.group_modes(t_c_g_b, 0, 3),
    )
    t_cr_b = tiled_mma.make_fragment_B(s_b)

    a_identity = cute.make_identity_tensor((CTA_TILE_M, CTA_TILE_K))
    partitioned_a_identity = thr_mma.partition_A(a_identity)
    acc_shape = tiled_mma.partition_shape_C(FUSED_TILE_MNK[:2])
    acc_template = tiled_mma.make_fragment_C(acc_shape)
    a_layout = tiled_mma.make_fragment_A(partitioned_a_identity.layout).layout

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=TRANSFORM_THREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(TOKEN_N384_TMEM_ALLOCATION_COLS)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    t_ct_acc_0 = cute.make_tensor(tmem_ptr, acc_template.layout)
    t_ct_acc_1 = cute.make_tensor(tmem_ptr + num_acc_tmem_cols, acc_template.layout)
    t_ct_acc_2 = cute.make_tensor(tmem_ptr + 2 * num_acc_tmem_cols, acc_template.layout)
    tmem_a_ptr = cute.recast_ptr(tmem_ptr + TOKEN_SLICES * num_acc_tmem_cols, dtype=cutlass.BFloat16)
    t_cr_a = cute.make_tensor(tmem_a_ptr, a_layout)

    store_atom = cute.make_copy_atom(
        tcgen05.St16x256bOp(tcgen05.Repetition(1), tcgen05.Unpack.NONE),
        cutlass.BFloat16,
    )
    tiled_store = tcgen05.make_tmem_copy(store_atom, t_cr_a)
    thread_store = tiled_store.get_slice(tidx)
    t_s_a_coords = thread_store.partition_S(partitioned_a_identity)
    t_d_a = thread_store.partition_D(t_cr_a)
    r_a = cute.make_rmem_tensor(t_s_a_coords.shape, cutlass.BFloat16)

    operand_pipeline = pipeline.PipelineAsyncUmma.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, TRANSFORM_THREADS),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        barrier_storage=storage.operand_mbar_ptr.data_ptr(),
    )
    b_pipeline = pipeline.PipelineTmaUmma.create(
        num_stages=TOKEN_N384_B_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=CTA_TILE_N * CTA_TILE_K * cutlass.BFloat16.width // 8,
        barrier_storage=storage.b_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
    )
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        num_stages=FUSED_ACCUMULATOR_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, TRANSFORM_THREADS),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    )
    operand_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
    operand_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
    b_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, TOKEN_N384_B_STAGES)
    b_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, TOKEN_N384_B_STAGES)
    acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, FUSED_ACCUMULATOR_STAGES)
    acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, FUSED_ACCUMULATOR_STAGES)

    if warp_idx == 0:
        acc_pipeline.producer_acquire(acc_producer_state)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

    scale_u32_per_row = weight_scale.shape[1] // SCALE_BYTES_PER_ROW_TILE
    scale_expert_row_base = expert_idx * weight_scale.shape[0]
    g_scale_u32 = cute.recast_ptr(weight_scale.iterator, dtype=cutlass.Uint32)
    s_scale_u32 = cute.recast_ptr(s_scale.iterator, dtype=cutlass.Uint32)
    num_k_tiles = grouped_work[10]
    for current_tile_k in cutlass.range(num_k_tiles, unroll=1):
        operand_pipeline.producer_acquire(operand_producer_state)
        if warp_idx == 0:
            b_pipeline.producer_acquire(b_producer_state)
            cute.copy(
                tma_atom_b,
                t_b_g_b[(None, 0, current_tile_k, 0)],
                t_b_s_b[(None, b_producer_state.index)],
                tma_bar_ptr=b_pipeline.producer_get_barrier(b_producer_state),
            )
            b_producer_state.advance()
            if has_second_token_slice:
                b_pipeline.producer_acquire(b_producer_state)
                cute.copy(
                    tma_atom_b,
                    t_b_g_b[(None, 1, current_tile_k, 0)],
                    t_b_s_b[(None, b_producer_state.index)],
                    tma_bar_ptr=b_pipeline.producer_get_barrier(b_producer_state),
                )
                b_producer_state.advance()
            if has_third_token_slice:
                b_pipeline.producer_acquire(b_producer_state)
                cute.copy(
                    tma_atom_b,
                    t_b_g_b[(None, 2, current_tile_k, 0)],
                    t_b_s_b[(None, b_producer_state.index)],
                    tma_bar_ptr=b_pipeline.producer_get_barrier(b_producer_state),
                )
                b_producer_state.advance()

        global_row = output_tile_m * cutlass.Int32(CTA_TILE_M) + tidx
        scale_u32_idx = (scale_expert_row_base + global_row) * scale_u32_per_row + current_tile_k
        if tidx < CTA_TILE_M:
            (s_scale_u32 + tidx).store((g_scale_u32 + scale_u32_idx).load())
        transform_inputs_ready = pipeline.NamedBarrier(barrier_id=1, num_threads=TRANSFORM_THREADS)
        transform_inputs_ready.arrive_and_wait()

        _fill_one_m64_a_fragment(
            packed_weight,
            s_scale,
            t_s_a_coords,
            r_a,
            output_tile_m,
            cutlass.Int32(0),
            current_tile_k,
            expert_idx,
        )
        cute.copy(thread_store, r_a, t_d_a)
        transform_store_done = pipeline.NamedBarrier(barrier_id=2, num_threads=TRANSFORM_THREADS)
        transform_store_done.arrive_and_wait()
        cute.arch.fence_view_async_tmem_store()
        operand_pipeline.producer_commit(operand_producer_state)
        operand_producer_state.advance()

        if warp_idx == 0:
            operand_pipeline.consumer_wait(operand_consumer_state)
            b_pipeline.consumer_wait(b_consumer_state)
            if current_tile_k == cutlass.Int32(0):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range_constexpr(cute.size(t_cr_a, mode=[2])):
                cute.gemm(
                    tiled_mma,
                    t_ct_acc_0,
                    t_cr_a[(None, None, k_block)],
                    t_cr_b[(None, None, k_block, b_consumer_state.index)],
                    t_ct_acc_0,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            b_pipeline.consumer_release(b_consumer_state)
            b_consumer_state.advance()

            if has_second_token_slice:
                b_pipeline.consumer_wait(b_consumer_state)
                if current_tile_k == cutlass.Int32(0):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range_constexpr(cute.size(t_cr_a, mode=[2])):
                    cute.gemm(
                        tiled_mma,
                        t_ct_acc_1,
                        t_cr_a[(None, None, k_block)],
                        t_cr_b[(None, None, k_block, b_consumer_state.index)],
                        t_ct_acc_1,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                b_pipeline.consumer_release(b_consumer_state)
                b_consumer_state.advance()

            if has_third_token_slice:
                b_pipeline.consumer_wait(b_consumer_state)
                if current_tile_k == cutlass.Int32(0):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range_constexpr(cute.size(t_cr_a, mode=[2])):
                    cute.gemm(
                        tiled_mma,
                        t_ct_acc_2,
                        t_cr_a[(None, None, k_block)],
                        t_cr_b[(None, None, k_block, b_consumer_state.index)],
                        t_ct_acc_2,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                b_pipeline.consumer_release(b_consumer_state)
                b_consumer_state.advance()

            operand_pipeline.consumer_release(operand_consumer_state)
            operand_consumer_state.advance()

    if warp_idx == 0:
        acc_pipeline.producer_commit(acc_producer_state)
    tmem.relinquish_alloc_permit()
    acc_pipeline.consumer_wait(acc_consumer_state)

    factor_f32 = factor[expert_idx].to(cutlass.Float32)
    output_0 = cute.domain_offset((cutlass.Int64(token_base), output_tile_m * cutlass.Int32(CTA_TILE_M)), physical_output)
    logical_output_0 = cute.make_tensor(
        output_0.iterator,
        cute.make_layout((CTA_TILE_M, CTA_TILE_N), stride=(1, physical_output.shape[1])),
    )
    _store_one_m64_accumulator_ragged(
        tiled_mma,
        t_ct_acc_0,
        logical_output_0,
        factor_f32,
        relu_square,
        first_store_n_extent,
    )
    if has_second_token_slice:
        output_1 = cute.domain_offset(
            (
                cutlass.Int64(token_base) + cutlass.Int64(CTA_TILE_N),
                output_tile_m * cutlass.Int32(CTA_TILE_M),
            ),
            physical_output,
        )
        logical_output_1 = cute.make_tensor(
            output_1.iterator,
            cute.make_layout((CTA_TILE_M, CTA_TILE_N), stride=(1, physical_output.shape[1])),
        )
        _store_one_m64_accumulator_ragged(
            tiled_mma,
            t_ct_acc_1,
            logical_output_1,
            factor_f32,
            relu_square,
            second_store_n_extent,
        )
    if has_third_token_slice:
        output_2 = cute.domain_offset(
            (
                cutlass.Int64(token_base) + cutlass.Int64(2 * CTA_TILE_N),
                output_tile_m * cutlass.Int32(CTA_TILE_M),
            ),
            physical_output,
        )
        logical_output_2 = cute.make_tensor(
            output_2.iterator,
            cute.make_layout((CTA_TILE_M, CTA_TILE_N), stride=(1, physical_output.shape[1])),
        )
        _store_one_m64_accumulator_ragged(
            tiled_mma,
            t_ct_acc_2,
            logical_output_2,
            factor_f32,
            relu_square,
            third_store_n_extent,
        )
    acc_pipeline.consumer_release(acc_consumer_state)

    pipeline.sync(barrier_id=3)
    tmem.free(tmem_ptr)


@cute.jit
def _launch_weight_only_nvfp4_grouped_token_n384(
    packed_weight: cute.Tensor,
    weight_scale: cute.Tensor,
    routed_tokens: cute.Tensor,
    first_token_offset: cute.Tensor,
    physical_output: cute.Tensor,
    factor: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """Validate and launch the exact-FC1 starts-only token-N384 kernel."""

    if cutlass.const_expr(
        cute.rank(packed_weight) != 3
        or packed_weight.shape[0] != FC1_NOUT
        or packed_weight.shape[1] != FC1_K
        or packed_weight.shape[2] < 1
        or packed_weight.shape[2] > GROUPED_MAX_STATIC_SCAN_GROUPS
        or packed_weight.stride != (FC1_K, 1, FC1_NOUT * FC1_K)
    ):
        raise ValueError("token-N384 FC1 packed weight must be checkpoint-native logical [1856,2688,E], E in [1,128]")
    if cutlass.const_expr(
        cute.rank(weight_scale) != 3
        or weight_scale.shape != (FC1_NOUT, FC1_K // NVFP4_GROUP_SIZE, packed_weight.shape[2])
        or weight_scale.stride
        != (
            FC1_K // NVFP4_GROUP_SIZE,
            1,
            FC1_NOUT * (FC1_K // NVFP4_GROUP_SIZE),
        )
    ):
        raise ValueError("token-N384 FC1 scale must use checkpoint-native [E,1856,168] storage")
    if cutlass.const_expr(cute.rank(routed_tokens) != 2 or routed_tokens.shape[1] != FC1_K or routed_tokens.stride != (FC1_K, 1)):
        raise ValueError("token-N384 FC1 tokens must be contiguous BF16 [S,2688]")
    if cutlass.const_expr(
        cute.rank(first_token_offset) != 1
        or first_token_offset.element_type != cutlass.Int32
        or first_token_offset.stride != (1,)
        or first_token_offset.shape[0] != packed_weight.shape[2]
    ):
        raise TypeError("token-N384 first_token_offset must be contiguous starts-only INT32 [G], E=G in [1,128]")
    if cutlass.const_expr(cute.rank(physical_output) != 2 or physical_output.shape[1] != FC1_NOUT or physical_output.stride != (FC1_NOUT, 1)):
        raise ValueError("token-N384 FC1 output must be contiguous row-major BF16 [S,1856]")
    if cutlass.const_expr(
        cute.rank(factor) != 3 or factor.shape != (packed_weight.shape[2], 1, 1) or factor.stride != (1, 1, 1) or factor.element_type != cutlass.Float32
    ):
        raise TypeError("token-N384 factor must be contiguous per-expert FP32 [E,1,1]")
    if cutlass.const_expr(packed_weight.element_type != cutlass.Float4E2M1FN):
        raise TypeError("token-N384 packed weight must be logical E2M1")
    if cutlass.const_expr(weight_scale.element_type != cutlass.Float8E4M3FN):
        raise TypeError("token-N384 weight scale must be E4M3")
    if cutlass.const_expr(routed_tokens.element_type != cutlass.BFloat16):
        raise TypeError("token-N384 routed tokens must be BF16")
    if cutlass.const_expr(physical_output.element_type != cutlass.BFloat16):
        raise TypeError("token-N384 physical output must be BF16")

    num_experts = packed_weight.shape[2]
    num_groups = first_token_offset.shape[0]
    factor_by_expert = cute.make_tensor(factor.iterator, cute.make_layout((num_experts,), stride=(1,)))
    mma_op = tcgen05.MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        FUSED_MMA_INSTRUCTION_MNK,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.TMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        FUSED_TILE_MNK,
        cutlass.BFloat16,
        TOKEN_N384_B_STAGES,
    )
    acc_shape = tiled_mma.partition_shape_C(FUSED_TILE_MNK[:2])
    fake_accumulator = tiled_mma.make_fragment_C(acc_shape)
    num_acc_tmem_cols = utils.get_num_tmem_alloc_cols(fake_accumulator, arch="sm_100")
    if cutlass.const_expr(num_acc_tmem_cols != 128):
        raise ValueError("token-N384 TMEM map requires exactly 128 columns per M64 accumulator")
    if cutlass.const_expr(TOKEN_SLICES * num_acc_tmem_cols + M64_A_TMEM_COLS > TOKEN_N384_TMEM_ALLOCATION_COLS):
        raise ValueError("token-N384 accumulators and decoded-A fragment exceed the 512-column TMEM allocation")
    tma_atom_b, tma_routed_tokens, cluster_layout_vmnk = _make_b_copy_arguments(
        tiled_mma,
        routed_tokens,
        b_smem_layout,
    )
    grid_y = cute.ceil_div(routed_tokens.shape[0], TOKEN_N384) + num_groups - 1
    _weight_only_nvfp4_grouped_token_n384_kernel(
        tiled_mma,
        tma_atom_b,
        tma_routed_tokens,
        cluster_layout_vmnk,
        packed_weight,
        weight_scale,
        routed_tokens,
        first_token_offset,
        physical_output,
        factor_by_expert,
        b_smem_layout,
        num_acc_tmem_cols,
        num_groups,
        num_experts,
        cutlass.Int32(routed_tokens.shape[0]),
        True,
    ).launch(
        grid=(FC1_GRID_X, grid_y, 1),
        block=(TRANSFORM_THREADS, 1, 1),
        cluster=(1, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


@cute.jit
def _logical_views_from_checkpoint_storage(
    packed_weight_storage: cute.Tensor,
    weight_scale_storage: cute.Tensor,
    routed_tokens_storage: cute.Tensor,
    first_token_offset_storage: cute.Tensor,
    physical_output_storage: cute.Tensor,
    factor: cute.Tensor,
    expected_nout: cutlass.Constexpr,
    expected_k: cutlass.Constexpr,
):
    """Validate checkpoint storage and create zero-copy logical kernel views."""

    if cutlass.const_expr(
        cute.rank(packed_weight_storage) != 3
        or packed_weight_storage.shape[1:] != (expected_nout, expected_k // 2)
        or packed_weight_storage.shape[0] < 1
        or packed_weight_storage.shape[0] > GROUPED_MAX_STATIC_SCAN_GROUPS
        or packed_weight_storage.stride != (expected_nout * (expected_k // 2), expected_k // 2, 1)
        or packed_weight_storage.element_type != cutlass.Uint8
    ):
        raise TypeError("packed weight storage must be contiguous UINT8 [E,Nout,K/2], E in [1,128]")
    num_experts = packed_weight_storage.shape[0]

    if cutlass.const_expr(
        cute.rank(weight_scale_storage) != 3
        or weight_scale_storage.shape != (num_experts, expected_nout, expected_k // NVFP4_GROUP_SIZE)
        or weight_scale_storage.stride
        != (
            expected_nout * (expected_k // NVFP4_GROUP_SIZE),
            expected_k // NVFP4_GROUP_SIZE,
            1,
        )
        or weight_scale_storage.element_type != cutlass.Float8E4M3FN
    ):
        raise TypeError("weight scale storage must be contiguous E4M3 [E,Nout,K/16]")

    if cutlass.const_expr(
        cute.rank(routed_tokens_storage) != 3
        or routed_tokens_storage.shape[0] != 1
        or routed_tokens_storage.shape[2] != expected_k
        or routed_tokens_storage.stride[1:] != (expected_k, 1)
        or routed_tokens_storage.element_type != cutlass.BFloat16
    ):
        raise TypeError("routed tokens must be contiguous BF16 [1,S,K]")

    num_groups = first_token_offset_storage.shape[0]
    if cutlass.const_expr(
        cute.rank(first_token_offset_storage) != 3
        or first_token_offset_storage.shape != (num_experts, 1, 1)
        or first_token_offset_storage.stride != (1, 1, 1)
        or first_token_offset_storage.element_type != cutlass.Int32
    ):
        raise TypeError("first_token_offset must be starts-only contiguous INT32 [G,1,1] with G=E")

    if cutlass.const_expr(
        cute.rank(physical_output_storage) != 3
        or physical_output_storage.shape[0] != 1
        or physical_output_storage.shape[2] != expected_nout
        or physical_output_storage.stride[1:] != (expected_nout, 1)
        or physical_output_storage.element_type != cutlass.BFloat16
    ):
        raise TypeError("physical output must be contiguous BF16 [1,S,Nout]")

    if cutlass.const_expr(cute.rank(factor) != 3 or factor.shape != (num_groups, 1, 1) or factor.stride != (1, 1, 1) or factor.element_type != cutlass.Float32):
        raise TypeError("factor must be contiguous FP32 [G,1,1]")

    packed_weight = cute.make_tensor(
        cute.recast_ptr(packed_weight_storage.iterator, dtype=cutlass.Float4E2M1FN),
        cute.make_layout(
            (expected_nout, expected_k, num_experts),
            stride=(expected_k, 1, expected_nout * expected_k),
        ),
    )
    weight_scale = cute.make_tensor(
        weight_scale_storage.iterator,
        cute.make_layout(
            (expected_nout, expected_k // NVFP4_GROUP_SIZE, num_experts),
            stride=(
                expected_k // NVFP4_GROUP_SIZE,
                1,
                expected_nout * (expected_k // NVFP4_GROUP_SIZE),
            ),
        ),
    )
    routed_tokens = cute.make_tensor(
        routed_tokens_storage.iterator,
        cute.make_layout(
            (routed_tokens_storage.shape[1], expected_k),
            stride=(expected_k, 1),
        ),
    )
    first_token_offset = cute.make_tensor(
        first_token_offset_storage.iterator,
        cute.make_layout((num_groups,), stride=(1,)),
    )
    physical_output = cute.make_tensor(
        physical_output_storage.iterator,
        cute.make_layout(
            (routed_tokens_storage.shape[1], expected_nout),
            stride=(expected_nout, 1),
        ),
    )
    return (
        packed_weight,
        weight_scale,
        routed_tokens,
        first_token_offset,
        physical_output,
    )


@cute.jit
def weight_only_nvfp4_lightning_fc1_sm100(
    packed_weight: cute.Tensor,
    weight_scale: cute.Tensor,
    routed_tokens: cute.Tensor,
    first_token_offset: cute.Tensor,
    physical_output: cute.Tensor,
    factor: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """Run Lightning FC1 with the expert-local token-N384 schedule."""

    packed_weight, weight_scale, routed_tokens, first_token_offset, physical_output = _logical_views_from_checkpoint_storage(
        packed_weight,
        weight_scale,
        routed_tokens,
        first_token_offset,
        physical_output,
        factor,
        FC1_NOUT,
        FC1_K,
    )
    _launch_weight_only_nvfp4_grouped_token_n384(
        packed_weight,
        weight_scale,
        routed_tokens,
        first_token_offset,
        physical_output,
        factor,
        stream,
    )


@cute.jit
def weight_only_nvfp4_lightning_fc2_sm100(
    packed_weight: cute.Tensor,
    weight_scale: cute.Tensor,
    routed_tokens: cute.Tensor,
    first_token_offset: cute.Tensor,
    physical_output: cute.Tensor,
    factor: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """Run Lightning FC2 with the output-M192 schedule."""

    packed_weight, weight_scale, routed_tokens, first_token_offset, physical_output = _logical_views_from_checkpoint_storage(
        packed_weight,
        weight_scale,
        routed_tokens,
        first_token_offset,
        physical_output,
        factor,
        FC2_NOUT,
        FC2_K,
    )
    _launch_weight_only_nvfp4_grouped_m192(
        packed_weight,
        weight_scale,
        routed_tokens,
        first_token_offset,
        physical_output,
        factor,
        stream,
    )


__all__: tuple[str, ...] = ()
