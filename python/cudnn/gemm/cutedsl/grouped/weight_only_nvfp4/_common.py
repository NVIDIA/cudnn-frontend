# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SM100 definitions for Lightning weight-only NVFP4 kernels.

The checkpoint stores packed weights physically as ``[E, Nout, K/2]`` and
E4M3 group-16 scales as ``[E, Nout, K/16]``.  The kernels consume logical
``[Nout, K, E]`` views over exactly that storage and decode one M64xK64 tile
directly into TMEM.  No decoded-weight workspace is materialized.
"""

import cutlass
from cutlass import cute
from cutlass.utils import blackwell_helpers as sm100_utils

NVFP4_GROUP_SIZE = 16
CTA_TILE_M = 64
CTA_TILE_N = 128
CTA_TILE_K = 64
CLUSTER_SHAPE_MN = (1, 1)
TRANSFORM_THREADS = 128
SCALE_BYTES_PER_ROW_TILE = CTA_TILE_K // NVFP4_GROUP_SIZE

FUSED_TILE_MNK = (CTA_TILE_M, CTA_TILE_N, CTA_TILE_K)
FUSED_MMA_INSTRUCTION_MNK = (CTA_TILE_M, CTA_TILE_N, 16)
FUSED_ACCUMULATOR_STAGES = 1

GROUPED_WORK_TILE_WORDS = 10
GROUPED_MAX_STATIC_SCAN_GROUPS = 128
GROUPED_SCHEDULER_BARRIER_ID = 4

FC1_NOUT = 1856
FC1_K = 2688
FC1_GRID_X = FC1_NOUT // CTA_TILE_M
FC2_NOUT = 2688
FC2_K = 1856

M_SLICES = 3
SUPERTILE_M = M_SLICES * CTA_TILE_M
FC2_M192_GRID_X = FC2_NOUT // SUPERTILE_M
M64_A_TMEM_COLS = 32
M192_TMEM_ALLOCATION_COLS = 512

TOKEN_SLICES = 3
TOKEN_N384 = TOKEN_SLICES * CTA_TILE_N
TOKEN_N384_B_STAGES = 3
TOKEN_N384_TMEM_ALLOCATION_COLS = 512
TOKEN_N384_WORK_TILE_WORDS = 11


@cute.struct
class _M192SharedStorage:
    operand_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    grouped_work_words: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, GROUPED_WORK_TILE_WORDS],
        16,
    ]
    tmem_holding_buf: cutlass.Int32


@cute.struct
class _TokenN384SharedStorage:
    operand_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, TOKEN_N384_B_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    grouped_work_words: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, TOKEN_N384_WORK_TILE_WORDS],
        16,
    ]
    tmem_holding_buf: cutlass.Int32


@cute.jit
def _fill_one_m64_a_fragment(
    packed_weight: cute.Tensor,
    s_scale: cute.Tensor,
    t_s_a_coords: cute.Tensor,
    r_a: cute.Tensor,
    output_tile_m: cutlass.Int32,
    scale_row_offset: cutlass.Int32,
    current_tile_k: cutlass.Int32,
    expert_idx: cutlass.Int32,
) -> None:
    """Decode one M64xK64 packed-weight slice into its RF fragment."""

    for k_block in cutlass.range_constexpr(CTA_TILE_K // 16):
        for row_group in cutlass.range_constexpr(2):
            coord_idx = (((0, row_group), 0), 0, 0, k_block)
            local_coord = t_s_a_coords[coord_idx]
            local_row = cutlass.Int32(cute.get(local_coord, mode=[0]))
            local_k = cutlass.Int32(cute.get(local_coord, mode=[1]))
            global_row = output_tile_m * cutlass.Int32(CTA_TILE_M) + local_row
            global_k = current_tile_k * cutlass.Int32(CTA_TILE_K) + local_k
            weight_u16_per_row = packed_weight.shape[1] // 4
            weight_u16_idx = (expert_idx * packed_weight.shape[0] + global_row) * weight_u16_per_row + global_k // 4
            g_weight_u16 = cute.recast_ptr(packed_weight.iterator, dtype=cutlass.Uint16)
            r_weight_u16 = cute.make_rmem_tensor((1,), cutlass.Uint16)
            r_weight_u16[0] = (g_weight_u16 + weight_u16_idx).load()
            r_fp4 = cute.make_tensor(
                cute.recast_ptr(r_weight_u16.iterator, dtype=cutlass.Float4E2M1FN),
                cute.make_layout((4,), stride=(1,)),
            )
            weight_f32 = r_fp4.load().to(cutlass.Float16).to(cutlass.Float32)
            scale_f32 = s_scale[(scale_row_offset + local_row, local_k // NVFP4_GROUP_SIZE)].to(cutlass.Float32)
            r_scale = cute.make_rmem_tensor((4,), cutlass.Float32)
            for elem in cutlass.range_constexpr(4):
                r_scale[elem] = scale_f32
            scaled_bf16 = (weight_f32 * r_scale.load()).to(cutlass.BFloat16)
            for elem in cutlass.range_constexpr(4):
                r_a[(((elem, row_group), 0), 0, 0, k_block)] = scaled_bf16[elem]


@cute.jit
def _make_b_copy_arguments(
    tiled_mma: cute.TiledMma,
    routed_tokens: cute.Tensor,
    b_smem_layout: cute.ComposedLayout,
):
    """Build the TMA descriptor for contiguous BF16 routed tokens."""

    cluster_layout_mnk = cute.make_layout((1, 1, 1))
    cluster_layout_vmnk = cute.tiled_divide(cluster_layout_mnk, (tiled_mma.thr_id,))
    b_nkl = cute.make_tensor(
        routed_tokens.iterator,
        cute.make_layout(
            (routed_tokens.shape[0], routed_tokens.shape[1], 1),
            stride=(
                routed_tokens.shape[1],
                1,
                routed_tokens.shape[0] * routed_tokens.shape[1],
            ),
        ),
    )
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(
        (*CLUSTER_SHAPE_MN, 1),
        tiled_mma.thr_id,
    )
    b_smem_layout_stage = cute.slice_(b_smem_layout, (None, None, None, 0))
    tma_atom_b, tma_routed_tokens = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        b_nkl,
        b_smem_layout_stage,
        FUSED_TILE_MNK,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    return tma_atom_b, tma_routed_tokens, cluster_layout_vmnk


__all__: tuple[str, ...] = ()
