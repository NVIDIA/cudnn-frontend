# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

"""
MoE Scheduler Extensions for Block-Scaled Grouped GEMM.

Bridges the MoE tile scheduler (MoEPersistentTileScheduler) with tensor-level
domain conversion and TMA descriptor selection.

Two concrete extensions are provided:

    DiscreteWeightScaledGemmSchedExtension
        B/SFB are per-expert pointer arrays → expert-wise TMA descriptors
        A/C/D/SFA are contiguous → global TMA descriptors with domain_offset

    ContiguousAndConsistentGroupedGemmSchedExtension
        ALL tensors are contiguous with consistent padding
        B/SFB are 3D (N, K, L) → domain_offset along L by expert_idx
        A/C/D/SFA are contiguous → domain_offset along M by token_offset
        All use global TMA descriptors (no per-expert workspace needed)

Architecture:

    Scheduler ──(produces)──> MoEWorkTileInfo
                                    │
                           expert_idx, tile_m, tile_n, k_cnt
                                    │
                                    v
    Extension ──(uses)──> OnlineTensormapDescCreator  (discrete only)
        │                         │
        │  get_gmem_tensor()      │  get_desc_ptr()
        │                         │  construct_and_write()
        │                         │
        └── internal calls ───────┘

    Kernel (caller): the only place that knows all three exist
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import Int32, extract_mlir_values, new_from_mlir_values

import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from .moe_utils import (
    OnlineTensormapDescCreator,
    tensormap_ptr_for_copy,
    compute_expert_token_range,
    rewrite_tensor_shape,
)
from .moe_persistent_scheduler import MoEWorkTileInfo


class MoESchedExtension(ABC):
    """
    Abstract base class for MoE scheduler extensions.

    Bridges MoEWorkTileInfo with tensor-level domain conversion and TMA
    descriptor selection.
    """

    def __init__(self, tensormap_ctor: "OnlineTensormapDescCreator | None" = None):
        super().__init__()
        self.tensormap_ctor = tensormap_ctor

    @abstractmethod
    def get_gmem_tensor(
        self,
        tensor_name: str,
        gmem_tensor_in_moe_view: cute.Tensor,
        offs: cute.Tensor,
        work_tile_info: MoEWorkTileInfo,
    ) -> Tuple[cute.Tensor, "Pointer | None"]:
        """
        Convert an MoE-view tensor to the real per-expert tensor for the
        current work tile, and return the appropriate TMA descriptor pointer.

        :param tensor_name: Identifies which tensor (e.g., "a", "b", "c", "sfa", "sfb")
        :param gmem_tensor_in_moe_view: Tensor in fake GEMM MNKL domain
        :param offs: Cumsum tensor (experts,) — padded_offsets for this kernel
        :param work_tile_info: Current work tile from the scheduler
        :return: (real_tensor, tma_desc_ptr_or_none)
        """
        ...


# =============================================================================
# Discrete Weight Scaled Grouped GEMM Extension
# =============================================================================


class DiscreteWeightScaledGemmSchedExtension(MoESchedExtension):
    """
    MoE scheduler extension for discrete-weight block-scaled grouped GEMM
    with GLU and quantization fusion.

    Handles domain conversion for: a, b, c, d, d_col, prob, dprob, sfa, sfd, sfd_col, sfb.

    B and SFB are discrete (per-expert pointer arrays) → use expert-wise
    TMA descriptors from workspace.

    A, C, D, SFA are contiguous across experts (indexed by padded M offset)
    → use global TMA descriptors with domain_offset.

    Domain conversion:
        A:               (total_padded_M, K, 1)     → domain_offset M by token_offset, global desc
        B:               template (N, K, 1)          → rewrite L to dynamic 1, expert-wise desc
        C/D/D_col/prob/dprob:
                         (total_padded_M, N_dim, 1)  → domain_offset M by token_offset, global desc
        SFA/SFD:         (total_padded_M, K_or_N, 1) → domain_offset M by token_offset,
                                                        tile_atom_to_shape_SF layout, global desc
        SFD_col:         (total_padded_M, N, 1)      → domain_offset M by token_offset,
                                                        BlockScaledBasicChunk layout, global desc
        SFB:             template                     → tile_atom_to_shape_SF layout, expert-wise desc

    :param tensormap_ctor: DiscreteWeightTensormapConstructor for B/SFB descs
    :param sf_vec_size: Scale factor vector size
    """

    def __init__(
        self,
        tensormap_ctor: OnlineTensormapDescCreator,
        sf_vec_size: int,
    ):
        super().__init__(tensormap_ctor)
        self.sf_vec_size = sf_vec_size

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.tensormap_ctor)

    def __new_from_mlir_values__(self, values):
        new_ctor = new_from_mlir_values(self.tensormap_ctor, values)
        return DiscreteWeightScaledGemmSchedExtension(
            tensormap_ctor=new_ctor,
            sf_vec_size=self.sf_vec_size,
        )

    def update_expert_info(self, offs, expert_idx):
        self.token_offset, self.tokens_i = compute_expert_token_range(offs, expert_idx)

    @cute.jit
    def get_gmem_tensor(
        self,
        tensor_name: str,
        gmem_tensor_in_moe_view: cute.Tensor,
        offs: cute.Tensor,
        work_tile_info: MoEWorkTileInfo,
    ):
        expert_idx = work_tile_info.expert_idx
        if cutlass.const_expr(hasattr(self, "token_offset")):
            token_offset, tokens_i = self.token_offset, self.tokens_i
        else:
            token_offset, tokens_i = compute_expert_token_range(offs, expert_idx)

        shape = gmem_tensor_in_moe_view.shape
        c1 = cutlass.Int32(1)

        if cutlass.const_expr(tensor_name == "a"):
            # A: (total_padded_M, K, 1) → offset M by token_offset, global desc
            real = cute.domain_offset((token_offset, 0, 0), gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(real, (tokens_i, shape[1], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "b"):
            # B: discrete — rewrite L to dynamic c1, expert-wise desc
            real = rewrite_tensor_shape(gmem_tensor_in_moe_view, (shape[0], shape[1], c1))
            desc = tensormap_ptr_for_copy(self.tensormap_ctor.get_desc_ptr("b", expert_idx))
            return (real, desc)

        elif cutlass.const_expr(tensor_name == "bias"):
            # Bias: (N, L) → domain_offset L by expert_idx, global desc
            real = cute.domain_offset((0, expert_idx), gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(real, (shape[0], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name in ("c", "d", "d_col", "prob", "dprob")):
            # C/D/D_col/prob: contiguous M, offset by token_offset, global desc
            real = cute.domain_offset((token_offset, 0, 0), gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(real, (tokens_i, shape[1], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name in ("sfa", "sfd")):
            # SFA/SFD: contiguous with tile_atom_to_shape_SF layout,
            # offset in GEMM-element units (NOT SF units),
            # because tile_atom_to_shape_SF layout's M stride is per-element.
            real = cute.domain_offset((token_offset, 0, 0), gmem_tensor_in_moe_view)
            per_expert_shape = (tokens_i, shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, self.sf_vec_size)
            stride = gmem_tensor_in_moe_view.stride
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "sfd_col"):
            # SFD Col with BlockScaledBasicChunk layout (non-atom):
            # domain_offset + rebuild with tile_to_shape using per-expert M
            real = cute.domain_offset((token_offset, 0, 0), gmem_tensor_in_moe_view)
            per_expert_shape = (tokens_i, shape[1], c1)
            sfd_col_layout = cute.tile_to_shape(
                blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size, OperandMajorMode.MN).layout,
                per_expert_shape,
                (1, 2, 3),
            )
            real = cute.make_tensor(real.iterator, cute.make_layout(sfd_col_layout.shape, stride=gmem_tensor_in_moe_view.stride))
            return (real, None)

        else:  # "sfb"
            # SFB: discrete — rewrite with tile_atom_to_shape_SF, expert-wise desc
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, self.sf_vec_size)
            stride = gmem_tensor_in_moe_view.stride
            real = cute.make_tensor(
                gmem_tensor_in_moe_view.iterator,
                cute.make_layout(sf_layout.shape, stride=stride),
            )
            desc = tensormap_ptr_for_copy(self.tensormap_ctor.get_desc_ptr("sfb", expert_idx))
            return (real, desc)


# =============================================================================
# Contiguous & Consistent Grouped GEMM Extension
# =============================================================================


class ContiguousAndConsistentGroupedGemmSchedExtension(MoESchedExtension):
    """
    MoE scheduler extension for contiguous block-scaled grouped GEMM where
    ALL tensors share a consistent padding scheme.

    All tensors use global TMA descriptors (no per-expert workspace needed).

    Domain conversion:
        A:               (total_padded_M, K, 1)  → domain_offset M by token_offset
        B:               (N, K, L)               → domain_offset L by expert_idx
        C/D/D_col/prob/dprob:
                         (total_padded_M, N, 1)  → domain_offset M by token_offset
        SFA/SFD:         (total_padded_M, K_or_N, 1) → domain_offset M, tile_atom_to_shape_SF
        SFB:             (N, K, L)               → domain_offset L by expert_idx,
                                                    tile_atom_to_shape_SF
        SFD_col:         (total_padded_M, N, 1)  → domain_offset M,
                                                    BlockScaledBasicChunk layout

    :param sf_vec_size: Scale factor vector size
    """

    def __init__(self, sf_vec_size: int):
        super().__init__(tensormap_ctor=None)
        self.sf_vec_size = sf_vec_size

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return ContiguousAndConsistentGroupedGemmSchedExtension(
            sf_vec_size=self.sf_vec_size,
        )

    def update_expert_info(self, offs, expert_idx):
        self.token_offset, self.tokens_i = compute_expert_token_range(offs, expert_idx)

    @cute.jit
    def get_gmem_tensor(
        self,
        tensor_name: str,
        gmem_tensor_in_moe_view: cute.Tensor,
        offs: cute.Tensor,
        work_tile_info: MoEWorkTileInfo,
    ):
        expert_idx = work_tile_info.expert_idx
        if cutlass.const_expr(hasattr(self, "token_offset")):
            token_offset, tokens_i = self.token_offset, self.tokens_i
        else:
            token_offset, tokens_i = compute_expert_token_range(offs, expert_idx)

        shape = gmem_tensor_in_moe_view.shape
        c1 = cutlass.Int32(1)

        if cutlass.const_expr(tensor_name == "a"):
            real = cute.domain_offset((token_offset, 0, 0), gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(real, (tokens_i, shape[1], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "b"):
            # B: (N, K, L) → domain_offset along L by expert_idx, global desc
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(real, (shape[0], shape[1], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "bias"):
            # Bias: (N, L) → domain_offset L by expert_idx, global desc
            real = cute.domain_offset((0, expert_idx), gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(real, (shape[0], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name in ("c", "d", "d_col", "prob", "dprob")):
            real = cute.domain_offset((token_offset, 0, 0), gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(real, (tokens_i, shape[1], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name in ("sfa", "sfd")):
            real = cute.domain_offset((token_offset, 0, 0), gmem_tensor_in_moe_view)
            per_expert_shape = (tokens_i, shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, self.sf_vec_size)
            stride = gmem_tensor_in_moe_view.stride
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "sfd_col"):
            real = cute.domain_offset((token_offset, 0, 0), gmem_tensor_in_moe_view)
            per_expert_shape = (tokens_i, shape[1], c1)
            sfd_col_layout = cute.tile_to_shape(
                blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size, OperandMajorMode.MN).layout,
                per_expert_shape,
                (1, 2, 3),
            )
            real = cute.make_tensor(real.iterator, cute.make_layout(sfd_col_layout.shape, stride=gmem_tensor_in_moe_view.stride))
            return (real, None)

        else:  # "sfb"
            # SFB: (N, K, L) → domain_offset along L by expert_idx,
            # then apply tile_atom_to_shape_SF layout, global desc
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, self.sf_vec_size)
            stride = gmem_tensor_in_moe_view.stride
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)
