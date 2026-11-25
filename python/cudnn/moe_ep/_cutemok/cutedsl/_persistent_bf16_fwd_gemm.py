# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Cluster-2 BF16 GEMM collective of the forward megakernel (SM100 / SM103): the stage counts, warp
roles and shared-memory sizes of a compute CTA, and the staged TMA block load."""

from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import cutlass.cute as cute
from cutlass import BFloat16, Float32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import LayoutEnum, block_copy

from ._compat.gemm_sm100 import GemmDefaultSm100, VarlenArguments


from .persistent_bf16_fwd_contract import (
    AB_STAGES,
    ACC_STAGES,
    AB_LOAD_WARP as TMA_WARP,
    BF16_IDLE_WARP,
    CLC_WARP,
    CLUSTER_SIZE,
    D_STAGES,
    FC1_CTA_M as CTA_M,
    FC1_EPILOGUE_N as EPILOGUE_N,
    FC1_LOGICAL_N as LOGICAL_N,
    FC1_PACKED_N as PACKED_N,
    MLP_K_TILE as K_TILE,
    MLP_TILE_ROWS as TILE_M,
    MMA_WARP,
    THREADS_PER_CTA as THREADS,
)

CLUSTER_SHAPE = (CLUSTER_SIZE, 1, 1)
IDLE_WARPS = (CLC_WARP, BF16_IDLE_WARP)

A_BYTES = AB_STAGES * CTA_M * K_TILE * 2
B_BYTES = AB_STAGES * LOGICAL_N * K_TILE * 2
D_BYTES = D_STAGES * CTA_M * EPILOGUE_N * 2
ARENA_BYTES = A_BYTES + B_BYTES + D_BYTES
assert ARENA_BYTES == 221184


@dsl_user_op
def _make_tma_block_load_fn(
    atom: cute.CopyAtom,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    *,
    tma_desc_ptr=None,
    loc=None,
    ip=None,
):
    """Bind one staged GMEM-to-SMEM load through CUTLASS ``block_copy``.

    ``tma_desc_ptr`` overrides the descriptor the atom carries, which is how a routed weight held
    in split per-expert allocations is read: the atom is built over one representative expert and
    ``routed_descriptors`` supplies the rest, since the allocations differ only in base address.
    Leaving it ``None`` uses the atom's own descriptor, which is every other load here.
    """

    src = cute.group_modes(
        src_tensor,
        0,
        cute.rank(src_tensor) - 1,
        loc=loc,
        ip=ip,
    )
    dst = cute.group_modes(
        dst_tensor,
        0,
        cute.rank(dst_tensor) - 1,
        loc=loc,
        ip=ip,
    )

    @dsl_user_op
    def copy_tma(src_idx, dst_idx, *, loc=None, ip=None, **kwargs):
        if tma_desc_ptr is not None:
            kwargs.setdefault("tma_desc_ptr", tma_desc_ptr)
        block_copy(
            atom,
            src[None, src_idx],
            dst[None, dst_idx],
            loc=loc,
            ip=ip,
            **kwargs,
        )

    return copy_tma


class _Fc1Collective(GemmDefaultSm100):
    """Cluster-2 collective for one Gate/Up tile."""

    @classmethod
    def _compute_stages(cls, *args, **kwargs):
        del args, kwargs
        return ACC_STAGES, AB_STAGES, D_STAGES, 0

    def __init__(self) -> None:
        super().__init__(
            acc_dtype=Float32,
            a_dtype=BFloat16,
            mma_tiler_mnk=(TILE_M, PACKED_N, K_TILE),
            cluster_shape_mnk=CLUSTER_SHAPE,
            use_clc_persistence=False,
            use_pdl=False,
        )
        self.mma_warp_id = MMA_WARP
        self.scheduler_warp_id = IDLE_WARPS[0]
        self.epi_load_warp_id = IDLE_WARPS[1]
        self.ab_load_warp_id = TMA_WARP
        assert self.epilog_warp_id == (0, 1, 2, 3)
        assert self.threads_per_cta == THREADS

    def configure(
        self,
        mA_mk: cute.Tensor,
        mB_gate_nk: cute.Tensor,
        mD_mn: cute.Tensor,
    ) -> None:
        self.a_dtype = mA_mk.element_type
        self.b_dtype = mB_gate_nk.element_type
        self.a_mma_dtype = self.a_dtype
        self.b_mma_dtype = self.b_dtype
        self.a_unpack = False
        self.b_unpack = False
        self.a_smem_dtype = self.a_dtype
        self.b_smem_dtype = self.b_dtype
        self.d_dtype = mD_mn.element_type
        self.c_dtype = None
        self.sf_dtype = None
        self.a_layout = LayoutEnum.from_tensor(mA_mk)
        self.b_layout = LayoutEnum.from_tensor(mB_gate_nk)
        self.d_layout = LayoutEnum.from_tensor(mD_mn)
        self.c_layout = None
        self.a_major_mode = self.a_layout.mma_major_mode()
        self.b_major_mode = self.b_layout.mma_major_mode()
        self.varlen_m = False
        self.varlen_k = False
        self._setup_attributes(self.EpilogueArguments(), VarlenArguments())

        assert self.a_dtype is BFloat16
        assert self.b_dtype is BFloat16
        assert self.d_dtype is BFloat16
        assert self.use_2cta_instrs
        assert self.cta_tile_shape_mnk == (CTA_M, PACKED_N, K_TILE)
        assert cute.cosize(self.a_smem_layout_staged.outer) == (AB_STAGES * CTA_M * K_TILE)
        assert cute.cosize(self.b_smem_layout_staged.outer) == (AB_STAGES * LOGICAL_N * K_TILE)


__all__ = ["_Fc1Collective", "_make_tma_block_load_fn"]
