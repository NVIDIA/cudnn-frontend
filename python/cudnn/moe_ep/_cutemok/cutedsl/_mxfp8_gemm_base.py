# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""BF16 GEMM collective of the MXFP8 megakernels: the layouts and copy atoms of a 2-CTA
256 x 256 x 64 GEMM with a six-stage A/B ring, one accumulator stage and three D staging tiles
(one for an FP32 D), and the D-tile store paths.

There is no task scheduler here: the megakernels create the pipelines, decode the tasks and issue
the synchronization themselves.  The store paths return once their bulk stores have finished
reading shared memory (the arena may be reused); ``finish_store`` waits for the stores to complete,
which has to precede any readiness or buffer-done arrival.
"""

from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import LayoutEnum, block_copy

from ._compat.gemm_sm100 import GemmDefaultSm100, VarlenArguments
from ._bwd_device_helpers import named_barrier_sync_unaligned

TILE_M = 256
CTA_B_ROWS = 128
TILE_N = 256
CTA_M = 128
K_TILE = 64
CLUSTER_SHAPE = (2, 1, 1)
THREADS = 256

# Every D tile is stored with the evict_first cache policy (createpolicy's fractional
# evict_first encoding).  The 1D communication copies in _tma_1d.py carry no cache hint.
TMA_CACHE_HINT_EVICT_FIRST = 0x12F0000000000000
AB_STAGES = 6
ACC_STAGES = 1
D_STAGES = 3
EPILOGUE_N = 32
MMA_WARP = 4
IDLE_WARPS = (5, 6)
TMA_WARP = 7

A_BYTES = AB_STAGES * CTA_M * K_TILE * 2
B_BYTES = AB_STAGES * CTA_B_ROWS * K_TILE * 2
D_BYTES = D_STAGES * CTA_M * EPILOGUE_N * 2
# One FP32 Wgrad staging tile has to fit where the three BF16 ones do.
assert CTA_M * EPILOGUE_N * 4 <= D_BYTES
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


class _Bf16Gemm(GemmDefaultSm100):
    """Layouts and copy atoms only (QuACK's SM100 GEMM with fixed stage counts and warp roles);
    the megakernels issue the synchronization explicitly."""

    #: What ``configure`` requires of D, and how many epilogue staging tiles it keeps.  The Wgrad
    #: collective writes FP32 when the weight gradients accumulate into float32 main_grad buffers.
    d_dtype_expected = BFloat16
    epi_stages = D_STAGES

    @classmethod
    def _compute_stages(cls, *args, **kwargs):
        del args, kwargs
        return ACC_STAGES, AB_STAGES, cls.epi_stages, 0

    def __init__(self) -> None:
        super().__init__(
            acc_dtype=Float32,
            a_dtype=BFloat16,
            mma_tiler_mnk=(TILE_M, TILE_N, K_TILE),
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
        mB_nk: cute.Tensor,
        mD_mn: cute.Tensor,
    ) -> None:
        self.a_dtype = mA_mk.element_type
        self.b_dtype = mB_nk.element_type
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
        self.b_layout = LayoutEnum.from_tensor(mB_nk)
        self.d_layout = LayoutEnum.from_tensor(mD_mn)
        self.c_layout = None
        self.a_major_mode = self.a_layout.mma_major_mode()
        self.b_major_mode = self.b_layout.mma_major_mode()
        self.varlen_m = False
        self.varlen_k = False
        self._setup_attributes(self.EpilogueArguments(), VarlenArguments())
        # Transaction bytes the leader CTA arms per stage for both CTAs
        # (PipelineTmaUmma tx_count).
        self.num_tma_load_bytes = (
            cute.size_in_bytes(BFloat16, cute.slice_(self.a_smem_layout_staged, (None, None, None, 0)))
            + cute.size_in_bytes(BFloat16, cute.slice_(self.b_smem_layout_staged, (None, None, None, 0)))
        ) * cute.size(self.tiled_mma.thr_id.shape)

        assert self.a_dtype is BFloat16
        assert self.b_dtype is BFloat16
        assert self.d_dtype is self.d_dtype_expected
        assert self.use_2cta_instrs
        assert self.cta_tile_shape_mnk == (CTA_M, TILE_N, K_TILE)
        assert cute.cosize(self.a_smem_layout_staged.outer) == (AB_STAGES * CTA_M * K_TILE)
        assert cute.cosize(self.b_smem_layout_staged.outer) == (AB_STAGES * CTA_B_ROWS * K_TILE)
        # store_acc_all walks the CTA tile as eight D subtiles rotating through epi_stages staging
        # tiles of the arena's D region; an FP32 D keeps one where a BF16 one keeps three.
        assert cute.size(cute.zipped_divide(cute.make_layout(self.cta_tile_shape_mnk[:2]), self.epi_tile).shape[1]) == 8
        assert (cute.cosize(self.epi_smem_layout_staged.outer) * self.d_dtype.width // 8) <= D_BYTES


class _Bf16GemmFp32D(_Bf16Gemm):
    """The Wgrad collective writing FP32 weight gradients straight into main_grad.

    The accumulator is FP32 already, so a float32 main_grad never sees a bfloat16 rounding.
    One staging tile where the BF16 epilogue keeps three: one FP32 tile is the smaller of the
    two, and more would spill.
    """

    d_dtype_expected = Float32
    epi_stages = 1


# The mainloop is driven by QuACK's PipelineTmaUmma / PipelineUmmaAsync and
# GemmDefaultSm100.load_tma / mma / epi_load_acc_subtile; only the D-tile store paths
# below are hand-written (eight subtile stores through the D staging tiles).


@cute.jit
def _is_store_warp(thread: Int32):
    """Whether this thread's warp (warp 0) issues the D stores and waits for them, as a warp-uniform value.

    The store paths wait for the staging tile's previous store right in front of an epilogue barrier.  Under a
    per-thread condition (``thread == 0``, ``thread < 32``) that wait is a branch the assembler has to treat as a
    possible divergence of the warp, and it then guards the unaligned barrier behind it with a divergence check
    and an out-of-line path; a branch on a warp-uniform value is none.  The whole warp may take it: the TMA store
    is issued once per warp, and a lane without bulk stores of its own does not wait."""
    return cute.arch.make_warp_uniform(thread >> Int32(5)) == Int32(0)


@cute.jit
def store_acc_all(collective, registers, tiled_copy_r2s, tRS_sD, copy_D, thread: Int32, copy_D_add=None, first_contribution: Int32 = 1):
    """Stores the CTA's D tile, held in ``registers``, as eight subtiles through the D staging
    tiles.  ``copy_D`` is the plain store; with ``copy_D_add`` (a Wgrad) a ``first_contribution``
    stores plainly and every other one reduce-adds.

    Returns when the stores have finished reading shared memory, which permits arena reuse.
    Call ``finish_store`` before publishing any global ready / buffer-done counter.
    """
    # The staging tiles rotate, so a subtile may only overwrite one whose store has finished
    # reading it: all but one of them may still be in flight.  Three BF16 tiles leave two, the
    # single FP32 tile of an accumulating Wgrad leaves none.
    in_flight = collective.epi_stages - 1
    is_store_warp = _is_store_warp(thread)
    for subtile in cutlass.range_constexpr(8):
        if is_store_warp:
            cute.arch.cp_async_bulk_wait_group(in_flight, read=True)
        named_barrier_sync_unaligned(collective.epilogue_barrier)
        cute.copy(tiled_copy_r2s, registers[None, None, None, subtile], tRS_sD[None, None, None, subtile % collective.epi_stages])
        cute.arch.fence_view_async_shared()
        named_barrier_sync_unaligned(collective.epilogue_barrier)
        if is_store_warp:
            if cutlass.const_expr(copy_D_add is not None):
                if first_contribution != Int32(0):
                    copy_D(src_idx=subtile % collective.epi_stages, dst_idx=(0, subtile), cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
                else:
                    copy_D_add(src_idx=subtile % collective.epi_stages, dst_idx=(0, subtile), cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
            else:
                copy_D(src_idx=subtile % collective.epi_stages, dst_idx=(0, subtile), cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
            cute.arch.cp_async_bulk_commit_group()
    if is_store_warp:
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    named_barrier_sync_unaligned(pipeline.NamedBarrier(barrier_id=4, num_threads=128))


@cute.jit
def store_acc_fp32(collective, tiled_copy_t2r, tiled_copy_r2s, tTR_tAcc, tTR_rAcc, tRS_rD, tRS_sD, copy_D_add, thread: Int32, acc_pipeline, acc_consumer_state):
    """The epilogue of a Wgrad accumulating into a float32 main_grad: eight reduce-adds.

    The Tensor Core accumulator is FP32 already, so a float32 main_grad never sees a bfloat16
    rounding -- the destination takes the accumulator as it is.  Where ``store_acc_all`` reads all
    eight subtiles out of TMEM first and then stores them, this reads one at a time and keeps a
    single register tile live: several FP32 epilogue tiles live at once spill to local memory
    (eight would be 256 registers a thread).

    Every contribution is additive -- main_grad already holds the earlier microbatches -- so
    ``copy_D_add`` is the only store, and one FP32 staging tile fits where the BF16 three do.
    """
    assert collective.epi_stages == 1
    is_store_warp = _is_store_warp(thread)
    for subtile in cutlass.range_constexpr(8):
        collective.epi_load_acc_subtile(tiled_copy_t2r, tiled_copy_r2s, tTR_tAcc, tTR_rAcc, tRS_rD, (0, subtile), acc_pipeline, acc_consumer_state, 7)
        if is_store_warp:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        named_barrier_sync_unaligned(collective.epilogue_barrier)
        cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, 0])
        cute.arch.fence_view_async_shared()
        named_barrier_sync_unaligned(collective.epilogue_barrier)
        if is_store_warp:
            copy_D_add(src_idx=0, dst_idx=(0, subtile), cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
            cute.arch.cp_async_bulk_commit_group()
    if is_store_warp:
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    named_barrier_sync_unaligned(pipeline.NamedBarrier(barrier_id=4, num_threads=128))


@cute.jit
def finish_store(thread: Int32):
    """Thread 0 waits until its bulk stores have completed, not only finished reading shared memory."""
    if thread == Int32(0):
        cute.arch.cp_async_bulk_wait_group(0, read=False)


__all__ = ["_Bf16Gemm", "_is_store_warp", "_make_tma_block_load_fn", "store_acc_all", "store_acc_fp32", "finish_store"]
