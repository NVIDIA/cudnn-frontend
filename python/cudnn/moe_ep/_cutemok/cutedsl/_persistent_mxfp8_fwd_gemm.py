# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Routed MXFP8 GEMM pieces of the MXFP8 megakernels (forward and backward).

The BF16 GEMM collective is ``_mxfp8_gemm_base``'s; this module adds the block-scaled collective
(``_Mxfp8Gemm``), the scale ring's loads, the per-stage TMEM scale slots, the block-scaled mainloop
(``mma_blockscaled``) and the Gate / Up epilogues: the forward's fused Gate / Up + SwiGLU epilogue
(``store_acc_fused_swiglu`` with a BF16 Gate / Up context, ``store_acc_fused_swiglu_e4m3_context`` with an e4m3
one) and ``store_acc_mxfp8`` (the BF16 tile and its e4m3 quantization, used by the backward).

Invariants:

- an e4m3 A/B stage is the same 16 KiB as a BF16 one, so both precisions share one A/B ring and one
  transaction count; the scales have their own ring;
- per K tile the mainloop copies the stage's scales to that stage's TMEM slots and releases the scale stage
  before it issues the stage's MMAs (the commit covers the copies' shared-memory reads);
- every epilogue that uses the D stages ends with a 128-thread barrier after its TMA stores have read them:
  the next epilogue writes the stages without a wait of its own.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Float32, Int32, Int64, Uint16, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import LayoutEnum

from ._compat import copy_utils
from ._compat.gemm_sm100 import GemmDefaultSm100, VarlenArguments
from ._compat.layout_utils import tile_atom_to_shape_SF_strided
from ._mxfp8_bwd_math import glu_forward

from ._mxfp8_quant import (
    PAIRS_PER_BLOCK,
    PAYLOAD_WORDS,
    QUANT_TILE,
    cvt_bf16x2,
    quantize_pairs,
    store_scale_word,
    store_shared_v4,
)
from ._mxfp8_gemm_base import (
    A_BYTES,
    AB_STAGES,
    ACC_STAGES,
    B_BYTES,
    CLUSTER_SHAPE,
    CTA_B_ROWS,
    CTA_M,
    D_BYTES,
    D_STAGES,
    EPILOGUE_N,
    IDLE_WARPS,
    MMA_WARP,
    THREADS,
    TILE_M,
    TILE_N,
    TMA_CACHE_HINT_EVICT_FIRST,
    TMA_WARP,
    _is_store_warp,
)
from ._bwd_device_helpers import named_barrier_sync_unaligned
from ._tma_1d import tma_store_1d_raw

# Routed MXFP8.  An e4m3 K tile of 128 is the same 16 KiB as a BF16 K tile of 64, so both
# precisions share one A/B ring.  The scale ring is separate: per stage one 512 B SFA tile
# (this CTA's 128 rows) and two SFB tiles (the cluster's 256 rows), then the D tiles 1 KiB aligned.
FP8_K_TILE = 128
SF_VEC_SIZE = 32
SF_TILE_BYTES = 512  # one scale tile: 128 MN rows x 128 K
SFA_BYTES = AB_STAGES * SF_TILE_BYTES
SFB_BYTES = AB_STAGES * CLUSTER_SHAPE[0] * SF_TILE_BYTES
SF_LOAD_BYTES = CLUSTER_SHAPE[0] * 3 * SF_TILE_BYTES  # a scale stage's transaction bytes
MXFP8_D_OFFSET = A_BYTES + B_BYTES + SFA_BYTES + SFB_BYTES
MXFP8_ARENA_BYTES = MXFP8_D_OFFSET + D_BYTES
# The fused Gate/Up + SwiGLU epilogue (store_acc_fused_swiglu) over the three 8 KiB D stages, each
# 128 x 32 BF16 tile under the 64 B swizzle (context_smem_layout):
# stage 0 holds the pair's BF16 Up context tile, then the BF16 Hidden fragment staged for the
# transposed quantization; stage 1 the Hidden e4m3 tiles (normal 128 x 32 B, transposed 32 x 128 B
# under the 128 B swizzle); stage 2 the pair's BF16 Gate context tile and, after the last pair, the
# two 512 B Hidden scale tiles.  The BF16 Gate / Up context is kept unquantized and leaves through
# TMA stores.
D_STAGE_BYTES = CTA_M * EPILOGUE_N * 2
FUSED_HIDDEN_BF16_STAGE = 0
FUSED_UP_STAGE = 0
FUSED_HIDDEN_FP8_STAGE = 1
FUSED_GATE_STAGE = 2
FUSED_SCALE_STAGE = 2
FUSED_HIDDEN_T_OFFSET = CTA_M * EPILOGUE_N  # after the normal e4m3 tile, 4 KiB in (1 KiB aligned)
FUSED_HIDDEN_SC_OFFSET = 0
FUSED_HIDDEN_T_SC_OFFSET = FUSED_HIDDEN_SC_OFFSET + SF_TILE_BYTES
assert FUSED_HIDDEN_T_SC_OFFSET + SF_TILE_BYTES <= D_STAGE_BYTES

# The e4m3-context variant of the fused epilogue (``gate_up_context="mxfp8"``: the
# ``*_e4m3_context`` functions below): stage 2 holds the Gate / Up e4m3 context tile (4 KiB, shared by
# the two) and the four 512 B scale tiles; stages 0 / 1 as above.
E4M3_CONTEXT_STAGE = 2
E4M3_CONTEXT_GATE_SC_OFFSET = CTA_M * EPILOGUE_N
E4M3_CONTEXT_UP_SC_OFFSET = E4M3_CONTEXT_GATE_SC_OFFSET + SF_TILE_BYTES
E4M3_CONTEXT_HIDDEN_SC_OFFSET = E4M3_CONTEXT_UP_SC_OFFSET + SF_TILE_BYTES
E4M3_CONTEXT_HIDDEN_T_SC_OFFSET = E4M3_CONTEXT_HIDDEN_SC_OFFSET + SF_TILE_BYTES
assert E4M3_CONTEXT_HIDDEN_T_SC_OFFSET + SF_TILE_BYTES <= D_STAGE_BYTES
HIDDEN_T_TILE_ROWS = EPILOGUE_N  # one pair's 32 Hidden columns = 32 rows of the transposed tile
assert FUSED_HIDDEN_T_OFFSET + HIDDEN_T_TILE_ROWS * QUANT_TILE <= D_STAGE_BYTES
FUSED_SUBTILES = QUANT_TILE // EPILOGUE_N  # four 32-column pairs make one 128 x 128 Hidden tile
assert MXFP8_D_OFFSET % 1024 == 0 and MXFP8_ARENA_BYTES == 230400
# TMEM columns: the accumulator at column 0, the A scales at 256, the B scales at 384, one
# slot per load stage (4 / 8 columns).
MXFP8_TMEM_COLUMNS = 512
SFA_TMEM_COLUMN = 256
SFB_TMEM_COLUMN = 384
SFA_TMEM_STAGE_COLUMNS = 4
SFB_TMEM_STAGE_COLUMNS = 8

# Named barriers of the forward kernel (ids 0 and 3 are the CuTe pipelines' / cluster's):
# the GEMM epilogue's 128 threads, the TMEM allocator's 256 and the D-store tail's 128.
EPILOGUE_BARRIER_ID = 1
TMEM_ALLOC_BARRIER_ID = 2
D_STORE_TAIL_BARRIER_ID = 4

# The routed mainloop issues tcgen05 directly (one elected lane, all of a K tile's
# instructions back to back).  The descriptors are the ones the CuTe DSL atoms build
# for these layouts (``configure`` checks the staged layouts they encode):
# - UMMA instruction descriptor, kind::mxf8f6f4 block-scaled, M 256 (m_dim 16 at bit 24),
#   N 256 (n_dim 32 at bit 17), e8m0 scales (bit 23), e4m3 A / B, both K-major, no negate;
#   the A / B scale-factor ids go to bits 29-30 / 4-5.
UMMA_IDESC = 0x10C00000
# - shared-memory descriptors: start address >> 4 in bits 0-13, LBO 16 B (bit 16), SBO
#   and the 128 B swizzle in the high word: A / B tiles (128 rows x 128 B, SBO 1 KiB) and
#   scale tiles (SBO 128 B).
AB_DESC_HIGH = 0x40004040
SF_DESC_HIGH = 0x00004008
# - bit 16 of the instruction descriptor marks an MN-major B, the sibling of the kind::f16 loop's
#   bit 15 for an MN-major A (``_swapab_fwd.IDESC_A_MN_MAJOR``).  An MN-major B tile is the same
#   128 row x 128 B stage under the same swizzle -- 128 e4m3 of MN fill one 128 B swizzle atom, so
#   there is a single slab and the leading-dimension offset stays unused -- but its K runs down the
#   rows, so one MMA's K of 32 steps 32 rows (4 KiB) instead of 32 B.
IDESC_B_MN_MAJOR = 1 << 16
B_K_STEP_UNITS = 2  # K-major: 32 e4m3 along a row
B_MN_K_STEP_UNITS = (32 * 128) >> 4  # MN-major: 32 rows of 128 B
AB_STAGE_BYTES = A_BYTES // AB_STAGES  # one e4m3 128 x 128 tile per CTA, A and B alike
assert AB_STAGE_BYTES == CTA_M * FP8_K_TILE == B_BYTES // AB_STAGES
SFB_STAGE_BYTES = CLUSTER_SHAPE[0] * SF_TILE_BYTES


class _Mxfp8Gemm(GemmDefaultSm100):
    """Block-scaled sibling of ``_Bf16Gemm``: layouts and atoms of the routed MXFP8 tasks.

    Same stage counts (six A/B, one accumulator, three D) and 2-CTA 256 x 256 tile; e4m3
    operands with a 128 K tile and e8m0 scales every 32 elements.  The mainloop is
    ``mma_blockscaled`` below, not QuACK's ``mma``: the scales are on their own ring.
    """

    @classmethod
    def _compute_stages(cls, *args, **kwargs):
        del args, kwargs
        return ACC_STAGES, AB_STAGES, D_STAGES, 0

    def __init__(self) -> None:
        super().__init__(
            acc_dtype=Float32,
            a_dtype=Float8E4M3FN,
            mma_tiler_mnk=(TILE_M, TILE_N, FP8_K_TILE),
            cluster_shape_mnk=CLUSTER_SHAPE,
            sf_vec_size=SF_VEC_SIZE,
            use_clc_persistence=False,
            use_pdl=False,
        )
        self.mma_warp_id = MMA_WARP
        self.scheduler_warp_id = IDLE_WARPS[0]
        self.epi_load_warp_id = IDLE_WARPS[1]
        self.ab_load_warp_id = TMA_WARP
        assert self.blockscaled
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
        self.sf_dtype = Float8E8M0FNU  # MXFP8: e8m0 by definition
        self.a_layout = LayoutEnum.from_tensor(mA_mk)
        self.b_layout = LayoutEnum.from_tensor(mB_nk)
        self.d_layout = LayoutEnum.from_tensor(mD_mn)
        self.c_layout = None
        self.a_major_mode = self.a_layout.mma_major_mode()
        self.b_major_mode = self.b_layout.mma_major_mode()
        self.varlen_m = False
        self.varlen_k = False
        self._setup_attributes(self.EpilogueArguments(), VarlenArguments())
        atom_thr_size = cute.size(self.tiled_mma.thr_id.shape)
        # The A/B stage's transaction bytes only; the scales arm their own ring (SF_LOAD_BYTES).
        self.num_tma_load_bytes = (
            cute.size_in_bytes(Float8E4M3FN, cute.slice_(self.a_smem_layout_staged, (None, None, None, 0)))
            + cute.size_in_bytes(Float8E4M3FN, cute.slice_(self.b_smem_layout_staged, (None, None, None, 0)))
        ) * atom_thr_size
        self.num_sf_load_bytes = (
            cute.size_in_bytes(Float8E8M0FNU, cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)))
            + cute.size_in_bytes(Float8E8M0FNU, cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)))
        ) * atom_thr_size

        assert self.a_dtype is Float8E4M3FN
        assert self.b_dtype is Float8E4M3FN
        assert self.d_dtype is BFloat16
        assert self.use_2cta_instrs
        assert self.cta_tile_shape_mnk == (CTA_M, TILE_N, FP8_K_TILE)
        assert cute.cosize(self.a_smem_layout_staged.outer) == AB_STAGES * CTA_M * FP8_K_TILE
        assert cute.cosize(self.b_smem_layout_staged.outer) == AB_STAGES * CTA_B_ROWS * FP8_K_TILE
        assert cute.cosize(self.sfa_smem_layout_staged) == SFA_BYTES
        assert cute.cosize(self.sfb_smem_layout_staged) == SFB_BYTES
        assert self.num_sf_load_bytes == SF_LOAD_BYTES
        # The hand-built descriptors below (UMMA_IDESC, AB_DESC_HIGH, SF_DESC_HIGH) encode these
        # staged layouts: K-major e4m3 A / B stages of CTA_M x FP8_K_TILE under the 128 B swizzle
        # (row pitch 128 B, SBO 1 KiB), one stage every AB_STAGE_BYTES; unswizzled 512 B scale
        # tiles (SBO 128 B); e8m0 scales every SF_VEC_SIZE = 32 elements (``scale_vec::1X``).
        a_stage = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_stage = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        assert "3,4,3" in str(a_stage.inner) and "3,4,3" in str(b_stage.inner), (a_stage.inner, b_stage.inner)
        assert cute.size_in_bytes(Float8E4M3FN, a_stage) == AB_STAGE_BYTES
        assert cute.size_in_bytes(Float8E4M3FN, b_stage) == AB_STAGE_BYTES
        assert cute.size_in_bytes(Float8E8M0FNU, cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))) == SF_TILE_BYTES
        assert cute.size_in_bytes(Float8E8M0FNU, cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))) == SFB_STAGE_BYTES
        assert self.a_major_mode == tcgen05.OperandMajorMode.K
        # B may be MN-major (a routed weight in TransformerEngine's native columnwise layout).  Its
        # stage is the same 128 x 128 B tile under the same swizzle, which is why one A/B ring and
        # one transaction count serve both; ``mma_blockscaled`` flips the descriptor bit and the K
        # step (IDESC_B_MN_MAJOR / B_MN_K_STEP_UNITS).
        assert self.b_major_mode in (tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN)
        assert self.sf_vec_size == SF_VEC_SIZE


@cute.jit
def scale_operand(scales: cute.Tensor, operand_shape: cute.Shape, k_offset=None) -> cute.Tensor:
    """The e8m0 view the block-scaled TMA loads, over the uint8 scale tiles.

    The scales are ``(L * MN / 128, K / 128, 32, 16)`` contiguous bytes (``mxfp8_quantize``);
    each 512 B tile is the tcgen05 scale atom, so only the outer strides are rebuilt for
    ``operand_shape`` (``(MN, K)`` or ``(MN, K, L)``).

    The per-L stride comes from the scale tensor's own extent, so ``operand_shape`` must cover the
    whole operand.  A combined FC1 passes its full ``2I`` rows and reads Up through a coordinate
    offset of ``I`` rows (or K columns, transposed), which, unlike an offset baked into the
    iterator here, survives a split expert's descriptor being retargeted.

    ``k_offset`` (a runtime multiple of 128, grouped ``(MN, K, L)`` operands only): the view of the
    operand's K columns from ``k_offset`` on -- the same strides, the iterator ``k_offset / 128`` K
    blocks in, the K extent shortened by it.  This is how the backward gives the transposed Up half
    of a combined FC1 its own scale view, which its K loop then tiles from 0.
    """
    k_blocks = Int32(cute.size(scales.shape[1]))
    row_blocks = Int32(cute.size(scales.shape[0]))
    if cutlass.const_expr(cute.rank(operand_shape) == 3):
        row_blocks = row_blocks // Int32(cute.size(operand_shape[2]))
    tile = Int32(SF_TILE_BYTES)
    strides = (row_blocks * k_blocks * tile, k_blocks * tile, tile, Int32(1))
    base = cute.recast_ptr(scales.iterator, dtype=Float8E8M0FNU)
    if cutlass.const_expr(k_offset is not None):
        base = base + Int64(k_offset // Int32(FP8_K_TILE)) * Int64(SF_TILE_BYTES)
        operand_shape = (operand_shape[0], Int32(operand_shape[1]) - k_offset, operand_shape[2])
    return cute.make_tensor(base, tile_atom_to_shape_SF_strided(operand_shape, SF_VEC_SIZE, strides))


@cute.jit
def make_sf_load_fns(
    collective,
    tiled_mma: cute.TiledMma,
    tiled_mma_sfb: cute.TiledMma,
    cluster_layout_vmnk: cute.Layout,
    cluster_layout_sfb_vmnk: cute.Layout,
    sfa_atom: cute.CopyAtom,
    sfa: cute.Tensor,
    sfb_atom: cute.CopyAtom,
    sfb: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    row: Int32,
    column: Int32,
    cta_rank: Int32,
    sfb_desc_ptr=None,
):
    """The two scale loads of one routed tile.

    SFA: this CTA's 128-row tile to itself (mask ``1 << cta_rank``).  SFB: CTA r loads
    the cluster tile's 128-row half r and multicasts it (mask ``0b11``) into half r of the
    stage's SFB tile pair on both CTAs.  All bytes land on CTA 0's barrier.

    ``sfb_desc_ptr`` overrides the descriptor of the B scale load alone, which is how a routed
    weight held in split per-expert allocations is read: its scale is that expert's own
    allocation, named by ``routed_descriptors`` rather than sliced out of an L mode.  The A scale
    is the activations', never split.
    """
    thr_mma = tiled_mma.get_slice(cta_rank)
    thr_mma_sfb = tiled_mma_sfb.get_slice(cta_rank)
    block_coord = cluster_layout_vmnk.get_flat_coord(cta_rank)
    block_coord_sfb = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank)
    sfa_tile = cute.local_tile(sfa, cute.select(collective.mma_tiler, [0, 2]), (row, None))
    copy_sfa, _, _ = copy_utils.tma_get_copy_fn(
        sfa_atom,
        cta_coord=block_coord[2],
        cta_layout=cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape),
        src_tensor=thr_mma.partition_A(sfa_tile),
        dst_tensor=sSFA,
        filter_zeros=True,
        mcast_mask=cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_coord, mcast_mode=2),
    )
    sfb_tile = cute.local_tile(sfb, cute.select(collective.mma_tiler_sfb, [1, 2]), (column, None))
    copy_sfb, _, _ = copy_utils.tma_get_copy_fn(
        sfb_atom,
        cta_coord=block_coord_sfb[1],
        cta_layout=cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape),
        src_tensor=thr_mma_sfb.partition_B(sfb_tile),
        dst_tensor=sSFB,
        filter_zeros=True,
        mcast_mask=cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_coord_sfb, mcast_mode=1),
        # Only when given: a caller that names the descriptor at every load (the backward, through its global
        # tensormap slots) must not find the keyword bound here as well.
        **({} if sfb_desc_ptr is None else {"tma_desc_ptr": sfb_desc_ptr}),
    )
    return copy_sfa, copy_sfb


@dsl_user_op
def _issue_scale_copies(sfa_tmem: Int32, sfb_tmem: Int32, sfa_desc: Int64, sfb_desc: Int64, empty_barrier: Int32, mask: Int32, *, loc=None, ip=None) -> None:
    """Three scale copies from shared memory to TMEM (``tcgen05.cp``) and the commit to the scale
    stage's empty barrier, issued by one elected lane: SFA, then the two SFB halves (the second
    512 B / 4 columns further)."""
    llvm.inline_asm(
        None,
        [
            Int32(sfa_tmem).ir_value(loc=loc, ip=ip),
            Int32(sfb_tmem).ir_value(loc=loc, ip=ip),
            Int64(sfa_desc).ir_value(loc=loc, ip=ip),
            Int64(sfb_desc).ir_value(loc=loc, ip=ip),
            Int32(empty_barrier).ir_value(loc=loc, ip=ip),
            Int32(mask).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .pred leader;\n.reg .b32 lane;\n.reg .b64 second;\n.reg .b16 mask;\n"
        "elect.sync lane|leader, 0xffffffff;\n"
        "add.s64 second, $3, 32;\n"
        "cvt.u16.u32 mask, $5;\n"
        "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [$0], $2;\n"
        "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [$1], $3;\n"
        "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [$1+4], second;\n"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$4], mask;\n"
        "}",
        "r,r,l,l,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _issue_mma_k_tile(
    d_tmem: Int32,
    a_desc: Int64,
    b_desc: Int64,
    sfa_tmem: Int32,
    sfb_tmem: Int32,
    accumulate: Int32,
    empty_barrier: Int32,
    mask: Int32,
    *,
    idesc: int = UMMA_IDESC,
    b_k_step: int = B_K_STEP_UNITS,
    loc=None,
    ip=None,
) -> None:
    """One K tile's four block-scaled MMAs (K 32 each) and the commit to the A/B stage's empty
    barrier, issued by one elected lane.

    MMA ``k`` reads the A descriptor ``+ 2k`` (32 B along K), the B descriptor ``+ b_k_step * k``
    (2 for a K-major B, ``B_MN_K_STEP_UNITS`` for an MN-major one) and scale-factor id ``k`` (TMEM
    address bits 30-31, mirrored in the instruction descriptor); ``accumulate`` is the first
    MMA's enable-input-d, the others accumulate.  ``idesc`` is the instruction descriptor's static
    part: ``UMMA_IDESC``, or that with ``IDESC_B_MN_MAJOR`` for an MN-major B.
    """
    llvm.inline_asm(
        None,
        [
            Int32(d_tmem).ir_value(loc=loc, ip=ip),
            Int64(a_desc).ir_value(loc=loc, ip=ip),
            Int64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(sfa_tmem).ir_value(loc=loc, ip=ip),
            Int32(sfb_tmem).ir_value(loc=loc, ip=ip),
            Int32(accumulate).ir_value(loc=loc, ip=ip),
            Int32(empty_barrier).ir_value(loc=loc, ip=ip),
            Int32(mask).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .pred leader, first, on;\n.reg .b32 lane, i0, i1, i2, i3, sa1, sa2, sa3, sb1, sb2, sb3;\n"
        ".reg .b64 a1, a2, a3, b1, b2, b3;\n.reg .b16 mask;\n"
        "elect.sync lane|leader, 0xffffffff;\n"
        # ``first`` (accumulate != 0) is the enable-input-D of the first MMA of the K tile; the three
        # that follow always accumulate, so ``on`` is the always-true predicate.
        "setp.ne.b32 first, $5, 0;\n"
        "setp.eq.b32 on, $5, $5;\n"
        "mov.b32 i0, " + str(idesc) + ";\n"
        "or.b32 i1, i0, 0x20000010;\nor.b32 i2, i0, 0x40000020;\nor.b32 i3, i0, 0x60000030;\n"
        "add.s64 a1, $1, 2;\nadd.s64 a2, $1, 4;\nadd.s64 a3, $1, 6;\n"
        "add.s64 b1, $2, " + str(b_k_step) + ";\nadd.s64 b2, $2, " + str(2 * b_k_step) + ";\nadd.s64 b3, $2, " + str(3 * b_k_step) + ";\n"
        "add.u32 sa1, $3, 0x40000000;\nadd.u32 sa2, $3, 0x80000000;\nadd.u32 sa3, $3, 0xC0000000;\n"
        "add.u32 sb1, $4, 0x40000000;\nadd.u32 sb2, $4, 0x80000000;\nadd.u32 sb3, $4, 0xC0000000;\n"
        "cvt.u16.u32 mask, $7;\n"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[$0], $1, $2, i0, [$3], [$4], first;\n"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[$0], a1, b1, i1, [sa1], [sb1], on;\n"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[$0], a2, b2, i2, [sa2], [sb2], on;\n"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[$0], a3, b3, i3, [sa3], [sb3], on;\n"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$6], mask;\n"
        "}",
        "r,l,l,r,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _smem_desc(address: Int32, high: cutlass.Constexpr[int]) -> Int64:
    """A shared-memory descriptor: ``address >> 4`` (14 bits), LBO 16 B, ``high`` the upper word."""
    low = ((address >> Int32(4)) & Int32(0x3FFF)) | Int32(0x00010000)
    return (Int64(high) << Int64(32)) | Int64(Uint32(low))


@cute.jit
def mma_blockscaled(
    ab_pipeline,
    sf_pipeline,
    acc_pipeline,
    ab_state,
    sf_state,
    acc_state,
    arena: cute.Pointer,
    acc_tmem: cute.Pointer,
    k_tiles: Int32,
    b_mn_major: cutlass.Constexpr[bool] = False,
):
    """The mainloop of one routed tile on the leader CTA.

    Per K tile: wait the scales, copy them to this stage's TMEM slots and release the
    scale stage (the commit covers the copies' shared-memory reads), wait A/B, issue
    the 128 / 32 block-scaled MMAs, release A/B.  The accumulator is committed once.
    ``arena`` holds the A and B rings, then the SFA and SFB rings (1 KiB aligned);
    ``acc_tmem`` is the accumulator, with the A / B scale slots at columns 256 / 384, one per stage.

    ``b_mn_major``: the B stage holds an MN-major tile (a routed weight in TransformerEngine's
    native columnwise layout, whose MN rather than K is contiguous).  The stage is the same
    128 x 128 B tile, so only the instruction descriptor's B-major bit and the descriptor's K step
    change; A and the scales are untouched.
    """
    idesc = UMMA_IDESC | IDESC_B_MN_MAJOR if b_mn_major else UMMA_IDESC
    b_k_step = B_MN_K_STEP_UNITS if b_mn_major else B_K_STEP_UNITS
    a_base = Int32(arena.toint())
    b_base = a_base + Int32(A_BYTES)
    sfa_base = b_base + Int32(B_BYTES)
    sfb_base = sfa_base + Int32(SFA_BYTES)
    d_tmem = Int32(acc_tmem.toint())
    sf_mask = Int32(sf_pipeline.consumer_mask)
    ab_mask = Int32(ab_pipeline.consumer_mask)
    acc_pipeline.producer_acquire(acc_state)
    for k_tile in cutlass.range(k_tiles, unroll=1):
        sf_pipeline.consumer_wait(sf_state)
        slot = sf_state.index
        sfa_tmem = d_tmem + Int32(SFA_TMEM_COLUMN) + slot * Int32(SFA_TMEM_STAGE_COLUMNS)
        sfb_tmem = d_tmem + Int32(SFB_TMEM_COLUMN) + slot * Int32(SFB_TMEM_STAGE_COLUMNS)
        _issue_scale_copies(
            sfa_tmem,
            sfb_tmem,
            _smem_desc(sfa_base + slot * Int32(SF_TILE_BYTES), SF_DESC_HIGH),
            _smem_desc(sfb_base + slot * Int32(SFB_STAGE_BYTES), SF_DESC_HIGH),
            Int32(sf_pipeline.sync_object_empty.get_barrier(slot).toint()),
            sf_mask,
        )
        ab_pipeline.consumer_wait(ab_state)
        stage = ab_state.index
        _issue_mma_k_tile(
            d_tmem,
            _smem_desc(a_base + stage * Int32(AB_STAGE_BYTES), AB_DESC_HIGH),
            _smem_desc(b_base + stage * Int32(AB_STAGE_BYTES), AB_DESC_HIGH),
            sfa_tmem,
            sfb_tmem,
            Int32(k_tile != Int32(0)),
            Int32(ab_pipeline.sync_object_empty.get_barrier(stage).toint()),
            ab_mask,
            idesc=idesc,
            b_k_step=b_k_step,
        )
        ab_state.advance()
        sf_state.advance()
    acc_pipeline.producer_commit(acc_state)
    acc_state.advance()
    return ab_state, sf_state, acc_state


@cute.jit
def store_tile(atom: cute.CopyAtom, tensor: cute.Tensor, tile_shape: cute.Tile, coord, smem_tile: cute.Tensor, cache_policy=None):
    """One shared-memory tile to ``tensor``'s tile ``coord`` (one thread, TMA)."""
    global_tile = cute.local_tile(tensor, tile_shape, coord)
    smem_part, global_part = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(smem_tile, 0, cute.rank(smem_tile)), cute.group_modes(global_tile, 0, cute.rank(global_tile))
    )
    if cutlass.const_expr(cache_policy is None):
        cute.copy(atom, smem_part, global_part)
    else:
        cute.copy(atom, smem_part, global_part, cache_policy=cache_policy)


@cute.jit
def store_acc_mxfp8(
    collective,
    tiled_copy_t2r,
    tiled_copy_r2s,
    tTR_tAcc,
    tTR_rAcc,
    tRS_rD,
    tRS_sD,
    copy_D,
    fp8_atom: cute.CopyAtom,
    fp8_tensor: cute.Tensor,
    d_scales: cute.Pointer,
    scale_columns: Int32,
    arena_fp8: cute.Pointer,
    tile_row: Int32,
    tile_column: Int32,
    acc_pipeline,
    acc_consumer_state,
    thread: Int32,
    has_tail: cutlass.Constexpr[bool] = True,
):
    """The routed Gate / Up epilogue that stores the BF16 tile and its e4m3 quantization.

    Each of the eight 32-column subtiles: TMEM -> FP32 -> BF16; the BF16 tile goes out
    through D stage ``i % 2`` (the SwiGLU reads it); once the previous subtile's e4m3 and scale
    stores have read their tiles, the same BF16 values are quantized, one 32-element block per
    row, into the e4m3 tile and the row's scale byte; every fourth subtile writes the 128-row
    scale tile (512 B, bulk copy).  The accumulator is released only after the last subtile.
    ``arena_fp8`` is D stage 2: the e4m3 tile (128 x 32 B) and two scale tiles.  ``tile_row`` is
    the 128-row tile, ``tile_column`` the 256-column tile of this CTA's output.

    ``has_tail`` (compile time): the output may end in a 128-column tail tile (I % 256 != 0), whose
    second scale tile is skipped.  A caller whose I is always a multiple of 256 passes False and
    gets the unguarded store.
    """
    fp8_tile = cute.make_tensor(cute.recast_ptr(arena_fp8, dtype=Float8E4M3FN), cute.make_layout((CTA_M, EPILOGUE_N), stride=(EPILOGUE_N, 1)))
    scale_tiles = arena_fp8 + CTA_M * EPILOGUE_N
    # A row's 32 e4m3 bytes go out as two 16-byte stores.  Rows 4-7 of every eight store
    # their halves in the other order, so a quarter warp's stores cover all 32 banks.
    fp8_row = arena_fp8 + thread * Int32(EPILOGUE_N)
    swap_half = (thread >> Int32(2)) & Int32(1)
    swap_mask = Int32(0) - swap_half
    pairs = cute.make_rmem_tensor((SF_VEC_SIZE // 2,), Uint32)
    payload = cute.make_rmem_tensor((SF_VEC_SIZE // 4,), Int32)
    chunk = cute.make_rmem_tensor((4,), Int32)
    r_bf16 = cute.make_rmem_tensor(tRS_rD.shape, BFloat16)
    acc_r2s = tiled_copy_r2s.retile(tTR_rAcc)
    # The dynamic loop below flattens every local whose method it calls (only ``self`` is
    # exempt), so it calls the retiled view, not the collective, and reads the barrier taken here.
    epilogue_barrier = collective.epilogue_barrier
    assert cute.size(tTR_rAcc) == SF_VEC_SIZE  # one 32x32b x32 load: a row's 32 columns
    acc_pipeline.consumer_wait(acc_consumer_state)
    scale_word = Int32(0)
    # unroll=1: one copy of the loop body for the eight subtiles.
    for subtile in cutlass.range(TILE_N // EPILOGUE_N, unroll=1):
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile)], tTR_rAcc)
        # Wait for this subtile's TMEM load before converting it (PTX orders the register use
        # itself; the wait also keeps the loads complete before the accumulator is released after
        # the loop).
        cute.arch.fence_view_async_tmem_load()
        bf16 = acc_r2s.load().to(BFloat16)
        named_barrier_sync_unaligned(epilogue_barrier)
        r_bf16.store(bf16)
        cute.copy(tiled_copy_r2s, r_bf16, tRS_sD[(None, None, None, subtile % Int32(2))])
        cute.arch.fence_view_async_shared()
        named_barrier_sync_unaligned(epilogue_barrier)
        if thread == Int32(0):
            copy_D(src_idx=subtile % Int32(2), dst_idx=(0, subtile), cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
            cute.arch.cp_async_bulk_commit_group()
            # The previous subtile's e4m3 and scale stores must have read their tiles.
            cute.arch.cp_async_bulk_wait_group(1, read=True)
        named_barrier_sync_unaligned(epilogue_barrier)
        for pair in cutlass.range_constexpr(SF_VEC_SIZE // 2):
            pairs[pair] = cvt_bf16x2(tTR_rAcc[2 * pair], tTR_rAcc[2 * pair + 1])
        scale_byte = quantize_pairs(pairs, payload)
        scale_word = scale_word | (scale_byte << ((subtile % Int32(4)) * Int32(8)))
        for half in cutlass.range_constexpr(2):
            # The words of half ``half ^ swap_half``, selected without branching.
            for word in cutlass.range_constexpr(4):
                own = payload[4 * half + word]
                chunk[word] = own ^ ((own ^ payload[4 * (1 - half) + word]) & swap_mask)
            store_shared_v4(fp8_row + (Int32(half) ^ swap_half) * Int32(16), chunk[0], chunk[1], chunk[2], chunk[3])
        if subtile % Int32(4) == Int32(3):
            store_scale_word(scale_tiles + (subtile // Int32(4)) * Int32(SF_TILE_BYTES), thread, scale_word)
            scale_word = Int32(0)
        cute.arch.fence_view_async_shared()
        named_barrier_sync_unaligned(epilogue_barrier)
        if thread == Int32(0):
            store_tile(
                fp8_atom,
                fp8_tensor,
                (CTA_M, EPILOGUE_N),
                (tile_row, tile_column * Int32(TILE_N // EPILOGUE_N) + subtile),
                fp8_tile,
                cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST),
            )
            cute.arch.cp_async_bulk_commit_group()
            if subtile % Int32(4) == Int32(3):
                column_block = tile_column * Int32(TILE_N // 128) + subtile // Int32(4)
                if cutlass.const_expr(has_tail):
                    # A tail tile (I % 256 != 0) has one 128-column half: its second half's scale tile
                    # would be the next row's first (a raw 1D store, not a clipped TMA tensor store).
                    if column_block < scale_columns:
                        tma_store_1d_raw(
                            d_scales.toint() + (Int64(tile_row) * Int64(scale_columns) + Int64(column_block)) * Int64(SF_TILE_BYTES),
                            scale_tiles + (subtile // Int32(4)) * Int32(SF_TILE_BYTES),
                            Int32(SF_TILE_BYTES),
                        )
                        cute.arch.cp_async_bulk_commit_group()
                else:
                    tma_store_1d_raw(
                        d_scales.toint() + (Int64(tile_row) * Int64(scale_columns) + Int64(column_block)) * Int64(SF_TILE_BYTES),
                        scale_tiles + (subtile // Int32(4)) * Int32(SF_TILE_BYTES),
                        Int32(SF_TILE_BYTES),
                    )
                    cute.arch.cp_async_bulk_commit_group()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()
    if _is_store_warp(thread):  # warp-uniform, in front of the barrier (see _is_store_warp); lane 0 holds the stores
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    named_barrier_sync_unaligned(pipeline.NamedBarrier(barrier_id=D_STORE_TAIL_BARRIER_ID, num_threads=128))
    return acc_consumer_state


def hidden_t_smem_layout() -> cute.ComposedLayout:
    """The transposed Hidden e4m3 staging tile: 32 rows (Hidden columns) x 128 B (tokens) under the
    128 B swizzle (Swizzle<3,4,3>: 16-byte chunk ``c`` of row ``r`` sits at chunk ``c ^ (r % 8)``), so
    a warp's 32 row writes spread over all banks and the TMA store un-swizzles it."""
    return cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, cute.make_layout((HIDDEN_T_TILE_ROWS, QUANT_TILE), stride=(QUANT_TILE, 1)))


@dsl_user_op
def store_shared_u8(address: cute.Pointer, value: Int32, *, loc=None, ip=None) -> None:
    """One byte to shared memory (``st.shared.u8``): the low byte of ``value``."""
    llvm.inline_asm(
        None,
        [address.toint(loc=loc, ip=ip).ir_value(), Int32(value).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 b;\ncvt.u16.u32 b, $1;\nst.shared.u8 [$0], b;\n}",
        "r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


def context_smem_layout() -> cute.ComposedLayout:
    """A 128 x 32 BF16 tile (the Gate / Up context, the Hidden staging) under the 64 B swizzle, the
    epilogue layout CuTe builds for it (16-byte chunk ``c`` of row ``r`` sits at chunk
    ``c ^ ((r / 2) % 4)``), so eight consecutive rows' 16-byte stores cover all 32 banks and one TMA
    store un-swizzles the tile."""
    layout = cute.select(cutlass.utils.make_smem_layout_epi(BFloat16, LayoutEnum.ROW_MAJOR, (CTA_M, EPILOGUE_N), 1), mode=[0, 1])
    # Trace time only: _store_bf16_row and the transposed pass's reads hard-code this swizzle (chunk ^ ((r / 2) % 4)).
    assert str(layout.inner) == str(cute.make_swizzle(2, 4, 3)), f"context tile swizzle {layout.inner}, not SW64"
    return layout


@cute.jit
def _store_bf16_row(stage: cute.Pointer, thread: Int32, pairs: cute.Tensor) -> None:
    """This thread's row of a ``context_smem_layout`` tile: its 32 BF16 values (``pairs``, 16 bf16x2
    words) as four swizzled 16-byte chunks."""
    row = stage + thread * Int32(2 * EPILOGUE_N)
    swizzle = (thread >> Int32(1)) & Int32(3)
    for chunk in cutlass.range_constexpr(4):
        store_shared_v4(
            row + ((Int32(chunk) ^ swizzle) << Int32(4)),
            pairs[4 * chunk].bitcast(Int32),
            pairs[4 * chunk + 1].bitcast(Int32),
            pairs[4 * chunk + 2].bitcast(Int32),
            pairs[4 * chunk + 3].bitcast(Int32),
        )


@cute.jit
def _store_row_block(row_base: cute.Pointer, swap_half: Int32, payload: cute.Tensor) -> None:
    """One thread's 32 e4m3 bytes (``payload``, eight words) as two 16-byte stores to a plain
    128 x 32 B tile row; rows 4-7 of every eight store their halves in the other order so a
    quarter warp covers all 32 banks."""
    swap_mask = Int32(0) - swap_half
    chunk = cute.make_rmem_tensor((4,), Int32)
    for half in cutlass.range_constexpr(2):
        for word in cutlass.range_constexpr(4):
            own = payload[4 * half + word]
            chunk[word] = own ^ ((own ^ payload[4 * (1 - half) + word]) & swap_mask)
        store_shared_v4(row_base + (Int32(half) ^ swap_half) * Int32(16), chunk[0], chunk[1], chunk[2], chunk[3])


@cute.jit
def _fused_pair(
    subtile: Int32,
    release_acc: cutlass.Constexpr[bool],
    tiled_copy_t2r,
    tTR_tAcc,
    tTR_rAcc,
    epilogue_barrier,
    acc_pipeline,
    acc_consumer_state,
    gate_atom,
    gate_bf16,
    up_atom,
    up_bf16,
    hidden_atom,
    hidden_fp8,
    hidden_t_atom,
    hidden_t,
    d_stages: cute.Pointer,
    tile_row: Int32,
    tile_column: Int32,
    is_clamped: cutlass.Constexpr[bool],
    swiglu_limit: Float32,
    thread: Int32,
    scale_words: cute.Tensor,
    gate_pairs: cute.Tensor,
    up_pairs: cute.Tensor,
    hidden_pairs: cute.Tensor,
    t_pairs: cute.Tensor,
    payload: cute.Tensor,
    situ: cutlass.Constexpr[bool] = False,
    situ_beta1=None,
    situ_beta2=None,
) -> None:
    """One 32-column pair of the fused epilogue: Gate subtile ``subtile`` and Up subtile
    ``subtile + 4`` to BF16, staged and TMA-stored as the BF16 context (the stores read shared memory
    while the SwiGLU runs), then Hidden = SwiGLU of the two BF16 values, quantized in the normal
    layout (this thread's row: one 32-element block) and the transposed layout (one Hidden column's
    32 rows: a warp's rows, through the BF16 staging tile).  ``scale_words[0]`` accumulates the
    normal Hidden scale bytes, ``scale_words[1]`` this thread's transposed ones (one per pair); the
    ``*_pairs`` and ``payload`` registers are the caller's (allocated once, outside its loop).
    ``release_acc``: the last pair releases the accumulator right after its Up load.
    """
    stage_bf16 = d_stages + FUSED_HIDDEN_BF16_STAGE * D_STAGE_BYTES
    stage_fp8 = d_stages + FUSED_HIDDEN_FP8_STAGE * D_STAGE_BYTES
    stage_gate = d_stages + FUSED_GATE_STAGE * D_STAGE_BYTES
    stage_up = d_stages + FUSED_UP_STAGE * D_STAGE_BYTES
    tile_layout = cute.make_layout((CTA_M, EPILOGUE_N), stride=(EPILOGUE_N, 1))
    ctx_layout = context_smem_layout()
    gate_tile = cute.make_tensor(cute.recast_ptr(stage_gate, ctx_layout.inner, dtype=BFloat16), ctx_layout.outer)
    up_tile = cute.make_tensor(cute.recast_ptr(stage_up, ctx_layout.inner, dtype=BFloat16), ctx_layout.outer)
    hidden_tile = cute.make_tensor(cute.recast_ptr(stage_fp8, dtype=Float8E4M3FN), tile_layout)
    ht_layout = hidden_t_smem_layout()
    hidden_t_tile = cute.make_tensor(cute.recast_ptr(stage_fp8 + FUSED_HIDDEN_T_OFFSET, ht_layout.inner, dtype=Float8E4M3FN), ht_layout.outer)
    row_offset = thread * Int32(EPILOGUE_N)
    swap_half = (thread >> Int32(2)) & Int32(1)
    shift = (subtile & Int32(3)) * Int32(8)
    column_tile = tile_column * Int32(FUSED_SUBTILES) + subtile
    # --- Gate: TMEM -> FP32 -> BF16 pairs ---
    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile)], tTR_rAcc)
    cute.arch.fence_view_async_tmem_load()
    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
        gate_pairs[pair] = cvt_bf16x2(tTR_rAcc[2 * pair], tTR_rAcc[2 * pair + 1])
    # --- Up: the same ---
    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile + Int32(FUSED_SUBTILES))], tTR_rAcc)
    cute.arch.fence_view_async_tmem_load()
    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
        up_pairs[pair] = cvt_bf16x2(tTR_rAcc[2 * pair], tTR_rAcc[2 * pair + 1])
    if cutlass.const_expr(release_acc):
        # Every TMEM read of this tile is complete (wait::ld above): the MMA may start the next tile.
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
    # --- Both context tiles through shared memory, one TMA store each.  The previous pair's Gate /
    # Up stores have read stages 2 / 0 (its wait below) and its transposed pass read stage 0 before
    # its last barrier.  Pair 0 relies on the previous user of the D stages ending with a 128-thread
    # barrier after its reads (store_acc_all's barrier, the D_STORE_TAIL barrier of this epilogue and of
    # store_acc_mxfp8): an epilogue that shares the D stages must keep that. ---
    _store_bf16_row(stage_gate, thread, gate_pairs)
    _store_bf16_row(stage_up, thread, up_pairs)
    cute.arch.fence_view_async_shared()
    named_barrier_sync_unaligned(epilogue_barrier)
    if thread == Int32(0):
        store_tile(gate_atom, gate_bf16, (CTA_M, EPILOGUE_N), (tile_row, column_tile), gate_tile, cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
        cute.arch.cp_async_bulk_commit_group()
        store_tile(up_atom, up_bf16, (CTA_M, EPILOGUE_N), (tile_row, column_tile), up_tile, cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
        cute.arch.cp_async_bulk_commit_group()
    # --- Hidden = SwiGLU(Gate BF16, Up BF16) -> BF16, computed on the BF16-rounded values ---
    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
        gate_low = (gate_pairs[pair] << Uint32(16)).bitcast(Float32)
        gate_high = (gate_pairs[pair] & Uint32(0xFFFF0000)).bitcast(Float32)
        up_low = (up_pairs[pair] << Uint32(16)).bitcast(Float32)
        up_high = (up_pairs[pair] & Uint32(0xFFFF0000)).bitcast(Float32)
        if cutlass.const_expr(is_clamped):
            # The clamp is a single min.f32 / max.f32 each.
            limit = Float32(swiglu_limit)
            gate_low = cute.arch.fmin(gate_low, limit)
            gate_high = cute.arch.fmin(gate_high, limit)
            up_low = cute.arch.fmin(cute.arch.fmax(up_low, -limit), limit)
            up_high = cute.arch.fmin(cute.arch.fmax(up_high, -limit), limit)
        hidden_pairs[pair] = cvt_bf16x2(
            glu_forward(gate_low, up_low, situ, situ_beta1, situ_beta2), glu_forward(gate_high, up_high, situ, situ_beta1, situ_beta2)
        )
    # Normal layout: this thread's row, these 32 columns are one block.  The previous pair's Hidden
    # stores must have read stage 1, and this pair's Up store stage 0, before they are rewritten.
    if thread == Int32(0):
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    named_barrier_sync_unaligned(epilogue_barrier)
    scale_words[0] = scale_words[0] | (quantize_pairs(hidden_pairs, payload) << shift)
    _store_row_block(stage_fp8 + row_offset, swap_half, payload)
    # Stage the BF16 row for the transposed pass (the 64 B swizzle of context_smem_layout).
    _store_bf16_row(stage_bf16, thread, hidden_pairs)
    named_barrier_sync_unaligned(epilogue_barrier)
    # Transposed layout: thread t quantizes Hidden column (t % 32) over its warp's 32 rows (one
    # block of hidden_t's row ``32 * subtile + t % 32``), the block division of ``quantize_tile``.  The
    # staging is read through a tensor view (ordinary loads the barrier orders).
    staging_halves = cute.make_tensor(cute.recast_ptr(stage_bf16, dtype=Uint16), cute.make_layout((CTA_M * EPILOGUE_N,)))
    column = thread & Int32(EPILOGUE_N - 1)
    row_group = thread >> Int32(5)
    column_chunk = column >> Int32(3)
    column_half = column & Int32(7)
    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
        row = row_group * Int32(32) + Int32(2 * pair)
        row_swizzle = (row >> Int32(1)) & Int32(3)
        # halves: 32 per row, 8 per 16-byte chunk
        offset = row * Int32(EPILOGUE_N) + ((column_chunk ^ row_swizzle) << Int32(3)) + column_half
        low = Uint32(staging_halves[offset])
        high = Uint32(staging_halves[offset + Int32(EPILOGUE_N)])
        t_pairs[pair] = low | (high << Uint32(16))
    scale_byte = quantize_pairs(t_pairs, payload)
    # hidden_t tile row ``column``, bytes [32 * row_group, +32): two swizzled 16-byte chunks.
    t_row = stage_fp8 + FUSED_HIDDEN_T_OFFSET + (column << Int32(7))
    for half in cutlass.range_constexpr(2):
        chunk_index = (row_group * Int32(2) + Int32(half)) ^ (column & Int32(7))
        store_shared_v4(t_row + (chunk_index << Int32(4)), payload[4 * half], payload[4 * half + 1], payload[4 * half + 2], payload[4 * half + 3])
    # Scale byte of hidden_t row 32 * subtile + column, block row_group; written after the last pair
    # (stage 2 holds the Gate tile until then).
    scale_words[1] = scale_words[1] | (scale_byte << shift)
    cute.arch.fence_view_async_shared()
    named_barrier_sync_unaligned(epilogue_barrier)
    if thread == Int32(0):
        store_tile(hidden_atom, hidden_fp8, (CTA_M, EPILOGUE_N), (tile_row, column_tile), hidden_tile, cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
        cute.arch.cp_async_bulk_commit_group()
        store_tile(
            hidden_t_atom, hidden_t, (HIDDEN_T_TILE_ROWS, QUANT_TILE), (column_tile, tile_row), hidden_t_tile, cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST)
        )
        cute.arch.cp_async_bulk_commit_group()


@cute.jit
def store_acc_fused_swiglu(
    collective,
    tiled_copy_t2r,
    tTR_tAcc,
    tTR_rAcc,
    gate_atom: cute.CopyAtom,
    gate_bf16: cute.Tensor,
    up_atom: cute.CopyAtom,
    up_bf16: cute.Tensor,
    hidden_atom: cute.CopyAtom,
    hidden_fp8: cute.Tensor,
    hidden_scales: cute.Pointer,
    hidden_t_atom: cute.CopyAtom,
    hidden_t: cute.Tensor,
    hidden_t_scales: cute.Pointer,
    scale_columns: Int32,
    scale_t_columns: Int32,
    d_stages: cute.Pointer,
    tile_row: Int32,
    tile_column: Int32,
    is_clamped: cutlass.Constexpr[bool],
    swiglu_limit: Float32,
    acc_pipeline,
    acc_consumer_state,
    thread: Int32,
    situ: cutlass.Constexpr[bool] = False,
    situ_beta1=None,
    situ_beta2=None,
):
    """The fused routed Gate/Up + SwiGLU epilogue of the MXFP8 forward.

    The accumulator holds this CTA's 128 rows x (Gate 128 columns | Up 128 columns) of one 128-column
    tile (CTA 0 loaded the Gate weights, CTA 1 the Up weights of the same rows).  Four 32-column
    pairs (``_fused_pair``) write the BF16 Gate / Up context (``gate_bf16`` / ``up_bf16``,
    macrobatch rows x intermediate), the Hidden e4m3 tile in both layouts and, after the last
    pair, the two 512 B Hidden scale tiles.  The routed experts therefore have no BF16 Gate / Up
    ring and no SwiGLU task; Hidden is the SwiGLU of the BF16-rounded Gate and Up.
    ``tile_row`` is the 128-row tile (macrobatch-local), ``tile_column`` the 128-column tile;
    ``scale_columns`` = intermediate / 128 (normal scale tiles), ``scale_t_columns`` =
    macrobatch / 128 (transposed).  ``situ`` (compile time): SiTU-GLU with the runtime
    ``situ_beta1`` / ``situ_beta2`` (Float32; None, and unread, otherwise) instead of SwiGLU.
    """
    epilogue_barrier = collective.epilogue_barrier
    scale_words = cute.make_rmem_tensor((2,), Int32)
    scale_words[0] = Int32(0)
    scale_words[1] = Int32(0)
    gate_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    up_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    hidden_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    t_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    payload = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
    assert cute.size(tTR_rAcc) == SF_VEC_SIZE  # one 32x32b x32 load: a row's 32 columns
    acc_pipeline.consumer_wait(acc_consumer_state)
    # unroll=1: one copy of the pair body for pairs 0-2, a second for the last pair,
    # which releases the accumulator right after its Up load.
    for subtile in cutlass.range(FUSED_SUBTILES - 1, unroll=1):
        _fused_pair(
            subtile,
            False,
            tiled_copy_t2r,
            tTR_tAcc,
            tTR_rAcc,
            epilogue_barrier,
            acc_pipeline,
            acc_consumer_state,
            gate_atom,
            gate_bf16,
            up_atom,
            up_bf16,
            hidden_atom,
            hidden_fp8,
            hidden_t_atom,
            hidden_t,
            d_stages,
            tile_row,
            tile_column,
            is_clamped,
            swiglu_limit,
            thread,
            scale_words,
            gate_pairs,
            up_pairs,
            hidden_pairs,
            t_pairs,
            payload,
            situ,
            situ_beta1,
            situ_beta2,
        )
    _fused_pair(
        Int32(FUSED_SUBTILES - 1),
        True,
        tiled_copy_t2r,
        tTR_tAcc,
        tTR_rAcc,
        epilogue_barrier,
        acc_pipeline,
        acc_consumer_state,
        gate_atom,
        gate_bf16,
        up_atom,
        up_bf16,
        hidden_atom,
        hidden_fp8,
        hidden_t_atom,
        hidden_t,
        d_stages,
        tile_row,
        tile_column,
        is_clamped,
        swiglu_limit,
        thread,
        scale_words,
        gate_pairs,
        up_pairs,
        hidden_pairs,
        t_pairs,
        payload,
        situ,
        situ_beta1,
        situ_beta2,
    )
    acc_consumer_state.advance()
    # The two Hidden scale tiles in stage 2 (the last pair's wait covered its Gate store): the normal
    # one (a row's four blocks are its four pairs) and the transposed one, byte (r % 32) * 16 +
    # (r / 32) * 4 + block of hidden_t row r = 32 * subtile + column.  Both leave through 512 B bulk copies.
    stage_sc = d_stages + FUSED_SCALE_STAGE * D_STAGE_BYTES
    store_scale_word(stage_sc + FUSED_HIDDEN_SC_OFFSET, thread, scale_words[0])
    column = thread & Int32(EPILOGUE_N - 1)
    row_group = thread >> Int32(5)
    for subtile in cutlass.range_constexpr(FUSED_SUBTILES):
        store_shared_u8(stage_sc + FUSED_HIDDEN_T_SC_OFFSET + column * Int32(16) + Int32(4 * subtile) + row_group, scale_words[1] >> Int32(8 * subtile))
    cute.arch.fence_view_async_shared()
    named_barrier_sync_unaligned(epilogue_barrier)
    if thread == Int32(0):
        normal_index = (Int64(tile_row) * Int64(scale_columns) + Int64(tile_column)) * Int64(SF_TILE_BYTES)
        tma_store_1d_raw(hidden_scales.toint() + normal_index, stage_sc + FUSED_HIDDEN_SC_OFFSET, Int32(SF_TILE_BYTES))
        cute.arch.cp_async_bulk_commit_group()
        tma_store_1d_raw(
            hidden_t_scales.toint() + (Int64(tile_column) * Int64(scale_t_columns) + Int64(tile_row)) * Int64(SF_TILE_BYTES),
            stage_sc + FUSED_HIDDEN_T_SC_OFFSET,
            Int32(SF_TILE_BYTES),
        )
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    named_barrier_sync_unaligned(pipeline.NamedBarrier(barrier_id=D_STORE_TAIL_BARRIER_ID, num_threads=128))
    return acc_consumer_state


@cute.jit
def _fused_pair_e4m3_context(
    subtile: Int32,
    release_acc: cutlass.Constexpr[bool],
    tiled_copy_t2r,
    tTR_tAcc,
    tTR_rAcc,
    epilogue_barrier,
    acc_pipeline,
    acc_consumer_state,
    gate_atom,
    gate_fp8,
    up_atom,
    up_fp8,
    hidden_atom,
    hidden_fp8,
    hidden_t_atom,
    hidden_t,
    d_stages: cute.Pointer,
    tile_row: Int32,
    tile_column: Int32,
    is_clamped: cutlass.Constexpr[bool],
    swiglu_limit: Float32,
    thread: Int32,
    scale_words: cute.Tensor,
    gate_pairs: cute.Tensor,
    up_pairs: cute.Tensor,
    hidden_pairs: cute.Tensor,
    t_pairs: cute.Tensor,
    payload: cute.Tensor,
    situ: cutlass.Constexpr[bool] = False,
    situ_beta1=None,
    situ_beta2=None,
) -> None:
    """One 32-column pair of the fused epilogue: Gate subtile ``subtile`` and Up subtile
    ``subtile + 4`` to BF16 and their e4m3 context, then Hidden = SwiGLU of the two BF16 values,
    quantized in the normal layout (this thread's row: one 32-element block) and the transposed
    layout (one Hidden column's 32 rows: a warp's rows, through the BF16 staging tile).
    ``scale_words[0..3]`` accumulate the Gate / Up / Hidden / transposed-Hidden scale bytes; the
    ``*_pairs`` and ``payload`` registers are the caller's (allocated once, outside its loop).
    ``release_acc``: the last pair releases the accumulator right after its Up load.
    """
    stage_bf16 = d_stages + FUSED_HIDDEN_BF16_STAGE * D_STAGE_BYTES
    stage_fp8 = d_stages + FUSED_HIDDEN_FP8_STAGE * D_STAGE_BYTES
    stage_ctx = d_stages + E4M3_CONTEXT_STAGE * D_STAGE_BYTES
    tile_layout = cute.make_layout((CTA_M, EPILOGUE_N), stride=(EPILOGUE_N, 1))
    ctx_tile = cute.make_tensor(cute.recast_ptr(stage_ctx, dtype=Float8E4M3FN), tile_layout)
    hidden_tile = cute.make_tensor(cute.recast_ptr(stage_fp8, dtype=Float8E4M3FN), tile_layout)
    ht_layout = hidden_t_smem_layout()
    hidden_t_tile = cute.make_tensor(cute.recast_ptr(stage_fp8 + FUSED_HIDDEN_T_OFFSET, ht_layout.inner, dtype=Float8E4M3FN), ht_layout.outer)
    row_offset = thread * Int32(EPILOGUE_N)
    swap_half = (thread >> Int32(2)) & Int32(1)
    shift = (subtile & Int32(3)) * Int32(8)
    column_tile = tile_column * Int32(FUSED_SUBTILES) + subtile
    # --- Gate: TMEM -> FP32 -> BF16 pairs, the e4m3 context tile through the context stage ---
    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile)], tTR_rAcc)
    cute.arch.fence_view_async_tmem_load()
    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
        gate_pairs[pair] = cvt_bf16x2(tTR_rAcc[2 * pair], tTR_rAcc[2 * pair + 1])
    if thread == Int32(0):
        # The previous pair's Up context store must have read the context stage.
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    named_barrier_sync_unaligned(epilogue_barrier)
    scale_words[0] = scale_words[0] | (quantize_pairs(gate_pairs, payload) << shift)
    _store_row_block(stage_ctx + row_offset, swap_half, payload)
    cute.arch.fence_view_async_shared()
    named_barrier_sync_unaligned(epilogue_barrier)
    if thread == Int32(0):
        store_tile(gate_atom, gate_fp8, (CTA_M, EPILOGUE_N), (tile_row, column_tile), ctx_tile, cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
        cute.arch.cp_async_bulk_commit_group()
    # --- Up: the same, after the Gate context store has read the stage ---
    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile + Int32(FUSED_SUBTILES))], tTR_rAcc)
    cute.arch.fence_view_async_tmem_load()
    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
        up_pairs[pair] = cvt_bf16x2(tTR_rAcc[2 * pair], tTR_rAcc[2 * pair + 1])
    if cutlass.const_expr(release_acc):
        # Every TMEM read of this tile is complete (wait::ld above): the MMA may start the next tile.
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
    if thread == Int32(0):
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    named_barrier_sync_unaligned(epilogue_barrier)
    scale_words[1] = scale_words[1] | (quantize_pairs(up_pairs, payload) << shift)
    _store_row_block(stage_ctx + row_offset, swap_half, payload)
    cute.arch.fence_view_async_shared()
    named_barrier_sync_unaligned(epilogue_barrier)
    if thread == Int32(0):
        store_tile(up_atom, up_fp8, (CTA_M, EPILOGUE_N), (tile_row, column_tile), ctx_tile, cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
        cute.arch.cp_async_bulk_commit_group()
    # --- Hidden = SwiGLU(Gate BF16, Up BF16) -> BF16, computed on the BF16-rounded values ---
    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
        gate_low = (gate_pairs[pair] << Uint32(16)).bitcast(Float32)
        gate_high = (gate_pairs[pair] & Uint32(0xFFFF0000)).bitcast(Float32)
        up_low = (up_pairs[pair] << Uint32(16)).bitcast(Float32)
        up_high = (up_pairs[pair] & Uint32(0xFFFF0000)).bitcast(Float32)
        if cutlass.const_expr(is_clamped):
            # The clamp is a single min.f32 / max.f32 each.
            limit = Float32(swiglu_limit)
            gate_low = cute.arch.fmin(gate_low, limit)
            gate_high = cute.arch.fmin(gate_high, limit)
            up_low = cute.arch.fmin(cute.arch.fmax(up_low, -limit), limit)
            up_high = cute.arch.fmin(cute.arch.fmax(up_high, -limit), limit)
        hidden_pairs[pair] = cvt_bf16x2(
            glu_forward(gate_low, up_low, situ, situ_beta1, situ_beta2), glu_forward(gate_high, up_high, situ, situ_beta1, situ_beta2)
        )
    # Normal layout: this thread's row, these 32 columns are one block.  The previous pair's Hidden
    # stores have read stage 1 (the wait above covered every pending group).
    scale_words[2] = scale_words[2] | (quantize_pairs(hidden_pairs, payload) << shift)
    _store_row_block(stage_fp8 + row_offset, swap_half, payload)
    # Stage the BF16 row for the transposed pass: 16-byte chunk k of row r at chunk k ^ ((r / 2) % 4),
    # so eight consecutive rows' stores cover all 32 banks.  The previous pair's reads of this
    # stage ended before its last barrier.
    staging_row = stage_bf16 + thread * Int32(2 * EPILOGUE_N)
    staging_swizzle = (thread >> Int32(1)) & Int32(3)
    for chunk in cutlass.range_constexpr(4):
        store_shared_v4(
            staging_row + ((Int32(chunk) ^ staging_swizzle) << Int32(4)),
            hidden_pairs[4 * chunk].bitcast(Int32),
            hidden_pairs[4 * chunk + 1].bitcast(Int32),
            hidden_pairs[4 * chunk + 2].bitcast(Int32),
            hidden_pairs[4 * chunk + 3].bitcast(Int32),
        )
    named_barrier_sync_unaligned(epilogue_barrier)
    # Transposed layout: thread t quantizes Hidden column (t % 32) over its warp's 32 rows (one
    # block of hidden_t's row ``32 * subtile + t % 32``), the block division of ``quantize_tile``.  The
    # staging is read through a tensor view (ordinary loads the barrier orders).
    staging_halves = cute.make_tensor(cute.recast_ptr(stage_bf16, dtype=Uint16), cute.make_layout((CTA_M * EPILOGUE_N,)))
    column = thread & Int32(EPILOGUE_N - 1)
    row_group = thread >> Int32(5)
    column_chunk = column >> Int32(3)
    column_half = column & Int32(7)
    for pair in cutlass.range_constexpr(PAIRS_PER_BLOCK):
        row = row_group * Int32(32) + Int32(2 * pair)
        row_swizzle = (row >> Int32(1)) & Int32(3)
        # halves: 32 per row, 8 per 16-byte chunk
        offset = row * Int32(EPILOGUE_N) + ((column_chunk ^ row_swizzle) << Int32(3)) + column_half
        low = Uint32(staging_halves[offset])
        high = Uint32(staging_halves[offset + Int32(EPILOGUE_N)])
        t_pairs[pair] = low | (high << Uint32(16))
    scale_byte = quantize_pairs(t_pairs, payload)
    # hidden_t tile row ``column``, bytes [32 * row_group, +32): two swizzled 16-byte chunks.
    t_row = stage_fp8 + FUSED_HIDDEN_T_OFFSET + (column << Int32(7))
    for half in cutlass.range_constexpr(2):
        chunk_index = (row_group * Int32(2) + Int32(half)) ^ (column & Int32(7))
        store_shared_v4(t_row + (chunk_index << Int32(4)), payload[4 * half], payload[4 * half + 1], payload[4 * half + 2], payload[4 * half + 3])
    # Scale byte of hidden_t row r = 32 * subtile + column, block row_group:
    # (r % 32) * 16 + (r / 32) * 4 + block = column * 16 + subtile * 4 + row_group.
    store_shared_u8(stage_ctx + E4M3_CONTEXT_HIDDEN_T_SC_OFFSET + column * Int32(16) + subtile * Int32(4) + row_group, scale_byte)
    cute.arch.fence_view_async_shared()
    named_barrier_sync_unaligned(epilogue_barrier)
    if thread == Int32(0):
        store_tile(hidden_atom, hidden_fp8, (CTA_M, EPILOGUE_N), (tile_row, column_tile), hidden_tile, cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST))
        cute.arch.cp_async_bulk_commit_group()
        store_tile(
            hidden_t_atom, hidden_t, (HIDDEN_T_TILE_ROWS, QUANT_TILE), (column_tile, tile_row), hidden_t_tile, cache_policy=Int64(TMA_CACHE_HINT_EVICT_FIRST)
        )
        cute.arch.cp_async_bulk_commit_group()


@cute.jit
def store_acc_fused_swiglu_e4m3_context(
    collective,
    tiled_copy_t2r,
    tTR_tAcc,
    tTR_rAcc,
    gate_atom: cute.CopyAtom,
    gate_fp8: cute.Tensor,
    gate_scales: cute.Pointer,
    up_atom: cute.CopyAtom,
    up_fp8: cute.Tensor,
    up_scales: cute.Pointer,
    hidden_atom: cute.CopyAtom,
    hidden_fp8: cute.Tensor,
    hidden_scales: cute.Pointer,
    hidden_t_atom: cute.CopyAtom,
    hidden_t: cute.Tensor,
    hidden_t_scales: cute.Pointer,
    scale_columns: Int32,
    scale_t_columns: Int32,
    d_stages: cute.Pointer,
    tile_row: Int32,
    tile_column: Int32,
    is_clamped: cutlass.Constexpr[bool],
    swiglu_limit: Float32,
    acc_pipeline,
    acc_consumer_state,
    thread: Int32,
    situ: cutlass.Constexpr[bool] = False,
    situ_beta1=None,
    situ_beta2=None,
):
    """The fused routed Gate/Up + SwiGLU epilogue of the MXFP8 forward.

    The accumulator holds this CTA's 128 rows x (Gate 128 columns | Up 128 columns) of one 128-column
    tile (CTA 0 loaded the Gate weights, CTA 1 the Up weights of the same rows).  Four 32-column
    pairs (``_fused_pair_e4m3_context``) write the Gate / Up e4m3 context, the Hidden e4m3 tile in both layouts
    and, after the last pair, the four 512 B scale tiles.  The routed experts therefore have no BF16
    Gate / Up ring and no SwiGLU task; Hidden is the SwiGLU of the BF16-rounded Gate and Up.
    ``tile_row`` is the 128-row tile (macrobatch-local), ``tile_column`` the 128-column tile;
    ``scale_columns`` = intermediate / 128 (normal scale tiles), ``scale_t_columns`` =
    macrobatch / 128 (transposed).  ``situ`` (compile time): SiTU-GLU with the runtime
    ``situ_beta1`` / ``situ_beta2`` (Float32; None, and unread, otherwise) instead of SwiGLU.
    """
    epilogue_barrier = collective.epilogue_barrier
    scale_words = cute.make_rmem_tensor((4,), Int32)
    for word in cutlass.range_constexpr(4):
        scale_words[word] = Int32(0)
    gate_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    up_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    hidden_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    t_pairs = cute.make_rmem_tensor((PAIRS_PER_BLOCK,), Uint32)
    payload = cute.make_rmem_tensor((PAYLOAD_WORDS,), Int32)
    assert cute.size(tTR_rAcc) == SF_VEC_SIZE  # one 32x32b x32 load: a row's 32 columns
    acc_pipeline.consumer_wait(acc_consumer_state)
    # unroll=1: one copy of the pair body for pairs 0-2, a second for the last pair,
    # which releases the accumulator right after its Up load.
    for subtile in cutlass.range(FUSED_SUBTILES - 1, unroll=1):
        _fused_pair_e4m3_context(
            subtile,
            False,
            tiled_copy_t2r,
            tTR_tAcc,
            tTR_rAcc,
            epilogue_barrier,
            acc_pipeline,
            acc_consumer_state,
            gate_atom,
            gate_fp8,
            up_atom,
            up_fp8,
            hidden_atom,
            hidden_fp8,
            hidden_t_atom,
            hidden_t,
            d_stages,
            tile_row,
            tile_column,
            is_clamped,
            swiglu_limit,
            thread,
            scale_words,
            gate_pairs,
            up_pairs,
            hidden_pairs,
            t_pairs,
            payload,
            situ,
            situ_beta1,
            situ_beta2,
        )
    _fused_pair_e4m3_context(
        Int32(FUSED_SUBTILES - 1),
        True,
        tiled_copy_t2r,
        tTR_tAcc,
        tTR_rAcc,
        epilogue_barrier,
        acc_pipeline,
        acc_consumer_state,
        gate_atom,
        gate_fp8,
        up_atom,
        up_fp8,
        hidden_atom,
        hidden_fp8,
        hidden_t_atom,
        hidden_t,
        d_stages,
        tile_row,
        tile_column,
        is_clamped,
        swiglu_limit,
        thread,
        scale_words,
        gate_pairs,
        up_pairs,
        hidden_pairs,
        t_pairs,
        payload,
        situ,
        situ_beta1,
        situ_beta2,
    )
    acc_consumer_state.advance()
    # The three normal scale tiles (a row's four blocks are its four pairs); the transposed one was
    # filled byte by byte above.  All four leave through 512 B bulk copies.
    stage_ctx = d_stages + E4M3_CONTEXT_STAGE * D_STAGE_BYTES
    store_scale_word(stage_ctx + E4M3_CONTEXT_GATE_SC_OFFSET, thread, scale_words[0])
    store_scale_word(stage_ctx + E4M3_CONTEXT_UP_SC_OFFSET, thread, scale_words[1])
    store_scale_word(stage_ctx + E4M3_CONTEXT_HIDDEN_SC_OFFSET, thread, scale_words[2])
    cute.arch.fence_view_async_shared()
    named_barrier_sync_unaligned(epilogue_barrier)
    if thread == Int32(0):
        normal_index = (Int64(tile_row) * Int64(scale_columns) + Int64(tile_column)) * Int64(SF_TILE_BYTES)
        for offset, scales in (
            (E4M3_CONTEXT_GATE_SC_OFFSET, gate_scales),
            (E4M3_CONTEXT_UP_SC_OFFSET, up_scales),
            (E4M3_CONTEXT_HIDDEN_SC_OFFSET, hidden_scales),
        ):
            tma_store_1d_raw(scales.toint() + normal_index, stage_ctx + offset, Int32(SF_TILE_BYTES))
            cute.arch.cp_async_bulk_commit_group()
        tma_store_1d_raw(
            hidden_t_scales.toint() + (Int64(tile_column) * Int64(scale_t_columns) + Int64(tile_row)) * Int64(SF_TILE_BYTES),
            stage_ctx + E4M3_CONTEXT_HIDDEN_T_SC_OFFSET,
            Int32(SF_TILE_BYTES),
        )
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    named_barrier_sync_unaligned(pipeline.NamedBarrier(barrier_id=D_STORE_TAIL_BARRIER_ID, num_threads=128))
    return acc_consumer_state


__all__ = [
    "FP8_K_TILE",
    "MXFP8_ARENA_BYTES",
    "MXFP8_D_OFFSET",
    "MXFP8_TMEM_COLUMNS",
    "SFA_BYTES",
    "SFA_TMEM_STAGE_COLUMNS",
    "SFB_TMEM_STAGE_COLUMNS",
    "D_STORE_TAIL_BARRIER_ID",
    "EPILOGUE_BARRIER_ID",
    "TMEM_ALLOC_BARRIER_ID",
    "SF_TILE_BYTES",
    "SF_VEC_SIZE",
    "make_sf_load_fns",
    "mma_blockscaled",
    "scale_operand",
    "store_acc_fused_swiglu",
    "store_acc_fused_swiglu_e4m3_context",
    "store_acc_mxfp8",
    "store_tile",
    "context_smem_layout",
    "hidden_t_smem_layout",
    "D_STAGE_BYTES",
    "FUSED_SUBTILES",
    "HIDDEN_T_TILE_ROWS",
]
