# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Swap-AB routed GEMMs of the BF16 mega kernels (device side): the forward's routed Gate/Up and
Down tiles and the backward's routed Gate/Up and Down dgrad tiles.

The routed tiles run with the expert weights as MMA operand A (M = 256 features, 128 per CTA) and the
token rows as operand B, so the instruction's N is the token count: align16(the expert's valid rows
left in this 256-row block), a runtime value.  Only each expert's last block shrinks; the padded row
space, the communication prefix and every counter do not depend on N.  The epilogues handle
the accumulator transposed (tokens as rows) and skip the 32-token subtiles at or past N: the forward
Down GEMM never reads those Hidden rows, the combine never pushes those y rows, and the backward's
reverse dispatch never pushes those dX rows.  Exceptions: a macrobatch-0 (context-saving) forward Gate/Up
tile stores those subtiles as zero Gate / Up / Hidden, since a backward reads the saved context of every
padded row; the backward's Down dgrad rounds N up to the 64-row Wgrad K block instead, since the routed
Wgrads read the dGate / dUp / w*Hidden rows its fused SwiGLU-backward epilogue writes up to that end.

Operand majors: the forward tiles have both operands K-major.  The backward's dgrads have the expert
weights as A with the output features as M and contiguous in memory ([Wg; Wu] with H for the Gate/Up dX,
Wd with I for the Down dH), i.e. an MN-major A, staged as two 64-feature x 64-K slabs per stage
(``A_MN_SLAB_*``; ``mma_dynamic_n(..., a_mn_major=True)`` sets the instruction descriptor's A-major bit
and builds the MN-major shared-memory descriptors).

Pieces (used by ``_persistent_bf16_fwd_mega`` and ``_persistent_bf16_bwd_mega``):
  * ``tile_valid_rows`` / ``mma_n``: a block's valid rows (from the scheduler's valid count of the expert
    and the expert's rows before the block) and the instruction's N.
  * ``nonleader_b_shift``: cta_group::2 splits N between the two CTAs at N / 2, so the non-leader CTA
    loads its B half from token N / 2 instead of 128.
  * ``mma_dynamic_n``: the mainloop of one routed tile: QuACK ``mma``'s pipeline protocol with raw
    ``tcgen05.mma.cta_group::2.kind::f16`` (four per 64-K tile, descriptors from the stage addresses, as
    the MXFP8 forward's mainloop does) and a runtime ``n_dim`` in the instruction descriptor.
  * ``D_LAYOUT`` / ``d_smem_layout`` / ``epilogue_copies``: the routed Down epilogue's transposed store.
    In MMA coordinates the tile's D is (features, tokens) with the features contiguous in y, i.e.
    M-major; CUTLASS's helpers pick the TMEM load (16x256b) and the shared-memory store (stmatrix
    .trans) for that layout and the swizzled shared-memory tile the TMA store reads.
  * ``store_tile``: one shared-memory tile to a global tile (one thread, TMA), for the transposed stores.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.warp import StMatrix8x8x16bOp
from cutlass.utils import LayoutEnum
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .persistent_bf16_fwd_contract import MLP_TILE_ROWS

TILE_TOKENS = MLP_TILE_ROWS  # a routed row block is the tile's full N
MMA_N_ALIGNMENT = 16  # kind::f16 cta_group::2 N granularity
# kind::f16 instruction descriptor, static part: F32 accumulator (bit 4), BF16 A (bit 7) and B (bit 10),
# both K-major, M 256 (m_dim 16 at bit 24); n_dim (N >> 3) is OR'd in at bit 17 at run time.  Bit 15
# (a_major) marks an MN-major A.
IDESC_BF16_M256 = 0x10000490
IDESC_N_SHIFT = 17
IDESC_A_MN_MAJOR = 1 << 15
# Shared-memory descriptor high word of the A / B stage tiles (128 rows x 128 B, 128 B swizzle, SBO 1 KiB):
# the same geometry as the MXFP8 forward's e4m3 tiles (_persistent_mxfp8_fwd_gemm.AB_DESC_HIGH).
AB_DESC_HIGH = 0x40004040
# K-major tiles advance 32 B (2 descriptor units) per K16 instruction; LBO is unused (1).
K_MAJOR_K_STEP_UNITS = 2
# MN-major A stage tile (the backward's weights with the output features as M): the 128 B swizzle atom spans
# 64 BF16 features, so this CTA's 128 features are two slabs of 64 features x 64 K rows (128 B each),
# 8 KiB apart (canonical ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO)): LBO = the slab stride, SBO = the 1 KiB
# 8-row K group as in the K-major tiles).  A K16 instruction advances 16 rows = 2 KiB.
A_MN_SLAB_ROWS = 64
A_MN_SLAB_BYTES = A_MN_SLAB_ROWS * 64 * 2
A_MN_LBO_UNITS = A_MN_SLAB_BYTES >> 4
A_MN_K_STEP_UNITS = (16 * 128) >> 4
assert A_MN_LBO_UNITS < (1 << 14)


@cute.jit
def tile_valid_rows(expert_valid_rows: Int32, expert_row_offset: Int32) -> Int32:
    """Valid rows of a routed 256-row block: the expert's valid (unpadded) rows minus its rows before
    the block, clamped to [0, 256]."""
    return cutlass.max(Int32(0), cutlass.min(expert_valid_rows - expert_row_offset, Int32(TILE_TOKENS)))


@cute.jit
def mma_n(valid_rows: Int32) -> Int32:
    """The routed tile's N: align16(valid rows), at least 16."""
    return cutlass.max(Int32(MMA_N_ALIGNMENT), (valid_rows + Int32(MMA_N_ALIGNMENT - 1)) & Int32(-MMA_N_ALIGNMENT))


@cute.jit
def nonleader_b_shift(n: Int32) -> Int32:
    """Token offset of the non-leader CTA's B half relative to the static split (128): n / 2 - 128."""
    return (n >> Int32(1)) - Int32(TILE_TOKENS // 2)


@cute.jit
def _smem_desc(address: Int32) -> Int64:
    """Shared-memory descriptor of a K-major tile: ``address >> 4`` (14 bits), LBO 16 B, AB_DESC_HIGH in
    the upper word."""
    low = ((address >> Int32(4)) & Int32(0x3FFF)) | Int32(0x00010000)
    return (Int64(AB_DESC_HIGH) << Int64(32)) | Int64(Uint32(low))


@cute.jit
def _smem_desc_a_mn(address: Int32) -> Int64:
    """Shared-memory descriptor of the MN-major A tile: ``address >> 4``, LBO the slab stride
    (A_MN_LBO_UNITS), the same SBO / swizzle high word as the K-major tiles."""
    low = ((address >> Int32(4)) & Int32(0x3FFF)) | Int32(A_MN_LBO_UNITS << 16)
    return (Int64(AB_DESC_HIGH) << Int64(32)) | Int64(Uint32(low))


@dsl_user_op
def _issue_k_tile(
    d_tmem: Int32,
    a_desc: Int64,
    b_desc: Int64,
    idesc: Int32,
    accumulate: Int32,
    empty_barrier: Int32,
    mask: Int32,
    *,
    a_k_step: int = K_MAJOR_K_STEP_UNITS,
    loc=None,
    ip=None,
) -> None:
    """One 64-K tile: four kind::f16 MMAs (K 16 each; the B descriptor + 2 per 32 B, the A descriptor
    + ``a_k_step`` units: 2 for a K-major A, A_MN_K_STEP_UNITS for the MN-major A) and the commit to
    the A/B stage's empty barrier (both CTAs), issued by one elected lane."""
    llvm.inline_asm(
        None,
        [
            Int32(d_tmem).ir_value(loc=loc, ip=ip),
            Int64(a_desc).ir_value(loc=loc, ip=ip),
            Int64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Int32(accumulate).ir_value(loc=loc, ip=ip),
            Int32(empty_barrier).ir_value(loc=loc, ip=ip),
            Int32(mask).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .pred leader, first, on;\n.reg .b32 lane;\n.reg .b64 a1, a2, a3, b1, b2, b3;\n.reg .b16 mask;\n"
        "elect.sync lane|leader, 0xffffffff;\n"
        "setp.ne.b32 first, $4, 0;\n"
        "setp.eq.b32 on, $4, $4;\n"
        f"add.s64 a1, $1, {a_k_step};\nadd.s64 a2, $1, {2 * a_k_step};\nadd.s64 a3, $1, {3 * a_k_step};\n"
        "add.s64 b1, $2, 2;\nadd.s64 b2, $2, 4;\nadd.s64 b3, $2, 6;\n"
        "cvt.u16.u32 mask, $6;\n"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [$0], $1, $2, $3, first;\n"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [$0], a1, b1, $3, on;\n"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [$0], a2, b2, $3, on;\n"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [$0], a3, b3, $3, on;\n"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$5], mask;\n"
        "}",
        "r,l,l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.jit
def mma_dynamic_n(
    ab_pipeline,
    acc_pipeline,
    ab_state,
    acc_state,
    a_base: Int32,
    b_base: Int32,
    stage_bytes: cutlass.Constexpr[int],
    d_tmem: Int32,
    k_tiles: Int32,
    n: Int32,
    is_leader_cta,
    a_mn_major: cutlass.Constexpr[bool] = False,
):
    """One routed tile.  Leader CTA: acquire the accumulator stage, per K tile wait A/B, issue the four
    MMAs with ``n_dim = n >> 3`` and commit the stage's empty barrier, then commit the accumulator.  Both
    CTAs advance the states as QuACK's ``mma`` does.  ``d_tmem`` is this accumulator stage's TMEM address.
    ``n`` comes from a global load; it is first used after the accumulator wait, so the load latency
    overlaps that wait.  ``a_mn_major``: the A stage holds the MN-major two-slab tile (A_MN_SLAB_*) instead
    of the K-major 128 x 128 B tile; B is K-major either way."""
    ab_mask = Int32(ab_pipeline.consumer_mask)
    if is_leader_cta:
        acc_pipeline.producer_acquire(acc_state)
    idesc = Int32(IDESC_BF16_M256) | ((n >> Int32(3)) << Int32(IDESC_N_SHIFT))
    if cutlass.const_expr(a_mn_major):
        idesc = idesc | Int32(IDESC_A_MN_MAJOR)
    for k_tile in cutlass.range(k_tiles, unroll=1):
        if is_leader_cta:
            ab_pipeline.consumer_wait(ab_state)
            stage = ab_state.index
            if cutlass.const_expr(a_mn_major):
                _issue_k_tile(
                    d_tmem,
                    _smem_desc_a_mn(a_base + stage * Int32(stage_bytes)),
                    _smem_desc(b_base + stage * Int32(stage_bytes)),
                    idesc,
                    Int32(k_tile != Int32(0)),
                    Int32(ab_pipeline.sync_object_empty.get_barrier(stage).toint()),
                    ab_mask,
                    a_k_step=A_MN_K_STEP_UNITS,
                )
            else:
                _issue_k_tile(
                    d_tmem,
                    _smem_desc(a_base + stage * Int32(stage_bytes)),
                    _smem_desc(b_base + stage * Int32(stage_bytes)),
                    idesc,
                    Int32(k_tile != Int32(0)),
                    Int32(ab_pipeline.sync_object_empty.get_barrier(stage).toint()),
                    ab_mask,
                )
        ab_state.advance()
    if is_leader_cta:
        acc_pipeline.producer_commit(acc_state)
    acc_state.advance()
    return ab_state, acc_state


D_LAYOUT = LayoutEnum.COL_MAJOR  # (features, tokens) with the features contiguous


def d_smem_layout(epi_tile: cute.Tile, stages: int):
    """Staged shared-memory layout of the transposed epilogue tiles (M-major, swizzled); the host's TMA
    store atom and the kernel's staging view are both built from it."""
    return sm100_utils.make_smem_layout_epi(BFloat16, D_LAYOUT, epi_tile, stages)


def epilogue_copies(cta_tile_shape_mnk, acc: cute.Tensor, epi_tile: cute.Tile, is_two_cta, s_d: cute.Tensor, thread: Int32):
    """The transposed epilogue's copies, as QuACK's epilog_tmem_copy_and_partition /
    epilog_smem_store_and_partition with D M-major: (t2r, tTR_tAcc, tTR_rAcc, r2s, tRS_rD, tRS_sD)."""
    t2r_atom = sm100_utils.get_tmem_load_op(cta_tile_shape_mnk, D_LAYOUT, BFloat16, Float32, epi_tile, is_two_cta)
    acc_epi = cute.flat_divide(acc[((None, None), 0, 0, None)], epi_tile)
    t2r = tcgen05.make_tmem_copy(t2r_atom, acc_epi[(None, None, 0, 0, 0)])
    thr_t2r = t2r.get_slice(thread)
    tTR_tAcc = thr_t2r.partition_S(acc_epi)
    c_acc_epi = cute.flat_divide(cute.make_identity_tensor(cta_tile_shape_mnk[:2]), epi_tile)
    tTR_rAcc = cute.make_rmem_tensor(thr_t2r.partition_D(c_acc_epi)[None, None, None, 0, 0].shape, Float32)
    r2s_atom = sm100_utils.get_smem_store_op(D_LAYOUT, BFloat16, Float32, t2r)
    assert isinstance(r2s_atom.op, StMatrix8x8x16bOp), f"expected stmatrix for the transposed store, got {r2s_atom.op}"
    r2s = cute.make_tiled_copy_D(r2s_atom, t2r)
    tRS_sD = r2s.get_slice(thread).partition_D(s_d)
    tRS_rD = r2s.retile(cute.make_rmem_tensor(tTR_rAcc.shape, Float32))
    return t2r, tTR_tAcc, tTR_rAcc, r2s, tRS_rD, tRS_sD


@cute.jit
def store_tile(atom: cute.CopyAtom, tensor: cute.Tensor, tile_shape: cute.Tile, coord, smem_tile: cute.Tensor):
    """One shared-memory tile to ``tensor``'s tile ``coord`` (one thread, TMA)."""
    global_tile = cute.local_tile(tensor, tile_shape, coord)
    smem_part, global_part = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(smem_tile, 0, cute.rank(smem_tile)), cute.group_modes(global_tile, 0, cute.rank(global_tile))
    )
    cute.copy(atom, smem_part, global_part)


__all__ = [
    "A_MN_SLAB_BYTES",
    "A_MN_SLAB_ROWS",
    "D_LAYOUT",
    "TILE_TOKENS",
    "d_smem_layout",
    "epilogue_copies",
    "mma_dynamic_n",
    "mma_n",
    "nonleader_b_shift",
    "store_tile",
    "tile_valid_rows",
]
