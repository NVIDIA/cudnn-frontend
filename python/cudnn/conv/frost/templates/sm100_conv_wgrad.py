# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsplit shape-generic wgrad: output (K,C), reduction L=N*Z*P*Q.

    dW[k, c, t, r, s] = sum over (n, z, p, q) of
        dY[n, k, z, p, q] * X[n, c,
                              z*stride[0] - padding[0] + t*dilation[0],
                              p*stride[1] - padding[1] + r*dilation[1],
                              q*stride[2] - padding[2] + s*dilation[2]]

Input coordinates that fall in the padding contribute zero (the im2col tensor map
masks them), so no host-side halo handling is needed.

The whole weight gradient is one persistent kernel over implicit GEMMs, one per
output tile:

    M = cta_tile_m (64 or 128)   K-out channels of dW, read from dY   (operand A)
    N = cta_tile_n (32..256)     C-in  channels of dW, read from X    (operand B)
    K = a_layout.reduction       output pixels, folded over n, z, p, q

The reduction is the large axis here, so it is neither split across CTAs nor
atomically accumulated: exactly one CTA owns each dW element and each element is
produced by two ordered chains of MMA issues folded once, so a relaunch over the
same shape reproduces the same bytes. The batch axis is part of that reduction
and carries no separate partial-sum contract. Work items are (k tile, c tile,
filter position t*r*s) dealt by the static persistent scheduler, so the compiled
artifact does not depend on the shape it runs on.

MN-major TMA boxes keep channels contiguous and pixels in the reduction axis.
OperandLayout derives TMA boxes, SMEM capacity, descriptors and MMA offsets
from the same TMA-box geometry. TF32 uses the required 32-byte-atomic swizzle.
Operand slots are released by MMA commit; every TMEM fragment participates in
its release dependency; output slots drain bulk reads before reuse.

Warp roles (block = 6 warps, 192 threads):

    warps 0-3   epilogue: TMEM -> registers -> SMEM -> TMA store of dW
    warp 4      MMA: issues the two tcgen05.mma chains and their commits
    warp 5      producer: TMA loads of the dY and X TMA boxes (im2col for X)

The reduction is accumulated as two independent chains, one per TMEM slot, and
the two partial sums are folded in IEEE FP32 by the epilogue. The tensor-core
FP32 accumulator truncates on every MMA issue, so a single chain multiplies its
per-issue error by the number of issues (a coherent error growing as L^1.5);
halving the chain length keeps that error under an atol-bound tolerance for
long reductions. When both chains of two work items fit in the 512-column TMEM
(cta_tile_n <= 128), two accumulator stages double-buffer consecutive work
items; epilogue slots release TMEM immediately after tcgen05.ld and register
consumption, before SMEM staging and TMA stores.

Coverage: test/python/conv/frost/test_frost_conv_wgrad.py (numerics against an
independent FP64 contraction, every dtype/tile/CTA-group combination, geometry,
epilogues, release rendering) and test_frost_conv_wgrad_determinism.py (replay,
CUDA-graph capture, no allocation and no host sync on the execute path).
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Type

from cuda.bindings import driver as cuda_driver
import cutlass
from cutlass import Numeric, utils
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims
import cutlass.experimental.cuda as cuda
from cudnn._cutlass_helpers import static_persistent_tile_scheduler as scheduler

from cudnn.frost.device import build_target, device_context
from cudnn.frost.tile_dsl.consume import consume_then_arrive, consume_then_arrive_on_peer, release_zero
from ..tile_config import ConvTileConfig
from .epilogue import get_epilogue_op
from .sm100_common import _EpiDatapath, _c_smem_swizzle, _mma_kind_for_dtype, _select_epi_datapath, _to_tensor_map_swizzle

DEFAULT_CONFIG = ConvTileConfig(128, 128, 128, 1)

# Hardware, layout, and swizzle constants
_BITS_PER_BYTE = 8
_TF32_BITS = 32
_MIN_SWIZZLE_BYTES = 32
_MAX_SWIZZLE_BYTES = 128
_MMA_INST_K_BYTES = 32
_SMEM_DESC_UNIT_BYTES = 16
_TENSOR_ALIGN_BYTES = 16
_TENSOR_ALIGN_BITS = _TENSOR_ALIGN_BYTES * _BITS_PER_BYTE
_SMEM_ALIGNMENT = 1024
_TMEM_COLS = 512

# Pipeline stage counts, warp roles, and synchronization constants
_NUM_AB_STAGES = 3
# Accumulation chains folded per dW element in TMEM (Rule 12). When two pairs
# of chains fit in 512 TMEM columns (cta_tile_n <= 128), two inter-tile
# accumulator stages double-buffer consecutive work items.
_NUM_ACC_CHAINS = 2
_MAX_ACC_STAGES = 2
_NUM_C_STAGES = 2
_WARP_SIZE = 32
_ALLOCATOR_WARP_ID = 0
_NUM_EPI_WARPS = 4
_MMA_WARP_ID = 4
_TMA_WARP_ID = 5
_NUM_WARPS = 6
_EPI_THREADS = _NUM_EPI_WARPS * _WARP_SIZE
_TMEM_ALLOC_SYNC_THREADS = (_NUM_EPI_WARPS + 1) * _WARP_SIZE
_TMEM_ALLOC_BAR_ID = 1
_EPI_SYNC_BAR_ID = 2
_MBAR_WAIT_TIMEOUT = 10_000_000

# Epilogue staging constants
_STMATRIX_ROW_LANES = 16
_STMATRIX_CHANNEL_HALVES = 2
_STMATRIX_WORDS = 4
_STMATRIX_ALIGN_BYTES = 32
_SWIZZLE_32B_MASK = 128
_SWIZZLE_32B_SHIFT = 3
_LDTM_CONTIGUOUS_CHANNELS = 32
_STORE_SWIZZLED_ALIGN_BYTES = 64


@dataclass(frozen=True)
class OperandLayout:
    """A TMA box is (contiguous MN channels, reduction pixels), swizzled in bytes.

    Both tcgen05 operands are MN-major (``a_major``/``b_major`` = 1), so a
    pipeline stage is a stack of whole swizzle TMA boxes: each TMA box is
    ``tma_box_channels`` channels by ``reduction`` pixels, laid out so that the
    channel axis of every pixel row is one contiguous, swizzled run.
    Channel counts that do not fill the last TMA box are padded up to it, which is
    why ``stage_bytes`` can exceed ``channels * reduction * bits / 8``.

    Every consumer of the TMA-box geometry -- the host TMA box, the SMEM stage
    capacity, the tcgen05 SMEM descriptors and the per-MMA-issue descriptor
    offsets -- reads this one object, so a change here cannot drift between
    them.
    """

    channels: int  # MN extent of this operand's tile: K-out channels for dY, C-in for X
    reduction: int  # reduction pixels per stage, i.e. the CTA K tile in elements
    bits: int  # element width of the operand
    minimum_tma_box_bytes: int = _MIN_SWIZZLE_BYTES  # floor for the swizzle width (X keeps 128 for alignment)

    @property
    def tma_box_bytes(self) -> int:
        """Swizzle width of one TMA box: 128 B, unless a channel row is narrower."""
        # TF32 must use the 32-byte-atomic 128 B swizzle; the 16-byte-atomic
        # modes take the largest power of two that still spans a whole channel
        # row, clamped to [minimum_tma_box_bytes, 128].
        if self.bits == _TF32_BITS:
            return _MAX_SWIZZLE_BYTES
        channel_row_bytes = self.channels * self.bits // _BITS_PER_BYTE
        floor_pow2_bytes = 1 << (channel_row_bytes.bit_length() - 1)
        return min(_MAX_SWIZZLE_BYTES, max(self.minimum_tma_box_bytes, floor_pow2_bytes))

    @property
    def tma_box_channels(self) -> int:
        """Channels in one TMA box: the width of one swizzled TMA-box row."""
        return self.tma_box_bytes * _BITS_PER_BYTE // self.bits

    @property
    def tma_boxes(self) -> int:
        """TMA boxes per stage; the last one is partially filled when channels do not divide."""
        return (self.channels + self.tma_box_channels - 1) // self.tma_box_channels

    @property
    def tma_box_size(self) -> int:
        """Bytes of one TMA box: ``tma_box_bytes`` per pixel row, ``reduction`` rows."""
        return self.tma_box_bytes * self.reduction

    @property
    def stage_bytes(self) -> int:
        """Bytes of one pipeline stage (a smem slot holding every TMA box)."""
        return self.tma_boxes * self.tma_box_size

    @property
    def swizzle(self) -> cuda.TensorMapSwizzle:
        """Tensor-map swizzle of the load box; matches ``descriptor_layout``."""
        if self.bits == _TF32_BITS:
            return cuda.TensorMapSwizzle.s128b_atom_32b
        return {
            32: cuda.TensorMapSwizzle.s32b,
            64: cuda.TensorMapSwizzle.s64b,
            128: cuda.TensorMapSwizzle.s128b,
        }[self.tma_box_bytes]

    @property
    def descriptor_layout(self) -> prims.Tcgen05SmemSwizzle:
        """The same swizzle as a tcgen05 SMEM-descriptor field."""
        # TensorMapSwizzle is also an IntEnum, with different numeric values.
        # Convert explicitly before build() can take its plain-integer branch.
        return self.swizzle.to(prims.Tcgen05SmemSwizzle)

    @property
    def stride_bytes(self) -> int:
        """Descriptor stride between neighbouring MN atom groups along the reduction axis."""
        # TF32's 32B-atomic MN atom spans four pixels; the others span eight.
        atom_pixels = 4 if self.bits == _TF32_BITS else 8
        return atom_pixels * self.tma_box_bytes

    @property
    def instruction_bytes(self) -> int:
        """Bytes one tcgen05 MMA issue consumes: a 32-byte K step over all TMA-box channels."""
        return (_MMA_INST_K_BYTES * _BITS_PER_BYTE // self.bits) * self.tma_box_bytes


def _select_acc_stages(cta_tile_n: int) -> int:
    return min(_MAX_ACC_STAGES, _TMEM_COLS // (_NUM_ACC_CHAINS * cta_tile_n))


def check_config(config: ConvTileConfig, output_dtype: Type[Numeric]) -> None:
    """The TMEM bands must contain complete output-store subtiles."""
    # The reducer loop issues cta_tile_k_bytes // 32 MMAs per stage, so only the
    # 64- and 128-byte stages are implemented (the catalog also lists 32/96).
    if config.cta_tile_k_bytes not in (64, 128):
        raise ValueError("frost_conv_wgrad requires 64- or 128-byte reduction stages")
    # The epilogue walks the accumulator in (warp_n_count x subtile_n) N bands;
    # a partial band would need a masked TMEM read and a masked store.
    epi_datapath = _select_epi_datapath(config.cta_tile_m, config.use_2cta_instrs, output_dtype)
    n_band_width = epi_datapath.warp_n_count * epi_datapath.subtile_n
    if config.cta_tile_n % n_band_width:
        raise ValueError(f"frost_conv_wgrad: TILE_N must be divisible by {n_band_width} for this epilogue datapath")


@cute.kernel
def _kernel(
    tile_sched_params: scheduler.PersistentTileSchedulerParams,
    reduction_tiles: cutlass.Int32,
    zpq_divisors: tuple[cute.FastDivmodDivisorV2, cute.FastDivmodDivisorV2, cute.FastDivmodDivisorV2],
    rs_divisors: tuple[cute.FastDivmodDivisorV2, cute.FastDivmodDivisorV2],
    dy_map: cutlass.GridConstant[cuda.TensorMap],
    x_map: cutlass.GridConstant[cuda.TensorMap],
    dw_map: cutlass.GridConstant[cuda.TensorMap],
    release_arg: cutlass.Int32,
    config: cutlass.Constexpr[ConvTileConfig],
    ab_dtype: cutlass.Constexpr[Type[Numeric]],
    c_dtype: cutlass.Constexpr[Type[Numeric]],
    a_layout: cutlass.Constexpr[OperandLayout],
    b_layout: cutlass.Constexpr[OperandLayout],
    epi_datapath: cutlass.Constexpr[_EpiDatapath],
    epilogue_op: cutlass.Constexpr[Callable[[Any], Any]],
    padding: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    stride: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    dilation: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
) -> None:
    # Uniform warp id: the role dispatch below branches on a value that is the
    # same for every lane of the warp, so each warp runs exactly one role.
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    thread_idx, _, _ = cute.arch.thread_idx()
    # Release-zero argument for the TMEM slot release in the epilogue (Rule 11).
    # The host always passes the literal 0 and this kernel never tests it: its
    # single use is inside consume_then_arrive*, where it masks a fold of every
    # register the warp loaded from the accumulator.
    release_zero_mask = release_zero(release_arg)

    # ---- Cluster shape -----------------------------------------------------
    # The launch is one cluster per tile, shaped (cta_group, 1, 1): a 2-CTA
    # tcgen05 group with its leader at even rank. Only the leader issues the MMA
    # chain; both ranks load their own TMA boxes and both drain their own
    # accumulator half.
    cta_group = prims.CTAGroup.CTA_2 if cutlass.const_expr(config.use_2cta_instrs) else prims.CTAGroup.CTA_1
    cta_rank_in_group = cute.arch.block_idx()[0] % config.cta_group
    is_leader_cta = cta_rank_in_group == 0
    commit_multicast_mask = 0b11 if cutlass.const_expr(config.use_2cta_instrs) else None  # commit arrives on both CTAs of the group

    # ---- Per-tile geometry -------------------------------------------------
    cta_tile_m = config.cta_tile_m  # K-out channel rows of one MMA tile
    cta_tile_n = config.cta_tile_n  # C-in channel columns of one MMA tile
    reduction_per_stage = a_layout.reduction  # reduction pixels per stage
    num_acc_stages = _select_acc_stages(cta_tile_n)  # inter-tile TMEM accumulator stages (2 when cta_tile_n <= 128, else 1)
    num_acc_slots = num_acc_stages * _NUM_ACC_CHAINS  # total TMEM chain slots across accumulator stages
    num_epi_subtiles = cta_tile_n // (epi_datapath.warp_n_count * epi_datapath.subtile_n)  # N subtiles one epilogue warp drains per accumulation slot
    c_stage_elements = epi_datapath.c_tile_rows * epi_datapath.subtile_n  # elements in one staging fragment

    # ---- Shared memory inventory (one flat allocation; the launch passes
    # smem_merge_branch_allocs so the warp-role branches share one buffer) ----
    #   tmem_dealloc_mbar[1]  init 32: the peer CTA's warp 0 arrives here with its 32
    #                         lanes, so neither half frees TMEM before the pair ends.
    #   ab_full_mbar[s]       init 1: the producer's arrive_expect_tx for stage s;
    #                         only the MMA leader waits on it.
    #   ab_empty_mbar[s]      init 1: tcgen05.commit once the MMAs that read stage s
    #                         have completed; each CTA's own producer waits on it.
    #   acc_full_mbar[a]      init 1: accumulator slot a holds one chain's share of
    #                         a work item's reduction; committed with multicast so
    #                         both CTAs' epilogues see it.
    #   acc_empty_mbar[a]     init 128*num_epi_subtiles*cta_group: one arrive per epilogue
    #                         lane per fragment, from BOTH CTAs, all on the leader's copy.
    #   tmem_ptr_slot[1]      the Int32 TMEM token written by tcgen05.alloc.
    tmem_dealloc_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    ab_full_mbar = cutlass.Array(cutlass.Int64, _NUM_AB_STAGES, space=cutlass.AddressSpace.smem)
    ab_empty_mbar = cutlass.Array(cutlass.Int64, _NUM_AB_STAGES, space=cutlass.AddressSpace.smem)
    acc_full_mbar = cutlass.Array(cutlass.Int64, num_acc_slots, space=cutlass.AddressSpace.smem)
    acc_empty_mbar = cutlass.Array(cutlass.Int64, num_acc_slots, space=cutlass.AddressSpace.smem)
    tmem_ptr_slot = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # Operand rings: 3 stages each, 1024-byte aligned so every TMA-box row starts
    # on a swizzle boundary.
    smem_a = cutlass.Array(cutlass.Int8, _NUM_AB_STAGES * a_layout.stage_bytes, space=cutlass.AddressSpace.smem, alignment=_SMEM_ALIGNMENT)
    smem_b = cutlass.Array(cutlass.Int8, _NUM_AB_STAGES * b_layout.stage_bytes, space=cutlass.AddressSpace.smem, alignment=_SMEM_ALIGNMENT)
    # Epilogue staging: two fragments, alternating, so a bulk store can still be
    # reading one while the next fragment is written.
    smem_c = cutlass.Array(c_dtype, _NUM_C_STAGES * c_stage_elements, space=cutlass.AddressSpace.smem, alignment=_SMEM_ALIGNMENT)
    if warp_idx == _ALLOCATOR_WARP_ID:
        if prims.elect_sync():
            # mbarrier_init is elect-safe: one lane per CTA initializes.
            prims.mbarrier_init(tmem_dealloc_mbar, _WARP_SIZE)
            for stage_idx in cutlass.range_constexpr(_NUM_AB_STAGES):
                prims.mbarrier_init(ab_full_mbar.subview(stage_idx), 1)
                prims.mbarrier_init(ab_empty_mbar.subview(stage_idx), 1)
            for acc_idx in cutlass.range_constexpr(num_acc_slots):
                prims.mbarrier_init(acc_full_mbar.subview(acc_idx), 1)
                # Counted for the whole CTA group: the peer's epilogue credits
                # the leader's barrier through mapa(.., 0) below.
                prims.mbarrier_init(acc_empty_mbar.subview(acc_idx), _EPI_THREADS * num_epi_subtiles * config.cta_group)
    # Barrier init is only CTA-visible until the pair rendezvous, and the first
    # TMA completion / tcgen05 commit can cross to the peer immediately.
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    # Static persistent schedule: work item = (k tile, c tile, filter position).
    tile_sched = scheduler.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
    work_tile = tile_sched.initial_work_tile_info()
    prims.barrier_cluster_wait()

    # ======================== TMA producer warp =============================
    if warp_idx == _TMA_WARP_ID:
        ab_stage_idx = 0  # index into the 3-deep operand ring
        ab_empty_phase = 1  # parity of the empty-barrier phase being awaited; flips when the ring wraps
        z_divisor, p_divisor, q_divisor = zpq_divisors
        r_divisor, s_divisor = rs_divisors
        while work_tile.is_valid_tile:
            # Filter position (t, r, s) from the work item's filter_idx.
            k_tile_idx, c_tile_idx, filter_idx = work_tile.tile_idx
            tr_idx, s_idx = divmod(filter_idx, s_divisor)
            t_idx, r_idx = divmod(tr_idx, r_divisor)
            for reduction_step in cutlass.range(reduction_tiles, unroll=1):
                mbar_full = ab_full_mbar.subview(ab_stage_idx)
                # Producer acquire: wait until the MMA released this stage.
                # try_wait_parity is a single non-blocking attempt, so it must
                # be retried in a loop.
                while not prims.mbarrier_try_wait_parity(ab_empty_mbar.subview(ab_stage_idx), ab_empty_phase, time_limit=_MBAR_WAIT_TIMEOUT):
                    pass
                # First reduction pixel of this stage, decomposed (n, z, p, q)
                # with q fastest -- the same order the A descriptor flattens.
                pixel_idx = reduction_step * reduction_per_stage
                nzp_idx, q_idx = divmod(pixel_idx, q_divisor)
                nz_idx, p_idx = divmod(nzp_idx, p_divisor)
                n_idx, z_idx = divmod(nz_idx, z_divisor)
                if prims.elect_sync():
                    # The group leader counts the transaction bytes of BOTH CTAs:
                    # with group="cta_2" the peer's complete_tx is routed back to
                    # the group leader's mbar (bit-24 collapse).
                    if is_leader_cta:
                        prims.mbarrier_arrive_expect_tx(mbar_full, (a_layout.stage_bytes + b_layout.stage_bytes) * config.cta_group)
                    # A (dY): one copy per swizzle TMA box, box = (reduction
                    # pixels, tma_box_channels channels) at (k row, pixel).
                    # Rank-independent because k_tile_idx already distinguishes the two
                    # CTAs' K-row bands.
                    for tma_box_idx in cutlass.range_constexpr(a_layout.tma_boxes):
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            smem_a.subview(ab_stage_idx * a_layout.stage_bytes + tma_box_idx * a_layout.tma_box_size),
                            dy_map.get_ptr(),
                            (k_tile_idx * cta_tile_m + tma_box_idx * a_layout.tma_box_channels, pixel_idx),
                            mbar_full,
                            [],
                            group=cta_group,
                        )
                    # B (X): im2col box = (tma_box_channels channels, reduction
                    # pixels). The descriptor's corners already encode padding
                    # and the output extent. The coordinate list is innermost
                    # input dimension first -- (channel, w, h, d, n) -- so each
                    # entry is the output pixel mapped back to the input, and the
                    # trailing list is the filter offset (s, r, t) the TMA adds
                    # per column. The channel base carries the CTA rank: each CTA
                    # of a 2-CTA group holds its own N half of the tile.
                    for tma_box_idx in cutlass.range_constexpr(b_layout.tma_boxes):
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            smem_b.subview(ab_stage_idx * b_layout.stage_bytes + tma_box_idx * b_layout.tma_box_size),
                            x_map.get_ptr(),
                            (
                                c_tile_idx * cta_tile_n + cta_rank_in_group * (cta_tile_n // config.cta_group) + tma_box_idx * b_layout.tma_box_channels,
                                q_idx * stride[2] - padding[2],
                                p_idx * stride[1] - padding[1],
                                z_idx * stride[0] - padding[0],
                                n_idx,
                            ),
                            mbar_full,
                            [s_idx * dilation[2], r_idx * dilation[1], t_idx * dilation[0]],
                            mode=prims.TMALoadMode.IM2COL,
                            group=cta_group,
                        )
                ab_stage_idx += 1
                if ab_stage_idx == _NUM_AB_STAGES:
                    ab_stage_idx = 0
                    ab_empty_phase ^= 1
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    # =========================== MMA warp ==================================
    elif warp_idx == _MMA_WARP_ID:
        # Waits for the TMEM token the allocator warp (epilogue warp 0) publishes.
        # Named barrier 1 is a 5-warp subset barrier (warps 0-4), never an
        # all-CTA one, so it uses the non-aligned form.
        prims.barrier_cta_sync(_TMEM_ALLOC_BAR_ID, thread_count=_TMEM_ALLOC_SYNC_THREADS)
        tmem_base_ptr = prims.make_tmem_ptr(tmem_ptr_slot.load(), cutlass.Float32)
        # SMEM descriptors of the operand rings. leading_byte_offset is one TMA box
        # (where the next MN group starts); stride_byte_offset is the MN atom
        # stride. The layout must be the swizzle the host TMA box used: a
        # mismatch makes the MMA read a transposed stage.
        desc_a = prims.Tcgen05SmemDesc.build(
            smem_a,
            leading_byte_offset=a_layout.tma_box_size,
            stride_byte_offset=a_layout.stride_bytes,
            layout=a_layout.descriptor_layout,
        )
        desc_b = prims.Tcgen05SmemDesc.build(
            smem_b,
            leading_byte_offset=b_layout.tma_box_size,
            stride_byte_offset=b_layout.stride_bytes,
            layout=b_layout.descriptor_layout,
        )
        # a_major/b_major = 1: both operands are MN-major (channels contiguous),
        # the layout OperandLayout built. m_dim spans the whole 2-CTA group.
        mma_instr_desc = prims.Tcgen05InstrDesc.build(
            a_dtype=ab_dtype,
            b_dtype=ab_dtype,
            c_dtype=cutlass.Float32,
            a_major=1,  # M-major
            b_major=1,  # N-major
            m_dim=cta_tile_m * config.cta_group,
            n_dim=cta_tile_n,
        )
        if is_leader_cta:
            ab_stage_idx, ab_full_phase, acc_stage_idx, acc_empty_phase = 0, 0, 0, 1
            while work_tile.is_valid_tile:
                acc_base_slot = acc_stage_idx * _NUM_ACC_CHAINS
                # Acquire every chain's slot for this accumulator stage; the
                # initial phase makes the first wave pass without blocking.
                for chain_idx in cutlass.range_constexpr(_NUM_ACC_CHAINS):
                    while not prims.mbarrier_try_wait_parity(
                        acc_empty_mbar.subview(acc_base_slot + chain_idx),
                        acc_empty_phase,
                        time_limit=_MBAR_WAIT_TIMEOUT,
                    ):
                        pass
                # One chain per accumulator slot in the active stage. Consecutive
                # MMA issues feed alternate chains, so the first stage seeds every
                # chain and the two partial sums cover disjoint halves of the reduction.
                acc_tmem_ptrs = [
                    cutlass.Array(
                        tmem_base_ptr.data_ptr() + (acc_base_slot + chain_idx) * cta_tile_n,
                        dtype=cutlass.Int32,
                        addrspace=cutlass.AddressSpace.tmem,
                    )
                    for chain_idx in range(_NUM_ACC_CHAINS)
                ]
                for reduction_step in cutlass.range(reduction_tiles, unroll=1):
                    while not prims.mbarrier_try_wait_parity(ab_full_mbar.subview(ab_stage_idx), ab_full_phase, time_limit=_MBAR_WAIT_TIMEOUT):
                        pass
                    # A stage holds cta_tile_k_bytes of reduction but one MMA
                    # consumes 32 bytes, so it is issued in that many steps, each
                    # advancing its descriptors by instruction_bytes. Issue i of
                    # the reduction feeds chain i % _NUM_ACC_CHAINS.
                    if prims.elect_sync():
                        for mma_k_idx in cutlass.range_constexpr(config.cta_tile_k_bytes // _MMA_INST_K_BYTES):
                            prims.tcgen05_mma(
                                _mma_kind_for_dtype(ab_dtype),
                                cta_group,
                                acc_tmem_ptrs[mma_k_idx % _NUM_ACC_CHAINS],
                                desc_a
                                + ab_stage_idx * (a_layout.stage_bytes // _SMEM_DESC_UNIT_BYTES)
                                + mma_k_idx * (a_layout.instruction_bytes // _SMEM_DESC_UNIT_BYTES),
                                desc_b
                                + ab_stage_idx * (b_layout.stage_bytes // _SMEM_DESC_UNIT_BYTES)
                                + mma_k_idx * (b_layout.instruction_bytes // _SMEM_DESC_UNIT_BYTES),
                                mma_instr_desc,
                                # enable_input_d: accumulate every issue except the
                                # first issue of each chain, which seeds it. The
                                # first stage therefore seeds one chain per issue.
                                (reduction_step > 0) | (mma_k_idx >= _NUM_ACC_CHAINS),
                            )
                        # Release the stage once the tracked MMAs have completed
                        # (they read A and B from it). Multicast so the peer's
                        # producer is released too.
                        prims.tcgen05_commit(ab_empty_mbar.subview(ab_stage_idx), group=cta_group, multicast_mask=commit_multicast_mask)
                    ab_stage_idx += 1
                    if ab_stage_idx == _NUM_AB_STAGES:
                        ab_stage_idx = 0
                        ab_full_phase ^= 1
                if prims.elect_sync():
                    for chain_idx in cutlass.range_constexpr(_NUM_ACC_CHAINS):
                        # This chain's share of the reduction is complete: hand it
                        # to both CTAs' epilogues.
                        prims.tcgen05_commit(
                            acc_full_mbar.subview(acc_base_slot + chain_idx),
                            group=cta_group,
                            multicast_mask=commit_multicast_mask,
                        )
                acc_stage_idx += 1
                if acc_stage_idx == num_acc_stages:
                    acc_stage_idx = 0
                    acc_empty_phase ^= 1
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

    # ========================= Epilogue warps ==============================
    elif warp_idx < _NUM_EPI_WARPS:
        # This warp's position in the (M sub-band, N band) grid. The TMEM rows
        # are selected by the warp's own TMEM sub-partition; the split decides
        # which staging slot and which N band the result is written to.
        warp_m_idx = warp_idx % epi_datapath.warp_m_count
        warp_n_idx = warp_idx // epi_datapath.warp_m_count
        vec_size = _TENSOR_ALIGN_BITS // c_dtype.width  # one 128-bit store per lane step
        # Thread -> staging placement for the 16-row fragment datapaths: thread
        # (warp w, low lane L = thread_idx % 16, half H = (thread_idx // 16) % 2) owns staging
        # row (w * 16 + L) at channel half H. Deriving it from thread_idx keeps the
        # byte layout identical for every C dtype because the step scales with
        # subtile_n. The general (32x32b) paths place row thread_idx instead.
        if cutlass.const_expr(epi_datapath.use_stmatrix or epi_datapath.use_fp8_simt):
            smem_thread_offset = (
                (thread_idx % _STMATRIX_ROW_LANES) * epi_datapath.subtile_n
                + ((thread_idx // _STMATRIX_ROW_LANES) % _STMATRIX_CHANNEL_HALVES) * (epi_datapath.subtile_n // _STMATRIX_CHANNEL_HALVES)
                + (thread_idx // _WARP_SIZE) * _STMATRIX_ROW_LANES * epi_datapath.subtile_n
            )
        if warp_idx == _ALLOCATOR_WARP_ID:
            # tcgen05_alloc is warp-collective and not elect-safe: it must stay
            # outside elect_sync and outside any rank-divergent branch.
            prims.tcgen05_alloc(tmem_ptr_slot, _TMEM_COLS, group=cta_group)
        prims.barrier_cta_sync(_TMEM_ALLOC_BAR_ID, thread_count=_TMEM_ALLOC_SYNC_THREADS)
        tmem_raw_addr = tmem_ptr_slot.load()
        acc_stage_idx, acc_full_phase, c_stage_idx = 0, 0, 0
        r_divisor, s_divisor = rs_divisors
        while work_tile.is_valid_tile:
            # Filter position of this work item; the dW store coordinates are
            # built from it directly.
            k_tile_idx, c_tile_idx, filter_idx = work_tile.tile_idx
            tr_idx, s_idx = divmod(filter_idx, s_divisor)
            t_idx, r_idx = divmod(tr_idx, r_divisor)
            acc_base_slot = acc_stage_idx * _NUM_ACC_CHAINS
            # Wait for every chain's commit: the reduction is complete and every
            # fragment below may be read.
            for chain_idx in cutlass.range_constexpr(_NUM_ACC_CHAINS):
                while not prims.mbarrier_try_wait_parity(
                    acc_full_mbar.subview(acc_base_slot + chain_idx),
                    acc_full_phase,
                    time_limit=_MBAR_WAIT_TIMEOUT,
                ):
                    pass
            # Drain the accumulator one N subtile at a time so the staging stays
            # small and the bulk store of one subtile overlaps the next read.
            for subtile_idx in cutlass.range(num_epi_subtiles, unroll=1):
                # Every chain holds a partial reduction of the same output tile;
                # fold them here in IEEE FP32 before the elementwise epilogue.
                chain_regs = []
                acc_regs = None
                for chain_idx in cutlass.range_constexpr(_NUM_ACC_CHAINS):
                    # TMEM view of this warp's 32-lane sub-partition at the chain's
                    # column base plus this subtile's N band.
                    subtile_tmem_ptr = prims.make_tmem_ptr_from_warp_row_col(
                        tmem_raw_addr,
                        warp_m_idx,
                        (tmem_raw_addr & 0xFFFF) + (acc_base_slot + chain_idx) * cta_tile_n + subtile_idx * epi_datapath.subtile_n,
                        cutlass.Float32,
                    )
                    # 16x32bx2 (fp8) is the only shape that takes a column offset.
                    regs = prims.tcgen05_ld(
                        epi_datapath.ldtm_shape,
                        subtile_tmem_ptr,
                        num=epi_datapath.ldtm_repx,
                        offset=epi_datapath.ldtm_offset if cutlass.const_expr(epi_datapath.use_fp8_simt) else None,
                    )
                    chain_regs.append(regs)
                    acc_regs = regs if acc_regs is None else acc_regs + regs
                # Release each chain's TMEM slot immediately after every register
                # loaded from it has been read (Rule 11), before SMEM staging and
                # TMA stores so the MMA warp can reuse the slot early.
                for chain_idx in cutlass.range_constexpr(_NUM_ACC_CHAINS):
                    slot_idx = acc_base_slot + chain_idx
                    if cutlass.const_expr(config.use_2cta_instrs):
                        consume_then_arrive_on_peer(acc_empty_mbar.subview(slot_idx), cutlass.Int32(0), chain_regs[chain_idx], release_zero_mask)
                    else:
                        consume_then_arrive(acc_empty_mbar.subview(slot_idx), chain_regs[chain_idx], release_zero_mask)
                # Elementwise epilogue on the FP32 accumulator before the cast.
                epi_regs = epilogue_op(acc_regs)
                if warp_m_idx == 0:
                    # Ensure the previous bulk read from this staging slot has
                    # completed before overwriting it.
                    prims.cp_async_bulk_wait_group(_NUM_C_STAGES - 1, read=True)
                prims.barrier_cta_sync(_EPI_SYNC_BAR_ID, thread_count=_EPI_THREADS)
                # This fragment's staging slot: two of them, swapped per subtile.
                smem_c_stage = smem_c.subview(c_stage_idx * c_stage_elements)
                if cutlass.const_expr(epi_datapath.use_stmatrix):
                    # LDTM delivers two interleaved rows per thread, which is the
                    # 8x8 fragment layout stmatrix consumes: one call writes 16 B
                    # (4 i32) per thread. store_swizzled cannot express that
                    # fragment shape, so the SW32 destination address is
                    # XOR-permuted here (bit 4 ^= bit 7) instead.
                    packed = cutlass.Array(c_dtype, vec_size, alignment=_STMATRIX_ALIGN_BYTES)
                    packed[0:vec_size] = epi_regs[0:vec_size].to(c_dtype)
                    addr = cutlass.Int64(llvm.ptrtoint(cutlass.Int64.mlir_type, smem_c_stage.subview(smem_thread_offset).data_ptr()))
                    addr = addr ^ ((addr & cutlass.Int64(_SWIZZLE_32B_MASK)) >> cutlass.Int64(_SWIZZLE_32B_SHIFT))
                    ptr = llvm.inttoptr(llvm.PointerType.get(cutlass.AddressSpace.smem), addr.ir_value())
                    words = cutlass.Array(packed.ir_value(), dtype=cutlass.Int32, shape=(_STMATRIX_WORDS,), addrspace=packed.space)[0:_STMATRIX_WORDS]
                    prims.stmatrix(ptr, [words[0], words[1], words[2], words[3]], prims.MMALayout.ROW, shape="m8n8")
                else:
                    # 32x32b: thread thread_idx holds 32 contiguous channels of one
                    # staging row. Cast, then write in 128-bit swizzled chunks
                    # with the same swizzle the host dW descriptor uses.
                    base_offset = smem_thread_offset if cutlass.const_expr(epi_datapath.use_fp8_simt) else thread_idx * epi_datapath.subtile_n
                    for vec_idx in cutlass.range_constexpr(_LDTM_CONTIGUOUS_CHANNELS // vec_size):
                        smem_c_stage.subview(base_offset + vec_idx * vec_size).data_ptr().store_swizzled(
                            epi_regs[vec_idx * vec_size : vec_idx * vec_size + vec_size].to(c_dtype),
                            alignment=_STORE_SWIZZLED_ALIGN_BYTES,
                            swizzle=_c_smem_swizzle(c_dtype, epi_datapath.subtile_n),
                        )
                # Make the generic-proxy stores visible to the async proxy (TMA).
                cute.arch.fence_view_async_shared()
                prims.barrier_cta_sync(_EPI_SYNC_BAR_ID, thread_count=_EPI_THREADS)
                if warp_m_idx == 0:
                    # One warp per N band commits that band: the staging fragment
                    # holds cta_tile_m rows per band, and this CTA's copy of the
                    # band is complete once all four epilogue warps have written.
                    if prims.elect_sync():
                        prims.cp_async_bulk_tensor_global_shared_cta(
                            dw_map.get_ptr(),
                            smem_c_stage.subview(warp_n_idx * cta_tile_m * epi_datapath.subtile_n),
                            (
                                c_tile_idx * cta_tile_n + warp_n_idx * (cta_tile_n // epi_datapath.warp_n_count) + subtile_idx * epi_datapath.subtile_n,
                                s_idx,
                                r_idx,
                                t_idx,
                                k_tile_idx * cta_tile_m,
                            ),
                        )
                    # Leave the just-issued store in flight; the wait at the
                    # top of the next staging write guarantees the previous
                    # fragment's SMEM reads are done before reuse.
                    prims.cp_async_bulk_commit_group()
                c_stage_idx ^= 1  # swap staging fragments for the next subtile
            acc_stage_idx += 1
            if acc_stage_idx == num_acc_stages:
                acc_stage_idx = 0
                acc_full_phase ^= 1
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        if warp_m_idx == 0:
            # Drain the last bulk read before this CTA's SMEM is released.
            prims.cp_async_bulk_wait_group(0, read=True)
        prims.barrier_cta_sync(_EPI_SYNC_BAR_ID, thread_count=_EPI_THREADS)
        if warp_idx == _ALLOCATOR_WARP_ID:
            # Every allocation must be released, and a CTA that relinquishes the
            # permit may not allocate again.
            prims.tcgen05_relinquish_alloc_permit(group=cta_group)
            if cutlass.const_expr(config.use_2cta_instrs):
                # The dealloc is pair-collective: each CTA arrives on its peer's
                # barrier (32 lanes = the init count) and waits for its own, so
                # neither half frees TMEM while the other still reads it.
                prims.mbarrier_arrive(prims.mapa(tmem_dealloc_mbar, cta_rank_in_group ^ 1), count=1, scope=prims.MemScope.CLUSTER)
                while not prims.mbarrier_try_wait_parity(tmem_dealloc_mbar, 0, time_limit=_MBAR_WAIT_TIMEOUT):
                    pass
            prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw_addr, cutlass.Float32), _TMEM_COLS, group=cta_group)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


class _WgradKernel:
    """Host side: descriptor construction, work-item space and the launch."""

    def __init__(
        self,
        config: ConvTileConfig,
        ab_dtype: Type[Numeric],
        c_dtype: Type[Numeric],
        epilogue: str,
        attrs: tuple[tuple[str, float], ...],
    ) -> None:
        self.config = config
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.epilogue = get_epilogue_op(epilogue, attrs)
        reduction_per_stage = config.cta_tile_k(ab_dtype.width)  # reduction pixels per stage
        # A (dY) tiles the K-out channels; B (X) tiles the C-in channels, split
        # over the CTA group.
        self.a_layout = OperandLayout(config.cta_tile_m, reduction_per_stage, ab_dtype.width)
        # Keep X TMA boxes 128-byte aligned across tile widths and CTA groups: the
        # im2col descriptor takes its channels_per_pixel from this TMA box, and
        # every stage then starts on a 128 B (whole swizzle) boundary.
        self.b_layout = OperandLayout(
            config.cta_tile_n // config.cta_group,
            reduction_per_stage,
            ab_dtype.width,
            minimum_tma_box_bytes=_MAX_SWIZZLE_BYTES,
        )
        # One descriptor for the epilogue datapath: staging width, LDTM form,
        # store kind and the epilogue warp split. The device SMEM sizing and the
        # host dW descriptor both read it, so they cannot drift.
        self.epi_datapath = _select_epi_datapath(config.cta_tile_m, config.use_2cta_instrs, c_dtype)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        dy: cute.Tensor,
        dw: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda_driver.CUstream,
        padding: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        post_padding: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        stride: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        dilation: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    ) -> None:
        n, z, p, q, k = dy.shape

        _, t, r, s, c = dw.shape
        reduction_length = n * z * p * q  # reduction extent L
        a_layout, b_layout, config, epi_datapath = self.a_layout, self.b_layout, self.config, self.epi_datapath
        # A: flatten the output-pixel axes into one row-major axis of L rows of K
        # channels, so the tiled TMA box and the reduction loop share one index.
        dy_flat = cute.make_tensor(dy.iterator, cute.make_layout((reduction_length, k), stride=(k, 1)))
        dy_map = cuda.create_tensor_map_tiled_from_view(
            dy_flat,
            box_dims=(a_layout.reduction, a_layout.tma_box_channels),
            stride_order=(1, 0),
            swizzle=a_layout.swizzle,
        )
        # B: native im2col tensor map. global_dims/global_strides are innermost
        # dimension first (c, w, h, d, n) and strides are in 16-byte units; the
        # corners bound the padded input over which the TMA walks with the
        # output traversal strides and the per-column filter offset.
        x_map = cuda.create_tensor_map_im2col(
            global_address=x.iterator.toint(),
            dtype=self.ab_dtype,
            global_dims=(x.shape[4], x.shape[3], x.shape[2], x.shape[1], x.shape[0]),
            global_strides=tuple(x.stride[i] * self.ab_dtype.width // _TENSOR_ALIGN_BITS for i in (3, 2, 1, 0)),
            lower_corner=[-padding[2], -padding[1], -padding[0]],
            upper_corner=[
                post_padding[2] - (s - 1) * dilation[2],
                post_padding[1] - (r - 1) * dilation[1],
                post_padding[0] - (t - 1) * dilation[0],
            ],
            channels_per_pixel=b_layout.tma_box_channels,
            pixels_per_column=b_layout.reduction,
            traversal_strides=(1, stride[2], stride[1], stride[0], 1),
            swizzle=b_layout.swizzle,
        )
        # C: one box is (cta_tile_m k rows x subtile_n c channels) at a single
        # filter position; stride_order makes the channel axis innermost. The
        # swizzle comes from the same helper the device store uses.
        dw_map = cuda.create_tensor_map_tiled_from_view(
            dw,
            box_dims=(config.cta_tile_m, 1, 1, 1, epi_datapath.subtile_n),
            stride_order=(4, 3, 2, 1, 0),
            swizzle=_to_tensor_map_swizzle(_c_smem_swizzle(self.c_dtype, epi_datapath.subtile_n)),
        )
        # Work item space: (k bands, c tiles, filter positions). The k extent is
        # a multiple of the CTA group so a 2-CTA cluster always gets two
        # adjacent K-row bands; the cluster shape only spans M.
        tile_sched_params = scheduler.PersistentTileSchedulerParams(
            (
                cute.ceil_div(k, config.cta_tile_m * config.cta_group) * config.cta_group,
                cute.ceil_div(c, config.cta_tile_n),
                t * r * s,
            ),
            (config.cta_group, 1, 1),
        )
        grid = scheduler.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)
        _kernel(
            tile_sched_params,
            cutlass.Int32(cute.ceil_div(reduction_length, a_layout.reduction)),  # reduction stages, last one partial
            tuple(cute.fast_divmod_create_divisor_v2(cutlass.Int32(v)) for v in (z, p, q)),
            tuple(cute.fast_divmod_create_divisor_v2(cutlass.Int32(v)) for v in (r, s)),
            dy_map,
            x_map,
            dw_map,
            cutlass.Int32(0),  # release-zero argument
            config,
            self.ab_dtype,
            self.c_dtype,
            a_layout,
            b_layout,
            epi_datapath,
            self.epilogue,
            padding,
            stride,
            dilation,
            # 6 warps: 4 epilogue + MMA + producer; one flat SMEM buffer shared
            # by the warp-role branches.
        ).launch(
            grid=grid,
            block=(_NUM_WARPS * _WARP_SIZE, 1, 1),
            cluster=(config.cta_group, 1, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )


def compile(
    ab_dtype: Type[Numeric] = cutlass.BFloat16,
    c_dtype: Type[Numeric] = cutlass.BFloat16,
    config: ConvTileConfig = DEFAULT_CONFIG,
    epilogue: str = "identity",
    epilogue_attrs: tuple[tuple[str, float], ...] = (),
) -> Any:
    check_config(config, c_dtype)
    device, target = build_target()
    return _compile_cached(device, target, ab_dtype, c_dtype, config, epilogue, epilogue_attrs)


@lru_cache(maxsize=None)
def _compile_cached(
    device: int,
    target: str,
    ab_dtype: Type[Numeric],
    c_dtype: Type[Numeric],
    config: ConvTileConfig,
    epilogue: str,
    epilogue_attrs: tuple[tuple[str, float], ...],
) -> Any:
    # Shapes and geometry never enter the artifact identity.
    # Every extent below is symbolic, so one artifact serves any conv of this
    # dtype/tile/epilogue and the runtime geometry arrives as kernel arguments.
    # Only the divisibility the engine guarantees is baked in.
    with device_context(device):
        n = cute.sym_int()
        c = cute.sym_int(divisibility=max(_TENSOR_ALIGN_BITS // ab_dtype.width, _TENSOR_ALIGN_BITS // c_dtype.width))
        k = cute.sym_int(divisibility=_TENSOR_ALIGN_BITS // ab_dtype.width)
        shapes = [
            (n, cute.sym_int(), cute.sym_int(), cute.sym_int(), c),  # x
            (n, cute.sym_int(), cute.sym_int(), cute.sym_int(), k),  # dy
            (k, cute.sym_int(), cute.sym_int(), cute.sym_int(), c),  # dw
        ]
        args = [
            make_fake_compact_tensor(dtype, shape, stride_order=(4, 3, 2, 1, 0), assumed_align=_TENSOR_ALIGN_BYTES)
            for dtype, shape in zip((ab_dtype, ab_dtype, c_dtype), shapes)
        ]
        # Neutral padding / post-padding / stride / dilation for the trace.
        neutral_coord = (cutlass.Int32(1),) * 3
        geometry = [neutral_coord, neutral_coord, neutral_coord, neutral_coord]
        return cute.compile(
            _WgradKernel(config, ab_dtype, c_dtype, epilogue, epilogue_attrs),
            *args,
            utils.HardwareInfo().get_max_active_clusters(config.cta_group),  # persistent grid bound
            make_fake_stream(),
            *geometry,
            options=f"--enable-tvm-ffi --gpu-arch {target}",
        )
