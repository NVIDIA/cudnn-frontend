# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# CTM-style FP16/BF16 implicit-GEMM fprop convolution kernel for Blackwell B200.
#
# Maps a 3D convolution onto an implicit GEMM: M = N*Z*P*Q, N_gemm = K,
# K_gemm = T*R*S*C. Warp-specialized with explicit roles (1 TMA producer warp,
# 1 MMA consumer warp, 4 epilogue warps), mbarrier-based producer/consumer
# synchronization, and a tcgen05.mma + tcgen05.commit handshake for the TMEM
# accumulator.
#
# Data movement:
#   - A (activations) is loaded via im2col TMA:
#     prims.cp_async_bulk_tensor_shared_cluster_global with TMALoadMode.IM2COL,
#     a 5D coord list (c, w, h, d, n), and an im2col-offset list (s, r, t).
#     The host builds the descriptor with cuda.create_tensor_map_im2col_from_view.
#   - B (filter) is loaded via a tiled TMA descriptor
#     (cuda.create_tensor_map_tiled_from_view).
#   - C (output) is stored via im2col TMA store:
#     prims.cp_async_bulk_tensor_global_shared_cta with TMAStoreMode.IM2COL and
#     a 5D coord list (k_off, q, p, z, n). The store op has no im2col-offset
#     operand; output spatial offsets live in the descriptor's zero corners.
#
# The cluster shape is static; tiles are assigned by a persistent scheduler.

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Tuple, Type

import cutlass
from cutlass.experimental import primitives as prims
from cutlass import Numeric, testing
from cutlass._mlir.dialects import llvm as llvm
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from .epilogue import get_epilogue_op
from ..tile_config import ConvTileConfig, _get_config

__all__ = ["compile"]


def _mma_kind_for_dtype(ab_dtype: Type[Numeric]) -> "prims.Tcgen05MMAKind":
    """Select the tcgen05 MMA kind from the A/B operand dtype.

    BF16 shares the ``f16`` kind (the instruction descriptor's a_dtype/b_dtype
    disambiguates it); the per-instruction K-shape is derived from the dtype
    width upstream, so only the MMA kind is chosen here. Accepted dtypes match
    check_supported_dtypes; any other dtype raises.
    """
    if ab_dtype in {cutlass.Float16, cutlass.BFloat16}:
        return prims.Tcgen05MMAKind.F16
    if ab_dtype is cutlass.Float32:
        return prims.Tcgen05MMAKind.TF32
    if ab_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
        return prims.Tcgen05MMAKind.F8F6F4
    raise TypeError(f"Unsupported tcgen05 MMA dtype: {ab_dtype!r}")


def _c_smem_swizzle(c_dtype: Type[Numeric], subtile_n: int) -> "cutlass.Swizzle":
    """Select the epilogue C-SMEM swizzle from the output dtype and subtile width.

    The store packs ``subtile_n`` output channels per row, so the swizzled
    region spans ``subtile_n * c_dtype.width // 8`` bytes. The tcgen05 SMEM
    swizzle family is named by that byte width: 32B/64B/128B. Keying off bytes
    (not a fixed 2-byte assumption) keeps fp32 (4B) and fp8 (1B) outputs
    correct, not just fp16/bf16. The host C TMA descriptor and the device
    ``store_swizzled`` both derive from this one value, so they stay in
    lockstep (a mismatch silently transposes the store; see descriptors.py).
    """
    row_bytes = subtile_n * c_dtype.width // 8
    if row_bytes == 32:
        return cutlass.Swizzle(1, 4, 3)  # s32b
    if row_bytes == 64:
        return cutlass.Swizzle(2, 4, 3)  # s64b
    if row_bytes == 128:
        return cutlass.Swizzle(3, 4, 3)  # s128b
    raise TypeError(f"No SMEM swizzle for {row_bytes}B C row " f"(subtile_n={subtile_n}, dtype={c_dtype!r}); " "expected 32/64/128B.")


def _to_tensor_map_swizzle(swizzle: "cutlass.Swizzle") -> "cuda.TensorMapSwizzle":
    """Convert the semantic SMEM swizzle to the local tensor-map enum.

    ``cutlass.Swizzle.to`` only recognizes the enum class exported by the
    installed ``cutlass.experimental.cuda`` package.  The convolution uses the
    pasted tensor-map module as a unit, so convert explicitly instead of mixing
    enum classes from the two implementations.
    """
    mapping = {
        cutlass.Swizzle(0, 0, 0): cuda.TensorMapSwizzle.none,
        cutlass.Swizzle(1, 4, 3): cuda.TensorMapSwizzle.s32b,
        cutlass.Swizzle(2, 4, 3): cuda.TensorMapSwizzle.s64b,
        cutlass.Swizzle(3, 4, 3): cuda.TensorMapSwizzle.s128b,
    }
    try:
        return mapping[swizzle]
    except KeyError as exc:
        raise ValueError(f"No local tensor-map swizzle preset for {swizzle!r}.") from exc


@dataclass(frozen=True)
class _EpiDatapath:
    """Single source of truth for the epilogue (TMEM->RMEM->SMEM) datapath.

    One place resolves the ``(m_per_cta, use_2cta, c_dtype.width)`` config into
    the concrete LDTM shape, store kind, subtile width, and epilogue-warp split.
    The four decision sites (device SMEM sizing, device epilogue, host attribute
    setup, host C-descriptor) all read this descriptor instead of recomputing the
    same predicate chain, so a stale copy can no longer silently transpose or
    drop the C store.

    store_kind selects the SMEM store path:
      - "stmatrix"  : M==64, 1cta, width>=16  -> stmatrix (16x256b/16x128b LDTM)
      - "fp8_simt"  : M==64, 1cta, width==8   -> SIMT store_swizzled (16x32bx2)
      - "swizzled"  : everything else (cfgA M==128, cfgC M==64+2cta) -> 32x32b
    """

    store_kind: Literal["stmatrix", "fp8_simt", "swizzled"]
    subtile_n: int
    c_tile_rows: int
    ldtm_shape: str
    ldtm_repx: int
    ldtm_offset: int
    warp_m_count: int
    warp_n_count: int

    @property
    def use_stmatrix(self) -> bool:
        return self.store_kind == "stmatrix"

    @property
    def use_fp8_simt(self) -> bool:
        return self.store_kind == "fp8_simt"


def _select_epi_datapath(
    m_per_cta: int,
    use_2cta_instrs: bool,
    c_dtype: Type[Numeric],
    *,
    num_epi_warps: int = 4,
) -> _EpiDatapath:
    """Resolve the epilogue datapath descriptor from the tile/dtype config.

    Every field is a Python constant resolved here, so each caller just reads the
    field it needs. ``num_epi_warps`` is the structural epilogue-warp count (the
    kernel uses warps 0-3).
    """
    is_cfgB = m_per_cta == 64 and not use_2cta_instrs
    if is_cfgB and c_dtype.width >= 16:
        return _EpiDatapath(
            store_kind="stmatrix",
            subtile_n=(128 // c_dtype.width) * 2,
            c_tile_rows=m_per_cta,
            ldtm_shape="16x256b" if c_dtype.width == 16 else "16x128b",
            ldtm_repx=2,
            ldtm_offset=0,
            warp_m_count=num_epi_warps,
            warp_n_count=1,
        )
    if is_cfgB and c_dtype.width == 8:
        return _EpiDatapath(
            store_kind="fp8_simt",
            subtile_n=64,
            c_tile_rows=m_per_cta,
            ldtm_shape="16x32bx2",
            ldtm_repx=32,
            ldtm_offset=32,
            warp_m_count=num_epi_warps,
            warp_n_count=1,
        )
    is_cfgC = m_per_cta == 64 and use_2cta_instrs
    return _EpiDatapath(
        store_kind="swizzled",
        subtile_n=32,
        c_tile_rows=32 * num_epi_warps,
        ldtm_shape="32x32b",
        ldtm_repx=32,
        ldtm_offset=0,
        warp_m_count=2 if is_cfgC else num_epi_warps,
        warp_n_count=2 if is_cfgC else 1,
    )


@cute.kernel
def _kernel(
    # Persistent tile scheduler parameters
    tile_sched_params: utils.PersistentTileSchedulerParams,
    # Constexpr knobs
    mma_tiler: cutlass.Constexpr[Tuple[int, int, int]],
    mma_inst_shape_mnk: cutlass.Constexpr[Tuple[int, int, int]],
    # GEMM-K tile count (= ceil_div(T*R*S*C, mma_tiler_k)), host-computed.
    k_tile_cnt: cutlass.Constexpr[int],
    num_ab_stage: cutlass.Constexpr[int],
    num_acc_stage: cutlass.Constexpr[int],
    num_c_stage: cutlass.Constexpr[int],
    use_2cta_instrs: cutlass.Constexpr[bool],
    ab_dtype: cutlass.Constexpr[Type[Numeric]],
    c_dtype: cutlass.Constexpr[Type[Numeric]],
    acc_dtype: cutlass.Constexpr[Type[Numeric]],
    # Conv geometry consumed by the im2col TMA producer / epilogue:
    # zpq      = output spatial dims (Z, P, Q) used to decompose linear M -> (n,z,p,q)
    # trs      = filter spatial dims (T, R, S) used to decompose K-tile index
    # lower_pad_dhw / stride_dhw / dilation_dhw = standard conv params
    lower_pad_dhw: cutlass.Constexpr[Tuple[int, int, int]],
    stride_dhw: cutlass.Constexpr[Tuple[int, int, int]],
    dilation_dhw: cutlass.Constexpr[Tuple[int, int, int]],
    zpq: cutlass.Constexpr[Tuple[int, int, int]],
    trs: cutlass.Constexpr[Tuple[int, int, int]],
    # Host-built TMA descriptors, consumed by the device-side
    # prims.cp_async_bulk_tensor_* / prims.prefetch_tensormap calls.
    tma_a_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_c_desc: cutlass.GridConstant[cuda.TensorMap],
    epilogue_op: cutlass.Constexpr,
) -> None:
    """CTM-style implicit-GEMM fprop kernel: TMA load + tcgen05 MMA + TMA store."""

    # Warp / thread / cluster identity
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx, _, _ = cute.arch.thread_idx()
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

    # CTA-group geometry, threaded through every tcgen05 call site below.
    # _atom_thr = MMA-atom CTA count (2 for the 2-CTA tcgen05 group, else 1).
    # Cluster ranks are M-fast (grid.x = cluster_m), so a 2-CTA group G spans
    # ranks {2G, 2G+1} with its leader at the even rank 2G. _cta_group is the
    # matching nvvm enum.
    _atom_thr = 2 if cutlass.const_expr(use_2cta_instrs) else 1
    _cta_group = prims.CTAGroup.CTA_2 if cutlass.const_expr(use_2cta_instrs) else prims.CTAGroup.CTA_1
    if cutlass.const_expr(use_2cta_instrs):
        # 2-CTA group leader = even cluster rank (covers multi-group clusters,
        # e.g. cluster_m > 2, where group 1's leader sits at rank 2).
        is_leader_cta = (cta_rank_in_cluster % _atom_thr) == 0
        # tcgen05_commit multicast mask: 3 << rank covers the issuing group's
        # own pair {2G, 2G+1}. Only the leader (even rank) issues the commit,
        # so this is the multi-group-safe form (hard-coded 3 would deadlock
        # groups G>0). See prims.tcgen05_commit docstring.
        _commit_mask = cutlass.Int32(3) << cta_rank_in_cluster
    else:
        # 1-CTA path: every CTA owns its own accumulator => its own leader.
        # constexpr True so the leader-gated scf.if branches elide entirely.
        is_leader_cta = True
        # None selects the CTA_1 no-multicast commit path (arrive on own mbar).
        _commit_mask = None

    # Warp role assignment: epilogue 0-3, MMA 4, TMA producer 5.
    epilogue_warp_ids = (0, 1, 2, 3)
    mma_warp_id = 4
    tma_warp_id = 5

    # Prefetch the three TMA descriptors from the MMA warp; the descriptor cache
    # is shared across the cluster.
    if warp_idx == mma_warp_id:
        prims.prefetch_tensormap(tma_a_desc.get_ptr())
        prims.prefetch_tensormap(tma_b_desc.get_ptr())
        prims.prefetch_tensormap(tma_c_desc.get_ptr())

    # ===== Shared memory: data tiles + mbarriers + cluster init =====
    #
    # Every SMEM object is a flat cutlass.Array(space=smem), laid out in
    # declaration order. Swizzle is expressed by the descriptors
    # (cuda.TensorMapSwizzle.* on the host, Tcgen05SmemDesc layout on device),
    # not by the SMEM tensor type.
    #
    # Layout: mbarriers first (Int64 each):
    #   - ab_full / ab_empty (one per AB pipeline stage)
    #   - acc_full / acc_empty (one per accumulator stage)
    #   - tmem_dealloc_mbar (single, for the tcgen05 dealloc handshake)
    # then the TMEM holding slot (Int32, where tcgen05_alloc writes the TMEM
    # ptr), then the A/B/C data tiles as Int8 byte arrays sized
    # num_bytes * num_stage. cutlass.Array offsets an element/byte view with
    # ``.subview(n)`` and hands out a raw Pointer with ``.data_ptr(n)`` (the
    # latter is masked for bit-24 CTA_2 routing).
    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, num_acc_stage, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, num_acc_stage, space=cutlass.AddressSpace.smem)
    tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    # Per-CTA per-stage SMEM byte sizes (integer; same formulas as the host
    # _setup_attributes, which must agree byte-for-byte).
    _m_per_cta = mma_tiler[0] // _atom_thr
    _n_per_cta = mma_tiler[1] // _atom_thr
    _k_tile = cute.size(mma_tiler, mode=[2])
    _a_stage_bytes = _m_per_cta * _k_tile * (ab_dtype.width // 8)
    _b_stage_bytes = _n_per_cta * _k_tile * (ab_dtype.width // 8)
    # Epilogue C tile sizing (must match the device epilogue store path + the
    # host C descriptor). The datapath descriptor is the single source of truth
    # for subtile_n / c_tile_rows (see _select_epi_datapath); deriving the SMEM
    # byte size here from the same descriptor keeps this site, the device
    # epilogue, and the host C descriptor byte-for-byte consistent.
    _epi = _select_epi_datapath(_m_per_cta, use_2cta_instrs, c_dtype, num_epi_warps=len(epilogue_warp_ids))
    _epi_subtile_n = _epi.subtile_n
    _c_tile_rows = _epi.c_tile_rows
    _c_stage_bytes = _c_tile_rows * _epi_subtile_n * (c_dtype.width // 8)

    # A/B/C data tiles as flat Int8 byte arrays (1024B aligned).
    sA = cutlass.Array(
        cutlass.Int8,
        _a_stage_bytes * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    sB = cutlass.Array(
        cutlass.Int8,
        _b_stage_bytes * num_ab_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    # Epilogue C SMEM (only when TMA store is used; num_c_stage stages).
    sC = None
    if cutlass.const_expr(num_c_stage > 0):
        sC = cutlass.Array(
            c_dtype,
            (_c_stage_bytes // (c_dtype.width // 8)) * num_c_stage,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )

    # mbarrier_init: one warp / one lane initializes every barrier.
    # arrive_count for acc_empty is (4 epilogue warps) × (2 CTAs if 2cta else 1).
    num_acc_empty_arrives = len(epilogue_warp_ids) * (2 if use_2cta_instrs else 1)
    if warp_idx == 0:
        if prims.elect_sync():
            prims.mbarrier_init(tmem_dealloc_mbar_ptr, cute.arch.WARP_SIZE)
            for i in cutlass.range_constexpr(num_acc_stage):
                prims.mbarrier_init(acc_empty_mbar_ptr.subview(i), num_acc_empty_arrives)
                prims.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
            for i in cutlass.range_constexpr(num_ab_stage):
                prims.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                prims.mbarrier_init(ab_empty_mbar_ptr.subview(i), 1)

    # Cluster sync sandwich: fence_mbarrier_init publishes the mbarrier_init
    # writes to the cluster; barrier_cluster_arrive_relaxed signals this CTA is
    # done initializing; barrier_cluster_wait (later, after tile geometry is
    # set up) blocks until every CTA in the cluster arrives.
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()

    # Initial pipeline phase bits — parity flips on stage wrap-around.
    ab_empty_phase_bit = 1
    ab_full_phase_bit = 0
    acc_empty_phase_bit = 1
    acc_full_phase_bit = 0

    # Pre-build the tcgen05 instruction descriptor (constexpr) so MMA / epilogue
    # warps can reference it without recomputing. mma_inst_shape_mnk encodes the
    # per-instruction (M, N, K) shape, e.g. (256, 32, 16) for FP16/2cta.
    idesc = prims.Tcgen05InstrDesc.build(
        a_dtype=ab_dtype,
        b_dtype=ab_dtype,
        c_dtype=acc_dtype,
        m_dim=mma_inst_shape_mnk[0],
        n_dim=mma_inst_shape_mnk[1],
    )
    # MMA kind is fixed by the operand dtype (constexpr) and shared by every
    # tcgen05.mma issued in the K-loop below.
    mma_kind = _mma_kind_for_dtype(ab_dtype)

    # Persistent tile scheduler.
    tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
    work_tile = tile_sched.initial_work_tile_info()

    # tcgen05 TMEM allocation knob (always 512 cols on Blackwell).
    num_tmem_cols = 512

    # Cluster wait — gates SMEM/mbarrier visibility before any warp loads/stores.
    prims.barrier_cluster_wait()

    # ===== Tile geometry (visible to all warps) =====
    #
    # The A/B loads and the C store are raw prims.cp_async_bulk_tensor_* calls
    # that take a descriptor pointer plus hand-computed coords; the epilogue
    # computes its im2col store coords (k_off, q, p, z, n) by hand. The MMA is
    # driven by prims.Tcgen05SmemDesc (A/B) + prims.make_tmem_ptr (acc). So no
    # partition tensors or MMA fragment objects are materialized here — only the
    # per-CTA tile scalars below.

    # Per-CTA tile sizes (used by raw TMA producer + tcgen05 desc strides).
    mma_tiler_per_cta_m = mma_tiler[0] // (2 if use_2cta_instrs else 1)
    n_per_cta = mma_tiler[1] // (2 if use_2cta_instrs else 1)
    k_tile_size = cute.size(mma_tiler, mode=[2])

    # Bytes-per-K-tile expectation. For the 2-CTA cluster the leader writes
    # expect_tx for both CTAs' loads (the peer slot is reached via the
    # mbarrier bit-24 clear inside the wrapper).
    a_bytes_per_stage = mma_tiler_per_cta_m * k_tile_size * (ab_dtype.width // 8)
    b_bytes_per_stage = n_per_cta * k_tile_size * (ab_dtype.width // 8)
    num_tma_copy_bytes = a_bytes_per_stage + b_bytes_per_stage
    if cutlass.const_expr(use_2cta_instrs):
        num_tma_copy_bytes = num_tma_copy_bytes * 2

    # ===== TMA producer warp =====
    # Per-CTA unicast (no multicast_mask) prims.cp_async_bulk_tensor loads.
    # Im2col coords are decomposed from the linear M index:
    # m_off_cta -> (n, z, p, q); spatial anchors are then q*str_w-pad_w (etc).
    # The filter coord (s, r, t) and C-element offset are loop-carried and
    # advanced by a colexicographic carry (add + compare + conditional reset),
    # so the loop body never divides the linear k by T*R*S.
    if warp_idx == tma_warp_id:
        ab_stage_idx = 0
        Z_out, P_out, Q_out = zpq
        T_filt, R_filt, S_filt = trs
        pad_d, pad_h, pad_w = lower_pad_dhw
        str_d, str_h, str_w = stride_dhw
        dil_d, dil_h, dil_w = dilation_dhw
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // (2 if use_2cta_instrs else 1),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )
            # M index for this CTA's portion of the M-tile.
            m_off_cta = mma_tile_coord_mnl[0] * mma_tiler[0] + (cta_rank_in_cluster % _atom_thr if use_2cta_instrs else 0) * mma_tiler_per_cta_m
            # Output (n, z, p, q) from linear M (col-major Q-fastest).
            n_idx = m_off_cta // (Q_out * P_out * Z_out)
            rem = m_off_cta % (Q_out * P_out * Z_out)
            z_idx = rem // (Q_out * P_out)
            rem = rem % (Q_out * P_out)
            p_idx = rem // Q_out
            q_idx = rem % Q_out
            # Spatial anchors for im2col TMA: idx*stride - pad_lower.
            w_anchor = q_idx * str_w - pad_w
            h_anchor = p_idx * str_h - pad_h
            d_anchor = z_idx * str_d - pad_d
            # B side: per-CTA N offset (KTRSC tiled descriptor consumes this).
            n_off_cta = mma_tile_coord_mnl[1] * mma_tiler[1] + (cta_rank_in_cluster % _atom_thr if use_2cta_instrs else 0) * n_per_cta

            # Filter coord (s, r, t) + C-chunk, carried across K-tiles and
            # advanced colexicographically (s fastest). Reset to origin per
            # work-tile since each tile sweeps GEMM-K from 0. The carry below
            # replaces k // (T*R*S) style division.
            s_idx = 0
            r_idx = 0
            t_idx = 0
            c_chunk_idx = 0
            for k in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                mbar_full = ab_full_mbar_ptr.subview(ab_stage_idx)
                mbar_empty = ab_empty_mbar_ptr.subview(ab_stage_idx)

                # Producer acquire: wait until consumer (MMA) released this stage.
                # try_wait_parity issues a single non-blocking attempt that may
                # hardware-suspend up to time_limit then return False; a blocking
                # wait must retry in a loop.
                while not prims.mbarrier_try_wait_parity(mbar_empty, ab_empty_phase_bit, time_limit=10000000):
                    pass

                # Filter/C offsets from the carried colex coord (no division).
                c_off = c_chunk_idx * k_tile_size
                # Im2col offsets fold dilation in.
                s_off = s_idx * dil_w
                r_off = r_idx * dil_h
                t_off = t_idx * dil_d

                # Per-stage SMEM slices as cutlass.Array byte offsets: sA/sB are
                # flat Int8 arrays, stage stride = bytes-per-stage.
                sA_stage = sA.subview(ab_stage_idx * a_bytes_per_stage)
                sB_stage = sB.subview(ab_stage_idx * b_bytes_per_stage)

                if prims.elect_sync():
                    # Group leader sets the byte expectation. In 2cta the
                    # per-group leader (even rank) counts both pair members'
                    # loads (num_tma_copy_bytes doubled, bit-24 routing folds
                    # the peer's complete_tx onto the leader mbar). In 1cta
                    # is_leader_cta is constexpr-True, so every CTA counts its
                    # own (undoubled) bytes on its own mbar.
                    if is_leader_cta:
                        prims.mbarrier_arrive_expect_tx(mbar_full, num_tma_copy_bytes)
                    # Per-CTA unicast load; _cta_group selects the 1-/2-CTA
                    # tcgen05 group (identical coords/box on both paths).
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sA_stage,
                        tma_a_desc.get_ptr(),
                        (c_off, w_anchor, h_anchor, d_anchor, n_idx),
                        mbar_full,
                        [s_off, r_off, t_off],
                        mode=prims.TMALoadMode.IM2COL,
                        group=_cta_group,
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sB_stage,
                        tma_b_desc.get_ptr(),
                        (c_off, s_idx, r_idx, t_idx, n_off_cta),
                        mbar_full,
                        [],
                        group=_cta_group,
                    )

                ab_stage_idx += 1
                if ab_stage_idx == num_ab_stage:
                    ab_stage_idx = 0
                    ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                # Colex carry of (s, r, t, c_chunk): bump s, ripple right on
                # wrap. Nested single-compare ifs (no boolean-and on traced
                # predicates) advance the coord over (S, R, T, *).
                s_idx += 1
                if s_idx == S_filt:
                    s_idx = 0
                    r_idx += 1
                    if r_idx == R_filt:
                        r_idx = 0
                        t_idx += 1
                        if t_idx == T_filt:
                            t_idx = 0
                            c_chunk_idx += 1

            # Advance to next persistent work-tile (static schedule).
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    # ---- MMA consumer warp ----
    elif warp_idx == mma_warp_id:
        # MMA + 4 epilogue warps participate in this barrier so the MMA warp
        # can pick up the tmem_ptr written by the allocator (epilogue warp 0).
        # Non-aligned subset barrier — only 5 warps × 32 threads = 160 lanes.
        tmem_bar_id = 1
        tmem_bar_threads = 32 * (1 + len(epilogue_warp_ids))
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)
        tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), acc_dtype)

        # AB / acc pipeline consumer state — tracks the current MMA stage.
        ab_stage_idx = 0
        acc_stage_idx = 0

        # tcgen05 SMEM descriptors for stage 0. tcgen05_mma encodes SMEM
        # addresses in 16-byte units, so per-stage / per-K-block increments
        # are bytes >> 4. (leading=16, stride=1024, layout=2) is the 128B-swizzle
        # K-major layout, matching cuda.TensorMapSwizzle.s128b on the A/B
        # descriptors. sA/sB are flat cutlass.Arrays, so the SMEM base address is
        # the array itself.
        desc_a_base = prims.Tcgen05SmemDesc.build(
            sA,
            leading_byte_offset=16,
            stride_byte_offset=1024,
            layout=2,
        )
        desc_b_base = prims.Tcgen05SmemDesc.build(
            sB,
            leading_byte_offset=16,
            stride_byte_offset=1024,
            layout=2,
        )

        # Per-stage descriptor delta (one full AB stage in SMEM, in 16B units).
        # a_bytes_per_stage / b_bytes_per_stage are already per-stage values,
        # so no division by num_ab_stage here.
        sA_increment_per_stage = a_bytes_per_stage >> 4
        sB_increment_per_stage = b_bytes_per_stage >> 4

        desc_a_cur = desc_a_base
        desc_b_cur = desc_b_base

        # Per-K-block descriptor delta inside one AB stage.
        inc_bytes_per_iter = mma_inst_shape_mnk[2] * ab_dtype.width // 8
        increment = inc_bytes_per_iter >> 4
        # Conv K-dim is structured (C,S,R,T); flatten via cute.size before //.
        num_k_blocks = cute.size(mma_tiler, mode=[2]) // mma_inst_shape_mnk[2]

        # Persistent loop: outer = work tiles (acc stages); inner = K-tiles.
        while work_tile.is_valid_tile:
            current_acc_stage = acc_stage_idx
            acc_empty_mbar_ptr_stage = acc_empty_mbar_ptr.subview(current_acc_stage)
            acc_full_mbar_ptr_stage = acc_full_mbar_ptr.subview(current_acc_stage)
            current_empty_phase_bit = acc_empty_phase_bit

            acc_stage_idx += 1
            if acc_stage_idx == num_acc_stage:
                acc_stage_idx = 0
                acc_empty_phase_bit = acc_empty_phase_bit ^ 1

            if is_leader_cta:
                # Wait until the previous result has been consumed (epilogue
                # released this acc stage). No elect_sync — every thread is fine.
                # try_wait_parity is one non-blocking attempt; loop until it
                # reports the phase advanced.
                while not prims.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr_stage,
                    current_empty_phase_bit,
                    time_limit=10000000,
                ):
                    pass

                # Per-stage TMEM pointer. A TMEM Pointer holds a packed i32
                # address token (col in bits [0:16), row in [16:32)); adding an
                # int advances the raw column id directly, with no element/byte
                # scaling. Each acc stage owns a full mma_tiler[1]-wide column
                # band, so the per-stage stride is exactly mma_tiler[1] columns.
                tmem_ptr_for_mma = tmem_ptr.data_ptr() + current_acc_stage * mma_tiler[1]
                tmem_ptr_curr = cutlass.Array(tmem_ptr_for_mma, dtype=cutlass.Int32, addrspace=6)

                # scale_d=False (overwrite) only on the very first MMA of this
                # C-tile's K-loop; True (accumulate) for every subsequent MMA.
                # We use a runtime expression on (k_idx, kb_idx) instead of a
                # mutable Python local — the local would freeze at trace time
                # and re-trigger overwrite on every dynamic outer iteration of
                # the partially-unrolled K-tile loop.
                for k_idx in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_full_mbar_ptr_stage = ab_full_mbar_ptr.subview(ab_stage_idx)
                    ab_empty_mbar_ptr_stage = ab_empty_mbar_ptr.subview(ab_stage_idx)

                    # Wait for producer (TMA warp) to fill this AB stage.
                    # try_wait_parity is one non-blocking attempt; loop until the
                    # producer's arrive advances the phase.
                    while not prims.mbarrier_try_wait_parity(ab_full_mbar_ptr_stage, ab_full_phase_bit, time_limit=10000000):
                        pass

                    # Issue all K-blocks for this AB stage.
                    desc_a_cur_ = desc_a_cur
                    desc_b_cur_ = desc_b_cur
                    for kb_idx in cutlass.range_constexpr(num_k_blocks):
                        scale_d_now = (k_idx > 0) | (kb_idx > 0)
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                mma_kind,
                                _cta_group,
                                tmem_ptr_curr,
                                desc_a_cur_,
                                desc_b_cur_,
                                idesc,
                                scale_d_now,
                            )
                        desc_a_cur_ = desc_a_cur_ + increment
                        desc_b_cur_ = desc_b_cur_ + increment

                    # Advance AB stage state.
                    ab_stage_idx += 1
                    desc_a_cur = desc_a_cur + sA_increment_per_stage
                    desc_b_cur = desc_b_cur + sB_increment_per_stage
                    if ab_stage_idx == num_ab_stage:
                        ab_stage_idx = 0
                        desc_a_cur = desc_a_base
                        desc_b_cur = desc_b_base
                        ab_full_phase_bit = ab_full_phase_bit ^ 1

                    if prims.elect_sync():
                        # Release this AB stage to the TMA producer. In 2cta
                        # _commit_mask = 3 << rank broadcasts to the issuing
                        # group's pair; in 1cta it is None (arrive on own mbar).
                        prims.tcgen05_commit(
                            ab_empty_mbar_ptr_stage,
                            multicast_mask=_commit_mask,
                            group=_cta_group,
                        )

                if prims.elect_sync():
                    # K-tile loop done — signal accumulator full to epilogue
                    # warps (both CTAs of the group in 2cta; own CTA in 1cta).
                    prims.tcgen05_commit(
                        acc_full_mbar_ptr_stage,
                        multicast_mask=_commit_mask,
                        group=_cta_group,
                    )

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # Producer tail: drain the remaining acc stages before exit so the
        # kernel doesn't race against epilogue warps still signalling
        # acc_empty after the MMA warp has died.
        tail_stage = acc_stage_idx
        tail_phase = acc_empty_phase_bit
        if is_leader_cta:
            for _ in cutlass.range_constexpr(num_acc_stage - 1):
                tail_stage = tail_stage + 1
                if tail_stage == num_acc_stage:
                    tail_stage = 0
                    tail_phase = tail_phase ^ 1
            if prims.elect_sync():
                while not prims.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(tail_stage),
                    tail_phase,
                    time_limit=10000000,
                ):
                    pass

    # ---- Epilogue warps (0-3) ----
    elif warp_idx < mma_warp_id:
        # Per-CTA M tile size — 2cta cluster halves mma_tiler[0] across the pair.
        mma_tiler_per_cta_m = mma_tiler[0] // 2 if cutlass.const_expr(use_2cta_instrs) else mma_tiler[0]
        # Epilogue datapath descriptor — resolved once by _select_epi_datapath
        # (called in the prologue, reused here). It owns every cfgA / cfgB
        # (stmatrix | fp8-SIMT) / cfgC decision: store kind, LDTM shape/repx/
        # offset, N-subtile width, and the (warp_m, warp_n) split. Reusing the
        # prologue descriptor is what keeps this site, the SMEM sizing, and the
        # host C descriptor in lockstep. See _EpiDatapath for the datapath table.
        use_stmatrix_epi = _epi.use_stmatrix
        use_fp8_simt_epi = _epi.use_fp8_simt
        warp_m_count = _epi.warp_m_count
        warp_n_count = _epi.warp_n_count
        # This warp's position in the (warp_m, warp_n) grid.
        warp_m_idx = warp_idx % warp_m_count
        warp_n_idx = warp_idx // warp_m_count
        # TMEM->RMEM (LDTM) shape + SMEM N-subtile width + column half-split
        # offset (only 16x32bx2 fp8 uses a non-zero offset).
        t2r_inst_shape = _epi.ldtm_shape
        t2r_inst_repx = _epi.ldtm_repx
        t2r_ld_offset = _epi.ldtm_offset
        subtile_n = _epi.subtile_n
        # N-subtiles per warp = warp's N-band width (full N / warp_n) / subtile_n.
        #   cfgB fp16: 256/16=16.  cfgB fp32: 256/8=32.  cfgB fp8: 256/64=4.
        #   cfgA: 256/32=8.  cfgC: 128/32=4.
        subtile_cnt = (mma_tiler[1] // warp_n_count) // subtile_n

        # Sync ids — tmem_bar_id=1 already used by the MMA consumer warp.
        # epilog_sync_bar_id=2 is a separate named barrier across the 4 epi warps.
        threads_in_epilogue = 32 * len(epilogue_warp_ids)
        epilog_sync_bar_id = 2
        allocator_warp_id = epilogue_warp_ids[0]
        tmem_bar_id = 1
        tmem_bar_threads = 32 * (1 + len(epilogue_warp_ids))

        # 128-bit (16-byte) SMEM store width — store_swizzled vectorizes per lane.
        vsize = 128 // c_dtype.width

        # per-CTA M=64 thread->SMEM placement (hoisted, CTA-invariant). Shared by
        # both cfgB datapaths because the 16x256b/16x128b stmatrix LDTM and the
        # 16x32bx2 fp8 LDTM route TMEM by the SAME thread->(row,channel-half) map:
        # thread (warp w, lane_lo L=tid%16, lane_hi H=(tid//16)%2) owns SMEM M-row
        # (w*16+L), channel-half H. Decomposing tidx (0..127): t_mod_16 = L,
        # (t_div_16 % 2) = H, (t_div_16 // 2) = w. So final_offset = L*subtile_n +
        # H*(subtile_n//2) + w*16*subtile_n = base of M-row (w*16+L) at channel
        # half H. Element offsets scale with subtile_n so the byte layout is the
        # same for fp16 (subtile_n=16, 8 f16), fp32 (subtile_n=8, 4 f32) and fp8
        # (subtile_n=64, 32 fp8 contiguous channels). The middle term is the
        # half-fragment / lane-half column step = subtile_n // 2.
        if cutlass.const_expr(use_stmatrix_epi or use_fp8_simt_epi):
            t_div_16 = tidx // 16
            t_mod_16 = tidx % 16
            final_offset = t_mod_16 * subtile_n + (t_div_16 % 2) * (subtile_n // 2) + (t_div_16 // 2) * 16 * subtile_n
        # stmatrix-only constants. stmatrix.8x8.x4 lays an 8x8 b16 fragment per
        # call (16 B/thread = 8 f16 or 4 f32) into SW32-swizzled C SMEM (s32b
        # descriptor), so the store address is XOR-permuted by hand: Swizzle(1,4,3)
        # => mask=128,shift=3. The fp8 SIMT path uses store_swizzled instead and
        # needs none of these.
        if cutlass.const_expr(use_stmatrix_epi):
            stmatrix_layout = prims.MMALayout.ROW
            stmatrix_shape = "m8n8"
            _swz_mask = cutlass.Int64(128)
            _swz_shift = cutlass.Int64(3)

        # C-side im2col store coords are computed by hand. The im2col STORE op
        # has no im2col-offset operand (unlike the A LOAD), and the C descriptor
        # uses zero corners, so output spatial coords are bare pixels (no pad
        # subtraction, no stride multiply). Decompose the linear per-CTA M index
        # into (n, z, p, q) per work-tile in the loop.
        Z_out, P_out, Q_out = zpq

        # Allocator warp (epi 0) reserves 512 TMEM cols and stashes the pointer
        # in tmem_holding_buf. Every CTA allocates from its own 512-col bank;
        # _cta_group arranges peer-side state for the 2-CTA group (no-op for
        # CTA_1). This op is warp-collective (NOT elect-safe) and must stay
        # outside any rank-divergent branch, so it is unconditional here.
        if warp_idx == allocator_warp_id:
            prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_cols, group=_cta_group)

        # 5-warp barrier: the MMA consumer warp also waits on this id, so it
        # picks up the same tmem_ptr right after the allocator publishes it.
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)

        tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), acc_dtype)
        tmem_raw_addr = tmem_ptr_i32.load()

        # Persistent loop: outer over work-tiles (one per acc stage), inner over
        # epilogue subtiles within each tile.
        acc_stage_idx = 0
        epi_stage_idx = 0
        while work_tile.is_valid_tile:
            current_acc_stage = acc_stage_idx
            acc_full_mbar_ptr_stage = acc_full_mbar_ptr.subview(current_acc_stage)
            acc_empty_mbar_ptr_stage = acc_empty_mbar_ptr.subview(current_acc_stage)
            current_full_phase_bit = acc_full_phase_bit

            acc_stage_idx += 1
            if acc_stage_idx == num_acc_stage:
                acc_stage_idx = 0
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // (2 if use_2cta_instrs else 1),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )

            # Wait for MMA warp to commit acc_full for this stage.
            # try_wait_parity is one non-blocking attempt; loop until the MMA
            # warp's commit advances the phase.
            while not prims.mbarrier_try_wait_parity(acc_full_mbar_ptr_stage, current_full_phase_bit, time_limit=10000000):
                pass

            # Per-stage TMEM column origin. Lower 16 bits of tmem_raw_addr is
            # the column id; upper 16 bits is the row id (always 0 for warp 0).
            base_col_id = (tmem_raw_addr & 0xFFFF) + (current_acc_stage * mma_tiler[1])

            # Per-tile output (im2col store) spatial coords. The 2CTA cluster
            # splits M (the NZPQ pixel axis) across the pair, so the spatial
            # coords carry cta_rank. The K-out channel axis (GEMM N) is NOT
            # split: each CTA's tcgen05 accumulator holds its own M-row band
            # over the full N columns.
            m_off_cta = mma_tile_coord_mnl[0] * mma_tiler[0] + (cta_rank_in_cluster % _atom_thr if use_2cta_instrs else 0) * mma_tiler_per_cta_m
            # Bare output pixels (col-major Q-fastest). No pad/stride: the C
            # descriptor uses zero corners, so output has no halo concept.
            n_out = m_off_cta // (Q_out * P_out * Z_out)
            rem_m = m_off_cta % (Q_out * P_out * Z_out)
            z_out = rem_m // (Q_out * P_out)
            rem_m = rem_m % (Q_out * P_out)
            p_out = rem_m // Q_out
            q_out = rem_m % Q_out
            # K-out channel base for this N-tile (no cta_rank, see above).
            k_off_base = mma_tile_coord_mnl[1] * mma_tiler[1]

            # Subtile loop on the N axis.
            for subtile_idx in cutlass.range(subtile_cnt):
                # Rotate through C SMEM stages so the previous TMA store can
                # drain in parallel with the next t2r/r2s.
                epi_stage_idx = (epi_stage_idx + 1) % num_c_stage

                # TMEM -> RMEM: per-warp LDTM. The column origin advances ONLY by
                # subtile_n and carries NO warp_n term: tcgen05_ld routes by the
                # executing warp's sub-partition, so each warp lands on its own
                # N-band without an explicit column add. An explicit warp_n column
                # offset would over-shoot into unwritten TMEM (reading back zero).
                # The global k128..255 offset is applied on the STORE side
                # (band_smem_base + k_off), not here.
                tmem_ctm = prims.make_tmem_ptr_from_warp_row_col(
                    tmem_raw_addr,
                    warp_m_idx,
                    base_col_id + subtile_idx * subtile_n,
                    cutlass.Float32,
                )
                # offset= is mandatory for 16x32bx2 (the fp8 column half-split)
                # and rejected for every other shape, so pass it only for fp8.
                t2r_rmem = prims.tcgen05_ld(
                    t2r_inst_shape,
                    tmem_ctm,
                    num=t2r_inst_repx,
                    offset=t2r_ld_offset if cutlass.const_expr(use_fp8_simt_epi) else None,
                )
                # Apply the user-supplied epilogue_op (constexpr fn) on the FP32
                # accumulator before casting to the output dtype (defaults to
                # identity).
                t2r_rmem = epilogue_op(t2r_rmem)

                # RMEM -> SMEM. sC is a flat cutlass.Array (c_dtype elements);
                # per-stage stride = _c_tile_rows * _epi_subtile_n elems
                # (_c_tile_rows = 128 for store_swizzled so cfgC's two N-bands
                # both fit; = _epi_m=64 for cfgB stmatrix). Two paths, keyed on
                # the datapath predicate (constexpr):
                smem_tile_base = sC.subview(epi_stage_idx * (_c_tile_rows * _epi_subtile_n))
                if cutlass.const_expr(use_fp8_simt_epi):
                    # fp8 cfgB SIMT store. This thread holds 32 contiguous
                    # channels of one M-row/half; final_offset is that row/half
                    # base in the C SMEM. Write them in vsize-wide store_swizzled
                    # chunks (fp8 vsize=16, so 2 stores). Same swizzle helper as
                    # the host C descriptor so the two layouts match.
                    for j in cutlass.range_constexpr(32 // vsize):
                        vec_f32 = t2r_rmem[j * vsize : j * vsize + vsize]
                        vec_io = vec_f32.to(c_dtype)
                        smem_thr_ptr = smem_tile_base.subview(cutlass.Int32(final_offset) + j * vsize)
                        smem_thr_ptr.data_ptr().store_swizzled(
                            vec_io,
                            alignment=64,
                            swizzle=_c_smem_swizzle(c_dtype, subtile_n),
                        )
                elif cutlass.const_expr(not use_stmatrix_epi):
                    # (4,1) path: 32x32b LDTM gives 1 contiguous row/thread. Each
                    # lane writes vsize elements per (i, j); tiles cover
                    # (t2r_inst_repx rows) x (subtile_n cols). The store_swizzled
                    # pattern is derived from the output dtype (same helper as the
                    # host C descriptor) so it matches for fp16/bf16 (s64b), fp32
                    # (s128b) and fp8 at m128/m64+2cta (s32b, subtile_n=32).
                    for i in cutlass.range_constexpr(t2r_inst_repx // 32):
                        for j in cutlass.range_constexpr(32 // vsize):
                            vec_f32 = t2r_rmem[i * 32 + j * vsize : i * 32 + j * vsize + vsize]
                            vec_io = vec_f32.to(c_dtype)
                            # offset by elements then take .data_ptr() for the
                            # cutlass.Pointer that exposes store_swizzled.
                            smem_thr_ptr = smem_tile_base.subview((tidx + i * 128) * subtile_n + j * vsize)
                            smem_thr_ptr.data_ptr().store_swizzled(
                                vec_io,
                                alignment=64,
                                swizzle=_c_smem_swizzle(c_dtype, subtile_n),
                            )
                else:
                    # per-CTA M=64 + 1cta: LDTM gives 2 interleaved rows/thread,
                    # which is the 8x8 fragment layout stmatrix expects. One
                    # stmatrix.8x8.x4 per thread writes a 16-byte fragment (4 i32)
                    # = 8 f16 or 4 f32, so the per-thread element count tracks the
                    # C dtype (= LDTM vector length = 128 // width). The C SMEM is
                    # SW32 (s32b descriptor), so the destination address is
                    # XOR-permuted by hand (store_swizzled cannot express the
                    # stmatrix fragment shape). final_offset is hoisted above.
                    _st_nelem = cutlass.const_expr(128 // c_dtype.width)
                    acc_f16 = cutlass.Array(c_dtype, _st_nelem, alignment=32)
                    acc_f16[0:_st_nelem] = t2r_rmem[0:_st_nelem].to(c_dtype)
                    smem_thr_ptr = smem_tile_base.subview(cutlass.Int32(final_offset))
                    smem_ptr_raw = smem_thr_ptr.data_ptr()
                    smem_addr0 = cutlass.Int64(llvm.ptrtoint(cutlass.Int64.mlir_type, smem_ptr_raw))
                    swz_addr0 = smem_addr0 ^ ((smem_addr0 & _swz_mask) >> _swz_shift)
                    dst0 = llvm.inttoptr(llvm.PointerType.get(3), swz_addr0.ir_value())
                    tmp_i32_0 = cutlass.Array(
                        acc_f16.ir_value(),
                        dtype=cutlass.Int32,
                        shape=(4,),
                        addrspace=acc_f16.space,
                    )
                    regs0 = tmp_i32_0[0:4]
                    prims.stmatrix(
                        dst0,
                        [regs0[0], regs0[1], regs0[2], regs0[3]],
                        stmatrix_layout,
                        shape=stmatrix_shape,
                    )

                # Make swizzled SMEM stores visible to the TMA proxy.
                cute.arch.fence_view_async_shared()
                prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)

                # Band-leader warps issue the raw im2col TMA store. Coord order
                # mirrors the A-load 5D layout (channel_off, spatial..., batch)
                # = (k_off, q, p, z, n); see cuda.create_tensor_map_im2col_from_view
                # docstring (stride-1 dim is the channel axis). The store op has
                # NO im2col-offset operand, so (s,r,t) are NOT passed — output
                # spatial offsets are baked into the descriptor's zero corners.
                # k_off advances by subtile_n per N-subtile (matches the C
                # descriptor's channels_per_pixel=subtile_n box).
                #
                # warp_m_idx==0 picks one leader per N-band: cfgA/cfgB (4,1) ->
                # warp 0 only, stores all mma_tiler_per_cta_m rows; cfgC (2,2) ->
                # warps 0 and 2, each stores its 64-row band from a per-band SMEM
                # offset (warp_n_idx*64 rows) to a k_off shifted by the N-band
                # width (warp_n_idx*128). The (q,p,z,n) pixel coords are shared by
                # both bands (the (2,2) split is on the K-out channel axis only).
                if warp_m_idx == 0:
                    k_off = k_off_base + warp_n_idx * (mma_tiler[1] // warp_n_count) + subtile_idx * subtile_n
                    band_smem_base = smem_tile_base.subview(warp_n_idx * mma_tiler_per_cta_m * subtile_n)
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        tma_c_desc.get_ptr(),
                        band_smem_base,
                        (k_off, q_out, p_out, z_out, n_out),
                        mode=prims.TMAStoreMode.IM2COL,
                    )
                    prims.cp_async_bulk_commit_group()
                    # Keep at most num_c_stage-1 inflight so the next iteration
                    # can reuse a drained SMEM stage.
                    prims.cp_async_bulk_wait_group(num_c_stage - 1, read=True)

                prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)

            # Signal acc_empty back to the group leader's mbarrier. In 2cta the
            # mbar lives in the even-rank leader's SMEM and arrive_count was set
            # to (4 epi warps * 2 CTAs); the per-group leader rank is
            # (rank // _atom_thr) * _atom_thr (degrades to 0 for the single
            # group, 2 for group 1, etc.). In 1cta _atom_thr=1 so this is the
            # CTA's own rank (count 4, own mbar).
            if prims.elect_sync():
                leader_cta_rank = (cta_rank_in_cluster // _atom_thr) * _atom_thr
                mbar_cluster_ptr = prims.mapa(acc_empty_mbar_ptr_stage, leader_cta_rank)
                prims.mbarrier_arrive(
                    mbar_cluster_ptr,
                    count=1,
                    scope=prims.MemScope.CLUSTER,
                )

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # Drain any inflight TMA stores before exiting the kernel.
        if warp_idx == epilogue_warp_ids[0]:
            prims.cp_async_bulk_wait_group(0, read=True)

        # tcgen05 dealloc. In 2cta both CTAs of the group must agree the TMEM
        # is no longer in use, so we route an arrival through mapa to the pair
        # partner's tmem_dealloc_mbar (partner = rank ^ 1, correct within each
        # consecutive {2G, 2G+1} pair) and wait on our own before freeing. In
        # 1cta each CTA owns its TMEM independently, so the handshake is elided
        # and we relinquish + dealloc directly.
        if warp_idx == allocator_warp_id:
            prims.tcgen05_relinquish_alloc_permit(group=_cta_group)

            if cutlass.const_expr(use_2cta_instrs):
                peer_cta_rank = cute.arch.make_warp_uniform(cta_rank_in_cluster ^ 1)
                peer_mbar = prims.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                prims.mbarrier_arrive(
                    peer_mbar,
                    count=1,
                    scope=prims.MemScope.CLUSTER,
                )

                # Wait until peer also signalled, then physically free the TMEM.
                # try_wait_parity is one non-blocking attempt; loop until the
                # peer's arrive advances the phase.
                while not prims.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10000000):
                    pass

            prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group=_cta_group)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# ---------------------------------------------------------------------------
# Host wrapper class + compile() / verify()
# ---------------------------------------------------------------------------
#
# FpropCTMKernel holds a ``PersistentConvSetup`` instance that derives the
# integer configuration (input attrs, mma/stage/byte sizes) and then launches
# the device ``kernel``. ``__call__`` materialises the kernel arguments from the
# setup attributes after ``_setup_attributes()``.

# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _check_tensor_alignment(
    c: int,
    k: int,
    ab_dtype: Type[Numeric],
    c_dtype: Type[Numeric],
    channel_tile: int,
):
    """Require 16-byte packed channel rows for im2col TMA descriptors."""

    def aligned(dtype, elements):
        return elements % (16 * 8 // dtype.width) == 0

    if not aligned(ab_dtype, c) or not aligned(c_dtype, k):
        raise testing.CantImplementError(f"C={c} and K={k} must provide 16-byte channel rows")

    # The producer advances filter coordinates once per complete mainloop K
    # tile. A partial channel tile would need a distinct carry path; decline it
    # until that path exists instead of skipping filter/channel contributions.
    if c % channel_tile:
        raise testing.CantImplementError(f"C={c} must be a multiple of the {channel_tile}-element mainloop tile for {ab_dtype}")


def _check_swizzle_size(
    m: int,
    n: int,
    mma_tiler_mn: Tuple[int, int],
    use_2cta_instrs: bool,
    preferred_cluster_shape_mn: Tuple[int, int],
    fallback_cluster_shape_mn: Tuple[int, int],
    swizzle_size: int,
    raster_along: str,
):
    """Check that swizzle_size does not exceed the cluster count in the swizzled dimension.

    :param m: GEMM M dimension (N*Z*P*Q for convolution)
    :param n: GEMM N dimension (K for convolution)
    :param mma_tiler_mn: MMA tiler shape (M, N)
    :param use_2cta_instrs: Whether 2-CTA instructions are used
    :param preferred_cluster_shape_mn: Preferred cluster shape (M, N)
    :param fallback_cluster_shape_mn: Fallback cluster shape (M, N)
    :param swizzle_size: Swizzle size to validate
    :param raster_along: Rasterization order ("m" or "n")
    """
    if swizzle_size <= 1:
        return

    cta_v = 2 if use_2cta_instrs else 1
    cta_tile_m = mma_tiler_mn[0] // cta_v
    cta_tile_n = mma_tiler_mn[1]
    m_tiles = -(-m // cta_tile_m)
    n_tiles = -(-n // cta_tile_n)

    for cs in [preferred_cluster_shape_mn, fallback_cluster_shape_mn]:
        if raster_along == "m":
            nclusters = -(-n_tiles // cs[1])
        else:
            nclusters = -(-m_tiles // cs[0])
        if nclusters < swizzle_size:
            dim_name = "N" if raster_along == "m" else "M"
            raise testing.CantImplementError(
                f"swizzle_size ({swizzle_size}) exceeds the number of "
                f"{dim_name} clusters ({nclusters}) for cluster shape "
                f"{cs}. Use a smaller swizzle_size or increase the "
                f"{dim_name} dimension."
            )


def _compute_zpq(
    dhw: Tuple[int, int, int],
    trs: Tuple[int, int, int],
    stride_dhw: Tuple[int, int, int],
    upper_padding_dhw: Tuple[int, int, int],
    lower_padding_dhw: Tuple[int, int, int],
    dilation_dhw: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Compute output spatial dimensions Z, P, and Q with asymmetric padding."""
    D, H, W = dhw
    T, R, S = trs
    Sd, Sh, Sw = stride_dhw
    UpperPadD, UpperPadH, UpperPadW = upper_padding_dhw
    LowerPadD, LowerPadH, LowerPadW = lower_padding_dhw
    DilD, DilH, DilW = dilation_dhw
    Z = ((D + UpperPadD + LowerPadD - DilD * (T - 1) - 1) // Sd) + 1
    P = ((H + UpperPadH + LowerPadH - DilH * (R - 1) - 1) // Sh) + 1
    Q = ((W + UpperPadW + LowerPadW - DilW * (S - 1) - 1) // Sw) + 1
    return Z, P, Q


def _compute_stages(
    a_bytes_per_stage: int,
    b_bytes_per_stage: int,
    c_bytes_per_stage: int,
    smem_capacity: int,
    occupancy: int,
) -> Tuple[int, int, int]:
    """Compute the number of (ACC, A/B, C) stages from SMEM-capacity heuristics.

    Byte sizes per stage are passed in (computed by the caller as
    ``per_cta_M * K_tile * dtype.width // 8``); the remaining SMEM after the
    mbarrier reserve and the initial C budget is split into A/B stages, then
    leftover SMEM is folded back into extra C stages.
    """
    # Default ACC stages
    num_acc_stage = 2

    # Default C stages (conv always uses im2col TMA store)
    num_c_stage = 2

    ab_bytes_per_stage = a_bytes_per_stage + b_bytes_per_stage
    mbar_helpers_bytes = 1024

    c_bytes = c_bytes_per_stage * num_c_stage

    # Calculate A/B stages from remaining smem after reserved + initial C bytes.
    num_ab_stage = (smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)) // ab_bytes_per_stage

    # Refine epilogue stages: add leftover smem to C.
    num_c_stage += (smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy * (mbar_helpers_bytes + c_bytes)) // (
        occupancy * c_bytes_per_stage
    )
    return num_acc_stage, num_ab_stage, num_c_stage


class _PersistentConvSetup:
    """Host-side config/setup for the implicit-GEMM fprop conv.

    Derives the integer configuration consumed by the device ``kernel``: input
    attributes, mma/tile shapes, SMEM byte sizes, pipeline stage counts, and
    can_implement checks.

    Conv->GEMM mapping: M=N*Z*P*Q, N=K, K=T*R*S*C.
    A is NDHWC C-major, B is KTRSC C-major, C is NZPQK K-major.

    :note: A and B tensor must be C major; C tensor must be K major.
    :note: A and B tensor must have the same data type.
    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
    """

    def __init__(
        self,
        acc_dtype: Type[Numeric],
        tile_config: ConvTileConfig,
        filter_trs: Tuple[int, int, int],
        upper_padding_dhw: Tuple[int, int, int],
        lower_padding_dhw: Tuple[int, int, int],
        stride_dhw: Tuple[int, int, int],
        dilation_dhw: Tuple[int, int, int],
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
    ):
        # --- base PersistentDenseGemmKernel.__init__ (split_k=1, TMA store) ---
        self.acc_dtype: Type[Numeric] = acc_dtype
        self.tile_config = tile_config
        self.use_2cta_instrs = tile_config.use_2cta_instrs
        cluster_shape_mn = tile_config.cluster_shape_mn
        # Base stores cluster_shape_mn = fallback (preferred-cluster wiring).
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along = raster_along
        self.split_k = 1
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mn = tile_config.mma_tiler_mn
        self.mma_tiler = (*self.mma_tiler_mn, 1)
        self.arch = "sm_100"
        # CTA group (1 or 2) is selected via use_2cta_instrs at the device
        # tcgen05 call sites (prims.CTAGroup.CTA_2).
        self.occupancy = 1
        # Set specialized warp ids
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                *self.epilogue_warp_id,
            )
        )
        # --- preferred-cluster overrides ---
        self.preferred_cluster_shape_mn = cluster_shape_mn
        self.fallback_cluster_shape_mn = cluster_shape_mn

        # --- conv parameters for im2col TMA ---
        self.filter_trs = filter_trs
        self.upper_padding_dhw = upper_padding_dhw
        self.lower_padding_dhw = lower_padding_dhw
        self.stride_dhw = stride_dhw
        self.dilation_dhw = dilation_dhw

    def check_supported_dtypes(
        self,
        a_dtype: Type[Numeric],
        b_dtype: Type[Numeric],
        c_dtype: Type[Numeric],
    ):
        """Check if the dtypes are valid.

        :raises testing.CantImplementError: If the dtypes are invalid
        """
        # int8/uint8 (and the Int32 accumulator they require) is intentionally
        # unsupported here: the TMA descriptor builder cannot encode an int8
        # instruction descriptor yet, and no primitives GEMM kernel runs int8
        # either. Re-add the int8/uint8 AB types, the Int32 accumulator, and
        # their accumulator/C compatibility entries once the primitives infra
        # and GEMM kernels gain int8 support.
        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float32,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }
        if a_dtype not in valid_ab_dtypes or b_dtype not in valid_ab_dtypes:
            raise testing.CantImplementError(f"Unsupported AB dtype: {a_dtype} and {b_dtype}")

        if self.acc_dtype not in {cutlass.Float32, cutlass.Float16}:
            raise testing.CantImplementError(f"Unsupported accumulator dtype: {self.acc_dtype}")

        # Define compatibility mapping between accumulator type and AB type
        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float32,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },  # Float32 accumulator supports floating point AB types only
            cutlass.Float16: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
        }
        # Check compatibility between accumulator type and AB type
        if a_dtype not in acc_ab_compatibility[self.acc_dtype] or b_dtype not in acc_ab_compatibility[self.acc_dtype]:
            raise testing.CantImplementError(f"Unsupported AB dtype: {a_dtype} and {b_dtype} for accumulator dtype: {self.acc_dtype}")

        # Define compatibility mapping between accumulator type and C type
        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.BFloat16,
                cutlass.Float16,
            },
        }
        # Check compatibility between accumulator type and C type
        if c_dtype not in acc_c_compatibility[self.acc_dtype]:
            raise testing.CantImplementError(f"Unsupported C dtype: {c_dtype} for accumulator dtype: {self.acc_dtype}")

    def check_mma_tiler_and_cluster_shape(self):
        """Check if the mma tiler and cluster shape are valid.

        :raises testing.CantImplementError: If the mma tiler and cluster shape are invalid
        """
        # Skip invalid mma tile shape
        if not ((not self.use_2cta_instrs and self.mma_tiler_mn[0] in [64, 128]) or (self.use_2cta_instrs and self.mma_tiler_mn[0] in [128, 256])):
            raise testing.CantImplementError(f"Invalid mma tiler & use_2cta_instrs: {self.mma_tiler_mn}, {self.use_2cta_instrs}")
        if self.mma_tiler_mn[1] not in range(32, 257, 32):
            raise testing.CantImplementError(f"Invalid mma tiler N: {self.mma_tiler_mn[1]}")
        # Epilogue path selection (per-CTA M = mma_tiler M / atom_thr); the
        # accumulator shape fixes the LDTM shape, which in turn picks the store:
        #   cfgA / 1cta-128 (per-CTA M==128): 32x32b LDTM +
        #     store_swizzled (32-col, s64b), warp (4,1).
        #   cfgC (per-CTA M==64 AND 2cta):     32x32b LDTM +
        #     store_swizzled (32-col, s64b), warp (2,2) — same datapath as cfgA.
        #     (2,2) is the standard TMEM-load warp tiling for M==64+2cta (every
        #     other per-CTA-M/2cta combo uses (4,1)).
        #   cfgB (per-CTA M==64 AND 1cta):      store/LDTM/subtile are
        #     dtype-dependent (stmatrix for >=16-bit, fp8_simt for fp8); see
        #     _select_epi_datapath for the resolved fields, warp (4,1).
        # The cfgB/cfgC split on the same per-CTA M=64 is gated by use_2cta_instrs
        # throughout the epilogue (use_stmatrix_epi = M==64 AND not 2cta).
        # Skip illegal cluster shape
        if self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0:
            raise testing.CantImplementError(f"Invalid cluster shape M: {self.cluster_shape_mn[0]}")

        # Skip invalid cluster shape (total cluster including split_k must be <= 16)
        def is_power_of_2(x: int) -> bool:
            return x > 0 and (x & (x - 1)) == 0

        total_cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1] * self.split_k
        if (
            total_cluster_size > 16
            or self.cluster_shape_mn[0] <= 0
            or self.cluster_shape_mn[1] <= 0
            or not is_power_of_2(self.cluster_shape_mn[0])
            or not is_power_of_2(self.cluster_shape_mn[1])
        ):
            raise testing.CantImplementError(f"Invalid cluster shape: {self.cluster_shape_mn} with split_k={self.split_k}")

    def _check_epilogue_tile(self, c_dtype: Type[Numeric]) -> None:
        """Require every epilogue warp band to contain whole store subtiles."""
        epi = _select_epi_datapath(self.tile_config.cta_tile_m, self.use_2cta_instrs, c_dtype)
        n_per_warp_band = self.tile_config.cta_tile_n // epi.warp_n_count
        if n_per_warp_band < epi.subtile_n or n_per_warp_band % epi.subtile_n:
            raise testing.CantImplementError(
                f"Tile N={self.tile_config.cta_tile_n} does not divide into {epi.warp_n_count} "
                f"epilogue band(s) of {epi.subtile_n}-column subtiles for {c_dtype}"
            )

    def _stage_counts(self, ab_dtype: Type[Numeric], c_dtype: Type[Numeric], smem_capacity: int) -> Tuple[int, int, int]:
        """Derive pipeline stages for this tile and dtype combination."""
        k_tile = self.tile_config.cta_tile_k(ab_dtype.width)
        m_per_cta = self.tile_config.cta_tile_m
        n_per_cta = self.tile_config.cta_tile_n // self.tile_config.cta_group
        epi = _select_epi_datapath(m_per_cta, self.use_2cta_instrs, c_dtype)
        a_bytes_per_stage = m_per_cta * k_tile * (ab_dtype.width // 8)
        b_bytes_per_stage = n_per_cta * k_tile * (ab_dtype.width // 8)
        c_bytes_per_stage = epi.c_tile_rows * epi.subtile_n * (c_dtype.width // 8)
        return _compute_stages(
            a_bytes_per_stage,
            b_bytes_per_stage,
            c_bytes_per_stage,
            smem_capacity,
            self.occupancy,
        )

    def can_implement(
        self,
        ncdhw: Tuple[int, int, int, int, int],
        k: int,
        ab_dtype: Type[Numeric],
        c_dtype: Type[Numeric],
    ) -> bool:
        """Determine if the given tensor configuration can be implemented."""
        try:
            self.check_supported_dtypes(ab_dtype, ab_dtype, c_dtype)
            # Validate fallback cluster shape (base stores it as cluster_shape_mn)
            self.check_mma_tiler_and_cluster_shape()
            self._check_epilogue_tile(c_dtype)
            channel_tile = self.tile_config.cta_tile_k(ab_dtype.width)
            _check_tensor_alignment(ncdhw[1], k, ab_dtype, c_dtype, channel_tile)
            _num_acc_stage, num_ab_stage, num_c_stage = self._stage_counts(
                ab_dtype,
                c_dtype,
                cutlass.memory.get_smem_capacity_in_bytes(),
            )
            if num_ab_stage < 1 or num_c_stage < 1:
                raise testing.CantImplementError(
                    f"Tile {self.tile_config.name} has no viable shared-memory pipeline " f"(AB stages={num_ab_stage}, C stages={num_c_stage})"
                )
            # Compute implicit GEMM M
            z, p, q = _compute_zpq(
                ncdhw[2:],
                self.filter_trs,
                self.stride_dhw,
                self.upper_padding_dhw,
                self.lower_padding_dhw,
                self.dilation_dhw,
            )
            _check_swizzle_size(
                ncdhw[0] * z * p * q,
                k,
                self.mma_tiler_mn,
                self.use_2cta_instrs,
                self.preferred_cluster_shape_mn,
                self.fallback_cluster_shape_mn,
                self.swizzle_size,
                self.raster_along,
            )
        except testing.CantImplementError:
            return False
        return True

    def _setup_conv_input_attrs(self, a, b, c):
        """Validate and set input-dependent attributes.

        Sets a_dtype, b_dtype, c_dtype (layouts are fixed by the conv contract).

        :param a: Input tensor A - (N, D, H, W, C) layout
        :param b: Filter tensor B - (K, T, R, S, C) layout
        :param c: Output tensor C - (N, Z, P, Q, K) layout
        """
        self.a_dtype: Type[Numeric] = a.element_type
        self.b_dtype: Type[Numeric] = b.element_type
        self.c_dtype: Type[Numeric] = c.element_type
        # Only C major accepted
        if cutlass.const_expr(a.leading_dim != 4):
            raise RuntimeError("The layout of a is not supported")
        if cutlass.const_expr(b.leading_dim != 4):
            raise RuntimeError("The layout of b is not supported")
        if cutlass.const_expr(c.leading_dim != 4):
            raise RuntimeError("The layout of c is not supported")
        # A/B are K-major (channel-contiguous), C is K-major (row-major), fixed
        # by the conv layout contract and baked into the TMA descriptors'
        # stride_order.

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

    def _setup_attributes(self):
        """Set up convolution-input-dependent configuration as plain integers
        (mma shapes, SMEM byte sizes, stage counts, multicast counts). The
        device kernel consumes only these scalars plus the TMA descriptors.
        """
        # Compute mma/cluster/tile shapes.
        # tcgen05 MMA K-instruction = 256 / dtype.width (FP16/BF16->16, FP8->32,
        # TF32->8).
        mma_inst_shape_k = 256 // self.a_dtype.width
        mma_inst_tile_k = self.tile_config.mma_inst_tile_k
        # atom_thr_size = number of CTAs cooperating on one MMA atom along M;
        # 2 for 2-CTA UTCMMA, else 1.
        atom_thr_size = 2 if self.use_2cta_instrs else 1
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            (mma_inst_shape_k * mma_inst_tile_k,),
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // atom_thr_size,
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.smem_capacity = cutlass.memory.get_smem_capacity_in_bytes()

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._stage_counts(self.a_dtype, self.c_dtype, self.smem_capacity)

        # SMEM A/B/C layouts are not materialized on the host: the device
        # kernel allocates flat cutlass.Array byte buffers sized from
        # a/b/c_bytes_per_stage and expresses swizzle via the TMA
        # descriptors + Tcgen05SmemDesc.


class _FpropCTMKernel:
    """CTM implicit-GEMM fprop convolution kernel (host entry point).

    Delegates configuration to a ``PersistentConvSetup`` instance, then launches
    the device ``kernel`` with a persistent tile scheduler.
    """

    def __init__(
        self,
        acc_dtype: Type[Numeric],
        tile_config: ConvTileConfig,
        filter_trs: Tuple[int, int, int],
        upper_padding_dhw: Tuple[int, int, int],
        lower_padding_dhw: Tuple[int, int, int],
        stride_dhw: Tuple[int, int, int],
        dilation_dhw: Tuple[int, int, int],
        swizzle_size: int = 1,
        raster_along: str = "m",
    ):
        self._setup = _PersistentConvSetup(
            acc_dtype=acc_dtype,
            tile_config=tile_config,
            filter_trs=filter_trs,
            upper_padding_dhw=upper_padding_dhw,
            lower_padding_dhw=lower_padding_dhw,
            stride_dhw=stride_dhw,
            dilation_dhw=dilation_dhw,
            swizzle_size=swizzle_size,
            raster_along=raster_along,
        )

    def can_implement(self, ncdhw, k, ab_dtype, c_dtype) -> bool:
        return self._setup.can_implement(ncdhw, k, ab_dtype, c_dtype)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Materialise kernel arguments from setup attrs and launch the device kernel."""
        p = self._setup
        # Derive the integer mma/stage/byte config.
        p._setup_conv_input_attrs(a, b, c)
        p._setup_attributes()

        # Host TMA descriptors, built directly from the raw input tensors via
        # cuda.create_tensor_map_*.
        T_filter, R_filter, S_filter = p.filter_trs
        upad_d, upad_h, upad_w = p.upper_padding_dhw
        lpad_d, lpad_h, lpad_w = p.lower_padding_dhw
        dil_d, dil_h, dil_w = p.dilation_dhw
        # Per-CTA M tile (use_2cta_instrs splits M between 2 CTAs along cluster_m).
        mma_tiler_per_cta_m = p.mma_tiler[0] // 2 if p.use_2cta_instrs else p.mma_tiler[0]
        # mma_tiler[2] is a sub-tuple (K_tile,) after _setup_attributes.
        k_tile = cute.size(p.mma_tiler, mode=[2])
        # Epilogue subtile width (K-mode of NZPQK), from the shared datapath
        # descriptor. Must match the device epilogue + the C descriptor swizzle
        # below; reading it from _select_epi_datapath (the same source the device
        # epilogue and SMEM sizing use) is what keeps channels_per_pixel and the
        # store_swizzled width in lockstep.
        subtile_n = _select_epi_datapath(mma_tiler_per_cta_m, p.use_2cta_instrs, p.c_dtype).subtile_n

        # Activation A: NDHWC raw, im2col descriptor.
        # stride_order=(4,3,2,1,0): C innermost, then W, H, D, N outermost.
        # lower/upper_corner are stride-ascending spatial dims (W, H, D).
        # Corner formula: lower = -pad_lower, upper = pad_upper -
        # (filter-1)*dilation. The window extent that the hardware walks per
        # output column is fixed by these corners, so a wrong upper drifts
        # every wrapped row by (filter-1) pixels.
        # The convenience ``*_from_view`` builder currently fixes traversal
        # strides to one. Build the activation descriptor explicitly so the
        # hardware advances adjacent output pixels by the convolution stride.
        # TMA orders the compact NDHWC view as C, W, H, D, N and takes global
        # strides in 16-byte units.
        tma_a_desc = cuda.create_tensor_map_im2col(
            global_address=a.iterator.toint(),
            dtype=p.a_dtype,
            global_dims=(a.shape[4], a.shape[3], a.shape[2], a.shape[1], a.shape[0]),
            global_strides=tuple(a.stride[i] * p.a_dtype.width // 128 for i in (3, 2, 1, 0)),
            lower_corner=[-lpad_w, -lpad_h, -lpad_d],
            upper_corner=[
                upad_w - (S_filter - 1) * dil_w,
                upad_h - (R_filter - 1) * dil_h,
                upad_d - (T_filter - 1) * dil_d,
            ],
            channels_per_pixel=k_tile,
            pixels_per_column=mma_tiler_per_cta_m,
            traversal_strides=(1, p.stride_dhw[2], p.stride_dhw[1], p.stride_dhw[0], 1),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )

        # Filter B: KTRSC raw, tiled descriptor.
        # box_dims is in tensor mode order (K, T, R, S, C). When C >= K_tile,
        # the GEMM-K tile fits inside C alone and (T,R,S) box dims are 1.
        n_per_cta = p.mma_tiler[1] // 2 if p.use_2cta_instrs else p.mma_tiler[1]
        tma_b_desc = cuda.create_tensor_map_tiled_from_view(
            b,
            box_dims=(n_per_cta, 1, 1, 1, k_tile),
            stride_order=(4, 3, 2, 1, 0),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )

        # Output C: NZPQK raw, im2col store descriptor with zero corners.
        # channels_per_pixel = subtile_n K-cols stored per TMA call. The swizzle
        # must match the per-CTA-M epilogue SMEM layout: it is derived from the
        # row byte width (subtile_n * c_dtype.width), which is why both subtile_n
        # and the swizzle come from the same source (_select_epi_datapath +
        # _c_smem_swizzle). Mismatch here vs the device store silently reads
        # transposed data, so both sides must derive from the same helper.
        c_swizzle = _to_tensor_map_swizzle(_c_smem_swizzle(p.c_dtype, subtile_n))
        tma_c_desc = cuda.create_tensor_map_im2col_from_view(
            c,
            lower_corner=[0, 0, 0],
            upper_corner=[0, 0, 0],
            channels_per_pixel=subtile_n,
            pixels_per_column=mma_tiler_per_cta_m,
            stride_order=(4, 3, 2, 1, 0),
            swizzle=c_swizzle,
        )

        # Persistent tile scheduler params — integer tile counts. Implicit-GEMM
        # mapping: M = N*Z*P*Q (rows of C), N(gemm) = K (channels of C).
        # c.shape = (N, Z, P, Q, K). cta tile = cta_tile_shape_mnk[:2].
        m_total = c.shape[0] * c.shape[1] * c.shape[2] * c.shape[3]
        n_total = c.shape[4]
        cta_m = p.cta_tile_shape_mnk[0]
        cta_n = p.cta_tile_shape_mnk[1]
        num_ctas_mnl = (
            (m_total + cta_m - 1) // cta_m,
            (n_total + cta_n - 1) // cta_n,
            1,
        )
        cluster_shape_mnl = (*p.preferred_cluster_shape_mn, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        # mma_inst_shape_mnk — per-MMA atomic instruction shape, used by the
        # kernel for the SMEM-desc K-block stride. tcgen05 atomic shape is
        # (mma_tiler_M, mma_tiler_N, K_inst); K_inst = 256 // dtype.width
        # (FP16/BF16->16, FP8->32, TF32->8).
        mma_inst_shape_mnk = (
            p.mma_tiler[0],
            p.mma_tiler[1],
            256 // p.a_dtype.width,
        )

        # GEMM-K tile count: GEMM-K = T*R*S*C, tiled by mma_tiler_k.
        mma_tiler_k = cute.size(p.mma_tiler, mode=[2])
        gemm_k = T_filter * R_filter * S_filter * a.shape[4]
        k_tile_cnt = (gemm_k + mma_tiler_k - 1) // mma_tiler_k

        # Output spatial dims for the im2col producer/epilogue coord decode.
        # c.shape = (N, Z, P, Q, K); filter_trs / padding / stride / dilation
        # live on the PersistentConvSetup instance.
        zpq = (c.shape[1], c.shape[2], c.shape[3])

        # Launch the kernel (argument order matches the device kernel
        # signature).
        _kernel(
            tile_sched_params,
            p.mma_tiler,
            mma_inst_shape_mnk,
            k_tile_cnt,
            p.num_ab_stage,
            p.num_acc_stage,
            p.num_c_stage,
            p.use_2cta_instrs,
            p.a_dtype,
            p.c_dtype,
            p.acc_dtype,
            p.lower_padding_dhw,
            p.stride_dhw,
            p.dilation_dhw,
            zpq,
            p.filter_trs,
            tma_a_desc,
            tma_b_desc,
            tma_c_desc,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[p.threads_per_cta, 1, 1],
            cluster=(*p.preferred_cluster_shape_mn, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )
        return


# ---------------------------------------------------------------------------
# Compile factory — required by the kernel_lint / test_kernel_compiles contract.
# ---------------------------------------------------------------------------

# Default config.
_DEFAULT_NCDHW: Tuple[int, int, int, int, int] = (1, 64, 6, 10, 10)
_DEFAULT_KTRS: Tuple[int, int, int, int] = (256, 3, 3, 3)
_DEFAULT_UPPER_PAD: Tuple[int, int, int] = (0, 0, 0)
_DEFAULT_LOWER_PAD: Tuple[int, int, int] = (0, 0, 0)
_DEFAULT_STRIDE: Tuple[int, int, int] = (1, 1, 1)
_DEFAULT_DILATION: Tuple[int, int, int] = (1, 1, 1)
_DEFAULT_DTYPE: Type[Numeric] = cutlass.BFloat16


def _dense_config_violation(tile_config: ConvTileConfig, channel_bytes: int) -> str | None:
    """Return why a tile cannot implement dense convolution, if anything."""
    if tile_config.cta_tile_k_bytes not in (64, 128):
        return f"CTA K must be 64 or 128 bytes, got {tile_config.cta_tile_k_bytes}"
    if channel_bytes <= 0:
        return f"channel_bytes must be positive, got {channel_bytes}"
    if channel_bytes % tile_config.cta_tile_k_bytes:
        return f"channel_bytes={channel_bytes} must be divisible by CTA K={tile_config.cta_tile_k_bytes} bytes"
    if tile_config.cta_tile_m == 64 and tile_config.cta_group == 2 and tile_config.cta_tile_n % 64:
        return f"M=64 2-CTA requires CTA N divisible by 64, got {tile_config.cta_tile_n}"
    return None


def _is_dense_selection_candidate(tile_config: ConvTileConfig, implicit_m: int, output_n: int, channel_bytes: int) -> bool:
    """Apply dense compatibility plus the automatic-selection axes."""
    n_choices = (128, 256) if output_n >= 512 else (32, 64, 128, 256)
    return (
        _dense_config_violation(tile_config, channel_bytes) is None
        and tile_config.cta_tile_m == 128
        and tile_config.cta_group == (2 if implicit_m > 128 else 1)
        and tile_config.cta_tile_n in n_choices
    )


@lru_cache(maxsize=None)
def compile(
    ncdhw: Tuple[int, int, int, int, int] = _DEFAULT_NCDHW,
    ktrs: Tuple[int, int, int, int] = _DEFAULT_KTRS,
    upper_padding_dhw: Tuple[int, int, int] = _DEFAULT_UPPER_PAD,
    lower_padding_dhw: Tuple[int, int, int] = _DEFAULT_LOWER_PAD,
    stride_dhw: Tuple[int, int, int] = _DEFAULT_STRIDE,
    dilation_dhw: Tuple[int, int, int] = _DEFAULT_DILATION,
    epilogue: str = "identity",
    epilogue_attrs: tuple[tuple[str, float], ...] = (),
    ab_dtype: Type[Numeric] = _DEFAULT_DTYPE,
    c_dtype: Type[Numeric] = _DEFAULT_DTYPE,
    tile_config: ConvTileConfig | None = None,
):  # noqa: A001
    """Compile the SM100 fprop kernel for the requested convolution.

    The shape is explicit because im2col TMA descriptors bake it into the
    artifact. Padding, stride, and dilation are likewise plan-time
    specializations. Input/output storage dtypes and an explicitly supplied
    ``tile_config`` are specialized independently; when no config is supplied,
    the shared predicate-based selector chooses one from the implicit-GEMM
    shape. The accumulator remains FP32.
    """
    acc_dtype = cutlass.Float32
    epilogue_op = get_epilogue_op(epilogue, epilogue_attrs)

    N, C, D, H, W = ncdhw
    K, T, R, S = ktrs
    Z, P, Q = _compute_zpq((D, H, W), (T, R, S), stride_dhw, upper_padding_dhw, lower_padding_dhw, dilation_dhw)
    implicit_m = N * Z * P * Q
    implicit_k = T * R * S * C
    channel_bytes = C * ab_dtype.width // 8
    if tile_config is None:
        tile_config = _get_config(
            implicit_m,
            K,
            implicit_k,
            predicate=lambda config: _is_dense_selection_candidate(config, implicit_m, K, channel_bytes),
        )
    else:
        violation = _dense_config_violation(tile_config, channel_bytes)
        if violation is not None:
            raise ValueError(f"dense convolution config {tile_config.name} is incompatible: {violation}")

    # NDHWC + KTRSC + NZPQK fake tensors (channel/K-fastest, contiguous layout).
    fake_a = make_fake_compact_tensor(ab_dtype, (N, D, H, W, C), stride_order=(4, 3, 2, 1, 0), assumed_align=16)
    fake_b = make_fake_compact_tensor(ab_dtype, (K, T, R, S, C), stride_order=(4, 3, 2, 1, 0), assumed_align=16)
    fake_c = make_fake_compact_tensor(c_dtype, (N, Z, P, Q, K), stride_order=(4, 3, 2, 1, 0), assumed_align=16)

    fprop_op = _FpropCTMKernel(
        acc_dtype=acc_dtype,
        tile_config=tile_config,
        filter_trs=ktrs[1:],
        upper_padding_dhw=upper_padding_dhw,
        lower_padding_dhw=lower_padding_dhw,
        stride_dhw=stride_dhw,
        dilation_dhw=dilation_dhw,
    )
    if not fprop_op.can_implement(ncdhw, ktrs[0], ab_dtype, c_dtype):
        raise testing.CantImplementError(
            f"Config {tile_config.name} not implementable: ncdhw={ncdhw}, ktrs={ktrs}, upper_padding_dhw={upper_padding_dhw}, "
            f"lower_padding_dhw={lower_padding_dhw}, stride_dhw={stride_dhw}, dilation_dhw={dilation_dhw}"
        )

    cluster_size = tile_config.cluster_shape_mn[0] * tile_config.cluster_shape_mn[1]
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
    return cute.compile(
        fprop_op,
        fake_a,
        fake_b,
        fake_c,
        max_active_clusters,
        make_fake_stream(),
        epilogue_op,
        options="--enable-tvm-ffi",
    )
