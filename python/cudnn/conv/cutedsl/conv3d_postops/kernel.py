# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 fused Conv3D epilogues for video-VAE inference."""

import math
from dataclasses import dataclass

import cutlass
from cuda.bindings import driver as cuda_driver
from cutlass import cute
from cutlass.experimental import cuda
from cutlass.experimental import primitives as prims

from cudnn._cutlass_helpers.static_persistent_tile_scheduler import (
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
)


def _contiguous_tma_layout(shape: tuple[int, ...], dtype: type[cutlass.Numeric]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Convert a C-contiguous shape to TMA dimensions and 16-byte strides.

    NDHWC becomes C,W,H,D,N; KTRSC becomes C,S,R,T,K.
    """
    if not 1 <= len(shape) <= 5 or any(type(extent) is not int or extent <= 0 for extent in shape):
        raise ValueError("Tensor rank must be between 1 and 5")
    dims = shape[::-1]
    stride_bytes = dtype.width // 8
    strides = []
    for extent in dims[:-1]:
        stride_bytes *= extent
        if stride_bytes % 16:
            raise ValueError("TMA inter-dimension strides must be multiples of 16 bytes")
        strides.append(stride_bytes // 16)
    return dims, tuple(strides)


@cute.kernel
def _conv3d_postops_kernel(
    # Leading specialization label remains visible before DSL name truncation.
    operation: cutlass.Constexpr[str],
    # Persistent tile scheduler parameters
    tile_sched_params: PersistentTileSchedulerParams,
    # Constexpr knobs
    mma_tiler: cutlass.Constexpr[tuple[int, int, int]],
    mma_inst_shape_mnk: cutlass.Constexpr[tuple[int, int, int]],
    # Host-computed: T*R*S*ceil_div(Ci, 64), including each channel tail.
    k_tile_cnt: cutlass.Constexpr[int],
    num_ab_stage: cutlass.Constexpr[int],
    num_c_stage: cutlass.Constexpr[int],
    use_2cta_instrs: cutlass.Constexpr[bool],
    # Output spatial dimensions (Z, P, Q) decompose linear M into (n, z, p, q).
    zpq: cutlass.Constexpr[tuple[int, int, int]],
    batches: cutlass.Constexpr[int],
    # Host-built TMA descriptors, consumed by the device-side
    # prims.cp_async_bulk_tensor_* / prims.prefetch_tensormap calls.
    tma_a_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_c_desc: cutlass.GridConstant[cuda.TensorMap],
    conv_bias: cute.Tensor = None,
    gamma: cute.Tensor = None,
    packed_x: cute.Tensor = None,
    packed_shape: cutlass.Constexpr = None,
    prep_padded: cute.Tensor = None,
    prep_cache: cute.Tensor = None,
    PREP_SHAPE: cutlass.Constexpr = None,
    residual: cute.Tensor = None,
    residual_bias: cute.Tensor = None,
    prep_residual: cute.Tensor = None,
    SPATIAL_SHAPE: cutlass.Constexpr = None,
) -> None:
    """Run valid 3x3x3 BF16 convolution with optional fused post-operations.

    Stride and dilation are fixed at one; any input padding is caller-provided.
    Tile geometry and pipeline stages are compile-time parameters. The host
    supplies matching tensor-map descriptors and the reduction tile count.
    """
    FUSE_NORM: cutlass.Constexpr = gamma is not None
    ab_dtype = cutlass.BFloat16
    c_dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    num_acc_stage = 2
    trs = (3, 3, 3)
    lower_pad_dhw = (0, 0, 0)
    stride_dhw = (1, 1, 1)
    dilation_dhw = (1, 1, 1)
    if cutlass.const_expr(mma_tiler[0] != (256 if use_2cta_instrs else 128)):
        raise ValueError("Convolution core requires 128 rows per CTA")
    if cutlass.const_expr(mma_tiler[1] != 160):
        raise ValueError("Fused convolution requires tile N=160")
    if cutlass.const_expr(mma_tiler[2] != 64):
        raise ValueError("Convolution core requires tile K=64")
    if cutlass.const_expr(mma_inst_shape_mnk != (mma_tiler[0], mma_tiler[1], 16)):
        raise ValueError("MMA instruction shape must be (tileM, tileN, 16)")
    if cutlass.const_expr(num_ab_stage < 1 or num_c_stage < 1 or num_c_stage > 8):
        raise ValueError("AB stages must be positive; C stages must be in [1, 8]")
    if cutlass.const_expr(k_tile_cnt <= 0):
        raise ValueError("Convolution core requires a positive K tile count")
    if cutlass.const_expr(packed_shape is not None):
        if cutlass.const_expr(use_2cta_instrs or FUSE_NORM or k_tile_cnt != 7):
            raise ValueError("Packed C12 gather requires raw 1CTA and seven K tiles")
        if cutlass.const_expr(packed_x is None):
            raise ValueError("Packed C12 gather requires an input tensor")
        if cutlass.const_expr(zpq != tuple(d - 2 for d in packed_shape[1:])):
            raise ValueError("Packed gather output geometry must match the padded input")
    if cutlass.const_expr(FUSE_NORM):
        if cutlass.const_expr(mma_tiler[1] != 160):
            raise ValueError("Fused normalization requires tileN=160")
        if cutlass.const_expr(conv_bias is None or gamma is None):
            raise ValueError("Fused normalization requires conv_bias and gamma")
        if cutlass.const_expr(conv_bias.element_type != cutlass.BFloat16 or gamma.element_type != cutlass.BFloat16):
            raise TypeError("Fused conv_bias and gamma must be BF16")
        if cutlass.const_expr(conv_bias.shape not in ((160,), (320,)) or gamma.shape != conv_bias.shape):
            raise ValueError("Fused conv_bias and gamma must have shape (160,) or (320,)")
    if cutlass.const_expr(PREP_SHAPE is not None):
        if cutlass.const_expr(not FUSE_NORM):
            raise ValueError("Next-convolution preparation requires normalization")
        if cutlass.const_expr(prep_padded is None or prep_cache is None):
            raise ValueError("Preparation requires padded/cache outputs")
    if cutlass.const_expr(residual is not None):  # noqa: SIM102 - preserve DSL specialization guard
        if cutlass.const_expr(SPATIAL_SHAPE is None and (not FUSE_NORM or prep_residual is None)):
            raise ValueError("Residual fusion requires normalization and a saved-sum output")
    if cutlass.const_expr(SPATIAL_SHAPE is not None and (FUSE_NORM or PREP_SHAPE is not None or residual is None or conv_bias is None or prep_padded is None)):
        raise ValueError("Spatial residual output requires bias/residual and output without norm/history")

    norm_channels: cutlass.Constexpr = gamma.shape[0] if FUSE_NORM else 160
    paired_norm: cutlass.Constexpr = FUSE_NORM and norm_channels == 320
    if cutlass.const_expr(paired_norm and not use_2cta_instrs):
        raise ValueError("C320 normalization requires 2CTA MMA")

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

    # Wide TMEM loads let four lanes share each fused-normalization row.
    epilogue_warp_ids = tuple(range((16 if residual is not None or paired_norm else 8) if FUSE_NORM else 4))
    mma_warp_id = len(epilogue_warp_ids)
    tma_warp_id = mma_warp_id + 1

    # Prefetch the three TMA descriptors from the MMA warp; the descriptor cache
    # is shared across the cluster.
    if warp_idx == mma_warp_id:
        if cutlass.const_expr(packed_shape is None):
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
    # ptr), then A/B as Int8 byte arrays and C as a BF16 array.
    # cutlass.Array offsets an element/byte view with
    # ``.subview(n)`` and hands out a raw Pointer with ``.data_ptr(n)`` (the
    # latter is masked for bit-24 CTA_2 routing).
    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, num_acc_stage, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, num_acc_stage, space=cutlass.AddressSpace.smem)
    tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    # Per-CTA per-stage SMEM byte sizes; host descriptors must agree.
    _m_per_cta = mma_tiler[0] // _atom_thr
    _n_per_cta = mma_tiler[1] // _atom_thr
    _k_tile = cute.size(mma_tiler, mode=[2])
    _a_stage_bytes = _m_per_cta * _k_tile * (ab_dtype.width // 8)
    _b_stage_bytes = _n_per_cta * _k_tile * (ab_dtype.width // 8)
    # Both epilogue mappings publish the same 128-row, 32-channel BF16 tile.
    _epi_subtile_n = 32
    _c_tile_rows = 128
    _c_stage_bytes = _c_tile_rows * _epi_subtile_n * (c_dtype.width // 8)

    # A/B byte arrays and the BF16 C array are all 1024B aligned.
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
    sC = cutlass.Array(
        c_dtype,
        (_c_stage_bytes // (c_dtype.width // 8)) * num_c_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    if cutlass.const_expr(paired_norm):
        norm_params = cutlass.Array(cutlass.BFloat16, 640, space=cutlass.AddressSpace.smem, alignment=16)
        if tidx < 160:
            norm_params.store(
                (conv_bias.iterator.raw_ptr() + tidx * 2).load(count=2, alignment=4),
                tidx * 2,
            )
            norm_params.store(
                (gamma.iterator.raw_ptr() + tidx * 2).load(count=2, alignment=4),
                320 + tidx * 2,
            )
        # Publish ordinary SMEM loads independently of the async TMA barriers.
        prims.barrier_cta_sync(
            3,
            thread_count=32 * (len(epilogue_warp_ids) + 2 + (4 if PREP_SHAPE is not None else 0)),
        )

    # mbarrier_init: one warp / one lane initializes every barrier.
    # Each epilogue warp in each cooperating CTA releases the accumulator.
    num_acc_empty_arrives = len(epilogue_warp_ids) * (2 if use_2cta_instrs else 1)
    if warp_idx == 0 and prims.elect_sync():
        prims.mbarrier_init(tmem_dealloc_mbar_ptr, cute.arch.WARP_SIZE)
        for i in cutlass.range_constexpr(num_acc_stage):
            prims.mbarrier_init(acc_empty_mbar_ptr.subview(i), num_acc_empty_arrives)
            prims.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
        for i in cutlass.range_constexpr(num_ab_stage):
            prims.mbarrier_init(
                ab_full_mbar_ptr.subview(i),
                128 if packed_shape is not None else 1,
            )
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
    # per-instruction (M, N, K) shape, e.g. (256, 32, 16) for BF16/2cta.
    idesc = prims.Tcgen05InstrDesc.build(
        a_dtype=ab_dtype,
        b_dtype=ab_dtype,
        c_dtype=acc_dtype,
        m_dim=mma_inst_shape_mnk[0],
        n_dim=mma_inst_shape_mnk[1],
    )
    # BF16 uses the F16 instruction kind; idesc selects BF16 operands.
    mma_kind = prims.Tcgen05MMAKind.F16

    # Persistent tile scheduler.
    tile_sched = StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
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
    producer_warp_end = 9 if cutlass.const_expr(packed_shape is not None) else tma_warp_id + 1
    auxiliary_threads = 64 if FUSE_NORM and not paired_norm else 128
    if warp_idx >= producer_warp_end:
        if cutlass.const_expr(SPATIAL_SHAPE is not None):
            batches, channels = SPATIAL_SHAPE
            z, h, w = zpq
            border_rows = h + w + 1
            bx, by, bz = cute.arch.block_idx()
            gx, gy, gz = cute.arch.grid_dim()
            linear_cta = cutlass.Int64(bx) + cutlass.Int64(gx) * (cutlass.Int64(by) + cutlass.Int64(gy) * bz)
            total_ctas = cutlass.Int64(gx) * gy * gz
            vector_idx = linear_cta * auxiliary_threads + tidx - producer_warp_end * 32
            while vector_idx < batches * z * border_rows * (channels // 8):
                channel = vector_idx % (channels // 8) * 8
                border_row = vector_idx // (channels // 8)
                nt = border_row // border_rows
                border = border_row % border_rows
                ph = cutlass.Int64(h)
                pw = border
                if border >= w + 1:
                    ph = border - w - 1
                    pw = cutlass.Int64(w)
                offset = ((nt * (h + 1) + ph) * (w + 1) + pw) * channels + channel
                zeros = cutlass.vector.full((8,), 0, cutlass.BFloat16)
                (prep_padded.iterator.raw_ptr() + offset).store(zeros, alignment=16)
                vector_idx += total_ctas * auxiliary_threads
        if cutlass.const_expr(PREP_SHAPE is not None):
            # The caller owns history interiors. Initialize only missing
            # history planes and spatial borders, disjoint from TMA stores.
            Z_out, P_out, Q_out = zpq
            batches, previous_frames = PREP_SHAPE
            padded_h = P_out + 2
            padded_w = Q_out + 2
            history_rows = (2 - previous_frames) * padded_h * padded_w
            border_rows = 2 * padded_w + 2 * P_out
            auxiliary_rows = history_rows + (Z_out + previous_frames) * border_rows
            bx, by, bz = cute.arch.block_idx()
            gx, gy, gz = cute.arch.grid_dim()
            # The persistent scheduler places clusters along grid Z, not X.
            # Row/vector coordinates fit in 32 bits for encoder shapes. Widen
            # only element offsets, which can exceed 2 Gi elements after padding.
            index_type: cutlass.Constexpr = cutlass.Int32 if batches * auxiliary_rows * (norm_channels // 8) < 2**31 else cutlass.Int64
            linear_cta = cutlass.Int64(bx) + cutlass.Int64(gx) * (cutlass.Int64(by) + cutlass.Int64(gy) * bz)
            total_ctas = cutlass.Int64(gx) * gy * gz
            vector_idx = linear_cta * auxiliary_threads + tidx - producer_warp_end * 32
            while vector_idx < batches * auxiliary_rows * (norm_channels // 8):
                coordinate_idx = index_type(vector_idx)
                channel = coordinate_idx % (norm_channels // 8) * 8
                auxiliary_row = coordinate_idx // (norm_channels // 8)
                batch = auxiliary_row // auxiliary_rows
                local_row = auxiliary_row % auxiliary_rows
                pt = index_type(0)
                ph = index_type(0)
                pw = index_type(0)
                if local_row < history_rows:
                    pt = local_row // (padded_h * padded_w)
                    ph = local_row // padded_w % padded_h
                    pw = local_row % padded_w
                else:
                    border = (local_row - history_rows) % border_rows
                    pt = (local_row - history_rows) // border_rows + 2 - previous_frames
                    if border < 2 * padded_w:
                        ph = border // padded_w * (P_out + 1)
                        pw = border % padded_w
                    else:
                        ph = (border - 2 * padded_w) // 2 + 1
                        pw = (border - 2 * padded_w) % 2 * (Q_out + 1)
                values = cutlass.vector.full((8,), 0, cutlass.BFloat16)
                destination = (((cutlass.Int64(batch) * (Z_out + 2) + pt) * padded_h + ph) * padded_w + pw) * norm_channels + channel
                (prep_padded.iterator.raw_ptr() + destination).store(values, alignment=16)
                vector_idx += total_ctas * auxiliary_threads
    elif warp_idx >= tma_warp_id:
        if cutlass.const_expr(packed_shape is not None):
            # Four producer warps gather the C16-padded activation into the
            # K64 shared-memory tiles consumed by the tensor-core pipeline.
            ab_stage_idx = 0
            producer_tid = tidx - 160
            batch_count, input_t, input_h, input_w = packed_shape
            Z_out, P_out, Q_out = zpq
            total_rows = batch_count * Z_out * P_out * Q_out
            while work_tile.is_valid_tile:
                m_start = work_tile.tile_idx[0] * 128
                n_start = work_tile.tile_idx[1] * mma_tiler[1]
                for k in cutlass.range(7, unroll=1):
                    full = ab_full_mbar_ptr.subview(ab_stage_idx)
                    empty = ab_empty_mbar_ptr.subview(ab_stage_idx)
                    while not prims.mbarrier_try_wait_parity(empty, ab_empty_phase_bit, time_limit=10000000):
                        pass
                    if producer_tid == 0:
                        prims.mbarrier_expect_tx(full, b_bytes_per_stage)
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            sB.subview(ab_stage_idx * b_bytes_per_stage),
                            tma_b_desc.get_ptr(),
                            (k * 64, n_start),
                            full,
                            [],
                            group=prims.CTAGroup.CTA_1,
                        )
                    local_k = producer_tid % 8 * 8
                    kk = k * 64 + local_k
                    channel = kk % 16
                    tap = kk // 16
                    dw = tap % 3
                    dh = tap // 3 % 3
                    dt = tap // 9
                    a_ptr = cutlass.inttoptr(
                        sA.data_ptr().toint(),
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    )
                    for chunk in cutlass.range_constexpr(8):
                        local_row = producer_tid // 8 + chunk * 16
                        row = m_start + local_row
                        q = row % Q_out
                        p = row // Q_out % P_out
                        z = row // (Q_out * P_out) % Z_out
                        batch = row // (Q_out * P_out * Z_out)
                        offset = cutlass.Int64(0)
                        source_bytes = cutlass.Int32(0)
                        if row < total_rows and tap < 27:
                            offset = cutlass.Int64((((batch * input_t + z + dt) * input_h + p + dh) * input_w + q + dw) * 16 + channel)
                            source_bytes = cutlass.Int32(16)
                        swizzled = local_row * 64 + (local_k ^ (local_row % 8 * 8))
                        prims.cp_async_shared_global(
                            a_ptr + ab_stage_idx * 8192 + swizzled,
                            packed_x.iterator.raw_ptr() + offset,
                            16,
                            "ca",
                            cp_size=source_bytes,
                        )
                    prims.cp_async_mbarrier_arrive(full, noinc=True)
                    ab_stage_idx += 1
                    if ab_stage_idx == num_ab_stage:
                        ab_stage_idx = 0
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        else:
            ab_stage_idx = 0
            Z_out, P_out, Q_out = zpq
            T_filt, R_filt, S_filt = trs
            pad_d, pad_h, pad_w = lower_pad_dhw
            str_d, str_h, str_w = stride_dhw
            dil_d, dil_h, dil_w = dilation_dhw
            while work_tile.is_valid_tile:
                for channel_half in cutlass.range_constexpr(2 if paired_norm else 1):
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // (2 if use_2cta_instrs else 1),
                        channel_half if paired_norm else cur_tile_coord[1],
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

                        # Acquire after the MMA consumer releases this stage.
                        # try_wait_parity issues a single non-blocking attempt that may
                        # hardware-suspend up to time_limit then return False;
                        # a blocking wait must retry in a loop.
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
        # MMA and all epilogue warps participate in this barrier so the MMA warp
        # can pick up the tmem_ptr written by the allocator (epilogue warp 0).
        # This subset has 160 raw or 288 fused threads, excluding producers.
        tmem_bar_id = 1
        tmem_bar_threads = 32 * (1 + len(epilogue_warp_ids))
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), acc_dtype)

        # AB / acc pipeline consumer state — tracks the current MMA stage.
        ab_stage_idx = 0
        acc_stage_idx = 0

        # tcgen05 SMEM descriptors for stage 0. tcgen05_mma encodes SMEM
        # addresses in 16-byte units, so per-stage / per-K-block increments
        # are bytes >> 4. (leading=16, stride=1024, layout=2) is the 128B-swizzle
        # K-major layout, matching the cuda.TensorMapSwizzle.s128b on the A/B
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
        # Four K=16 instructions consume each K=64 AB stage.
        num_k_blocks = cute.size(mma_tiler, mode=[2]) // mma_inst_shape_mnk[2]

        # Persistent loop: outer = work tiles (acc stages); inner = K-tiles.
        while work_tile.is_valid_tile:
            for channel_half in cutlass.range_constexpr(2 if paired_norm else 1):
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
                    # Order the epilogue's completed TMEM reads before reuse.
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

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
                        while not prims.mbarrier_try_wait_parity(
                            ab_full_mbar_ptr_stage,
                            ab_full_phase_bit,
                            time_limit=10000000,
                        ):
                            pass
                        if cutlass.const_expr(packed_shape is not None):
                            # Publish the cp.async gather to the MMA async proxy.
                            cute.arch.fence_view_async_shared()
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

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

    # ---- Epilogue warps: raw 0-3, fused 0-7 ----
    elif warp_idx < mma_warp_id:
        # Per-CTA M tile size — 2cta cluster halves mma_tiler[0] across the pair.
        mma_tiler_per_cta_m = mma_tiler[0] // 2 if cutlass.const_expr(use_2cta_instrs) else mma_tiler[0]
        # Raw warps own 32 rows; fused warps own 16 channel-cooperative rows.
        subtile_n = _epi_subtile_n
        subtile_cnt = mma_tiler[1] // subtile_n

        # Sync ids — tmem_bar_id=1 already used by the MMA consumer warp.
        # epilog_sync_bar_id=2 is separate and covers all epilogue warps.
        threads_in_epilogue = 32 * len(epilogue_warp_ids)
        epilog_sync_bar_id = 2
        allocator_warp_id = epilogue_warp_ids[0]
        tmem_bar_id = 1
        tmem_bar_threads = 32 * (1 + len(epilogue_warp_ids))

        # 128-bit (16-byte) SMEM store width — store_swizzled vectorizes per lane.
        vsize = 128 // c_dtype.width
        # Keep the same elected lane through every store and the final drain.
        store_issuer = prims.elect_sync()

        # C-side im2col store coords are computed by hand. The im2col STORE op
        # has no im2col-offset operand (unlike the A LOAD), and the C descriptor
        # uses zero corners, so output spatial coords are bare pixels (no pad
        # subtraction, no stride multiply). Decompose the linear per-CTA M index
        # into (n, z, p, q) per work-tile in the loop.
        Z_out, P_out, Q_out = zpq

        if cutlass.const_expr(PREP_SHAPE is not None):
            batches, previous_frames = PREP_SHAPE
            cache_frames = min(2, Z_out + previous_frames)

        # Allocator warp (epi 0) reserves 512 TMEM cols and stashes the pointer
        # in tmem_ptr_i32. Every CTA allocates from its own 512-col bank;
        # _cta_group arranges peer-side state for the 2-CTA group (no-op for
        # CTA_1). This op is warp-collective (NOT elect-safe) and must stay
        # outside any rank-divergent branch, so it is unconditional here.
        if warp_idx == allocator_warp_id:
            prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_cols, group=_cta_group)

        # The MMA consumer warp also waits with all epilogue warps, so it
        # picks up the same tmem_ptr right after the allocator publishes it.
        prims.barrier_cta_sync(tmem_bar_id, thread_count=tmem_bar_threads)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

        tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), acc_dtype)
        tmem_raw_addr = tmem_ptr_i32.load()

        # Persistent loop: outer over work-tiles (one per acc stage), inner over
        # epilogue subtiles within each tile.
        acc_stage_idx = 0
        epi_stage_idx = 0
        while work_tile.is_valid_tile:
            if cutlass.const_expr(paired_norm):
                # Both N160 halves belong to this M task. Keep BF16 N0 values
                # and contiguous-channel FP32 partials live until N1 arrives.
                retained = cutlass.Array(cutlass.Uint32, 40, space=cutlass.AddressSpace.rmem)
                partials = cutlass.Array(cutlass.Float32, 32, space=cutlass.AddressSpace.rmem)
            for channel_half in cutlass.range_constexpr(2 if paired_norm else 1):
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
                    channel_half if paired_norm else cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Wait for MMA warp to commit acc_full for this stage.
                # try_wait_parity is one non-blocking attempt; loop until the MMA
                # warp's commit advances the phase.
                while not prims.mbarrier_try_wait_parity(acc_full_mbar_ptr_stage, current_full_phase_bit, time_limit=10000000):
                    pass
                # The MMA completion handshake precedes this warp's TMEM loads.
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

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

                if cutlass.const_expr(FUSE_NORM):
                    # Residual loads need more latency hiding: sixteen warps own
                    # one row/lane, versus eight warps and two rows for norm-only.
                    rows_per_lane: cutlass.Constexpr = 1 if residual is not None or paired_norm else 2
                    lane = tidx % 32
                    lane_col = lane % 4
                    # Warp rank modulo four fixes the accessible 32-row TMEM band.
                    # The second warpgroup drains its upper 16 rows.
                    row_base = (warp_idx % 4) * 32 + ((warp_idx // 4) % 2) * 16
                    drain_half = warp_idx // 8
                    if cutlass.const_expr(not paired_norm):
                        retained = cutlass.Array(
                            cutlass.Uint32,
                            20 * rows_per_lane,
                            space=cutlass.AddressSpace.rmem,
                        )
                    if cutlass.const_expr(residual is not None):
                        # Drain before global residual traffic. Reuse the same packed
                        # words for raw values, then overwrite them with summed values.
                        for drain_subtile in cutlass.range_constexpr(5):
                            drain_addr = (((tmem_raw_addr >> 16) + row_base) << 16) | (base_col_id + drain_subtile * 32)
                            drain_ptr = cutlass.inttoptr(drain_addr, 6, cutlass.Float32)
                            drain_values = prims.tcgen05_ld("16x256b", drain_ptr, num=4)
                            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                            drain_selected = cutlass.Array(cutlass.BFloat16, 8, space=cutlass.AddressSpace.rmem)
                            if drain_half == 0:
                                for j in cutlass.range_constexpr(8):
                                    drain_selected[j] = drain_values[(j // 2) * 4 + j % 2].to(cutlass.BFloat16)
                            else:
                                for j in cutlass.range_constexpr(8):
                                    drain_selected[j] = drain_values[(j // 2) * 4 + j % 2 + 2].to(cutlass.BFloat16)
                            retained.store(
                                drain_selected.load(0, 8).bitcast(cutlass.Uint32),
                                (channel_half * 5 + drain_subtile) * 4,
                            )
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if prims.elect_sync():
                            drain_leader = (cta_rank_in_cluster // _atom_thr) * _atom_thr
                            prims.mbarrier_arrive(
                                prims.mapa(acc_empty_mbar_ptr_stage, drain_leader),
                                count=1,
                                scope=prims.MemScope.CLUSTER,
                            )
                    if cutlass.const_expr(not paired_norm):
                        partials = cutlass.Array(
                            cutlass.Float32,
                            32 * rows_per_lane,
                            space=cutlass.AddressSpace.rmem,
                        )
                    if cutlass.const_expr(channel_half == 0):
                        for part in cutlass.range_constexpr(32 * rows_per_lane):
                            partials[part] = cutlass.Float32(0.0)
                    for norm_subtile in cutlass.range_constexpr(5):
                        rounded = cutlass.Array(
                            cutlass.BFloat16,
                            8 * rows_per_lane,
                            space=cutlass.AddressSpace.rmem,
                        )
                        if cutlass.const_expr(residual is None):
                            norm_addr = (((tmem_raw_addr >> 16) + row_base) << 16) | (base_col_id + norm_subtile * 32)
                            norm_ptr = cutlass.inttoptr(norm_addr, 6, cutlass.Float32)
                            norm_rmem = prims.tcgen05_ld("16x256b", norm_ptr, num=4)
                            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                            if cutlass.const_expr(paired_norm):
                                if drain_half == 0:
                                    for j in cutlass.range_constexpr(8):
                                        rounded[j] = norm_rmem[(j // 2) * 4 + j % 2].to(cutlass.BFloat16)
                                else:
                                    for j in cutlass.range_constexpr(8):
                                        rounded[j] = norm_rmem[(j // 2) * 4 + j % 2 + 2].to(cutlass.BFloat16)
                            else:
                                rounded.store(norm_rmem.to(cutlass.BFloat16), 0)
                        else:
                            rounded.store(
                                retained.load((channel_half * 5 + norm_subtile) * 4, 4).bitcast(cutlass.BFloat16),
                                0,
                            )
                        math_width = 1 if paired_norm else 2
                        for j in cutlass.range_constexpr(8 * rows_per_lane // math_width):
                            scalar = j * math_width
                            channel = channel_half * 160 + norm_subtile * 32 + (scalar // (2 * rows_per_lane)) * 8 + lane_col * 2 + scalar % 2
                            if cutlass.const_expr(paired_norm):
                                bias_pair = (norm_params.data_ptr() + channel).load(count=math_width, alignment=2 * math_width)
                            else:
                                bias_pair = (conv_bias.iterator.raw_ptr() + channel).load(count=2, alignment=4)
                            value = rounded.load(scalar, math_width).to(cutlass.Float32)
                            rounded.store(
                                (value + bias_pair.to(cutlass.Float32)).to(cutlass.BFloat16),
                                scalar,
                            )
                        if cutlass.const_expr(residual is not None):
                            for pair in cutlass.range_constexpr(4):
                                skip_row = row_base + lane // 4 + drain_half * 8
                                skip_pixel = cutlass.Int64(m_off_cta) + skip_row
                                if skip_pixel < batches * Z_out * P_out * Q_out:
                                    channel = channel_half * 160 + norm_subtile * 32 + pair * 8 + lane_col * 2
                                    offset = skip_pixel * norm_channels + channel
                                    skip = (residual.iterator.raw_ptr() + offset).load(count=2, alignment=4)
                                    if cutlass.const_expr(residual_bias is not None):
                                        skip_bias = (residual_bias.iterator.raw_ptr() + channel).load(count=2, alignment=4)
                                        skip = (skip.to(cutlass.Float32) + skip_bias.to(cutlass.Float32)).to(cutlass.BFloat16)
                                    summed_pair = (rounded.load(pair * 2, 2).to(cutlass.Float32) + skip.to(cutlass.Float32)).to(cutlass.BFloat16)
                                    rounded.store(summed_pair, pair * 2)
                                    (prep_residual.iterator.raw_ptr() + offset).store(summed_pair, alignment=4)
                        for j in cutlass.range_constexpr(8 * rows_per_lane // math_width):
                            scalar = j * math_width
                            value = rounded.load(scalar, math_width).to(cutlass.Float32)
                            # Torch accumulates c, c+128, c+256 independently
                            # for each of four adjacent channels per virtual lane.
                            # Physical lanes retain two channels from each group
                            # of eight; neighboring pairs complete a virtual lane.
                            channel_group = (channel_half * 160 + norm_subtile * 32 + (scalar // (2 * rows_per_lane)) * 8) // 8 % 16
                            part = ((scalar // 2) % rows_per_lane) * 32 + channel_group * 2 + scalar % 2
                            partials.store(partials.load(part, math_width) + value * value, part)
                        retained.store(
                            rounded.load(0, 8 * rows_per_lane).bitcast(cutlass.Uint32),
                            (channel_half * 5 + norm_subtile) * 4 * rows_per_lane,
                        )

                    # All accumulators are now register-owned. Release TMEM before
                    # normalization and stores, allowing MMA to reuse this stage.
                    if cutlass.const_expr(residual is None):
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if prims.elect_sync():
                            leader_cta_rank = (cta_rank_in_cluster // _atom_thr) * _atom_thr
                            mbar_cluster_ptr = prims.mapa(acc_empty_mbar_ptr_stage, leader_cta_rank)
                            prims.mbarrier_arrive(mbar_cluster_ptr, count=1, scope=prims.MemScope.CLUSTER)

                    # Reconstruct Torch's contiguous-channel warp tree without
                    # changing the four-physical-lanes-per-row TMEM mapping.
                    denominators = cutlass.Array(cutlass.Float32, rows_per_lane, space=cutlass.AddressSpace.rmem)
                    for row_half in cutlass.range_constexpr(rows_per_lane):
                        virtual_lanes = cutlass.Array(cutlass.Float32, 16, space=cutlass.AddressSpace.rmem)
                        for q in cutlass.range_constexpr(16):
                            lo = partials[row_half * 32 + q * 2]
                            hi = partials[row_half * 32 + q * 2 + 1]
                            # Even physical lanes produce ((p0+p1)+p2)+p3.
                            # Odd-lane values are not consumed by the final sum.
                            virtual_lanes[q] = ((lo + hi) + cute.arch.shuffle_sync(lo, lane ^ 1)) + cute.arch.shuffle_sync(hi, lane ^ 1)
                        for offset in cutlass.range_constexpr(4):
                            step = 8 >> offset
                            for q in cutlass.range_constexpr(step):
                                virtual_lanes[q] = virtual_lanes[q] + virtual_lanes[q + step]
                        total = cute.arch.shuffle_sync(virtual_lanes[0], (lane // 4) * 4) + cute.arch.shuffle_sync(virtual_lanes[0], (lane // 4) * 4 + 2)
                        denominator = cute.math.sqrt(total, fastmath=False)
                        if denominator < 1e-12:
                            denominator = cutlass.Float32(1e-12)
                        denominators[row_half] = denominator

                if cutlass.const_expr(not paired_norm or channel_half == 1):
                    # Subtile loop on the N axis.
                    for subtile_idx in cutlass.range(subtile_cnt * (2 if paired_norm else 1), unroll_full=FUSE_NORM):
                        # Rotate through C SMEM stages so the previous TMA store can
                        # drain in parallel with the next t2r/r2s.
                        epi_stage_idx = (epi_stage_idx + 1) % num_c_stage

                        if cutlass.const_expr(not FUSE_NORM):
                            tmem_ctm = prims.make_tmem_ptr_from_warp_row_col(
                                tmem_raw_addr,
                                warp_idx,
                                base_col_id + subtile_idx * subtile_n,
                                cutlass.Float32,
                            )
                            t2r_rmem = prims.tcgen05_ld("32x32b", tmem_ctm, num=32)
                            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)

                        if cutlass.const_expr(FUSE_NORM):
                            retained_values = retained.load(subtile_idx * 4 * rows_per_lane, 4 * rows_per_lane).bitcast(cutlass.BFloat16)
                            fused_rmem = cutlass.Array(
                                c_dtype,
                                8 * rows_per_lane,
                                space=cutlass.AddressSpace.rmem,
                                alignment=4,
                            )
                            math_width = 1 if paired_norm else 2
                            for j in cutlass.range_constexpr(8 * rows_per_lane // math_width):
                                scalar = j * math_width
                                col = subtile_idx * 32 + (scalar // (2 * rows_per_lane)) * 8 + lane_col * 2 + scalar % 2
                                value = retained_values[scalar : scalar + math_width].to(cutlass.Float32)
                                normalized = (value / denominators[(scalar // 2) % rows_per_lane]).to(cutlass.BFloat16)
                                scaled = (normalized.to(cutlass.Float32) * (norm_channels**0.5)).to(cutlass.BFloat16)
                                if cutlass.const_expr(paired_norm):
                                    gamma_pair = (norm_params.data_ptr() + 320 + col).load(count=math_width, alignment=2 * math_width)
                                else:
                                    gamma_pair = (gamma.iterator.raw_ptr() + col).load(count=2, alignment=4)
                                affine = (scaled.to(cutlass.Float32) * gamma_pair.to(cutlass.Float32)).to(cutlass.BFloat16)
                                value = affine.to(cutlass.Float32)
                                activated = (value / (1.0 + cute.math.exp(-value, fastmath=False))).to(cutlass.BFloat16)
                                fused_rmem.store(activated, scalar)

                        # Raw lanes store four 16B vectors. Fused lanes store eight
                        # adjacent BF16 pairs, distributed across four-lane row groups.
                        smem_tile_base = sC.subview(epi_stage_idx * (_c_tile_rows * _epi_subtile_n))
                        if cutlass.const_expr(FUSE_NORM):
                            for pair in cutlass.range_constexpr(4 * rows_per_lane):
                                row = row_base + lane // 4 + (pair % rows_per_lane + drain_half) * 8
                                col = (pair // rows_per_lane) * 8 + lane_col * 2
                                vec_io = fused_rmem[pair * 2 : 2]
                                smem_thr_ptr = smem_tile_base.subview(row * subtile_n + col)
                                smem_thr_ptr.data_ptr().store_swizzled(
                                    vec_io,
                                    alignment=4,
                                    swizzle=cutlass.Swizzle(2, 4, 3),
                                )
                                if cutlass.const_expr(PREP_SHAPE is not None):
                                    pixel = cutlass.Int64(m_off_cta) + row
                                    batch = pixel // (Z_out * P_out * Q_out)
                                    time = pixel // (P_out * Q_out) % Z_out
                                    spatial = pixel % (P_out * Q_out)
                                    if batch < batches and time >= Z_out - cache_frames:
                                        cache_offset = (
                                            ((batch * cache_frames + time - Z_out + cache_frames) * P_out * Q_out + spatial) * norm_channels
                                            + subtile_idx * 32
                                            + col
                                        )
                                        (prep_cache.iterator.raw_ptr() + cache_offset).store(vec_io, alignment=4)
                        else:
                            for j in cutlass.range_constexpr(32 // vsize):
                                vec_io = t2r_rmem[j * vsize : j * vsize + vsize].to(c_dtype)
                                if cutlass.const_expr(SPATIAL_SHAPE is not None):
                                    spatial_pixel = cutlass.Int64(m_off_cta) + tidx
                                    spatial_channel = k_off_base + subtile_idx * subtile_n + j * vsize
                                    if spatial_pixel < SPATIAL_SHAPE[0] * Z_out * P_out * Q_out:
                                        bias_vec = (conv_bias.iterator.raw_ptr() + spatial_channel).load(count=vsize, alignment=16)
                                        vec_io = (vec_io.to(cutlass.Float32) + bias_vec.to(cutlass.Float32)).to(c_dtype)
                                        spatial_skip = (residual.iterator.raw_ptr() + spatial_pixel * SPATIAL_SHAPE[1] + spatial_channel).load(
                                            count=vsize, alignment=16
                                        )
                                        if cutlass.const_expr(residual_bias is not None):
                                            spatial_skip_bias = (residual_bias.iterator.raw_ptr() + spatial_channel).load(count=vsize, alignment=16)
                                            spatial_skip = (spatial_skip.to(cutlass.Float32) + spatial_skip_bias.to(cutlass.Float32)).to(c_dtype)
                                        vec_io = (vec_io.to(cutlass.Float32) + spatial_skip.to(cutlass.Float32)).to(c_dtype)
                                smem_thr_ptr = smem_tile_base.subview(tidx * subtile_n + j * vsize)
                                smem_thr_ptr.data_ptr().store_swizzled(
                                    vec_io,
                                    alignment=16,
                                    swizzle=cutlass.Swizzle(2, 4, 3),
                                )

                        # Make swizzled SMEM stores visible to the TMA proxy.
                        cute.arch.fence_view_async_shared()
                        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)

                        # One issuer owns the TMA store, commit, and wait sequence.
                        if warp_idx == epilogue_warp_ids[0] and store_issuer:
                            k_off = (0 if paired_norm else k_off_base) + subtile_idx * subtile_n
                            prims.cp_async_bulk_tensor_global_shared_cta(
                                tma_c_desc.get_ptr(),
                                smem_tile_base,
                                (k_off, q_out, p_out, z_out, n_out),
                                mode=prims.TMAStoreMode.IM2COL,
                            )
                            prims.cp_async_bulk_commit_group()
                            # Bound outstanding reads before the next SMEM reuse.
                            prims.cp_async_bulk_wait_group(num_c_stage - 1, read=True)

                        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)

                # All lanes must order completed TMEM loads before releasing it.
                if cutlass.const_expr(not FUSE_NORM):
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)

                # Signal acc_empty back to the group leader's mbarrier. In 2cta the
                # mbar lives in the even-rank leader's SMEM and arrive_count was set
                # to (4 epi warps * 2 CTAs); the per-group leader rank is
                # (rank // _atom_thr) * _atom_thr (degrades to 0 for the single
                # group, 2 for group 1, etc.). In 1cta _atom_thr=1 so this is the
                # CTA's own rank (count 4, own mbar).
                if cutlass.const_expr(not FUSE_NORM) and prims.elect_sync():
                    leader_cta_rank = (cta_rank_in_cluster // _atom_thr) * _atom_thr
                    mbar_cluster_ptr = prims.mapa(acc_empty_mbar_ptr_stage, leader_cta_rank)
                    prims.mbarrier_arrive(
                        mbar_cluster_ptr,
                        count=1,
                        scope=prims.MemScope.CLUSTER,
                    )

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # The issuing lane drains global writes, not only SMEM reads.
        if warp_idx == epilogue_warp_ids[0] and store_issuer:
            prims.cp_async_bulk_wait_group(0, read=False)

        # All epilogue readers must finish before the allocator frees TMEM.
        prims.barrier_cta_sync(epilog_sync_bar_id, thread_count=threads_in_epilogue)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

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


_conv3d_postops_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@dataclass(frozen=True)
class Conv3dConfig:
    """Padded NTHWC dimensions for the fixed 3x3x3, N160, 2CTA kernel."""

    n: int
    t: int
    h: int
    w: int
    ci: int
    co: int
    max_active_clusters: int

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Return the contiguous NTHWC convolution input shape."""
        return self.n, self.t, self.h, self.w, self.ci

    @property
    def packed_weight_shape(self) -> tuple[int, ...]:
        """OTRSC filters with each input-channel row padded to K64."""
        return self.co, 3, 3, 3, (self.ci + 63) // 64 * 64

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Return the NTHWC shape after valid 3x3x3 convolution."""
        return self.n, self.t - 2, self.h - 2, self.w - 2, self.co


class Conv3dPostOpsLaunch:
    """Specialize the convolution launch for its geometry and fused post-operations."""

    def __init__(
        self,
        config: Conv3dConfig,
        fuse_norm: bool = False,
        prepare_output: bool = False,
        previous_frames: int | None = None,
        has_residual: bool = False,
        has_residual_bias: bool = False,
        spatial_output: bool = False,
    ) -> None:
        """Record convolution geometry and validate the selected fusion options."""
        self.config = config
        self.fuse_norm = fuse_norm
        self.prepare_output = prepare_output
        self.previous_frames = previous_frames
        self.has_residual = has_residual
        self.has_residual_bias = has_residual_bias
        self.spatial_output = spatial_output
        if self.prepare_output and not self.fuse_norm:
            raise ValueError("Prepared output requires normalization")
        if self.previous_frames is not None and not self.prepare_output:
            raise ValueError("History is only valid for prepared output")

    def __repr__(self) -> str:
        """Identify the launch specialization by shape, schedule, and fusion options."""
        cfg = self.config
        operation = "Raw"
        if self.prepare_output:
            operation = "NormSiluPad"
        elif self.fuse_norm:
            operation = "NormSilu"
        elif self.spatial_output:
            operation = "BiasResidualPad"
        return (
            f"Conv3d{operation}_"
            f"{cfg.n}x{cfg.t}x{cfg.h}x{cfg.w}x{cfg.ci}_{cfg.co}_"
            f"2cta_n160_k3_clusters{cfg.max_active_clusters}_prev{self.previous_frames}"
            f"_res{self.has_residual}_rb{self.has_residual_bias}"
            f"_spatial{self.spatial_output}"
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda_driver.CUstream,
        conv_bias: cute.Tensor = None,
        gamma: cute.Tensor = None,
        prep_padded: cute.Tensor = None,
        prep_cache: cute.Tensor = None,
        residual: cute.Tensor = None,
        residual_bias: cute.Tensor = None,
        prep_residual: cute.Tensor = None,
    ) -> None:
        """Build tensor maps and launch the shape-specialized convolution pipeline."""
        cfg = self.config
        tma_a_desc, tma_b_desc, tma_c_desc = _make_tensor_maps(
            cfg,
            a,
            b,
            c,
            prepared_output=self.prepare_output,
            spatial_output=self.spatial_output,
        )
        groups = 2
        tiles = (
            cute.ceil_div(math.prod(cfg.output_shape[:-1]), 128),
            1 if self.fuse_norm else cute.ceil_div(cfg.co, 160),
            1,
        )
        scheduler = PersistentTileSchedulerParams(tiles, (groups, 1, 1))
        grid = StaticPersistentTileScheduler.get_grid_shape(scheduler, cfg.max_active_clusters)
        ab_stages, c_stages = 8, 2
        if cutlass.const_expr(self.fuse_norm and cfg.co == 320 and cfg.ci == 320):
            # The longer K135 convolution benefits from a deeper output-store
            # queue; C160->C320 keeps its measured faster eight/two schedule.
            ab_stages, c_stages = 6, 6
        # Each spatial filter position has its own independently zero-filled
        # channel tail. Flattening TRS*C before ceil-div truncates C=160 work.
        reduction_tiles = 3 * 3 * 3 * ((cfg.ci + 63) // 64)
        _conv3d_postops_kernel(
            (
                "conv_bias_residual_pad"
                if self.spatial_output
                else (
                    "conv_residual_norm_silu_pad"
                    if self.has_residual and self.prepare_output
                    else (
                        "conv_norm_silu_pad"
                        if self.prepare_output
                        else "conv_residual_norm_silu" if self.has_residual else "conv_norm_silu" if self.fuse_norm else "conv_raw"
                    )
                )
            ),
            scheduler,
            (128 * groups, 160, 64),
            (128 * groups, 160, 16),
            reduction_tiles,
            ab_stages,
            c_stages,
            True,
            cfg.output_shape[1:4],
            cfg.n,
            tma_a_desc,
            tma_b_desc,
            tma_c_desc,
            conv_bias,
            gamma,
            prep_padded=prep_padded,
            prep_cache=prep_cache,
            PREP_SHAPE=(cfg.n, self.previous_frames) if self.prepare_output else None,
            residual=residual,
            residual_bias=residual_bias,
            prep_residual=prep_residual,
            SPATIAL_SHAPE=(cfg.n, cfg.co) if self.spatial_output else None,
        ).launch(
            grid=grid,
            block=(
                (
                    ((576 if cfg.co == 320 or self.has_residual else 320) + ((128 if cfg.co == 320 else 64) if self.prepare_output else 0))
                    if self.fuse_norm
                    else (320 if self.spatial_output else 192)
                ),
                1,
                1,
            ),
            min_blocks_per_mp=1 if self.fuse_norm or self.spatial_output else 0,
            cluster=(groups, 1, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )


def _make_tensor_maps(
    cfg: Conv3dConfig,
    x: cute.Tensor,
    weight: cute.Tensor,
    output: cute.Tensor,
    *,
    prepared_output: bool = False,
    spatial_output: bool = False,
) -> tuple[cuda.TensorMap, cuda.TensorMap, cuda.TensorMap]:
    """Build public tensor maps inside the JIT host wrapper."""
    dtype = cutlass.BFloat16
    a_dims, a_strides = _contiguous_tma_layout(cfg.input_shape, dtype)
    b_dims, b_strides = _contiguous_tma_layout(cfg.packed_weight_shape, dtype)
    c_dims, c_strides = _contiguous_tma_layout(cfg.output_shape, dtype)
    output_offset = 0
    if prepared_output or spatial_output:
        # Logical T/H/W omit the halo, while physical pitches include it.
        # An IM2COL store can therefore walk 128 logical pixels across rows,
        # frames and batches without ever touching the separately owned halo.
        _, t, h, w, c = cfg.output_shape
        pad_hw = 2 if prepared_output else 1
        pad_t = 2 if prepared_output else 0
        c_strides_bytes = (
            c * dtype.width // 8,
            (w + pad_hw) * c * dtype.width // 8,
            (h + pad_hw) * (w + pad_hw) * c * dtype.width // 8,
            (t + pad_t) * (h + pad_hw) * (w + pad_hw) * c * dtype.width // 8,
        )
        if any(stride % 16 for stride in c_strides_bytes):
            raise ValueError("Output TMA strides must be multiples of 16 bytes")
        c_strides = tuple(stride // 16 for stride in c_strides_bytes)
        if prepared_output:
            output_offset = (2 * (h + 2) * (w + 2) + (w + 2) + 1) * c

    tma_a_desc = cuda.create_tensor_map_im2col(
        global_address=x.iterator.toint(),
        dtype=dtype,
        global_dims=a_dims,
        global_strides=a_strides,
        lower_corner=(0, 0, 0),
        upper_corner=(-2, -2, -2),
        channels_per_pixel=64,
        pixels_per_column=128,
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_b_desc = cuda.create_tensor_map_tiled(
        global_address=weight.iterator.toint(),
        dtype=dtype,
        global_dims=b_dims,
        global_strides=b_strides,
        box_dims=(64, 1, 1, 1, 80),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_c_desc = cuda.create_tensor_map_im2col(
        global_address=(output.iterator + output_offset).toint(),
        dtype=dtype,
        global_dims=c_dims,
        global_strides=c_strides,
        lower_corner=(0, 0, 0),
        upper_corner=(0, 0, 0),
        channels_per_pixel=32,
        pixels_per_column=128,
        swizzle=cuda.TensorMapSwizzle.s64b,
    )
    return tma_a_desc, tma_b_desc, tma_c_desc
