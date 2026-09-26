# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications are licensed under Apache-2.0. Pre-existing code retains
# its MIT terms; see LICENSING.md and THIRD_PARTY_LICENSES.txt.

"""The FROST SM120 SDPA backward kernel chain around the fused main pass:
the deterministic two-kernel dQ GEMM and the dQ / dBias convert kernels.

The arch-neutral members of the chain -- ``dot`` (delta preprocess), the GQA
dK/dV group reduce and ``dsink`` -- live in ``..bprop_chain_common`` (they are
shared with the SM100 chain) and are re-exported here so this module's public
names are unchanged."""

from cudnn._cutlass_compat import get_smem_capacity_in_bytes
from typing import Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.utils
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

from cudnn.sdpa.bwd.config_sm120 import ROW_ROUND
from cudnn.sdpa.bwd.kernels.sm120._common import (
    _COPY_ELEMS,
    ceil_div,
    copy16_smem_to_gmem,
    load_a_frag,
    mma_bstream,
    pack_half2,
    tile_ptr,
)
from cudnn.sdpa.bwd.kernels.bprop_chain_common import (  # noqa: F401 -- re-exported; the chain's public names are unchanged
    dkv_reduce_host,
    dkv_reduce_kernel,
    dot_do_o_host,
    dot_do_o_kernel,
    dsink_host,
    dsink_kernel,
)


class SM120DetDqGemmKernel:
    """Deterministic dQ GEMM over the main kernel's dS workspace.

    Q-stationary: one CTA per ``q_tile`` rows (grid ``(num_q_tiles, H_q,
    B)``), streaming (K tile, dS panel) pairs in ascending kv order through
    a 2-stage TMA pipeline; dQ accumulates in registers and stores directly
    in the io dtype (no atomics, no fp32 workspace, no convert kernel).
    """

    def __init__(
        self,
        in_dtype: Type[cutlass.Numeric],
        is_causal: bool,
        causal_top_left: bool,
        right_slack: int,
        head_dim: int,
        q_tile: int,
        kv_tile: int,
        ws_q_tile: int,
        use_pdl: bool,
    ):
        self.in_dtype = in_dtype
        self.is_causal = bool(is_causal)
        self.causal_top_left = bool(causal_top_left)
        self.right_slack = int(right_slack)
        self.d = head_dim
        self.q_tile = q_tile
        self.kv_tile = kv_tile
        self.ws_q_tile = ws_q_tile
        if kv_tile % 64:
            raise ValueError(f"deterministic dQ GEMM: kv_tile must be a multiple of 64; got {kv_tile}")
        if q_tile % 16:
            raise ValueError(f"deterministic dQ GEMM: q_tile must be a multiple of 16; got {q_tile}")
        if q_tile % ws_q_tile or ws_q_tile % 16:
            raise ValueError(f"deterministic dQ GEMM: q_tile ({q_tile}) must be a multiple of ws_q_tile ({ws_q_tile}), ws_q_tile a multiple of 16")
        self.use_pdl = bool(use_pdl)
        self.chunk_elems = 64 if head_dim % 64 == 0 else 32
        self.ds_chunk_elems = 64  # chunk width along kv (kv_tile is a multiple of 64)
        self.tma_swizzle = cuda.TensorMapSwizzle.s128b if self.chunk_elems == 64 else cuda.TensorMapSwizzle.s64b

        # One 16-row MMA block per compute warp, plus the TMA producer warp.
        self.num_compute_warps = q_tile // 16
        self.load_warp_id = self.num_compute_warps
        self.num_warps = self.num_compute_warps + 1
        self.threads = 32 * self.num_warps
        self.threads_pipeline = 32 * (self.num_compute_warps + 1)
        self.stages = 2
        self.k_tile_elems = kv_tile * head_dim
        self.ds_tile_elems = q_tile * kv_tile
        smem_bytes = self.stages * (self.k_tile_elems + self.ds_tile_elems) * in_dtype.bytes + self.stages * 8
        cap = get_smem_capacity_in_bytes("sm_120")
        if smem_bytes > cap:
            raise ValueError(f"deterministic dQ GEMM: smem {smem_bytes} bytes exceeds the sm_120 cap of {cap} bytes")
        self.min_blocks = 1

    @cute.jit
    def load_stage(
        self,
        sK,
        sDS,
        tma_mbar,
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_ds_desc: cutlass.GridConstant[cuda.TensorMap],
        batch,
        kv_head,
        q_head,
        stage,
        kv_seq,
        q_base,
    ) -> None:
        """Issue one pipeline stage: a K tile and a dS panel (joint mbarrier)."""
        mbar = tma_mbar.subview(stage)
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(mbar, (self.k_tile_elems + self.ds_tile_elems) * self.in_dtype.bytes)
        k_elems_per_chunk = self.kv_tile * self.chunk_elems
        for chunk in cutlass.range_constexpr(self.d // self.chunk_elems):
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sK.subview(stage * self.k_tile_elems + chunk * k_elems_per_chunk),
                    tma_k_desc.get_ptr(),
                    (chunk * self.chunk_elems, kv_head, kv_seq, batch),
                    mbar,
                )
        ds_elems_per_chunk = self.q_tile * self.ds_chunk_elems
        for chunk in cutlass.range_constexpr(self.kv_tile // self.ds_chunk_elems):
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sDS.subview(stage * self.ds_tile_elems + chunk * ds_elems_per_chunk),
                    tma_ds_desc.get_ptr(),
                    (kv_seq + chunk * self.ds_chunk_elems, q_base, q_head, batch),
                    mbar,
                )

    @cute.kernel
    def kernel(
        self,
        k: cute.Tensor,  # [B, S_KV, H_KV, D] io dtype
        ds_ws: cute.Tensor,  # [B, H_Q, S_Q, S_KV_r128] io dtype (main-kernel dS, unscaled)
        dq: cute.Tensor,  # [B, S_Q, H_Q, D] io dtype out
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_ds_desc: cutlass.GridConstant[cuda.TensorMap],
        attn_scale: cutlass.Float32,
    ) -> None:
        io_dtype = self.in_dtype
        D = self.d
        Q_TILE = self.q_tile
        KV_TILE = self.kv_tile
        CHUNK_ELEMS = self.chunk_elems
        DS_CHUNK_ELEMS = self.ds_chunk_elems
        STAGES = self.stages
        KV_CHUNKS = KV_TILE // 16
        DQ_COL_FRAGS = D // 8

        tidx, _, _ = cute.arch.thread_idx()
        q_block, q_head, batch = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        g_lane = lane // 4
        p_lane = lane % 4

        S_Q = dq.shape[1]
        S_KV = k.shape[1]
        H_Q = dq.shape[2]
        H_KV = k.shape[2]
        GROUP = H_Q // H_KV
        kv_head = q_head // GROUP
        q_base = q_block * Q_TILE
        PARTIAL_Q = (S_Q % Q_TILE) != 0

        # kv range: dense reads all; causal stops at this tile's last-row
        # (right-band widened) diagonal — same geometry as the main kernel.
        if cutlass.const_expr(self.is_causal):
            if cutlass.const_expr(self.causal_top_left):
                diag_off = cutlass.Int32(0)
            else:
                diag_off = S_KV - S_Q
            kv_end = cute.math.min(cutlass.Int32(S_KV), q_base + Q_TILE + diag_off + self.right_slack)
            kv_end = cute.math.max(kv_end, cutlass.Int32(0))
        else:
            kv_end = cutlass.Int32(S_KV)
        num_kv_blocks = ceil_div(kv_end, KV_TILE)

        sK = cutlass.Array(io_dtype, self.k_tile_elems * STAGES, space=cutlass.AddressSpace.smem, alignment=128)
        sDS = cutlass.Array(io_dtype, self.ds_tile_elems * STAGES, space=cutlass.AddressSpace.smem, alignment=128)
        tma_mbar = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8)

        if warp == self.load_warp_id:
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_k_desc.get_ptr())
                prims.prefetch_tensormap(tma_ds_desc.get_ptr())
                for st in cutlass.range_constexpr(STAGES):
                    prims.mbarrier_init(tma_mbar.subview(st), 1)
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)

        if warp == self.load_warp_id:
            # The dS panels are the main kernel's output: wait for its grid
            # before the first TMA touches the workspace.
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            issue = cutlass.Int32(0)
            while (issue < cutlass.Int32(STAGES)) & (issue < num_kv_blocks):
                self.load_stage(sK, sDS, tma_mbar, tma_k_desc, tma_ds_desc, batch, kv_head, q_head, issue % STAGES, issue * KV_TILE, q_base)
                issue += 1
            if num_kv_blocks > 0:
                while not prims.mbarrier_try_wait_parity(tma_mbar.subview(0), cutlass.Int32(0)):
                    pass
            ready = cutlass.Int32(1)
            done = cutlass.Int32(0)
            while done < num_kv_blocks:
                cute.arch.barrier(barrier_id=2, number_of_threads=self.threads_pipeline)
                if ready < num_kv_blocks:
                    if ready >= STAGES:
                        self.load_stage(sK, sDS, tma_mbar, tma_k_desc, tma_ds_desc, batch, kv_head, q_head, ready % STAGES, ready * KV_TILE, q_base)
                    while not prims.mbarrier_try_wait_parity(tma_mbar.subview(ready % STAGES), (ready // STAGES) & cutlass.Int32(1)):
                        pass
                    ready += 1
                done += 1

        elif warp < self.load_warp_id:  # compute warps
            row0 = warp * 16  # this warp's q rows within the tile
            acc_dq = cutlass.Array(cutlass.Float32, DQ_COL_FRAGS * 4, alignment=16)
            for i in cutlass.range_constexpr(DQ_COL_FRAGS * 4):
                acc_dq[i] = cutlass.Float32(0.0)

            if cutlass.const_expr(self.is_causal and self.ws_q_tile != Q_TILE):
                ws_end = q_base + (row0 // self.ws_q_tile + 1) * self.ws_q_tile
                kv_end_warp = cute.math.min(cutlass.Int32(S_KV), ws_end + diag_off + self.right_slack)
                kv_end_warp = cute.math.max(kv_end_warp, cutlass.Int32(0))
                # This warp's kv-block count, causal-trimmed at ws_q_tile granularity.
                num_kv_blocks_warp = ceil_div(kv_end_warp, KV_TILE)
            else:
                num_kv_blocks_warp = num_kv_blocks

            kv_block = cutlass.Int32(0)
            while kv_block < num_kv_blocks:
                stage = kv_block % STAGES
                sK_stage = sK.subview(stage * self.k_tile_elems)
                sDS_stage = sDS.subview(stage * self.ds_tile_elems)
                cute.arch.barrier(barrier_id=2, number_of_threads=self.threads_pipeline)
                if kv_block < num_kv_blocks_warp:
                    for k_chunk in cutlass.range_constexpr(KV_CHUNKS):
                        a_frag = load_a_frag(sDS_stage, k_chunk, row0, lane, rows=Q_TILE, chunk_elems=DS_CHUNK_ELEMS)
                        mma_bstream(
                            acc_dq,
                            [a_frag[0], a_frag[1], a_frag[2], a_frag[3]],
                            sK_stage,
                            b_k_step=k_chunk,
                            M=16,
                            N=D,
                            b_trans=True,
                            b_rows=KV_TILE,
                            b_chunk_elems=CHUNK_ELEMS,
                            lane=lane,
                            ab_dtype=io_dtype,
                        )
                kv_block += 1

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            r0 = q_base + row0 + g_lane
            r8 = r0 + 8
            dq_ptr = dq.iterator.raw_ptr()
            for col_frag in cutlass.range_constexpr(DQ_COL_FRAGS):
                col = col_frag * 8 + 2 * p_lane
                off = col_frag * 4
                base_top = ((batch * S_Q + r0) * H_Q + q_head) * D + col
                base_bot = ((batch * S_Q + r8) * H_Q + q_head) * D + col
                if (not cutlass.const_expr(PARTIAL_Q)) or (r0 < S_Q):
                    (dq_ptr + base_top).store(
                        pack_half2(acc_dq[off + 0] * attn_scale, acc_dq[off + 1] * attn_scale, io_dtype),
                        alignment=4,
                    )
                if (not cutlass.const_expr(PARTIAL_Q)) or (r8 < S_Q):
                    (dq_ptr + base_bot).store(
                        pack_half2(acc_dq[off + 2] * attn_scale, acc_dq[off + 3] * attn_scale, io_dtype),
                        alignment=4,
                    )

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,
        ds_ws: cute.Tensor,
        dq: cute.Tensor,
        attn_scale: cutlass.Float32,
        stream: cuda_driver.CUstream,
    ) -> None:
        box_k = (1, self.kv_tile, 1, self.chunk_elems)
        tma_k_desc = cuda.create_tensor_map_tiled_from_view(k, box_dims=box_k, stride_order=(3, 2, 1, 0), swizzle=self.tma_swizzle)
        box_ds = (1, 1, self.q_tile, self.ds_chunk_elems)
        tma_ds_desc = cuda.create_tensor_map_tiled_from_view(ds_ws, box_dims=box_ds, stride_order=(3, 2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
        n_q_tiles = ceil_div(dq.shape[1], self.q_tile)
        self.kernel(k, ds_ws, dq, tma_k_desc, tma_ds_desc, attn_scale).launch(
            grid=(n_q_tiles, dq.shape[2], dq.shape[0]),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.min_blocks,
            use_pdl=self.use_pdl,
        )


# ---------------------------------------------------------------------------
# Convert kernel: scrambled dq_accum (fp32) -> dQ (io dtype)
# ---------------------------------------------------------------------------


@cute.kernel
def convert_dq_kernel(
    dq_accum: cute.Tensor,  # [B*S_Q_r128*H*D] fp32
    dq: cute.Tensor,  # [B, S_Q, H, D] io dtype out
    q_tile: cutlass.Constexpr[int],
    D_QK: cutlass.Constexpr[int],
    chunk_elems: cutlass.Constexpr[int],
    warps_m_dq: cutlass.Constexpr[int],
    attn_scale: cutlass.Float32,
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
    q_block, head, batch = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    g_lane = lane // 4
    p_lane = lane % 4
    S_Q = dq.shape[1]
    H = dq.shape[2]
    S_Q_R = ceil_div(S_Q, ROW_ROUND) * ROW_ROUND
    Q_TILE = q_tile
    WM_DQ = warps_m_dq
    DQ_ROW_BLOCKS = Q_TILE // (16 * WM_DQ)
    DQ_COLS = D_QK * WM_DQ // 8
    DQ_COL_FRAGS = DQ_COLS // 8
    wm_dq = warp % WM_DQ
    wn_dq = warp // WM_DQ

    dq_accum_ptr = dq_accum.iterator.raw_ptr()
    dq_ptr = dq.iterator.raw_ptr()

    sdQ = cutlass.Array(io_dtype, Q_TILE * D_QK, space=cutlass.AddressSpace.smem, alignment=128)

    t_row = tidx // 32
    t_col = tidx % 32
    dq_accum_base = ((batch * S_Q_R + q_block * Q_TILE) * H + head) * D_QK
    for row_blk in cutlass.range_constexpr(DQ_ROW_BLOCKS):
        for col_frag in cutlass.range_constexpr(DQ_COL_FRAGS):
            frag = cutlass.Array(cutlass.Float32, 4)
            for hf in cutlass.range_constexpr(2):
                i_pair = hf + row_blk * 2 + col_frag * 2 * DQ_ROW_BLOCKS
                if cutlass.const_expr(D_QK >= 64):
                    pair_row = i_pair % (Q_TILE // 8)
                    pair_col = i_pair // (Q_TILE // 8)
                    addr = dq_accum_base + (t_row + pair_row * 8) * (H * D_QK) + t_col * 2 + pair_col * 64
                else:
                    addr = dq_accum_base + (t_row + (t_col // 16) * 8 + i_pair * 16) * (H * D_QK) + (t_col % 16) * 2
                pair = (dq_accum_ptr + addr).load(count=2)
                frag[hf * 2 + 0] = pair[0] * attn_scale
                frag[hf * 2 + 1] = pair[1] * attn_scale
            r0 = wm_dq * 16 + row_blk * 16 * WM_DQ + g_lane
            r8 = r0 + 8
            c0 = wn_dq * DQ_COLS + col_frag * 8 + 2 * p_lane
            tile_ptr(sdQ, r0, c0, chunk_elems=chunk_elems, rows=Q_TILE).store(pack_half2(frag[0], frag[1], io_dtype), alignment=4)
            tile_ptr(sdQ, r8, c0, chunk_elems=chunk_elems, rows=Q_TILE).store(pack_half2(frag[2], frag[3], io_dtype), alignment=4)
    prims.barrier_cta_sync(0)

    q_left = S_Q - q_block * Q_TILE
    dq_batch_stride, dq_seq_stride, dq_head_stride, _ = dq.stride
    g_base = batch * dq_batch_stride + (q_block * Q_TILE) * dq_seq_stride + head * dq_head_stride
    chunks_per_row = D_QK // _COPY_ELEMS
    for i in cutlass.range_constexpr(Q_TILE * chunks_per_row // 256):
        chunk = i * 256 + tidx
        row = chunk // chunks_per_row
        col = (chunk % chunks_per_row) * _COPY_ELEMS
        if row < q_left:
            if cutlass.const_expr(dq.shape[3] != D_QK):
                # Head dim is padded: dQ is only dq.shape[3] wide (pad columns are zero).
                if col < dq.shape[3]:
                    copy16_smem_to_gmem(
                        tile_ptr(sdQ, row, col, chunk_elems=chunk_elems, rows=Q_TILE),
                        dq_ptr + g_base + row * dq_seq_stride + col,
                    )
            else:
                copy16_smem_to_gmem(
                    tile_ptr(sdQ, row, col, chunk_elems=chunk_elems, rows=Q_TILE),
                    dq_ptr + g_base + row * dq_seq_stride + col,
                )


convert_dq_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def convert_dq_host(
    dq_accum: cute.Tensor,
    dq: cute.Tensor,
    q_tile: cutlass.Constexpr[int],
    D_QK: cutlass.Constexpr[int],
    chunk_elems: cutlass.Constexpr[int],
    warps_m_dq: cutlass.Constexpr[int],
    attn_scale: cutlass.Float32,
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    q_blocks = ceil_div(dq.shape[1], q_tile)
    convert_dq_kernel(dq_accum, dq, q_tile, D_QK, chunk_elems, warps_m_dq, attn_scale, io_dtype, use_pdl).launch(
        grid=(q_blocks, dq.shape[2], dq.shape[0]),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


# ---------------------------------------------------------------------------
# Convert kernel: dbias_accum (fp32) -> dBias (io dtype; fp32 outputs skip it)
# ---------------------------------------------------------------------------


@cute.kernel
def convert_dbias_kernel(
    dbias_accum: cute.Tensor,  # [total] fp32 (flat view of [1|B, H_Q, S_Q, S_KV])
    dbias: cute.Tensor,  # [total] out dtype (flat view, same layout)
    out_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    total = dbias.shape[0]
    acc_ptr = dbias_accum.iterator.raw_ptr()
    out_ptr = dbias.iterator.raw_ptr()
    gidx = bidx * 256 + tidx
    if gidx < total:
        (out_ptr + gidx).store((acc_ptr + gidx).load().to(out_dtype))


convert_dbias_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def convert_dbias_host(
    dbias_accum: cute.Tensor,
    dbias: cute.Tensor,
    out_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    convert_dbias_kernel(dbias_accum, dbias, out_dtype, use_pdl).launch(
        grid=(ceil_div(dbias.shape[0], 256), 1, 1),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )
