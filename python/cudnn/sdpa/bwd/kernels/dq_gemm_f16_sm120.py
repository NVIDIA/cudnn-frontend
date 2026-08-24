# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""det_2kernel dQ GEMM for the FROST SM120 SDPA backward (fp16 / bf16)."""

from typing import Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.utils
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

from cudnn.frost.tile_dsl.mma import mma_m16n8k16_f32
from cudnn.frost.tile_dsl.swizzle import swizzle_xor

_LOG2E = 1.4426950408889634
_COPY_ELEMS = 8  # 16-byte gmem<->smem chunk (8 fp16/bf16)


@cute.jit
def tile_ptr(
    sbuf,
    row: cutlass.Int32,
    col: cutlass.Int32,
    *,
    page: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
):
    """Element pointer into a swizzled smem tile."""
    pg = col // page
    in_col = col % page
    off = pg * (rows * page) + row * page + swizzle_xor(row, in_col, page, 2)
    return sbuf.subview(off).data_ptr()


@cute.jit
def pack_half2(lo, hi, dtype: cutlass.Constexpr[Type[cutlass.Numeric]]):
    """Pack two fp32 into a 2-element io-dtype vector (one 4 B store)."""
    return cutlass.Vector.from_elements((lo.to(dtype), hi.to(dtype)), dtype)


@cute.jit
def load_a_frag(
    sbuf,
    kc: cutlass.Constexpr[int],
    row0,
    lane,
    *,
    rows: cutlass.Constexpr[int],
    page: cutlass.Constexpr[int],
):
    """ldmatrix.x4 one (16 x 16) row-major A fragment."""
    row = row0 + lane % 16
    col = kc * 16 + (lane // 16) * 8
    return prims.ldmatrix(tile_ptr(sbuf, row, col, page=page, rows=rows), 4, prims.MMALayout.ROW)


@cute.jit
def load_a_frag_transposed(
    sbuf,
    kc: cutlass.Constexpr[int],
    col0,
    lane,
    *,
    rows: cutlass.Constexpr[int],
    page: cutlass.Constexpr[int],
):
    """ldmatrix.trans.x4: A[M, K] from a tile stored physically [K, M]."""
    row = kc * 16 + lane % 16
    col = col0 + (lane // 16) * 8
    return prims.ldmatrix(tile_ptr(sbuf, row, col, page=page, rows=rows), 4, prims.MMALayout.COL)


@cute.jit
def copy16_smem_to_gmem(sptr, gptr):
    """One 16-byte smem->gmem chunk."""
    v = sptr.load(count=8)
    gptr.store(v, alignment=16)


@cute.jit
def mma_bstream(
    acc,
    a_frag,
    sB,
    *,
    b_k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    b_trans: cutlass.Constexpr[bool],
    b_rows: cutlass.Constexpr[int],
    b_page: cutlass.Constexpr[int],
    lane,
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    col_base=0,
    row_base=0,
):
    """One k=16 step of an (M x N) MMA, B streamed from smem via
    ldmatrix.x4 (2 adjacent 8-column n-frags per fetch).

    acc:    ``(M//16) * (N//8) * 4`` fp32, m-rep major then n-frag.
    a_frag: ``(M//16) * 4`` Int32 (one 16x16 A fragment per m-rep).
    """
    M_BLOCKS = M // 16
    N_FRAGS = N // 8
    PAIRS = N_FRAGS // 2
    a_stride = len(a_frag) // M_BLOCKS

    if cutlass.const_expr(b_trans):
        b_row = lane % 16
        n_offset = lane // 16
        layout_flag = prims.MMALayout.COL
    else:
        b_row = lane % 8
        b_col_subchunk = (lane // 8) % 2
        n_offset = lane // 16
        layout_flag = prims.MMALayout.ROW

    for pair in cutlass.range_constexpr(PAIRS):
        n_frag = pair * 2
        if cutlass.const_expr(b_trans):
            row = b_k_step * 16 + b_row
            col = (n_frag + n_offset) * 8 + col_base
        else:
            row = row_base + (n_frag + n_offset) * 8 + b_row
            col = b_k_step * 16 + b_col_subchunk * 8 + col_base
        b_ptr = tile_ptr(sB, row, col, page=b_page, rows=b_rows)
        b_v = prims.ldmatrix(b_ptr, 4, layout_flag)
        for m_block in cutlass.range_constexpr(M_BLOCKS):
            a_off = m_block * a_stride
            for half in cutlass.range_constexpr(2):
                s = (m_block * N_FRAGS + n_frag + half) * 4
                c0, c1, c2, c3 = mma_m16n8k16_f32(
                    a_frag[a_off + 0],
                    a_frag[a_off + 1],
                    a_frag[a_off + 2],
                    a_frag[a_off + 3],
                    b_v[half * 2 + 0],
                    b_v[half * 2 + 1],
                    acc[s + 0],
                    acc[s + 1],
                    acc[s + 2],
                    acc[s + 3],
                    ab_dtype,
                )
                acc[s + 0] = c0
                acc[s + 1] = c1
                acc[s + 2] = c2
                acc[s + 3] = c3


@cute.jit
def mma_abregs(
    acc,
    a_frag,
    b_frag,
    *,
    b_k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    """One k=16 MMA step with both operands resident in registers."""
    M_BLOCKS = M // 16
    N_FRAGS = N // 8
    PAIRS = N_FRAGS // 2
    a_stride = len(a_frag) // M_BLOCKS
    b_k_stride = PAIRS * 4

    for pair in cutlass.range_constexpr(PAIRS):
        n_frag = pair * 2
        b_off = b_k_step * b_k_stride + pair * 4
        for m_block in cutlass.range_constexpr(M_BLOCKS):
            a_off = m_block * a_stride
            for half in cutlass.range_constexpr(2):
                s = (m_block * N_FRAGS + n_frag + half) * 4
                c0, c1, c2, c3 = mma_m16n8k16_f32(
                    a_frag[a_off + 0],
                    a_frag[a_off + 1],
                    a_frag[a_off + 2],
                    a_frag[a_off + 3],
                    b_frag[b_off + half * 2 + 0],
                    b_frag[b_off + half * 2 + 1],
                    acc[s + 0],
                    acc[s + 1],
                    acc[s + 2],
                    acc[s + 3],
                    ab_dtype,
                )
                acc[s + 0] = c0
                acc[s + 1] = c1
                acc[s + 2] = c2
                acc[s + 3] = c3


class SM120DetDqGemmKernel:
    """Deterministic dQ GEMM over the det_2kernel dS workspace.

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
            raise ValueError(f"det_2kernel dQ kv_tile must be a multiple of 64; got {kv_tile}")
        if q_tile % 16:
            raise ValueError(f"det_2kernel dQ q_tile must be a multiple of 16; got {q_tile}")
        if q_tile % ws_q_tile or ws_q_tile % 16:
            raise ValueError(f"det_2kernel dQ q_tile ({q_tile}) must be a multiple of ws_q_tile ({ws_q_tile}), ws_q_tile a multiple of 16")
        self.use_pdl = bool(use_pdl)
        self.page = 64 if head_dim % 64 == 0 else 32
        self.ds_page = 64  # panel pages along kv (kv_tile is a multiple of 64)
        self.tma_swizzle = cuda.TensorMapSwizzle.s128b if self.page == 64 else cuda.TensorMapSwizzle.s64b

        # One 16-row MMA block per compute warp, plus the TMA producer warp.
        self.num_compute_warps = q_tile // 16
        self.load_warp_id = self.num_compute_warps
        self.num_warps = self.num_compute_warps + 1
        self.threads = 32 * self.num_warps
        self.threads_pipeline = 32 * (self.num_compute_warps + 1)
        self.stages = 2
        self.k_tile_elems = kv_tile * head_dim
        self.ds_tile_elems = q_tile * kv_tile
        smem_bytes = self.stages * (self.k_tile_elems + self.ds_tile_elems) * in_dtype.bytes
        cap = cutlass.utils.get_smem_capacity_in_bytes("sm_120")
        if smem_bytes > cap:
            raise ValueError(f"det_2kernel dQ smem {smem_bytes} bytes exceeds the sm_120 cap of {cap} bytes")
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
        k_page_elems = self.kv_tile * self.page
        for pg in cutlass.range_constexpr(self.d // self.page):
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sK.subview(stage * self.k_tile_elems + pg * k_page_elems),
                    tma_k_desc.get_ptr(),
                    (pg * self.page, kv_head, kv_seq, batch),
                    mbar,
                )
        ds_page_elems = self.q_tile * self.ds_page
        for pg in cutlass.range_constexpr(self.kv_tile // self.ds_page):
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sDS.subview(stage * self.ds_tile_elems + pg * ds_page_elems),
                    tma_ds_desc.get_ptr(),
                    (kv_seq + pg * self.ds_page, q_base, q_head, batch),
                    mbar,
                )

    @cute.kernel
    def kernel(
        self,
        k: cute.Tensor,  # [B, SKV, HKV, D] io dtype
        ds_ws: cute.Tensor,  # [B, HQ, SQ, SKV_pad] io dtype (main-kernel dS, unscaled)
        dq: cute.Tensor,  # [B, SQ, HQ, D] io dtype out
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_ds_desc: cutlass.GridConstant[cuda.TensorMap],
        attn_scale: cutlass.Float32,
    ) -> None:
        io_dtype = self.in_dtype
        d = self.d
        M = self.q_tile
        N = self.kv_tile
        PAGE = self.page
        DS_PAGE = self.ds_page
        STAGES = self.stages
        KV_CHUNKS = N // 16
        DQ_NF = d // 8

        tidx, _, _ = cute.arch.thread_idx()
        m_block, q_head, batch = cute.arch.block_idx()
        lane = tidx % 32
        warp = cute.arch.warp_idx()
        g_lane = lane // 4
        p_lane = lane % 4

        SQ = dq.shape[1]
        SKV = k.shape[1]
        HQ = dq.shape[2]
        HKV = k.shape[2]
        GROUP = HQ // HKV
        kv_head = q_head // GROUP
        q_base = m_block * M
        PARTIAL_Q = (SQ % M) != 0

        # kv range: dense reads all; causal stops at this tile's last-row
        # (right-band widened) diagonal — same geometry as the main kernel.
        if cutlass.const_expr(self.is_causal):
            if cutlass.const_expr(self.causal_top_left):
                diag_off = cutlass.Int32(0)
            else:
                diag_off = SKV - SQ
            kv_hi = cute.math.min(cutlass.Int32(SKV), q_base + M + diag_off + self.right_slack)
            kv_hi = cute.math.max(kv_hi, cutlass.Int32(0))
        else:
            kv_hi = cutlass.Int32(SKV)
        n_iters = (kv_hi + N - 1) // N

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
            jp = cutlass.Int32(0)
            while (jp < cutlass.Int32(STAGES)) & (jp < n_iters):
                self.load_stage(sK, sDS, tma_mbar, tma_k_desc, tma_ds_desc, batch, kv_head, q_head, jp % STAGES, jp * N, q_base)
                jp += 1
            if n_iters > 0:
                while not prims.mbarrier_try_wait_parity(tma_mbar.subview(0), cutlass.Int32(0)):
                    pass
            jw = cutlass.Int32(1)
            jc = cutlass.Int32(0)
            while jc < n_iters:
                cute.arch.barrier(barrier_id=2, number_of_threads=self.threads_pipeline)
                if jw < n_iters:
                    if jw >= STAGES:
                        self.load_stage(sK, sDS, tma_mbar, tma_k_desc, tma_ds_desc, batch, kv_head, q_head, jw % STAGES, jw * N, q_base)
                    while not prims.mbarrier_try_wait_parity(tma_mbar.subview(jw % STAGES), (jw // STAGES) & cutlass.Int32(1)):
                        pass
                    jw += 1
                jc += 1

        elif warp < self.load_warp_id:  # compute warps
            row0 = warp * 16  # this warp's q rows within the tile
            acc_dq = cutlass.Array(cutlass.Float32, DQ_NF * 4, alignment=16)
            for i in cutlass.range_constexpr(DQ_NF * 4):
                acc_dq[i] = cutlass.Float32(0.0)

            if cutlass.const_expr(self.is_causal and self.ws_q_tile != M):
                ws_end = q_base + (row0 // self.ws_q_tile + 1) * self.ws_q_tile
                kv_hi_w = cute.math.min(cutlass.Int32(SKV), ws_end + diag_off + self.right_slack)
                kv_hi_w = cute.math.max(kv_hi_w, cutlass.Int32(0))
                n_iters_w = (kv_hi_w + N - 1) // N
            else:
                n_iters_w = n_iters

            jj = cutlass.Int32(0)
            while jj < n_iters:
                stage = jj % STAGES
                sK_stage = sK.subview(stage * self.k_tile_elems)
                sDS_stage = sDS.subview(stage * self.ds_tile_elems)
                cute.arch.barrier(barrier_id=2, number_of_threads=self.threads_pipeline)
                if jj < n_iters_w:
                    for kc in cutlass.range_constexpr(KV_CHUNKS):
                        af = load_a_frag(sDS_stage, kc, row0, lane, rows=M, page=DS_PAGE)
                        mma_bstream(
                            acc_dq,
                            [af[0], af[1], af[2], af[3]],
                            sK_stage,
                            b_k_step=kc,
                            M=16,
                            N=d,
                            b_trans=True,
                            b_rows=N,
                            b_page=PAGE,
                            lane=lane,
                            ab_dtype=io_dtype,
                        )
                jj += 1

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            r0 = q_base + row0 + g_lane
            r8 = r0 + 8
            dq_ptr = dq.iterator.raw_ptr()
            for nf in cutlass.range_constexpr(DQ_NF):
                col = nf * 8 + 2 * p_lane
                off = nf * 4
                base_top = ((batch * SQ + r0) * HQ + q_head) * d + col
                base_bot = ((batch * SQ + r8) * HQ + q_head) * d + col
                if (not cutlass.const_expr(PARTIAL_Q)) or (r0 < SQ):
                    (dq_ptr + base_top).store(
                        pack_half2(acc_dq[off + 0] * attn_scale, acc_dq[off + 1] * attn_scale, io_dtype),
                        alignment=4,
                    )
                if (not cutlass.const_expr(PARTIAL_Q)) or (r8 < SQ):
                    (dq_ptr + base_bot).store(
                        pack_half2(acc_dq[off + 2] * attn_scale, acc_dq[off + 3] * attn_scale, io_dtype),
                        alignment=4,
                    )

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,
        ds_ws: cute.Tensor,
        dq: cute.Tensor,
        attn_scale: cutlass.Float32,
        stream: cuda_driver.CUstream,
    ) -> None:
        box_k = (1, self.kv_tile, 1, self.page)
        tma_k_desc = cuda.create_tensor_map_tiled_from_view(k, box_dims=box_k, stride_order=(3, 2, 1, 0), swizzle=self.tma_swizzle)
        box_ds = (1, 1, self.q_tile, self.ds_page)
        tma_ds_desc = cuda.create_tensor_map_tiled_from_view(ds_ws, box_dims=box_ds, stride_order=(3, 2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
        n_q_tiles = cute.ceil_div(dq.shape[1], self.q_tile)
        self.kernel(k, ds_ws, dq, tma_k_desc, tma_ds_desc, attn_scale).launch(
            grid=(n_q_tiles, dq.shape[2], dq.shape[0]),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.min_blocks,
            use_pdl=self.use_pdl,
        )
