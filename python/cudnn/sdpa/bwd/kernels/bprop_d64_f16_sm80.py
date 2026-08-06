# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 (A100) SDPA backward for head-dim 64 — fp16 / bf16.

Fused dQ/dK/dV backward over a ``[B, S, H, D=64]`` BSHD layout.  One CTA per
``(kv-tile, head, batch)``: each CTA owns an ``n_block=128`` band of KV rows,
streams the full Q sequence in m-blocks of 64 q-rows, accumulates dK/dV for its
KV band, and scatter-adds dQ partials.

Per m-block (64 q-rows), accumulators in fp32, operands staged in swizzled SMEM
(Swz128B), the dataflow is:

    S  = Q · Kᵀ                       (acc_S  [q=64, kv=128], lane = q)
    P  = exp2(scale·S − lse_q)        (softmax row; lse precomputed in the fwd)
    dP = dO · Vᵀ                      (acc_dP [q=64, kv=128])
    dS = scale · P · (dP − Dq)        (Dq = rowsum(O∘dO), precomputed)
    dV += Pᵀ · dO                     (accumulated over q-blocks)
    dQ  = dS · K                      (scatter-added into dQ)
    dK += dSᵀ · Q                     (accumulated over q-blocks)

``softmax_scale`` is folded into dS so dQ and dK come out pre-scaled (dV uses the
unscaled P).  Q/dO are double-buffered with cp.async (Q 1-deep, dO 2-deep); the
next-tile loads are issued INSIDE the dP / dQ MMAs to overlap the global loads
with compute.  P and dS are staged to SMEM in natural ``[q, kv]`` layout so the
BMM2 gemms read them transposed via ``ldmatrix.trans``.

dQ accumulation: each warp's dQ C-fragment is atomicAdd-ed thread-major into a
PERMUTED-flat scratch ``dQ_acc`` ``[B, H, SQ, D]`` — a warp's 32 stores then hit
consecutive addresses (coalesced, no SMEM staging and no barrier in the hot
loop).  A separate ``_unpermute`` kernel reads that scratch back in fragment
order and writes the row-major dQ output (with the dtype cast).  Different
kv-tiles add to the same flat slot for the same ``(thread, element)``, so the
cross-tile accumulation is exact.

Config: m_block(q)=64, n_block(kv)=128, 256 threads (8 warps), d_qk == d_v == 64,
MHA (no GQA), dense (no mask).  Asserts ``S_q % 64 == 0`` and ``S_kv % 128 == 0``.

ABI: ``backward(Q,K,V,dO,O,lse,*,scale=None,do_dot=None) -> (dQ,dK,dV)`` — BSHD
in/out; ``lse`` natural-log ``[B,H,S_q]``.  The do_dot (rowsum O∘dO) preprocessing
reuses the shared ``bprop_f16_sm80`` device kernel.

Validation: f16 rel ~3.8e-4, bf16 ~3.4e-3 (dQ/dK/dV) at S=512; multi-tile
(S_q=768) also validated.
"""

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack as _from_dlpack_raw


def from_dlpack(t, **kw):
    """Vendoring shim: the kernels compile with --enable-tvm-ffi, so host-side
    conversions must produce TVM-FFI tensors regardless of the
    CUTE_DSL_ENABLE_TVM_FFI environment latch."""
    kw.setdefault("enable_tvm_ffi", True)
    return _from_dlpack_raw(t, **kw)


from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.mma import load_b_smem_x4, mma_step  # noqa: E402
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16  # noqa: E402
from cudnn.frost.tile_dsl.tma import load_tile_2d, cp_async_commit, cp_async_wait  # noqa: E402

from cudnn.sdpa.bwd.kernels import bprop_f16_sm80 as _base  # noqa: E402

_LOG2E = 1.4426950408889634
_ELEM_BYTES = 2
_COPY_ELEMS = 8  # 16-byte cp.async chunk (8 fp16)

# Fixed d=64 tile shape.
M_BLOCK = 64  # q rows per m-block
N_BLOCK = 128  # kv rows per CTA tile
NUM_WARPS = 8
NUM_THREADS = NUM_WARPS * 32  # 256


@cute.kernel
def _bprop_kernel(
    Q: cute.Tensor,  # [B, SQ,  H, D]  io_dtype
    K: cute.Tensor,  # [B, SKV, H, D]  io_dtype
    V: cute.Tensor,  # [B, SKV, H, D]  io_dtype
    dO: cute.Tensor,  # [B, SQ,  H, D]  io_dtype
    dQ_acc: cute.Tensor,  # [B, H, SQ, D]  fp32 PERMUTED-flat (coalesced atomic target)
    dK: cute.Tensor,  # [B, SKV, H, D]  io_dtype (output)
    dV: cute.Tensor,  # [B, SKV, H, D]  io_dtype (output)
    LSE: cute.Tensor,  # [B, H, SQ] fp32 (natural-log)
    DO_DOT: cute.Tensor,  # [B, H, SQ] fp32 (sum_d O*dO)
    d: cutlass.Constexpr[int],  # head dim (64)
    io_dtype: cutlass.Constexpr,
    n_q_tiles: cutlass.Int32,
    softmax_scale_log2: cutlass.Float32,  # scale * log2(e)
    attn_scale: cutlass.Float32,  # scale
):
    # ---- compile-time derived counts --------------------------------------
    D_CHUNKS = d // 16  # BMM1 k-reduce over d (4)
    _DLOG2 = (d).bit_length() - 1  # log2(d) for the linear-swizzle B-loads (d=64→6)
    Q_CHUNKS = M_BLOCK // 16  # BMM2 dV/dK k-reduce over q (4)
    KV_CHUNKS = N_BLOCK // 16  # dQ k-reduce over kv (8)
    # SdP warp grid: 4 warp-rows (q) × 2 warp-cols (kv).
    SDP_N_PER_WARP = N_BLOCK // 2  # 64 kv per warp
    SDP_N_FRAGS = SDP_N_PER_WARP // 8  # 8
    # dV/dK warp grid: 8 warp-rows (kv = 8×16=128) × 1 col (N = d = 64).
    DKV_N_FRAGS = d // 8  # 8
    # dQ warp grid: 4 warp-rows (q = 64) × 2 warp-cols (d split 2×32).
    DQ_N_PER_WARP = d // 2  # 32
    DQ_N_FRAGS = DQ_N_PER_WARP // 8  # 4

    bx, by, bz = cute.arch.block_idx()
    kv_tile = bx
    head = by
    batch = bz

    SQ = Q.shape[1]
    H = Q.shape[2]
    SKV = K.shape[1]

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = tidx // 32
    lane = tidx % 32
    g_lane = lane // 4  # 0..7  (C-frag row group)
    p_lane = lane % 4  # 0..3  (C-frag col pos)
    a_row = lane % 16  # ldmatrix.x4 A-frag row selector
    a_col = lane // 16  # ldmatrix.x4 A-frag col-subchunk selector

    # SdP warp bands.
    sdp_wr = warp_idx % 4  # 0..3 → q M-band ×16
    sdp_wc = warp_idx // 4  # 0..1 → kv N-band ×64
    # dQ warp bands (same wr/wc split, but wc → d N-band ×32).
    dq_wr = warp_idx % 4
    dq_wc = warp_idx // 4

    Q_view = cutlass.make_array_view(Q)
    K_view = cutlass.make_array_view(K)
    V_view = cutlass.make_array_view(V)
    dO_view = cutlass.make_array_view(dO)
    dQ_view = cutlass.make_array_view(dQ_acc)
    dK_view = cutlass.make_array_view(dK)
    dV_view = cutlass.make_array_view(dV)
    LSE_view = cutlass.make_array_view(LSE)
    DOT_view = cutlass.make_array_view(DO_DOT)

    QK_RS = cutlass.Int32(H) * cutlass.Int32(d)  # GMEM row stride for Q/K/dO/V (BSHD)
    kv_base = kv_tile * cutlass.Int32(N_BLOCK)

    # K/V tile base GMEM element pointers for this (batch, head).
    kv_row0_base = ((batch * cutlass.Int32(SKV) + kv_base) * cutlass.Int32(H) + head) * cutlass.Int32(d)
    k_tile_gmem = K_view.data_ptr() + kv_row0_base
    v_tile_gmem = V_view.data_ptr() + kv_row0_base

    lse_head_base = (batch * cutlass.Int32(H) + head) * cutlass.Int32(SQ)
    dot_head_base = lse_head_base

    bhead_q = (batch * cutlass.Int32(SQ) * cutlass.Int32(H) + head) * cutlass.Int32(d)

    # ---- SMEM tiles -------------------------------------------------------
    QSTAGE = M_BLOCK * d
    sK = cutlass.Array(io_dtype, N_BLOCK * d, alignment=128, space=cutlass.AddressSpace.smem)
    sV = cutlass.Array(io_dtype, N_BLOCK * d, alignment=128, space=cutlass.AddressSpace.smem)
    # Q ring (2 stages) + dO ring (2 stages).
    sQ = cutlass.Array(io_dtype, 2 * QSTAGE, alignment=128, space=cutlass.AddressSpace.smem)
    sdO = cutlass.Array(io_dtype, 2 * QSTAGE, alignment=128, space=cutlass.AddressSpace.smem)
    # P / dS staging in NATURAL [q, kv] layout (Swz128B), q=64 rows × kv=128 cols.
    sP = cutlass.Array(io_dtype, M_BLOCK * N_BLOCK, alignment=128, space=cutlass.AddressSpace.smem)
    sdS = cutlass.Array(io_dtype, M_BLOCK * N_BLOCK, alignment=128, space=cutlass.AddressSpace.smem)

    # ---- load K, V (full 128 rows; all 256 threads cooperate) -------------
    load_tile_2d(
        sK,
        k_tile_gmem,
        rows=N_BLOCK,
        elems_per_row=d,
        gmem_row_stride_elems=QK_RS,
        tidx=tidx,
        num_threads=NUM_THREADS,
        elems_per_copy=_COPY_ELEMS,
        elem_bytes=_ELEM_BYTES,
        swizzle=True,
    )
    load_tile_2d(
        sV,
        v_tile_gmem,
        rows=N_BLOCK,
        elems_per_row=d,
        gmem_row_stride_elems=QK_RS,
        tidx=tidx,
        num_threads=NUM_THREADS,
        elems_per_copy=_COPY_ELEMS,
        elem_bytes=_ELEM_BYTES,
        swizzle=True,
    )
    cp_async_commit()  # group: K / V

    SQ_rt = cutlass.Int32(SQ)
    HD = cutlass.Int32(H) * cutlass.Int32(d)
    # Pipeline fill: Q is 1-deep (tile 0), dO is 2-deep (tiles 0 AND 1).  Each
    # tensor load is its OWN cp.async commit group so the mainloop's
    # ``cp_async_wait(1)`` drains them in order ([K/V],[Q0],[dO0],[dO1]); the
    # next-tile loads are issued inside the dP / dQ MMAs below (not at top-of-loop).
    load_tile_2d(
        sQ,
        Q_view.data_ptr() + bhead_q,
        rows=M_BLOCK,
        elems_per_row=d,
        gmem_row_stride_elems=QK_RS,
        tidx=tidx,
        num_threads=NUM_THREADS,
        elems_per_copy=_COPY_ELEMS,
        elem_bytes=_ELEM_BYTES,
        swizzle=True,
        valid_rows=SQ_rt,
        row_base=cutlass.Int32(0),
    )
    cp_async_commit()  # group: Q tile 0 → slot 0
    load_tile_2d(
        sdO,
        dO_view.data_ptr() + bhead_q,
        rows=M_BLOCK,
        elems_per_row=d,
        gmem_row_stride_elems=QK_RS,
        tidx=tidx,
        num_threads=NUM_THREADS,
        elems_per_copy=_COPY_ELEMS,
        elem_bytes=_ELEM_BYTES,
        swizzle=True,
        valid_rows=SQ_rt,
        row_base=cutlass.Int32(0),
    )
    cp_async_commit()  # group: dO tile 0 → slot 0
    load_tile_2d(
        sdO.subview(cutlass.Int32(QSTAGE)),
        dO_view.data_ptr() + bhead_q + cutlass.Int32(M_BLOCK) * HD,
        rows=M_BLOCK,
        elems_per_row=d,
        gmem_row_stride_elems=QK_RS,
        tidx=tidx,
        num_threads=NUM_THREADS,
        elems_per_copy=_COPY_ELEMS,
        elem_bytes=_ELEM_BYTES,
        swizzle=True,
        valid_rows=SQ_rt,
        row_base=cutlass.Int32(M_BLOCK),
    )
    cp_async_commit()  # group: dO tile 1 → slot 1 (2-deep prefetch)

    # ---- accumulators (LOCAL, persistent across q-loop) -------------------
    # dV [kv=128, d=64]: each warp owns 16 kv-rows × d=64 → DKV_N_FRAGS*4.
    # dK [kv=128, d=64]: same.
    acc_dV = cutlass.Array(cutlass.Float32, DKV_N_FRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
    acc_dK = cutlass.Array(cutlass.Float32, DKV_N_FRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(DKV_N_FRAGS * 4):
        acc_dV[i] = cutlass.Float32(0.0)
        acc_dK[i] = cutlass.Float32(0.0)

    # =======================================================================
    # q-loop.
    # =======================================================================
    for q_iter in cutlass.range(n_q_tiles, unroll=1):
        q_base = q_iter * cutlass.Int32(M_BLOCK)
        stage_cur = q_iter % cutlass.Int32(2)
        stage_nxt = (q_iter + cutlass.Int32(1)) % cutlass.Int32(2)
        sQ_cur = sQ.subview(stage_cur * cutlass.Int32(QSTAGE))
        sdO_cur = sdO.subview(stage_cur * cutlass.Int32(QSTAGE))

        # Next-tile prefetch bases — issued INSIDE the dP / dQ MMAs below (to
        # overlap the cp.async with compute), NOT at the top of the loop:
        #   Q is 1-deep  → tile i+1 → ring slot stage_nxt (loaded during dP MMA);
        #   dO is 2-deep → tile i+2 → ring slot stage_cur (loaded during dQ MMA,
        #                  AFTER dV has consumed this iter's dO from stage_cur).
        q_next_base = q_base + cutlass.Int32(M_BLOCK)
        do_next_base = q_base + cutlass.Int32(2 * M_BLOCK)

        q_row_t = sdp_wr * cutlass.Int32(16) + g_lane  # C-frag q-row (top/bottom)
        q_row_b = q_row_t + cutlass.Int32(8)
        kv_col_base = sdp_wc * cutlass.Int32(SDP_N_PER_WARP)
        a_qrow = sdp_wr * cutlass.Int32(16) + a_row  # A-frag SMEM row (q), kc-invariant
        a_qrow_swz = a_qrow & cutlass.Int32(7)  # swizzle (row&7), kc-invariant
        a_qrow_off = a_qrow * cutlass.Int32(d)  # row*d, hoisted out of kc

        # ---- BMM1 S = Q · Kᵀ → acc_S [q=64, kv=128]; wait for Q tile then load A
        cp_async_wait(1)
        nvvm.barrier_cta_sync()
        a_S = cutlass.Array(cutlass.Int32, D_CHUNKS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
        for kc in cutlass.range_constexpr(D_CHUNKS):
            # col = kc*16 + a_col*8 → chunk = kc*2 + a_col; smem_col = (chunk ^ (row&7))*8.
            smem_col = ((cutlass.Int32(kc * 2) + a_col) ^ a_qrow_swz) * cutlass.Int32(8)
            p = sQ_cur.subview(a_qrow_off + smem_col)
            vv = nvvm.ldmatrix(p.data_ptr(), 4, nvvm.MMALayout.ROW)
            a_S[kc * 4 + 0] = vv[0]
            a_S[kc * 4 + 1] = vv[1]
            a_S[kc * 4 + 2] = vv[2]
            a_S[kc * 4 + 3] = vv[3]
        a_S_list = [a_S[i] for i in range(D_CHUNKS * 4)]

        acc_S = cutlass.Array(cutlass.Float32, SDP_N_FRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(SDP_N_FRAGS * 4):
            acc_S[i] = cutlass.Float32(0.0)
        sK_warp = sK.subview(sdp_wc * cutlass.Int32(SDP_N_PER_WARP) * cutlass.Int32(d))
        b_cur = load_b_smem_x4(
            sK_warp, k_step=0, N=SDP_N_PER_WARP, sB_elems_per_row=cutlass.Int32(d), b_trans=False, lane=lane, swizzle=True, row_stride_log2=_DLOG2
        )
        for kc in cutlass.range_constexpr(D_CHUNKS):
            if cutlass.const_expr(kc + 1 < D_CHUNKS):
                b_nxt = load_b_smem_x4(
                    sK_warp, k_step=kc + 1, N=SDP_N_PER_WARP, sB_elems_per_row=cutlass.Int32(d), b_trans=False, lane=lane, swizzle=True, row_stride_log2=_DLOG2
                )
            mma_step(acc_S, a_S_list, b_cur, k_step=kc, M=16, N=SDP_N_PER_WARP, ab_dtype=io_dtype)
            if cutlass.const_expr(kc + 1 < D_CHUNKS):
                b_cur = b_nxt

        # ---- softmax: P = exp2(scale·S − lse_q) → p_f (regs; written to sP after dP)
        p_f = cutlass.Array(cutlass.Float32, SDP_N_FRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
        lse_t = Pointer(LSE_view.data_ptr() + lse_head_base + q_base + q_row_t, dtype=cutlass.Float32).load() * cutlass.Float32(_LOG2E)
        lse_b = Pointer(LSE_view.data_ptr() + lse_head_base + q_base + q_row_b, dtype=cutlass.Float32).load() * cutlass.Float32(_LOG2E)
        for nf in cutlass.range_constexpr(SDP_N_FRAGS):
            off = nf * 4
            p_f[off + 0] = cute.math.exp2(acc_S[off + 0] * softmax_scale_log2 - lse_t, fastmath=True)
            p_f[off + 1] = cute.math.exp2(acc_S[off + 1] * softmax_scale_log2 - lse_t, fastmath=True)
            p_f[off + 2] = cute.math.exp2(acc_S[off + 2] * softmax_scale_log2 - lse_b, fastmath=True)
            p_f[off + 3] = cute.math.exp2(acc_S[off + 3] * softmax_scale_log2 - lse_b, fastmath=True)

        # ---- BMM1 dP = dO · Vᵀ → acc_dP [q=64, kv=128]; the next-iter Q tile is
        #      prefetched at k==0 of this MMA (overlap the cp.async with the HMMAs)
        cp_async_wait(1)
        nvvm.barrier_cta_sync()
        a_dP = cutlass.Array(cutlass.Int32, D_CHUNKS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
        for kc in cutlass.range_constexpr(D_CHUNKS):
            smem_col = ((cutlass.Int32(kc * 2) + a_col) ^ a_qrow_swz) * cutlass.Int32(8)
            p = sdO_cur.subview(a_qrow_off + smem_col)
            vv = nvvm.ldmatrix(p.data_ptr(), 4, nvvm.MMALayout.ROW)
            a_dP[kc * 4 + 0] = vv[0]
            a_dP[kc * 4 + 1] = vv[1]
            a_dP[kc * 4 + 2] = vv[2]
            a_dP[kc * 4 + 3] = vv[3]
        a_dP_list = [a_dP[i] for i in range(D_CHUNKS * 4)]

        acc_dP = cutlass.Array(cutlass.Float32, SDP_N_FRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(SDP_N_FRAGS * 4):
            acc_dP[i] = cutlass.Float32(0.0)
        sV_warp = sV.subview(sdp_wc * cutlass.Int32(SDP_N_PER_WARP) * cutlass.Int32(d))
        b_cur = load_b_smem_x4(
            sV_warp, k_step=0, N=SDP_N_PER_WARP, sB_elems_per_row=cutlass.Int32(d), b_trans=False, lane=lane, swizzle=True, row_stride_log2=_DLOG2
        )
        for kc in cutlass.range_constexpr(D_CHUNKS):
            if cutlass.const_expr(kc == 0):
                # prefetch Q tile i+1 → slot stage_nxt (1-deep), overlapped with the dP MMA.
                load_tile_2d(
                    sQ.subview(stage_nxt * cutlass.Int32(QSTAGE)),
                    Q_view.data_ptr() + bhead_q + q_next_base * HD,
                    rows=M_BLOCK,
                    elems_per_row=d,
                    gmem_row_stride_elems=QK_RS,
                    tidx=tidx,
                    num_threads=NUM_THREADS,
                    elems_per_copy=_COPY_ELEMS,
                    elem_bytes=_ELEM_BYTES,
                    swizzle=True,
                    valid_rows=SQ_rt,
                    row_base=q_next_base,
                )
                cp_async_commit()
            if cutlass.const_expr(kc + 1 < D_CHUNKS):
                b_nxt = load_b_smem_x4(
                    sV_warp, k_step=kc + 1, N=SDP_N_PER_WARP, sB_elems_per_row=cutlass.Int32(d), b_trans=False, lane=lane, swizzle=True, row_stride_log2=_DLOG2
                )
            mma_step(acc_dP, a_dP_list, b_cur, k_step=kc, M=16, N=SDP_N_PER_WARP, ab_dtype=io_dtype)
            if cutlass.const_expr(kc + 1 < D_CHUNKS):
                b_cur = b_nxt

        # ---- write sP (from p_f) → barrier → write sdS (dS computed inline).
        # The barrier between the two SMEM writes lets dV (which reads sP) overlap
        # with the sdS store; only p_f + acc_dP are held live (no separate dS array).
        sp_t_off = q_row_t * cutlass.Int32(N_BLOCK)
        sp_t_swz = q_row_t & cutlass.Int32(7)
        sp_b_off = q_row_b * cutlass.Int32(N_BLOCK)
        sp_b_swz = q_row_b & cutlass.Int32(7)
        for nf in cutlass.range_constexpr(SDP_N_FRAGS):
            off = nf * 4
            # kv_c = kv_col_base + nf*8 + 2p; chunk = kv_col_base//8 + nf, in_chunk = 2p.
            chunk = kv_col_base // cutlass.Int32(8) + cutlass.Int32(nf)
            in_chunk = cutlass.Int32(2) * p_lane
            scol_t = (chunk ^ sp_t_swz) * cutlass.Int32(8) + in_chunk
            scol_b = (chunk ^ sp_b_swz) * cutlass.Int32(8) + in_chunk
            Pointer(sP.subview(sp_t_off + scol_t).data_ptr(), dtype=cutlass.Int32).store(fp32_to_fp16(p_f[off + 0], p_f[off + 1], dtype=io_dtype), alignment=4)
            Pointer(sP.subview(sp_b_off + scol_b).data_ptr(), dtype=cutlass.Int32).store(fp32_to_fp16(p_f[off + 2], p_f[off + 3], dtype=io_dtype), alignment=4)

        nvvm.barrier_cta_sync()  # P fully written before dV reads Pᵀ

        # ---- dS = scale · P · (dP − Dq) → sdS (natural [q, kv]) ------------
        dd_t = Pointer(DOT_view.data_ptr() + dot_head_base + q_base + q_row_t, dtype=cutlass.Float32).load()
        dd_b = Pointer(DOT_view.data_ptr() + dot_head_base + q_base + q_row_b, dtype=cutlass.Float32).load()
        for nf in cutlass.range_constexpr(SDP_N_FRAGS):
            off = nf * 4
            ds0 = attn_scale * (acc_dP[off + 0] - dd_t) * p_f[off + 0]
            ds1 = attn_scale * (acc_dP[off + 1] - dd_t) * p_f[off + 1]
            ds2 = attn_scale * (acc_dP[off + 2] - dd_b) * p_f[off + 2]
            ds3 = attn_scale * (acc_dP[off + 3] - dd_b) * p_f[off + 3]
            chunk = kv_col_base // cutlass.Int32(8) + cutlass.Int32(nf)
            in_chunk = cutlass.Int32(2) * p_lane
            scol_t = (chunk ^ sp_t_swz) * cutlass.Int32(8) + in_chunk
            scol_b = (chunk ^ sp_b_swz) * cutlass.Int32(8) + in_chunk
            Pointer(sdS.subview(sp_t_off + scol_t).data_ptr(), dtype=cutlass.Int32).store(fp32_to_fp16(ds0, ds1, dtype=io_dtype), alignment=4)
            Pointer(sdS.subview(sp_b_off + scol_b).data_ptr(), dtype=cutlass.Int32).store(fp32_to_fp16(ds2, ds3, dtype=io_dtype), alignment=4)

        # ---- BMM2 dV += Pᵀ · dO  [kv=128, d=64], K=q=64 -------------------
        # Warp owns kv-rows [warp_idx*16, +16).  A = Pᵀ[kv, q]: P SMEM is natural
        # [q, kv]; ldmatrix.trans (COL) at SMEM (row=q, col=kv) yields the [kv, q]
        # A-frag.  B = dO[q, d] read transposed (contract q) → b_trans=True.
        # The transposed-A swizzle is kc-INVARIANT (SMEM col = kv n-band, SMEM row =
        # q contract, (kc*16+a_row)&7 = a_row&7), so hoist the swizzled column-offset
        # once; only the row offset (kc*16+a_row)*N_BLOCK moves per k-step.
        kvb = warp_idx * cutlass.Int32(16)
        tr_swz = (kvb // cutlass.Int32(8) + a_col) ^ (a_row & cutlass.Int32(7))  # invariant
        tr_scol = tr_swz * cutlass.Int32(8)  # in_chunk=0

        def _trans_a(sBase, kc):
            row = cutlass.Int32(kc * 16) + a_row
            v = nvvm.ldmatrix(sBase.subview(row * cutlass.Int32(N_BLOCK) + tr_scol).data_ptr(), 4, nvvm.MMALayout.COL)
            return [v[0], v[2], v[1], v[3]]  # COL→(TL,TR,BL,BR); mma wants (TL,BL,TR,BR)

        a_dV_cur = _trans_a(sP, 0)
        bdO_cur = load_b_smem_x4(sdO_cur, k_step=0, N=d, sB_elems_per_row=cutlass.Int32(d), b_trans=True, lane=lane, swizzle=True, row_stride_log2=_DLOG2)
        for kc in cutlass.range_constexpr(Q_CHUNKS):
            if cutlass.const_expr(kc + 1 < Q_CHUNKS):
                a_dV_nxt = _trans_a(sP, kc + 1)
                bdO_nxt = load_b_smem_x4(
                    sdO_cur, k_step=kc + 1, N=d, sB_elems_per_row=cutlass.Int32(d), b_trans=True, lane=lane, swizzle=True, row_stride_log2=_DLOG2
                )
            mma_step(acc_dV, a_dV_cur, bdO_cur, k_step=0, M=16, N=d, ab_dtype=io_dtype)
            if cutlass.const_expr(kc + 1 < Q_CHUNKS):
                a_dV_cur = a_dV_nxt
                bdO_cur = bdO_nxt

        nvvm.barrier_cta_sync()  # dS fully written before dQ / dK read sdS

        # ---- dQ = dS · K  [q=64, d=64], K=kv=128 → coalesced atomicAdd --------
        # Warp grid: dq_wr (0..3) → q M-band ×16; dq_wc (0..1) → d N-band ×32.
        # A = dS[q, kv] (this warp's q-band), via ldmatrix.x4 over KV_CHUNKS.
        # B = K[kv, d] read transposed (contract kv) → b_trans=True, this warp's
        #     d-band dq_wc*32.  The next-iter dO tile is prefetched at k==0 of this MMA.
        acc_dQ = cutlass.Array(cutlass.Float32, DQ_N_FRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(DQ_N_FRAGS * 4):
            acc_dQ[i] = cutlass.Float32(0.0)
        dq_qrow0 = dq_wr * cutlass.Int32(16) + a_row  # q SMEM row (kc-invariant)
        dq_qrow_off = dq_qrow0 * cutlass.Int32(N_BLOCK)  # row*N_BLOCK, hoisted
        dq_qrow_swz = dq_qrow0 & cutlass.Int32(7)  # (row&7), hoisted

        def _dq_a(kc):
            # col = kc*16 + a_col*8 → chunk = kc*2 + a_col; smem_col = (chunk^(row&7))*8.
            smem_col = ((cutlass.Int32(kc * 2) + a_col) ^ dq_qrow_swz) * cutlass.Int32(8)
            v = nvvm.ldmatrix(sdS.subview(dq_qrow_off + smem_col).data_ptr(), 4, nvvm.MMALayout.ROW)
            return [v[0], v[1], v[2], v[3]]

        a_dQ_cur = _dq_a(0)
        dq_colbase = dq_wc * cutlass.Int32(DQ_N_PER_WARP)
        bK_cur = load_b_smem_x4(
            sK, k_step=0, N=DQ_N_PER_WARP, sB_elems_per_row=cutlass.Int32(d), b_trans=True, lane=lane, swizzle=True, col_base=dq_colbase, row_stride_log2=_DLOG2
        )
        for kc in cutlass.range_constexpr(KV_CHUNKS):
            if cutlass.const_expr(kc == 0):
                # prefetch dO tile i+2 → slot stage_cur (2-deep), overlapped with the dQ MMA.
                # Safe to overwrite stage_cur: this iter's dV already consumed dO
                # from stage_cur, and dK (below) reads sQ_cur, not sdO.
                load_tile_2d(
                    sdO.subview(stage_cur * cutlass.Int32(QSTAGE)),
                    dO_view.data_ptr() + bhead_q + do_next_base * HD,
                    rows=M_BLOCK,
                    elems_per_row=d,
                    gmem_row_stride_elems=QK_RS,
                    tidx=tidx,
                    num_threads=NUM_THREADS,
                    elems_per_copy=_COPY_ELEMS,
                    elem_bytes=_ELEM_BYTES,
                    swizzle=True,
                    valid_rows=SQ_rt,
                    row_base=do_next_base,
                )
                cp_async_commit()
            if cutlass.const_expr(kc + 1 < KV_CHUNKS):
                a_dQ_nxt = _dq_a(kc + 1)
                bK_nxt = load_b_smem_x4(
                    sK,
                    k_step=kc + 1,
                    N=DQ_N_PER_WARP,
                    sB_elems_per_row=cutlass.Int32(d),
                    b_trans=True,
                    lane=lane,
                    swizzle=True,
                    col_base=dq_colbase,
                    row_stride_log2=_DLOG2,
                )
            mma_step(acc_dQ, a_dQ_cur, bK_cur, k_step=0, M=16, N=DQ_N_PER_WARP, ab_dtype=io_dtype)
            if cutlass.const_expr(kc + 1 < KV_CHUNKS):
                a_dQ_cur = a_dQ_nxt
                bK_cur = bK_nxt

        # ---- dQ coalesced atomicAdd into the PERMUTED-flat scratch -----------
        # Write acc_dQ thread-major: thread tidx, fragment element j →
        # flat[(b*H+h)*SQ*d + q_base*d + j*NUM_THREADS + tidx], so a warp's 32
        # atomicAdds for a fixed j hit consecutive addresses and coalesce (no SMEM
        # staging, no barrier).  kv-tiles add to the same slot for the same (tidx,j)
        # → cross-tile accumulation is exact; the (tidx,j)→(q-row,d-col) permutation
        # is undone by the _unpermute postprocess kernel.
        dq_perm_base = ((batch * cutlass.Int32(H) + head) * cutlass.Int32(SQ) + q_base) * cutlass.Int32(d) + tidx
        for nf in cutlass.range_constexpr(DQ_N_FRAGS):
            off = nf * 4
            for s in cutlass.range_constexpr(4):
                _base._atomic_add_f32(dQ_view.data_ptr() + dq_perm_base + cutlass.Int32((nf * 4 + s) * NUM_THREADS), acc_dQ[off + s])

        # ---- BMM2 dK += dSᵀ · Q  [kv=128, d=64], K=q=64 ------------------
        a_dK_cur = _trans_a(sdS, 0)
        bQ_cur = load_b_smem_x4(sQ_cur, k_step=0, N=d, sB_elems_per_row=cutlass.Int32(d), b_trans=True, lane=lane, swizzle=True, row_stride_log2=_DLOG2)
        for kc in cutlass.range_constexpr(Q_CHUNKS):
            if cutlass.const_expr(kc + 1 < Q_CHUNKS):
                a_dK_nxt = _trans_a(sdS, kc + 1)
                bQ_nxt = load_b_smem_x4(
                    sQ_cur, k_step=kc + 1, N=d, sB_elems_per_row=cutlass.Int32(d), b_trans=True, lane=lane, swizzle=True, row_stride_log2=_DLOG2
                )
            mma_step(acc_dK, a_dK_cur, bQ_cur, k_step=0, M=16, N=d, ab_dtype=io_dtype)
            if cutlass.const_expr(kc + 1 < Q_CHUNKS):
                a_dK_cur = a_dK_nxt
                bQ_cur = bQ_nxt

    cp_async_wait(0)
    nvvm.barrier_cta_sync()

    # =======================================================================
    # Epilogue — write dV / dK.  Each warp owns kv-rows [warp_idx*16, +16),
    # full d=64 cols.  C-frag: kv-row = warp_idx*16 + g_lane[+8]; d-col = nf*8 +
    # 2p[+1].  Vectorized 64-bit (2 adjacent fp32→half2) stores.
    # =======================================================================
    kv_r_t = warp_idx * cutlass.Int32(16) + g_lane
    kv_r_b = kv_r_t + cutlass.Int32(8)
    base_t = ((batch * cutlass.Int32(SKV) + kv_base + kv_r_t) * cutlass.Int32(H) + head) * cutlass.Int32(d)
    base_b = ((batch * cutlass.Int32(SKV) + kv_base + kv_r_b) * cutlass.Int32(H) + head) * cutlass.Int32(d)
    for nf in cutlass.range_constexpr(DKV_N_FRAGS):
        off = nf * 4
        dcol = cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane
        Pointer((dV_view.data_ptr() + base_t + dcol), dtype=cutlass.Int32).store(fp32_to_fp16(acc_dV[off + 0], acc_dV[off + 1], dtype=io_dtype), alignment=4)
        Pointer((dV_view.data_ptr() + base_b + dcol), dtype=cutlass.Int32).store(fp32_to_fp16(acc_dV[off + 2], acc_dV[off + 3], dtype=io_dtype), alignment=4)
        Pointer((dK_view.data_ptr() + base_t + dcol), dtype=cutlass.Int32).store(fp32_to_fp16(acc_dK[off + 0], acc_dK[off + 1], dtype=io_dtype), alignment=4)
        Pointer((dK_view.data_ptr() + base_b + dcol), dtype=cutlass.Int32).store(fp32_to_fp16(acc_dK[off + 2], acc_dK[off + 3], dtype=io_dtype), alignment=4)


@cute.kernel
def _unpermute_dq_kernel(
    dQ_acc: cute.Tensor,  # [B, H, SQ, d] fp32 PERMUTED-flat scratch (atomic target)
    dQ_out: cute.Tensor,  # [B, SQ, H, d] io_dtype (row-major output)
    d: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
):
    # Read the permuted-flat dQ scratch back in C-fragment order, cast to io_dtype,
    # and write the row-major dQ output.  A SEPARATE launch so no transpose lands
    # in the main backward hot loop.  Inverts the main kernel's
    # (tidx, j=nf*4+s) → (q-row, d-col) C-fragment mapping:
    #   warp=tidx//32, lane=tidx%32; dq_wr=warp%4, dq_wc=warp//4;
    #   g=lane//4, p=lane%4; row = dq_wr*16 + g + (8 if s>=2); col = dq_wc*32 + nf*8 + 2p + (s&1).
    qt, head, batch = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    SQ = dQ_out.shape[1]
    H = dQ_out.shape[2]
    DQ_N_FRAGS = (d // 2) // 8
    warp = tidx // 32
    lane = tidx % 32
    dq_wr = warp % 4
    dq_wc = warp // 4
    g_lane = lane // 4
    p_lane = lane % 4

    acc_view = cutlass.make_array_view(dQ_acc)
    out_view = cutlass.make_array_view(dQ_out)
    # Permuted-flat read base for this (batch, head, q-tile): flat = base + j*256 + tidx.
    perm_base = ((batch * cutlass.Int32(H) + head) * cutlass.Int32(SQ) + qt * cutlass.Int32(M_BLOCK)) * cutlass.Int32(d) + tidx
    q_abs_t = qt * cutlass.Int32(M_BLOCK) + dq_wr * cutlass.Int32(16) + g_lane
    q_abs_b = q_abs_t + cutlass.Int32(8)
    out_t = ((batch * cutlass.Int32(SQ) + q_abs_t) * cutlass.Int32(H) + head) * cutlass.Int32(d)
    out_b = ((batch * cutlass.Int32(SQ) + q_abs_b) * cutlass.Int32(H) + head) * cutlass.Int32(d)
    for nf in cutlass.range_constexpr(DQ_N_FRAGS):
        v0 = Pointer((acc_view.data_ptr() + perm_base + cutlass.Int32((nf * 4 + 0) * NUM_THREADS)), dtype=cutlass.Float32).load()
        v1 = Pointer((acc_view.data_ptr() + perm_base + cutlass.Int32((nf * 4 + 1) * NUM_THREADS)), dtype=cutlass.Float32).load()
        v2 = Pointer((acc_view.data_ptr() + perm_base + cutlass.Int32((nf * 4 + 2) * NUM_THREADS)), dtype=cutlass.Float32).load()
        v3 = Pointer((acc_view.data_ptr() + perm_base + cutlass.Int32((nf * 4 + 3) * NUM_THREADS)), dtype=cutlass.Float32).load()
        dcol = dq_wc * cutlass.Int32(32) + cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane
        Pointer((out_view.data_ptr() + out_t + dcol), dtype=cutlass.Int32).store(fp32_to_fp16(v0, v1, dtype=io_dtype), alignment=4)
        Pointer((out_view.data_ptr() + out_b + dcol), dtype=cutlass.Int32).store(fp32_to_fp16(v2, v3, dtype=io_dtype), alignment=4)


@cute.jit
def _unpermute_host(
    dQ_acc: cute.Tensor, dQ_out: cute.Tensor, d: cutlass.Constexpr[int], io_dtype: cutlass.Constexpr, n_q_tiles: cutlass.Int32, stream: cuda.CUstream
):
    H = dQ_out.shape[2]
    B = dQ_out.shape[0]
    _unpermute_dq_kernel(dQ_acc, dQ_out, d, io_dtype).launch(grid=(n_q_tiles, H, B), block=(NUM_THREADS, 1, 1), stream=stream)


@cute.jit
def _bprop_host(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    dO: cute.Tensor,
    dQ_acc: cute.Tensor,
    dK: cute.Tensor,
    dV: cute.Tensor,
    LSE: cute.Tensor,
    DO_DOT: cute.Tensor,
    d: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    n_q_tiles: cutlass.Int32,
    softmax_scale_log2: cutlass.Float32,
    attn_scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    SKV = K.shape[1]
    H = Q.shape[2]
    B = Q.shape[0]
    n_kv_tiles = (SKV + N_BLOCK - 1) // N_BLOCK
    _bprop_kernel(Q, K, V, dO, dQ_acc, dK, dV, LSE, DO_DOT, d, io_dtype, n_q_tiles, softmax_scale_log2, attn_scale).launch(
        grid=(n_kv_tiles, H, B), block=(NUM_THREADS, 1, 1), stream=stream
    )


@lru_cache(maxsize=None)
def _compile_main(B, H, SQ, SKV, d, io_is_bf16):
    io_dtype = cutlass.BFloat16 if io_is_bf16 else cutlass.Float16
    mk = cute.runtime.make_fake_compact_tensor
    fq = mk(io_dtype, (B, SQ, H, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    fk = mk(io_dtype, (B, SKV, H, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    fv = mk(io_dtype, (B, SKV, H, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    fdo = mk(io_dtype, (B, SQ, H, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    # dQ_acc is the PERMUTED-flat accumulator [B, H, SQ, d] (NOT the row-major
    # [B,SQ,H,d] output) — the main kernel atomicAdds into it thread-major.
    fdq = mk(cutlass.Float32, (B, H, SQ, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    fdk = mk(io_dtype, (B, SKV, H, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    fdv = mk(io_dtype, (B, SKV, H, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    fl = mk(cutlass.Float32, (B, H, SQ), stride_order=(2, 1, 0), assumed_align=16)
    fdt = mk(cutlass.Float32, (B, H, SQ), stride_order=(2, 1, 0), assumed_align=16)
    return cute.compile(
        _bprop_host,
        fq,
        fk,
        fv,
        fdo,
        fdq,
        fdk,
        fdv,
        fl,
        fdt,
        d,
        io_dtype,
        cutlass.Int32(0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cuda.CUstream(0),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=None)
def _compile_unpermute(B, H, SQ, d, io_is_bf16):
    io_dtype = cutlass.BFloat16 if io_is_bf16 else cutlass.Float16
    mk = cute.runtime.make_fake_compact_tensor
    fdq_acc = mk(cutlass.Float32, (B, H, SQ, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    fdq_out = mk(io_dtype, (B, SQ, H, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    return cute.compile(_unpermute_host, fdq_acc, fdq_out, d, io_dtype, cutlass.Int32(0), cuda.CUstream(0), options="--enable-tvm-ffi")


def backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    O: torch.Tensor,
    lse: torch.Tensor,
    *,
    scale: Optional[float] = None,
    do_dot: Optional[torch.Tensor] = None,
    **_ignored,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SDPA backward for head-dim 64, fp16/bf16.  BSHD in/out; ``lse`` natural-log
    [B,H,S_q].  Returns (dQ, dK, dV)."""
    assert Q.dtype in (torch.float16, torch.bfloat16)
    assert K.dtype == Q.dtype == V.dtype == dO.dtype == O.dtype
    io_is_bf16 = Q.dtype == torch.bfloat16
    B, SQ, H, D = Q.shape
    _, SKV, Hk, D_v = V.shape
    assert Hk == H, "MHA only (no GQA)"
    assert D == D_v == 64, "this kernel is d_qk == d_v == 64 only"
    assert SQ % M_BLOCK == 0, f"S_q ({SQ}) must be a multiple of {M_BLOCK}"
    assert SKV % N_BLOCK == 0, f"S_kv ({SKV}) must be a multiple of {N_BLOCK}"
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    scale_log2 = scale * math.log2(math.e)

    dev = Q.device
    # PERMUTED-flat dQ scratch [B, H, SQ, D] — the main kernel atomicAdds into it
    # thread-major; the _unpermute kernel casts it → row-major dQ.
    dQ_acc = torch.zeros(B, H, SQ, D, dtype=torch.float32, device=dev)
    dK = torch.empty(B, SKV, H, D, dtype=Q.dtype, device=dev)
    dV = torch.empty(B, SKV, H, D, dtype=Q.dtype, device=dev)
    dQ = torch.empty(B, SQ, H, D, dtype=Q.dtype, device=dev)
    lse_t = lse.to(dtype=torch.float32, device=dev).contiguous()

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # do_dot (rowsum O∘dO) preprocessing reuses the shared device kernel.
    if do_dot is None:
        dot_t = torch.empty(B, H, SQ, dtype=torch.float32, device=dev)
        dd_fn = _base._compile_do_dot(B, H, SQ, D, io_is_bf16)
        dd_fn(from_dlpack(O), from_dlpack(dO), from_dlpack(dot_t), cutlass.Int32(B * H * SQ), stream)
    else:
        dot_t = do_dot.to(dtype=torch.float32, device=dev).contiguous()

    fn = _compile_main(B, H, SQ, SKV, D, io_is_bf16)
    fn(
        from_dlpack(Q),
        from_dlpack(K),
        from_dlpack(V),
        from_dlpack(dO),
        from_dlpack(dQ_acc),
        from_dlpack(dK),
        from_dlpack(dV),
        from_dlpack(lse_t),
        from_dlpack(dot_t),
        cutlass.Int32((SQ + M_BLOCK - 1) // M_BLOCK),
        cutlass.Float32(scale_log2),
        cutlass.Float32(scale),
        stream,
    )

    # Un-permute the dQ scratch → row-major dQ + cast (separate launch).
    up_fn = _compile_unpermute(B, H, SQ, D, io_is_bf16)
    up_fn(from_dlpack(dQ_acc), from_dlpack(dQ), cutlass.Int32((SQ + M_BLOCK - 1) // M_BLOCK), stream)
    return dQ, dK, dV
