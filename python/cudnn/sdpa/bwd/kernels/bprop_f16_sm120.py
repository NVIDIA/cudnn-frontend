# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FROST SM120 SDPA backward kernel template (fp16 / bf16).

A fused multi-head attention (FMHA) backward for the NVIDIA Blackwell
GeForce SM120 family (SM120 and SM121) using TMA loads and a
warp-specialized producer/consumer schedule.

The backward follows the FlashAttention-2 algorithm (Dao, 2023;
https://github.com/Dao-AILab/flash-attention, BSD-3-Clause): a
KV-stationary, seq-KV-parallel single pass in which each CTA owns one KV
tile and walks the query tiles in descending order, computing the five
chained GEMMs (S = Q*K^T, dP = dO*V^T, dV += P^T*dO, dQ = dS*K,
dK += dS^T*Q) with the softmax VJP fused in registers. dK/dV accumulate
in registers across the whole pass (no atomics); dQ is reduced through an
fp32 workspace and finalized by a small convert kernel.

Constraints:
* Supported input dtypes: Float16 and BFloat16 (output dtype matches)
* Head dimension must be one of 32, 64, 128, 192, or 256; any other
  multiple of 8 up to 256 computes on the next of those sizes with the TMA
  envelope zero-filling the pad columns in place, so every multiple of 8 serves
  without staging copies.
* GQA/MQA: H_q must be a multiple of H_kv
* No dropout/alibi/softcap
* Optional causal (top-left or bottom-right), right-band-widened causal
  (window_size_right), sliding-window masks and padding masks.
* LSE input is the natural-log forward stats, fp32 (B, H, SQ); any
  non-broadcast layout
* io tensors may declare any dense layout whose head dim is
  innermost-contiguous and whose batch/seq/head strides are 16-byte
  multiples (TMA's global-stride rule)

One backward call is three kernel launches through the per-shape
``compile()`` cache at the bottom of this module: ``dot`` (delta =
rowsum(dO*O), also zeroes the dq_accum workspace), ``main`` (the fused
five-GEMM pass writing dK/dV), and ``cvt`` (dq_accum fp32 -> dQ io dtype);
GQA adds ``reduce`` and a dSink_token output adds ``dsink``.
"""

from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.utils
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.experimental import primitives as prims
from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16
from cudnn.frost.tile_dsl.mma import mma_m16n8k16_f32
from cudnn.frost.tile_dsl.swizzle import swizzle_xor
from cudnn.sdpa.bwd.config_sm120 import SUPPORTED_HEAD_DIMS, TemplateParams, padded_head_dims, validate_params

# The FROST loader injects one immutable specialization before executing this
# module. A direct import uses the dense FP16 defaults.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
validate_params(PARAMS)

STORAGE_DTYPE = {DTYPE_FP16: cutlass.Float16, DTYPE_BF16: cutlass.BFloat16}[PARAMS.dtype_qkv]

_LOG2E = 1.4426950408889634
_COPY_ELEMS = 8  # 16-byte gmem<->smem chunk (8 fp16/bf16)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def largest_warp_partition(m_dim: int, *n_dims: int) -> int:
    """Largest valid 2-D warp-partition factor A for one GEMM's (M, N) pair.

    The warp grid is (A, 8 // A): the A warps split m_dim into 16-row MMA
    (m16n8k16) blocks, so m_dim must be a multiple of 16 * A, and the
    per-warp N slice (n_dim * A / 8) must be a multiple of 16 (ldmatrix.x4
    pairs). The dK/dV pair shares one partition, so it passes both head
    dims as ``n_dims`` and A must validate for each. Used when a macro-tile
    override deviates from the per-head-dim CONFIG default, whose
    hand-tuned partitions only validate for the default tiles; "largest
    valid" is a heuristic, not a sweep winner.
    """

    for a in (8, 4, 2, 1):
        if m_dim % (16 * a) == 0 and all((n_dim * a // 8) % 16 == 0 for n_dim in n_dims):
            return a
    raise ValueError(f"no valid warp partition for M{m_dim} N{n_dims}")


@cute.jit
def tile_ptr(
    sbuf,
    row: cutlass.Int32,
    col: cutlass.Int32,
    *,
    page: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
):
    """Element pointer into a paged+swizzled smem tile."""
    pg = col // page
    in_col = col % page
    off = pg * (rows * page) + row * page + swizzle_xor(row, in_col, page, 2)
    return sbuf.subview(off).data_ptr()


@cute.jit
def pack_half2(lo, hi, dtype: cutlass.Constexpr[Type[cutlass.Numeric]]):
    """Pack two fp32 into a 2-element io-dtype vector (one 4 B store)."""
    return cutlass.Vector.from_elements((lo.to(dtype), hi.to(dtype)), dtype)


@cute.jit
def _red_add_f32x2(ptr, v0: cutlass.Float32, v1: cutlass.Float32) -> None:
    """One red.global.add.v2.f32 covering a thread's adjacent (c0,c1) pair."""
    prims.inline_ptx(
        "red.global.add.v2.f32 [$0], {$1, $2};",
        read_only_args=[ptr, v0, v1],
    )


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


@cute.jit
def _bwd_gemm4_dq(
    acc_dq,
    sdS,
    sK,
    wq,
    wd_q,
    lane,
    *,
    DQ_REPS: cutlass.Constexpr[int],
    DQ_NF: cutlass.Constexpr[int],
    KV_CHUNKS: cutlass.Constexpr[int],
    WM_DQ: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    PDS: cutlass.Constexpr[int],
    PAGE: cutlass.Constexpr[int],
    DQ_PER: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    """GEMM 4: acc_dq = dS @ K^T (reads only sdS/sK, never sQ)."""
    for i in cutlass.range_constexpr(DQ_REPS * DQ_NF * 4):
        acc_dq[i] = cutlass.Float32(0.0)
    for kc in cutlass.range_constexpr(KV_CHUNKS):
        af = []
        for rep in cutlass.range_constexpr(DQ_REPS):
            sf = load_a_frag(sdS, kc, wq * 16 + rep * 16 * WM_DQ, lane, rows=M, page=PDS)
            af = af + [sf[0], sf[1], sf[2], sf[3]]
        mma_bstream(
            acc_dq,
            af,
            sK,
            b_k_step=kc,
            M=16 * DQ_REPS,
            N=DQ_PER,
            b_trans=True,
            b_rows=N,
            b_page=PAGE,
            lane=lane,
            ab_dtype=io_dtype,
            col_base=wd_q * DQ_PER,
        )


@cute.jit
def _bwd_dq_scatter(
    acc_dq,
    dqa_ptr,
    dqa_base,
    math_tidx,
    H,
    *,
    DQ_REPS: cutlass.Constexpr[int],
    DQ_NF: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    d_qk: cutlass.Constexpr[int],
):
    """dQ accumulate into the scrambled dq_accum workspace."""
    t_r = math_tidx // 32
    t_c = math_tidx % 32
    for rep in cutlass.range_constexpr(DQ_REPS):
        for nf in cutlass.range_constexpr(DQ_NF):
            for hv in cutlass.range_constexpr(2):
                i_pair = hv + rep * 2 + nf * 2 * DQ_REPS
                if cutlass.const_expr(d_qk >= 64):
                    jm = i_pair % (M // 8)
                    jn = i_pair // (M // 8)
                    addr = dqa_base + (t_r + jm * 8) * (H * d_qk) + t_c * 2 + jn * 64
                else:
                    addr = dqa_base + (t_r + (t_c // 16) * 8 + i_pair * 16) * (H * d_qk) + (t_c % 16) * 2
                poff = (rep * DQ_NF + nf) * 4 + hv * 2
                _red_add_f32x2(dqa_ptr + addr, acc_dq[poff + 0], acc_dq[poff + 1])


@cute.jit
def _bwd_gemm5_dk(
    acc_dk,
    sdS,
    sQ_st,
    wn_k,
    wd_k,
    lane,
    *,
    Q_CHUNKS: cutlass.Constexpr[int],
    DKV_REPS: cutlass.Constexpr[int],
    WM_DKV: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    PDS: cutlass.Constexpr[int],
    PAGE: cutlass.Constexpr[int],
    DK_PER: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    """GEMM 5: acc_dk += dS^T @ Q (the iteration's last sQ reader)."""
    for kc in cutlass.range_constexpr(Q_CHUNKS):
        af = []
        for rep in cutlass.range_constexpr(DKV_REPS):
            sf = load_a_frag_transposed(
                sdS,
                kc,
                wn_k * 16 + rep * 16 * WM_DKV,
                lane,
                rows=M,
                page=PDS,
            )
            af = af + [sf[0], sf[2], sf[1], sf[3]]
        mma_bstream(
            acc_dk,
            af,
            sQ_st,
            b_k_step=kc,
            M=16 * DKV_REPS,
            N=DK_PER,
            b_trans=True,
            b_rows=M,
            b_page=PAGE,
            lane=lane,
            ab_dtype=io_dtype,
            col_base=wd_k * DK_PER,
        )


@cute.jit
def _bwd_det_wait(det_sem, m_block, det_turn, warp):
    """Deterministic-relay entry: block the 8 compute warps until it is this
    CTA's turn for q-tile ``m_block`` (FA3 / cuDNN-SM90 STAGES=4 scheme).

    One elected lane of warp 0 spins on an acquire load of the turn counter;
    barrier 6 (compute warps only — never the producer's 288-thread
    barriers) releases the other warps into the dQ scatter."""
    if warp == 0:
        if prims.elect_sync():
            while (
                prims.load_ext(
                    det_sem + m_block,
                    order=prims.MemOrder.ACQUIRE,
                    scope=prims.MemScope.GPU,
                )
                != det_turn
            ):
                pass
    cute.arch.barrier(barrier_id=6, number_of_threads=256)


@cute.jit
def _bwd_det_release(det_sem, m_block, det_turn, warp):
    """Deterministic-relay exit: pass the turn for ``m_block`` to the next
    kv tile once every compute warp has issued its dQ reds.

    The release store alone orders the relaxed reds before the handoff: the
    barrier sequences the other warps' reds against this thread (CTA scope)
    and st.release makes them cumulatively visible at GPU scope (its own
    membar; an extra fence here doubled the per-handoff drain cost)."""
    cute.arch.barrier(barrier_id=6, number_of_threads=256)
    if warp == 0:
        if prims.elect_sync():
            prims.store_ext(
                (det_turn + 1).ir_value(),
                det_sem + m_block,
                order=prims.MemOrder.RELEASE,
                scope=prims.MemScope.GPU,
            )


# ---------------------------------------------------------------------------
# Main kernel.
# ---------------------------------------------------------------------------


class SM120FusedMultiHeadAttentionFP16Backward:
    """Configure and launch the SM120 FMHA backward kernel chain."""

    DEFAULT_TILES = {
        32: (128, 64),
        64: (64, 128),
        128: (64, 64),
        192: (32, 64),
        256: (32, 64),
    }
    # (d_qk, q_tile, kv_tile) -> (warps_m_sdp, warps_m_dkv, warps_m_dq): for each
    # GEMM the 8 compute warps form an (A, 8 // A) grid; the value is A, the
    # warp count along that GEMM's own M (row) axis.
    CONFIG = {
        (32, 128, 64): (4, 4, 8),  # default for d32
        (32, 128, 128): (4, 8, 4),  # for very long S
        (64, 64, 128): (4, 8, 4),  # default for d64
        (64, 128, 64): (8, 2, 4),  # For underfilled grids, kv64 can double CTA counts
        (128, 64, 64): (2, 1, 4),  # default for d128
        (192, 32, 64): (2, 4, 2),  # default for d192
        (256, 32, 64): (2, 4, 2),  # default for d256
    }

    def __init__(
        self,
        in_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        is_causal: bool = False,
        causal_top_left: bool = False,
        window_size_left: int | None = None,
        window_size_right: int | None = None,
        deterministic: bool = False,
        head_dim_qk: int = 128,
        head_dim_v: int = 0,  # 0 = same as head_dim_qk.
        use_pdl: bool = True,
        q_tile: int = 0,
        kv_tile: int = 0,
        seq_kv_lens_present: bool = False,
        seq_q_lens_present: bool = False,
        sink_present: bool = False,
    ):
        self.in_dtype = in_dtype
        self.is_causal = is_causal
        self.causal_top_left = bool(causal_top_left)
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.right_slack = window_size_right if window_size_right is not None else 0
        self.deterministic = bool(deterministic)
        self.seq_kv_lens_present = bool(seq_kv_lens_present)
        self.seq_q_lens_present = bool(seq_q_lens_present)
        # sink LSE is finite on padded rows; trim them explicitly (LSE := +inf, P = 0)
        self.trim_q_rows = bool(sink_present) and self.seq_q_lens_present
        # Use padded head dim for compute
        self.d_qk_orig = int(head_dim_qk)
        self.d_v_orig = int(head_dim_v) or int(head_dim_qk)
        for orig, tag in ((self.d_qk_orig, "d_qk"), (self.d_v_orig, "d_v")):
            if orig % 8 or orig <= 0:
                raise ValueError(f"{tag} must be a positive multiple of 8; got {orig}")
        pads = padded_head_dims(self.d_qk_orig, self.d_v_orig)
        if pads is None:
            raise ValueError(f"head dims must be <= {max(SUPPORTED_HEAD_DIMS)}; got d_qk={self.d_qk_orig}, d_v={self.d_v_orig}")
        self.d_qk, self.d_v = pads
        head_dim_qk = self.d_qk
        # current MLA requires both to be multiples of 64 so one smem swizzle serves every tile.
        if self.d_v != head_dim_qk and (head_dim_qk % 64 or self.d_v % 64):
            raise ValueError(f"unequal head dims must both be multiples of 64; got d_qk={head_dim_qk}, d_v={self.d_v}")
        self.qk_envelope = self.d_qk_orig != self.d_qk
        self.v_envelope = self.d_v_orig != self.d_v
        self.use_pdl = bool(use_pdl)
        self.q_tile, self.kv_tile = self.DEFAULT_TILES[head_dim_qk]
        if q_tile:
            self.q_tile = int(q_tile)
        if kv_tile:
            self.kv_tile = int(kv_tile)
        if 128 % self.q_tile:
            # dq_accum is scrambled in 128-row blocks (SQ_R is rounded to 128
            # and the convert kernel unscrambles per block).
            raise ValueError(f"q_tile must divide 128; got {self.q_tile}")
        # Warp layouts: the sweep-tuned triple for this exact tile choice
        # when we have one, else the largest-valid derivation
        tuned = self.CONFIG.get((head_dim_qk, self.q_tile, self.kv_tile))
        if tuned is not None:
            self.warps_m_sdp, self.warps_m_dkv, self.warps_m_dq = tuned
            # The tuned dK partition does not tile the dV head dim.
            if (self.d_v * self.warps_m_dkv // 8) % 16:
                self.warps_m_dkv = largest_warp_partition(self.kv_tile, head_dim_qk, self.d_v)
        else:
            self.warps_m_sdp = largest_warp_partition(self.q_tile, self.kv_tile)
            self.warps_m_dkv = largest_warp_partition(self.kv_tile, head_dim_qk, self.d_v)
            self.warps_m_dq = largest_warp_partition(self.q_tile, head_dim_qk)
        M_, N_, d_qk_ = self.q_tile, self.kv_tile, head_dim_qk
        for a_, m_dim, n_dim, tag in (
            (self.warps_m_sdp, M_, N_, "warps_m_sdp"),
            (self.warps_m_dkv, N_, d_qk_, "warps_m_dkv"),
            (self.warps_m_dkv, N_, self.d_v, "warps_m_dkv (dV)"),
            (self.warps_m_dq, M_, d_qk_, "warps_m_dq"),
        ):
            if 8 % a_ or m_dim % (16 * a_) or (n_dim * a_ // 8) % 16:
                raise ValueError(f"invalid {tag}={a_} for M{M_} N{N_} d_qk{d_qk_} d_v{self.d_v}")
        self.page = 64 if head_dim_qk % 64 == 0 and self.d_v % 64 == 0 else 32
        self.threads = 384
        self.num_consumer_warps = 8
        self.load_warp_id = self.num_consumer_warps
        self.tma_swizzle = cuda.TensorMapSwizzle.s128b if self.page == 64 else cuda.TensorMapSwizzle.s64b

        M, N, d_qk, d_v = self.q_tile, self.kv_tile, self.d_qk, self.d_v
        # Double-buffer Q when SMEM allows (prefetch hides the TMA latency);
        # single-buffered Q pays an end-of-iteration rendezvous (d256's only fit).
        cap = cutlass.utils.get_smem_capacity_in_bytes("sm_120")
        for q_stages in (2, 1):
            smem_elems = q_stages * M * d_qk + M * d_v + N * d_qk + max(N * d_v, 2 * M * N)
            if smem_elems * in_dtype.bytes <= cap:
                break
        else:
            raise ValueError(f"smem {smem_elems * in_dtype.bytes} bytes exceeds the sm_120 cap of {cap} bytes even single-buffered")
        # smem element offsets.
        self.q_stages = q_stages
        self.off_sQ = 0  # `q_stages` buffers
        self.off_sdO = q_stages * M * d_qk
        self.off_sK = self.off_sdO + M * d_v
        self.off_sV = self.off_sK + N * d_qk
        self.off_sdS = self.off_sV  # aliases sV (V is in regs)
        self.off_sP = self.off_sV + M * N
        self.smem_elems = smem_elems

    @cute.jit
    def load_tma_tile(self, s_dst, tma_desc, mbar, batch, head, seq, rows: cutlass.Constexpr[int], cols: cutlass.Constexpr[int]):
        """Load one paged/swizzled `(rows, cols)` tile with TMA."""
        elems_per_page = rows * self.page
        for pg in cutlass.range_constexpr(cols // self.page):
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_dst.subview(pg * elems_per_page),
                    tma_desc.get_ptr(),
                    (pg * self.page, head, seq, batch),
                    mbar,
                )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,  # [B, SQ,  HQ, D] io dtype (BSHD)
        k: cute.Tensor,  # [B, SKV, HKV, D]
        v: cute.Tensor,  # [B, SKV, HKV, DV]
        do: cute.Tensor,  # [B, SQ,  HQ, DV]
        lse: cute.Tensor,  # [B, HQ, SQ] fp32 (natural-log LSE)
        delta: cute.Tensor,  # [B, HQ, SQ_r128] fp32 (dot_do_o output)
        dq_accum: cute.Tensor,  # [B*SQ_r128*HQ*D] fp32 (scrambled, zeroed)
        dq_sem: cute.Tensor,  # [B*HQ*num_q_tiles] int32 relay turn counters, one per (batch, head, q-tile); zeroed by dot (deterministic only)
        dk_ws: cute.Tensor,  # [B, SKV, HQ, D] dK destination: dk itself when MHA; per-q-head partials summed by _dkv_reduce_kernel when GQA
        dv_ws: cute.Tensor,  # [B, SKV, HQ, DV] dV destination (same as dK)
        seq_q_lens: Optional[cute.Tensor],  # [B] int32 per-batch Q lengths; None unless seq_q_lens_present
        seq_kv_lens: Optional[cute.Tensor],  # [B] int32 per-batch KV lengths; None unless seq_kv_lens_present
        tma_q_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_do_desc: cutlass.GridConstant[cuda.TensorMap],
        softmax_scale_log2: cutlass.Float32,  # scale * log2(e)
        attn_scale: cutlass.Float32,  # linear scale (dq/dk output)
    ) -> None:
        io_dtype = self.in_dtype
        d_qk = self.d_qk
        d_v = self.d_v
        M = self.q_tile
        N = self.kv_tile
        PAGE = self.page
        Q_STAGES = self.q_stages
        PDS = 64 if self.kv_tile >= 64 else self.kv_tile
        WM_SDP = self.warps_m_sdp  # S/dP warp grid (WM_SDP, 8//WM_SDP), WM_SDP along q rows
        WM_DKV = self.warps_m_dkv  # dK/dV warp grid (WM_DKV, 8//WM_DKV), WM_DKV along kv rows
        WM_DQ = self.warps_m_dq  # dQ warp grid (WM_DQ, 8//WM_DQ), WM_DQ along q rows

        # SdP: warp (wm, wn); wm steps over SDP_REPS 16-row MMA blocks
        # interleaved by 16*WM_SDP; wn covers SDP_NPER contiguous kv columns.
        SDP_REPS = M // (16 * WM_SDP)
        SDP_NPER = N * WM_SDP // 8
        SDP_NF = SDP_NPER // 8
        # dKV: warp (wn2, wd); DKV_REPS 16-row MMA blocks interleaved by 16*WM_DKV.
        DKV_REPS = N // (16 * WM_DKV)
        DK_PER = d_qk * WM_DKV // 8
        DK_NF = DK_PER // 8
        DV_PER = d_v * WM_DKV // 8
        DV_NF = DV_PER // 8
        # dQ: warp (wq, wdq).
        DQ_REPS = M // (16 * WM_DQ)
        DQ_PER = d_qk * WM_DQ // 8
        DQ_NF = DQ_PER // 8

        DQK_CHUNKS = d_qk // 16  # S = Q @ K^T k-reduce (D_QK)
        DV_CHUNKS = d_v // 16  # dP = dO @ V^T k-reduce (D_V)
        Q_CHUNKS = M // 16  # dK/dV k-reduce
        KV_CHUNKS = N // 16  # dQ k-reduce
        VREG_PAIRS = SDP_NPER // 16  # V-in-regs frag pairs / chunk

        tidx, _, _ = cute.arch.thread_idx()
        n_block, q_head, batch = cute.arch.block_idx()
        lane = tidx % 32
        warp = cute.arch.warp_idx()
        g_lane = lane // 4
        p_lane = lane % 4

        SQ = q.shape[1]
        SKV = k.shape[1]
        HQ = q.shape[2]
        HKV = k.shape[2]
        GROUP = HQ // HKV  # query heads per KV head (1 = plain MHA)
        SQ_R = ((SQ + 127) // 128) * 128
        kv_base = n_block * N
        qk_row_stride = HQ * d_qk  # row stride of HQ-headed D_QK-wide BSHD tensors (Q; also dk_ws, whose head axis is HQ)
        v_row_stride = HQ * d_v  # row stride of HQ-headed D_V-wide BSHD tensors (dO; also dv_ws)

        # Per-batch actual lengths (Padding mask)
        seqlen_q = SQ
        seqlen_kv = SKV
        if cutlass.const_expr(self.seq_kv_lens_present):
            seqlen_kv = cute.math.max(cutlass.Int32(0), cute.math.min(seq_kv_lens[batch], cutlass.Int32(SKV)))
        if cutlass.const_expr(self.seq_q_lens_present):
            seqlen_q = cute.math.max(cutlass.Int32(0), cute.math.min(seq_q_lens[batch], cutlass.Int32(SQ)))

        lse_ptr = lse.iterator.raw_ptr()
        lse_batch_stride, lse_head_stride, lse_seq_stride = lse.stride
        lse_strided = (lse_batch_stride, lse_head_stride, lse_seq_stride) != (HQ * SQ, SQ, 1)
        dd_ptr = delta.iterator.raw_ptr()
        dqa_ptr = dq_accum.iterator.raw_ptr()
        dkws_ptr = dk_ws.iterator.raw_ptr()
        dvws_ptr = dv_ws.iterator.raw_ptr()
        if cutlass.const_expr(self.deterministic):
            dqsem_ptr = dq_sem.iterator.raw_ptr()

        PARTIAL_Q = (SQ % M) != 0
        PARTIAL_KV = (SKV % N) != 0
        # Skip the kv >= SKV check when padding already masks kv >= seqlen_kv (<= SKV).
        MASK_KV_GLOBAL = PARTIAL_KV and not self.seq_kv_lens_present
        # Fully-masked rows have LSE = -inf; flip to +inf so P = exp2(finite - inf) = 0, not NaN.
        FLIP_MASKED_LSE = (
            (self.is_causal and not self.causal_top_left) or self.window_size_left is not None or self.seq_kv_lens_present or self.seq_q_lens_present
        )

        m_block_max = (seqlen_q + M - 1) // M
        if cutlass.const_expr(self.is_causal or self.window_size_left is not None):
            if cutlass.const_expr(self.is_causal and not self.causal_top_left):
                # The diagonal anchors at the actual lengths.
                diag_off = seqlen_kv - seqlen_q
            else:
                diag_off = cutlass.Int32(0)
        if cutlass.const_expr(self.is_causal):
            # kv is visible to q when kv <= q + diag_off + right_slack
            m_block_min = cute.math.max(kv_base - diag_off - self.right_slack, cutlass.Int32(0)) // M
        else:
            m_block_min = cutlass.Int32(0)
        if cutlass.const_expr(self.window_size_left is not None):
            # q <= k - diag_off + W
            last_q_row = kv_base + N - 1 - diag_off + self.window_size_left
            rows_hi = cute.math.max(last_q_row + 1, cutlass.Int32(0))
            m_block_max = cute.math.min(m_block_max, (rows_hi + M - 1) // M)
        n_iters = cute.math.max(m_block_max - m_block_min, cutlass.Int32(0))
        if cutlass.const_expr(self.seq_kv_lens_present):
            # Fully-padded KV tile: skip the loop; the epilogue writes zero dK/dV.
            if kv_base >= seqlen_kv:
                n_iters = cutlass.Int32(0)

        smem = cutlass.Array(io_dtype, self.smem_elems, space=cutlass.AddressSpace.smem, alignment=128)
        sQ = smem  # Q_STAGES * M * d_qk
        sdO = smem.subview(self.off_sdO)  # M * d_v
        sK = smem.subview(self.off_sK)  # N * d_qk
        sV = smem.subview(self.off_sV)  # N * d_v (region max(N * d_v, 2 * M * N))
        sdS = smem.subview(self.off_sdS)  # M * N (aliases sV)
        sP = smem.subview(self.off_sP)  # M * N
        tma_mbar = cutlass.Array(cutlass.Int64, 5, space=cutlass.AddressSpace.smem, alignment=8)
        k_mbar = tma_mbar
        v_mbar = tma_mbar.subview(1)
        q_full = tma_mbar.subview(2)
        do_full = tma_mbar.subview(4)

        if warp == self.load_warp_id:
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_k_desc.get_ptr())
                prims.prefetch_tensormap(tma_v_desc.get_ptr())
                prims.prefetch_tensormap(tma_q_desc.get_ptr())
                prims.prefetch_tensormap(tma_do_desc.get_ptr())
                prims.mbarrier_init(k_mbar, 1)
                prims.mbarrier_init(v_mbar, 1)
                prims.mbarrier_init(q_full, 1)
                prims.mbarrier_init(q_full.subview(1), 1)
                prims.mbarrier_init(do_full, 1)
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)

        kv_head = q_head // GROUP
        if cutlass.const_expr(lse_strided):
            lse_base = batch * lse_batch_stride + q_head * lse_head_stride
        else:
            lse_base = (batch * HQ + q_head) * SQ
        dd_base = (batch * HQ + q_head) * SQ_R
        if cutlass.const_expr(self.deterministic):
            det_sem = dqsem_ptr + (batch * HQ + q_head) * ((SQ + M - 1) // M)

        if warp == self.load_warp_id:
            prims.setmaxregister(24, prims.SetMaxRegisterAction.DECREASE)
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(v_mbar, N * d_v * io_dtype.bytes)
                prims.mbarrier_arrive_expect_tx(k_mbar, N * d_qk * io_dtype.bytes)
            self.load_tma_tile(sV, tma_v_desc, v_mbar, batch, kv_head, kv_base, rows=N, cols=d_v)
            self.load_tma_tile(sK, tma_k_desc, k_mbar, batch, kv_head, kv_base, rows=N, cols=d_qk)

            if n_iters > 0:
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(q_full, M * d_qk * io_dtype.bytes)
                self.load_tma_tile(sQ, tma_q_desc, q_full, batch, q_head, (m_block_max - 1) * M, rows=M, cols=d_qk)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(do_full, M * d_v * io_dtype.bytes)
                self.load_tma_tile(
                    sdO,
                    tma_do_desc,
                    do_full,
                    batch,
                    q_head,
                    (m_block_max - 1) * M,
                    rows=M,
                    cols=d_v,
                )
            while not prims.mbarrier_try_wait_parity(v_mbar, cutlass.Int32(0)):
                pass
            while not prims.mbarrier_try_wait_parity(k_mbar, cutlass.Int32(0)):
                pass
            for load_j in cutlass.range(n_iters, unroll=1):
                if cutlass.const_expr(Q_STAGES == 2):
                    load_stage = load_j & cutlass.Int32(1)
                    q_phase_p = (load_j // 2) & cutlass.Int32(1)
                else:
                    load_stage = cutlass.Int32(0)
                    q_phase_p = load_j & cutlass.Int32(1)
                while not prims.mbarrier_try_wait_parity(q_full.subview(load_stage), q_phase_p):
                    pass
                do_phase_p = load_j & cutlass.Int32(1)
                while not prims.mbarrier_try_wait_parity(do_full, do_phase_p):
                    pass
                # Loop-top: stage load_j ready AND load_j-1 consumed.
                cute.arch.barrier(barrier_id=3, number_of_threads=288)
                next_m = m_block_max - 2 - load_j
                if cutlass.const_expr(Q_STAGES == 2):
                    # when double-buffer for q, prefetch for the next iteration
                    if load_j + 1 < n_iters:
                        next_stage = (load_j + 1) & cutlass.Int32(1)
                        next_q_full = q_full.subview(next_stage)
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(next_q_full, M * d_qk * io_dtype.bytes)
                        self.load_tma_tile(
                            sQ.subview(next_stage * M * d_qk),
                            tma_q_desc,
                            next_q_full,
                            batch,
                            q_head,
                            next_m * M,
                            rows=M,
                            cols=d_qk,
                        )
                # Post-GEMM3 (dV += P^T*dO): every consumer is done with sdO.
                cute.arch.barrier(barrier_id=4, number_of_threads=288)
                if load_j + 1 < n_iters:
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(do_full, M * d_v * io_dtype.bytes)
                    self.load_tma_tile(sdO, tma_do_desc, do_full, batch, q_head, next_m * M, rows=M, cols=d_v)
                if cutlass.const_expr(Q_STAGES == 1):
                    # Post-GEMM5 (dK += dS^T*Q): every consumer is done with sQ.
                    cute.arch.barrier(barrier_id=5, number_of_threads=288)
                    if load_j + 1 < n_iters:
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(q_full, M * d_qk * io_dtype.bytes)
                        self.load_tma_tile(sQ, tma_q_desc, q_full, batch, q_head, next_m * M, rows=M, cols=d_qk)

        elif warp < self.load_warp_id:
            prims.setmaxregister(240, prims.SetMaxRegisterAction.INCREASE)
            # LSE for the first (highest) m-block: per-thread direct loads at
            # this thread's C-fragment rows
            math_warp = warp
            math_tidx = tidx
            m_block = m_block_max - 1
            if cutlass.const_expr(self.window_size_left is not None or self.seq_q_lens_present):
                # m_block_max can be 0 in bottom-right SWA, or seq_len_q[b] == 0
                m_block = cute.math.max(m_block, cutlass.Int32(0))
            wm_s = math_warp % WM_SDP
            wn_s = math_warp // WM_SDP
            lse_r = cutlass.Array(cutlass.Float32, 2 * SDP_REPS)
            dd_r = cutlass.Array(cutlass.Float32, 2 * SDP_REPS)
            for rep in cutlass.range_constexpr(SDP_REPS):
                for hf in cutlass.range_constexpr(2):
                    r_loc = wm_s * 16 + rep * 16 * WM_SDP + g_lane + hf * 8
                    r_abs = m_block * M + r_loc
                    if cutlass.const_expr(PARTIAL_Q):
                        r_cl = cute.math.min(r_abs, SQ - 1)
                        if cutlass.const_expr(lse_strided):
                            val = (lse_ptr + lse_base + r_cl * lse_seq_stride).load()
                        else:
                            val = (lse_ptr + lse_base + r_cl).load()
                        inf = cutlass.Float32(float("inf"))
                        # branchless (r_abs < SQ) ? 1 : 0 via arith.select
                        ok32 = cutlass.Int32(
                            arith.select(
                                (r_abs < SQ).ir_value(),
                                cutlass.Int32(1).ir_value(),
                                cutlass.Int32(0).ir_value(),
                            )
                        )
                        if ok32 == 0:
                            val = inf
                    elif cutlass.const_expr(lse_strided):
                        val = (lse_ptr + lse_base + r_abs * lse_seq_stride).load()
                    else:
                        val = (lse_ptr + lse_base + r_abs).load()
                    if cutlass.const_expr(FLIP_MASKED_LSE):
                        # A fully masked row has LSE = -inf, which can produce
                        # -inf - (-inf) = NaN.  Use +inf to reconstruct P = 0.
                        if val == cutlass.Float32(float("-inf")):
                            val = cutlass.Float32(float("inf"))
                    if cutlass.const_expr(self.trim_q_rows):
                        # explicit padded-row trim (sink LSE is finite there)
                        if r_abs >= seqlen_q:
                            val = cutlass.Float32(float("inf"))
                    lse_r[rep * 2 + hf] = val * cutlass.Float32(_LOG2E)

            while not prims.mbarrier_try_wait_parity(v_mbar, cutlass.Int32(0)):
                pass
            while not prims.mbarrier_try_wait_parity(k_mbar, cutlass.Int32(0)):
                pass

            # V -> registers.
            v_persist = cutlass.Array(cutlass.Int32, DV_CHUNKS * VREG_PAIRS * 4, alignment=16)
            for kc in cutlass.range_constexpr(DV_CHUNKS):
                for pair in cutlass.range_constexpr(VREG_PAIRS):
                    n_frag = pair * 2
                    row = wn_s * SDP_NPER + (n_frag + lane // 16) * 8 + lane % 8
                    col = kc * 16 + ((lane // 8) % 2) * 8
                    vf = prims.ldmatrix(
                        tile_ptr(sV, row, col, page=PAGE, rows=N),
                        4,
                        prims.MMALayout.ROW,
                    )
                    v_off = (kc * VREG_PAIRS + pair) * 4
                    v_persist[v_off + 0] = vf[0]
                    v_persist[v_off + 1] = vf[1]
                    v_persist[v_off + 2] = vf[2]
                    v_persist[v_off + 3] = vf[3]
            cute.arch.barrier(barrier_id=1, number_of_threads=256)

            # dK/dV accumulators.
            wn_k = math_warp % WM_DKV
            wd_k = math_warp // WM_DKV
            acc_dk = cutlass.Array(cutlass.Float32, DKV_REPS * DK_NF * 4, alignment=16)
            acc_dv = cutlass.Array(cutlass.Float32, DKV_REPS * DV_NF * 4, alignment=16)
            for i in cutlass.range_constexpr(DKV_REPS * DK_NF * 4):
                acc_dk[i] = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(DKV_REPS * DV_NF * 4):
                acc_dv[i] = cutlass.Float32(0.0)

            wq = math_warp % WM_DQ
            wd_q = math_warp // WM_DQ

            acc_s = cutlass.Array(cutlass.Float32, SDP_REPS * SDP_NF * 4, alignment=16)
            acc_dp = cutlass.Array(cutlass.Float32, SDP_REPS * SDP_NF * 4, alignment=16)
            acc_dq = cutlass.Array(cutlass.Float32, DQ_REPS * DQ_NF * 4, alignment=16)

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            # ---- main loop: m_block descending --------------------------------
            j = cutlass.Int32(0)
            while j < n_iters:
                m_block = m_block_max - 1 - j
                if cutlass.const_expr(Q_STAGES == 2):
                    stage = j & cutlass.Int32(1)
                else:
                    stage = cutlass.Int32(0)
                sQ_st = sQ.subview(stage * M * d_qk)
                q_row0 = m_block * M

                cute.arch.barrier(barrier_id=3, number_of_threads=288)

                # dP_sum per-thread loads (delta buffer is 128-rounded).
                for rep in cutlass.range_constexpr(SDP_REPS):
                    for hf in cutlass.range_constexpr(2):
                        r_loc = wm_s * 16 + rep * 16 * WM_SDP + g_lane + hf * 8
                        dd_r[rep * 2 + hf] = (dd_ptr + dd_base + q_row0 + r_loc).load()

                # GEMM 1: acc_s = Q @ K^T.
                for i in cutlass.range_constexpr(SDP_REPS * SDP_NF * 4):
                    acc_s[i] = cutlass.Float32(0.0)
                for kc in cutlass.range_constexpr(DQK_CHUNKS):
                    af = []
                    for rep in cutlass.range_constexpr(SDP_REPS):
                        qf = load_a_frag(
                            sQ_st,
                            kc,
                            wm_s * 16 + rep * 16 * WM_SDP,
                            lane,
                            rows=M,
                            page=PAGE,
                        )
                        af = af + [qf[0], qf[1], qf[2], qf[3]]
                    mma_bstream(
                        acc_s,
                        af,
                        sK,
                        b_k_step=kc,
                        M=16 * SDP_REPS,
                        N=SDP_NPER,
                        b_trans=False,
                        b_rows=N,
                        b_page=PAGE,
                        lane=lane,
                        ab_dtype=io_dtype,
                        row_base=wn_s * SDP_NPER,
                    )

                # Mask + softmax (scores -> P, unscaled by attn_scale) and the
                # P store to smem.
                if cutlass.const_expr(self.is_causal):
                    do_mask_causal = (m_block * M) < (kv_base + N - diag_off - self.right_slack)
                if cutlass.const_expr(self.window_size_left is not None):
                    do_mask_window = kv_base < (m_block * M + M - 1 + diag_off - self.window_size_left)
                if cutlass.const_expr(self.seq_kv_lens_present):
                    do_mask_pad = (kv_base + N) > seqlen_kv
                neg_inf = cutlass.Float32(float("-inf"))
                for rep in cutlass.range_constexpr(SDP_REPS):
                    for nf in cutlass.range_constexpr(SDP_NF):
                        off = (rep * SDP_NF + nf) * 4
                        kv_c0 = wn_s * SDP_NPER + nf * 8 + 2 * p_lane
                        kv_a0 = kv_base + kv_c0
                        kv_a1 = kv_a0 + 1
                        r0 = q_row0 + wm_s * 16 + rep * 16 * WM_SDP + g_lane
                        r8 = r0 + 8
                        s0 = acc_s[off + 0]
                        s1 = acc_s[off + 1]
                        s2 = acc_s[off + 2]
                        s3 = acc_s[off + 3]
                        if cutlass.const_expr(self.is_causal):
                            if do_mask_causal:
                                hi0 = r0 + diag_off + self.right_slack
                                hi8 = r8 + diag_off + self.right_slack
                                if kv_a0 > hi0:
                                    s0 = neg_inf
                                if kv_a1 > hi0:
                                    s1 = neg_inf
                                if kv_a0 > hi8:
                                    s2 = neg_inf
                                if kv_a1 > hi8:
                                    s3 = neg_inf
                        if cutlass.const_expr(self.window_size_left is not None):
                            if do_mask_window:
                                lo0 = r0 + diag_off - self.window_size_left
                                lo8 = r8 + diag_off - self.window_size_left
                                if kv_a0 < lo0:
                                    s0 = neg_inf
                                if kv_a1 < lo0:
                                    s1 = neg_inf
                                if kv_a0 < lo8:
                                    s2 = neg_inf
                                if kv_a1 < lo8:
                                    s3 = neg_inf
                        if cutlass.const_expr(self.seq_kv_lens_present):
                            if do_mask_pad:
                                if kv_a0 >= seqlen_kv:
                                    s0 = neg_inf
                                    s2 = neg_inf
                                if kv_a1 >= seqlen_kv:
                                    s1 = neg_inf
                                    s3 = neg_inf
                        if cutlass.const_expr(MASK_KV_GLOBAL):
                            if kv_a0 >= SKV:
                                s0 = neg_inf
                                s2 = neg_inf
                            if kv_a1 >= SKV:
                                s1 = neg_inf
                                s3 = neg_inf
                        lse0 = lse_r[rep * 2 + 0]
                        lse8 = lse_r[rep * 2 + 1]
                        p0 = cute.math.exp2(s0 * softmax_scale_log2 - lse0, fastmath=True)
                        p1 = cute.math.exp2(s1 * softmax_scale_log2 - lse0, fastmath=True)
                        p2 = cute.math.exp2(s2 * softmax_scale_log2 - lse8, fastmath=True)
                        p3 = cute.math.exp2(s3 * softmax_scale_log2 - lse8, fastmath=True)
                        acc_s[off + 0] = p0
                        acc_s[off + 1] = p1
                        acc_s[off + 2] = p2
                        acc_s[off + 3] = p3
                        # sP store: each (2p, 2p+1) pair packed into one 4 B
                        # store to the swizzled tile
                        pr0 = wm_s * 16 + rep * 16 * WM_SDP + g_lane
                        pr8 = pr0 + 8
                        sw0 = tile_ptr(sP, pr0, kv_c0, page=PDS, rows=M)
                        sw8 = tile_ptr(sP, pr8, kv_c0, page=PDS, rows=M)
                        sw0.store(pack_half2(p0, p1, io_dtype), alignment=4)
                        sw8.store(pack_half2(p2, p3, io_dtype), alignment=4)

                # GEMM 2: acc_dp = dO @ V^T (V in registers).
                for i in cutlass.range_constexpr(SDP_REPS * SDP_NF * 4):
                    acc_dp[i] = cutlass.Float32(0.0)
                for kc in cutlass.range_constexpr(DV_CHUNKS):
                    af = []
                    for rep in cutlass.range_constexpr(SDP_REPS):
                        dof = load_a_frag(
                            sdO,
                            kc,
                            wm_s * 16 + rep * 16 * WM_SDP,
                            lane,
                            rows=M,
                            page=PAGE,
                        )
                        af = af + [dof[0], dof[1], dof[2], dof[3]]
                    mma_abregs(
                        acc_dp,
                        af,
                        v_persist,
                        b_k_step=kc,
                        M=16 * SDP_REPS,
                        N=SDP_NPER,
                        ab_dtype=io_dtype,
                    )

                # dS = P * (dP - dP_sum)
                for rep in cutlass.range_constexpr(SDP_REPS):
                    for nf in cutlass.range_constexpr(SDP_NF):
                        off = (rep * SDP_NF + nf) * 4
                        dd0 = dd_r[rep * 2 + 0]
                        dd8 = dd_r[rep * 2 + 1]
                        acc_dp[off + 0] = acc_s[off + 0] * (acc_dp[off + 0] - dd0)
                        acc_dp[off + 1] = acc_s[off + 1] * (acc_dp[off + 1] - dd0)
                        acc_dp[off + 2] = acc_s[off + 2] * (acc_dp[off + 2] - dd8)
                        acc_dp[off + 3] = acc_s[off + 3] * (acc_dp[off + 3] - dd8)

                # dS -> fp16 -> sdS.
                for rep in cutlass.range_constexpr(SDP_REPS):
                    for nf in cutlass.range_constexpr(SDP_NF):
                        off = (rep * SDP_NF + nf) * 4
                        kv_c0 = wn_s * SDP_NPER + nf * 8 + 2 * p_lane
                        pr0 = wm_s * 16 + rep * 16 * WM_SDP + g_lane
                        pr8 = pr0 + 8
                        sw0 = tile_ptr(sdS, pr0, kv_c0, page=PDS, rows=M)
                        sw8 = tile_ptr(sdS, pr8, kv_c0, page=PDS, rows=M)
                        sw0.store(
                            pack_half2(acc_dp[off + 0], acc_dp[off + 1], io_dtype),
                            alignment=4,
                        )
                        sw8.store(
                            pack_half2(acc_dp[off + 2], acc_dp[off + 3], io_dtype),
                            alignment=4,
                        )
                cute.arch.barrier(barrier_id=1, number_of_threads=256)

                # GEMM 3: acc_dv += P^T @ dO.
                for kc in cutlass.range_constexpr(Q_CHUNKS):
                    af = []
                    for rep in cutlass.range_constexpr(DKV_REPS):
                        pf = load_a_frag_transposed(
                            sP,
                            kc,
                            wn_k * 16 + rep * 16 * WM_DKV,
                            lane,
                            rows=M,
                            page=PDS,
                        )
                        af = af + [pf[0], pf[2], pf[1], pf[3]]
                    mma_bstream(
                        acc_dv,
                        af,
                        sdO,
                        b_k_step=kc,
                        M=16 * DKV_REPS,
                        N=DV_PER,
                        b_trans=True,
                        b_rows=M,
                        b_page=PAGE,
                        lane=lane,
                        ab_dtype=io_dtype,
                        col_base=wd_k * DV_PER,
                    )

                # GEMM3 is the final dO consumer; this rendezvous lets
                # the producer refill the single dO buffer.
                cute.arch.barrier(barrier_id=4, number_of_threads=288)

                # Deterministic relay turn for this q-tile: the dQ adds of a
                # (batch, head, q-tile) happen in ascending kv-tile order.
                if cutlass.const_expr(self.deterministic):
                    if cutlass.const_expr(self.window_size_left is not None):
                        # SWA clamps m_block_max, so a q-tile's visitors start
                        # at kv tile n_lo = max((m_block*M + diag_off - W) // N, 0)
                        # (inverts the clamp); count turns from there.
                        det_turn = n_block - cute.math.max(
                            (m_block * M + diag_off - self.window_size_left) // N,
                            cutlass.Int32(0),
                        )
                    else:
                        det_turn = n_block

                if cutlass.const_expr(Q_STAGES == 1):
                    # Single Q buffer: GEMM5 (sQ's last reader) first, so the
                    # Q refill hides behind GEMM4 + the dQ scatter. (2-stage
                    # keeps GEMM4-first: atomics drain during GEMM5 instead.)
                    _bwd_gemm5_dk(
                        acc_dk,
                        sdS,
                        sQ_st,
                        wn_k,
                        wd_k,
                        lane,
                        Q_CHUNKS=Q_CHUNKS,
                        DKV_REPS=DKV_REPS,
                        WM_DKV=WM_DKV,
                        M=M,
                        PDS=PDS,
                        PAGE=PAGE,
                        DK_PER=DK_PER,
                        io_dtype=io_dtype,
                    )
                    cute.arch.barrier(barrier_id=5, number_of_threads=288)
                    _bwd_gemm4_dq(
                        acc_dq,
                        sdS,
                        sK,
                        wq,
                        wd_q,
                        lane,
                        DQ_REPS=DQ_REPS,
                        DQ_NF=DQ_NF,
                        KV_CHUNKS=KV_CHUNKS,
                        WM_DQ=WM_DQ,
                        M=M,
                        N=N,
                        PDS=PDS,
                        PAGE=PAGE,
                        DQ_PER=DQ_PER,
                        io_dtype=io_dtype,
                    )
                    # Reload LSE for the next (lower) m-block: overlaps the
                    # in-flight Q refill issued at barrier 5.
                    if j + 1 < n_iters:
                        nq0 = (m_block - 1) * M
                        for rep in cutlass.range_constexpr(SDP_REPS):
                            for hf in cutlass.range_constexpr(2):
                                r_loc = wm_s * 16 + rep * 16 * WM_SDP + g_lane + hf * 8
                                if cutlass.const_expr(lse_strided):
                                    val = (lse_ptr + lse_base + (nq0 + r_loc) * lse_seq_stride).load()
                                else:
                                    val = (lse_ptr + lse_base + nq0 + r_loc).load()
                                if cutlass.const_expr(FLIP_MASKED_LSE):
                                    # A fully masked row has LSE = -inf, which can produce
                                    # -inf - (-inf) = NaN.  Use +inf to reconstruct P = 0.
                                    if val == cutlass.Float32(float("-inf")):
                                        val = cutlass.Float32(float("inf"))
                                if cutlass.const_expr(self.trim_q_rows):
                                    # explicit padded-row trim (sink LSE is finite there)
                                    if nq0 + r_loc >= seqlen_q:
                                        val = cutlass.Float32(float("inf"))
                                lse_r[rep * 2 + hf] = val * cutlass.Float32(_LOG2E)
                    dqa_base = ((batch * SQ_R + q_row0) * HQ + q_head) * d_qk
                    if cutlass.const_expr(self.deterministic):
                        _bwd_det_wait(det_sem, m_block, det_turn, warp)
                    _bwd_dq_scatter(acc_dq, dqa_ptr, dqa_base, math_tidx, HQ, DQ_REPS=DQ_REPS, DQ_NF=DQ_NF, M=M, d_qk=d_qk)
                    if cutlass.const_expr(self.deterministic):
                        _bwd_det_release(det_sem, m_block, det_turn, warp)
                else:
                    _bwd_gemm4_dq(
                        acc_dq,
                        sdS,
                        sK,
                        wq,
                        wd_q,
                        lane,
                        DQ_REPS=DQ_REPS,
                        DQ_NF=DQ_NF,
                        KV_CHUNKS=KV_CHUNKS,
                        WM_DQ=WM_DQ,
                        M=M,
                        N=N,
                        PDS=PDS,
                        PAGE=PAGE,
                        DQ_PER=DQ_PER,
                        io_dtype=io_dtype,
                    )
                    # Reload LSE for the next (lower) m-block (develop-exact
                    # position: between GEMM4 and the scatter, hiding the
                    # global-load latency behind the atomic drain + GEMM5).
                    if j + 1 < n_iters:
                        nq0 = (m_block - 1) * M
                        for rep in cutlass.range_constexpr(SDP_REPS):
                            for hf in cutlass.range_constexpr(2):
                                r_loc = wm_s * 16 + rep * 16 * WM_SDP + g_lane + hf * 8
                                if cutlass.const_expr(lse_strided):
                                    val = (lse_ptr + lse_base + (nq0 + r_loc) * lse_seq_stride).load()
                                else:
                                    val = (lse_ptr + lse_base + nq0 + r_loc).load()
                                if cutlass.const_expr(FLIP_MASKED_LSE):
                                    # A fully masked row has LSE = -inf, which can produce
                                    # -inf - (-inf) = NaN.  Use +inf to reconstruct P = 0.
                                    if val == cutlass.Float32(float("-inf")):
                                        val = cutlass.Float32(float("inf"))
                                if cutlass.const_expr(self.trim_q_rows):
                                    # explicit padded-row trim (sink LSE is finite there)
                                    if nq0 + r_loc >= seqlen_q:
                                        val = cutlass.Float32(float("inf"))
                                lse_r[rep * 2 + hf] = val * cutlass.Float32(_LOG2E)
                    dqa_base = ((batch * SQ_R + q_row0) * HQ + q_head) * d_qk
                    if cutlass.const_expr(self.deterministic):
                        _bwd_det_wait(det_sem, m_block, det_turn, warp)
                    _bwd_dq_scatter(acc_dq, dqa_ptr, dqa_base, math_tidx, HQ, DQ_REPS=DQ_REPS, DQ_NF=DQ_NF, M=M, d_qk=d_qk)
                    if cutlass.const_expr(self.deterministic):
                        _bwd_det_release(det_sem, m_block, det_turn, warp)
                    _bwd_gemm5_dk(
                        acc_dk,
                        sdS,
                        sQ_st,
                        wn_k,
                        wd_k,
                        lane,
                        Q_CHUNKS=Q_CHUNKS,
                        DKV_REPS=DKV_REPS,
                        WM_DKV=WM_DKV,
                        M=M,
                        PDS=PDS,
                        PAGE=PAGE,
                        DK_PER=DK_PER,
                        io_dtype=io_dtype,
                    )

                j += 1

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            # epilogue: dK/dV through smem (sdK aliases sK, sdV aliases sV).
            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            sdK = sK
            sdV = sV
            for rep in cutlass.range_constexpr(DKV_REPS):
                for nf in cutlass.range_constexpr(max(DK_NF, DV_NF)):
                    r0 = wn_k * 16 + rep * 16 * WM_DKV + g_lane
                    r8 = r0 + 8
                    if cutlass.const_expr(nf < DK_NF):
                        off = (rep * DK_NF + nf) * 4
                        c0 = wd_k * DK_PER + nf * 8 + 2 * p_lane
                        dk0 = acc_dk[off + 0] * attn_scale
                        dk1 = acc_dk[off + 1] * attn_scale
                        dk2 = acc_dk[off + 2] * attn_scale
                        dk3 = acc_dk[off + 3] * attn_scale
                        tile_ptr(sdK, r0, c0, page=PAGE, rows=N).store(pack_half2(dk0, dk1, io_dtype), alignment=4)
                        tile_ptr(sdK, r8, c0, page=PAGE, rows=N).store(pack_half2(dk2, dk3, io_dtype), alignment=4)
                    if cutlass.const_expr(nf < DV_NF):
                        off_v = (rep * DV_NF + nf) * 4
                        c0_v = wd_k * DV_PER + nf * 8 + 2 * p_lane
                        tile_ptr(sdV, r0, c0_v, page=PAGE, rows=N).store(
                            pack_half2(acc_dv[off_v + 0], acc_dv[off_v + 1], io_dtype),
                            alignment=4,
                        )
                        tile_ptr(sdV, r8, c0_v, page=PAGE, rows=N).store(
                            pack_half2(acc_dv[off_v + 2], acc_dv[off_v + 3], io_dtype),
                            alignment=4,
                        )
            cute.arch.barrier(barrier_id=2, number_of_threads=256)

            # smem -> gmem. dk_ws/dv_ws rows are HQ-headed: dk/dv themselves
            # when MHA (HQ == HKV and q_head == kv_head), one slot per q head
            # under GQA — the same addressing covers both.
            dk_batch_stride, dk_seq_stride, dk_head_stride, _ = dk_ws.stride
            dv_batch_stride, dv_seq_stride, dv_head_stride, _ = dv_ws.stride
            dkv_strided = (dk_batch_stride, dk_seq_stride, dk_head_stride) != (SKV * qk_row_stride, qk_row_stride, d_qk) or (
                dv_batch_stride,
                dv_seq_stride,
                dv_head_stride,
            ) != (
                SKV * v_row_stride,
                v_row_stride,
                d_v,
            )
            if cutlass.const_expr(d_qk == d_v and not dkv_strided and dk_ws.shape[3] == d_qk and dv_ws.shape[3] == d_v):
                chunks_per_row = d_qk // _COPY_ELEMS
                total = N * chunks_per_row
                # workspace's head-dim base offset
                whd_base = (batch * SKV + kv_base) * qk_row_stride + q_head * d_qk
                for i in cutlass.range_constexpr(total // 256):
                    chunk = i * 256 + math_tidx
                    row = chunk // chunks_per_row
                    col = (chunk % chunks_per_row) * _COPY_ELEMS
                    if (not cutlass.const_expr(PARTIAL_KV)) or (kv_base + row < SKV):
                        w_off = whd_base + row * qk_row_stride + col
                        copy16_smem_to_gmem(tile_ptr(sdK, row, col, page=PAGE, rows=N), dkws_ptr + w_off)
                        copy16_smem_to_gmem(tile_ptr(sdV, row, col, page=PAGE, rows=N), dvws_ptr + w_off)
            else:
                # d_qk != d_v, strided, or enveloped
                k_chunks_per_row = d_qk // _COPY_ELEMS
                base_k = batch * dk_batch_stride + kv_base * dk_seq_stride + q_head * dk_head_stride
                for i in cutlass.range_constexpr(N * k_chunks_per_row // 256):
                    chunk = i * 256 + math_tidx
                    row = chunk // k_chunks_per_row
                    col = (chunk % k_chunks_per_row) * _COPY_ELEMS
                    if (not cutlass.const_expr(PARTIAL_KV)) or (kv_base + row < SKV):
                        if cutlass.const_expr(self.qk_envelope):
                            # dK is only d_qk_orig wide; the pad columns are zero anyway.
                            if col < self.d_qk_orig:
                                copy16_smem_to_gmem(tile_ptr(sdK, row, col, page=PAGE, rows=N), dkws_ptr + base_k + row * dk_seq_stride + col)
                        else:
                            copy16_smem_to_gmem(tile_ptr(sdK, row, col, page=PAGE, rows=N), dkws_ptr + base_k + row * dk_seq_stride + col)
                v_chunks_per_row = d_v // _COPY_ELEMS
                base_v = batch * dv_batch_stride + kv_base * dv_seq_stride + q_head * dv_head_stride
                for i in cutlass.range_constexpr(N * v_chunks_per_row // 256):
                    chunk = i * 256 + math_tidx
                    row = chunk // v_chunks_per_row
                    col = (chunk % v_chunks_per_row) * _COPY_ELEMS
                    if (not cutlass.const_expr(PARTIAL_KV)) or (kv_base + row < SKV):
                        if cutlass.const_expr(self.v_envelope):
                            if col < self.d_v_orig:
                                copy16_smem_to_gmem(tile_ptr(sdV, row, col, page=PAGE, rows=N), dvws_ptr + base_v + row * dv_seq_stride + col)
                        else:
                            copy16_smem_to_gmem(tile_ptr(sdV, row, col, page=PAGE, rows=N), dvws_ptr + base_v + row * dv_seq_stride + col)
        else:
            prims.setmaxregister(24, prims.SetMaxRegisterAction.DECREASE)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        do: cute.Tensor,
        lse: cute.Tensor,
        delta: cute.Tensor,
        dq_accum: cute.Tensor,
        dq_sem: cute.Tensor,
        dk_ws: cute.Tensor,
        dv_ws: cute.Tensor,
        seq_q_lens: Optional[cute.Tensor],
        seq_kv_lens: Optional[cute.Tensor],
        softmax_scale_log2: cutlass.Float32,
        attn_scale: cutlass.Float32,
        stream: cuda_driver.CUstream,
    ) -> None:
        box_kv = (1, self.kv_tile, 1, self.page)
        box_q = (1, self.q_tile, 1, self.page)
        tma_q_desc = cuda.create_tensor_map_tiled_from_view(q, box_dims=box_q, stride_order=(3, 2, 1, 0), swizzle=self.tma_swizzle)
        tma_k_desc = cuda.create_tensor_map_tiled_from_view(k, box_dims=box_kv, stride_order=(3, 2, 1, 0), swizzle=self.tma_swizzle)
        tma_v_desc = cuda.create_tensor_map_tiled_from_view(v, box_dims=box_kv, stride_order=(3, 2, 1, 0), swizzle=self.tma_swizzle)
        tma_do_desc = cuda.create_tensor_map_tiled_from_view(do, box_dims=box_q, stride_order=(3, 2, 1, 0), swizzle=self.tma_swizzle)
        n_blocks = cute.ceil_div(k.shape[1], self.kv_tile)
        self.kernel(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq_accum,
            dq_sem,
            dk_ws,
            dv_ws,
            seq_q_lens,
            seq_kv_lens,
            tma_q_desc,
            tma_k_desc,
            tma_v_desc,
            tma_do_desc,
            softmax_scale_log2,
            attn_scale,
        ).launch(
            grid=(n_blocks, q.shape[2], q.shape[0]),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )


# ---------------------------------------------------------------------------
# Preprocess kernel: delta = rowsum(dO * O) + dq_accum / dq_sem zeroing
# ---------------------------------------------------------------------------


@cute.kernel
def _dot_do_o_kernel(
    o: cute.Tensor,  # [B, SQ, H, DV]
    do: cute.Tensor,  # [B, SQ, H, DV]
    delta: cute.Tensor,  # [B, H, SQ_r128] fp32 out
    dq_accum: cute.Tensor,  # [B*SQ_r128*H*D] fp32 (zeroed here)
    dq_sem: cute.Tensor,  # [B*H*num_q_tiles] int32 relay turn counters (zeroed here when deterministic)
    q_tile: cutlass.Constexpr[int],
    d_qk: cutlass.Constexpr[int],  # D_QK: dq_accum's head dim
    d_v: cutlass.Constexpr[int],  # D_V: O/dO's head dim
    page: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    deterministic: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_launch_dependents()
    m_block, head, batch = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    SQ = o.shape[1]
    H = o.shape[2]
    SQ_R = ((SQ + 127) // 128) * 128
    M = q_tile

    o_ptr = o.iterator.raw_ptr()
    do_ptr = do.iterator.raw_ptr()
    dd_ptr = delta.iterator.raw_ptr()
    dqa_ptr = dq_accum.iterator.raw_ptr()

    o_batch_stride, o_seq_stride, o_head_stride, _ = o.stride
    do_batch_stride, do_seq_stride, do_head_stride, _ = do.stride
    compact = (SQ * H * d_v, H * d_v, d_v)
    io_strided = o.shape[3] != d_v or (o_batch_stride, o_seq_stride, o_head_stride) != compact or (do_batch_stride, do_seq_stride, do_head_stride) != compact
    if cutlass.const_expr(io_strided):
        o_base = batch * o_batch_stride + (m_block * M) * o_seq_stride + head * o_head_stride
        do_base = batch * do_batch_stride + (m_block * M) * do_seq_stride + head * do_head_stride
    else:
        row_stride = H * d_v
        base = ((batch * SQ + m_block * M) * H + head) * d_v
    dd_base = (batch * H + head) * SQ_R + m_block * M
    q_left = SQ - m_block * M

    threads_per_row = page // _COPY_ELEMS
    rows_per_pass = 256 // threads_per_row
    col0 = (tidx % threads_per_row) * _COPY_ELEMS
    row0 = tidx // threads_per_row
    n_pages = d_v // page
    for rp in cutlass.range_constexpr(M // rows_per_pass):
        row = row0 + rp * rows_per_pass
        acc = cutlass.Float32(0.0)
        if row < q_left:
            if cutlass.const_expr(io_strided):
                o_off = o_base + row * o_seq_stride + col0
                do_off = do_base + row * do_seq_stride + col0
                for pg in cutlass.range_constexpr(n_pages):
                    if cutlass.const_expr(o.shape[3] != d_v):
                        # Envelope: rows are only o.shape[3] wide
                        if col0 + pg * page < o.shape[3]:
                            ov = (o_ptr + o_off + pg * page).load(count=_COPY_ELEMS)
                            dov = (do_ptr + do_off + pg * page).load(count=_COPY_ELEMS)
                            for kk in cutlass.range_constexpr(_COPY_ELEMS):
                                acc = acc + ov[kk].to(cutlass.Float32) * dov[kk].to(cutlass.Float32)
                    else:
                        ov = (o_ptr + o_off + pg * page).load(count=_COPY_ELEMS)
                        dov = (do_ptr + do_off + pg * page).load(count=_COPY_ELEMS)
                        for kk in cutlass.range_constexpr(_COPY_ELEMS):
                            acc = acc + ov[kk].to(cutlass.Float32) * dov[kk].to(cutlass.Float32)
            else:
                g_off = base + row * row_stride + col0
                for pg in cutlass.range_constexpr(n_pages):
                    ov = (o_ptr + g_off + pg * page).load(count=_COPY_ELEMS)
                    dov = (do_ptr + g_off + pg * page).load(count=_COPY_ELEMS)
                    for kk in cutlass.range_constexpr(_COPY_ELEMS):
                        acc = acc + ov[kk].to(cutlass.Float32) * dov[kk].to(cutlass.Float32)
        # Allreduce over the threads sharing the row (lane-contiguous).
        n_sh = 3 if cutlass.const_expr(threads_per_row == 8) else 2
        for sh in cutlass.range_constexpr(n_sh):
            acc = acc + prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=acc,
                offset=1 << (n_sh - 1 - sh),
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
            )
        if tidx % threads_per_row == 0:
            (dd_ptr + dd_base + row).store(acc)

    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    zero_rows_per_pass = 32 if cutlass.const_expr(d_qk == 32) else 16
    zero_threads_per_row = 256 // zero_rows_per_pass
    zero_row0 = tidx // zero_threads_per_row
    zero_col0 = (tidx % zero_threads_per_row) * 4
    zero4 = cutlass.Vector.from_elements(
        (
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
        ),
        cutlass.Float32,
    )
    dqa_base = ((batch * SQ_R + m_block * M) * H + head) * d_qk
    for im in cutlass.range_constexpr(M // zero_rows_per_pass):
        for jn in cutlass.range_constexpr(d_qk // (zero_threads_per_row * 4)):
            addr = dqa_base + (zero_row0 + im * zero_rows_per_pass) * (H * d_qk) + zero_col0 + jn * zero_threads_per_row * 4
            (dqa_ptr + addr).store(zero4, alignment=16)

    if cutlass.const_expr(deterministic):
        # Reset this q-tile's relay turn counter (PDL-ordered before the main
        # kernel's first acquire, like the dq_accum zeroing above).
        if tidx == 0:
            num_q_tiles = (SQ + M - 1) // M
            sem_ptr = dq_sem.iterator.raw_ptr()
            (sem_ptr + (batch * H + head) * num_q_tiles + m_block).store(cutlass.Int32(0))


@cute.jit
def _dot_do_o_host(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    dq_accum: cute.Tensor,
    dq_sem: cute.Tensor,
    q_tile: cutlass.Constexpr[int],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    page: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    deterministic: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    m_blocks = cute.ceil_div(o.shape[1], q_tile)
    _dot_do_o_kernel(o, do, delta, dq_accum, dq_sem, q_tile, d_qk, d_v, page, use_pdl, deterministic).launch(
        grid=(m_blocks, o.shape[2], o.shape[0]),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


# ---------------------------------------------------------------------------
# Convert kernel: scrambled dq_accum (fp32) -> dQ (io dtype)
# ---------------------------------------------------------------------------


@cute.kernel
def _convert_dq_kernel(
    dq_accum: cute.Tensor,  # [B*SQ_r128*H*D] fp32
    dq: cute.Tensor,  # [B, SQ, H, D] io dtype out
    q_tile: cutlass.Constexpr[int],
    d_qk: cutlass.Constexpr[int],
    page: cutlass.Constexpr[int],
    warps_m_dq: cutlass.Constexpr[int],
    attn_scale: cutlass.Float32,
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
    m_block, head, batch = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    warp = cute.arch.warp_idx()
    g_lane = lane // 4
    p_lane = lane % 4
    SQ = dq.shape[1]
    H = dq.shape[2]
    SQ_R = ((SQ + 127) // 128) * 128
    M = q_tile
    WM_DQ = warps_m_dq
    DQ_REPS = M // (16 * WM_DQ)
    DQ_PER = d_qk * WM_DQ // 8
    DQ_NF = DQ_PER // 8
    wq = warp % WM_DQ
    wd_q = warp // WM_DQ

    dqa_ptr = dq_accum.iterator.raw_ptr()
    dq_ptr = dq.iterator.raw_ptr()

    sdQ = cutlass.Array(io_dtype, M * d_qk, space=cutlass.AddressSpace.smem, alignment=128)

    t_r = tidx // 32
    t_c = tidx % 32
    dqa_base = ((batch * SQ_R + m_block * M) * H + head) * d_qk
    for rep in cutlass.range_constexpr(DQ_REPS):
        for nf in cutlass.range_constexpr(DQ_NF):
            frag = cutlass.Array(cutlass.Float32, 4)
            for hv in cutlass.range_constexpr(2):
                i_pair = hv + rep * 2 + nf * 2 * DQ_REPS
                if cutlass.const_expr(d_qk >= 64):
                    jm = i_pair % (M // 8)
                    jn = i_pair // (M // 8)
                    addr = dqa_base + (t_r + jm * 8) * (H * d_qk) + t_c * 2 + jn * 64
                else:
                    addr = dqa_base + (t_r + (t_c // 16) * 8 + i_pair * 16) * (H * d_qk) + (t_c % 16) * 2
                pv = (dqa_ptr + addr).load(count=2)
                frag[hv * 2 + 0] = pv[0] * attn_scale
                frag[hv * 2 + 1] = pv[1] * attn_scale
            r0 = wq * 16 + rep * 16 * WM_DQ + g_lane
            r8 = r0 + 8
            c0 = wd_q * DQ_PER + nf * 8 + 2 * p_lane
            tile_ptr(sdQ, r0, c0, page=page, rows=M).store(pack_half2(frag[0], frag[1], io_dtype), alignment=4)
            tile_ptr(sdQ, r8, c0, page=page, rows=M).store(pack_half2(frag[2], frag[3], io_dtype), alignment=4)
    prims.barrier_cta_sync(0)

    q_left = SQ - m_block * M
    dq_batch_stride, dq_seq_stride, dq_head_stride, _ = dq.stride
    g_base = batch * dq_batch_stride + (m_block * M) * dq_seq_stride + head * dq_head_stride
    chunks_per_row = d_qk // _COPY_ELEMS
    for i in cutlass.range_constexpr(M * chunks_per_row // 256):
        chunk = i * 256 + tidx
        row = chunk // chunks_per_row
        col = (chunk % chunks_per_row) * _COPY_ELEMS
        if row < q_left:
            if cutlass.const_expr(dq.shape[3] != d_qk):
                # Envelope: dQ is only dq.shape[3] wide (pad columns are zero).
                if col < dq.shape[3]:
                    copy16_smem_to_gmem(
                        tile_ptr(sdQ, row, col, page=page, rows=M),
                        dq_ptr + g_base + row * dq_seq_stride + col,
                    )
            else:
                copy16_smem_to_gmem(
                    tile_ptr(sdQ, row, col, page=page, rows=M),
                    dq_ptr + g_base + row * dq_seq_stride + col,
                )


@cute.jit
def _convert_dq_host(
    dq_accum: cute.Tensor,
    dq: cute.Tensor,
    q_tile: cutlass.Constexpr[int],
    d_qk: cutlass.Constexpr[int],
    page: cutlass.Constexpr[int],
    warps_m_dq: cutlass.Constexpr[int],
    attn_scale: cutlass.Float32,
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    m_blocks = cute.ceil_div(dq.shape[1], q_tile)
    _convert_dq_kernel(dq_accum, dq, q_tile, d_qk, page, warps_m_dq, attn_scale, io_dtype, use_pdl).launch(
        grid=(m_blocks, dq.shape[2], dq.shape[0]),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


# ---------------------------------------------------------------------------
# Reduce kernel: per-q-head dk_ws/dv_ws partials (io dtype) -> dK/dV over the group
# ---------------------------------------------------------------------------


@cute.jit
def _reduce_group_vec(
    ws_ptr,
    out_ptr,
    idx,
    hkv,
    hq,
    *,
    d: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    out_batch_stride: cutlass.Constexpr[int] = 0,
    out_seq_stride: cutlass.Constexpr[int] = 0,
    out_head_stride: cutlass.Constexpr[int] = 0,
    out_strided: cutlass.Constexpr[bool] = False,
    skv: cutlass.Constexpr[int] = 0,
):
    """Sum one 16 B output vector over the group's q-head partials (fp32,
    fixed order -> deterministic) and store it in the io dtype."""
    VEC = 8  # 8 elements per vector (16 bytes)
    pos = idx * VEC
    col = pos % d
    rowh = pos // d  # (b*SKV + s)*HKV + kv_head
    kh = rowh % hkv
    bs = rowh // hkv
    in0 = (bs * hq + kh * group) * d + col
    acc = cutlass.Array(cutlass.Float32, VEC)
    for e in cutlass.range_constexpr(VEC):
        acc[e] = cutlass.Float32(0.0)
    for g in cutlass.range_constexpr(group):
        w = (ws_ptr + in0 + g * d).load(count=VEC)
        for e in cutlass.range_constexpr(VEC):
            acc[e] = acc[e] + w[e].to(cutlass.Float32)
    vec = cutlass.Vector.from_elements(tuple(acc[e].to(io_dtype) for e in range(VEC)), io_dtype)
    if cutlass.const_expr(out_strided):
        s_row = bs % skv
        b_idx = bs // skv
        (out_ptr + b_idx * out_batch_stride + s_row * out_seq_stride + kh * out_head_stride + col).store(vec, alignment=16)
    else:
        (out_ptr + pos).store(vec, alignment=16)


@cute.jit
def _reduce_group_vec_guarded(
    ws_ptr,
    out_ptr,
    idx,
    hkv,
    hq,
    *,
    d: cutlass.Constexpr[int],
    d_out: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    out_batch_stride: cutlass.Constexpr[int],
    out_seq_stride: cutlass.Constexpr[int],
    out_head_stride: cutlass.Constexpr[int],
    out_strided: cutlass.Constexpr[bool],
    skv: cutlass.Constexpr[int],
):
    """_reduce_group_vec, skipping the pad-column vectors when the output is
    narrower than the padded ws rows (envelope: those columns are zero and
    the user tensor has no room for them)."""
    if cutlass.const_expr(d_out != d):
        if (idx * 8) % d < d_out:
            _reduce_group_vec(
                ws_ptr,
                out_ptr,
                idx,
                hkv,
                hq,
                d=d,
                group=group,
                io_dtype=io_dtype,
                out_batch_stride=out_batch_stride,
                out_seq_stride=out_seq_stride,
                out_head_stride=out_head_stride,
                out_strided=out_strided,
                skv=skv,
            )
    else:
        _reduce_group_vec(
            ws_ptr,
            out_ptr,
            idx,
            hkv,
            hq,
            d=d,
            group=group,
            io_dtype=io_dtype,
            out_batch_stride=out_batch_stride,
            out_seq_stride=out_seq_stride,
            out_head_stride=out_head_stride,
            out_strided=out_strided,
            skv=skv,
        )


@cute.kernel
def _dkv_reduce_kernel(
    dk_ws: cute.Tensor,  # [B, SKV, HQ, D] io dtype (one dK partial per q head)
    dv_ws: cute.Tensor,  # [B, SKV, HQ, DV] io dtype (one dV partial per q head)
    dk: cute.Tensor,  # [B, SKV, HKV, D] io dtype out
    dv: cute.Tensor,  # [B, SKV, HKV, DV] io dtype out
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
):
    # one thread per 16 B output vector, serial fp32 accumulation over the group's q-head slices (fixed order -> deterministic).
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    B = dk.shape[0]
    SKV = dk.shape[1]
    HKV = dk.shape[2]
    HQ = HKV * group
    VEC = 8  # 8 elements per vector (16 bytes)
    dkws_ptr = dk_ws.iterator.raw_ptr()
    dvws_ptr = dv_ws.iterator.raw_ptr()
    dk_ptr = dk.iterator.raw_ptr()
    dv_ptr = dv.iterator.raw_ptr()
    dk_batch_stride, dk_seq_stride, dk_head_stride, _ = dk.stride
    dv_batch_stride, dv_seq_stride, dv_head_stride, _ = dv.stride
    dk_strided = (dk_batch_stride, dk_seq_stride, dk_head_stride) != (SKV * HKV * d_qk, HKV * d_qk, d_qk)
    dv_strided = (dv_batch_stride, dv_seq_stride, dv_head_stride) != (SKV * HKV * d_v, HKV * d_v, d_v)
    gidx = bidx * 256 + tidx  # host launch 256 threads
    if cutlass.const_expr(d_qk == d_v):
        OUT_VECS = B * SKV * HKV * d_qk // VEC
        if gidx < OUT_VECS:
            _reduce_group_vec_guarded(
                dkws_ptr,
                dk_ptr,
                gidx,
                HKV,
                HQ,
                d=d_qk,
                d_out=dk.shape[3],
                group=group,
                io_dtype=io_dtype,
                out_batch_stride=dk_batch_stride,
                out_seq_stride=dk_seq_stride,
                out_head_stride=dk_head_stride,
                out_strided=dk_strided,
                skv=SKV,
            )
            _reduce_group_vec_guarded(
                dvws_ptr,
                dv_ptr,
                gidx,
                HKV,
                HQ,
                d=d_qk,
                d_out=dv.shape[3],
                group=group,
                io_dtype=io_dtype,
                out_batch_stride=dv_batch_stride,
                out_seq_stride=dv_seq_stride,
                out_head_stride=dv_head_stride,
                out_strided=dv_strided,
                skv=SKV,
            )
    else:
        # Unequal head dims: dK and dV vectors index different row widths, so
        # the flat thread range covers dK's vectors first, then dV's.
        K_VECS = B * SKV * HKV * d_qk // VEC
        V_VECS = B * SKV * HKV * d_v // VEC
        if gidx < K_VECS:
            _reduce_group_vec_guarded(
                dkws_ptr,
                dk_ptr,
                gidx,
                HKV,
                HQ,
                d=d_qk,
                d_out=dk.shape[3],
                group=group,
                io_dtype=io_dtype,
                out_batch_stride=dk_batch_stride,
                out_seq_stride=dk_seq_stride,
                out_head_stride=dk_head_stride,
                out_strided=dk_strided,
                skv=SKV,
            )
        else:
            if gidx < K_VECS + V_VECS:
                _reduce_group_vec_guarded(
                    dvws_ptr,
                    dv_ptr,
                    gidx - K_VECS,
                    HKV,
                    HQ,
                    d=d_v,
                    d_out=dv.shape[3],
                    group=group,
                    io_dtype=io_dtype,
                    out_batch_stride=dv_batch_stride,
                    out_seq_stride=dv_seq_stride,
                    out_head_stride=dv_head_stride,
                    out_strided=dv_strided,
                    skv=SKV,
                )


@cute.jit
def _dkv_reduce_host(
    dk_ws: cute.Tensor,
    dv_ws: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    group: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    if cutlass.const_expr(d_qk == d_v):
        out_vecs = cute.ceil_div(dk.shape[0] * dk.shape[1] * dk.shape[2] * d_qk, 8)
    else:
        # Split index space: one thread per dK vector plus one per dV vector.
        out_vecs = cute.ceil_div(dk.shape[0] * dk.shape[1] * dk.shape[2] * (d_qk + d_v), 8)
    _dkv_reduce_kernel(dk_ws, dv_ws, dk, dv, d_qk, d_v, group, io_dtype, use_pdl).launch(
        grid=(cute.ceil_div(out_vecs, 256), 1, 1),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.kernel
def _dsink_kernel(
    lse: cute.Tensor,  # [B, HQ, SQ] fp32 (natural-log, sink folded in by the fwd)
    delta: cute.Tensor,  # [B, HQ, SQ_r128] fp32 (dot_do_o output)
    sink: cute.Tensor,  # [HQ] fp32 sink logits
    dsink: cute.Tensor,  # [HQ] fp32 out
    seq_q_lens: Optional[cute.Tensor],  # [B] int32; None unless seq_q_lens_present
    use_pdl: cutlass.Constexpr[bool],
):
    """dsink[h] = -sum_{b,q} exp(sink[h] - lse[b,h,q]) * delta[b,h,q].

    One warp per query head, fixed reduction order -> bitwise deterministic."""
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
    head, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    B = lse.shape[0]
    HQ = lse.shape[1]
    SQ = lse.shape[2]
    SQ_R = delta.shape[2]
    lse_batch_stride, lse_head_stride, lse_seq_stride = lse.stride
    lse_ptr = lse.iterator.raw_ptr()
    delta_ptr = delta.iterator.raw_ptr()
    s_val = (sink.iterator.raw_ptr() + head).load()
    inf = cutlass.Float32(float("inf"))
    acc = cutlass.Float32(0.0)
    batch = cutlass.Int32(0)
    # batch loop
    while batch < B:
        lse_base = batch * lse_batch_stride + head * lse_head_stride
        delta_base = (batch * HQ + head) * SQ_R
        q_bound = SQ
        if cutlass.const_expr(seq_q_lens is not None):
            q_bound = cute.math.max(cutlass.Int32(0), cute.math.min(seq_q_lens[batch], cutlass.Int32(SQ)))
        q = cutlass.Int32(tidx)
        while q < q_bound:
            lv = (lse_ptr + lse_base + q * lse_seq_stride).load()
            # Padded / trimmed rows carry LSE = -inf: skip them, exp(sink + inf) * 0 = NaN
            if lv > -inf and lv < inf:
                dd = (delta_ptr + delta_base + q).load()
                acc = acc + cute.math.exp2((s_val - lv) * cutlass.Float32(_LOG2E), fastmath=True) * dd
            q = q + 32
        batch = batch + 1
    for sh in cutlass.range_constexpr(5):
        acc = acc + prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=acc,
            offset=1 << (4 - sh),
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
    if tidx == 0:
        (dsink.iterator.raw_ptr() + head).store(-acc)


@cute.jit
def _dsink_host(
    lse: cute.Tensor,
    delta: cute.Tensor,
    sink: cute.Tensor,
    dsink: cute.Tensor,
    seq_q_lens: Optional[cute.Tensor],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    _dsink_kernel(lse, delta, sink, dsink, seq_q_lens, use_pdl).launch(
        grid=(lse.shape[1], 1, 1),
        block=(32, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    compute_capability: tuple[int, int],
    b: int = 1,
    qh: int = 1,
    sq: int = 128,
    skv: int = 128,
    d_qk: int = 128,
    d_v: int = 0,  # the V/O/dO/dV head dim (0 = ``d_qk``; unequal dims serve MLA).
    kvh: int = 0,  # the KV head count for GQA/MQA (0 = ``qh``, plain MHA).
    lse_strides: "tuple[int, int, int] | None" = None,  # non-contiguous LSE (B, H, S) strides; None = contiguous
    # Non-compact io ports: declared BSHD (batch, seq, head) element strides; None = compact BSHD.
    q_strides: "tuple[int, int, int] | None" = None,
    k_strides: "tuple[int, int, int] | None" = None,
    v_strides: "tuple[int, int, int] | None" = None,
    o_strides: "tuple[int, int, int] | None" = None,
    do_strides: "tuple[int, int, int] | None" = None,
    dq_strides: "tuple[int, int, int] | None" = None,
    dk_strides: "tuple[int, int, int] | None" = None,
    dv_strides: "tuple[int, int, int] | None" = None,
) -> SimpleNamespace:
    """Compile and cache the backward chain for one BSHD shape (dot, main,
    cvt, plus the group-reduce kernel when GQA). Non-compact ports carry
    their declared strides as compile-time constants.
    """

    kvh = int(kvh) or int(qh)
    d_v = int(d_v) or int(d_qk)
    if qh % kvh:
        raise ValueError(f"GQA requires qh to be a multiple of kvh; got qh={qh}, kvh={kvh}")
    has_gqa = kvh != qh
    bwd = SM120FusedMultiHeadAttentionFP16Backward(
        in_dtype=STORAGE_DTYPE,
        is_causal=PARAMS.is_causal,
        causal_top_left=PARAMS.causal_top_left,
        window_size_left=PARAMS.window_size_left,
        window_size_right=PARAMS.window_size_right,
        deterministic=PARAMS.deterministic,
        head_dim_qk=d_qk,
        head_dim_v=d_v,
        use_pdl=PARAMS.use_pdl,
        q_tile=PARAMS.q_tile,
        kv_tile=PARAMS.kv_tile,
        seq_kv_lens_present=PARAMS.seq_kv_lens_present,
        seq_q_lens_present=PARAMS.seq_q_lens_present,
        sink_present=PARAMS.sink_present,
    )
    d_qk_orig, d_v_orig = bwd.d_qk_orig, bwd.d_v_orig
    d_qk, d_v = bwd.d_qk, bwd.d_v
    sq_r = ceil_div(sq, 128) * 128

    def _fake(dtype, shape, strides=None):
        """Compact fake, or one carrying a port's declared (batch, seq, head)
        strides (the TMA descriptors and pointer math then address that
        layout natively)."""
        if strides is None:
            return make_fake_compact_tensor(
                dtype,
                shape,
                stride_order=tuple(range(len(shape) - 1, -1, -1)),
                assumed_align=16,
            )
        batch_stride, seq_stride, head_stride = strides
        return make_fake_tensor(dtype, shape, (batch_stride, seq_stride, head_stride, 1), assumed_align=16)

    fake_q = _fake(STORAGE_DTYPE, (b, sq, qh, d_qk_orig), q_strides)
    fake_k = _fake(STORAGE_DTYPE, (b, skv, kvh, d_qk_orig), k_strides)
    fake_v = _fake(STORAGE_DTYPE, (b, skv, kvh, d_v_orig), v_strides)
    fake_o = _fake(STORAGE_DTYPE, (b, sq, qh, d_v_orig), o_strides)
    fake_do = _fake(STORAGE_DTYPE, (b, sq, qh, d_v_orig), do_strides)
    fake_dq = _fake(STORAGE_DTYPE, (b, sq, qh, d_qk_orig), dq_strides)
    fake_dk = _fake(STORAGE_DTYPE, (b, skv, kvh, d_qk_orig), dk_strides)
    fake_dv = _fake(STORAGE_DTYPE, (b, skv, kvh, d_v_orig), dv_strides)
    if lse_strides is not None:
        # f32 scalars -> 4 B alignment
        # A strided stats view can start at any fp32 element; the compact
        # branch keeps the historical allocation-backed 16-byte assumption.
        fake_lse = make_fake_tensor(cutlass.Float32, (b, qh, sq), tuple(lse_strides), assumed_align=4)
    else:
        fake_lse = _fake(cutlass.Float32, (b, qh, sq))
    fake_delta = _fake(cutlass.Float32, (b, qh, sq_r))
    fake_dq_accum = _fake(cutlass.Float32, (b * sq_r * qh * d_qk,))
    # Sized for the smallest legal q-tile (32) so one formula covers every
    # tile choice; must match the adapter's carve (scratch_workspace_bytes).
    fake_dq_sem = _fake(cutlass.Int32, (b * qh * ceil_div(sq, 32),))
    # Main-kernel dK/dV destinations, always HQ-headed: alias dk/d_v for MHA
    # (qh == kvh); per-q-head partials summed by _dkv_reduce_kernel for GQA.
    fake_dk_ws = _fake(STORAGE_DTYPE, (b, skv, qh, d_qk)) if has_gqa else fake_dk
    fake_dv_ws = _fake(STORAGE_DTYPE, (b, skv, qh, d_v)) if has_gqa else fake_dv
    fake_seq_q_lens = _fake(cutlass.Int32, (b,)) if PARAMS.seq_q_lens_present else None
    fake_seq_kv_lens = _fake(cutlass.Int32, (b,)) if PARAMS.seq_kv_lens_present else None
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    options = "--enable-tvm-ffi"

    compiled_dot = cute.compile(
        _dot_do_o_host,
        fake_o,
        fake_do,
        fake_delta,
        fake_dq_accum,
        fake_dq_sem,
        bwd.q_tile,
        d_qk,
        d_v,
        bwd.page,
        bwd.use_pdl,
        bwd.deterministic,
        fake_stream,
        options=options,
    )
    compiled_main = cute.compile(
        bwd,
        fake_q,
        fake_k,
        fake_v,
        fake_do,
        fake_lse,
        fake_delta,
        fake_dq_accum,
        fake_dq_sem,
        fake_dk_ws,
        fake_dv_ws,
        fake_seq_q_lens,
        fake_seq_kv_lens,
        cutlass.Float32(1.0),
        cutlass.Float32(1.0),
        fake_stream,
        options=options,
    )
    compiled_cvt = cute.compile(
        _convert_dq_host,
        fake_dq_accum,
        fake_dq,
        bwd.q_tile,
        d_qk,
        bwd.page,
        bwd.warps_m_dq,
        cutlass.Float32(1.0),
        STORAGE_DTYPE,
        bwd.use_pdl,
        fake_stream,
        options=options,
    )
    compiled_reduce = None
    if has_gqa:
        compiled_reduce = cute.compile(
            _dkv_reduce_host,
            fake_dk_ws,
            fake_dv_ws,
            fake_dk,
            fake_dv,
            d_qk,
            d_v,
            qh // kvh,
            STORAGE_DTYPE,
            bwd.use_pdl,
            fake_stream,
            options=options,
        )
    compiled_dsink = None
    if PARAMS.dsink_present:
        fake_sink = _fake(cutlass.Float32, (qh,))
        fake_dsink = _fake(cutlass.Float32, (qh,))
        compiled_dsink = cute.compile(
            _dsink_host,
            fake_lse,
            fake_delta,
            fake_sink,
            fake_dsink,
            fake_seq_q_lens,
            bwd.use_pdl,
            fake_stream,
            options=options,
        )
    return SimpleNamespace(dot=compiled_dot, main=compiled_main, cvt=compiled_cvt, reduce=compiled_reduce, dsink=compiled_dsink)
