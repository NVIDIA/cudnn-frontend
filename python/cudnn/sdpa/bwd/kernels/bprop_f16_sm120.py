# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FROST SM120 SDPA backward kernel template (fp16 / bf16).

A fused multi-head attention (FMHA) backward for the NVIDIA Blackwell
GeForce SM120 family (SM120 and SM121), built on TMA loads and a
warp-specialized producer/consumer schedule.

The algorithm is FlashAttention-2 (Dao, 2023;
https://github.com/Dao-AILab/flash-attention, BSD-3-Clause), KV-stationary
and parallel over seq-KV: each CTA owns one KV tile and walks the query
tiles in descending order, computing the five chained GEMMs (S = Q*K^T,
dP = dO*V^T, dV += P^T*dO, dQ = dS*K, dK += dS^T*Q) with the softmax VJP
fused in registers. dK/dV accumulate in registers across the whole pass
(no atomics); dQ is reduced through an fp32 workspace and finalized by a
small convert kernel.

Support:
* Input dtypes fp16 and bf16 (output dtype matches).
* Head dims 32, 64, 128, 192, and 256 natively; any other multiple of 8
  up to 256 runs on the next of those sizes, with the TMA envelope
  zero-filling the pad columns in place (no staging copies).
* GQA/MQA (H_q a multiple of H_kv).
* Causal masks (top-left or bottom-right), right-band-widened causal
  (window_size_right), sliding-window masks, and padding masks.
* Additive bias ([1|B, H, S_Q, S_KV], post-scale pre-softmax), optionally
  with a dBias output (fp32 red.add accumulator; a broadcast batch dim
  reduces over B for free, so dBias is non-deterministic when B > 1).
* LSE input as the natural-log forward stats, fp32 (B, H, S_Q), in any
  non-broadcast layout.
* Any dense io layout whose head dim is innermost-contiguous and whose
  batch/seq/head strides are 16-byte multiples (TMA's global-stride rule).
* Not supported: dropout, ALiBi, softcap.

Kernels (compiled through the per-shape ``compile()`` cache at the bottom
of this module): every backward call launches ``dot`` (delta =
rowsum(dO*O), which also zeroes the dq_accum workspace), ``main`` (the
fused five-GEMM pass writing dK/dV), and ``cvt`` (dq_accum fp32 -> the dQ
io dtype). GQA adds ``reduce``, a dSink_token output adds ``dsink``, and a
dBias output adds ``dbias_cvt`` (fp32 accumulator -> the io dtype; an fp32
dBias output doubles as the accumulator and needs no convert).

Deterministic dQ comes in two flavors. The relay (default) keeps the
fused chain and orders the dq_accum adds per q-tile with turn counters.
The ``det_2kernel`` split instead streams unscaled dS (io dtype) to a
``[B, H_q, S_q, S_kv]`` workspace, drops the main kernel's dQ section
(GEMM 4, scatter, relay), and computes dQ = attn_scale * dS @ K in a
separate ``dq2k`` kernel (``bprop_chain_f16_sm120.py``) that replaces
``cvt``. The adapter falls back to the relay when the dS workspace does
not fit in device memory.
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
from cudnn.sdpa.bwd.config_sm120 import ROW_ROUND, SUPPORTED_HEAD_DIMS, TemplateParams, padded_head_dims, validate_params
from cudnn.sdpa.bwd.kernels._common_sm120 import (
    _COPY_ELEMS,
    _LOG2E,
    ceil_div,
    copy16_smem_to_gmem,
    load_a_frag,
    load_a_frag_transposed,
    mma_abregs,
    mma_bstream,
    pack_half2,
    tile_ptr,
)
from cudnn.sdpa.bwd.kernels.bprop_chain_f16_sm120 import (
    SM120DetDqGemmKernel,
    convert_dbias_host,
    convert_dq_host,
    dkv_reduce_host,
    dot_do_o_host,
    dsink_host,
)

# The FROST loader injects one immutable specialization before executing this
# module. A direct import uses the dense FP16 defaults.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
validate_params(PARAMS)

STORAGE_DTYPE = {DTYPE_FP16: cutlass.Float16, DTYPE_BF16: cutlass.BFloat16}[PARAMS.dtype_qkv]


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
def _red_add_f32x2(ptr, v0: cutlass.Float32, v1: cutlass.Float32) -> None:
    """One red.global.add.v2.f32 covering a thread's adjacent (c0,c1) pair."""
    prims.inline_ptx(
        "red.global.add.v2.f32 [$0], {$1, $2};",
        read_only_args=[ptr, v0, v1],
    )


@cute.jit
def _red_add_f32(ptr, v0: cutlass.Float32) -> None:
    prims.inline_ptx(
        "red.global.add.f32 [$0], $1;",
        read_only_args=[ptr, v0],
    )


@cute.jit
def _bwd_gemm4_dq(
    acc_dq,
    sdS,
    sK,
    wm_dq,
    wn_dq,
    lane,
    *,
    DQ_ROW_BLOCKS: cutlass.Constexpr[int],
    DQ_COL_FRAGS: cutlass.Constexpr[int],
    KV_CHUNKS: cutlass.Constexpr[int],
    WM_DQ: cutlass.Constexpr[int],
    Q_TILE: cutlass.Constexpr[int],
    KV_TILE: cutlass.Constexpr[int],
    DS_CHUNK_ELEMS: cutlass.Constexpr[int],
    CHUNK_ELEMS: cutlass.Constexpr[int],
    DQ_COLS: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    """GEMM 4: acc_dq = dS @ K^T (reads only sdS/sK, never sQ)."""
    for i in cutlass.range_constexpr(DQ_ROW_BLOCKS * DQ_COL_FRAGS * 4):
        acc_dq[i] = cutlass.Float32(0.0)
    for k_chunk in cutlass.range_constexpr(KV_CHUNKS):
        a_frag = []
        for row_blk in cutlass.range_constexpr(DQ_ROW_BLOCKS):
            sf = load_a_frag(sdS, k_chunk, wm_dq * 16 + row_blk * 16 * WM_DQ, lane, rows=Q_TILE, chunk_elems=DS_CHUNK_ELEMS)
            a_frag = a_frag + [sf[0], sf[1], sf[2], sf[3]]
        mma_bstream(
            acc_dq,
            a_frag,
            sK,
            b_k_step=k_chunk,
            M=16 * DQ_ROW_BLOCKS,
            N=DQ_COLS,
            b_trans=True,
            b_rows=KV_TILE,
            b_chunk_elems=CHUNK_ELEMS,
            lane=lane,
            ab_dtype=io_dtype,
            col_base=wn_dq * DQ_COLS,
        )


@cute.jit
def _bwd_dq_scatter(
    acc_dq,
    dq_accum_ptr,
    dq_accum_base,
    math_tidx,
    H,
    *,
    DQ_ROW_BLOCKS: cutlass.Constexpr[int],
    DQ_COL_FRAGS: cutlass.Constexpr[int],
    Q_TILE: cutlass.Constexpr[int],
    D_QK: cutlass.Constexpr[int],
):
    """dQ accumulate into the scrambled dq_accum workspace."""
    t_row = math_tidx // 32
    t_col = math_tidx % 32
    for row_blk in cutlass.range_constexpr(DQ_ROW_BLOCKS):
        for col_frag in cutlass.range_constexpr(DQ_COL_FRAGS):
            for hf in cutlass.range_constexpr(2):
                i_pair = hf + row_blk * 2 + col_frag * 2 * DQ_ROW_BLOCKS
                if cutlass.const_expr(D_QK >= 64):
                    pair_row = i_pair % (Q_TILE // 8)
                    pair_col = i_pair // (Q_TILE // 8)
                    addr = dq_accum_base + (t_row + pair_row * 8) * (H * D_QK) + t_col * 2 + pair_col * 64
                else:
                    addr = dq_accum_base + (t_row + (t_col // 16) * 8 + i_pair * 16) * (H * D_QK) + (t_col % 16) * 2
                poff = (row_blk * DQ_COL_FRAGS + col_frag) * 4 + hf * 2
                _red_add_f32x2(dq_accum_ptr + addr, acc_dq[poff + 0], acc_dq[poff + 1])


@cute.jit
def _bwd_gemm5_dk(
    acc_dk,
    sdS,
    sQ_stage,
    wm_dkv,
    wn_dkv,
    lane,
    *,
    Q_CHUNKS: cutlass.Constexpr[int],
    DKV_ROW_BLOCKS: cutlass.Constexpr[int],
    WM_DKV: cutlass.Constexpr[int],
    Q_TILE: cutlass.Constexpr[int],
    DS_CHUNK_ELEMS: cutlass.Constexpr[int],
    CHUNK_ELEMS: cutlass.Constexpr[int],
    DK_COLS: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    """GEMM 5: acc_dk += dS^T @ Q (the iteration's last sQ reader)."""
    for k_chunk in cutlass.range_constexpr(Q_CHUNKS):
        a_frag = []
        for row_blk in cutlass.range_constexpr(DKV_ROW_BLOCKS):
            sf = load_a_frag_transposed(
                sdS,
                k_chunk,
                wm_dkv * 16 + row_blk * 16 * WM_DKV,
                lane,
                rows=Q_TILE,
                chunk_elems=DS_CHUNK_ELEMS,
            )
            a_frag = a_frag + [sf[0], sf[2], sf[1], sf[3]]
        mma_bstream(
            acc_dk,
            a_frag,
            sQ_stage,
            b_k_step=k_chunk,
            M=16 * DKV_ROW_BLOCKS,
            N=DK_COLS,
            b_trans=True,
            b_rows=Q_TILE,
            b_chunk_elems=CHUNK_ELEMS,
            lane=lane,
            ab_dtype=io_dtype,
            col_base=wn_dkv * DK_COLS,
        )


@cute.jit
def _bwd_relay_wait(relay_sem, q_block, relay_turn, warp):
    """Deterministic-relay entry: block the 8 compute warps until it is this
    CTA's turn for q-tile ``q_block`` (FA3 / cuDNN-SM90 STAGES=4 scheme).

    One elected lane of warp 0 spins on an acquire load of the turn counter;
    barrier 6 (compute warps only — never the producer's 288-thread
    barriers) releases the other warps into the dQ scatter."""
    if warp == 0:
        if prims.elect_sync():
            while (
                prims.load_ext(
                    relay_sem + q_block,
                    order=prims.MemOrder.ACQUIRE,
                    scope=prims.MemScope.GPU,
                )
                != relay_turn
            ):
                pass
    cute.arch.barrier(barrier_id=6, number_of_threads=256)


@cute.jit
def _bwd_relay_release(relay_sem, q_block, relay_turn, warp):
    """Deterministic-relay exit: pass the turn for ``q_block`` to the next
    kv tile once every compute warp has issued its dQ reds.

    The release store alone orders the relaxed reds before the handoff: the
    barrier sequences the other warps' reds against this thread (CTA scope)
    and st.release makes them cumulatively visible at GPU scope (its own
    membar; an extra fence here doubled the per-handoff drain cost)."""
    cute.arch.barrier(barrier_id=6, number_of_threads=256)
    if warp == 0:
        if prims.elect_sync():
            prims.store_ext(
                (relay_turn + 1).ir_value(),
                relay_sem + q_block,
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
        det_2kernel: bool = False,
        bias_present: bool = False,
        dbias_present: bool = False,
        bias_is_fp32: bool = False,
    ):
        self.in_dtype = in_dtype
        # causal mask
        self.is_causal = is_causal
        self.causal_top_left = bool(causal_top_left)
        # sliding window
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        # right-band widening of causal mask
        self.right_slack = window_size_right if window_size_right is not None else 0
        # deterministic dQ
        self.deterministic = bool(deterministic)
        # two-kernel deterministic mode
        self.det_2k = bool(det_2kernel)
        # padding mask
        self.seq_kv_lens_present = bool(seq_kv_lens_present)
        self.seq_q_lens_present = bool(seq_q_lens_present)
        # additive bias and its gradient output
        self.bias_present = bool(bias_present)
        self.dbias_present = bool(dbias_present)
        if self.dbias_present and not self.bias_present:
            raise ValueError("dbias_present requires bias_present")
        self.bias_dtype = cutlass.Float32 if bias_is_fp32 else in_dtype
        # sink LSE is finite on padded rows; trim them explicitly (LSE := +inf, P = 0)
        self.trim_q_rows = bool(sink_present) and self.seq_q_lens_present
        # use padded head dim for compute
        self.d_qk_orig = int(head_dim_qk)
        self.d_v_orig = int(head_dim_v) or int(head_dim_qk)
        for orig, tag in ((self.d_qk_orig, "d_qk"), (self.d_v_orig, "d_v")):
            if orig % 8 or orig <= 0:
                raise ValueError(f"{tag} must be a positive multiple of 8; got {orig}")
        pads = padded_head_dims(self.d_qk_orig, self.d_v_orig)
        if pads is None:
            raise ValueError(f"head dims must be <= {max(SUPPORTED_HEAD_DIMS)}; got d_qk={self.d_qk_orig}, d_v={self.d_v_orig}")
        self.d_qk, self.d_v = pads
        # current MLA requires both to be multiples of 64 so one smem swizzle serves every tile.
        if self.d_v != self.d_qk and (self.d_qk % 64 or self.d_v % 64):
            raise ValueError(f"unequal head dims must both be multiples of 64; got d_qk={self.d_qk}, d_v={self.d_v}")
        # envelope flags: the TMA load zero-fills the pad columns in place
        self.qk_envelope = self.d_qk_orig != self.d_qk
        self.v_envelope = self.d_v_orig != self.d_v
        # PDL
        self.use_pdl = bool(use_pdl)
        # default tile config for given head size
        self.q_tile, self.kv_tile = self.DEFAULT_TILES[self.d_qk]
        if q_tile:
            self.q_tile = int(q_tile)
        if kv_tile:
            self.kv_tile = int(kv_tile)
        if 128 % self.q_tile:
            # dq_accum is scrambled in 128-row blocks (S_Q_R is rounded to 128
            # and the convert kernel unscrambles per block).
            raise ValueError(f"q_tile must divide 128; got {self.q_tile}")
        # Warp layouts: the sweep-tuned triple for this exact tile choice
        # when we have one, else the largest-valid derivation
        tuned = self.CONFIG.get((self.d_qk, self.q_tile, self.kv_tile))
        if tuned is not None:
            self.warps_m_sdp, self.warps_m_dkv, self.warps_m_dq = tuned
            # The tuned dK partition does not tile the dV head dim.
            if (self.d_v * self.warps_m_dkv // 8) % 16:
                self.warps_m_dkv = largest_warp_partition(self.kv_tile, self.d_qk, self.d_v)
        else:
            self.warps_m_sdp = largest_warp_partition(self.q_tile, self.kv_tile)
            self.warps_m_dkv = largest_warp_partition(self.kv_tile, self.d_qk, self.d_v)
            self.warps_m_dq = largest_warp_partition(self.q_tile, self.d_qk)
        for a_, m_dim, n_dim, tag in (
            (self.warps_m_sdp, self.q_tile, self.kv_tile, "warps_m_sdp"),
            (self.warps_m_dkv, self.kv_tile, self.d_qk, "warps_m_dkv"),
            (self.warps_m_dkv, self.kv_tile, self.d_v, "warps_m_dkv (dV)"),
            (self.warps_m_dq, self.q_tile, self.d_qk, "warps_m_dq"),
        ):
            if 8 % a_ or m_dim % (16 * a_) or (n_dim * a_ // 8) % 16:
                raise ValueError(f"invalid {tag}={a_} for q_tile={self.q_tile} kv_tile={self.kv_tile} d_qk={self.d_qk} d_v={self.d_v}")
        # tma swizzle
        self.chunk_elems = 64 if self.d_qk % 64 == 0 and self.d_v % 64 == 0 else 32
        self.tma_swizzle = cuda.TensorMapSwizzle.s128b if self.chunk_elems == 64 else cuda.TensorMapSwizzle.s64b
        # 1 TMA warp, 8 compute warps and 3 unused warps
        self.threads = 384
        self.num_consumer_warps = 8
        self.load_warp_id = self.num_consumer_warps

        Q_TILE, KV_TILE, D_QK, D_V = self.q_tile, self.kv_tile, self.d_qk, self.d_v
        # Double-buffer Q when SMEM allows (prefetch hides the TMA latency);
        # single-buffered Q pays an end-of-iteration rendezvous (d256's only fit).
        cap = cutlass.utils.get_smem_capacity_in_bytes("sm_120")
        for q_stages in (2, 1):
            smem_elems = q_stages * Q_TILE * D_QK + Q_TILE * D_V + KV_TILE * D_QK + max(KV_TILE * D_V, 2 * Q_TILE * KV_TILE)
            if smem_elems * in_dtype.bytes <= cap:
                break
        else:
            raise ValueError(f"smem {smem_elems * in_dtype.bytes} bytes exceeds the sm_120 cap of {cap} bytes even single-buffered")
        # det_2k: double-buffer dO too when it still fits.
        self.do_stages = 1
        if self.det_2k and q_stages == 2 and (smem_elems + Q_TILE * D_V) * in_dtype.bytes <= cap:
            self.do_stages = 2
            smem_elems += Q_TILE * D_V
        # smem element offsets.
        self.q_stages = q_stages
        self.off_sQ = 0  # `q_stages` buffers
        self.off_sdO = q_stages * Q_TILE * D_QK  # `do_stages` buffers
        self.off_sK = self.off_sdO + self.do_stages * Q_TILE * D_V
        self.off_sV = self.off_sK + KV_TILE * D_QK
        self.off_sdS = self.off_sV  # aliases sV (V is in regs)
        self.off_sP = self.off_sV + Q_TILE * KV_TILE
        self.smem_elems = smem_elems

    @cute.jit
    def load_tma_tile(self, s_dst, tma_desc, mbar, batch, head, seq, rows: cutlass.Constexpr[int], cols: cutlass.Constexpr[int]):
        """Load one chunked/swizzled `(rows, cols)` tile with TMA."""
        elems_per_chunk = rows * self.chunk_elems
        for chunk in cutlass.range_constexpr(cols // self.chunk_elems):
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_dst.subview(chunk * elems_per_chunk),
                    tma_desc.get_ptr(),
                    (chunk * self.chunk_elems, head, seq, batch),
                    mbar,
                )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,  # [B, S_Q,  H_Q, D] io dtype (BSHD)
        k: cute.Tensor,  # [B, S_KV, H_KV, D]
        v: cute.Tensor,  # [B, S_KV, H_KV, DV]
        do: cute.Tensor,  # [B, S_Q,  H_Q, DV]
        lse: cute.Tensor,  # [B, H_Q, S_Q] fp32 (natural-log LSE)
        delta: cute.Tensor,  # [B, H_Q, S_Q_r128] fp32 (dot_do_o output)
        dq_accum: Optional[cute.Tensor],  # [B*S_Q_r128*H_Q*D_QK] fp32 (scrambled, zeroed; relay deterministic only)
        dq_sem: Optional[cute.Tensor],  # [B*H_Q*num_q_tiles] int32 relay turn counters, one per (batch, head, q-tile); zeroed by dot (relay deterministic only)
        ds_ws: Optional[cute.Tensor],  # [B, H_Q, S_Q, SKV_pad] io dtype dS out (det_2kernel only)
        dk_ws: cute.Tensor,  # [B, S_KV, H_Q, D] dK destination: dk itself when MHA; per-q-head partials summed by dkv_reduce_kernel when GQA
        dv_ws: cute.Tensor,  # [B, S_KV, H_Q, DV] dV destination (same as dK)
        seq_q_lens: Optional[cute.Tensor],  # [B] int32 per-batch Q lengths; None unless seq_q_lens_present
        seq_kv_lens: Optional[cute.Tensor],  # [B] int32 per-batch KV lengths; None unless seq_kv_lens_present
        bias: Optional[cute.Tensor],  # [1|B, H_Q, S_Q, S_KV] additive bias (contiguous); None unless bias_present
        dbias_accum: Optional[cute.Tensor],  # [1|B, H_Q, S_Q, S_KV] fp32 dBias accumulator; None unless dbias_present
        tma_q_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_do_desc: cutlass.GridConstant[cuda.TensorMap],
        softmax_scale_log2: cutlass.Float32,  # scale * log2(e)
        attn_scale: cutlass.Float32,  # linear scale (dq/dk output)
        tma_dsws_desc: Optional[cutlass.GridConstant[cuda.TensorMap]],  # dS workspace store map (det_2kernel only)
    ) -> None:
        io_dtype = self.in_dtype
        D_QK = self.d_qk
        D_V = self.d_v
        Q_TILE = self.q_tile
        KV_TILE = self.kv_tile
        CHUNK_ELEMS = self.chunk_elems
        Q_STAGES = self.q_stages
        DO_STAGES = self.do_stages
        # sdS's chunk elems in kv_tile
        DS_CHUNK_ELEMS = 64 if self.kv_tile >= 64 else self.kv_tile
        WM_SDP = self.warps_m_sdp  # S/dP warp grid (WM_SDP, 8//WM_SDP), WM_SDP along q rows
        WM_DKV = self.warps_m_dkv  # dK/dV warp grid (WM_DKV, 8//WM_DKV), WM_DKV along kv rows
        WM_DQ = self.warps_m_dq  # dQ warp grid (WM_DQ, 8//WM_DQ), WM_DQ along q rows

        # SdP: warp (wm_sdp, wn_sdp); wm_sdp steps over SDP_ROW_BLOCKS 16-row MMA blocks
        # interleaved by 16*WM_SDP; wn_sdp covers SDP_COLS contiguous kv columns.
        SDP_ROW_BLOCKS = Q_TILE // (16 * WM_SDP)
        SDP_COLS = KV_TILE * WM_SDP // 8
        SDP_COL_FRAGS = SDP_COLS // 8
        # dKV: warp (wm_dkv, wn_dkv); DKV_ROW_BLOCKS 16-row MMA blocks interleaved by 16*WM_DKV.
        DKV_ROW_BLOCKS = KV_TILE // (16 * WM_DKV)
        DK_COLS = D_QK * WM_DKV // 8
        DK_COL_FRAGS = DK_COLS // 8
        DV_COLS = D_V * WM_DKV // 8
        DV_COL_FRAGS = DV_COLS // 8
        # dQ: warp (wm_dq, wn_dq).
        DQ_ROW_BLOCKS = Q_TILE // (16 * WM_DQ)
        DQ_COLS = D_QK * WM_DQ // 8
        DQ_COL_FRAGS = DQ_COLS // 8

        D_QK_CHUNKS = D_QK // 16  # S = Q @ K^T k-reduce (D_QK)
        D_V_CHUNKS = D_V // 16  # dP = dO @ V^T k-reduce (D_V)
        Q_CHUNKS = Q_TILE // 16  # dK/dV k-reduce
        KV_CHUNKS = KV_TILE // 16  # dQ k-reduce
        V_ROW_BLOCKS = SDP_COLS // 16  # V-in-regs: 16-row ldmatrix.x4 blocks per D_V k-chunk

        tidx, _, _ = cute.arch.thread_idx()
        kv_block, q_head, batch = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        g_lane = lane // 4
        p_lane = lane % 4

        S_Q = q.shape[1]
        S_KV = k.shape[1]
        H_Q = q.shape[2]
        H_KV = k.shape[2]
        GROUP = H_Q // H_KV  # query heads per KV head (1 = plain MHA)
        S_Q_R = ceil_div(S_Q, ROW_ROUND) * ROW_ROUND
        kv_base = kv_block * KV_TILE
        qk_row_stride = H_Q * D_QK  # row stride of H_Q-headed D_QK-wide BSHD tensors (Q; also dk_ws, whose head axis is H_Q)
        v_row_stride = H_Q * D_V  # row stride of H_Q-headed D_V-wide BSHD tensors (dO; also dv_ws)

        # Per-batch actual lengths (Padding mask)
        seqlen_q = S_Q
        seqlen_kv = S_KV
        if cutlass.const_expr(self.seq_kv_lens_present):
            seqlen_kv = cute.math.max(cutlass.Int32(0), cute.math.min(seq_kv_lens[batch], cutlass.Int32(S_KV)))
        if cutlass.const_expr(self.seq_q_lens_present):
            seqlen_q = cute.math.max(cutlass.Int32(0), cute.math.min(seq_q_lens[batch], cutlass.Int32(S_Q)))

        lse_ptr = lse.iterator.raw_ptr()
        lse_batch_stride, lse_head_stride, lse_seq_stride = lse.stride
        lse_strided = (lse_batch_stride, lse_head_stride, lse_seq_stride) != (H_Q * S_Q, S_Q, 1)
        delta_ptr = delta.iterator.raw_ptr()
        if cutlass.const_expr(not self.det_2k):
            dq_accum_ptr = dq_accum.iterator.raw_ptr()
        dk_ws_ptr = dk_ws.iterator.raw_ptr()
        dv_ws_ptr = dv_ws.iterator.raw_ptr()
        if cutlass.const_expr(self.deterministic and not self.det_2k):
            dq_sem_ptr = dq_sem.iterator.raw_ptr()
        if cutlass.const_expr(self.bias_present):
            bias_ptr = bias.iterator.raw_ptr()
            if cutlass.const_expr(bias.shape[0] == 1):  # broadcast
                bias_hq_base = q_head * (S_Q * S_KV)
            else:
                bias_hq_base = (batch * H_Q + q_head) * (S_Q * S_KV)
        if cutlass.const_expr(self.dbias_present):
            dbias_ptr = dbias_accum.iterator.raw_ptr()

        PARTIAL_Q = (S_Q % Q_TILE) != 0
        PARTIAL_KV = (S_KV % KV_TILE) != 0
        # Skip the kv >= S_KV check when padding already masks kv >= seqlen_kv (<= S_KV).
        MASK_KV_GLOBAL = PARTIAL_KV and not self.seq_kv_lens_present
        # Fully-masked rows have LSE = -inf; flip to +inf so P = exp2(finite - inf) = 0, not NaN.
        # Any feature below can fully mask a row -- including a bias used as an
        # additive mask whose row is all -inf.
        FLIP_MASKED_LSE = (
            (self.is_causal and not self.causal_top_left)
            or self.window_size_left is not None
            or self.seq_kv_lens_present
            or self.seq_q_lens_present
            or self.bias_present
        )

        q_block_max = ceil_div(seqlen_q, Q_TILE)
        if cutlass.const_expr(self.is_causal or self.window_size_left is not None):
            if cutlass.const_expr(self.is_causal and not self.causal_top_left):
                # The diagonal anchors at the actual lengths.
                diag_off = seqlen_kv - seqlen_q
            else:
                diag_off = cutlass.Int32(0)
        if cutlass.const_expr(self.is_causal):
            # kv is visible to q when kv <= q + diag_off + right_slack
            q_block_min = cute.math.max(kv_base - diag_off - self.right_slack, cutlass.Int32(0)) // Q_TILE
        else:
            q_block_min = cutlass.Int32(0)
        if cutlass.const_expr(self.window_size_left is not None):
            # q <= k - diag_off + W
            last_q_row = kv_base + KV_TILE - 1 - diag_off + self.window_size_left
            rows_hi = cute.math.max(last_q_row + 1, cutlass.Int32(0))
            q_block_max = cute.math.min(q_block_max, ceil_div(rows_hi, Q_TILE))
        num_q_blocks = cute.math.max(q_block_max - q_block_min, cutlass.Int32(0))
        if cutlass.const_expr(self.seq_kv_lens_present):
            # Fully-padded KV tile: skip the loop; the epilogue writes zero dK/dV.
            if kv_base >= seqlen_kv:
                num_q_blocks = cutlass.Int32(0)

        smem = cutlass.Array(io_dtype, self.smem_elems, space=cutlass.AddressSpace.smem, alignment=128)
        sQ = smem  # Q_STAGES * Q_TILE * D_QK
        sdO = smem.subview(self.off_sdO)  # Q_TILE * D_V
        sK = smem.subview(self.off_sK)  # KV_TILE * D_QK
        sV = smem.subview(self.off_sV)  # KV_TILE * D_V (region max(KV_TILE * D_V, 2 * Q_TILE * KV_TILE))
        sdS = smem.subview(self.off_sdS)  # Q_TILE * KV_TILE (aliases sV)
        sP = smem.subview(self.off_sP)  # Q_TILE * KV_TILE
        tma_mbar = cutlass.Array(cutlass.Int64, 6, space=cutlass.AddressSpace.smem, alignment=8)
        k_mbar = tma_mbar
        v_mbar = tma_mbar.subview(1)
        q_full = tma_mbar.subview(2)  # Q_STAGES barriers
        do_full = tma_mbar.subview(4)  # DO_STAGES barriers

        if warp == self.load_warp_id:
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_k_desc.get_ptr())
                prims.prefetch_tensormap(tma_v_desc.get_ptr())
                prims.prefetch_tensormap(tma_q_desc.get_ptr())
                prims.prefetch_tensormap(tma_do_desc.get_ptr())
                prims.mbarrier_init(k_mbar, 1)
                prims.mbarrier_init(v_mbar, 1)
                prims.mbarrier_init(q_full, 1)
                if cutlass.const_expr(Q_STAGES == 2):
                    prims.mbarrier_init(q_full.subview(1), 1)
                prims.mbarrier_init(do_full, 1)
                if cutlass.const_expr(DO_STAGES == 2):
                    prims.mbarrier_init(do_full.subview(1), 1)
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)

        kv_head = q_head // GROUP
        if cutlass.const_expr(lse_strided):
            lse_base = batch * lse_batch_stride + q_head * lse_head_stride
        else:
            lse_base = (batch * H_Q + q_head) * S_Q
        delta_base = (batch * H_Q + q_head) * S_Q_R
        if cutlass.const_expr(self.deterministic and not self.det_2k):
            relay_sem = dq_sem_ptr + (batch * H_Q + q_head) * ceil_div(S_Q, Q_TILE)

        if warp == self.load_warp_id:
            prims.setmaxregister(24, prims.SetMaxRegisterAction.DECREASE)
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(v_mbar, KV_TILE * D_V * io_dtype.bytes)
                prims.mbarrier_arrive_expect_tx(k_mbar, KV_TILE * D_QK * io_dtype.bytes)
            self.load_tma_tile(sV, tma_v_desc, v_mbar, batch, kv_head, kv_base, rows=KV_TILE, cols=D_V)
            self.load_tma_tile(sK, tma_k_desc, k_mbar, batch, kv_head, kv_base, rows=KV_TILE, cols=D_QK)

            if num_q_blocks > 0:
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(q_full, Q_TILE * D_QK * io_dtype.bytes)
                self.load_tma_tile(sQ, tma_q_desc, q_full, batch, q_head, (q_block_max - 1) * Q_TILE, rows=Q_TILE, cols=D_QK)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(do_full, Q_TILE * D_V * io_dtype.bytes)
                self.load_tma_tile(
                    sdO,
                    tma_do_desc,
                    do_full,
                    batch,
                    q_head,
                    (q_block_max - 1) * Q_TILE,
                    rows=Q_TILE,
                    cols=D_V,
                )
            while not prims.mbarrier_try_wait_parity(v_mbar, cutlass.Int32(0)):
                pass
            while not prims.mbarrier_try_wait_parity(k_mbar, cutlass.Int32(0)):
                pass
            for load_step in cutlass.range(num_q_blocks, unroll=1):
                if cutlass.const_expr(Q_STAGES == 2):
                    q_stage = load_step & cutlass.Int32(1)
                    q_phase = (load_step // 2) & cutlass.Int32(1)
                else:
                    q_stage = cutlass.Int32(0)
                    q_phase = load_step & cutlass.Int32(1)
                while not prims.mbarrier_try_wait_parity(q_full.subview(q_stage), q_phase):
                    pass
                if cutlass.const_expr(DO_STAGES == 2):
                    do_stage = load_step & cutlass.Int32(1)
                    do_phase = (load_step // 2) & cutlass.Int32(1)
                else:
                    do_stage = cutlass.Int32(0)
                    do_phase = load_step & cutlass.Int32(1)
                while not prims.mbarrier_try_wait_parity(do_full.subview(do_stage), do_phase):
                    pass
                # Loop-top: stage load_step ready AND load_step-1 consumed.
                cute.arch.barrier(barrier_id=3, number_of_threads=288)
                next_q_block = q_block_max - 2 - load_step
                if cutlass.const_expr(Q_STAGES == 2):
                    # prefetch the next iteration's Q into the other buffer
                    if load_step + 1 < num_q_blocks:
                        next_stage = (load_step + 1) & cutlass.Int32(1)
                        next_q_full = q_full.subview(next_stage)
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(next_q_full, Q_TILE * D_QK * io_dtype.bytes)
                        self.load_tma_tile(
                            sQ.subview(next_stage * Q_TILE * D_QK),
                            tma_q_desc,
                            next_q_full,
                            batch,
                            q_head,
                            next_q_block * Q_TILE,
                            rows=Q_TILE,
                            cols=D_QK,
                        )
                if cutlass.const_expr(DO_STAGES == 2):
                    # prefetch the next iteration's dO into the other buffer
                    if load_step + 1 < num_q_blocks:
                        next_do = (load_step + 1) & cutlass.Int32(1)
                        next_do_full = do_full.subview(next_do)
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(next_do_full, Q_TILE * D_V * io_dtype.bytes)
                        self.load_tma_tile(
                            sdO.subview(next_do * Q_TILE * D_V), tma_do_desc, next_do_full, batch, q_head, next_q_block * Q_TILE, rows=Q_TILE, cols=D_V
                        )
                if cutlass.const_expr(DO_STAGES == 1):
                    # Post-GEMM3 (dV += P^T*dO): every consumer is done with sdO.
                    cute.arch.barrier(barrier_id=4, number_of_threads=288)
                    if load_step + 1 < num_q_blocks:
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(do_full, Q_TILE * D_V * io_dtype.bytes)
                        self.load_tma_tile(sdO, tma_do_desc, do_full, batch, q_head, next_q_block * Q_TILE, rows=Q_TILE, cols=D_V)
                if cutlass.const_expr(Q_STAGES == 1):
                    # Post-GEMM5 (dK += dS^T*Q): every consumer is done with sQ.
                    cute.arch.barrier(barrier_id=5, number_of_threads=288)
                    if load_step + 1 < num_q_blocks:
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(q_full, Q_TILE * D_QK * io_dtype.bytes)
                        self.load_tma_tile(sQ, tma_q_desc, q_full, batch, q_head, next_q_block * Q_TILE, rows=Q_TILE, cols=D_QK)

        elif warp < self.load_warp_id:
            prims.setmaxregister(240, prims.SetMaxRegisterAction.INCREASE)
            # LSE for the first (highest) q-block: per-thread direct loads at
            # this thread's C-fragment rows
            math_warp = warp
            math_tidx = tidx
            q_block = q_block_max - 1
            if cutlass.const_expr(self.window_size_left is not None or self.seq_q_lens_present):
                # q_block_max can be 0 in bottom-right SWA, or seq_len_q[b] == 0
                q_block = cute.math.max(q_block, cutlass.Int32(0))
            wm_sdp = math_warp % WM_SDP
            wn_sdp = math_warp // WM_SDP
            lse_r = cutlass.Array(cutlass.Float32, 2 * SDP_ROW_BLOCKS)
            delta_r = cutlass.Array(cutlass.Float32, 2 * SDP_ROW_BLOCKS)
            for row_blk in cutlass.range_constexpr(SDP_ROW_BLOCKS):
                for hf in cutlass.range_constexpr(2):
                    r_loc = wm_sdp * 16 + row_blk * 16 * WM_SDP + g_lane + hf * 8
                    r_abs = q_block * Q_TILE + r_loc
                    if cutlass.const_expr(PARTIAL_Q):
                        r_cl = cute.math.min(r_abs, S_Q - 1)
                        if cutlass.const_expr(lse_strided):
                            val = (lse_ptr + lse_base + r_cl * lse_seq_stride).load()
                        else:
                            val = (lse_ptr + lse_base + r_cl).load()
                        inf = cutlass.Float32(float("inf"))
                        # branchless (r_abs < S_Q) ? 1 : 0 via arith.select
                        ok32 = cutlass.Int32(
                            arith.select(
                                (r_abs < S_Q).ir_value(),
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
                    lse_r[row_blk * 2 + hf] = val * cutlass.Float32(_LOG2E)

            while not prims.mbarrier_try_wait_parity(v_mbar, cutlass.Int32(0)):
                pass
            while not prims.mbarrier_try_wait_parity(k_mbar, cutlass.Int32(0)):
                pass

            # V -> registers.
            v_persist = cutlass.Array(cutlass.Int32, D_V_CHUNKS * V_ROW_BLOCKS * 4, alignment=16)
            for k_chunk in cutlass.range_constexpr(D_V_CHUNKS):
                for v_row_blk in cutlass.range_constexpr(V_ROW_BLOCKS):
                    n_frag = v_row_blk * 2
                    row = wn_sdp * SDP_COLS + (n_frag + lane // 16) * 8 + lane % 8
                    col = k_chunk * 16 + ((lane // 8) % 2) * 8
                    vf = prims.ldmatrix(
                        tile_ptr(sV, row, col, chunk_elems=CHUNK_ELEMS, rows=KV_TILE),
                        4,
                        prims.MMALayout.ROW,
                    )
                    v_off = (k_chunk * V_ROW_BLOCKS + v_row_blk) * 4
                    v_persist[v_off + 0] = vf[0]
                    v_persist[v_off + 1] = vf[1]
                    v_persist[v_off + 2] = vf[2]
                    v_persist[v_off + 3] = vf[3]
            cute.arch.barrier(barrier_id=1, number_of_threads=256)

            # dK/dV accumulators.
            wm_dkv = math_warp % WM_DKV
            wn_dkv = math_warp // WM_DKV
            acc_dk = cutlass.Array(cutlass.Float32, DKV_ROW_BLOCKS * DK_COL_FRAGS * 4, alignment=16)
            acc_dv = cutlass.Array(cutlass.Float32, DKV_ROW_BLOCKS * DV_COL_FRAGS * 4, alignment=16)
            for i in cutlass.range_constexpr(DKV_ROW_BLOCKS * DK_COL_FRAGS * 4):
                acc_dk[i] = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(DKV_ROW_BLOCKS * DV_COL_FRAGS * 4):
                acc_dv[i] = cutlass.Float32(0.0)

            wm_dq = math_warp % WM_DQ
            wn_dq = math_warp // WM_DQ

            acc_s = cutlass.Array(cutlass.Float32, SDP_ROW_BLOCKS * SDP_COL_FRAGS * 4, alignment=16)
            acc_dp = cutlass.Array(cutlass.Float32, SDP_ROW_BLOCKS * SDP_COL_FRAGS * 4, alignment=16)
            acc_dq = cutlass.Array(cutlass.Float32, DQ_ROW_BLOCKS * DQ_COL_FRAGS * 4, alignment=16)

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            # ---- main loop: q_block descending --------------------------------
            math_step = cutlass.Int32(0)
            while math_step < num_q_blocks:
                q_block = q_block_max - 1 - math_step
                if cutlass.const_expr(Q_STAGES == 2):
                    stage = math_step & cutlass.Int32(1)
                else:
                    stage = cutlass.Int32(0)
                sQ_stage = sQ.subview(stage * Q_TILE * D_QK)
                if cutlass.const_expr(DO_STAGES == 2):
                    sdO_stage = sdO.subview((math_step & cutlass.Int32(1)) * Q_TILE * D_V)
                else:
                    sdO_stage = sdO
                q_row0 = q_block * Q_TILE

                if cutlass.const_expr(self.det_2k):
                    # The previous tile's dS TMA store must finish reading sdS
                    # before this iteration's pack overwrites it.
                    if warp == 0:
                        prims.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(barrier_id=3, number_of_threads=288)

                # dP_sum per-thread loads (delta buffer is 128-rounded).
                # det_2k prefetches the next block's delta alongside its LSE
                # reload, so only the first iteration loads here.
                if (not cutlass.const_expr(self.det_2k)) or math_step == 0:
                    for row_blk in cutlass.range_constexpr(SDP_ROW_BLOCKS):
                        for hf in cutlass.range_constexpr(2):
                            r_loc = wm_sdp * 16 + row_blk * 16 * WM_SDP + g_lane + hf * 8
                            delta_r[row_blk * 2 + hf] = (delta_ptr + delta_base + q_row0 + r_loc).load()

                # GEMM 1: acc_s = Q @ K^T.
                for i in cutlass.range_constexpr(SDP_ROW_BLOCKS * SDP_COL_FRAGS * 4):
                    acc_s[i] = cutlass.Float32(0.0)
                for k_chunk in cutlass.range_constexpr(D_QK_CHUNKS):
                    a_frag = []
                    for row_blk in cutlass.range_constexpr(SDP_ROW_BLOCKS):
                        qf = load_a_frag(
                            sQ_stage,
                            k_chunk,
                            wm_sdp * 16 + row_blk * 16 * WM_SDP,
                            lane,
                            rows=Q_TILE,
                            chunk_elems=CHUNK_ELEMS,
                        )
                        a_frag = a_frag + [qf[0], qf[1], qf[2], qf[3]]
                    mma_bstream(
                        acc_s,
                        a_frag,
                        sK,
                        b_k_step=k_chunk,
                        M=16 * SDP_ROW_BLOCKS,
                        N=SDP_COLS,
                        b_trans=False,
                        b_rows=KV_TILE,
                        b_chunk_elems=CHUNK_ELEMS,
                        lane=lane,
                        ab_dtype=io_dtype,
                        row_base=wn_sdp * SDP_COLS,
                    )

                # Mask + softmax (scores -> P, unscaled by attn_scale) and the
                # P store to smem.
                if cutlass.const_expr(self.is_causal):
                    do_mask_causal = (q_block * Q_TILE) < (kv_base + KV_TILE - diag_off - self.right_slack)
                if cutlass.const_expr(self.window_size_left is not None):
                    do_mask_window = kv_base < (q_block * Q_TILE + Q_TILE - 1 + diag_off - self.window_size_left)
                if cutlass.const_expr(self.seq_kv_lens_present):
                    do_mask_pad = (kv_base + KV_TILE) > seqlen_kv
                neg_inf = cutlass.Float32(float("-inf"))
                for row_blk in cutlass.range_constexpr(SDP_ROW_BLOCKS):
                    for col_frag in cutlass.range_constexpr(SDP_COL_FRAGS):
                        off = (row_blk * SDP_COL_FRAGS + col_frag) * 4
                        kv_c0 = wn_sdp * SDP_COLS + col_frag * 8 + 2 * p_lane
                        kv_a0 = kv_base + kv_c0
                        kv_a1 = kv_a0 + 1
                        r0 = q_row0 + wm_sdp * 16 + row_blk * 16 * WM_SDP + g_lane
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
                            if kv_a0 >= S_KV:
                                s0 = neg_inf
                                s2 = neg_inf
                            if kv_a1 >= S_KV:
                                s1 = neg_inf
                                s3 = neg_inf
                        lse0 = lse_r[row_blk * 2 + 0]
                        lse8 = lse_r[row_blk * 2 + 1]
                        if cutlass.const_expr(self.bias_present):
                            if cutlass.const_expr(PARTIAL_Q):
                                bias_r0 = cute.math.min(r0, cutlass.Int32(S_Q - 1))
                                bias_r8 = cute.math.min(r8, cutlass.Int32(S_Q - 1))
                            else:
                                bias_r0 = r0
                                bias_r8 = r8
                            if cutlass.const_expr(PARTIAL_KV):
                                bias_k0 = cute.math.min(kv_a0, cutlass.Int32(S_KV - 1))
                                bias_k1 = cute.math.min(kv_a1, cutlass.Int32(S_KV - 1))
                            else:
                                bias_k0 = kv_a0
                                bias_k1 = kv_a1
                            bias_row0 = bias_hq_base + bias_r0 * S_KV
                            bias_row8 = bias_hq_base + bias_r8 * S_KV
                            b00 = (bias_ptr + bias_row0 + bias_k0).load().to(cutlass.Float32) * cutlass.Float32(_LOG2E)
                            b01 = (bias_ptr + bias_row0 + bias_k1).load().to(cutlass.Float32) * cutlass.Float32(_LOG2E)
                            b80 = (bias_ptr + bias_row8 + bias_k0).load().to(cutlass.Float32) * cutlass.Float32(_LOG2E)
                            b81 = (bias_ptr + bias_row8 + bias_k1).load().to(cutlass.Float32) * cutlass.Float32(_LOG2E)
                            p0 = cute.math.exp2(s0 * softmax_scale_log2 + b00 - lse0, fastmath=True)
                            p1 = cute.math.exp2(s1 * softmax_scale_log2 + b01 - lse0, fastmath=True)
                            p2 = cute.math.exp2(s2 * softmax_scale_log2 + b80 - lse8, fastmath=True)
                            p3 = cute.math.exp2(s3 * softmax_scale_log2 + b81 - lse8, fastmath=True)
                        else:
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
                        pr0 = wm_sdp * 16 + row_blk * 16 * WM_SDP + g_lane
                        pr8 = pr0 + 8
                        sw0 = tile_ptr(sP, pr0, kv_c0, chunk_elems=DS_CHUNK_ELEMS, rows=Q_TILE)
                        sw8 = tile_ptr(sP, pr8, kv_c0, chunk_elems=DS_CHUNK_ELEMS, rows=Q_TILE)
                        sw0.store(pack_half2(p0, p1, io_dtype), alignment=4)
                        sw8.store(pack_half2(p2, p3, io_dtype), alignment=4)

                # GEMM 2: acc_dp = dO @ V^T (V in registers).
                for i in cutlass.range_constexpr(SDP_ROW_BLOCKS * SDP_COL_FRAGS * 4):
                    acc_dp[i] = cutlass.Float32(0.0)
                for k_chunk in cutlass.range_constexpr(D_V_CHUNKS):
                    a_frag = []
                    for row_blk in cutlass.range_constexpr(SDP_ROW_BLOCKS):
                        dof = load_a_frag(
                            sdO_stage,
                            k_chunk,
                            wm_sdp * 16 + row_blk * 16 * WM_SDP,
                            lane,
                            rows=Q_TILE,
                            chunk_elems=CHUNK_ELEMS,
                        )
                        a_frag = a_frag + [dof[0], dof[1], dof[2], dof[3]]
                    mma_abregs(
                        acc_dp,
                        a_frag,
                        v_persist,
                        b_k_step=k_chunk,
                        M=16 * SDP_ROW_BLOCKS,
                        N=SDP_COLS,
                        ab_dtype=io_dtype,
                    )

                # dS = P * (dP - dP_sum)
                for row_blk in cutlass.range_constexpr(SDP_ROW_BLOCKS):
                    for col_frag in cutlass.range_constexpr(SDP_COL_FRAGS):
                        off = (row_blk * SDP_COL_FRAGS + col_frag) * 4
                        dd0 = delta_r[row_blk * 2 + 0]
                        dd8 = delta_r[row_blk * 2 + 1]
                        acc_dp[off + 0] = acc_s[off + 0] * (acc_dp[off + 0] - dd0)
                        acc_dp[off + 1] = acc_s[off + 1] * (acc_dp[off + 1] - dd0)
                        acc_dp[off + 2] = acc_s[off + 2] * (acc_dp[off + 2] - dd8)
                        acc_dp[off + 3] = acc_s[off + 3] * (acc_dp[off + 3] - dd8)

                # dS -> fp16 -> sdS.
                for row_blk in cutlass.range_constexpr(SDP_ROW_BLOCKS):
                    for col_frag in cutlass.range_constexpr(SDP_COL_FRAGS):
                        off = (row_blk * SDP_COL_FRAGS + col_frag) * 4
                        kv_c0 = wn_sdp * SDP_COLS + col_frag * 8 + 2 * p_lane
                        pr0 = wm_sdp * 16 + row_blk * 16 * WM_SDP + g_lane
                        pr8 = pr0 + 8
                        sw0 = tile_ptr(sdS, pr0, kv_c0, chunk_elems=DS_CHUNK_ELEMS, rows=Q_TILE)
                        sw8 = tile_ptr(sdS, pr8, kv_c0, chunk_elems=DS_CHUNK_ELEMS, rows=Q_TILE)
                        sw0.store(pack_half2(acc_dp[off + 0], acc_dp[off + 1], io_dtype), alignment=4)
                        sw8.store(pack_half2(acc_dp[off + 2], acc_dp[off + 3], io_dtype), alignment=4)
                        if cutlass.const_expr(self.dbias_present):
                            db_q0 = q_row0 + pr0
                            db_k0 = kv_base + kv_c0
                            db_row0 = bias_hq_base + db_q0 * S_KV + db_k0
                            db_row8 = db_row0 + 8 * S_KV
                            db_q0_ok = True
                            db_q8_ok = True
                            if cutlass.const_expr(PARTIAL_Q):
                                db_q0_ok = db_q0 < S_Q
                                db_q8_ok = db_q0 + 8 < S_Q
                            db_k0_ok = True
                            db_k1_ok = True
                            if cutlass.const_expr(PARTIAL_KV):
                                db_k0_ok = db_k0 < S_KV
                                db_k1_ok = db_k0 + 1 < S_KV
                            if db_q0_ok:
                                if db_k0_ok:
                                    _red_add_f32(dbias_ptr + db_row0, acc_dp[off + 0])
                                if db_k1_ok:
                                    _red_add_f32(dbias_ptr + db_row0 + 1, acc_dp[off + 1])
                            if db_q8_ok:
                                if db_k0_ok:
                                    _red_add_f32(dbias_ptr + db_row8, acc_dp[off + 2])
                                if db_k1_ok:
                                    _red_add_f32(dbias_ptr + db_row8 + 1, acc_dp[off + 3])
                cute.arch.barrier(barrier_id=1, number_of_threads=256)

                if cutlass.const_expr(self.det_2k):
                    # tma store dS
                    if warp == 0:
                        cute.arch.fence_proxy("async.shared", space="cta")
                        if prims.elect_sync():
                            for chunk in cutlass.range_constexpr(KV_TILE // DS_CHUNK_ELEMS):
                                prims.cp_async_bulk_tensor_global_shared_cta(
                                    tma_dsws_desc.get_ptr(),
                                    sdS.subview(chunk * (Q_TILE * DS_CHUNK_ELEMS)),
                                    (kv_base + chunk * DS_CHUNK_ELEMS, q_row0, q_head, batch),
                                )
                            prims.cp_async_bulk_commit_group()

                # GEMM 3: acc_dv += P^T @ dO.
                for k_chunk in cutlass.range_constexpr(Q_CHUNKS):
                    a_frag = []
                    for row_blk in cutlass.range_constexpr(DKV_ROW_BLOCKS):
                        pf = load_a_frag_transposed(
                            sP,
                            k_chunk,
                            wm_dkv * 16 + row_blk * 16 * WM_DKV,
                            lane,
                            rows=Q_TILE,
                            chunk_elems=DS_CHUNK_ELEMS,
                        )
                        a_frag = a_frag + [pf[0], pf[2], pf[1], pf[3]]
                    mma_bstream(
                        acc_dv,
                        a_frag,
                        sdO_stage,
                        b_k_step=k_chunk,
                        M=16 * DKV_ROW_BLOCKS,
                        N=DV_COLS,
                        b_trans=True,
                        b_rows=Q_TILE,
                        b_chunk_elems=CHUNK_ELEMS,
                        lane=lane,
                        ab_dtype=io_dtype,
                        col_base=wn_dkv * DV_COLS,
                    )

                if cutlass.const_expr(DO_STAGES == 1):
                    # GEMM3 is the final dO consumer; this rendezvous lets
                    # the producer refill the single dO buffer.
                    cute.arch.barrier(barrier_id=4, number_of_threads=288)

                # Deterministic relay turn for this q-tile: the dQ adds of a
                # (batch, head, q-tile) happen in ascending kv-tile order.
                if cutlass.const_expr(self.deterministic and not self.det_2k):
                    if cutlass.const_expr(self.window_size_left is not None):
                        # SWA clamps q_block_max, so a q-tile's visitors start
                        # at kv tile n_lo = max((q_block*Q_TILE + diag_off - W) // KV_TILE, 0)
                        # (inverts the clamp); count turns from there.
                        relay_turn = kv_block - cute.math.max(
                            (q_block * Q_TILE + diag_off - self.window_size_left) // KV_TILE,
                            cutlass.Int32(0),
                        )
                    else:
                        relay_turn = kv_block

                if cutlass.const_expr(self.det_2k):
                    # dS is already in the workspace; the dQ section lives in
                    # dq2k. Only GEMM 5 remains (sQ's last reader).
                    _bwd_gemm5_dk(
                        acc_dk,
                        sdS,
                        sQ_stage,
                        wm_dkv,
                        wn_dkv,
                        lane,
                        Q_CHUNKS=Q_CHUNKS,
                        DKV_ROW_BLOCKS=DKV_ROW_BLOCKS,
                        WM_DKV=WM_DKV,
                        Q_TILE=Q_TILE,
                        DS_CHUNK_ELEMS=DS_CHUNK_ELEMS,
                        CHUNK_ELEMS=CHUNK_ELEMS,
                        DK_COLS=DK_COLS,
                        io_dtype=io_dtype,
                    )
                    if cutlass.const_expr(Q_STAGES == 1):
                        cute.arch.barrier(barrier_id=5, number_of_threads=288)
                    # Reload LSE and delta for the next (lower) m-block behind
                    # the dS workspace stores draining (the loop-top delta
                    # load would expose its gmem latency).
                    if math_step + 1 < num_q_blocks:
                        nq0 = (q_block - 1) * Q_TILE
                        for row_blk in cutlass.range_constexpr(SDP_ROW_BLOCKS):
                            for hf in cutlass.range_constexpr(2):
                                r_loc = wm_sdp * 16 + row_blk * 16 * WM_SDP + g_lane + hf * 8
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
                                lse_r[row_blk * 2 + hf] = val * cutlass.Float32(_LOG2E)
                                delta_r[row_blk * 2 + hf] = (delta_ptr + delta_base + nq0 + r_loc).load()
                elif cutlass.const_expr(Q_STAGES == 1):
                    # Single Q buffer: GEMM5 (sQ's last reader) first, so the
                    # Q refill hides behind GEMM4 + the dQ scatter. (2-stage
                    # keeps GEMM4-first: atomics drain during GEMM5 instead.)
                    _bwd_gemm5_dk(
                        acc_dk,
                        sdS,
                        sQ_stage,
                        wm_dkv,
                        wn_dkv,
                        lane,
                        Q_CHUNKS=Q_CHUNKS,
                        DKV_ROW_BLOCKS=DKV_ROW_BLOCKS,
                        WM_DKV=WM_DKV,
                        Q_TILE=Q_TILE,
                        DS_CHUNK_ELEMS=DS_CHUNK_ELEMS,
                        CHUNK_ELEMS=CHUNK_ELEMS,
                        DK_COLS=DK_COLS,
                        io_dtype=io_dtype,
                    )
                    cute.arch.barrier(barrier_id=5, number_of_threads=288)
                    _bwd_gemm4_dq(
                        acc_dq,
                        sdS,
                        sK,
                        wm_dq,
                        wn_dq,
                        lane,
                        DQ_ROW_BLOCKS=DQ_ROW_BLOCKS,
                        DQ_COL_FRAGS=DQ_COL_FRAGS,
                        KV_CHUNKS=KV_CHUNKS,
                        WM_DQ=WM_DQ,
                        Q_TILE=Q_TILE,
                        KV_TILE=KV_TILE,
                        DS_CHUNK_ELEMS=DS_CHUNK_ELEMS,
                        CHUNK_ELEMS=CHUNK_ELEMS,
                        DQ_COLS=DQ_COLS,
                        io_dtype=io_dtype,
                    )
                    # Reload LSE for the next (lower) m-block: overlaps the
                    # in-flight Q refill issued at barrier 5.
                    if math_step + 1 < num_q_blocks:
                        nq0 = (q_block - 1) * Q_TILE
                        for row_blk in cutlass.range_constexpr(SDP_ROW_BLOCKS):
                            for hf in cutlass.range_constexpr(2):
                                r_loc = wm_sdp * 16 + row_blk * 16 * WM_SDP + g_lane + hf * 8
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
                                lse_r[row_blk * 2 + hf] = val * cutlass.Float32(_LOG2E)
                    dq_accum_base = ((batch * S_Q_R + q_row0) * H_Q + q_head) * D_QK
                    if cutlass.const_expr(self.deterministic):
                        _bwd_relay_wait(relay_sem, q_block, relay_turn, warp)
                    _bwd_dq_scatter(
                        acc_dq, dq_accum_ptr, dq_accum_base, math_tidx, H_Q, DQ_ROW_BLOCKS=DQ_ROW_BLOCKS, DQ_COL_FRAGS=DQ_COL_FRAGS, Q_TILE=Q_TILE, D_QK=D_QK
                    )
                    if cutlass.const_expr(self.deterministic):
                        _bwd_relay_release(relay_sem, q_block, relay_turn, warp)
                else:
                    _bwd_gemm4_dq(
                        acc_dq,
                        sdS,
                        sK,
                        wm_dq,
                        wn_dq,
                        lane,
                        DQ_ROW_BLOCKS=DQ_ROW_BLOCKS,
                        DQ_COL_FRAGS=DQ_COL_FRAGS,
                        KV_CHUNKS=KV_CHUNKS,
                        WM_DQ=WM_DQ,
                        Q_TILE=Q_TILE,
                        KV_TILE=KV_TILE,
                        DS_CHUNK_ELEMS=DS_CHUNK_ELEMS,
                        CHUNK_ELEMS=CHUNK_ELEMS,
                        DQ_COLS=DQ_COLS,
                        io_dtype=io_dtype,
                    )
                    # Reload LSE for the next (lower) m-block (develop-exact
                    # position: between GEMM4 and the scatter, hiding the
                    # global-load latency behind the atomic drain + GEMM5).
                    if math_step + 1 < num_q_blocks:
                        nq0 = (q_block - 1) * Q_TILE
                        for row_blk in cutlass.range_constexpr(SDP_ROW_BLOCKS):
                            for hf in cutlass.range_constexpr(2):
                                r_loc = wm_sdp * 16 + row_blk * 16 * WM_SDP + g_lane + hf * 8
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
                                lse_r[row_blk * 2 + hf] = val * cutlass.Float32(_LOG2E)
                    dq_accum_base = ((batch * S_Q_R + q_row0) * H_Q + q_head) * D_QK
                    if cutlass.const_expr(self.deterministic):
                        _bwd_relay_wait(relay_sem, q_block, relay_turn, warp)
                    _bwd_dq_scatter(
                        acc_dq, dq_accum_ptr, dq_accum_base, math_tidx, H_Q, DQ_ROW_BLOCKS=DQ_ROW_BLOCKS, DQ_COL_FRAGS=DQ_COL_FRAGS, Q_TILE=Q_TILE, D_QK=D_QK
                    )
                    if cutlass.const_expr(self.deterministic):
                        _bwd_relay_release(relay_sem, q_block, relay_turn, warp)
                    _bwd_gemm5_dk(
                        acc_dk,
                        sdS,
                        sQ_stage,
                        wm_dkv,
                        wn_dkv,
                        lane,
                        Q_CHUNKS=Q_CHUNKS,
                        DKV_ROW_BLOCKS=DKV_ROW_BLOCKS,
                        WM_DKV=WM_DKV,
                        Q_TILE=Q_TILE,
                        DS_CHUNK_ELEMS=DS_CHUNK_ELEMS,
                        CHUNK_ELEMS=CHUNK_ELEMS,
                        DK_COLS=DK_COLS,
                        io_dtype=io_dtype,
                    )

                math_step += 1

            if cutlass.const_expr(self.det_2k):
                if warp == 0:
                    prims.cp_async_bulk_wait_group(0)
                cute.arch.barrier(barrier_id=6, number_of_threads=256)
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            # epilogue: dK/dV through smem (sdK aliases sK, sdV aliases sV).
            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            sdK = sK
            sdV = sV
            for row_blk in cutlass.range_constexpr(DKV_ROW_BLOCKS):
                for col_frag in cutlass.range_constexpr(max(DK_COL_FRAGS, DV_COL_FRAGS)):
                    r0 = wm_dkv * 16 + row_blk * 16 * WM_DKV + g_lane
                    r8 = r0 + 8
                    if cutlass.const_expr(col_frag < DK_COL_FRAGS):
                        off = (row_blk * DK_COL_FRAGS + col_frag) * 4
                        c0 = wn_dkv * DK_COLS + col_frag * 8 + 2 * p_lane
                        dk0 = acc_dk[off + 0] * attn_scale
                        dk1 = acc_dk[off + 1] * attn_scale
                        dk2 = acc_dk[off + 2] * attn_scale
                        dk3 = acc_dk[off + 3] * attn_scale
                        tile_ptr(sdK, r0, c0, chunk_elems=CHUNK_ELEMS, rows=KV_TILE).store(pack_half2(dk0, dk1, io_dtype), alignment=4)
                        tile_ptr(sdK, r8, c0, chunk_elems=CHUNK_ELEMS, rows=KV_TILE).store(pack_half2(dk2, dk3, io_dtype), alignment=4)
                    if cutlass.const_expr(col_frag < DV_COL_FRAGS):
                        off_v = (row_blk * DV_COL_FRAGS + col_frag) * 4
                        c0_v = wn_dkv * DV_COLS + col_frag * 8 + 2 * p_lane
                        tile_ptr(sdV, r0, c0_v, chunk_elems=CHUNK_ELEMS, rows=KV_TILE).store(
                            pack_half2(acc_dv[off_v + 0], acc_dv[off_v + 1], io_dtype),
                            alignment=4,
                        )
                        tile_ptr(sdV, r8, c0_v, chunk_elems=CHUNK_ELEMS, rows=KV_TILE).store(
                            pack_half2(acc_dv[off_v + 2], acc_dv[off_v + 3], io_dtype),
                            alignment=4,
                        )
            cute.arch.barrier(barrier_id=2, number_of_threads=256)

            # smem -> gmem. dk_ws/dv_ws rows are H_Q-headed: dk/dv themselves
            # when MHA (H_Q == H_KV and q_head == kv_head), one slot per q head
            # under GQA — the same addressing covers both.
            dk_batch_stride, dk_seq_stride, dk_head_stride, _ = dk_ws.stride
            dv_batch_stride, dv_seq_stride, dv_head_stride, _ = dv_ws.stride
            dkv_strided = (dk_batch_stride, dk_seq_stride, dk_head_stride) != (S_KV * qk_row_stride, qk_row_stride, D_QK) or (
                dv_batch_stride,
                dv_seq_stride,
                dv_head_stride,
            ) != (
                S_KV * v_row_stride,
                v_row_stride,
                D_V,
            )
            if cutlass.const_expr(D_QK == D_V and not dkv_strided and dk_ws.shape[3] == D_QK and dv_ws.shape[3] == D_V):
                chunks_per_row = D_QK // _COPY_ELEMS
                total = KV_TILE * chunks_per_row
                # workspace's head-dim base offset
                whd_base = (batch * S_KV + kv_base) * qk_row_stride + q_head * D_QK
                for i in cutlass.range_constexpr(total // 256):
                    chunk = i * 256 + math_tidx
                    row = chunk // chunks_per_row
                    col = (chunk % chunks_per_row) * _COPY_ELEMS
                    if (not cutlass.const_expr(PARTIAL_KV)) or (kv_base + row < S_KV):
                        w_off = whd_base + row * qk_row_stride + col
                        copy16_smem_to_gmem(tile_ptr(sdK, row, col, chunk_elems=CHUNK_ELEMS, rows=KV_TILE), dk_ws_ptr + w_off)
                        copy16_smem_to_gmem(tile_ptr(sdV, row, col, chunk_elems=CHUNK_ELEMS, rows=KV_TILE), dv_ws_ptr + w_off)
            else:
                # D_QK != D_V, strided, or enveloped
                k_chunks_per_row = D_QK // _COPY_ELEMS
                base_k = batch * dk_batch_stride + kv_base * dk_seq_stride + q_head * dk_head_stride
                for i in cutlass.range_constexpr(KV_TILE * k_chunks_per_row // 256):
                    chunk = i * 256 + math_tidx
                    row = chunk // k_chunks_per_row
                    col = (chunk % k_chunks_per_row) * _COPY_ELEMS
                    if (not cutlass.const_expr(PARTIAL_KV)) or (kv_base + row < S_KV):
                        if cutlass.const_expr(self.qk_envelope):
                            # dK is only d_qk_orig wide; the pad columns are zero anyway.
                            if col < self.d_qk_orig:
                                copy16_smem_to_gmem(
                                    tile_ptr(sdK, row, col, chunk_elems=CHUNK_ELEMS, rows=KV_TILE), dk_ws_ptr + base_k + row * dk_seq_stride + col
                                )
                        else:
                            copy16_smem_to_gmem(tile_ptr(sdK, row, col, chunk_elems=CHUNK_ELEMS, rows=KV_TILE), dk_ws_ptr + base_k + row * dk_seq_stride + col)
                v_chunks_per_row = D_V // _COPY_ELEMS
                base_v = batch * dv_batch_stride + kv_base * dv_seq_stride + q_head * dv_head_stride
                for i in cutlass.range_constexpr(KV_TILE * v_chunks_per_row // 256):
                    chunk = i * 256 + math_tidx
                    row = chunk // v_chunks_per_row
                    col = (chunk % v_chunks_per_row) * _COPY_ELEMS
                    if (not cutlass.const_expr(PARTIAL_KV)) or (kv_base + row < S_KV):
                        if cutlass.const_expr(self.v_envelope):
                            if col < self.d_v_orig:
                                copy16_smem_to_gmem(
                                    tile_ptr(sdV, row, col, chunk_elems=CHUNK_ELEMS, rows=KV_TILE), dv_ws_ptr + base_v + row * dv_seq_stride + col
                                )
                        else:
                            copy16_smem_to_gmem(tile_ptr(sdV, row, col, chunk_elems=CHUNK_ELEMS, rows=KV_TILE), dv_ws_ptr + base_v + row * dv_seq_stride + col)
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
        dq_accum: Optional[cute.Tensor],
        dq_sem: Optional[cute.Tensor],
        ds_ws: Optional[cute.Tensor],
        dk_ws: cute.Tensor,
        dv_ws: cute.Tensor,
        seq_q_lens: Optional[cute.Tensor],
        seq_kv_lens: Optional[cute.Tensor],
        bias: Optional[cute.Tensor],
        dbias_accum: Optional[cute.Tensor],
        softmax_scale_log2: cutlass.Float32,
        attn_scale: cutlass.Float32,
        stream: cuda_driver.CUstream,
    ) -> None:
        box_kv = (1, self.kv_tile, 1, self.chunk_elems)
        box_q = (1, self.q_tile, 1, self.chunk_elems)
        tma_q_desc = cuda.create_tensor_map_tiled_from_view(q, box_dims=box_q, stride_order=(3, 2, 1, 0), swizzle=self.tma_swizzle)
        tma_k_desc = cuda.create_tensor_map_tiled_from_view(k, box_dims=box_kv, stride_order=(3, 2, 1, 0), swizzle=self.tma_swizzle)
        tma_v_desc = cuda.create_tensor_map_tiled_from_view(v, box_dims=box_kv, stride_order=(3, 2, 1, 0), swizzle=self.tma_swizzle)
        tma_do_desc = cuda.create_tensor_map_tiled_from_view(do, box_dims=box_q, stride_order=(3, 2, 1, 0), swizzle=self.tma_swizzle)
        tma_dsws_desc = None
        if cutlass.const_expr(ds_ws is not None):
            # dS workspace store map: one (q_tile x 64) chunk_elems per issue, same swizzle as dq2k's load side.
            box_ds = (1, 1, self.q_tile, 64)
            tma_dsws_desc = cuda.create_tensor_map_tiled_from_view(ds_ws, box_dims=box_ds, stride_order=(3, 2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
        kv_blocks = ceil_div(k.shape[1], self.kv_tile)
        self.kernel(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq_accum,
            dq_sem,
            ds_ws,
            dk_ws,
            dv_ws,
            seq_q_lens,
            seq_kv_lens,
            bias,
            dbias_accum,
            tma_q_desc,
            tma_k_desc,
            tma_v_desc,
            tma_do_desc,
            softmax_scale_log2,
            attn_scale,
            tma_dsws_desc,
        ).launch(
            grid=(kv_blocks, q.shape[2], q.shape[0]),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
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
    bias_batch: int = 0,  # bias / dBias batch dim (1 = broadcast over B, b = per-batch); 0 unless bias_present.
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
    det_2k = PARAMS.det_2kernel
    if det_2k and d_qk not in (64, 128, 192, 256):
        raise ValueError(f"det_2kernel serves d_qk in (64, 128, 192, 256), any main-kernel-legal d_v (use the relay path otherwise); got d_qk={d_qk}")
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
        det_2kernel=det_2k,
        bias_present=PARAMS.bias_present,
        dbias_present=PARAMS.dbias_present,
        bias_is_fp32=PARAMS.bias_is_fp32,
    )
    if PARAMS.bias_present and bias_batch not in (1, b):
        raise ValueError(f"bias_batch must be 1 (broadcast) or B={b}; got {bias_batch}")
    d_qk_orig, d_v_orig = bwd.d_qk_orig, bwd.d_v_orig
    d_qk, d_v = bwd.d_qk, bwd.d_v
    sq_r = ceil_div(sq, ROW_ROUND) * ROW_ROUND

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
    if det_2k:
        fake_dq_accum = None
        fake_dq_sem = None
        skv_r = ceil_div(skv, ROW_ROUND) * ROW_ROUND
        fake_ds_ws = _fake(STORAGE_DTYPE, (b, qh, sq, skv_r))
    else:
        fake_dq_accum = _fake(cutlass.Float32, (b * sq_r * qh * d_qk,))
        # Sized for the smallest legal q-tile (32) so one formula covers every
        # tile choice; must match the adapter's carve (scratch_workspace_bytes).
        fake_dq_sem = _fake(cutlass.Int32, (b * qh * ceil_div(sq, 32),))
        fake_ds_ws = None
    # Main-kernel dK/dV destinations, always H_Q-headed: alias dk/d_v for MHA
    # (qh == kvh); per-q-head partials summed by dkv_reduce_kernel for GQA.
    fake_dk_ws = _fake(STORAGE_DTYPE, (b, skv, qh, d_qk)) if has_gqa else fake_dk
    fake_dv_ws = _fake(STORAGE_DTYPE, (b, skv, qh, d_v)) if has_gqa else fake_dv
    fake_seq_q_lens = _fake(cutlass.Int32, (b,)) if PARAMS.seq_q_lens_present else None
    fake_seq_kv_lens = _fake(cutlass.Int32, (b,)) if PARAMS.seq_kv_lens_present else None
    # bias / dBias: contiguous [1|B, H_Q, S_Q, S_KV]
    bias_dtype = cutlass.Float32 if PARAMS.bias_is_fp32 else STORAGE_DTYPE
    fake_bias = _fake(bias_dtype, (bias_batch, qh, sq, skv)) if PARAMS.bias_present else None
    fake_dbias_accum = _fake(cutlass.Float32, (bias_batch, qh, sq, skv)) if PARAMS.dbias_present else None
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    options = "--enable-tvm-ffi"

    compiled_dot = cute.compile(
        dot_do_o_host,
        fake_o,
        fake_do,
        fake_delta,
        fake_dq_accum,
        fake_dq_sem,
        bwd.q_tile,
        d_qk,
        d_v,
        bwd.chunk_elems,
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
        fake_ds_ws,
        fake_dk_ws,
        fake_dv_ws,
        fake_seq_q_lens,
        fake_seq_kv_lens,
        fake_bias,
        fake_dbias_accum,
        cutlass.Float32(1.0),
        cutlass.Float32(1.0),
        fake_stream,
        options=options,
    )
    compiled_cvt = None
    compiled_dq2k = None
    if det_2k:
        dq_gemm = SM120DetDqGemmKernel(
            in_dtype=STORAGE_DTYPE,
            is_causal=PARAMS.is_causal,
            causal_top_left=PARAMS.causal_top_left,
            right_slack=bwd.right_slack,
            head_dim=d_qk,
            q_tile=128 if d_qk <= 128 else 64,
            kv_tile=min(64, bwd.kv_tile),
            ws_q_tile=bwd.q_tile,
            use_pdl=PARAMS.use_pdl,
        )
        compiled_dq2k = cute.compile(
            dq_gemm,
            fake_k,
            fake_ds_ws,
            fake_dq,
            cutlass.Float32(1.0),
            fake_stream,
            options=options,
        )
    else:
        compiled_cvt = cute.compile(
            convert_dq_host,
            fake_dq_accum,
            fake_dq,
            bwd.q_tile,
            d_qk,
            bwd.chunk_elems,
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
            dkv_reduce_host,
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
    compiled_dbias_cvt = None
    if PARAMS.dbias_present and not PARAMS.dbias_is_fp32:
        dbias_total = bias_batch * qh * sq * skv
        compiled_dbias_cvt = cute.compile(
            convert_dbias_host,
            _fake(cutlass.Float32, (dbias_total,)),
            _fake(STORAGE_DTYPE, (dbias_total,)),
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
            dsink_host,
            fake_lse,
            fake_delta,
            fake_sink,
            fake_dsink,
            fake_seq_q_lens,
            bwd.use_pdl,
            fake_stream,
            options=options,
        )
    return SimpleNamespace(
        dot=compiled_dot,
        main=compiled_main,
        cvt=compiled_cvt,
        dq2k=compiled_dq2k,
        reduce=compiled_reduce,
        dbias_cvt=compiled_dbias_cvt,
        dsink=compiled_dsink,
    )
