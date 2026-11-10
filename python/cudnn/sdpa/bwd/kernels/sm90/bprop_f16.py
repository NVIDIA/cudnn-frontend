# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 (Hopper) SDPA backward main kernel -- FP16/BF16, head-dim envelopes 64 and 128.

The SM80 backward's two-sub-group pipeline (``kernels/sm80/bprop_f16.py``) on
warpgroup MMA.  One CTA owns a 64-row KV tile of one (batch | sequence, query
head) and streams 64-row Q tiles; each of its two warpgroups is one sub-group:

    WG0:  Sᵀ  = K · Qᵀ        (SS, K-major A/B)   → P = exp2(scale·S − LSE)
          dV += Pᵀ · dO        (RS: P from registers, dO MN-major)
    WG1:  dPᵀ = V · dOᵀ       (SS)                → dS = scale·(dP − delta)·P
          dK += dSᵀ · Q        (RS: dS from registers, Q MN-major)
    both: dQ  = dS · K         (SS: dSᵀ tile MN-major A, K MN-major B; WG w owns
                                 head columns [w·D/2, (w+1)·D/2)) → fp32 atomics

Sᵀ and dPᵀ put the KV row on the WGMMA M axis, so Pᵀ and dSᵀ come out of the
first GEMM as C fragments in exactly the register-A order the dV / dK GEMMs
consume (no SMEM round trip).  WG1 reads the unrounded fp32 P through a
thread-linear SMEM exchange (the two C fragments share one layout), and stores
dSᵀ once with ``stmatrix`` as the dQ GEMM's A operand.

Everything outside the main kernel is the SM80 chain, imported unchanged: the
do_dot / dSink preprocess, the fp32 dQ accumulator layout ``[B, S_q, H, D_QK]``
and its cast, the per-query-head dK / dV partials and their GQA reduction, the
deterministic dQ relay protocol, the grid schedulers and the THD origins.  The
prologue of ``_bprop_kernel`` (grid decode, sequence bounds, mask scaffolding,
Q-loop bounds, operand bases) is the SM80 one, so both rows serve one mask and
layout contract.

Loads are ``cp.async`` into 128-byte-swizzled 64-column slabs (the WGMMA SW128
layout); every generic-proxy SMEM write is published to the async proxy with
``fence.proxy.async`` before the CTA barrier that precedes the WGMMA reading it.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16  # noqa: E402
from cudnn.frost.tile_dsl.tma import load_tile_2d, cp_async_commit, cp_async_wait  # noqa: E402
from cudnn.frost.tile_dsl.mask import MASK_NONE, MASK_PADDED, MASK_CAUSAL, MASK_SWA, compute_q_loop_bounds, q_tile_kv_hi  # noqa: E402
from cudnn.frost.tile_dsl.scheduler import lpt_l2_tile_coords  # noqa: E402
from cudnn.frost.tile_dsl.thd import thd_port_origin  # noqa: E402
from cudnn.frost.tile_dsl.wgmma import wgmma_m64nNk16_f32, wgmma_smem_desc  # noqa: E402
from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL  # noqa: E402
from cudnn.sdpa.fwd.kernels.sm90._common_hopper import store_fragment  # noqa: E402

# The SM80 chain: preprocess, cast, reduction and dSink hosts, and the shared
# per-cell helpers.  None of them reads the template parameters.
from cudnn.sdpa.bwd.kernels.sm80.bprop_f16 import (  # noqa: E402,F401
    _atomic_add_f32,
    _cast_host,
    _cast_thd_host,
    _cast_thd_org_host,
    _clamp_lt,
    _dkv_reduce_host,
    _dkv_reduce_thd_host,
    _dkv_reduce_thd_org_host,
    _do_dot_host,
    _do_dot_thd_host,
    _dq_sem_wait,
    _dsink_host,
    _mask_p,
    _zero_if_ge,
)

SCHED_DEFAULT = SCHED_NATURAL
_LOG2E = 1.4426950408889634
# One 64-row x 64-column SW128 slab (elements); every SMEM operand is built from them.
_SLAB = 64 * 64
_WG_THREADS = 128

from cudnn.sdpa.bwd.config_sm80 import TemplateParams  # noqa: E402

PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())


def _load_slabs(dst, src, d, gmem_d, row_stride, tidx, valid_rows, row_base):
    """cp.async 64 rows x ``d`` columns into ``d // 64`` SW128 slabs of ``dst``.

    Columns at or past the operand's logical width ``gmem_d`` (the flavor
    envelope pads the head dim) and rows the ``valid_rows`` gate rejects are
    zero-filled without a global read."""
    for slab in range(d // 64):  # plain Python: this helper is traced inline
        load_tile_2d(
            dst.subview(slab * _SLAB),
            src + slab * 64,
            rows=64,
            elems_per_row=64,
            gmem_row_stride_elems=row_stride,
            valid_cols=gmem_d if gmem_d != d else None,
            col_base=cutlass.Int32(slab * 64) if gmem_d != d else None,
            tidx=tidx,
            num_threads=2 * _WG_THREADS,
            elems_per_copy=8,
            elem_bytes=2,
            swizzle=True,
            valid_rows=valid_rows,
            row_base=row_base,
        )


@cute.jit
def _c_row(t, i: cutlass.Constexpr[int]):
    """M row of C slot ``i`` for warpgroup-local thread ``t``."""
    return 16 * (t // 32) + (t % 32) // 4 + 8 * ((i // 2) % 2)


@cute.jit
def _c_col(t, i: cutlass.Constexpr[int]):
    """N column of C slot ``i`` for warpgroup-local thread ``t``."""
    return 2 * (t % 4) + (i % 2) + 8 * (i // 4)


@cute.jit
def _store_grad_pair(
    acc,
    i: cutlass.Constexpr[int],
    t,
    view,
    batch_base,
    row_stride,
    head_stride,
    width: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    kv_base,
    head,
    io_dtype: cutlass.Constexpr,
    row_live,
):
    """Store C slots ``i, i+1`` (one row, two adjacent head columns) of a dV / dK
    accumulator; columns past the logical ``width`` and gated rows are skipped."""
    row = _c_row(t, i)
    d0 = _c_col(t, i)
    base = batch_base + (cutlass.Int64(kv_base) + cutlass.Int64(row)) * row_stride + cutlass.Int64(head) * cutlass.Int64(head_stride) + d0
    word = fp32_to_fp16(acc[i], acc[i + 1], dtype=io_dtype)
    if cutlass.const_expr(width != d):
        if d0 < cutlass.Int32(width):
            if cutlass.const_expr(row_live is not None):
                if row_live:
                    Pointer(view.data_ptr() + base, dtype=cutlass.Int32).store(word, alignment=4)
            else:
                Pointer(view.data_ptr() + base, dtype=cutlass.Int32).store(word, alignment=4)
    elif cutlass.const_expr(row_live is not None):
        if row_live:
            Pointer(view.data_ptr() + base, dtype=cutlass.Int32).store(word, alignment=4)
    else:
        Pointer(view.data_ptr() + base, dtype=cutlass.Int32).store(word, alignment=4)


@cute.kernel
def _bprop_kernel(
    Q: cute.Tensor,  # [B, SQ,  H, D_QK]  io_dtype
    K: cute.Tensor,  # [B, SKV, H, D_QK]  io_dtype
    V: cute.Tensor,  # [B, SKV, H, D_V]   io_dtype
    dO: cute.Tensor,  # [B, SQ,  H, D_V]   io_dtype
    dQ_acc: cute.Tensor,  # [B, SQ,  H, D_QK]  fp32 (atomicAdd target)
    dK: cute.Tensor,  # [B, SKV, H, D_QK]  io_dtype (output)
    dV: cute.Tensor,  # [B, SKV, H, D_V]   io_dtype (output)
    LSE: cute.Tensor,  # [B, H, SQ] fp32 (natural-log)
    DO_DOT: cute.Tensor,  # [B, H, SQ] fp32 (sum_d O*dO, "delta")
    SEQ_KV_LENS: Optional[cute.Tensor],  # [B] int32 (per-batch KV length; dummy if unused)
    BIAS: Optional[cute.Tensor],  # [1|B, H, SQ, SKV] additive bias (dummy if unused)
    DBIAS: Optional[cute.Tensor],  # [1|B, H, SQ, SKV] fp32 bias-grad accumulator (atomicAdd)
    ROPE_CS: Optional[cute.Tensor],  # [max_s, d_qk//2, 2] fp32 (cos,sin); dummy if unused
    CU_Q: Optional[cute.Tensor],  # [B+1] int32 cumulative Q seqlens (THD/varlen); dummy otherwise.
    CU_K: Optional[cute.Tensor],  # [B+1] int32 cumulative KV seqlens (THD/varlen); dummy otherwise.
    SEQ_LEN_Q: Optional[cute.Tensor],  # [B] int32 (per-batch Q length; PADDED dense — dummy if unused)
    DQ_SEM: Optional[cute.Tensor],  # [n_seq*H*sem_q_stride] int32 deterministic-dQ relay counter; dummy (1-elem) if !deterministic
    ORG: Optional[cute.Tensor],  # THD: Int64 [rows, n_seq] caller-buffer token origins (see org_rows); None otherwise
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    tile_kv: cutlass.Constexpr[int],
    tile_q: cutlass.Constexpr[int],
    warps_per_sg: cutlass.Constexpr[int],
    qo_stages: cutlass.Constexpr[int],  # Q/dO SMEM ring depth (2 = double-buffer; 1 = load-wait-compute, for big-d SMEM fit)
    dq_smem_coalesce: cutlass.Constexpr[bool],  # stage dQ through sDQ for coalesced atomicAdd (False = direct scattered atomicAdd, saves tile_q*d_qk*4 B)
    io_dtype: cutlass.Constexpr,
    mask_flags: cutlass.Constexpr[int],  # MASK_* bitmask (compile-time)
    swa_window: cutlass.Constexpr[int],  # SWA window W (compile-time)
    causal_bottom_right: cutlass.Constexpr[int],  # bottom-right causal alignment
    has_seq_kv_lens: cutlass.Constexpr[bool],  # read SEQ_KV_LENS[batch] for PADDED
    has_bias: cutlass.Constexpr[bool],  # add BIAS[.,h,q,k] to S; dump dBias
    bias_is_fp32: cutlass.Constexpr[bool],  # BIAS dtype (fp32 vs io_dtype)
    has_rope: cutlass.Constexpr[bool],  # rotate Q/K in SMEM; un-rotate dQ/dK
    has_seq_len_q: cutlass.Constexpr[bool],  # read SEQ_LEN_Q[batch] → per-batch q-pad (dense PADDED)
    THD_VARLEN: cutlass.Constexpr[bool],  # packed [1,T,H,D] + CU_Q/CU_K; per-batch S_q/S_kv
    LSE_TOKEN_MAJOR: cutlass.Constexpr[bool],  # THD Stats packing: (T, H) token-major (True) or (1, H, head_stride) head-major (False)
    deterministic: cutlass.Constexpr[bool],  # gate the dQ semaphore relay (folds out → byte-identical fast path)
    sched_policy: cutlass.Constexpr[int],  # SCHED_DEFAULT / SCHED_LPT / SCHED_LPT_L2 grid decode
    sched_l2_bytes: cutlass.Constexpr[int],  # L2 budget sizing the SCHED_LPT_L2 head groups
    org_rows: cutlass.Constexpr,  # THD: ORG row per ORIGIN_PORTS entry (-1 = compact cu), or None
    n_q_tiles: cutlass.Int32,  # runtime — ceil(SQ / tile_q) (dense); THD recomputes per-batch
    softmax_scale_log2: cutlass.Float32,  # = attn_scale * log2(e)  (for P)
    attn_scale: cutlass.Float32,  # linear scale (for dS)
    right_bound: cutlass.Int32,  # causal right-band widening (k <= q+rb)
    bias_bstride: cutlass.Int64,  # bias batch stride (0 = broadcast over B)
    sem_q_stride: cutlass.Int32,  # DQ_SEM per-(seq,head,partition) stride
    grad_dim_splits: cutlass.Constexpr[int] = 1,
):
    # ---- compile-time derived counts --------------------------------------
    """SM90 SDPA backward main kernel: one CTA per (kv-tile, head, batch|sequence) owns dK/dV
    and streams Q-tiles, accumulating dQ across CTAs (atomics or the deterministic relay)."""
    threads = 2 * warps_per_sg * 32
    DQK_CHUNKS = d_qk // 16  # BMM1 K-reduce over d (S = K·Qᵀ)
    DV_CHUNKS = d_v // 16  # BMM1' K-reduce over d (dP = V·dOᵀ)
    Q_CHUNKS = tile_q // 16  # BMM2 K-reduce over q (dV/dK)
    KV_CHUNKS = tile_kv // 16  # dQ K-reduce over kv
    GRAD_DQK = d_qk // grad_dim_splits
    GRAD_DV = d_v // grad_dim_splits
    DQ_N = GRAD_DQK // 2  # dQ N per sub-group and column partition
    KV_PER_WARP = tile_kv // warps_per_sg  # rows of S/dP a warp owns (== 16)
    Q_PER_WARP = tile_q // warps_per_sg  # rows of dQ a warp owns (16 or 32)
    # dQ M-blocks per warp.  dQ is [tile_q, d_qk]; warps_per_sg warps each own
    # an m16n8 row group, so they cover warps_per_sg*16 q-rows per m-block.
    # When tile_q > that (e.g. gptoss d=64 with TILE_Q=128, warps_per_sg=4 →
    # 64 rows/m-block), each warp runs DQ_M_BLOCKS m-blocks (rows
    # mb*warps_per_sg*16 + warp_local*16 + ...).  Llama (tile_q=64) → 1 (the
    # single-m-block path is byte-identical to before).  BMM1 / dV / dK keep
    # M = tile_kv = warps_per_sg*16 (1 m-block); only dQ tiles along M.
    DQ_M_BLOCKS = tile_q // (warps_per_sg * 16)
    M_STRIDE = warps_per_sg * 16  # q-row stride between dQ m-blocks

    bx, by, bz = cute.arch.block_idx()
    grad_col_idx = bx % grad_dim_splits
    bx = bx // grad_dim_splits
    B = Q.shape[0]
    SQ = Q.shape[1]
    H = Q.shape[2]  # query heads (grid head dim; Q/dO/dQ/dK_ws/dV_ws)
    SKV = K.shape[1]
    Hkv = K.shape[2]  # KV heads (K/V); GQA/MQA when Hkv < H

    # ---- Partial-tile (arbitrary seqlen) gates -- compile-time (SQ/SKV are JIT
    #      shapes, tile_q/tile_kv constexprs).  When a seqlen isn't a tile
    #      multiple the LAST tile straddles the boundary: K/V/Q/dO loads
    #      zero-fill OOB rows (valid_rows), P is masked to 0 for kv>=SKV / q>=SQ,
    #      LSE/do_dot/bias reads are clamped to <SQ/<SKV, and the dV/dK/dQ/dBias
    #      GMEM stores are row-gated so no OOB write lands.  Both False ⇒ every
    #      branch folds out ⇒ the dense (tile-multiple) path is byte-identical.
    # THD compiles SQ/SKV as sym_int DYNAMICS (issue #604) so the modulus is
    # not a compile-time value there; force the static gates False — THD's
    # per-sequence bounds ride GATE_Q/GATE_KV via THD_VARLEN instead (and the
    # only pure PARTIAL_* consumer, the dBias store gate, is dense-only).
    PARTIAL_Q = False if cutlass.const_expr(THD_VARLEN) else ((SQ % tile_q) != 0)
    PARTIAL_KV = False if cutlass.const_expr(THD_VARLEN) else ((SKV % tile_kv) != 0)

    # Grid decode.  SCHED_DEFAULT: plain 3-D (kv_tile, head, batch).  SCHED_LPT:
    # 1-D kv-major flat grid, kv_tile = bx // (H*B), so all (head, batch) of
    # kv=0 come first (heaviest).  SCHED_LPT_L2: the same order within L2-sized
    # (batch, kv_head) groups, one group block after another.
    if cutlass.const_expr(sched_policy == SCHED_LPT):
        _hb = cutlass.Int32(H) * cutlass.Int32(B)
        kv_tile = bx // _hb
        _r = bx % _hb
        head = _r % cutlass.Int32(H)
        batch = _r // cutlass.Int32(H)
    elif cutlass.const_expr(sched_policy == SCHED_LPT_L2):
        # Block-cyclic over L2-sized (batch, kv_head) groups, kv-major within a
        # block: the shared fwd helper decodes (row_rank, head, batch) with the
        # row REVERSED for the forward's heavy-high-q order; the backward's heavy
        # tiles are LOW kv (kv-tile 0 attends every q), so undo the reversal.
        # per_group = this group's Q + dO + fp32 dQ stream over its heads_per_kv
        # query heads -- the bytes the group's CTAs re-touch every Q-iter.
        _nkv = cutlass.Int32((SKV + tile_kv - 1) // tile_kv)
        _hpk = H // Hkv
        _row, head, batch = lpt_l2_tile_coords(
            bx,
            cutlass.Int32(H),
            cutlass.Int32(B),
            _nkv,
            _hpk,
            cutlass.Int32(SQ),
            _hpk * (2 * d_qk + 2 * d_v + 4 * d_qk),
            sched_l2_bytes,
        )
        kv_tile = (_nkv - cutlass.Int32(1)) - _row
    else:
        kv_tile = bx
        if cutlass.const_expr(deterministic):
            # The deterministic relay hands each dQ slot from the highest visiting
            # kv-tile down, so a kv-tile waits on HIGHER ones: schedule those first
            # (lower blockIdx) to keep every awaited predecessor resident.
            kv_tile = cutlass.Int32(cute.arch.grid_dim()[0]) // grad_dim_splits - cutlass.Int32(1) - bx
        head = by
        batch = bz
    # GQA: CTA processes ONE query head `head`; its KV head = head // (H//Hkv).
    # dK/dV are written PER-QUERY-HEAD into an H-head buffer (each CTA owns a
    # unique query-head slice → no atomics) and summed over the query-head group
    # by a separate reduction kernel (see _dkv_reduce_*).  MHA (Hkv==H) → ratio
    # 1, kv_head==head, the reduction is identity (host skips it).
    gqa_ratio = H // Hkv
    kv_head = head // cutlass.Int32(gqa_ratio)

    # ---- THD / varlen: per-batch packed seq origins + lengths.  Q/K/V/dO/O are
    #      packed [1,T,H,D]; sequence `batch` owns Q rows [cu_q[b],cu_q[b+1]),
    #      KV rows [cu_k[b],cu_k[b+1]).  Under THD, SQ/SKV (== Q.shape[1]/K.shape[1])
    #      are the PACKED totals T_q/T_kv; the per-batch lengths are s_q_b/s_kv_b
    #      and the masking/clamping/gating bounds become RUNTIME (vs the dense
    #      compile-time SQ/SKV).  Dense: origin = batch*S, bound = S.  The whole
    #      block folds out at THD_VARLEN=False.
    if cutlass.const_expr(THD_VARLEN):
        _cuq = Pointer(cutlass.make_array_view(CU_Q).data_ptr(), dtype=cutlass.Int32)
        _cuk = Pointer(cutlass.make_array_view(CU_K).data_ptr(), dtype=cutlass.Int32)
        cu_q_b = _cuq[batch]
        cu_k_b = _cuk[batch]
        s_q_b = _cuq[batch + cutlass.Int32(1)] - cu_q_b
        s_kv_b = _cuk[batch + cutlass.Int32(1)] - cu_k_b
        q_row_origin = cutlass.Int64(cu_q_b)
        kv_row_origin = cutlass.Int64(cu_k_b)
        q_bound = s_q_b
        kv_bound = s_kv_b
        n_q_tiles_eff = (s_q_b + cutlass.Int32(tile_q - 1)) // cutlass.Int32(tile_q)
    else:
        q_row_origin = cutlass.Int64(batch) * cutlass.Int64(SQ)
        kv_row_origin = cutlass.Int64(batch) * cutlass.Int64(SKV)
        # PADDED dense: per-batch live Q length (mirrors eff_skv on the kv side).
        # rows q >= q_bound get P zeroed (→ dQ/dK/dV all 0 for padded q) and the
        # dQ store gated.  Dense full (has_seq_len_q=False) → q_bound = SQ.
        q_bound = (
            Pointer(cutlass.make_array_view(SEQ_LEN_Q).data_ptr() + batch, dtype=cutlass.Int32).load()
            if cutlass.const_expr(has_seq_len_q)
            else cutlass.Int32(SQ)
        )
        kv_bound = cutlass.Int32(SKV)
        n_q_tiles_eff = n_q_tiles
    # Predicate the partial-tile path when EITHER a static partial tile (dense,
    # SQ/SKV not tile multiples) OR THD (every packed sequence is potentially
    # partial, with a runtime per-batch bound).  q_bound/kv_bound carry the live
    # row count either way.
    GATE_Q = cutlass.const_expr(PARTIAL_Q or THD_VARLEN or has_seq_len_q)
    GATE_KV = cutlass.const_expr(PARTIAL_KV or THD_VARLEN)

    # Mask scaffolding (folds out entirely when mask_flags == MASK_NONE).
    # eff_skv = per-batch KV length under PADDED, else the full SKV.
    if cutlass.const_expr(has_seq_kv_lens):
        eff_skv = Pointer(cutlass.make_array_view(SEQ_KV_LENS).data_ptr() + batch, dtype=cutlass.Int32).load()
    else:
        eff_skv = cutlass.Int32(SKV)
    # Bottom-right causal diagonal base.  Aligns the diagonal to the bottom-right
    # corner = (effective KV length) - (effective Q length), per-batch under
    # padding (matches the forward's br_base = eff_skv - eff_sq); THD uses the
    # packed per-seq s_kv_b - s_q_b; dense full folds to SKV - SQ (eff_skv=SKV,
    # q_bound=SQ).  Only consulted for causal_bottom_right (TL uses q_lim=q_abs).
    causal_diag = (s_kv_b - s_q_b) if cutlass.const_expr(THD_VARLEN) else (eff_skv - q_bound)
    # Bias / dBias base offset for this (batch, head): [.,H,SQ,SKV] row-major.
    # bias_bstride == 0 ⇒ bias broadcast over batch (all batches share the same
    # [1,H,SQ,SKV] slice → dBias atomicAdd reduces over batch for free).
    if cutlass.const_expr(has_bias):
        bias_base = cutlass.Int64(batch) * cutlass.Int64(bias_bstride) + cutlass.Int64(head) * cutlass.Int64(SQ * SKV)
        _bias_ptr = cutlass.make_array_view(BIAS).data_ptr()
        _dbias_ptr = cutlass.make_array_view(DBIAS).data_ptr()

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = tidx // 32
    lane = tidx % 32
    sg_id = warp_idx // warps_per_sg  # 0 (warps 0..) or 1
    warp_local = warp_idx % warps_per_sg  # 0..warps_per_sg-1
    g_lane = lane // 4  # 0..7  (mma D-frag row group)
    p_lane = lane % 4  # 0..3  (mma D-frag col pos)
    is_sg0 = sg_id == cutlass.Int32(0)

    # ---- GMEM views + row strides (BSHD, token stride per operand) ---------
    Q_view = cutlass.make_array_view(Q)
    K_view = cutlass.make_array_view(K)
    V_view = cutlass.make_array_view(V)
    dO_view = cutlass.make_array_view(dO)
    dQ_view = cutlass.make_array_view(dQ_acc)
    dK_view = cutlass.make_array_view(dK)
    dV_view = cutlass.make_array_view(dV)
    LSE_view = cutlass.make_array_view(LSE)
    DOT_view = cutlass.make_array_view(DO_DOT)

    # Declared element strides, widened before multiplication. Dense layouts
    # carry independent batch/head strides; packed THD uses sequence origins.
    QK_RS = cutlass.Int64(Q.stride[1])
    V_RS = cutlass.Int64(dO.stride[1])
    K_RS = cutlass.Int64(K.stride[1])
    VV_RS = cutlass.Int64(V.stride[1])
    DK_RS = cutlass.Int64(dK.stride[1])
    DV_RS = cutlass.Int64(dV.stride[1])
    if cutlass.const_expr(THD_VARLEN):
        # Caller ports sit at their own token origins (bound ragged offsets);
        # a port without one, and every internal buffer, uses the compact cu.
        rows = org_rows if cutlass.const_expr(org_rows is not None) else (-1,) * 9
        # Q_BATCH / DO_BATCH serve the compact case; with origins the Q/dO tile
        # loads reload theirs at each use (QO_RELOAD below).
        Q_BATCH = thd_port_origin(ORG, rows[0], batch, cu_q_b) * QK_RS
        DO_BATCH = thd_port_origin(ORG, rows[4], batch, cu_q_b) * V_RS
        K_BATCH = thd_port_origin(ORG, rows[1], batch, cu_k_b) * K_RS
        V_BATCH = thd_port_origin(ORG, rows[2], batch, cu_k_b) * VV_RS
        DK_BATCH = thd_port_origin(ORG, rows[6], batch, cu_k_b) * DK_RS
        DV_BATCH = thd_port_origin(ORG, rows[7], batch, cu_k_b) * DV_RS
        lse_origin = thd_port_origin(ORG, rows[8], batch, cu_q_b)
    else:
        Q_BATCH = cutlass.Int64(batch) * cutlass.Int64(Q.stride[0])
        DO_BATCH = cutlass.Int64(batch) * cutlass.Int64(dO.stride[0])
        K_BATCH = cutlass.Int64(batch) * cutlass.Int64(K.stride[0])
        V_BATCH = cutlass.Int64(batch) * cutlass.Int64(V.stride[0])
        DK_BATCH = cutlass.Int64(batch) * cutlass.Int64(dK.stride[0])
        DV_BATCH = cutlass.Int64(batch) * cutlass.Int64(dV.stride[0])
    kv_base = kv_tile * cutlass.Int32(tile_kv)

    # ---- Q-loop bounds (shared tile_dsl arithmetic, the KV-major dual of the
    #      forward's compute_kv_loop_bounds): causal trims the q-loop from below
    #      (q above the diagonal never attends this kv-tile; ~2x on causal
    #      bprop), the sliding window trims it from above (the forward trims
    #      kv_left the same way).  The j-counter runs from 0 (q_iter = q_lo_tile
    #      + j) so the double-buffer stage parity is relative to the loop start.
    #      Non-causal, no window -> [0, n_q_tiles_eff) (byte-identical).  A
    #      kv-tile past every attending q (SKV > SQ causal edge, or a window that
    #      ends before it) runs 0 iters and the epilogue stores dK = dV = 0.
    #      Under THD the per-sequence lengths feed the diagonal and tile count.
    _qb = compute_q_loop_bounds(
        kv_base,
        seqlen_q=s_q_b if cutlass.const_expr(THD_VARLEN) else q_bound,
        seq_kv_len=s_kv_b if cutlass.const_expr(THD_VARLEN) else eff_skv,
        n_q_tiles=n_q_tiles_eff,
        window_left=swa_window,
        mask_flags=mask_flags,
        tile_q=tile_q,
        tile_kv=tile_kv,
        bottom_right=bool(causal_bottom_right),
        window_right=right_bound,
    )
    q_lo_tile = _qb.lo
    n_iters = cutlass.Int32(arith.maxsi((_qb.hi - _qb.lo).ir_value(), cutlass.Int32(0).ir_value()))
    # THD over-provisioned grid: this CTA's kv-tile may start past the packed
    # sequence's KV length (n_kv_tiles = ceil(max_skv/tile_kv) covers the longest
    # sequence).  Force 0 q-iters for such tiles → no compute, and the dV/dK
    # epilogue's kv<kv_bound gate suppresses every store.
    if cutlass.const_expr(THD_VARLEN):
        n_iters = cutlass.Int32(arith.select((kv_base >= s_kv_b).ir_value(), cutlass.Int32(0).ir_value(), n_iters.ir_value()))
    q_lo_base = q_lo_tile * cutlass.Int32(tile_q)
    if cutlass.const_expr(deterministic):
        # The deterministic relay's last kv-tile that runs any q-iter: THD skips
        # tiles past the packed sequence's KV length; dense tiles past a padded
        # length still iterate (and relay).
        if cutlass.const_expr(THD_VARLEN):
            relay_kv_last = (s_kv_b + cutlass.Int32(tile_kv - 1)) // cutlass.Int32(tile_kv) - cutlass.Int32(1)
        else:
            relay_kv_last = cutlass.Int32(cute.arch.grid_dim()[0]) // grad_dim_splits - cutlass.Int32(1)

    # K/V tile (row 0) GMEM element pointers — at the KV head (GQA-mapped).
    # kv_row_origin = batch*SKV (dense) or cu_k[b] (THD packed).
    k_tile_base = K_BATCH + cutlass.Int64(kv_base) * K_RS + cutlass.Int64(kv_head) * cutlass.Int64(K.stride[2])
    v_tile_base = V_BATCH + cutlass.Int64(kv_base) * VV_RS + cutlass.Int64(kv_head) * cutlass.Int64(V.stride[2])
    k_tile_gmem = K_view.data_ptr() + k_tile_base
    v_tile_gmem = V_view.data_ptr() + v_tile_base

    # LSE: STRIDE-AWARE on the dense path (a graph Stats input may carry any
    # dense-compatible [B,H,SQ] layout; the compile-time strides come from the
    # fake — a compact fake folds them to the packed constants, so the packed
    # case is byte-identical).  THD reads the PACKED Stats in either packing
    # Rule S1 admits, both plan-time (the fake's shape and strides):
    #   head-major  (1, H, head_stride): row = head*head_stride + cu_q[b] + q,
    #               head_stride == the packed extent when compact (the
    #               wrapper's [1,H,T_q]) or the caller's wider capacity;
    #   token-major (T, H): row = (cu_q[b] + q)*H + head (cuDNN's ragged recipe).
    # The grid `batch` is the LOGICAL sequence index, not the tensor's batch-1
    # dim.  do_dot is an internal packed buffer → packed math always.
    if cutlass.const_expr(THD_VARLEN):
        if cutlass.const_expr(LSE_TOKEN_MAJOR):
            lse_head_base = lse_origin * cutlass.Int64(LSE.stride[0]) + cutlass.Int64(head)
            LSE_Q_STRIDE = cutlass.Int64(LSE.stride[0])
        else:
            lse_head_base = cutlass.Int64(head) * cutlass.Int64(LSE.stride[1]) + lse_origin
            LSE_Q_STRIDE = cutlass.Int64(1)
        dot_head_base = cutlass.Int64(head) * cutlass.Int64(SQ) + cutlass.Int64(cu_q_b)
    else:
        lse_head_base = cutlass.Int64(batch) * cutlass.Int64(LSE.stride[0]) + cutlass.Int64(head) * cutlass.Int64(LSE.stride[1])
        LSE_Q_STRIDE = cutlass.Int64(LSE.stride[2])
        dot_head_base = (cutlass.Int64(batch) * cutlass.Int64(H) + cutlass.Int64(head)) * cutlass.Int64(SQ)

    # =======================================================================
    # SM90 body.  Everything above is the SM80 prologue (grid decode, bounds,
    # mask scaffolding, operand bases).  Below, warpgroup ``wg`` plays the SM80
    # sub-group ``sg_id``; ``t`` is the warpgroup-local thread of the C layout.
    # =======================================================================
    if cutlass.const_expr(has_bias or has_rope):
        raise NotImplementedError("sdpa_bwd_sm90: bias / dBias and RoPE are not implemented")
    if cutlass.const_expr(grad_dim_splits != 1):
        raise NotImplementedError("sdpa_bwd_sm90: gradient column partitions are not implemented")
    if cutlass.const_expr(tile_kv != 64 or tile_q != 64 or warps_per_sg != 4):
        raise ValueError("sdpa_bwd_sm90: the warpgroup pipeline is fixed at tile_kv = tile_q = 64, two warpgroups")
    if cutlass.const_expr(d_qk not in (64, 128) or d_v not in (64, 128)):
        raise ValueError(f"sdpa_bwd_sm90: head-dim envelopes are 64 and 128; got ({d_qk}, {d_v})")
    wg = tidx // _WG_THREADS
    t = tidx % _WG_THREADS
    DQ_N = d_qk // 2  # dQ columns per warpgroup
    S_SLOTS = tile_q // 2  # C slots of the m64 x n64 S / dP tile
    QSTAGE_Q = tile_q * d_qk
    QSTAGE_O = tile_q * d_v

    # ---- SMEM (SW128 slabs; 1024-byte aligned) ------------------------------
    sK = cutlass.Array(io_dtype, tile_kv * d_qk, alignment=1024, space=cutlass.AddressSpace.smem)
    sV = cutlass.Array(io_dtype, tile_kv * d_v, alignment=1024, space=cutlass.AddressSpace.smem)
    sQ = cutlass.Array(io_dtype, 2 * QSTAGE_Q, alignment=1024, space=cutlass.AddressSpace.smem)
    sdO = cutlass.Array(io_dtype, 2 * QSTAGE_O, alignment=1024, space=cutlass.AddressSpace.smem)
    # fp32 P, WG0 -> WG1, slot-major and thread-linear: both C fragments share
    # one layout, so thread t of WG1 reads what thread t of WG0 wrote.
    sP = cutlass.Array(cutlass.Float32, S_SLOTS * _WG_THREADS, alignment=16, space=cutlass.AddressSpace.smem)
    # dSᵀ [kv, q] in one SW128 slab: the dQ GEMM's MN-major A operand.
    sdS = cutlass.Array(io_dtype, tile_kv * tile_q, alignment=1024, space=cutlass.AddressSpace.smem)

    # ---- K / V once (rows past the live KV length zero-filled, as SM80) -----
    kv_valid = kv_bound if cutlass.const_expr(GATE_KV) else None
    kv_rbase = kv_base if cutlass.const_expr(GATE_KV) else None
    if cutlass.const_expr(has_seq_kv_lens and not THD_VARLEN):
        kv_valid = cute.math.max(cutlass.Int32(0), cute.math.min(eff_skv, cutlass.Int32(SKV)))
        kv_rbase = kv_base
    _load_slabs(sK, k_tile_gmem, d_qk, K.shape[3], K_RS, tidx, kv_valid, kv_rbase)
    _load_slabs(sV, v_tile_gmem, d_v, V.shape[3], VV_RS, tidx, kv_valid, kv_rbase)
    cp_async_commit()

    SQ_rt = q_bound
    bhead_qk = Q_BATCH + cutlass.Int64(head) * cutlass.Int64(Q.stride[2])
    bhead_v = DO_BATCH + cutlass.Int64(head) * cutlass.Int64(dO.stride[2])
    QO_RELOAD = cutlass.const_expr(THD_VARLEN and org_rows is not None and (org_rows[0] >= 0 or org_rows[4] >= 0))
    if cutlass.const_expr(QO_RELOAD):
        q_src = Q_view.data_ptr() + thd_port_origin(ORG, rows[0], batch, cu_q_b) * QK_RS + cutlass.Int64(head) * cutlass.Int64(Q.stride[2]) + q_lo_base * QK_RS
        do_src = dO_view.data_ptr() + thd_port_origin(ORG, rows[4], batch, cu_q_b) * V_RS + cutlass.Int64(head) * cutlass.Int64(dO.stride[2]) + q_lo_base * V_RS
    else:
        q_src = Q_view.data_ptr() + bhead_qk + q_lo_base * QK_RS
        do_src = dO_view.data_ptr() + bhead_v + q_lo_base * V_RS
    _load_slabs(sQ, q_src, d_qk, Q.shape[3], QK_RS, tidx, SQ_rt, q_lo_base)
    _load_slabs(sdO, do_src, d_v, dO.shape[3], V_RS, tidx, SQ_rt, q_lo_base)
    cp_async_commit()

    # ---- descriptors that do not move with the Q ring -----------------------
    k_desc = wgmma_smem_desc(sK, leading_byte_offset=16, stride_byte_offset=1024)
    v_desc = wgmma_smem_desc(sV, leading_byte_offset=16, stride_byte_offset=1024)
    ds_desc = wgmma_smem_desc(sdS, leading_byte_offset=_SLAB * 2, stride_byte_offset=1024)
    # dQ B = K[kv, wg*DQ_N : (wg+1)*DQ_N], MN-major; the column offset is in
    # 16-byte start units (slab, then 8 columns per unit inside the slab).
    dq_col0 = wg * cutlass.Int32(DQ_N)
    kq_desc = cutlass.Int64(wgmma_smem_desc(sK, leading_byte_offset=_SLAB * 2, stride_byte_offset=1024)) + cutlass.Int64(
        (dq_col0 // 64) * (_SLAB * 2 // 16) + (dq_col0 % 64) // 8
    )

    # dV (WG0) / dK (WG1) accumulator, persistent across the Q loop.
    GRAD_SLOTS = (d_qk if cutlass.const_expr(d_qk >= d_v) else d_v) // 2
    acc = cutlass.Array(cutlass.Float32, GRAD_SLOTS, alignment=16)
    for i in cutlass.range_constexpr(GRAD_SLOTS):
        acc[i] = cutlass.Float32(0.0)
    scores = cutlass.Array(cutlass.Float32, S_SLOTS, alignment=16)

    for j in cutlass.range(n_iters, unroll=1):
        q_iter = q_lo_tile + j
        q_base = q_iter * cutlass.Int32(tile_q)
        stage_cur = j % cutlass.Int32(2)
        stage_nxt = (j + cutlass.Int32(1)) % cutlass.Int32(2)
        q_cur = sQ.subview(stage_cur * cutlass.Int32(QSTAGE_Q))
        o_cur = sdO.subview(stage_cur * cutlass.Int32(QSTAGE_O))

        # ---- B0: prefetch tile j+1 (row-gated past the bound), wait for tile j.
        # The stage it overwrites was last read by tile j-1's WGMMAs, which every
        # thread completed before tile j-1's B2.
        nxt_qbase = q_base + cutlass.Int32(tile_q)
        if cutlass.const_expr(QO_RELOAD):
            q_src = (
                Q_view.data_ptr() + thd_port_origin(ORG, rows[0], batch, cu_q_b) * QK_RS + cutlass.Int64(head) * cutlass.Int64(Q.stride[2]) + nxt_qbase * QK_RS
            )
            do_src = (
                dO_view.data_ptr() + thd_port_origin(ORG, rows[4], batch, cu_q_b) * V_RS + cutlass.Int64(head) * cutlass.Int64(dO.stride[2]) + nxt_qbase * V_RS
            )
        else:
            q_src = Q_view.data_ptr() + bhead_qk + nxt_qbase * QK_RS
            do_src = dO_view.data_ptr() + bhead_v + nxt_qbase * V_RS
        _load_slabs(sQ.subview(stage_nxt * cutlass.Int32(QSTAGE_Q)), q_src, d_qk, Q.shape[3], QK_RS, tidx, SQ_rt, nxt_qbase)
        _load_slabs(sdO.subview(stage_nxt * cutlass.Int32(QSTAGE_O)), do_src, d_v, dO.shape[3], V_RS, tidx, SQ_rt, nxt_qbase)
        cp_async_commit()
        cp_async_wait(1)
        nvvm.fence_proxy(nvvm.Proxy.ASYNC_SHARED, space=nvvm.SharedSpace.shared_cta)
        nvvm.barrier_cta_sync()

        # ---- first GEMM: WG0 Sᵀ = K·Qᵀ, WG1 dPᵀ = V·dOᵀ (k over the head dim) --
        if wg == cutlass.Int32(0):
            c = cutlass.Vector.from_elements((cutlass.Float32(0.0),) * S_SLOTS, cutlass.Float32)
            q_desc = wgmma_smem_desc(q_cur, leading_byte_offset=16, stride_byte_offset=1024)
            nvvm.wgmma_fence_aligned()
            for k in cutlass.range_constexpr(d_qk // 16):
                off = (k // 4) * (_SLAB * 2 // 16) + (k % 4) * 2
                c = wgmma_m64nNk16_f32(c, k_desc + off, q_desc + off, io_dtype, accumulate=k != 0)
            nvvm.wgmma_commit_group_sync_aligned()
            nvvm.wgmma_wait_group_sync_aligned(0)
            scores.store(c)
        else:
            c = cutlass.Vector.from_elements((cutlass.Float32(0.0),) * S_SLOTS, cutlass.Float32)
            o_desc = wgmma_smem_desc(o_cur, leading_byte_offset=16, stride_byte_offset=1024)
            nvvm.wgmma_fence_aligned()
            for k in cutlass.range_constexpr(d_v // 16):
                off = (k // 4) * (_SLAB * 2 // 16) + (k % 4) * 2
                c = wgmma_m64nNk16_f32(c, v_desc + off, o_desc + off, io_dtype, accumulate=k != 0)
            nvvm.wgmma_commit_group_sync_aligned()
            nvvm.wgmma_wait_group_sync_aligned(0)
            scores.store(c)

        kv_r0 = kv_base + _c_row(t, 0)
        kv_r1 = kv_base + _c_row(t, 2)
        # ---- WG0: P = exp2(scale·S − LSE) with the SM80 cell masks → sP; dV ----
        if wg == cutlass.Int32(0):
            lse = [None] * 16
            for u in cutlass.range_constexpr(16):
                qc = q_base + _c_col(t, (u % 2) + 4 * (u // 2))
                qr = _clamp_lt(qc, q_bound) if cutlass.const_expr(GATE_Q) else qc
                lse[u] = Pointer(LSE_view.data_ptr() + lse_head_base + cutlass.Int64(qr) * LSE_Q_STRIDE, dtype=cutlass.Float32).load()
            probs = []
            for i in cutlass.range_constexpr(S_SLOTS):
                qc = q_base + _c_col(t, i)
                kv_a = kv_r0 if cutlass.const_expr((i // 2) % 2 == 0) else kv_r1
                row_lse = lse[(i % 2) + 2 * (i // 4)]
                p = cute.math.exp2(scores[i] * softmax_scale_log2 - row_lse * cutlass.Float32(_LOG2E), fastmath=True)
                # Rows with an empty softmax (LSE = -inf) have no probability mass.
                live = row_lse > cutlass.Float32(-3.0e38)
                p = cutlass.Float32(arith.select(live.ir_value(), p.ir_value(), cutlass.Float32(0.0).ir_value()))
                if cutlass.const_expr(mask_flags != MASK_NONE):
                    p = _mask_p(
                        p,
                        kv_a,
                        qc,
                        mask_flags=mask_flags,
                        swa_window=swa_window,
                        causal_bottom_right=causal_bottom_right,
                        causal_diag=causal_diag,
                        eff_skv=eff_skv,
                        right_bound=right_bound,
                    )
                if cutlass.const_expr(GATE_KV):
                    p = _zero_if_ge(p, kv_a, kv_bound)
                if cutlass.const_expr(GATE_Q):
                    p = _zero_if_ge(p, qc, q_bound)
                Pointer(sP.subview(cutlass.Int32(i * _WG_THREADS) + t).data_ptr(), dtype=cutlass.Float32).store(p)
                probs.append(p)
            p_words = cutlass.Vector.from_elements(probs, cutlass.Float32).to(io_dtype).bitcast(cutlass.Uint32)
            # dV += Pᵀ · dO  (A = P words, B = dO[q, d_v] MN-major); WG1 only
            # needs sP, so this runs ahead of B1.
            od_mn = wgmma_smem_desc(o_cur, leading_byte_offset=_SLAB * 2, stride_byte_offset=1024)
            g = acc.load(0, d_v // 2)
            nvvm.wgmma_fence_aligned()
            for k in cutlass.range_constexpr(tile_q // 16):
                g = wgmma_m64nNk16_f32(g, p_words[4 * k : 4 * k + 4], od_mn + 128 * k, io_dtype, transpose_b=True)
            nvvm.wgmma_commit_group_sync_aligned()
            nvvm.wgmma_wait_group_sync_aligned(0)
            acc.store(g)

        # ---- B1: sP published --------------------------------------------------
        nvvm.barrier_cta_sync()

        if wg == cutlass.Int32(1):
            # dS = scale·(dP − delta)·P
            dot = [None] * 16
            for u in cutlass.range_constexpr(16):
                qc = q_base + _c_col(t, (u % 2) + 4 * (u // 2))
                qr = _clamp_lt(qc, q_bound) if cutlass.const_expr(GATE_Q) else qc
                dot[u] = Pointer(DOT_view.data_ptr() + dot_head_base + qr, dtype=cutlass.Float32).load()
            grads = []
            for i in cutlass.range_constexpr(S_SLOTS):
                p = Pointer(sP.subview(cutlass.Int32(i * _WG_THREADS) + t).data_ptr(), dtype=cutlass.Float32).load()
                ds = attn_scale * ((scores[i] - dot[(i % 2) + 2 * (i // 4)]) * p)
                if cutlass.const_expr(has_seq_len_q):
                    # An empty dense member has no live delta row to clamp to and
                    # its O/dO/Stats may be NaN: masked P = 0 must give dS = 0.
                    ds = cutlass.Float32(arith.select((p == cutlass.Float32(0.0)).ir_value(), cutlass.Float32(0.0).ir_value(), ds.ir_value()))
                grads.append(ds)
            ds_words = cutlass.Vector.from_elements(grads, cutlass.Float32).to(io_dtype).bitcast(cutlass.Uint32)
            # dK += dSᵀ · Q  (A = dS words, B = Q[q, d_qk] MN-major)
            qd_mn = wgmma_smem_desc(q_cur, leading_byte_offset=_SLAB * 2, stride_byte_offset=1024)
            g = acc.load(0, d_qk // 2)
            nvvm.wgmma_fence_aligned()
            for k in cutlass.range_constexpr(tile_q // 16):
                g = wgmma_m64nNk16_f32(g, ds_words[4 * k : 4 * k + 4], qd_mn + 128 * k, io_dtype, transpose_b=True)
            nvvm.wgmma_commit_group_sync_aligned()
            # dSᵀ for dQ while the dK group runs.
            store_fragment(ds_words, sdS, t)
            nvvm.wgmma_wait_group_sync_aligned(0)
            acc.store(g)
            nvvm.fence_proxy(nvvm.Proxy.ASYNC_SHARED, space=nvvm.SharedSpace.shared_cta)

        # ---- B2: sdS published; every read of the Q / dO stage is complete ----
        nvvm.barrier_cta_sync()

        # ---- dQ = dS · K (k over the 64 KV rows), this warpgroup's columns -----
        dq = cutlass.Vector.from_elements((cutlass.Float32(0.0),) * (DQ_N // 2), cutlass.Float32)
        nvvm.wgmma_fence_aligned()
        for k in cutlass.range_constexpr(tile_kv // 16):
            dq = wgmma_m64nNk16_f32(dq, ds_desc + 128 * k, kq_desc + 128 * k, io_dtype, transpose_a=True, transpose_b=True, accumulate=k != 0)
        nvvm.wgmma_commit_group_sync_aligned()
        nvvm.wgmma_wait_group_sync_aligned(0)

        # Deterministic relay (acquire), exactly the SM80 protocol.
        if cutlass.const_expr(deterministic):
            sem_idx = (batch * cutlass.Int32(H) + head) * sem_q_stride + q_iter
            sem_ptr = Pointer(cutlass.make_array_view(DQ_SEM).data_ptr() + sem_idx, dtype=cutlass.Int32)
            relay_kv_hi = q_tile_kv_hi(
                q_iter,
                seqlen_q=s_q_b if cutlass.const_expr(THD_VARLEN) else q_bound,
                seq_kv_len=s_kv_b if cutlass.const_expr(THD_VARLEN) else eff_skv,
                kv_last=relay_kv_last,
                mask_flags=mask_flags,
                tile_q=tile_q,
                tile_kv=tile_kv,
                bottom_right=bool(causal_bottom_right),
                window_right=right_bound,
            )
            relay_turn = relay_kv_hi - kv_tile
            if tidx == cutlass.Int32(0):
                _dq_sem_wait(sem_ptr, relay_turn)
            nvvm.barrier_cta_sync()

        for i in cutlass.range_constexpr(DQ_N // 2):
            qrow = _c_row(t, i)
            col = dq_col0 + _c_col(t, i)
            g_off = ((q_row_origin + q_base + qrow) * cutlass.Int32(H) + head) * cutlass.Int32(d_qk) + col
            if cutlass.const_expr(GATE_Q):
                if (q_base + qrow) < q_bound:
                    _atomic_add_f32(dQ_view.data_ptr() + g_off, dq[i])
            else:
                _atomic_add_f32(dQ_view.data_ptr() + g_off, dq[i])

        if cutlass.const_expr(deterministic):
            nvvm.barrier_cta_sync()
            if tidx == cutlass.Int32(0):
                nvvm.store_ext((relay_turn + cutlass.Int32(1)).ir_value(), sem_ptr, order=nvvm.MemOrder.RELEASE, scope=nvvm.MemScope.GPU)

    # Drain the last (row-gated) prefetch.
    cp_async_wait(0)

    # =======================================================================
    # Epilogue: WG0 writes dV, WG1 dK; the CTA owns its KV rows (no atomics).
    # =======================================================================
    if cutlass.const_expr(GATE_KV):
        kv_top_ok = (kv_base + _c_row(t, 0)) < kv_bound
        kv_bot_ok = (kv_base + _c_row(t, 2)) < kv_bound
    if wg == cutlass.Int32(0):
        for w in cutlass.range_constexpr(d_v // 4):
            _store_grad_pair(
                acc,
                2 * w,
                t,
                dV_view,
                DV_BATCH,
                DV_RS,
                dV.stride[2],
                dV.shape[3],
                d_v,
                kv_base,
                head,
                io_dtype,
                (kv_top_ok if cutlass.const_expr(w % 2 == 0) else kv_bot_ok) if cutlass.const_expr(GATE_KV) else None,
            )
    else:
        for w in cutlass.range_constexpr(d_qk // 4):
            _store_grad_pair(
                acc,
                2 * w,
                t,
                dK_view,
                DK_BATCH,
                DK_RS,
                dK.stride[2],
                dK.shape[3],
                d_qk,
                kv_base,
                head,
                io_dtype,
                (kv_top_ok if cutlass.const_expr(w % 2 == 0) else kv_bot_ok) if cutlass.const_expr(GATE_KV) else None,
            )


_bprop_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
    SEQ_KV_LENS: Optional[cute.Tensor],
    BIAS: Optional[cute.Tensor],
    DBIAS: Optional[cute.Tensor],
    ROPE_CS: Optional[cute.Tensor],
    CU_Q: Optional[cute.Tensor],
    CU_K: Optional[cute.Tensor],
    SEQ_LEN_Q: Optional[cute.Tensor],
    DQ_SEM: Optional[cute.Tensor],
    ORG: Optional[cute.Tensor],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    tile_kv: cutlass.Constexpr[int],
    tile_q: cutlass.Constexpr[int],
    warps_per_sg: cutlass.Constexpr[int],
    qo_stages: cutlass.Constexpr[int],
    dq_smem_coalesce: cutlass.Constexpr[bool],
    io_dtype: cutlass.Constexpr,
    mask_flags: cutlass.Constexpr[int],
    swa_window: cutlass.Constexpr[int],
    causal_bottom_right: cutlass.Constexpr[int],
    has_seq_kv_lens: cutlass.Constexpr[bool],
    has_bias: cutlass.Constexpr[bool],
    bias_is_fp32: cutlass.Constexpr[bool],
    has_rope: cutlass.Constexpr[bool],
    has_seq_len_q: cutlass.Constexpr[bool],
    thd_varlen: cutlass.Constexpr[bool],
    lse_token_major: cutlass.Constexpr[bool],
    deterministic: cutlass.Constexpr[bool],
    sched_policy: cutlass.Constexpr[int],
    sched_l2_bytes: cutlass.Constexpr[int],
    org_rows: cutlass.Constexpr,
    n_q_tiles: cutlass.Int32,
    softmax_scale_log2: cutlass.Float32,
    attn_scale: cutlass.Float32,
    right_bound: cutlass.Int32,
    bias_bstride: cutlass.Int64,
    sem_q_stride: cutlass.Int32,
    grid_kv_tiles: cutlass.Int32,
    grid_batch: cutlass.Int32,
    stream: cuda.CUstream,
    grad_dim_splits: cutlass.Constexpr[int] = 1,
):
    SKV = K.shape[1]
    H = Q.shape[2]
    B = Q.shape[0]
    threads = 2 * warps_per_sg * 32
    # Dense grid is shape-derived; THD over-provisions to (ceil(max_skv/tile_kv),
    # H, n_seq) from host-computed runtime counts (packed B == 1).  THD uses the
    # 3-D grid only (SCHED_DEFAULT); LPT+THD is a future scheduler tweak.
    if cutlass.const_expr(thd_varlen):
        grid = (grid_kv_tiles * grad_dim_splits, H, grid_batch)
    elif cutlass.const_expr(sched_policy == SCHED_LPT or sched_policy == SCHED_LPT_L2):
        n_kv_tiles = (SKV + tile_kv - 1) // tile_kv
        grid = (n_kv_tiles * H * B * grad_dim_splits, 1, 1)
    else:
        n_kv_tiles = (SKV + tile_kv - 1) // tile_kv
        grid = (n_kv_tiles * grad_dim_splits, H, B)
    _bprop_kernel(
        Q,
        K,
        V,
        dO,
        dQ_acc,
        dK,
        dV,
        LSE,
        DO_DOT,
        SEQ_KV_LENS,
        BIAS,
        DBIAS,
        ROPE_CS,
        CU_Q,
        CU_K,
        SEQ_LEN_Q,
        DQ_SEM,
        ORG,
        d_qk,
        d_v,
        tile_kv,
        tile_q,
        warps_per_sg,
        qo_stages,
        dq_smem_coalesce,
        io_dtype,
        mask_flags,
        swa_window,
        causal_bottom_right,
        has_seq_kv_lens,
        has_bias,
        bias_is_fp32,
        has_rope,
        has_seq_len_q,
        thd_varlen,
        lse_token_major,
        deterministic,
        sched_policy,
        sched_l2_bytes,
        org_rows,
        n_q_tiles,
        softmax_scale_log2,
        attn_scale,
        right_bound,
        bias_bstride,
        sem_q_stride,
        grad_dim_splits,
    ).launch(grid=grid, block=(threads, 1, 1), stream=stream)
