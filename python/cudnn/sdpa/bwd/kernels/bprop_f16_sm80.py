# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 (Ampere / A100) SDPA BACKWARD — Llama d_qk=d_v=128 / GPT-OSS d_qk=d_v=64, FP16/BF16.

Companion to the forward ``prefill_f16_sm80.py``.  Computes dQ / dK / dV
for non-causal MHA (KH = VH = QH) from saved ``LSE`` (forward log-sum-exp) and
``do_dot[q] = sum_d O[q,d]*dO[q,d]`` (the bprop "delta", fed in — see
``bprop_do_dot`` follow-up).  No mbarriers / clusters (Ampere) — pure
``cp.async`` + ``__syncthreads`` + warp-collective ``mma.sync.m16n8k16``.

Pipeline (the user's 2-sub-group design)
----------------------------------------
1 CTA = one (batch, head, **KV-tile**); the CTA loops over Q-tiles.  K/V are
loaded once and reused across the whole Q-loop; Q/dO are reloaded per Q-iter.

Two sub-groups of ``WARPS_PER_SG`` warps each (256 threads at the 4+4 default):

    sg0:  S  = K · Qᵀ    → P = exp2(scale·S − LSE_q)   → write sP
                                                        → dV += P · dO   (accum)
    sg1:  dP = V · dOᵀ                                  → read sP,
          dS = scale·(dP − do_dot_q)·P                  → write sdS / sdSᵀ
                                                        → dK += dS · Q   (accum)
    both: dQ += dSᵀ · K   (sg0 = d-cols 0:d/2, sg1 = d/2:d)  → atomicAdd dQ_acc

All S/dP/P/dS tiles are ``[TILE_KV, TILE_Q]``.  P and dS are exchanged through
SMEM (``sP`` sg0→sg1; ``sdS`` for dK, ``sdSᵀ`` transposed for dQ's A operand).
``dV``/``dK`` accumulate in registers across the Q-loop and are written ONCE at
CTA end (the CTA owns its KV slice → no atomics).  ``dQ`` is ``atomicAdd``-ed
into a FP32 GMEM accumulator (every KV-tile contributes to every Q-row's dQ);
a separate cast kernel (:func:`_cast_kernel`) narrows ``dQ_acc`` → FP16/BF16.

dQ atomicAdd is **COALESCED via SMEM staging** (``sDQ``): the dQ MMA C-fragment
scatters across 8 rows per warp → 8 L2 sectors per atomic request (NCU).  Staging
the tile to ``sDQ`` and then atomicAdd-ing in row-major order (consecutive lanes
→ consecutive dQ columns) drops it to 4 sectors/request → halves the dQ atomic L2
traffic.  This is what closes the gap to cuDNN (whose dQ atomic also coalesces):
+18 % @ B2H16S4096 (75→89 TFLOPS, ~1.0× cuDNN), +16 % @ B1H16S2048 (1.08×).

**Deterministic dQ** (``backward(deterministic=True)``): the cross-KV-tile dQ
atomicAdd is order-non-deterministic (fp32 add is non-associative → bitwise
varies run-to-run once a sequence spans >1 KV-tile).  The deterministic path
orders the adds by ``kv_tile`` via a per-(seq,head,q_tile) int32 GMEM semaphore
(``DQ_SEM``): each KV-tile CTA spins (acquire) until the counter == its
``kv_tile``, does its (coalesced/direct) atomicAdds, then releases the token to
``kv_tile+1`` (FlashAttention-2 deterministic pattern).  Ordering by
``kv_tile == blockIdx.x`` of the SCHED_DEFAULT 3-D grid makes the awaited
predecessor strictly lower-blockIdx (scheduled first) → deadlock-free; the relay
is q-skip-safe because ``q_lo_tile`` is monotonic in ``kv_tile`` so every lower
KV-tile reaching a q-tile relays before us (wait target is exactly ``kv_tile``).
Forces SCHED_DEFAULT (LPT remaps ``kv_tile`` and would break the order); zero
extra GMEM beyond the small counter; perf-insensitive (gated knob).  dK/dV are
already deterministic (each CTA owns distinct K rows → no cross-tile atomic).
All semaphore code folds out under ``const_expr(deterministic)`` → the default
(non-deterministic) path is byte-identical.

Why two sub-groups: register capacity.  dV_acc + dK_acc + frags at d=128 bust
the 255-reg/thread Ampere cap in a single group; splitting halves the live
accumulator footprint (sg0 holds dV, sg1 holds dK — one shared LOCAL array).

Named-"barrier" table (all ``nvvm.barrier_cta_sync()`` — full 256-thread CTA):
  * B0  after Q/dO ``cp.async`` load          (top of each Q-iter)
  * B1  after sg0 writes ``sP``               (sg1's dS + sg0's dV both read sP)
  * B2  after sg1 writes ``sdS``/``sdSᵀ``     (dQ reads sdSᵀ; sP/sdO read done)
  * B2b after both sg stage ``sDQ``           (before the coalesced dQ atomicAdd)
  * B3  after dQ atomicAdds                    (before next Q-iter reloads Q/dO)
ALL barriers are at top level (never inside a divergent ``if sg0`` arm) so the
per-thread barrier count is identical across both sub-groups.

v1 envelope: f16/bf16, dense (no mask), MHA, SQ % TILE_Q == 0, SKV % TILE_KV == 0,
d_qk == d_v, (d_qk//2) % 16 == 0 (llama d=128, gptoss d=64).  do_dot + LSE fed in
(computed by the reference / forward kernel for now; on-device do_dot is a
follow-up).

The kernel is fully parameterized on d_qk/d_v/tile_kv/tile_q/warps_per_sg — d=64
(gptoss) and d=128 (llama) run the SAME code; d=64 uses strictly less SMEM /
registers.  The only d-specific subtlety was the dQ d-col-split swizzle, now
handled d-agnostically via load_b_smem_x4(col_base=...).
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
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass._mlir.dialects import arith

from cudnn.frost.tile_dsl.mma import load_b_smem_x4, mma_step  # noqa: E402
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16  # noqa: E402
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b  # noqa: E402
from cudnn.frost.tile_dsl.tma import load_tile_2d, cp_async_commit, cp_async_wait  # noqa: E402
from cudnn.frost.tile_dsl.mask import MASK_NONE, MASK_PADDED, MASK_CAUSAL, MASK_SWA  # noqa: E402
from cudnn.frost.tile_dsl.rope import rope_rotate_smem_tile  # noqa: E402

# Pull the default geometry from the flavor config (single source of truth).
from cudnn.sdpa.bwd.config_sm80 import LLAMA_CFG as _LLAMA_CFG  # noqa: E402

_LOG2E = 1.4426950408889634
_ELEM_BYTES = 2
_COPY_ELEMS = 8  # 16-byte cp.async chunk (8 fp16)

# Scheduler policy (compile-time).  The bprop parallel axis is the KV-tile; under
# causal the lowest kv-tile is the heaviest (most q-iters after the causal
# skip), so an LPT schedule is KV-MAJOR (all kv=0 tiles first → the light high-kv
# tiles land in the last wave, minimizing the makespan tail).
SCHED_DEFAULT = 0  # 3-D grid (kv_tile, head, batch); no reorder (byte-identical)
SCHED_LPT = 1  # 1-D kv-major grid for causal load-balance


def _mask_p(p, kv_abs, q_abs, *, mask_flags: int, swa_window: int, causal_bottom_right: int, causal_diag, eff_skv, right_bound):
    """Zero a recomputed softmax probability ``p`` if its ``(kv_abs, q_abs)``
    cell is masked.  Mirrors ``tile_dsl.mask.apply_mask_chunk`` but per-scalar
    (the bprop sg0 frag holds 4 scattered cells/lane).  Masking ``P=0`` makes
    ``dS, dK, dV, dQ, dBias`` inherit the mask for free (they're all ∝ P).

    All ``mask_flags`` / ``swa_window`` / ``causal_bottom_right`` are Python
    constexprs (arms fold out); ``causal_diag`` (= SKV-SQ), ``eff_skv``,
    ``right_bound`` are runtime ``cutlass.Int32``.  ``mask_flags == 0`` → no IR.
    """
    if cutlass.const_expr(mask_flags == MASK_NONE):
        return p
    masked = None
    if cutlass.const_expr(mask_flags & MASK_PADDED):
        t = kv_abs >= eff_skv
        masked = t if masked is None else (masked | t)
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        q_lim = (q_abs + causal_diag) if cutlass.const_expr(causal_bottom_right) else q_abs
        t = kv_abs > (q_lim + right_bound)
        masked = t if masked is None else (masked | t)
    if cutlass.const_expr(mask_flags & MASK_SWA):
        # SWA lower bound: keep kv >= q + (br diagonal) - W.  Under bottom-right
        # the window floor shifts by causal_diag just like the causal upper bound
        # (mirrors the forward: q + br_base - swa_window).  Top-left: causal_diag
        # is not added (anchor = q_abs).
        swa_anchor = (q_abs + causal_diag) if cutlass.const_expr(causal_bottom_right) else q_abs
        t = kv_abs < (swa_anchor - cutlass.Int32(swa_window))
        masked = t if masked is None else (masked | t)
    return cutlass.Float32(arith.select(masked.ir_value(), cutlass.Float32(0.0).ir_value(), p.ir_value()))


def _zero_if_ge(p, idx, lim):
    """Partial-tile bound: zero ``p`` when ``idx >= lim`` (runtime cutlass.Int32).
    Used to mask padded kv-rows (kv >= SKV) / q-rows (q >= SQ) on the last
    partial tile of an arbitrary-seqlen problem.  ``idx`` runtime, ``lim``
    runtime; caller const_expr-gates the whole call so dense folds out."""
    m = idx >= lim
    return cutlass.Float32(arith.select(m.ir_value(), cutlass.Float32(0.0).ir_value(), p.ir_value()))


def _clamp_lt(idx, bound_rt):
    """Clamp an index to ``[0, bound_rt-1]`` so a GMEM read indexed by it stays
    in bounds on the partial last tile (rows q >= SQ / kv >= SKV) or a short
    packed THD sequence.  ``bound_rt`` is a runtime ``cutlass.Int32`` (the live row
    count: SQ/SKV for dense-partial, s_q_b/s_kv_b for THD); the masked P zeroes
    the result so the clamped value is only ever read, never used live.

    Floored at 0: a per-batch q-length / kv-length of 0 (empty padded sequence,
    e.g. seq_len_q[b]==0) makes bound_rt-1 == -1; without the max() the read
    index goes negative → illegal GMEM access (the read value is unused, but the
    address must stay in bounds).  P is masked to 0 for those rows regardless."""
    hi = arith.minsi(idx.ir_value(), (bound_rt - cutlass.Int32(1)).ir_value())
    return cutlass.Int32(arith.maxsi(hi, cutlass.Int32(0).ir_value()))


# ===========================================================================
# Main backward kernel.
# ===========================================================================
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
    SEQ_KV_LENS: cute.Tensor,  # [B] int32 (per-batch KV length; dummy if unused)
    BIAS: cute.Tensor,  # [1|B, H, SQ, SKV] additive bias (dummy if unused)
    DBIAS: cute.Tensor,  # [1|B, H, SQ, SKV] fp32 bias-grad accumulator (atomicAdd)
    ROPE_CS: cute.Tensor,  # [max_s, d_qk//2, 2] fp32 (cos,sin); dummy if unused
    CU_Q: cute.Tensor,  # [B+1] int32 cumulative Q seqlens (THD/varlen); dummy otherwise.
    CU_K: cute.Tensor,  # [B+1] int32 cumulative KV seqlens (THD/varlen); dummy otherwise.
    SEQ_LEN_Q: cute.Tensor,  # [B] int32 (per-batch Q length; PADDED dense — dummy if unused)
    DQ_SEM: cute.Tensor,  # [n_seq*H*sem_q_stride] int32 deterministic-dQ relay counter; dummy (1-elem) if !deterministic
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
    deterministic: cutlass.Constexpr[bool],  # gate the dQ semaphore relay (folds out → byte-identical fast path)
    sched_policy: cutlass.Constexpr[int],  # SCHED_DEFAULT / SCHED_LPT grid decode
    n_q_tiles: cutlass.Int32,  # runtime — ceil(SQ / tile_q) (dense); THD recomputes per-batch
    softmax_scale_log2: cutlass.Float32,  # = attn_scale * log2(e)  (for P)
    attn_scale: cutlass.Float32,  # linear scale (for dS)
    right_bound: cutlass.Int32,  # causal right-band widening (k <= q+rb)
    inv_softmax_scale: cutlass.Float32,  # 1/scale (fold bias into UNSCALED acc1)
    bias_bstride: cutlass.Int32,  # bias batch stride (0 = broadcast over B)
    sem_q_stride: cutlass.Int32,  # DQ_SEM per-(seq,head) stride = ceil(max_SQ/tile_q) (deterministic only)
):
    # ---- compile-time derived counts --------------------------------------
    threads = 2 * warps_per_sg * 32
    DQK_CHUNKS = d_qk // 16  # BMM1 K-reduce over d (S = K·Qᵀ)
    DV_CHUNKS = d_v // 16  # BMM1' K-reduce over d (dP = V·dOᵀ)
    Q_CHUNKS = tile_q // 16  # BMM2 K-reduce over q (dV/dK)
    KV_CHUNKS = tile_kv // 16  # dQ K-reduce over kv
    DQ_N = d_qk // 2  # dQ N per sub-group
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
    PARTIAL_Q = (SQ % tile_q) != 0
    PARTIAL_KV = (SKV % tile_kv) != 0

    # Grid decode.  SCHED_DEFAULT: plain 3-D (kv_tile, head, batch).  SCHED_LPT:
    # 1-D kv-major flat grid — bx = kv_tile*(H*B) + head*B... no: kv-major means
    # kv_tile = bx // (H*B), so all (head,batch) of kv=0 come first (heaviest).
    if cutlass.const_expr(sched_policy == SCHED_LPT):
        _hb = cutlass.Int32(H) * cutlass.Int32(B)
        kv_tile = bx // _hb
        _r = bx % _hb
        head = _r % cutlass.Int32(H)
        batch = _r // cutlass.Int32(H)
    else:
        kv_tile = bx
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
        q_row_origin = cu_q_b
        kv_row_origin = cu_k_b
        q_bound = s_q_b
        kv_bound = s_kv_b
        n_q_tiles_eff = (s_q_b + cutlass.Int32(tile_q - 1)) // cutlass.Int32(tile_q)
    else:
        q_row_origin = batch * cutlass.Int32(SQ)
        kv_row_origin = batch * cutlass.Int32(SKV)
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
        bias_base = batch * bias_bstride + head * cutlass.Int32(SQ * SKV)
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

    # ---- GMEM views + row strides (BSHD packed) ---------------------------
    Q_view = cutlass.make_array_view(Q)
    K_view = cutlass.make_array_view(K)
    V_view = cutlass.make_array_view(V)
    dO_view = cutlass.make_array_view(dO)
    dQ_view = cutlass.make_array_view(dQ_acc)
    dK_view = cutlass.make_array_view(dK)
    dV_view = cutlass.make_array_view(dV)
    LSE_view = cutlass.make_array_view(LSE)
    DOT_view = cutlass.make_array_view(DO_DOT)

    QK_RS = cutlass.Int32(H) * cutlass.Int32(d_qk)  # Q / dQ / dK_ws row stride (H_q heads)
    V_RS = cutlass.Int32(H) * cutlass.Int32(d_v)  # dO / dV_ws row stride (H_q heads)
    K_RS = cutlass.Int32(Hkv) * cutlass.Int32(d_qk)  # K read row stride (H_kv heads)
    VV_RS = cutlass.Int32(Hkv) * cutlass.Int32(d_v)  # V read row stride (H_kv heads)
    kv_base = kv_tile * cutlass.Int32(tile_kv)

    # ---- Causal compute-skip: a kv-tile [kv_base, kv_base+tile_kv) is attended
    #      only by q >= kv_base (top-left causal) or q >= kv_base - causal_diag
    #      (bottom-right).  Start the q-loop at the first such q-tile and run a
    #      j-counter (q_iter = q_lo_tile + j) so the double-buffer stage parity is
    #      relative to the loop start, not the absolute q-tile.  ~2x on causal
    #      bprop.  Non-causal → q_lo_tile=0, n_iters=n_q_tiles (byte-identical).
    #      SKV>SQ edge: kv-tiles past SQ get q_lo_tile >= n_q_tiles → 0 iters →
    #      dV/dK epilogue writes 0 (those kv attend no q under causal).  The
    #      prologue's valid_rows row-gate makes the OOB prefetch a zero-size copy.
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        # A kv-tile [kv_base, kv_base+tile_kv) is attended by q whenever
        # kv <= q + causal_diag + right_bound (band widening), i.e. the first
        # attending q is (kv_base - causal_diag) - right_bound (BR) / kv_base -
        # right_bound (TL).  Subtract right_bound so band-right widening doesn't
        # drop the right_bound queries that straddle the previous kv-tile (else
        # their dQ + this tile's dK/dV lose those contributions).  right_bound==0
        # → byte-identical to the plain-causal skip.
        _q_lo_abs = ((kv_base - causal_diag) if cutlass.const_expr(causal_bottom_right) else kv_base) - right_bound
        _q_lo_t = _q_lo_abs // cutlass.Int32(tile_q)
        q_lo_tile = cutlass.Int32(arith.maxsi(_q_lo_t.ir_value(), cutlass.Int32(0).ir_value()))
    else:
        q_lo_tile = cutlass.Int32(0)
    n_iters = cutlass.Int32(arith.maxsi((n_q_tiles_eff - q_lo_tile).ir_value(), cutlass.Int32(0).ir_value()))
    # THD over-provisioned grid: this CTA's kv-tile may start past the packed
    # sequence's KV length (n_kv_tiles = ceil(max_skv/tile_kv) covers the longest
    # sequence).  Force 0 q-iters for such tiles → no compute, and the dV/dK
    # epilogue's kv<kv_bound gate suppresses every store.
    if cutlass.const_expr(THD_VARLEN):
        n_iters = cutlass.Int32(arith.select((kv_base >= s_kv_b).ir_value(), cutlass.Int32(0).ir_value(), n_iters.ir_value()))
    q_lo_base = q_lo_tile * cutlass.Int32(tile_q)

    # K/V tile (row 0) GMEM element pointers — at the KV head (GQA-mapped).
    # kv_row_origin = batch*SKV (dense) or cu_k[b] (THD packed).
    k_tile_base = ((kv_row_origin + kv_base) * cutlass.Int32(Hkv) + kv_head) * cutlass.Int32(d_qk)
    v_tile_base = ((kv_row_origin + kv_base) * cutlass.Int32(Hkv) + kv_head) * cutlass.Int32(d_v)
    k_tile_gmem = K_view.data_ptr() + k_tile_base
    v_tile_gmem = V_view.data_ptr() + v_tile_base

    # LSE / do_dot: dense [B,H,SQ] → (batch*H+head)*SQ; THD packed [1,H,T_q]
    # (T_q == SQ here) → head*T_q + cu_q[b].  Read below at + the in-seq q index.
    lse_head_base = (head * cutlass.Int32(SQ) + cu_q_b) if cutlass.const_expr(THD_VARLEN) else (batch * cutlass.Int32(H) + head) * cutlass.Int32(SQ)
    dot_head_base = lse_head_base

    # ---- SMEM tiles -------------------------------------------------------
    # Q∪dO is MERGED into one array so BMM1/dV/dK's per-sub-group B operand is a
    # runtime sg_id *offset* on one base (no `if is_sg0` around the mma) — d_qk==
    # d_v makes the two sub-groups structurally identical.  Layout:
    #   sQdO[t][stage] : t=0 Q, t=1 dO ; base = sQdO + (t*2 + stage)*QSTAGE
    # The BMM2 A operand (P for dV, dS for dK) is NOT staged through SMEM — it is
    # the BMM1 C-fragment, already in registers (built as half2 in softmax/dS),
    # and reused directly as the next mma's A (FA2 trick; same layout).  So only
    # sP (sg0→sg1, for dS) and sdSᵀ (transposed, for dQ) live in SMEM; sdS is gone.
    # Q and dO are staged in SEPARATE arrays (d_qk vs d_v wide) — the old merged
    # sQdO + runtime sg_id-offset trick required d_qk == d_v.  Each is a ring of
    # `qo_stages` tiles: stage s of Q at sQ + s*QSTAGE_Q, dO at sdO + s*QSTAGE_O.
    QSTAGE_Q = tile_q * d_qk
    QSTAGE_O = tile_q * d_v
    PT = tile_kv * tile_q
    sK = cutlass.Array(io_dtype, tile_kv * d_qk, alignment=128, space=cutlass.AddressSpace.smem)
    sV = cutlass.Array(io_dtype, tile_kv * d_v, alignment=128, space=cutlass.AddressSpace.smem)
    sQ = cutlass.Array(io_dtype, qo_stages * QSTAGE_Q, alignment=128, space=cutlass.AddressSpace.smem)
    sdO = cutlass.Array(io_dtype, qo_stages * QSTAGE_O, alignment=128, space=cutlass.AddressSpace.smem)
    sP = cutlass.Array(io_dtype, PT, alignment=128, space=cutlass.AddressSpace.smem)  # sg0→sg1
    sdST = cutlass.Array(io_dtype, tile_q * tile_kv, alignment=128, space=cutlass.AddressSpace.smem)
    # dQ atomicAdd-coalescing staging buffer (FP32 [tile_q, d_qk]).  The dQ MMA
    # C-fragment scatters across 8 rows per warp → 8 L2 sectors per atomic
    # request (NCU).  Staging the tile to SMEM, then having all threads atomicAdd
    # in row-major COALESCED order (consecutive lanes → consecutive dQ cols)
    # cuts it to 4 sectors/request — matching cuDNN's atomic pattern.  At large d
    # (dsv3 d_qk=192 / qwen d=256) this tile_q*d_qk*4 B buffer blows the 164 KiB
    # A100 cap, so `dq_smem_coalesce=False` drops it and atomicAdds the dQ frag
    # directly (scattered, slower, but correct + fits).  RoPE still needs SMEM, so
    # under rope we keep a small staging path even when coalesce is off (rope is
    # dense-only + d_qk≤128 there, where the buffer fits).
    sDQ = (
        cutlass.Array(cutlass.Float32, tile_q * d_qk, alignment=128, space=cutlass.AddressSpace.smem)
        if cutlass.const_expr(dq_smem_coalesce or has_rope)
        else None
    )

    # ---- load K, V once (all threads cooperate) ---------------------------
    # Partial last kv-tile (or short THD seq): gate rows >= kv_bound → zero-fill
    # SMEM (no GMEM read).  Zero-filled K/V keep dP = V·dOᵀ finite (0) for padded
    # kv-rows so dQ's dSᵀ·K term over those rows is a clean 0 (no NaN leak into
    # valid dQ).  Under THD row_base is the packed in-sequence row (kv_base).
    kv_valid = kv_bound if cutlass.const_expr(GATE_KV) else None
    kv_rbase = kv_base if cutlass.const_expr(GATE_KV) else None
    load_tile_2d(
        sK,
        k_tile_gmem,
        rows=tile_kv,
        elems_per_row=d_qk,
        gmem_row_stride_elems=K_RS,
        tidx=tidx,
        num_threads=threads,
        elems_per_copy=_COPY_ELEMS,
        elem_bytes=_ELEM_BYTES,
        swizzle=True,
        valid_rows=kv_valid,
        row_base=kv_rbase,
    )
    load_tile_2d(
        sV,
        v_tile_gmem,
        rows=tile_kv,
        elems_per_row=d_v,
        gmem_row_stride_elems=VV_RS,
        tidx=tidx,
        num_threads=threads,
        elems_per_copy=_COPY_ELEMS,
        elem_bytes=_ELEM_BYTES,
        swizzle=True,
        valid_rows=kv_valid,
        row_base=kv_rbase,
    )
    cp_async_commit()
    cp_async_wait(0)
    nvvm.barrier_cta_sync()

    # ---- RoPE: rotate K in SMEM once (before a_persist).  Q is rotated per
    #      Q-iter (positions differ per tile).  dQ/dK are un-rotated before
    #      store (gradient contract = w.r.t. UN-rotated Q/K).  Folds out at
    #      has_rope=False. ------------------------------------------------------
    if cutlass.const_expr(has_rope):
        rope_cs_ptr = Pointer(cutlass.make_array_view(ROPE_CS).data_ptr(), dtype=cutlass.Float32)
        rope_rotate_smem_tile(sK, rope_cs_ptr, kv_base, rows=tile_kv, d_qk=d_qk, tidx=tidx, threads=threads, io_dtype=io_dtype, elem_bytes=_ELEM_BYTES)
        nvvm.barrier_cta_sync()

    # ---- persistent BMM1 A-fragment (sg0: K[kv,d]; sg1: V[kv,d]) ----------
    # Built once; reused every Q-iter.  Stored in a LOCAL array so it stays
    # visible across the runtime Q-loop and the sg branches.
    # Sized for the larger sub-group K-reduce (sg0: DQK_CHUNKS over d_qk; sg1:
    # DV_CHUNKS over d_v).  At d_qk != d_v the two differ; size to the max.
    A_CHUNKS = DQK_CHUNKS if cutlass.const_expr(DQK_CHUNKS >= DV_CHUNKS) else DV_CHUNKS
    a_persist = cutlass.Array(cutlass.Int32, A_CHUNKS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
    a_row = lane % 16
    a_col = lane // 16
    if is_sg0:
        for kc in cutlass.range_constexpr(DQK_CHUNKS):
            row = warp_local * 16 + a_row
            col = kc * 16 + a_col * 8
            ptr = sK.subview(row * cutlass.Int32(d_qk) + swizzle_xor_128b(row, col, elem_bytes=_ELEM_BYTES))
            v = nvvm.ldmatrix(ptr.data_ptr(), 4, nvvm.MMALayout.ROW)
            a_persist[kc * 4 + 0] = v[0]
            a_persist[kc * 4 + 1] = v[1]
            a_persist[kc * 4 + 2] = v[2]
            a_persist[kc * 4 + 3] = v[3]
    else:
        for kc in cutlass.range_constexpr(DV_CHUNKS):
            row = warp_local * 16 + a_row
            col = kc * 16 + a_col * 8
            ptr = sV.subview(row * cutlass.Int32(d_v) + swizzle_xor_128b(row, col, elem_bytes=_ELEM_BYTES))
            v = nvvm.ldmatrix(ptr.data_ptr(), 4, nvvm.MMALayout.ROW)
            a_persist[kc * 4 + 0] = v[0]
            a_persist[kc * 4 + 1] = v[1]
            a_persist[kc * 4 + 2] = v[2]
            a_persist[kc * 4 + 3] = v[3]

    # ---- accumulators (LOCAL, persistent across Q-loop) -------------------
    # acc_grad: sg0 → dV[kv, d_v]; sg1 → dK[kv, d_qk].  Sized for the larger
    # of the two N (d_qk != d_v on dsv3 → size to max; equal on llama/qwen).
    # One m-block, N//8 n_frags.
    GRAD_N = d_qk if cutlass.const_expr(d_qk >= d_v) else d_v
    GRAD_NFRAGS = GRAD_N // 8
    acc_grad = cutlass.Array(cutlass.Float32, GRAD_NFRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(GRAD_NFRAGS * 4):
        acc_grad[i] = cutlass.Float32(0.0)

    # acc1: BMM1 result — sg0 → S[kv,q]; sg1 → dP[kv,q].  N = tile_q.
    S_NFRAGS = tile_q // 8
    acc1 = cutlass.Array(cutlass.Float32, S_NFRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
    # bmm2_a: BMM2 A operand (P for dV / dS for dK) as half2 i32, built directly
    # from the BMM1 C-fragment in softmax/dS — reused as the next mma's A with
    # NO SMEM round-trip (FA2-style).  Layout = mma A-frag: [nf*2+0]=half2(top),
    # [nf*2+1]=half2(bot); mma_step(k_step=kc) reads slots kc*4.. (n_frags 2kc,2kc+1).
    bmm2_a = cutlass.Array(cutlass.Int32, S_NFRAGS * 2, alignment=16, space=cutlass.AddressSpace.rmem)
    # dQ partial — N = DQ_N per sub-group, tiled DQ_M_BLOCKS along M (q-rows).
    DQ_NFRAGS = DQ_N // 8
    dq_acc = cutlass.Array(cutlass.Float32, DQ_M_BLOCKS * DQ_NFRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)

    a_list = [a_persist[i] for i in range(A_CHUNKS * 4)]

    sg_d_base = sg_id * cutlass.Int32(DQ_N)  # dQ d-col base for this sub-group
    # Q/dO row-gate bound (per-seq under THD) + packed seq origin for the GMEM
    # base (q_row_origin = batch*SQ dense / cu_q[b] THD).
    SQ_rt = q_bound
    HD_QK = cutlass.Int32(H) * cutlass.Int32(d_qk)
    HD_V = cutlass.Int32(H) * cutlass.Int32(d_v)
    bhead_qk = (q_row_origin * cutlass.Int32(H) + head) * cutlass.Int32(d_qk)
    bhead_v = (q_row_origin * cutlass.Int32(H) + head) * cutlass.Int32(d_v)

    # Prologue: prefetch Q/dO tile 0 into ring stage 0.  Predicated row-gate
    # (rows >= SQ → zero-size cp.async) keeps the per-iter group count uniform
    # so the last-iter prefetch needs no OOB GMEM read / branch.
    # Q → sQ[stage0]; dO → sdO[stage0].  Separate arrays (d_qk vs d_v wide).
    # (Loads are inlined — the DSL forbids closures that capture runtime vars inside
    # the dynamic q-loop.)
    load_tile_2d(
        sQ,
        Q_view.data_ptr() + bhead_qk + q_lo_base * HD_QK,
        rows=tile_q,
        elems_per_row=d_qk,
        gmem_row_stride_elems=QK_RS,
        tidx=tidx,
        num_threads=threads,
        elems_per_copy=_COPY_ELEMS,
        elem_bytes=_ELEM_BYTES,
        swizzle=True,
        valid_rows=SQ_rt,
        row_base=q_lo_base,
    )
    load_tile_2d(
        sdO,
        dO_view.data_ptr() + bhead_v + q_lo_base * HD_V,
        rows=tile_q,
        elems_per_row=d_v,
        gmem_row_stride_elems=V_RS,
        tidx=tidx,
        num_threads=threads,
        elems_per_copy=_COPY_ELEMS,
        elem_bytes=_ELEM_BYTES,
        swizzle=True,
        valid_rows=SQ_rt,
        row_base=q_lo_base,
    )
    cp_async_commit()

    # =======================================================================
    # Q-loop.  qo_stages=2 → double-buffer (prefetch tile i+1 into the other
    # ring slot while computing i).  qo_stages=1 (big-d SMEM fit) → load tile i
    # into the single slot, wait, compute, then reload (a barrier guards reuse).
    # j-counter from 0; q_iter = q_lo_tile + j (causal skip) so the ring-stage
    # parity is relative to the loop start (prologue loaded stage 0 = q_lo_tile).
    # =======================================================================
    for j in cutlass.range(n_iters, unroll=1):
        q_iter = q_lo_tile + j
        q_base = q_iter * cutlass.Int32(tile_q)
        stage_cur = j % cutlass.Int32(qo_stages)
        stage_nxt = (j + cutlass.Int32(1)) % cutlass.Int32(qo_stages)
        # Per-sub-group B operand bases (separate sQ/sdO; d_qk != d_v safe):
        #   BMM1 B = Q(sg0, d_qk) / dO(sg1, d_v); dV/dK B = dO(sg0) / Q(sg1).
        q_cur = sQ.subview(stage_cur * cutlass.Int32(QSTAGE_Q))
        o_cur = sdO.subview(stage_cur * cutlass.Int32(QSTAGE_O))

        # ---- B0: prefetch tile i+1 (into the other stage), wait for tile i --
        nxt_qbase = q_base + cutlass.Int32(tile_q)
        if cutlass.const_expr(qo_stages == 2):
            load_tile_2d(
                sQ.subview(stage_nxt * cutlass.Int32(QSTAGE_Q)),
                Q_view.data_ptr() + bhead_qk + nxt_qbase * HD_QK,
                rows=tile_q,
                elems_per_row=d_qk,
                gmem_row_stride_elems=QK_RS,
                tidx=tidx,
                num_threads=threads,
                elems_per_copy=_COPY_ELEMS,
                elem_bytes=_ELEM_BYTES,
                swizzle=True,
                valid_rows=SQ_rt,
                row_base=nxt_qbase,
            )
            load_tile_2d(
                sdO.subview(stage_nxt * cutlass.Int32(QSTAGE_O)),
                dO_view.data_ptr() + bhead_v + nxt_qbase * HD_V,
                rows=tile_q,
                elems_per_row=d_v,
                gmem_row_stride_elems=V_RS,
                tidx=tidx,
                num_threads=threads,
                elems_per_copy=_COPY_ELEMS,
                elem_bytes=_ELEM_BYTES,
                swizzle=True,
                valid_rows=SQ_rt,
                row_base=nxt_qbase,
            )
            cp_async_commit()
            cp_async_wait(1)  # leave the just-issued prefetch in flight
        else:
            cp_async_wait(0)  # 1-stage: just wait for tile i
        nvvm.barrier_cta_sync()

        # ---- RoPE: rotate THIS Q tile (stage_cur) in place; dO is NOT rotated.
        #      Both sg0 (S=K'·Q'ᵀ) and sg1 (dK=dS·Q') read it. ----
        if cutlass.const_expr(has_rope):
            rope_rotate_smem_tile(q_cur, rope_cs_ptr, q_base, rows=tile_q, d_qk=d_qk, tidx=tidx, threads=threads, io_dtype=io_dtype, elem_bytes=_ELEM_BYTES)
            nvvm.barrier_cta_sync()

        # ---- BMM1: acc1 = A_persist · Bᵀ.  sg0 → S=K·Qᵀ (B=Q, d_qk, DQK_CHUNKS);
        # sg1 → dP=V·dOᵀ (B=dO, d_v, DV_CHUNKS).  Warp-uniform branch (sg_id is
        # per-warp) — needed because B width + K-reduce trip differ at d_qk != d_v.
        for i in cutlass.range_constexpr(S_NFRAGS * 4):
            acc1[i] = cutlass.Float32(0.0)
        if is_sg0:
            for kc in cutlass.range_constexpr(DQK_CHUNKS):
                b = load_b_smem_x4(q_cur, k_step=kc, N=tile_q, sB_elems_per_row=cutlass.Int32(d_qk), b_trans=False, lane=lane, swizzle=True)
                mma_step(acc1, a_list, b, k_step=kc, M=16, N=tile_q, ab_dtype=io_dtype)
        else:
            for kc in cutlass.range_constexpr(DV_CHUNKS):
                b = load_b_smem_x4(o_cur, k_step=kc, N=tile_q, sB_elems_per_row=cutlass.Int32(d_v), b_trans=False, lane=lane, swizzle=True)
                mma_step(acc1, a_list, b, k_step=kc, M=16, N=tile_q, ab_dtype=io_dtype)

        # ---- sg0: softmax recompute P = exp2(scale·S − LSE_q) → write sP ----
        if is_sg0:
            kv_row_g = warp_local * 16 + g_lane
            kv_row_g8 = kv_row_g + 8
            # Absolute kv indices for this lane's two fragment rows (nf-invariant).
            kv_a = kv_base + kv_row_g
            kv_a8 = kv_base + kv_row_g8
            # Clamped kv for the (partial last kv-tile) bias read so kv >= SKV
            # doesn't index past bias[.,.,.,SKV]; P is masked to 0 for kv >= SKV.
            # (bias is dense-only; kv_bound == SKV here.)
            ka = _clamp_lt(kv_a, kv_bound) if cutlass.const_expr(GATE_KV) else kv_a
            ka8 = _clamp_lt(kv_a8, kv_bound) if cutlass.const_expr(GATE_KV) else kv_a8
            for nf in cutlass.range_constexpr(S_NFRAGS):
                qc0 = q_base + cutlass.Int32(nf * 8 + 0) + cutlass.Int32(2) * p_lane
                qc1 = qc0 + cutlass.Int32(1)
                # Partial last q-tile (or short THD seq): clamp the q index for
                # GMEM reads (LSE / bias) to < q_bound so rows q >= S_q don't
                # read past the packed/dense LSE; P is masked to 0 below so the
                # clamped value is dead.
                qr0 = _clamp_lt(qc0, q_bound) if cutlass.const_expr(GATE_Q) else qc0
                qr1 = _clamp_lt(qc1, q_bound) if cutlass.const_expr(GATE_Q) else qc1
                lse0 = Pointer(LSE_view.data_ptr() + lse_head_base + qr0, dtype=cutlass.Float32).load() * cutlass.Float32(_LOG2E)
                lse1 = Pointer(LSE_view.data_ptr() + lse_head_base + qr1, dtype=cutlass.Float32).load() * cutlass.Float32(_LOG2E)
                off = nf * 4
                # Bias: add bias/scale to UNSCALED acc1 (post-scale → +bias),
                # matching the forward.  Cells: (kv_a,qc0)(kv_a,qc1)(kv_a8,qc0)(kv_a8,qc1).
                if cutlass.const_expr(has_bias):
                    _bdt = cutlass.Float32 if cutlass.const_expr(bias_is_fp32) else io_dtype
                    b00 = Pointer(_bias_ptr + bias_base + qr0 * cutlass.Int32(SKV) + ka, dtype=_bdt).load()
                    b01 = Pointer(_bias_ptr + bias_base + qr1 * cutlass.Int32(SKV) + ka, dtype=_bdt).load()
                    b10 = Pointer(_bias_ptr + bias_base + qr0 * cutlass.Int32(SKV) + ka8, dtype=_bdt).load()
                    b11 = Pointer(_bias_ptr + bias_base + qr1 * cutlass.Int32(SKV) + ka8, dtype=_bdt).load()
                    acc1[off + 0] = acc1[off + 0] + b00.to(cutlass.Float32) * inv_softmax_scale
                    acc1[off + 1] = acc1[off + 1] + b01.to(cutlass.Float32) * inv_softmax_scale
                    acc1[off + 2] = acc1[off + 2] + b10.to(cutlass.Float32) * inv_softmax_scale
                    acc1[off + 3] = acc1[off + 3] + b11.to(cutlass.Float32) * inv_softmax_scale
                p0 = cute.math.exp2(acc1[off + 0] * softmax_scale_log2 - lse0, fastmath=True)
                p1 = cute.math.exp2(acc1[off + 1] * softmax_scale_log2 - lse1, fastmath=True)
                p2 = cute.math.exp2(acc1[off + 2] * softmax_scale_log2 - lse0, fastmath=True)
                p3 = cute.math.exp2(acc1[off + 3] * softmax_scale_log2 - lse1, fastmath=True)
                # Fully-masked / padded q-rows have LSE = -inf (the forward writes
                # -inf for an empty softmax denom: padded rows, or BR-causal rows
                # that attend zero valid KV when eff_sq > eff_skv).  exp2(scale·S -
                # (-inf)) = +inf; the cell masks zero it, but do_dot/dS would still
                # see +inf on any unmasked cell of such a row.  Force P=0 directly
                # when LSE is non-finite so dS/dQ/dK can't pick up a +inf -> NaN.
                _f0 = lse0 > cutlass.Float32(-3.0e38)
                _f1 = lse1 > cutlass.Float32(-3.0e38)
                p0 = cutlass.Float32(arith.select(_f0.ir_value(), p0.ir_value(), cutlass.Float32(0.0).ir_value()))
                p2 = cutlass.Float32(arith.select(_f0.ir_value(), p2.ir_value(), cutlass.Float32(0.0).ir_value()))
                p1 = cutlass.Float32(arith.select(_f1.ir_value(), p1.ir_value(), cutlass.Float32(0.0).ir_value()))
                p3 = cutlass.Float32(arith.select(_f1.ir_value(), p3.ir_value(), cutlass.Float32(0.0).ir_value()))
                # Mask: zero P on masked cells (causal / SWA / padded).  dS/dK/dV/
                # dQ/dBias all ∝ P so they inherit the mask.  Folds out at NONE.
                if cutlass.const_expr(mask_flags != MASK_NONE):
                    _mk = dict(
                        mask_flags=mask_flags,
                        swa_window=swa_window,
                        causal_bottom_right=causal_bottom_right,
                        causal_diag=causal_diag,
                        eff_skv=eff_skv,
                        right_bound=right_bound,
                    )
                    p0 = _mask_p(p0, kv_a, qc0, **_mk)
                    p1 = _mask_p(p1, kv_a, qc1, **_mk)
                    p2 = _mask_p(p2, kv_a8, qc0, **_mk)
                    p3 = _mask_p(p3, kv_a8, qc1, **_mk)
                # Partial-tile / THD bounds: zero P for padded kv-rows
                # (kv >= S_kv) and padded q-cols (q >= S_q).  Keeps dV/dS/dBias
                # clean; the dQ dSᵀ·K term over padded kv is already 0 via the K
                # zero-fill.  Bounds are runtime (SQ/SKV dense, s_q_b/s_kv_b THD).
                if cutlass.const_expr(GATE_KV):
                    p0 = _zero_if_ge(p0, kv_a, kv_bound)
                    p1 = _zero_if_ge(p1, kv_a, kv_bound)
                    p2 = _zero_if_ge(p2, kv_a8, kv_bound)
                    p3 = _zero_if_ge(p3, kv_a8, kv_bound)
                if cutlass.const_expr(GATE_Q):
                    p0 = _zero_if_ge(p0, qc0, q_bound)
                    p1 = _zero_if_ge(p1, qc1, q_bound)
                    p2 = _zero_if_ge(p2, qc0, q_bound)
                    p3 = _zero_if_ge(p3, qc1, q_bound)
                h01 = fp32_to_fp16(p0, p1, dtype=io_dtype)  # row gid  (q 2p, 2p+1)
                h23 = fp32_to_fp16(p2, p3, dtype=io_dtype)  # row gid+8
                # BMM2 (dV) A-fragment, in registers (no ldmatrix round-trip).
                bmm2_a[nf * 2 + 0] = h01
                bmm2_a[nf * 2 + 1] = h23
                # sP write — sg1 still reads P here to form dS.
                col = cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane
                a_top = sP.subview(kv_row_g * cutlass.Int32(tile_q) + swizzle_xor_128b(kv_row_g, col, elem_bytes=_ELEM_BYTES))
                a_bot = sP.subview(kv_row_g8 * cutlass.Int32(tile_q) + swizzle_xor_128b(kv_row_g8, col, elem_bytes=_ELEM_BYTES))
                Pointer(a_top.data_ptr(), dtype=cutlass.Int32).store(h01, alignment=4)
                Pointer(a_bot.data_ptr(), dtype=cutlass.Int32).store(h23, alignment=4)

        # ---- B1: sP ready ---------------------------------------------------
        nvvm.barrier_cta_sync()

        # ---- sg1: dS = scale·(dP − do_dot_q)·P  → bmm2_a regs + sdSᵀ -------
        if not is_sg0:
            kv_row_g = warp_local * 16 + g_lane
            kv_row_g8 = kv_row_g + 8
            kv_a = kv_base + kv_row_g
            kv_a8 = kv_base + kv_row_g8
            for nf in cutlass.range_constexpr(S_NFRAGS):
                qc0 = q_base + cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane
                qc1 = qc0 + cutlass.Int32(1)
                # Clamp the do_dot read index to < q_bound on the partial last
                # q-tile / short THD seq (rows q >= S_q would read past the LSE).
                # P was masked to 0 for those rows so dS / dBias come out 0.
                qr0 = _clamp_lt(qc0, q_bound) if cutlass.const_expr(GATE_Q) else qc0
                qr1 = _clamp_lt(qc1, q_bound) if cutlass.const_expr(GATE_Q) else qc1
                dd0 = Pointer(DOT_view.data_ptr() + dot_head_base + qr0, dtype=cutlass.Float32).load()
                dd1 = Pointer(DOT_view.data_ptr() + dot_head_base + qr1, dtype=cutlass.Float32).load()
                col = cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane
                swz_top = swizzle_xor_128b(kv_row_g, col, elem_bytes=_ELEM_BYTES)
                swz_bot = swizzle_xor_128b(kv_row_g8, col, elem_bytes=_ELEM_BYTES)
                a_top = sP.subview(kv_row_g * cutlass.Int32(tile_q) + swz_top)
                a_bot = sP.subview(kv_row_g8 * cutlass.Int32(tile_q) + swz_bot)
                ptop = Pointer(a_top.data_ptr(), dtype=io_dtype).load(count=2)
                pbot = Pointer(a_bot.data_ptr(), dtype=io_dtype).load(count=2)
                p0 = ptop[0].to(cutlass.Float32)
                p1 = ptop[1].to(cutlass.Float32)
                p2 = pbot[0].to(cutlass.Float32)
                p3 = pbot[1].to(cutlass.Float32)
                off = nf * 4
                # Un-scaled softmax-input gradient = (dP − do_dot)·P.  This IS
                # dBias (bias adds post-scale → dBias = dS').  dS for dQ/dK folds
                # in attn_scale.  (cells: (kv_a,qc0)(kv_a,qc1)(kv_a8,qc0)(kv_a8,qc1))
                db0 = (acc1[off + 0] - dd0) * p0
                db1 = (acc1[off + 1] - dd1) * p1
                db2 = (acc1[off + 2] - dd0) * p2
                db3 = (acc1[off + 3] - dd1) * p3
                if cutlass.const_expr(has_bias):
                    if cutlass.const_expr(PARTIAL_Q or PARTIAL_KV):
                        # Padded cells have db == 0 (P masked) but their dBias
                        # GMEM slot is OOB ([.,H,SQ,SKV]) → row/col-gate the
                        # atomicAdd so no out-of-bounds write lands.
                        _q0 = qc0 < cutlass.Int32(SQ)
                        _q1 = qc1 < cutlass.Int32(SQ)
                        _k0 = kv_a < cutlass.Int32(SKV)
                        _k8 = kv_a8 < cutlass.Int32(SKV)
                        if _q0 & _k0:
                            _atomic_add_f32(_dbias_ptr + bias_base + qc0 * cutlass.Int32(SKV) + kv_a, db0)
                        if _q1 & _k0:
                            _atomic_add_f32(_dbias_ptr + bias_base + qc1 * cutlass.Int32(SKV) + kv_a, db1)
                        if _q0 & _k8:
                            _atomic_add_f32(_dbias_ptr + bias_base + qc0 * cutlass.Int32(SKV) + kv_a8, db2)
                        if _q1 & _k8:
                            _atomic_add_f32(_dbias_ptr + bias_base + qc1 * cutlass.Int32(SKV) + kv_a8, db3)
                    else:
                        _atomic_add_f32(_dbias_ptr + bias_base + qc0 * cutlass.Int32(SKV) + kv_a, db0)
                        _atomic_add_f32(_dbias_ptr + bias_base + qc1 * cutlass.Int32(SKV) + kv_a, db1)
                        _atomic_add_f32(_dbias_ptr + bias_base + qc0 * cutlass.Int32(SKV) + kv_a8, db2)
                        _atomic_add_f32(_dbias_ptr + bias_base + qc1 * cutlass.Int32(SKV) + kv_a8, db3)
                ds0 = attn_scale * db0
                ds1 = attn_scale * db1
                ds2 = attn_scale * db2
                ds3 = attn_scale * db3
                # BMM2 (dK) A-fragment, in registers (no SMEM round-trip for dS).
                bmm2_a[nf * 2 + 0] = fp32_to_fp16(ds0, ds1, dtype=io_dtype)
                bmm2_a[nf * 2 + 1] = fp32_to_fp16(ds2, ds3, dtype=io_dtype)
                # sdSᵀ [q, kv] — scalar fp16, the ONLY SMEM dS copy (dQ needs the
                # transpose); swizzled per (q, kv) so the dQ ldmatrix is conflict-free.
                q0 = cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane
                q1 = q0 + cutlass.Int32(1)
                Pointer(sdST.subview(q0 * cutlass.Int32(tile_kv) + swizzle_xor_128b(q0, kv_row_g, elem_bytes=_ELEM_BYTES)).data_ptr(), dtype=io_dtype).store(
                    ds0.to(io_dtype)
                )
                Pointer(sdST.subview(q1 * cutlass.Int32(tile_kv) + swizzle_xor_128b(q1, kv_row_g, elem_bytes=_ELEM_BYTES)).data_ptr(), dtype=io_dtype).store(
                    ds1.to(io_dtype)
                )
                Pointer(sdST.subview(q0 * cutlass.Int32(tile_kv) + swizzle_xor_128b(q0, kv_row_g8, elem_bytes=_ELEM_BYTES)).data_ptr(), dtype=io_dtype).store(
                    ds2.to(io_dtype)
                )
                Pointer(sdST.subview(q1 * cutlass.Int32(tile_kv) + swizzle_xor_128b(q1, kv_row_g8, elem_bytes=_ELEM_BYTES)).data_ptr(), dtype=io_dtype).store(
                    ds3.to(io_dtype)
                )

        # ---- BMM2: acc_grad += A(=P sg0 / dS sg1, FROM REGS) · Bᵀ.  sg0 → dV=Pᵀ·dO
        #      (B=dO=o_cur, N=d_v); sg1 → dK=dSᵀ·Q (B=Q=q_cur, N=d_qk).  A is bmm2_a
        #      (BMM1 C-frag reused — no SMEM round-trip).  K-reduce = Q_CHUNKS (over
        #      q) for both.  Warp-uniform branch: B + N differ at d_qk != d_v.
        a_bmm2 = [bmm2_a[i] for i in range(S_NFRAGS * 2)]
        if is_sg0:
            for kc in cutlass.range_constexpr(Q_CHUNKS):
                b = load_b_smem_x4(o_cur, k_step=kc, N=d_v, sB_elems_per_row=cutlass.Int32(d_v), b_trans=True, lane=lane, swizzle=True)
                mma_step(acc_grad, a_bmm2, b, k_step=kc, M=16, N=d_v, ab_dtype=io_dtype)
        else:
            for kc in cutlass.range_constexpr(Q_CHUNKS):
                b = load_b_smem_x4(q_cur, k_step=kc, N=d_qk, sB_elems_per_row=cutlass.Int32(d_qk), b_trans=True, lane=lane, swizzle=True)
                mma_step(acc_grad, a_bmm2, b, k_step=kc, M=16, N=d_qk, ab_dtype=io_dtype)

        # ---- B2: sdS / sdSᵀ ready -------------------------------------------
        nvvm.barrier_cta_sync()

        # ---- dQ += dSᵀ · K  (both sg; sg splits the d-cols, tiled DQ_M_BLOCKS
        #      along q-rows).  A = dSᵀ[q,kv] (ldmatrix per m-block), B = K half-
        #      column.  Each warp owns rows mb*M_STRIDE + warp_local*16 (+ a_row)
        #      for mb in 0..DQ_M_BLOCKS — at tile_q=64 (llama) DQ_M_BLOCKS=1 so
        #      this is the byte-identical single-m-block path.
        for i in cutlass.range_constexpr(DQ_M_BLOCKS * DQ_NFRAGS * 4):
            dq_acc[i] = cutlass.Float32(0.0)
        for kc in cutlass.range_constexpr(KV_CHUNKS):
            adst = [None] * (DQ_M_BLOCKS * 4)
            col = kc * 16 + a_col * 8
            for mb in cutlass.range_constexpr(DQ_M_BLOCKS):
                row = cutlass.Int32(mb * M_STRIDE) + warp_local * 16 + a_row
                vv = nvvm.ldmatrix(
                    sdST.subview(row * cutlass.Int32(tile_kv) + swizzle_xor_128b(row, col, elem_bytes=_ELEM_BYTES)).data_ptr(), 4, nvvm.MMALayout.ROW
                )
                adst[mb * 4 + 0] = vv[0]
                adst[mb * 4 + 1] = vv[1]
                adst[mb * 4 + 2] = vv[2]
                adst[mb * 4 + 3] = vv[3]
            # dQ B = K[kv, sg_d_base : sg_d_base+DQ_N] — a half-column slice of
            # the Swz128B sK tile.  Pass the d-col offset as `col_base` (swizzle
            # on the TRUE column) rather than offsetting sB_base, so the read is
            # correct for ANY d (a base-pointer offset only commutes with the
            # XOR when sg_d_base % 64 == 0, i.e. d_qk//2 % 64 == 0 — breaks at
            # d=64).  At d=128 (sg_d_base ∈ {0,64}) this is byte-identical.
            # B is SHARED across the DQ_M_BLOCKS m-blocks (one ldmatrix-B load).
            bk = load_b_smem_x4(sK, k_step=kc, N=DQ_N, sB_elems_per_row=cutlass.Int32(d_qk), b_trans=True, lane=lane, swizzle=True, col_base=sg_d_base)
            mma_step(dq_acc, adst, bk, k_step=0, M=16 * DQ_M_BLOCKS, N=DQ_N, ab_dtype=io_dtype)

        # ---- dQ atomicAdd ---------------------------------------------------
        # Deterministic relay (acquire): wait until every lower kv-tile has added
        # its contribution to THIS (seq,head,q_tile) slot, so the fp32 atomicAdds
        # land in fixed kv order → bitwise-reproducible dQ.  q-skip-safe: q_lo_tile
        # is monotonic in kv_tile, so every kv' < kv_tile that reaches this q-tile
        # relays before us → the wait target is exactly kv_tile.  Folds out (byte-
        # identical fast path) when deterministic=False.
        if cutlass.const_expr(deterministic):
            sem_idx = (batch * cutlass.Int32(H) + head) * sem_q_stride + q_iter
            sem_ptr = Pointer(cutlass.make_array_view(DQ_SEM).data_ptr() + sem_idx, dtype=cutlass.Int32)
            if tidx == cutlass.Int32(0):
                _dq_sem_wait(sem_ptr, kv_tile)
            nvvm.barrier_cta_sync()
        if cutlass.const_expr(dq_smem_coalesce or has_rope):
            # COALESCED via SMEM staging (sDQ).  Stage the dq_acc C-fragment into
            # sDQ[tile_q, d_qk] (each sg writes its d-col half).  The frag layout
            # scatters (8 rows / warp) — cheap SMEM scatter.  Then ALL threads
            # atomicAdd sDQ → GMEM dQ in row-major order: consecutive lanes hit
            # consecutive dQ columns, so each warp's atomic request coalesces to
            # 4 L2 sectors (vs 8 direct) → halves the dQ atomic L2 traffic.  Also
            # the ONLY path that supports RoPE un-rotate (needs SMEM scratch).
            for mb in cutlass.range_constexpr(DQ_M_BLOCKS):
                q_row_g = cutlass.Int32(mb * M_STRIDE) + warp_local * 16 + g_lane
                q_row_g8 = q_row_g + 8
                for nf in cutlass.range_constexpr(DQ_NFRAGS):
                    col = sg_d_base + cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane
                    off = (mb * DQ_NFRAGS + nf) * 4
                    r0 = q_row_g * cutlass.Int32(d_qk) + col
                    r1 = q_row_g8 * cutlass.Int32(d_qk) + col
                    Pointer(sDQ.subview(r0).data_ptr(), dtype=cutlass.Float32).store(dq_acc[off + 0])
                    Pointer(sDQ.subview(r0 + cutlass.Int32(1)).data_ptr(), dtype=cutlass.Float32).store(dq_acc[off + 1])
                    Pointer(sDQ.subview(r1).data_ptr(), dtype=cutlass.Float32).store(dq_acc[off + 2])
                    Pointer(sDQ.subview(r1 + cutlass.Int32(1)).data_ptr(), dtype=cutlass.Float32).store(dq_acc[off + 3])

            nvvm.barrier_cta_sync()  # sDQ fully staged (both sg)

            # ---- RoPE: un-rotate dQ' in sDQ (R(-pos_q)) before the atomicAdd. --
            if cutlass.const_expr(has_rope):
                _d2 = d_qk // 2
                _npairs = tile_q * _d2
                for u in cutlass.range_constexpr(_npairs // threads):
                    pid = tidx + cutlass.Int32(u * threads)
                    qrow = pid // cutlass.Int32(_d2)
                    i = pid % cutlass.Int32(_d2)
                    _cs = (q_base + qrow).to(cutlass.Int64) * cutlass.Int64(_d2 * 2) + i.to(cutlass.Int64) * cutlass.Int64(2)
                    c = rope_cs_ptr[_cs]
                    s = rope_cs_ptr[_cs + cutlass.Int64(1)]
                    lo_off = qrow * cutlass.Int32(d_qk) + i
                    hi_off = lo_off + cutlass.Int32(_d2)
                    lo = sDQ[lo_off]
                    hi = sDQ[hi_off]
                    sDQ[lo_off] = lo * c + hi * s  # un-rotate: sin sign flipped
                    sDQ[hi_off] = hi * c - lo * s
                nvvm.barrier_cta_sync()

            # Coalesced atomicAdd: thread t owns elements {t, t+threads, …}; a
            # warp's 32 lanes span 32 consecutive dQ cols → one coalesced request.
            for s in cutlass.range_constexpr((tile_q * d_qk) // threads):
                e = cutlass.Int32(s * threads) + tidx
                row = e // cutlass.Int32(d_qk)
                col = e % cutlass.Int32(d_qk)
                gaddr = ((q_row_origin + q_base + row) * cutlass.Int32(H) + head) * cutlass.Int32(d_qk) + col
                val = Pointer(sDQ.subview(e).data_ptr(), dtype=cutlass.Float32).load()
                # Partial last q-tile / short THD seq: skip rows q >= S_q (OOB).
                if cutlass.const_expr(GATE_Q):
                    if (q_base + row) < q_bound:
                        _atomic_add_f32(dQ_view.data_ptr() + gaddr, val)
                else:
                    _atomic_add_f32(dQ_view.data_ptr() + gaddr, val)
        else:
            # DIRECT scattered atomicAdd from the dq_acc C-frag (no sDQ — saves
            # tile_q*d_qk*4 B so big-d, e.g. dsv3 192 / qwen 256, fits 164 KiB).
            # 8 L2 sectors/req (vs 4 coalesced) — perf follow-up.  No RoPE here
            # (rope routes to the sDQ branch above).
            for mb in cutlass.range_constexpr(DQ_M_BLOCKS):
                q_row_g = cutlass.Int32(mb * M_STRIDE) + warp_local * 16 + g_lane
                q_row_g8 = q_row_g + 8
                for nf in cutlass.range_constexpr(DQ_NFRAGS):
                    col = sg_d_base + cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane
                    off = (mb * DQ_NFRAGS + nf) * 4
                    for qrow, va, vb in ((q_row_g, dq_acc[off + 0], dq_acc[off + 1]), (q_row_g8, dq_acc[off + 2], dq_acc[off + 3])):
                        g = ((q_row_origin + q_base + qrow) * cutlass.Int32(H) + head) * cutlass.Int32(d_qk) + col
                        if cutlass.const_expr(GATE_Q):
                            if (q_base + qrow) < q_bound:
                                _atomic_add_f32(dQ_view.data_ptr() + g, va)
                                _atomic_add_f32(dQ_view.data_ptr() + g + cutlass.Int32(1), vb)
                        else:
                            _atomic_add_f32(dQ_view.data_ptr() + g, va)
                            _atomic_add_f32(dQ_view.data_ptr() + g + cutlass.Int32(1), vb)

        # ---- Deterministic relay (release): all of this kv-tile's dQ atomicAdds
        #      to the slot have completed (atom.global.add returns → done at L2);
        #      the CTA barrier orders all 256 threads' adds before we publish, and
        #      the GPU acq_rel fence makes them visible before the next kv-tile's
        #      acquire returns.  Hand the token to kv_tile+1. ---------------------
        if cutlass.const_expr(deterministic):
            nvvm.barrier_cta_sync()
            cute.arch.fence_acq_rel_gpu()
            if tidx == cutlass.Int32(0):
                cute.arch.atomic_exch(sem_ptr, kv_tile + cutlass.Int32(1), sem="release", scope="gpu")

        # ---- B3: before next Q-iter overwrites the ring stage ---------------
        nvvm.barrier_cta_sync()

        # 1-stage: reload the NEXT tile into the single slot now that every warp
        # has finished reading stage_cur (B3 barrier above).  2-stage already
        # prefetched it at B0.  Guard j+1 < n_iters so the last iter issues no
        # OOB load (the row-gate would zero-size it anyway, but skip for clarity).
        if cutlass.const_expr(qo_stages == 1):
            if (j + cutlass.Int32(1)) < n_iters:
                load_tile_2d(
                    sQ,
                    Q_view.data_ptr() + bhead_qk + nxt_qbase * HD_QK,
                    rows=tile_q,
                    elems_per_row=d_qk,
                    gmem_row_stride_elems=QK_RS,
                    tidx=tidx,
                    num_threads=threads,
                    elems_per_copy=_COPY_ELEMS,
                    elem_bytes=_ELEM_BYTES,
                    swizzle=True,
                    valid_rows=SQ_rt,
                    row_base=nxt_qbase,
                )
                load_tile_2d(
                    sdO,
                    dO_view.data_ptr() + bhead_v + nxt_qbase * HD_V,
                    rows=tile_q,
                    elems_per_row=d_v,
                    gmem_row_stride_elems=V_RS,
                    tidx=tidx,
                    num_threads=threads,
                    elems_per_copy=_COPY_ELEMS,
                    elem_bytes=_ELEM_BYTES,
                    swizzle=True,
                    valid_rows=SQ_rt,
                    row_base=nxt_qbase,
                )
                cp_async_commit()

    # Drain the last iter's (clamped/predicated) trailing prefetch.
    cp_async_wait(0)

    # =======================================================================
    # Epilogue — write dV (sg0) / dK (sg1).  CTA owns its KV-tile → no atomics.
    # =======================================================================
    kv_row_g = warp_local * 16 + g_lane
    kv_row_g8 = kv_row_g + 8
    # Partial last kv-tile / short THD seq: gate dV/dK stores so padded rows
    # kv >= S_kv (which hold P·dO / dS·Q of zero-filled K/V = garbage) don't
    # write OOB into the packed/dense grad buffers.  Also covers the THD
    # over-provisioned tiles (kv_base >= s_kv_b → all rows gated, no store).
    # Predicate is nf-invariant (per kv-row).  kv_row_origin = batch*SKV (dense)
    # / cu_k[b] (THD packed).
    if cutlass.const_expr(GATE_KV):
        kv_top_ok = (kv_base + kv_row_g) < kv_bound
        kv_bot_ok = (kv_base + kv_row_g8) < kv_bound
    if is_sg0:
        for nf in cutlass.range_constexpr(d_v // 8):
            d0 = cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane
            off = nf * 4
            base_top = ((kv_row_origin + kv_base + kv_row_g) * cutlass.Int32(H) + head) * cutlass.Int32(d_v) + d0
            base_bot = ((kv_row_origin + kv_base + kv_row_g8) * cutlass.Int32(H) + head) * cutlass.Int32(d_v) + d0
            _vt = fp32_to_fp16(acc_grad[off + 0], acc_grad[off + 1], dtype=io_dtype)
            _vb = fp32_to_fp16(acc_grad[off + 2], acc_grad[off + 3], dtype=io_dtype)
            if cutlass.const_expr(GATE_KV):
                if kv_top_ok:
                    Pointer((dV_view.data_ptr() + base_top), dtype=cutlass.Int32).store(_vt, alignment=4)
                if kv_bot_ok:
                    Pointer((dV_view.data_ptr() + base_bot), dtype=cutlass.Int32).store(_vb, alignment=4)
            else:
                Pointer((dV_view.data_ptr() + base_top), dtype=cutlass.Int32).store(_vt, alignment=4)
                Pointer((dV_view.data_ptr() + base_bot), dtype=cutlass.Int32).store(_vb, alignment=4)
    else:
        # ---- RoPE: un-rotate dK' (R(-pos_kv)) IN-FRAGMENT before the store.
        #      Col i (<d2, n-frag nf<n_frags/2) pairs with col i+d2 (n-frag
        #      nf+n_frags/2) — both in this lane's acc_grad.  Un-rotation = the
        #      forward rotate with sin negated; cos/sin at (kv pos, i). ---------
        if cutlass.const_expr(has_rope):
            _d2 = d_qk // 2
            _half = d_qk // 16  # n_frags // 2
            for nf in cutlass.range_constexpr(_half):
                off = nf * 4
                off_hi = (nf + _half) * 4
                d0 = cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane  # rope angle index i (<d2)
                for ridx, krow in ((0, kv_row_g), (2, kv_row_g8)):
                    pos = (kv_base + krow).to(cutlass.Int64)
                    cs0 = pos * cutlass.Int64(_d2 * 2) + d0.to(cutlass.Int64) * cutlass.Int64(2)
                    c0 = rope_cs_ptr[cs0]
                    s0 = rope_cs_ptr[cs0 + cutlass.Int64(1)]
                    cs1 = cs0 + cutlass.Int64(2)
                    c1 = rope_cs_ptr[cs1]
                    s1 = rope_cs_ptr[cs1 + cutlass.Int64(1)]
                    lo0 = acc_grad[off + ridx]
                    hi0 = acc_grad[off_hi + ridx]
                    lo1 = acc_grad[off + ridx + 1]
                    hi1 = acc_grad[off_hi + ridx + 1]
                    acc_grad[off + ridx] = lo0 * c0 + hi0 * s0
                    acc_grad[off_hi + ridx] = hi0 * c0 - lo0 * s0
                    acc_grad[off + ridx + 1] = lo1 * c1 + hi1 * s1
                    acc_grad[off_hi + ridx + 1] = hi1 * c1 - lo1 * s1
        for nf in cutlass.range_constexpr(d_qk // 8):
            d0 = cutlass.Int32(nf * 8) + cutlass.Int32(2) * p_lane
            off = nf * 4
            base_top = ((kv_row_origin + kv_base + kv_row_g) * cutlass.Int32(H) + head) * cutlass.Int32(d_qk) + d0
            base_bot = ((kv_row_origin + kv_base + kv_row_g8) * cutlass.Int32(H) + head) * cutlass.Int32(d_qk) + d0
            _kt = fp32_to_fp16(acc_grad[off + 0], acc_grad[off + 1], dtype=io_dtype)
            _kb = fp32_to_fp16(acc_grad[off + 2], acc_grad[off + 3], dtype=io_dtype)
            if cutlass.const_expr(GATE_KV):
                if kv_top_ok:
                    Pointer((dK_view.data_ptr() + base_top), dtype=cutlass.Int32).store(_kt, alignment=4)
                if kv_bot_ok:
                    Pointer((dK_view.data_ptr() + base_bot), dtype=cutlass.Int32).store(_kb, alignment=4)
            else:
                Pointer((dK_view.data_ptr() + base_top), dtype=cutlass.Int32).store(_kt, alignment=4)
                Pointer((dK_view.data_ptr() + base_bot), dtype=cutlass.Int32).store(_kb, alignment=4)


@cute.jit
def _atomic_add_f32(addr_i64, val):
    """``atom.global.add.f32`` on a 32-bit GMEM float element.  ``addr_i64`` is
    an element pointer offset (Int64 byte address via the view's data_ptr +
    element offset); the returned old value is discarded."""
    inline_ptx(
        "atom.global.add.f32 $0, [$1], $2;",
        write_only_types=[cutlass.Float32],
        read_only_args=[addr_i64, val],
    )


@cute.jit
def _dq_sem_wait(sem_ptr, target):
    """Spin (acquire) on an int32 GMEM semaphore until it reads ``target``.

    The FlashAttention-2 deterministic-dQ relay: every kv-tile CTA adds its dQ
    contribution to a (seq,head,q_tile) slot in strict ``kv_tile`` order, gated by
    this counter.  ``sem_ptr`` is a ``Pointer(Int32)`` to the slot.  Ordering by
    ``kv_tile == blockIdx.x`` (deterministic 3-D grid) makes the awaited
    predecessor strictly lower-blockIdx (scheduled first) → deadlock-free."""
    cur = cute.arch.atomic_add(sem_ptr, cutlass.Int32(0), sem="acquire", scope="gpu")
    wl = cutlass.while_generate([cur], lambda c: c != target)
    with wl as [c]:
        c = cute.arch.atomic_add(sem_ptr, cutlass.Int32(0), sem="acquire", scope="gpu")
        cutlass.yield_out([c])


# ===========================================================================
# do_dot prologue: delta[b,h,q] = sum_d O[b,q,h,d] * dO[b,q,h,d]  (raw, f32).
# One WARP per (b,h,q) row: the 32 lanes stride the contiguous D vector
# (lane l reads d = l, l+32, …) so the GMEM reads are coalesced, then a
# butterfly shfl reduces the per-lane partials.  (The old 1-thread-per-row
# form was uncoalesced — adjacent threads read rows d_v apart → ~10% BW SOL.)
# ===========================================================================
_DODOT_WARPS = 8  # warps per block (8 rows/block at 256 threads)


@cute.kernel
def _do_dot_kernel(
    O: cute.Tensor,  # [B, SQ, H, D_V] io_dtype (BSHD)
    dO: cute.Tensor,  # [B, SQ, H, D_V] io_dtype
    DELTA: cute.Tensor,  # [B, H, SQ] fp32 (output, = bprop "do_dot")
    d_v: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    n_rows: cutlass.Int32,  # B * H * SQ
):
    bx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // 32
    lane = tidx % 32
    row = bx * cutlass.Int32(_DODOT_WARPS) + warp
    if row < n_rows:
        SQ = O.shape[1]
        H = O.shape[2]
        HSQ = cutlass.Int32(H) * cutlass.Int32(SQ)
        b = row // HSQ
        r = row % HSQ
        h = r // cutlass.Int32(SQ)
        q = r % cutlass.Int32(SQ)
        in_base = ((b * cutlass.Int32(SQ) + q) * cutlass.Int32(H) + h) * cutlass.Int32(d_v)
        Ob = cutlass.make_array_view(O).data_ptr()
        dOb = cutlass.make_array_view(dO).data_ptr()
        acc = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr(d_v // 32):
            idx = in_base + lane + cutlass.Int32(k * 32)  # coalesced across lanes
            o = Pointer(Ob + idx, dtype=io_dtype).load()
            d = Pointer(dOb + idx, dtype=io_dtype).load()
            acc = acc + o.to(cutlass.Float32) * d.to(cutlass.Float32)
        # Warp butterfly reduce → all lanes hold the row sum; lane 0 stores.
        for off in cutlass.range_constexpr(5):
            acc = acc + nvvm.shfl_sync(0xFFFFFFFF, acc, 1 << (4 - off), 0x1F, nvvm.Shfl.BFLY)
        if lane == cutlass.Int32(0):
            Pointer(cutlass.make_array_view(DELTA).data_ptr() + row, dtype=cutlass.Float32).store(acc)


# ===========================================================================
# Attention-sink gradient: dSink_h = -Σ_{b,q} exp(sink_h - lse[b,h,q]) *
# do_dot[b,h,q].  Standalone (dQ/dK/dV are already sink-correct from the
# sink-aware LSE — no main-kernel change).  One warp per (b,h) row; lanes
# stride q; warp-reduce; lane 0 atomicAdds -acc into dSink[h] (sums over b).
# ===========================================================================
@cute.kernel
def _dsink_kernel(
    LSE: cute.Tensor,  # [B, H, SQ] fp32 (sink-aware, natural-log)
    DO_DOT: cute.Tensor,  # [B, H, SQ] fp32
    SINKS: cute.Tensor,  # [H] fp32 (natural-log sink logits)
    DSINK: cute.Tensor,  # [H] fp32 (atomicAdd target; zero-init)
    SQ: cutlass.Constexpr[int],
    n_bh: cutlass.Int32,  # B * H
):
    bx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // 32
    lane = tidx % 32
    row = bx * cutlass.Int32(_DODOT_WARPS) + warp  # = b*H + h
    if row < n_bh:
        H = SINKS.shape[0]
        h = row % cutlass.Int32(H)
        sink_h = Pointer(cutlass.make_array_view(SINKS).data_ptr() + h, dtype=cutlass.Float32).load()
        base = row * cutlass.Int32(SQ)
        lse_p = cutlass.make_array_view(LSE).data_ptr()
        dot_p = cutlass.make_array_view(DO_DOT).data_ptr()
        acc = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr((SQ + 31) // 32):
            q = lane + cutlass.Int32(k * 32)
            if q < cutlass.Int32(SQ):
                lse_q = Pointer(lse_p + base + q, dtype=cutlass.Float32).load()
                dd_q = Pointer(dot_p + base + q, dtype=cutlass.Float32).load()
                # Padded / fully-masked q-rows have LSE = -inf (forward writes -inf
                # for an empty softmax denom).  exp2((sink - (-inf))·log2e) = +inf →
                # NaN dSink.  Those rows don't exist (or attend nothing) → contribute
                # 0.  Select the term to 0 when LSE is non-finite.
                term = cute.math.exp2((sink_h - lse_q) * cutlass.Float32(_LOG2E), fastmath=True) * dd_q
                acc = acc + cutlass.Float32(arith.select((lse_q > cutlass.Float32(-3.0e38)).ir_value(), term.ir_value(), cutlass.Float32(0.0).ir_value()))
        for off in cutlass.range_constexpr(5):
            acc = acc + nvvm.shfl_sync(0xFFFFFFFF, acc, 1 << (4 - off), 0x1F, nvvm.Shfl.BFLY)
        if lane == cutlass.Int32(0):
            _atomic_add_f32(cutlass.make_array_view(DSINK).data_ptr() + h, -acc)


# ===========================================================================
# dQ_acc (fp32) → dQ (io_dtype) cast kernel.
# ===========================================================================
@cute.kernel
def _cast_kernel(
    dQ_acc: cute.Tensor,  # [B, SQ, H, D_QK] fp32
    dQ_out: cute.Tensor,  # [B, SQ, H, D_QK] io_dtype
    io_dtype: cutlass.Constexpr,
    n_vecs: cutlass.Int32,  # total / 2  (each thread casts a half2)
):
    bx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    gid = bx * cutlass.Int32(256) + tidx
    if gid < n_vecs:
        src = cutlass.make_array_view(dQ_acc).data_ptr()
        dst = cutlass.make_array_view(dQ_out).data_ptr()
        v = Pointer(src + gid * cutlass.Int32(2), dtype=cutlass.Float32).load(count=2)
        Pointer(dst + gid * cutlass.Int32(2), dtype=cutlass.Int32).store(fp32_to_fp16(v[0], v[1], dtype=io_dtype), alignment=4)


# ===========================================================================
# GQA dK/dV head reduction: sum the per-query-head workspace over each
# query-head group → [B, SKV, Hk, d].  IN[B,SKV,H,d] (io_dtype) → OUT[B,SKV,Hk,d].
# One thread per OUT element; fp32 accumulate over the `ratio` group heads
# (spaced `d` apart in the [.,.,H,d] layout).
# ===========================================================================
@cute.kernel
def _dkv_reduce_kernel(
    IN: cute.Tensor,  # [B, SKV, H, d] io_dtype (per-query-head dK/dV)
    OUT: cute.Tensor,  # [B, SKV, Hk, d] io_dtype (reduced)
    d: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    Hk: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    n_out: cutlass.Int32,  # B * SKV * Hk * d
):
    ratio = H // Hk
    bx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    e = bx * cutlass.Int32(256) + tidx
    if e < n_out:
        di = e % cutlass.Int32(d)
        rest = e // cutlass.Int32(d)
        hk = rest % cutlass.Int32(Hk)
        rest2 = rest // cutlass.Int32(Hk)  # = b*SKV + s
        in_base = (rest2 * cutlass.Int32(H) + hk * cutlass.Int32(ratio)) * cutlass.Int32(d) + di
        src = cutlass.make_array_view(IN).data_ptr()
        acc = cutlass.Float32(0.0)
        for g in cutlass.range_constexpr(ratio):
            acc = acc + Pointer(src + in_base + cutlass.Int32(g * d), dtype=io_dtype).load().to(cutlass.Float32)
        Pointer(cutlass.make_array_view(OUT).data_ptr() + e, dtype=io_dtype).store(acc.to(io_dtype))


# ===========================================================================
# Hosts.
# ===========================================================================
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
    SEQ_KV_LENS: cute.Tensor,
    BIAS: cute.Tensor,
    DBIAS: cute.Tensor,
    ROPE_CS: cute.Tensor,
    CU_Q: cute.Tensor,
    CU_K: cute.Tensor,
    SEQ_LEN_Q: cute.Tensor,
    DQ_SEM: cute.Tensor,
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
    deterministic: cutlass.Constexpr[bool],
    sched_policy: cutlass.Constexpr[int],
    n_q_tiles: cutlass.Int32,
    softmax_scale_log2: cutlass.Float32,
    attn_scale: cutlass.Float32,
    right_bound: cutlass.Int32,
    inv_softmax_scale: cutlass.Float32,
    bias_bstride: cutlass.Int32,
    sem_q_stride: cutlass.Int32,
    grid_kv_tiles: cutlass.Int32,
    grid_batch: cutlass.Int32,
    stream: cuda.CUstream,
):
    SKV = K.shape[1]
    H = Q.shape[2]
    B = Q.shape[0]
    threads = 2 * warps_per_sg * 32
    # Dense grid is shape-derived; THD over-provisions to (ceil(max_skv/tile_kv),
    # H, n_seq) from host-computed runtime counts (packed B == 1).  THD uses the
    # 3-D grid only (SCHED_DEFAULT); LPT+THD is a future scheduler tweak.
    if cutlass.const_expr(thd_varlen):
        grid = (grid_kv_tiles, H, grid_batch)
    elif cutlass.const_expr(sched_policy == SCHED_LPT):
        n_kv_tiles = (SKV + tile_kv - 1) // tile_kv
        grid = (n_kv_tiles * H * B, 1, 1)
    else:
        n_kv_tiles = (SKV + tile_kv - 1) // tile_kv
        grid = (n_kv_tiles, H, B)
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
        deterministic,
        sched_policy,
        n_q_tiles,
        softmax_scale_log2,
        attn_scale,
        right_bound,
        inv_softmax_scale,
        bias_bstride,
        sem_q_stride,
    ).launch(grid=grid, block=(threads, 1, 1), stream=stream)


@cute.jit
def _do_dot_host(
    O: cute.Tensor,
    dO: cute.Tensor,
    DELTA: cute.Tensor,
    d_v: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    n_rows: cutlass.Int32,
    stream: cuda.CUstream,
):
    n_blocks = (n_rows + _DODOT_WARPS - 1) // _DODOT_WARPS
    _do_dot_kernel(O, dO, DELTA, d_v, io_dtype, n_rows).launch(grid=(n_blocks, 1, 1), block=(_DODOT_WARPS * 32, 1, 1), stream=stream)


@cute.jit
def _cast_host(
    dQ_acc: cute.Tensor,
    dQ_out: cute.Tensor,
    io_dtype: cutlass.Constexpr,
    n_vecs: cutlass.Int32,
    stream: cuda.CUstream,
):
    n_blocks = (n_vecs + 255) // 256
    _cast_kernel(dQ_acc, dQ_out, io_dtype, n_vecs).launch(grid=(n_blocks, 1, 1), block=(256, 1, 1), stream=stream)


@cute.jit
def _dkv_reduce_host(
    IN: cute.Tensor,
    OUT: cute.Tensor,
    d: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    Hk: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    n_out: cutlass.Int32,
    stream: cuda.CUstream,
):
    n_blocks = (n_out + 255) // 256
    _dkv_reduce_kernel(IN, OUT, d, H, Hk, io_dtype, n_out).launch(grid=(n_blocks, 1, 1), block=(256, 1, 1), stream=stream)


@cute.jit
def _dsink_host(
    LSE: cute.Tensor,
    DO_DOT: cute.Tensor,
    SINKS: cute.Tensor,
    DSINK: cute.Tensor,
    SQ: cutlass.Constexpr[int],
    n_bh: cutlass.Int32,
    stream: cuda.CUstream,
):
    n_blocks = (n_bh + _DODOT_WARPS - 1) // _DODOT_WARPS
    _dsink_kernel(LSE, DO_DOT, SINKS, DSINK, SQ, n_bh).launch(grid=(n_blocks, 1, 1), block=(_DODOT_WARPS * 32, 1, 1), stream=stream)


# ===========================================================================
# Compile cache.
# ===========================================================================
@lru_cache(maxsize=None)
def _compile_main(
    B,
    H,
    SQ,
    SKV,
    d_qk,
    d_v,
    tile_kv,
    tile_q,
    warps_per_sg,
    io_is_bf16,
    qo_stages=2,
    dq_smem_coalesce=True,
    Hk=None,
    mask_flags=MASK_NONE,
    swa_window=0,
    causal_bottom_right=0,
    has_seq_kv_lens=False,
    has_bias=False,
    bias_is_fp32=True,
    bias_batch=1,
    has_rope=False,
    rope_max_s=1,
    has_seq_len_q=False,
    thd_varlen=False,
    n_seq=1,
    deterministic=False,
    sem_units=1,
    sched_policy=SCHED_DEFAULT,
):
    io_dtype = cutlass.BFloat16 if io_is_bf16 else cutlass.Float16
    if Hk is None:
        Hk = H
    # Q/dO/dQ + dK/dV WORKSPACE carry H_q heads; K/V READ at H_kv heads (GQA).
    fq = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SQ, H, d_qk), stride_order=(3, 2, 1, 0), assumed_align=16)
    fk = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SKV, Hk, d_qk), stride_order=(3, 2, 1, 0), assumed_align=16)
    fv = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SKV, Hk, d_v), stride_order=(3, 2, 1, 0), assumed_align=16)
    fdo = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SQ, H, d_v), stride_order=(3, 2, 1, 0), assumed_align=16)
    fdq = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (B, SQ, H, d_qk), stride_order=(3, 2, 1, 0), assumed_align=16)
    fdk = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SKV, H, d_qk), stride_order=(3, 2, 1, 0), assumed_align=16)
    fdv = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SKV, H, d_v), stride_order=(3, 2, 1, 0), assumed_align=16)
    fl = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (B, H, SQ), stride_order=(2, 1, 0), assumed_align=16)
    fdt = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (B, H, SQ), stride_order=(2, 1, 0), assumed_align=16)
    # seq_kv_lens [B] int32 — a real tensor only when PADDED; a 1-elem dummy
    # otherwise (the kernel never reads it when has_seq_kv_lens is False).
    fsk = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (B if has_seq_kv_lens else 1,), stride_order=(0,), assumed_align=16)
    # BIAS / DBIAS [bias_batch, H, SQ, SKV] (bias_batch ∈ {1 (broadcast), B}); a
    # 1-elem dummy when has_bias is False.  bias_dt = fp32 or io_dtype.
    bias_dt = cutlass.Float32 if bias_is_fp32 else io_dtype
    if has_bias:
        fbias = cute.runtime.make_fake_compact_tensor(bias_dt, (bias_batch, H, SQ, SKV), stride_order=(3, 2, 1, 0), assumed_align=16)
        fdbias = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (bias_batch, H, SQ, SKV), stride_order=(3, 2, 1, 0), assumed_align=16)
    else:
        fbias = cute.runtime.make_fake_compact_tensor(bias_dt, (1,), stride_order=(0,), assumed_align=16)
        fdbias = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=16)
    # rope_cs [max_s, d_qk//2, 2] fp32 (cos,sin) — 1-elem dummy when no rope.
    if has_rope:
        frope = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (rope_max_s, d_qk // 2, 2), stride_order=(2, 1, 0), assumed_align=16)
    else:
        frope = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=16)
    # cu_q / cu_k [n_seq+1] int32 (THD packed cumulative seqlens); 1-elem dummy
    # otherwise (the kernel reads them only when thd_varlen).
    _cu_n = (n_seq + 1) if thd_varlen else 1
    fcuq = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (_cu_n,), stride_order=(0,), assumed_align=16)
    fcuk = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (_cu_n,), stride_order=(0,), assumed_align=16)
    # seq_len_q [B] int32 — real only for dense PADDED (per-batch q-pad); 1-elem
    # dummy otherwise (the kernel reads it only when has_seq_len_q).
    fsq = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (B if has_seq_len_q else 1,), stride_order=(0,), assumed_align=16)
    # DQ_SEM [n_seq*H*sem_q_stride] int32 deterministic-dQ relay counter; 1-elem
    # dummy when !deterministic (the kernel touches it only under deterministic).
    fsem = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sem_units if deterministic else 1,), stride_order=(0,), assumed_align=16)
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
        fsk,
        fbias,
        fdbias,
        frope,
        fcuq,
        fcuk,
        fsq,
        fsem,
        d_qk,
        d_v,
        tile_kv,
        tile_q,
        warps_per_sg,
        int(qo_stages),
        bool(dq_smem_coalesce),
        io_dtype,
        int(mask_flags),
        int(swa_window),
        int(causal_bottom_right),
        bool(has_seq_kv_lens),
        bool(has_bias),
        bool(bias_is_fp32),
        bool(has_rope),
        bool(has_seq_len_q),
        bool(thd_varlen),
        bool(deterministic),
        int(sched_policy),
        cutlass.Int32(0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cuda.CUstream(0),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=None)
def _compile_do_dot(B, H, SQ, d_v, io_is_bf16):
    io_dtype = cutlass.BFloat16 if io_is_bf16 else cutlass.Float16
    fo = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SQ, H, d_v), stride_order=(3, 2, 1, 0), assumed_align=16)
    fdo = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SQ, H, d_v), stride_order=(3, 2, 1, 0), assumed_align=16)
    fdt = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (B, H, SQ), stride_order=(2, 1, 0), assumed_align=16)
    return cute.compile(_do_dot_host, fo, fdo, fdt, d_v, io_dtype, cutlass.Int32(0), cuda.CUstream(0), options="--enable-tvm-ffi")


@lru_cache(maxsize=None)
def _compile_cast(B, H, SQ, d_qk, io_is_bf16):
    io_dtype = cutlass.BFloat16 if io_is_bf16 else cutlass.Float16
    fdq = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (B, SQ, H, d_qk), stride_order=(3, 2, 1, 0), assumed_align=16)
    fout = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SQ, H, d_qk), stride_order=(3, 2, 1, 0), assumed_align=16)
    return cute.compile(_cast_host, fdq, fout, io_dtype, cutlass.Int32(0), cuda.CUstream(0), options="--enable-tvm-ffi")


@lru_cache(maxsize=None)
def _compile_dkv_reduce(B, SKV, H, Hk, d, io_is_bf16):
    io_dtype = cutlass.BFloat16 if io_is_bf16 else cutlass.Float16
    fin = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SKV, H, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    fout = cute.runtime.make_fake_compact_tensor(io_dtype, (B, SKV, Hk, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    return cute.compile(_dkv_reduce_host, fin, fout, d, H, Hk, io_dtype, cutlass.Int32(0), cuda.CUstream(0), options="--enable-tvm-ffi")


@lru_cache(maxsize=None)
def _compile_dsink(B, H, SQ):
    fl = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (B, H, SQ), stride_order=(2, 1, 0), assumed_align=16)
    fdt = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (B, H, SQ), stride_order=(2, 1, 0), assumed_align=16)
    fsk = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H,), stride_order=(0,), assumed_align=16)
    fds = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H,), stride_order=(0,), assumed_align=16)
    return cute.compile(_dsink_host, fl, fdt, fsk, fds, SQ, cutlass.Int32(0), cuda.CUstream(0), options="--enable-tvm-ffi")


# ===========================================================================
# Python entry point.
# ===========================================================================
def backward(
    Q: torch.Tensor,  # [B, SQ,  H, D] io_dtype (BSHD)
    K: torch.Tensor,  # [B, SKV, H, D]
    V: torch.Tensor,  # [B, SKV, H, D]
    dO: torch.Tensor,  # [B, SQ,  H, D]
    O: torch.Tensor,  # [B, SQ,  H, D] forward output (for on-device do_dot)
    lse: torch.Tensor,  # [B, H, SQ] fp32 (forward log-sum-exp, natural-log)
    *,
    scale: Optional[float] = None,
    do_dot: Optional[torch.Tensor] = None,  # [B,H,SQ] fp32 — pass to skip the
    # on-device do_dot kernel (debug).
    mask: str = "none",  # none / causal / swa / causal_swa
    deterministic: bool = False,  # bitwise-reproducible dQ via a kv-ordered gmem-semaphore relay (forces SCHED_DEFAULT; perf-insensitive)
    sched: str = "auto",  # auto / default / lpt — kv-major LPT grid
    swa_window: int = 0,  # SWA window W (compile-time)
    right_bound: int = 0,  # causal right-band widening (k <= q+rb)
    causal_bottom_right: bool = False,  # bottom-right causal alignment
    seq_kv_lens: Optional[torch.Tensor] = None,  # [B] int32 per-batch KV length
    seq_len_q: Optional[torch.Tensor] = None,  # [B] int32 per-batch Q length (dense PADDED)
    bias: Optional[torch.Tensor] = None,  # [1|B, H, SQ, SKV] additive bias
    rope_freqs: Optional[torch.Tensor] = None,  # [max_s, .., d_qk] RoPE angles
    sinks: Optional[torch.Tensor] = None,  # [H] fp32 sink logits → dSink output
    cu_seqlens_q: Optional[torch.Tensor] = None,  # [n_seq+1] int32 cumulative Q seqlens (THD)
    cu_seqlens_k: Optional[torch.Tensor] = None,  # [n_seq+1] int32 cumulative KV seqlens (THD)
    tile_kv: int = _LLAMA_CFG.TILE_KV,
    tile_q: int = _LLAMA_CFG.TILE_Q,
    warps_per_sg: int = _LLAMA_CFG.WARPS_PER_SG,
):
    """Full SDPA backward — only ``O`` and ``lse`` (forward outputs) + ``dO`` are
    needed beyond Q/K/V.  ``do_dot`` is computed on-device from O·dO unless
    supplied.  ``mask`` / ``swa_window`` / ``right_bound`` / ``causal_bottom_right``
    / ``seq_kv_lens`` mirror the forward kernel's masking (P=0 on masked cells →
    dQ/dK/dV inherit it).  Returns (dQ, dK, dV) as io_dtype BSHD tensors."""
    assert Q.dtype in (torch.float16, torch.bfloat16)
    assert K.dtype == Q.dtype and V.dtype == Q.dtype and dO.dtype == Q.dtype and O.dtype == Q.dtype
    io_is_bf16 = Q.dtype == torch.bfloat16
    B, SQ, H, D = Q.shape
    _, SKV, Hk, D_K = K.shape
    _, _, Hv, _ = V.shape
    assert D_K == D, f"K head dim ({D_K}) must match Q head dim ({D}); every K offset is computed from Q's"
    d_qk, d_v = D, V.shape[3]
    # GQA / MQA: H query heads share Hk KV heads (H % Hk == 0).  Each query head
    # is processed by its own CTA; dK/dV are written per-query-head then summed
    # over the query-head group by a reduction kernel.  K and V must share the
    # same KV-head count (standard GQA).
    assert Hk == Hv, f"K/V head counts must match for GQA (got Hk={Hk}, Hv={Hv})"
    assert H % Hk == 0, f"H ({H}) must be divisible by KV heads ({Hk}) for GQA/MQA"
    gqa_ratio = H // Hk
    # d_qk may exceed d_v (dsv3 192/128); the two sub-groups are specialized per d.
    # Envelope: d_qk in {64,128,192,256}, d_v <= d_qk, both multiples of 16.
    assert d_qk >= d_v, f"bprop: d_qk ({d_qk}) must be >= d_v ({d_v})"
    assert d_qk % 16 == 0 and d_v % 16 == 0, f"bprop: d_qk/d_v must be mult of 16 (got {d_qk}/{d_v})"
    # SMEM fit on A100 (164 KiB): drop the dQ-coalescing sDQ staging once d_qk>128,
    # and the Q/dO double-buffer (qo_stages 2→1) once d_qk>=256 (qwen).  d<=128
    # (llama/gptoss) keeps both (byte-identical fast path).
    dq_smem_coalesce = d_qk <= 128
    qo_stages = 1 if d_qk >= 256 else 2
    # ---- THD / varlen: packed [1,T,H,D] Q/K/V/dO/O + cu_seqlens.  B==1 packed;
    #      n_seq logical sequences drive the grid + cu_* sizing.  Q.shape[1]/
    #      K.shape[1] are the packed totals T_q/T_kv (== the kernel's SQ/SKV). ---
    thd = cu_seqlens_q is not None
    if thd:
        assert cu_seqlens_k is not None, "THD needs both cu_seqlens_q and cu_seqlens_k"
        assert B == 1, f"THD: Q/K/V/dO/O must be packed [1,T,H,D]; got batch dim {B}"
        # GQA/MQA work under THD: the per-query-head dK_ws/dV_ws workspace
        # ([1,T_kv,H,d]) + the _dkv_reduce over the query-head group are
        # layout-agnostic (B=1, SKV=T_kv packed); the kv-tile CTA reads the
        # GQA-mapped kv_head and writes its query-head slice exactly as in the
        # dense path.  (Was conservatively asserted MHA-only.)
        cu_q_host = cu_seqlens_q.to(dtype=torch.int32, device="cpu")
        cu_k_host = cu_seqlens_k.to(dtype=torch.int32, device="cpu")
        n_seq = cu_q_host.numel() - 1
        assert cu_k_host.numel() == n_seq + 1, "cu_seqlens_q / cu_seqlens_k length mismatch"
        max_skv = int((cu_k_host[1:] - cu_k_host[:-1]).max())
        max_sq = int((cu_q_host[1:] - cu_q_host[:-1]).max())
    else:
        n_seq = B
        max_sq = SQ
    # ---- mask string → compile-time bitmask -------------------------------
    _MASK_TOK = {
        "none": MASK_NONE,
        "causal": MASK_CAUSAL,
        "swa": MASK_SWA,
        "causal_swa": MASK_CAUSAL | MASK_SWA,
    }
    assert mask in _MASK_TOK, f"backward: mask must be one of {list(_MASK_TOK)}; got {mask!r}"
    mask_flags = _MASK_TOK[mask]
    # Scheduler: kv-major LPT grid balances the causal load (light high-kv tiles
    # land in the last wave).  'auto' uses LPT only when causal (where the q-skip
    # makes work uneven); non-causal work is uniform so DEFAULT is fine.
    assert sched in ("auto", "default", "natural", "lpt"), f"backward: bad sched {sched!r}"
    if sched == "lpt":
        sched_policy = SCHED_LPT
    elif sched in ("default", "natural"):
        sched_policy = SCHED_DEFAULT
    else:  # auto
        sched_policy = SCHED_LPT if (mask_flags & MASK_CAUSAL) else SCHED_DEFAULT
    # Deterministic dQ orders the per-(seq,head,q_tile) atomicAdd by kv_tile == the
    # 3-D grid's blockIdx.x — so it REQUIRES the SCHED_DEFAULT decode (kv_tile=bx).
    # The kv-major LPT 1-D flat grid would remap kv_tile and break both the order
    # and the deadlock-freedom (predecessor = lower blockIdx).  Force it off.
    if deterministic:
        sched_policy = SCHED_DEFAULT
    has_seq_kv_lens = seq_kv_lens is not None
    has_seq_len_q = seq_len_q is not None
    if has_seq_kv_lens:
        mask_flags |= MASK_PADDED
    cbr = 1 if causal_bottom_right else 0
    has_bias = bias is not None
    if has_bias:
        assert bias.dim() == 4 and bias.shape[1] == H, f"bias must be [1|B, H, SQ, SKV]; got {tuple(bias.shape)}"
        bias_is_fp32 = bias.dtype == torch.float32
        bias_batch = bias.shape[0]  # 1 (broadcast over B) or B
        assert bias_batch in (1, B), f"bias batch dim must be 1 or B={B}; got {bias_batch}"
        bias_bstride = 0 if bias_batch == 1 else H * SQ * SKV
    else:
        bias_is_fp32, bias_batch, bias_bstride = True, 1, 0
    # RoPE: build the [max_s, d_qk//2, 2] (cos,sin) table from the angles
    # (mirrors the forward).  Backward rotates Q/K on load + un-rotates dQ/dK.
    has_rope = rope_freqs is not None
    if has_rope:
        assert seq_kv_lens is None, "RoPE is dense-only (no THD/padded) on SM80 bprop"
        d2 = d_qk // 2
        rf = rope_freqs.to(dtype=torch.float32, device=Q.device).reshape(rope_freqs.shape[0], -1)
        rope_max_s = rf.shape[0]
        assert rf.shape[1] >= d2, f"rope_freqs last dim ({rf.shape[1]}) must be >= d_qk//2 ({d2})"
        assert rope_max_s >= max(SQ, SKV), f"rope_freqs max_s ({rope_max_s}) must cover max(SQ={SQ}, SKV={SKV})"
        angles = rf[:, :d2]
        rope_cs_t = torch.stack([angles.cos(), angles.sin()], dim=-1).contiguous()
    else:
        rope_max_s = 1
        rope_cs_t = torch.zeros(1, dtype=torch.float32, device=Q.device)
    # Attention sink: dQ/dK/dV need NO kernel change (P recomputed from the
    # sink-aware LSE the caller passes); only dSink is computed (standalone).
    has_sink = sinks is not None
    if has_sink:
        sinks_t = sinks.to(dtype=torch.float32, device=Q.device).reshape(H).contiguous()
        dsink_t = torch.zeros(H, dtype=torch.float32, device=Q.device)
    # THD is dense-feature-only for now (bias/rope/sink/seq_kv_lens are
    # dense-only); per-sequence padding is handled by the packed bounds, not
    # the PADDED mask.  THD uses SCHED_DEFAULT (LPT+THD is a future tweak).
    if thd:
        assert not (
            has_bias or has_rope or has_sink or has_seq_kv_lens or has_seq_len_q
        ), "THD/varlen bprop: bias/rope/sink/seq_kv_lens/seq_len_q not supported yet"
        sched_policy = SCHED_DEFAULT
    # RoPE staging reuses the sDQ SMEM buffer, which the d_qk > 128 configs
    # drop (dq_smem_coalesce) to stay under the 164 KiB dynamic-SMEM cap —
    # re-enabling it via has_rope would exceed the budget at launch
    # (~48-64 KiB over at d=192/256).
    assert not (has_rope and d_qk > 128), "RoPE bprop requires d_qk <= 128 (the sDQ SMEM staging exceeds the A100 budget beyond that)"
    # seq_len_q [B] int32 — per-batch live Q length (dense PADDED).  Dummy 1-elem
    # when unused (kernel never reads it at has_seq_len_q=False).
    if has_seq_len_q:
        seqq_t = seq_len_q.to(dtype=torch.int32, device=Q.device).contiguous()
    else:
        seqq_t = torch.zeros(1, dtype=torch.int32, device=Q.device)
    assert d_qk % 2 == 0
    # dQ splits d-cols across the two sub-groups → each reads a DQ_N = d_qk//2
    # column slice of sK.  load_b_smem_x4 takes the d-col offset as `col_base`
    # (swizzle computed on the TRUE column), so the Swz128B half-column read is
    # correct for ANY d — no longer requires d_qk//2 % 64 == 0.  The only
    # remaining shape constraint is DQ_N % 16 == 0 (load_b_smem_x4 N//8 even):
    # d=128→64 ✓, d=64→32 ✓.
    assert (d_qk // 2) % 16 == 0, f"d_qk//2 ({d_qk//2}) must be a multiple of 16 (ldmatrix.x4 N//8 even)"
    assert d_v % 32 == 0, f"d_v ({d_v}) must be a multiple of 32 (do_dot warp reduce)"
    # Partial-tile (arbitrary seqlen): SQ/SKV need NOT be tile multiples.  The
    # last straddling tile zero-fills OOB rows on load, masks P=0 for kv>=SKV /
    # q>=SQ, clamps LSE/do_dot/bias reads, and row-gates the dV/dK/dQ/dBias
    # stores (all compile-time gated on SQ%tile_q / SKV%tile_kv → dense path is
    # byte-identical).  RoPE still requires alignment (see below).
    if has_rope and (SQ % tile_q or SKV % tile_kv):
        raise NotImplementedError("SM80 bprop: RoPE requires SQ/SKV tile-aligned")
    # dQ M-tiling: tile_q rows are covered by warps_per_sg warps × 16 × M_BLOCKS,
    # so tile_q must be an exact multiple of warps_per_sg*16 (llama 64→1 block,
    # gptoss 128→2).  dV/dK/BMM1 keep M=tile_kv=warps_per_sg*16 (1 block).
    assert tile_q % (warps_per_sg * 16) == 0, f"tile_q ({tile_q}) must be a multiple of warps_per_sg*16 ({warps_per_sg*16})"
    assert tile_kv == warps_per_sg * 16, f"tile_kv ({tile_kv}) must equal warps_per_sg*16 ({warps_per_sg*16})"
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    scale_log2 = scale * math.log2(math.e)
    inv_scale = 1.0 / float(scale)

    dQ_acc = torch.zeros(B, SQ, H, d_qk, dtype=torch.float32, device=Q.device)
    dQ = torch.empty(B, SQ, H, d_qk, dtype=Q.dtype, device=Q.device)
    # Deterministic-dQ relay counter: one int32 per (seq, head, q_tile), zeroed
    # per launch.  Stride = ceil(max_SQ/tile_q) so the per-seq q_iter (THD) or the
    # dense q_iter both index in-bounds; n_seq sequences (= B dense).  1-elem dummy
    # (never touched) on the fast path so it costs nothing.
    sem_q_stride = (max_sq + tile_q - 1) // tile_q if deterministic else 0
    sem_units = n_seq * H * sem_q_stride if deterministic else 1
    dq_sem = torch.zeros(max(sem_units, 1), dtype=torch.int32, device=Q.device)
    # dK/dV write buffers have H_q heads (one slice per query head — no atomics).
    # MHA (gqa_ratio==1): they ARE the outputs.  GQA: a per-query-head workspace
    # that a reduction kernel sums over the group → [B,SKV,Hk,d] outputs.
    dK_ws = torch.empty(B, SKV, H, d_qk, dtype=Q.dtype, device=Q.device)
    dV_ws = torch.empty(B, SKV, H, d_v, dtype=Q.dtype, device=Q.device)
    if gqa_ratio == 1:
        dK, dV = dK_ws, dV_ws
    else:
        dK = torch.empty(B, SKV, Hk, d_qk, dtype=Q.dtype, device=Q.device)
        dV = torch.empty(B, SKV, Hk, d_v, dtype=Q.dtype, device=Q.device)

    lse_t = lse.to(dtype=torch.float32, device=Q.device).contiguous()
    # seq_kv_lens [B] int32 (or 1-elem dummy when not padded — never read).
    if has_seq_kv_lens:
        seqk_t = seq_kv_lens.to(dtype=torch.int32, device=Q.device).contiguous()
    else:
        seqk_t = torch.zeros(1, dtype=torch.int32, device=Q.device)
    # cu_seqlens [n_seq+1] int32 (THD) or 1-elem dummy (dense — never read).  The
    # over-provisioned THD grid covers the longest sequence (ceil(max_skv/tile_kv)
    # kv-tiles) × H × n_seq; short sequences early-out per kv-tile.
    if thd:
        cu_q_t = cu_seqlens_q.to(dtype=torch.int32, device=Q.device).contiguous()
        cu_k_t = cu_seqlens_k.to(dtype=torch.int32, device=Q.device).contiguous()
        grid_kv_tiles = (max_skv + tile_kv - 1) // tile_kv
        grid_batch = n_seq
    else:
        cu_q_t = torch.zeros(1, dtype=torch.int32, device=Q.device)
        cu_k_t = torch.zeros(1, dtype=torch.int32, device=Q.device)
        grid_kv_tiles = 0
        grid_batch = 0
    # Bias + dBias (fp32 accumulator, same shape as bias; atomicAdd reduces over
    # batch when bias is broadcast [1,H,SQ,SKV]).
    if has_bias:
        bias_t = bias.contiguous()
        dbias_t = torch.zeros(bias_batch, H, SQ, SKV, dtype=torch.float32, device=Q.device)
    else:
        # Dummy must match the fake tensor _compile_main builds at has_bias=False
        # (bias_is_fp32 defaults True → fp32).
        bias_t = torch.zeros(1, dtype=torch.float32, device=Q.device)
        dbias_t = torch.zeros(1, dtype=torch.float32, device=Q.device)

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # ---- do_dot: on-device (default) or caller-supplied ------------------
    if do_dot is None:
        dot_t = torch.empty(B, H, SQ, dtype=torch.float32, device=Q.device)
        dd_fn = _compile_do_dot(B, H, SQ, d_v, io_is_bf16)
        dd_fn(from_dlpack(O), from_dlpack(dO), from_dlpack(dot_t), cutlass.Int32(B * H * SQ), stream)
    else:
        dot_t = do_dot.to(dtype=torch.float32, device=Q.device).contiguous()

    # dSink (standalone reduction; dQ/dK/dV are already sink-correct via LSE).
    if has_sink:
        ds_fn = _compile_dsink(B, H, SQ)
        ds_fn(from_dlpack(lse_t), from_dlpack(dot_t), from_dlpack(sinks_t), from_dlpack(dsink_t), cutlass.Int32(B * H), stream)

    fn = _compile_main(
        B,
        H,
        SQ,
        SKV,
        d_qk,
        d_v,
        tile_kv,
        tile_q,
        warps_per_sg,
        io_is_bf16,
        qo_stages=qo_stages,
        dq_smem_coalesce=dq_smem_coalesce,
        Hk=Hk,
        mask_flags=mask_flags,
        swa_window=swa_window,
        causal_bottom_right=cbr,
        has_seq_kv_lens=has_seq_kv_lens,
        has_bias=has_bias,
        bias_is_fp32=bias_is_fp32,
        bias_batch=bias_batch,
        has_rope=has_rope,
        rope_max_s=rope_max_s,
        has_seq_len_q=has_seq_len_q,
        thd_varlen=thd,
        n_seq=n_seq,
        deterministic=deterministic,
        sem_units=max(sem_units, 1),
        sched_policy=sched_policy,
    )
    fn(
        from_dlpack(Q),
        from_dlpack(K),
        from_dlpack(V),
        from_dlpack(dO),
        from_dlpack(dQ_acc),
        from_dlpack(dK_ws),
        from_dlpack(dV_ws),
        from_dlpack(lse_t),
        from_dlpack(dot_t),
        from_dlpack(seqk_t),
        from_dlpack(bias_t),
        from_dlpack(dbias_t),
        from_dlpack(rope_cs_t),
        from_dlpack(cu_q_t),
        from_dlpack(cu_k_t),
        from_dlpack(seqq_t),
        from_dlpack(dq_sem),
        cutlass.Int32((SQ + tile_q - 1) // tile_q),
        cutlass.Float32(scale_log2),
        cutlass.Float32(scale),
        cutlass.Int32(right_bound),
        cutlass.Float32(inv_scale),
        cutlass.Int32(bias_bstride),
        cutlass.Int32(sem_q_stride),
        cutlass.Int32(grid_kv_tiles),
        cutlass.Int32(grid_batch),
        stream,
    )

    cast_fn = _compile_cast(B, H, SQ, d_qk, io_is_bf16)
    n_vecs = (B * SQ * H * d_qk) // 2
    cast_fn(from_dlpack(dQ_acc), from_dlpack(dQ), cutlass.Int32(n_vecs), stream)

    # GQA: sum dK_ws/dV_ws over each query-head group → [B,SKV,Hk,d] outputs.
    if gqa_ratio > 1:
        rk_fn = _compile_dkv_reduce(B, SKV, H, Hk, d_qk, io_is_bf16)
        rk_fn(from_dlpack(dK_ws), from_dlpack(dK), cutlass.Int32(B * SKV * Hk * d_qk), stream)
        rv_fn = _compile_dkv_reduce(B, SKV, H, Hk, d_v, io_is_bf16)
        rv_fn(from_dlpack(dV_ws), from_dlpack(dV), cutlass.Int32(B * SKV * Hk * d_v), stream)

    # Optional grads appended in a FIXED order (dBias, then dSink); the caller
    # reconstructs positions from has_bias / has_sink (which it passed in).
    outs = [dQ, dK, dV]
    if has_bias:
        outs.append(dbias_t)  # fp32 [1|B, H, SQ, SKV]
    if has_sink:
        outs.append(dsink_t)  # fp32 [H]
    return tuple(outs)
