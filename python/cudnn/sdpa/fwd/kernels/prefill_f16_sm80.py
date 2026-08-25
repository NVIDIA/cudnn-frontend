# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 (Ampere / A100) SDPA prefill, FP16 in / FP16 out.

This file is a TEMPLATE: ``frost.template_loader.load_template`` re-executes
it as a fresh module per ``config_sm80.TemplateParams`` (injected as the
``FROST_TEMPLATE_PARAMS`` module global), so the feature/config axes fold at
trace time; the remaining SHAPE axes compile through the module's own
``compile()`` (per-shape ``@lru_cache``; THD packed token totals are DYNAMIC
via ``cute.sym_int`` — plan-time-only keys, Hard Rule 4).  The adapter
(``api_dsl.SdpaFwdDslSm80``) owns validation, operand binding and launch.

Online flash-attention with rowwise max / sum tracked per warp lane
(2 M-rows per lane), threadquad butterfly reductions across the 4 lanes
that share a row, exp2-based softmax with the input ``softmax_scale * log2(e)``
folded into the exponent (so the kernel uses ``exp2`` not ``exp``), and
``O`` rescaled by ``exp2(scale_log2 * (m_old - m_new))`` at each iter.
The first iter zeros the accumulator implicitly via ``m_state = -1e30``
→ ``alpha ≈ 0`` → ``O *= 0`` before the first SV mma accumulates.

Pipeline shape (Llama-class, d_qk = d_v = 128; DSv3 = d_qk=192, d_v=128):

  ┌─────────────────────────────────────────────────────────────────┐
  │  1 CTA = 1 (batch, head, q_tile) work item.  No persistent loop.│
  │  8 warps × 32 threads = 256 threads/CTA (default; flavor-tuned),│
  │  no warp specialization.                                        │
  │  TILE_M = 128 (Q rows / CTA — each warp owns 16)                │
  │  TILE_N = 64  (KV rows / iter)                                  │
  │  D_QK   = 128 (head_dim, also Q→reg storage dim)                │
  │  D_V    = 128 (V head_dim, = D_QK on Llama)                     │
  │                                                                 │
  │  SMEM:  sQ_buf    32 KiB  (Q during prologue;  splits into     │
  │                            sV0 + sV1 = double-buffered V ring)  │
  │         sK_buf    32 KiB  (K[i] / K[i+1] double-buffered ring;  │
  │                            stage picked at runtime via ptr math │
  │                            since kv_iter is a runtime value)    │
  │         total =   64 KiB  (well under A100's 164 KiB opt-in)    │
  │         P fragment lives in regs only — no SMEM round-trip.     │
  │                                                                 │
  │  RF/thread:                                                     │
  │         Q regs-resident          32 i32  (fp16x2 packed)        │
  │         O fp32 accumulator       64 fp32                        │
  │         S fp32 (transient)       32 fp32                        │
  │         total fixed = 128 regs; ~128 regs left for K/V/P frags  │
  │         and ldmatrix temporaries — well under the 256/thread    │
  │         cap on SM80.                                            │
  │                                                                 │
  │  Pipeline timeline (FA-style, K + V both double-buffered):      │
  │    prologue:    cp.async Q (commit)                             │
  │                 cp.async K[0] (commit)                          │
  │                 wait_group(1) → Q done, K[0] still in flight    │
  │                 ldmatrix Q → regs                               │
  │                 barrier (Q reads done before iter 0 overwrites  │
  │                          sV0/sV1 with V[0])                     │
  │                                                                 │
  │    loop iter i: cp.async V[i]   into sV[i%2]    (commit)        │
  │                 wait_group(1)  → K[i] done                      │
  │                 barrier                                         │
  │                 QK mma  ← overlaps V[i] cp.async                │
  │                 softmax → P packed half2 in regs (16 i32/lane)  │
  │                 cp.async K[i+1] into sK[next] (commit, if any)  │
  │                 wait_group(1 if !last else 0) → V[i] done       │
  │                 barrier                                         │
  │                 SV mma  ← overlaps K[i+1] cp.async              │
  │                 (no end-of-iter barrier — V[i+2] write to       │
  │                  sV[i%2] is fenced by iter (i+1)'s barriers)    │
  │                                                                 │
  │    post-loop:   barrier (all warps finished last SV mma)        │
  │                                                                 │
  │    epilogue:    cast O f32→f16, STG to gmem.                    │
  └─────────────────────────────────────────────────────────────────┘

Synchronisation: ``cp.async.commit_group`` / ``cp.async.wait_group(0)``
only — no mbarriers, no clusters (clusters are SM90+).  Standard Ampere
ldgsts idiom.

Layout choices:
- SMEM K / V row-major [N_kv_row][d_col] with the 128B XOR swizzle
  (``load_tile_2d(..., swizzle=True)`` + ``swizzle_xor_128b`` at ldmatrix)
  to keep the ldmatrix reads bank-conflict-free.
- ldmatrix.x4 .row for A operands (Q at QK time; P at SV time is built
  directly from QK D-frag regs — no ldmatrix at all).
- ldmatrix.x2 layout for B operands depends on which SMEM axis aligns
  with the mma K direction (the reduction axis):
    - K (QK time): SMEM[N_kv, d_qk].  mma's K is d_qk → SMEM col, mma's
      N is N_kv → SMEM row.  Use ``.row`` (no trans) — lane (g, p)
      gets source[row=g, cols=2p..2p+1] = K[N=g, d=2p..2p+1], which
      already matches mma B col-major b0 at (K=2p..2p+1, N=g).
    - V (SV time): SMEM[N_kv, d_v].   mma's K is N_kv → SMEM row, mma's
      N is d_v → SMEM col.  Use ``.col`` (= .trans) — the swap makes
      the per-lane data land at (rows=2p..2p+1, col=g) = V[N=2p..2p+1, d=g],
      matching mma B col-major b0 at (K=2p..2p+1, N=g).
"""

from functools import lru_cache
from typing import Optional


import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack as _from_dlpack_raw


def from_dlpack(t, **kw):
    """Vendoring shim: the kernels compile with --enable-tvm-ffi, so host-side
    conversions must produce TVM-FFI tensors regardless of the
    CUTE_DSL_ENABLE_TVM_FFI environment latch."""
    kw.setdefault("enable_tvm_ffi", True)
    return _from_dlpack_raw(t, **kw)  # was from_dlpack pre-DKG-bump


from cutlass.base_dsl.typing import Pointer  # was the DSL's Pointer pre-DKG-bump
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass._mlir.dialects import arith

# Pick up the in-tree tile_dsl wrappers (mma, load_tile_2d, cp_async_*).
from cudnn.frost.tile_dsl.mma import load_b_smem_x4, mma_step  # noqa: E402
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16  # noqa: E402
from cudnn.frost.tile_dsl.rope import rope_rotate_smem_tile  # noqa: E402
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b  # noqa: E402
from cudnn.frost.tile_dsl.tma import load_tile_2d, cp_async_commit, cp_async_wait  # noqa: E402
from cudnn.frost.tile_dsl.mask import (  # noqa: E402
    MASK_NONE,
    MASK_PADDED,
    MASK_CAUSAL,
    MASK_SWA,
)
from cudnn.frost.tile_dsl.constants import (  # noqa: E402
    SCHED_LPT,
    SCHED_NATURAL,
)
from cudnn.sdpa.fwd.config_sm80 import TemplateParams, validate_params  # noqa: E402

ELEM_BYTES = 2  # fp16 (Phase 2/3 baseline)
ELEMS_PER_LD = 8  # 8 fp16 per cp.async = 16 B (max throughput)


# Scheduler policy IDs — the shared frost vocabulary (tile_dsl.constants;
# identical 0/1/2 values the kernel always used).
SCHED_DEFAULT = SCHED_NATURAL  # the kernel's plain 3-D grid (q_tile, head, batch)


# ---------------------------------------------------------------------------
# Template identity.
# ---------------------------------------------------------------------------
# Injected by ``frost.template_loader.load_template``; the module-level
# default keeps a direct import (tests, tooling) importable at the
# llama-flavor baseline.  Tile geometry, head-dim envelope, dtype, mask
# family, operand presence, scheduler policy and LSE presence all live here
# and fold at trace time — ``compile()`` below carries only shape axes.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
validate_params(PARAMS)


# ---------------------------------------------------------------------------
# Kernel.
# ---------------------------------------------------------------------------
@cute.kernel
def _sdpa_kernel(
    Q: cute.Tensor,  # [B, SQ,  HQ, D_QK] fp16
    K: cute.Tensor,  # [B, SKV, HQ, D_QK] fp16    (MHA: HK == HQ)
    V: cute.Tensor,  # [B, SKV, HQ, D_V]  fp16
    O: cute.Tensor,  # [B, SQ,  HQ, D_V]  fp16  (output)
    LSE: Optional[cute.Tensor],  # [B, HQ, SQ] fp32 (LSE in natural log); THD
    # packed [1, HQ, T].  None ⇒ no Stats output —
    # the whole LSE compute + store is compiled out.
    seq_kv_lens: cute.Tensor,  # [B] int32 — per-batch effective KV length (padded
    # mask).  Consulted only when has_seq_kv_lens; a
    # 1-elem dummy is passed otherwise.
    seq_len_q: cute.Tensor,  # [B] int32 — per-batch effective Q length.  Consulted
    # only when has_seq_len_q (BR diagonal under padding);
    # a 1-elem dummy is passed otherwise.
    sinks: cute.Tensor,  # [H] fp32 — per-Q-head sink logit in log2 units
    # (= sink_h * log2(e), the same log2-of-scaled-logit
    # units as row_max).  Consulted only when has_sink;
    # joins the softmax DENOMINATOR only (V_sink = 0).
    # 1-elem dummy when unused.
    bias: cute.Tensor,  # [1, H, SQ, SKV] additive attention bias (broadcast
    # over batch).  io_dtype (fp16/bf16) OR fp32 per
    # bias_is_fp32.  Consulted only when has_bias; a
    # 1-elem dummy otherwise.  Added pre-scale into S_acc
    # as bias * inv_softmax_scale so the late
    # * softmax_scale_log2 reproduces the reference's
    # post-scale ``s += bias`` in the log2 domain.
    cu_q: cute.Tensor,  # [B+1] int32 cumulative Q seqlens (THD/varlen).  Q/O
    # are packed [1,T,H,D]; sequence b owns rows
    # [cu_q[b], cu_q[b+1]).  Consulted only when
    # THD_VARLEN; a 1-elem dummy otherwise.
    cu_k: cute.Tensor,  # [B+1] int32 cumulative KV seqlens (THD/varlen).
    rope_cs: cute.Tensor,  # [max_s, d_qk//2, 2] fp32 — host-precomputed
    # (cos, sin) RoPE table (torch full-range cos/sin
    # of the angles).  Consulted only when has_rope;
    # a 1-elem dummy is passed otherwise.  Applied as
    # the half-split rotate_half to Q AND K in SMEM
    # before ldmatrix (dense-only; no THD).
    tile_m: cutlass.Constexpr[int],  # Q rows / CTA  (64 or 128)
    num_warps: cutlass.Constexpr[int],  # warps / CTA   (4  or 8)
    tile_n: cutlass.Constexpr[int],  # KV rows / iter (64 default; 128 doubles
    # arithmetic intensity at cost of RF + SMEM)
    d_qk: cutlass.Constexpr[int],  # head dim for Q/K  (Llama 128, GPT-OSS 64)
    d_v: cutlass.Constexpr[int],  # head dim for V    (== d_qk on the
    # currently-supported flavors)
    io_dtype: cutlass.Constexpr,  # cutlass.Float16 / cutlass.BFloat16 — Q/K/V/O
    # element type.  bf16 reuses the entire
    # fp16 pipeline (same 2-B SMEM/cp.async/
    # ldmatrix); only the mma.sync variant
    # (ab_dtype) and the fp32→16b casts flip.
    is_even_mn: cutlass.Constexpr[bool],  # True ⇒ SQ%tile_m==0 AND SKV%tile_n==0;
    # False ⇒ predicate Q/K/V loads + STG + mask
    # OOB K-cols (MASK_PADDED implicit).
    is_even_k: cutlass.Constexpr[bool],  # True ⇒ D == D_QK == D_V; False ⇒ predicate
    # the K-dim cp.async (assumes D % 8 == 0).
    mask_flags: cutlass.Constexpr[int],  # MASK_NONE / CAUSAL / SWA bitmask (PADDED
    # is auto-ORed when ~is_even_mn).
    swa_window: cutlass.Constexpr[int],  # SWA window width (used only when MASK_SWA).
    causal_bottom_right: cutlass.Constexpr[bool],  # MASK_CAUSAL only: align the causal
    # diagonal to the bottom-right corner
    # (k <= q + (SKV - SQ)) instead of top-left
    # (k <= q).  Folds to a runtime diag offset.
    has_seq_kv_lens: cutlass.Constexpr[bool],  # True ⇒ read per-batch effective KV
    # length from seq_kv_lens[batch] and mask
    # cols >= it (user per-batch padded mask).
    has_seq_len_q: cutlass.Constexpr[bool],  # True ⇒ read per-batch effective Q
    # length from seq_len_q[batch] (BR diagonal
    # base = eff_skv - eff_sq under padding).
    has_sink: cutlass.Constexpr[bool],  # True ⇒ add the per-head sink logit to the
    # softmax denominator at finalize.
    has_bias: cutlass.Constexpr[bool],  # True ⇒ add the additive attention bias
    # tile to S_acc (pre-scale) every kv-iter.
    bias_is_fp32: cutlass.Constexpr[bool],  # bias element type: True=fp32, False=io_dtype.
    THD_VARLEN: cutlass.Constexpr[bool],  # True ⇒ packed [1,T,H,D] Q/K/V/O + cu_q/cu_k
    # cumulative seqlens; per-batch S_q/S_kv come
    # from the cu_* diffs, GMEM seq coords are
    # offset by cu_q[b]/cu_k[b], and over-provisioned
    # tiles (q_row_base >= S_q_b) run 0 kv-iters.
    has_rope: cutlass.Constexpr[bool],  # True ⇒ rotate Q + K in SMEM via the
    # half-split RoPE before ldmatrix, using
    # the precomputed rope_cs (cos,sin) table.
    sched_policy: cutlass.Constexpr[int],  # SCHED_DEFAULT / LPT / LPT_L2.
    sched_l2_bytes: cutlass.Constexpr[int],  # L2 budget for SCHED_LPT_L2 sizing.
    n_kv_tiles: cutlass.Int32,  # runtime — host computes round_up(SKV/tile_n)
    softmax_scale_log2: cutlass.Float32,  # = scale * log2(e), folded into exp2
    sq_runtime: cutlass.Int32,  # actual SQ  (consulted only when ~is_even_mn)
    skv_runtime: cutlass.Int32,  # actual SKV (consulted when ~is_even_mn OR
    # MASK_PADDED / CAUSAL)
    d_runtime: cutlass.Int32,  # actual D   (consulted only when ~is_even_k)
    causal_band_right: cutlass.Int32,  # MASK_CAUSAL: extra right band — admit up to
    # this many future tokens (k <= q + right).
    # Adds to causal_diag; 0 = plain causal / BR.
    inv_softmax_scale: cutlass.Float32,  # = 1 / scale.  Multiplies the additive bias
    # before it is added to the un-scaled S_acc
    # (consulted only when has_bias).
):
    # ---- derived shape / layout constants ---------------------------------
    # Folded at trace time — one compiled fn per (tile_m, num_warps, tile_n,
    # d_qk, d_v) tuple.
    m_per_warp = tile_m // num_warps  # 16 (TILE_M=64/4w or 128/8w) or 32 (128/4w)
    m_blocks = m_per_warp // 16  # 1 in 64/4w & 128/8w cases; 2 in 128/4w
    threads = num_warps * 32  # 128 or 256
    # Per-d derived layout constants — fold to Python ints at trace time.
    # K and V can differ along the head dim (DSv3: d_qk=192, d_v=128).  All
    # other flavors keep d_qk == d_v.
    ELEMS_PER_ROW_Q = d_qk  # fp16 per Q row (Llama 128, GPT-OSS 64, DSv3 192)
    ELEMS_PER_ROW_K = d_qk  # fp16 per K row (== d_qk by construction)
    ELEMS_PER_ROW_V = d_v  # fp16 per V row (DSv3: 128 != d_qk=192)
    QK_K_CHUNKS = d_qk // 16  # k_chunks of 16 fp16 along QK reduction
    SV_N_FRAGS = d_v // 8  # n_frags of 8 cols (SV output N)
    elems_q = tile_m * ELEMS_PER_ROW_Q
    # Tile_n-derived constants.
    QK_N_FRAGS = tile_n // 8  # 8 (tile_n=64) or 16 (tile_n=128)
    SV_K_CHUNKS = tile_n // 16  # 4 or 8
    ELEMS_KV_K = tile_n * ELEMS_PER_ROW_K  # K ring slot size in fp16 elems
    ELEMS_KV_V = tile_n * ELEMS_PER_ROW_V  # V ring slot size in fp16 elems

    # ---- block / thread indices -------------------------------------------
    # Tile-scheduling policy — selected at trace time via Constexpr.
    #
    # SCHED_DEFAULT: plain 3-D grid (q_tile, head, batch).  No reorder.
    #   Best for MASK_NONE (uniform-length mainloops; reverse-row HURTS
    #   L2 reuse on K/V).
    # SCHED_LPT: 1-D grid, reverse-row across (head, batch).  Best for
    #   small (B, H) and longest-tile-first matters most (causal SQ=2K).
    #   Mirrors decode_linear_tile() in the upstream C++ prefill kernel.
    # SCHED_LPT_L2: 1-D grid, block-cyclic over L2-sized (batch, head)
    #   groups.  Picks ``active_groups = sched_l2_bytes / per_group`` so
    #   each block's K+V resident set fits in L2.  Reverse-row WITHIN a
    #   block.  Best for larger SQ where K/V GMEM footprint exceeds L2.
    if cutlass.const_expr(sched_policy == SCHED_DEFAULT):
        bx, by, bz = cute.arch.block_idx()
        q_tile_idx = bx
        head_idx = by
        batch_idx = bz
    else:
        bx, _, _ = cute.arch.block_idx()
        SQ_for_decode = Q.shape[1]
        H_for_decode = Q.shape[2]  # H_q
        Hkv_for_decode = K.shape[2]  # H_kv (may differ)
        B_for_decode = Q.shape[0]
        heads_per_kv_decode = H_for_decode // Hkv_for_decode
        q_tiles_i32 = cutlass.Int32((SQ_for_decode + tile_m - 1) // tile_m)
        H_i32 = cutlass.Int32(H_for_decode)
        HB = cutlass.Int32(H_for_decode * B_for_decode)
        if cutlass.const_expr(sched_policy == SCHED_LPT):
            row_rank = bx // HB
            within_hb = bx % HB
            q_tile_idx = (q_tiles_i32 - cutlass.Int32(1)) - row_rank
            head_idx = within_hb % H_i32
            batch_idx = within_hb // H_i32
        else:
            # SCHED_LPT_L2: per_group bytes = SKV * (D_QK + D_V) * BPE_K_V.
            # ``per_group`` is the K + V GMEM footprint for ONE
            # (batch, kv_head) group across all KV rows.  Under GQA/MQA the
            # group's K/V is shared by ``heads_per_kv`` Q-heads, so a
            # single block can fit MORE useful Q work than under MHA.
            # Use the compile-time D_QK upper bound rather than d_runtime —
            # the L2 budget is dimensioned for the BLOB the kernel reads
            # from GMEM (full D_QK columns even when D < D_QK).
            #
            # active_groups = clamp(sched_l2_bytes / per_group, 1, num_groups).
            BPE_K_V = 2  # fp16 K, fp16 V
            SKV_i32 = cutlass.Int32(K.shape[1])
            per_group = SKV_i32 * cutlass.Int32((d_qk + d_v) * BPE_K_V)
            l2_budget = cutlass.Int32(sched_l2_bytes)
            num_groups = cutlass.Int32(Hkv_for_decode * B_for_decode)
            ag_raw = l2_budget // per_group
            ag_min1 = cutlass.Int32(
                arith.select(
                    (ag_raw < cutlass.Int32(1)).ir_value(),
                    cutlass.Int32(1).ir_value(),
                    ag_raw.ir_value(),
                )
            )
            active_groups = cutlass.Int32(
                arith.select(
                    (ag_min1 > num_groups).ir_value(),
                    num_groups.ir_value(),
                    ag_min1.ir_value(),
                )
            )
            tiles_per_grp = q_tiles_i32 * cutlass.Int32(heads_per_kv_decode)
            tiles_per_blk = active_groups * tiles_per_grp
            num_blocks = (num_groups + active_groups - cutlass.Int32(1)) // active_groups
            block_idx_ = bx // tiles_per_blk
            within_blk = bx % tiles_per_blk
            # Last block may have fewer than ``active_groups`` groups.
            is_last_block = (block_idx_ + cutlass.Int32(1)) == num_blocks
            agroup_eff_lb = num_groups - block_idx_ * active_groups
            agroup_eff = cutlass.Int32(
                arith.select(
                    is_last_block.ir_value(),
                    agroup_eff_lb.ir_value(),
                    active_groups.ir_value(),
                )
            )
            row_rank = within_blk // (agroup_eff * cutlass.Int32(heads_per_kv_decode))
            in_rank = within_blk % (agroup_eff * cutlass.Int32(heads_per_kv_decode))
            # in_rank lays out as (sub_head, kv_group) so all Q-heads
            # sharing the same KV head/batch land in the same block.
            sub_head = in_rank // agroup_eff
            kv_group = (in_rank % agroup_eff) + block_idx_ * active_groups
            kv_head = kv_group % cutlass.Int32(Hkv_for_decode)
            batch_idx = kv_group // cutlass.Int32(Hkv_for_decode)
            head_idx = kv_head * cutlass.Int32(heads_per_kv_decode) + sub_head
            q_tile_idx = (q_tiles_i32 - cutlass.Int32(1)) - row_rank

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = tidx // 32
    lane = tidx % 32

    # ---- SMEM allocations -------------------------------------------------
    # 4-buffer layout: 2 K + 2 V = 64 KiB regardless of tile_m.  Both K
    # and V live in ONE contiguous 32 KiB SMEM block each, with the
    # per-stage slot picked at runtime via pointer arithmetic
    # (``cur_stage = kv_iter & 1``).  Single allocs (``sK_buf``,
    # ``sQ_buf``) replace the old paired ``sK0`` / ``sK1`` because
    # ``kv_iter`` is now a runtime value (was a constexpr unroll variable)
    # to slash JIT compile time.
    #   * sQ_buf doubles as Q's SMEM during the prologue and is reused as
    #     the V double-buffer ring (sV0 = sQ_buf, sV1 = sQ_buf + ELEMS_KV_V).
    #     For asymmetric flavors (DSv3: d_qk=192, d_v=128) sQ_buf is sized
    #     to fit both Q (tile_m·d_qk) and the V ring (2·tile_n·d_v) and the
    #     sO staging buffer (tile_m·d_v) used in the epilogue.
    #   * TILE_M=128 (Q tile = 32 KiB on Llama, 48 KiB on DSv3).
    #   * TILE_M=64  (Q tile = 16 KiB on Llama) → Q fills sV0 only; sV1 sits
    #     idle during prologue and gets used for V[1] in the mainloop.
    # Double-buffering V removes the need for a per-iter end-of-iter
    # barrier (V[i+2]'s write to sV[i%2] is naturally fenced by iter
    # (i+1)'s K-wait + V-wait barriers).
    SQ_BUF_ELEMS = max(elems_q, 2 * ELEMS_KV_V, tile_m * d_v)
    SK_BUF_ELEMS = 2 * ELEMS_KV_K
    sQ_buf = cutlass.Array(io_dtype, SQ_BUF_ELEMS, alignment=128, space=cutlass.AddressSpace.smem)
    sK_buf = cutlass.Array(io_dtype, SK_BUF_ELEMS, alignment=128, space=cutlass.AddressSpace.smem)

    # ---- GMEM base pointers + row strides ---------------------------------
    Q_view = cutlass.make_array_view(Q)
    K_view = cutlass.make_array_view(K)
    V_view = cutlass.make_array_view(V)
    O_view = cutlass.make_array_view(O)

    # Shapes (B, S, H, D).  GQA/MQA: H_q (Q heads) may exceed H_kv (K/V
    # heads) when H_q % H_kv == 0.  ``heads_per_kv`` Q-heads share each
    # KV-head; the kernel maps ``kv_head_idx = head_idx // heads_per_kv``
    # at GMEM-offset time.  MHA is the H_q == H_kv special case.
    SQ = Q.shape[1]
    SKV = K.shape[1]
    H = Q.shape[2]  # H_q
    H_kv = K.shape[2]  # H_kv (== H for MHA)
    heads_per_kv = H // H_kv  # Python int — folds at trace time
    kv_head_idx = head_idx if heads_per_kv == 1 else (head_idx // cutlass.Int32(heads_per_kv))

    # Additive-bias per-head base pointer (element-addressed).  bias is
    # [1, H, SQ, SKV] (broadcast over batch): head stride = SQ*SKV, q stride =
    # SKV, col stride = 1.  Materialize the per-head ELEMENT base here; the
    # mainloop loads the per-lane tile frag (overlapped with the QK mma) and
    # the softmax injects it pre-scale.  bias_dt selects the GMEM load width.
    bias_dt = cutlass.Float32 if cutlass.const_expr(bias_is_fp32) else io_dtype
    if cutlass.const_expr(has_bias):
        _bias_ptr0 = cutlass.make_array_view(bias).data_ptr()
        bias_head_e = cutlass.Int64(head_idx) * cutlass.Int64(SQ) * cutlass.Int64(SKV)
        bias_base = _bias_ptr0 + bias_head_e  # per-head element ptr

    # RoPE (cos, sin) table base — shared by all heads/batches (position-only).
    # [max_s, d_qk//2, 2] fp32; element (pos, i, {0,1}) at (pos*d2+i)*2 (+1).
    if cutlass.const_expr(has_rope):
        rope_cs_ptr = Pointer(cutlass.make_array_view(rope_cs).data_ptr(), dtype=cutlass.Float32)

    # THD/varlen: read per-batch cumulative seqlens → packed seq origins +
    # per-batch lengths.  Q/O are packed [1,T,H,D]: sequence b's rows live at
    # [cu_q[b], cu_q[b+1]); K/V at [cu_k[b], cu_k[b+1]).  The GMEM bases below
    # use the seq origin (in rows) instead of the dense batch*S term; the dense
    # path keeps batch*S.  s_q_b / s_kv_b feed the per-batch eff lengths +
    # predication below.
    if cutlass.const_expr(THD_VARLEN):
        _cuq = Pointer(cutlass.make_array_view(cu_q).data_ptr(), dtype=cutlass.Int32)
        _cuk = Pointer(cutlass.make_array_view(cu_k).data_ptr(), dtype=cutlass.Int32)
        cu_q_b = _cuq[batch_idx]
        s_q_b = _cuq[batch_idx + cutlass.Int32(1)] - cu_q_b
        cu_k_b = _cuk[batch_idx]
        s_kv_b = _cuk[batch_idx + cutlass.Int32(1)] - cu_k_b
        q_seq_origin = cutlass.Int64(cu_q_b)
        kv_seq_origin = cutlass.Int64(cu_k_b)
    else:
        s_q_b = cutlass.Int32(sq_runtime)
        s_kv_b = cutlass.Int32(skv_runtime)
        q_seq_origin = cutlass.Int64(batch_idx) * cutlass.Int64(SQ)
        kv_seq_origin = cutlass.Int64(batch_idx) * cutlass.Int64(SKV)

    # Row stride along S axis in ELEMENTS (fp16) — ``+ N`` on a Float16-
    # typed pointer advances ``N * 2`` bytes.  Earlier byte-unit code
    # crashed with cudaErrorIllegalAddress at SKV > 1 tile (2x overshoot).
    # Q/K strides are derived from the RUNTIME D (= d_runtime) so the kernel
    # supports D < compile-time d_qk without slicing the user tensor; the
    # compile-time d_qk still governs SMEM, regs, and MMA shapes (the K-axis
    # tile is always d_qk fp16 wide, with cols ≥ d_runtime zero-padded via
    # cp.async predication).  V/O strides use the COMPILE-TIME d_v — the
    # kernel does not support partial d_v, so V's last dim is always exactly
    # d_v (matches the user tensor shape).  Q/O use H_q, K/V use H_kv (only
    # differs for GQA/MQA).
    Q_ROW_STRIDE_E = cutlass.Int32(H) * d_runtime
    K_ROW_STRIDE_E = cutlass.Int32(H_kv) * d_runtime
    V_ROW_STRIDE_E = cutlass.Int32(H_kv * d_v)
    O_ROW_STRIDE_E = cutlass.Int32(H * d_v)
    d_runtime64 = d_runtime.to(cutlass.Int64)
    d_v64 = cutlass.Int64(d_v)

    # Per-tile GMEM element offsets to (batch, q_row_base, head, 0).
    q_row_base = q_tile_idx * tile_m
    SKV_i64 = cutlass.Int64(SKV)
    H_kv_i64 = cutlass.Int64(H_kv)
    # Seq origin (in rows) absorbs both the dense batch*S term and the THD
    # cu_*[b] packed offset — the row-stride multiply is identical either way.
    q_seq_abs64 = q_seq_origin + cutlass.Int64(q_row_base)
    Q_BASE = q_seq_abs64 * Q_ROW_STRIDE_E.to(cutlass.Int64) + cutlass.Int64(head_idx) * d_runtime64
    O_BASE = q_seq_abs64 * O_ROW_STRIDE_E.to(cutlass.Int64) + cutlass.Int64(head_idx) * d_v64
    # LSE: dense [B, H_q, SQ]; THD packed [1, H_q, T]. Dense strides are
    # compile-time layout metadata; THD remains compact.
    # None-specialized: no Stats output ⇒ view/base/pointer are compiled out
    # together with the epilogue store below.
    if cutlass.const_expr(LSE is not None):
        LSE_view = cutlass.make_array_view(LSE)
        LSE_S_STRIDE_E = cutlass.Int64(LSE.stride[2])
        if cutlass.const_expr(THD_VARLEN):
            LSE_BASE = cutlass.Int64(head_idx) * cutlass.Int64(LSE.stride[1]) + (q_seq_origin + cutlass.Int64(q_row_base)) * LSE_S_STRIDE_E
        else:
            LSE_BASE = (
                cutlass.Int64(batch_idx) * cutlass.Int64(LSE.stride[0])
                + cutlass.Int64(head_idx) * cutlass.Int64(LSE.stride[1])
                + cutlass.Int64(q_row_base) * LSE_S_STRIDE_E
            )
        lse_gmem = LSE_view.data_ptr() + LSE_BASE

    # K/V offsets parameterised on kv_row_base (variable in mainloop).  Uses
    # H_kv and kv_head_idx (Q-heads sharing a KV head land at the same K/V
    # tile but accumulate into distinct Q rows + O rows).  K uses d_runtime
    # (allowed to be < d_qk); V uses compile-time d_v.  ``kv_seq_origin``
    # carries the dense batch*SKV or the THD cu_k[b] packed row offset.
    K_BATCH_OFF = kv_seq_origin * K_ROW_STRIDE_E.to(cutlass.Int64)
    K_HEAD_OFF = kv_head_idx.to(cutlass.Int64) * d_runtime64
    V_BATCH_OFF = kv_seq_origin * V_ROW_STRIDE_E.to(cutlass.Int64)
    V_HEAD_OFF = kv_head_idx.to(cutlass.Int64) * d_v64

    q_gmem = Q_view.data_ptr() + Q_BASE
    o_gmem = O_view.data_ptr() + O_BASE
    k_gmem_batch_head = K_view.data_ptr() + K_BATCH_OFF + K_HEAD_OFF
    v_gmem_batch_head = V_view.data_ptr() + V_BATCH_OFF + V_HEAD_OFF
    # Per-iter element advance: ``+ TILE_N rows`` on a Float16-typed ptr.
    # K and V have separate tile strides when d_qk != d_v (DSv3).  Stays
    # in Int32 (max value ``128 (max kv_iter) * 64 (TILE_N) * 64 (max H) *
    # 192 (max D) ~ 2^27``), so the mainloop only pays one Int32 mul +
    # Int64 sign-extend per iter instead of the old full Int64 multiply.
    K_TILE_STRIDE_E = cutlass.Int32(tile_n) * K_ROW_STRIDE_E
    V_TILE_STRIDE_E = cutlass.Int32(tile_n) * V_ROW_STRIDE_E

    # ---- MAINLOOP BOUNDS — causal / SWA trimming -------------------------
    # Mirror ``compute_kv_loop_bounds`` from the upstream C++ prefill
    # kernel.  Each Q tile only needs the KV blocks that
    # could contribute to its valid (q, k) pairs after masking:
    #   - causal: trim ``kv_right`` to the tile containing q_row_base+tile_m-1.
    #   - SWA   : trim ``kv_left`` to the tile containing q_row_base-W (only
    #             tiles whose last col is < q_row_base-W are fully masked out).
    # When neither flag is set both bounds collapse to [0, n_kv_tiles) — fast
    # path is byte-identical to the previous build.  Mask elements that
    # straddle the boundary tile still get -inf'd inside the softmax pre-pass.
    # Effective per-batch KV / Q lengths.  eff_skv: per-batch seq_kv_lens[batch]
    # when supplied, else the physical SKV (drives the column compare + KV
    # tile-prune).  eff_sq: per-batch seq_len_q[batch] when supplied, else the
    # physical SQ (only used for the bottom-right diagonal base below).
    if cutlass.const_expr(THD_VARLEN):
        eff_skv = s_kv_b  # cu_k[b+1] - cu_k[b]
    elif cutlass.const_expr(has_seq_kv_lens):
        _seqk_ptr = cutlass.make_array_view(seq_kv_lens).data_ptr()
        eff_skv = Pointer(_seqk_ptr, dtype=cutlass.Int32)[batch_idx]
    else:
        eff_skv = cutlass.Int32(skv_runtime)
    if cutlass.const_expr(THD_VARLEN):
        eff_sq = s_q_b  # cu_q[b+1] - cu_q[b]
    elif cutlass.const_expr(has_seq_len_q):
        _seqq_ptr = cutlass.make_array_view(seq_len_q).data_ptr()
        eff_sq = Pointer(_seqq_ptr, dtype=cutlass.Int32)[batch_idx]
    else:
        eff_sq = cutlass.Int32(sq_runtime)
    # Over-provisioned THD grid: a CTA whose q-tile starts past this sequence's
    # length does no work (its rows belong to the next sequence).  Mark it
    # invalid → the KV tile range collapses to 0 iters below and the O / LSE
    # stores are predicated off (q_row_base >= eff_sq → no row passes).
    thd_invalid = cutlass.Int32(q_row_base) >= eff_sq if cutlass.const_expr(THD_VARLEN) else cutlass.Int32(0) != cutlass.Int32(0)
    # Bottom-right diagonal base: BR shifts the whole attention band right by
    # (eff_SKV - eff_SQ) — PER-BATCH under padding (matching cuDNN); top-left is
    # 0.  ``br_base`` shifts BOTH the causal upper bound AND the SWA lower bound
    # (so BR + sliding-window / padding aligns to the bottom-right corner).  The
    # runtime ``causal_band_right`` additionally widens only the causal UPPER
    # bound (k <= q + br_base + right).
    br_base = (eff_skv - eff_sq) if cutlass.const_expr(causal_bottom_right) else cutlass.Int32(0)
    causal_diag = br_base + cutlass.Int32(causal_band_right)
    kv_left = cutlass.Int32(0)
    kv_right = n_kv_tiles
    if cutlass.const_expr(has_seq_kv_lens or THD_VARLEN):
        # Drop KV tiles entirely past the per-batch valid length:
        # kv_right = ceil_div(eff_skv, tile_n).
        padded_kv_hi = (eff_skv + cutlass.Int32(tile_n - 1)) // cutlass.Int32(tile_n)
        cond_pad_hi = padded_kv_hi < kv_right
        kv_right = cutlass.Int32(
            arith.select(
                cond_pad_hi.ir_value(),
                padded_kv_hi.ir_value(),
                kv_right.ir_value(),
            )
        )
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        # Last absolute Q row in this CTA = q_row_base + tile_m - 1.
        # K cols beyond (last_q + causal_diag) are masked, so
        # kv_right = ceil_div(last_q + 1 + causal_diag, tile_n)
        #          = ceil_div(q_row_base + tile_m + causal_diag, tile_n).
        causal_kv_hi = (cutlass.Int32(q_row_base) + cutlass.Int32(tile_m) + causal_diag + cutlass.Int32(tile_n - 1)) // cutlass.Int32(tile_n)
        cond_caus = causal_kv_hi < kv_right
        kv_right = cutlass.Int32(
            arith.select(
                cond_caus.ir_value(),
                causal_kv_hi.ir_value(),
                kv_right.ir_value(),
            )
        )
    if cutlass.const_expr(mask_flags & MASK_SWA):
        # SWA trims LEFT.  For a CTA covering Q rows [q_row_base,
        # q_row_base + tile_m), the LOWEST Q row (q_row_base) has the
        # smallest first-valid K col = q_row_base - W; tiles entirely
        # below that col are fully masked for *every* row in the CTA.
        # Matches C++ `prefill_sdpa_util.cuh`: anchor = q_row_coord (the
        # CTA-base Q row), not q_row_coord + TILE_M - 1.  ``br_base`` shifts the
        # window to the bottom-right corner under BR alignment (0 for top-left).
        swa_anchor = cutlass.Int32(q_row_base) + br_base - cutlass.Int32(swa_window)
        swa_neg = swa_anchor < cutlass.Int32(0)
        swa_lo_raw = swa_anchor // cutlass.Int32(tile_n)
        swa_lo = cutlass.Int32(
            arith.select(
                swa_neg.ir_value(),
                cutlass.Int32(0).ir_value(),
                swa_lo_raw.ir_value(),
            )
        )
        cond_swa = swa_lo > kv_left
        kv_left = cutlass.Int32(
            arith.select(
                cond_swa.ir_value(),
                swa_lo.ir_value(),
                kv_left.ir_value(),
            )
        )
    # Empty-loop clamp: if every block was masked out (degenerate causal/SWA
    # combination, e.g. q_row_base + tile_m < swa_window for SWA-only) push
    # kv_left up to kv_right so the mainloop runs zero iters and the LSE +
    # epilogue path sees row_sum=0 → STG writes inf for affected rows.
    empty_cond = kv_left > kv_right
    kv_left = cutlass.Int32(
        arith.select(
            empty_cond.ir_value(),
            kv_right.ir_value(),
            kv_left.ir_value(),
        )
    )
    # THD over-provisioned tile past this sequence → force 0 kv-iters.  The
    # row_sum stays 0 so the predicated O / LSE stores below write nothing for
    # these rows (they belong to the next packed sequence).
    if cutlass.const_expr(THD_VARLEN):
        kv_left = cutlass.Int32(
            arith.select(
                thd_invalid.ir_value(),
                kv_right.ir_value(),
                kv_left.ir_value(),
            )
        )
    # Non-negative clamp.  A very negative causal_diag (BR with eff_skv ≪ eff_sq,
    # e.g. a packed sequence whose KV length is far below its Q length) drives
    # causal_kv_hi negative → kv_right < 0.  The prologue below seeds tile
    # kv_right-1, so a negative kv_right would load K/V at a negative tile index
    # → negative GMEM offset → OOB read.  Such a CTA is fully masked (every q
    # attends k ≤ q + causal_diag < 0); clamp to 0 iters so the prologue seeds a
    # valid (unused) tile and the epilogue writes the masked-row result.
    kv_right = cutlass.Int32(
        arith.select(
            (kv_right < cutlass.Int32(0)).ir_value(),
            cutlass.Int32(0).ir_value(),
            kv_right.ir_value(),
        )
    )
    kv_left = cutlass.Int32(
        arith.select(
            (kv_left < cutlass.Int32(0)).ir_value(),
            cutlass.Int32(0).ir_value(),
            kv_left.ir_value(),
        )
    )
    kv_left = cutlass.Int32(
        arith.select(
            (kv_left > kv_right).ir_value(),
            kv_right.ir_value(),
            kv_left.ir_value(),
        )
    )

    # ---- INTERIOR vs BOUNDARY iter split for mask pre-pass ---------------
    # ``compute_kv_loop_bounds`` already trims fully-masked iters off the
    # ends.  The REMAINING iters split into:
    #   - INTERIOR iters [unmasked_lo, unmasked_hi) — every (q, k) pair
    #     is in-bounds; the per-element softmax mask pre-pass is a no-op
    #     (every select returns the original S value).  Skip it.
    #   - BOUNDARY iters [kv_left, unmasked_lo) ∪ [unmasked_hi, kv_right)
    #     — straddle the causal diagonal / SWA-window edge / padded SKV
    #     edge; apply the mask.
    # We keep ONE cute.range over [kv_left, kv_right) but gate the mask
    # pre-pass on a runtime ``needs_mask`` flag (a few cmp + or per iter
    # vs ~32 cmp + sel per element on interior iters → ~7-9 % win on
    # causal-heavy mainloops).
    unmasked_lo = kv_left
    unmasked_hi = kv_right
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        # Mask needed when MAX_k_in_tile > MIN_q_in_CTA + causal_diag, i.e.,
        #   (kv_iter + 1) * tile_n > q_row_base + 1 + causal_diag
        #   kv_iter >= (q_row_base + 1 + causal_diag) // tile_n
        causal_hi = (cutlass.Int32(q_row_base) + cutlass.Int32(1) + causal_diag) // cutlass.Int32(tile_n)
        cond = causal_hi < unmasked_hi
        unmasked_hi = cutlass.Int32(
            arith.select(
                cond.ir_value(),
                causal_hi.ir_value(),
                unmasked_hi.ir_value(),
            )
        )
    if cutlass.const_expr((mask_flags & MASK_SWA) != 0):
        # SWA: mask needed iff MIN_k_in_tile < MAX_q_in_CTA - W
        #   kv_iter * tile_n < q_row_base + tile_m - 1 - W
        #   kv_iter < (q_row_base + tile_m - 1 - W + tile_n) // tile_n
        # I.e., kv_iter < ceil((q_row_base + br_base + tile_m - W) / tile_n).
        swa_lo_raw = (cutlass.Int32(q_row_base) + br_base + cutlass.Int32(tile_m - swa_window) + cutlass.Int32(tile_n - 1)) // cutlass.Int32(tile_n)
        # Clamp to >= kv_left.
        cond_lo = swa_lo_raw > unmasked_lo
        unmasked_lo = cutlass.Int32(
            arith.select(
                cond_lo.ir_value(),
                swa_lo_raw.ir_value(),
                unmasked_lo.ir_value(),
            )
        )
    # PADDED needs the per-element mask on the boundary tile that straddles
    # eff_skv.  Implicit ~is_even_mn padding only ever overhangs on the LAST
    # tile; per-batch seq_kv_lens trims kv_right to ceil(eff_skv/tile_n) above
    # so the boundary tile is again the last kept iter.  Either way pull
    # unmasked_hi back to kv_right - 1 so that last iter applies the mask.
    if cutlass.const_expr((not is_even_mn) or has_seq_kv_lens or THD_VARLEN):
        last_iter = kv_right - cutlass.Int32(1)
        cond_pad = last_iter < unmasked_hi
        unmasked_hi = cutlass.Int32(
            arith.select(
                cond_pad.ir_value(),
                last_iter.ir_value(),
                unmasked_hi.ir_value(),
            )
        )
    # unmasked_lo <= unmasked_hi clamp (degenerate combo).
    clamp_cond = unmasked_lo > unmasked_hi
    unmasked_lo = cutlass.Int32(
        arith.select(
            clamp_cond.ir_value(),
            unmasked_hi.ir_value(),
            unmasked_lo.ir_value(),
        )
    )

    # ---- PROLOGUE: cp.async Q + K[0] -------------------------------------
    # Q and K[0] go into separate cp.async-groups so we can wait for Q
    # alone (group 0) while K[0] (group 1) continues to drain in the
    # background — that hides ~16 KiB of K load latency behind ldmatrix Q.
    # Row/col predication is enabled only when the corresponding Constexpr
    # ``is_even_*`` flag is False — fully eliminated otherwise.
    # THD packs sequences back-to-back, so valid_rows must be the PER-BATCH
    # length (eff_sq / eff_skv) — over-reading into the next sequence's rows
    # (or past T on the last sequence) would corrupt / fault.  Dense uses the
    # physical SQ / SKV.
    if cutlass.const_expr(THD_VARLEN):
        q_valid_rows = eff_sq
        kv_valid_rows = eff_skv
    else:
        q_valid_rows = None if cutlass.const_expr(is_even_mn) else sq_runtime
        kv_valid_rows = None if cutlass.const_expr(is_even_mn) else skv_runtime
    valid_cols = None if cutlass.const_expr(is_even_k) else d_runtime
    q_row_base_i = cutlass.Int32(q_row_base)
    load_tile_2d(
        sQ_buf,
        q_gmem,
        rows=tile_m,
        elems_per_row=ELEMS_PER_ROW_Q,
        gmem_row_stride_elems=Q_ROW_STRIDE_E,
        tidx=tidx,
        num_threads=threads,
        elems_per_copy=ELEMS_PER_LD,
        elem_bytes=ELEM_BYTES,
        cache=nvvm.LoadCacheModifier.CG,
        swizzle=True,
        valid_rows=q_valid_rows,
        valid_cols=valid_cols,
        row_base=q_row_base_i,
    )
    cp_async_commit()  # group 0: Q
    # ---- REVERSE iter direction (FA-style) ------------------------------
    # The mainloop walks kv_iter from kv_right - 1 down to kv_left.  Why
    # reverse?  For MASK_CAUSAL / MASK_SWA the boundary tiles (where the
    # mask actually kicks in) sit at the END of the [kv_left, kv_right)
    # range.  Iterating reverse places those at the TOP of the loop;
    # combined with the two-loop split that immediately follows the
    # boundary, the bulk-interior loop carries no mask code at all.
    #
    # SMEM-stage alternation uses the LOOP COUNTER ``i`` (0..N-1), NOT
    # ``kv_iter``, so the prologue can always seed stage 0 regardless of
    # kv_right's parity.
    # max(kv_right-1, 0): an empty loop (kv_left==kv_right==0, fully-masked CTA)
    # still seeds tile 0 — a valid, in-bounds load the 0-iter mainloop ignores.
    prologue_kv_iter = kv_right - cutlass.Int32(1)
    prologue_kv_iter = cutlass.Int32(
        arith.select(
            (prologue_kv_iter < cutlass.Int32(0)).ir_value(),
            cutlass.Int32(0).ir_value(),
            prologue_kv_iter.ir_value(),
        )
    )
    prologue_kv_i64 = prologue_kv_iter.to(cutlass.Int64)
    k_prologue_off64 = prologue_kv_i64 * K_TILE_STRIDE_E.to(cutlass.Int64)
    v_prologue_off64 = prologue_kv_i64 * V_TILE_STRIDE_E.to(cutlass.Int64)
    k_prologue_gmem = k_gmem_batch_head + k_prologue_off64
    kv_prologue_row_abs = prologue_kv_iter * cutlass.Int32(tile_n)
    load_tile_2d(
        sK_buf,
        k_prologue_gmem,
        rows=tile_n,
        elems_per_row=ELEMS_PER_ROW_K,
        gmem_row_stride_elems=K_ROW_STRIDE_E,
        tidx=tidx,
        num_threads=threads,
        elems_per_copy=ELEMS_PER_LD,
        elem_bytes=ELEM_BYTES,
        cache=nvvm.LoadCacheModifier.CG,
        swizzle=True,
        valid_rows=kv_valid_rows,
        valid_cols=valid_cols,
        row_base=kv_prologue_row_abs,
    )
    cp_async_commit()  # group 1: K[kv_right-1]
    cp_async_wait(1)  # leave K in flight
    nvvm.barrier_cta_sync()

    # ---- RoPE: rotate Q in SMEM (before ldmatrix Q) -----------------------
    # Q is rotated ONCE in the prologue (one CTA owns one Q tile).  Position of
    # Q row 0 = q_row_base.  The extra barrier fences the in-place rotation
    # writes against the cross-thread ldmatrix reads below.
    if cutlass.const_expr(has_rope):
        rope_rotate_smem_tile(
            sQ_buf, rope_cs_ptr, cutlass.Int32(q_row_base), rows=tile_m, d_qk=d_qk, tidx=tidx, threads=threads, io_dtype=io_dtype, elem_bytes=ELEM_BYTES
        )
        nvvm.barrier_cta_sync()

    # ---- ldmatrix Q → regs ------------------------------------------------
    # Each warp owns Q rows [warp_m_base, warp_m_base + m_per_warp).
    # For TILE_M=128, m_per_warp=32 = 2 m16n8k16 row-blocks per warp →
    # we ldmatrix BOTH m-blocks now (m_block * 16 row offset) and lay the
    # result out flat in Q_frag with m-block-major stride for the matching
    # ``mma_step`` indexing: ``Q_frag[m_block * QK_K_CHUNKS*4 + k_chunk*4 + i]``.
    # Per (m_block, k_chunk): 4 i32/lane via one ldmatrix.x4 .row.
    Q_frag = [None] * (m_blocks * QK_K_CHUNKS * 4)

    warp_m_base = warp_idx * m_per_warp  # 0, 32, 64, 96  for TILE_M=128
    a_row = lane % 16
    a_col_chunk = lane // 16
    for m_block in cutlass.range_constexpr(m_blocks):
        for k_chunk in cutlass.range_constexpr(QK_K_CHUNKS):
            # Same Swz128B XOR as the Q cp.async writer: read lands at the
            # physical SMEM slot the writer chose, but the per-warp
            # ldmatrix m8n8 atom now spreads its 8 row-ptrs across 8
            # disjoint bank groups.
            q_row = warp_m_base + m_block * 16 + a_row
            q_col_log = k_chunk * 16 + a_col_chunk * 8
            q_col_swz = swizzle_xor_128b(q_row, q_col_log, elem_bytes=ELEM_BYTES)
            ptr = sQ_buf.subview(q_row * ELEMS_PER_ROW_Q + q_col_swz)
            v = nvvm.ldmatrix(ptr.data_ptr(), 4, nvvm.MMALayout.ROW)
            base = m_block * QK_K_CHUNKS * 4 + k_chunk * 4
            Q_frag[base + 0] = v[0]
            Q_frag[base + 1] = v[1]
            Q_frag[base + 2] = v[2]
            Q_frag[base + 3] = v[3]

    # End-of-prologue barrier: all warps must finish ldmatrix Q before iter 0
    # issues V[0] cp.async into sQ_or_V (whose SMEM still holds Q data).
    nvvm.barrier_cta_sync()

    # ---- O accumulator (16 m16n8 frags / warp × 4 fp32/lane) --------------
    # O accumulator — m_blocks * SV_N_FRAGS * 4 fp32 per lane, m-block major.
    # mma_step indexes acc as ``acc[m_block * SV_N_FRAGS*4 + n_frag*4 + i]``.
    O_acc = cutlass.Array(cutlass.Float32, m_blocks * SV_N_FRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(m_blocks * SV_N_FRAGS * 4):
        O_acc[i] = cutlass.Float32(0.0)

    # ---- Online-softmax per-lane state ------------------------------------
    # Each m-block contributes 2 M-rows of softmax state per lane (mma D-frag
    # top + bottom 8x8 sub-blocks).  So per warp, per lane we track
    # ``2 * m_blocks`` row scalars for both m_max and row_sum.  Storage
    # layout: ``row_max[m_block * 2 + 0]`` = top row, ``+ 1`` = bot row.
    # Local-mem allocation (not Python literals) — the -inf init must
    # reach the trace as a real fp32 Value or the iter-0 α-rescale folds
    # away.
    row_max = cutlass.Array(cutlass.Float32, m_blocks * 2, alignment=16, space=cutlass.AddressSpace.rmem)
    row_sum = cutlass.Array(cutlass.Float32, m_blocks * 2, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(m_blocks * 2):
        row_max[i] = -cutlass.Float32.inf
        row_sum[i] = cutlass.Float32(0.0)
    g_lane = lane // 4  # 0..7 (mma D-frag row group within an m_block)
    p_lane = lane % 4  # 0..3 (mma D-frag col pos)

    def _imin_i32(a, b):  # min(a, b) for Int32 (IMNMX) — bias OOB clamp
        return cutlass.Int32(arith.select((a < b).ir_value(), a.ir_value(), b.ir_value()))

    # ---- MAINLOOP ---------------------------------------------------------
    # Both K and V are double-buffered (sK_buf split halves, sQ_buf split
    # halves).  At iter-i top we issue cp.async for both V[i] AND K[i+1]
    # (the next K), giving K[i+1] a full-iter drain window (QK + softmax +
    # V-wait + SV).  V[i] gets its usual half-iter drain (QK + softmax)
    # before the V-wait.
    #
    # ``n_kv_tiles`` is a RUNTIME value (host computes ``round_up(SKV /
    # TILE_N)`` and passes it in) so the kernel's outer loop uses a
    # runtime ``cutlass.range(..., unroll=1)`` — this collapses JIT
    # compile time from ~60 s (constexpr unroll over 128 iters at
    # SQ=8192) to a few seconds (one shared loop body trace).
    # Stage selection ``cur_stage = kv_iter & 1`` is therefore a runtime
    # bit-test, and stage pointers come from runtime ptr arithmetic.
    #
    # EXP-A: running pointers — V starts at v_gmem_batch_head (iter-0
    # K/V row), K-next starts at +TILE_N (iter-0 K[1] prefetch).  Each
    # iter advances both by KV_TILE_STRIDE_E.  If cute traces Python
    # rebind correctly across runtime ``cutlass.range`` iters, this
    # drops one Int32 mul per iter as well.  Tested: validation must
    # PASS (FAIL = rebind doesn't carry, revert).
    # Reverse iter: kv_iter walks kv_right-1 down to kv_left.  Running
    # pointers START at the kv_right-1 V/K block and DECREMENT each iter.
    # The K[i-1] prefetch (PREDECESSOR, not successor) is the one we issue
    # mid-loop; on the LAST iter (kv_iter == kv_left), the predecessor
    # K[kv_left - 1] would go OOB so we predicate it with cp_size=0.
    K_STRIDE_64 = K_TILE_STRIDE_E.to(cutlass.Int64)
    V_STRIDE_64 = V_TILE_STRIDE_E.to(cutlass.Int64)
    v_cur_gmem = v_gmem_batch_head + v_prologue_off64
    # K[i-1] prefetch starts at K[kv_right - 2] (= prologue's K - 1 stride).
    k_next_gmem = k_gmem_batch_head + k_prologue_off64 - K_STRIDE_64
    iter_count = kv_right - kv_left
    for i in cutlass.range(cutlass.Int32(0), iter_count, cutlass.Int32(1), unroll=1):
        kv_iter = prologue_kv_iter - i
        # SMEM stage alternates on the LOOP COUNTER so prologue's stage-0
        # write is the iter-0 read regardless of kv_right's parity.
        cur_stage = i & cutlass.Int32(1)
        next_stage = cutlass.Int32(1) - cur_stage
        sK_cur = sK_buf.subview(cur_stage * cutlass.Int32(ELEMS_KV_K))
        sK_next = sK_buf.subview(next_stage * cutlass.Int32(ELEMS_KV_K))
        sV_cur = sQ_buf.subview(cur_stage * cutlass.Int32(ELEMS_KV_V))
        sV_next = sQ_buf.subview(next_stage * cutlass.Int32(ELEMS_KV_V))

        # ---- QK mma : S[M=m_per_warp, N=TILE_N] per warp, regs accumulator
        # mma_step internally loops m_blocks m16n8k16 sub-calls per k_step,
        # so the kernel-side call is the same one-line shape it was at
        # TILE_M=64.  S_acc is flat, m-block major:
        #   ``S_acc[m_block * QK_N_FRAGS*4 + n_frag*4 + i]``.
        S_acc = cutlass.Array(cutlass.Float32, m_blocks * QK_N_FRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(m_blocks * QK_N_FRAGS * 4):
            S_acc[i] = cutlass.Float32(0.0)

        # ---- Step 1: issue V[i] cp.async ---------------------------------
        # V[i] writes into sV_cur (= sV[i%2]).  Iter (i-2)'s SV-mma read
        # of this same slot is fenced by iter (i-1)'s K-wait + V-wait
        # barriers, so no per-iter end-of-iter barrier is needed.
        # v_cur_gmem advanced at iter-end below (running-pointer form).
        # row_base = kv_iter * tile_n (V[i]'s absolute starting row).
        kv_row_base_cur = kv_iter * cutlass.Int32(tile_n)
        load_tile_2d(
            sV_cur,
            v_cur_gmem,
            rows=tile_n,
            elems_per_row=ELEMS_PER_ROW_V,
            gmem_row_stride_elems=V_ROW_STRIDE_E,
            tidx=tidx,
            num_threads=threads,
            elems_per_copy=ELEMS_PER_LD,
            elem_bytes=ELEM_BYTES,
            cache=nvvm.LoadCacheModifier.CG,
            swizzle=True,
            valid_rows=kv_valid_rows,
            valid_cols=None,  # V is always the full compile-time d_v wide (asserted)
            row_base=kv_row_base_cur,
        )
        cp_async_commit()  # group: V[i]

        # ---- Step 1.5: prefetch K[i-1] (predicated via cp_size) ----------
        # Always issue the prefetch cp.async and predicate it via
        # ``cp_size_bytes``: PTX cp.async with source-bytes < dst-bytes
        # zero-fills the remainder (here ALL bytes), making the predicated
        # instruction a benign SMEM zero-store with NO GMEM dereference.
        # Issued before this iter's K-wait so its drain is covered by the
        # whole QK + softmax + SV pipeline of this iteration.
        bytes_per_copy_full = ELEMS_PER_LD * ELEM_BYTES  # 16
        # Reverse iter: prefetch K[i-1] (the PREDECESSOR, not successor).
        # On the LAST iter (kv_iter == kv_left), K[kv_left-1] would go OOB
        # → predicate with cp_size=0.  Row predication (when ~is_even_mn)
        # ALSO zeroes OOB rows within the LAST valid K tile — they compose
        # multiplicatively inside load_tile_2d.
        not_last = cutlass.Int32(kv_iter > kv_left)
        cp_size_bytes = not_last * cutlass.Int32(bytes_per_copy_full)
        # k_next_gmem holds iter-i's K[i-1] base (decremented at iter-end).
        kv_row_base_prev = (kv_iter - cutlass.Int32(1)) * cutlass.Int32(tile_n)
        load_tile_2d(
            sK_next,
            k_next_gmem,
            rows=tile_n,
            elems_per_row=ELEMS_PER_ROW_K,
            gmem_row_stride_elems=K_ROW_STRIDE_E,
            tidx=tidx,
            num_threads=threads,
            elems_per_copy=ELEMS_PER_LD,
            elem_bytes=ELEM_BYTES,
            cache=nvvm.LoadCacheModifier.CG,
            swizzle=True,
            cp_size_bytes=cp_size_bytes,
            valid_rows=kv_valid_rows,
            valid_cols=valid_cols,
            row_base=kv_row_base_prev,
        )
        cp_async_commit()  # group: K[i-1] (no-op on first kv_iter == kv_left)

        # ---- Step 2: wait for K[i] ---------------------------------------
        # Pending groups (oldest first): K[i] (from the prev iter's prefetch
        # or the prologue), V[i] (Step 1), K[i-1] (the prefetch just issued).
        # wait_group(2) drains exactly K[i]; V[i] and K[i-1] keep streaming
        # through QK + softmax.
        cp_async_wait(2)
        nvvm.barrier_cta_sync()

        # ---- RoPE: rotate K[i] in SMEM (before ldmatrix K) ----------------
        # Each K tile is loaded once and consumed once, so rotating sK_cur in
        # place per kv-iter is correct (the prefetch wrote sK_next, a disjoint
        # ring slot).  Position of K row 0 in this tile = kv_iter*tile_n.  The
        # extra barrier fences the rotation writes against the ldmatrix reads.
        if cutlass.const_expr(has_rope):
            rope_rotate_smem_tile(
                sK_cur, rope_cs_ptr, kv_row_base_cur, rows=tile_n, d_qk=d_qk, tidx=tidx, threads=threads, io_dtype=io_dtype, elem_bytes=ELEM_BYTES
            )
            nvvm.barrier_cta_sync()

        # ---- Bias frag register-prefetch (overlap GMEM read with QK mma) --
        # Load THIS kv-iter's additive-bias tile into a fp32 frag (layout ==
        # S_acc), folding inv_softmax_scale in at load time.  Issued BEFORE
        # the QK mma below so the GMEM latency hides behind the mma compute
        # window; injected in the softmax.  OOB on the last uneven tile
        # (q >= SQ / col >= SKV) is clamped to physical bounds — those cells
        # are masked / not stored, so the clamped value is discarded.
        if cutlass.const_expr(has_bias):
            bias_frag = cutlass.Array(cutlass.Float32, m_blocks * QK_N_FRAGS * 4, alignment=16, space=cutlass.AddressSpace.rmem)
            bias_pp = Pointer(bias_base, dtype=bias_dt)
            bias_kv_col_base = kv_iter * cutlass.Int32(tile_n)
            for m_block in cutlass.range_constexpr(m_blocks):
                bbase = m_block * QK_N_FRAGS * 4
                bq_top = cutlass.Int32(q_row_base) + cutlass.Int32(warp_m_base) + cutlass.Int32(m_block * 16) + g_lane
                bq_bot = bq_top + cutlass.Int32(8)
                if cutlass.const_expr(not is_even_mn):
                    bq_top = _imin_i32(bq_top, sq_runtime - cutlass.Int32(1))
                    bq_bot = _imin_i32(bq_bot, sq_runtime - cutlass.Int32(1))
                bq_top64 = cutlass.Int64(bq_top) * cutlass.Int64(SKV)
                bq_bot64 = cutlass.Int64(bq_bot) * cutlass.Int64(SKV)
                for k in cutlass.range_constexpr(QK_N_FRAGS):
                    bca = bias_kv_col_base + cutlass.Int32(k * 8) + cutlass.Int32(2) * p_lane
                    bcb = bca + cutlass.Int32(1)
                    if cutlass.const_expr(not is_even_mn):
                        bca = _imin_i32(bca, skv_runtime - cutlass.Int32(1))
                        bcb = _imin_i32(bcb, skv_runtime - cutlass.Int32(1))
                    bca64 = cutlass.Int64(bca)
                    bcb64 = cutlass.Int64(bcb)
                    bias_frag[bbase + k * 4 + 0] = bias_pp[bq_top64 + bca64].to(cutlass.Float32) * inv_softmax_scale
                    bias_frag[bbase + k * 4 + 1] = bias_pp[bq_top64 + bcb64].to(cutlass.Float32) * inv_softmax_scale
                    bias_frag[bbase + k * 4 + 2] = bias_pp[bq_bot64 + bca64].to(cutlass.Float32) * inv_softmax_scale
                    bias_frag[bbase + k * 4 + 3] = bias_pp[bq_bot64 + bcb64].to(cutlass.Float32) * inv_softmax_scale

        # Outer k-loop with one-ahead K-frag prefetch: at iter k we already
        # hold K[k]'s frag and issue ldmatrix(K[k+1]) BEFORE calling
        # mma_step on K[k], so the next-step's SMEM→reg latency overlaps
        # the current mma's compute window.  Prologue load primes K[0];
        # the last iter skips the prefetch since K[K_CHUNKS] doesn't exist.
        # ``b_trans=False`` for K — K SMEM row = N_kv = mma N, col = d_qk =
        # mma K dim → ldmatrix.x2 .row, no transpose at load.
        # B-operand (K) via ldmatrix.x4 — halves the inner-loop ldmatrix
        # instruction count vs the x2 variant (load 2 adjacent n_frags per
        # call).  tile_n must be a multiple of 16 (N//8 even).
        K_frag_cur = load_b_smem_x4(sK_cur, k_step=0, N=tile_n, sB_elems_per_row=ELEMS_PER_ROW_K, b_trans=False, lane=lane, swizzle=True, elem_bytes=ELEM_BYTES)
        for k_chunk in cutlass.range_constexpr(QK_K_CHUNKS):
            if cutlass.const_expr(k_chunk + 1 < QK_K_CHUNKS):
                K_frag_next = load_b_smem_x4(
                    sK_cur, k_step=k_chunk + 1, N=tile_n, sB_elems_per_row=ELEMS_PER_ROW_K, b_trans=False, lane=lane, swizzle=True, elem_bytes=ELEM_BYTES
                )
            mma_step(S_acc, Q_frag, K_frag_cur, k_step=k_chunk, M=m_per_warp, N=tile_n, ab_dtype=io_dtype)
            if cutlass.const_expr(k_chunk + 1 < QK_K_CHUNKS):
                K_frag_cur = K_frag_next

        # ---- ONLINE SOFTMAX -----------------------------------------------
        # Per-lane: rowwise max → reduce → update m_state → rescale O & l →
        # exp2 to P → reduce sum → update l_state → pack P→half2 in regs.
        # All scaling via ``softmax_scale_log2`` is folded into the exp2
        # input late (so the row_max state stays un-scaled), matching the
        # DKG blackwell_geforce FMHA reference convention.

        # Per m_block: each warp owns 2 m16-row blocks worth of softmax
        # state.  Run the full max/butterfly/α-rescale/exp/pack pipeline
        # m_block at a time — the K[i+1] prefetch already overlapped the
        # earlier QK mma so we don't need to interleave m_blocks here.
        # P_frag layout: m-block major, ``P_frag[m_block * QK_N_FRAGS*2 +
        # k*2 + half]``.
        # ``MASK_PADDED`` is auto-ORed when ~is_even_mn so OOB-K columns
        # (last KV block when SKV%tile_n != 0) get -FLT_MAX before the
        # row-max reduction.  When both is_even_mn AND the user-supplied
        # mask_flags are clean, ``_effective_mask == MASK_NONE`` and the
        # mask pre-pass below traces to nothing.
        _effective_mask = mask_flags
        if cutlass.const_expr(not is_even_mn):
            _effective_mask = _effective_mask | MASK_PADDED
        if cutlass.const_expr(has_seq_kv_lens):
            _effective_mask = _effective_mask | MASK_PADDED
        if cutlass.const_expr(THD_VARLEN):
            # Mask KV cols >= per-batch eff_skv (the boundary tile straddles
            # the sequence's KV length in the packed layout).
            _effective_mask = _effective_mask | MASK_PADDED
        NEG_MASK_VAL = cutlass.Float32(-3.4028235e38)
        kv_col_base = kv_iter * cutlass.Int32(tile_n)

        # Runtime check: do we need the mask pre-pass for THIS iter?  The
        # [unmasked_lo, unmasked_hi) range was computed at kernel entry as
        # the iters where every (q, k) pair in the tile is in-bounds.  When
        # the mask is off entirely (_effective_mask == NONE) the entire
        # block below is constexpr-eliminated and ``needs_mask`` never
        # materializes.
        if cutlass.const_expr(_effective_mask != MASK_NONE):
            needs_mask = (kv_iter < unmasked_lo) | (kv_iter >= unmasked_hi)

        P_frag = [None] * (m_blocks * QK_N_FRAGS * 2)
        for m_block in cutlass.range_constexpr(m_blocks):
            s_base = m_block * QK_N_FRAGS * 4
            o_base = m_block * SV_N_FRAGS * 4
            p_base = m_block * QK_N_FRAGS * 2
            row_state_lo = m_block * 2  # row_max/sum[row_state_lo+0] = top
            row_state_hi = m_block * 2 + 1  #                    [+1] = bot

            # 0z) Additive attention bias (pre-scale).  bias_frag was loaded
            #     at top-of-iter (overlapping the QK mma) with inv_softmax_scale
            #     already folded in; its layout matches S_acc, so a flat add
            #     over the 4*QK_N_FRAGS lane elements injects the whole tile.
            #     Added before the mask block (masked cols overwritten to
            #     -FLT_MAX) — matches the reference order ``s += bias`` then mask.
            if cutlass.const_expr(has_bias):
                for j in cutlass.range_constexpr(QK_N_FRAGS * 4):
                    S_acc[s_base + j] = S_acc[s_base + j] + bias_frag[s_base + j]

            # 0) Per-element attention mask (PADDED / CAUSAL / SWA).
            #    Sets S_acc[masked_col] = -FLT_MAX so the column does not
            #    contribute to row max / row sum.  Each branch is gated by
            #    a Constexpr — fully eliminated when the flag bit is off.
            #    Nested runtime ``if needs_mask:`` so INTERIOR iters
            #    (where every k is in-bounds) skip the per-element selects
            #    entirely — saves ~32 cmp+sel / iter / m_block.
            if cutlass.const_expr(_effective_mask != MASK_NONE):
                if needs_mask:
                    # Absolute Q-rows owned by this lane within this m_block:
                    #   top = q_row_base + warp_m_base + m_block*16 + g_lane
                    #   bot = top + 8
                    q_top_abs = cutlass.Int32(q_row_base) + cutlass.Int32(warp_m_base) + cutlass.Int32(m_block * 16) + g_lane
                    q_bot_abs = q_top_abs + cutlass.Int32(8)
                    # SWA lower bound: keep col >= q + br_base - swa_window (br_base
                    # shifts the window to the BR corner; 0 for top-left).
                    q_top_minus_w = q_top_abs + br_base - cutlass.Int32(swa_window) if (_effective_mask & MASK_SWA) else None
                    q_bot_minus_w = q_bot_abs + br_base - cutlass.Int32(swa_window) if (_effective_mask & MASK_SWA) else None
                    for k in cutlass.range_constexpr(QK_N_FRAGS):
                        col_a = kv_col_base + cutlass.Int32(k * 8) + cutlass.Int32(2) * p_lane
                        col_b = col_a + cutlass.Int32(1)

                        # Per-pair (col_a or col_b, q_top_abs or q_bot_abs) →
                        # build the masked-out boolean from active mask bits.
                        def _mask_term(col, q_abs, q_mw):
                            m = None
                            if cutlass.const_expr(_effective_mask & MASK_PADDED):
                                t = col >= eff_skv
                                m = t
                            if cutlass.const_expr(_effective_mask & MASK_CAUSAL):
                                # causal_diag folds bottom-right (SKV-SQ) and/or the
                                # runtime right band; on plain top-left causal it is 0
                                # so q_lim == q_abs (SASS unchanged).
                                q_lim = q_abs + causal_diag
                                t = col > q_lim
                                m = t if m is None else (m | t)
                            if cutlass.const_expr(_effective_mask & MASK_SWA):
                                t = col < q_mw
                                m = t if m is None else (m | t)
                            return m

                        for off, col_e, q_abs_e, q_mw_e in (
                            (0, col_a, q_top_abs, q_top_minus_w),
                            (1, col_b, q_top_abs, q_top_minus_w),
                            (2, col_a, q_bot_abs, q_bot_minus_w),
                            (3, col_b, q_bot_abs, q_bot_minus_w),
                        ):
                            idx = s_base + k * 4 + off
                            masked = _mask_term(col_e, q_abs_e, q_mw_e)
                            S_acc[idx] = cutlass.Float32(
                                arith.select(
                                    masked.ir_value(),
                                    NEG_MASK_VAL.ir_value(),
                                    S_acc[idx].ir_value(),
                                )
                            )

            # 1) Within-lane row max over the 16 N elements this lane owns
            #    per row (8 n_frags × 2 cols-per-frag).  Wrap operands in
            #    ``cutlass.Float32(...)`` — required for cute.math.max (the
            #    underlying NVVM op rejects compile-time constexprs; this
            #    coerces to a proper Value).  ftz=True keeps the FMNMX3 fusion.
            m_top_iter = cutlass.Float32(-cutlass.Float32.inf)
            m_bot_iter = cutlass.Float32(-cutlass.Float32.inf)
            for k in cutlass.range_constexpr(QK_N_FRAGS):
                s0 = S_acc[s_base + k * 4 + 0]
                s1 = S_acc[s_base + k * 4 + 1]
                s2 = S_acc[s_base + k * 4 + 2]
                s3 = S_acc[s_base + k * 4 + 3]
                m_top_iter = cutlass.Float32(cute.math.max(m_top_iter, cutlass.Float32(cute.math.max(s0, s1, ftz=True)), ftz=True))
                m_bot_iter = cutlass.Float32(cute.math.max(m_bot_iter, cutlass.Float32(cute.math.max(s2, s3, ftz=True)), ftz=True))

            # 2) Cross-lane butterfly max across the 4 lanes that share a
            #    row (lanes 4g..4g+3): XOR-2 then XOR-1 → all 4 lanes hold
            #    the row's max.  Fold softmax_scale_log2 into the iter max
            #    once here so row_max state is stored pre-scaled — saves
            #    one mul each in alpha and in neg_m below.
            m_top_iter = cutlass.Float32(cute.math.max(m_top_iter, nvvm.shfl_sync(0xFFFFFFFF, m_top_iter, 2, 0x1F, nvvm.Shfl.BFLY), ftz=True))
            m_top_iter = cutlass.Float32(cute.math.max(m_top_iter, nvvm.shfl_sync(0xFFFFFFFF, m_top_iter, 1, 0x1F, nvvm.Shfl.BFLY), ftz=True))
            m_top_iter = m_top_iter * softmax_scale_log2
            m_bot_iter = cutlass.Float32(cute.math.max(m_bot_iter, nvvm.shfl_sync(0xFFFFFFFF, m_bot_iter, 2, 0x1F, nvvm.Shfl.BFLY), ftz=True))
            m_bot_iter = cutlass.Float32(cute.math.max(m_bot_iter, nvvm.shfl_sync(0xFFFFFFFF, m_bot_iter, 1, 0x1F, nvvm.Shfl.BFLY), ftz=True))
            m_bot_iter = m_bot_iter * softmax_scale_log2

            # 3) Merge with global row-max state under RESCALE_THRESHOLD skip
            #    (only update the running max when the new iter exceeds it
            #    by > THRESH; otherwise α=1 and the O / row_sum rescale is
            #    skipped entirely via warp vote).  row_max stores the SCALED
            #    max (m * scale_log2), so the threshold is in log2 units.
            #    8.0 matches the upstream
            #    `_common.rescale_threshold(...)` value for BF16/FP16/TF32:
            #    defer the rescale to amortize the α=1 fast path.
            RESCALE_THRESHOLD = cutlass.Float32(8.0)
            m_top_prev = row_max[row_state_lo]
            m_bot_prev = row_max[row_state_hi]
            update_top = (m_top_iter - m_top_prev) > RESCALE_THRESHOLD
            update_bot = (m_bot_iter - m_bot_prev) > RESCALE_THRESHOLD
            m_top_new = cutlass.Float32(
                arith.select(
                    update_top.ir_value(),
                    m_top_iter.ir_value(),
                    m_top_prev.ir_value(),
                )
            )
            m_bot_new = cutlass.Float32(
                arith.select(
                    update_bot.ir_value(),
                    m_bot_iter.ir_value(),
                    m_bot_prev.ir_value(),
                )
            )
            # When the running max did NOT update, m_prev - m_new = 0 →
            # exp2(0) = 1 (exact, no rescale needed).  On iter 0, m_prev =
            # -inf → exp2(-inf) = 0 (zeroes the O acc — already zero-inited
            # in prologue, so no-op).
            alpha_top = cute.math.exp2(m_top_prev - m_top_new, fastmath=True)
            alpha_bot = cute.math.exp2(m_bot_prev - m_bot_new, fastmath=True)
            row_max[row_state_lo] = m_top_new
            row_max[row_state_hi] = m_bot_new

            # 4) Warp-wide vote: if EVERY lane sees α==1 for both rows,
            #    the rescale of O / row_sum is a no-op and we can skip
            #    the full O sweep (64 fmul / lane).
            alpha_is_one = (alpha_top == cutlass.Float32(1.0)) & (alpha_bot == cutlass.Float32(1.0))
            all_alpha_one = vote_sync(
                0xFFFFFFFF,
                alpha_is_one,
                VoteSync.ALL,
            )

            # 6) Compute P = exp2(scale_log2 * S - m_new_scaled) and pack
            #    into per-n_frag half2 register fragments for SV mma.
            #    m_*_new is already pre-scaled, so subtract directly (saves
            #    the prep mul + the neg-stash temp).  Per QK n_frag k:
            #    P_frag[p_base + k*2 + 0/1] holds (top, bot).  SV's k_step
            #    K consumes QK n_frags 2K, 2K+1 via the (top, bot, top,
            #    bot) tiling — the m-block-major stride in P_frag is
            #    QK_N_FRAGS*2.
            l_top_iter = cutlass.Float32(0.0)
            l_bot_iter = cutlass.Float32(0.0)
            for k in cutlass.range_constexpr(QK_N_FRAGS):
                x_top_0 = S_acc[s_base + k * 4 + 0] * softmax_scale_log2 - m_top_new
                x_top_1 = S_acc[s_base + k * 4 + 1] * softmax_scale_log2 - m_top_new
                x_bot_0 = S_acc[s_base + k * 4 + 2] * softmax_scale_log2 - m_bot_new
                x_bot_1 = S_acc[s_base + k * 4 + 3] * softmax_scale_log2 - m_bot_new
                p_top_0 = cute.math.exp2(x_top_0, fastmath=True)
                p_top_1 = cute.math.exp2(x_top_1, fastmath=True)
                p_bot_0 = cute.math.exp2(x_bot_0, fastmath=True)
                p_bot_1 = cute.math.exp2(x_bot_1, fastmath=True)
                l_top_iter = l_top_iter + p_top_0 + p_top_1
                l_bot_iter = l_bot_iter + p_bot_0 + p_bot_1
                P_frag[p_base + k * 2 + 0] = fp32_to_fp16(p_top_0, p_top_1, dtype=io_dtype)
                P_frag[p_base + k * 2 + 1] = fp32_to_fp16(p_bot_0, p_bot_1, dtype=io_dtype)

            # 5) Rescale O accumulator + l_state for this m_block (skipped
            #    when no lane saw a max update this iter).
            if ~all_alpha_one:
                for n in cutlass.range_constexpr(SV_N_FRAGS):
                    O_acc[o_base + n * 4 + 0] = O_acc[o_base + n * 4 + 0] * alpha_top
                    O_acc[o_base + n * 4 + 1] = O_acc[o_base + n * 4 + 1] * alpha_top
                    O_acc[o_base + n * 4 + 2] = O_acc[o_base + n * 4 + 2] * alpha_bot
                    O_acc[o_base + n * 4 + 3] = O_acc[o_base + n * 4 + 3] * alpha_bot

            # 7) Accumulate this iter's PER-LANE partial l into row_sum.
            #    Threadquad butterfly deferred to AFTER the mainloop (same
            #    correctness arg as TILE_M=64 — α is row-wide via the
            #    butterflied m_max).
            row_sum[row_state_lo] = row_sum[row_state_lo] * alpha_top + l_top_iter
            row_sum[row_state_hi] = row_sum[row_state_hi] * alpha_bot + l_bot_iter

        # ---- Step 5: wait for V[i] --------------------------------------
        # Pending: V[i], K[i+1] (always — even if last iter where K[i+1]
        # is a cp_size=0 no-op).  wait_group(1) drains V[i] (oldest) and
        # leaves K[i+1] streaming through SV mma + next iter (or harmless
        # zero-fill no-op on the last iter).
        cp_async_wait(1)
        nvvm.barrier_cta_sync()

        # ---- Step 6: SV mma ---------------------------------------------
        # V[i] is now resident in sV_cur (= sV[i%2]).  Same one-ahead
        # V-frag ldmatrix prefetch within the k-loop as QK does for K.
        # SV B-operand (V) also via ldmatrix.x4 — bigger win here than QK
        # because d_v // 8 is large (32 at d_v=256, 16 at d_v=128), so the
        # halving cuts the per-k-step ldmatrix count from {32,16} to {16,8}.
        V_frag_cur = load_b_smem_x4(sV_cur, k_step=0, N=d_v, sB_elems_per_row=ELEMS_PER_ROW_V, b_trans=True, lane=lane, swizzle=True, elem_bytes=ELEM_BYTES)
        for k_chunk in cutlass.range_constexpr(SV_K_CHUNKS):
            if cutlass.const_expr(k_chunk + 1 < SV_K_CHUNKS):
                V_frag_next = load_b_smem_x4(
                    sV_cur, k_step=k_chunk + 1, N=d_v, sB_elems_per_row=ELEMS_PER_ROW_V, b_trans=True, lane=lane, swizzle=True, elem_bytes=ELEM_BYTES
                )
            mma_step(O_acc, P_frag, V_frag_cur, k_step=k_chunk, M=m_per_warp, N=d_v, ab_dtype=io_dtype)
            if cutlass.const_expr(k_chunk + 1 < SV_K_CHUNKS):
                V_frag_cur = V_frag_next

        # No per-iter end-of-iter barrier:  with V double-buffered, iter
        # (i+2)'s V cp.async write to sV[i%2] is naturally fenced by
        # iter (i+1)'s K-wait + V-wait barriers — every warp finishes
        # iter i's SV-mma read before iter (i+2) issues its V cp.async.

        # EXP-A: advance running pointers for iter (i+1).  Int64 add only
        # (K/V stride already cast outside the loop).  K and V have separate
        # strides for asymmetric d_qk != d_v (DSv3).
        v_cur_gmem = v_cur_gmem - V_STRIDE_64
        k_next_gmem = k_next_gmem - K_STRIDE_64

    # ---- Post-mainloop sync ----------------------------------------------
    # One barrier here replaces the (dropped) per-iter end-of-iter barrier:
    # we just need all warps to have finished the LAST iter's SV mma before
    # the epilogue starts.  Without this, a fast warp could begin LSE/STG
    # writes while a slow warp is still in the last SV mma — harmless for
    # the per-warp data flow (each warp owns disjoint M rows), but cleaner
    # to fence here as a single sync point.
    cp_async_wait(0)
    nvvm.barrier_cta_sync()

    # ---- Final cross-lane butterfly sum (deferred from the mainloop) -----
    # All 4 lanes of a threadquad share one (row_top, row_bot) pair per
    # m_block; sum their per-lane partials via two XOR-2 / XOR-1
    # butterflies.  ``row_sum[m_block * 2 + 0/1]`` covers top/bot.
    for i in cutlass.range_constexpr(m_blocks * 2):
        row_sum[i] = row_sum[i] + nvvm.shfl_sync(0xFFFFFFFF, row_sum[i], 2, 0x1F, nvvm.Shfl.BFLY)
        row_sum[i] = row_sum[i] + nvvm.shfl_sync(0xFFFFFFFF, row_sum[i], 1, 0x1F, nvvm.Shfl.BFLY)

    # ---- Attention sink (denominator-only, finalized post-mainloop) -------
    # The per-Q-head sink logit joins the softmax DENOMINATOR (V_sink = 0 → no
    # numerator).  ``sinks[head]`` is in log2-of-scaled-logit units (host
    # pre-multiplied by log2(e)), matching row_max.  Lift the running max with
    # the sink (== reference ``max(row_max, sink)``) so empty / low-max rows
    # stay numerically safe, rescale O_acc + row_sum by ``exp2(old_max -
    # lifted)``, then add the sink's ``exp2(sink - lifted)`` to the
    # denominator.  os == 1 whenever sink <= max (the common case), so the
    # O rescale is a no-op there; for a fully-masked row (max = -inf) os = 0
    # zeroes O and the denominator becomes exp2(0) = 1 → O = 0, lse = sink.
    if cutlass.const_expr(has_sink):
        _sink_ptr = cutlass.make_array_view(sinks).data_ptr()
        sink_log2_h = Pointer(_sink_ptr, dtype=cutlass.Float32)[head_idx]
        for m_block in cutlass.range_constexpr(m_blocks):
            sk_o_base = m_block * SV_N_FRAGS * 4
            sk_lo = m_block * 2
            sk_hi = m_block * 2 + 1
            m_top_lift = cutlass.Float32(cute.math.max(row_max[sk_lo], sink_log2_h, ftz=True))
            m_bot_lift = cutlass.Float32(cute.math.max(row_max[sk_hi], sink_log2_h, ftz=True))
            os_top = cute.math.exp2(row_max[sk_lo] - m_top_lift, fastmath=True)
            os_bot = cute.math.exp2(row_max[sk_hi] - m_bot_lift, fastmath=True)
            row_sum[sk_lo] = row_sum[sk_lo] * os_top + cute.math.exp2(sink_log2_h - m_top_lift, fastmath=True)
            row_sum[sk_hi] = row_sum[sk_hi] * os_bot + cute.math.exp2(sink_log2_h - m_bot_lift, fastmath=True)
            row_max[sk_lo] = m_top_lift
            row_max[sk_hi] = m_bot_lift
            for n in cutlass.range_constexpr(SV_N_FRAGS):
                O_acc[sk_o_base + n * 4 + 0] = O_acc[sk_o_base + n * 4 + 0] * os_top
                O_acc[sk_o_base + n * 4 + 1] = O_acc[sk_o_base + n * 4 + 1] * os_top
                O_acc[sk_o_base + n * 4 + 2] = O_acc[sk_o_base + n * 4 + 2] * os_bot
                O_acc[sk_o_base + n * 4 + 3] = O_acc[sk_o_base + n * 4 + 3] * os_bot

    # ---- LSE output: lse[r] = scale * m_max[r] + ln(row_sum[r]) ----------
    # row_max stores the pre-scaled max (m * scale_log2), so scale*m_max =
    # ln(2)*row_max — fold both terms under one LN2 multiply.
    # Per m_block: write 2 fp32 (top, bot) — 4 lanes share each row's state
    # so the per-row write is benignly multi-issued by the threadquad.
    LN2 = cutlass.Float32(0.6931471805599453)
    q_row_base_i32 = cutlass.Int32(q_row_base)
    # Store row bound: dense writes padded rows (q < SQ) as zeros into the
    # [B,SQ,H,D] tensor; THD must NOT write rows past this sequence (they
    # belong to the next packed sequence) → bound by the per-batch eff_sq.
    sq_store_bound = eff_sq if cutlass.const_expr(THD_VARLEN) else sq_runtime
    # LSE output is None-specialized: no Stats output -> the whole block
    # (compute + stores) is compiled out.
    if cutlass.const_expr(LSE is not None):
        for m_block in cutlass.range_constexpr(m_blocks):
            block_warp_row = warp_m_base + m_block * 16
            block_row_top = block_warp_row + g_lane
            block_row_bot = block_warp_row + g_lane + 8
            row_state_lo = m_block * 2
            row_state_hi = m_block * 2 + 1
            lse_top = LN2 * (row_max[row_state_lo] + cute.math.log2(row_sum[row_state_lo]))
            lse_bot = LN2 * (row_max[row_state_hi] + cute.math.log2(row_sum[row_state_hi]))
            # Dense PADDED (per-batch seq_len_q): padded query rows q >= eff_sq are
            # within this batch's [B,SQ,..] slice but don't exist → lse = -inf
            # (log-sum-exp of a fully-masked row; matches cuDNN >= 9.14, and what
            # test_mhas_v2 expects for stats on padded rows).  Applied BEFORE the
            # store-path split below: the is_even_mn fast path stores every row, so
            # trimming only in the predicated path left finite LSE on padded rows
            # whenever SQ is tile-aligned.  THD is bounded by sq_store_bound==eff_sq
            # instead (must NOT write the next packed seq's rows), so this select is
            # gated on has_seq_len_q only.  The SM80 bprop reads this lse and masks
            # P=0 for padded rows via a select (inf->0, no NaN), so the -inf is safe
            # downstream.
            if cutlass.const_expr(has_seq_len_q):
                _ninf = cutlass.Float32(float("-inf"))
                trim_top = q_row_base_i32 + block_row_top
                trim_bot = q_row_base_i32 + block_row_bot
                lse_top = cutlass.Float32(arith.select((trim_top < eff_sq).ir_value(), lse_top.ir_value(), _ninf.ir_value()))
                lse_bot = cutlass.Float32(arith.select((trim_bot < eff_sq).ir_value(), lse_bot.ir_value(), _ninf.ir_value()))
            lse_top_ptr = lse_gmem + cutlass.Int64(block_row_top) * LSE_S_STRIDE_E
            lse_bot_ptr = lse_gmem + cutlass.Int64(block_row_bot) * LSE_S_STRIDE_E

            # Row predication for OOB Q-rows when ~is_even_mn.  All 4 lanes of a
            # threadquad share the same row → predicate is uniform within the
            # threadquad, so the if-branch traces identically across them.
            def _stf(ptr, val):
                cutlass.Array(ptr, (1,), dtype=cutlass.Float32)[0] = val

            if cutlass.const_expr(is_even_mn):
                _stf(lse_top_ptr, lse_top)
                _stf(lse_bot_ptr, lse_bot)
            else:
                top_abs = q_row_base_i32 + block_row_top
                bot_abs = q_row_base_i32 + block_row_bot
                if top_abs < sq_store_bound:
                    _stf(lse_top_ptr, lse_top)
                if bot_abs < sq_store_bound:
                    _stf(lse_bot_ptr, lse_bot)

    # ---- Final normalization + SMEM-staged STG.128 epilogue --------------
    # Naïve scalar per-lane STG (the previous epilogue) emits 2-fp16 stores
    # at row_top / row_bot positions that are 8 rows apart, yielding ~2.05
    # wavefronts/request and ~27 % wasted store bandwidth (NCU flagged this).
    #
    # New path: stage the warp's mma D-frag through SMEM with the Swz128B
    # XOR layout, then have ALL ``threads`` lanes cooperate on 16-byte
    # STG.128 stores (ld.shared.v4.b32 → st.global.v4.b32).  Re-uses
    # ``sQ_buf`` (32 KiB) as the O staging buffer — it's free after the
    # last SV mma + post-loop barrier.  Layout: ``sO_buf[TILE_M][D_V]``
    # row-major fp16, swizzled so the per-lane 2-fp16 D-frag stores and
    # the cooperative 8-fp16 reads are both bank-conflict-free.
    #
    # Layout details (Swz128B, elem_bytes=2, chunk_elems=8):
    #   * SMEM write: lane (g, p), per n_frag, writes 2 fp16 (= 1 b32) at
    #     (row = warp_row+g[+8], col = n_frag*8 + 2p).  chunk_idx = n_frag.
    #     swz_chunk = n_frag XOR (row & 7).
    #   * SMEM read: per-thread 16-B chunk at (row = chunk_glob / 16,
    #     col_chunk = chunk_glob % 16).  swz_chunk = col_chunk XOR (row & 7).
    sO_buf = sQ_buf

    ELEMS_PER_CHUNK = 16 // ELEM_BYTES  # 8 fp16 per 16-B chunk
    CHUNKS_PER_ROW = d_v // ELEMS_PER_CHUNK  # chunks per O row (16 @ d=128, 8 @ d=64)
    TOTAL_CHUNKS = tile_m * CHUNKS_PER_ROW  # full O tile in 16-B chunks
    # Constexpr-fold the wave count for the cooperative STG.128 loop.
    n_waves = TOTAL_CHUNKS // threads

    # Per-warp SMEM write of mma D-frag, with inv_l fused in at cast time.
    for m_block in cutlass.range_constexpr(m_blocks):
        o_base = m_block * SV_N_FRAGS * 4
        row_state_lo = m_block * 2
        row_state_hi = m_block * 2 + 1
        # Fully-masked Q rows (causal / SWA can leave a CTA with zero
        # unmasked KV blocks → row_sum stays at 0).  Without this guard
        # ``1 / 0 == inf`` propagates into O as NaN.  Select 0 instead so
        # those rows output zeros (and LSE writes ``-inf`` naturally via
        # ``log2(0) = -inf``).  When mask_flags == NONE the row_sum is
        # always > 0 (every Q row has at least one valid K col) and the
        # select traces to a single cmp+sel — negligible on the fast path.
        rs_top = row_sum[row_state_lo]
        rs_bot = row_sum[row_state_hi]
        inv_l_top = cutlass.Float32(
            arith.select(
                (rs_top > cutlass.Float32(0.0)).ir_value(),
                (cutlass.Float32(1.0) / rs_top).ir_value(),
                cutlass.Float32(0.0).ir_value(),
            )
        )
        inv_l_bot = cutlass.Float32(
            arith.select(
                (rs_bot > cutlass.Float32(0.0)).ir_value(),
                (cutlass.Float32(1.0) / rs_bot).ir_value(),
                cutlass.Float32(0.0).ir_value(),
            )
        )
        # Dense PADDED (per-batch seq_len_q): a padded query row q >= eff_sq is a
        # nonexistent query that still computed a normal softmax (row_sum > 0) over
        # the valid K cols → its O would be a meaningful attention output rather
        # than 0.  cuDNN zeroes padded output rows (and we already write lse = -inf
        # for them above); force inv_l = 0 so O = O_acc * 0 = 0, matching cuDNN and
        # keeping the full [B,SQ,H,D] tensor well-defined for direct users.  THD is
        # bounded by sq_store_bound == eff_sq (padded rows never stored), so gate on
        # has_seq_len_q only; folds away on the fast path.
        if cutlass.const_expr(has_seq_len_q):
            _row_top_abs = q_row_base_i32 + warp_m_base + cutlass.Int32(m_block * 16) + g_lane
            _row_bot_abs = _row_top_abs + cutlass.Int32(8)
            inv_l_top = cutlass.Float32(arith.select((_row_top_abs < eff_sq).ir_value(), inv_l_top.ir_value(), cutlass.Float32(0.0).ir_value()))
            inv_l_bot = cutlass.Float32(arith.select((_row_bot_abs < eff_sq).ir_value(), inv_l_bot.ir_value(), cutlass.Float32(0.0).ir_value()))
        block_warp_row = warp_m_base + m_block * 16
        block_row_top = block_warp_row + g_lane
        block_row_bot = block_warp_row + g_lane + 8
        # Swz128B chunk-XOR mask for this lane's two target rows.
        swz_top_mask = block_row_top & cutlass.Int32(7)
        swz_bot_mask = block_row_bot & cutlass.Int32(7)
        for n_frag in cutlass.range_constexpr(SV_N_FRAGS):
            o_off = o_base + n_frag * 4
            h_top = fp32_to_fp16(O_acc[o_off + 0] * inv_l_top, O_acc[o_off + 1] * inv_l_top, dtype=io_dtype)
            h_bot = fp32_to_fp16(O_acc[o_off + 2] * inv_l_bot, O_acc[o_off + 3] * inv_l_bot, dtype=io_dtype)
            # chunk_idx for col = n_frag*8 + 2p (2p ∈ {0,2,4,6} < 8) is n_frag.
            swz_chunk_top = cutlass.Int32(n_frag) ^ swz_top_mask
            swz_chunk_bot = cutlass.Int32(n_frag) ^ swz_bot_mask
            smem_top = sO_buf.subview(block_row_top * cutlass.Int32(d_v) + swz_chunk_top * cutlass.Int32(ELEMS_PER_CHUNK) + cutlass.Int32(2 * p_lane))
            smem_bot = sO_buf.subview(block_row_bot * cutlass.Int32(d_v) + swz_chunk_bot * cutlass.Int32(ELEMS_PER_CHUNK) + cutlass.Int32(2 * p_lane))
            # Re-type pointer to Int32 → single st.shared.b32 of the
            # half2 register pair.  alignment=4 (4-byte aligned write).
            Pointer(smem_top.data_ptr(), dtype=cutlass.Int32).store(h_top, alignment=4)
            Pointer(smem_bot.data_ptr(), dtype=cutlass.Int32).store(h_bot, alignment=4)

    # Drain warp-private SMEM writes before the cooperative read.
    nvvm.barrier_cta_sync()

    # ---- STG.128 epilogue ------------------------------------------------
    # All ``threads`` lanes read a 16-B chunk from sO_buf and store it to
    # o_gmem.  Per wave: thread ``tidx`` handles chunk_glob = wave*threads
    # + tidx, mapping to (row = chunk_glob / 16, col_chunk = chunk_glob %
    # 16).  Apply the inverse Swz128B XOR on the SMEM read to undo the
    # write-side permutation; the GMEM write uses the un-swizzled
    # ``col_chunk * ELEMS_PER_CHUNK`` directly.
    chunks_per_row_i32 = cutlass.Int32(CHUNKS_PER_ROW)
    elems_per_chunk_i32 = cutlass.Int32(ELEMS_PER_CHUNK)
    row_stride64 = O_ROW_STRIDE_E.to(cutlass.Int64)
    for wave in cutlass.range_constexpr(n_waves):
        chunk_glob = cutlass.Int32(wave * threads) + tidx
        row = chunk_glob // chunks_per_row_i32
        col_chunk = chunk_glob % chunks_per_row_i32
        swz_chunk = col_chunk ^ (row & cutlass.Int32(7))
        smem_addr = sO_buf.subview(row * cutlass.Int32(d_v) + swz_chunk * elems_per_chunk_i32)
        gmem_addr = o_gmem + row.to(cutlass.Int64) * row_stride64 + (col_chunk * elems_per_chunk_i32).to(cutlass.Int64)
        # SMEM load is unconditional — sO_buf is sized for the full tile_m ×
        # D_V and any OOB lanes wrote dummy data there.  GMEM STG.128 must
        # skip OOB rows so we don't touch GMEM past the user tensor.  Columns
        # are never trimmed: O is always the compile-time d_v wide (the
        # adapter pads V up to it, and padded-V columns compute to exact
        # zero), while d_runtime is the Q/K head dim — trimming on it dropped
        # the real O columns [D, d_v) whenever a graph declared d_qk < d_v.
        vec = Pointer(smem_addr.data_ptr(), dtype=cutlass.Int32).load(alignment=16, count=4)
        if cutlass.const_expr(is_even_mn):
            Pointer(gmem_addr, dtype=cutlass.Int32).store(vec, alignment=16)
        else:
            row_abs_i32 = q_row_base_i32 + row
            if row_abs_i32 < sq_store_bound:
                Pointer(gmem_addr, dtype=cutlass.Int32).store(vec, alignment=16)


# ---------------------------------------------------------------------------
# Host wrapper.
# ---------------------------------------------------------------------------
@cute.jit
def _sdpa_host(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    O: cute.Tensor,
    LSE: Optional[cute.Tensor],
    seq_kv_lens: cute.Tensor,
    seq_len_q: cute.Tensor,
    sinks: cute.Tensor,
    bias: cute.Tensor,
    cu_q: cute.Tensor,
    cu_k: cute.Tensor,
    rope_cs: cute.Tensor,
    tile_m: cutlass.Constexpr[int],
    num_warps: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    is_even_mn: cutlass.Constexpr[bool],
    is_even_k: cutlass.Constexpr[bool],
    mask_flags: cutlass.Constexpr[int],
    swa_window: cutlass.Constexpr[int],
    causal_bottom_right: cutlass.Constexpr[bool],
    has_seq_kv_lens: cutlass.Constexpr[bool],
    has_seq_len_q: cutlass.Constexpr[bool],
    has_sink: cutlass.Constexpr[bool],
    has_bias: cutlass.Constexpr[bool],
    bias_is_fp32: cutlass.Constexpr[bool],
    THD_VARLEN: cutlass.Constexpr[bool],
    has_rope: cutlass.Constexpr[bool],
    sched_policy: cutlass.Constexpr[int],
    sched_l2_bytes: cutlass.Constexpr[int],
    n_kv_tiles: cutlass.Int32,  # runtime — collapses JIT compile time
    softmax_scale_log2: cutlass.Float32,
    sq_runtime: cutlass.Int32,
    skv_runtime: cutlass.Int32,
    d_runtime: cutlass.Int32,
    causal_band_right: cutlass.Int32,
    inv_softmax_scale: cutlass.Float32,
    thd_q_tiles: cutlass.Int32,  # THD: ceil(max_s_q / tile_m) (over-provisioned)
    thd_n_batch: cutlass.Int32,  # THD: logical sequence count B
    stream: cuda.CUstream,
):
    SQ = Q.shape[1]
    H = Q.shape[2]
    B = Q.shape[0]
    threads = num_warps * 32
    q_tiles = (SQ + tile_m - 1) // tile_m
    # Grid shape is policy-dependent:
    #   * THD_VARLEN → over-provisioned 3-D grid (ceil(max_s_q/tile_m), H, B)
    #     with B the LOGICAL sequence count (packed Q.shape[0] == 1); tiles past
    #     a sequence's length early-out in the kernel.  Runtime grid dims.
    #   * SCHED_DEFAULT → 3-D grid (q_tiles, H, B), no decode in kernel.
    #   * SCHED_LPT / SCHED_LPT_L2 → 1-D grid, kernel does the decode.
    if cutlass.const_expr(THD_VARLEN):
        grid = (thd_q_tiles, cutlass.Int32(H), thd_n_batch)
    elif cutlass.const_expr(sched_policy == SCHED_DEFAULT):
        grid = (q_tiles, H, B)
    else:
        grid = (q_tiles * H * B, 1, 1)
    _sdpa_kernel(
        Q,
        K,
        V,
        O,
        LSE,
        seq_kv_lens,
        seq_len_q,
        sinks,
        bias,
        cu_q,
        cu_k,
        rope_cs,
        tile_m,
        num_warps,
        tile_n,
        d_qk,
        d_v,
        io_dtype,
        is_even_mn,
        is_even_k,
        mask_flags,
        swa_window,
        causal_bottom_right,
        has_seq_kv_lens,
        has_seq_len_q,
        has_sink,
        has_bias,
        bias_is_fp32,
        THD_VARLEN,
        has_rope,
        sched_policy,
        sched_l2_bytes,
        n_kv_tiles,
        softmax_scale_log2,
        sq_runtime,
        skv_runtime,
        d_runtime,
        causal_band_right,
        inv_softmax_scale,
    ).launch(
        grid=grid,
        block=(threads, 1, 1),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Per-shape compile cache.
# ---------------------------------------------------------------------------
# ``cute.compile`` is expensive (trace + MLIR + NVVM + PTX → SASS, ~1-2 s on
# A100).  The FEATURE / config axes are module identity — one loaded template
# module per ``TemplateParams`` via ``frost.template_loader`` — so this cache
# covers the remaining SHAPE axes only.  Every key component is PLAN-TIME
# data (AGENTS.md Hard Rule 4): under ``PARAMS.thd_varlen`` the packed token
# totals compile DYNAMIC (``cute.sym_int``) and are never part of the key —
# callers pass ``sq = skv = 0`` there (a stray runtime total must not be
# passed: it would only mint a redundant cache entry for the same artifact).
@lru_cache(maxsize=None)
def compile(  # noqa: A001 — the template contract's entry point (matches the SM100 kernels)
    b: int,
    h: int,
    h_kv: int,
    sq: int,
    skv: int,
    d: int,
    swa_window: int = 0,
    rope_max_s: int = 0,
    n_batch_logical: int = 0,
    lse_stride: Optional[tuple[int, int, int]] = None,
):
    """Compile (or fetch) this template specialization for one shape.

    ``d`` is the ACTUAL Q/K head dim and may be < ``PARAMS.d_qk``: the
    SMEM/reg tile stays ``PARAMS.d_qk`` wide and the missing columns
    zero-fill via cp.async predication.  V/O are always exactly
    ``PARAMS.d_v`` wide.  ``swa_window`` is the left-window width W (keep
    k in [q-W, q]) — plan-time graph data, baked exactly as the old
    ``forward()`` did.  Dense evenness is derived here the same way the old
    entry point derived it: ``is_even_mn = (sq % tile_m == 0) and
    (skv % tile_n == 0)``; ``is_even_k = (d == PARAMS.d_qk)``.

    THD (``PARAMS.thd_varlen``): q/k/v/o are packed ``[1, T, H, D]`` and the
    LSE is packed ``[1, H, T]``; the token extents compile DYNAMIC — one
    ``cute.sym_int`` symbol shared by the Q/O/LSE group and one for K/V — so
    one artifact re-binds any packed totals (issue #604).  Pass ``b = 1``,
    ``sq = skv = 0``; ``n_batch_logical`` (the logical sequence count) sizes
    the ``cu_seqlens`` ABI and IS plan-time.  THD always takes the
    predicated-store path (``is_even_mn = False``) and the over-provisioned
    SCHED_DEFAULT grid, driven by the runtime ``thd_q_tiles``/``thd_n_batch``
    launch arguments.

    ``PARAMS.has_lse = False`` compiles the LSE store out entirely (the LSE
    argument is None-specialized) — no buffer and no dummy at any level.
    """
    p = PARAMS
    if p.thd_varlen and p.has_bias:
        raise ValueError("sm80: bias + THD is not supported (varlen has no single [1,H,SQ,SKV] bias shape)")
    io_dtype = cutlass.BFloat16 if p.io_bf16 else cutlass.Float16
    mask_flags = (MASK_CAUSAL if p.is_causal else MASK_NONE) | (MASK_SWA if p.has_swa else 0)
    sched_l2_bytes = p.sched_l2_mib * 1024 * 1024
    is_even_k = d == p.d_qk
    if p.thd_varlen:
        # One symbol per ragged group: Q/O (and the LSE's T axis) share t_q,
        # K/V share t_kv — a new packed total re-binds the same artifact.
        is_even_mn = False
        t_q = cute.sym_int(divisibility=1)
        t_kv = cute.sym_int(divisibility=1)
        _b, _sq, _skv = 1, t_q, t_kv
    else:
        is_even_mn = (sq % p.tile_m == 0) and (skv % p.tile_n == 0)
        _b, _sq, _skv = b, sq, skv

    # Q and K share the QK head dim (= d, possibly < PARAMS.d_qk when
    # ~is_even_k); V and O follow PARAMS.d_v (DSv3: d_qk != d_v).
    fake_q = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (_b, _sq, h, d),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_k = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (_b, _skv, h_kv, d),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_v = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (_b, _skv, h_kv, p.d_v),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_o = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (_b, _sq, h, p.d_v),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    # LSE: dense [B, H, SQ]; THD packed [1, H, T] (shares the Q/O token
    # symbol, which is exactly what the kernel's LSE.shape[2] read needs).
    if p.has_lse:
        fake_lse = (
            cute.runtime.make_fake_tensor(cutlass.Float32, (_b, h, _sq), lse_stride, assumed_align=4)
            if lse_stride is not None and not p.thd_varlen
            else cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (_b, h, _sq),
                stride_order=(2, 1, 0),
                assumed_align=16,
            )
        )
    else:
        fake_lse = None
    # Per-batch KV / Q lengths [B] int32 (or a 1-elem dummy when unused).
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (b if p.has_seq_kv_lens else 1,),
        stride_order=(0,),
        assumed_align=4,
    )
    fake_seq_len_q = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (b if p.has_seq_q_lens else 1,),
        stride_order=(0,),
        assumed_align=4,
    )
    # Per-Q-head sink logit [H] fp32 (log2 units) (or a 1-elem dummy when unused).
    fake_sinks = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (h if p.has_sink else 1,),
        stride_order=(0,),
        assumed_align=4,
    )
    # Additive bias [1, H, SQ, SKV] in io_dtype or fp32 (or a 1-elem dummy).
    bias_io_dtype = cutlass.Float32 if p.bias_is_fp32 else io_dtype
    fake_bias = cute.runtime.make_fake_compact_tensor(
        bias_io_dtype,
        ((1, h, sq, skv) if p.has_bias else (1,)),
        stride_order=((3, 2, 1, 0) if p.has_bias else (0,)),
        assumed_align=16,
    )
    # THD cumulative seqlens [B_logical + 1] int32 (or 1-elem dummies).
    _cu_len = (n_batch_logical + 1) if p.thd_varlen else 1
    fake_cu_q = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (_cu_len,), stride_order=(0,), assumed_align=4)
    fake_cu_k = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (_cu_len,), stride_order=(0,), assumed_align=4)
    # RoPE (cos, sin) table [max_s, d_qk//2, 2] fp32 (or a 1-elem dummy).
    fake_rope_cs = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        ((rope_max_s, p.d_qk // 2, 2) if p.has_rope else (1,)),
        stride_order=((2, 1, 0) if p.has_rope else (0,)),
        assumed_align=16,
    )
    fake_n_kv_tiles = cutlass.Int32(0)
    fake_scale = cutlass.Float32(0.0)
    fake_sq_rt = cutlass.Int32(0)
    fake_skv_rt = cutlass.Int32(0)
    fake_d_rt = cutlass.Int32(0)
    fake_band_rt = cutlass.Int32(0)
    fake_inv_scale = cutlass.Float32(0.0)
    fake_thd_qt = cutlass.Int32(0)
    fake_thd_nb = cutlass.Int32(0)
    # Stream not used during trace; passed through to launch().  Use a
    # null stream sentinel — cute.compile only inspects type, not value.
    fake_stream = cuda.CUstream(0)
    return cute.compile(
        _sdpa_host,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        fake_seq_kv_lens,
        fake_seq_len_q,
        fake_sinks,
        fake_bias,
        fake_cu_q,
        fake_cu_k,
        fake_rope_cs,
        p.tile_m,
        p.num_warps,
        p.tile_n,
        p.d_qk,
        p.d_v,
        io_dtype,
        is_even_mn,
        is_even_k,
        mask_flags,
        swa_window,
        p.causal_bottom_right,
        p.has_seq_kv_lens,
        p.has_seq_q_lens,
        p.has_sink,
        p.has_bias,
        p.bias_is_fp32,
        p.thd_varlen,
        p.has_rope,
        p.sched_policy,
        sched_l2_bytes,
        fake_n_kv_tiles,
        fake_scale,
        fake_sq_rt,
        fake_skv_rt,
        fake_d_rt,
        fake_band_rt,
        fake_inv_scale,
        fake_thd_qt,
        fake_thd_nb,
        fake_stream,
        options="--enable-tvm-ffi",
    )
