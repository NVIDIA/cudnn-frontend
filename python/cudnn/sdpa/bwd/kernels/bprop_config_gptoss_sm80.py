# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 (Ampere/A100) BPROP config — GPT-OSS d_qk=d_v=64, FP16/BF16.

Companion to ``bprop_config_llama_sm80.py``; same frozen-dataclass + module
singleton ``CFG`` convention and the SAME 2-sub-group backward pipeline shape
(``bprop_f16_sm80.py`` runs both flavors unchanged — d is read from the
tensor shapes).  GPT-OSS only differs by head dim (64 vs llama's 128):

  * 1 CTA owns one **KV-tile** (TILE_KV rows of K/V) for a (batch, head);
    the CTA loops over Q-tiles (TILE_Q rows of Q/dO/O).
  * Two sub-groups of ``WARPS_PER_SG`` warps each (256 threads at the
    default 4+4 split): sg0 runs ``S = K·Qᵀ`` + softmax + ``dV = P·dO``; sg1
    runs ``dP = V·dOᵀ`` + dSoftmax + ``dK = dS·Q``.  Both groups split ``dQ``
    by d-col (DQ_N = d_qk//2 = 32 at d=64) — the swizzle is handled
    d-agnostically via ``load_b_smem_x4(col_base=...)``.

At d=64 the per-thread register / SMEM footprint is roughly half of llama's,
which buys a **128-row Q-tile** (vs llama's 64): TILE_Q=128 halves the number
of Q-iters, so the fixed per-iter overhead (5 CTA barriers + Q/dO reload +
do_dot/LSE reads) is amortized over 2× the MMA work — the dominant cost at
d=64 where each tile's BMM flops halve.  The dQ MMA (M = TILE_Q) then runs
DQ_M_BLOCKS = TILE_Q // (WARPS_PER_SG*16) = 2 m-blocks per warp (the same
M_BLOCKS trick the forward gptoss SM80 kernel uses at d=64).  TILE_KV stays 64
(= WARPS_PER_SG*16, one m-block for BMM1 / dV / dK).  SMEM at TILE_Q=128 is
~144 KiB < the A100 163 KiB dynamic-SMEM cap.

  SQ must be a multiple of TILE_Q=128 (all of 2K/4K/8K qualify); SKV a
  multiple of TILE_KV=64.

NOTE: the shipped gptoss-sm80 (d=64) BPROP kernel is ``bprop_d64_f16_sm80.py``,
which uses a FIXED tile shape (m_block=64, n_block=128) and does not read this
config.  The fields below (TILE_KV=64, TILE_Q=128) configure the shared
2-sub-group path (``bprop_f16_sm80.py``) when run at d=64.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    D_QK: int = 64  # head dim for Q/K (= matmul K-dim of BMM1)
    D_V: int = 64  # head dim for V/dO/O (= N of dV, dP K-dim of BMM2)
    TILE_KV: int = 64  # KV rows owned per CTA (M of dV/dK; K-reduce of dQ)
    TILE_Q: int = 128  # Q rows per inner iter (N of S/dP; M of dQ → 2 m-blocks)
    WARPS_PER_SG: int = 4  # warps per sub-group → 2 sub-groups = 8 warps = 256 thr


CFG = Cfg()
