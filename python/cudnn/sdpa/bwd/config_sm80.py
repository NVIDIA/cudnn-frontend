# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile-time flavor configs for the SM80 SDPA backward kernel.

Single source of truth for the shared 2-sub-group backward pipeline
(``kernels/bprop_f16_sm80.py``); all shape-dependent sizes are derived in the
kernel from these fields — do NOT inline ``128`` / ``64`` literals there.
The pipeline shape the knobs describe:

  * 1 CTA owns one **KV-tile** (TILE_KV rows of K/V) for a (batch, head);
    the CTA loops over Q-tiles (TILE_Q rows of Q/dO/O).
  * Two sub-groups of ``WARPS_PER_SG`` warps each (256 threads at the default
    4+4 split): sg0 runs ``S = K·Qᵀ`` + softmax + ``dV = P·dO``; sg1 runs
    ``dP = V·dOᵀ`` + dSoftmax + ``dK = dS·Q``. Both groups split ``dQ`` by
    d-col — the swizzle is handled d-agnostically via
    ``load_b_smem_x4(col_base=...)``.

Flavors (d is read from the tensor shapes; the kernel runs both unchanged):

- **llama** (d_qk = d_v = 128): TILE_Q=64.
- **gptoss** (d_qk = d_v = 64): at d=64 the per-thread register / SMEM
  footprint is roughly half of llama's, which buys a **128-row Q-tile**:
  TILE_Q=128 halves the number of Q-iters, so the fixed per-iter overhead
  (5 CTA barriers + Q/dO reload + do_dot/LSE reads) is amortized over 2× the
  MMA work — the dominant cost at d=64 where each tile's BMM flops halve.
  The dQ MMA then runs DQ_M_BLOCKS = TILE_Q // (WARPS_PER_SG*16) = 2 m-blocks
  per warp (the same M_BLOCKS trick the forward gptoss kernel uses at d=64).
  SMEM at TILE_Q=128 is ~144 KiB < the A100 163 KiB dynamic-SMEM cap. SQ must
  be a multiple of TILE_Q=128; SKV a multiple of TILE_KV=64.
  NOTE: the shipped gptoss-sm80 (d=64) BPROP kernel is
  ``kernels/bprop_d64_f16_sm80.py``, which uses a FIXED tile shape
  (m_block=64, n_block=128) and does not read this config. GPTOSS_CFG
  configures the shared path when run at d=64.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    D_QK: int  # head dim for Q/K (= matmul K-dim of BMM1)
    D_V: int  # head dim for V/dO/O (= N of dV, dP K-dim of BMM2)
    TILE_KV: int  # KV rows owned per CTA (M of dV/dK; K-reduce of dQ)
    TILE_Q: int  # Q rows per inner iter (N of S/dP; M of dQ)
    WARPS_PER_SG: int  # warps per sub-group → 2 sub-groups = 8 warps = 256 thr


LLAMA_CFG = Cfg(D_QK=128, D_V=128, TILE_KV=64, TILE_Q=64, WARPS_PER_SG=4)
GPTOSS_CFG = Cfg(D_QK=64, D_V=64, TILE_KV=64, TILE_Q=128, WARPS_PER_SG=4)
