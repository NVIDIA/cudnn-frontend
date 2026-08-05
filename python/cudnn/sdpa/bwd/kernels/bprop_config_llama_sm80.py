# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 (Ampere/A100) BPROP config — Llama d_qk=d_v=128, FP16/BF16.

Single source of truth for the SM80 SDPA backward kernel
(`bprop_sdpa_f16_sm80.py`).  Mirrors the forward adapter's llama knob defaults
convention (frozen dataclass + module singleton `CFG`), but the knobs
describe the *backward* pipeline shape:

  * 1 CTA owns one **KV-tile** (TILE_KV rows of K/V) for a (batch, head);
    the CTA loops over Q-tiles (TILE_Q rows of Q/dO/O).
  * Two sub-groups of `WARPS_PER_SG` warps each (256 threads total at the
    default 4+4 split): sg0 runs `S = K·Qᵀ` + softmax + `dV = P·dO`; sg1
    runs `dP = V·dOᵀ` + dSoftmax + `dK = dS·Q`.  Both groups split `dQ`.

All shape-dependent sizes are derived in the kernel from these fields —
do NOT inline `128` / `64` literals there.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    D_QK: int = 128  # head dim for Q/K (= matmul K-dim of BMM1)
    D_V: int = 128  # head dim for V/dO/O (= N of dV, dP K-dim of BMM2)
    TILE_KV: int = 64  # KV rows owned per CTA (M of dV/dK; K-reduce of dQ)
    TILE_Q: int = 64  # Q rows per inner iter (N of S/dP; M of dQ; K-reduce of dV/dK)
    WARPS_PER_SG: int = 4  # warps per sub-group → 2 sub-groups = 8 warps = 256 thr


CFG = Cfg()
