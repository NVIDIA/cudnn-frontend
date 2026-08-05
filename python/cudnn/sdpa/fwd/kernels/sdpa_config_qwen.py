# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen SDPA flavor config for SM80 — d_qk = d_v = 256, FP16.

Qwen ships d=256 per head.  Same kernel skeleton as llama / dsv3 / gptoss
(``prefill_sdpa_f16_sm80.py``) — the shared Q-in-regs + double-buffered
K + V-aliased-with-Q pipeline handles d=256 cleanly once m_blocks stays
at 1 (tile_m = num_warps * 16).

SMEM at the pinned point: sQ_buf (64 KiB, aliased with sV ring after
prologue Q→reg) + sK_buf (64 KiB, 2-stage) = 128 KiB.  Well under
A100's 164 KiB opt-in.

Register footprint per thread:
  Q_frag  m_blocks*QK_K_CHUNKS*4 = 1*16*4 = 64 i32   (resident regs)
  O acc   m_blocks*SV_N_FRAGS*4  = 1*32*4 = 128 fp32 (resident regs)
  S acc transient                          32 fp32
  P frag transient                         16 i32
  + K/V frag transients + control          ~30
  Total ~ 240-250 regs/thread — under SM80's 256/thread cap.

Knob defaults: ``(tile_m=128, tile_n=64, num_warps=8)``.  m_per_warp = 16
→ m_blocks = 1.  TILE_M must equal num_warps*16 at d=256 (any larger
m_blocks pushes O_acc + Q_frag over the 256/thread cap).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    D_QK: int = 256
    D_V: int = 256
    TILE_M: int = 128
    TILE_N: int = 64
    NUM_WARPS: int = 8


CFG = Cfg()
