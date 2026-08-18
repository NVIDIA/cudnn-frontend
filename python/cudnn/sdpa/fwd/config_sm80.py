# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile-time flavor configs for the SM80 SDPA forward kernels.

One frozen dataclass per kernel flavor (the model families the head-dim
envelopes are named after), all served by the same kernel skeleton
(``kernels/prefill_f16_sm80.py``; qwen d=256 by ``prefill_d256_f16_sm80.py``).
Knob provenance per flavor:

- **gptoss** (d_qk = d_v = 64): picked from the A100-PCIE-40GB perf sweep. At
  d=64 the SV mma has 8 n_frags instead of 16, so M_BLOCKS=2 per warp fits the
  register budget that spills at d=128. Best point ``(tile_m=128, tile_n=64,
  num_warps=4)`` matches FA to within ~5-7 % on H=32 H_kv=2 GQA shapes at
  SQ ∈ {2K, 4K, 8K}. For causal the sweep additionally wants ``--sched lpt_l2
  --sched-l2-mib 16`` on top — driver-side override, not pinned here.
- **llama** (d_qk = d_v = 128): swept on A100-PCIE-40GB; ``(tile_m=128,
  tile_n=64, num_warps=8)`` matches Phase-3 baseline of ~97 % FA-2.8.3 at
  SQ=8K — keep unless a perf-node sweep finds a better point.
- **dsv3** (d_qk = 192, d_v = 128): first flavor with d_qk != d_v; the shared
  skeleton's K vs V row-stride plumbing handles the asymmetry without a fork.
  Knobs are an initial guess — not swept yet; the perf-node sweep will pin it.
- **qwen** (d_qk = d_v = 256): SMEM at the pinned point: sQ_buf (64 KiB,
  aliased with the sV ring after the prologue Q→reg) + sK_buf (64 KiB,
  2-stage) = 128 KiB, under A100's 164 KiB opt-in. ~240-250 regs/thread —
  under SM80's 256/thread cap, but only at m_blocks = 1: TILE_M must equal
  num_warps*16 at d=256 (any larger m_blocks pushes O_acc + Q_frag over).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    D_QK: int
    D_V: int
    TILE_M: int
    TILE_N: int
    NUM_WARPS: int


GPTOSS_CFG = Cfg(D_QK=64, D_V=64, TILE_M=128, TILE_N=64, NUM_WARPS=4)
LLAMA_CFG = Cfg(D_QK=128, D_V=128, TILE_M=128, TILE_N=64, NUM_WARPS=8)
DSV3_CFG = Cfg(D_QK=192, D_V=128, TILE_M=128, TILE_N=64, NUM_WARPS=8)
QWEN_CFG = Cfg(D_QK=256, D_V=256, TILE_M=128, TILE_N=64, NUM_WARPS=8)
