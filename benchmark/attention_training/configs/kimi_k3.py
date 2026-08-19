# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Kimi K3 MLA SDPA Benchmark Configuration (training, unabsorbed).

K3 is a hybrid of 69 KDA (linear attention) layers and 24 gated-MLA layers.
The MLA layers train unabsorbed with 96 heads, d_qk = 192 (qk_nope 128 + 64
unrotated "pe" channels — K3's MLA is NoPE, the channels stay allocated) and
d_v = 128. Values from the official moonshotai/Kimi-K3 config.json
(kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
num_attention_heads=96). Versus Kimi K2.6 this bumps heads 64 -> 96.

The absorbed decode shape (576/512 shared record) is covered by
benchmark/attention_inference's kimi_k3 config.

Usage:
    python -m benchmark.attention_training.runner --config kimi_k3
    python -m benchmark.attention_training.runner --config kimi_k3 --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

KIMI_K3 = ModelPreset(
    name="kimi_k3",
    num_q_heads=96,
    num_kv_heads=96,
    head_dim_qk=192,
    head_dim_vo=128,
)

CONFIG = BenchmarkConfig(
    name="kimi_k3",
    models=[KIMI_K3],
    seqlens=[
        (32768, 32768),
        (16384, 16384),
        (8192, 8192),
        (4096, 4096),
        (2048, 2048),
    ],
    backends=["cudnn", "flash_attention_4"],
    data_types=["bfloat16", "fp8", "mxfp8"],
    attn_masks=["top_left", "no_mask"],
    profile_pass="both",
    deterministic_bwd=[False, True],
    batch_size=2,
    num_iterations=10,
    output_dir="results",
)
