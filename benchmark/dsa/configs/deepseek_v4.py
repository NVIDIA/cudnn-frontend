# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V4 sparse attention (DSA) benchmark configuration.

V4's sparse core is flat MQA over one shared K=V record of head_dim=512 (RoPE
is applied in place on the trailing channels; the head is not widened to 576
as in V3.2). Per query, the lightning indexer selects the top entries — 512
for V4-Flash, 1024 for V4-Pro — and the softmax runs over the union of those
entries, the last 128 tokens (sliding window) and a per-head sink logit.
Folding the window into the index list gives the kernel a logical K of 640
(Flash) / 1152 (Pro); ``indexer_topk`` marks the indexer-selected prefix whose
LSE the forward kernel also emits (the indexer's training signal).

Heads: 64 (Flash) / 128 (Pro), from the official HF configs. The kernel
gathers K rows of the shared record per query; whether those rows are raw
tokens or CSA compressed entries does not change its work, so the sweep
sizes the KV pool as ``s_kv == s_q``. Every query gathers the full top-k,
i.e. the timed work is the per-token upper bound (no causal-prefix
shortening at the start of the sequence).

Usage:
    python -m benchmark.dsa.runner --config deepseek_v4
    python -m benchmark.dsa.runner --config deepseek_v4 --dry-run
    python -m benchmark.dsa.runner --config deepseek_v4 --filter flash --pass fwd
"""

from ..config_types import DsaBenchmarkConfig, ModelPreset

DSV4_FLASH = ModelPreset(
    name="dsv4_flash",
    num_q_heads=64,
    head_dim_qk=512,
    head_dim_vo=512,
    topk=640,  # 512 indexer-selected + 128 sliding window
    indexer_topk=512,
    has_sink=True,
)

DSV4_PRO = ModelPreset(
    name="dsv4_pro",
    num_q_heads=128,
    head_dim_qk=512,
    head_dim_vo=512,
    topk=1152,  # 1024 indexer-selected + 128 sliding window
    indexer_topk=1024,
    has_sink=True,
)

CONFIG = DsaBenchmarkConfig(
    name="deepseek_v4",
    models=[DSV4_FLASH, DSV4_PRO],
    seqlens=[
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (32768, 32768),
    ],
    backends=["cudnn"],
    data_types=["bfloat16"],
    profile_pass="both",
    deterministic_bwd=[False],
    use_topk_length=True,
    num_iterations=20,
    num_warmup_iterations=5,
    output_dir="results",
)
