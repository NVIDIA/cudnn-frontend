# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V4.1 sparse attention (DSA) benchmark configuration.

V4.1-Flash keeps V4's sparse core — flat MQA over one shared K=V record of
head_dim=512 with RoPE in place on the trailing 64 channels, a 128-token
sliding window and a per-head sink — and changes what the indexer selects
from: the pool is 2x-compressed (layers 2-19) or uncompressed raw tokens
(layers 20-39), the KV record and the index picks are produced by a few
source layers and reused by the layers after them, and a candidate set of
2048 blocks of 8 tokens narrows the later layers' top-512 selection
(``compress_ratios``, ``kv_source_layer_ids``, ``index_source_layer_ids``,
``candidate_topk_blocks`` in the official HF config). None of that changes
the per-query gather the kernel performs — 512 indexer-selected rows plus
the 128-token window — so the preset is V4-Flash's: logical K = 640 with
``indexer_topk = 512``, 64 heads. Only a Flash checkpoint is published for
V4.1; add a Pro preset when one appears.

Usage:
    python -m benchmark.dsa.runner --config deepseek_v41
    python -m benchmark.dsa.runner --config deepseek_v41 --dry-run
    python -m benchmark.dsa.runner --config deepseek_v41 --pass bwd
"""

from ..config_types import DsaBenchmarkConfig, ModelPreset

DSV41_FLASH = ModelPreset(
    name="dsv41_flash",
    num_q_heads=64,
    head_dim_qk=512,
    head_dim_vo=512,
    topk=640,  # 512 indexer-selected + 128 sliding window
    indexer_topk=512,
    has_sink=True,
)

CONFIG = DsaBenchmarkConfig(
    name="deepseek_v41",
    models=[DSV41_FLASH],
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
