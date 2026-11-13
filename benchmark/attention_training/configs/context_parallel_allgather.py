# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Context-parallel (AllGather) THD SDPA Benchmark Configuration

The per-rank attention calls of Transformer Engine's load-balanced AllGather context parallelism on a 64 x 8192-token
causal THD batch (32 query / 8 KV heads, d256): each rank owns two Q chunks of 8192 / (2 * CP) tokens per sequence and
attends each to the gathered KV prefix that ends at the chunk, bottom-right aligned. Q 1024 is CP4, Q 512 is CP8; KV
8192 is the last chunk (the longest call) and 4096 a middle one. Equal-length and ragged (half to full) sequences.

Usage:
    python -m benchmark.attention_training.runner --config context_parallel_allgather
    python -m benchmark.attention_training.runner --config context_parallel_allgather --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

CP_ALLGATHER_D256 = ModelPreset(name="cp_allgather_d256", num_q_heads=32, num_kv_heads=8, head_dim=256)

CONFIG = BenchmarkConfig(
    name="context_parallel_allgather",
    models=[CP_ALLGATHER_D256],
    seqlens=[
        (1024, 8192),
        (1024, 4096),
        (512, 8192),
        (512, 4096),
    ],
    backends=["cudnn", "cudnn_oss", "cudnn_auto"],
    data_types=["bfloat16"],
    attn_masks=["bottom_right"],
    profile_pass="both",
    layout="thd",
    thd_min_fills=[1.0, 0.5],
    batch_size=64,
    num_iterations=20,
    output_dir="results",
)
