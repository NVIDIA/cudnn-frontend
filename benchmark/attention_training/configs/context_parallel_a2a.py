# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Context-parallel (A2A) THD SDPA Benchmark Configuration

The per-rank attention of Transformer Engine's CP4 all-to-all context parallelism on a 64 x 8192-token causal THD
batch (524288 tokens, 32 query / 8 KV heads): after the all-to-all each rank holds every full sequence and a quarter of
the heads (8 query / 2 KV). Equal-length sequences (TE's uniform-THD path) and ragged ones (half to full length).

Usage:
    python -m benchmark.attention_training.runner --config context_parallel_a2a
    python -m benchmark.attention_training.runner --config context_parallel_a2a --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

CP4_A2A_D128 = ModelPreset(name="cp4_a2a_d128", num_q_heads=8, num_kv_heads=2, head_dim=128)
CP4_A2A_D256 = ModelPreset(name="cp4_a2a_d256", num_q_heads=8, num_kv_heads=2, head_dim=256)

CONFIG = BenchmarkConfig(
    name="context_parallel_a2a",
    models=[CP4_A2A_D128, CP4_A2A_D256],
    seqlens=[(8192, 8192)],
    backends=["cudnn", "cudnn_oss", "cudnn_auto"],
    data_types=["bfloat16"],
    attn_masks=["top_left"],
    profile_pass="both",
    layout="thd",
    thd_min_fills=[1.0, 0.5],
    batch_size=64,
    num_iterations=10,
    output_dir="results",
)
