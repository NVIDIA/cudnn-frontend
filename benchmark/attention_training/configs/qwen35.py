# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen 3.5 SDPA Benchmark Configuration

Benchmarks Qwen 3.5-style GQA attention with causal (top_left) mask only.
32 Q heads and 2 KV heads (16:1 GQA) with head_dim 256.
bfloat16 and mxfp8; the mxfp8 backward at head_dim=256 is served only by the
FROST engine (``cudnn_oss`` backend, sdpa_bwd_sm100_mxfp8), the native cuDNN
backend has no plan for it.

Usage:
    python -m benchmark.attention_training.runner --config qwen35
    python -m benchmark.attention_training.runner --config qwen35 --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig, fa2_on_ampere

QWEN35 = ModelPreset(
    name="qwen35",
    num_q_heads=32,
    num_kv_heads=2,
    head_dim=256,
)

CONFIG = BenchmarkConfig(
    name="qwen35",
    models=[QWEN35],
    seqlens=[
        (32768, 32768),
        (16384, 16384),
        (8192, 8192),
        (4096, 4096),
        (2048, 2048),
    ],
    backends=["cudnn", "cudnn_oss", "flash_attention_4"] + fa2_on_ampere(),
    # Blackwell at head_dim=256: fa4's sm100 forward kernel asserts on tmem
    # exhaustion regardless of batch (rows kept, they fail, to document the
    # limitation). mxfp8 backward has no native cuDNN plan at this head dim;
    # the cudnn_oss backend pins the FROST engine (sdpa_bwd_sm100_mxfp8).
    data_types=["bfloat16", "mxfp8"],
    attn_masks=["top_left"],  # Causal only
    profile_pass="both",
    deterministic_bwd=[False, True],
    batch_size=1,
    num_iterations=10,
    output_dir="results",
)
