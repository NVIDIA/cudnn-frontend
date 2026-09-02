# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Llama 3.1 SDPA Benchmark Configuration

Benchmarks Llama 3.1 405B-style GQA attention with both causal and non-causal masks.
Includes forward and backward pass benchmarking with deterministic mode options.

Usage:
    python -m benchmark.attention_training.runner --config llama
    python -m benchmark.attention_training.runner --config llama --dry-run
"""

import torch

from ..config_types import ModelPreset, BenchmarkConfig

# flash_attention (FA2) is benchmarked only on Ampere: FA4's backward asserts
# SM90+ (flash_attn/cute/interface.py, _flash_attn_bwd) while its forward
# supports SM80, so without FA2 the Ampere dashboard has no flash-attention
# backward reference. On SM90+ FA4 is the reference and FA2 would only add
# runtime. The 26.07 base container ships flash-attn 2.x natively.
_IS_AMPERE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 8

LLAMA3_1 = ModelPreset(
    name="llama3.1",
    num_q_heads=64,
    num_kv_heads=8,
    head_dim=128,
)

CONFIG = BenchmarkConfig(
    name="llama3.1",
    models=[LLAMA3_1],
    seqlens=[
        (32768, 32768),
        (16384, 16384),
        (8192, 8192),
        (4096, 4096),
        (2048, 2048),
    ],
    backends=["cudnn", "cudnn_oss", "flash_attention_4"]
    + (["flash_attention"] if _IS_AMPERE else []),
    data_types=["bfloat16", "fp8", "mxfp8"],
    attn_masks=["top_left", "no_mask"],  # Both causal and non-causal
    profile_pass="both",  # Forward and backward
    deterministic_bwd=[False, True],
    batch_size=2,
    num_iterations=10,
    output_dir="results",
)
