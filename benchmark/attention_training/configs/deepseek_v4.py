# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V4 SDPA Benchmark Configuration (training, dense core).

V4 replaces V3's MLA with shared-K=V multi-query attention: a single KV head
of head_dim=512 (the same tensor is read as both key and value), partial RoPE
on the trailing 64 channels carved out of the 512, and a sliding-window
branch of 128 tokens. Values from the official deepseek-ai/DeepSeek-V4-Flash
and -Pro configs (head_dim=512, qk_rope_head_dim=64, num_key_value_heads=1).

This config benchmarks the dense trainable core (the sliding-window MQA every
layer runs); the CSA/HCA compressed pools and lightning indexer are separate
kernels (see benchmark/csa). The harness allocates separate K and V tensors,
so KV read traffic is counted twice versus the served shared record — shapes
and FLOPs are identical.

d=512 heads also make this the support-surface probe for wide-head attention:
unsupported (backend, pass) combos are recorded in the CSV rather than hidden.

Usage:
    python -m benchmark.attention_training.runner --config deepseek_v4
    python -m benchmark.attention_training.runner --config deepseek_v4 --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

DSV4_FLASH = ModelPreset(
    name="dsv4_flash",
    num_q_heads=64,
    num_kv_heads=1,
    head_dim=512,
)

DSV4_PRO = ModelPreset(
    name="dsv4_pro",
    num_q_heads=128,
    num_kv_heads=1,
    head_dim=512,
)

CONFIG = BenchmarkConfig(
    name="deepseek_v4",
    models=[DSV4_FLASH, DSV4_PRO],
    seqlens=[
        (16384, 16384),
        (8192, 8192),
        (4096, 4096),
        (2048, 2048),
    ],
    backends=["cudnn", "flash_attention_4"],
    data_types=["bfloat16", "fp8", "mxfp8"],
    attn_masks=["top_left"],
    profile_pass="both",
    deterministic_bwd=[False],
    batch_size=2,
    num_iterations=10,
    sliding_window_size=128,
    output_dir="results",
)
