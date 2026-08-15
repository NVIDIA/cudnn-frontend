# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-VL Vision Encoder (ViT) SDPA Benchmark Configuration

Benchmarks the self-attention of the Qwen3-VL vision encoder as exercised
by image/video inference. ViT self-attention is bidirectional over the
patchified image tokens, so only ``no_mask`` is benchmarked, forward pass
only (inference workload).

Architecture (vision tower):
    - hidden dim           = 1152
    - num_attention_heads  = 16   (MHA, no GQA)
    - attention head_dim   = 72   (16 x 72 = 1152)
    - benchmarked head_dim = 80   (72 zero-padded to the next 16-byte
      multiple: fp8 kernels require 16B-aligned head dims, and production
      integrations run this padded contract. Reported TFLOPS count d=80.)

Sequence lengths are per-image patch-grid token counts taken from a
production inference trace (native-resolution images; token count =
grid_h x grid_w). Single-image (batch 1) forwards dominate that trace,
and these six lengths span its per-forward FLOPs distribution from the
10th to the 99th percentile:

    ( 8836,  94 x  94 grid)
    (15376, 124 x 124 grid)   <- most frequent single-image forward
    (24336, 156 x 156 grid)
    (35344, 188 x 188 grid)   <- FLOPs-median forward
    (47376, non-square grid, e.g. 168 x 282)
    (62500, 250 x 250 grid)

Usage:
    python -m benchmark.sdpa_benchmark_training.runner --config qwen3vl_vit
    python -m benchmark.sdpa_benchmark_training.runner --config qwen3vl_vit --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

QWEN3VL_VIT = ModelPreset(
    name="qwen3vl_vit",
    num_q_heads=16,
    num_kv_heads=16,
    head_dim=80,
)

CONFIG = BenchmarkConfig(
    name="qwen3vl_vit",
    models=[QWEN3VL_VIT],
    seqlens=[
        (8836, 8836),  # 94x94 patch grid
        (15376, 15376),  # 124x124 (most frequent)
        (24336, 24336),  # 156x156
        (35344, 35344),  # 188x188 (FLOPs median)
        (47376, 47376),  # non-square grid (168x282)
        (62500, 62500),  # 250x250
    ],
    backends=["cudnn", "cudnn_oss", "flash_attention_4"],
    data_types=["bfloat16", "fp8"],
    attn_masks=["no_mask"],
    profile_pass="fwd",
    batch_size=1,
    num_iterations=10,
    num_warmup_iterations=5,
)
