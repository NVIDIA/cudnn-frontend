# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-VL Vision Encoder (ViT) self-attention — inference.

The vision tower's self-attention is bidirectional over the patchified image
tokens and runs forward-only: pure context-phase (full prefill) work, so this
config sweeps ONLY the context phase — an encoder has no KV cache to decode
against, so there is no generation phase and no kv-cache dtype axis.

Architecture (vision tower): 16 MHA heads, head_dim 72 zero-padded to 80
(fp8 kernels require 16-byte-aligned head dims; production integrations run
the padded contract, and reported TFLOPS count d=80). Sequence lengths are
per-image patch-grid token counts from a production inference trace
(94x94 .. 250x250 grids), spanning that trace's per-forward FLOPs
distribution from the 10th to the 99th percentile. The tower is not
head-shardable in deployments (whole-model per device), so no TP sweep.

Usage:
    python -m benchmark.attention_inference.runner --config qwen3vl_vit
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset

QWEN3VL_VIT = ModelPreset(
    name="qwen3vl_vit",
    num_q_heads=16,
    num_kv_heads=16,
    head_dim=80,
)

CONFIG = InferenceBenchmarkConfig(
    name="qwen3vl_vit",
    models=[QWEN3VL_VIT],
    context_seqlens=[
        8836,  # 94x94 patch grid
        15376,  # 124x124 (most frequent single-image forward)
        24336,  # 156x156
        35344,  # 188x188 (FLOPs-median forward)
        47376,  # non-square grid (e.g. 168x282)
        62500,  # 250x250
    ],
    context_chunked_shapes=[],  # encoder: no chunked prefill against a cache
    generation_shapes=[],  # encoder: no decode phase
    context_causal=False,  # bidirectional ViT self-attention
    # bf16 only: this suite expresses fp8 solely as the generation-phase
    # kv-cache axis (the fp8 attention graph), and an encoder has no
    # generation phase — the context path has no fp8 route today. The training
    # suite's fp8 forward numbers for this model are dropped by the move;
    # restoring them here needs fp8-context support in
    # benchmark_single_attention first.
    data_types=["bfloat16"],
    backends=["cudnn", "cudnn_oss"],
)
