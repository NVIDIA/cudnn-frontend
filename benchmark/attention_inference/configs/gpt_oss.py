# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS attention, inference phases.

The model interleaves sliding-window (W=128) and full-attention layers, and
every layer carries per-head attention sinks; both layer types are modeled,
swept across tensor-parallel shards (TP 1/2/4/8 -> 64/8, 32/4, 16/2, 8/1
heads). Serving deployments hold the KV cache in fp8.
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

GPT_OSS_SWA = ModelPreset(
    name="gpt_oss_swa",
    num_q_heads=64,
    num_kv_heads=8,
    head_dim=64,
    sliding_window_size=128,
    has_sink=True,
)

GPT_OSS_FULL = ModelPreset(
    name="gpt_oss_full",
    num_q_heads=64,
    num_kv_heads=8,
    head_dim=64,
    has_sink=True,
)

CONFIG = InferenceBenchmarkConfig(
    name="gpt_oss",
    models=with_tp_shards(GPT_OSS_SWA, [1, 2, 4, 8]) + with_tp_shards(GPT_OSS_FULL, [1, 2, 4, 8]),
    context_seqlens=[2048, 8192, 32768],
    context_chunked_shapes=[
        (512, 65536),
        (512, 131072),
        (1024, 65536),
        (1024, 131072),
    ],
    # q_tokens = 1 + MTP, for MTP 0..3.
    generation_shapes=[(q, 131072) for q in (1, 2, 3, 4)],
    # b1 = latency anchor (single sequence; exercises split-KV fill), b128 =
    # bandwidth plateau. Batches between are a smooth occupancy ramp and kv
    # lengths >=32k measure identically at the plateau, so one long cache
    # (128k) covers the throughput story (B300-measured).
    # NB the W=128 sliding window caps effective KV reads, so the batch axis
    # is the occupancy axis here and the b1 -> b128 spread is the story.
    generation_batch_sizes=[1, 128],
    kv_cache_dtypes=["bfloat16", "fp8_e4m3"],
    page_size=16,
    backends=["cudnn", "cudnn_oss"],
)
