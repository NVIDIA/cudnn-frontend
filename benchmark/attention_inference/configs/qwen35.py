# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen 3.5 GQA attention (head_dim=256), inference phases.

Swept across tensor-parallel shards (TP 1/2/4/8 -> 32/2, 16/1, 8/1, 4/1 heads).
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

QWEN3_5 = ModelPreset(
    name="qwen3.5",
    num_q_heads=32,
    num_kv_heads=2,
    head_dim=256,
)

CONFIG = InferenceBenchmarkConfig(
    name="qwen35",
    models=with_tp_shards(QWEN3_5, [1, 2, 4, 8]),
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
    generation_batch_sizes=[1, 128],
    kv_cache_dtypes=["bfloat16", "fp8_e4m3"],
    page_size=16,
    backends=["cudnn", "cudnn_oss"],
)
