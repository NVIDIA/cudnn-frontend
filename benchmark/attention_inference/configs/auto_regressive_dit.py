# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Autoregressive video DiT cross-attention.

Chunked autoregression: each step's queries (985..8192 new tokens) attend
bidirectionally to a long cached KV (62208 tokens). That is chunked-prefill-
shaped work, so it lives in the context phase's chunked kind here (the config
is bidirectional: context_causal=False). Full prefill covers the self-
attention over the whole clip, and generation sweeps the standard MTP widths
against the cached clip for completeness. The training suite carries its own
forward-only sweep of the same model (`attention_training`).
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset

AR_DIT = ModelPreset(
    name="ar_dit",
    num_q_heads=9,
    num_kv_heads=9,
    head_dim=128,
)

CONFIG = InferenceBenchmarkConfig(
    name="auto_regressive_dit",
    # No TP shards: 9 heads don't head-shard, and video DiT deployments use
    # sequence/context parallelism — attention runs whole-model per device.
    models=[AR_DIT],
    context_seqlens=[8192, 62208],
    context_chunked_shapes=[
        (985, 62208),
        (2048, 62208),
        (4096, 62208),
        (8192, 62208),
    ],
    # q_tokens = 1 + MTP, for MTP 0..3, against the cached clip.
    generation_shapes=[(q, 62208) for q in (1, 2, 3, 4)],
    generation_batch_sizes=[1],
    context_causal=False,
    backends=["cudnn", "cudnn_oss"],
)
