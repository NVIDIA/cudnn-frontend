# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V4 attention, inference phases.

V4 replaces V3's MLA with shared-K=V multi-query attention: a single KV head
of head_dim=512 where the same tensor is read as both key and value, partial
RoPE on the trailing 64 channels (carved out of the 512, not added on top),
and a sliding-window branch of 128. Config values from the official
deepseek-ai/DeepSeek-V4-Flash and -Pro configs (head_dim=512,
qk_rope_head_dim=64, num_key_value_heads=1).

The presets benchmark the dense shared-K=V core UNWINDOWED: the production
W=128 sliding window reads at most 128 cached tokens, which is not a
KV-cache-bound problem, so the unwindowed sweep is the upper-bound stress on
the shared record (the CSA/HCA compressed-pool + lightning-indexer machinery
is likewise out of scope for a single-kernel benchmark). Modeled as
kind="mla_absorbed" so KV bytes are counted once for the shared record.
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

DSV4_FLASH = ModelPreset(
    name="dsv4_flash",
    num_q_heads=64,
    num_kv_heads=1,
    head_dim_qk=512,
    head_dim_vo=512,
    kind="mla_absorbed",  # shared K=V record
)

DSV4_PRO = ModelPreset(
    name="dsv4_pro",
    num_q_heads=128,
    num_kv_heads=1,
    head_dim_qk=512,
    head_dim_vo=512,
    kind="mla_absorbed",
)

CONFIG = InferenceBenchmarkConfig(
    name="deepseek_v4",
    models=with_tp_shards(DSV4_FLASH, [1, 2, 4, 8]) + with_tp_shards(DSV4_PRO, [1, 2, 4, 8]),
    # Context: unwindowed core at moderate lengths (see module docstring).
    context_seqlens=[2048, 8192],
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
    # fp8 shared-KV record: the serving config for the d=512 dense core.
    kv_cache_dtypes=["bfloat16", "fp8_e4m3"],
    backends=["cudnn", "cudnn_oss"],
)
