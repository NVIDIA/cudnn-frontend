# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Kimi K3 MLA layers, inference phases (absorbed decode).

K3 is a hybrid: 69 KDA (linear attention) layers + 24 gated-MLA layers.
The MLA layers use NoPE (no rotation is applied) but keep the 64 "rope"
channels allocated, so the absorbed decode record is kv_lora_rank 512 + 64 =
576 wide, V reads the leading 512, and the softmax scale is anchored to the
raw QK width 192 (qk_nope 128 + 64). 96 query heads. Config values from the
official moonshotai/Kimi-K3 config.json.

Generation runs the absorbed shape (576/512 shared record, MQA-style).
Context (prefill) for latent-absorbed MLA runs unabsorbed — that is the
training suite's kimi_k3 config (96 heads, d_qk=192, d_v=128). The context
shapes below are still swept so the coverage gap shows up as blank slots in
the charts rather than silently missing.
"""

import math

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

KIMI_K3_MLA_ABSORBED = ModelPreset(
    name="kimi_k3",
    num_q_heads=96,
    num_kv_heads=1,
    head_dim_qk=576,  # 512 latent + 64 unrotated pe channels
    head_dim_vo=512,
    kind="mla_absorbed",
    sm_scale=1.0 / math.sqrt(192.0),  # raw qk width: qk_nope 128 + pe 64
)

CONFIG = InferenceBenchmarkConfig(
    name="kimi_k3",
    # TP shards of the 96 query heads (kv is the single shared record).
    models=with_tp_shards(KIMI_K3_MLA_ABSORBED, [1, 2, 4, 8]),
    # Absorbed-MLA prefill (576/512) is structurally unabsorbed at context
    # time (training suite's kimi_k3 config); the cases are still swept so
    # the charts show the gap explicitly.
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
    page_size=32,
    backends=["cudnn", "cudnn_oss"],
)
