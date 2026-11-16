# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V3-style MLA prefill (unabsorbed): per-head K of 192 (128 nope + 64 rope) and V of 128.

This is the (192, 128) head-dim pair FlashInfer's ragged prefill ``auto`` routes to cuDNN, so the FlashInfer legs read the
ragged (``dense``) path. Shards follow the vLLM recipes (TP1 = DEP, TP8 = TP / TEP).

Same attention shape, so these rows also stand for:
- deepseek_v3_prefill (128 heads): DeepSeek-V3, V3.1, R1, V3.2 (its prefill is dense below the top-k length).
- kimi_k2_prefill (64 heads): Kimi-K2, K2.5, K2.6; Ling-2.6-1T, Ling-3.1-flash, Ring-2.6-1T.
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

DEEPSEEK_V3_PREFILL = ModelPreset(name="deepseek_v3_prefill", num_q_heads=128, num_kv_heads=128, head_dim_qk=192, head_dim_vo=128)
KIMI_K2_PREFILL = ModelPreset(name="kimi_k2_prefill", num_q_heads=64, num_kv_heads=64, head_dim_qk=192, head_dim_vo=128)

CONFIG = InferenceBenchmarkConfig(
    name="mla_prefill",
    models=with_tp_shards(DEEPSEEK_V3_PREFILL, [1, 8]) + with_tp_shards(KIMI_K2_PREFILL, [1, 8]),
    context_seqlens=[2048, 8192, 32768],
    context_chunked_shapes=[(512, 65536), (1024, 131072)],
    generation_shapes=[],
    backends=["cudnn", "cudnn_oss", "cudnn_auto", "flashinfer", "flashinfer_fa2", "flashinfer_cudnn"],
    kv_layouts=["dense"],
)
