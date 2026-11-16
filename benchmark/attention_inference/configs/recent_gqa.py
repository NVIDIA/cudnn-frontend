# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GQA shapes of recent MoE / dense releases that the other configs do not cover: query-to-KV group sizes that are not a power
of two (6, 9, 10), sliding windows of 512 to 2048 at head_dim 64 to 192, and MTP up to 15 draft tokens.

Shapes from the HF configs (num_attention_heads, num_key_value_heads or num_attention_groups, per-layer head counts,
sliding_window, layer_types). Shards follow the vLLM recipes; TP1 is also the DEP deployment of the MoE models.

Same attention shape, so these rows also stand for:
- minimax_m2 (48/8): MiniMax-M2, M2.1, M2.5; Jais-3-375B-A23B, K2-Horizon-375B-A23B; Laguna full-attention layers.
- laguna_swa (72/8, W=512): Laguna-S-2.1, Laguna-XS-2.1, Laguna-XS.2.
- ripple (80/8, W=1024 on 4 of 5 layers): Ripple-65B-Preview; the full preset also Seed-OSS-36B.
- step37_swa (64/8, W=512): Step-3.5-Flash, Step-3.7-Flash (their full-attention layers are the llama config's 64/8).
- step5 (64/4, head_dim 192, W=512 on 3 of 4 layers): Step-5-Preview.
- muse_spark (128/16, head_dim 64, W=2048): Muse-Spark-Open-816B-A42B.
- qwen36_27b (24/4, head_dim 256; the full-attention layers of a linear-attention hybrid): Qwen3.5-27B, Qwen3.6-27B,
  Qwen3.8-27B, JEV-27B-VL.
- dense_8b (32/8): Llama-3.1-8B, Jais-3-3.7B / 7B / MoVA-36B-A4B, K2-Horizon-3.7B / 7B, and others.
- qwen3_14b (40/8): Qwen3-14B, QwQ-32B, Qwen2.5-32B, and Llama-4-Scout's global layers.
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

MINIMAX_M2 = ModelPreset(name="minimax_m2", num_q_heads=48, num_kv_heads=8, head_dim=128)
LAGUNA_SWA = ModelPreset(name="laguna_swa", num_q_heads=72, num_kv_heads=8, head_dim=128, sliding_window_size=512)
RIPPLE_FULL = ModelPreset(name="ripple_full", num_q_heads=80, num_kv_heads=8, head_dim=128)
RIPPLE_SWA = ModelPreset(name="ripple_swa", num_q_heads=80, num_kv_heads=8, head_dim=128, sliding_window_size=1024)
STEP37_SWA = ModelPreset(name="step37_swa", num_q_heads=64, num_kv_heads=8, head_dim=128, sliding_window_size=512)
STEP5_FULL = ModelPreset(name="step5_full", num_q_heads=64, num_kv_heads=4, head_dim=192)
STEP5_SWA = ModelPreset(name="step5_swa", num_q_heads=64, num_kv_heads=4, head_dim=192, sliding_window_size=512)
MUSE_SPARK_FULL = ModelPreset(name="muse_spark_full", num_q_heads=128, num_kv_heads=16, head_dim=64)
MUSE_SPARK_SWA = ModelPreset(name="muse_spark_swa", num_q_heads=128, num_kv_heads=16, head_dim=64, sliding_window_size=2048)
QWEN36_27B = ModelPreset(name="qwen36_27b", num_q_heads=24, num_kv_heads=4, head_dim=256)
DENSE_8B = ModelPreset(name="dense_8b", num_q_heads=32, num_kv_heads=8, head_dim=128)
QWEN3_14B = ModelPreset(name="qwen3_14b", num_q_heads=40, num_kv_heads=8, head_dim=128)

CONFIG = InferenceBenchmarkConfig(
    name="recent_gqa",
    models=with_tp_shards(MINIMAX_M2, [1, 8])
    + [LAGUNA_SWA]
    + with_tp_shards(RIPPLE_FULL, [1, 8])
    + with_tp_shards(RIPPLE_SWA, [1, 8])
    + with_tp_shards(STEP37_SWA, [1, 4])
    + [STEP5_FULL, STEP5_SWA]
    + with_tp_shards(MUSE_SPARK_FULL, [1, 8])
    + with_tp_shards(MUSE_SPARK_SWA, [1, 8])
    + [QWEN36_27B, DENSE_8B, QWEN3_14B],
    context_seqlens=[2048, 8192, 32768],
    context_chunked_shapes=[(512, 131072), (1024, 131072)],
    # q_tokens = 1 + MTP; the recipes ship MTP 3 and 7 (MTP 15 only for MiniMax-M2 / Laguna).
    generation_shapes=[(q, 131072) for q in (1, 4, 8)],
    generation_batch_sizes=[1, 128],
    kv_cache_dtypes=["bfloat16", "fp8_e4m3"],
    page_size=16,
    backends=["cudnn", "cudnn_oss", "cudnn_auto", "flashinfer"],
    kv_layouts=["paged"],
    kv_min_fills=[1.0, 0.25],
)
