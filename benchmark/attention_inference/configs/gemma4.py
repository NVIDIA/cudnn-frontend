# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Google Gemma 4 attention, inference phases.

Five of every six layers are sliding-window (W=1024) at head_dim 256; the rest are global layers at head_dim 512 with fewer
KV heads, whose K and V are one tensor (attention_k_eq_v). The global presets read K and V as separate tensors, so their
KV bytes are counted twice. Shapes from the HF configs (head_dim, global_head_dim, num_global_key_value_heads,
sliding_window). Shards follow the vLLM recipes (31B at TP1 / TP4, 26B-A4B at TP1).

Same attention shape, so these rows also stand for:
- gemma4_26b: gemma-4-26B-A4B-it, diffusiongemma-26B-A4B-it; the SWA preset also gemma-4-12B-it.
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

GEMMA4_31B_SWA = ModelPreset(name="gemma4_31b_swa", num_q_heads=32, num_kv_heads=16, head_dim=256, sliding_window_size=1024)
GEMMA4_31B_GLOBAL = ModelPreset(name="gemma4_31b_global", num_q_heads=32, num_kv_heads=4, head_dim=512)
GEMMA4_26B_SWA = ModelPreset(name="gemma4_26b_swa", num_q_heads=16, num_kv_heads=8, head_dim=256, sliding_window_size=1024)
GEMMA4_26B_GLOBAL = ModelPreset(name="gemma4_26b_global", num_q_heads=16, num_kv_heads=2, head_dim=512)

CONFIG = InferenceBenchmarkConfig(
    name="gemma4",
    models=with_tp_shards(GEMMA4_31B_SWA, [1, 4]) + with_tp_shards(GEMMA4_31B_GLOBAL, [1, 4]) + [GEMMA4_26B_SWA, GEMMA4_26B_GLOBAL],
    context_seqlens=[2048, 8192, 32768],
    context_chunked_shapes=[(512, 131072), (1024, 131072)],
    # q_tokens = 1 + MTP; the recipes ship MTP 3, 4 and 7.
    generation_shapes=[(q, 131072) for q in (1, 4, 8)],
    generation_batch_sizes=[1, 128],
    kv_cache_dtypes=["bfloat16", "fp8_e4m3"],
    page_size=16,
    backends=["cudnn", "cudnn_oss", "cudnn_auto", "flashinfer"],
    kv_layouts=["paged"],
    kv_min_fills=[1.0, 0.25],
)
