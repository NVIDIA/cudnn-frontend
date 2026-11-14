# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Xiaomi MiMo-V2 attention, inference phases: head_dim 192 for Q / K and 128 for V / O.

Full-attention layers (about 1 in 5) use fewer KV heads than the sliding-window ones; the sliding-window layers (W=128)
carry per-head attention sinks. Shapes from the HF configs (head_dim 192, v_head_dim 128, swa_num_key_value_heads,
add_swa_attention_sink_bias). Shards follow the vLLM recipes: Flash at TP4, Pro at TP8, and TP1 for DEP.

Same attention shape, so these rows also stand for:
- mimo_flash: MiMo-V2-Flash, MiMo-V2.5, MiMo-V2.6-Flash-RL.
- mimo_pro: MiMo-V2.5-Pro, MiMo-V2.6-Pro-RL.
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

MIMO_FLASH_FULL = ModelPreset(name="mimo_flash_full", num_q_heads=64, num_kv_heads=4, head_dim_qk=192, head_dim_vo=128)
MIMO_FLASH_SWA = ModelPreset(name="mimo_flash_swa", num_q_heads=64, num_kv_heads=8, head_dim_qk=192, head_dim_vo=128, sliding_window_size=128, has_sink=True)
MIMO_PRO_FULL = ModelPreset(name="mimo_pro_full", num_q_heads=128, num_kv_heads=8, head_dim_qk=192, head_dim_vo=128)
MIMO_PRO_SWA = ModelPreset(name="mimo_pro_swa", num_q_heads=128, num_kv_heads=8, head_dim_qk=192, head_dim_vo=128, sliding_window_size=128, has_sink=True)

CONFIG = InferenceBenchmarkConfig(
    name="mimo_v2",
    models=with_tp_shards(MIMO_FLASH_FULL, [1, 4])
    + with_tp_shards(MIMO_FLASH_SWA, [1, 4])
    + [MIMO_PRO_FULL]  # its TP8 shard (16/1) is mimo_flash_full-tp4
    + with_tp_shards(MIMO_PRO_SWA, [1, 8]),
    context_seqlens=[2048, 8192, 32768],
    context_chunked_shapes=[(512, 131072), (1024, 131072)],
    # q_tokens = 1 + MTP; the recipes ship MTP 1 and 7, and 7 is the bandwidth-heavier.
    generation_shapes=[(q, 131072) for q in (1, 8)],
    generation_batch_sizes=[1, 128],
    kv_cache_dtypes=["bfloat16", "fp8_e4m3"],
    page_size=16,
    backends=["cudnn", "cudnn_oss", "cudnn_auto", "flashinfer"],
    # The paged prefill leg runs V / O at the Q / K width, so the context phase also needs the dense form.
    kv_layouts=["dense", "paged"],
    kv_min_fills=[1.0, 0.25],
)
