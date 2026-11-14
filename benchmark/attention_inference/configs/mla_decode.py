# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V3-style MLA decode (absorbed, 576/512), the form vLLM serves at generation.

Every query head attends to one shared latent record: K reads 512 latent + 64 rope channels, V the first 512 of the same
record, and the softmax scale is the unabsorbed QK width (1/sqrt(192)). Context runs unabsorbed; see ``mla_prefill``.

Shapes from the HF configs (kv_lora_rank 512, qk_rope_head_dim 64, qk_nope_head_dim 128, v_head_dim 128). Shards follow the
vLLM recipes: TP1 is the DEP deployment (attention data-parallel, all heads per GPU), TP8 the TP / TEP one.

Same attention shape, so these rows also stand for:
- deepseek_v3 (128 heads): DeepSeek-V3, DeepSeek-V3.1, DeepSeek-R1.
- kimi_k2 (64 heads): Kimi-K2, K2.5, K2.6; Ling-2.6-1T, Ling-3.1-flash, Ring-2.6-1T.
The sparse variants (DeepSeek-V3.2, GLM-5.x, Mistral-Large-4) read this record through a top-k index; the ``dsa`` suite
benchmarks that path.
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

DEEPSEEK_V3 = ModelPreset(
    name="deepseek_v3",
    num_q_heads=128,
    num_kv_heads=1,
    head_dim_qk=576,
    head_dim_vo=512,
    kind="mla_absorbed",
    sm_scale=192**-0.5,
)

KIMI_K2 = ModelPreset(
    name="kimi_k2",
    num_q_heads=64,
    num_kv_heads=1,
    head_dim_qk=576,
    head_dim_vo=512,
    kind="mla_absorbed",
    sm_scale=192**-0.5,
)

CONFIG = InferenceBenchmarkConfig(
    name="mla_decode",
    models=with_tp_shards(DEEPSEEK_V3, [1, 8]) + with_tp_shards(KIMI_K2, [1, 8]),
    # q_tokens = 1 + MTP; the recipes ship MTP 3 and Eagle3 7.
    generation_shapes=[(q, 131072) for q in (1, 4, 8)],
    generation_batch_sizes=[1, 128],
    kv_cache_dtypes=["bfloat16", "fp8_e4m3"],
    page_size=64,
    backends=["cudnn", "cudnn_oss", "cudnn_auto", "flashinfer", "flash_mla"],
    kv_layouts=["paged"],
    kv_min_fills=[1.0, 0.25],
)
