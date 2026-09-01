# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Popular-model attention presets for the sdpa/suites framework.

This is a GLOBAL/FULL attention suite: presets pin the head/dim geometry of a
model's dense attention layers and feed the same fuzzer/exec path as the
random suites (batch, sequence lengths, layout, mask flavor and data stay
fuzzed). Sparse-attention variants (DSA/NSA/indexer top-k paths of the
DeepSeek/Kimi/GLM 5.x families) are out of scope here — for those models the
preset covers their dense/full-attention layers only.

Field names mirror benchmark/attention_training's ModelPreset so the two
catalogs stay conciliable.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPreset:
    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    # Mask-flavor weights for the fuzzer (model-appropriate: e.g. gpt-oss
    # weights sliding-window causal heavily, DiT-style models use no mask).
    mask_weights: dict = field(default_factory=lambda: dict(causal=10, no_mask=2))
    # Attention sinks (gpt-oss style). Fuzzed {True,False} when the model has
    # them, pinned False otherwise (and always False for s_q==1 decode).
    with_sink: bool = False


LLAMA31 = ModelPreset(
    name="llama31",
    num_q_heads=64,
    num_kv_heads=8,
    head_dim_qk=128,
    head_dim_vo=128,
)

QWEN35 = ModelPreset(
    name="qwen35",
    num_q_heads=32,
    num_kv_heads=2,
    head_dim_qk=256,
    head_dim_vo=256,
)

GPT_OSS = ModelPreset(
    name="gpt_oss",
    num_q_heads=64,
    num_kv_heads=8,
    head_dim_qk=64,
    head_dim_vo=64,
    # Alternating dense-causal / sliding-window-causal layers, with sinks.
    mask_weights=dict(causal=5, left_window_only=10, no_mask=1),
    with_sink=True,
)

DSV3 = ModelPreset(
    name="dsv3",
    num_q_heads=128,
    num_kv_heads=128,
    head_dim_qk=192,
    head_dim_vo=128,
)

KIMI_K26 = ModelPreset(
    name="kimi_k26",
    num_q_heads=64,
    num_kv_heads=64,
    head_dim_qk=192,
    head_dim_vo=128,
)

GLM46 = ModelPreset(
    name="glm46",
    num_q_heads=96,
    num_kv_heads=8,
    head_dim_qk=128,
    head_dim_vo=128,
)

CATALOG = [LLAMA31, QWEN35, GPT_OSS, DSV3, KIMI_K26, GLM46]
