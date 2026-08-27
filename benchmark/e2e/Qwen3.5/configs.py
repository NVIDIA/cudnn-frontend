# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Published Qwen3.5 / Qwen3.6 layer dimensions, read from each model's `config.json`.

Only NVIDIA-checkable published configurations from the `Qwen` org are listed; community
quantizations and derivatives are not. Qwen3.6-27B and Qwen3.6-35B-A3B are field-for-field
identical to their Qwen3.5 counterparts and are aliases rather than separate entries.

Every one of these shares the same linear-attention and short-convolution geometry: key and
value head dim 128, 16 key heads, convolution width 4. Only the value-head count moves, so
the convolution's channel count is one of four values:

    conv_dim = 2 * linear_num_key_heads * 128 + linear_num_value_heads * 128
             = 6144 (v=16) | 8192 (v=32) | 10240 (v=48) | 12288 (v=64)

Full attention is head_dim 256 with GQA in all of them, and `num_attention_heads * 256`
exceeds `hidden_size` in six of the eight -- which is why these are measured as blocks
rather than through a config whose attention derives `head_dim = hidden_size // num_heads`.
"""

from __future__ import annotations


def _dense(hidden, layers, intermediate, v_heads, q_heads, kv_heads):
    return {
        "kind": "dense",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "intermediate_size": intermediate,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": v_heads,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "num_attention_heads": q_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": 256,
        "vocab_size": 248320,
    }


def _moe(hidden, layers, v_heads, q_heads, kv_heads, experts, top_k, moe_inter, shared_inter):
    cfg = _dense(hidden, layers, None, v_heads, q_heads, kv_heads)
    cfg.update(
        kind="moe",
        num_experts=experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_inter,
    )
    return cfg


MODELS = {
    # dense
    "Qwen3.5-0.8B": _dense(1024, 24, 3584, 16, 8, 2),
    "Qwen3.5-2B": _dense(2048, 24, 6144, 16, 8, 2),
    "Qwen3.5-4B": _dense(2560, 32, 9216, 32, 16, 4),
    "Qwen3.5-9B": _dense(4096, 32, 12288, 32, 16, 4),  # intermediate_size verified against config.json
    "Qwen3.5-27B": _dense(5120, 64, 17408, 48, 24, 4),
    # mixture of experts
    "Qwen3.5-35B-A3B": _moe(2048, 40, 32, 16, 2, 256, 8, 512, 512),
    "Qwen3.5-122B-A10B": _moe(3072, 48, 64, 32, 2, 256, 8, 1024, 1024),
    "Qwen3.5-397B-A17B": _moe(4096, 60, 64, 32, 2, 512, 10, 1024, 1024),
}

# Qwen3.6 published no new layer geometry.
ALIASES = {"Qwen3.6-27B": "Qwen3.5-27B", "Qwen3.6-35B-A3B": "Qwen3.5-35B-A3B"}


def get(name):
    """Resolve a model name (or alias) to its layer dimensions."""
    name = ALIASES.get(name, name)
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS) + sorted(ALIASES)}")
    cfg = dict(MODELS[name])
    cfg["name"] = name
    cfg["conv_dim"] = 2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"] + cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    interval = cfg["full_attention_interval"]
    cfg["num_full_attention_layers"] = cfg["num_hidden_layers"] // interval
    cfg["num_linear_layers"] = cfg["num_hidden_layers"] - cfg["num_full_attention_layers"]
    return cfg
