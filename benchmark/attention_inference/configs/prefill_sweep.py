# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill sweep for the FROST placement tree: chunked prefill (small incoming chunk over a long cache) across chunk sizes,
cache lengths and head counts, plus dense squares, so the context-phase crossover (few heads / long KV wins, dense squares
lose) is located rather than read off two chunk sizes. bf16, batch 1; llama (d128) at TP 1/2/4/8, qwen35 (d256) at TP 1/4.
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

LLAMA3_1 = ModelPreset(name="llama3.1", num_q_heads=64, num_kv_heads=8, head_dim=128)
QWEN35 = ModelPreset(name="qwen35", num_q_heads=32, num_kv_heads=2, head_dim=256)

CONFIG = InferenceBenchmarkConfig(
    name="prefill_sweep",
    models=with_tp_shards(LLAMA3_1, [1, 2, 4, 8]) + with_tp_shards(QWEN35, [1, 4]),
    context_seqlens=[2048, 4096, 8192, 16384],
    context_chunked_shapes=[(q, kv) for q in (256, 512, 1024, 2048) for kv in (8192, 32768, 65536, 131072)],
    generation_shapes=[],
    kv_cache_dtypes=["bfloat16"],
    page_size=16,
    backends=["cudnn", "cudnn_oss"],
)
