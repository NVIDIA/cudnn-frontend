# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode-shaped sweep for the FROST graduation line: pure decode (q=1) and deeper
spec-decode verify widths (q=8, 16) across KV lengths and batch sizes, so the
s_q == 1 decision is not read off one (kv=128k, b in {1, 128}) point.
bf16 KV only; llama (d128) and qwen35 (d256) at TP1 / TP4 shards.
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

LLAMA3_1 = ModelPreset(name="llama3.1", num_q_heads=64, num_kv_heads=8, head_dim=128)
QWEN35 = ModelPreset(name="qwen35", num_q_heads=32, num_kv_heads=2, head_dim=256)

CONFIG = InferenceBenchmarkConfig(
    name="decode_sweep",
    models=with_tp_shards(LLAMA3_1, [1, 4]) + with_tp_shards(QWEN35, [1, 4]),
    context_seqlens=[],
    context_chunked_shapes=[],
    generation_shapes=[(q, kv) for q in (1, 8, 16) for kv in (2048, 8192, 32768, 131072)],
    generation_batch_sizes=[1, 8, 32, 128],
    kv_cache_dtypes=["bfloat16"],
    page_size=16,
    backends=["cudnn", "cudnn_oss"],
)
