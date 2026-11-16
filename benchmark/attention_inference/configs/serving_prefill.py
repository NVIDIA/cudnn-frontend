# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving prefill: the batched, mixed-length paged prefill a serving stack hands cuDNN through FlashInfer.

The model configs' context phase is one full-length request (batch 1, Q >= 256, exact total-Q declaration). FlashInfer's
paged prefill instead declares a batch envelope (max Q / max KV, no total-Q) over requests of different lengths: short
chunks over long caches (prefix-cache hits, chunked prefill), page 16 or 64. This config runs that form for the cudnn legs
and, as references on the same inputs, FlashInfer's FA2 backend and FlashInfer's own cudnn backend. The cases follow the
SM100 / Rubin placement studies of 2026-11-12 (heuristics.py / placement.py docstrings).
"""

from ..config_types import InferenceBenchmarkConfig, ModelPreset, with_tp_shards

LLAMA3_1 = ModelPreset(name="llama3.1", num_q_heads=64, num_kv_heads=8, head_dim=128)
LLAMA3_2_1B = ModelPreset(name="llama3.2-1b", num_q_heads=32, num_kv_heads=8, head_dim=64)
QWEN35 = ModelPreset(name="qwen35", num_q_heads=32, num_kv_heads=2, head_dim=256)

SHORT_CHUNKS = [
    dict(batch_size=b, q_tokens=q, kv_len=kv, kv_min_fill=f, q_min_fill=f) for b in (4, 16) for q in (16, 128) for kv in (8192, 65536) for f in (1.0, 0.5)
]
PAGE64 = [dict(batch_size=8, q_tokens=64, kv_len=kv, kv_min_fill=0.5, q_min_fill=0.5, page_size=64) for kv in (32768, 131072)]
LONG_CACHE = [dict(batch_size=4, q_tokens=128, kv_len=131072)]
CHUNK = [dict(batch_size=4, q_tokens=512, kv_len=32768, kv_min_fill=0.5, q_min_fill=0.5)]
FULL = [dict(batch_size=8, q_tokens=2048, kv_len=2048, kv_min_fill=0.25)]  # full prefills of mixed length (Q = KV each)

CONFIG = InferenceBenchmarkConfig(
    name="serving_prefill",
    models=with_tp_shards(LLAMA3_1, [1, 8]) + [LLAMA3_2_1B, QWEN35],
    context_cases=SHORT_CHUNKS + PAGE64 + LONG_CACHE + CHUNK + FULL,
    generation_shapes=[],
    kv_cache_dtypes=["bfloat16"],
    page_size=16,
    kv_layouts=["paged"],
    cudnn_context_total_q="none",
    backends=["cudnn", "cudnn_oss", "cudnn_auto", "flashinfer_fa2", "flashinfer_cudnn"],
)
