# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.8-Flash-Next QSA (Qwen sparse attention) benchmark configuration.

QSA is GQA, not flat MQA: 24 query heads over 2 KV heads with SEPARATE K and
V tensors of head_dim 256 (partial rotary on 64 channels). Each query
gathers 2048 tokens selected as 512 four-token blocks (``index_granularity
= 4``, forced tail block included), no sink. Its indexer is 4 MQA heads x
128 over one bf16 key per four-token block (``indexer_pool_ratio = 4``),
budget 512 blocks. 12 of 48 layers are sparse.

Every cell of this config is a coverage row today: the DSA core kernels
serve the shared K=V latent record only (``kv_layout = "separate"`` is
declined with its own reason) and the indexer kernels require 32 or 64
query heads per KV head. The rows document the gap with per-token byte
footprints (K+V record 2048 / 1056 / 576 bytes for bf16 / mxfp8 / nvfp4;
index key 64 / 33 / 17 bytes per token amortized over the pool). The
follow-up that turns the core rows into measurements is the compute
capability 10.7 sparse GQA d256 kernel (``cudnn.sdpa`` block-sparse GQA
forward: BSHD operands, 4-token block ids, paged pools with a block table),
which has a different operand model from the flat-row setup this suite
runs today.

Not in ``run_all.sh``'s default list; run it by name:
    python -m benchmark.dsa.runner --config qwen38_qsa
    python -m benchmark.dsa.runner --config qwen38_qsa --dry-run
    benchmark/dsa/run_all.sh <arch> <gpu> qwen38_qsa
"""

from ..config_types import DsaBenchmarkConfig, ModelPreset

QSA = ModelPreset(
    name="qsa_qwen38",
    num_q_heads=24,
    head_dim_qk=256,
    head_dim_vo=256,
    topk=2048,  # 512 four-token blocks
    indexer_topk=0,
    has_sink=False,
    num_kv_heads=2,
    kv_layout="separate",
    index_granularity=4,
    indexer_heads=4,
    indexer_pool_ratio=4,
)

CONFIG = DsaBenchmarkConfig(
    name="qwen38_qsa",
    models=[QSA],
    seqlens=[
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (32768, 32768),
    ],
    backends=["cudnn"],
    data_types=["bfloat16"],
    profile_pass="both",
    deterministic_bwd=[False],
    use_topk_length=True,
    num_iterations=20,
    num_warmup_iterations=5,
    output_dir="results",
    components=["core", "indexer"],
    phases=["context", "generation", "bprop"],
    kv_dtypes=["bfloat16", "mxfp8", "nvfp4"],
    paged=[False, True],
)
