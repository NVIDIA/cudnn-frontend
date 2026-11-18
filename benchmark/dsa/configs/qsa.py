# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Query Sparse Attention (QSA) benchmark configuration.

QSA is the block-sparse attention of the packed Miles training interface
(docs/operations/sparse_attention/qsa.md): BF16 GQA with d=256 per head, K/V
kept per KV head (no shared latent record), and a per-query selection of
four-token blocks — up to 512 complete causal blocks from the indexer plus
the partial causal tail — so a query gathers at most ~2k tokens whatever the
sequence length. The model's geometry is 24 query heads over 2 KV heads.

Swept across tensor-parallel shards the way the inference suite does (TP
1/2/4/8 -> 24/2, 12/1, 6/1, 3/1 heads): heads shard, tokens do not, so every
rank runs the full packed sequence with its slice of heads; the two KV heads
divide at TP 2 and replicate from TP 4 on. The kernels serve 24/2, 12/1 and
6/1; the TP 8 shard (3/1) is declined by the contract and lands as a skipped
row, so the gap stays visible in the CSV rather than silently absent.

One packed sequence per case, 2k..32k tokens. Per-token work is bounded by
the block budget, so time grows about linearly with length once the budget
binds (short sequences gather fewer blocks than the budget).

Usage:
    python -m benchmark.dsa.runner --config qsa
    python -m benchmark.dsa.runner --config qsa --dry-run
    python -m benchmark.dsa.runner --config qsa --filter tp4 --pass bwd
"""

from ..config_types import DsaBenchmarkConfig, ModelPreset, with_tp_shards

QSA_24Q_2KV = ModelPreset(
    name="qsa",
    kind="qsa",
    num_q_heads=24,
    num_kv_heads=2,
    head_dim_qk=256,
    head_dim_vo=256,
    topk=512,  # block budget per query: complete causal blocks, plus the partial tail block
    block_size=4,
    indexer_topk=0,
    has_sink=False,
)

CONFIG = DsaBenchmarkConfig(
    name="qsa",
    models=with_tp_shards(QSA_24Q_2KV, [1, 2, 4, 8]),
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
    use_topk_length=False,
    num_iterations=20,
    num_warmup_iterations=5,
    output_dir="results",
)
