# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GLM-5.3 sparse attention (DSA) benchmark configuration.

GLM-5.3 (same architecture as GLM-5.2) runs DeepSeek-V3.2-style DSA on the
MLA latent: the absorbed query attends a shared 576-wide record — the 512 KV
latent plus 64 RoPE channels — and the 512 latent channels are read back as
V. The indexer (32 heads x 128) selects 2048 tokens per query; there is no
sliding-window fold and no sink, so the whole logical K is indexer-selected
and the plain LSE is the indexer's training signal (``indexer_topk = 0``).

GLM-5.3-Flash is the hybrid sibling: 34 KDA layers and 11 sparse NoPE MLA
layers whose latent has no RoPE slice (``qk_rope_head_dim = 0``), so the
record is 512 wide. Its indexer scores 4-token pools (``index_kpool = 4``)
and selects 512 of them = 2048 raw tokens, always adding the incomplete tail
pool (at most 3 tokens, left out of the preset's round 2048).

Heads: 64 for both, from the official HF configs (zai-org/GLM-5.3,
zai-org/GLM-5.3-Flash). The sweep sizes the KV pool as ``s_kv == s_q`` and
every query gathers the full top-k, as in ``deepseek_v4``.

Usage:
    python -m benchmark.dsa.runner --config glm53
    python -m benchmark.dsa.runner --config glm53 --dry-run
    python -m benchmark.dsa.runner --config glm53 --filter flash --pass bwd
"""

from ..config_types import DsaBenchmarkConfig, ModelPreset

GLM53 = ModelPreset(
    name="glm53",
    num_q_heads=64,
    head_dim_qk=576,  # 512 KV latent + 64 RoPE
    head_dim_vo=512,
    topk=2048,
    indexer_topk=0,
    has_sink=False,
)

GLM53_FLASH = ModelPreset(
    name="glm53_flash",
    num_q_heads=64,
    head_dim_qk=512,  # NoPE latent, no RoPE slice
    head_dim_vo=512,
    topk=2048,  # 512 pools x 4 tokens; forced tail (<= 3 tokens) omitted
    indexer_topk=0,
    has_sink=False,
)

CONFIG = DsaBenchmarkConfig(
    name="glm53",
    models=[GLM53, GLM53_FLASH],
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
)
