# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GLM-5.3 sparse attention (DSA) benchmark configuration.

GLM-5.3 (GLM-5.2's architecture) runs V3.2-style DSA on the MLA latent: the
absorbed query attends a shared 576-wide record (512 KV latent + 64 RoPE)
and reads the 512 latent channels back as V; token top-2048 with no window
fold and no sink, so the plain LSE is the indexer's training signal and
``indexer_topk = 0``. Its indexer has 32 heads x 128 over one key per token
and selects 2048 entries; generation rows carry two query rows per request
(``mtp_rows = 2``, one MTP draft).

GLM-5.3-Flash's sparse layers are NoPE MLA, so the record is 512 wide; its
indexer scores one key per four-token pool (``indexer_pool_ratio = 4``) and
selects 512 pools = 2048 raw tokens (``index_granularity = 4``; the
incomplete tail pool, at most 3 tokens, is omitted). 64 heads for both.

As in ``deepseek_v4``, ``s_kv == s_q`` and every context / bprop query
gathers its full top-k, the per-token upper bound: ``glm53`` as independent
unique random rows, ``glm53_flash`` as 512 random complete 4-token blocks
(``index_granularity = 4``), so its pool locality is modeled.

Axes: components core + indexer; phases context, generation, bprop;
kv_dtypes bfloat16, mxfp8, nvfp4; flat and paged pools. Cells no kernel
serves are written as skipped coverage rows with the reason.

Usage:
    python -m benchmark.dsa.runner --config glm53
    python -m benchmark.dsa.runner --config glm53 --dry-run
    python -m benchmark.dsa.runner --config glm53 --filter flash --component indexer
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
    indexer_heads=32,
    mtp_rows=2,
)

GLM53_FLASH = ModelPreset(
    name="glm53_flash",
    num_q_heads=64,
    head_dim_qk=512,  # NoPE latent
    head_dim_vo=512,
    topk=2048,  # 512 pools x 4 raw tokens
    indexer_topk=0,
    has_sink=False,
    index_granularity=4,
    indexer_heads=32,
    indexer_pool_ratio=4,
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
    components=["core", "indexer"],
    phases=["context", "generation", "bprop"],
    kv_dtypes=["bfloat16", "mxfp8", "nvfp4"],
    paged=[False, True],
)
