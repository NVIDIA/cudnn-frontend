# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V4.1 sparse attention (DSA) benchmark configuration.

V4.1-Flash keeps V4's sparse core: flat MQA over a shared K=V record of
head_dim=512, 512 indexer-selected entries plus a 128-token window folded
into the index list, and a per-head sink. Its changes — 2x-compressed and
uncompressed pools, KV and index picks shared across layers — alter what
the indexer selects from, not the per-query gather, so the core preset is
V4-Flash's (K = 640, indexer_topk = 512, 64 heads). Only a Flash
checkpoint is published.

This config also enumerates the lightning indexer (32 heads x 128, one
shared key per entry, top-k 512) in both of V4.1's modes: ``dense`` scores
every causally visible key then takes the top-k; ``hierarchical`` is the
candidate-consumer step of layers 24/28/32/36, which score only the 16384
candidate positions the source layer published and take their own top-k.
Generation rows carry two query rows per request (``mtp_rows = 2``: the
next token and one MTP draft).

Axes: components core + indexer; phases context, generation, bprop;
kv_dtypes bfloat16, mxfp8, nvfp4; flat and paged pools. Cells no kernel
serves (NVFP4 anywhere, MXFP8 core, paged indexer, low-precision bprop) are
written as skipped coverage rows with the reason, not dropped.

Usage:
    python -m benchmark.dsa.runner --config deepseek_v41
    python -m benchmark.dsa.runner --config deepseek_v41 --dry-run
    python -m benchmark.dsa.runner --config deepseek_v41 --component indexer --phase generation
"""

from ..config_types import DsaBenchmarkConfig, ModelPreset

DSV41_FLASH = ModelPreset(
    name="dsv41_flash",
    num_q_heads=64,
    head_dim_qk=512,
    head_dim_vo=512,
    topk=640,  # 512 indexer-selected + 128 sliding window
    indexer_topk=512,
    has_sink=True,
    indexer_heads=32,
    indexer_modes=("dense", "hierarchical"),
    mtp_rows=2,
)

CONFIG = DsaBenchmarkConfig(
    name="deepseek_v41",
    models=[DSV41_FLASH],
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
