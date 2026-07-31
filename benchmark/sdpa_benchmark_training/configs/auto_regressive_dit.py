# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Autoregressive video DiT SDPA Benchmark Configuration

Benchmarks the self-attention in an autoregressive (world-model / next-frame)
video DiT. Unlike the bidirectional video DiTs (LTX-2, Wan 2.2), an
autoregressive DiT generates one frame at a time conditioned on a growing
frame-history KV cache, so each attention call has a short query (the new
frame's patch tokens) against a long key/value tensor (concatenated past
frames):

    s_q   in {985, 1024, 2048, 4096, 8192}    # one new frame, varied resolution
    s_kv  = 62208                              # cached history of past frames
    num_heads = 9
    head_dim  = 128

Tokens for the new frame = (H/p) * (W/p) for typical 720p..1440p patchified
inputs at patch sizes 1..4. The KV cache size of 62208 (= 486 * 128) is
representative of ~30s of generation history at the resolutions above.

Within a single attention call, the new-frame query tokens attend to the
entire past-frame KV cache (no causality across frames is enforced inside
the operator; frame-level autoregression is the outer loop), so the
operator-level mask is ``no_mask``.

Usage:
    python -m benchmark.sdpa_benchmark_training.runner --config auto_regressive_dit
    python -m benchmark.sdpa_benchmark_training.runner --config auto_regressive_dit --dry-run
"""

from ..config_types import ModelPreset, BenchmarkConfig

AR_DIT = ModelPreset(
    name="auto_regressive_dit",
    num_q_heads=9,
    num_kv_heads=9,
    head_dim=128,
)

CONFIG = BenchmarkConfig(
    name="auto_regressive_dit",
    models=[AR_DIT],
    seqlens=[
        (985, 62208),
        (1024, 62208),
        (2048, 62208),
        (4096, 62208),
        (8192, 62208),
    ],
    backends=["cudnn", "flash_attention_4"],
    data_types=["bfloat16", "fp8", "mxfp8"],
    attn_masks=["no_mask"],
    profile_pass="fwd",
    batch_size=1,
    num_iterations=10,
    output_dir="results",
)
