# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Local identity and declaration boundaries of the batch-pair work item."""

NAME = "sdpa_fwd_sm107_small_batch_pair"
TEMPLATE_NAME = "prefill_d128_f16"
KERNEL_FILE = "sm107/small_f16_batch_pair/kernel.py"


def mismatch(facts, knobs):
    if facts.d_qk not in (32, 64) or not facts.uniform_dtype:
        return "batch-pair requires same-dtype D32/D64 half inputs"
    if not facts.has_bias or len(facts.bias_dim) != 4 or facts.bias_dim[0] != 1 or facts.b < 2:
        return "batch-pair requires a batch-broadcast bias and at least two batch entries"
    if facts.thd or facts.has_paged_kv or facts.has_epilogue_gate:
        return "batch-pair requires dense, unpaged, ungated graphs"
    if knobs is not None and (
        knobs.cga not in (None, 1) or knobs.tile_m not in (None, 128) or knobs.pack_gqa or (knobs.split_kv or 1) > 1 or knobs.warp_spec_cfg not in (None, 0)
    ):
        return "batch-pair requires CTA1, TILE_M=128, unsplit and unpacked heads"
    return None
