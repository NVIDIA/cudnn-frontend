# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 key-group half decode identity; no runtime imports."""

NAME = "sdpa_fwd_sm120_decode_key_group"
TEMPLATE_NAME = "decode_f16"
KERNEL_FILE = "sm120/decode_f16_key_group/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm120"
QUERY_TILES = (16, 32)
DEFAULT_QUERY_TILE = 16


def mismatch(facts, knobs):
    if knobs is not None and knobs.tile_m not in (None, *QUERY_TILES):
        return f"{NAME} runs TILE_M in {QUERY_TILES}"
    return None
