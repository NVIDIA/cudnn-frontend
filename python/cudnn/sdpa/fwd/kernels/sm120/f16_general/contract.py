# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 general half identity; no runtime imports."""

NAME = "sdpa_fwd_sm120_f16_general"
TEMPLATE_NAME = "prefill_f16"
KERNEL_FILE = "sm120/f16_general/kernel.py"
LEGACY_ROW = "sdpa_fwd_prefill_sm120"
QUERY_TILES = (64, 128)
DEFAULT_QUERY_TILE = 128


def mismatch(facts, knobs):
    if knobs is not None and knobs.tile_m not in (None, *QUERY_TILES):
        return f"{NAME} runs TILE_M in {QUERY_TILES}"
    return None
