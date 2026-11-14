# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lookahead body's public identity and local geometry contract."""

NAME = "sdpa_fwd_sm100_d256_lookahead"
TEMPLATE_NAME = "prefill_d256_f16"
KERNEL_FILE = "sm100/d256_f16_lookahead/kernel.py"
QUERY_TILE = 128
CGA = 2


def mismatch(facts, knobs):
    if knobs is not None and knobs.tile_m not in (None, QUERY_TILE):
        return "D256 lookahead TILE_M must be 128"
    return None
