# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The SM107 d512_f16_2x2 implementation's local identity and admission."""

from ..half_flavor_contract import padded_shape

NAME = "sdpa_fwd_sm107_d512_2x2"
TEMPLATE_NAME = "prefill_d512_f16_2x2"
KERNEL_FILE = "sm107/d512_f16_2x2/kernel.py"
QUERY_TILE = 64
CTA_MMA = 2


def mismatch(facts, knobs):
    if padded_shape(facts) not in ((512, 512), (576, 512)):
        return "SM107 2x2 serves the D512/V512 envelope or exact D576/V512"
    if knobs is not None and knobs.tile_m not in (None, QUERY_TILE):
        return f"this SM107 implementation requires TILE_M={QUERY_TILE}"
    return None
