# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The role-split D512 half implementation's identity and local contract."""

NAME = "sdpa_fwd_sm100_d512_role_split"
TEMPLATE_NAME = "prefill_d512_f16"
KERNEL_FILE = "sm100/d512_f16_role_split/kernel.py"
QUERY_TILE = 128
CTA_MMA = 2
CLUSTER_M = 4


def mismatch(facts, knobs):
    if (facts.d_qk, facts.d_v) != (512, 512):
        return "the role-split implementation serves exact D_qk=D_v=512"
    if knobs is not None and knobs.tile_m not in (None, QUERY_TILE):
        return "D512 role-split TILE_M must be 128"
    return None
