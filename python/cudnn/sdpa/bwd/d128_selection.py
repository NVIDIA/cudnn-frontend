# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declaration-only selection for the dense small-half backward implementations."""

from .d128_contract import ACCUMULATION, PAIR, PAIR_CCS, RECOMPUTE, select_implementation

LEGACY_ROW = "sdpa_bwd_sm100_d128"
RECOMPUTE_OWNER = "sdpa_bwd_sm100_d128_recompute"


def default_dense_implementation(natural, cc, head_dim, skv, window_left, deterministic):
    """Rank legal dense bodies after the shared accumulation admission checks."""
    window_bounded = window_left is None or window_left + 1 + skv + 2 * 128 <= 2**31 - 1
    if deterministic:
        return PAIR if natural == ACCUMULATION and window_bounded and tuple(cc) in PAIR_CCS else RECOMPUTE
    window_in_envelope = window_left is None or (head_dim == 64 and window_bounded)
    return natural if window_in_envelope else RECOMPUTE


def legal_implementations(facts, version):
    """Return the current default first, with recompute available for dense MHA."""
    if facts.thd or facts.h_q != facts.h_kv:
        return (LEGACY_ROW,)
    scale = facts.d_qk**-0.5 if facts.scale is None else facts.scale
    natural = select_implementation(facts.device_cc, version, facts.b, facts.h_q, facts.s_q, facts.s_kv, scale, d_qk=facts.d_qk, d_v=facts.d_v)
    chosen = default_dense_implementation(natural, facts.device_cc, facts.d_qk, facts.s_kv, facts.window_left, facts.deterministic)
    return (RECOMPUTE_OWNER,) if chosen == RECOMPUTE else (LEGACY_ROW, RECOMPUTE_OWNER)
