# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declaration-only selection for the dense small-half backward implementations."""

from .d128_contract import ACCUMULATION, PAIR, PAIR_CCS, RECOMPUTE, select_implementation

LEGACY_ROW = "sdpa_bwd_sm100_d128"
RECOMPUTE_OWNER = "sdpa_bwd_sm100_d128_recompute"
PAIR_OWNER = "sdpa_bwd_sm100_d128_pair"


def default_dense_implementation(natural, cc, head_dim, skv, window_left, deterministic):
    """Rank legal dense bodies after the shared accumulation admission checks."""
    window_bounded = window_left is None or window_left + 1 + skv + 2 * 128 <= 2**31 - 1
    if deterministic:
        return PAIR if natural == ACCUMULATION and window_bounded and tuple(cc) in PAIR_CCS else RECOMPUTE
    window_in_envelope = window_left is None or (head_dim == 64 and window_bounded)
    return natural if window_in_envelope else RECOMPUTE


def pair_available(natural, cc, head_dim, skv, window_left):
    """The fixed pair's envelope, independent of its default ranking."""
    return default_dense_implementation(natural, cc, head_dim, skv, window_left, True) == PAIR


def legal_implementations(facts, version):
    """Return the existing default first and all admitted fixed dense bodies."""
    if facts.thd:
        return (LEGACY_ROW,)
    scale = facts.d_qk**-0.5 if facts.scale is None else facts.scale
    natural = select_implementation(facts.device_cc, version, facts.b, facts.h_q, facts.s_q, facts.s_kv, scale, d_qk=facts.d_qk, d_v=facts.d_v)
    pair = pair_available(natural, facts.device_cc, facts.d_qk, facts.s_kv, facts.window_left)
    if facts.h_q != facts.h_kv:
        return (PAIR_OWNER,) if facts.deterministic and pair else (LEGACY_ROW,)
    chosen = default_dense_implementation(natural, facts.device_cc, facts.d_qk, facts.s_kv, facts.window_left, facts.deterministic)
    if chosen == PAIR:
        return (PAIR_OWNER, RECOMPUTE_OWNER)
    owners = (RECOMPUTE_OWNER,) if chosen == RECOMPUTE else (LEGACY_ROW, RECOMPUTE_OWNER)
    return (*owners, PAIR_OWNER) if pair else owners
