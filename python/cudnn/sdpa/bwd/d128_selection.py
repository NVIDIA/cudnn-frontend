# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declaration-only selection for the small-half backward implementations."""

from .d128_contract import ACCUMULATION, PAIR, PAIR_CCS, RECOMPUTE, select_implementation

LEGACY_ROW = "sdpa_bwd_sm100_d128"
RECOMPUTE_OWNER = "sdpa_bwd_sm100_d128_recompute"
PAIR_OWNER = "sdpa_bwd_sm100_d128_pair"
ACCUMULATION_OWNER = "sdpa_bwd_sm100_d128_accumulation"
TWO_CTA_OWNER = "sdpa_bwd_sm100_d128_accumulation_2cta"


def default_dense_implementation(natural, cc, head_dim, skv, window_left, deterministic):
    """Rank legal dense bodies after the shared accumulation admission checks."""
    window_bounded = window_left is None or window_left + 1 + skv + 2 * 128 <= 2**31 - 1
    if deterministic:
        return PAIR if natural == ACCUMULATION and window_bounded and tuple(cc) in PAIR_CCS else RECOMPUTE
    return ACCUMULATION if accumulation_available(natural, head_dim, skv, window_left) else RECOMPUTE


def pair_available(natural, cc, head_dim, skv, window_left):
    """The fixed pair's envelope, independent of its default ranking."""
    return default_dense_implementation(natural, cc, head_dim, skv, window_left, True) == PAIR


def accumulation_available(natural, head_dim, skv, window_left):
    """Safety envelope for the fixed one-CTA dense body."""
    window_bounded = window_left is None or window_left + 1 + skv + 2 * 128 <= 2**31 - 1
    return natural == ACCUMULATION and (window_left is None or (head_dim == 64 and window_bounded))


def grouped_implementations(facts):
    """Offer both feasible bodies; the existing sequence policy only orders them."""
    from .d128_grouped_2cta_contract import CtaPairDecline, PAIR_BODY, dense_capacities, select_cta_pair, validate_thd_bounds

    layout = "thd" if facts.thd else "dense"
    if facts.deterministic or facts.h_q == facts.h_kv:
        return (ACCUMULATION_OWNER,)
    try:
        capacities = (
            (
                min(facts.b * facts.s_q, facts.max_total_seq_len_q),
                min(facts.b * facts.s_kv, facts.max_total_seq_len_kv),
                facts.max_total_seq_len_q,
                facts.max_total_seq_len_kv,
            )
            if facts.thd
            else dense_capacities(facts.b, facts.s_q, facts.s_kv)
        )
        validate_thd_bounds(facts.b, facts.h_q, facts.h_kv, facts.s_q, facts.s_kv, *capacities)
        # Declaration admission assumes a cluster can be launched. The fixed
        # host performs the real device occupancy query and declines on failure.
        select_cta_pair("on", layout=layout, s_q_max=facts.s_q, s_kv_max=facts.s_kv, cc=facts.device_cc, max_active_pairs=1)
    except (ValueError, CtaPairDecline):
        return (ACCUMULATION_OWNER,)
    preferred = select_cta_pair("auto", layout=layout, s_q_max=facts.s_q, s_kv_max=facts.s_kv, cc=facts.device_cc, max_active_pairs=1)
    return (TWO_CTA_OWNER, ACCUMULATION_OWNER) if preferred.body == PAIR_BODY else (ACCUMULATION_OWNER, TWO_CTA_OWNER)


def legal_implementations(facts, version):
    """Return the existing default first and all admitted fixed bodies."""
    if facts.thd:
        return grouped_implementations(facts) if facts.h_q != facts.h_kv else (ACCUMULATION_OWNER,)
    scale = facts.d_qk**-0.5 if facts.scale is None else facts.scale
    natural = select_implementation(facts.device_cc, version, facts.b, facts.h_q, facts.s_q, facts.s_kv, scale, d_qk=facts.d_qk, d_v=facts.d_v)
    pair = pair_available(natural, facts.device_cc, facts.d_qk, facts.s_kv, facts.window_left)
    if facts.h_q != facts.h_kv:
        return (PAIR_OWNER,) if facts.deterministic and pair else grouped_implementations(facts)
    chosen = default_dense_implementation(natural, facts.device_cc, facts.d_qk, facts.s_kv, facts.window_left, facts.deterministic)
    if chosen == PAIR:
        return (PAIR_OWNER, RECOMPUTE_OWNER)
    owners = (RECOMPUTE_OWNER,) if chosen == RECOMPUTE else (ACCUMULATION_OWNER, RECOMPUTE_OWNER)
    return (*owners, PAIR_OWNER) if pair else owners
