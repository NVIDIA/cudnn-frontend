# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which SM80 half backward body a declaration runs; no runtime imports."""

LEGACY_ROW = "sdpa_bwd_sm80"
D64 = "sdpa_bwd_sm80_f16_d64"
TWO_SUBGROUP = "sdpa_bwd_sm80_f16_two_subgroup"
# The D64 body's fixed tile (f16_d64/kernel.py M_BLOCK, N_BLOCK); it requires whole tiles.
D64_TILE = (64, 128)


def _packed_stats(facts) -> bool:
    if facts.stats_t is None:
        return True
    expected = 1
    for size, stride in reversed(tuple(zip(facts.stats_t.get_dim(), facts.stats_t.get_stride()))):
        if size != 1:
            if stride != expected:
                return False
            expected *= size
    return True


def select_implementation(facts):
    """The D64 body serves plain dense MHA D64 graphs on whole tiles with packed Stats; the two-subgroup body serves
    every other graph. The adapter's own choice (api_dsl._sm80_d64_fast_path_eligible) is the same rule."""
    plain = not (facts.causal or facts.window_left is not None or facts.bottom_right or facts.padded)
    plain = plain and not (facts.has_bias or facts.has_sink or facts.deterministic)
    whole_tiles = facts.s_q % D64_TILE[0] == 0 and facts.s_kv % D64_TILE[1] == 0
    if not facts.thd and (facts.d_qk, facts.d_v) == (64, 64) and facts.h_q == facts.h_kv and whole_tiles and plain and _packed_stats(facts):
        return D64
    return TWO_SUBGROUP
