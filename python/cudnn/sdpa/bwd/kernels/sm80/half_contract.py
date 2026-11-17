# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which SM80 half backward body a declaration runs; no runtime imports."""

LEGACY_ROW = "sdpa_bwd_sm80"
D64 = "sdpa_bwd_sm80_f16_d64"
TWO_SUBGROUP = "sdpa_bwd_sm80_f16_two_subgroup"
# The D64 body's fixed tile (f16_d64/kernel.py M_BLOCK, N_BLOCK); it requires whole tiles.
D64_TILE = (64, 128)


def uses_flat_grid(*, is_causal, deterministic, thd):
    """The two-subgroup LPT schedule is dense, causal and nondeterministic."""
    return is_causal and not deterministic and not thd


def launch_grid_mismatch(batch, heads, *, is_causal, deterministic, thd):
    from cudnn.sdpa.bwd.launch_grid import yz_grid_mismatch

    if uses_flat_grid(is_causal=is_causal, deterministic=deterministic, thd=thd):
        return None
    return yz_grid_mismatch(batch, heads)


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


def d64_fast_path_eligible(*, d_qk, d_v, h_q, h_kv, s_q, s_kv, mask_token, right_bound, causal_bottom_right, bw_kwargs) -> bool:
    """Whether the D64 body can serve this call EXACTLY: it computes a plain dense MHA backward and nothing else, so
    every condition here guards a feature it would silently ignore. The adapter's choice and the graph's are this rule."""
    if (d_qk, d_v) != (64, 64) or h_q != h_kv:
        return False
    if s_q % D64_TILE[0] != 0 or s_kv % D64_TILE[1] != 0:
        return False
    if mask_token != "none" or right_bound != 0 or causal_bottom_right:
        return False
    for feature in ("seq_kv_lens", "seq_len_q", "bias", "sinks", "rope_freqs"):
        if bw_kwargs.get(feature) is not None:
            return False
    return not bw_kwargs.get("deterministic")


def select_implementation(facts):
    """The body the SM80 half row runs: the adapter's rule on the arguments lowering hands it (lower_dsl_bwd:
    is_causal = causal or right_band_widening, window_size_right = right_bound under a widened band). THD and
    strided Stats always take the two-subgroup body, as in the adapter."""
    is_causal = facts.causal or facts.right_band_widening
    eligible = d64_fast_path_eligible(
        d_qk=facts.d_qk,
        d_v=facts.d_v,
        h_q=facts.h_q,
        h_kv=facts.h_kv,
        s_q=facts.s_q,
        s_kv=facts.s_kv,
        mask_token="none" if not is_causal and facts.window_left is None else "masked",
        right_bound=(facts.right_bound or 0) if facts.right_band_widening else 0,
        causal_bottom_right=facts.bottom_right,
        bw_kwargs=dict(
            seq_kv_lens=True if facts.padded else None,
            bias=True if facts.has_bias else None,
            sinks=True if facts.has_sink else None,
            deterministic=facts.deterministic,
        ),
    )
    return D64 if eligible and not facts.thd and _packed_stats(facts) else TWO_SUBGROUP
