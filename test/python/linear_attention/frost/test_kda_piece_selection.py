# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side invariants for KDA's packed-batch piece selection."""

import pytest

from cudnn.linear_attention.frost.common.piece_chain import KDA_RAGGED_MAX_PIECES, choose_kda_pieces

pytestmark = pytest.mark.L0


def _choice(**overrides):
    geometry = dict(
        num_seqs=8,
        heads_out=32,
        num_sm=148,
        total_tokens=32768,
        b_t=16,
        cadence_tokens=0,
        batch_invariant=False,
        expand_num=1,
        min_tokens=1024,
        enable_ragged_fallback=True,
    )
    geometry.update(overrides)
    return choose_kda_pieces(**geometry)


def test_kda_ragged_fallback_keeps_a_device_length_rule():
    """An oversubscribed packed shape retains pieces for device-side length selection."""
    pieces, unit_chunks, length_rule = _choice()
    assert 1 < pieces <= KDA_RAGGED_MAX_PIECES
    assert unit_chunks == 1
    assert length_rule


def test_kda_ragged_fallback_is_sm100_opt_in():
    """A caller that has not qualified the fallback preserves the old uncut result."""
    assert _choice(enable_ragged_fallback=False) == (0, 1, False)


def test_kda_ragged_fallback_does_not_replace_an_ordinary_chain():
    """An underfilled shape keeps the ordinary non-length-rule schedule."""
    pieces, unit_chunks, length_rule = _choice(num_seqs=1)
    assert pieces > 0
    assert unit_chunks == 1
    assert not length_rule


@pytest.mark.parametrize(
    "overrides",
    [
        {"total_tokens": 8192},
        {"num_seqs": 19},
    ],
    ids=["one-length-piece", "more-than-four-nominal-waves"],
)
def test_kda_ragged_fallback_stays_bounded(overrides):
    """The fallback avoids fixed overhead on one-piece and heavily oversubscribed shapes."""
    assert _choice(**overrides) == (0, 1, False)


def test_kda_batch_invariant_selection_is_unchanged():
    """The public batch-invariant request owns its length rule on every architecture."""
    enabled = _choice(batch_invariant=True)
    disabled = _choice(batch_invariant=True, enable_ragged_fallback=False)
    assert enabled == disabled
    assert enabled[0] > 0 and enabled[2]


def test_kda_ragged_fallback_caps_workspace_growth():
    """Long packed capacities do not grow the automatic fallback past its measured cap."""
    pieces, unit_chunks, length_rule = _choice(total_tokens=256 * 1024)
    assert pieces == KDA_RAGGED_MAX_PIECES
    assert unit_chunks == 1
    assert length_rule
