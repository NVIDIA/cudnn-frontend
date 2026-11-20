# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from cudnn.linear_attention import kda_heuristics

pytestmark = pytest.mark.L0


OFFERED = {
    "kda_frost": 0,
    "kda_cutile": 1,
    "kda_summary_frost": 2,
    "kda_hopper": 4,
    "kda_hopper_cuda": 5,
    "kda_summary_hopper_cuda": 7,
    "kda_summary_hopper_scan": 8,
}


def facts(**changes):
    values = dict(
        op="KDA_SUMMARY",
        is_bwd=True,
        wants_transition=False,
        d_final_state_dtype=None,
        output_context=False,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_hopper_g_only_promotes_compact_scan(monkeypatch):
    monkeypatch.setattr(kda_heuristics, "_current_sm", lambda: 90)
    assert kda_heuristics._ordered_names(facts(), OFFERED)[0] == "kda_summary_hopper_scan"


@pytest.mark.parametrize(
    "changes",
    [
        {"wants_transition": True},
        {"d_final_state_dtype": object()},
        {"output_context": True},
        {"is_bwd": False},
        {"op": "KDA"},
    ],
)
def test_non_g_only_graphs_keep_existing_hopper_order(monkeypatch, changes):
    monkeypatch.setattr(kda_heuristics, "_current_sm", lambda: 90)
    ordered = kda_heuristics._ordered_names(facts(**changes), OFFERED)
    assert ordered[:2] == ["kda_hopper_cuda", "kda_hopper"]
    assert ordered.index("kda_summary_hopper_scan") == len(ordered) - 1


def test_non_hopper_keeps_manifest_order(monkeypatch):
    monkeypatch.setattr(kda_heuristics, "_current_sm", lambda: 100)
    assert kda_heuristics._ordered_names(facts(), OFFERED) == list(OFFERED)
