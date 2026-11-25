# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""benchmark/placement_gen scoring: a cell's measured default counts as FROST for forward and backward plan names."""

import importlib.util
from pathlib import Path

import pytest

_GEN = Path(__file__).resolve().parents[4] / "benchmark" / "placement_gen" / "gen.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("placement_gen_gen", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.L0
@pytest.mark.parametrize("plan, frost", [("sdpa_fwd_sm80", True), ("sdpa_bwd_sm80", True), ("eng7", False), (None, False)])
def test_default_plan_direction(gen, plan, frost):
    assert gen._ran_frost(plan) is frost


@pytest.mark.L0
@pytest.mark.parametrize("default_plan", ["sdpa_bwd_sm80", "eng7"])
def test_changed_cells_follow_the_default(gen, default_plan):
    """A rule that switches the measured default is a changed cell in both directions; one that keeps it is not."""
    x = dict(be=2.0, fr=1.0, au=1.0 if default_plan.startswith("sdpa_") else 2.0, dflt=gen._ran_frost(default_plan))
    keep, switch = ("always", "never") if x["dflt"] else ("never", "always")
    assert gen._changed([x], (keep, None))[1] == []
    assert len(gen._changed([x], (switch, None))[1]) == 1
