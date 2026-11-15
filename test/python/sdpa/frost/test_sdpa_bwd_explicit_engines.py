# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward explicit engines (no GPU): a multi-body row and the engines that own its bodies partition the row's graphs by
the body its declaration selects, and the owners inherit the row's heuristics decisions."""

from types import SimpleNamespace

import pytest

from cudnn.sdpa.bwd import engines as bwd_engines
from cudnn.sdpa.bwd.engines import Capabilities, EngineSpec, explicit_engine, mismatch, policy_name

pytestmark = pytest.mark.L0


def _lower(*args, **kwargs):
    raise AssertionError("not lowered")


def _row(delegated=frozenset()):
    caps = Capabilities(sm_lo=120, sm_hi=121, d=frozenset({64}), select_implementation=lambda facts: facts.body, delegated=delegated)
    return EngineSpec("row", caps, _lower)


def test_selected_body_partitions_row_and_owner(monkeypatch):
    monkeypatch.setattr(bwd_engines, "_declared_mismatch", lambda caps, facts, requested: None)
    row = _row(delegated=frozenset({"owner_a"}))
    owner = explicit_engine(row, "owner_a", _lower)
    a, b = SimpleNamespace(body="owner_a"), SimpleNamespace(body="b")
    assert mismatch(owner.capabilities, a) is None
    assert mismatch(row.capabilities, a) == "this graph is owned by the explicit implementation engine owner_a"
    assert mismatch(owner.capabilities, b) == "owner_a does not run this graph; its declaration selects b"
    assert mismatch(row.capabilities, b) is None


def test_declared_reason_comes_before_the_selection(monkeypatch):
    monkeypatch.setattr(bwd_engines, "_declared_mismatch", lambda caps, facts, requested: "declared")
    row = _row(delegated=frozenset({"owner_a"}))
    assert mismatch(row.capabilities, SimpleNamespace(body="owner_a")) == "declared"


def test_owner_names_its_body_and_inherits_the_row_policy():
    row = _row(delegated=frozenset({"owner_a"}))
    owner = explicit_engine(row, "owner_a", _lower, d=frozenset({32, 64}))
    assert (owner.name, owner.capabilities.implementation, owner.capabilities.delegated) == ("owner_a", "owner_a", frozenset())
    assert owner.capabilities.d == frozenset({32, 64}) and owner.capabilities.select_implementation is row.capabilities.select_implementation
    assert (policy_name(owner), policy_name(row)) == ("row", "row")
