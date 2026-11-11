# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-suite conftest for the FROST DSL SDPA engines."""

import pytest


@pytest.fixture(autouse=True)
def _frost_opt_in(monkeypatch):
    """The FROST manifest rows are opt-in; this suite exercises them.

    Per test rather than at import, so the flag cannot leak into the rest of a
    full ``pytest test/python`` session and silently opt everything in."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def pytest_collection_modifyitems(config, items):
    """The d512 2x2 twin arm (``d512_arm`` in test_sdpa_fwd_dsl_sm100.py) changes only what a d512 plan lowers to:
    drop the ``two_by_two`` cell of every callspec whose id names another flavor, so `-k "d512 or dsv4"` lists each
    d512 id exactly twice and the other flavors' ids stay single."""
    keep, drop = [], []
    for item in items:
        cs = getattr(item, "callspec", None)
        twin = cs is not None and cs.params.get("d512_arm") == "two_by_two"
        if twin and not ("d512" in item.nodeid or "dsv4" in item.nodeid):
            drop.append(item)
        else:
            keep.append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
