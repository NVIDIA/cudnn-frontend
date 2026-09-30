# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-suite conftest for the gated attention block (a FROST frontend-only API)."""

import pytest


@pytest.fixture(autouse=True)
def _frost_opt_in(monkeypatch):
    """Every FROST engine row is opt-in and the block drives them; opt in PER TEST.

    A module-level ``os.environ.setdefault`` leaked the flag into the rest of a
    full ``pytest test/python`` session (test_dispatch.py asserts it does not),
    so the opt-in lives here, scoped to each test by ``monkeypatch``."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
