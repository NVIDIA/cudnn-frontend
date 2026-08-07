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
