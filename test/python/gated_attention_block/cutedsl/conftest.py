# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-suite conftest for the gated attention block (a FROST frontend-only API)."""

import pytest
import torch

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


@pytest.fixture(autouse=True)
def _frost_opt_in(monkeypatch):
    """Every FROST engine row is opt-in and the block drives them; opt in PER TEST.

    A module-level ``os.environ.setdefault`` leaked the flag into the rest of a
    full ``pytest test/python`` session (test_dispatch.py asserts it does not),
    so the opt-in lives here, scoped to each test by ``monkeypatch``."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_rubin: gated_attention_block ACCEPT test -- needs the SM107 (Rubin) GPU the block targets; skipped on every other part "
        "(the reject / host tests carry no marker and run anywhere)",
    )


def pytest_collection_modifyitems(config, items):
    """The ONE hoisted ``requires_rubin`` marker of this suite: ``@pytest.mark.requires_rubin`` on an accept
    test skips it wherever the device is not SM107.

    The older modules each re-define ``requires_rubin = pytest.mark.skipif(_cc() != _SM107, ...)`` at module
    level (``test_block_end_to_end.py``, ``test_block_fp8.py``, ``test_block_mxfp8.py``, ``test_sdpa_stage_sm107.py``,
    ``test_proj_gemm.py``, ...); those stay as they are.  New modules spell ``requires_rubin = pytest.mark.requires_rubin``
    (a REGISTERED marker, see ``pytest_configure``) and get the same skip from here -- one definition of "Rubin" for
    every module that lands HERE from now on, instead of a copy per file.

    SCOPE: the modules under THIS directory (``test/python/gated_attention_block/cutedsl/``, where every gated-block
    CuTeDSL module lives) -- pytest registers and applies the marker from this conftest only.  A module elsewhere that
    spells ``@pytest.mark.requires_rubin`` gets an unregistered-marker warning and NO skip (its Rubin accept tests would
    RUN, and fail to compile, on an A100 / GB200 host): give it its own ``skipif``, or land it here."""
    cc = _cc()
    if cc == _SM107:
        return
    skip = pytest.mark.skip(reason=f"the block targets SM107 only; found {cc}")
    for item in items:
        if "requires_rubin" in item.keywords:
            item.add_marker(skip)
