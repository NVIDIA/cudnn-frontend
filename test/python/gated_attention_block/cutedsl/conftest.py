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


def _dense_serving_ccs():
    """The parts the block's dense bf16 / f16 forms are served on -- ONE definition, the api's (``DENSE_SERVING_CCS``: cc 10.7 and, since
    2026-10-29, cc 10.0); read lazily so collecting this directory never imports the package before a module asks for it."""
    from cudnn.gated_attention_block import DENSE_SERVING_CCS

    return tuple(tuple(int(x) for x in cc) for cc in DENSE_SERVING_CCS)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_rubin: gated_attention_block ACCEPT test -- needs the SM107 (Rubin) GPU the block targets; skipped on every other part "
        "(the reject / host tests carry no marker and run anywhere)",
    )
    config.addinivalue_line(
        "markers",
        "requires_blackwell: gated_attention_block ACCEPT test of the dense bf16 / f16 serving forms on a Blackwell-line part the block "
        "admits (a cc of DENSE_SERVING_CCS other than SM107: cc 10.0 today); skipped everywhere else -- the cc 10.0 accept rows CI runs",
    )
    config.addinivalue_line(
        "markers",
        "requires_dense_part: gated_attention_block ACCEPT test the dense bf16 / f16 forms run on EVERY served part (any cc of "
        "DENSE_SERVING_CCS: SM107 and the admitted Blackwell-line parts); skipped on every other part",
    )


def pytest_collection_modifyitems(config, items):
    """The hoisted markers of this suite: ``@pytest.mark.requires_rubin`` on an accept test skips it wherever the device is
    not SM107; ``@pytest.mark.requires_blackwell`` (2026-10-29) skips wherever the device is not an admitted Blackwell-line part
    of ``DENSE_SERVING_CCS`` (cc 10.0 today: the dense bf16 / f16 forms' accept rows there); ``@pytest.mark.requires_dense_part``
    skips wherever the device is not in ``DENSE_SERVING_CCS`` at all (the cells both parts run).

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
    served = _dense_serving_ccs()
    skip_rubin = pytest.mark.skip(reason=f"a Rubin (SM107) accept test; found {cc}")
    skip_blackwell = pytest.mark.skip(
        reason=f"an accept test of the dense forms on an admitted Blackwell-line part {tuple(c for c in served if c != _SM107)}; found {cc}"
    )
    skip_dense = pytest.mark.skip(reason=f"an accept test of the dense forms on a served part {served}; found {cc}")
    for item in items:
        if "requires_rubin" in item.keywords and cc != _SM107:
            item.add_marker(skip_rubin)
        if "requires_blackwell" in item.keywords and not (cc in served and cc != _SM107):
            item.add_marker(skip_blackwell)
        if "requires_dense_part" in item.keywords and cc not in served:
            item.add_marker(skip_dense)
