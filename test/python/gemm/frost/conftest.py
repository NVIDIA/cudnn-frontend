# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make ``cudnn.gemm.frost`` importable: overlay the source ``python/cudnn``
dir onto the installed ``cudnn.__path__`` and the nested ``cudnn.gemm.__path__``.
Unnecessary once the engine ships in the built frontend package."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import cudnn

_SRC_CUDNN = Path(__file__).resolve().parents[4] / "python" / "cudnn"
if _SRC_CUDNN.is_dir():
    if str(_SRC_CUDNN) not in cudnn.__path__:
        cudnn.__path__.append(str(_SRC_CUDNN))
    import cudnn.gemm

    _src_gemm = str(_SRC_CUDNN / "gemm")
    if _src_gemm not in cudnn.gemm.__path__:
        cudnn.gemm.__path__.append(_src_gemm)


@pytest.fixture(autouse=True)
def _frost_opt_in(monkeypatch):
    """The FROST manifest rows are opt-in; this suite exercises them.

    Per test rather than at import: a module-level ``os.environ[...] = "1"``
    runs during COLLECTION and would leave the flag set for every other test in
    the same pytest process, quietly turning a default-path run into an opt-in
    one."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


# A template family that does not run on the active GPU is a capability gap, not
# a defect. Every frost jit path declines it with the one message
# kernel_registry.KernelTemplate.arch_active_reject spells, so a test that pins
# a config of another family (an sm100 config on a consumer SM 12.x part, or the
# reverse) reports that message and nothing else. Report it as skipped, the way
# the arch markers do for the families a test knows to gate on, so a run on a
# part the config was never meant for reads as what it is.
_ARCH_DECLINE = re.compile(r"runs only on \d+ <= SM < \d+.* but the active GPU is sm_\d+")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when != "call" or not rep.failed or call.excinfo is None:
        return
    exc = call.excinfo.value
    if isinstance(exc, NotImplementedError) and _ARCH_DECLINE.search(str(exc)):
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1], f"SKIPPED: {exc}")
