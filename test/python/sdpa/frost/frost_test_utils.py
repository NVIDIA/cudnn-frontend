# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-condition markers for the FROST SDPA suites.

One gate for the suite, aligned with what the ENGINES declare
(``Capabilities.sm_lo``/``sm_hi`` in sdpa/fwd/engines.py) rather than
re-derived per file. Five files each carried their own copy pinned to exactly
(10, 0), so every one skipped on sm103 -- and would have on Rubin and Thor --
while the engines they test serve the whole line.
"""

import pytest


def _active_sm():
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


_SM = _active_sm()

requires_blackwell = pytest.mark.skipif(
    _SM is None or not (100 <= _SM <= 119),
    reason="needs an SM100-line GPU (100 <= SM <= 119), have " + ("none" if _SM is None else f"sm_{_SM}"),
)
requires_blackwell_geforce = pytest.mark.skipif(
    _SM is None or not (120 <= _SM <= 129),
    reason="needs an SM120-line GPU, have " + ("none" if _SM is None else f"sm_{_SM}"),
)


def _dsl_usable():
    """``(usable, why_not)`` for the DSL these engines lower through.

    Version too, not just presence: the extra is deliberately NOT pinned to the
    floor (that would make cudnn-frontend incompatible with anything holding the
    DSL back -- quack-kernels pins ==4.6.0), so an environment can legitimately
    have an older one. The engines decline it; these tests must skip rather than
    fail, and for the same reason.
    """
    from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed:
        return False, "needs the cutedsl extra (nvidia-cutlass-dsl)"
    if cutedsl_too_old(version):
        want = ".".join(str(v) for v in CUTEDSL_MIN_VERSION)
        return False, f"needs nvidia-cutlass-dsl >= {want}, have {version[1]}"
    return True, ""


_DSL_OK, _DSL_WHY = _dsl_usable()
requires_dsl = pytest.mark.skipif(not _DSL_OK, reason=_DSL_WHY or "cutedsl available")


def _dsl_installed() -> bool:
    """For the few call sites that gate inside a test body rather than on it."""
    return _DSL_OK
