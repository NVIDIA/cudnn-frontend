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


def _dsl_installed() -> bool:
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


requires_dsl = pytest.mark.skipif(not _dsl_installed(), reason="needs the cutedsl extra (nvidia-cutlass-dsl)")
