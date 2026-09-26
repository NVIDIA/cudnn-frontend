# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 8 ratchet for ``sdpa/fwd/api_dsl.py`` (GPU-free).

The grandfathered ``_dummy`` cached-zeros sites are pinned by count so no new
caller lands before Part 3 retires them. They go away with their family's
kernel-side change, never with a workspace borrow: the fp8/mxfp8 SM100/SM107
sites through the fp8 pointer-ABI migration (dead slots bind ``0``, absent
amax/scales compile out), the SM120 and SM80 sites through ``Optional``-typing
the kernel slots (recipe R3(1)). Borrowing instead would flip a dense plan's
``get_workspace_size()`` from 0 -- tests pin 0 for dense fp8 and SM100 f16.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.L0

_API_DSL = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "sdpa" / "fwd" / "api_dsl.py"
_GRANDFATHERED_DUMMY_CALLERS = 23


@pytest.mark.no_workspace_shim
def test_no_new_dummy_callers():
    if not _API_DSL.is_file():
        pytest.skip(f"source tree not present: {_API_DSL}")
    count = len(re.findall(r"self\._dummy\(", _API_DSL.read_text()))
    assert count <= _GRANDFATHERED_DUMMY_CALLERS, (
        f"{count} `self._dummy(` callers in api_dsl.py, ratchet is {_GRANDFATHERED_DUMMY_CALLERS}: a dead ABI slot is "
        "compiled out (Optional kernel parameter + None fake), bound to 0 on a pointer ABI, or borrowed from a "
        "workspace the path already carves -- never a new cached torch.zeros dummy (AGENTS.md Rule 8, R3)"
    )
    assert (
        count == _GRANDFATHERED_DUMMY_CALLERS
    ), f"{_GRANDFATHERED_DUMMY_CALLERS - count} `_dummy` caller(s) retired -- lower _GRANDFATHERED_DUMMY_CALLERS to {count} so the ratchet stays tight"
