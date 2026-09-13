# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM80 SDPA wrappers decline a too-old CuTe DSL by version (AGENTS.md Rule 7).

Host-only: the wrapper entry is an independent caller-facing path to the kernel
modules (no engine check_support in front of it), so it must raise an error
that names the installed version *before* importing a kernel module, where a
4.5.x wheel would otherwise fail with the DSL's own TypeError.
"""

import pytest
import torch

from cudnn.frost import buffers

pytestmark = pytest.mark.L0


@pytest.fixture
def old_dsl(monkeypatch):
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))


def test_bwd_wrapper_declines_by_version(old_dsl):
    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_sm80

    t = torch.zeros(1, 1, 8, 16, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match=r"found 4\.6\.2"):
        sdpa_bwd_wrapper_sm80(q_tensor=t, k_tensor=t, v_tensor=t, o_tensor=t, do_tensor=t, lse_tensor=torch.zeros(1, 1, 8, 1))


def test_fwd_wrapper_declines_by_version(old_dsl):
    from cudnn.sdpa.fwd.api_dsl import sdpa_fwd_wrapper_sm80

    t = torch.zeros(1, 1, 8, 16, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match=r"found 4\.6\.2"):
        sdpa_fwd_wrapper_sm80(t, t, t)
