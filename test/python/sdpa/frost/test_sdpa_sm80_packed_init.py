# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared initialization preserves the packed wrappers' zeroed capacity."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backward", [False, True])
def test_packed_wrapper_uses_prepared_initialization(dtype, backward, monkeypatch):
    if backward:
        from cudnn.sdpa.bwd import api_dsl
        from test_sdpa_sm80_thd_wrapper_prepared import test_wrapper_rebind_capture_and_capacity_tails as check

        name = "_sm80_thd_backward"
    else:
        from cudnn.sdpa.fwd import api_dsl
        from test_sdpa_sm80_thd_forward_prepared import test_thd_wrapper_rebind_and_capture as check

        name = "_sm80_thd_forward"
    original = getattr(api_dsl, name)

    def run(*args, **kwargs):
        with monkeypatch.context() as guard:
            for entry in ("zeros", "zeros_like"):
                guard.setattr(torch, entry, lambda *a, **k: pytest.fail("packed wrapper rebuilt Torch zero initialization"))
            return original(*args, **kwargs)

    monkeypatch.setattr(api_dsl, name, run)
    if backward:
        check(96, 96, dtype, monkeypatch)
    else:
        check(96, 96, dtype, True, monkeypatch)
