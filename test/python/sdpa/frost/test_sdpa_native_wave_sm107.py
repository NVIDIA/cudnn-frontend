# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native initial-wave policy under broad shape declarations and changed replay."""

import pytest
import torch

from frost_test_utils import requires_dsl
from test_sdpa_prepared_thd import _check_thd_cache_shape_grid

pytestmark = [pytest.mark.L1, requires_dsl]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("backend_lowering", [False, True])
def test_native_initial_waves_bind_current_work(packed, backend_lowering, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("native SM107 initial-wave policy")
    _check_thd_cache_shape_grid(128, "sm107", backend_lowering, monkeypatch, native_packed=packed)
