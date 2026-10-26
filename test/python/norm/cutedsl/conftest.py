# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FROST norm kernels lower through Hopper-and-later instructions (mbarrier
transaction counts, bulk async copies, thread-block clusters), so their native
tests run on SM90 or newer only. On an older GPU each test skips instead of
failing to compile."""

import pytest
import torch

_MIN_CC = (9, 0)


@pytest.fixture(autouse=True)
def _frost_norm_kernels_need_sm90():
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc < _MIN_CC:
            pytest.skip(f"FROST norm kernels need SM90 or newer (mbarrier transaction counts, bulk async copies); have sm_{cc[0]}{cc[1]}")
