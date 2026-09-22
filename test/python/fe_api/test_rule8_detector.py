# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The R9 detector itself (``fe_api/conftest.py``): an ``APIBase`` subclass defined AFTER the
conftest imported -- the shape of every test body's lazy ``from cudnn import X`` -- is guarded
at class creation, so a host sync inside its ``execute()`` raises regardless of which test
imported the API first; ``allow_host_sync`` disarms it."""

import pytest
import torch

from cudnn.api_base import APIBase


def _late_api():
    class LateAPI(APIBase):
        def __init__(self):  # no samples: the detector only cares about execute()
            pass

        def check_support(self) -> bool:
            return True

        def compile(self) -> None:
            pass

        def execute(self):
            return torch.ones(1, device="cuda").item()  # D2H read: the sync the detector must catch

    return LateAPI()


@pytest.mark.L0
def test_sync_in_execute_of_a_late_defined_api_is_caught():
    if not torch.cuda.is_available():
        pytest.skip("Environment not supported: CUDA device required")
    with pytest.raises(RuntimeError, match="synchronizing"):
        _late_api().execute()


@pytest.mark.L0
@pytest.mark.allow_host_sync
def test_allow_host_sync_disarms_the_detector():
    if not torch.cuda.is_available():
        pytest.skip("Environment not supported: CUDA device required")
    assert _late_api().execute() == 1.0
