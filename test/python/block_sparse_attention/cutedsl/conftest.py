# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture(autouse=True)
def _isolated_rng():
    """Restore the CPU and CUDA generators after each test, so a test that seeds them does not leak its state."""
    with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
        yield


@pytest.fixture(autouse=True)
def _fp32_reference_matmul():
    """Run the torch references with full FP32 matmuls (``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`` turns TF32 on by default).

    The flag is restored by hand: ``torch.backends.cuda.matmul`` serves ``allow_tf32`` through ``__getattr__`` /
    ``__setattr__``, so since pytest 9.2 ``monkeypatch.setattr`` finds no instance value to restore and its teardown
    deletes the attribute.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous
