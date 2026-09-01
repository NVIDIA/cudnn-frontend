# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation (decode / paged) suites, fp8 (e4m3/e5m2)."""

import pytest

from sdpa.suites.common import run_suite, suite_seeds


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("generation.fp8.decode"), ids=lambda p: f"test{p[0]}"
)
def test_generation_fp8_decode(env_info, test_no, request, cudnn_handle):
    run_suite("generation.fp8.decode", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("generation.fp8.paged"), ids=lambda p: f"test{p[0]}"
)
def test_generation_fp8_paged(env_info, test_no, request, cudnn_handle):
    run_suite("generation.fp8.paged", env_info, test_no, request, cudnn_handle)
