# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context (prefill forward) suites, fp8 (e4m3/e5m2)."""

import pytest

from sdpa.suites.common import run_suite, suite_seeds


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("context.fp8.dense"), ids=lambda p: f"test{p[0]}"
)
def test_context_fp8_dense(env_info, test_no, request, cudnn_handle):
    run_suite("context.fp8.dense", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("context.fp8.thd"), ids=lambda p: f"test{p[0]}"
)
def test_context_fp8_thd(env_info, test_no, request, cudnn_handle):
    run_suite("context.fp8.thd", env_info, test_no, request, cudnn_handle)
