# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation (decode / small-s_q forward) suites, fp16."""

import pytest

from sdpa.suites.common import run_suite, suite_seeds


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("generation.fp16.decode"), ids=lambda p: f"test{p[0]}"
)
def test_generation_fp16_decode(env_info, test_no, request, cudnn_handle):
    run_suite("generation.fp16.decode", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("generation.fp16.lean"), ids=lambda p: f"test{p[0]}"
)
def test_generation_fp16_lean(env_info, test_no, request, cudnn_handle):
    run_suite("generation.fp16.lean", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("generation.fp16.thd_chunked"), ids=lambda p: f"test{p[0]}"
)
def test_generation_fp16_thd_chunked(env_info, test_no, request, cudnn_handle):
    run_suite("generation.fp16.thd_chunked", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("generation.fp16.paged"), ids=lambda p: f"test{p[0]}"
)
def test_generation_fp16_paged(env_info, test_no, request, cudnn_handle):
    run_suite("generation.fp16.paged", env_info, test_no, request, cudnn_handle)
