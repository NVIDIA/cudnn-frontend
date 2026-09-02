# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation (decode / small-s_q forward) suites, bf16 — same seeds/knobs as
the fp16 siblings, so both dtypes sweep identical geometry."""

import pytest

from sdpa.suites.common import run_suite, suite_seeds


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("generation.bf16.decode"), ids=lambda p: f"test{p[0]}"
)
def test_generation_bf16_decode(env_info, test_no, request, cudnn_handle):
    run_suite("generation.bf16.decode", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("generation.bf16.lean"), ids=lambda p: f"test{p[0]}"
)
def test_generation_bf16_lean(env_info, test_no, request, cudnn_handle):
    run_suite("generation.bf16.lean", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("generation.bf16.thd_chunked"), ids=lambda p: f"test{p[0]}"
)
def test_generation_bf16_thd_chunked(env_info, test_no, request, cudnn_handle):
    run_suite("generation.bf16.thd_chunked", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("generation.bf16.paged"), ids=lambda p: f"test{p[0]}"
)
def test_generation_bf16_paged(env_info, test_no, request, cudnn_handle):
    run_suite("generation.bf16.paged", env_info, test_no, request, cudnn_handle)
