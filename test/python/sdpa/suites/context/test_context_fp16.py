# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context (prefill forward) suites, fp16. See sdpa/suites/registry.py for the
master list of what each suite fuzzes."""

import pytest

from sdpa.suites.common import run_suite, suite_seeds


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("context.fp16.dense"), ids=lambda p: f"test{p[0]}"
)
def test_context_fp16_dense(env_info, test_no, request, cudnn_handle):
    run_suite("context.fp16.dense", env_info, test_no, request, cudnn_handle)


@pytest.mark.L1
@pytest.mark.parametrize(
    "test_no", suite_seeds("context.fp16.unified"), ids=lambda p: f"test{p[0]}"
)
def test_context_fp16_unified(env_info, test_no, request, cudnn_handle):
    run_suite("context.fp16.unified", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("context.fp16.thd"), ids=lambda p: f"test{p[0]}"
)
def test_context_fp16_thd(env_info, test_no, request, cudnn_handle):
    run_suite("context.fp16.thd", env_info, test_no, request, cudnn_handle)


@pytest.mark.L1
@pytest.mark.parametrize(
    "test_no", suite_seeds("context.fp16.thd_unified"), ids=lambda p: f"test{p[0]}"
)
def test_context_fp16_thd_unified(env_info, test_no, request, cudnn_handle):
    run_suite("context.fp16.thd_unified", env_info, test_no, request, cudnn_handle)


@pytest.mark.L1
@pytest.mark.parametrize(
    "test_no", suite_seeds("context.fp16.thd_offset_mult"), ids=lambda p: f"test{p[0]}"
)
def test_context_fp16_thd_offset_mult(env_info, test_no, request, cudnn_handle):
    run_suite("context.fp16.thd_offset_mult", env_info, test_no, request, cudnn_handle)
