# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation (decode / small-s_q forward) suites, f16 — the 16-bit family:
each config draws fp16 or bf16 (data_type fuzz), like fp8 draws e4m3/e5m2."""

import pytest

from sdpa.suites.common import run_suite, suite_seeds


@pytest.mark.L0
@pytest.mark.parametrize("test_no", suite_seeds("generation.f16.decode"), ids=lambda p: f"test{p[0]}")
def test_generation_f16_decode(env_info, test_no, request, cudnn_handle):
    run_suite("generation.f16.decode", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", suite_seeds("generation.f16.lean"), ids=lambda p: f"test{p[0]}")
def test_generation_f16_lean(env_info, test_no, request, cudnn_handle):
    run_suite("generation.f16.lean", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", suite_seeds("generation.f16.thd_chunked"), ids=lambda p: f"test{p[0]}")
def test_generation_f16_thd_chunked(env_info, test_no, request, cudnn_handle):
    run_suite("generation.f16.thd_chunked", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", suite_seeds("generation.f16.paged"), ids=lambda p: f"test{p[0]}")
def test_generation_f16_paged(env_info, test_no, request, cudnn_handle):
    run_suite("generation.f16.paged", env_info, test_no, request, cudnn_handle)
