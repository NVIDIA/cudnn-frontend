# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bprop (training fwd+bwd) suites, f16 — the 16-bit family: each config
draws fp16 or bf16 (data_type fuzz), like fp8 draws e4m3/e5m2."""

import pytest

from sdpa.suites.common import run_suite, suite_seeds


@pytest.mark.L0
@pytest.mark.parametrize("test_no", suite_seeds("bprop.f16.dense"), ids=lambda p: f"test{p[0]}")
def test_bprop_f16_dense(env_info, test_no, request, cudnn_handle):
    run_suite("bprop.f16.dense", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("test_no", suite_seeds("bprop.f16.thd"), ids=lambda p: f"test{p[0]}")
def test_bprop_f16_thd(env_info, test_no, request, cudnn_handle):
    run_suite("bprop.f16.thd", env_info, test_no, request, cudnn_handle)
