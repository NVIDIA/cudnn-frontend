# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bprop (training fwd+bwd) suites, bf16 — same seeds/knobs as the fp16
siblings, so both dtypes sweep identical geometry."""

import pytest

from sdpa.suites.common import run_suite, suite_seeds


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("bprop.bf16.dense"), ids=lambda p: f"test{p[0]}"
)
def test_bprop_bf16_dense(env_info, test_no, request, cudnn_handle):
    run_suite("bprop.bf16.dense", env_info, test_no, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize(
    "test_no", suite_seeds("bprop.bf16.thd"), ids=lambda p: f"test{p[0]}"
)
def test_bprop_bf16_thd(env_info, test_no, request, cudnn_handle):
    run_suite("bprop.bf16.thd", env_info, test_no, request, cudnn_handle)
