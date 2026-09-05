# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bprop (training fwd+bwd) suites, mxfp8 (block-scaled fp8; SM100+)."""

import pytest

from sdpa.suites.common import run_suite, suite_seeds


@pytest.mark.L0
@pytest.mark.parametrize("test_no", suite_seeds("bprop.mxfp8.dense"), ids=lambda p: f"test{p[0]}")
def test_bprop_mxfp8_dense(env_info, test_no, request, cudnn_handle):
    run_suite("bprop.mxfp8.dense", env_info, test_no, request, cudnn_handle)
