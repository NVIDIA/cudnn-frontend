# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect shared ordered-binding smoke cases in the FROST SDPA CI target."""

import cudnn
import pytest

import test_execute_overload as ordered_cases
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.fixture
def frost_attention_case(cudnn_handle):
    original_stream = cudnn.get_stream(cudnn_handle)
    try:
        graph, tensors = ordered_cases._graph(cudnn_handle, "frost")
        yield graph, tensors, cudnn_handle, "frost"
    finally:
        cudnn.set_stream(cudnn_handle, original_stream)


def test_frost_ordered_matches_mapping_with_fresh_buffers_and_uid_order(frost_attention_case):
    ordered_cases.test_ordered_matches_mapping_with_fresh_buffers_and_uid_order(frost_attention_case)


def test_frost_ordered_overrides_follow_mutable_values(frost_attention_case):
    ordered_cases.test_ordered_overrides_follow_mutable_values(frost_attention_case)
