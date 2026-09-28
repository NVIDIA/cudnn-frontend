# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect the shared torch metadata producer in the FROST SDPA CI gate."""

import pytest

import sdpa.torch.test_varlen_metadata as shared_cases
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("return_lse", [False, True])
def test_frost_torch_varlen_metadata_rebind_and_replay(return_lse, monkeypatch):
    shared_cases.test_shared_metadata_serves_both_forward_providers("frost", return_lse, monkeypatch)
