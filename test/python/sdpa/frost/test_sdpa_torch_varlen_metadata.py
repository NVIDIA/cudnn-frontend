# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect the shared torch metadata producer in the FROST SDPA CI gate."""

import pytest

import sdpa.torch.test_varlen_metadata as shared_cases
import sdpa.torch.test_aux_metadata as auxiliary_cases
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.mark.parametrize("return_lse", [False, True])
@requires_pre_rubin_blackwell
def test_frost_torch_varlen_metadata_rebind_and_replay(return_lse, monkeypatch):
    shared_cases.test_shared_metadata_serves_both_forward_providers("frost", return_lse, monkeypatch)


@pytest.mark.parametrize("role", ["lengths", "sinks"])
@pytest.mark.parametrize("strided", [False, True])
def test_frost_torch_auxiliary_metadata_rebind_and_replay(role, strided, monkeypatch):
    auxiliary_cases.test_forward_auxiliary_storage_matches_graph("frost", role, strided, monkeypatch)
