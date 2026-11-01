# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wide SM80 forward retains dense, paged and independent-origin contracts."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_sdpa_fwd_origins_sm80 import test_independent_origins_rebind_replay as _origins
from test_sdpa_fwd_paged_sm80 import test_native_paged_adapter_rebind_replay as _paged
from test_sdpa_fwd_single_q_sm80 import test_dense_single_q_rebind_replay as _single_q
from test_sdpa_sm80_padding_nan import test_runtime_padding_nan_rebind_replay as _padding

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]

_DIMS = [(512, 512), (576, 512), (288, 384), (520, 504)]
_POINTS = [
    (512, 512, 16, "HND", "NHD", "none"),
    (576, 512, 64, "NHD", "HND", "causal"),
    (288, 384, 128, "HND", "HND", "br_swa"),
    (520, 504, 16, "NHD", "NHD", "br_swa"),
]


@pytest.mark.parametrize("d,dv", _DIMS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_wide_dense_padding(d, dv, dtype, causal, cudnn_handle):
    _padding(d, dv, dtype, 33, 65, causal, cudnn_handle)


@pytest.mark.parametrize("d,dv", _DIMS[:2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["plain", "features", "o_only"])
def test_wide_single_q(d, dv, dtype, mode, cudnn_handle, monkeypatch):
    _single_q(d, dv, dtype, mode, cudnn_handle, monkeypatch)


@pytest.mark.parametrize("point", _POINTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_wide_paged_rebind_replay(point, dtype, thd, graph_mode, cudnn_handle, monkeypatch):
    _paged(point, dtype, thd, graph_mode, cudnn_handle, monkeypatch, case_index=8 + _POINTS.index(point))


@pytest.mark.parametrize("d,dv", _DIMS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("token_major", [False, True])
def test_wide_thd_origins(d, dv, dtype, token_major, cudnn_handle, monkeypatch):
    _origins(d, dv, dtype, token_major, cudnn_handle, monkeypatch)
