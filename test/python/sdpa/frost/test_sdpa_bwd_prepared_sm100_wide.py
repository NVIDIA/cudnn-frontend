# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical Int64 addressing across every large-head backward stage."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from test_sdpa_bwd_dsl_sm100 import _prepared_case, _check_prepared

pytestmark = [pytest.mark.L1, pytest.mark.gpu_exclusive, requires_pre_rubin_blackwell, requires_dsl]


@pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "dq", "dk", "dv"])
@pytest.mark.parametrize("wide_product", [False, True], ids=["wide_stride", "wide_product"])
@pytest.mark.parametrize("hkv", [4, 2], ids=["mha", "gqa"])
def test_prepared_sm100_physical_batch_stride(role, wide_product, hkv):
    case = _prepared_case(hkv=hkv, wide=role, wide_product=wide_product)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            case.graph.execute(case.pack, case.workspace)
        for name in ("dq", "dk", "dv"):
            case.tensors[name].fill_(float("nan"))
        case.workspace.fill_(0xBD)
        capture.replay()
        _check_prepared(case)
    finally:
        capture.reset()
