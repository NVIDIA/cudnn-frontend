# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect the shared graph contract in the SM120 FROST CI lane."""

import pytest

from sdpa.graph.test_mhas_v2 import _PHYSICAL_ORIGIN_CASES, _exec_physical_origin_case, _physical_origin_cfg
from frost_test_utils import requires_blackwell_geforce, requires_dsl
import frost_routing

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


@pytest.mark.parametrize("dtype_name,width", _PHYSICAL_ORIGIN_CASES)
def test_shared_physical_origins(dtype_name, width, request, cudnn_handle):
    _check_native_origins(dtype_name, width, request, cudnn_handle)


@pytest.mark.parametrize("dtype_name,width", [case for case in _PHYSICAL_ORIGIN_CASES if case[0].startswith("float8_")])
def test_shared_physical_origins_mxfp8(dtype_name, width, request, cudnn_handle):
    _check_native_origins(dtype_name, width, request, cudnn_handle, mxfp8=True)


def _check_native_origins(dtype_name, width, request, cudnn_handle, *, mxfp8=False, engine=None):
    cfg = _physical_origin_cfg(dtype_name, width)
    if engine is None:
        engine = "sdpa_fwd_prefill_sm120_mxfp8" if mxfp8 else "sdpa_fwd_prefill_sm120_fp8" if cfg.data_type.itemsize == 1 else "sdpa_fwd_prefill_sm120"
    cfg.plan_pin = {"engine": engine, "knobs": {"PACK_GQA": 0, "SPLIT_KV": 1}}
    key = "frost:" + engine
    before = frost_routing.snapshot().get(key, 0)
    try:
        _exec_physical_origin_case(cfg, request, cudnn_handle, mxfp8=mxfp8)
    except pytest.skip.Exception as error:
        if request.config.option.dryrun:
            raise
        pytest.fail(f"physical-origin regression must execute {engine}: {error}")
    assert frost_routing.snapshot().get(key, 0) == before + 1
