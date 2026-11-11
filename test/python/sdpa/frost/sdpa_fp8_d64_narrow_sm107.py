# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native per-tensor D64 narrow KV tiles, collected by the SM107 CI entry."""

import pytest
import torch

from test_sdpa_sm107_native_d64_fp8 import _CASES, test_native_d64_fp8_rebind_replay, test_standalone_d64_padded_default_tile_n


@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize("in_key,out_dtype,mode,d,packed,splits,scheduler", _CASES + [("e4m3", torch.bfloat16, "split_sink", 64, True, 5, 1)])
def narrow_rebind_replay(in_key, out_dtype, mode, d, packed, splits, scheduler, cga, cudnn_handle, monkeypatch):
    test_native_d64_fp8_rebind_replay(in_key, out_dtype, mode, d, packed, splits, scheduler, cga, cudnn_handle, monkeypatch, tile_n=64)


@pytest.mark.parametrize("sk", [129, 192, 257])
@pytest.mark.parametrize("in_key,cga", [("e4m3", 1), ("e5m2", 2)])
def narrow_standalone(in_key, cga, sk):
    test_standalone_d64_padded_default_tile_n(in_key, cga, sk, tile_n=64)
