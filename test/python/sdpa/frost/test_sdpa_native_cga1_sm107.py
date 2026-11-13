# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native single-CTA THD prefill: storage rebinding, lengths and retained captures."""

from contextlib import nullcontext

import pytest
import torch

from frost_test_utils import requires_dsl, requires_rubin
from test_sdpa_prepared_thd import _check_rubin_ragged_pack_rebind, _check_thd_cache_shape_grid

pytestmark = [pytest.mark.L1, requires_dsl, requires_rubin]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stats_layout", ["token", "head", "padded"])
@pytest.mark.parametrize("stats_log2", [False, True])
@pytest.mark.parametrize("geometry", ["mixed", "partial12", "single_large", "single_unpacked", "two_wave_pair"])
def test_native_cga1_ragged_replay(dtype, stats_layout, stats_log2, geometry):
    _check_rubin_ragged_pack_rebind(dtype, stats_layout, stats_log2, geometry, cga=1)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("backend_lowering", [False, True])
@pytest.mark.parametrize("heads,workloads", [((16, 4), None), ((64, 8), ([1025], [769], [1025]))], ids=["ragged", "many_heads"])
def test_native_cga1_broad_declaration_replay(packed, backend_lowering, heads, workloads, monkeypatch):
    _check_thd_cache_shape_grid(128, "sm107", backend_lowering, monkeypatch, native_packed=packed, native_heads=heads, native_workloads=workloads, native_cga=1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("sched_policy", [0, 1, 2])
@pytest.mark.parametrize("causal,bottom_right,window_left", [(False, False, None), (True, True, None), (True, True, 17), (True, False, 32)])
def test_native_cga1_mask_and_scheduler_replay(dtype, packed, sched_policy, causal, bottom_right, window_left):
    # The graph row exposes NATURAL/LPT. LPT-L2 remains an explicit decline
    # even though the underlying template can compile that scheduler.
    expected = (
        pytest.raises(NotImplementedError, match=r"requested sched_policy=2 is outside this engine's domain \[0, 1\]") if sched_policy == 2 else nullcontext()
    )
    with expected:
        _check_rubin_ragged_pack_rebind(
            dtype,
            "padded",
            True,
            "partial12",
            cga=1,
            causal=causal,
            bottom_right=bottom_right,
            window_left=window_left,
            packed=packed,
            sched_policy=sched_policy,
        )
