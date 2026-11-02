# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal Q-subtile tails with physical origins and retained graph replay."""

import pytest


@pytest.mark.parametrize(
    "dtype_name,tile_n,bottom_right,window_right,scale,growth,cga,packed",
    [
        pytest.param("bfloat16", 64, False, 0, None, 2.0, 2, False, id="n64-top-left-rescale"),
        pytest.param("float16", 128, False, 0, 1.0, 1.0, 2, False, id="n128-top-left-unit-scale"),
        pytest.param("float16", 64, True, 0, None, 1.5, 2, False, id="n64-bottom-right-rescale"),
        pytest.param("bfloat16", 128, True, 17, 0.125, 1.0, 2, False, id="n128-right-band"),
        pytest.param("bfloat16", 64, False, 0, None, 1.0, 1, False, id="single-cta-control"),
        pytest.param("float16", 64, True, 0, None, 1.0, 2, True, id="packed-control"),
    ],
)
def test_causal_tail_origins_and_replay(dtype_name, tile_n, bottom_right, window_right, scale, growth, cga, packed):
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture

    test_paged_thd_origins_rebind_and_capture(
        128,
        dtype_name,
        "mixed",
        "head" if bottom_right else "token",
        cga,
        1,
        packed,
        page_size=16,
        tile_n=tile_n,
        sched_policy=1,
        cpu_reference=True,
        attention_scale=scale,
        bottom_right=bottom_right,
        window_right=window_right,
        kv_bound=512,
        key_growth=growth,
    )
