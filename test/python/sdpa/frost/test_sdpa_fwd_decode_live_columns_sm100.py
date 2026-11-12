# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged D128 swap-AB lowering with eight live query columns.

Exercise the public graph and assert the actual compiled geometry. The shared
D256 helper compares O and Stats against an independent dense reference and
checks that execution performs no blocking device-to-host reads.
"""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from test_sdpa_fwd_decode_d256_sm100 import DECODE, _run_graph

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]


@pytest.fixture
def live_column_route(monkeypatch):
    from cudnn.sdpa.fwd import api_dsl

    original = api_dsl._load_sm100_kernel_module
    observed = []

    def load(*args, **kwargs):
        mod = original(*args, **kwargs)
        observed.append(mod)
        return mod

    monkeypatch.setattr(api_dsl, "_load_sm100_kernel_module", load)
    yield
    assert observed, "the test did not compile a FROST kernel"
    assert any(
        mod.__file__.endswith("sm100/decode_d256_f16.py") and mod.CFG.TILE_K == mod.CFG.TILE_O == 128 and mod.N_Q == 16 and mod.COLS == 8 for mod in observed
    ), "the graph did not execute the D128 live-column route"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "sq,h,kh,d,page,hnd",
    [
        (1, 8, 1, 128, 8, True),
        (1, 16, 2, 128, 16, False),
        (1, 6, 2, 128, 32, True),
        (1, 4, 4, 128, 64, False),
        (2, 8, 2, 128, 16, True),
        (4, 4, 2, 128, 128, False),
        (8, 2, 2, 128, 16, True),
        (1, 8, 1, 80, 16, False),
    ],
    ids=["page8-g8", "nhd-g8", "g3", "mha", "mtp2-g4", "mtp4-g2", "mtp8-mha", "d80-envelope"],
)
def test_live_columns_graph(dtype, sq, h, kh, d, page, hnd, live_column_route):
    _run_graph(
        B=4, H=h, KH=kh, s_q=sq, lens=[513, 129, 1, 0], d=d, page=page, hnd=hnd, dtype=dtype, q_lens=[sq, max(sq - 1, 0), sq, 0], causal_br=True, expect=DECODE
    )


@pytest.mark.parametrize("feature", ["no_stats", "stats_log2", "sink", "window", "right_band", "deep_split"])
def test_live_columns_graph_features(feature, live_column_route):
    options = {
        "no_stats": dict(stats=False),
        "stats_log2": dict(stats_log2=True),
        "sink": dict(sink=-100.0),
        "window": dict(causal_br=True, window_left=73),
        "right_band": dict(window_right=3),
        "deep_split": dict(sink=True),
    }[feature]
    plan = _run_graph(
        B=3, H=8, KH=1, s_q=1, lens=[4096 if feature == "deep_split" else 513, 0, 129], d=128, page=16, dtype=torch.bfloat16, expect=DECODE, **options
    )
    if feature == "deep_split":
        assert plan.knobs.split_kv > 1


@pytest.mark.parametrize("sq,h,kh,page", [(1, 16, 1, 16), (9, 1, 1, 16), (1, 8, 1, 0)], ids=["g16", "q9", "dense"])
def test_live_columns_keep_existing_routes(sq, h, kh, page):
    _run_graph(B=2, H=h, KH=kh, s_q=sq, lens=[513, 129], d=128, page=page, expect="decode_d128_f16")
