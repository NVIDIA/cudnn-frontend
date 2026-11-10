# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native dense-query D128 paging coverage for the SM107 CI entry."""

import pytest
import torch


@pytest.mark.parametrize(
    "cga,packed,split,tile_n",
    [
        (1, False, 1, 128),
        (2, False, 1, 128),
        (1, True, 1, 128),
        (2, True, 1, 128),
        (1, False, 4, 128),
        (2, False, 4, 128),
        (1, True, 4, 128),
        (2, True, 4, 128),
        (1, False, 1, 64),
        (2, False, 1, 64),
        (1, True, 1, 64),
        (2, True, 1, 64),
        (1, False, 4, 64),
        (2, False, 4, 64),
        (1, True, 4, 64),
        (2, True, 4, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_replay(cga, packed, split, tile_n, dtype, monkeypatch):
    import cudnn
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from test_sdpa_fwd_paged_sm100 import _run_graph
    from sm107_half_test_utils import _rubin_only

    _rubin_only()
    original = SdpaFwdDslSm100.compile
    seen = []

    def compile_native(api):
        original(api)
        assert api.flavor == (128, 128)
        assert api._k_mod.__file__.endswith("sm107/prefill_d128_f16.py")
        assert api._k_mod.CFG.PAGED_KV and not api._k_mod.CFG.THD_VARLEN
        assert api._k_mod.CFG.CTA_MMA == cga
        seen.append(True)

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    _run_graph(
        4,
        24,
        2,
        128,
        16,
        64,
        [1024, 257, 65, 0],
        cga == 2,
        dtype=dtype,
        stats=True,
        s_q=17,
        sink=-120.0,
        window_left=73,
        causal_br=True,
        pack_gqa=packed,
        v_hnd=cga != 2,
        replay=True,
        arch="sm107",
        explicit_knobs={
            cudnn.knob_type.TILE_CGA_M: cga,
            cudnn.knob_type.SPLIT_KV: split,
            cudnn.knob_type.PACK_GQA: int(packed),
            cudnn.knob_type.TILE_N: tile_n,
        },
    )
    assert seen


@pytest.mark.parametrize(
    "dims,page,sq,cga,split,packed,policy,empty",
    [
        ((128, 128), 8, 1, 1, 8, True, 0, False),
        ((128, 128), 256, 129, 2, 1, False, 1, False),
        ((96, 80), 32, 2, 1, 4, True, 0, False),
        ((64, 64), 64, 4, 2, 1, True, 1, False),
        ((128, 96), 128, 257, 2, 1, False, 0, False),
        ((128, 128), 8, 1, 2, 4, True, 0, True),
        ((128, 128), 1024, 2, 1, 1, False, 1, False),
        ((32, 32), 16, 1, 1, 1, False, 0, False),
    ],
)
def test_page_boundaries(dims, page, sq, cga, split, packed, policy, empty, monkeypatch):
    import cudnn
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from test_sdpa_fwd_paged_sm100 import _run_graph
    from sm107_half_test_utils import _rubin_only

    _rubin_only()
    original = SdpaFwdDslSm100.compile

    def compile_native(api):
        original(api)
        assert api.flavor == (128, 128)
        assert api._k_mod.__file__.endswith("sm107/prefill_d128_f16.py")
        assert api._k_mod.CFG.PAGED_KV and not api._k_mod.CFG.THD_VARLEN

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    _run_graph(
        4,
        24,
        2,
        dims[0],
        page,
        max(2, 1024 // page),
        [0] * 4 if empty else [1024, 257, 1, 0],
        cga == 2,
        dtype=torch.bfloat16,
        stats=True,
        s_q=sq,
        sink=False,
        window_left=73 if policy else None,
        causal_br=True,
        d_v=dims[1],
        pack_gqa=packed,
        v_hnd=cga != 2,
        replay=True,
        arch="sm107",
        explicit_knobs={
            cudnn.knob_type.TILE_CGA_M: cga,
            cudnn.knob_type.SPLIT_KV: split,
            cudnn.knob_type.PACK_GQA: int(packed),
            cudnn.knob_type.SCHED_POLICY: policy,
        },
    )
