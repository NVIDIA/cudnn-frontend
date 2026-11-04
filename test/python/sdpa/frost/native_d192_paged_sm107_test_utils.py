# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SM107 D192/V128 paged regression cells for the architecture CI entrypoint."""

import pytest
import torch


@pytest.mark.parametrize("dims", [(192, 128), (160, 96)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("hnd,page,stats", [(False, 16, True), (True, 64, False)])
def test_dense_replay(dtype, packed, hnd, page, stats, dims, monkeypatch):
    import cudnn
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from test_sdpa_fwd_paged_sm100 import _run_graph

    compile_plan = SdpaFwdDslSm100.compile
    seen = []

    def compile_native(api):
        compile_plan(api)
        assert api.flavor == (192, 128)
        assert api._k_mod.__file__.endswith("sm107/prefill_d128_f16.py")
        assert (api._k_mod.CFG.TILE_K, api._k_mod.CFG.TILE_O) == (192, 128)
        assert api._k_mod.CFG.STAGES_KV == 2 and api._k_mod.CFG.CTA_MMA == 2
        seen.append(api.flavor)

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    _run_graph(
        4,
        24,
        2,
        dims[0],
        page,
        1024 // page,
        [1024, 257, 65, 0],
        hnd,
        dtype=dtype,
        stats=stats,
        s_q=129,
        d_v=dims[1],
        sink=True,
        window_left=73 if stats else None,
        causal_br=True,
        pack_gqa=packed,
        v_hnd=not hnd,
        replay=True,
        arch="sm107",
        explicit_knobs={cudnn.knob_type.TILE_CGA_M: 2, cudnn.knob_type.SPLIT_KV: 1, cudnn.knob_type.PACK_GQA: int(packed)},
    )
    assert seen


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("pool_layout,page,stats", [("mixed", 16, "token"), ("hnd", 64, "head")])
def test_independent_origins(dtype_name, packed, pool_layout, page, stats):
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture

    test_paged_thd_origins_rebind_and_capture(
        (192, 128),
        dtype_name,
        pool_layout,
        stats,
        2,
        1,
        packed,
        page_size=page,
        window_left=47 if stats == "head" else None,
        sink=-120.0,
    )
