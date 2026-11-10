# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native half D512 paging on the SM107 2x2 pipeline."""

import pytest
import torch

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed,split,layout", [(False, 1, "mixed"), (True, 1, "hnd"), (False, 4, "nhd"), (True, 4, "mixed")])
def test_dense(dtype, packed, split, layout, monkeypatch):
    import cudnn
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from test_sdpa_fwd_paged_sm100 import _run_graph
    from sm107_half_test_utils import _rubin_only

    _rubin_only()
    original = SdpaFwdDslSm100.compile
    seen = []

    def compile_native(api):
        original(api)
        assert api.flavor == (512, 512)
        assert api._k_mod.__file__.endswith("sm107/prefill_d512_f16_2x2.py")
        assert api._k_mod.CFG.PAGED_KV and api._k_mod.CFG.KV_SHARE == 2
        assert api._k_mod.DESC_VERSION == 1
        seen.append(True)

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    _run_graph(
        4,
        24,
        2,
        512,
        16,
        64,
        [1024, 257, 65, 0],
        layout != "nhd",
        dtype=dtype,
        stats=True,
        s_q=17,
        sink=-120.0,
        window_left=73,
        causal_br=True,
        pack_gqa=packed,
        v_hnd=layout == "hnd",
        replay=True,
        arch="sm107",
        explicit_knobs={cudnn.knob_type.TILE_CGA_M: 2, cudnn.knob_type.SPLIT_KV: split, cudnn.knob_type.PACK_GQA: int(packed)},
    )
    assert seen


@pytest.mark.parametrize("dtype_name,split", [("float16", 1), ("bfloat16", 1), ("float16", 4), ("bfloat16", 4)])
def test_determinism(dtype_name, split):
    from native_d128_dense_paged_determinism import test_replay_batch_and_golden

    test_replay_batch_and_golden(dtype_name, dtype_name, 2, split, 128, head_dim=512)


@pytest.mark.parametrize(
    "dims,page,sq,split,packed,empty",
    [
        ((512, 512), 8, 1, 4, True, False),
        ((512, 512), 64, 2, 1, False, False),
        ((512, 512), 128, 129, 1, True, False),
        ((512, 512), 256, 257, 1, False, False),
        ((512, 512), 1024, 2, 4, True, False),
        ((512, 512), 8, 1, 4, True, True),
        ((504, 480), 32, 2, 1, False, False),
        ((264, 256), 16, 4, 4, True, False),
    ],
)
def test_boundaries(dims, page, sq, split, packed, empty, monkeypatch):
    import cudnn
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from sm107_half_test_utils import _rubin_only
    from test_sdpa_fwd_paged_sm100 import _run_graph

    _rubin_only()
    original = SdpaFwdDslSm100.compile

    def compile_native(api):
        original(api)
        assert api.flavor == (512, 512)
        assert api._k_mod.__file__.endswith("sm107/prefill_d512_f16_2x2.py")
        assert api._k_mod.CFG.PAGED_KV and api._k_mod.CFG.KV_SHARE == 2

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    _run_graph(
        4,
        24,
        2,
        dims[0],
        page,
        max(2, 1024 // page),
        [0] * 4 if empty else [1024, 257, 1, 0],
        True,
        dtype=torch.bfloat16,
        stats=True,
        s_q=sq,
        sink=False,
        window_left=73,
        causal_br=True,
        d_v=dims[1],
        pack_gqa=packed,
        v_hnd=False,
        replay=True,
        arch="sm107",
        explicit_knobs={cudnn.knob_type.TILE_CGA_M: 2, cudnn.knob_type.SPLIT_KV: split, cudnn.knob_type.PACK_GQA: int(packed), cudnn.knob_type.SCHED_POLICY: 0},
    )


def test_admission_and_config_boundaries():
    from test_sdpa_fwd_dsl_sm107 import _caps, _f16_facts
    from cudnn.sdpa.fwd import engines, config_sm100, config_sm107

    caps = _caps("sdpa_fwd_prefill_sm107")
    for thd in (False, True):
        for cga in (1, 2):
            for pack in (False, True):
                for split in (1, 4):
                    facts = _f16_facts(d_qk=512, d_v=512, thd=thd, padded=True, has_paged_kv=True, page_size=16)
                    knobs = engines.SdpaFwdKnobs(cga=cga, pack_gqa=pack, split_kv=split)
                    why = engines.mismatch(caps, facts, knobs)
                    assert (why is None) == (not thd and cga == 2), (thd, knobs, why)
    params = config_sm100.TemplateParams(dtype_qkv=2, dtype_o=2, cta_mma=2, mma_2x2=True, paged_kv=True, page_size=16, seq_kv_lens_present=True)
    cfg, _ = config_sm107.make_cfg_d512_2x2(params)
    assert cfg.PAGED_KV and cfg.PAGE_SIZE == 16 and cfg.KV_SHARE == 2
    # Native SM107 admission must not expose SM100's still-unwired D512 2x2 arm.
    with pytest.raises(ValueError, match="mma_2x2 does not serve paged"):
        config_sm100.make_cfg_d512_2x2(params)
