# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native half D576 paging on the SM107 2x2 pipeline."""

import pytest
import torch

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed,split,layout", [(False, 1, "mixed"), (True, 1, "hnd"), (False, 4, "nhd"), (True, 4, "mixed")])
def test_dense(dtype, packed, split, layout, monkeypatch):
    import cudnn
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    import test_sdpa_fwd_paged_sm100 as paged
    from sm107_half_test_utils import _rubin_only

    _rubin_only()
    original = SdpaFwdDslSm100.compile
    seen = []

    def compile_native(api):
        original(api)
        assert api.flavor == (576, 512)
        assert api._k_mod.__file__.endswith("sm107/prefill_d512_f16_2x2.py")
        assert api._k_mod.CFG.PAGED_KV and api._k_mod.CFG.KV_SHARE == 1
        assert api._k_mod.CFG.N_BMM1_CHUNKS == 3
        assert api._k_mod.DESC_VERSION == 1
        seen.append(True)

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    paged._run_graph(
        4,
        24,
        2,
        576,
        16,
        64,
        [1024, 257, 65, 0],
        layout != "nhd",
        dtype=dtype,
        d_v=512,
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

    test_replay_batch_and_golden(dtype_name, dtype_name, 2, split, 128, head_dim=576, value_dim=512)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("folded,split", [(False, 1), (True, 1), (False, 4), (True, 4)])
def test_nonpaged(dtype, folded, split):
    import math
    from sm107_half_test_utils import _half_prefold_api, _half_oracle, _rubin_only
    from determinism_utils import assert_bitwise, snapshot

    _rubin_only()
    api, t = _half_prefold_api(
        576,
        512,
        dtype,
        prefolded=folded,
        with_stats=True,
        split_kv=split,
        b=2,
        hq=12,
        hkv=1,
        s_q=129,
        s_kv=256,
        causal=True,
        bottom_right=True,
        window_left=73,
        sink=True,
        pack_gqa=True,
        seed=7231,
    )
    api.check_support()
    api.compile()
    assert api._k_mod.__file__.endswith("sm107/prefill_d512_f16_2x2.py")
    assert api._k_mod.CFG.TILE_K == 576 and api._k_mod.CFG.N_BMM1_CHUNKS == 3
    assert api._k_mod.CFG.CGA_M == 2 and not api._k_mod.CFG.PAGED_KV
    ws = torch.empty(max(1, api.scratch_workspace_bytes()), device="cuda", dtype=torch.uint8)

    def run():
        api.execute(t["q"], t["k"], t["v"], t["out"], t["lse"], sinks=t["sinks"], workspace=ws)

    def check():
        qi = torch.arange(129)[:, None] + 256 - 129
        ki = torch.arange(256)[None, :]
        expected, stats = _half_oracle(
            t["q"].cpu(),
            t["k"].cpu(),
            t["v"].cpu(),
            logit_scale=math.log(2) if folded else t["attn_scale"],
            masked=(ki > qi) | (ki < qi - 73),
            sinks=t["sinks"].cpu(),
        )
        torch.testing.assert_close(t["out"].cpu().double(), expected, atol=0.005, rtol=0.02)
        torch.testing.assert_close(t["lse"].cpu().double(), stats, atol=0.0006, rtol=0)

    run()
    check()
    old = snapshot({"o": t["out"], "lse": t["lse"]})
    run()
    assert_bitwise({"o": t["out"], "lse": t["lse"]}, old, "fixed-input replay")
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            torch.cuda.set_sync_debug_mode("error")
            try:
                run()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        t["q"].mul_(0.5)
        t["out"].fill_(torch.nan)
        t["lse"].fill_(torch.nan)
        graph.replay()
        check()
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize(
    "page,sq,split,packed,empty",
    [
        (8, 1, 4, True, False),
        (32, 2, 1, False, False),
        (64, 129, 1, True, False),
        (128, 257, 1, False, False),
        (256, 17, 4, True, False),
        (1024, 2, 1, False, False),
        (8, 1, 4, True, True),
    ],
)
def test_boundaries(page, sq, split, packed, empty, monkeypatch):
    import cudnn
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    import test_sdpa_fwd_paged_sm100 as paged
    from sm107_half_test_utils import _rubin_only

    _rubin_only()
    original = SdpaFwdDslSm100.compile

    def compile_native(api):
        original(api)
        assert api.flavor == (576, 512) and api._k_mod.CFG.N_BMM1_CHUNKS == 3
        assert api._k_mod.__file__.endswith("sm107/prefill_d512_f16_2x2.py")

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    paged._run_graph(
        4,
        24,
        2,
        576,
        page,
        max(2, 1024 // page),
        [0] * 4 if empty else [1024, 257, 1, 0],
        True,
        d_v=512,
        dtype=torch.bfloat16,
        stats=True,
        s_q=sq,
        sink=-120.0,
        window_left=73,
        causal_br=True,
        pack_gqa=packed,
        v_hnd=False,
        replay=True,
        arch="sm107",
        explicit_knobs={cudnn.knob_type.TILE_CGA_M: 2, cudnn.knob_type.SPLIT_KV: split, cudnn.knob_type.PACK_GQA: int(packed), cudnn.knob_type.SCHED_POLICY: 0},
    )


def test_admission():
    from test_sdpa_fwd_dsl_sm107 import _caps, _f16_facts
    from cudnn.sdpa.fwd import engines, config_sm107
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    import cudnn

    caps = _caps("sdpa_fwd_prefill_sm107")
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def desc(width):
        return graph.tensor(dim=(2, 8, 512, width), stride=(512 * 8 * width, width, 8 * width, 1), data_type=cudnn.data_type.HALF)

    operands = dict(q_t=desc(576), k_t=desc(576), v_t=desc(512), o_t=desc(512))
    for paged in (False, True):
        for thd in (False, True):
            for cga in (1, 2):
                for pack in (False, True):
                    for split in (1, 4):
                        facts = _f16_facts(d_qk=576, d_v=512, thd=thd, padded=True, has_paged_kv=paged, page_size=16 if paged else 0, **operands)
                        knobs = engines.SdpaFwdKnobs(cga=cga, pack_gqa=pack, split_kv=split)
                        why = engines.mismatch(caps, facts, knobs)
                        assert (why is None) == (not thd and cga == 2), (paged, thd, knobs, why)
    params = TemplateParams(d_flavor=576, mma_2x2=True, cta_mma=2)
    cfg, _ = config_sm107.make_cfg_d512(params)
    assert cfg.TILE_K == 576 and cfg.N_BMM1_CHUNKS == 3 and cfg.CGA_M == 2 and cfg.KV_SHARE == 1
    assert cfg.DESC_VERSION == 1
    with pytest.raises(ValueError, match="three K chunks and one CTA pair"):
        config_sm107.make_cfg_d512_2x2(params, cga_m=4)


@pytest.mark.parametrize("dtype,heads,sq,hnd", [(torch.bfloat16, 128, 1, True), (torch.float16, 96, 2, False), (torch.bfloat16, 128, 16, False)])
def test_shared_mla_record(dtype, heads, sq, hnd, monkeypatch):
    """Actual absorbed MLA: V aliases the first512 columns of each576-wide K row."""
    import test_sdpa_fwd_paged_sm100 as paged
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from sm107_half_test_utils import _rubin_only

    _rubin_only()
    original_pools = paged._pools
    original_compile = SdpaFwdDslSm100.compile

    def pools(*args, **kwargs):
        k, _, kc, _, table = original_pools(*args, **kwargs)
        return k, k[..., :512], kc, kc[..., :512], table

    def compile_native(api):
        original_compile(api)
        assert api.flavor == (576, 512)
        assert api._k_mod.__file__.endswith("sm107/prefill_d512_f16_2x2.py")
        assert api._k_mod.CFG.N_BMM1_CHUNKS == 3

    monkeypatch.setattr(paged, "_pools", pools)
    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_native)
    paged._run_graph(
        4,
        heads,
        1,
        576,
        64,
        16,
        [1024, 257, 1, 0],
        hnd,
        dtype=dtype,
        d_v=512,
        stats=True,
        s_q=sq,
        causal_br=True,
        pack_gqa=True,
        v_hnd=hnd,
        replay=True,
        arch="sm107",
    )
