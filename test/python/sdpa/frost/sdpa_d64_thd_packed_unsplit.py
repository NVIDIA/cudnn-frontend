# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""D64 packed THD prefill writes final O/Stats without split scratch."""

from dataclasses import replace

import pytest
import torch


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize("paged", [False, True])
def test_contract(cc, cga, paged):
    import cudnn
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    facts = SdpaGraphFacts(
        b=2,
        h_q=8,
        h_kv=2,
        s_q=129,
        s_kv=2048,
        d_qk=64,
        d_v=64,
        dtype=cudnn.data_type.BFLOAT16,
        thd=True,
        padded=True,
        has_paged_kv=paged,
        page_size=16 if paged else 0,
        device_cc=cc,
        device_sm_count=148,
    )
    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")
    knobs = engines.SdpaFwdKnobs(cga=cga, split_kv=1, pack_gqa=True, tile_n=128, sched_policy=0)
    assert engines.mismatch(caps, facts, knobs) is None, engines.mismatch(caps, facts, knobs)
    assert engines.mismatch(caps, facts, replace(knobs, tile_n=64)) is not None
    assert engines.mismatch(caps, replace(facts, h_q=12), knobs) is not None  # whole-group packing only
    params = TemplateParams(
        dtype_qkv=2,
        dtype_o=2,
        d_flavor=64,
        cta_mma=cga,
        split_kv=1,
        thd_varlen=True,
        pack_gqa=True,
        qh_per_kh=4,
        seq_kv_lens_present=True,
        paged_kv=paged,
        page_size=16 if paged else 0,
    )
    module = _load_sm100_kernel_module((64, 64), params, part=cc)
    assert module.__file__.endswith("sm100/prefill_d128_f16.py")
    assert module.CFG.TILE_K == module.CFG.TILE_O == 64
    assert module.CFG.PACK_G == 4 and module.CFG.THD_VARLEN
    assert module.CFG.CTA_MMA == cga and not module._FP32_PARTIALS


def _assert_direct_body(monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    original = SdpaFwdDslSm100.compile
    seen = []

    def compile_direct(api):
        original(api)
        assert api._k_mod.__file__.endswith("sm100/prefill_d128_f16.py")
        assert api._k_mod.CFG.TILE_K == api._k_mod.CFG.TILE_O == 64
        assert api._k_mod.CFG.PACK_GQA and api._k_mod.CFG.THD_VARLEN
        assert not api._k_mod._FP32_PARTIALS and not api.packed_thd_split
        assert api._thd_spec.native is not None and api._thd_spec.origin_specs is not None
        assert api._thd_spec.split_workspace is None
        seen.append(api)

    monkeypatch.setattr(SdpaFwdDslSm100, "compile", compile_direct)
    return seen


@pytest.mark.parametrize(
    "dtype,cga,layout,stats,page,heads,scheduler,q_bound",
    [
        ("bfloat16", 1, "mixed", "token", 16, 8, 0, None),
        ("float16", 2, "nhd", "head", 64, 16, 1, None),
        ("float16", 1, "hnd", "token", 8, 32, 2, 64),
        ("bfloat16", 2, "mixed", "head", 256, 64, 0, 64),
    ],
    ids=["bf16-cga1-g2-page16", "fp16-cga2-g4-page64", "fp16-cga1-g8-short-page8", "bf16-cga2-g16-short-page256"],
)
def test_paged_origins(dtype, cga, layout, stats, page, heads, scheduler, q_bound, monkeypatch):
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture as run

    seen = _assert_direct_body(monkeypatch)
    run(
        64,
        dtype,
        layout,
        stats,
        cga,
        1,
        True,
        page_size=page,
        heads_override=heads,
        sched_policy=scheduler,
        q_bound=q_bound,
        kv_bound=1024,
        window_left=127,
        sink=0.4,
        cpu_reference=True,
    )
    assert seen


@pytest.mark.parametrize("dtype,cga,stats", [("bfloat16", 1, "token"), ("float16", 2, "head")])
def test_nonpaged_origins(dtype, cga, stats, monkeypatch):
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    seen = _assert_direct_body(monkeypatch)
    run(64, dtype, stats, True, True, 1, engine_name="sdpa_fwd_prefill_sm100", cga=cga)
    assert seen


@pytest.mark.parametrize("dtype,cga", [("bfloat16", 1), ("float16", 2)])
def test_standalone(dtype, cga, monkeypatch):
    from native_d64_sm107_test_utils import test_native_d64_standalone_compact_thd as run

    seen = _assert_direct_body(monkeypatch)
    run(getattr(torch, dtype), cga, True, heads=8, splits=1, cpu_reference=True)
    assert seen


@pytest.mark.parametrize("dtype,cga", [("bfloat16", 1), ("float16", 2)])
@pytest.mark.parametrize("paged", [False, True], ids=["nonpaged", "paged"])
def test_batch_and_golden(dtype, cga, paged, monkeypatch):
    from test_sdpa_single_sequence_setup_sm107 import test_single_sequence_batch_bytes as run

    seen = _assert_direct_body(monkeypatch)
    run(64, dtype, cga, True, 128, engine_name="sdpa_fwd_prefill_sm100", paged=paged)
    assert seen


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_warm_refill(dtype, monkeypatch):
    from test_sdpa_prepared_thd import _buffers, _compact_thd_api, _execute_compact_thd
    from warm_stress import StressCase, assert_warm_stress

    seen = _assert_direct_body(monkeypatch)
    b, sq, sk, h, hk, d = 4, 1025, 1025, 8, 2, 64
    values = _buffers(b, sq, sk, h, hk, d, seed=38217, dtype=dtype)
    api = _compact_thd_api(values, b, sq, sk, h, hk, d, cga=1, pack_gqa=True)
    workspace = torch.empty(max(1, api.scratch_workspace_bytes()), dtype=torch.uint8, device="cuda")

    def launch():
        _execute_compact_thd(api, values, workspace)

    launch()
    for batch in range(b):
        q = values["q"][batch * sq : (batch + 1) * sq].cpu().double().transpose(0, 1)
        k = values["k"][batch * sk : (batch + 1) * sk].cpu().double().repeat_interleave(h // hk, 1).transpose(0, 1)
        v = values["v"][batch * sk : (batch + 1) * sk].cpu().double().repeat_interleave(h // hk, 1).transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) * d**-0.5
        scores.masked_fill_(torch.arange(sk)[None, :] > torch.arange(sq)[:, None], -torch.inf)
        reference = (scores.softmax(-1) @ v).transpose(0, 1)
        stats = scores.logsumexp(-1).T
        torch.testing.assert_close(values["o"][batch * sq : (batch + 1) * sq].cpu().double(), reference, atol=0.004, rtol=0.02)
        torch.testing.assert_close(values["lse"][batch * sq : (batch + 1) * sq].cpu().double(), stats, atol=0.0006, rtol=0)
    assert seen
    # Late query tiles traverse nine KV tiles through a four-stage ring.
    assert_warm_stress(
        StressCase("D64 packed THD prefill", launch, {name: values[name] for name in ("o", "lse")}, thread_blocks=api._thd_plan().units, refills_per_launch=5),
        warm=256,
        poisoned=256,
    )
