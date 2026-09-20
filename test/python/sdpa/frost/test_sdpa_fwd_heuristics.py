# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDPA-forward heuristic's recommend contract.

Unit tier (no GPU): recommend() emits ordered COMPLETE knob assignments — the
same engine repeated with different sets, every set admissible, no cartesian
blowup, cross-axis constraints never emitted, mode never on an entry.

Executable tier (SM100): the ranked list carries knob-suffixed duplicates of
one cell; the split_kv entry, pinned by name, builds, carves its partial slabs
from the caller workspace, recombines correctly (O and Stats), and its
(engine_id, knobs) tuple replays on a fresh graph.
"""

import math

import pytest
import torch

import cudnn
from cudnn.engines import manifest
from cudnn.engines.base import PlanConfig
from cudnn.sdpa.fwd import engines
from cudnn.engines.heuristics import _assemble
from cudnn.sdpa.fwd.heuristics import _MAX_SETS_PER_ENGINE, recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2

_F16 = "sdpa_fwd_prefill_sm100"
_OFFERED = {_F16: 20500, "sdpa_fwd_prefill_sm100_fp8": 20501}


def _facts(**over):
    base = dict(
        b=1,
        h_q=4,
        h_kv=4,
        s_q=128,
        s_kv=8192,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.HALF,
        causal=True,
        device_cc=(10, 0),
        device_sm_count=148,
    )
    base.update(over)
    return SdpaGraphFacts(**base)


@pytest.mark.L0
def test_recommend_emits_multiple_complete_sets_per_engine():
    plans = recommend("A", _facts(), _OFFERED)
    f16 = [p for p in plans if p.engine_id == 20500]
    assert len(f16) >= 3, "expected sched + split runners behind the primary"
    for p in f16:
        k = p.knobs
        # Complete assignment: every axis the row declares carries a value.
        assert None not in (k.sched_policy, k.tile_m, k.tile_n, k.cga, k.split_kv)
        assert p.mode is None and p.cpp_index is None
    assert len({p.knobs for p in f16}) == len(f16), "duplicate knob sets emitted"
    assert len(f16) <= _MAX_SETS_PER_ENGINE


@pytest.mark.L0
def test_recommend_primary_reproduces_the_derived_scheduler():
    # Behavior preservation on the UNSPLIT leg: the first set carries exactly
    # what the adapter's internal derivation historically chose (causal + small
    # working set -> LPT_L2; mask-free -> NATURAL with no sched runners). A
    # grid that fills the machine never splits, so it reads the derivation
    # straight off the primary.
    causal = recommend("A", _facts(s_q=8192), _OFFERED)
    assert causal[0].knobs.split_kv == 1 and causal[0].knobs.sched_policy == 2  # SCHED_LPT_L2
    dense = recommend("A", _facts(causal=False), _OFFERED)
    dense_f16 = [p for p in dense if p.engine_id == 20500]
    assert dense_f16[0].knobs.sched_policy == 0  # SCHED_NATURAL
    assert all(p.knobs.sched_policy == 0 for p in dense_f16), "mask-free graphs gain nothing from LPT runners"


@pytest.mark.L0
def test_recommend_packs_partial_gqa_group_on_decode_shapes():
    # 96 query heads over 8 KV heads (G=12) at S_q=1: 12 does not divide the
    # 128-row tile, but 4 does -- the d128 f16 flavor packs 4 heads per token
    # row-group (partial PackGQA), and the packed set leads like any other
    # decode-shaped GQA graph, unpacked as the runner-up.
    f16 = [p for p in recommend("A", _facts(h_q=96, h_kv=8, s_q=1, causal=False), _OFFERED) if p.engine_id == 20500]
    assert f16[0].knobs.pack_gqa is True and False in {p.knobs.pack_gqa for p in f16}, [p.knobs for p in f16]
    # G=3 shares no factor with the tile: no packed set is proposed.
    f16 = [p for p in recommend("A", _facts(h_q=24, h_kv=8, s_q=1, causal=False), _OFFERED) if p.engine_id == 20500]
    assert f16 and all(p.knobs.pack_gqa is False for p in f16), [p.knobs for p in f16]


@pytest.mark.L0
def test_split_model_sees_the_partial_pack_group_not_the_gqa_ratio(monkeypatch):
    """The split-KV wave-cost model is fed the PACKED launch.  Under partial
    PackGQA the kernel launches ``h_q // p`` packed heads (96/8, p=4: 24 packed
    heads of s_q*4 rows), not ``h_q // G`` (8): fed G, the model saw a 3x
    smaller grid and over-proposed the split on the GLM decode shape (b=1:
    split 8 instead of 2; b=2..4: a split where the packed grid already fills
    the machine).  Both launches the model compares -- the split leg and the
    unsplit runner-up -- must carry p.  The packed leg rides the d128 DECODE
    tile (S_q * p = 4 <= 128 rows: TILE_CGA_M=1, one CTA per tile), so the
    model is fed ctas_per_tile=1 and levels the 24 x b grid over 148 SMs with
    split 4 / 2 / 1 at b = 1 / 2 / 4 (2 / 1 / 1 while the leg rode the cga2
    prefill tile)."""
    import cudnn.sdpa.fwd.heuristics as heur

    seen = []
    real = heur.choose_split_kv

    def recording(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(heur, "choose_split_kv", recording)
    for b, want in ((1, 4), (2, 2), (4, 1)):
        seen.clear()
        facts = _facts(b=b, h_q=96, h_kv=8, s_q=1, s_kv=4096, causal=False, dtype=cudnn.data_type.BFLOAT16)
        f16 = [p for p in recommend("A", facts, _OFFERED) if p.engine_id == 20500]
        assert seen, "the split leg never consulted the wave-cost model"
        # Geometry runners now consult the model too. Packed candidates carry
        # p=4 (24 head groups), unpacked ones carry all 96 heads; no candidate
        # may mistake the GQA ratio G=12 for p and model only eight heads.
        assert {kw["heads_q"] for kw in seen} == {24, 96}
        for kw in seen:
            assert kw["q_tiles"] == 1
            assert kw["unsplit_launch"] is not None and kw["unsplit_launch"].heads_q == kw["heads_q"], kw["unsplit_launch"]
            # The combine still reduces the graph's own (S_q, H, B) rows.
            assert kw["combine_rows"] == 1 * 96 * b
        assert f16[0].knobs.pack_gqa is True and f16[0].knobs.cga == 1 and f16[0].knobs.split_kv == want, (b, f16[0].knobs)
        # Exactly the split the model gives the 24-packed-head geometry on the decode tile.
        assert f16[0].knobs.split_kv == real(q_tiles=1, heads_q=24, batch=b, kv_tiles=32, sm_count=148, combine_rows=96 * b, ctas_per_tile=1)


@pytest.mark.L0
def test_split_and_scheduler_stay_coupled_whichever_leads():
    """A split set rides the plain scheduler — structural, so it must bind the
    PRIMARY too, not just the runner-ups. config_sm120 raises outright on
    split_kv > 1 under an LPT remap, so an LPT+split set is unbuildable there.
    Regression: flipping the split to lead once let it inherit the derived
    LPT_L2 policy on causal graphs."""
    for f in (_facts(), _facts(causal=False), _facts(s_q=8192), _facts(h_q=1, h_kv=1)):
        for p in recommend("A", f, _OFFERED):
            if (p.knobs.split_kv or 1) > 1:
                assert p.knobs.sched_policy == 0, f"split set on a non-plain scheduler: {p.knobs}"


@pytest.mark.L0
def test_recommend_split_leads_and_respects_structure():
    # A split the wave-cost model asks for is what a plain build_plans() runs,
    # with no-split behind it for autotune / select_plan. Sweep justifying the
    # lead (B300, ar_dit chunked prefill, bf16 B1xH9xD128, S_kv=62208, no mask):
    #   S_q=985   0.955 ms -> 0.556 ms (split 4, 1.72x)
    #   S_q=2048  0.960 ms -> 0.722 ms (split 2, 1.33x)
    #   S_q>=4096 unchanged (model declines to split a full grid)
    plans = [p for p in recommend("A", _facts(causal=False), _OFFERED) if p.engine_id == 20500]
    assert plans[0].knobs.split_kv > 1, "an underfilled grid runs the split the model chose"
    assert any(p.knobs.split_kv == 1 for p in plans), "no-split must stay reachable as the runner-up"
    for bad in (dict(has_sink=True), dict(thd=True, padded=True), dict(padded=True), dict(s_q=8192)):
        got = [p for p in recommend("A", _facts(**bad), _OFFERED) if p.engine_id == 20500]
        assert all(p.knobs.split_kv == 1 for p in got), f"split emitted under {bad}"


@pytest.mark.L0
def test_recommend_every_set_is_admissible():
    facts = _facts()
    for p in recommend("A", facts, _OFFERED):
        spec = next(s for s in engines.ENGINE_SPECS if _OFFERED.get(s.name) == p.engine_id)
        assert engines.mismatch(spec.capabilities, facts, p.knobs) is None


@pytest.mark.L0
@pytest.mark.parametrize(
    "engine_name,dtype,is_fp8",
    [
        ("sdpa_fwd_prefill_sm120", cudnn.data_type.HALF, False),
        ("sdpa_fwd_prefill_sm120_fp8", cudnn.data_type.FP8_E4M3, True),
    ],
)
def test_sm120_d192_keeps_sm120_cga_domain(engine_name, dtype, is_fp8):
    facts = _facts(
        s_q=256,
        s_kv=256,
        d_qk=192,
        d_v=128,
        dtype=dtype,
        dtype_o=cudnn.data_type.HALF,
        is_fp8=is_fp8,
        device_cc=(12, 0),
        device_sm_count=84,
    )
    offered = {engine_name: 20504}
    plans = recommend("A", facts, offered)
    assert plans
    assert all(plan.knobs.cga == 1 for plan in plans)
    spec = next(spec for spec in engines.ENGINE_SPECS if spec.name == engine_name)
    assert all(engines.mismatch(spec.capabilities, facts, plan.knobs) is None for plan in plans)


@pytest.mark.L0
def test_sm120_d512_flavor_pins_its_one_cta_tile():
    """The d512 flavor is built for (64, 32) alone: the grid rule's tile_m and
    the largest-fitting tile_n both yield to the kernel table
    (config_sm120.tile_domain) on full and underfilled grids, and no runner-up
    proposes another tile. At d128 the same table keeps kv_tile 32 out."""
    spec = next(spec for spec in engines.ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm120")
    offered = {"sdpa_fwd_prefill_sm120": 20504}
    for d_qk, d_v in ((264, 264), (264, 512), (512, 264), (272, 320), (384, 448), (496, 504), (512, 512)):
        for s in (128, 16384):
            facts = _facts(s_q=s, s_kv=s, h_q=64, h_kv=1, d_qk=d_qk, d_v=d_v, device_cc=(12, 0), device_sm_count=188)
            plans = recommend("A", facts, offered)
            assert plans
            assert {(p.knobs.tile_m, p.knobs.tile_n) for p in plans} == {(64, 32)}
            assert all(engines.mismatch(spec.capabilities, facts, p.knobs) is None for p in plans)
    facts = _facts(d_qk=128, d_v=128, device_cc=(12, 0), device_sm_count=188)
    plans = recommend("A", facts, offered)
    assert plans and all(p.knobs.tile_n != 32 for p in plans)


@pytest.mark.L0
def test_sm120_d512_sliding_window_packs_gqa():
    """The d512 flavor packs the GQA group into its 64-row Q tile under a sliding
    window (the packed unit's key span shrinks from 64 + W to 1 + W tokens for
    MQA) and walks windowed units, packed or not, with plain LPT; a 128-head
    group cannot pack into 64 rows, MHA has nothing to pack, and without a
    window the decode rule (s_q < tile) and the L2 scheduler rule decide as
    before."""
    offered = {"sdpa_fwd_prefill_sm120": 20504}
    spec = next(spec for spec in engines.ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm120")

    def primary(**over):
        facts = _facts(**{**dict(s_q=16384, s_kv=16384, h_q=64, h_kv=1, d_qk=512, d_v=512, device_cc=(12, 0), device_sm_count=188), **over})
        plans = recommend("A", facts, offered)
        assert plans and engines.mismatch(spec.capabilities, facts, plans[0].knobs) is None
        return plans[0].knobs

    packed = primary(window_left=128)
    assert (packed.pack_gqa, packed.tile_m, packed.sched_policy) == (True, 64, SCHED_LPT)  # packed window units walk plain LPT
    assert primary(window_left=128, h_q=8, h_kv=1).pack_gqa is True
    for d_qk, d_v in ((264, 512), (512, 264), (320, 384)):
        packed = primary(window_left=128, d_qk=d_qk, d_v=d_v)
        assert (packed.pack_gqa, packed.tile_m, packed.tile_n, packed.sched_policy) == (True, 64, 32, SCHED_LPT)
    pro = primary(window_left=128, h_q=128, h_kv=1)  # G=128 does not divide the 64-row tile; windowed units still walk LPT
    assert (pro.pack_gqa, pro.sched_policy) == (False, SCHED_LPT)
    assert primary(window_left=128, h_q=64, h_kv=64).pack_gqa is False  # MHA
    full_causal = primary()  # full causal prefill: decode rule only, L2 rule for the scheduler
    assert (full_causal.pack_gqa, full_causal.sched_policy) == (False, SCHED_LPT_L2)
    assert primary(window_left=128, d_qk=128, d_v=128).pack_gqa is False  # the rule is the d512 flavor's


@pytest.mark.L0
def test_sm120_fp8_d512_flavor_and_tile_contract():
    """FP8 D512 has a dedicated 64x64 tile and a full-width shared-memory budget."""
    from cudnn.sdpa.fwd.config_sm120 import pick_flavor, smem_bytes, tile_domain

    assert pick_flavor(512, 512, fp8=True) == (512, 512)
    assert tile_domain(512, 512, fp8=True) == frozenset({(64, 64)})
    assert smem_bytes(512, 512, 64, 64, itemsize=1, out_itemsize=2) == 98328
    assert smem_bytes(272, 496, 64, 64, itemsize=1, out_itemsize=1) == 98328
    name = "sdpa_fwd_prefill_sm120_fp8"
    spec = next(spec for spec in engines.ENGINE_SPECS if spec.name == name)
    for d_qk, d_v in ((272, 272), (496, 512), (512, 496), (512, 512)):
        facts = _facts(
            d_qk=d_qk, d_v=d_v, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.FP8_E4M3, is_fp8=True, device_cc=(12, 0), device_sm_count=188
        )
        plans = recommend("A", facts, {name: 20504})
        assert plans, (d_qk, d_v)
        assert {(plan.knobs.tile_m, plan.knobs.tile_n) for plan in plans} == {(64, 64)}
        assert all(engines.mismatch(spec.capabilities, facts, plan.knobs) is None for plan in plans)
    for d_qk, d_v in ((256, 512), (512, 256), (264, 512), (512, 528)):
        facts = _facts(
            d_qk=d_qk, d_v=d_v, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.FP8_E4M3, is_fp8=True, device_cc=(12, 0), device_sm_count=188
        )
        assert engines.mismatch(spec.capabilities, facts) is not None, (d_qk, d_v)
    for dim in (128, 256):
        assert pick_flavor(dim, dim, fp8=True) is None
        assert (128, 128) in tile_domain(dim, dim, fp8=True)


@pytest.mark.L0
@pytest.mark.parametrize("dim", [256, 512])
def test_sm120_fp8_dense_layouts_and_split_output(dim):
    """FP8 layouts normalize to compact storage before main and split-combine kernels."""
    from types import SimpleNamespace

    shape = (1, 4, 16, dim)
    head_major = (64 * dim, 16 * dim, dim, 1)
    padded = (64 * (dim + 1), 16 * (dim + 1), dim + 1, 1)
    query = SimpleNamespace(get_dim=lambda: shape, get_stride=lambda: padded)
    output = SimpleNamespace(get_dim=lambda: shape, get_stride=lambda: head_major)
    facts = _facts(
        s_q=16,
        s_kv=32768,
        h_q=4,
        h_kv=1,
        d_qk=dim,
        d_v=dim,
        q_t=query,
        o_t=output,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_fp8=True,
        device_cc=(12, 0),
        device_sm_count=188,
    )
    name = "sdpa_fwd_prefill_sm120_fp8"
    spec = next(spec for spec in engines.ENGINE_SPECS if spec.name == name)
    knobs = engines.SdpaFwdKnobs(sched_policy=0, tile_m=64, tile_n=64, cga=1, pack_gqa=False, split_kv=2)
    assert engines.mismatch(spec.capabilities, facts, knobs) is None
    plans = recommend("A", facts, {name: 20504})
    assert plans and any((plan.knobs.split_kv or 1) > 1 for plan in plans)


@pytest.mark.L0
@pytest.mark.parametrize("mxfp8", [False, True], ids=["per_tensor", "block_scale"])
@pytest.mark.parametrize(
    ("d_qk", "d_v", "expected_cga"),
    [(128, 128, 2), (256, 256, 1)],
    ids=["d128", "d256"],
)
def test_quantized_cga_follows_selected_native_flavor(mxfp8, d_qk, d_v, expected_cga):
    """A unified dtype-family engine must advertise the geometry it launches."""

    name = engines.engine_name(mxfp8=mxfp8, fp8=not mxfp8)
    facts = _facts(
        d_qk=d_qk,
        d_v=d_v,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.HALF,
        is_mxfp8=mxfp8,
        is_fp8=not mxfp8,
    )
    plans = recommend("A", facts, {name: 20510})
    assert plans
    assert {plan.knobs.cga for plan in plans} == {expected_cga}

    spec = next(spec for spec in engines.ENGINE_SPECS if spec.name == name)
    assert engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(cga=expected_cga)) is None
    wrong_cga = 1 if expected_cga == 2 else 2
    assert "outside this engine's domain" in engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(cga=wrong_cga))


@pytest.mark.L0
def test_per_tensor_fp8_envelope_uses_d256_cga1():
    """The D256 flavor's geometry (cga = 1) reaches the plan list for its exact
    shape; a non-native shape in its padded envelope is declined (the flavor is
    floored to exact shapes until its padded paths are validated), so no plan is
    proposed for it at all."""

    name = engines.engine_name(fp8=True)
    exact = _facts(d_qk=256, d_v=256, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.HALF, is_fp8=True)
    plans = recommend("A", exact, {name: 20510})
    assert plans
    assert {plan.knobs.cga for plan in plans} == {1}
    padded = _facts(d_qk=224, d_v=224, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.HALF, is_fp8=True)
    assert not recommend("A", padded, {name: 20510})


@pytest.mark.L0
def test_d256_fp8_config_requires_cga1():
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3, DTYPE_FP16
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d256, make_cfg_d256_mxfp8

    params = TemplateParams(dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_FP16, cta_mma=1)
    cfg_pt = make_cfg_d256(params)[0]
    cfg_mx = make_cfg_d256_mxfp8(params)[0]
    assert cfg_pt.CGA_M == cfg_pt.CTA_MMA == 1
    assert cfg_mx.CGA_M == cfg_mx.CTA_MMA == 1
    with pytest.raises(ValueError, match="FP8/MXFP8 requires cta_mma=1"):
        make_cfg_d256(TemplateParams(dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_FP16, cta_mma=2))


@pytest.mark.L0
@pytest.mark.parametrize("mxfp8", [False, True], ids=["per_tensor", "block_scale"])
@pytest.mark.parametrize("sched_policy", [0, 1, 2], ids=["natural", "lpt", "lpt_l2"])
def test_d256_config_honors_explicit_scheduler(mxfp8, sched_policy):
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3, DTYPE_FP16
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d256, make_cfg_d256_mxfp8

    params = TemplateParams(
        dtype_qkv=DTYPE_E4M3,
        dtype_o=DTYPE_FP16,
        window_right=0,
        sched_policy=sched_policy,
        cta_mma=1,
    )
    make_cfg = make_cfg_d256_mxfp8 if mxfp8 else make_cfg_d256
    assert make_cfg(params)[0].SCHEDULER_POLICY == sched_policy


@pytest.mark.L0
@pytest.mark.parametrize(
    ("mxfp8", "expected_sched"),
    [(False, 2), (True, 1)],
    ids=["per_tensor_lpt_l2", "block_scale_lpt"],
)
def test_d256_quantized_primary_uses_measured_scheduler(mxfp8, expected_sched):
    name = engines.engine_name(mxfp8=mxfp8, fp8=not mxfp8)
    facts = _facts(
        s_q=8192,
        d_qk=256,
        d_v=256,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.HALF,
        is_mxfp8=mxfp8,
        is_fp8=not mxfp8,
    )
    plans = recommend("A", facts, {name: 20510})
    assert plans[0].knobs.cga == 1
    assert plans[0].knobs.split_kv == 1
    assert plans[0].knobs.sched_policy == expected_sched


@pytest.mark.L0
def test_d512_mxfp8_primary_uses_measured_scheduler():
    name = engines.engine_name(mxfp8=True)
    offered = {name: 20510}
    base = dict(
        s_q=8192,
        d_qk=512,
        d_v=512,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.HALF,
        is_mxfp8=True,
    )

    causal = recommend("A", _facts(**base), offered)
    dense = recommend("A", _facts(causal=False, **base), offered)
    thd = recommend("A", _facts(thd=True, padded=True, **base), offered)

    assert (causal[0].knobs.cga, causal[0].knobs.sched_policy) == (1, 1)
    assert (dense[0].knobs.cga, dense[0].knobs.sched_policy) == (1, 0)
    assert (thd[0].knobs.cga, thd[0].knobs.sched_policy) == (1, 0)

    rubin_name = engines.engine_name(mxfp8=True, arch="sm107")
    rubin = recommend("A", _facts(device_cc=(10, 7), **base), {rubin_name: 20511})
    assert rubin
    assert all((plan.knobs.cga, plan.knobs.sched_policy) == (2, 0) for plan in rubin)


@pytest.mark.L0
def test_assemble_strips_mode_dedups_and_our_proposals_lead():
    """Placement is the SHARED layer's job (engines/heuristics._assemble):
    proposals lead the backend's entries inside each mode block by standing
    assumption, the delegating entry never leads an OPENSOURCE block, one
    config repeated across blocks keeps its first position, and no final
    entry carries a mode."""
    ours = [PlanConfig(20500, "set-a"), PlanConfig(20500, "set-b")]
    backend = [
        PlanConfig(-1, None),  # delegating (mode None)
        PlanConfig(7, {"k": 1}, cpp_index=0, mode=cudnn.heur_mode.A),
        PlanConfig(7, {"k": 1}, cpp_index=1, mode=cudnn.heur_mode.FALLBACK),  # same config, later block
    ]
    final = _assemble([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK], lambda kind: ours if kind == "A" else [], backend)
    assert all(p.mode is None for p in final), "mode must never reach final entries"
    assert [p.engine_id for p in final[:2]] == [20500, 20500], "our proposals lead the backend inside the block"
    assert sum(1 for p in final if p.engine_id == 7) == 1, "one backend config repeated across modes must dedup"
    assert next(p for p in final if p.engine_id == 7).cpp_index == 0, "first position wins"
    assert sum(1 for p in final if p.engine_id == -1) == 1
    # OPENSOURCE: ours + delegating, and never the backend's own entries.
    oss = _assemble([cudnn.heur_mode.OPENSOURCE], lambda kind: ours, backend)
    assert [p.engine_id for p in oss] == [20500, 20500, -1]


@pytest.mark.L0
def test_fallback_kind_is_least_demanding():
    for p in recommend("FALLBACK", _facts(), _OFFERED):
        assert p.knobs.split_kv == 1
        assert p.knobs.sched_policy == 0  # SCHED_NATURAL


# ---------------------------------------------------------------------------
# Executable tier — SM100 graph path
# ---------------------------------------------------------------------------


def _is_sm100() -> bool:
    # Pre-Rubin only: the executable tier drives the f16 family, which has no
    # Rubin lowering (Rubin serves per-tensor FP8 only).
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return major == 10 and minor <= 6


def _dsl_available() -> bool:
    try:
        import cutlass.experimental  # noqa: F401
    except ImportError:
        return False
    return True


def _build_decodeish_graph(*, causal=True):
    B, H, SQ, SKV, D = 1, 4, 128, 8192, 128
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = g.tensor(dim=(B, H, SQ, D), stride=(SQ * H * D, D, H * D, 1), data_type=cudnn.data_type.HALF, name="q")
    k = g.tensor(dim=(B, H, SKV, D), stride=(SKV * H * D, D, H * D, 1), data_type=cudnn.data_type.HALF, name="k")
    v = g.tensor(dim=(B, H, SKV, D), stride=(SKV * H * D, D, H * D, 1), data_type=cudnn.data_type.HALF, name="v")
    o, st = g.sdpa(name="sdpa", q=q, k=k, v=v, attn_scale=1.0 / math.sqrt(D), is_inference=False, use_causal_mask=causal)
    o.set_output(True).set_dim((B, H, SQ, D)).set_stride((SQ * H * D, D, H * D, 1))
    st.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    return g, (q, k, v, o, st), (B, H, SQ, SKV, D)


@pytest.mark.L1
@pytest.mark.skipif(not (_is_sm100() and _dsl_available()), reason="needs an SM100 device and nvidia-cutlass-dsl")
def test_split_kv_plan_pinned_by_name_matches_reference():
    """Issue F-2 regression: the split plan is graph-reachable, carves its
    slabs from the caller workspace, and recombines exactly."""
    g, (q, k, v, o, st), (B, H, SQ, SKV, D) = _build_decodeish_graph(causal=False)
    # The split value depends on this device's SM count — ask the chooser
    # rather than hard-coding one that only holds at one part's geometry.
    from cudnn._device import device_info
    from cudnn.sdpa.fwd.config_sm100 import cga_tile_m
    from cudnn.sdpa.fwd.heuristics import choose_split_kv

    want = choose_split_kv(
        q_tiles=-(-SQ // cga_tile_m(D)),
        heads_q=H,
        batch=B,
        kv_tiles=-(-SKV // 128),
        sm_count=device_info(torch.cuda.current_device()).sm_count,
        combine_rows=SQ * H * B,
        ctas_per_tile=2,
    )
    if want == 1:
        pytest.skip("this part is small enough that the shape already fills it")
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    f16 = [n for n in names if n.split("[")[0] == "sdpa_fwd_prefill_sm100"]
    assert len(f16) >= 3, f"expected knob-suffixed duplicates of the f16 family engine: {f16}"
    split_idx = next(i for i, n in enumerate(names) if n.split("[")[0] == "sdpa_fwd_prefill_sm100" and f"split_kv={want}" in n)
    g.select_plan(split_idx)
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() > 0, "the split plan must report its partial-slab workspace"

    torch.manual_seed(0)
    q_gpu = torch.randn(B, SQ, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    k_gpu = torch.randn(B, SKV, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    v_gpu = torch.randn(B, SKV, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    o_gpu = torch.empty(B, SQ, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    st_gpu = torch.empty(B, H, SQ, 1, device="cuda", dtype=torch.float32)
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu, st: st_gpu}, ws)
    torch.cuda.synchronize()

    s = torch.einsum("bhqd,bhkd->bhqk", q_gpu.float(), k_gpu.float()) / math.sqrt(D)
    torch.testing.assert_close(o_gpu, torch.einsum("bhqk,bhkd->bhqd", torch.softmax(s, dim=-1), v_gpu.float()).half(), atol=5e-2, rtol=3e-2)
    torch.testing.assert_close(st_gpu.squeeze(-1), torch.logsumexp(s, dim=-1), atol=2e-3, rtol=2e-3)

    # Autotune replay: the split entry round-trips through (engine_id, knobs).
    eng_id, knobs = g.get_engine_and_knobs_at_index(split_idx)
    assert knobs.split_kv == want
    g2, _handles2, _ = _build_decodeish_graph(causal=False)
    cfg = g2.create_execution_plan(eng_id, knobs)
    assert cfg is not None


@pytest.mark.L1
@pytest.mark.skipif(not (_is_sm100() and _dsl_available()), reason="needs an SM100 device and nvidia-cutlass-dsl")
def test_runner_up_sched_plan_builds_and_matches_the_winner():
    """select_plan on a runner-up knob set compiles the adapter with exactly
    that set and executes correctly — honored, not silently degraded."""
    g, (q, k, v, o, st), (B, H, SQ, SKV, D) = _build_decodeish_graph()
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    nat_idx = next(i for i, n in enumerate(names) if n.split("[")[0] == "sdpa_fwd_prefill_sm100" and "sched_policy=0" in n and "split_kv=1" in n)
    g.select_plan(nat_idx)
    g.check_support()
    g.build_plans()
    eng_id, knobs = g.get_engine_and_knobs_at_index(nat_idx)
    assert knobs.sched_policy == 0 and knobs.split_kv == 1

    torch.manual_seed(0)
    q_gpu = torch.randn(B, SQ, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    k_gpu = torch.randn(B, SKV, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    v_gpu = torch.randn(B, SKV, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    o_gpu = torch.empty(B, SQ, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    st_gpu = torch.empty(B, H, SQ, 1, device="cuda", dtype=torch.float32)
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({q: q_gpu, k: k_gpu, v: v_gpu, o: o_gpu, st: st_gpu}, ws)
    torch.cuda.synchronize()
    s = torch.einsum("bhqd,bhkd->bhqk", q_gpu.float(), k_gpu.float()) / math.sqrt(D)
    i = torch.arange(SQ, device="cuda").view(SQ, 1)
    j = torch.arange(SKV, device="cuda").view(1, SKV)
    s = s.masked_fill(j > i, float("-inf"))
    torch.testing.assert_close(o_gpu, torch.einsum("bhqk,bhkd->bhqd", torch.softmax(s, dim=-1), v_gpu.float()).half(), atol=5e-2, rtol=3e-2)


# --- Epilogue gate (PR-A): a FACT the heuristics must never trade against ------

# The REAL Rubin ids, derived from the manifest (recommend() only keys on ``offered.get(spec.name)``, so any
# labels would pass -- but a reader takes literals here for engine ids, and the file's older ``_OFFERED`` labels
# already predate the slot re-cut).  20515 = base + slot 15 (f16), 20514 = base + slot 14 (fp8) today.
_SDPA_FWD_FAMILY = next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_fwd")
_RUBIN_F16, _RUBIN_FP8 = "sdpa_fwd_prefill_sm107", "sdpa_fwd_prefill_sm107_fp8"
_RUBIN_OFFERED = {name: _SDPA_FWD_FAMILY.engine_id + _SDPA_FWD_FAMILY.slots[name].slot for name in (_RUBIN_F16, _RUBIN_FP8)}


@pytest.mark.L0
def test_heuristics_never_propose_split_or_pack_for_a_gated_graph():
    """The fused O * sigmoid(G) epilogue lives in the UNSPLIT, UNPACKED kernel:
    a split's combine would write the un-gated O and a packed tile interleaves
    (token, head) rows the gate's TMA box cannot address.  mismatch() declines
    both pairs, and -- rule 4, never PROPOSE a knob the row cannot honour -- the
    proposal helpers must not emit them either, so no plan is ever listed only
    to be filtered.  Pinned on the proposal helpers with a synthetic row that
    WOULD split / pack an ungated graph, then on the real Rubin rows."""
    import dataclasses

    from cudnn.sdpa.fwd.heuristics import _pack_gqa_eligible, _split_points

    gated = dict(has_epilogue_gate=True, epilogue_gate_dtype=cudnn.data_type.HALF, d_qk=256, d_v=256, device_cc=(10, 7))
    row = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    permissive = dataclasses.replace(row, split_kv_supported=True, split_d_shapes=None, pack_gqas=frozenset({False, True}), pack_gqa_d_shapes=None)
    # Ungated: the underfilled causal grid (s_q=128, s_kv=8192, 148 SMs) asks for a split; gated: never.
    assert any(p > 1 for p in _split_points(permissive, _facts(d_qk=256, d_v=256, device_cc=(10, 7)), 128, 128, 2)), "the control must split"
    assert _split_points(permissive, _facts(**gated), 128, 128, 2) == [1]
    assert _pack_gqa_eligible(permissive, _facts(h_q=8, h_kv=2, d_qk=256, d_v=256, device_cc=(10, 7)), 128) is True, "the control must pack"
    assert _pack_gqa_eligible(permissive, _facts(h_q=8, h_kv=2, **gated), 128) is False

    # The real rows: every emitted set is unsplit and unpacked, and admissible.
    for facts in (_facts(**gated), _facts(h_q=8, h_kv=2, **gated), _facts(causal=False, **gated)):
        plans = recommend("A", facts, _RUBIN_OFFERED)
        assert plans, "the Rubin f16 row serves the gated d256 graph"
        assert {p.engine_id for p in plans} == {_RUBIN_OFFERED[_RUBIN_F16]}
        for p in plans:
            assert (p.knobs.split_kv or 1) == 1 and not p.knobs.pack_gqa, p.knobs
            spec = next(s for s in engines.ENGINE_SPECS if _RUBIN_OFFERED.get(s.name) == p.engine_id)
            assert engines.mismatch(spec.capabilities, facts, p.knobs) is None
    fp8 = dict(gated, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_fp8=True, epilogue_gate_dtype=cudnn.data_type.BFLOAT16)
    plans = recommend("A", _facts(h_q=8, h_kv=2, **fp8), _RUBIN_OFFERED)
    assert plans and {p.engine_id for p in plans} == {_RUBIN_OFFERED[_RUBIN_FP8]}
    assert all((p.knobs.split_kv or 1) == 1 and not p.knobs.pack_gqa for p in plans), [p.knobs for p in plans]
    # ...and a gated graph on a flavor that does not carry the gate proposes nothing at all.
    assert not recommend("A", _facts(**dict(gated, d_qk=128, d_v=128)), _RUBIN_OFFERED)


def _decode_d256_facts(**over):
    """Qwen3.5 decode as served: 32/2 heads (packed 16:1), d=256, S_q=1, paged
    (page 16) over a 4096-key table, on a 148-SM SM100 part."""
    base = dict(
        b=32,
        h_q=32,
        h_kv=2,
        s_q=1,
        s_kv=4096,
        d_qk=256,
        d_v=256,
        dtype=cudnn.data_type.BFLOAT16,
        causal=False,
        padded=True,
        has_paged_kv=True,
        page_size=16,
        device_cc=(10, 0),
        device_sm_count=148,
    )
    base.update(over)
    return SdpaGraphFacts(**base)


@pytest.mark.L0
def test_decode_tile_split_points_lead_with_the_eager_safe_choice():
    """A d256 graph the decode tile serves gets the decode split model
    (choose_decode_tile_split_kv), not the prefill fit: the LEADING set is the
    choice that also pays for the split path's second host launch, the captured
    caller's optimum follows as a runner-up (select_plan / autotune reach it),
    and no-split closes the list. The same graph one token wider (S_q=2: 32
    packed rows, the compiled-but-unrouted 32-column tile) or longer (S_q=3: 48
    rows) is the prefill tile's launch and keeps the prefill model, which does
    not split either shape at b=32."""

    def sets(**over):
        return [p.knobs for p in recommend("A", _decode_d256_facts(**over), _OFFERED) if p.engine_id == 20500]

    serving = sets()
    assert serving[0].split_kv == 1 and serving[0].pack_gqa is True, serving[0]
    assert [k.split_kv for k in serving if k.split_kv > 1] == [2], serving
    small = sets(b=8)
    assert small[0].split_kv == 8, small[0]
    assert any(k.split_kv == 1 for k in small[1:]), small
    saturated = sets(b=128)
    assert all(k.split_kv == 1 for k in saturated), saturated
    long_kv = sets(s_kv=16384)
    assert long_kv[0].split_kv == 2, long_kv[0]
    # The 16-row MTP step (Qwen3-Next 16/2 at S_q=2 bottom-right: 8:1 packing)
    # is decode-shaped and follows the serving shape's policy.
    mtp16 = sets(h_q=16, s_q=2, causal=True, bottom_right=True)
    assert mtp16[0].split_kv == 1 and mtp16[0].pack_gqa is True, mtp16[0]
    assert [k.split_kv for k in mtp16 if k.split_kv > 1] == [2], mtp16
    # The 32-row MTP step (32/2 at S_q=2) is NOT routed onto the decode tile
    # (config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS): the prefill model, unsplit.
    mtp32 = sets(s_q=2, causal=True, bottom_right=True)
    assert mtp32[0].split_kv == 1 and mtp32[0].pack_gqa is True and all(k.split_kv == 1 for k in mtp32), mtp32
    prefill = sets(s_q=3)
    assert prefill[0].split_kv == 1 and all(k.split_kv == 1 for k in prefill), prefill


@pytest.mark.L0
def test_decode_tile_model_counts_the_whole_packed_group():
    """96/8 (G = 12) at d256: the prefill tile packs gcd(12, 128) = 4 heads per
    row-group (partial PackGQA), the decode tile packs all 12 (HEADS_PER_TILE =
    QH_PER_KH, no partial form), so its routing test and its split model count
    S_q x 12 like the adapter does: S_q = 1 is a 12-row decode launch of b x 8
    units, S_q = 2 (24 rows) is the prefill tile's graph.  Fed the partial group
    the model would call S_q = 2 an 8-row decode launch and cost it with the
    decode model while the adapter lowers it onto the prefill tile."""
    from cudnn.sdpa.fwd.heuristics import _d256_decode_tile_selected, _decode_tile_pack_g, _pack_gqa_group

    row = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _F16)
    one, two = _decode_d256_facts(h_q=96, h_kv=8, s_q=1), _decode_d256_facts(h_q=96, h_kv=8, s_q=2)
    partial = _pack_gqa_group(row, one, 128, True)
    assert partial == 4, partial
    assert _decode_tile_pack_g(one, partial) == 12 and _decode_tile_pack_g(one, 1) == 1
    assert _d256_decode_tile_selected(row, one, _decode_tile_pack_g(one, partial))
    assert _d256_decode_tile_selected(row, two, partial), "the control: the partial group would admit the 24-row graph"
    assert not _d256_decode_tile_selected(row, two, _decode_tile_pack_g(two, partial))
