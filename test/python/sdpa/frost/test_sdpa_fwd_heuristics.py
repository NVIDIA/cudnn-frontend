# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDPA-forward heuristic's recommend contract.

Unit tier (no GPU): recommend() emits ordered COMPLETE knob assignments — the
same engine repeated with different sets, every set admissible, no cartesian
blowup, cross-axis constraints never emitted, mode never on an entry.

Executable tier (SM100): explicit public knob assignments build independently
of recommendation order. The split plan carves its partial slabs from caller
workspace, recombines correctly (O and Stats), and its (engine_id, knobs)
record replays on a fresh graph.
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
from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

_F16 = "sdpa_fwd_prefill_sm100"
_OFFERED = {_F16: 20500, "sdpa_fwd_prefill_sm100_fp8": 20501}


@pytest.fixture
def sm107_metadata_target(monkeypatch):
    """Heuristic unit tests model a supported target independently of the worker's DSL."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


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
    # d128 f16 causal at S_kv=8192 (4 MiB of K+V per head) now leads with plain
    # LPT (heuristics._SM100_D128_LPT_L2_MIN_BYTES: B200-measured, LPT_L2 only
    # pays off from ~16 MiB per head); the 32K graph keeps the L2 grouping.
    causal = recommend("A", _facts(s_q=8192), _OFFERED)
    assert causal[0].knobs.split_kv == 1 and causal[0].knobs.sched_policy == 1  # SCHED_LPT
    long_causal = recommend("A", _facts(s_q=32768, s_kv=32768), _OFFERED)
    assert long_causal[0].knobs.split_kv == 1 and long_causal[0].knobs.sched_policy == 2  # SCHED_LPT_L2
    dense = recommend("A", _facts(causal=False), _OFFERED)
    dense_f16 = [p for p in dense if p.engine_id == 20500]
    assert dense_f16[0].knobs.sched_policy == 0  # SCHED_NATURAL
    assert all(p.knobs.sched_policy == 0 for p in dense_f16), "mask-free graphs gain nothing from LPT runners"


@pytest.mark.L0
def test_recommend_packs_gqa_under_a_band_on_sm100():
    """SM100 rows pack a GQA group under a diagonal band at prefill S_q (llama
    3.1 layer: 64/8 heads, S=2048 causal), unpacked as the runner-up; a dense
    graph of the same shape keeps the decode rule (unpacked first); MHA never
    packs (heuristics._sm100_banded_gqa_packs)."""
    llama = dict(b=2, h_q=64, h_kv=8, s_q=2048, s_kv=2048)
    rows = (
        (20500, dict(dtype=cudnn.data_type.HALF)),
        (20501, dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.FP8_E4M3, is_fp8=True)),
    )
    for eid, dt in rows:
        plans = [p for p in recommend("A", _facts(causal=True, **dt, **llama), _OFFERED) if p.engine_id == eid]
        assert plans and plans[0].knobs.pack_gqa is True and False in {p.knobs.pack_gqa for p in plans}, (eid, [p.knobs for p in plans])
        window = [p for p in recommend("A", _facts(causal=True, window_left=127, **dt, **llama), _OFFERED) if p.engine_id == eid]
        assert window and window[0].knobs.pack_gqa is True, (eid, window[0].knobs)
        dense = [p for p in recommend("A", _facts(causal=False, **dt, **llama), _OFFERED) if p.engine_id == eid]
        assert dense and dense[0].knobs.pack_gqa is False, (eid, dense[0].knobs)
        mha = [p for p in recommend("A", _facts(causal=True, **dt, **{**llama, "h_kv": 64}), _OFFERED) if p.engine_id == eid]
        assert mha and all(p.knobs.pack_gqa is False for p in mha), (eid, [p.knobs for p in mha])


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
@pytest.mark.parametrize(
    ("mxfp8", "d_qk", "d_v", "expected_cga"),
    # Per-tensor FP8 d128 runs its unsplit leg at cga1 (one 256-row CTA, the
    # geometry cuDNN's fp8 kernel uses; B200: llama causal S=2K 1.18x -> 1.14x,
    # AR-DiT no-split 1.07x -> 1.05x); MXFP8 d128 keeps the cga2 pair.
    [(False, 128, 128, 1), (True, 128, 128, 2), (False, 256, 256, 1), (True, 256, 256, 1)],
    ids=["per_tensor-d128", "block_scale-d128", "per_tensor-d256", "block_scale-d256"],
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
    unsplit = [plan for plan in plans if (plan.knobs.split_kv or 1) == 1]
    assert unsplit and {plan.knobs.cga for plan in unsplit} == {expected_cga}, [plan.knobs for plan in plans]
    if not mxfp8 and (d_qk, d_v) == (128, 128):
        # Per-tensor FP8 d128 offers both widths; the split leg stays on the cga2
        # pair (split_cgas_by_d_shape), so the plan list may carry both.
        assert {plan.knobs.cga for plan in plans} <= {1, 2}
    else:
        assert {plan.knobs.cga for plan in plans} == {expected_cga}

    spec = next(spec for spec in engines.ENGINE_SPECS if spec.name == name)
    assert engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(cga=expected_cga)) is None
    wrong_cga = 1 if expected_cga == 2 else 2
    if not mxfp8 and (d_qk, d_v) == (128, 128):
        assert engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(cga=wrong_cga)) is None, "per-tensor FP8 d128 builds at both widths"
    else:
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
def test_rubin_mxfp8_row_names_its_thd_and_paged_gap(sm107_metadata_target):
    """The cc 10.7 MXFP8 row claims THD (at d256 only, #1488's thd_d_shapes) AND paged pools (d128 / d256 with dense
    queries), so its former gap clause has no live case and is gone; every decline reads the clause that governs it:
    THD at another head dim the generic THD-shape clause, a THD query over pools the paged MXFP8 clause, a page that
    does not hold whole 128-row SF atoms the page-size clause.  The SM100 MXFP8 row, which serves THD and pools, is
    untouched."""
    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    rubin, sm100 = caps[engines.engine_name(mxfp8=True, arch="sm107")], caps[engines.engine_name(mxfp8=True)]
    thd_leg = "THD (ragged) rides the packed native-tile leg on this engine (shapes [(256, 256)]); the head-dim envelope is dense-only"
    quant = dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_mxfp8=True)
    thd = dict(thd=True, padded=True, **quant)
    paged = dict(has_paged_kv=True, padded=True, page_size=128, **quant)
    paged64 = dict(has_paged_kv=True, padded=True, page_size=64, **quant)
    assert engines.mismatch(rubin, _facts(device_cc=(10, 7), **thd)) == thd_leg, "THD at d128: the THD-shape clause"
    assert engines.mismatch(rubin, _facts(device_cc=(10, 7), d_qk=256, d_v=256, **thd)) is None, "THD at d256 is served (#1488)"
    assert engines.mismatch(rubin, _facts(device_cc=(10, 7), **paged)) is None, "page-128 pools with dense queries are served"
    assert engines.mismatch(rubin, _facts(device_cc=(10, 7), d_qk=256, d_v=256, thd=True, **paged)) == "paged MXFP8 KV with THD queries is not wired"
    assert engines.mismatch(rubin, _facts(device_cc=(10, 7), **paged64)) == "paged MXFP8 KV needs page_size to be a multiple of 128; got 64"
    assert engines.mismatch(rubin, _facts(device_cc=(10, 7), **quant)) is None, "dense BSHD stays admitted"
    assert engines.mismatch(sm100, _facts(**thd)) is None
    assert engines.mismatch(sm100, _facts(**paged)) is None


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_rubin_d256_mxfp8_masked_sets_stay_in_the_flavor_sched_domain(kind, sm107_metadata_target):
    """The measured D256 picker says LPT for a masked block-scale graph; the cc 10.7 MXFP8 row claims NATURAL only at
    (256, 256), so every proposed set must be clamped into that domain -- the FALLBACK block had no entry at all and
    the A block kept only its NATURAL runner (test_d256_quantized_primary_uses_measured_scheduler keeps the SM100 row's
    LPT, which that row's domain honours)."""
    name = engines.engine_name(mxfp8=True, arch="sm107")
    spec = next(s for s in engines.ENGINE_SPECS if s.name == name)
    facts = _facts(
        b=2,
        h_q=8,
        h_kv=2,
        s_q=4096,
        s_kv=4096,
        d_qk=256,
        d_v=256,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_mxfp8=True,
        device_cc=(10, 7),
        device_sm_count=216,
    )
    plans = recommend(kind, facts, {name: 20516})
    assert plans, f"{kind}: the row admits the graph (mismatch={engines.mismatch(spec.capabilities, facts)!r}) and must propose a set"
    assert all(engines.mismatch(spec.capabilities, facts, p.knobs) is None for p in plans)
    assert (plans[0].knobs.sched_policy, plans[0].knobs.cga, plans[0].knobs.split_kv) == (SCHED_NATURAL, 1, 1)


@pytest.mark.L0
def test_d128_mxfp8_causal_primary_uses_measured_scheduler():
    """sm100 d128 MXFP8, causal: the FIRST proposed policy is plain LPT and LPT_L2 stays in the ranking as the
    autotune runner (MEASURED on B200, 2026-09-22: LPT over the L2-budget arm's LPT_L2 +4.3..+10.4 % on six shapes,
    GQA 1 and 3, S=4K..32K, controls <= 0.35 %; O / Stats / Amax_O bitwise identical across policies).  The notch is
    an explicit oracle arm, not a domain effect (unlike d256: the d128 kernel serves LPT_L2), so the row's domain must
    still list all three.  Everything around it keeps the old proposal: the per-tensor FP8 d128 row measured
    shape-dependent and leads with LPT_L2; the sm100 d192x128 MXFP8 flavor and the Rubin d128 MXFP8 row are unmeasured
    and lead with LPT_L2 on a GQA graph; a mask-free d128 MXFP8 graph proposes NATURAL alone."""
    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
    from cudnn.sdpa.fwd.heuristics import _sched_points

    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    mx_name, fp8_name = engines.engine_name(mxfp8=True), engines.engine_name(fp8=True)
    quant = dict(s_q=8192, h_q=24, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16)
    for h_kv in (8, 24):  # GQA 3 and no GQA, both measured
        mx = _facts(is_mxfp8=True, h_kv=h_kv, **quant)
        assert engines.effective_sched_policies(caps[mx_name], mx) == frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}), "domain unchanged"
        assert _sched_points(caps[mx_name], mx) == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL], h_kv
        plans = recommend("A", mx, {mx_name: 20510})
        assert (plans[0].knobs.split_kv, plans[0].knobs.sched_policy) == (1, SCHED_LPT), plans[0].knobs
        assert SCHED_LPT_L2 in {p.knobs.sched_policy for p in plans}, "LPT_L2 must stay an autotune runner"
        # The per-tensor FP8 d128 row: 2 MiB per head here sits under
        # _SM100_D128_LPT_L2_MIN_BYTES, so it too leads with plain LPT (B200:
        # e4m3 64/64 S=2K cga1 LPT 1.19x vs LPT_L2 1.33x of cuDNN; 64/8 S=8K
        # packed equal); above 8 MiB per head it keeps the L2-budget arm.
        fp8 = _facts(is_fp8=True, h_kv=h_kv, **quant)
        assert _sched_points(caps[fp8_name], fp8) == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL], h_kv
        assert recommend("A", fp8, {fp8_name: 20501})[0].knobs.sched_policy == SCHED_LPT
        fp8_long = _facts(is_fp8=True, h_kv=h_kv, **{**quant, "s_q": 65536, "s_kv": 65536})
        assert _sched_points(caps[fp8_name], fp8_long) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL], h_kv
    # Scope: d128 only, SM100 row only, causal only.
    assert _sched_points(caps[mx_name], _facts(is_mxfp8=True, h_kv=8, d_qk=192, d_v=128, **quant)) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL]
    rubin_mx = caps[engines.engine_name(mxfp8=True, arch="sm107")]
    assert _sched_points(rubin_mx, _facts(is_mxfp8=True, h_kv=8, device_cc=(10, 7), **quant)) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL]
    assert _sched_points(caps[mx_name], _facts(is_mxfp8=True, h_kv=8, causal=False, **quant)) == [SCHED_NATURAL]


@pytest.mark.L0
def test_d512_mxfp8_primary_uses_measured_scheduler(sm107_metadata_target):
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
    """Placement is the SHARED layer's job (engines/heuristics._assemble): a
    list WITHOUT the BACKEND marker keeps the historical order (ours lead the
    backend's entries inside each mode block), the delegating entry never
    leads an OPENSOURCE block, one config repeated across blocks keeps its
    first position, and no final entry carries a mode."""
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
def test_assemble_places_the_backend_block_where_the_marker_sits():
    """The BACKEND marker: ``[BACKEND, ours]`` puts the delegating entry and the
    mode's backend entries ahead of ours; ``[ours, BACKEND]`` is the historical
    order; the marker itself never reaches the list, a repeat is dropped, an
    OPENSOURCE block ignores it (python-only + delegating), FALLBACK expands to
    the FALLBACK entries, and with no backend entries the block is empty."""
    from cudnn.engines.heuristics import BACKEND, is_backend_block

    ours = [PlanConfig(20500, "set-a"), PlanConfig(20500, "set-b")]
    backend = [
        PlanConfig(-1, None),
        PlanConfig(7, {"k": 1}, cpp_index=0, mode=cudnn.heur_mode.A),
        PlanConfig(8, {"k": 2}, cpp_index=1, mode=cudnn.heur_mode.FALLBACK),
    ]
    trail = _assemble([cudnn.heur_mode.A], lambda kind: [BACKEND] + ours, backend)
    assert [p.engine_id for p in trail] == [-1, 7, 20500, 20500]
    lead = _assemble([cudnn.heur_mode.A], lambda kind: ours + [BACKEND], backend)
    assert [p.engine_id for p in lead] == [20500, 20500, -1, 7]
    assert not any(is_backend_block(p) for p in trail + lead)
    twice = _assemble([cudnn.heur_mode.A], lambda kind: [BACKEND, ours[0], BACKEND, ours[1]], backend)
    assert [p.engine_id for p in twice] == [-1, 7, 20500, 20500]
    oss = _assemble([cudnn.heur_mode.OPENSOURCE], lambda kind: [BACKEND] + ours, backend)
    assert [p.engine_id for p in oss] == [20500, 20500, -1], "OPENSOURCE stays python-only + delegating"
    both = _assemble([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK], lambda kind: [BACKEND] + ours if kind == "A" else [BACKEND, ours[0]], backend)
    assert [p.engine_id for p in both] == [-1, 7, 20500, 20500, 8], "FALLBACK block expands to the FALLBACK entries; dedup keeps first positions"
    alone = _assemble([cudnn.heur_mode.A], lambda kind: [BACKEND] + ours, [])
    assert [p.engine_id for p in alone] == [20500, 20500], "no backend entries: the block is empty and ours stay"


@pytest.mark.L0
def test_fallback_kind_is_least_demanding():
    for p in recommend("FALLBACK", _facts(), _OFFERED):
        assert p.knobs.split_kv == 1
        assert p.knobs.sched_policy == 0  # SCHED_NATURAL


# ---------------------------------------------------------------------------
# d128 decode-shaped launches (FlashInfer paged GQA decode, MTP S_q in [2, 8])
# ---------------------------------------------------------------------------
#
# The SM100 f16 row's d128 flavor: cga1 is the decode tile (one 128-row CTA per
# (batch, packed head) unit, PR #1094), cga2 the prefill pipeline (512 rows per
# cluster). The rules under test read the unit's live rows S_q * PACK_G: the
# width (one decode tile), the causal scheduler (NATURAL first while one cga2
# cluster covers the unit) and the split model's true CTA count plus its two
# combine corrections. The measured B200 numbers live in the heuristics module
# next to the rules; test_sdpa_fwd_decode_d128_sm100.py pins the decode tile's
# own selection and kernel.


def _decode_facts(**over):
    """FlashInfer's paged GQA decode graph: b=32, 64/4 heads, d128 bf16, 4k KV."""
    base = dict(b=32, h_q=64, h_kv=4, s_q=1, s_kv=4096, dtype=cudnn.data_type.BFLOAT16, causal=False, padded=True, has_paged_kv=True, page_size=16)
    base.update(over)
    return _facts(**base)


def _f16_plans(facts):
    plans = [p for p in recommend("A", facts, _OFFERED) if p.engine_id == 20500]
    assert plans, "the f16 row must serve this graph"
    spec = next(s for s in engines.ENGINE_SPECS if s.name == _F16)
    assert all(engines.mismatch(spec.capabilities, facts, p.knobs) is None for p in plans)
    return plans


_DENSE = dict(has_paged_kv=False, padded=False, page_size=0)


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [
        dict(),  # S_q=1, G=16 packed: 16 live rows
        dict(h_kv=8),  # G=8
        dict(h_q=8, h_kv=8),  # MHA, nothing to pack: 1 live row
        dict(h_q=96, h_kv=8),  # G=12 does not divide the tile -> packs 4 of 12 (partial PackGQA): 4 live rows per packed head
        dict(s_q=4, causal=True, bottom_right=True),  # MTP
        dict(s_q=8),  # 8 * 16 = 128 rows: exactly one decode tile
        dict(**_DENSE),  # dense decode
        dict(s_q=8, causal=True, bottom_right=True, window_left=255, has_paged_kv=False, padded=True, page_size=0),  # dense padded MTP + SWA
    ],
    ids=["fi_64_4", "fi_64_8", "mha", "partial_pack_96_8", "mtp4_br", "one_tile_exactly", "dense_decode", "dense_mtp_swa"],
)
def test_d128_decode_shaped_launch_leads_with_cga1(over):
    """S_q * PACK_G <= 128: every set rides the decode tile (cga1 -- the only
    width in this band, as test_sdpa_fwd_decode_d128_sm100 pins) and the lead
    walks the plain scheduler even under a causal band (the one-cluster rule;
    the LPT variants stay behind it as runners for autotune, as on every
    causal graph). A pinned cga2 is still honored (test_d128_cga_request_domain)."""
    plans = _f16_plans(_decode_facts(**over))
    lead = plans[0].knobs
    assert lead.cga == 1, lead
    assert lead.sched_policy == 0, lead  # SCHED_NATURAL, even under a causal band
    assert all(p.knobs.cga == 1 for p in plans), [p.knobs for p in plans]


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [
        dict(s_q=129),  # 129 rows overflow one decode tile unpacked too (16 * 129 packed)
        dict(s_q=300, h_q=8, h_kv=8),  # paged prefill, MHA
        dict(s_q=4096, s_kv=4096, b=1, h_q=32, h_kv=8, causal=True, **_DENSE),  # prefill
    ],
    ids=["past_one_tile", "paged_prefill", "prefill_4k"],
)
def test_d128_prefill_shaped_launch_keeps_cga2(over):
    """Above one decode tile's rows on every leg the plan list is what it was:
    cga2 throughout. (A packed leg that overflows while its unpacked runner-up
    fits is test_sdpa_fwd_decode_d128_sm100's per-candidate case.)"""
    plans = _f16_plans(_decode_facts(**over))
    assert all(p.knobs.cga == 2 for p in plans), [p.knobs for p in plans]


@pytest.mark.L0
def test_d128_causal_scheduler_rule_stops_at_one_q_cluster():
    """The one-cluster NATURAL rule ends exactly where the LPT remaps gain rows
    to balance: at S_q * PACK_G <= 512 (one cga2 cluster) NATURAL leads with
    the LPT variants behind it as runners -- on the prefill tile here, past
    the decode tile's 128 rows -- and one row more restores the L2-budget
    rule's causal primary with its runners: plain LPT at this 2 MiB-per-head
    working set (_SM100_D128_LPT_L2_MIN_BYTES), LPT_L2 on a 32k cache; a 4k
    causal prefill is untouched."""
    one = _f16_plans(_decode_facts(s_q=32, causal=True, bottom_right=True))  # 32 * 16 = 512 rows
    assert one[0].knobs.sched_policy == 0, one[0].knobs
    assert {SCHED_LPT, SCHED_LPT_L2} <= {p.knobs.sched_policy for p in one}, [p.knobs for p in one]
    assert one[0].knobs.cga == 2, "512 packed rows are four decode tiles' worth: the prefill tile leads"
    two = _f16_plans(_decode_facts(s_q=33, causal=True, bottom_right=True))  # 528 rows: two clusters
    assert two[0].knobs.sched_policy == SCHED_LPT, two[0].knobs
    assert {SCHED_LPT_L2, 0} <= {p.knobs.sched_policy for p in two}, [p.knobs for p in two]
    two_long = _f16_plans(_decode_facts(s_q=33, s_kv=32768, causal=True, bottom_right=True))  # 16 MiB per head
    assert two_long[0].knobs.sched_policy == SCHED_LPT_L2, two_long[0].knobs
    prefill = _f16_plans(_decode_facts(s_q=4096, s_kv=4096, b=1, h_q=32, h_kv=8, causal=True, **_DENSE))
    assert (prefill[0].knobs.sched_policy, prefill[0].knobs.cga) == (SCHED_LPT, 2), prefill[0].knobs


@pytest.mark.L0
def test_d128_small_batch_units_split_only_where_the_combine_is_cheap():
    """On the decode tile the wave model sees the true CTA count, so a small
    batch splits finer than it did at cga2 -- right where the combine is one
    wave, wrong where the output rows make it many: the unsplit leg runs no
    combine and must not be charged one. Pinned to the B200 kernel times in
    the heuristics module: b=8 h=32/8 S_q=64 paged bottom-right (256 rows per
    unit -- one cga2 cluster, the prefill tile -- 16384 combine rows) and its
    dense causal twin lead UNSPLIT on the plain scheduler (their split 2
    measured 11% slower); S_q=16 at the same batch (64 rows: the decode tile,
    4096 combine rows) leads with the split that pays, no-split reachable
    behind it; the b=8 h=64/4 decode keeps its four splits; and the dense b=4
    h=32/8 S_q=128 causal launch -- a cga2 unit -- stops proposing the split 2
    that measured 29% slower than unsplit."""
    chunk = dict(b=8, h_q=32, h_kv=8, s_q=64, causal=True)
    for facts in (_decode_facts(bottom_right=True, **chunk), _decode_facts(**chunk, **_DENSE)):
        lead = _f16_plans(facts)[0].knobs
        assert (lead.cga, lead.split_kv, lead.pack_gqa, lead.sched_policy) == (2, 1, True, 0), lead
    short = _f16_plans(_decode_facts(bottom_right=True, **{**chunk, "s_q": 16}))
    assert (short[0].knobs.cga, short[0].knobs.split_kv) == (1, 2), short[0].knobs
    assert any(p.knobs.cga == 1 and p.knobs.split_kv == 1 for p in short), [p.knobs for p in short]
    decode = _f16_plans(_decode_facts(b=8))[0].knobs
    assert (decode.cga, decode.split_kv) == (1, 4), decode
    two_ctas = _f16_plans(_decode_facts(b=4, h_q=32, h_kv=8, s_q=128, causal=True, **_DENSE))
    assert two_ctas[0].knobs.cga == 2, two_ctas[0].knobs
    assert all(p.knobs.split_kv == 1 for p in two_ctas), [p.knobs for p in two_ctas]


@pytest.mark.L0
def test_d128_few_unit_long_kv_splits_to_the_combine_latency_floor():
    """b=1 with a handful of KV heads: the whole launch is a few decode-tile
    CTAs, so the split leg fills the wave with partials -- and once each split is a few
    KV tiles, the combine's serial walk over them costs more than the loop
    saves. The wave model prices that walk at a lone block's latency
    (choose_split_kv's COMBINE_FLOOR), so the lead stops one power of two short
    of the full wave where that is what measures (B200, paged bf16, kernel
    time, heuristic lead against the split one step finer): h=16/2 S_kv=32k
    32 splits at 39.7 us (64: 50.8); h=64/4 S_kv=4k 8 at 20.3 us (16: 22.3),
    the same at S_q=4 MTP (25.4 vs 28.3) and for h=32/8 (20.3 vs 23.6). Where
    the wave boundary already stops the split the lead is what it was:
    h=64/4 S_kv=32k 32 splits (43.0 us; the base cga2 split 16: 48.5), b=4
    h=64/4 S_kv=4k 8 (21.5 vs base 26.9) and b=8 4 (26.6 vs base 38.9)."""

    def lead(**over):
        knobs = _f16_plans(_decode_facts(**{"b": 1, **over}))[0].knobs
        assert knobs.cga == 1 and knobs.pack_gqa is True and knobs.sched_policy == 0, knobs
        return knobs.split_kv

    assert lead(h_q=16, h_kv=2, s_kv=32768) == 32
    assert lead(h_q=64, h_kv=4, s_kv=4096) == 8
    assert lead(h_q=64, h_kv=4, s_kv=4096, s_q=4, causal=True, bottom_right=True) == 8
    assert lead(h_q=32, h_kv=8, s_kv=4096) == 8
    assert lead(h_q=64, h_kv=4, s_kv=32768) == 32
    assert lead(b=4, h_q=64, h_kv=4, s_kv=4096) == 8
    assert lead(b=8, h_q=64, h_kv=4, s_kv=4096) == 4


@pytest.mark.L0
def test_unsplit_leg_accounting_moves_the_wide_head_flavors_too():
    """The unsplit leg paying no combine (choose_split_kv) is shared by every
    flavor's split leg, so the d192x128 and d256 f16 leads it moves belong to
    this change: few-unit 2k-KV chunks with thousands of output rows, whose
    split 2 the phantom combine wave-set used to buy. Measured B200 bf16 dense
    (heuristic lead against the old split 2 pinned): d256 b=1 h=64/4 S_q=128
    mask-free 51.7 vs 75.3 us; d192 b=1 h=32/8 S_q=256 causal 35.3 vs 60.9 us.
    Both stay on cga2 -- the cga1 rule is d128's alone -- and neither list
    proposes a split any more."""
    wide = dict(b=1, s_kv=2048, **_DENSE)
    d256 = _f16_plans(_decode_facts(d_qk=256, d_v=256, h_q=64, h_kv=4, s_q=128, **wide))
    d192 = _f16_plans(_decode_facts(d_qk=192, d_v=128, h_q=32, h_kv=8, s_q=256, causal=True, **wide))
    for plans in (d256, d192):
        assert (plans[0].knobs.cga, plans[0].knobs.split_kv) == (2, 1), plans[0].knobs
        assert all(p.knobs.split_kv == 1 for p in plans), [p.knobs for p in plans]


@pytest.mark.L0
def test_d128_cga_request_domain():
    """The f16 row admits cga1 AND cga2 on d128, split or not (cga1 is the
    decode tile, validated split and unsplit); a width the flavor has no
    configuration for is declined, as is cga1 on the f16 d256 flavor, whose
    kernel mandates CTA2. The fp8 d128 row keeps cga2 only
    (test_quantized_cga_follows_selected_native_flavor pins that side)."""
    spec = next(s for s in engines.ENGINE_SPECS if s.name == _F16)
    for facts in (_decode_facts(), _decode_facts(s_q=4096, s_kv=4096, b=1, h_q=32, h_kv=8, causal=True, **_DENSE)):
        for cga in (1, 2):
            assert engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(cga=cga)) is None
            assert engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(cga=cga, split_kv=2)) is None
        assert "outside this engine's domain" in engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(cga=4))
    d256 = _decode_facts(d_qk=256, d_v=256)
    assert engines.mismatch(spec.capabilities, d256, engines.SdpaFwdKnobs(cga=2)) is None
    assert "outside this engine's domain" in engines.mismatch(spec.capabilities, d256, engines.SdpaFwdKnobs(cga=1))


@pytest.mark.L0
def test_d128_width_rule_is_one_rule_for_graph_and_adapter():
    """select_d128_auto_cga is the rule behind both the graph path
    (_d128_decode_tile_fits derives pack_g from the candidate's packing) and
    the standalone adapter's default width (api_dsl.SdpaFwdDslSm100, which
    derives it from its pack_gqa request through pack_gqa_group_size): one
    decode tile's rows decide (cga_tile_m(128, 1) == _D128_DECODE_TILE_ROWS);
    a ragged graph keeps cga2 unless it is the decode tile's ragged-Q leg.
    (The adapter's cga DOMAIN is #1094's
    test_standalone_cga_domain_admits_cga1_on_d128_f16_only.)"""
    from cudnn.sdpa.fwd.config_sm100 import cga_tile_m, pack_gqa_group_size
    from cudnn.sdpa.fwd.heuristics import _D128_DECODE_TILE_ROWS, _d128_decode_tile_fits, select_d128_auto_cga

    assert cga_tile_m(128, 1) == _D128_DECODE_TILE_ROWS == 128
    assert cga_tile_m(128, 2) == 512
    caps = next(s for s in engines.ENGINE_SPECS if s.name == _F16).capabilities
    cases = (
        (1, 64, 4, True),
        (8, 64, 4, True),
        (9, 64, 4, True),
        (9, 64, 4, False),
        (32, 96, 8, True),
        (33, 96, 8, True),
        (128, 8, 8, False),
        (129, 8, 8, False),
    )
    for s_q, h_q, h_kv, packed in cases:
        facts = _decode_facts(s_q=s_q, h_q=h_q, h_kv=h_kv)
        pack_g = pack_gqa_group_size(h_q // h_kv, 128, partial=True) if packed else 1
        want = 1 if _d128_decode_tile_fits(caps, facts, packed) else 2
        assert select_d128_auto_cga(s_q=s_q, pack_g=pack_g, thd=False) == want, (s_q, h_q, h_kv, packed, want)
        assert want == (1 if s_q * pack_g <= 128 else 2), (s_q, h_q, h_kv, packed, want)
    assert select_d128_auto_cga(s_q=1, pack_g=1, thd=True) == 2
    assert select_d128_auto_cga(s_q=1, pack_g=16, thd=True, thd_decode_leg=True) == 1
    assert select_d128_auto_cga(s_q=129, pack_g=1, thd=True, thd_decode_leg=True) == 2


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
    return g, (q, k, v, o, st), (B, H, SQ, SKV, D)


def _append_explicit_f16_plan(g, *, split_kv):
    engine = next(e for e in manifest.engines_for(g) if e.name == _F16)
    knobs = {
        cudnn.knob_type.TILE_M: 128,
        cudnn.knob_type.TILE_N: 128,
        cudnn.knob_type.TILE_CGA_M: 1,
        cudnn.knob_type.SCHED_POLICY: 0,
        cudnn.knob_type.SPLIT_KV: split_kv,
        cudnn.knob_type.PACK_GQA: 0,
    }
    g.create_execution_plan(engine.engine_id, knobs)
    index = g.get_execution_plan_count() - 1
    assert g.get_engine_and_knobs_at_index(index) == (engine.engine_id, knobs)
    return index


@pytest.mark.L1
@pytest.mark.skipif(not (_is_sm100() and _dsl_available()), reason="needs an SM100 device and nvidia-cutlass-dsl")
def test_explicit_split_kv_plan_matches_reference_and_roundtrips():
    """Issue F-2 regression: the split plan is graph-reachable, carves its
    slabs from the caller workspace, and recombines exactly."""
    g, (q, k, v, o, st), (B, H, SQ, SKV, D) = _build_decodeish_graph(causal=False)
    split_idx = _append_explicit_f16_plan(g, split_kv=2)
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
    assert knobs[cudnn.knob_type.SPLIT_KV] == 2
    g2, (q2, k2, v2, o2, st2), _ = _build_decodeish_graph(causal=False)
    g2.create_execution_plan(eng_id, knobs)
    replay_idx = g2.get_execution_plan_count() - 1
    assert g2.get_engine_and_knobs_at_index(replay_idx) == (eng_id, knobs)
    g2.select_plan(replay_idx)
    g2.check_support()
    g2.build_plans()
    replay_o, replay_st = torch.empty_like(o_gpu), torch.empty_like(st_gpu)
    replay_ws = torch.empty(g2.get_workspace_size(), device="cuda", dtype=torch.uint8)
    g2.execute({q2: q_gpu, k2: k_gpu, v2: v_gpu, o2: replay_o, st2: replay_st}, replay_ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_o, o_gpu, atol=0, rtol=0)
    torch.testing.assert_close(replay_st, st_gpu, atol=0, rtol=0)


@pytest.mark.L1
@pytest.mark.skipif(not (_is_sm100() and _dsl_available()), reason="needs an SM100 device and nvidia-cutlass-dsl")
def test_explicit_natural_unsplit_plan_builds_and_matches_reference():
    """A pinned scheduler is honored regardless of the recommendation list."""
    g, (q, k, v, o, st), (B, H, SQ, SKV, D) = _build_decodeish_graph()
    nat_idx = _append_explicit_f16_plan(g, split_kv=1)
    g.select_plan(nat_idx)
    g.check_support()
    g.build_plans()
    eng_id, knobs = g.get_engine_and_knobs_at_index(nat_idx)
    assert knobs[cudnn.knob_type.SCHED_POLICY] == 0 and knobs[cudnn.knob_type.SPLIT_KV] == 1

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
def test_heuristics_never_propose_split_or_pack_for_a_gated_graph(sm107_metadata_target):
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
    # Half nonpaged Rubin graphs cannot pack even without a gate. Use the
    # FP8 row with its D256 packing restriction relaxed for this paired probe.
    fp8 = dict(gated, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_fp8=True, epilogue_gate_dtype=cudnn.data_type.BFLOAT16)
    pack_row = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _RUBIN_FP8)
    pack_row = dataclasses.replace(pack_row, pack_gqa_d_shapes=None)
    pack_facts = _facts(h_q=8, h_kv=2, **fp8)
    assert _pack_gqa_eligible(pack_row, dataclasses.replace(pack_facts, has_epilogue_gate=False), 128) is True, "the control must pack"
    assert _pack_gqa_eligible(pack_row, pack_facts, 128) is False

    # The real rows: every emitted set is unsplit and unpacked, and admissible.
    for facts in (_facts(**gated), _facts(h_q=8, h_kv=2, **gated), _facts(causal=False, **gated)):
        plans = recommend("A", facts, _RUBIN_OFFERED)
        assert plans, "the Rubin f16 row serves the gated d256 graph"
        assert {p.engine_id for p in plans} == {_RUBIN_OFFERED[_RUBIN_F16]}
        for p in plans:
            assert (p.knobs.split_kv or 1) == 1 and not p.knobs.pack_gqa, p.knobs
            spec = next(s for s in engines.ENGINE_SPECS if _RUBIN_OFFERED.get(s.name) == p.engine_id)
            assert engines.mismatch(spec.capabilities, facts, p.knobs) is None
    plans = recommend("A", _facts(h_q=8, h_kv=2, **fp8), _RUBIN_OFFERED)
    assert plans and {p.engine_id for p in plans} == {_RUBIN_OFFERED[_RUBIN_FP8]}
    assert all((p.knobs.split_kv or 1) == 1 and not p.knobs.pack_gqa for p in plans), [p.knobs for p in plans]
    # ...and a gated graph on a flavor that does not carry the gate proposes nothing at all.
    assert not recommend("A", _facts(**dict(gated, d_qk=128, d_v=128)), _RUBIN_OFFERED)
    # The ONE exception: a DECODE-shaped gated half graph (S_q x G <= 16 packed rows) splits on the d256 decode
    # tile, whose combine applies the gate -- the split proposal LEADS, floored at 2, packed; an unsplit set stays
    # unpacked (the prefill kernel's fused epilogue) and exists on a dense cache only.  The decode suite pins the
    # full contract; here the proposal helpers and the admissibility of every emitted set.
    decode = dict(
        gated,
        dtype=cudnn.data_type.BFLOAT16,
        epilogue_gate_dtype=cudnn.data_type.BFLOAT16,
        s_q=1,
        s_kv=4096,
        b=4,
        h_q=24,
        h_kv=2,
        causal=False,
        device_sm_count=204,
    )
    for cache in (dict(has_paged_kv=True, page_size=16, padded=True), dict(padded=False)):
        f = _facts(**decode, **cache)
        assert _pack_gqa_eligible(row, f, 128, 2) and not _pack_gqa_eligible(row, f, 128, 1) and _pack_gqa_eligible(row, f, 128), cache  # None = some plan
        points = _split_points(row, f, 128, 128, 2, pack_g=12)
        assert points[0] >= 2 and (1 in points) == (not f.has_paged_kv), (cache, points)
        plans = recommend("A", f, _RUBIN_OFFERED)
        assert plans and plans[0].knobs.pack_gqa is True and plans[0].knobs.split_kv >= 2, [p.knobs for p in plans]
        for p in plans:
            assert engines.mismatch(row, f, p.knobs) is None, (cache, p.knobs, engines.mismatch(row, f, p.knobs))
            assert not (p.knobs.pack_gqa and (p.knobs.split_kv or 1) == 1), p.knobs
        assert any((p.knobs.split_kv or 1) == 1 for p in plans) == (not f.has_paged_kv), [p.knobs for p in plans]


@pytest.mark.L0
def test_rubin_gated_decode_tile_split_declines_execute_time_shape_overrides(sm107_metadata_target):
    """The gate-in-combine split binds G to the plan's declared (B, H_q, S_q, D_v) (prepared.CombineGate), while a graph
    that permits execute-time geometry (shape overrides) may change either at execute: the row never serves such a graph
    on the decode tile's split -- the facts-level predicate says so (one definition for mismatch, the heuristics and the
    adapter), mismatch names the overrides, the lowering-side admission agrees, and the proposal helpers emit no split or
    packed set for it (its unsplit plan keeps the fused-gate prefill kernel).  The same graph WITHOUT overrides rides the
    tile's split -- the control."""
    import dataclasses

    from cudnn.sdpa.fwd.engines import _prepared_decline_reason, d256_decode_tile_selected
    from cudnn.sdpa.fwd.heuristics import _pack_gqa_eligible, _split_points

    row = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _RUBIN_F16)
    gated = _decode_d256_facts(
        device_cc=(10, 7),
        device_sm_count=204,
        b=4,
        h_q=24,
        has_epilogue_gate=True,
        epilogue_gate_dtype=cudnn.data_type.BFLOAT16,
        has_paged_kv=False,
        page_size=0,
        padded=False,
    )
    assert d256_decode_tile_selected(row, gated, 12, 2) and engines.mismatch(row, gated, engines.SdpaFwdKnobs(pack_gqa=True, split_kv=2)) is None, "the control"
    assert _split_points(row, gated, 128, 128, 2, pack_g=12)[0] >= 2 and _pack_gqa_eligible(row, gated, 128, 2), "the control proposes the gated split"
    over = dataclasses.replace(gated, shape_overrides=True)
    assert not d256_decode_tile_selected(row, over, 12, 2) and not d256_decode_tile_selected(row, over, 1, 2) and not d256_decode_tile_selected(row, over, 12)
    why = engines.mismatch(row, over, engines.SdpaFwdKnobs(split_kv=2))
    assert why and "shape overrides" in why, why
    why = engines.mismatch(row, over, engines.SdpaFwdKnobs(pack_gqa=True, split_kv=2))
    assert why and "decode tile" in why, why  # the packed request falls at the row's PackGQA rule first (the tile is not selected)
    why = _prepared_decline_reason(row, over, 2)
    assert why and "shape overrides" in why, why
    assert "shape overrides" not in (_prepared_decline_reason(row, over, 1) or ""), "the unsplit question is not this decline"
    assert _split_points(row, over, 128, 128, 2, pack_g=12) == [1], _split_points(row, over, 128, 128, 2, pack_g=12)
    assert not _pack_gqa_eligible(row, over, 128, 2) and not _pack_gqa_eligible(row, over, 128)


# --- cc 10.7 dense d128 half: the shared decode tile and PackGQA (issue #1472) --------------------------------------


def _sm107_d128_facts(**over):
    """Issue #1472's verify cell on cc 10.7: dense bf16 d128 64/8 at S_q 8 over a 2056-token cache, bottom-right
    causal, an attention sink, 216 SMs (the board the levers were measured on)."""
    base = dict(
        b=128,
        h_q=64,
        h_kv=8,
        s_q=8,
        s_kv=2056,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        causal=True,
        bottom_right=True,
        has_sink=True,
        device_cc=(10, 7),
        device_sm_count=216,
    )
    base.update(over)
    return _facts(**base)


def _sm107_f16_plans(facts):
    """The cc 10.7 half row's proposals for ``facts``, each re-admitted by mismatch() (honored-or-never-listed)."""
    plans = [p for p in recommend("A", facts, _RUBIN_OFFERED) if p.engine_id == _RUBIN_OFFERED[_RUBIN_F16]]
    assert plans, "the cc 10.7 half row must serve this graph"
    spec = next(s for s in engines.ENGINE_SPECS if s.name == _RUBIN_F16)
    assert all(engines.mismatch(spec.capabilities, facts, p.knobs) is None for p in plans), [p.knobs for p in plans]
    return plans


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [dict(), dict(h_kv=4, s_q=4, s_kv=2052), dict(s_q=1, causal=False, bottom_right=False), dict(s_q=16, h_kv=16), dict(padded=True), dict(h_q=8, h_kv=8)],
    ids=["issue_64_8_q8", "issue_64_4_q4", "plain_decode_q1", "q16_g4_one_tile", "padded_mtp", "mha_q8"],
)
def test_sm107_d128_decode_shaped_sets_ride_the_decode_tile(sm107_metadata_target, over):
    """S_q * pack_g <= 128 on the cc 10.7 half row (issue #1472): every proposed set runs the shared decode tile (cga1 --
    the tile-fit contract, as test_sdpa_fwd_decode_d128_sm100 pins on the SM100 row), none splits (the sink), a GQA
    graph lists BOTH packings (admission; which leads is the ranking's business), an MHA graph none packed, and a causal
    graph lists NATURAL and LPT both (which leads follows the measured tables,
    test_sm107_shared_d128_scheduler_follows_the_measured_tables; both reachable for autotune)."""
    facts = _sm107_d128_facts(**over)
    plans = _sm107_f16_plans(facts)
    assert all(p.knobs.cga == 1 for p in plans), [p.knobs for p in plans]
    assert all((p.knobs.split_kv or 1) == 1 for p in plans), [p.knobs for p in plans]
    packings = {bool(p.knobs.pack_gqa) for p in plans}
    assert packings == ({True, False} if facts.h_q != facts.h_kv else {False}), [p.knobs for p in plans]
    scheds = {p.knobs.sched_policy for p in plans}
    assert 0 in scheds, scheds  # SCHED_NATURAL
    if facts.causal:
        assert SCHED_LPT in scheds, scheds


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [512, 129], ids=["q512", "q129"])
def test_sm107_d128_prefill_shaped_sets_keep_cga2(sm107_metadata_target, s_q):
    """Above one decode tile's rows on every leg (129 rows overflow the unpacked tile too) the cc 10.7 half row keeps the
    cga2 prefill width throughout and still lists a packed set (the shared SM100 prefill body) next to the unpacked
    Rubin body."""
    plans = _sm107_f16_plans(_sm107_d128_facts(b=1, h_kv=4, s_q=s_q, s_kv=max(512, s_q)))
    assert all(p.knobs.cga == 2 for p in plans), [p.knobs for p in plans]
    assert {bool(p.knobs.pack_gqa) for p in plans} == {True, False}, [p.knobs for p in plans]


@pytest.mark.L0
def test_sm107_d128_prefolded_graphs_keep_the_native_body(sm107_metadata_target):
    """The pre-folded scale (attn_scale_prefolded) is an arm of the Rubin prefill body only -- the shared SM100 bodies
    apply the scale in-kernel -- so a pre-folded dense d128 graph keeps the cga2 domain, proposes no packed set, and an
    explicit cga=1 or PackGQA pin is a typed decline (the heuristics never propose them; the cc 10.7 lever sweeps'
    FLOAT+fold cases keep their plan)."""
    from cudnn.sdpa.fwd.engines import effective_cgas

    spec = next(s for s in engines.ENGINE_SPECS if s.name == _RUBIN_F16)
    facts = _sm107_d128_facts(attn_scale_prefolded=True)
    assert effective_cgas(spec.capabilities, facts) == frozenset({2})
    plans = _sm107_f16_plans(facts)
    assert all(p.knobs.cga == 2 and not p.knobs.pack_gqa for p in plans), [p.knobs for p in plans]
    assert "outside this engine's domain" in engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(cga=1))
    assert "pre-folded" in engines.mismatch(spec.capabilities, facts, engines.SdpaFwdKnobs(pack_gqa=True))


@pytest.mark.L0
def test_sm107_d128_cga_and_pack_admission(sm107_metadata_target):
    """Explicit pins on the cc 10.7 half row (issue #1472): dense d128 half admits cga=1 (the shared decode tile) and
    cga=2, and PackGQA for every group that divides the 128-row tile (4 / 8 / 16; the d64 envelope too); a group that
    does not (96/8) is declined with the divisibility reason, dense d256 packs on the d256 decode tile within its route
    (S_q x G within two token units of the 32-column tile) and keeps its paged-THD-only PackGQA past it, THD nonpaged
    keeps the cga2 prefill pipeline (cga=1 outside its domain, no packed leg), and dense PAGED queries -- not wired on
    cc 10.7 outside the d256 decode tile -- never see the d128 tile."""
    from cudnn.sdpa.fwd.engines import effective_cgas

    caps = next(s for s in engines.ENGINE_SPECS if s.name == _RUBIN_F16).capabilities
    pack = engines.SdpaFwdKnobs(pack_gqa=True)
    for cga in (1, 2):
        assert engines.mismatch(caps, _sm107_d128_facts(), engines.SdpaFwdKnobs(cga=cga)) is None, cga
    for h_kv in (16, 8, 4):
        assert engines.mismatch(caps, _sm107_d128_facts(h_kv=h_kv), pack) is None, h_kv
    assert engines.mismatch(caps, _sm107_d128_facts(d_qk=64, d_v=64), pack) is None
    assert "divide" in engines.mismatch(caps, _sm107_d128_facts(h_q=96, h_kv=8), pack)
    # dense d256 (64/8 at S_q 8 = 64 packed rows = two token units of the 32-column tile): the d256 decode tile packs the whole
    # group, so the request is ADMITTED; past the route (S_q 16 = 128 rows) the dense d256 prefill kernel runs unpacked and the
    # packed request is a typed decline naming the decode tile among the routes
    assert engines.mismatch(caps, _sm107_d128_facts(d_qk=256, d_v=256), pack) is None
    past_route = engines.mismatch(caps, _sm107_d128_facts(d_qk=256, d_v=256, s_q=16), pack)
    assert past_route is not None and "decode tile" in past_route, past_route
    thd = _sm107_d128_facts(thd=True, padded=True)
    assert "outside this engine's domain" in engines.mismatch(caps, thd, engines.SdpaFwdKnobs(cga=1))
    assert engines.mismatch(caps, thd, pack) is not None
    assert 1 not in effective_cgas(caps, _sm107_d128_facts(has_paged_kv=True, page_size=16, padded=True))


@pytest.mark.L0
def test_sm107_d128_sink_free_decode_sets_are_admissible(sm107_metadata_target):
    """Sink-free small-batch long-KV decode on cc 10.7 (b=1, 64/8, S_q 1, 32k keys, no mask): the proposals ride the
    decode tile, a packed set is among them, and every set -- the split ones included (the shared dense combine on
    cc 10.7) -- passes mismatch().  The split COUNT is the wave model's choice and is not asserted."""
    facts = _sm107_d128_facts(b=1, s_q=1, s_kv=32768, causal=False, bottom_right=False, has_sink=False)
    plans = _sm107_f16_plans(facts)
    assert all(p.knobs.cga == 1 for p in plans), [p.knobs for p in plans]
    assert any(p.knobs.pack_gqa for p in plans), [p.knobs for p in plans]


@pytest.mark.L0
@pytest.mark.parametrize(
    "over, first_sched",
    [
        (dict(), SCHED_LPT),
        (dict(b=1, s_q=1, s_kv=32768), SCHED_LPT),
        (dict(b=32, s_q=16, s_kv=8192), SCHED_LPT),
        (dict(b=8, s_q=1, s_kv=1024), 0),
        (dict(b=8, s_q=4, s_kv=4096, d_qk=64, d_v=64, window_left=128), 0),
        (dict(h_kv=4, s_q=16, s_kv=2048), 0),
        (dict(b=2, s_q=4096, s_kv=4096, window_left=1024), 0),
        (dict(b=2, s_q=4096, s_kv=4096, window_left=1024, has_sink=False), 0),
    ],
    ids=[
        "issue_cell_lpt",
        "long_cache_b1_lpt",
        "q16_g8_one_tile_lpt",
        "short_cache_natural",
        "swa_decode_natural",
        "packed_cga2_band_natural",
        "swa_prefill_natural",
        "swa_prefill_sink_free_natural",
    ],
)
def test_sm107_shared_d128_scheduler_follows_the_measured_tables(sm107_metadata_target, over, first_sched):
    """The scheduler lead of the cc 10.7 shared dense d128 legs (issue #1472's tables, heuristics._SM107_DECODE_TILE_LPT_MIN_KV
    and the two cc 10.7 arms of _sched_points): the packed decode tile walks plain LPT first from a 2k cache on; a short
    cache, a sliding window (on the decode tile and on the packed cga2 prefill body alike) and the one-cluster packed cga2
    band keep NATURAL first.  The other policy is always listed as a runner and every proposed set stays admissible; the
    SM100 row's one-cluster NATURAL lead on the same decode shape is untouched."""
    facts = _sm107_d128_facts(**over)
    plans = _sm107_f16_plans(facts)
    first = plans[0].knobs
    assert first.pack_gqa is True, first
    assert first.sched_policy == first_sched, [p.knobs for p in plans]
    other = SCHED_LPT if first_sched == 0 else 0
    assert any(p.knobs.sched_policy == other for p in plans), [p.knobs for p in plans]
    sm100 = _facts(
        b=32, h_q=64, h_kv=4, s_q=4, s_kv=4096, dtype=cudnn.data_type.BFLOAT16, causal=True, bottom_right=True, padded=True, has_paged_kv=True, page_size=16
    )
    sm100_plans = [p for p in recommend("A", sm100, {_F16: 20500}) if p.engine_id == 20500]
    assert sm100_plans and sm100_plans[0].knobs.sched_policy in (None, 0), sm100_plans[0].knobs


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
    # A caller that declared CUDA-graph replay (pygraph(is_cuda_graph_replay_expected=True)
    # -> facts.cuda_graph_replay) pays the second launch once at capture: the
    # captured optimum LEADS and the eager-safe choice follows as a runner-up
    # (after the tile / scheduler / packing runners, like any later split
    # point). The saturated, small-batch and long-cache verdicts do not move
    # (both models agree there already).
    replay = sets(cuda_graph_replay=True)
    assert replay[0].split_kv == 2 and replay[0].pack_gqa is True, replay[0]
    assert any(k.split_kv == 1 and k.pack_gqa is True for k in replay[1:]), replay
    assert [k.split_kv for k in replay if k.split_kv > 1] == [2], replay
    assert sets(b=8, cuda_graph_replay=True)[0].split_kv == 8
    assert all(k.split_kv == 1 for k in sets(b=128, cuda_graph_replay=True))
    assert sets(s_kv=16384, cuda_graph_replay=True)[0].split_kv == 2
    mtp16r = sets(h_q=16, s_q=2, causal=True, bottom_right=True, cuda_graph_replay=True)
    assert mtp16r[0].split_kv == 2 and any(k.split_kv == 1 and k.pack_gqa is True for k in mtp16r[1:]), mtp16r
    # The prefill tile's shapes have no launch term to waive: the hint moves nothing.
    assert all(k.split_kv == 1 for k in sets(s_q=3, cuda_graph_replay=True))


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


@pytest.mark.L0
def test_rubin_decode_tile_packs_the_whole_group_and_takes_the_decode_split_model(sm107_metadata_target):
    """The Rubin half row's twin of the two SM100 decode-tile tests above, on cc 10.7 at 204 SMs:
    the row claims PackGQA at d256 for the decode tile ONLY (pack_gqa_d_shapes carries (256, 256);
    the d256 prefill kernel runs unpacked), so a decode-shaped graph -- paged OR dense, the dense
    cache no longer excluded -- leads PACKED with the decode split model's eager-safe choice (the
    captured optimum under cuda_graph_replay), every emitted set admissible by mismatch; the 24/2
    geometry (G = 12, which no prefill tile can pack: 128 % 12 != 0 and the Rubin row has no partial
    form) packs the WHOLE group, its group read as 12 so the model sees 2 x B units; the MTP steps
    ride the 32-column tile on this row (config_sm107.decode_d256_q_tile): 32/2 at S_q = 2 (32 rows)
    packs in ONE unit, 24/2 at S_q = 4 (48 rows) in TWO token units of two tokens, so the model sees
    2 x 2 x B units streaming the KV range -- the split picks follow; past the route (24/2 at S_q = 5:
    three units) there is no packed set and a packed request is a typed decline; a prefill-shaped
    dense d256 graph stays unpacked / unsplit; the padded dense cache and the sink keep the shared
    no-split rules."""
    from cudnn.sdpa.fwd.heuristics import _d256_decode_tile_selected, _pack_gqa_eligible, _pack_gqa_group

    row = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _RUBIN_F16)
    assert (256, 256) in row.pack_gqa_d_shapes and (256, 256) in row.split_d_shapes

    def facts(**over):
        return _decode_d256_facts(device_cc=(10, 7), device_sm_count=204, **over)

    def sets(**over):
        f = facts(**over)
        out = [p.knobs for p in recommend("A", f, _RUBIN_OFFERED) if p.engine_id == _RUBIN_OFFERED[_RUBIN_F16]]
        assert out, over
        for k in out:
            assert engines.mismatch(row, f, k) is None, (over, k, engines.mismatch(row, f, k))
        return out

    serving = sets()
    assert serving[0].pack_gqa is True and serving[0].split_kv == 1, serving[0]
    assert [k.split_kv for k in serving if k.split_kv > 1] == [2], serving
    replay = sets(cuda_graph_replay=True)
    assert replay[0].pack_gqa is True and replay[0].split_kv == 2 and any(k.split_kv == 1 and k.pack_gqa is True for k in replay[1:]), replay
    small = sets(b=8)
    assert small[0].pack_gqa is True and small[0].split_kv == 8 and any(k.split_kv == 1 for k in small[1:]), small
    assert sets(b=3)[0].split_kv == 16
    assert all(k.split_kv == 1 for k in sets(b=128))
    # The 24/2 geometry: the whole group, dense or paged.
    q24 = facts(h_q=24, b=3, s_kv=1000)
    assert _pack_gqa_eligible(row, q24, 128) and _pack_gqa_group(row, q24, 128, True) == 12
    assert _d256_decode_tile_selected(row, q24, 12) and _d256_decode_tile_selected(row, facts(h_q=24, s_q=2), 12)  # S_q = 2: the 32-column tile
    assert _d256_decode_tile_selected(row, facts(h_q=24, s_q=4), 12) and not _d256_decode_tile_selected(row, facts(h_q=24, s_q=5), 12)
    assert sets(h_q=24, b=3, s_kv=1000)[0].pack_gqa is True
    dense_padded = sets(h_q=24, b=3, s_kv=1000, has_paged_kv=False, page_size=0)
    assert dense_padded[0].pack_gqa is True and all(k.split_kv == 1 for k in dense_padded), dense_padded
    dense_unpadded = sets(h_q=24, b=3, s_kv=1024, has_paged_kv=False, page_size=0, padded=False)
    assert dense_unpadded[0].pack_gqa is True and any(k.split_kv > 1 for k in dense_unpadded), dense_unpadded
    # MTP: 16 rows pack and follow the serving policy; 32 rows (32/2 at S_q = 2) pack on the 32-column tile
    # in one unit (the Rubin route), 48 rows (24/2 at S_q = 4) in two TOKEN UNITS the model counts as streams.
    mtp16 = sets(h_q=16, s_q=2, causal=True, bottom_right=True)
    assert mtp16[0].pack_gqa is True and mtp16[0].split_kv == 1 and [k.split_kv for k in mtp16 if k.split_kv > 1] == [2], mtp16
    mtp32 = sets(s_q=2, causal=True, bottom_right=True)
    assert mtp32[0].pack_gqa is True, mtp32
    assert _pack_gqa_eligible(row, facts(s_q=2), 128) and _pack_gqa_group(row, facts(s_q=2), 128, True) == 16
    assert engines.mismatch(row, facts(s_q=2), engines.SdpaFwdKnobs(pack_gqa=True)) is None
    assert engines.mismatch(row, facts(s_q=2, has_paged_kv=False, page_size=0), engines.SdpaFwdKnobs(pack_gqa=True)) is None
    from cudnn.sdpa.fwd.config_sm100 import decode_d256_q_units
    from cudnn.sdpa.fwd.heuristics import _split_points, choose_decode_tile_split_kv
    from cudnn.sdpa.fwd.config_sm107 import decode_d256_q_tile as rubin_q_tile

    mtp4 = facts(h_q=24, s_q=4, causal=True, bottom_right=True)  # 24/2 at S_q = 4: two units of two tokens x 12 heads
    assert rubin_q_tile(4, 12) == 32 and decode_d256_q_units(4, 12, 32) == 2
    assert _pack_gqa_eligible(row, mtp4, 128) and _pack_gqa_group(row, mtp4, 128, True) == 12
    want = choose_decode_tile_split_kv(units=32 * 2 * 2, kv_tiles=4096 // 128, sm_count=204, q_tile=32)
    assert _split_points(row, mtp4, 128, 128, 2, pack_g=12)[0] == want, (_split_points(row, mtp4, 128, 128, 2, pack_g=12), want)
    assert sets(h_q=24, s_q=4, causal=True, bottom_right=True)[0].pack_gqa is True
    # Past the route (a third token unit): no packed set, the packed request a typed decline.
    mtp5 = facts(h_q=24, s_q=5, causal=True, bottom_right=True)
    assert rubin_q_tile(5, 12) == 0 and not _pack_gqa_eligible(row, mtp5, 128)
    assert all(k.pack_gqa is not True for k in sets(h_q=24, s_q=5, causal=True, bottom_right=True))
    why = engines.mismatch(row, mtp5, engines.SdpaFwdKnobs(pack_gqa=True))
    assert why and "decode tile" in why, why
    # MHA, the sink, and a prefill-shaped dense graph: unpacked / unsplit as the shared rules say.
    assert all(k.pack_gqa is not True for k in sets(h_q=4, h_kv=4, b=3, s_kv=1000))
    sink = sets(h_q=16, b=3, s_kv=700, has_sink=True)
    assert sink[0].pack_gqa is True and all(k.split_kv == 1 for k in sink), sink
    prefill = facts(s_q=512, h_q=16, causal=True, has_paged_kv=False, page_size=0, padded=False)
    assert all(k.pack_gqa is not True and k.split_kv == 1 for k in sets(s_q=512, h_q=16, causal=True, has_paged_kv=False, page_size=0, padded=False))
    why = engines.mismatch(row, prefill, engines.SdpaFwdKnobs(split_kv=2))
    assert why and "decode tile" in why, why
    # The paged THD d256 packed graph is NOT the decode tile's (it has no THD scheduler): unsplit at CGA2 it is served by
    # the paged half prefill pipeline's own D256 PackGQA (config_sm100.supports_paged_d256_pack_gqa); a SPLIT packed
    # request has no route on either path and the decline names both.
    assert engines.mismatch(row, facts(thd=True), engines.SdpaFwdKnobs(pack_gqa=True)) is None
    assert engines.mismatch(row, facts(thd=True), engines.SdpaFwdKnobs(pack_gqa=True, cga=2, split_kv=1)) is None
    why = engines.mismatch(row, facts(thd=True), engines.SdpaFwdKnobs(pack_gqa=True, split_kv=2))
    assert why and "decode tile" in why and "paged half THD" in why, why


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [1, 4, 128, 512])
def test_rubin_paged_thd_d256_keeps_its_packed_candidate(sm107_metadata_target, s_q):
    """The paged half THD d256 graph on cc 10.7 packs on the paged prefill pipeline (CGA2, unsplit;
    config_sm100.supports_paged_d256_pack_gqa) -- a route the Rubin d256 decode tile never touched, so the
    proposal helpers keep emitting its packed candidate exactly as they did before the tile's d256 PackGQA
    rule arrived (B = 32, 32/2 heads, page 128, 4096 keys, S_q from one token to a long prefill): one
    pack_gqa=True set at CGA2 unsplit, admissible, and no packed SPLIT (the route has none).  The explicit
    packed knobs passing mismatch() is not enough -- the candidate LIST is what a caller without knobs sees."""
    from cudnn.sdpa.fwd.heuristics import _pack_gqa_eligible

    row = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _RUBIN_F16)
    f = _decode_d256_facts(device_cc=(10, 7), device_sm_count=212, s_q=s_q, thd=True, page_size=128, causal=True, bottom_right=True)
    plans = [p.knobs for p in recommend("A", f, _RUBIN_OFFERED) if p.engine_id == _RUBIN_OFFERED[_RUBIN_F16]]
    assert plans, "the Rubin half row serves the paged THD d256 graph"
    for k in plans:
        assert engines.mismatch(row, f, k) is None, (k, engines.mismatch(row, f, k))
    packed = [k for k in plans if k.pack_gqa is True]
    assert packed and all(k.cga == 2 and (k.split_kv or 1) == 1 for k in packed), plans
    assert _pack_gqa_eligible(row, f, 128) and _pack_gqa_eligible(row, f, 128, 1, 2), "the route: some plan / CGA2 unsplit"
    assert not _pack_gqa_eligible(row, f, 128, 2) and not _pack_gqa_eligible(row, f, 128, 1, 1), "no packed split, no CGA1 packing"
    assert engines.mismatch(row, f, engines.SdpaFwdKnobs(pack_gqa=True, cga=2, split_kv=1)) is None


@pytest.mark.L0
@pytest.mark.parametrize("d", [96, 128, 200, 256])
@pytest.mark.parametrize("paged", [False, True])
def test_thd_half_admits_explicit_live_worklist_policies(d, paged):
    """Policy support and candidate validity are independent of heuristic ranking."""
    facts = _facts(d_qk=d, d_v=d, s_q=2048, h_q=16, h_kv=2, thd=True, padded=True, has_paged_kv=paged, page_size=128)
    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _F16)
    for policy in (0, SCHED_LPT, SCHED_LPT_L2):
        assert engines.mismatch(caps, facts, engines.SdpaFwdKnobs(sched_policy=policy)) is None
    plans = recommend("A", facts, {_F16: 20500})
    assert plans and all(engines.mismatch(caps, facts, p.knobs) is None for p in plans)


@pytest.mark.L0
@pytest.mark.parametrize("quant", ["fp8", "mxfp8"])
def test_quantized_thd_proposals_are_admissible(quant):
    name = engines.engine_name(**{quant: True})
    facts = _facts(thd=True, padded=True, s_q=2048, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, **{"is_" + quant: True})
    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == name)
    plans = recommend("A", facts, {name: 20501 if quant == "fp8" else 20510})
    assert plans and all(engines.mismatch(caps, facts, p.knobs) is None for p in plans)


def _sm107_paged_thd_facts(**over):
    """The paged THD + sink serving contract on cc 10.7 (issue #1472's paged table): packed queries over page-16 HND pools,
    bottom-right causal, 2056-token caches, b128 64/8 with four tokens per request."""
    base = dict(
        b=128,
        h_q=64,
        h_kv=8,
        s_q=4,
        s_kv=2056,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        causal=True,
        bottom_right=True,
        has_sink=True,
        thd=True,
        padded=True,
        has_paged_kv=True,
        page_size=16,
        device_cc=(10, 7),
        device_sm_count=216,
    )
    base.update(over)
    return _facts(**base)


@pytest.mark.L0
@pytest.mark.parametrize(
    "over, cga",
    [
        (dict(), 1),
        (dict(s_q=1), 1),
        (dict(s_q=8), 1),
        (dict(has_sink=False), 1),
        (dict(h_kv=4), 1),
        (dict(b=24, h_kv=4, s_q=8), 2),
        (dict(dtype=cudnn.data_type.HALF), 2),
        (dict(page_size=128), 1),
    ],
    ids=["b128_64_8_q4_sink", "q1_sink", "q8_sink", "sink_free", "gqa16_b128", "gqa16_b24_one_wave", "f16_keeps_cga2", "page128"],
)
def test_sm107_paged_thd_sink_sets_pack_and_order_like_their_sink_free_twins(sm107_metadata_target, over, cga):
    """The cc 10.7 paged THD plan ordering measured with and without an attention sink (issue #1472's paged table): the
    first proposal packs the group (GQA 8 and 16 alike), walks the live tiles LPT, and takes the two-slab cga1 tile where
    the wave rule prefers it (b128: 1024 / 512 units over 216 SMs) -- cga2 where one wave fits either way (b24 64/4: 96
    units) or outside the measured bf16 family (f16).  A sink changes none of that (the sink cells ranked like their
    sink-free twins), and every proposed set stays admissible."""
    from cudnn.sdpa.fwd.heuristics import _prefer_thd_pack_gqa, _sm107_paged_half

    facts = _sm107_paged_thd_facts(**over)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == _RUBIN_F16)
    assert _sm107_paged_half(spec.capabilities, facts) and _prefer_thd_pack_gqa(spec.capabilities, facts)
    plans = _sm107_f16_plans(facts)
    first = plans[0].knobs
    assert first.pack_gqa is True, first
    assert first.sched_policy == SCHED_LPT, first
    assert first.cga == cga, first
    assert first.split_kv in (None, 1), first
    # the unpacked set stays listed as a runner (autotune), never first
    assert any(p.knobs.pack_gqa is False for p in plans), [p.knobs for p in plans]


@pytest.mark.L0
def test_sm107_paged_thd_rules_do_not_move_the_sm100_row():
    """The cc 10.7 paged measurements widen nothing on the SM100 line: paged GQA16 stays unpacked-first there, a sink keeps
    the NATURAL lead and the cga2 width (its own measured family, #1468 / PR #1469's domain)."""
    from cudnn.sdpa.fwd.heuristics import _prefer_thd_pack_gqa, _sm107_paged_half

    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == _F16)
    sm100 = dict(device_cc=(10, 0), device_sm_count=148)
    gqa16 = _sm107_paged_thd_facts(h_kv=4, has_sink=False, **sm100)
    assert not _sm107_paged_half(caps, gqa16) and not _prefer_thd_pack_gqa(caps, gqa16)
    sink = _sm107_paged_thd_facts(**sm100)
    plans = [p for p in recommend("A", sink, {_F16: 20500}) if p.engine_id == 20500]
    assert plans and all(engines.mismatch(caps, sink, p.knobs) is None for p in plans)
    assert plans[0].knobs.sched_policy in (None, 0) and plans[0].knobs.cga == 2, plans[0].knobs


@pytest.mark.L0
@pytest.mark.parametrize(
    "over, packed_first",
    [
        (dict(), True),
        (dict(b=4, h_kv=4, s_q=1024, s_kv=1024), True),
        (dict(b=1, h_q=32, s_q=4096, s_kv=4096), True),
        (dict(b=1, h_kv=4, s_q=512, s_kv=512, has_sink=False), True),
        (dict(causal=False, bottom_right=False), False),
        (dict(h_q=8, h_kv=8), None),
    ],
    ids=["b2_64_8_s2048", "b4_64_4_s1024", "b1_32_8_s4096", "b1_64_4_s512_sink_free", "no_band", "mha"],
)
def test_sm107_dense_d128_gqa_packs_under_a_band_at_prefill_depth(sm107_metadata_target, over, packed_first):
    """The cc 10.7 shared dense d128 leg packs a GQA group under a diagonal band at any S_q (the SM100 rule,
    heuristics._sm100_banded_gqa_packs, measured on cc 10.7 at S 512-4096: packed 6-16 % ahead of the unpacked Rubin
    tile, with and without a sink); a mask-free graph keeps the unpacked lead, an MHA graph proposes no packed set."""
    facts = _sm107_d128_facts(**{**dict(b=2, s_q=2048, s_kv=2048), **over})
    plans = _sm107_f16_plans(facts)
    if packed_first is None:
        assert all(p.knobs.pack_gqa is not True for p in plans), [p.knobs for p in plans]
        return
    assert plans[0].knobs.pack_gqa is packed_first, [p.knobs for p in plans]
    assert {p.knobs.pack_gqa for p in plans} >= {True, False}, [p.knobs for p in plans]
    assert all(p.knobs.cga == 2 for p in plans), [p.knobs for p in plans]


@pytest.mark.L0
@pytest.mark.parametrize(
    "over, first",
    [
        (dict(s_q=64), (False, 1)),
        (dict(s_q=128), (False, 1)),
        (dict(s_q=256), (True, 2)),
        (dict(b=8, h_kv=4, s_q=128, s_kv=2048), (False, 1)),
        (dict(s_q=128, causal=True, bottom_right=True), (True, 2)),
        (dict(b=128, s_q=8, s_kv=2056), (True, 1)),
    ],
    ids=["q64_unpacked_tile", "q128_unpacked_tile", "q256_packed_body", "gqa16_q128_unpacked_tile", "banded_q128_packs", "decode_fit_packs"],
)
def test_sm107_mask_free_d128_gqa_rides_the_unpacked_decode_tile_where_only_it_fits(sm107_metadata_target, over, first):
    """Mask-free cc 10.7 dense d128 GQA (the opt-in order; placement keeps the backend first): when one head's rows fit the
    shared decode tile while the packed unit overflows it (S_q <= 128 < S_q x G) the unpacked decode tile leads, unsplit
    (measured 32 us against the packed body's 91 / 98 at b2 64/8 q128 KV 4k); past the tile the packed cga2 body keeps the
    lead, a band packs first (_sm100_banded_gqa_packs) and a packed unit that fits the tile packs first.  The other packing
    stays listed."""
    facts = _sm107_d128_facts(**{**dict(b=2, s_kv=4096, causal=False, bottom_right=False, has_sink=False), **over})
    plans = _sm107_f16_plans(facts)
    assert (bool(plans[0].knobs.pack_gqa), plans[0].knobs.cga) == first, [p.knobs for p in plans]
    if first[1] == 1:
        assert plans[0].knobs.split_kv in (None, 1), plans[0].knobs  # the decode tile's unsplit leg (its split arms measured slower)
    assert {bool(p.knobs.pack_gqa) for p in plans} == {True, False}, [p.knobs for p in plans]


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
@pytest.mark.parametrize("s_q", [2, 257])
@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
def test_explicit_paged_prefill_cga1_domain_and_tile_geometry(dtype, s_q, cc):
    """An admitted unsplit plan uses both prefill slabs; split keeps its own tile."""
    from cudnn.sdpa.fwd.heuristics import _pack_gqa_tile_q

    caps = next(s for s in engines.ENGINE_SPECS if s.name == _F16).capabilities
    facts = _facts(dtype=dtype, device_cc=cc, s_q=s_q, h_q=16, h_kv=4, thd=True, padded=True, has_paged_kv=True, page_size=16)
    reason = engines.mismatch(caps, facts, engines.SdpaFwdKnobs(cga=1, split_kv=1, pack_gqa=True))
    if cc == (10, 0):
        assert reason is None, reason
        assert _pack_gqa_tile_q(caps, facts, 128, cga=1, split_kv=1) == 2 * 128
    else:
        assert reason is not None
    assert _pack_gqa_tile_q(caps, facts, 128, cga=1, split_kv=2) == 128
