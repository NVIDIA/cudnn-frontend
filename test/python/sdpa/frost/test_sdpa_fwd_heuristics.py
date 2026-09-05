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
from cudnn.engines.base import PlanConfig
from cudnn.sdpa.fwd import engines
from cudnn.engines.heuristics import _assemble
from cudnn.sdpa.fwd.heuristics import _MAX_SETS_PER_ENGINE, recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

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
    g2, handles2, _ = _build_decodeish_graph(causal=False)
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
