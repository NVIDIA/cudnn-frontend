# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM120 half DECODE tile (``sm120/decode_f16.py``, ``tile_m`` 16 / 32).

KNOB ROUTE: the adapter runs the decode tile when the plan's ``tile_m`` names
it, exactly as ``lower_dsl_prefill`` forwards a plan's knobs; the kernel tests
pass the knobs explicitly and compare against an fp32 oracle bounded to one
query head. The facts-only tests check knob admission and rejection and the
validity of the heuristics' proposals; the graph test at the bottom runs a
decode-tile plan through CUDA Graph capture with changed inputs and storage.
"""

import math
from typing import NamedTuple, Optional

import pytest
import torch

from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [requires_dsl]

_SM120_ROW = "sdpa_fwd_prefill_sm120"


def _reference(q, k, v, *, causal, bottom_right=True, window_left=None, window_right=0, scale=None, sinks=None, seq_q=None, seq_kv=None):
    """fp32 O and natural-log LSE of BHSD q/k/v, one query head at a time.

    ``causal`` keeps keys up to the diagonal plus ``window_right``, anchored
    bottom-right by default (the decode convention: the query rows are the
    last ``S_q`` positions of the sequence) or top-left; ``window_left`` is the
    inclusive left bound; ``scale`` defaults to 1/sqrt(D); ``sinks`` (fp32
    [H_q]) join the softmax denominator; ``seq_q`` / ``seq_kv`` (int [B]) bound
    each batch's queries and keys, and a padded query row is O := 0 / LSE := -inf.
    """
    b, h_q, s_q, d = q.shape
    h_kv, s_kv = k.shape[1:3]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    out = torch.zeros((b, h_q, s_q, v.shape[-1]), device=q.device, dtype=torch.float32)
    lse = torch.full((b, h_q, s_q), float("-inf"), device=q.device, dtype=torch.float32)
    for batch in range(b):
        kv_len = s_kv if seq_kv is None else int(seq_kv[batch])
        q_len = s_q if seq_q is None else int(seq_q[batch])
        i = torch.arange(s_q, device=q.device).view(s_q, 1) + ((kv_len - q_len) if bottom_right else 0)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        visible = j < kv_len
        if causal:
            visible = visible & (j <= i + window_right)
        if window_left is not None:
            visible = visible & (j >= i - window_left)
        for head in range(h_q):
            kv_head = head // (h_q // h_kv)
            scores = torch.matmul(q[batch, head].float(), k[batch, kv_head].float().transpose(0, 1)) * scale
            scores = scores.masked_fill(~visible, float("-inf"))
            m = scores.amax(-1, keepdim=True)
            if sinks is not None:
                m = torch.maximum(m, sinks[head].view(1, 1))
            m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            p = torch.exp(scores - m_safe)
            den = p.sum(-1, keepdim=True)
            if sinks is not None:
                den = den + torch.exp(sinks[head].view(1, 1) - m_safe)
            out[batch, head] = torch.where(den > 0, torch.matmul(p, v[batch, kv_head].float()) / den.clamp_min(1e-30), torch.zeros_like(out[batch, head]))
            lse[batch, head] = torch.where(den[:, 0] > 0, (m_safe + torch.log(den.clamp_min(1e-30)))[:, 0], torch.full_like(den[:, 0], float("-inf")))
        out[batch, :, q_len:] = 0
        lse[batch, :, q_len:] = float("-inf")
    return out, lse


class _DecodeRun(NamedTuple):
    """One explicit plan's outputs, its fp32 oracle and the adapter that ran it."""

    o: torch.Tensor
    lse: Optional[torch.Tensor]
    ref_o: torch.Tensor
    ref_lse: torch.Tensor
    api: object
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


def _decode_case(
    h_q: int,
    h_kv: int,
    s_q: int,
    s_kv: int,
    *,
    d: int = 128,
    d_v: Optional[int] = None,
    tile_m: int = 16,
    tile_n: Optional[int] = None,
    pack_gqa: bool = True,
    split_kv: int = 1,
    sink: bool = False,
    causal: Optional[bool] = None,
    bottom_right: bool = True,
    window_left: Optional[int] = None,
    window_right: Optional[int] = None,
    scale: Optional[float] = None,
    with_lse: bool = False,
    stats_log2: bool = False,
    dtype=torch.float16,
    batch: int = 1,
    seq_q: Optional[list] = None,
    seq_kv: Optional[list] = None,
    nan_kv_tail: bool = False,
) -> _DecodeRun:
    """Run one explicit plan through SdpaFwdDslSm120 over the serving layout
    (BSHD storage viewed as BHSD). ``nan_kv_tail`` hands the kernel K / V whose
    rows past each batch's ``seq_kv`` length are NaN; the oracle reads the
    finite originals."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    d_v = d if d_v is None else d_v
    causal = s_q > 1 if causal is None else causal
    torch.manual_seed(0)
    q = (torch.randn(batch, s_q, h_q, d, device="cuda", dtype=dtype) * 0.5).transpose(1, 2)
    k = (torch.randn(batch, s_kv, h_kv, d, device="cuda", dtype=dtype) * 0.5).transpose(1, 2)
    v = (torch.randn(batch, s_kv, h_kv, d_v, device="cuda", dtype=dtype) * 0.5).transpose(1, 2)
    o = torch.full((batch, s_q, h_q, d_v), float("nan"), device="cuda", dtype=dtype).transpose(1, 2)
    lse = torch.full((batch, h_q, s_q), float("nan"), device="cuda", dtype=torch.float32) if with_lse else None
    sinks = torch.randn(h_q, device="cuda", dtype=torch.float32) if sink else None
    seq_q_t = None if seq_q is None else torch.tensor(seq_q, device="cuda", dtype=torch.int32)
    seq_kv_t = None if seq_kv is None else torch.tensor(seq_kv, device="cuda", dtype=torch.int32)
    api = SdpaFwdDslSm120(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse,
        is_causal=causal,
        causal_bottom_right=causal and bottom_right,
        window_size_left=window_left,
        window_size_right=window_right,
        scale_softmax=scale,
        has_sink=sink,
        seq_q_lens_present=seq_q is not None,
        seq_kv_lens_present=seq_kv is not None,
        stats_log2=stats_log2,
        tile_m=tile_m,
        tile_n=tile_n,
        pack_gqa=pack_gqa,
        split_kv=split_kv,
        sched_policy=0,
    )
    assert api.check_support()
    api.compile()
    ws_bytes = api.scratch_workspace_bytes()
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda") if ws_bytes else None
    k_run, v_run = k, v
    if nan_kv_tail:
        k_run, v_run = k.clone(), v.clone()
        for batch_idx, kv_len in enumerate(seq_kv):
            k_run[batch_idx, :, kv_len:] = float("nan")
            v_run[batch_idx, :, kv_len:] = float("nan")
    api.execute(q_tensor=q, k_tensor=k_run, v_tensor=v_run, o_tensor=o, lse_tensor=lse, sinks=sinks, seq_q_lens=seq_q_t, seq_kv_lens=seq_kv_t, workspace=ws)
    torch.cuda.synchronize()
    ref_o, ref_lse = _reference(
        q,
        k,
        v,
        causal=causal,
        bottom_right=bottom_right,
        window_left=window_left,
        window_right=window_right or 0,
        scale=scale,
        sinks=sinks,
        seq_q=seq_q,
        seq_kv=seq_kv,
    )
    if stats_log2:
        ref_lse = ref_lse * math.log2(math.e)
    return _DecodeRun(o, lse, ref_o, ref_lse, api, q, k, v)


def _assert_close(run: _DecodeRun, dtype=torch.float16):
    assert not torch.isnan(run.o).any(), "a live O element was never written"
    atol = 2e-2 if dtype == torch.float16 else 4e-2
    torch.testing.assert_close(run.o.float(), run.ref_o, rtol=0, atol=atol)
    if run.lse is not None:
        finite = torch.isfinite(run.ref_lse)
        assert torch.equal(torch.isfinite(run.lse), finite), "LSE is -inf exactly on rows without a visible key or sink"
        torch.testing.assert_close(run.lse[finite], run.ref_lse[finite], rtol=0, atol=2e-2)


def _facts(**over):
    """Dense SM120 decode facts (no GPU): GQA 4 at d128 over 8K keys unless overridden."""
    import cudnn
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    base = dict(b=1, h_q=32, h_kv=8, s_q=1, s_kv=8192, d_qk=128, d_v=128, dtype=cudnn.data_type.HALF, device_cc=(12, 0), device_sm_count=188)
    base.update(over)
    if base["s_q"] > 1:
        base.setdefault("causal", True)
        base.setdefault("bottom_right", True)
    return SdpaGraphFacts(**base)


def _spec(name: str = _SM120_ROW):
    from cudnn.sdpa.fwd import engines

    return next(s for s in engines.ENGINE_SPECS if s.name == name)


# --- config and eligibility (no GPU) ------------------------------------------


@pytest.mark.L0
def test_decode_stage_widths_and_ring_fit():
    from cudnn.sdpa.fwd.config_sm120 import (
        SMEM_CAPACITY_BYTES,
        decode_kv_stage,
        decode_num_stages,
        smem_bytes,
        tile_domain,
    )

    assert [decode_kv_stage(d, d) for d in (16, 32, 48, 64, 80, 128, 256)] == [64, 64, 32, 32, 16, 16, 16]
    assert decode_kv_stage(192, 128) == 16
    for d in range(16, 257, 16):
        for q_tile in (16, 32):
            # Every key group owns at least one slot (decode_slots_per_group).
            assert decode_num_stages(d, d, q_tile) >= 4 and decode_num_stages(d, d, q_tile) % (4 // (q_tile // 16)) == 0, (d, q_tile)
            assert smem_bytes(d, d, q_tile, decode_kv_stage(d, d)) <= SMEM_CAPACITY_BYTES
    # Half graphs list the decode tiles at the stage width; FP8 and d512 do not.
    assert {(16, 16), (32, 16)} <= tile_domain(128, 128, fp8=False)
    assert not any(m in (16, 32) for m, _ in tile_domain(128, 128, fp8=True))
    assert not any(m in (16, 32) for m, _ in tile_domain(512, 512, fp8=False))
    # The decode tile rings head tiles up to 256 on both sides, whichever template serves the graph.
    assert not any(m in (16, 32) for m, _ in tile_domain(288, 128, fp8=False))


@pytest.mark.L0
def test_each_sm120_template_validates_its_own_q_tiles():
    """validate_params backs every SM120 template: a prefill template is built for the sequence Q tiles and the
    decode tile for its own, so a direct load with the other family's Q tile declines instead of tracing a body
    whose row count the tile does not match."""
    from cudnn.sdpa.fwd.config_sm120 import TemplateParams, validate_params

    validate_params(TemplateParams(q_tile=64, kv_tile=128))
    validate_params(TemplateParams(q_tile=16, kv_tile=16), decode_tile=True)
    with pytest.raises(ValueError, match="q_tile must be one of"):
        validate_params(TemplateParams(q_tile=16, kv_tile=16))
    with pytest.raises(ValueError, match="q_tile must be one of"):
        validate_params(TemplateParams(q_tile=64, kv_tile=64), decode_tile=True)


@pytest.mark.L0
@pytest.mark.parametrize(
    "facts_over,knobs_over,reason",
    [
        ({}, dict(tile_m=16, tile_n=64), "16-key K/V stages"),
        ({}, dict(tile_m=64, tile_n=16), "has no tile_m=64, tile_n=16"),
        ({}, dict(tile_m=64, tile_n=32), "has no tile_m=64, tile_n=32"),
        (dict(d_qk=512, d_v=512), dict(tile_m=128), "has no tile_m=128"),
        (dict(d_qk=512, d_v=512), dict(tile_m=16), "head dims up to 256"),
        (dict(h_q=32, h_kv=1), dict(tile_m=16, pack_gqa=True), "does not pack at tile_m=16"),
        (dict(s_q=4), dict(tile_m=16, sched_policy="lpt"), "requested sched_policy=1 is outside this engine's domain [0]"),
        (dict(b=2, thd=True, padded=True), dict(tile_m=16), "dense-only"),
    ],
)
def test_sm120_tile_requests_the_kernels_cannot_serve_are_ineligible(facts_over, knobs_over, reason):
    """Each tile knob passes its own domain, but the template serving the head dims takes only the pairs
    config_sm120.tile_domain lists, and the decode tile has graph conditions of its own: mismatch() declines
    these before lowering, so the adapter never raises past eligibility."""
    from dataclasses import replace

    from cudnn.frost.tile_dsl.constants import SCHED_LPT
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, mismatch

    if knobs_over.get("sched_policy") == "lpt":
        knobs_over = dict(knobs_over, sched_policy=SCHED_LPT)
    knobs = replace(SdpaFwdKnobs(cga=1, split_kv=1, pack_gqa=False), **knobs_over)
    got = mismatch(_spec().capabilities, _facts(**facts_over), knobs)
    assert got is not None and reason in got, got


@pytest.mark.L0
def test_sm120_knob_admission_for_the_decode_tile():
    """The half row admits the decode tile at its stage width, packed (G = 6 included) or not, split and with a
    sink; the FP8 row has no decode tile."""
    import cudnn
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, mismatch

    half = _spec().capabilities

    def knobs(tile_m, tile_n, *, split=1, pack=True):
        return SdpaFwdKnobs(sched_policy=0, tile_m=tile_m, tile_n=tile_n, cga=1, pack_gqa=pack, split_kv=split)

    assert mismatch(half, _facts(), knobs(16, 16, split=4)) is None
    assert mismatch(half, _facts(), knobs(32, 16, pack=False)) is None
    assert mismatch(half, _facts(h_q=24, h_kv=4, d_qk=256, d_v=256), knobs(16, 16)) is None
    assert mismatch(half, _facts(h_q=64, h_kv=8, d_qk=64, d_v=64), knobs(32, 32)) is None
    assert mismatch(half, _facts(has_sink=True), knobs(16, 16, split=4)) is None
    fp8 = next(s for s in _fp8_specs()).capabilities
    fp8_facts = _facts(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.HALF, is_fp8=True)
    assert "outside this engine's domain" in mismatch(fp8, fp8_facts, knobs(16, 16, pack=False))


def _fp8_specs():
    from cudnn.sdpa.fwd import engines

    return [s for s in engines.ENGINE_SPECS if s.capabilities.sm_lo == 120 and s.capabilities.is_fp8]


_HEURISTIC_SHAPES = [
    dict(),
    dict(h_q=24, h_kv=4, d_qk=256, d_v=256),
    dict(s_q=3, h_q=32, h_kv=4),
    dict(b=64, s_q=2, h_q=32, h_kv=8),
    dict(s_q=16, h_q=24, h_kv=4),
    dict(s_q=20, h_q=24, h_kv=4, s_kv=131072),
    dict(h_q=64, h_kv=8, d_qk=64, d_v=64, has_sink=True),
    dict(h_q=64, h_kv=8, d_qk=64, d_v=64, has_sink=True, causal=True, bottom_right=True, window_left=128),
    dict(b=4, h_q=64, h_kv=8, d_qk=64, d_v=64, has_sink=True, s_kv=1024),
    dict(s_q=2048, s_kv=2048),
]


@pytest.mark.L0
@pytest.mark.parametrize("over", _HEURISTIC_SHAPES)
def test_heuristics_propose_admissible_sm120_plans(over):
    """Facts-only: every SM120 knob set the heuristics PROPOSE (before recommend() drops what mismatch() declines)
    is admissible -- a decode-tile set included, whatever scheduler the graph's prefill tiles prefer -- names a
    pair from the kernel table, and packs only a group that fits its tile."""
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.config_sm120 import tile_domain
    from cudnn.sdpa.fwd.engines import mismatch

    spec = _spec()
    facts = _facts(**over)
    proposed = heuristics._knob_sets(spec, facts)
    assert proposed
    for knobs in proposed:
        assert mismatch(spec.capabilities, facts, knobs) is None, knobs
        assert (knobs.tile_m, knobs.tile_n) in tile_domain(facts.d_qk, facts.d_v, fp8=False), knobs
        if knobs.pack_gqa:
            assert facts.h_q // facts.h_kv <= knobs.tile_m, knobs


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [dict(), dict(d_qk=64, d_v=64), dict(d_qk=512, d_v=512), dict(b=2, thd=True, padded=True), dict(dtype="fp8"), dict(dtype="fp8", d_qk=512, d_v=512)],
)
def test_sm120_fallback_builds_a_kernel_table_tile(over):
    """The FALLBACK plan of every SM120 row names a (tile_m, tile_n) pair the template serving the graph's head
    dims is built for, and is admissible."""
    import cudnn
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.config_sm120 import tile_domain
    from cudnn.sdpa.fwd.heuristics import recommend

    fp8 = over.pop("dtype", None) == "fp8"
    if fp8:
        over.update(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.HALF, is_fp8=True)
    facts = _facts(**over)
    offered = {s.name: 20504 + i for i, s in enumerate(engines.ENGINE_SPECS) if s.capabilities.sm_lo == 120}
    plans = recommend("FALLBACK", facts, offered)
    assert plans
    for plan in plans:
        assert (plan.knobs.tile_m, plan.knobs.tile_n) in tile_domain(facts.d_qk, facts.d_v, fp8=fp8), plan.knobs


@pytest.mark.L0
@pytest.mark.parametrize("over", [dict(), dict(h_q=64, h_kv=8, d_qk=64, d_v=64, has_sink=True), dict(s_q=3, h_q=24, h_kv=4, d_qk=256, d_v=256)])
@pytest.mark.parametrize("chosen", [1, 4])
def test_decode_tile_plans_carry_the_split_choice(monkeypatch, over, chosen):
    """The decode-tile split chooser (mocked) sees the launch the kernel runs -- one unit per whole tokens-per-tile
    slice of S_q per packed KV head and batch, over decode_kv_stage-key stages -- and every proposed decode-tile
    plan carries its answer or the unsplit runner. A sink graph is no exception."""
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.config_sm120 import decode_kv_stage, head_tiles, HEAD_TILE_GRANULE
    from cudnn.sdpa.fwd.engines import sm120_decode_tile

    seen = []

    def recording(**kw):
        seen.append(kw)
        return chosen

    monkeypatch.setattr(heuristics, "choose_sm120_decode_split_kv", recording)
    spec = _spec()
    facts = _facts(**over)
    plans = heuristics.recommend("A", facts, {spec.name: 20504})
    decode = [p.knobs for p in plans if sm120_decode_tile(spec.capabilities, p.knobs.tile_m)]
    assert decode and seen
    stage = decode_kv_stage(*head_tiles(facts.d_qk, facts.d_v, HEAD_TILE_GRANULE))
    group = facts.h_q // facts.h_kv
    expected_units = {
        math.ceil(facts.s_q / (k.tile_m // group if k.pack_gqa else k.tile_m)) * facts.b * (facts.h_kv if k.pack_gqa else facts.h_q) for k in decode
    }
    assert {kw["units"] for kw in seen} <= expected_units | {math.ceil(facts.s_q / m) * facts.b * facts.h_q for m in (16, 32)}
    assert all(kw["kv_stages"] == math.ceil(facts.s_kv / stage) for kw in seen)
    assert {k.split_kv for k in decode} <= {chosen, 1}


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [
        dict(h_q=24, h_kv=4, d_qk=256, d_v=256, s_kv=65536),
        dict(s_q=2, h_q=24, h_kv=4, s_kv=16384),
        dict(h_q=64, h_kv=8, d_qk=64, d_v=64, has_sink=True, s_kv=131072),
    ],
)
def test_each_decode_tile_set_is_split_for_its_own_launch(monkeypatch, over):
    """A runner that changes the packing changes the unit count (unpacked: G x the packed launch's), so it carries the
    split chooser's answer for its own launch: the packed lead's split would launch G x the CTAs. Every set the
    heuristics build is checked, the ones past the per-engine cap included."""
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import sm120_decode_tile

    def by_units(*, units, **_):
        return max(1, 256 // units)

    monkeypatch.setattr(heuristics, "choose_sm120_decode_split_kv", by_units)
    monkeypatch.setattr(heuristics, "_MAX_SETS_PER_ENGINE", 64)
    spec = _spec()
    facts = _facts(**over)
    group = facts.h_q // facts.h_kv
    decode = [k for k in heuristics._knob_sets(spec, facts) if sm120_decode_tile(spec.capabilities, k.tile_m)]
    assert any(k.pack_gqa for k in decode) and any(not k.pack_gqa for k in decode), decode
    for k in decode:
        units = math.ceil(facts.s_q / (k.tile_m // group if k.pack_gqa else k.tile_m)) * facts.b * (facts.h_kv if k.pack_gqa else facts.h_q)
        assert k.split_kv in (by_units(units=units), 1), (k, units)


@pytest.mark.L0
@pytest.mark.parametrize(
    "over,lead",
    [
        (dict(), (16, 16)),  # 4 packed rows
        (dict(s_q=4), (16, 16)),  # 16 rows
        (dict(s_q=5), (32, 16)),  # 20 rows
        (dict(h_q=24, h_kv=4, d_qk=256, d_v=256, s_q=5), (32, 16)),  # GQA 6: 30 live rows of 32
        (dict(h_q=64, h_kv=8, d_qk=64, d_v=64, has_sink=True), (16, 32)),  # 32-key stages at d64
        (dict(s_q=8), (32, 16)),  # 32 rows
        (dict(s_q=9, s_kv=4096), (128, None)),  # 36 rows: a prefill tile, packing on the largest that admits it
        (dict(h_q=64, h_kv=1), (128, None)),  # MQA: 64 rows
    ],
)
def test_sm120_decode_shaped_graphs_lead_on_the_decode_tile(over, lead):
    """A half graph whose packed rows per KV head (S_q x G) fit 32 leads packed on the smallest decode tile that
    holds them, at its stage width; past 32 rows the packed lead keeps the largest prefill tile that admits the
    ratio. FP8 and THD graphs never reach the decode tile."""
    import cudnn
    from cudnn.sdpa.fwd import heuristics

    facts = _facts(**over)
    lead_set = heuristics._knob_sets(_spec(), facts)[0]
    assert lead_set.pack_gqa is True, lead_set
    assert lead_set.tile_m == lead[0] and (lead[1] is None or lead_set.tile_n == lead[1]), lead_set
    fp8 = _fp8_specs()[0]
    fp8_facts = _facts(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.HALF, is_fp8=True, **over)
    assert all(k.tile_m >= 64 for k in heuristics._knob_sets(fp8, fp8_facts))
    thd = _facts(**{**over, "b": 2, "thd": True, "padded": True})
    assert all(k.tile_m >= 64 for k in heuristics._knob_sets(_spec(), thd))


@pytest.mark.L0
def test_sm120_decode_split_model_anchors():
    """The decode tile's split model, unmocked: a launch that fills the machine stays unsplit, a lone KV stream over
    128K keys splits until the DRAM bound takes over, and a one-stage unit never splits."""
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.config_sm120 import decode_stage_bytes

    stage = decode_stage_bytes(128, 128)
    choose = heuristics.choose_sm120_decode_split_kv
    assert choose(units=2 * 188, kv_stages=8192, stage_bytes=stage, d_v=128, sm_count=188) == 1
    split = choose(units=8, kv_stages=8192, stage_bytes=stage, d_v=128, sm_count=188)
    assert split > 1 and split & (split - 1) == 0, split
    assert choose(units=8, kv_stages=1, stage_bytes=stage, d_v=128, sm_count=188) == 1
    # Through the heuristics: b = 1, GQA 4 over 128K keys leads with that split on the packed decode tile.
    lead_set = heuristics._knob_sets(_spec(), _facts(s_kv=131072))[0]
    assert (lead_set.tile_m, lead_set.pack_gqa, lead_set.split_kv) == (16, True, split), lead_set


# --- the kernel through the adapter -----------------------------------------

_L0_CASES = [
    # (h_q, h_kv, s_q, s_kv, d, tile_m, split, sink)
    (32, 8, 1, 4096, 128, 16, 1, False),  # GQA 4, one decode step
    (32, 8, 1, 4133, 128, 16, 8, False),  # ragged tail under a split
    (24, 4, 1, 2000, 256, 16, 4, False),  # GQA 6: 6 live rows of 16, d256 16-key stages
    (24, 4, 2, 777, 128, 16, 1, False),  # GQA 6 x 2 tokens = 12 live rows, causal
    (24, 4, 5, 1500, 128, 16, 2, False),  # GQA 6 over three tiles: dead rows name the next tile's first token
    (64, 8, 1, 3000, 64, 16, 1, True),  # d64 sink, unsplit fold
    (64, 8, 1, 3000, 64, 16, 6, True),  # d64 sink folded by the combine
    (64, 8, 3, 515, 64, 32, 3, True),  # 24 packed rows on the 32-row tile, causal, split sink
    (16, 2, 2, 4096, 256, 16, 2, False),  # GQA 8 x 2 tokens = 16 rows, causal, d256
    (8, 8, 5, 300, 96, 16, 1, False),  # MHA, five tokens per tile
    (32, 16, 1, 1, 128, 16, 1, False),  # one key
]


@requires_blackwell_geforce
@pytest.mark.L0
@pytest.mark.parametrize("h_q,h_kv,s_q,s_kv,d,tile_m,split,sink", _L0_CASES)
def test_decode_tile_matches_reference(h_q, h_kv, s_q, s_kv, d, tile_m, split, sink):
    run = _decode_case(h_q, h_kv, s_q, s_kv, d=d, tile_m=tile_m, split_kv=split, sink=sink, with_lse=True)
    assert run.api.q_tile == tile_m and run.api.split_kv == split
    _assert_close(run)


@requires_blackwell_geforce
@pytest.mark.parametrize(
    "d,d_v,split",
    [
        pytest.param(72, 72, 1, marks=pytest.mark.L0),  # head tile 80: TMA zero-fills the pad columns, one copy per swizzle chunk
        pytest.param(88, 88, 3, marks=pytest.mark.L1),  # head tile 96, 16-key stages
        pytest.param(128, 120, 2, marks=pytest.mark.L1),  # V envelope only
        pytest.param(16, 16, 1, marks=pytest.mark.L1),  # 64-key stages, 32-byte swizzle
        pytest.param(32, 32, 4, marks=pytest.mark.L1),  # 64-key stages, 64-byte swizzle
        pytest.param(48, 48, 2, marks=pytest.mark.L1),  # 32-key stages, 32-byte swizzle
    ],
)
def test_decode_tile_head_dim_envelopes_and_stage_widths(d, d_v, split):
    _assert_close(_decode_case(32, 8, 2, 3001, d=d, d_v=d_v, split_kv=split, sink=True, with_lse=True))


@requires_blackwell_geforce
@pytest.mark.L1
@pytest.mark.parametrize("tile_m,s_q", [(16, 5), (32, 7)])
@pytest.mark.parametrize("bottom_right", [True, False])
@pytest.mark.parametrize("split", [1, 2])
def test_decode_tile_dead_rows_never_store_over_the_next_tiles_token(tile_m, s_q, bottom_right, split):
    """GQA 6 over several Q tiles: a 16-row tile holds two tokens in 12 live rows (32 rows: five tokens in 30), so
    its dead rows name the first token of the next tile. Only the live rows may store O and LSE. One such case runs
    at L0 in ``_L0_CASES``."""
    run = _decode_case(24, 4, s_q, 1500, d=128, tile_m=tile_m, split_kv=split, bottom_right=bottom_right, with_lse=True)
    assert run.api.q_tile == tile_m and run.api.pack_gqa
    _assert_close(run)


@requires_blackwell_geforce
@pytest.mark.parametrize(
    "dtype,stats_log2",
    [
        pytest.param(torch.bfloat16, True, marks=pytest.mark.L0),
        pytest.param(torch.bfloat16, False, marks=pytest.mark.L1),
        pytest.param(torch.float16, True, marks=pytest.mark.L1),
        pytest.param(torch.float16, False, marks=pytest.mark.L1),
    ],
)
def test_decode_tile_dtypes_and_stats_base(dtype, stats_log2):
    for split in (1, 4):
        _assert_close(_decode_case(32, 4, 1, 5000, d=128, split_kv=split, with_lse=True, stats_log2=stats_log2, dtype=dtype), dtype)


@requires_blackwell_geforce
@pytest.mark.L1
def test_decode_tile_mixed_head_dims_and_unpacked_gqa():
    _assert_close(_decode_case(16, 4, 1, 1500, d=192, d_v=128, split_kv=3, with_lse=True))
    # Unpacked: one query head per CTA, each streaming its KV head.
    _assert_close(_decode_case(16, 4, 2, 1500, d=128, pack_gqa=False, split_kv=2, with_lse=True))


@requires_blackwell_geforce
@pytest.mark.L1
def test_decode_tile_window_and_padded_lengths():
    # A left window cuts the leading stages; per-batch lengths bound the keys (unsplit: padded graphs do not split).
    _assert_close(_decode_case(32, 8, 2, 4096, d=128, window_left=129, with_lse=True))
    _assert_close(_decode_case(32, 8, 1, 1024, d=128, batch=3, seq_kv=[1024, 17, 0], sink=True, with_lse=True))


@requires_blackwell_geforce
@pytest.mark.L0
@pytest.mark.parametrize("h_q,h_kv,s_q,d,tile_m", [(32, 8, 1, 128, 16), (64, 8, 3, 64, 32), (24, 4, 2, 256, 16)])
def test_decode_tile_keeps_a_nan_kv_tail_out_of_o(h_q, h_kv, s_q, d, tile_m):
    """K / V rows past a batch's KV length are memory the TMA loads (it zero-fills only past the tensor), and a masked
    key's P = 0 does not cancel a NaN V row in P @ V. The batches end inside a stage, at a stage edge and on the
    first key."""
    stage = {64: 32, 128: 16, 256: 16}[d]
    seq_kv = [1000, 3 * stage + 5, 2 * stage, 1]
    _assert_close(_decode_case(h_q, h_kv, s_q, 1000, d=d, tile_m=tile_m, batch=len(seq_kv), seq_kv=seq_kv, nan_kv_tail=True, with_lse=True))


@requires_blackwell_geforce
@pytest.mark.L0
@pytest.mark.parametrize(
    "case",
    [
        dict(h_q=32, h_kv=8, s_q=3, s_kv=600, bottom_right=False, split_kv=2),  # top-left diagonal
        dict(h_q=32, h_kv=8, s_q=3, s_kv=900, window_left=64, window_right=5, split_kv=2),  # right band + left window
        dict(h_q=24, h_kv=4, s_q=1, s_kv=3000, d=256, scale=-0.09, split_kv=4, sink=True),  # negative scale
        dict(h_q=32, h_kv=8, s_q=4, s_kv=700, batch=3, seq_q=[4, 1, 0], seq_kv=[700, 33, 5], sink=True),  # padded queries
        dict(h_q=32, h_kv=8, s_q=4, s_kv=2, split_kv=2),  # rows 0-1 see no key; one split chunk is empty
        dict(h_q=32, h_kv=8, s_q=4, s_kv=2, split_kv=2, sink=True),  # the same rows land on the sink
    ],
)
def test_decode_tile_masks_scale_and_padded_queries(case):
    case = dict(case)
    shape = [case.pop(name) for name in ("h_q", "h_kv", "s_q", "s_kv")]
    _assert_close(_decode_case(*shape, with_lse=True, **case))


@requires_blackwell_geforce
@pytest.mark.L1
@pytest.mark.parametrize("d,tile_m,split", [(128, 16, 1), (128, 16, 16), (256, 16, 4), (128, 32, 8), (64, 16, 2)])
def test_decode_tile_ring_reuse_is_race_free(d, tile_m, split):
    """A long KV reuses every ring slot hundreds of times. Each slot has one waiting key group: a parity wait cannot
    tell a phase from the one two later, so a slot shared by two groups lets one of them read a stale stage and
    release the slot early -- wrong O, then a hang or a fault. The kernel is deterministic, so repeated launches must
    match bitwise, and the first must match the reference."""
    h_q, h_kv, s_q = (32, 8, 1) if tile_m == 16 else (32, 4, 3)
    run = _decode_case(h_q, h_kv, s_q, 65536 + 37, d=d, tile_m=tile_m, split_kv=split, with_lse=True)
    _assert_close(run)
    ws = torch.empty(max(run.api.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
    first = run.o.clone()
    for launch in range(50):
        run.api.execute(q_tensor=run.q, k_tensor=run.k, v_tensor=run.v, o_tensor=run.o, lse_tensor=run.lse, workspace=ws)
        torch.cuda.synchronize()
        assert torch.equal(run.o, first), f"launch {launch} of a deterministic kernel diverged"


@requires_blackwell_geforce
@pytest.mark.L0
def test_decode_tile_more_splits_than_stages():
    """Splits past the stage count are empty chunks: O := 0 / LSE := -inf partials the combine absorbs."""
    _assert_close(_decode_case(32, 8, 1, 70, d=128, split_kv=8, sink=True, with_lse=True))


@requires_blackwell_geforce
@pytest.mark.L0
def test_decode_tile_declines_what_it_cannot_serve():
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    q = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 64, 128, device="cuda", dtype=torch.float16)
    o = torch.empty_like(q)
    # A 32-head group does not fit the 16-row tile.
    with pytest.raises(ValueError, match=r"h_q/h_kv <= q_tile \(16\)"):
        SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=k, sample_o=o, tile_m=16, pack_gqa=True).check_support()
    # The stage width is fixed by the head tiles.
    with pytest.raises(ValueError, match="16-key K/V stages"):
        SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=k, sample_o=o, tile_m=16, tile_n=128).check_support()


@requires_blackwell_geforce
@pytest.mark.L0
def test_decode_tile_wrapper_keys_an_unset_kv_tile_apart_from_an_explicit_one():
    """The standalone wrapper caches one adapter per key. An unset kv_tile resolves to the decode tile's stage width,
    so it keys apart from any explicit value: a kv_tile the decode tile cannot run still raises after an unset call
    built and cached the 16-key plan."""
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_dsl_sm120

    torch.manual_seed(0)
    q = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(1, 512, 8, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
    v = torch.randn(1, 512, 8, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
    o_tensor, _ = sdpa_fwd_wrapper_dsl_sm120(q, k, v, q_tile=16)
    ref_o, _ = _reference(q, k, v, causal=False)
    torch.testing.assert_close(o_tensor.float(), ref_o, rtol=0, atol=2e-2)
    with pytest.raises(ValueError, match="16-key K/V stages"):
        sdpa_fwd_wrapper_dsl_sm120(q, k, v, q_tile=16, kv_tile=128)


@requires_blackwell_geforce
@pytest.mark.L1
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("h_q,h_kv", [(32, 8), (24, 4), (16, 2), (32, 16), (64, 8)])
@pytest.mark.parametrize("s_q", [1, 2, 4])
@pytest.mark.parametrize("s_kv", [1, 255, 8191])
@pytest.mark.parametrize("split", [1, 7])
@pytest.mark.parametrize("sink", [False, True])
def test_decode_tile_sweep(d, h_q, h_kv, s_q, s_kv, split, sink):
    rows = s_q * (h_q // h_kv)
    if rows > 32:
        pytest.skip("past the 32-row decode tile")
    _assert_close(_decode_case(h_q, h_kv, s_q, s_kv, d=d, tile_m=16 if rows <= 16 else 32, split_kv=split, sink=sink, with_lse=True))


# --- the graph route ----------------------------------------------------------


@requires_blackwell_geforce
@pytest.mark.L0
@pytest.mark.parametrize("split,sink", [(1, False), (4, True)])
def test_decode_tile_graph_capture_reads_current_inputs_and_storage(split, sink, monkeypatch):
    """A decode-tile plan (explicit knobs, as an autotuner replays a record) runs through the graph: the first
    execute is captured, a replay after changing V reads the new values, and fresh Q/K/V/O/Stats storage is
    rebound on the next capture. Execute never compiles, and an eager execute allocates nothing."""
    import cudnn
    import cutlass.cute as cute

    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, engine_name
    from frost_test_utils import select_engine

    b, h_q, h_kv, s_kv, d = 2, 32, 8, 4133, 128
    torch.manual_seed(5)
    storage = {
        "q": torch.randn(b, 1, h_q, d, device="cuda", dtype=torch.float16) * 0.5,
        "k": torch.randn(b, s_kv, h_kv, d, device="cuda", dtype=torch.float16) * 0.5,
        "v": torch.randn(b, s_kv, h_kv, d, device="cuda", dtype=torch.float16) * 0.5,
        "o": torch.full((b, 1, h_q, d), float("nan"), device="cuda", dtype=torch.float16),
        "lse": torch.full((b, h_q, 1, 1), float("nan"), device="cuda", dtype=torch.float32),
    }
    sinks = torch.randn(1, h_q, 1, 1, device="cuda", dtype=torch.float32) if sink else None

    def view(name):
        return storage[name] if name == "lse" else storage[name].transpose(1, 2)

    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tensors = {name: g.tensor_like(view(name), name=name) for name in ("q", "k", "v")}
    kw = {"sink_token": g.tensor_like(sinks)} if sink else {}
    o, lse = g.sdpa(name="sdpa", q=tensors["q"], k=tensors["k"], v=tensors["v"], attn_scale=d**-0.5, generate_stats=True, **kw)
    o.set_output(True).set_dim(view("o").shape).set_stride(view("o").stride())
    lse.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(view("lse").shape).set_stride(view("lse").stride())
    tensors["o"], tensors["lse"] = o, lse
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    row = select_engine(g, engine_name(arch="sm120"))
    g.create_execution_plan(row.engine_id, SdpaFwdKnobs(sched_policy=0, tile_m=16, tile_n=16, cga=1, pack_gqa=True, split_kv=split))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    # The pin is strict: the built plan is the decode tile, on the prepared pointer launch.
    knobs = g.plans[g._plan_index].knobs
    assert (knobs.tile_m, knobs.split_kv) == (16, split), knobs
    assert g._compiled_plans[g._plan_index]._prepared is not None
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute must not compile"))

    def check():
        ref_o, ref_lse = _reference(view("q"), view("k"), view("v"), causal=False, sinks=None if sinks is None else sinks.view(h_q))
        torch.testing.assert_close(view("o").float(), ref_o, rtol=0, atol=2e-2)
        torch.testing.assert_close(view("lse")[..., 0], ref_lse, rtol=0, atol=2e-2)

    for iteration in range(2):
        if iteration:
            for name in storage:
                storage[name] = storage[name].clone()
            ws = torch.empty_like(ws)
        vp = {tensors[name]: view(name) for name in tensors}
        if sink:
            vp[kw["sink_token"]] = sinks
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        captured = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(captured, stream=stream):
                g.execute(vp, ws)
            torch.cuda.current_stream().wait_stream(stream)
            storage["v"].mul_(0.7)
            storage["o"].fill_(float("nan"))
            storage["lse"].fill_(float("nan"))
            captured.replay()
            torch.cuda.synchronize()
            check()
        finally:
            captured.reset()
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        g.execute(vp, ws)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode("default")
