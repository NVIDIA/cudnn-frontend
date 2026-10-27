# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDPA-backward heuristic's recommend contract (no GPU): every eligible
row lists exactly one entry, and the entry spells the tiles the lowering would
pick when handed no knobs -- so a recorded (engine_id, knobs) pins the kernel."""

import pytest

import cudnn
from cudnn.sdpa.bwd import engines as bwd_engines
from cudnn.sdpa.bwd.config_sm120 import DEFAULT_TILES, padded_head_dims
from cudnn.sdpa.bwd.engines import SdpaBwdKnobs, mismatch
from cudnn.sdpa.bwd.heuristics import _DEFAULT_TILE_RESOLVERS, default_knobs, recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]  # mismatch() declines every row without the DSL


@pytest.fixture(autouse=True)
def _mock_target_for_cross_arch_contracts(monkeypatch):
    # This module probes Rubin rows on non-Rubin hosts too (the analyzer's cc faked to 10.7).  bwd mismatch() now carries the
    # fwd rows' sm_107a DSL gate (AGENTS.md Rule 7), so match the fake device with a fake compiler target -- exactly as the
    # fwd suites do; real Rubin runs use the real build.
    import torch
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


_OFFERED = {
    "sdpa_bwd_sm120": 20600,
    "sdpa_bwd_sm80": 20601,
    "sdpa_bwd_sm100": 20602,
    "sdpa_bwd_sm100_mxfp8": 20603,
    "sdpa_bwd_sm107": 20604,
    "sdpa_bwd_sm107_fp8": 20605,
    "sdpa_bwd_sm107_mxfp8": 20606,
    "sdpa_bwd_sm100_d256": 20607,
    "sdpa_bwd_sm107_d512": 20608,
    "sdpa_bwd_sm100_d128": 20609,
}


def _facts(**over):
    base = dict(is_backward=True, b=2, h_q=8, h_kv=8, s_q=256, s_kv=256, d_qk=128, d_v=128, dtype=cudnn.data_type.HALF, causal=True, device_cc=(12, 0))
    base.update(over)
    return SdpaGraphFacts(**base)


@pytest.mark.parametrize("d", [32, 40, 64, 128, 192, 256])
def test_sm120_row_lists_the_kernels_per_head_dim_default(d):
    plans = recommend("A", _facts(d_qk=d, d_v=d), _OFFERED)
    assert [p.engine_id for p in plans] == [20600]
    q_tile, kv_tile = DEFAULT_TILES[padded_head_dims(d, d)[0]]  # d=40 runs at the padded 64
    assert plans[0].knobs == SdpaBwdKnobs(tile_m=q_tile, tile_n=kv_tile)
    kt = cudnn.knob_type
    assert plans[0].knobs.to_public() == {kt.TILE_M: q_tile, kt.TILE_N: kv_tile}
    assert plans[0].mode is None and plans[0].cpp_index is None


def test_fixed_geometry_row_lists_no_knobs():
    # sm100 f16: the kernel takes no tile choice, so {} is the complete record.
    plans = recommend("A", _facts(d_qk=512, d_v=512, device_cc=(10, 0)), _OFFERED)
    assert [(p.engine_id, p.knobs) for p in plans] == [(20602, None)]


def test_single_point_domain_row_lists_its_sole_tiles():
    facts = _facts(d_qk=256, d_v=256, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.HALF, is_mxfp8=True, device_cc=(10, 0))
    plans = recommend("A", facts, _OFFERED)
    assert [(p.engine_id, p.knobs) for p in plans] == [(20603, SdpaBwdKnobs(tile_m=128, tile_n=128))]


def test_listed_knobs_are_admissible_and_kind_does_not_matter():
    for facts in (_facts(), _facts(d_qk=256, d_v=256), _facts(d_qk=512, d_v=512, device_cc=(10, 0))):
        a = recommend("A", facts, _OFFERED)
        assert a and a == recommend("FALLBACK", facts, _OFFERED)
        for p in a:
            spec = next(s for s in bwd_engines.ENGINE_SPECS if _OFFERED[s.name] == p.engine_id)
            assert mismatch(spec.capabilities, facts, p.knobs) is None


def test_unoffered_rows_are_not_listed():
    assert recommend("A", _facts(), {"sdpa_bwd_sm100": 20602}) == []


def test_every_row_with_a_tile_choice_resolves_a_default():
    """A new row that advertises a tile domain must either be a single point
    or register a resolver -- otherwise its plan would be listed as {} while
    the kernel picks a tile of its own, the drift this heuristic removes."""
    for spec in bwd_engines.ENGINE_SPECS:
        caps = spec.capabilities
        if not caps.tile_ms and not caps.tile_ns:
            assert default_knobs(spec, _facts()) is None
            continue
        single_point = len(caps.tile_ms) == 1 and len(caps.tile_ns) == 1
        assert single_point or spec.name in _DEFAULT_TILE_RESOLVERS, spec.name


@pytest.mark.parametrize(
    "cc, want",
    [((10, 7), [20604]), ((11, 0), [20604]), ((10, 0), [20607]), ((10, 3), [20607]), ((12, 0), [20600])],
    ids=["sm107", "sm110", "sm100", "sm103", "sm120"],
)
def test_sm107_half_row_lists_one_knobless_entry_on_the_rubin_line(cc, want):
    """The Rubin d256 bf16 / fp16 backward row (slot 4 -> 20604) is fixed-geometry: on cc 10.7-11.x it lists exactly one
    entry with NO knobs (``{}`` is the complete record), and off the Rubin line it is absent -- on the SM100 line the
    d256 half graph lists exactly the 2x2-datapath row (slot 7 -> 20607, knobless too; the sm100 d512 row's envelope floor
    is exclusive at 256), on SM120 only that line's own continuum row (20600, with its tiles) lists."""
    assert any(s.name == "sdpa_bwd_sm107" for s in bwd_engines.ENGINE_SPECS), "sdpa_bwd_sm107 is not registered (plan s7)"
    plans = recommend("A", _facts(d_qk=256, d_v=256, dtype=cudnn.data_type.BFLOAT16, causal=False, device_cc=cc), _OFFERED)
    assert [p.engine_id for p in plans] == want
    assert all(p.knobs is None for p in plans if p.engine_id in (20604, 20607)), "the d256 half rows have no tile axis: {} is the complete record"
    assert all(p.mode is None and p.cpp_index is None for p in plans)


@pytest.mark.parametrize(
    "cc, listed",
    [((10, 0), True), ((10, 3), True), ((10, 4), True), ((10, 5), True), ((10, 6), True), ((10, 7), False), ((12, 0), False), ((8, 0), False)],
    ids=["cc10.0", "cc10.3", "cc10.4", "cc10.5", "cc10.6", "cc10.7", "cc12.0", "cc8.0"],
)
def test_sm100_d256_row_lists_one_knobless_entry_on_the_sm100_line(cc, listed):
    """The SM100 d256 bf16 / fp16 backward row (slot 7 -> 20607, the 2x2-datapath body) lists one knob-less entry on the
    SM100 line (cc 10.0-10.6: the row's ``sm_hi = 106`` admits every pre-Rubin Blackwell datacenter cc, like the MXFP8
    row) for the causal GQA graph too, and nowhere else (the other lines list their own rows); on the SM100 line it never
    lists for d != 256 or for a decode-shaped graph."""
    assert any(s.name == "sdpa_bwd_sm100_d256" for s in bwd_engines.ENGINE_SPECS), "sdpa_bwd_sm100_d256 is not registered"
    facts = _facts(d_qk=256, d_v=256, h_kv=2, dtype=cudnn.data_type.BFLOAT16, causal=True, device_cc=cc)
    plans = recommend("A", facts, _OFFERED)
    assert (20607 in [p.engine_id for p in plans]) == listed, plans
    assert all(p.knobs is None for p in plans if p.engine_id == 20607), "fixed geometry: {} is the complete record"
    if listed:
        assert [p.engine_id for p in plans] == [20607]
        assert 20607 not in [p.engine_id for p in recommend("A", _facts(d_qk=512, d_v=512, device_cc=cc), _OFFERED)]
        assert [p.engine_id for p in recommend("A", _facts(d_qk=128, d_v=128, device_cc=cc), _OFFERED)] == [20609]
        assert recommend("A", _facts(d_qk=256, d_v=256, s_q=1, device_cc=cc), _OFFERED) == []


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 6)])
def test_sm100_d128_dense_row_lists_one_knobless_entry(monkeypatch, cc):
    monkeypatch.setattr(bwd_engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    facts = _facts(device_cc=cc)
    assert [(p.engine_id, p.knobs) for p in recommend("A", facts, _OFFERED)] == [(20609, None)]
    grouped = recommend("A", _facts(device_cc=cc, h_kv=2, dtype=cudnn.data_type.BFLOAT16), _OFFERED)
    assert [(p.engine_id, p.knobs) for p in grouped] == ([(20609, None)] if cc in ((10, 0), (10, 3)) else [])
    assert recommend("A", _facts(device_cc=cc, h_kv=2, dtype=cudnn.data_type.HALF), _OFFERED) == []


@pytest.mark.parametrize(
    "cc, d, want",
    [
        ((10, 7), 512, [20608]),
        ((11, 0), 512, [20608]),
        ((10, 7), 264, [20608]),
        ((10, 7), 504, [20608]),
        ((10, 0), 512, [20602]),
        ((10, 3), 512, [20602]),
        ((10, 7), 256, [20604]),
        ((10, 7), 128, []),
        ((12, 0), 512, []),
    ],
    ids=["sm107-d512", "sm110-d512", "sm107-d264", "sm107-d504", "sm100-d512", "sm103-d512", "sm107-d256", "sm107-d128", "sm120-d512"],
)
def test_sm107_d512_row_lists_one_knobless_entry_on_the_rubin_line(cc, d, want):
    """The cc 10.7 d512 bf16 / fp16 backward row (slot 8 -> 20608) is fixed-geometry: on the Rubin line (cc 10.7-11.x) a
    d in (256, 512] graph lists exactly one entry with NO knobs (``{}`` is the complete record); the SM100 line keeps its
    own d512 row (20602); d = 256 on the Rubin line stays the d256 row's (20604: the envelope floor is exclusive at 256);
    d = 128 and SM120 list no python row for the half d512 graph."""
    assert any(s.name == "sdpa_bwd_sm107_d512" for s in bwd_engines.ENGINE_SPECS), "sdpa_bwd_sm107_d512 is not registered"
    plans = recommend("A", _facts(d_qk=d, d_v=d, dtype=cudnn.data_type.BFLOAT16, causal=False, device_cc=cc), _OFFERED)
    assert [p.engine_id for p in plans] == want
    assert all(p.knobs is None for p in plans if p.engine_id in (20602, 20604, 20608)), "the fixed-geometry rows have no tile axis: {} is the complete record"
    assert all(p.mode is None and p.cpp_index is None for p in plans)


@pytest.mark.parametrize("cc, want", [((10, 7), [20605]), ((10, 0), []), ((12, 0), [])], ids=["sm107", "sm100", "sm120"])
def test_sm107_fp8_row_lists_one_knobless_entry_on_the_rubin_line(cc, want):
    """The per-tensor FP8 d256 backward row (slot 5 -> 20605): one knob-less entry for an E4M3 ``sdpa_fp8_backward``
    graph with FP8 gradients on Rubin; no python row serves that graph off the Rubin line."""
    assert any(s.name == "sdpa_bwd_sm107_fp8" for s in bwd_engines.ENGINE_SPECS), "sdpa_bwd_sm107_fp8 is not registered (plan s7)"
    facts = _facts(d_qk=256, d_v=256, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.FP8_E4M3, is_fp8=True, causal=True, device_cc=cc)
    plans = recommend("A", facts, _OFFERED)
    assert [(p.engine_id, p.knobs) for p in plans] == [(e, None) for e in want]


@pytest.mark.parametrize(
    "cc, want",
    [((10, 7), [20606]), ((11, 0), [20606]), ((10, 0), [20603]), ((10, 3), [20603]), ((12, 0), [])],
    ids=["sm107", "sm110", "sm100", "sm103", "sm120"],
)
def test_sm107_mxfp8_row_lists_one_knobless_entry_on_the_rubin_line(cc, want):
    """The block-scale MXFP8 d256 backward row (slot 6 -> 20606): one knob-less entry for an E4M3 ``sdpa_mxfp8_backward`` graph
    with bf16 gradients on the Rubin line (the sm107 rows have no tile axis, unlike the SM100 MXFP8 row's single-point domain);
    on the SM100 line only that line's own row (20603) lists; nothing on SM120."""
    assert any(s.name == "sdpa_bwd_sm107_mxfp8" for s in bwd_engines.ENGINE_SPECS), "sdpa_bwd_sm107_mxfp8 is not registered"
    facts = _facts(d_qk=256, d_v=256, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_mxfp8=True, causal=True, device_cc=cc)
    plans = recommend("A", facts, _OFFERED)
    assert [p.engine_id for p in plans] == want
    assert all(p.knobs is None for p in plans if p.engine_id == 20606), "the Rubin MXFP8 row has no tile axis: {} is the complete record"


def test_sm100_d128_leads_the_backend_only_on_unwindowed_d128_thd(monkeypatch):
    """The default d128 backward row ranks ahead of the backend for d128 THD without a window and behind it elsewhere;
    the FROST opt-in keeps it first everywhere (cudnn.sdpa.bwd.heuristics._leads)."""
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.bwd import heuristics

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert not family.slots["sdpa_bwd_sm100_d128"].opt_in
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    eid = _OFFERED["sdpa_bwd_sm100_d128"]
    monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered: [PlanConfig(eid, None)])

    def order(**over):
        plans = heuristics.propose("A", _facts(device_cc=(10, 0), **over), _OFFERED)
        return ["backend" if is_backend_block(p) else "frost" for p in plans]

    assert order(thd=True) == ["frost"]
    assert order() == ["backend", "frost"]  # dense d128: parity
    assert order(thd=True, window_left=1024) == ["backend", "frost"]  # a window loses ~2x
    assert order(thd=True, d_qk=64, d_v=64) == ["backend", "frost"]
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert order() == ["frost"]


def test_sm100_d256_leads_the_backend_only_inside_its_measured_box(monkeypatch):
    """The default d256 backward row ranks ahead of the backend for d256 on cc 10.0 and behind it for D192x128, other parts,
    unmeasured features and mask-free ragged THD past S_q 4096; the FROST opt-in keeps it first everywhere."""
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.bwd import heuristics

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert not family.slots["sdpa_bwd_sm100_d256"].opt_in
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    eid = _OFFERED["sdpa_bwd_sm100_d256"]
    monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered: [PlanConfig(eid, None)])

    def order(**over):
        base = dict(device_cc=(10, 0), d_qk=256, d_v=256, dtype=cudnn.data_type.BFLOAT16, b=8, h_q=16, h_kv=2, s_q=4096, s_kv=4096)
        plans = heuristics.propose("A", _facts(**{**base, **over}), _OFFERED)
        return ["backend" if is_backend_block(p) else "frost" for p in plans]

    lead, trail = ["frost"], ["backend", "frost"]
    assert order() == lead
    assert order(causal=False) == lead
    assert order(thd=True, causal=False, max_total_seq_len_q=8 * 4096) == lead  # full-length THD
    assert order(thd=True, causal=False, b=32, s_q=1024, s_kv=1024, max_total_seq_len_q=24 * 1024) == lead  # short ragged
    assert order(thd=True, causal=False, max_total_seq_len_q=6 * 4096) == trail  # mask-free ragged past 4096
    assert order(thd=True, max_total_seq_len_q=6 * 4096) == lead  # causal ragged
    assert order(d_qk=192, d_v=128, h_kv=16) == trail
    assert order(device_cc=(10, 3)) == trail
    for unmeasured in (dict(deterministic=True), dict(has_bias=True), dict(has_dropout=True), dict(window_left=1024), dict(has_sink=True)):
        assert order(**unmeasured) == trail, unmeasured
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert order(d_qk=192, d_v=128, h_kv=16) == lead


def test_trail_only_rows_rank_behind_the_backend(monkeypatch):
    """The default 16-bit rows with no measured lead shard rank behind the backend's block (a correctness fallback for
    graphs the backend declines); the FROST opt-in keeps them first."""
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.bwd import heuristics

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    for name in heuristics.TRAIL_ONLY_ROWS:
        assert not family.slots[name].opt_in, name
        eid = _OFFERED[name]
        monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered, eid=eid: [PlanConfig(eid, None)])
        order = ["backend" if is_backend_block(p) else "frost" for p in heuristics.propose("A", _facts(), _OFFERED)]
        assert order == ["backend", "frost"], name
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    order = ["backend" if is_backend_block(p) else "frost" for p in heuristics.propose("A", _facts(), _OFFERED)]
    assert order == ["frost"]
