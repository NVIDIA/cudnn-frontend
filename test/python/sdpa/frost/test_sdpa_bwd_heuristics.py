# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDPA-backward heuristic's recommend contract (no GPU): every eligible
row lists exactly one entry, and the entry spells the tiles the lowering would
pick when handed no knobs -- so a recorded (engine_id, knobs) pins the kernel."""

import itertools

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
    "sdpa_bwd_sm90": 20634,
    "sdpa_bwd_sm100": 20602,
    "sdpa_bwd_sm100_d512_role_split": 20626,
    "sdpa_bwd_sm100_d512_2x2": 20627,
    "sdpa_bwd_sm100_mxfp8": 20603,
    "sdpa_bwd_sm107": 20604,  # retained record; delegated to the fixed native owner
    "sdpa_bwd_sm107_d256_f16_4x1": 20632,
    "sdpa_bwd_sm107_fp8": 20605,
    "sdpa_bwd_sm107_mxfp8": 20606,
    "sdpa_bwd_sm107_d192": 20610,
    "sdpa_bwd_sm100_d256": 20607,
    "sdpa_bwd_sm107_d512": 20608,
    "sdpa_bwd_sm100_d128": 20609,
    "sdpa_bwd_sm100_d128_recompute": 20628,
    "sdpa_bwd_sm100_d128_pair": 20629,
    "sdpa_bwd_sm100_d128_accumulation": 20630,
    "sdpa_bwd_sm100_d128_accumulation_2cta": 20631,
    "sdpa_bwd_sm120_f16_fused": 20618,
    "sdpa_bwd_sm80_f16_d64": 20619,
    "sdpa_bwd_sm80_f16_two_subgroup": 20620,
    "sdpa_bwd_sm120_f16_two_subgroup": 20621,
    "sdpa_bwd_sm100_d128_fp8_pair": 20622,
    "sdpa_bwd_sm100_d128_fp8_accumulation": 20623,
    "sdpa_bwd_sm100_mxfp8_pair": 20624,
    "sdpa_bwd_sm100_d256_mxfp8": 20625,
    "sdpa_bwd_sm89": 20635,
    "sdpa_bwd_sm89_f16_d64": 20636,
}


def _facts(**over):
    base = dict(is_backward=True, b=2, h_q=8, h_kv=8, s_q=256, s_kv=256, d_qk=128, d_v=128, dtype=cudnn.data_type.HALF, causal=True, device_cc=(12, 0))
    base.update(over)
    return SdpaGraphFacts(**base)


@pytest.mark.parametrize("d", [32, 40, 64, 128, 192, 256])
def test_sm120_row_lists_the_kernels_per_head_dim_default(d):
    plans = recommend("A", _facts(d_qk=d, d_v=d), _OFFERED)
    assert [p.engine_id for p in plans] == [20618]  # the fused body; the SM120 row keeps its packed and D > 256 bodies
    q_tile, kv_tile = DEFAULT_TILES[padded_head_dims(d, d)[0]]  # d=40 runs at the padded 64
    assert plans[0].knobs == SdpaBwdKnobs(tile_m=q_tile, tile_n=kv_tile)
    kt = cudnn.knob_type
    assert plans[0].knobs.to_public() == {kt.TILE_M: q_tile, kt.TILE_N: kv_tile}
    assert plans[0].mode is None and plans[0].cpp_index is None


def test_fixed_geometry_row_lists_no_knobs():
    # sm100 f16: the kernel takes no tile choice, so {} is the complete record.
    plans = recommend("A", _facts(d_qk=512, d_v=512, device_cc=(10, 0)), _OFFERED)
    assert [(p.engine_id, p.knobs) for p in plans] == [(20626, None), (20627, None)]


def test_single_point_domain_row_lists_its_sole_tiles():
    facts = _facts(d_qk=256, d_v=256, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.HALF, is_mxfp8=True, device_cc=(10, 0))
    plans = recommend("A", facts, _OFFERED)
    assert [(p.engine_id, p.knobs) for p in plans] == [(20625, SdpaBwdKnobs(tile_m=128, tile_n=128))]  # the MXFP8 row's D256 chain


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
        assert single_point or bwd_engines.policy_name(spec) in _DEFAULT_TILE_RESOLVERS, spec.name


@pytest.mark.parametrize(
    "cc, want",
    [((10, 7), [20632]), ((11, 0), [20607]), ((10, 0), [20607]), ((10, 3), [20607]), ((12, 0), [20618])],
    ids=["sm107", "sm110", "sm100", "sm103", "sm120"],
)
def test_sm107_half_row_lists_one_knobless_entry_on_the_rubin_line(cc, want):
    """The Rubin d256 bf16 / fp16 backward row (slot 4 -> 20632) is fixed-geometry: on cc 10.7-11.x it lists exactly one
    entry with NO knobs (``{}`` is the complete record), and off the Rubin line it is absent -- on the SM100 line the
    d256 half graph lists exactly the 2x2-datapath row (slot 7 -> 20607, knobless too; the sm100 d512 row's envelope floor
    is exclusive at 256), on SM120 only that line's own fused continuum body (20618, with its tiles) lists."""
    assert any(s.name == "sdpa_bwd_sm107" for s in bwd_engines.ENGINE_SPECS), "sdpa_bwd_sm107 is not registered (plan s7)"
    plans = recommend("A", _facts(d_qk=256, d_v=256, dtype=cudnn.data_type.BFLOAT16, causal=False, device_cc=cc), _OFFERED)
    assert [p.engine_id for p in plans] == want
    assert all(p.knobs is None for p in plans if p.engine_id in (20632, 20607)), "the d256 half rows have no tile axis: {} is the complete record"
    assert all(p.mode is None and p.cpp_index is None for p in plans)


@pytest.mark.parametrize(
    "cc, listed",
    [((10, 0), True), ((10, 3), True), ((10, 4), True), ((10, 5), True), ((10, 6), True), ((10, 7), False), ((12, 0), False), ((8, 0), False)],
    ids=["cc10.0", "cc10.3", "cc10.4", "cc10.5", "cc10.6", "cc10.7", "cc12.0", "cc8.0"],
)
def test_sm100_d256_row_lists_one_knobless_entry_on_the_sm100_line(monkeypatch, cc, listed):
    """The SM100 d256 bf16 / fp16 backward row (slot 7 -> 20607, the 2x2-datapath body) lists one knob-less entry on the
    SM100 line (cc 10.0-10.6: the row's ``sm_hi = 106`` admits every pre-Rubin Blackwell datacenter cc, like the MXFP8
    row) for the causal GQA graph too, and nowhere else (the other lines list their own rows); on the SM100 line it never
    lists for d != 256 or for a decode-shaped graph."""
    monkeypatch.setattr(bwd_engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    assert any(s.name == "sdpa_bwd_sm100_d256" for s in bwd_engines.ENGINE_SPECS), "sdpa_bwd_sm100_d256 is not registered"
    facts = _facts(d_qk=256, d_v=256, h_kv=2, dtype=cudnn.data_type.BFLOAT16, causal=True, device_cc=cc)
    plans = recommend("A", facts, _OFFERED)
    assert (20607 in [p.engine_id for p in plans]) == listed, plans
    assert all(p.knobs is None for p in plans if p.engine_id == 20607), "fixed geometry: {} is the complete record"
    if listed:
        assert [p.engine_id for p in plans] == [20607]
        assert 20607 not in [p.engine_id for p in recommend("A", _facts(d_qk=512, d_v=512, device_cc=cc), _OFFERED)]
        expected = [20630, 20628, 20629] if cc in ((10, 0), (10, 3)) else [20628]
        assert [p.engine_id for p in recommend("A", _facts(d_qk=128, d_v=128, device_cc=cc), _OFFERED)] == expected
        assert recommend("A", _facts(d_qk=256, d_v=256, s_q=1, device_cc=cc), _OFFERED) == []


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)], ids=["cc10.0", "cc10.3"])
def test_sm100_d256_row_lists_ahead_of_the_d192_half_row_on_dense_graphs(monkeypatch, cc):
    """With the opt-in d192 half row offered too, a half-precision (192, 128) MHA backward lists the d256 row (its native
    192/128 geometry) before the d192 accumulation row on a dense graph, and the d192 THD row first on a packed one."""
    monkeypatch.setattr(bwd_engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    offered = {**_OFFERED, "sdpa_bwd_sm100_d192": 20612}
    base = dict(d_qk=192, d_v=128, h_q=16, h_kv=16, dtype=cudnn.data_type.BFLOAT16, device_cc=cc)
    thd = dict(thd=True, max_total_seq_len_q=2 * 256, max_total_seq_len_kv=2 * 256)
    for causal in (False, True):
        assert [p.engine_id for p in recommend("A", _facts(**base, causal=causal), offered)] == [20607, 20612], causal
        assert [p.engine_id for p in recommend("A", _facts(**base, causal=causal, **thd), offered)] == [20612, 20607], causal


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 6), (10, 7), (11, 0)], ids=["cc10.0", "cc10.3", "cc10.6", "cc10.7", "cc11.0"])
def test_sm100_d128_dense_owners_keep_default_order(monkeypatch, cc):
    """Dense MHA offers recompute explicitly; grouped and THD retain the accumulation row."""
    from cudnn.sdpa.bwd.d128_contract import ACCUMULATION_CCS

    monkeypatch.setattr(bwd_engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    facts = _facts(device_cc=cc)
    expected = [(20630, None), (20628, None)] if cc in ACCUMULATION_CCS else [(20628, None)]
    if cc in ((10, 0), (10, 3)):
        expected.append((20629, None))
    assert [(p.engine_id, p.knobs) for p in recommend("A", facts, _OFFERED)] == expected
    grouped = recommend("A", _facts(device_cc=cc, h_kv=2, dtype=cudnn.data_type.BFLOAT16), _OFFERED)
    two_cta = [(20631, None)] if cc != (11, 0) else []  # the two-CTA accumulation row keeps its named parts (no Thor)
    assert [(p.engine_id, p.knobs) for p in grouped] == ([(20630, None)] + two_cta if cc in ACCUMULATION_CCS else [])
    assert ((10, 7) in ACCUMULATION_CCS) and ((10, 6) not in ACCUMULATION_CCS)  # the admitted parts are named, the envelope's hypothetical ccs are not
    assert recommend("A", _facts(device_cc=cc, h_kv=2, dtype=cudnn.data_type.HALF), _OFFERED) == []


@pytest.mark.parametrize(
    "cc, d, want",
    [
        ((10, 7), 512, [20608]),
        ((11, 0), 512, [20626, 20627]),  # Thor: the SM100 d512 explicit rows (sm_extra)
        ((10, 7), 264, [20608]),
        ((10, 7), 504, [20608]),
        ((10, 0), 512, [20626, 20627]),
        ((10, 3), 512, [20626, 20627]),
        ((10, 7), 256, [20632]),
        ((10, 7), 128, [20630, 20628]),  # accumulation default, followed by explicit recompute; never the d512 row
        ((11, 0), 128, [20630, 20628]),  # Thor: the SM100 d128 rows (sm_extra), the cc 10.7 order
        ((12, 0), 512, [20621]),  # SM120's two-subgroup body
    ],
    ids=["sm107-d512", "sm110-d512", "sm107-d264", "sm107-d504", "sm100-d512", "sm103-d512", "sm107-d256", "sm107-d128", "sm110-d128", "sm120-d512"],
)
def test_sm107_d512_row_lists_one_knobless_entry_on_the_rubin_line(monkeypatch, cc, d, want):
    """The cc 10.7 d512 bf16 / fp16 backward row (slot 8 -> 20608) is fixed-geometry: on the Rubin line (cc 10.7-11.x) a
    d in (256, 512] graph lists exactly one entry with NO knobs (``{}`` is the complete record); the SM100 line keeps its
    explicit d512 owners (20626 then 20627); d = 256 on the Rubin line stays the d256 row's (20632: the envelope floor is exclusive at 256);
    d = 128 on cc 10.7 lists accumulation then explicit recompute; past that range (cc 11.0) neither is listed,
    on SM120 the native wide row lists its 16x16 tile pair."""
    monkeypatch.setattr(bwd_engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    assert any(s.name == "sdpa_bwd_sm107_d512" for s in bwd_engines.ENGINE_SPECS), "sdpa_bwd_sm107_d512 is not registered"
    plans = recommend("A", _facts(d_qk=d, d_v=d, dtype=cudnn.data_type.BFLOAT16, causal=False, device_cc=cc), _OFFERED)
    assert [p.engine_id for p in plans] == want
    assert all(
        p.knobs is None for p in plans if p.engine_id in (20626, 20627, 20632, 20608)
    ), "the fixed-geometry rows have no tile axis: {} is the complete record"
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
    [((10, 7), [20606]), ((11, 0), []), ((10, 0), [20625]), ((10, 3), [20625]), ((12, 0), [])],
    ids=["sm107", "sm110", "sm100", "sm103", "sm120"],
)
def test_sm107_mxfp8_row_lists_one_knobless_entry_on_the_rubin_line(cc, want):
    """The block-scale MXFP8 d256 backward row (slot 6 -> 20606): one knob-less entry for an E4M3 ``sdpa_mxfp8_backward`` graph
    with bf16 gradients on the Rubin line (the sm107 rows have no tile axis, unlike the SM100 MXFP8 row's single-point domain);
    on the SM100 line only that line's own D256 chain (20625, the 20603 row's body) lists; nothing on SM120."""
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
    eid = _OFFERED["sdpa_bwd_sm100_d128_accumulation"]
    monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered: [PlanConfig(eid, None)])

    def order(**over):
        plans = heuristics.propose("A", _facts(**{"device_cc": (10, 0), **over}), _OFFERED)
        return ["backend" if is_backend_block(p) else "frost" for p in plans]

    assert order(thd=True) == ["frost", "backend"]
    assert order() == ["backend", "frost"]  # dense d128: parity
    assert order(thd=True, window_left=1024) == ["backend", "frost"]  # a window loses ~2x
    assert order(thd=True, d_qk=64, d_v=64) == ["backend", "frost"]
    assert order(thd=True, device_cc=(10, 3)) == ["frost", "backend"]  # cc 10.3 runs the cc 10.0 rule (the sm103 precedent)
    # cc 10.7 (Rubin, measured 2026-11-03): the backend serves no d128 THD backward plan there, so the row leads its box as on cc 10.0;
    # dense d128 read parity (+1.6 % at 16K, noise at 8K) and trails (heuristics.D128_LEAD_CCS)
    assert order(thd=True, device_cc=(10, 7)) == ["frost", "backend"]
    assert order(device_cc=(10, 7)) == ["backend", "frost"]
    assert order(thd=True, device_cc=(10, 7), window_left=1024) == ["backend", "frost"]
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert order() == ["frost", "backend"]


def test_sm100_d128_deterministic_entries_key_apart(monkeypatch):
    """Study sm100_det_bwd_20261122 owns deterministic dense d64 / d128 bf16 classes of ``sdpa_bwd_sm100_d128`` on cc 10.0 and cc 10.3.
    The row is in ``DETERMINISTIC_KEYED_ROWS``: a deterministic graph keys ``prefill_det``, so only these entries answer it, and through
    ``propose`` the kernel pair goes ahead of the backend exactly where its entry leads; every entry leads some graph of its class. The
    same graph without determinism keeps the hand rule (dense trails), as do the classes the study did not ship."""
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block
    from cudnn.sdpa import placement_table as pt
    from cudnn.sdpa.bwd import heuristics

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    row, study = "sdpa_bwd_sm100_d128", "sm100_det_bwd_20261122"
    eid = _OFFERED["sdpa_bwd_sm100_d128_pair"]
    monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered: [PlanConfig(eid, None)])
    assert row in pt.DETERMINISTIC_KEYED_ROWS

    def facts(**over):
        return _facts(**{"dtype": cudnn.data_type.BFLOAT16, "deterministic": True, **over})

    def first(f):
        return "backend" if is_backend_block(heuristics.propose("A", f, _OFFERED)[0]) else "frost"

    entries = {k: e for k, e in pt.TABLE.items() if e.study == study}
    assert len(entries) == 42
    led = set()
    for key, entry in entries.items():
        _, cc, _, (d, _), phase, layout, mask, _, group, off = key
        assert phase == "prefill_det" and layout == "dense" and mask in ("causal_br", "none", "window")
        heads = ((32, 32), (128, 128)) if group == "mha" else ((64, 8), (48, 3), (32, 2))
        for b, s, (h_q, h_kv) in itertools.product((1, 2, 8), (128, 1024, 8192), heads):
            shape = dict(device_cc=cc, d_qk=d, d_v=d, b=b, h_q=h_q, h_kv=h_kv, s_q=s + off, s_kv=s + off, causal=mask != "none")
            f = facts(**shape, window_left=127 if mask == "window" else None)
            assert pt.class_key(row, f) == key and pt.matched_entry(row, f) is entry
            lead = pt.lookup(row, f)
            assert first(f) == ("frost" if lead else "backend"), (key, b, s, h_q, h_kv)
            assert first(facts(**shape, deterministic=False, window_left=f.window_left)) == "backend"
            if lead:
                led.add(key)
    assert led == set(entries)
    # the GB nightlies' deterministic rows: wan 2.2 / ltx 2 (d128 MHA mask-free) and gpt-oss (d64 MHA, 128-key window) lead; llama's
    # d128 GQA 64/8 causal keeps the backend; classes without an entry (d64 causal GQA, cc 10.0 d128 mask-free GQA) and f16 trail
    for cc in ((10, 0), (10, 3)):
        assert first(facts(device_cc=cc, b=1, h_q=40, h_kv=40, s_q=7800, s_kv=7800, causal=False)) == "frost"
        assert first(facts(device_cc=cc, d_qk=64, d_v=64, b=2, h_q=128, h_kv=128, s_q=4096, s_kv=4096, window_left=127)) == "frost"
        assert first(facts(device_cc=cc, b=2, h_q=64, h_kv=8, s_q=4096, s_kv=4096)) == "backend"
        assert pt.matched_entry(row, facts(device_cc=cc, d_qk=64, d_v=64, b=2, h_q=64, h_kv=8, s_q=4096, s_kv=4096)) is None
        assert first(facts(device_cc=cc, b=1, h_q=40, h_kv=40, s_q=7800, s_kv=7800, causal=False, dtype=cudnn.data_type.HALF)) == "backend"
    assert pt.matched_entry(row, facts(device_cc=(10, 0), b=2, h_q=64, h_kv=8, s_q=4096, s_kv=4096, causal=False)) is None


def test_sm100_d256_leads_the_backend_only_inside_its_measured_box(monkeypatch):
    """The default d256 backward row ranks ahead of the backend for d256 on cc 10.0 and behind it for D192x128, other parts,
    unmeasured features and mask-free ragged THD past S_q 4096; on cc 10.3 the placement table decides bf16 THD graphs, and on
    both parts it decides the dense classes study sm100_d256_bwd_dense_20261122 shipped. The FROST opt-in keeps it first everywhere."""
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

    lead, trail = ["frost", "backend"], ["backend", "frost"]
    assert order() == lead
    assert order(causal=False) == lead
    assert order(thd=True, causal=False, max_total_seq_len_q=8 * 4096) == lead  # full-length THD
    assert order(thd=True, causal=False, b=32, s_q=1024, s_kv=1024, max_total_seq_len_q=24 * 1024) == lead  # short ragged
    assert order(thd=True, causal=False, max_total_seq_len_q=6 * 4096) == trail  # mask-free ragged past 4096
    assert order(thd=True, max_total_seq_len_q=6 * 4096) == lead  # causal ragged
    assert order(d_qk=192, d_v=128, h_kv=16) == trail
    assert order(device_cc=(10, 3)) == lead  # dense causal on cc 10.3: a table entry
    assert order(device_cc=(10, 3), causal=False) == trail  # dense mask-free on cc 10.3: no entry, and the box is cc 10.0's
    assert order(deterministic=True) == lead  # deterministic dense causal: a class of its own, with an entry
    assert order(deterministic=True, causal=False) == trail  # deterministic dense mask-free: no entry
    for unmeasured in (dict(has_bias=True), dict(has_dropout=True), dict(window_left=1024), dict(has_sink=True)):
        assert order(**unmeasured) == trail, unmeasured
    # cc 10.3 bf16 THD on the 128-key tile: the placement table's own entries (study sm103_d256_bwd_v2_20261031) decide
    thd103 = dict(device_cc=(10, 3), thd=True, max_total_seq_len_q=8 * 4096)
    assert order(**thd103) == lead
    assert order(device_cc=(10, 3), thd=True, causal=False, b=1, h_q=16, h_kv=16, s_q=1024, s_kv=1024, max_total_seq_len_q=1024) == lead
    assert order(device_cc=(10, 3), thd=True, causal=False, b=1, h_q=8, h_kv=8, s_q=256, s_kv=256, max_total_seq_len_q=256) == trail
    for unmeasured in (dict(deterministic=True), dict(has_bias=True), dict(window_left=1024), dict(dtype=cudnn.data_type.HALF)):
        assert order(**thd103, **unmeasured) == trail, unmeasured
    # off the 128-key tile: the twin entries (study sm103_d256_bwd_thd_offtile_20261031)
    assert order(device_cc=(10, 3), thd=True, s_q=4000, s_kv=4000, max_total_seq_len_q=8 * 4000) == lead
    assert order(device_cc=(10, 3), thd=True, causal=False, b=1, h_q=16, h_kv=16, s_q=3000, s_kv=3000, max_total_seq_len_q=3000) == lead
    assert order(device_cc=(10, 3), thd=True, causal=False, b=1, h_q=8, h_kv=8, s_q=1000, s_kv=1000, max_total_seq_len_q=1000) == trail
    assert order(device_cc=(10, 3), s_q=4000, s_kv=4000) == lead  # dense causal off the tile: a table entry
    assert order(device_cc=(10, 3), causal=False, s_q=4000, s_kv=4000) == trail  # dense mask-free off the tile: no entry
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert order(d_qk=192, d_v=128, h_kv=16) == lead


def test_sm100_d256_dense_entries_key_deterministic_graphs_apart(monkeypatch):
    """Study sm100_d256_bwd_dense_20261122 owns the dense d256 bf16 classes it shipped on cc 10.0 and cc 10.3. The row is in
    ``DETERMINISTIC_KEYED_ROWS``, so a deterministic graph keys ``prefill_det`` and only a deterministic entry answers it. Each entry
    leads a large graph of its class. The probe's small causal square (b 8, 8/8, S 128), slower on FROST on both parts, keeps the
    backend in both modes, including cc 10.0's non-deterministic graph, which the hand-written box led before the study."""
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block
    from cudnn.sdpa import placement_table as pt
    from cudnn.sdpa.bwd import heuristics

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    row, study = "sdpa_bwd_sm100_d256", "sm100_d256_bwd_dense_20261122"
    eid = _OFFERED[row]
    monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered: [PlanConfig(eid, None)])
    assert row in pt.DETERMINISTIC_KEYED_ROWS

    def facts(**over):
        base = dict(d_qk=256, d_v=256, dtype=cudnn.data_type.BFLOAT16, b=4, h_q=32, h_kv=8, s_q=8192, s_kv=8192)
        return _facts(**{**base, **over})

    def first(f):
        return "backend" if is_backend_block(heuristics.propose("A", f, _OFFERED)[0]) else "frost"

    entries = {k: e for k, e in pt.TABLE.items() if e.study == study}
    assert len(entries) == 18
    for key, entry in entries.items():
        _, cc, _, _, phase, layout, mask, _, group, off = key
        assert layout == "dense" and mask in ("causal_br", "none") and group in ("gqa", "mha")
        s = 8192 + off
        f = facts(device_cc=cc, causal=mask == "causal_br", h_kv=8 if group == "gqa" else 32, s_q=s, s_kv=s, deterministic=phase == "prefill_det")
        assert pt.class_key(row, f) == key
        assert pt.matched_entry(row, f) is entry
        assert first(f) == "frost", key
    small = dict(b=8, h_q=8, h_kv=8, s_q=128, s_kv=128, causal=True)
    qwen = dict(b=1, h_q=32, h_kv=2, causal=True)  # the GB nightlies' d256 training backward rows
    for cc in ((10, 0), (10, 3)):
        for det in (False, True):
            assert pt.matched_entry(row, facts(device_cc=cc, deterministic=det, **small)).study == study
            assert first(facts(device_cc=cc, deterministic=det, **small)) == "backend", (cc, det)
            assert first(facts(device_cc=cc, deterministic=det, **qwen)) == "frost", (cc, det)
    # classes the study did not ship, and graphs it did not time, keep the backend first on cc 10.3
    assert pt.matched_entry(row, facts(device_cc=(10, 3), causal=False)) is None
    assert first(facts(device_cc=(10, 3), causal=False)) == "backend"
    assert first(facts(device_cc=(10, 3), causal=True, thd=True, max_total_seq_len_q=4 * 8192, deterministic=True)) == "backend"
    assert first(facts(device_cc=(10, 3), causal=True, dtype=cudnn.data_type.HALF)) == "backend"


def test_trail_only_rows_rank_behind_the_backend(monkeypatch):
    """The default rows with no measured lead shard sit after the LAST_RESORT marker: behind every requested mode's
    backend block (a correctness fallback for graphs the backend declines); the FROST opt-in keeps them first."""
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block, is_last_resort
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.bwd import heuristics

    tag = lambda p: "backend" if is_backend_block(p) else "last_resort" if is_last_resort(p) else "frost"
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    specs = {s.name: s for s in bwd_engines.ENGINE_SPECS}
    for name in heuristics.TRAIL_ONLY_ROWS:
        # A delegating row never runs itself: its default-offered explicit owners are what trails.
        owners = sorted(specs[name].capabilities.delegated) or [name]
        offered = [n for n in owners if not family.slots[n].opt_in]
        assert offered, name
        for owner in offered:
            eid = _OFFERED[owner]
            monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered, eid=eid: [PlanConfig(eid, None)])
            # d256: no study timed these head dims, so no placement_table entry lifts the row
            assert [tag(p) for p in heuristics.propose("A", _facts(d_qk=256, d_v=256), _OFFERED)] == ["backend", "last_resort", "frost"], owner
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert [tag(p) for p in heuristics.propose("A", _facts(), _OFFERED)] == ["frost", "backend"]


def test_measured_lead_lifts_a_trail_only_row_only_inside_its_classes(monkeypatch):
    """A placement_table entry puts a trail-only row ahead of the backend on the measured part and classes; any other graph
    keeps the row behind LAST_RESORT."""
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block, is_last_resort
    from cudnn.sdpa.bwd import heuristics

    tag = lambda p: "backend" if is_backend_block(p) else "last_resort" if is_last_resort(p) else "frost"
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    eid = _OFFERED["sdpa_bwd_sm120"]
    monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered: [PlanConfig(eid, None)])
    order = lambda **over: [tag(p) for p in heuristics.propose("A", _facts(**{"device_cc": (12, 1), **over}), _OFFERED)]
    trails = ["backend", "last_resort", "frost"]

    for dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
        for d in (64, 128):
            assert order(dtype=dtype, d_qk=d, d_v=d)[0] == "frost"
            assert order(dtype=dtype, d_qk=d, d_v=d, causal=False)[0] == "frost"
            assert order(dtype=dtype, d_qk=d, d_v=d, window_left=256)[0] == "frost"
            assert order(dtype=dtype, d_qk=d, d_v=d, h_q=32, h_kv=8, s_q=300, s_kv=300)[0] == "frost"
    assert order(device_cc=(10, 0)) == trails  # another part
    assert order(d_qk=256, d_v=256) == trails  # head dims no study timed
    assert order(s_q=4, s_kv=4) == trails  # MTP-shaped
    assert order(thd=True, max_total_seq_len_q=512) == trails  # packed layout
    assert order(padded=True) == trails  # runtime lengths
    assert order(seq_q_trim=True) == trails
    assert order(causal=False, s_q=512, s_kv=1024)[0] == "frost"  # the GB10 studies are not square-only


@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
def test_sm89_fixed_d64_leads_only_for_measured_square_class(monkeypatch, dtype):
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block
    from cudnn.sdpa import placement_table
    from cudnn.sdpa.bwd import heuristics

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    fixed = _OFFERED["sdpa_bwd_sm89_f16_d64"]
    generic = _OFFERED["sdpa_bwd_sm89"]
    offer = {name: _OFFERED[name] for name in ("sdpa_bwd_sm89_f16_d64", "sdpa_bwd_sm89")}
    base = dict(d_qk=64, d_v=64, dtype=dtype, causal=False, device_cc=(8, 9))

    def order(**overrides):
        return heuristics.propose("A", _facts(**{**base, **overrides}), offer)

    short_dtype = "f16" if dtype == cudnn.data_type.HALF else "bf16"
    study = placement_table.matched_entry("sdpa_bwd_sm89_f16_d64", _facts(**base))
    assert study.study == f"sm89_bwd_d64_4090_{short_dtype}_20261118"
    assert placement_table.class_key("sdpa_bwd_sm89_f16_d64", _facts(**base, s_q=192, s_kv=192)) is None
    for length in (128, 768, 2048):
        assert order(s_q=length, s_kv=length)[0].engine_id == fixed
    for overrides in (
        dict(s_q=128, s_kv=256),
        dict(s_q=256, s_kv=128),
        dict(s_q=192, s_kv=192),
        dict(s_q=2176, s_kv=2176),
        dict(b=5),
        dict(h_q=3, h_kv=3),
        dict(h_q=17, h_kv=17),
        dict(causal=True),
        dict(padded=True),
        dict(deterministic=True),
        dict(h_kv=4),
        dict(d_qk=128, d_v=128),
    ):
        assert is_backend_block(order(**overrides)[0]), overrides

    # A placement entry for the fixed body must not lift the generic owner.
    monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered: [PlanConfig(generic, None)])
    assert is_backend_block(order()[0])


def test_rtx5090_backward_entries_lift_the_sm120_row_on_cc_12_0(monkeypatch):
    """cc 12.0 has its own measured dense d64 / d128 classes for the trail-only row, separate from cc 12.1; anything outside them keeps trailing."""
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block, is_last_resort
    from cudnn.sdpa import placement_table as pt
    from cudnn.sdpa.bwd import heuristics

    tag = lambda p: "backend" if is_backend_block(p) else "last_resort" if is_last_resort(p) else "frost"
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    eid = _OFFERED["sdpa_bwd_sm120"]
    monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered: [PlanConfig(eid, None)])
    facts = lambda **over: _facts(**{"device_cc": (12, 0), **over})
    order = lambda **over: [tag(p) for p in heuristics.propose("A", facts(**over), _OFFERED)]
    trails = ["backend", "last_resort", "frost"]

    for dtype, study in ((cudnn.data_type.HALF, "sm120_bwd_f16_20261107"), (cudnn.data_type.BFLOAT16, "sm120_bwd_bf16_20261107")):
        for d in (64, 128):
            assert order(dtype=dtype, d_qk=d, d_v=d)[0] == "frost"
            assert order(dtype=dtype, d_qk=d, d_v=d, causal=False)[0] == "frost"
            assert order(dtype=dtype, d_qk=d, d_v=d, window_left=256)[0] == "frost"
            assert order(dtype=dtype, d_qk=d, d_v=d, h_q=32, h_kv=8, s_q=300, s_kv=300)[0] == "frost"
            assert pt.TABLE[pt.class_key("sdpa_bwd_sm120", facts(dtype=dtype, d_qk=d, d_v=d))].study == study
            for s_q, s_kv in ((512, 1024), (1024, 512)):  # the key does not see squareness, the guard does
                assert pt.lookup("sdpa_bwd_sm120", facts(dtype=dtype, d_qk=d, d_v=d, causal=False, s_q=s_q, s_kv=s_kv)) is True
                assert order(dtype=dtype, d_qk=d, d_v=d, causal=False, s_q=s_q, s_kv=s_kv) == trails
            # Deterministic graphs are classes of their own (sm120_det_bwd_20261117, bf16 only; no holdout cell exercised d128 causal MHA).
            det = order(dtype=dtype, d_qk=d, d_v=d, deterministic=True)
            if (dtype, d) == (cudnn.data_type.BFLOAT16, 64):
                assert det[0] == "frost", det
            else:
                assert det == trails, (dtype, d, det)
            assert order(dtype=dtype, d_qk=d, d_v=d, deterministic=True, s_q=512, s_kv=1024) == trails  # square-only study
    assert order(d_qk=256, d_v=256) == trails  # head dims no study timed
    assert order(s_q=4, s_kv=4) == trails  # MTP-shaped
    assert order(thd=True, max_total_seq_len_q=512) == trails  # packed layout
    assert order(padded=True) == trails  # runtime lengths
    assert order(seq_q_trim=True) == trails


def test_a100_backward_window_entries_lift_only_square_sm80_graphs(monkeypatch):
    """Every cc 8.0 class with a square study entry leads on a square graph and trails on a non-square graph of the same class, which lookup still answers True for."""
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block, is_last_resort
    from cudnn.sdpa import placement_table as pt
    from cudnn.sdpa.bwd import heuristics

    tag = lambda p: "backend" if is_backend_block(p) else "last_resort" if is_last_resort(p) else "frost"
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    eid = _OFFERED["sdpa_bwd_sm80"]
    monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered: [PlanConfig(eid, None)])
    facts = lambda **over: _facts(**{"device_cc": (8, 0), "window_left": 256, "bottom_right": True, **over})
    order = lambda **over: [tag(p) for p in heuristics.propose("A", facts(**over), _OFFERED)]
    trails = ["backend", "last_resort", "frost"]
    dtypes = {"bf16": cudnn.data_type.BFLOAT16, "f16": cudnn.data_type.HALF}

    entries = [k for k, e in pt.TABLE.items() if e.study.startswith("sm80_bwd_square_")]
    assert entries
    for key in entries:
        dtype, (d, _), group, off_tile = dtypes[key[2]], key[3], key[8], key[9]
        kv = 4096 + (37 if off_tile else 0)
        common = dict(dtype=dtype, d_qk=d, d_v=d, h_q=32, h_kv=8 if group == "gqa" else 32)
        assert pt.class_key("sdpa_bwd_sm80", facts(s_q=kv, s_kv=kv, **common)) == key
        assert order(s_q=kv, s_kv=kv, **common)[0] == "frost"
        assert pt.class_key("sdpa_bwd_sm80", facts(s_q=512, s_kv=kv, **common)) == key  # the key does not see squareness ...
        assert pt.lookup("sdpa_bwd_sm80", facts(s_q=512, s_kv=kv, **common)) is True
        assert order(s_q=512, s_kv=kv, **common) == trails  # ... and the guard does
        assert order(s_q=kv, s_kv=kv, padded=True, **common) == trails
        assert order(s_q=kv, s_kv=kv, window_left=None, **common) == trails
        assert order(s_q=kv, s_kv=kv, device_cc=(8, 6), **common) == trails


def test_rubin_d256_backward_entries_lead_thd_and_full_length_bshd(monkeypatch):
    """cc 10.7 d256 bf16 backward (studies sm107_d256_bwd_20261122 and its dense causal follow-up): the trail-only sm107 row's d256 body
    leads the backend on every class the studies shipped, THD graphs included (the graph analyzer marks every THD graph padded). A BSHD
    graph of the same class with runtime lengths keeps trailing, as do f16, windowed and deterministic graphs, which no study timed."""
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import is_backend_block, is_last_resort
    from cudnn.sdpa import placement_table as pt
    from cudnn.sdpa.bwd import heuristics

    tag = lambda p: "backend" if is_backend_block(p) else "last_resort" if is_last_resort(p) else "frost"
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    eid = _OFFERED["sdpa_bwd_sm107_d256_f16_4x1"]
    monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered: [PlanConfig(eid, None)])
    facts = lambda **over: _facts(**{"device_cc": (10, 7), "dtype": cudnn.data_type.BFLOAT16, "d_qk": 256, "d_v": 256, **over})
    order = lambda **over: [tag(p) for p in heuristics.propose("A", facts(**over), _OFFERED)]
    trails = ["backend", "last_resort", "frost"]

    studies = ("sm107_d256_bwd_20261122", "sm107_d256_bwd_causal_v2_20261122")
    entries = [k for k, e in pt.TABLE.items() if e.study in studies]
    assert len(entries) == 24  # dense / thd / thd_ragged x causal / mask-free x MHA / GQA x on / off the tile
    for key in entries:
        layout, mask, group, off_tile = key[5], key[6], key[8], key[9]
        s = 8192 + (1 if off_tile else 0)  # long enough to pass every floor the studies fitted
        shape = dict(b=4, h_q=32, h_kv=8 if group == "gqa" else 32, s_q=s, s_kv=s, causal=mask == "causal_br")
        if layout != "dense":
            shape.update(thd=True, padded=True, max_total_seq_len_q=4 * s if layout == "thd" else 2 * s)
        assert pt.class_key("sdpa_bwd_sm107", facts(**shape)) == key
        assert order(**shape)[0] == "frost", key
        if layout == "dense":  # the key does not see per-batch lengths; the guard does
            assert order(**shape, padded=True) == trails, key
            assert order(**shape, seq_q_trim=True) == trails, key
        assert order(**shape, dtype=cudnn.data_type.HALF) == trails, key
        assert order(**shape, window_left=1024, bottom_right=True) == trails, key
        assert order(**shape, deterministic=True) == trails, key

    # The Rubin nightly's d256 backward shapes: context-parallel THD chunks, ragged THD squares and dense causal GQA squares lead.
    nightly = (
        (dict(b=64, h_q=32, h_kv=8, s_q=1024, s_kv=8192, bottom_right=True, thd=True, padded=True, max_total_seq_len_q=64 * 1024), studies[0]),
        (dict(b=64, h_q=8, h_kv=2, s_q=8192, s_kv=8192, thd=True, padded=True, max_total_seq_len_q=32 * 8192), studies[0]),
        (dict(b=1, h_q=32, h_kv=2, s_q=8192, s_kv=8192), studies[1]),
    )
    for shape, study in nightly:
        assert pt.matched_entry("sdpa_bwd_sm107", facts(**shape)).study == study, shape
        assert order(**shape)[0] == "frost", shape
    # The dense causal off-tile chunks the first study's holdout lost (1.149 and 1.132 of the backend) keep the backend.
    assert order(b=32, h_q=6, h_kv=3, s_q=64, s_kv=257, bottom_right=True) == trails  # small work
    assert order(b=4, h_q=40, h_kv=40, s_q=512, s_kv=4097, bottom_right=True) == trails  # few query tokens


@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("mask", ["none", "causal", "window"])
@pytest.mark.parametrize("h_kv", [6, 2])
def test_sm120_short_square_class_keeps_default_and_long_square_leads(monkeypatch, dtype, d, mask, h_kv):
    """All 24 off-tile classes distinguish an unqualified first partial tile from qualified long squares."""
    from cudnn.engines.heuristics import BACKEND
    from cudnn.sdpa import placement_table as pt
    from cudnn.sdpa.bwd.heuristics import propose

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    for s in (5, 16, 63, 64, 127, 128, 129, 2037):
        facts = _facts(dtype=dtype, d_qk=d, d_v=d, h_q=6, h_kv=h_kv, s_q=s, s_kv=s, causal=mask != "none", window_left=3 if mask == "window" else None)
        key = pt.class_key("sdpa_bwd_sm120", facts)
        plans = propose("A", facts, {"sdpa_bwd_sm120": 20600, "sdpa_bwd_sm120_f16_fused": 20618})
        if s < 128:
            assert key[4] == "prefill_short_square"
            assert pt.lookup("sdpa_bwd_sm120", facts) is None
            assert plans[0] is BACKEND
        else:
            assert key[4] == "prefill"
            assert pt.lookup("sdpa_bwd_sm120", facts) is True
            assert plans[0].engine_id == 20618  # the fused body inherits the row's measured entries


@pytest.mark.parametrize("row,cc", [("sdpa_bwd_sm120", (12, 1)), ("sdpa_bwd_sm80", (8, 0)), ("sdpa_fwd_prefill_sm120", (12, 0))])
def test_short_square_class_does_not_reclassify_other_measured_rows(row, cc):
    from cudnn.sdpa import placement_table as pt

    for s in (5, 16, 63, 127, 128, 129):
        assert pt.class_key(row, _facts(s_q=s, s_kv=s, device_cc=cc))[4] == "prefill"


@pytest.mark.parametrize("sq,sk,phase", [(1, 1, "q1"), (4, 4, "mtp"), (5, 7, "prefill"), (127, 128, "prefill")])
def test_sm120_short_square_split_preserves_other_phases(sq, sk, phase):
    from cudnn.sdpa import placement_table as pt

    assert pt.class_key("sdpa_bwd_sm120", _facts(s_q=sq, s_kv=sk))[4] == phase
