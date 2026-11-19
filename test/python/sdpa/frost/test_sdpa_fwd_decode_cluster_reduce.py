# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLUSTER_K_REDUCE on the SM100 D256 swap-AB decode tile: the knob, the heuristics rule, the declines and the binding.

No device and no kernel build: the rule is a pure function of graph facts, so every cell here pins POLICY on synthetic
facts. The device cells are in ``sdpa_decode_cluster_reduce_sm100.py``.
"""

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.engines.manifest import MANIFEST
from cudnn.sdpa.fwd import engines, heuristics as heur, placement
from cudnn.sdpa.fwd import prepared as prep
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from cudnn.sdpa.fwd.kernels.sm100.d256_f16_contract import LOOKAHEAD, SWAP_AB
from cudnn.sdpa.fwd.kernels.sm100.d256_f16_swap_ab import contract as swap_ab
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0

SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()
NAME_OF = {engine_id: name for name, engine_id in OFFERED.items()}


def facts(b=1, s_q=1, s_kv=2048, sm=148, replay=False, cc=(10, 0), **overrides):
    """A paged 24 / 4-head D256 bf16 decode graph; a multi-token step carries the bottom-right causal mask."""
    kw = dict(causal=True, bottom_right=True) if s_q > 1 else {}
    kw.update(overrides)
    return SdpaGraphFacts(
        **dict(
            dict(
                b=b,
                h_q=24,
                h_kv=4,
                s_q=s_q,
                s_kv=s_kv,
                d_qk=256,
                d_v=256,
                dtype=cudnn.data_type.BFLOAT16,
                device_cc=cc,
                device_sm_count=sm,
                padded=True,
                has_paged_kv=True,
                page_size=64,
                cuda_graph_replay=replay,
            ),
            **kw,
        )
    )


def _plans(f):
    return [(NAME_OF[p.engine_id], p.knobs) for p in heur.recommend("A", f, OFFERED)]


def _lead_swap_ab(f):
    return next((k for name, k in _plans(f) if name == SWAP_AB), None)


GRID = [
    (b, s_q, s_kv, sm, replay)
    for sm in (148, 152)
    for replay in (False, True)
    for s_q in (1, 2, 3, 6, 8)
    for b in (1, 2, 4, 8, 16, 64)
    for s_kv in (128, 512, 1024, 1536, 2048, 3072, 4096, 8192, 32768)
]


def test_knob_is_a_public_zero_one_knob_that_an_unset_record_does_not_carry():
    kt = cudnn.knob_type
    assert SdpaFwdKnobs().cluster_reduce is None and kt.CLUSTER_K_REDUCE not in SdpaFwdKnobs(split_kv=8).to_public()
    knobs = SdpaFwdKnobs(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=True, split_kv=8, cluster_reduce=True)
    public = knobs.to_public()
    assert public[kt.CLUSTER_K_REDUCE] == 1 and SdpaFwdKnobs.from_public(public) == knobs
    assert SdpaFwdKnobs.from_public({int(k): v for k, v in public.items()}) == knobs
    assert SdpaFwdKnobs.from_public({kt.CLUSTER_K_REDUCE: 0}).cluster_reduce is False
    assert SdpaFwdKnobs.from_public({kt.CLUSTER_K_REDUCE: True}).cluster_reduce is True
    with pytest.raises(ValueError, match="CLUSTER_K_REDUCE must be 0 or 1"):
        SdpaFwdKnobs.from_public({kt.CLUSTER_K_REDUCE: 2})
    with pytest.raises(ValueError, match="must be an int"):
        SdpaFwdKnobs.from_public({kt.CLUSTER_K_REDUCE: "1"})


def test_only_the_swap_ab_row_carries_the_form():
    """The capability domain: one row has True, and every other row declines a request with the same reason."""
    with_form = [s.name for s in engines.ENGINE_SPECS if True in s.capabilities.cluster_reduces]
    assert with_form == [SWAP_AB]
    assert all(False in s.capabilities.cluster_reduces for s in engines.ENGINE_SPECS)
    request = SdpaFwdKnobs(split_kv=4, cluster_reduce=True)
    for spec in engines.ENGINE_SPECS:
        if spec.name != SWAP_AB:
            assert "is a form of the SM100 D256 swap-AB split decode tile" in engines.mismatch(spec.capabilities, facts(), request), spec.name


@pytest.mark.parametrize(
    "overrides,knobs,reason",
    [
        ({}, dict(split_kv=1), "needs 2 to 8 KV chunks"),
        ({}, dict(split_kv=16), "needs 2 to 8 KV chunks"),
        ({}, dict(split_kv=2, tile_m=32), "16-column D256 swap-AB rendering"),
        (dict(cc=(10, 3)), dict(split_kv=8), "verified on compute capability 10.0 only"),
        (dict(has_stats_log2=True), dict(split_kv=4), "does not carry a sink, base-2 Stats"),
    ],
    ids=["unsplit", "split16", "tile32", "cc103", "log2-stats"],
)
def test_row_declines_a_request_outside_the_form(overrides, knobs, reason):
    caps = SPECS[SWAP_AB].capabilities
    request = SdpaFwdKnobs(**{**dict(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=True, cluster_reduce=True), **knobs})
    f = facts(**overrides)
    assert reason in engines.mismatch(caps, f, request)
    # only the request is declined: the same set without the knob is served
    assert engines.mismatch(caps, f, dataclasses.replace(request, cluster_reduce=None)) is None


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        (dict(thd=True), "does not carry"),
        (dict(has_sink=True), "does not carry"),
        (dict(gated=True), "does not carry"),
        (dict(o_matches_qkv=False), "does not carry"),
        (dict(use_pdl=True), "programmatic-dependent-launch"),
        (dict(d_shape=(128, 128)), "16-column D256"),
        (dict(split_kv=9), "2 to 8"),
    ],
)
def test_contract_predicate_rows(kwargs, reason):
    base = dict(device_cc=(10, 0), d_shape=(256, 256), tile=16, split_kv=8)
    assert swap_ab.cluster_reduce_decline(**base) is None
    assert all(swap_ab.cluster_reduce_decline(**{**base, "split_kv": s}) is None for s in range(2, 9))
    assert reason in swap_ab.cluster_reduce_decline(**{**base, **kwargs})


def test_rule_adds_removes_and_reorders_nothing(monkeypatch):
    """Clearing the knob in every listed set gives the list of a tree without the rule, entry for entry: no set appears,
    disappears or moves, and no split count changes."""
    marked = {}
    for cell in GRID:
        f = facts(*cell)
        marked[cell] = _plans(f)
    monkeypatch.setattr(heur, "fused_selected", lambda *_: False)
    count = 0
    for cell in GRID:
        plain = _plans(facts(*cell))
        assert all(not k.cluster_reduce for _, k in plain)
        cleared = [(name, dataclasses.replace(k, cluster_reduce=None)) for name, k in marked[cell]]
        assert cleared == plain, cell
        count += sum(bool(k.cluster_reduce) for _, k in marked[cell])
    assert count > 0


def test_rule_is_pure_and_every_marked_set_is_honoured(monkeypatch):
    """No device query on the planning path, and a set the rule marks is a set the row admits."""
    from cudnn.frost import occupancy

    def refuse(*_args, **_kwargs):
        raise AssertionError("the planning path queried the device")

    monkeypatch.setattr(occupancy, "occupancy_map", refuse)
    monkeypatch.setattr(occupancy, "max_active_clusters", refuse)
    caps = SPECS[SWAP_AB].capabilities
    for cell in GRID:
        f = facts(*cell)
        for name, knobs in _plans(f):
            if knobs.cluster_reduce:
                assert name == SWAP_AB and knobs.pack_gqa and knobs.tile_m == 16 and 2 <= knobs.split_kv <= 8, cell
                assert engines.mismatch(caps, f, knobs) is None, cell
                units = f.b * 4
                assert units <= heur.fused_active_clusters(f.device_sm_count, knobs.split_kv), cell


@pytest.mark.parametrize(
    "b,s_q,s_kv,replay,split,reduced",
    [
        # one user, one or two tokens: unsplit up to 1,024 declared keys, then today's 4 / 8 chunks reduced in the launch
        (1, 1, 512, False, 1, False),
        (1, 1, 1024, False, 1, False),
        (1, 1, 1536, False, 4, True),
        (1, 2, 2048, False, 8, True),
        (1, 1, 3072, False, 8, True),
        (1, 1, 4096, False, 16, False),  # 16 chunks are past one cluster: two launches as before
        (1, 2, 1024, True, 4, True),  # a replay-declared graph splits earlier
        # two and four users inside the conservative one-wave bound
        (2, 1, 1536, False, 4, True),
        (2, 2, 6144, False, 8, True),
        (2, 1, 8192, False, 16, False),
        (4, 2, 1536, False, 4, True),
        (4, 1, 2560, False, 4, True),
        (4, 1, 3072, False, 8, False),  # 16 clusters of 8 are past the bound without an occupancy row
        (8, 1, 2048, False, 4, False),
        (16, 1, 8192, False, 2, False),
        (64, 1, 2048, False, 1, False),
    ],
)
def test_rule_table_on_148_sms(b, s_q, s_kv, replay, split, reduced):
    lead = _lead_swap_ab(facts(b, s_q, s_kv, 148, replay))
    assert lead.pack_gqa and (lead.split_kv, bool(lead.cluster_reduce)) == (split, reduced)


def test_unpacked_and_multi_token_plans_are_not_marked():
    """Three to eight tokens: the first plan stays the prefill kernel, and the decode tile's unpacked set keeps two launches unless
    a caller sets the knob, which the row then honours."""
    for s_q in (3, 4, 6, 8):
        f = facts(1, s_q, 2048)
        plans = _plans(f)
        assert plans[0][0] == LOOKAHEAD
        unpacked = [k for name, k in plans if name == SWAP_AB and not k.pack_gqa]
        assert unpacked and all(not k.cluster_reduce for k in unpacked)
        assert engines.mismatch(SPECS[SWAP_AB].capabilities, f, dataclasses.replace(unpacked[0], cluster_reduce=True)) is None


def test_occupancy_row_and_measured_losses_steer_the_rule(monkeypatch):
    assert heur.FUSED_ACTIVE_CLUSTERS.get((152, 8)) is None or heur.FUSED_ACTIVE_CLUSTERS[(152, 8)] >= 9
    assert [heur.fused_active_clusters(1000, s) for s in (2, 4, 8)] == [250, 125, 62]  # a part without a row: half the SMs
    f = facts(4, 1, 3072, sm=1000)
    assert _lead_swap_ab(f).split_kv > 1
    # a recorded occupancy admits what the closed form holds back, and nothing else moves
    monkeypatch.setattr(heur, "FUSED_ACTIVE_CLUSTERS", {(148, 8): 18})
    lead = _lead_swap_ab(facts(4, 1, 3072))
    assert (lead.split_kv, lead.cluster_reduce) == (8, True)
    assert _lead_swap_ab(facts(8, 1, 4096)).cluster_reduce is None
    # a split count measured behind the two-launch form keeps it
    monkeypatch.setattr(heur, "FUSED_LOSES", frozenset({8}))
    assert (_lead_swap_ab(facts(1, 1, 2048)).split_kv, _lead_swap_ab(facts(1, 1, 2048)).cluster_reduce) == (8, None)
    assert _lead_swap_ab(facts(1, 1, 1536)).cluster_reduce is True


def test_rule_proposes_nothing_off_the_verified_part():
    for cell in GRID:
        f = facts(*cell, cc=(10, 3))
        assert all(not k.cluster_reduce for _, k in _plans(f)), cell
    assert not heur.fused_selected(facts(sm=0), SdpaFwdKnobs(tile_m=16, pack_gqa=True, split_kv=8))


@pytest.mark.parametrize("s_kv", [128, 2048, 16384])
def test_default_placement_is_unchanged(monkeypatch, s_kv):
    """With the engines' opt-in flag unset: one user x one token still ranks the backend first up to 16,384 declared keys
    (the marked plan does not change its class), two tokens lead as before, and six tokens lead with the prefill kernel."""
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    spec = SPECS[SWAP_AB]
    one = facts(1, 1, s_kv)
    assert placement.place(spec, one, _lead_swap_ab(one)) == placement.TRAIL
    two = facts(1, 2, s_kv)
    assert placement.place(spec, two, _lead_swap_ab(two)) == placement.LEAD
    six = _plans(facts(1, 6, s_kv))
    assert six[0][0] == LOOKAHEAD


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("requested", [False, True])
def test_adapter_passes_the_keywords_only_for_a_requested_plan(stats, requested):
    """The Blackwell-line tile reduces in the launch on request, never by itself; without the request the compile
    call is the one of a tree without the form (no extra compile key per Stats presence)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    def compile(*, has_lse=True, lse_kind="dense", d_qk=256, d_v=256, paged_hnd=False, use_pdl=False, cluster_combine=False, has_final_lse=True):  # noqa: A001
        raise AssertionError("host-only selection must not compile")

    km = SimpleNamespace(compile=compile, cluster_combine_supported=lambda dq, dv: True, CLUSTER_COMBINE_BY_KNOB=True)
    api = SimpleNamespace(
        packed_thd_split=False,
        thd=False,
        thd_decode_leg=False,
        _k_mod=km,
        lse_desc=object() if stats else None,
        paged=True,
        _paged_pool_stride=lambda desc: (1, 2, 3, 4),
        k_desc=None,
        use_pdl=False,
        stats_log2=False,
        gate_desc=None,
        _device_cc=(10, 0),
        s_q_max=2,
        head_dim_qk=256,
        head_dim_v=256,
        split_kv=8,
        cluster_reduce=requested,
        _not_implemented_error_if=lambda cond, msg: (_ for _ in ()).throw(NotImplementedError(msg)) if cond else None,
    )
    result = SdpaFwdDslSm100._explicit_compile_kwargs(api)
    assert api._cluster_combine is requested
    if requested:
        assert result["cluster_combine"] is True and result["has_final_lse"] is stats
    else:
        assert "cluster_combine" not in result and "has_final_lse" not in result
    assert result["has_lse"] is True
    km.cluster_combine_supported = lambda dq, dv: False
    if requested:
        with pytest.raises(NotImplementedError, match="no in-launch reduction"):
            SdpaFwdDslSm100._explicit_compile_kwargs(api)


@pytest.mark.parametrize(
    "change,reason",
    [
        (None, None),
        ("old_extension", "rebuild the extension"),
        ("no_cluster", "cannot run a thread-block cluster of 8 blocks"),
        ("prefill", "does not lower onto it"),
        ("rubin", "does not lower onto it"),
        ("pdl", "programmatic-dependent-launch"),
        ("sink", "does not carry a sink"),
        ("split16", "2 to 8 KV chunks"),
    ],
)
def test_standalone_adapter_declines_like_the_row_and_adds_the_installation_checks(monkeypatch, change, reason):
    """The adapter's twin of the row decline, on a stand-in plan: the same contract predicate, then the two answers only an
    installation can give -- an extension that binds the reduced launch over pages and lengths, and a device that can launch
    a cluster of that size (a device query at support, never on the planning path)."""
    from cudnn.frost import occupancy
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    binder = SimpleNamespace() if change == "old_extension" else SimpleNamespace(supports_fused_split_paged=True)
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaDenseBinder", binder)
    table = [148, 74, 45, 33, 26, 22, 15, 0 if change == "no_cluster" else 15] + [0] * 8
    monkeypatch.setattr(occupancy, "occupancy_map", lambda device=None: table)
    half = SimpleNamespace(dtype="bfloat16", device=SimpleNamespace(index=0))
    api = SimpleNamespace(
        _fp8=False,
        _mixed_fp8_kv=False,
        _device_cc=(10, 7) if change == "rubin" else (10, 0),
        _decode_q_tile=lambda: 0 if change == "prefill" else 16,
        flavor=(256, 256),
        split_kv=16 if change == "split16" else 8,
        thd=False,
        has_sink=change == "sink",
        stats_log2=False,
        gate_desc=None,
        o_desc=half,
        q_desc=half,
        use_pdl=change == "pdl",
    )
    got = SdpaFwdDslSm100._cluster_reduce_decline(api)
    assert got is None if reason is None else reason in got


def _paged_fused_fixture(stats):
    """A split-8 paged D256 spec with per-batch KV and Q lengths whose host order is this tile's own, bound fused."""
    from test_sdpa_native_split_binding import _split_fixture

    spec, f, frames, combined = _split_fixture(d=256, splits=8, paged=True, hnd=True, lengths=True, lse=stats)
    path = Path(prep.__file__).parent / "kernels/sm100/d256_f16_swap_ab/kernel.py"
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_host")
    values = dict(zip(spec.order, spec.template))
    spec.order = [arg.arg for arg in host.args.args if "Constexpr" not in ast.unparse(arg.annotation)]
    assert spec.order[-5:] == ["final_o_ptr", "final_lse_ptr", "final_o_strides", "final_lse_strides", "stream"]
    spec.index = {name: i for i, name in enumerate(spec.order)}
    spec.template = [values.get(name) for name in spec.order]
    spec.fused_split = True
    spec.native = cudnn._pybind_module._SdpaDenseBinder(spec)
    return spec, f, frames, combined


@pytest.mark.parametrize("stats", [False, True])
def test_native_and_python_frames_agree_over_pages_and_lengths(stats):
    """The reduced launch binds the final O / Stats beside the partial slabs over a paged cache with per-batch lengths, the
    native binder and the Python frame agree slot for slot, and exactly one launch is issued."""
    import sdpa_binding_reference as reference
    from test_sdpa_native_dense_binding import _pack

    if not getattr(cudnn._pybind_module._SdpaDenseBinder, "supports_fused_split_paged", False):
        pytest.skip("the installed extension predates the reduced launch over pages and lengths")
    spec, f, frames, combined = _paged_fused_fixture(stats)
    assert spec.paged and spec.seq_kv_present and spec.seq_q_present
    for offset, stream in [(0, 17), (0x200000, 29)]:
        current = {role: fact._replace(ptr=fact.ptr + offset) for role, fact in f.items()}
        workspace = 2**40 + offset
        expected, combine = reference.bind_dense_split(spec, current, workspace, stream, stream)
        for name, value in {
            "final_o_ptr": current["o"].ptr,
            "final_lse_ptr": current["lse"].ptr if stats else None,
            "final_o_strides": combine[6][:3],
            "final_lse_strides": combine[7],
        }.items():
            expected[spec.index[name]] = value
        actual = spec.native.bind_split(_pack(current), prep._NATIVE_DENSE_INDICES, workspace, stream)
        python_frame, python_combine = prep.bind_dense_split(spec, current, workspace, stream, stream)
        assert list(actual[0]) == python_frame == expected
        assert actual[1] == python_combine == combine
        spec.native.execute(_pack(current), prep._NATIVE_DENSE_INDICES, stream, workspace=workspace)
        assert frames[-1] == tuple(expected)
    assert len(frames) == 2 and combined == []
    # the reduction reads the partial rows as 16-byte groups: both frames refuse an unaligned workspace, as for every split
    with pytest.raises(ValueError, match="16-byte aligned"):
        prep.bind_dense_split(spec, f, 2**40 + 8, 17, 17)
    with pytest.raises(ValueError, match="16-byte aligned"):
        spec.native.bind_split(_pack(f), prep._NATIVE_DENSE_INDICES, 2**40 + 8, 17)
    with pytest.raises(ValueError, match="overlaps"):
        spec.native.execute(_pack(f), prep._NATIVE_DENSE_INDICES, 17, workspace=f["o"].ptr)


def test_reduced_frame_still_refuses_a_sink():
    from test_sdpa_native_split_binding import _split_fixture

    spec, _, _, _ = _split_fixture(d=256, splits=8, lengths=False, lse=True)
    spec.fused_split, spec.has_sink = True, True
    with pytest.raises(ValueError, match="without a sink"):
        cudnn._pybind_module._SdpaDenseBinder(spec)
