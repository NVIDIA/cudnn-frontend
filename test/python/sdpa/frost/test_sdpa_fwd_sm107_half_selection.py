# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Rubin fixed-body records preserve native/shared and bias-budget boundaries."""

from dataclasses import replace

import pytest
import cudnn
from cudnn.sdpa.fwd import engines, heuristics
from cudnn.sdpa.fwd.kernels.sm107 import mid_half_contract as mid, small_half_contract as small
from cudnn.sdpa.fwd.kernels.sm107.shared_half_contract import SINGLE_Q, LOOKAHEAD as SHARED_LOOKAHEAD
from sdpa.frost.test_sdpa_fwd_sm107_small_implementation_contract import facts, biased

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "d,changes,knobs,owner",
    [
        (32, {}, dict(cga=1), small.MULTI_Q),
        (64, dict(s_q=128), dict(cga=1), SINGLE_Q),
        (64, dict(s_q=129), dict(cga=1), small.MULTI_Q),
        (64, dict(s_q=128), dict(cga=2), small.MULTI_Q),
        (64, dict(thd=True), dict(cga=1, split_kv=3), SINGLE_Q),
        (64, dict(thd=True), dict(cga=1), small.MULTI_Q),
        (128, {}, dict(cga=1), SINGLE_Q),
        (128, dict(thd=True), dict(cga=1), small.MULTI_Q),
        (128, dict(thd=True), dict(cga=1, warp_spec_cfg=1), mid.LOOKAHEAD),
        (128, dict(thd=True), dict(cga=1, split_kv=3), SINGLE_Q),
        (192, {}, dict(cga=2), mid.D192),
        (192, dict(thd=True), dict(cga=1), SINGLE_Q),
        (192, dict(has_paged_kv=True, padded=True), dict(cga=2), small.MULTI_Q),
        (256, dict(s_q=1), dict(cga=1), mid.SWAP_AB),
        (256, dict(s_q=129), dict(cga=2), mid.LOOKAHEAD),
        (256, dict(s_q=129, has_paged_kv=True), dict(cga=2), SHARED_LOOKAHEAD),
        (256, dict(thd=True), dict(cga=2, split_kv=3), SHARED_LOOKAHEAD),
        (256, dict(thd=True), dict(cga=2), mid.LOOKAHEAD),
    ],
)
def test_main_body_identity_at_layout_and_topology_boundaries(d, changes, knobs, owner):
    f = facts(d, **changes)
    k = engines.SdpaFwdKnobs(tile_n=128, **knobs)
    geometry = heuristics.select_sm107_half_geometry(f, k)
    assert geometry.implementation == owner
    assert geometry.lowering_row_name == "sdpa_fwd_prefill_sm107"
    canonical = geometry.canonical_knobs(k)
    assert engines.SdpaFwdKnobs.from_public(canonical.to_public()) == canonical


def test_logical_d32_uses_the_existing_d64_thd_envelope():
    f = facts(32, thd=True)
    k = engines.SdpaFwdKnobs(cga=1, tile_n=128)
    policy = engines.sm100_legacy_policy_spec(small.LEGACY_ROW)
    assert engines._selected_d_shape(policy.capabilities, f, k) == (64, 64)
    g = heuristics.select_sm107_half_geometry(f, k)
    assert g.implementation == small.MULTI_Q and g.d_flavor == 64


@pytest.mark.parametrize(
    "owner,physical,dims,changes,knobs",
    [
        (small.MULTI_Q, 32, (24, 16), {}, dict(cga=1)),
        (small.MULTI_Q, 128, (80, 112), {}, dict(cga=2)),
        (small.BATCH_PAIR, 64, (48, 40), dict(b=4, has_bias=True, bias_dim=(1, 32, 128, 256), s_kv=256), dict(cga=1)),
        (mid.D192, 192, (176, 112), {}, dict(cga=2)),
        (mid.LOOKAHEAD, 256, (224, 208), {}, dict(cga=2)),
        (mid.SWAP_AB, 256, (224, 208), dict(s_q=3), dict(cga=1, tile_m=16)),
        (SINGLE_Q, 128, (112, 96), {}, dict(cga=1)),
    ],
)
def test_envelope_record_preserves_physical_host_geometry(monkeypatch, owner, physical, dims, changes, knobs):
    from cudnn.sdpa.fwd.config_common import TemplateParams

    spec = next(s for s in engines.ENGINE_SPECS if s.name == owner)
    native = facts(physical, **changes)
    envelope = replace(native, d_qk=dims[0], d_v=dims[1])
    k = engines.SdpaFwdKnobs(tile_n=128, **knobs)
    assert engines.mismatch(spec.capabilities, envelope, k) is None
    legacy = next(s for s in engines.ENGINE_SPECS if s.name == small.LEGACY_ROW)
    assert "owned by" in engines.mismatch(legacy.capabilities, envelope, k)
    seen = []
    monkeypatch.setitem(spec.lower.keywords, "lower_impl", lambda *a, **kw: seen.append((a, kw["selected_kernel"])))

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed envelope record consulted main-body selection")

    monkeypatch.setattr(heuristics, "select_sm107_half_geometry", forbidden)
    for f in (native, envelope):
        spec.lower(spec, f, engines.SdpaFwdKnobs.from_public(k.to_public()))
    (native_args, expected), (args, got) = seen
    assert args[1] == envelope  # Logical load/store bounds are not replaced by physical widths.
    assert args[2] == native_args[2]
    assert got == expected
    assert got.configure_params(TemplateParams()) == expected.configure_params(TemplateParams())


@pytest.mark.parametrize("dims", [(24, 16), (48, 40), (112, 96), (176, 112), (224, 208)])
def test_half_envelope_does_not_expand_mixed_kv(dims):
    f = replace(facts(128), d_qk=dims[0], d_v=dims[1], uniform_dtype=False, kv_dtype=cudnn.data_type.FP8_E4M3)
    owners = [s for s in engines.ENGINE_SPECS if s.capabilities.implementation in small.IMPLEMENTATIONS | mid.IMPLEMENTATIONS | {SINGLE_Q, SHARED_LOOKAHEAD}]
    assert all(engines.mismatch(s.capabilities, f) for s in owners)


@pytest.mark.parametrize(
    "owner,dims,knobs",
    [
        (small.MULTI_Q, (24, 16), dict(cga=1)),
        (mid.LOOKAHEAD, (112, 96), dict(cga=1, warp_spec_cfg=1)),
        (SHARED_LOOKAHEAD, (224, 208), dict(cga=2, split_kv=3)),
    ],
)
def test_envelope_transfer_keeps_existing_thd_declines(owner, dims, knobs):
    f = replace(facts(128, thd=True), d_qk=dims[0], d_v=dims[1])
    k = engines.SdpaFwdKnobs(tile_n=128, **knobs)
    policy = engines.sm100_legacy_policy_spec(small.LEGACY_ROW)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == owner)
    assert engines.mismatch(policy.capabilities, f, k) is not None
    assert engines.mismatch(spec.capabilities, f, k) is not None


def test_omitted_d128_tile_width_keeps_legacy_default():
    g = heuristics.select_sm107_half_geometry(facts(128), engines.SdpaFwdKnobs())
    assert (g.implementation, g.tile_n, g.cga) == (small.MULTI_Q, 128, 2)


@pytest.mark.parametrize("sq,owner", [(128, small.BATCH_PAIR), (256, small.MULTI_Q), (384, small.BATCH_PAIR)])
def test_batch_pair_crossover_is_only_in_central_selection(sq, owner, monkeypatch):
    f = biased(32, s_q=sq, has_kv_mask=False)
    k = engines.SdpaFwdKnobs(tile_n=128, cga=1)
    assert heuristics.select_sm107_half_geometry(f, k).implementation == owner

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed replay consulted the batch-pair crossover")

    monkeypatch.setattr(heuristics, "select_sm107_half_geometry", forbidden)
    from cudnn.sdpa.fwd import config_sm107

    monkeypatch.setattr(config_sm107, "sm107_n_chunk_pick", forbidden)
    for name in (small.MULTI_Q, small.BATCH_PAIR):
        g = small.explicit_geometry(name, f, k)
        assert g.implementation == name
        assert (g.n_chunk > 0) == (name == small.BATCH_PAIR)


@pytest.mark.parametrize("sq", [128, 256])
def test_bias_support_uses_the_selected_work_items_budget(sq, monkeypatch):
    # D64 CTA1 ordinary has only a 128-column resident bias and no streamed
    # ring. Pair's 384-column block fits this request, regardless of SQ.
    f = biased(64, s_q=sq, s_kv=256, has_kv_mask=False)
    k = engines.SdpaFwdKnobs(cga=1, tile_n=128, sched_policy=0)
    caps = engines.sm100_legacy_policy_spec(small.LEGACY_ROW).capabilities
    from cudnn.sdpa.fwd import config_sm107

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed support consulted the batch-pair crossover")

    monkeypatch.setattr(config_sm107, "sm107_pair_item_active", forbidden)
    ordinary = engines._bias_arm_decline(replace(caps, implementation=small.MULTI_Q), f, k)
    pair = engines._bias_arm_decline(replace(caps, implementation=small.BATCH_PAIR), f, k)
    assert ordinary and "no bias form fits" in ordinary
    assert pair is None


def test_mixed_kv_and_split_gate_preserve_native_owners():
    f = facts(256, s_q=1, has_paged_kv=True, uniform_dtype=False, kv_dtype=cudnn.data_type.FP8_E4M3)
    assert heuristics.select_sm107_half_geometry(f, engines.SdpaFwdKnobs(cga=2)).implementation == mid.LOOKAHEAD
    gated = facts(256, s_q=1, has_epilogue_gate=True)
    assert heuristics.select_sm107_half_geometry(gated, engines.SdpaFwdKnobs(split_kv=1)).implementation == mid.LOOKAHEAD
    assert heuristics.select_sm107_half_geometry(gated, engines.SdpaFwdKnobs(split_kv=3)).implementation == mid.SWAP_AB


@pytest.mark.parametrize(
    "name,d,changes,knobs",
    [
        (small.MULTI_Q, 64, {}, dict(cga=2)),
        (small.MULTI_Q, 32, dict(thd=True), dict(cga=1)),
        (small.BATCH_PAIR, 32, dict(b=4, has_bias=True, bias_dim=(1, 32, 128, 512), s_kv=512), dict(cga=1)),
        (mid.D192, 192, {}, dict(cga=2)),
        (mid.LOOKAHEAD, 256, {}, dict(cga=2)),
        (mid.LOOKAHEAD, 128, dict(thd=True), dict(cga=1, warp_spec_cfg=1)),
        (mid.SWAP_AB, 256, dict(s_q=1), dict(cga=1, tile_m=16)),
        (SINGLE_Q, 128, {}, dict(cga=1)),
        (SHARED_LOOKAHEAD, 256, dict(thd=True), dict(cga=2, split_kv=3)),
    ],
)
def test_registered_host_replays_public_knobs_without_selecting_another_body(monkeypatch, name, d, changes, knobs):
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd import config_sm107

    offered = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()
    assert name in offered
    spec = next(s for s in engines.ENGINE_SPECS if s.name == name)
    k = engines.SdpaFwdKnobs(tile_n=128, **knobs)
    restored = engines.SdpaFwdKnobs.from_public(k.to_public())
    calls = []
    monkeypatch.setitem(spec.lower.keywords, "lower_impl", lambda *args, **kw: calls.append((args, kw)))

    def forbidden(*args, **kwargs):
        raise AssertionError("a registered fixed host consulted the main-body chooser")

    monkeypatch.setattr(heuristics, "select_sm107_half_geometry", forbidden)
    monkeypatch.setattr(config_sm107, "sm107_n_chunk_pick", forbidden)
    spec.lower(spec, facts(d, **changes), restored)
    assert len(calls) == 1
    args, kw = calls[0]
    selected = kw["selected_kernel"]
    assert selected.implementation == name
    assert engines.SdpaFwdKnobs.from_public(args[2].to_public()) == args[2]
    if name == small.MULTI_Q and d == 32:
        assert selected.d_flavor == 64  # THD uses the existing larger envelope.
    if name in (SINGLE_Q, SHARED_LOOKAHEAD):
        assert selected.kernel_file.startswith("sm100/")
    else:
        assert selected.kernel_file.startswith("sm107/")


@pytest.mark.parametrize("d", [32, 64])
def test_dense_n64_split_keeps_the_native_ordinary_body(d):
    f = facts(d, s_q=257)
    k = engines.SdpaFwdKnobs(cga=1, tile_n=64, split_kv=3)
    g = heuristics.select_sm107_half_geometry(f, k)
    assert g.implementation == small.MULTI_Q
    spec = next(s for s in engines.ENGINE_SPECS if s.name == g.implementation)
    assert engines.mismatch(spec.capabilities, f, g.canonical_knobs(k)) is None
    single_q = next(s for s in engines.ENGINE_SPECS if s.name == SINGLE_Q)
    assert engines.mismatch(single_q.capabilities, f, k) is not None


@pytest.mark.parametrize("d,cga", [(32, 1), (128, 1), (192, 2), (256, 2)])
def test_placement_does_not_import_selection_from_candidate_generator(monkeypatch, d, cga):
    import builtins
    from cudnn.sdpa.fwd import placement

    f = facts(d, s_q=129)
    geometry = heuristics.select_sm107_half_geometry(f, engines.SdpaFwdKnobs(cga=cga, tile_n=128))
    knobs = geometry.canonical_knobs(engines.SdpaFwdKnobs(cga=cga, tile_n=128))
    spec = next(s for s in engines.ENGINE_SPECS if s.name == geometry.implementation)
    assert engines.mismatch(spec.capabilities, f, knobs) is None
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("heuristics") and "select_sm107_half_geometry" in fromlist:
            raise AssertionError("placement imports its body selector from the candidate generator")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert placement.place(spec, f, knobs) in (placement.LEAD, placement.TRAIL)


def test_selection_module_imports_without_registry_or_candidate_generator(monkeypatch):
    import builtins
    import importlib.util
    from pathlib import Path

    source = Path(engines.__file__).with_name("sm107_half_selection.py")
    spec = importlib.util.spec_from_file_location("cudnn.sdpa.fwd._selection_import_probe", source)
    module = importlib.util.module_from_spec(spec)
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name in ("engines", "heuristics", "placement")
            or name.startswith("cudnn.sdpa.fwd.")
            and name.rsplit(".", 1)[-1] in ("engines", "heuristics", "placement")
        ):
            raise AssertionError("body policy imports the registry or candidate generator")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec.loader.exec_module(module)


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("stats_layout", ["token", "head"])
@pytest.mark.parametrize("compact", [False, True])
def test_d64_origins_helper_keeps_architecture_scheduler_domain(monkeypatch, cc, dtype_name, stats_layout, compact):
    import torch
    from cudnn.frost import buffers
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
    import sdpa_d64_thd_split as helper
    import test_sdpa_fwd_thd_origins as origins

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: cc)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    f = SdpaGraphFacts(
        b=2,
        h_q=8,
        h_kv=2,
        s_q=129,
        s_kv=2048,
        d_qk=64,
        d_v=64,
        dtype=cudnn.data_type.BFLOAT16 if dtype_name == "bfloat16" else cudnn.data_type.HALF,
        thd=True,
        device_cc=cc,
        device_sm_count=148,
    )
    seen = []

    def record(graph, engine_name, knobs=None):
        k = engines.SdpaFwdKnobs.from_public(knobs)
        seen.append(k.sched_policy)
        spec = next(s for s in engines.ENGINE_SPECS if s.name == engine_name)
        assert engines.mismatch(spec.capabilities, f, k) is None

    def probe(d, dtype, layout, packed_stats, packed, splits, *, engine_name, cga, compact):
        assert (d, dtype, layout, packed, splits) == (64, dtype_name, stats_layout, True, 9)
        spec = next(s for s in engines.ENGINE_SPECS if s.name == engine_name)
        k = engines.SdpaFwdKnobs(cga=cga, tile_n=128, pack_gqa=packed, split_kv=splits)
        cudnn.pygraph.create_execution_plan(None, spec.name, k.to_public())

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plan", record)
    monkeypatch.setattr(origins, "test_independent_thd_origins_rebind_and_capture", probe)
    helper.test_origins_and_replay(dtype_name, stats_layout, True, 9, 2, compact, monkeypatch)
    assert seen == [1 if cc == (10, 7) else 2]
