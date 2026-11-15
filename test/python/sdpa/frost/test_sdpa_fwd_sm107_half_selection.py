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
