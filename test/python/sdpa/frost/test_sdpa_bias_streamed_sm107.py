# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The STREAMED bias ring of the Rubin (SM107) f16/bf16 small head-dim tiles -- ``TemplateParams.bias_mode = 2`` on
``sm107/prefill_d128_f16.py``: the fallback above the RESIDENT row-block's budget.

Above the resident budget (S_kv 768 / 384 for a batch-broadcast bias at the 32- / 64-wide tile -- the (n, n+1) pair item; 384 / 128 /
256 for a per-batch bias -- the one-batch-entry item) the bias row-block is no longer held for the item's whole KV range: the TMA-LDG
warp streams ONE ``BIAS_BLOCK_ROWS x TILE_N`` bias tile per kv tile through a ``STAGES_BIAS``-deep SMEM ring (2 or 3 stages; the deepest
the carveout holds beside the operands), both softmax sub-tiles read the stage (the pair item: rows n / n + 1 read the SAME 128-row tile)
and release it per kv tile.  Same TMA boxes, same 128-byte swizzle, same lane loads and the same pre-scale packed FFMA as the resident
block, so O / Stats are BITWISE the resident form's wherever both fit and bias = 0 stays bitwise the plain kernel; S_kv is bounded by
nothing but the payload / stride rules.  The pick is ONE host function (``config_sm107.sm107_bias_mode_pick``) the engine row, the
heuristics' width pick and the adapter read: the resident block where it holds S_kv, else the ring where a 2-stage ring fits, else a
typed decline (the one-batch-entry item at the 64-wide tile's cga1: two 64 KiB stages exceed its spare -- the heuristics take cga2).

Three tiers: (1) CPU contract -- the derived ring depth per (tile, cga, item form), the mode pick and its two test seams, the engine
row's accept / decline table (the cells the ring newly serves and the one width that still declines), the heuristics' width pick, the
config factory's couplings, the adapter twin with its ``bias_mode=`` / ``bias_stages=`` requests, the appended fields, the kernel's one
const_expr axis; (2) host rendering pins -- every streamed rendering keeps the register split, no cluster drain, no new spills over the
resident rendering of the same arm, the packed bias add and the bit-word mask where a mask is on; the RESIDENT renderings stay
byte-identical to their record (the ring's seams fold out of them); (3) GPU cells on a Rubin device -- the fp64 oracle at S_kv above
the resident budget (both item forms, both tiles, the cga2 ring, the tail chunk, a zero-length row, the kv_mask, fp16), the bitwise
pins (resident vs streamed where both fit, 2 vs 3 stages, bias = 0 vs the plain kernel), the served / declined real graphs.
"""

import dataclasses
import math
import re
from pathlib import Path

import pytest
import torch

from frost_test_utils import assert_no_new_spills, requires_dsl, requires_rubin, run_sass_probe, sass_probe_source
import test_sdpa_bias_sm107 as _s13

pytestmark = [pytest.mark.L0, requires_dsl]

_ROW = "sdpa_fwd_prefill_sm107"
_KERNEL = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "sdpa" / "fwd" / "kernels" / "sm107" / "prefill_d128_f16.py"
_RENDERINGS = Path(__file__).resolve().parent / "renderings"
# (head-dim tile, cga, pair item?) -> the deepest streamed ring the carveout holds beside the operands (0 = no ring).
_RING = {(32, 1, False): 3, (32, 1, True): 3, (64, 1, False): 0, (64, 1, True): 3, (64, 2, False): 2}
# The resident budgets the ring sits above (test_sdpa_bias_sm107.py / test_sdpa_bias_nchunk_sm107.py pin them).
_RESIDENT = {(32, 1, False): 384, (32, 1, True): 768, (64, 1, False): 128, (64, 1, True): 384, (64, 2, False): 256}


@pytest.fixture(autouse=True)
def _sm107_target_for_metadata(monkeypatch):
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


@pytest.fixture
def _pins(monkeypatch):
    """Pin the bias FORM, the ring depth and the chunk length a knob-less plan gets (the config module's three private, process-local
    test seams; the public knobs need the frontend-band enum in both pybind extensions).  ``None`` = the pick."""
    from cudnn.sdpa.fwd import config_sm107 as c

    def _set(mode=None, stages=None, n_chunk=None):
        monkeypatch.setattr(c, "_BIAS_MODE_OVERRIDE_FOR_SWEEPS", mode)
        monkeypatch.setattr(c, "_BIAS_STAGES_OVERRIDE_FOR_SWEEPS", stages)
        monkeypatch.setattr(c, "_N_CHUNK_OVERRIDE_FOR_SWEEPS", n_chunk)

    return _set


@pytest.fixture
def _s13_rubin(monkeypatch):
    """The standalone adapter reads the device capability at construction: model a Rubin device on any host (the bias module's fixture)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))


# --------------------------------------------------------------------------- tier 1: CPU contract


def test_ring_depth_and_mode_pick_are_derived():
    """One arithmetic for the row, the heuristics, the adapter and the factory: the ring depth is what the carveout leaves beside the
    operands in whole tiles of the item form, capped at 3 and 0 below 2; the pick is resident inside the resident budget, streamed above
    it where a ring fits, 0 where neither form fits."""
    from cudnn.sdpa.fwd import config_sm107 as c

    assert c.BIAS_MODES == {0, 1, 2} and c.BIAS_RING_STAGES == (2, 3)
    for (d, cga, pair), stages in _RING.items():
        assert c.sm107_bias_ring_stages(d, cga, pair=pair) == stages, (d, cga, pair)
        tiles = c.sm107_bias_resident_tiles(d, cga, pair=pair)
        assert stages == (min(3, tiles) if tiles >= 2 else 0), (d, cga, pair, tiles)
        s_res = _RESIDENT[(d, cga, pair)]
        # the knob-less pick: resident inside the budget -- except the one-batch-entry item at the 32-wide tile, which streams wherever
        # its ring fits (the measured pick, +6..+13 % over its resident block at 2-3 kv tiles); a RESIDENT request still gets the block
        streams_inside = (not pair) and cga == 1 and d in c.SM107_ONE_BATCH_ENTRY_STREAMING_D_FLAVORS and stages and s_res > 128
        assert c.sm107_bias_mode_pick(d, cga, s_res, pair=pair) == (c.BIAS_MODE_STREAMED if streams_inside else c.BIAS_MODE_RESIDENT), (d, cga, pair)
        assert c.sm107_bias_mode_pick(d, cga, s_res, pair=pair, request=1) == c.BIAS_MODE_RESIDENT
        above = c.sm107_bias_mode_pick(d, cga, s_res + 128, pair=pair)
        assert above == (c.BIAS_MODE_STREAMED if stages else c.BIAS_MODE_NONE), (d, cga, pair, above)
        assert c.sm107_bias_mode_pick(d, cga, 4096, pair=pair) == above
        # a REQUEST is honoured where that form fits and 0 elsewhere -- never degraded to the other form
        assert c.sm107_bias_mode_pick(d, cga, s_res, pair=pair, request=2) == (c.BIAS_MODE_STREAMED if stages else c.BIAS_MODE_NONE)
        assert c.sm107_bias_mode_pick(d, cga, s_res + 128, pair=pair, request=1) == c.BIAS_MODE_NONE
        assert c.sm107_bias_mode_pick(d, cga, s_res, pair=pair, request=3) == c.BIAS_MODE_NONE
        # the depth pick: the request in the domain and fitting, else the derived deepest for the pair item and 2 for the one-batch-entry
        # item (the measured pick: its 3-stage ring is +1.5..+3.6 % slower); a request that does not fit is 0
        assert c.sm107_bias_stages_pick(d, cga, pair=pair) == (stages if pair else min(stages, c.SM107_ONE_BATCH_ENTRY_RING_STAGES))
        for req in (2, 3):
            assert c.sm107_bias_stages_pick(d, cga, pair=pair, request=req) == (req if req <= stages else 0)
        assert c.sm107_bias_stages_pick(d, cga, pair=pair, request=4) == 0
    # TILE_N 64 halves every stage: the one-batch-entry item at the 64-wide tile's cga1 gets a ring there
    assert c.sm107_bias_ring_stages(64, 1, tile_n=64) == 3
    # the SMEM extent by form
    assert c.bias_smem_tiles(c._CfgSm107()) == 0


def test_the_n_chunk_pick_keeps_the_one_batch_entry_item_where_both_forms_stream():
    """The measured guard: at S_q % 256 == 0 the one-batch-entry item wherever a form of it fits at the width (+8.6 % over the pair's RESIDENT
    block at d32 S 768, +6..7 % over the pair's ring at 1024 / 2048 / 4096); the pair stays where S_q is not a multiple of 256 (384, 896) and at
    the 64-wide tile's cga1 above its 128-column block (no one-batch-entry ring there)."""
    from cudnn.sdpa.fwd import config_sm107 as c

    for s in (512, 768, 1024, 2048, 4096):
        assert c.sm107_n_chunk_pick(1, s, 4, s, 212, 1, s_kv=s, d_flavor=32) == 0, s
        assert not c.sm107_pair_item_active(1, s, 4, s, 212, 1, s_kv=s, d_flavor=32), s
        assert c.sm107_pair_item_active(1, s, 4, s, 212, 1, n_chunk_request=8, s_kv=s, d_flavor=32), "a request still selects the pair"
    assert c.sm107_n_chunk_pick(1, 896, 4, 896, 212, 1, s_kv=896, d_flavor=32) >= 2, "S_q 896 is not a multiple of 256: the pair streams"
    assert c.sm107_n_chunk_pick(1, 384, 4, 384, 212, 1, s_kv=384, d_flavor=32) >= 2, "S_q 384 wastes a quarter of a super-tile: the pair (+30 % measured)"
    assert c.sm107_n_chunk_pick(1, 1024, 4, 1024, 212, 1, s_kv=1024, d_flavor=64) >= 2, "d64 cga1: no one-batch-entry ring, the pair streams"
    assert c.sm107_n_chunk_pick(1, 256, 4, 256, 212, 1, s_kv=128, d_flavor=64) == 0, "d64 S_kv 128: the one-batch-entry block fits (the resident-block guard)"


def test_the_auto_width_takes_the_streamed_fallback_at_the_narrowest_width_whose_ring_fits():
    from cudnn.sdpa.fwd import config_sm107 as c

    assert c.sm107_bias_auto_cga(64, 512, {1, 2}, pair=False) == 2, "a per-batch bias at d64 above 256: cga2's 2-stage ring (cga1 has none)"
    assert c.sm107_bias_auto_cga(64, 512, {1, 2}, pair=True) == 1, "the pair item's 3-stage ring at cga1"
    assert c.sm107_bias_auto_cga(64, 128, {1, 2}) == 1 and c.sm107_bias_auto_cga(64, 256, {1, 2}) == 2, "the resident pass first"
    assert c.sm107_bias_auto_cga(32, 4096, {1}, pair=True) == 1 and c.sm107_bias_auto_cga(32, 1024, {1}) == 1
    assert c.sm107_bias_auto_cga(128, 256, {1, 2}) is None, "the d128 tile has no spare for either form"


def test_seams_pin_the_form_and_the_depth(_pins):
    from cudnn.sdpa.fwd import config_sm107 as c

    _pins(mode=2)
    assert c.sm107_bias_mode_pick(32, 1, 384, pair=True) == 2, "the streamed arm where the resident block would fit (the A/B)"
    assert c.sm107_bias_mode_pick(64, 1, 128, pair=False) == 0, "...but never where no ring fits"
    _pins(mode=1)
    assert c.sm107_bias_mode_pick(32, 1, 896, pair=True) == 0, "the resident arm declines above its budget"
    assert c.sm107_bias_mode_pick(32, 1, 896, pair=True, request=2) == 2, "an explicit request outranks the seam"
    _pins(stages=2)
    assert c.sm107_bias_stages_pick(32, 1, pair=True) == 2 and c.sm107_bias_stages_pick(32, 1, pair=True, request=3) == 3
    _pins()
    assert c.sm107_bias_mode_pick(32, 1, 896, pair=True) == 2 and c.sm107_bias_stages_pick(32, 1, pair=True) == 3


# (name, facts kwargs, knobs kwargs, accept?) -- the cells the ring newly serves beside the one width that still declines.
_CELLS = [
    ("pair d32 S896: the streamed ring", dict(s_q=896, s_kv=896, **_s13._bias_kw(s_q=896, s_kv=896)), None, True),
    ("pair d32 S1024 causal", dict(s_q=1024, s_kv=1024, causal=True, **_s13._bias_kw(s_q=1024, s_kv=1024)), None, True),
    ("pair d32 S4096", dict(s_q=4096, s_kv=4096, **_s13._bias_kw(s_q=4096, s_kv=4096)), None, True),
    ("pair d32 S2048 with a kv_mask", dict(s_q=2048, s_kv=2048, **_s13._bias_kw(s_q=2048, s_kv=2048), **_s13._mask_kw(s_kv=2048)), None, True),
    ("per-batch bias d32 S1024: the one-batch-entry item's ring", dict(s_q=1024, s_kv=1024, **_s13._bias_kw(b=2, s_q=1024, s_kv=1024)), None, True),
    ("one flattened row d32 S1024", dict(b=1, s_q=1024, s_kv=1024, **_s13._bias_kw(s_q=1024, s_kv=1024)), None, True),
    ("pair d64 S1024", dict(d_qk=64, d_v=64, s_q=1024, s_kv=1024, **_s13._bias_kw(s_q=1024, s_kv=1024)), None, True),
    ("per-batch bias d64 S512 knob-less: cga2's ring", dict(d_qk=64, d_v=64, s_q=512, s_kv=512, **_s13._bias_kw(b=2, s_q=512, s_kv=512)), None, True),
    ("per-batch bias d64 S512 at a requested cga2", dict(d_qk=64, d_v=64, s_q=512, s_kv=512, **_s13._bias_kw(b=2, s_q=512, s_kv=512)), dict(cga=2), True),
    (
        "per-batch bias d64 S512 at a requested cga1: no ring fits (the resident bias block holds 128)",
        dict(d_qk=64, d_v=64, s_q=512, s_kv=512, **_s13._bias_kw(b=2, s_q=512, s_kv=512)),
        dict(cga=1),
        False,
    ),
    ("bias d128 S1024: no form on the d128 tile", dict(d_qk=128, d_v=128, s_q=1024, s_kv=1024, **_s13._bias_kw(s_q=1024, s_kv=1024)), None, False),
    ("bias S1028 compact: the TMA stride rule is form-independent", dict(s_q=1028, s_kv=1028, **_s13._bias_kw(s_q=1028, s_kv=1028)), None, False),
]


@pytest.mark.parametrize("name, fkw, kkw, accept", _CELLS, ids=[c[0] for c in _CELLS])
def test_row_serves_the_streamed_ring_and_declines_where_no_form_fits(name, fkw, kkw, accept):
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _s13._caps()
    f = _s13._facts(**_s13._resolve_dtypes(fkw))
    knobs = SdpaFwdKnobs(**kkw) if kkw else None
    why = engines.mismatch(caps, f, knobs)
    assert (why is None) == accept, (name, why)
    if not accept and "no ring" in name:
        assert "no bias form fits" in why and "resident bias block" in why and "streamed ring" in why, (name, why)


def test_heuristics_take_the_streamed_width():
    """The width pick agrees with the row's eligibility (rule 8b'): a per-batch bias at the 64-wide tile above 256 lands on cga2
    (the only width with a ring for the 256-row item), the pair item stays cga1 at any S_kv, a per-batch bias at the 32-wide tile
    above 384 stays on its cga1-only tile."""
    from cudnn.sdpa.fwd import heuristics

    offered = {s.name: i for i, s in enumerate(heuristics.ENGINE_SPECS)}
    for fkw, want in (
        (dict(d_qk=64, d_v=64, s_q=512, s_kv=512, **_s13._bias_kw(b=2, s_q=512, s_kv=512)), {2}),
        (dict(d_qk=64, d_v=64, s_q=512, s_kv=512, **_s13._bias_kw(s_q=512, s_kv=512)), {1}),
        (dict(s_q=1024, s_kv=1024, **_s13._bias_kw(b=2, s_q=1024, s_kv=1024)), {1}),
        (dict(s_q=2048, s_kv=2048, **_s13._bias_kw(s_q=2048, s_kv=2048)), {1}),
    ):
        plans = heuristics.recommend("A", _s13._facts(device_sm_count=204, **_s13._resolve_dtypes(fkw)), offered)
        assert {p.knobs.cga for p in plans if offered[_ROW] == p.engine_id} == want, (fkw, [p.knobs for p in plans])


def test_config_factory_derives_the_ring_and_raises_on_each_coupling():
    """Every ring field follows the tile, the cga and the item form; each coupling moved alone raises with the failure it would produce
    (frost-kernels s5 "bind the couplings"; the SMEM term names the ring)."""
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    for d, cga, kw, stages, rows in (
        (32, 1, dict(n_chunk=4), 3, 128),
        (32, 1, dict(n_chunk=4, bias_stages=2), 2, 128),
        (32, 1, dict(), 2, 256),  # the one-batch-entry item's ring: 2 stages by the measured pick (sm107_bias_stages_pick)
        (32, 1, dict(bias_stages=3), 3, 256),  # ...the deepest ring still a served request
        (64, 1, dict(n_chunk=2), 3, 128),
        (64, 2, dict(), 2, 256),
    ):
        cfg, _ = {32: c.make_cfg_d32, 64: c.make_cfg_d64}[d](TemplateParams(dtype_qkv=2, cta_mma=cga, bias_mode=2, **kw))
        assert (cfg.BIAS_MODE, cfg.STAGES_BIAS, cfg.BIAS_RESIDENT_TILES, cfg.BIAS_BLOCK_ROWS, cfg.BIAS_BOXES_PER_TILE, cfg.BIAS_EMPTY_ARRIVERS) == (
            2,
            stages,
            0,
            rows,
            2,
            8,
        ), (d, cga, kw, cfg)
        assert c.bias_smem_tiles(cfg) == stages and c.bias_resident_block_bytes(cfg) == stages * rows * 128 * 2
        assert c.operand_region_end_bytes(cfg) <= c.TCGEN05_V0_ADDR_LIMIT
        assert c.operand_region_end_bytes(cfg) + c.bias_resident_block_bytes(cfg) + c._SMEM_FIXED_OVERHEAD <= c.SMEM_USABLE_BYTES
        for field, value, word in (
            ("STAGES_BIAS", 4, "stages"),
            ("STAGES_BIAS", 1, "stages"),
            ("BIAS_RESIDENT_TILES", 1, "no resident block"),
            ("BIAS_EMPTY_ARRIVERS", 4, "softmax warp count"),
            ("BIAS_MODE", 3, "BIAS_MODE must be"),
        ):
            with pytest.raises(ValueError, match=re.escape(word)):
                c._validate_bias(dataclasses.replace(cfg, **{field: value}), "probe")
    # the resident form keeps its fields and carries no ring
    res, _ = c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=1, n_chunk=4))
    assert (res.BIAS_MODE, res.STAGES_BIAS, res.BIAS_RESIDENT_TILES) == (1, 0, 6)
    with pytest.raises(ValueError, match="no ring"):
        c._validate_bias(dataclasses.replace(res, STAGES_BIAS=2), "probe")
    with pytest.raises(ValueError, match="bias_stages=2 is the STREAMED ring's depth"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=1, bias_stages=2))
    with pytest.raises(ValueError, match="no \\[2, 3\\]-stage streamed bias ring"):
        c.make_cfg_d64(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=2))
    with pytest.raises(ValueError, match="requested bias_stages=4"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=2, n_chunk=4, bias_stages=4))
    with pytest.raises(ValueError, match="bias_stages=3 is the STREAMED ring's depth"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_stages=3))
    plain, _ = c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1))
    assert (plain.BIAS_MODE, plain.STAGES_BIAS, plain.BIAS_RESIDENT_TILES) == (0, 0, 0)


# (adapter kwargs, accept?, expected bias_mode, expected bias_stages, expected cta_mma): the adapter is the row's twin and honours a
# requested form / depth where it fits, declining typed elsewhere (never degraded).
_ADAPTER_CELLS = [
    (dict(s_q=896, s_kv=896, bias="h"), True, 2, 3, 1),
    (dict(s_q=896, s_kv=896, bias="h", bias_stages=2), True, 2, 2, 1),
    (dict(s_q=1024, s_kv=1024, bias="b"), True, 2, 2, 1),  # the one-batch-entry ring runs 2 stages (the measured pick)
    (dict(s_q=1024, s_kv=1024, bias="h", b=1), True, 2, 2, 1),
    (dict(s_q=1024, s_kv=1024, bias="b", bias_stages=3), True, 2, 3, 1),  # the deepest ring as a request
    (dict(s_q=384, s_kv=384, bias="h"), True, 1, 0, 1),
    (dict(s_q=384, s_kv=384, bias="h", bias_mode=2), True, 2, 3, 1),
    (dict(s_q=384, s_kv=384, bias="h", bias_mode=2, bias_stages=2), True, 2, 2, 1),
    (dict(s_q=896, s_kv=896, bias="h", bias_mode=1), False, None, None, None),
    (dict(s_q=384, s_kv=384, bias="h", bias_stages=2), False, None, None, None),
    (dict(s_q=896, s_kv=896, bias="h", bias_stages=4), False, None, None, None),
    (dict(s_q=896, s_kv=896, bias="h", bias_mode=3), False, None, None, None),
    (dict(d=64, s_q=1024, s_kv=1024, bias="h"), True, 2, 3, 1),
    (dict(d=64, s_q=512, s_kv=512, bias="b"), True, 2, 2, 2),
    (dict(d=64, s_q=512, s_kv=512, bias="b", cga=2), True, 2, 2, 2),
    (dict(d=64, s_q=512, s_kv=512, bias="b", cga=1), False, None, None, None),
    (dict(d=64, s_q=512, s_kv=512, bias="b", cga=2, bias_stages=3), False, None, None, None),
]


@pytest.mark.parametrize(
    "kw, accept, mode, stages, cga",
    _ADAPTER_CELLS,
    ids=[
        "d32_S896_pair_ring",
        "d32_S896_pair_ring_2_stages",
        "d32_S1024_per_batch_ring",
        "d32_S1024_one_row_ring",
        "d32_S1024_per_batch_ring_3_stages_requested",
        "d32_S384_resident_pick",
        "d32_S384_ring_requested",
        "d32_S384_ring_2_stages_requested",
        "d32_S896_resident_requested_declines",
        "d32_S384_stages_on_a_resident_plan_declines",
        "d32_S896_stages_4_declines",
        "d32_S896_mode_3_declines",
        "d64_S1024_pair_ring",
        "d64_S512_per_batch_knobless_cga2",
        "d64_S512_per_batch_cga2",
        "d64_S512_per_batch_cga1_declines",
        "d64_S512_per_batch_cga2_stages_3_declines",
    ],
)
def test_adapter_check_support_is_the_rows_twin(_s13_rubin, kw, accept, mode, stages, cga):
    kw = dict(kw)
    d = kw.pop("d", 32)
    form = kw.pop("bias")
    b = kw.pop("b", 2)
    h = 4
    api = _s13._api(d, b=b, h=h, bias=((b if form == "b" else 1), h, kw["s_q"], kw["s_kv"]), **kw)
    if not accept:
        with pytest.raises((NotImplementedError, ValueError)):
            api.check_support()
        return
    api.check_support()
    params = api.template_params()
    assert (params.bias_mode, params.bias_stages, params.cta_mma) == (mode, stages, cga), params
    if mode == 2 and form == "h" and b >= 2 and cga == 1:
        assert params.n_chunk >= 2, "the pair item streams too"
    if form == "b" or b == 1:
        assert params.n_chunk == 0, "the one-batch-entry item streams its 256-row tiles"


def test_template_params_cfg_and_bars_are_appended():
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.kernels._common_blackwell import Bars

    fields = [f.name for f in dataclasses.fields(TemplateParams)]
    assert (
        fields[-7:] == ["n_chunk", "bias_stages", "padded_tail_form", "dead_subtile_quiet", "bias_max_form", "softmax_wgs", "s_slots"]
        and TemplateParams().bias_stages == 0
    ), "append-only; 0 renders every existing record byte-identically"
    assert c._CfgSm107.STAGES_BIAS == 0
    assert Bars._fields[-2:] == ("mb_bias_ring_full", "mb_bias_ring_empty") and Bars._field_defaults["mb_bias_ring_full"] is None


def test_kernel_carries_the_streamed_ring_as_one_const_expr_axis():
    """The ring is one const_expr axis of the bias arm: ONE fill helper called from EVERY kv-tile path (the prologue and the mainloop of
    both item forms -- the lockstep with the softmax side's per-tile wait / release the P14 statement rests on), ONE wait before the first
    read and ONE elected release after the last word in the softmax body, the state advanced once per kv tile by each caller, the two
    ring rows initialised stage by stage, the resident rows folded on their own guard."""
    src = _KERNEL.read_text()
    assert "_BIAS_RESIDENT = CFG.BIAS_MODE == BIAS_MODE_RESIDENT" in src and "_BIAS_STREAMED = CFG.BIAS_MODE == BIAS_MODE_STREAMED" in src
    assert "def _bias_ring_fill(" in src
    assert src.count("bias_ring_state = _bias_ring_fill(") == 4, "one fill per kv tile on the four kv-tile paths"
    assert src.count("bars.mb_bias_ring_full[bias_ring_state.idx].wait(bias_ring_state.phase, spin=SPIN_RING_WAITS)") == 1
    assert src.count("bars.mb_bias_ring_empty[bias_ring_state.idx].wait(bias_ring_state.phase, spin=SPIN_RING_WAITS)") == 1
    assert src.count("bars.mb_bias_ring_empty[bias_ring_state.idx].arrive()") == 1
    assert src.count("bias_ring_state = advance(bias_ring_state, CFG.STAGES_BIAS)") == 2 and src.count("return advance(bias_ring_state, CFG.STAGES_BIAS)") == 1
    assert "for bs in cutlass.range_constexpr(CFG.STAGES_BIAS):" in src
    assert (
        "make_classic_bars(CFG, s_stages=CFG.TILES_Q * CFG.S_SLOTS, bias_resident=_BIAS_RESIDENT, bias_ring_stages=CFG.STAGES_BIAS if _BIAS_STREAMED else 0)"
        in src
    )
    assert "n_tiles=bias_smem_tiles(CFG)" in src
    assert "if cutlass.const_expr(_BIAS_ARM):\n                if item_live:" not in src, "the item-level rows fold on the RESIDENT guard"
    assert "if cutlass.const_expr(_BIAS_RESIDENT):" in src


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_PROBE_BODY = """
d = params_kw.pop("__d")
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
print("BIAS_MODE", mod.CFG.BIAS_MODE)
print("KV_MASK_FORM", mod.CFG.KV_MASK_FORM)
print("N_CHUNK", mod.CFG.N_CHUNK)
print("BIAS_RESIDENT_TILES", mod.CFG.BIAS_RESIDENT_TILES)
print("STAGES_BIAS", mod.CFG.STAGES_BIAS)
print("BIAS_BLOCK_ROWS", mod.CFG.BIAS_BLOCK_ROWS)
mod.compile(d_qk=d, d_v=d, has_lse=True, lse_kind="dense", has_bias=bool(mod.CFG.BIAS_MODE), has_kv_mask=bool(mod.CFG.KV_MASK_FORM))
"""
_COUNTS = {
    "USETMAXREG": ("USETMAXREG",),
    "STL": (" STL",),
    "LDL": (" LDL",),
    "FMNMX3": ("FMNMX3",),
    "R2P": (" R2P ",),
    "CGAERRBAR": ("CGAERRBAR",),
    "FFMA2": (" FFMA2",),
    "UTMALDG": ("UTMALDG",),
}
# The RESIDENT pair renderings' committed record (renderings/md5_fwd_d128_file_bias_resident_sm_107a.txt) -> their TemplateParams.
_RESIDENT_RECORD_ARMS = {
    "d32_pair4": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 1, "n_chunk": 4},
    "d32_pair4_mask_padded_causal": {
        "__d": 32,
        "cta_mma": 1,
        "dtype_qkv": 2,
        "bias_mode": 1,
        "kv_mask_form": 1,
        "seq_kv_lens_present": True,
        "window_right": 0,
        "n_chunk": 4,
    },
}


def _record():
    f = _RENDERINGS / "md5_fwd_d128_file_bias_resident_sm_107a.txt"
    assert f.is_file(), f
    dsl, want = None, {}
    for ln in f.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(r"fwd_d128_file sm_107a (\S+) rc=0 cubin_md5=([0-9a-f]{32})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


def _probe(tmp_path, params, tag):
    return run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch="sm_107a", params=params, tag=tag)


def test_resident_record_is_committed_and_complete():
    dsl, want = _record()
    assert dsl and dsl.startswith("nvidia-cutlass-dsl "), dsl
    assert set(want) == set(_RESIDENT_RECORD_ARMS), (sorted(want), sorted(_RESIDENT_RECORD_ARMS))


@pytest.mark.parametrize("arm", list(_RESIDENT_RECORD_ARMS))
def test_resident_renderings_are_byte_identical_to_their_record(tmp_path, arm):
    """The ring's seams fold OUT of the resident form: a resident rendering traces byte-identically to the tree BEFORE the ring landed
    (the pass-through ring state of the pair item is canonicalised away; host trace-compile)."""
    from cudnn.frost.buffers import cutedsl_state

    dsl, want = _record()
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: the cubin is a function of the DSL build, re-render the record")
    probe = _probe(tmp_path, _RESIDENT_RECORD_ARMS[arm], f"resident_{arm}")
    assert probe.expect["BIAS_MODE"] == 1 and probe.expect["STAGES_BIAS"] == 0
    assert probe.cubin_md5 == want[arm], f"{arm}: cubin md5 {probe.cubin_md5} != the resident record's {want[arm]} -- the resident rendering changed"


@pytest.mark.parametrize(
    "d, cta, extra, base_extra, stages",
    [
        (32, 1, {"n_chunk": 4}, {"n_chunk": 4}, 3),
        (32, 1, {"n_chunk": 4, "bias_stages": 2}, {"n_chunk": 4}, 2),
        (
            32,
            1,
            {"n_chunk": 4, "kv_mask_form": 1, "seq_kv_lens_present": True, "window_right": 0},
            {"n_chunk": 4, "kv_mask_form": 1, "seq_kv_lens_present": True, "window_right": 0},
            3,
        ),
        (32, 1, {}, {}, 2),
        (32, 1, {"bias_stages": 3}, {}, 3),
        (64, 1, {"n_chunk": 4}, {"n_chunk": 4}, 3),
        (64, 2, {}, {}, 2),
    ],
    ids=[
        "d32_pair_3",
        "d32_pair_2",
        "d32_pair_mask_padded_causal",
        "d32_one_batch_entry_2",
        "d32_one_batch_entry_3_requested",
        "d64_pair_3",
        "d64_one_batch_entry_cga2_2",
    ],
)
def test_streamed_renderings_keep_the_split_the_mask_form_and_the_spill_count(tmp_path, d, cta, extra, base_extra, stages):
    """Each streamed rendering traced for sm_107a keeps the per-role register split (USETMAXREG > 0), no cluster-scope drain (CGAERRBAR
    0), spills no more than the RESIDENT rendering of the same arm, carries the packed bias add (FFMA2 >= 128) and the software row max
    (FMNMX3 > 0), the bit-word form (R2P > 0) where a mask is on, more TMA issue sites than the resident form (the per-kv-tile fill on the
    prologue AND the mainloop) and the derived ring depth with no resident tiles."""
    base = _probe(tmp_path, {"__d": d, "cta_mma": cta, "dtype_qkv": 2, "bias_mode": 1, **base_extra}, f"resident_{d}_{cta}_{'_'.join(sorted(base_extra))}")
    probe = _probe(tmp_path, {"__d": d, "cta_mma": cta, "dtype_qkv": 2, "bias_mode": 2, **extra}, f"streamed_{d}_{cta}_{'_'.join(sorted(extra))}")
    assert probe.expect["BIAS_MODE"] == 2 and probe.expect["STAGES_BIAS"] == stages and probe.expect["BIAS_RESIDENT_TILES"] == 0
    assert probe.expect["BIAS_BLOCK_ROWS"] == (128 if extra.get("n_chunk") else 256)
    assert probe.cubin_md5 != base.cubin_md5
    assert probe.stats["USETMAXREG"] > 0 and probe.stats["CGAERRBAR"] == 0, probe.stats
    assert_no_new_spills(probe.stats, base.stats, tag=f"d{d} cga{cta} streamed {sorted(extra)} vs the resident rendering: ")
    assert probe.stats["FMNMX3"] > 0 and probe.stats["FFMA2"] >= 128, probe.stats
    assert probe.stats["UTMALDG"] > base.stats["UTMALDG"], (probe.stats, base.stats)
    if extra.get("kv_mask_form") or extra.get("seq_kv_lens_present"):
        assert probe.stats["R2P"] > 0, probe.stats


# --------------------------------------------------------------------------- tier 3: GPU cells (a Rubin device)

# The streamed ring's degenerate-input matrix: S_kv ABOVE the resident budget of the item form (the pair item above 768 / 384 at the
# 32- / 64-wide tile; the one-batch-entry item -- a per-batch bias ("b") or a single row -- above 384 / 128 / 256), the tail chunk, a
# zero-length row, the padded-Q trim, the kv_mask, the cga2 ring, fp16; plus the cells where BOTH forms fit (the bitwise pins).
# ``mode`` / ``stages`` / ``n_chunk``: the seams' pins (None = the pick).
_CASES = {
    "pair_896_causal": dict(b=4, h=2, h_kv=2, s_q=896, s_kv=896, causal=True),
    "pair_1024_dense": dict(b=4, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=4),
    "pair_1024_2_stages": dict(b=4, h=2, h_kv=2, s_q=1024, s_kv=1024, stages=2, n_chunk=4),
    "one_batch_entry_pick_1024_dense": dict(b=4, h=2, h_kv=2, s_q=1024, s_kv=1024),
    "pair_1152_tail_chunk_zero_row": dict(b=5, h=2, h_kv=2, s_q=1152, s_kv=1152, n_chunk=4, causal=True, kv_lens=[1152, 0, 700, 1152, 1]),
    "pair_1024_padded_q_trim": dict(b=4, h=4, h_kv=2, s_q=1024, s_kv=1024, n_chunk=2, causal=True, q_lens=[1024, 300, 0, 1024], kv_lens=[1024, 900, 1024, 0]),
    "pair_1000_tail_tile": dict(b=4, h=2, h_kv=2, s_q=1000, s_kv=1000, n_chunk=2),
    "pair_2048_causal": dict(b=2, h=2, h_kv=2, s_q=2048, s_kv=2048, causal=True, n_chunk=2),
    # the ring at 32 / 48 kv tiles: the pair item, the knob-less pick (the one-batch-entry ring at S_q % 256 == 0) and a causal 6144
    "pair_4096_dense": dict(b=2, h=2, h_kv=2, s_q=4096, s_kv=4096, n_chunk=2),
    "one_batch_entry_pick_4096_dense": dict(b=2, h=2, h_kv=2, s_q=4096, s_kv=4096),
    "pair_6144_causal": dict(b=2, h=1, h_kv=1, s_q=6144, s_kv=6144, causal=True, n_chunk=2),
    "pair_1024_swa_padded": dict(b=4, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=2, causal=True, window_left=256, kv_lens=[1024, 500, 1024, 777]),
    "pair_1024_head_broadcast_bias": dict(b=4, h=4, h_kv=4, s_q=1024, s_kv=1024, bias="1", n_chunk=4),
    "pair_1024_bias_x10": dict(b=4, h=2, h_kv=2, s_q=1024, s_kv=1024, bias_scale=10.0, n_chunk=4),
    "pair_1024_bh_gt_1_sink_padded": dict(b=4, h=4, h_kv=2, s_q=1024, s_kv=1024, n_chunk=2, causal=True, kv_lens=[1024, 0, 800, 1024], sink=True),
    "one_batch_entry_per_batch_512": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512, bias="b"),
    "one_batch_entry_per_batch_1024_padded": dict(b=3, h=2, h_kv=2, s_q=1024, s_kv=1024, bias="b", kv_lens=[1024, 0, 500]),
    "one_batch_entry_single_row_1024_causal": dict(b=1, h=2, h_kv=2, s_q=1024, s_kv=1024, causal=True),
    "d64_pair_512": dict(b=4, h=2, h_kv=2, s_q=512, s_kv=512, d=64, n_chunk=2),
    "d64_pair_1024_causal": dict(b=4, h=2, h_kv=2, s_q=1024, s_kv=1024, d=64, causal=True),  # d64 cga1: the pick stays the pair (no one-batch-entry ring)
    "d64_one_batch_entry_per_batch_512_cga2_ring": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512, d=64, bias="b"),
    "d64_one_batch_entry_per_batch_384_cga2_ring_padded": dict(b=3, h=2, h_kv=2, s_q=384, s_kv=384, d=64, bias="b", kv_lens=[384, 0, 200]),
    # both forms fit: the streamed arm pinned by the seam (the bitwise pins below compare against the resident pick)
    "both_fit_pair_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, n_chunk=4, mode=2),
    "both_fit_pair_768_causal": dict(b=4, h=2, h_kv=2, s_q=768, s_kv=768, n_chunk=4, causal=True, mode=2),
    "both_fit_one_batch_entry_per_batch_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384, bias="b", mode=2),
    "both_fit_d64_pair_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, d=64, n_chunk=2, mode=2),
    "both_fit_d64_one_batch_entry_per_batch_256_cga2": dict(b=2, h=2, h_kv=2, s_q=256, s_kv=256, d=64, bias="b", mode=2),
}


def _case_inputs(case, dtype, *, kv_mask_p=None):
    return _s13._case_inputs(case, dtype, kv_mask_p=kv_mask_p, table=_CASES)


def _run(q, k, v, kw, c, _pins, *, mode=None, stages=None, n_chunk=None):
    _pins(
        mode=c.get("mode") if mode is None else mode,
        stages=c.get("stages") if stages is None else stages,
        n_chunk=c.get("n_chunk") if n_chunk is None else n_chunk,
    )
    return _s13._run(q, k, v, **kw)


def _ring_plan(out):
    """The plan's TemplateParams are not on the graph's public surface; the ring's presence is checked on the OUTPUT (the oracle) and the
    width on the knobs -- the form that ran is pinned by the seam the cell set."""
    return out["knobs"]


@requires_rubin
@pytest.mark.parametrize("case", [c for c in _CASES if not c.startswith("both_fit")])
def test_streamed_ring_matches_the_fp64_oracle(case, _pins):
    """The streamed ring over its degenerate-input matrix (bf16): O and Stats against the fp64 oracle, keyless rows exactly 0 / -inf,
    two launches bitwise; every cell sits ABOVE the resident budget of its item form, so the ring is the only form that serves it."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    out = _run(q, k, v, kw, c, _pins)
    assert out["knobs"].cga == (2 if "cga2" in case else 1), (case, out["knobs"])
    _s13._check(out, ref)


@requires_rubin
@pytest.mark.parametrize("case", ["pair_1024_dense", "pair_1152_tail_chunk_zero_row", "one_batch_entry_per_batch_1024_padded", "d64_pair_1024_causal"])
def test_streamed_ring_matches_the_fp64_oracle_fp16(case, _pins):
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.float16)
    out = _run(q, k, v, kw, c, _pins)
    _s13._check(out, ref)


@requires_rubin
@pytest.mark.parametrize(
    "case", ["pair_1024_dense", "pair_1152_tail_chunk_zero_row", "one_batch_entry_per_batch_1024_padded", "d64_one_batch_entry_per_batch_512_cga2_ring"]
)
def test_streamed_ring_composes_with_the_kv_mask(case, _pins):
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16, kv_mask_p=0.3)
    out = _run(q, k, v, kw, c, _pins)
    _s13._check(out, ref)


@requires_rubin
@pytest.mark.parametrize("case", [c for c in _CASES if c.startswith("both_fit")])
def test_streamed_is_bitwise_the_resident_block_where_both_fit(case, _pins):
    """Where both forms fit, O / Stats are BITWISE identical: every row's kv loop runs the same tiles in the same order and the bias
    enters by the same FFMA from the same 128-byte swizzled row slice -- only the SMEM extent and the two mbarrier rings differ."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    streamed = _run(q, k, v, kw, c, _pins, mode=2)
    _s13._check(streamed, ref)
    resident = _run(q, k, v, kw, c, _pins, mode=1)
    assert torch.equal(streamed["o"], resident["o"]) and torch.equal(
        streamed["stats"], resident["stats"]
    ), f"{case}: the streamed ring differs from the resident block"
    assert streamed["knobs"].cga == resident["knobs"].cga


@requires_rubin
@pytest.mark.parametrize("case", ["pair_1024_dense", "pair_2048_causal", "one_batch_entry_per_batch_512", "d64_pair_1024_causal"])
def test_two_and_three_stages_are_bitwise(case, _pins):
    """The ring depth is a pure latency knob (no accumulator parity is indexed from it): 2 and 3 stages give bitwise-identical O / Stats."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    deep = _run(q, k, v, kw, c, _pins, stages=3)
    _s13._check(deep, ref)
    shallow = _run(q, k, v, kw, c, _pins, stages=2)
    assert torch.equal(deep["o"], shallow["o"]) and torch.equal(deep["stats"], shallow["stats"]), f"{case}: 2 stages differ from 3"


@requires_rubin
@pytest.mark.parametrize("dtype", _s13._DTYPES, ids=_s13._DTYPE_IDS)
@pytest.mark.parametrize("case", ["pair_1024_dense", "pair_896_causal", "one_batch_entry_per_batch_512", "d64_pair_1024_causal"])
def test_identity_bias_zero_is_bitwise_the_plain_kernel(case, dtype, _pins):
    """bias = 0 through the streamed ring is BITWISE the plain kernel on O and Stats (the -inf masked body with fma(0, 1/scale, s) == s)."""
    c = _CASES[case]
    q, k, v, kw, _ = _case_inputs(case, dtype)
    _pins()
    plain = _s13._run(q, k, v, **{**kw, "bias": None})
    zero = torch.zeros(1, 1, c["s_q"], c["s_kv"], device="cuda", dtype=dtype)
    out = _run(q, k, v, {**kw, "bias": zero}, c, _pins)
    assert torch.equal(out["o"], plain["o"]) and torch.equal(out["stats"], plain["stats"])


@requires_rubin
def test_served_and_declined_on_real_graphs(_pins):
    """A batch-broadcast bias at S_kv 896 / 2048 (over the pair's resident budget), a per-batch bias at S_kv 1024 and a single-row graph at
    S_kv 1024 (over the one-batch-entry budget) are SERVED by the FROST row (the ring); a per-batch bias at the 64-wide tile at S_kv 512 is
    served at cga2 (the only width with a ring for its 256-row tiles); a compact bias at S_kv % 8 != 0 stays a typed decline (the TMA
    stride rule is form-independent)."""
    import cudnn

    _pins()
    torch.manual_seed(5)

    def plans(b, s, bias_b, d=32):
        q, k, v = (_s13._bhsd(b, 2, s, d, torch.bfloat16) for _ in range(3))
        bias = torch.randn(bias_b, 2, s, s, device="cuda").to(torch.bfloat16)
        g, *_ = _s13._build(q, k, v, scale=1.0 / math.sqrt(d), bias=bias)
        try:
            g.create_execution_plans([cudnn.heur_mode.A])
            names, ours = _s13._frost_plans(g)
            return names, ours, [g.plans[i].knobs.cga for i in ours]
        except cudnn.cudnnGraphNotSupportedError:
            return [], [], []

    for b, s, bias_b, d in ((2, 896, 1, 32), (2, 2048, 1, 32), (2, 1024, 2, 32), (1, 1024, 1, 32), (2, 1024, 1, 64)):
        names, ours, cgas = plans(b, s, bias_b, d)
        assert ours and 1 in cgas, ("served by the streamed ring at cga1", b, s, bias_b, d, names)
    names, ours, cgas = plans(2, 512, 2, 64)
    assert ours and set(cgas) == {2}, ("a per-batch bias at the 64-wide tile above 256: cga2's ring only", names, cgas)
    names, ours, _ = plans(2, 1028, 1, 32)
    assert not ours, ("a compact bias at S_kv % 8 != 0 stays declined", names)
