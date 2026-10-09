# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The (n, n+1) PAIR work item of the resident-bias arm on the Rubin (SM107) f16/bf16 small head-dim tiles --
``TemplateParams.n_chunk >= 1`` on ``sm107/prefill_d128_f16.py``.

A work item is (a 128-row q tile, a head, a CHUNK of up to ``n_chunk`` consecutive entries of the flattened batch axis); the two
softmax sub-tiles are rows nb / nb + 1 of the SAME q tile, so the bias row-block shrinks to the q tile's 128 rows (twice the kv
columns fit: S_kv <= 768 at the 32-wide tile, <= 384 at the 64-wide one, from 384 / 128) and is loaded ONCE per item for every
row of the chunk -- the per-item fixed cost and the bias feed divided by ``n_chunk``.  The form needs a batch-BROADCAST bias
(``[1, 1|H, S_q, S_kv]``: two rows share one block), at least two rows and cga1; a per-batch bias, a single row or a cga2 plan keeps
the one-batch-entry item (``n_chunk = 0``, every one-batch-entry cell and cubin record unchanged).  Nothing numerically changes: every row's
kv loop runs the same tiles in the same order, so O / Stats are BITWISE identical across ``n_chunk`` values, against the
one-batch-entry item and (bias = 0) against the plain kernel.

Three tiers: (1) CPU contract -- the derived pair budget and the form predicate, the chunk-length rule, the engine row's
accept / decline table (the widened cells and the cells that keep the one-batch-entry budget), the config factory's couplings,
the adapter twin; (2) host rendering pins -- every pair rendering keeps the register split, no cluster drain, no spills above
the one-batch-entry bias rendering, the packed bias add and the bit-word mask where a mask is on; (3) GPU cells on a Rubin
device -- the fp64 oracle under n_chunk in {1, 2, 8, N} and the degenerate rows (one chunk, the tail chunk with a dead sub-tile,
S_kv not a multiple of the tile, B x H > 1 with n_kv = 1, a zero-length row inside a chunk, the padded-Q trim, S_kv above the
one-batch-entry budget), the bitwise pins, the typed declines on real graphs.
"""

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
# (head-dim tile, cga) -> the resident block's S_kv budget on the PAIR item (128-row block), beside the one-batch-entry one.
_PAIR_BUDGET = {(32, 1): 768, (64, 1): 384}
_LEGACY_BUDGET = {(32, 1): 384, (64, 1): 128, (64, 2): 256}


@pytest.fixture(autouse=True)
def _sm107_target_for_metadata(monkeypatch):
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


@pytest.fixture
def _n_chunk(monkeypatch):
    """Pin the chunk length the heuristics would pick (the process-local sweep seam; the public knob needs the frontend-band
    enum in both pybind extensions).  ``None`` = the rule."""
    from cudnn.sdpa.fwd import config_sm107 as c

    def _set(value):
        monkeypatch.setattr(c, "_N_CHUNK_OVERRIDE_FOR_SWEEPS", value)

    return _set


# --------------------------------------------------------------------------- tier 1: CPU contract


def test_pair_block_is_derived_and_doubles_the_budget():
    """One arithmetic for the row, the adapter and the factory: the 128-row block holds twice the kv columns of the 256-row one at the
    same operands; the one-batch-entry numbers are untouched."""
    from cudnn.sdpa.fwd import config_sm107 as c

    assert c.sm107_bias_block_rows(pair=True) == 128 and c.sm107_bias_block_rows() == 256
    for (d, cga), s_max in _PAIR_BUDGET.items():
        tiles = c.sm107_bias_resident_tiles(d, cga, pair=True)
        assert c.sm107_bias_resident_s_kv_max(d, cga, pair=True) == s_max == tiles * 128, (d, cga, tiles)
        stages = c._stages_kv_d128(2, cga, mxfp8=False, tile_k=d)
        operands = c.sdpa_smem_bytes(128, 128, d, d, 2, stages, cga, 2, 2)
        assert tiles == (c.SMEM_USABLE_BYTES - c._SMEM_FIXED_OVERHEAD - operands) // (128 * 128 * 2)
        assert c.sm107_bias_resident_tiles(d, cga, pair=True) >= 2 * c.sm107_bias_resident_tiles(d, cga)
    for (d, cga), s_max in _LEGACY_BUDGET.items():
        assert c.sm107_bias_resident_s_kv_max(d, cga) == s_max


def test_pair_form_predicate_and_the_chunk_rule():
    from cudnn.sdpa.fwd import config_sm107 as c

    assert c.sm107_bias_pair_form(1, 2, 1) and c.sm107_bias_pair_form(1, 384, 1)
    assert not c.sm107_bias_pair_form(2, 2, 1), "a per-batch bias keeps the one-batch-entry item"
    assert not c.sm107_bias_pair_form(1, 1, 1), "a single flattened row would run a dead sub-tile on every item"
    assert not c.sm107_bias_pair_form(1, 4, 2), "the pair runs cga1 only"
    # the rule: the largest power of two <= H * ceil(S_q / 128) * B_flat / (3 * SMs), clamped to [2, B_flat] and even
    assert c.sm107_n_chunk_rule(384, 4, 384, 212) == 4 and c.sm107_n_chunk_rule(512, 4, 512, 212) == 8
    assert c.sm107_n_chunk_rule(256, 4, 256, 212) == 2 and c.sm107_n_chunk_rule(128, 4, 128, 212) == 2
    assert c.sm107_n_chunk_rule(384, 4, 384, 152) == 8 and c.sm107_n_chunk_rule(1024, 4, 1024, 212) == 32
    assert c.sm107_n_chunk_rule(1, 4, 384, 212) == 0 and c.sm107_n_chunk_rule(3, 4, 384, 212) == 2
    assert c.sm107_n_chunk_rule(384, 4, 384, 0) == c.sm107_n_chunk_rule(384, 4, 384, 148)
    assert c.sm107_n_chunk_pick(1, 384, 4, 384, 212, 1) == 4 and c.sm107_n_chunk_pick(2, 384, 4, 384, 212, 1) == 0
    assert c.sm107_n_chunk_pick(1, 384, 4, 384, 212, 2) == 0 and c.sm107_n_chunk_pick(1, 1, 4, 384, 212, 1) == 0


def test_n_chunk_override_seam_is_read_by_the_pick(_n_chunk):
    from cudnn.sdpa.fwd import config_sm107 as c

    _n_chunk(8)
    assert c.sm107_n_chunk_pick(1, 384, 4, 384, 212, 1) == 8
    assert c.sm107_n_chunk_pick(2, 384, 4, 384, 212, 1) == 0, "the seam never turns a per-batch bias into the pair item"
    _n_chunk(None)
    assert c.sm107_n_chunk_pick(1, 384, 4, 384, 212, 1) == 4
    _n_chunk(0)
    assert c.sm107_n_chunk_pick(1, 384, 4, 384, 212, 1) == 0, "0 pins the one-batch-entry item (its A/B arm)"
    assert not c.sm107_pair_item_active(1, 384, 4, 384, 212, 1), "...and the row / adapter evaluate THAT item's budget"
    assert c.sm107_pair_item_active(1, 384, 4, 384, 212, 1, n_chunk_request=4)
    _n_chunk(None)
    assert c.sm107_pair_item_active(1, 384, 4, 384, 212, 1) and not c.sm107_pair_item_active(2, 384, 4, 384, 212, 1)
    # the MEASURED guard of the default pick: a full 256-row super-tile inside the one-batch-entry budget keeps that item (its two
    # sub-tiles share K/V); a request or the sweep seam still selects the pair there
    assert c.sm107_n_chunk_pick(1, 256, 4, 256, 212, 1, s_kv=256, d_flavor=32) == 0
    assert (
        c.sm107_n_chunk_pick(1, 256, 4, 256, 212, 1, s_kv=512, d_flavor=32) == 0
    ), "over the one-batch-entry BLOCK: its streamed ring still wins at S_q % 256 == 0"
    assert c.sm107_n_chunk_pick(1, 256, 4, 256, 212, 1, s_kv=512, d_flavor=64) == 2, "d64 cga1: no one-batch-entry ring above 128 -> the pair"
    assert c.sm107_n_chunk_pick(1, 384, 4, 384, 212, 1, s_kv=384, d_flavor=32) == 4, "a partly empty super-tile: the pair"
    assert c.sm107_n_chunk_pick(1, 256, 4, 256, 212, 1, s_kv=256, d_flavor=64) == 2, "d64 at cga1 holds 128 on the one-batch-entry item: the pair"
    assert c.sm107_pair_item_active(1, 256, 4, 256, 212, 1, n_chunk_request=4, s_kv=256, d_flavor=32)
    _n_chunk(4)
    assert c.sm107_n_chunk_pick(1, 256, 4, 256, 212, 1, s_kv=256, d_flavor=32) == 4
    _n_chunk(None)


# (name, facts kwargs, knobs kwargs, accept?) -- the widened cells and the cells that keep the one-batch-entry budget.
_CELLS = [
    ("pair d32 S512 (6-tile block)", dict(s_q=512, s_kv=512, **_s13._bias_kw(s_q=512, s_kv=512)), None, True),
    ("pair d32 S768 = the 6-tile budget", dict(s_q=768, s_kv=768, **_s13._bias_kw(s_q=768, s_kv=768)), None, True),
    ("pair d32 S896 over the 6-tile budget: the streamed ring", dict(s_q=896, s_kv=896, **_s13._bias_kw(s_q=896, s_kv=896)), None, True),
    ("per-batch bias d32 S512 keeps the 3-tile budget: the streamed ring", dict(s_q=512, s_kv=512, **_s13._bias_kw(b=2, s_q=512, s_kv=512)), None, True),
    ("one flattened row d32 S512 keeps the 3-tile budget: the streamed ring", dict(b=1, s_q=512, s_kv=512, **_s13._bias_kw(s_q=512, s_kv=512)), None, True),
    ("pair d64 S384 cga1 (3-tile block)", dict(d_qk=64, d_v=64, s_q=384, s_kv=384, **_s13._bias_kw(s_q=384, s_kv=384)), None, True),
    (
        "pair d64 S384 at a requested cga2 keeps the 2-tile budget: the cga2 streamed ring",
        dict(d_qk=64, d_v=64, s_q=384, s_kv=384, **_s13._bias_kw(s_q=384, s_kv=384)),
        dict(cga=2),
        True,
    ),
    ("pair d64 S512 over both budgets: the pair's streamed ring", dict(d_qk=64, d_v=64, s_q=512, s_kv=512, **_s13._bias_kw(s_q=512, s_kv=512)), None, True),
    ("pair d32 S512 with a kv_mask", dict(s_q=512, s_kv=512, **_s13._bias_kw(s_q=512, s_kv=512), **_s13._mask_kw(s_kv=512)), None, True),
    ("pair d32 S512 head-broadcast bias", dict(s_q=512, s_kv=512, **_s13._bias_kw(h=1, s_q=512, s_kv=512)), None, True),
]


@pytest.mark.parametrize("name, fkw, kkw, accept", _CELLS, ids=[c[0] for c in _CELLS])
def test_row_serves_the_pair_budget_and_keeps_the_one_batch_entry_cells(name, fkw, kkw, accept):
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _s13._caps()
    f = _s13._facts(**_s13._resolve_dtypes(fkw))
    knobs = SdpaFwdKnobs(**kkw) if kkw else None
    why = engines.mismatch(caps, f, knobs)
    assert (why is None) == accept, (name, why)
    if not accept and "budget" in name:
        assert "resident bias block" in why and ("pair item" in why or "one-batch-entry" in why), (name, why)


def test_heuristics_pick_cga1_for_a_pair_eligible_d64_bias_at_s_kv_256():
    """The pair's 3-tile cga1 block holds S_kv 256 at the 64-wide tile, so a batch-broadcast bias no longer needs the cga2 width the
    one-batch-entry item took before the pair item existed; a per-batch bias still lands on cga2."""
    from cudnn.sdpa.fwd import heuristics

    offered = {s.name: i for i, s in enumerate(heuristics.ENGINE_SPECS)}
    plans = heuristics.recommend("A", _s13._facts(**_s13._resolve_dtypes(dict(d_qk=64, d_v=64, **_s13._bias_kw()))), offered)
    assert {p.knobs.cga for p in plans if offered[_ROW] == p.engine_id} == {1}
    plans = heuristics.recommend("A", _s13._facts(**_s13._resolve_dtypes(dict(d_qk=64, d_v=64, **_s13._bias_kw(b=2)))), offered)
    assert {p.knobs.cga for p in plans if offered[_ROW] == p.engine_id} == {2}


def test_config_factory_derives_the_pair_block_and_raises_on_each_coupling():
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    cfg, _ = c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=1, n_chunk=4))
    assert (cfg.N_CHUNK, cfg.BIAS_BLOCK_ROWS, cfg.BIAS_RESIDENT_TILES) == (4, 128, 6)
    assert c.bias_resident_block_bytes(cfg) == 6 * 128 * 128 * 2 and c.operand_region_end_bytes(cfg) < c.TCGEN05_V0_ADDR_LIMIT
    cfg64, _ = c.make_cfg_d64(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=1, n_chunk=8))
    assert (cfg64.N_CHUNK, cfg64.BIAS_BLOCK_ROWS, cfg64.BIAS_RESIDENT_TILES) == (8, 128, 3)
    legacy, _ = c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=1))
    assert (legacy.N_CHUNK, legacy.BIAS_BLOCK_ROWS, legacy.BIAS_RESIDENT_TILES) == (0, 256, 3)
    with pytest.raises(ValueError, match="needs bias_mode=1"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, n_chunk=4))
    with pytest.raises(ValueError, match="cga1 only"):
        c.make_cfg_d64(TemplateParams(dtype_qkv=2, cta_mma=2, bias_mode=1, n_chunk=4))
    with pytest.raises(ValueError, match="bottom-right"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=1, n_chunk=4, window_right=0, bottom_right=True))
    # the block-rows coupling is a validator predicate, not a comment
    import dataclasses

    bad = dataclasses.replace(cfg, BIAS_BLOCK_ROWS=256)
    with pytest.raises(ValueError, match="pair item"):
        c._validate_bias(bad, "sm107 d32")


@pytest.mark.parametrize(
    "kw, accept, n_chunk_expect, bias_mode_expect",
    [
        (dict(s_q=512, s_kv=512, bias="h"), True, 0, 2),  # S_q % 256 == 0: the one-batch-entry item streams (its block holds 384)
        (dict(s_q=512, s_kv=512, bias="h", n_chunk=8), True, 8, 1),
        (dict(s_q=512, s_kv=512, bias="h", n_chunk=0), True, 0, 2),  # the one-batch-entry item above ITS budget: the streamed ring
        (dict(s_q=768, s_kv=768, bias="h"), True, 0, 2),  # S_q % 256 == 0: the one-batch-entry ring (+8.6 % over the pair's resident block)
        (
            dict(s_q=896, s_kv=896, bias="h"),
            True,
            "rule",
            3,
        ),  # over the 6-tile budget: the pair's streamed ring in its cluster-multicast form (the measured default at D 32)
        (dict(s_q=512, s_kv=512, bias="b"), True, 0, 2),  # a per-batch bias: the one-batch-entry item's streamed ring
        (dict(s_q=256, s_kv=256, bias="b", n_chunk=4), False, None, None),
        (dict(s_q=256, s_kv=256, bias="h", b=1, n_chunk=4), False, None, None),
        (dict(d=64, s_q=384, s_kv=384, bias="h"), True, "rule", 1),
        (dict(d=64, s_q=384, s_kv=384, bias="h", cga=2), True, 0, 2),  # a requested cga2 keeps the one-batch-entry item: its 2-stage ring
        (dict(d=64, s_q=256, s_kv=256, bias="h"), True, "rule", 1),
    ],
    ids=[
        "d32_S512_one_batch_entry_ring",
        "d32_S512_n8",
        "d32_S512_n0_legacy_streamed",
        "d32_S768_one_batch_entry_ring",
        "d32_S896_over_streamed",
        "d32_S512_per_batch_streamed",
        "per_batch_n4",
        "one_row_n4",
        "d64_S384_pair",
        "d64_S384_cga2_legacy_streamed",
        "d64_S256_pair_cga1",
    ],
)
def test_adapter_check_support_is_the_rows_twin(_s13_rubin, kw, accept, n_chunk_expect, bias_mode_expect):
    from cudnn.sdpa.fwd import config_sm107 as c

    kw = dict(kw)
    d = kw.pop("d", 32)
    form = kw.pop("bias")
    b = kw.pop("b", 2)
    n_chunk = kw.pop("n_chunk", None)
    h = 4
    bias_shape = ((b if form == "b" else 1), h, kw["s_q"], kw["s_kv"])
    api = _s13._api(d, b=b, h=h, bias=bias_shape, n_chunk=n_chunk, **kw)
    if not accept:
        with pytest.raises((NotImplementedError, ValueError)):
            api.check_support()
        return
    api.check_support()
    params = api.template_params()
    assert params.bias_mode == bias_mode_expect and params.cta_mma == kw.get("cga", 1)
    if n_chunk_expect == "rule":
        # the rule at THIS host's SM count: even, >= 2, <= the flattened batch (the exact value depends on the device)
        assert params.n_chunk >= 2 and params.n_chunk % 2 == 0 and params.n_chunk <= b, params.n_chunk
    else:
        assert params.n_chunk == n_chunk_expect


@pytest.fixture
def _s13_rubin(monkeypatch):
    """The standalone adapter reads the device capability at construction: model a Rubin device on any host (the bias module's fixture, repeated here)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))


def test_adapter_declines_n_chunk_without_a_bias(_s13_rubin):
    api = _s13._api(32, b=2, h=4, s_q=256, s_kv=256, n_chunk=4)
    with pytest.raises(NotImplementedError, match="no bias"):
        api.check_support()


def test_template_params_and_cfg_are_appended():
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    import dataclasses

    fields = [f.name for f in dataclasses.fields(TemplateParams)]
    assert (
        fields[-7:] == ["n_chunk", "bias_stages", "padded_tail_form", "dead_subtile_quiet", "bias_max_form", "softmax_wgs", "s_slots"]
        and TemplateParams().n_chunk == 0
    ), "append-only; 0 renders every existing record byte-identically"
    assert c._CfgSm107.N_CHUNK == 0


def test_kernel_carries_the_pair_arm_as_one_const_expr_axis():
    """The pair item is one const_expr axis of the five warp roles: every role branches on _PAIR_ARM OUTSIDE its persistent loop (a
    dead arm inside the loop would add loop-carried names to the one-batch-entry rendering and move its cubin), the row / bounds
    arithmetic is shared by every role, the block row of the pair is the lane's row of the q tile."""
    src = _KERNEL.read_text()
    assert "_PAIR_ARM = CFG.N_CHUNK > 0" in src
    for fn in (
        "_pair_item_rows",
        "_pair_step_rows",
        "_pair_item_bounds",
        "_pair_step_bounds",
        "_tmaldg_item_pair",
        "_mma_item_pair",
        "_softmax_item_pair",
        "_corr_item_pair",
        "_tmastg_item_pair",
    ):
        assert f"def {fn}(" in src, fn
    assert src.count("    if cutlass.const_expr(_PAIR_ARM):\n") == 5, "one function-level branch per warp role"
    assert len(re.findall(r"^        while is_valid_tile > cutlass\.Int32\(0\):", src, re.M)) == 10, "five pair loops beside five one-batch-entry loops"
    assert "nb = rows[sub_tile_id]" in src and "block_row" in src  # the step's rows tuple (two, or three with the third softmax warp group)


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_PROBE_BODY = """
d = params_kw.pop("__d")
params = TemplateParams(**params_kw)
mod = _load_sm100_kernel_module((d, d), params, rubin=True)
print("BIAS_MODE", mod.CFG.BIAS_MODE)
print("KV_MASK_FORM", mod.CFG.KV_MASK_FORM)
print("N_CHUNK", mod.CFG.N_CHUNK)
print("BIAS_RESIDENT_TILES", mod.CFG.BIAS_RESIDENT_TILES)
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
    "VOTE": ("VOTE",),
}


def _probe(tmp_path, params, tag):
    return run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch="sm_107a", params=params, tag=tag)


@pytest.mark.parametrize(
    "d, extra, tiles",
    [
        (32, {"n_chunk": 4}, 6),
        (32, {"n_chunk": 4, "kv_mask_form": 1, "seq_kv_lens_present": True, "window_right": 0}, 6),
        (32, {"n_chunk": 2, "seq_kv_lens_present": True, "seq_q_lens_present": True}, 6),
        (64, {"n_chunk": 8}, 3),
    ],
    ids=["d32_pair4_dense", "d32_pair4_mask_padded_causal", "d32_pair2_padded_qtrim", "d64_pair8_dense"],
)
def test_pair_renderings_keep_the_split_the_mask_form_and_the_spill_count(tmp_path, d, extra, tiles):
    """Each pair rendering traced for sm_107a keeps the per-role register split (USETMAXREG > 0), no cluster-scope drain (CGAERRBAR 0),
    spills no more than the one-batch-entry BIAS rendering of the same width, the packed bias add (FFMA2 >= 128) and the software row
    max (FMNMX3 > 0); the bit-word form (R2P > 0) wherever a mask is on; and the 128-row block with the doubled tile count."""
    base = _probe(tmp_path, {"__d": d, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 1}, f"legacy_bias_d{d}")
    params = {"__d": d, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 1}
    params.update(extra)
    probe = _probe(tmp_path, params, f"pair_d{d}_{'_'.join(sorted(extra))}")
    assert probe.expect["N_CHUNK"] == extra["n_chunk"] and probe.expect["BIAS_BLOCK_ROWS"] == 128 and probe.expect["BIAS_RESIDENT_TILES"] == tiles
    assert base.expect["BIAS_BLOCK_ROWS"] == 256
    assert probe.cubin_md5 != base.cubin_md5
    assert probe.stats["USETMAXREG"] > 0 and probe.stats["CGAERRBAR"] == 0, probe.stats
    assert_no_new_spills(probe.stats, base.stats, tag=f"d{d} pair {sorted(extra)} vs the one-batch-entry bias rendering: ")
    assert probe.stats["FMNMX3"] > 0 and probe.stats["FFMA2"] >= 128, probe.stats
    if extra.get("kv_mask_form") or extra.get("seq_kv_lens_present"):
        assert probe.stats["R2P"] > 0, probe.stats


# --------------------------------------------------------------------------- tier 3: GPU cells (a Rubin device)

# The degenerate-input matrix of the pair item.  ``b`` is the FLATTENED batch (the rows the chunks partition); the bias is
# batch-broadcast [1, H, S, S] unless ``bias`` says otherwise.  ``n_chunk``: the pinned chunk length (None = the rule).
_CASES = {
    "pure_n_loop_one_tile_128": dict(b=8, h=2, h_kv=2, s_q=128, s_kv=128, n_chunk=4),
    "one_chunk_equals_n": dict(b=6, h=2, h_kv=2, s_q=256, s_kv=256, n_chunk=6),
    "chunk_of_one_dead_subtile_every_item": dict(b=4, h=2, h_kv=2, s_q=256, s_kv=256, n_chunk=1),
    "tail_chunk_dead_subtile": dict(b=5, h=2, h_kv=2, s_q=384, s_kv=384, n_chunk=4, causal=True),
    "bh_gt_1_nkv_1": dict(b=8, h=4, h_kv=4, s_q=128, s_kv=128, n_chunk=2),
    "tail_tile_304": dict(b=4, h=2, h_kv=2, s_q=304, s_kv=304, n_chunk=2),
    "zero_length_row_inside_a_chunk": dict(b=6, h=2, h_kv=2, s_q=256, s_kv=256, n_chunk=4, kv_lens=[256, 0, 100, 256, 0, 256]),
    "mixed_lengths_pair": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, n_chunk=4, kv_lens=[384, 130, 257, 1]),
    "padded_q_trim_pair": dict(
        b=6, h=4, h_kv=2, s_q=256, s_kv=256, n_chunk=4, causal=True, q_lens=[256, 100, 0, 256, 129, 200], kv_lens=[256, 200, 256, 0, 256, 128]
    ),
    "swa_causal_padded_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, n_chunk=2, causal=True, window_left=128, kv_lens=[384, 200, 384, 300]),
    "above_the_legacy_budget_512": dict(b=4, h=2, h_kv=2, s_q=512, s_kv=512, n_chunk=2),
    "the_6_tile_budget_768": dict(b=4, h=2, h_kv=2, s_q=768, s_kv=768, n_chunk=2, causal=True),
    "d64_3_tile_budget_384": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, d=64, n_chunk=2),
    "d64_256_cga1_pair": dict(b=4, h=2, h_kv=2, s_q=256, s_kv=256, d=64, n_chunk=2),
    "bias_x10_causal": dict(b=4, h=2, h_kv=2, s_q=384, s_kv=384, n_chunk=4, causal=True, bias_scale=10.0),
    "head_broadcast_bias": dict(b=4, h=4, h_kv=4, s_q=256, s_kv=256, n_chunk=2, bias="1"),
    "sink_padded": dict(b=4, h=2, h_kv=2, s_q=256, s_kv=256, n_chunk=2, causal=True, kv_lens=[256, 0, 200, 256], sink=True),
    "rule_pick_384": dict(b=8, h=4, h_kv=4, s_q=384, s_kv=384, n_chunk=None),
}


def _case_inputs(case, dtype, *, kv_mask_p=None):
    return _s13._case_inputs(case, dtype, kv_mask_p=kv_mask_p, table=_CASES)


def _run_pair(q, k, v, kw, n_chunk, _n_chunk):
    _n_chunk(n_chunk)
    return _s13._run(q, k, v, **kw)


@requires_rubin
@pytest.mark.parametrize("case", list(_CASES))
def test_pair_item_matches_the_fp64_oracle(case, _n_chunk):
    """The pair item over its degenerate-input matrix (bf16): O and Stats against the fp64 oracle, keyless rows exactly 0 / -inf, two
    launches bitwise, the plan on cga1."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    out = _run_pair(q, k, v, kw, c["n_chunk"], _n_chunk)
    assert out["knobs"].cga == 1, out["knobs"]
    _s13._check(out, ref)


@requires_rubin
@pytest.mark.parametrize("case", ["tail_chunk_dead_subtile", "zero_length_row_inside_a_chunk", "above_the_legacy_budget_512", "d64_3_tile_budget_384"])
def test_pair_item_matches_the_fp64_oracle_fp16(case, _n_chunk):
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.float16)
    out = _run_pair(q, k, v, kw, c["n_chunk"], _n_chunk)
    _s13._check(out, ref)


@requires_rubin
@pytest.mark.parametrize("case", ["tail_chunk_dead_subtile", "zero_length_row_inside_a_chunk", "swa_causal_padded_384", "above_the_legacy_budget_512"])
def test_pair_item_composes_with_the_kv_mask(case, _n_chunk):
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16, kv_mask_p=0.3)
    out = _run_pair(q, k, v, kw, c["n_chunk"], _n_chunk)
    _s13._check(out, ref)


@requires_rubin
@pytest.mark.parametrize("case", ["one_chunk_equals_n", "tail_chunk_dead_subtile", "zero_length_row_inside_a_chunk", "bh_gt_1_nkv_1", "swa_causal_padded_384"])
def test_o_and_stats_are_bitwise_across_n_chunk_and_the_one_batch_entry_item(case, _n_chunk):
    """Every row's kv loop runs the same tiles in the same order whatever the chunking, so O / Stats are BITWISE identical across
    n_chunk in {1, 2, 8, N} and against the one-batch-entry item (n_chunk = 0: the 256-row block) where its budget serves the cell."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    base = _run_pair(q, k, v, kw, c["b"], _n_chunk)
    _s13._check(base, ref)
    for n in (1, 2, 8):
        out = _run_pair(q, k, v, kw, n, _n_chunk)
        assert torch.equal(out["o"], base["o"]) and torch.equal(out["stats"], base["stats"]), f"{case}: n_chunk {n} differs from n_chunk = N"
    if c["s_kv"] <= 384 and c.get("d", 32) == 32:
        legacy = _run_pair(q, k, v, kw, 0, _n_chunk)
        assert torch.equal(legacy["o"], base["o"]) and torch.equal(legacy["stats"], base["stats"]), f"{case}: the one-batch-entry item differs"


@requires_rubin
@pytest.mark.parametrize("dtype", _s13._DTYPES, ids=_s13._DTYPE_IDS)
@pytest.mark.parametrize("case", ["one_chunk_equals_n", "tail_chunk_dead_subtile", "above_the_legacy_budget_512"])
def test_identity_bias_zero_is_bitwise_the_plain_kernel(case, dtype, _n_chunk):
    """bias = 0 on the pair item is BITWISE the plain kernel on O and Stats (the -inf masked body with fma(0, 1/scale, s) == s)."""
    c = _CASES[case]
    q, k, v, kw, _ = _case_inputs(case, dtype)
    _n_chunk(None)
    plain = _s13._run(q, k, v, **{**kw, "bias": None})
    zero = torch.zeros(1, 1, c["s_q"], c["s_kv"], device="cuda", dtype=dtype)
    out = _run_pair(q, k, v, {**kw, "bias": zero}, c["n_chunk"] or 4, _n_chunk)
    assert torch.equal(out["o"], plain["o"]) and torch.equal(out["stats"], plain["stats"])


@requires_rubin
def test_declines_on_real_graphs(_n_chunk):
    """A batch-broadcast bias at S_kv 896 (over the 6-tile budget), a per-batch bias at S_kv 512 (the one-batch-entry budget) and a
    single-row graph at S_kv 512 used to leave the FROST row out; since the STREAMED ring they are SERVED too (test_sdpa_bias_streamed_sm107.py
    pins their form) -- the resident cells S_kv 512 / 768 on >= 2 rows stay served."""
    import cudnn

    _n_chunk(None)
    torch.manual_seed(3)
    d = 32

    def plans(b, s, bias_b):
        q, k, v = (_s13._bhsd(b, 2, s, d, torch.bfloat16) for _ in range(3))
        bias = torch.randn(bias_b, 2, s, s, device="cuda").to(torch.bfloat16)
        g, *_ = _s13._build(q, k, v, scale=1.0 / math.sqrt(d), bias=bias)
        try:
            g.create_execution_plans([cudnn.heur_mode.A])
            return _s13._frost_plans(g)
        except cudnn.cudnnGraphNotSupportedError:
            return [], []

    for b, s, bias_b in ((2, 896, 1), (2, 512, 2), (1, 512, 1)):
        names, ours = plans(b, s, bias_b)
        assert ours, ("served by the streamed ring", b, s, bias_b, names)
    for b, s in ((2, 512), (2, 768), (4, 640)):
        names, ours = plans(b, s, 1)
        assert ours, (b, s, names)
