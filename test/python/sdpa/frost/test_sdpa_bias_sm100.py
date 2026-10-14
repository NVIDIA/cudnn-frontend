# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The additive attention bias and the per-key kv_mask on the SM100 (cc 10.0 / 10.3) f16/bf16 small head-dim tiles -- the RESIDENT
row-block arm of ``sm100/prefill_d128_f16.py`` (``TemplateParams.bias_mode = 1`` / ``kv_mask_form`` 1 / 2 / 3; the first body seam of the
Rubin small tiles' port onto the SM100 file).  The Rubin module ``test_sdpa_bias_sm107`` carries the arch-neutral pieces (the fp64 oracle of
the whole contract, the graph builder, the degenerate-input cells, the checks); this module runs them against the SM100 row at the SM100
budget: the 32-wide tile holds S_kv <= 256 at cga1, the 64-wide tile S_kv <= 128 at cga2 (``config_sm100.sm100_bias_resident_s_kv_max``:
the shared arithmetic at 227 KiB), the kv_mask alone any S_kv (nothing staged in SMEM).  What the SM100 file does not render yet -- the
streamed / multicast ring, the (n, n+1) pair item, the 64-wide KV tile, the fp32 additive vector together with a bias -- is a typed decline
on the row and on the adapter (rule 8b twins), pinned here on real graphs and on the standalone adapter.

Tiers: the host contract (the row's claims, the adapter's ``check_support`` twin under a modelled cc 10.0 device, the kernel module's
constants and its not-yet-wired backstops), then the GPU cells on a cc 10.0 / 10.3 device (the dev box is SHARED: functional only, never a
number): keyless rows exactly 0 / -inf, live rows against the fp64 oracle, two launches bitwise.
"""

import pytest
import torch

import test_sdpa_bias_sm107 as _t7
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

_ROW = "sdpa_fwd_prefill_sm100"
requires_sm100_line = pytest.mark.skipif(_SM not in (100, 103), reason="needs a cc 10.0 / 10.3 device, have " + ("none" if _SM is None else f"sm_{_SM}"))


def _caps():
    from cudnn.sdpa.fwd import engines

    return {s.name: s.capabilities for s in engines.ENGINE_SPECS}[_ROW]


# --------------------------------------------------------------------------- tier 1: the row's claims (host)


def test_row_claims_the_resident_bias_on_the_derived_shapes_and_budgets():
    """The SM100 row claims the bias on the shapes the SM100 budget derives (``SM100_BIAS_D_SHAPES`` = the 32- and the 64-wide tile), the
    io-dtype bias only, the kv_mask in its four dtypes; the resident S_kv budgets are the shared arithmetic at 227 KiB: 256 at the
    32-wide tile's cga1, none at the 64-wide tile's cga1, 128 at its cga2."""
    import cudnn
    from cudnn.sdpa.fwd import config_sm100 as c0

    caps = _caps()
    assert caps.bias is True and caps.kv_mask is True and caps.bias_io_dtype_only is True
    assert caps.bias_d_shapes == c0.SM100_BIAS_D_SHAPES == frozenset({(32, 32), (64, 64)})
    assert caps.kv_mask_dtypes == frozenset({cudnn.data_type.BOOLEAN, cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT})
    assert c0.sm100_bias_resident_s_kv_max(32, 1) == 256
    assert c0.sm100_bias_resident_s_kv_max(64, 1) == 0
    assert c0.sm100_bias_resident_s_kv_max(64, 2) == 128
    # the streamed ring (the one-batch-entry item): 2 stages at the 32-wide tile's cga1, none at the 64-wide tile (cga1 holds no form, cga2 one tile)
    assert c0.sm100_bias_ring_stages(32, 1) == 2 and c0.sm100_bias_ring_stages(64, 1) == 0 and c0.sm100_bias_ring_stages(64, 2) == 0
    assert c0.sm100_bias_mode_pick(32, 1, 384) == 2 and c0.sm100_bias_mode_pick(32, 1, 256) == 1 and c0.sm100_bias_mode_pick(64, 2, 256) == 0
    # the heuristics' width pick = the row's: the narrowest width whose block holds S_kv
    assert c0.sm100_bias_auto_cga(32, 256, frozenset({1})) == 1
    assert c0.sm100_bias_auto_cga(64, 128, frozenset({1, 2})) == 2
    # the kernel's preload constant: the Rubin table as the prior (0 at the 128-wide tile)
    assert c0.sm100_bias_preload_chunks(128) == 0 and c0.sm100_bias_preload_chunks(64) == 1


def test_kernel_module_constants_and_the_not_yet_wired_backstops():
    """The SM100 file's bias arms fold on the record (the module constants), and the seams the file does not render yet raise at
    module load with a message naming them (the config validators accept the records; the kernel states its own envelope)."""
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    def _mod(**kw):
        return _load_sm100_kernel_module(
            (kw.get("d_flavor", 32),) * 2, TemplateParams(dtype_qkv=2, cta_mma=1, d_flavor=32, **{k: v for k, v in kw.items() if k != "d_flavor"}), rubin=False
        )

    plain = _mod()
    assert not plain._BIAS_ARM and not plain._KV_MASK_ARM and not plain._INF_MASK_ARM and plain._BIAS_GG is None
    res = _mod(bias_mode=1)
    assert res._BIAS_ARM and res._BIAS_RESIDENT and res._INF_MASK_ARM and not res._HW_UPPER_MAX
    assert res._BIAS_GG.n_tiles == res.CFG.BIAS_RESIDENT_TILES == 2 and res._BIAS_GG.block_rows == 256 and res._BIAS_GG.boxes_per_tile == 2
    assert res.BIAS_PRELOAD_CHUNKS == 0 and res.CFG.BIAS_EMPTY_ARRIVERS == 8
    mask = _mod(kv_mask_form=1, bias_max_form=1)
    assert mask._KV_MASK_BOOL and mask._HW_UPPER_MAX and not mask._BIAS_ARM and mask._INF_MASK_ARM
    ring = _mod(bias_mode=2, bias_stages=2)
    assert ring._BIAS_STREAMED and not ring._BIAS_RESIDENT and ring.CFG.STAGES_BIAS == 2 and ring._BIAS_GG.n_tiles == 2
    with pytest.raises(ValueError, match="multicast bias ring"):
        _mod(bias_mode=3, bias_stages=2, n_chunk=2)
    with pytest.raises(ValueError, match="pair item"):
        _mod(bias_mode=1, n_chunk=2)
    t64 = _mod(bias_mode=1, tile_n=64)
    assert t64.CFG.TILE_N == 64 and t64.CFG.N_BMM2_CHUNKS == 1 and t64._BIAS_GG.boxes_per_tile == 1 and t64.BIAS_PRELOAD_CHUNKS == 1
    assert t64.CFG.BIAS_RESIDENT_TILES == 5  # the 32-wide tile holds 5 x 64 = 320 kv columns resident at the 64-wide KV tile


# --------------------------------------------------------------------------- tier 2: the adapter's check_support twin (host, a modelled cc 10.0 device)


@pytest.fixture
def _sm100_device_for_the_adapter(monkeypatch):
    """The standalone adapter reads the device capability at construction: model a cc 10.0 device on any host."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))


def _api(d=32, *, s_q=256, s_kv=256, b=2, h=4, h_kv=None, dtype=torch.bfloat16, bias=None, kv_mask=None, kv_mask_dtype=torch.bool, **kw):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    h_kv = h if h_kv is None else h_kv
    q, o = (_t7._desc((b, h, s_q, d), dtype, n) for n in ("q", "o"))
    k, v = (_t7._desc((b, h_kv, s_kv, d), dtype, n) for n in ("k", "v"))
    if bias is not None:
        kw["sample_bias"] = _t7._desc(bias, dtype, "bias")
    if kv_mask is not None:
        kw["sample_kv_mask"] = _t7._desc(kv_mask, kv_mask_dtype, "kv_mask")
    return SdpaFwdDslSm100(q, k, v, o, None, **kw)


# (name, adapter kwargs, accept?, flavor, bias_mode, kv_mask_form, cta_mma): the budget pick at the SM100 constants; a request the file
# does not render is declined typed, never degraded.
_ADAPTER_CELLS = [
    ("bias d32 S256 (the resident edge)", dict(bias=(1, 4, 256, 256)), True, (32, 32), 1, 0, 1),
    ("bias d32 S128 per batch", dict(s_q=128, s_kv=128, bias=(2, 4, 128, 128)), True, (32, 32), 1, 0, 1),
    ("bias d32 S384 -> the 2-stage streamed ring", dict(s_q=384, s_kv=384, bias=(1, 4, 384, 384)), True, (32, 32), 2, 0, 1),
    ("bias d32 S1024 -> the streamed ring", dict(s_q=1024, s_kv=1024, bias=(1, 4, 1024, 1024)), True, (32, 32), 2, 0, 1),
    ("bias_mode=2 request at S256 (honoured)", dict(bias=(1, 4, 256, 256), bias_mode=2), True, (32, 32), 2, 0, 1),
    ("bias d64 S128 -> cga2 block", dict(d=64, s_q=128, s_kv=128, bias=(1, 4, 128, 128)), True, (64, 64), 1, 0, 2),
    ("bias d64 S256 (no form on this line yet)", dict(d=64, s_q=256, s_kv=256, bias=(1, 4, 256, 256)), False, None, None, None, None),
    ("bias d64 S128 at a requested cga1 (holds no block)", dict(d=64, s_q=128, s_kv=128, bias=(1, 4, 128, 128), cga=1), False, None, None, None, None),
    ("kv_mask alone d32 S1024 (no budget)", dict(s_q=1024, s_kv=1024, kv_mask=(2, 1, 1, 1024)), True, (32, 32), 0, 1, 1),
    ("bias + bool kv_mask d32 S256", dict(bias=(1, 4, 256, 256), kv_mask=(2, 1, 1, 256)), True, (32, 32), 1, 1, 1),
    ("bias + bf16 additive kv_mask d32 S256", dict(bias=(1, 4, 256, 256), kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.bfloat16), True, (32, 32), 1, 2, 1),
    ("fp32 additive kv_mask alone d32 S256", dict(kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.float32), True, (32, 32), 0, 3, 1),
    (
        "bias + fp32 additive kv_mask -> the 64-wide KV tile (the pick)",
        dict(bias=(1, 4, 256, 256), kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.float32),
        True,
        (32, 32),
        1,
        3,
        1,
    ),
    (
        "bias + fp32 additive kv_mask at a requested tile_n 128",
        dict(bias=(1, 4, 256, 256), kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.float32, tile_n=128),
        False,
        None,
        None,
        None,
        None,
    ),
    # (the standalone adapter needs a DENSE S_kv at a multiple of 128 without a padding mask -- the graph path synthesizes it; the GPU cells carry the tails)
    ("bias d32 S256 at tile_n 64 (4 of the 5 resident tiles)", dict(bias=(1, 4, 256, 256), tile_n=64), True, (32, 32), 1, 0, 1),
    ("bias d32 S640 at tile_n 64 (the 3-stage ring)", dict(s_q=640, s_kv=640, bias=(1, 4, 640, 640), tile_n=64), True, (32, 32), 2, 0, 1),
    (
        "bias d64 S128 at tile_n 64 -> cga1 (2 of its 3 tiles; the 128-wide tile picks cga2)",
        dict(d=64, s_q=128, s_kv=128, bias=(1, 4, 128, 128), tile_n=64),
        True,
        (64, 64),
        1,
        0,
        1,
    ),
    ("bias_mode=3 request (the multicast ring)", dict(bias=(1, 1, 256, 256), bias_mode=3), False, None, None, None, None),
    ("bias_stages=3 request (only 2 fit)", dict(s_q=384, s_kv=384, bias=(1, 4, 384, 384), bias_stages=3), False, None, None, None, None),
    ("n_chunk=2 request (the pair item)", dict(bias=(1, 1, 256, 256), n_chunk=2), False, None, None, None, None),
    ("tile_n=256 request (outside the domain)", dict(bias=(1, 4, 256, 256), tile_n=256), False, None, None, None, None),
    ("softmax_wgs=3 request", dict(bias=(1, 1, 256, 256), softmax_wgs=3), False, None, None, None, None),
]


@pytest.mark.parametrize("name,kw,accept,flavor,bias_mode,kv_mask_form,cta_mma", _ADAPTER_CELLS, ids=[c[0] for c in _ADAPTER_CELLS])
def test_adapter_check_support_is_the_rows_twin(_sm100_device_for_the_adapter, name, kw, accept, flavor, bias_mode, kv_mask_form, cta_mma):
    """The standalone adapter's ``check_support`` on a cc 10.0 device (rule 8b twin): the same cells accept with the compile-time record
    carrying ``bias_mode`` / ``kv_mask_form`` on the expected flavor at the budget's width, or decline with one of the two TYPED exceptions."""
    if not accept:
        with pytest.raises((NotImplementedError, ValueError)):
            _api(**kw).check_support()
        return
    api = _api(**kw)
    assert api.check_support() is True, name
    assert api.flavor == flavor, (name, api.flavor)
    params = api.template_params()
    assert (params.bias_mode, params.kv_mask_form) == (bias_mode, kv_mask_form), (name, params)
    _want_tile_n = kw.get("tile_n", 64 if (kw.get("kv_mask_dtype") == torch.float32 and kw.get("bias") is not None) else 128)
    assert params.d_flavor == flavor[0] and params.cta_mma == cta_mma and params.n_chunk == 0 and params.tile_n == _want_tile_n, (name, params)
    _want_stages = 0 if bias_mode != 2 else (3 if _want_tile_n == 64 else 2)  # the derived deepest ring: 3 stages at the 64-wide KV tile, 2 at the 128-wide
    assert params.bias_stages == _want_stages and params.s_slots == 1 and params.q_in_tmem == 0 and params.softmax_wgs == 2, (name, params)


# --------------------------------------------------------------------------- tier 3: the GPU cells (a cc 10.0 / 10.3 device)

# The Rubin module's degenerate-input matrix INSIDE the SM100 resident budget (S_kv <= 256 at the 32-wide tile, <= 128 at the 64-wide one):
# the KV ring sweep n_kv 1 / 2 with B*H > 1, ONE CTA / one tile in the whole launch (S_kv 128 and 64), a tail tile with a padded bias row
# stride (217 / 100), GQA, a zero-length KV batch (an EMPTY work item: no bias barrier is touched), a zero-key batch WITH a sink and the
# padded-Q trim, the bias in its four broadcast forms, the magnitude cells, the 64-wide tile at cga2 (one tile), D 24 on the 32-wide tile.
_IN_BUDGET = [
    "dense_one_tile_bh1",
    "dense_nkv1",
    "dense_nkv2",
    "causal_gqa_8_2",
    "padded_kv_zero_batch",
    "causal_sink_padded_q_trim",
    "bias_head_broadcast",
    "bias_per_batch",
    "bias_per_batch_head_broadcast",
    "bias_x10",
    "bias_neg1e9_key_fold",
    "d64_dense_128",
    "d64_padded_zero_batch_128",
    "d24_dense_256",
    "dense_one_tile_64",
    "tail_217_padded_stride_bias",
    "causal_tail_100_padded_stride_bias",
]
# Above the SM100 resident budget (the Rubin module streams or pairs these): a typed decline of the row on a real graph -- no FROST plan.
# The one-batch-entry item ABOVE the resident budget at the 32-wide tile: the 2-stage STREAMED ring (n_kv 3 = one stage past the ring, 4 = two
# wraps, 5, 8 with a tail; the causal band; a padded zero-length batch on the ring: an empty item touches no ring row).
_RING_CASES = {
    "ring_dense_nkv3_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384),
    "ring_causal_384": dict(b=1, h=4, h_kv=4, s_q=384, s_kv=384, causal=True),
    "ring_x10_causal_384": dict(b=1, h=4, h_kv=4, s_q=384, s_kv=384, causal=True, bias_scale=10.0),
    "ring_dense_nkv4_512": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512),
    "ring_causal_nkv5_640": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640, causal=True),
    # S_kv 904 (not 900): a COMPACT bias needs S_kv % 8 == 0 for its TMA row stride (the row declines 900 typed; the Rubin module runs 900 on the mask-only arm)
    "ring_padded_nkv8_tail_904_zero_batch": dict(b=3, h=2, h_kv=2, s_q=904, s_kv=904, kv_lens=[904, 0, 500]),
    "ring_dense_nkv8_1024_head_broadcast": dict(b=2, h=2, h_kv=2, s_q=1024, s_kv=1024, bias="1"),
}
# No form at any width of the flavor's domain: the 64-wide tile's cga2 holds one tile and no ring; cga1 none -- a typed decline on real graphs.
_ABOVE_BUDGET = ["d64_dense_256_cga2"]


def _expected_cga(case):
    c = _t7._CASES[case]
    return 2 if c.get("d", 32) == 64 else 1  # the 64-wide tile's bias block lives at cga2 on this line; the 32-wide tile at its cga1


def _run(q, k, v, *, scale, bias=None, kv_mask=None, sdpa_kwargs=None, seq_len_kv=None, seq_len_q=None, sink=None, stats=True, cga=None, tile_n=None):
    """The Rubin module's ``_run`` against the SM100 row: build the graph, pin the row's unsplit plan (the heuristics' pick; a cga / tile_n
    request through the public autotune record), launch TWICE on sentinel-filled outputs (the two-launch delta) and return O, Stats and
    the plan's knobs."""
    import cudnn
    from types import SimpleNamespace

    g, vp, o, o_gpu, stats_gpu = _t7._build(
        q, k, v, scale=scale, bias=bias, kv_mask=kv_mask, sdpa_kwargs=sdpa_kwargs, seq_len_kv=seq_len_kv, seq_len_q=seq_len_q, sink=sink, stats=stats
    )
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    ours = [i for i, n in enumerate(names) if n == _ROW or n.startswith(_ROW + "[")]
    idx = next((i for i in ours if g.plans[i].knobs.split_kv in (None, 1)), None)
    assert idx is not None, f"no unsplit {_ROW} plan in {names}"
    picked = g.plans[idx].knobs
    picked_cga, picked_tile_n = picked.cga, picked.tile_n
    want_cga = picked_cga if cga is None else cga
    want_tile_n = picked_tile_n if tile_n is None else tile_n
    if (want_cga, want_tile_n) != (picked_cga, picked_tile_n):
        engine_id, record = g.get_engine_and_knobs_at_index(idx)
        record = dict(record)
        if want_cga != picked_cga:
            record[cudnn.knob_type.TILE_CGA_M] = want_cga
        if want_tile_n != picked_tile_n:
            record[cudnn.knob_type.TILE_N] = want_tile_n
        g.create_execution_plan(engine_id, record)
        idx = g.get_execution_plan_count() - 1
        picked_cga, picked_tile_n = want_cga, want_tile_n
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    vp[o] = o_gpu
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

    def launch():
        o_gpu.fill_(float("nan"))  # sentinel: an unwritten live row stays NaN
        if stats_gpu is not None:
            stats_gpu.fill_(float("nan"))
        g.execute(vp, ws)
        torch.cuda.synchronize()
        return o_gpu.clone(), (stats_gpu.clone() if stats_gpu is not None else None)

    o1, st1 = launch()
    o2, st2 = launch()
    assert torch.equal(o1, o2) and (st1 is None or torch.equal(st1, st2)), "two-launch delta on O / Stats"
    return dict(o=o_gpu, stats=stats_gpu, knobs=SimpleNamespace(cga=picked_cga, tile_n=picked_tile_n), names=names)


def _frost_plan_names(q, k, v, kw):
    import cudnn

    g, *_ = _t7._build(
        q,
        k,
        v,
        scale=kw["scale"],
        bias=kw["bias"],
        kv_mask=kw["kv_mask"],
        sdpa_kwargs=kw["sdpa_kwargs"],
        seq_len_kv=kw["seq_len_kv"],
        seq_len_q=kw["seq_len_q"],
        sink=kw["sink"],
    )
    try:
        g.create_execution_plans([cudnn.heur_mode.A])
    except Exception as exc:  # noqa: BLE001 -- the typed graph-level decline
        if type(exc).__name__ == "cudnnGraphNotSupportedError":
            # no engine at all proposed a plan (a kv_mask graph has no backend lowering): the row's decline IS the whole answer
            return []
        raise
    return [g.get_plan_name_at_index(i) for i in range(len(g.plans))]


# (case, dtype) pairs: every in-budget case in bf16; fp16 on every case but the -1e9 key fold (not representable in fp16 -- the models'
# fp16 fold is a smaller finite value; the Rubin module runs that case in bf16 only too).
_BIAS_CELLS = [(c, torch.bfloat16) for c in _IN_BUDGET] + [(c, torch.float16) for c in _IN_BUDGET if c != "bias_neg1e9_key_fold"]


@requires_sm100_line
@pytest.mark.parametrize("case,dtype", _BIAS_CELLS, ids=[f"{c}-{'bf16' if d == torch.bfloat16 else 'fp16'}" for c, d in _BIAS_CELLS])
def test_bias_matches_the_fp64_oracle_on_sm100(case, dtype):
    """The additive bias over the degenerate-input matrix inside the SM100 resident budget, both dtypes: the plan is the row's unsplit plan
    at the budget's width (cga1 on the 32-wide tile, cga2 on the 64-wide one), O and Stats against the fp64 oracle, keyless rows exactly
    0 / -inf, two launches bitwise; a zero-length KV batch on the output directly."""
    q, k, v, kw, ref = _t7._case_inputs(case, dtype)
    kw.pop("cga", None)  # the heuristics' width on this line (the Rubin cells' cga requests are the Rubin budget's)
    out = _run(q, k, v, **kw)
    assert out["knobs"].cga == _expected_cga(case) and out["knobs"].tile_n == 128, (case, out["knobs"])
    _t7._check(out, ref)
    c = _t7._CASES[case]
    if "kv_lens" in c and 0 in c["kv_lens"]:
        empty_b = c["kv_lens"].index(0)
        st = out["stats"][empty_b].squeeze(-1)
        assert (out["o"][empty_b] == 0).all(), "a zero-length KV batch: O = 0 (a SELECT, never a multiply by zero)"
        if kw["sink"] is None:
            assert torch.isneginf(st).all(), "a zero-length KV batch without a sink: Stats = -inf"
        else:
            q_len = c["q_lens"][empty_b] if "q_lens" in c else c["s_q"]
            torch.testing.assert_close(st[:, :q_len], kw["sink"].flatten().to(st.dtype).view(-1, 1).expand(c["h"], q_len), **_t7._TOL_LSE)
            assert torch.isneginf(st[:, q_len:]).all(), "the padded-Q trim: Stats = -inf at and past seq_len_q"


@requires_sm100_line
@pytest.mark.parametrize("case", ["dense_nkv2", "padded_kv_zero_batch", "causal_gqa_8_2", "d64_dense_128", "causal_sink_padded_q_trim"])
@pytest.mark.parametrize("with_bias", [False, True], ids=["mask_only", "mask_and_bias"])
def test_kv_mask_matches_the_fp64_oracle_on_sm100(case, with_bias):
    """The per-key boolean mask (30 % of the keys False) alone and composed with the bias, the band mask, the padding and a zero-length
    batch: a masked key is exactly a padded one in the oracle.  The mask-only cells run the hardware upper-bound row max (the pick's
    default on that form), the bias cells the software tree."""
    q, k, v, kw, ref = _t7._case_inputs(case, torch.bfloat16, kv_mask_p=0.3)
    kw.pop("cga", None)
    if not with_bias:
        kw["bias"] = None
        ref = _t7._ref_of(_t7._CASES[case], q, k, v, kw)
    out = _run(q, k, v, **kw)
    # a bias plan takes the budget's width; the kv_mask alone stages no block and keeps the tiles' own cga1 default on this line
    assert out["knobs"].cga == (_expected_cga(case) if with_bias else 1), (case, out["knobs"])
    _t7._check(out, ref)


@requires_sm100_line
@pytest.mark.parametrize("case", list(_t7._MASK_RING_CASES))
def test_kv_mask_alone_serves_any_s_kv_across_the_ring_wrap_on_sm100(case):
    """The kv_mask ALONE has no resident budget (nothing staged in SMEM): S_kv 512 / 640 / 900 / 1024 = n_kv 4 (the ring depth), 5 (one
    past it), 8 (two wraps; 900 = a tail + right padding) on the -inf masked body, 30 % of the keys False, against the fp64 oracle."""
    q, k, v, kw, _ = _t7._case_inputs(case, torch.bfloat16, kv_mask_p=0.3, table=_t7._MASK_RING_CASES)
    kw.pop("cga", None)
    kw["bias"] = None
    out = _run(q, k, v, **kw)
    _t7._check(out, _t7._ref_of(_t7._MASK_RING_CASES[case], q, k, v, kw))


@requires_sm100_line
@pytest.mark.parametrize("case", list(_t7._MASK_ONLY_CASES))
def test_kv_mask_composes_with_bottom_right_causal_keyless_rows_and_the_sink_on_sm100(case):
    """The per-key mask under the bottom-right band: the rows above the diagonal (no key) read Stats = sink_logit / O = 0, the trimmed
    rows -inf / 0, the live rows match the oracle with 30 % of the keys masked."""
    q, k, v, kw, _ = _t7._case_inputs(case, torch.bfloat16, kv_mask_p=0.3, table=_t7._MASK_ONLY_CASES)
    kw.pop("cga", None)
    kw["bias"] = None
    out = _run(q, k, v, **kw)
    _t7._check(out, _t7._ref_of(_t7._MASK_ONLY_CASES[case], q, k, v, kw))


@requires_sm100_line
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_additive_kv_mask_vector_matches_the_fp64_oracle_on_sm100(dtype):
    """The ADDITIVE per-key vector (the models' finite fold in the io dtype: 0 on a live key, -1e9 on a masked one) alone and with a bias,
    and the fp32 vector alone (with a bias it is the 64-wide tile's form: a typed decline on this line, pinned below)."""
    for with_bias, vec_dtype in ((False, dtype), (True, dtype), (False, torch.float32)):
        q, k, v, kw, _ = _t7._case_inputs("dense_nkv2", dtype, kv_mask_p=0.3)
        kw.pop("cga", None)
        keep = kw["kv_mask"]
        fold = -1e9 if vec_dtype != torch.float16 else -6.0e4  # the models' finite fold; fp16 holds no -1e9
        kw["kv_mask"] = torch.where(keep, torch.zeros((), device="cuda", dtype=vec_dtype), torch.full((), fold, device="cuda", dtype=vec_dtype)).to(vec_dtype)
        if not with_bias:
            kw["bias"] = None
        ref_kw = dict(kw, kv_mask=keep)
        out = _run(q, k, v, **kw)
        _t7._check(out, _t7._ref_of(_t7._CASES["dense_nkv2"], q, k, v, ref_kw))


@requires_sm100_line
@pytest.mark.parametrize("case", list(_RING_CASES))
@pytest.mark.parametrize("with_mask", [False, True], ids=["bias", "bias_and_mask"])
def test_streamed_ring_matches_the_fp64_oracle_on_sm100(case, with_mask):
    """The 2-stage streamed bias ring of the one-batch-entry item at the 32-wide tile (S_kv above the resident 256): the ring wrap axis
    n_kv 3 / 4 / 5 / 8 (with a tail and a zero-length batch), the causal band, the magnitude cell, the head-broadcast form, with and without
    the bit-word mask; the plan's record carries bias_mode 2 at cga1; O and Stats against the fp64 oracle, two launches bitwise."""
    q, k, v, kw, ref = _t7._case_inputs(case, torch.bfloat16, kv_mask_p=0.3 if with_mask else None, table=_RING_CASES)
    kw.pop("cga", None)
    out = _run(q, k, v, **kw)
    assert out["knobs"].cga == 1 and out["knobs"].tile_n == 128, (case, out["knobs"])
    _t7._check(out, ref)


# The 64-wide KV tile (the TILE_N knob through the public autotune record): the resident block holds 5 tiles (S_kv 320) at the 32-wide tile and
# 3 (192) at the 64-wide tile's cga1, the ring 3 stages above; the one-chunk softmax body; the fp32 vector WITH a bias is served HERE only.
_T64_CASES = {
    "t64_dense_nkv2_256": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256),
    "t64_causal_320": dict(b=2, h=2, h_kv=2, s_q=320, s_kv=320, causal=True),
    "t64_ring_dense_640": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640),
    "t64_ring_padded_1024_zero_batch": dict(b=3, h=2, h_kv=2, s_q=1024, s_kv=1024, kv_lens=[1024, 0, 700]),
    "t64_d64_dense_192_cga1": dict(b=2, h=2, h_kv=2, s_q=192, s_kv=192, d=64),
    "t64_d64_causal_tail_200": dict(b=2, h=2, h_kv=2, s_q=200, s_kv=200, d=64, causal=True),
}


@requires_sm100_line
@pytest.mark.parametrize("case", list(_T64_CASES))
@pytest.mark.parametrize("with_mask", [False, True], ids=["bias", "bias_and_mask"])
def test_tile_n_64_matches_the_fp64_oracle_on_sm100(case, with_mask):
    """The 64-wide KV tile on the SM100 small tiles (TILE_N=64 through the autotune record): the resident block (S_kv 256 / 320 at D 32, 192 at D 64
    cga1) and the 3-stage ring above it (640, 1024 with a zero-length batch), the one-chunk body, with and without the bit-word mask."""
    q, k, v, kw, ref = _t7._case_inputs(case, torch.bfloat16, kv_mask_p=0.3 if with_mask else None, table=_T64_CASES)
    kw.pop("cga", None)
    out = _run(q, k, v, tile_n=64, **kw)
    assert out["knobs"].tile_n == 64 and out["knobs"].cga == 1, (case, out["knobs"])
    _t7._check(out, ref)


@requires_sm100_line
def test_fp32_additive_vector_with_a_bias_runs_at_tile_n_64_on_sm100():
    """The fp32 additive kv_mask WITH a bias: a knob-less graph lands on the 64-wide KV tile (the pick; the 128-wide rendering spills) and matches the
    oracle; the same graph at an explicit TILE_N 128 proposes no plan of the row."""
    q, k, v, kw, _ = _t7._case_inputs("dense_nkv2", torch.bfloat16, kv_mask_p=0.3)
    kw.pop("cga", None)
    keep = kw["kv_mask"]
    kw["kv_mask"] = torch.where(keep, torch.zeros((), device="cuda"), torch.full((), -1e9, device="cuda")).float()
    out = _run(q, k, v, **kw)
    assert out["knobs"].tile_n == 64, out["knobs"]
    _t7._check(out, _t7._ref_of(_t7._CASES["dense_nkv2"], q, k, v, dict(kw, kv_mask=keep)))


@requires_sm100_line
@pytest.mark.parametrize("case", _ABOVE_BUDGET)
def test_declines_above_the_resident_budget_on_real_graphs(case):
    """A bias graph no form holds at any width of the flavor's domain (the 64-wide tile at S_kv 256: cga2's block holds 128 and no ring fits)
    proposes NO plan of the row -- a typed decline on the real graph, never a plan that fails at build."""
    q, k, v, kw, _ = _t7._case_inputs(case, torch.bfloat16)
    names = _frost_plan_names(q, k, v, kw)
    assert not any(n == _ROW or n.startswith(_ROW + "[") for n in names), (case, names)


@requires_sm100_line
def test_declines_the_fp32_additive_vector_with_a_bias_at_tile_n_128_on_a_real_graph():
    """The fp32 additive kv_mask WITH a bias at an EXPLICIT TILE_N 128 (the autotune record): the row declines it typed (the 128-wide rendering
    spills), never a degraded plan -- the knob-less graph runs at 64 (the test above)."""
    import cudnn

    q, k, v, kw, _ = _t7._case_inputs("dense_nkv2", torch.bfloat16, kv_mask_p=0.3)
    kw["kv_mask"] = torch.where(kw["kv_mask"], torch.zeros((), device="cuda"), torch.full((), -1e9, device="cuda")).float()
    kw.pop("cga", None)
    out = _run(q, k, v, **kw)  # the pick: 64
    g, *_ = _t7._build(
        q,
        k,
        v,
        scale=kw["scale"],
        bias=kw["bias"],
        kv_mask=kw["kv_mask"],
        sdpa_kwargs=kw["sdpa_kwargs"],
        seq_len_kv=kw["seq_len_kv"],
        seq_len_q=kw["seq_len_q"],
        sink=kw["sink"],
    )
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    idx = next(i for i, nm in enumerate(names) if nm == _ROW or nm.startswith(_ROW + "["))
    engine_id, record = g.get_engine_and_knobs_at_index(idx)
    record = dict(record)
    record[cudnn.knob_type.TILE_N] = 128
    with pytest.raises(Exception):  # the typed decline of the row on the replayed record (NotImplementedError / the graph's not-supported error)
        g.create_execution_plan(engine_id, record)
        g.select_plan(g.get_execution_plan_count() - 1)
        g.check_support()
        g.build_plans()
    assert out["knobs"].tile_n == 64
