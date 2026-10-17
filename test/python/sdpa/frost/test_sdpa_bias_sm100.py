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
number): keyless rows exactly 0 / -inf, live rows against the fp64 oracle, two launches bitwise -- one oracle per form / body / dtype member
(the Rubin module's matrix trimmed symmetrically, Rule S6), and the per-batch Q trim under the quiet body (the stored dead rows).
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
    # the width pick: 128 by default; 64 for the fp32 vector WITH a bias, and by the BUDGET rule where only the 64-wide tile holds a form (D 64 above S_kv 128)
    dom = frozenset({64, 128})
    assert c0.sm100_tile_n_pick(32, 256, has_bias=True, has_kv_mask=False, domain=dom) == 128
    assert c0.sm100_tile_n_pick(64, 128, has_bias=True, has_kv_mask=False, domain=dom) == 128  # cga2 holds one 128-wide tile
    assert c0.sm100_tile_n_pick(64, 192, has_bias=True, has_kv_mask=False, domain=dom) == 64  # 3 resident 64-wide tiles at cga1
    assert c0.sm100_tile_n_pick(64, 1024, has_bias=True, has_kv_mask=False, domain=dom) == 64  # the 3-stage ring at the 64-wide tile
    assert c0.sm100_tile_n_pick(64, 192, has_bias=False, has_kv_mask=True, domain=dom) == 128  # the mask alone has no budget
    assert c0.sm100_tile_n_pick(32, 256, has_bias=True, has_kv_mask=True, kv_mask_fp32=True, domain=dom) == 64
    # the PADDING leg (per-row kv_lengths) with a bias on the 32-wide tile: the 64-wide KV tile while its resident block holds S_kv (<= 320 at cga1;
    # MEASURED on GB200 2026-10-14: +21.8 % at S 256 f 0.75), the 128-wide tile above it (the 64-wide ring lost there); D 64 keeps the budget rule
    assert c0.sm100_tile_n_pick(32, 256, has_bias=True, has_kv_mask=False, padded=True, domain=dom) == c0.SM100_PADDED_BIAS_TILE_N == 64
    assert c0.sm100_tile_n_pick(32, 320, has_bias=True, has_kv_mask=True, padded=True, domain=dom) == 64
    assert c0.sm100_tile_n_pick(32, 384, has_bias=True, has_kv_mask=False, padded=True, domain=dom) == 128
    assert c0.sm100_tile_n_pick(32, 256, has_bias=False, has_kv_mask=False, padded=True, domain=dom) == 128  # the plain padded body: unmeasured, keeps 128
    assert c0.sm100_tile_n_pick(64, 128, has_bias=True, has_kv_mask=False, padded=True, domain=dom) == 128  # D 64 at S_kv 128: cga2's one 128-wide tile
    # the heuristics' width pick = the row's: the narrowest width whose block holds S_kv
    assert c0.sm100_bias_auto_cga(32, 256, frozenset({1})) == 1
    assert c0.sm100_bias_auto_cga(64, 128, frozenset({1, 2})) == 2
    # the D 64 flavor at ITS width (the 64-wide tile): cga1 -- the block up to 192, the 3-stage ring above it (measured over the cga-2 block, GB200 2026-10-14)
    assert c0.sm100_bias_auto_cga(64, 192, frozenset({1, 2}), tile_n=64) == 1 and c0.sm100_bias_auto_cga(64, 256, frozenset({1, 2}), tile_n=64) == 1
    assert c0.sm100_bias_mode_pick(64, 1, 256, tile_n=64) == 2 and c0.sm100_bias_stages_pick(64, 1, tile_n=64) == 3
    # the kernel's preload constant: the Rubin table as the prior (0 at the 128-wide tile)
    assert c0.sm100_bias_preload_chunks(128) == 0 and c0.sm100_bias_preload_chunks(64) == 1


def test_kernel_module_constants_and_the_not_yet_wired_backstops(_sm100_device_for_the_adapter):
    """The SM100 file's bias arms fold on the record (the module constants), the seams the file does not render yet raise at
    module load with a message naming them (the config validators accept the records; the kernel states its own envelope), and the
    d64 decode tile refuses either operand (the fix pin at the end)."""
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
    tail = _mod(seq_kv_lens_present=True, padded_tail_form=1)
    assert tail._PADDED_TAIL_PREMASK and not tail._INF_MASK_ARM and tail._INF_BODY_MASK_FLAGS == 0
    # the pre-masked tail on the BIAS body (the Rubin twin's arm): the -inf body renders its band mask at MASK_NONE (the padding arm's words fold out
    # of every kv tile; the tail tile is pre-masked in TMEM); a kv_mask is refused with either body (its keep words fold the length in already)
    btail = _mod(seq_kv_lens_present=True, padded_tail_form=1, bias_mode=2, bias_stages=2)
    assert btail._PADDED_TAIL_PREMASK and btail._BIAS_STREAMED and btail._INF_MASK_ARM and btail._INF_BODY_MASK_FLAGS == 0 and btail.CFG.MASK_FLAGS != 0
    bword = _mod(seq_kv_lens_present=True, bias_mode=2, bias_stages=2)
    assert not bword._PADDED_TAIL_PREMASK and bword._INF_BODY_MASK_FLAGS == bword.CFG.MASK_FLAGS != 0
    with pytest.raises(ValueError, match="kv_mask"):
        _mod(seq_kv_lens_present=True, padded_tail_form=1, kv_mask_form=1)
    with pytest.raises(ValueError, match="kv_mask"):
        _mod(seq_kv_lens_present=True, padded_tail_form=1, bias_mode=2, bias_stages=2, kv_mask_form=1)
    # the small tiles' ring-wait form: the BIAS body spins (MEASURED on GB200 2026-10-14: +5..+10 % on the bias cells, +8..+18 % with the kv_mask word);
    # the plain small tile and the kv_mask-only body keep the d64 flavor's sleeping form (unmeasured); never a literal at a call site
    assert res.SPIN_RING_WAITS is True and btail.SPIN_RING_WAITS is True and bword.SPIN_RING_WAITS is True
    assert plain.SPIN_RING_WAITS is False and mask.SPIN_RING_WAITS is False and tail.SPIN_RING_WAITS is False
    # the dead-sub-tile quiet body: wired on the -inf arms (a bias / a kv_mask), refused on the plain body of this file (the factory's admission)
    quiet = _mod(bias_mode=2, bias_stages=2, dead_subtile_quiet=1)
    assert quiet._DEAD_SUBTILE_QUIET and quiet._BIAS_STREAMED and not res._DEAD_SUBTILE_QUIET
    assert _mod(kv_mask_form=1, dead_subtile_quiet=1)._DEAD_SUBTILE_QUIET
    with pytest.raises(ValueError, match="-inf body"):
        _mod(dead_subtile_quiet=1)
    with pytest.raises(ValueError, match="pair item"):
        _mod(bias_mode=1, n_chunk=2)
    t64 = _mod(bias_mode=1, tile_n=64)
    assert t64.CFG.TILE_N == 64 and t64.CFG.N_BMM2_CHUNKS == 1 and t64._BIAS_GG.boxes_per_tile == 1 and t64.BIAS_PRELOAD_CHUNKS == 1
    assert t64.CFG.BIAS_RESIDENT_TILES == 5  # the 32-wide tile holds 5 x 64 = 320 kv columns resident at the 64-wide KV tile
    # The fix pin: ``_d64_decode_tile`` refuses the 128-row decode tile with a bias / kv_mask bound (the tile carries neither arm; a kv_mask-only
    # d64 graph at S_q 128 used to lower onto it and fail at build on cc 10.0): the adapter at d64, S_q 128 takes the decode tile plain and stays
    # on the prefill tile with either operand.
    for operands, decode in ((dict(), True), (dict(kv_mask=(2, 1, 1, 128)), False), (dict(bias=(1, 4, 128, 128)), False)):
        api = _api(d=64, s_q=128, s_kv=128, **operands)
        assert api.check_support() is True and api._d64_decode_tile() is decode, (operands, api.template_params())


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
    (
        "bias d64 S256 -> the 64-wide tile by the budget rule: the cga-1 3-stage STREAMED RING (measured +77.5 % over the cga-2 block, GB200 2026-10-14)",
        dict(d=64, s_q=256, s_kv=256, bias=(1, 4, 256, 256)),
        True,
        (64, 64),
        2,
        0,
        1,
    ),
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
    _fp32_bias = kw.get("kv_mask_dtype") == torch.float32 and kw.get("bias") is not None
    _budget_64 = kw.get("d", 32) == 64 and kw.get("bias") is not None and kw.get("s_kv", 256) > 128
    _want_tile_n = kw.get("tile_n", 64 if (_fp32_bias or _budget_64) else 128)
    assert params.d_flavor == flavor[0] and params.cta_mma == cta_mma and params.n_chunk == 0 and params.tile_n == _want_tile_n, (name, params)
    _want_stages = 0 if bias_mode != 2 else (3 if _want_tile_n == 64 else 2)  # the derived deepest ring: 3 stages at the 64-wide KV tile, 2 at the 128-wide
    assert params.bias_stages == _want_stages and params.s_slots == 1 and params.q_in_tmem == 0 and params.softmax_wgs == 2, (name, params)


def test_padded_tail_form_pick_and_request_on_sm100(_sm100_device_for_the_adapter, monkeypatch):
    """The pre-masked tail is reached by request or by the sweep seam on the SM100 small tiles' padded plans WITHOUT a kv_mask -- the plain body
    (default 0) and the BIAS body (the Rubin twin's arm; its knob-less default per KV tile width = ``SM100_BIAS_PADDED_TAIL_FORM_BY_TILE_N``); a
    plan with a kv_mask, a band, or one the d64 decode tile would take, is not eligible (a request there is a typed decline)."""
    from cudnn.sdpa.fwd import config_sm100 as c0

    assert c0.sm100_padded_tail_form_pick(32, padded=True, has_bias=False, has_kv_mask=False) == 0
    assert c0.sm100_padded_tail_form_pick(32, padded=True, has_bias=False, has_kv_mask=False, request=1) == 1
    # the BIAS body: a request and the seam are honoured at either width; the knob-less default reads the per-width table; a kv_mask / a band: 0
    assert c0.sm100_padded_tail_form_pick(32, padded=True, has_bias=True, has_kv_mask=False, request=1) == 1
    assert c0.sm100_padded_tail_form_pick(64, padded=True, has_bias=True, has_kv_mask=False, tile_n=64, request=1) == 1
    for tn in (64, 128):
        assert c0.sm100_padded_tail_form_pick(32, padded=True, has_bias=True, has_kv_mask=False, tile_n=tn) == c0.SM100_BIAS_PADDED_TAIL_FORM_BY_TILE_N[tn]
        assert c0.sm100_padded_tail_form_pick(64, padded=True, has_bias=True, has_kv_mask=False, tile_n=tn) == c0.SM100_BIAS_PADDED_TAIL_FORM_BY_TILE_N[tn]
    assert c0.sm100_padded_tail_form_pick(32, padded=True, has_bias=True, has_kv_mask=True, request=1) == 0
    assert c0.sm100_padded_tail_form_pick(32, padded=True, has_bias=True, has_kv_mask=False, causal=True, request=1) == 0
    assert c0.sm100_padded_tail_form_pick(32, padded=False, has_bias=True, has_kv_mask=False, request=1) == 0
    monkeypatch.setattr(c0, "_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS", 1)
    assert c0.sm100_padded_tail_form_pick(32, padded=True, has_bias=True, has_kv_mask=False, tile_n=128) == 1
    assert c0.sm100_padded_tail_form_pick(32, padded=True, has_bias=True, has_kv_mask=True) == 0, "the seam never reaches the kv_mask plan"
    monkeypatch.setattr(c0, "_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS", None)
    api = _api(s_q=256, s_kv=256, seq_kv_lens_present=True, padded_tail_form=1)
    assert api.check_support() is True and api.template_params().padded_tail_form == 1
    api0 = _api(s_q=256, s_kv=256, seq_kv_lens_present=True)
    assert api0.check_support() is True and api0.template_params().padded_tail_form == 0
    # the bias body through the adapter: the request honoured (the resident block at S 256), the knob-less plan at its width's default
    apib = _api(s_q=256, s_kv=256, seq_kv_lens_present=True, bias=(1, 4, 256, 256), padded_tail_form=1)
    assert apib.check_support() is True and apib.template_params().padded_tail_form == 1 and apib.template_params().bias_mode != 0
    apib0 = _api(s_q=256, s_kv=256, seq_kv_lens_present=True, bias=(1, 4, 256, 256))
    assert apib0.check_support() is True
    assert apib0.template_params().padded_tail_form == c0.SM100_BIAS_PADDED_TAIL_FORM_BY_TILE_N[apib0.template_params().tile_n]
    with pytest.raises((NotImplementedError, ValueError)):
        _api(s_q=256, s_kv=256, seq_kv_lens_present=True, bias=(1, 4, 256, 256), kv_mask=(2, 1, 1, 256), padded_tail_form=1).check_support()
    with pytest.raises((NotImplementedError, ValueError)):
        _api(d=64, s_q=128, s_kv=128, seq_kv_lens_present=True, padded_tail_form=1).check_support()  # the d64 decode tile's plan


def _paged_d64_api(**kw):
    """A paged (64, 64) plan on the standalone adapter: K / V as page pools (pages, H_kv, page, D), the padding mask on (paged KV requires it)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    page, pages = 128, 4
    q, o = (_t7._desc((2, 4, 256, 64), torch.bfloat16, n) for n in ("q", "o"))
    k, v = (_t7._desc((pages, 4, page, 64), torch.bfloat16, n) for n in ("k", "v"))
    return SdpaFwdDslSm100(q, k, v, o, None, seq_kv_lens_present=True, paged_page_size=page, paged_max_seq_len_kv=pages * page, **kw)


def test_padded_tail_form_request_declines_typed_on_the_d64_flavors_other_legs_on_sm100(_sm100_device_for_the_adapter):
    """``check_support`` asks the SM100 branch of ``_padded_tail_form_eligible`` for an explicit ``padded_tail_form=`` on every leg of the
    (64, 64) flavor and BEFORE it resolves the GQA group: a PackGQA, paged or split-KV padded d64 plan is declined TYPED there (the
    small-tile factory refuses those legs, so an admitted request failed at compile() instead) -- RED on the tree whose predicate reached
    ``_d64_decode_tile``'s ``h_q // h_kv`` with both still None (a raw TypeError) and admitted the paged request; the knob-less paged plan
    keeps its plain d64 record (``padded_tail_form`` 0, the d64 factory accepts it), so no served rendering moves."""
    from cudnn.sdpa.fwd import config_sm100 as c0

    with pytest.raises(NotImplementedError, match="padded_tail_form=1"):
        _api(d=64, s_q=256, s_kv=256, h=8, h_kv=2, pack_gqa=True, seq_kv_lens_present=True, padded_tail_form=1).check_support()
    with pytest.raises(NotImplementedError, match="padded_tail_form=1"):
        _paged_d64_api(padded_tail_form=1).check_support()
    with pytest.raises((NotImplementedError, ValueError)):
        _api(d=64, s_q=256, s_kv=256, seq_kv_lens_present=True, split_kv=2, padded_tail_form=1).check_support()
    api = _paged_d64_api()
    assert api.check_support() is True
    params = api.template_params()
    assert params.padded_tail_form == 0 and params.paged_kv is True and params.d_flavor == 64, params
    c0.make_cfg_d64(params)  # the plain d64 record: accepted by the factory (the small-tile one would refuse a paged record)


def test_dead_subtile_quiet_pick_and_request_on_the_inf_body_on_sm100(_sm100_device_for_the_adapter, monkeypatch):
    """The dead-sub-tile quiet body is served on the SM100 small tiles' -inf body (a bias or a kv_mask) by request or by the sweep seam when the
    plan's S_q leaves a dead sub-tile; the knob-less default is 0 until measured; the plain body (its quiet form is not wired here) and a plan
    without a dead sub-tile are 0 (a request on the plain body is a typed decline)."""
    from cudnn.sdpa.fwd import config_sm100 as c0

    assert c0.sm100_dead_subtile_quiet_pick(32, has_bias=True, has_kv_mask=False, s_q=384) == c0.SM100_INF_BODY_DEAD_SUBTILE_QUIET_DEFAULT == 1
    assert (
        c0.sm100_dead_subtile_quiet_pick(32, has_bias=True, has_kv_mask=False, s_q=384, padded=True, tile_n=128) == 0
    ), "the 128-wide padded body keeps the full body (measured a loss)"
    assert (
        c0.sm100_dead_subtile_quiet_pick(64, has_bias=True, has_kv_mask=False, s_q=384, padded=True, tile_n=64) == 1
    ), "the 64-wide padded body: measured a win"
    assert c0.sm100_dead_subtile_quiet_pick(32, has_bias=True, has_kv_mask=False, s_q=384, padded=True, tile_n=128, request=1) == 1
    assert c0.sm100_dead_subtile_quiet_pick(32, has_bias=True, has_kv_mask=False, s_q=384, request=1) == 1
    assert c0.sm100_dead_subtile_quiet_pick(64, has_bias=False, has_kv_mask=True, s_q=640, request=1) == 1
    assert c0.sm100_dead_subtile_quiet_pick(32, has_bias=False, has_kv_mask=False, s_q=384, request=1) == 0  # the plain body: not wired here
    assert c0.sm100_dead_subtile_quiet_pick(128, has_bias=True, has_kv_mask=False, s_q=384, request=1) == 0
    monkeypatch.setattr(c0, "_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS", 1)
    assert c0.sm100_dead_subtile_quiet_pick(32, has_bias=True, has_kv_mask=False, s_q=384) == 1
    assert c0.sm100_dead_subtile_quiet_pick(64, has_bias=True, has_kv_mask=False, s_q=640) == 1
    assert c0.sm100_dead_subtile_quiet_pick(32, has_bias=True, has_kv_mask=False, s_q=256) == 0, "no dead sub-tile at a 256 multiple"
    assert (
        c0.sm100_dead_subtile_quiet_pick(32, has_bias=True, has_kv_mask=False, s_q=200) == 0
    ), "S_q % 256 = 200 > 128: sub-tile 1 (rows 128..255) holds live rows"
    assert c0.sm100_dead_subtile_quiet_pick(32, has_bias=True, has_kv_mask=False, s_q=100) == 1, "rows 128..255 of the only item are dead"
    assert c0.sm100_dead_subtile_quiet_pick(32, has_bias=True, has_kv_mask=False, s_q=400) == 0, "S_q % 256 = 144 > 128: both sub-tiles live"
    assert (
        c0.sm100_dead_subtile_quiet_pick(64, has_bias=True, has_kv_mask=False, s_q=300, cga=2) == 1
    ), "cga2: the cluster's 512 rows, 300 leaves dead sub-tiles"
    assert c0.sm100_dead_subtile_quiet_pick(32, has_bias=False, has_kv_mask=False, s_q=384) == 0, "the seam never reaches the plain body"
    api = _api(s_q=384, s_kv=384, bias=(1, 4, 384, 384))
    assert api.check_support() is True and api.template_params().dead_subtile_quiet == 1 and api.template_params().bias_mode != 0
    monkeypatch.setattr(c0, "_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS", None)
    api0 = _api(s_q=384, s_kv=384, bias=(1, 4, 384, 384))
    assert api0.check_support() is True and api0.template_params().dead_subtile_quiet == c0.SM100_INF_BODY_DEAD_SUBTILE_QUIET_DEFAULT == 1
    apip = _api(s_q=384, s_kv=384, seq_kv_lens_present=True, bias=(1, 4, 384, 384))
    assert (
        apip.check_support() is True and apip.template_params().tile_n == 128 and apip.template_params().dead_subtile_quiet == 0
    ), "the 128-wide padded plan keeps the full body"
    apip64 = _api(d=64, s_q=384, s_kv=384, seq_kv_lens_present=True, bias=(1, 4, 384, 384))
    assert (
        apip64.check_support() is True and apip64.template_params().tile_n == 64 and apip64.template_params().dead_subtile_quiet == 1
    ), "the 64-wide padded plan takes the quiet body"
    api256 = _api(s_q=256, s_kv=256, bias=(1, 4, 256, 256))
    assert api256.check_support() is True and api256.template_params().dead_subtile_quiet == 0, "no dead sub-tile at a 256 multiple"
    apir = _api(s_q=384, s_kv=384, bias=(1, 4, 384, 384), dead_subtile_quiet=1)
    assert apir.check_support() is True and apir.template_params().dead_subtile_quiet == 1
    apim = _api(s_q=640, s_kv=640, kv_mask=(2, 1, 1, 640), dead_subtile_quiet=1)
    assert apim.check_support() is True and apim.template_params().dead_subtile_quiet == 1 and apim.template_params().kv_mask_form != 0
    with pytest.raises((NotImplementedError, ValueError)):
        _api(s_q=384, s_kv=384, dead_subtile_quiet=1).check_support()  # the plain body: a typed decline on this line


def test_dead_subtile_quiet_pick_sees_the_width_a_bias_plan_runs_at_on_sm100(_sm100_device_for_the_adapter, monkeypatch):
    """The dead-sub-tile rows rule (a cluster tile of 2 x tile_m x cga rows) is evaluated at the width the plan RUNS at -- the ``cta_mma``
    arm of ``template_params``: a bias plan without a width request takes the heuristics' width (D 64 at S_kv <= 128 is the cga-2 resident
    block), a kv_mask-only plan its request or the small tiles' cga1.  A knob-less D 64 bias plan at S_kv 128 and S_q 256 therefore takes
    the quiet body (the cluster's 512 rows leave CTA 1 of the pair without a live row), as the pick at cga 2 says and as the explicit
    width request already did; at S_q 300 too (sub-tile 1 of CTA 1 dead); at S_q 512 the full body (every sub-tile live).  The
    kv_mask-only plan keeps the small tiles' cga1 and its own rule (no dead sub-tile at a 256 multiple, one at S_q 300)."""
    from cudnn.sdpa.fwd import config_sm100 as c0

    monkeypatch.setattr(c0, "_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS", None)
    assert c0.sm100_dead_subtile_quiet_pick(64, has_bias=True, has_kv_mask=False, s_q=256, cga=2) == 1
    assert c0.sm100_dead_subtile_quiet_pick(64, has_bias=True, has_kv_mask=False, s_q=256, cga=1) == 0
    api = _api(d=64, s_q=256, s_kv=128, bias=(1, 4, 256, 128))
    assert api.check_support() is True
    p = api.template_params()
    assert (p.cta_mma, p.tile_n, p.bias_mode) == (2, 128, 1), p
    assert p.dead_subtile_quiet == 1, "the pick is evaluated at the cga-2 width the plan runs at (the cta_mma arm), not at the small tiles' cga1"
    apir = _api(d=64, s_q=256, s_kv=128, bias=(1, 4, 256, 128), cga=2)
    assert apir.check_support() is True and apir.template_params().dead_subtile_quiet == 1, "the explicit width request agrees"
    api300 = _api(d=64, s_q=300, s_kv=128, bias=(1, 4, 300, 128))
    assert api300.check_support() is True and api300.template_params().dead_subtile_quiet == 1
    api512 = _api(d=64, s_q=512, s_kv=128, bias=(1, 4, 512, 128))
    assert api512.check_support() is True
    assert (api512.template_params().cta_mma, api512.template_params().dead_subtile_quiet) == (2, 0), "S_q 512 fills the cluster tile"
    apim = _api(d=64, s_q=256, s_kv=128, kv_mask=(2, 1, 1, 128))
    assert apim.check_support() is True
    pm = apim.template_params()
    assert (pm.cta_mma, pm.kv_mask_form, pm.dead_subtile_quiet) == (1, 1, 0), "a kv_mask-only plan keeps cga1: no dead sub-tile at a 256 multiple"
    apim300 = _api(d=64, s_q=300, s_kv=128, kv_mask=(2, 1, 1, 128))
    assert apim300.check_support() is True
    assert (apim300.template_params().cta_mma, apim300.template_params().dead_subtile_quiet) == (1, 1), "sub-tile 1 of the second item dead at cga1"


# --------------------------------------------------------------------------- tier 3: the GPU cells (a cc 10.0 / 10.3 device)

# The Rubin module's degenerate-input matrix INSIDE the SM100 resident budget (S_kv <= 256 at the 32-wide tile, <= 128 at the 64-wide one):
# the KV ring sweep n_kv 1 / 2 with B*H > 1, ONE CTA / one tile in the whole launch (S_kv 128 and 64), a tail tile with a padded bias row
# stride (217 / 100), GQA, a zero-length KV batch (an EMPTY work item: no bias barrier is touched), a zero-key batch WITH a sink and the
# padded-Q trim, the bias in its four broadcast forms, the magnitude cells, the 64-wide tile at cga2 (one tile).  (D 24 rides the 32-wide
# tile by the zero-fill contract, bitwise-pinned on the plain body in test_sdpa_small_d_sm100.py; the bias arm does not read D.)
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
    "dense_one_tile_64",
    "tail_217_padded_stride_bias",
    "causal_tail_100_padded_stride_bias",
]
# Above the SM100 resident budget (the Rubin module streams or pairs these): a typed decline of the row on a real graph -- no FROST plan.
# The one-batch-entry item ABOVE the resident budget at the 32-wide tile: the 2-stage STREAMED ring (n_kv 3 = one stage past the ring, 4 = two
# wraps, 5, 8 with a tail; the causal band; a padded zero-length batch on the ring: an empty item touches no ring row).  One (form x ring class)
# each, with and without the word where the body differs; the magnitude / head-broadcast forms are pinned on the resident arm (the ring feeds
# the same FFMA).
_RING_CASES = {
    "ring_dense_nkv3_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384),
    "ring_dense_nkv4_512": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512),
    "ring_causal_nkv5_640": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640, causal=True),
    # S_kv 904 (not 900): a COMPACT bias needs S_kv % 8 == 0 for its TMA row stride (the row declines 900 typed; the Rubin module runs 900 on the mask-only arm)
    "ring_padded_nkv8_tail_904_zero_batch": dict(b=3, h=2, h_kv=2, s_q=904, s_kv=904, kv_lens=[904, 0, 500]),
    # the D 64 flavor's knob-less default above its cga-1 resident budget: the 64-wide tile's 3-stage ring at cga1 (the measured preference over the cga-2 block)
    "ring_d64_nkv4_256_t64": dict(b=2, h=2, h_kv=2, s_q=256, s_kv=256, d=64),
}
# No form at any width of the flavor's domain: the 64-wide tile's cga2 holds one tile and no ring; cga1 none -- a typed decline on real graphs.
_ABOVE_BUDGET = ["d64_dense_256_cga2"]


def _expected_cga(case):
    c = _t7._CASES[case]
    return 2 if c.get("d", 32) == 64 else 1  # the 64-wide tile's bias block lives at cga2 on this line; the 32-wide tile at its cga1


def _run(
    q,
    k,
    v,
    *,
    scale,
    bias=None,
    kv_mask=None,
    sdpa_kwargs=None,
    seq_len_kv=None,
    seq_len_q=None,
    sink=None,
    stats=True,
    cga=None,
    tile_n=None,
    two_launch_assert=True,
):
    """The Rubin module's ``_run`` against the SM100 row: build the graph, pin the row's unsplit plan (the heuristics' pick; a cga / tile_n
    request through the public autotune record), launch TWICE on sentinel-filled outputs (the two-launch delta) and return O, Stats and
    the plan's knobs.  ``two_launch_assert=False`` leaves the two-launch identity to the caller (the first launch rides along as
    ``o_first`` / ``stats_first``), so a cell can report a NaN residue as what it is before the identity check."""
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
    if two_launch_assert:
        assert torch.equal(o1, o2) and (st1 is None or torch.equal(st1, st2)), "two-launch delta on O / Stats"
    return dict(o=o_gpu, stats=stats_gpu, knobs=SimpleNamespace(cga=picked_cga, tile_n=picked_tile_n), names=names, o_first=o1, stats_first=st1)


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


# (case, dtype) pairs: every in-budget case in bf16; the fp16 member on the dense base and on the fp16 O cast of keyless rows under the sink
# and the padded-Q trim (the same two cells as the Rubin module).
_BIAS_CELLS = [(c, torch.bfloat16) for c in _IN_BUDGET] + [("dense_nkv2", torch.float16), ("causal_sink_padded_q_trim", torch.float16)]


@requires_sm100_line
@pytest.mark.parametrize("case,dtype", _BIAS_CELLS, ids=[f"{c}-{'bf16' if d == torch.bfloat16 else 'fp16'}" for c, d in _BIAS_CELLS])
def test_bias_matches_the_fp64_oracle_on_sm100(case, dtype):
    """The additive bias over the degenerate-input matrix inside the SM100 resident budget (bf16, the fp16 member twice): the plan is the row's unsplit plan
    at the budget's width (cga1 on the 32-wide tile, cga2 on the 64-wide one), O and Stats against the fp64 oracle, keyless rows exactly
    0 / -inf, two launches bitwise; a zero-length KV batch on the output directly."""
    q, k, v, kw, ref = _t7._case_inputs(case, dtype)
    kw.pop("cga", None)  # the heuristics' width on this line (the Rubin cells' cga requests are the Rubin budget's)
    out = _run(q, k, v, **kw)
    # the knob-less width: 128, except the plain PADDING leg with a bias on the 32-wide tile inside the 64-wide resident budget (per-row kv_lengths:
    # padded_kv_zero_batch at S_kv 256 -> the 64-wide tile at cga1; MEASURED on GB200 2026-10-14).  The GQA 8 / 4 + sink + padded-Q-trim composition
    # (causal_sink_padded_q_trim) keeps the 128-wide tile on the graph path (recorded on the cc 10.0 dev box the same day); its width is not claimed by the rule.
    want_tn = {"padded_kv_zero_batch": 64}.get(case, 128)
    assert out["knobs"].cga == _expected_cga(case) and out["knobs"].tile_n == want_tn, (case, out["knobs"], want_tn)
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


# The HW-max kv_mask-only body (the dense base, the zero-length batch, the d64 graph at S_q 128 -- the cell that used to lower onto the decode
# tile and fail at build) and the software-tree bias + mask body (GQA under the band; keyless rows under the sink and the padded-Q trim).
_KV_MASK_CELLS = [
    (False, "dense_nkv2"),
    (False, "padded_kv_zero_batch"),
    (False, "d64_dense_128"),
    (True, "causal_gqa_8_2"),
    (True, "causal_sink_padded_q_trim"),
]


@requires_sm100_line
@pytest.mark.parametrize("with_bias, case", _KV_MASK_CELLS, ids=[f"{'mask_and_bias' if w else 'mask_only'}-{c}" for w, c in _KV_MASK_CELLS])
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
@pytest.mark.parametrize("form", _t7._SIGNED_SCALE_FORMS)
@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive_scale", "negative_scale"])
def test_signed_attn_scale_keeps_the_additive_arms_sign_on_sm100(sign, form):
    """attn_scale = +0.125 and -0.125 on the SM100 row's resident bias and on its additive kv_mask vector, on the public graph route (the
    review's repro: zero Q / K, S 128, D 32, +2 / -2 by key half, V +1 / -1): the negated plan (BMM1 negates S, |scale| in the kernel) must
    bind the additive arms' 1 / scale at the same effective magnitude -- O = tanh(2) on every lane at either sign; the positive cell is the
    control.  The Rubin twin runs the same cells on its row, and the host pin of the template itself lives in the Rubin module."""
    q, k, v, kw, ref = _t7._signed_scale_inputs(sign, form, torch.bfloat16)
    _t7._check_signed_scale(_run(q, k, v, **kw), ref)


_RING_CELLS = [
    (False, "ring_dense_nkv3_384"),
    (False, "ring_causal_nkv5_640"),
    (False, "ring_padded_nkv8_tail_904_zero_batch"),
    (False, "ring_d64_nkv4_256_t64"),
    (True, "ring_dense_nkv4_512"),
    (True, "ring_padded_nkv8_tail_904_zero_batch"),
    (True, "ring_d64_nkv4_256_t64"),
]


@requires_sm100_line
@pytest.mark.parametrize("with_mask, case", _RING_CELLS, ids=[f"{'bias_and_mask' if m else 'bias'}-{c}" for m, c in _RING_CELLS])
def test_streamed_ring_matches_the_fp64_oracle_on_sm100(case, with_mask):
    """The 2-stage streamed bias ring of the one-batch-entry item at the 32-wide tile (S_kv above the resident 256): the ring wrap axis
    n_kv 3 / 4 / 5 / 8 (with a tail and a zero-length batch), the causal band, with and without the bit-word mask; the plan's record carries
    bias_mode 2 at cga1; O and Stats against the fp64 oracle, two launches bitwise."""
    q, k, v, kw, ref = _t7._case_inputs(case, torch.bfloat16, kv_mask_p=0.3 if with_mask else None, table=_RING_CASES)
    kw.pop("cga", None)
    out = _run(q, k, v, **kw)
    # the 32-wide tile rings at the 128-wide KV tile; the D 64 flavor's ring is the 64-wide tile's (its budget width), cga1 on both
    assert out["knobs"].cga == 1 and out["knobs"].tile_n == (64 if _RING_CASES[case].get("d", 32) == 64 else 128), (case, out["knobs"])
    _t7._check(out, ref)


# The 64-wide KV tile (the TILE_N knob through the public autotune record): the resident block holds 5 tiles (S_kv 320) at the 32-wide tile and
# 3 (192) at the 64-wide tile's cga1, the ring 3 stages above; the one-chunk softmax body; the fp32 vector WITH a bias is served HERE only.
_T64_CASES = {
    "t64_dense_nkv2_256": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256),
    "t64_causal_320": dict(b=2, h=2, h_kv=2, s_q=320, s_kv=320, causal=True),
    "t64_ring_dense_640": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640),
    "t64_ring_padded_1024_zero_batch": dict(b=3, h=2, h_kv=2, s_q=1024, s_kv=1024, kv_lens=[1024, 0, 700]),
    "t64_d64_dense_192_cga1": dict(b=2, h=2, h_kv=2, s_q=192, s_kv=192, d=64),  # 3 resident tiles at cga1
    "t64_d64_causal_tail_200": dict(
        b=2, h=2, h_kv=2, s_q=200, s_kv=200, d=64, causal=True
    ),  # 4 tiles: the cga-1 ring (the measured preference over the cga-2 block)
}


# each (resident / ring / d64 cga1) x (bias / both) class once
_T64_CELLS = [
    (False, "t64_dense_nkv2_256"),
    (False, "t64_ring_padded_1024_zero_batch"),
    (False, "t64_d64_causal_tail_200"),
    (True, "t64_causal_320"),
    (True, "t64_ring_dense_640"),
    (True, "t64_d64_dense_192_cga1"),
]


@requires_sm100_line
@pytest.mark.parametrize("with_mask, case", _T64_CELLS, ids=[f"{'bias_and_mask' if m else 'bias'}-{c}" for m, c in _T64_CELLS])
def test_tile_n_64_matches_the_fp64_oracle_on_sm100(case, with_mask):
    """The 64-wide KV tile on the SM100 small tiles (TILE_N=64 through the autotune record): the resident block (S_kv 256 / 320 at D 32, 192 at D 64
    cga1) and the 3-stage ring above it (640, 1024 with a zero-length batch), the one-chunk body, with and without the bit-word mask."""
    q, k, v, kw, ref = _t7._case_inputs(case, torch.bfloat16, kv_mask_p=0.3 if with_mask else None, table=_T64_CASES)
    kw.pop("cga", None)
    out = _run(q, k, v, tile_n=64, **kw)
    # the width at the 64-wide tile = the budget pick's own answer (the row's, the heuristics' and the adapter's one arithmetic): cga1 while the
    # resident block holds S_kv there (D 32 always; D 64 up to 3 tiles = 192) and, at D 64, the cga-1 ring above it (D 64 S 200: measured faster than
    # the cga-2 4-tile block on GB200, 2026-10-14) -- cga 1 on every case of this table
    from cudnn.sdpa.fwd import config_sm100 as c0

    c = _T64_CASES[case]
    d = c.get("d", 32)
    want_cga = c0.sm100_bias_auto_cga(d, c["s_kv"], c0.SM100_SMALL_D_CGAS[(d, d)], tile_n=64)
    assert want_cga == 1, (case, want_cga)
    assert out["knobs"].tile_n == 64 and out["knobs"].cga == want_cga, (case, out["knobs"])
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


# The PLAIN padded body's pre-masked tail is NOT this line's default (``SM100_PADDED_TAIL_FORM_DEFAULT`` = 0; the public ``padded_tail_form=``
# kwarg reaches it, the sweep seam in-process here): one cell, a tail tile with a zero-length batch at the 128-wide tile -- against the oracle
# and BITWISE against the legacy form (the same values: the premask's -inf columns are the masked body's; the manual max over them is the
# software max over the masked chunks).
_TAIL_CASES = {
    "tail_padded_217_zero_batch": dict(b=3, h=2, h_kv=2, s_q=217, s_kv=217, kv_lens=[217, 0, 100]),
}


@requires_sm100_line
def test_pre_masked_tail_matches_the_oracle_and_the_legacy_body_bitwise_on_sm100(monkeypatch):
    from cudnn.sdpa.fwd import config_sm100 as c0

    case, tile_n = "tail_padded_217_zero_batch", 128
    q, k, v, kw, ref = _t7._case_inputs(case, torch.bfloat16, table=_TAIL_CASES)
    kw.pop("cga", None)
    kw["bias"] = None
    ref = _t7._ref_of(_TAIL_CASES[case], q, k, v, kw)
    monkeypatch.setattr(c0, "_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS", 0)
    legacy = _run(q, k, v, tile_n=tile_n, **kw)
    monkeypatch.setattr(c0, "_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS", 1)
    pre = _run(q, k, v, tile_n=tile_n, **kw)
    _t7._check(pre, ref)
    assert torch.equal(pre["o"], legacy["o"]) and torch.equal(pre["stats"], legacy["stats"]), (case, tile_n)


# The BIAS body's pre-masked tail (the Rubin twin's arm, ledger A44; the sweep seam in-process): a padded bias row at D 32 / D 64 (the resident block,
# the streamed ring across a wrap, a zero-length batch), both widths where the bias form fits -- against the oracle and BITWISE against the word body
# (the same -inf scores: the pre-masked columns are the band words' columns, the bias FFMA reads -inf through, the software max and the exp2 see the
# same values; the FFMA order on the live columns is unchanged).  D 64 at the 128-wide tile holds no bias form above S_kv 128 (declined typed): 64 only.
# S_kv and every row length at a multiple of 8 (the COMPACT bias TMA's 16-byte rule: S 217 / 385 / 300 are typed declines of the row with a compact
# bias -- the plain tail cases above carry them bias-less) and off the tile multiple at the width under test (a tail tile with 8..120 valid columns).
_BIAS_TAIL_CASES = {
    "bias_tail_216_zero_batch": dict(b=3, h=2, h_kv=2, s_q=216, s_kv=216, kv_lens=[216, 0, 104]),  # the resident block at both widths; a zero-length batch
    "bias_tail_304_d64": dict(b=2, h=2, h_kv=2, s_q=256, s_kv=304, d=64, kv_lens=[304, 128]),  # the 64-wide cga-1 ring; one row at the tile multiple (no tail)
    "bias_tail_392": dict(b=2, h=4, h_kv=4, s_q=392, s_kv=392, kv_lens=[392, 200]),  # the ring at both widths (2 st at 128, 3 st at 64); 8 valid tail columns
    "bias_ring_tail_904_zero_batch": dict(b=3, h=2, h_kv=2, s_q=904, s_kv=904, kv_lens=[904, 0, 500]),  # the 2-stage ring across a wrap; a zero-length batch
}
# The bias body's form 1 is the DEFAULT at both widths on this line: the d32 default (the padded bias leg at D 32 picks 64 while the 64-wide
# block holds S_kv), the d64 default (the budget rule), the ring default at 128, and the resident block at 128 through the public TILE_N knob.
_BIAS_TAIL_PAIRS = [
    ("bias_tail_216_zero_batch", 128),
    ("bias_tail_304_d64", 64),
    ("bias_tail_392", 64),
    ("bias_ring_tail_904_zero_batch", 128),
]


@requires_sm100_line
@pytest.mark.parametrize("case,tile_n", _BIAS_TAIL_PAIRS, ids=[f"{c}-t{t}" for c, t in _BIAS_TAIL_PAIRS])
def test_pre_masked_tail_on_the_bias_body_matches_the_oracle_and_the_word_body_bitwise_on_sm100(case, tile_n, monkeypatch):
    from cudnn.sdpa.fwd import config_sm100 as c0

    q, k, v, kw, ref = _t7._case_inputs(case, torch.bfloat16, table=_BIAS_TAIL_CASES)
    kw.pop("cga", None)
    assert kw["bias"] is not None
    monkeypatch.setattr(c0, "_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS", 0)
    word = _run(q, k, v, tile_n=tile_n, **kw)
    monkeypatch.setattr(c0, "_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS", 1)
    pre = _run(q, k, v, tile_n=tile_n, **kw)
    assert word["knobs"].tile_n == pre["knobs"].tile_n == tile_n, (case, word["knobs"], pre["knobs"])
    _t7._check(pre, ref)
    assert torch.equal(pre["o"], word["o"]) and torch.equal(pre["stats"], word["stats"]), (case, tile_n)


# The dead-sub-tile QUIET body on the -inf arms (the sweep seam in-process): a dead sub-tile 1 (S_q % 256 in 1..128) on the resident block, the
# streamed rings at both widths, the D 64 cga-1 ring at the losing cell's shape, with the kv_mask word and with the pre-masked padded tail,
# and the cga-2 RESIDENT block (D 64 at S_kv <= 128: the heuristics' width, so the one-batch-entry item's cluster tile is 512 rows and the
# dead-sub-tile rule is evaluated at cga 2) -- against the oracle and BITWISE against the full body (the dead rows are never stored; every
# live row's O / Stats come from the same body).
_QUIET_CASES = {
    "quiet_bias_384": dict(b=2, h=2, h_kv=2, s_q=384, s_kv=384),  # sub-tile 1 of the 2nd item dead: the 2-stage ring at 128 / the 3-stage ring at 64
    "quiet_bias_pad_384": dict(
        b=2, h=2, h_kv=2, s_q=384, s_kv=392, kv_lens=[392, 200]
    ),  # sub-tile 1 of the 2nd item dead, with the pre-masked padded tail (both arms on)
    "quiet_bias_d64_300_cga2": dict(
        b=2, h=2, h_kv=2, s_q=300, s_kv=128, d=64
    ),  # cga2: CTA 1 of the pair keeps sub-tile 0 live (rows 256..299), its sub-tile 1 dead
    "quiet_bias_d64_256_cga2": dict(
        b=2, h=2, h_kv=2, s_q=256, s_kv=128, d=64
    ),  # cga2: CTA 1 of the pair holds NO live row (both sub-tiles quiet); the collective MMA still runs
}
_QUIET_PAIRS = [
    ("quiet_bias_384", 128, False),
    ("quiet_bias_384", 64, True),
    ("quiet_bias_pad_384", 128, False),
    ("quiet_bias_d64_300_cga2", 128, False),
    ("quiet_bias_d64_300_cga2", 128, True),
    ("quiet_bias_d64_256_cga2", 128, False),
]


@requires_sm100_line
@pytest.mark.parametrize("case,tile_n,with_mask", _QUIET_PAIRS, ids=[f"{c}-t{t}{'-mask' if m else ''}" for c, t, m in _QUIET_PAIRS])
def test_quiet_dead_sub_tile_on_the_inf_body_matches_the_oracle_and_the_full_body_bitwise_on_sm100(case, tile_n, with_mask, monkeypatch):
    from cudnn.sdpa.fwd import config_sm100 as c0

    q, k, v, kw, _ = _t7._case_inputs(case, torch.bfloat16, table=_QUIET_CASES, kv_mask_p=(0.3 if with_mask else None))
    kw.pop("cga", None)
    assert kw["bias"] is not None
    ref = _t7._ref_of(_QUIET_CASES[case], q, k, v, kw)
    if "kv_lens" in _QUIET_CASES[case]:
        monkeypatch.setattr(c0, "_PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS", 1)
    monkeypatch.setattr(c0, "_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS", 0)
    full = _run(q, k, v, tile_n=tile_n, **kw)
    monkeypatch.setattr(c0, "_DEAD_SUBTILE_QUIET_OVERRIDE_FOR_SWEEPS", 1)
    quiet = _run(q, k, v, tile_n=tile_n, **kw)
    assert full["knobs"].tile_n == quiet["knobs"].tile_n == tile_n, (case, full["knobs"], quiet["knobs"])
    want_cga = 2 if case.endswith("_cga2") else 1
    assert full["knobs"].cga == quiet["knobs"].cga == want_cga, (case, full["knobs"], quiet["knobs"])
    _t7._check(quiet, ref)
    assert torch.equal(quiet["o"], full["o"]) and torch.equal(quiet["stats"], full["stats"]), (case, tile_n, with_mask)


# The per-batch Q TRIM under the quiet body (the line's knob-less default on every -inf body).  Under bottom-right causal -- the one form under
# which the per-batch length reaches the softmax's dead-sub-tile decision (``_resolve_seqlen_q`` folds to S_q otherwise) -- the rows in
# [seq_len_q[b], S_q) of a dead sub-tile ARE stored, and the trim must write them as EXACT zeros through a SELECT after the scale: a zero folded
# into ``inv_sum`` is ``NaN x 0 = NaN`` wherever the quiet sub-tile's BMM2 read a NaN residue out of the S slot (the quiet body arrives
# ``mb_bmm2_ready`` without writing P).  The route a caller takes there: a bottom-right graph with the kv_mask word (a bias is not a graph under
# bottom-right), per-batch Q = KV lengths, D 32, S_q 640 (640 % 256 = 128: the pick admits the quiet body) at the 64-wide KV tile through the
# public TILE_N record (the knob-less padded kv_mask width is 128, where the pick keeps the full body); batch 1 ends at 300, so item 1 (rows
# 256..511) keeps sub-tile 0 live and sub-tile 1 (rows 384..511) dead AND stored.
_TRIM_CASES = {
    "mask_br_640_q300": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640, bottom_right=True, q_lens=[640, 300], kv_lens=[640, 300]),
}
_TRIM_DEAD_ROWS = (1, 384, 512)  # (batch, lo, hi): the quiet sub-tile's STORED rows


@requires_sm100_line
def test_quiet_body_under_per_batch_q_lengths_stores_zero_on_trimmed_rows(monkeypatch):
    """RED on the tree whose trim folds its zero into the scale, GREEN with the select after it.  Asserted most specific first -- the quiet
    sub-tile's stored rows finite and exactly +0, every trimmed row 0 / -inf, the whole tensor against the fp64 oracle, two launches bitwise --
    so a NaN residue is reported as what it is; the plan is the row's unsplit plan at the 64-wide tile and the pick's answer for it is the
    quiet body (the spy on the pick)."""
    from cudnn.sdpa.fwd import api_dsl

    calls = []
    real = api_dsl._sm100_dead_subtile_quiet_pick

    def spy(*a, **kw):
        r = real(*a, **kw)
        calls.append((int(kw.get("tile_n", 128)), int(r)))
        return r

    monkeypatch.setattr(api_dsl, "_sm100_dead_subtile_quiet_pick", spy)
    case = "mask_br_640_q300"
    c = _TRIM_CASES[case]
    q, k, v, kw, _ = _t7._case_inputs(case, torch.bfloat16, kv_mask_p=0.3, table=_TRIM_CASES)
    kw["bias"] = None  # bottom-right causal + a bias is not a graph: the kv_mask word alone carries the -inf body
    kw.pop("cga", None)
    ref = _t7._ref_of(c, q, k, v, kw)
    out = _run(q, k, v, tile_n=64, two_launch_assert=False, **kw)
    assert out["knobs"].tile_n == 64 and out["knobs"].cga == 1, out["knobs"]
    assert (64, 1) in calls, f"the plan did not take the quiet body: {calls}"
    o, st = out["o"], out["stats"].squeeze(-1)
    b, lo, hi = _TRIM_DEAD_ROWS
    rows = o[b, :, lo:hi, :]
    n_bad = int((~torch.isfinite(rows)).sum())
    assert n_bad == 0, f"{n_bad} non-finite elements in the quiet sub-tile's stored rows O[{b}, :, {lo}:{hi}] (a residue reached the store)"
    assert bool((rows.view(torch.int16) == 0).all()), f"the quiet sub-tile's stored rows O[{b}, :, {lo}:{hi}] are not exactly +0"
    q_len = c["q_lens"][b]
    assert bool((o[b, :, q_len:, :].view(torch.int16) == 0).all()) and bool(torch.isneginf(st[b, :, q_len:]).all()), "the trimmed rows: O = +0, Stats = -inf"
    _t7._check(out, ref)
    assert torch.equal(out["o"], out["o_first"]) and torch.equal(out["stats"], out["stats_first"]), "two-launch delta on O / Stats"


@requires_sm100_line
@pytest.mark.parametrize("case", _ABOVE_BUDGET)
def test_d64_bias_above_128_lands_on_the_64_wide_tile_by_the_budget_rule(case):
    """A D 64 bias graph above S_kv 128 fits no form at the 128-wide tile (cga1 holds no block, cga2 one tile and no ring): the knob-less plan
    takes the 64-wide KV tile by the budget rule -- tile_n 64 -- at cga1 on its 3-stage STREAMED RING (the measured preference over the cga-2 resident
    block: +77.5 % on GB200, 2026-10-14) and matches the oracle (the Rubin cell ran the same graph at the 128-wide tile's cga2 on its 320 KiB budget)."""
    q, k, v, kw, ref = _t7._case_inputs(case, torch.bfloat16)
    kw.pop("cga", None)  # the Rubin cell's cga2 request is the Rubin budget's; the knob-less pick is the point here
    out = _run(q, k, v, **kw)
    # the pick at the 64-wide tile: cga1 whenever a form fits there (S_kv 256 = the 3-stage ring; the cga-2 four-tile block measured 1.8x slower)
    assert out["knobs"].tile_n == 64 and out["knobs"].cga == 1, (case, out["knobs"])
    _t7._check(out, ref)


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
