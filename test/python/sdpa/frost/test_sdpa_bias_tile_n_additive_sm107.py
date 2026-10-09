# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The second half of the Rubin small-tile bias / kv_mask arm (``sm107/prefill_d128_f16.py``; the first half is
``test_sdpa_bias_sm107.py``, whose helpers this module reuses): the TILE_N = 64 renderings as REAL cells, the ADDITIVE per-key
``kv_mask`` vector, and the strided outer axes.

* **TILE_N 64** (``TemplateParams.tile_n``; the ``TILE_N`` knob, per-flavor domain ``Capabilities.tile_ns_by_d_shape`` =
  ``config_sm107.SM107_TILE_NS_BY_D_SHAPE``): the one-chunk softmax arm the body folds to at ``N_BMM2_CHUNKS == 1`` -- the S
  accumulator 64 columns per sub-tile, P in its 32-column tail, a K/V ring tile of 64 rows, two mask words per tile, one
  bias chunk.  Trace-compiled (the host pins) and run on cc 10.7 (the GPU cells) at n_kv = 1 / 2 / 4 (the ring depth) / 6,
  plain / bias / bool mask / additive mask / causal / padded tail, d32 and d64 at cga1 / cga2, against the fp64 oracle and
  against the 128-wide rendering of the same graph (within tolerance: the online-softmax sums combine in a different order).
  TILE_N 256 is a typed decline on the row, in the adapter and in the config factory (two sub-tiles x 256 S columns + the O
  slots + STATS exceed the 576-column TMEM) -- pinned, not served; the 128 default of the heuristics is unchanged.
* **The additive kv_mask** (``kv_mask_form`` 2 / 3): a 16-bit (the Q/K/V dtype) or fp32 vector ``[1|B, 1, 1, S_kv]`` folded
  into the raw scores pre-scale like the bias (``s + v * inv_scale``, one packed FFMA per pair) from warp-uniform 16-byte
  granule loads -- nothing staged in SMEM, no barrier row, the bit word's lane ledger untouched.  The models' finite -1e9
  fold is preserved as a VALUE (a fully -1e9 row is the uniform average of V, not the empty-row contract); a -inf vector
  reproduces the BOOLEAN form BITWISE.  Typed declines: a 16-bit vector that is not the Q/K/V dtype, S_kv % 8 != 0, a batch
  stride that is not a 16-byte multiple, and an EXPLICIT TILE_N 128 for the fp32 vector WITH a bias (that rendering spills on the
  softmax role -- the measured counts are the pins below; the heuristics and the adapter run that form at TILE_N 64, its only served
  width, and the io-dtype vector is the fast path).
* **Strided outer axes**: the activations stay in the row's ``bshd`` envelope (Q/K/V/O BSHD-physical, a 16-byte padded head
  stride included; a (B, H, S, D)-CONTIGUOUS activation is the row's pre-existing typed decline, pinned below -- the
  ``dense_flex`` envelope is the SM100 line's, not this row's) with the arm on, a bias in a permuted outer order
  (``[B, S_q, H, S_kv]``-physical) or with a padded row stride, a kv_mask with a padded batch stride -- graph FACTS the row
  judges and the kernels address through the declared strides (TMA descriptors from the caller's 4-D strides, the granule
  loads from the batch stride): no reshape, no copy; a non-16-byte stride is a typed decline on a real graph.
"""

import math

import pytest
import torch

from frost_test_utils import SPILL_TOLERANCE, assert_no_new_spills, requires_dsl, requires_rubin, run_sass_probe, sass_probe_source
from test_sdpa_bias_sm107 import (  # noqa: F401  (the two fixtures are re-exported so they apply here too)
    _PROBE_BODY,
    _RECORD_ARMS,
    _ROW,
    _TOL_LSE,
    _TOL_O,
    _IrTensor,
    _bhsd,
    _bias_kw,
    _build,
    _caps,
    _check,
    _desc,
    _facts,
    _frost_plans,
    _mask_kw,
    _ref,
    _rubin_device_for_the_adapter,
    _sm107_target_for_metadata,
)

pytestmark = [pytest.mark.L0, requires_dsl]


# --------------------------------------------------------------------------- tier 1: CPU contract


def test_tile_n_and_kv_mask_form_constants_are_shared_by_row_adapter_and_factory():
    """ONE constant per fact (rule 8b'): the row's per-shape TILE_N domain IS ``config_sm107.SM107_TILE_NS_BY_D_SHAPE``; the kv_mask
    dtype set is the bit word plus the two additive forms; the factory renders the 64-wide tile and every kv_mask form, and
    derives the one-chunk arm's geometry (P in the 32-column tail of a 64-column S slot) and a LARGER resident block (the K/V
    ring halves in bytes) -- the budgets every row / adapter gate asks at TILE_N 128 are therefore conservative at 64."""
    import cudnn
    from cudnn.sdpa.fwd import config_sm107 as c

    caps = _caps()
    assert caps.tile_ns_by_d_shape == c.SM107_TILE_NS_BY_D_SHAPE == (((32, 32), frozenset({64, 128})), ((64, 64), frozenset({64, 128})))
    assert caps.tile_ns == frozenset({128}), "the row-wide floor = the heuristics' default width"
    assert caps.kv_mask_dtypes == frozenset({cudnn.data_type.BOOLEAN, cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT})
    assert (c.KV_MASK_BOOL_WORD, c.KV_MASK_ADDITIVE_16, c.KV_MASK_ADDITIVE_F32) == (1, 2, 3) and c.KV_MASK_ADDITIVE_GRANULE_ELEMS == 8
    for d, cga in ((32, 1), (64, 1), (64, 2)):
        for form in (1, 2, 3):
            cfg, _ = {32: c.make_cfg_d32, 64: c.make_cfg_d64}[d](TemplateParams(dtype_qkv=2, cta_mma=cga, bias_mode=1, kv_mask_form=form, tile_n=64))
            lay = c.d128_family_tmem_layout(cfg)
            assert cfg.TILE_N == 64 and cfg.N_BMM2_CHUNKS == 1 and cfg.BMM2_CHUNK_SIZE == 64 and cfg.KV_MASK_FORM == form
            assert lay.S_OFF == (0, 64) and lay.P_OFF == (32, 96) and lay.O_OFF[0] == 128, lay
            assert cfg.BIAS_BOXES_PER_TILE == 1 and cfg.BIAS_RESIDENT_TILES * 64 > c.sm107_bias_resident_s_kv_max(
                d, cga
            ), "the 64-wide block holds more columns"
            assert c.sm107_bias_resident_s_kv_max(d, cga, tile_n=64) == cfg.BIAS_RESIDENT_TILES * 64
    assert {(d, cga): c.sm107_bias_resident_s_kv_max(d, cga, tile_n=64) for d, cga in ((32, 1), (64, 1), (64, 2))} == {(32, 1): 448, (64, 1): 320, (64, 2): 384}
    with pytest.raises(ValueError, match="kv_mask_form"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, kv_mask_form=4))
    with pytest.raises(ValueError, match="STATS"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, tile_n=256))
    with pytest.raises(ValueError, match="TILE_N"):
        c.make_cfg_d128(TemplateParams(dtype_qkv=2, cta_mma=2, tile_n=64))


def _add_kw(b=2, s_kv=256, dt=None, stride0=None):
    """An ADDITIVE kv_mask's facts: a bf16 (default) / fp32 / fp16 vector ``[b, 1, 1, s_kv]`` with an optional padded batch stride."""
    import cudnn

    dt = cudnn.data_type.BFLOAT16 if dt is None else dt
    st0 = s_kv if stride0 is None else stride0
    return dict(has_kv_mask=True, kv_mask_t=_IrTensor(dt), kv_mask_dim=(b, 1, 1, s_kv), kv_mask_stride=(st0, st0, st0, 1), kv_mask_dtype=dt)


def _bias_strided_kw(dims, strides, dt=None):
    import cudnn

    dt = cudnn.data_type.BFLOAT16 if dt is None else dt
    return dict(has_bias=True, bias_t=_IrTensor(dt), bias_dim=dims, bias_stride=strides, bias_dtype=dt)


def _cells():
    import cudnn

    DT = cudnn.data_type
    S = 256
    # (name, facts kwargs, knobs kwargs, accept?)
    return [
        # the TILE_N knob
        ("tile_n 64 on d32", dict(), dict(tile_n=64), True),
        ("tile_n 64 on d64 cga1", dict(d_qk=64, d_v=64), dict(tile_n=64, cga=1), True),
        ("tile_n 64 on d64 cga2", dict(d_qk=64, d_v=64), dict(tile_n=64, cga=2), True),
        ("tile_n 64 + bias + bool mask d32", dict(**_bias_kw(), **_mask_kw()), dict(tile_n=64), True),
        ("tile_n 64 on the d128 flavor", dict(d_qk=128, d_v=128), dict(tile_n=64), False),
        ("tile_n 256 on d32", dict(), dict(tile_n=256), False),
        ("tile_n 96 on d32", dict(), dict(tile_n=96), False),
        # the additive kv_mask
        ("additive bf16 vector", dict(**_add_kw()), None, True),
        ("additive fp32 vector", dict(**_add_kw(dt=DT.FLOAT)), None, True),
        ("additive fp16 vector on a bf16 graph", dict(**_add_kw(dt=DT.HALF)), None, False),
        ("additive fp16 vector on an fp16 graph", dict(dtype=DT.HALF, dtype_o=DT.HALF, **_add_kw(dt=DT.HALF)), None, True),
        ("additive bf16 vector + bias", dict(**_bias_kw(), **_add_kw()), None, True),
        # the fp32 vector WITH a bias: the 128-wide rendering spills on the softmax role (MEASURED, the pins below) -> served at
        # TILE_N 64 only -- the heuristics' pick and the adapter's default for the form (a knob-less probe accepts), an EXPLICIT 128
        # a typed decline (never degraded); the fp32 vector ALONE serves at 128
        ("additive fp32 vector + bias (knob-less: the heuristics pick TILE_N 64)", dict(**_bias_kw(), **_add_kw(dt=DT.FLOAT)), None, True),
        ("additive fp32 vector + bias, TILE_N 128 requested (the spilling rendering)", dict(**_bias_kw(), **_add_kw(dt=DT.FLOAT)), dict(tile_n=128), False),
        ("additive fp32 vector + bias at TILE_N 64", dict(**_bias_kw(), **_add_kw(dt=DT.FLOAT)), dict(tile_n=64), True),
        ("additive bf16 vector + bias at TILE_N 128 (the io-dtype fast path)", dict(**_bias_kw(), **_add_kw()), dict(tile_n=128), True),
        ("additive bf16 vector S_kv 1024 (no budget)", dict(s_q=1024, s_kv=1024, **_add_kw(s_kv=1024)), None, True),
        ("additive bf16 vector S_kv 100 (not whole granules)", dict(s_q=100, s_kv=100, **_add_kw(s_kv=100)), None, False),
        ("additive bf16 vector, padded 16-byte batch stride 320", dict(**_add_kw(stride0=320)), None, True),
        ("additive bf16 vector, batch stride 260 (not 16 bytes)", dict(**_add_kw(stride0=260)), None, False),
        ("additive fp32 vector, batch stride 260 (16 bytes at 4 B)", dict(**_add_kw(dt=DT.FLOAT, stride0=260)), None, True),
        ("additive vector broadcast [1, 1, 1, S]", dict(**_add_kw(b=1)), None, True),
        ("additive vector on d128", dict(d_qk=128, d_v=128, **_add_kw()), None, False),
        ("additive vector + score outputs", dict(has_score_max=True, **_add_kw()), None, False),
        # the strided outer axes of the bias (every live stride a 16-byte multiple)
        ("bias [B, S_q, H, S_kv]-physical", dict(**_bias_strided_kw((2, 4, S, S), (S * 4 * S, S, 4 * S, 1))), None, True),
        ("bias [H, S_q, S_kv] with a padded row stride 320", dict(**_bias_strided_kw((1, 4, S, S), (4 * S * 320, S * 320, 320, 1))), None, True),
        ("bias with a padded q-row stride 260 (520 B, not 16 bytes)", dict(**_bias_strided_kw((1, 4, S, S), (4 * S * 260, S * 260, 260, 1))), None, False),
        (
            "bias head stride below the row stride, padded 16 bytes",
            dict(**_bias_strided_kw((1, 4, S, S), (S * 4 * (S + 8), S + 8, 4 * (S + 8), 1))),
            None,
            True,
        ),
        ("bool kv_mask with a padded batch stride 300", dict(**_mask_kw()) | dict(kv_mask_stride=(300, 300, 300, 1)), None, True),
    ]


@pytest.mark.parametrize("name, fkw, kkw, accept", _cells(), ids=[c[0] for c in _cells()])
def test_row_accepts_and_declines_tile_n_additive_and_strided_cells(name, fkw, kkw, accept):
    """The engine row's table for the three arms: an accept is ``mismatch() is None`` on a bias shape, a decline names its
    reason (domain / dtype / granule / stride); the adapter twin below agrees on every cell it can construct."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps()
    f = _facts(**fkw)
    knobs = SdpaFwdKnobs(**kkw) if kkw else None
    why = engines.mismatch(caps, f, knobs)
    assert (why is None) == accept, (name, why)
    if accept and (f.has_bias or f.has_kv_mask):
        assert engines._selected_d_shape(caps, f, knobs) in caps.bias_d_shapes, name


def _api2(
    d=32, *, s_q=256, s_kv=256, b=2, h=4, dtype=torch.bfloat16, bias=None, bias_stride=None, kv_mask=None, kv_mask_dtype=torch.bool, kv_mask_stride=None, **kw
):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    q, o = (_desc((b, h, s_q, d), dtype, n) for n in ("q", "o"))
    k, v = (_desc((b, h, s_kv, d), dtype, n) for n in ("k", "v"))
    if bias is not None:
        kw["sample_bias"] = _desc(bias, dtype, "bias", stride=bias_stride)
    if kv_mask is not None:
        kw["sample_kv_mask"] = _desc(kv_mask, kv_mask_dtype, "kv_mask", stride=kv_mask_stride)
    return SdpaFwdDslSm100(q, k, v, o, None, **kw)


# (name, adapter kwargs, accept?, (tile_n, kv_mask_form, cta_mma) of the record)
_ADAPTER_CELLS = [
    ("tile_n 64 d32", dict(tile_n=64), True, (64, 0, 1)),
    ("tile_n 64 d32 + bias + bool mask", dict(tile_n=64, bias=(1, 4, 256, 256), kv_mask=(2, 1, 1, 256)), True, (64, 1, 1)),
    ("tile_n 64 d64 cga2 + bias", dict(d=64, tile_n=64, cga=2, bias=(1, 4, 256, 256)), True, (64, 0, 2)),
    ("tile_n 64 d128", dict(d=128, tile_n=64), False, None),
    ("tile_n 256 d32", dict(tile_n=256), False, None),
    # the 64-wide block holds 7 tiles = 448 columns (a factory fact pinned above), but the row AND the adapter ask the budget at the
    # default TILE_N 128 -- conservative, never wrong -- so S_kv 448 with the TILE_N 64 knob is a typed decline on both (the sweep that
    # would make the plan's width the budget's is the knob sweep's)
    ("tile_n 64 bias at S_kv 448 (the 64-wide block's 7 tiles; both gates ask at 128)", dict(s_q=448, s_kv=448, tile_n=64, bias=(1, 4, 448, 448)), False, None),
    ("additive bf16 vector", dict(kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.bfloat16), True, (128, 2, 1)),
    ("additive fp32 vector", dict(kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.float32), True, (128, 3, 1)),
    (
        "additive fp32 vector + bias (no tile_n: the adapter's default for the form is 64)",
        dict(bias=(1, 4, 256, 256), kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.float32),
        True,
        (64, 3, 1),
    ),
    (
        "additive fp32 vector + bias + tile_n 128 (the spilling rendering)",
        dict(tile_n=128, bias=(1, 4, 256, 256), kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.float32),
        False,
        None,
    ),
    ("additive fp32 vector + bias + tile_n 64", dict(tile_n=64, bias=(1, 4, 256, 256), kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.float32), True, (64, 3, 1)),
    ("additive fp16 vector on a bf16 graph", dict(kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.float16), False, None),
    ("additive fp16 vector on an fp16 graph", dict(dtype=torch.float16, kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.float16), True, (128, 2, 1)),
    ("additive bf16 vector + bias + tile_n 64", dict(tile_n=64, bias=(1, 4, 256, 256), kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.bfloat16), True, (64, 2, 1)),
    ("additive bf16 S_kv 100", dict(s_q=100, s_kv=100, kv_mask=(2, 1, 1, 100), kv_mask_dtype=torch.bfloat16, seq_kv_lens_present=True), False, None),
    ("additive bf16 padded batch stride 320", dict(kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.bfloat16, kv_mask_stride=(320, 320, 320, 1)), True, (128, 2, 1)),
    ("additive bf16 batch stride 260", dict(kv_mask=(2, 1, 1, 256), kv_mask_dtype=torch.bfloat16, kv_mask_stride=(260, 260, 260, 1)), False, None),
    ("bias [B, S_q, H, S_kv]-physical", dict(bias=(2, 4, 256, 256), bias_stride=(256 * 4 * 256, 256, 4 * 256, 1)), True, (128, 0, 1)),
    ("bias padded q-row stride 260 (520 B, not 16 bytes)", dict(bias=(1, 4, 256, 256), bias_stride=(4 * 256 * 260, 256 * 260, 260, 1)), False, None),
    ("bool kv_mask padded batch stride 300", dict(kv_mask=(2, 1, 1, 256), kv_mask_stride=(300, 300, 300, 1)), True, (128, 1, 1)),
]


@pytest.mark.parametrize("name, kw, accept, record", _ADAPTER_CELLS, ids=[c[0] for c in _ADAPTER_CELLS])
def test_adapter_check_support_is_the_rows_twin_for_the_three_arms(_rubin_device_for_the_adapter, name, kw, accept, record):
    """The standalone adapter's ``check_support`` (rule 8b twin): the same cells accept with the compile-time record carrying
    ``tile_n`` / ``kv_mask_form`` / ``cta_mma`` as expected, or decline with a typed exception (the S_kv 448 cell: both gates ask
    the bias budget at the default TILE_N 128, so the 64-wide block's seventh tile is declined by both -- conservative, consistent)."""
    if not accept:
        with pytest.raises((NotImplementedError, ValueError)):
            _api2(**kw).check_support()
        return
    api = _api2(**kw)
    assert api.check_support() is True, name
    params = api.template_params()
    assert (params.tile_n, params.kv_mask_form, params.cta_mma) == record, (name, params)


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)

_COUNTS = {
    "USETMAXREG": ("USETMAXREG",),
    "STL": (" STL",),
    "LDL": (" LDL",),
    "FMNMX3": ("FMNMX3",),
    "R2P": (" R2P ",),
    "CGAERRBAR": ("CGAERRBAR",),
    "FFMA2": (" FFMA2",),
    "VOTE": ("VOTE",),
    "LDG_128": (" LDG.E.128",),
    "MUFU_EX2": ("MUFU.EX2",),
}


def _probe(tmp_path, params, tag):
    return run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch="sm_107a", params=params, tag=tag)


_T64_ARMS = [
    # (d, cta, extra, base (the same mask arm at TILE_N 128, from the pre-bias record))
    (32, 1, {"tile_n": 64}, "d32_dense_bf16"),
    (32, 1, {"tile_n": 64, "seq_kv_lens_present": True}, "d32_padded_bf16"),
    (32, 1, {"tile_n": 64, "bias_mode": 1, "kv_mask_form": 1, "seq_kv_lens_present": True}, "d32_padded_bf16"),
    (64, 2, {"tile_n": 64, "bias_mode": 1, "kv_mask_form": 1, "seq_kv_lens_present": True}, "d64_padded_bf16_cga2"),
]


@pytest.mark.parametrize(
    "d, cta, extra, base_arm", _T64_ARMS, ids=["d32_t64_dense", "d32_t64_padded", "d32_t64_bias_kvmask_padded", "d64_t64_bias_kvmask_cga2_padded"]
)
def test_tile_n_64_renderings_fold_to_one_chunk_and_keep_the_split_and_the_spill_count(tmp_path, d, cta, extra, base_arm):
    """Each TILE_N = 64 rendering traced for sm_107a: a different cubin from the 128-wide one, the per-role register split
    (USETMAXREG > 0), no cluster drain, spills no more than the 128-wide rendering of the same mask arm, and HALF the exp
    work per body (MUFU.EX2 = one 64-column chunk per body) -- the one-chunk arm, not a 128-wide body reading garbage."""
    base = _probe(tmp_path, _RECORD_ARMS[base_arm], f"base_{base_arm}")
    params = dict(_RECORD_ARMS[base_arm])
    params.update({"__d": d, "cta_mma": cta, "dtype_qkv": 2})
    params.update(extra)
    probe = _probe(tmp_path, params, f"t64_{d}_{cta}_{'_'.join(sorted(extra))}")
    assert probe.cubin_md5 != base.cubin_md5
    assert probe.stats["USETMAXREG"] > 0 and probe.stats["CGAERRBAR"] == 0, probe.stats
    assert_no_new_spills(probe.stats, base.stats, tag=f"TILE_N 64 d{d} cga{cta} {sorted(extra)} vs {base_arm}: ")
    assert probe.stats["MUFU_EX2"] * 2 <= base.stats["MUFU_EX2"] + 8, (probe.stats, base.stats)
    if extra.get("kv_mask_form") or extra.get("seq_kv_lens_present"):
        assert probe.stats["R2P"] > 0 and probe.stats["FMNMX3"] > 0, probe.stats


# The fp32 vector WITH the bias at TILE_N 128 spills (MEASURED on this branch's toolchain, cutlass-dsl 4.8.0 + the CUDA 13.5 ptxas,
# sm_107a): the sixteen 128-bit granule loads of a 64-column chunk are 64 fp32 registers beside the 128 live S values, the 32
# pre-loaded bias words and the band words -- over the softmax role's 192 -- where the io-dtype form packs two values per register
# and stays at the bit word's count.  Neither named lever moves it (MEASURED, same toolchain): BIAS_PRELOAD_CHUNKS = 0 leaves
# 23 / 27, 15 / 15 and 16 / 16 unchanged (the peak is the SECOND chunk's, after the pre-load is consumed) and a streamed
# one-granule-at-a-time fold makes it WORSE (37 / 40, 25 / 25, 38 / 38 -- ptxas hoists the loads anyway).  So the rendering is NOT
# SERVED: the heuristics pick TILE_N 64 for the fp32 vector WITH a bias and the adapter defaults to it (64 live S: 3 / 3 padded = the
# base), the row and the adapter decline an EXPLICIT 128 typed (`engines._bias_arm_decline` / `check_support`), and the io-dtype vector
# stays the fast path (cast on the host).  The
# three renderings stay pinned here as the decline's REASON, from both sides: no more spills than measured (the upper bound, the
# regression detector) and still MORE than the bit-word twin plus the tolerance (the lower bound) -- when a toolchain stops
# spilling here, this test says so and the decline can be lifted.
_FP32_BIAS_SPILL_PINS = {(32, 1, False): {"STL": 23, "LDL": 27}, (32, 1, True): {"STL": 15, "LDL": 15}, (64, 2, True): {"STL": 16, "LDL": 16}}
_ADDITIVE_ARMS = [
    # (d, cta, extra, the bool-form twin's extra) -- the additive rendering is pinned against the SAME arm with kv_mask_form = 1
    (32, 1, {"kv_mask_form": 2}, {"kv_mask_form": 1}),
    (32, 1, {"kv_mask_form": 3}, {"kv_mask_form": 1}),
    (32, 1, {"kv_mask_form": 2, "bias_mode": 1, "seq_kv_lens_present": True}, {"kv_mask_form": 1, "bias_mode": 1, "seq_kv_lens_present": True}),
    (
        32,
        1,
        {"kv_mask_form": 2, "bias_mode": 1, "tile_n": 64, "seq_kv_lens_present": True},
        {"kv_mask_form": 1, "bias_mode": 1, "tile_n": 64, "seq_kv_lens_present": True},
    ),
    (
        32,
        1,
        {"kv_mask_form": 3, "bias_mode": 1, "tile_n": 64, "seq_kv_lens_present": True},
        {"kv_mask_form": 1, "bias_mode": 1, "tile_n": 64, "seq_kv_lens_present": True},
    ),
    (32, 1, {"kv_mask_form": 3, "bias_mode": 1, "window_right": 0}, {"kv_mask_form": 1, "bias_mode": 1, "window_right": 0}),
    (32, 1, {"kv_mask_form": 3, "bias_mode": 1, "seq_kv_lens_present": True}, {"kv_mask_form": 1, "bias_mode": 1, "seq_kv_lens_present": True}),
    (64, 2, {"kv_mask_form": 3, "bias_mode": 1, "seq_kv_lens_present": True}, {"kv_mask_form": 1, "bias_mode": 1, "seq_kv_lens_present": True}),
    # the additive form at d64 cga1 (served; no rendering existed before): the fp32 vector alone, the io-dtype vector with the bias
    (64, 1, {"kv_mask_form": 3}, {"kv_mask_form": 1}),
    (64, 1, {"kv_mask_form": 2, "bias_mode": 1}, {"kv_mask_form": 1, "bias_mode": 1}),
]
_ADDITIVE_IDS = [
    "d32_add16",
    "d32_addf32",
    "d32_add16_bias_padded",
    "d32_t64_add16_bias_padded",
    "d32_t64_addf32_bias_padded",
    "d32_addf32_bias_causal",
    "d32_addf32_bias_padded",
    "d64_addf32_bias_cga2_padded",
    "d64_addf32_cga1",
    "d64_add16_bias_cga1",
]


@pytest.mark.parametrize("d, cta, extra, twin", _ADDITIVE_ARMS, ids=_ADDITIVE_IDS)
def test_additive_kv_mask_renderings_fold_the_vector_as_packed_ffma_from_128_bit_loads(tmp_path, d, cta, extra, twin):
    """Each additive-vector rendering against the bit-word rendering of the SAME arm: the register split kept, no cluster drain,
    no new spills (the fp32 form with the bias at TILE_N 128: its MEASURED bound, ``_FP32_BIAS_SPILL_PINS``), MORE packed FFMA
    (the vector fold, one per pair), 128-bit granule loads present (never scalarised), the ballots gone (no bit word) -- and a
    different cubin."""
    base_params = {"__d": d, "cta_mma": cta, "dtype_qkv": 2, **twin}
    base = _probe(tmp_path, base_params, f"bool_{d}_{cta}_{'_'.join(sorted(twin))}")
    params = {"__d": d, "cta_mma": cta, "dtype_qkv": 2, **extra}
    probe = _probe(tmp_path, params, f"add_{d}_{cta}_{'_'.join(sorted(extra))}")
    assert probe.expect["KV_MASK_FORM"] == extra["kv_mask_form"] and base.expect["KV_MASK_FORM"] == 1
    assert probe.cubin_md5 != base.cubin_md5
    assert probe.stats["USETMAXREG"] > 0 and probe.stats["CGAERRBAR"] == 0, probe.stats
    pins = base.stats
    unserved = extra["kv_mask_form"] == 3 and extra.get("bias_mode") and extra.get("tile_n", 128) == 128
    if unserved:
        pins = _FP32_BIAS_SPILL_PINS[(d, cta, bool(extra.get("seq_kv_lens_present")))]
    assert_no_new_spills(
        probe.stats, pins, tag=f"additive d{d} cga{cta} {sorted(extra)} vs {'the measured fp32 bound' if pins is not base.stats else 'the bit word'}: "
    )
    if unserved:
        assert probe.stats["STL"] > base.stats["STL"] + SPILL_TOLERANCE, (
            f"the fp32 + bias rendering at TILE_N 128 no longer spills beyond the bit word ({probe.stats['STL']} vs {base.stats['STL']} STL): "
            "the typed decline in engines._bias_arm_decline / SdpaFwdDslSm100.check_support can be lifted"
        )
    assert probe.stats["FFMA2"] > base.stats["FFMA2"], (probe.stats, base.stats)
    assert probe.stats["LDG_128"] > 0, probe.stats
    assert probe.stats["VOTE"] < base.stats["VOTE"], (probe.stats, base.stats)


# --------------------------------------------------------------------------- tier 3: GPU cells (a Rubin device)


def _executed_tile_n(g, idx):
    """The KV tile width the EXECUTED plan was compiled with -- never the record the test asked for, so a 64-wide cell is evidenced by
    the rendering that ran, not inferred from the adapter tier.  Read from the compiled kernel module's own ``CFG.TILE_N`` through the
    prepared launch spec (``prepared.DenseLaunchSpec.tile_n``); a plan the lowering built WITHOUT a prepared launch (one KV tile, S_kv
    64) exposes the adapter it compiled through the executor closure, whose ``template_params().tile_n`` is the width the kernel
    module was loaded with (``_load_sm100_kernel_module((d, d), params)``)."""
    plan = g._compiled_plans[idx]
    spec = getattr(getattr(plan, "_prepared", None), "spec", None)
    if spec is not None and hasattr(spec, "tile_n"):
        return int(spec.tile_n)
    cells = getattr(plan._compiled, "__closure__", None) or ()
    api = next((c.cell_contents for c in cells if type(c.cell_contents).__name__.startswith("SdpaFwdDsl")), None)
    assert api is not None, f"the compiled plan carries neither a prepared launch spec nor the adapter it compiled: {type(plan).__name__}"
    return int(api.template_params().tile_n)


def _run2(q, k, v, *, scale, bias=None, kv_mask=None, sdpa_kwargs=None, seq_len_kv=None, seq_len_q=None, sink=None, stats=True, cga=None, tile_n=None):
    """``test_sdpa_bias_sm107._run`` with the TILE_N knob: build the graph, pin the Rubin f16 row (a cga and / or a TILE_N through
    the public autotune record), launch TWICE on sentinel-filled outputs (the two-launch delta) and return O, Stats, the plan's OWN
    knob record and the width the compiled kernel actually carries (``executed_tile_n``)."""
    import cudnn

    g, vp, o, o_gpu, stats_gpu = _build(
        q, k, v, scale=scale, bias=bias, kv_mask=kv_mask, sdpa_kwargs=sdpa_kwargs, seq_len_kv=seq_len_kv, seq_len_q=seq_len_q, sink=sink, stats=stats
    )
    g.create_execution_plans([cudnn.heur_mode.A])
    names, ours = _frost_plans(g)
    idx = next((i for i in ours if g.plans[i].knobs.split_kv in (None, 1)), None)
    assert idx is not None, f"no unsplit {_ROW} plan in {names}"
    picked = g.plans[idx].knobs
    want_cga = picked.cga if cga is None else cga
    want_tile_n = picked.tile_n if tile_n is None else tile_n
    if (want_cga, want_tile_n) != (picked.cga, picked.tile_n):
        engine_id, record = g.get_engine_and_knobs_at_index(idx)
        record = dict(record)
        record[cudnn.knob_type.TILE_CGA_M] = want_cga
        record[cudnn.knob_type.TILE_N] = want_tile_n
        g.create_execution_plan(engine_id, record)
        idx = g.get_execution_plan_count() - 1
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    knobs = g.plans[idx].knobs
    assert knobs.tile_n == want_tile_n and knobs.cga == want_cga, (knobs, want_tile_n, want_cga)
    # The EXECUTED rendering's width from the compiled kernel's CFG (a knob-less plan runs the row's default 128).
    executed_tile_n = _executed_tile_n(g, idx)
    assert executed_tile_n == (want_tile_n if want_tile_n is not None else 128), (executed_tile_n, want_tile_n, knobs)
    vp[o] = o_gpu
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

    def launch():
        o_gpu.fill_(float("nan"))
        if stats_gpu is not None:
            stats_gpu.fill_(float("nan"))
        g.execute(vp, ws)
        torch.cuda.synchronize()
        return o_gpu.clone(), (stats_gpu.clone() if stats_gpu is not None else None)

    o1, st1 = launch()
    o2, st2 = launch()
    assert torch.equal(o1, o2) and (st1 is None or torch.equal(st1, st2)), "two-launch delta on O / Stats"
    return dict(o=o_gpu, stats=stats_gpu, knobs=knobs, executed_tile_n=executed_tile_n, names=names)


def _ref_add(q, k, v, *, scale, bias=None, add=None, kv_mask=None, causal=False, seq_kv_lens=None):
    """The fp64 oracle with an ADDITIVE per-key vector folded as the eager models fold it: fp32 scores + the fp32 vector (a
    |s| < 100 score against -1e9 is swamped in fp32 exactly as the kernel's fma swamps it), then the fp64 softmax; the other
    terms through ``_ref``."""
    if add is None:
        return _ref(q, k, v, scale=scale, bias=bias, kv_mask=kv_mask, causal=causal, seq_kv_lens=seq_kv_lens)
    b, h_q, s_q, _ = q.shape
    s_kv = k.shape[2]
    scores = (torch.matmul(q.double(), k.double().transpose(-1, -2)) * scale).float()
    if bias is not None:
        scores = scores + bias.float()
    scores = (scores + add.float().view(add.shape[0], 1, 1, s_kv)).double()
    dev = q.device
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    masked = torch.zeros(1, 1, s_q, s_kv, dtype=torch.bool, device=dev) | (j > i if causal else False)
    if seq_kv_lens is not None:
        masked = masked | (j >= seq_kv_lens.flatten().to(torch.int64).view(b, 1, 1, 1))
    if kv_mask is not None:
        masked = masked | (~kv_mask.view(kv_mask.shape[0], 1, 1, s_kv))
    scores = scores.masked_fill(masked, float("-inf"))
    smax = scores.max(dim=-1).values
    ssum = torch.exp(scores - smax.unsqueeze(-1)).sum(dim=-1)
    lse = smax + torch.log(ssum)
    o = torch.matmul(torch.softmax(scores, dim=-1).nan_to_num(0.0), v.double())
    return dict(o=o, lse=lse, empty=torch.isneginf(lse))


def _inputs(c, dtype=torch.bfloat16, *, seed=0):
    """Inputs of one cell: BSHD-physical activations (``layout`` = "bhsd" for contiguous (B, H, S, D), "headpad" for a 16-byte
    padded head stride), an optional bias (per-head; ``bias_perm`` = the [B, S_q, H, S_kv]-physical view; ``bias_row`` = a
    padded row stride), a bool / additive kv_mask (``add`` = "bf16" / "f32" / "inf" / "full"; ``mask_pad`` = a padded batch stride)."""
    torch.manual_seed(seed)
    b, h, s, d = c.get("b", 2), c.get("h", 4), c["s"], c.get("d", 32)
    layout = c.get("layout", "bshd")
    if layout == "bhsd":
        mk = lambda: torch.randn(b, h, s, d, device="cuda", dtype=dtype)  # noqa: E731
    elif layout == "headpad":
        mk = lambda: torch.randn(b, s, h, d + 8, device="cuda", dtype=dtype)[..., :d].transpose(1, 2)  # noqa: E731
    else:
        mk = lambda: _bhsd(b, h, s, d, dtype)  # noqa: E731
    q, k, v = mk(), mk(), mk()
    bias = None
    if c.get("bias"):
        if c.get("bias_perm"):
            bias = torch.randn(b, s, h, s, device="cuda", dtype=torch.float32).to(dtype).transpose(1, 2)
        else:
            row = c.get("bias_row", s)
            bias = torch.randn(1, h, s, row, device="cuda", dtype=torch.float32).to(dtype)[..., :s]
    kv_mask = add = None
    pad = c.get("mask_pad", 0)
    keep = torch.rand(b, 1, 1, s + pad, device="cuda") > 0.3
    if c.get("mask"):
        kv_mask = keep[..., :s].contiguous() if not pad else keep[..., :s]
    form = c.get("add")
    if form is not None:
        vdt = torch.float32 if form == "f32" else dtype
        # the finite fold: -1e9 where the dtype holds it (fp32, bf16), the dtype's most negative finite value where it does not (fp16: 65504)
        fill = float("-inf") if form == "inf" else (-1e9 if torch.finfo(vdt).max > 1e9 else float(torch.finfo(vdt).min))
        vec = torch.where(keep, torch.zeros((), device="cuda", dtype=vdt), torch.full((), fill, device="cuda", dtype=vdt))
        if form == "full":
            vec[1] = -1e9  # batch 1: every key -1e9 -> the eager uniform average, NOT the empty-row contract
        add = vec[..., :s]
    kv_lens = torch.tensor(c["kv_lens"], dtype=torch.int32, device="cuda").view(b, 1, 1, 1) if "kv_lens" in c else None
    scale = 1.0 / math.sqrt(d)
    sdpa_kwargs = dict(use_causal_mask=True) if c.get("causal") else {}
    return (
        q,
        k,
        v,
        dict(scale=scale, bias=bias, kv_mask=kv_mask, add=add, keep=keep[..., :s], kv_lens=kv_lens, sdpa_kwargs=sdpa_kwargs, causal=bool(c.get("causal"))),
    )


def _run_cell(c, io, *, tile_n=None, add_as_mask=True):
    """Run one cell; the additive vector travels as the graph's ``kv_mask`` (``add_as_mask``), the bool one otherwise."""
    q, k, v, kw = _inputs(c, io)
    mask_arg = kw["add"] if (kw["add"] is not None and add_as_mask) else kw["kv_mask"]
    out = _run2(
        q, k, v, scale=kw["scale"], bias=kw["bias"], kv_mask=mask_arg, sdpa_kwargs=kw["sdpa_kwargs"], seq_len_kv=kw["kv_lens"], cga=c.get("cga"), tile_n=tile_n
    )
    ref = _ref_add(
        q,
        k,
        v,
        scale=kw["scale"],
        bias=kw["bias"],
        add=kw["add"] if add_as_mask else None,
        kv_mask=kw["kv_mask"],
        causal=kw["causal"],
        seq_kv_lens=kw["kv_lens"],
    )
    return out, ref, kw


# TILE_N = 64 cells: n_kv = 1 / 2 / 4 (= the ring depth) / 6 at the 64-wide tile with B*H > 1, a padded tail (200 / 150 of
# 256: a partial 64-wide tile), plain / bias / bool mask / both / causal, d64 at cga1 and cga2 (the bias at S_kv 256 at cga2 =
# the 128-wide budget's edge, served at 64 too).
_T64_CASES = {
    "t64_dense_64_nkv1": dict(s=64),
    "t64_dense_128_nkv2": dict(s=128),
    "t64_dense_256_nkv4": dict(s=256),
    "t64_dense_384_nkv6": dict(s=384, h=2),
    "t64_causal_384": dict(s=384, h=2, causal=True),
    "t64_padded_tail_256": dict(s=256, kv_lens=[200, 150]),
    "t64_bias_256": dict(s=256, bias=True),
    "t64_mask_256": dict(s=256, mask=True),
    "t64_both_384": dict(s=384, h=2, bias=True, mask=True),
    "t64_both_causal_padded_384": dict(s=384, h=2, bias=True, mask=True, causal=True, kv_lens=[384, 300]),
    "t64_d64_both_128_cga1": dict(d=64, s=128, bias=True, mask=True),
    "t64_d64_both_256_cga2": dict(d=64, s=256, h=2, bias=True, mask=True, cga=2),
    "t64_f16_both_256": dict(s=256, bias=True, mask=True, io=torch.float16),  # the 64-wide rendering on an fp16 graph
}


@requires_rubin
@pytest.mark.parametrize("case", list(_T64_CASES))
def test_tile_n_64_cells_match_the_fp64_oracle_and_the_128_wide_rendering(case):
    """Every TILE_N = 64 cell: the TILE_N knob honoured (the plan's record reads 64), O / Stats against the fp64 oracle, keyless
    rows exactly 0 / -inf, two launches bitwise -- and within the same tolerance of the 128-wide rendering of the same graph
    (never bitwise: the online softmax combines its chunk sums in a different order)."""
    c = _T64_CASES[case]
    io = c.get("io", torch.bfloat16)
    out64, ref, _ = _run_cell(c, io, tile_n=64)
    assert out64["knobs"].tile_n == 64 and out64["knobs"].cga == c.get("cga", 1), out64["knobs"]
    assert out64["executed_tile_n"] == 64, "the compiled kernel of the executed plan must carry TILE_N 64"
    _check(out64, ref)
    out128, _, _ = _run_cell(c, io, tile_n=128)
    assert out128["executed_tile_n"] == 128
    live = ~ref["empty"]
    torch.testing.assert_close(out64["o"][live].double(), out128["o"][live].double(), **_TOL_O)
    torch.testing.assert_close(out64["stats"].squeeze(-1)[live].double(), out128["stats"].squeeze(-1)[live].double(), **_TOL_LSE)


@requires_rubin
def test_tile_n_64_bias_zero_and_mask_all_true_are_bitwise_the_plain_64_wide_kernel():
    """The identity check at TILE_N 64: bias = 0 / mask = all True on the 64-wide arms are BITWISE the plain 64-wide kernel."""
    c = dict(s=256)
    q, k, v, kw = _inputs(c, torch.bfloat16)
    plain = _run2(q, k, v, scale=kw["scale"], tile_n=64)
    zero = torch.zeros(1, 1, 256, 256, device="cuda", dtype=torch.bfloat16)
    ones = torch.ones(2, 1, 1, 256, device="cuda", dtype=torch.bool)
    zero_vec = torch.zeros(2, 1, 1, 256, device="cuda", dtype=torch.bfloat16)
    for label, b_arg, m_arg in (("bias=0 + mask=all-true", zero, ones), ("bias=0", zero, None), ("mask=all-true", None, ones), ("additive=0", None, zero_vec)):
        out = _run2(q, k, v, scale=kw["scale"], bias=b_arg, kv_mask=m_arg, tile_n=64)
        assert torch.equal(out["o"], plain["o"]) and torch.equal(out["stats"], plain["stats"]), label


@requires_rubin
def test_tile_n_256_is_a_typed_decline_on_a_real_graph():
    """The TILE_N 256 request through the public autotune record is refused by the plan creation (the row's domain), never
    silently degraded to 128."""
    import cudnn

    q, k, v, kw = _inputs(dict(s=256), torch.bfloat16)
    g, vp, o, o_gpu, stats_gpu = _build(q, k, v, scale=kw["scale"])
    g.create_execution_plans([cudnn.heur_mode.A])
    names, ours = _frost_plans(g)
    assert ours, names
    engine_id, record = g.get_engine_and_knobs_at_index(ours[0])
    record = dict(record)
    record[cudnn.knob_type.TILE_N] = 256
    with pytest.raises(Exception) as excinfo:
        g.create_execution_plan(engine_id, record)
        g.select_plan(g.get_execution_plan_count() - 1)
        g.check_support()
        g.build_plans()
    assert "tile_n" in str(excinfo.value) or "TILE_N" in str(excinfo.value) or "domain" in str(excinfo.value), str(excinfo.value)[:400]


# The additive vector: bf16 and fp32, the -1e9 fold alone / with the bias / causal / padded, across the ring wrap (n_kv 4), at
# TILE_N 64, d64 at cga2, a padded 16-byte batch stride, an fp16 graph with an fp16 vector.
_ADD_CASES = {
    "add_bf16_256": dict(s=256, add="bf16"),
    "add_f32_256": dict(s=256, add="f32"),
    "add_bf16_bias_384": dict(s=384, h=2, add="bf16", bias=True),
    "add_f32_bias_causal_384": dict(s=384, h=2, add="f32", bias=True, causal=True),  # the fp32 + bias form: the heuristics pick TILE_N 64
    "add_f32_bias_causal_t64_384": dict(s=384, h=2, add="f32", bias=True, causal=True, tile_n=64),  # ... and the explicit knob
    "add_f32_d64_128_cga1": dict(d=64, s=128, add="f32"),
    "add_bf16_bias_d64_128_cga1": dict(d=64, s=128, add="bf16", bias=True),
    "add_bf16_padded_384": dict(s=384, h=2, add="bf16", kv_lens=[384, 200]),
    "add_bf16_512_nkv4": dict(s=512, h=2, add="bf16"),
    "add_bf16_t64_256": dict(s=256, add="bf16", tile_n=64),
    "add_f32_d64_256_cga2": dict(d=64, s=256, h=2, add="f32", cga=2),
    "add_bf16_batchpad_256": dict(s=256, add="bf16", mask_pad=64),
}


@requires_rubin
@pytest.mark.parametrize("case", list(_ADD_CASES))
def test_additive_kv_mask_matches_the_fp64_oracle_with_the_vector_folded_in_fp32(case):
    """The additive vector (-1e9 on 30 % of the keys, 0 elsewhere) folded like the eager models fold it: O / Stats against the
    fp64 oracle with the fp32 add, keyless rows (none here: a -1e9 key is finite) absent, two launches bitwise."""
    from cudnn.sdpa.fwd import config_sm107 as cfg107

    c = _ADD_CASES[case]
    out, ref, _ = _run_cell(c, torch.bfloat16, tile_n=c.get("tile_n"))
    assert out["knobs"].cga == c.get("cga", 1)
    # A knob-less cell runs the width the heuristics' table picks -- ONE arithmetic, ``config_sm107.sm107_tile_n_pick`` (the vector
    # travels as the graph's kv_mask: the narrow tile for a bias WITH a mask of any dtype and for a mask alone at one 128-wide kv
    # tile, 128 elsewhere; test_sdpa_heuristics_small_tile_sm107.py pins the table itself) -- and the fp32 vector WITH a bias has ONE
    # served width, 64, whether the knob is given or the table picks it.
    want = c.get("tile_n")
    if want is None:
        want = cfg107.sm107_tile_n_pick(c.get("d", 32), c["s"], has_bias=bool(c.get("bias")), has_kv_mask=True, padded="kv_lens" in c)
    if c.get("add") == "f32" and c.get("bias"):
        assert want == cfg107.SM107_FP32_VECTOR_WITH_BIAS_TILE_N == 64, (case, want)
    assert out["executed_tile_n"] == want, (case, want, out["knobs"])
    _check(out, ref)


@requires_rubin
def test_fp32_additive_vector_with_a_bias_runs_tile_n_64_and_declines_an_explicit_128():
    """The fp32 vector WITH a bias: the heuristics' plan carries TILE_N 64 (the only served width of the form), the EXECUTED kernel's
    CFG.TILE_N is 64, and a record asking TILE_N 128 (the spilling rendering) is refused at plan creation with the two served routes
    named -- never degraded."""
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd import engines

    q, k, v, kw = _inputs(dict(s=256, add="f32", bias=True), torch.bfloat16)
    out = _run2(q, k, v, scale=kw["scale"], bias=kw["bias"], kv_mask=kw["add"])
    assert out["knobs"].tile_n == 64 and out["executed_tile_n"] == 64, out["knobs"]
    g, *_ = _build(q, k, v, scale=kw["scale"], bias=kw["bias"], kv_mask=kw["add"])
    g.create_execution_plans([cudnn.heur_mode.A])
    names, ours = _frost_plans(g)
    assert ours and all("TILE_N=64" in names[i] for i in ours), names
    engine_id, record = g.get_engine_and_knobs_at_index(ours[0])
    record = dict(record)
    record[cudnn.knob_type.TILE_N] = 128
    with pytest.raises(Exception) as excinfo:
        g.create_execution_plan(engine_id, record)
        g.select_plan(g.get_execution_plan_count() - 1)
        g.check_support()
        g.build_plans()
    assert "TILE_N 64" in str(excinfo.value) or "tile_n=64" in str(excinfo.value) or "spills" in str(excinfo.value), str(excinfo.value)[:400]
    why = engines.mismatch(_caps(), ga.analyze(g), engines.SdpaFwdKnobs(tile_n=128))
    assert why is not None and "TILE_N 64" in why and "fp32" in why, why
    assert engines.mismatch(_caps(), ga.analyze(g)) is None, "the knob-less probe accepts: the heuristics pick the served width"


@requires_rubin
def test_additive_kv_mask_fp16_vector_on_an_fp16_graph():
    c = dict(s=256, add="bf16", bias=True)  # "bf16" = the io dtype's vector: fp16 here
    out, ref, _ = _run_cell(c, torch.float16)
    _check(out, ref)


@requires_rubin
def test_additive_minus_inf_image_is_bitwise_the_boolean_form():
    """A -inf / 0 vector is the bit word's image: ``fma(-inf, 1/scale, s) = -inf`` lands exactly where the keep word's select
    puts -inf, so O and Stats are BITWISE the BOOLEAN form (alone, with the bias, at TILE_N 64, through the ring wrap)."""
    for c in (dict(s=256, add="inf"), dict(s=384, h=2, add="inf", bias=True), dict(s=256, add="inf", tile_n=64), dict(s=512, h=2, add="inf")):
        q, k, v, kw = _inputs(c, torch.bfloat16)
        add = _run2(q, k, v, scale=kw["scale"], bias=kw["bias"], kv_mask=kw["add"], tile_n=c.get("tile_n"))
        word = _run2(q, k, v, scale=kw["scale"], bias=kw["bias"], kv_mask=kw["keep"].contiguous(), tile_n=c.get("tile_n"))
        assert torch.equal(add["o"], word["o"]) and torch.equal(add["stats"], word["stats"]), c
        _check(add, _ref(q, k, v, scale=kw["scale"], bias=kw["bias"], kv_mask=kw["keep"]))


@requires_rubin
def test_additive_fully_masked_row_is_the_eager_uniform_average_not_the_empty_row():
    """A batch whose vector is -1e9 on EVERY key (the models' fold on a padding-only row): the kernel keeps the finite value, so
    the row is the uniform average of V with Stats ~ -1e9 + ln(S_kv) -- eager math -- while the BOOLEAN all-False twin reads the
    empty-row contract (O = 0, Stats = -inf); the other batch matches the oracle."""
    c = dict(s=256, add="full")
    q, k, v, kw = _inputs(c, torch.bfloat16)
    out = _run2(q, k, v, scale=kw["scale"], kv_mask=kw["add"])
    s = 256
    uniform = v[1].double().mean(dim=-2, keepdim=True).expand(-1, s, -1)  # (H, S_q, D): every row the mean of V over the keys
    torch.testing.assert_close(out["o"][1].double(), uniform, atol=5e-2, rtol=3e-2)
    lse1 = out["stats"][1].squeeze(-1).double()
    folded = kw["add"][1].flatten()[0].double().item()  # the vector's VALUE in its dtype: bf16(-1e9) = -998244352, what the kernel folds
    assert torch.isfinite(lse1).all() and ((lse1 - (folded + math.log(s))).abs() < 512).all(), f"Stats ~ {folded} + ln(S_kv): the finite fold, not -inf"
    ref = _ref_add(q, k, v, scale=kw["scale"], add=kw["add"])
    torch.testing.assert_close(out["o"][0].double(), ref["o"][0], **_TOL_O)
    torch.testing.assert_close(out["stats"][0].squeeze(-1).double(), ref["lse"][0], **_TOL_LSE)
    keep = kw["keep"].clone()
    keep[1] = False
    word = _run2(q, k, v, scale=kw["scale"], kv_mask=keep.contiguous())
    assert (word["o"][1] == 0).all() and torch.isneginf(word["stats"][1]).all(), "the BOOLEAN all-False twin: the empty-row contract"


# The strided outer axes: BHSD-contiguous activations, a 16-byte padded head stride, the permuted and the padded-row bias, a
# padded kv_mask batch stride (bool and additive) -- every one a graph FACT the row judges and the kernels address.
_STRIDED_CASES = {
    "headpad8_bias_mask_256": dict(s=256, layout="headpad", bias=True, mask=True),
    "bias_perm_bshs_256": dict(s=256, bias=True, bias_perm=True),
    "bias_perm_bshs_causal_t64_384": dict(s=384, h=2, bias=True, bias_perm=True, causal=True, tile_n=64),
    "bias_padded_row_320_256": dict(s=256, bias=True, bias_row=320),
    "mask_batchpad_64_256": dict(s=256, mask=True, mask_pad=64),
    "add_f32_batchpad_8_256": dict(s=256, add="f32", mask_pad=8),
    "headpad8_add_bf16_bias_perm_256": dict(s=256, layout="headpad", add="bf16", bias=True, bias_perm=True),
}


@requires_rubin
@pytest.mark.parametrize("case", list(_STRIDED_CASES))
def test_strided_outer_axes_are_addressed_through_the_declared_strides(case):
    c = _STRIDED_CASES[case]
    out, ref, kw = _run_cell(c, torch.bfloat16, tile_n=c.get("tile_n"))
    _check(out, ref)
    if c.get("bias_perm"):
        assert kw["bias"].stride()[1] < kw["bias"].stride()[2], "the permuted bias stays a permuted VIEW (no copy by the test)"


@requires_rubin
def test_bhsd_contiguous_activations_are_the_rows_pre_existing_typed_decline():
    """The row's layout envelope is ``bshd`` (Q/K/V/O BSHD-physical, padded 16-byte strides included -- the "layout_left" customer
    form): a (B, H, S, D)-CONTIGUOUS activation is the row's pre-existing typed decline ("must be BSHD-physical"), with or without
    the bias; with the python-only kv_mask bound nothing else serves it either."""
    import cudnn

    q, k, v, kw = _inputs(dict(s=256, layout="bhsd", bias=True), torch.bfloat16)
    g, *_ = _build(q, k, v, scale=kw["scale"], bias=kw["bias"])
    try:
        g.create_execution_plans([cudnn.heur_mode.A])
        names, ours = _frost_plans(g)
    except cudnn.cudnnGraphNotSupportedError:
        names, ours = [], []
    assert not ours, names
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd import engines

    why = engines.mismatch(_caps(), ga.analyze(g))
    assert why is not None and "BSHD" in why, why


def _no_frost_service(g):
    """Assert the graph is NOT served by the FROST row and say where the typed decline fell: at plan-list time (no FROST plan, or no
    plan at all) or at check_support / build time (the plan list offered the row, the build refused it).  A plan that BUILDS fails
    the test -- the kernel must never run on these operands."""
    import cudnn

    try:
        g.create_execution_plans([cudnn.heur_mode.A])
    except cudnn.cudnnGraphNotSupportedError as exc:
        return f"list: {exc}"
    names, ours = _frost_plans(g)
    if not ours:
        return f"list: no FROST plan in {names}"
    g.select_plan(ours[0])
    with pytest.raises((cudnn.cudnnGraphNotSupportedError, NotImplementedError, ValueError, RuntimeError)) as excinfo:
        g.check_support()
        g.build_plans()
    return f"build: {excinfo.value}"


@requires_rubin
@pytest.mark.parametrize("arm", ["both", "mask"])
def test_one_kv_tile_of_64_with_the_resident_arms_is_served_or_declined_typed(arm):
    """S_kv = 64 at TILE_N 64 (one KV tile, the acceptance grid's 64) with the bias + bool mask / the mask alone: the plan is either
    SERVED through the prepared launch (then O / Stats match the oracle, two launches bitwise, the executed CFG.TILE_N reads 64) or it
    is a TYPED decline at list / check_support / build time -- never a plan that builds and fails at execute (the staging-copy path
    carries neither resident-arm operand).  The outcome is recorded by the assertion message, not assumed."""
    import cudnn

    c = dict(s=64, bias=arm == "both", mask=True)
    q, k, v, kw = _inputs(c, torch.bfloat16)
    g, vp, o, o_gpu, stats_gpu = _build(q, k, v, scale=kw["scale"], bias=kw["bias"], kv_mask=kw["kv_mask"])
    try:
        g.create_execution_plans([cudnn.heur_mode.A])
        names, ours = _frost_plans(g)
    except cudnn.cudnnGraphNotSupportedError as exc:
        pytest.skip(f"S_kv 64 + TILE_N 64 + {arm}: a typed decline at list time -- {exc}")
    if not ours:
        pytest.skip(f"S_kv 64 + TILE_N 64 + {arm}: no FROST plan at list time ({names})")
    try:
        out = _run_cell(c, torch.bfloat16, tile_n=64)
    except (cudnn.cudnnGraphNotSupportedError, NotImplementedError, ValueError) as exc:
        assert "prepared" in str(exc) or "domain" in str(exc) or "TILE_N" in str(exc), f"an untyped decline: {exc}"
        pytest.skip(f"S_kv 64 + TILE_N 64 + {arm}: a typed decline at plan creation / build -- {str(exc)[:200]}")
    res, ref, _ = out
    assert res["executed_tile_n"] == 64
    _check(res, ref)


@requires_rubin
def test_non_16_byte_outer_strides_are_typed_declines_on_real_graphs():
    """A head stride of D + 4 bf16 (72 B) on Q/K/V/O, a bias row stride of S + 4 and an additive vector with a 260-element
    batch stride are not TMA / granule addressable: with the python-only kv_mask bound no engine serves them, and the bias /
    activation forms leave the FROST row out -- a typed decline at plan-list time or at check_support / build time, never a
    launch (``_no_frost_service`` reports where)."""
    import cudnn

    d, s = 32, 256
    # (a) a non-16-byte head stride with the kv_mask bound: no engine serves it -- the row declines it at plan-LIST time (the resident
    # arms bind through the prepared launch only; the staging-copy path does not carry the kv_mask, so a plan that built there used to
    # fail at EXECUTE time with "kv_mask_tensor is required by this compiled specialization")
    q, k, v = (torch.randn(2, s, 4, d + 4, device="cuda", dtype=torch.bfloat16)[..., :d].transpose(1, 2) for _ in range(3))
    mask = torch.ones(2, 1, 1, s, device="cuda", dtype=torch.bool)
    g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), kv_mask=mask)
    where = _no_frost_service(g)
    assert where.startswith("list:"), where
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd import engines

    why = engines.mismatch(_caps(), ga.analyze(g))
    assert why is not None and "prepared launch" in why and "bind without a copy" in why, why
    # (b) a bias with a row stride of S + 4: the FROST row is out (the backend may serve it)
    q, k, v = (_bhsd(2, 4, s, d, torch.bfloat16) for _ in range(3))
    bias = torch.randn(1, 4, s, s + 4, device="cuda", dtype=torch.bfloat16)[..., :s]
    g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), bias=bias)
    try:
        g.create_execution_plans([cudnn.heur_mode.A])
        names, ours = _frost_plans(g)
    except cudnn.cudnnGraphNotSupportedError:
        names, ours = [], []
    assert not ours, names
    # (c) an additive vector with a 260-element batch stride: no engine
    vec = torch.zeros(2, 1, 1, s + 4, device="cuda", dtype=torch.bfloat16)[..., :s]
    g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), kv_mask=vec)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g.create_execution_plans([cudnn.heur_mode.A])
        names, ours = _frost_plans(g)
        assert not ours, names
        raise cudnn.cudnnGraphNotSupportedError("no FROST plan for a 520-byte batch stride and the backend has no kv_mask field")
    # (d) an additive fp16 vector on a bf16 graph and an S_kv of 100: no engine
    for vec in (torch.zeros(2, 1, 1, s, device="cuda", dtype=torch.float16),):
        g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), kv_mask=vec)
        with pytest.raises(cudnn.cudnnGraphNotSupportedError):
            g.create_execution_plans([cudnn.heur_mode.A])
            names, ours = _frost_plans(g)
            assert not ours, names
            raise cudnn.cudnnGraphNotSupportedError("no FROST plan for an fp16 vector on a bf16 graph")
    q, k, v = (_bhsd(2, 4, 100, d, torch.bfloat16) for _ in range(3))
    g, *_ = _build(q, k, v, scale=1.0 / math.sqrt(d), kv_mask=torch.zeros(2, 1, 1, 100, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g.create_execution_plans([cudnn.heur_mode.A])
        names, ours = _frost_plans(g)
        assert not ours, names
        raise cudnn.cudnnGraphNotSupportedError("no FROST plan for an additive vector at S_kv 100")


# --------------------------------------------------------------------------- the PLAIN padded body's tail form (TemplateParams.padded_tail_form)
#
# The padding leg's measured pathology: at TILE_N 128 the padded PLAIN rendering runs the tail kv tile through a SECOND, software-max
# softmax body beside the HW-max body (two full bodies per warp group -- 4139 vs 2776 SASS instructions on the d32 bf16 rendering, over
# the per-SM instruction-cache cliff), so an S_kv off the tile multiple costs MORE than the dense next size up and more than the same
# shape WITH a kv_mask (whose -inf body is one body).  Form 1 = the pre-masked tail: ONE loop with the HW-max body, the tail tile's
# out-of-range S columns rewritten as -inf in TMEM before the fused max load.  The pins: the eligibility (one arithmetic for the
# config pick, the factory and the adapter), the typed declines, the rendering (one body: the dense rendering's exp count, no
# software row max, the bit-word pre-mask, no new spills), and the GPU cells against the fp64 oracle and the legacy form.


def test_padded_tail_form_pick_is_one_arithmetic_with_the_factory():
    """``sm107_padded_tail_form_pick`` returns 0 off the PLAIN padded small-tile body (dense, a bias, a kv_mask, a band, d128) whatever
    the request or the seam say; on it the request wins over the seam, the seam over the default; and the factory refuses form 1 on
    exactly the shapes the pick returns 0 for (a band, a bias, a kv_mask, the d128 flavor) while rendering form 0 / 1 on the eligible one."""
    from cudnn.sdpa.fwd import config_sm107 as cfg_mod
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.config_sm107 import (
        PADDED_TAIL_FORMS,
        SM107_PADDED_TAIL_FORM_DEFAULT,
        make_cfg_d32,
        make_cfg_d64,
        make_cfg_d128,
        sm107_padded_tail_form_pick,
    )

    assert SM107_PADDED_TAIL_FORM_DEFAULT in PADDED_TAIL_FORMS
    for d in (32, 64):
        assert sm107_padded_tail_form_pick(d, padded=True, has_bias=False, has_kv_mask=False) == SM107_PADDED_TAIL_FORM_DEFAULT
        assert sm107_padded_tail_form_pick(d, padded=True, has_bias=False, has_kv_mask=False, request=1) == 1
        assert sm107_padded_tail_form_pick(d, padded=True, has_bias=False, has_kv_mask=False, request=0) == 0
        for off in (dict(padded=False), dict(has_bias=True), dict(has_kv_mask=True), dict(causal=True), dict(swa=True)):
            kw = dict(padded=True, has_bias=False, has_kv_mask=False)
            kw.update(off)
            assert sm107_padded_tail_form_pick(d, request=1, **kw) == 0, (d, off)
    assert sm107_padded_tail_form_pick(128, padded=True, has_bias=False, has_kv_mask=False, request=1) == 0
    saved = cfg_mod._PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS
    try:
        cfg_mod._PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS = 1
        assert sm107_padded_tail_form_pick(32, padded=True, has_bias=False, has_kv_mask=False) == 1
        assert sm107_padded_tail_form_pick(32, padded=True, has_bias=False, has_kv_mask=False, request=0) == 0
        assert sm107_padded_tail_form_pick(32, padded=False, has_bias=False, has_kv_mask=False) == 0
    finally:
        cfg_mod._PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS = saved
    base = dict(cta_mma=1, dtype_qkv=2, seq_kv_lens_present=True)
    for form in (0, 1):
        cfg, _ = make_cfg_d32(TemplateParams(**base, padded_tail_form=form))
        assert cfg.PADDED_TAIL_FORM == form and cfg.MASK_FLAGS == 1
    cfg, _ = make_cfg_d64(TemplateParams(**{**base, "cta_mma": 2}, padded_tail_form=1))
    assert cfg.PADDED_TAIL_FORM == 1
    assert make_cfg_d32(TemplateParams(cta_mma=1, dtype_qkv=2))[0].PADDED_TAIL_FORM == 0  # a record without the field renders the legacy form
    for bad in (
        dict(cta_mma=1, dtype_qkv=2, padded_tail_form=1),  # dense: no padded tail
        dict(**base, window_right=0, padded_tail_form=1),  # a causal band
        dict(**base, bias_mode=1, padded_tail_form=1),  # the bias arm (-inf body)
        dict(**base, kv_mask_form=1, padded_tail_form=1),  # the kv_mask arm (-inf body)
        dict(**base, padded_tail_form=2),  # not a form
    ):
        with pytest.raises(ValueError, match="padded_tail_form"):
            make_cfg_d32(TemplateParams(**bad))
    with pytest.raises(ValueError, match="padded_tail_form"):
        make_cfg_d128(TemplateParams(**base, padded_tail_form=1))


_PADDED_TAIL_ADAPTER_CELLS = [
    # (name, adapter kwargs, accept?, the TemplateParams.padded_tail_form of the record)
    ("padded d32, form 1 requested", dict(s_q=217, s_kv=217, seq_kv_lens_present=True, padded_tail_form=1), True, 1),
    ("padded d32, form 0 requested", dict(s_q=217, s_kv=217, seq_kv_lens_present=True, padded_tail_form=0), True, 0),
    ("padded d64 cga2, form 1 requested", dict(d=64, s_q=300, s_kv=300, cga=2, seq_kv_lens_present=True, padded_tail_form=1), True, 1),
    ("dense d32, form 1 requested", dict(padded_tail_form=1), False, None),
    ("padded d32 + bias, form 1 requested", dict(s_q=224, s_kv=224, seq_kv_lens_present=True, bias=(1, 4, 224, 224), padded_tail_form=1), False, None),
    ("padded d32 + kv_mask, form 1 requested", dict(s_q=217, s_kv=217, seq_kv_lens_present=True, kv_mask=(2, 1, 1, 217), padded_tail_form=1), False, None),
    ("padded d32 causal, form 1 requested", dict(s_q=217, s_kv=217, seq_kv_lens_present=True, is_causal=True, padded_tail_form=1), False, None),
    ("padded d128, form 1 requested", dict(d=128, s_q=217, s_kv=217, seq_kv_lens_present=True, padded_tail_form=1), False, None),
    ("padded d32, form 2 requested", dict(s_q=217, s_kv=217, seq_kv_lens_present=True, padded_tail_form=2), False, None),
]


@requires_rubin
@pytest.mark.parametrize("name, kw, accept, record", _PADDED_TAIL_ADAPTER_CELLS, ids=[c[0] for c in _PADDED_TAIL_ADAPTER_CELLS])
def test_adapter_padded_tail_form_requests_are_honoured_or_declined_typed(name, kw, accept, record):
    """A ``padded_tail_form=`` request is honoured on the PLAIN padded small-tile plan (the record carries it) and declined typed
    everywhere else (a dense S_kv, a bias, a kv_mask, a band, the d128 flavor, a value outside 0 / 1) -- never degraded."""
    if not accept:
        with pytest.raises((NotImplementedError, ValueError), match="padded_tail_form"):
            _api2(**kw).check_support()
        return
    api = _api2(**kw)
    assert api.check_support() is True, name
    assert api.template_params().padded_tail_form == record, (name, api.template_params())


@requires_rubin
def test_adapter_padded_tail_form_default_is_the_config_pick():
    """Without a request the adapter's record carries the config pick (the default constant, or the sweep seam when a driver pins it);
    the seam never reaches an ineligible plan."""
    from cudnn.sdpa.fwd import config_sm107 as cfg_mod
    from cudnn.sdpa.fwd.config_sm107 import SM107_PADDED_TAIL_FORM_DEFAULT

    api = _api2(s_q=217, s_kv=217, seq_kv_lens_present=True)
    assert api.check_support() is True
    assert api.template_params().padded_tail_form == SM107_PADDED_TAIL_FORM_DEFAULT
    saved = cfg_mod._PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS
    try:
        cfg_mod._PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS = 1
        api = _api2(s_q=217, s_kv=217, seq_kv_lens_present=True)
        assert api.check_support() is True
        assert api.template_params().padded_tail_form == 1
        api = _api2(s_q=217, s_kv=217, seq_kv_lens_present=True, kv_mask=(2, 1, 1, 217))
        assert api.check_support() is True
        assert api.template_params().padded_tail_form == 0, "the seam must not reach the kv_mask (-inf body) plan"
    finally:
        cfg_mod._PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS = saved


@pytest.mark.parametrize(
    "d, cta, tile_n",
    [(32, 1, 128), (32, 1, 64), (64, 2, 128)],
    ids=["d32_t128", "d32_t64", "d64_cga2_t128"],
)
def test_padded_tail_form_1_rendering_has_one_softmax_body(tmp_path, d, cta, tile_n):
    """The pre-masked tail rendering traced for sm_107a carries ONE softmax body per warp group: its exp count equals the DENSE
    rendering's of the same width (the legacy padded rendering carries twice as many -- the second, software-max body), it has NO
    software row max (FMNMX3 0), the bit-word pre-mask (R2P > 0), the register split, no cluster drain, and spills no more than
    the legacy padded rendering; and it is a different cubin from both."""
    base_arm = {32: "d32_padded_bf16", 64: "d64_padded_bf16_cga2"}[d]
    dense_arm = {32: "d32_dense_bf16", 64: "d64_dense_bf16"}[d]
    legacy = dict(_RECORD_ARMS[base_arm])
    legacy.update({"__d": d, "cta_mma": cta, "dtype_qkv": 2, "tile_n": tile_n})
    dense = dict(_RECORD_ARMS[dense_arm])
    dense.update({"__d": d, "cta_mma": cta, "dtype_qkv": 2, "tile_n": tile_n})
    form1 = dict(legacy)
    form1["padded_tail_form"] = 1
    p_legacy = _probe(tmp_path, legacy, f"tail0_{d}_{cta}_{tile_n}")
    p_dense = _probe(tmp_path, dense, f"dense_{d}_{cta}_{tile_n}")
    p_form1 = _probe(tmp_path, form1, f"tail1_{d}_{cta}_{tile_n}")
    assert len({p_legacy.cubin_md5, p_dense.cubin_md5, p_form1.cubin_md5}) == 3
    assert p_form1.stats["MUFU_EX2"] == p_dense.stats["MUFU_EX2"], (p_form1.stats, p_dense.stats)
    assert p_legacy.stats["MUFU_EX2"] >= 2 * p_dense.stats["MUFU_EX2"] - 8, (p_legacy.stats, p_dense.stats)
    assert p_form1.stats["FMNMX3"] == 0 and p_form1.stats["R2P"] > 0, p_form1.stats
    assert p_form1.stats["USETMAXREG"] > 0 and p_form1.stats["CGAERRBAR"] == 0, p_form1.stats
    assert_no_new_spills(p_form1.stats, p_legacy.stats, tag=f"padded_tail_form 1 d{d} cga{cta} TILE_N {tile_n} vs the legacy tail: ")


# The padded tail cells: a partial 128-wide tail (89 / 72 / 44 / 66 / 88 valid columns of the last tile), per-batch lengths with
# a FULL tile for one entry (the pre-mask must fire on the ragged entry only), a one-tile shape, d64 at cga2, the 64-wide tile.
_PADDED_TAIL_CASES = {
    "tail_217": dict(s=217),
    "tail_200_B2_lens": dict(s=256, kv_lens=[200, 256]),
    "tail_300": dict(s=300, h=2),
    "tail_450": dict(s=450, h=2),
    "tail_one_tile_100": dict(s=100),
    "tail_d64_300_cga2": dict(d=64, s=300, h=2, cga=2),
    "tail_t64_217": dict(s=217, tile_n=64),
    "tail_f16_217": dict(s=217, io=torch.float16),
}


def _executed_padded_tail_form(g, idx):
    """The ``padded_tail_form`` the EXECUTED plan's kernel module was loaded with, read like ``_executed_tile_n`` from the adapter the
    compiled plan carries (the prepared launch spec does not record it); None when the plan exposes no adapter."""
    plan = g._compiled_plans[idx]
    cells = getattr(plan._compiled, "__closure__", None) or ()
    api = next((c.cell_contents for c in cells if type(c.cell_contents).__name__.startswith("SdpaFwdDsl")), None)
    return None if api is None else int(api.template_params().padded_tail_form)


@requires_rubin
@pytest.mark.parametrize("case", list(_PADDED_TAIL_CASES))
def test_padded_tail_form_1_cells_match_the_fp64_oracle_and_the_legacy_tail(case):
    """Every PLAIN padded cell under the pre-masked tail (the sweep seam pins form 1 for the plan): O / Stats against the fp64 oracle,
    keyless rows exactly 0 / -inf, two launches bitwise -- and against the legacy tail form of the same graph within the oracle
    tolerance (the same online softmax over the same -inf-masked scores; the HW and the software row max agree exactly, so the
    forms are expected bitwise -- asserted at the oracle tolerance, the bitwise equality recorded)."""
    from cudnn.sdpa.fwd import config_sm107 as cfg_mod

    c = _PADDED_TAIL_CASES[case]
    io = c.get("io", torch.bfloat16)
    tile_n = c.get("tile_n")
    s = c["s"]
    cc = dict(c)
    cc.pop("tile_n", None)
    cc.pop("io", None)
    if "kv_lens" not in cc:
        cc["kv_lens"] = [s] * 2  # the SAME per-batch lengths the engine synthesises for an S_kv off the tile multiple
    saved = cfg_mod._PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS
    try:
        cfg_mod._PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS = 1
        out1, ref, _ = _run_cell(cc, io, tile_n=tile_n)
        cfg_mod._PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS = 0
        out0, _, _ = _run_cell(cc, io, tile_n=tile_n)
    finally:
        cfg_mod._PADDED_TAIL_FORM_OVERRIDE_FOR_SWEEPS = saved
    _check(out1, ref)
    _check(out0, ref)
    live = ~ref["empty"]
    torch.testing.assert_close(out1["o"][live].double(), out0["o"][live].double(), **_TOL_O)
    torch.testing.assert_close(out1["stats"].squeeze(-1)[live].double(), out0["stats"].squeeze(-1)[live].double(), **_TOL_LSE)
    bitwise = torch.equal(out1["o"], out0["o"]) and torch.equal(out1["stats"], out0["stats"])
    print(f"\n{case}: pre-masked tail vs legacy tail bitwise={bitwise}")
