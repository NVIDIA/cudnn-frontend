# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pins for ``sdpa.softmax_knobs.served_softmax_knob_sets``: the served-domain mirror the cc 10.7
test_mhas_v2 sweeps draw their ``(softmax_precision, attn_scale_prefolded)`` sets from.

A knob set the mirror admits and no FROST row then serves FAILS the sweep (not a skip), so the
mirror must agree with the engine rows (``ENGINE_SPECS``: ``softmax_precisions`` and
``attn_scale_prefolded_d_shapes``), the config backstop tables (``config_sm107``) and the adapter's
flavor tables (``api_dsl``).  Pure python; runs on every lane."""

import ast
import collections
import itertools

import cudnn
import pytest

from sdpa import softmax_knobs as sk
from sdpa.random_config import ExecConfig

pytestmark = pytest.mark.L0

HALF = cudnn.data_type.HALF
DEFAULT, HALF_SET, FOLD_SET, BOTH = (None, False), (HALF, False), (None, True), (HALF, True)
FLAVORS = sorted(sk.EXACT_FLAVORS)
CONFIG_FLAVOR_NAME = {(128, 128): "sm107 d128", (192, 128): "sm107 d192xd128", (256, 256): "sm107 d256", (512, 512): "sm107 d512"}


def _cc107_row(family):
    from cudnn.sdpa.fwd import engines

    name = engines.engine_name(arch="sm107", fp8=family == "fp8", mxfp8=family == "mxfp8")
    return next(spec.capabilities for spec in engines.ENGINE_SPECS if spec.name == name)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_half_row_serves_the_fold_only(flavor):
    assert sk.served_softmax_knob_sets("half", *flavor) == (DEFAULT, FOLD_SET)
    assert sk.served_softmax_knob_sets("half", *flavor, paged=True) == (DEFAULT,), "the paged bodies apply the scale in-kernel"
    expect_thd = (DEFAULT,) if flavor == (192, 128) else (DEFAULT, FOLD_SET)
    assert sk.served_softmax_knob_sets("half", *flavor, thd=True) == expect_thd, "THD (192, 128) is kept FLOAT-only (single-Q leg)"
    assert sk.served_softmax_knob_sets("half", *flavor, thd=True, paged=True) == (DEFAULT,)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_fp8_row_serves_the_half_exponent_only(flavor):
    assert sk.served_softmax_knob_sets("fp8", *flavor) == (DEFAULT, HALF_SET)
    assert sk.served_softmax_knob_sets("fp8", *flavor, thd=True) == (DEFAULT, HALF_SET), "FP8 THD is served on every cc 10.7 flavor"
    assert sk.served_softmax_knob_sets("fp8", *flavor, paged=True) == (DEFAULT,), "no paged KV on the cc 10.7 FP8 row"


@pytest.mark.parametrize("flavor", FLAVORS)
def test_mxfp8_row_serves_all_four_sets_dense_and_the_half_exponent_over_pools(flavor):
    assert sk.served_softmax_knob_sets("mxfp8", *flavor) == (DEFAULT, HALF_SET, FOLD_SET, BOTH)
    assert sk.served_softmax_knob_sets("mxfp8", *flavor, thd=True) == (
        (DEFAULT, HALF_SET, FOLD_SET, BOTH) if flavor == (256, 256) else (DEFAULT,)
    ), "MXFP8 THD on cc 10.7 is the d256 body only (#1488)"
    assert sk.served_softmax_knob_sets("mxfp8", *flavor, thd=True, paged=True) == (DEFAULT,), "no MXFP8 THD over pools"
    assert sk.served_softmax_knob_sets("mxfp8", *flavor, paged=True) == (
        (DEFAULT, HALF_SET) if flavor in ((128, 128), (256, 256)) else (DEFAULT,)
    ), "MXFP8 pools on cc 10.7: the f16x2 exponent composes with the page loader; the fold stays declined over paged KV"


def test_inexact_flavors_and_unknown_families():
    for family in sk.FAMILIES:
        assert sk.served_softmax_knob_sets(family, 64, 64) == (DEFAULT,)
        assert sk.served_softmax_knob_sets(family, 256, 128) == (DEFAULT,)
    with pytest.raises(ValueError, match="family"):
        sk.served_softmax_knob_sets("bf16", 128, 128)


@pytest.mark.parametrize("family", sk.FAMILIES)
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("path", ["dense", "thd", "paged"])
def test_mirror_agrees_with_the_engine_rows_and_config_tables(family, flavor, path):
    """Derive the served set from the SOURCES the mirror hard-codes: the row's softmax_precisions /
    attn_scale_prefolded_d_shapes claim, the path claims (thd / paged_kv + their d-shape sets), the
    config backstop tables and the adapter's flavor tables, plus the two rules only the mirror
    states (per-tensor FP8 never folds -- its row claims None; half THD (192, 128) kept FLOAT-only)."""
    from cudnn.sdpa.fwd import api_dsl, config_sm107

    caps = _cc107_row(family)
    cfg_name = CONFIG_FLAVOR_NAME[flavor] + (" mxfp8" if family == "mxfp8" else "")
    thd, paged = path == "thd", path == "paged"
    path_served = (not thd or (caps.thd and (caps.thd_d_shapes is None or flavor in caps.thd_d_shapes))) and (
        not paged or (caps.paged_kv and (caps.paged_d_shapes is None or flavor in caps.paged_d_shapes))
    )
    half_exp = (
        path_served and HALF in caps.softmax_precisions and cfg_name in config_sm107.SM107_SOFTMAX_F16_FLAVORS and flavor in api_dsl._SM107_HALF_SOFTMAX_FLAVORS
    )
    fold = (
        path_served
        and not paged  # engines.mismatch: "attn_scale_prefolded is not wired in the paged-KV kernel bodies"
        and caps.attn_scale_prefolded_d_shapes is not None
        and flavor in caps.attn_scale_prefolded_d_shapes
        and cfg_name in config_sm107.SM107_SCALE_PREFOLDED_FLAVORS
        and flavor in api_dsl._SM107_PREFOLDED_FLAVORS
        and not (family == "half" and thd and flavor == (192, 128))  # the mirror's simplification of the single-Q leg rule
    )
    expected = [DEFAULT] + ([HALF_SET] if half_exp else []) + ([FOLD_SET] if fold else []) + ([BOTH] if half_exp and fold else [])
    assert sk.served_softmax_knob_sets(family, *flavor, thd=thd, paged=paged) == tuple(expected)
    # The quantized rows gate the paged / THD paths the mirror folds to the default set.
    if family == "fp8":
        assert caps.attn_scale_prefolded_d_shapes is None, "per-tensor FP8 must keep the fold declined (descale contract)"
    if family == "mxfp8":
        assert caps.thd and caps.thd_d_shapes == frozenset({(256, 256)}) and caps.paged_kv and caps.paged_d_shapes == frozenset({(128, 128), (256, 256)})


def test_every_served_set_is_well_formed():
    for family, flavor, thd, paged in itertools.product(sk.FAMILIES, FLAVORS, (False, True), (False, True)):
        served = sk.served_softmax_knob_sets(family, *flavor, thd=thd, paged=paged)
        assert served[0] == sk.DEFAULT_KNOB_SET and sk.is_default_knob_set(served[0])
        assert len(set(served)) == len(served)
        assert all(not sk.is_default_knob_set(s) for s in served[1:])
        assert all(s[0] in (None, HALF) and isinstance(s[1], bool) for s in served)
        if family == "half":
            assert all(s[0] is None for s in served), "half inputs never take the f16x2 exponent"
        if family == "fp8":
            assert all(not s[1] for s in served), "per-tensor FP8 never folds the scale"
        if paged:
            assert all(not s[1] for s in served), "paged bodies apply the scale in-kernel"


@pytest.mark.parametrize("served", [(DEFAULT,), (DEFAULT, FOLD_SET), (DEFAULT, HALF_SET), (DEFAULT, HALF_SET, FOLD_SET, BOTH)])
def test_case_index_assignment_is_uniform_and_rng_free(served):
    n = 3 * len(served)
    counts = collections.Counter(sk.knob_set_for_case(served, i) for i in range(1, n + 1))
    assert counts == {s: 3 for s in served}
    assert sk.knob_set_for_case(served, 1) == DEFAULT, "case 1 always runs the default set"


def test_labels_and_factor():
    assert [sk.knob_set_label(s) for s in (DEFAULT, HALF_SET, FOLD_SET, BOTH)] == ["FLOAT", "HALF", "FLOAT+fold", "HALF+fold"]
    assert sk.knob_set_label((cudnn.data_type.FLOAT, False)) == "FLOAT" and sk.is_default_knob_set((cudnn.data_type.FLOAT, False))
    assert sk.prefold_factor(0.125) == pytest.approx(0.125 * 1.4426950408889634)
    assert sk.LN2 * sk.LOG2E == pytest.approx(1.0)


def test_exec_config_round_trips_the_knob_fields():
    """The repro contract: ExecConfig serializes the two lever fields (the enum through _ENUM_FIELDS)
    and the forward-Stats request, and ``test_repro``'s deserialize path restores them."""
    cfg = ExecConfig(
        data_type=None,
        batches=2,
        h_q=4,
        h_k=4,
        h_v=4,
        s_q=64,
        s_kv=64,
        d_qk=128,
        d_v=128,
        softmax_precision=HALF,
        attn_scale_prefolded=True,
        fwd_stats=False,
        bshd_layout=True,
    )
    cfg.fill_derived_fields()
    serialized = cfg.serialize()
    assert serialized["softmax_precision"] == "cudnn.data_type.HALF" and serialized["attn_scale_prefolded"] is True and serialized["fwd_stats"] is False
    assert serialized["bshd_layout"] is True
    back = ExecConfig.deserialize(ast.literal_eval(repr(serialized)))
    assert back.softmax_precision == HALF and back.attn_scale_prefolded is True and back.fwd_stats is False and back.bshd_layout is True
    repro = cfg.to_repro_cmd("test_mhas_v2.py")
    assert "'softmax_precision': 'cudnn.data_type.HALF'" in repro and "'attn_scale_prefolded': True" in repro
    # The defaults round-trip as the default set (None / False / None), so an old repro string without the fields still deserializes.
    plain = ExecConfig.deserialize({"batches": 1, "h_q": 1, "h_k": 1, "h_v": 1, "s_q": 1, "s_kv": 1, "d_qk": 128, "d_v": 128})
    assert (plain.softmax_precision, plain.attn_scale_prefolded, plain.fwd_stats) == (None, False, None)
