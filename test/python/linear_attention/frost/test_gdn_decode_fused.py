# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused GDN decode step (``cudnn.linear_attention.frost.gdn_decode_fused``): the typed declines of its host entry (host-only,
CPU tensors), the plan-time-only compile key, the config knob validation (the occupancy pad and the co-residency on every part reading), the DSL-floor
decline, and -- on a cc 10.x or 12.x device -- the null-slot contract, the reader tickets, determinism, CUDA-graph replay over distinct
layers, a loose torch reference of the fused math, and the bitwise agreement of every knob arm the part's shared memory holds (the
shared-memory state path, the streamed two-CTAs-per-SM form, the PDL order, the TPR mappings, the 128-thread form, the tap-contiguous
conv-state layout) with the entry's default pick on one operand set that carries a null row and non-monotone slots -- an arm the part
does not hold is its typed decline, reported as a skip.  The default's per-rows pick (the pair, one CTA per SM while the rows x 48 grid
fits the SMs, past that the capped two-CTAs-per-SM form where two of its CTAs share an SM, else the one-CTA form in two waves) is pinned
on the host for both validated parts' readings over several SM counts and on the device.  The geometry (``Geometry(K_HEADS, V_HEADS,
HIDDEN)`` of ``kernel/gdn_geometry.py``) is pinned on the host too: the default geometry's constants and forms unchanged, and for the
per-rank geometries of the larger dense models (1 K head / 8 V heads / hidden 8192 at 16-way tensor parallelism, 2 / 16 / 8192, 4 / 32 /
8192, 4 / 16 / 4096) the derived constants, the operand shapes the entry accepts, the per-geometry pick ladder, the entry's forms and keys,
and the BA partition's coverage of every hidden index -- and, on the device, the torch model and determinism per rows, every form family
bitwise with the one-CTA form, and batch invariance at each of those geometries."""

import math
from collections import Counter
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.linear_attention.frost import gdn_decode_fused as entry  # noqa: E402
from cudnn.linear_attention.frost.kernel import gdn_decode_fused_config as kcfg  # noqa: E402
from cudnn.linear_attention.frost.kernel import gdn_geometry  # noqa: E402

pytestmark = [pytest.mark.L0]

ROWS = 2
SLOTS = ROWS + 1
# the per-rank geometries of the larger dense models: (K heads, V heads, hidden) at 16- / 8- / 4-way tensor parallelism of a hidden-8192
# model and at 4- / 8-way of a hidden-4096 one (8-way = two trays of four GPUs serving the 397B-class checkpoint: the rehearsal geometry)
RANK_GEOMETRIES = [
    gdn_geometry.Geometry(1, 8, 8192),
    gdn_geometry.Geometry(2, 16, 8192),
    gdn_geometry.Geometry(4, 32, 8192),
    gdn_geometry.Geometry(4, 16, 4096),
    gdn_geometry.Geometry(2, 8, 4096),
]


def _gid(g):
    return f"k{g.K_HEADS}v{g.V_HEADS}h{g.HIDDEN}"


# the cc 10.0 part reading (228 KB per SM, 227 KB opt-in), a hypothetical larger part (336 KiB per SM, 335 KiB opt-in) and the cc 12.0 part
# reading (100 KB per SM, 99 KB opt-in): the pad must do what it claims on each, and a form two of which cannot share the SM is a typed
# decline where the caller asks for two
SM100_READING = (kcfg.SM100_SMEM_PER_SM_BYTES, kcfg.SM100_SMEM_OPTIN_BYTES)  # 233,472 / 232,448
SM120_READING = (kcfg.SM120_SMEM_PER_SM_BYTES, kcfg.SM120_SMEM_OPTIN_BYTES)  # 102,400 / 101,376
PART_READINGS = [SM100_READING, (344064, 343040), SM120_READING]


def _on(reading):
    """The ``Cfg`` fields of a part reading ``(shared memory per SM, per-CTA opt-in)``."""
    return dict(SMEM_PER_SM_BYTES=reading[0], SMEM_OPTIN_BYTES=reading[1])


def make_operands(rows=ROWS, device="cpu", slots=SLOTS, seed=0, geometry=None, **overrides):
    """The entry's operand set in the serving layout at ``geometry`` (the default geometry when None: x = the QKV part of a [rows, 16384]
    slab, z its Z part viewed [rows, 48, 128], cs the [slots, 3, 10240] conv buffer transposed to [slots, 10240, 3]; the same draws in the
    same order as before the geometry was a parameter, so every seeded cell keeps its operands)."""
    geo = geometry or kcfg.DEFAULT_GEOMETRY
    QKV, NV, HID = geo.QKV_CHANNELS, geo.V_HEADS, geo.HIDDEN
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g).to(device)

    qkvz = (rn(rows, QKV + geo.OUT_ROW) * 2).bfloat16()
    conv = rn(slots, 3, QKV).bfloat16()
    ops = dict(
        rows=rows,
        x=qkvz[:, :QKV],
        cs=conv.transpose(-1, -2),
        cw=(0.3 * rn(QKV, 4)).bfloat16(),
        al=rn(NV) * 0.5,
        dt=(rn(NV) * 0.5).bfloat16(),
        state=rn(slots, NV, 128, 128) * 0.1,
        idx=torch.arange(1, rows + 1, dtype=torch.int32).to(device),
        z=qkvz[:, QKV:].view(rows, NV, 128),
        w=(1 + 0.2 * rn(128)).bfloat16(),
        scale=torch.tensor([0.02]).to(device),
        hidden=rn(rows, HID).bfloat16(),
        baw=(rn(2 * NV, HID) / math.sqrt(HID)).bfloat16(),
    )
    ops.update(overrides)
    return ops


def _binding_args(ops):
    """The entry's positional operands: ``(rows, x, cs, cw, al, dt, state, idx, z, w, scale, hidden, baw)``."""
    return tuple(ops[name] for name in ("rows", "x", "cs", "cw", "al", "dt", "state", "idx", "z", "w", "scale", "hidden", "baw"))


def call_check(ops, device_cc=(10, 0)):
    return entry.check_support(*_binding_args(ops), device_cc=device_cc)


# ---------------------------------------------------------------------------------------------------------------------------
# host-only
# ---------------------------------------------------------------------------------------------------------------------------
def test_served_geometry_passes_the_host_checks():
    call_check(make_operands())


def test_default_cfg_is_the_measured_pick_per_rows():
    """The three forms of the default (measured on a cc 10.0 part with 148 SMs, every form bitwise with the reference kernel): the
    two-CTAs-per-head pair while ``rows x 96 <= SM count`` (the fastest form at rows 1), one CTA per SM while ``rows x 48 <= SM count``
    and the register-resident form under the 128-register cap past that (two CTAs per SM, one wave; ahead of the streamed form); every
    form bulk, TPR 4, the single traced tail.  PDL stays off in all (the caller's stream contract)."""
    cfg = kcfg.CFG
    assert (cfg.THREADS, cfg.TPR, cfg.STATE_LOAD_PATH, cfg.STATE_STREAM, cfg.MIN_CTAS_PER_SM, cfg.CTAS_PER_SM) == (256, 4, "bulk", False, 1, 1)
    assert (cfg.PDL, cfg.PDL_RELEASE, cfg.PIN_ENTRY_LOADS, cfg.K_RESIDENT, cfg.WARPGROUP_SPLIT) == (False, False, False, True, True)
    assert (cfg.SINGLE_TAIL, cfg.CONV_FIRST, cfg.STORE_PIPELINE, cfg.TWO_CTAS_PER_HEAD) == (True, False, False, False)
    assert (cfg.R, cfg.C, cfg.RG, cfg.BOX_ROWS) == (2, 32, 8, 16)
    pair = kcfg.PAIR_CFG
    assert pair == replace(cfg, TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0) and pair.SMEM_PAD_BYTES == 0 and pair.CTA_ROWS == 64
    capped = kcfg.CAPPED_CFG
    assert capped == replace(cfg, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0) and capped.SMEM_PAD_BYTES == 0 and not capped.STATE_STREAM
    streamed = kcfg.STREAMED_CFG
    assert streamed == replace(capped, STATE_STREAM=True)
    assert all(form.PDL is False for form in (pair, cfg, capped, streamed))
    assert kcfg.make_cfg() == cfg


@pytest.mark.parametrize("sm_count", (148, 152))
def test_default_form_switches_where_the_grid_outgrows_the_sms(sm_count):
    """96 cluster CTAs at rows 1 fit 148 / 152 SMs (the pair), 192 at rows 2 do not; ``rows x 48`` CTAs: 144 at rows 3 fit (one CTA
    per SM), 192 at rows 4 do not (the capped two-CTAs-per-SM form)."""
    assert kcfg.default_cfg(1, sm_count) == kcfg.PAIR_CFG
    for rows in (2, 3):
        assert kcfg.default_cfg(rows, sm_count) == kcfg.CFG, (rows, sm_count)
    for rows in (4, 8, 64):
        assert kcfg.default_cfg(rows, sm_count) == kcfg.CAPPED_CFG, (rows, sm_count)
    pairs = sm_count // (2 * kcfg.V_HEADS)  # the last rows count whose cluster grid has an SM per CTA
    fits = sm_count // kcfg.V_HEADS  # the last rows count whose one-CTA grid fits
    assert kcfg.default_cfg(pairs, sm_count) == kcfg.PAIR_CFG and kcfg.default_cfg(pairs + 1, sm_count) == kcfg.CFG
    assert kcfg.default_cfg(fits, sm_count) == kcfg.CFG and kcfg.default_cfg(fits + 1, sm_count) == kcfg.CAPPED_CFG
    assert kcfg.default_cfg(2, 192) == kcfg.PAIR_CFG and kcfg.default_cfg(3, 192) == kcfg.CFG  # a 192-SM part would pair rows 2 too
    with pytest.raises(ValueError, match="rows"):
        kcfg.default_cfg(0, sm_count)
    with pytest.raises(ValueError, match="SM count"):
        kcfg.default_cfg(1, 0)


def test_default_cfg_on_the_cc100_reading_is_the_measured_pick_at_every_sm_count():
    """The cc 10.0 readings, explicit and as the defaults: the capped form IS two per SM there (three footprints by shared memory, two by
    the cap), so the pick is PAIR_CFG / CFG / CAPPED_CFG by the SM-count rule alone at every rows count -- the byte-identity pin of the
    budget-aware pick with the measured cc 10.0 pick, over six SM counts."""
    per_sm, optin = SM100_READING
    for sms in (84, 148, 152, 170, 188):
        for rows in range(1, 17):
            want = kcfg.PAIR_CFG if rows * 96 <= sms else (kcfg.CFG if rows * 48 <= sms else kcfg.CAPPED_CFG)
            assert kcfg.default_cfg(rows, sms) == kcfg.default_cfg(rows, sms, per_sm, optin) == want, (sms, rows)
    assert (
        kcfg.default_cfg(2, 192) == kcfg.PAIR_CFG and kcfg.default_cfg(3, 192) == kcfg.CFG and kcfg.default_cfg(5, 192) == kcfg.CAPPED_CFG
    )  # a 192-SM part would pair rows 2
    assert kcfg.default_cfg(1, 84) == kcfg.CFG and kcfg.default_cfg(2, 84) == kcfg.CAPPED_CFG  # 84 SMs hold no 96-CTA cluster grid
    assert kcfg.first_fitting(kcfg.ONE_CTA_FORMS) == kcfg.CFG and kcfg.first_fitting(kcfg.BEYOND_FORMS, per_sm, optin) == kcfg.CAPPED_CFG
    assert kcfg.admissible(kcfg.CAPPED_CFG) and kcfg.admissible(kcfg.STREAMED_CFG, per_sm, optin) and kcfg.admissible(kcfg.PAIR_CFG, per_sm, optin)
    assert (kcfg.PAIR_FORMS, kcfg.ONE_CTA_FORMS, kcfg.BEYOND_FORMS) == ((kcfg.PAIR_CFG,), (kcfg.CFG,), (kcfg.CAPPED_CFG, kcfg.CFG))
    assert kcfg.part_readings(per_sm, optin) == _on(SM100_READING)


def test_default_cfg_on_the_cc120_reading_never_picks_a_cap_the_part_cannot_use():
    """The cc 12.0 readings (100 KB per SM, 99 KB opt-in) on 170 / 188 SMs: rows 1 the pair, rows 2 and up the one-CTA form -- two waves
    from rows 4 rather than CAPPED_CFG / STREAMED_CFG, whose cap keeps nothing co-resident on a 100 KB SM and spills; on 84 SMs the
    one-CTA form at every rows count; a two-per-SM occupancy pad (``CTAS_PER_SM`` 2, valid on the 228 KB readings) is inadmissible there by
    the validator's fit term; a part no form fits is the typed decline naming its opt-in."""
    per_sm, optin = SM120_READING
    for sms in (170, 188):
        assert kcfg.default_cfg(1, sms, per_sm, optin) == kcfg.PAIR_CFG
        for rows in range(2, 65):
            assert kcfg.default_cfg(rows, sms, per_sm, optin) == kcfg.CFG, (sms, rows)
    for rows in range(1, 17):
        assert kcfg.default_cfg(rows, 84, per_sm, optin) == kcfg.CFG, rows
    assert not kcfg.admissible(kcfg.CAPPED_CFG, per_sm, optin) and not kcfg.admissible(kcfg.STREAMED_CFG, per_sm, optin)
    assert kcfg.admissible(kcfg.CFG, per_sm, optin) and kcfg.admissible(kcfg.PAIR_CFG, per_sm, optin)
    assert kcfg.admissible(
        kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, MIN_CTAS_PER_SM=2), per_sm, optin
    ), "the one capped form two of which share a 100 KB SM"
    assert not kcfg.admissible(kcfg.make_cfg(CTAS_PER_SM=2), per_sm, optin), "two whole-slab footprints plus the reserves exceed the 100 KB SM"
    assert kcfg.first_fitting(kcfg.BEYOND_FORMS, per_sm, optin) == kcfg.CFG
    for call in (
        lambda: kcfg.default_cfg(1, 170, 40960, 40000),
        lambda: kcfg.default_cfg(4, 170, 40960, 40000),
        lambda: kcfg.first_fitting(kcfg.ONE_CTA_FORMS, 40960, 40000),
    ):
        with pytest.raises(NotImplementedError, match="40000-byte shared-memory opt-in per CTA .* holds no decode form"):
            call()
    with pytest.raises(ValueError, match="cannot exceed the SM"):
        kcfg.default_cfg(1, 170, 100000, 200000)
    with pytest.raises(ValueError, match="smem_optin"):
        kcfg.default_cfg(1, 170, per_sm, 0)


def test_cudnn_level_export_resolves_to_the_entry():
    assert cudnn.GdnDecodeFused is entry.GdnDecodeFused, "the lazy cudnn-level export (import cudnn; cudnn.GdnDecodeFused) is the entry class"
    from cudnn.linear_attention import frost

    assert frost.GdnGeometry is gdn_geometry.Geometry, "the geometry type is exported beside the entries"


# ---------------------------------------------------------------------------------------------------------------------------
# the geometry: the default unchanged, the per-rank geometries derived, admitted and keyed
# ---------------------------------------------------------------------------------------------------------------------------
def test_the_default_geometry_is_the_validated_one_and_its_constants_are_unchanged():
    """The default ``Geometry`` is the 16 K / 48 V / hidden 5120 one every form was validated at; the module constants name it; every
    module form carries it (the SAME object: a default-geometry pick is the module form itself, so the forms cache and the compile keys of
    existing callers are untouched); head dim and conv width are pinned."""
    g = kcfg.DEFAULT_GEOMETRY
    assert g is gdn_geometry.DEFAULT_GEOMETRY and (g.K_HEADS, g.V_HEADS, g.HIDDEN) == (16, 48, 5120) == (kcfg.K_HEADS, kcfg.V_HEADS, kcfg.HIDDEN)
    assert (kcfg.QKV_CHANNELS, kcfg.K_CHANNEL0, kcfg.V_CHANNEL0, kcfg.BA_CHUNKS, kcfg.TICKET_LIMIT, kcfg.TICKET_STRIDE) == (10240, 2048, 4096, 5, 2, 512)
    assert (g.QKV_CHANNELS, g.K_CHANNEL0, g.V_CHANNEL0, g.OUT_ROW, g.VK_RATIO, g.BA_CHUNKS_DECODE, g.BA_CHUNKS_VERIFY) == (10240, 2048, 4096, 6144, 3, 5, 3)
    assert g.BA_VERIFY_HALF_LIVE == (True, True, False) and g.TICKET_STRIDE == 512
    assert (kcfg.HEAD_DIM, kcfg.CONV_WIDTH) == (gdn_geometry.HEAD_DIM, gdn_geometry.CONV_WIDTH) == (128, 4)
    for form in (kcfg.CFG, kcfg.PAIR_CFG, kcfg.CAPPED_CFG, kcfg.STREAMED_CFG, kcfg.make_cfg(), kcfg.Cfg()):
        assert form.GEOM is g
    assert (kcfg.CFG.TICKET_LIMIT, kcfg.PAIR_CFG.TICKET_LIMIT) == (2, 5)
    assert kcfg.default_cfg(1, 148) is kcfg.PAIR_CFG and kcfg.default_cfg(2, 148, geometry=g) is kcfg.CFG and kcfg.default_cfg(4, 148) is kcfg.CAPPED_CFG
    assert kcfg.with_geometry(kcfg.CFG, g) is kcfg.CFG
    assert g.describe() == "16 K heads, 48 V heads x 128, hidden 5120"


@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_a_rank_geometry_derives_its_constants_and_moves_nothing_but_the_geometry(g):
    """The derived constants of a per-rank geometry (the QKV channel layout, the output row, the V-per-K ratio, the BA chunk counts, the
    ticket limits) follow from (K, V, HIDDEN); a form compiled for it differs from the module form in ``GEOM`` alone -- the shared-memory
    budget, the lane map, the box pad and the co-residency carry no head-count or hidden term -- and is a distinct compile key."""
    K, V, H = g.K_HEADS, g.V_HEADS, g.HIDDEN
    assert (g.QKV_CHANNELS, g.K_CHANNEL0, g.V_CHANNEL0, g.OUT_ROW) == (2 * K * 128 + V * 128, K * 128, 2 * K * 128, V * 128)
    assert g.VK_RATIO * K == V and g.BA_CHUNKS_DECODE * 1024 == H and g.BA_CHUNKS_VERIFY == -(-H // 2048)
    assert g.TICKET_STRIDE == 512 and K * 32 <= 512, "the reader-ticket layout counts[row * 512 + kh * 32] holds every K head"
    assert g.describe() == f"{K} K heads, {V} V heads x 128, hidden {H}"
    cfg, pair, capped = (kcfg.with_geometry(f, g) for f in (kcfg.CFG, kcfg.PAIR_CFG, kcfg.CAPPED_CFG))
    assert cfg == replace(kcfg.CFG, GEOM=g) and cfg.GEOM is g and cfg is not kcfg.CFG and cfg != kcfg.CFG and hash(cfg) != hash(kcfg.CFG)
    assert (cfg.TICKET_LIMIT, pair.TICKET_LIMIT, capped.TICKET_LIMIT) == (g.VK_RATIO - 1, 2 * g.VK_RATIO - 1, g.VK_RATIO - 1)
    for a, b in ((cfg, kcfg.CFG), (pair, kcfg.PAIR_CFG), (capped, kcfg.CAPPED_CFG)):
        kcfg.validate_cfg(a)
        same = (
            "NATURAL_SMEM_BYTES",
            "SMEM_BYTES",
            "SMEM_PAD_BYTES",
            "R",
            "C",
            "RG",
            "BOX_ROWS",
            "STATE_BOX_PITCH_ELEMS",
            "STATE_SLAB_BYTES_CTA",
            "co_resident_ctas",
        )
        assert [getattr(a, n) for n in same] == [getattr(b, n) for n in same], b
    assert kcfg.make_cfg(GEOM=g).GEOM is g and kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, GEOM=g) == pair
    assert kcfg.admissible(cfg) and kcfg.admissible(pair) and kcfg.first_fitting((cfg,)) is cfg


def test_geometry_envelope_declines_by_field():
    """Outside the envelope (a non-positive count, more than 16 K heads, more than 128 V heads, V heads not a multiple of the K heads, a
    hidden width that is not a multiple of 1024) is a ``ValueError`` naming the field at construction; the field-wise form of the same
    answer (``geometry_fields_off``) is what the FlashInfer backend's decline names; a config or pick handed something else than a
    ``Geometry`` is a ``ValueError`` too."""
    for bad, token in (
        ((0, 8, 8192), "K_HEADS must be a positive int"),
        ((17, 17, 8192), "K_HEADS=17 exceeds the 16 K heads"),
        ((1, 129, 8192), "V_HEADS=129 exceeds"),
        ((3, 8, 8192), "V_HEADS=8 must be a multiple of K_HEADS=3"),
        ((1, 8, 5000), "HIDDEN=5000 must be a multiple of 1024"),
        ((1, 8, 0), "HIDDEN must be a positive int"),
        ((True, 8, 8192), "K_HEADS must be a positive int"),
        ((1, 8.0, 8192), "V_HEADS must be a positive int"),
    ):
        with pytest.raises(ValueError, match=token):
            gdn_geometry.Geometry(*bad)
    off = gdn_geometry.geometry_fields_off
    assert off(16, 48, 5120) == {} and off(1, 8, 8192) == {} and off(16, 128, 8192) == {}
    assert off(17, 17, 8192) == {"K_HEADS": 17} and off(1, 129, 8192) == {"V_HEADS": 129} and off(3, 8, 8192) == {"V_HEADS": 8}
    assert off(1, 8, 5000) == {"HIDDEN": 5000} and off(0, 8, 5000) == {"K_HEADS": 0, "HIDDEN": 5000} and off(2, 3, 1023) == {"V_HEADS": 3, "HIDDEN": 1023}
    assert gdn_geometry.geometry_error(1, 8, 8192) is None and "1024" in gdn_geometry.geometry_error(1, 8, 1000)
    assert (gdn_geometry.MAX_K_HEADS, gdn_geometry.MAX_V_HEADS, gdn_geometry.HIDDEN_GRANULE) == (16, 128, 1024)
    with pytest.raises(ValueError, match="GEOM"):
        kcfg.make_cfg(GEOM="k1v8h8192")
    with pytest.raises(ValueError, match="geometry"):
        kcfg.default_cfg(1, 148, geometry=(1, 8, 8192))
    with pytest.raises(ValueError, match="geometry"):
        kcfg.with_geometry(kcfg.CFG, None)


@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_default_cfg_ladder_follows_the_geometry_v_heads(g):
    """The pick's regime switches move with the V heads per rank by the one rule (the pair while ``rows x 2 V <= SMs``, one CTA per SM while
    ``rows x V <= SMs``, the capped form past that where the part holds two; the budget rule, unmeasured at these geometries until their
    own cells run), every pick carrying the geometry; on the cc 12.0 readings the cap is never picked, as at the default geometry."""
    V = g.V_HEADS
    for sms in (148, 152, 170, 188):
        for rows in range(1, 41):
            want = kcfg.PAIR_CFG if rows * 2 * V <= sms else (kcfg.CFG if rows * V <= sms else kcfg.CAPPED_CFG)
            got = kcfg.default_cfg(rows, sms, geometry=g)
            assert got == replace(want, GEOM=g) and got.GEOM is g, (sms, rows)
            assert kcfg.default_cfg(rows, sms, *SM100_READING, g) == got
    pairs, fits = 152 // (2 * V), 152 // V
    assert kcfg.default_cfg(pairs, 152, geometry=g).TWO_CTAS_PER_HEAD and not kcfg.default_cfg(pairs + 1, 152, geometry=g).TWO_CTAS_PER_HEAD
    assert kcfg.default_cfg(fits, 152, geometry=g).MIN_CTAS_PER_SM == 1 and kcfg.default_cfg(fits + 1, 152, geometry=g).MIN_CTAS_PER_SM == 2
    per_sm, optin = SM120_READING
    assert kcfg.default_cfg(fits + 1, 170, per_sm, optin, g) == replace(kcfg.CFG, GEOM=g) and kcfg.default_cfg(1, 170, per_sm, optin, g) == replace(
        kcfg.PAIR_CFG, GEOM=g
    )


@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_host_checks_accept_the_rank_geometry_operands_and_decline_a_mismatch(g):
    """The entry's binding checks at a per-rank geometry: its operand shapes pass (the QKV slab width, the state slab per V head, the BA
    weight's two rows per V head over the hidden width), the default geometry's checks refuse them by name and vice versa, a hidden or
    BA weight of another width is declined by name, and a geometry that is not a ``Geometry`` is a ``TypeError``."""
    ops = make_operands(rows=3, geometry=g)
    assert (
        tuple(ops["x"].shape) == (3, g.QKV_CHANNELS) and tuple(ops["z"].shape) == (3, g.V_HEADS, 128) and tuple(ops["baw"].shape) == (2 * g.V_HEADS, g.HIDDEN)
    )
    assert tuple(ops["state"].shape[1:]) == (g.V_HEADS, 128, 128) and tuple(ops["state"].stride()) == (g.V_HEADS * 16384, 16384, 128, 1)
    assert ops["x"].stride(0) == g.QKV_CHANNELS + g.OUT_ROW and tuple(ops["cs"].shape[1:]) == (g.QKV_CHANNELS, 3)
    entry.check_support(*_binding_args(ops), device_cc=(10, 0), geometry=g)
    entry.check_binding(*_binding_args(ops), geometry=g)
    with pytest.raises(NotImplementedError, match="x as"):
        entry.check_support(*_binding_args(ops), device_cc=(10, 0))  # the default geometry's checks refuse the rank's slab width
    with pytest.raises(NotImplementedError, match="x as"):
        entry.check_support(*_binding_args(make_operands(rows=3)), device_cc=(10, 0), geometry=g)
    with pytest.raises(NotImplementedError, match="hidden"):
        entry.check_support(*_binding_args(dict(ops, hidden=torch.zeros(3, g.HIDDEN + 1024, dtype=torch.bfloat16))), device_cc=(10, 0), geometry=g)
    with pytest.raises(NotImplementedError, match="BA weight"):
        entry.check_support(*_binding_args(dict(ops, baw=torch.zeros(2 * g.V_HEADS, 5120, dtype=torch.bfloat16))), device_cc=(10, 0), geometry=g)
    with pytest.raises(NotImplementedError, match="A_log"):
        entry.check_support(*_binding_args(dict(ops, al=torch.zeros(g.V_HEADS - 1))), device_cc=(10, 0), geometry=g)
    with pytest.raises(TypeError, match="geometry"):
        entry.check_support(*_binding_args(ops), device_cc=(10, 0), geometry=(g.K_HEADS, g.V_HEADS, g.HIDDEN))


@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_entry_takes_the_geometry_and_keys_every_form_by_it(g):
    """An entry built for a per-rank geometry stamps it into every form it picks (the pick ladder at the geometry's V heads), sizes its
    output buffers by it, carries it in every compile key (a default-geometry entry's key differs), takes it from a given ``cfg`` when
    none is named, refuses a ``cfg`` and a ``geometry`` that disagree, and keeps the PDL launch knobs on top."""
    V = g.V_HEADS
    k = entry.GdnDecodeFused("cpu", 1e-6, geometry=g)
    assert k.geometry is g and k.cfg == replace(kcfg.CFG, GEOM=g) and k.cfg_for(1).GEOM is g and k.cfg_for(1) == k.cfg_for(64)
    k.sm_count = 152
    assert k.cfg_for(1) == replace(kcfg.PAIR_CFG, GEOM=g) and k.cfg_for(152 // (2 * V) + 1) == replace(kcfg.CFG, GEOM=g)
    assert k.cfg_for(152 // V + 1) == replace(kcfg.CAPPED_CFG, GEOM=g) and k.cfg_for(152 // V + 1) is k.cfg_for(
        152 // V + 2
    ), "one stamped form per module form"
    out = k.outputs(2)
    assert tuple(out["qout"].shape) == (2, V, 128) == tuple(out["out"].shape) and out["counts"].shape == (2 * kcfg.TICKET_STRIDE,)
    dbg = entry.GdnDecodeFused("cpu", 1e-6, geometry=g, debug=True).outputs(2)
    assert tuple(dbg["raw"].shape) == (2, V, 128) and tuple(dbg["convout"].shape) == (2, g.QKV_CHANNELS) and tuple(dbg["baout"].shape) == (2, 2 * V)
    ops, default_ops = make_operands(rows=2, geometry=g), make_operands(rows=2)
    default = entry.GdnDecodeFused("cpu", 1e-6)
    assert default.geometry is kcfg.DEFAULT_GEOMETRY
    key = k.key(2, ops["x"], ops["cs"], ops["state"], ops["z"])
    assert key[1].GEOM is g and key != default.key(2, default_ops["x"], default_ops["cs"], default_ops["state"], default_ops["z"])
    given = kcfg.make_cfg(TPR=8, GEOM=g)
    assert entry.GdnDecodeFused("cpu", 1e-6, cfg=given).geometry is g and entry.GdnDecodeFused("cpu", 1e-6, cfg=given, geometry=g).cfg == given
    assert entry.GdnDecodeFused("cpu", 1e-6, cfg=kcfg.make_cfg(TPR=8)).geometry is kcfg.DEFAULT_GEOMETRY
    with pytest.raises(ValueError, match="compiled for"):
        entry.GdnDecodeFused("cpu", 1e-6, cfg=given, geometry=kcfg.DEFAULT_GEOMETRY)
    with pytest.raises(TypeError, match="geometry"):
        entry.GdnDecodeFused("cpu", 1e-6, geometry="k1v8h8192")
    pdl = entry.GdnDecodeFused("cpu", 1e-6, geometry=g, pdl=True)
    pdl.sm_count = 152
    assert pdl.cfg_for(1) == replace(kcfg.PAIR_CFG, GEOM=g, PDL=True, PDL_RELEASE=True)


@pytest.mark.parametrize("g", [kcfg.DEFAULT_GEOMETRY] + RANK_GEOMETRIES, ids=_gid)
def test_decode_ba_partition_covers_every_hidden_index_once(g):
    """The reference 128-thread BA partition the kernel loads and accumulates (``_ba_weight_loads`` / ``_ba_hidden_loads`` / ``_ba_dots``:
    thread ``t`` of 128, chunk ``c`` of ``BA_CHUNKS_DECODE`` -> elements ``c * 1024 + 8 t + j``) covers ``0 .. HIDDEN - 1`` exactly once at
    every served hidden width (a pure-Python mirror of the kernel's index arithmetic, no GPU)."""
    seen = Counter()
    for t in range(128):
        for c in range(g.BA_CHUNKS_DECODE):
            kk = c * 1024 + t * 8
            seen.update(range(kk, kk + 8))
    assert seen == Counter(range(g.HIDDEN)), g


def test_rows_are_bounded_by_the_int32_row_address_headroom_at_wide_geometries():
    """The kernel forms its row addresses in 32-bit arithmetic: ``check_binding`` declines, by type and naming the geometry, a rows count
    whose widest row stride overflows 2^31 elements -- below ``MAX_ROWS`` only off the default geometry.  At the default geometry
    (QKVZ row 16384) the bound is ``MAX_ROWS`` itself; at 16 K / 128 V heads with hidden 8192 and a fused QKVZ view (row 36864) it is 58254;
    on contiguous operands it drops below ``MAX_ROWS`` only once the hidden width passes 32768 (63550 at 33792, the geometry-only bound
    FlashInfer's probe answers).  The operands at the bound live on the ``meta`` device: ``check_binding`` reads shape, stride, dtype,
    device and alignment only (a ``meta`` tensor's ``data_ptr`` is 0), while a real tensor at the bound holds 2^31 elements -- torch's
    ``as_strided`` refuses a view past its storage -- 4 GiB this host tier has no use for."""
    d = kcfg.DEFAULT_GEOMETRY
    assert entry.max_rows_for(d.QKV_CHANNELS + d.OUT_ROW, d.QKV_CHANNELS + d.OUT_ROW, d) == entry.MAX_ROWS == 65535
    wide = gdn_geometry.Geometry(16, 128, 8192)
    slab = wide.QKV_CHANNELS + wide.OUT_ROW  # 20480 + 16384
    assert slab == 36864 and entry.max_rows_for(slab, slab, wide) == (2**31 - 1) // 36864 == 58254
    assert (
        (2**31 - 1) // wide.QKV_CHANNELS == 104857 > entry.MAX_ROWS == entry.max_rows_for(wide.QKV_CHANNELS, wide.OUT_ROW, wide)
    ), "contiguous x / z: the grid-Y cap binds first"
    deep = gdn_geometry.Geometry(1, 8, 33792)  # the hidden row is the widest term: the geometry-only bound on contiguous operands
    assert entry.max_rows_for(deep.QKV_CHANNELS, deep.OUT_ROW, deep) == (2**31 - 1) // 33792 == 63550 < entry.MAX_ROWS
    edge = gdn_geometry.Geometry(1, 8, 32768)
    assert entry.max_rows_for(edge.QKV_CHANNELS, edge.OUT_ROW, edge) == entry.MAX_ROWS, "32768 is the last hidden width the grid-Y cap binds at"

    def meta(*shape):
        return torch.empty(*shape, dtype=torch.bfloat16, device="meta")

    rows = entry.max_rows_for(slab, slab, wide) + 1
    ops = make_operands(rows=1, geometry=wide, device="meta")
    # the row count claims rows rows of the (one-row) operands: the stride check comes first, so size the strided views to hold them
    x = meta(rows, slab)
    ops.update(
        rows=rows,
        x=x[:, : wide.QKV_CHANNELS],
        z=x[:, wide.QKV_CHANNELS :].view(rows, wide.V_HEADS, 128),
        idx=torch.ones(rows, dtype=torch.int32, device="meta"),
        hidden=meta(rows, wide.HIDDEN),
    )
    with pytest.raises(NotImplementedError, match="32-bit arithmetic: at most 58254 rows .* 36864 / 36864 .* hidden 8192; got rows=58255"):
        entry.check_binding(*_binding_args(ops), geometry=wide)
    ops["rows"] = rows - 1
    entry.check_binding(*_binding_args(ops), geometry=wide)  # the bound itself is served
    rows = 63551
    ops = make_operands(rows=1, geometry=deep, device="meta")
    ops.update(rows=rows, x=meta(rows, deep.QKV_CHANNELS), z=meta(rows, deep.V_HEADS, 128), idx=torch.ones(rows, dtype=torch.int32, device="meta"))
    with pytest.raises(NotImplementedError, match="at most 63550 rows for x / z row strides 1280 / 1024 .* hidden 33792; got rows=63551"):
        entry.check_binding(*_binding_args(ops), geometry=deep)
    ops.update(rows=rows - 1, hidden=meta(rows - 1, deep.HIDDEN))
    entry.check_binding(*_binding_args(ops), geometry=deep)  # 63550 rows of contiguous operands are served


def test_geometry_envelope_takes_any_integral_and_holds_plain_ints():
    """A numpy integer (a config's field) constructs the same Geometry as the Python int; a bool never does; the fields are plain ints."""
    np = pytest.importorskip("numpy")
    g = gdn_geometry.Geometry(np.int64(1), np.int32(8), np.int64(8192))
    assert g == gdn_geometry.Geometry(1, 8, 8192) and hash(g) == hash(gdn_geometry.Geometry(1, 8, 8192))
    assert all(type(v) is int for v in (g.K_HEADS, g.V_HEADS, g.HIDDEN))
    assert gdn_geometry.geometry_fields_off(np.int64(1), np.int64(8), np.int64(8192)) == {}
    with pytest.raises(ValueError, match="K_HEADS must be a positive int"):
        gdn_geometry.Geometry(True, 8, 8192)
    assert gdn_geometry.geometry_fields_off(True, 8, 8192) == {"K_HEADS": True}


def test_host_entry_keeps_the_one_cta_form_and_a_given_cfg_overrides_the_pick():
    k = entry.GdnDecodeFused("cpu", 1e-6)
    assert k.sm_count is None and not k.cfg_given
    assert k.cfg_for(1) == k.cfg_for(64) == kcfg.CFG  # no CUDA device to pick by
    k.sm_count = 148  # a 148-SM reading: the pick follows rows (the device cell below checks the reading itself)
    assert k.cfg_for(1) == kcfg.PAIR_CFG and k.cfg_for(3) == kcfg.CFG and k.cfg_for(4) == kcfg.CAPPED_CFG and k.cfg_for(2) is k.cfg_for(3)
    given = kcfg.make_cfg(TPR=8)
    k8 = entry.GdnDecodeFused("cpu", 1e-6, cfg=given)
    k8.sm_count = 148
    assert k8.cfg_given and k8.cfg_for(1) == k8.cfg_for(4) == given  # the override wins at every rows


def test_entry_on_a_99kb_part_picks_the_one_cta_form_past_the_pair_regime(monkeypatch):
    """Host-only, the device readers stubbed to the cc 12.0 readings and 170 SMs: the entry's pick reads the part -- the pair at rows 1,
    the one-CTA form from rows 2 on (two waves from rows 4, never CAPPED_CFG: the whole slab is one CTA per SM by shared memory there,
    ``co_resident_ctas`` 1) -- every form stamped with the readings; the same entry on the cc 10.0 readings picks CAPPED_CFG at rows 4; a
    part holding no form is the typed decline at construction, and a caller's explicit form is validated as given."""
    monkeypatch.setattr(entry, "device_shared_memory", lambda device: SM120_READING)
    monkeypatch.setattr(entry, "device_multiprocessor_count", lambda device: 170)
    k = entry.GdnDecodeFused("cuda:0", 1e-6)
    assert (k.readings, k.sm_count) == (SM120_READING, 170) and k.cfg == replace(kcfg.CFG, **_on(SM120_READING))
    assert replace(k.cfg_for(1), **_on(SM100_READING)) == kcfg.PAIR_CFG and k.cfg_for(1).co_resident_ctas == 1
    for rows in (2, 3, 4, 8, 64):
        got = k.cfg_for(rows)
        assert replace(got, **_on(SM100_READING)) == kcfg.CFG, rows
        assert (got.MIN_CTAS_PER_SM, got.co_resident_ctas, got.SMEM_PAD_BYTES, got.SMEM_BYTES) == (1, 1, 0, 74848), rows
        assert (got.SMEM_PER_SM_BYTES, got.SMEM_OPTIN_BYTES) == SM120_READING, "every pick carries the part's readings"
    assert k.cfg_for(4) is k.cfg_for(8), "one stamped form per module form: the forms cache"
    assert replace(kcfg.CAPPED_CFG, **_on(SM120_READING)).co_resident_ctas == 1, "the cap would keep nothing co-resident on this part"
    ops = make_operands(rows=4)
    assert k.key(4, ops["x"], ops["cs"], ops["state"], ops["z"])[1] == k.cfg_for(4)
    # the cc 10.0 readings on 148 SMs: the measured picks, the capped form two per SM at rows 4
    monkeypatch.setattr(entry, "device_shared_memory", lambda device: SM100_READING)
    monkeypatch.setattr(entry, "device_multiprocessor_count", lambda device: 148)
    k100 = entry.GdnDecodeFused("cuda:0", 1e-6)
    assert k100.cfg == kcfg.CFG and k100.readings == SM100_READING
    assert k100.cfg_for(1) == kcfg.PAIR_CFG and k100.cfg_for(2) == kcfg.CFG and k100.cfg_for(4) == kcfg.CAPPED_CFG and k100.cfg_for(4).co_resident_ctas == 2
    # a part that holds no form: the typed decline at construction; a caller's explicit form is validated as given (its own typed error)
    monkeypatch.setattr(entry, "device_shared_memory", lambda device: (40960, 40000))
    with pytest.raises(NotImplementedError, match="40000-byte shared-memory opt-in per CTA .* holds no decode form"):
        entry.GdnDecodeFused("cuda:0", 1e-6)
    with pytest.raises(ValueError, match="exceeds the per-CTA opt-in maximum 40000"):
        entry.GdnDecodeFused("cuda:0", 1e-6, cfg=kcfg.PAIR_CFG)


def test_pdl_launch_knobs_apply_on_top_of_the_pick_and_of_a_given_cfg():
    """``pdl`` / ``pdl_release`` are launch knobs on top of whatever form serves: ``pdl=True`` sets ``PDL`` and ``PDL_RELEASE`` on every
    pick (the pair, the one-CTA and the capped forms alike; the release follows ``pdl`` unless said otherwise), a knob left None keeps the
    form's own fields (the picks: off; a given cfg's: as given), and the knob changes the compile key (two compiled kernels per form)."""
    plain, pdl = entry.GdnDecodeFused("cpu", 1e-6), entry.GdnDecodeFused("cpu", 1e-6, pdl=True)
    plain.sm_count = pdl.sm_count = 148  # the pick on a 148-SM part, from the host
    for rows, form in ((1, kcfg.PAIR_CFG), (2, kcfg.CFG), (4, kcfg.CAPPED_CFG), (8, kcfg.CAPPED_CFG)):
        got = pdl.cfg_for(rows)
        assert (got.PDL, got.PDL_RELEASE) == (True, True) and (plain.cfg_for(rows).PDL, plain.cfg_for(rows).PDL_RELEASE) == (False, False)
        assert got == replace(plain.cfg_for(rows), PDL=True, PDL_RELEASE=True), f"rows {rows}: the knob changes the launch fields only"
        assert (got.TWO_CTAS_PER_HEAD, got.MIN_CTAS_PER_SM) == (form.TWO_CTAS_PER_HEAD, form.MIN_CTAS_PER_SM), f"rows {rows}: the pick stays"
    k = entry.GdnDecodeFused("cpu", 1e-6, pdl=True, pdl_release=False)
    assert (k.cfg_for(1).PDL, k.cfg_for(1).PDL_RELEASE) == (True, False)
    k = entry.GdnDecodeFused("cpu", 1e-6, pdl_release=True)
    assert (k.cfg_for(1).PDL, k.cfg_for(1).PDL_RELEASE) == (False, True), "the release knob alone: a PDL primary that is no secondary"
    given = kcfg.make_cfg(PDL=True, PDL_RELEASE=True)
    assert entry.GdnDecodeFused("cpu", 1e-6, cfg=given).cfg_for(1).PDL is True, "a given cfg's launch fields stay without the knob"
    off = entry.GdnDecodeFused("cpu", 1e-6, cfg=given, pdl=False).cfg_for(1)
    assert (off.PDL, off.PDL_RELEASE) == (False, False), "pdl=False turns a given cfg's launch off"
    ops = make_operands(rows=2)
    assert plain.key(2, ops["x"], ops["cs"], ops["state"], ops["z"]) != pdl.key(2, ops["x"], ops["cs"], ops["state"], ops["z"])


def test_cfg_knobs_validate():
    for tpr in (2, 4, 8, 16):  # at 256 threads: 1 x 64, 2 x 32, 4 x 16, 8 x 8 state values per thread
        cfg = kcfg.make_cfg(TPR=tpr)
        assert cfg.R * cfg.C * cfg.THREADS == 128 * 128 and cfg.C % 8 == 0 and cfg.R * cfg.C == 64
        assert cfg.RG * cfg.TPR == 32 and cfg.BOX_ROWS == cfg.R * cfg.WARPS and cfg.NORM_PARTS == cfg.RG * cfg.WARPS
    with pytest.raises(ValueError, match="does not tile"):
        kcfg.make_cfg(TPR=1)  # one thread per row needs the 128-thread form
    cfg = kcfg.make_cfg(THREADS=128, TPR=1, WARPGROUP_SPLIT=False)
    assert (cfg.R, cfg.C, cfg.RG, cfg.BOX_ROWS) == (1, 128, 32, 4)
    with pytest.raises(ValueError, match="THREADS"):
        kcfg.make_cfg(THREADS=512)
    with pytest.raises(ValueError, match="TPR"):
        kcfg.make_cfg(TPR=3)
    with pytest.raises(ValueError, match="WARPGROUP_SPLIT"):
        kcfg.make_cfg(THREADS=128, TPR=1, WARPGROUP_SPLIT=True)
    with pytest.raises(ValueError, match="CTAS_PER_SM"):
        kcfg.make_cfg(CTAS_PER_SM=3)
    with pytest.raises(ValueError, match="MIN_CTAS_PER_SM"):
        kcfg.make_cfg(MIN_CTAS_PER_SM=3)
    with pytest.raises(ValueError, match="contradicts"):
        kcfg.make_cfg(MIN_CTAS_PER_SM=2, CTAS_PER_SM=1)
    with pytest.raises(ValueError, match="STATE_LOAD_PATH"):
        kcfg.make_cfg(STATE_LOAD_PATH="tma")
    with pytest.raises(ValueError, match="STATE_STREAM"):
        kcfg.make_cfg(STATE_LOAD_PATH="ldg128", STATE_STREAM=True)
    with pytest.raises(ValueError, match="PIN_ENTRY_LOADS"):
        kcfg.make_cfg(PIN_ENTRY_LOADS=True)  # the default's state path is bulk: nothing to pin
    with pytest.raises(ValueError, match="CONV_FIRST"):
        kcfg.make_cfg(STATE_LOAD_PATH="ldg128", CONV_FIRST=True)  # no slab boxes to order the conv copies ahead of
    assert kcfg.make_cfg(CONV_FIRST=True).CONV_FIRST and not kcfg.CFG.CONV_FIRST
    assert kcfg.CFG.SINGLE_TAIL and not kcfg.make_cfg(SINGLE_TAIL=False).SINGLE_TAIL
    assert kcfg.make_cfg(STATE_LOAD_PATH="ldg128").SINGLE_TAIL  # accepted as a no-op there (the kernel keeps the twice-traced body)
    with pytest.raises(ValueError, match="STORE_PIPELINE"):
        kcfg.make_cfg(STATE_LOAD_PATH="ldg128", STORE_PIPELINE=True)  # no shared-memory boxes to store
    with pytest.raises(ValueError, match="STORE_PIPELINE"):
        kcfg.make_cfg(THREADS=128, TPR=1, WARPGROUP_SPLIT=False, STORE_PIPELINE=True)  # one warpgroup: no halves
    assert kcfg.make_cfg(STORE_PIPELINE=True).STORE_PIPELINE and not kcfg.CFG.STORE_PIPELINE
    with pytest.raises(ValueError, match="TWO_CTAS_PER_HEAD"):
        kcfg.make_cfg(STATE_LOAD_PATH="ldg128", TWO_CTAS_PER_HEAD=True)
    with pytest.raises(ValueError, match="does not tile"):
        kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, TPR=2)  # 64 rows x 2 threads per row < 256 threads
    two = kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0)
    assert (two.CTAS_PER_HEAD, two.CTA_ROWS, two.R, two.C, two.RG, two.BOX_ROWS, two.V_PERM_STRIDE) == (2, 64, 1, 32, 8, 8, 16)
    assert (two.STATE_BOX_BYTES, two.STATE_SLAB_BYTES_CTA, two.TICKET_LIMIT, kcfg.CFG.TICKET_LIMIT) == (4096, 32768, 5, 2)
    assert two.NATURAL_SMEM_BYTES == 74848 - 65664 + 32 + 8 * (4096 + 16) and two.SMEM_BYTES == two.NATURAL_SMEM_BYTES  # 42,112 B: five fit an SM
    two8 = kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, TPR=8, CTAS_PER_SM=0)
    assert (two8.R, two8.C, two8.RG, two8.BOX_ROWS, two8.V_PERM_STRIDE, two8.STATE_BOX_BYTES) == (2, 16, 4, 16, 8, 8192)
    assert kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, THREADS=128, TPR=2, WARPGROUP_SPLIT=False).R == 1
    assert kcfg.NORM_MBAR_ARRIVERS == 1 and kcfg.NORM_XFER_BYTES == 4
    with pytest.raises(ValueError, match="SMEM_PER_SM_BYTES"):
        kcfg.make_cfg(SMEM_PER_SM_BYTES=100000, SMEM_OPTIN_BYTES=200000)
    assert kcfg.make_cfg(CTAS_PER_SM=0).SMEM_PAD_BYTES == 0
    ldg2 = kcfg.make_cfg(STATE_LOAD_PATH="ldg128", TPR=2)
    assert ldg2.NATURAL_SMEM_BYTES == 9392  # the TPR-2 ldg128 build's buffers in allocation order, 16-byte slots
    ldg8 = kcfg.make_cfg(STATE_LOAD_PATH="ldg128", TPR=8)
    assert ldg8.NATURAL_SMEM_BYTES == 9200  # TPR 8: 1280 B of q / k slices, 128 B of norm partials, a 16-B state slot
    bulk = kcfg.make_cfg(TPR=8)
    assert bulk.STATE_BOX_BYTES == 16384 and bulk.STATE_BOX_PITCH_ELEMS == 4100 and bulk.STATE_SMEM_ELEMS == 4 * 4100
    assert bulk.NATURAL_SMEM_BYTES == 9200 - 16 + 65600  # the 64 KB slab + 4 x 16 B box pad
    default = kcfg.CFG  # TPR 4: 8 boxes of 16 rows, each padded by 16 B
    assert default.STATE_BOX_BYTES == 8192 and default.STATE_BOX_PITCH_ELEMS == 2052 and default.STATE_SMEM_ELEMS == 8 * 2052
    assert default.NATURAL_SMEM_BYTES == 74848 and kcfg.STREAMED_CFG.SMEM_BYTES == 74848  # the streamed form's 74.85 KB (the device reading)
    assert default.SMEM_BYTES == 119808  # one CTA per SM on the 228 KB reading: the 119.8 KB the device reports for this form
    assert kcfg.make_cfg(STATE_LOAD_PATH="bulk", STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=2).MIN_CTAS_PER_SM == 2
    assert kcfg.make_cfg(STATE_LOAD_PATH="bulk", MIN_CTAS_PER_SM=2, CTAS_PER_SM=0).SMEM_PAD_BYTES == 0
    assert kcfg.CONV_MBAR_ARRIVERS == 1 and kcfg.STATE_MBAR_ARRIVERS == 1
    assert kcfg.CONV_BULK_BYTES == 6144 and kcfg.STATE_SLAB_BYTES == 65536


@pytest.mark.parametrize("per_sm, optin", PART_READINGS)
@pytest.mark.parametrize("path", kcfg.STATE_LOAD_PATHS)
def test_occupancy_pad_caps_the_ctas_per_sm_on_every_part_reading(per_sm, optin, path):
    """``CTAS_PER_SM`` n pads the footprint so that n + 1 CTAs exceed the SM while n fit it; where even the NATURAL footprints of n CTAs do
    not fit (the whole-slab bulk form at n = 2 on the cc 12.0 reading: 2 x (74,848 + 1,024) B > 102,400 B) the knob is a typed decline,
    never a silent one-per-SM launch."""
    for n in (1, 2):
        natural = kcfg.make_cfg(CTAS_PER_SM=0, STATE_LOAD_PATH=path, SMEM_PER_SM_BYTES=per_sm, SMEM_OPTIN_BYTES=optin).NATURAL_SMEM_BYTES
        if n * (natural + kcfg.SMEM_CTA_RESERVE_BYTES) > per_sm:
            with pytest.raises(ValueError, match="do not fit the SM"):
                kcfg.make_cfg(CTAS_PER_SM=n, STATE_LOAD_PATH=path, SMEM_PER_SM_BYTES=per_sm, SMEM_OPTIN_BYTES=optin)
            continue
        c = kcfg.make_cfg(CTAS_PER_SM=n, STATE_LOAD_PATH=path, SMEM_PER_SM_BYTES=per_sm, SMEM_OPTIN_BYTES=optin)
        per_cta = c.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES
        assert (n + 1) * per_cta > per_sm, "one CTA more than allowed does not fit the SM"
        assert n * per_cta <= per_sm, "the allowed count does"
        assert c.SMEM_BYTES <= optin
        assert c.SMEM_PAD_BYTES % 16 == 0
        assert per_sm // per_cta == n and c.co_resident_ctas == 1, "the pad admits exactly n by shared memory; an uncapped form guarantees one by registers"


def test_cc120_reading_footprints_and_co_residency():
    """The cc 12.0 readings (100 KB per SM, 99 KB per-CTA opt-in) through the module's own arithmetic: the launch footprints of the forms
    (no pad fits beside a slab there: the natural footprints launch), which of them two can share an SM -- ``co_resident_ctas`` reads BOTH
    budgets, the part's shared memory and the register cap the launch asks for -- and the knob arms that cannot be asked for two."""
    per_sm, optin = SM120_READING
    assert SM120_READING == (102400, 101376) and SM100_READING == (233472, 232448)
    pair, cfg, capped, streamed = (replace(f, **_on(SM120_READING)) for f in (kcfg.PAIR_CFG, kcfg.CFG, kcfg.CAPPED_CFG, kcfg.STREAMED_CFG))
    assert (pair.SMEM_BYTES, cfg.SMEM_BYTES, capped.SMEM_BYTES, streamed.SMEM_BYTES) == (42112, 74848, 74848, 74848)
    assert (pair.SMEM_PAD_BYTES, cfg.SMEM_PAD_BYTES, capped.SMEM_PAD_BYTES) == (0, 0, 0)
    for form in (pair, cfg, capped, streamed):
        kcfg.validate_cfg(form)  # every form fits the 99 KB opt-in at one CTA per SM
    # the whole-slab forms are one per SM by shared memory alone (2 x 75,872 B > 102,400 B), cap or no cap; the uncapped pair guarantees one
    assert (pair.co_resident_ctas, cfg.co_resident_ctas, capped.co_resident_ctas, streamed.co_resident_ctas) == (1, 1, 1, 1)
    assert 2 * (capped.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES) == 151744 > per_sm
    pair2 = kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, MIN_CTAS_PER_SM=2, **_on(SM120_READING))
    assert pair2.co_resident_ctas == 2 and 2 * (pair2.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES) == 86272 <= per_sm < 3 * (
        pair2.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES
    )
    ldg2 = kcfg.make_cfg(STATE_LOAD_PATH="ldg128", TPR=16, CTAS_PER_SM=2, MIN_CTAS_PER_SM=2, **_on(SM120_READING))
    assert (ldg2.NATURAL_SMEM_BYTES, ldg2.SMEM_PAD_BYTES, ldg2.SMEM_BYTES, ldg2.co_resident_ctas) == (9392, 27824, 37216, 2)
    assert kcfg.make_cfg(STATE_LOAD_PATH="ldg128", **_on(SM120_READING)).SMEM_BYTES == 54272  # CTAS_PER_SM 1: padded to one per SM
    for arm in (
        dict(MIN_CTAS_PER_SM=2, CTAS_PER_SM=2),
        dict(STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=2),
        dict(TPR=8, MIN_CTAS_PER_SM=2, CTAS_PER_SM=2),
    ):
        with pytest.raises(ValueError, match="do not fit the SM"):
            kcfg.make_cfg(**arm, **_on(SM120_READING))  # two whole slabs asked for by the pad: the typed decline
    # the cc 10.0 readings: the capped whole-slab forms ARE two per SM (three by shared memory, two by the cap); the uncapped forms one
    assert replace(kcfg.CAPPED_CFG, **_on(SM100_READING)).co_resident_ctas == replace(kcfg.STREAMED_CFG, **_on(SM100_READING)).co_resident_ctas == 2
    assert replace(kcfg.CFG, **_on(SM100_READING)).co_resident_ctas == replace(kcfg.PAIR_CFG, **_on(SM100_READING)).co_resident_ctas == 1
    assert kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, MIN_CTAS_PER_SM=2, **_on(SM100_READING)).co_resident_ctas == 2


def test_state_box_bank_map_is_conflict_free_at_every_tpr():
    """The 8 lanes of a 128-bit shared-memory phase read 8 distinct bank quads: bank = 4 rg + (128 / TPR) tpr + 4 c4 (mod 32)."""
    forms = [(256, tpr, False) for tpr in (2, 4, 8, 16)] + [(128, 1, False)] + [(256, tpr, True) for tpr in (4, 8, 16)] + [(128, 2, True)]
    for threads, tpr, two in forms:
        c = kcfg.make_cfg(THREADS=threads, TPR=tpr, WARPGROUP_SPLIT=threads == 256, STATE_LOAD_PATH="bulk", TWO_CTAS_PER_HEAD=two)
        pitch_words = c.STATE_BOX_PITCH_ELEMS
        for c4 in range(c.C // 4):
            for m in range(c.R):
                for phase in range(4):
                    quads = set()
                    for lane in range(8 * phase, 8 * phase + 8):
                        tp, rg = lane // c.RG, lane % c.RG
                        word = rg * pitch_words + (c.R * 0 + m) * 128 + tp * c.C + 4 * c4  # warp 0; the warp term is a multiple of 32
                        quads.add((word % 32) // 4)
                    assert len(quads) == 8, (threads, tpr, two, c4, m, phase, sorted(quads))


@pytest.mark.parametrize(
    "mutation, exc, token",
    [
        (lambda o: dict(x=torch.zeros(ROWS, 8192, dtype=torch.bfloat16)), NotImplementedError, "x"),
        (lambda o: dict(x=o["x"].to(torch.float16)), TypeError, "x"),
        (lambda o: dict(x=torch.zeros(ROWS, 10248, dtype=torch.bfloat16)[:, 1:10241]), ValueError, "x"),
        (lambda o: dict(x=torch.zeros(ROWS, 10244, dtype=torch.bfloat16)[:, :10240]), NotImplementedError, "x"),
        (lambda o: dict(cs=torch.zeros(SLOTS, 10240, 2, dtype=torch.bfloat16)), NotImplementedError, "conv state"),
        (lambda o: dict(cs=torch.zeros(SLOTS, 3, 10240, dtype=torch.bfloat16)), NotImplementedError, "conv state"),
        (
            lambda o: dict(cs=torch.zeros(10240, SLOTS, 3, dtype=torch.bfloat16).transpose(0, 1)),
            NotImplementedError,
            "conv-state layout",
        ),  # right shape, strides (3, 9, 1)
        (lambda o: dict(cs=o["cs"].to(torch.float16)), TypeError, "cs"),
        (lambda o: dict(cw=torch.zeros(10240, 3, dtype=torch.bfloat16)), NotImplementedError, "conv weight"),
        (lambda o: dict(al=torch.zeros(48, dtype=torch.bfloat16)), TypeError, "A_log"),
        (lambda o: dict(dt=torch.zeros(47, dtype=torch.bfloat16)), NotImplementedError, "dt_bias"),
        (lambda o: dict(state=torch.zeros(SLOTS, 48, 64, 64)), NotImplementedError, "state"),
        (lambda o: dict(state=torch.zeros(SLOTS, 48, 128, 128, dtype=torch.float16)), TypeError, "state"),
        (lambda o: dict(state=torch.zeros(SLOTS, 48, 128, 128).transpose(2, 3)), NotImplementedError, "head slabs"),
        (lambda o: dict(idx=torch.arange(1, ROWS + 1, dtype=torch.int64)), TypeError, "idx"),
        (lambda o: dict(idx=torch.ones(ROWS - 1, dtype=torch.int32)), ValueError, "idx"),
        (lambda o: dict(idx=torch.ones(ROWS + 8, dtype=torch.int32)[1 : ROWS + 1]), ValueError, "idx"),  # right shape, a 4-byte-offset slice
        (lambda o: dict(w=torch.ones(128 + 8, dtype=torch.bfloat16)[1:129]), ValueError, "w"),  # right shape, a 2-byte-offset slice
        (lambda o: dict(z=torch.zeros(ROWS, 48, 64, dtype=torch.bfloat16)), NotImplementedError, "z"),
        (lambda o: dict(w=torch.zeros(64, dtype=torch.bfloat16)), NotImplementedError, "norm weight"),
        (lambda o: dict(scale=torch.zeros(0)), ValueError, "scale"),
        (lambda o: dict(hidden=torch.zeros(ROWS, 4096, dtype=torch.bfloat16)), NotImplementedError, "hidden"),
        (lambda o: dict(baw=torch.zeros(96, 4096, dtype=torch.bfloat16)), NotImplementedError, "BA weight"),
        (lambda o: dict(rows=0), ValueError, "rows"),
        (lambda o: dict(rows=entry.MAX_ROWS + 1), NotImplementedError, "65535"),  # one grid row per request: CUDA's grid Y limit
    ],
)
def test_typed_declines_name_the_operand(mutation, exc, token):
    ops = make_operands()
    ops.update(mutation(ops))
    with pytest.raises(exc, match=token):
        call_check(ops)


def test_declines_an_unserved_compute_capability_by_name():
    """The admission is two closed ranges of ``10 * major + minor``: SM100-class (cc 10.x) and SM120-class (cc 12.x) -- cc 12.0 is served,
    cc 12.1 structurally with it, while cc 9.x, 11.x (these kernels never ran there) and 13.x are declined by name, the decline naming
    both served families and the device's capability; the single-major constant is gone (the ranges are the one source)."""
    for cc in ((9, 0), (8, 0), (11, 0), (13, 0)):
        with pytest.raises(NotImplementedError, match=rf"10\.x .*12\.x.*{cc[0]}\.{cc[1]}"):
            call_check(make_operands(), device_cc=cc)
        assert not entry.cc_supported(cc), cc
    for cc in ((10, 0), (10, 3), (10, 7), (12, 0), (12, 1)):
        call_check(make_operands(), device_cc=cc)
        assert entry.cc_supported(cc), cc
    assert entry.SUPPORTED_CC_RANGES == ((100, 109), (120, 129)) and "10.x" in entry.SUPPORTED_CC_TEXT and "12.x" in entry.SUPPORTED_CC_TEXT
    assert not entry.cc_supported(None) and not hasattr(entry, "SUPPORTED_CC_MAJOR")


def test_declines_cpu_operands_when_no_capability_is_given():
    with pytest.raises(NotImplementedError, match="CUDA tensors only"):
        call_check(make_operands(), device_cc=None)


def test_every_launch_re_checks_the_binding():
    """A compiled DSL entry accepts a tensor of another dtype or footprint silently, so the entry re-runs the binding checks on every
    launch, before the compiled kernel sees the pointers: with a kernel planted under the binding's compile key (rows, strides, the
    config -- none of these mutations changes it), a mismatched operand is a typed decline, never a launch."""
    k = entry.GdnDecodeFused("cpu", 1e-6)
    ops = make_operands()
    key = k.key(ROWS, ops["x"], ops["cs"], ops["state"], ops["z"])

    def launched(*args):
        raise AssertionError("the planted kernel was launched on a mismatched binding")

    k.compiled[key] = launched
    for mutation, exc, token in (
        (dict(hidden=ops["hidden"].to(torch.float16)), TypeError, "hidden"),  # a dtype the key does not carry
        (dict(idx=ops["idx"][:1]), ValueError, "idx"),  # fewer slots than rows
        (dict(w=torch.ones(128 + 8, dtype=torch.bfloat16)[1:129]), ValueError, "w"),  # the right shape at a 2-byte offset: misaligned
        (dict(cw=torch.zeros(10240, 3, dtype=torch.bfloat16)), NotImplementedError, "conv weight"),
        (dict(state=ops["state"].cpu().to("meta")), ValueError, "one device"),
    ):
        bad = dict(ops, **mutation)
        assert k.key(ROWS, bad["x"], bad["cs"], bad["state"], bad["z"]) == key, "the mutation keeps the compile key: only the launch check sees it"
        with pytest.raises(exc, match=token):
            _run(k, bad)


def test_binding_check_requires_every_operand_on_the_given_device():
    """With the entry's device given (a concrete CUDA ordinal), every operand must live on exactly it, type AND index: CPU operands of
    the served shapes and strides are a ``ValueError`` naming the operand and both devices -- host-only, no CUDA needed."""
    ops = make_operands()
    with pytest.raises(ValueError, match=r"one device, the entry's \(cuda:0\); x is on cpu"):
        entry.check_binding(*_binding_args(ops), device=torch.device("cuda", 0))
    entry.check_binding(*_binding_args(ops), device="cpu")  # the operands' own device passes
    with pytest.raises(ValueError, match=r"the entry's \(cpu\); state is on meta"):
        entry.check_binding(*_binding_args(dict(ops, state=ops["state"].to("meta"))), device="cpu")
    with pytest.raises(ValueError, match=r"x's \(cpu\); idx is on meta"):
        entry.check_binding(*_binding_args(dict(ops, idx=ops["idx"].to("meta"))))  # without a device: x's


def test_entry_device_is_a_concrete_ordinal_that_guards_a_warm_cache_too(monkeypatch):
    """A CUDA entry resolves ``cuda`` to the current ordinal at construction, and a WARM entry (a kernel under the binding's compile key)
    still runs the device check before the compiled callable is looked up: CPU operands of the same shapes and strides (the key matches)
    are a ``ValueError`` on the launch path and on the compile path, the planted kernel never invoked.  Host-only: the device readers
    (the part's shared-memory readings, the SM count, the current ordinal) are stubbed."""
    monkeypatch.setattr(entry, "device_shared_memory", lambda device: PART_READINGS[0])
    monkeypatch.setattr(entry, "device_multiprocessor_count", lambda device: 148)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    k = entry.GdnDecodeFused("cuda", 1e-6)
    assert k.device == torch.device("cuda", 1) and k.sm_count == 148
    assert entry.GdnDecodeFused("cuda:0", 1e-6).device == torch.device("cuda", 0)  # a given ordinal stays
    assert entry.GdnDecodeFused("cpu", 1e-6).device == torch.device("cpu")  # a host entry: nothing to resolve
    ops = make_operands()
    key = k.key(ROWS, ops["x"], ops["cs"], ops["state"], ops["z"])
    calls = []
    k.compiled[key] = lambda *args: calls.append(args)
    with pytest.raises(ValueError, match=r"the entry's \(cuda:1\); x is on cpu"):
        _run(k, ops)
    with pytest.raises(ValueError, match=r"the entry's \(cuda:1\); x is on cpu"):
        k.compile(*_binding_args(ops))
    assert not calls, "the planted kernel was never invoked"


def test_declines_a_dsl_below_the_floor_by_version(monkeypatch):
    """The kernel's 4.8.0 floor runs through the library's one version parser (cudnn.frost.buffers.cutedsl_too_old with the kernel's
    floor), so a prerelease of the floor compares as the floor and an internal build is never judged by the public numbering."""
    from cudnn.frost import buffers

    def installed(state):
        monkeypatch.setattr(buffers, "cutedsl_state", lambda: state)
        monkeypatch.setattr(entry, "cutedsl_state", lambda: state)

    installed((True, ("nvidia-cutlass-dsl", "4.7.1")))
    msg = entry._dsl_version_error()
    assert msg is not None and "4.8.0" in msg and "4.7.1" in msg and "pip install" in msg
    installed((True, ("nvidia-cutlass-dsl", "4.8.0")))
    assert entry._dsl_version_error() is None
    installed((True, ("nvidia-cutlass-dsl", "4.8.0rc1")))
    assert entry._dsl_version_error() is None
    installed((True, ("nvidia-cutlass-dsl-internal", "0.3.0+2026")))
    assert entry._dsl_version_error() is None
    installed((False, None))
    assert "not installed" in entry._dsl_version_error()
    assert entry.DSL_FLOOR > buffers.CUTEDSL_MIN_VERSION, "the kernel's floor is above the library's: one call covers both"
    assert buffers.cutedsl_too_old(("nvidia-cutlass-dsl", "4.7.9"), floor=entry.DSL_FLOOR) and not buffers.cutedsl_too_old(("nvidia-cutlass-dsl", "4.7.9"))


def test_compile_key_is_plan_time_only():
    k = entry.GdnDecodeFused("cpu", 1e-6)
    a = make_operands(seed=1)
    b = make_operands(seed=2)
    assert k.key(ROWS, a["x"], a["cs"], a["state"], a["z"]) == k.key(ROWS, b["x"], b["cs"], b["state"], b["z"])  # values never enter
    c = make_operands(slots=SLOTS + 3)
    assert k.key(ROWS, a["x"], a["cs"], a["state"], a["z"]) == k.key(ROWS, c["x"], c["cs"], c["state"], c["z"])  # nor the slot count
    wide = torch.zeros(ROWS, 16384 + 1024, dtype=torch.bfloat16)
    assert k.key(ROWS, wide[:, :10240], a["cs"], a["state"], a["z"]) != k.key(ROWS, a["x"], a["cs"], a["state"], a["z"])  # a stride does
    assert k.key(ROWS + 1, a["x"], a["cs"], a["state"], a["z"]) != k.key(ROWS, a["x"], a["cs"], a["state"], a["z"])
    k2 = entry.GdnDecodeFused("cpu", 1e-6, cfg=kcfg.make_cfg(TPR=8))
    assert k2.key(ROWS, a["x"], a["cs"], a["state"], a["z"]) != k.key(ROWS, a["x"], a["cs"], a["state"], a["z"])
    assert k.debug is False and replace(k.cfg, DEBUG=True) == entry.GdnDecodeFused("cpu", 1e-6, debug=True).cfg
    assert k.cfg.SMEM_PER_SM_BYTES == kcfg.SM100_SMEM_PER_SM_BYTES  # a host entry keeps the module reading; a CUDA device replaces it


def test_debug_buffers_exist_under_debug_only():
    """Without DEBUG the kernel's three debug slots are compiled out (None), never bound to an allocation; a debug entry allocates them."""
    plain, debug = entry.GdnDecodeFused("cpu", 1e-6), entry.GdnDecodeFused("cpu", 1e-6, debug=True)
    assert set(plain.outputs(ROWS)) == {"qout", "out", "counts"}
    assert plain._debug_args(plain.outputs(ROWS)) == [None, None, None]
    assert set(debug.outputs(ROWS)) == {"qout", "out", "counts", "raw", "convout", "baout"}
    assert tuple(debug.outputs(ROWS)["convout"].shape) == (ROWS, kcfg.QKV_CHANNELS) and tuple(debug.outputs(ROWS)["baout"].shape) == (ROWS, 2 * kcfg.V_HEADS)


def test_flat_view_covers_the_footprint_with_16_byte_alignment():
    from cudnn.frost.buffers import cutedsl_requirement_error

    msg = cutedsl_requirement_error("the _flat view test")
    if msg:
        pytest.skip(msg)
    slab = torch.zeros(3, 16384, dtype=torch.bfloat16)
    view = slab[:, 10240:].view(3, 48, 128)
    flat = entry._flat(view)
    # the footprint of a strided view: 1 + sum((d - 1) * s) elements from its first element
    assert flat.shape[0] == 1 + 2 * 16384 + 47 * 128 + 127


# ---------------------------------------------------------------------------------------------------------------------------
# device cells
# ---------------------------------------------------------------------------------------------------------------------------
def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    cc = tuple(torch.cuda.get_device_capability())
    if not entry.cc_supported(cc):
        pytest.skip(f"the fused GDN decode step serves compute capability {entry.SUPPORTED_CC_TEXT}, this device is {cc}")
    msg = entry._dsl_version_error()
    if msg:
        pytest.skip(msg)
    return torch.device("cuda")


def _run(k, ops):
    return k(*_binding_args(ops))


def _reference(ops, eps=1e-6):
    """A plain torch (fp32 / fp64) model of the fused math; the kernel rounds exactly where the serving chain rounds (BA in bf16,
    the conv taps and SiLU in bf16, the core output in bf16, the normed value in bf16), the model's SiLU and transcendental forms
    are torch's -- so the comparison is loose, a function check, never a bit check.  The geometry is read off the operands (the
    state's V heads, the QKV slab's K heads, the hidden width), so a per-rank cell reuses it."""
    rows = ops["rows"]
    x, cs, cw, al, dt, st, idx, z, w, scale, hid, baw = (ops[k] for k in ("x", "cs", "cw", "al", "dt", "state", "idx", "z", "w", "scale", "hidden", "baw"))
    D = gdn_geometry.HEAD_DIM
    NV, QKV = int(st.shape[1]), int(x.shape[1])
    NK = (QKV - NV * D) // (2 * D)
    VK = NV // NK  # V heads per K head: the K head of V head h is h // VK
    K0, V0 = NK * D, 2 * NK * D  # the first K / V channel of the QKV slab
    out = torch.zeros(rows, NV, D, dtype=torch.float64, device=x.device)
    state_new = st.clone()
    cs_new = cs.clone()
    for b in range(rows):
        slot = int(idx[b])
        if slot <= 0:
            continue
        taps = cs[slot].float()  # [QKV, 3]
        xb = x[b].float()
        acc = torch.zeros(QKV, dtype=torch.float32, device=x.device)
        for j in range(3):
            acc = acc + (taps[:, j] * cw[:, j].float()).bfloat16().float()
        acc = acc + (xb * cw[:, 3].float()).bfloat16().float()
        conv = torch.nn.functional.silu(acc).bfloat16().float()
        cs_new[slot, :, 0] = cs[slot, :, 1]
        cs_new[slot, :, 1] = cs[slot, :, 2]
        cs_new[slot, :, 2] = x[b]
        for h in range(NV):
            kh = h // VK
            bval = (hid[b].float() @ baw[h].float()).bfloat16().float()
            aval = (hid[b].float() @ baw[NV + h].float()).bfloat16().float()
            av = aval + dt[h].float()
            soft = torch.nn.functional.softplus(av) if float(av) <= 20 else av
            decay = torch.exp(-torch.exp(al[h]) * soft)
            beta = torch.sigmoid(bval)
            q = conv[kh * D : (kh + 1) * D]
            k_ = conv[K0 + kh * D : K0 + (kh + 1) * D]
            v = conv[V0 + h * D : V0 + (h + 1) * D]
            q = q * (D**-0.5 / torch.sqrt((q * q).sum() + 1e-6))
            k_ = k_ / torch.sqrt((k_ * k_).sum() + 1e-6)
            S = st[slot, h].double() * decay.double()
            dot = S @ k_.double()
            inn = (v.double() - dot) * beta.double()
            S = S + torch.outer(inn, k_.double())
            state_new[slot, h] = S.float()
            o = (S @ q.double()).float().bfloat16().float()
            inv = torch.rsqrt((o * o).mean() + eps)
            gate = z[b, h].float()
            out[b, h] = ((o * inv) * w.float() * (gate / (1 + torch.exp(-gate)))).bfloat16().double()
    return out, state_new, cs_new


def test_entry_reads_the_device_shared_memory():
    dev = _device_or_skip()
    k = entry.GdnDecodeFused(dev, 1e-6)
    props = torch.cuda.get_device_properties(dev)
    assert k.cfg.SMEM_PER_SM_BYTES == props.shared_memory_per_multiprocessor
    assert k.cfg.SMEM_OPTIN_BYTES == props.shared_memory_per_block_optin
    assert k.readings == (props.shared_memory_per_multiprocessor, props.shared_memory_per_block_optin)
    assert k.cfg.SMEM_BYTES <= k.cfg.SMEM_OPTIN_BYTES and 2 * (k.cfg.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES) > k.cfg.SMEM_PER_SM_BYTES
    assert k.cfg == replace(kcfg.first_fitting(kcfg.ONE_CTA_FORMS, *k.readings), **k.part), "the start form: the one-CTA form this part holds"


def test_entry_picks_the_form_per_rows_on_this_device():
    """The entry's pick per rows on THIS part equals the module pick on the part's SM count and readings; the two-per-SM premise of the
    capped form is asserted only where the pick carries the register cap (the cc 10.0 readings), and where it does not (a 100 KB SM keeps
    the whole slab at one CTA per SM) the pick past the one-CTA grid is the one-CTA form itself, two waves, ``co_resident_ctas`` 1."""
    dev = _device_or_skip()
    k = entry.GdnDecodeFused(dev, 1e-6)
    props = torch.cuda.get_device_properties(dev)
    sms = props.multi_processor_count
    assert k.sm_count == sms and not k.cfg_given and k.readings == (props.shared_memory_per_multiprocessor, props.shared_memory_per_block_optin)
    for rows in (1, 2, 3, 4, 8, 64):
        want, got = kcfg.default_cfg(rows, sms, *k.readings), k.cfg_for(rows)
        assert replace(got, **_on(SM100_READING)) == want, rows
        assert (got.STATE_LOAD_PATH, got.TPR, got.PDL, got.SINGLE_TAIL) == ("bulk", 4, False, True)
        assert (got.SMEM_PER_SM_BYTES, got.SMEM_OPTIN_BYTES) == k.readings  # the part's readings
    pair = k.cfg_for(1)
    assert pair.TWO_CTAS_PER_HEAD == (2 * kcfg.V_HEADS <= sms)  # the pair while its 96 cluster CTAs have an SM each
    if pair.TWO_CTAS_PER_HEAD:
        assert pair.SMEM_PAD_BYTES == 0 and 2 * (pair.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES) <= pair.SMEM_PER_SM_BYTES
    one = k.cfg_for(sms // (2 * kcfg.V_HEADS) + 1)  # the first rows count whose cluster grid outgrows the SMs (rows 2 on 148 / 152 / 170 SMs)
    assert not one.TWO_CTAS_PER_HEAD and one.co_resident_ctas == 1 and 2 * (one.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES) > one.SMEM_PER_SM_BYTES
    beyond = k.cfg_for(sms // kcfg.V_HEADS + 1)  # the first rows count whose one-CTA grid outgrows the SMs
    if beyond.MIN_CTAS_PER_SM == 2:  # the capped form: this part's shared memory holds two of its CTAs per SM (the cc 10.0 readings)
        assert beyond.co_resident_ctas == 2 and beyond.SMEM_PAD_BYTES == 0 and 2 * (beyond.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES) <= beyond.SMEM_PER_SM_BYTES
    else:  # the whole slab is one CTA per SM by shared memory on this part, so the cap left the pick: the one-CTA form in two waves
        assert beyond == one and replace(kcfg.CAPPED_CFG, **k.part).co_resident_ctas == 1 and not kcfg.admissible(kcfg.CAPPED_CFG, *k.readings)


def test_warm_entry_rejects_cpu_operands_before_the_compiled_kernel_is_looked_up():
    """After a real compile and launch, the same entry called with CPU operands of the same shapes and strides (the compile key matches:
    a cache hit) is a ``ValueError`` from the device check on the launch path and on the compile path, before the compiled callable is
    looked up (swapped for a recorder: no call); the entry's device is a concrete ordinal, and it launches again afterwards."""
    dev = _device_or_skip()
    k = entry.GdnDecodeFused(dev, 1e-6)
    assert k.device == torch.device("cuda", torch.cuda.current_device())
    ops = make_operands(device=dev)
    _run(k, ops)
    torch.cuda.synchronize()
    key = k.key(ROWS, ops["x"], ops["cs"], ops["state"], ops["z"])
    cpu = make_operands()
    assert k.key(ROWS, cpu["x"], cpu["cs"], cpu["state"], cpu["z"]) == key, "the same shapes and strides: a warm key"
    real, calls = k.compiled[key], []
    k.compiled[key] = lambda *args: calls.append(args)
    try:
        with pytest.raises(ValueError, match=rf"the entry's \({k.device}\); x is on cpu"):
            _run(k, cpu)
        with pytest.raises(ValueError, match=rf"the entry's \({k.device}\); x is on cpu"):
            k.compile(*_binding_args(cpu))
    finally:
        k.compiled[key] = real
    assert not calls, "the compiled kernel was never invoked on CPU operands"
    _run(k, ops)
    torch.cuda.synchronize()


def test_operands_on_another_cuda_device_are_rejected_before_any_launch():
    """Operands on another CUDA device than the entry's (same shapes and strides: the compile key matches) are a ``ValueError`` naming
    both ordinals, on the launch path and on the compile path, the compiled kernel never invoked.  Needs two CUDA devices."""
    dev = _device_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    k = entry.GdnDecodeFused(dev, 1e-6)
    other = torch.device("cuda", (k.device.index + 1) % torch.cuda.device_count())
    ops = make_operands(device=dev)
    _run(k, ops)
    torch.cuda.synchronize()
    key = k.key(ROWS, ops["x"], ops["cs"], ops["state"], ops["z"])
    foreign = make_operands(device=other)
    assert k.key(ROWS, foreign["x"], foreign["cs"], foreign["state"], foreign["z"]) == key
    real, calls = k.compiled[key], []
    k.compiled[key] = lambda *args: calls.append(args)
    try:
        with pytest.raises(ValueError, match=rf"the entry's \({k.device}\); x is on {other}"):
            _run(k, foreign)
        with pytest.raises(ValueError, match=rf"the entry's \({k.device}\); x is on {other}"):
            k.compile(*_binding_args(foreign))
    finally:
        k.compiled[key] = real
    assert not calls, "the compiled kernel was never invoked on another device's operands"


@pytest.mark.parametrize("form", ["one-cta", "two-ctas-per-head"])
def test_null_slot_rows_are_zero_and_leave_the_state_and_conv_state_untouched(form):
    dev = _device_or_skip()
    ops = make_operands(rows=3, slots=4, device=dev, idx=torch.tensor([2, 0, 3], dtype=torch.int32, device=dev))
    state0, cs0 = ops["state"].clone(), ops["cs"].clone()
    k = entry.GdnDecodeFused(dev, 1e-6, cfg=None if form == "one-cta" else kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0))
    bufs = k.outputs(3)
    bufs["qout"].view(torch.uint8).fill_(0xAB)
    bufs["out"].view(torch.int16).fill_(0x7FC0)
    q = _run(k, ops)
    torch.cuda.synchronize()
    assert not q.view(torch.uint8)[1].any() and not bufs["out"][1].view(torch.int16).any(), "the null row writes zeros"
    assert q.view(torch.uint8)[0].any() and q.view(torch.uint8)[2].any(), "the live rows are written"
    assert torch.equal(ops["state"][0], state0[0]) and torch.equal(
        ops["state"][1], state0[1]
    ), "slots 0 / 1 (the null block and an unreferenced slot) are untouched"
    assert not torch.equal(ops["state"][2], state0[2]) and not torch.equal(ops["state"][3], state0[3]), "the live slots advance"
    assert torch.equal(ops["cs"][0], cs0[0]) and torch.equal(ops["cs"][1], cs0[1])
    assert not torch.equal(ops["cs"][2], cs0[2])
    assert not bufs["counts"].any(), "the reader tickets return to zero"


def test_deterministic_and_matches_the_torch_model():
    dev = _device_or_skip()
    ops_a = make_operands(rows=2, device=dev, seed=7)
    ops_b = make_operands(rows=2, device=dev, seed=7)
    ref_out, ref_state, ref_cs = _reference(ops_a)
    k = entry.GdnDecodeFused(dev, 1e-6)
    qa = _run(k, ops_a).clone()
    out_a = k.outputs(2)["out"].clone()
    qb = _run(k, ops_b).clone()
    out_b = k.outputs(2)["out"].clone()
    torch.cuda.synchronize()
    assert torch.equal(qa.view(torch.uint8), qb.view(torch.uint8)) and torch.equal(out_a, out_b), "two identical operand sets are byte-identical"
    assert torch.equal(ops_a["state"], ops_b["state"]) and torch.equal(ops_a["cs"], ops_b["cs"])
    assert torch.equal(ops_a["cs"], ref_cs), "the conv history shifts by one tap for every channel"
    live = ref_out.abs().max()
    err = (out_a.double() - ref_out).abs().max()
    assert err <= 2e-2 * live + 1e-3, f"out max |err| {err} vs max |ref| {live}"
    serr = (ops_a["state"][1:].double() - ref_state[1:].double()).abs().max()
    assert serr <= 1e-3 * ref_state.abs().max() + 1e-5, f"state max |err| {serr}"
    assert torch.isfinite(qa.float()).all()


def test_graph_replay_over_distinct_layers_matches_eager():
    dev = _device_or_skip()
    layers = [make_operands(rows=1, device=dev, seed=100 + i) for i in range(8)]
    eager = [make_operands(rows=1, device=dev, seed=100 + i) for i in range(8)]
    k = entry.GdnDecodeFused(dev, 1e-6)
    _run(k, layers[0])  # compile + warm up
    torch.cuda.synchronize()
    layers[0] = make_operands(rows=1, device=dev, seed=100)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for lay in layers:
            _run(k, lay)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    graph_state = [lay["state"].clone() for lay in layers]
    graph_q = k.outputs(1)["qout"].view(torch.uint8).clone()
    for lay in eager:
        _run(k, lay)
    torch.cuda.synchronize()
    eager_q = k.outputs(1)["qout"].view(torch.uint8).clone()
    assert torch.equal(graph_q, eager_q), "the last layer's fp8 output replays bitwise"
    for gs, lay in zip(graph_state, eager):
        assert torch.equal(gs, lay["state"]), "every layer's state advances identically under graph replay (per-layer operands re-bound per launch)"


# --- every knob arm is bitwise with the entry's default pick on the B4 set with a null row and non-monotone slots [3, 0, 4, 1] ---
B4_IDX = [3, 0, 4, 1]


def _snapshot(k, ops):
    q = _run(k, ops)
    torch.cuda.synchronize()
    b = k.outputs(ops["rows"])
    assert not b["counts"].any(), "the reader tickets return to zero"
    return dict(qout=q.view(torch.uint8).clone(), out=b["out"].view(torch.int16).clone(), state=ops["state"].clone(), cs=ops["cs"].clone())


def _bytes(t):
    """The tensor's bytes in the standard layout (a transposed conv-state view cannot be re-viewed as bytes in place)."""
    return t.contiguous().view(torch.uint8)


def _assert_bitwise(a, b, what):
    for key in ("qout", "out", "state", "cs"):
        assert torch.equal(_bytes(a[key]), _bytes(b[key])), f"{what}: {key} differs from the default arm"


def _default_snapshot(dev, seed=31):
    ops = make_operands(rows=4, slots=5, device=dev, seed=seed, idx=torch.tensor(B4_IDX, dtype=torch.int32, device=dev))
    return _snapshot(entry.GdnDecodeFused(dev, 1e-6), ops)


@pytest.mark.parametrize(
    "arm",
    [  # Cfg deltas from the one-CTA form (bulk, TPR 4); the pick at rows 4 is the streamed form on a part with fewer than 192 SMs
        dict(),  # the one-CTA form itself at rows 4 (two waves)
        dict(STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),  # the streamed form, explicit
        dict(STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=2),
        dict(STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0, PDL=True, PDL_RELEASE=True),
        dict(TPR=8),
        dict(TPR=8, MIN_CTAS_PER_SM=2, CTAS_PER_SM=2),  # register-resident under the 128 cap (spill-free with one tail; the bytes agree)
        dict(TPR=8, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=2),
        dict(TPR=8, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),
        dict(PDL=True),
        dict(PDL=True, PDL_RELEASE=True),
        dict(TPR=16),
        dict(TPR=2),
        dict(THREADS=128, TPR=1, WARPGROUP_SPLIT=False, STATE_STREAM=True),
        dict(WARPGROUP_SPLIT=False),
        dict(K_RESIDENT=False),
        dict(STATE_LOAD_PATH="ldg128", TPR=8),  # the register state path at its measured-best TPR
        dict(STATE_LOAD_PATH="ldg128", TPR=8, PDL=True),
        dict(STATE_LOAD_PATH="ldg128", TPR=2, PIN_ENTRY_LOADS=True),
        dict(STATE_LOAD_PATH="ldg128", TPR=16),
        dict(STATE_LOAD_PATH="ldg128", THREADS=128, TPR=1, WARPGROUP_SPLIT=False),
        dict(CONV_FIRST=True),  # the conv-operand copies ahead of the slab boxes, one issuing lane
        dict(CONV_FIRST=True, PDL=True),
        dict(CONV_FIRST=True, TPR=8),
        dict(CONV_FIRST=True, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),  # the streamed form with the lever
        dict(CONV_FIRST=True, WARPGROUP_SPLIT=False),  # the no-split form: warp 0's lane issues conv copies then the slab
        dict(CONV_FIRST=True, THREADS=128, TPR=1, WARPGROUP_SPLIT=False, STATE_STREAM=True),
        dict(SINGLE_TAIL=False),  # the twice-traced body (the default's A/B twin)
        dict(SINGLE_TAIL=False, CONV_FIRST=True),
        dict(SINGLE_TAIL=False, PDL=True, PDL_RELEASE=True),
        dict(SINGLE_TAIL=False, TPR=8),
        dict(SINGLE_TAIL=False, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),  # the streamed form, twice-traced
        dict(CONV_FIRST=True, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),
        dict(MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),  # the capped register-resident form = CAPPED_CFG (the rows >= 4 pick), explicit
        dict(SINGLE_TAIL=False, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),  # its twice-traced twin
        dict(SINGLE_TAIL=False, WARPGROUP_SPLIT=False),  # the no-split form, twice-traced
        dict(THREADS=128, TPR=1, WARPGROUP_SPLIT=False, STATE_STREAM=True),  # SINGLE_TAIL is a no-op on one warpgroup
        dict(STORE_PIPELINE=True),  # each warpgroup's half-boxes stored as soon as its pass 2 is published
        dict(STORE_PIPELINE=True, SINGLE_TAIL=False),
        dict(STORE_PIPELINE=True, CONV_FIRST=True),
        dict(STORE_PIPELINE=True, PDL=True, PDL_RELEASE=True),
        dict(STORE_PIPELINE=True, TPR=8),
        dict(STORE_PIPELINE=True, WARPGROUP_SPLIT=False),  # the no-split form: warpgroup 1 holds the second halves too
        dict(STORE_PIPELINE=True, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),
        dict(STORE_PIPELINE=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0),  # = PAIR_CFG (the rows-1 pick), explicit: two CTAs per head, the norm partial by st.async
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, SINGLE_TAIL=False),  # its twice-traced twin
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, CONV_FIRST=True),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, STORE_PIPELINE=True),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, CONV_FIRST=True, STORE_PIPELINE=True),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, STATE_STREAM=True, MIN_CTAS_PER_SM=2),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, PDL=True, PDL_RELEASE=True),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, TPR=8),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, TPR=16),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, WARPGROUP_SPLIT=False),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, THREADS=128, TPR=2, WARPGROUP_SPLIT=False),
    ],
    ids=lambda a: ",".join(f"{k}={v}" for k, v in a.items()) or "one-cta-form",
)
def test_knob_arms_the_part_holds_are_bitwise_with_the_default_pick_at_b4(arm):
    """Every knob arm this part's shared memory holds writes the bytes of the entry's pick; an arm it does not hold (a two-slab pad on a
    100 KB SM: three arms on the cc 12.0 readings, none on the cc 10.0 ones) is asserted to be the typed decline at construction, never a
    launch, and then reported as a SKIP with that reason, so the count shows which arms ran on the part."""
    dev = _device_or_skip()
    cfg = kcfg.make_cfg(**arm)
    try:
        kcfg.validate_cfg(replace(cfg, **_on(entry.device_shared_memory(dev))))
    except ValueError as exc:
        assert "do not fit the SM" in str(exc), exc
        with pytest.raises(ValueError, match="do not fit the SM"):
            entry.GdnDecodeFused(dev, 1e-6, cfg=cfg)
        pytest.skip(f"this part's shared memory does not hold the arm (declined by type at construction): {exc}")
    base = _default_snapshot(dev)
    ops = make_operands(rows=4, slots=5, device=dev, seed=31, idx=torch.tensor(B4_IDX, dtype=torch.int32, device=dev))
    got = _snapshot(entry.GdnDecodeFused(dev, 1e-6, cfg=cfg), ops)
    _assert_bitwise(base, got, str(arm))
    assert not got["qout"][1].any() and got["qout"][0].any(), "the null row is zero, the live rows written"


@pytest.mark.parametrize("rows", [1, 4])
def test_pdl_launch_knob_entry_is_bitwise_with_the_plain_pick(rows):
    """The entry's ``pdl=True`` (the serving chain's launch mode when the deployment asks for it) on the pick of ``rows`` -- the pair form
    at rows 1, the capped form at rows 4 -- writes the same bytes as the plain pick.  The stream is drained before the PDL launch: the
    kernel's caller obligation (no in-flight predecessor writes the state or the conv pool) holds in the test as in the chain."""
    dev = _device_or_skip()
    idx = torch.tensor(B4_IDX if rows == 4 else [3], dtype=torch.int32, device=dev)
    ops = make_operands(rows=rows, slots=5, device=dev, seed=37, idx=idx.clone())
    torch.cuda.synchronize()
    base = _snapshot(entry.GdnDecodeFused(dev, 1e-6), ops)
    ops = make_operands(rows=rows, slots=5, device=dev, seed=37, idx=idx.clone())
    k = entry.GdnDecodeFused(dev, 1e-6, pdl=True)
    assert k.cfg_for(rows).PDL and k.cfg_for(rows).PDL_RELEASE
    torch.cuda.synchronize()
    got = _snapshot(k, ops)
    _assert_bitwise(base, got, f"pdl=True entry at rows {rows}")
    assert got["qout"][0].any(), "the live row is written"


def test_tap_contiguous_conv_state_layout_is_bitwise_with_the_serving_layout():
    dev = _device_or_skip()
    base = _default_snapshot(dev)
    ops = make_operands(rows=4, slots=5, device=dev, seed=31, idx=torch.tensor(B4_IDX, dtype=torch.int32, device=dev))
    ops["cs"] = ops["cs"].contiguous()  # [slots, 10240, 3] with the taps contiguous per channel: the 9-copy arm of the conv operands
    assert tuple(ops["cs"].stride()) == (30720, 3, 1)
    for arm in (dict(STATE_LOAD_PATH="ldg128"), dict(STATE_LOAD_PATH="bulk"), dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0)):
        ops_p = dict(ops, cs=ops["cs"].clone(), state=make_operands(rows=4, slots=5, device=dev, seed=31)["state"])
        got = _snapshot(entry.GdnDecodeFused(dev, 1e-6, cfg=kcfg.make_cfg(**arm)), ops_p)
        _assert_bitwise(base, got, f"tap-contiguous conv state, {arm}")  # the values agree layout to layout (compared in one layout)


# ---------------------------------------------------------------------------------------------------------------------------
# the per-rank geometries on the device: the torch model and determinism per rows, every form family bitwise with the one-CTA form on a
# set with a null row and non-monotone slots, the rows of a step batch-invariant -- the 27B cells above are unchanged (the geometry is
# the compile key, and the default geometry's renderings are byte-identical to the pre-geometry kernels)
# ---------------------------------------------------------------------------------------------------------------------------
RANK_ROWS = [1, 2, 4, 8, 16]
# the form families (Cfg deltas from the one-CTA form): the three pick forms (pair / capped / streamed), the per-row thread mappings, the
# register state path, the 128-thread form, the PDL order, the twice-traced tail, the conv-first and store-pipeline issue orders, the
# no-split and non-resident-K bodies, and the pair with the mappings it is picked with
RANK_ARMS = [
    dict(),  # the one-CTA form itself, a second entry
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0),  # = PAIR_CFG: the rows-1 pick (rows x 2 V cluster CTAs)
    dict(MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),  # = CAPPED_CFG: the beyond-the-grid pick on the cc 10.0 readings
    dict(STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),  # = STREAMED_CFG
    dict(TPR=8),
    dict(TPR=16),
    dict(TPR=2),
    dict(TPR=8, MIN_CTAS_PER_SM=2, CTAS_PER_SM=2),
    dict(STATE_LOAD_PATH="ldg128", TPR=8),
    dict(THREADS=128, TPR=1, WARPGROUP_SPLIT=False, STATE_STREAM=True),
    dict(PDL=True, PDL_RELEASE=True),
    dict(SINGLE_TAIL=False),
    dict(CONV_FIRST=True),
    dict(STORE_PIPELINE=True),
    dict(WARPGROUP_SPLIT=False),
    dict(K_RESIDENT=False),
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, TPR=8),
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, CONV_FIRST=True, STORE_PIPELINE=True),
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, PDL=True, PDL_RELEASE=True),
]


def _rank_operands(g, rows, dev, seed, idx=None):
    """``make_operands`` at geometry ``g`` on ``dev``; ``idx`` (a slot per row, 0 = a null row) sizes the pools to hold it."""
    if idx is None:
        return make_operands(rows=rows, slots=rows + 1, device=dev, seed=seed, geometry=g)
    return make_operands(rows=rows, slots=max(idx) + 1, device=dev, seed=seed, geometry=g, idx=torch.tensor(idx, dtype=torch.int32, device=dev))


@pytest.mark.parametrize("rows", RANK_ROWS)
@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_rank_geometry_matches_the_torch_model_and_replays_bitwise(g, rows):
    """The entry's pick for ``rows`` at a per-rank geometry (the pair, one CTA per SM or the capped form by ``rows x V`` against this
    part's SM count: the module pick) against the torch model at the default geometry's bands (the function check), the conv history
    shifted bitwise, the reader tickets back at zero -- and twice on fresh identical operands: byte-identical (fp8 / bf16 outputs, every
    state slot, the conv pool)."""
    dev = _device_or_skip()
    k = entry.GdnDecodeFused(dev, 1e-6, geometry=g)
    cfg = k.cfg_for(rows)
    assert cfg.GEOM == g and replace(cfg, **_on(SM100_READING)) == kcfg.default_cfg(rows, k.sm_count, *k.readings, g)
    ops_a = _rank_operands(g, rows, dev, seed=500 + rows)
    ops_b = _rank_operands(g, rows, dev, seed=500 + rows)
    ref_out, ref_state, ref_cs = _reference(ops_a)
    snap_a = _snapshot(k, ops_a)
    snap_b = _snapshot(k, ops_b)
    _assert_bitwise(snap_a, snap_b, f"{_gid(g)} rows {rows}: two identical operand sets")
    assert torch.equal(ops_a["cs"], ref_cs), "the conv history shifts by one tap for every channel"
    out = snap_a["out"].view(torch.bfloat16)
    live = ref_out.abs().max()
    err = (out.double() - ref_out).abs().max()
    assert err <= 2e-2 * live + 1e-3, f"out max |err| {err} vs max |ref| {live}"
    serr = (ops_a["state"][1:].double() - ref_state[1:].double()).abs().max()
    assert serr <= 1e-3 * ref_state.abs().max() + 1e-5, f"state max |err| {serr}"
    assert torch.isfinite(snap_a["qout"].view(torch.float8_e4m3fn).float()).all()
    assert snap_a["qout"].any(), "the fp8 output is written"


@pytest.mark.parametrize("arm", RANK_ARMS, ids=lambda a: ",".join(f"{k}={v}" for k, v in a.items()) or "one-cta-form")
@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_rank_geometry_knob_arms_are_bitwise_with_the_one_cta_form(g, arm):
    """Every form family this part holds, at a per-rank geometry, writes the bytes of the explicit one-CTA form (``CFG`` at that geometry)
    on the 4-row set with a null row and non-monotone slots [3, 0, 4, 1]: the fp8 / bf16 outputs, every state slot, the conv pool; the
    null row zero, the tickets back at zero (at 1 K head per rank, 8 or 16 readers share one K head's conv history per row).  An arm the
    part's shared memory does not hold is its typed decline at construction, reported as a skip."""
    dev = _device_or_skip()
    cfg = kcfg.with_geometry(kcfg.make_cfg(**arm), g)  # the form compiled for the geometry (a form of another geometry beside geometry= is refused)
    try:
        kcfg.validate_cfg(replace(cfg, **_on(entry.device_shared_memory(dev))))
    except ValueError as exc:
        assert "do not fit the SM" in str(exc), exc
        with pytest.raises(ValueError, match="do not fit the SM"):
            entry.GdnDecodeFused(dev, 1e-6, cfg=cfg)
        pytest.skip(f"this part's shared memory does not hold the arm (declined by type at construction): {exc}")
    base = _snapshot(entry.GdnDecodeFused(dev, 1e-6, cfg=kcfg.with_geometry(kcfg.CFG, g)), _rank_operands(g, 4, dev, seed=31, idx=B4_IDX))
    got = entry.GdnDecodeFused(dev, 1e-6, cfg=cfg)
    assert got.geometry == g and got.cfg.GEOM == g
    got = _snapshot(got, _rank_operands(g, 4, dev, seed=31, idx=B4_IDX))
    _assert_bitwise(base, got, f"{_gid(g)} {arm}")
    assert not got["qout"][1].any() and got["qout"][0].any() and got["qout"][2].any() and got["qout"][3].any(), "the null row is zero, the live rows written"


@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_rank_geometry_rows_are_batch_invariant(g):
    """Batch invariance at a per-rank geometry: every row of a 4-row step (the entry's pick at rows 4, non-monotone slots) writes the
    bytes it writes alone (rows 1: the pair pick) from the same pools -- its fp8 / bf16 outputs, its state slot, its conv slot -- so a
    request's result does not depend on which requests share its step."""
    dev = _device_or_skip()
    k = entry.GdnDecodeFused(dev, 1e-6, geometry=g)
    idx = [2, 4, 1, 3]
    snap4 = _snapshot(k, _rank_operands(g, 4, dev, seed=61, idx=idx))
    for r in range(4):
        ops = _rank_operands(g, 4, dev, seed=61, idx=idx)  # the same draws; row r alone, the pools whole (the serving layout's views kept)
        one = dict(ops, rows=1, x=ops["x"][r : r + 1], z=ops["z"][r : r + 1], hidden=ops["hidden"][r : r + 1], idx=ops["idx"][r : r + 1])
        snap1 = _snapshot(k, one)
        assert torch.equal(snap1["qout"][0], snap4["qout"][r]) and torch.equal(snap1["out"][0], snap4["out"][r]), f"{_gid(g)} row {r}: the outputs"
        assert torch.equal(snap1["state"][idx[r]], snap4["state"][idx[r]]), f"{_gid(g)} row {r}: the state slot"
        assert torch.equal(_bytes(snap1["cs"][idx[r]]), _bytes(snap4["cs"][idx[r]])), f"{_gid(g)} row {r}: the conv slot"
