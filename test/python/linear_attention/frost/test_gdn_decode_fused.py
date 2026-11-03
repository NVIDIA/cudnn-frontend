# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused GDN decode step (``cudnn.linear_attention.frost.gdn_decode_fused``): the typed declines of its host entry (host-only,
CPU tensors), the plan-time-only compile key, the config knob validation (the occupancy pad on both part readings), the DSL-floor
decline, and -- on a cc 10.x device -- the null-slot contract, the reader tickets, determinism, CUDA-graph replay over distinct
layers, a loose torch reference of the fused math, and the bitwise agreement of every knob arm (the shared-memory state path, the
streamed two-CTAs-per-SM form, the PDL order, the TPR mappings, the 128-thread form, the tap-contiguous conv-state layout) with the
entry's default pick on one operand set that carries a null row and non-monotone slots.  The default's per-rows pick (one CTA per
SM while the rows x 48 grid fits the SMs, the streamed two-CTAs-per-SM form past that) is pinned on the host for two SM counts and
on the device."""

import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.linear_attention.frost import gdn_decode_fused as entry  # noqa: E402
from cudnn.linear_attention.frost.kernel import gdn_decode_fused_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

ROWS = 2
SLOTS = ROWS + 1
# the cc 10.0 part reading (228 KB per SM, 227 KB opt-in) and a hypothetical larger part (336 KiB per SM, 335 KiB opt-in): the pad must do
# what it claims on both
PART_READINGS = [(233472, 232448), (344064, 343040)]


def make_operands(rows=ROWS, device="cpu", slots=SLOTS, seed=0, **overrides):
    """The entry's operand set in the serving layout (x = the QKV part of a [rows, 16384] slab, z its Z part viewed [rows, 48, 128],
    cs the [slots, 3, 10240] conv buffer transposed to [slots, 10240, 3])."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g).to(device)

    qkvz = (rn(rows, 16384) * 2).bfloat16()
    conv = rn(slots, 3, 10240).bfloat16()
    ops = dict(
        rows=rows,
        x=qkvz[:, :10240],
        cs=conv.transpose(-1, -2),
        cw=(0.3 * rn(10240, 4)).bfloat16(),
        al=rn(48) * 0.5,
        dt=(rn(48) * 0.5).bfloat16(),
        state=rn(slots, 48, 128, 128) * 0.1,
        idx=torch.arange(1, rows + 1, dtype=torch.int32).to(device),
        z=qkvz[:, 10240:].view(rows, 48, 128),
        w=(1 + 0.2 * rn(128)).bfloat16(),
        scale=torch.tensor([0.02]).to(device),
        hidden=rn(rows, 5120).bfloat16(),
        baw=(rn(96, 5120) / math.sqrt(5120)).bfloat16(),
    )
    ops.update(overrides)
    return ops


def call_check(ops, device_cc=(10, 0)):
    return entry.check_support(
        ops["rows"],
        ops["x"],
        ops["cs"],
        ops["cw"],
        ops["al"],
        ops["dt"],
        ops["state"],
        ops["idx"],
        ops["z"],
        ops["w"],
        ops["scale"],
        ops["hidden"],
        ops["baw"],
        device_cc=device_cc,
    )


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
def test_occupancy_pad_caps_the_ctas_per_sm_on_both_part_readings(per_sm, optin, path):
    for n in (1, 2):
        c = kcfg.make_cfg(CTAS_PER_SM=n, STATE_LOAD_PATH=path, SMEM_PER_SM_BYTES=per_sm, SMEM_OPTIN_BYTES=optin)
        per_cta = c.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES
        assert (n + 1) * per_cta > per_sm, "one CTA more than allowed does not fit the SM"
        assert n * per_cta <= per_sm, "the allowed count does"
        assert c.SMEM_BYTES <= optin
        assert c.SMEM_PAD_BYTES % 16 == 0


def test_state_box_bank_map_is_conflict_free_at_every_tpr():
    """The 8 lanes of a 128-bit shared-memory phase read 8 distinct bank quads: bank = 4 rg + (128 / TPR) tpr + 4 c4 (mod 32)."""
    forms = [(256, tpr, False) for tpr in (2, 4, 8, 16)] + [(128, 1, False)] + [(256, tpr, True) for tpr in (4, 8, 16)] + [(128, 2, True)]
    for threads, tpr, two in forms:
        if True:
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
    with pytest.raises(NotImplementedError, match=r"9\.0"):
        call_check(make_operands(), device_cc=(9, 0))
    with pytest.raises(NotImplementedError, match=r"12\.0"):
        call_check(make_operands(), device_cc=(12, 0))
    assert entry.cc_supported((10, 0)) and entry.cc_supported((10, 3)) and not entry.cc_supported((9, 0)) and not entry.cc_supported(None)


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
        pytest.skip(f"the fused GDN decode step serves SM100-class parts (cc 10.x), this device is {cc}")
    msg = entry._dsl_version_error()
    if msg:
        pytest.skip(msg)
    return torch.device("cuda")


def _run(k, ops):
    return k(
        ops["rows"], ops["x"], ops["cs"], ops["cw"], ops["al"], ops["dt"], ops["state"], ops["idx"], ops["z"], ops["w"], ops["scale"], ops["hidden"], ops["baw"]
    )


def _reference(ops, eps=1e-6):
    """A plain torch (fp32 / fp64) model of the fused math; the kernel rounds exactly where the serving chain rounds (BA in bf16,
    the conv taps and SiLU in bf16, the core output in bf16, the normed value in bf16), the model's SiLU and transcendental forms
    are torch's -- so the comparison is loose, a function check, never a bit check."""
    rows = ops["rows"]
    x, cs, cw, al, dt, st, idx, z, w, scale, hid, baw = (ops[k] for k in ("x", "cs", "cw", "al", "dt", "state", "idx", "z", "w", "scale", "hidden", "baw"))
    out = torch.zeros(rows, 48, 128, dtype=torch.float64, device=x.device)
    state_new = st.clone()
    cs_new = cs.clone()
    for b in range(rows):
        slot = int(idx[b])
        if slot <= 0:
            continue
        taps = cs[slot].float()  # [10240, 3]
        xb = x[b].float()
        acc = torch.zeros(10240, dtype=torch.float32, device=x.device)
        for j in range(3):
            acc = acc + (taps[:, j] * cw[:, j].float()).bfloat16().float()
        acc = acc + (xb * cw[:, 3].float()).bfloat16().float()
        conv = torch.nn.functional.silu(acc).bfloat16().float()
        cs_new[slot, :, 0] = cs[slot, :, 1]
        cs_new[slot, :, 1] = cs[slot, :, 2]
        cs_new[slot, :, 2] = x[b]
        for h in range(48):
            kh = h // 3
            bval = (hid[b].float() @ baw[h].float()).bfloat16().float()
            aval = (hid[b].float() @ baw[48 + h].float()).bfloat16().float()
            av = aval + dt[h].float()
            soft = torch.nn.functional.softplus(av) if float(av) <= 20 else av
            decay = torch.exp(-torch.exp(al[h]) * soft)
            beta = torch.sigmoid(bval)
            q = conv[kh * 128 : (kh + 1) * 128]
            k_ = conv[2048 + kh * 128 : 2048 + (kh + 1) * 128]
            v = conv[4096 + h * 128 : 4096 + (h + 1) * 128]
            q = q * (128**-0.5 / torch.sqrt((q * q).sum() + 1e-6))
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
    assert k.cfg.SMEM_BYTES <= k.cfg.SMEM_OPTIN_BYTES and 2 * (k.cfg.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES) > k.cfg.SMEM_PER_SM_BYTES


def test_entry_picks_the_form_per_rows_on_this_device():
    dev = _device_or_skip()
    k = entry.GdnDecodeFused(dev, 1e-6)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    assert k.sm_count == sms and not k.cfg_given
    for rows in (1, 2, 3, 4, 8):
        want, got = kcfg.default_cfg(rows, sms), k.cfg_for(rows)
        assert (got.TWO_CTAS_PER_HEAD, got.MIN_CTAS_PER_SM, got.CTAS_PER_SM) == (want.TWO_CTAS_PER_HEAD, want.MIN_CTAS_PER_SM, want.CTAS_PER_SM), rows
        assert (got.STATE_LOAD_PATH, got.TPR, got.PDL, got.SINGLE_TAIL) == ("bulk", 4, False, True)
        assert (got.SMEM_PER_SM_BYTES, got.SMEM_OPTIN_BYTES) == (k.cfg.SMEM_PER_SM_BYTES, k.cfg.SMEM_OPTIN_BYTES)  # the part's readings
    pair = k.cfg_for(1)
    assert pair.TWO_CTAS_PER_HEAD and pair.SMEM_PAD_BYTES == 0 and 2 * (pair.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES) <= pair.SMEM_PER_SM_BYTES
    one = k.cfg_for(sms // (2 * kcfg.V_HEADS) + 1)  # the first rows count whose cluster grid outgrows the SMs (rows 2 on 148 / 152 SMs)
    assert not one.TWO_CTAS_PER_HEAD and 2 * (one.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES) > one.SMEM_PER_SM_BYTES  # one CTA per SM by shared memory
    two = k.cfg_for(sms // kcfg.V_HEADS + 1)  # the first rows count whose one-CTA grid outgrows the SMs
    assert two.MIN_CTAS_PER_SM == 2 and two.SMEM_PAD_BYTES == 0 and 2 * (two.SMEM_BYTES + kcfg.SMEM_CTA_RESERVE_BYTES) <= two.SMEM_PER_SM_BYTES


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
def test_knob_arms_are_bitwise_with_the_default_pick_at_b4(arm):
    dev = _device_or_skip()
    base = _default_snapshot(dev)
    ops = make_operands(rows=4, slots=5, device=dev, seed=31, idx=torch.tensor(B4_IDX, dtype=torch.int32, device=dev))
    got = _snapshot(entry.GdnDecodeFused(dev, 1e-6, cfg=kcfg.make_cfg(**arm)), ops)
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
