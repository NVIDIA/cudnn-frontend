# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused GDN verify step (``cudnn.linear_attention.frost.gdn_verify_fused``): the typed declines of its host entry (host-only,
CPU tensors), the plan-time-only compile key, the config knob validation and the per-T shared-memory budget (the occupancy pad on
both part readings), the DSL-floor decline, and -- on a cc 10.x device -- the semantics against a pure-torch reference of the fused
chain (the FP64 recurrence with the chain's rounding points; its tolerances are the ones the reference verify kernel was validated
with: the bf16 output within 2^-6 relative + 1e-3, fp8 bytes equal on > 99.9 % of the live outputs and bitwise equal to the static
quantization of the kernel's own bf16 output, the conv pool bitwise, written state slots within 1e-4, untouched slots untouched,
padding outputs zero) over requests 1 / 2 / 4 x tokens 2 / 4 / 6 with random acceptance counts, padding requests, skipped slot
writes and non-monotone slot ids, the two-CTAs-per-SM form at 8 and 16 requests, determinism, CUDA-graph replay over distinct
layers, the bitwise agreement of every knob arm with the entry's default pick, and the two-CTAs-per-head PAIR form (the pick at one
request on a 148- / 152-SM part) against the one-CTA form at one request.  When ``CUDNN_FROST_GDN_VERIFY_REFERENCE_DIR``
names a directory of saved operand sets with the reference verify kernel's outputs (``set_<tag>.pt`` / ``ref_<tag>.pt``), every set
is replayed and compared BYTE for byte (fp8 output, bf16 output, every fp32 state slot, the conv pool, the reader tickets)."""

import math
import os
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.linear_attention.frost import gdn_verify_fused as entry  # noqa: E402
from cudnn.linear_attention.frost.kernel import gdn_verify_fused_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

# the cc 10.0 / 10.3 part reading (228 KB per SM, 227 KB opt-in) and a hypothetical larger part (336 KiB per SM, 335 KiB opt-in): the pad
# must do what it claims on both
PART_READINGS = [(233472, 232448), (344064, 343040)]
REFERENCE_DIR_ENV = "CUDNN_FROST_GDN_VERIFY_REFERENCE_DIR"
FP8_MAX = 448.0
AMBIGUOUS_REL = 2.0**-16


# ---------------------------------------------------------------------------------------------------------------------------
# operands (the reference verify kernel's test distributions) and the torch reference of the fused chain
# ---------------------------------------------------------------------------------------------------------------------------
def make_operands(requests=2, tokens=4, device="cpu", seed=0, accepted="random", pad="none", skip="none", param_dtypes=None, slots=None):
    """One seeded verify step in the serving layout: ``qkv`` / ``z`` strided views of one [rows, 16384] QKVZ slab, the conv pool
    stored ``[slots, T + 2, 10240]`` and viewed ``[slots, 10240, T + 2]`` (channel-contiguous), distinct random (non-monotone) live
    slots, ``hidden`` in {-1, 0, 1} and ``ba_weight`` in k / 256 (the BA dot exact before its bf16 rounding).  ``accepted``: random in
    [1, T], ``full`` (T) or ``one``; ``pad``: ``last`` / ``mid`` makes that request a padding one (every slot 0); ``skip``: ``first`` /
    ``mid`` / ``last`` zeroes one slot (t >= 1, never the source slot) of every live request -- a skipped state write."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a_dt, dt_dt, w_dt = param_dtypes or (torch.float32, torch.bfloat16, torch.bfloat16)
    rows = requests * tokens
    slots = slots or rows + 3
    L = tokens + kcfg.CONV_WIDTH - 2

    def rn(*shape):
        return torch.randn(*shape, generator=g)

    qkvz = rn(rows, 16384).bfloat16().to(device)  # moved as one slab so qkv / z stay strided views of it (the serving layout)
    conv = rn(slots, L, kcfg.QKV_CHANNELS).bfloat16().to(device)
    idx = (torch.randperm(slots - 1, generator=g)[:rows] + 1).int().view(requests, tokens)
    if accepted == "full":
        acc = torch.full((requests,), tokens, dtype=torch.int32)
    elif accepted == "one":
        acc = torch.ones(requests, dtype=torch.int32)
    else:
        acc = torch.randint(1, tokens + 1, (requests,), generator=g, dtype=torch.int32)
    if pad == "last" and requests > 1:
        idx[-1] = 0
    if pad == "mid" and requests > 1:
        idx[requests // 2] = 0
    if skip != "none" and tokens >= 2:
        for n in range(requests):
            if int(idx[n, 0]) <= 0:
                continue
            want = {"first": 1, "mid": tokens // 2, "last": tokens - 1}[skip]
            if want == int(acc[n]) - 1:  # never the source slot
                want = want + 1 if want + 1 < tokens else want - 1
            if 1 <= want < tokens and want != int(acc[n]) - 1:
                idx[n, want] = 0
    ops = dict(
        qkv=qkvz[:, : kcfg.QKV_CHANNELS],
        z=qkvz[:, kcfg.QKV_CHANNELS :].view(rows, kcfg.V_HEADS, kcfg.HEAD_DIM),
        hidden=torch.randint(-1, 2, (rows, kcfg.HIDDEN), generator=g).bfloat16().to(device),
        ba_weight=(torch.randint(-8, 9, (2 * kcfg.V_HEADS, kcfg.HIDDEN), generator=g).float() / 256).bfloat16().to(device),
        A_log=(rn(kcfg.V_HEADS) * 0.5).to(a_dt).to(device),
        dt_bias=(rn(kcfg.V_HEADS) * 0.5).to(dt_dt).to(device),
        ssm_state=(rn(slots, kcfg.V_HEADS, kcfg.HEAD_DIM, kcfg.HEAD_DIM) * 0.1).to(device),
        state_indices=idx.to(device),
        cu_seqlens=torch.arange(0, rows + 1, tokens, dtype=torch.int32).to(device),
        num_accepted_tokens=acc.to(device),
        norm_weight=(1 + 0.2 * rn(kcfg.HEAD_DIM)).to(w_dt).to(device),
        out_scale=torch.tensor([0.05]).to(device),
        conv_state=conv.transpose(1, 2),
        conv_weight=(rn(kcfg.QKV_CHANNELS, kcfg.CONV_WIDTH) * 0.3).bfloat16().to(device),
    )
    return ops


def _conv_silu(window, weight):
    """``window [..., C, K]`` (oldest tap first), ``weight [C, K]`` bf16 -> (the exact SiLU in FP64, its bf16): every tap product
    rounded to bf16, the taps summed in FP32 in tap order."""
    prod = (window.float() * weight.float()).bfloat16().float()
    acc = prod[..., 0]
    for j in range(1, prod.shape[-1]):
        acc = acc + prod[..., j]
    a = acc.double()
    s = a / (1.0 + torch.exp(-a))
    return s, s.float().bfloat16()


def _windows(ops):
    """``[(request, row slice, conv slot, source slot, window [T, C, K])]`` of every live request."""
    requests, tokens = ops["state_indices"].shape
    K = ops["conv_weight"].shape[1]
    idx = ops["state_indices"].tolist()
    acc = ops["num_accepted_tokens"].tolist()
    out = []
    for n in range(requests):
        a = acc[n]
        src = idx[n][a - 1]
        if src <= 0 or idx[n][0] <= 0:
            continue
        rows = slice(n * tokens, (n + 1) * tokens)
        seq = torch.cat([ops["conv_state"][idx[n][0]][:, a - 1 : a + K - 2], ops["qkv"][rows].T], dim=1)
        out.append((n, rows, idx[n][0], src, seq.unfold(1, K, 1).transpose(0, 1)))
    return out


def _ambiguous(s):
    bits = s.float().view(torch.int32)
    mid = ((bits & ~0xFFFF) | 0x8000).view(torch.float32).double()
    return (s - mid).abs() <= AMBIGUOUS_REL * s.abs()


def condition_conv_inputs(ops, max_rounds=32):
    """Move the newest conv input of every live (token, channel) whose exact SiLU lies within 2^-16 relative of a bf16 rounding
    midpoint (the kernels evaluate it with ``ex2.approx``) until none is left, so the reference's bf16 q / k / v are the kernel's."""
    moved = 0
    for _ in range(max_rounds):
        hits = 0
        for _, rows, _, _, win in _windows(ops):
            s, _ = _conv_silu(win, ops["conv_weight"])
            bad = _ambiguous(s)
            if bad.any():
                x = ops["qkv"][rows]
                v = x[bad].float()
                x[bad] = (v + 0.0625 * v.abs().clamp_min(1.0)).bfloat16()
                hits += int(bad.sum())
        if hits == 0:
            return moved
        moved += hits
    raise AssertionError("conv input conditioning did not converge")


def quant_fp8(x, scale):
    """Static per-tensor FP8: ``satfinite(x * fp32(1 / scale))`` rounded to nearest even."""
    inv = 1.0 / scale.float().reshape(())
    return (x.float() * inv).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def reference(ops, eps=1e-6):
    """The fused chain on ``ops``: ``(out_bf16, out_fp8, ssm_state, conv_state, live_rows)``.  Rounding points as the fused kernels':
    bf16 tap products and SiLU, the BA projection to bf16, the recurrence in FP64, the core output to bf16, the normed value to bf16."""
    hv, dv, dk = kcfg.V_HEADS, kcfg.HEAD_DIM, kcfg.HEAD_DIM
    hk = kcfg.K_HEADS
    K = ops["conv_weight"].shape[1]
    rows = ops["qkv"].shape[0]
    dev = ops["qkv"].device
    out = torch.zeros(rows, hv, dv, dtype=torch.bfloat16, device=dev)
    live = torch.zeros(rows, dtype=torch.bool, device=dev)
    ssm = ops["ssm_state"].clone()
    conv = ops["conv_state"].clone()
    ba = (ops["hidden"].double() @ ops["ba_weight"].double().T).float().bfloat16().double()
    A_log, dt_bias, w, z = ops["A_log"].double(), ops["dt_bias"].double(), ops["norm_weight"].double(), ops["z"].double()
    idx = ops["state_indices"].tolist()
    acc = ops["num_accepted_tokens"].tolist()
    for n, rows_n, cslot, src, win in _windows(ops):
        _, x = _conv_silu(win, ops["conv_weight"])
        x = x.double()
        a = acc[n]
        tokens = win.shape[0]
        old = ops["conv_state"][cslot]
        conv[cslot] = torch.cat([old[:, a : a + K - 2], ops["qkv"][rows_n].T], dim=1)
        h = ops["ssm_state"][src].double()
        for t in range(tokens):
            r = rows_n.start + t
            q = x[t, : hk * dk].view(hk, dk)
            k = x[t, hk * dk : 2 * hk * dk].view(hk, dk)
            v = x[t, 2 * hk * dk :].view(hv, dv)
            q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * dk**-0.5
            k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            q = q.repeat_interleave(hv // hk, 0)
            k = k.repeat_interleave(hv // hk, 0)
            gate = -A_log.exp() * torch.nn.functional.softplus(ba[r, hv:] + dt_bias)
            beta = torch.sigmoid(ba[r, :hv])
            h = h * gate.exp()[:, None, None]
            innov = (v - (h * k[:, None, :]).sum(-1)) * beta[:, None]
            h = h + innov[:, :, None] * k[:, None, :]
            if idx[n][t] > 0:
                ssm[idx[n][t]] = h.float()
            core = (h * q[:, None, :]).sum(-1).float().bfloat16().double()
            inv = torch.rsqrt(core.square().mean(-1, keepdim=True) + eps)
            zr = z[r]
            out[r] = (core * inv * w * zr * torch.sigmoid(zr)).float().bfloat16()
            live[r] = True
    return out, quant_fp8(out, ops["out_scale"]), ssm, conv, live


def pools_before(ops):
    """Clones of the two pools the kernel updates in place (the reference starts from them; the kernel's results are compared)."""
    return ops["ssm_state"].clone(), ops["conv_state"].clone()


def check_against_reference(ops, q, out, ssm0, conv0, eps=1e-6):
    """The reference verify kernel's acceptance bars against the torch reference: ``ssm0`` / ``conv0`` are the pools BEFORE the launch
    (the reference runs on them), ``ops`` holds the kernel's updated pools."""
    ssm, conv = ops["ssm_state"], ops["conv_state"]
    ref_out, ref_q, ref_ssm, ref_conv, live = reference(dict(ops, ssm_state=ssm0, conv_state=conv0), eps)
    diff = (out.float() - ref_out.float()).abs()
    bad = diff > ref_out.float().abs() * 2**-6 + 1e-3
    assert not bad.any(), (int(bad.sum()), float(diff.max()))
    same = (q[live].view(torch.uint8) == ref_q[live].view(torch.uint8)).float().mean()
    assert float(same) > 0.999, float(same)
    assert torch.equal(
        q.view(torch.uint8), quant_fp8(out, ops["out_scale"]).view(torch.uint8)
    ), "fp8 is the static quantization of the kernel's own bf16 output"
    assert not q[~live].view(torch.uint8).any() and not out[~live].any(), "padding rows are zero"
    assert torch.equal(conv, ref_conv), "the conv pool shifts bitwise"
    idx = ops["state_indices"]
    written = torch.zeros(ssm.shape[0], dtype=torch.bool, device=ssm.device)
    for n, _, _, _, _ in _windows(ops):
        for t in range(idx.shape[1]):
            if int(idx[n, t]) > 0:
                written[int(idx[n, t])] = True
    assert torch.equal(ssm[~written], ssm0[~written]), "untouched slots (padding requests, skipped writes, unreferenced) stay"
    err = float((ssm[written] - ref_ssm[written]).abs().max()) if written.any() else 0.0
    assert err < 1e-4, err


# ---------------------------------------------------------------------------------------------------------------------------
# host-only
# ---------------------------------------------------------------------------------------------------------------------------
def call_check(ops, device_cc=(10, 0), cfg=None):
    return entry.check_support(**ops, device_cc=device_cc, cfg=cfg)


def test_served_geometry_passes_the_host_checks():
    call_check(make_operands())
    call_check(make_operands(requests=1, tokens=8))
    call_check(make_operands(requests=4, tokens=6, param_dtypes=(torch.bfloat16, torch.float32, torch.float32)))


def test_default_cfg_picks_the_pair_then_the_one_cta_form_then_the_two_ctas_per_sm_form():
    cfg = kcfg.CFG
    assert (cfg.TPR, cfg.NBUF, cfg.STATE_STREAM, cfg.MIN_CTAS_PER_SM, cfg.CTAS_PER_SM, cfg.BA_PREFETCH_CHUNKS) == (4, 3, False, 1, 1, 3)
    assert (cfg.PDL, cfg.PDL_RELEASE, cfg.K_RESIDENT, cfg.NORM_FORM, cfg.STATE_LOAD_PATH) == (False, False, True, None, "bulk")
    assert (cfg.STATE_STORE_PATH, cfg.BULK_ISSUE, cfg.TWO_CTAS_PER_HEAD, cfg.ROLLED_TOKENS) == ("bulk", "single", False, True)
    assert (cfg.R, cfg.C, cfg.RG, cfg.BOX_ROWS, cfg.GROUPS, cfg.CTAS_PER_HEAD, cfg.CTA_ROWS, cfg.V_PERM_STRIDE, cfg.TICKET_LIMIT) == (
        2,
        32,
        8,
        16,
        4,
        1,
        128,
        8,
        2,
    )
    two = kcfg.TWO_CTA_CFG
    assert two.MIN_CTAS_PER_SM == 2 and two.CTAS_PER_SM == 0 and two.PDL is False and not two.TWO_CTAS_PER_HEAD
    pair = kcfg.PAIR_CFG
    assert pair == replace(cfg, TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True) and pair.PDL is False
    assert (pair.CTAS_PER_HEAD, pair.CTA_ROWS, pair.R, pair.C, pair.RG, pair.BOX_ROWS, pair.V_PERM_STRIDE, pair.GROUPS) == (2, 64, 1, 32, 8, 8, 16, 4)
    assert (pair.STATE_BOX_BYTES, pair.STATE_SLAB_BYTES_CTA, pair.TICKET_LIMIT, pair.STATE_BOX_PITCH_ELEMS) == (4096, 32768, 5, 1028)
    for sm_count in (148, 152):
        assert kcfg.default_cfg(1, sm_count) == pair, sm_count  # 96 cluster CTAs have an SM each
        for requests in (2, 3):
            assert kcfg.default_cfg(requests, sm_count) == cfg, (requests, sm_count)  # 192 would not; 96 / 144 one-CTA CTAs do
        for requests in (4, 8, 16):
            assert kcfg.default_cfg(requests, sm_count) == two, (requests, sm_count)
        pairs = sm_count // (2 * kcfg.V_HEADS)  # the last requests count whose cluster grid has an SM per CTA
        fits = sm_count // kcfg.V_HEADS  # the last requests count whose one-CTA grid fits
        assert kcfg.default_cfg(pairs, sm_count) == pair and kcfg.default_cfg(pairs + 1, sm_count) == cfg, sm_count
        assert kcfg.default_cfg(fits, sm_count) == cfg and kcfg.default_cfg(fits + 1, sm_count) == two, sm_count
    assert kcfg.default_cfg(2, 192) == pair and kcfg.default_cfg(3, 192) == cfg  # a 192-SM part would pair two requests too
    assert kcfg.default_cfg(1, 95) == cfg  # fewer SMs than the pair's 96 CTAs: the one-CTA form
    with pytest.raises(ValueError, match="requests"):
        kcfg.default_cfg(0, 148)
    with pytest.raises(ValueError, match="SM count"):
        kcfg.default_cfg(1, 0)


def test_norm_form_follows_the_reference_ctas_policy_and_resolves_per_requests():
    assert kcfg.norm_form_for(1) == "ctas2" and kcfg.norm_form_for(2) == "ctas1" and kcfg.norm_form_for(16) == "ctas1"
    assert kcfg.norm_chain_rows("ctas2") == 4 and kcfg.norm_chain_rows("ctas1") == 8
    one = kcfg.resolve_cfg(kcfg.CFG, 1)
    two = kcfg.resolve_cfg(kcfg.CFG, 2)
    assert (one.NORM_FORM, one.NORM_CHAINS, one.NORM_CHAIN_ROWS) == ("ctas2", 32, 4)
    assert (two.NORM_FORM, two.NORM_CHAINS, two.NORM_CHAIN_ROWS) == ("ctas1", 16, 8)
    pinned = kcfg.make_cfg(NORM_FORM="ctas1")
    assert kcfg.resolve_cfg(pinned, 1) == pinned  # a given form is kept
    with pytest.raises(ValueError, match="requests"):
        kcfg.resolve_cfg(kcfg.CFG, 0)


def test_host_entry_keeps_the_one_cta_form_and_a_given_cfg_overrides_the_pick():
    k = entry.GdnVerifyFused("cpu", 1e-6)
    assert k.sm_count is None and not k.cfg_given
    assert k.cfg_for(1).NORM_FORM == "ctas2" and k.cfg_for(4).NORM_FORM == "ctas1"
    assert replace(k.cfg_for(4), NORM_FORM=None) == kcfg.CFG  # no CUDA device to pick by: the one-CTA form at every requests count
    k.sm_count = 148  # a 148-SM reading: the pick follows requests (the device cell below checks the reading itself)
    assert replace(k.cfg_for(1), NORM_FORM=None) == kcfg.PAIR_CFG and k.cfg_for(1).NORM_FORM == "ctas2"
    assert replace(k.cfg_for(3), NORM_FORM=None) == kcfg.CFG and replace(k.cfg_for(4), NORM_FORM=None) == kcfg.TWO_CTA_CFG
    given = kcfg.make_cfg(TPR=8, NBUF=1)
    k8 = entry.GdnVerifyFused("cpu", 1e-6, cfg=given)
    k8.sm_count = 148
    assert k8.cfg_given and replace(k8.cfg_for(1), NORM_FORM=None) == replace(k8.cfg_for(8), NORM_FORM=None) == given


def test_pdl_launch_knobs_apply_on_top_of_the_pick_and_of_a_given_cfg():
    """``pdl`` / ``pdl_release`` are launch knobs on top of whatever form serves: ``pdl=True`` sets ``PDL`` and ``PDL_RELEASE`` on the pick
    of every requests count (the one-CTA and the two-CTA forms; the release follows ``pdl`` unless said otherwise), a knob left None
    keeps the form's own fields, and the knob changes the compile key (two compiled kernels per form)."""
    plain, pdl = entry.GdnVerifyFused("cpu", 1e-6), entry.GdnVerifyFused("cpu", 1e-6, pdl=True)
    plain.sm_count = pdl.sm_count = 148
    for requests in (1, 3, 4, 8):
        got, ref = pdl.cfg_for(requests), plain.cfg_for(requests)
        assert (got.PDL, got.PDL_RELEASE, ref.PDL, ref.PDL_RELEASE) == (True, True, False, False)
        assert got == replace(ref, PDL=True, PDL_RELEASE=True), f"requests {requests}: the knob changes the launch fields only"
    k = entry.GdnVerifyFused("cpu", 1e-6, pdl=True, pdl_release=False)
    assert (k.cfg_for(1).PDL, k.cfg_for(1).PDL_RELEASE) == (True, False)
    given = kcfg.make_cfg(PDL=True, PDL_RELEASE=True)
    assert entry.GdnVerifyFused("cpu", 1e-6, cfg=given).cfg_for(1).PDL is True
    assert entry.GdnVerifyFused("cpu", 1e-6, cfg=given, pdl=False).cfg_for(1).PDL is False
    ops = make_operands(2, 4)
    assert _key(plain, ops) != _key(pdl, ops)


def test_cfg_knobs_validate_and_the_smem_budget_follows_the_tokens():
    for tpr in (2, 4, 8, 16):
        c = kcfg.make_cfg(TPR=tpr, NBUF=1)
        assert c.R * c.C * kcfg.THREADS == 128 * 128 and c.C % 8 == 0 and c.RG * c.TPR == 32 and c.BOX_ROWS == c.R * kcfg.WARPS
    with pytest.raises(ValueError, match="TPR"):
        kcfg.make_cfg(TPR=3)
    with pytest.raises(ValueError, match="NBUF"):
        kcfg.make_cfg(NBUF=5)
    with pytest.raises(ValueError, match="STATE_STREAM"):
        kcfg.make_cfg(STATE_STREAM=True, NBUF=3)
    with pytest.raises(ValueError, match="MIN_CTAS_PER_SM"):
        kcfg.make_cfg(MIN_CTAS_PER_SM=3)
    with pytest.raises(ValueError, match="CTAS_PER_SM"):
        kcfg.make_cfg(CTAS_PER_SM=3)
    with pytest.raises(ValueError, match="contradicts"):
        kcfg.make_cfg(MIN_CTAS_PER_SM=2, CTAS_PER_SM=1)
    with pytest.raises(ValueError, match="BA_PREFETCH_CHUNKS"):
        kcfg.make_cfg(BA_PREFETCH_CHUNKS=0)
    with pytest.raises(ValueError, match="NORM_FORM"):
        kcfg.make_cfg(NORM_FORM="ctas4")
    with pytest.raises(ValueError, match="STATE_LOAD_PATH"):
        kcfg.make_cfg(STATE_LOAD_PATH="ldg128")
    with pytest.raises(ValueError, match="STATE_STORE_PATH"):
        kcfg.make_cfg(STATE_STORE_PATH="tma")
    with pytest.raises(ValueError, match="stg"):
        kcfg.make_cfg(STATE_STORE_PATH="stg")  # the default ring depth 3: the stg path keeps the landing slab only
    with pytest.raises(ValueError, match="stg"):
        kcfg.make_cfg(STATE_STORE_PATH="stg", NBUF=1, STATE_STREAM=True)
    with pytest.raises(ValueError, match="BULK_ISSUE"):
        kcfg.make_cfg(BULK_ISSUE="all")
    stg = kcfg.make_cfg(STATE_STORE_PATH="stg", NBUF=1)
    assert not stg.BULK_STORE and not stg.SPREAD_ISSUE and stg.natural_smem_bytes(6) == 93344
    assert kcfg.make_cfg(BULK_ISSUE="spread").SPREAD_ISSUE
    with pytest.raises(ValueError, match="SMEM_PER_SM_BYTES"):
        kcfg.make_cfg(SMEM_PER_SM_BYTES=100000, SMEM_OPTIN_BYTES=200000)
    assert kcfg.conv_bulk_bytes(6) == 9984 and kcfg.conv_bulk_copies(6) == 30
    assert kcfg.CONV_MBAR_ARRIVERS == 1 and kcfg.STATE_MBAR_ARRIVERS == 1 and kcfg.STATE_SLAB_BYTES == 65536
    one = kcfg.CFG
    assert one.STATE_BOX_BYTES == 8192 and one.STATE_BOX_PITCH_ELEMS == 2052 and one.SLAB_SMEM_ELEMS == 8 * 2052 and one.RING_SMEM_ELEMS == 3 * 8 * 2052
    assert one.natural_smem_bytes(6) == 224672 and one.smem_pad_bytes(6) == 0  # the ring alone keeps one CTA per SM
    assert one.max_tokens() == 8 and one.natural_smem_bytes(8) <= one.SMEM_OPTIN_BYTES < one.natural_smem_bytes(9)
    kcfg.validate_tokens(one, 8)
    with pytest.raises(NotImplementedError, match="at most 8"):
        kcfg.validate_tokens(one, 9)
    with pytest.raises(NotImplementedError, match="at least 2"):
        kcfg.validate_tokens(one, 1)
    two = kcfg.TWO_CTA_CFG
    assert two.natural_smem_bytes(6) == 93344 and 2 * (two.smem_bytes(6) + kcfg.SMEM_CTA_RESERVE_BYTES) <= two.SMEM_PER_SM_BYTES
    assert kcfg.make_cfg(NBUF=2).natural_smem_bytes(6) == 159008
    res8 = kcfg.make_cfg(TPR=8, NBUF=1, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0)
    assert res8.STATE_BOX_BYTES == 16384 and res8.STATE_BOX_PITCH_ELEMS == 4100 and res8.natural_smem_bytes(6) == 94048
    # the two-CTAs-per-head pair: half the rows per CTA, the same box pad, the norm exchange's slots and stages
    pair = kcfg.PAIR_CFG
    assert pair.natural_smem_bytes(6) == 126512 and pair.smem_pad_bytes(6) == 0 and pair.max_tokens() == 35
    assert (
        pair.natural_smem_bytes(6) == one.natural_smem_bytes(6) - 3 * 8 * (16 * 512 + 16) + 3 * 8 * (8 * 512 + 16) + 2 * 16 * 4 + 2 * 8
    )  # half-size boxes + the exchange's slots and stages
    assert 2 * (pair.smem_bytes(6) + kcfg.SMEM_CTA_RESERVE_BYTES) > pair.SMEM_PER_SM_BYTES  # one CTA per SM by shared memory alone
    assert (kcfg.NORM_MBAR_ARRIVERS, kcfg.NORM_MBAR_STAGES, kcfg.NORM_RECV_MAX_FLOATS) == (1, 2, 16)
    for form, chains, per_cta in (("ctas2", 32, 16), ("ctas1", 16, 8)):
        p = kcfg.resolve_cfg(pair, 1 if form == "ctas2" else 2)
        assert (p.NORM_FORM, p.NORM_CHAINS, p.NORM_CHAINS_CTA, p.NORM_XFER_FLOATS, p.NORM_XFER_BYTES) == (form, chains, per_cta, per_cta, 4 * per_cta)
        assert p.CTA_ROWS % p.NORM_CHAIN_ROWS == 0  # a chain never straddles the CTA boundary
    assert kcfg.resolve_cfg(one, 1).NORM_XFER_FLOATS == 0 and one.NORM_XFER_BYTES == 0
    with pytest.raises(ValueError, match="does not tile"):
        kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, TPR=2)  # 64 rows x 2 threads per row < 256 threads
    with pytest.raises(ValueError, match="ROLLED_TOKENS"):
        kcfg.make_cfg(ROLLED_TOKENS=True, STATE_STREAM=True, NBUF=1)  # no register-resident rows to carry
    assert (
        not kcfg.make_cfg().ROLLED_TOKENS and kcfg.PAIR_CFG.ROLLED_TOKENS and kcfg.CFG.ROLLED_TOKENS and not kcfg.TWO_CTA_CFG.ROLLED_TOKENS
    )  # the knob is off in the field defaults, on in the two register-resident picks
    pair8 = kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, TPR=8, CTAS_PER_SM=0)
    assert (pair8.R, pair8.C, pair8.RG, pair8.BOX_ROWS, pair8.V_PERM_STRIDE, pair8.STATE_BOX_BYTES) == (2, 16, 4, 16, 8, 8192)
    assert kcfg.make_cfg(TWO_CTAS_PER_HEAD=True, TPR=16, CTAS_PER_SM=0).R == 4


@pytest.mark.parametrize("per_sm, optin", PART_READINGS)
@pytest.mark.parametrize("tokens", (2, 6))
def test_occupancy_pad_caps_the_ctas_per_sm_on_both_part_readings(per_sm, optin, tokens):
    for n, base in ((1, dict(NBUF=1)), (2, dict(NBUF=1, STATE_STREAM=True, MIN_CTAS_PER_SM=2))):
        c = kcfg.make_cfg(CTAS_PER_SM=n, SMEM_PER_SM_BYTES=per_sm, SMEM_OPTIN_BYTES=optin, **base)
        per_cta = c.smem_bytes(tokens) + kcfg.SMEM_CTA_RESERVE_BYTES
        assert (n + 1) * per_cta > per_sm, "one CTA more than allowed does not fit the SM"
        assert n * per_cta <= per_sm, "the allowed count does"
        assert c.smem_bytes(tokens) <= optin and c.smem_pad_bytes(tokens) % 16 == 0


def test_state_box_bank_map_is_conflict_free_at_every_tpr():
    """The 8 lanes of a 128-bit shared-memory phase read 8 distinct bank quads: bank = 4 rg + (128 / TPR) tpr + 4 c4 (mod 32) -- on the
    one-CTA form and on the two-CTAs-per-head pair (half-size boxes, the same 16-B pad)."""
    for tpr, two in [(t, False) for t in (2, 4, 8, 16)] + [(t, True) for t in (4, 8, 16)]:
        c = kcfg.make_cfg(TPR=tpr, NBUF=1, TWO_CTAS_PER_HEAD=two, CTAS_PER_SM=0 if two else 1)
        for c4 in range(c.C // 4):
            for m in range(c.R):
                for phase in range(4):
                    quads = set()
                    for lane in range(8 * phase, 8 * phase + 8):
                        tp, rg = lane // c.RG, lane % c.RG
                        word = rg * c.STATE_BOX_PITCH_ELEMS + m * 128 + tp * c.C + 4 * c4
                        quads.add((word % 32) // 4)
                    assert len(quads) == 8, (tpr, two, c4, m, phase, sorted(quads))


@pytest.mark.parametrize(
    "mutation, exc, token",
    [
        (lambda o: dict(qkv=torch.zeros(8, 8192, dtype=torch.bfloat16)), NotImplementedError, "qkv"),
        (lambda o: dict(qkv=o["qkv"].to(torch.float16)), TypeError, "qkv"),
        (lambda o: dict(qkv=torch.zeros(8, 10248, dtype=torch.bfloat16)[:, 1:10241]), ValueError, "qkv"),
        (lambda o: dict(qkv=torch.zeros(8, 10244, dtype=torch.bfloat16)[:, :10240]), NotImplementedError, "qkv"),
        (lambda o: dict(qkv=o["qkv"][:4]), NotImplementedError, "rows"),
        (lambda o: dict(z=torch.zeros(8, 48, 64, dtype=torch.bfloat16)), NotImplementedError, "z"),
        (lambda o: dict(hidden=torch.zeros(8, 4096, dtype=torch.bfloat16)), NotImplementedError, "hidden"),
        (lambda o: dict(ba_weight=torch.zeros(96, 4096, dtype=torch.bfloat16)), NotImplementedError, "ba_weight"),
        (lambda o: dict(A_log=torch.zeros(48, dtype=torch.float16)), TypeError, "A_log"),
        (lambda o: dict(dt_bias=torch.zeros(47, dtype=torch.bfloat16)), NotImplementedError, "dt_bias"),
        (lambda o: dict(ssm_state=torch.zeros(11, 48, 64, 64)), NotImplementedError, "ssm_state"),
        (lambda o: dict(ssm_state=torch.zeros(11, 48, 128, 128, dtype=torch.float16)), TypeError, "ssm_state"),
        (lambda o: dict(ssm_state=torch.zeros(11, 48, 128, 128).transpose(2, 3)), NotImplementedError, "head slabs"),
        (lambda o: dict(state_indices=torch.ones(8, dtype=torch.int32)), NotImplementedError, "state_indices"),
        (lambda o: dict(state_indices=torch.ones(2, 4, dtype=torch.int64)), TypeError, "state_indices"),
        (lambda o: dict(state_indices=torch.ones(4, 2, dtype=torch.int32).T), NotImplementedError, "contiguous in state_indices"),
        (lambda o: dict(state_indices=torch.ones(entry.MAX_REQUESTS + 1, 4, dtype=torch.int32)), NotImplementedError, "65535"),  # CUDA's grid Y limit
        (
            lambda o: dict(state_indices=torch.ones(2, 1, dtype=torch.int32), conv_state=torch.zeros(5, 3, 10240, dtype=torch.bfloat16).transpose(1, 2)),
            NotImplementedError,
            "at least 2",
        ),
        (lambda o: dict(cu_seqlens=torch.zeros(2, dtype=torch.int32)), ValueError, "cu_seqlens"),
        (lambda o: dict(cu_seqlens=torch.zeros(3, dtype=torch.int64)), TypeError, "cu_seqlens"),
        (lambda o: dict(num_accepted_tokens=torch.ones(1, dtype=torch.int32)), ValueError, "num_accepted_tokens"),
        (lambda o: dict(norm_weight=torch.ones(64, dtype=torch.bfloat16)), NotImplementedError, "norm_weight"),
        (lambda o: dict(norm_weight=torch.ones(136, dtype=torch.bfloat16)[1:129]), ValueError, "norm_weight"),  # right shape, a 2-byte-offset slice
        (lambda o: dict(out_scale=torch.zeros(0)), ValueError, "out_scale"),
        (
            lambda o: dict(conv_state=torch.zeros(11, 10240, 6, dtype=torch.bfloat16).transpose(1, 2).contiguous().transpose(1, 2)[:, :, :5]),
            NotImplementedError,
            "history columns",
        ),
        (lambda o: dict(conv_state=torch.zeros(11, 10240, 6, dtype=torch.bfloat16)), NotImplementedError, "channel-contiguous"),  # dim-first pool
        (lambda o: dict(conv_state=torch.zeros(11, 6, 10244, dtype=torch.bfloat16)[:, :, :10240].transpose(1, 2)), NotImplementedError, "channel-contiguous"),
        (lambda o: dict(conv_state=o["conv_state"].to(torch.float16)), TypeError, "conv_state"),
        (lambda o: dict(conv_weight=torch.zeros(10240, 3, dtype=torch.bfloat16)), NotImplementedError, "conv_weight"),
        (lambda o: dict(out=torch.zeros(8, 48, 128, dtype=torch.bfloat16)), TypeError, "out"),
        (lambda o: dict(out=torch.zeros(4, 48, 128, dtype=torch.float8_e4m3fn)), ValueError, "out"),
        (lambda o: dict(out_bf16=torch.zeros(8, 48, 128, dtype=torch.float32)), TypeError, "out_bf16"),
    ],
)
def test_typed_declines_name_the_operand(mutation, exc, token):
    ops = make_operands(requests=2, tokens=4)
    ops.update(mutation(ops))
    with pytest.raises(exc, match=token):
        call_check(ops)


def test_declines_a_tokens_count_the_form_cannot_hold():
    ops = make_operands(requests=1, tokens=9)
    with pytest.raises(NotImplementedError, match="at most 8"):
        call_check(ops)
    call_check(ops, cfg=kcfg.make_cfg(NBUF=2))  # a shallower ring holds it


def test_rows_bounds_a_ragged_or_padded_step_instead_of_requests_times_tokens():
    """A serving engine's ragged step (its warm-up pass: two speculative requests of ONE token each, so the operands carry 2 rows
    for requests x T = 8) or a padded one (rows beyond the requests) is bounded by the caller's ``rows``; without it the FlashInfer
    contract (every request exactly T tokens) is checked as before."""
    ops = make_operands(requests=2, tokens=4)
    ragged = dict(ops, qkv=ops["qkv"][:2], z=ops["z"][:2], hidden=ops["hidden"][:2], cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32))
    with pytest.raises(NotImplementedError, match="rows >= requests x T = 8"):
        call_check(ragged)
    entry.check_support(**ragged, device_cc=(10, 0), rows=2)
    out2 = torch.empty(2, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.float8_e4m3fn)
    entry.check_support(**ragged, out=out2, device_cc=(10, 0), rows=2)
    with pytest.raises(ValueError, match="rows >= the step's rows = 2"):
        entry.check_support(**ragged, out=out2[:1], device_cc=(10, 0), rows=2)
    with pytest.raises(NotImplementedError, match="the step's rows = 3"):
        entry.check_support(**ragged, device_cc=(10, 0), rows=3)  # the operands carry 2 rows
    with pytest.raises(ValueError, match="positive int"):
        entry.check_support(**ragged, device_cc=(10, 0), rows=0)
    # padded: 10 rows of operands for requests x T = 8, an output pair of 10 rows
    pad = make_operands(requests=2, tokens=4, slots=20)
    qkvz = torch.randn(10, 16384).bfloat16()
    padded = dict(
        pad,
        qkv=qkvz[:, : kcfg.QKV_CHANNELS],
        z=qkvz[:, kcfg.QKV_CHANNELS :].view(10, kcfg.V_HEADS, kcfg.HEAD_DIM),
        hidden=torch.zeros(10, kcfg.HIDDEN).bfloat16(),
    )
    out10 = torch.empty(10, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="rows >= requests x T = 8"):
        entry.check_support(**padded, out=out10[:7], device_cc=(10, 0))
    entry.check_support(**padded, out=out10, out_bf16=torch.empty(10, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.bfloat16), device_cc=(10, 0), rows=10)


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
    launch, before the compiled kernel sees the pointers: with a kernel planted under the binding's compile key (the step shape, the
    strides, the parameter dtypes, the config -- none of these mutations changes it), a mismatched operand is a typed decline, never a
    launch; the caller's output slots are checked too."""
    k = entry.GdnVerifyFused("cpu", 1e-6)
    ops = make_operands(requests=2, tokens=4)
    key = _key(k, ops)

    def launched(*args):
        raise AssertionError("the planted kernel was launched on a mismatched binding")

    k.compiled[key] = launched
    for mutation, exc, token in (
        (dict(hidden=ops["hidden"].to(torch.float16)), TypeError, "hidden"),  # a dtype the key does not carry
        (dict(cu_seqlens=ops["cu_seqlens"][:2]), ValueError, "cu_seqlens"),  # fewer than requests + 1 offsets
        (dict(conv_weight=torch.zeros(10240, 3, dtype=torch.bfloat16)), NotImplementedError, "conv_weight"),
        (dict(norm_weight=torch.ones(136, dtype=torch.bfloat16)[1:129]), ValueError, "norm_weight"),  # the right shape at a 2-byte offset
    ):
        bad = dict(ops, **mutation)
        assert _key(k, bad) == key, "the mutation keeps the compile key: only the launch check sees it"
        with pytest.raises(exc, match=token):
            _run(k, bad)
    with pytest.raises(TypeError, match="out"):
        _run(k, ops, out=torch.zeros(8, 48, 128, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="out_bf16"):
        _run(k, ops, out_bf16=torch.zeros(4, 48, 128, dtype=torch.bfloat16))  # fewer rows than the step
    with pytest.raises(ValueError, match="rows"):
        k(**ops, rows=0)


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


def _key(k, o):
    return k.key(o["qkv"], o["z"], o["A_log"], o["dt_bias"], o["ssm_state"], o["state_indices"], o["norm_weight"], o["conv_state"])


def test_compile_key_is_plan_time_only():
    k = entry.GdnVerifyFused("cpu", 1e-6)
    a = make_operands(seed=1)
    b = make_operands(seed=2)
    assert _key(k, a) == _key(k, b)  # values never enter
    c = make_operands(slots=20)
    assert _key(k, a) == _key(k, c)  # nor the slot count
    assert _key(k, a) != _key(k, make_operands(requests=4))  # the step shape does
    assert _key(k, a) != _key(k, make_operands(tokens=6))
    wide = torch.zeros(8, 16384 + 1024, dtype=torch.bfloat16)
    assert _key(k, dict(a, qkv=wide[:, :10240])) != _key(k, a)  # a stride does
    assert _key(k, dict(a, A_log=a["A_log"].bfloat16())) != _key(k, a)  # a parameter dtype does
    k2 = entry.GdnVerifyFused("cpu", 1e-6, cfg=kcfg.make_cfg(TPR=8, NBUF=1))
    assert _key(k2, a) != _key(k, a)
    assert k.debug is False and replace(k.cfg, DEBUG=True) == entry.GdnVerifyFused("cpu", 1e-6, debug=True).cfg
    assert k.cfg.SMEM_PER_SM_BYTES == kcfg.SM100_SMEM_PER_SM_BYTES  # a host entry keeps the module reading; a CUDA device replaces it
    assert entry.step_shape(a["state_indices"]) == (2, 4)


def test_torch_reference_conditioning_and_padding_bookkeeping():
    ops = make_operands(requests=4, tokens=4, seed=3, pad="last", skip="mid")
    moved = condition_conv_inputs(ops)
    assert moved >= 0 and condition_conv_inputs(ops) == 0
    wins = _windows(ops)
    assert len(wins) == 3 and all(n != 3 for n, *_ in wins), "the padding request has no window"
    idx = ops["state_indices"]
    for n in range(3):
        assert int(idx[n, 0]) > 0 and int(idx[n, int(ops["num_accepted_tokens"][n]) - 1]) > 0
        assert int((idx[n] == 0).sum()) == 1, "one skipped slot per live request"
    out, q, ssm, conv, live = reference(ops)
    assert live[:12].all() and not live[12:].any()
    assert torch.equal(q.view(torch.uint8), quant_fp8(out, ops["out_scale"]).view(torch.uint8))


# ---------------------------------------------------------------------------------------------------------------------------
# device cells
# ---------------------------------------------------------------------------------------------------------------------------
def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    cc = tuple(torch.cuda.get_device_capability())
    if not entry.cc_supported(cc):
        pytest.skip(f"the fused GDN verify step serves SM100-class parts (cc 10.x), this device is {cc}")
    msg = entry._dsl_version_error()
    if msg:
        pytest.skip(msg)
    return torch.device("cuda")


def _run(k, ops, out=None, out_bf16=None):
    return k(**ops, out=out, out_bf16=out_bf16)


def _snapshot(k, ops):
    """Run once; the bytes of every output (the fp8 / bf16 outputs, the state and conv pools, the tickets back at zero)."""
    rows = ops["qkv"].shape[0]
    outb = torch.empty(rows, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.bfloat16, device=ops["qkv"].device)
    q = _run(k, ops, out_bf16=outb)
    torch.cuda.synchronize()
    requests, tokens = ops["state_indices"].shape
    assert not k.outputs(requests, tokens)["counts"].any(), "the reader tickets return to zero"
    # the conv pool in its STORAGE layout [slots, T + 2, 10240] (the kernel sees the channel-contiguous [slots, 10240, T + 2] view of it)
    return dict(
        qout=q.view(torch.uint8).clone(),
        out=outb.view(torch.int16).clone(),
        ssm=ops["ssm_state"].clone(),
        conv=ops["conv_state"].transpose(1, 2).contiguous().clone(),
    )


def _assert_bitwise(a, b, what):
    for key in ("qout", "out", "ssm", "conv"):
        assert torch.equal(a[key].contiguous().view(torch.uint8), b[key].contiguous().view(torch.uint8)), f"{what}: {key} differs"


def test_entry_reads_the_device_and_picks_the_form_per_requests():
    dev = _device_or_skip()
    k = entry.GdnVerifyFused(dev, 1e-6)
    props = torch.cuda.get_device_properties(dev)
    assert k.cfg.SMEM_PER_SM_BYTES == props.shared_memory_per_multiprocessor and k.cfg.SMEM_OPTIN_BYTES == props.shared_memory_per_block_optin
    assert k.sm_count == props.multi_processor_count and not k.cfg_given
    for requests in (1, 2, 3, 4, 8, 16):
        want, got = kcfg.default_cfg(requests, k.sm_count), k.cfg_for(requests)
        assert replace(got, NORM_FORM=None, SMEM_PER_SM_BYTES=want.SMEM_PER_SM_BYTES, SMEM_OPTIN_BYTES=want.SMEM_OPTIN_BYTES) == want, requests
        assert got.NORM_FORM == kcfg.norm_form_for(requests) and got.PDL is False
    one = k.cfg_for(1)
    assert one.TWO_CTAS_PER_HEAD == (kcfg.V_HEADS * 2 <= k.sm_count)  # the pair while its 96 cluster CTAs have an SM each
    assert 2 * (one.smem_bytes(6) + kcfg.SMEM_CTA_RESERVE_BYTES) > one.SMEM_PER_SM_BYTES  # one CTA per SM by shared memory (pair or not)
    two = k.cfg_for(k.sm_count // kcfg.V_HEADS + 1)  # the first requests count whose grid outgrows the SMs
    assert two.MIN_CTAS_PER_SM == 2 and 2 * (two.smem_bytes(6) + kcfg.SMEM_CTA_RESERVE_BYTES) <= two.SMEM_PER_SM_BYTES


REFERENCE_CELLS = [
    (1, 6, "random", "none", "none"),
    (2, 6, "random", "last", "none"),
    (4, 6, "random", "last", "mid"),
    (4, 6, "full", "mid", "last"),
    (1, 2, "full", "none", "none"),
    (2, 2, "one", "last", "first"),
    (4, 4, "random", "last", "mid"),
    (1, 4, "one", "none", "last"),
    (2, 8, "random", "none", "mid"),
]


@pytest.mark.parametrize("requests, tokens, accepted, pad, skip", REFERENCE_CELLS, ids=lambda v: str(v))
def test_matches_the_torch_reference(requests, tokens, accepted, pad, skip):
    """The fused chain against the FP64 torch reference at the reference kernel's bars; random acceptance counts, a padding request,
    one skipped slot write per live request, non-monotone slots."""
    dev = _device_or_skip()
    ops = make_operands(requests, tokens, dev, seed=100 * requests + 10 * tokens + len(pad), accepted=accepted, pad=pad, skip=skip)
    condition_conv_inputs(ops)
    ssm0, conv0 = pools_before(ops)
    k = entry.GdnVerifyFused(dev, 1e-6)
    outb = torch.empty(requests * tokens, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.bfloat16, device=dev)
    q = _run(k, ops, out_bf16=outb)
    torch.cuda.synchronize()
    assert not k.outputs(requests, tokens)["counts"].any()
    check_against_reference(ops, q, outb, ssm0, conv0)


@pytest.mark.parametrize("param_dtypes", [(torch.bfloat16, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32)])
def test_parameter_dtypes_are_keys_and_match_the_reference(param_dtypes):
    dev = _device_or_skip()
    ops = make_operands(2, 4, dev, seed=77, param_dtypes=param_dtypes, pad="last")
    condition_conv_inputs(ops)
    ssm0, conv0 = pools_before(ops)
    k = entry.GdnVerifyFused(dev, 1e-6)
    outb = torch.empty(8, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.bfloat16, device=dev)
    q = _run(k, ops, out_bf16=outb)
    torch.cuda.synchronize()
    check_against_reference(ops, q, outb, ssm0, conv0)


@pytest.mark.parametrize("requests", [8, 16])
def test_two_cta_form_beyond_the_reference_kernel_envelope_matches_the_torch_reference(requests):
    """8 and 16 requests (the reference verify kernel declines them): the two-CTAs-per-SM form against the torch reference, the
    same bars; the gated-norm association is the reference's one-CTA-per-head chain order (ctas1)."""
    dev = _device_or_skip()
    ops = make_operands(requests, 6, dev, seed=800 + requests, pad="last", skip="mid")
    condition_conv_inputs(ops)
    ssm0, conv0 = pools_before(ops)
    k = entry.GdnVerifyFused(dev, 1e-6)
    assert k.cfg_for(requests).MIN_CTAS_PER_SM == 2 or requests * kcfg.V_HEADS <= k.sm_count
    outb = torch.empty(requests * 6, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.bfloat16, device=dev)
    q = _run(k, ops, out_bf16=outb)
    torch.cuda.synchronize()
    check_against_reference(ops, q, outb, ssm0, conv0)


@pytest.mark.parametrize("form", ["pick", "pair"])
def test_padding_and_skipped_slots_touch_nothing_else(form):
    """The entry's pick at 4 requests (the two-CTAs-per-SM form on a 148- / 152-SM part) and the two-CTAs-per-head pair forced: a
    padding request (both CTAs of a pair null together: the cluster wait pairs the entry arrive, nothing shipped) zeroes its rows and
    touches nothing else; a skipped slot stays."""
    dev = _device_or_skip()
    ops = make_operands(4, 6, dev, seed=5, accepted="random", pad="mid", skip="last")
    idx = ops["state_indices"]
    ssm0, conv0 = ops["ssm_state"].clone(), ops["conv_state"].contiguous().clone()
    k = entry.GdnVerifyFused(dev, 1e-6, cfg=None if form == "pick" else kcfg.PAIR_CFG)
    bufs = k.outputs(4, 6)
    bufs["qout"].view(torch.uint8).fill_(0xAB)
    outb = torch.full((24, kcfg.V_HEADS, kcfg.HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=dev)
    q = _run(k, ops, out_bf16=outb)
    torch.cuda.synchronize()
    pad_rows = slice(2 * 6, 3 * 6)
    assert not q.view(torch.uint8)[pad_rows].any() and not outb[pad_rows].view(torch.int16).any(), "the padding request's rows are zero"
    assert q.view(torch.uint8)[:12].any() and torch.isfinite(outb[:12].float()).all(), "the live rows are written"
    touched = set()
    for n in range(4):
        if int(idx[n, 0]) > 0:
            touched.update(int(s) for s in idx[n].tolist() if s > 0)
    for s in range(ops["ssm_state"].shape[0]):
        if s in touched:
            assert not torch.equal(ops["ssm_state"][s], ssm0[s]), f"slot {s} advances"
        else:
            assert torch.equal(ops["ssm_state"][s], ssm0[s]), f"slot {s} (skipped, padding or unreferenced) is untouched"
    conv = ops["conv_state"].contiguous()
    for s in range(conv.shape[0]):
        is_conv_slot = any(int(idx[n, 0]) == s for n in range(4) if int(idx[n, 0]) > 0)
        assert torch.equal(conv[s], conv0[s]) != is_conv_slot, s
    assert not bufs["counts"].any()


def test_deterministic_and_graph_replay_over_distinct_layers_matches_eager():
    dev = _device_or_skip()
    k = entry.GdnVerifyFused(dev, 1e-6)
    a = make_operands(2, 4, dev, seed=11, pad="last")
    b = make_operands(2, 4, dev, seed=11, pad="last")
    sa, sb = _snapshot(k, a), _snapshot(k, b)
    _assert_bitwise(sa, sb, "two identical operand sets")
    layers = [make_operands(1, 6, dev, seed=200 + i) for i in range(6)]
    eager = [make_operands(1, 6, dev, seed=200 + i) for i in range(6)]
    outs_g = [torch.empty(6, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev) for _ in layers]
    outs_e = [torch.empty(6, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev) for _ in layers]
    warm = make_operands(1, 6, dev, seed=999)
    _run(k, warm)  # compile + warm up this key
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for lay, o in zip(layers, outs_g):
            _run(k, lay, out=o)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    for lay, o in zip(eager, outs_e):
        _run(k, lay, out=o)
    torch.cuda.synchronize()
    for i, (lg, le) in enumerate(zip(layers, eager)):
        assert torch.equal(outs_g[i].view(torch.uint8), outs_e[i].view(torch.uint8)), f"layer {i}: the fp8 output replays bitwise"
        assert torch.equal(lg["ssm_state"], le["ssm_state"]) and torch.equal(lg["conv_state"].contiguous(), le["conv_state"].contiguous()), f"layer {i}"


# --- every knob arm is bitwise with the entry's default pick on a 4-request set with a padding request, a skipped slot and non-monotone slots ---
def _default_snapshot(dev, seed=31):
    ops = make_operands(4, 6, dev, seed=seed, accepted="random", pad="last", skip="mid")
    return _snapshot(entry.GdnVerifyFused(dev, 1e-6), ops)


@pytest.mark.parametrize(
    "arm",
    [
        dict(),  # the one-CTA form with the UNROLLED token loop at 4 requests (two waves on a part with fewer than 192 SMs)
        dict(ROLLED_TOKENS=True, CTAS_PER_SM=1),  # = CFG (the pick at 2-3 requests) at 4 requests
        dict(NBUF=2),
        dict(NBUF=1),
        dict(K_RESIDENT=False),
        dict(BA_PREFETCH_CHUNKS=1),
        dict(BA_PREFETCH_CHUNKS=2),
        dict(TPR=2, NBUF=2),
        dict(TPR=8, NBUF=1),
        dict(TPR=16, NBUF=1),
        dict(TPR=8, NBUF=1, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0, BA_PREFETCH_CHUNKS=1),  # register-resident under the 128 cap: the gate decides, the bytes agree
        dict(TPR=4, NBUF=1, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0, BA_PREFETCH_CHUNKS=1),  # the two-CTA form, explicit
        dict(TPR=4, NBUF=2, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0, BA_PREFETCH_CHUNKS=1),
        dict(TPR=4, NBUF=1, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=2),
        dict(TPR=8, NBUF=2, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0),
        dict(STATE_STREAM=True, NBUF=2),
        dict(PDL=True),
        dict(PDL=True, PDL_RELEASE=True),
        dict(CTAS_PER_SM=0),
        dict(BULK_ISSUE="spread"),
        dict(BULK_ISSUE="spread", NBUF=1),
        dict(TPR=4, NBUF=1, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0, BA_PREFETCH_CHUNKS=1, BULK_ISSUE="spread"),
        dict(STATE_STORE_PATH="stg", NBUF=1),  # the per-thread store path: no ring, no fence, no bulk store
        dict(STATE_STORE_PATH="stg", NBUF=1, CTAS_PER_SM=0),
        dict(STATE_STORE_PATH="stg", NBUF=1, TPR=8),
        dict(STATE_STORE_PATH="stg", NBUF=1, PDL=True),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0),  # = PAIR_CFG: a cluster of two CTAs per head, half the rows each, the norm chains exchanged by st.async
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, NBUF=2),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, NBUF=1),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, TPR=8),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, TPR=16, NBUF=2),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, K_RESIDENT=False),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, BA_PREFETCH_CHUNKS=1),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, PDL=True, PDL_RELEASE=True),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, BULK_ISSUE="spread"),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, STATE_STREAM=True, NBUF=1),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, STATE_STORE_PATH="stg", NBUF=1),
        dict(ROLLED_TOKENS=True),  # the token body traced once (the one-CTA form)
        dict(ROLLED_TOKENS=True, NBUF=1),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True),  # the pair with the rolled loop: the per-token exchange from a dynamic token index
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True, TPR=8),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True, PDL=True, PDL_RELEASE=True),
        dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True, BULK_ISSUE="spread"),
    ],
    ids=lambda a: ",".join(f"{k}={v}" for k, v in a.items()) or "one-cta-form",
)
def test_knob_arms_are_bitwise_with_the_default_pick(arm):
    dev = _device_or_skip()
    base = _default_snapshot(dev)
    ops = make_operands(4, 6, dev, seed=31, accepted="random", pad="last", skip="mid")
    got = _snapshot(entry.GdnVerifyFused(dev, 1e-6, cfg=kcfg.make_cfg(**arm)), ops)
    _assert_bitwise(base, got, str(arm))
    assert not got["qout"][18:24].any() and got["qout"][:18].any(), "the padding request is zero, the live rows written"


@pytest.mark.parametrize("requests", [1, 4])
def test_pdl_launch_knob_entry_is_bitwise_with_the_plain_pick(requests):
    """The entry's ``pdl=True`` (the serving default of the fused verify step behind the specialized decode) on the pick of ``requests``
    -- the rolled cluster pair at 1 (while 96 x requests <= the SM count), the two-CTA form at 4 -- writes the same bytes as the plain pick,
    on a set with a padding request, a skipped slot and non-monotone slots at 4.  The stream is drained before the PDL launch: the kernel's
    caller obligation (no in-flight predecessor writes the state, the conv pool or the slot table) holds in the test as in the chain."""
    dev = _device_or_skip()
    kw = dict(accepted="random", pad="last", skip="mid") if requests == 4 else dict(accepted="random")
    ops = make_operands(requests, 6, dev, seed=37, **kw)
    torch.cuda.synchronize()
    base = _snapshot(entry.GdnVerifyFused(dev, 1e-6), ops)
    ops = make_operands(requests, 6, dev, seed=37, **kw)
    k = entry.GdnVerifyFused(dev, 1e-6, pdl=True)
    assert k.cfg_for(requests).PDL and k.cfg_for(requests).PDL_RELEASE
    torch.cuda.synchronize()
    got = _snapshot(k, ops)
    _assert_bitwise(base, got, f"pdl=True entry at {requests} requests")
    assert got["qout"][:6].any(), "the first request's rows are written"


@pytest.mark.parametrize("tokens", [6, 2])
def test_pair_and_one_cta_forms_are_bitwise_at_one_request(tokens):
    """One request = the pair's own regime (the pick on a 148- / 152-SM part; the ctas2 norm association on both forms): the pair (rolled =
    PAIR_CFG, and unrolled), the one-CTA form and the entry's pick agree byte for byte on the fp8 / bf16 outputs, every state slot and the conv pool."""
    dev = _device_or_skip()
    ops = [make_operands(1, tokens, dev, seed=41, accepted="random", skip="last") for _ in range(3)]
    pick = entry.GdnVerifyFused(dev, 1e-6)
    assert pick.cfg_for(1).TWO_CTAS_PER_HEAD == (2 * kcfg.V_HEADS <= pick.sm_count)
    got_pick = _snapshot(pick, ops[0])
    got_pair = _snapshot(entry.GdnVerifyFused(dev, 1e-6, cfg=kcfg.PAIR_CFG), ops[1])
    got_one = _snapshot(entry.GdnVerifyFused(dev, 1e-6, cfg=kcfg.CFG), ops[2])
    _assert_bitwise(got_one, got_pair, "the pair vs the one-CTA form at one request")
    _assert_bitwise(got_one, got_pick, "the pick vs the one-CTA form at one request")
    ops_u = make_operands(1, tokens, dev, seed=41, accepted="random", skip="last")
    got_unrolled = _snapshot(entry.GdnVerifyFused(dev, 1e-6, cfg=replace(kcfg.PAIR_CFG, ROLLED_TOKENS=False)), ops_u)
    _assert_bitwise(got_one, got_unrolled, "the unrolled pair vs the one-CTA form at one request")


def test_debug_dumps_do_not_change_the_outputs():
    dev = _device_or_skip()
    base = _default_snapshot(dev)
    ops = make_operands(4, 6, dev, seed=31, accepted="random", pad="last", skip="mid")
    k = entry.GdnVerifyFused(dev, 1e-6, debug=True)
    got = _snapshot(k, ops)
    _assert_bitwise(base, got, "DEBUG")
    b = k.outputs(4, 6)
    assert b["raw"][:18].view(torch.int16).any() and b["convout"][:18].view(torch.int16).any() and b["baout"][:18].view(torch.int16).any()


# --- the reference verify kernel's saved outputs (produced by its own interpreter on saved operand sets), byte for byte ---
def _reference_sets():
    d = os.environ.get(REFERENCE_DIR_ENV)
    if not d or not os.path.isdir(d):
        return []
    tags = sorted(f[len("set_") : -3] for f in os.listdir(d) if f.startswith("set_") and f.endswith(".pt"))
    return [t for t in tags if os.path.exists(os.path.join(d, f"ref_{t}.pt"))]


def test_cfg_knobs_reject_a_ring_the_opt_in_cannot_hold():
    with pytest.raises(ValueError, match="does not fit"):
        kcfg.make_cfg(NBUF=4)  # four 64 KB slabs: 262 KB of ring at any TPR (a buffer is a whole slab)
    assert kcfg.make_cfg(NBUF=3).natural_smem_bytes(2) <= kcfg.SM100_SMEM_OPTIN_BYTES


def load_reference_set(d, tag, device):
    """A saved operand set (CPU tensors in the serving layout) on ``device`` plus the reference kernel's saved outputs."""
    s = torch.load(os.path.join(d, f"set_{tag}.pt"), map_location="cpu")
    ref = torch.load(os.path.join(d, f"ref_{tag}.pt"), map_location="cpu")
    if "outputs" not in ref:
        pytest.skip(f"the reference kernel declined this set ({ref.get('declined')}): the torch reference is its oracle")
    i = s["inputs"]
    rows = i["qkvz"].shape[0]
    ops = dict(
        qkv=i["qkvz"].to(device)[:, : kcfg.QKV_CHANNELS],
        z=i["qkvz"].to(device)[:, kcfg.QKV_CHANNELS :].view(rows, kcfg.V_HEADS, kcfg.HEAD_DIM),
        hidden=i["hidden"].to(device),
        ba_weight=i["ba_weight"].to(device),
        A_log=i["A_log"].to(device),
        dt_bias=i["dt_bias"].to(device),
        ssm_state=i["ssm_state"].to(device),
        state_indices=i["state_indices"].to(device),
        cu_seqlens=i["cu_seqlens"].to(device),
        num_accepted_tokens=i["num_accepted_tokens"].to(device),
        norm_weight=i["norm_weight"].to(device),
        out_scale=i["out_scale"].to(device),
        conv_state=i["conv_state"].to(device).transpose(1, 2),
        conv_weight=i["conv_weight"].to(device),
    )
    return ops, ref["outputs"], s.get("meta", {})


@pytest.mark.parametrize(
    "tag", _reference_sets() or [pytest.param("none", marks=pytest.mark.skip(reason=f"{REFERENCE_DIR_ENV} names no saved reference-kernel output set"))]
)
def test_bitwise_with_the_reference_verify_kernel_outputs(tag):
    dev = _device_or_skip()
    ops, ref, meta = load_reference_set(os.environ[REFERENCE_DIR_ENV], tag, dev)
    k = entry.GdnVerifyFused(dev, float(meta.get("eps", 1e-6)))
    got = _snapshot(k, ops)
    for key, mine in (("qout", got["qout"]), ("out", got["out"]), ("ssm_state", got["ssm"]), ("conv_state", got["conv"])):
        theirs = ref[key].to(dev)
        assert torch.equal(
            mine.contiguous().view(torch.uint8), theirs.contiguous().view(torch.uint8)
        ), f"{tag}: {key} differs from the reference kernel's bytes"
    if "counts" in ref:
        assert not ref["counts"].any()
