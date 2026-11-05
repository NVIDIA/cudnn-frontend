# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused GDN verify step (``cudnn.linear_attention.frost.gdn_verify_fused``): the typed declines of its host entry (host-only,
CPU tensors), the plan-time-only compile key, the config knob validation and the per-T shared-memory budget (the occupancy pad and the
co-residency on every part reading), the DSL-floor decline, and -- on a cc 10.x or 12.x device -- the semantics against a pure-torch reference of the fused
chain (the FP64 recurrence with the chain's rounding points; its tolerances are the ones the reference verify kernel was validated
with: the bf16 output within 2^-6 relative + 1e-3, fp8 bytes equal on > 99.9 % of the live outputs and bitwise equal to the static
quantization of the kernel's own bf16 output, the conv pool bitwise, written state slots within 1e-4, untouched slots untouched,
padding outputs zero) over requests 1 / 2 / 4 x tokens 2 / 4 / 6 with random acceptance counts, padding requests, skipped slot
writes and non-monotone slot ids, the pick past the one-CTA grid at 8 and 16 requests, determinism, CUDA-graph replay over distinct
layers, the bitwise agreement of every knob arm the part's shared memory holds with the entry's default pick (an arm the part does not
hold is its typed decline, reported as a skip), and the two-CTAs-per-head pair form the part holds (the pick at one request) against
its one-CTA form at one request.  When ``CUDNN_FROST_GDN_VERIFY_REFERENCE_DIR``
names a directory of saved operand sets with the reference verify kernel's outputs (``set_<tag>.pt`` / ``ref_<tag>.pt``), every set
is replayed and compared BYTE for byte (fp8 output, bf16 output, every fp32 state slot, the conv pool, the reader tickets).  The geometry
(``Geometry(K_HEADS, V_HEADS, HIDDEN)`` of ``kernel/gdn_geometry.py``) is pinned on the host too: the default geometry's constants and
forms unchanged, and for the per-rank geometries of the larger dense models the derived constants, the pick ladder and token bounds per
regime, the entry's forms and keys, the operand shapes the entry accepts -- and the BA partition's coverage of every hidden index through
the kernel's ONE half-chunk predicate, with the stale "last chunk is the half chunk" form shown to drop 1024 inputs at hidden 8192 -- and, on
the device, the torch reference and determinism per step shape, every form family bitwise with the one-CTA form, and batch invariance at
each of those geometries."""

import importlib.util
import os
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.linear_attention.frost import gdn_verify_fused as entry  # noqa: E402
from cudnn.linear_attention.frost.kernel import gdn_geometry  # noqa: E402
from cudnn.linear_attention.frost.kernel import gdn_verify_fused_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

# the per-rank geometries of the larger dense models: (K heads, V heads, hidden) at 16- / 8- / 4-way tensor parallelism of a hidden-8192
# model and at 4-way of a hidden-4096 one
RANK_GEOMETRIES = [
    gdn_geometry.Geometry(1, 8, 8192),
    gdn_geometry.Geometry(2, 16, 8192),
    gdn_geometry.Geometry(4, 32, 8192),
    gdn_geometry.Geometry(4, 16, 4096),
]


def _gid(g):
    return f"k{g.K_HEADS}v{g.V_HEADS}h{g.HIDDEN}"


# the cc 10.0 / 10.3 part reading (228 KB per SM, 227 KB opt-in), a hypothetical larger part (336 KiB per SM, 335 KiB opt-in) and the cc
# 12.0 part reading (100 KB per SM, 99 KB opt-in): the pad must do what it claims on each, and a form two of which cannot share the SM is
# a typed decline where the caller asks for two
SM100_READING = (kcfg.SM100_SMEM_PER_SM_BYTES, kcfg.SM100_SMEM_OPTIN_BYTES)  # 233,472 / 232,448
SM120_READING = (kcfg.SM120_SMEM_PER_SM_BYTES, kcfg.SM120_SMEM_OPTIN_BYTES)  # 102,400 / 101,376
PART_READINGS = [SM100_READING, (344064, 343040), SM120_READING]
REFERENCE_DIR_ENV = "CUDNN_FROST_GDN_VERIFY_REFERENCE_DIR"


def _on(reading):
    """The ``Cfg`` fields of a part reading ``(shared memory per SM, per-CTA opt-in)``."""
    return dict(SMEM_PER_SM_BYTES=reading[0], SMEM_OPTIN_BYTES=reading[1])


FP8_MAX = 448.0
AMBIGUOUS_REL = 2.0**-16


# ---------------------------------------------------------------------------------------------------------------------------
# operands (the reference verify kernel's test distributions) and the torch reference of the fused chain
# ---------------------------------------------------------------------------------------------------------------------------
def make_operands(requests=2, tokens=4, device="cpu", seed=0, accepted="random", pad="none", skip="none", param_dtypes=None, slots=None, geometry=None):
    """One seeded verify step in the serving layout at ``geometry`` (the default geometry when None): ``qkv`` / ``z`` strided views of
    one [rows, QKV + V x 128] QKVZ slab ([rows, 16384] at the default), the conv pool stored ``[slots, T + 2, QKV]`` and viewed ``[slots,
    QKV, T + 2]`` (channel-contiguous), distinct random (non-monotone) live slots, ``hidden`` in {-1, 0, 1} and ``ba_weight`` in k / 256
    (the BA dot exact before its bf16 rounding); the same draws in the same order as before the geometry was a parameter, so every seeded
    cell keeps its operands.  ``accepted``: random in [1, T], ``full`` (T) or ``one``; ``pad``: ``last`` / ``mid`` makes that request a
    padding one (every slot 0); ``skip``: ``first`` / ``mid`` / ``last`` zeroes one slot (t >= 1, never the source slot) of every live
    request -- a skipped state write."""
    geo = geometry or kcfg.DEFAULT_GEOMETRY
    QKV, NV, HID = geo.QKV_CHANNELS, geo.V_HEADS, geo.HIDDEN
    g = torch.Generator(device="cpu").manual_seed(seed)
    a_dt, dt_dt, w_dt = param_dtypes or (torch.float32, torch.bfloat16, torch.bfloat16)
    rows = requests * tokens
    slots = slots or rows + 3
    L = tokens + kcfg.CONV_WIDTH - 2

    def rn(*shape):
        return torch.randn(*shape, generator=g)

    qkvz = rn(rows, QKV + geo.OUT_ROW).bfloat16().to(device)  # moved as one slab so qkv / z stay strided views of it (the serving layout)
    conv = rn(slots, L, QKV).bfloat16().to(device)
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
        qkv=qkvz[:, :QKV],
        z=qkvz[:, QKV:].view(rows, NV, kcfg.HEAD_DIM),
        hidden=torch.randint(-1, 2, (rows, HID), generator=g).bfloat16().to(device),
        ba_weight=(torch.randint(-8, 9, (2 * NV, HID), generator=g).float() / 256).bfloat16().to(device),
        A_log=(rn(NV) * 0.5).to(a_dt).to(device),
        dt_bias=(rn(NV) * 0.5).to(dt_dt).to(device),
        ssm_state=(rn(slots, NV, kcfg.HEAD_DIM, kcfg.HEAD_DIM) * 0.1).to(device),
        state_indices=idx.to(device),
        cu_seqlens=torch.arange(0, rows + 1, tokens, dtype=torch.int32).to(device),
        num_accepted_tokens=acc.to(device),
        norm_weight=(1 + 0.2 * rn(kcfg.HEAD_DIM)).to(w_dt).to(device),
        out_scale=torch.tensor([0.05]).to(device),
        conv_state=conv.transpose(1, 2),
        conv_weight=(rn(QKV, kcfg.CONV_WIDTH) * 0.3).bfloat16().to(device),
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
    bf16 tap products and SiLU, the BA projection to bf16, the recurrence in FP64, the core output to bf16, the normed value to bf16.
    The geometry is read off the operands (the state's V heads, the QKV slab's K heads), so a per-rank cell reuses it."""
    hv, dv, dk = int(ops["ssm_state"].shape[1]), kcfg.HEAD_DIM, kcfg.HEAD_DIM
    hk = (int(ops["qkv"].shape[1]) - hv * dv) // (2 * dk)
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


def test_default_cfg_on_the_cc100_reading_is_the_measured_pick_at_every_sm_count():
    """The cc 10.0 readings, explicit and as the defaults: the primary form of every regime is admissible there, so the pick is PAIR_CFG /
    CFG / TWO_CTA_CFG by the SM-count rule alone at every requests count -- whatever the step's token count says -- and the served token
    envelope is the published 35 / 8 / 45 (the byte-identity pin of the budget-aware pick with the measured cc 10.0 pick)."""
    per_sm, optin = SM100_READING
    envelope = {kcfg.PAIR_CFG: 35, kcfg.CFG: 8, kcfg.TWO_CTA_CFG: 45}
    for sms in (84, 148, 152, 170, 188):
        for requests in range(1, 17):
            want = kcfg.PAIR_CFG if requests * 96 <= sms else (kcfg.CFG if requests * 48 <= sms else kcfg.TWO_CTA_CFG)
            assert kcfg.default_cfg(requests, sms) == want, (sms, requests)
            for tokens in (None, 2, 6, 9, 17, 18, 40, 64):
                assert kcfg.default_cfg(requests, sms, per_sm, optin, tokens) == want, (sms, requests, tokens)
            assert kcfg.max_tokens_served(requests, sms) == kcfg.max_tokens_served(requests, sms, per_sm, optin) == envelope[want], (sms, requests)
    assert (
        kcfg.default_cfg(2, 192) == kcfg.PAIR_CFG and kcfg.default_cfg(3, 192) == kcfg.CFG and kcfg.default_cfg(5, 192) == kcfg.TWO_CTA_CFG
    )  # a 192-SM part would pair two
    assert kcfg.first_fitting(kcfg.ONE_CTA_FORMS) == kcfg.CFG and kcfg.first_fitting(kcfg.PAIR_FORMS, per_sm, optin) == kcfg.PAIR_CFG
    assert kcfg.first_fitting(kcfg.BEYOND_FORMS, per_sm, optin) == kcfg.TWO_CTA_CFG
    assert kcfg.admissible(kcfg.TWO_CTA_CFG) and kcfg.admissible(kcfg.CFG, per_sm, optin) and kcfg.admissible(kcfg.PAIR_CFG, per_sm, optin)
    assert kcfg.PAIR_FORMS == (kcfg.PAIR_CFG, kcfg.PAIR_NBUF2_CFG, kcfg.PAIR_NBUF1_CFG)
    assert kcfg.ONE_CTA_FORMS == (kcfg.CFG, kcfg.ONE_CTA_NBUF1_CFG, kcfg.PAIR_NBUF1_CFG)
    assert kcfg.BEYOND_FORMS == (kcfg.TWO_CTA_CFG, kcfg.PAIR_NBUF2_CFG, kcfg.ONE_CTA_NBUF1_CFG, kcfg.PAIR_NBUF1_CFG)
    assert kcfg.BEYOND_FORMS == (kcfg.TWO_CTA_CFG,) + kcfg.BEYOND_SMALL_SMEM_ORDER  # the cc 10.0 primary first: byte-identical there
    assert kcfg.part_readings(per_sm, optin) == _on(SM100_READING)


def test_default_cfg_on_the_cc120_reading_walks_the_ladders_token_aware():
    """The cc 12.0 readings (100 KB per SM, 99 KB opt-in) on 84 / 170 / 188 SMs: no regime's primary form is admissible (the 3-deep rings
    do not fit; two TWO_CTA_CFG footprints do not share the SM), so the pick walks the ladder, token-aware -- the pair with the 2-deep
    ring at one request AND past the one-CTA grid (the measured wave-quantization exception, ``BEYOND_SMALL_SMEM_ORDER``), the one-CTA
    landing-slab form while the ``requests x 48`` grid has an SM per CTA, all for T <= 8 (T 8 itself lands on the 2-deep-ring pair, which
    holds it), the landing-slab pair for T 9 .. 17 at every requests count, T 18 the deepest form (so the typed decline names 17); never
    TWO_CTA_CFG / PAIR_CFG / CFG; ``max_tokens_served`` 17 everywhere; a part with no form is the typed decline naming its opt-in."""
    per_sm, optin = SM120_READING
    on = _on(SM120_READING)
    for sms in (84, 170, 188):
        for requests in range(1, 17):
            shallow = kcfg.PAIR_NBUF2_CFG if (requests * 96 <= sms or requests * 48 > sms) else kcfg.ONE_CTA_NBUF1_CFG
            for tokens in (None, 2, 3, 4, 6, 8):
                assert kcfg.default_cfg(requests, sms, per_sm, optin, tokens) == shallow, (sms, requests, tokens)
            for tokens in range(9, 19):
                assert kcfg.default_cfg(requests, sms, per_sm, optin, tokens) == kcfg.PAIR_NBUF1_CFG, (sms, requests, tokens)
            assert kcfg.max_tokens_served(requests, sms, per_sm, optin) == 17, (sms, requests)
            for tokens in (None, 2, 8, 17, 18, 64):
                assert kcfg.default_cfg(requests, sms, per_sm, optin, tokens) not in (kcfg.TWO_CTA_CFG, kcfg.PAIR_CFG, kcfg.CFG), (sms, requests, tokens)
    assert kcfg.default_cfg(1, 170, per_sm, optin) == kcfg.default_cfg(1, 170, per_sm, optin, 2) == kcfg.PAIR_NBUF2_CFG
    assert kcfg.default_cfg(1, 84, per_sm, optin) == kcfg.ONE_CTA_NBUF1_CFG  # 84 SMs hold no 96-CTA cluster grid: the one-CTA ladder
    assert kcfg.default_cfg(2, 84, per_sm, optin, 6) == kcfg.PAIR_NBUF2_CFG  # 96 one-CTA CTAs outgrow 84 SMs: past the grid, the pair again
    # past the one-CTA grid on 170 / 188 SMs: the 2-deep-ring pair for T 2 .. 8 (T 8 fits: 100,688 B), never the spilling cap, and never the
    # one-CTA form's nearly empty second wave (192 CTAs over 170 SMs); the one-CTA form stays the pick while its grid is one wave (R 2 / 3)
    for sms in (170, 188):
        for tokens in (None, 2, 4, 6, 8):
            assert kcfg.default_cfg(4, sms, per_sm, optin, tokens) == kcfg.default_cfg(8, sms, per_sm, optin, tokens) == kcfg.PAIR_NBUF2_CFG, (sms, tokens)
            assert kcfg.default_cfg(2, sms, per_sm, optin, tokens) == kcfg.default_cfg(3, sms, per_sm, optin, tokens) == kcfg.ONE_CTA_NBUF1_CFG, (sms, tokens)
        assert kcfg.default_cfg(4, sms, per_sm, optin, 9) == kcfg.default_cfg(2, sms, per_sm, optin, 9) == kcfg.PAIR_NBUF1_CFG, sms
    assert replace(kcfg.PAIR_NBUF2_CFG, **on).max_tokens() == 8 and replace(kcfg.default_cfg(4, 170, per_sm, optin, 8), **on).max_tokens() == 8
    pick18 = replace(kcfg.default_cfg(1, 170, per_sm, optin, 18), **on)
    kcfg.validate_tokens(pick18, 17)
    with pytest.raises(NotImplementedError, match="at most 17 on this part"):
        kcfg.validate_tokens(pick18, 18)
    with pytest.raises(NotImplementedError, match="at most 8 on this part"):
        kcfg.validate_tokens(replace(kcfg.default_cfg(2, 170, per_sm, optin, 8), **on), 9)  # the T-aware pick holds T 9; the T-8 pick does not
    assert (
        kcfg.first_fitting(kcfg.ONE_CTA_FORMS, per_sm, optin) == kcfg.ONE_CTA_NBUF1_CFG
        and kcfg.first_fitting(kcfg.PAIR_FORMS, per_sm, optin) == kcfg.PAIR_NBUF2_CFG
    )
    assert kcfg.first_fitting(kcfg.BEYOND_FORMS, per_sm, optin) == kcfg.first_fitting(kcfg.BEYOND_SMALL_SMEM_ORDER, per_sm, optin) == kcfg.PAIR_NBUF2_CFG
    assert (
        not kcfg.admissible(kcfg.TWO_CTA_CFG, per_sm, optin)
        and not kcfg.admissible(kcfg.CFG, per_sm, optin)
        and not kcfg.admissible(kcfg.PAIR_CFG, per_sm, optin)
    )
    assert kcfg.admissible(kcfg.PAIR_NBUF1_CFG, per_sm, optin) and not kcfg.admissible(
        replace(kcfg.ONE_CTA_NBUF1_CFG, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0), per_sm, optin
    )
    assert kcfg.admissible(replace(kcfg.PAIR_NBUF1_CFG, MIN_CTAS_PER_SM=2), per_sm, optin)  # the one capped form two of which share a 100 KB SM (T <= 3)
    for call in (
        lambda: kcfg.default_cfg(1, 170, 40960, 40000),
        lambda: kcfg.default_cfg(4, 170, 40960, 40000, 6),
        lambda: kcfg.max_tokens_served(4, 170, 40960, 40000),
        lambda: kcfg.first_fitting(kcfg.ONE_CTA_FORMS, 40960, 40000),
    ):
        with pytest.raises(NotImplementedError, match="40000-byte shared-memory opt-in per CTA .* holds no verify form at 2 tokens per request"):
            call()
    with pytest.raises(ValueError, match="tokens must be a positive int"):
        kcfg.default_cfg(1, 170, per_sm, optin, 0)
    with pytest.raises(ValueError, match="cannot exceed the SM"):
        kcfg.default_cfg(1, 170, 100000, 200000)
    with pytest.raises(ValueError, match="smem_optin"):
        kcfg.max_tokens_served(1, 170, per_sm, 0)


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


def test_cudnn_level_export_resolves_to_the_entry():
    assert cudnn.GdnVerifyFused is entry.GdnVerifyFused, "the lazy cudnn-level export (import cudnn; cudnn.GdnVerifyFused) is the entry class"


# ---------------------------------------------------------------------------------------------------------------------------
# the geometry: the default unchanged, the per-rank geometries derived, admitted and keyed; the BA partition's coverage
# ---------------------------------------------------------------------------------------------------------------------------
def test_the_default_geometry_is_the_validated_one_and_its_constants_are_unchanged():
    """The default ``Geometry`` (16 K / 48 V / hidden 5120, the one every form was validated at) is the one the module constants name and
    every module form carries (the SAME object, so a default-geometry pick is the module form itself: the forms cache and the compile keys
    of existing callers are untouched); the verify BA partition at that hidden width has three chunks, the last a half chunk."""
    g = kcfg.DEFAULT_GEOMETRY
    assert g is gdn_geometry.DEFAULT_GEOMETRY and (g.K_HEADS, g.V_HEADS, g.HIDDEN) == (16, 48, 5120) == (kcfg.K_HEADS, kcfg.V_HEADS, kcfg.HIDDEN)
    assert (kcfg.QKV_CHANNELS, kcfg.K_CHANNEL0, kcfg.V_CHANNEL0, kcfg.OUT_ROW, kcfg.BA_CHUNK, kcfg.BA_CHUNKS) == (10240, 2048, 4096, 6144, 2048, 3)
    assert (kcfg.TICKET_LIMIT, kcfg.TICKET_STRIDE, kcfg.HEAD_DIM, kcfg.CONV_WIDTH) == (2, 512, 128, 4)
    assert (g.BA_CHUNKS_VERIFY, g.BA_VERIFY_HALF_LIVE, g.BA_CHUNKS_DECODE, g.VK_RATIO) == (3, (True, True, False), 5, 3)
    for form in (kcfg.CFG, kcfg.PAIR_CFG, kcfg.TWO_CTA_CFG, kcfg.PAIR_NBUF2_CFG, kcfg.PAIR_NBUF1_CFG, kcfg.ONE_CTA_NBUF1_CFG, kcfg.make_cfg(), kcfg.Cfg()):
        assert form.GEOM is g
    assert (kcfg.CFG.TICKET_LIMIT, kcfg.PAIR_CFG.TICKET_LIMIT) == (2, 5)
    assert kcfg.default_cfg(1, 148) is kcfg.PAIR_CFG and kcfg.default_cfg(2, 148, geometry=g) is kcfg.CFG and kcfg.default_cfg(4, 148) is kcfg.TWO_CTA_CFG
    assert kcfg.with_geometry(kcfg.CFG, g) is kcfg.CFG and kcfg.resolve_cfg(kcfg.CFG, 1).GEOM is g
    assert kcfg.max_tokens_served(1, 148, geometry=g) == kcfg.max_tokens_served(1, 148) == 35


@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_a_rank_geometry_derives_its_forms_ladder_and_token_bounds(g):
    """A form compiled for a per-rank geometry differs from the module form in ``GEOM`` alone (the shared-memory budget per token count,
    the token bound, the lane map and the co-residency carry no head-count or hidden term; the ticket limit follows the V-per-K ratio); the
    pick's regime switches move with the V heads per rank by the one rule, the served token bound with them (35 / 8 / 8 by regime on the
    cc 10.0 readings: past the one-CTA grid a per-rank geometry takes the one-CTA form in two waves, never the two-CTAs-per-SM form, which
    is unmeasured off the default geometry and spills at hidden 8192 -- the sm_100a / sm_103a census: 128 registers / 0 stack / 0 STL-LDL
    at 16 / 48 / 5120 and at 4 / 16 / 4096 against 128 / 56 B / 14 STL + 14 LDL at 1 / 8 / 8192, 2 / 16 / 8192 and 4 / 32 / 8192, a
    hidden-width term), the resolved norm form rides on the geometry form, and the cc 12.0 readings walk the same ladder."""
    V = g.V_HEADS
    cfg, pair, two = (kcfg.with_geometry(f, g) for f in (kcfg.CFG, kcfg.PAIR_CFG, kcfg.TWO_CTA_CFG))
    assert cfg == replace(kcfg.CFG, GEOM=g) and cfg.GEOM is g and cfg != kcfg.CFG and hash(cfg) != hash(kcfg.CFG)
    assert (cfg.TICKET_LIMIT, pair.TICKET_LIMIT, two.TICKET_LIMIT) == (g.VK_RATIO - 1, 2 * g.VK_RATIO - 1, g.VK_RATIO - 1)
    for a, b in ((cfg, kcfg.CFG), (pair, kcfg.PAIR_CFG), (two, kcfg.TWO_CTA_CFG)):
        kcfg.validate_cfg(a)
        assert [a.natural_smem_bytes(t) for t in (2, 6, 8)] == [b.natural_smem_bytes(t) for t in (2, 6, 8)] and a.max_tokens() == b.max_tokens()
        assert (a.R, a.C, a.RG, a.BOX_ROWS, a.STATE_BOX_PITCH_ELEMS, a.co_resident_ctas(2)) == (
            b.R,
            b.C,
            b.RG,
            b.BOX_ROWS,
            b.STATE_BOX_PITCH_ELEMS,
            b.co_resident_ctas(2),
        )
    envelope = {kcfg.PAIR_CFG: 35, kcfg.CFG: 8}
    assert kcfg.BEYOND_FORMS_OFF_DEFAULT[0] is kcfg.CFG and kcfg.TWO_CTA_CFG not in kcfg.BEYOND_FORMS_OFF_DEFAULT and kcfg.BEYOND_FORMS[0] is kcfg.TWO_CTA_CFG
    for sms in (148, 152, 170, 188):
        for requests in range(1, 41):
            want = kcfg.PAIR_CFG if requests * 2 * V <= sms else kcfg.CFG  # the one-CTA form past the grid too (two waves)
            got = kcfg.default_cfg(requests, sms, geometry=g)
            assert got == replace(want, GEOM=g) and got.GEOM is g and got.MIN_CTAS_PER_SM == 1, (sms, requests)
            assert kcfg.default_cfg(requests, sms, *SM100_READING, 6, g) == got
            assert kcfg.max_tokens_served(requests, sms, geometry=g) == envelope[want], (sms, requests)
            if requests * V > sms:
                assert kcfg._ladder(requests, sms, g) is kcfg.BEYOND_FORMS_OFF_DEFAULT
    # the default geometry's ladders are untouched: the two-CTAs-per-SM form past the grid, 45 tokens
    assert kcfg._ladder(4, 152) is kcfg.BEYOND_FORMS and kcfg.default_cfg(4, 152) is kcfg.TWO_CTA_CFG and kcfg.max_tokens_served(4, 152) == 45
    one = kcfg.resolve_cfg(kcfg.default_cfg(1, 152, geometry=g), 1)
    assert (one.NORM_FORM, one.GEOM, one.TWO_CTAS_PER_HEAD) == ("ctas2", g, True)
    per_sm, optin = SM120_READING
    assert kcfg.default_cfg(1, 170, per_sm, optin, 6, g) == replace(kcfg.PAIR_NBUF2_CFG, GEOM=g)
    assert kcfg.default_cfg(1, 170, per_sm, optin, 12, g) == replace(kcfg.PAIR_NBUF1_CFG, GEOM=g)
    assert kcfg.max_tokens_served(1, 170, per_sm, optin, g) == 17
    assert kcfg.make_cfg(TPR=8, NBUF=1, GEOM=g).GEOM is g
    # the BA prefetch depth is the knob bounded by the geometry's chunk count (two chunks at hidden 4096 cannot have three in flight)
    assert cfg.BA_PREFETCH_DEPTH == min(3, g.BA_CHUNKS_VERIFY) and two.BA_PREFETCH_DEPTH == 1 and kcfg.CFG.BA_PREFETCH_DEPTH == kcfg.CFG.BA_PREFETCH_CHUNKS == 3
    assert kcfg.make_cfg(BA_PREFETCH_CHUNKS=2, GEOM=g).BA_PREFETCH_DEPTH == 2 and kcfg.make_cfg(GEOM=g).BA_PREFETCH_DEPTH <= g.BA_CHUNKS_VERIFY
    with pytest.raises(ValueError, match="GEOM"):
        kcfg.make_cfg(GEOM=(g.K_HEADS, g.V_HEADS, g.HIDDEN))
    with pytest.raises(ValueError, match="geometry"):
        kcfg.max_tokens_served(1, 152, geometry="k1v8h8192")


@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_host_checks_accept_the_rank_geometry_operands_and_decline_a_mismatch(g):
    """The entry's binding checks at a per-rank geometry: its operand shapes pass -- named by ``geometry`` or carried by the ``cfg`` --
    the default geometry's checks refuse them by name and vice versa, another hidden width is declined by name, a ``cfg`` and a
    ``geometry`` that disagree are a ``ValueError``, a geometry that is not a ``Geometry`` a ``TypeError``."""
    ops = make_operands(requests=2, tokens=4, geometry=g)
    assert (
        tuple(ops["qkv"].shape) == (8, g.QKV_CHANNELS)
        and tuple(ops["z"].shape) == (8, g.V_HEADS, 128)
        and tuple(ops["ba_weight"].shape) == (2 * g.V_HEADS, g.HIDDEN)
    )
    assert tuple(ops["ssm_state"].shape[1:]) == (g.V_HEADS, 128, 128) and tuple(ops["conv_state"].shape[1:]) == (g.QKV_CHANNELS, 6)
    entry.check_support(**ops, device_cc=(10, 0), geometry=g)
    entry.check_support(**ops, device_cc=(10, 0), cfg=kcfg.resolve_cfg(kcfg.with_geometry(kcfg.CFG, g), 2))  # the cfg's own geometry
    entry.check_binding(**ops, geometry=g)
    with pytest.raises(NotImplementedError, match="qkv as"):
        entry.check_support(**ops, device_cc=(10, 0))  # the default geometry's checks refuse the rank's slab width
    with pytest.raises(NotImplementedError, match="qkv as"):
        entry.check_support(**make_operands(requests=2, tokens=4), device_cc=(10, 0), geometry=g)
    with pytest.raises(NotImplementedError, match="hidden"):
        entry.check_support(**dict(ops, hidden=torch.zeros(8, g.HIDDEN + 1024, dtype=torch.bfloat16)), device_cc=(10, 0), geometry=g)
    with pytest.raises(NotImplementedError, match="ba_weight"):
        entry.check_support(**dict(ops, ba_weight=torch.zeros(2 * g.V_HEADS, 5120, dtype=torch.bfloat16)), device_cc=(10, 0), geometry=g)
    with pytest.raises(ValueError, match="compiled for"):
        entry.check_support(**ops, device_cc=(10, 0), cfg=kcfg.resolve_cfg(kcfg.CFG, 2), geometry=g)
    with pytest.raises(TypeError, match="geometry"):
        entry.check_support(**ops, device_cc=(10, 0), geometry=(g.K_HEADS, g.V_HEADS, g.HIDDEN))
    assert entry.resolve_geometry(None, kcfg.resolve_cfg(kcfg.with_geometry(kcfg.CFG, g), 2)) is g and entry.resolve_geometry(None) is kcfg.DEFAULT_GEOMETRY


@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_entry_takes_the_geometry_and_keys_every_form_by_it(g):
    """An entry built for a per-rank geometry stamps it into every form it picks (the pick ladder and token bounds at the geometry's V
    heads), sizes its buffers by it, carries it in every compile key (a default-geometry entry's key differs), takes it from a given
    ``cfg`` when none is named and refuses a ``cfg`` and a ``geometry`` that disagree."""
    V = g.V_HEADS
    k = entry.GdnVerifyFused("cpu", 1e-6, geometry=g)
    assert k.geometry is g and replace(k.cfg_for(1), NORM_FORM=None) == replace(kcfg.CFG, GEOM=g) and k.cfg_for(1).NORM_FORM == "ctas2"
    k.sm_count = 152
    assert replace(k.cfg_for(1), NORM_FORM=None) == replace(kcfg.PAIR_CFG, GEOM=g) and k.cfg_for(1).GEOM is g
    assert replace(k.cfg_for(152 // (2 * V) + 1), NORM_FORM=None) == replace(kcfg.CFG, GEOM=g)
    beyond = k.cfg_for(152 // V + 1, 6)  # past the one-CTA grid: the one-CTA form in two waves (the capped form: unmeasured off the default, spills at 8192)
    assert replace(beyond, NORM_FORM=None) == replace(kcfg.CFG, GEOM=g) and beyond.NORM_FORM == "ctas1" and beyond.MIN_CTAS_PER_SM == 1
    assert (k.max_tokens_for(1), k.max_tokens_for(152 // (2 * V) + 1), k.max_tokens_for(152 // V + 1)) == (35, 8, 8)
    with pytest.raises(NotImplementedError, match="at most 8 on this part"):
        kcfg.validate_tokens(beyond, 9)  # the two-wave pick's bound past the grid (the default geometry's capped form would hold 45)
    out = k.outputs(2, 4)
    assert tuple(out["qout"].shape) == (8, V, 128) == tuple(out["out"].shape) and out["counts"].shape == (2 * kcfg.TICKET_STRIDE,)
    dbg = entry.GdnVerifyFused("cpu", 1e-6, geometry=g, debug=True).outputs(2, 4)
    assert tuple(dbg["raw"].shape) == (8, V, 128) and tuple(dbg["convout"].shape) == (8, g.QKV_CHANNELS) and tuple(dbg["baout"].shape) == (8, 2 * V)
    ops = make_operands(requests=2, tokens=4, geometry=g)
    default = entry.GdnVerifyFused("cpu", 1e-6)
    assert default.geometry is kcfg.DEFAULT_GEOMETRY
    assert _key(k, ops)[2].GEOM is g and _key(k, ops) != _key(default, make_operands(requests=2, tokens=4))
    given = kcfg.make_cfg(TPR=8, NBUF=1, GEOM=g)
    assert entry.GdnVerifyFused("cpu", 1e-6, cfg=given).geometry is g and entry.GdnVerifyFused("cpu", 1e-6, cfg=given, geometry=g).cfg == given
    with pytest.raises(ValueError, match="compiled for"):
        entry.GdnVerifyFused("cpu", 1e-6, cfg=given, geometry=kcfg.DEFAULT_GEOMETRY)
    with pytest.raises(TypeError, match="geometry"):
        entry.GdnVerifyFused("cpu", 1e-6, geometry="k1v8h8192")


def verify_ba_partition(g, half_live):
    """The hidden indices the verify BA role accumulates, as ``_ba_role`` traces them: virtual thread ``u`` (0 .. 127) sums elements
    ``c * 2048 + 8 u + j`` of every chunk ``c``; virtual thread ``u + 128`` sums ``c * 2048 + 1024 + 8 u + j`` of the chunks with
    ``half_live(c)`` -- the predicate under test (a pure-Python mirror of the kernel's index arithmetic, no GPU)."""
    seen = Counter()
    for c in range(g.BA_CHUNKS_VERIFY):
        for u in range(128):
            kk = c * 2048 + 8 * u
            seen.update(range(kk, kk + 8))
            if half_live(c):
                seen.update(range(kk + 1024, kk + 1024 + 8))
    return seen


@pytest.mark.parametrize("hidden", [1024, 2048, 3072, 4096, 5120, 6144, 8192])
def test_verify_ba_partition_covers_every_hidden_index_once_with_the_kernels_half_chunk_predicate(hidden):
    """With the kernel's one predicate (``Geometry.ba_verify_half_live``: a chunk has its second half unless it is a trailing half chunk)
    the verify BA partition covers ``0 .. HIDDEN - 1`` exactly once at every served hidden width.  The RED half: the predicate the dot
    chains used to carry -- "the LAST chunk is the half chunk", ``c < BA_CHUNKS - 1`` -- holds at hidden 5120 only; at a hidden width of
    whole chunks (8192, 4096, ...) it never accumulates the last chunk's second half: 1024 inputs loaded, dropped, no error."""
    g = gdn_geometry.Geometry(1, 8, hidden)
    assert verify_ba_partition(g, g.ba_verify_half_live) == Counter(range(hidden)), hidden
    assert g.BA_VERIFY_HALF_LIVE == tuple(c * 2048 + 1024 < hidden for c in range(g.BA_CHUNKS_VERIFY))
    assert g.ba_verify_half_live(g.BA_CHUNKS_VERIFY - 1) == (hidden % 2048 == 0), "the last chunk has a second half iff it is a whole chunk"

    def stale(c):
        return c < g.BA_CHUNKS_VERIFY - 1

    covered = verify_ba_partition(g, stale)
    if hidden % 2048 == 1024:
        assert covered == Counter(range(hidden)), "at a trailing half chunk the stale predicate happens to agree"
    else:
        missing = sorted(set(range(hidden)) - set(covered))
        assert missing == list(range(hidden - 1024, hidden)), (hidden, len(missing))


def test_verify_kernel_reads_the_one_half_chunk_predicate_at_every_ba_site():
    """Source tripwire (no DSL import): the verify kernel's BA role spells the second virtual thread's liveness through
    ``ba_verify_half_live`` at every site -- the weight loads, the prefetched and the streamed hidden loads, the weight unpack, the dot
    chains -- and never through the stale ``BA_CHUNKS - 1`` form or a bare module constant of the default geometry."""
    spec = importlib.util.find_spec("cudnn.linear_attention.frost.kernel.gdn_verify_fused_f16")
    src = Path(spec.origin).read_text()
    assert "BA_CHUNKS - 1" not in src and "BA_CHUNKS -1" not in src
    assert src.count("ba_verify_half_live(") >= 5, src.count("ba_verify_half_live(")
    imports = src[src.index("from .gdn_verify_fused_config import (") :]
    imports = imports[: imports.index(")")]  # the kernel's one import block from its config
    for name in ("HIDDEN", "V_HEADS", "QKV_CHANNELS", "K_CHANNEL0", "V_CHANNEL0", "OUT_ROW", "BA_CHUNKS"):
        assert not re.search(rf"\b{name}\b", imports), f"the kernel imports the default geometry's {name}; every geometry fact reaches it through cfg.GEOM"


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
    # a host entry keeps the cc 10.0 readings; its bound follows the pick once an SM count is set (148 here), a given cfg's its own
    assert k.readings == k8.readings == SM100_READING and (k.max_tokens_for(1), k.max_tokens_for(3), k.max_tokens_for(4)) == (35, 8, 45)
    assert k8.max_tokens_for(1) == k8.max_tokens_for(8) == given.max_tokens() and entry.GdnVerifyFused("cpu", 1e-6).max_tokens_for(1) == 8


def test_entry_on_a_99kb_part_starts_from_the_fitting_one_cta_form_and_picks_per_step_shape(monkeypatch):
    """Host-only, the device readers stubbed to the cc 12.0 readings and 170 SMs: the entry constructs (its start form is the one-CTA
    landing-slab form, not CFG, whose 3-deep ring the part does not hold), ``cfg_for`` is token-aware (the 2-deep-ring pair at one request
    for T <= 8, the one-CTA landing-slab form at two and three (the 2-deep-ring pair again from four), the landing-slab pair for T 9 .. 17
    at every requests count),
    ``max_tokens_for`` is 17, the compile key carries the T-aware form, and the step shape is bounded by that form -- while a host caller
    of ``check_support`` without a form keeps the module's one-CTA bound.  On the cc 10.0 readings the start form is CFG and the picks
    are the measured ones, tokens ignored; a part holding no form is the typed decline at construction."""
    monkeypatch.setattr(entry, "device_shared_memory", lambda device: SM120_READING)
    monkeypatch.setattr(entry, "device_multiprocessor_count", lambda device: 170)
    k = entry.GdnVerifyFused("cuda:0", 1e-6)
    assert (k.readings, k.sm_count, k.cfg.SMEM_PER_SM_BYTES, k.cfg.SMEM_OPTIN_BYTES) == (SM120_READING, 170, 102400, 101376)
    assert replace(k.cfg, **_on(SM100_READING)) == kcfg.ONE_CTA_NBUF1_CFG and (k.cfg.NBUF, k.cfg.TWO_CTAS_PER_HEAD, k.cfg.ROLLED_TOKENS) == (1, False, True)

    def strip(c):
        return replace(c, NORM_FORM=None, **_on(SM100_READING))

    assert strip(k.cfg_for(1)) == strip(k.cfg_for(1, 2)) == strip(k.cfg_for(1, 6)) == strip(k.cfg_for(1, 8)) == kcfg.PAIR_NBUF2_CFG
    assert strip(k.cfg_for(1, 9)) == strip(k.cfg_for(1, 17)) == strip(k.cfg_for(4, 12)) == strip(k.cfg_for(16, 17)) == kcfg.PAIR_NBUF1_CFG
    assert strip(k.cfg_for(2)) == strip(k.cfg_for(2, 8)) == strip(k.cfg_for(3, 6)) == kcfg.ONE_CTA_NBUF1_CFG  # the one-CTA grid in one wave
    assert strip(k.cfg_for(4)) == strip(k.cfg_for(4, 6)) == strip(k.cfg_for(4, 8)) == strip(k.cfg_for(16)) == kcfg.PAIR_NBUF2_CFG  # past it: the pair
    assert k.cfg_for(1).NORM_FORM == k.cfg_for(1, 12).NORM_FORM == "ctas2" and k.cfg_for(4, 12).NORM_FORM == "ctas1"
    assert k.cfg_for(1, 9) is k.cfg_for(1, 12), "one resolved form per (module form, requests): the forms cache"
    assert k.cfg_for(1, 18) == k.cfg_for(1, 17), "beyond the ladder the deepest form answers, so the decline names its bound"
    assert all(k.max_tokens_for(requests) == 17 for requests in (1, 2, 3, 4, 8, 16))
    assert (k.cfg_for(1, 9).SMEM_PER_SM_BYTES, k.cfg_for(1, 9).SMEM_OPTIN_BYTES) == SM120_READING, "every pick carries the part's readings"
    ops = make_operands(requests=1, tokens=12)
    assert _key(k, ops)[2] == k.cfg_for(1, 12) and _key(k, ops) != _key(k, make_operands(requests=1, tokens=6))
    # the step is bounded by the T-aware pick: T 12 at one request passes the host checks with it, T 18 is the typed decline naming 17,
    # and a host caller without a form is bounded by the module default form (8) -- said as such, never as a statement about the part
    entry.check_support(**ops, device_cc=(10, 0), cfg=k.cfg_for(1, 12))
    with pytest.raises(NotImplementedError, match="at most 17 on this part"):
        entry.check_support(**make_operands(requests=1, tokens=18), device_cc=(10, 0), cfg=k.cfg_for(1, 18))
    with pytest.raises(
        NotImplementedError,
        match=r"without a form the step is bounded by the module default form \(TPR 4, NBUF 3\) on the cc 10\.0 readings, which holds at most 8 tokens per request; got 12 -- an entry bounds a step by its own pick for the part",
    ) as e:
        call_check(ops)
    assert "on this part" not in str(e.value) and "17 on a 99 KB per-CTA opt-in" in str(e.value)
    # the launch path resolves the pick from the table's shape: a planted kernel under the T-aware key is reached only after the
    # binding checks (CPU operands against a CUDA entry: the device check declines first, the token bound never did)
    calls = []
    k.compiled[_key(k, ops)] = lambda *args: calls.append(args)
    with pytest.raises(ValueError, match=r"the entry's \(cuda:0\); qkv is on cpu"):
        _run(k, ops)
    assert not calls
    # the cc 10.0 readings: the start form is CFG, the picks the measured ones whatever the token count, the envelope 35 / 8 / 45
    monkeypatch.setattr(entry, "device_shared_memory", lambda device: SM100_READING)
    monkeypatch.setattr(entry, "device_multiprocessor_count", lambda device: 148)
    k100 = entry.GdnVerifyFused("cuda:0", 1e-6)
    assert k100.cfg == kcfg.CFG and k100.readings == SM100_READING
    assert replace(k100.cfg_for(1, 30), NORM_FORM=None) == kcfg.PAIR_CFG and replace(k100.cfg_for(2, 8), NORM_FORM=None) == kcfg.CFG
    assert replace(k100.cfg_for(4, 6), NORM_FORM=None) == replace(k100.cfg_for(4, 40), NORM_FORM=None) == kcfg.TWO_CTA_CFG
    assert (k100.max_tokens_for(1), k100.max_tokens_for(2), k100.max_tokens_for(3), k100.max_tokens_for(4)) == (35, 8, 8, 45)
    # a part that holds no form: the typed decline at construction; a caller's explicit form is validated as given (its own typed error)
    monkeypatch.setattr(entry, "device_shared_memory", lambda device: (40960, 40000))
    with pytest.raises(NotImplementedError, match="40000-byte shared-memory opt-in per CTA .* holds no verify form"):
        entry.GdnVerifyFused("cuda:0", 1e-6)
    with pytest.raises(ValueError, match="does not fit 2 tokens per request on a 40000-byte opt-in"):
        entry.GdnVerifyFused("cuda:0", 1e-6, cfg=kcfg.PAIR_NBUF1_CFG)


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
def test_occupancy_pad_caps_the_ctas_per_sm_on_every_part_reading(per_sm, optin, tokens):
    """``CTAS_PER_SM`` n pads the footprint so that n + 1 CTAs exceed the SM while n fit it; where even the NATURAL footprints of n CTAs do
    not fit at the minimum token count (the streamed whole-slab form at n = 2 on the cc 12.0 reading: 2 x (79,200 + 1,024) B > 102,400 B)
    the knob is a typed decline, never a silent one-per-SM launch."""
    for n, base in ((1, dict(NBUF=1)), (2, dict(NBUF=1, STATE_STREAM=True, MIN_CTAS_PER_SM=2))):
        natural = kcfg.make_cfg(CTAS_PER_SM=0, SMEM_PER_SM_BYTES=per_sm, SMEM_OPTIN_BYTES=optin, **base).natural_smem_bytes(kcfg.TOKENS_MIN)
        if n * (natural + kcfg.SMEM_CTA_RESERVE_BYTES) > per_sm:
            with pytest.raises(ValueError, match="does not fit"):
                kcfg.make_cfg(CTAS_PER_SM=n, SMEM_PER_SM_BYTES=per_sm, SMEM_OPTIN_BYTES=optin, **base)
            continue
        c = kcfg.make_cfg(CTAS_PER_SM=n, SMEM_PER_SM_BYTES=per_sm, SMEM_OPTIN_BYTES=optin, **base)
        per_cta = c.smem_bytes(tokens) + kcfg.SMEM_CTA_RESERVE_BYTES
        assert (n + 1) * per_cta > per_sm, "one CTA more than allowed does not fit the SM"
        assert n * per_cta <= per_sm, "the allowed count does"
        assert c.smem_bytes(tokens) <= optin and c.smem_pad_bytes(tokens) % 16 == 0
        assert per_sm // per_cta == n and c.co_resident_ctas(tokens) == min(n, c.MIN_CTAS_PER_SM), "the pad admits exactly n by shared memory"


def test_cc120_reading_footprints_and_co_residency():
    """The cc 12.0 readings (100 KB per SM, 99 KB per-CTA opt-in) through the module's own arithmetic: the footprint law ``6,464 + 3,536 T
    + ring (+ 144 on the pair)`` with ``ring = NBUF x 65,664`` (one CTA) or ``NBUF x 32,896`` (the pair), the token bounds of the forms a
    99 KB opt-in holds, which of them two can share an SM -- ``co_resident_ctas`` reads BOTH budgets, the part's shared memory and the
    register cap the launch asks for -- and the three SM100 picks that do not hold even two tokens there."""
    per_sm, optin = SM120_READING
    assert SM120_READING == (102400, 101376) and SM100_READING == (233472, 232448)
    # the module forms a 99 KB opt-in holds: the pair with a 2-deep ring, the pair and the one-CTA form with the landing slab only
    assert kcfg.PAIR_NBUF2_CFG == replace(kcfg.PAIR_CFG, NBUF=2) and kcfg.PAIR_NBUF1_CFG == replace(kcfg.PAIR_CFG, NBUF=1)
    assert kcfg.ONE_CTA_NBUF1_CFG == replace(kcfg.CFG, NBUF=1) and kcfg.ONE_CTA_NBUF1_CFG.CTAS_PER_SM == 1 and not kcfg.ONE_CTA_NBUF1_CFG.TWO_CTAS_PER_HEAD
    assert all(f.ROLLED_TOKENS and f.MIN_CTAS_PER_SM == 1 and f.PDL is False for f in (kcfg.PAIR_NBUF2_CFG, kcfg.PAIR_NBUF1_CFG, kcfg.ONE_CTA_NBUF1_CFG))
    pair_nbuf2, pair_nbuf1, one_nbuf1 = (replace(f, **_on(SM120_READING)) for f in (kcfg.PAIR_NBUF2_CFG, kcfg.PAIR_NBUF1_CFG, kcfg.ONE_CTA_NBUF1_CFG))
    for form in (pair_nbuf2, pair_nbuf1, one_nbuf1):
        kcfg.validate_cfg(form)
    assert [pair_nbuf2.natural_smem_bytes(t) for t in (2, 4, 6)] == [79472, 86544, 93616] and pair_nbuf2.max_tokens() == 8
    assert [one_nbuf1.natural_smem_bytes(t) for t in (2, 4, 6)] == [79200, 86272, 93344] and one_nbuf1.max_tokens() == 8
    assert [pair_nbuf1.natural_smem_bytes(t) for t in (2, 3, 4, 6)] == [46576, 50112, 53648, 60720] and pair_nbuf1.max_tokens() == 17
    assert pair_nbuf1.natural_smem_bytes(17) == 99616 <= optin < pair_nbuf1.natural_smem_bytes(18) == 103152
    assert pair_nbuf2.natural_smem_bytes(9) == 104224 > optin and one_nbuf1.natural_smem_bytes(9) == 103952 > optin
    assert all(f.smem_pad_bytes(t) == 0 for f in (pair_nbuf2, pair_nbuf1, one_nbuf1) for t in (2, 6)), "no pad fits beside a slab on this part"
    for t in (2, 4, 6):
        assert one_nbuf1.natural_smem_bytes(t) == 6464 + 3536 * t + 65664 and pair_nbuf1.natural_smem_bytes(t) == 6464 + 3536 * t + 144 + 32896
        assert pair_nbuf2.natural_smem_bytes(t) == 6464 + 3536 * t + 144 + 2 * 32896
    # co-residency: every whole-slab form is one per SM by shared memory alone; the NBUF-1 pair is two by shared memory at T <= 3 and one
    # from T 4 -- and the host guarantees two only under the register cap (the uncapped pair measured 187 registers on sm_120a)
    two = replace(kcfg.TWO_CTA_CFG, **_on(SM120_READING))
    assert two.max_tokens() == 8 and two.co_resident_ctas(2) == 1 and 2 * (two.smem_bytes(2) + kcfg.SMEM_CTA_RESERVE_BYTES) == 160448 > per_sm
    assert [pair_nbuf2.co_resident_ctas(t) for t in (2, 6)] == [1, 1] and [one_nbuf1.co_resident_ctas(t) for t in (2, 6)] == [1, 1]
    assert [pair_nbuf1.co_resident_ctas(t) for t in (2, 3, 4)] == [1, 1, 1]
    capped = replace(pair_nbuf1, MIN_CTAS_PER_SM=2)
    assert [capped.co_resident_ctas(t) for t in (2, 3, 4, 6)] == [2, 2, 1, 1] and 2 * (capped.smem_bytes(3) + kcfg.SMEM_CTA_RESERVE_BYTES) == 102272
    for form in (kcfg.PAIR_CFG, kcfg.CFG):  # the 3-deep rings: 112,368 / 210,528 B at T 2 -- typed declines, by the opt-in
        assert replace(form, **_on(SM120_READING)).max_tokens() == 0
        with pytest.raises(ValueError, match="does not fit 2 tokens per request on a 101376-byte opt-in"):
            kcfg.validate_cfg(replace(form, **_on(SM120_READING)))
    # the cc 10.0 readings: TWO_CTA_CFG is two per SM to T 12 (two by shared memory, two by the cap); the uncapped picks guarantee one
    two100 = replace(kcfg.TWO_CTA_CFG, **_on(SM100_READING))
    assert (two100.co_resident_ctas(2), two100.co_resident_ctas(12), two100.co_resident_ctas(13), two100.max_tokens()) == (2, 2, 1, 45)
    assert replace(kcfg.CFG, **_on(SM100_READING)).co_resident_ctas(2) == replace(kcfg.PAIR_CFG, **_on(SM100_READING)).co_resident_ctas(2) == 1


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
    with pytest.raises(NotImplementedError, match="module default form .* at most 8"):
        call_check(ops)  # no form given: the module default form's bound, named as such
    call_check(ops, cfg=kcfg.make_cfg(NBUF=2))  # a shallower ring holds it
    with pytest.raises(NotImplementedError, match="at most 8 on this part"):
        call_check(ops, cfg=kcfg.CFG)  # the same form given explicitly: validate_tokens' per-form bound


def test_a_table_of_fewer_tokens_than_the_minimum_is_check_bindings_typed_decline_on_a_cuda_entry(monkeypatch):
    """A ``[requests, 0]`` slot table on an entry that picks per step shape without a given form: the pick runs at the minimum token count
    (``cfg_for`` clamps the table's count, never raising on it) so ``check_binding`` keeps the typed ``NotImplementedError`` naming the
    minimum on the launch and compile paths; ``key()`` carries the minimum-count pick without raising.  Host-only: a host entry steered to
    each part's SM count and readings (as the host cells above steer theirs), so the CPU operands and the entry share one device and the
    token bound is the first check that can decline."""
    for reading, sms in ((SM120_READING, 170), (SM100_READING, 148)):
        k = entry.GdnVerifyFused("cpu", 1e-6)
        k.sm_count, k.readings, k.part = sms, reading, _on(reading)  # the pick's inputs on this part; the forms it stamps carry the reading
        ops = dict(make_operands(requests=1, tokens=2), state_indices=torch.zeros(1, 0, dtype=torch.int32))
        assert entry.step_shape(ops["state_indices"]) == (1, 0) and ops["state_indices"].stride(1) == 1
        with pytest.raises(NotImplementedError, match="serves at least 2 tokens per request"):
            _run(k, ops)
        with pytest.raises(NotImplementedError, match="serves at least 2 tokens per request"):
            k.compile(**ops)
        assert _key(k, ops)[1:3] == (0, k.cfg_for(1, 2)), reading  # the key carries the minimum-count pick; nothing was compiled or launched
        assert k.cfg_for(1, 0) is k.cfg_for(1, 1) is k.cfg_for(1, 2) is k.cfg_for(1) and not k.compiled
        with pytest.raises(ValueError, match="tokens must be an int or None"):
            k.cfg_for(1, 2.0)


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


def test_binding_check_requires_every_operand_on_the_given_device():
    """With the entry's device given (a concrete CUDA ordinal), every operand must live on exactly it, type AND index: CPU operands of
    the served shapes and strides are a ``ValueError`` naming the operand and both devices -- host-only, no CUDA needed."""
    ops = make_operands()
    with pytest.raises(ValueError, match=r"one device, the entry's \(cuda:0\); qkv is on cpu"):
        entry.check_binding(**ops, device=torch.device("cuda", 0))
    entry.check_binding(**ops, device="cpu")  # the operands' own device passes
    with pytest.raises(ValueError, match=r"the entry's \(cpu\); ssm_state is on meta"):
        entry.check_binding(**dict(ops, ssm_state=ops["ssm_state"].to("meta")), device="cpu")
    with pytest.raises(ValueError, match=r"qkv's \(cpu\); cu_seqlens is on meta"):
        entry.check_binding(**dict(ops, cu_seqlens=ops["cu_seqlens"].to("meta")))  # without a device: qkv's


def test_entry_device_is_a_concrete_ordinal_that_guards_a_warm_cache_too(monkeypatch):
    """A CUDA entry resolves ``cuda`` to the current ordinal at construction, and a WARM entry (a kernel under the binding's compile key)
    still runs the device check before the compiled callable is looked up: CPU operands of the same shapes, strides and dtypes (the key
    matches) are a ``ValueError`` on the launch path and on the compile path, the planted kernel never invoked.  Host-only: the device
    readers (the part's shared-memory readings, the SM count, the current ordinal) are stubbed."""
    monkeypatch.setattr(entry, "device_shared_memory", lambda device: PART_READINGS[0])
    monkeypatch.setattr(entry, "device_multiprocessor_count", lambda device: 148)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    k = entry.GdnVerifyFused("cuda", 1e-6)
    assert k.device == torch.device("cuda", 1) and k.sm_count == 148
    assert entry.GdnVerifyFused("cuda:0", 1e-6).device == torch.device("cuda", 0)  # a given ordinal stays
    assert entry.GdnVerifyFused("cpu", 1e-6).device == torch.device("cpu")  # a host entry: nothing to resolve
    ops = make_operands(requests=2, tokens=4)
    key = _key(k, ops)
    calls = []
    k.compiled[key] = lambda *args: calls.append(args)
    with pytest.raises(ValueError, match=r"the entry's \(cuda:1\); qkv is on cpu"):
        _run(k, ops)
    with pytest.raises(ValueError, match=r"the entry's \(cuda:1\); qkv is on cpu"):
        k.compile(**ops)
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


def test_debug_buffers_exist_under_debug_only():
    """Without DEBUG the kernel's three debug slots are compiled out (None), never bound to an allocation; a debug entry allocates them."""
    plain, debug = entry.GdnVerifyFused("cpu", 1e-6), entry.GdnVerifyFused("cpu", 1e-6, debug=True)
    assert set(plain.outputs(2, 4)) == {"qout", "out", "counts"}
    assert plain._debug_args(plain.outputs(2, 4)) == [None, None, None]
    assert set(debug.outputs(2, 4)) == {"qout", "out", "counts", "raw", "convout", "baout"}
    assert tuple(debug.outputs(2, 4)["raw"].shape) == (8, kcfg.V_HEADS, kcfg.HEAD_DIM) and tuple(debug.outputs(2, 4)["baout"].shape) == (8, 2 * kcfg.V_HEADS)


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
        pytest.skip(f"the fused GDN verify step serves compute capability {entry.SUPPORTED_CC_TEXT}, this device is {cc}")
    msg = entry._dsl_version_error()
    if msg:
        pytest.skip(msg)
    return torch.device("cuda")


def _run(k, ops, out=None, out_bf16=None):
    return k(**ops, out=out, out_bf16=out_bf16)


def _part_form(dev, forms):
    """The first of ``forms`` this device's shared memory holds (``PAIR_CFG`` / ``CFG`` on a cc 10.0 part; the 2-deep-ring pair / the
    landing-slab one-CTA form on a 99 KB opt-in): the explicit form a cell compares the pick against on THIS part."""
    return kcfg.first_fitting(forms, *entry.device_shared_memory(dev))


def _fits_or_declines(dev, cfg, tokens, geometry=None):
    """``cfg`` stamped with this part's readings when it holds ``tokens`` per request; else the typed decline is asserted (a form whose
    shared memory the part does not hold at the minimum token count is validate_cfg's ValueError at construction, one that holds too few
    tokens is validate_tokens' NotImplementedError at the call) and the cell is SKIPPED with that reason -- the count then shows which
    arms ran on the part (every arm on the cc 10.0 readings; 17 of 43 on the cc 12.0 ones), and an unheld arm is never a launch."""
    on_part = replace(cfg, **_on(entry.device_shared_memory(dev)), **({} if geometry is None else dict(GEOM=geometry)))
    try:
        kcfg.validate_cfg(on_part)
    except ValueError as exc:
        assert "does not fit" in str(exc), exc
        with pytest.raises(ValueError, match="does not fit"):
            entry.GdnVerifyFused(dev, 1e-6, cfg=cfg, geometry=geometry)
        pytest.skip(f"this part's shared memory does not hold the arm (declined by type at construction): {exc}")
    if tokens > on_part.max_tokens():
        with pytest.raises(NotImplementedError, match=f"at most {on_part.max_tokens()} on this part"):
            kcfg.validate_tokens(on_part, tokens)
        pytest.skip(f"this part's shared memory holds the arm for at most {on_part.max_tokens()} tokens per request, not {tokens} (declined by type)")
    return on_part


def _snapshot(k, ops):
    """Run once; the bytes of every output (the fp8 / bf16 outputs, the state and conv pools, the tickets back at zero)."""
    rows = ops["qkv"].shape[0]
    outb = torch.empty(rows, int(ops["ssm_state"].shape[1]), kcfg.HEAD_DIM, dtype=torch.bfloat16, device=ops["qkv"].device)  # the operands' V heads
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
    assert k.readings == (props.shared_memory_per_multiprocessor, props.shared_memory_per_block_optin)
    assert k.cfg == replace(kcfg.first_fitting(kcfg.ONE_CTA_FORMS, *k.readings), **k.part), "the start form: the first one-CTA form this part holds"
    for requests in (1, 2, 3, 4, 8, 16):
        for tokens in (None, 2, 6, 12):
            want, got = kcfg.default_cfg(requests, k.sm_count, *k.readings, tokens), k.cfg_for(requests, tokens)
            assert replace(got, NORM_FORM=None, **_on(SM100_READING)) == want, (requests, tokens)
            assert got.NORM_FORM == kcfg.norm_form_for(requests) and got.PDL is False
            assert (got.SMEM_PER_SM_BYTES, got.SMEM_OPTIN_BYTES) == k.readings
        assert k.max_tokens_for(requests) == kcfg.max_tokens_served(requests, k.sm_count, *k.readings)
    one = k.cfg_for(1)
    assert one.TWO_CTAS_PER_HEAD == (kcfg.V_HEADS * 2 <= k.sm_count)  # the pair while its 96 cluster CTAs have an SM each
    assert (
        one.co_resident_ctas(6) == 1 and 2 * (one.smem_bytes(6) + kcfg.SMEM_CTA_RESERVE_BYTES) > one.SMEM_PER_SM_BYTES
    )  # one CTA per SM by shared memory at T 6
    beyond = k.cfg_for(k.sm_count // kcfg.V_HEADS + 1)  # the first requests count whose grid outgrows the SMs
    if beyond.MIN_CTAS_PER_SM == 2:  # the capped form: the part's shared memory holds two of its CTAs per SM (the cc 10.0 readings)
        assert (
            beyond.co_resident_ctas(kcfg.TOKENS_MIN) == 2 and 2 * (beyond.smem_bytes(kcfg.TOKENS_MIN) + kcfg.SMEM_CTA_RESERVE_BYTES) <= beyond.SMEM_PER_SM_BYTES
        )
    else:  # the part keeps every whole-slab form at one CTA per SM, so the cap left the pick (a 99 KB opt-in): the 2-deep-ring pair serves
        assert beyond.co_resident_ctas(kcfg.TOKENS_MIN) == 1 and replace(kcfg.TWO_CTA_CFG, **k.part).co_resident_ctas(kcfg.TOKENS_MIN) == 1
        assert not kcfg.admissible(kcfg.TWO_CTA_CFG, *k.readings) and beyond.TWO_CTAS_PER_HEAD and beyond.NBUF == 2
        assert replace(beyond, NORM_FORM=None, **_on(SM100_READING)) == kcfg.first_fitting(kcfg.BEYOND_SMALL_SMEM_ORDER, *k.readings)


def test_warm_entry_rejects_cpu_operands_before_the_compiled_kernel_is_looked_up():
    """After a real compile and launch, the same entry called with CPU operands of the same shapes, strides and dtypes (the compile key
    matches: a cache hit) is a ``ValueError`` from the device check on the launch path and on the compile path, before the compiled
    callable is looked up (swapped for a recorder: no call); the entry's device is a concrete ordinal, and it launches again afterwards."""
    dev = _device_or_skip()
    k = entry.GdnVerifyFused(dev, 1e-6)
    assert k.device == torch.device("cuda", torch.cuda.current_device())
    ops = make_operands(requests=2, tokens=4, device=dev)
    _run(k, ops)
    torch.cuda.synchronize()
    key = _key(k, ops)
    cpu = make_operands(requests=2, tokens=4)
    assert _key(k, cpu) == key, "the same shapes, strides and dtypes: a warm key"
    real, calls = k.compiled[key], []
    k.compiled[key] = lambda *args: calls.append(args)
    try:
        with pytest.raises(ValueError, match=rf"the entry's \({k.device}\); qkv is on cpu"):
            _run(k, cpu)
        with pytest.raises(ValueError, match=rf"the entry's \({k.device}\); qkv is on cpu"):
            k.compile(**cpu)
    finally:
        k.compiled[key] = real
    assert not calls, "the compiled kernel was never invoked on CPU operands"
    _run(k, ops)
    torch.cuda.synchronize()


def test_operands_on_another_cuda_device_are_rejected_before_any_launch():
    """Operands on another CUDA device than the entry's (same shapes, strides and dtypes: the compile key matches) are a ``ValueError``
    naming both ordinals, on the launch path and on the compile path, the compiled kernel never invoked.  Needs two CUDA devices."""
    dev = _device_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    k = entry.GdnVerifyFused(dev, 1e-6)
    other = torch.device("cuda", (k.device.index + 1) % torch.cuda.device_count())
    ops = make_operands(requests=2, tokens=4, device=dev)
    _run(k, ops)
    torch.cuda.synchronize()
    key = _key(k, ops)
    foreign = make_operands(requests=2, tokens=4, device=other)
    assert _key(k, foreign) == key
    real, calls = k.compiled[key], []
    k.compiled[key] = lambda *args: calls.append(args)
    try:
        with pytest.raises(ValueError, match=rf"the entry's \({k.device}\); qkv is on {other}"):
            _run(k, foreign)
        with pytest.raises(ValueError, match=rf"the entry's \({k.device}\); qkv is on {other}"):
            k.compile(**foreign)
    finally:
        k.compiled[key] = real
    assert not calls, "the compiled kernel was never invoked on another device's operands"


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


@pytest.mark.parametrize("requests, tokens, accepted, pad, skip", REFERENCE_CELLS, ids=str)
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
def test_pick_beyond_the_reference_kernel_envelope_matches_the_torch_reference(requests):
    """8 and 16 requests (the reference verify kernel declines them): the pick past the one-CTA grid -- the two-CTAs-per-SM form on a cc
    10.0 part, the 2-deep-ring pair on a 99 KB opt-in (768 / 1,536 cluster CTAs over the SMs) -- against the torch reference, the same bars;
    the gated-norm association is the reference's one-CTA-per-head chain order (ctas1)."""
    dev = _device_or_skip()
    ops = make_operands(requests, 6, dev, seed=800 + requests, pad="last", skip="mid")
    condition_conv_inputs(ops)
    ssm0, conv0 = pools_before(ops)
    k = entry.GdnVerifyFused(dev, 1e-6)
    assert replace(k.cfg_for(requests, 6), NORM_FORM=None, **_on(SM100_READING)) == kcfg.default_cfg(requests, k.sm_count, *k.readings, 6)
    assert k.cfg_for(requests, 6).NORM_FORM == "ctas1" and requests * kcfg.V_HEADS > k.sm_count
    outb = torch.empty(requests * 6, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.bfloat16, device=dev)
    q = _run(k, ops, out_bf16=outb)
    torch.cuda.synchronize()
    check_against_reference(ops, q, outb, ssm0, conv0)


@pytest.mark.parametrize("requests, tokens", [(1, 10), (2, 12), (4, 17)], ids=str)
def test_long_steps_on_the_landing_slab_pair_match_the_torch_reference(requests, tokens):
    """Steps of 10 .. 17 tokens per request against the torch reference: where this part's pick holds them (a 99 KB opt-in: the
    landing-slab pair, the T 9 .. 17 fall-through at every requests count; a cc 10.0 part: the 3-deep pair at one request, the
    two-CTAs-per-SM form from four) the pick serves; where it does not (a cc 10.0 part at two requests holds 8) the landing-slab pair is
    forced as an explicit form -- so the fall-through form of the 99 KB parts runs on every part, never a skip."""
    dev = _device_or_skip()
    k = entry.GdnVerifyFused(dev, 1e-6)
    if tokens > k.max_tokens_for(requests):
        k = entry.GdnVerifyFused(dev, 1e-6, cfg=kcfg.PAIR_NBUF1_CFG)
        assert k.cfg_for(requests, tokens).max_tokens() >= tokens
    ops = make_operands(requests, tokens, dev, seed=900 + 10 * requests + tokens, accepted="random", pad="last" if requests > 1 else "none", skip="mid")
    condition_conv_inputs(ops)
    ssm0, conv0 = pools_before(ops)
    outb = torch.empty(requests * tokens, kcfg.V_HEADS, kcfg.HEAD_DIM, dtype=torch.bfloat16, device=dev)
    q = _run(k, ops, out_bf16=outb)
    torch.cuda.synchronize()
    assert not k.outputs(requests, tokens)["counts"].any()
    check_against_reference(ops, q, outb, ssm0, conv0)


@pytest.mark.parametrize("form", ["pick", "pair"])
def test_padding_and_skipped_slots_touch_nothing_else(form):
    """The entry's pick at 4 requests (the two-CTAs-per-SM form on a 148- / 152-SM part; the 2-deep-ring pair on a 99 KB opt-in) and the
    two-CTAs-per-head pair this part holds forced (on a 99 KB opt-in that forced pair IS the 2-deep-ring pick, so both arms run one form
    there; the arm is kept as the explicit pair): a padding request (both CTAs of a pair null together: the cluster wait pairs
    the entry arrive, nothing shipped) zeroes its rows and touches nothing else; a skipped slot stays."""
    dev = _device_or_skip()
    ops = make_operands(4, 6, dev, seed=5, accepted="random", pad="mid", skip="last")
    idx = ops["state_indices"]
    ssm0, conv0 = ops["ssm_state"].clone(), ops["conv_state"].contiguous().clone()
    k = entry.GdnVerifyFused(dev, 1e-6, cfg=None if form == "pick" else _part_form(dev, kcfg.PAIR_FORMS))
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
def test_knob_arms_the_part_holds_are_bitwise_with_the_default_pick(arm):
    """Every knob arm this part's shared memory holds at T 6 writes the bytes of the entry's pick; an arm it does not hold (the 3-deep
    rings and the two-slab pads on a 99 KB opt-in: 26 of the 43 arms) is asserted to be the typed decline and reported as a skip with that
    reason (``_fits_or_declines``), never a launch -- so the count shows which arms ran on the part."""
    dev = _device_or_skip()
    cfg = kcfg.make_cfg(**arm)
    _fits_or_declines(dev, cfg, 6)
    base = _default_snapshot(dev)
    ops = make_operands(4, 6, dev, seed=31, accepted="random", pad="last", skip="mid")
    got = _snapshot(entry.GdnVerifyFused(dev, 1e-6, cfg=cfg), ops)
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
    """One request = the pair's own regime (the pick on a 148- / 152-SM part; the ctas2 norm association on both forms): the pair this
    part holds (rolled = PAIR_CFG, or the 2-deep-ring pair on a 99 KB opt-in; and unrolled), the one-CTA form it holds (CFG, or the
    landing-slab form) and the entry's pick agree byte for byte on the fp8 / bf16 outputs, every state slot and the conv pool."""
    dev = _device_or_skip()
    pair_form, one_form = _part_form(dev, kcfg.PAIR_FORMS), _part_form(dev, kcfg.ONE_CTA_FORMS)
    ops = [make_operands(1, tokens, dev, seed=41, accepted="random", skip="last") for _ in range(3)]
    pick = entry.GdnVerifyFused(dev, 1e-6)
    assert pick.cfg_for(1, tokens).TWO_CTAS_PER_HEAD == (2 * kcfg.V_HEADS <= pick.sm_count)
    got_pick = _snapshot(pick, ops[0])
    got_pair = _snapshot(entry.GdnVerifyFused(dev, 1e-6, cfg=pair_form), ops[1])
    got_one = _snapshot(entry.GdnVerifyFused(dev, 1e-6, cfg=one_form), ops[2])
    _assert_bitwise(got_one, got_pair, "the pair vs the one-CTA form at one request")
    _assert_bitwise(got_one, got_pick, "the pick vs the one-CTA form at one request")
    ops_u = make_operands(1, tokens, dev, seed=41, accepted="random", skip="last")
    got_unrolled = _snapshot(entry.GdnVerifyFused(dev, 1e-6, cfg=replace(pair_form, ROLLED_TOKENS=False)), ops_u)
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


# ---------------------------------------------------------------------------------------------------------------------------
# the per-rank geometries on the device: the torch reference and determinism per step shape, every form family bitwise with the one-CTA
# form on a set with a padding request, a skipped slot and non-monotone slots, the requests of a step batch-invariant -- the 27B cells
# above are unchanged (the geometry is the compile key, and the default geometry's renderings are byte-identical to the pre-geometry kernels)
# ---------------------------------------------------------------------------------------------------------------------------
RANK_VERIFY_SHAPES = [(requests, tokens) for requests in (1, 2, 4) for tokens in (2, 4, 6)] + [(8, 6)]  # 8 x 6: past the one-CTA grid at 32 V heads
# the form families (Cfg deltas from the unrolled one-CTA form): the three pick forms (rolled pair / rolled one-CTA / two CTAs per SM),
# the shallower rings, the per-row thread mappings, the register-capped mappings, the streamed state path, the PDL order, the spread
# bulk issue, the per-thread store path, non-resident K, the BA prefetch depths, and the pair with the knobs it is picked with
RANK_VERIFY_ARMS = [
    dict(),  # the one-CTA form with the unrolled token loop
    dict(ROLLED_TOKENS=True),  # = CFG: the base itself, a second entry
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True),  # = PAIR_CFG
    dict(TPR=4, NBUF=1, STATE_STREAM=True, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0, BA_PREFETCH_CHUNKS=1),  # = TWO_CTA_CFG
    dict(ROLLED_TOKENS=True, NBUF=1),  # = ONE_CTA_NBUF1_CFG
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True, NBUF=2),  # = PAIR_NBUF2_CFG
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True, NBUF=1),  # = PAIR_NBUF1_CFG
    dict(NBUF=2),
    dict(K_RESIDENT=False),
    dict(BA_PREFETCH_CHUNKS=1),
    dict(BA_PREFETCH_CHUNKS=2),
    dict(TPR=2, NBUF=2),
    dict(TPR=8, NBUF=1),
    dict(TPR=16, NBUF=1),
    dict(TPR=8, NBUF=1, MIN_CTAS_PER_SM=2, CTAS_PER_SM=0, BA_PREFETCH_CHUNKS=1),
    dict(STATE_STREAM=True, NBUF=2),
    dict(PDL=True, PDL_RELEASE=True),
    dict(BULK_ISSUE="spread"),
    dict(STATE_STORE_PATH="stg", NBUF=1),
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, TPR=8),
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, BA_PREFETCH_CHUNKS=1),
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True, PDL=True, PDL_RELEASE=True),
    dict(TWO_CTAS_PER_HEAD=True, CTAS_PER_SM=0, ROLLED_TOKENS=True, BULK_ISSUE="spread"),
]
RANK_SET = dict(accepted="random", pad="last", skip="mid")  # the 4-request set: a padding request, a skipped slot write, non-monotone slots


@pytest.mark.parametrize("requests, tokens", RANK_VERIFY_SHAPES, ids=str)
@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_rank_geometry_matches_the_torch_reference_and_replays_bitwise(g, requests, tokens):
    """The entry's pick for the step shape at a per-rank geometry (the pair, one CTA per SM, or -- past the one-CTA grid, 8 requests at 32
    V heads on 152 SMs -- the one-CTA form in two waves, by ``requests x V`` against this part's SM count: the module pick) against the
    FP64 torch reference at the reference kernel's bars (random acceptance
    counts, a padding request from two requests on, a skipped slot write, non-monotone slots), the tickets back at zero -- and twice on
    fresh identical operands: byte-identical (fp8 / bf16 outputs, every state slot, the conv pool)."""
    dev = _device_or_skip()
    kw = dict(accepted="random", pad="last" if requests > 1 else "none", skip="mid")
    seed = 700 + 10 * requests + tokens
    ops = make_operands(requests, tokens, dev, seed=seed, geometry=g, **kw)
    condition_conv_inputs(ops)
    twin = make_operands(requests, tokens, dev, seed=seed, geometry=g, **kw)
    twin["qkv"].copy_(ops["qkv"])  # the same conditioned inputs (the conditioning moves qkv values only)
    ssm0, conv0 = pools_before(ops)
    k = entry.GdnVerifyFused(dev, 1e-6, geometry=g)
    cfg = k.cfg_for(requests, tokens)
    assert cfg.GEOM == g and replace(cfg, NORM_FORM=None, **_on(SM100_READING)) == kcfg.default_cfg(requests, k.sm_count, *k.readings, tokens, g)
    got = _snapshot(k, ops)
    check_against_reference(ops, got["qout"].view(torch.float8_e4m3fn), got["out"].view(torch.bfloat16), ssm0, conv0)
    again = _snapshot(k, twin)
    _assert_bitwise(got, again, f"{_gid(g)} {requests} x {tokens}: two identical operand sets")


@pytest.mark.parametrize("arm", RANK_VERIFY_ARMS, ids=lambda a: ",".join(f"{k}={v}" for k, v in a.items()) or "unrolled-one-cta-form")
@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_rank_geometry_knob_arms_are_bitwise_with_the_one_cta_form(g, arm):
    """Every form family this part holds at T 6, at a per-rank geometry, writes the bytes of the one-CTA form THIS part holds at that
    geometry (``CFG``, the rolled 3-deep-ring form, on a cc 10.0 part; the landing-slab one-CTA form on a 99 KB opt-in -- ``_part_form``,
    as the 27B cells take it: the footprints carry no geometry term) on the 4-request set with a padding request, a skipped slot and
    non-monotone slots: the fp8 / bf16 outputs, every state slot, the conv pool; the padding request zero, the tickets back at zero.  An
    arm the part does not hold is its typed decline, reported as a skip (``_fits_or_declines``), and so is a part whose one-CTA form does
    not hold T 6."""
    dev = _device_or_skip()
    cfg = kcfg.with_geometry(kcfg.make_cfg(**arm), g)  # the form compiled for the geometry (a form of another geometry beside geometry= is refused)
    _fits_or_declines(dev, cfg, 6)
    base_form = kcfg.with_geometry(_part_form(dev, kcfg.ONE_CTA_FORMS), g)  # the one-CTA form this part holds (CFG on cc 10.0), at the geometry
    _fits_or_declines(dev, base_form, 6)
    base = _snapshot(entry.GdnVerifyFused(dev, 1e-6, cfg=base_form), make_operands(4, 6, dev, seed=31, geometry=g, **RANK_SET))
    got = entry.GdnVerifyFused(dev, 1e-6, cfg=cfg)
    assert got.geometry == g and got.cfg.GEOM == g
    got = _snapshot(got, make_operands(4, 6, dev, seed=31, geometry=g, **RANK_SET))
    _assert_bitwise(base, got, f"{_gid(g)} {arm}")
    assert not got["qout"][18:24].any() and got["qout"][:18].any(), "the padding request is zero, the live rows written"


@pytest.mark.parametrize("g", RANK_GEOMETRIES, ids=_gid)
def test_rank_geometry_requests_are_batch_invariant(g):
    """Batch invariance at a per-rank geometry: every live request of a 4-request T-6 step (the entry's pick at four requests; random
    acceptance counts, a skipped slot, non-monotone slots) writes the bytes it writes alone from the same pools -- its fp8 / bf16 rows,
    its written state slots, its conv slot -- so a request's result does not depend on which requests share its step.  Under ONE gated-norm
    association: the lone request runs the one-CTA form this part holds (``CFG`` on a cc 10.0 part, the landing-slab form on a 99 KB
    opt-in -- ``_part_form``) with the multi-request association (``ctas1``, the pick's at four requests), because the one-request pick
    itself takes the reference kernel's two-CTAs-per-head association (``ctas2``: 4-row chains) by policy, which rounds the normed
    output's last bits differently by design (``norm_form_for``); the forms' agreement under one association is the cross-form cell above
    and the 27B's one-request pair-vs-one-CTA cell."""
    dev = _device_or_skip()
    requests, tokens = 4, 6
    k = entry.GdnVerifyFused(dev, 1e-6, geometry=g)
    assert k.cfg_for(requests, tokens).NORM_FORM == "ctas1"
    one_cta = kcfg.with_geometry(_part_form(dev, kcfg.ONE_CTA_FORMS), g)  # the one-CTA form this part holds, at the geometry
    _fits_or_declines(dev, one_cta, tokens)
    alone = entry.GdnVerifyFused(dev, 1e-6, cfg=replace(one_cta, NORM_FORM="ctas1"))
    kw = dict(accepted="random", skip="mid")
    ops = make_operands(requests, tokens, dev, seed=71, geometry=g, **kw)
    idx = ops["state_indices"].clone()
    snap4 = _snapshot(k, ops)
    for n in range(requests):
        again = make_operands(requests, tokens, dev, seed=71, geometry=g, **kw)  # the same draws; request n alone, the pools whole
        rows = slice(n * tokens, (n + 1) * tokens)
        one = dict(
            again,
            qkv=again["qkv"][rows],  # row slices of the slab keep its 16-byte-multiple row stride (the serving layout's views)
            z=again["z"][rows],
            hidden=again["hidden"][rows],
            state_indices=again["state_indices"][n : n + 1].clone(),  # fresh allocations: a sliced int32 table is not 16-byte aligned at odd n
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=dev),
            num_accepted_tokens=again["num_accepted_tokens"][n : n + 1].clone(),
        )
        snap1 = _snapshot(alone, one)
        assert torch.equal(snap1["qout"], snap4["qout"][rows]) and torch.equal(snap1["out"], snap4["out"][rows]), f"{_gid(g)} request {n}: the outputs"
        for t in range(tokens):
            slot = int(idx[n, t])
            if slot > 0:
                assert torch.equal(snap1["ssm"][slot], snap4["ssm"][slot]), f"{_gid(g)} request {n}: state slot {slot}"
        assert torch.equal(snap1["conv"][int(idx[n, 0])], snap4["conv"][int(idx[n, 0])]), f"{_gid(g)} request {n}: the conv slot"
