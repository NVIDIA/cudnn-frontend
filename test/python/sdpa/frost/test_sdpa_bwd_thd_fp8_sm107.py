# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm107_fp8`` THD / varlen: the packed d = 256 per-tensor FP8 E4M3 backward on the Rubin line, end to end.

The twin of ``test_sdpa_bwd_thd_sm107.py`` (bf16 / fp16) for the fp8 row -- reuse by import, never copy: the tail sentinel, the
stage-3 band twins and the case tables come from there; the scalar set, the amax set, the tolerance recipe and the ``ds_knob``
fixture from ``test_sdpa_bwd_fp8_sm107.py``.

Two surfaces:

* ``_run_fp8_direct`` drives ``SdpaBwdDslSm107Fp8(thd=True, ...)`` DIRECTLY over PACKED e4m3 operands, the twelve scalars and the
  requested amax outputs -- the numerics of the chain one layer below the plan machinery, so a failure localises to the kernels.
  Every accept cell of this module runs on this tier.
* ``_run_fp8_graph`` goes through the ragged ``sdpa_fp8_backward`` GRAPH with the engine pinned.  The fp8 backward node can
  declare its packed totals only once ``SDPA_fp8_backward_attributes`` carries ``max_total_seq_len_q/kv`` (the bf16 node has had
  them); until that attribute lands the graph tier is a typed-decline probe
  (``test_graph_thd_fp8_tier_declines_typed_until_the_totals_attribute_lands``) and the graph accept cells skip, naming it.

The reference is the repo's fp8 backward oracle (``sdpa.fp8_ref.compute_ref_backward``) run PER SEQUENCE over the packed live
tokens under the fp8 row's recipe (``_FP8_GRAD_TOL`` through ``assert_close_fp8_grad`` with its midpoint-flip budget; ``_AMAX_DS_TOL``
on ``amax_dP``) -- the dense fp8 suite's bounds, never a looser one.  The packed quantization is the forward THD row's convention:
ONE scalar per operand over the packed live tokens, so every sequence shares the twelve scalars the kernel reads -- and therefore
ONE ``scale_dP`` per packed batch: the per-sequence oracle rounds its dS at that global scale (``dP_scale=``), the gradients are
quantized at the global ``scale_dQ / dK / dV`` (``quantize_grads=False`` + this module's cast), and the four ``amax_*`` are compared
with the maximum over the packed LIVE region of the oracle's pre-quant tensors -- dead units, pad rows, pad columns and the NaN
capacity tail excluded (sdpa-invariants s5).  The oracle yields the forward's convention on a fully masked row (``O = 0``,
``LSE = -inf`` -> ``P = 0`` -> ``dQ = 0``), and the cells with such rows assert the exact zeros on the OUTPUT directly.  No case
table carries a length-1 sequence (a (1 x 1) matmul in the sibling module's fp64 oracle trips a Triton codegen defect before any
kernel runs); five rows is the shortest sequence here too.
"""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl, requires_rubin, requires_sm80, select_engine
from test_sdpa_bwd_dsl_sm107 import _adapter
from test_sdpa_bwd_fp8_sm107 import _AMAX, _AMAX_DS_TOL, _DS_KNOBS, _FP8_GRAD_TOL, _SCALARS, _ds_dtype_code, ds_knob  # noqa: F401  (ds_knob: fixture by import)
from test_sdpa_bwd_thd_sm107 import (
    _GQA_TWIN_CASES,
    _NO_KEY_ROW_CASES,
    _TRIM_TWIN_CASES,
    _assert_empty_sequence_exactly_zero,
    _assert_tails_untouched,
    _plan_index,
    _rows_with_keys,
    _sentinel_tails,
)
from test_sdpa_bwd_thd_sm107 import test_stage3_thd_band_arithmetic as _band_arithmetic  # the tk-parametrized body, called with the fp8 K tile

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3

pytestmark = [pytest.mark.L0, requires_dsl]

_ENGINE = "sdpa_bwd_sm107_fp8"
_D = 256
_RUBIN_CC = (10, 7)
_T_E4M3 = torch.float8_e4m3fn
_FP8 = cudnn.data_type.FP8_E4M3
_CUDNN_GRAD = {_T_E4M3: _FP8, torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float16: cudnn.data_type.HALF}
_GRAD_IDS = {_T_E4M3: "e4m3", torch.bfloat16: "bf16", torch.float16: "fp16"}


@pytest.fixture(autouse=True)
def _mock_target_for_cross_arch_contracts(monkeypatch):
    # The host rejects probe the Rubin row off the Rubin line (the analyzer's cc faked to 10.7); mismatch() carries the sm_107a
    # DSL gate, so the fake device gets a fake compiler target too -- exactly as the sibling sm107 suites do.
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != _RUBIN_CC:
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


def _spec(name=_ENGINE):
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == name), None)
    assert spec is not None, f"{name} is not in cudnn.sdpa.bwd.engines.ENGINE_SPECS"
    return spec


def _cu(lens):
    c = [0]
    for n in lens:
        c.append(c[-1] + int(n))
    return c


def _binding_declares_totals() -> bool:
    """Whether the NATIVE ``sdpa_fp8_backward`` binding takes ``max_total_seq_len_q/kv``: the python-native graph captures any kwarg
    and the analyzer reads the totals off the captured record, but the C++ replay forwards every kwarg to this binding -- so the
    binding's own signature (pybind writes it into the docstring) is the fact that decides whether a ragged fp8 graph can declare
    its totals."""
    return "max_total_seq_len_q" in (cudnn._pybind_module.backend_graph.sdpa_fp8_backward.__doc__ or "")


# --------------------------------------------------------------------------- the packed fp8 case + the per-sequence fp8 oracle


def _quant_packed(x_live):
    """ONE e4m3 scale per operand over the packed LIVE tokens (the forward THD row's convention); returns (codes, descale)."""
    from sdpa.helpers import get_fp8_scale_factor

    amax = x_live.abs().max().item() if x_live.numel() else 0.0
    scale = get_fp8_scale_factor(amax, _T_E4M3)
    return (x_live * scale).to(_T_E4M3), 1.0 / scale


def _thd_fp8_case(
    lens_q,
    lens_kv,
    h,
    hkv=None,
    *,
    cap_q=None,
    cap_kv=None,
    poison=False,
    seed=7,
    causal=False,
    bottom_right=False,
    window_left=None,
    grad_dtype=_T_E4M3,
    quantize_ds=True,
    device="cuda",
    oracle=True,
):
    """Packed e4m3 Q / K / V / dO (one scalar per operand over the packed live tokens), the forward's packed e4m3 O and packed
    natural-log Stats (per sequence through ``sdpa.fp8_ref.compute_ref``, O quantized ONCE at the packed amax), the twelve
    scalars by delayed scaling from the per-sequence oracle's amaxes (ONE ``scale_dP`` / ``scale_dQ / dK / dV`` per packed batch),
    and the per-sequence reference gradients quantized at those global scales.

    ``cap_*`` over-allocates the packed buffers past the real totals; with ``poison`` the slack is NaN (e4m3 has a NaN code) --
    the declared totals only bound the buffers, so the rows between ``cu_*[B]`` and the capacity have to be kept out of reach by
    the kernels' OWN device-side descriptor clamps.  ``window_left`` is the graph's band bound = the number of keys a row keeps
    (the oracle's ``left_bound``; the adapter takes ``window_size_left = window_left - 1``).  ``quantize_ds`` selects the dS
    rounding the oracle composes (``True`` = the shipped e4m3-dS chain, ``False`` = the bf16-dS twin).  Unit-normal inputs on a
    CPU generator (the dataset does not depend on the GPU's SM count).  ``oracle=False`` builds the GEOMETRY only (unit scalars, no
    forward, no reference) -- for a host probe that lowers the ragged graph and never runs it."""
    from sdpa.fp8_ref import compute_ref, compute_ref_backward
    from sdpa.helpers import get_fp8_scale_factor

    dev, b = device, len(lens_q)
    lens_q, lens_kv = [int(x) for x in lens_q], [int(x) for x in lens_kv]
    assert len(lens_kv) == b
    hkv = h if hkv is None else hkv
    t_q, t_kv = sum(lens_q), sum(lens_kv)
    cap_q, cap_kv = cap_q or t_q, cap_kv or t_kv
    assert cap_q >= t_q and cap_kv >= t_kv
    cu_q, cu_k = _cu(lens_q), _cu(lens_kv)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(n, nh):
        return torch.randn(max(n, 1), nh, _D, generator=gen)[:n].to(dev)

    q32, do32, k32, v32 = draw(t_q, h), draw(t_q, h), draw(t_kv, hkv), draw(t_kv, hkv)
    q8, q_ds = _quant_packed(q32)
    k8, k_ds = _quant_packed(k32)
    v8, v_ds = _quant_packed(v32)
    do8, do_ds = _quant_packed(do32)

    def pack(live8, cap, nh):  # [1, cap, nh, D] e4m3: the live rows leading, the capacity tail NaN (poison) or zero
        stor = torch.full((1, cap, nh, _D), float("nan") if poison else 0.0, device=dev).to(_T_E4M3)
        stor[0, : live8.shape[0]] = live8
        return stor

    scale = 1.0 / math.sqrt(_D)
    s_scale = get_fp8_scale_factor(1.0, _T_E4M3)  # P <= 1
    s_descale = 1.0 / s_scale
    right = 0 if causal else None
    align = (cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT) if causal else None
    # The forward, per sequence: O in fp32 (quantized below ONCE at the packed amax -- the kernel reads one descale_o), Stats in
    # natural log.  A fully masked row comes out as O = 0, LSE = -inf (the forward's contract) from the oracle itself.
    o32 = torch.zeros(1, cap_q, h, _D, device=dev, dtype=torch.float32)
    lse = torch.full((1, h, cap_q), float("nan") if poison else float("-inf"), device=dev, dtype=torch.float32)
    o_amax = 0.0
    for i in range(b):
        sq_, sk_ = lens_q[i], lens_kv[i]
        if sq_ == 0 or sk_ == 0 or not oracle:
            if sq_:
                lse[0, :, cu_q[i] : cu_q[i] + sq_] = float("-inf") if sk_ == 0 else 0.0  # a sequence without keys: every row is dead
            continue
        slq, slk = slice(cu_q[i], cu_q[i] + sq_), slice(cu_k[i], cu_k[i] + sk_)
        o_i, st_i, amax_i = compute_ref(
            q8[slq][None], k8[slk][None], v8[slk][None], scale, q_ds, k_ds, v_ds, s_scale, s_descale, _T_E4M3, _T_E4M3,
            left_bound=window_left, right_bound=right, diag_align=align, quantize_o=False,
        )  # fmt: skip
        o32[0, slq] = o_i[0].float()
        lse[0, :, slq] = st_i[0, :, :, 0].float()
        o_amax = max(o_amax, float(amax_i))
    o_scale = get_fp8_scale_factor(o_amax, _T_E4M3)
    o_ds = 1.0 / o_scale
    o8 = pack((o32[0, :t_q] * o_scale).to(_T_E4M3), cap_q, h)
    q_p, k_p, v_p, do_p = pack(q8, cap_q, h), pack(k8, cap_kv, hkv), pack(v8, cap_kv, hkv), pack(do8, cap_q, h)

    def ref_bwd(i, *, dP_scale=None, return_intermediates=False, quantize_grads=True, quantize_ds_=None):
        """The fp8 oracle on sequence ``i`` alone ([1, S, H, D] slices of the packed codes, the sequence's own Stats)."""
        slq, slk = slice(cu_q[i], cu_q[i] + lens_q[i]), slice(cu_k[i], cu_k[i] + lens_kv[i])
        return compute_ref_backward(
            q8[slq][None], k8[slk][None], v8[slk][None], o8[0, slq][None], do8[slq][None], scale,
            q_ds, k_ds, v_ds, s_scale, s_descale, _T_E4M3, o_ds, do_ds, grad_dtype,
            left_bound=window_left, right_bound=right, diag_align=align, stats=lse[0, :, slq][None, :, :, None],
            return_intermediates=return_intermediates, quantize_ds=quantize_ds if quantize_ds_ is None else quantize_ds_,
            dP_scale=dP_scale, quantize_grads=quantize_grads,
        )  # fmt: skip

    live = [i for i in range(b) if lens_q[i] > 0 and lens_kv[i] > 0] if oracle else []
    # Pass 1: the packed dP amax -> ONE scale_dP for the whole batch (the kernel has one scalar; a per-sequence scale would be a
    # different rounding of dS for every sequence but one).
    dp_amax = max([ref_bwd(i)[4] for i in live], default=0.0)
    dp_scale = get_fp8_scale_factor(dp_amax, _T_E4M3)
    # Pass 2: the fp32 gradients per sequence at that dS scale, with the intermediates (``ds_scaled`` = the fp32 dS the contract's
    # amax_dP reduces, scaled) -- then the global output scales and the quantized references.
    grads32, amax32, ds_amax = {}, dict(dQ=0.0, dK=0.0, dV=0.0), 0.0
    for i in live:
        dq_i, dk_i, dv_i, _dsink, _dp, dq_a, dk_a, dv_a, inter = ref_bwd(i, dP_scale=dp_scale, return_intermediates=True, quantize_grads=False)
        grads32[i] = dict(dQ=dq_i[0], dK=dk_i[0], dV=dv_i[0])  # [S, H, D] fp32
        for name, a in (("dQ", dq_a), ("dK", dk_a), ("dV", dv_a)):
            amax32[name] = max(amax32[name], float(a))
        ds_amax = max(ds_amax, inter["ds_scaled"].abs().max().item() / dp_scale)
    grad_scale = {name: get_fp8_scale_factor(amax32[name], grad_dtype) for name in ("dQ", "dK", "dV")}  # 1.0 for bf16 / fp16
    refs = {i: {name: (g32 * grad_scale[name]).to(grad_dtype) for name, g32 in grads32[i].items()} for i in live}
    scalars = dict(
        descale_q=q_ds,
        descale_k=k_ds,
        descale_v=v_ds,
        descale_o=o_ds,
        descale_dO=do_ds,
        descale_s=s_descale,
        descale_dP=1.0 / dp_scale,
        scale_s=s_scale,
        scale_dQ=grad_scale["dQ"],
        scale_dK=grad_scale["dK"],
        scale_dV=grad_scale["dV"],
        scale_dP=dp_scale,
    )
    deq = dict(q=q8.float() * q_ds, k=k8.float() * k_ds, dO=do8.float() * do_ds)
    return SimpleNamespace(
        b=b, h=h, hkv=hkv, d=_D, grad_dtype=grad_dtype, dtype=grad_dtype, scale=scale, causal=causal, bottom_right=bottom_right, window_left=window_left,
        lens_q=lens_q, lens_kv=lens_kv, cu_q=cu_q, cu_k=cu_k, t_q=t_q, t_kv=t_kv, cap_q=cap_q, cap_kv=cap_kv, live=live,
        q=q_p, k=k_p, v=v_p, o=o8, do=do_p, lse=lse, scalars=scalars, dp_scale=dp_scale, s_descale=s_descale, grad_scale=grad_scale,
        descales={name: 1.0 / s for name, s in grad_scale.items()}, refs=refs, ref_amax=dict(dQ=amax32["dQ"], dK=amax32["dK"], dV=amax32["dV"], dP=ds_amax),
        deq=deq, ref_bwd=ref_bwd, quantize_ds=quantize_ds,
    )  # fmt: skip


def _finite(t):
    return bool(torch.isfinite(t.float()).all())


def _check_fp8(case, dq, dk, dv, amax=None):
    """Per-sequence comparison under the fp8 row's recipe, COLLECTED and asserted at the end (which gradient of which sequence is
    wrong is the attribution: dK and dQ come from the two stage-3 renderings, dV from the main kernel), then the four amax outputs
    against the maximum over the packed LIVE region of the oracle's pre-quant tensors.  A sequence empty on either side has no
    reference and is checked by its caller."""
    from sdpa.fp8 import assert_close_fp8_grad

    bad, verdicts = [], []
    for i in case.live:
        slq = slice(case.cu_q[i], case.cu_q[i] + case.lens_q[i])
        slk = slice(case.cu_k[i], case.cu_k[i] + case.lens_kv[i])
        for name, got, operand, keys, flip in (
            ("dQ", dq[0, slq], case.deq["k"][slk], case.lens_kv[i], 1.0 / case.dp_scale),
            ("dK", dk[0, slk], case.deq["q"][slq], case.lens_q[i], 1.0 / case.dp_scale),
            ("dV", dv[0, slk], case.deq["dO"][slq], case.lens_q[i], case.s_descale),
        ):
            tag = f"seq {i} {name} (lens q={case.lens_q[i]} kv={case.lens_kv[i]})"
            want = case.refs[i][name]
            g_, w_ = got[None].float() * case.descales[name], want[None].float() * case.descales[name]  # [1, S, H, D] true units
            if not _finite(g_):
                bad.append(f"{tag}: non-finite output")
                verdicts.append(bad[-1])
                continue
            try:
                assert_close_fp8_grad(
                    g_,
                    w_,
                    _FP8_GRAD_TOL["atol"],
                    _FP8_GRAD_TOL["rtol"],
                    tag=tag,
                    kind=name,
                    keys=keys,
                    operand=operand[None],
                    flip_unit=flip,
                    intermediates=lambda selection, i=i: case.ref_bwd(i, dP_scale=case.dp_scale, return_intermediates=selection, quantize_grads=False)[8],
                    fp8_dtype=_T_E4M3,
                    out_dtype=case.grad_dtype,
                )
                verdicts.append(f"{tag}: ok")
            except AssertionError as exc:
                bad.append(f"{tag} vs the fp8 oracle: {str(exc).splitlines()[0]}")
                verdicts.append(bad[-1])
    assert not bad, "\n".join(verdicts)
    if amax is not None:
        _check_amax(case, amax)


def _check_amax(case, amax):
    """Every requested amax: finite, and the maximum over the packed LIVE region (the oracle's pre-quant tensors) within the
    recipe -- ``_FP8_GRAD_TOL`` on dQ / dK / dV, ``_AMAX_DS_TOL`` on dP (the fp32 ``dS`` before its ``scale_dP`` cast)."""
    for name in ("dQ", "dK", "dV", "dP"):
        if name not in amax:
            continue
        a, r = float(amax[name]), case.ref_amax[name]
        assert math.isfinite(a), f"amax_{name} = {a}: not written, or a dead unit / pad row / NaN capacity tail reached the fold"
        tol = _AMAX_DS_TOL if name == "dP" else _FP8_GRAD_TOL
        assert (
            abs(a - r) <= tol["atol"] + tol["rtol"] * r
        ), f"amax_{name} {a:.6f} vs the live-region oracle max {r:.6f} (sdpa-invariants s5: the LIVE region only)"


def _check_fp8_rows_with_keys(case, dq, dk, dv, amax=None):
    """The oracle for a case with FULLY MASKED q rows (bottom-right with ``s_q > s_kv``, a top-left window past the keys): the rows
    without a key get ``dQ = 0`` EXACTLY -- asserted on the output directly (a diff against a reference proves nothing there) --
    and contribute nothing to dK / dV; the whole sequence then passes the per-sequence oracle, which yields the same zeros on
    those rows (``LSE = -inf`` -> ``P = 0``)."""
    for i in case.live:
        r0, r1 = _rows_with_keys(case, i)
        q0, k0 = case.cu_q[i], case.cu_k[i]
        for lo, hi in ((0, r0), (r1, case.lens_q[i])):
            dead = dq[0, q0 + lo : q0 + hi].float()
            if dead.numel():
                assert (
                    _finite(dead) and not dead.any()
                ), f"seq {i}: dQ rows [{lo}, {hi}) have no key and must be exactly zero (max |.| {dead.abs().max().item():.3e})"
        if r1 <= r0:
            for name, got in (("dK", dk[0, k0 : k0 + case.lens_kv[i]]), ("dV", dv[0, k0 : k0 + case.lens_kv[i]])):
                assert _finite(got) and not got.float().any(), f"seq {i}: no row has a key, {name} must be exactly zero"
    _check_fp8(case, dq, dk, dv, amax)


def _bitwise(name, a, b):
    """Byte-exact equality (e4m3, bf16 and fp16 alike): a difference is a race or a changed rounding, never noise."""
    a8, b8 = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    n_diff = int((a8 != b8).sum())
    assert n_diff == 0, f"{name}: {n_diff} of {a8.numel()} bytes differ (max|diff|={(a.float() - b.float()).abs().max().item():.3e})"


# --------------------------------------------------------------------------- the direct adapter surface


def _envelope_samples(case, envelope_q=None, envelope_kv=None):
    """The adapter's SAMPLES declare the ENVELOPE (B, H, S_max, D) the way a ragged graph does; the packed buffers only show up at
    execute.  e4m3 payloads, the gradient dtype on dQ / dK / dV, contiguous fp32 Stats.  ``envelope_q`` / ``envelope_kv`` name the
    envelopes when the longest sequence would not (a batch with no query anywhere would declare S_q = 1, a decode shape no prefill
    row serves; a capacity-padded packing needs ``B * S_max >= capacity``: the adapter tightens the declared total to ``B * S_max``,
    and the standalone surface holds the packed buffers to the plan's exact capacity)."""
    dev, b, h, hkv = "cuda", case.b, case.h, case.hkv
    s_max_q, s_max_kv = envelope_q or max(max(case.lens_q), 1), envelope_kv or max(max(case.lens_kv), 1)

    def env(n, s, nh, dt):
        return torch.empty(1, n, s, nh, _D, device=dev, dtype=dt)[0].permute(0, 2, 1, 3)

    eq, ekv = env(b, s_max_q, h, _T_E4M3), env(b, s_max_kv, hkv, _T_E4M3)
    gq, gkv = env(b, s_max_q, h, case.grad_dtype), env(b, s_max_kv, hkv, case.grad_dtype)
    e_stats = torch.empty(b, h, s_max_q, 1, device=dev, dtype=torch.float32)
    return eq, ekv, gq, gkv, e_stats


def _scalar_tensors(scalars):
    return {name: torch.tensor([float(v)], dtype=torch.float32, device="cuda") for name, v in scalars.items()}


def _build_direct_api(case, *, token_major_stats=False, request_amax=_AMAX, envelope_q=None):
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    # The packed buffers are allocated at ``cap_q`` / ``cap_kv`` tokens (the live totals plus a capacity tail); the envelope must
    # cover them (``B * S_max >= cap``) or the adapter tightens the plan's capacity below the buffers and the standalone surface
    # refuses the runtime geometry.
    env_q = max(envelope_q or max(max(case.lens_q), 1), -(-case.cap_q // case.b))
    env_kv = max(max(max(case.lens_kv), 1), -(-case.cap_kv // case.b))
    eq, ekv, gq, gkv, e_stats = _envelope_samples(case, env_q, env_kv)
    api = SdpaBwdDslSm107Fp8(
        sample_q=eq,
        sample_k=ekv,
        sample_v=ekv,
        sample_o=eq,
        sample_do=eq,
        sample_stats=e_stats,
        sample_dq=gq,
        sample_dk=gkv,
        sample_dv=gkv,
        scale_softmax=case.scale,
        is_causal=case.causal,
        causal_bottom_right=case.bottom_right,
        window_size_left=None if case.window_left is None else case.window_left - 1,
        thd=True,
        max_total_seq_len_q=case.cap_q,
        max_total_seq_len_kv=case.cap_kv,
        thd_stats_token_major=token_major_stats,
        # The declared totals are a MAXIMUM the adapter tightens to the envelope's B * S_max tokens (``_thd_total``), so a
        # head-major Stats buffer allocated at the capacity names its own head stride, as the graph path derives it from the Stats
        # port's stride; token-major (T, H) Stats is compact and takes no stride (``_run_fp8_direct`` slices it to the plan's cap).
        thd_stats_head_stride=None if token_major_stats else case.cap_q,
        amax_requested=tuple(request_amax),
    )
    assert api.check_support()
    return api


def _run_fp8_direct(
    lens_q,
    lens_kv,
    *,
    h=2,
    hkv=None,
    token_major_stats=False,
    causal=False,
    bottom_right=False,
    window_left=None,
    poison=False,
    pad_cap=0,
    poison_outputs=False,
    grad_dtype=_T_E4M3,
    request_amax=_AMAX,
    runs=1,
    ws_fill=0xFF,
    check=None,
    seed=7,
    envelope_q=None,
):
    """Build the case, drive ``SdpaBwdDslSm107Fp8(thd=True)`` directly on PACKED views over a 0xFF-poisoned workspace (NaN in every
    dtype the chain stores: a stage reading a scratch region before writing it surfaces as NaN), the gradient tails past the packed
    totals under the finite sentinel, the amax outputs NaN-filled per run; compare per sequence under the fp8 recipe.  The dS
    rounding the oracle composes follows the ``ds_knob`` fixture (``api_dsl_sm107.FP8_DS_DTYPE`` at construction).  Returns the
    run (case, gradients, amax values, every run's outputs, the api and its arguments) for the caller's extra assertions."""
    dev = "cuda"
    case = _thd_fp8_case(
        lens_q, lens_kv, h, hkv, cap_q=sum(lens_q) + pad_cap, cap_kv=sum(lens_kv) + pad_cap, poison=poison, seed=seed, causal=causal,
        bottom_right=bottom_right, window_left=window_left, grad_dtype=grad_dtype, quantize_ds=(_ds_dtype_code() == DTYPE_E4M3),
    )  # fmt: skip
    api = _build_direct_api(case, token_major_stats=token_major_stats, request_amax=request_amax, envelope_q=envelope_q)
    view = lambda t: t.permute(0, 2, 1, 3)  # noqa: E731  [1,T,H,D] -> logical [1,H,T,D], the dense path's orientation
    fill = float("nan") if poison_outputs else 0.0
    dq = torch.empty(1, case.cap_q, case.h, _D, device=dev, dtype=grad_dtype)
    dk = torch.empty(1, case.cap_kv, case.hkv, _D, device=dev, dtype=grad_dtype)
    dv = torch.empty(1, case.cap_kv, case.hkv, _D, device=dev, dtype=grad_dtype)
    # TRANSPOSED, not reshaped: lse is head-major [1, H, T]; a reshape to (T, H) would reinterpret the memory.  The compact
    # token-major form holds exactly the plan's packed capacity of rows (a leading-dim slice stays contiguous).
    stats = case.lse[0].transpose(0, 1).contiguous()[: api._t_q_cap] if token_major_stats else case.lse
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), dtype=torch.uint8, device=dev)
    scalars = _scalar_tensors(case.scalars)
    amax_t = {name: torch.full((1,), float("nan"), device=dev, dtype=torch.float32) for name in request_amax}
    lq = torch.tensor(case.lens_q, dtype=torch.int32, device=dev)
    lk = torch.tensor(case.lens_kv, dtype=torch.int32, device=dev)
    tensors = (view(case.q), view(case.k), view(case.v), view(case.o), view(case.do), stats, view(dq), view(dk), view(dv))
    kwargs = dict(workspace=ws, seq_q_lens=lq, seq_kv_lens=lk, **scalars, **{name: amax_t.get(name) for name in _AMAX})
    outs, amaxes = [], []
    for _ in range(runs):
        for x in (dq, dk, dv):
            x.fill_(fill)
        _sentinel_tails(case, dq, dk, dv)
        for a in amax_t.values():
            a.fill_(float("nan"))
        ws.fill_(ws_fill)
        api.execute(*tensors, **kwargs)
        torch.cuda.synchronize()
        outs.append(tuple(x.clone() for x in (dq, dk, dv)))
        amaxes.append({name: a.item() for name, a in amax_t.items()})
    for name, x, live in (("dQ", dq, case.t_q), ("dK", dk, case.t_kv), ("dV", dv, case.t_kv)):
        assert _finite(x[0, :live]), f"{name} has non-finite values in the packed region"
    _assert_tails_untouched(case, dq, dk, dv)
    amax = {name[len("amax_") :]: v for name, v in amaxes[-1].items()}
    (check or _check_fp8)(case, dq, dk, dv, amax)
    return SimpleNamespace(case=case, dq=dq, dk=dk, dv=dv, amax=amax, amaxes=amaxes, outs=outs, api=api, tensors=tensors, kwargs=kwargs, ws=ws, amax_t=amax_t)


@requires_rubin
def test_thd_fp8_self_attention(ds_knob):
    """Three sequences of unequal length, none a tile multiple -- on both dS workspace dtypes of the row."""
    _run_fp8_direct((300, 128, 200), (300, 128, 200))


@requires_rubin
def test_thd_fp8_cross_attention(ds_knob):
    """Unequal Q and KV lengths, and unequal packed totals with them."""
    _run_fp8_direct((256, 100), (180, 300))


@requires_rubin
def test_thd_fp8_single_sequence_matches_dense_shape():
    """B == 1 is the degenerate packing: it must agree with the dense answer."""
    _run_fp8_direct((512,), (512,))


@requires_rubin
def test_thd_fp8_stats_token_major():
    """The other packed Stats layout the forward can emit."""
    _run_fp8_direct((300, 128), (300, 128), token_major_stats=True)


@requires_rubin
@pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_thd_fp8_half_gradients(grad_dtype):
    """bf16 / fp16 gradients on the packed fp8 chain (``_FP8_GRAD_DTYPES``): the stage-3 QUANT epilogue stores the half dtype with
    ``scale_* = 1``; one fp16 cell, the bf16 one on the dense THD shape."""
    _run_fp8_direct((300, 128, 200), (300, 128, 200), grad_dtype=grad_dtype)


@requires_rubin
def test_thd_fp8_dead_units_run_one_masked_tile():
    """FEWER live units than clusters (one 256-row kv block of one head on an occupancy-sized grid): every other cluster's first
    unit is past the device live total and runs ONE forced fully-masked tile.  Poisoned outputs and workspace: a dead unit that
    stored anything into a live row, folded anything into an amax, or a live unit that skipped a store, surfaces as NaN or a
    wrong sequence."""
    _run_fp8_direct((256,), (256,), h=1, poison_outputs=True)


@requires_rubin
def test_thd_fp8_zero_length_sequence():
    """A sequence with no tokens on either side must not corrupt its neighbours; its own gradients are exact zeros."""
    run = _run_fp8_direct((256, 0, 128), (256, 0, 128), poison_outputs=True)
    _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, 1)


@requires_rubin
@pytest.mark.parametrize("lens_q,lens_kv", (((0, 128, 256), (192, 128, 256)), ((192, 128, 256), (0, 128, 256))), ids=("empty_q_side", "empty_kv_side"))
def test_thd_fp8_one_sided_empty_sequence(lens_q, lens_kv):
    """A sequence empty on ONE side only.  Empty Q: its kv blocks still get units (one forced fully-masked q tile each -> dS = 0
    into their own rows, dV = 0 stored to the sequence's rows), the dK GEMM's reduction is empty and must store zeros by SELECT
    (not residue, and its EPI_QUANT fold must not move amax_dK), dQ has no rows.  Empty KV: no unit at all, the dQ GEMM's
    reduction is empty -> zeros by select, dK / dV have no rows.  Zero is the answer, not a convention; the poisoned outputs make
    an unwritten live row NaN."""
    run = _run_fp8_direct(lens_q, lens_kv, poison_outputs=True)
    _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, 0)


@requires_rubin
def test_thd_fp8_nan_capacity_tail():
    """Declared totals larger than the live packing, with a NaN tail in every e4m3 payload, O and Stats: the kernels' device-side
    descriptor clamps keep the rows between ``cu_*[B]`` and the capacity out of every MMA -- and out of every amax fold."""
    _run_fp8_direct((256, 128), (256, 128), poison=True, pad_cap=384)


@requires_rubin
def test_thd_fp8_nan_capacity_tail_unaligned_last_sequence():
    """The capacity tail reached by a K-TILE OVERSHOOT (a 100-token last sequence: the fp8 K64 arm's 128-element e4m3 K tile reads
    28 rows past the packed total) on both GEMM orientations, and by the main kernel's last kv block / q tile."""
    _run_fp8_direct((256, 100), (256, 100), poison=True, pad_cap=384)


@requires_rubin
@pytest.mark.parametrize("hkv", (2, 1), ids=("gqa_group2", "mqa"))
def test_thd_fp8_gqa(hkv):
    """Packed GQA / MQA: bf16 dK / dV partials ONE PER Q HEAD over the packed kv axis (``EPI_DESCALE`` on dK), folded and quantized
    onto the KV heads by the bounded ``fold_quant``; the reference SUMS a group's contributions."""
    _run_fp8_direct((300, 128, 200), (300, 128, 200), h=4, hkv=hkv)


@requires_rubin
def test_thd_fp8_gqa_causal(ds_knob):
    _run_fp8_direct((256, 100), (256, 100), h=4, hkv=2, causal=True)


@requires_rubin
def test_thd_fp8_gqa_cross_attention_and_zero_length():
    """GQA over unequal Q / KV totals with an empty sequence in the middle: the dK / dV partials are sized on the packed KV
    capacity while dQ rides the Q one."""
    run = _run_fp8_direct((256, 0, 100), (180, 0, 300), h=4, hkv=2, poison_outputs=True)
    _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, 1)


@requires_rubin
def test_thd_fp8_gqa_capacity_tail_untouched():
    """GQA with declared totals past the live packing and a NaN tail on BOTH sides.  The per-Q-head partials past ``cu_k[B]`` are
    never written, so the fold + quantize pass must stop at the live kv total ON DEVICE (``fold_quant`` with its row limit): an
    unbounded fold reads the 0xFF-poisoned workspace there, writes NaN into the caller's dK / dV capacity tail -- which only the
    FINITE tail sentinel can see -- and folds NaN into ``amax_dV`` / ``amax_dK``."""
    _run_fp8_direct((300, 128, 200), (300, 128, 200), h=4, hkv=2, poison=True, pad_cap=384, poison_outputs=True)


@requires_rubin
def test_thd_fp8_every_sequence_empty_q_with_nan_in_dO_row_0():
    """No query anywhere (``cu_q[B] = 0``), keys in every sequence, the Q / dO capacity all NaN.  The clamped Q / dO descriptors keep
    an extent of 1 (0 is invalid), so a forced tile that addressed ``cu_q[b] = 0`` would load the NaN row and ``dV = P^T . dO``
    would be ``0 * NaN`` on every live kv row.  The kernel routes every load of a unit without query rows past the clamped extent
    instead (zero-filled): dK and dV exact zeros, every amax 0, nothing past the packed totals written."""
    run = _run_fp8_direct((0, 0), (128, 256), poison=True, pad_cap=128, poison_outputs=True, envelope_q=128)
    for i in range(run.case.b):
        _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, i)
    for name, value in run.amax.items():
        assert value == 0.0, f"amax_{name} = {value}: a batch without a single live cell must fold nothing"


@requires_rubin
@pytest.mark.parametrize("lens", [(200, 512, 700, 1024, 2048)], ids=["kv-blocks-1-2-3-4-8"])
def test_thd_fp8_kv_blocks_per_sequence(lens):
    """kv blocks per sequence 1, 2, 3, 4 and 8 in ONE packed batch (sdpa-invariants s9: the parity reuse and the ring wrap of the
    per-sequence block walk), q tiles per block from 1 to 4 across them."""
    _run_fp8_direct(lens, lens)


@requires_rubin
def test_thd_fp8_q_tiles_per_kv_block():
    """q tiles per kv block = 1, 2, 3 (``STAGES_Q``) and 4 in one packed batch against one kv block each: the three 3-deep Q / dO
    rings and the 2-stage P ring wrap at different depths per sequence."""
    _run_fp8_direct((128, 256, 384, 512), (256, 256, 256, 256), h=1)


# --- causal family: the per-sequence diagonal is the whole risk, and stage 3 trims its K range per sequence over a workspace whose
# masked tiles the kernel never wrote (0xFF-poisoned here, never zero-filled: a read past the band lands NaN in dQ / dK).  Unequal
# lengths, none a tile multiple: with equal lengths a diagonal from the envelope would agree with the per-sequence one.


@requires_rubin
def test_thd_fp8_causal(ds_knob):
    _run_fp8_direct((300, 128, 200), (300, 128, 200), causal=True)


@requires_rubin
def test_thd_fp8_causal_zero_length_sequence():
    """Causal plus an empty sequence: the masked tiles the kernel never wrote and the zero-length sequence's missing unit /
    extent-1 descriptor are independent mechanisms, both live at once."""
    run = _run_fp8_direct((256, 0, 128), (256, 0, 128), causal=True, poison_outputs=True)
    _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, 1)


@requires_rubin
def test_thd_fp8_causal_cross_attention():
    """Top-left aligned, so the diagonal does not move -- but S_kv[b] != S_q[b] makes the live kv blocks per q row differ per
    sequence, which a trim keyed on absolute workspace rows would get wrong."""
    _run_fp8_direct((256, 100), (180, 300), causal=True)


@requires_rubin
def test_thd_fp8_causal_bottom_right():
    """Bottom-right alignment: the diagonal offset IS per sequence (56 and 200 here), read from the metadata per sequence by the
    kernel (``_causal_diag(eff_q, eff_kv)`` per tile) and by stage 3's trim (``thd_causal_bottom_right``)."""
    _run_fp8_direct((200, 100), (256, 300), causal=True, bottom_right=True)


@requires_rubin
def test_thd_fp8_causal_bottom_right_ragged_s_q(ds_knob):
    """Bottom-right at a RAGGED S_q per sequence (200 and 300: neither a q-tile multiple): the fp8 body now threads the real q
    length (``seqlen_q_real`` / the per-sequence ``s_q[b]``) into its diagonal and its q-tile trim -- what the dense row's
    ``bottom_right_s_q_multiple = 128`` decline used to stand in for.  The last q tile's pad columns are select-masked by the THD q
    band."""
    _run_fp8_direct((200, 300), (456, 300), causal=True, bottom_right=True)


@requires_rubin
def test_thd_fp8_causal_swa():
    """Sliding window: a LEFT bound on top of the causal right one -- the band's second per-sequence edge, through the kernel's
    q-tile trim from above and stage 3's per-sequence window edge, so the window-skipped tiles are never read."""
    _run_fp8_direct((300, 128, 200), (300, 128, 200), causal=True, window_left=64)


@requires_rubin
def test_thd_fp8_causal_nan_capacity_tail():
    """Causal with a NaN tail past the declared totals: a different set of workspace rows than the dense tail case, read through
    the per-sequence trim with no zero-fill in between."""
    _run_fp8_direct((256, 128), (256, 128), poison=True, pad_cap=384, causal=True)


@requires_rubin
@pytest.mark.parametrize(
    "lens_q,lens_kv,kw",
    [
        ((0, 128, 256), (192, 128, 256), dict(causal=True, bottom_right=True)),
        ((192, 128, 256), (0, 128, 256), dict(causal=True, window_left=64)),
        ((256, 0, 128), (256, 0, 128), dict(causal=True, bottom_right=True, window_left=64)),
        ((300, 5, 200), (300, 5, 200), dict(causal=True)),
        ((300, 65, 200), (300, 129, 200), dict(causal=True, window_left=2)),  # s_q <= s_kv: every row keeps a key
    ],
    ids=["br-empty-q-side", "swa-empty-kv-side", "br-swa-zero-length", "five-row-sequence", "window-of-two-keys-odd-tiles"],
)
def test_thd_fp8_trimmed_degenerate_sequences(lens_q, lens_kv, kw):
    """The degenerate-input matrix under the trimmed stage 3 (sdpa-invariants): an empty q side (dK / dV rows exist, the reduction
    is empty -> exact zeros through the select, no amax contribution), an empty kv side (no unit, no rows), a zero-length sequence
    inside the batch, a FIVE-row sequence (one q tile, one kv block -- never a one-row one), the narrowest served window (two
    keys) on odd q-tile counts -- all over a poisoned workspace and poisoned outputs, every live gradient held to the oracle."""
    run = _run_fp8_direct(lens_q, lens_kv, poison_outputs=True, **kw)
    for i in range(run.case.b):
        if run.case.lens_q[i] == 0 or run.case.lens_kv[i] == 0:
            _assert_empty_sequence_exactly_zero(run.case, run.dq, run.dk, run.dv, i)


@requires_rubin
@pytest.mark.parametrize("case", list(_NO_KEY_ROW_CASES), ids=list(_NO_KEY_ROW_CASES))
def test_thd_fp8_rows_without_a_key_are_exactly_zero_and_fold_no_amax(case, monkeypatch):
    """Per-sequence geometries with FULLY MASKED q rows -- the trailing rows of a top-left window past ``s_kv + W`` and the leading
    rows of a bottom-right sequence with ``s_q > s_kv``.  Their dQ is exactly zero: the trim hands those M tiles an EMPTY K range
    (the select-zero store), and -- the fp8 arm's own hazard -- the EPI_QUANT epilogue must not fold the tile's TMEM residue into
    ``amax_dQ`` (sdpa-invariants s5: the fold is gated per row on the sequence's live rows).  The untrimmed, zero-filled twin
    (``STAGE3_CAUSAL_TRIM = False``) must agree bitwise, amax values included."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = {k: v for k, v in _NO_KEY_ROW_CASES[case].items() if k not in ("lens_q", "lens_kv")}
    lens_q, lens_kv = _NO_KEY_ROW_CASES[case]["lens_q"], _NO_KEY_ROW_CASES[case]["lens_kv"]
    run_kw = dict(
        h=kw.get("h", 2), hkv=kw.get("hkv"), causal=bool(kw.get("use_causal_mask") or kw.get("use_causal_mask_bottom_right")),
        bottom_right=bool(kw.get("use_causal_mask_bottom_right")), window_left=kw.get("sliding_window_length"), poison_outputs=True, check=_check_fp8_rows_with_keys,
    )  # fmt: skip
    trimmed = _run_fp8_direct(lens_q, lens_kv, **run_kw)
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", False)
    untrimmed = _run_fp8_direct(lens_q, lens_kv, **run_kw)
    for name, a, b in zip(("dQ", "dK", "dV"), (trimmed.dq, trimmed.dk, trimmed.dv), (untrimmed.dq, untrimmed.dk, untrimmed.dv)):
        _bitwise(f"{name} (trimmed vs untrimmed, rows without a key)", a, b)
    assert trimmed.amax == untrimmed.amax, f"amax: trimmed {trimmed.amax} vs untrimmed {untrimmed.amax}"


def _graph_kw_to_direct(kw):
    """The sibling module's case tables spell the GRAPH's flags; the direct surface takes the adapter's."""
    return dict(
        h=kw.get("h", 2),
        hkv=kw.get("hkv"),
        causal=bool(kw.get("use_causal_mask") or kw.get("use_causal_mask_bottom_right")),
        bottom_right=bool(kw.get("use_causal_mask_bottom_right")),
        window_left=kw.get("sliding_window_length"),
        poison_outputs=bool(kw.get("poison_outputs", False)),
    )


@requires_rubin
@pytest.mark.parametrize("case", list(_TRIM_TWIN_CASES), ids=list(_TRIM_TWIN_CASES))
def test_thd_fp8_trimmed_stage3_is_bitwise_the_untrimmed_rendering_over_a_poisoned_workspace(case, monkeypatch, ds_knob):
    """The per-sequence K-trim on the fp8 K64 arm reads ONLY dS tiles the main kernel wrote: the workspace is 0xFF-poisoned (NaN in
    e4m3 and bf16) and the shipped chain runs WITHOUT the zero-fill, so a GEMM reaching a skipped tile lands NaN in dQ / dK.  The
    untrimmed twin (``STAGE3_CAUSAL_TRIM = False``: ``CAUSAL_K_NONE`` on both GEMMs, the fill back on) walks every k tile of the
    zero-filled workspace -- the skipped tiles are exact zeros, so the two accumulate the same values in the same order: the
    gradients must be the SAME BITS and the four amax values EQUAL.  The amax equality is what catches an EPI_QUANT epilogue that
    folds a tile with an EMPTY K range (residue) or a tile past the sequence's rows: both renderings would still agree on the
    gradients (the store select), only the amax would move."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = dict(_TRIM_TWIN_CASES[case])
    lens_q, lens_kv = kw.pop("lens_q"), kw.pop("lens_kv")
    assert sm107.STAGE3_CAUSAL_TRIM, "the per-sequence trim is what ships; the pin flips it OFF for the twin"
    trimmed = _run_fp8_direct(lens_q, lens_kv, **_graph_kw_to_direct(kw))
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", False)
    untrimmed = _run_fp8_direct(lens_q, lens_kv, **_graph_kw_to_direct(kw))
    for name, a, b in zip(("dQ", "dK", "dV"), (trimmed.dq, trimmed.dk, trimmed.dv), (untrimmed.dq, untrimmed.dk, untrimmed.dv)):
        _bitwise(f"{name} (trimmed stage 3, no fill vs untrimmed, zero-filled)", a, b)
    assert trimmed.amax == untrimmed.amax, f"amax: trimmed {trimmed.amax} vs untrimmed {untrimmed.amax} -- a fold reached a tile the trim skips"


@requires_rubin
@pytest.mark.parametrize("case", list(_GQA_TWIN_CASES), ids=list(_GQA_TWIN_CASES))
def test_thd_fp8_single_launch_dq_is_bitwise_the_per_member_launches(case, monkeypatch, ds_knob):
    """Under GQA the THD dQ GEMM is ONE launch per head chunk -- its rendering indexes the packed B = K by ``h // group`` over the
    whole dS and dQ chunk -- where it used to be one launch per group MEMBER.  On the e4m3 K64 arm the QUANT epilogue folds
    ``amax_dQ`` with per-TENSOR scalars (descale_dP, descale_k, scale_dQ), so the single launch reads the same scalars the
    per-member launches did: dQ the SAME BITS (and dK / dV, untouched), the four amax values equal.  ``DQ_SINGLE_LAUNCH = False``
    is the per-member twin.  Covers GQA 32/2 and 64/8 at small S, dense / causal / bottom-right / a window, an empty side."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = dict(_GQA_TWIN_CASES[case])
    lens_q, lens_kv = kw.pop("lens_q"), kw.pop("lens_kv")
    assert sm107.DQ_SINGLE_LAUNCH, "one dQ launch per chunk is what ships; the pin flips it OFF for the twin"
    single = _run_fp8_direct(lens_q, lens_kv, **_graph_kw_to_direct(kw))
    monkeypatch.setattr(sm107, "DQ_SINGLE_LAUNCH", False)
    members = _run_fp8_direct(lens_q, lens_kv, **_graph_kw_to_direct(kw))
    for name, a, b in zip(("dQ", "dK", "dV"), (single.dq, single.dk, single.dv), (members.dq, members.dk, members.dv)):
        _bitwise(f"{name} (single dQ launch vs per-member launches)", a, b)
    assert single.amax == members.amax, f"amax: single {single.amax} vs per-member {members.amax}"


@requires_rubin
def test_thd_fp8_two_launches_are_bitwise_and_race_free():
    """Launch 2 vs 1 = the two-launch race trick, launch 3 vs 2 = the determinism to show before claiming it; the amax outputs
    (an atomicMax over a fixed set of fp32 values) must be bitwise stable too."""
    run = _run_fp8_direct((300, 128, 200), (300, 128, 200), h=4, hkv=2, causal=True, runs=3, poison_outputs=True)
    for which, i, j in (("launch 2 vs 1 (race)", 1, 0), ("launch 3 vs 2 (determinism)", 2, 1)):
        for name, a, b in zip(("dQ", "dK", "dV"), run.outs[i], run.outs[j]):
            _bitwise(f"{name} {which}", a, b)
        assert run.amaxes[i] == run.amaxes[j], f"amax {which}: {run.amaxes[i]} vs {run.amaxes[j]}"


@requires_rubin
def test_thd_fp8_ds_workspace_dtypes_agree(monkeypatch):
    """The two chains side by side on one packed input: dV does not depend on dS and amax_dP is folded BEFORE the scale and the
    cast, so both are BITWISE across the knob (a difference is a kernel change, not a GEMM-arm change); amax_dV likewise.  dQ /
    dK each pass the recipe against their OWN oracle inside ``_run_fp8_direct``."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107

    runs = {}
    for knob in (DTYPE_E4M3, DTYPE_BF16):
        with monkeypatch.context() as patch:
            patch.setattr(sm107, "FP8_DS_DTYPE", knob)
            runs[knob] = _run_fp8_direct((300, 128, 200), (300, 128, 200), h=4, hkv=2, causal=True)
    e4m3, bf16 = runs[DTYPE_E4M3], runs[DTYPE_BF16]
    _bitwise("dV across the dS workspace dtype", e4m3.dv, bf16.dv)
    assert e4m3.amax["dP"] == bf16.amax["dP"], "amax_dP is the pre-quant fp32 max on both chains"
    assert e4m3.amax["dV"] == bf16.amax["dV"]


@requires_rubin
def test_thd_fp8_amax_is_the_live_region_max(ds_knob):
    """The amax contract under THD (sdpa-invariants s5), on the shape that exposes every one of its FOUR sites: dense THD (no mask,
    so every q tile of every kv block is live and the stage-3 renderings run ``CAUSAL_K_NONE`` -- the arm whose dQ groups walk
    EVERY M tile of the envelope, the tiles past a shorter sequence's own rows included), two sequences of DIFFERENT lengths (300
    and 128: the shorter one's envelope tiles past its rows hold products of unwritten / foreign blocked rows), a 0xFF-poisoned
    workspace and a NaN capacity tail.  ``amax_dQ`` (the stage-3 EPI_QUANT fold), ``amax_dK`` (the same fold at MHA), ``amax_dV``
    (the bounded ``fold_quant`` pass) and ``amax_dP`` (the kernel's row-gated fold) must each equal the maximum over the packed LIVE
    region of the oracle's pre-quant tensors within the recipe -- on both dS knobs (the bf16-dS twin folds all three through
    ``fold_quant``)."""
    run = _run_fp8_direct((300, 128), (300, 128), poison=True, pad_cap=384, poison_outputs=True)
    assert set(run.amax) == {"dQ", "dK", "dV", "dP"}
    _check_amax(run.case, run.amax)  # the assertion _check_fp8 already made, spelled as the test's own verdict
    # and no amax exceeds the maximum of the DEQUANTIZED live output by more than the output's own rounding (the fold is over the
    # same values the store quantizes)
    for name, got in (("dQ", run.dq[0, : run.case.t_q]), ("dK", run.dk[0, : run.case.t_kv]), ("dV", run.dv[0, : run.case.t_kv])):
        live_max = (got.float() * run.case.descales[name]).abs().max().item()
        assert run.amax[name] >= live_max * (1.0 - 2.0**-3) - 1e-6, f"amax_{name} {run.amax[name]:.6f} is below the live output's own max {live_max:.6f}"


@requires_rubin
@pytest.mark.parametrize("request_amax", [(), ("amax_dP",), ("amax_dQ", "amax_dK", "amax_dV")], ids=["none", "dP", "dQ-dK-dV"])
def test_thd_fp8_binds_only_the_requested_amax(request_amax):
    """An amax the plan did not request is None-specialized out of the THD artifact: the gradients are BITWISE the all-requested
    plan's and every requested amax equals its value there."""
    full = _run_fp8_direct((300, 128, 200), (300, 128, 200), h=4, hkv=2, causal=True)
    part = _run_fp8_direct((300, 128, 200), (300, 128, 200), h=4, hkv=2, causal=True, request_amax=request_amax)
    spec = part.api._prepared
    bound = {role for role, op in zip(spec.roles, spec.operands) if role.startswith("amax_") and op is not None}
    assert bound == set(request_amax), f"the artifact binds the amax operands {sorted(bound)}; the plan requested {sorted(request_amax)}"
    for name, a, b in zip(("dQ", "dK", "dV"), (part.dq, part.dk, part.dv), (full.dq, full.dk, full.dv)):
        _bitwise(f"{name} depends on which amax outputs are requested", a, b)
    for name, value in part.amax.items():
        assert value == full.amax[name], f"amax_{name} differs from the all-requested plan's"


@requires_rubin
@pytest.mark.L1
@pytest.mark.parametrize("batch", [33, 129])
def test_thd_fp8_batched_descriptors(batch):
    """More sequences than setup warps (the per-sequence dV descriptors and stage 3's are shared round-robin), empty sequences
    interleaved, a poisoned capacity tail."""
    lens_q = [[0, 17, 65, 129][i % 4] for i in range(batch)]
    lens_kv = [[33, 0, 127, 257][i % 4] for i in range(batch)]
    _run_fp8_direct(lens_q, lens_kv, poison=True, pad_cap=256)


# --------------------------------------------------------------------------- the prepared plan: rebind, replay, length forms, artifact


@requires_rubin
@pytest.mark.parametrize("token_major", [False, True])
def test_prepared_thd_fp8_standalone_switches_length_and_prefix_form(token_major, monkeypatch):
    """The standalone surface takes ``(B,)`` lengths or ``(B+1,)`` prefixes per side without recompiling (the form is host metadata,
    derived from numel), and replays under CUDA-graph capture -- the twelve scalars and the amax outputs rebound by NAME."""
    import cutlass.cute as cute

    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    original = SdpaBwdDslSm107Fp8.execute

    def poison(args, kwargs):
        for tensor in args[6:9]:
            tensor.fill_(float("nan"))
        for name in _AMAX:
            if kwargs.get(name) is not None:
                kwargs[name].fill_(float("nan"))
        kwargs["workspace"].fill_(0xBD)

    def exercise(api, *args, **kwargs):
        original(api, *args, **kwargs)
        artifact = api._prepared.artifact
        for name in ("seq_q_lens", "seq_kv_lens"):
            lens = kwargs[name]
            kwargs[name] = torch.cat((torch.zeros(1, device=lens.device, dtype=torch.int32), lens.cumsum(0, dtype=torch.int32)))
        poison(args, kwargs)
        capture = torch.cuda.CUDAGraph()
        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(cute, "compile", lambda *a, **k: pytest.fail("length/prefix form must reuse the compiled host"))
                original(api, *args, **kwargs)
                with torch.cuda.graph(capture):
                    original(api, *args, **kwargs)
            poison(args, kwargs)
            capture.replay()
            assert api._prepared.artifact is artifact
        finally:
            capture.reset()

    monkeypatch.setattr(SdpaBwdDslSm107Fp8, "execute", exercise)
    _run_fp8_direct((129, 63, 97), (113, 75, 141), token_major_stats=token_major)


@requires_rubin
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_prepared_thd_fp8_rebind_lengths_and_replay(causal, monkeypatch):
    """The prepared THD plan rebinds new buffers AND new lengths (every length fact is a device value) without rebuilding tensor
    operands, allocating, synchronizing or compiling, and replays under CUDA-graph capture with changed lengths and data.  GQA
    with a padded capacity: the bounded fold must stop at every rebound live total."""
    import cutlass.cute as cute

    from cudnn.sdpa.bwd.api_dsl import WorkspaceCarver

    first = _run_fp8_direct((129, 97, 63), (143, 83, 79), h=4, hkv=2, poison=True, pad_cap=64, causal=causal)
    api, cap_q, cap_kv = first.api, first.case.cap_q, first.case.cap_kv
    grads = dict(dq=first.dq, dk=first.dk, dv=first.dv)
    buffers = dict(q=first.case.q, k=first.case.k, v=first.case.v, o=first.case.o, do=first.case.do, lse=first.case.lse)
    scalars = {name: first.kwargs[name] for name in _SCALARS}
    amax_t = first.amax_t
    lens = dict(q=first.kwargs["seq_q_lens"], kv=first.kwargs["seq_kv_lens"])
    ws = first.ws

    def load(lens_q, lens_kv):
        """A NEW case into the SAME buffers (copies, no new operands) with the same packed capacities and the first case's scalars."""
        case = _thd_fp8_case(lens_q, lens_kv, 4, 2, cap_q=cap_q, cap_kv=cap_kv, poison=True, causal=causal, quantize_ds=first.case.quantize_ds)
        for name in ("q", "k", "v", "o", "do", "lse"):
            buffers[name].copy_(getattr(case, name))
        for name, value in case.scalars.items():
            scalars[name].fill_(float(value))
        lens["q"].copy_(torch.tensor(lens_q, dtype=torch.int32, device="cuda"))
        lens["kv"].copy_(torch.tensor(lens_kv, dtype=torch.int32, device="cuda"))
        for x in grads.values():
            x.fill_(float("nan"))
        _sentinel_tails(case, grads["dq"], grads["dk"], grads["dv"])
        for a in amax_t.values():
            a.fill_(float("nan"))
        ws.fill_(0xBD)
        return case

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared THD rebuilt tensor operands, allocated, synchronized or compiled")

    def execute_guarded():
        with monkeypatch.context() as patcher:
            for name in ("view", "reshape", "as_strided", "permute", "transpose", "copy_", "zero_"):
                patcher.setattr(torch.Tensor, name, forbidden)
            for name in ("empty", "empty_like", "zeros", "zeros_like"):
                patcher.setattr(torch, name, forbidden)
            patcher.setattr(WorkspaceCarver, "__init__", forbidden)
            patcher.setattr(cute, "compile", forbidden)
            api.execute(*first.tensors, **first.kwargs)

    def verify(case):
        torch.cuda.synchronize()
        _assert_tails_untouched(case, grads["dq"], grads["dk"], grads["dv"])
        _check_fp8(case, grads["dq"], grads["dk"], grads["dv"], {name[len("amax_") :]: a.item() for name, a in amax_t.items()})

    case = load((63, 0, 121), (37, 81, 0))
    torch.cuda.set_sync_debug_mode("error")
    try:
        execute_guarded()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    verify(case)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            execute_guarded()
        case = load((31, 123, 41), (113, 67, 109))
        capture.replay()
        verify(case)
    finally:
        capture.reset()


class _StandaloneGraphShim:
    """The standalone adapter wearing the graph's prepared-plan surface (``_compiled_plans[_plan_index]._prepared.spec``,
    ``execute(pack, workspace)``) for the artifact-reload child of ``prepared_bwd_cache_utils``: the graph tier of this row needs
    the ``max_total_seq_len_*`` attribute on the fp8 backward node, the artifact does not."""

    def __init__(self, api):
        self._api = api
        self._plan_index = 0
        self._compiled_plans = [SimpleNamespace(_prepared=SimpleNamespace(spec=api._prepared))]

    def execute(self, pack, workspace):
        self._api.execute(**pack, workspace=workspace)


def _prepared_fp8_thd_case():
    """One executed THD fp8 plan (GQA, causal, poisoned capacity) as the reload child wants it: ``graph`` (the standalone shim),
    ``pack`` (the execute kwargs), ``workspace``, the live ``outs_t`` / ``amax_t`` and the first run's bits in ``outs`` / ``amax``."""
    run = _run_fp8_direct((129, 97, 63), (143, 83, 79), h=4, hkv=2, poison=True, pad_cap=64, causal=True)
    names = ("q_tensor", "k_tensor", "v_tensor", "o_tensor", "do_tensor", "stats_tensor", "dq_tensor", "dk_tensor", "dv_tensor")
    pack = dict(zip(names, run.tensors))
    pack.update({k: v for k, v in run.kwargs.items() if k != "workspace" and v is not None})
    case = SimpleNamespace(graph=_StandaloneGraphShim(run.api), pack=pack, workspace=run.ws, outs_t=dict(dQ=run.dq, dK=run.dk, dV=run.dv), amax_t=run.amax_t)
    case.outs = {name: t.clone() for name, t in case.outs_t.items()}
    case.amax = {name: t.clone() for name, t in case.amax_t.items()}
    case.live = dict(dQ=run.case.t_q, dK=run.case.t_kv, dV=run.case.t_kv)  # packed token totals: the rows the chain writes
    return case


def _check_prepared_fp8_thd(case):
    """The live outputs are BITWISE the first run's (two-launch pin) and every requested amax is.  Live = the packed totals: the
    reload protocol NaN-fills the whole output tensors before a replay, and the capacity tail past the totals is never written
    (the tails' untouched pin is ``_assert_tails_untouched`` on the first run), so the comparison stops at the live total."""
    torch.cuda.synchronize()
    for name, t in case.outs_t.items():
        live = case.live[name]
        _bitwise(f"{name}: a re-execute / replay of the prepared THD launch changed the bits", t[:, :live], case.outs[name][:, :live])
    for name, t in case.amax_t.items():
        assert torch.equal(t, case.amax[name]), f"{name}: a re-execute / replay changed the bits"


@requires_rubin
def test_prepared_thd_fp8_artifact_reloads_in_fresh_process(tmp_path):
    from prepared_bwd_cache_utils import check_backward_artifact_reload

    check_backward_artifact_reload("sm107", "fp8_thd", "float8_e4m3fn", tmp_path)


# --------------------------------------------------------------------------- the graph path: ragged fp8 tensors -> lower_dsl_bwd_fp8 -> packed views


def _build_thd_fp8_graph(case, *, stats_layout="head_major", declare_totals=True, envelope_q=None, request_amax=_AMAX, **sdpa_kwargs):
    """A ragged ``sdpa_fp8_backward`` graph over ``case``'s packed buffers: everything declared as the ENVELOPE (B, H, S_max, D)
    plus a per-tensor ragged offset -- how cuDNN spells a packed tensor -- the twelve scalars as (1, 1, 1, 1) fp32 tensors, the
    requested amax ports real.  ``declare_totals`` passes ``max_total_seq_len_q/kv`` (the python-native graph captures the kwarg;
    the C++ replay needs the node attribute -- ``_binding_declares_totals``)."""
    b, h, hkv, d, dev = case.b, case.h, case.hkv, case.d, case.q.device
    grad = _CUDNN_GRAD[case.grad_dtype]
    s_max_q, s_max_kv = envelope_q or max(max(case.lens_q), 1), max(max(case.lens_kv), 1)
    st_q = [s_max_q * h * d, d, h * d, 1]
    st_kv = [s_max_kv * hkv * d, d, hkv * d, 1]
    g = cudnn.pygraph(io_data_type=_FP8, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    ro_q_t = (torch.tensor(case.cu_q, dtype=torch.int64, device=dev) * (h * d)).view(b + 1, 1, 1, 1)
    ro_k_t = (torch.tensor(case.cu_k, dtype=torch.int64, device=dev) * (hkv * d)).view(b + 1, 1, 1, 1)
    vp, t = {}, {}

    def _ragged(name, s_max, stride, nh, ro_t):
        x = g.tensor(name=name, dim=[b, nh, s_max, d], stride=stride, data_type=_FP8)
        ro = g.tensor(name=f"{name}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
        x.set_ragged_offset(ro)
        vp[ro] = ro_t
        return x

    for n in ("q", "o", "dO"):
        t[n] = _ragged(n, s_max_q, st_q, h, ro_q_t)
    for n in ("k", "v"):
        t[n] = _ragged(n, s_max_kv, st_kv, hkv, ro_k_t)
    t_cap = max(64, -(-case.cap_q // 64) * 64)
    if stats_layout == "head_major":
        stats_stride = [h * t_cap, t_cap, 1, 1]
        stats_stor = torch.zeros(h * t_cap, dtype=torch.float32, device=dev)
        stats_stor.as_strided((1, h, case.cap_q), (h * t_cap, t_cap, 1)).copy_(case.lse[:, :, : case.cap_q])
        stats_ro_t = torch.tensor(case.cu_q, dtype=torch.int64, device=dev).view(b + 1, 1, 1, 1)
    else:
        stats_stride = [s_max_q * h, 1, h, 1]
        stats_stor = case.lse[0].transpose(0, 1).contiguous().reshape(-1)
        stats_ro_t = (torch.tensor(case.cu_q, dtype=torch.int64, device=dev) * h).view(b + 1, 1, 1, 1)
    st = g.tensor(name="stats", dim=[b, h, s_max_q, 1], stride=stats_stride, data_type=cudnn.data_type.FLOAT)
    st_ro = g.tensor(name="stats_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
    st.set_ragged_offset(st_ro)
    vp[st_ro], vp[st] = stats_ro_t, stats_stor
    for name in _SCALARS:
        t[name] = g.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name=name)
        vp[t[name]] = torch.tensor([[[[float(case.scalars[name])]]]], dtype=torch.float32, device=dev)
    slq = torch.tensor(case.lens_q, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    slk = torch.tensor(case.lens_kv, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    tq_len = g.tensor(name="seq_len_q", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    tk_len = g.tensor(name="seq_len_kv", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    vp[tq_len], vp[tk_len] = slq, slk
    kw = dict(name="fb", attn_scale=case.scale, use_padding_mask=True, seq_len_q=tq_len, seq_len_kv=tk_len)
    if declare_totals:
        kw.update(max_total_seq_len_q=case.cap_q, max_total_seq_len_kv=case.cap_kv)
    kw.update(sdpa_kwargs)
    outs = g.sdpa_fp8_backward(**{k: t[k] for k in ("q", "k", "v", "o", "dO", *_SCALARS)}, stats=st, **kw)
    t.update(zip(("dQ", "dK", "dV") + _AMAX, outs))
    for name, s_max, stride, nh, ro_t in (("dQ", s_max_q, st_q, h, ro_q_t), ("dK", s_max_kv, st_kv, hkv, ro_k_t), ("dV", s_max_kv, st_kv, hkv, ro_k_t)):
        t[name].set_output(True).set_data_type(grad).set_dim([b, nh, s_max, d]).set_stride(stride)
        ro = g.tensor(name=f"{name}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
        t[name].set_ragged_offset(ro)
        vp[ro] = ro_t
    for name in _AMAX:
        t[name].set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
        if name in request_amax:
            t[name].set_output(True)
    vp.update({t["q"]: case.q, t["k"]: case.k, t["v"]: case.v, t["o"]: case.o, t["dO"]: case.do})
    return g, vp, t


def _run_fp8_graph(
    lens_q, lens_kv, *, h=2, hkv=None, stats_layout="head_major", poison=False, pad_cap=0, poison_outputs=False, runs=1, grad_dtype=_T_E4M3, check=None, **kw
):
    """Build the ragged fp8 graph, PIN the engine, execute, compare per sequence.  Skips, naming the attribute, while the fp8
    backward node cannot declare its packed totals (the row's ``thd_declared_totals`` makes a graph without them a typed decline)."""
    if not _binding_declares_totals():
        pytest.skip(
            "the native sdpa_fp8_backward binding does not declare max_total_seq_len_q/kv yet (SDPA_fp8_backward_attributes); the direct tier carries the numerics"
        )
    case = _thd_fp8_case(
        lens_q, lens_kv, h, hkv, cap_q=sum(lens_q) + pad_cap, cap_kv=sum(lens_kv) + pad_cap, poison=poison,
        causal=bool(kw.get("use_causal_mask") or kw.get("use_causal_mask_bottom_right")), bottom_right=bool(kw.get("use_causal_mask_bottom_right")),
        window_left=kw.get("left_bound"), grad_dtype=grad_dtype, quantize_ds=(_ds_dtype_code() == DTYPE_E4M3),
    )  # fmt: skip
    g, vp, t = _build_thd_fp8_graph(case, stats_layout=stats_layout, **kw)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    assert _plan_index(g, _ENGINE) is not None, f"{_ENGINE} not offered; plans = {[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    fill = float("nan") if poison_outputs else 0.0
    dev = "cuda"
    dq = torch.empty(1, case.cap_q, case.h, _D, device=dev, dtype=grad_dtype)
    dk, dv = (torch.empty(1, case.cap_kv, case.hkv, _D, device=dev, dtype=grad_dtype) for _ in range(2))
    amax_t = {name: torch.full((1, 1, 1, 1), float("nan"), device=dev, dtype=torch.float32) for name in _AMAX}
    vp.update({t["dQ"]: dq, t["dK"]: dk, t["dV"]: dv})
    vp.update({t[name]: amax_t[name] for name in _AMAX})
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    outs = []
    for _ in range(runs):
        for x in (dq, dk, dv):
            x.fill_(fill)
        _sentinel_tails(case, dq, dk, dv)
        for a in amax_t.values():
            a.fill_(float("nan"))
        ws.fill_(0xFF)
        g.execute(vp, ws)
        torch.cuda.synchronize()
        outs.append(tuple(x.clone() for x in (dq, dk, dv)))
    for name, x, live in (("dQ", dq, case.t_q), ("dK", dk, case.t_kv), ("dV", dv, case.t_kv)):
        assert _finite(x[0, :live]), f"{name} has non-finite values in the packed region"
    _assert_tails_untouched(case, dq, dk, dv)
    amax = {name[len("amax_") :]: a.item() for name, a in amax_t.items()}
    (check or _check_fp8)(case, dq, dk, dv, amax)
    return SimpleNamespace(case=case, dq=dq, dk=dk, dv=dv, amax=amax, outs=outs, graph=g)


@requires_rubin
def test_graph_thd_fp8_self_attention(ds_knob):
    """The whole point of the graph tier: a ragged fp8 graph reaches the kernels through the engine."""
    _run_fp8_graph((300, 128, 200), (300, 128, 200))


@requires_rubin
@pytest.mark.parametrize("layout", ("head_major", "token_major"))
def test_graph_thd_fp8_stats_packings(layout):
    _run_fp8_graph((300, 128), (300, 128), stats_layout=layout)


@requires_rubin
def test_graph_thd_fp8_causal_gqa_nan_tail():
    _run_fp8_graph((300, 128, 200), (300, 128, 200), h=4, hkv=2, poison=True, pad_cap=384, use_causal_mask=True)


@requires_rubin
def test_graph_thd_fp8_two_launches_are_bitwise():
    run = _run_fp8_graph((300, 128, 200), (300, 128, 200), h=4, hkv=2, use_causal_mask=True, runs=2)
    for name, a, b in zip(("dQ", "dK", "dV"), run.outs[0], run.outs[1]):
        _bitwise(f"{name} differs between two launches", a, b)


# --------------------------------------------------------------------------- rejects and host pins (no Rubin GPU needed)


def _thd_adapter(cls=None, b=2, h=2, s_max=256, dt=_T_E4M3, grad_dt=_T_E4M3, **kw):
    """A THD fp8 adapter over the ENVELOPE (B, H, S_max, D) as ``TensorDesc``s -- no buffer, no device: the host rejects check shapes,
    dtypes and plan facts, nothing executes."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    return _adapter(SdpaBwdDslSm107Fp8 if cls is None else cls, b=b, hq=h, sq=s_max, skv=s_max, dt=dt, grad_dt=grad_dt, thd=True, **kw)


_TOTALS = dict(max_total_seq_len_q=400, max_total_seq_len_kv=400)


def test_capabilities_claim_thd_and_a_ragged_bottom_right():
    """The fp8 row claims THD with declared totals and bottom-right at ANY S_q (``bottom_right_s_q_multiple = 1``: the body threads
    the real q length now), and keeps declining the dense padding graph (it carries ``seq_len_q``, which no body threads) and the
    forward-only ``cu_seq_len`` ports.  Rule S2: the tracker rows change with these."""
    c = _spec().capabilities
    assert c.is_fp8 and c.thd and c.thd_declared_totals, "the fp8 row serves THD with declared packed totals"
    assert not c.cu_seq_len, "no BACKWARD node carries cu_seq_len_* (forward-only ports)"
    assert c.bottom_right_s_q_multiple == 1, "bottom-right at a ragged S_q is served: the fp8 body threads seqlen_q_real"
    assert not c.padded, "the dense padding graph stays declined (seq_len_q by construction; the standalone per-batch kv lengths are served)"


def test_fp8_thd_plan_facts_are_in_the_constructors_own_signature():
    """The engines' lowering forwards a plan-time fact to the adapter only when it appears in the constructor's OWN signature
    (``inspect.signature(adapter_cls.__init__)``; a bare ``**kwargs`` hides the base constructor's parameters).  The fp8 ctor
    re-declares the THD facts and ``external_delta`` next to its own ``amax_requested``."""
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    own = inspect.signature(SdpaBwdDslSm107Fp8.__init__).parameters
    base = inspect.signature(SdpaBwdDsl.__init__).parameters
    for name in ("thd", "max_total_seq_len_q", "max_total_seq_len_kv", "thd_stats_token_major", "thd_stats_head_stride"):
        assert name in own and name in base, f"{name} must be in the fp8 row's own constructor signature (the lowering forwards only those)"
        assert own[name].default == base[name].default, name
    assert "external_delta" in own and "amax_requested" in own


def test_fp8_thd_requires_declared_totals():
    """THD without ``max_total_seq_len_*`` is DECLINED, not silently mis-sized (the kv-blocked workspace, the packed delta and the
    bf16 partials are fixed at build time from the packed token capacity)."""
    with pytest.raises(ValueError, match="max_total_seq_len"):
        _thd_adapter().check_support()
    assert _thd_adapter(**_TOTALS).check_support()


def test_fp8_thd_refuses_the_dense_length_flags():
    """THD carries its lengths in the metadata buffer: ``seq_kv_lens_present`` / ``seq_q_lens_present`` with THD are refused
    (two sources of truth drift apart) -- while each alone is served on this row now (per-batch kv lengths) or declined for its
    own reason (per-batch Q lengths: no body threads them)."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _thd_adapter(seq_kv_lens_present=True, **_TOTALS).check_support()
    with pytest.raises(ValueError, match="mutually exclusive"):
        _thd_adapter(seq_q_lens_present=True, **_TOTALS).check_support()


def test_fp8_thd_declines_the_external_delta():
    """A caller's delta is a DENSE contract ([B, H_q, S_q_pad] fp32); the THD chain's delta is its own scaled ``dot_do_o`` over the
    packed e4m3 O / dO in the head-major ``[1, H_q, ceil128(T_q)]`` layout, and no producer emits that packed layout -- declined
    typed before any plan is built, on the fp8 row exactly as on the bf16 row.  The THD roles carry no delta slot; the dense fp8
    roles carry it LAST, after the appended per-batch kv lengths."""
    from cudnn.sdpa.bwd.prepared_sm107 import ATTRIBUTES_FP8_THD, EXTERNAL_DELTA_ROLE, ROLES, ROLES_FP8, ROLES_FP8_THD

    with pytest.raises(ValueError, match="external_delta is not served on the packed chain"):
        _thd_adapter(external_delta=True, **_TOTALS).check_support()
    api = _thd_adapter(**_TOTALS)
    assert api.check_support() and api.external_delta is False
    assert "delta" in [name for name, _n, _d in api._scratch_plan()], "the THD carve keeps the chain's own (packed) delta region"
    assert EXTERNAL_DELTA_ROLE not in ROLES_FP8_THD and EXTERNAL_DELTA_ROLE not in ATTRIBUTES_FP8_THD
    assert ROLES_FP8[-2:] == ("seq_kv", EXTERNAL_DELTA_ROLE), "the dense fp8 roles: the lengths, then the delta (appended)"
    assert ROLES_FP8_THD[:11] == ROLES[:9] + ("seq_q", "seq_kv"), "the THD roles put the two length operands at slots 9 / 10 (the bf16 THD spec's order)"
    assert ATTRIBUTES_FP8_THD[9:11] == ("seq_len_q", "seq_len_kv") and len(ATTRIBUTES_FP8_THD) == len(ROLES_FP8_THD)
    with pytest.raises(ValueError, match="external_delta=False"):
        api._check_external_delta(torch.zeros(1, 2, 128))


def test_fp8_thd_execute_requires_both_lengths(monkeypatch):
    """A THD plan's execute needs ``seq_q_lens`` AND ``seq_kv_lens`` -- refused typed before compile, ahead of the scalar checks."""
    api = _thd_adapter(**_TOTALS)
    monkeypatch.setattr(api, "compile", lambda: pytest.fail("refused before compile"))
    lens = torch.zeros(2, dtype=torch.int32)
    scalars = {name: lens for name in _SCALARS}
    with pytest.raises(ValueError, match="seq_q_lens AND seq_kv_lens"):
        api.execute(*([None] * 9), seq_kv_lens=lens, **scalars)
    with pytest.raises(ValueError, match="seq_q_lens AND seq_kv_lens"):
        api.execute(*([None] * 9), seq_q_lens=lens, **scalars)


@pytest.mark.parametrize("knob", _DS_KNOBS)
def test_fp8_thd_scratch_plan_is_the_packed_carve(knob, monkeypatch):
    """The THD workspace of the fp8 row, in carve order: the packed head-major delta, ONE head chunk of the kv-BLOCKED e4m3 (or
    bf16-twin) dS workspace, the metadata + the main kernel's (5 + B) tensor maps, stage 3's (B + 1) descriptors, the bf16 dV
    partials ALWAYS (the fold + quantize pass is unconditional on the fp8 row), the GQA dK partials, the bf16-dS twin's three
    extra buffers, and the amax scratch.  ``regions[seq_kv].numel`` is pinned to the family's map-slot count."""
    from cudnn.frost.tile_dsl.thd import THD_BWD_MAPS_META_WORDS
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107
    from cudnn.sdpa.fwd.api_dsl import ws_align

    monkeypatch.setattr(sm107, "FP8_DS_DTYPE", knob)
    b, h, hkv = 2, 4, 2
    api = _thd_adapter(b=b, h=h, hkv=hkv, **_TOTALS)
    assert api.check_support()
    plan = {name: (tuple(int(x) for x in shape), dt) for name, shape, dt in api._scratch_shapes()}
    names = [name for name, _s, _d in api._scratch_shapes()]
    tq, tkv = api._t_q_cap, api._t_kv_cap
    assert names[:4] == ["delta", "ds_ws", "seq_kv", "desc_words"], names
    assert plan["delta"] == ((1, h, -(-tq // 128) * 128), torch.float32)
    assert plan["ds_ws"][0] == (1, api._qh_chunk, api._ws_rows_cap, api._sq_pad) and plan["ds_ws"][1] == (_T_E4M3 if knob == DTYPE_E4M3 else torch.bfloat16)
    assert plan["seq_kv"] == ((THD_BWD_MAPS_META_WORDS(b, 5 + b),), torch.int32), "the metadata words + (5 + B) tensor maps, the bf16 THD layout"
    assert plan["desc_words"] == (((b + 1) * 16,), torch.int64)
    assert plan["dv_part"] == ((1, tkv, h, _D), torch.bfloat16), "the per-Q-head dV_true partials over the packed kv capacity, bf16, always"
    assert plan["dk_part"] == ((1, tkv, h, _D), torch.bfloat16), "GQA: the per-Q-head dK partials (EPI_DESCALE true units) over the packed kv capacity"
    assert plan["amax_scratch"] == ((8,), torch.float32)
    twin = {"dq_ws": (1, tq, h, _D), "q_bf16": (1, tq, h, _D), "k_bf16": (1, tkv, hkv, _D)}
    if knob == DTYPE_E4M3:
        assert not (set(twin) & set(plan)), "the e4m3 chain quantizes dQ / MHA dK in the GEMM epilogue: no bf16 staging"
    else:
        for name, shape in twin.items():
            assert plan[name] == (shape, torch.bfloat16), name
    assert all(
        name not in plan for name in ("q_pad", "do_pad", "lse_pad", "k_pad", "v_pad")
    ), "no staging under THD: the packed path reads the caller's buffers"
    assert api.scratch_workspace_bytes() == sum(ws_align(math.prod(s) * dt.itemsize) for s, dt in plan.values())


@pytest.mark.parametrize("b", [1, 2, 3, 8])
def test_fp8_thd_map_slot_count_agrees_between_the_body_and_the_adapter(b):
    """The THD tensor-map slot count is spelled on both sides of the metadata buffer: the body's ``THD_MAP_SLOTS(b)`` (its setup
    kernel writes that many 128-B maps, ``_host`` views them) and the adapter's ``_thd_map_slots(b)`` (the scratch carve and the
    host's maps view).  A row that moves one side only -- the MXFP8 body's five extra clamped scale-factor maps are the planned
    override -- lets the maps overrun the words after them, silent and faultless; so the two are pinned equal per row, read from
    the LOADED body, never from a literal."""
    from test_sdpa_bwd_dsl_sm107 import _load_kernel

    api = _thd_adapter(b=b, h=2, hkv=2, max_total_seq_len_q=128 * b + 72, max_total_seq_len_kv=128 * b + 72)
    mod = _load_kernel("fp8", thd_varlen=True)
    assert mod.THD_INPUT_SLOTS == 5, "five packed-total-clamped input maps (Q, dO, dO_dv, K, V) ahead of the per-sequence dV maps"
    assert mod.THD_MAP_SLOTS(b) == api._thd_map_slots(b) == mod.THD_INPUT_SLOTS + b, (mod.THD_MAP_SLOTS(b), api._thd_map_slots(b))


def test_stage3_thd_band_arithmetic_fp8_k128():
    """The THD K-trim's contract on the sibling module's ~28 per-sequence geometries with the fp8 K64 arm's K TILE (128 e4m3
    elements = 128 bytes, ``cta_tile_mnk[2]`` of the fp8 rendering; the bf16 arm's is 64): every kept cell covered, every tile read
    written by the sequence's own kv blocks, an EMPTY range where the tile has no kept cell."""
    _band_arithmetic(128)


def test_stage3_fp8_thd_records_are_admitted_and_spelled():
    """The stage-3 template admits the fp8 THD renderings (``dtype_qkv=DTYPE_E4M3``, the QUANT / DESCALE epilogues, ``thd_varlen``
    with ``thd_rows_kv``) and the adapter's THD records are the dense fp8 records with the THD arm on: ``epi_modes`` kept
    (``EPI_QUANT`` dQ; dK ``EPI_QUANT`` at MHA, ``EPI_DESCALE`` under GQA), the gradient dtype kept, ``causal_shift`` 0 (the
    per-sequence diagonal is read from the metadata: ``thd_causal_bottom_right``), the window KEPT, dQ's ``b_head_group`` the
    GQA group under the single launch -- and the bf16-dS twin's records the plain bf16 THD renderings."""
    import types

    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107
    from cudnn.sdpa.bwd.config_sm100 import (
        CAUSAL_K_HI,
        CAUSAL_K_LO,
        CAUSAL_K_NONE,
        EPI_DESCALE,
        EPI_NONE,
        EPI_QUANT,
        MatmulTemplateParams,
        validate_matmul_params,
    )

    fp8 = dict(dtype_qkv=DTYPE_E4M3, cgrp_tile_mn=(256, 256), thd_varlen=True, thd_rows_kv=True)
    for ok in (dict(**fp8, epi_mode=EPI_QUANT), dict(**fp8, epi_mode=EPI_QUANT, dtype_out=DTYPE_E4M3), dict(**fp8, epi_mode=EPI_DESCALE)):
        validate_matmul_params(MatmulTemplateParams(**ok))
    validate_matmul_params(
        MatmulTemplateParams(
            **fp8, epi_mode=EPI_QUANT, causal_mode=CAUSAL_K_HI, causal_gran=256, a_is_m_major=True, thd_causal_bottom_right=True, b_head_group=2
        )
    )
    mod = types.SimpleNamespace(CFG=types.SimpleNamespace(TILE_M=128, CTA_MMA=2))
    for hkv, dk_mode in ((2, EPI_QUANT), (1, EPI_DESCALE)):
        api = _thd_adapter(h=2, hkv=hkv, is_causal=True, causal_bottom_right=True, window_size_left=63, **_TOTALS)
        assert api.check_support()
        dk, dq = api._stage3_records(mod, (256, 256))
        assert dk.thd_varlen and dq.thd_varlen and dk.thd_rows_kv and dq.thd_rows_kv
        assert dk.dtype_qkv == dq.dtype_qkv == DTYPE_E4M3
        assert (dk.epi_mode, dq.epi_mode) == (dk_mode, EPI_QUANT)
        assert dq.dtype_out == DTYPE_E4M3 and dk.dtype_out == (
            DTYPE_E4M3 if dk_mode == EPI_QUANT else -1
        ), "QUANT stores the gradient dtype; DESCALE the inherited bf16"

        assert (dk.causal_mode, dq.causal_mode) == (CAUSAL_K_LO, CAUSAL_K_HI) and dk.causal_shift == dq.causal_shift == 0
        assert dk.thd_causal_bottom_right and dq.thd_causal_bottom_right
        assert dk.causal_window == dq.causal_window == 63, "the window edge is KEPT under THD (anchored per sequence by the trim)"
        assert dq.b_head_group == (2 // hkv if sm107.DQ_SINGLE_LAUNCH else 1) and dk.b_head_group == 1
    api = _thd_adapter(h=2, hkv=2, **_TOTALS)
    assert api.check_support()
    dk, dq = api._stage3_records(mod, (256, 256))
    assert (dk.causal_mode, dq.causal_mode) == (CAUSAL_K_NONE, CAUSAL_K_NONE) and not dk.thd_causal_bottom_right
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sm107, "FP8_DS_DTYPE", DTYPE_BF16)
        api = _thd_adapter(h=2, hkv=2, is_causal=True, **_TOTALS)
        assert api.check_support()
        dk, dq = api._stage3_records(mod, (256, 256))
        assert dk.dtype_qkv == dq.dtype_qkv == DTYPE_BF16 and dk.thd_varlen and dq.thd_varlen and dk.epi_mode == dq.epi_mode == EPI_NONE


def test_fp8_thd_stage3_host_threads_the_epilogue_operands():
    """Source pin on the fp8 THD host: its stage-3 call hands the epilogue operands (``dk_epi`` / ``dq_epi``: the descale and
    quantize scalars, the amax pointers) to the THD stage-3 helper -- the dense fp8 host does; a THD host that reused the bf16
    helper's call shape would silently render EPI_NONE-shaped GEMMs (dQ wrong by ``descale_dP * descale_k``)."""
    from pathlib import Path

    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host

    src = Path(prepared_host.__file__).read_text(encoding="utf-8")
    assert "def host_fp8_thd(" in src, "the fp8 row's THD host is a sibling artifact (host_fp8_thd), like host_f16_thd"
    body = src[src.index("def host_fp8_thd(") :]
    body = body[: body.index("\ndef ", 1)]
    assert "_stage3_thd(" in body and "dk_epi" in body and "dq_epi" in body, "host_fp8_thd must pass dk_epi / dq_epi into the THD stage-3 call"
    assert "fold_quant_host(" in body and "THD_CU_K_TOTAL_OFF" in body, "the fold + quantize pass is bounded at the live kv total on device"
    assert "dot_do_o_scaled_host(" in body, "the packed delta is the scaled dot over the packed e4m3 O / dO"


# --- the graph tier: served once the fp8 backward node declares its totals; a typed decline at mismatch() meanwhile

_requires_cuda_device = requires_sm80


def _thd_fp8_mismatch(lens_q=(256, 128), lens_kv=(256, 128), *, h=2, hkv=None, declare_totals=True, monkeypatch=None, **kw):
    """``mismatch()`` for a ragged fp8 backward graph on the row (the analyzer's cc faked to 10.7 off the Rubin line), or None if
    served.  The case lives on the CPU: the graph needs its geometry, never its data."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    if monkeypatch is not None:
        monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    case = _thd_fp8_case(lens_q, lens_kv, h, hkv, device="cpu", oracle=False)
    g, _vp, _t = _build_thd_fp8_graph(case, declare_totals=declare_totals, **kw)
    try:
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError as exc:
        return f"refused by the node: {exc}"
    facts = ga.analyze(g)
    assert facts is not None and facts.is_fp8 and facts.thd
    return mismatch(_spec().capabilities, facts)


@_requires_cuda_device
def test_graph_thd_fp8_tier_declines_typed_until_the_totals_attribute_lands(monkeypatch):
    """The ragged fp8 backward graph on the row: with the packed totals declared it is SERVED (the row claims ``thd`` and
    ``thd_declared_totals``); WITHOUT them it is a typed decline naming ``max_total_seq_len`` -- the state every ragged fp8 graph is
    in while ``SDPA_fp8_backward_attributes`` cannot carry the totals (the C++ replay forwards the kwarg to the native binding), so
    a caller gets a decline it can act on, never a mis-sized workspace.  Self-inverting on the binding's signature."""
    reason = _thd_fp8_mismatch(monkeypatch=monkeypatch, declare_totals=False)
    assert reason is not None and "max_total_seq_len" in reason, reason
    if _binding_declares_totals():
        assert _thd_fp8_mismatch(monkeypatch=monkeypatch) is None
        assert _thd_fp8_mismatch(monkeypatch=monkeypatch, use_causal_mask_bottom_right=True) is None
        assert _thd_fp8_mismatch(monkeypatch=monkeypatch, h=4, hkv=2, use_causal_mask=True, left_bound=64) is None
    else:
        # the python-native graph captured the kwarg: the row's eligibility is satisfied; only the C++ replay would refuse it
        assert _thd_fp8_mismatch(monkeypatch=monkeypatch) is None, "the row's own mismatch() serves a ragged fp8 graph that declares its totals"


@_requires_cuda_device
def test_graph_thd_fp8_backend_replay_needs_the_totals_attribute(monkeypatch):
    """The OTHER half of the graph tier, below ``mismatch()``: the python-native graph captures ``max_total_seq_len_q/kv`` on the
    fp8 backward node and the C++ replay forwards every captured kwarg to the native ``sdpa_fp8_backward`` binding.  While that
    binding lacks the two arguments (``SDPA_fp8_backward_attributes`` without the fields), ``create_execution_plans`` on a ragged fp8
    graph that declared its totals dies with a bare ``TypeError`` (``incompatible function arguments``) -- UNTYPED, on any device,
    before any engine is asked (MEASURED on the A100 host and pinned here so the state is visible); without the totals the same
    graph is a typed ``cudnnGraphNotSupportedError`` (the row declines ``max_total_seq_len``, the backend has no plan).  Once the
    binding declares the totals the replay must get past this point: the graph is served on the Rubin line and a typed decline
    elsewhere.  Self-inverting on the binding's signature."""
    from cudnn.sdpa import graph_analyzer as ga

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    case = _thd_fp8_case((256, 128), (256, 128), 2, device="cuda", oracle=False)
    g, _vp, _t = _build_thd_fp8_graph(case, declare_totals=True)
    g.validate()
    g.build_operation_graph()
    if not _binding_declares_totals():
        with pytest.raises(TypeError, match="sdpa_fp8_backward"):
            g.create_execution_plans([cudnn.heur_mode.A])
        return
    try:
        g.create_execution_plans([cudnn.heur_mode.A])
    except cudnn.cudnnGraphNotSupportedError:
        assert torch.cuda.get_device_capability() != _RUBIN_CC, "a Rubin device must offer the fp8 THD plan"
        return
    assert _plan_index(g, _ENGINE) is not None, [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]


@_requires_cuda_device
def test_reject_thd_fp8_conjunctions_the_row_declines(monkeypatch):
    """Under THD the row keeps its dense declines: right-band widening, the sink token, E5M2 payloads; and a ragged graph
    whose packed Stats lost its ragged offset is declined on the absent offset."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    reason = _thd_fp8_mismatch(monkeypatch=monkeypatch, right_bound=16)
    assert reason is not None and "right-band" in reason
    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    case = _thd_fp8_case((256, 128), (256, 128), 2, device="cpu", oracle=False)
    g, _vp, _t = _build_thd_fp8_graph(case)
    g.nodes[0].inputs["stats"].set_ragged_offset(None)
    try:
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError:
        return
    facts = ga.analyze(g)
    reason = mismatch(_spec().capabilities, facts)
    assert reason is not None and "ragged" in reason
