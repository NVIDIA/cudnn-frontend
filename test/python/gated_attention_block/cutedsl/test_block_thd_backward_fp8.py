# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PACKED (THD / varlen) per-tensor fp8 block backward: ``GatedAttentionBlockBwd(thd=True, quant=QuantSpec, ...)`` over the
packed per-tensor fp8 training record AS WRITTEN (``test_block_thd.py``'s ``_run_fp8_thd(training=True)``: e4m3 ``saved.h``, the
bf16 slab / ``o`` / ``lse`` / ``rstd``, ``saved.seq_lens`` + ``saved.seq_lens_form``).

What runs is the dense fp8 backward's stage list at ``B = 1, S = T`` (``test_block_backward_fp8.py``: the fused prologue, the
quantizers, the four e4m3 GEMMs, the gate backward with its mandatory delta, the norm backward, the fused epilogue) with ONE
stage declared packed: the fp8 SDPA row's THD chain over the envelope ``(num_sequences, max_seq_len)``, reading the gate
backward's delta -- at ``B = 1, S = T`` the packed head-major ``[1, H_q, ceil128(T)]`` delta the packed chain reads at the packed
token index -- as its external one (no ``dot`` pre-pass over the e4m3 payloads: one rounding of dO, the dense arm's contract).

The oracle is the dense suite's MODELLED fp8 oracle (``gated_attention_block_fp8_bwd_reference(modelled=True)`` fed the record's
exact LSE, bf16 pre-gate O, bf16 GATE band, e4m3 ``q8 / k8 / v8`` and the block's bf16 dO, ``scale_dP`` and the SAME delta) run
PER SEQUENCE on the packed rows: ``dh`` packed back row by row, every weight gradient the SUM over the sequences, the ``dW_norm``
masses combined as ``sqrt(sum mass_i^2)``, ``amax_dP`` the max over the sequences.  Tolerances are the dense fp8 suite's,
imported and never re-literalled: the fp8 SDPA row's recipe on the SDPA stage per sequence (``_row_tol``: ``_FP8_GRAD_TOL`` with
``assert_close_fp8_grad``'s flip budget, ``amax_dP`` under ``_AMAX_DS_TOL``) plus the bf16 block's bound form on the stage's bf16
output, the (M) end-to-end in the row-budgeted form (``1e-5 x rows x keys``) on ``dh / dw_qkvg / dw_o``, the seeded oracle under the
bf16 block's bound on ``dh / dw_o / dW_norm``; the quantizers, every scalar and the delta BITWISE (the dense suite's layer, unchanged
at ``B = 1, S = T``).  ``B = 1`` packed is pinned ``torch.equal`` the dense fp8 backward over the same bytes (a difference is a
finding to investigate at the SDPA stage first, never a tolerance); ``grad_scaling="delayed"`` replays the current run bitwise;
the launch count is MEASURED (CUPTI) against the formula from the adapter's own facts.

Accept tests are ``requires_rubin``; the host tests build CUDA tensors for a DECLARED backward (``requires_cuda``, no compile).
"""

import dataclasses
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block packed fp8 backward tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import GatedAttentionBlockBwd, GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QuantSpec  # noqa: E402
from cudnn.gated_attention_block.api import MxQuantSpec, _cols, _view  # noqa: E402
from cudnn.gated_attention_block.api_bwd import _check_saved_record  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import RefGeometry, gated_attention_block_fp8_bwd_reference, make_packed_inputs, quantize_block_inputs, sequence_slices  # noqa: E402
from test_block_backward import _alloc_grads, _assert_dw_norm_close, _assert_grad_close, _cos, _make_dy  # noqa: E402
from test_block_backward_fp8 import (  # noqa: E402
    _E4M3,
    _api_const,
    _assert_quantizers_scalars_delta_bitwise,
    _calibrated_scale_dp,
    _declare_fp8_bwd,
    _delta,
    _dev_scalar,
    _execute_fp8,
    _print_end_to_end,
    _report_close,
    _report_stage_difference,
    _row_keys,
    _row_tol,
    _slots,
    _test_python_root,
)
from test_block_thd import _COMMON, _LENS, _alloc_packed_saved, _no_device_sync, _run_fp8_thd, _thd_kw  # noqa: E402
from test_block_training_forward import _alloc_saved  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture(autouse=True)
def _no_tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


# ---------------------------------------------------------------------------
# Building a packed fp8 backward
# ---------------------------------------------------------------------------


def _geom_kw(*, causal=True, h_kv=2) -> dict:
    """The packed suite's geometry (``_COMMON``: d_model 512, h_q 8, D 256, rope 64, QK-norm) with the mask and the GQA group varied."""
    return {**_COMMON, "is_causal": bool(causal), "h_kv": int(h_kv)}


_MATRIX = pytest.mark.parametrize(
    "causal, h_kv", [(True, 2), (False, 2), (True, 8), (False, 8)], ids=["causal-gqa_8_2", "dense-gqa_8_2", "causal-mha", "dense-mha"]
)

_MEMO: dict = {}


def _backward_fp8_thd(lens=_LENS, *, causal=True, h_kv=2, grad_scaling="current", scale_dp="calibrated", scales=None, max_seq_len=None, memo=True, **bwd_kw):
    """Run the packed per-tensor fp8 TRAINING forward (the record, KEPT with its workspace alive: its e4m3 bytes are the reference
    the recomputed ``q8 / k8 / v8`` are pinned against), declare / compile / run the packed fp8 backward over the record AS WRITTEN
    and read its scalar block back.  ``scale_dp``: a float or ``"calibrated"`` (the dense suite's recipe: one warm-up at 1.0, then
    ``get_fp8_scale_factor(amax_dP)``, then the measured run).  Memoised per declaration."""
    key = (tuple(lens), causal, h_kv, grad_scaling, str(scale_dp), tuple(sorted((scales or {}).items())), max_seq_len, tuple(sorted(bwd_kw.items())))
    if memo and key in _MEMO:
        return _MEMO[key]
    geom_kw = _geom_kw(causal=causal, h_kv=h_kv)
    r = _run_fp8_thd(lens, training=True, max_seq_len=max_seq_len, geom_kw=geom_kw)
    inp, saved, meta, g = r.inp, r.saved, r.meta, r.geom
    assert saved.h is inp["h"] and saved.h.dtype == _E4M3 and saved.seq_lens is r.seq_lens and saved.seq_lens_form == "lengths"
    dy = _make_dy(r.out)
    blk = _declare_fp8_bwd(dy, saved, inp, g, quant=r.spec, grad_scaling=grad_scaling, **_thd_kw(meta), **bwd_kw)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    scale_dp_t = _dev_scalar(1.0 if scale_dp == "calibrated" else scale_dp)
    scale_ts = {k: _dev_scalar(v) for k, v in (scales or {}).items()}
    _execute_fp8(blk, inp, saved, dy, grads, ws, scale_dp=scale_dp_t, **scale_ts)
    if scale_dp == "calibrated":
        scale_dp_t.fill_(_calibrated_scale_dp(blk, ws))
        _execute_fp8(blk, inp, saved, dy, grads, ws, scale_dp=scale_dp_t, **scale_ts)
    torch.cuda.synchronize()
    res = SimpleNamespace(
        blk=blk,
        fwd=r,
        inp=inp,
        spec=r.spec,
        saved=saved,
        dy=dy,
        out=r.out,
        ws=ws,
        grads=grads,
        scale_dp=float(scale_dp_t.item()),
        scale_dp_t=scale_dp_t,
        scale_ts=scale_ts,
        scalars={k: float(v.item()) for k, v in blk.quant_scalars(ws).items()},
        geom=g,
        geom_kw=geom_kw,
        batch=1,
        seq_len=meta["t"],
        meta=meta,
        lens=meta["lens"],
        grad_scaling=grad_scaling,
    )
    if memo:
        _MEMO[key] = res
    return res


def _twin_fp8_thd(res, *, poison=0xFF, grad_scaling=None, scales=None, **bwd_kw):
    """A second packed fp8 block over the SAME record / dy / inputs / scale_dp as ``res`` (different knobs or recipe), compiled and
    run once into a poisoned workspace and NaN-filled gradients; returns ``(blk, ws, grads)`` -- the bitwise comparand of ``res``."""
    blk = _declare_fp8_bwd(res.dy, res.saved, res.inp, res.geom, quant=res.spec, grad_scaling=grad_scaling or res.grad_scaling, **_thd_kw(res.meta), **bwd_kw)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(poison)
    grads = _alloc_grads(blk, fill=float("nan"))
    scale_ts = {k: _dev_scalar(v) for k, v in (scales or {}).items()} if scales is not None else res.scale_ts
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws, scale_dp=res.scale_dp_t, **scale_ts)
    torch.cuda.synchronize()
    return blk, ws, grads


def _scalar_block(blk, ws) -> torch.Tensor:
    return _view(ws, blk._layout().quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32)


def _gate_rows(res) -> torch.Tensor:
    """The record's bf16 GATE band as the gate kernels read it: ``[T, H_q, D]``, strided over the slab."""
    g, t = res.geom, res.seq_len
    _o_q, o_g, _o_k, _o_v = g.qkvg_offsets
    return _cols(res.saved.proj_slab.view(t, g.n_qkvg), o_g, g.h_q, g.d_head)


# ---------------------------------------------------------------------------
# The oracles, per sequence
# ---------------------------------------------------------------------------


def _oracle_m_packed(res, *, seeded=False) -> dict:
    """The dense suite's MODELLED fp8 oracle PER SEQUENCE, fed the kernels' own inputs sliced to the sequence's rows: the record's
    exact LSE columns, bf16 pre-gate O rows, bf16 GATE rows, the block's recomputed e4m3 ``q8 / k8 / v8`` rows and its bf16 dO rows,
    the block's read-back gradient scales, ``2 ** FP8_SCALE_S_LOG2``, its ``scale_dp`` and the SAME ``delta`` columns the kernel
    consumed.  ``dh`` is packed back row by row, every weight gradient is the SUM over the sequences, the ``dW_norm`` masses combine as
    ``sqrt(sum mass_i^2)``, ``amax_dp`` is the max over the sequences; ``per_seq`` keeps each sequence's dict.  ``seeded`` substitutes
    the block's own bf16 dQ / dK / dV rows per sequence (``amax_dp`` is then None)."""
    g, t, sc, d = res.geom, res.seq_len, res.scalars, res.geom.d_head
    v = _slots(res)
    gate = _gate_rows(res)
    delta = _delta(res)
    ref_geom = RefGeometry(**res.geom_kw)
    s_scale = 2.0 ** _api_const("FP8_SCALE_S_LOG2")
    per_seq, total, amax_dp = [], None, 0.0
    dh = torch.zeros(1, t, g.d_model, dtype=torch.float64, device=res.dy.device)
    for lo, hi in sequence_slices(res.lens):
        if hi == lo:
            per_seq.append(None)
            continue
        n = hi - lo
        inp_i = dict(res.inp, h=res.inp["h"][:, lo:hi], cos=res.inp["cos"][:, lo:hi], sin=res.inp["sin"][:, lo:hi])
        record = dict(
            lse=res.saved.lse[:, :, lo:hi],
            o=res.saved.o[:, lo:hi],
            gate=gate[lo:hi],
            q8=v["q8"][lo:hi],
            k8=v["k8"][lo:hi],
            v8=v["v8"][lo:hi],
            do=v["do"][lo:hi],
        )
        seed = (
            dict(dq=v["dq"][lo:hi].view(1, n, g.h_q, d), dk=v["dk"][lo:hi].view(1, n, g.h_kv, d), dv=v["dv"][lo:hi].view(1, n, g.h_kv, d)) if seeded else None
        )
        o = gated_attention_block_fp8_bwd_reference(
            inp_i,
            ref_geom,
            res.spec,
            res.dy[:, lo:hi],
            scale_dy=sc["scale_dy"],
            scale_do=sc["scale_do"],
            scale_dqkvg=sc["scale_dqkvg"],
            scale_s=s_scale,
            scale_dp=res.scale_dp,
            delta=delta[..., lo:hi],
            modelled=True,
            seeded=seed,
            **record,
        )
        per_seq.append(o)
        dh[0, lo:hi] = o["dh"].reshape(n, g.d_model)
        if o.get("amax_dp") is not None:
            amax_dp = max(amax_dp, float(o["amax_dp"]))
        if total is None:
            total = {k: (o[k].clone() if o[k] is not None else None) for k in ("dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm")}
            total.update({k: (o[k] ** 2 if o.get(k) is not None else None) for k in ("dw_q_norm_mass", "dw_k_norm_mass")})
        else:
            for k in ("dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"):
                if total[k] is not None:
                    total[k] += o[k]
            for k in ("dw_q_norm_mass", "dw_k_norm_mass"):
                if total.get(k) is not None:
                    total[k] += o[k] ** 2
    assert total is not None, "every sequence is empty"
    for k in ("dw_q_norm_mass", "dw_k_norm_mass"):
        if total.get(k) is not None:
            total[k] = total[k].sqrt()
    total["dh"] = dh
    total["per_seq"] = per_seq
    total["amax_dp"] = None if seeded else amax_dp
    return total


def _row_reference_seq(res, v: dict, lo: int, hi: int):
    """The fp8 row's reference over ONE sequence of the block's own operands and conditions (the dense suite's ``_row_reference``
    on the rows ``[lo, hi)``): e4m3 ``q8 / k8 / v8 / do8`` rows, the forward's exact LSE columns, the block's ``delta`` columns (the
    SAME values the kernel consumed), ``scale_s = 2 ** FP8_SCALE_S_LOG2``, the caller's ``scale_dp``, bf16 gradients.  Returns
    ``(dq, dk, dv)`` fp32 BSHD at ``(1, n, H, D)``, the fp32 ``max |dS|`` of the sequence and ``ref_bwd(selection)`` -- the reference
    re-run with ``return_intermediates=selection``, the evidence ``assert_close_fp8_grad`` consults to PROVE a bad d-row is one e4m3
    midpoint flip."""
    _test_python_root()
    from sdpa.fp8_ref import compute_ref_backward

    g, sc, sp = res.geom, res.scalars, res.spec
    n, d = hi - lo, g.d_head
    s_scale = 2.0 ** _api_const("FP8_SCALE_S_LOG2")
    bshd = lambda x, h: x[lo:hi].reshape(1, n, h, d)  # noqa: E731
    o_dead = v["og8"] if v["og8"] is not None else v["do8"]

    def ref_bwd(return_intermediates=True):
        return compute_ref_backward(
            bshd(v["q8"], g.h_q),
            bshd(v["k8"], g.h_kv),
            bshd(v["v8"], g.h_kv),
            bshd(o_dead, g.h_q),
            bshd(v["do8"], g.h_q),
            g.scale,
            1.0 / sp.scale_q,
            1.0 / sp.scale_k,
            1.0 / sp.scale_v,
            s_scale,
            1.0 / s_scale,
            _E4M3,
            1.0 / sp.scale_o,
            sc["descale_do"],
            torch.bfloat16,
            left_bound=None,
            right_bound=0 if g.is_causal else None,
            diag_align=None,
            stats=res.saved.lse[:, :, lo:hi, None],
            return_intermediates=return_intermediates,
            quantize_ds=True,
            dP_scale=res.scale_dp,
            quantize_grads=False,
            delta=_delta(res)[..., lo:hi],
        )

    dq, dk, dv, _dsink, _dp_amax_raw, _dqa, _dka, _dva, inter = ref_bwd(True)
    amax_ds = float(inter["ds_scaled"].abs().max()) / res.scale_dp
    return dq.to(torch.bfloat16).float(), dk.to(torch.bfloat16).float(), dv.to(torch.bfloat16).float(), amax_ds, ref_bwd


def _assert_sdpa_stage_per_sequence(res, v: dict) -> None:
    """The SDPA stage's bf16 ``dq / dk / dv`` rows of EVERY sequence against the fp8 row's reference over the sequence's own operands
    under the row's recipe (``_FP8_GRAD_TOL`` + ``assert_close_fp8_grad``'s flip budget, the flip proof fed the sequence's operands)
    AND under the bf16 block's bound form (the stricter pin: the row recipe's absolute ``atol 0.08`` cannot reject an all-zero dQ / dK
    at this geometry); ``amax_dP`` -- the max over the sequences of ``max |dS|`` -- under ``_AMAX_DS_TOL``."""
    g, sc, sp, d = res.geom, res.scalars, res.spec, res.geom.d_head
    grad_tol, amax_tol, assert_close_fp8_grad = _row_tol()
    s_scale = 2.0 ** _api_const("FP8_SCALE_S_LOG2")
    amax_ref = 0.0
    for i, (lo, hi) in enumerate(sequence_slices(res.lens)):
        if hi == lo:
            continue
        n = hi - lo
        dq_ref, dk_ref, dv_ref, amax_ds_i, ref_bwd = _row_reference_seq(res, v, lo, hi)
        amax_ref = max(amax_ref, amax_ds_i)
        refs = dict(dq=dq_ref, dk=dk_ref, dv=dv_ref)
        operands = dict(
            dq=v["k8"][lo:hi].view(1, n, g.h_kv, d).float() / sp.scale_k,
            dk=v["q8"][lo:hi].view(1, n, g.h_q, d).float() / sp.scale_q,
            dv=v["do8"][lo:hi].view(1, n, g.h_q, d).float() * sc["descale_do"],
        )
        flip_unit = dict(dq=1.0 / res.scale_dp, dk=1.0 / res.scale_dp, dv=1.0 / s_scale)
        for name, h, tag in (("dq", g.h_q, "dQ"), ("dk", g.h_kv, "dK"), ("dv", g.h_kv, "dV")):
            got = v[name][lo:hi].view(1, n, h, d)
            assert_close_fp8_grad(
                got.float(),
                refs[name],
                grad_tol["atol"],
                grad_tol["rtol"],
                f"{tag} seq {i} [{lo}:{hi}]",
                keys=n,
                operand=operands[name],
                flip_unit=flip_unit[name],
                intermediates=lambda sel, rb=ref_bwd: rb(sel)[8],
                fp8_dtype=_E4M3,
                out_dtype=torch.bfloat16,
            )
            _report_stage_difference(got, refs[name], f"{tag} seq {i} [{lo}:{hi}]")
            _assert_grad_close(got, refs[name].double(), f"{tag} seq {i} [{lo}:{hi}] vs the row's reference (the bf16 bound form)")
    amax_dp = sc["amax_dp"]
    print(f"amax_dP {amax_dp:.6g} vs the reference's max over the sequences of max|dS| {amax_ref:.6g} (scale_dp {res.scale_dp:g})")
    assert abs(amax_dp - amax_ref) <= amax_tol["atol"] + amax_tol["rtol"] * amax_ref, (amax_dp, amax_ref)


def _assert_m_row_budgeted(tag: str, res, ref: dict) -> dict:
    """The (M) end-to-end in the dense suite's ONE form: the bf16 block's bound with the SDPA stage's flip class propagated and budgeted
    by ROWS (``1e-5 x rows x keys``, at least 1) on ``dh / dw_qkvg / dw_o`` -- every output, both GQA fold paths included (the row folds
    its per-Q-head partials from fp32, rounding once like the reference) -- never widened.  Returns the per-output report."""
    m = _print_end_to_end(tag, res.grads, ref, keys=_row_keys(res))
    assert m, tag
    over = {
        n: (m[n]["rows_outside"], m[n]["rows"], m[n]["row_budget"]) for n in ("dh", "dw_qkvg", "dw_o") if n in m and m[n]["rows_outside"] > m[n]["row_budget"]
    }
    assert not over, f"{tag}: (M) rows outside the bf16 bound exceed the 1e-5 x rows x keys row budget (rows outside, rows, budget): {over}"
    return m


def _assert_seeded_under_the_bf16_bound(tag: str, res, ref: dict) -> dict:
    """Downstream of the SDPA stage: ``dh / dw_o`` vs the oracle SEEDED with the block's own per-sequence dQ / dK / dV under the bf16
    block's bound, ``dW_norm`` under the noise bound with the combined mass; ``dw_qkvg`` printed (the (M) row budget is its pin: the slab's
    single near-amax e4m3 flips land there, one weight row each)."""
    worst = {}
    for name in ("dh", "dw_o"):
        if res.grads[name] is not None:
            worst[name] = _assert_grad_close(res.grads[name], ref[name], f"{tag} {name} vs the seeded oracle")
    if res.grads["dw_qkvg"] is not None:
        worst["dw_qkvg (printed)"] = _report_close(res.grads["dw_qkvg"], ref["dw_qkvg"], f"{tag} dw_qkvg vs the seeded oracle")
    for name in ("dw_q_norm", "dw_k_norm"):
        if res.grads[name] is not None:
            worst[name] = _assert_dw_norm_close(res.grads[name], ref[name], ref[name + "_mass"], f"{tag} {name} vs the seeded oracle")
    print(f"{tag}: worst cells (fraction of the bound) {worst}")
    return worst


# ---------------------------------------------------------------------------
# Host pins -- any CUDA device, no compile
# ---------------------------------------------------------------------------


def _declare_fp8_thd_bwd(lens=_LENS, *, causal=True, h_kv=2, quant="spec", record_kw=None, **bwd_kw):
    """A DECLARED (not compiled) packed fp8 backward over a packed fp8 record placeholder (e4m3 ``h``, bf16 activations, the lengths
    tensor and its form) with a unit-scale ``QuantSpec`` (a declaration needs no calibration); ``quant`` = ``"spec"`` / an object / None."""
    geom_kw = _geom_kw(causal=causal, h_kv=h_kv)
    g = GatedAttentionBlockGeometry(**geom_kw)
    inp, meta = make_packed_inputs(RefGeometry(**geom_kw), lens)
    inp8, desc = quantize_block_inputs(inp)
    spec = QuantSpec(**desc, scale_q=1.0, scale_k=1.0, scale_v=1.0, scale_o=1.0)
    saved = _alloc_packed_saved(g, inp8, meta, seq_lens=meta["seq_lens"], form="lengths", act_dtype=torch.bfloat16)
    if record_kw:
        saved = dataclasses.replace(saved, **record_kw)
    out = torch.empty(1, meta["t"], g.d_model, device="cuda", dtype=torch.bfloat16)
    dy = _make_dy(out)
    kw = dict(_thd_kw(meta))
    kw.update(bwd_kw)
    if quant == "spec":
        kw["quant"] = spec
    elif quant is not None:
        kw["quant"] = quant
    blk = _declare_fp8_bwd(dy, saved, inp8, g, **kw)
    return SimpleNamespace(blk=blk, inp=inp8, spec=spec, saved=saved, dy=dy, meta=meta, geom=g, geom_kw=geom_kw)


@requires_cuda
def test_thd_fp8_backward_declares_the_packed_fp8_chain():
    """``GatedAttentionBlockBwd(thd=True, quant=QuantSpec)`` CONSTRUCTS (the construction decline is the MXFP8 arm's only) with the
    dense fp8 stage list at ``B = 1, S = T`` and the SDPA stage's PACKED declaration over the fp8 row: ``thd`` and the envelope
    passed through, ``external_delta=True`` with the adapter's packed head-major ``[1, H_q, ceil128(T)]`` delta shape (the dense
    layout at ``B = 1, S = T``), ``amax_dP`` requested, both packed totals ``T``, Stats declared ``(B, H_q, S_max, 1)``, e4m3
    payload samples and bf16 gradient samples -- read off the declared stage without a compile or a device read."""
    r = _declare_fp8_thd_bwd()
    blk, g, t, b, s_max = r.blk, r.geom, r.meta["t"], r.meta["b"], r.meta["max_seq_len"]
    with _no_device_sync():
        assert blk.thd and blk.quant is r.spec and (blk.batch, blk.seq_len) == (1, t) and (blk.num_sequences, blk.max_seq_len) == (b, s_max)
        assert blk.act_dtype == torch.bfloat16 and blk.w_dtype == _E4M3
        st = blk._sdpa
        assert type(st).__name__ == "_SdpaBwdFp8" and st.thd and (st.num_sequences, st.max_seq_len, st.cu_seqlens) == (b, s_max, False)
        assert type(blk._prologue).__name__ == "_QuantPrologue" and type(blk._epilogue).__name__ == "_QuantEpilogue"
        assert blk._gate_bwd.want_delta and blk._gate_bwd.seq_len == t, "the gate backward's delta at s = T IS the packed delta"
        impl = st._ensure_impl()
        assert impl.thd is True and impl.external_delta is True and impl.seq_kv_lens_present is False and impl.amax_requested == frozenset({"amax_dP"})
        assert (impl.max_total_seq_len_q, impl.max_total_seq_len_kv) == (t, t)
        t_pad = -(-t // 128) * 128
        assert tuple(impl.external_delta_shape) == st.delta_shape == (1, g.h_q, t_pad), "the packed delta: the dense layout at B = 1, S = T"
        assert tuple(int(x) for x in impl.stats_desc.shape) == (b, g.h_q, s_max, 1), "Stats declared over the envelope (B, H_q, S_max, 1)"
        assert impl.q_desc.dtype == impl.k_desc.dtype == impl.v_desc.dtype == impl.o_desc.dtype == impl.do_desc.dtype == _E4M3
        assert impl.dq_desc.dtype == impl.dk_desc.dtype == impl.dv_desc.dtype == torch.bfloat16
        assert tuple(int(x) for x in impl.q_desc.shape) == (b, g.h_q, s_max, g.d_head), "the payload samples carry the envelope"


@requires_cuda
def test_thd_fp8_record_contracts_are_typed():
    """The packed fp8 record AS WRITTEN (e4m3 ``h``, bf16 activations, ``seq_lens`` + ``seq_lens_form``) passes ``_check_saved_record``
    at declaration and at execute under ``quant`` (``h_dtype`` = the e4m3 codes); a record claiming the DENSE form (``seq_lens_form=None``)
    is refused by the packed fp8 backward naming the form, one without its lengths naming ``saved.seq_lens``, a bf16-``h`` record under
    ``quant`` naming the quantized forward's record -- each BEFORE any device read; ``seq_lens_present`` with ``thd`` is the mutual
    exclusion.  On Rubin the as-written record's ``check_support`` passes; elsewhere it stops at the arch gate."""
    r = _declare_fp8_thd_bwd()
    g, t, b, dev = r.geom, r.meta["t"], r.meta["b"], r.dy.device
    deq_h = torch.empty(1, t, g.d_model, dtype=torch.bfloat16, device="cuda")
    # The declarations (device ops: the inputs are drawn and quantized) sit OUTSIDE the sync guard; every CHECK inside it.
    r_dense_form = _declare_fp8_thd_bwd(record_kw=dict(seq_lens_form=None))
    r_no_lengths = _declare_fp8_thd_bwd(record_kw=dict(seq_lens=None))
    r_present = _declare_fp8_thd_bwd(seq_lens_present=True)
    with _no_device_sync():
        for at in ("declaration", "execute"):
            proj, o_flat = _check_saved_record(r.saved, g, 1, t, torch.bfloat16, dev, at=at, thd=True, num_sequences=b, cu_seqlens=False, h_dtype=_E4M3)
            assert proj.data_ptr() == r.saved.proj_slab.data_ptr() and o_flat.data_ptr() == r.saved.o.data_ptr()
        with pytest.raises(ValueError, match="seq_lens_form is None"):
            r_dense_form.blk.check_support()
        with pytest.raises(ValueError, match=r"SavedForBackward\.seq_lens must be"):
            r_no_lengths.blk.check_support()
        with pytest.raises(ValueError, match="e4m3 codes"):
            _check_saved_record(dataclasses.replace(r.saved, h=deq_h), g, 1, t, torch.bfloat16, dev, at="declaration", thd=True, num_sequences=b, h_dtype=_E4M3)
        with pytest.raises(ValueError, match="mutually exclusive"):
            r_present.blk.check_support()
        if _cc() == _SM107:
            assert r.blk.check_support()
        else:
            with pytest.raises(NotImplementedError, match="Rubin"):
                r.blk.check_support()


@requires_cuda
def test_thd_fp8_backward_declines_an_mxquantspec():
    """``thd=True`` with an ``MxQuantSpec`` stays the typed decline at construction, naming both attributes and the two reasons (no
    packed MXFP8 training record; the SDPA-layout MX quantizes have no packed per-sequence arm) and pointing at the served packed
    per-tensor fp8 backward; the adapter's text never surfaces."""
    with pytest.raises(ValueError, match="thd") as ei:
        _declare_fp8_thd_bwd(quant=MxQuantSpec(descale_w_o=0.03, scale_o=1.0))
    msg = str(ei.value)
    assert "MxQuantSpec" in msg and "QuantSpec" in msg and "scale-factor" in msg, msg
    assert "external_delta" not in msg, msg


# ---------------------------------------------------------------------------
# Accept -- Rubin only
# ---------------------------------------------------------------------------


@requires_rubin
@_MATRIX
def test_thd_fp8_gradients_match_the_per_sequence_modelled_oracle(causal, h_kv):
    """``(300, 128, 200)`` B=3 (a tail tile on every sequence) through the packed fp8 training record, causal and dense, GQA 8/2 and
    MHA 8/8, at the calibrated ``scale_dp``: the dense fp8 suite's layers at ``B = 1, S = T`` -- the quantizers, every scalar and the
    delta BITWISE (the delta the chain's own ``dot_do_o`` over the packed bf16 O / dO with an exactly-zero tail); the SDPA stage's bf16
    dQ / dK / dV of EVERY sequence under the fp8 row's recipe and the bf16 bound form, ``amax_dP`` the max over the sequences; the (M)
    end-to-end row-budgeted on ``dh / dw_qkvg / dw_o`` against the per-sequence modelled oracle (``dh`` per sequence, the weight
    gradients the SUM over the sequences); ``dh / dw_o / dW_norm`` vs the seeded per-sequence oracle under the bf16 block's bound.
    Magnitudes are printed on every cell; no bound is this module's own."""
    res = _backward_fp8_thd(_LENS, causal=causal, h_kv=h_kv)
    assert res.blk.thd and res.blk._sdpa.thd and res.blk._sdpa._impl.thd and res.blk._sdpa._impl.external_delta is True
    for name, ten in res.grads.items():
        if ten is not None:
            assert torch.isfinite(ten.float()).all(), f"{name}: non-finite cells"
    v = _assert_quantizers_scalars_delta_bitwise(res)
    _assert_sdpa_stage_per_sequence(res, v)
    tag = f"thd fp8 {tuple(res.lens)} {'causal' if causal else 'dense'} h_kv={h_kv}"
    _assert_m_row_budgeted(f"{tag} (M)", res, _oracle_m_packed(res))
    _assert_seeded_under_the_bf16_bound(tag, res, _oracle_m_packed(res, seeded=True))


@requires_rubin
def test_thd_fp8_delayed_replays_current_bitwise():
    """A ``grad_scaling="delayed"`` packed block fed the "current" run's READ-BACK ``scale_dy / scale_do / scale_dqkvg`` gives
    ``torch.equal`` gradients and an equal scalar block (the dense pin, packed)."""
    res = _backward_fp8_thd(_LENS)
    sc = res.scalars
    blk, ws, grads = _twin_fp8_thd(res, grad_scaling="delayed", scales=dict(scale_dy=sc["scale_dy"], scale_do=sc["scale_do"], scale_dqkvg=sc["scale_dqkvg"]))
    assert blk.grad_scaling == "delayed" and blk.thd
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: the delayed replay differs from the current run"
    assert torch.equal(_scalar_block(blk, ws), _scalar_block(res.blk, res.ws))


@requires_rubin
def test_thd_fp8_b1_bwd_is_bitwise_the_dense_fp8_block():
    """``B = 1`` packed ``(512,)`` against the dense ``B=1, S=512`` fp8 backward over the SAME bytes (the e4m3 inputs, the QuantSpec, the
    record the two training forwards wrote, the same dy and ``scale_dp``): the two training records bitwise first (slab, O, LSE, rstd,
    the forward's e4m3 ``q8 / k8 / v8 / o8``), then every intermediate of the backward -- the SDPA stage's ``dq / dk / dv`` BEFORE anything
    downstream, so a difference is localised to the row or to the block -- the scalar block, the delta and every gradient ``torch.equal``.
    ``S % 128 == 0``: the dense chain has no staging pads.  A difference is a finding to investigate at the first differing stage, never a
    tolerance."""
    t = 512
    res = _backward_fp8_thd((t,), causal=True, h_kv=2)
    inp8, spec, g = res.inp, res.spec, res.geom
    # --- the dense fp8 TRAINING forward over the same e4m3 inputs and spec ---------------------------------------------------
    out_d = torch.empty(1, t, g.d_model, device="cuda", dtype=torch.bfloat16)
    fwd_d = GatedAttentionBlockFwd(
        inp8["h"], inp8["w_qkvg"], inp8["w_q_norm"], inp8["w_k_norm"], inp8["cos"], inp8["sin"], inp8["w_o"], out_d, g, quant=spec, save_for_backward=True
    )
    saved_d = _alloc_saved(g, inp8, 1, t, save_mode="proj_slab", act_dtype=torch.bfloat16)
    fwd_d.check_support()
    fwd_d.compile()
    ws_fd = torch.empty(fwd_d.get_workspace_size(), dtype=torch.uint8, device="cuda")
    fwd_d.execute(inp8["h"], inp8["w_qkvg"], inp8["w_q_norm"], inp8["w_k_norm"], inp8["cos"], inp8["sin"], inp8["w_o"], out_d, ws_fd, saved=saved_d)
    torch.cuda.synchronize()
    assert torch.equal(out_d, res.out), "the forward output differs between packed B=1 and dense"
    for name, a, b_ in (("proj_slab", res.saved.proj_slab, saved_d.proj_slab), ("o", res.saved.o, saved_d.o), ("lse", res.saved.lse, saved_d.lse)):
        assert torch.equal(a, b_), f"saved.{name}: the packed B=1 training record differs from the dense one"
    if g.qk_norm:
        assert torch.equal(res.saved.rstd_q, saved_d.rstd_q) and torch.equal(res.saved.rstd_k, saved_d.rstd_k)
    lay_p, lay_d = res.fwd.blk._layout(), fwd_d._layout()
    for name, h in (("q8", g.h_q), ("k8", g.h_kv), ("v8", g.h_kv), ("o8", g.h_q)):
        off_p, off_d = getattr(lay_p, name, -1), getattr(lay_d, name, -1)
        if off_p >= 0 and off_d >= 0:
            a = _view(res.fwd.ws, off_p, (t, h, g.d_head), _E4M3).view(torch.uint8)
            b_ = _view(ws_fd, off_d, (t, h, g.d_head), _E4M3).view(torch.uint8)
            assert torch.equal(a, b_), f"forward {name}: the packed B=1 forward's codes differ from the dense forward's"
    # --- the dense fp8 backward: the same dy, the packed run's calibrated scale_dp, a poisoned workspace -----------------------
    blk_d = _declare_fp8_bwd(res.dy, saved_d, inp8, g, quant=spec)
    assert not blk_d.thd
    blk_d.check_support()
    blk_d.compile()
    ws_d = torch.empty(blk_d.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(0xFF)
    grads_d = _alloc_grads(blk_d, fill=float("nan"))
    _execute_fp8(blk_d, inp8, saved_d, res.dy, grads_d, ws_d, scale_dp=res.scale_dp_t)
    torch.cuda.synchronize()
    v_p, v_d = _slots(res), _slots(SimpleNamespace(blk=blk_d, ws=ws_d, geom=g))
    first_diff = None
    # the stage order of the backward: the pre-SDPA intermediates, the SDPA stage's outputs, then everything downstream
    for name in ("dy8", "do", "do8", "og8", "q8", "k8", "v8", "dq", "dk", "dv", "dqkvg", "dqkvg8"):
        a, b_ = v_p[name], v_d[name]
        if a is None or b_ is None:
            assert a is None and b_ is None, name
            continue
        same = torch.equal(a.view(torch.uint8), b_.view(torch.uint8)) if a.dtype == _E4M3 else torch.equal(a, b_)
        if not same and first_diff is None:
            first_diff = name
        print(f"{name}: {'bitwise' if same else 'DIFFERS'} (packed B=1 vs dense)")
    d_p, d_d = _delta(res), _view(ws_d, blk_d._layout().delta, tuple(blk_d._sdpa.delta_shape), torch.float32)
    print(
        f"delta: {'bitwise' if torch.equal(d_p, d_d) else 'DIFFERS'}; scalar block: {'bitwise' if torch.equal(_scalar_block(res.blk, res.ws), _scalar_block(blk_d, ws_d)) else 'DIFFERS'}"
    )
    assert (
        first_diff is None
    ), f"the first differing intermediate between the packed B=1 and the dense fp8 backward is {first_diff} -- a finding, not a tolerance"
    assert torch.equal(d_p, d_d), "the delta differs between packed B=1 and dense"
    assert torch.equal(_scalar_block(res.blk, res.ws), _scalar_block(blk_d, ws_d)), "the scalar block differs between packed B=1 and dense"
    for name, ten in res.grads.items():
        if ten is not None:
            d = (ten.float() - grads_d[name].float()).abs().max().item()
            assert torch.equal(ten, grads_d[name]), f"{name}: packed B=1 differs from the dense fp8 backward (max|diff| {d:.3e}) -- a finding, not a tolerance"


@requires_rubin
def test_thd_fp8_launch_count_is_honest():
    """CUPTI kernel records of one packed fp8 backward == the launch table recomputed from the adapter's own facts: the block's own
    launches (one per stage but the SDPA's: the fused prologue, the two quantizes, the four e4m3 GEMMs, the gate backward, the norm
    backward, the fused epilogue -- 10 with every gradient) + the fp8 row's packed chain ``2 + [zero-fill] + c x (own setup + main +
    (patch + dK) + q x (patch + dQ)) + 1`` -- the THD metadata setup AND the amax resets (two launches: under THD the dense row's kv-length
    fill is gone, the resets stay their own launch), NO ``dot`` (the delta is the gate backward's), the fold launch (dV, and dK under GQA)
    on every group; ``q`` read off the dQ record.  MEASURED on the first run: 19 kernels at the test geometry (``(300, 128, 200)``, GQA 8/2,
    c = 1, q = 1, no zero-fill) = 10 + 9, and recorded in the module docstring of the API; the names are printed; a typed skip when CUPTI
    records nothing on this node; no hidden memcpy and no memset.  Profiled over ``execute`` ALONE (the buffers exist before the
    profiled region)."""
    from torch.profiler import ProfilerActivity, profile

    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import _dq_launches

    res = _backward_fp8_thd(_LENS)
    blk, g = res.blk, res.geom
    impl = blk._sdpa._impl
    assert impl.external_delta is True and impl.thd is True
    grp = g.h_q // g.h_kv
    c = -(-g.h_q // impl._qh_chunk)
    dq = _dq_launches(grp, impl._dq_b_head_group)
    chain = 2 + (1 if impl._zero_ws else 0) + c * (1 + 1 + 2 + 2 * dq) + 1  # THD setup + amax resets, [zero-fill], per chunk, the fold
    block_own = len(blk._stages) - 1
    formula = block_own + chain
    grads = _alloc_grads(blk)
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, res.ws, scale_dp=res.scale_dp_t)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _execute_fp8(blk, res.inp, res.saved, res.dy, grads, res.ws, scale_dp=res.scale_dp_t)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    memsets = [n for n in names if "memset" in n.lower()]
    memcpys = [n for n in names if "memcpy" in n.lower()]
    kernels = [n for n in names if n not in memsets and n not in memcpys]
    print(
        f"\n{len(kernels)} kernels (formula {formula}: {block_own} block + {chain} chain; c={c}, q={dq}, zero_ws={impl._zero_ws}, grp={grp}), "
        f"{len(memsets)} memsets, {len(memcpys)} memcpys:\n  " + "\n  ".join(names)
    )
    assert not memcpys, f"a hidden copy on the execute path: {memcpys}"
    assert not memsets, f"a hidden memset on the execute path (the scalar init and the row's fills are kernels): {memsets}"
    assert len(kernels) == formula, (len(kernels), formula, kernels)
