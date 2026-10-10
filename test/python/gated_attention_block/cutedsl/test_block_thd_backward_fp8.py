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
output, the (M) end-to-end in the row-budgeted form (``1e-5 x rows x keys``) on ``dh / dw_qkvg``, the seeded oracle under the
bf16 block's bound on ``dh / dW_norm``, ``dw_o`` on both layers in the dense suite's e4m3 FLIP-CLASS form (``_assert_dw_o_flip_structured``:
every cell outside the bf16 bound in a column an ``og8`` flip touched, the residual after the flips' exact rank-1 term inside the bound,
the flip class itself capped -- the og8 cast's one-code flips are COLUMN events a row count at a short reduction does not describe; a
kernel-vs-kernel reference keeps ``dw_o`` row-budgeted) and, on ``dw_qkvg``, in the row-budgeted form with the dense suite's flip ATTRIBUTION
(every row outside the bound a ``dqkvg8`` flip's, its pre-cast slab column inside its band's bound, and -- the magnitude guard -- the whole ``dW_qkvg`` within the GEMM suite's bound
of its own dequantized codes) -- where that budget is its FLOOR
(fewer than 20 tokens: the 5-token cell) the ``dw_qkvg`` row count is REPORTED, not asserted, and the attribution carries the pin on
both layers (``_row_budget_floored``: the cast's flip class leaves a handful of rows outside at every length, so a floor of one row is
not its bound); the quantizers, every scalar
and the delta BITWISE (the dense suite's layer, unchanged at ``B = 1, S = T``).  ``B = 1`` packed is pinned ``torch.equal`` the dense
fp8 backward over the same bytes (a difference is a
finding to investigate at the SDPA stage first, never a tolerance); ``grad_scaling="delayed"`` replays the current run bitwise;
the launch count is MEASURED (CUPTI) against the formula from the adapter's own facts.

The sweep.  Uniform ``B = 4`` packed vs the dense ``B = 4`` fp8 backward over the same bytes is the SPLIT pin: the token-wise
stages bitwise, the SDPA-derived ones reported in stage order -- MEASURED bitwise on every intermediate and gradient at both
shapes (the padded-mask arm's per-sequence bounds and the per-sequence kv-blocked workspace coincide with the dense tile walk at
uniform lengths); the split form is the convention, so a reordering of the per-sequence trim or the bounded fold would surface as a
reported difference -- the gradients kernel-vs-kernel in the row-budgeted form.  Zero-length sequences
(a middle one, a first one, two trailing ones behind a 5-token sequence; each packing under both mask arms at GQA 8/2, the MHA
fold path on two of them) run under the suite's ``timeout`` (a zero-length
sequence is a ``seq_kv_len == 0`` entry for the per-tensor fp8 d256 kernels): finite, the live sequences under the whole
per-sequence chain, every gradient and ``amax_dP`` ``torch.equal`` the same tokens packed WITHOUT the empty sequence, and a
twin over a 0xFF workspace with NaN-filled gradients bitwise the clean run (no unwritten tile is read, nothing leaks).  The
dense fp8 suite's contracts hold packed: two executes and a fresh block bitwise under every knob set, ``fuse_wgrad_overlap``
and ``fuse_gate_bwd`` (inert under quant) bitwise, the lengths and the prefix record forms bitwise, a CUDA graph replaying
bitwise -- over a NEW packing and a NEW ``scale_dp`` written through the captured pointers too -- and the workspace size exact
(every e4m3 region and the delta written in full, nothing allocated on the execute path).  The remaining rejects are typed
before any device read.

Accept tests are ``requires_rubin``; the host tests build CUDA tensors for a DECLARED backward (``requires_cuda``, no compile).
"""

import dataclasses
import gc
import os
import sys
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block packed fp8 backward tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QuantSpec  # noqa: E402
from cudnn.gated_attention_block.api import MxQuantSpec, _cols, _view  # noqa: E402
from cudnn.gated_attention_block.api_bwd import _check_saved_record  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    RefGeometry,
    assert_packing_contract,
    cu_seqlens_of,
    gated_attention_block_fp8_bwd_reference,
    make_packed_inputs,
    packed_rope_tables,
    quantize_block_inputs,
    sequence_slices,
)
from test_block_backward import _KNOBS, _alloc_grads, _assert_dw_norm_close, _assert_grad_close, _make_dy  # noqa: E402
from test_block_backward_fp8 import (  # noqa: E402
    _E4M3,
    _api_const,
    _assert_dw_o_flip_structured,
    _assert_dw_qkvg_floor_guard,
    _assert_quantizers_scalars_delta_bitwise,
    _assert_seeded_dw_qkvg_row_budgeted,
    _calibrated_scale_dp,
    _declare_fp8_bwd,
    _declare_then_check,
    _delta,
    _dev_scalar,
    _execute_fp8,
    _floored_flip_fixture,
    _print_end_to_end,
    _report_seeded_intermediates,
    _report_stage_difference,
    _row_budget_floored,
    _row_keys,
    _row_tol,
    _rows_outside_mask,
    _slots,
    _test_python_root,
)
from test_block_thd import _COMMON, _LENS, _alloc_packed_saved, _no_device_sync, _run_fp8_thd, _thd_kw  # noqa: E402
from test_block_training_forward import _alloc_saved  # noqa: E402

_SM107 = (10, 7)
# The oracle's per-sequence slab bands (fp64 ``[n, H, D]``) and ``og8`` (e4m3), packed back row by row by ``_oracle_m_packed`` -- the
# evidence the dense suite's seeded dW_qkvg attribution (``_report_seeded_intermediates`` / ``_assert_seeded_dw_qkvg_row_budgeted``)
# reads at ``t = B * S = T`` rows.
_PACKED_BANDS = ("dq_pre", "dg", "dk_pre", "dv_band", "og8")


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


def _twin_fp8_thd(res, *, poison=0xFF, grad_scaling=None, scales=None, saved=None, cu=False, **bwd_kw):
    """A second packed fp8 block over the SAME record / dy / inputs / scale_dp as ``res`` (different knobs or recipe), compiled and
    run once into a poisoned workspace and NaN-filled gradients; returns ``(blk, ws, grads)`` -- the bitwise comparand of ``res``.
    ``saved`` (appended) substitutes a record over the same bytes (the prefix form of the same packing), ``cu`` the matching
    ``cu_seqlens`` constructor fact."""
    saved = res.saved if saved is None else saved
    blk = _declare_fp8_bwd(
        res.dy, saved, res.inp, res.geom, quant=res.spec, grad_scaling=grad_scaling or res.grad_scaling, **_thd_kw(res.meta, cu=cu), **bwd_kw
    )
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(poison)
    grads = _alloc_grads(blk, fill=float("nan"))
    scale_ts = {k: _dev_scalar(v) for k, v in (scales or {}).items()} if scales is not None else res.scale_ts
    _execute_fp8(blk, res.inp, saved, res.dy, grads, ws, scale_dp=res.scale_dp_t, **scale_ts)
    torch.cuda.synchronize()
    return blk, ws, grads


def _dense_fp8_run(res, b: int, s: int):
    """The DENSE per-tensor fp8 training forward and backward at ``(b, s)`` over the SAME bytes as the packed run ``res``
    (``b * s == T``): the e4m3 inputs viewed ``[b, s, .]`` -- at uniform lengths the packed per-token RoPE tables, positions
    restarting at every sequence, ARE the dense ``[b, s, rope]`` table -- the QuantSpec, the packed run's ``dy`` and its calibrated
    ``scale_dp``; the backward into a 0xFF workspace and NaN-filled gradients.  Returns the forward block and workspace, the dense
    record and output, the backward block, workspace and gradients."""
    inp8, spec, g, t = res.inp, res.spec, res.geom, b * s
    assert t == res.seq_len, (b, s, res.seq_len)
    inp_d = dict(inp8, h=inp8["h"].view(b, s, g.d_model), cos=inp8["cos"].view(b, s, -1), sin=inp8["sin"].view(b, s, -1))
    out_d = torch.empty(b, s, g.d_model, device="cuda", dtype=torch.bfloat16)
    fwd = GatedAttentionBlockFwd(
        inp_d["h"],
        inp_d["w_qkvg"],
        inp_d["w_q_norm"],
        inp_d["w_k_norm"],
        inp_d["cos"],
        inp_d["sin"],
        inp_d["w_o"],
        out_d,
        g,
        quant=spec,
        save_for_backward=True,
    )
    saved = _alloc_saved(g, inp_d, b, s, save_mode="proj_slab", act_dtype=torch.bfloat16)
    fwd.check_support()
    fwd.compile()
    ws_f = torch.empty(fwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
    fwd.execute(inp_d["h"], inp_d["w_qkvg"], inp_d["w_q_norm"], inp_d["w_k_norm"], inp_d["cos"], inp_d["sin"], inp_d["w_o"], out_d, ws_f, saved=saved)
    torch.cuda.synchronize()
    dy_d = res.dy.view(b, s, -1)
    blk = _declare_fp8_bwd(dy_d, saved, inp_d, g, quant=spec)
    assert not blk.thd and (blk.batch, blk.seq_len) == (b, s)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(0xFF)
    grads = _alloc_grads(blk, fill=float("nan"))
    _execute_fp8(blk, inp_d, saved, dy_d, grads, ws, scale_dp=res.scale_dp_t)
    torch.cuda.synchronize()
    return SimpleNamespace(fwd=fwd, ws_f=ws_f, saved=saved, out=out_d, inp=inp_d, blk=blk, ws=ws, grads=grads)


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
    ``sqrt(sum mass_i^2)``, ``amax_dp`` is the max over the sequences; ``per_seq`` keeps each sequence's dict; the slab bands
    ``dq_pre / dg / dk_pre / dv_band`` and ``og8`` are packed back row by row (``_PACKED_BANDS``: the seeded attribution's evidence).
    ``seeded`` substitutes the block's own bf16 dQ / dK / dV rows per sequence (``amax_dp`` is then None)."""
    g, t, sc, d = res.geom, res.seq_len, res.scalars, res.geom.d_head
    v = _slots(res)
    gate = _gate_rows(res)
    delta = _delta(res)
    ref_geom = RefGeometry(**res.geom_kw)
    s_scale = 2.0 ** _api_const("FP8_SCALE_S_LOG2")
    per_seq, total, amax_dp = [], None, 0.0
    dh = torch.zeros(1, t, g.d_model, dtype=torch.float64, device=res.dy.device)
    bands = {k: [] for k in _PACKED_BANDS}
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
        for k in _PACKED_BANDS:
            bands[k].append(None if o.get(k) is None else o[k].reshape(n, -1))
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
    for k in _PACKED_BANDS:
        # [T, H * D] at the packed rows (e4m3 og8 concatenated through its byte view): the element order the dense attribution
        # helpers reshape from
        if any(x is None for x in bands[k]):
            total[k] = None
        elif bands[k][0].dtype == _E4M3:
            total[k] = torch.cat([x.view(torch.uint8) for x in bands[k]], dim=0).view(_E4M3)
        else:
            total[k] = torch.cat(bands[k], dim=0)
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


def _assert_m_row_budgeted(tag: str, res, ref: dict, flip_ev: Optional[dict] = None, v: Optional[dict] = None) -> dict:
    """The (M) end-to-end in the dense suite's ONE form: the bf16 block's bound with the SDPA stage's flip class propagated and budgeted
    by ROWS (``1e-5 x rows x keys``, at least 1) on ``dh / dw_qkvg`` -- both GQA fold paths included (the row folds its per-Q-head
    partials from fp32, rounding once like the reference) -- never widened.  Where ``dw_qkvg``'s budget is its FLOOR
    (``_row_budget_floored``: fewer than 20 tokens) the count is REPORTED and every row outside is held to the cast's flip class instead
    -- a row a ``dqkvg8`` flip touched, its pre-cast slab column inside its band's bound (the dense suite's attribution conditions (2)
    and (3), read from ``flip_ev`` = ``_report_seeded_intermediates``'s evidence, REQUIRED there) AND to the dense suite's magnitude guard
    (4): the whole ``dW_qkvg`` within the GEMM suite's bound of ``dqkvg8^T . h8`` on the block's own codes (``v`` = the intermediates,
    REQUIRED there; ``_assert_dw_qkvg_floor_guard`` -- the residual form needs the oracle's codes and is the seeded layer's) -- the honest
    form of the pin at a length whose proportional budget does not describe the class; ``dh`` keeps the row-budgeted form at every
    length.  ``dw_o`` takes the dense suite's e4m3 flip-class form (``_assert_dw_o_flip_structured``: the og8 cast's one-code flips are
    COLUMN events, so a row count at a short reduction does not describe them) against an ORACLE reference, which carries the ``og8``
    the form attributes by; a kernel-vs-kernel reference (the uniform packed ``B = 4`` block against the dense ``B = 4`` block) has no
    ``og8`` and keeps ``dw_o`` row-budgeted as before.  Returns the per-output report."""
    m = _print_end_to_end(tag, res.grads, ref, keys=_row_keys(res))
    assert m, tag
    floored = "dw_qkvg" in m and _row_budget_floored(m["dw_qkvg"]["rows"], _row_keys(res)["dw_qkvg"])
    # dW_o: the e4m3 flip-class form at every length -- against an ORACLE reference, which carries the og8 the form attributes by; a
    # kernel-vs-kernel reference (the uniform packed B=4 block vs the dense B=4 block) has no og8 and keeps the row-budgeted form
    flip_form = res.grads.get("dw_o") is not None and ref.get("og8") is not None
    budgeted = ("dh",) if floored else ("dh", "dw_qkvg")
    if res.grads.get("dw_o") is not None and not flip_form:
        budgeted = budgeted + ("dw_o",)
    over = {n: (m[n]["rows_outside"], m[n]["rows"], m[n]["row_budget"]) for n in budgeted if n in m and m[n]["rows_outside"] > m[n]["row_budget"]}
    assert not over, f"{tag}: (M) rows outside the bf16 bound exceed the 1e-5 x rows x keys row budget (rows outside, rows, budget): {over}"
    if flip_form:
        _assert_dw_o_flip_structured(res, _slots(res) if v is None else v, ref, f"{tag} dw_o")
    if floored:
        assert flip_ev is not None, f"{tag}: the dw_qkvg row budget is its floor here -- the flip evidence is required for the attribution form"
        rows_out = torch.nonzero(_rows_outside_mask(res.grads["dw_qkvg"], ref["dw_qkvg"])).flatten()
        unexplained = rows_out[~torch.isin(rows_out, flip_ev["dw_qkvg_rows"].to(rows_out.device))]
        band_worst = flip_ev["band_col_worst"].to(rows_out.device)[rows_out]  # each row's PRE-cast slab column: worst cell / its band's bound
        not_the_casts = rows_out[band_worst > 1.0]
        print(
            f"{tag} dw_qkvg: {int(rows_out.numel())} of {m['dw_qkvg']['rows']} rows outside the bf16 bound at a FLOORED row budget "
            f"({m['dw_qkvg']['row_budget']:.3g}) -- the count is reported; {int(unexplained.numel())} untouched by a dqkvg8 flip, "
            f"{int(not_the_casts.numel())} with the pre-cast slab column outside its band's bound"
        )
        assert unexplained.numel() == 0, (
            f"{tag}: (M) dw_qkvg rows {unexplained.tolist()} are outside the bf16 bound and no dqkvg8 flip touched them (not the cast's flip "
            f"class) -- rows outside {rows_out.tolist()}"
        )
        assert not_the_casts.numel() == 0, (
            f"{tag}: (M) dw_qkvg rows {not_the_casts.tolist()} are outside the bf16 bound and their PRE-cast slab columns are themselves outside "
            f"the band's bound against the seeded oracle ({[round(x, 3) for x in band_worst[band_worst > 1.0].tolist()]} of it): a miss upstream "
            f"of the cast, not the cast's flip class -- rows outside {rows_out.tolist()}"
        )
        assert v is not None, f"{tag}: the dw_qkvg row budget is its floor here -- the block's dqkvg8 codes are required for the magnitude guard"
        _assert_dw_qkvg_floor_guard(res, v, res.grads["dw_qkvg"], rows_out, None, f"{tag} dw_qkvg")
    return m


def _assert_seeded_under_the_bf16_bound(tag: str, res, ref: dict, v: Optional[dict] = None, flip_ev: Optional[dict] = None) -> dict:
    """Downstream of the SDPA stage, the dense suite's seeded layer at ``B = 1, S = T``: ``dh`` vs the oracle SEEDED with the block's
    own per-sequence dQ / dK / dV under the bf16 block's bound, ``dw_o`` in the e4m3 flip-class form (``_assert_dw_o_flip_structured``),
    ``dW_norm`` under the noise bound with the combined mass, and
    ``dw_qkvg`` in the ROW-BUDGETED form WITH its attribution (``_assert_seeded_dw_qkvg_row_budgeted`` over the packed slab: the rows
    with a cell outside the bound within ``1e-5 x rows x keys``, EVERY such row one a ``dqkvg8`` flip touched, its PRE-cast slab column
    inside its band's bound -- the cast's rounding, not a band's miss; the flip evidence from ``_report_seeded_intermediates`` over the
    packed bands, which also pins the slab's V band bitwise the block's own dV slot).  ``v`` = the materialised intermediates
    (``_slots(res)`` when omitted); ``flip_ev`` = that evidence when the caller computed it already (shared with the (M) layer's
    floored form), else it is computed here."""
    v = _slots(res) if v is None else v
    worst = {}
    if res.grads["dh"] is not None:
        worst["dh"] = _assert_grad_close(res.grads["dh"], ref["dh"], f"{tag} dh vs the seeded oracle")
    if res.grads["dw_o"] is not None:
        worst["dw_o"] = _assert_dw_o_flip_structured(res, v, ref, f"{tag} dw_o vs the seeded oracle")  # the e4m3 flip-class form
    if res.grads["dw_qkvg"] is not None:
        flip_ev = _report_seeded_intermediates(res, v, ref) if flip_ev is None else flip_ev
        worst["dw_qkvg"] = _assert_seeded_dw_qkvg_row_budgeted(res, v, ref, flip_ev, f"{tag} dw_qkvg vs the seeded oracle")
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
    """``thd=True`` with an ``MxQuantSpec`` stays the typed decline at construction, naming both attributes and the one remaining
    reason (the backward's SDPA-layout MX quantizes run the quantizer's dense arm only; the packed MXFP8 training record exists)
    and pointing at the served packed per-tensor fp8 backward; the adapter's text never surfaces."""
    with pytest.raises(ValueError, match="thd") as ei:
        _declare_fp8_thd_bwd(quant=MxQuantSpec(descale_w_o=0.03, scale_o=1.0))
    msg = str(ei.value)
    assert "MxQuantSpec" in msg and "QuantSpec" in msg and "scale-factor" in msg, msg
    assert "external_delta" not in msg, msg


def test_thd_fp8_floored_m_form_guard_rejects_a_corrupted_row():
    """The (M) layer's tiny-``T`` form has the dense suite's magnitude guard (host, no device): on ``_floored_flip_fixture``'s inputs --
    two legitimate single-step ``dqkvg8`` flips, exactly their two ``dW_qkvg`` rows outside the bf16 bound at a row budget of one --
    ``_assert_m_row_budgeted`` passes with the block's codes handed in, REQUIRES them at the floor, and REJECTS the same inputs with the
    two flip-touched rows replaced by 1e6 (the GEMM bound on the block's own codes), which the attribution conditions alone accept."""
    res, v, ref, flip_ev = _floored_flip_fixture()
    assert _row_budget_floored(res.grads["dw_qkvg"].shape[0], _row_keys(res)["dw_qkvg"]), "the fixture must sit at the floor"
    _assert_m_row_budgeted("floor control (M)", res, ref, flip_ev, v=v)
    with pytest.raises(AssertionError, match="codes are required"):
        _assert_m_row_budgeted("floor control (M) without the codes", res, ref, flip_ev)
    bad = SimpleNamespace(**{**vars(res), "grads": {"dw_qkvg": res.grads["dw_qkvg"].clone()}})
    bad.grads["dw_qkvg"][[7, 4000]] = 1e6
    with pytest.raises(AssertionError, match="magnitude guard"):
        _assert_m_row_budgeted("floor mutant (M)", bad, ref, flip_ev, v=v)


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
    end-to-end row-budgeted on ``dh / dw_qkvg`` against the per-sequence modelled oracle (``dh`` per sequence, the weight gradients
    the SUM over the sequences) and ``dw_o`` in the e4m3 flip-class form; ``dh / dW_norm`` vs the seeded per-sequence oracle under the
    bf16 block's bound, ``dw_o`` in the flip-class form again,
    ``dw_qkvg`` in the row-budgeted form with its flip attribution (every row outside a ``dqkvg8`` flip's, its pre-cast slab column
    inside its band's bound).
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
    _assert_seeded_under_the_bf16_bound(tag, res, _oracle_m_packed(res, seeded=True), v)


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
    g = res.geom
    # --- the dense fp8 TRAINING forward + backward over the same e4m3 inputs, spec, dy and scale_dp -------------------------
    dn = _dense_fp8_run(res, 1, t)
    out_d, saved_d, ws_fd, fwd_d, blk_d, ws_d, grads_d = dn.out, dn.saved, dn.ws_f, dn.fwd, dn.blk, dn.ws, dn.grads
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
    # --- the dense fp8 backward (the same dy, the packed run's calibrated scale_dp, a poisoned workspace), stage by stage ------
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


# ---------------------------------------------------------------------------
# The sweep -- the split pin, zero-length sequences over poisoned buffers, the contracts, the remaining rejects
# ---------------------------------------------------------------------------

_KNOB_SETS = pytest.mark.parametrize("knobs", list(_KNOBS.values()), ids=list(_KNOBS))
# The zero-length cells: three packings, each under BOTH mask arms (causal / dense: a ``seq_kv_len == 0`` entry meets every ``MASK_FLAGS``
# arm the block serves at d = 256) at GQA 8/2, and the MHA 8/8 fold path (``grp = 1``: the dV-only fold) on two of them.
_EMPTY_SHAPES = {"middle_empty": ((300, 0, 200), 300), "first_empty": ((0, 256, 128), 256), "trailing_empties_T5": ((5, 0, 0), 5)}
_EMPTY_ARMS = [
    ("middle_empty", True, 2),
    ("middle_empty", False, 2),
    ("middle_empty", True, 8),
    ("first_empty", True, 2),
    ("first_empty", False, 2),
    ("first_empty", False, 8),
    ("trailing_empties_T5", True, 2),
    ("trailing_empties_T5", False, 2),
]
_EMPTY_CELLS = pytest.mark.parametrize(
    "lens, s_max, causal, h_kv",
    [(*_EMPTY_SHAPES[k], c, h) for k, c, h in _EMPTY_ARMS],
    ids=[f"{k}-{'causal' if c else 'dense'}-{'mha' if h == 8 else f'gqa_8_{h}'}" for k, c, h in _EMPTY_ARMS],
)
# The backward's intermediates split by what they depend on: the token-wise ones (the same launches over the same tokens: bitwise
# between a uniform packing and the dense block) and the ones that read the forward's O (bitwise exactly when the two forwards' O
# is); everything downstream of the SDPA backward (``dq / dk / dv``, ``dqkvg / dqkvg8``, the gradients) is reported or budgeted.
_TOKEN_WISE = ("dy8", "do", "do8", "q8", "k8", "v8")
_O_DEPENDENT = ("og8", "dg", "delta")


def _packed_view_of_dense(name: str, ten: torch.Tensor) -> torch.Tensor:
    """A dense block's ``[B, ...]`` tensor in the packed arm's element order: ``saved.lse`` / ``delta`` are ``[B, H_q, S]`` and the
    packed arm's ``[1, H_q, T]`` is head-major, so they permute; everything else flattens its batch axis onto the token axis."""
    if name in ("saved.lse", "delta"):
        return ten.permute(1, 0, 2).reshape(1, ten.shape[1], -1)
    return ten.reshape(1, -1, *ten.shape[2:])


@requires_rubin
@pytest.mark.parametrize("s", [128, 256], ids=["one_tile_each", "two_q_tiles_each"])
def test_thd_fp8_uniform_b4_bwd_matches_the_dense_fp8_block_per_sequence(s):
    """Uniform ``B = 4`` packed ``(s,)*4`` against the dense ``B=4, S=s`` fp8 backward over the SAME bytes (the e4m3 inputs, the
    QuantSpec, the dy, the packed run's calibrated ``scale_dp``) at ``s = 128`` (one q tile and one kv block per sequence: the
    ``B*H > 1`` with ``n_kv == 1`` shape of the invariants) and ``s = 256`` (two q tiles per sequence under the per-sequence stage-3
    trim) -- the SPLIT pin: the token-wise stages BITWISE (the record's ``proj_slab`` / ``rstd``; the backward's ``dy8``, the gated
    ``dO``, ``do8``, the recomputed ``q8 / k8 / v8``: the same launches over the same tokens), the SDPA-derived ones REPORTED in
    stage order with the first difference named -- ``O`` / ``LSE`` / ``out``, the O-dependent ``og8`` / ``dG`` / delta (asserted
    bitwise whenever the two forwards' O is), then ``dq / dk / dv`` and ``dqkvg / dqkvg8``: whether the packed chain's per-sequence
    trim and bounded fold reorder anything against the dense chain is MEASURED here, never assumed -- and the gradients ``dh /
    dw_qkvg / dw_o`` against the dense block's in the dense suite's row-budgeted form (kernel vs kernel: the (M) budget ``1e-5 x
    rows x keys``, ``dW_norm`` printed).  The pin keeps the split form by convention: a reordering is a reported difference, a
    budget miss a finding.  The packed arm also runs the matrix cell's whole per-sequence chain at each shape."""
    b = 4
    res = _backward_fp8_thd((s,) * b, causal=True, h_kv=2)
    g, t = res.geom, b * s
    d = _dense_fp8_run(res, b, s)
    assert torch.equal(d.saved.proj_slab, res.saved.proj_slab), "proj_slab differs: the token-wise stage-(1) GEMM is not the same launch"
    if g.qk_norm:
        assert torch.equal(res.saved.rstd_q, d.saved.rstd_q.reshape(1, t, -1)) and torch.equal(res.saved.rstd_k, d.saved.rstd_k.reshape(1, t, -1))
    report = []

    def verdict(name, a, b_):
        same = torch.equal(a.view(torch.uint8), b_.view(torch.uint8)) if a.dtype == _E4M3 else torch.equal(a, b_)
        md = 0.0 if same else (a.float() - b_.float()).abs().max().item()
        report.append((name, same, md))
        print(f"{name}: {'bitwise' if same else f'DIFFERS (max|diff| {md:.3e})'} (packed uniform B=4 vs dense B=4)")
        return same

    verdict("saved.o", res.saved.o, _packed_view_of_dense("saved.o", d.saved.o))
    verdict("saved.lse", res.saved.lse, _packed_view_of_dense("saved.lse", d.saved.lse))
    verdict("out", res.out, d.out.reshape(1, t, -1))
    v_p, v_d = _slots(res), _slots(SimpleNamespace(blk=d.blk, ws=d.ws, geom=g))
    for name in ("dy8", "do", "do8", "og8", "q8", "k8", "v8", "dg", "dq", "dk", "dv", "dqkvg", "dqkvg8"):
        a, b_ = v_p[name], v_d[name]
        if a is None or b_ is None:
            assert a is None and b_ is None, name
            continue
        same = verdict(name, a, b_)
        if name in _TOKEN_WISE:
            assert same, f"{name}: a token-wise stage differs between the packed uniform B=4 and the dense B=4 fp8 backward -- a finding, not a tolerance"
    delta_d = _view(d.ws, d.blk._layout().delta, tuple(d.blk._sdpa.delta_shape), torch.float32)
    verdict("delta", _delta(res), _packed_view_of_dense("delta", delta_d))
    o_same = report[0][1]
    for name, same, _md in report:
        if name in _O_DEPENDENT and o_same:
            assert same, f"{name}: the two forwards' O is bitwise, so this O-dependent token-wise stage must be too -- a finding, not a tolerance"
    first = next(((n, md) for n, same, md in report if not same), None)
    print(f"\nuniform B=4 vs dense: first SDPA-derived difference {first} (None = bitwise)")
    ref_d = {k: (ten.reshape(res.grads[k].shape).double() if ten is not None else None) for k, ten in d.grads.items()}
    for name, ten in res.grads.items():
        if ten is not None:
            assert torch.isfinite(ten.float()).all() and torch.isfinite(d.grads[name].float()).all(), name
    _assert_m_row_budgeted(f"thd fp8 uniform ({s},)*4 vs the dense B=4 fp8 block", res, ref_d)
    # the packed arm under the matrix cell's whole chain at this shape
    v = _assert_quantizers_scalars_delta_bitwise(res)
    _assert_sdpa_stage_per_sequence(res, v)
    tag = f"thd fp8 {(s,) * b} causal h_kv=2"
    _assert_m_row_budgeted(f"{tag} (M)", res, _oracle_m_packed(res))
    _assert_seeded_under_the_bf16_bound(tag, res, _oracle_m_packed(res, seeded=True), v)


@requires_rubin
@_EMPTY_CELLS
def test_thd_fp8_zero_length_sequences(lens, s_max, causal, h_kv):
    """Zero-length sequences in the packed fp8 backward -- a middle one ``(300, 0, 200)``, a first one ``(0, 256, 128)``, two
    trailing ones behind a 5-token sequence ``(5, 0, 0)`` at ``S_max = 5`` (the shortest legal sequence; never 1) -- each packing
    under both mask arms (causal and dense) at GQA 8/2, the MHA 8/8 fold path (``grp = 1``, the dV-only fold) on two of them.  A
    zero-length sequence is a ``seq_kv_len == 0`` entry for the per-tensor fp8 d256 kernels, so the cell runs under the suite's ``timeout``
    (a hang is a barrier-table finding, never a raised limit; if it ever flakes, count exit codes over >= 8 fresh processes).
    Asserted: every gradient finite; the live sequences under the whole per-sequence chain (the quantizers / scalars / delta
    bitwise, the SDPA stage per sequence under the row recipe, the (M) row budget, the seeded layer with the ``dw_qkvg``
    attribution -- at the 5-token cell the ``dw_qkvg`` row budget is its FLOOR, so its count is REPORTED and every row outside is
    held to the cast's flip class on both layers, ``_row_budget_floored``); the NEIGHBOURS EXACT -- the same tokens packed WITHOUT
    the empty sequence (same ``T``, same bytes, same
    calibrated QuantSpec and ``scale_dp``) give a bitwise record, ``torch.equal`` gradients, an equal scalar block and the SAME
    ``amax_dP`` (an empty sequence contributes nothing); and a twin over a 0xFF workspace with NaN-filled gradients bitwise the
    clean run on every gradient, on the SDPA stage's slots and on the delta (no unwritten tile is read, nothing leaks -- the
    unwritten-tile hazard of the per-sequence stage-3 trim)."""
    res = _backward_fp8_thd(lens, causal=causal, h_kv=h_kv, max_seq_len=s_max)
    assert 0 in res.lens and res.blk.thd and res.blk._sdpa._impl.external_delta is True
    assert res.geom.is_causal is bool(causal) and res.geom.h_kv == h_kv and res.geom.h_q // res.geom.h_kv == (1 if h_kv == 8 else 4)
    for name, ten in res.grads.items():
        if ten is not None:
            assert torch.isfinite(ten.float()).all(), f"{name}: non-finite cells"
    v = _assert_quantizers_scalars_delta_bitwise(res)
    _assert_sdpa_stage_per_sequence(res, v)
    tag = f"thd fp8 {tuple(lens)} s_max={s_max} {'causal' if causal else 'dense'} h_kv={h_kv}"
    ref_m, ref_seeded = _oracle_m_packed(res), _oracle_m_packed(res, seeded=True)
    flip_ev = _report_seeded_intermediates(res, v, ref_seeded)  # computed once: the (M) layer's floored form and the seeded layer share it
    _assert_m_row_budgeted(f"{tag} (M)", res, ref_m, flip_ev, v=v)
    _assert_seeded_under_the_bf16_bound(tag, res, ref_seeded, v, flip_ev)
    # the neighbours exact: the same tokens packed without the empty sequence(s)
    live = tuple(n for n in lens if n)
    twin = _backward_fp8_thd(live, causal=causal, h_kv=h_kv, max_seq_len=s_max)
    assert twin.spec == res.spec and twin.scale_dp == res.scale_dp, (twin.spec, res.spec, twin.scale_dp, res.scale_dp)
    assert torch.equal(twin.inp["h"].view(torch.uint8), res.inp["h"].view(torch.uint8)) and torch.equal(twin.dy, res.dy)
    for name, a, b_ in (("proj_slab", res.saved.proj_slab, twin.saved.proj_slab), ("o", res.saved.o, twin.saved.o), ("lse", res.saved.lse, twin.saved.lse)):
        assert torch.equal(a, b_), f"saved.{name}: the empty sequence changed the live sequences' record -- a finding, not a tolerance"
    assert res.scalars["amax_dp"] == twin.scalars["amax_dp"], f"amax_dP changed by the empty sequence: {res.scalars['amax_dp']} vs {twin.scalars['amax_dp']}"
    assert torch.equal(_scalar_block(res.blk, res.ws), _scalar_block(twin.blk, twin.ws)), "the scalar block differs with the empty sequence"
    for name, ten in res.grads.items():
        if ten is not None:
            md = (ten.float() - twin.grads[name].float()).abs().max().item()
            assert torch.equal(
                ten, twin.grads[name]
            ), f"{name}: the empty sequence changed a live sequence's gradient (max|diff| {md:.3e}) -- a finding, not a tolerance"
    # the poisoned twin: 0xFF workspace, NaN-filled gradients
    blk_p, ws_p, grads_p = _twin_fp8_thd(res)
    for name, ten in grads_p.items():
        if ten is not None:
            assert torch.isfinite(ten.float()).all(), f"{name}: non-finite cells over a poisoned workspace"
            assert torch.equal(ten, res.grads[name]), f"{name}: the poisoned twin differs from the clean run"
    poisoned = SimpleNamespace(blk=blk_p, ws=ws_p, geom=res.geom)
    v_pz = _slots(poisoned)
    for name in ("dq", "dk", "dv"):
        assert torch.isfinite(v_pz[name].float()).all() and torch.equal(v_pz[name], v[name]), f"{name}: the SDPA stage's slot differs over a poisoned workspace"
    assert torch.equal(_delta(poisoned), _delta(res)) and torch.equal(_scalar_block(blk_p, ws_p), _scalar_block(res.blk, res.ws))


@requires_rubin
@_KNOB_SETS
def test_thd_fp8_two_runs_are_bitwise(knobs):
    """Two executes of the SAME packed fp8 block over the same record / dy / scale_dp -- the second into a workspace poisoned 0xFF and
    NaN-filled gradients -- are ``torch.equal`` on every gradient AND on the scalar block, and so is a FRESH block over the same
    record (compiled anew, poisoned the same way), under every knob set (``fuse_gate_bwd`` is inert under quant; ``fuse_wgrad_overlap``
    moves B1 / B7 to the side stream): the packed chain's metadata and dS workspace included, nothing an execute reads survives from
    the previous one."""
    res = _backward_fp8_thd(_LENS, **knobs)
    assert res.blk.fuse_wgrad_overlap is bool(knobs.get("fuse_wgrad_overlap", False)) and res.blk.fuse_gate_bwd is bool(knobs.get("fuse_gate_bwd", False))
    sb1 = _scalar_block(res.blk, res.ws).clone()
    ws2 = torch.empty_like(res.ws).fill_(0xFF)
    grads2 = _alloc_grads(res.blk, fill=float("nan"))
    _execute_fp8(res.blk, res.inp, res.saved, res.dy, grads2, ws2, scale_dp=res.scale_dp_t, **res.scale_ts)
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.isfinite(ten.float()).all() and torch.equal(
                ten, res.grads[name]
            ), f"{name}: a second execute of the same packed block differs (knobs={knobs})"
    assert torch.equal(_scalar_block(res.blk, ws2), sb1), "the scalar block differs between two executes of one packed block"
    blk, ws, grads = _twin_fp8_thd(res, **knobs)
    for name, ten in grads.items():
        if ten is not None:
            assert torch.isfinite(ten.float()).all() and torch.equal(ten, res.grads[name]), f"{name}: two packed runs differ (knobs={knobs})"
    assert torch.equal(_scalar_block(blk, ws), sb1), "the scalar block differs between two packed runs"


@requires_rubin
def test_thd_fp8_fuse_wgrad_overlap_is_bitwise_the_in_order_block():
    """``fuse_wgrad_overlap=True`` under THD is a scheduling knob: the side stream exists, every gradient and the scalar block
    ``torch.equal`` the in-order packed block's over the same record (the workspace poisoned, the gradients NaN-filled first)."""
    res = _backward_fp8_thd(_LENS)
    blk, ws, grads = _twin_fp8_thd(res, fuse_wgrad_overlap=True)
    assert blk.thd and blk.fuse_wgrad_overlap and blk._side is not None
    for name, ten in grads.items():
        if ten is not None:
            assert torch.isfinite(ten.float()).all() and torch.equal(ten, res.grads[name]), f"{name}: fuse_wgrad_overlap differs from the in-order packed block"
    assert torch.equal(_scalar_block(blk, ws), _scalar_block(res.blk, res.ws))


@requires_rubin
def test_thd_fp8_fuse_gate_bwd_is_inert_under_quant():
    """The external delta is MANDATORY under quant, so ``fuse_gate_bwd`` has no second arm packed either: both values construct, the
    stage's adapter carries ``external_delta=True`` either way, and the gradients, the scalar block and the delta region are
    ``torch.equal``."""
    res = _backward_fp8_thd(_LENS)
    assert not res.blk.fuse_gate_bwd and res.blk._sdpa._impl.external_delta is True
    blk, ws, grads = _twin_fp8_thd(res, fuse_gate_bwd=True)
    assert blk.fuse_gate_bwd and blk._sdpa._impl.external_delta is True
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: fuse_gate_bwd changed the packed gradients under quant"
    assert torch.equal(_scalar_block(blk, ws), _scalar_block(res.blk, res.ws)) and torch.equal(_delta(SimpleNamespace(blk=blk, ws=ws)), _delta(res))


@requires_rubin
@pytest.mark.parametrize("cu_base", [0, 100], ids=["prefix", "prefix_nonzero_base"])
def test_thd_fp8_lengths_and_prefix_forms_are_bitwise(cu_base):
    """The same packing through the ``[B]`` lengths record and the ``[B+1]`` prefix record (base 0 and base 100: a prefix tensor
    sliced from a larger one) gives ``torch.equal`` gradients and an equal scalar block; the prefix block declares ``cu_seqlens`` on
    the block and on the fp8 SDPA stage."""
    res = _backward_fp8_thd(_LENS)
    cu_t = torch.tensor(cu_seqlens_of(res.lens, base=cu_base), dtype=torch.int32, device="cuda")
    assert_packing_contract(cu_t, res.seq_len, res.meta["max_seq_len"], res.meta["b"], cu=True)
    saved_cu = dataclasses.replace(res.saved, seq_lens=cu_t, seq_lens_form="prefix")
    blk, ws, grads = _twin_fp8_thd(res, saved=saved_cu, cu=True)
    assert blk.thd and blk.cu_seqlens and blk._sdpa.cu_seqlens and saved_cu.seq_lens_form == "prefix" and cu_t.numel() == res.meta["b"] + 1
    for name, ten in grads.items():
        if ten is not None:
            assert torch.isfinite(ten.float()).all() and torch.equal(ten, res.grads[name]), f"{name} differs between the lengths and the prefix form"
    assert torch.equal(_scalar_block(blk, ws), _scalar_block(res.blk, res.ws)), "the scalar block differs between the lengths and the prefix form"


@requires_rubin
def test_thd_fp8_cuda_graph_replay_with_new_lengths_and_scale():
    """One packed fp8 backward captured into a CUDA graph on a side torch stream replays bitwise the eager run (the capture itself
    launches nothing); a replay over a NEW packing -- new lengths and RoPE tables written through the record's own tensors, the
    forward re-run eagerly over them so the record matches (same ``B``, ``sum == T``, each ``<= max_seq_len``) -- and a NEW
    ``scale_dp`` (the new packing's calibrated one) written through the captured scalar equals a fresh eager backward over it and
    is finite: the setup kernel rebuilds the packed metadata per execute from the device lengths, the prologue's scalar init, the
    amax passes and the quantize publishes are device work.  The new packing then passes the per-sequence layers (the quantizers /
    scalars / delta bitwise, the SDPA stage per sequence under the row recipe, the (M) row budget); a last replay at the DOUBLED
    ``scale_dp`` through the same captured scalar equals eager at that scale (the scalar is read live even where the new packing's
    calibrated value coincides with the old one)."""
    res = _backward_fp8_thd(_LENS, memo=False)
    blk, inp, saved, dy, g, meta = res.blk, res.inp, res.saved, res.dy, res.geom, res.meta
    sdp = res.scale_dp_t.clone()
    ws = torch.empty_like(res.ws)
    grads = _alloc_grads(blk, fill=float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _execute_fp8(blk, inp, saved, dy, grads, ws, scale_dp=sdp)  # warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for ten in grads.values():
        if ten is not None:
            ten.fill_(float("nan"))
    ws.fill_(0xFF)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            _execute_fp8(blk, inp, saved, dy, grads, ws, scale_dp=sdp)
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isnan(ten).all(), f"{name}: the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.equal(ten, res.grads[name]), f"{name}: the replay differs from the eager run"
        assert torch.equal(_scalar_block(blk, ws), _scalar_block(res.blk, res.ws)), "the scalar block differs between the replay and the eager run"
        # a new packing through the captured pointers: new tables + lengths, the forward re-run eagerly into the same record
        new_lens = [200, 300, 128]
        assert_packing_contract(new_lens, meta["t"], meta["max_seq_len"], meta["b"])
        cos2, sin2 = packed_rope_tables(new_lens, g.rope_dim, base=RefGeometry(**res.geom_kw).rope_base)
        inp["cos"].copy_(cos2.view_as(inp["cos"]))
        inp["sin"].copy_(sin2.view_as(inp["sin"]))
        assert saved.seq_lens is res.fwd.seq_lens
        res.fwd.seq_lens.copy_(torch.tensor(new_lens, dtype=torch.int32, device="cuda"))
        res.fwd.blk.execute(
            inp["h"],
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            res.out,
            res.fwd.ws,
            seq_lens=res.fwd.seq_lens,
            saved=saved,
        )
        torch.cuda.synchronize()
        # the new packing's calibrated scale_dp (the chart recipe over one eager run at 1.0 into scratch buffers), through the captured scalar
        ws_cal = torch.empty_like(ws)
        _execute_fp8(blk, inp, saved, dy, _alloc_grads(blk), ws_cal, scale_dp=_dev_scalar(1.0))
        new_scale = _calibrated_scale_dp(blk, ws_cal)
        print(f"\nnew packing {new_lens}: scale_dp {res.scale_dp:g} -> {new_scale:g}")
        sdp.fill_(new_scale)
        ws.fill_(0xFF)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        ref = _alloc_grads(blk, fill=float("nan"))
        ws_ref = torch.empty_like(ws).fill_(0xFF)
        _execute_fp8(blk, inp, saved, dy, ref, ws_ref, scale_dp=_dev_scalar(new_scale))
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isfinite(ten.float()).all() and torch.equal(ten, ref[name]), f"{name}: the replay over the new packing differs from eager"
        assert torch.equal(_scalar_block(blk, ws), _scalar_block(blk, ws_ref)), "the scalar block differs between the replay over the new packing and eager"
        res2 = SimpleNamespace(**vars(res))
        res2.ws, res2.grads, res2.lens, res2.meta = ws, grads, new_lens, dict(meta, lens=new_lens)
        res2.scale_dp, res2.scale_dp_t = float(new_scale), sdp
        res2.scalars = {k: float(x.item()) for k, x in blk.quant_scalars(ws).items()}
        v = _assert_quantizers_scalars_delta_bitwise(res2)
        _assert_sdpa_stage_per_sequence(res2, v)
        _assert_m_row_budgeted(f"thd fp8 graph replay over the new packing {new_lens} (M)", res2, _oracle_m_packed(res2))
        # the captured scalar is live even where the calibrated value coincides: the doubled scale_dp through the same pointer
        sdp.fill_(new_scale * 2.0)
        ws.fill_(0xFF)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        ref2 = _alloc_grads(blk, fill=float("nan"))
        ws_ref2 = torch.empty_like(ws).fill_(0xFF)
        _execute_fp8(blk, inp, saved, dy, ref2, ws_ref2, scale_dp=_dev_scalar(new_scale * 2.0))
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isfinite(ten.float()).all() and torch.equal(ten, ref2[name]), f"{name}: the replay at the doubled scale_dp differs from eager"
        assert torch.equal(_scalar_block(blk, ws), _scalar_block(blk, ws_ref2)), "the scalar block differs between the replay at the doubled scale_dp and eager"
    finally:  # a graph left to the cyclic GC resets itself inside a later test's capture
        graph.reset()


@requires_rubin
@_KNOB_SETS
def test_thd_fp8_workspace_size_is_honest(knobs):
    """``get_workspace_size()`` is exact and never exceeded on the packed fp8 backward: the carve (every quant region present, aligned,
    the scalar block) + the fp8 adapter's PACKED scratch (``scratch_workspace_bytes()``; the block's own ``delta`` region ALWAYS
    present under quant with the adapter's packed ``(1, H_q, ceil128(T))`` shape, ``fuse_gate_bwd`` or not) + the GEMM scratch (the
    max over the four K64 fp8 plans); a buffer 4096 B larger keeps its tail untouched; two executes allocate nothing; every e4m3
    region is WRITTEN in full (no 0xFF byte survives), and so are the fp32 delta region -- its zero tail included -- and the scalar
    block; one byte less is a typed ``ValueError``; bitwise the memoised gradients -- under every knob set."""
    from cudnn.gated_attention_block.api import _WS_ALIGN

    res = _backward_fp8_thd(_LENS, **knobs)
    blk, g, t = res.blk, res.geom, res.seq_len
    size = blk.get_workspace_size()
    lay = blk._layout()
    t_pad = -(-t // 128) * 128
    print(
        f"\nworkspace {size} B; sdpa packed scratch {lay.sdpa_bwd_bytes} B; gemm scratch {lay.gemm_scratch_bytes} B; delta region {lay.delta} (knobs {knobs})"
    )
    assert size == lay.total_bytes and size % _WS_ALIGN == 0
    assert lay.sdpa_bwd_bytes == blk._sdpa.scratch_workspace_bytes() > 0
    assert lay.delta >= 0 and lay.delta_shape == tuple(blk._sdpa._impl.external_delta_shape) == (1, g.h_q, t_pad), "the packed delta region: always under quant"
    assert (
        lay.quant_scalars >= 0
        and lay.quant_scalars % 256 == 0
        and lay.o_gated == -1
        and lay.recompute_v == -1
        and lay.recompute == -1
        and lay.recompute_k == -1
    )
    plans = blk.gemm_plans
    assert len(plans) == 4 and all(p.mma_tile_k_bytes == 64 and p.has_alpha for p in plans.values())
    assert lay.gemm_scratch_bytes == max(p.workspace_bytes for p in plans.values()) >= 1
    for name in ("dy8", "do8", "q8", "k8", "v8", "dqkvg8"):
        assert getattr(lay, name) >= 0 and getattr(lay, name) % _WS_ALIGN == 0, name
    ws = torch.full((size + 4096,), 0xFF, dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws[:size], scale_dp=res.scale_dp_t, **res.scale_ts)
    torch.cuda.synchronize()
    # The allocation pin in the caching allocator's COUNTER form: the cumulative allocation count cannot be lowered by an unrelated
    # release and still rises for a temporary the execute frees before returning; the allocator peak is the second witness.
    gc.collect()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    n0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws[:size], scale_dp=res.scale_dp_t, **res.scale_ts)
    _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws[:size], scale_dp=res.scale_dp_t, **res.scale_ts)
    torch.cuda.synchronize()
    n1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    peak = torch.cuda.max_memory_allocated()
    assert n1 == n0, f"the packed fp8 backward made {n1 - n0} CUDA allocation(s) on the execute path (allocation.all.allocated {n0} -> {n1})"
    assert peak <= live, f"a temporary on the packed fp8 backward's execute path: the allocator peak rose from {live} to {peak} bytes"
    assert torch.equal(ws[size:], torch.full((4096,), 0xFF, dtype=torch.uint8, device="cuda")), "bytes past get_workspace_size() were written"
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), name
    sb = _view(ws[:size], lay.quant_scalars, (len(_api_const("QUANT_SCALAR_SLOTS")),), torch.float32)
    assert torch.isfinite(sb).all(), "a scalar slot was never written (0xFF = NaN)"
    d = g.d_head
    regions = dict(dy8=(lay.dy8, t * g.d_model), do8=(lay.do8, t * g.h_q * d), q8=(lay.q8, t * g.h_q * d), k8=(lay.k8, t * g.h_kv * d))
    regions.update(v8=(lay.v8, t * g.h_kv * d), dqkvg8=(lay.dqkvg8, t * g.n_qkvg))
    if lay.og8 >= 0:
        regions["og8"] = (lay.og8, t * g.h_q * d)
    for name, (off, nbytes) in regions.items():
        survivors = int((ws[off : off + nbytes] == 0xFF).sum())
        assert survivors == 0, f"{name}: {survivors} of {nbytes} e4m3 bytes still hold the 0xFF poison -- never written"
    delta = _view(ws[:size], lay.delta, lay.delta_shape, torch.float32)
    assert torch.isfinite(delta).all(), "a delta element (the zero tail included) was never written (0xFFFFFFFF = NaN)"
    assert torch.equal(delta[:, :, t:], torch.zeros_like(delta[:, :, t:])), "the tail [T, ceil128(T)) of the packed delta is not zero"
    with pytest.raises(ValueError, match="workspace is"):
        _execute_fp8(blk, res.inp, res.saved, res.dy, grads, ws[: size - 1], scale_dp=res.scale_dp_t, **res.scale_ts)


@requires_cuda
def test_thd_fp8_rejects_are_typed_before_any_device_read():
    """The remaining rejects of the packed fp8 backward -- each the dense decline, reached through the SAME code path under ``thd``,
    before any device read: an fp16 ``dy`` under ``quant`` (the quantized record is bf16) names ``dy``; a bf16 ``saved.h`` under
    ``quant`` through the BLOCK names the e4m3 codes the quantized forward's record carries; an e5m2 ``QuantSpec`` surfaces the
    spec's own decline; ``need_*`` all False leaves no work.  The declarations' inputs are drawn outside the sync guard, every
    decline fires inside it."""
    r = _declare_fp8_thd_bwd()
    t, g = r.meta["t"], r.geom
    dy16 = r.dy.to(torch.float16)
    deq_h = torch.empty(1, t, g.d_model, dtype=torch.bfloat16, device="cuda")
    e5 = dataclasses.replace(r.spec, dtype=torch.float8_e5m2)
    kw = dict(quant=r.spec, **_thd_kw(r.meta))
    with _no_device_sync():
        with pytest.raises((ValueError, NotImplementedError), match="dy"):
            _declare_then_check(lambda: _declare_fp8_bwd(dy16, r.saved, r.inp, g, **kw))
        with pytest.raises(ValueError, match="e4m3 codes"):
            _declare_then_check(lambda: _declare_fp8_bwd(r.dy, dataclasses.replace(r.saved, h=deq_h), r.inp, g, **kw))
        with pytest.raises(NotImplementedError, match="QuantSpec|e5m2|quant"):
            _declare_then_check(lambda: _declare_fp8_bwd(r.dy, r.saved, r.inp, g, quant=e5, **_thd_kw(r.meta)))
        with pytest.raises(ValueError, match="need_dh"):
            _declare_then_check(
                lambda: _declare_fp8_bwd(r.dy, r.saved, r.inp, g, need_dh=False, need_dw_qkvg=False, need_dw_o=False, need_dw_norms=False, **kw)
            )
