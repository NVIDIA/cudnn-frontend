# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The UNFUSED block backward (``GatedAttentionBlockBwd``: bf16 / fp16, proj_slab save mode) against fp64 autograd.

Every stage is individually tested (``test_proj_gemm_bwd.py``, ``test_sigmoid_gate_bwd.py``,
``test_qk_norm_rope_bwd.py``, the sdpa bwd suite); what is under test here is the ASSEMBLY -- the
workspace carve, the compact-vs-band handovers (B4 writes compact slots, the elementwise kernel writes
the ``dqkvg`` bands), the stage order, the one launch stream -- and the block's typed contract.

The oracle is ``gated_attention_block_reference(..., acc_dtype=torch.float64)`` on fp64 COPIES of the
bf16 inputs with ``requires_grad`` on ``h`` / ``w_qkvg`` / ``w_q_norm`` / ``w_k_norm`` / ``w_o``,
differentiated with ``torch.autograd.grad`` -- which also yields the gradients w.r.t. the oracle's
exposed intermediates (``q_pre`` / ``k_pre`` / ``gate`` / ``v`` / ``o``), so a failure localises to
B3 / B4 / B5+B6 instead of "the block". TF32 is pinned off and printed.

Tolerances -- an INFERRED start ("calibrate on the first Rubin run, never widen"), CALIBRATED ONCE
from the first two full Rubin runs (cc 10.7, 212 SMs, SM clock locked at 2376 MHz; identical numbers on both), magnitudes
printed on every cell, never widened again:

* bf16 gradients ``rtol 2^-6``, ``atol 2^-7 * max|ref|``, cosine >= 0.999 -- MEASURED worst cells 0.40-0.66 of the
  bound over every accept cell (dh 0.51-0.66, dW_qkvg 0.40-0.59, dW_o 0.41-0.51; cos >= 0.99998). The chain rounds
  to bf16 at the saved slab, dO, dQ / dK / dV, the bands and the outputs, on top of the SDPA backward's own bf16 dS --
  hence a bound twice the GEMM drivers'.
* the fp32 ``dW_norm`` is an fp32 SUM over ``T*H`` rows of products of two bf16-rounded factors (``g = RoPE^T(dQ)``
  and ``x_hat``), so its noise is ABSOLUTE and scales with the column's ``sqrt(sum(terms^2))`` (``mass``) -- not with
  ``max|ref|``: a near-cancelling column legitimately deviates by ~2e-2 of the terms' mass while ``cos`` stays
  0.99998. The inferred ``rtol 1e-2 / atol 1e-3 * max|ref|`` read that as a 2.3-6.1x miss on 11 of 11 norm
  cells (max|diff| 0.009-0.026 against max|ref| 1.9-5.2; cos 0.99998-0.99999); the bound here is
  ``|diff| <= _DW_NORM_NOISE * mass + 1e-2 * |ref|`` with ``mass`` from the fp64 oracle's own per-row terms. MEASURED
  ``|diff| / mass`` over the 11 norm cells: 0.0156-0.0195 (max 0.01949, the swa640 dense dW_k_norm; the spread is
  the max-of-256-columns statistic of B4's bf16-dS rounding, consistent with dh's ~2^-7 per-element noise), so
  ``_DW_NORM_NOISE = 2^-5`` (0.03125, 1.6x the largest measured ratio). Every cell still prints its ratio and the
  inferred bound's fraction for the record. ``test_stage_localisation_via_autograd_grad`` pins the MECHANISM: rebuilt
  in fp64 from the block's OWN bf16 dQ / dK slots, the saved rstd and the slab bands, ``dW_norm`` matches the kernel
  + reduce at fp32 tightness (``1e-4 * mass + 1e-3 * |ref|``), so the whole deviation from the fp64 oracle is B4's
  bf16 rounding, not B5+B6 or the reduce.
* the fp16 arm (``test_gradients_match_fp64_autograd_fp16``) runs the same chain at ``_RTOL[float16]`` /
  ``_ATOL_FRAC[float16]`` (8x the bf16 tightness: three more mantissa bits at every rounding point) -- MEASURED (Rubin cc 10.7,
  S=256, B=1): dh 0.50, dW_qkvg 0.56, dW_o 0.45 of the bound, cos 1.000000; its dW_norm ``|diff| / mass``
  0.0019-0.0023 (8x below bf16's, the same three bits), 0.05-0.06 of the noise bound.

A DENSE ``S % 128 != 0`` has no training record: the FORWARD's SDPA row declines it typed (its KV tail would be
unmasked on the SM100 DSL), so the two dense S=1000 cells of the S sweep pin that decline instead of a gradient.

Accept tests are ``requires_rubin`` (the block binds ONE engine, the Rubin d256 backward -- AGENTS.md
Rule 9); reject tests build CUDA tensors for a DECLARED block (``requires_cuda``, no compile);
the pure-carve tests run anywhere. ``torch.exp2`` / ``torch.log2`` are deliberately absent from the
tolerance helpers (they fail through nvrtc on cc 10.7).
"""

import dataclasses
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import (
    GatedAttentionBlockBwd,
    GatedAttentionBlockGeometry,
    RecomputePolicy,
    SavedForBackward,
    gated_attention_block_backward,
)  # noqa: E402
from cudnn.gated_attention_block.api import _WS_ALIGN, _cols, _view  # noqa: E402
from cudnn.gated_attention_block.api_bwd import _BwdIntermediates, _GemmStage, _plan_bwd_workspace  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import RefGeometry, gated_attention_block_reference  # noqa: E402
from gated_block_stream_probe import park_the_default_stream  # noqa: E402
from test_block_training_forward import _alloc_saved, _declare, _run_training  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


# The REGISTERED marker of cutedsl/conftest.py (the skip is applied at collection).
requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_COMMON = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)  # g = 4
_GEOM_397B = dict(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64)
_QK_NORM = pytest.mark.parametrize("qk_norm", [True, False], ids=["norm", "rope_only"])
_CAUSAL = pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
_FORCED_TILE = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"

# The bounds (module docstring): one per output dtype, tensor-scaled atol, plus a cosine floor.
# bf16: worst cells 0.40-0.66 of the bound over every accept cell; fp16 (8x tighter): 0.45-0.56 -- both MEASURED on Rubin cc 10.7.
_RTOL = {torch.bfloat16: 2.0**-6, torch.float16: 2.0**-9}
_ATOL_FRAC = {torch.bfloat16: 2.0**-7, torch.float16: 2.0**-10}
_COS_MIN = 0.999
_DW_NORM_RTOL = 1e-2  # the systematic part (a wrong scale / a missing term shows up as a RELATIVE miss on the large cells)
# |diff| per unit of sqrt(sum(terms^2)) -- the bf16 noise floor of the reduction. CALIBRATED ONCE (Rubin cc 10.7, 212 SMs) from the
# measured 0.0156-0.0195 over 11 norm cells (two identical runs): 2^-5 = 0.03125 leaves a 1.6x margin. Never widen.
_DW_NORM_NOISE = 2.0**-5
_DW_NORM_SPEC_ATOL_FRAC = 1e-3  # the initially INFERRED atol (as a fraction of max|ref|): printed for the record, no longer asserted


@pytest.fixture(autouse=True)
def _no_tf32():
    """An fp32 torch reference is a TF32 reference on Blackwell+ unless pinned (``allow_tf32``); the oracle is fp64 but the pin is printed anyway."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"\nallow_tf32={torch.backends.cuda.matmul.allow_tf32}")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _assert_grad_close(got: torch.Tensor, ref64: torch.Tensor, what: str, *, rtol=None, atol_frac=None, cos_min=_COS_MIN) -> float:
    """``assert_close`` at ``rtol`` / ``atol = atol_frac * max|ref|`` plus a cosine floor; prints the magnitude (worst cell as a
    fraction of the bound) so a calibration or a regression is a number, never a silent widening."""
    rtol = _RTOL[got.dtype] if rtol is None else rtol
    atol_frac = _ATOL_FRAC[got.dtype] if atol_frac is None else atol_frac
    got64, ref64 = got.detach().double(), ref64.detach().double()
    assert torch.isfinite(got64).all(), f"{what}: non-finite cells"
    ref_max = ref64.abs().max().item()
    atol = atol_frac * ref_max
    diff = (got64 - ref64).abs()
    worst = (diff / (atol + rtol * ref64.abs())).max().item()
    cos = _cos(got64, ref64)
    msg = f"{what}: max|diff|={diff.max().item():.4g} max|ref|={ref_max:.4g} worst cell {worst:.3f} of the bound (rtol={rtol}, atol={atol_frac}*max|ref|) cos={cos:.6f} {got.dtype}"
    print(msg)
    assert cos >= cos_min, msg
    torch.testing.assert_close(got64, ref64, rtol=rtol, atol=atol, msg=msg)
    return worst


# ---------------------------------------------------------------------------
# Building a backward: the training forward supplies the record exactly as a user would
# ---------------------------------------------------------------------------


def _make_dy(out: torch.Tensor, seed: int = 1) -> torch.Tensor:
    g = torch.Generator(device=out.device).manual_seed(seed)
    return torch.randn(out.shape, generator=g, device=out.device, dtype=torch.float32).to(out.dtype)


def _alloc_grads(blk: GatedAttentionBlockBwd, fill=None) -> dict:
    g, b, s, dev, act = blk.geom, blk.batch, blk.seq_len, blk.device, blk.act_dtype

    def buf(*shape, dt=act):
        t = torch.empty(*shape, dtype=dt, device=dev)
        if fill is not None:
            t.fill_(fill)
        return t

    return dict(
        dh=buf(b, s, g.d_model) if blk.need_dh else None,
        dw_qkvg=buf(g.n_qkvg, g.d_model) if blk.need_dw_qkvg else None,
        dw_o=buf(g.d_model, g.h_q * g.d_head) if blk.need_dw_o else None,
        dw_q_norm=buf(g.d_head, dt=torch.float32) if blk.need_dw_norms else None,
        dw_k_norm=buf(g.d_head, dt=torch.float32) if blk.need_dw_norms else None,
    )


def _execute(blk, inp, saved, dy, grads, ws, *, seq_lens=None, current_stream=None):
    blk.execute(
        dy,
        saved,
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        workspace=ws,
        seq_lens=seq_lens,
        current_stream=current_stream,
        **grads,
    )


def _fp64_oracle(inp: dict, geom_kw: dict, dy: torch.Tensor) -> dict:
    """fp64 autograd through the oracle on fp64 copies of the (bf16-rounded) inputs: the five gradients, plus the
    gradients w.r.t. the exposed intermediates the block materialises (the ``dqkvg`` bands and dO) for localisation."""
    g64 = RefGeometry(**geom_kw)
    leaf = lambda x: None if x is None else x.detach().double().requires_grad_(True)  # noqa: E731
    h, w_qkvg, w_q, w_k, w_o = (leaf(inp[k]) for k in ("h", "w_qkvg", "w_q_norm", "w_k_norm", "w_o"))
    ref = gated_attention_block_reference(h, w_qkvg, w_q, w_k, inp["cos"].double(), inp["sin"].double(), w_o, geom=g64, acc_dtype=torch.float64)
    wanted = [h, w_qkvg, w_o] + ([w_q, w_k] if g64.qk_norm else []) + [ref.q_pre, ref.k_pre, ref.gate, ref.v, ref.o, ref.q, ref.k]
    grads = list(torch.autograd.grad(ref.out, wanted, dy.double()))
    out = dict(dh=grads.pop(0), dw_qkvg=grads.pop(0), dw_o=grads.pop(0))
    out.update(dw_q_norm=grads.pop(0), dw_k_norm=grads.pop(0)) if g64.qk_norm else out.update(dw_q_norm=None, dw_k_norm=None)
    b, s = dy.shape[0], dy.shape[1]
    t = b * s
    out.update(
        dq_pre=grads.pop(0).reshape(t, g64.h_q, g64.d_head),
        dk_pre=grads.pop(0).reshape(t, g64.h_kv, g64.d_head),
        dg=grads.pop(0).reshape(t, g64.h_q, g64.d_head),
        dv=grads.pop(0).reshape(t, g64.h_kv, g64.d_head),
        do=grads.pop(0).reshape(t, g64.h_q, g64.d_head),
        ref=ref,
    )
    dq_post, dk_post = grads.pop(0), grads.pop(0)  # w.r.t. the post-norm / post-RoPE q / k: B4's outputs, the norm backward's inputs
    if g64.qk_norm:
        out["dw_q_norm_mass"] = _dw_norm_noise_mass(dq_post, ref.q_pre, ref.rstd_q, inp["cos"].double(), inp["sin"].double(), g64.rope_dim)
        out["dw_k_norm_mass"] = _dw_norm_noise_mass(dk_post, ref.k_pre, ref.rstd_k, inp["cos"].double(), inp["sin"].double(), g64.rope_dim)
    return out


def _rope_adjoint(dy: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """The exact RoPE adjoint on ``[B, S, H, D]`` (the kernel's form): ``dy*cos - rotate_half(dy*sin)`` on the leading
    ``rope_dim`` columns, pass-through beyond."""
    rot, rest = dy[..., :rope_dim], dy[..., rope_dim:]
    c, sn = cos[:, :, None, :rope_dim], sin[:, :, None, :rope_dim]
    ys = rot * sn
    half = rope_dim // 2
    g_rot = rot * c - torch.cat((-ys[..., half:], ys[..., :half]), dim=-1)
    return torch.cat((g_rot, rest), dim=-1) if rest.shape[-1] else g_rot


def _dw_norm_noise_mass(dq_post: torch.Tensor, x_pre: torch.Tensor, rstd: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Per column ``d``: ``sqrt(sum over rows of (g * x_hat)^2)`` in fp64 -- the natural unit of the bf16 rounding noise of
    ``dW[d] = sum_rows g * x_hat`` when both factors carry bf16-rounded inputs (``g = RoPE^T(dQ)``, ``x_hat = x * rstd``)."""
    g = _rope_adjoint(dq_post.double(), cos, sin, rope_dim)
    x_hat = x_pre.double() * rstd.double()[..., None]
    terms = (g * x_hat).reshape(-1, x_pre.shape[-1])
    return terms.pow(2).sum(0).sqrt()


def _assert_dw_norm_close(got: torch.Tensor, ref64: torch.Tensor, mass: torch.Tensor, what: str) -> float:
    """``|diff| <= _DW_NORM_NOISE * mass + _DW_NORM_RTOL * |ref|`` per column plus the cosine floor; prints the measured
    ``|diff| / mass`` (the calibration quantity) and, for the record, the worst cell of the initially inferred bound."""
    got64, ref64, mass = got.detach().double(), ref64.detach().double(), mass.detach().double()
    assert torch.isfinite(got64).all(), f"{what}: non-finite cells"
    diff = (got64 - ref64).abs()
    ref_max = ref64.abs().max().item()
    noise_ratio = (diff / mass.clamp_min(1e-300)).max().item()
    bound = _DW_NORM_NOISE * mass + _DW_NORM_RTOL * ref64.abs()
    worst = (diff / bound).max().item()
    spec_worst = (diff / (_DW_NORM_SPEC_ATOL_FRAC * ref_max + _DW_NORM_RTOL * ref64.abs())).max().item()
    cos = _cos(got64, ref64)
    msg = (
        f"{what}: max|diff|={diff.max().item():.4g} max|ref|={ref_max:.4g} mass[min,max]=[{mass.min().item():.4g},{mass.max().item():.4g}] "
        f"|diff|/mass max={noise_ratio:.4g} worst cell {worst:.3f} of the bound ({_DW_NORM_NOISE:.4g}*mass + {_DW_NORM_RTOL}*|ref|) "
        f"[inferred bound atol=1e-3*max|ref|: {spec_worst:.3f}] cos={cos:.6f} {got.dtype}"
    )
    print(msg)
    assert cos >= _COS_MIN, msg
    assert bool((diff <= bound).all()), msg
    return worst


def _declare_bwd(geom_kw, batch, seq_len, *, save_mode="proj_slab", dtype=torch.bfloat16, seq_lens=None, **bwd_kw):
    """A DECLARED (not compiled) backward over a DECLARED training forward's record -- CUDA tensors, no launch, any CUDA device."""
    fwd, inp, out = _declare(geom_kw, batch, seq_len, save_mode=save_mode, dtype=dtype, seq_lens_present=seq_lens is not None)
    saved = _alloc_saved(fwd.geom, inp, batch, seq_len, save_mode=save_mode, seq_lens=seq_lens)
    dy = _make_dy(out)
    bwd = GatedAttentionBlockBwd(dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], fwd.geom, **bwd_kw)
    return SimpleNamespace(blk=bwd, fwd=fwd, inp=inp, saved=saved, dy=dy, out=out, geom=fwd.geom, geom_kw=geom_kw, batch=batch, seq_len=seq_len)


_MEMO: dict = {}


def _backward(geom_kw, batch, seq_len, *, dtype=torch.bfloat16, memo=True, **bwd_kw):
    """Run the training forward (the record), declare / compile / run the backward, and differentiate the fp64 oracle.
    Memoised per declaration so the contract tests reuse one compiled block (the compile is the expensive part)."""
    key = (tuple(sorted(geom_kw.items())), batch, seq_len, str(dtype), tuple(sorted(bwd_kw.items())))
    if memo and key in _MEMO:
        return _MEMO[key]
    out, _ref32, fwd, saved, inp = _run_training(geom_kw, batch, seq_len, save_mode="proj_slab", dtype=dtype)
    dy = _make_dy(out)
    blk = GatedAttentionBlockBwd(dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], fwd.geom, **bwd_kw)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute(blk, inp, saved, dy, grads, ws)
    torch.cuda.synchronize()
    res = SimpleNamespace(
        blk=blk,
        fwd=fwd,
        inp=inp,
        saved=saved,
        dy=dy,
        out=out,
        ws=ws,
        grads=grads,
        oracle=_fp64_oracle(inp, geom_kw, dy),
        geom=fwd.geom,
        geom_kw=geom_kw,
        batch=batch,
        seq_len=seq_len,
    )
    if memo:
        _MEMO[key] = res
    return res


def _check_all_grads(res) -> dict:
    """Every produced gradient against the fp64 oracle; returns the worst-cell fractions for the report."""
    worst = {}
    for name in ("dh", "dw_qkvg", "dw_o"):
        if res.grads[name] is not None:
            worst[name] = _assert_grad_close(res.grads[name], res.oracle[name], name)
    for name in ("dw_q_norm", "dw_k_norm"):
        if res.grads[name] is not None:
            worst[name] = _assert_dw_norm_close(res.grads[name], res.oracle[name], res.oracle[name + "_mass"], name)
    return worst


def _bands(res) -> dict:
    """The block's materialised intermediates after execute, read out of the workspace: the four ``dqkvg`` bands and dO."""
    blk, g = res.blk, res.geom
    t, d, act = blk.batch * blk.seq_len, g.d_head, blk.act_dtype
    lay = blk._layout()
    dqkvg = _view(res.ws, lay.dqkvg, (t, g.n_qkvg), act)
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    return dict(
        dq_pre=_cols(dqkvg, o_q, g.h_q, d),
        dg=_cols(dqkvg, o_g, g.h_q, d),
        dk_pre=_cols(dqkvg, o_k, g.h_kv, d),
        dv=_cols(dqkvg, o_v, g.h_kv, d),
        do=_view(res.ws, lay.do_gated, (t, g.h_q, d), act),
    )


# ---------------------------------------------------------------------------
# Gradients vs fp64 autograd (Rubin)
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("seq_len", [256, 1000, 2048])
@_CAUSAL
@_QK_NORM
def test_gradients_match_fp64_autograd(qk_norm, causal, seq_len):
    """``dh``, ``dW_qkvg``, ``dW_o`` (bf16) and ``dW_q_norm`` / ``dW_k_norm`` (fp32, norm arm) against fp64 autograd at
    the test geometry, B=2, norm | rope_only x causal | dense x S in {256 (one kv block), 1000 (S % 128 != 0: the
    adapter's padded staging), 2048}. The magnitudes are printed; the bounds are the module's, never widened.

    A DENSE ``S % 128 != 0`` has no training record to differentiate: the FORWARD's SDPA row declines it typed (its KV
    tail would be unmasked on the SM100 DSL: "S_kv (1000) must be a multiple of 128 unless a padding mask ... or the
    causal mask covers the KV tail"), so those two cells pin the forward's decline instead of skipping (rejections are
    asserted, not skipped); the causal S=1000 cells run the adapter's padded launches."""
    geom_kw = {**_COMMON, "qk_norm": qk_norm, "is_causal": causal}
    if not causal and seq_len % 128:
        with pytest.raises(ValueError, match="multiple of 128"):
            _backward(geom_kw, batch=2, seq_len=seq_len)
        return
    res = _backward(geom_kw, batch=2, seq_len=seq_len)
    worst = _check_all_grads(res)
    assert (res.grads["dw_q_norm"] is None) == (not qk_norm)
    print(f"worst cells: {worst}")


@requires_rubin
def test_gradients_match_fp64_autograd_fp16():
    """The fp16 activation arm (``_ACT_DTYPES`` claims it; ``act_dtype`` keys every dtype check of the record contract
    and every kernel artifact of the chain): the record comes from the fp16 training forward, the SDPA backward runs
    its fp16 dS, the GEMMs fp16 io, the outputs are fp16. Same oracle, the fp16 bounds (``_RTOL`` / ``_ATOL_FRAC``:
    8x the bf16 tightness) and the dW_norm noise bound; the fractions are printed -- a miss is reported, never
    widened."""
    res = _backward(dict(_COMMON), batch=1, seq_len=256, dtype=torch.float16)
    assert res.grads["dh"].dtype == res.grads["dw_qkvg"].dtype == res.grads["dw_o"].dtype == torch.float16
    assert res.blk.act_dtype == torch.float16 and res.saved.proj_slab.dtype == torch.float16
    worst = _check_all_grads(res)
    print(f"fp16 worst cells: {worst}")


@requires_rubin
@_QK_NORM
def test_stage_localisation_via_autograd_grad(qk_norm):
    """The block's intermediates against the oracle's: the ``dqkvg`` bands (dQ_pre / dG / dK_pre / dV, written by B3
    and B5+B6 from B4's compact slots) and dO (the ``do_gated`` slot after B3) vs ``torch.autograd.grad`` w.r.t. the
    oracle's ``q_pre`` / ``gate`` / ``k_pre`` / ``v`` / ``o``. A miss here names the stage; a miss in the outputs
    alone would name the GEMMs.

    Under ``qk_norm`` it also localises ``dW_norm``: rebuilt in fp64 from the block's OWN compact bf16 dQ / dK slots
    (B4's outputs, intact after execute -- B5+B6 read them and nothing writes them again), the slab's bf16 pre-norm
    bands, the saved fp32 rstd and the bf16 cos / sin -- ``dW[d] = sum_rows RoPE^T(dq)[r, d] * (x_pre[r, d] * rstd[r])``
    -- the kernel + the fixed-order reduce must match at fp32 tightness (an fp32 sum of products that are exact in
    fp32). A pass pins the whole deviation from the fp64 oracle (the module docstring's ``|diff| / mass`` ~ 2e-2) on
    B4's bf16 rounding; a miss names B5+B6 / the reduce."""
    res = _backward({**_COMMON, "qk_norm": qk_norm}, batch=2, seq_len=256)
    bands = _bands(res)
    for name in ("do", "dg", "dv", "dq_pre", "dk_pre"):
        _assert_grad_close(bands[name], res.oracle[name], f"band {name}")
    if qk_norm:
        blk, g, inp, saved = res.blk, res.geom, res.inp, res.saved
        b, s, d, act = res.batch, res.seq_len, g.d_head, blk.act_dtype
        t = b * s
        lay = blk._layout()
        proj = saved.proj_slab.view(t, g.n_qkvg)
        o_q, _o_g, o_k, _o_v = g.qkvg_offsets
        cos64, sin64 = inp["cos"].double(), inp["sin"].double()
        for name, slot, heads, off, rstd in (("dw_q_norm", lay.dq, g.h_q, o_q, saved.rstd_q), ("dw_k_norm", lay.dk, g.h_kv, o_k, saved.rstd_k)):
            dq_slot = _view(res.ws, slot, (t, heads, d), act).view(b, s, heads, d).double()
            g_rows = _rope_adjoint(dq_slot, cos64, sin64, g.rope_dim)
            x_hat = _cols(proj, off, heads, d).view(b, s, heads, d).double() * rstd.double()[..., None]
            terms = (g_rows * x_hat).reshape(-1, d)
            ref_slot, mass = terms.sum(0), terms.pow(2).sum(0).sqrt()
            got = res.grads[name].double()
            diff = (got - ref_slot).abs()
            bound = 1e-4 * mass + 1e-3 * ref_slot.abs()
            worst = (diff / bound).max().item()
            print(
                f"{name} vs the block's own dq slot: max|diff|={diff.max().item():.4g} max|ref|={ref_slot.abs().max().item():.4g} "
                f"|diff|/mass max={(diff / mass).max().item():.3g} worst cell {worst:.3f} of the fp32 bound (1e-4*mass + 1e-3*|ref|)"
            )
            assert bool(
                (diff <= bound).all()
            ), f"{name}: the norm+RoPE backward kernel / reduce deviates from the fp64 sum over the block's own dq slot ({worst:.3f} of the bound)"


@requires_rubin
@pytest.mark.parametrize("h_kv", [1, 2])
@pytest.mark.parametrize("batch", [1, 3])
def test_gqa_ratio_and_batch(h_kv, batch):
    """S = 256 pins ``n_kv == 1`` (ONE 256-row kv block) with ``B * H >= 2`` -- the ring-phase-drift repro shape
    -- across the GQA fold (g = 8 and 4) and the adapter's batch chunking."""
    res = _backward({**_COMMON, "h_kv": h_kv}, batch=batch, seq_len=256)
    _check_all_grads(res)


@requires_rubin
def test_padded_s_with_batch():
    """S = 1000 with B = 2: the adapter's +7 padded launches (q / dO / lse pads, k / v pads, the GQA fold copy-outs)
    run under batch chunking; the same bound."""
    res = _backward(dict(_COMMON), batch=2, seq_len=1000)
    _check_all_grads(res)
    _bands(res)  # the bands are readable (the carve covers the padded shape)


@requires_rubin
@_CAUSAL
def test_gradients_match_fp64_autograd_swa640(causal):
    """A sliding window of 640 at S = 1024 (with and without the causal diagonal) through the oracle's window arm
    (``RefGeometry.window_left``, pinned vs ``F.scaled_dot_product_attention`` by ``test_reference_window_arm_matches_torch_sdpa``):
    a graph flag with a matching reference branch."""
    res = _backward({**_COMMON, "is_causal": causal, "window_left": 640}, batch=1, seq_len=1024)
    _check_all_grads(res)


@requires_rubin
def test_bottom_right_causal_and_save_all_policy():
    """Two more arms of the served set at S=256, B=1 (every served arm, not only the common one):
    ``causal_bottom_right=True`` (== top-left at S_q == S_kv; mapped one-to-one onto the adapter's flag, the oracle's own
    arm) and ``recompute=RecomputePolicy.SAVE_ALL`` (both served policies read the slab bands in P0; the default cells
    cover only ``RECOMPUTE_QK_PRE``) -- the main bound."""
    res = _backward({**_COMMON, "is_causal": True, "causal_bottom_right": True}, batch=1, seq_len=256)
    assert res.blk._sdpa._impl.causal_bottom_right
    _check_all_grads(res)
    res = _backward(dict(_COMMON), batch=1, seq_len=256, recompute=RecomputePolicy.SAVE_ALL)
    assert res.blk.recompute is RecomputePolicy.SAVE_ALL
    _check_all_grads(res)


@requires_rubin
def test_partial_needs_skip_whole_stages():
    """``need_dw_o=False, need_dw_norms=False``: B1 and the reduce do not exist, the gate kernel is traced without its
    third output (``has_og=False``) and the norm kernel without partials (``want_dw=False``) -- both directions typed
    by the kernels -- and ``dh`` / ``dW_qkvg`` still match the oracle."""
    res = _backward(dict(_COMMON), batch=1, seq_len=256, need_dw_o=False, need_dw_norms=False)
    assert res.grads["dw_o"] is None and res.grads["dw_q_norm"] is None
    assert res.blk._layout().o_gated == -1 and res.blk._layout().dw_partials_q == -1
    assert "out_proj_wgrad" not in res.blk.gemm_plans
    _check_all_grads(res)


@requires_rubin
def test_two_runs_are_bitwise():
    """Determinism by construction (no atomics anywhere on the chain, fixed-order folds and reduces): two executes
    with the workspace POISONED in between (0xFF bytes = bf16 NaN) give bitwise-equal gradients -- which also proves
    no region depends on prior workspace content."""
    res = _backward(dict(_COMMON), batch=2, seq_len=256)
    res.ws.fill_(0xFF)
    grads2 = _alloc_grads(res.blk, fill=float("nan"))
    _execute(res.blk, res.inp, res.saved, res.dy, grads2, res.ws)
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), name
            assert torch.equal(ten, res.grads[name]), f"{name} differs across two runs"


@requires_rubin
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_a_caller_stream_orders_every_stage(how):
    """Every stage -- the CuTe-DSL kernels, the four FROST GEMMs and the SDPA backward adapter -- launches on ONE
    stream, the caller's: ambient (``with torch.cuda.stream(s):``) or explicit (``current_stream=``). The default
    stream is parked behind a long spin and the workspace is zeroed on the side stream right after the block, so a
    stage enqueued on the default stream runs late (a late producer leaves zeros for its consumers; a late consumer
    reads the zeros written over the workspace) and the gradients differ from the default-stream run -- which they
    must equal BITWISE (the block is deterministic)."""
    import cuda.bindings.driver as cuda_drv

    res = _backward(dict(_COMMON), batch=1, seq_len=256)  # default stream, synchronized
    side = torch.cuda.Stream()
    ws2 = torch.zeros_like(res.ws)
    grads2 = _alloc_grads(res.blk, fill=0)
    torch.cuda.synchronize()  # the fills above ran on the default stream: finish them before the park, or a late fill could overwrite a stage's output
    park_the_default_stream()
    with torch.cuda.stream(side):
        dy2 = res.dy.clone()  # written on the side stream right before the block reads it
        if how == "ambient":
            _execute(res.blk, res.inp, res.saved, dy2, grads2, ws2)
        else:
            _execute(res.blk, res.inp, res.saved, dy2, grads2, ws2, current_stream=cuda_drv.CUstream(side.cuda_stream))
        ws2.zero_()
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), f"{name}: a stage escaped the caller's stream ({how})"


@requires_rubin
def test_workspace_size_is_honest():
    """``get_workspace_size()`` is exact and never exceeded: a buffer 4096 B larger keeps its tail
    untouched; two executes allocate nothing (``memory_allocated`` delta 0 after a warm-up); the result over a
    sentinel-filled buffer is bitwise the memoised one; ``gemm_scratch`` covers ``max(plan.workspace_bytes)`` (12 MiB
    at this geometry: the backend heuristic's split-K partials, never launched on the forced JIT path)."""
    res = _backward(dict(_COMMON), batch=2, seq_len=256)
    blk = res.blk
    size = blk.get_workspace_size()
    lay = blk._layout()
    assert size == lay.total_bytes and size % _WS_ALIGN == 0
    plans = blk.gemm_plans
    assert len(plans) == 4
    need_gemm = max(p.workspace_bytes for p in plans.values())
    print(f"workspace {size} B; gemm_scratch {lay.gemm_scratch_bytes} B (plans need {need_gemm}); sdpa scratch {lay.sdpa_bwd_bytes} B")
    assert lay.gemm_scratch_bytes >= need_gemm >= 1
    assert lay.sdpa_bwd_bytes == blk._sdpa.scratch_workspace_bytes()
    ws = torch.full((size + 4096,), 0xAB, dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute(blk, res.inp, res.saved, res.dy, grads, ws)  # warm-up: first-use artefacts, if any
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    _execute(blk, res.inp, res.saved, res.dy, grads, ws)
    _execute(blk, res.inp, res.saved, res.dy, grads, ws)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "execute allocated on the hot path"
    assert torch.equal(ws[size:], torch.full((4096,), 0xAB, dtype=torch.uint8, device="cuda")), "bytes past get_workspace_size() were written"
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), name


@requires_rubin
def test_plans_are_the_forced_tile():
    """All four GEMM plans run the block's FORCED tile on the JIT route at one split-K slice: a
    fallback to the heuristic is a FAILURE, never a skip -- it changes the config, the route and possibly adds a
    split-K reducer whose workspace the carve would then have to carry."""
    res = _backward(dict(_COMMON), batch=2, seq_len=256)
    plans = res.blk.gemm_plans
    assert set(plans) == {"out_proj_dgrad", "out_proj_wgrad", "qkv_gate_wgrad", "qkv_gate_dgrad"}
    majors = {"out_proj_dgrad": ("k", "n"), "qkv_gate_dgrad": ("k", "n"), "out_proj_wgrad": ("m", "n"), "qkv_gate_wgrad": ("m", "n")}
    for label, plan in plans.items():
        print(
            f"{label}: tile={plan.tile_config_name} route={plan.route} majors=({plan.a_major},{plan.b_major}) split_k={plan.split_k} ws={plan.workspace_bytes}"
        )
        assert plan.tile_config_name == _FORCED_TILE, (label, plan.tile_config_name)
        assert plan.route == "graph+jit" and plan.jit is not None, (label, plan.route)
        assert plan.split_k == 0 and (plan.a_major, plan.b_major) == majors[label]


@requires_rubin
def test_dw_partial_planes_are_sized_to_n_ctas_for():
    """After ``compile()`` the two fp32 partial planes have EXACTLY ``n_ctas_for(recipe, T)`` rows (the reduce sums
    every row it is handed and is launched with ``t=T``, which cross-checks them) -- the kernel's contract."""
    from cudnn.gated_attention_block.kernels.qk_norm_rope_bwd import n_ctas_for

    res = _backward(dict(_COMMON), batch=2, seq_len=256)
    lay = res.blk._layout()
    n_q, n_k, _n_v = n_ctas_for(res.blk._norm_bwd._recipe, res.batch * res.seq_len)
    assert (lay.n_ctas_q, lay.n_ctas_k) == (n_q, n_k)
    assert lay.dw_partials_k - lay.dw_partials_q >= n_q * _COMMON["d_head"] * 4


@requires_rubin
@pytest.mark.parametrize("seq_len, expected", [(256, 15), (1000, 22)])
def test_launch_count_is_honest(seq_len, expected):
    """CUPTI kernel records of one execute == the launch table: ``12 + c*(2+q)`` (15 at this geometry: c = 1 and q = 1 dQ
    GEMM launch per chunk under the adapter's single-launch dQ rendering -- its ``b_head_group`` is the GQA group; the
    per-member twin would make it g = 4) plus the adapter's padded launches at S = 1000 (+3 q / dO / lse pads, +2 k / v
    pads, +2 GQA fold copy-outs = 22). Profiled on a warm block; no hidden memcpy. The formula is ALSO recomputed from the
    adapter's own facts (chunks, the dQ record's ``b_head_group``, padding, zero-fill) so a change in either side is
    visible."""
    from torch.profiler import ProfilerActivity, profile

    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import _dq_launches

    res = _backward(dict(_COMMON), batch=2, seq_len=seq_len)
    blk, g = res.blk, res.geom
    impl = blk._sdpa._impl
    grp = g.h_q // g.h_kv
    c = (blk.batch // impl._b_chunk) * (g.h_q // impl._qh_chunk)
    dq_launches = _dq_launches(grp, impl._dq_b_head_group)  # 1 under the single-launch dQ rendering, grp per group member
    formula = 9 + 2 + c * (2 + dq_launches) + (1 if grp > 1 else 0) + (3 if impl._q_padded else 0) + (2 if impl._kv_padded else 0)
    formula += (2 if (impl._kv_padded and grp > 1) else 1 if impl._kv_padded else 0) + (1 if impl._zero_ws else 0)
    grads = _alloc_grads(blk)
    _execute(blk, res.inp, res.saved, res.dy, grads, res.ws)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _execute(blk, res.inp, res.saved, res.dy, grads, res.ws)
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    names = [e.name for e in evs]
    if not names:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    memsets = [n for n in names if "memset" in n.lower()]
    memcpys = [n for n in names if "memcpy" in n.lower()]
    kernels = [n for n in names if n not in memsets and n not in memcpys]
    print(f"\n{len(kernels)} kernels (formula {formula}, expected {expected}), {len(memsets)} memsets, {len(memcpys)} memcpys:\n  " + "\n  ".join(names))
    assert not memcpys, f"a hidden copy on the execute path: {memcpys}"
    assert formula == expected, (formula, expected)
    assert len(kernels) == expected, (len(kernels), expected, kernels)


@requires_rubin
def test_convenience_wrapper_matches_the_class():
    """``gated_attention_block_backward`` allocates, caches the compiled block and delegates: its outputs are
    ``torch.equal`` the class path's; which gradients exist follows ``requires_grad`` on the forward inputs."""
    res = _backward(dict(_COMMON), batch=1, seq_len=256)
    inp, saved = res.inp, res.saved
    for t in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
        t.requires_grad_(True)
    try:
        out = gated_attention_block_backward(res.dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], res.geom)
        torch.cuda.synchronize()
        dh, dw_qkvg, dw_o, dw_q_norm, dw_k_norm = out
        for name, ten in (("dh", dh), ("dw_qkvg", dw_qkvg), ("dw_o", dw_o), ("dw_q_norm", dw_q_norm), ("dw_k_norm", dw_k_norm)):
            assert torch.equal(ten, res.grads[name]), name
        assert out["dh"] is dh
        # requires_grad decides the set: no weight gradients wanted -> None entries, dh alone still computed
        for t in (inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
            t.requires_grad_(False)
        out2 = gated_attention_block_backward(res.dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], res.geom)
        torch.cuda.synchronize()
        assert out2["dw_qkvg"] is None and out2["dw_o"] is None and out2["dw_q_norm"] is None
        assert torch.equal(out2["dh"], res.grads["dh"])
        # a PADDED record through the wrapper: the record's seq_lens presence is in the cache key, so this is a NEW
        # declaration, declined typed at its check_support -- never the cached dense block's chain
        lens = torch.full((1,), 256, dtype=torch.int32, device="cuda")
        with pytest.raises(NotImplementedError, match="sdpa_bwd_sm107"):
            gated_attention_block_backward(
                res.dy, dataclasses.replace(saved, seq_lens=lens), inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], res.geom
            )
    finally:
        for t in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
            t.requires_grad_(False)


def _record_cuda_allocation_streams(monkeypatch) -> list:
    """Record torch's CURRENT stream at every ``torch.empty`` / ``torch.empty_like`` issued while patched: the caching
    allocator tags a block with that stream and orders its reuse against that stream alone."""
    seen: list = []
    real_empty, real_empty_like = torch.empty, torch.empty_like

    def empty(*a, **k):
        t = real_empty(*a, **k)
        if t.is_cuda:
            seen.append(torch.cuda.current_stream(t.device).cuda_stream)
        return t

    def empty_like(src, *a, **k):
        t = real_empty_like(src, *a, **k)
        if t.is_cuda:
            seen.append(torch.cuda.current_stream(t.device).cuda_stream)
        return t

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch, "empty_like", empty_like)
    return seen


@requires_cuda
def test_convenience_wrapper_allocates_on_the_launch_stream(monkeypatch):
    """Rule 5 / recipe R2 on the convenience path: with ``current_stream`` naming a SIDE stream while torch's current
    stream is still the default one, the per-call workspace and gradients are allocated -- and the block is run --
    under that side stream. The workspace reference dies at return: allocated on the ambient stream it would be freed
    into the DEFAULT stream's pool and could back the caller's next allocation while the backward is still writing it
    (silent gradient corruption under load). The compiled block is replaced by a recorder so the detector runs on any
    CUDA device: RED on an ambient-stream allocation, GREEN on the launch-stream one."""
    import cuda.bindings.driver as cuda_drv
    from cudnn.gated_attention_block import api_bwd as api_bwd_mod

    dec = _declare_bwd(dict(_COMMON), batch=1, seq_len=256)
    inp, saved = dec.inp, dec.saved
    seen: list = []

    class _Recorder:
        def __init__(self, *a, **k):
            pass

        def check_support(self):
            pass

        def compile(self):
            pass

        def get_workspace_size(self):
            return 4096

        def execute(self, *a, workspace=None, current_stream=None, **k):
            seen.append(
                ("execute", torch.cuda.current_stream().cuda_stream, int(current_stream) if current_stream is not None else None, int(workspace.numel()))
            )

    monkeypatch.setattr(api_bwd_mod, "GatedAttentionBlockBwd", _Recorder)
    monkeypatch.setattr(api_bwd_mod, "_BWD_CACHE", {})
    allocs = _record_cuda_allocation_streams(monkeypatch)
    side = torch.cuda.Stream()
    ambient = torch.cuda.current_stream()
    assert ambient.cuda_stream != side.cuda_stream
    for t in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
        t.requires_grad_(True)
    try:
        out = gated_attention_block_backward(
            dec.dy,
            saved,
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            dec.geom,
            current_stream=cuda_drv.CUstream(side.cuda_stream),
        )
    finally:
        for t in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
            t.requires_grad_(False)
    assert torch.cuda.current_stream().cuda_stream == ambient.cuda_stream, "the wrapper leaked its stream context"
    assert len(allocs) >= 4, allocs  # the workspace + dh + dW_qkvg + dW_o (+ the two dW_norm under qk_norm)
    assert all(s == side.cuda_stream for s in allocs), f"an allocation on the ambient stream {ambient.cuda_stream}: {allocs} (launch stream {side.cuda_stream})"
    assert seen == [("execute", side.cuda_stream, side.cuda_stream, 4096)], seen
    assert out["dh"].shape == saved.h.shape and out["dw_qkvg"].shape == inp["w_qkvg"].shape


@requires_rubin
def test_convenience_wrapper_explicit_stream_from_the_default_stream(monkeypatch):
    """The wrapper twin of ``test_a_caller_stream_orders_every_stage[explicit]``: ``current_stream`` names a side
    stream while the caller stays on the (parked) default stream. Every stage and every per-call allocation follows
    the handle -- the gradients are bitwise the default-stream run's and nothing was allocated on the ambient stream."""
    import cuda.bindings.driver as cuda_drv

    res = _backward(dict(_COMMON), batch=1, seq_len=256)  # default stream, synchronized; the wrapper's block is cached
    inp, saved = res.inp, res.saved
    allocs = _record_cuda_allocation_streams(monkeypatch)
    side = torch.cuda.Stream()
    for t in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
        t.requires_grad_(True)
    try:
        torch.cuda.synchronize()
        park_the_default_stream()
        out = gated_attention_block_backward(
            res.dy,
            saved,
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            res.geom,
            current_stream=cuda_drv.CUstream(side.cuda_stream),
        )
        torch.cuda.synchronize()
    finally:
        for t in (saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
            t.requires_grad_(False)
    assert allocs and all(s == side.cuda_stream for s in allocs), f"an allocation escaped the explicit stream: {allocs} (side {side.cuda_stream})"
    for name in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"):
        assert torch.equal(out[name], res.grads[name]), f"{name}: a stage or an allocation escaped the explicit stream"


@requires_rubin
def test_execute_contracts_on_a_compiled_block():
    """Rule 1 both directions at EXECUTE, before any launch: a missing needed output, an output for a ``need_*`` the
    block was declared without, a ``dw_*_norm`` in the wrong dtype (naming ``dw_norm_dtype``), ``seq_lens`` on a block
    declared without padding, a short workspace, and an output aliasing an operand (the overlap check)."""
    res = _backward(dict(_COMMON), batch=1, seq_len=256)
    blk, inp, saved, dy, ws = res.blk, res.inp, res.saved, res.dy, res.ws
    grads = _alloc_grads(blk)
    with pytest.raises(ValueError, match="dh is required"):
        _execute(blk, inp, saved, dy, {**grads, "dh": None}, ws)
    with pytest.raises(ValueError, match="dw_norm_dtype"):
        _execute(blk, inp, saved, dy, {**grads, "dw_q_norm": grads["dw_q_norm"].to(torch.bfloat16)}, ws)
    lens = torch.full((1,), 256, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="seq_lens"):
        _execute(blk, inp, saved, dy, grads, ws, seq_lens=lens)
    # At execute: a PADDED record (saved.seq_lens a tensor) contradicts the dense declaration -- refused before any
    # launch, from the tensor's presence alone (a padded record through the dense chain would be NaN on its dead rows)
    with pytest.raises(ValueError, match="saved.seq_lens"):
        _execute(blk, inp, dataclasses.replace(saved, seq_lens=lens), dy, grads, ws)
    with pytest.raises(ValueError, match="workspace is"):
        _execute(blk, inp, saved, dy, grads, ws[:-256])
    with pytest.raises(ValueError, match="overlaps"):
        _execute(blk, inp, saved, dy, {**grads, "dh": dy}, ws)
    # workspace aliasing a caller buffer: the record's o inside the workspace storage
    ws_big = torch.empty(ws.numel() + saved.o.numel() * 2, dtype=torch.uint8, device="cuda")
    o_alias = ws_big[: saved.o.numel() * 2].view(torch.bfloat16).view(saved.o.shape)
    saved_alias = dataclasses.replace(saved, o=o_alias)
    with pytest.raises(ValueError, match="overlaps"):
        _execute(blk, inp, saved_alias, dy, grads, ws_big)
    # a need_dw_o=False block refuses a dw_o
    res2 = _backward(dict(_COMMON), batch=1, seq_len=256, need_dw_o=False, need_dw_norms=False)
    with pytest.raises(ValueError, match="need_dw_o=False"):
        _execute(res2.blk, res2.inp, res2.saved, res2.dy, {**res2.grads, "dw_o": torch.empty_like(grads["dw_o"])}, res2.ws)
    with pytest.raises(ValueError, match="need_dw_norms=False"):
        _execute(res2.blk, res2.inp, res2.saved, res2.dy, {**res2.grads, "dw_q_norm": grads["dw_q_norm"], "dw_k_norm": grads["dw_k_norm"]}, res2.ws)


# ---------------------------------------------------------------------------
# The typed contract at declaration / check_support (any CUDA device; no compile)
# ---------------------------------------------------------------------------


def test_workspace_carve_is_the_declared_composition():
    """``_plan_bwd_workspace`` on any device: regions in the documented order, each padded to the 256 B carve
    alignment, the dW planes exactly ``n_ctas x D x 4`` bytes, ``gemm_scratch`` never 0, the appended fields
    resolving to -1 when a ``need_*`` is False."""
    g = GatedAttentionBlockGeometry(**_COMMON)
    b, s, e = 2, 256, 2
    t, d = b * s, g.d_head
    lay = _plan_bwd_workspace(
        g,
        b,
        s,
        torch.bfloat16,
        RecomputePolicy.RECOMPUTE_QK_PRE,
        need=dict(dw_o=True, dw_norms=True),
        sdpa_bwd_bytes=1000,
        gemm_scratch_bytes=0,
        n_ctas_q=7,
        n_ctas_k=3,
    )
    assert isinstance(lay, _BwdIntermediates)
    al = lambda n: -(-n // _WS_ALIGN) * _WS_ALIGN  # noqa: E731
    order = [
        "do_gated",
        "dqkvg",
        "o_gated",
        "recompute",
        "recompute_k",
        "recompute_v",
        "dq",
        "dk",
        "dv",
        "dw_partials_q",
        "dw_partials_k",
        "sdpa_bwd_ws",
        "gemm_scratch",
    ]
    sizes = dict(
        do_gated=t * g.h_q * d * e,
        dqkvg=t * g.n_qkvg * e,
        o_gated=t * g.h_q * d * e,
        recompute=t * g.h_q * d * e,
        recompute_k=t * g.h_kv * d * e,
        recompute_v=t * g.h_kv * d * e,
        dq=t * g.h_q * d * e,
        dk=t * g.h_kv * d * e,
        dv=t * g.h_kv * d * e,
        dw_partials_q=7 * d * 4,
        dw_partials_k=3 * d * 4,
        sdpa_bwd_ws=1000,
        gemm_scratch=1,
    )
    off = 0
    for name in order:
        assert getattr(lay, name) == off, (name, getattr(lay, name), off)
        off += al(sizes[name])
    assert lay.total_bytes == off and lay.do == -1 and lay.base_align == _WS_ALIGN
    assert (lay.n_ctas_q, lay.n_ctas_k, lay.sdpa_bwd_bytes, lay.gemm_scratch_bytes) == (7, 3, 1000, 1)
    lean = _plan_bwd_workspace(
        g, b, s, torch.bfloat16, RecomputePolicy.SAVE_ALL, need=dict(dw_o=False, dw_norms=False), sdpa_bwd_bytes=1000, gemm_scratch_bytes=4096
    )
    assert lean.o_gated == -1 and lean.dw_partials_q == lean.dw_partials_k == -1 and lean.n_ctas_q == 0
    assert lean.total_bytes == lay.total_bytes - al(sizes["o_gated"]) - al(sizes["dw_partials_q"]) - al(sizes["dw_partials_k"]) + al(4096) - al(1)
    with pytest.raises(ValueError, match="n_ctas"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.SAVE_ALL, need=dict(dw_norms=True), sdpa_bwd_bytes=1000)
    with pytest.raises(ValueError, match="scratch"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.SAVE_ALL, need=dict(dw_norms=False), sdpa_bwd_bytes=0)


def test_bwd_constructor_no_longer_stubs():
    """The INVERTED pin of the stub era: construction SUCCEEDS on placeholders (the four declaration contracts still
    fire first -- test_block_end_to_end.py keeps them), ``need_dw_norms`` resolves from the knob, and ``execute`` /
    ``get_workspace_size`` before ``compile`` are typed ``RuntimeError``s naming ``compile()``."""
    d = _COMMON["d_head"]
    dy = torch.empty(1, 8, _COMMON["d_model"])
    z = torch.empty(0)
    w = torch.ones(d)
    geom_on = GatedAttentionBlockGeometry(**_COMMON)
    geom_off = GatedAttentionBlockGeometry(**{**_COMMON, "qk_norm": False})
    saved = lambda rstd: SavedForBackward(h=z, gate=z, o=z, lse=z, rstd_q=rstd, rstd_k=rstd)  # noqa: E731
    for geom, wq, rstd, kw, want in ((geom_on, w, z, {}, True), (geom_on, w, z, {"need_dw_norms": False}, False), (geom_off, None, None, {}, False)):
        blk = GatedAttentionBlockBwd(dy, saved(rstd), z, wq, wq, z, z, z, geom, **kw)
        assert blk.need_dw_norms is want
        assert (blk.batch, blk.seq_len, blk.act_dtype) == (1, 8, dy.dtype)
        with pytest.raises(RuntimeError, match="compile"):
            blk.get_workspace_size()
        with pytest.raises(RuntimeError, match="compile"):
            blk.execute(dy, saved(rstd), z, wq, wq, z, z, z)
    with pytest.raises(ValueError, match="sample_dy must be"):
        GatedAttentionBlockBwd(z, saved(z), z, w, w, z, z, z, geom_on)
    with pytest.raises(TypeError, match="RecomputePolicy"):
        GatedAttentionBlockBwd(dy, saved(z), z, w, w, z, z, z, geom_on, recompute=1)


@requires_cuda
def test_get_workspace_size_needs_compile():
    """As built, the plane rows and the four ``plan.workspace_bytes`` exist only once the artifacts do: the query
    before ``compile()`` is a ``RuntimeError`` naming ``compile()`` (the forward's own order)."""
    res = _declare_bwd(dict(_COMMON), 1, 256)
    with pytest.raises(RuntimeError, match=r"compile\(\)"):
        res.blk.get_workspace_size()


@requires_cuda
def test_need_flags_at_declaration():
    """``need_*`` all False -> ``ValueError`` (no work) at check_support; ``need_dw_norms=True`` under ``qk_norm=False``
    -> ``ValueError`` at construction (the existing contract); ``need_dw_norms=None`` follows the knob."""
    res = _declare_bwd(dict(_COMMON), 1, 256, need_dh=False, need_dw_qkvg=False, need_dw_o=False, need_dw_norms=False)
    with pytest.raises(ValueError, match="no work"):
        res.blk.check_support()
    with pytest.raises(ValueError, match="need_dw_norms=True"):
        _declare_bwd({**_COMMON, "qk_norm": False}, 1, 256, need_dw_norms=True)
    assert _declare_bwd({**_COMMON, "qk_norm": False}, 1, 256).blk.need_dw_norms is False


@requires_cuda
@pytest.mark.parametrize("policy", [RecomputePolicy.SAVE_ALL, RecomputePolicy.RECOMPUTE_QK_PRE])
def test_recompute_policy_declines(policy):
    """``RECOMPUTE_GATE`` is reserved (typed); a gate-copy record (``proj_slab=None``) is a typed decline naming
    the follow-up PR under EITHER served policy -- P0 = proj_slab records (V is a slab band with no field of its own)."""
    res = _declare_bwd(dict(_COMMON), 1, 256, recompute=RecomputePolicy.RECOMPUTE_GATE)
    with pytest.raises(NotImplementedError, match="RECOMPUTE_GATE"):
        res.blk.check_support()
    res = _declare_bwd(dict(_COMMON), 1, 256, save_mode="gate_copy", recompute=policy)
    assert res.saved.proj_slab is None
    with pytest.raises(NotImplementedError, match="gate-copy"):
        res.blk.check_support()


@requires_cuda
def test_dw_norm_dtype_is_fp32_only_in_p0():
    """The appended ``dw_norm_dtype`` knob: anything but fp32 is a typed ``NotImplementedError`` naming the knob."""
    res = _declare_bwd(dict(_COMMON), 1, 256, dw_norm_dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="dw_norm_dtype"):
        res.blk.check_support()


@requires_cuda
def test_padding_is_a_typed_decline():
    """Padding is declined at DECLARATION -- ``seq_lens_present=True``, or a record whose ``seq_lens`` is a tensor
    (the padded forward's identity-verified field) -- naming the ``sdpa_bwd_sm107`` row, with NO device sync
    (``set_sync_debug_mode('error')``: the tensor's presence is the fact, never its values)."""
    res = _declare_bwd(dict(_COMMON), 1, 256, seq_lens_present=True)
    with pytest.raises(NotImplementedError, match="sdpa_bwd_sm107"):
        res.blk.check_support()
    lens = torch.full((1,), 256, dtype=torch.int32, device="cuda")
    res = _declare_bwd(dict(_COMMON), 1, 256, seq_lens=lens)
    assert res.saved.seq_lens is lens
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(NotImplementedError, match="sdpa_bwd_sm107"):
            res.blk.check_support()
    finally:
        torch.cuda.set_sync_debug_mode(prev)


@requires_cuda
def test_window_knobs_the_row_cannot_serve_are_typed():
    """``window_left=0`` and ``window_right > 0`` are declined by the BLOCK, naming its own geometry field, before the
    adapter's message could."""
    res = _declare_bwd({**_COMMON, "window_left": 0}, 1, 256)
    with pytest.raises(NotImplementedError, match="window_left"):
        res.blk.check_support()
    res = _declare_bwd({**_COMMON, "is_causal": True, "window_right": 64}, 1, 256)
    with pytest.raises(NotImplementedError, match="window_right"):
        res.blk.check_support()


@requires_cuda
def test_prefill_and_forced_tile_preconditions_are_typed():
    """Two limits the block declines BEFORE the adapter or the GEMM driver could speak: ``seq_len = 1`` (decode -- the
    ``sdpa_bwd_sm107`` prefill bodies serve S_q >= 2; the adapter's own message is an untyped ``ValueError``), and a
    ``d_model`` the forced 256-wide GEMM tile cannot take (``d_model % 256 != 0``: the driver's heuristic fallback would
    change tile, route and possibly add a split-K reducer -- the block serves the forced tile only)."""
    res = _declare_bwd(dict(_COMMON), 1, 1)
    with pytest.raises(NotImplementedError, match="seq_len=1"):
        res.blk.check_support()
    res = _declare_bwd({**_COMMON, "d_model": 384}, 1, 256)
    with pytest.raises(NotImplementedError, match="multiple of 256"):
        res.blk.check_support()


@requires_cuda
def test_record_and_operand_contracts_are_typed():
    """The record and operand checks name their field: a wrong dtype / shape / device / misaligned or non-contiguous
    buffer, a band that does not alias the slab, a rope_dim of 0 -- all before any launch."""
    res = _declare_bwd(dict(_COMMON), 1, 256)
    blk, saved, inp = res.blk, res.saved, res.inp
    # every miss below is checked before the Rubin gate, so the message is the field's on any CUDA device
    b, s, g = 1, 256, blk.geom
    bad = dataclasses.replace(saved, o=saved.o.float())
    blk._samples["saved"] = bad
    with pytest.raises(ValueError, match="saved.o must be"):
        blk.check_support()
    blk._samples["saved"] = dataclasses.replace(saved, gate=saved.gate[:, :, :, 1:])
    with pytest.raises(ValueError, match="saved.gate must alias"):
        blk.check_support()
    blk._samples["saved"] = dataclasses.replace(saved, lse=saved.lse.view(b, s, g.h_q))
    with pytest.raises(ValueError, match="saved.lse must be"):
        blk.check_support()
    blk._samples["saved"] = saved
    blk._samples["dy"] = res.dy[:, 1:]
    with pytest.raises(ValueError, match="dy must be"):
        blk.check_support()
    blk._samples["dy"] = res.dy
    blk._samples["w_o"] = inp["w_o"].t().contiguous()
    with pytest.raises(ValueError, match="w_o must be"):
        blk.check_support()
    blk._samples["w_o"] = inp["w_o"]
    res_e4 = _declare_bwd(dict(_COMMON), 1, 256)
    res_e4.blk._samples["dy"] = res_e4.dy.float()
    res_e4.blk.act_dtype = torch.float32
    with pytest.raises(NotImplementedError, match="bf16 / fp16"):
        res_e4.blk.check_support()


@pytest.mark.skipif(_cc() == _SM107, reason="the everywhere-reject twin runs on every part BUT Rubin")
@requires_cuda
def test_declines_every_arch_but_rubin():
    """The block binds ONE engine class (the Rubin d256 backward) and declines every other device typed, naming Rubin
    (AGENTS.md Rule 9: no silent fallback to the cuDNN backend's d=256 backward)."""
    res = _declare_bwd(dict(_COMMON), 1, 256)
    with pytest.raises(NotImplementedError, match="Rubin"):
        res.blk.check_support()


def test_gemm_stage_declaration_rules():
    """The TMA 16-byte rule falls on the MN-major GEMM operands at DECLARATION (a message naming the operand), not at
    the JIT; the four stages spell the drivers' majors."""
    from cudnn.gated_attention_block.api_bwd import _OutProjDgrad, _OutProjWgrad, _QkvGateDgrad, _QkvGateWgrad

    assert _OutProjWgrad(m=512, k=2048, n=2048, dtype=torch.bfloat16, label="x").majors == ("m", "n")
    assert _QkvGateWgrad(m=5120, k=2048, n=512, dtype=torch.bfloat16, label="x").majors == ("m", "n")
    assert _OutProjDgrad(m=2048, k=512, n=2048, dtype=torch.bfloat16, label="x").majors == ("k", "n")
    assert _QkvGateDgrad(m=2048, k=5120, n=512, dtype=torch.bfloat16, label="x").majors == ("k", "n")
    with pytest.raises(ValueError, match="M=516"):
        _OutProjWgrad(m=516, k=2048, n=2048, dtype=torch.bfloat16, label="x").check_support()
    with pytest.raises(ValueError, match="N=2052"):
        _OutProjDgrad(m=2048, k=512, n=2052, dtype=torch.bfloat16, label="x").check_support()
    with pytest.raises(NotImplementedError, match="bf16 / fp16"):
        _OutProjDgrad(m=2048, k=512, n=2048, dtype=torch.float32, label="x").check_support()
    assert isinstance(_OutProjDgrad(m=8, k=8, n=8, dtype=torch.bfloat16, label="x"), _GemmStage)
