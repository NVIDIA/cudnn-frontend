# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The block backward (``GatedAttentionBlockBwd``: bf16 / fp16, proj_slab save mode) against fp64 autograd -- the unfused
assembly, and its ``fuse_gate_bwd`` and ``fuse_wgrad_overlap`` knobs pinned BITWISE against it.

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

``fuse_wgrad_overlap`` (the two weight-gradient GEMMs on a block-owned side stream, forked / joined through events) is a
SCHEDULING knob: the same launches, so it is pinned by bitwise equality with the in-order block (bf16 / fp16 x dense /
causal, B=3 GQA, composed with ``fuse_gate_bwd``), by the stream-order probe under the knob (both arms: the join is
what the probe's post-block zeroing would expose), by the unchanged CUPTI launch count, by a CUDA-graph capture whose
replay is bitwise the eager run, and -- on any CUDA device -- by a recorder test of the fork / join protocol itself
(which stage goes to which stream, and that the launch stream waits both join events after the last stage).

A QUANTIZED record (the unfused per-tensor FP8 / MXFP8 training forward, ``test_block_training_forward.py``) is the bf16
record with ``h`` as e4m3 codes; this bf16 backward consumes it given the dequantized bf16 ``h`` and weights
(``test_gradients_over_a_quantized_record_match_the_record_seeded_fp64_oracle``).  Its oracle seeds the SDPA stage with the
record's own pre-gate ``O`` and LSE (an autograd Function whose forward VALUE is ``saved.o`` and whose backward is the exact
attention backward over ``saved.lse`` -- B3 / B4 by construction), because the quantized forward's ``O`` carries the kernels'
e4m3 P that no oracle models; the record's slab bands, ``rstd`` and the whole assembly are then held to THE SAME bf16 bounds,
and the cosine against the plain fp64 oracle is printed for the record, never asserted.

Accept tests are ``requires_rubin`` (the block binds ONE engine, the Rubin d256 backward -- AGENTS.md
Rule 9); reject tests build CUDA tensors for a DECLARED block (``requires_cuda``, no compile);
the pure-carve tests run anywhere. ``torch.exp2`` / ``torch.log2`` are deliberately absent from the
tolerance helpers (they fail through nvrtc on cc 10.7).
"""

import dataclasses
import gc
import inspect
import math
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
from cudnn.gated_attention_block.api import _WS_ALIGN, Fp4Format, MxQuantSpec, QuantSpec, _cols, _view  # noqa: E402
from cudnn.gated_attention_block.api_bwd import (  # noqa: E402
    QUANT_CONST_SLOTS,
    QUANT_SCALAR_SLOTS,
    QUANT_SCALAR_STRIDE,
    QUANT_SCALARS_BYTES,
    _BwdIntermediates,
    _GemmStage,
    _plan_bwd_workspace,
)
from cudnn._torch_stream import as_torch_stream  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import RefGeometry, gated_attention_block_reference, qk_norm_rope_reference  # noqa: E402
from gated_block_stream_probe import park_the_default_stream  # noqa: E402
from test_block_training_forward import _alloc_saved, _declare, _declare_quant, _dequantized_bf16_inputs, _run_training, _run_training_quant  # noqa: E402

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
# The knob sets the determinism / stream / workspace pins run under: both knobs are performance-only, so every one of them
# must give the SAME gradients (bitwise) and the same caller-visible stream semantics.
_KNOBS = {
    "unfused": {},
    "fuse_gate_bwd": {"fuse_gate_bwd": True},
    "fuse_wgrad_overlap": {"fuse_wgrad_overlap": True},
    "both": {"fuse_gate_bwd": True, "fuse_wgrad_overlap": True},
}
_KNOB_SETS = pytest.mark.parametrize("knobs", list(_KNOBS.values()), ids=list(_KNOBS))

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


def _twin(res, *, poison=0xFF, **bwd_kw):
    """A second block over the SAME record / dy / inputs as ``res`` (different knobs), compiled and run once into a
    poisoned workspace and NaN-filled gradients; returns ``(blk, ws, grads)`` -- the bitwise comparand of ``res``."""
    inp = res.inp
    blk = GatedAttentionBlockBwd(res.dy, res.saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], res.geom, **bwd_kw)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(poison)
    grads = _alloc_grads(blk, fill=float("nan"))
    _execute(blk, inp, res.saved, res.dy, grads, ws)
    torch.cuda.synchronize()
    return blk, ws, grads


def _adapter_delta(blk, ws) -> torch.Tensor:
    """The SDPA backward's ``delta`` as the block's chain sees it: the adapter's own workspace region (knob off) or the
    block's ``delta`` region B3 wrote (``fuse_gate_bwd``), both ``[B, H_q, S_pad]`` fp32."""
    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import R_DELTA

    lay, impl = blk._layout(), blk._sdpa._impl
    shape = tuple(impl.external_delta_shape)
    if blk.fuse_gate_bwd:
        assert lay.delta >= 0 and lay.delta_shape == shape
        return _view(ws, lay.delta, shape, torch.float32)
    offset, r_shape, _strides = prepared_sm107._regions(impl, prepared_sm107._REGION_SLOTS_F16)[0][R_DELTA]
    assert r_shape == shape and lay.delta == -1
    return _view(ws, lay.sdpa_bwd_ws + offset, shape, torch.float32)


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
@_KNOB_SETS
def test_two_runs_are_bitwise(knobs):
    """Determinism by construction (no atomics anywhere on the chain, fixed-order folds and reduces): two executes
    with the workspace POISONED in between (0xFF bytes = bf16 NaN) give bitwise-equal gradients -- which also proves
    no region depends on prior workspace content (the fused chain's ``delta`` region and the side-stream GEMMs'
    ``gemm_scratch_side`` included).  Under every knob set."""
    res = _backward(dict(_COMMON), batch=2, seq_len=256, **knobs)
    res.ws.fill_(0xFF)
    grads2 = _alloc_grads(res.blk, fill=float("nan"))
    _execute(res.blk, res.inp, res.saved, res.dy, grads2, res.ws)
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), name
            assert torch.equal(ten, res.grads[name]), f"{name} differs across two runs"


@requires_rubin
@_KNOB_SETS
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_a_caller_stream_orders_every_stage(how, knobs):
    """Every stage -- the CuTe-DSL kernels, the four FROST GEMMs and the SDPA backward adapter -- launches on ONE
    stream, the caller's: ambient (``with torch.cuda.stream(s):``) or explicit (``current_stream=``). The default
    stream is parked behind a long spin and the workspace is zeroed on the side stream right after the block, so a
    stage enqueued on the default stream runs late (a late producer leaves zeros for its consumers; a late consumer
    reads the zeros written over the workspace) and the gradients differ from the default-stream run -- which they
    must equal BITWISE (the block is deterministic).  Under ``fuse_gate_bwd`` the delta hand-off (B3 -> the adapter)
    is one more producer / consumer pair on that stream.  Under ``fuse_wgrad_overlap`` the block's OWN side stream
    carries the two wgrad GEMMs: the caller's zeroing right after the block is what a missing JOIN would expose
    (B7 reading a zeroed ``dqkvg`` slab, ``dW_o`` / ``dW_qkvg`` landing after the caller moved on), and the parked
    default stream what a missing FORK or a GEMM escaping to the default stream would."""
    import cuda.bindings.driver as cuda_drv

    res = _backward(dict(_COMMON), batch=1, seq_len=256, **knobs)  # default stream, synchronized
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
            assert torch.equal(ten, res.grads[name]), f"{name}: a stage escaped the caller's stream ({how}, knobs={knobs})"
    if knobs.get("fuse_wgrad_overlap"):
        # the side stream the block used is neither the caller's nor the default one
        used = res.blk._side.side.cuda_stream
        assert used not in (side.cuda_stream, torch.cuda.default_stream().cuda_stream), (used, side.cuda_stream)


@requires_rubin
@pytest.mark.parametrize(
    "dtype, causal, batch",
    [
        (torch.bfloat16, True, 1),
        (torch.bfloat16, False, 1),
        (torch.float16, True, 1),
        (torch.float16, False, 1),
        (torch.bfloat16, True, 3),
        (torch.bfloat16, False, 3),
    ],
    ids=["bf16-causal-b1", "bf16-dense-b1", "fp16-causal-b1", "fp16-dense-b1", "bf16-causal-b3-gqa", "bf16-dense-b3-gqa"],
)
def test_fused_gate_bwd_is_bitwise_the_unfused_block(dtype, causal, batch):
    """``fuse_gate_bwd=True`` computes the SAME function: every gradient ``torch.equal`` the unfused block's over the same
    record (bf16 and fp16, dense and causal, B=1 and B=3 under GQA 8/2), and the ``delta`` the gate-backward kernel wrote
    is bitwise the adapter's own ``dot_do_o`` (read out of the unfused block's adapter scratch) -- the same fp32 products
    summed in the same order over the same bf16-rounded dO.  Both workspaces are poisoned first; the fused block's
    adapter carries no ``delta`` region, the block's own one takes its place, and the total shrinks by nothing but the
    carve alignment (same bytes, moved)."""
    from cudnn.gated_attention_block.api import _WS_ALIGN as align

    geom_kw = {**_COMMON, "is_causal": causal}
    res = _backward(geom_kw, batch=batch, seq_len=256, dtype=dtype)
    fused, ws_f, grads_f = _twin(res, fuse_gate_bwd=True)
    assert fused.fuse_gate_bwd and fused._gate_bwd.want_delta and fused._sdpa.external_delta and fused._sdpa._impl.external_delta
    for name, ten in grads_f.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), name
            assert torch.equal(ten, res.grads[name]), f"{name}: the fused block differs from the unfused one"
    d_unfused, d_fused = _adapter_delta(res.blk, res.ws), _adapter_delta(fused, ws_f)
    assert torch.isfinite(d_fused).all() and torch.equal(d_fused, d_unfused), "the gate kernel's delta is not the adapter's dot_do_o"
    lay_u, lay_f = res.blk._layout(), fused._layout()
    assert lay_f.sdpa_bwd_bytes < lay_u.sdpa_bwd_bytes and lay_f.delta_shape == tuple(fused._sdpa._impl.external_delta_shape)
    delta_bytes = 4 * d_fused.numel()
    assert lay_u.sdpa_bwd_bytes - lay_f.sdpa_bwd_bytes == -(-delta_bytes // 128) * 128
    assert abs(lay_f.total_bytes - lay_u.total_bytes) <= 2 * align


@requires_rubin
@pytest.mark.parametrize(
    "dtype, causal, batch, base",
    [
        (torch.bfloat16, True, 1, {}),
        (torch.bfloat16, False, 1, {}),
        (torch.float16, True, 1, {}),
        (torch.float16, False, 1, {}),
        (torch.bfloat16, True, 3, {}),
        (torch.bfloat16, False, 3, {}),
        (torch.bfloat16, True, 1, {"fuse_gate_bwd": True}),
    ],
    ids=["bf16-causal-b1", "bf16-dense-b1", "fp16-causal-b1", "fp16-dense-b1", "bf16-causal-b3-gqa", "bf16-dense-b3-gqa", "bf16-causal-b1-on-fuse_gate_bwd"],
)
def test_fuse_wgrad_overlap_is_bitwise_the_in_order_block(dtype, causal, batch, base):
    """``fuse_wgrad_overlap=True`` computes the SAME function with the SAME launches: every gradient ``torch.equal`` the
    in-order block's over the same record (bf16 and fp16, dense and causal, B=1 and B=3 under GQA 8/2, and composed
    with ``fuse_gate_bwd`` on in both arms).  The deterministic GEMMs land the same bytes from the side stream; the
    workspace is poisoned first, the gradients NaN-filled.  The side layout appends ``gemm_scratch_side`` (sized to the
    two wgrad plans, never 0) after everything else, and nothing else in the carve moves."""
    geom_kw = {**_COMMON, "is_causal": causal}
    res = _backward(geom_kw, batch=batch, seq_len=256, dtype=dtype, **base)
    on, ws_on, grads_on = _twin(res, fuse_wgrad_overlap=True, **base)
    assert on.fuse_wgrad_overlap and on._side is not None and on._side.side is not None
    assert on._side.side.cuda_stream != torch.cuda.current_stream().cuda_stream  # the executed side stream was not the launch stream
    for name, ten in grads_on.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), name
            assert torch.equal(ten, res.grads[name]), f"{name}: the overlapped block differs from the in-order one"
    lay_i, lay_o = res.blk._layout(), on._layout()
    assert lay_i.gemm_scratch_side == -1 and lay_i.gemm_scratch_side_bytes == 0
    wgrad_need = max(p.workspace_bytes for k, p in on.gemm_plans.items() if k.endswith("_wgrad"))
    assert lay_o.gemm_scratch_side >= 0 and lay_o.gemm_scratch_side_bytes == max(wgrad_need, 1)
    assert lay_o.gemm_scratch_side >= max(lay_o.gemm_scratch + lay_o.gemm_scratch_bytes, lay_o.delta)  # appended LAST
    for f in ("do_gated", "dqkvg", "o_gated", "recompute", "recompute_k", "recompute_v", "dq", "dk", "dv", "sdpa_bwd_ws", "gemm_scratch", "delta"):
        assert getattr(lay_i, f) == getattr(lay_o, f), f
    jit_ws = {k: int(getattr(p.jit, "workspace_bytes", 0) or 0) for k, p in on.gemm_plans.items()}
    print(f"gemm_scratch_side {lay_o.gemm_scratch_side_bytes} B (wgrad plans report {wgrad_need}); the forced-tile JITs carve {jit_ws} bytes of scratch")


@requires_rubin
@_KNOB_SETS
def test_workspace_size_is_honest(knobs):
    """``get_workspace_size()`` is exact and never exceeded: a buffer 4096 B larger keeps its tail
    untouched; two executes allocate nothing (the allocator's allocation COUNTER unchanged after a warm-up -- under ``fuse_wgrad_overlap``
    that is also the pin that the side stream and its events exist from ``compile()``, never per execute); the result
    over a sentinel-filled buffer is bitwise the memoised one; ``gemm_scratch`` covers ``max(plan.workspace_bytes)`` (12 MiB
    at this geometry: the backend heuristic's split-K partials, never launched on the forced JIT path).  Under every knob
    set: the fused layouts append their ``delta`` / ``gemm_scratch_side`` regions LAST, the position where a size slip
    would run past the end."""
    res = _backward(dict(_COMMON), batch=2, seq_len=256, **knobs)
    blk = res.blk
    size = blk.get_workspace_size()
    lay = blk._layout()
    assert size == lay.total_bytes and size % _WS_ALIGN == 0
    plans = blk.gemm_plans
    assert len(plans) == 4
    need_gemm = max(p.workspace_bytes for p in plans.values())
    print(
        f"workspace {size} B; gemm_scratch {lay.gemm_scratch_bytes} B (plans need {need_gemm}); sdpa scratch {lay.sdpa_bwd_bytes} B; side scratch {lay.gemm_scratch_side_bytes} B"
    )
    assert lay.gemm_scratch_bytes >= need_gemm >= 1
    assert lay.sdpa_bwd_bytes == blk._sdpa.scratch_workspace_bytes()
    if knobs.get("fuse_wgrad_overlap"):
        assert lay.gemm_scratch_side_bytes >= max(p.workspace_bytes for k, p in plans.items() if k.endswith("_wgrad"))
    else:
        assert lay.gemm_scratch_side == -1
    ws = torch.full((size + 4096,), 0xAB, dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute(blk, res.inp, res.saved, res.dy, grads, ws)  # warm-up: first-use artefacts, if any
    torch.cuda.synchronize()
    # The allocation pin in the caching allocator's COUNTER form (test_block_training_forward.py): the cumulative allocation
    # count cannot be lowered by an unrelated release and still rises for a temporary the execute frees before returning; the
    # allocator peak is the second witness for such a temporary's bytes.  Every object the execute reads stays alive across it.
    gc.collect()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    n0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    _execute(blk, res.inp, res.saved, res.dy, grads, ws)
    _execute(blk, res.inp, res.saved, res.dy, grads, ws)
    torch.cuda.synchronize()
    n1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    peak = torch.cuda.max_memory_allocated()
    assert n1 == n0, f"the backward made {n1 - n0} CUDA allocation(s) on the execute path (allocation.all.allocated {n0} -> {n1})"
    assert peak <= live, f"a temporary on the backward's execute path: the allocator peak rose from {live} to {peak} bytes"
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
@pytest.mark.parametrize(
    "knobs, seq_len, expected",
    [({}, 256, 15), ({}, 1000, 22), ({"fuse_gate_bwd": True}, 256, 14), ({"fuse_gate_bwd": True}, 1000, 21), ({"fuse_wgrad_overlap": True}, 256, 15)],
    ids=["unfused-256", "unfused-1000", "fuse_gate_bwd-256", "fuse_gate_bwd-1000", "fuse_wgrad_overlap-256"],
)
def test_launch_count_is_honest(seq_len, expected, knobs):
    """CUPTI kernel records of one execute == the launch table: ``12 + c*(2+q)`` (15 at this geometry: c = 1 and q = 1 dQ
    GEMM launch per chunk under the adapter's single-launch dQ rendering -- its ``b_head_group`` is the GQA group; the
    per-member twin would make it g = 4) plus the adapter's padded launches at S = 1000 (+3 q / dO / lse pads, +2 k / v
    pads, +2 GQA fold copy-outs = 22); ONE fewer under ``fuse_gate_bwd`` (the adapter's ``dot_do_o`` is gone: 14 / 21);
    UNCHANGED under ``fuse_wgrad_overlap`` (a scheduling knob: the two wgrad GEMMs move to the side stream, nothing is
    added or removed -- the fork / join events are not kernels).  Profiled on a warm block; no hidden memcpy. The
    formula is ALSO recomputed from the adapter's own facts (chunks, the dQ record's ``b_head_group``, padding,
    zero-fill, the external delta) so a change in either side is visible."""
    from torch.profiler import ProfilerActivity, profile

    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import _dq_launches

    fuse = bool(knobs.get("fuse_gate_bwd", False))
    res = _backward(dict(_COMMON), batch=2, seq_len=seq_len, **knobs)
    blk, g = res.blk, res.geom
    impl = blk._sdpa._impl
    assert impl.external_delta is fuse
    grp = g.h_q // g.h_kv
    c = (blk.batch // impl._b_chunk) * (g.h_q // impl._qh_chunk)
    dq_launches = _dq_launches(grp, impl._dq_b_head_group)  # 1 under the single-launch dQ rendering, grp per group member
    stage1 = 1 + (0 if impl.external_delta else 1)  # the seq_kv fill, + the adapter's own dot_do_o unless the caller's delta replaces it
    formula = 9 + stage1 + c * (2 + dq_launches) + (1 if grp > 1 else 0) + (3 if impl._q_padded else 0) + (2 if impl._kv_padded else 0)
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
        # the fusion knob threads through (and is part of the cache key: a DIFFERENT compiled block), bitwise the class path
        from cudnn.gated_attention_block.api_bwd import _BWD_CACHE

        n_cached = len(_BWD_CACHE)
        out3 = gated_attention_block_backward(
            res.dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], res.geom, fuse_gate_bwd=True
        )
        torch.cuda.synchronize()
        assert len(_BWD_CACHE) == n_cached + 1 and torch.equal(out3["dh"], res.grads["dh"])
        assert any(b.fuse_gate_bwd for b in _BWD_CACHE.values())
        # the scheduling knob threads through too (its own cache entry), over EVERY gradient: the weights' grads back on
        for t in (inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
            t.requires_grad_(True)
        out4 = gated_attention_block_backward(
            res.dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], res.geom, fuse_wgrad_overlap=True
        )
        torch.cuda.synchronize()
        assert len(_BWD_CACHE) == n_cached + 2 and all(torch.equal(out4[k], res.grads[k]) for k in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"))
        assert any(b.fuse_wgrad_overlap for b in _BWD_CACHE.values())
        # FROZEN weights + the knob: no weight-gradient GEMM exists, so the wrapper runs the in-order block (the typed
        # decline is the class path's, for an explicit need_* declaration) -- the EFFECTIVE knob is in the cache key, so
        # this is out2's cached block, no new entry
        for t in (inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"]):
            t.requires_grad_(False)
        n_cached = len(_BWD_CACHE)
        out5 = gated_attention_block_backward(
            res.dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], res.geom, fuse_wgrad_overlap=True
        )
        torch.cuda.synchronize()
        assert len(_BWD_CACHE) == n_cached and out5["dw_qkvg"] is None and out5["dw_o"] is None and out5["dw_q_norm"] is None
        assert torch.equal(out5["dh"], res.grads["dh"])
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

        def release_workspace_views(self):
            pass  # the wrapper drops the block's cached views of its per-call workspace at return; no stream work

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
@pytest.mark.parametrize("knobs", [_KNOBS["unfused"], _KNOBS["fuse_wgrad_overlap"], _KNOBS["both"]], ids=["unfused", "fuse_wgrad_overlap", "both"])
def test_cuda_graph_capture_replays_bitwise(knobs):
    """One ``execute`` captured into a CUDA graph on a side torch stream replays bitwise the eager run -- and recomputes
    NEW inputs written through the captured pointers.  Under ``fuse_wgrad_overlap`` the fork / join (an event recorded
    on the capturing stream, waited by the block's side stream, and the join the launch stream waits before ``execute``
    returns) is the canonical cross-stream capture pattern: the side stream joins the capture and is joined back before
    it ends, so the capture succeeds and the replay carries both wgrad GEMMs.  The capture itself launches nothing (the
    NaN-filled gradients are still NaN after it); the block allocates nothing, so no private-pool allocation is
    captured either."""
    res = _backward(dict(_COMMON), batch=1, seq_len=256, **knobs)
    blk = res.blk
    dy2 = res.dy.clone()
    ws = torch.empty_like(res.ws)
    grads = _alloc_grads(blk, fill=float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _execute(blk, res.inp, res.saved, dy2, grads, ws)  # warm-up on the capture stream (torch's own capture recipe)
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
            _execute(blk, res.inp, res.saved, dy2, grads, ws)
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isnan(ten).all(), f"{name}: the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.equal(ten, res.grads[name]), f"{name}: the replay differs from the eager run (knobs={knobs})"
        # new inputs through the captured pointers: a second replay == a fresh eager run over the new dy
        dy3 = _make_dy(res.out, seed=7)
        dy2.copy_(dy3)
        ws.fill_(0xFF)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        ref = _alloc_grads(blk, fill=float("nan"))
        ws_ref = torch.empty_like(ws).fill_(0xFF)
        _execute(blk, res.inp, res.saved, dy3, ref, ws_ref)
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isfinite(ten).all() and torch.equal(ten, ref[name]), f"{name}: the replay over new inputs differs from eager (knobs={knobs})"
    finally:  # on every path: a graph left to the cyclic GC resets itself inside a later test's capture (_err)
        graph.reset()


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
    # fuse_gate_bwd: the fp32 delta region, appended LAST (after gemm_scratch), sized (B, H_q, S_pad) x 4 from the adapter's shape
    fused = _plan_bwd_workspace(
        g,
        b,
        s,
        torch.bfloat16,
        RecomputePolicy.SAVE_ALL,
        need=dict(dw_o=False, dw_norms=False),
        sdpa_bwd_bytes=1000,
        gemm_scratch_bytes=4096,
        delta_shape=(b, g.h_q, 384),
    )
    assert lean.delta == -1 and lean.delta_shape == ()
    assert fused.delta == lean.total_bytes and fused.delta_shape == (b, g.h_q, 384) and fused.total_bytes == lean.total_bytes + al(b * g.h_q * 384 * 4)
    with pytest.raises(ValueError, match="delta_shape"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.SAVE_ALL, need=dict(dw_norms=False), sdpa_bwd_bytes=1000, delta_shape=(b, g.h_q, s - 1))
    with pytest.raises(ValueError, match="delta_shape"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.SAVE_ALL, need=dict(dw_norms=False), sdpa_bwd_bytes=1000, delta_shape=(b, g.h_kv, s))
    # fuse_wgrad_overlap: the side-stream GEMMs' own scratch, appended LAST (after delta), max(.., 1) like gemm_scratch
    assert fused.gemm_scratch_side == -1 and fused.gemm_scratch_side_bytes == 0 and lean.gemm_scratch_side == -1
    for want, side_bytes in ((1, 0), (12345, 12345)):
        side = _plan_bwd_workspace(
            g,
            b,
            s,
            torch.bfloat16,
            RecomputePolicy.SAVE_ALL,
            need=dict(dw_o=False, dw_norms=False),
            sdpa_bwd_bytes=1000,
            gemm_scratch_bytes=4096,
            delta_shape=(b, g.h_q, 384),
            side_gemm_scratch_bytes=side_bytes,
        )
        assert side.gemm_scratch_side == fused.total_bytes and side.gemm_scratch_side_bytes == want
        assert side.total_bytes == fused.total_bytes + al(want) and side.delta == fused.delta and side.gemm_scratch == fused.gemm_scratch


@requires_cuda
def test_fuse_gate_bwd_knob_is_wired_at_declaration():
    """The knob (appended, default False) reaches both stages at construction -- the gate-backward kernel's ``want_delta``
    and the adapter's ``external_delta`` -- and nothing else changes; no compile, any CUDA device."""
    off = _declare_bwd(dict(_COMMON), 1, 256).blk
    on = _declare_bwd(dict(_COMMON), 1, 256, fuse_gate_bwd=True).blk
    assert off.fuse_gate_bwd is False and off._gate_bwd.want_delta is False and off._sdpa.external_delta is False
    assert on.fuse_gate_bwd is True and on._gate_bwd.want_delta is True and on._sdpa.external_delta is True
    assert on._sdpa.delta_shape == (1, _COMMON["h_q"], 256) and on._sdpa._impl.external_delta is True
    assert [type(st).__name__ for st in on._stages] == [type(st).__name__ for st in off._stages]


@requires_cuda
def test_fuse_wgrad_overlap_knob_is_wired_at_declaration():
    """The knob (appended, keyword-only, default False) is a declaration fact: stored, no side stream before ``compile()``
    (Rule 1: the stream and its events belong to the compiled block), the stage list unchanged (a scheduling knob adds
    no stage), and a typed ``ValueError`` at ``check_support`` when no weight-gradient GEMM exists to overlap
    (``need_dw_o=False, need_dw_qkvg=False``) -- before the Rubin gate, so it fires on any CUDA device.  Each
    single-wgrad declaration is accepted (one GEMM to overlap is work enough)."""
    off = _declare_bwd(dict(_COMMON), 1, 256).blk
    on = _declare_bwd(dict(_COMMON), 1, 256, fuse_wgrad_overlap=True).blk
    assert off.fuse_wgrad_overlap is False and off._side is None
    assert on.fuse_wgrad_overlap is True and on._side is None
    assert [type(st).__name__ for st in on._stages] == [type(st).__name__ for st in off._stages]
    none = _declare_bwd(dict(_COMMON), 1, 256, fuse_wgrad_overlap=True, need_dw_o=False, need_dw_qkvg=False).blk
    with pytest.raises(ValueError, match="fuse_wgrad_overlap=True with need_dw_o=False and need_dw_qkvg=False"):
        none.check_support()
    for kw in (dict(need_dw_o=False), dict(need_dw_qkvg=False)):
        one = _declare_bwd(dict(_COMMON), 1, 256, fuse_wgrad_overlap=True, **kw).blk
        try:
            one.check_support()
        except NotImplementedError as exc:  # the Rubin gate (or a stage's) on a non-Rubin device: the knob was accepted
            assert "fuse_wgrad_overlap" not in str(exc)
        except ValueError as exc:
            assert "fuse_wgrad_overlap" not in str(exc)


def _wire_bwd_recorders(dec, sink, touch=None) -> None:
    """A DECLARED backward made executable on ANY CUDA device: a small hand-planned workspace, every stage's ``execute``
    (and the norm reduce) replaced by a recorder that calls ``sink((name, launch-stream handle))``, and -- under
    ``fuse_wgrad_overlap`` -- the real ``_WgradSideStream`` the compiled block would own.  The protocol tests of the
    knob run on this (the artifacts need Rubin; the fork / join protocol is host logic).  With ``touch`` (a two-element
    CUDA tensor) every recorder also runs one tiny kernel on its stage's stream -- ``touch[0] += 1`` for a launch-stream
    stage, ``touch[1] += 1`` for a side-stream one; the two streams run concurrently, so they must not share an element
    -- so a CUDA-graph capture of the recorder-wired block has real nodes and its replay is observable.  Two LAUNCH
    streams share element 0: a caller orders their executes itself (a host synchronize between warm-ups)."""
    from cudnn.gated_attention_block import api_bwd as api_bwd_mod

    blk, g = dec.blk, dec.geom
    blk._ws = _plan_bwd_workspace(
        g,
        dec.batch,
        dec.seq_len,
        torch.bfloat16,
        RecomputePolicy.RECOMPUTE_QK_PRE,
        need=dict(dw_o=True, dw_norms=True),
        sdpa_bwd_bytes=4096,
        gemm_scratch_bytes=1,
        n_ctas_q=4,
        n_ctas_k=2,
        side_gemm_scratch_bytes=1 if blk.fuse_wgrad_overlap else None,
    )
    blk._compiled_kernel = blk._ws
    if blk.fuse_wgrad_overlap:
        blk._side = api_bwd_mod._WgradSideStream(blk.device)

    side_handle = blk._side.handle if blk.fuse_wgrad_overlap else None

    def rec(name, key):
        def f(*a, **k):
            handle = int(k[key])
            if touch is not None:
                with torch.cuda.stream(as_torch_stream(handle, touch.device)):
                    touch[1 if handle == side_handle else 0].add_(1)
            sink((name, handle))

        return f

    for name, attr, key in (
        ("B2", "_out_proj_dgrad", "stream"),
        ("B3", "_gate_bwd", "stream"),
        ("B1", "_out_proj_wgrad", "stream"),
        ("recompute", "_recompute_qk", "current_stream"),
        ("compact_v", "_compact_v", "current_stream"),
        ("B4", "_sdpa", "stream"),
        ("B5+B6", "_norm_bwd", "stream"),
        ("B7", "_qkv_gate_wgrad", "stream"),
        ("B8", "_qkv_gate_dgrad", "stream"),
    ):
        setattr(getattr(blk, attr), "execute", rec(name, key))
    blk._norm_bwd.reduce = rec("reduce", "stream")


@requires_cuda
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_fuse_wgrad_overlap_fork_join_protocol(how, monkeypatch):
    """The fork / join protocol of ``fuse_wgrad_overlap`` on ANY CUDA device, with every stage replaced by a recorder
    (the artifacts need Rubin; the protocol is host logic): B1 and B7 launch on the block's side stream and every other
    stage on the launch stream (ambient or explicit); B1's fork event is recorded on the launch stream AFTER B3 and
    waited by the side stream BEFORE B1; B7's after B5+B6 and before B7 (so B7 overlaps the reduce and B8); each join
    event is recorded on the side stream right after its GEMM; and the launch stream waits BOTH join events after the
    last stage (B8) -- the last things ``execute`` does.  The side stream is never the launch stream.  With the knob
    off the same recorders see every stage on the launch stream and no event at all."""
    import cuda.bindings.driver as cuda_drv

    real_record, real_wait = torch.cuda.Event.record, torch.cuda.Stream.wait_event
    logs = []

    def record(ev, stream=None):
        st = stream if stream is not None else torch.cuda.current_stream()
        for log in logs:
            log.append(("record", id(ev), int(st.cuda_stream)))
        return real_record(ev, stream)

    def wait_event(stream, ev):
        for log in logs:
            log.append(("wait", int(stream.cuda_stream), id(ev)))
        return real_wait(stream, ev)

    monkeypatch.setattr(torch.cuda.Event, "record", record)
    monkeypatch.setattr(torch.cuda.Stream, "wait_event", wait_event)
    launch = torch.cuda.Stream()
    L = launch.cuda_stream

    def run(dec, log):
        logs.clear()
        logs.append(log)
        ws = torch.empty(dec.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
        grads = _alloc_grads(dec.blk)
        if how == "ambient":
            with torch.cuda.stream(launch):
                _execute(dec.blk, dec.inp, dec.saved, dec.dy, grads, ws)
        else:
            _execute(dec.blk, dec.inp, dec.saved, dec.dy, grads, ws, current_stream=cuda_drv.CUstream(L))
        logs.clear()

    # knob ON
    on = _declare_bwd(dict(_COMMON), 1, 256, fuse_wgrad_overlap=True)
    log = []
    _wire_bwd_recorders(on, log.append)
    run(on, log)
    side = on.blk._side
    S = side.handle
    assert S != L and S != torch.cuda.default_stream().cuda_stream
    stages = [(e[0], e[1]) for e in log if e[0] not in ("record", "wait")]
    assert [n for n, _ in stages] == ["B2", "B3", "B1", "recompute", "compact_v", "B4", "B5+B6", "B7", "reduce", "B8"], stages
    assert all(st == (S if n in ("B1", "B7") else L) for n, st in stages), stages
    idx = {e[0]: i for i, e in enumerate(log) if e[0] not in ("record", "wait")}
    fo, jo, fq, jq = (id(side.ev_fork["o"]), id(side.ev_join["o"]), id(side.ev_fork["qkvg"]), id(side.ev_join["qkvg"]))
    events = [(i, e) for i, e in enumerate(log) if e[0] in ("record", "wait")]
    assert [e for _, e in events] == [
        ("record", fo, L),  # fork dW_o: after B3 ...
        ("wait", S, fo),
        ("record", jo, S),  # ... join recorded right after B1 on the side stream
        ("record", fq, L),  # fork dW_qkvg: after B5+B6 ...
        ("wait", S, fq),
        ("record", jq, S),  # ... join recorded right after B7
        ("wait", L, jo),  # the launch stream waits both joins after B8 -- the last things execute does
        ("wait", L, jq),
    ], events
    pos = {e: i for i, e in events}
    assert idx["B3"] < pos[("record", fo, L)] < pos[("wait", S, fo)] < idx["B1"] < pos[("record", jo, S)] < idx["recompute"]
    assert idx["B5+B6"] < pos[("record", fq, L)] < pos[("wait", S, fq)] < idx["B7"] < pos[("record", jq, S)] < idx["reduce"] < idx["B8"]
    assert idx["B8"] < pos[("wait", L, jo)] < pos[("wait", L, jq)] == len(log) - 1
    # knob OFF: the same recorders, one stream, no event
    off = _declare_bwd(dict(_COMMON), 1, 256)
    log_off = []
    _wire_bwd_recorders(off, log_off.append)
    run(off, log_off)
    assert [e[0] for e in log_off] == ["B2", "B3", "B1", "recompute", "compact_v", "B4", "B5+B6", "reduce", "B7", "B8"], log_off
    assert all(e[1] == L for e in log_off), log_off


@requires_cuda
def test_wgrad_side_stream_is_dedicated_and_released():
    """The side stream of ``fuse_wgrad_overlap`` is the block's OWN: created through the driver -- never one of torch's
    32 round-robin pool streams, which a caller's ``torch.cuda.Stream()`` could coincide with -- non-blocking, at the
    device's lowest priority (a filler below any launch stream), wrapped once as an ``ExternalStream`` for the event
    pairs; the fork / join pair allocates nothing (the events exist from construction); and the stream is released
    (``cuStreamDestroy``) when the owner is collected -- no CUDA object outlives the block."""
    import gc

    import cuda.bindings.driver as cuda_drv
    from cudnn.gated_attention_block.api_bwd import _WgradSideStream

    side = _WgradSideStream(torch.device("cuda"))
    pool = {torch.cuda.Stream().cuda_stream for _ in range(40)}  # the whole default-priority pool (32 streams) and then some
    assert side.handle not in pool and side.side.cuda_stream == side.handle
    assert side.handle not in (0, torch.cuda.default_stream().cuda_stream, torch.cuda.current_stream().cuda_stream)
    err, flags = cuda_drv.cuStreamGetFlags(cuda_drv.CUstream(side.handle))
    assert int(err) == 0 and int(flags) & int(cuda_drv.CUstream_flags.CU_STREAM_NON_BLOCKING), (err, flags)
    err, prio = cuda_drv.cuStreamGetPriority(cuda_drv.CUstream(side.handle))
    err2, least, _greatest = cuda_drv.cuCtxGetStreamPriorityRange()
    assert int(err) == 0 and int(err2) == 0 and int(prio) == int(least) == side.priority, (prio, least, side.priority)
    launch = torch.cuda.current_stream()
    gc.collect()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    n0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    for tag in side.TAGS:
        with side.issue(launch, tag):
            pass
        side.join(launch, tag)
    torch.cuda.synchronize()
    n1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    assert n1 == n0, f"a fork / join made {n1 - n0} CUDA allocation(s) (allocation.all.allocated {n0} -> {n1})"
    assert torch.cuda.max_memory_allocated() <= live, "a temporary on the fork / join path"
    fin = side._finalizer
    assert fin.alive
    del side
    gc.collect()
    assert not fin.alive, "the side stream was not released with its owner"
    again = _WgradSideStream(torch.device("cuda"))  # a fresh owner gets a fresh stream
    assert again.handle not in pool
    del again


@requires_cuda
def test_fuse_wgrad_overlap_is_reentrant_across_launch_streams(monkeypatch):
    """ONE block (the convenience wrapper caches one per declaration for the process) driven from TWO host threads on
    TWO launch streams keeps every fork its own.  Recorder-wired like the protocol test, with every event record slowed
    down (a sleep after it) so the threads interleave at exactly the hazardous point: the record a thread's side-stream
    wait consumes -- the LATEST record of that fork event before the wait -- must be its own, on its own launch stream;
    a stranger's record would point the side stream at the other launch stream's producer and this thread's dW GEMM
    could run before its own operand exists.  Both executes see B1 / B7 on the one side stream and every other stage on
    their own launch stream, in order.  The join needs no such pin (a record on the one side stream is after this
    execute's GEMM whichever thread issued it); each thread's own join record precedes its join wait."""
    import threading
    import time

    import cuda.bindings.driver as cuda_drv

    dec = _declare_bwd(dict(_COMMON), 1, 256, fuse_wgrad_overlap=True)
    log = []
    guard = threading.Lock()

    def sink(entry):
        with guard:
            log.append((threading.get_ident(),) + tuple(entry) + (None,) * (3 - len(entry)))

    _wire_bwd_recorders(dec, sink)
    side = dec.blk._side
    S = side.handle
    forks = {id(ev) for ev in side.ev_fork.values()}
    joins = {id(ev) for ev in side.ev_join.values()}
    real_record, real_wait = torch.cuda.Event.record, torch.cuda.Stream.wait_event

    def record(ev, stream=None):
        st = stream if stream is not None else torch.cuda.current_stream()
        out = real_record(ev, stream)
        sink(("record", id(ev), int(st.cuda_stream)))
        time.sleep(0.02)  # the window between a fork's record and its wait: without the fork lock the other thread's record lands here
        return out

    def wait_event(stream, ev):
        sink(("wait", int(stream.cuda_stream), id(ev)))
        return real_wait(stream, ev)

    monkeypatch.setattr(torch.cuda.Event, "record", record)
    monkeypatch.setattr(torch.cuda.Stream, "wait_event", wait_event)
    launches = [torch.cuda.Stream(), torch.cuda.Stream()]
    assert launches[0].cuda_stream != launches[1].cuda_stream
    gate = threading.Barrier(2)
    errors = []

    def worker(launch):
        try:
            ws = torch.empty(dec.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
            grads = _alloc_grads(dec.blk)
            gate.wait()
            _execute(dec.blk, dec.inp, dec.saved, dec.dy, grads, ws, current_stream=cuda_drv.CUstream(launch.cuda_stream))
        except Exception as exc:  # noqa: BLE001 -- reported by the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(launch,)) for launch in launches]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    assert not errors, errors
    by_thread = {}
    for tid, kind, a, b in log:
        by_thread.setdefault(tid, []).append((kind, a, b))
    assert len(by_thread) == 2, by_thread.keys()
    stage_names = ["B2", "B3", "B1", "recompute", "compact_v", "B4", "B5+B6", "B7", "reduce", "B8"]
    own_launch = {}
    for tid, entries in by_thread.items():
        stages = [(kind, a) for kind, a, _b in entries if kind not in ("record", "wait")]
        L = stages[0][1]
        own_launch[tid] = L
        assert [n for n, _ in stages] == stage_names, stages
        assert all(st == (S if n in ("B1", "B7") else L) for n, st in stages), (tid, stages)
        for kind, a, b in entries:
            if kind == "record":
                assert (a in forks and b == L) or (a in joins and b == S), (tid, kind, a, b)
            elif kind == "wait":
                assert (a == S and b in forks) or (a == L and b in joins), (tid, kind, a, b)
    assert len(set(own_launch.values())) == 2 and set(own_launch.values()) == {launches[0].cuda_stream, launches[1].cuda_stream}
    # the fork pin, over the GLOBAL order: the latest record of the event before each side-stream wait is this thread's
    for i, (tid, kind, a, b) in enumerate(log):
        if kind == "wait" and a == S:
            latest = max(j for j, (_t, k2, a2, _b2) in enumerate(log[:i]) if k2 == "record" and a2 == b)
            assert log[latest][0] == tid and log[latest][3] == own_launch[tid], (i, log[latest], log[i])
        if kind == "wait" and b in joins:
            assert any(k2 == "record" and a2 == b and b2 == S for k2, a2, b2 in by_thread[tid][: by_thread[tid].index((kind, a, b))]), (tid, kind, a, b)


def _err(exc: BaseException) -> str:
    """A thread's error for the main thread's assertions, as a STRING.  Never store the exception object: its traceback
    keeps the thread's frame alive, the frame keeps every local -- a ``torch.cuda.CUDAGraph`` among them -- and the dict
    that stores the exception closes the cycle, so the graph outlives the test and is reclaimed by the cyclic GC at an
    uncontrolled later moment; a ``CUDAGraph.reset()`` that lands inside a LATER test's capture invalidates it (the
    intermittent ``operation not permitted when stream is capturing (function reset)`` warning, two device classes).
    Same rule for the graphs themselves: reset every one explicitly, on every path -- the capture tests do it in a
    ``finally`` that also releases every parked thread and joins them, so a failing assertion leaks neither a thread
    nor a capture nor a graph into the next test."""
    return f"{type(exc).__name__}: {exc}"


@requires_cuda
def test_wgrad_side_stream_refuses_a_foreign_capture():
    """The ONE side stream of a block cannot be shared between a CUDA-graph capture and anything else.  A capturing
    thread's fork puts the side stream inside its capture until the join; in that window an eager fork from another
    thread (1), an eager join (2) and a fork from a SECOND capture (3) are each a typed ``RuntimeError`` raised BEFORE
    the record / wait touches the capture -- so the capture survives: its own second fork and its joins pass (both
    streams report the same capture id), it ends, launches nothing, and replays both side-stream kernels; afterwards
    the eager fork / join pass again.  Without the guard (verified): (1) is ``cudaErrorStreamCaptureIsolation`` for
    the eager thread and the capture is INVALIDATED; (2) joins the eager launch stream to the capture silently (its
    later work lands in the graph); (3) invalidates the capture too.  Two torch details make (3) reachable at all:
    the second capture is begun with the low-level ``CUDAGraph.capture_begin`` (``torch.cuda.graph.__enter__``
    synchronizes the device, which invalidates ANY open capture -- through torch's own recipe the case cannot arise),
    and the first capture runs ``thread_local`` (a ``global`` one forbids the other thread's ``capture_end``
    bookkeeping and is invalidated by it).  The guard reads the same capture status under every mode (verified on two
    device classes); the capturing thread's own forks and joins under the default ``global`` mode are what
    ``test_cuda_graph_capture_replays_bitwise`` exercises.  Every graph is reset explicitly and thread errors are
    recorded as strings (``_err``): a graph left to the cyclic GC invalidates a later capture."""
    import threading

    from cudnn.gated_attention_block.api_bwd import _WgradSideStream

    side = _WgradSideStream(torch.device("cuda"))
    buf = torch.zeros(1, device="cuda")
    s_a, s_b, s_c = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    a_inside = threading.Event()
    done = threading.Barrier(3, timeout=120)
    out = {}

    def thread_a():  # the capture: fork -> side kernel -> join record, held open for B and C, then its own second fork
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=s_a, capture_error_mode="thread_local"):
                with side.issue(s_a, "o"):  # the side stream is now inside A's capture
                    with torch.cuda.stream(side.side):
                        buf.add_(1)
                a_inside.set()
                done.wait()
                with side.issue(s_a, "qkvg"):  # the capturing thread's own second fork: both streams in the same capture
                    with torch.cuda.stream(side.side):
                        buf.add_(1)
                side.join(s_a, "o")
                side.join(s_a, "qkvg")
            out["graph"] = graph
        except Exception as exc:  # noqa: BLE001 -- reported by the main thread
            out["A"], out["A_first"] = _err(exc), repr(exc.__context__)  # capture_end's error REPLACES the body's: keep the first
            graph.reset()
            a_inside.set()
            done.abort()

    def thread_b():  # (1) an eager fork and (2) an eager join on another launch stream, inside A's window
        try:
            a_inside.wait(timeout=60)
            try:
                with side.issue(s_b, "o"):
                    out["B_fork"] = None  # unreachable: the guard raises before the section opens
            except RuntimeError as exc:
                out["B_fork"] = _err(exc)
            try:
                side.join(s_b, "o")
                out["B_join"] = None
            except RuntimeError as exc:
                out["B_join"] = _err(exc)
        finally:
            done.wait()

    def thread_c():  # (3) a fork from a SECOND capture, inside A's window (low-level begin / end: no device synchronize)
        try:
            a_inside.wait(timeout=60)
            graph2 = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s_c):
                graph2.capture_begin()
                try:
                    buf.add_(1000)  # a node of C's own (captured, never replayed) so its capture is not empty
                    with side.issue(s_c, "qkvg"):
                        out["C_fork"] = None  # unreachable
                except RuntimeError as exc:
                    out["C_fork"] = _err(exc)
                finally:
                    try:
                        graph2.capture_end()
                    finally:
                        graph2.reset()  # never leave an instantiated graph to the GC (_err)
        except Exception as exc:  # noqa: BLE001 -- reported by the main thread
            out["C_other"] = _err(exc)
        finally:
            done.wait()

    threads = [threading.Thread(target=f) for f in (thread_a, thread_b, thread_c)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join(timeout=180)
        assert "A" not in out, ("the capture failed", out.get("A"), "first error in its body:", out.get("A_first"))
        for key in ("B_fork", "B_join"):
            assert str(out.get(key)).startswith("RuntimeError: ") and "an eager execute cannot share it" in str(out.get(key)), (key, out.get(key))
        assert "C_other" not in out, out.get("C_other")
        assert str(out.get("C_fork")).startswith("RuntimeError: ") and "two concurrent captures" in str(out.get("C_fork")), out.get("C_fork")
        torch.cuda.synchronize()
        assert int(buf.item()) == 0, "the capture launched work"
        out["graph"].replay()
        torch.cuda.synchronize()
        assert int(buf.item()) == 2, "the replay does not carry both side-stream kernels"
        with side.issue(s_b, "o"):  # the capture has ended: the eager issue / join pass again
            pass
        side.join(s_b, "o")
        torch.cuda.synchronize()
    finally:  # a failing assertion must leak neither the capture nor the graph into the next test (_err)
        a_inside.set()
        done.abort()  # frees any thread still at the barrier
        for t in threads:
            t.join(timeout=180)
        if "graph" in out:
            out["graph"].reset()


@requires_cuda
def test_wgrad_side_stream_refuses_a_dead_capture():
    """A capture that has been INVALIDATED still holds the side stream, and the driver reports no id for it: an eager
    fork onto it from another thread is the typed ``RuntimeError`` naming the dead capture -- without the guard the
    eager wait onto the dead capture's stream proceeds silently (no error anywhere; the eager execute's GEMM would be
    swallowed by a capture that never instantiates).  The capturing thread invalidates its own capture with an event
    query (a "potentially unsafe" call under any non-relaxed mode; its ``capture_end`` then reports the error).  Once
    that capture has ended the side stream is free again and the eager fork / join pass."""
    import threading

    from cudnn.gated_attention_block.api_bwd import _WgradSideStream

    side = _WgradSideStream(torch.device("cuda"))
    s_a, s_b = torch.cuda.Stream(), torch.cuda.Stream()
    a_dead, b_done = threading.Event(), threading.Event()
    out = {}

    def thread_a():
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=s_a, capture_error_mode="thread_local"):
                with side.issue(s_a, "o"):  # the side stream is inside A's capture
                    pass
                probe = torch.cuda.Event()
                probe.record(torch.cuda.Stream())
                try:
                    probe.query()  # the capturing thread's own unsafe call: the capture is now INVALIDATED
                except RuntimeError as exc:
                    out["A_invalidator"] = _err(exc)
                a_dead.set()
                b_done.wait(timeout=60)
            out["A_end"] = None  # unreachable: capture_end raises for an invalidated capture
        except RuntimeError as exc:
            out["A_end"] = _err(exc)
            a_dead.set()
        finally:
            graph.reset()  # never leave a graph to the GC (_err)

    def thread_b():
        try:
            a_dead.wait(timeout=60)
            try:
                with side.issue(s_b, "o"):
                    out["B_fork"] = None  # unreachable
            except RuntimeError as exc:
                out["B_fork"] = _err(exc)
        finally:
            b_done.set()

    ta, tb = threading.Thread(target=thread_a), threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta.join(timeout=180)
    tb.join(timeout=180)
    assert "capturing" in str(out.get("A_invalidator")), out.get("A_invalidator")  # the invalidating call itself erred
    assert str(out.get("B_fork")).startswith("RuntimeError: ") and "has been invalidated" in str(out.get("B_fork")), out.get("B_fork")
    assert out.get("A_end") is not None and "capture" in str(out.get("A_end")), out.get("A_end")  # the dead capture's own capture_end reports it
    with side.issue(s_b, "o"):  # the capture has ended: the side stream is free again
        pass
    side.join(s_b, "o")
    torch.cuda.synchronize()


@requires_cuda
def test_fuse_wgrad_overlap_capture_and_eager_executes_do_not_overlap():
    """ONE compiled block (the convenience wrapper caches one per declaration for the process), recorder-wired like the
    protocol test with every stage one tiny kernel on its stream: thread A captures ``execute`` into a CUDA graph and
    is held open past its first fork (inside the SDPA stage) while thread B executes eagerly on its own launch stream
    -> B's execute is the typed ``RuntimeError`` at its first fork (the block's side stream is inside A's capture; B
    ran its two stages before the fork and nothing on the side stream), and A's capture is untouched: its second fork
    and both joins pass, the capture launches nothing, and the replay runs all ten stages.  Once A's capture has ended
    the same eager execute passes.  The message names the rule: capture and eager executes of one compiled block must
    not overlap.  The capture is ``thread_local``: a ``global`` one is invalidated by a "potentially unsafe" CUDA call
    (a device synchronize, an event query, a ``cudaFree``) from ANY thread of the process, and this test holds its
    capture open across another thread's work -- in a shared pytest process (allocator cache trims, profiler
    teardown) an exposure the product's own capture, a few milliseconds of enqueue, never has.  The guard reads the
    same capture status under every mode."""
    import threading

    import cuda.bindings.driver as cuda_drv

    dec = _declare_bwd(dict(_COMMON), 1, 256, fuse_wgrad_overlap=True)
    touch = torch.zeros(2, device="cuda")  # [launch-stream stage kernels, side-stream stage kernels]
    log, guard = [], threading.Lock()

    def sink(entry):
        with guard:
            log.append((threading.get_ident(),) + tuple(entry))

    _wire_bwd_recorders(dec, sink, touch=touch)
    blk = dec.blk
    side_handle = blk._side.handle
    hold = {"tid": None}
    a_inside, b_done = threading.Event(), threading.Event()
    sdpa_rec = blk._sdpa.execute

    def sdpa_held(*a, **k):  # A's SDPA stage -- inside its capture, past the dW_o fork: hold the capture open for B
        sdpa_rec(*a, **k)
        if threading.get_ident() == hold["tid"]:
            a_inside.set()
            b_done.wait(timeout=60)

    blk._sdpa.execute = sdpa_held
    s_a, s_b = torch.cuda.Stream(), torch.cuda.Stream()
    ws_a = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    ws_b = torch.empty_like(ws_a)
    grads_a, grads_b = _alloc_grads(blk), _alloc_grads(blk)
    # warm-up on both streams (torch's capture recipe; nothing is created or loaded inside the capture window); the two
    # launch streams share touch[0], so their executes are ordered by a host synchronize, as is the fill before them
    torch.cuda.synchronize()
    _execute(blk, dec.inp, dec.saved, dec.dy, grads_a, ws_a, current_stream=cuda_drv.CUstream(s_a.cuda_stream))
    torch.cuda.synchronize()
    _execute(blk, dec.inp, dec.saved, dec.dy, grads_b, ws_b, current_stream=cuda_drv.CUstream(s_b.cuda_stream))
    torch.cuda.synchronize()
    assert touch.tolist() == [16.0, 4.0], touch.tolist()  # two warm-ups x (8 launch-stream stages + 2 side-stream GEMMs)
    log.clear()
    out = {}

    def thread_a():
        hold["tid"] = threading.get_ident()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=s_a, capture_error_mode="thread_local"):
                _execute(blk, dec.inp, dec.saved, dec.dy, grads_a, ws_a)
            out["graph"] = graph
        except Exception as exc:  # noqa: BLE001 -- reported by the main thread
            out["A"], out["A_first"] = _err(exc), repr(exc.__context__)  # capture_end's error REPLACES the body's: keep the first
            graph.reset()
            a_inside.set()

    def thread_b():
        try:
            a_inside.wait(timeout=60)
            _execute(blk, dec.inp, dec.saved, dec.dy, grads_b, ws_b, current_stream=cuda_drv.CUstream(s_b.cuda_stream))
            out["B"] = None
        except Exception as exc:  # noqa: BLE001
            out["B"], out["B_first"] = _err(exc), repr(exc.__context__)
        finally:
            b_done.set()

    ta, tb = threading.Thread(target=thread_a), threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    try:
        ta.join(timeout=180)
        tb.join(timeout=180)
        assert "A" not in out, ("the capture failed", out.get("A"), "first error in its body:", out.get("A_first"))
        assert str(out.get("B")).startswith("RuntimeError: ") and "capture and eager executes of one compiled block must not overlap" in str(out.get("B")), (
            out.get("B"),
            out.get("B_first"),
        )
        torch.cuda.synchronize()
        assert touch.tolist() == [18.0, 4.0], ("the capture launched work, or B ran past its first fork", touch.tolist())
        out["graph"].replay()
        torch.cuda.synchronize()
        assert touch.tolist() == [26.0, 6.0], ("the replay does not carry all ten stages", touch.tolist())
        by_thread = {}
        for tid, name, handle in log:
            by_thread.setdefault(tid, []).append((name, handle))
        stages_a = by_thread[ta.ident]
        assert [n for n, _ in stages_a] == ["B2", "B3", "B1", "recompute", "compact_v", "B4", "B5+B6", "B7", "reduce", "B8"], stages_a
        assert all(h == (side_handle if n in ("B1", "B7") else s_a.cuda_stream) for n, h in stages_a), stages_a
        assert by_thread[tb.ident] == [("B2", s_b.cuda_stream), ("B3", s_b.cuda_stream)], by_thread[tb.ident]  # B stopped at its dW_o fork
        hold["tid"] = None  # the capture has ended: the same eager execute passes
        _execute(blk, dec.inp, dec.saved, dec.dy, grads_b, ws_b, current_stream=cuda_drv.CUstream(s_b.cuda_stream))
        torch.cuda.synchronize()
        assert touch.tolist() == [34.0, 8.0], touch.tolist()
    finally:  # a failing assertion must leak neither the capture nor the graph into the next test (_err)
        a_inside.set()
        b_done.set()
        ta.join(timeout=180)
        tb.join(timeout=180)
        if "graph" in out:
            out["graph"].reset()


@requires_cuda
def test_fuse_wgrad_overlap_capture_cannot_absorb_an_eager_side_gemm():
    """The guard is atomic with the side GEMM's enqueue: thread B is parked INSIDE its ``issue`` section (its fork done,
    its GEMM not yet launched, the issue lock held) while thread A captures ``execute`` -- A's first fork must wait
    for B's section, so B's GEMM launches eagerly onto a side stream no capture holds and can never become a node of
    A's graph.  Without the lock across the enqueue, A's fork would slip in between B's guard and B's launch and the
    replay would run B's GEMM on B's buffers.  Pinned by the counters: A's replay carries exactly A's ten stages (two
    side-stream kernels, not three), B's eager side GEMM ran once, B's own ``issue`` of its second GEMM is the typed
    error (A's capture, held open past its fork, then owns the side stream), and B's side GEMM precedes A's in the log.
    B waits, OUT of its section, until A has forked before going on, so its second issue meets A's capture
    deterministically (which thread re-acquires the lock first is otherwise a race, and the guard would fire at B's
    join instead -- correct, but a different line).  Streams that share a counter element never run concurrently (host
    synchronizes around the warm-ups)."""
    import threading

    import cuda.bindings.driver as cuda_drv

    dec = _declare_bwd(dict(_COMMON), 1, 256, fuse_wgrad_overlap=True)
    touch = torch.zeros(2, device="cuda")  # [launch-stream stage kernels, side-stream stage kernels]
    log, guard = [], threading.Lock()

    def sink(entry):
        with guard:
            log.append((threading.get_ident(),) + tuple(entry))

    _wire_bwd_recorders(dec, sink, touch=touch)
    blk = dec.blk
    side_handle = blk._side.handle
    tids = {"a": None, "b": None}
    b_in_issue, release_b, a_past_fork, b_done = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    b1_rec, recompute_rec, sdpa_rec = blk._out_proj_wgrad.execute, blk._recompute_qk.execute, blk._sdpa.execute

    def b1_parked(*a, **k):  # the side GEMM: B parks here INSIDE its issue section (lock held, fork done, GEMM not launched)
        if threading.get_ident() == tids["b"]:
            b_in_issue.set()
            release_b.wait(timeout=60)
        b1_rec(*a, **k)
        if threading.get_ident() == tids["a"]:
            a_past_fork.set()

    def recompute_gated(*a, **k):  # B, out of its first issue section: go on only once A has forked (A's B1 ran)
        if threading.get_ident() == tids["b"]:
            a_past_fork.wait(timeout=60)
        recompute_rec(*a, **k)

    def sdpa_held(*a, **k):  # A, inside its capture and past its fork: hold the capture open until B is done
        sdpa_rec(*a, **k)
        if threading.get_ident() == tids["a"]:
            b_done.wait(timeout=60)

    blk._out_proj_wgrad.execute, blk._recompute_qk.execute, blk._sdpa.execute = b1_parked, recompute_gated, sdpa_held
    s_a, s_b = torch.cuda.Stream(), torch.cuda.Stream()
    ws_a = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    ws_b = torch.empty_like(ws_a)
    grads_a, grads_b = _alloc_grads(blk), _alloc_grads(blk)
    torch.cuda.synchronize()  # the touch fill ran on the default stream
    _execute(blk, dec.inp, dec.saved, dec.dy, grads_a, ws_a, current_stream=cuda_drv.CUstream(s_a.cuda_stream))
    torch.cuda.synchronize()  # the two warm-ups both increment touch[0]: never concurrently
    _execute(blk, dec.inp, dec.saved, dec.dy, grads_b, ws_b, current_stream=cuda_drv.CUstream(s_b.cuda_stream))
    torch.cuda.synchronize()
    assert touch.tolist() == [16.0, 4.0], touch.tolist()
    log.clear()
    out = {}

    def thread_b():
        tids["b"] = threading.get_ident()
        try:
            _execute(blk, dec.inp, dec.saved, dec.dy, grads_b, ws_b, current_stream=cuda_drv.CUstream(s_b.cuda_stream))
            out["B"] = None
        except Exception as exc:  # noqa: BLE001 -- reported by the main thread
            out["B"] = _err(exc)
        finally:
            b_in_issue.set()
            b_done.set()

    def thread_a():
        tids["a"] = threading.get_ident()
        graph = torch.cuda.CUDAGraph()
        try:
            b_in_issue.wait(timeout=60)
            with torch.cuda.graph(graph, stream=s_a, capture_error_mode="thread_local"):
                _execute(blk, dec.inp, dec.saved, dec.dy, grads_a, ws_a)
            out["graph"] = graph
        except Exception as exc:  # noqa: BLE001 -- reported by the main thread
            out["A"], out["A_first"] = _err(exc), repr(exc.__context__)
            graph.reset()
        finally:
            a_past_fork.set()

    tb, ta = threading.Thread(target=thread_b), threading.Thread(target=thread_a)
    tb.start()
    ta.start()
    try:
        assert b_in_issue.wait(timeout=60)
        assert not a_past_fork.wait(timeout=1.0), "A's capture forked onto the side stream while B's issue section was open"
        release_b.set()  # B launches its GEMM eagerly, records its join and leaves the section; only then can A fork
        ta.join(timeout=180)
        tb.join(timeout=180)
        assert "A" not in out, ("the capture failed", out.get("A"), "first error in its body:", out.get("A_first"))
        assert str(out.get("B")).startswith("RuntimeError: ") and "(at the fork of dW_qkvg)" in str(out.get("B")), out.get("B")
        torch.cuda.synchronize()
        # B ran eagerly up to its second issue: 6 launch-stream stages (B2, B3, recompute, compact_v, B4, B5+B6) and ONE side GEMM
        assert touch.tolist() == [22.0, 5.0], ("the capture launched work, or B's eager GEMM did not run", touch.tolist())
        out["graph"].replay()
        torch.cuda.synchronize()
        assert touch.tolist() == [30.0, 7.0], ("the graph does not carry exactly A's ten stages", touch.tolist())
        b1 = [(tid, name) for tid, name, _h in log if name == "B1"]
        assert b1 == [(tb.ident, "B1"), (ta.ident, "B1")], b1  # B's eager side GEMM before A's captured one: A's fork waited
    finally:  # a failing assertion must leak neither a parked thread, the capture nor the graph into the next test (_err)
        b_in_issue.set()
        release_b.set()  # never leave B parked inside its section, nor A held inside its capture
        b_done.set()
        ta.join(timeout=180)
        tb.join(timeout=180)
        if "graph" in out:
            out["graph"].reset()


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


@requires_cuda
def test_a_quantized_record_handed_through_with_its_e4m3_h_is_a_typed_decline():
    """The per-tensor FP8 / MXFP8 training forward keeps ``saved.h`` as the caller's e4m3 codes; this bf16 backward consumes
    such a record given the DEQUANTIZED bf16 ``h`` (``dataclasses.replace(saved, h=...)``, the accept test
    ``test_gradients_over_a_quantized_record_match_the_record_seeded_fp64_oracle``).  Handed the record as written, it raises
    a ``ValueError`` naming the record and that contract -- not the generic dtype mismatch -- at declaration and at execute,
    before any launch, on any CUDA device; the dequantized-h record passes the same check."""
    from cudnn.gated_attention_block.api_bwd import _check_saved_record

    res = _declare_bwd(dict(_COMMON), 1, 256)
    blk, saved = res.blk, res.saved
    as_written = dataclasses.replace(saved, h=saved.h.to(torch.float8_e4m3fn))
    blk._samples["saved"] = as_written
    with pytest.raises(ValueError, match="e4m3 codes") as ei:
        blk.check_support()
    assert "dataclasses.replace(saved, h=h_dequantized)" in str(ei.value) and "DEQUANTIZED torch.bfloat16 h" in str(ei.value)
    with pytest.raises(ValueError, match="e4m3 codes"):
        _check_saved_record(as_written, blk.geom, 1, 256, torch.bfloat16, saved.h.device, at="execute")
    _check_saved_record(saved, blk.geom, 1, 256, torch.bfloat16, saved.h.device, at="execute")


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


# ---------------------------------------------------------------------------
# A QUANTIZED record through the bf16 backward (Rubin)
# ---------------------------------------------------------------------------


class _AttentionFromRecord(torch.autograd.Function):
    """The SDPA stage as the block backward sees it: the forward VALUE is the record's pre-gate ``O`` and the backward is the
    exact attention backward over the record's LSE -- ``delta = rowsum(dO * O)``, ``P = exp(S - LSE)`` (masked), ``dV = P^T dO``,
    ``dP = dO V^T``, ``dS = P (dP - delta)``, ``dQ = dS K scale``, ``dK = dS^T Q scale`` -- which is what ``api_bwd`` computes by
    construction (B3 takes ``saved.o``; B4 takes ``saved.o`` / ``saved.lse`` and the Q / K recomputed from the slab).  Operands
    are BHSD fp64 with K / V already GQA-broadcast (the group sum comes back through ``repeat_interleave``'s autograd)."""

    @staticmethod
    def forward(ctx, q, k, v, o_rec, lse_rec, scale, causal):
        """Save the operands and the record's ``O`` / ``LSE`` for the backward; the stage's VALUE is the record's pre-gate ``O``
        (a clone), not a recomputed attention."""
        ctx.save_for_backward(q, k, v, o_rec, lse_rec)
        ctx.scale, ctx.causal = float(scale), bool(causal)
        return o_rec.clone()

    @staticmethod
    def backward(ctx, do):
        """The exact attention backward over the record's LSE (the class docstring's chain, causal-masked when the geometry is):
        ``dQ``, ``dK``, ``dV``, and ``None`` for ``o_rec`` / ``lse_rec`` / ``scale`` / ``causal``."""
        q, k, v, o, lse = ctx.saved_tensors
        s_ = torch.matmul(q, k.transpose(-1, -2)) * ctx.scale
        if ctx.causal:
            s_ = s_.masked_fill(~torch.tril(torch.ones(s_.shape[-2:], dtype=torch.bool, device=s_.device)), float("-inf"))
        p = torch.exp(s_ - lse[..., None])  # masked cells: exp(-inf) == 0; no dead rows here (dense / causal, no padding)
        delta = (do * o).sum(-1, keepdim=True)
        dv = torch.matmul(p.transpose(-1, -2), do)
        dp = torch.matmul(do, v.transpose(-1, -2))
        ds = p * (dp - delta)
        dq = torch.matmul(ds, k) * ctx.scale
        dk = torch.matmul(ds.transpose(-1, -2), q) * ctx.scale
        return dq, dk, dv, None, None, None, None


def _fp64_oracle_from_record(inp: dict, geom_kw: dict, dy: torch.Tensor, o_rec: torch.Tensor, lse_rec: torch.Tensor) -> dict:
    """``_fp64_oracle``'s twin for a record whose SDPA stage is SEEDED: fp64 autograd through the block's chain on fp64 copies
    of the (bf16) inputs, with the attention replaced by :class:`_AttentionFromRecord` over the record's ``O`` / ``LSE``.  Same
    outputs (the five gradients, the post-norm ``dq`` / ``dk`` and the ``dW_norm`` noise masses)."""
    g64 = RefGeometry(**geom_kw)
    leaf = lambda x: None if x is None else x.detach().double().requires_grad_(True)  # noqa: E731
    h, w_qkvg, w_q, w_k, w_o = (leaf(inp[k]) for k in ("h", "w_qkvg", "w_q_norm", "w_k_norm", "w_o"))
    cos, sin = inp["cos"].double(), inp["sin"].double()
    b, s, dm = h.shape
    hq, hkv, d = g64.h_q, g64.h_kv, g64.d_head
    o_q, o_g, o_k, o_v = g64.offsets
    proj = h.reshape(b * s, dm) @ w_qkvg.t()  # fp64, unrounded (the oracle's acc_dtype=float64 form)
    q_pre = proj[:, o_q : o_q + hq * d].reshape(b, s, hq, d)
    gate = proj[:, o_g : o_g + hq * d].reshape(b, s, hq, d)
    k_pre = proj[:, o_k : o_k + hkv * d].reshape(b, s, hkv, d)
    v = proj[:, o_v : o_v + hkv * d].reshape(b, s, hkv, d)
    q, rstd_q = qk_norm_rope_reference(q_pre, w_q, cos, sin, g64.rope_dim, g64.qk_norm_eps, qk_norm=g64.qk_norm, acc_dtype=torch.float64)
    k, rstd_k = qk_norm_rope_reference(k_pre, w_k, cos, sin, g64.rope_dim, g64.qk_norm_eps, qk_norm=g64.qk_norm, acc_dtype=torch.float64)
    rep = hq // hkv
    qb = q.transpose(1, 2)
    kb = k.transpose(1, 2).repeat_interleave(rep, 1)
    vb = v.transpose(1, 2).repeat_interleave(rep, 1)
    o = _AttentionFromRecord.apply(qb, kb, vb, o_rec.detach().double().transpose(1, 2), lse_rec.detach().double(), g64.scale, g64.is_causal).transpose(1, 2)
    out = (o * torch.sigmoid(gate)).reshape(b, s, hq * d) @ w_o.t()
    wanted = [h, w_qkvg, w_o] + ([w_q, w_k] if g64.qk_norm else []) + [q, k]
    grads = list(torch.autograd.grad(out, wanted, dy.double().reshape(b, s, dm)))
    res = dict(dh=grads.pop(0), dw_qkvg=grads.pop(0), dw_o=grads.pop(0))
    res.update(dw_q_norm=grads.pop(0), dw_k_norm=grads.pop(0)) if g64.qk_norm else res.update(dw_q_norm=None, dw_k_norm=None)
    dq_post, dk_post = grads.pop(0), grads.pop(0)
    if g64.qk_norm:
        res["dw_q_norm_mass"] = _dw_norm_noise_mass(dq_post, q_pre, rstd_q, cos, sin, g64.rope_dim)
        res["dw_k_norm_mass"] = _dw_norm_noise_mass(dk_post, k_pre, rstd_k, cos, sin, g64.rope_dim)
    return res


@requires_rubin
@pytest.mark.parametrize("family", ["fp8", "mxfp8"])
@_CAUSAL
def test_gradients_over_a_quantized_record_match_the_record_seeded_fp64_oracle(family, causal):
    """The bf16 backward CONSUMES the quantized training forward's record: the per-tensor FP8 / MXFP8 forward writes the bf16
    record (slab with PRE-norm Q/K bands, pre-gate ``O``, exact LSE, ``rstd``) and this backward is handed it with the
    dequantized bf16 ``h`` (``dataclasses.replace(saved, h=...)``) and weights.  ``dh``, ``dW_qkvg``, ``dW_o`` and the fp32
    ``dW_norm`` are held to the module's bf16 bounds against the record-seeded fp64 oracle (the exact function of the record;
    the quantized forward's ``O`` carries the kernels' e4m3 P that no oracle models, so the attention stage is seeded with the
    record's own ``O`` / ``LSE`` and everything else -- the bands, the recompute, ``rstd``, the eight stages -- is under test).
    The cosine against the PLAIN fp64 oracle (the unquantized chain on the dequantized inputs) is printed for the record."""
    geom_kw = {**_COMMON, "qk_norm": True, "is_causal": causal}
    b, s = 2, 512
    r = _run_training_quant(geom_kw, b, s, family)
    deq = _dequantized_bf16_inputs(r.inp, r.spec, family)
    saved = dataclasses.replace(r.saved, h=deq["h"])  # everything else is the quantized forward's record, as written
    assert saved.h.dtype == torch.bfloat16 and saved.proj_slab is r.saved.proj_slab and saved.o is r.saved.o and saved.lse is r.saved.lse
    dy = _make_dy(r.out)
    blk = GatedAttentionBlockBwd(dy, saved, deq["w_qkvg"], deq["w_q_norm"], deq["w_k_norm"], deq["cos"], deq["sin"], deq["w_o"], r.geom)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute(blk, deq, saved, dy, grads, ws)
    torch.cuda.synchronize()
    res = SimpleNamespace(grads=grads, oracle=_fp64_oracle_from_record(deq, geom_kw, dy, r.saved.o, r.saved.lse))
    worst = _check_all_grads(res)
    plain = _fp64_oracle(deq, geom_kw, dy)
    cos_plain = {nm: _cos(grads[nm], plain[nm]) for nm in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm")}
    print(f"{family} {'causal' if causal else 'dense'} record: worst cells {worst}; cos vs the PLAIN fp64 oracle (reported) {cos_plain}")


# ---------------------------------------------------------------------------
# The quantized (per-tensor fp8) backward -- the API's host cells (any CUDA device, or no device for the pure carve)
# ---------------------------------------------------------------------------

_E4M3 = torch.float8_e4m3fn
# A plausible per-tensor spec (the forward's static scales): the declaration reads its dtype and its positivity only.
_QSPEC = QuantSpec(descale_h=0.25, descale_w_qkvg=0.5, descale_w_o=0.125, scale_q=2.0, scale_k=4.0, scale_v=8.0, scale_o=16.0)
# The quantized backward's stage list, in launch order (the module docstring's table): the host pin of the wiring.
_FP8_STAGES = [
    "_QuantPrologue",
    "_QuantizeGrad",
    "_OutProjDgrad",
    "_SigmoidGateBwd",
    "_QuantizeGrad",
    "_OutProjWgrad",
    "_SdpaBwdFp8",
    "_QkNormRopeBwd",
    "_QuantEpilogue",
    "_QkvGateWgrad",
    "_QkvGateDgrad",
]
_BF16_STAGES = [
    "_OutProjDgrad",
    "_SigmoidGateBwd",
    "_OutProjWgrad",
    "_QkNormRope",
    "_VCompaction",
    "_SdpaBwd",
    "_QkNormRopeBwd",
    "_QkvGateWgrad",
    "_QkvGateDgrad",
]


def _declare_bwd_fp8(geom_kw, batch, seq_len, *, spec=_QSPEC, h_dtype=_E4M3, w_dtype=_E4M3, dy_dtype=torch.bfloat16, saved_replace=None, **bwd_kw):
    """A DECLARED (not compiled) per-tensor fp8 backward over a bf16 forward's record whose ``h`` and weights are CAST to the
    code dtype -- the declaration reads dtypes / shapes / None-ness only, never a code's value -- plus the QuantSpec; CUDA tensors,
    no launch, any CUDA device.  ``saved_replace`` edits the record before the declaration (a packed record for the THD cell)."""
    fwd, inp, out = _declare(geom_kw, batch, seq_len, save_mode="proj_slab", dtype=dy_dtype)
    saved = _alloc_saved(fwd.geom, inp, batch, seq_len, save_mode="proj_slab")
    saved = dataclasses.replace(saved, h=saved.h.to(h_dtype), **(saved_replace or {}))
    inp = {**inp, "h": saved.h, "w_qkvg": inp["w_qkvg"].to(w_dtype), "w_o": inp["w_o"].to(w_dtype)}
    dy = _make_dy(out)
    blk = GatedAttentionBlockBwd(dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], fwd.geom, quant=spec, **bwd_kw)
    return SimpleNamespace(blk=blk, fwd=fwd, inp=inp, saved=saved, dy=dy, out=out, geom=fwd.geom, geom_kw=geom_kw, batch=batch, seq_len=seq_len)


def _passes_the_block_level_checks(blk, *, attr: str) -> None:
    """``check_support`` on a DECLARED quantized block: every block-level decline passes (none names ``attr``); what remains is
    the Rubin gate on this host (a ``NotImplementedError`` naming Rubin) or, on Rubin, the stages' own contracts."""
    try:
        blk.check_support()
    except NotImplementedError as exc:
        assert attr not in str(exc), str(exc)
    except ValueError as exc:
        assert attr not in str(exc), str(exc)


@requires_cuda
def test_fp8_declaration_declines_are_typed():
    """Every typed decline of the quantized backward's DECLARATION, on any CUDA device, before the Rubin gate and before any
    stage is asked -- and the ONE place its message texts are pinned (every other test matches the attribute name only).
    At construction: a ``quant`` of a wrong type (``TypeError``), an ``MxQuantSpec`` (a typed ``NotImplementedError``: the
    MXFP8 backward is a follow-up), e5m2 codes (``QuantSpec.validate``), ``grad_scaling`` outside its vocabulary or given
    without ``quant``.  At ``check_support``: an fp16 ``sample_dy`` under ``quant`` (the quantized backward is bf16); the
    record / weight dtype gates BOTH ways (a bf16 ``saved.h`` with a spec, e4m3 codes without one -- the Q0 message extended
    with the ``quant=QuantSpec`` declaration --, bf16 weights with a spec, e4m3 weights without one); ``thd=True`` with
    ``quant`` (names BOTH attributes and never the row's flag, so the caller is not told to drop the delta the block
    requires).  There is NO ``B*S % 16`` decline: the weight-gradient GEMMs are MN-major (no K-contiguous operand), so S = 1000
    at B = 1 passes the block-level checks with its weight gradients, with one of them, without them, and at B = 2 alike."""
    b, s = 1, 256
    # -- construction --
    with pytest.raises(TypeError, match="quant"):
        _declare_bwd_fp8(dict(_COMMON), b, s, spec=object())
    # an MxQuantSpec is the MXFP8 backward's own declaration now (test_mxfp8_declaration_declines_are_typed pins its declines)
    assert isinstance(_declare_bwd_fp8(dict(_COMMON), b, s, spec=MxQuantSpec(descale_w_o=0.125, scale_o=16.0)).blk.quant, MxQuantSpec)
    with pytest.raises(NotImplementedError, match="QuantSpec"):
        _declare_bwd_fp8(dict(_COMMON), b, s, spec=dataclasses.replace(_QSPEC, dtype=torch.float8_e5m2))
    with pytest.raises(ValueError, match="grad_scaling"):
        _declare_bwd_fp8(dict(_COMMON), b, s, grad_scaling="static")
    with pytest.raises(ValueError, match="grad_scaling"):
        _declare_bwd(dict(_COMMON), b, s, grad_scaling="delayed")  # the attribute belongs to the quantized backward
    assert _declare_bwd_fp8(dict(_COMMON), b, s, grad_scaling="delayed").blk.grad_scaling == "delayed"
    # -- the activation dtype under quant --
    r = _declare_bwd_fp8(dict(_COMMON), b, s, dy_dtype=torch.float16)
    with pytest.raises(ValueError, match="quant=QuantSpec") as ei:
        r.blk.check_support()
    assert "bfloat16" in str(ei.value) and "float16" in str(ei.value)
    # -- the record's h, both directions --
    r = _declare_bwd_fp8(dict(_COMMON), b, s, h_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="saved.h") as ei:
        r.blk.check_support()
    assert "e4m3 codes" in str(ei.value) and "quant=QuantSpec" in str(ei.value)
    r = _declare_bwd(dict(_COMMON), b, s)
    r.blk._samples["saved"] = dataclasses.replace(r.saved, h=r.saved.h.to(_E4M3))
    with pytest.raises(ValueError, match="e4m3 codes") as ei:
        r.blk.check_support()
    msg = str(ei.value)  # the Q0 contract kept verbatim, extended with the native declaration
    assert "dataclasses.replace(saved, h=h_dequantized)" in msg and "DEQUANTIZED torch.bfloat16 h" in msg and "quant=QuantSpec" in msg
    # -- the weights, both directions --
    r = _declare_bwd_fp8(dict(_COMMON), b, s, w_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="w_qkvg") as ei:
        r.blk.check_support()
    assert "quant=QuantSpec" in str(ei.value) and "descale_w" in str(ei.value)
    r = _declare_bwd(dict(_COMMON), b, s)
    r.blk._samples["w_o"] = r.inp["w_o"].to(_E4M3)
    with pytest.raises(ValueError, match="w_o") as ei:
        r.blk.check_support()
    assert "without quant" in str(ei.value) and "DEQUANTIZED" in str(ei.value)
    # -- thd + quant: both attributes named, the row's flag never (the message must not tell the caller to drop the delta) --
    lens = torch.tensor([128, 128], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError) as ei:  # at CONSTRUCTION (before any stage is built), whatever the record carries
        _declare_bwd_fp8(dict(_COMMON), 1, 256, thd=True, num_sequences=2, max_seq_len=128, saved_replace=dict(seq_lens=lens, seq_lens_form="lengths"))
    msg = str(ei.value)
    assert "thd=True" in msg and "quant=QuantSpec" in msg and "dense-only" in msg, msg
    assert "external_delta" not in msg, msg
    z = torch.empty(0, device="cuda")  # a placeholder record without a proj_slab gets the same answer, not the gate-copy decline
    placeholder = SavedForBackward(
        h=torch.empty(256, _COMMON["d_model"], dtype=_E4M3, device="cuda"), gate=z, o=z, lse=z, rstd_q=z, rstd_k=z, seq_lens=lens, seq_lens_form="lengths"
    )
    inp = _declare_bwd(dict(_COMMON), 1, 256).inp
    with pytest.raises(ValueError, match="thd=True with quant=QuantSpec"):
        GatedAttentionBlockBwd(
            torch.empty(256, _COMMON["d_model"], dtype=torch.bfloat16, device="cuda"),
            placeholder,
            inp["w_qkvg"].to(_E4M3),
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"].to(_E4M3),
            GatedAttentionBlockGeometry(**_COMMON),
            quant=_QSPEC,
            thd=True,
            num_sequences=2,
            max_seq_len=256,
        )
    # -- no B*S % 16 rule: S = 1000 at B = 1 passes the block-level checks WITH its weight gradients (T = 1000 is a ragged K the
    #    MN-major wgrads zero-fill), with one of them, without them, and at B = 2 --
    for kw, (bb, ss) in (({}, (1, 1000)), (dict(need_dw_qkvg=False), (1, 1000)), (dict(need_dw_o=False, need_dw_qkvg=False), (1, 1000)), ({}, (2, 1000))):
        r = _declare_bwd_fp8(dict(_COMMON), bb, ss, **kw)
        _passes_the_block_level_checks(r.blk, attr="B*S")
        _passes_the_block_level_checks(r.blk, attr="% 16")
    # -- every bf16 decline is unchanged (one spot check: padding under quant is the dense block's padding decline) --
    r = _declare_bwd_fp8(dict(_COMMON), b, s, seq_lens_present=True)
    with pytest.raises(NotImplementedError, match="sdpa_bwd_sm107"):
        r.blk.check_support()


@requires_cuda
def test_fp8_declaration_wires_the_quant_stage_list():
    """``quant=QuantSpec`` builds the quantized backward's stage list in its launch order -- the fused PROLOGUE (scalar init,
    the dY amax as per-CTA partials, the Q / K rebuild with its e4m3 epilogue, v8: no bf16 rebuild, no V compaction, no
    standalone quantizers), the dY quantize (reduces the partials; two alpha products), the e4m3 out_proj dgrad, the gate
    backward's fp8 arm (``og8`` iff ``need_dw_o``, the dO and dG amax folds as per-CTA partials, the delta ALWAYS), the dO quantize
    (no amax pass of its own: it reduces B3's dO partials), the e4m3 out_proj wgrad, the fp8 SDPA
    stage, the norm backward (the bands' amax as per-CTA partials), the fused EPILOGUE (the dW_norm reduce + the dqkvg quantize
    over the dG and band partials, two alpha products), the two e4m3 qkv_gate GEMMs --
    every e4m3 GEMM stage declared ``alpha=True`` with a bf16 output and the EXPLICIT 64-byte MMA K (never derived from the
    dtype); ``"delayed"`` flips the quantizers to the caller's scale and keeps the amax folds; the bf16 declaration is
    untouched."""
    from cudnn.gated_attention_block.api_bwd import _FP8_GEMM_MMA_TILE_K_BYTES, _QuantEpilogue, _QuantizeGrad, _QuantPrologue, _SdpaBwdFp8

    on = _declare_bwd_fp8(dict(_COMMON), 1, 256).blk
    assert [type(st).__name__ for st in on._stages] == _FP8_STAGES
    assert on.quant is _QSPEC and on.grad_scaling == "current" and on.w_dtype == _E4M3 and on.act_dtype == torch.bfloat16
    assert on._compact_v is None and on._recompute_qk is None and isinstance(on._sdpa, _SdpaBwdFp8) and on._quant_vals is None  # VALUES at compile()
    # the prologue's init job zeroes every slot and stores the plan-time constants (the tail of the slot tuple) from its kernel arguments
    assert (on._prologue.n_slots, on._prologue.const_slot0, on._prologue.n_consts) == (len(QUANT_SCALAR_SLOTS), 15, len(QUANT_CONST_SLOTS))
    assert (
        not hasattr(on, "_quant_dev") and not hasattr(on, "_quant_consts") and not hasattr(on, "_init_scalars")
    ), "no compile-time device constants, no standalone init stage"
    for st in on._stages:
        if isinstance(st, _GemmStage):
            assert (st.dtype, st.alpha, st.out_dtype, st.mma_tile_k_bytes) == (_E4M3, True, torch.bfloat16, _FP8_GEMM_MMA_TILE_K_BYTES), st.label
    assert _FP8_GEMM_MMA_TILE_K_BYTES == 64
    gb = on._gate_bwd
    assert (gb.want_og, gb.og_fp8, gb.want_amax_do, gb.want_amax_dg, gb.want_delta) == (True, True, True, True, True)
    d = _COMMON["d_head"]
    # neither quantize owns an amax pass: the dY cast reduces the prologue's partials, the dO cast B3's; neither is persistent
    # (the one-row-group-per-block grid of the standalone quantize: the measured faster form for a <= SMs x 8 partials reduce)
    qdy, qdo = on._quant_dy, on._quant_do
    assert (qdy.heads, qdy.n_alpha, qdy.own_amax, qdy.amax_src, qdy.scale_src, qdy.persistent) == (_COMMON["d_model"] // d, 2, False, "partials", "amax", False)
    assert (qdo.heads, qdo.n_alpha, qdo.own_amax, qdo.amax_src, qdo.scale_src, qdo.persistent) == (_COMMON["h_q"], 0, False, "partials", "amax", False)
    # the fused launches: the prologue carries the scalar block's width and the forward's own rebuild stage (TMA-tiled here); the
    # epilogue the dqkvg quantize (two alpha products) and the reduce; the norm backward folds the bands' amax
    pro, epi = on._prologue, on._epilogue
    assert isinstance(pro, _QuantPrologue) and pro.n_slots == len(QUANT_SCALAR_SLOTS) and pro._rebuild.want_rstd is False
    assert pro._rebuild.resolve_tile_rows() > 0  # the geometry tiles for the TMA rebuild (the decline otherwise is pinned in the fp8 module)
    assert isinstance(epi, _QuantEpilogue) and (epi.want_dw, epi.n_alpha, epi.scale_src) == (True, 2, "amax")
    assert on._norm_bwd.want_amax is True
    lean = _declare_bwd_fp8(dict(_COMMON), 1, 256, need_dw_o=False).blk
    assert (lean._gate_bwd.want_og, lean._gate_bwd.og_fp8, lean._gate_bwd.want_delta) == (False, False, True)
    assert [type(st).__name__ for st in lean._stages] == [n for n in _FP8_STAGES if n != "_OutProjWgrad"]
    delayed = _declare_bwd_fp8(dict(_COMMON), 1, 256, grad_scaling="delayed").blk
    assert all(st.scale_src == "given" for st in delayed._stages if isinstance(st, (_QuantizeGrad, _QuantEpilogue)))
    # the partials come from the producers under both recipes; the delayed casts reduce and publish them the same way
    assert all((st.own_amax, st.amax_src, st.persistent) == (False, "partials", False) for st in (delayed._quant_dy, delayed._quant_do))
    off = _declare_bwd(dict(_COMMON), 1, 256).blk
    assert off.quant is None and off.grad_scaling == "current" and off.w_dtype == torch.bfloat16 and off._prologue is None and off._epilogue is None
    assert off._quant_dy is None and off._quant_do is None and off._recompute_qk is not None and off._compact_v is not None
    assert [type(st).__name__ for st in off._stages] == _BF16_STAGES


@requires_cuda
def test_fuse_gate_bwd_has_no_effect_under_quant():
    """Under ``quant`` the gate backward's delta is MANDATORY (it is the fp8 SDPA row's external delta), so ``fuse_gate_bwd``
    has no second arm: both declarations resolve the same stage list, the gate stage wants the delta under both, and the
    SDPA stage is built with the external delta under both.  Accepted with either value (a knob is performance-only: the same
    function under any value); the Rubin cells pin the gradients bitwise."""
    off = _declare_bwd_fp8(dict(_COMMON), 1, 256).blk
    on = _declare_bwd_fp8(dict(_COMMON), 1, 256, fuse_gate_bwd=True).blk
    assert (off.fuse_gate_bwd, on.fuse_gate_bwd) == (False, True)
    assert [type(st).__name__ for st in on._stages] == [type(st).__name__ for st in off._stages] == _FP8_STAGES
    assert off._gate_bwd.want_delta is True and on._gate_bwd.want_delta is True
    for blk in (off, on):
        # The stage constructs the fp8 adapter on any device (its delta shape is shape arithmetic), so this pin RUNS here:
        # a construction-time error would fail the external-delta pin, never skip it.
        assert blk._sdpa.delta_shape == (1, _COMMON["h_q"], 256) and blk._sdpa._impl.external_delta is True


def test_workspace_carve_under_quant_is_the_declared_composition():
    """``_plan_bwd_workspace(quant=QuantSpec)`` on any device -- the pure-carve twin of
    ``test_workspace_carve_is_the_declared_composition``: every bf16 region keeps its place and size except the four the
    quantized backward does not write (``o_gated`` -> -1: B3's third output is the e4m3 ``og8``; ``recompute`` / ``recompute_k``
    -> -1: the fused prologue writes the e4m3 ``q8`` / ``k8`` straight out of its registers; ``recompute_v`` -> -1: ``v8`` IS V's
    compaction), ``delta`` is MANDATORY, and the e4m3 regions plus the 256-B fp32 scalar block are appended
    AFTER every bf16 region in the documented order, each padded to the carve alignment; ``og8`` follows ``need_dw_o``
    (or an explicit ``need_og8``); the ``quant=None`` layout is byte-identical to before (every appended field -1)."""
    g = GatedAttentionBlockGeometry(**_COMMON)
    b, s, e = 2, 256, 2
    t, d, n = b * s, g.d_head, g.n_qkvg
    al = lambda x: -(-x // _WS_ALIGN) * _WS_ALIGN  # noqa: E731
    common = dict(sdpa_bwd_bytes=1000, gemm_scratch_bytes=4096, n_ctas_q=7, n_ctas_k=3, delta_shape=(b, g.h_q, 384), side_gemm_scratch_bytes=512)
    lay16 = _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_o=True, dw_norms=True), **common)
    lay8 = _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_o=True, dw_norms=True), quant=_QSPEC, **common)
    # quant=None: every appended field is -1 (the bf16 layout is untouched)
    for f in ("dy8", "do8", "og8", "q8", "k8", "v8", "dqkvg8", "quant_scalars"):
        assert getattr(lay16, f) == -1, f
    # the bf16 regions, in the documented order, with o_gated, recompute, recompute_k and recompute_v dropped under quant
    sizes = dict(
        do_gated=t * g.h_q * d * e,
        dqkvg=t * n * e,
        dq=t * g.h_q * d * e,
        dk=t * g.h_kv * d * e,
        dv=t * g.h_kv * d * e,
        dw_partials_q=7 * d * 4,
        dw_partials_k=3 * d * 4,
        sdpa_bwd_ws=1000,
        gemm_scratch=4096,
        delta=b * g.h_q * 384 * 4,
        gemm_scratch_side=512,
        # the quantized backward's regions, 1 B/elem, then the scalar block
        dy8=t * g.d_model,
        do8=t * g.h_q * d,
        og8=t * g.h_q * d,
        q8=t * g.h_q * d,
        k8=t * g.h_kv * d,
        v8=t * g.h_kv * d,
        dqkvg8=t * n,
        quant_scalars=QUANT_SCALARS_BYTES,
    )
    order = [
        "do_gated",
        "dqkvg",
        "dq",
        "dk",
        "dv",
        "dw_partials_q",
        "dw_partials_k",
        "sdpa_bwd_ws",
        "gemm_scratch",
        "delta",
        "gemm_scratch_side",
        "dy8",
        "do8",
        "og8",
        "q8",
        "k8",
        "v8",
        "dqkvg8",
        "quant_scalars",
    ]
    off = 0
    for name in order:
        assert getattr(lay8, name) == off, (name, getattr(lay8, name), off)
        off += al(sizes[name])
    assert lay8.total_bytes == off and lay8.o_gated == -1 and lay8.recompute == -1 and lay8.recompute_k == -1 and lay8.recompute_v == -1
    assert lay8.do == -1 and lay8.base_align == _WS_ALIGN
    assert lay8.delta >= 0 and lay8.delta_shape == (b, g.h_q, 384) and lay8.quant_scalars % _WS_ALIGN == 0
    assert len(QUANT_SCALAR_SLOTS) * QUANT_SCALAR_STRIDE <= QUANT_SCALARS_BYTES and len(QUANT_SCALAR_SLOTS) == 15 + len(QUANT_CONST_SLOTS) == 29
    # the shared prefix up to dqkvg is byte-identical; past it the four dropped bf16 regions (o_gated, recompute, recompute_k,
    # recompute_v) shift every later bf16 region by their padded sizes
    assert (lay8.do_gated, lay8.dqkvg) == (lay16.do_gated, lay16.dqkvg)
    assert lay8.dq == lay16.dq - 2 * al(t * g.h_q * d * e) - 2 * al(t * g.h_kv * d * e)
    # the per-token delta of the quantized carve at this geometry: +(dy8 + do8 + og8 + q8 + k8 + v8 + dqkvg8) - (o_gated + recompute +
    # recompute_k + recompute_v) bytes (the fused prologue writes q8 / k8 straight out of its registers: no bf16 rebuild buffers)
    added = t * (g.d_model + 3 * g.h_q * d + 2 * g.h_kv * d + n) - t * (2 * g.h_q * d * e + 2 * g.h_kv * d * e)
    assert lay8.total_bytes - lay16.total_bytes == added + al(QUANT_SCALARS_BYTES), (lay8.total_bytes - lay16.total_bytes, added)
    # the dY amax partials (appended, DEFAULTED): the default carves none -- the layout above is the pure carve --, a positive
    # count carves fp32 [n] LAST (after the scalar block), a count without quant or a negative / bool one is typed
    assert (lay8.amax_partials, lay8.amax_partials_n) == (-1, 0) and (lay16.amax_partials, lay16.amax_partials_n) == (-1, 0)
    cap = 204 * 8  # SMs x 8, what compile() passes from the prologue recipe on a 204-SM part
    with_p = _plan_bwd_workspace(
        g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_o=True, dw_norms=True), quant=_QSPEC, amax_partials_n=cap, **common
    )
    assert (with_p.amax_partials, with_p.amax_partials_n) == (lay8.total_bytes, cap) and with_p.total_bytes == lay8.total_bytes + al(cap * 4)
    assert with_p.quant_scalars == lay8.quant_scalars and with_p.dqkvg8 == lay8.dqkvg8  # everything before it is untouched
    with pytest.raises(ValueError, match="amax_partials_n"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), amax_partials_n=cap, **common)
    for bad in (-1, True):
        with pytest.raises(ValueError, match="amax_partials_n"):
            _plan_bwd_workspace(
                g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), quant=_QSPEC, amax_partials_n=bad, **common
            )
    # the producers' partials (appended, DEFAULTED): the gate backward's dO and dG arrays (its cap, one region each) then the norm
    # backward's band partials (its grid), in that order AFTER the dY partials; the defaults carve none; typed without quant / bad
    for lay in (lay8, lay16, with_p):
        assert (lay.amax_partials_do, lay.amax_partials_dg, lay.gate_partials_n, lay.amax_partials_bands, lay.band_partials_n) == (-1, -1, 0, -1, 0)
    with_all = _plan_bwd_workspace(
        g,
        b,
        s,
        torch.bfloat16,
        RecomputePolicy.RECOMPUTE_QK_PRE,
        need=dict(dw_o=True, dw_norms=True),
        quant=_QSPEC,
        amax_partials_n=cap,
        gate_partials_n=cap,
        band_partials_n=4896,
        **common,
    )
    base_off = with_p.total_bytes
    assert (with_all.amax_partials, with_all.amax_partials_n) == (with_p.amax_partials, cap)
    assert (with_all.amax_partials_do, with_all.amax_partials_dg, with_all.gate_partials_n) == (base_off, base_off + al(cap * 4), cap)
    assert (with_all.amax_partials_bands, with_all.band_partials_n) == (base_off + 2 * al(cap * 4), 4896)
    assert with_all.total_bytes == base_off + 2 * al(cap * 4) + al(4896 * 4)
    only_bands = _plan_bwd_workspace(
        g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_o=True, dw_norms=True), quant=_QSPEC, band_partials_n=12, **common
    )
    assert (only_bands.amax_partials_do, only_bands.amax_partials_dg, only_bands.gate_partials_n) == (-1, -1, 0)
    assert (only_bands.amax_partials_bands, only_bands.band_partials_n, only_bands.total_bytes) == (lay8.total_bytes, 12, lay8.total_bytes + al(48))
    with pytest.raises(ValueError, match="gate_partials_n"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), gate_partials_n=cap, **common)
    for bad in (-1, True):
        with pytest.raises(ValueError, match="band_partials_n"):
            _plan_bwd_workspace(
                g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), quant=_QSPEC, band_partials_n=bad, **common
            )
    # og8 follows need_dw_o (-1 without the wgrad), and need_og8 overrides it only in the direction B1 can live with
    lean = _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_o=False, dw_norms=False), quant=_QSPEC, **common)
    assert lean.og8 == -1 and lean.o_gated == -1 and lean.q8 == lean.do8 + al(t * g.h_q * d) and lean.dw_partials_q == -1
    forced = _plan_bwd_workspace(
        g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_o=False, dw_norms=False), quant=_QSPEC, need_og8=True, **common
    )
    assert forced.og8 == lean.do8 + al(t * g.h_q * d) and forced.total_bytes == lean.total_bytes + al(t * g.h_q * d)
    with pytest.raises(ValueError, match="need_og8"):
        _plan_bwd_workspace(
            g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_o=True, dw_norms=False), quant=_QSPEC, need_og8=False, **common
        )
    with pytest.raises(ValueError, match="need_og8"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), need_og8=True, **common)
    # delta is mandatory under quant (the gate backward's delta is the fp8 row's external delta); a wrong quant type is typed
    no_delta = {k: v for k, v in common.items() if k != "delta_shape"}
    with pytest.raises(ValueError, match="delta_shape"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), quant=_QSPEC, **no_delta)
    with pytest.raises(ValueError, match="QuantSpec"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), quant=object(), **common)


def test_workspace_carve_pins_the_slot_stride_to_the_fp32_element_size(monkeypatch):
    """The scalar block's slot stride is coupled to the fp32 element size in three places (``_scalar()``'s offsets, the
    contiguous ``[n_slots]`` view the init launch zeroes, the init kernel's 4-byte store pitch): the carve pins the equality,
    so a stride moved on its own (16 B, say) raises at declaration naming the constant -- instead of readers sitting on bytes
    the init never zeroed (an amax slot that never grows).  The pin fires under ``quant`` only; the bf16 carve never reads it."""
    import cudnn.gated_attention_block.api_bwd as api_bwd_mod

    g = GatedAttentionBlockGeometry(**_COMMON)
    common = dict(sdpa_bwd_bytes=1000, gemm_scratch_bytes=4096, n_ctas_q=7, n_ctas_k=3, delta_shape=(1, g.h_q, 256))
    assert QUANT_SCALAR_STRIDE == torch.empty((), dtype=torch.float32).element_size()
    _plan_bwd_workspace(g, 1, 256, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), quant=_QSPEC, **common)
    monkeypatch.setattr(api_bwd_mod, "QUANT_SCALAR_STRIDE", 16)
    with pytest.raises(ValueError, match="QUANT_SCALAR_STRIDE"):
        _plan_bwd_workspace(g, 1, 256, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), quant=_QSPEC, **common)
    _plan_bwd_workspace(g, 1, 256, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), **common)  # bf16: untouched


def _stand_in_for_compile(blk):
    """A host-side stand-in for ``compile()`` on a DECLARED block (the stages' bodies need Rubin): the plan-time constants and a
    carve of plausible sizes, so ``execute``'s host checks and ``quant_scalars()`` can be exercised before any launch."""
    g, b, s = blk.geom, blk.batch, blk.seq_len
    blk._quant_vals = blk._quant_const_values()
    blk._ws = _plan_bwd_workspace(
        g,
        b,
        s,
        blk.act_dtype,
        blk.recompute,
        need=dict(dw_o=blk.need_dw_o, dw_norms=blk.need_dw_norms),
        sdpa_bwd_bytes=4096,
        gemm_scratch_bytes=1,
        n_ctas_q=1 if blk.need_dw_norms else 0,
        n_ctas_k=1 if blk.need_dw_norms else 0,
        delta_shape=(b, g.h_q, -(-s // 128) * 128) if (blk.quant is not None or blk.fuse_gate_bwd) else None,
        side_gemm_scratch_bytes=1 if blk.fuse_wgrad_overlap else None,
        quant=blk.quant,
        amax_partials_n=8 if blk.quant is not None else 0,  # compile() passes the prologue recipe's SMs x 8 cap; one SM's worth stands in
        mx_prologue_arm=blk.mx_prologue_arm,  # the MXFP8 prologue's arm keys the rebuild regions (None on the bf16 / fp8 arms)
    )
    blk._compiled_kernel = blk._ws
    blk._samples = None
    return blk


def _exec_fp8(r, ws, grads, **scalars):
    inp = r.inp
    r.blk.execute(r.dy, r.saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], workspace=ws, **grads, **scalars)


@requires_cuda
def test_fp8_execute_scalar_contracts_are_typed():
    """Rule 1 BOTH directions for the appended ``execute`` scalars, at execute and before any launch (on a declared block whose
    compile is stood in for; the scalar checks sit before the workspace and the record checks): ``scale_dp`` required under
    ``quant`` and refused without; ``scale_dy`` / ``scale_do`` / ``scale_dqkvg`` required under ``grad_scaling="delayed"``,
    refused under ``"current"`` and without ``quant``; a CPU, an fp64 or a 2-element scalar is typed, naming the input; the
    scalars join the overlap check's read side (a scalar inside the workspace is refused)."""
    r = _declare_bwd_fp8(dict(_COMMON), 1, 256)
    blk = _stand_in_for_compile(r.blk)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    ok = torch.ones(1, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="scale_dp is required"):
        _exec_fp8(r, ws, grads)
    with pytest.raises(ValueError, match="scale_dy was given"):
        _exec_fp8(r, ws, grads, scale_dp=ok, scale_dy=ok)
    with pytest.raises(ValueError, match="scale_dqkvg was given") as ei:
        _exec_fp8(r, ws, grads, scale_dp=ok, scale_dqkvg=ok)
    assert "quant_scalars()" in str(ei.value)  # the "current" recipe's scales are read back, not handed in
    for bad in (torch.ones(1, dtype=torch.float32), torch.ones(1, dtype=torch.float64, device="cuda"), torch.ones(2, dtype=torch.float32, device="cuda")):
        with pytest.raises(ValueError, match="scale_dp must be"):
            _exec_fp8(r, ws, grads, scale_dp=bad)
    inside = _view(ws, 0, (1,), torch.float32)  # a scalar INSIDE the workspace: the scalar block's init would clobber it
    with pytest.raises(ValueError, match="overlaps"):
        _exec_fp8(r, ws, grads, scale_dp=inside)
    # "delayed": the three gradient scales are required (each named)
    rd = _declare_bwd_fp8(dict(_COMMON), 1, 256, grad_scaling="delayed")
    _stand_in_for_compile(rd.blk)
    wsd = torch.empty(rd.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="scale_dy is required") as ei:
        _exec_fp8(rd, wsd, _alloc_grads(rd.blk), scale_dp=ok)
    assert "delayed" in str(ei.value)
    with pytest.raises(ValueError, match="scale_do is required"):
        _exec_fp8(rd, wsd, _alloc_grads(rd.blk), scale_dp=ok, scale_dy=ok, scale_dqkvg=ok)
    # the bf16 backward takes none of them
    r16 = _declare_bwd(dict(_COMMON), 1, 256)
    _stand_in_for_compile(r16.blk)
    ws16 = torch.empty(r16.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="scale_dp was given") as ei:
        _exec_fp8(r16, ws16, _alloc_grads(r16.blk), scale_dp=ok)
    assert "without quant" in str(ei.value)
    with pytest.raises(ValueError, match="scale_do was given"):
        _exec_fp8(r16, ws16, _alloc_grads(r16.blk), scale_do=ok)


@requires_cuda
def test_quant_scalars_are_zero_copy_views_of_the_workspace():
    """``quant_scalars(workspace)`` hands out 1-element fp32 VIEWS of the scalar block -- one per ``QUANT_SCALAR_SLOTS`` name,
    in slot order, at ``quant_scalars + QUANT_SCALAR_STRIDE * i`` (4-byte aligned), allocating nothing; a write through a view
    lands in the workspace; the rest of the 256-B region is untouched; the same workspace checks as ``execute``; a bf16 block
    has no scalar block (typed), and the call needs ``compile()`` first."""
    blk = _stand_in_for_compile(_declare_bwd_fp8(dict(_COMMON), 1, 256).blk)
    lay = blk._layout()
    assert lay.quant_scalars >= 0 and lay.quant_scalars % _WS_ALIGN == 0 and lay.o_gated == -1 and lay.recompute_v == -1 and lay.delta >= 0
    ws = torch.zeros(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    views = blk.quant_scalars(ws)
    assert list(views) == list(QUANT_SCALAR_SLOTS) and len(views) == 29 and list(views)[15:] == list(QUANT_CONST_SLOTS)
    for i, (name, v) in enumerate(views.items()):
        assert v.dtype == torch.float32 and v.numel() == 1 and v.device == ws.device and v.data_ptr() % 4 == 0, name
        # zero-copy: the view's storage IS the workspace's (a structural fact; a memory_allocated() delta would also see what
        # the other tests of the process free or allocate in between)
        assert v.untyped_storage().data_ptr() == ws.untyped_storage().data_ptr(), name
        assert v.data_ptr() == ws.data_ptr() + lay.quant_scalars + QUANT_SCALAR_STRIDE * i, name
        v.fill_(float(i + 1))
    block = _view(ws, lay.quant_scalars, (len(QUANT_SCALAR_SLOTS),), torch.float32)
    assert torch.equal(block, torch.arange(1, len(QUANT_SCALAR_SLOTS) + 1, dtype=torch.float32, device="cuda"))
    used = len(QUANT_SCALAR_SLOTS) * QUANT_SCALAR_STRIDE
    assert bool(ws[lay.quant_scalars + used : lay.quant_scalars + QUANT_SCALARS_BYTES].eq(0).all())
    assert views["descale_dp"].item() == 15.0 and QUANT_SCALAR_SLOTS.index("descale_dp") == 14 and QUANT_SCALAR_SLOTS.index("amax_dp") == 3
    with pytest.raises(ValueError, match="workspace is"):
        blk.quant_scalars(ws[:-256])
    with pytest.raises(ValueError, match="quant"):
        _stand_in_for_compile(_declare_bwd(dict(_COMMON), 1, 256).blk).quant_scalars(ws)
    with pytest.raises(RuntimeError, match="compile"):
        _declare_bwd_fp8(dict(_COMMON), 1, 256).blk.quant_scalars(ws)


# ---------------------------------------------------------------------------
# The MXFP8 backward (quant=MxQuantSpec): its declaration and declines (any CUDA device), its carve (anywhere)
# ---------------------------------------------------------------------------

# The MXFP8 backward's stage list in launch order (module docstring of api_bwd, "The MXFP8 backward"): the fp8 chain's shape -- the fused
# PROLOGUE and EPILOGUE, one dual-axis dO quantize; the two projection GEMMs are the block-scale stages, the out-projection ones the
# per-tensor fp8 stages.
_MXFP8_STAGES = [
    "_MxQuantPrologue",  # init | dY amax partials | the Q / K rebuild's MX epilogue (q8 / q_T8 / k8 / k_T8 with their blobs) | v8
    "_QuantizeGrad",  # dY, per-tensor
    "_OutProjDgrad",
    "_SigmoidGateBwd",
    "_QuantizeMxfp8",  # dO, DUAL-AXIS: rowwise + columnwise from one read
    "_OutProjWgrad",
    "_SdpaBwdMxfp8",
    "_QkNormRopeBwd",
    "_MxQuantEpilogue",  # dW_norm reduce | the dual-axis canonical dQKVG cast (rowwise + transposed)
    "_QkvGateWgrad",
    "_QkvGateDgrad",
]
# (name, axis, sf_layout, transposed, heads-of) of the ONE standalone MXFP8 quantize stage of the fused chain -- the dual-axis dO
# launch (its columnwise half rides the same stage: `dual=True`); the other eight block quantizations are jobs of the fused PROLOGUE
# (q / q_T / k / k_T / v) and EPILOGUE (dqkvg / dqkvg_T)
_MXFP8_QUANTIZES = [
    ("quantize_mxfp8_do", "row", "sdpa", False, "h_q"),
]
# The eight slots the MXFP8 arm writes a non-zero value to (the dY point and the three live constants); the other 21 read 0.0.
_MXFP8_LIVE_SLOTS = ("amax_dy", "scale_dy", "descale_dy", "alpha_b1", "alpha_b2", "scale_o", "descale_o", "descale_w_o")
_MXFP8_LIVE_CONSTS = ("scale_o", "descale_o", "descale_w_o")


def _declare_bwd_mxfp8(geom_kw, batch, seq_len, *, spec="record", h_dtype=_E4M3, w_dtype=_E4M3, dy_dtype=torch.bfloat16, saved_replace=None, **bwd_kw):
    """A DECLARED (not compiled) MXFP8 backward over a DECLARED MXFP8 training forward's record -- the e4m3 codes and the two
    scale-factor blobs as the forward's test tree builds them, the record's own calibrated ``MxQuantSpec`` (``spec="record"``) or
    the given one; CUDA tensors, no launch, any CUDA device.  ``h_dtype`` / ``w_dtype`` re-cast the record's ``h`` / the weights
    (the declaration reads dtypes / shapes / None-ness only, never a code's value); ``saved_replace`` edits the record first."""
    r = _declare_quant(geom_kw, batch, seq_len, "mxfp8")
    saved = _alloc_saved(r.geom, r.inp, batch, seq_len, save_mode="proj_slab", act_dtype=torch.bfloat16)
    saved = dataclasses.replace(saved, h=saved.h.to(h_dtype), **(saved_replace or {}))
    inp = {**r.inp, "h": saved.h, "w_qkvg": r.inp["w_qkvg"].to(w_dtype), "w_o": r.inp["w_o"].to(w_dtype)}
    dy = _make_dy(r.out).to(dy_dtype)
    kw = dict(bwd_kw)
    kw["quant"] = r.spec if spec == "record" else spec
    blk = GatedAttentionBlockBwd(dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], r.geom, **kw)
    return SimpleNamespace(
        blk=blk, fwd=r, inp=inp, inp16=r.inp16, spec=r.spec, saved=saved, dy=dy, out=r.out, geom=r.geom, geom_kw=geom_kw, batch=batch, seq_len=seq_len
    )


def _mx_artifacts(inp16: dict, geom, *, h_t=True, w_qkvg_t=True) -> dict:
    """The MXFP8 backward's four caller artifacts built the way the convergence harness does -- ``h`` re-quantized along TOKENS
    (the contiguous e4m3 ``[d_model, T]`` with its padded blob), ``W_qkvg`` re-quantized along N (``[d_model, N]`` + blob) -- from the
    bf16 inputs the record was quantized from; the oracle's own rowwise quantizer and blob builder (``gated_block_reference``)."""
    from gated_block_reference import mx_quantize_rowwise_2d, mx_swizzle_sf_rowwise_padded

    out = {}
    if h_t:
        h = inp16["h"]
        t, dm = h.shape[0] * h.shape[1], h.shape[2]
        codes, e = mx_quantize_rowwise_2d(h.reshape(t, dm).t().contiguous())
        out.update(h_t=codes.contiguous(), h_t_sf=mx_swizzle_sf_rowwise_padded(e))
    if w_qkvg_t:
        codes, e = mx_quantize_rowwise_2d(inp16["w_qkvg"].t().contiguous())
        out.update(w_qkvg_t=codes.contiguous(), w_qkvg_t_sf=mx_swizzle_sf_rowwise_padded(e))
    return out


@requires_cuda
def test_mxfp8_declaration_declines_are_typed():
    """Every typed decline of the MXFP8 backward's DECLARATION, on any CUDA device, before the Rubin gate and before any stage is
    asked -- and the ONE place its message texts are pinned (every other test matches the attribute name only).  At construction:
    the fp4 weight modes CONSTRUCT (``w_qkvg_dtype`` e2m1, ``o_fp4``: their backward is ``test_block_backward_fp4.py``'s) and an e4m3
    weight handed to such a block is ``check_support``'s decline naming the weight and the field, e5m2 codes (``MxQuantSpec.validate``),
    ``thd=True`` with an MxQuantSpec (names BOTH attributes, says dense-only, and never the row's flag), ``grad_scaling`` outside its
    vocabulary.  At ``check_support``: an fp16 ``sample_dy`` (the MXFP8 backward is bf16); the record / weight dtype gates BOTH
    ways (a bf16 ``saved.h`` with an MxQuantSpec, e4m3 codes without one -- the bf16 backward's message now names ``quant=MxQuantSpec``
    and the transposed artifacts --, bf16 weights with an MxQuantSpec); ``B*S % 32 != 0`` EXACTLY when ``need_dw_qkvg`` (the weight
    gradient contracts over the tokens through the block-scale GEMM: declined with the three fixes named, constructed and passing
    the block-level checks with ``need_dw_qkvg=False``; the rule binds ``B*S``, so S = 1008 at B = 2 and S = 992 at B = 1 pass while S =
    1000 at B = 1 and at B = 2 are declined); every bf16 decline unchanged (padding spot-checked)."""
    b, s = 1, 256
    # -- construction: the fp4 weight modes CONSTRUCT (their backward is served); an e4m3 weight under them is check_support's decline, by name --
    r4 = _declare_bwd_mxfp8(dict(_COMMON), b, s, spec=MxQuantSpec(descale_w_o=0.125, w_qkvg_dtype=torch.float4_e2m1fn_x2))
    assert r4.blk.w_qkvg_dtype == torch.float4_e2m1fn_x2 and r4.blk.w_o_dtype == _E4M3 and r4.blk.o_fp4 is None
    with pytest.raises(ValueError, match="w_qkvg") as ei:
        r4.blk.check_support()
    assert "float4_e2m1fn_x2" in str(ei.value) and "w_qkvg_dtype" in str(ei.value), str(ei.value)
    r4 = _declare_bwd_mxfp8(dict(_COMMON), b, s, spec=MxQuantSpec(descale_w_o=1.0, scale_o=1.0, o_fp4=Fp4Format.NVFP4))
    assert r4.blk.o_fp4 is Fp4Format.NVFP4 and r4.blk.w_o_dtype == torch.float4_e2m1fn_x2 and r4.blk.w_qkvg_dtype == _E4M3
    with pytest.raises(ValueError, match="w_o") as ei:
        r4.blk.check_support()
    assert "float4_e2m1fn_x2" in str(ei.value) and "o_fp4" in str(ei.value), str(ei.value)
    # -- construction: e5m2, thd + spec, grad_scaling --
    with pytest.raises(NotImplementedError, match="MxQuantSpec"):
        _declare_bwd_mxfp8(dict(_COMMON), b, s, spec=MxQuantSpec(descale_w_o=0.125, dtype=torch.float8_e5m2))
    with pytest.raises(ValueError, match="grad_scaling"):
        _declare_bwd_mxfp8(dict(_COMMON), b, s, grad_scaling="static")
    assert _declare_bwd_mxfp8(dict(_COMMON), b, s, grad_scaling="delayed").blk.grad_scaling == "delayed"
    lens = torch.tensor([128, 128], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError) as ei:  # at CONSTRUCTION, whatever the record carries
        _declare_bwd_mxfp8(dict(_COMMON), 1, 256, thd=True, num_sequences=2, max_seq_len=128, saved_replace=dict(seq_lens=lens, seq_lens_form="lengths"))
    msg = str(ei.value)
    assert "thd=True" in msg and "quant=MxQuantSpec" in msg and "dense-only" in msg, msg
    assert "external_delta" not in msg, msg
    # -- the activation dtype under an MxQuantSpec --
    r = _declare_bwd_mxfp8(dict(_COMMON), b, s, dy_dtype=torch.float16)
    with pytest.raises(ValueError, match="quant=MxQuantSpec") as ei:
        r.blk.check_support()
    assert "bfloat16" in str(ei.value) and "float16" in str(ei.value)
    # -- the record's h, both directions --
    r = _declare_bwd_mxfp8(dict(_COMMON), b, s, h_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="saved.h") as ei:
        r.blk.check_support()
    assert "e4m3 codes" in str(ei.value) and "MxQuantSpec" in str(ei.value)
    r = _declare_bwd(dict(_COMMON), b, s)
    r.blk._samples["saved"] = dataclasses.replace(r.saved, h=r.saved.h.to(_E4M3))
    with pytest.raises(ValueError, match="e4m3 codes") as ei:
        r.blk.check_support()
    msg = str(ei.value)  # the bf16 backward's message names BOTH native declarations and the MXFP8 arm's artifacts
    assert "quant=QuantSpec" in msg and "quant=MxQuantSpec" in msg and "h_t" in msg and "w_qkvg_t" in msg, msg
    # -- the weights, both directions --
    r = _declare_bwd_mxfp8(dict(_COMMON), b, s, w_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="w_qkvg") as ei:
        r.blk.check_support()
    assert "quant=MxQuantSpec" in str(ei.value) and "w_qkvg_t" in str(ei.value)
    # -- B*S % 32: declined with need_dw_qkvg (the public text, the three fixes), served without; the rule binds B*S --
    for bb, ss in ((1, 1000), (2, 1000)):
        r = _declare_bwd_mxfp8(dict(_COMMON), bb, ss)
        with pytest.raises(ValueError, match="need_dw_qkvg") as ei:
            r.blk.check_support()
        msg = str(ei.value)
        assert f"T = B*S = {bb * ss}" in msg and f"B*S must be a multiple of 32 (got {bb * ss})" in msg, msg
        assert "need_dw_qkvg=False" in msg and "quant=QuantSpec" in msg and "pad or batch the sequence" in msg, msg
        lean = _declare_bwd_mxfp8(dict(_COMMON), bb, ss, need_dw_qkvg=False).blk
        _passes_the_block_level_checks(lean, attr="need_dw_qkvg")
        _passes_the_block_level_checks(lean, attr="% 32")
    for bb, ss in ((1, 992), (2, 1008), (1, 256)):
        _passes_the_block_level_checks(_declare_bwd_mxfp8(dict(_COMMON), bb, ss).blk, attr="need_dw_qkvg")
    # -- every bf16 decline is unchanged (one spot check: padding under an MxQuantSpec is the dense block's padding decline) --
    r = _declare_bwd_mxfp8(dict(_COMMON), b, s, seq_lens_present=True)
    with pytest.raises(NotImplementedError, match="sdpa_bwd_sm107"):
        r.blk.check_support()


@requires_cuda
def test_mxfp8_declaration_wires_the_stage_list():
    """``quant=MxQuantSpec`` builds the MXFP8 backward's stage list in its launch order -- the fused PROLOGUE (the scalar init, the
    dY amax partials, the Q / K rebuild's MX epilogue and the v quantize as its jobs), the per-tensor dY quantize, the e4m3 out_proj
    dgrad, the gate backward's fp8 arm WITHOUT the amax folds, the DUAL-AXIS dO block quantize, the e4m3 out_proj wgrad, the MXFP8
    SDPA stage, the norm backward (no amax fold), the fused EPILOGUE (the dW_norm reduce and the dual-axis canonical dQKVG cast), the
    two block-scale projection GEMMs -- with every declaration fact: the out-projection GEMMs per-tensor e4m3 (``alpha=True``, bf16
    out, the 64-byte MMA K), the projection GEMMs ``block_scale=True`` (``alpha=False``, bf16 out, the same K64); the one standalone
    quantize's (axis, layout, transposed, heads, dual); the prologue's slot facts and arm (the carve's key); the epilogue's three jobs
    following the needs -- the needs dropping exactly their launches (a cast half folds out, the epilogue launch stays while a job
    remains, and goes when none does); every standalone launch of the unfused chain unbuilt; ``"delayed"`` flipping the dY quantize
    alone; the bf16 and the fp8 declarations untouched."""
    from cudnn.gated_attention_block.api_bwd import _FP8_GEMM_MMA_TILE_K_BYTES, _MxQuantEpilogue, _MxQuantPrologue, _QuantizeGrad
    from cudnn.gated_attention_block.api import _QuantizeMxfp8

    on = _declare_bwd_mxfp8(dict(_COMMON), 1, 256).blk
    assert [type(st).__name__ for st in on._stages] == _MXFP8_STAGES
    assert isinstance(on.quant, MxQuantSpec) and on.grad_scaling == "current" and on.w_dtype == _E4M3 and on.act_dtype == torch.bfloat16
    assert isinstance(on._prologue, _MxQuantPrologue) and isinstance(on._epilogue, _MxQuantEpilogue) and on._quant_vals is None  # VALUES at compile()
    assert on._stages[0] is on._prologue and on._stages[-3] is on._epilogue
    # the unfused chain's standalone launches are the fused launches' jobs now: none is built
    assert on._scalar_init is None and on._amax_dy is None and on._recompute_qk is None and on._compact_v is None and on._quant_do_T is None
    assert on._quant_q is None and on._quant_q_T is None and on._quant_k is None and on._quant_k_T is None and on._quant_v is None
    assert on._quant_dqkvg is None and on._quant_dqkvg_T is None
    assert (on._prologue.n_slots, on._prologue.const_slot0, on._prologue.n_consts) == (len(QUANT_SCALAR_SLOTS), 15, len(QUANT_CONST_SLOTS))
    assert on._prologue.arm == on.mx_prologue_arm == "mx_epilogue" and on._prologue.dtype == torch.bfloat16 and on._prologue._recipe is None
    assert (on._epilogue.want_dw, on._epilogue.want_row, on._epilogue.want_col) == (True, True, True) and on._epilogue.dtype == torch.bfloat16
    d = _COMMON["d_head"]
    qdy = on._quant_dy
    assert isinstance(qdy, _QuantizeGrad)
    assert (qdy.heads, qdy.n_alpha, qdy.own_amax, qdy.amax_src, qdy.scale_src, qdy.persistent) == (_COMMON["d_model"] // d, 2, False, "partials", "amax", False)
    for st in on._stages:
        if isinstance(st, _GemmStage):
            if st.label in ("out_proj_dgrad", "out_proj_wgrad"):
                assert (st.dtype, st.alpha, st.out_dtype, st.mma_tile_k_bytes, st.block_scale) == (
                    _E4M3,
                    True,
                    torch.bfloat16,
                    _FP8_GEMM_MMA_TILE_K_BYTES,
                    False,
                ), st.label
            else:
                assert (st.dtype, st.alpha, st.out_dtype, st.mma_tile_k_bytes, st.block_scale) == (
                    _E4M3,
                    False,
                    torch.bfloat16,
                    _FP8_GEMM_MMA_TILE_K_BYTES,
                    True,
                ), st.label
                assert st.majors == ("k", "k"), st.label
    gb = on._gate_bwd
    assert (gb.want_og, gb.og_fp8, gb.want_amax_do, gb.want_amax_dg, gb.want_delta) == (True, True, False, False, True)
    assert on._norm_bwd.want_amax is False and on._norm_bwd.want_dw is True
    heads_of = dict(h_q=_COMMON["h_q"], h_kv=_COMMON["h_kv"], n_per_d=on.geom.n_qkvg // d)
    quantizes = [st for st in on._stages if isinstance(st, _QuantizeMxfp8)]
    assert [(st.name, st.axis, st.sf_layout, st.transposed, st.heads) for st in quantizes] == [
        (name, axis, layout, transposed, heads_of[h]) for name, axis, layout, transposed, h in _MXFP8_QUANTIZES
    ]
    assert all(st.dtype_in == torch.bfloat16 for st in quantizes)
    assert quantizes == [on._quant_do] and on._quant_do.dual is True and on._quant_do.transposed_second is False  # the dO dual-axis launch
    # the needs drop exactly their launches; a cast half of the epilogue folds out, the launch stays while one of its jobs remains
    no_wo = _declare_bwd_mxfp8(dict(_COMMON), 1, 256, need_dw_o=False).blk
    assert [type(st).__name__ for st in no_wo._stages] == [n for n in _MXFP8_STAGES if n != "_OutProjWgrad"]
    assert (no_wo._gate_bwd.want_og, no_wo._gate_bwd.og_fp8, no_wo._gate_bwd.want_delta) == (False, False, True)
    no_wq = _declare_bwd_mxfp8(dict(_COMMON), 1, 256, need_dw_qkvg=False).blk
    assert no_wq._qkv_gate_wgrad is None and (no_wq._epilogue.want_dw, no_wq._epilogue.want_row, no_wq._epilogue.want_col) == (True, True, False)
    assert len(no_wq._stages) == len(_MXFP8_STAGES) - 1
    no_dh = _declare_bwd_mxfp8(dict(_COMMON), 1, 256, need_dh=False).blk
    assert no_dh._qkv_gate_dgrad is None and (no_dh._epilogue.want_dw, no_dh._epilogue.want_row, no_dh._epilogue.want_col) == (True, False, True)
    assert len(no_dh._stages) == len(_MXFP8_STAGES) - 1
    rope = _declare_bwd_mxfp8({**_COMMON, "qk_norm": False}, 1, 256).blk
    assert rope._norm_bwd.want_dw is False and [type(st).__name__ for st in rope._stages] == _MXFP8_STAGES
    assert (rope._epilogue.want_dw, rope._epilogue.want_row, rope._epilogue.want_col) == (False, True, True)  # the epilogue stays for the cast
    only_wo = _declare_bwd_mxfp8(dict(_COMMON), 1, 256, need_dh=False, need_dw_qkvg=False, need_dw_norms=False).blk
    assert only_wo._epilogue is None and [type(st).__name__ for st in only_wo._stages] == [
        n for n in _MXFP8_STAGES if n not in ("_MxQuantEpilogue", "_QkvGateWgrad", "_QkvGateDgrad")
    ]
    delayed = _declare_bwd_mxfp8(dict(_COMMON), 1, 256, grad_scaling="delayed").blk
    assert delayed._quant_dy.scale_src == "given" and (delayed._quant_dy.own_amax, delayed._quant_dy.amax_src) == (False, "partials")
    # the bf16 and the fp8 declarations are untouched
    off = _declare_bwd(dict(_COMMON), 1, 256).blk
    assert off.quant is None and off._scalar_init is None and off._amax_dy is None and [type(st).__name__ for st in off._stages] == _BF16_STAGES
    fp8 = _declare_bwd_fp8(dict(_COMMON), 1, 256).blk
    assert fp8._scalar_init is None and fp8._amax_dy is None and fp8._quant_q is None and [type(st).__name__ for st in fp8._stages] == _FP8_STAGES


def test_mxfp8_fused_stage_contracts_are_typed():
    """Host, no GPU (the two fused stage classes alone): ``_MxQuantPrologue.check_support`` declines, GEOMETRY FIRST and device-free -- a
    non-bf16 record (``NotImplementedError``), the slot arithmetic (no slot; constants outside the block; more constants than the init
    ABI's arguments), ``d_model % d_head`` (the dY view), a ``d_head`` the MX token tile cannot serve (``validate_mx_shape``: the
    warp-per-row rule) -- and then the arch (the TMA ring: SM90 or newer; on an older part that is the one decline left, naming TMA);
    ``_MxQuantEpilogue.check_support`` declines a non-bf16 gradient, a dW_norm job without the norm, nothing to launch and a
    transposed half at ``B*S % 32 != 0``.  The prologue's ``arm`` is ``"mx_epilogue"``, a member of ``MX_PROLOGUE_ARMS``; both stages
    refuse ``execute`` before ``compile``."""
    from cudnn.gated_attention_block.api_bwd import MX_PROLOGUE_ARMS, _MxQuantEpilogue, _MxQuantPrologue
    from cudnn.gated_attention_block.kernels.quantize import MAX_INIT_CONSTS

    g = GatedAttentionBlockGeometry(**_COMMON)
    n_slots = len(QUANT_SCALAR_SLOTS)
    ok = dict(batch=1, seq_len=256, dtype=torch.bfloat16, n_slots=n_slots, const_slot0=15, n_consts=14)
    pro = _MxQuantPrologue(g, **ok)
    assert pro.arm == "mx_epilogue" and pro.arm in MX_PROLOGUE_ARMS and pro.name == "mxfp8_bwd_prologue"
    with pytest.raises(NotImplementedError, match="bf16"):
        _MxQuantPrologue(g, **{**ok, "dtype": torch.float16}).check_support()
    with pytest.raises(ValueError, match="n_slots"):
        _MxQuantPrologue(g, **{**ok, "n_slots": 0}).check_support()
    with pytest.raises(ValueError, match="inside"):
        _MxQuantPrologue(g, **{**ok, "const_slot0": n_slots - 3}).check_support()
    with pytest.raises(ValueError, match="ABI"):
        _MxQuantPrologue(g, **{**ok, "n_slots": 64, "const_slot0": 0, "n_consts": MAX_INIT_CONSTS + 1}).check_support()
    with pytest.raises(ValueError, match="d_model"):
        _MxQuantPrologue(GatedAttentionBlockGeometry(**{**_COMMON, "d_model": 640}), **ok).check_support()
    with pytest.raises(ValueError, match="d_head"):
        _MxQuantPrologue(GatedAttentionBlockGeometry(**{**_COMMON, "d_head": 128}), **ok).check_support()
    with pytest.raises(RuntimeError, match="compile"):
        pro.n_partials()
    with pytest.raises(RuntimeError, match="compile"):
        pro.execute(**{k: None for k in ("slots", "dy", "partials", "q_pre", "k_pre", "w_q", "w_k", "cos", "sin", "q8", "sf_q", "q_T8", "sf_q_T", "k8", "sf_k", "k_T8", "sf_k_T", "v", "v8", "sf_v", "stream")})  # fmt: skip
    cc = _cc()
    if cc is not None:  # the arch decision below asks the ambient device; without one the device-free checks above are the test
        if cc[0] < 9:
            with pytest.raises(NotImplementedError, match="SM90"):
                pro.check_support()  # the geometry passed; the TMA ring is the one decline left on this part
        else:
            pro.check_support()
    epi = dict(batch=1, seq_len=256, dtype=torch.bfloat16, want_dw=True, want_row=True, want_col=True)
    assert _MxQuantEpilogue(g, **epi).name == "mxfp8_bwd_epilogue"
    _MxQuantEpilogue(g, **epi).check_support()
    with pytest.raises(NotImplementedError, match="bf16"):
        _MxQuantEpilogue(g, **{**epi, "dtype": torch.float16}).check_support()
    with pytest.raises(ValueError, match="qk_norm"):
        _MxQuantEpilogue(GatedAttentionBlockGeometry(**{**_COMMON, "qk_norm": False}), **epi).check_support()
    with pytest.raises(ValueError, match="nothing to launch"):
        _MxQuantEpilogue(g, **{**epi, "want_dw": False, "want_row": False, "want_col": False}).check_support()
    with pytest.raises(ValueError, match="32"):
        _MxQuantEpilogue(g, **{**epi, "seq_len": 1000}).check_support()
    _MxQuantEpilogue(g, **{**epi, "seq_len": 1000, "want_col": False}).check_support()  # a dgrad-only block serves a ragged T
    with pytest.raises(RuntimeError, match="compile"):
        _MxQuantEpilogue(g, **epi).execute(
            plane_q=None, plane_k=None, dw_q_norm=None, dw_k_norm=None, src=None, dst=None, sf=None, dst_t=None, sf_t=None, stream=0
        )


@requires_cuda
def test_fuse_gate_bwd_has_no_effect_under_mxfp8():
    """Under an ``MxQuantSpec`` the gate backward's delta is MANDATORY (it is the MXFP8 SDPA row's external delta), so
    ``fuse_gate_bwd`` has no second arm: both declarations resolve the same stage list, the gate stage wants the delta under both,
    and the SDPA stage is built with the external delta under both.  Accepted with either value; the Rubin cells pin the
    gradients bitwise."""
    off = _declare_bwd_mxfp8(dict(_COMMON), 1, 256).blk
    on = _declare_bwd_mxfp8(dict(_COMMON), 1, 256, fuse_gate_bwd=True).blk
    assert (off.fuse_gate_bwd, on.fuse_gate_bwd) == (False, True)
    assert [type(st).__name__ for st in on._stages] == [type(st).__name__ for st in off._stages] == _MXFP8_STAGES
    assert off._gate_bwd.want_delta is True and on._gate_bwd.want_delta is True
    for blk in (off, on):
        assert blk._sdpa.delta_shape == (1, _COMMON["h_q"], 256) and blk._sdpa._impl.external_delta is True


def test_workspace_carve_under_mxfp8_is_the_declared_composition():
    """``_plan_bwd_workspace(quant=MxQuantSpec)`` on any device -- the pure-carve twin of
    ``test_workspace_carve_under_quant_is_the_declared_composition``: the bf16 regions keep their places and sizes except the two
    the MXFP8 backward does not write (``o_gated`` -> -1: B3's third output is the e4m3 ``og8``; ``recompute_v`` -> -1: ``v8`` IS V's
    compaction) -- the bf16 ``recompute`` / ``recompute_k`` ARE carved under the default and the bf16-rebuild prologue arm (the bf16
    rebuild feeds the block quantizes there) and NOT under the MX-epilogue arm the block runs (``mx_prologue_arm``: the fused prologue
    writes the four Q / K payloads out of registers), every later region then moving up by exactly their two aligned sizes --, ``delta``
    is MANDATORY, and the MXFP8 regions are appended AFTER every bf16 region in the documented order, every payload right before its
    scale-factor blob (the SDPA-layout blobs at ``_sf_slot_bytes``, the canonical ones at ``sf_blob_bytes``), each padded to the carve
    alignment, then the scalar block and the dY partials; ``og8`` follows ``need_dw_o``; the dO / dG / band partials are never carved
    and a non-zero count for them is typed; an arm outside ``MX_PROLOGUE_ARMS`` or an arm without an MxQuantSpec is typed; the
    ``quant=None`` and the ``quant=QuantSpec`` layouts are byte-identical to before (every MXFP8 field -1 on both)."""
    from cudnn.gated_attention_block.api import _sf_slot_bytes
    from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes

    g = GatedAttentionBlockGeometry(**_COMMON)
    b, s, e = 2, 256, 2
    t, d, n = b * s, g.d_head, g.n_qkvg
    mx = MxQuantSpec(descale_w_o=0.125, scale_o=16.0)
    al = lambda x: -(-x // _WS_ALIGN) * _WS_ALIGN  # noqa: E731
    common = dict(sdpa_bwd_bytes=1000, gemm_scratch_bytes=4096, n_ctas_q=7, n_ctas_k=3, delta_shape=(b, g.h_q, 384), side_gemm_scratch_bytes=512)
    need = dict(dw_o=True, dw_norms=True)
    lay16 = _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=need, **common)
    lay8 = _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=need, quant=_QSPEC, **common)
    laymx = _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=need, quant=mx, **common)
    mx_fields = ("sf_do", "do_T8", "sf_do_T", "sf_q", "q_T8", "sf_q_T", "sf_k", "k_T8", "sf_k_T", "sf_v", "sf_dqkvg", "dqkvg_t8", "sf_dqkvg_t")
    for lay in (lay16, lay8):
        for f in mx_fields:
            assert getattr(lay, f) == -1, f
    sf_q, sf_kv = _sf_slot_bytes(b, g.h_q, s, d), _sf_slot_bytes(b, g.h_kv, s, d)
    sizes = dict(
        do_gated=t * g.h_q * d * e,
        dqkvg=t * n * e,
        recompute=t * g.h_q * d * e,
        recompute_k=t * g.h_kv * d * e,
        dq=t * g.h_q * d * e,
        dk=t * g.h_kv * d * e,
        dv=t * g.h_kv * d * e,
        dw_partials_q=7 * d * 4,
        dw_partials_k=3 * d * 4,
        sdpa_bwd_ws=1000,
        gemm_scratch=4096,
        delta=b * g.h_q * 384 * 4,
        gemm_scratch_side=512,
        dy8=t * g.d_model,
        do8=t * g.h_q * d,
        sf_do=sf_q,
        do_T8=t * g.h_q * d,
        sf_do_T=sf_q,
        og8=t * g.h_q * d,
        q8=t * g.h_q * d,
        sf_q=sf_q,
        q_T8=t * g.h_q * d,
        sf_q_T=sf_q,
        k8=t * g.h_kv * d,
        sf_k=sf_kv,
        k_T8=t * g.h_kv * d,
        sf_k_T=sf_kv,
        v8=t * g.h_kv * d,
        sf_v=sf_kv,
        dqkvg8=t * n,
        sf_dqkvg=sf_blob_bytes(t, n),
        dqkvg_t8=n * t,
        sf_dqkvg_t=sf_blob_bytes(n, t),
        quant_scalars=QUANT_SCALARS_BYTES,
    )
    order = [
        "do_gated",
        "dqkvg",
        "recompute",
        "recompute_k",
        "dq",
        "dk",
        "dv",
        "dw_partials_q",
        "dw_partials_k",
        "sdpa_bwd_ws",
        "gemm_scratch",
        "delta",
        "gemm_scratch_side",
    ]
    order += ["dy8", "do8", "sf_do", "do_T8", "sf_do_T", "og8", "q8", "sf_q", "q_T8", "sf_q_T", "k8", "sf_k", "k_T8", "sf_k_T", "v8", "sf_v"]
    order += ["dqkvg8", "sf_dqkvg", "dqkvg_t8", "sf_dqkvg_t", "quant_scalars"]
    off = 0
    for name in order:
        assert getattr(laymx, name) == off, (name, getattr(laymx, name), off)
        assert off % _WS_ALIGN == 0, name
        off += al(sizes[name])
    assert laymx.total_bytes == off and laymx.o_gated == -1 and laymx.recompute_v == -1 and laymx.recompute >= 0 and laymx.recompute_k >= 0
    assert laymx.do == -1 and laymx.base_align == _WS_ALIGN and laymx.delta >= 0 and laymx.delta_shape == (b, g.h_q, 384)
    assert (laymx.amax_partials, laymx.amax_partials_n) == (-1, 0)  # the pure carve
    assert (laymx.amax_partials_do, laymx.amax_partials_dg, laymx.gate_partials_n, laymx.amax_partials_bands, laymx.band_partials_n) == (-1, -1, 0, -1, 0)
    # the shared prefix up to dqkvg is byte-identical to the bf16 carve, the rebuild slots too; o_gated's drop shifts the rest
    assert (laymx.do_gated, laymx.dqkvg) == (lay16.do_gated, lay16.dqkvg) and laymx.recompute == lay16.recompute - al(t * g.h_q * d * e)
    # the dY partials: a positive count carves fp32 [n] LAST; the producers' partials are refused under an MxQuantSpec
    cap = 204 * 8
    with_p = _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=need, quant=mx, amax_partials_n=cap, **common)
    assert (with_p.amax_partials, with_p.amax_partials_n) == (laymx.total_bytes, cap) and with_p.total_bytes == laymx.total_bytes + al(cap * 4)
    assert with_p.quant_scalars == laymx.quant_scalars and with_p.sf_dqkvg_t == laymx.sf_dqkvg_t
    for bad_kw in (dict(gate_partials_n=cap), dict(band_partials_n=12)):
        with pytest.raises(ValueError, match="MxQuantSpec"):
            _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=need, quant=mx, **bad_kw, **common)
    # og8 follows need_dw_o; need_og8 overrides it in the direction B1 can live with only
    lean = _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_o=False, dw_norms=False), quant=mx, **common)
    assert lean.og8 == -1 and lean.o_gated == -1 and lean.q8 == lean.sf_do_T + al(sf_q) and lean.dw_partials_q == -1
    forced = _plan_bwd_workspace(
        g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_o=False, dw_norms=False), quant=mx, need_og8=True, **common
    )
    assert forced.og8 == lean.sf_do_T + al(sf_q) and forced.total_bytes == lean.total_bytes + al(t * g.h_q * d)
    with pytest.raises(ValueError, match="need_og8"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_o=True, dw_norms=False), quant=mx, need_og8=False, **common)
    # delta is mandatory under an MxQuantSpec (the gate backward's delta is the row's external delta)
    no_delta = {k: v for k, v in common.items() if k != "delta_shape"}
    with pytest.raises(ValueError, match="delta_shape"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=dict(dw_norms=False), quant=mx, **no_delta)
    # the per-token delta vs the bf16 carve at this geometry: the appended payloads + blobs + the scalar block, minus o_gated and recompute_v
    added = sum(al(sizes[name]) for name in order if name in ("dy8", "do8", "sf_do", "do_T8", "sf_do_T", "og8", "q8", "sf_q", "q_T8", "sf_q_T"))
    added += sum(al(sizes[name]) for name in ("k8", "sf_k", "k_T8", "sf_k_T", "v8", "sf_v", "dqkvg8", "sf_dqkvg", "dqkvg_t8", "sf_dqkvg_t", "quant_scalars"))
    assert laymx.total_bytes - lay16.total_bytes == added - al(t * g.h_q * d * e) - al(t * g.h_kv * d * e)
    # the fused prologue's MX-epilogue ARM (the block's: GatedAttentionBlockBwd.mx_prologue_arm) carves NEITHER bf16 rebuild region, and
    # every later region moves up by exactly their two aligned sizes; the bf16-rebuild arm is the default carve, byte for byte; a bad arm
    # and an arm without an MxQuantSpec are typed
    from cudnn.gated_attention_block.api_bwd import MX_PROLOGUE_ARM_BF16_REBUILD, MX_PROLOGUE_ARM_MX_EPILOGUE, MX_PROLOGUE_ARMS

    assert MX_PROLOGUE_ARMS == (MX_PROLOGUE_ARM_MX_EPILOGUE, MX_PROLOGUE_ARM_BF16_REBUILD) == ("mx_epilogue", "bf16_rebuild")
    fused = _plan_bwd_workspace(
        g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=need, quant=mx, mx_prologue_arm=MX_PROLOGUE_ARM_MX_EPILOGUE, **common
    )
    shift = al(sizes["recompute"]) + al(sizes["recompute_k"])
    assert fused.recompute == -1 and fused.recompute_k == -1 and fused.total_bytes == laymx.total_bytes - shift
    moved = False
    for name in order:
        if name in ("recompute", "recompute_k"):
            moved = True
            continue
        assert getattr(fused, name) == getattr(laymx, name) - (shift if moved else 0), (name, getattr(fused, name), getattr(laymx, name))
    assert (fused.do_gated, fused.dqkvg) == (laymx.do_gated, laymx.dqkvg) and fused.dq == laymx.dqkvg + al(sizes["dqkvg"])
    kept = _plan_bwd_workspace(
        g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=need, quant=mx, mx_prologue_arm=MX_PROLOGUE_ARM_BF16_REBUILD, **common
    )
    assert kept == laymx
    with pytest.raises(ValueError, match="mx_prologue_arm"):
        _plan_bwd_workspace(g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=need, quant=mx, mx_prologue_arm="rowwise", **common)
    for q_other in (None, _QSPEC):
        with pytest.raises(ValueError, match="MxQuantSpec"):
            _plan_bwd_workspace(
                g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, need=need, quant=q_other, mx_prologue_arm=MX_PROLOGUE_ARM_MX_EPILOGUE, **common
            )
    # the fp8 carve is untouched by the MXFP8 arm (its own test pins the composition; here: the same bytes as before, no MX field)
    assert lay8.recompute == -1 and lay8.recompute_k == -1 and lay8.o_gated == -1 and lay8.recompute_v == -1


def test_workspace_carve_under_mxfp8_follows_the_projection_weight_gradient_and_a_padded_s():
    """``_plan_bwd_workspace(quant=MxQuantSpec)`` anywhere (no device): (a) the TRANSPOSED dQKVG payload and its canonical blob are B7's
    operands alone, carved iff ``need["dw_qkvg"]`` (missing = True, like the other flags) -- a dgrad-only block at a ragged token count
    (T = 1000, the matrix's dgrad-only cell) carves neither and raises nothing, while the same T WITH the weight gradient is refused by
    the canonical blob builder (whole 32-token blocks along T; on a real block ``check_support``'s ``B*S % 32`` rule fires first with the
    public text); at a served T the dgrad-only layout is the full one minus exactly those two regions, the scalar block moving up into
    the transposed payload's place; the bf16 and fp8 carves ignore the flag.  (b) A padded S (992: q- and kv-padded on the row) carves
    every MXFP8 region: the SDPA-layout blobs at the forward's ``_sf_slot_bytes`` (its 128-row atom pad), the canonical blobs at
    ``sf_blob_bytes`` over the REAL token count (992 = 31 x 32) -- the block's own carve never pads tokens; the row's kv-padded
    per-Q-head partials live inside the adapter's opaque ``sdpa_bwd_ws`` slot, which is why the matrix compares them on the live rows."""
    from cudnn.gated_attention_block.api import _sf_slot_bytes
    from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes

    g = GatedAttentionBlockGeometry(**_COMMON)
    mx = MxQuantSpec(descale_w_o=0.125, scale_o=16.0)
    al = lambda x: -(-x // _WS_ALIGN) * _WS_ALIGN  # noqa: E731
    d, n = g.d_head, g.n_qkvg
    plan = lambda b, s, **kw: _plan_bwd_workspace(  # noqa: E731
        g, b, s, torch.bfloat16, RecomputePolicy.RECOMPUTE_QK_PRE, sdpa_bwd_bytes=1000, gemm_scratch_bytes=4096, n_ctas_q=7, n_ctas_k=3, **kw
    )
    # (a) the transposed pair follows need_dw_qkvg: a ragged T is served without the weight gradient, refused with it
    b, s = 1, 1000
    t = b * s
    delta = dict(delta_shape=(b, g.h_q, 1024), side_gemm_scratch_bytes=512)
    with pytest.raises(ValueError, match="32"):
        plan(b, s, need=dict(dw_norms=True), quant=mx, **delta)
    lean = plan(b, s, need=dict(dw_norms=True, dw_qkvg=False), quant=mx, **delta)
    assert lean.dqkvg_t8 == -1 and lean.sf_dqkvg_t == -1
    assert lean.dqkvg8 >= 0 and lean.sf_dqkvg == lean.dqkvg8 + al(t * n) and lean.quant_scalars == lean.sf_dqkvg + al(sf_blob_bytes(t, n))
    assert lean.dqkvg >= 0, "the bf16 dqkvg slab stays: B8 reads its rowwise quantization"
    for f in ("dy8", "do8", "sf_do", "do_T8", "sf_do_T", "og8", "q8", "sf_q", "q_T8", "sf_q_T", "k8", "sf_k", "k_T8", "sf_k_T", "v8", "sf_v", "delta"):
        assert getattr(lean, f) >= 0, f
    # at a served T the dgrad-only layout is the full one minus exactly the two regions
    b, s = 2, 256
    t = b * s
    delta = dict(delta_shape=(b, g.h_q, 256), side_gemm_scratch_bytes=512)
    full = plan(b, s, need=dict(dw_norms=True), quant=mx, **delta)
    lean = plan(b, s, need=dict(dw_norms=True, dw_qkvg=False), quant=mx, **delta)
    assert full.dqkvg_t8 == full.sf_dqkvg + al(sf_blob_bytes(t, n)) and full.sf_dqkvg_t == full.dqkvg_t8 + al(n * t)
    assert lean.quant_scalars == full.dqkvg_t8 and lean.total_bytes == full.total_bytes - al(n * t) - al(sf_blob_bytes(n, t))
    for f in ("dqkvg8", "sf_dqkvg", "dy8", "og8", "q8", "sf_v", "delta", "recompute", "recompute_k", "sdpa_bwd_ws", "gemm_scratch_side"):
        assert getattr(lean, f) == getattr(full, f), f
    for q in (None, _QSPEC):  # no transposed pair on the bf16 / fp8 carves: the flag is inert there
        assert plan(b, s, need=dict(dw_norms=True), quant=q, **delta) == plan(b, s, need=dict(dw_norms=True, dw_qkvg=False), quant=q, **delta)
    # (b) a padded S: every region carved; SDPA-layout blobs at the 128-row atom pad, canonical blobs over the real token count
    b, s = 1, 992
    t = b * s
    pad = plan(b, s, need=dict(dw_norms=True), quant=mx, delta_shape=(b, g.h_q, 1024), side_gemm_scratch_bytes=512)
    sf_q, sf_kv = _sf_slot_bytes(b, g.h_q, s, d), _sf_slot_bytes(b, g.h_kv, s, d)
    assert sf_q == _sf_slot_bytes(b, g.h_q, 1024, d) and sf_kv == _sf_slot_bytes(b, g.h_kv, 1024, d), "the SDPA-layout blob pads S to the 128-row atom"
    assert pad.sf_do == pad.do8 + al(t * g.h_q * d) and pad.do_T8 == pad.sf_do + al(sf_q) and pad.sf_v == pad.v8 + al(t * g.h_kv * d)
    assert pad.dqkvg8 == pad.sf_v + al(sf_kv) and pad.sf_dqkvg == pad.dqkvg8 + al(t * n)
    assert pad.dqkvg_t8 == pad.sf_dqkvg + al(sf_blob_bytes(t, n)) and pad.sf_dqkvg_t == pad.dqkvg_t8 + al(n * t)
    assert pad.quant_scalars == pad.sf_dqkvg_t + al(sf_blob_bytes(n, t)) and pad.delta_shape == (b, g.h_q, 1024) and pad.total_bytes % _WS_ALIGN == 0
    assert sf_blob_bytes(n, t) == sf_blob_bytes(n, 1024), "the canonical blob pads its K blocks to 4 (992 / 32 = 31 -> 32 blocks)"


@requires_cuda
def test_mxfp8_plan_time_constants_are_init_launch_arguments():
    """Host, any CUDA device, no compile and no launch: under an ``MxQuantSpec`` the plan-time constants are the SAME 14 slots
    (``QUANT_CONST_SLOTS``, the tail of the unchanged 29-slot tuple), their VALUES Python floats resolved at ``compile()`` -- the three
    live ones (``scale_o``, ``descale_o = 1 / scale_o``, ``descale_w_o``) at the MxQuantSpec's values and the eleven dead ones exactly 0.0
    (never 1.0) -- handed to the fused PROLOGUE's init job as kernel ARGUMENTS (``_execute_mxfp8`` spells ``consts=tuple(vals[n]
    for n in QUANT_CONST_SLOTS)``), the prologue declared over the whole block with the constants' slot range and WITHOUT the
    ``descale_dp`` division (no dP scalar); ``compile()`` writes nothing to the device."""
    r = _declare_bwd_mxfp8(dict(_COMMON), 1, 256)
    blk, spec = r.blk, r.spec
    assert blk._quant_vals is None
    vals = blk._quant_const_values()
    assert tuple(vals) == QUANT_CONST_SLOTS and len(vals) == 14
    assert all(type(v) is float and math.isfinite(v) for v in vals.values()), vals
    want = {name: 0.0 for name in QUANT_CONST_SLOTS}
    want.update(scale_o=float(spec.scale_o), descale_o=1.0 / float(spec.scale_o), descale_w_o=float(spec.descale_w_o))
    assert vals == want, (vals, want)
    assert all(vals[n] != 0.0 for n in _MXFP8_LIVE_CONSTS) and all(vals[n] == 0.0 for n in QUANT_CONST_SLOTS if n not in _MXFP8_LIVE_CONSTS)
    init = blk._prologue  # the fused prologue's first job is the scalar init: it carries the init body's slot facts
    assert blk._stages[0] is init and (init.n_slots, init.const_slot0, init.n_consts) == (len(QUANT_SCALAR_SLOTS), 15, 14)
    exe = inspect.getsource(GatedAttentionBlockBwd._execute_mxfp8)
    assert "consts=tuple(vals[n] for n in QUANT_CONST_SLOTS)" in exe, "the init launch is handed every plan-time constant, in slot order"
    assert "descale_dp" not in exe.replace("no descale_dp", "").replace("no scale_dp, no descale_dp", ""), "the MXFP8 arm derives no descale_dp"
    src = inspect.getsource(GatedAttentionBlockBwd.compile) + inspect.getsource(GatedAttentionBlockBwd._quant_const_values)
    assert (
        "torch.full" not in src and "torch.empty" not in src and "torch.zeros" not in src and "device=" not in src
    ), "compile() must write nothing to the device"
    assert set(_MXFP8_LIVE_SLOTS) <= set(QUANT_SCALAR_SLOTS) and len(_MXFP8_LIVE_SLOTS) == 8


@requires_cuda
def test_mxfp8_execute_contracts_are_typed():
    """Rule 1 BOTH directions for the appended ``execute`` scalars and artifacts under an ``MxQuantSpec``, at execute and before any
    launch (a declared block whose compile is stood in for; the checks sit before the workspace and the record checks): ``scale_dp``
    / ``scale_do`` / ``scale_dqkvg`` REFUSED (no dP scalar; block-scaled dO / dQKVG), ``scale_dy`` refused under ``"current"`` and
    required under ``"delayed"``; ``h_t`` / ``h_t_sf`` required iff ``need_dw_qkvg`` and ``w_qkvg_t`` / ``w_qkvg_t_sf`` iff ``need_dh``
    (each named with its need), refused when the need is off and on a QuantSpec / bf16 block; a ``.t()`` VIEW of the un-transposed
    codes refused by name (the strides), a wrong blob byte count / dtype by ``_check_sf_blob``'s message, a wrong code dtype / shape /
    device; an artifact inside the workspace joins the overlap check."""
    from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes

    r = _declare_bwd_mxfp8(dict(_COMMON), 1, 256)
    blk = _stand_in_for_compile(r.blk)
    g, t = blk.geom, blk.batch * blk.seq_len
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    art = _mx_artifacts(r.inp16, g)
    ok = torch.ones(1, dtype=torch.float32, device="cuda")

    def run(**kw):
        kw = {**art, **kw}
        _exec_fp8(r, ws, grads, **kw)

    for name in ("scale_dp", "scale_do", "scale_dqkvg"):
        with pytest.raises(ValueError, match=f"{name} was given") as ei:
            run(**{name: ok})
        assert "MxQuantSpec" in str(ei.value), str(ei.value)
    with pytest.raises(ValueError, match="scale_dy was given") as ei:
        run(scale_dy=ok)
    assert "quant_scalars()" in str(ei.value)
    # the artifacts: required with their need, each named with it
    for name in ("h_t", "h_t_sf", "w_qkvg_t", "w_qkvg_t_sf"):
        with pytest.raises(ValueError, match=f"{name} is required") as ei:
            run(**{name: None})
        assert ("need_dw_qkvg" if name.startswith("h_t") else "need_dh") in str(ei.value), str(ei.value)
    # a .t() view of the un-transposed codes: refused by name (the strides)
    with pytest.raises(ValueError, match="h_t has strides") as ei:
        run(h_t=r.inp["h"].reshape(t, g.d_model).t())
    assert ".t() view" in str(ei.value)
    with pytest.raises(ValueError, match="w_qkvg_t has strides"):
        run(w_qkvg_t=r.inp["w_qkvg"].t())
    # wrong code dtype / shape / device
    with pytest.raises(ValueError, match="h_t must be the torch.float8_e4m3fn codes"):
        run(h_t=art["h_t"].to(torch.bfloat16))
    with pytest.raises(ValueError, match="w_qkvg_t must be \\[d_model"):
        run(w_qkvg_t=art["w_qkvg_t"][:, :256].contiguous())
    with pytest.raises(ValueError, match="h_t must live on"):
        run(h_t=art["h_t"].cpu())
    # the blobs: _check_sf_blob's message on a wrong byte count / dtype, the device check
    with pytest.raises(ValueError, match="h_t_sf has") as ei:
        run(h_t_sf=art["h_t_sf"][:-512])
    assert f"{sf_blob_bytes(g.d_model, t)}" in str(ei.value), str(ei.value)
    with pytest.raises(ValueError, match="w_qkvg_t_sf must be"):
        run(w_qkvg_t_sf=art["w_qkvg_t_sf"].to(torch.bfloat16))
    with pytest.raises(ValueError, match="h_t_sf must live on"):
        run(h_t_sf=art["h_t_sf"].cpu())
    # an artifact inside the workspace: the overlap check names it
    inside = _view(ws, 0, tuple(art["h_t"].shape), _E4M3)
    with pytest.raises(ValueError, match="overlaps"):
        run(h_t=inside)
    # "delayed": scale_dy is required (and the artifacts still are)
    rd = _declare_bwd_mxfp8(dict(_COMMON), 1, 256, grad_scaling="delayed")
    _stand_in_for_compile(rd.blk)
    wsd = torch.empty(rd.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="scale_dy is required") as ei:
        _exec_fp8(rd, wsd, _alloc_grads(rd.blk), **art)
    assert "delayed" in str(ei.value)
    # the needs off: the artifact is refused by its need
    lean = _declare_bwd_mxfp8(dict(_COMMON), 1, 256, need_dw_qkvg=False)
    _stand_in_for_compile(lean.blk)
    wsl = torch.empty(lean.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="h_t was given") as ei:
        _exec_fp8(lean, wsl, _alloc_grads(lean.blk), **art)
    assert "need_dw_qkvg=False" in str(ei.value)
    only_h = {k: v for k, v in art.items() if k.startswith("h_t")}
    lean2 = _declare_bwd_mxfp8(dict(_COMMON), 1, 256, need_dh=False)
    _stand_in_for_compile(lean2.blk)
    wsl2 = torch.empty(lean2.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="w_qkvg_t was given") as ei:
        _exec_fp8(lean2, wsl2, _alloc_grads(lean2.blk), **art)
    assert "need_dh=False" in str(ei.value)
    assert only_h  # the h_t pair alone is what that block takes
    # a QuantSpec block and a bf16 block refuse every artifact
    r8 = _declare_bwd_fp8(dict(_COMMON), 1, 256)
    _stand_in_for_compile(r8.blk)
    ws8 = torch.empty(r8.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="h_t was given") as ei:
        _exec_fp8(r8, ws8, _alloc_grads(r8.blk), scale_dp=ok, h_t=art["h_t"])
    assert "quant=QuantSpec" in str(ei.value) and "MxQuantSpec" in str(ei.value)
    r16 = _declare_bwd(dict(_COMMON), 1, 256)
    _stand_in_for_compile(r16.blk)
    ws16 = torch.empty(r16.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="w_qkvg_t_sf was given") as ei:
        _exec_fp8(r16, ws16, _alloc_grads(r16.blk), w_qkvg_t_sf=art["w_qkvg_t_sf"])
    assert "without quant" in str(ei.value)
