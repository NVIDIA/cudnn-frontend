# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's TRAINING-LOOP surface: the per-step recalibration API of both blocks and the gradient-scale margin
attribute of the quantized backward.

A training step moves every scale a quantized block was declared with -- the weights change, so ``descale_w_*`` (and ``descale_h``)
are recomputed from the tensors quantized this step, and a per-tensor fp8 recipe recalibrates ``scale_q / scale_k / scale_v / scale_o``
from a previous step's activation amax.  ``GatedAttentionBlockFwd.update_quant_scales(spec, *, current_stream=None)`` re-points a
COMPILED forward at a new spec with an in-place write of its device scalars on the launch stream; ``GatedAttentionBlockBwd.
update_quant_scales(spec)`` re-resolves the plan-time constants the backward's prologue launch stores from its kernel arguments.
``GatedAttentionBlockBwd(grad_scale_margin_log2=)`` is the appended declaration attribute that drives the "current" recipe's
power-of-two headroom through EVERY gradient quantize (a post-import rebind of the module constant reaches the fused epilogue's dQKVG
quantize only -- the signature defaults of the dY / dO stages were bound at import -- which is a mixed recipe no gate catches).

What is pinned here:

* the forward API rewrites the device scalars: ``out`` after ``update_quant_scales(B)`` is BITWISE the ``out`` of a block DECLARED
  with ``B`` (unfused training pipeline and fully fused inference fork, per-tensor fp8 and MXFP8), the scalars read back as ``B``'s,
  the fused fp8 fork's scale vector follows, and under ``o_fp4`` the call validates and records (nothing to write);
* the refusals, typed, BEFORE any write (host: any CUDA device);
* stream ordering: an update + execute on a side stream while the default stream is parked equals the eager result, the FIRST
  execute of a compiled block writes the scalars on its own launch stream (the allocated scalars poisoned on purpose), and a
  recalibration on a side stream outlives pending work on the stream ``compile()`` ran on (it allocates and never fills);
* the backward API changes the prologue's constants: ``quant_scalars()`` reads ``B``'s values and the gradients equal a backward
  DECLARED with ``B``, bitwise;
* the margin reaches every quantize launch: ``scale_dy / scale_do / scale_dqkvg`` (``scale_dy`` alone under MXFP8) are
  ``grad_scale_from_amax(amax, 2)`` bitwise and NOT the margin-0 value, and the default-margin block is bitwise a block declared
  without the attribute; the convenience wrapper's cache tells the margin ``1`` from ``True`` and ``1.0`` (host).

And the TRAINING LOOP over the toy decoder of ``gated_block_train`` (the convergence harness core, a tracked helper of this directory):

* the bf16 block as the attention layer of a small decoder vs the pure-torch reference model from one seed: the step-0 master
  gradients of ``W_qkvg`` / ``W_o`` of every layer within the bf16 backward bound of the accept suite (asserted), the norm-weight
  gradients under the suite's cosine floor and a worst-cell ceiling of that bound, the per-step losses within a relative bound;
* three steps of the bf16 and the MXFP8 arm finite, the per-layer lists live (``descale_w_o`` differs between layers), the bf16
  run with both backward knobs bitwise the default-knob run;
* the deterministic-replay pin: two runs from one seed digest equal (every row, ``grad_sha256`` included), for the torch control on
  any CUDA device and for the bf16 / MXFP8 arms on Rubin;
* the feed-less ``delayed`` gradient recipe bootstraps its scales from a ladder of discarded step-0 passes so that its logged step 0
  is BITWISE the ``current`` recipe's (fp8 and MXFP8), completes, and collapses no gradient;
* a CUDA graph that captured an execute of a WARMED-UP block replays with the values a later eager ``update_quant_scales`` wrote,
  while a graph that captured the block's FIRST execute replays its capture-time scalars (the documented limitation);
* the harness core imports only from the standard library, torch, the ``cudnn`` package and this test tree.

Accept tests are ``requires_rubin`` (the block targets SM107 only); the refusal tests, the torch-control pin and the import
detector carry no marker and run on any CUDA device.
"""

import ast
import dataclasses
import importlib
import inspect
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block training-loop tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import (  # noqa: E402
    GatedAttentionBlockBwd,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    MxQuantSpec,
    gated_attention_block_backward,
)
from cudnn.gated_attention_block import api_bwd as _api_bwd  # noqa: E402
from cudnn.gated_attention_block.api import Fp4Format  # noqa: E402
from cudnn.gated_attention_block.kernels import quantize as _quantize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gated_block_train as gbt  # noqa: E402
from gated_block_stream_probe import park_the_default_stream  # noqa: E402
from test_block_backward import _ATOL_FRAC, _COMMON, _COS_MIN, _RTOL, _alloc_grads, _assert_grad_close, _declare_bwd  # noqa: E402
from test_block_backward_fp8 import _backward_fp8, _declare_fp8_bwd, _execute_fp8, _f32, _fp8_decl  # noqa: E402
from test_block_backward_mxfp8 import _backward_mxfp8, _declare_mx_bwd, _execute_mx  # noqa: E402
from test_block_fp8 import _require_fp8_forks  # noqa: E402
from test_block_mxfp8 import _mxfp8_fork_missing  # noqa: E402
from test_block_training_forward import _alloc_saved, _declare, _declare_quant, _run_training_quant  # noqa: E402

_E4M3 = torch.float8_e4m3fn

# The REGISTERED marker of cutedsl/conftest.py (the skip is applied at collection).
requires_rubin = pytest.mark.requires_rubin

_FAMILY = pytest.mark.parametrize("family", ["fp8", "mxfp8"])
_GEOM = dict(_COMMON, qk_norm=True, is_causal=True)  # the smallest accept cell of the quantized suites (norm, GQA 8/2, causal)
_B, _S = 1, 256
_API_CONST = _api_bwd.FP8_GRAD_SCALE_MARGIN_LOG2

# The training loop: the accept suites' head geometry at the small model size (2 layers, 2 x 1024 tokens, a 4096 vocabulary), three steps.
_SMOKE = gbt.GEOMETRIES["smoke"]
_STEPS = 3
# The block's norm-weight gradients, model vs model: the suite's own dW_norm bound is a MASS bound over the fp64 oracle's per-row terms,
# unavailable here, so they are held to the suite's cosine floor plus a worst-cell CEILING of the bf16 bound -- CALIBRATED ONCE (Rubin
# cc 10.7) from the measured 1.105 / 1.155 (a [D] reduction over T bf16 products whose noise does not shrink with layer 1's 8x smaller
# max|ref|): 2.0 leaves a 1.7x margin, like the suite's _DW_NORM_NOISE.  Never widen.
_NORM_W_BOUND_CEIL = 2.0
# The per-step relative loss difference between the bf16 block model and the torch reference model: CALIBRATED ONCE from the measured
# 1.44e-5 over three steps (a 7x margin).  Never widen.
_LOSS_REL_BOUND = 1e-4
# The row keys a feed-less delayed run's step 0 must share with the current recipe's step 0 (the ladder seeds the kernels' own rule).
_STEP0_KEYS = ("grad_sha256", "loss", "grad_norm_total", "quant_scalars", "grad_norms", "spec", "scale_dp", "act_amax", "sat", "n_clip")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_b(spec, *, fused: bool = False):
    """A second spec of ``spec``'s class whose SCALES differ (every plan fact untouched) and which moves the output: powers of two, so
    every value stays exact in fp32 and valid.  On the fully fused MXFP8 path only ``descale_w_o`` may move (``scale_o`` is pinned
    to 1.0 there)."""
    if isinstance(spec, MxQuantSpec):
        if fused:
            return dataclasses.replace(spec, descale_w_o=spec.descale_w_o * 2.0)
        return dataclasses.replace(spec, descale_w_o=spec.descale_w_o * 2.0, scale_o=spec.scale_o * 0.5)
    return dataclasses.replace(
        spec,
        descale_h=spec.descale_h * 2.0,
        descale_w_qkvg=spec.descale_w_qkvg * 0.5,
        descale_w_o=spec.descale_w_o * 2.0,
        scale_q=spec.scale_q * 0.5,
        scale_k=spec.scale_k * 2.0,
        scale_v=spec.scale_v * 0.5,
        scale_o=spec.scale_o * 0.5,
    )


def _skip_without_the_fused_fork(family: str) -> None:
    """The fully fused forks are kernel files of their own: skip (never fail) where a fork has not landed."""
    if family == "fp8":
        _require_fp8_forks()
    else:
        missing = _mxfp8_fork_missing()
        if missing:
            pytest.skip(str(missing))


def _fwd_block_with(r, spec, *, training: bool, **blk_kw):
    """A block over ``r``'s inputs declared with ``spec`` (plus the MXFP8 blobs when the family needs them) and a fresh output buffer."""
    inp = r.inp
    extra = dict(sample_h_sf=inp["h_sf"], sample_w_qkvg_sf=inp["w_qkvg_sf"]) if r.family == "mxfp8" else {}
    out = torch.empty_like(r.out)
    kw = dict(quant=spec, **extra, **blk_kw)
    if training:
        kw["save_for_backward"] = True
    blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, r.geom, **kw)
    return blk, out


def _record(r):
    return _alloc_saved(r.geom, r.inp, r.batch, r.seq_len, save_mode="proj_slab", act_dtype=torch.bfloat16)


def _compile(blk) -> torch.Tensor:
    blk.check_support()
    blk.compile()
    return torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")


def _execute_fwd(r, blk, out, ws, *, saved=None, current_stream=None) -> None:
    inp = r.inp
    sf = dict(h_sf=inp["h_sf"], w_qkvg_sf=inp["w_qkvg_sf"]) if r.family == "mxfp8" else {}
    blk.execute(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, saved=saved, current_stream=current_stream, **sf
    )


def _dev_values(blk) -> dict:
    torch.cuda.synchronize()
    return {k: float(v.item()) for k, v in blk._quant_dev.items()}


def _f32_values(vals: dict) -> dict:
    return {k: _f32(v) for k, v in vals.items()}


def _assert_grads_equal(got: dict, want: dict, what: str) -> None:
    for name, ten in got.items():
        if ten is not None:
            assert torch.equal(ten, want[name]), f"{name}: {what}"


def _grads_differ(got: dict, other: dict) -> bool:
    return any(ten is not None and not torch.equal(ten, other[name]) for name, ten in got.items())


# ---------------------------------------------------------------------------
# The surface (host, no GPU kernel)
# ---------------------------------------------------------------------------


def test_the_recalibration_surface_is_appended_and_keyword_only():
    """Host: the two methods' signatures, and ``grad_scale_margin_log2`` as the LAST parameter of ``GatedAttentionBlockBwd.__init__``
    and of the convenience wrapper (keyword-only, defaulted to the module constant); ``_QuantEpilogue(margin_log2=)`` appended and
    defaulted the same way (every existing caller traces the same artifact)."""
    fwd = inspect.signature(GatedAttentionBlockFwd.update_quant_scales).parameters
    assert (
        list(fwd) == ["self", "spec", "current_stream"]
        and fwd["current_stream"].kind is inspect.Parameter.KEYWORD_ONLY
        and fwd["current_stream"].default is None
    )
    assert list(inspect.signature(GatedAttentionBlockBwd.update_quant_scales).parameters) == ["self", "spec"]
    for fn in (GatedAttentionBlockBwd.__init__, gated_attention_block_backward):
        p = inspect.signature(fn).parameters
        last = list(p)[-1]
        assert last == "grad_scale_margin_log2", (fn.__qualname__, last)
        assert p[last].kind is inspect.Parameter.KEYWORD_ONLY and p[last].default == _API_CONST, fn.__qualname__
    pe = inspect.signature(_api_bwd._QuantEpilogue.__init__).parameters
    assert list(pe)[-1] == "margin_log2" and pe["margin_log2"].default == _API_CONST
    assert _api_bwd._GRAD_SCALE_MARGIN_LOG2_MAX == 8


# ---------------------------------------------------------------------------
# The refusals (host, any CUDA device)
# ---------------------------------------------------------------------------


def test_update_quant_scales_refusals():
    """Every refusal of both methods fires BEFORE any write, typed, on DECLARED blocks: a bf16 block (``ValueError``), a spec of the
    other class (``TypeError``), a differing plan fact (``ValueError`` naming ``dtype`` / ``block_size`` / ``w_qkvg_dtype`` / ``o_fp4``),
    a zero / inf / NaN / negative scale (the ``QuantSpec`` / ``MxQuantSpec`` class's own ``ValueError``, reached through the method), a block not yet compiled
    (``RuntimeError``); the fully fused MXFP8 forward applies ``validate(fused=True)`` exactly as its declaration did (``scale_o != 1.0``
    is its typed ``NotImplementedError``)."""
    fp8 = _declare_quant(_GEOM, _B, _S, "fp8")
    mx = _declare_quant(_GEOM, _B, _S, "mxfp8")
    bf16, _inp, _out = _declare(_GEOM, _B, _S)
    with pytest.raises(ValueError, match="without quant"):
        bf16.update_quant_scales(fp8.spec)
    with pytest.raises(TypeError, match="QuantSpec"):
        fp8.blk.update_quant_scales(mx.spec)
    with pytest.raises(TypeError, match="MxQuantSpec"):
        mx.blk.update_quant_scales(fp8.spec)
    with pytest.raises(ValueError, match="dtype"):
        fp8.blk.update_quant_scales(dataclasses.replace(fp8.spec, dtype=torch.float8_e5m2))
    for name, val in (("dtype", torch.float8_e5m2), ("block_size", 16), ("w_qkvg_dtype", torch.float4_e2m1fn_x2), ("o_fp4", Fp4Format.NVFP4)):
        with pytest.raises(ValueError, match=name):
            mx.blk.update_quant_scales(dataclasses.replace(mx.spec, **{name: val}))
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="scale_q"):
            fp8.blk.update_quant_scales(dataclasses.replace(fp8.spec, scale_q=bad))
        with pytest.raises(ValueError, match="descale_w_o"):
            mx.blk.update_quant_scales(dataclasses.replace(mx.spec, descale_w_o=bad))
    with pytest.raises(RuntimeError, match="compile"):
        fp8.blk.update_quant_scales(fp8.spec)
    with pytest.raises(RuntimeError, match="compile"):
        mx.blk.update_quant_scales(mx.spec)
    assert fp8.blk.quant is fp8.spec and mx.blk.quant is mx.spec, "a refused call must leave the declared spec in place"
    mxf = _declare_quant(_GEOM, _B, _S, "mxfp8", training=False, fuse_norm_rope=True, fuse_gate=True, scale_o=1.0)
    with pytest.raises(NotImplementedError, match="scale_o"):
        mxf.blk.update_quant_scales(dataclasses.replace(mxf.spec, scale_o=2.0))
    # the backward's method, the same refusals
    d = _fp8_decl(_GEOM, _B, _S)
    bwd16 = _declare_bwd(_GEOM, _B, _S).blk
    with pytest.raises(ValueError, match="without quant"):
        bwd16.update_quant_scales(d.spec)
    with pytest.raises(TypeError, match="QuantSpec"):
        d.blk.update_quant_scales(mx.spec)
    with pytest.raises(ValueError, match="dtype"):
        d.blk.update_quant_scales(dataclasses.replace(d.spec, dtype=torch.float8_e5m2))
    with pytest.raises(ValueError, match="scale_q"):
        d.blk.update_quant_scales(dataclasses.replace(d.spec, scale_q=0.0))
    with pytest.raises(RuntimeError, match="compile"):
        d.blk.update_quant_scales(d.spec)
    assert d.blk.quant is d.spec and d.blk._quant_vals is None


def test_grad_scale_margin_refusals_and_plumbing():
    """Host: ``grad_scale_margin_log2`` is validated at declaration -- an int in [0, 8] (a float, a bool, a negative value, 9 and a
    string are ``ValueError``), refused non-default without ``quant`` (a bf16 block quantizes nothing) and under
    ``grad_scaling="delayed"`` (the caller's scales carry their own headroom) -- and a legal value reaches the three gradient quantize
    stages and the fused epilogue of a DECLARED fp8 backward; the default is the module constant."""
    for bad in (1.5, True, -1, 9, "2"):
        with pytest.raises(ValueError, match="grad_scale_margin_log2"):
            _fp8_decl(_GEOM, _B, _S, grad_scale_margin_log2=bad)
    with pytest.raises(ValueError, match="grad_scale_margin_log2"):
        _fp8_decl(_GEOM, _B, _S, grad_scaling="delayed", grad_scale_margin_log2=2)
    with pytest.raises(ValueError, match="grad_scale_margin_log2"):
        _declare_bwd(_GEOM, _B, _S, grad_scale_margin_log2=2)
    _declare_bwd(_GEOM, _B, _S, grad_scale_margin_log2=_API_CONST)  # the default spelled out is fine on a bf16 block
    for margin in (0, 2, 8):
        blk = _fp8_decl(_GEOM, _B, _S, grad_scale_margin_log2=margin).blk
        assert blk.grad_scale_margin_log2 == margin
        assert (blk._quant_dy.margin_log2, blk._quant_do.margin_log2, blk._epilogue.margin_log2) == (margin, margin, margin)
    default = _fp8_decl(_GEOM, _B, _S).blk
    assert default.grad_scale_margin_log2 == _API_CONST == default._quant_dy.margin_log2 == default._quant_do.margin_log2 == default._epilogue.margin_log2


def test_the_backward_wrapper_cache_keys_the_margin_by_type(monkeypatch):
    """Host: the convenience wrapper caches compiled blocks by a key that tells the margin ``1`` from ``True`` and from ``1.0`` (Python
    compares the three equal), so a block cached at an int margin is never handed to a caller passing a bool or a float -- they reach
    the class's typed ``ValueError`` on their own miss.  The class is stubbed and the cache spied: no device compile runs, the pin is
    the key itself."""
    seen = []

    class _SpyCache(dict):
        def get(self, key, default=None):
            seen.append(key)
            return None

    class _Stop(Exception):
        pass

    class _StubBwd:
        def __init__(self, *args, **kwargs):
            raise _Stop(kwargs.get("grad_scale_margin_log2"))

    monkeypatch.setattr(_api_bwd, "_BWD_CACHE", _SpyCache())
    monkeypatch.setattr(_api_bwd, "GatedAttentionBlockBwd", _StubBwd)
    d = _fp8_decl(_GEOM, _B, _S)
    inp = d.inp
    grads_of = (d.saved.h, inp["w_qkvg"], inp["w_o"], inp["w_q_norm"], inp["w_k_norm"])
    for t_ in grads_of:
        t_.requires_grad_(True)
    try:
        for margin in (1, True, 1.0):
            with pytest.raises(_Stop):
                gated_attention_block_backward(
                    d.dy,
                    d.saved,
                    inp["w_qkvg"],
                    inp["w_q_norm"],
                    inp["w_k_norm"],
                    inp["cos"],
                    inp["sin"],
                    inp["w_o"],
                    d.geom,
                    quant=d.spec,
                    grad_scale_margin_log2=margin,
                )
    finally:
        for t_ in grads_of:
            t_.requires_grad_(False)
    assert len(seen) == 3 and len(set(seen)) == 3, "the wrapper's cache key does not tell the margin 1 from True and 1.0"
    assert 1 == True == 1.0  # the aliasing the key must see through  # noqa: E712


# ---------------------------------------------------------------------------
# The forward API on the device
# ---------------------------------------------------------------------------


@requires_rubin
@_FAMILY
@pytest.mark.parametrize("pipeline", ["unfused_training", "fused_inference"])
def test_update_quant_scales_rewrites_the_device_scalars(family, pipeline):
    """Declare + compile a block with spec ``A``, execute; ``update_quant_scales(B)``, execute again -> ``out`` is BITWISE the ``out`` of a
    block DECLARED with ``B`` (and differs from the ``A`` run: ``B`` moved the output), the device scalars read back as ``B``'s fp32
    values and ``blk.quant`` is ``B``.  Two pipelines per family: the UNFUSED training forward (the record written through) and the
    FULLY FUSED inference fork, whose per-tensor fp8 twin reads ``[alpha_qkvg, scale_q, scale_k, scale_v]`` from the projection
    stage's own vector -- rewritten too."""
    fused = pipeline == "fused_inference"
    if fused:
        _skip_without_the_fused_fork(family)
    kw = dict(fuse_norm_rope=True, fuse_gate=True) if fused else {}
    r = _declare_quant(_GEOM, _B, _S, family, training=not fused, scale_o=1.0 if (fused and family == "mxfp8") else None, **kw)
    a = r.spec
    saved_a = _record(r) if not fused else None
    ws = _compile(r.blk)
    out_a = torch.empty_like(r.out)
    _execute_fwd(r, r.blk, out_a, ws, saved=saved_a)
    torch.cuda.synchronize()
    assert torch.isfinite(out_a.float()).all()
    b = _spec_b(a, fused=fused)
    r.blk.update_quant_scales(b)
    assert r.blk.quant is b
    want = r.blk._quant_dev_values(b)
    assert set(want) == set(r.blk._quant_dev) and _dev_values(r.blk) == _f32_values(want)
    if fused and family == "fp8":
        assert r.blk._proj.quant is b
        assert r.blk._proj._qscal.tolist() == [_f32(v) for v in (b.alpha_qkvg, b.scale_q, b.scale_k, b.scale_v)]
    out_ab = torch.empty_like(r.out)
    saved_ab = _record(r) if not fused else None
    _execute_fwd(r, r.blk, out_ab, ws, saved=saved_ab)
    blk_b, out_b = _fwd_block_with(r, b, training=not fused, **kw)
    ws_b = _compile(blk_b)
    saved_b = _record(r) if not fused else None
    _execute_fwd(r, blk_b, out_b, ws_b, saved=saved_b)
    torch.cuda.synchronize()
    assert torch.isfinite(out_b.float()).all()
    assert torch.equal(
        out_ab, out_b
    ), f"out after update_quant_scales(B) differs from a block declared with B: max {(out_ab.float() - out_b.float()).abs().max().item():.3e}"
    assert not torch.equal(out_ab, out_a), "B must move the output, or the pin proves nothing"
    if not fused:
        assert torch.equal(saved_ab.o, saved_b.o) and torch.equal(saved_ab.lse, saved_b.lse), "the record after the update differs from the B block's"


@requires_rubin
def test_update_quant_scales_under_o_fp4_records_the_spec():
    """Under ``MxQuantSpec.o_fp4`` both per-tensor scales of the out projection are pinned to 1.0 and the block holds NO device scalar
    (``{}``): the call validates, records ``spec`` and writes nothing; a second execute is bitwise the first.  The plan fact ``o_fp4``
    and ``MxQuantSpec``'s own unit-scale rule stay typed through the method."""
    from test_block_backward_fp4 import _CFGS, _fp4_inputs, _fwd_block, _fwd_execute  # that module skips itself without torch's fp4 dtypes

    cfg = next(c for c in _CFGS if c.name == "o_nvfp4")
    cell = SimpleNamespace(geom_kw=_GEOM, b=_B, s=_S)
    _inp16, mx, spec = _fp4_inputs(cfg, cell)
    geom = GatedAttentionBlockGeometry(**_GEOM)
    out = torch.empty(_B, _S, geom.d_model, device="cuda", dtype=torch.bfloat16)
    blk = _fwd_block(cfg, cell, mx, spec, out, save=False)
    ws = _compile(blk)
    assert blk._quant_dev == {} and blk._quant_dev_values(spec) == {}
    _fwd_execute(cfg, blk, mx, out, ws)
    torch.cuda.synchronize()
    out_a = out.clone()
    assert torch.isfinite(out_a.float()).all()
    b = dataclasses.replace(spec)  # the only legal B under o_fp4 carries A's unit scales
    blk.update_quant_scales(b)
    assert blk.quant is b and blk._quant_dev == {}
    _fwd_execute(cfg, blk, mx, out, ws)
    torch.cuda.synchronize()
    assert torch.equal(out, out_a)
    with pytest.raises(ValueError, match="o_fp4"):
        blk.update_quant_scales(dataclasses.replace(spec, o_fp4=None))
    with pytest.raises(ValueError, match="descale_w_o"):
        blk.update_quant_scales(dataclasses.replace(spec, descale_w_o=2.0))
    assert blk.quant is b


@requires_rubin
@_FAMILY
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_update_quant_scales_is_stream_ordered(family, how):
    """The default stream is parked behind a long spin; ``update_quant_scales(B)`` -- on the ambient side stream, or with the side stream
    passed as ``current_stream`` while the ambient stream stays the parked default stream (the handle alone carries the side stream)
    -- and the execute that follows run on the side stream, the workspace is zeroed right after.  A write
    that escaped to the default stream would land LATE (after the execute consumed the old scalars) and the result would differ from
    the eager ``B`` result, which it must equal BITWISE."""
    import cuda.bindings.driver as cuda_drv

    r = _run_training_quant(_GEOM, _B, _S, family)
    b = _spec_b(r.spec)
    blk_b, out_b = _fwd_block_with(r, b, training=True)
    ws_b = _compile(blk_b)
    saved_b = _record(r)
    _execute_fwd(r, blk_b, out_b, ws_b, saved=saved_b)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    out2 = torch.zeros_like(r.out)
    saved2 = _record(r)
    ws2 = torch.zeros_like(r.ws)
    torch.cuda.synchronize()
    park_the_default_stream()
    if how == "ambient":
        with torch.cuda.stream(side):
            r.blk.update_quant_scales(b)
            _execute_fwd(r, r.blk, out2, ws2, saved=saved2)
            ws2.zero_()
    else:
        # STRICT: the ambient stream stays the (parked) default stream and the side stream reaches the API through the handle
        # alone -- an update or a stage that used torch's current stream would land on the parked stream, late
        assert torch.cuda.current_stream() == torch.cuda.default_stream()
        cs = cuda_drv.CUstream(side.cuda_stream)
        r.blk.update_quant_scales(b, current_stream=cs)
        _execute_fwd(r, r.blk, out2, ws2, saved=saved2, current_stream=cs)
        with torch.cuda.stream(side):
            ws2.zero_()
    torch.cuda.synchronize()
    assert torch.equal(out2, out_b), f"the update or a stage escaped the caller's stream ({how})"
    assert torch.equal(saved2.o, saved_b.o) and torch.equal(saved2.lse, saved_b.lse), f"the record written on the side stream differs ({how})"


@requires_rubin
@_FAMILY
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_first_execute_writes_the_scales_on_its_launch_stream(family, how, tmp_path):
    """``compile()`` only ALLOCATES the device scalars; the FIRST execute writes their VALUES on ITS launch stream, so a block compiled
    on one stream and first executed on another reads what its execution stream wrote.  Pinned
    without a timing window: the allocated scalars are POISONED to NaN (synchronously) after ``compile()`` and the first execute runs
    on a side stream (ambient, or explicit with the ambient stream left at the default stream) -- its output is bitwise the
    synchronised run's and the scalars read back as ``spec``'s.
    The write is one-shot per compile: a second poison is NOT repaired by the second execute (``update_quant_scales`` is the caller's
    path to a new value), which pins where the write sits.  With CUPTI available, every device-side record of the first execute ran on
    ONE stream (the fills included)."""
    import cuda.bindings.driver as cuda_drv
    from torch.profiler import ProfilerActivity, profile

    ref = _run_training_quant(_GEOM, _B, _S, family)
    blk, out = _fwd_block_with(ref, ref.spec, training=True)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(0xFF)
    saved = _record(ref)
    out.fill_(float("nan"))
    for ten in blk._quant_dev.values():
        ten.fill_(float("nan"))
    torch.cuda.synchronize()
    assert all(v != v for v in _dev_values(blk).values()), "the poison did not land"
    side = torch.cuda.Stream()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        if how == "ambient":
            with torch.cuda.stream(side):
                _execute_fwd(ref, blk, out, ws, saved=saved)
        else:
            # STRICT: the ambient stream stays the default stream and the side stream reaches execute through the handle alone,
            # so a fill or a stage that used torch's current stream shows up as a SECOND stream in the trace below
            assert torch.cuda.current_stream() == torch.cuda.default_stream()
            _execute_fwd(ref, blk, out, ws, saved=saved, current_stream=cuda_drv.CUstream(side.cuda_stream))
        side.synchronize()
    torch.cuda.synchronize()
    assert torch.equal(out, ref.out), f"the first execute on the side stream read the poisoned compile-time scalars ({how})"
    assert _dev_values(blk) == _f32_values(blk._quant_dev_values(ref.spec))
    assert blk._quant_dev_on_launch_stream is True
    prof.export_chrome_trace(str(tmp_path / f"first_execute_{family}_{how}.json"))
    import json

    with open(tmp_path / f"first_execute_{family}_{how}.json") as f:
        events = json.load(f)["traceEvents"]
    device = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if device:
        streams = {e.get("args", {}).get("stream") for e in device}
        assert len(streams) == 1, f"the first execute ran on {len(streams)} streams ({streams}): device work escaped the execution stream"
    # one-shot: the second execute does not rewrite the scalars
    for ten in blk._quant_dev.values():
        ten.fill_(float("nan"))
    torch.cuda.synchronize()
    out2 = torch.empty_like(out)
    _execute_fwd(ref, blk, out2, ws, saved=_record(ref))
    torch.cuda.synchronize()
    assert not torch.isfinite(out2.float()).all(), "the second execute repaired the poisoned scalars: the first-execute write is not one-shot"
    # and the caller's path repairs them
    blk.update_quant_scales(ref.spec)
    _execute_fwd(ref, blk, out2, ws, saved=_record(ref))
    torch.cuda.synchronize()
    assert torch.equal(out2, ref.out)


@requires_rubin
@_FAMILY
@pytest.mark.parametrize("pipeline", ["unfused_training", "fused_inference"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["plain", "deterministic_alloc"])
def test_update_quant_scales_outlives_pending_work_on_the_compile_stream(family, pipeline, deterministic):
    """``compile()`` ALLOCATES the device scalars and writes nothing: a block compiled while its ambient stream is PARKED behind a long
    spin, recalibrated with ``update_quant_scales(B)`` and executed on a side stream, reads ``B`` -- on that first execute AND on a
    later one after the parked stream has drained.  A compile-time fill of the scalars (or of the fused fp8 fork's scale vector) on the
    parked stream would land AFTER the update and silently restore ``A`` for every later execute (the first-execute write is one-shot)
    while ``blk.quant`` still said ``B``.  Both pipelines of both families; the park must still be pending at the update (asserted, so
    the cell cannot pass vacuously).  The ``deterministic_alloc`` arm compiles under ``torch.use_deterministic_algorithms(True)``
    (``fill_uninitialized_memory`` on, the default): there ``torch.empty`` itself enqueues a NaN fill of the fresh scalars on the parked
    stream, which the writers must order themselves behind (the event ``compile()`` records) -- without it the drained park overwrites
    a completed recalibration with NaN.  The process settings are restored whatever happens."""
    fused = pipeline == "fused_inference"
    if fused:
        _skip_without_the_fused_fork(family)
    kw = dict(fuse_norm_rope=True, fuse_gate=True) if fused else {}
    r = _declare_quant(_GEOM, _B, _S, family, training=not fused, scale_o=1.0 if (fused and family == "mxfp8") else None, **kw)
    a = r.spec
    b = _spec_b(a, fused=fused)
    blk_b, out_b = _fwd_block_with(r, b, training=not fused, **kw)
    ws_b = _compile(blk_b)  # also warms the artifact caches, so the compile behind the park below is short against it
    saved_b = _record(r) if not fused else None
    _execute_fwd(r, blk_b, out_b, ws_b, saved=saved_b)
    torch.cuda.synchronize()
    assert not torch.equal(out_b, r.out) or fused, "B must move the output, or the pin proves nothing"
    side = torch.cuda.Stream()
    out1, out2 = torch.empty_like(r.out), torch.empty_like(r.out)
    saved1 = _record(r) if not fused else None
    saved2 = _record(r) if not fused else None
    # the tested block's workspace is allocated BEFORE the park (the same declaration as blk_b up to the scales, so the same size): under
    # the deterministic arm an allocation behind the park carries its own NaN fill, which no event orders and which could land in the
    # middle of the first side-stream execute -- a workspace hazard unrelated to the scale ordering this cell pins
    ws = torch.empty_like(ws_b)
    torch.cuda.synchronize()
    assert torch.cuda.current_stream() == torch.cuda.default_stream()
    prev_det, prev_fill = torch.are_deterministic_algorithms_enabled(), torch.utils.deterministic.fill_uninitialized_memory
    try:
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.utils.deterministic.fill_uninitialized_memory = True
            probe = torch.empty(4, dtype=torch.float32, device="cuda")  # the arm's premise: an allocation that fills itself
            torch.cuda.synchronize()
            if not torch.isnan(probe).all():
                pytest.skip("this torch does not fill torch.empty under deterministic mode; the plain arm covers the race")
        park_the_default_stream(seconds=6.0)  # compile()'s ambient stream: anything it enqueued there lands only after the spin
        r.blk.check_support()
        r.blk.compile()  # under the deterministic arm: its torch.empty scalar allocations carry a NaN fill queued behind the park
        assert r.blk.get_workspace_size() == ws.numel()
    finally:
        torch.use_deterministic_algorithms(prev_det)
        torch.utils.deterministic.fill_uninitialized_memory = prev_fill
    with torch.cuda.stream(side):
        r.blk.update_quant_scales(b)
        _execute_fwd(r, r.blk, out1, ws, saved=saved1)
    assert not torch.cuda.default_stream().query(), "the park drained before the update: the race window closed, lengthen the park"
    side.synchronize()
    torch.cuda.synchronize()  # the park drains: whatever compile() enqueued on the default stream has landed by now
    with torch.cuda.stream(side):
        _execute_fwd(r, r.blk, out2, ws, saved=saved2)
    torch.cuda.synchronize()
    assert torch.equal(out1, out_b), "the first execute after the cross-stream update does not read B"
    assert torch.equal(
        out2, out_b
    ), f"an execute after the compile stream drained reads scales a compile-time write restored: max {(out2.float() - out_b.float()).abs().max().item():.3e}"
    assert _dev_values(r.blk) == _f32_values(r.blk._quant_dev_values(b)), "the device scalars were overwritten after the update"
    if fused and family == "fp8":
        assert r.blk._proj._qscal.tolist() == [_f32(v) for v in (b.alpha_qkvg, b.scale_q, b.scale_k, b.scale_v)]
    if not fused:
        assert torch.equal(saved2.o, saved_b.o) and torch.equal(saved2.lse, saved_b.lse)


# ---------------------------------------------------------------------------
# The backward API and the margin on the device
# ---------------------------------------------------------------------------


@requires_rubin
@_FAMILY
def test_bwd_update_quant_scales_changes_the_prologue_constants(family):
    """After ``bwd.update_quant_scales(B)`` + execute, the scalar block's plan-time constants read ``B``'s values -- ``scale_q`` /
    ``descale_h`` / ``descale_w_o`` (fp8), ``descale_w_o`` / ``scale_o`` / ``descale_o`` (MXFP8, through the fused prologue's kernel
    arguments) -- and every gradient equals a backward DECLARED with ``B`` over the same record, bitwise (and differs from the ``A``
    run).  The memoised block is restored to ``A`` afterwards."""
    if family == "fp8":
        res = _backward_fp8(_GEOM, _B, _S)
        b = _spec_b(res.spec)
        declare, execute, extra = _declare_fp8_bwd, _execute_fp8, dict(scale_dp=res.scale_dp_t, **res.scale_ts)
        names = ("scale_q", "scale_k", "scale_v", "scale_o", "descale_h", "descale_w_qkvg", "descale_w_o")
    else:
        res = _backward_mxfp8(_GEOM, _B, _S)
        b = _spec_b(res.spec)
        declare, execute, extra = _declare_mx_bwd, (lambda *a, **kw: _execute_mx(*a, res.art, **kw)), dict(scale_dy=res.scale_dy_t)
        names = ("scale_o", "descale_o", "descale_w_o")
    want = {n: _f32(v) for n, v in res.blk._quant_const_values().items()}
    try:
        res.blk.update_quant_scales(b)
        assert res.blk.quant is b
        vals = res.blk._quant_vals
        assert vals == res.blk._quant_const_values() and all(_f32(vals[n]) != want[n] for n in names), "B must move every named constant"
        ws = torch.empty_like(res.ws).fill_(0xFF)
        grads = _alloc_grads(res.blk, fill=float("nan"))
        execute(res.blk, res.inp, res.saved, res.dy, grads, ws, **extra)
        torch.cuda.synchronize()
        sc = {k: float(v.item()) for k, v in res.blk.quant_scalars(ws).items()}
        for n in names:
            assert sc[n] == _f32(vals[n]), (n, sc[n], vals[n])
        blk_b = declare(res.dy, res.saved, res.inp, res.geom, quant=b)
        ws_b = _compile(blk_b).fill_(0xFF)
        grads_b = _alloc_grads(blk_b, fill=float("nan"))
        execute(blk_b, res.inp, res.saved, res.dy, grads_b, ws_b, **extra)
        torch.cuda.synchronize()
        for name, ten in grads_b.items():
            if ten is not None:
                assert torch.isfinite(ten.float()).all(), name
        _assert_grads_equal(grads, grads_b, "update_quant_scales(B) differs from a backward declared with B")
        assert torch.equal(res.blk.quant_scalars(ws)["descale_w_o"], blk_b.quant_scalars(ws_b)["descale_w_o"])
        assert _grads_differ(grads, res.grads), "B must move the gradients, or the pin proves nothing"
    finally:
        res.blk.update_quant_scales(res.spec)
    assert res.blk._quant_vals == res.blk._quant_const_values() and res.blk.quant is res.spec


@requires_rubin
@_FAMILY
def test_grad_scale_margin_reaches_every_quantize_launch(family):
    """A backward declared ``grad_scaling="current", grad_scale_margin_log2=2`` publishes ``scale_dy / scale_do / scale_dqkvg`` (fp8;
    ``scale_dy`` alone under MXFP8) EQUAL to ``grad_scale_from_amax(amax, 2)`` bitwise and NOT equal to the margin-0 value wherever the
    amax is positive -- so every quantize launch took the margin (the mixed recipe of a rebound module constant, margin 0 on dY / dO
    and 2 on dQKVG, fails here); the amax passes are margin-free (equal to the default block's); and a block declared with the default
    spelled out is bitwise the block declared without the attribute (gradients and scalar block)."""
    run = _backward_fp8 if family == "fp8" else _backward_mxfp8
    base = run(_GEOM, _B, _S)
    m2 = run(_GEOM, _B, _S, grad_scale_margin_log2=2)
    m0 = run(_GEOM, _B, _S, grad_scale_margin_log2=_API_CONST)
    names = ("dy", "do", "dqkvg") if family == "fp8" else ("dy",)
    assert m2.blk.grad_scale_margin_log2 == 2 and m2.blk._quant_dy.margin_log2 == 2 and m2.blk._quant_dy._quant.margin_log2 == 2
    if family == "fp8":
        assert m2.blk._quant_do.margin_log2 == 2 and m2.blk._quant_do._quant.margin_log2 == 2 and m2.blk._epilogue.margin_log2 == 2
    for n in names:
        amax = m2.scalars[f"amax_{n}"]
        assert amax > 0.0 and amax == base.scalars[f"amax_{n}"], (n, amax, base.scalars[f"amax_{n}"])
        scale = m2.scalars[f"scale_{n}"]
        print(f"{family}: amax_{n} = {amax:.6g}, scale_{n} = {scale:g} (margin 2), {base.scalars[f'scale_{n}']:g} (margin 0)")
        assert scale == _f32(_quantize.grad_scale_from_amax(amax, 2)), (n, scale, _quantize.grad_scale_from_amax(amax, 2))
        assert scale != _f32(_quantize.grad_scale_from_amax(amax, 0)), f"{n}: the quantize launch ignored the margin"
        assert base.scalars[f"scale_{n}"] == _f32(_quantize.grad_scale_from_amax(amax, 0))
        assert amax * scale * 4.0 <= 448.0 < amax * scale * 8.0, f"{n}: the product is not two octaves under the e4m3 maximum"
        assert m2.scalars[f"descale_{n}"] == _f32(1.0 / scale)
    # REPORTED, never asserted: a power-of-two margin shifts e4m3 EXPONENTS, so the gradients move only where an element falls into the
    # e4m3 subnormal range or underflows at the coarser scale -- data-dependent (the fp4 suite pins the same equivariance for dY)
    print(f"{family}: gradients at margin 2 {'differ from' if _grads_differ(m2.grads, base.grads) else 'equal'} the margin-0 block's")
    _assert_grads_equal(m0.grads, base.grads, "the default margin spelled out differs from the block declared without it")
    assert m0.scalars == base.scalars


# ---------------------------------------------------------------------------
# The training loop (gated_block_train): the toy decoder with the block as its attention layer
# ---------------------------------------------------------------------------


@pytest.fixture
def deterministic_cublas(monkeypatch):
    """The harness runs under ``torch.use_deterministic_algorithms(True)``, which refuses cuBLAS GEMMs unless ``CUBLAS_WORKSPACE_CONFIG``
    names a deterministic workspace; a process that already names one keeps it, any other gets ``:4096:8`` for the test."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in gbt.DETERMINISTIC_CUBLAS_CONFIGS:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", gbt.DETERMINISTIC_CUBLAS_CONFIGS[0])
    return os.environ["CUBLAS_WORKSPACE_CONFIG"]


def _train(arm, *, steps=_STEPS, **kw):
    """``steps`` steps of ``arm`` at the small geometry from seed 0 (``kw`` -> ``run_training``: ``replica``, ``name``, ``keep_grads_at``, ...)."""
    return gbt.run_training(arm, _SMOKE, steps=steps, seed=0, **kw)


def _finite(x) -> bool:
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    if isinstance(x, list):
        return all(_finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def _digested(row: dict) -> str:
    """The row as the digest sees it (minus the non-deterministic keys and the replica metadata), in the digest's canonical JSON --
    compared as TEXT, the way the digest compares it: a dict comparison would read the torch arm's documented ``NaN`` block gradient
    norms as unequal to themselves."""
    return gbt.canonical_json({k: v for k, v in row.items() if k not in gbt.NONDET_KEYS | gbt.ROW_META_KEYS})


def _numeric(row: dict) -> str:
    """The row minus the digest's version-keyed part too (``NUMERIC_COMPARE_IGNORE``), canonical JSON: what two recipes of one arm agree on."""
    return gbt.canonical_json({k: v for k, v in row.items() if k not in gbt.NONDET_KEYS | gbt.ROW_META_KEYS | gbt.NUMERIC_COMPARE_IGNORE})


def _bound_fraction(got: torch.Tensor, ref: torch.Tensor, rtol: float, atol_frac: float):
    """The accept suite's bound, read without asserting: ``(worst cell as a fraction of rtol * |ref| + atol_frac * max|ref|, cosine)``."""
    got64, ref64 = got.detach().double(), ref.detach().double()
    atol = atol_frac * ref64.abs().max().item()
    worst = ((got64 - ref64).abs() / (atol + rtol * ref64.abs())).max().item()
    cos = (got64.flatten() @ ref64.flatten() / (got64.norm() * ref64.norm())).item()
    return worst, cos


def _assert_rows_bitwise(a, b, what: str) -> None:
    assert a.manifest["run_digest"] == b.manifest["run_digest"], f"{what}: run_digest {a.manifest['run_digest'][:16]} vs {b.manifest['run_digest'][:16]}"
    assert len(a.rows) == len(b.rows) == _STEPS
    for ra, rb in zip(a.rows, b.rows):
        assert ra["row_digest"] == rb["row_digest"] and ra["grad_sha256"] == rb["grad_sha256"], f"{what}: step {ra['step']}"
        assert _digested(ra) == _digested(rb), f"{what}: step {ra['step']} differs on a digested key"
        assert "wall_ms" in ra and "wall_ms" in rb  # present, informational, outside the digest
    assert a.manifest["deterministic_algorithms"] is True and a.manifest["cublas_workspace_config"] in gbt.DETERMINISTIC_CUBLAS_CONFIGS


def test_the_training_loop_imports_only_from_the_test_tree_and_the_package():
    """Host: the harness core is a TRACKED helper of this test tree, so every module it imports is the standard library, torch, the
    ``cudnn`` package or a sibling of the tree (resolved by import, not by name) -- a helper reaching into an untracked directory
    imports in ONE checkout only, and a test that imports it would be collected green there while asserting nothing anywhere else.
    The file carries no absolute path, and it is a helper, not a collected test."""
    path = os.path.abspath(gbt.__file__)
    src = open(path, encoding="utf-8").read()
    test_tree = os.path.dirname(os.path.dirname(os.path.dirname(path)))  # <repo>/test/python
    assert os.path.basename(test_tree) == "python" and os.path.basename(os.path.dirname(test_tree)) == "test"
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    outside, absolute = [], []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("/"):
            absolute.append(node.value)  # an absolute POSIX path literal: a checkout- or host-specific location
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports in a sys.path helper"
            names = [node.module]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in stdlib or top in ("torch", "cudnn"):
                continue
            mod_file = os.path.abspath(importlib.import_module(name).__file__)
            if not mod_file.startswith(test_tree + os.sep):
                outside.append((name, mod_file))
    assert not outside, f"imports resolved outside the test tree: {outside}"
    assert not absolute, f"absolute path literals in the harness core: {absolute}"
    assert not os.path.basename(path).startswith("test_"), "the helper would be collected as a test module"


def test_training_loop_torch_control_two_runs_are_bitwise(deterministic_cublas):
    """Any CUDA device: the pure-torch control arm (the reference block under autograd) trained three steps twice from one seed digests
    equal -- every row, ``grad_sha256`` included -- and its twin driven through the ``GatedBlockFn`` plumbing (the reference block's
    output into the layer buffer, its autograd gradients into the gradient buffers) reproduces its losses, learning rates, token counts
    and total gradient norms bitwise: the trainer plumbing is exercised on hosts that cannot run the block."""
    a = _train(gbt.ARMS["torch-bf16"])
    b = _train(gbt.ARMS["torch-bf16"], replica=1)
    _assert_rows_bitwise(a, b, "torch control")
    fn = _train(gbt.ARMS["torch-bf16-fn"])
    for r, s in zip(a.rows, fn.rows):
        assert (r["loss"], r["lr"], r["tokens_seen"], r["grad_norm_total"]) == (s["loss"], s["lr"], s["tokens_seen"], s["grad_norm_total"]), r["step"]
        # the torch arm has no block ``dh`` buffer: its ``dh`` norm is the documented NaN; everything else is finite
        assert all(math.isnan(v) for v in r["grad_norms"]["dh"])
        assert _finite({k: v for k, v in r.items() if k != "grad_norms"}) and _finite({g: v for g, v in r["grad_norms"].items() if g != "dh"})
    print(f"torch control: losses {[round(r['loss'], 6) for r in a.rows]} run_digest {a.manifest['run_digest'][:16]} ({deterministic_cublas})")


@requires_rubin
def test_training_loop_bf16_matches_the_torch_reference_model(deterministic_cublas):
    """Three steps of the bf16 block model and of the torch reference model from ONE seed (same init, data and optimizer): the step-0
    fp32 master gradients of the block's ``W_qkvg`` / ``W_o`` of every layer within the bf16 backward bound of the accept suite
    (``_assert_grad_close``: rtol 2^-6, atol 2^-7 * max|ref|, cos >= 0.999 -- the numbers the single-step accept tests hold the backward
    to), the block's ``W_q_norm`` / ``W_k_norm`` gradients under the suite's cosine floor and a worst-cell ceiling of that bound
    (``_NORM_W_BOUND_CEIL``), the per-step relative loss difference under ``_LOSS_REL_BOUND``; the model's other parameters' gradients are
    printed, not asserted."""
    frost = _train(gbt.ARMS["bf16"], keep_grads_at=(0,))
    ref = _train(gbt.ARMS["torch-bf16"], keep_grads_at=(0,))
    rtol, atol_frac = _RTOL[torch.bfloat16], _ATOL_FRAC[torch.bfloat16]
    g_f, g_r = frost.kept_grads[0], ref.kept_grads[0]
    assert (
        set(g_f) == set(g_r) and len(g_f) == 8 * _SMOKE.n_layers + 2
    )  # per layer ln1 / w_qkvg / w_o / w_q_norm / w_k_norm / ln2 / mlp_w1 / mlp_w2, plus emb / lnf
    worst_block, worst_norm, reported = {}, {}, {}
    for name in sorted(g_f):
        fam = name.split(".")[0]
        if fam in ("w_qkvg", "w_o"):
            worst_block[name] = _assert_grad_close(g_f[name], g_r[name], f"step-0 master gradient {name}", rtol=rtol, atol_frac=atol_frac)
            continue
        worst, cos = _bound_fraction(g_f[name], g_r[name], rtol, atol_frac)
        print(f"step-0 master gradient {name}: worst cell {worst:.3f} of the bf16 bound, cos {cos:.6f}")
        if fam in ("w_q_norm", "w_k_norm"):
            worst_norm[name] = worst
            assert (
                cos >= _COS_MIN and worst <= _NORM_W_BOUND_CEIL
            ), f"{name}: worst cell {worst:.3f} of the bf16 bound (ceiling {_NORM_W_BOUND_CEIL}), cos {cos:.6f}"
        else:
            reported[name] = (worst, cos)
    rel = [abs(a["loss"] - b["loss"]) / abs(b["loss"]) for a, b in zip(frost.rows, ref.rows)]
    print(
        f"bf16 block vs torch: W_qkvg / W_o worst {max(worst_block.values()):.3f} of the bound; norm weights worst {max(worst_norm.values()):.3f} "
        f"(ceiling {_NORM_W_BOUND_CEIL}); reported {len(reported)} other parameters, worst {max(w for w, _c in reported.values()):.3f}; "
        f"losses {[round(r['loss'], 6) for r in frost.rows]} vs {[round(r['loss'], 6) for r in ref.rows]}, max relative difference {max(rel):.3e}"
    )
    assert max(rel) <= _LOSS_REL_BOUND, f"loss relative difference {max(rel):.3e} > {_LOSS_REL_BOUND:.0e}"


@requires_rubin
@pytest.mark.parametrize("arm_name", ["bf16", "mxfp8"])
def test_training_loop_three_steps_bf16_and_mxfp8_finite(arm_name, deterministic_cublas):
    """Three steps of the arm: every logged value finite, every per-layer list of length ``n_layers``; under MXFP8 the per-layer specs
    are live (``descale_w_o`` differs between the two layers -- each layer's ``W_o`` has its own amax), ``scale_o`` stays 1.0, no
    calibration pass runs, and the one gradient scale derived in-kernel keeps ``amax * scale <= 448``; the bf16 run with both
    performance-only backward knobs (``fuse_gate_bwd`` + ``fuse_wgrad_overlap``) is BITWISE the default-knob run on every numeric key
    (the knobs sit in the recipe, so the digests differ by construction)."""
    arm = gbt.ARMS[arm_name]
    res = _train(arm)
    L = _SMOKE.n_layers
    for r in res.rows:
        assert _finite(r), f"step {r['step']}: a non-finite logged value"
        assert all(len(v) == L for v in r["grad_norms"].values())
        assert all(v > 0.0 for vals in r["grad_norms"].values() for v in vals)
        assert r["calib_fwd"] is False and r["calib_bwd"] is False
    if arm_name == "mxfp8":
        for r in res.rows:
            assert len(r["spec"]["descale_w_o"]) == L and r["spec"]["descale_w_o"][0] != r["spec"]["descale_w_o"][1], r["spec"]
            assert r["spec"]["scale_o"] == [1.0] * L
            assert all(len(v) == L for v in r["quant_scalars"].values())
            for l in range(L):
                prod = r["quant_scalars"]["amax_dy"][l] * r["quant_scalars"]["scale_dy"][l]
                assert 0.0 < prod <= 448.0, (r["step"], l, prod)
        print(f"mxfp8: descale_w_o per step {[r['spec']['descale_w_o'] for r in res.rows]} losses {[round(r['loss'], 6) for r in res.rows]}")
        return
    knobs = _train(arm.with_(bwd_knobs=("fuse_gate_bwd", "fuse_wgrad_overlap")), name="bf16_knobs")
    for r, s in zip(res.rows, knobs.rows):
        assert r["recipe"] != s["recipe"] and r["row_digest"] != s["row_digest"]
        assert _numeric(r) == _numeric(s), f"step {r['step']}: the backward knobs moved a numeric key"
    print(f"bf16: losses {[round(r['loss'], 6) for r in res.rows]}; the knob run bitwise on every numeric key")


@requires_rubin
@pytest.mark.parametrize("arm_name", ["bf16", "mxfp8"])
def test_training_loop_two_runs_are_bitwise(arm_name, deterministic_cublas):
    """The deterministic-replay pin: two three-step runs of the arm from one seed (replica 0 and 1) agree on ``run_digest``, on every
    ``row_digest`` and ``grad_sha256`` and on every digested key; ``wall_ms`` is present and outside the digest; the harness turned
    ``torch.use_deterministic_algorithms`` on for the run and restored the process's setting."""
    prev = torch.are_deterministic_algorithms_enabled()
    a = _train(gbt.ARMS[arm_name])
    b = _train(gbt.ARMS[arm_name], replica=1)
    assert torch.are_deterministic_algorithms_enabled() == prev
    _assert_rows_bitwise(a, b, arm_name)
    assert a.rows[0]["replica"] == 0 and b.rows[0]["replica"] == 1
    print(f"{arm_name}: run_digest {a.manifest['run_digest']} x2, last loss {a.rows[-1]['loss']:.6f}")


@requires_rubin
@pytest.mark.parametrize("arm_name", ["fp8-recal1", "mxfp8"])
def test_training_loop_feedless_delayed_step0_is_bitwise_the_current_recipe(arm_name, deterministic_cublas):
    """``grad_scaling="delayed"`` without a scale feed bootstraps its gradient scales at step 0 from DISCARDED backward passes in
    dependency order (``bootstrap_rungs``: fp8 ``scale_dy`` -> ``scale_do`` -> ``scale_dp`` -> ``scale_dqkvg``, MXFP8 ``scale_dy``), each
    seeded by the kernels' own ``grad_scale_from_amax`` over that pass's published amax, so the LOGGED step 0 is BITWISE the ``current``
    recipe's step 0 (the kernel derives the same rule in-kernel) and the seeded scales are the published ones; the run completes with
    every block gradient of every layer finite and non-zero (a scale seeded from a flushed pass zeroed a layer's gradients exactly), and
    the lagged recipe's saturation is counted per step, layer and gradient (``grad_sat``), never asserted."""
    arm = gbt.ARMS[arm_name]
    L = _SMOKE.n_layers
    cur = _train(arm)
    dl = _train(arm.with_(grad_scaling="delayed"), name=f"{arm_name}_delayed")
    rungs = ["scale_dy", "scale_do", "scale_dp", "scale_dqkvg"] if arm.family == "fp8" else ["scale_dy"]
    assert dl.manifest["bootstrap_rungs"] == rungs and cur.manifest["bootstrap_rungs"] == (["scale_dp"] if arm.family == "fp8" else [])
    assert dl.rows[0]["calib_bwd"] is True and all(r["calib_bwd"] is False for r in dl.rows[1:])
    differ = [k for k in _STEP0_KEYS if gbt.canonical_json(cur.rows[0].get(k)) != gbt.canonical_json(dl.rows[0].get(k))]
    assert not differ, f"the feed-less delayed step 0 differs from the current recipe's on {differ}"
    for n in arm.grad_scale_names:
        assert dl.rows[0]["given_scales"][n] == dl.rows[0]["quant_scalars"][n], n  # the ladder's seed is what the kernel published
    assert "given_scales" not in cur.rows[0] and "grad_sat" not in cur.rows[0]
    for r in dl.rows:
        assert set(r["grad_sat"]) == set(arm.grad_scale_names) and all(len(v) == L for v in r["grad_sat"].values())
        for g, vals in r["grad_norms"].items():
            assert len(vals) == L and all(math.isfinite(v) and v > 0.0 for v in vals), (r["step"], g, vals)
    sat = {n: sum(r["grad_sat"][n][l] for r in dl.rows for l in range(L)) for n in arm.grad_scale_names}
    ratios = {
        g: [round(d["grad_norms"][g][l] / c["grad_norms"][g][l], 4) for d, c in zip(dl.rows, cur.rows) for l in range(L)] for g in cur.rows[0]["grad_norms"]
    }
    print(
        f"{arm_name}: ladder {rungs}; step 0 bitwise the current recipe's; grad_sat counts over {_STEPS} x {L} step-layers {sat} (reported); "
        f"delayed / current block-gradient norm ratios {ratios}"
    )


@requires_rubin
@_FAMILY
def test_update_quant_scales_reaches_a_captured_execute_of_a_warmed_up_block(family):
    """The two halves of the API's CUDA-graph statement.  (1) A graph that captured an execute of a WARMED-UP block (one eager execute
    on the capture stream first) replays at ``A`` bitwise, and after an EAGER ``update_quant_scales(B)`` between two replays the next
    replay is BITWISE the ``out`` (and the record) of a block DECLARED with ``B``: the scalars are read, not baked into the graph.  (2) A
    graph that captured the block's FIRST execute captures its one-time scalar write with the capture-time values: it replays at ``A``,
    and after the same eager ``update_quant_scales(B)`` the replay STAYS at ``A`` and re-writes the device scalars to ``A`` -- the
    documented limitation (warm up before capturing, or re-capture after a recalibration); outside the graph the caller's path still
    works (an eager update + execute runs at ``B``)."""
    r = _run_training_quant(_GEOM, _B, _S, family)
    a = r.spec
    b = _spec_b(a)
    blk_b, out_b = _fwd_block_with(r, b, training=True)
    ws_b = _compile(blk_b)
    saved_b = _record(r)
    _execute_fwd(r, blk_b, out_b, ws_b, saved=saved_b)
    torch.cuda.synchronize()
    assert not torch.equal(out_b, r.out), "B must move the output, or the pin proves nothing"
    want_a, want_b = _f32_values(r.blk._quant_dev_values(a)), _f32_values(r.blk._quant_dev_values(b))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # (1) the warmed-up block
    out_g, saved_g, ws_g = torch.empty_like(r.out), _record(r), torch.empty_like(r.ws)
    with torch.cuda.stream(stream):
        _execute_fwd(r, r.blk, out_g, ws_g, saved=saved_g)  # warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert torch.equal(out_g, r.out)
    graph, graph2 = torch.cuda.CUDAGraph(), None
    try:
        with torch.cuda.graph(graph, stream=stream):
            _execute_fwd(r, r.blk, out_g, ws_g, saved=saved_g)
        out_g.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_g, r.out), "the warmed-up capture does not replay the eager A result"
        r.blk.update_quant_scales(b)  # eager, on the default stream, between two replays
        torch.cuda.synchronize()
        assert _dev_values(r.blk) == want_b
        out_g.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            out_g, out_b
        ), f"the replay after update_quant_scales(B) is not the B block's out: max {(out_g.float() - out_b.float()).abs().max().item():.3e}"
        assert torch.equal(saved_g.o, saved_b.o) and torch.equal(saved_g.lse, saved_b.lse), "the record written by the replay differs from the B block's"
        assert _dev_values(r.blk) == want_b
        # (2) the FIRST execute captured: the documented limitation
        blk2, out2 = _fwd_block_with(r, a, training=True)
        ws2 = _compile(blk2)
        saved2 = _record(r)
        torch.cuda.synchronize()
        assert blk2._quant_dev_on_launch_stream is False
        graph2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph2, stream=stream):
            _execute_fwd(r, blk2, out2, ws2, saved=saved2)
        assert blk2._quant_dev_on_launch_stream is True, "the one-shot write was not part of the captured first execute"
        out2.fill_(float("nan"))
        torch.cuda.synchronize()
        graph2.replay()
        torch.cuda.synchronize()
        assert torch.equal(out2, r.out)
        blk2.update_quant_scales(b)
        torch.cuda.synchronize()
        assert _dev_values(blk2) == want_b
        out2.fill_(float("nan"))
        torch.cuda.synchronize()
        graph2.replay()
        torch.cuda.synchronize()
        assert torch.equal(out2, r.out), "a graph that captured the FIRST execute no longer replays its capture-time scales: the documented limitation moved"
        assert _dev_values(blk2) == want_a, "the replayed one-time write did not re-write the capture-time values"
        out3 = torch.empty_like(r.out)
        _execute_fwd(r, blk2, out3, ws2, saved=_record(r))
        torch.cuda.synchronize()
        assert torch.equal(out3, r.out)  # an eager execute right after that replay reads the replayed A scalars
        blk2.update_quant_scales(b)
        _execute_fwd(r, blk2, out3, ws2, saved=_record(r))
        torch.cuda.synchronize()
        assert torch.equal(out3, out_b)  # and the caller's path outside the graph still reaches B
    finally:
        # test-owned graphs are reset here, assertion or not: a graph collected from a reference cycle inside a LATER test's
        # capture would invalidate that capture (the teardown hook flags an unreset graph)
        graph.reset()
        if graph2 is not None:
            graph2.reset()
