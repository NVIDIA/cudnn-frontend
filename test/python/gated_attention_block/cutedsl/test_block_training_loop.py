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
* stream ordering: an update + execute on a side stream while the default stream is parked equals the eager result, and the FIRST
  execute of a compiled block writes the scalars on its own launch stream (compile-time fills poisoned on purpose);
* the backward API changes the prologue's constants: ``quant_scalars()`` reads ``B``'s values and the gradients equal a backward
  DECLARED with ``B``, bitwise;
* the margin reaches every quantize launch: ``scale_dy / scale_do / scale_dqkvg`` (``scale_dy`` alone under MXFP8) are
  ``grad_scale_from_amax(amax, 2)`` bitwise and NOT the margin-0 value, and the default-margin block is bitwise a block declared
  without the attribute.

Accept tests are ``requires_rubin`` (the block targets SM107 only); the refusal tests carry no marker and run on any CUDA device.
The training-loop tests over the toy decoder join this module next to the pins above.
"""

import dataclasses
import inspect
import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import (  # noqa: E402
    GatedAttentionBlockBwd,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    MxQuantSpec,
    QuantSpec,
    gated_attention_block_backward,
)
from cudnn.gated_attention_block import api_bwd as _api_bwd  # noqa: E402
from cudnn.gated_attention_block.api import Fp4Format  # noqa: E402
from cudnn.gated_attention_block.kernels import quantize as _quantize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_stream_probe import park_the_default_stream  # noqa: E402
from test_block_backward import _COMMON, _alloc_grads, _declare_bwd  # noqa: E402
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
    passed as ``current_stream`` -- and the execute that follows run on the side stream, the workspace is zeroed right after.  A write
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
    with torch.cuda.stream(side):
        cs = None if how == "ambient" else cuda_drv.CUstream(side.cuda_stream)
        r.blk.update_quant_scales(b, current_stream=cs)
        _execute_fwd(r, r.blk, out2, ws2, saved=saved2, current_stream=cs)
        ws2.zero_()
    torch.cuda.synchronize()
    assert torch.equal(out2, out_b), f"the update or a stage escaped the caller's stream ({how})"
    assert torch.equal(saved2.o, saved_b.o) and torch.equal(saved2.lse, saved_b.lse), f"the record written on the side stream differs ({how})"


@requires_rubin
@_FAMILY
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_first_execute_writes_the_scales_on_its_launch_stream(family, how, tmp_path):
    """``compile()`` fills the device scalars on whatever stream is ambient then; the FIRST execute writes their VALUES again on ITS
    launch stream, so a block compiled on one stream and first executed on another reads what its execution stream wrote.  Pinned
    without a timing window: the compile-time values are POISONED to NaN (synchronously) after ``compile()`` and the first execute runs
    on a side stream (ambient, or explicit) -- its output is bitwise the synchronised run's and the scalars read back as ``spec``'s.
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
    cs = None if how == "ambient" else cuda_drv.CUstream(side.cuda_stream)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.cuda.stream(side):
            _execute_fwd(ref, blk, out, ws, saved=saved, current_stream=cs)
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
