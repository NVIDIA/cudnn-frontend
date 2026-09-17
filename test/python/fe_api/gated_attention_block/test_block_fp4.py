# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's fp4 modes -- the WEIGHT arms (``MxQuantSpec.w_qkvg_dtype = torch.float4_e2m1fn_x2``,
first half of this module) and the OUTPUT arms (``MxQuantSpec.o_fp4 = Fp4Format.NVFP4 | MXFP4``, second half).

An MXFP4 ``W_qkvg`` (E2M1 codes, two per byte, per-32-block E8M0 scales in the SAME F8_128x4 blob
the MXFP8 weight carries) against the e4m3 ``h`` rides the FROST block-scale catalog's MIXED row
(``fp8_e4m3 x fp4_e2m1``, E8M0 / 32) through the UNFUSED MXFP8 pipeline's stage (1) -- the same
nine stages, the same workspace, byte for byte (plan config row 7).  The fused MXFP8 projection
fork is rendered for an e4m3 B, so ``w_qkvg_dtype=fp4 + fuse_norm_rope`` is a typed decline
(row 11), feature-detected on ``NormRopeFusionParams.weight_fp4`` so the pin INVERTS when the arm
lands.

What is pinned here, and where the GPU matters:

* the appended ``MxQuantSpec`` field, its default and its typed declines (no GPU);
* the per-weight dtype / shape contract at ``check_support`` -- an e2m1 weight is checked against its
  STORAGE shape ``[n_qkvg, d_model // 2]`` (``APIBase._make_tensor_desc`` reports the LOGICAL shape of a
  packed-fp4 tensor; ``test_proj_gemm_fp4.py`` pins that fact and ``_fp4_storage_shape`` inverts it),
  a ``uint8`` blob declines with the ``.view(torch.float4_e2m1fn_x2)`` hint (torch 2.13 can view,
  not cast), an fp4 weight under a ``QuantSpec`` / bf16 block names ``MxQuantSpec`` (any CUDA device);
* the generalised ``_check_sf_blob(block=, sf_dtypes=)`` (no GPU);
* the ``GatedAttentionBlockFwd.execute`` body dispatching the nine frozen stages once each with an
  fp4 weight (monkeypatched trace; any CUDA device);
* numerics on Rubin: the e4m3 spelling is BITWISE the MXFP8 suite's run (the field's default changes
  nothing), and the MXFP4 weight matches the fake-quant oracle at the MXFP8 suite's shapes with a
  sentinel-filled output plus the format floor ``cos(kernel, bf16 ref) >= cos(oracle, bf16 ref) - 0.005``.

The oracle for an fp4 weight: ``gated_attention_block_mxfp8_reference`` dequantizes ``W_qkvg`` THROUGH
the blob it is handed, and every E2M1 value ``{0, .5, 1, 1.5, 2, 3, 4, 6}`` is EXACTLY an e4m3 value, so the
oracle is fed the fp4 weight's values as e4m3 "shadow" codes with the fp4 blob -- ``deq(shadow, blob)
== fp4_dequant_rowwise_2d(packed, blob)`` is asserted on every run, so the shadow is the plan's fp4
dequant path, not an approximation of it.

The O arms (plan config rows 8-10) pin, in the same three tiers: ``Fp4Format`` and the appended ``o_fp4`` field
with its unit-scale rules; the both-halves rule (``o_fp4`` <-> ``sample_w_o_sf`` / ``w_o_sf``); the per-weight
contract for an e2m1 ``w_o`` and its blob at the FORMAT's block and scale dtype; the stage list (``quantize_fp4_o``
in ``quantize_o``'s place, 9 launches unfused / 4 fused with the gated MXFP8 SDPA writing bf16 O); the workspace
(``o8`` not reserved, ``o4`` / ``sf_o`` appended, ``sf_o`` sized by the GEMM's ``sf_blob_bytes`` -- and the frozen
snapshot of EVERY pre-existing layout, test and 397B geometry); the execute body's fp4 hand-off (``o4`` as A,
``sf_a=sf_o``, ``sf_w=w_o_sf``, no alpha); and on Rubin the fake-quant oracle (``gated_attention_block_mxfp8_reference
(..., o_fp4=)``) with the format floor, the dead entry ``out[1] == 0`` EXACTLY, ragged padding, a bitwise second
execute and the 9 / 4 launch counts.

The fixtures / helpers of the MXFP8 suite are reused by importing that module by NAME (the unique-
basename rule); nothing here re-derives a shape or a stage list.
"""

import dataclasses
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import Fp4Format, GatedAttentionBlockFwd, GatedAttentionBlockGeometry, MxQuantSpec, QuantSpec  # noqa: E402
from cudnn.gated_attention_block.api import _FusedQkvProjection, _check_sf_blob  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes  # noqa: E402

E4M3 = torch.float8_e4m3fn
_E8M0 = getattr(torch, "float8_e8m0fnu", None)
_FP4 = getattr(torch, "float4_e2m1fn_x2", None)
if _FP4 is None or _E8M0 is None:
    # The oracle module reads both dtypes at import, so a torch without them must skip HERE (collection), not error.
    pytest.skip(f"torch {torch.__version__} has no float4_e2m1fn_x2 / float8_e8m0fnu", allow_module_level=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_block_mxfp8 as mx_suite  # noqa: E402  -- the MXFP8 suite's fixtures, by module name
from gated_block_reference import (  # noqa: E402
    RefGeometry,
    fp4_dequant_rowwise_2d,
    fp4_quantize_rowwise_2d,
    gated_attention_block_mxfp8_reference,
    gated_attention_block_reference,
    make_inputs,
    mx_dequant_rowwise_2d,
    mx_swizzle_sf_rowwise_padded,
    mxfp8_calibrated_scale_o,
    quantize_block_inputs_mxfp8,
    unpack_e2m1,
)

_GEOM, _FUSED, _SENTINEL, _MX_STAGES, _ABSENT = mx_suite._GEOM, mx_suite._FUSED, mx_suite._SENTINEL, mx_suite._MX_STAGES, mx_suite._ABSENT
_cc, requires_rubin = mx_suite._cc, mx_suite.requires_rubin
requires_fp4 = pytest.mark.skipif(
    _FP4 is None or _E8M0 is None, reason="this torch has no float4_e2m1fn_x2 / float8_e8m0fnu"
)  # always true past the module skip; kept as the per-test statement
MXFP4_BLOCK = 32  # the mixed row's scale block == the MX block (the weight blob is the MXFP8 one, unchanged)


def _fp4_w_spec(**kw) -> MxQuantSpec:
    kw.setdefault("descale_w_o", mx_suite._SPEC.descale_w_o)
    return MxQuantSpec(w_qkvg_dtype=_FP4, **kw)


def _decl_tensors_fp4w(geom_kw=_GEOM, batch=1, seq_len=256, *, w_storage: str = "fp4"):
    """The MXFP8 suite's declaration tensors with ``w_qkvg`` swapped for an e2m1 weight: ``"fp4"`` = the contract
    (storage ``[N, K/2]`` viewed ``float4_e2m1fn_x2``), ``"uint8"`` = the same bytes left as uint8, ``"logical"`` =
    an fp4x2 tensor allocated at the LOGICAL ``[N, K]`` (twice the data)."""
    geom, args, hsf, wsf = mx_suite._decl_tensors(geom_kw, batch, seq_len)
    n, k = geom.n_qkvg, geom.d_model
    u8 = torch.zeros(n, k // 2, dtype=torch.uint8, device="cuda")
    w = {"fp4": u8.view(_FP4), "uint8": u8, "logical": torch.zeros(n, k, dtype=torch.uint8, device="cuda").view(_FP4)}[w_storage]
    args = list(args)
    args[1] = w
    return geom, tuple(args), hsf, wsf


def _decl_block_fp4w(geom_kw=_GEOM, *, w_storage: str = "fp4", **kw) -> GatedAttentionBlockFwd:
    geom, args, hsf, wsf = _decl_tensors_fp4w(geom_kw, w_storage=w_storage)
    kw.setdefault("quant", _fp4_w_spec())
    kw.setdefault("sample_h_sf", hsf)
    kw.setdefault("sample_w_qkvg_sf", wsf)
    return GatedAttentionBlockFwd(*args, geom, **{k: v for k, v in kw.items() if v is not _ABSENT})


def _fork_has_fp4_arm() -> bool:
    return _FusedQkvProjection._fork_supports_field(_FusedQkvProjection._WEIGHT_FP4_FIELD)


# ---------------------------------------------------------------------------
# MxQuantSpec.w_qkvg_dtype (no GPU)
# ---------------------------------------------------------------------------


def test_w_qkvg_dtype_is_appended_after_the_mxfp8_fields_with_an_e4m3_default():
    """Append-only: the field follows the four MXFP8 fields (``o_fp4``, the O slice's field, comes after it), defaults to
    e4m3, and every existing spelling (positional included) is unchanged -- the MXFP8 suite's ``_SPEC`` is the same record
    with ``w_qkvg_fp4 == False``."""
    names = [f.name for f in dataclasses.fields(MxQuantSpec)]
    assert names[4] == "w_qkvg_dtype" and names[:4] == ["descale_w_o", "scale_o", "dtype", "block_size"]
    assert MxQuantSpec(descale_w_o=0.03).w_qkvg_dtype == E4M3 and not MxQuantSpec(descale_w_o=0.03).w_qkvg_fp4
    assert MxQuantSpec(0.03, 1.0, E4M3, 32) == MxQuantSpec(descale_w_o=0.03) == mx_suite._SPEC
    mx_suite._SPEC.validate()
    mx_suite._SPEC.validate(fused=True)


@requires_fp4
def test_w_qkvg_dtype_accepts_e2m1_and_declines_every_other_dtype_typed():
    """fp4x2 validates (unfused AND fused -- the fused decline is the FORK's, at check_support, not the spec's);
    e5m2 / bf16 / uint8 are ``NotImplementedError`` naming the field and the mixed catalog row; the uint8 message
    carries the ``.view(torch.float4_e2m1fn_x2)`` spelling."""
    spec = _fp4_w_spec()
    assert spec.w_qkvg_fp4 and spec.w_qkvg_dtype == _FP4 and spec.dtype == E4M3 and spec.block_size == 32
    spec.validate()
    spec.validate(fused=True)
    _fp4_w_spec(scale_o=2.0).validate()  # unfused scale_o stays free with an fp4 weight
    for bad in (torch.float8_e5m2, torch.bfloat16, torch.uint8):
        with pytest.raises(NotImplementedError, match="w_qkvg_dtype") as ei:
            MxQuantSpec(descale_w_o=1.0, w_qkvg_dtype=bad).validate()
        assert "mixed row" in str(ei.value) and "fp4_e2m1" in str(ei.value)
        if bad is torch.uint8:
            assert ".view(torch.float4_e2m1fn_x2)" in str(ei.value)
    # the other rules still fire first / unchanged with an fp4 weight
    with pytest.raises(NotImplementedError, match="e4m3"):
        MxQuantSpec(descale_w_o=1.0, dtype=torch.float8_e5m2, w_qkvg_dtype=_FP4).validate()
    with pytest.raises(ValueError, match="block_size"):
        MxQuantSpec(descale_w_o=1.0, block_size=16, w_qkvg_dtype=_FP4).validate()
    with pytest.raises(NotImplementedError, match="scale_o"):
        _fp4_w_spec(scale_o=2.0).validate(fused=True)


@requires_fp4
def test_fp4_weight_declaration_is_declined_at_declaration_only_where_the_spec_is():
    """The spec's ``validate`` runs in the constructor, so a bad ``w_qkvg_dtype`` is refused at DECLARATION
    (before any descriptor is read), while the three-halves / both-halves rules are untouched by the field."""
    with pytest.raises(NotImplementedError, match="w_qkvg_dtype"):
        _decl_block_fp4w(quant=MxQuantSpec(descale_w_o=1.0, w_qkvg_dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="three halves"):
        _decl_block_fp4w(sample_w_qkvg_sf=_ABSENT)
    with pytest.raises(NotImplementedError, match="inference-only"):
        _decl_block_fp4w(save_for_backward=True, inplace_qkv=False)


# ---------------------------------------------------------------------------
# check_support: per-weight dtype / STORAGE shape (any CUDA device)
# ---------------------------------------------------------------------------


@requires_fp4
def test_fp4_weight_declaration_accepts_the_storage_shape_and_wires_the_mixed_gemm():
    """``[n_qkvg, d_model // 2]`` fp4x2 passes the declaration checks; stage (1) is the block-scale GEMM on
    A e4m3 x W e2m1 at block 32 (the mixed row; ``_Projection.check_support`` accepts the pairing on any device);
    the stage list and the workspace are the MXFP8 block's, byte for byte (config row 7: no new slot)."""
    blk = _decl_block_fp4w()
    blk._check_declaration()
    assert blk.mxfp8 and not blk.mxfp8_fused and blk.quant.w_qkvg_fp4
    p = blk._proj
    assert p.block_scale and not p.alpha and p.dtype == E4M3 and p.w_dtype == _FP4 and p.block_size == MXFP4_BLOCK and p.sf_dtype is None
    p.check_support()  # the pairing table serves (e4m3, e2m1, E8M0, 32)
    assert [s.name for s in blk._stages] == _MX_STAGES
    assert blk._expected_weight_dtypes() == {"w_qkvg": _FP4, "w_o": E4M3}
    ref = mx_suite._decl_block()  # the e4m3 spelling
    assert ref._expected_weight_dtypes() == {"w_qkvg": E4M3, "w_o": E4M3}
    assert blk._layout() == ref._layout(), "an fp4 weight adds no workspace slot and moves no offset"
    assert blk._layout().total_bytes == ref._layout().total_bytes  # get_workspace_size() is arch-gated (it runs the stages' checks)
    # w_qkvg_sf is UNCHANGED: the same E8M0 / 32 blob over n_qkvg x d_model the MXFP8 weight carries
    assert blk._descs["w_qkvg_sf"].shape == ref._descs["w_qkvg_sf"].shape == (sf_blob_bytes(blk.geom.n_qkvg, blk.geom.d_model, MXFP4_BLOCK),)
    blk._check_mxfp8_declaration()


@requires_fp4
def test_fp4_storage_shape_inverts_what_the_descriptor_reports():
    """``_make_tensor_desc`` reports the LOGICAL ``(N, K)`` of a ``[N, K/2]`` fp4x2 tensor (the tier-1 pin in
    ``test_proj_gemm_fp4.py``); ``_fp4_storage_shape`` gives the caller's ``(N, K/2)`` back, for rank 2 and for a
    unit leading dim -- the two spellings the block's shape check speaks in."""
    blk = _decl_block_fp4w()
    n, k = blk.geom.n_qkvg, blk.geom.d_model
    d = blk._descs["w_qkvg"]
    assert tuple(d.shape) == (n, k) and d.dtype == _FP4 and blk._is_fp4x2(d)
    assert blk._fp4_storage_shape(d) == (n, k // 2)
    d3 = blk._make_tensor_desc(torch.zeros(1, n, k // 2, dtype=torch.uint8, device="cuda").view(_FP4), name="w3")
    assert tuple(d3.shape) == (1, n, k) and blk._fp4_storage_shape(d3) == (1, n, k // 2)


@requires_fp4
@pytest.mark.parametrize(
    "case",
    ["uint8_storage", "logical_shape", "e4m3_weight_under_fp4_spec", "fp4_weight_under_default_spec", "fp4_weight_under_quantspec", "fp4_weight_bf16_block"],
)
def test_fp4_weight_declaration_rejects_wrong_dtype_or_shape_typed(case):
    """Every wrong spelling is a ``ValueError`` at ``check_support`` (the declaration part -- device-agnostic), never a
    silent reinterpretation: a uint8 blob names the ``.view`` fix; the LOGICAL ``[N, K]`` fp4 tensor names the storage
    shape; an e4m3 weight under an fp4 spec, and an fp4 weight under a default (e4m3) spec / a ``QuantSpec`` / a bf16
    block, each name ``MxQuantSpec.w_qkvg_dtype``.  Only the LAST two have no MXFP8 blobs -- the field is unrepresentable
    there, so the dtype loop is the fence (plan section 2)."""
    if case == "uint8_storage":
        blk = _decl_block_fp4w(w_storage="uint8")
        with pytest.raises(ValueError, match=r"\.view\(torch\.float4_e2m1fn_x2\)") as ei:
            blk._check_declaration()
        assert "w_qkvg" in str(ei.value) and "uint8" in str(ei.value)
    elif case == "logical_shape":
        blk = _decl_block_fp4w(w_storage="logical")
        with pytest.raises(ValueError, match="LOGICAL") as ei:
            blk._check_declaration()
        n, k = blk.geom.n_qkvg, blk.geom.d_model
        assert f"[{n}, {k} // 2 = {k // 2}]" in str(ei.value) and f"storage ({n}, {k})" in str(ei.value)
    elif case == "e4m3_weight_under_fp4_spec":
        blk = mx_suite._decl_block(quant=_fp4_w_spec())  # the MXFP8 suite's e4m3 W with an fp4-naming spec
        with pytest.raises(ValueError, match="w_qkvg must be torch.float4_e2m1fn_x2") as ei:
            blk._check_declaration()
        assert "MxQuantSpec.w_qkvg_dtype" in str(ei.value)
    elif case == "fp4_weight_under_default_spec":
        blk = _decl_block_fp4w(quant=mx_suite._SPEC)
        with pytest.raises(ValueError, match=r"MxQuantSpec\(w_qkvg_dtype=torch\.float4_e2m1fn_x2\)"):
            blk._check_declaration()
    elif case == "fp4_weight_under_quantspec":
        geom, args, _, _ = _decl_tensors_fp4w()
        blk = GatedAttentionBlockFwd(*args, geom, quant=mx_suite._PT_SPEC)  # e4m3 h, per-tensor spec, fp4 W, no blobs
        assert not blk.mxfp8
        with pytest.raises(ValueError, match="MxQuantSpec") as ei:
            blk._check_declaration()
        assert "QuantSpec / bf16 block has no GEMM row" in str(ei.value)
    else:
        geom, args, _, _ = _decl_tensors_fp4w()
        bf16 = [a.to(torch.bfloat16) if a.dtype == E4M3 else a for a in args]  # h / w_o bf16, W_qkvg stays fp4
        blk = GatedAttentionBlockFwd(*bf16, geom)
        assert blk.quant is None
        with pytest.raises(ValueError, match="MxQuantSpec"):
            blk._check_declaration()
    # the healthy spelling of the same block still passes, so each decline above is the ONE wrong thing
    _decl_block_fp4w()._check_declaration()


@requires_fp4
def test_fp4_w_o_under_an_fp4_weight_alone_names_the_output_mode():
    """``w_o`` expects h's dtype under an fp4 ``W_qkvg`` alone (the fp4 out projection is the O mode's, ``o_fp4``): an
    e2m1 ``w_o`` is a ValueError that names ``MxQuantSpec(o_fp4=...)`` + ``sample_w_o_sf``, not a shape mismatch."""
    geom, args, hsf, wsf = _decl_tensors_fp4w()
    args = list(args)
    args[6] = torch.zeros(geom.d_model, geom.h_q * geom.d_head // 2, dtype=torch.uint8, device="cuda").view(_FP4)
    blk = GatedAttentionBlockFwd(*args, geom, quant=_fp4_w_spec(), sample_h_sf=hsf, sample_w_qkvg_sf=wsf)
    with pytest.raises(ValueError, match="w_o is torch.float4_e2m1fn_x2") as ei:
        blk._check_declaration()
    assert "fp4 O" in str(ei.value) and "MxQuantSpec(o_fp4=Fp4Format.NVFP4 | MXFP4) plus sample_w_o_sf" in str(ei.value)


def test_check_declaration_is_the_pre_stage_half_of_check_support():
    """``check_support`` == ``_check_declaration`` + every stage's ``check_support``: the MXFP8 suite's block passes the
    declaration half on any device, and the full call reaches the stages (Rubin accepts; elsewhere the SDPA stage's own
    arch gate is what declines -- typed)."""
    blk = mx_suite._decl_block()
    blk._check_declaration()
    if _cc() == mx_suite._SM107:
        assert blk.check_support() is True and blk._is_supported
    else:
        with pytest.raises(NotImplementedError, match="SM107"):
            blk.check_support()
        assert not getattr(blk, "_is_supported", False)


# ---------------------------------------------------------------------------
# fused fork: row 11 typed decline, feature-detected
# ---------------------------------------------------------------------------


@requires_fp4
def test_fp4_weight_with_fuse_norm_rope_declines_typed_until_the_fork_arm_lands():
    """Config row 11: ``w_qkvg_dtype=fp4 + fuse_norm_rope`` is a ``NotImplementedError`` from the fused projection's
    ``check_support`` (and its ``params()`` -- no e4m3-B artifact can be rendered for an e2m1 weight), naming the e4m3 B
    and the unfused pipeline.  Feature-detected on ``NormRopeFusionParams.weight_fp4``, so this test INVERTS the day
    the fork arm lands: then the params carry the field and the fork's own gates take over."""
    blk = _decl_block_fp4w(**_FUSED)
    assert blk.mxfp8_fused and blk._proj.mxfp8 and blk._proj.quant.w_qkvg_fp4
    if not _fork_has_fp4_arm():
        with pytest.raises(NotImplementedError, match="e4m3 B") as ei:
            blk.check_support()
        msg = str(ei.value)
        assert "unfused" in msg and "weight_fp4" in msg and "float4_e2m1fn_x2" in msg
        with pytest.raises(NotImplementedError, match="e4m3 B"):
            blk._proj.params()
        with pytest.raises(NotImplementedError, match="e4m3 B"):
            blk._proj.compile()
    else:  # pragma: no cover -- the fork's e2m1-B arm has landed
        assert "weight_fp4" in str(blk._proj.params())
        if _cc() == mx_suite._SM107:
            blk._proj.check_support()
    # the e4m3 spelling of the SAME fused block never consults the fp4 arm (byte-identical params to the MXFP8 suite)
    e4 = mx_suite._decl_block(**_FUSED)
    assert "weight_fp4" not in str(e4._proj.params()) and "quant_mxfp8" in str(e4._proj.params())


# ---------------------------------------------------------------------------
# _check_sf_blob(block=, sf_dtypes=) (no GPU beyond a CUDA pointer)
# ---------------------------------------------------------------------------


@requires_fp4
def test_check_sf_blob_generalises_the_block_and_the_scale_dtypes():
    """Defaults = the MXFP8 contract (block 32, uint8 | e8m0) -- unchanged for every existing caller.  ``block=16,
    sf_dtypes=(uint8, e4m3)`` is the NVFP4 blob: its byte count follows ``sf_blob_bytes(rows, k, 16)``, an e4m3 view
    is accepted, an e8m0 view is refused (the declared scale dtype selects the MMA's scale format), and the size
    message names the block."""
    rows, k = 1000, 512
    dev = "cuda"
    b32 = torch.zeros(sf_blob_bytes(rows, k), dtype=torch.uint8, device=dev)
    _check_sf_blob(b32, "sf", rows, k)
    _check_sf_blob(b32.view(_E8M0), "sf", rows, k)
    with pytest.raises(ValueError, match="uint8 or torch.float8_e8m0fnu"):
        _check_sf_blob(b32.view(torch.int8), "sf", rows, k)
    with pytest.raises(ValueError, match="at block 32"):
        _check_sf_blob(b32[:-16], "sf", rows, k)
    b16 = torch.zeros(sf_blob_bytes(rows, k, 16), dtype=torch.uint8, device=dev)
    assert b16.numel() == 2 * b32.numel() != b32.numel()
    nv = dict(block=16, sf_dtypes=(torch.uint8, E4M3))
    _check_sf_blob(b16, "w_o_sf", rows, k, **nv)
    _check_sf_blob(b16.view(E4M3), "w_o_sf", rows, k, **nv)
    with pytest.raises(ValueError, match="uint8 or torch.float8_e4m3fn"):
        _check_sf_blob(b16.view(_E8M0), "w_o_sf", rows, k, **nv)
    with pytest.raises(ValueError, match="at block 16") as ei:
        _check_sf_blob(b32, "w_o_sf", rows, k, **nv)  # the 32-block byte count is HALF the 16-block one
    assert f"{sf_blob_bytes(rows, k, 16)}" in str(ei.value) and "K/16 blocks" in str(ei.value)
    with pytest.raises(ValueError, match="contiguous"):
        _check_sf_blob(torch.zeros(2 * b16.numel(), dtype=torch.uint8, device=dev)[::2], "w_o_sf", rows, k, **nv)


# ---------------------------------------------------------------------------
# execute body (monkeypatched trace; any CUDA device)
# ---------------------------------------------------------------------------


def _dispatch_trace_fp4w(blk, monkeypatch):
    """``mx_suite._dispatch_trace`` with the fp4 WEIGHT as the execute argument (that helper builds its arguments from
    the MXFP8 suite's e4m3 ``w_qkvg``, so the fp4 tensor would never enter ``execute``).  Returns
    ``(calls, w, proj_args)``: the ``(stage, method)`` trace, the fp4 weight handed in, and the positional arguments
    the projection stage's recorder received."""
    lay = blk._layout()
    blk._ws = lay
    blk._is_supported = True
    monkeypatch.setattr(blk, "get_workspace_size", lambda: lay.total_bytes)
    q = blk.quant
    blk._quant_dev = dict(
        alpha_o=torch.full((1,), float(q.alpha_o), dtype=torch.float32, device="cuda"),
        scale_o=torch.full((1,), float(q.scale_o), dtype=torch.float32, device="cuda"),
    )
    calls, proj_args = [], []
    for st in blk._stages:
        for meth in ("execute", "execute_mxfp8", "execute_fp8"):
            if hasattr(st, meth):

                def _rec(*a, _n=st.name, _m=meth, **k):
                    calls.append((_n, _m))
                    if _n == _MX_STAGES[0]:
                        proj_args.append(a)

                monkeypatch.setattr(st, meth, _rec)
    _, args, hsf, wsf = _decl_tensors_fp4w(batch=blk.batch, seq_len=blk.seq_len)
    ws = torch.zeros(lay.total_bytes, dtype=torch.uint8, device="cuda")
    blk.execute(*args, ws, h_sf=hsf, w_qkvg_sf=wsf)
    return calls, args[1], proj_args


@requires_fp4
def test_fp4_weight_execute_dispatches_the_nine_mxfp8_stages_once_in_order(monkeypatch):
    """Row 7 is the MXFP8 execute body unchanged: nine stage calls, pipeline order, plain ``execute`` each, and the
    execute body carries the fp4 weight tensor THROUGH to the projection stage unchecked -- ``execute`` performs no
    declaration-vs-argument dtype check of its own, so the only execute-time guard on the weight is
    ``run_proj_gemm``'s operand check (``test_proj_gemm_fp4.py``); the recorder here stands in for that stage."""
    blk = _decl_block_fp4w()
    calls, w, proj_args = _dispatch_trace_fp4w(blk, monkeypatch)
    assert [n for n, _ in calls] == _MX_STAGES and all(m == "execute" for _, m in calls)
    assert w.dtype == _FP4 and len(proj_args) == 1
    got = proj_args[0][1]  # execute(h, w_qkvg, proj, ws, ...): the weight is the second positional argument
    assert got.dtype == _FP4 and got.data_ptr() == w.data_ptr() and tuple(got.shape) == tuple(w.shape)


# ---------------------------------------------------------------------------
# Numerics -- Rubin
# ---------------------------------------------------------------------------


def _fp4w_inputs(geom_kw, batch, seq_len):
    """The MXFP8 suite's quantized inputs with ``W_qkvg`` re-quantized to MXFP4 (E2M1 codes packed ``[N, K/2]``,
    viewed ``float4_e2m1fn_x2``; the E8M0 / 32 blob in the SAME F8_128x4 layout).  Returns ``(inp, kern, orac, desc)``:
    ``kern`` is what the block gets, ``orac`` the same dict with the weight's values as e4m3 SHADOW codes for the
    oracle -- every E2M1 value is an e4m3 value exactly, so ``deq(shadow, blob) == fp4_dequant(packed, blob)``
    (asserted): the oracle dequantizes exactly the bytes the GEMM reads."""
    inp = make_inputs(RefGeometry(**geom_kw), batch=batch, seq_len=seq_len, dtype=torch.bfloat16)
    mx, desc = quantize_block_inputs_mxfp8(inp)
    packed, e = fp4_quantize_rowwise_2d(inp["w_qkvg"].float(), "mxfp4")
    blob = mx_swizzle_sf_rowwise_padded(e, block=MXFP4_BLOCK)
    assert blob.numel() == sf_blob_bytes(*inp["w_qkvg"].shape, MXFP4_BLOCK) == mx["w_qkvg_sf"].numel()
    values = unpack_e2m1(packed)
    shadow = values.to(E4M3)
    assert torch.equal(shadow.float(), values), "E2M1 values are e4m3-exact by construction"
    assert torch.equal(mx_dequant_rowwise_2d(shadow, blob), fp4_dequant_rowwise_2d(packed, blob, "mxfp4"))
    kern = dict(mx, w_qkvg=packed.view(_FP4), w_qkvg_sf=blob)
    orac = dict(mx, w_qkvg=shadow, w_qkvg_sf=blob)
    return inp, kern, orac, desc


def _run_fp4w_block(geom_kw, batch, seq_len, *, w_fp4: bool, seq_lens=None, sentinel: bool = False, **blk_kw):
    """Declare / check / compile / execute the MXFP8 block with ``w_qkvg_dtype`` spelled explicitly (e4m3 or fp4);
    return ``(out, oracle, bf16_ref, blk, kern, spec)``.  Mirrors ``test_block_mxfp8._run_mx_block``."""
    geom = GatedAttentionBlockGeometry(**geom_kw)
    rg = mx_suite._ref_geom(geom)
    if w_fp4:
        inp, kern, orac, desc = _fp4w_inputs(geom_kw, batch, seq_len)
    else:
        inp, kern, desc = mx_suite._mx_inputs(geom_kw, batch, seq_len)
        orac = kern
    scale_o = mxfp8_calibrated_scale_o(orac, rg, seq_lens=seq_lens)
    spec = MxQuantSpec(**desc, scale_o=scale_o, w_qkvg_dtype=_FP4 if w_fp4 else E4M3)
    ref = gated_attention_block_mxfp8_reference(orac, rg, descale_w_o=spec.descale_w_o, scale_o=spec.scale_o, seq_lens=seq_lens)
    bf16_ref = gated_attention_block_reference(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], rg, seq_lens=seq_lens
    ).out
    out = torch.empty(batch, seq_len, geom.d_model, device="cuda", dtype=torch.bfloat16)
    if sentinel:
        out.fill_(_SENTINEL)
    blk = GatedAttentionBlockFwd(
        kern["h"],
        kern["w_qkvg"],
        kern["w_q_norm"],
        kern["w_k_norm"],
        kern["cos"],
        kern["sin"],
        kern["w_o"],
        out,
        geom,
        quant=spec,
        seq_lens_present=seq_lens is not None,
        sample_h_sf=kern["h_sf"],
        sample_w_qkvg_sf=kern["w_qkvg_sf"],
        **blk_kw,
    )
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    mx_suite._execute(blk, kern, out, ws, seq_lens=seq_lens)
    torch.cuda.synchronize()
    return out, ref, bf16_ref, blk, kern, spec


@requires_rubin
@requires_fp4
def test_e4m3_weight_spelling_is_bitwise_the_mxfp8_suites_run():
    """``MxQuantSpec(w_qkvg_dtype=torch.float8_e4m3fn)`` (the default, spelled out) runs the SAME plan on the SAME data as
    the MXFP8 suite's ``_run_mx_block`` and the outputs are bit-identical -- the appended field changes nothing for an
    existing caller (row 1-6: byte-identical)."""
    geom_kw = {**_GEOM, "is_causal": True}
    out_mx, _, blk_mx, _, spec_mx = mx_suite._run_mx_block(geom_kw, batch=2, seq_len=256)
    out, _, _, blk, _, spec = _run_fp4w_block(geom_kw, 2, 256, w_fp4=False)
    assert spec == spec_mx and blk._proj.w_dtype == blk_mx._proj.w_dtype == E4M3
    torch.testing.assert_close(out, out_mx, rtol=0, atol=0)


@requires_rubin
@requires_fp4
@pytest.mark.parametrize("seq_len, causal", [(256, True), (1000, True), (256, False), (1024, False)])
def test_mxfp4_weight_block_matches_the_fake_quant_oracle(seq_len, causal):
    """The MXFP8 suite's shapes (B=2, causal KV tail at S=1000, dense at S % 128 == 0) with an MXFP4 ``W_qkvg``: no
    sentinel survivor, finite, ``cos > 0.99`` vs the fake-quant oracle (``max_rel`` printed), and the FORMAT FLOOR --
    ``cos(kernel, bf16 ref) >= cos(oracle, bf16 ref) - 0.005`` with the oracle's own cosine printed, so an fp4
    weight's cost against the bf16 block is a number in the log, not a tolerance.  A second execute is bitwise."""
    out, ref, bf16_ref, blk, kern, spec = _run_fp4w_block({**_GEOM, "is_causal": causal}, 2, seq_len, w_fp4=True, sentinel=True)
    assert [s.name for s in blk._stages] == _MX_STAGES and blk._proj.w_dtype == _FP4 and blk._proj._plan.w_dtype == _FP4
    assert not (out == _SENTINEL).any(), f"{(out == _SENTINEL).sum().item()} output cells were never written"
    assert torch.isfinite(out.float()).all()
    c = mx_suite._cos(out, ref)
    rel = ((out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-30)).item()
    c_k, c_o = mx_suite._cos(out, bf16_ref), mx_suite._cos(ref, bf16_ref)
    print(
        f"\nmxfp4-weight block S={seq_len} causal={causal}: cos(kernel, oracle)={c:.6f} max_rel={rel:.3e} scale_o={spec.scale_o:.4g} "
        f"| format floor cos(oracle, bf16)={c_o:.6f} cos(kernel, bf16)={c_k:.6f} | proj route={blk._proj._plan.route}"
    )
    assert c > 0.99, f"mxfp4-weight block cos {c}"
    assert c_k >= c_o - 0.005, f"the kernel sits below the fp4 format floor: cos(kernel, bf16)={c_k} < cos(oracle, bf16)={c_o} - 0.005"
    out2 = torch.full_like(out, _SENTINEL)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    mx_suite._execute(blk, kern, out2, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(out2, out, rtol=0, atol=0)


@requires_rubin
@requires_fp4
def test_mxfp4_weight_dead_padded_entry_is_exactly_zero():
    """sdpa-invariants S1/S2 through the fp4-weight block: ``seq_lens=[s, 0]`` -> ``out[1] == 0`` EXACTLY (the weight's
    dtype cannot move a select), the live entry on the oracle."""
    s = 512
    seq_lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    out, ref, _, blk, _, _ = _run_fp4w_block({**_GEOM, "is_causal": False}, 2, s, w_fp4=True, seq_lens=seq_lens, sentinel=True)
    assert blk._sdpa.seq_lens_present and not (out == _SENTINEL).any() and torch.isfinite(out.float()).all()
    assert (out[1] == 0).all(), f"the dead entry must be EXACTLY zero; max|out[1]| = {out[1].abs().max().item()}"
    c = mx_suite._cos(out[0], ref[0])
    print(f"\nmxfp4-weight block dead entry S={s}: live cos={c:.6f}")
    assert c > 0.99, f"live entry cos {c}"


# ===========================================================================
# O ARMS: MxQuantSpec.o_fp4 = Fp4Format.NVFP4 | MXFP4 (plan config rows 8-10)
# ===========================================================================
#
# The gated O is block-quantized to E2M1 (``quantize_fp4_o``, ONE launch) and the out projection is the
# fp4 x fp4 block-scale GEMM against an e2m1 ``W_o`` ``[d_model, h_q*d_head // 2]`` with its F8_128x4 blob
# (``sample_w_o_sf`` / ``w_o_sf``).  UNFUSED: the frozen nine stages with ``quantize_o`` -> ``quantize_fp4_o``
# (row 8; with an fp4 W_qkvg too = row 9).  FULLY FUSED: the gated MXFP8 SDPA writes bf16 O, then the fp4
# quantize, then the fp4 out projection -- FOUR launches (row 10).  Workspace: ``o8`` is not reserved, ``o4`` /
# ``sf_o`` are appended at the END of either arm (the fused arm gets a bf16 ``o`` in ``o8``'s place); every
# layout WITHOUT ``o_fp4`` is byte-identical to before (the frozen snapshot below).

_FMTS = list(Fp4Format)
_FMT_IDS = [m.name for m in _FMTS]
_FP4_O_UNFUSED_STAGES = _MX_STAGES[:-2] + ["quantize_fp4_o", "out_proj"]
_FP4_O_FUSED_STAGES = ["qkv_gate_proj_norm_rope", "sdpa", "quantize_fp4_o", "out_proj"]


def _fp4_o_spec(fmt, **kw) -> MxQuantSpec:
    kw.setdefault("descale_w_o", 1.0)
    return MxQuantSpec(o_fp4=fmt, **kw)


def _w_o_sf_bytes(geom: GatedAttentionBlockGeometry, fmt) -> int:
    return sf_blob_bytes(geom.d_model, geom.h_q * geom.d_head, fmt.block_size)


def _decl_tensors_fp4o(fmt, geom_kw=_GEOM, batch=1, seq_len=256, *, w_o_storage: str = "fp4", w_qkvg_fp4: bool = False):
    """The MXFP8 suite's declaration tensors with ``w_o`` swapped for an e2m1 weight (``"fp4"`` = storage ``[d_model, K/2]``
    viewed ``float4_e2m1fn_x2``; ``"uint8"`` the same bytes left as uint8; ``"logical"`` an fp4x2 tensor at the LOGICAL
    ``[d_model, K]``; ``"e4m3"`` the MXFP8 suite's per-tensor e4m3 W_o) plus the e2m1 W_o's blob at ``fmt``'s block."""
    if w_qkvg_fp4:
        geom, args, hsf, wsf = _decl_tensors_fp4w(geom_kw, batch, seq_len)
    else:
        geom, args, hsf, wsf = mx_suite._decl_tensors(geom_kw, batch, seq_len)
    n, k = geom.d_model, geom.h_q * geom.d_head
    u8 = torch.zeros(n, k // 2, dtype=torch.uint8, device="cuda")
    w_o = {
        "fp4": u8.view(_FP4),
        "uint8": u8,
        "logical": torch.zeros(n, k, dtype=torch.uint8, device="cuda").view(_FP4),
        "e4m3": args[6],
    }[w_o_storage]
    args = list(args)
    args[6] = w_o
    wosf = torch.zeros(_w_o_sf_bytes(geom, fmt), dtype=torch.uint8, device="cuda")
    return geom, tuple(args), hsf, wsf, wosf


def _decl_block_fp4o(fmt, geom_kw=_GEOM, *, w_o_storage: str = "fp4", w_qkvg_fp4: bool = False, **kw) -> GatedAttentionBlockFwd:
    geom, args, hsf, wsf, wosf = _decl_tensors_fp4o(fmt, geom_kw, w_o_storage=w_o_storage, w_qkvg_fp4=w_qkvg_fp4)
    kw.setdefault("quant", _fp4_o_spec(fmt, w_qkvg_dtype=_FP4 if w_qkvg_fp4 else E4M3))
    kw.setdefault("sample_h_sf", hsf)
    kw.setdefault("sample_w_qkvg_sf", wsf)
    kw.setdefault("sample_w_o_sf", wosf)
    return GatedAttentionBlockFwd(*args, geom, **{k: v for k, v in kw.items() if v is not _ABSENT})


# ---------------------------------------------------------------------------
# Fp4Format + MxQuantSpec.o_fp4 (no GPU)
# ---------------------------------------------------------------------------


def test_fp4format_members_are_the_two_catalog_pairs_and_agree_with_the_quantize_kernel():
    """ONE member = (codes, scale dtype, block): NVFP4 = e4m3 / 16, MXFP4 = E8M0 / 32 -- and nothing else is spellable.
    ``kernels/quantize_fp4.py``'s ``fp4_format(member)`` resolves the member by NAME and cross-checks ``block_size``, so the
    enum and the kernel cannot drift; the cudnn scale dtype is what the GEMM declares its SF tensors in.  Exported."""
    import cudnn
    import cudnn.gated_attention_block as pkg
    from cudnn.gated_attention_block.kernels.quantize_fp4 import FORMATS, fp4_format

    assert pkg.Fp4Format is Fp4Format and "Fp4Format" in pkg.__all__
    assert [m.name for m in Fp4Format] == ["NVFP4", "MXFP4"] and len(Fp4Format) == 2
    nv, mx = Fp4Format.NVFP4, Fp4Format.MXFP4
    assert (nv.block_size, nv.sf_torch_dtype, nv.sf_cudnn_dtype, nv.fmt_name) == (16, E4M3, cudnn.data_type.FP8_E4M3, "nvfp4")
    assert (mx.block_size, mx.sf_torch_dtype, mx.sf_cudnn_dtype, mx.fmt_name) == (32, _E8M0, cudnn.data_type.FP8_E8M0, "mxfp4")
    for m in Fp4Format:
        name, block, sf_e4m3 = fp4_format(m)
        assert (name, block, sf_e4m3) == (m.fmt_name, m.block_size, m is Fp4Format.NVFP4)
        assert FORMATS[name] == (m.block_size, m is Fp4Format.NVFP4)
        assert m.block_size % 4 == 0 and _GEOM["d_head"] % (4 * m.block_size) == 0  # d=256 passes both formats' whole-word rule


def test_o_fp4_is_appended_last_defaults_to_none_and_the_e4m3_spelling_is_unchanged():
    """Append-only: ``o_fp4`` is the LAST field, after ``w_qkvg_dtype``; every existing spelling (positional included) is
    unchanged and reads ``o_fp4 is None``."""
    names = [f.name for f in dataclasses.fields(MxQuantSpec)]
    assert names[-2:] == ["w_qkvg_dtype", "o_fp4"] and names[:4] == ["descale_w_o", "scale_o", "dtype", "block_size"]
    assert MxQuantSpec(descale_w_o=0.03).o_fp4 is None and MxQuantSpec(0.03, 1.0, E4M3, 32, E4M3).o_fp4 is None
    assert MxQuantSpec(0.03, 1.0, E4M3, 32, E4M3, None) == mx_suite._SPEC


@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_o_fp4_validate_accepts_both_members_and_pins_the_two_scales_to_one(fmt):
    """Both members validate unfused AND fused with unit scales (D8 is skipped under o_fp4: scale_o == 1.0 is already
    pinned); ``scale_o != 1`` and ``descale_w_o != 1`` are ``ValueError`` naming the reason (no global scale on a block-scaled
    side); a non-``Fp4Format`` is a ``TypeError``; an fp4 W_qkvg composes with an fp4 O (row 9)."""
    spec = _fp4_o_spec(fmt)
    spec.validate()
    spec.validate(fused=True)
    assert spec.alpha_o == 1.0 and spec.scale_o == 1.0 and spec.descale_w_o == 1.0
    _fp4_o_spec(fmt, w_qkvg_dtype=_FP4).validate()
    with pytest.raises(ValueError, match="scale_o=1.0") as ei:
        _fp4_o_spec(fmt, scale_o=2.0).validate()
    assert "block-scaled O has no per-tensor scale" in str(ei.value)
    with pytest.raises(ValueError, match="descale_w_o=1.0") as ei:
        _fp4_o_spec(fmt, descale_w_o=0.5).validate()
    assert "w_o_sf" in str(ei.value)
    for bad in (fmt.name.lower(), fmt.block_size, object(), (16, E4M3)):
        with pytest.raises(TypeError, match="Fp4Format"):
            MxQuantSpec(descale_w_o=1.0, o_fp4=bad).validate()
    # the pre-existing rules still fire first with an o_fp4 set
    with pytest.raises(NotImplementedError, match="e4m3"):
        MxQuantSpec(descale_w_o=1.0, dtype=torch.float8_e5m2, o_fp4=fmt).validate()
    with pytest.raises(ValueError, match="positive"):
        MxQuantSpec(descale_w_o=0.0, o_fp4=fmt).validate()


@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_o_fp4_declines_typed_on_a_torch_without_the_packed_e2m1_dtype(fmt, monkeypatch):
    """A torch without ``float4_e2m1fn_x2`` cannot view the o4 buffer or W_o: ``validate`` declines with a
    ``NotImplementedError`` naming the dtype and the torch version (mirrors ``quantize_fp4.check_torch_fp4_dtypes``),
    instead of ``_expected_weight_dtypes`` reporting a ``None`` weight dtype at declaration.  ``o_fp4=None`` is untouched."""
    import cudnn.gated_attention_block.api as api_mod

    monkeypatch.setattr(api_mod, "_FP4_X2", None)
    with pytest.raises(NotImplementedError, match="float4_e2m1fn_x2") as ei:
        _fp4_o_spec(fmt).validate()
    assert torch.__version__ in str(ei.value)
    mx_suite._SPEC.validate()  # the e4m3 spelling never touches the fp4 dtype


def test_o_fp4_needs_both_halves_at_declaration():
    """``o_fp4`` without ``sample_w_o_sf`` and ``sample_w_o_sf`` without ``o_fp4`` are each a ``ValueError`` at declaration; a
    ``QuantSpec`` / bf16 block can name the blob but never the mode (the field lives on ``MxQuantSpec``) -- same message,
    which then says so; the spec's own declines (``TypeError`` / the unit-scale rules) fire BEFORE the both-halves rule."""
    nv = Fp4Format.NVFP4
    with pytest.raises(ValueError, match="both halves") as ei:
        _decl_block_fp4o(nv, sample_w_o_sf=_ABSENT)
    assert "sample_w_o_sf=None" in str(ei.value) and "o_fp4=Fp4Format.NVFP4" in str(ei.value)
    with pytest.raises(ValueError, match="both halves") as ei:
        _decl_block_fp4o(nv, quant=mx_suite._SPEC)  # blob, no mode
    assert "o_fp4=None" in str(ei.value)
    geom, args, hsf, wsf, wosf = _decl_tensors_fp4o(nv, w_o_storage="e4m3")
    with pytest.raises(ValueError, match="only an MxQuantSpec carries o_fp4"):
        GatedAttentionBlockFwd(*args, geom, quant=mx_suite._PT_SPEC, sample_w_o_sf=wosf)
    bf16 = [a.to(torch.bfloat16) if a.dtype == E4M3 else a for a in args]
    with pytest.raises(ValueError, match="only an MxQuantSpec carries o_fp4"):
        GatedAttentionBlockFwd(*bf16, geom, sample_w_o_sf=wosf)
    with pytest.raises(TypeError, match="Fp4Format"):
        _decl_block_fp4o(nv, quant=MxQuantSpec(descale_w_o=1.0, o_fp4="nvfp4"), sample_w_o_sf=_ABSENT)
    with pytest.raises(ValueError, match="scale_o=1.0"):
        _decl_block_fp4o(nv, quant=_fp4_o_spec(nv, scale_o=4.0))
    # the pre-existing MXFP8 rules are untouched by the new halves
    with pytest.raises(ValueError, match="three halves"):
        _decl_block_fp4o(nv, sample_w_qkvg_sf=_ABSENT)
    with pytest.raises(NotImplementedError, match="fuse_norm_rope"):
        _decl_block_fp4o(nv, fuse_gate=True)


def test_o_fp4_save_for_backward_declines_typed():
    """Inference-only, inherited from the quantized pipelines' rule -- and the fused knobs stay under their own guards."""
    for fmt in Fp4Format:
        with pytest.raises(NotImplementedError, match="inference-only"):
            _decl_block_fp4o(fmt, save_for_backward=True, inplace_qkv=False)
        with pytest.raises(ValueError, match="incompatible with save_for_backward"):
            _decl_block_fp4o(fmt, save_for_backward=True, **_FUSED)  # the fused knobs' own training guards fire first


# ---------------------------------------------------------------------------
# Declaration: stage list, out projection, workspace (any CUDA device)
# ---------------------------------------------------------------------------


@requires_fp4
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_o_fp4_declaration_wires_the_fp4_quantize_stage_and_the_block_scale_out_proj(fmt, fused):
    """Rows 8 / 10: ``quantize_o`` -> ``quantize_fp4_o`` (``_QuantizeFp4`` on the format, h_q heads, bf16 in), the out projection
    is the fp4 x fp4 block-scale GEMM at the format's block and scale dtype with NO alpha (``_Projection.check_support``
    accepts the pairing on any device), ``_quant_dev`` holds nothing, and on the fused arm the gated MXFP8 SDPA is compiled
    for a **bf16** O (the adapter's bf16-O-for-fp8-input path; ``impl.dtype_o``) with the compact bf16 gate."""
    from cudnn.gated_attention_block.api import _QuantizeFp4, _Sdpa

    blk = _decl_block_fp4o(fmt, **(_FUSED if fused else {}))
    blk._check_declaration()
    assert blk.mxfp8 and blk.mxfp8_fused == fused and blk.o_fp4 is fmt and blk.quant.o_fp4 is fmt
    assert [s.name for s in blk._stages] == (_FP4_O_FUSED_STAGES if fused else _FP4_O_UNFUSED_STAGES)
    q = blk._quant_o
    assert isinstance(q, _QuantizeFp4) and q.name == "quantize_fp4_o" and q.fmt is fmt and q.heads == blk.geom.h_q and q.dtype_in == torch.bfloat16
    assert q._format() == (fmt.fmt_name, fmt.block_size, fmt is Fp4Format.NVFP4)
    q.check_support()
    assert q.code_bytes() == blk.batch * blk.seq_len * blk.geom.h_q * blk.geom.d_head // 2
    assert q.sf_bytes() == sf_blob_bytes(blk.batch * blk.seq_len, blk.geom.h_q * blk.geom.d_head, fmt.block_size)
    p = blk._out_proj
    assert p.block_scale and not p.alpha and p.dtype == _FP4 and p.w_dtype == _FP4 and p.block_size == fmt.block_size and p.sf_dtype == fmt.sf_cudnn_dtype
    assert p.out_dtype == torch.bfloat16 and p.k == blk.geom.h_q * blk.geom.d_head and p.n == blk.geom.d_model
    p.check_support()  # the pairing table serves (e2m1, e2m1, E4M3/16) and (e2m1, e2m1, E8M0/32)
    assert blk._expected_weight_dtypes() == {"w_qkvg": E4M3, "w_o": _FP4}
    assert blk._make_quant_dev() == {}, "neither alpha_o nor scale_o exists under o_fp4 (both pinned 1.0)"
    st = blk._sdpa
    assert isinstance(st, _Sdpa) and st.mxfp8 and st.fuse_gate == fused and st.o_dtype == torch.bfloat16
    impl = st._build_impl()
    assert impl._pertensor is False and impl.dtype_o == torch.bfloat16 and impl.has_amax_o is False
    assert (impl.gate_desc is not None) == fused
    if fused:
        assert impl.gate_desc.dtype == torch.bfloat16 and blk._gate is None and blk._quant_q is None
        assert "quant_mxfp8" in str(blk._proj.params()) and "weight_fp4" not in str(blk._proj.params())
    else:
        assert blk._gate is not None and blk._quant_q is not None


@requires_fp4
def test_o_fp4_with_an_fp4_w_qkvg_is_row_nine():
    """Both fp4 fields together (unfused): the mixed GEMM at stage (1) AND the fp4 tail -- the stage list of row 8, the
    weight contract per weight (``w_qkvg`` e2m1 by ``w_qkvg_dtype``, ``w_o`` e2m1 by ``o_fp4``), ``w_qkvg_sf`` unchanged."""
    blk = _decl_block_fp4o(Fp4Format.MXFP4, w_qkvg_fp4=True)
    blk._check_declaration()
    assert blk.quant.w_qkvg_fp4 and blk.o_fp4 is Fp4Format.MXFP4
    assert [s.name for s in blk._stages] == _FP4_O_UNFUSED_STAGES
    assert blk._expected_weight_dtypes() == {"w_qkvg": _FP4, "w_o": _FP4}
    assert blk._proj.w_dtype == _FP4 and blk._proj.dtype == E4M3 and blk._out_proj.dtype == blk._out_proj.w_dtype == _FP4
    blk._proj.check_support()
    blk._out_proj.check_support()
    assert blk._layout() == _decl_block_fp4o(Fp4Format.MXFP4)._layout(), "the fp4 weight adds no slot; the fp4 O's slots are the same"
    # row 11 stays a typed decline on top of the fp4 O
    with pytest.raises(NotImplementedError, match="e4m3 B"):
        _decl_block_fp4o(Fp4Format.MXFP4, w_qkvg_fp4=True, **_FUSED).check_support() if not _fork_has_fp4_arm() else pytest.skip("fork arm landed")


@requires_fp4
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_o_fp4_workspace_drops_o8_and_appends_o4_and_sf_o_at_the_end(fmt, fused):
    """``o8 == -1`` (never written, never reserved); ``o4`` (``t*h_q*d/2`` bytes) then ``sf_o`` -- sized by the GEMM's
    ``sf_blob_bytes(t, h_q*d, block)``, NOT the SDPA's ``_sf_slot_bytes`` -- are the LAST two slots; every slot ahead of ``o8``'s
    old position keeps the MXFP8 layout's offset, and the slots after it move up by exactly the dropped ``o8`` (unfused) or by
    the bf16 ``o`` that replaces it (fused, where ``o >= 0`` because the gated SDPA writes bf16 O)."""
    from cudnn.gated_attention_block.api import _align_up, _plan_workspace, _sf_slot_bytes

    blk = _decl_block_fp4o(fmt, **(_FUSED if fused else {}))
    ref = mx_suite._decl_block(**(_FUSED if fused else {}))
    lay, mx = blk._layout(), ref._layout()
    g, b, s = blk.geom, blk.batch, blk.seq_len
    t = b * s
    code_bytes, sf_bytes = t * g.h_q * g.d_head // 2, sf_blob_bytes(t, g.h_q * g.d_head, fmt.block_size)
    assert lay.o8 == -1 and lay.o4 >= 0 and lay.sf_o >= 0
    assert lay.sf_o - lay.o4 == _align_up(code_bytes) and lay.total_bytes == lay.engine_scratch == lay.sf_o + _align_up(sf_bytes)
    assert lay.o4 == lay.sf_v + _align_up(_sf_slot_bytes(b, g.h_kv, s, g.d_head)), "o4 follows sf_v: the fp4 slots are appended at the END"
    assert sf_bytes == blk._quant_o.sf_bytes() and code_bytes == blk._quant_o.code_bytes()
    if fused:
        assert lay.o >= 0 and lay.o == mx.o8 and (lay.q8, lay.k8, lay.v8, lay.gate16) == (mx.q8, mx.k8, mx.v8, mx.gate16) and lay.proj == -1
        shift = _align_up(t * g.h_q * g.d_head * 2) - _align_up(t * g.h_q * g.d_head)  # bf16 o in place of e4m3 o8
    else:
        assert (lay.proj, lay.q8, lay.k8, lay.v8, lay.o) == (mx.proj, mx.q8, mx.k8, mx.v8, mx.o) and lay.q == lay.v == -1
        shift = -_align_up(t * g.h_q * g.d_head)  # o8 dropped
    assert (lay.sf_q, lay.sf_k, lay.sf_v) == (mx.sf_q + shift, mx.sf_k + shift, mx.sf_v + shift)
    assert lay.o4 == mx.total_bytes + shift
    assert lay == _plan_workspace(g, b, s, torch.bfloat16, False, False, True, fp8=True, fp8_fused=fused, mxfp8=True, o_fp4=fmt)
    # R7: two SF contracts coexist.  sf_o is the GEMM's blob (rows padded to 128 over T = B*S, K/block blocks padded to 4),
    # never the SDPA's per-(b, h, s_tile) count.  The two COUNTS coincide at some shapes (MXFP4 at B=1; NVFP4 at a ragged B=2
    # where ceil(B*S/128) == B*ceil(S/128)/2) while the byte ORDER never does, so the inequality is pinned at a separating
    # shape per format: the suite shape for NVFP4 (2x), a ragged B=2, S=64 for MXFP4 (ceil(128/128) = 1 < 2*ceil(64/128) = 2).
    sb, ss = (b, s) if fmt is Fp4Format.NVFP4 else (2, 64)
    sep = _plan_workspace(g, sb, ss, torch.bfloat16, False, False, True, fp8=True, fp8_fused=fused, mxfp8=True, o_fp4=fmt)
    sep_sf = sf_blob_bytes(sb * ss, g.h_q * g.d_head, fmt.block_size)
    assert sep.total_bytes - sep.sf_o == _align_up(sep_sf) and sep_sf != _sf_slot_bytes(sb, g.h_q, ss, g.d_head)


# The frozen offsets of EVERY existing pipeline (o_fp4=None) at the suite geometry (B=2, S=1000) and the 397B geometry
# (B=1, S=4096), computed before the fp4 slots existed and pasted as literals on purpose: a change to any number here is a
# workspace-layout change for a caller that never asked for fp4.  Keys: (geometry, arm); values: the _Intermediates fields.
_GEOM_397B = dict(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64)
_SNAPSHOT_SHAPES = {"test": (_GEOM, 2, 1000), "397b": (_GEOM_397B, 1, 4096)}
_SNAPSHOT_ARMS = {
    "bf16_inplace": dict(inplace_qkv=True),
    "bf16_compact": dict(inplace_qkv=False),
    "fp8_unfused": dict(inplace_qkv=True, fp8=True),
    "fp8_fused": dict(inplace_qkv=True, fp8=True, fp8_fused=True),
    "mxfp8_unfused": dict(inplace_qkv=True, fp8=True, mxfp8=True),
    "mxfp8_fused": dict(inplace_qkv=True, fp8=True, fp8_fused=True, mxfp8=True),
}
_COMMON = dict(gate=-1, o_gated=-1, base_align=256, o4=-1, sf_o=-1)
_SNAPSHOT = {
    ("test", "bf16_inplace"): dict(
        proj=0, q=-1, k=-1, v=-1, o=20480000, engine_scratch=28672000, total_bytes=28672000, q8=-1, k8=-1, v8=-1, o8=-1, gate16=-1, sf_q=-1, sf_k=-1, sf_v=-1
    ),
    ("test", "bf16_compact"): dict(
        proj=0,
        q=20480000,
        k=28672000,
        v=30720000,
        o=32768000,
        engine_scratch=40960000,
        total_bytes=40960000,
        q8=-1,
        k8=-1,
        v8=-1,
        o8=-1,
        gate16=-1,
        sf_q=-1,
        sf_k=-1,
        sf_v=-1,
    ),
    ("test", "fp8_unfused"): dict(
        proj=0,
        q=-1,
        k=-1,
        v=-1,
        o=26624000,
        engine_scratch=38912000,
        total_bytes=38912000,
        q8=20480000,
        k8=24576000,
        v8=25600000,
        o8=34816000,
        gate16=-1,
        sf_q=-1,
        sf_k=-1,
        sf_v=-1,
    ),
    ("test", "fp8_fused"): dict(
        proj=-1,
        q=-1,
        k=-1,
        v=-1,
        o=-1,
        engine_scratch=18432000,
        total_bytes=18432000,
        q8=0,
        k8=4096000,
        v8=5120000,
        o8=14336000,
        gate16=6144000,
        sf_q=-1,
        sf_k=-1,
        sf_v=-1,
    ),
    ("test", "mxfp8_unfused"): dict(
        proj=0,
        q=-1,
        k=-1,
        v=-1,
        o=26624000,
        engine_scratch=39108608,
        total_bytes=39108608,
        q8=20480000,
        k8=24576000,
        v8=25600000,
        o8=34816000,
        gate16=-1,
        sf_q=38912000,
        sf_k=39043072,
        sf_v=39075840,
    ),
    ("test", "mxfp8_fused"): dict(
        proj=-1,
        q=-1,
        k=-1,
        v=-1,
        o=-1,
        engine_scratch=18628608,
        total_bytes=18628608,
        q8=0,
        k8=4096000,
        v8=5120000,
        o8=14336000,
        gate16=6144000,
        sf_q=18432000,
        sf_k=18563072,
        sf_v=18595840,
    ),
    ("397b", "bf16_inplace"): dict(
        proj=0, q=-1, k=-1, v=-1, o=142606336, engine_scratch=209715200, total_bytes=209715200, q8=-1, k8=-1, v8=-1, o8=-1, gate16=-1, sf_q=-1, sf_k=-1, sf_v=-1
    ),
    ("397b", "bf16_compact"): dict(
        proj=0,
        q=142606336,
        k=209715200,
        v=213909504,
        o=218103808,
        engine_scratch=285212672,
        total_bytes=285212672,
        q8=-1,
        k8=-1,
        v8=-1,
        o8=-1,
        gate16=-1,
        sf_q=-1,
        sf_k=-1,
        sf_v=-1,
    ),
    ("397b", "fp8_unfused"): dict(
        proj=0,
        q=-1,
        k=-1,
        v=-1,
        o=180355072,
        engine_scratch=281018368,
        total_bytes=281018368,
        q8=142606336,
        k8=176160768,
        v8=178257920,
        o8=247463936,
        gate16=-1,
        sf_q=-1,
        sf_k=-1,
        sf_v=-1,
    ),
    ("397b", "fp8_fused"): dict(
        proj=-1,
        q=-1,
        k=-1,
        v=-1,
        o=-1,
        engine_scratch=138412032,
        total_bytes=138412032,
        q8=0,
        k8=33554432,
        v8=35651584,
        o8=104857600,
        gate16=37748736,
        sf_q=-1,
        sf_k=-1,
        sf_v=-1,
    ),
    ("397b", "mxfp8_unfused"): dict(
        proj=0,
        q=-1,
        k=-1,
        v=-1,
        o=180355072,
        engine_scratch=282198016,
        total_bytes=282198016,
        q8=142606336,
        k8=176160768,
        v8=178257920,
        o8=247463936,
        gate16=-1,
        sf_q=281018368,
        sf_k=282066944,
        sf_v=282132480,
    ),
    ("397b", "mxfp8_fused"): dict(
        proj=-1,
        q=-1,
        k=-1,
        v=-1,
        o=-1,
        engine_scratch=139591680,
        total_bytes=139591680,
        q8=0,
        k8=33554432,
        v8=35651584,
        o8=104857600,
        gate16=37748736,
        sf_q=138412032,
        sf_k=139460608,
        sf_v=139526144,
    ),
}


@pytest.mark.parametrize("arm", list(_SNAPSHOT_ARMS))
@pytest.mark.parametrize("shape", list(_SNAPSHOT_SHAPES))
def test_workspace_layout_is_byte_identical_without_fp4(shape, arm):
    """Every field of ``_plan_workspace`` for every pre-existing pipeline equals the frozen snapshot (test AND 397B geometry,
    unfused AND fused arms), and the appended ``o_fp4=None`` spells the same layout as not passing it.  The two new fields
    read ``-1`` there.  No GPU."""
    from cudnn.gated_attention_block.api import _plan_workspace

    geom_kw, b, s = _SNAPSHOT_SHAPES[shape]
    kw = dict(_SNAPSHOT_ARMS[arm])
    inplace = kw.pop("inplace_qkv")
    geom = GatedAttentionBlockGeometry(**geom_kw)
    lay = _plan_workspace(geom, b, s, torch.bfloat16, False, False, inplace, **kw)
    assert dataclasses.asdict(lay) == {**_COMMON, **_SNAPSHOT[(shape, arm)]}
    assert lay == _plan_workspace(geom, b, s, torch.bfloat16, False, False, inplace, o_fp4=None, **kw)
    # and the same layout with an fp4 O keeps every slot ahead of o8's position where it was (the fp4 arms cannot move an
    # existing pipeline's offsets because they are only ever reached with o_fp4 set); o8 is gone, o4 / sf_o are appended
    if kw.get("mxfp8"):
        from cudnn.gated_attention_block.api import _align_up

        t = b * s
        for fmt in Fp4Format:
            f4 = _plan_workspace(geom, b, s, torch.bfloat16, False, False, inplace, o_fp4=fmt, **kw)
            assert f4.o8 == -1 and f4.o4 >= 0 and f4.sf_o > f4.o4
            assert (f4.proj, f4.q8, f4.k8, f4.v8, f4.gate16) == (lay.proj, lay.q8, lay.k8, lay.v8, lay.gate16)
            o_bytes = _align_up(t * geom.h_q * geom.d_head * 2) if kw.get("fp8_fused") else 0  # the fused arm's bf16 o in o8's place
            tail = _align_up(t * geom.h_q * geom.d_head // 2) + _align_up(sf_blob_bytes(t, geom.h_q * geom.d_head, fmt.block_size))
            assert f4.total_bytes == lay.total_bytes - _align_up(t * geom.h_q * geom.d_head) + o_bytes + tail


# ---------------------------------------------------------------------------
# Declaration declines: W_o dtype / storage shape, w_o_sf bytes / dtype / contiguity, d_head (any CUDA device)
# ---------------------------------------------------------------------------


@requires_fp4
@pytest.mark.parametrize("case", ["e4m3_w_o", "uint8_storage", "logical_shape"])
def test_o_fp4_rejects_a_wrong_w_o_dtype_or_storage_shape_typed(case):
    """Under ``o_fp4`` the out-projection weight is checked against ``float4_e2m1fn_x2`` and the STORAGE shape
    ``[d_model, h_q*d_head // 2]`` -- an e4m3 / uint8 / LOGICAL-shaped ``w_o`` is a ``ValueError`` naming ``MxQuantSpec.o_fp4``
    (uint8 with the ``.view`` hint), at the declaration half of ``check_support`` -- never a silent reinterpretation."""
    blk = _decl_block_fp4o(Fp4Format.NVFP4, w_o_storage={"e4m3_w_o": "e4m3", "uint8_storage": "uint8", "logical_shape": "logical"}[case])
    if case == "e4m3_w_o":
        with pytest.raises(ValueError, match="w_o must be torch.float4_e2m1fn_x2") as ei:
            blk._check_declaration()
        assert "MxQuantSpec.o_fp4" in str(ei.value)
    elif case == "uint8_storage":
        with pytest.raises(ValueError, match=r"\.view\(torch\.float4_e2m1fn_x2\)") as ei:
            blk._check_declaration()
        assert str(ei.value).startswith("w_o must be torch.float4_e2m1fn_x2")
    else:
        with pytest.raises(ValueError, match="LOGICAL") as ei:
            blk._check_declaration()
        n, k = blk.geom.d_model, blk.geom.h_q * blk.geom.d_head
        assert f"[{n}, {k} // 2 = {k // 2}]" in str(ei.value) and f"storage ({n}, {k})" in str(ei.value)
    _decl_block_fp4o(Fp4Format.NVFP4)._check_declaration()  # the healthy spelling passes: each decline above is the ONE wrong thing


@requires_fp4
def test_fp4_w_o_without_o_fp4_names_the_output_mode():
    """The converse: an e2m1 ``w_o`` on a block WITHOUT ``o_fp4`` (the plain MXFP8 spec, no blob) is a ``ValueError`` that names
    the mode to declare (``MxQuantSpec(o_fp4=...)`` + ``sample_w_o_sf``), not a shape mismatch."""
    geom, args, hsf, wsf, _ = _decl_tensors_fp4o(Fp4Format.NVFP4)
    blk = GatedAttentionBlockFwd(*args, geom, quant=mx_suite._SPEC, sample_h_sf=hsf, sample_w_qkvg_sf=wsf)
    with pytest.raises(ValueError, match="w_o is torch.float4_e2m1fn_x2") as ei:
        blk._check_declaration()
    assert "MxQuantSpec(o_fp4=Fp4Format.NVFP4 | MXFP4) plus sample_w_o_sf" in str(ei.value)


@requires_fp4
@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_o_fp4_rejects_a_wrong_w_o_sf_byte_count_or_dtype_and_accepts_the_formats_own_scale_dtype(fmt):
    """``sample_w_o_sf`` is the PADDED F8_128x4 blob over ``d_model`` rows x ``K = h_q*d_head`` at the FORMAT's block: one byte
    short, the OTHER format's byte count, a non-scale dtype, or the OTHER format's scale dtype (the declared scale dtype
    selects the MMA's scale format -- an e4m3 blob read as E8M0 miscomputes silently) are each a typed ``ValueError`` at
    check_support; ``uint8`` and the format's own fp8 storage dtype are accepted.  Non-contiguity is refused at execute
    (a descriptor carries no pointer)."""
    other = Fp4Format.MXFP4 if fmt is Fp4Format.NVFP4 else Fp4Format.NVFP4
    geom, args, hsf, wsf, wosf = _decl_tensors_fp4o(fmt)
    mk = lambda blob: GatedAttentionBlockFwd(*args, geom, quant=_fp4_o_spec(fmt), sample_h_sf=hsf, sample_w_qkvg_sf=wsf, sample_w_o_sf=blob)  # noqa: E731
    with pytest.raises(ValueError, match="sf_blob_bytes") as ei:
        mk(wosf[:-16])._check_declaration()
    assert f"at block {fmt.block_size}" in str(ei.value) and f"is {wosf.numel()}" in str(ei.value)
    with pytest.raises(ValueError, match="sf_blob_bytes"):
        mk(torch.zeros(_w_o_sf_bytes(geom, other), dtype=torch.uint8, device="cuda"))._check_declaration()
    with pytest.raises(ValueError, match="sample_w_o_sf must be torch.uint8 or") as ei:
        mk(wosf.view(torch.int8))._check_declaration()
    assert str(fmt.sf_torch_dtype) in str(ei.value)
    with pytest.raises(ValueError, match="sample_w_o_sf must be torch.uint8 or"):
        mk(wosf.view(other.sf_torch_dtype))._check_declaration()
    mk(wosf)._check_declaration()
    mk(wosf.view(fmt.sf_torch_dtype))._check_declaration()
    # the two MXFP8 blobs keep their own checks (block 32, E8M0) with an o_fp4 set
    with pytest.raises(ValueError, match="sample_h_sf"):
        GatedAttentionBlockFwd(*args, geom, quant=_fp4_o_spec(fmt), sample_h_sf=hsf[:-16], sample_w_qkvg_sf=wsf, sample_w_o_sf=wosf)._check_declaration()


@requires_fp4
def test_o_fp4_declines_a_head_dim_that_is_not_whole_scale_words():
    """``d_head % (4 * block) != 0`` -> ``NotImplementedError`` naming the format, the rule and the head dim: one head's scales must
    be whole 4-block F8_128x4 words so the per-head quantize CTA owns whole atoms.  d=64 passes NVFP4 (4*16) and fails MXFP4
    (4*32); d=256 passes both (the suite geometry)."""
    geom_kw = {**_GEOM, "d_head": 64, "rope_dim": 32}
    blk = _decl_block_fp4o(Fp4Format.MXFP4, geom_kw)
    with pytest.raises(NotImplementedError, match="d_head % 128 == 0") as ei:
        blk._check_declaration()
    assert "MXFP4" in str(ei.value) and "d_head=64" in str(ei.value)
    with pytest.raises(NotImplementedError, match="4"):
        blk._quant_o.check_support()  # the stage carries the same rule (the kernel's own contract)
    _decl_block_fp4o(Fp4Format.NVFP4, geom_kw)._check_declaration()  # 64 % 64 == 0


@requires_fp4
@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_o_fp4_execute_requires_w_o_sf_and_a_block_without_o_fp4_refuses_it(fmt):
    """Both directions at execute, before any launch: the fp4-O block needs ``w_o_sf`` (checked for bytes at the format's block,
    dtype, contiguity, 16-B alignment, device); an MXFP8 block declared without ``o_fp4`` refuses it (a silently dropped scale
    blob is a wrong answer nobody reports)."""
    blk = _decl_block_fp4o(fmt)
    blk._ws = blk._layout()  # bypass compile: the contract checks run first
    geom, args, hsf, wsf, wosf = _decl_tensors_fp4o(fmt)
    ws = torch.empty(16, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="execute needs w_o_sf") as ei:
        blk.execute(*args, ws, h_sf=hsf, w_qkvg_sf=wsf)
    assert fmt.name in str(ei.value)
    with pytest.raises(ValueError, match=f"at block {fmt.block_size}"):
        blk.execute(*args, ws, h_sf=hsf, w_qkvg_sf=wsf, w_o_sf=wosf[:-16])
    with pytest.raises(ValueError, match="contiguous"):
        blk.execute(*args, ws, h_sf=hsf, w_qkvg_sf=wsf, w_o_sf=torch.zeros(2 * wosf.numel(), dtype=torch.uint8, device="cuda")[::2])
    other = Fp4Format.MXFP4 if fmt is Fp4Format.NVFP4 else Fp4Format.NVFP4
    with pytest.raises(ValueError, match="w_o_sf must be"):
        blk.execute(*args, ws, h_sf=hsf, w_qkvg_sf=wsf, w_o_sf=wosf.view(other.sf_torch_dtype))
    # the MXFP8 blobs are still required ahead of it
    with pytest.raises(ValueError, match="both scale-factor blobs"):
        blk.execute(*args, ws, h_sf=hsf, w_o_sf=wosf)
    plain = mx_suite._decl_block()
    plain._ws = plain._layout()
    _, pargs, phsf, pwsf = mx_suite._decl_tensors()
    with pytest.raises(ValueError, match="without MxQuantSpec.o_fp4"):
        plain.execute(*pargs, ws, h_sf=phsf, w_qkvg_sf=pwsf, w_o_sf=wosf)


# ---------------------------------------------------------------------------
# execute body (monkeypatched trace; any CUDA device)
# ---------------------------------------------------------------------------


def _dispatch_trace_fp4o(blk, monkeypatch, fmt):
    """``mx_suite._dispatch_trace`` for an fp4-O block: every stage's launch replaced by a recorder that keeps the
    positional / keyword arguments; the block's own ``_make_quant_dev`` (``{}``) stands in for ``compile``.
    Returns ``[(stage, method, args, kwargs)]`` in call order."""
    lay = blk._layout()
    blk._ws = lay
    blk._is_supported = True
    monkeypatch.setattr(blk, "get_workspace_size", lambda: lay.total_bytes)
    blk._quant_dev = blk._make_quant_dev()
    calls = []
    for st in blk._stages:
        for meth in ("execute", "execute_mxfp8", "execute_fp8"):
            if hasattr(st, meth):
                monkeypatch.setattr(st, meth, lambda *a, _n=st.name, _m=meth, **k: calls.append((_n, _m, a, k)))
    _, args, hsf, wsf, wosf = _decl_tensors_fp4o(fmt, batch=blk.batch, seq_len=blk.seq_len)
    ws = torch.zeros(lay.total_bytes, dtype=torch.uint8, device="cuda")
    blk.execute(*args, ws, h_sf=hsf, w_qkvg_sf=wsf, w_o_sf=wosf)
    return calls, wosf, ws


@requires_fp4
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_o_fp4_execute_dispatches_every_stage_once_in_pipeline_order_with_the_fp4_tail(fmt, fused, monkeypatch):
    """CUPTI-free twin of the launch-count test.  UNFUSED: the nine frozen stages with ``quantize_fp4_o`` in ``quantize_o``'s
    place; FUSED: projection through ``execute_mxfp8``, the gated SDPA writing the **bf16** ``o`` view, ``quantize_fp4_o``,
    ``out_proj`` -- FOUR calls.  The fp4 tail's hand-off is pinned on the recorded arguments: the quantizer gets the bf16 O,
    the ``float4_e2m1fn_x2 [T, K/2]`` ``o4`` view and the ``sf_blob_bytes`` uint8 ``sf_o`` view; the out projection gets that
    SAME ``o4`` as A, ``sf_a=sf_o`` and ``sf_w=w_o_sf`` (the caller's tensor, by pointer), and NO alpha."""
    blk = _decl_block_fp4o(fmt, **(_FUSED if fused else {}))
    calls, wosf, ws = _dispatch_trace_fp4o(blk, monkeypatch, fmt)
    names = [n for n, _, _, _ in calls]
    assert names == (_FP4_O_FUSED_STAGES if fused else _FP4_O_UNFUSED_STAGES)
    assert [m for _, m, _, _ in calls] == (["execute_mxfp8"] if fused else ["execute"]) + ["execute"] * (len(names) - 1)
    g, t = blk.geom, blk.batch * blk.seq_len
    k = g.h_q * g.d_head
    sdpa_o = next(a for n, _, a, _ in calls if n == "sdpa")[3]
    assert sdpa_o.dtype == torch.bfloat16 and tuple(sdpa_o.shape) == (blk.batch, blk.seq_len, g.h_q, g.d_head)
    _, _, qa, qk = next(c for c in calls if c[0] == "quantize_fp4_o")
    src, o4, sfo = qa
    assert src.dtype == torch.bfloat16 and tuple(src.shape) == (t, g.h_q, g.d_head) and src.data_ptr() == sdpa_o.data_ptr()
    assert o4.dtype == _FP4 and tuple(o4.shape) == (t, k // 2) and o4.is_contiguous()
    assert sfo.dtype == torch.uint8 and sfo.numel() == sf_blob_bytes(t, k, fmt.block_size) and sfo.is_contiguous()
    lay = blk._ws
    assert o4.data_ptr() == ws.data_ptr() + lay.o4 and sfo.data_ptr() == ws.data_ptr() + lay.sf_o
    _, _, pa, pk = calls[-1]
    assert pa[0].data_ptr() == o4.data_ptr() and pa[0].dtype == _FP4 and tuple(pa[0].shape) == (t, k // 2)
    assert pa[1].dtype == _FP4 and tuple(pa[2].shape) == (t, g.d_model)
    assert pk["sf_a"].data_ptr() == sfo.data_ptr() and pk["sf_w"].data_ptr() == wosf.data_ptr() and pk.get("alpha") is None
    assert blk._quant_dev == {}


# ---------------------------------------------------------------------------
# Numerics -- Rubin
# ---------------------------------------------------------------------------


def _fp4o_inputs(geom_kw, batch, seq_len, fmt):
    """The MXFP8 suite's quantized inputs with ``W_o`` re-quantized to ``fmt`` (packed E2M1 ``[d_model, K/2]`` viewed
    ``float4_e2m1fn_x2`` + the format's padded blob) -- what the block gets AND what the oracle dequantizes through."""
    inp = make_inputs(RefGeometry(**geom_kw), batch=batch, seq_len=seq_len, dtype=torch.bfloat16)
    mx, desc = quantize_block_inputs_mxfp8(inp, o_fp4=fmt)
    assert mx["w_o"].dtype == _FP4 and mx["w_o_sf"].numel() == sf_blob_bytes(*inp["w_o"].shape, fmt.block_size) and desc == dict(descale_w_o=1.0)
    return inp, mx, desc


def _execute_fp4o(blk, mx, out, ws, seq_lens=None):
    blk.execute(
        mx["h"],
        mx["w_qkvg"],
        mx["w_q_norm"],
        mx["w_k_norm"],
        mx["cos"],
        mx["sin"],
        mx["w_o"],
        out,
        ws,
        seq_lens=seq_lens,
        h_sf=mx["h_sf"],
        w_qkvg_sf=mx["w_qkvg_sf"],
        w_o_sf=mx["w_o_sf"],
    )


def _run_fp4o_block(geom_kw, batch, seq_len, fmt, *, seq_lens=None, sentinel: bool = False, **blk_kw):
    """Declare / check / compile / execute the fp4-O block; return ``(out, oracle, bf16_ref, blk, mx, spec)``."""
    geom = GatedAttentionBlockGeometry(**geom_kw)
    rg = mx_suite._ref_geom(geom)
    fused = bool(blk_kw.get("fuse_gate")) and bool(blk_kw.get("fuse_norm_rope"))
    inp, mx, desc = _fp4o_inputs(geom_kw, batch, seq_len, fmt)
    spec = MxQuantSpec(**desc, o_fp4=fmt)
    ref = gated_attention_block_mxfp8_reference(mx, rg, descale_w_o=1.0, scale_o=1.0, seq_lens=seq_lens, fused=fused, o_fp4=fmt)
    bf16_ref = gated_attention_block_reference(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], rg, seq_lens=seq_lens
    ).out
    out = torch.empty(batch, seq_len, geom.d_model, device="cuda", dtype=torch.bfloat16)
    if sentinel:
        out.fill_(_SENTINEL)
    blk = GatedAttentionBlockFwd(
        mx["h"],
        mx["w_qkvg"],
        mx["w_q_norm"],
        mx["w_k_norm"],
        mx["cos"],
        mx["sin"],
        mx["w_o"],
        out,
        geom,
        quant=spec,
        seq_lens_present=seq_lens is not None,
        sample_h_sf=mx["h_sf"],
        sample_w_qkvg_sf=mx["w_qkvg_sf"],
        sample_w_o_sf=mx["w_o_sf"],
        **blk_kw,
    )
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute_fp4o(blk, mx, out, ws, seq_lens=seq_lens)
    torch.cuda.synchronize()
    return out, ref, bf16_ref, blk, mx, spec


def _skip_fused_if_twin_missing(fused: bool) -> None:
    if fused and mx_suite._mxfp8_fork_missing() is not None:
        pytest.skip(f"fully fused MXFP8 needs the GEMM fork twin (PR-B S6): {mx_suite._mxfp8_fork_missing()}")


@requires_rubin
@requires_fp4
@pytest.mark.parametrize("seq_len, causal", [(256, True), (1000, True), (1024, False)])
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_fp4_o_block_matches_the_fake_quant_oracle(fmt, fused, seq_len, causal):
    """Rows 8 and 10, both formats: sentinel-filled output fully written, finite, ``cos > 0.99`` vs the fake-quant oracle (which
    quantizes the bf16 gated O to the format and dequantizes THROUGH the blobs), ``max_rel`` printed, and the FORMAT FLOOR
    ``cos(kernel, bf16 ref) >= cos(oracle, bf16 ref) - 0.005`` with the oracle's own cosine printed -- so an fp4 O's cost against
    the bf16 block is a number in the log, not a tolerance.  Fused at B=1 for S=1000 (the fused path declines ``S % 128 != 0``
    at B > 1); a second execute is bitwise.  Row 10 is the FIRST place the gated MXFP8 SDPA's bf16-O arm runs inside the block:
    if this fails numerically ONLY on the fused arm, row 10 becomes a typed decline (plan section 8, commit 5)."""
    _skip_fused_if_twin_missing(fused)
    b = 1 if (fused and seq_len % 128) else 2
    out, ref, bf16_ref, blk, mx, spec = _run_fp4o_block({**_GEOM, "is_causal": causal}, b, seq_len, fmt, sentinel=True, **(_FUSED if fused else {}))
    assert [s.name for s in blk._stages] == (_FP4_O_FUSED_STAGES if fused else _FP4_O_UNFUSED_STAGES)
    assert (
        blk._out_proj._plan.block_scale
        and blk._out_proj._plan.dtype == blk._out_proj._plan.w_dtype == _FP4
        and blk._out_proj._plan.block_size == fmt.block_size
    )
    tp = blk._sdpa._impl.template_params()
    assert tp.epilogue_gate is fused and blk._sdpa._impl.dtype_o == torch.bfloat16 and blk._sdpa._impl._pertensor is False
    assert not (out == _SENTINEL).any(), f"{(out == _SENTINEL).sum().item()} output cells were never written"
    assert torch.isfinite(out.float()).all()
    c = mx_suite._cos(out, ref)
    rel = ((out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-30)).item()
    c_k, c_o = mx_suite._cos(out, bf16_ref), mx_suite._cos(ref, bf16_ref)
    print(
        f"\n{fmt.name} O {'FUSED' if fused else 'unfused'} block B={b} S={seq_len} causal={causal}: cos(kernel, oracle)={c:.6f} max_rel={rel:.3e} "
        f"| format floor cos(oracle, bf16)={c_o:.6f} cos(kernel, bf16)={c_k:.6f} | out_proj route={blk._out_proj._plan.route}"
    )
    assert c > 0.99, f"{fmt.name} O block cos {c}"
    assert c_k >= c_o - 0.005, f"the kernel sits below the fp4 format floor: cos(kernel, bf16)={c_k} < cos(oracle, bf16)={c_o} - 0.005"
    out2 = torch.full_like(out, _SENTINEL)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute_fp4o(blk, mx, out2, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(out2, out, rtol=0, atol=0)


@requires_rubin
@requires_fp4
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_fp4_o_dead_padded_entry_is_exactly_zero(fmt, fused):
    """sdpa-invariants S1/S2 through the fp4-O block: ``seq_lens=[s, 0]`` -> the SDPA SELECTS ``O := 0`` on the dead entry, the
    gate multiplies an exact zero, the fp4 quantizer sees amax 0 (MXFP4: SF byte ``0x00``; NVFP4: the ``2^-9`` floor) and writes
    code 0 everywhere, and the block-scale GEMM propagates it -- ``out[1] == 0`` EXACTLY.  The live entry stays on the oracle.
    Fresh-process intermittency counts: the untracked dead-entry sweep's fp4 arms."""
    _skip_fused_if_twin_missing(fused)
    s = 512
    seq_lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    out, ref, _, blk, _, _ = _run_fp4o_block({**_GEOM, "is_causal": False}, 2, s, fmt, seq_lens=seq_lens, sentinel=True, **(_FUSED if fused else {}))
    assert blk._sdpa.seq_lens_present and blk.mxfp8_fused == fused
    assert not (out == _SENTINEL).any()
    assert torch.isfinite(out.float()).all(), "dead rows leaked a non-finite value into the output"
    assert (out[1] == 0).all(), f"the dead entry must be EXACTLY zero (select, not residue * sigmoid); max|out[1]| = {out[1].abs().max().item()}"
    c = mx_suite._cos(out[0], ref[0])
    print(f"\n{fmt.name} O {'fused' if fused else 'unfused'} block dead entry S={s}: live cos={c:.6f}")
    assert c > 0.99, f"live entry cos {c}"


@requires_rubin
@requires_fp4
@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_fp4_o_ragged_padded_entries_match_the_oracle(fmt):
    """Dense S=1000 with ``seq_lens=[1000, 700]`` through the fp4 tail (unfused; the fused pipeline declines S % 128 != 0 at
    B > 1): the quantizer's pad rows / the GEMM's TMA-clipped padded M rows do not disturb the live rows -- cosine per ENTRY."""
    s = 1000
    seq_lens = torch.tensor([s, 700], device="cuda", dtype=torch.int32)
    out, ref, _, blk, _, _ = _run_fp4o_block({**_GEOM, "is_causal": False}, 2, s, fmt, seq_lens=seq_lens, sentinel=True)
    assert blk._sdpa.seq_lens_present and not blk.mxfp8_fused
    assert not (out == _SENTINEL).any() and torch.isfinite(out.float()).all()
    cs = [mx_suite._cos(out[i], ref[i]) for i in range(2)]
    print(f"\n{fmt.name} O block ragged padding S={s} seq_lens={seq_lens.tolist()}: cos per entry={[f'{c:.6f}' for c in cs]}")
    assert all(c > 0.99 for c in cs), f"ragged padded entries cos {cs}"


@requires_rubin
@requires_fp4
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("fmt", _FMTS, ids=_FMT_IDS)
def test_fp4_o_launch_count_is_nine_unfused_and_four_fused(fmt, fused):
    """One CUDA kernel per stage and NO hidden launch: 9 unfused (``quantize_fp4_o`` replaces ``quantize_o`` one-for-one), 4 fused
    (the fp4 quantize is the ONE launch the fp4 O adds to the 3-launch fused pipeline), no memset (amax folded out), no memcpy.
    Profiled on the SECOND execute; typed skip where CUPTI records nothing (the Rubin dev / perf nodes)."""
    from torch.profiler import ProfilerActivity, profile

    _skip_fused_if_twin_missing(fused)
    out, _, _, blk, mx, _ = _run_fp4o_block(_GEOM, 1, 512, fmt, **(_FUSED if fused else {}))
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute_fp4o(blk, mx, out, ws)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _execute_fp4o(blk, mx, out, ws)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    memsets = [n for n in names if "memset" in n.lower()]
    memcpys = [n for n in names if "memcpy" in n.lower()]
    kernels = [n for n in names if n not in memsets and n not in memcpys]
    print(f"\n{fmt.name} O {'fused' if fused else 'unfused'} CUDA events:\n  " + "\n  ".join(names))
    assert len(kernels) == len(blk._stages) == (4 if fused else 9), (len(kernels), kernels)
    assert not memcpys, f"a hidden copy on the execute path: {memcpys}"
    assert not memsets, f"unexpected memset(s) on the execute path: {memsets}"
