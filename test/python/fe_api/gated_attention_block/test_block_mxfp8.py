# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 block (E4M3 codes + per-32-block E8M0 scales, cuDNN F8_128x4) against a fake-quant oracle.

The caller hands the block PRE-quantized ``h`` / ``W_qkvg`` codes plus their PADDED
F8_128x4 scale-factor blobs (``sample_h_sf`` / ``sample_w_qkvg_sf`` at declaration,
``h_sf`` / ``w_qkvg_sf`` at execute -- ``kernels.proj_gemm.sf_blob_bytes`` bytes each)
and an :class:`MxQuantSpec`.  The oracle (``reference.gated_attention_block_mxfp8_reference``)
applies the SAME quantization points -- block-dequant of h / W THROUGH the blobs the
kernels read, Q / K block-quantized ROWWISE along D, V COLUMNWISE along S on the
S-padded tensor, per-tensor O and W_o (PR-B D1) -- so what is left is the kernels'
internal rounding (bf16 slab / O on the UNFUSED pipeline) and the MXFP8 SDPA's
unit-scale e4m3 P (not replicated).  Hence a cosine gate (0.99 floor, as the FP8
block; ``max_rel`` printed), never bit-identity.

Two MXFP8 configurations exist and both are covered end to end here.  UNFUSED (9 stages
= 9 kernel launches: block-scale GEMM, norm+RoPE, THREE ``quantize_mxfp8`` launches, the
production ``sm107/prefill_d256_mxfp8.py``, gate, per-tensor ``quantize_o``, FP8
``out_proj``) and FULLY FUSED (``fuse_norm_rope=True, fuse_gate=True``, 3 launches through
the MXFP8 GEMM fork twin ``run_fused_proj_gemm_mxfp8`` and the gated production MXFP8
SDPA).  The fused oracle tests SKIP typed on a checkout without the twin; the two
fused-only DECLINE branches (twin missing, adapter without ``sample_gate``) are pinned
DETERMINISTICALLY by monkeypatching the feature detection, so they stay red-capable on
every checkout (engine-contract rule 2: a decline that no test exercises can silently
turn into an untyped escape).

Padding (``seq_lens_present``) is served on BOTH pipelines: the dead-entry test
(``seq_lens=[S, 0]`` -> ``out[dead] == 0`` EXACTLY) runs unfused and fused, and the
ragged case (dense S=1000 with ``seq_lens=[1000, 700]``, plan D6) exercises the
quantizer's 0x00 pad-tile SF through the block's ``quantize_mxfp8 -> SDPA`` chain.
The >= 12-fresh-process intermittency evidence for the admission is
``frost_dev/sweep_block_dead_entry.py`` (cited in ``api.py``), not this file.

The SF-order S-sweep is the detector for the two silent-wrong-answer layouts of
``mma-tma-matrix.md`` S7: a cosine that DEGRADES with S is V's D-plane-major scale
factors read per-tile; a constant offset is a Q/K atom bug.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.L0


import sys  # noqa: E402

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, MxQuantSpec, QuantSpec  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import (  # noqa: E402
    RefGeometry,
    gated_attention_block_mxfp8_reference,
    make_inputs,
    mx_swizzle_sf_rowwise_padded,
    mx_unswizzle_sf_rowwise,
    mxfp8_calibrated_scale_o,
    quantize_block_inputs_mxfp8,
)

_SM107 = (10, 7)
E4M3 = torch.float8_e4m3fn
_SENTINEL = 1.5e30  # a finite magnitude no correct output cell can hold; survivors localize an unwritten region
_FUSED = dict(fuse_norm_rope=True, fuse_gate=True)
_GEOM = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)
# The unfused MXFP8 stage list, in pipeline order (FROZEN: block_perf_table / block_fusion_table code against it).
_MX_STAGES = ["qkv_gate_proj", "qk_norm_rope", "quantize_mxfp8_q", "quantize_mxfp8_k", "quantize_mxfp8_v", "sdpa", "sigmoid_gate", "quantize_o", "out_proj"]


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the block targets SM107 only; found {_cc()}")


def _mxfp8_fork_missing():
    """The reason the fully fused MXFP8 block declines on this checkout, or None once S6 has landed."""
    from cudnn.gated_attention_block.api import _FusedQkvProjection

    st = _FusedQkvProjection.__new__(_FusedQkvProjection)
    return st._mxfp8_fork_available()


def _ref_geom(geom: GatedAttentionBlockGeometry) -> RefGeometry:
    return RefGeometry(
        d_model=geom.d_model,
        h_q=geom.h_q,
        h_kv=geom.h_kv,
        d_head=geom.d_head,
        rope_dim=geom.rope_dim,
        qk_norm_eps=geom.qk_norm_eps,
        attn_scale=geom.attn_scale,
        is_causal=geom.is_causal,
        qk_norm=geom.qk_norm,
    )


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _mx_inputs(geom_kw, batch, seq_len):
    """bf16 inputs from the shared generator -> e4m3 codes + PADDED F8_128x4 blobs for h / W_qkvg, per-tensor e4m3 W_o."""
    inp = make_inputs(RefGeometry(**geom_kw), batch=batch, seq_len=seq_len, dtype=torch.bfloat16)
    mx, desc = quantize_block_inputs_mxfp8(inp)
    return inp, mx, desc


def _run_mx_block(geom_kw, batch, seq_len, seq_lens=None, sentinel: bool = False, scale_o=None, **blk_kw):
    """Declare / check / compile / execute the MXFP8 block; return ``(out, oracle, blk, mx, spec)``."""
    geom = GatedAttentionBlockGeometry(**geom_kw)
    rg = _ref_geom(geom)
    inp, mx, desc = _mx_inputs(geom_kw, batch, seq_len)
    fused = bool(blk_kw.get("fuse_gate")) and bool(blk_kw.get("fuse_norm_rope"))
    if scale_o is None:
        # D8: the fully fused path writes e4m3 O UNSCALED -> scale_o must be 1.0; the
        # unfused path takes a static per-tensor scale (calibrated here on the oracle's O).
        scale_o = 1.0 if fused else mxfp8_calibrated_scale_o(mx, rg, seq_lens=seq_lens)
    spec = MxQuantSpec(**desc, scale_o=scale_o)
    ref = gated_attention_block_mxfp8_reference(mx, rg, descale_w_o=spec.descale_w_o, scale_o=spec.scale_o, seq_lens=seq_lens, fused=fused)
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
        **blk_kw,
    )
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute(blk, mx, out, ws, seq_lens=seq_lens)
    torch.cuda.synchronize()
    return out, ref, blk, mx, spec


def _execute(blk, mx, out, ws, seq_lens=None):
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
    )


# ---------------------------------------------------------------------------
# Contract (no GPU kernel needed)
# ---------------------------------------------------------------------------

_SPEC = MxQuantSpec(descale_w_o=0.03)
_PT_SPEC = QuantSpec(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=1.0, scale_k=2.0, scale_v=3.0, scale_o=4.0)


def _decl_tensors(geom_kw=_GEOM, batch=1, seq_len=256):
    geom = GatedAttentionBlockGeometry(**geom_kw)
    dev = "cuda"
    bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)  # noqa: E731
    h8 = torch.zeros(batch, seq_len, geom.d_model, dtype=E4M3, device=dev)
    w8 = torch.zeros(geom.n_qkvg, geom.d_model, dtype=E4M3, device=dev)
    wo8 = torch.zeros(geom.d_model, geom.h_q * geom.d_head, dtype=E4M3, device=dev)
    hsf = torch.zeros(sf_blob_bytes(batch * seq_len, geom.d_model), dtype=torch.uint8, device=dev)
    wsf = torch.zeros(sf_blob_bytes(geom.n_qkvg, geom.d_model), dtype=torch.uint8, device=dev)
    return (
        geom,
        (h8, w8, bf(geom.d_head), bf(geom.d_head), bf(batch, seq_len, geom.rope_dim), bf(batch, seq_len, geom.rope_dim), wo8, bf(batch, seq_len, geom.d_model)),
        hsf,
        wsf,
    )


def _decl_block(geom_kw=_GEOM, **kw):
    geom, args, hsf, wsf = _decl_tensors(geom_kw)
    kw.setdefault("quant", _SPEC)
    kw.setdefault("sample_h_sf", hsf)
    kw.setdefault("sample_w_qkvg_sf", wsf)
    return GatedAttentionBlockFwd(*args, geom, **{k: v for k, v in kw.items() if v is not _ABSENT})


_ABSENT = object()


def test_both_spec_classes_are_exported():
    import cudnn.gated_attention_block as pkg

    assert pkg.QuantSpec is QuantSpec and pkg.MxQuantSpec is MxQuantSpec
    assert "QuantSpec" in pkg.__all__ and "MxQuantSpec" in pkg.__all__


def test_mxquantspec_validation():
    """Positive finite floats, E4M3 only (E5M2 is a typed decline), block 32, and D8 on the fused path."""
    with pytest.raises(ValueError, match="positive"):
        MxQuantSpec(descale_w_o=0.0).validate()
    with pytest.raises(ValueError, match="positive"):
        MxQuantSpec(descale_w_o=1.0, scale_o=float("inf")).validate()
    with pytest.raises(NotImplementedError, match="e4m3"):
        MxQuantSpec(descale_w_o=1.0, dtype=torch.float8_e5m2).validate()
    with pytest.raises(ValueError, match="block_size"):
        MxQuantSpec(descale_w_o=1.0, block_size=16).validate()
    MxQuantSpec(descale_w_o=1.0, scale_o=2.0).validate()  # unfused: any positive scale_o
    with pytest.raises(NotImplementedError, match="scale_o"):
        MxQuantSpec(descale_w_o=1.0, scale_o=2.0).validate(fused=True)
    MxQuantSpec(descale_w_o=1.0).validate(fused=True)
    assert MxQuantSpec(descale_w_o=0.03, scale_o=4.0).alpha_o == pytest.approx(0.25 * 0.03)
    assert MxQuantSpec(descale_w_o=0.03).scale_o == 1.0 and MxQuantSpec(descale_w_o=0.03).block_size == 32


def test_mxfp8_needs_all_three_halves():
    """e4m3 codes + MxQuantSpec + BOTH scale-factor blobs, or nothing of the kind -- every partial
    declaration is a typed ValueError that names the missing half."""
    with pytest.raises(ValueError, match="three halves"):
        _decl_block(sample_h_sf=_ABSENT, sample_w_qkvg_sf=_ABSENT)  # codes + MxQuantSpec, no blobs
    with pytest.raises(ValueError, match="three halves"):
        _decl_block(sample_w_qkvg_sf=_ABSENT)  # one blob only
    with pytest.raises(ValueError, match="three halves"):
        _decl_block(quant=_PT_SPEC)  # per-tensor QuantSpec + blobs
    geom, args, hsf, wsf = _decl_tensors()
    with pytest.raises(ValueError, match="both halves"):
        _decl_block(quant=_ABSENT)  # e4m3 codes, no spec at all
    bf16 = [a.to(torch.bfloat16) if a.dtype == E4M3 else a for a in args]
    with pytest.raises(ValueError, match="both halves"):
        GatedAttentionBlockFwd(*bf16, geom, quant=_SPEC, sample_h_sf=hsf, sample_w_qkvg_sf=wsf)  # MxQuantSpec + bf16 h
    with pytest.raises(TypeError, match="QuantSpec"):
        _decl_block(quant=object())


def test_mxfp8_declines_training_and_mixed_fusions():
    with pytest.raises(NotImplementedError, match="inference-only"):
        _decl_block(save_for_backward=True, inplace_qkv=False)
    for kw in (dict(fuse_gate=True), dict(fuse_norm_rope=True)):
        with pytest.raises(NotImplementedError, match="fuse_norm_rope") as ei:
            _decl_block(**kw)
        assert "fuse_gate" in str(ei.value) and "MXFP8" in str(ei.value)


def test_mxfp8_e5m2_codes_are_declined():
    with pytest.raises(NotImplementedError, match="e4m3"):
        _decl_block(quant=MxQuantSpec(descale_w_o=1.0, dtype=torch.float8_e5m2))


def test_mxfp8_scale_o_is_pinned_to_one_on_the_fused_path_only():
    """D8: the gated production MXFP8 SDPA writes e4m3 O UNSCALED, so a fully fused block refuses
    ``scale_o != 1.0`` AT DECLARATION; the unfused block applies any positive scale_o in quantize_o."""
    with pytest.raises(NotImplementedError, match="scale_o"):
        _decl_block(quant=MxQuantSpec(descale_w_o=1.0, scale_o=2.0), **_FUSED)
    blk = _decl_block(quant=MxQuantSpec(descale_w_o=1.0, scale_o=2.0))
    assert blk.quant.scale_o == 2.0 and blk.mxfp8 and not blk.mxfp8_fused


def test_mxfp8_padding_mask_is_accepted():
    """``seq_lens_present`` is served under MXFP8 (the d256 MXFP8 leading-zero-length-KV L0 test passes on
    Rubin); the dead-entry contract is ``test_mxfp8_dead_padded_entry_is_exactly_zero`` below."""
    blk = _decl_block(seq_lens_present=True)
    assert blk.seq_lens_present and blk._sdpa.seq_lens_present and blk._sdpa._build_impl().seq_kv_lens_present is True


def test_mxfp8_declines_d_model_not_a_multiple_of_128():
    """Whole F8_128x4 atoms along the contraction (4 blocks of 32): the v1 caller contract."""
    geom_kw = {**_GEOM, "d_model": 576}
    blk = _decl_block(geom_kw)  # the declaration is geometry-agnostic; check_support pins the rule
    with pytest.raises(NotImplementedError, match="d_model % 128"):
        blk.check_support()


def test_mxfp8_rejects_a_wrong_scale_factor_byte_count_and_dtype():
    """The blobs are PADDED F8_128x4 (``sf_blob_bytes``): one byte short, or a non-E8M0 dtype, is a typed ValueError
    at check_support, naming the expected size -- never a TMA over-read at execute."""
    geom, args, hsf, wsf = _decl_tensors()
    blk = GatedAttentionBlockFwd(*args, geom, quant=_SPEC, sample_h_sf=hsf[:-16], sample_w_qkvg_sf=wsf)
    with pytest.raises(ValueError, match="sf_blob_bytes"):
        blk.check_support()
    blk = GatedAttentionBlockFwd(*args, geom, quant=_SPEC, sample_h_sf=hsf, sample_w_qkvg_sf=torch.zeros(wsf.numel(), dtype=torch.int8, device="cuda"))
    with pytest.raises(ValueError, match="uint8"):
        blk.check_support()
    # the e8m0 view of the same bytes is accepted for the blob (the FROST GEMM's own scale dtype)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        blk = GatedAttentionBlockFwd(*args, geom, quant=_SPEC, sample_h_sf=hsf.view(e8m0), sample_w_qkvg_sf=wsf)
        blk._check_mxfp8_declaration()


def test_mxfp8_execute_requires_both_blobs_and_non_mx_refuses_them():
    """Both directions at execute, before any launch: MX needs h_sf AND w_qkvg_sf; a per-tensor FP8 / bf16
    block refuses them (a silently dropped scale factor is a wrong answer nobody reports)."""
    blk = _decl_block()
    blk._ws = blk._layout()  # bypass compile: the contract check runs first
    geom, args, hsf, wsf = _decl_tensors()
    ws = torch.empty(16, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="both scale-factor blobs"):
        blk.execute(*args, ws, h_sf=hsf)
    with pytest.raises(ValueError, match="sf_blob_bytes"):
        blk.execute(*args, ws, h_sf=hsf[:-16], w_qkvg_sf=wsf)
    pt = GatedAttentionBlockFwd(*args, geom, quant=_PT_SPEC)
    pt._ws = pt._layout()
    with pytest.raises(ValueError, match="without an MxQuantSpec"):
        pt.execute(*args, ws, h_sf=hsf, w_qkvg_sf=wsf)


def test_sdpa_stage_refuses_per_tensor_scalars_under_mx_and_blobs_otherwise():
    """``_Sdpa(mxfp8=True)`` REQUIRES sf_q/k/v and REFUSES descale_*/scale_o (the adapter's MXFP8 path would
    silently ignore the scalars); a per-tensor stage refuses the blobs.  Checked before the adapter is touched."""
    from cudnn.gated_attention_block.api import _Sdpa

    g = GatedAttentionBlockGeometry(**_GEOM)
    dev = torch.device("cuda")
    t = torch.empty(1, 256, g.h_q, g.d_head, device=dev, dtype=E4M3)
    kv = torch.empty(1, 256, g.h_kv, g.d_head, device=dev, dtype=E4M3)
    one = torch.ones(1, device=dev)
    sf = torch.zeros(16, dtype=torch.uint8, device=dev)

    class _NeverReached:
        def __getattr__(self, attr):
            pytest.fail(f"_Sdpa.execute touched the adapter (._impl.{attr}) before validating its own arguments")

    mx = _Sdpa(g, batch=1, seq_len=256, dtype=E4M3, device=dev, want_lse=False, mxfp8=True)
    pt = _Sdpa(g, batch=1, seq_len=256, dtype=E4M3, device=dev, want_lse=False)
    mx._impl = pt._impl = _NeverReached()
    assert mx.mxfp8 and mx.fp8 and not mx.pertensor and pt.pertensor and not pt.mxfp8
    with pytest.raises(ValueError, match="sf_q/sf_k/sf_v"):
        mx.execute(t, kv, kv, t)
    with pytest.raises(ValueError, match="per-tensor FP8 scalars"):
        mx.execute(t, kv, kv, t, sf_q=sf, sf_k=sf, sf_v=sf, descale_q=one)
    with pytest.raises(ValueError, match="per-tensor FP8 scalars"):
        mx.execute(t, kv, kv, t, sf_q=sf, sf_k=sf, sf_v=sf, scale_o=one)
    with pytest.raises(ValueError, match="MXFP8 scale-factor blobs"):
        pt.execute(t, kv, kv, t, descale_q=one, descale_k=one, descale_v=one, sf_q=sf)
    with pytest.raises(ValueError, match="mxfp8=True needs e4m3"):
        _Sdpa(g, batch=1, seq_len=256, dtype=torch.bfloat16, device=dev, want_lse=False, mxfp8=True)


def test_mxfp8_stage_list_and_workspace():
    """The frozen 9-stage list; quantize_o is an EXPLICIT per-tensor stage (never the rowwise MX recipe);
    the workspace carries the FP8 layout plus THREE SF slots of ``b*h*ceil(s/128)*(128*d/32)`` bytes,
    appended so every FP8 offset is unchanged.  Sizes derived, never literal."""
    from cudnn.gated_attention_block.api import _Quantize, _QuantizeMxfp8, _align_up, _sf_slot_bytes
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import sf_bytes

    blk = _decl_block()
    assert [s.name for s in blk._stages] == _MX_STAGES
    assert isinstance(blk._quant_q, _QuantizeMxfp8) and isinstance(blk._quant_k, _QuantizeMxfp8) and isinstance(blk._quant_v, _QuantizeMxfp8)
    assert (blk._quant_q.axis, blk._quant_k.axis, blk._quant_v.axis) == ("row", "row", "col")
    assert (blk._quant_q.heads, blk._quant_k.heads, blk._quant_v.heads) == (blk.geom.h_q, blk.geom.h_kv, blk.geom.h_kv)
    assert isinstance(blk._quant_o, _Quantize) and blk._quant_o is not blk._quant_q and blk._quant_o.name == "quantize_o"
    assert blk._quant_kv is None and blk._compact_v is None
    # the projection is the block-scale GEMM (no alpha); the out projection stays per-tensor (alpha_o)
    assert blk._proj.block_scale and not blk._proj.alpha and blk._out_proj.alpha and not blk._out_proj.block_scale
    lay = blk._layout()
    g, b, s = blk.geom, blk.batch, blk.seq_len
    for nm in ("sf_q", "sf_k", "sf_v", "q8", "k8", "v8", "o8", "proj", "o"):
        assert getattr(lay, nm) >= 0, nm
    assert lay.gate16 == -1 and lay.q == -1 and lay.v == -1
    q_sf, kv_sf = _sf_slot_bytes(b, g.h_q, s, g.d_head), _sf_slot_bytes(b, g.h_kv, s, g.d_head)
    assert q_sf == sf_bytes(b, g.h_q, s, g.d_head) == b * g.h_q * -(-s // 128) * (128 * g.d_head // 32)
    assert kv_sf == sf_bytes(b, g.h_kv, s, g.d_head) == blk._quant_k.sf_bytes() == blk._quant_v.sf_bytes()
    assert q_sf == blk._quant_q.sf_bytes()
    assert (lay.sf_k - lay.sf_q, lay.sf_v - lay.sf_k) == (_align_up(q_sf), _align_up(kv_sf))
    # == the FP8 unfused layout + the three slots (the FP8 offsets are byte-identical)
    geom, args, _, _ = _decl_tensors()
    fp8 = GatedAttentionBlockFwd(*args, geom, quant=_PT_SPEC)._layout()
    assert (lay.proj, lay.q8, lay.k8, lay.v8, lay.o, lay.o8) == (fp8.proj, fp8.q8, fp8.k8, fp8.v8, fp8.o, fp8.o8)
    assert lay.total_bytes == fp8.total_bytes + _align_up(q_sf) + 2 * _align_up(kv_sf) == lay.engine_scratch
    assert lay.sf_q == fp8.total_bytes
    # The SDPA stage is the production adapter on the BLOCK-SCALE kernel: pertensor_fp8=False, bf16 O, no amax.
    st = blk._sdpa
    assert st.mxfp8 and not st.pertensor and st.token_stride == 0 and not st.fuse_gate and st.o_dtype == torch.bfloat16
    impl = st._build_impl()
    assert impl._pertensor is False and impl.dtype_o == torch.bfloat16 and impl.has_amax_o is False and impl.gate_desc is None


def test_mxfp8_fused_stage_list_and_workspace():
    """FULLY FUSED (declaration only): three stages, the FP8-fused five slots + the three SF slots; the SDPA is the
    gated production adapter on the block-scale kernel with an e4m3 UNSCALED O and a compact bf16 gate."""
    from cudnn.gated_attention_block.api import _align_up, _sf_slot_bytes
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    blk = _decl_block(**_FUSED)
    assert blk.mxfp8_fused and blk.quant_fused and not blk.fp8_fused
    assert [s.name for s in blk._stages] == ["qkv_gate_proj_norm_rope", "sdpa", "out_proj"]
    assert blk._proj.mxfp8 and not blk._proj.fp8 and blk._quant_q is None and blk._quant_o is None and blk._gate is None
    assert "quant_mxfp8" in str(blk._proj.params())
    lay = blk._layout()
    g, b, s = blk.geom, blk.batch, blk.seq_len
    for nm in ("q8", "k8", "v8", "gate16", "o8", "sf_q", "sf_k", "sf_v"):
        assert getattr(lay, nm) >= 0, nm
    assert lay.proj == -1 and lay.o == -1
    geom, args, _, _ = _decl_tensors()
    fp8 = GatedAttentionBlockFwd(*args, geom, quant=_PT_SPEC, **_FUSED)._layout()
    assert (lay.q8, lay.k8, lay.v8, lay.gate16, lay.o8) == (fp8.q8, fp8.k8, fp8.v8, fp8.gate16, fp8.o8)
    q_sf, kv_sf = _sf_slot_bytes(b, g.h_q, s, g.d_head), _sf_slot_bytes(b, g.h_kv, s, g.d_head)
    assert lay.sf_q == fp8.total_bytes and lay.total_bytes == fp8.total_bytes + _align_up(q_sf) + 2 * _align_up(kv_sf)
    st = blk._sdpa
    assert st.mxfp8 and st.fuse_gate and st.o_dtype == E4M3 and st.gate_dtype == torch.bfloat16 and st.gate_token_stride == g.h_q * g.d_head
    impl = st._build_impl()
    assert isinstance(impl, SdpaFwdDslSm100)
    assert impl._pertensor is False and impl.dtype_o == E4M3 and impl.has_amax_o is False
    assert impl.gate_desc is not None and impl.gate_desc.dtype == torch.bfloat16
    hd = g.h_q * g.d_head
    assert tuple(impl.gate_desc.shape) == (b, g.h_q, s, g.d_head) and tuple(impl.gate_desc.stride) == (s * hd, g.d_head, hd, 1)


def _dispatch_trace(blk, monkeypatch):
    """Run ``blk.execute`` with every stage's launch replaced by a recorder; return the stage names in call order.

    The block's execute body is the thing under test -- the S5 bring-up defect was a stage silently SKIPPED by a
    sibling ``elif`` (GEMM exact, SDPA exact, end-to-end cos 0.73), which no per-stage test can see.  No kernel
    runs, so this holds on any CUDA device; the compiled-record facts the profiler test would add (one launch per
    stage, ``has_amax_o=False`` folding the amax memset out) are pinned on Rubin by the bitwise test."""
    lay = blk._layout()
    blk._ws = lay
    blk._is_supported = True
    monkeypatch.setattr(blk, "get_workspace_size", lambda: lay.total_bytes)
    q = blk.quant
    blk._quant_dev = dict(
        alpha_o=torch.full((1,), float(q.alpha_o), dtype=torch.float32, device="cuda"),
        scale_o=torch.full((1,), float(q.scale_o), dtype=torch.float32, device="cuda"),
    )
    calls = []
    for st in blk._stages:
        for meth in ("execute", "execute_mxfp8", "execute_fp8"):
            if hasattr(st, meth):
                monkeypatch.setattr(st, meth, lambda *a, _n=st.name, _m=meth, **k: calls.append((_n, _m)))
    _, args, hsf, wsf = _decl_tensors(batch=blk.batch, seq_len=blk.seq_len)
    ws = torch.zeros(lay.total_bytes, dtype=torch.uint8, device="cuda")
    blk.execute(*args, ws, h_sf=hsf, w_qkvg_sf=wsf)
    return calls


def test_mxfp8_unfused_execute_dispatches_every_stage_once_in_pipeline_order(monkeypatch):
    """CUPTI-free twin of ``test_mxfp8_unfused_launch_count``: the unfused MXFP8 execute body calls each of the
    NINE frozen stages exactly once, in pipeline order, through its plain ``execute`` -- in particular the shared
    norm+RoPE stage between the block-scale GEMM and the three quantize launches, and the EXPLICIT per-tensor
    ``quantize_o`` before ``out_proj``.  Each stage is one kernel launch (the quantize runner, the gate kernel,
    the adapter with the amax folded out), so 9 stage calls == 9 launches."""
    blk = _decl_block()
    calls = _dispatch_trace(blk, monkeypatch)
    assert [n for n, _ in calls] == _MX_STAGES
    assert all(m == "execute" for _, m in calls)


def test_mxfp8_fused_execute_dispatches_the_three_stages_through_the_mx_runner(monkeypatch):
    """The fully fused MXFP8 execute body: the projection through ``execute_mxfp8`` (the frozen SF-output runner ABI,
    never the per-tensor ``execute_fp8`` / bf16 ``execute``), then the gated SDPA, then ``out_proj`` -- three calls."""
    blk = _decl_block(**_FUSED)
    calls = _dispatch_trace(blk, monkeypatch)
    assert calls == [("qkv_gate_proj_norm_rope", "execute_mxfp8"), ("sdpa", "execute"), ("out_proj", "execute")]


@pytest.mark.parametrize("what", ["runner", "fork_file"])
def test_mxfp8_fused_declines_typed_when_the_fork_twin_is_missing(monkeypatch, what):
    """The fully fused MXFP8 block feature-detects the GEMM fork twin (the runner ``run_fused_proj_gemm_mxfp8``
    with the frozen SF-output ABI, and the fork file) and declines at check_support with a NotImplementedError
    naming what is missing.  On a checkout WITH the twin this branch is dead unless the detection is
    monkeypatched -- so it is, deterministically, for each of the two probes; a skip-until-landed pin
    would never run again once S6 landed (which it has)."""
    from cudnn.gated_attention_block.api import _FusedQkvProjection

    if what == "runner":
        monkeypatch.setattr(_FusedQkvProjection, "_MXFP8_RUNNER", "run_fused_proj_gemm_mxfp8_NOT_HERE")
    else:
        monkeypatch.setattr(_FusedQkvProjection, "_FORK_PATH_MXFP8", _FusedQkvProjection._FORK_PATH_MXFP8 + ".not_here")
    blk = _decl_block(**_FUSED)
    reason = _mxfp8_fork_missing()
    assert reason is not None
    with pytest.raises(NotImplementedError, match="fork twin") as ei:
        blk.check_support()
    msg = str(ei.value)
    assert reason in msg
    assert ("run_fused_proj_gemm_mxfp8" in msg) if what == "runner" else ("proj_gemm_norm_rope_mxfp8" in msg)
    # the same block ACCEPTS once the detection is restored (declaration is not the gate; check_support is)
    monkeypatch.undo()
    assert _mxfp8_fork_missing() is None, "the twin is expected in this checkout"
    if _cc() == _SM107:
        blk._proj.check_support()  # the fork's own cc gate follows the twin check, so only Rubin reaches the accept


def test_mxfp8_gated_sdpa_declines_typed_without_the_adapters_sample_gate(monkeypatch):
    """``_Sdpa(mxfp8=True, fuse_gate=True)`` reaches the block-scale kernel ONLY through the production adapter's
    ``sample_gate`` (S7); an adapter without it has no gated MXFP8 path and the stage must decline TYPED
    (never reach for the per-tensor gated fork, which cannot take block scales).  The branch is dead on this
    checkout, so the adapter's signature is monkeypatched to one without ``sample_gate``.  Geometry-level: the
    decline fires before the cc gate, so it reads the same on every device."""
    from cudnn.gated_attention_block.api import _Sdpa
    from cudnn.sdpa.fwd import api_dsl

    def _init_without_sample_gate(self, *args, **kwargs):  # pragma: no cover -- never called; only its signature is read
        raise AssertionError("the adapter must not be constructed on the decline path")

    monkeypatch.setattr(api_dsl.SdpaFwdDslSm100, "__init__", _init_without_sample_gate)
    g = GatedAttentionBlockGeometry(**_GEOM)
    st = _Sdpa(g, batch=1, seq_len=256, dtype=E4M3, device=torch.device("cuda"), want_lse=False, fuse_gate=True, gate_dtype=torch.bfloat16, mxfp8=True)
    with pytest.raises(NotImplementedError, match="sample_gate") as ei:
        st.check_support()
    assert "fuse_gate=False" in str(ei.value)
    # the UNGATED MXFP8 stage does not consult the gate hook at all
    st_plain = _Sdpa(g, batch=1, seq_len=256, dtype=E4M3, device=torch.device("cuda"), want_lse=False, mxfp8=True)
    if _cc() != _SM107:
        with pytest.raises(NotImplementedError, match="SM107"):
            st_plain.check_support()


def test_mxfp8_gate_reads_the_mxfp8_row_and_the_production_adapter():
    """The gated MXFP8 SDPA stage reads the MXFP8 row's claims (gate at (256, 256), bf16 G) -- not the per-tensor
    row's -- and reaches the block-scale kernel through the adapter's ``sample_gate`` only (no fork).  A head dim
    the MXFP8 row does not claim declines naming the head dim, on every device."""
    from cudnn.gated_attention_block.api import _Sdpa
    from cudnn.sdpa.fwd import engines

    g = GatedAttentionBlockGeometry(**_GEOM)
    st = _Sdpa(g, batch=1, seq_len=256, dtype=E4M3, device=torch.device("cuda"), want_lse=False, fuse_gate=True, gate_dtype=torch.bfloat16, mxfp8=True)
    assert [a for a in dir(st) if "fork" in a.lower()] == [], "no fork machinery on the stage"
    caps = st._row_capabilities("sm107", True, True)
    assert caps is next(sp.capabilities for sp in engines.ENGINE_SPECS if sp.name == engines.engine_name(arch="sm107", mxfp8=True))
    assert caps.is_mxfp8 and caps.epilogue_gate and (g.d_head, g.d_head) in caps.epilogue_gate_d_shapes
    st._check_gate_geometry()  # accepted
    with pytest.raises(NotImplementedError, match="gate_dtype"):
        _Sdpa(
            g, batch=1, seq_len=256, dtype=E4M3, device=torch.device("cuda"), want_lse=False, fuse_gate=True, gate_dtype=E4M3, mxfp8=True
        )._check_gate_geometry()
    g128 = GatedAttentionBlockGeometry(**{**_GEOM, "d_head": 128, "rope_dim": 64})
    with pytest.raises(NotImplementedError, match="128"):
        _Sdpa(
            g128, batch=1, seq_len=256, dtype=E4M3, device=torch.device("cuda"), want_lse=False, fuse_gate=True, gate_dtype=torch.bfloat16, mxfp8=True
        )._check_gate_geometry()


def test_mxfp8_sched_policy_is_natural_at_d256_read_off_the_mxfp8_row():
    """D5: the MXFP8 (256, 256) row claims NATURAL only, so the stage requests NATURAL explicitly -- the per-tensor
    FP8 row's LPT does not transfer, and ``sched_policy=None`` would let the adapter's auto knobs pick LPT.
    Read off the row at the same time so the two cannot drift (the row is the truth; this pins the reading)."""
    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
    from cudnn.gated_attention_block.api import _Sdpa
    from cudnn.sdpa.fwd import engines

    caps = next(sp.capabilities for sp in engines.ENGINE_SPECS if sp.name == engines.engine_name(arch="sm107", mxfp8=True))
    domain = dict(caps.sched_policies_by_d_shape).get((256, 256), caps.sched_policies)
    assert domain == frozenset({SCHED_NATURAL}), "the MXFP8 (256,256) row grew a policy: re-validate the block's request"
    blk = _decl_block()
    if _cc() == _SM107:
        assert blk._sdpa._sched_policy() == SCHED_NATURAL
        assert blk._sdpa._build_impl().sched_policy == SCHED_NATURAL


def test_sf_swizzle_round_trips_and_pads_to_whole_atoms():
    """The reference's F8_128x4 helpers (what builds the caller's blobs AND what the oracle dequantizes through):
    unswizzle(swizzle(e)) == e for ragged rows / blocks, and the blob is exactly ``sf_blob_bytes``."""
    for rows, k in ((1000, 512), (256, 544), (392, 512), (200, 256)):
        e = torch.randint(0, 255, (rows, k // 32), dtype=torch.uint8, device="cuda")
        blob = mx_swizzle_sf_rowwise_padded(e)
        assert blob.numel() == sf_blob_bytes(rows, k) and blob.dtype == torch.uint8 and blob.is_contiguous()
        assert torch.equal(mx_unswizzle_sf_rowwise(blob, rows, k), e)
    # pad rows / blocks are 0x00 (E8M0 2^-127) -- an unwritten byte would be 0xFF = E8M0 NaN
    e = torch.full((200, 8), 0x7F, dtype=torch.uint8, device="cuda")
    blob = mx_swizzle_sf_rowwise_padded(e)
    assert int((blob == 0x7F).sum()) == 200 * 8 and int((blob == 0).sum()) == blob.numel() - 200 * 8


# ---------------------------------------------------------------------------
# Numerics -- Rubin
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("seq_len, causal", [(256, True), (1000, True), (256, False), (1024, False)])
def test_mxfp8_block_matches_the_fake_quant_oracle(seq_len, causal):
    """Causal covers a KV tail (S=1000: the quantize stages write the pad rows' SF as 0x00 exactly like the
    torch oracle pads); dense needs S % 128 == 0 (see the decline test).  B=2 so V's D-plane-major SF stride
    (B*KH*n_tiles*512) is exercised past the (b=0, h=0) coincidence."""
    out, ref, blk, _, spec = _run_mx_block({**_GEOM, "is_causal": causal}, batch=2, seq_len=seq_len, sentinel=True)
    assert [s.name for s in blk._stages] == _MX_STAGES
    assert not (out == _SENTINEL).any(), f"{(out == _SENTINEL).sum().item()} output cells were never written"
    assert torch.isfinite(out.float()).all()
    c = _cos(out, ref)
    rel = ((out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-30)).item()
    print(f"\nmxfp8 block S={seq_len} causal={causal}: cos={c:.6f} max_rel={rel:.3e} scale_o={spec.scale_o:.4g} proj route={blk._proj._plan.route}")
    assert c > 0.99, f"mxfp8 block cos {c}"


@requires_rubin
def test_mxfp8_sf_order_s_sweep_does_not_degrade_with_s():
    """The mma-tma-matrix S7 detector at B=1, dense: a cosine DEGRADING with S is V's D-plane-major scale factors
    read per-tile (plane stride grows with S); a CONSTANT offset is a Q/K atom bug.  Every S must pass the gate
    AND the sweep must be flat."""
    cs = {}
    for s in (128, 256, 384, 512):
        out, ref, _, _, _ = _run_mx_block({**_GEOM, "is_causal": False}, batch=1, seq_len=s)
        cs[s] = _cos(out, ref)
    print("\nmxfp8 SF-order S-sweep (B=1 dense): " + "  ".join(f"S={s}: cos={c:.6f}" for s, c in cs.items()))
    assert all(c > 0.99 for c in cs.values()), cs
    assert max(cs.values()) - min(cs.values()) < 5e-3, f"cosine drifts with S -- an SF layout bug: {cs}"


@requires_rubin
@pytest.mark.parametrize("kw", [{}, _FUSED], ids=["unfused", "fused"])
def test_mxfp8_dead_padded_entry_is_exactly_zero(kw):
    """sdpa-invariants S1/S2 through the MXFP8 block, UNFUSED and FULLY FUSED: batch entry 1 has NO valid KV
    column (``seq_lens=[s, 0]``), so the MXFP8 SDPA runs zero KV iterations and must SELECT ``O := 0``; the gate
    (stage (5), or the gated kernel's epilogue AFTER the select on the fused path) multiplies an exact zero,
    quantize_o / out_proj (or the e4m3 O straight into out_proj) propagate it, and ``out[1]`` is EXACTLY zero.
    On the fused path this also drives the fork twin's per-(b, s_tile) SF stores for a fully-masked entry at
    B=2 and the block's ``seq_lens`` hand-off into the gated MXFP8 adapter.  Asserted on the OUTPUT (a diff
    against a reference that is itself NaN on the dead rows proves nothing).  The live entry stays on the
    oracle.  Fresh-process intermittency counts: ``frost_dev/sweep_block_dead_entry.py``."""
    if kw and _mxfp8_fork_missing() is not None:
        pytest.skip(f"fully fused MXFP8 needs the GEMM fork twin (PR-B S6): {_mxfp8_fork_missing()}")
    s = 512
    seq_lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    out, ref, blk, _, _ = _run_mx_block({**_GEOM, "is_causal": False}, batch=2, seq_len=s, seq_lens=seq_lens, sentinel=True, **kw)
    assert blk._sdpa.seq_lens_present and blk.mxfp8_fused == bool(kw)
    assert not (out == _SENTINEL).any()
    assert torch.isfinite(out.float()).all(), "dead rows leaked a non-finite value into the output"
    assert (out[1] == 0).all(), f"the dead entry must be EXACTLY zero (select, not residue * sigmoid); max|out[1]| = {out[1].abs().max().item()}"
    c = _cos(out[0], ref[0])
    print(f"\nmxfp8 {'fused' if kw else 'unfused'} block dead entry S={s}: live cos={c:.6f}")
    assert c > 0.99, f"live entry cos {c}"


@requires_rubin
def test_mxfp8_ragged_padded_entries_match_the_oracle():
    """Plan D6's distinguishing MXFP8 padding case, through the BLOCK's ``quantize_mxfp8 -> SDPA`` chain (the SDPA-level
    probe covers it alone): dense S=1000 (S % 128 != 0, admitted because a padding mask is present) with
    ``seq_lens=[1000, 700]`` -- a live ragged entry whose last KV tile is 104 rows + 24 pad rows, and a partially
    padded entry.  The quantizers write the pad rows' SF as 0x00 (they read those rows as 0), the SDPA reads
    whole 1024-B SF tiles and masks the padded KV columns by ``seq_lens``; every Q row of entry 1 attends its
    700 valid columns.  Cosine per ENTRY against the oracle with the same ``seq_lens``; unfused only (the fused
    pipeline declines S % 128 != 0 at B > 1)."""
    s = 1000
    seq_lens = torch.tensor([s, 700], device="cuda", dtype=torch.int32)
    out, ref, blk, _, spec = _run_mx_block({**_GEOM, "is_causal": False}, batch=2, seq_len=s, seq_lens=seq_lens, sentinel=True)
    assert blk._sdpa.seq_lens_present and not blk.mxfp8_fused
    assert not (out == _SENTINEL).any(), f"{(out == _SENTINEL).sum().item()} output cells were never written"
    assert torch.isfinite(out.float()).all()
    cs = [_cos(out[i], ref[i]) for i in range(2)]
    print(f"\nmxfp8 block ragged padding S={s} seq_lens={seq_lens.tolist()}: cos per entry={[f'{c:.6f}' for c in cs]} scale_o={spec.scale_o:.4g}")
    assert all(c > 0.99 for c in cs), f"ragged padded entries cos {cs}"


@requires_rubin
def test_mxfp8_block_second_execute_agrees_bitwise_and_runs_natural_cga1():
    """Same plan, fresh output + workspace, bit-identical (no per-execute compile / allocation; a cold-cache race
    probe); the compiled record is the MXFP8 row's NATURAL policy at cta_mma 1 (the adapter's quantized d256 rule),
    the block-scale kernel (pertensor False), amax folded out."""
    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL

    out, _, blk, mx, _ = _run_mx_block(_GEOM, batch=1, seq_len=512)
    tp = blk._sdpa._impl.template_params()
    assert tp.sched_policy == SCHED_NATURAL and tp.cta_mma == 1 and tp.epilogue_gate is False
    assert blk._sdpa._impl._pertensor is False and blk._sdpa._impl.has_amax_o is False
    # The compiled record folded the amax OUT (kernel carries `has_amax`, caller asked for none): the adapter then
    # binds None in the amax slot and issues NO per-execute `amax_o.zero_()` memset -- the CUPTI-free half of the
    # "9 launches, 0 memsets" claim (the profiler test needs a node where CUPTI works).
    assert getattr(blk._sdpa._impl, "_amax_folded_out", None) is True
    out2 = torch.full_like(out, _SENTINEL)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute(blk, mx, out2, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(out2, out, rtol=0, atol=0)


@requires_rubin
def test_mxfp8_unfused_launch_count():
    """One CUDA kernel per stage of the frozen list (9: the three quantize launches are three stages) and NO
    hidden launch: the adapter's per-execute ``amax_o.zero_()`` memset is gone under ``has_amax_o=False`` on the
    Rubin d256 kernel (the atomic is compiled out), and nothing else may allocate or copy.  Profiled on the
    SECOND execute (the first materialises the adapter's cached dummy operands)."""
    from torch.profiler import ProfilerActivity, profile

    out, _, blk, mx, _ = _run_mx_block(_GEOM, batch=1, seq_len=512)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute(blk, mx, out, ws)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _execute(blk, mx, out, ws)
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    names = [e.name for e in evs]
    if not names:
        # torch.profiler needs CUPTI; on the Rubin dev / perf nodes it reports
        # CUPTI_ERROR_INVALID_DEVICE and records NO device events (frost-gotchas: CUPTI
        # is unavailable on c09 too).  A typed skip, not a green pass on an empty list.
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    memsets = [n for n in names if "memset" in n.lower()]
    memcpys = [n for n in names if "memcpy" in n.lower()]
    kernels = [n for n in names if n not in memsets and n not in memcpys]
    print("\nmxfp8 unfused CUDA events:\n  " + "\n  ".join(names))
    assert len(kernels) == len(_MX_STAGES) == 9, (len(kernels), kernels)
    assert not memcpys, f"a hidden copy on the execute path: {memcpys}"
    assert len(memsets) == 0, f"unexpected memset(s) on the execute path (amax is folded out under has_amax_o=False): {memsets}"


@requires_rubin
@pytest.mark.parametrize("seq_len, causal", [(1000, True), (1024, False)])
def test_mxfp8_fused_block_matches_the_fake_quant_oracle(seq_len, causal):
    """The FULLY FUSED MXFP8 block (3 launches) against the fused-numerics oracle (one rounding per output, e4m3 O
    UNSCALED -- scale_o 1.0, D8), same cosine floor.  Typed SKIP on a checkout without the MXFP8 GEMM fork twin
    (feature-detected); the decline itself is pinned deterministically by
    ``test_mxfp8_fused_declines_typed_when_the_fork_twin_is_missing``.  B=1 at S=1000: the fused path declines
    ``S % 128 != 0 and B > 1``."""
    missing = _mxfp8_fork_missing()
    if missing is not None:
        pytest.skip(f"fully fused MXFP8 needs the GEMM fork twin (PR-B S6): {missing}")
    b = 1 if seq_len % 128 else 2
    out, ref, blk, mx, spec = _run_mx_block({**_GEOM, "is_causal": causal}, batch=b, seq_len=seq_len, sentinel=True, **_FUSED)
    assert blk.mxfp8_fused and len(blk._stages) == 3 and spec.scale_o == 1.0
    assert blk._sdpa._impl.template_params().epilogue_gate is True and blk._sdpa._impl._pertensor is False
    assert not (out == _SENTINEL).any(), f"{(out == _SENTINEL).sum().item()} output cells were never written"
    assert torch.isfinite(out.float()).all()
    c = _cos(out, ref)
    rel = ((out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-30)).item()
    print(f"\nmxfp8 FUSED block B={b} S={seq_len} causal={causal}: cos={c:.6f} max_rel={rel:.3e}")
    assert c > 0.99, f"fused mxfp8 block cos {c}"
    out2 = torch.full_like(out, _SENTINEL)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute(blk, mx, out2, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(out2, out, rtol=0, atol=0)


def test_mxfp8_fused_declines_ragged_s_at_b_gt_1():
    """Plan 3.1 / Q9: the fork's SF stores are decoded from the flat GEMM row, so an M-tile straddling two sequences
    is declined (typed) rather than scattered; the unfused pipeline serves the shape.  Geometry-level, so it reads
    the same on every device -- and it fires BEFORE the fork-missing decline, so it holds after S6 lands too."""
    geom, args, _, _ = _decl_tensors(batch=2, seq_len=1000)
    hsf = torch.zeros(sf_blob_bytes(2000, geom.d_model), dtype=torch.uint8, device="cuda")
    wsf = torch.zeros(sf_blob_bytes(geom.n_qkvg, geom.d_model), dtype=torch.uint8, device="cuda")
    b = GatedAttentionBlockFwd(*args, geom, quant=_SPEC, sample_h_sf=hsf, sample_w_qkvg_sf=wsf, **_FUSED)
    with pytest.raises(NotImplementedError, match="S % 128 == 0 when B > 1"):
        b.check_support()


@requires_rubin
def test_mxfp8_dense_kv_tail_is_declined_not_computed_wrong():
    """The Rubin MXFP8 SDPA leaves a dense KV tail unmasked, so the adapter DECLINES S % 128 != 0 without a padding
    mask or a causal mask -- the block surfaces that at check_support, typed."""
    with pytest.raises((ValueError, NotImplementedError), match="multiple of 128"):
        _run_mx_block({**_GEOM, "is_causal": False}, batch=1, seq_len=1000)
