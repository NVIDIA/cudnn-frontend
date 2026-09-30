# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FP8 (E4M3, static per-tensor scales) block against a fake-quantized fp32 oracle.

The oracle applies the SAME quantization points with the SAME scales the block
uses (exact saturating e4m3 casts, dequantized back to fp32), so the only
differences left are the kernels' internal roundings: bf16 slab/O rounding
(replicated for the UNFUSED pipeline; skipped for the FULLY FUSED one, which
rounds once per output) and the FP8 SDPA's fp8 cast of P (not replicated --
that is why the gate is a cosine, as in the FROST FP8 SDPA suite's d256
tolerance).

Two FP8 configurations exist and both are covered here: UNFUSED (7 stages = 9
kernel launches; the Q/K/V quantize stage is three launches) and FULLY FUSED
(``fuse_norm_rope=True, fuse_gate=True``, 3 launches: the
projection fork quantizes Q/K/V in its epilogue into COMPACT per-tensor
``q8`` / ``k8`` / ``v8`` -- round 2; round 1's single strided slab is gone --
the production FP8 d256 SDPA reads them compact, gates in its ``epilogue_gate``
and writes e4m3 O).  They are NOT bit-identical by design; each is scored
against its own oracle variant, never against the other bitwise.  The fully
fused tests SKIP (with the reason) while the FP8 projection fork -- or the
round-2 runner ABI (``out_k8``) -- has not landed.

Padding (``seq_lens_present``) is SERVED under FP8 since 2026-09-15 (the Rubin FP8
d256 SDPA's empty-KV-entry hang is gone); the dead-entry test asserts ``out[dead]``
is EXACTLY zero on both pipelines, the same contract the bf16 block is held to.
"""

import os

import pytest
import torch
from cuda.bindings import driver as cuda

pytestmark = pytest.mark.L0


import sys  # noqa: E402

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry  # noqa: E402
from cudnn.gated_attention_block.api import MxQuantSpec, QuantSpec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    RefGeometry,
    amax_scale,
    dequant_e4m3,
    gated_attention_block_fp8_reference,
    make_fp8_inputs,
    make_inputs,
    qk_norm_rope_reference,
    quant_e4m3,
    quantize_block_inputs_mxfp8,
)

_SM107 = (10, 7)
E4M3 = torch.float8_e4m3fn
FMAX = 448.0
_SENTINEL = 1.5e30  # a finite magnitude no correct output cell can hold; survivors localize an unwritten region
# qk_norm arms: RoPE-only Q/K (None norm weights, no rstd) rides the SAME FP8 pipelines.
_QK_NORM = pytest.mark.parametrize("qk_norm", [True, False], ids=["norm", "rope_only"])


def _fork_supports_qk_norm() -> bool:
    """True once ``NormRopeFusionParams.qk_norm`` exists (PR-B slice S2); until then the
    fully fused block declines ``qk_norm=False`` with a typed NotImplementedError."""
    from cudnn.gated_attention_block.api import _FusedQkvProjection

    return _FusedQkvProjection._fork_supports_qk_norm()


# The FP8 projection fork + the runner ABI the fully fused block needs.  Its
# absence is a SKIP with the reason, not an error: this file lands with the
# wiring, the fork lands in parallel.  (The SDPA side is the production
# `sdpa/fwd/kernels/sm107/prefill_d256_fp8.py` behind `epilogue_gate` -- always present.)
_FP8_FORK_FILES = ("proj_gemm_norm_rope_fp8.py",)
_FUSED = dict(fuse_norm_rope=True, fuse_gate=True)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the block targets SM107 only; found {_cc()}")

_GEOM = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)


def _require_fp8_forks():
    """Skip unless the FP8 projection fork and the ROUND-2 FP8 runner ABI are in this checkout.

    The block's ``execute_fp8`` hands the runner compact ``out_q8, out_k8, out_v8, out_gate16``
    positionally; the round-1 runner took one ``out_qkv8`` slab in that position, so a
    checkout with the old signature must SKIP (not mis-bind, not error)."""
    import inspect

    import cudnn.gated_attention_block.kernels as kernels_pkg
    from cudnn.gated_attention_block.kernels import proj_gemm

    base = os.path.dirname(os.path.abspath(kernels_pkg.__file__))
    missing = [f for f in _FP8_FORK_FILES if not os.path.exists(os.path.join(base, f))]
    if missing:
        pytest.skip(f"FP8 fork kernel(s) not landed in kernels/: {missing}")
    fn = getattr(proj_gemm, "run_fused_proj_gemm_fp8", None)
    if fn is None:
        pytest.skip("kernels/proj_gemm.py has no run_fused_proj_gemm_fp8 yet")
    if "out_k8" not in inspect.signature(fn).parameters:
        pytest.skip("run_fused_proj_gemm_fp8 still has the round-1 slab ABI (no out_k8); the compact q8/k8/v8 GEMM fork has not landed")


def _amax_scale(x: torch.Tensor) -> float:
    return amax_scale(x)


def _quant(x: torch.Tensor, scale: float) -> torch.Tensor:
    return quant_e4m3(x, scale)


def _dequant(x8: torch.Tensor, descale: float) -> torch.Tensor:
    return dequant_e4m3(x8, descale)


def _make_fp8_inputs(geom_kw, batch, seq_len):
    """bf16 inputs from the shared generator, then h / W_qkvg / W_o quantized to e4m3 with amax scales."""
    return make_fp8_inputs(RefGeometry(**geom_kw), batch=batch, seq_len=seq_len)


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


def _fp8_oracle(inp, geom: GatedAttentionBlockGeometry, spec: QuantSpec, seq_lens=None, fused: bool = False) -> torch.Tensor:
    """fp32 chain with the block's exact quantization points (shared: ``reference.gated_attention_block_fp8_reference``)."""
    return gated_attention_block_fp8_reference(
        inp,
        _ref_geom(geom),
        descale_h=spec.descale_h,
        descale_w_qkvg=spec.descale_w_qkvg,
        descale_w_o=spec.descale_w_o,
        scale_q=spec.scale_q,
        scale_k=spec.scale_k,
        scale_v=spec.scale_v,
        scale_o=spec.scale_o,
        seq_lens=seq_lens,
        fused=fused,
    )


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _calibrated_spec(inp, desc, geom: GatedAttentionBlockGeometry, batch, seq_len) -> QuantSpec:
    """Static activation scales calibrated on this very data through the oracle's
    own pre-quant activations (offline calibration stand-in)."""
    spec0 = QuantSpec(**desc, scale_q=1.0, scale_k=1.0, scale_v=1.0, scale_o=1.0)
    t = batch * seq_len
    h32 = _dequant(inp["h"], spec0.descale_h).view(t, geom.d_model)
    proj = (h32 @ _dequant(inp["w_qkvg"], spec0.descale_w_qkvg).t()).to(torch.bfloat16)
    o_q, o_g, o_k, o_v = geom.qkvg_offsets
    q = proj[:, o_q : o_q + geom.h_q * geom.d_head].reshape(batch, seq_len, geom.h_q, geom.d_head)
    k = proj[:, o_k : o_k + geom.h_kv * geom.d_head].reshape(batch, seq_len, geom.h_kv, geom.d_head)
    v = proj[:, o_v : o_v + geom.h_kv * geom.d_head]
    qn, _ = qk_norm_rope_reference(q, inp["w_q_norm"], inp["cos"], inp["sin"], geom.rope_dim, geom.qk_norm_eps, qk_norm=geom.qk_norm)
    kn, _ = qk_norm_rope_reference(k, inp["w_k_norm"], inp["cos"], inp["sin"], geom.rope_dim, geom.qk_norm_eps, qk_norm=geom.qk_norm)
    return QuantSpec(**desc, scale_q=_amax_scale(qn), scale_k=_amax_scale(kn), scale_v=_amax_scale(v), scale_o=_amax_scale(v) * 0.5)


def _run_fp8_block(geom_kw, batch, seq_len, seq_lens=None, sentinel: bool = False, **blk_kw):
    geom = GatedAttentionBlockGeometry(**geom_kw)
    inp, desc = _make_fp8_inputs(geom_kw, batch, seq_len)
    spec = _calibrated_spec(inp, desc, geom, batch, seq_len)
    fused = bool(blk_kw.get("fuse_gate")) and bool(blk_kw.get("fuse_norm_rope"))
    ref = _fp8_oracle(inp, geom, spec, seq_lens=seq_lens, fused=fused)

    out = torch.empty(batch, seq_len, geom.d_model, device="cuda", dtype=torch.bfloat16)
    if sentinel:
        out.fill_(_SENTINEL)
    blk = GatedAttentionBlockFwd(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        out,
        geom,
        quant=spec,
        seq_lens_present=seq_lens is not None,
        **blk_kw,
    )
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, seq_lens=seq_lens)
    torch.cuda.synchronize()
    return out, ref, blk


# ---------------------------------------------------------------------------
# Contract (no GPU kernel needed)
# ---------------------------------------------------------------------------


def _decl_block(**kw):
    geom = GatedAttentionBlockGeometry(**_GEOM)
    dev = "cuda"
    bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)  # noqa: E731
    h8 = torch.zeros(1, 256, geom.d_model, dtype=E4M3, device=dev)
    w8 = torch.zeros(geom.n_qkvg, geom.d_model, dtype=E4M3, device=dev)
    wo8 = torch.zeros(geom.d_model, geom.h_q * geom.d_head, dtype=E4M3, device=dev)
    return GatedAttentionBlockFwd(h8, w8, bf(256), bf(256), bf(1, 256, 64), bf(1, 256, 64), wo8, bf(1, 256, geom.d_model), geom, **kw)


_SPEC = QuantSpec(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=1.0, scale_k=2.0, scale_v=3.0, scale_o=4.0)


def test_fp8_h_without_quantspec_is_refused():
    with pytest.raises(ValueError, match="both halves"):
        _decl_block()


def test_quantspec_with_bf16_h_is_refused():
    geom = GatedAttentionBlockGeometry(**_GEOM)
    inp = make_inputs(RefGeometry(**_GEOM), batch=1, seq_len=256, dtype=torch.bfloat16)
    out = torch.empty(1, 256, geom.d_model, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="both halves"):
        GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom, quant=_SPEC)


def test_fp8_declines_training():
    """FP8 is inference-only (no q_pre/k_pre/pre-gate O contract under quantization)."""
    with pytest.raises(NotImplementedError, match="inference-only"):
        _decl_block(quant=_SPEC, save_for_backward=True, inplace_qkv=False)


def test_fp8_accepts_both_fusions():
    """INVERTED from ``test_fp8_declines_fusions_and_training`` (2026-09-11): the two fusion
    knobs are now ACCEPTED under FP8 -- together.  The block records the fully fused
    configuration and declares its SDPA gated (the production FP8 kernel's epilogue_gate)."""
    blk = _decl_block(quant=_SPEC, **_FUSED)
    assert blk.fuse_norm_rope and blk.fuse_gate and blk.fp8_fused
    assert blk._sdpa.fp8 and blk._sdpa.fuse_gate and blk._sdpa.o_dtype == E4M3
    assert blk._proj.fp8 and blk._proj.name == "qkv_gate_proj_norm_rope"


@pytest.mark.parametrize("kw", [dict(fuse_gate=True), dict(fuse_norm_rope=True)])
def test_fp8_fusions_are_both_or_neither(kw):
    """Under FP8 the block serves exactly two configurations -- unfused and fully
    fused.  A single knob is a typed decline that names BOTH knobs, so the caller
    learns which one to flip rather than getting a half-fused specialization
    nobody validated."""
    with pytest.raises(NotImplementedError, match="fuse_norm_rope") as ei:
        _decl_block(quant=_SPEC, **kw)
    assert "fuse_gate" in str(ei.value) and "fuse_norm_rope" in str(ei.value)


def test_quantspec_validation():
    with pytest.raises(ValueError, match="positive"):
        QuantSpec(descale_h=0.0, descale_w_qkvg=1.0, descale_w_o=1.0, scale_q=1.0, scale_k=1.0, scale_v=1.0, scale_o=1.0).validate()
    with pytest.raises(NotImplementedError, match="e4m3"):
        QuantSpec(descale_h=1.0, descale_w_qkvg=1.0, descale_w_o=1.0, scale_q=1.0, scale_k=1.0, scale_v=1.0, scale_o=1.0, dtype=torch.float8_e5m2).validate()
    assert _SPEC.alpha_qkvg == pytest.approx(0.01 * 0.02) and _SPEC.alpha_o == pytest.approx(0.25 * 0.03)


def test_fp8_rope_only_declares_the_same_stage_list_with_no_norm_weights():
    """``qk_norm=False`` under FP8: the norm is folded out of the qk_norm_rope
    kernel, not a stage removed -- same 7-launch list, None weights, no rstd."""
    geom = GatedAttentionBlockGeometry(**{**_GEOM, "qk_norm": False})
    dev = "cuda"
    bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)  # noqa: E731
    h8 = torch.zeros(1, 256, geom.d_model, dtype=E4M3, device=dev)
    w8 = torch.zeros(geom.n_qkvg, geom.d_model, dtype=E4M3, device=dev)
    wo8 = torch.zeros(geom.d_model, geom.h_q * geom.d_head, dtype=E4M3, device=dev)
    blk = GatedAttentionBlockFwd(h8, w8, None, None, bf(1, 256, 64), bf(1, 256, 64), wo8, bf(1, 256, geom.d_model), geom, quant=_SPEC)
    assert [s.name for s in blk._stages] == ["qkv_gate_proj", "qk_norm_rope", "quantize_q", "quantize_kv", "sdpa", "sigmoid_gate", "out_proj"]
    assert blk._norm_rope.want_rstd is False and blk._descs["w_q_norm"] is None
    with pytest.raises(ValueError, match="qk_norm=False"):
        GatedAttentionBlockFwd(h8, w8, bf(256), bf(256), bf(1, 256, 64), bf(1, 256, 64), wo8, bf(1, 256, geom.d_model), geom, quant=_SPEC)


def test_fp8_stage_list_and_workspace():
    blk = _decl_block(quant=_SPEC)
    assert [s.name for s in blk._stages] == ["qkv_gate_proj", "qk_norm_rope", "quantize_q", "quantize_kv", "sdpa", "sigmoid_gate", "out_proj"]
    lay = blk._layout()
    assert lay.q8 >= 0 and lay.k8 >= 0 and lay.v8 >= 0 and lay.o8 >= 0 and lay.q == -1 and lay.v == -1
    assert lay.gate16 == -1 and lay.proj >= 0 and lay.o >= 0
    assert not hasattr(lay, "qkv8"), "the round-1 qkv8 slab slot must not come back"
    # The unfused pipeline reads compact e4m3 Q/K/V at token_stride 0 through the shipped adapter, ungated, bf16 O.
    assert blk._sdpa.token_stride == 0 and not blk._sdpa.fuse_gate and blk._sdpa.o_dtype == torch.bfloat16


def test_fp8_fused_stage_list_and_workspace():
    """Fully fused (round 2): THREE stages and FIVE data slots -- COMPACT e4m3
    q8 / k8 / v8 (the unfused pipeline's own slots, so the SDPA reads exactly what
    the shipped FP8 SDPA reads), the bf16 GATE, the e4m3 O -- plus the 36-B quant
    slot LAST, and no bf16 slab, no bf16 O, no round-1 ``qkv8`` slab.  The expected
    total is derived from the geometry, never a literal."""
    from cudnn.gated_attention_block.api import _QUANT_SLOT_BYTES, _align_up

    blk = _decl_block(quant=_SPEC, **_FUSED)
    assert [s.name for s in blk._stages] == ["qkv_gate_proj_norm_rope", "sdpa", "out_proj"]
    assert blk._norm_rope is None and blk._gate is None and blk._quant_q is None and blk._quant_kv is None and blk._quant_o is None
    lay = blk._layout()
    assert lay.q8 >= 0 and lay.k8 >= 0 and lay.v8 >= 0 and lay.gate16 >= 0 and lay.o8 >= 0
    assert lay.proj == -1 and lay.o == -1 and lay.q == -1 and lay.k == -1 and lay.v == -1 and lay.gate == -1
    assert not hasattr(lay, "qkv8"), "the round-1 qkv8 slab slot must not come back"
    g = blk.geom
    t = blk.batch * blk.seq_len
    e = torch.finfo(torch.bfloat16).bits // 8
    q8_b, kv8_b, gate_b, o8_b = t * g.h_q * g.d_head, t * g.h_kv * g.d_head, t * g.h_q * g.d_head * e, t * g.h_q * g.d_head
    # Reserved in write order: q8, k8, v8, gate16, o8; each slot aligned; no gaps, no hidden slot.
    assert (lay.q8, lay.k8, lay.v8) == (0, _align_up(q8_b), _align_up(q8_b) + _align_up(kv8_b))
    assert lay.gate16 == lay.v8 + _align_up(kv8_b) and lay.o8 == lay.gate16 + _align_up(gate_b)
    want = _align_up(q8_b) + 2 * _align_up(kv8_b) + _align_up(gate_b) + _align_up(o8_b)
    assert lay.quant == want and lay.total_bytes == want + _align_up(_QUANT_SLOT_BYTES) == lay.engine_scratch, (lay.total_bytes, want)
    # q8 + k8 + v8 is the Q|K|V width per token -- the same e4m3 bytes round 1's slab held, per tensor now.
    assert q8_b + 2 * kv8_b == t * g.n_qkv and g.n_qkv == g.n_qkvg - g.h_q * g.d_head
    # -51 % of the unfused FP8 workspace (no bf16 slab, no bf16 O; the e4m3 slots are shared).
    unfused = _decl_block(quant=_SPEC)._layout()
    assert lay.total_bytes < 0.5 * unfused.total_bytes
    assert lay.total_bytes == unfused.total_bytes - _align_up(t * g.n_qkvg * e) - _align_up(t * g.h_q * g.d_head * e) + _align_up(gate_b)
    # The SDPA is the gated production FP8 kernel (fuse_gate), reading COMPACT Q/K/V (token_stride 0,
    # like the unfused path) and the compact bf16 GATE out of gate16, writing e4m3 O.
    assert blk._sdpa.fuse_gate and blk._sdpa.fp8
    assert blk._sdpa.token_stride == 0 and blk._sdpa.gate_token_stride == g.h_q * g.d_head and blk._sdpa.o_dtype == E4M3
    assert blk._sdpa.gate_dtype == torch.bfloat16


def test_fp8_fused_sdpa_is_the_production_adapter_gated_and_without_amax():
    """The fully fused block's SDPA stage is the shipped adapter with THREE things
    declared: ``pertensor_fp8`` + e4m3 ``dtype_o``, a COMPACT bf16 gate descriptor
    (the gate16 buffer, BSHD ``(S*H*D, H*D, D, 1)`` seen as BHSD), and
    ``has_amax_o=False`` -- the block runs static scales and reads no ``Amax_O``,
    so the kernel's atomicMax is compiled out.  The adapter's constructor touches
    no device, so this runs everywhere; no fork attribute survives on the stage."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    blk = _decl_block(quant=_SPEC, **_FUSED)
    g, b, s = blk.geom, blk.batch, blk.seq_len
    st = blk._sdpa
    assert [a for a in dir(st) if "fork" in a.lower()] == [], "the retired SDPA fork machinery must not survive on the stage"
    impl = st._build_impl()
    assert isinstance(impl, SdpaFwdDslSm100)
    assert impl._pertensor and impl.dtype_o == E4M3 and impl.has_amax_o is False
    assert impl.gate_desc is not None and impl.gate_desc.dtype == torch.bfloat16
    hd = g.h_q * g.d_head
    assert tuple(impl.gate_desc.shape) == (b, g.h_q, s, g.d_head) and tuple(impl.gate_desc.stride) == (s * hd, g.d_head, hd, 1)
    # The unfused FP8 SDPA is the same adapter, ungated, still without amax.
    impl_u = _decl_block(quant=_SPEC)._sdpa._build_impl()
    assert impl_u.gate_desc is None and impl_u.has_amax_o is False and impl_u.dtype_o == torch.bfloat16


# ---------------------------------------------------------------------------
# Numerics -- Rubin
# ---------------------------------------------------------------------------


@requires_rubin
@_QK_NORM
@pytest.mark.parametrize("seq_len, causal", [(256, True), (1000, True), (256, False), (1024, False)])
def test_fp8_block_matches_the_fake_quant_oracle(seq_len, causal, qk_norm):
    """Causal covers a KV tail (S=1000); the dense FP8 kernel needs S % 128 == 0 (see the decline test).
    ``rope_only``: None norm weights, the oracle skips the norm, same cosine bar."""
    out, ref, blk = _run_fp8_block({**_GEOM, "is_causal": causal, "qk_norm": qk_norm}, batch=2, seq_len=seq_len)
    assert blk._norm_rope._recipe.apply_norm is qk_norm
    assert torch.isfinite(out.float()).all()
    c = _cos(out, ref)
    rel = ((out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-30)).item()
    print(f"\nfp8 block S={seq_len} causal={causal} qk_norm={qk_norm}: cos={c:.6f} max_rel={rel:.3e}")
    assert c > 0.99, f"fp8 block cos {c}"


@requires_rubin
@_QK_NORM
@pytest.mark.parametrize("seq_len, causal", [(1000, True), (1024, False)])  # 1000: tail tile + rows past S in both forks' epilogues
def test_fp8_fused_block_matches_the_fake_quant_oracle(seq_len, causal, qk_norm):
    """The FULLY FUSED FP8 block (3 launches) against the fused-numerics oracle
    (one rounding per output), at the SAME cosine floor as the unfused test.
    Causal AND dense: the projection fork's and the gated SDPA's mask arms are
    const_expr-folded, so a dense PASS proves nothing about the causal arm and
    vice versa.  Also: sentinel survivors (an unwritten region of `out`), and a
    second execute bitwise (no per-execute compile/allocation; a cold-cache race probe).
    ``rope_only`` INVERTS while the FP8 fork lacks ``NormRopeFusionParams.qk_norm``
    (PR-B slice S2): a typed decline naming the knob, never a norm-ON artifact."""
    _require_fp8_forks()
    geom_kw = {**_GEOM, "is_causal": causal, "qk_norm": qk_norm}
    if not qk_norm and not _fork_supports_qk_norm():
        with pytest.raises(NotImplementedError, match="qk_norm"):
            _run_fp8_block(geom_kw, batch=2, seq_len=seq_len, sentinel=True, **_FUSED)
        return
    out, ref_fused, blk = _run_fp8_block(geom_kw, batch=2, seq_len=seq_len, sentinel=True, **_FUSED)
    assert blk.fp8_fused and len(blk._stages) == 3
    # The SDPA ran the production kernel's gated, amax-free specialization.
    assert blk._sdpa._impl.template_params().epilogue_gate is True and blk._sdpa._impl.has_amax_o is False
    assert not (out == _SENTINEL).any(), f"{(out == _SENTINEL).sum().item()} output cells were never written"
    assert torch.isfinite(out.float()).all()
    c = _cos(out, ref_fused)
    rel = ((out.float() - ref_fused.float()).abs().max() / ref_fused.float().abs().max().clamp_min(1e-30)).item()
    # Reported, not asserted: the unfused-numerics oracle (bf16 slab/O roundings) and
    # the unfused FP8 block are both "within quant noise" of the fused result by design.
    inp, desc = _make_fp8_inputs(geom_kw, 2, seq_len)
    geom = GatedAttentionBlockGeometry(**geom_kw)
    spec = _calibrated_spec(inp, desc, geom, 2, seq_len)
    c_unfused_oracle = _cos(out, _fp8_oracle(inp, geom, spec, fused=False))
    print(f"\nfp8 FUSED block S={seq_len} causal={causal}: cos={c:.6f} (vs unfused-numerics oracle {c_unfused_oracle:.6f}) max_rel={rel:.3e}")
    assert c > 0.99, f"fused fp8 block cos {c}"
    # Two-launch: same plan, fresh output + workspace, bit-identical.
    out2 = torch.full_like(out, _SENTINEL)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out2, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(out2, out, rtol=0, atol=0)


@requires_rubin
def test_fp8_fused_within_quant_noise_of_unfused():
    """Fused vs unfused FP8 blocks on the same inputs: close (cosine), NOT bit-identical
    -- the unfused path rounds to bf16 twice where the fused one rounds once.
    Asserting bit-identity here is the documented trap."""
    _require_fp8_forks()
    out_f, _, _ = _run_fp8_block(_GEOM, batch=1, seq_len=512, **_FUSED)
    out_u, _, _ = _run_fp8_block(_GEOM, batch=1, seq_len=512)
    c = _cos(out_f, out_u)
    print(f"\nfp8 fused vs unfused S=512 causal: cos={c:.6f}")
    assert c > 0.99, f"fused vs unfused fp8 cos {c}"


@requires_rubin
@pytest.mark.parametrize("kw", [{}, _FUSED], ids=["unfused", "fused"])
def test_fp8_dense_kv_tail_is_declined_not_computed_wrong(kw):
    """The Rubin FP8 SDPA leaves a dense KV tail unmasked, so the adapter DECLINES S % 128 != 0 without a
    padding mask or a causal mask -- the block must surface that at check_support, typed, not compute garbage.
    The gated path IS the adapter (with a gate descriptor), so the fused block declines identically."""
    if kw:
        _require_fp8_forks()
    with pytest.raises((ValueError, NotImplementedError), match="multiple of 128"):
        _run_fp8_block({**_GEOM, "is_causal": False}, batch=1, seq_len=1000, **kw)


@pytest.mark.parametrize("kw", [{}, _FUSED], ids=["unfused", "fused"])
def test_fp8_padding_mask_is_accepted(kw):
    """INVERTED from ``test_fp8_padding_mask_is_declined_because_the_kernel_hangs_on_an_empty_entry``
    (2026-09-15): the Rubin FP8 d256 SDPA's empty-KV-entry hang is gone on the rebased tree (8/8 fresh
    processes at S=1000 and S=512 with ``seq_kv_lens=[S, 0]``), so ``seq_lens_present`` is SERVED under FP8
    -- unfused AND fully fused -- and the declaration records it on the SDPA stage.  The dead-entry
    contract itself is ``test_fp8_dead_padded_entry_is_exactly_zero`` below."""
    blk = _decl_block(quant=_SPEC, seq_lens_present=True, **kw)
    assert blk.seq_lens_present and blk._sdpa.seq_lens_present
    assert blk._sdpa._build_impl().seq_kv_lens_present is True


@requires_rubin
@pytest.mark.parametrize("kw", [{}, _FUSED], ids=["unfused", "fused"])
def test_fp8_dead_padded_entry_is_exactly_zero(kw):
    """sdpa-invariants S1/S2 through the FP8 block: batch entry 1 has NO valid KV column
    (``seq_lens=[s, 0]``), so the FP8 SDPA runs zero KV iterations and must SELECT ``O := 0`` --
    the gate (stage (5), or the gated kernel's epilogue AFTER the select on the fused path) then
    multiplies an exact zero, ``quantize_o`` / the out projection propagate it, and ``out[1]`` is
    EXACTLY zero: never accumulator residue times a sigmoid, never a floored denominator.  The
    live entry stays on the fake-quant oracle.  Asserted on the OUTPUT directly (a diff against a
    reference that is itself NaN on the dead rows proves nothing).  Unfused dense S=512 needs no
    causal cover (padding carries the lengths)."""
    if kw:
        _require_fp8_forks()
    s = 512
    seq_lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    out, ref, blk = _run_fp8_block({**_GEOM, "is_causal": False}, batch=2, seq_len=s, seq_lens=seq_lens, sentinel=True, **kw)
    assert blk._sdpa.seq_lens_present
    assert not (out == _SENTINEL).any(), f"{(out == _SENTINEL).sum().item()} output cells were never written"
    assert torch.isfinite(out.float()).all(), "dead rows leaked a non-finite value into the output"
    assert (out[1] == 0).all(), f"the dead entry must be EXACTLY zero (select, not residue * sigmoid); max|out[1]| = {out[1].abs().max().item()}"
    c = _cos(out[0], ref[0])
    print(f"\nfp8 {'fused' if kw else 'unfused'} block dead entry S={s}: live cos={c:.6f}")
    assert c > 0.99, f"live entry cos {c}"


@requires_rubin
def test_fp8_block_second_execute_agrees_bitwise():
    out, _, blk = _run_fp8_block(_GEOM, batch=1, seq_len=512)
    # rerun into a fresh output with the same plan
    inp, _ = _make_fp8_inputs(_GEOM, 1, 512)  # same seed path -> same inputs
    out2 = torch.empty_like(out)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out2, ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(out2, out, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Rule 8: the quant scalars live in the workspace, filled at execute (R4 + R2); the plan owns no device memory
# ---------------------------------------------------------------------------


def _quant_words(spec: QuantSpec) -> list:
    """The nine fp32 words of the quant slot in ``_QUANT_WORDS`` order (``descale_* = 1 / scale_*``)."""
    return [spec.alpha_qkvg, spec.scale_q, spec.scale_k, spec.scale_v, spec.alpha_o, spec.scale_o, 1.0 / spec.scale_q, 1.0 / spec.scale_k, 1.0 / spec.scale_v]


def _bf16_block() -> GatedAttentionBlockFwd:
    geom = GatedAttentionBlockGeometry(**_GEOM)
    inp = make_inputs(RefGeometry(**_GEOM), batch=1, seq_len=256, dtype=torch.bfloat16)
    out = torch.empty(1, 256, geom.d_model, device="cuda", dtype=torch.bfloat16)
    return GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom)


@pytest.mark.parametrize("kw", [{}, _FUSED], ids=["unfused", "fused"])
def test_quant_slot_is_the_last_workspace_slot(kw):
    """Declaration-time arithmetic, any device: the nine-word slot (36 B, one 256-B alignment unit) is reserved
    LAST in both FP8 layouts, so every pre-existing offset is unchanged and the carve grows by exactly 256 B; a
    bf16 block reserves none (``quant == -1``) and has no values to fill."""
    import dataclasses

    from cudnn.gated_attention_block.api import _QUANT_SLOT_BYTES, _QUANT_WORDS, _align_up

    assert _QUANT_WORDS == ("alpha_qkvg", "scale_q", "scale_k", "scale_v", "alpha_o", "scale_o", "descale_q", "descale_k", "descale_v")
    assert _QUANT_SLOT_BYTES == 4 * 9 and _align_up(_QUANT_SLOT_BYTES) == 256
    lay = _decl_block(quant=_SPEC, **kw)._layout()
    assert lay.quant >= 0 and lay.quant % 256 == 0
    assert lay.total_bytes == lay.engine_scratch == lay.quant + 256
    others = [v for k, v in dataclasses.asdict(lay).items() if k not in ("quant", "total_bytes", "engine_scratch", "base_align") and v >= 0]
    assert max(others) < lay.quant, "the quant slot is appended after every data slot"
    bf16 = _bf16_block()
    assert bf16._layout().quant == -1 and bf16._quant_values() is None


@pytest.mark.parametrize("kw", [{}, _FUSED], ids=["unfused", "fused"])
def test_fill_quant_slot_writes_the_nine_words_on_the_given_stream(kw):
    """R4 on any GPU: ``_fill_quant_slot`` writes the nine fp32 words with driver memsets on the stream it is
    given -- no torch op, no host sync, no allocation -- touches nothing outside the 36 B, and returns 1-element
    views of the words (plus ``qscal`` = words 0..3, 16-B aligned) that alias the workspace."""
    from cudnn.gated_attention_block.api import _QUANT_SLOT_BYTES, _QUANT_WORDS

    blk = _decl_block(quant=_SPEC, **kw)
    lay = blk._layout()
    blk._ws = lay
    ws = torch.full((lay.total_bytes,), 0xFF, dtype=torch.uint8, device="cuda")
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        qd = blk._fill_quant_slot(ws, side.cuda_stream)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    side.synchronize()
    want = torch.tensor(_quant_words(_SPEC), dtype=torch.float32)
    got = ws[lay.quant : lay.quant + _QUANT_SLOT_BYTES].view(torch.float32).cpu()
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert blk._quant_values() == dict(zip(_QUANT_WORDS, _quant_words(_SPEC)))
    assert set(qd) == set(_QUANT_WORDS) | {"qscal"}
    for i, name in enumerate(_QUANT_WORDS):
        v = qd[name]
        assert v.dtype == torch.float32 and tuple(v.shape) == (1,) and v.data_ptr() == ws.data_ptr() + lay.quant + 4 * i
    qs = qd["qscal"]
    assert tuple(qs.shape) == (4,) and qs.is_contiguous() and qs.data_ptr() == ws.data_ptr() + lay.quant and qs.data_ptr() % 16 == 0
    torch.testing.assert_close(qs.cpu(), want[:4], rtol=0, atol=0)
    assert (ws[: lay.quant] == 0xFF).all() and (ws[lay.quant + _QUANT_SLOT_BYTES :] == 0xFF).all(), "only the nine words were written"


def test_quant_scalars_are_not_plan_owned():
    """The contract this batch changes: ``compile()`` owns no device scalars (no ``_quant_dev`` /
    ``_make_quant_dev``), ``_FusedQkvProjection`` takes no ``device`` and ``execute_fp8`` takes ``qscal`` as a
    required keyword (the block's workspace view)."""
    import inspect

    from cudnn.gated_attention_block.api import _FusedQkvProjection

    blk = _decl_block(quant=_SPEC, **_FUSED)
    assert not hasattr(blk, "_quant_dev") and not hasattr(GatedAttentionBlockFwd, "_make_quant_dev")
    assert "device" not in inspect.signature(_FusedQkvProjection.__init__).parameters and not hasattr(blk._proj, "_qscal")
    p = inspect.signature(_FusedQkvProjection.execute_fp8).parameters
    assert p["qscal"].kind is inspect.Parameter.KEYWORD_ONLY and p["qscal"].default is inspect.Parameter.empty


def _declare_fp8_block(geom_kw, batch, seq_len, **blk_kw):
    """A DECLARED (not checked, not compiled) FP8 block on calibrated data; returns ``(blk, args, oracle)``
    with ``args`` the positional execute arguments through ``out``."""
    geom = GatedAttentionBlockGeometry(**geom_kw)
    inp, desc = _make_fp8_inputs(geom_kw, batch, seq_len)
    spec = _calibrated_spec(inp, desc, geom, batch, seq_len)
    fused = bool(blk_kw.get("fuse_gate")) and bool(blk_kw.get("fuse_norm_rope"))
    ref = _fp8_oracle(inp, geom, spec, fused=fused)
    out = torch.empty(batch, seq_len, geom.d_model, device="cuda", dtype=torch.bfloat16)
    args = (inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out)
    return GatedAttentionBlockFwd(*args, geom, quant=spec, **blk_kw), args, ref


def _declare_mxfp8_fused_block(geom_kw, batch, seq_len):
    """A DECLARED fully fused MXFP8 block (``scale_o == 1.0``, D8); returns ``(blk, args, execute kwargs)``."""
    geom = GatedAttentionBlockGeometry(**geom_kw)
    mx, desc = quantize_block_inputs_mxfp8(make_inputs(RefGeometry(**geom_kw), batch=batch, seq_len=seq_len, dtype=torch.bfloat16))
    out = torch.empty(batch, seq_len, geom.d_model, device="cuda", dtype=torch.bfloat16)
    args = (mx["h"], mx["w_qkvg"], mx["w_q_norm"], mx["w_k_norm"], mx["cos"], mx["sin"], mx["w_o"], out)
    blk = GatedAttentionBlockFwd(*args, geom, quant=MxQuantSpec(**desc, scale_o=1.0), sample_h_sf=mx["h_sf"], sample_w_qkvg_sf=mx["w_qkvg_sf"], **_FUSED)
    return blk, args, dict(h_sf=mx["h_sf"], w_qkvg_sf=mx["w_qkvg_sf"])


def _check_support_or_skip(blk) -> None:
    """A fork that has not landed in this checkout is a SKIP with the block's own reason, not a failure."""
    try:
        blk.check_support()
    except NotImplementedError as exc:
        if "not landed" in str(exc):
            pytest.skip(str(exc))
        raise


@requires_rubin
@pytest.mark.parametrize("kw", [{}, _FUSED], ids=["unfused", "fused"])
def test_quant_slot_is_filled_on_launch_stream_and_matches_spec(kw):
    """After an execute on a side stream (ambient AND ``current_stream``), the quant slot holds the nine words
    of the block's ``QuantSpec`` and the alignment tail behind it is untouched; the output is bit-identical to
    the default-stream run (the fills are ordered with the kernels that read them)."""
    from cudnn.gated_attention_block.api import _QUANT_SLOT_BYTES

    if kw:
        _require_fp8_forks()
    out_ref, ref, blk = _run_fp8_block(_GEOM, batch=1, seq_len=512, **kw)
    inp, _ = _make_fp8_inputs(_GEOM, 1, 512)  # same seed path -> same inputs
    ws = torch.full((blk.get_workspace_size(),), 0xFF, dtype=torch.uint8, device="cuda")
    out = torch.empty_like(out_ref)
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        blk.execute(
            inp["h"],
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            out,
            ws,
            current_stream=cuda.CUstream(side.cuda_stream),
        )
    side.synchronize()
    lay = blk._ws
    got = ws[lay.quant : lay.quant + _QUANT_SLOT_BYTES].view(torch.float32).cpu()
    torch.testing.assert_close(got, torch.tensor(_quant_words(blk.quant), dtype=torch.float32), rtol=0, atol=0)
    assert (ws[lay.quant + _QUANT_SLOT_BYTES : lay.total_bytes] == 0xFF).all(), "the slot's alignment tail must stay untouched"
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    assert _cos(out, ref) > 0.99


@requires_rubin
@pytest.mark.parametrize("cfg", ["fp8_unfused", "fp8_fused", "mxfp8_fused"])
def test_execute_allocates_nothing_and_never_synchronizes(cfg, compile_allocates_nothing):
    """R9 around BUILD and EXECUTE (Rule 8): ``compile()`` makes no torch allocation (the quant scalars are no
    longer plan-owned tensors) and three warm executes allocate nothing and never block the host."""
    if cfg == "mxfp8_fused":
        blk, args, exec_kw = _declare_mxfp8_fused_block(_GEOM, 1, 512)
    else:
        if cfg == "fp8_fused":
            _require_fp8_forks()
        blk, args, _ = _declare_fp8_block(_GEOM, 1, 512, **(_FUSED if cfg == "fp8_fused" else {}))
        exec_kw = {}
    _check_support_or_skip(blk)
    compile_allocates_nothing(blk)
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    blk.execute(*args, ws, **exec_kw)  # warm: the adapter's cached dummies, plan caches
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            blk.execute(*args, ws, **exec_kw)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before, "execute() allocated (Rule 8 / R9)"
    assert torch.isfinite(args[-1].float()).all()


@requires_rubin
@pytest.mark.parametrize("kw", [{}, _FUSED], ids=["unfused", "fused"])
def test_first_execute_inside_capture(kw):
    """Rule 8's capture detector: ``compile()``, then the FIRST execute of the plan runs inside ``torch.cuda.graph``
    (the quant fills are memset nodes; the graph-route handle already exists), and the replay is bit-identical to
    an eager execute of the same plan and stable across replays."""
    if kw:
        _require_fp8_forks()
    blk, args, ref = _declare_fp8_block(_GEOM, 1, 512, **kw)
    _check_support_or_skip(blk)
    blk.compile()
    out = args[-1]
    ws = torch.zeros(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        blk.execute(*args, ws)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()
    assert torch.isfinite(replayed.float()).all(), "the captured first execute replayed a non-finite output"
    out2 = torch.empty_like(out)
    blk.execute(*args[:-1], out2, torch.zeros_like(ws))
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, out2, rtol=0, atol=0)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, replayed, rtol=0, atol=0)
    assert _cos(replayed, ref) > 0.99
