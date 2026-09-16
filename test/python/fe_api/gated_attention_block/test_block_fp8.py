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

Two FP8 configurations exist and both are covered here: UNFUSED (7 launches)
and FULLY FUSED (``fuse_norm_rope=True, fuse_gate=True``, 3 launches: the
projection fork quantizes Q/K/V in its epilogue into COMPACT per-tensor
``q8`` / ``k8`` / ``v8`` -- round 2; round 1's single strided slab is gone --
the production FP8 d256 SDPA reads them compact, gates in its ``epilogue_gate``
and writes e4m3 O).  They are NOT bit-identical by design; each is scored
against its own oracle variant, never against the other bitwise.  The fully
fused tests SKIP (with the reason) while the FP8 projection fork -- or the
round-2 runner ABI (``out_k8``) -- has not landed.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.L0


import sys  # noqa: E402

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry  # noqa: E402
from cudnn.gated_attention_block.api import QuantSpec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import (  # noqa: E402
    RefGeometry,
    amax_scale,
    dequant_e4m3,
    gated_attention_block_fp8_reference,
    make_fp8_inputs,
    make_inputs,
    qk_norm_rope_reference,
    quant_e4m3,
)

_SM107 = (10, 7)
E4M3 = torch.float8_e4m3fn
FMAX = 448.0
_SENTINEL = 1.5e30  # a finite magnitude no correct output cell can hold; survivors localize an unwritten region

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
    qn, _ = qk_norm_rope_reference(q, inp["w_q_norm"], inp["cos"], inp["sin"], geom.rope_dim, geom.qk_norm_eps)
    kn, _ = qk_norm_rope_reference(k, inp["w_k_norm"], inp["cos"], inp["sin"], geom.rope_dim, geom.qk_norm_eps)
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
    """Fully fused (round 2): THREE stages and FIVE workspace slots -- COMPACT e4m3
    q8 / k8 / v8 (the unfused pipeline's own slots, so the SDPA reads exactly what
    the shipped FP8 SDPA reads), the bf16 GATE, the e4m3 O -- and no bf16 slab,
    no bf16 O, no round-1 ``qkv8`` slab.  The expected total is derived from the
    geometry, never a literal."""
    from cudnn.gated_attention_block.api import _align_up

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
    assert lay.total_bytes == want == lay.engine_scratch, (lay.total_bytes, want)
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
@pytest.mark.parametrize("seq_len, causal", [(256, True), (1000, True), (256, False), (1024, False)])
def test_fp8_block_matches_the_fake_quant_oracle(seq_len, causal):
    """Causal covers a KV tail (S=1000); the dense FP8 kernel needs S % 128 == 0 (see the decline test)."""
    out, ref, blk = _run_fp8_block({**_GEOM, "is_causal": causal}, batch=2, seq_len=seq_len)
    assert torch.isfinite(out.float()).all()
    c = _cos(out, ref)
    rel = ((out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-30)).item()
    print(f"\nfp8 block S={seq_len} causal={causal}: cos={c:.6f} max_rel={rel:.3e}")
    assert c > 0.99, f"fp8 block cos {c}"


@requires_rubin
@pytest.mark.parametrize("seq_len, causal", [(1000, True), (1024, False)])  # 1000: tail tile + rows past S in both forks' epilogues
def test_fp8_fused_block_matches_the_fake_quant_oracle(seq_len, causal):
    """The FULLY FUSED FP8 block (3 launches) against the fused-numerics oracle
    (one rounding per output), at the SAME cosine floor as the unfused test.
    Causal AND dense: the projection fork's and the gated SDPA's mask arms are
    const_expr-folded, so a dense PASS proves nothing about the causal arm and
    vice versa.  Also: sentinel survivors (an unwritten region of `out`), and a
    second execute bitwise (no per-execute compile/allocation; a cold-cache race probe)."""
    _require_fp8_forks()
    geom_kw = {**_GEOM, "is_causal": causal}
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
def test_fp8_padding_mask_is_declined_because_the_kernel_hangs_on_an_empty_entry(kw):
    """The Rubin FP8 d256 SDPA hangs on seq_kv_lens == 0 (probe_sdpa_fp8_d256.py --lsepad-dead: S=1000 first
    launch, S=512 second launch).  A hang is worse than a decline, so the block refuses padding under FP8
    at declaration -- unfused AND fully fused (the gate epilogue does not touch the kernel's
    empty-mainloop path).  INVERT this test (assert the dead entry is exactly 0, as the bf16 test does)
    when the kernel's empty-mainloop path is fixed; until then no FP8 dead-row test can exercise
    gate-after-select on the gated FP8 kernel."""
    with pytest.raises(NotImplementedError, match="hangs on an empty"):
        _decl_block(quant=_SPEC, seq_lens_present=True, **kw)


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
