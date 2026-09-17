# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stages (2)+(3) — the fused QK-RMSNorm + partial RoPE FROST kernel.

The kernel is plain vectorized LDG/STG plus one butterfly shuffle: no tcgen05,
no TMA, no arch-specific path. So unlike stage (4) it runs anywhere CuTe DSL
does, and these tests deliberately do NOT gate on Rubin — catching a lane-group
or tail bug on whatever device is at hand is worth more than arch purity.

Every numerics test runs in two arms, ``norm`` and ``rope_only``
(``GatedAttentionBlockGeometry.qk_norm`` / ``compile_qk_norm_rope(apply_norm=)``):
the RoPE-only artifact traces no RMSNorm at all, takes ``None`` for both norm
weights, emits no rstd, and must copy the passthrough dims ``[rope_dim, D)``
BIT-EXACTLY (there is no fp32 op between load and store on them).
"""

import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gated_attention_block.kernels.qk_norm_rope import (
    QkNormRopeRecipe,
    build_qk_norm_rope,
    compile_qk_norm_rope,
    lanes_per_row,
    moved_bytes,
    run_qk_norm_rope,
    validate_shape,
    vec_chunks,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import qk_norm_rope_reference  # noqa: E402

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_EPS = 1e-6
_QK_NORM = pytest.mark.parametrize("qk_norm", [True, False], ids=["norm", "rope_only"])


def _weights(w_q, w_k, qk_norm):
    """The two norm-weight slots as the kernel takes them: tensors, or ``None`` for RoPE-only."""
    return (w_q, w_k) if qk_norm else (None, None)


def _make(t, h_q, h_kv, d, rope_dim, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32).to(dtype)

    q = rnd(t, h_q, d)
    k = rnd(t, h_kv, d)
    w_q = rnd(d)
    w_k = rnd(d)
    if rope_dim:
        half = rope_dim // 2
        ang = torch.randn(t, half, generator=g, device="cuda", dtype=torch.float32)
        emb = torch.cat((ang, ang), dim=-1)
        cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    else:
        cos = sin = torch.zeros(t, 1, device="cuda", dtype=dtype)
    return q, k, w_q, w_k, cos, sin


def _ref(x, w, cos, sin, rope_dim, qk_norm=True):
    """The [T, H, D] oracle: the reference takes [B, S, H, D], so borrow B=1.
    ``qk_norm=False`` returns ``(rope_only_y, None)``."""
    y, rstd = qk_norm_rope_reference(x[None], w, cos[None], sin[None], rope_dim, _EPS, qk_norm=qk_norm)
    return y[0], (None if rstd is None else rstd[0])


def _check(got, want, dtype):
    # bf16 has 8 mantissa bits; the kernel and the oracle differ only in fp32
    # op order, so a couple of ulps is the whole budget.
    atol = 8e-3 if dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(got.float(), want.float(), rtol=0, atol=atol)


# ---------------------------------------------------------------------------
# Shape algebra — no GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d, lanes, chunks", [(64, 8, 1), (128, 16, 1), (256, 32, 1), (512, 32, 2)])
def test_lane_partition(d, lanes, chunks):
    """One 16-byte access per lane per chunk, and a row never straddles a warp."""
    assert lanes_per_row(d) == lanes
    assert vec_chunks(d) == chunks
    assert lanes * chunks * 8 == d


@pytest.mark.parametrize(
    "d, rope_dim, threads, match",
    [
        (250, 64, 256, "multiple of 8"),
        (256, 24, 256, "multiple of 16"),
        (256, 96, 256, "power of two"),  # 96/16 = 6
        (256, 512, 256, "first access chunk"),
        (256, 64, 100, "multiple of the 32 lanes"),
    ],
)
def test_validate_shape_rejects(d, rope_dim, threads, match):
    with pytest.raises(ValueError, match=match):
        validate_shape(d, rope_dim, threads)


def test_moved_bytes_counts_a_read_and_a_write_of_q_and_k():
    t, h_q, h_kv, d = 8192, 32, 2, 256
    rows = t * (h_q + h_kv)
    assert moved_bytes(t, h_q, h_kv, d, want_rstd=False) == 2 * rows * d * 2
    assert moved_bytes(t, h_q, h_kv, d, want_rstd=True) == 2 * rows * d * 2 + rows * 4


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@requires_cuda
@_QK_NORM
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("t, h_q, h_kv, d, rope_dim", [(64, 8, 2, 256, 64), (64, 4, 1, 128, 32), (64, 2, 2, 64, 16)])
def test_matches_the_fused_oracle(dtype, t, h_q, h_kv, d, rope_dim, qk_norm):
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, dtype)
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    # rstd exists only where a norm exists: the RoPE-only artifact takes None.
    rstd_q = torch.empty(t, h_q, device="cuda", dtype=torch.float32) if qk_norm else None
    rstd_k = torch.empty(t, h_kv, device="cuda", dtype=torch.float32) if qk_norm else None
    wq, wk = _weights(w_q, w_k, qk_norm)

    r = build_qk_norm_rope(q, k, q_out, k_out, wq, wk, cos, sin, rstd_q, rstd_k, rope_dim=rope_dim, eps=_EPS, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert r.apply_norm is qk_norm and r.want_rstd is qk_norm

    want_q, want_rstd_q = _ref(q, wq, cos, sin, rope_dim, qk_norm)
    want_k, want_rstd_k = _ref(k, wk, cos, sin, rope_dim, qk_norm)
    _check(q_out, want_q, dtype)
    _check(k_out, want_k, dtype)
    if qk_norm:
        torch.testing.assert_close(rstd_q, want_rstd_q, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(rstd_k, want_rstd_k, rtol=1e-5, atol=1e-6)
    else:
        assert want_rstd_q is None and want_rstd_k is None
        # No fp32 op touches the passthrough dims on the RoPE-only path.
        assert torch.equal(q_out[..., rope_dim:], q[..., rope_dim:]) and torch.equal(k_out[..., rope_dim:], k[..., rope_dim:])


@requires_cuda
def test_rope_actually_rotates():
    """Guards the whole butterfly: if the shuffle partner or the sign were
    wrong the result would still be finite and plausible, so compare against a
    norm-only oracle and require the rope band to DIFFER while the passthrough
    band matches exactly."""
    t, h_q, h_kv, d, rope_dim = 32, 4, 2, 256, 64
    dtype = torch.bfloat16
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=7)
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    build_qk_norm_rope(q, k, q_out, k_out, w_q, w_k, cos, sin, rope_dim=rope_dim, eps=_EPS, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    norm_only, _ = _ref(q, w_q, cos, sin, 0)
    assert not torch.allclose(q_out[..., :rope_dim].float(), norm_only[..., :rope_dim].float(), atol=1e-2)
    _check(q_out[..., rope_dim:], norm_only[..., rope_dim:], dtype)


@requires_cuda
@_QK_NORM
def test_in_place_matches_out_of_place(qk_norm):
    """``q_out is q`` is the block's default: every lane reads its whole row
    before any lane stores, and the RoPE shuffle stays inside the row."""
    t, h_q, h_kv, d, rope_dim = 96, 8, 2, 256, 64
    dtype = torch.bfloat16
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=3)
    wq, wk = _weights(w_q, w_k, qk_norm)
    q_ref, k_ref = torch.empty_like(q), torch.empty_like(k)
    build_qk_norm_rope(q, k, q_ref, k_ref, wq, wk, cos, sin, rope_dim=rope_dim, eps=_EPS, stream=torch.cuda.current_stream().cuda_stream)
    q_ip, k_ip = q.clone(), k.clone()
    build_qk_norm_rope(q_ip, k_ip, q_ip, k_ip, wq, wk, cos, sin, rope_dim=rope_dim, eps=_EPS, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(q_ip, q_ref)
    torch.testing.assert_close(k_ip, k_ref)


@requires_cuda
@pytest.mark.parametrize("t", [1, 3, 13, 257])
def test_ragged_tail_rows(t):
    """A CTA covers 8 rows at d=256; these token counts leave a partial last
    CTA, whose clamped loads must not corrupt the rows that do exist."""
    h_q, h_kv, d, rope_dim, dtype = 8, 2, 256, 64, torch.bfloat16
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=5)
    q_out = torch.full_like(q, 1.5e3)
    k_out = torch.full_like(k, 1.5e3)
    # rstd ON: a ragged group also breaks the vectorized rstd store's
    # "R consecutive rows in one tensor" precondition, so this is the case that
    # exercises its scalar fallback and the Q/K-seam guard.
    rstd_q = torch.full((t, h_q), -1.0, device="cuda", dtype=torch.float32)
    rstd_k = torch.full((t, h_kv), -1.0, device="cuda", dtype=torch.float32)
    build_qk_norm_rope(q, k, q_out, k_out, w_q, w_k, cos, sin, rstd_q, rstd_k, rope_dim=rope_dim, eps=_EPS, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    want_q, want_rq = _ref(q, w_q, cos, sin, rope_dim)
    want_k, want_rk = _ref(k, w_k, cos, sin, rope_dim)
    _check(q_out, want_q, dtype)
    _check(k_out, want_k, dtype)
    torch.testing.assert_close(rstd_q, want_rq, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rstd_k, want_rk, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_rstd_is_optional():
    """Inference asks for no rstd and must compile a kernel that never stores it."""
    t, h_q, h_kv, d, rope_dim, dtype = 64, 8, 2, 256, 64, torch.bfloat16
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, dtype)
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    build_qk_norm_rope(q, k, q_out, k_out, w_q, w_k, cos, sin, rope_dim=rope_dim, eps=_EPS, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    want_q, _ = _ref(q, w_q, cos, sin, rope_dim)
    _check(q_out, want_q, dtype)


# ---------------------------------------------------------------------------
# RoPE-only (qk_norm=False): the norm folded out at trace time
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("t", [64, 257])  # 257: a ragged last CTA on the RoPE-only path too
def test_rope_only_passthrough_dims_are_bit_exact(t):
    """With no RMSNorm there is no fp32 op between the load and the store of the
    dims ``[rope_dim, D)``: widen, narrow, same value -- so ``torch.equal``, not a
    tolerance. The rope band must still rotate (differ from the input, match the
    RoPE-only oracle), and a second launch is bit-identical to the first."""
    h_q, h_kv, d, rope_dim, dtype = 8, 2, 256, 64, torch.bfloat16
    q, k, _, _, cos, sin = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=11)
    q_out, k_out = torch.full_like(q, 1.5e3), torch.full_like(k, 1.5e3)
    stream = torch.cuda.current_stream().cuda_stream
    r = build_qk_norm_rope(q, k, q_out, k_out, None, None, cos, sin, rope_dim=rope_dim, eps=_EPS, stream=stream)
    torch.cuda.synchronize()
    assert r.apply_norm is False and r.want_rstd is False
    assert torch.equal(q_out[..., rope_dim:], q[..., rope_dim:]), "RoPE-only Q passthrough dims are not a bit-exact copy"
    assert torch.equal(k_out[..., rope_dim:], k[..., rope_dim:]), "RoPE-only K passthrough dims are not a bit-exact copy"
    want_q, none_q = _ref(q, None, cos, sin, rope_dim, qk_norm=False)
    want_k, _ = _ref(k, None, cos, sin, rope_dim, qk_norm=False)
    assert none_q is None
    _check(q_out, want_q, dtype)
    _check(k_out, want_k, dtype)
    assert not torch.allclose(q_out[..., :rope_dim].float(), q[..., :rope_dim].float(), atol=1e-2), "the rope band did not rotate"
    # Two launches, same plan: bit-identical (also a cheap cold-cache race probe).
    q2, k2 = torch.empty_like(q), torch.empty_like(k)
    run_qk_norm_rope(r, q, k, q2, k2, None, None, cos, sin, stream=stream)
    torch.cuda.synchronize()
    assert torch.equal(q2, q_out) and torch.equal(k2, k_out)


@requires_cuda
def test_rope_only_and_norm_are_distinct_cache_entries():
    """``apply_norm`` is IN the compile key: a cached norm-on artifact bound to
    None weights would dereference a null pointer, so the two must never alias."""
    kw = dict(dtype=torch.bfloat16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, want_rstd=False)
    on, off = compile_qk_norm_rope(**kw), compile_qk_norm_rope(**kw, apply_norm=False)
    assert on.apply_norm is True and off.apply_norm is False
    assert on.compiled is not off.compiled
    assert compile_qk_norm_rope(**kw, apply_norm=False).compiled is off.compiled, "the RoPE-only recipe must itself be a cache hit"


def test_rope_only_rejects_weights_and_rstd():
    """Both directions, typed, and BEFORE any compile (so this runs on every box):

    * a RoPE-only artifact cannot emit rstd (compile-time ``ValueError``);
    * RoPE-only with ``rope_dim=0`` is an identity copy -- refused, not launched;
    * at execute the weights must agree with the recipe: None on a norm-on
      recipe, tensors on a RoPE-only one, or one of the two missing, all raise;
    * rstd handed to a recipe compiled without it raises (it would be ignored).
    """
    base = dict(dtype=torch.bfloat16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS)
    with pytest.raises(ValueError, match="emits no rstd"):
        compile_qk_norm_rope(**base, want_rstd=True, apply_norm=False)
    with pytest.raises(ValueError, match="identity copy"):
        compile_qk_norm_rope(**{**base, "rope_dim": 0}, want_rstd=False, apply_norm=False)

    # Hand-built recipes: the checks run before `compiled` is ever touched.
    rope_only = QkNormRopeRecipe(compiled=None, h_q=8, h_kv=2, d=256, eps=_EPS, rows_per_cta=8, want_rstd=False, apply_norm=False)
    normed = QkNormRopeRecipe(compiled=None, h_q=8, h_kv=2, d=256, eps=_EPS, rows_per_cta=8, want_rstd=False)  # apply_norm defaults True
    assert normed.apply_norm is True
    x = torch.empty(4, 8, 256, dtype=torch.bfloat16)
    kx = torch.empty(4, 2, 256, dtype=torch.bfloat16)
    tab = torch.empty(4, 64, dtype=torch.bfloat16)
    w = torch.ones(256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="WITHOUT the RMSNorm"):
        run_qk_norm_rope(rope_only, x, kx, x, kx, w, w, tab, tab, stream=0)
    with pytest.raises(ValueError, match="WITH the RMSNorm"):
        run_qk_norm_rope(normed, x, kx, x, kx, None, None, tab, tab, stream=0)
    with pytest.raises(ValueError, match="together"):
        run_qk_norm_rope(normed, x, kx, x, kx, w, None, tab, tab, stream=0)
    with pytest.raises(ValueError, match="together"):
        run_qk_norm_rope(rope_only, x, kx, x, kx, None, w, tab, tab, stream=0)
    with pytest.raises(ValueError, match="WITHOUT rstd"):
        run_qk_norm_rope(rope_only, x, kx, x, kx, None, None, tab, tab, torch.empty(4, 8), torch.empty(4, 2), stream=0)
    # The convenience builder refuses a half-given pair before compiling anything.
    with pytest.raises(ValueError, match="together"):
        build_qk_norm_rope(x, kx, x, kx, w, None, tab, tab, rope_dim=64, eps=_EPS, stream=0)

    # The TMA twin enforces the same contract at compile time.
    from cudnn.gated_attention_block.kernels.qk_norm_rope_tma import QkNormRopeTmaRecipe, compile_qk_norm_rope_tma, run_qk_norm_rope_tma

    with pytest.raises(ValueError, match="emits no rstd"):
        compile_qk_norm_rope_tma(**base, want_rstd=True, apply_norm=False, tile_rows=8)  # 8: the fitted tile for h_q=8
    tma_rope_only = QkNormRopeTmaRecipe(
        compiled=None, h_q=8, h_kv=2, d=256, eps=_EPS, tile_rows=8, stages=2, stages_o=1, threads=128, want_rstd=False, ctas_per_sm=8, apply_norm=False
    )
    with pytest.raises(ValueError, match="WITHOUT the RMSNorm"):
        run_qk_norm_rope_tma(tma_rope_only, x, kx, x, kx, w, w, tab, tab, stream=0)

    # And the block-level knob: RoPE-only with no RoPE is refused at the geometry.
    from cudnn.gated_attention_block import GatedAttentionBlockGeometry
    from cudnn.gated_attention_block.api import _QkNormRope

    with pytest.raises(ValueError, match="identity copy"):
        GatedAttentionBlockGeometry(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=0, qk_norm=False).validate()
    g0 = GatedAttentionBlockGeometry(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64, qk_norm=False)
    with pytest.raises(ValueError, match="emits no rstd"):
        _QkNormRope(g0, batch=1, seq_len=64, dtype=torch.bfloat16, want_rstd=True, impl="ldg").check_support()
    _QkNormRope(g0, batch=1, seq_len=64, dtype=torch.bfloat16, want_rstd=False, impl="ldg").check_support()


@pytest.mark.L0
def test_block_default_tracks_the_kernel_default():
    """``api.py`` re-declares the deferred-load default so ``import cudnn`` stays
    cheap (no eager kernel import). A duplicated constant is a drift hazard, and
    the drift is SILENT: the two would simply compile different artifacts, both
    correct, one 33% slower. This is the only tripwire."""
    from cudnn.gated_attention_block import api
    from cudnn.gated_attention_block.kernels import qk_norm_rope as kern

    from cudnn.gated_attention_block.kernels import qk_norm_rope_tma as kern_tma

    assert api._QK_NORM_ROPE_DEFER_SECONDARY_LOADS is kern.DEFAULT_DEFER_SECONDARY_LOADS
    assert api._QK_NORM_ROPE_THREADS == kern.DEFAULT_THREADS_PER_CTA
    assert api._QK_NORM_ROPE_ROWS_PER_GROUP == kern.DEFAULT_ROWS_PER_GROUP
    assert api._QK_NORM_ROPE_TILE_ROWS == kern_tma.DEFAULT_TILE_ROWS
    assert api._QK_NORM_ROPE_STAGES == kern_tma.DEFAULT_STAGES


# --- the two implementations ------------------------------------------------
#
# Both kernels are kept ON PURPOSE: `tma` needs sm_90+ and is the faster one
# wherever it runs; `ldg` is the only option on SM80 and the fallback for any
# geometry the TMA tiling cannot express. The contract that makes keeping both
# safe is that they compute the SAME function -- so it is asserted, not assumed.


def _stage(impl, *, h_q=8, h_kv=2, d=256, rope=64, s=512, qk_norm=True, want_rstd=None):
    """``want_rstd=None`` follows ``qk_norm`` (rstd exists only where a norm exists)."""
    from cudnn.gated_attention_block import GatedAttentionBlockGeometry
    from cudnn.gated_attention_block.api import _QkNormRope

    g = GatedAttentionBlockGeometry(d_model=512, h_q=h_q, h_kv=h_kv, d_head=d, rope_dim=rope, qk_norm=qk_norm)
    g.validate()
    want_rstd = qk_norm if want_rstd is None else want_rstd
    return _QkNormRope(g, batch=1, seq_len=s, dtype=torch.bfloat16, want_rstd=want_rstd, impl=impl), g


@pytest.mark.L0
def test_impl_auto_resolves_and_explicit_tma_declines_loudly():
    """`auto` is a SELECTION; an explicit `tma` that cannot be served RAISES."""
    st, _ = _stage("auto")
    assert st.resolve_impl() in ("ldg", "tma")
    assert _stage("ldg")[0].resolve_impl() == "ldg"
    with pytest.raises(ValueError, match="impl must be one of"):
        _stage("nonsense")[0].resolve_impl()
    # a geometry the TMA tiling cannot express: warp-per-row needs d_head == 256
    narrow, _ = _stage("auto", h_q=4, h_kv=2, d=64, rope=16)
    assert narrow.resolve_impl() == "ldg"
    with pytest.raises(NotImplementedError, match="cannot tile this geometry|needs sm_90"):
        _stage("tma", h_q=4, h_kv=2, d=64, rope=16)[0].resolve_impl()


@pytest.mark.L0
def test_tile_rows_is_fitted_when_left_default_and_honored_when_given():
    """h_q=8 cannot tile at the measured optimum of 16. Left at None that must
    FIT down to 8 rather than silently abandon the TMA kernel; passed explicitly
    it must raise instead of being quietly replaced."""
    from cudnn.gated_attention_block.api import _QK_NORM_ROPE_TILE_ROWS

    assert _stage("auto", h_q=32)[0].resolve_tile_rows() == _QK_NORM_ROPE_TILE_ROWS
    assert _stage("auto", h_q=8)[0].resolve_tile_rows() == 8
    st, _ = _stage("auto", h_q=8)
    st.tile_rows = 16
    with pytest.raises(NotImplementedError, match="must divide h_q"):
        st.resolve_tile_rows()


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@_QK_NORM
def test_ldg_and_tma_are_bit_identical(qk_norm):
    """The two kernels must agree EXACTLY -- not within a tolerance. They run the
    same fp32 math and round once, so any difference is a bug in one of them.
    Under ``rope_only`` both take None weights, emit no rstd, and must ALSO copy
    the passthrough dims bit-exactly."""
    tma_stage, g = _stage("auto", qk_norm=qk_norm)
    if tma_stage.resolve_impl() != "tma":
        pytest.skip("this device has no TMA path; nothing to cross-check")
    t = tma_stage.seq_len
    dev = torch.device("cuda")
    q = torch.randn(1, t, g.h_q, g.d_head, dtype=torch.bfloat16, device=dev)
    k = torch.randn(1, t, g.h_kv, g.d_head, dtype=torch.bfloat16, device=dev)
    wq = torch.randn(g.d_head, dtype=torch.bfloat16, device=dev) if qk_norm else None
    wk = torch.randn(g.d_head, dtype=torch.bfloat16, device=dev) if qk_norm else None
    cos = torch.randn(1, t, g.rope_dim, dtype=torch.bfloat16, device=dev)
    sin = torch.randn(1, t, g.rope_dim, dtype=torch.bfloat16, device=dev)
    outs = {}
    for impl in ("ldg", "tma"):
        st, _ = _stage(impl, qk_norm=qk_norm)
        st.check_support()
        st.compile()
        assert st._recipe.apply_norm is qk_norm
        qo, ko = torch.zeros_like(q), torch.zeros_like(k)
        rq = torch.zeros(1, t, g.h_q, dtype=torch.float32, device=dev) if qk_norm else None
        rk = torch.zeros(1, t, g.h_kv, dtype=torch.float32, device=dev) if qk_norm else None
        st.execute(q, k, wq, wk, cos, sin, q_out=qo, k_out=ko, rstd_q=rq, rstd_k=rk)
        torch.cuda.synchronize()
        outs[impl] = (qo, ko, rq, rk) if qk_norm else (qo, ko)
        if not qk_norm:
            assert torch.equal(qo[..., g.rope_dim :], q[..., g.rope_dim :]), f"{impl}: RoPE-only passthrough dims are not bit-exact"
            assert torch.equal(ko[..., g.rope_dim :], k[..., g.rope_dim :]), f"{impl}: RoPE-only passthrough dims are not bit-exact"
    names = ("q", "k", "rstd_q", "rstd_k") if qk_norm else ("q", "k")
    for name, a, b in zip(names, outs["ldg"], outs["tma"]):
        assert torch.equal(a, b), f"{name}: ldg and tma disagree, max|diff| = {(a.float() - b.float()).abs().max().item():g}"


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_tma_is_deterministic_and_bitwise_ldg_when_ctas_reuse_the_output_stage():
    """The race the 640-tile shape above can never reach: a CTA that processes >= 2
    tiles rewrites its ``sOut`` stage, and the shipped ``fused_store_wait=True`` at
    ``stages_o=1`` let tile i+1's lanes overwrite it under tile i's in-flight bulk
    store -- ~0.1 % of Q rows carried the reference row of token ``tok +
    n_ctas / q_tiles_per_token``, run-to-run non-deterministic (found 2026-09-15 as a
    "long-S SDPA cosine drop" in the gated attention block).  h_q=32 at S=4096 is
    8192 Q tiles against SMs x 8 CTAs (1664 on Rubin), so every CTA reuses the
    stage several times: two TMA runs must be bit-identical to each other AND to
    the LDG kernel."""
    tma_stage, g = _stage("auto", h_q=32, h_kv=2, s=4096)
    if tma_stage.resolve_impl() != "tma":
        pytest.skip("this device has no TMA path; nothing to cross-check")
    t = tma_stage.seq_len
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(7)
    q = torch.randn(1, t, g.h_q, g.d_head, dtype=torch.bfloat16, device=dev, generator=gen)
    k = torch.randn(1, t, g.h_kv, g.d_head, dtype=torch.bfloat16, device=dev, generator=gen)
    wq = torch.randn(g.d_head, dtype=torch.bfloat16, device=dev, generator=gen)
    wk = torch.randn(g.d_head, dtype=torch.bfloat16, device=dev, generator=gen)
    cos = torch.randn(1, t, g.rope_dim, dtype=torch.bfloat16, device=dev, generator=gen)
    sin = torch.randn(1, t, g.rope_dim, dtype=torch.bfloat16, device=dev, generator=gen)
    outs = []
    for impl in ("tma", "tma", "ldg"):
        st, _ = _stage(impl, h_q=32, h_kv=2, s=4096)
        st.check_support()
        st.compile()
        qo, ko = torch.zeros_like(q), torch.zeros_like(k)
        rq = torch.zeros(1, t, g.h_q, dtype=torch.float32, device=dev)
        rk = torch.zeros(1, t, g.h_kv, dtype=torch.float32, device=dev)
        st.execute(q, k, wq, wk, cos, sin, q_out=qo, k_out=ko, rstd_q=rq, rstd_k=rk)
        torch.cuda.synchronize()
        outs.append((qo, ko, rq, rk))
    for name, a, b in zip(("q", "k", "rstd_q", "rstd_k"), outs[0], outs[1]):
        bad = (a != b).any(dim=-1).sum().item() if a.dim() == 4 else (a != b).sum().item()
        assert torch.equal(a, b), f"{name}: two TMA runs differ in {bad} rows -- the sOut write-after-read race"
    for name, a, b in zip(("q", "k", "rstd_q", "rstd_k"), outs[0], outs[2]):
        assert torch.equal(a, b), f"{name}: tma and ldg disagree at the multi-tile grid, max|diff| = {(a.float() - b.float()).abs().max().item():g}"


@pytest.mark.L0
def test_tma_refuses_the_fused_drain_at_one_output_stage():
    """``fused_store_wait=True`` drains AFTER the lanes wrote ``sOut``, so it can only
    protect the NEXT tile's stage when there is a second one: at ``stages_o=1`` the
    pair is the write-after-read race above and must be a typed refusal, before any
    kernel is compiled (CPU-only)."""
    from cudnn.gated_attention_block.kernels.qk_norm_rope_tma import DEFAULT_FUSED_STORE_WAIT, compile_qk_norm_rope_tma

    assert DEFAULT_FUSED_STORE_WAIT is False, "the racy fused drain must not be the default"
    with pytest.raises(ValueError, match="stages_o >= 2"):
        compile_qk_norm_rope_tma(dtype=torch.bfloat16, h_q=32, h_kv=2, d=256, rope_dim=64, eps=1e-6, want_rstd=False, stages_o=1, fused_store_wait=True)


# --- the three PR #1102 review findings, each pinned ------------------------


@requires_cuda
@pytest.mark.parametrize(
    "t, h_q, h_kv, rows_per_group",
    [
        (3, 3, 1, 2),  # n_q_rows=9: rows 10..11 are whole-K at K offset 1 -> 4 B, misaligned for st.global.v2
        (5, 3, 1, 4),  # n_q_rows=15: rows 16..19 are whole-K at K offset 1 -> 4 B, misaligned for st.global.v4
        (1, 3, 1, 2),  # n_q_rows=3: the odd group straddles the Q/K seam -> the scalar path regardless
    ],
)
def test_rstd_vector_store_needs_an_aligned_k_offset(t, h_q, h_kv, rows_per_group):
    """The vectorized rstd store was gated on the group being CONTIGUOUS in one tensor
    (whole-Q or whole-K, not the ragged tail) and nothing else. Contiguous is not
    aligned: ``row0`` is a multiple of R so the Q side is, but the K side lands at
    ``mRstdK + (row0 - n_q_rows) * 4``, which is R*4-aligned only when ``n_q_rows =
    T*h_q`` is a multiple of R. At T=3, h_q=3, R=2 it is 9 and the ``st.global.v2``
    faulted with ``cudaErrorMisalignedAddress`` (CodeRabbit + Codex on PR #1102,
    reproduced on an A100). Such groups must take the scalar path, and the rows they
    cover must still be right."""
    d, rope_dim, dtype = 256, 0, torch.bfloat16
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=11)
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    rstd_q = torch.full((t, h_q), -1.0, device="cuda", dtype=torch.float32)
    rstd_k = torch.full((t, h_kv), -1.0, device="cuda", dtype=torch.float32)
    build_qk_norm_rope(
        q,
        k,
        q_out,
        k_out,
        w_q,
        w_k,
        cos,
        sin,
        rstd_q,
        rstd_k,
        rope_dim=rope_dim,
        eps=_EPS,
        rows_per_group=rows_per_group,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    want_q, want_rq = _ref(q, w_q, cos, sin, rope_dim)
    want_k, want_rk = _ref(k, w_k, cos, sin, rope_dim)
    _check(q_out, want_q, dtype)
    _check(k_out, want_k, dtype)
    torch.testing.assert_close(rstd_q, want_rq, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rstd_k, want_rk, rtol=1e-5, atol=1e-6)


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_tma_k_tail_tile_reads_no_rope_table_row_past_t():
    """T=3, h_kv=2, tile_rows=8: the single K tile is ``tile_rows // h_kv = 4`` whole
    tokens, so rows 6..7 belong to token 3, which does not exist. TMA clips that
    token's load (zero-fill) and store, and the rstd write is predicated on
    ``tok < n_tokens`` -- but the cos/sin TABLE loads are plain ``ld.global``, and
    unclamped they read 16 B per rope lane past the end of a ``[T, ROPE]`` table
    (compute-sanitizer memcheck on SM100, Codex on PR #1102). The kernel now clamps
    the padded rows' token to ``T-1``.

    The tables are allocated EXACTLY ``[T, ROPE]`` -- no slack row -- so the read is
    addressable by memcheck (``PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer
    --tool memcheck --padding 32``), and the real rows must be BITWISE the LDG
    kernel's, whose tail rows clamp to the last valid row by construction."""
    from cudnn.gated_attention_block.kernels.qk_norm_rope_tma import compile_qk_norm_rope_tma, run_qk_norm_rope_tma, tile_counts

    t, h_q, h_kv, d, rope_dim, tile_rows, dtype = 3, 8, 2, 256, 64, 8, torch.bfloat16
    if _stage("auto", h_q=h_q, h_kv=h_kv, d=d, rope=rope_dim, s=t)[0].resolve_impl() != "tma":
        pytest.skip("this device has no TMA path; nothing to cross-check")
    n_q_tiles, n_tiles = tile_counts(t, h_q, h_kv, tile_rows)
    assert (n_q_tiles, n_tiles) == (3, 4), "the shape must leave ONE K tile that overshoots T"
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=13)
    assert tuple(cos.shape) == (t, rope_dim) and cos.is_contiguous() and sin.is_contiguous(), "the tables must carry no slack row"
    stream = torch.cuda.current_stream().cuda_stream
    outs = {}
    for impl in ("ldg", "tma"):
        q_out, k_out = torch.full_like(q, 1.5e3), torch.full_like(k, 1.5e3)
        rstd_q = torch.full((t, h_q), -1.0, device="cuda", dtype=torch.float32)
        rstd_k = torch.full((t, h_kv), -1.0, device="cuda", dtype=torch.float32)
        if impl == "ldg":
            build_qk_norm_rope(q, k, q_out, k_out, w_q, w_k, cos, sin, rstd_q, rstd_k, rope_dim=rope_dim, eps=_EPS, stream=stream)
        else:
            r = compile_qk_norm_rope_tma(dtype=dtype, h_q=h_q, h_kv=h_kv, d=d, rope_dim=rope_dim, eps=_EPS, want_rstd=True, tile_rows=tile_rows)
            assert r.tile_rows == tile_rows
            run_qk_norm_rope_tma(r, q, k, q_out, k_out, w_q, w_k, cos, sin, rstd_q, rstd_k, stream=stream)
        torch.cuda.synchronize()
        outs[impl] = (q_out, k_out, rstd_q, rstd_k)
    want_k, want_rk = _ref(k, w_k, cos, sin, rope_dim)
    _check(outs["tma"][1], want_k, dtype)
    torch.testing.assert_close(outs["tma"][3], want_rk, rtol=1e-5, atol=1e-6)
    for name, a, b in zip(("q", "k", "rstd_q", "rstd_k"), outs["ldg"], outs["tma"]):
        assert torch.equal(a, b), f"{name}: ldg and tma disagree at the overshooting K tile, max|diff| = {(a.float() - b.float()).abs().max().item():g}"


@pytest.mark.L0
def test_tma_compile_cache_keys_on_the_output_ring_depth(monkeypatch):
    """``stages_o`` is a Constexpr of the artifact -- it sizes ``sOut_raw`` and bounds
    the store drain -- and it was missing from the compile cache key, so the SECOND
    depth requested in a process was served the FIRST one's artifact under a recipe
    that reported the second (CodeRabbit on PR #1102). CPU-only: ``cute.compile`` is
    stubbed so the key logic is tested on its own, and the stub binds the launch
    signature by NAME to read back the ``stages_o`` that actually reached it."""
    import inspect

    from cudnn.gated_attention_block.kernels import qk_norm_rope_tma as kern_tma

    compiled_stages_o = []

    def fake_compile(fn, *args, **kwargs):
        bound = inspect.signature(fn).bind(*args)
        compiled_stages_o.append(bound.arguments["stages_o"])
        return object()  # a fresh, distinct artifact per compile

    monkeypatch.setattr(kern_tma, "compiled_cache", {})
    monkeypatch.setattr(kern_tma, "current_device", lambda: 0)
    monkeypatch.setattr(kern_tma.cute, "compile", fake_compile)
    common = dict(dtype=torch.bfloat16, h_q=32, h_kv=2, d=256, rope_dim=64, eps=1e-6, want_rstd=False, fused_store_wait=False)
    r1 = kern_tma.compile_qk_norm_rope_tma(stages_o=1, **common)
    r2 = kern_tma.compile_qk_norm_rope_tma(stages_o=2, **common)
    assert (r1.stages_o, r2.stages_o) == (1, 2)
    assert r1.compiled is not r2.compiled, "stages_o=2 was served the stages_o=1 artifact"
    assert compiled_stages_o == [1, 2], "the recipe must report the depth that was COMPILED"
    r1_again = kern_tma.compile_qk_norm_rope_tma(stages_o=1, **common)
    assert r1_again.compiled is r1.compiled and len(compiled_stages_o) == 2, "a repeat request is a cache hit, not a recompile"


@pytest.mark.L0
def test_stage_execute_checks_weights_against_the_geometry_both_ways():
    """``_QkNormRope.execute`` names ``geometry.qk_norm`` in a typed ValueError
    before it touches the recipe -- on any device, pre-compile."""
    on, g = _stage("ldg")
    off, _ = _stage("ldg", qk_norm=False)
    on._recipe = off._recipe = object()  # never dereferenced: the check comes first
    t = torch.empty(1, 4, g.h_q, g.d_head, dtype=torch.bfloat16)
    kv = torch.empty(1, 4, g.h_kv, g.d_head, dtype=torch.bfloat16)
    tab = torch.empty(1, 4, g.rope_dim, dtype=torch.bfloat16)
    w = torch.ones(g.d_head, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="qk_norm=True"):
        on.execute(t, kv, None, None, tab, tab)
    with pytest.raises(ValueError, match="qk_norm=False"):
        off.execute(t, kv, w, w, tab, tab)
    with pytest.raises(ValueError, match="together"):
        on.execute(t, kv, w, None, tab, tab)
