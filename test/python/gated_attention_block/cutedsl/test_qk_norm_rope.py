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
    # a geometry no tile fits by HEAD COUNT (tile_rows must divide h_q, be a multiple of h_kv and of the 4 warps; nothing in
    # 16..1 does for h_q = 20 MHA or h_q = 6 over h_kv = 2): the fit is 0, `auto` resolves to ldg on EVERY arch and check_support
    # passes -- the TMA validator types tile_rows=0 instead of dividing by it (a ZeroDivisionError no caller caught)
    for h_q, h_kv in ((20, 20), (6, 2)):
        wide, _ = _stage("auto", h_q=h_q, h_kv=h_kv)
        assert wide.resolve_tile_rows() == 0 and wide.resolve_impl() == "ldg", (h_q, h_kv)
        wide.check_support()
        with pytest.raises(NotImplementedError, match="cannot tile this geometry|needs sm_90") as ei:
            _stage("tma", h_q=h_q, h_kv=h_kv)[0].resolve_impl()
        if "cannot tile" in str(ei.value):  # on sm_90+ the shape is the reason, and it names the head counts
            assert f"h_q={h_q}" in str(ei.value) and f"h_kv={h_kv}" in str(ei.value), str(ei.value)


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


# ---------------------------------------------------------------------------
# The TMA kernel's e4m3 epilogue (the quantized backward's Q / K rebuild)
# ---------------------------------------------------------------------------


def _fp8_and_tma_available() -> bool:
    return torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) >= (9, 0)


requires_fp8_tma = pytest.mark.skipif(not _fp8_and_tma_available(), reason="the e4m3 epilogue needs the TMA kernel (sm_90+) and the fp8 cvt (sm_89+)")


@requires_fp8_tma
@_QK_NORM
@pytest.mark.parametrize("t", [64, 1003], ids=["aligned", "k-tail-overshoot"])
def test_tma_e4m3_epilogue_is_bitwise_the_bf16_output_quantized(qk_norm, t):
    """``q8`` / ``k8`` of the ``fp8_out`` artifact == the quantize pass over the bf16 artifact's output at the same static scales,
    BYTE FOR BYTE (the epilogue rounds to bf16 first, then applies the pass's own multiply and cvt), under both arms; the K tail
    tile that overshoots ``T`` stores nothing past the last token (the NaN poison past the tensor is the allocation's -- here
    every row of the exact-size tensors is written: no 0x7F survivor); a scale change changes the bytes without a recompile;
    the compile cache keys on ``fp8_out``."""
    from cudnn.gated_attention_block.kernels.qk_norm_rope_tma import compile_qk_norm_rope_tma, run_qk_norm_rope_tma
    from cudnn.gated_attention_block.kernels.quantize import compile_quantize, run_quantize

    h_q, h_kv, d, rope_dim, tile_rows = 8, 2, 256, 64, 8
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, torch.bfloat16, seed=51)
    w_q, w_k = _weights(w_q, w_k, qk_norm)
    st = torch.cuda.current_stream().cuda_stream
    kw = dict(dtype=torch.bfloat16, h_q=h_q, h_kv=h_kv, d=d, rope_dim=rope_dim, eps=_EPS, want_rstd=False, tile_rows=tile_rows, apply_norm=qk_norm)
    r16 = compile_qk_norm_rope_tma(**kw)
    r8 = compile_qk_norm_rope_tma(**kw, fp8_out=True)
    assert r8.fp8_out is True and r16.fp8_out is False and r8.compiled is not r16.compiled
    assert compile_qk_norm_rope_tma(**kw, fp8_out=False).compiled is r16.compiled
    q16, k16 = torch.empty_like(q), torch.empty_like(k)
    run_qk_norm_rope_tma(r16, q, k, q16, k16, w_q, w_k, cos, sin, stream=st)
    sq = torch.tensor([7.0], device="cuda")
    sk = torch.tensor([0.25], device="cuda")
    q8 = torch.full((t, h_q, d), 0x7F, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    k8 = torch.full((t, h_kv, d), 0x7F, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    run_qk_norm_rope_tma(r8, q, k, None, None, w_q, w_k, cos, sin, stream=st, q8=q8, k8=k8, scale_q=sq, scale_k=sk)
    torch.cuda.synchronize()

    def _quant(src, scale):
        dst = torch.empty(src.shape, dtype=torch.float8_e4m3fn, device="cuda")
        run_quantize(compile_quantize(dtype_in=src.dtype, h=int(src.shape[1]), d=d), src, dst, scale, stream=st)
        torch.cuda.synchronize()
        return dst

    assert not (q8.view(torch.uint8) == 0x7F).all(dim=-1).any() and not (k8.view(torch.uint8) == 0x7F).all(dim=-1).any(), "a row kept its poison"
    assert torch.equal(q8.view(torch.uint8), _quant(q16, sq).view(torch.uint8)), "q8 differs from the quantize pass over the bf16 rebuild"
    assert torch.equal(k8.view(torch.uint8), _quant(k16, sk).view(torch.uint8)), "k8 differs from the quantize pass over the bf16 rebuild"
    sq.fill_(14.0)
    run_qk_norm_rope_tma(r8, q, k, None, None, w_q, w_k, cos, sin, stream=st, q8=q8, k8=k8, scale_q=sq, scale_k=sk)
    torch.cuda.synchronize()
    assert torch.equal(q8.view(torch.uint8), _quant(q16, sq).view(torch.uint8)), "the scale is not read from the tensor"


# ---------------------------------------------------------------------------
# The MX epilogue (mx_out): the 32-token tile, the rowwise + columnwise MXFP8 quantizes of Q and K
# ---------------------------------------------------------------------------

requires_mx_tma = pytest.mark.skipif(
    not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) < (10, 0),
    reason="the MX epilogue needs the TMA kernel (sm_90+), the fp8 cvt (sm_89+) and cvt.rp.satfinite.ue8m0x2 (sm_100+)",
)


def _mx_blobs(b, s, h_q, h_kv, d, poison=0xFF):
    """0xFF-poisoned MX outputs: the four e4m3 payloads and the four SF blobs (every byte a NaN in its format until written)."""
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import sf_bytes

    t = b * s
    pay = lambda h: torch.full((t, h, d), poison, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)  # noqa: E731
    blob = lambda h: torch.full((sf_bytes(b, h, s, d),), poison, dtype=torch.uint8, device="cuda")  # noqa: E731
    return dict(q8=pay(h_q), sf_q=blob(h_q), q_T8=pay(h_q), sf_q_T=blob(h_q), k8=pay(h_kv), sf_k=blob(h_kv), k_T8=pay(h_kv), sf_k_T=blob(h_kv))


def _standalone_mx(src, *, batch, seq_len, h, d):
    """The standalone rowwise and columnwise MXFP8 quantizes of a compact ``[T, H, D]`` buffer: ``(row_d, row_sf, col_d, col_sf)``."""
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import compile_quantize_mxfp8, run_quantize_mxfp8, sf_bytes

    st = torch.cuda.current_stream().cuda_stream
    out = []
    for axis in ("row", "col"):
        r = compile_quantize_mxfp8(dtype_in=src.dtype, h=h, d=d, axis=axis)
        dst = torch.full((batch * seq_len, h, d), 0xFF, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
        sf = torch.full((sf_bytes(batch, h, seq_len, d),), 0xFF, dtype=torch.uint8, device="cuda")
        run_quantize_mxfp8(r, src, dst, sf, batch=batch, seq_len=seq_len, stream=st)
        out += [dst, sf]
    torch.cuda.synchronize()
    return tuple(out)


@requires_mx_tma
@_QK_NORM
@pytest.mark.parametrize(
    "b, s, h_q, h_kv, strided",
    [(1, 512, 8, 2, False), (2, 1008, 8, 2, False), (1, 992, 8, 2, True), (1, 256, 32, 2, True), (2, 96, 8, 2, False)],
    ids=["s512", "s1008-b2", "s992-slab", "h32-slab", "s96-b2"],
)
def test_tma_mx_epilogue_is_bitwise_the_rebuild_plus_four_quantizes(qk_norm, b, s, h_q, h_kv, strided):
    """The ``mx_out`` artifact's eight outputs == {the bf16 TMA rebuild, then the standalone rowwise AND columnwise MXFP8
    quantizes of ``q16`` and of ``k16``}, byte for byte, on 0xFF-POISONED destinations: every SF byte of every ceil128(S) unit is
    written (``s992``: the 992..1023 block of every (b, h) is a pad block -> ``0x00``), every payload byte is written, and at
    ``s1008`` with B = 2 batch 1's first 16 tokens keep THEIR bytes (an unpredicated pad-row store would land there).  ``strided``
    reads the bands as column slices of a ``[T, N]`` slab (the backward's source); ``h32`` is the 397B head count."""
    from cudnn.gated_attention_block.kernels.qk_norm_rope_tma import compile_qk_norm_rope_tma, mx_tile_counts, run_qk_norm_rope_tma

    d, rope_dim = 256, 64
    t = b * s
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, torch.bfloat16, seed=1000 * b + s + h_q)
    if strided:
        n = (2 * h_q + 2 * h_kv) * d
        slab = torch.randn(t, n, device="cuda").to(torch.bfloat16)
        qb = torch.as_strided(slab, (t, h_q, d), (n, d, 1), 0)
        kb = torch.as_strided(slab, (t, h_kv, d), (n, d, 1), 2 * h_q * d)
        qb.copy_(q)
        kb.copy_(k)
        q, k = qb, kb
    w_q, w_k = _weights(w_q, w_k, qk_norm)
    st = torch.cuda.current_stream().cuda_stream
    kw = dict(dtype=torch.bfloat16, h_q=h_q, h_kv=h_kv, d=d, rope_dim=rope_dim, eps=_EPS, want_rstd=False, tile_rows=8, apply_norm=qk_norm)
    r16 = compile_qk_norm_rope_tma(**kw)
    rmx = compile_qk_norm_rope_tma(**kw, mx_out=True)
    assert rmx.mx_out is True and r16.mx_out is False and rmx.compiled is not r16.compiled
    q16 = torch.empty(t, h_q, d, dtype=torch.bfloat16, device="cuda")
    k16 = torch.empty(t, h_kv, d, dtype=torch.bfloat16, device="cuda")
    run_qk_norm_rope_tma(r16, q, k, q16, k16, w_q, w_k, cos, sin, stream=st)
    o = _mx_blobs(b, s, h_q, h_kv, d)
    run_qk_norm_rope_tma(rmx, q, k, None, None, w_q, w_k, cos, sin, stream=st, batch=b, seq_len=s, **o)
    torch.cuda.synchronize()
    rq_d, rq_sf, cq_d, cq_sf = _standalone_mx(q16, batch=b, seq_len=s, h=h_q, d=d)
    rk_d, rk_sf, ck_d, ck_sf = _standalone_mx(k16, batch=b, seq_len=s, h=h_kv, d=d)
    u8 = torch.uint8
    assert torch.equal(o["q8"].view(u8), rq_d.view(u8)), "q8 differs from the rowwise quantize of the bf16 rebuild"
    assert torch.equal(o["sf_q"], rq_sf), "sf_q differs from the rowwise quantize of the bf16 rebuild"
    assert torch.equal(o["q_T8"].view(u8), cq_d.view(u8)), "q_T8 differs from the columnwise quantize of the bf16 rebuild"
    assert torch.equal(o["sf_q_T"], cq_sf), "sf_q_T differs from the columnwise quantize of the bf16 rebuild"
    assert torch.equal(o["k8"].view(u8), rk_d.view(u8)), "k8 differs from the rowwise quantize of the bf16 rebuild"
    assert torch.equal(o["sf_k"], rk_sf), "sf_k differs from the rowwise quantize of the bf16 rebuild"
    assert torch.equal(o["k_T8"].view(u8), ck_d.view(u8)), "k_T8 differs from the columnwise quantize of the bf16 rebuild"
    assert torch.equal(o["sf_k_T"], ck_sf), "sf_k_T differs from the columnwise quantize of the bf16 rebuild"
    for name in ("sf_q", "sf_q_T", "sf_k", "sf_k_T"):
        assert not (o[name] == 0xFF).any(), f"{name}: an SF byte was never written (E8M0 NaN under the SDPA's whole-tile SF TMA)"
    for name in ("q8", "q_T8", "k8", "k_T8"):
        assert not (o[name].view(u8) == 0xFF).all(dim=-1).any(), f"{name}: a row kept its poison"
    n_blk, n_sft, n_tiles = mx_tile_counts(b, s, h_q, h_kv)
    assert (n_blk, n_sft, n_tiles) == (4 * -(-s // 128), -(-s // 128), b * 4 * -(-s // 128) * (h_q + h_kv))
    if s % 128:
        # the pad blocks' SF bytes are 0x00 (the standalone's byte for a zeroed block); the rowwise tile's pad ROWS likewise
        from cudnn.gated_attention_block.kernels.quantize_mxfp8 import sf_byte_columnwise, sf_byte_rowwise

        pad_rows = [
            sf_byte_rowwise(bb, hh, ss, dd, n_heads=h_q, n_tiles=n_sft, d=d)
            for bb in range(b)
            for hh in range(h_q)
            for ss in range(s, n_sft * 128)
            for dd in range(0, d, 32)
        ]
        assert int(o["sf_q"][pad_rows].max().item()) == 0
        pad_cols = [
            sf_byte_columnwise(bb, hh, ss, dd, n_heads=h_q, n_tiles=n_sft, batch=b)
            for bb in range(b)
            for hh in range(h_q)
            for ss in range(-(-s // 32) * 32, n_sft * 128, 32)
            for dd in range(d)
        ]
        if pad_cols:
            assert int(o["sf_q_T"][pad_cols].max().item()) == 0


@requires_mx_tma
def test_tma_mx_epilogue_subnormal_block_takes_the_exact_arm():
    """A RoPE passthrough block (d >= rope_dim, RoPE-only: the values pass bit-exact) whose amax is the bf16 min normal 2^-126 and
    which holds the bf16 SUBNORMAL 2^-130: the E8M0 byte is 0x00 (scale 2^127) and the subnormal element quantizes to e4m3 0.125 =
    byte 0x20 on the EXACT ``x * rcp -> cvt.rn.satfinite.e4m3x2`` arm -- the fused scaled cvt would flush it to 0x00 (its docstring:
    fp32-subnormal inputs are flushed), which is why the MX epilogue never uses it.  Pinned against the standalone quantizer too."""
    from cudnn.gated_attention_block.kernels.qk_norm_rope_tma import compile_qk_norm_rope_tma, run_qk_norm_rope_tma
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import sf_byte_columnwise, sf_byte_rowwise

    b, s, h_q, h_kv, d, rope_dim = 1, 128, 8, 2, 256, 64
    t = b * s
    q, k, _, _, cos, sin = _make(t, h_q, h_kv, d, rope_dim, torch.bfloat16, seed=3)
    q.zero_()
    q[5, 1, 64] = 2.0**-126  # block c = 2 (d 64..95) of token 5, head 1: the amax
    q[5, 1, 65] = 2.0**-130  # a bf16 subnormal in the same block
    q[9, 1, 70] = 2.0**-130  # the columnwise block of (head 1, d 70), tokens 0..31: amax 2^-130 itself -> its scale 2^127 too, code 0x20
    assert float(q[5, 1, 64]) == 2.0**-126 and float(q[5, 1, 65]) == 2.0**-130 and float(q[9, 1, 70]) == 2.0**-130
    st = torch.cuda.current_stream().cuda_stream
    rmx = compile_qk_norm_rope_tma(
        dtype=torch.bfloat16, h_q=h_q, h_kv=h_kv, d=d, rope_dim=rope_dim, eps=_EPS, want_rstd=False, tile_rows=8, apply_norm=False, mx_out=True
    )
    o = _mx_blobs(b, s, h_q, h_kv, d)
    run_qk_norm_rope_tma(rmx, q, k, None, None, None, None, cos, sin, stream=st, batch=b, seq_len=s, **o)
    # the reference is the standalone chain over the REBUILT bf16 rows (the RoPE of an all-zero row yields -0.0 where cos < 0, which both
    # chains quantize to e4m3 0x80 -- the raw input is not the quantizer's input), exactly as the bitwise test above
    r16 = compile_qk_norm_rope_tma(dtype=torch.bfloat16, h_q=h_q, h_kv=h_kv, d=d, rope_dim=rope_dim, eps=_EPS, want_rstd=False, tile_rows=8, apply_norm=False)
    q16 = torch.empty(t, h_q, d, dtype=torch.bfloat16, device="cuda")
    k16 = torch.empty(t, h_kv, d, dtype=torch.bfloat16, device="cuda")
    run_qk_norm_rope_tma(r16, q, k, q16, k16, None, None, cos, sin, stream=st)
    torch.cuda.synchronize()
    assert torch.equal(q16[:, :, rope_dim:], q[:, :, rope_dim:]), "the passthrough dims are not bit-exact through the rebuild"
    rq_d, rq_sf, cq_d, cq_sf = _standalone_mx(q16, batch=b, seq_len=s, h=h_q, d=d)
    assert torch.equal(o["q8"].view(torch.uint8), rq_d.view(torch.uint8)) and torch.equal(o["sf_q"], rq_sf)
    assert torch.equal(o["q_T8"].view(torch.uint8), cq_d.view(torch.uint8)) and torch.equal(o["sf_q_T"], cq_sf)
    n_sft = 1
    assert int(o["sf_q"][sf_byte_rowwise(0, 1, 5, 64, n_heads=h_q, n_tiles=n_sft, d=d)].item()) == 0x00, "the block's E8M0 is not 0x00 (scale 2^127)"
    assert int(o["q8"].view(torch.uint8)[5, 1, 64].item()) == 0x40, "2^-126 * 2^127 = 2.0 is e4m3 0x40"
    assert int(o["q8"].view(torch.uint8)[5, 1, 65].item()) == 0x20, "the subnormal element must quantize to 0.125 (0x20): the exact arm, not a flushed 0x00"
    assert int(o["sf_q_T"][sf_byte_columnwise(0, 1, 9, 70, n_heads=h_q, n_tiles=n_sft, batch=b)].item()) == 0x00
    assert int(o["q_T8"].view(torch.uint8)[9, 1, 70].item()) == 0x20, "the columnwise pass must take the exact arm too"


@pytest.mark.L0
def test_tma_mx_epilogue_contract_is_typed():
    """``mx_out`` at compile (a bool, exclusive with ``fp8_out``, no rstd, the default refill placement only, the MX geometry rules)
    and at execute both ways (Rule 1): the eight MX outputs + batch / seq_len REQUIRED on an ``mx_out`` artifact and refused on a
    bf16 one; the bf16 outputs and the per-tensor scales refused on it; an SF blob of the wrong size; ``T != batch * seq_len``.
    The shape algebra: ``mx_tile_counts`` over the PADDED sequence."""
    from cudnn.gated_attention_block.kernels.qk_norm_rope_tma import (
        MX_TILE_TOKENS,
        QkNormRopeTmaRecipe,
        compile_qk_norm_rope_tma,
        mx_tile_counts,
        run_qk_norm_rope_tma,
        validate_mx_shape,
    )

    assert MX_TILE_TOKENS == 32
    assert mx_tile_counts(1, 128, 8, 2) == (4, 1, 40) and mx_tile_counts(2, 992, 8, 2) == (32, 8, 640) and mx_tile_counts(1, 1000, 32, 2) == (32, 8, 1088)
    validate_mx_shape(256, 64, 128)
    validate_mx_shape(256, 0, 256)
    with pytest.raises(ValueError, match="warp-per-row"):
        validate_mx_shape(128, 64, 128)
    with pytest.raises(ValueError, match="spread evenly over the 3 warps"):
        validate_mx_shape(256, 64, 96)
    with pytest.raises(ValueError, match="rope_dim must be a multiple"):
        validate_mx_shape(256, 24, 128)
    base = dict(dtype=torch.bfloat16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, want_rstd=False, tile_rows=8)
    with pytest.raises(ValueError, match="mx_out must be a bool"):
        compile_qk_norm_rope_tma(**base, mx_out=1)
    with pytest.raises(ValueError, match="two epilogues of one tile"):
        compile_qk_norm_rope_tma(**base, fp8_out=True, mx_out=True)
    with pytest.raises(ValueError, match="emits no rstd"):
        compile_qk_norm_rope_tma(**{**base, "want_rstd": True}, mx_out=True)
    with pytest.raises(ValueError, match="refills after its second barrier only"):
        compile_qk_norm_rope_tma(**base, refill_pos=0, mx_out=True)
    if not torch.cuda.is_available():
        return
    rb = dict(compiled=None, h_q=8, h_kv=2, d=256, eps=_EPS, tile_rows=8, stages=2, stages_o=1, threads=128, want_rstd=False, ctas_per_sm=8, apply_norm=True)
    r16 = QkNormRopeTmaRecipe(**rb)
    rmx = QkNormRopeTmaRecipe(**rb, mx_out=True)
    assert r16.mx_out is False and rmx.fp8_out is False
    b, s = 2, 64
    t = b * s
    x = torch.empty(t, 8, 256, dtype=torch.bfloat16, device="cuda")
    kx = torch.empty(t, 2, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(256, dtype=torch.bfloat16, device="cuda")
    tab = torch.empty(t, 64, dtype=torch.bfloat16, device="cuda")
    o = _mx_blobs(b, s, 8, 2, 256)
    with pytest.raises(ValueError, match="WITH the MX epilogue .*must all be bound"):
        run_qk_norm_rope_tma(rmx, x, kx, None, None, w, w, tab, tab, stream=0, batch=b, seq_len=s, **{**o, "sf_k_T": None})
    with pytest.raises(ValueError, match="WITH the MX epilogue .*must all be bound"):
        run_qk_norm_rope_tma(rmx, x, kx, None, None, w, w, tab, tab, stream=0, seq_len=s, **o)
    with pytest.raises(ValueError, match="pass q_out=k_out=None and no"):
        run_qk_norm_rope_tma(rmx, x, kx, x, kx, w, w, tab, tab, stream=0, batch=b, seq_len=s, **o)
    with pytest.raises(ValueError, match="pass q_out=k_out=None and no"):
        run_qk_norm_rope_tma(rmx, x, kx, None, None, w, w, tab, tab, stream=0, batch=b, seq_len=s, scale_q=torch.ones(1, device="cuda"), **o)
    with pytest.raises(ValueError, match="T must equal batch\\*seq_len"):
        run_qk_norm_rope_tma(rmx, x, kx, None, None, w, w, tab, tab, stream=0, batch=b, seq_len=s + 1, **o)
    with pytest.raises(ValueError, match="sf_q_T must hold exactly"):
        run_qk_norm_rope_tma(rmx, x, kx, None, None, w, w, tab, tab, stream=0, batch=b, seq_len=s, **{**o, "sf_q_T": o["sf_q_T"][:-16]})
    with pytest.raises(ValueError, match="q_T8 must be a torch.float8_e4m3fn"):
        run_qk_norm_rope_tma(rmx, x, kx, None, None, w, w, tab, tab, stream=0, batch=b, seq_len=s, **{**o, "q_T8": x})
    with pytest.raises(ValueError, match="WITHOUT the MX epilogue"):
        run_qk_norm_rope_tma(r16, x, kx, x, kx, w, w, tab, tab, stream=0, sf_q=o["sf_q"])
    with pytest.raises(ValueError, match="WITHOUT the MX epilogue"):
        run_qk_norm_rope_tma(r16, x, kx, x, kx, w, w, tab, tab, stream=0, batch=b, seq_len=s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_tma_e4m3_epilogue_contract_is_typed():
    """``fp8_out`` both directions at execute (Rule 1): the e4m3 outputs and scales REQUIRED on an fp8_out artifact and refused on a
    bf16 one; ``q_out`` / ``k_out`` refused on an fp8_out artifact; a non-e4m3 / misaligned ``q8``; ``fp8_out`` must be a bool."""
    from cudnn.gated_attention_block.kernels.qk_norm_rope_tma import QkNormRopeTmaRecipe, compile_qk_norm_rope_tma, run_qk_norm_rope_tma

    base = dict(compiled=None, h_q=8, h_kv=2, d=256, eps=_EPS, tile_rows=8, stages=2, stages_o=1, threads=128, want_rstd=False, ctas_per_sm=8, apply_norm=True)
    r16 = QkNormRopeTmaRecipe(**base)
    r8 = QkNormRopeTmaRecipe(**base, fp8_out=True)
    assert r16.fp8_out is False
    t = 4
    x = torch.empty(t, 8, 256, dtype=torch.bfloat16, device="cuda")
    kx = torch.empty(t, 2, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(256, dtype=torch.bfloat16, device="cuda")
    tab = torch.empty(t, 64, dtype=torch.bfloat16, device="cuda")
    q8 = torch.empty(t, 8, 256, dtype=torch.float8_e4m3fn, device="cuda")
    k8 = torch.empty(t, 2, 256, dtype=torch.float8_e4m3fn, device="cuda")
    sc = torch.ones(1, device="cuda")
    with pytest.raises(ValueError, match="WITH the e4m3 epilogue .*must all be bound"):
        run_qk_norm_rope_tma(r8, x, kx, None, None, w, w, tab, tab, stream=0)
    with pytest.raises(ValueError, match="pass q_out=k_out=None"):
        run_qk_norm_rope_tma(r8, x, kx, x, kx, w, w, tab, tab, stream=0, q8=q8, k8=k8, scale_q=sc, scale_k=sc)
    with pytest.raises(ValueError, match="WITHOUT the e4m3 epilogue"):
        run_qk_norm_rope_tma(r16, x, kx, x, kx, w, w, tab, tab, stream=0, q8=q8, k8=k8, scale_q=sc, scale_k=sc)
    with pytest.raises(ValueError, match="q8 must be a torch.float8_e4m3fn"):
        run_qk_norm_rope_tma(r8, x, kx, None, None, w, w, tab, tab, stream=0, q8=x, k8=k8, scale_q=sc, scale_k=sc)
    with pytest.raises(ValueError, match=r"k8 must be \[T=4, H=2, D=256\]"):
        run_qk_norm_rope_tma(r8, x, kx, None, None, w, w, tab, tab, stream=0, q8=q8, k8=q8, scale_q=sc, scale_k=sc)
    with pytest.raises(ValueError, match="scale_k must be a 1-element fp32 CUDA tensor"):
        run_qk_norm_rope_tma(r8, x, kx, None, None, w, w, tab, tab, stream=0, q8=q8, k8=k8, scale_q=sc, scale_k=torch.ones(1))
    odd8 = torch.empty(t, 8 * 256 + 8, dtype=torch.float8_e4m3fn, device="cuda")[:, : 8 * 256].view(t, 8, 256)
    with pytest.raises(ValueError, match=r"q8 \(e4m3\) must be"):
        run_qk_norm_rope_tma(r8, x, kx, None, None, w, w, tab, tab, stream=0, q8=odd8, k8=k8, scale_q=sc, scale_k=sc)
    with pytest.raises(ValueError, match="fp8_out must be a bool"):
        compile_qk_norm_rope_tma(dtype=torch.bfloat16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, want_rstd=False, tile_rows=8, fp8_out=1)
