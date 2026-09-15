# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stages (2)+(3) — the fused QK-RMSNorm + partial RoPE FROST kernel.

The kernel is plain vectorized LDG/STG plus one butterfly shuffle: no tcgen05,
no TMA, no arch-specific path. So unlike stage (4) it runs anywhere CuTe DSL
does, and these tests deliberately do NOT gate on Rubin — catching a lane-group
or tail bug on whatever device is at hand is worth more than arch purity.
"""

import os
import sys

import pytest
import torch

from cudnn.gated_attention_block.kernels.qk_norm_rope import (
    build_qk_norm_rope,
    lanes_per_row,
    moved_bytes,
    validate_shape,
    vec_chunks,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import qk_norm_rope_reference  # noqa: E402

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_EPS = 1e-6


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


def _ref(x, w, cos, sin, rope_dim):
    """The [T, H, D] oracle: the reference takes [B, S, H, D], so borrow B=1."""
    y, rstd = qk_norm_rope_reference(x[None], w, cos[None], sin[None], rope_dim, _EPS)
    return y[0], rstd[0]


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
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("t, h_q, h_kv, d, rope_dim", [(64, 8, 2, 256, 64), (64, 4, 1, 128, 32), (64, 2, 2, 64, 16)])
def test_matches_the_fused_oracle(dtype, t, h_q, h_kv, d, rope_dim):
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, dtype)
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    rstd_q = torch.empty(t, h_q, device="cuda", dtype=torch.float32)
    rstd_k = torch.empty(t, h_kv, device="cuda", dtype=torch.float32)

    build_qk_norm_rope(q, k, q_out, k_out, w_q, w_k, cos, sin, rstd_q, rstd_k, rope_dim=rope_dim, eps=_EPS, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    want_q, want_rstd_q = _ref(q, w_q, cos, sin, rope_dim)
    want_k, want_rstd_k = _ref(k, w_k, cos, sin, rope_dim)
    _check(q_out, want_q, dtype)
    _check(k_out, want_k, dtype)
    torch.testing.assert_close(rstd_q, want_rstd_q, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rstd_k, want_rstd_k, rtol=1e-5, atol=1e-6)


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
def test_in_place_matches_out_of_place():
    """``q_out is q`` is the block's default: every lane reads its whole row
    before any lane stores, and the RoPE shuffle stays inside the row."""
    t, h_q, h_kv, d, rope_dim = 96, 8, 2, 256, 64
    dtype = torch.bfloat16
    q, k, w_q, w_k, cos, sin = _make(t, h_q, h_kv, d, rope_dim, dtype, seed=3)
    q_ref, k_ref = torch.empty_like(q), torch.empty_like(k)
    build_qk_norm_rope(q, k, q_ref, k_ref, w_q, w_k, cos, sin, rope_dim=rope_dim, eps=_EPS, stream=torch.cuda.current_stream().cuda_stream)
    q_ip, k_ip = q.clone(), k.clone()
    build_qk_norm_rope(q_ip, k_ip, q_ip, k_ip, w_q, w_k, cos, sin, rope_dim=rope_dim, eps=_EPS, stream=torch.cuda.current_stream().cuda_stream)
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


def _stage(impl, *, h_q=8, h_kv=2, d=256, rope=64, s=512):
    from cudnn.gated_attention_block import GatedAttentionBlockGeometry
    from cudnn.gated_attention_block.api import _QkNormRope

    g = GatedAttentionBlockGeometry(d_model=512, h_q=h_q, h_kv=h_kv, d_head=d, rope_dim=rope)
    g.validate()
    return _QkNormRope(g, batch=1, seq_len=s, dtype=torch.bfloat16, want_rstd=True, impl=impl), g


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
def test_ldg_and_tma_are_bit_identical():
    """The two kernels must agree EXACTLY -- not within a tolerance. They run the
    same fp32 math and round once, so any difference is a bug in one of them."""
    tma_stage, g = _stage("auto")
    if tma_stage.resolve_impl() != "tma":
        pytest.skip("this device has no TMA path; nothing to cross-check")
    t = tma_stage.seq_len
    dev = torch.device("cuda")
    q = torch.randn(1, t, g.h_q, g.d_head, dtype=torch.bfloat16, device=dev)
    k = torch.randn(1, t, g.h_kv, g.d_head, dtype=torch.bfloat16, device=dev)
    wq = torch.randn(g.d_head, dtype=torch.bfloat16, device=dev)
    wk = torch.randn(g.d_head, dtype=torch.bfloat16, device=dev)
    cos = torch.randn(1, t, g.rope_dim, dtype=torch.bfloat16, device=dev)
    sin = torch.randn(1, t, g.rope_dim, dtype=torch.bfloat16, device=dev)
    outs = {}
    for impl in ("ldg", "tma"):
        st, _ = _stage(impl)
        st.check_support()
        st.compile()
        qo, ko = torch.zeros_like(q), torch.zeros_like(k)
        rq = torch.zeros(1, t, g.h_q, dtype=torch.float32, device=dev)
        rk = torch.zeros(1, t, g.h_kv, dtype=torch.float32, device=dev)
        st.execute(q, k, wq, wk, cos, sin, q_out=qo, k_out=ko, rstd_q=rq, rstd_k=rk)
        torch.cuda.synchronize()
        outs[impl] = (qo, ko, rq, rk)
    for name, a, b in zip(("q", "k", "rstd_q", "rstd_k"), outs["ldg"], outs["tma"]):
        assert torch.equal(a, b), f"{name}: ldg and tma disagree, max|diff| = {(a.float() - b.float()).abs().max().item():g}"
