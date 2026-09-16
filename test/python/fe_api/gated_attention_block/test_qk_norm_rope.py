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
