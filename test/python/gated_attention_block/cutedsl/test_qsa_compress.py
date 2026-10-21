# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The QSA block-compression kernel (``kernels/qsa_compress.py``): mean-pool
``POOL`` raw indexer keys -> RMSNorm over 128 -> partial RoPE on 64 dims at the
block's start position, one 16-bit rounding at the end.

The oracle is re-derived in this module from the HF Qwen4 reference
(``Qwen4ExpTextQSAIndexer.forward``: ``key_groups.float().mean(dim=1)``, the
``k_layernorm`` with its zero-centred ``(1 + w)``, ``apply_rotary_pos_emb`` at
the group's first position) with ONE deliberate difference: HF casts the pooled
key to the key dtype BEFORE the norm, this oracle (like the kernel) keeps fp32
and rounds once at the end.  The acceptance is "bitwise-close": every written
element within ONE ulp of the io dtype of the oracle's rounded value (the kernel
and the oracle differ only in fp32 op order), and nothing written where the
kernel owes no block.

The kernel is plain vectorized LDG/STG plus warp shuffles -- no tcgen05, no TMA
-- so these tests deliberately do NOT gate on Rubin: they run on whatever CUDA
device is at hand.
"""

import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gated_attention_block.kernels.qsa_compress import (
    DEFAULT_POOL,
    QsaCompressRecipe,
    build_qsa_compress,
    compile_qsa_compress,
    moved_bytes,
    run_qsa_compress,
    validate_compress_shape,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import build_rope_tables  # noqa: E402

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_EPS = 1e-6
_D = 128  # the indexer head dim
_ROPE = 64  # partial RoPE width
_POOL = DEFAULT_POOL
_THETA = 1e7  # the model's rope_theta
_SENTINEL = -7777.0


def _stream():
    return torch.cuda.current_stream().cuda_stream


# ---------------------------------------------------------------------------
# Oracle (re-derived from the HF reference, fp32 end to end, one rounding)
# ---------------------------------------------------------------------------


def compress_reference(k_raw, w, cos, sin, seq_lens, *, pool=_POOL, eps=_EPS, norm_weight_offset=0.0, rope_dim=_ROPE):
    """fp32 compressed keys ``[B, floor(S / pool), D]`` and the per-entry valid
    block counts ``[B]``.

    ``k_raw`` ``[B, S, D]``; ``w`` ``[D]`` (``norm_weight_offset`` is added in
    fp32: 0.0 for a pre-folded ``(1 + w)``, 1.0 for the raw zero-centred
    weight); ``cos`` / ``sin`` ``[1 or B, S, rope_dim]`` duplicated-half
    tables, read at the block START position ``pool * b``; ``seq_lens`` ``[B]``
    or None (= S).  Blocks at or past an entry's count are not the kernel's
    to write -- the caller masks them out of the comparison.
    """
    batch, s_tok, d = k_raw.shape
    nb = s_tok // pool
    pooled = k_raw[:, : nb * pool].float().reshape(batch, nb, pool, d).mean(dim=2)
    rstd = torch.rsqrt(pooled.pow(2).mean(-1, keepdim=True) + eps)
    y = pooled * rstd * (w.float() + norm_weight_offset)
    if rope_dim:
        pos = torch.arange(nb, device=k_raw.device) * pool
        c = cos[:, pos, :rope_dim].float()
        s = sin[:, pos, :rope_dim].float()
        yr = y[..., :rope_dim]
        half = rope_dim // 2
        rotated = torch.cat((-yr[..., half:], yr[..., :half]), dim=-1)
        y = torch.cat((yr * c + rotated * s, y[..., rope_dim:]), dim=-1)
    lens = torch.full((batch,), s_tok, device=k_raw.device, dtype=torch.int64) if seq_lens is None else seq_lens.to(torch.int64).clamp(0, s_tok)
    return y, lens // pool


_SUBNORMAL_SPACING_F16 = 2.0**-24
"""fp16's fixed spacing below its smallest normal (2^-14): a value that
cancels down to ~1e-5 lands here, where a frexp-based ulp underestimates the
grid.  The one-ulp budget is floored at it for BOTH io dtypes (an absolute
6e-8, far below anything a consumer resolves)."""


def _ulp(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The spacing of ``dtype`` at each value of ``x`` (bf16: 8 significand
    bits, f16: 11), floored at fp16's subnormal spacing."""
    mant = 8 if dtype is torch.bfloat16 else 11
    _, e = torch.frexp(x.float())  # x = m * 2^e, m in [0.5, 1): the binade [2^(e-1), 2^e)
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), e - mant).clamp_min(_SUBNORMAL_SPACING_F16)


def _assert_one_rounding(got: torch.Tensor, want32: torch.Tensor, mask: torch.Tensor, what: str) -> float:
    """Every masked element of ``got`` within one io-dtype ulp of the oracle's
    rounded value; returns the bitwise-equal fraction (reported, not gated)."""
    assert torch.isfinite(want32).all(), f"{what}: the oracle itself is non-finite"
    assert torch.isfinite(got[mask]).all(), f"{what}: non-finite kernel output"
    want = want32.to(got.dtype)
    diff = (got.float() - want.float()).abs()
    bad = (diff > _ulp(want, got.dtype)) & mask
    if bad.any():
        where = bad.nonzero()[:8].tolist()
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {int(mask.sum())} elements beyond one {got.dtype} ulp of the oracle; "
            f"max |diff| among them {diff[bad].max().item():.3e}, first indices {where}"
        )
    return (got[mask] == want[mask]).float().mean().item()


def _valid_mask(n_blocks: torch.Tensor, nb: int) -> torch.Tensor:
    batch = int(n_blocks.shape[0])
    return (torch.arange(nb, device=n_blocks.device)[None, :] < n_blocks[:, None])[:, :, None].expand(batch, nb, _D)


def _make(batch, s_tok, dtype, *, seed=0, slab=False, table_batch=None):
    """Raw keys (optionally as the INDEX band of a ``[B, S, 640]`` slab, token
    stride 640), a small zero-centred weight, and real RoPE tables at the
    model's theta (identical rows across the batch)."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32)

    if slab:
        k_raw = rnd(batch, s_tok, 640).to(dtype)[:, :, 512:640]
    else:
        k_raw = rnd(batch, s_tok, _D).to(dtype)
    w = (0.05 * rnd(_D)).to(dtype)
    cos, sin = build_rope_tables(s_tok, _ROPE, base=_THETA, batch=batch if table_batch is None else table_batch, device="cuda", dtype=dtype)
    return k_raw, w, cos, sin


def _sentinel(batch, nb, dtype):
    return torch.full((batch, nb, _D), _SENTINEL, device="cuda", dtype=dtype)


# ---------------------------------------------------------------------------
# Shape algebra -- no GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, rope_dim, pool, threads, match",
    [
        (128, 64, 3, 128, "power of two"),
        (128, 64, 0, 128, "power of two"),
        (124, 64, 4, 128, "multiple of 8"),
        (128, 24, 4, 128, "multiple of 16"),
        (128, 64, 4, 100, "multiple of the 16 lanes"),
    ],
)
def test_validate_compress_shape_rejects(d, rope_dim, pool, threads, match):
    with pytest.raises(ValueError, match=match):
        validate_compress_shape(d, rope_dim, pool, threads)


def test_validate_compress_shape_accepts_the_indexer_geometry():
    validate_compress_shape(_D, _ROPE, _POOL, 128)
    validate_compress_shape(_D, 0, 1, 128)


def test_moved_bytes_is_pool_plus_one_rows_per_block():
    # 8192 blocks x (4 rows in + 1 row out) x 256 B = 10 MiB at S = 32768.
    assert moved_bytes(1, 32768, _D) == 8192 * 5 * _D * 2 == 10 * 1024 * 1024
    assert moved_bytes(3, 2051, _D) == 3 * 512 * 5 * _D * 2
    assert moved_bytes(2, 3, _D) == 0


def test_compile_rejects_unsupported_dtypes():
    with pytest.raises(ValueError, match="bf16/f16"):
        compile_qsa_compress(dtype=torch.float32, d=_D, rope_dim=_ROPE)
    with pytest.raises(ValueError, match="16-bit cos"):
        compile_qsa_compress(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, table_dtype=torch.float32)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("offset", [0.0, 1.0], ids=["folded_1_plus_w", "zero_centred_w"])
@pytest.mark.parametrize("s_tok", [5, 512, 2051, 4096])
def test_matches_the_oracle_within_one_rounding(s_tok, offset, dtype):
    """Three entries: a full one, one 7 tokens short (a ragged tail), an EMPTY
    one.  Every block an entry owns is within one ulp of the oracle; every
    other row of ``out`` keeps its sentinel."""
    batch = 3
    k_raw, w, cos, sin = _make(batch, s_tok, dtype)
    w_in = (1.0 + w.float()).to(dtype) if offset == 0.0 else w
    seq_lens = torch.tensor([s_tok, max(s_tok - 7, 0), 0], device="cuda", dtype=torch.int32)
    nb = s_tok // _POOL
    out = _sentinel(batch, nb, dtype)
    sentinel = out.clone()

    r = build_qsa_compress(k_raw, out, w_in, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=offset, stream=_stream())
    torch.cuda.synchronize()
    assert r.has_seq_lens and r.pool == _POOL and r.d == _D

    want, n_blocks = compress_reference(k_raw, w_in, cos, sin, seq_lens, norm_weight_offset=offset)
    assert n_blocks.tolist() == [s_tok // _POOL, max(s_tok - 7, 0) // _POOL, 0]
    mask = _valid_mask(n_blocks, nb)
    exact = _assert_one_rounding(out, want, mask, f"S={s_tok} {dtype} offset={offset}")
    assert torch.equal(out[~mask], sentinel[~mask]), "a block the kernel does not own was written"
    print(f"S={s_tok} {dtype} offset={offset}: {int(mask.sum())} elements checked, {exact * 100:.3f}% bitwise equal to the rounded oracle")


@requires_cuda
@pytest.mark.parametrize("s_tok", [5, 2051])
def test_dense_artifact_without_seq_lens(s_tok):
    """``seq_lens=None`` traces the dense artifact: every entry has
    ``floor(S / POOL)`` blocks and the load is folded out."""
    batch, dtype = 2, torch.bfloat16
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=11)
    nb = s_tok // _POOL
    out = _sentinel(batch, nb, dtype)
    r = build_qsa_compress(k_raw, out, w, cos, sin, None, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
    torch.cuda.synchronize()
    assert not r.has_seq_lens
    want, n_blocks = compress_reference(k_raw, w, cos, sin, None, norm_weight_offset=1.0)
    assert n_blocks.tolist() == [nb, nb]
    _assert_one_rounding(out, want, torch.ones_like(out, dtype=torch.bool), f"dense S={s_tok}")


@requires_cuda
def test_rope_actually_rotates():
    """Guards the butterfly: a wrong partner or sign is finite and plausible,
    so the rope band must DIFFER from a norm-only oracle while the passthrough
    band ``[64, 128)`` matches it."""
    batch, s_tok, dtype = 2, 256, torch.bfloat16
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=7)
    nb = s_tok // _POOL
    out = torch.empty(batch, nb, _D, device="cuda", dtype=dtype)
    build_qsa_compress(k_raw, out, w, cos, sin, None, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
    torch.cuda.synchronize()
    norm_only, _ = compress_reference(k_raw, w, cos, sin, None, norm_weight_offset=1.0, rope_dim=0)
    assert not torch.allclose(out[..., :_ROPE].float(), norm_only[..., :_ROPE], atol=1e-2)
    full = torch.ones_like(out, dtype=torch.bool)
    _assert_one_rounding(out[..., _ROPE:], norm_only[..., _ROPE:], full[..., _ROPE:], "passthrough band")


@requires_cuda
def test_no_write_for_incomplete_groups_and_empty_sequences():
    """S = 13 holds 3 complete blocks and a 1-token tail; lengths
    ``[13, 0, 3, 9]`` own ``[3, 0, 0, 2]`` blocks.  Everything else keeps the
    sentinel, and the tail token is never read into a block."""
    dtype, s_tok = torch.bfloat16, 13
    seq_lens = torch.tensor([13, 0, 3, 9], device="cuda", dtype=torch.int32)
    batch = int(seq_lens.shape[0])
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=3)
    # Poison the tail token: if it were pooled into a block the output would be non-finite.
    k_raw[:, 12, :] = float("nan")
    nb = s_tok // _POOL
    out = _sentinel(batch, nb, dtype)
    sentinel = out.clone()
    build_qsa_compress(k_raw, out, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
    torch.cuda.synchronize()
    want, n_blocks = compress_reference(k_raw, w, cos, sin, seq_lens, norm_weight_offset=1.0)
    assert n_blocks.tolist() == [3, 0, 0, 2]
    mask = _valid_mask(n_blocks, nb)
    _assert_one_rounding(out, want, mask, "S=13")
    assert torch.equal(out[~mask], sentinel[~mask])
    assert torch.equal(out[1], sentinel[1]) and torch.equal(out[2], sentinel[2])


@requires_cuda
def test_seq_lens_are_clamped_on_device():
    """A length above S behaves as S; a negative one as 0 -- no fault, no
    out-of-entry read."""
    dtype, s_tok = torch.bfloat16, 64
    seq_lens = torch.tensor([s_tok + 1000, -5], device="cuda", dtype=torch.int32)
    k_raw, w, cos, sin = _make(2, s_tok, dtype, seed=5)
    nb = s_tok // _POOL
    out = _sentinel(2, nb, dtype)
    sentinel = out.clone()
    build_qsa_compress(k_raw, out, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
    torch.cuda.synchronize()
    want, n_blocks = compress_reference(k_raw, w, cos, sin, seq_lens, norm_weight_offset=1.0)
    assert n_blocks.tolist() == [nb, 0]
    _assert_one_rounding(out[0], want[0], torch.ones_like(out[0], dtype=torch.bool), "clamped above S")
    assert torch.equal(out[1], sentinel[1])


@requires_cuda
def test_no_complete_block_launches_nothing_and_extra_rows_stay_untouched():
    dtype = torch.bfloat16
    # S = 3: no complete block anywhere -> no launch, an empty output is legal.
    k_raw, w, cos, sin = _make(2, 3, dtype, seed=9)
    out0 = torch.empty(2, 0, _D, device="cuda", dtype=dtype)
    r = build_qsa_compress(k_raw, out0, w, cos, sin, None, rope_dim=_ROPE, eps=_EPS, stream=_stream())
    assert isinstance(r, QsaCompressRecipe)
    # S = 9 -> 2 complete blocks into a 5-block cache: rows 2..4 are never written.
    k_raw, w, cos, sin = _make(2, 9, dtype, seed=10)
    out = _sentinel(2, 5, dtype)
    sentinel = out.clone()
    build_qsa_compress(k_raw, out, w, cos, sin, None, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
    torch.cuda.synchronize()
    want, _ = compress_reference(k_raw, w, cos, sin, None, norm_weight_offset=1.0)
    _assert_one_rounding(out[:, :2], want, torch.ones_like(out[:, :2], dtype=torch.bool), "S=9 into a 5-block cache")
    assert torch.equal(out[:, 2:], sentinel[:, 2:])


@requires_cuda
def test_slab_token_stride_is_bitwise_the_compact_run():
    """The raw keys as the INDEX band of a ``[B, S, 640]`` projection slab
    (token stride 640) and as a compact copy give bitwise-equal output: one
    artifact, symbolic strides, no repack."""
    batch, s_tok, dtype = 2, 300, torch.bfloat16
    k_slab, w, cos, sin = _make(batch, s_tok, dtype, seed=4, slab=True)
    assert k_slab.stride(1) == 640
    k_compact = k_slab.contiguous()
    nb = s_tok // _POOL
    out_slab = torch.empty(batch, nb, _D, device="cuda", dtype=dtype)
    out_compact = torch.empty_like(out_slab)
    r1 = build_qsa_compress(k_slab, out_slab, w, cos, sin, None, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
    r2 = build_qsa_compress(k_compact, out_compact, w, cos, sin, None, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
    torch.cuda.synchronize()
    assert r1.compiled is r2.compiled
    assert torch.equal(out_slab, out_compact)
    want, _ = compress_reference(k_compact, w, cos, sin, None, norm_weight_offset=1.0)
    _assert_one_rounding(out_slab, want, torch.ones_like(out_slab, dtype=torch.bool), "slab")


@requires_cuda
def test_broadcast_table_matches_per_entry_tables():
    """``[1, S, R]``, an expanded ``[B, S, R]`` view (batch stride 0) and a
    materialised ``[B, S, R]`` table are bitwise interchangeable."""
    batch, s_tok, dtype = 3, 200, torch.float16
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=6, table_batch=1)
    assert cos.shape[0] == 1
    nb = s_tok // _POOL
    outs = []
    for c, s in (
        (cos, sin),
        (cos.expand(batch, -1, -1), sin.expand(batch, -1, -1)),
        (cos.expand(batch, -1, -1).contiguous(), sin.expand(batch, -1, -1).contiguous()),
    ):
        out = torch.empty(batch, nb, _D, device="cuda", dtype=dtype)
        build_qsa_compress(k_raw, out, w, c, s, None, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
    want, _ = compress_reference(k_raw, w, cos, sin, None, norm_weight_offset=1.0)
    _assert_one_rounding(outs[0], want, torch.ones_like(outs[0], dtype=torch.bool), "broadcast table")


@requires_cuda
def test_two_launches_are_bitwise():
    batch, s_tok, dtype = 2, 1024, torch.bfloat16
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=8)
    seq_lens = torch.tensor([1024, 1000], device="cuda", dtype=torch.int32)
    nb = s_tok // _POOL
    a = _sentinel(batch, nb, dtype)
    b = _sentinel(batch, nb, dtype)
    build_qsa_compress(k_raw, a, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
    build_qsa_compress(k_raw, b, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(a, b)


@requires_cuda
def test_one_artifact_serves_every_shape():
    r_small = compile_qsa_compress(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, has_seq_lens=True)
    r_again = compile_qsa_compress(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, has_seq_lens=True)
    assert r_small.compiled is r_again.compiled
    for s_tok in (5, 77, 4096):
        k_raw, w, cos, sin = _make(1, s_tok, torch.bfloat16, seed=s_tok)
        seq_lens = torch.tensor([s_tok], device="cuda", dtype=torch.int32)
        out = torch.empty(1, s_tok // _POOL, _D, device="cuda", dtype=torch.bfloat16)
        run_qsa_compress(r_small, k_raw, out, w, cos, sin, seq_lens, eps=_EPS, norm_weight_offset=1.0, stream=_stream())
        torch.cuda.synchronize()
        want, _ = compress_reference(k_raw, w, cos, sin, seq_lens, norm_weight_offset=1.0)
        _assert_one_rounding(out, want, torch.ones_like(out, dtype=torch.bool), f"one artifact, S={s_tok}")


# ---------------------------------------------------------------------------
# Host-side contract: every refusal is typed and names the operand
# ---------------------------------------------------------------------------


@requires_cuda
def test_run_rejects_contract_violations():
    batch, s_tok, dtype = 2, 64, torch.bfloat16
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=12)
    nb = s_tok // _POOL
    out = torch.empty(batch, nb, _D, device="cuda", dtype=dtype)
    dense = compile_qsa_compress(dtype=dtype, d=_D, rope_dim=_ROPE, has_seq_lens=False)
    ragged = compile_qsa_compress(dtype=dtype, d=_D, rope_dim=_ROPE, has_seq_lens=True)
    seq_lens = torch.full((batch,), s_tok, device="cuda", dtype=torch.int32)

    def run(r, k=k_raw, o=out, wt=w, c=cos, s=sin, lens=None):
        run_qsa_compress(r, k, o, wt, c, s, lens, eps=_EPS, stream=_stream())

    with pytest.raises(ValueError, match="WITHOUT seq_lens"):
        run(dense, lens=seq_lens)
    with pytest.raises(ValueError, match="WITH per-batch seq_lens"):
        run(ragged)
    with pytest.raises(ValueError, match="int32"):
        run(ragged, lens=seq_lens.to(torch.int64))
    with pytest.raises(ValueError, match="holds .* blocks"):
        run(dense, o=torch.empty(batch, nb - 1, _D, device="cuda", dtype=dtype))
    with pytest.raises(ValueError, match="dtype"):
        run(dense, k=k_raw.float())
    with pytest.raises(ValueError, match="positions"):
        run(dense, c=cos[:, : s_tok - 8], s=sin[:, : s_tok - 8])
    with pytest.raises(ValueError, match="batch dim"):
        run(dense, c=cos[:1].expand(batch + 1, -1, -1), s=sin[:1].expand(batch + 1, -1, -1))
    with pytest.raises(ValueError, match="share a shape"):
        run(dense, s=sin[:1])
    with pytest.raises(ValueError, match="norm weight"):
        run(dense, wt=w[:64])
    with pytest.raises(ValueError, match="multiple of 8 elements"):
        run(dense, k=torch.randn(batch, s_tok, 132, device="cuda").to(dtype)[:, :, :_D])
    with pytest.raises(ValueError, match="compact"):
        run(dense, o=torch.empty(batch, nb, 2 * _D, device="cuda", dtype=dtype)[:, :, :_D])
    with pytest.raises(ValueError, match=r"\[B, S, D\]"):
        run(dense, k=k_raw[0])
