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
and the oracle differ only in fp32 op order) -- the ulp measured at the LARGER
ROTATION OPERAND's magnitude on the rotated dims (``_assert_one_rounding``: the
fp32 op-order difference of the operands survives a cancelling ``y cos + partner
sin`` unscaled while the result's own ulp shrinks with the result) -- and nothing
written where the kernel owes no block.

The kernel is plain vectorized LDG/STG plus warp shuffles -- no tcgen05, no TMA
-- so these tests deliberately do NOT gate on Rubin: they run on whatever CUDA
device is at hand.

The e4m3 STORE arm (``compile_qsa_compress(out_dtype=torch.float8_e4m3fn)``: the unfused decode step's pool-1 query path, and the
compressed rows at pool 4; sm_89+ -- the cell skips below it and is UNCOVERED there): the e4m3 run is bitwise the bf16 run's rows
``.to(float8_e4m3fn)`` (the e4m3 cast of the bf16 row, ``|bf16 row| < 448`` asserted first); ``out_dtype`` is a compile key (a distinct
artifact; the bf16 spelling is the io dtype's artifact); a wrong ``out`` dtype is refused both ways naming the recipe's; f16 io has no
e4m3 arm; float32 stays a ``ValueError``; below sm_89 the arm is a typed decline naming the instruction.

The rounding attribute (``compile_qsa_compress(rounding=)``, ``INDEX_COMPRESS_ROUNDINGS``; 2026-10-29): ``"single"`` is the chain above;
``"hf"`` is the HF / vLLM-prepare dtype flow -- the pooled key cast to the io dtype before its norm, the normed row cast before its
rotation, the rotation in the io dtype -- the oracle's ``hf_rounding=True`` points.  The ``"hf"`` arm reproduces the oracle's ROUNDING
POINTS, not its bits (the fp32 reduction orders differ: the 16 per-lane partial sums of squares + the fast-math rsqrt vs torch's reduce), so its
gate is a STATISTIC with a hard cap -- at least ``_HF_BITWISE_FLOOR`` (99.9 %) of the elements bitwise the HF-rounding oracle and every
element within ``_HF_ULP_CAP`` (6) io ulps AT THE LARGER ROTATION OPERAND'S MAGNITUDE (the derivation in ``_HF_BITWISE_FLOOR``'s
docstring: a one-ulp flip of an operand moves ``io(io(y cos) + io(partner sin))`` by at most two of THAT operand's ulps plus the
result's own rounding, both operands flipping by five; the result's own ulp is no yardstick, a cancelling sum makes it arbitrarily
small) -- while the ``"single"`` arm against the SAME oracle agrees on ~70 % of the elements (the FI README's "one bf16 ulp on a third
of the elements"): the two conventions are told apart by the statistic, at pool 4 (the key) and pool 1 (the query form).  The attribute
is a per-recipe compile key extended off the default only (the default artifacts keep their keys).
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
    INDEX_COMPRESS_ROUNDINGS,
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
_CC = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
requires_sm89 = pytest.mark.skipif(_CC is None or _CC < (8, 9), reason=f"the e4m3 store arm needs sm_89+ (found {_CC})")
_E4M3 = torch.float8_e4m3fn

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


def _normed_rows(k_raw, w, *, pool, eps, norm_weight_offset, hf_rounding):
    """The pooled, RMSNormed rows ``[B, floor(S / pool), D]`` BEFORE the rotation: fp32 (``hf_rounding=False``) or the io dtype with HF's two
    casts -- the pooled key before its norm, the normed row before its rotation (``hf_rounding=True``)."""
    batch, s_tok, d = k_raw.shape
    nb = s_tok // pool
    io = k_raw.dtype
    pooled = k_raw[:, : nb * pool].float().reshape(batch, nb, pool, d).mean(dim=2)
    if hf_rounding:
        pooled = pooled.to(io).float()
    rstd = torch.rsqrt(pooled.pow(2).mean(-1, keepdim=True) + eps)
    y = pooled * rstd * (w.float() + norm_weight_offset)
    return y.to(io) if hf_rounding else y


def hf_rotation_operand_magnitude(k_raw, w, *, pool=_POOL, eps=_EPS, norm_weight_offset=0.0, rope_dim=_ROPE, hf_rounding=True):
    """Per element of the oracle's output, the magnitude of the LARGER rotation operand -- ``max(|y_i|, |y_partner|)`` on the rotated
    dims (the two terms of ``y * cos + rotate_half(y) * sin`` can cancel, so the result's own ulp is no yardstick for a one-ulp flip
    of an operand), ``|y_i|`` on the passthrough dims -- the unit the ``"hf"`` gate's cap is stated in (``hf_rounding=True``: the
    HF-rounding oracle's io-dtype operands) and, appended, the unit of the ``"single"`` arm's one-rounding budget
    (``hf_rounding=False``: the fp32 oracle's operands; :func:`_assert_one_rounding`)."""
    y = _normed_rows(k_raw, w, pool=pool, eps=eps, norm_weight_offset=norm_weight_offset, hf_rounding=hf_rounding).float().abs()
    if rope_dim:
        half = rope_dim // 2
        partner = torch.cat((y[..., half:rope_dim], y[..., :half]), dim=-1)
        y = torch.cat((torch.maximum(y[..., :rope_dim], partner), y[..., rope_dim:]), dim=-1)
    return y


def compress_reference(k_raw, w, cos, sin, seq_lens, *, pool=_POOL, eps=_EPS, norm_weight_offset=0.0, rope_dim=_ROPE, hf_rounding=False):
    """fp32 compressed keys ``[B, floor(S / pool), D]`` and the per-entry valid
    block counts ``[B]``.

    ``k_raw`` ``[B, S, D]``; ``w`` ``[D]`` (``norm_weight_offset`` is added in
    fp32: 0.0 for a pre-folded ``(1 + w)``, 1.0 for the raw zero-centred
    weight); ``cos`` / ``sin`` ``[1 or B, S, rope_dim]`` duplicated-half
    tables, read at the block START position ``pool * b``; ``seq_lens`` ``[B]``
    or None (= S).  Blocks at or past an entry's count are not the kernel's
    to write -- the caller masks them out of the comparison.

    ``hf_rounding=True`` (appended): HF's dtype flow -- the pooled key cast to the key dtype BEFORE its norm, the normed row (every
    dim) cast to it before the rotation, the rotation in that dtype (``rot * c + rotate_half(rot) * s``, each op rounded, as
    ``gated_block_qsa_reference.qsa_indexer_reference(hf_rounding=True)`` spells it); the values returned are fp32 but io-representable.
    """
    batch, s_tok, d = k_raw.shape
    nb = s_tok // pool
    y = _normed_rows(k_raw, w, pool=pool, eps=eps, norm_weight_offset=norm_weight_offset, hf_rounding=hf_rounding)
    if rope_dim:
        pos = torch.arange(nb, device=k_raw.device) * pool
        c = cos[:, pos, :rope_dim].to(y.dtype)
        s = sin[:, pos, :rope_dim].to(y.dtype)
        yr = y[..., :rope_dim]
        half = rope_dim // 2
        rotated = torch.cat((-yr[..., half:], yr[..., :half]), dim=-1)
        y = torch.cat((yr * c + rotated * s, y[..., rope_dim:]), dim=-1)
    y = y.float()
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


def _assert_one_rounding(got: torch.Tensor, want32: torch.Tensor, mask: torch.Tensor, what: str, operand_mag=None) -> float:
    """Every masked element of ``got`` within one io-dtype ulp of the oracle's rounded value; returns the bitwise-equal fraction
    (reported, not gated).  ``operand_mag`` (appended; :func:`hf_rotation_operand_magnitude` with ``hf_rounding=False``) states the
    ulp at the LARGER of the rotation operand's magnitude and the two results': the kernel's fp32 chain and torch's differ by a few
    fp32 ulps OF THE OPERANDS (the 16 per-lane partial sums + the fast rsqrt vs torch's reduce), a difference a cancelling
    ``y cos + partner sin`` keeps unscaled while the result's own ulp shrinks with the result -- measured against the result's ulp
    alone, one f16 element in 524288 (an H100 host, S 2048, pool 1, a rotated dim, |result| ~ 2e-4) sat TWO of ITS ulps off (2.4e-7)
    with every other element bitwise or one ulp off.  Without ``operand_mag`` the yardstick is the result's own (the passthrough band)."""
    assert torch.isfinite(want32).all(), f"{what}: the oracle itself is non-finite"
    assert torch.isfinite(got[mask]).all(), f"{what}: non-finite kernel output"
    want = want32.to(got.dtype)
    diff = (got.float() - want.float()).abs()
    at = want.float().abs() if operand_mag is None else torch.maximum(operand_mag.float(), torch.maximum(got.float().abs(), want.float().abs()))
    bad = (diff > _ulp(at, got.dtype)) & mask
    if bad.any():
        where = bad.nonzero()[:8].tolist()
        unit = "" if operand_mag is None else " at the larger rotation operand's magnitude"
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {int(mask.sum())} elements beyond one {got.dtype} ulp of the oracle{unit}; "
            f"max |diff| among them {diff[bad].max().item():.3e}, first indices {where}"
        )
    return (got[mask] == want[mask]).float().mean().item()


_HF_BITWISE_FLOOR = 0.999
"""The ``"hf"`` arm's gate against the HF-rounding oracle: the fraction of elements bitwise equal (measured 99.997 .. 100.000 % at S 5 ..
4096, bf16 and f16, on an A100 host; the flips are the rare io roundings an fp32 reduction-order / fast-rsqrt difference moves), with
every element within ``_HF_ULP_CAP`` io ulps OF THE LARGER ROTATION OPERAND (:func:`hf_rotation_operand_magnitude`): a one-ulp flip of an
operand moves the io-dtype rotation's result by at most two of that operand's ulps plus the result's own rounding, both operands flipping
by five -- never measured against the result's own ulp, which a cancelling sum makes arbitrarily small (one f16 element at S 512 / 4096 sat
beyond four of ITS ulps while within one of its operand's)."""
_HF_ULP_CAP = 6.0
_SINGLE_VS_HF_CEILING = 0.95
"""The discrimination: the ``"single"`` arm against the HF-rounding oracle agrees on ~70 % of the elements (measured 0.712); a kernel
that ignored the attribute would pass the first gate and fail this one."""


def _bitwise_fraction(got: torch.Tensor, want32: torch.Tensor, mask: torch.Tensor) -> float:
    want = want32.to(got.dtype)
    return (got[mask] == want[mask]).float().mean().item()


def _assert_hf_rounding_points(got: torch.Tensor, want32: torch.Tensor, mask: torch.Tensor, what: str, operand_mag: torch.Tensor) -> float:
    """The ``"hf"`` arm's gate: ``>= _HF_BITWISE_FLOOR`` of the masked elements bitwise the HF-rounding oracle's rounded value and every
    one within ``_HF_ULP_CAP`` io ulps at the larger of the rotation operand's magnitude (``operand_mag``) and the two results';
    returns the bitwise fraction."""
    assert torch.isfinite(want32).all(), f"{what}: the oracle itself is non-finite"
    assert torch.isfinite(got[mask]).all(), f"{what}: non-finite kernel output"
    want = want32.to(got.dtype)
    diff = (got.float() - want.float()).abs()
    ulp = _ulp(torch.maximum(operand_mag.float(), torch.maximum(got.float().abs(), want.float().abs())), got.dtype)
    over = (diff > _HF_ULP_CAP * ulp) & mask
    if over.any():
        where = over.nonzero()[:8].tolist()
        raise AssertionError(f"{what}: {int(over.sum())} elements beyond {_HF_ULP_CAP} io ulps of the HF-rounding oracle; first indices {where}")
    frac = _bitwise_fraction(got, want32, mask)
    assert frac >= _HF_BITWISE_FLOOR, f"{what}: only {frac * 100:.3f} % of the elements bitwise the HF-rounding oracle (floor {_HF_BITWISE_FLOOR * 100:.1f} %)"
    return frac


def _valid_mask(n_blocks: torch.Tensor, nb: int) -> torch.Tensor:
    batch = int(n_blocks.shape[0])
    return (torch.arange(nb, device=n_blocks.device)[None, :] < n_blocks[:, None])[:, :, None].expand(batch, nb, _D)


def _make(batch, s_tok, dtype, *, seed=0, slab=False, table_batch=None, table_dtype=None):
    """Raw keys (optionally as the INDEX band of a ``[B, S, 640]`` slab, token
    stride 640), a small zero-centred weight, and real RoPE tables at the
    model's theta (identical rows across the batch); ``table_dtype`` (appended) builds the tables in another 16-bit width than the io."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32)

    if slab:
        k_raw = rnd(batch, s_tok, 640).to(dtype)[:, :, 512:640]
    else:
        k_raw = rnd(batch, s_tok, _D).to(dtype)
    w = (0.05 * rnd(_D)).to(dtype)
    cos, sin = build_rope_tables(
        s_tok, _ROPE, base=_THETA, batch=batch if table_batch is None else table_batch, device="cuda", dtype=dtype if table_dtype is None else table_dtype
    )
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
    # the e4m3 store arm's rows: a non-io, non-e4m3 output dtype and f16 io with an e4m3 output are ValueErrors naming the rule
    with pytest.raises(ValueError, match="must hold the io dtype .* or torch.float8_e4m3fn"):
        compile_qsa_compress(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, out_dtype=torch.float32)
    with pytest.raises(ValueError, match="needs bf16 io"):
        compile_qsa_compress(dtype=torch.float16, d=_D, rope_dim=_ROPE, out_dtype=_E4M3)


@requires_cuda
def test_e4m3_out_arm_is_a_typed_decline_below_sm89_and_keyed_above_it():
    """Below sm_89 the e4m3 store arm is a ``NotImplementedError`` naming the instruction (never a ptxas error); from sm_89 it compiles as
    a DISTINCT artifact whose recipe records ``out_dtype``, while the bf16 spelling of the io dtype is the io artifact itself."""
    if _CC < (8, 9):
        with pytest.raises(NotImplementedError, match=r"cvt.rn.satfinite.e4m3x2.f32\) needs sm_89 or later"):
            compile_qsa_compress(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, out_dtype=_E4M3)
        return
    bf = compile_qsa_compress(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, pool=1, has_seq_lens=False)
    assert compile_qsa_compress(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, pool=1, has_seq_lens=False, out_dtype=torch.bfloat16).compiled is bf.compiled
    e4 = compile_qsa_compress(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, pool=1, has_seq_lens=False, out_dtype=_E4M3)
    assert e4.compiled is not bf.compiled and e4.out_dtype == _E4M3 and bf.out_dtype == torch.bfloat16
    assert compile_qsa_compress(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, pool=1, has_seq_lens=False, out_dtype=_E4M3).compiled is e4.compiled


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("rounding", ["single", "hf"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("offset", [0.0, 1.0], ids=["folded_1_plus_w", "zero_centred_w"])
@pytest.mark.parametrize("s_tok", [5, 512, 2051, 4096])
def test_matches_the_oracle_within_one_rounding(s_tok, offset, dtype, rounding):
    """Three entries: a full one, one 7 tokens short (a ragged tail), an EMPTY
    one.  ``"single"``: every block an entry owns is within one ulp of the fp32 oracle; ``"hf"``: the HF-rounding oracle's points
    (the statistic + the cap of the module docstring); every other row of ``out`` keeps its sentinel."""
    batch = 3
    k_raw, w, cos, sin = _make(batch, s_tok, dtype)
    w_in = (1.0 + w.float()).to(dtype) if offset == 0.0 else w
    seq_lens = torch.tensor([s_tok, max(s_tok - 7, 0), 0], device="cuda", dtype=torch.int32)
    nb = s_tok // _POOL
    out = _sentinel(batch, nb, dtype)
    sentinel = out.clone()

    r = build_qsa_compress(k_raw, out, w_in, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=offset, stream=_stream(), rounding=rounding)
    torch.cuda.synchronize()
    assert r.has_seq_lens and r.pool == _POOL and r.d == _D and r.rounding == rounding

    want, n_blocks = compress_reference(k_raw, w_in, cos, sin, seq_lens, norm_weight_offset=offset, hf_rounding=rounding == "hf")
    assert n_blocks.tolist() == [s_tok // _POOL, max(s_tok - 7, 0) // _POOL, 0]
    mask = _valid_mask(n_blocks, nb)
    if rounding == "hf":
        mag = hf_rotation_operand_magnitude(k_raw, w_in, norm_weight_offset=offset)
        exact = _assert_hf_rounding_points(out, want, mask, f"hf S={s_tok} {dtype} offset={offset}", mag)
    else:
        mag32 = hf_rotation_operand_magnitude(k_raw, w_in, norm_weight_offset=offset, hf_rounding=False)
        exact = _assert_one_rounding(out, want, mask, f"S={s_tok} {dtype} offset={offset}", mag32)
    assert torch.equal(out[~mask], sentinel[~mask]), "a block the kernel does not own was written"
    print(f"S={s_tok} {dtype} offset={offset} rounding={rounding}: {int(mask.sum())} elements checked, {exact * 100:.3f}% bitwise equal to the rounded oracle")


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("pool, s_tok", [(4, 4096), (1, 2048)], ids=["pool4_keys", "pool1_queries"])
def test_hf_rounding_is_the_hf_oracles_rounding_points_and_single_is_not(pool, s_tok, dtype):
    """The DISCRIMINATION at the key form (pool 4) and the query form (pool 1): the ``"hf"`` arm against the HF-rounding oracle passes the
    statistic (>= ``_HF_BITWISE_FLOOR`` bitwise, every element within ``_HF_ULP_CAP`` (6) io ulps at the larger rotation operand's
    magnitude -- the unit and its derivation in ``_HF_BITWISE_FLOOR``'s docstring) while the ``"single"`` arm against the SAME oracle
    agrees on well under 95 % of the elements (measured ~71 %), and the converse -- the ``"single"`` arm is within one ulp of the fp32 oracle at the
    larger rotation operand's magnitude (``_assert_one_rounding``'s yardstick) AND bitwise the default compile's output (the same
    artifact: the honest statement that the single path did not change).  A kernel that ignored the attribute would fail the first
    statement; one that applied it to the wrong site would fail the gate's cap."""
    batch = 2
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=50 + pool)
    seq_lens = None if pool == 1 else torch.tensor([s_tok, s_tok - 3], device="cuda", dtype=torch.int32)
    nb = s_tok // pool
    outs = {}
    for rounding in INDEX_COMPRESS_ROUNDINGS:
        out = _sentinel(batch, nb, dtype)
        build_qsa_compress(k_raw, out, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, pool=pool, stream=_stream(), rounding=rounding)
        outs[rounding] = out
    torch.cuda.synchronize()
    want_hf, n_blocks = compress_reference(k_raw, w, cos, sin, seq_lens, pool=pool, norm_weight_offset=1.0, hf_rounding=True)
    want_32, _ = compress_reference(k_raw, w, cos, sin, seq_lens, pool=pool, norm_weight_offset=1.0)
    mask = _valid_mask(n_blocks, nb)
    mag = hf_rotation_operand_magnitude(k_raw, w, pool=pool, norm_weight_offset=1.0)
    frac_hf = _assert_hf_rounding_points(outs["hf"], want_hf, mask, f"hf pool {pool} {dtype}", mag)
    frac_single_vs_hf = _bitwise_fraction(outs["single"], want_hf, mask)
    assert (
        frac_single_vs_hf < _SINGLE_VS_HF_CEILING
    ), f"the single arm agrees with the HF-rounding oracle on {frac_single_vs_hf * 100:.2f} %: the attribute changed nothing"
    mag32 = hf_rotation_operand_magnitude(k_raw, w, pool=pool, norm_weight_offset=1.0, hf_rounding=False)
    _assert_one_rounding(outs["single"], want_32, mask, f"single pool {pool} {dtype}", mag32)
    # the deterministic twin: the "single" arm IS the default compile's output (the same artifact under the same key), bitwise
    default = _sentinel(batch, nb, dtype)
    build_qsa_compress(k_raw, default, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, pool=pool, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(outs["single"], default), "the single arm differs from the default compile's output"
    assert not torch.equal(outs["single"], outs["hf"])
    print(f"pool {pool} {dtype}: hf vs HF oracle {frac_hf * 100:.3f} % bitwise; single vs HF oracle {frac_single_vs_hf * 100:.2f} % bitwise")


@requires_cuda
@pytest.mark.parametrize(
    "io_dtype, table_dtype", [(torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)], ids=["bf16_io_f16_tables", "f16_io_bf16_tables"]
)
@pytest.mark.parametrize("pool, batch, s_tok, seed", [(4, 1, 512, 54), (4, 2, 2048, 55), (1, 1, 1024, 56)], ids=["pool4_S512", "pool4_S2048", "pool1_S1024"])
def test_hf_rounding_rounds_a_mixed_dtype_table_to_the_io_dtype(io_dtype, table_dtype, pool, batch, s_tok, seed):
    """The compile APIs accept a 16-bit cos / sin table of the OTHER width than the io (``table_dtype`` is a compile key): under ``"hf"`` the
    kernel rounds the table values to the io dtype BEFORE the rotation -- the HF reference rotates with the tables cast to the key dtype
    and the oracle (``hf_rounding=True``) does the same -- so the statistic gate holds (f16 tables under bf16 io sat at 93.98 % bitwise before
    the rounding was added: a systematic rounding-point difference, 986 of 16384 elements at pool 4 / S 512 / seed 54); under ``"single"``
    the fp32 chain takes the table's exact value and stays within one io ulp of the fp32 oracle."""
    k_raw, w, cos, sin = _make(batch, s_tok, io_dtype, seed=seed, table_dtype=table_dtype)
    assert cos.dtype == table_dtype and k_raw.dtype == io_dtype
    seq_lens = torch.tensor([s_tok] * batch, device="cuda", dtype=torch.int32) if pool == 4 else None
    nb = s_tok // pool
    outs = {}
    for rounding in INDEX_COMPRESS_ROUNDINGS:
        out = _sentinel(batch, nb, io_dtype)
        r = build_qsa_compress(
            k_raw, out, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=1.0, pool=pool, stream=_stream(), rounding=rounding
        )
        assert r.table_dtype == table_dtype and r.dtype == io_dtype and r.rounding == rounding
        outs[rounding] = out
    torch.cuda.synchronize()
    want_hf, n_blocks = compress_reference(k_raw, w, cos, sin, seq_lens, pool=pool, norm_weight_offset=1.0, hf_rounding=True)
    want_32, _ = compress_reference(k_raw, w, cos, sin, seq_lens, pool=pool, norm_weight_offset=1.0)
    mask = _valid_mask(n_blocks, nb)
    frac_hf = _assert_hf_rounding_points(
        outs["hf"],
        want_hf,
        mask,
        f"hf {io_dtype} io / {table_dtype} tables pool {pool}",
        hf_rotation_operand_magnitude(k_raw, w, pool=pool, norm_weight_offset=1.0),
    )
    frac_single = _assert_one_rounding(
        outs["single"],
        want_32,
        mask,
        f"single {io_dtype} io / {table_dtype} tables pool {pool}",
        hf_rotation_operand_magnitude(k_raw, w, pool=pool, norm_weight_offset=1.0, hf_rounding=False),
    )
    print(
        f"{io_dtype} io / {table_dtype} tables pool {pool} S {s_tok}: hf {frac_hf * 100:.3f} % bitwise the HF-rounding oracle; single {frac_single * 100:.3f} % bitwise the fp32 oracle"
    )


@requires_cuda
def test_rounding_is_a_per_recipe_key_and_the_default_key_is_unchanged():
    """``compile_qsa_compress(rounding="single")`` returns TODAY's artifact (the same cache key as no argument), ``"hf"`` a distinct one at
    pool 4 AND at pool 1 (per recipe: the key recipe and the query recipe each take it), the e4m3 arm composes with it; an unknown value
    is a ``ValueError`` naming ``INDEX_COMPRESS_ROUNDINGS``."""
    assert INDEX_COMPRESS_ROUNDINGS == ("single", "hf")
    for pool in (4, 1):
        base = dict(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, pool=pool, has_seq_lens=pool == 4)
        r0 = compile_qsa_compress(**base)
        assert compile_qsa_compress(rounding="single", **base).compiled is r0.compiled and r0.rounding == "single"
        rh = compile_qsa_compress(rounding="hf", **base)
        assert rh.compiled is not r0.compiled and rh.rounding == "hf"
        assert compile_qsa_compress(rounding="hf", **base).compiled is rh.compiled
        if _CC is not None and _CC >= (8, 9):
            e4 = compile_qsa_compress(rounding="hf", out_dtype=_E4M3, **base)
            assert e4.compiled is not rh.compiled and e4.compiled is not compile_qsa_compress(out_dtype=_E4M3, **base).compiled
    with pytest.raises(ValueError, match=r"rounding must be one of \('single', 'hf'\)"):
        compile_qsa_compress(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, rounding="vllm")


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
    _assert_one_rounding(
        out,
        want,
        torch.ones_like(out, dtype=torch.bool),
        f"dense S={s_tok}",
        hf_rotation_operand_magnitude(k_raw, w, norm_weight_offset=1.0, hf_rounding=False),
    )


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
    _assert_one_rounding(out, want, mask, "S=13", hf_rotation_operand_magnitude(k_raw, w, norm_weight_offset=1.0, hf_rounding=False))
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
    mag32 = hf_rotation_operand_magnitude(k_raw, w, norm_weight_offset=1.0, hf_rounding=False)
    _assert_one_rounding(out[0], want[0], torch.ones_like(out[0], dtype=torch.bool), "clamped above S", mag32[0])
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
    mag32 = hf_rotation_operand_magnitude(k_raw, w, norm_weight_offset=1.0, hf_rounding=False)
    _assert_one_rounding(out[:, :2], want, torch.ones_like(out[:, :2], dtype=torch.bool), "S=9 into a 5-block cache", mag32)
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
    _assert_one_rounding(
        out_slab,
        want,
        torch.ones_like(out_slab, dtype=torch.bool),
        "slab",
        hf_rotation_operand_magnitude(k_compact, w, norm_weight_offset=1.0, hf_rounding=False),
    )


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
    _assert_one_rounding(
        outs[0],
        want,
        torch.ones_like(outs[0], dtype=torch.bool),
        "broadcast table",
        hf_rotation_operand_magnitude(k_raw, w, norm_weight_offset=1.0, hf_rounding=False),
    )


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


@requires_sm89
@pytest.mark.parametrize(
    "pool, s_tok, offset", [(1, 4, 0.0), (1, 77, 0.0), (4, 2051, 1.0), (4, 4096, 1.0)], ids=["pool1_T4", "pool1_T77", "pool4_S2051", "pool4_S4096"]
)
def test_e4m3_out_is_bitwise_the_bf16_run_cast(pool, s_tok, offset):
    """The e4m3 store arm against the bf16 arm on the SAME inputs: every written row of the e4m3 ``out`` is bitwise ``bf16_out.to(e4m3)``
    (``|bf16 row| < 448`` first), the unwritten rows keep the 8-bit sentinel (-448: the dtype's max magnitude, never a written row); pool 1 at a token
    count = the unfused decode step's indexer-query path (``[T, H_i, D]`` as ``[B=T, S=H_i, D]``), pool 4 the compressed rows; the
    bf16 run is the one-rounding oracle's as before.  A wrong ``out`` dtype is refused BOTH ways, naming the recipe's dtype."""
    dtype, batch = torch.bfloat16, 3
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=40 + s_tok)
    seq_lens = torch.tensor([s_tok, max(s_tok - 7 * (pool // 4), 0), 0], device="cuda", dtype=torch.int32) if pool == 4 else None
    nb = s_tok // pool
    out16 = _sentinel(batch, nb, dtype)
    out8 = torch.full(
        (batch, nb, _D), -448.0, device="cuda", dtype=_E4M3
    )  # the e4m3 sentinel: its max magnitude (torch.full refuses -7777 at e4m3); never a written row
    r16 = build_qsa_compress(k_raw, out16, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=offset, pool=pool, stream=_stream())
    r8 = build_qsa_compress(k_raw, out8, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=offset, pool=pool, stream=_stream())
    torch.cuda.synchronize()
    assert r8.out_dtype == _E4M3 and r16.out_dtype == dtype and r8.compiled is not r16.compiled
    want, n_blocks = compress_reference(k_raw, w, cos, sin, seq_lens, pool=pool, norm_weight_offset=offset)
    mask = _valid_mask(n_blocks, nb)
    _assert_one_rounding(
        out16, want, mask, f"bf16 twin pool={pool} S={s_tok}", hf_rotation_operand_magnitude(k_raw, w, pool=pool, norm_weight_offset=offset, hf_rounding=False)
    )
    assert float(out16[mask].float().abs().max()) < 448.0, "a bf16 row at or past the e4m3 range"
    bits8 = out8.view(torch.int8)
    assert torch.equal(bits8[mask], out16.to(_E4M3).view(torch.int8)[mask]), "the e4m3 out is not the e4m3 cast of the bf16 rows"
    sent8 = -448.0
    assert bool((out8.float()[~mask] == sent8).all()), "a row the kernel does not own was written"
    with pytest.raises(ValueError, match=r"out must have dtype torch.float8_e4m3fn \(the recipe's\), got torch.bfloat16"):
        run_qsa_compress(r8, k_raw, out16, w, cos, sin, seq_lens, eps=_EPS, norm_weight_offset=offset, stream=_stream())
    with pytest.raises(ValueError, match=r"out must have dtype torch.bfloat16 \(the recipe's\), got torch.float8_e4m3fn"):
        run_qsa_compress(r16, k_raw, out8, w, cos, sin, seq_lens, eps=_EPS, norm_weight_offset=offset, stream=_stream())
    print(f"e4m3 store arm pool={pool} S={s_tok}: {int(mask.sum())} elements bitwise the bf16 cast")


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
        _assert_one_rounding(
            out,
            want,
            torch.ones_like(out, dtype=torch.bool),
            f"one artifact, S={s_tok}",
            hf_rotation_operand_magnitude(k_raw, w, norm_weight_offset=1.0, hf_rounding=False),
        )


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
