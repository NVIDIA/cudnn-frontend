# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-1 (no GPU) pins for the FP4 half of the gated block's reference oracle.

``gated_block_reference`` rounds to E2M1 BY HAND (torch 2.13 cannot cast to or from
``float4_e2m1fn_x2``), so before any kernel is measured against it the oracle itself is pinned
against two INDEPENDENT in-tree spellings of the same facts: the torchao port
``_f32_to_floatx_unpacked(x, 2, 1)`` (``test/python/test_low_precision_matmul.py``) for the E2M1
rounding, and the FROST GEMM suite's ``unpack_fp4`` / ``to_blocked`` (``gemm_test_utils``) for the
nibble order and the F8_128x4 blob.  Both are loaded BY PATH so this module keeps the block
suite's import shape (its own directory only on ``sys.path``).

The blob helpers gained ``block=`` (16 for NVFP4) here; the default stays 32 and the MXFP8 suites
are bitwise pins on it, so the default path is asserted equal to the explicit one.
"""

import importlib.util
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.L0

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gated_block_reference import (  # noqa: E402
    E2M1_GRID,
    E4M3_MIN_SUBNORMAL,
    FP4_FORMATS,
    FP4_INV_MAX,
    MX_ATOM_COLS,
    MX_ATOM_ROWS,
    e2m1_codes,
    e2m1_rne,
    e2m1_values,
    fp4_dequant_rowwise_2d,
    fp4_format,
    fp4_quantize_rowwise_2d,
    mx_sf_padded_dims,
    mx_swizzle_sf_rowwise_padded,
    mx_unswizzle_sf_rowwise,
    pack_e2m1,
    unpack_e2m1,
)

_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# midpoint -> the EVEN code's value (ties-to-even on the CODE, not on the value)
_MIDPOINT_TABLE = {0.25: 0.0, 0.75: 1.0, 1.25: 1.0, 1.75: 2.0, 2.5: 2.0, 3.5: 4.0, 5.0: 4.0}


def _load_by_path(module_name: str, *rel: str):
    """Load a sibling test helper by path (no ``sys.path`` widening; the file may import ``cudnn``)."""
    path = os.path.join(_HERE, *rel)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def torchao_port():
    """``test/python/test_low_precision_matmul.py``: the torchao ``_f32_to_floatx_unpacked`` /
    ``_floatx_unpacked_to_f32`` pair (``test_utils`` resolves through ``test/python``'s conftest)."""
    return _load_by_path("_fp4_ref_low_precision_matmul", "..", "..", "test_low_precision_matmul.py")


@pytest.fixture(scope="module")
def gemm_utils():
    """``test/python/gemm/frost/gemm_test_utils.py``: ``E2M1``, ``unpack_fp4``, ``to_blocked``."""
    return _load_by_path("_fp4_ref_gemm_test_utils", "..", "..", "gemm", "frost", "gemm_test_utils.py")


# ---------------------------------------------------------------------------
# E2M1 rounding
# ---------------------------------------------------------------------------


def test_e2m1_constants_are_the_kernels():
    assert E2M1_GRID == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    inv = torch.tensor(FP4_INV_MAX, dtype=torch.float32)
    assert inv.view(torch.int32).item() & 0xFFFFFFFF == 0x3E2AAAAB, "fp32(1/6) -- the kernel's opaque_fp4_max_rcp"
    assert (torch.tensor(6.0) * inv).item() == 1.0  # amax == 6 * 2^p scales EXACTLY to 2^p (no E8M0 round-up)
    assert E4M3_MIN_SUBNORMAL == 2.0**-9
    assert torch.tensor(E4M3_MIN_SUBNORMAL).to(torch.float8_e4m3fn).view(torch.uint8).item() == 0x01
    assert torch.tensor(E4M3_MIN_SUBNORMAL / 2).to(torch.float8_e4m3fn).float().item() == 0.0, "why the floor exists"
    assert FP4_FORMATS == {"nvfp4": (16, torch.float8_e4m3fn), "mxfp4": (32, torch.float8_e8m0fnu)}


def test_e2m1_midpoint_table_ties_to_the_even_code():
    x = torch.tensor(_MIDPOINTS, dtype=torch.float32)
    want = torch.tensor([_MIDPOINT_TABLE[m] for m in _MIDPOINTS], dtype=torch.float32)
    assert torch.equal(e2m1_rne(x), want)
    assert torch.equal(e2m1_rne(-x), -want)
    # saturation, sign of zero, exact grid points unchanged
    assert torch.equal(e2m1_rne(torch.tensor([6.0, 7.0, 1e9, float("inf")])), torch.full((4,), 6.0))
    assert torch.equal(e2m1_rne(torch.tensor([-6.5, -1e9])), torch.full((2,), -6.0))
    grid = torch.tensor(E2M1_GRID)
    assert torch.equal(e2m1_rne(grid), grid) and torch.equal(e2m1_rne(-grid), -grid)


def test_reference_refuses_nan_and_non_finite_blocks_instead_of_fabricating_codes():
    """``satfinite`` saturates ``+-inf`` to ``+-6`` (pinned above) but its NaN byte is unspecified, and a non-finite
    element makes its block's amax non-finite (E8M0 ``0xFF`` / E4M3 ``0x7F`` scales, every code 6.0).  The oracle
    REFUSES rather than encodes: a bitwise kernel tier must report "bad input", never a fabricated code mismatch."""
    with pytest.raises(ValueError, match="NaN input"):
        e2m1_codes(torch.tensor([1.0, float("nan")]))
    for bad in (float("nan"), float("inf"), -float("inf")):
        for fmt in ("nvfp4", "mxfp4"):
            x = torch.randn(2, 64)
            x[1, 5] = bad
            with pytest.raises(ValueError, match="non-finite input"):
                fp4_quantize_rowwise_2d(x, fmt)
    # finite input of any magnitude still quantizes (saturation is the contract, not an error)
    for fmt in ("nvfp4", "mxfp4"):
        packed, sf = fp4_quantize_rowwise_2d(torch.tensor([[1e30, -1e30] + [0.0] * 30]), fmt)
        assert packed.shape == (1, 16) and bool(torch.isfinite(unpack_e2m1(packed)).all())
    neg_zero = e2m1_rne(torch.tensor([-0.2, -0.0]))
    assert torch.equal(neg_zero, torch.zeros(2)) and bool(torch.signbit(neg_zero).all()), "sign kept through -0.0"
    assert e2m1_codes(torch.tensor([-0.2])).item() == 0x8


def test_e2m1_codes_match_the_torchao_port(torchao_port):
    """Every midpoint, every grid point, both signs, plus a dense random sweep -- code for code
    against ``_f32_to_floatx_unpacked(x, 2, 1)``, and value for value against its inverse."""
    torch.manual_seed(0)
    pieces = [torch.tensor(_MIDPOINTS), torch.tensor(E2M1_GRID), torch.rand(4096) * 8.0, torch.randn(4096) * 3.0, torch.tensor([0.0, 1e-30, 6.5, 100.0])]
    x = torch.cat([p.float() for p in pieces])
    x = torch.cat([x, -x])
    ours = e2m1_codes(x)
    theirs = torchao_port._f32_to_floatx_unpacked(x.contiguous(), 2, 1)
    assert ours.dtype == theirs.dtype == torch.uint8
    assert torch.equal(ours, theirs)
    assert torch.equal(e2m1_rne(x), torchao_port._floatx_unpacked_to_f32(theirs, 2, 1))
    assert torch.equal(e2m1_values(ours), e2m1_rne(x))


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------


def test_pack_unpack_identity_and_low_nibble_is_the_even_k(gemm_utils):
    torch.manual_seed(1)
    codes = torch.randint(0, 16, (7, 64), dtype=torch.uint8)
    packed = pack_e2m1(codes)
    assert packed.shape == (7, 32) and packed.dtype == torch.uint8
    assert torch.equal(pack_e2m1(torch.tensor([[1, 2]], dtype=torch.uint8)), torch.tensor([[0x21]], dtype=torch.uint8)), "low nibble = even k"
    assert torch.equal(unpack_e2m1(packed), e2m1_values(codes))
    # the FROST GEMM suite's reader agrees byte for byte
    lut = torch.tensor(gemm_utils.E2M1, dtype=torch.float32)
    assert list(gemm_utils.E2M1) == list(E2M1_GRID) + [-v for v in E2M1_GRID]
    assert torch.equal(gemm_utils.unpack_fp4(packed, lut), unpack_e2m1(packed))
    # a float4_e2m1fn_x2 VIEW of the bytes unpacks the same (what the kernel-side tensor is)
    assert torch.equal(unpack_e2m1(packed.view(torch.float4_e2m1fn_x2)), unpack_e2m1(packed))
    with pytest.raises(ValueError, match="even"):
        pack_e2m1(torch.zeros(3, 5, dtype=torch.uint8))


# ---------------------------------------------------------------------------
# F8_128x4 blob at block 16 and 32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", [16, 32])
@pytest.mark.parametrize("rows,k", [(1, 256), (200, 256), (256, 8192), (1000, 512), (392, 544)])
def test_blob_round_trips_at_both_blocks_and_matches_to_blocked(gemm_utils, block, rows, k):
    torch.manual_seed(rows * 31 + k + block)
    e = torch.randint(0, 255, (rows, k // block), dtype=torch.uint8)
    blob = mx_swizzle_sf_rowwise_padded(e, block=block)
    rows_pad, cols_pad = mx_sf_padded_dims(rows, k, block)
    assert rows_pad == -(-rows // MX_ATOM_ROWS) * MX_ATOM_ROWS and cols_pad == -(-(k // block) // MX_ATOM_COLS) * MX_ATOM_COLS
    assert blob.numel() == rows_pad * cols_pad and blob.dtype == torch.uint8 and blob.is_contiguous()
    assert torch.equal(mx_unswizzle_sf_rowwise(blob, rows, k, block), e)
    assert torch.equal(blob, gemm_utils.to_blocked(e)), "one blob builder == the GEMM suite's"
    # pad rows / blocks are 0x00
    assert int((blob != 0).sum()) <= e.numel()


def test_blob_default_block_is_32_and_unchanged():
    """The MXFP8 suites are bitwise pins on the default path -- ``block=`` must not have moved it."""
    torch.manual_seed(2)
    e = torch.randint(0, 255, (392, 16), dtype=torch.uint8)
    assert mx_sf_padded_dims(392, 512) == mx_sf_padded_dims(392, 512, 32) == (512, 16)
    assert torch.equal(mx_swizzle_sf_rowwise_padded(e), mx_swizzle_sf_rowwise_padded(e, block=32))
    assert torch.equal(mx_unswizzle_sf_rowwise(mx_swizzle_sf_rowwise_padded(e), 392, 512), e)
    # the same logical bytes over the same K mean a DIFFERENT matrix at block 16 (twice the columns)
    assert mx_sf_padded_dims(392, 512, 16) == (512, 32)
    with pytest.raises(ValueError, match="multiple"):
        mx_sf_padded_dims(8, 40, 16)
    with pytest.raises(ValueError, match="blob has"):
        mx_unswizzle_sf_rowwise(torch.zeros(512 * 16, dtype=torch.uint8), 392, 512, 16)


def test_blob_builder_views_fp8_scales_as_bytes_not_values():
    s = torch.full((4, 8), 0.013671875).to(torch.float8_e4m3fn)  # 6/448: value-casting to uint8 would give 0
    blob = mx_swizzle_sf_rowwise_padded(s, block=16)
    assert torch.equal(mx_unswizzle_sf_rowwise(blob, 4, 128, 16), s.view(torch.uint8))
    e = torch.full((4, 4), 130, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    assert torch.equal(mx_unswizzle_sf_rowwise(mx_swizzle_sf_rowwise_padded(e), 4, 128), e.view(torch.uint8))


# ---------------------------------------------------------------------------
# quantize / dequantize, both formats
# ---------------------------------------------------------------------------


def test_fp4_format_spellings():
    assert fp4_format("nvfp4") == ("nvfp4", 16, torch.float8_e4m3fn)
    assert fp4_format("MXFP4") == ("mxfp4", 32, torch.float8_e8m0fnu)

    class _Enum:  # an enum member NAMED like the block's Fp4Format
        name = "NVFP4"

    assert fp4_format(_Enum()) == ("nvfp4", 16, torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="unknown fp4 format"):
        fp4_format("fp4")
    with pytest.raises(ValueError, match="multiple of the 32-element mxfp4 block"):
        fp4_quantize_rowwise_2d(torch.zeros(2, 48), "mxfp4")
    with pytest.raises(ValueError, match="multiple of the 16-element nvfp4 block"):
        fp4_quantize_rowwise_2d(torch.zeros(2, 40), "nvfp4")


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_all_zero_block_quantizes_to_zero_codes_and_the_format_floor_scale(fmt):
    """amax 0: MXFP4 -> E8M0 ``0x00`` (2^-127); NVFP4 -> the E4M3 min-subnormal ``0x01`` (2^-9), never a
    zero scale (an infinite encode scale).  Codes 0, dequant EXACTLY 0 -- the block's dead-entry contract."""
    _, block, _ = fp4_format(fmt)
    k = 8 * block
    x = torch.zeros(5, k)
    x[2, block : 2 * block] = torch.tensor(E2M1_GRID)[torch.arange(block) % 8] * 0.75  # one live block in a dead row
    packed, sf = fp4_quantize_rowwise_2d(x, fmt)
    assert packed.shape == (5, k // 2) and sf.shape == (5, 8) and sf.dtype == torch.uint8
    floor = 0x00 if fmt == "mxfp4" else 0x01
    dead = torch.ones(5, 8, dtype=torch.bool)
    dead[2, 1] = False
    assert bool((sf[dead] == floor).all()) and int(sf[2, 1]) != floor
    assert bool((unpack_e2m1(packed)[x == 0] == 0).all())
    blob = mx_swizzle_sf_rowwise_padded(sf, block=block)
    deq = fp4_dequant_rowwise_2d(packed, blob, fmt)
    assert torch.equal(deq[x == 0], torch.zeros_like(deq[x == 0]))
    assert bool(torch.isfinite(deq).all())


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_grid_values_at_a_power_of_two_scale_round_trip_exactly(fmt):
    """A block holding the E2M1 grid x 2^p has amax 6 x 2^p, whose fp32(1/6) multiple is EXACTLY 2^p, so both
    scale rules land on 2^p and every code dequantizes to its input bit for bit."""
    _, block, _ = fp4_format(fmt)
    torch.manual_seed(3)
    rows, nblk = 6, 5
    p = torch.randint(-6, 6, (rows, nblk)).float()  # NVFP4's E4M3 covers 2^-6 .. 2^8 as normals
    grid = torch.tensor(E2M1_GRID)
    signs = torch.where(torch.rand(rows, nblk, block) < 0.5, -1.0, 1.0)
    x = grid[torch.randint(0, 8, (rows, nblk, block))] * signs * torch.pow(2.0, p).unsqueeze(-1)
    x[..., 0] = 6.0 * torch.pow(2.0, p)  # every block reaches the top of the grid
    x = x.reshape(rows, nblk * block)
    packed, sf = fp4_quantize_rowwise_2d(x, fmt)
    if fmt == "mxfp4":
        assert torch.equal(sf.int(), (127 + p).int())
    else:
        assert torch.equal(sf.view(torch.float8_e4m3fn).float(), torch.pow(2.0, p))
    deq = fp4_dequant_rowwise_2d(packed, mx_swizzle_sf_rowwise_padded(sf, block=block), fmt)
    assert torch.equal(deq.abs(), x.abs()) and torch.equal(torch.signbit(deq[x != 0]), torch.signbit(x[x != 0]))


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_random_blocks_dequantize_within_half_an_e2m1_ulp(fmt):
    """Random data whose block scales are NORMAL e4m3 / in-range E8M0: every code is within half a grid step
    (at its magnitude) x scale of the input, and the dequant goes THROUGH the blob.  (The subnormal-scale
    regime, where NVFP4 saturates, is pinned separately below.)"""
    _, block, sf_dtype = fp4_format(fmt)
    torch.manual_seed(4)
    rows, k = 300, 8 * block
    x = torch.randn(rows, k) * torch.pow(2.0, torch.empty(rows, 1).uniform_(-1.0, 7.0))
    inv = torch.tensor(FP4_INV_MAX, dtype=torch.float32)
    assert bool((x.reshape(rows, -1, block).abs().amax(-1) * inv >= 2.0**-6).all()), "precondition: normal e4m3 scales"
    packed, sf = fp4_quantize_rowwise_2d(x, fmt)
    blob = mx_swizzle_sf_rowwise_padded(sf, block=block)
    deq = fp4_dequant_rowwise_2d(packed, blob, fmt)
    sf_full = sf.view(sf_dtype).float().repeat_interleave(block, dim=-1)
    q = x / sf_full
    # MXFP4 rounds the exponent UP, so |q| <= 6 up to the fp32(1/6) product's ulp; NVFP4's e4m3 RN can round the
    # scale DOWN by up to 2^-4 relative, so |q| may exceed 6 by that much and saturate (still within half a step)
    assert q.abs().max() <= 6.0 * (1 + (2.0**-20 if fmt == "mxfp4" else 2.0**-4))
    # half-ulp bound on the scaled value: step is 0.5 below 2, 1 in [2, 4), 2 in [4, 6]
    qa = q.abs()
    step = torch.where(qa < 2.0, 0.5, torch.where(qa < 4.0, 1.0, 2.0))
    err = (deq - x).abs() / sf_full
    assert bool((err <= step / 2 + 1e-6).all())
    cos = torch.nn.functional.cosine_similarity(deq.flatten(), x.flatten(), dim=0).item()
    assert cos > 0.98, cos
    # the oracle reads the blob: a corrupted scale byte moves the dequant of exactly that block
    bad = blob.clone()
    bad[0] ^= 0x7F
    diff = (fp4_dequant_rowwise_2d(packed, bad, fmt) != deq).any(dim=-1)
    assert diff.sum() >= 1


def test_nvfp4_subnormal_scale_regime_saturates_without_a_global_scale():
    """Below ``amax/6 < 2^-6`` the e4m3 scale is SUBNORMAL (step 2^-9), so RN can round it DOWN by up to
    2^-10 absolute: a block with ``amax = 6 x 1.4 x 2^-9`` gets scale ``2^-9`` (``0x01``), its top value scales
    to 8.4 and SATURATES to 6 -- resolution the format loses without a per-tensor scale (the plan's risk 6,
    ``MxQuantSpec.o_global_scale`` is the append-only fix).  Pinned so the loss is a known quantity."""
    x = torch.zeros(1, 16)
    x[0, 0] = 6.0 * 1.4 * 2.0**-9
    x[0, 1] = 0.5 * 2.0**-9
    packed, sf = fp4_quantize_rowwise_2d(x, "nvfp4")
    assert sf.view(torch.uint8).item() == 0x01
    deq = fp4_dequant_rowwise_2d(packed, mx_swizzle_sf_rowwise_padded(sf, block=16), "nvfp4")
    assert deq[0, 0].item() == 6.0 * 2.0**-9 < x[0, 0].item()  # saturated
    assert deq[0, 1].item() == 0.5 * 2.0**-9  # the small value is exact at this scale
    # and just BELOW the floor the scale stays 2^-9 rather than collapsing to 0 (an infinite encode scale)
    y = torch.full((1, 16), 2.0**-12)
    packed, sf = fp4_quantize_rowwise_2d(y, "nvfp4")
    assert sf.view(torch.uint8).item() == 0x01 and bool(torch.isfinite(unpack_e2m1(packed)).all())


def test_nvfp4_uses_a_true_division_so_exact_midpoints_round_to_even():
    """``0.5859375 / 0.46875 == 1.25`` exactly -> code 1.0 under RNE; a reciprocal-multiply gives
    1.2500001 -> 1.5.  amax 2.8125 makes the e4m3 scale exactly 0.46875 (the kernel's ``div.rn`` case)."""
    x = torch.zeros(1, 16)
    x[0, 0] = 2.8125
    x[0, 1] = 0.5859375
    x[0, 2] = -0.5859375
    packed, sf = fp4_quantize_rowwise_2d(x, "nvfp4")
    assert sf.view(torch.float8_e4m3fn).float().item() == 0.46875
    vals = unpack_e2m1(packed)
    assert vals[0, 1].item() == 1.0 and vals[0, 2].item() == -1.0
    # and the reciprocal-multiply really would have flipped it
    rcp = torch.tensor(1.0) / torch.tensor(0.46875)
    assert e2m1_rne(torch.tensor(0.5859375) * rcp).item() == 1.5


def test_mxfp4_scale_is_the_shared_e8m0_ceil_rule():
    """The MXFP4 exponent is ``e8m0_ceil(amax * fp32(1/6))`` -- the same TE / cuDNN rounding-UP rule the
    MXFP8 oracle uses (with 1/448); an amax just above 6 x 2^p lands on 2^(p+1)."""
    x = torch.zeros(1, 64)
    x[0, 0] = 6.0
    x[0, 32] = 6.0 * (1 + 2.0**-20)
    _, sf = fp4_quantize_rowwise_2d(x, "mxfp4")
    assert sf.tolist() == [[127, 128]]
    x[0, 32] = 2.0**-130  # below 2^-127 x 6: the exponent stays at the E8M0 floor
    _, sf = fp4_quantize_rowwise_2d(x, "mxfp4")
    assert sf[0, 1].item() == 0
