# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for the torch-only MXFP8 quantizer + F8_128x4 swizzle
(test/python/sdpa/mxfp8_quant.py) that replaced the TransformerEngine
dependency of the MXFP8 SDPA tests.

Covers: E8M0 ceil rounding (incl. amax=0 / power-of-two / inf / nan cases),
round-trip dequantization error bounds, TE storage shapes, the 128x4 swizzle
against hand-computed layouts, and — when TransformerEngine >= 2.12 happens to
be installed — bit-exact parity with TE's MXFP8Quantizer and
tex.swizzle_scales_for_gemm_.
"""

import math

import pytest
import torch

# MXFP8 (block-scale FP8 + F8_128x4 scale reordering) is a Blackwell-and-newer
# feature: nothing on Hopper/Ampere can consume the quantized inputs, so the
# whole module skips there (the quantizer itself is pure torch and
# arch-agnostic — only its consumers are arch-bound).
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="MXFP8 requires Blackwell (SM100+)",
)

from sdpa.mxfp8_quant import (
    FP8_MAX_NORM,
    e8m0_ceil,
    e8m0_to_float,
    quantize_blocks,
    quantize_mxfp8_2d,
    quantize_to_mxfp8,
    swizzle_sf_columnwise,
    swizzle_sf_rowwise,
)

# fmt: off

def _has_te():
    try:
        from looseversion import LooseVersion
        import transformer_engine

        return LooseVersion(transformer_engine.__version__) >= LooseVersion("2.12.0")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# E8M0 scale computation
# ---------------------------------------------------------------------------

@pytest.mark.L0
def test_e8m0_ceil_special_cases():
    v = torch.tensor([0.0, 1.0, 1.5, 2.0, 0.75, 2.0**-127, 2.0**127, float("inf"), float("nan")], dtype=torch.float32)
    e = e8m0_ceil(v)
    assert e[0].item() == 0x00          # amax == 0 -> byte 0 (scale 2^-127)
    assert e[1].item() == 127           # exact power of two is NOT rounded up
    assert e[2].item() == 128           # 1.5 -> ceil -> 2.0
    assert e[3].item() == 128
    assert e[4].item() == 127           # 0.75 -> ceil -> 1.0
    assert e[5].item() == 0x00          # 2^-127 stays at the bottom (TE subnormal rule)
    assert e[6].item() == 254
    assert e[7].item() == 0xFE          # inf -> satfinite
    assert e[8].item() == 0xFF          # nan


@pytest.mark.L0
def test_e8m0_ceil_matches_math_log2():
    torch.manual_seed(0)
    v = torch.rand(4096, dtype=torch.float32) * 1000 + 1e-6
    e = e8m0_ceil(v)
    for x, eb in zip(v.tolist(), e.tolist()):
        assert eb == min(254, math.ceil(math.log2(x)) + 127), f"x={x}"


# ---------------------------------------------------------------------------
# Block quantization round-trip
# ---------------------------------------------------------------------------

@pytest.mark.L0
@pytest.mark.parametrize("fp8", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_roundtrip_error_bound(fp8):
    torch.manual_seed(1)
    x = torch.randn(64, 8, 32, dtype=torch.float32) * torch.logspace(-6, 6, 64).reshape(64, 1, 1)
    data, e = quantize_blocks(x, fp8)
    scale = e8m0_to_float(e).unsqueeze(-1)
    deq = data.float() * scale
    # ceil scale guarantees |x|/scale <= max_norm; the RN cast error is at most
    # half an fp8 ULP at magnitude max_norm: max_norm * 2^-(1+mantissa_bits).
    mant_bits = 3 if fp8 == torch.float8_e4m3fn else 2
    bound = scale * FP8_MAX_NORM[fp8] * 2.0 ** -(1 + mant_bits)
    assert ((x - deq).abs() <= bound + 1e-30).all()
    # no saturation: |quantized| never exceeds max_norm
    assert data.float().abs().max().item() <= FP8_MAX_NORM[fp8]


@pytest.mark.L0
def test_zero_block():
    x = torch.zeros(4, 32)
    data, e = quantize_blocks(x, torch.float8_e4m3fn)
    assert (e == 0).all() and (data.float() == 0).all()


@pytest.mark.L0
def test_shapes_2d():
    x = torch.randn(256, 128)
    rd, re, cd, ce = quantize_mxfp8_2d(x, torch.float8_e4m3fn)
    assert rd.shape == (256, 128) and re.shape == (256, 4)
    assert cd.shape == (256, 128) and ce.shape == (8, 128)
    # columnwise data is quantized along dim0: dequant must reproduce x closely
    deq_c = cd.float() * e8m0_to_float(ce).repeat_interleave(32, dim=0)
    assert (x - deq_c).abs().max().item() <= (x.abs().max().item() / 8)


# ---------------------------------------------------------------------------
# F8_128x4 swizzle vs hand-computed layout
# ---------------------------------------------------------------------------

def _expected_atom_byte(r, c):
    return (r % 32) * 16 + ((r // 32) % 4) * 4 + (c % 4)


@pytest.mark.L0
def test_swizzle_rowwise_single_atom():
    R, C = 128, 4
    sf = (torch.arange(R * C) % 251).to(torch.uint8).reshape(R, C)
    out = swizzle_sf_rowwise(sf).flatten()
    for r in range(R):
        for c in range(C):
            assert out[_expected_atom_byte(r, c)].item() == sf[r, c].item(), (r, c)


@pytest.mark.L0
def test_swizzle_rowwise_atom_grid():
    # 256 rows x 8 cols -> 2x2 atoms, row-major atom order
    R, C = 256, 8
    sf = torch.randint(0, 255, (R, C), dtype=torch.uint8)
    out = swizzle_sf_rowwise(sf).flatten()
    for r in range(R):
        for c in range(C):
            atom = (r // 128) * (C // 4) + (c // 4)
            off = atom * 512 + _expected_atom_byte(r % 128, c % 4)
            assert out[off].item() == sf[r, c].item(), (r, c)


@pytest.mark.L0
def test_swizzle_columnwise():
    # storage [M//32, K] with M=256, K=128 -> transposed matrix [K=128, MB=8],
    # atoms row-major over (K/128=1, MB/4=2); atom row = k, atom col = m-block.
    MB, K = 8, 128
    sf = torch.randint(0, 255, (MB, K), dtype=torch.uint8)
    out = swizzle_sf_columnwise(sf).flatten()
    for mb in range(MB):
        for k in range(K):
            atom = (k // 128) * (MB // 4) + (mb // 4)
            off = atom * 512 + _expected_atom_byte(k % 128, mb % 4)
            assert out[off].item() == sf[mb, k].item(), (mb, k)


@pytest.mark.L0
def test_quantize_to_mxfp8_end_to_end_shapes():
    b, h, s, d = 2, 3, 200, 96  # unaligned s/d exercise the padding path
    x = torch.randn(b, h, s, d)
    data_d, ref_d, swz_d, data_s, ref_s, swz_s = quantize_to_mxfp8(x, b, h, s, d)
    l = b * h
    s_pad, d_pad = 256, 128
    assert data_d.shape == (b, h, s, d) and data_s.shape == (b, h, s, d)
    assert ref_d.shape == (l, s, d) and ref_s.shape == (l, s, d)
    assert swz_d.numel() == l * s_pad * (d_pad // 32)
    assert swz_s.numel() == l * (s_pad // 32) * d_pad
    # round-trip through the per-element ref scales
    deq_d = data_d.float().reshape(l, s, d) * ref_d
    assert (x.reshape(l, s, d) - deq_d).abs().max().item() <= x.abs().max().item() / 8


# ---------------------------------------------------------------------------
# TransformerEngine parity (only when TE >= 2.12 is installed)
# ---------------------------------------------------------------------------

@pytest.mark.L0
@pytest.mark.parametrize("fp8", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.skipif(not _has_te(), reason="TransformerEngine >= 2.12 not installed")
def test_te_parity(fp8):
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    torch.manual_seed(2)
    M, K = 512, 256
    x = (torch.randn(M, K, device="cuda") * torch.logspace(-4, 4, M, device="cuda").unsqueeze(-1)).float()
    x[::7] = 0  # exercise amax=0 blocks

    te_dt = tex.DType.kFloat8E4M3 if fp8 == torch.float8_e4m3fn else tex.DType.kFloat8E5M2
    q = MXFP8Quantizer(fp8_dtype=te_dt, rowwise=True, columnwise=True)
    r = q(x)

    rd, re, cd, ce = quantize_mxfp8_2d(x, fp8)
    assert torch.equal(r._rowwise_data.view(torch.uint8), rd.view(torch.uint8)), "rowwise data mismatch vs TE"
    assert torch.equal(r._rowwise_scale_inv.view(torch.uint8), re), "rowwise scale mismatch vs TE"
    assert torch.equal(r._columnwise_data.view(torch.uint8), cd.view(torch.uint8)), "columnwise data mismatch vs TE"
    assert torch.equal(r._columnwise_scale_inv.view(torch.uint8), ce), "columnwise scale mismatch vs TE"

    tex.swizzle_scales_for_gemm_(r)
    assert torch.equal(r._rowwise_scale_inv.view(torch.uint8).flatten(), swizzle_sf_rowwise(re).flatten()), "rowwise swizzle mismatch vs TE"
    assert torch.equal(r._columnwise_scale_inv.view(torch.uint8).flatten(), swizzle_sf_columnwise(ce).flatten()), "columnwise swizzle mismatch vs TE"
