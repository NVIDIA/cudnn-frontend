# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the weight formats of the model samples (``samples/models/model_samples/weights.py``): the bf16 copies a
multiply reads, against a decode of the stored codes written here from the format definitions -- 8-bit E4M3 with a scale;
packed 4-bit E2M1, low nibble first, with E4M3 block scales and a global scale -- with an all-zero and a saturated block,
both merge rules, and the reference multiply on a product computed by hand.  The provider stack and the reference stack of a
sample read the same copies and use the same quantisers, so the tie-out between them cannot see an error in either: these
tests are the pin.  No GPU, no weights."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.L0]

MODELS = Path(__file__).resolve().parents[3] / "samples" / "models"
#: the eight magnitudes of an E2M1 code by its three low bits; bit 3 is the sign
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.fixture(scope="module", autouse=True)
def helper_layer():
    """``samples/models`` on ``sys.path`` for this module only: the path and every ``model_samples`` module leave with it."""
    sys.path.insert(0, str(MODELS))
    yield
    sys.path.remove(str(MODELS))
    for name in [n for n in sys.modules if n == "model_samples" or n.startswith("model_samples.")]:
        del sys.modules[name]


def _e4m3_value(byte: int) -> float:
    """The value of one E4M3 code byte from the format: a sign bit, four exponent bits with bias 7, three mantissa bits,
    subnormals at exponent 0, no infinity."""
    sign, exponent, mantissa = byte >> 7, (byte >> 3) & 0xF, byte & 0x7
    value = (mantissa / 8.0) * 2.0**-6 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return -value if sign else value


def _e4m3(codes):
    return torch.tensor([_e4m3_value(byte) for byte in range(256)], dtype=torch.float64)[codes.long()]


def _e4m3_step(values):
    """The spacing of E4M3 at these magnitudes."""
    return torch.pow(2.0, torch.floor(torch.log2(values.abs().clamp_min(2.0**-6))) - 3)


def _e2m1(packed):
    """Packed 4-bit codes -> values: element ``2k`` is the low nibble of byte ``k``, element ``2k + 1`` the high one."""
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    return torch.tensor([*E2M1, *(-v for v in E2M1)], dtype=torch.float64)[nibbles.long()]


def test_an_8_bit_copy_is_the_decode_of_its_codes():
    from model_samples import weights

    x = torch.randn((64, 80), generator=torch.Generator().manual_seed(1)) * 0.02
    x[0, 0], x[0, 1] = 1.0, -1.0  # beyond 448 times the scale: the quantiser saturates
    scale = 0.1 / 448.0
    stored = weights.quantize_fp8(x, scale)
    assert stored.codes.dtype == torch.uint8 and stored.codes[0, :2].tolist() == [0x7E, 0xFE], "saturated at +-448, never the NaN code"
    copy = weights.dequantize(stored)
    mine = _e4m3(stored.codes)
    assert copy.data.dtype == torch.bfloat16 and copy.scale == scale
    assert torch.equal(copy.data.double(), mine), "the copy holds the code values exactly: four significant bits in bf16"
    exact = x.double() / scale
    inside = (mine - exact).abs() <= 0.5 * _e4m3_step(mine) + 1e-6 * exact.abs()  # round to nearest, in code units
    assert bool(inside.flatten()[2:].all()) and mine[0, :2].tolist() == [448.0, -448.0]


def test_a_4_bit_copy_is_the_decode_of_its_nibbles_low_nibble_first():
    from model_samples import weights

    # one block of 16 whose values are the code values themselves: the block scale is 1 and nothing is rounded
    block = torch.tensor([[0.5, 6.0, -1.0, 1.5, 2.0, -3.0, 4.0, 0.0, -0.5, 1.0, -6.0, 3.0, -1.5, -2.0, -4.0, 0.5]])
    stored = weights.quantize_nvfp4(block, 1.0)

    def nibble(value):
        return E2M1.index(abs(value)) | (8 if value < 0 else 0)

    pairs = zip(block[0, 0::2].tolist(), block[0, 1::2].tolist())
    assert stored.codes.tolist() == [[nibble(low) | (nibble(high) << 4) for low, high in pairs]], "low nibble first"
    assert stored.block_scales.tolist() == [[0x38]], "the E4M3 code of 1"
    assert stored.shape == (1, 16) and weights.dequantize(stored).data.tolist() == block.tolist()

    x = torch.randn((64, 160), generator=torch.Generator().manual_seed(2)) * 0.02
    x[3, 16:32] = 0.0  # an all-zero block
    x[5, 0] = 50.0  # a block whose block scale saturates
    scale = 6.5 * 0.02 / (448.0 * 6.0)
    stored = weights.quantize_nvfp4(x, scale)
    copy = weights.dequantize(stored)
    block_scales = _e4m3(stored.block_scales).repeat_interleave(16, dim=-1)
    mine = _e2m1(stored.codes) * block_scales
    assert copy.data.dtype == torch.bfloat16 and copy.scale == scale
    assert torch.equal(copy.data.double(), mine), "code value times block scale is exact in bf16: at most six significant bits"
    assert int(stored.block_scales[3, 1]) == 0 and not bool(stored.codes[3, 8:16].any()) and not bool(copy.data[3, 16:32].any())
    assert int(stored.block_scales[5, 0]) == 0x7E and float(mine[5, 0]) == 6.0 * 448.0, "saturated: the largest code at the largest block scale"
    error = (mine * scale - x.double()).abs()
    error[5, :16] = 0.0
    assert bool((error <= block_scales * scale * (1.0 + 1e-5)).all()), "within one code step of its block where the block scale is not saturated"


def test_the_merge_rules():
    from model_samples import weights

    generator = torch.Generator().manual_seed(3)
    small = weights.quantize_fp8(torch.randn((8, 32), generator=generator) * 0.01, 0.01 / 100)
    large = weights.quantize_fp8(torch.randn((4, 32), generator=generator) * 0.03, 0.03 / 100)
    merged = weights.merge_scaled([small, large])  # the default rule: one scale, the largest
    assert merged.scale == large.scale and merged.codes.shape == (12, 32)
    assert torch.equal(merged.codes[8:], large.codes), "the part that has the largest scale keeps its codes"
    before, after = _e4m3(small.codes) * small.scale, _e4m3(merged.codes[:8])
    exact = before / merged.scale
    assert bool(((after - exact).abs() <= 0.5 * _e4m3_step(after) + 1e-6 * exact.abs()).all()), "the other part is re-quantised to nearest at that scale"
    assert not torch.equal(merged.codes[:8], small.codes)
    own = weights.merge_scaled([small, large], "own_scale")
    assert torch.equal(own.codes, torch.cat([small.codes, large.codes])), "every part keeps its codes"
    assert torch.allclose(own.scale, torch.tensor([small.scale] * 8 + [large.scale] * 4)) and own.scale.dtype == torch.float32
    with pytest.raises(ValueError, match="merge rule"):
        weights.merge_scaled([small, large], "smallest_scale")
    with pytest.raises(ValueError, match="one scale each"):
        weights.merge_scaled([own, large])
    gate = weights.quantize_nvfp4(torch.randn((2, 16), generator=generator), 0.1)
    up = weights.quantize_nvfp4(torch.randn((3, 16), generator=generator), 0.1)
    both = weights.merge_block_scaled([gate, up])
    assert both.scale == 0.1 and torch.equal(both.codes, torch.cat([gate.codes, up.codes]))
    assert torch.equal(both.block_scales, torch.cat([gate.block_scales, up.block_scales]))
    with pytest.raises(ValueError, match="share their global scale"):
        weights.merge_block_scaled([gate, weights.quantize_nvfp4(torch.randn((3, 16), generator=generator), 0.2)])


def test_the_reference_multiply_on_a_product_computed_by_hand():
    from model_samples import weights
    from model_samples.providers import torch_ref

    operand = weights.quantize_fp8(torch.tensor([[1.0, -2.0, 0.5, 4.0]]), 0.5)  # the codes 2, -4, 1, 8
    assert _e4m3(operand.codes).tolist() == [[2.0, -4.0, 1.0, 8.0]]
    rows = torch.tensor([[2.0, 1.0, 0.0, -1.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.bfloat16)
    # (2 * 2 - 4 * 1 + 0 - 8) * 0.5 * 0.25 = -1 and (2 - 4 + 1 + 8) * 0.5 * 0.25 = 0.875
    for acc in (torch.float64, torch.float32):
        product = torch_ref.linear(operand, weights.Dequantized(rows, 0.25), acc)
        assert product.dtype == torch.bfloat16 and product.tolist() == [[-1.0, 0.875]]
    per_row = weights.Dequantized(rows, torch.tensor([0.25, 0.5]))  # a merged weight that kept one scale per row
    assert torch_ref.linear(operand, per_row, torch.float64).tolist() == [[-1.0, 1.75]]
    packed = weights.quantize_nvfp4(torch.tensor([[0.5, 6.0, -1.0, 1.5, 2.0, -3.0, 4.0, 0.0, -0.5, 1.0, -6.0, 3.0, -1.5, -2.0, -4.0, 0.5]]), 2.0)
    ones = weights.Dequantized(torch.ones((1, 16), dtype=torch.bfloat16), 1.0)
    assert torch_ref.linear(packed, ones, torch.float64).tolist() == [[0.5]], "a 4-bit operand: its code values, its block scale and its global scale"
