# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DeepSeek-V4 normalized H128 plus MXFP4 QDQ operation."""

import pytest
import torch

pytestmark = pytest.mark.L0


def _require_supported_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the current inline E2M1 conversion path requires Blackwell+")


def _pow2_ceil_positive(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    mantissa = bits & 0x7FFFFF
    ceil_exponent = exponent + (mantissa != 0).to(torch.int32)
    return ((ceil_exponent + 127) << 23).contiguous().view(torch.float32)


def _e2m1_rne(values: torch.Tensor) -> torch.Tensor:
    sign = torch.where(values < 0, -1.0, 1.0)
    magnitude = values.abs()
    quantized = torch.where(
        magnitude <= 0.25,
        0.0,
        torch.where(
            magnitude < 0.75,
            0.5,
            torch.where(
                magnitude <= 1.25,
                1.0,
                torch.where(
                    magnitude < 1.75,
                    1.5,
                    torch.where(
                        magnitude <= 2.5,
                        2.0,
                        torch.where(
                            magnitude < 3.5,
                            3.0,
                            torch.where(magnitude <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return quantized * sign


def _reference(input_tensor: torch.Tensor) -> torch.Tensor:
    rows = input_tensor.numel() // 128
    transformed = input_tensor.view(rows, 128).float()
    for half_width in (1, 2, 4, 8, 16, 32, 64):
        pairs = transformed.reshape(rows, -1, 2, half_width)
        low = pairs[:, :, 0, :]
        high = pairs[:, :, 1, :]
        transformed = torch.cat((low + high, low - high), dim=-1).reshape(rows, 128)
    transformed = (transformed * (128.0**-0.5)).to(torch.bfloat16).float()

    groups = transformed.reshape(rows, 4, 32)
    amax = groups.abs().amax(dim=-1).clamp_min(6.0 * (2.0**-126))
    scale = _pow2_ceil_positive(amax * (1.0 / 6.0))
    normalized = (groups / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    output = _e2m1_rne(normalized) * scale.unsqueeze(-1)
    return output.reshape(input_tensor.shape).to(torch.bfloat16)


@pytest.mark.parametrize("shape", [(1, 128), (31, 128), (33, 128), (2, 3, 4, 128)])
def test_fwht_mxfp4_qdq_matches_exact_recipe(shape):
    _require_supported_device()
    try:
        from cudnn.ops import fwht_mxfp4_qdq
    except ImportError as error:
        pytest.skip(f"CuTe DSL optional dependencies are unavailable: {error}")

    torch.manual_seed(20260829)
    input_tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    expected = _reference(input_tensor)
    actual = fwht_mxfp4_qdq(input_tensor)

    assert actual.shape == input_tensor.shape
    assert actual.dtype == input_tensor.dtype
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


def test_fwht_mxfp4_qdq_empty_does_not_launch():
    _require_supported_device()
    from cudnn.ops import fwht_mxfp4_qdq

    input_tensor = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    output = fwht_mxfp4_qdq(input_tensor)
    assert output.shape == input_tensor.shape
    assert output.numel() == 0


def test_fwht_mxfp4_qdq_rejects_implicit_layout_conversion():
    _require_supported_device()
    from cudnn.ops import fwht_mxfp4_qdq

    input_tensor = torch.empty((128, 7), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    assert input_tensor.shape == (7, 128)
    assert not input_tensor.is_contiguous()
    with pytest.raises(ValueError, match="must be contiguous"):
        fwht_mxfp4_qdq(input_tensor)


def test_fwht_mxfp4_qdq_rejects_misaligned_contiguous_view():
    _require_supported_device()
    from cudnn.ops import fwht_mxfp4_qdq

    storage = torch.empty((129,), device="cuda", dtype=torch.bfloat16)
    input_tensor = storage[1:].view(1, 128)
    assert input_tensor.is_contiguous()
    assert input_tensor.data_ptr() % 32 != 0
    with pytest.raises(ValueError, match="32-byte aligned"):
        fwht_mxfp4_qdq(input_tensor)


def test_fwht_mxfp4_qdq_is_explicitly_inference_only():
    _require_supported_device()
    from cudnn.ops import fwht_mxfp4_qdq

    input_tensor = torch.randn((1, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(NotImplementedError, match="inference-only"):
        fwht_mxfp4_qdq(input_tensor)
