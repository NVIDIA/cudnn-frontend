# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cudnn
import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

_dsl_error = cutedsl_requirement_error("NVFP4 block-scale conversion")
if _dsl_error is not None:
    pytest.skip(_dsl_error, allow_module_level=True)


def _active_sm() -> int | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


_SM = _active_sm()
requires_sm100_or_newer = pytest.mark.skipif(
    _SM is None or _SM < 100,
    reason="needs SM100 or newer, have " + ("none" if _SM is None else f"sm_{_SM}"),
)


def _representable_input(m, k):
    levels = torch.tensor(
        (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device="cuda",
        dtype=torch.float32,
    )
    groups = k // 16
    scale_lut = torch.tensor((0.125, 0.25, 0.5, 1.0), device="cuda", dtype=torch.float32)
    rows = torch.arange(m, device="cuda").unsqueeze(1)
    columns = torch.arange(groups, device="cuda").unsqueeze(0)
    logical_scales = scale_lut[(rows + columns) & 3]
    return (logical_scales.unsqueeze(-1) * levels).reshape(1, m, k).to(torch.bfloat16)


@requires_sm100_or_newer
@pytest.mark.L0
def test_nvfp4_conversion_is_exposed_through_semantic_ops_namespace():
    assert cudnn.ops.nvfp4_block_scale_quantize_forward.__name__ == "nvfp4_block_scale_quantize_forward"
    assert cudnn.ops.nvfp4_block_scale_dequantize_forward.__name__ == "nvfp4_block_scale_dequantize_forward"
    assert not hasattr(cudnn.ops, "Nvfp4BlockScaleQuantizer")
    assert not hasattr(cudnn.ops, "Nvfp4BlockScaleDequantizer")


@requires_sm100_or_newer
@pytest.mark.L0
def test_nvfp4_quantize_rejects_non_contiguous_input():
    bad_x = torch.empty((1, 2048, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    encode = torch.ones((1, 1, 1), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="contiguous"):
        cudnn.ops.nvfp4_block_scale_quantize_forward(bad_x, encode)


@requires_sm100_or_newer
@pytest.mark.parametrize(
    "m,k", [pytest.param(128, 2048, marks=pytest.mark.L0), pytest.param(256, 5376, marks=pytest.mark.L1), pytest.param(128, 8192, marks=pytest.mark.L1)]
)
def test_nvfp4_quantize_dequantize_round_trip_on_representable_values(m, k):
    x = _representable_input(m, k)
    encode = torch.full((1, 1, 1), 2.0, device="cuda", dtype=torch.float32)
    decode = torch.full((1, 1, 1), 0.5, device="cuda", dtype=torch.float32)

    result = cudnn.ops.nvfp4_block_scale_quantize_forward(x, encode)
    packed, scales = result
    restored = cudnn.ops.nvfp4_block_scale_dequantize_forward(packed, scales, decode)
    torch.cuda.synchronize()

    assert tuple(packed.shape) == (1, m, k // 2)
    assert tuple(scales.shape) == (1, m, k // 16)
    assert result["packed_tensor"] is packed and result["scale_tensor"] is scales
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


@requires_sm100_or_newer
@pytest.mark.L1
def test_nvfp4_dequantize_accepts_uint8_carrier():
    x = _representable_input(128, 2048)
    one = torch.ones((1, 1, 1), device="cuda", dtype=torch.float32)
    packed, scales = cudnn.ops.nvfp4_block_scale_quantize_forward(x, one)
    restored = cudnn.ops.nvfp4_block_scale_dequantize_forward(packed.view(torch.uint8), scales, one)
    torch.cuda.synchronize()

    torch.testing.assert_close(restored, x, rtol=0, atol=0)


@requires_sm100_or_newer
@pytest.mark.L1
def test_nvfp4_reuses_compiled_k_across_runtime_m():
    one = torch.ones((1, 1, 1), device="cuda", dtype=torch.float32)
    for m in (128, 256):
        x = _representable_input(m, 2048)
        packed, scales = cudnn.ops.nvfp4_block_scale_quantize_forward(x, one)
        restored = cudnn.ops.nvfp4_block_scale_dequantize_forward(packed, scales, one)
        torch.cuda.synchronize()
        torch.testing.assert_close(restored, x, rtol=0, atol=0)
