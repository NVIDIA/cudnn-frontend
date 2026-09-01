# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cudnn
import pytest
import torch


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


def _make_buffers(m=128, k=2048):
    x = torch.empty((1, m, k), device="cuda", dtype=torch.bfloat16)
    encode = torch.ones((1, 1, 1), device="cuda", dtype=torch.float32)
    carrier = torch.empty((1, m, k // 2), device="cuda", dtype=torch.uint8)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    packed = carrier.view(fp4_dtype) if fp4_dtype is not None else carrier
    scales = torch.empty((1, m, k // 16), device="cuda", dtype=torch.float8_e4m3fn)
    output = torch.empty_like(x)
    return x, encode, packed, scales, output


@requires_sm100_or_newer
@pytest.mark.L0
def test_nvfp4_conversion_is_exposed_through_semantic_ops_namespace():
    assert cudnn.ops.nvfp4_block_scale_quantize.__name__ == "nvfp4_block_scale_quantize"
    assert cudnn.ops.nvfp4_block_scale_dequantize.__name__ == "nvfp4_block_scale_dequantize"


@requires_sm100_or_newer
@pytest.mark.L0
def test_nvfp4_plan_rejects_non_materializable_contracts():
    x, encode, packed, scales, _ = _make_buffers()
    bad_x = torch.empty((1, x.shape[-1], x.shape[-2]), device="cuda", dtype=x.dtype).transpose(1, 2)
    plan = cudnn.ops.Nvfp4BlockScaleQuantizer(bad_x, encode, packed, scales)
    with pytest.raises(ValueError, match="input must be C-contiguous"):
        plan.check_support()


@requires_sm100_or_newer
@pytest.mark.L1
@pytest.mark.parametrize("m,k", [(128, 2048), (256, 5376), (128, 8192)])
def test_nvfp4_quantize_dequantize_round_trip_on_representable_values(m, k):
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
    x = (logical_scales.unsqueeze(-1) * levels).reshape(1, m, k).to(torch.bfloat16)
    encode = torch.full((1, 1, 1), 2.0, device="cuda", dtype=torch.float32)
    decode = torch.full((1, 1, 1), 0.5, device="cuda", dtype=torch.float32)

    result = cudnn.ops.nvfp4_block_scale_quantize(x, encode)
    packed, scales = result
    restored = cudnn.ops.nvfp4_block_scale_dequantize(packed, scales, decode)
    torch.cuda.synchronize()

    assert tuple(packed.shape) == (1, m, k // 2)
    assert tuple(scales.shape) == (1, m, k // 16)
    assert result["packed_tensor"] is packed and result["scale_tensor"] is scales
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


@requires_sm100_or_newer
@pytest.mark.L1
def test_nvfp4_prepared_plans_accept_explicit_uint8_carrier():
    m, k = 128, 2048
    levels = torch.tensor(
        (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device="cuda",
        dtype=torch.float32,
    )
    x = (levels.repeat(m, k // 16) * 0.25).reshape(1, m, k).to(torch.bfloat16)
    one = torch.ones((1, 1, 1), device="cuda", dtype=torch.float32)
    packed = torch.empty((1, m, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.empty((1, m, k // 16), device="cuda", dtype=torch.float8_e4m3fn)
    restored = torch.empty_like(x)

    quantizer = cudnn.ops.Nvfp4BlockScaleQuantizer(x, one, packed, scales)
    dequantizer = cudnn.ops.Nvfp4BlockScaleDequantizer(packed, scales, one, restored)
    assert quantizer.check_support() and dequantizer.check_support()
    quantizer.compile()
    dequantizer.compile()
    quantizer.execute(x, one, packed, scales)
    dequantizer.execute(packed, scales, one, restored)
    torch.cuda.synchronize()

    assert packed.dtype == torch.uint8
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


@requires_sm100_or_newer
@pytest.mark.L1
def test_nvfp4_prepared_plans_reuse_compiled_k_across_runtime_m():
    x0, encode0, packed0, scales0, output0 = _make_buffers(m=128)
    quantizer = cudnn.ops.Nvfp4BlockScaleQuantizer(x0, encode0, packed0, scales0)
    dequantizer = cudnn.ops.Nvfp4BlockScaleDequantizer(packed0, scales0, encode0, output0)
    assert quantizer.check_support() and dequantizer.check_support()
    quantizer.compile()
    dequantizer.compile()

    x1, encode1, packed1, scales1, output1 = _make_buffers(m=256)
    x1.normal_()
    quantizer.execute(x1, encode1, packed1, scales1)
    dequantizer.execute(packed1, scales1, encode1, output1)
    torch.cuda.synchronize()

    assert torch.isfinite(output1).all()
    assert output1.abs().sum() > 0
