# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end correctness coverage for public NVFP4 convolution graphs."""

from __future__ import annotations

import cudnn
import pytest
import torch

from epilogue_cases import EPILOGUE_CASES, EpilogueCase
from utils import requires_block_scale_cutedsl, requires_sm100, select_and_build_frost_plan

_NCDHW = (1, 128, 1, 12, 16)
_KCTRS = (192, 128, 3, 3, 3)
_PADDING = (1, 1, 1)
_BLOCK_SIZE = 16
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

_OUTPUT_DTYPES = (
    (torch.float32, cudnn.data_type.FLOAT, "fp32"),
    (torch.float16, cudnn.data_type.HALF, "fp16"),
    (torch.bfloat16, cudnn.data_type.BFLOAT16, "bf16"),
    (torch.float8_e4m3fn, cudnn.data_type.FP8_E4M3, "fp8-e4m3"),
    (torch.float8_e5m2, cudnn.data_type.FP8_E5M2, "fp8-e5m2"),
    (None, cudnn.data_type.FP4_E2M1, "fp4-e2m1"),
)

pytestmark = [pytest.mark.L1, requires_sm100, requires_block_scale_cutedsl]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _channels_last_stride(shape) -> tuple[int, ...]:
    _n, c, d, h, w = shape
    return (c * d * h * w, 1, h * w * c, w * c, c)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Expand low-nibble-first E2M1 pairs without changing logical order."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    low = lut[(packed & 0xF).long()]
    high = lut[(packed >> 4).long()]
    return torch.stack((low, high), dim=-1).flatten(-2)


def _reorder_f8_128x4(scales: torch.Tensor) -> torch.Tensor:
    """Reorder a logical scale matrix into the caller-owned F8_128x4 blob."""
    rows, columns = scales.shape
    row_blocks, column_blocks = _ceil_div(rows, 128), _ceil_div(columns, 4)
    padded = torch.zeros(row_blocks * 128, column_blocks * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :columns] = scales
    blocks = padded.view(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


@pytest.fixture(scope="module")
def operands_and_reference():
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("PyTorch lacks native float4_e2m1fn_x2")

    n, c, d, h, w = _NCDHW
    k, _c, t, r, s = _KCTRS
    operands = {}
    for profile_index, profile in enumerate(("signed", "positive", "logical")):
        generator = torch.Generator(device="cuda").manual_seed(2026 + profile_index)
        if profile == "signed":
            image_bytes = torch.randint(0, 256, (n, d, h, w, c // 2), dtype=torch.uint8, device="cuda", generator=generator)
            weight_bytes = torch.randint(0, 256, (k, t, r, s, c // 2), dtype=torch.uint8, device="cuda", generator=generator)
        else:
            image_codes = torch.randint(0 if profile == "logical" else 1, 4, (n, d, h, w, c), dtype=torch.uint8, device="cuda", generator=generator)
            image_bytes = image_codes[..., 0::2] | (image_codes[..., 1::2] << 4)
            if profile == "positive":
                weight_codes = torch.randint(1, 4, (k, t, r, s, c), dtype=torch.uint8, device="cuda", generator=generator)
                weight_bytes = weight_codes[..., 0::2] | (weight_codes[..., 1::2] << 4)
            else:
                # Make the convolution copy input channel zero so logical_not
                # sees both exact zero and nonzero accumulator values.
                weight_bytes = torch.zeros((k, t, r, s, c // 2), dtype=torch.uint8, device="cuda")
                weight_bytes[:, t // 2, r // 2, s // 2, 0] = 2

        image = image_bytes.view(torch.float4_e2m1fn_x2).permute(0, 4, 1, 2, 3)
        weight = weight_bytes.view(torch.float4_e2m1fn_x2).permute(0, 4, 1, 2, 3)

        sf_c = c // _BLOCK_SIZE
        sfa_logical = (torch.randint(1, 3, (n, d, h, w, sf_c), dtype=torch.int32, device="cuda", generator=generator) / 32).to(torch.float8_e4m3fn)
        sfb_logical = (torch.randint(1, 3, (k, t, r, s, sf_c), dtype=torch.int32, device="cuda", generator=generator) / 32).to(torch.float8_e4m3fn)

        sfa_columns = _ceil_div(sf_c, 4) * 4
        sfa = torch.zeros(n * d * h * w, sfa_columns, 1, dtype=torch.float8_e4m3fn, device="cuda")
        sfa[:, :sf_c, 0] = sfa_logical.reshape(-1, sf_c)
        sfb = _reorder_f8_128x4(sfb_logical.reshape(k, -1))

        image_dequant = _unpack_fp4(image_bytes) * sfa_logical.float().repeat_interleave(_BLOCK_SIZE, dim=-1)
        weight_dequant = _unpack_fp4(weight_bytes) * sfb_logical.float().repeat_interleave(_BLOCK_SIZE, dim=-1)
        reference = torch.nn.functional.conv3d(
            image_dequant.permute(0, 4, 1, 2, 3),
            weight_dequant.permute(0, 4, 1, 2, 3),
            padding=_PADDING,
        )
        operands[profile] = (image, weight, sfa, sfb, reference)

    assert torch.all(operands["positive"][-1] > 0)
    assert torch.any(operands["logical"][-1] == 0)
    assert torch.any(operands["logical"][-1] != 0)
    return operands


def _build_graph(output_shape, output_dtype, epilogue: str | EpilogueCase):
    n, c, d, h, w = _NCDHW
    k, _c, t, r, s = _KCTRS
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    image = graph.tensor(name="A", dim=_NCDHW, stride=_channels_last_stride(_NCDHW), data_type=cudnn.data_type.FP4_E2M1)
    weight = graph.tensor(name="B", dim=_KCTRS, stride=_channels_last_stride(_KCTRS), data_type=cudnn.data_type.FP4_E2M1)
    sfa = graph.tensor(
        name="SFA",
        dim=(n, c // _BLOCK_SIZE, d, h, w),
        stride=_channels_last_stride((n, c // _BLOCK_SIZE, d, h, w)),
        data_type=cudnn.data_type.FP8_E4M3,
    )
    sfb = graph.tensor(
        name="SFB",
        dim=(k, c // _BLOCK_SIZE, t, r, s),
        stride=_channels_last_stride((k, c // _BLOCK_SIZE, t, r, s)),
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    dequant_image = graph.block_scale_dequantize(
        input=image,
        descale=sfa,
        block_size=(1, _BLOCK_SIZE, 1, 1, 1),
        compute_data_type=cudnn.data_type.FLOAT,
        name="dequantize_A",
    )
    dequant_weight = graph.block_scale_dequantize(
        input=weight,
        descale=sfb,
        block_size=(1, _BLOCK_SIZE, 1, 1, 1),
        compute_data_type=cudnn.data_type.FLOAT,
        name="dequantize_B",
    )
    convolution = graph.conv_fprop(
        dequant_image,
        dequant_weight,
        name="nvfp4_conv",
        pre_padding=_PADDING,
        post_padding=_PADDING,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        compute_data_type=cudnn.data_type.FLOAT,
    )

    result = convolution
    if epilogue != "implicit_identity":
        convolution.set_data_type(cudnn.data_type.FLOAT)
        if isinstance(epilogue, EpilogueCase):
            result = epilogue.graph_fn(graph, convolution)
        else:
            result = (
                graph.identity(convolution, compute_data_type=cudnn.data_type.FLOAT, name="identity")
                if epilogue == "explicit_identity"
                else graph.relu(convolution, compute_data_type=cudnn.data_type.FLOAT, name="relu")
            )
        result.set_dim(output_shape).set_stride(_channels_last_stride(output_shape))

    output_scale = None
    if output_dtype == cudnn.data_type.FP4_E2M1:
        result.set_data_type(cudnn.data_type.FLOAT)
        output, output_scale = graph.block_scale_quantize(input=result, block_size=_BLOCK_SIZE, axis=1, name="quantize_D")
        output.set_output(True).set_data_type(output_dtype).set_dim(output_shape).set_stride(_channels_last_stride(output_shape))
        output_scale_shape = (n, k // _BLOCK_SIZE, *output_shape[2:])
        output_scale.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3).set_dim(output_scale_shape).set_stride(_channels_last_stride(output_scale_shape))
    else:
        output = result
        output.set_output(True).set_data_type(output_dtype).set_dim(output_shape).set_stride(_channels_last_stride(output_shape))

    return graph, (image, weight, sfa, sfb, output, output_scale)


def _activated(reference: torch.Tensor, epilogue: str | EpilogueCase) -> torch.Tensor:
    if isinstance(epilogue, EpilogueCase):
        return epilogue.reference_fn(reference)
    return torch.relu(reference) if epilogue == "relu" else reference


_CORRECTNESS_CASES = tuple(
    pytest.param(epilogue, torch_dtype, cudnn_dtype, id=f"{epilogue}-{dtype_id}")
    for epilogue in ("implicit_identity", "explicit_identity", "relu")
    for torch_dtype, cudnn_dtype, dtype_id in _OUTPUT_DTYPES
) + tuple(
    pytest.param(case, torch.bfloat16, cudnn.data_type.BFLOAT16, id=f"{case.name}-bf16") for case in EPILOGUE_CASES if case.name not in ("identity", "relu")
)


@pytest.mark.parametrize("epilogue,torch_dtype,cudnn_dtype", _CORRECTNESS_CASES)
def test_block_scale_conv_correctness(monkeypatch, operands_and_reference, epilogue, torch_dtype, cudnn_dtype) -> None:
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    input_profile = epilogue.input_profile if isinstance(epilogue, EpilogueCase) else "signed"
    image, weight, sfa, sfb, reference = operands_and_reference[input_profile]
    expected_accumulator = _activated(reference, epilogue)
    graph, (image_desc, weight_desc, sfa_desc, sfb_desc, output_desc, output_scale_desc) = _build_graph(reference.shape, cudnn_dtype, epilogue)

    n, k, d, h, w = reference.shape
    if cudnn_dtype == cudnn.data_type.FP4_E2M1:
        output_bytes = torch.empty((n, d, h, w, k // 2), dtype=torch.uint8, device="cuda")
        output = output_bytes.view(torch.float4_e2m1fn_x2).permute(0, 4, 1, 2, 3)
        output_scale = torch.empty((n * d * h * w, k // _BLOCK_SIZE, 1), dtype=torch.float8_e4m3fn, device="cuda")
    else:
        output_bytes = None
        output = torch.empty(reference.shape, dtype=torch_dtype, device="cuda", memory_format=torch.channels_last_3d)
        output_scale = None

    select_and_build_frost_plan(graph, "frost_conv")
    variant_pack = {
        image_desc: image,
        weight_desc: weight,
        sfa_desc: sfa,
        sfb_desc: sfb,
        output_desc: output,
    }
    if output_scale_desc is not None:
        variant_pack[output_scale_desc] = output_scale
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    if cudnn_dtype == cudnn.data_type.FP4_E2M1:
        actual = _unpack_fp4(output_bytes) * output_scale.view(n, d, h, w, k // _BLOCK_SIZE).float().repeat_interleave(_BLOCK_SIZE, dim=-1)
        actual = actual.permute(0, 4, 1, 2, 3)
        error = (actual - expected_accumulator).abs()
        tolerance = 0.34 * expected_accumulator.abs() + 0.05 * expected_accumulator.abs().max()
        assert (error <= tolerance).float().mean().item() > 0.999
    else:
        expected = expected_accumulator.to(torch_dtype).float()
        epsilon = float(torch.finfo(torch_dtype).eps)
        activation_tolerance = 2e-2 if isinstance(epilogue, EpilogueCase) else 2e-3
        torch.testing.assert_close(
            output.float(),
            expected,
            atol=max(activation_tolerance, 2 * epsilon),
            rtol=max(activation_tolerance, 2 * epsilon),
        )
