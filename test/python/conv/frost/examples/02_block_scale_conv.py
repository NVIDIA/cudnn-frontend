# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 02: NVFP4 block-scaled 3D convolution with NVFP4 output."""

from __future__ import annotations

import cudnn
import torch

from common import build_frost_conv_plans

NCDHW = (1, 128, 1, 12, 16)
KCTRS = (192, 128, 3, 3, 3)
PADDING = (1, 1, 1)
BLOCK_SIZE = 16
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _channels_last_stride(shape) -> tuple[int, ...]:
    _n, c, d, h, w = shape
    return (c * d * h * w, 1, h * w * c, w * c, c)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Expand low-nibble-first E2M1 pairs without changing their logical order."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    low = lut[(packed & 0xF).long()]
    high = lut[(packed >> 4).long()]
    return torch.stack((low, high), dim=-1).flatten(-2)


def _reorder_f8_128x4(scales: torch.Tensor) -> torch.Tensor:
    """Reorder a logical (rows, columns) scale matrix into one dense F8_128x4 blob."""
    rows, columns = scales.shape
    row_blocks, column_blocks = _ceil_div(rows, 128), _ceil_div(columns, 4)
    padded = torch.zeros(row_blocks * 128, column_blocks * 4, dtype=scales.dtype, device=scales.device)
    padded[:rows, :columns] = scales
    blocks = padded.view(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _make_operands():
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("this example requires torch.float4_e2m1fn_x2")

    n, c, d, h, w = NCDHW
    k, _c, t, r, s = KCTRS
    generator = torch.Generator(device="cuda").manual_seed(2026)

    # Allocate packed bytes in the kernel's physical NDHWC/KTRSC order, then
    # expose zero-copy NCDHW/KCTRS views to graph.execute().
    image_bytes = torch.randint(0, 256, (n, d, h, w, c // 2), dtype=torch.uint8, device="cuda", generator=generator)
    weight_bytes = torch.randint(0, 256, (k, t, r, s, c // 2), dtype=torch.uint8, device="cuda", generator=generator)
    image = image_bytes.view(torch.float4_e2m1fn_x2).permute(0, 4, 1, 2, 3)
    weight = weight_bytes.view(torch.float4_e2m1fn_x2).permute(0, 4, 1, 2, 3)

    sf_c = c // BLOCK_SIZE
    sfa_logical = torch.randint(1, 3, (n, d, h, w, sf_c), dtype=torch.int32, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    sfb_logical = torch.randint(1, 3, (k, t, r, s, sf_c), dtype=torch.int32, device="cuda", generator=generator).to(torch.float8_e4m3fn)

    # SFA is unswizzled with one padded scale row per input pixel. C=128 gives
    # eight scales, already a multiple of the kernel's four-scale load group.
    sfa_columns = _ceil_div(sf_c, 4) * 4
    sfa = torch.zeros(n * d * h * w, sfa_columns, 1, dtype=torch.float8_e4m3fn, device="cuda")
    sfa[:, :sf_c, 0] = sfa_logical.reshape(-1, sf_c)

    # SFB is reordered by the caller. Its columns flatten (T,R,S,C/16), with
    # C/16 fastest, exactly matching the filter's KTRSC order.
    sfb = _reorder_f8_128x4(sfb_logical.reshape(k, -1))

    image_dequant = _unpack_fp4(image_bytes) * sfa_logical.float().repeat_interleave(BLOCK_SIZE, dim=-1)
    weight_dequant = _unpack_fp4(weight_bytes) * sfb_logical.float().repeat_interleave(BLOCK_SIZE, dim=-1)
    reference = torch.nn.functional.conv3d(
        image_dequant.permute(0, 4, 1, 2, 3),
        weight_dequant.permute(0, 4, 1, 2, 3),
        padding=PADDING,
    )
    return image, weight, sfa, sfb, reference


def _build_graph(output_shape):
    n, c, d, h, w = NCDHW
    k, _c, t, r, s = KCTRS
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    image = graph.tensor(name="A", dim=NCDHW, stride=_channels_last_stride(NCDHW), data_type=cudnn.data_type.FP4_E2M1)
    weight = graph.tensor(name="B", dim=KCTRS, stride=_channels_last_stride(KCTRS), data_type=cudnn.data_type.FP4_E2M1)
    sfa = graph.tensor(
        name="SFA",
        dim=(n, c // BLOCK_SIZE, d, h, w),
        stride=_channels_last_stride((n, c // BLOCK_SIZE, d, h, w)),
        data_type=cudnn.data_type.FP8_E4M3,
    )
    sfb = graph.tensor(
        name="SFB",
        dim=(k, c // BLOCK_SIZE, t, r, s),
        stride=_channels_last_stride((k, c // BLOCK_SIZE, t, r, s)),
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    dequant_image = graph.block_scale_dequantize(
        input=image,
        descale=sfa,
        block_size=(1, BLOCK_SIZE, 1, 1, 1),
        compute_data_type=cudnn.data_type.FLOAT,
        name="dequantize_A",
    )
    dequant_weight = graph.block_scale_dequantize(
        input=weight,
        descale=sfb,
        block_size=(1, BLOCK_SIZE, 1, 1, 1),
        compute_data_type=cudnn.data_type.FLOAT,
        name="dequantize_B",
    )
    convolution = graph.conv_fprop(
        dequant_image,
        dequant_weight,
        name="nvfp4_conv",
        pre_padding=PADDING,
        post_padding=PADDING,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        compute_data_type=cudnn.data_type.FLOAT,
    )
    output, output_scale = graph.block_scale_quantize(
        input=convolution,
        block_size=BLOCK_SIZE,
        axis=1,
        name="quantize_D",
    )
    output.set_output(True).set_data_type(cudnn.data_type.FP4_E2M1).set_dim(output_shape).set_stride(_channels_last_stride(output_shape))
    output_scale_shape = (n, k // BLOCK_SIZE, *output_shape[2:])
    output_scale.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3).set_dim(output_scale_shape).set_stride(_channels_last_stride(output_scale_shape))
    return graph, (image, weight, sfa, sfb, output, output_scale)


def main() -> None:
    image, weight, sfa, sfb, reference = _make_operands()
    graph, (image_desc, weight_desc, sfa_desc, sfb_desc, output_desc, output_scale_desc) = _build_graph(reference.shape)
    build_frost_conv_plans(graph)

    n, k, d, h, w = reference.shape
    output_bytes = torch.empty((n, d, h, w, k // 2), dtype=torch.uint8, device="cuda")
    output = output_bytes.view(torch.float4_e2m1fn_x2).permute(0, 4, 1, 2, 3)
    output_scale = torch.empty((n * d * h * w, k // BLOCK_SIZE, 1), dtype=torch.float8_e4m3fn, device="cuda")
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    graph.execute(
        {
            image_desc: image,
            weight_desc: weight,
            sfa_desc: sfa,
            sfb_desc: sfb,
            output_desc: output,
            output_scale_desc: output_scale,
        },
        workspace,
    )
    torch.cuda.synchronize()

    output_dequant = _unpack_fp4(output_bytes) * output_scale.view(n, d, h, w, k // BLOCK_SIZE).float().repeat_interleave(BLOCK_SIZE, dim=-1)
    output_dequant = output_dequant.permute(0, 4, 1, 2, 3)
    error = (output_dequant - reference).abs()
    # NVFP4 is very coarse. Relative error can be as large as ~33% when rounding between 0.5 and 1.0. Values near zero need an absolute tolerance, hence the
    # additional term.
    tolerance = 0.34 * reference.abs() + 0.05 * reference.abs().max()
    assert (error <= tolerance).float().mean().item() > 0.999
    print("[02] PASS    NVFP4 block-scaled conv_fprop with NVFP4 output through cudnn.pygraph")


if __name__ == "__main__":
    main()
