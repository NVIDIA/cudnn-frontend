# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an MXFP8 dequantize, matmul, and block-scale quantize graph."""

import cudnn
import torch


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def sample() -> None:
    batch, rows, columns, inner = 1, 128, 128, 128
    block_size = 32
    packed_rows = round_up(rows, 128)
    packed_columns = round_up(columns, 128)
    packed_inner = round_up((inner + block_size - 1) // block_size, 4)
    packed_output_columns = round_up((columns + block_size - 1) // block_size, 4)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = graph.tensor(
        name="A",
        dim=[batch, rows, inner],
        stride=[rows * inner, inner, 1],
        data_type=cudnn.data_type.FP8_E4M3,
    )
    B = graph.tensor(
        name="B",
        dim=[batch, inner, columns],
        stride=[inner * columns, 1, inner],
        data_type=cudnn.data_type.FP8_E4M3,
    )
    descale_a = graph.tensor(
        name="descale_a",
        dim=[batch, packed_rows, packed_inner],
        stride=[packed_rows * packed_inner, packed_inner, 1],
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    descale_b = graph.tensor(
        name="descale_b",
        dim=[batch, packed_inner, packed_columns],
        stride=[packed_inner * packed_columns, 1, packed_inner],
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )

    dequantized_a = graph.block_scale_dequantize(
        A,
        descale_a,
        block_size=[1, block_size],
        name="dequantize_a",
    )
    dequantized_b = graph.block_scale_dequantize(
        B,
        descale_b,
        block_size=[block_size, 1],
        name="dequantize_b",
    )
    product = graph.matmul(
        dequantized_a,
        dequantized_b,
        compute_data_type=cudnn.data_type.FLOAT,
        name="matmul",
    )
    quantized, output_scale = graph.block_scale_quantize(
        product,
        block_size=block_size,
        axis=2,
        transpose=False,
        name="quantize_output",
    )
    quantized.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    output_scale.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0).set_reordering_type(
        cudnn.tensor_reordering.F8_128x4
    )
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

    # The Graph API binds raw storage by pointer. uint8 tensors hold the FP8
    # bit patterns, while PyTorch's E8M0 dtype holds scale values.
    a_data = torch.zeros((batch, rows, inner), device="cuda", dtype=torch.uint8)
    b_data = torch.zeros((batch, inner, columns), device="cuda", dtype=torch.uint8)
    scale_a = torch.ones(
        (batch, packed_rows, packed_inner), device="cuda", dtype=torch.float8_e8m0fnu
    )
    scale_b = torch.ones(
        (batch, packed_inner, packed_columns), device="cuda", dtype=torch.float8_e8m0fnu
    )
    quantized_data = torch.empty((batch, rows, columns), device="cuda", dtype=torch.uint8)
    scale_data = torch.empty(
        (batch, packed_rows, packed_output_columns),
        device="cuda",
        dtype=torch.float8_e8m0fnu,
    )
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(
        {
            A: a_data,
            B: b_data,
            descale_a: scale_a,
            descale_b: scale_b,
            quantized: quantized_data,
            output_scale: scale_data,
        },
        workspace,
    )
    torch.cuda.synchronize()

    # Zero inputs must remain zero through dequantize, matmul, and requantize.
    assert torch.count_nonzero(quantized_data).item() == 0
    print("MXFP8 block-scaled matmul ok: zero result and output scales produced")


if __name__ == "__main__":
    sample()
