# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute and verify MoE grouped weight gradients in NONE routing mode."""

import math
import cudnn
import torch


def reference(
    doutput: torch.Tensor,
    token: torch.Tensor,
    offsets: list[int],
    experts: int,
    rows: int,
    hidden: int,
    output: int,
) -> torch.Tensor:
    token_2d = token.view(rows, hidden).float()
    doutput_2d = doutput.view(rows, output).float()
    dweight = torch.zeros(experts, hidden, output, device=token.device, dtype=torch.float32)
    bounds = offsets + [rows]
    for expert in range(experts):
        start, end = bounds[expert], bounds[expert + 1]
        if end > start:
            dweight[expert] = token_2d[start:end].t() @ doutput_2d[start:end]
    return dweight


def sample() -> None:
    experts, rows, hidden, output = 36, 2000, 520, 248
    offsets = list(range(19)) + [
        127,
        255,
        383,
        483,
        515,
        643,
        718,
        924,
        1100,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        1900,
    ]

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    doutput_tensor = graph.tensor(
        name="doutput",
        dim=[1, rows, output],
        stride=[rows * output, output, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    token_tensor = graph.tensor(
        name="token",
        dim=[1, rows, hidden],
        stride=[rows * hidden, hidden, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    offsets_tensor = graph.tensor(
        name="first_token_offset",
        dim=[experts, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    dweight_tensor = graph.moe_grouped_matmul_bwd(
        doutput_tensor,
        token_tensor,
        offsets_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe_grouped_matmul_bwd",
    )
    dweight_tensor.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    graph.build([cudnn.heur_mode.A])

    torch.manual_seed(7)
    doutput = torch.randn(rows * output, device="cuda", dtype=torch.bfloat16)
    token = torch.randn(rows * hidden, device="cuda", dtype=torch.bfloat16)
    first_token_offset = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    dweight = torch.empty(experts * hidden * output, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(
        {
            doutput_tensor: doutput,
            token_tensor: token,
            offsets_tensor: first_token_offset,
            dweight_tensor: dweight,
        },
        workspace,
    )
    torch.cuda.synchronize()

    actual = dweight.view(experts, output, hidden).transpose(1, 2).float()
    expected = reference(doutput, token, offsets, experts, rows, hidden, output)
    tolerance = 2e-2 * max(1.0, math.sqrt(rows / 128.0))
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    print("moe grouped matmul backward ok: dweight verified per expert")


if __name__ == "__main__":
    sample()
