# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute and numerically check MoE grouped matmul in NONE routing mode."""

import torch

import cudnn


def expert_weight(weight: torch.Tensor, expert: int, hidden: int, output: int) -> torch.Tensor:
    block = weight[expert * hidden * output : (expert + 1) * hidden * output]
    return block.view(output, hidden).t().float()


def reference(
    token: torch.Tensor,
    weight: torch.Tensor,
    offsets: list[int],
    experts: int,
    rows: int,
    hidden: int,
    output: int,
) -> torch.Tensor:
    token_2d = token.view(rows, hidden).float()
    result = torch.zeros(rows, output, device=token.device, dtype=torch.float32)
    bounds = offsets + [rows]
    for expert in range(experts):
        start, end = bounds[expert], bounds[expert + 1]
        if end > start:
            result[start:end] = token_2d[start:end] @ expert_weight(weight, expert, hidden, output)
    return result


def main() -> None:
    experts, rows, hidden, output = 2, 64, 128, 64
    offsets = [0, rows // 2]

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    token_tensor = graph.tensor(
        name="token",
        dim=[1, rows, hidden],
        stride=[rows * hidden, hidden, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    weight_tensor = graph.tensor(
        name="weight",
        dim=[experts, hidden, output],
        stride=[hidden * output, 1, hidden],
        data_type=cudnn.data_type.BFLOAT16,
    )
    offsets_tensor = graph.tensor(
        name="first_token_offset",
        dim=[experts, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    result_tensor = graph.moe_grouped_matmul(
        token_tensor,
        weight_tensor,
        offsets_tensor,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    result_tensor.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    graph.build([cudnn.heur_mode.A])
    token = torch.randn(rows * hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(experts * hidden * output, device="cuda", dtype=torch.bfloat16)
    first_token_offset = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    result = torch.empty(rows * output, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(
        {
            token_tensor: token,
            weight_tensor: weight,
            offsets_tensor: first_token_offset,
            result_tensor: result,
        },
        workspace,
    )
    torch.cuda.synchronize()

    expected = reference(token, weight, offsets, experts, rows, hidden, output)
    torch.testing.assert_close(result.view(rows, output).float(), expected, rtol=3e-2, atol=3e-2)
    print("moe grouped matmul ok: NONE mode")


if __name__ == "__main__":
    main()
