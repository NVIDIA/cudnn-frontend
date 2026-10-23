# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the OSS RMSNorm-SiLU Graph fusion and check a PyTorch reference."""

import torch
import torch.nn.functional as F

import cudnn


def main() -> None:
    torch.manual_seed(17)
    tokens, width = 1560, 512
    epsilon_value = 1e-6 / width

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_tensor = graph.tensor(
        name="X",
        dim=[tokens, width, 1, 1],
        stride=[width, 1, 1, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    scale_tensor = graph.tensor(
        name="scale",
        dim=[1, width, 1, 1],
        stride=[width, 1, 1, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    epsilon_tensor = graph.tensor(
        name="epsilon",
        dim=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    normalized = graph.rmsnorm(
        input=x_tensor,
        scale=scale_tensor,
        epsilon=epsilon_tensor,
        norm_forward_phase=cudnn.norm_forward_phase.INFERENCE,
    )[0]
    normalized.set_dim([tokens, width, 1, 1])
    normalized.set_stride([width, 1, 1, 1])
    normalized.set_data_type(cudnn.data_type.BFLOAT16)
    output_tensor = graph.swish(input=normalized, swish_beta=1.0)
    output_tensor.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    graph.build([cudnn.heur_mode.OPENSOURCE])

    x = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16) * 3.0 + 1.0
    scale = torch.rand(width, device="cuda", dtype=torch.bfloat16) + 0.5
    epsilon = torch.full((1, 1, 1, 1), epsilon_value, dtype=torch.float32)
    output = torch.empty(tokens, width, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(
        {
            x_tensor: x.view(tokens, width, 1, 1),
            scale_tensor: scale.view(1, width, 1, 1),
            epsilon_tensor: epsilon,
            output_tensor: output.view(tokens, width, 1, 1),
        },
        workspace,
    )
    torch.cuda.synchronize()

    rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + epsilon_value)
    expected = F.silu(x.float() * rms * scale.float()).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
    print("OSS RMSNorm-SiLU ok: BF16 Graph fusion")


if __name__ == "__main__":
    main()
