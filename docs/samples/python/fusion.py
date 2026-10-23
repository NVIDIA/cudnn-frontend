# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute matmul, bias, and ReLU in one cuDNN graph."""

import cudnn
import torch


def main() -> None:
    a_values = torch.arange(2 * 4 * 8, device="cuda").reshape(2, 4, 8)
    b_values = torch.arange(8 * 6, device="cuda").reshape(1, 8, 6)
    bias_values = torch.arange(6, device="cuda").reshape(1, 1, 6)
    a_data = (((a_values % 17) - 8) / 8).to(torch.bfloat16)
    b_data = (((b_values % 13) - 6) / 8).to(torch.bfloat16)
    bias_data = ((bias_values - 3) / 4).to(torch.bfloat16)

    with cudnn.Graph(
        io_data_type=torch.bfloat16,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        heuristics=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK],
        inputs=[a_data, b_data, bias_data],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        product = graph.matmul(A=a_data, B=b_data, compute_data_type=torch.float32, name="matmul")
        shifted = graph.bias(
            input=product,
            bias=bias_data,
            compute_data_type=torch.float32,
            name="bias_add",
        )
        result = graph.relu(input=shifted, compute_data_type=torch.float32, name="relu")
        result.set_name("Y").set_output(True).set_data_type(torch.bfloat16)

    actual = graph(a_data, b_data, bias_data)
    expected = torch.relu(torch.matmul(a_data.float(), b_data.float()) + bias_data.float()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    print(f"fusion ok: shape={tuple(actual.shape)}, workspace={graph.get_workspace_size()} bytes")


if __name__ == "__main__":
    main()
