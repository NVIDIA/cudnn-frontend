# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/rmsnorm.md
API: rmsnorm
Use case: Basic
Source: adapted from `samples/llm_coverage/test_rmsnorm_b+s+d.py`.
"""

import sys

try:
    import cudnn
    import torch
except Exception as exc:
    print(f"SKIP: could not import cudnn or torch: {exc}")
    raise SystemExit(0)


def run_sample(sample):
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available to PyTorch")
        return
    try:
        sample()
    except Exception as exc:
        if "--strict" in sys.argv:
            raise
        print(f"SKIP: sample could not run on this installation: {exc}")


def sample():
    shape = (8, 64, 1, 1)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn((1, 64, 1, 1), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(scale)
    epsilon = torch.full((1, 1, 1, 1), 1e-3, dtype=torch.float32)
    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    scale_desc = graph.tensor_like(scale)
    bias_desc = graph.tensor_like(bias)
    epsilon_desc = graph.tensor_like(epsilon)
    y_desc, inv_var_desc = graph.rmsnorm(
        norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
        input=x_desc,
        scale=scale_desc,
        bias=bias_desc,
        epsilon=epsilon_desc,
    )
    y_desc.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    inv_var_desc.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    y = torch.empty_like(x)
    inv_var = torch.empty((8, 1, 1, 1), device="cuda", dtype=torch.float32)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(
        {
            x_desc: x,
            scale_desc: scale,
            bias_desc: bias,
            epsilon_desc: epsilon,
            y_desc: y,
            inv_var_desc: inv_var,
        },
        workspace,
    )
    torch.cuda.synchronize()
    expected_inv = torch.rsqrt((x.float() * x.float()).mean((1, 2, 3), keepdim=True) + 1e-3)
    expected = (x.float() * expected_inv * scale.float() + bias.float()).to(torch.bfloat16)
    torch.testing.assert_close(y, expected, atol=0.125, rtol=0.125)
    print(f"rmsnorm output={tuple(y.shape)}, inv_var={tuple(inv_var.shape)}")


if __name__ == "__main__":
    run_sample(sample)
