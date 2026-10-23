# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete standalone Python sample.
API page: docs/api/python/rmsnorm_backward.md
API: rmsnorm_backward
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
    grad = torch.randn_like(x)
    inv_var = torch.rsqrt((x.float() * x.float()).mean((1, 2, 3), keepdim=True) + 1e-3)
    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    grad_desc = graph.tensor_like(grad)
    x_desc = graph.tensor_like(x)
    scale_desc = graph.tensor_like(scale)
    inv_var_desc = graph.tensor_like(inv_var)
    dx_desc, dscale_desc, dbias_desc = graph.rmsnorm_backward(
        grad=grad_desc,
        input=x_desc,
        scale=scale_desc,
        inv_variance=inv_var_desc,
        has_dbias=True,
    )
    dx_desc.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    dscale_desc.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    dbias_desc.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    dx = torch.empty_like(x)
    dscale = torch.empty_like(scale)
    dbias = torch.empty_like(scale)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(
        {
            grad_desc: grad,
            x_desc: x,
            scale_desc: scale,
            inv_var_desc: inv_var,
            dx_desc: dx,
            dscale_desc: dscale,
            dbias_desc: dbias,
        },
        workspace,
    )
    torch.cuda.synchronize()
    print(f"rmsnorm backward dx={tuple(dx.shape)}, dscale={tuple(dscale.shape)}")


if __name__ == "__main__":
    run_sample(sample)
