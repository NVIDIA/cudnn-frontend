# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ComputeLab compile and correctness smoke for dense causal-conv backward."""

from pathlib import Path

import cudnn
import torch
import torch.nn.functional as F

SOURCE_CUDNN = Path(__file__).resolve().parents[1] / "python" / "cudnn"
if str(SOURCE_CUDNN) not in cudnn.__path__:
    cudnn.__path__.insert(0, str(SOURCE_CUDNN))

from cudnn.causal_conv1d_bulk_sm100.backward import (
    compile_causal_conv1d_bulk_bwd_prototype,
)
from fla.modules.conv.triton.ops import causal_conv1d_bwd as fla_bwd


def reference(x, weight, dy):
    x_ref = x.float().requires_grad_()
    weight_ref = weight.float().requires_grad_()
    tokens = x.shape[1]
    z = F.conv1d(
        x_ref.transpose(1, 2),
        weight_ref.unsqueeze(1),
        padding=3,
        groups=x.shape[2],
    )[
        ..., :tokens
    ].transpose(1, 2)
    F.silu(z).backward(dy.float())
    return x_ref.grad, weight_ref.grad


@torch.no_grad()
def main():
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    x = torch.randn((1, 257, 256), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    weight = torch.randn((256, 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    dy = torch.randn(x.shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    with torch.enable_grad():
        reference_dx, reference_dw = reference(x, weight, dy)

    fla_dx, fla_dw = fla_bwd(
        x=x,
        dy=dy,
        dht=None,
        weight=weight,
        activation="silu",
    )[:2]
    fla_dx_diff = fla_dx.float() - reference_dx
    fla_dw_diff = fla_dw.float() - reference_dw
    fla_metrics = {
        "dx_max_abs": float(fla_dx_diff.abs().max()),
        "dx_rel_l2": float(torch.linalg.vector_norm(fla_dx_diff) / torch.linalg.vector_norm(reference_dx)),
        "dw_max_abs": float(fla_dw_diff.abs().max()),
        "dw_rel_l2": float(torch.linalg.vector_norm(fla_dw_diff) / torch.linalg.vector_norm(reference_dw)),
    }

    rows = {}
    for schedule in ("t32", "t64", "t128", "t64-partial"):
        backend = compile_causal_conv1d_bulk_bwd_prototype(x, weight, dy, schedule=schedule)
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight, dtype=torch.float32)
        workspace = None
        if backend.dweight_workspace_numel:
            workspace = torch.empty(backend.dweight_workspace_numel, device="cuda", dtype=torch.float32)
        backend.execute(x, weight, dy, dx, dw, dweight_workspace=workspace)
        torch.cuda.synchronize()
        dx_diff = dx.float() - reference_dx
        dw_diff = dw - reference_dw
        metrics = {
            "dx_max_abs": float(dx_diff.abs().max()),
            "dx_rel_l2": float(torch.linalg.vector_norm(dx_diff) / torch.linalg.vector_norm(reference_dx)),
            "dw_max_abs": float(dw_diff.abs().max()),
            "dw_rel_l2": float(torch.linalg.vector_norm(dw_diff) / torch.linalg.vector_norm(reference_dw)),
            "workspace_bytes": backend.dweight_workspace_bytes,
        }
        torch.testing.assert_close(dx.float(), reference_dx, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dw, reference_dw, atol=1e-1, rtol=5e-2)
        rows[schedule] = metrics
    print(
        {
            "device": torch.cuda.get_device_name(),
            "shape": list(x.shape),
            "fla_vs_fp32": fla_metrics,
            "results": rows,
        }
    )


if __name__ == "__main__":
    main()
