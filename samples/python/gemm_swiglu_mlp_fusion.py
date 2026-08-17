# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo for the dense bf16 SwiGLU-MLP cuDNN autograd op (``cudnn.gemm.ops.swiglu_mlp``).

The op computes ``out = (silu(x @ Wg^T) * (x @ Wu^T)) @ Wd^T`` with every GEMM and the
SwiGLU on cuDNN. What the cuDNN graph fuses today (no new kernel):
  * forward  ``silu(x @ Wg^T) * (x @ Wu^T)`` -> gate GEMM + up GEMM + SiLU + mul compile
    into ONE cuDNN kernel (the FORT-native runtime-fusion engine on SM100); the demo prints
    the actual launch count + kernel name so "1 kernel" is evidence, not assumed. The down
    projection is a separate GEMM (a 3-GEMM single graph does not compile).
  * backward dSwiGLU -> two single-output cuDNN pointwise kernels; as a matmul EPILOGUE the
    ``matmul(dout,Wd)->dh`` fused with the dSwiGLU is ~2.3x the unfused GEMM + elementwise.

Measured on B200 at the Qwen3.5-27B MLP shape (H5120 I17408), vs torch+cuBLAS:
  * forward fused is a real win: 1.05-1.20x eager AND under CUDA-graph replay, M in {2048..16384}
    (fusing gate+up+act into one kernel beats 2 cuBLAS GEMMs + act);
  * forward+backward eager is ~0.86x (a regression) NOT because the kernels are slow — the cuDNN
    backend execute is already at cuBLAS parity (~8.4us vs ~7.6us for a 256^3 matmul); the gap is
    removable per-call FE wrapper cost across 6-8 plain GEMMs plus a bwd recompute-vs-save tradeoff.
So the remaining levers are (a) a memoized matmul hot path to erase the per-call wrapper cost
(cf. the grouped-GEMM/cuTeDSL fast paths), and (b) a cuBLAS-class fused GEMM+epilogue for the
backward. The dSwiGLU itself already fuses well, so it is not the gap. See
docs/framework_integration_performance.md.

Numerically matches torch to bf16 noise (fwd + all four gradients). Verified on B200.
"""

import torch
import torch.nn.functional as F

from cudnn.gemm.ops.swiglu_mlp import swiglu_mlp, _swiglu_act


def _demo():
    dev = next((torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count()) if torch.cuda.get_device_properties(i).major >= 10), None)
    if dev is None:
        print("no SM100+ (Blackwell) GPU found; the fused runtime-fusion engine needs one — skipping.")
        return
    M, H, interm = 2048, 5120, 17408  # Qwen3.5-27B MLP shape
    torch.manual_seed(0)
    x = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
    Wg = (torch.randn(interm, H, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wu = (torch.randn(interm, H, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wd = (torch.randn(H, interm, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    do = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16)
    xr, Wgr, Wur, Wdr = (t.detach().clone().requires_grad_(True) for t in (x, Wg, Wu, Wd))

    swiglu_mlp(x, Wg, Wu, Wd).backward(do)
    ((F.silu(xr @ Wgr.t()) * (xr @ Wur.t())) @ Wdr.t()).backward(do)

    def rel(a, b):
        return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-9)

    print(f"device {torch.cuda.get_device_properties(dev).name}; SwiGLU-MLP M{M} H{H} I{interm}")
    print(f"fwd  rel={rel(swiglu_mlp(x, Wg, Wu, Wd), (F.silu(xr @ Wgr.t()) * (xr @ Wur.t())) @ Wdr.t()):.2e}")
    for n, a, b in [("dx", x.grad, xr.grad), ("dWg", Wg.grad, Wgr.grad), ("dWu", Wu.grad, Wur.grad), ("dWd", Wd.grad, Wdr.grad)]:
        print(f"bwd {n:3} rel={rel(a, b):.2e}")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        _swiglu_act(x.detach().reshape(M, H), Wg.detach(), Wu.detach())
        torch.cuda.synchronize()
    # count actual GPU launches (ev.count), not distinct keys, and name them — so "1 kernel"
    # is evidence, and the kernel name shows which engine served the fusion (not assumed).
    kernels = [(ev.key, ev.count) for ev in prof.key_averages() if ev.self_device_time_total > 0]
    launches = sum(c for _, c in kernels)
    print(f"fused swiglu_act (gate+up+silu+mul) -> {launches} GPU launch(es): {', '.join(k[:60] for k, _ in kernels)}")


if __name__ == "__main__":
    _demo()
