# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A compiled FROST GEMM launched from an autograd backward runs on the autograd
worker thread. When the GEMM is the first CUDA work on that thread, the thread has
no current CUDA context. Without a bind, the launch aborts the process
(``cudaErrorDeviceUninitialized - invalid device context``). ``swiglu_mlp``'s
x-only / Wg-only / Wu-only backward reached exactly this. The child runs in its
own process so the abort fails this test instead of killing the pytest worker."""

import subprocess
import sys

import pytest

from gemm_test_utils import requires_sm100

pytestmark = [pytest.mark.L0, requires_sm100]

_CHILD = r"""
import torch, cudnn
from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.tile_config import CATALOG

M = N = K = 256
torch.manual_seed(0)
a = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(1, K, N, device="cuda", dtype=torch.bfloat16)
g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
A = g.tensor(name="a", dim=[1, M, K], stride=[M * K, K, 1])
B = g.tensor(name="b", dim=[1, K, N], stride=[K * N, N, 1])
C = g.matmul(A=A, B=B, name="mm")
C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
cfg = next(
    c
    for c in CATALOG
    if (c.pipeline, c.cta_tile_m, c.cta_tile_n, c.cta_tile_k_bytes, c.cga_size_m, c.cga_size_n, c.cta_group) == ("sm100", 128, 128, 128, 1, 1, 1)
)
compiled = jit_from_cudnn_graph(g, config=cfg)
bd = compiled.binding
c = torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16)

def gemm():
    compiled({bd.a_operands[0]: a, bd.b_operands[0]: b, bd.outputs[0]: c}, stream=torch.cuda.current_stream().cuda_stream)

gemm()  # compile and first launch on the main thread
torch.cuda.synchronize()
c.zero_()

class LaunchInBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        gemm()  # the first CUDA work on the autograd worker
        return grad

x = torch.zeros(4, device="cuda", requires_grad=True)
LaunchInBackward.apply(x).sum().backward()
torch.cuda.synchronize()
torch.testing.assert_close(c.float(), a.float() @ b.float(), rtol=0.02, atol=0.1)
print("launched on the autograd worker")
"""


def test_compiled_gemm_launches_on_a_cold_autograd_worker():
    out = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, (out.returncode, out.stdout[-2000:], out.stderr[-3000:])
    assert "launched on the autograd worker" in out.stdout
