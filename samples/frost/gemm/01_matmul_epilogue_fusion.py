# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 01: matmul + fused pointwise epilogues (pure cuDNN frontend API).

Two cases: a single-output ``matmul -> relu`` chain, and a
``matmul -> bias (per-row) -> gelu_approx_tanh`` chain that also taps the raw
matmul output to its own buffer (multi-output: ``set_output(True)`` on an
intermediate tensor, per-output dtypes).
"""

from __future__ import annotations

import cudnn
import torch


def _build_plans(g) -> None:
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    g.select_plan(names.index("frost_gemm"))  # pin the FROST entry
    g.check_support()
    g.build_plans()


def _mkdata(M: int, N: int, K: int):
    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    return a, b


def _relu_case(M: int, N: int, K: int) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    Y = g.relu(input=C, name="r")
    Y.set_output(True)
    _build_plans(g)

    a, b = _mkdata(M, N, K)
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, Y: c}, workspace)
    torch.cuda.synchronize()

    ref = torch.relu(torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[01] PASS  matmul->relu                M={M} N={N} K={K}")


def _bias_gelu_tap_case(M: int, N: int, K: int) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    bias = g.tensor(name="bias", dim=[1, M, 1], stride=[M, 1, 1])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    Cb = g.bias(input=C, bias=bias, name="b")
    Y = g.gelu_approx_tanh(input=Cb, name="g")
    Y.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _build_plans(g)

    a, b = _mkdata(M, N, K)
    # Two GMEM outputs: terminal Y (FP32) + matmul tap C (BF16).
    c_term = torch.empty(1, M, N, dtype=torch.float32, device="cuda")
    c_tap = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    bias_t = torch.randn(1, M, 1, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, bias: bias_t, Y: c_term, C: c_tap}, workspace)
    torch.cuda.synchronize()

    mm = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref_term = torch.nn.functional.gelu(mm + bias_t.to(torch.float32), approximate="tanh")
    torch.testing.assert_close(c_term, ref_term, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(c_tap, mm.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    print(f"[01] PASS  matmul->bias->gelu (+ tap)  M={M} N={N} K={K}")


def main(M: int = 256, N: int = 256, K: int = 128) -> None:
    _relu_case(M, N, K)
    _bias_gelu_tap_case(M, N, K)


if __name__ == "__main__":
    main()
