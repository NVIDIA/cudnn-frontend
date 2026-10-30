# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 08: transpose-aware operand fusion, without a transpose kernel.

Graph: transpose(leaky_relu(X[1,K,M])) @ leaky_relu(transpose(W[1,N,K])).
FROST absorbs rank-3 M/N transposes into input views and retains each activation's
BF16 rounding. Runtime storage stays in the graph-input coordinates; no caller
transpose, contiguous conversion or intermediate allocation is required.

Run on data-center Blackwell:
    CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 python samples/frost/gemm/08_matmul_transpose_mainloop.py
"""

import argparse
import warnings

import cudnn
import torch


def _build_graph(M, N, K):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = g.tensor(name="X", dim=[1, K, M], stride=[K * M, 1, K])
    w = g.tensor(name="W", dim=[1, N, K], stride=[N * K, K, 1])
    a = g.leaky_relu(input=x, negative_slope=0.125).set_data_type(cudnn.data_type.BFLOAT16)
    a = g.transpose(input=a, permutation=[0, 2, 1])
    b = g.transpose(input=w, permutation=[0, 2, 1])
    b = g.leaky_relu(input=b, negative_slope=0.125).set_data_type(cudnn.data_type.BFLOAT16)
    y = g.matmul(A=a, B=b).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return g, x, w, y


def _build_plans(g):
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    frost = [i for i in range(len(g.plans)) if g.get_plan_name_at_index(i).split("[")[0] == "frost_gemm"]
    if not frost:
        raise RuntimeError("No FROST GEMM plan; set CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 and check GPU/DSL support")
    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()


def main(M=128, N=192, K=136):
    if min(M, N, K) <= 0 or K % 8:
        raise ValueError("M/N/K must be positive and K must be divisible by 8 (BF16 TMA alignment)")
    if not torch.cuda.is_available():
        print("[08] SKIP  needs a CUDA GPU")
        return
    major, minor = torch.cuda.get_device_capability()
    if not 100 <= major * 10 + minor < 120:
        print("[08] SKIP  needs a data-center Blackwell-family GPU (100 <= SM < 120)")
        return
    g, x, w, y = _build_graph(M, N, K)
    _build_plans(g)
    torch.manual_seed(0)
    # X's strided layout is declared by the graph; create that storage once, not during execute.
    xv = torch.empty_strided((1, K, M), (K * M, 1, K), device="cuda", dtype=torch.bfloat16)
    xv.copy_(torch.randint(-3, 4, xv.shape, device="cuda"))
    wv = torch.randint(-3, 4, (1, N, K), device="cuda").to(torch.bfloat16)
    out = torch.empty(1, M, N, device="cuda", dtype=torch.float32)
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    pack = {x: xv, w: wv, y: out}
    stream, captured, handle = torch.cuda.Stream(), torch.cuda.CUDAGraph(), cudnn.create_handle()
    try:
        stream.wait_stream(torch.cuda.current_stream())
        cudnn.set_stream(handle, stream.cuda_stream)
        with torch.cuda.stream(stream):
            g.execute(pack, workspace, handle=handle)
        stream.synchronize()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
            with torch.cuda.graph(captured, stream=stream):
                g.execute(pack, workspace, handle=handle)
        for _ in range(2):
            xv.neg_()
            out.fill_(float("nan"))
            captured.replay()
            a = torch.nn.functional.leaky_relu(xv.float(), 0.125).bfloat16().transpose(1, 2)
            b = torch.nn.functional.leaky_relu(wv.float().transpose(1, 2), 0.125).bfloat16()
            torch.testing.assert_close(out, a.float() @ b.float(), atol=0, rtol=0)
    finally:
        captured.reset()
        cudnn.destroy_handle(handle)
    print(f"[08] PASS  transpose-aware mainloop  M={M} N={N} K={K}; changed-input capture replay")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--k", type=int, default=136)
    args = parser.parse_args()
    main(args.m, args.n, args.k)
