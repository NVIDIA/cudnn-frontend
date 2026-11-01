# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-N GEMM + all logical axis indices, in one FROST epilogue.

Y[b,h,m,n] = (A @ B)[b,h,m,n] + b + h + m + n.
The kernel folds [B,H] into one batch coordinate; gen_index keeps graph axes.
The batch prefix is fixed at preparation; this is not dynamic reshape support.
"""

import warnings
from math import prod

import cudnn
import torch


def build_graph(prefix=(2, 3)):
    """Build rank-2, rank-3 or higher-rank graphs through the same public API."""
    shape = (*prefix, 128, 128)
    stride = [prod(shape[i + 1 :]) for i in range(len(shape))]
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    a = graph.tensor(name="A", dim=list(shape), stride=stride)
    b = graph.tensor(name="B", dim=list(shape), stride=stride)
    value = graph.matmul(A=a, B=b)
    for axis in range(len(shape)):
        value = graph.add(a=value, b=graph.gen_index(input=value, axis=axis))
    y = value.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return graph, a, b, y


def main():
    """Execute and check nondefault-stream changed-input CUDA graph replay."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11, 12):
        print("[10] SKIP: full FROST GEMM requires Blackwell")
        return
    graph, a, b, y = build_graph()
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    candidates = [i for i in range(len(graph.plans)) if graph.get_plan_name_at_index(i).split("[")[0] == "frost_gemm"]
    if not candidates:
        print("[10] SKIP: enable FROST engines and check GPU/DSL support")
        return
    graph.select_plan(candidates[0])
    graph.check_support()
    graph.build_plans()
    shape = tuple(y.get_dim())
    av = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    bv = torch.zeros_like(av)
    out = torch.empty(shape, device="cuda", dtype=torch.float32)
    indices = torch.zeros_like(out)
    for axis, extent in enumerate(shape):
        view = [1] * len(shape)
        view[axis] = extent
        indices += torch.arange(extent, device="cuda", dtype=torch.float32).reshape(view)
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    pack = {a: av, b: bv, y: out}
    stream, captured, handle = torch.cuda.Stream(), torch.cuda.CUDAGraph(), cudnn.create_handle()
    try:
        stream.wait_stream(torch.cuda.current_stream())
        cudnn.set_stream(handle, stream.cuda_stream)
        with torch.cuda.stream(stream):
            graph.execute(pack, workspace, handle=handle)
        stream.synchronize()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
            with torch.cuda.graph(captured, stream=stream):
                graph.execute(pack, workspace, handle=handle)
        for fill in (0.0, 0.0625):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                av.fill_(fill)
                bv.fill_(fill)
                out.fill_(float("nan"))
                captured.replay()
            stream.synchronize()
            torch.testing.assert_close(out, indices + shape[-1] * fill * fill, atol=0, rtol=0)
    finally:
        stream.synchronize()
        captured.reset()
        cudnn.destroy_handle(handle)
    print("[10] PASS: logical batch/matrix indices and changed-input replay")


if __name__ == "__main__":
    main()
