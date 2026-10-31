# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join row axes through a logical reshape without materializing activated storage.

Run with CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1. The reshape source dimensions
are fixed by the prepared graph; fresh storage can be rebound without copying.
No performance claim is made by this correctness example.
"""

import warnings
import cudnn
import torch


def build_graph():
    """Build [B,H,S,K] -> activation -> [B,H*S,K] -> GEMM."""
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = graph.tensor(name="X", dim=[2, 2, 64, 64], stride=[8192, 4096, 64, 1])
    w = graph.tensor(name="W", dim=[2, 64, 128], stride=[8192, 128, 1])
    value = graph.leaky_relu(input=x, negative_slope=0.125).set_data_type(cudnn.data_type.BFLOAT16)
    value = graph.reshape(input=value, reshape_mode=cudnn.reshape_mode.LOGICAL).set_dim([2, 128, 64]).set_stride([8192, 64, 1])
    y = graph.matmul(A=value, B=w).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return graph, x, w, y


def main():
    """Execute the public graph API and verify changed-input capture replay."""
    if not torch.cuda.is_available():
        print("[09] SKIP: CUDA is required")
        return
    major, minor = torch.cuda.get_device_capability()
    if not 100 <= major * 10 + minor < 120:
        print("[09] SKIP: full GEMM requires data-center Blackwell")
        return
    graph, x, w, y = build_graph()
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    candidates = [i for i in range(len(graph.plans)) if graph.get_plan_name_at_index(i).split("[")[0] == "frost_gemm"]
    if not candidates:
        print("[09] SKIP: enable FROST engines and check GPU/DSL support")
        return
    graph.select_plan(candidates[0])
    graph.check_support()
    graph.build_plans()
    xv = torch.randint(-3, 4, (2, 2, 64, 64), device="cuda").bfloat16()
    wv = torch.randint(-3, 4, (2, 64, 128), device="cuda").bfloat16()
    out = torch.empty((2, 128, 128), device="cuda", dtype=torch.float32)
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    pack = {x: xv, w: wv, y: out}
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
        for _ in range(2):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                xv.neg_()
                out.fill_(float("nan"))
                captured.replay()
            stream.synchronize()
            activated = torch.nn.functional.leaky_relu(xv.float(), 0.125).bfloat16().view(2, 128, 64)
            torch.testing.assert_close(out, activated.float() @ wv.float(), atol=0, rtol=0)
    finally:
        stream.synchronize()
        captured.reset()
        cudnn.destroy_handle(handle)
    print("[09] PASS: logical reshape mainloop fusion and changed-input replay")


if __name__ == "__main__":
    main()
