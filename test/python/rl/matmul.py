# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import cudnn


def build_matmul(a, b, bias=None):
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    x = graph.tensor_like(a)
    w = graph.tensor_like(b)
    y = graph.matmul(A=x, B=w)
    inputs = {x: a, w: b}
    if bias is not None:
        y.set_data_type(cudnn.data_type.FLOAT)
        v = graph.tensor_like(bias)
        inputs[v] = bias
        y = graph.add(a=y, b=v)
    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    graph.select_numeric_notes([cudnn.numerical_note.RL])
    graph.build()
    assert graph.selected_engine.name == "rl_gemm"
    work = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=a.device)

    def run(out=None):
        if out is None:
            out = torch.empty((*a.shape[:-1], b.shape[-1]), dtype=a.dtype, device=a.device)
        graph.execute({**inputs, y: out}, work)
        return {"out": out}

    return run
