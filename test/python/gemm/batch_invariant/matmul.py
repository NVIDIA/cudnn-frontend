# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cudnn
from cudnn.engines import is_backend_engine


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
    graph.select_numeric_notes([cudnn.numerical_note.BATCH_INVARIANT])
    graph.build()
    assert graph.selected_engine.name == "batch_invariant_gemm"
    work = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=a.device)

    def run(out=None):
        if out is None:
            out = torch.empty((*a.shape[:-1], b.shape[-1]), dtype=a.dtype, device=a.device)
        graph.execute({**inputs, y: out}, work)
        return {"out": out}

    return run


def execute_native(graph, bindings, device):
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    declined = []
    for index, plan in enumerate(graph.plans):
        if not is_backend_engine(plan.engine_id):
            continue
        try:
            graph.build_plan_at_index(index)
        except cudnn.cudnnGraphNotSupportedError as exc:
            declined.append((plan.engine_id, str(exc)))
            continue
        assert graph.selected_engine is None
        workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=device)
        graph.execute(bindings, workspace)
        return
    pytest.fail(f"No native cuDNN plan built. Declined backend plans: {declined}")


def native_matmul(a, b):
    graph = cudnn.pygraph(compute_data_type=cudnn.data_type.FLOAT)
    left, right = graph.tensor_like(a), graph.tensor_like(b)
    product = graph.matmul(A=left, B=right).set_output(True).set_data_type(cudnn.data_type.BFLOAT16 if a.dtype == torch.bfloat16 else cudnn.data_type.HALF)
    out = a.new_empty((a.shape[0], a.shape[1], b.shape[2]))
    execute_native(graph, {left: a, right: b, product: out}, a.device)
    return out
