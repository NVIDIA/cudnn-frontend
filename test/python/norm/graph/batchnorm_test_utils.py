# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared graph lifecycle helpers for BatchNorm tests."""

from functools import wraps

import cudnn
import pytest
import torch

_HEURISTICS = [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]


def preserve_handle_stream(test):
    @wraps(test)
    def wrapped(*, cudnn_handle, **kwargs):
        original_stream = cudnn.get_stream(cudnn_handle)
        try:
            return test(cudnn_handle=cudnn_handle, **kwargs)
        finally:
            cudnn.set_stream(handle=cudnn_handle, stream=original_stream)

    return wrapped


def torch_to_cudnn_data_type(dtype: torch.dtype):
    dtype_map = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.int32: cudnn.data_type.INT32,
        torch.int64: cudnn.data_type.INT64,
    }
    if hasattr(torch, "float8_e4m3fn"):
        dtype_map[torch.float8_e4m3fn] = cudnn.data_type.FP8_E4M3
    if hasattr(torch, "float8_e5m2"):
        dtype_map[torch.float8_e5m2] = cudnn.data_type.FP8_E5M2

    try:
        return dtype_map[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported tensor data type: {dtype}") from error


def new_batchnorm_graph(cudnn_handle, io_data_type: torch.dtype = torch.float16):
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)
    return cudnn.pygraph(
        io_data_type=torch_to_cudnn_data_type(io_data_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )


def _prepare_graph(graph) -> None:
    graph.validate()
    graph.build_operation_graph()


def _create_and_check_plans(graph) -> None:
    graph.create_execution_plans(_HEURISTICS)
    graph.check_support()


def finalize_graph(graph) -> None:
    _prepare_graph(graph)
    _create_and_check_plans(graph)
    graph.build_plans()


def finalize_graph_or_skip(graph) -> None:
    _prepare_graph(graph)
    try:
        _create_and_check_plans(graph)
    except cudnn.cudnnGraphNotSupportedError as error:
        print(f"TEST WAIVED: unsupported graph. {error}")
        pytest.skip("TEST WAIVED: unsupported graph.")
    graph.build_plans()


def execute_graph(graph, variant_pack, cudnn_handle) -> None:
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
