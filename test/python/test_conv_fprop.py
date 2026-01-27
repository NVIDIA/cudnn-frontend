"""
Test for conv fprop using tvm-ffi based execute API.

This test validates that the TVM-FFI migration for PyGraph::execute
works correctly by running a simple convolution forward pass.
"""

import cudnn
import pytest
import torch
from test_utils import torch_fork_set_rng


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_conv_fprop_tvm_ffi(cudnn_handle):
    """Test conv fprop using the tvm-ffi based execute API."""
    # Setup tensors
    X_gpu = torch.randn(4, 16, 32, 32, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    W_gpu = torch.randn(32, 16, 3, 3, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    padding = [1, 1]
    stride = [1, 1]
    dilation = [1, 1]

    # Reference result using PyTorch
    Y_expected = torch.nn.functional.conv2d(X_gpu, W_gpu, padding=padding, stride=stride, dilation=dilation)

    # Set stream
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    # Build cudnn graph
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    X = graph.tensor_like(X_gpu)
    W = graph.tensor_like(W_gpu)

    conv_output = graph.conv_fprop(image=X, weight=W, padding=padding, stride=stride, dilation=dilation)
    conv_output.set_output(True)

    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

    Y_actual = torch.zeros_like(Y_expected)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    # Execute using the tvm-ffi based execute API
    graph.execute(
        {X: X_gpu, W: W_gpu, conv_output: Y_actual},
        workspace,
        handle=cudnn_handle,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-3, rtol=1e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_conv_fprop_execute_plan_at_index_tvm_ffi(cudnn_handle):
    """Test conv fprop using execute_plan_at_index with tvm-ffi."""
    # Setup tensors
    X_gpu = torch.randn(2, 8, 16, 16, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    W_gpu = torch.randn(16, 8, 3, 3, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    padding = [1, 1]
    stride = [1, 1]
    dilation = [1, 1]

    # Reference result using PyTorch
    Y_expected = torch.nn.functional.conv2d(X_gpu, W_gpu, padding=padding, stride=stride, dilation=dilation)

    # Set stream
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    # Build cudnn graph
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    X = graph.tensor_like(X_gpu)
    W = graph.tensor_like(W_gpu)

    conv_output = graph.conv_fprop(image=X, weight=W, padding=padding, stride=stride, dilation=dilation)
    conv_output.set_output(True)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    Y_actual = torch.zeros_like(Y_expected)
    workspace = torch.empty(graph.get_workspace_size_plan_at_index(0), device="cuda", dtype=torch.uint8)

    # Execute using execute_plan_at_index with tvm-ffi
    graph.execute_plan_at_index(
        {X: X_gpu, W: W_gpu, conv_output: Y_actual},
        workspace,
        index=0,
        handle=cudnn_handle,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-3, rtol=1e-3)
