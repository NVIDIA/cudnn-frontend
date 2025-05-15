import cudnn
import pytest
import torch
from functools import lru_cache, wraps
import functools

from test_utils import torch_fork_set_rng


def conv_reduce_cache_key(handle, X_gpu, W_gpu):
    """Custom key function for conv_reduce_graph"""
    return (
        tuple(X_gpu.shape),
        tuple(X_gpu.stride()),
        tuple(W_gpu.shape),
        tuple(W_gpu.stride()),
        X_gpu.dtype,
        W_gpu.dtype,
    )


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.B])
@cudnn.graph_cache(key_fn=conv_reduce_cache_key)
def create_conv_reduce_graph(handle, X_gpu, W_gpu):
    with cudnn.graph(handle) as (g, _):
        print(
            f"Creating graph with X_gpu shape: {X_gpu.shape} and W_gpu shape: {W_gpu.shape}"
        )
        X = g.tensor_like(X_gpu)
        W = g.tensor_like(W_gpu)
        Y_conv = g.conv_fprop(X, W, padding=[1, 1], stride=[1, 1], dilation=[1, 1])
        Y = g.reduction(Y_conv, mode=cudnn.reduction_mode.ADD)
        n, _, h, w = X.get_dim()
        Y.set_output(True).set_dim([n, 1, h, w]).set_data_type(cudnn.data_type.FLOAT)
    return g, [X, W, Y]  # Return raw graph and tensors


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_reduction(cudnn_handle):

    # Define tensor dimensions
    N, K, C, H, W = 4, 32, 16, 64, 64
    R, S = 3, 3
    padding = stride = dilation = [1, 1]

    # Reference
    X_gpu = torch.randn(N, C, H, W, dtype=torch.float16, device="cuda").to(
        memory_format=torch.channels_last
    )
    W_gpu = torch.randn(K, C, R, S, dtype=torch.float16, device="cuda").to(
        memory_format=torch.channels_last
    )
    # Perform convolution using FP32 computation while input and filter remain in FP16
    with torch.cuda.amp.autocast(dtype=torch.float32):
        conv_output = torch.nn.functional.conv2d(
            X_gpu, W_gpu, padding=padding, stride=stride, dilation=dilation
        )
        Y_expected = conv_output.sum(dim=1)

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    g, uids = create_conv_reduce_graph(cudnn_handle, X_gpu, W_gpu)

    print(f"Graph created with UIDs: {uids}")

    X_uid, W_uid, Y_uid = uids

    X_gpu_2 = torch.randn(N, C, H, W, dtype=torch.float16, device="cuda").to(
        memory_format=torch.channels_last
    )
    W_gpu_2 = torch.randn(K, C, R, S, dtype=torch.float16, device="cuda").to(
        memory_format=torch.channels_last
    )

    g2, uids2 = create_conv_reduce_graph(cudnn_handle, X_gpu_2, W_gpu_2)

    X_uid2, W_uid2, Y_uid2 = uids2

    assert X_uid == X_uid2
    assert W_uid == W_uid2
    assert Y_uid == Y_uid2
    assert g == g2

    Y_actual = torch.zeros_like(Y_expected)
    Y_actual_2 = torch.zeros_like(Y_expected)

    workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)

    g.execute(
        {X_uid: X_gpu, W_uid: W_gpu, Y_uid: Y_actual}, workspace, handle=cudnn_handle
    )

    # g.execute(
    #     {X_uid: X_gpu_2, W_uid: W_gpu_2, Y_uid: Y_actual_2}, workspace, handle=cudnn_handle
    # )

    torch.cuda.synchronize()
    # Compare
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-3, rtol=1e-3)
