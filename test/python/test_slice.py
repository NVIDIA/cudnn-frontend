import cudnn
import itertools
import pytest
import torch
from looseversion import LooseVersion

from test_utils import torch_fork_set_rng


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 9, reason="requires Hopper or newer arch"
)
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_int8_bf16_matmul_slice(cudnn_handle):

    # matmul problem size
    Batch, M, N, K = 16, 32, 64, 128
    slice_B = slice(8, None)
    slice_M = slice(16, None)
    slice_N = slice(32, None)
    slice_K = slice(None, None)

    # Initialize input tensors
    A_gpu = (
        2
        * torch.randn(
            Batch, M, K, requires_grad=False, device="cuda", dtype=torch.bfloat16
        )
        - 0.25
    )
    A_slice_gpu = A_gpu[slice_B, slice_M, :]

    B_gpu = (
        3
        * torch.randn(
            Batch, K, N, requires_grad=False, device="cuda", dtype=torch.bfloat16
        )
        - 1.25
    )
    B_slice_gpu = B_gpu[slice_B, :, slice_N]

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    # Make cudnn graph
    graph = cudnn.pygraph(handle=cudnn_handle)

    # Create the two non-virtual input tensors A and B.
    # There are read from global memory.
    A = graph.tensor_like(A_gpu)
    A_slice = graph.slice(A, [slice_B, slice_M, slice_K], name="A_slice")
    B = graph.tensor_like(B_gpu)
    B_slice = graph.slice(B, [slice_B, slice_K, slice_N], name="B_slice")

    C = graph.matmul(
        name="matmul", A=A_slice, B=B_slice, compute_data_type=cudnn.data_type.FLOAT
    )
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    graph.build([cudnn.heur_mode.A])

    # Run pyt reference
    C_expected = torch.matmul(
        A_slice_gpu.to(torch.bfloat16), B_slice_gpu.to(torch.bfloat16)
    )

    # Run cudnn graph
    C_actual = torch.zeros_like(C_expected)
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute({A: A_gpu, B: B_gpu, C: C_actual}, workspace, handle=cudnn_handle)
    print(A_gpu.data_ptr())
    torch.cuda.synchronize()
    # compare'em
    torch.testing.assert_close(C_expected, C_actual)
