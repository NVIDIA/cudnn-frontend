import cudnn
import itertools
import pytest
import torch
from looseversion import LooseVersion

from test_utils import torch_fork_set_rng


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.bool:
        return cudnn.data_type.BOOLEAN
    elif torch_type == torch.uint8:
        return cudnn.data_type.UINT8
    elif torch_type == torch.int8:
        return cudnn.data_type.INT8
    else:
        raise ValueError("Unsupported tensor data type.")


def get_cc():
    (major, minor) = torch.cuda.get_device_capability()
    return major * 10 + minor


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "8.9.6",
    reason="requires cudnn 8.9.6 or higher",
)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 9, reason="requires Hopper or newer arch"
)
@torch_fork_set_rng(seed=0)
def test_int8_bf16_matmul():

    # matmul problem size
    B, M, N, K = 16, 32, 64, 128

    # Initialize input tensors
    A_gpu = (
        torch.randint(
            3, (B, M, K), requires_grad=False, device="cuda", dtype=torch.int8
        )
        - 2
    )
    B_gpu = (
        3
        * torch.randn(B, K, N, requires_grad=False, device="cuda", dtype=torch.bfloat16)
        - 1.25
    )

    handle = cudnn.create_handle()
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=handle, stream=stream)

    # Make cudnn graph
    graph = cudnn.pygraph(handle=handle)

    # Create the two non-virtual input tensors A and B.
    # There are read from global memory.
    A = graph.tensor_like(A_gpu)
    B = graph.tensor_like(B_gpu)

    # Cast the input tensors to required mma precision
    A_casted = graph.identity(input=A, compute_data_type=cudnn.data_type.FLOAT)
    A_casted.set_data_type(cudnn.data_type.BFLOAT16)

    C = graph.matmul(
        name="matmul", A=A_casted, B=B, compute_data_type=cudnn.data_type.FLOAT
    )
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    graph.build([cudnn.heur_mode.A])

    # Run pyt reference
    C_expected = torch.matmul(A_gpu.to(torch.bfloat16), B_gpu.to(torch.bfloat16))

    # Run cudnn graph
    C_actual = torch.zeros_like(C_expected)
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute({A: A_gpu, B: B_gpu, C: C_actual}, workspace, handle=handle)

    torch.cuda.synchronize()
    # compare'em
    torch.testing.assert_close(C_expected, C_actual)
    cudnn.destroy_handle(handle)


A_data_type_options = [torch.int8, torch.bfloat16, torch.float16]
B_data_type_options = [torch.int8, torch.bfloat16, torch.float16]
MMA_data_type_options = [torch.bfloat16, torch.float16, torch.float32]


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "8.9.6",
    reason="requires cudnn 8.9.6 or higher",
)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 9, reason="requires Hopper or newer arch"
)
@pytest.mark.parametrize("A_data_type", A_data_type_options)
@pytest.mark.parametrize("B_data_type", B_data_type_options)
@pytest.mark.parametrize("MMA_data_type", MMA_data_type_options)
@torch_fork_set_rng(seed=0)
def test_mixed_precision_matmul(A_data_type, B_data_type, MMA_data_type):

    # matmul problem size
    B, M, N, K = 16, 32, 64, 128

    # Initialize input tensors
    if A_data_type != torch.int8:
        A_gpu = (
            2
            * torch.randn(
                B, M, K, requires_grad=False, device="cuda", dtype=A_data_type
            )
            - 0.5
        )
    else:
        A_gpu = (
            torch.randint(
                4, (B, M, K), requires_grad=False, device="cuda", dtype=A_data_type
            )
            - 1
        )

    if B_data_type != torch.int8:
        B_gpu_strided = (
            3
            * torch.randn(
                B, K, N, requires_grad=False, device="cuda", dtype=B_data_type
            )
            - 1.25
        )
    else:
        B_gpu_strided = (
            torch.randint(
                3, (B, K, N), requires_grad=False, device="cuda", dtype=B_data_type
            ).contiguous()
            - 2
        )

    B_gpu = torch.as_strided(B_gpu_strided, (B, K, N), (N * K, 1, N))

    handle = cudnn.create_handle()
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=handle, stream=stream)

    # Make cudnn graph
    graph = cudnn.pygraph(handle=handle)

    # Create the two non-virtual input tensors A and B.
    # There are read from global memory.
    A = graph.tensor_like(A_gpu)
    B = graph.tensor_like(B_gpu)

    # Cast the input tensors to required mma precision
    A_casted = graph.identity(input=A, compute_data_type=cudnn.data_type.FLOAT)
    A_casted.set_data_type(convert_to_cudnn_type(MMA_data_type))

    # Casting input tensor B is only supported from cudnn v9
    if (
        B_data_type != MMA_data_type
        and LooseVersion(cudnn.backend_version_string()) < "9"
    ):
        pytest.skip("mixed precision on B only supported from cudnn v9.")

    if LooseVersion(cudnn.backend_version_string()) < "9":
        # Do not create a cast node
        B_casted = B
    else:
        # Cast the input tensors to required mma precision
        B_casted = graph.identity(input=B, compute_data_type=cudnn.data_type.FLOAT)
        B_casted.set_data_type(convert_to_cudnn_type(MMA_data_type))

    # CAUTION: Hardcodes to fp32 as tests today dont cover inputs that are casted to ints.
    # In case your usecase does cast inputs to int8, use int32 as compute type here.
    C = graph.matmul(
        name="matmul", A=A_casted, B=B_casted, compute_data_type=cudnn.data_type.FLOAT
    )
    C.set_output(True).set_data_type(convert_to_cudnn_type(MMA_data_type))

    graph.build([cudnn.heur_mode.A])

    # Run pyt reference
    C_expected = torch.matmul(A_gpu.to(MMA_data_type), B_gpu.to(MMA_data_type))

    # Run cudnn graph
    C_actual = torch.zeros_like(C_expected)
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute({A: A_gpu, B: B_gpu, C: C_actual}, workspace, handle=handle)

    torch.cuda.synchronize()
    # compare'em
    torch.testing.assert_close(C_expected, C_actual, atol=1e-4, rtol=1e-4)
    cudnn.destroy_handle(handle)


problem_size_options = [(1, 128, 768), (16, 512, 1600), (1, 128, 1024)]
input_type_options = [torch.bfloat16, torch.float16]

all_options = [
    elem for elem in itertools.product(*[problem_size_options, input_type_options])
]


@pytest.fixture(params=all_options)
def param_extract(request):
    return request.param


@torch_fork_set_rng(seed=0)
def test_matmul_bias_relu(param_extract):

    problem_size_options, input_type = param_extract
    b, s, e = problem_size_options

    if b > 1 and LooseVersion(cudnn.backend_version_string()) < "8.9.6":
        pytest.skip("matmul broadcast only supported 8.9.6 onwards.")

    # Regression in cudnn backend where ampere does not support matmul broadcast starting 8.9.6
    if b > 1 and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("matmul broadcast on ampere with 8.9.6 is not supported.")

    X_gpu = torch.randn(b, s, e, requires_grad=False, device="cuda", dtype=input_type)
    W_gpu = torch.randn(
        1, e, e * 4, requires_grad=False, device="cuda", dtype=input_type
    )
    B_gpu = torch.randn(
        1, 1, e * 4, requires_grad=False, device="cuda", dtype=input_type
    )
    Y_expected = torch.nn.functional.linear(
        X_gpu, W_gpu.squeeze().T, bias=B_gpu.squeeze()
    )

    handle = cudnn.create_handle()
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=handle, stream=stream)

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )

    X = graph.tensor(
        name="X",
        dim=X_gpu.size(),
        stride=X_gpu.stride(),
        data_type=convert_to_cudnn_type(input_type),
    )
    W = graph.tensor(
        name="W",
        dim=W_gpu.size(),
        stride=W_gpu.stride(),
        data_type=convert_to_cudnn_type(input_type),
    )
    B = graph.tensor(
        name="B",
        dim=B_gpu.size(),
        stride=B_gpu.stride(),
        data_type=convert_to_cudnn_type(input_type),
    )

    response = graph.matmul(name="matmul", A=X, B=W)
    Y = graph.bias(name="bias", input=response, bias=B)
    Y.set_output(True).set_data_type(convert_to_cudnn_type(input_type))

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    Y_actual = torch.zeros_like(Y_expected)

    graph.execute({X: X_gpu, W: W_gpu, B: B_gpu, Y: Y_actual}, workspace, handle=handle)

    atol = 0.0625 if get_cc() == 89 else 1e-3
    rtol = 1e-2 if input_type == torch.bfloat16 else 1e-3

    torch.cuda.synchronize()

    torch.testing.assert_close(Y_expected, Y_actual, atol=atol, rtol=rtol)

    cudnn.destroy_handle(handle)


if __name__ == "__main__":
    test_matmul_bias_relu(((1, 128, 1600), torch.float16))
