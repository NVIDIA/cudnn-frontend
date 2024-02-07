import cudnn
import itertools
import pytest
import torch

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
    return major*10 + minor 

def compare_tensors(expected, actual, name, rtol=2e-2, atol=2e-2, fudge=1e-9):
    assert expected.shape == actual.shape

    expected = expected.float().cuda().flatten()
    actual = actual.float().cuda().flatten()

    n_elem = torch.numel(expected)

    mae = (expected - actual).abs().mean().item()
    perr = ((expected - actual).abs().sum() / expected.abs().sum()).item()
    snr = (expected**2).mean().sqrt() / ((expected - actual) ** 2).mean().sqrt()
    snr_db = (10 * torch.log10(snr)).item()

    absolute_error = (expected - actual).abs()
    relative_error = absolute_error / torch.where(expected.abs() < fudge, fudge, expected.abs())

    abs_error_indices = absolute_error > atol
    rel_error_indices = relative_error > rtol
    n_abs_errors = torch.sum(abs_error_indices)
    n_rel_errors = torch.sum(rel_error_indices)
    error_indices = torch.logical_and(abs_error_indices, rel_error_indices)
    n_errors = torch.sum(error_indices)

    n_nans = torch.isnan(actual).sum()
    n_zeros = n_elem - torch.count_nonzero(actual)

    if n_errors != 0:
        print(f"========== Comparison for {name} ==========")
        print(f"Absolute Tolerance = {atol}")
        print(f"Relative Tolerance = {rtol}")
        print(f"Number of elements = {n_elem}")
        print(f"Number of absolute errors = {n_abs_errors} ({n_abs_errors * 100 / n_elem:.2f}%)")
        print(f"Number of relative errors = {n_rel_errors} ({n_rel_errors * 100 / n_elem:.2f}%)")
        print(f"Number of errors (absolute and relative) = {n_errors} ({(n_errors * 100)/n_elem:.2f}%)")
        print(f"Maximum absolute error = {absolute_error.max():.4f}")
        print(f"Maximum relative error = {relative_error.max():.4f}")
        print(f"Mean average error = {mae:.4f}")
        print(f"Perr error = {perr:.4f} = 1/{(1/perr) if perr != 0 else float('inf'):.2f}")
        print(f"Signal to noise ratio = {snr.item():.2f} = {snr_db:.2f}dB")
        print(f"Number of Nans = {n_nans} ({n_nans * 100 / n_elem:.2f}%)")
        print(f"Number of Zeros = {n_zeros} ({n_zeros * 100 / n_elem:.2f}%)")
        print("===================================\n")

    return n_errors

@pytest.mark.skipif(cudnn.backend_version() < 8906, reason="requires cudnn 8.9.6 or higher")
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="requires Hopper or newer arch")
def test_int8_bf16_matmul():
    # matmul problem size 
    B, M, N, K = 16, 32, 64, 128

    # Initialize input tensors
    A_gpu = torch.randint(3, (B, M, K), requires_grad=False, device="cuda", dtype=torch.int8) - 2
    B_gpu = 3 * torch.randn(B, K, N, requires_grad=False, device="cuda", dtype=torch.bfloat16) - 1.25
    
    # Make cudnn graph
    graph = cudnn.pygraph()

    # Create the two non-virtual input tensors A and B.
    # There are read from global memory.
    A = graph.tensor_like(A_gpu)
    B = graph.tensor_like(B_gpu)
    
    # Cast the input tensors to required mma precision
    A_casted = graph.identity(input = A, compute_data_type=cudnn.data_type.FLOAT)
    A_casted.set_data_type(cudnn.data_type.BFLOAT16)
    
    C = graph.matmul(name = "matmul", A = A_casted, B = B, compute_data_type=cudnn.data_type.FLOAT)
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    
    graph.build([cudnn.heur_mode.A])

    # Run pyt reference
    C_expected = torch.matmul(A_gpu.to(torch.bfloat16), B_gpu.to(torch.bfloat16))
    
    # Run cudnn graph
    C_actual = torch.zeros_like(C_expected)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute({A: A_gpu, B:  B_gpu, C:  C_actual}, workspace)

    # compare'em
    torch.testing.assert_close(C_expected, C_actual)

A_data_type_options = [torch.int8, torch.bfloat16, torch.float16]
B_data_type_options = [torch.int8, torch.bfloat16, torch.float16]
MMA_data_type_options = [torch.bfloat16, torch.float16, torch.float32]

@pytest.mark.skipif(cudnn.backend_version() < 8906, reason="requires cudnn 8.9.6 or higher")
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="requires Hopper or newer arch")
@pytest.mark.parametrize("A_data_type", A_data_type_options)
@pytest.mark.parametrize("B_data_type", B_data_type_options)
@pytest.mark.parametrize("MMA_data_type", MMA_data_type_options)
def test_mixed_precision_matmul(A_data_type, B_data_type, MMA_data_type):

    # matmul problem size 
    B, M, N, K = 16, 32, 64, 128

    # Initialize input tensors
    if A_data_type != torch.int8:
        A_gpu = 2 * torch.randn(B, M, K, requires_grad=False, device="cuda", dtype=A_data_type) - 0.5
    else:
        A_gpu = torch.randint(4, (B, M, K), requires_grad=False, device="cuda", dtype=A_data_type) - 1

    if B_data_type != torch.int8:
        B_gpu_strided = 3 * torch.randn(B, K, N, requires_grad=False, device="cuda", dtype=B_data_type) - 1.25
    else:
        B_gpu_strided = torch.randint(3, (B, K, N), requires_grad=False, device="cuda", dtype=B_data_type).contiguous() - 2
    
    B_gpu = torch.as_strided(B_gpu_strided, (B, K, N), (N*K, 1, N))
    
    # Make cudnn graph
    graph = cudnn.pygraph()

    # Create the two non-virtual input tensors A and B.
    # There are read from global memory.
    A = graph.tensor_like(A_gpu)
    B = graph.tensor_like(B_gpu)
    
    # Cast the input tensors to required mma precision
    A_casted = graph.identity(input = A, compute_data_type=convert_to_cudnn_type(MMA_data_type))
    A_casted.set_data_type(convert_to_cudnn_type(MMA_data_type))
    
    # Casting input tensor B is only supported from cudnn v9
    if B_data_type != MMA_data_type and cudnn.backend_version() < 90000:
        pytest.skip("mixed precision on B only supported from cudnn v9.")
    
    if cudnn.backend_version() < 90000:
        # Do not create a cast node
        B_casted = B
    else:
        B_casted = graph.identity(input = B, compute_data_type=convert_to_cudnn_type(MMA_data_type))
        B_casted.set_data_type(convert_to_cudnn_type(MMA_data_type))

    C = graph.matmul(name = "matmul", A = A_casted, B = B_casted, compute_data_type=cudnn.data_type.FLOAT)
    C.set_output(True).set_data_type(convert_to_cudnn_type(MMA_data_type))
    
    graph.build([cudnn.heur_mode.A])

    # Run pyt reference
    C_expected = torch.matmul(A_gpu.to(MMA_data_type), B_gpu.to(MMA_data_type))
    
    # Run cudnn graph
    C_actual = torch.zeros_like(C_expected)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute({A: A_gpu, B:  B_gpu, C:  C_actual}, workspace)

    # compare'em
    compare_tensors(C_expected, C_actual, "output", atol=1e-4, rtol=1e-4)

problem_size_options = [(1, 128, 768)
                        , (16, 512, 1600)
                        , (1, 128, 1024)]
input_type_options = [torch.bfloat16, torch.float16]

all_options = [elem for elem in itertools.product(*[problem_size_options, input_type_options])]

@pytest.fixture(params=all_options)
def param_extract(request):
  return request.param

def test_matmul_bias_relu(param_extract):
    problem_size_options, input_type = param_extract
    b, s, e = problem_size_options

    if b > 1 and cudnn.backend_version() < 8906:
        pytest.skip("matmul broadcast only supported 8.9.6 onwards.")

    # Regression in cudnn backend where ampere does not support matmul broadcast starting 8.9.6
    if b > 1 and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("matmul broadcast on ampere with 8.9.6 is not supported.")
        
    X_gpu = torch.randn(b,s,e, requires_grad=False, device="cuda", dtype=input_type)
    W_gpu = torch.randn(1,e,e*4, requires_grad=False, device="cuda", dtype=input_type)
    B_gpu = torch.randn(1,1,e*4, requires_grad=False, device="cuda", dtype=input_type)
    Y_expected = torch.nn.functional.linear(X_gpu, W_gpu.squeeze().T, bias=B_gpu.squeeze())

    graph = cudnn.pygraph(intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    X = graph.tensor(name = "X", dim = X_gpu.size(), stride = X_gpu.stride(), data_type = convert_to_cudnn_type(input_type))
    W = graph.tensor(name = "W", dim = W_gpu.size(), stride = W_gpu.stride(), data_type = convert_to_cudnn_type(input_type))
    B = graph.tensor(name = "B", dim = B_gpu.size(), stride = B_gpu.stride(), data_type = convert_to_cudnn_type(input_type))

    response = graph.matmul(name = "matmul", A = X, B = W)
    Y = graph.bias(name = "bias", input = response, bias = B)
    Y.set_output(True).set_data_type(convert_to_cudnn_type(input_type))
    
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    Y_actual = torch.zeros_like(Y_expected)

    graph.execute({X: X_gpu, W:  W_gpu, B:  B_gpu, Y:  Y_actual}, workspace)

    atol = 0.0625 if get_cc() == 89 else 1e-3
    rtol = 1e-2 if input_type == torch.bfloat16 else 1e-3
    torch.testing.assert_close(Y_expected, Y_actual, atol=atol, rtol=rtol)
        
if __name__ == "__main__":
    test_matmul_bias_relu(((1,128,1600), torch.float16))
