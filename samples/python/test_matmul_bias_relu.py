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
    else:
        raise ValueError("Unsupported tensor data type.")
    
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
    plans = graph.get_execution_plan_list([cudnn.heur_mode.A])
    plans.check_support()
    graph.set_execution_plans(plans)

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    Y_actual = torch.zeros_like(Y_expected)

    graph.execute({X: X_gpu, W:  W_gpu, B:  B_gpu, Y:  Y_actual}, workspace)

    rtol = 1e-2 if input_type == torch.bfloat16 else 1e-3
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-3, rtol=rtol)
        
if __name__ == "__main__":
    test_matmul_bias_relu(((1,128,1600), torch.float16))