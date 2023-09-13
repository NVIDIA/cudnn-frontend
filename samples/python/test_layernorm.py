import cudnn
import pytest
import torch
import itertools

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


embedding_dim_options = [768, 1024, 1280, 1600]
input_type_options = [torch.bfloat16, torch.float16]

all_options = [elem for elem in itertools.product(*[embedding_dim_options, input_type_options])]

@pytest.fixture(params=all_options)
def param_extract(request):
  return request.param

@pytest.mark.skipif(cudnn.backend_version() < 8905, reason="LN not supported below cudnn 8.9.5")
def test_ln(param_extract):
    embedding_dim, input_type = param_extract
    
    batch_size, seq_size = 16, 128
    N,C,H,W = batch_size * seq_size, embedding_dim, 1, 1
    
    epsilon_value = 1e-3

    x_gpu = torch.randn(N, C, H, W, device="cuda", dtype=input_type).to(memory_format=torch.channels_last)
    scale_gpu = torch.randn(1, C, H, W, requires_grad=False, device="cuda", dtype=input_type).to(memory_format=torch.channels_last)
    bias_gpu = torch.randn(1, C, H, W, requires_grad=False, device="cuda", dtype=input_type).to(memory_format=torch.channels_last)
    epsilon_cpu = torch.full((1, 1, 1, 1), epsilon_value, requires_grad=False, device="cpu", dtype=torch.float32)

    print("Running reference")
        
    Y_expected = torch.nn.functional.layer_norm(x_gpu, [C, H, W], weight=scale_gpu.squeeze(0), bias=bias_gpu.squeeze(0), eps=epsilon_value)
    
    print("Building cudnn graph")

    graph = cudnn.pygraph(intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    X = graph.tensor(name = "X", dim = x_gpu.size(), stride = x_gpu.stride(), data_type = convert_to_cudnn_type(x_gpu.dtype))
    scale = graph.tensor(name = "scale", dim = scale_gpu.size(), stride = scale_gpu.stride(), data_type = convert_to_cudnn_type(scale_gpu.dtype))
    bias = graph.tensor(name = "bias", dim = bias_gpu.size(), stride = bias_gpu.stride(), data_type = convert_to_cudnn_type(bias_gpu.dtype))
    epsilon = graph.tensor(name = "epsilon", dim = epsilon_cpu.size(), stride = epsilon_cpu.stride(), is_pass_by_value = True, data_type = convert_to_cudnn_type(epsilon_cpu.dtype))

    Y, mean, inv_var = graph.layernorm(name = "LN", 
                            norm_forward_phase = cudnn.norm_forward_phase.INFERENCE,
                            input = X,
                            scale = scale, 
                            bias = bias,
                            epsilon = epsilon)
    
    Y.set_output(True).set_data_type(convert_to_cudnn_type(x_gpu.dtype))
    assert mean is None, "Forward mode of inference should not output mean tensor"
    assert inv_var is None, "Forward mode of inference should not output inv_var tensor"

    graph.check_support()
    graph.build()
    
    Y_actual = torch.zeros_like(x_gpu)
    
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    print("Executing cudnn graph")
    
    graph.execute({
                X : x_gpu
                , scale : scale_gpu
                , bias : bias_gpu
                , epsilon: epsilon_cpu
                , Y : Y_actual
            }, workspace)
    
    print("Comparing with reference")
    torch.testing.assert_close(Y_expected, Y_actual, atol=2e-2, rtol=2e-2)
    print("Success!!")
    
    
if __name__ == "__main__":
    test_ln((1600, torch.bfloat16))