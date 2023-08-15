import cudnn
import pytest
import torch

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    else:
        raise ValueError("Unsupported tensor data type.")


n = 4
c = 32
k = 64
padding  = [1,1]
stride   = [1,1]
dilation = [1,1]

@pytest.mark.skipif(cudnn.get_cudnn_version() < 8800, reason="requires cudnn 8.8 or higher")
def test_scale_bias_relu_wgrad():
    print("Running test_scale_bias_relu_wgrad")

    # Reference
    X_gpu  = torch.randn(n, c, 32, 32, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    DY_gpu = torch.randn(n, k, 32, 32, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    DW_gpu = torch.randn(k, c, 3, 3, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    scale  = torch.randn(1, c, 1, 1, device = "cuda", dtype = torch.float16).to(memory_format=torch.channels_last) * 0.01
    bias   = torch.randn(1, c, 1, 1, device = "cuda", dtype = torch.float16).to(memory_format=torch.channels_last) * 0.01

    graph = cudnn.pygraph(io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    X  = graph.tensor(name = "X",  dim = X_gpu.size(), stride = X_gpu.stride(), data_type = convert_to_cudnn_type(X_gpu.dtype))
    DY = graph.tensor(name = "DY", dim = DY_gpu.size(), stride = DY_gpu.stride(), data_type = convert_to_cudnn_type(DY_gpu.dtype))
    B  = graph.tensor(name = "B", dim = bias.size(), stride = bias.stride(), data_type = convert_to_cudnn_type(bias.dtype))
    S  = graph.tensor(name = "S", dim = scale.size(), stride = scale.stride(), data_type = convert_to_cudnn_type(scale.dtype))

    scale_output = graph.scale(name = "scale", input = X, scale = S)
    bias_output  = graph.bias(name = "bias", input = scale_output, bias = B)

    relu_output  = graph.relu(name = "relu", input = bias_output)

    wgrad_output = graph.conv_wgrad(name = "wgrad", image = relu_output, loss = DY, padding = padding, stride = stride, dilation = dilation)
    wgrad_output.set_output(True)

    graph.check_support()
    
    graph.build()

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    DW_actual = torch.zeros_like(X_gpu)

    print("Executing test_scale_bias_relu_wgrad")
    graph.execute({X: X_gpu, DY: DY_gpu, B: bias, S: scale, wgrad_output: DW_actual}, workspace)

if __name__ == "__main__":
    test_scale_bias_relu_wgrad()