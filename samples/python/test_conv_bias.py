import cudnn
import torch

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    else:
        raise ValueError("Unsupported tensor data type.")

class CSBR(torch.nn.Module):
    def forward(self, x, w, b = None, padding = [1,1], stride = [1,1], dilation = [1,1]):
        if b is not None:
            b = b.reshape(-1) # Conv2d needs a 1D tensor
        conv_output = torch.nn.functional.conv2d(x, w, bias = b, padding=padding, stride=stride, dilation=dilation)
        return torch.nn.functional.relu(conv_output)

def test_conv_bias_relu():
    # Reference code
    X_gpu = torch.randn(4, 16, 56, 56, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    W_gpu = torch.randn(16, 16, 3, 3, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    B_gpu = torch.randn(1, 16, 1, 1, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    padding = [0,1]
    stride = [2,3]
    dilation = [1,1]
    model = CSBR().eval().to("cuda").to(torch.float16)
    Y_expected = model(X_gpu, W_gpu, b = B_gpu, padding = padding, stride = stride, dilation = dilation)

    graph = cudnn.pygraph(io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    X = graph.tensor(name = "X", dim = X_gpu.size(), stride = X_gpu.stride(), data_type = convert_to_cudnn_type(X_gpu.dtype))
    W = graph.tensor(name = "W", dim = W_gpu.size(), stride = W_gpu.stride(), data_type = convert_to_cudnn_type(W_gpu.dtype))
    B = graph.tensor(name = "B", dim = B_gpu.size(), stride = B_gpu.stride(), data_type = convert_to_cudnn_type(B_gpu.dtype))

    conv_output = graph.conv_fprop(image = X, weight = W, padding = padding, stride = stride, dilation = dilation)

    bias_output = graph.bias(name = "bias", input = conv_output, bias = B)

    Y = graph.relu(name = "relu", input = bias_output)
    Y.set_output(True)
    
    graph.check_support()

    graph.build()

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    Y_actual = torch.zeros_like(Y_expected)
    graph.execute({X: X_gpu, W: W_gpu, B: B_gpu, Y: Y_actual}, workspace)

    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)
    
def test_conv_relu():
    # Reference code
    X_gpu = torch.randn(20, 40, 30, 40, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    W_gpu = torch.randn(54, 40, 3, 4, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    padding = [0,1]
    stride = [2,3]
    dilation = [1,1]
    model = CSBR().eval().to("cuda").to(torch.float16)
    Y_expected = model(X_gpu, W_gpu, padding = padding, stride = stride, dilation = dilation)

    # Cudnn code
    graph = cudnn.pygraph(io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    X = graph.tensor(name = "X", dim = X_gpu.size(), stride = X_gpu.stride(), data_type = convert_to_cudnn_type(X_gpu.dtype))
    W = graph.tensor(name = "W", dim = W_gpu.size(), stride = W_gpu.stride(), data_type = convert_to_cudnn_type(W_gpu.dtype))
    
    conv_output = graph.conv_fprop(image = X, weight = W, padding = padding, stride = stride, dilation = dilation)

    Y = graph.relu(name = "relu", input = conv_output)
    Y.set_output(True)
    
    graph.check_support()
    
    graph.build()

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    Y_actual = torch.zeros_like(Y_expected)
    graph.execute({X: X_gpu, W: W_gpu, Y: Y_actual}, workspace)

    # Compare
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    test_conv_relu()
    test_conv_bias_relu()