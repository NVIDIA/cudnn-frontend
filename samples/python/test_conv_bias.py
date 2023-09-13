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
    
    handle = cudnn.create_handle()

    graph = cudnn.pygraph(io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT, handle = handle)

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
    
    cudnn.destroy_handle(handle)
    
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

def test_conv3d_bias_leaky_relu():
    N, C, D, H, W = 4, 16, 52, 54, 56
    K, R, S, T = 32, 3, 3, 3
    padding = [0,1,2]
    stride = [2,3,4]
    dilation = [1,1,1]
    negative_slope = 0.01
    
    # Reference code
    X_gpu = torch.randn(N, D, H, W, C, requires_grad=False, device="cuda", dtype=torch.float16).permute(0, 4, 1, 2, 3)
    W_gpu = torch.randn(K, R, S, T, C, requires_grad=False, device="cuda", dtype=torch.float16).permute(0, 4, 1, 2, 3)
    B_gpu = torch.randn(1, 1, 1, 1, K, requires_grad=False, device="cuda", dtype=torch.float16).permute(0, 4, 1, 2, 3)
    
    conv_out_expected = torch.nn.functional.conv3d(X_gpu, W_gpu, bias = B_gpu.reshape(-1), padding=padding, stride=stride, dilation=dilation).to("cuda").to(torch.float16)
    Y_expected = torch.nn.functional.leaky_relu(conv_out_expected, negative_slope = negative_slope)
    
    graph = cudnn.pygraph(io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    X = graph.tensor(name = "X", dim = X_gpu.size(), stride = X_gpu.stride(), data_type = convert_to_cudnn_type(X_gpu.dtype))
    Weight = graph.tensor(name = "W", dim = W_gpu.size(), stride = W_gpu.stride(), data_type = convert_to_cudnn_type(W_gpu.dtype))
    B = graph.tensor(name = "B", dim = B_gpu.size(), stride = B_gpu.stride(), data_type = convert_to_cudnn_type(B_gpu.dtype))

    conv_output = graph.conv_fprop(image = X, weight = Weight, padding = padding, stride = stride, dilation = dilation)

    bias_output = graph.bias(name = "bias", input = conv_output, bias = B)

    Y = graph.leaky_relu(name = "relu", input = bias_output, negative_slope = negative_slope)
    Y.set_output(True)
    
    graph.check_support()

    graph.build()

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    Y_actual = torch.zeros_like(Y_expected)
    graph.execute({X: X_gpu, Weight: W_gpu, B: B_gpu, Y: Y_actual}, workspace)

    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)

def test_leaky_relu_backward():
    N, C, H, W = 4, 16, 56, 56
    negative_slope = 0.01
    
    # Reference code
    loss_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    input_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)

    def dleaky_relu(grad: torch.Tensor, mask: torch.Tensor, negative_slope: float):
        return torch.ones_like(grad).masked_fill_(mask <= 0.0, negative_slope) * grad
        
    Y_expected = dleaky_relu(loss_gpu, input_gpu, negative_slope)
    
    graph = cudnn.pygraph(io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    loss = graph.tensor(name = "loss", dim = loss_gpu.size(), stride = loss_gpu.stride(), data_type = convert_to_cudnn_type(loss_gpu.dtype))
    input = graph.tensor(name = "input", dim = input_gpu.size(), stride = input_gpu.stride(), data_type = convert_to_cudnn_type(input_gpu.dtype))

    Y = graph.leaky_relu_backward(loss = loss, input = input, negative_slope = negative_slope)
    Y.set_output(True)
    
    graph.check_support()

    graph.build()

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    Y_actual = torch.zeros_like(Y_expected)
    graph.execute({loss: loss_gpu, input: input_gpu, Y: Y_actual}, workspace)

    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    # test_conv_relu()
    # test_conv_bias_relu()
    # test_conv3d_bias_leaky_relu()
    for i in range(100):
        print(i)
        test_leaky_relu_backward()