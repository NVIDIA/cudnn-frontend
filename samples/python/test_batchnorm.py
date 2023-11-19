import cudnn
import pytest
import torch

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.bool:
        return cudnn.data_type.BOOLEAN
    elif torch_type == torch.uint8:
        return cudnn.data_type.UINT8
    else:
        raise ValueError("Unsupported tensor data type.")

class SGBN(torch.nn.Module):
    def forward(self, input, running_mean, running_var, weight, bias, eps, momentum):
        return torch.nn.functional.batch_norm(input, running_mean, running_var, weight=weight, bias=bias, training=True, momentum=momentum, eps=eps)

@pytest.mark.skipif(cudnn.backend_version() < 8800, reason="BN with mask output not supported below cudnn 8.8")
def test_bn_relu_with_mask():
    N, C, H, W = 4, 16, 56, 56
    x_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    scale_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    bias_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    running_mean_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    running_var_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)

    epsilon_value = 1e-03
    epsilon_cpu = torch.full((1, 1, 1, 1), epsilon_value, requires_grad=False, device="cpu", dtype=torch.float32)
    momentum_cpu = torch.full((1, 1, 1, 1), 0.1, requires_grad=False, device="cpu", dtype=torch.float32)


    # Cudnn code
    graph = cudnn.pygraph(io_data_type = cudnn.data_type.FLOAT, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    X = graph.tensor(name = "X", dim = x_gpu.size(), stride = x_gpu.stride(), data_type = convert_to_cudnn_type(x_gpu.dtype))
    scale = graph.tensor(name = "scale", dim = scale_gpu.size(), stride = scale_gpu.stride())
    bias = graph.tensor(name = "bias", dim = bias_gpu.size(), stride = bias_gpu.stride())
    in_running_mean = graph.tensor(name = "in_running_mean", dim = running_mean_gpu.size(), stride = running_mean_gpu.stride())
    in_running_var = graph.tensor(name = "in_running_var", dim = running_var_gpu.size(), stride = running_var_gpu.stride())
    epsilon = graph.tensor(name = "epsilon", dim = epsilon_cpu.size(), stride = epsilon_cpu.stride(), is_pass_by_value = True)
    momentum = graph.tensor(name = "momentum", dim = momentum_cpu.size(), stride = momentum_cpu.stride(), is_pass_by_value = True)
    comparison = graph.tensor(name = "zeros", dim = x_gpu.size(), stride = x_gpu.stride(), data_type = convert_to_cudnn_type(x_gpu.dtype))
    
    (Y_before_relu, saved_mean, saved_inv_var, out_running_mean, out_running_var) = graph.batchnorm(name = "BN"
                                                                                        , input = X
                                                                                        , scale = scale, bias = bias
                                                                                        , in_running_mean = in_running_mean, in_running_var = in_running_var
                                                                                        , epsilon = epsilon, momentum = momentum)
    Y = graph.relu(name = "relu", 
                   input = Y_before_relu)
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)
    saved_mean.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    saved_inv_var.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    out_running_mean.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    out_running_var.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    
    mask = graph.cmp_gt(name="cmp",
                        input = Y,
                        comparison = comparison)
    mask.set_output(True).set_data_type(cudnn.data_type.BOOLEAN)

    
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    # Reference code execution
    model = SGBN().eval().to("cuda")
    Y_expected_before_relu = model(x_gpu, running_mean_gpu, running_var_gpu, scale_gpu, bias_gpu, epsilon_cpu.item(), momentum_cpu.item())
    mean_expected = x_gpu.to(torch.float32).mean(dim=(0, 2, 3), keepdim=True)
    inv_var_expected = torch.rsqrt(torch.var(x_gpu.to(torch.float32), dim=(0, 2, 3), keepdim=True) + epsilon_value)
    Y_expected = torch.relu(Y_expected_before_relu)
    mask_expected = (Y_expected > 0)
    
    # cudnn graph execution
    saved_mean_actual = torch.zeros_like(scale_gpu)
    saved_inv_var_actual = torch.zeros_like(scale_gpu)
    Y_actual = torch.zeros_like(Y_expected)
    # pytorch does not allow bool tensors in dlpack
    assert C%8 == 0
    mask_actual_uint8 = torch.empty(N, C // 8, H, W, requires_grad=False, device="cuda", dtype=torch.uint8).to(memory_format=torch.channels_last)

    zeros = torch.zeros_like(Y_expected)

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    graph.execute({
                    X : x_gpu
                    , scale : scale_gpu
                    , bias : bias_gpu
                    , in_running_mean: running_mean_gpu
                    , in_running_var: running_var_gpu
                    , epsilon: epsilon_cpu
                    , momentum: momentum_cpu
                    , out_running_mean: running_mean_gpu
                    , out_running_var: running_var_gpu
                    , saved_mean : saved_mean_actual
                    , saved_inv_var : saved_inv_var_actual
                    , Y : Y_actual
                    , comparison: zeros
                    , mask : mask_actual_uint8
                }, workspace)

    # Compare
    print("Comparing outputs")
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(mean_expected, saved_mean_actual, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(inv_var_expected, saved_inv_var_actual, atol=1e-3, rtol=1e-3)

@pytest.mark.skipif(cudnn.backend_version() < 8900, reason="DBN fusions not supported below cudnn 8.9")
def test_drelu_dadd_dbn():
    # Tensors
    N, C, H, W = 4, 16, 56, 56

    x_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    scale_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    mean_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    inv_variance_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    dy_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)

    # Cudnn code
    graph = cudnn.pygraph(io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    # Bool type is not supported by dlpack
    x_mask_gpu = torch.randint(0, 255, [N, int(C / 8), H, W], requires_grad=False, device="cuda", dtype=torch.uint8).to(memory_format=torch.channels_last)
    X_mask = graph.tensor(name = "X_mask", dim = [N, C, H, W], stride = x_gpu.stride(), data_type = cudnn.data_type.BOOLEAN)

    X = graph.tensor(name = "X", dim = x_gpu.size(), stride = x_gpu.stride(), data_type = convert_to_cudnn_type(x_gpu.dtype))
    DY = graph.tensor(name = "DY", dim = dy_gpu.size(), stride = dy_gpu.stride(), data_type = convert_to_cudnn_type(dy_gpu.dtype))
    scale = graph.tensor(name = "scale", dim = scale_gpu.size(), stride = scale_gpu.stride(), data_type = convert_to_cudnn_type(scale_gpu.dtype))
    mean = graph.tensor(name = "mean", dim = mean_gpu.size(), stride = mean_gpu.stride(), data_type = convert_to_cudnn_type(mean_gpu.dtype))
    inv_variance = graph.tensor(name = "inv_variance", dim = inv_variance_gpu.size(), stride = inv_variance_gpu.stride(), data_type = convert_to_cudnn_type(inv_variance_gpu.dtype))
    
    DX_drelu = graph.scale(name = "drelu"
                         , input = DY
                         , scale = X_mask)
    
    # NOTE: Toggle DADD output to dump to gmem
    should_dump_dx_drelu = False
    DX_drelu.set_output(should_dump_dx_drelu).set_data_type(cudnn.data_type.HALF)

    (DX, DScale, DBias) = graph.batchnorm_backward(name = "DBN"
                                                    , grad = DX_drelu
                                                    , input = X
                                                    , scale = scale
                                                    , mean = mean
                                                    , inv_variance = inv_variance
                                                )

    DX.set_output(True)
    DScale.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    DBias.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    DScale_actual = torch.zeros_like(scale_gpu)
    DBias_actual = torch.zeros_like(scale_gpu)
    DX_actual = torch.zeros_like(dy_gpu)

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    device_buffers = {
                    X : x_gpu
                    , X_mask : x_mask_gpu
                    , DY : dy_gpu
                    , scale : scale_gpu
                    , mean : mean_gpu
                    , inv_variance : inv_variance_gpu
                    , DX : DX_actual
                    , DScale : DScale_actual
                    , DBias : DBias_actual
                }
    if should_dump_dx_drelu is True:
        DX_drelu_actual = torch.zeros_like(dy_gpu)
        device_buffers[DX_drelu] = DX_drelu_actual
    graph.execute(device_buffers, workspace)

@pytest.mark.skipif(cudnn.backend_version() < 8904, reason="BN_infer-Drelu-DBN not supported below cudnn 8.9.4")
def test_bn_infer_drelu_dbn():
    # Tensors
    N, C, H, W = 4, 16, 56, 56

    bn_x_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    scale_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    bias_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    mean_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    inv_variance_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    dy_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)

    # Cudnn code
    graph = cudnn.pygraph(io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    # Bool type is not supported by dlpack
    BN_X = graph.tensor(name = "BN_X", dim = bn_x_gpu.size(), stride = bn_x_gpu.stride(), data_type = convert_to_cudnn_type(bn_x_gpu.dtype))
    DY = graph.tensor(name = "DY", dim = dy_gpu.size(), stride = dy_gpu.stride(), data_type = convert_to_cudnn_type(dy_gpu.dtype))
    scale = graph.tensor(name = "scale", dim = scale_gpu.size(), stride = scale_gpu.stride(), data_type = convert_to_cudnn_type(scale_gpu.dtype))
    bias = graph.tensor(name = "bias", dim = bias_gpu.size(), stride = bias_gpu.stride(), data_type = convert_to_cudnn_type(bias_gpu.dtype))
    mean = graph.tensor(name = "mean", dim = mean_gpu.size(), stride = mean_gpu.stride(), data_type = convert_to_cudnn_type(mean_gpu.dtype))
    inv_variance = graph.tensor(name = "inv_variance", dim = inv_variance_gpu.size(), stride = inv_variance_gpu.stride(), data_type = convert_to_cudnn_type(inv_variance_gpu.dtype))

    BN_Y = graph.batchnorm_inference(input = BN_X, mean = mean, inv_variance = inv_variance, scale = scale, bias = bias)    

    DX_drelu = graph.relu_backward(loss = DY, input = BN_Y)
    
    DX_drelu.set_data_type(cudnn.data_type.HALF)

    (DX, DScale, DBias) = graph.batchnorm_backward(name = "DBN"
                                                    , grad = DX_drelu
                                                    , input = BN_X
                                                    , scale = scale
                                                    , mean = mean
                                                    , inv_variance = inv_variance
                                                )

    DX.set_output(True)
    DScale.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    DBias.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    DScale_actual = torch.zeros_like(scale_gpu)
    DBias_actual = torch.zeros_like(scale_gpu)
    DX_actual = torch.zeros_like(dy_gpu)

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    device_buffers = {
                    BN_X : bn_x_gpu
                    , DY : dy_gpu
                    , scale : scale_gpu
                    , bias : bias_gpu
                    , mean : mean_gpu
                    , inv_variance : inv_variance_gpu
                    , DX : DX_actual
                    , DScale : DScale_actual
                    , DBias : DBias_actual
                }
    graph.execute(device_buffers, workspace)
        
if __name__ == "__main__":
    test_bn_relu_with_mask()
    test_drelu_dadd_dbn()
    test_bn_infer_drelu_dbn()