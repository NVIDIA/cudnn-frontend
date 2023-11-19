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


input_type_options = [torch.bfloat16, torch.float16]

all_options = [elem for elem in itertools.product(*[input_type_options,])]

@pytest.fixture(params=all_options)
def param_extract(request):
  return request.param

@pytest.mark.skipif(cudnn.backend_version() < 8905, reason="IN not supported below cudnn 8.9.5")
def test_in(param_extract):
    torch.manual_seed(0)

    input_type, = param_extract
    print(input_type)

    if input_type == torch.bfloat16:
        atol, rtol = 0.125, 0.125
    else:
        atol, rtol = 1e-2, 1e-2

    N,C,H,W = 16, 32, 64, 64
    
    epsilon_value = 1e-5

    x_gpu       = torch.randn((N, C, H, W), requires_grad=True, device="cuda", dtype=input_type).to(memory_format=torch.channels_last)
    scale_gpu   = torch.randn((1, C, 1, 1), requires_grad=True, device="cuda", dtype=input_type).to(memory_format=torch.channels_last)
    bias_gpu    = torch.randn((1, C, 1, 1), requires_grad=True, device="cuda", dtype=input_type).to(memory_format=torch.channels_last)
    epsilon_cpu = torch.full((1, 1, 1, 1), epsilon_value, requires_grad=False, device="cpu", dtype=torch.float32)

    print("Running reference")
        
    Y_expected = torch.nn.functional.instance_norm(x_gpu, weight = scale_gpu.view(C), bias = bias_gpu.view(C))
    mean_expected = x_gpu.to(torch.float32).mean(dim=(2, 3), keepdim=True)
    inv_var_expected = torch.rsqrt(torch.var(x_gpu.to(torch.float32), dim=(2, 3), keepdim=True) + epsilon_value)
    print("Building cudnn graph")

    graph = cudnn.pygraph(intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    X = graph.tensor_like(x_gpu.detach())
    scale = graph.tensor_like(scale_gpu.detach())
    bias = graph.tensor_like(bias_gpu.detach())
    epsilon = graph.tensor_like(epsilon_cpu)
    
    Y, mean, inv_var = graph.instancenorm(name = "IN", 
                            norm_forward_phase = cudnn.norm_forward_phase.TRAINING,
                            input = X,
                            scale = scale, 
                            bias = bias,
                            epsilon = epsilon)
    
    Y.set_output(True).set_data_type(convert_to_cudnn_type(x_gpu.dtype))
    mean.set_output(True).set_data_type(convert_to_cudnn_type(mean_expected.dtype))
    inv_var.set_output(True).set_data_type(convert_to_cudnn_type(inv_var_expected.dtype))
    
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    
    Y_actual = torch.empty_like(x_gpu)
    mean_actual = torch.empty_like(mean_expected)
    inv_var_actual = torch.empty_like(inv_var_expected)
    
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    print("Executing cudnn graph")
    
    graph.execute({
                X : x_gpu.detach()
                , scale : scale_gpu.detach()
                , bias : bias_gpu.detach()
                , epsilon: epsilon_cpu
                , Y : Y_actual
                , mean: mean_actual
                , inv_var: inv_var_actual
            }, workspace)
    
    print("Comparing with reference")
    torch.testing.assert_close(Y_expected, Y_actual, atol=atol, rtol=rtol)
    torch.testing.assert_close(mean_expected, mean_actual, atol=atol, rtol=rtol)
    torch.testing.assert_close(inv_var_expected, inv_var_actual, atol=atol, rtol=rtol)
    print("Success!!")

    target = torch.randn_like(Y_expected)
    criterion = torch.nn.MSELoss()
    loss = criterion(Y_expected, target)
    
    Y_expected.retain_grad()
    x_gpu.retain_grad()
    scale_gpu.retain_grad()
    bias_gpu.retain_grad()
    
    loss.backward()
    
    bwd_graph = cudnn.pygraph(intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    # https://github.com/pytorch/pytorch/issues/72341
    # PyT does not preserve layout for IN
    DY_gpu = Y_expected.grad.to(memory_format=torch.channels_last)

    DY = bwd_graph.tensor_like(DY_gpu)
    X_bwd = bwd_graph.tensor_like(x_gpu.detach())
    scale_bwd = bwd_graph.tensor_like(scale_gpu.detach())
    mean_bwd = bwd_graph.tensor_like(mean_actual.detach())
    inv_var_bwd = bwd_graph.tensor_like(inv_var_actual.detach())
    epsilon_bwd = bwd_graph.tensor_like(epsilon_cpu)

    DX, Dscale, Dbias = bwd_graph.instancenorm_backward(name = "DIN", 
                            grad = DY,
                            input = X_bwd,
                            scale = scale_bwd, 
                            mean = mean_bwd,
                            inv_variance = inv_var_bwd)
    
    DX.set_output(True).set_data_type(convert_to_cudnn_type(x_gpu.dtype))
    Dscale.set_output(True).set_data_type(convert_to_cudnn_type(scale_gpu.dtype))
    Dbias.set_output(True).set_data_type(convert_to_cudnn_type(bias_gpu.dtype))

    bwd_graph.validate()
    bwd_graph.build_operation_graph()
    bwd_graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    bwd_graph.check_support()
    bwd_graph.build_plans()
    
    DX_actual = torch.empty_like(x_gpu)
    DScale_actual = torch.empty_like(scale_gpu)
    Dbias_actual = torch.empty_like(bias_gpu)

    workspace = torch.empty(bwd_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    print("Executing cudnn bwd_graph")
    
    bwd_graph.execute({
                X_bwd : x_gpu.detach()
                , scale_bwd : scale_gpu.detach()
                , DY : DY_gpu
                , mean_bwd: mean_actual.detach()
                , inv_var_bwd: inv_var_actual.detach()
                , epsilon_bwd: epsilon_cpu
                , DX: DX_actual
                , Dscale: DScale_actual
                , Dbias: Dbias_actual
            }, workspace)

    torch.cuda.synchronize()
    print("Comparing with reference")
    torch.testing.assert_close(x_gpu.grad, DX_actual, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(scale_gpu.grad, DScale_actual, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(bias_gpu.grad, Dbias_actual, atol=2e-3, rtol=2e-3)
    print("Success!!")

if __name__ == "__main__":
    test_in((torch.float16, ))