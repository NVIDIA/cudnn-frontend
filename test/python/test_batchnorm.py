import cudnn
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
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "8.8",
    reason="BN with mask output not supported below cudnn 8.8",
)
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_bn_relu_with_mask(cudnn_handle):
    n, c, h, w = 4, 16, 56, 56
    input_type = torch.float16

    epsilon_value = 1e-3
    momentum_value = 1e-1

    # input tensors
    x_gpu = torch.randn(n, c, h, w, dtype=input_type, device="cuda")
    x_gpu = x_gpu.to(memory_format=torch.channels_last)
    scale_gpu = torch.randn(1, c, 1, 1, device="cuda")
    bias_gpu = torch.randn_like(scale_gpu)
    running_mean_gpu = torch.randn_like(scale_gpu)
    running_var_gpu = torch.randn_like(scale_gpu)

    comparison_gpu = torch.zeros_like(x_gpu, dtype=input_type, device="cuda")

    epsilon_cpu = torch.full((1, 1, 1, 1), epsilon_value)
    momentum_cpu = torch.full((1, 1, 1, 1), momentum_value)

    # output tensors
    saved_mean_gpu = torch.empty_like(running_mean_gpu, device="cuda")
    saved_inv_var_gpu = torch.empty_like(running_var_gpu, device="cuda")
    y_gpu = torch.empty_like(x_gpu, dtype=input_type, device="cuda")
    mask_gpu = torch.empty_like(x_gpu, dtype=torch.bool, device="cuda")

    # cudnn graph
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(input_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    x = graph.tensor_like(x_gpu)
    scale = graph.tensor_like(scale_gpu)
    bias = graph.tensor_like(bias_gpu)

    in_running_mean = graph.tensor_like(running_mean_gpu)
    in_running_var = graph.tensor_like(running_var_gpu)
    epsilon = graph.tensor_like(epsilon_cpu)
    momentum = graph.tensor_like(momentum_cpu)
    comparison = graph.tensor_like(x_gpu)

    y_before_relu, saved_mean, saved_inv_var, out_running_mean, out_running_var = (
        graph.batchnorm(
            name="BN",
            input=x,
            scale=scale,
            bias=bias,
            in_running_mean=in_running_mean,
            in_running_var=in_running_var,
            epsilon=epsilon,
            momentum=momentum,
        )
    )
    y = graph.relu(name="relu", input=y_before_relu)
    mask = graph.cmp_gt(name="cmp", input=y, comparison=comparison)

    y.set_output(True)
    saved_mean.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    saved_inv_var.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    out_running_mean.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    out_running_var.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    mask.set_output(True).set_data_type(cudnn.data_type.BOOLEAN)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    # cudnn graph execution
    variant_pack = {
        x: x_gpu,
        scale: scale_gpu,
        bias: bias_gpu,
        in_running_mean: running_mean_gpu,
        in_running_var: running_var_gpu,
        epsilon: epsilon_cpu,
        momentum: momentum_cpu,
        out_running_mean: running_mean_gpu,
        out_running_var: running_var_gpu,
        saved_mean: saved_mean_gpu,
        saved_inv_var: saved_inv_var_gpu,
        y: y_gpu,
        comparison: comparison_gpu,
        mask: mask_gpu,
    }
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute(
        variant_pack,
        workspace,
        handle=cudnn_handle,
    )
    torch.cuda.synchronize()

    # reference computation
    x_ref = x_gpu.clone().float()
    running_mean_ref = running_mean_gpu.clone().float()
    running_var_ref = running_var_gpu.clone().float()

    y_before_relu_ref = torch.nn.functional.batch_norm(
        x_ref,
        running_mean_ref,  # running_mean is both input and output
        running_var_ref,  # running_var is both input and output
        weight=scale_gpu,
        bias=bias_gpu,
        training=True,
        momentum=momentum_cpu.item(),
        eps=epsilon_cpu.item(),
    )

    mean_ref = torch.mean(x_ref, dim=(0, 2, 3), keepdim=True)
    inv_var_ref = torch.var(x_ref, dim=(0, 2, 3), keepdim=True)
    inv_var_ref = torch.rsqrt(inv_var_ref + epsilon_value)
    y_ref = torch.relu(y_before_relu_ref)
    mask_ref = y_ref > 0

    # Compare
    # fmt: off
    torch.testing.assert_close(y_ref, y_gpu.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(mean_ref, saved_mean_gpu.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(inv_var_ref, saved_inv_var_gpu.float(), atol=1e-3, rtol=1e-3)
    # torch.testing.assert_close(mask_ref, mask_gpu.float(), atol=1e-3, rtol=1e-3)
    # fmt: on


@pytest.mark.parametrize(
    "dump_dX_dRelu", [True, False], ids=lambda p: f"dump_dX_dRelu{int(p)}"
)
@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "8.9",
    reason="DBN fusions not supported below cudnn 8.9",
)
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_drelu_dadd_dbn(dump_dX_dRelu, cudnn_handle):
    n, c, h, w = 4, 16, 56, 56
    input_type = torch.float16

    # input tensors
    x_gpu = torch.randn(n, c, h, w, dtype=input_type, device="cuda")
    x_gpu = x_gpu.to(memory_format=torch.channels_last)
    x_mask_gpu = torch.randn_like(x_gpu) > 0.0
    scale_gpu = torch.randn(1, c, 1, 1, device="cuda")
    mean_gpu = torch.randn_like(scale_gpu)
    inv_var_gpu = torch.randn_like(scale_gpu)
    dY_gpu = torch.randn_like(x_gpu)

    # output tensors
    dScale_ref = torch.empty_like(scale_gpu)
    dBias_ref = torch.empty_like(scale_gpu)
    dX_ref = torch.empty_like(dY_gpu)

    if dump_dX_dRelu:
        dX_dRelu_gpu = torch.empty_like(dY_gpu)

    # cudnn graph
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(input_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    x = graph.tensor_like(x_gpu)
    x_mask = graph.tensor_like(x_mask_gpu)
    scale = graph.tensor_like(scale_gpu)
    mean = graph.tensor_like(mean_gpu)
    inv_var = graph.tensor_like(inv_var_gpu)
    dY = graph.tensor_like(dY_gpu)

    dX_drelu = graph.scale(name="drelu", input=dY, scale=x_mask)
    dX_drelu.set_data_type(cudnn.data_type.HALF)

    if dump_dX_dRelu:
        dX_drelu.set_output(True)

    dX, dScale, dBias = graph.batchnorm_backward(
        name="DBN",
        grad=dX_drelu,
        input=x,
        scale=scale,
        mean=mean,
        inv_variance=inv_var,
    )

    dX.set_output(True).set_data_type(cudnn.data_type.HALF)
    dScale.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    dBias.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        x: x_gpu,
        x_mask: x_mask_gpu,
        dY: dY_gpu,
        scale: scale_gpu,
        mean: mean_gpu,
        inv_var: inv_var_gpu,
        dX: dX_ref,
        dScale: dScale_ref,
        dBias: dBias_ref,
    }
    if dump_dX_dRelu:
        variant_pack[dX_drelu] = dX_dRelu_gpu

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "8.9.4",
    reason="BN_infer-Drelu-DBN not supported below cudnn 8.9.4",
)
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_bn_infer_drelu_dbn(cudnn_handle):
    n, c, h, w = 4, 16, 56, 56
    input_type = torch.float16

    # input tensors
    x_gpu = torch.randn(n, c, h, w, dtype=input_type, device="cuda")
    x_gpu = x_gpu.to(memory_format=torch.channels_last)
    scale_gpu = torch.randn(1, c, 1, 1, device="cuda")
    bias_gpu = torch.randn_like(scale_gpu)
    mean_gpu = torch.randn_like(scale_gpu)
    inv_var_gpu = torch.randn_like(scale_gpu)
    dY_gpu = torch.randn_like(x_gpu)

    # output tensors
    dScale_gpu = torch.empty_like(scale_gpu)
    dBias_gpu = torch.empty_like(scale_gpu)
    dX_gpu = torch.empty_like(x_gpu)

    # cudnn graph
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    x = graph.tensor(
        name="x",
        dim=x_gpu.size(),
        stride=x_gpu.stride(),
        data_type=x_gpu.dtype,
    )
    dY = graph.tensor(
        name="dY", dim=dY_gpu.size(), stride=dY_gpu.stride(), data_type=dY_gpu.dtype
    )
    scale = graph.tensor(
        name="scale",
        dim=scale_gpu.size(),
        stride=scale_gpu.stride(),
        data_type=scale_gpu.dtype,
    )
    bias = graph.tensor(
        name="bias",
        dim=bias_gpu.size(),
        stride=bias_gpu.stride(),
        data_type=bias_gpu.dtype,
    )
    mean = graph.tensor(
        name="mean",
        dim=mean_gpu.size(),
        stride=mean_gpu.stride(),
        data_type=mean_gpu.dtype,
    )
    inv_variance = graph.tensor(
        name="inv_variance",
        dim=inv_var_gpu.size(),
        stride=inv_var_gpu.stride(),
        data_type=inv_var_gpu.dtype,
    )

    y = graph.batchnorm_inference(
        input=x, mean=mean, inv_variance=inv_variance, scale=scale, bias=bias
    )

    dX_dRelu = graph.relu_backward(loss=dY, input=y)

    dX_dRelu.set_data_type(cudnn.data_type.HALF)

    dX, dScale, dBias = graph.batchnorm_backward(
        name="DBN",
        grad=dX_dRelu,
        input=x,
        scale=scale,
        mean=mean,
        inv_variance=inv_variance,
    )

    dX.set_output(True)
    dScale.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    dBias.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        x: x_gpu,
        dY: dY_gpu,
        scale: scale_gpu,
        bias: bias_gpu,
        mean: mean_gpu,
        inv_variance: inv_var_gpu,
        dX: dX_gpu,
        dScale: dScale_gpu,
        dBias: dBias_gpu,
    }

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
