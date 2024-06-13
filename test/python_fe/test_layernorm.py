import cudnn
import pytest
import torch
import itertools
from looseversion import LooseVersion

from test_utils import torch_fork_set_rng

embedding_dim_options = [768, 1024, 1280, 1600]
input_type_options = [torch.bfloat16, torch.float16]

all_options = [
    elem for elem in itertools.product(*[embedding_dim_options, input_type_options])
]


@pytest.fixture(params=all_options)
def param_extract(request):
    return request.param


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "8.9.5",
    reason="LN not supported below cudnn 8.9.5",
)
@torch_fork_set_rng(seed=0)
def test_layernorm(param_extract):

    embedding_dim, input_type = param_extract

    if input_type == torch.bfloat16:
        atol, rtol = 0.125, 0.125
    else:
        atol, rtol = 1e-2, 1e-2

    batch_size, seq_size = 16, 128
    N, C, H, W = batch_size * seq_size, embedding_dim, 1, 1

    epsilon_value = 1e-3

    x_gpu = (
        3
        * torch.randn(
            N, C, H, W, requires_grad=True, device="cuda", dtype=input_type
        ).to(memory_format=torch.channels_last)
        - 0.5
    )
    scale_gpu = (
        5
        * torch.randn(
            1, C, H, W, requires_grad=True, device="cuda", dtype=input_type
        ).to(memory_format=torch.channels_last)
        - 1
    )
    bias_gpu = (
        7
        * torch.randn(
            1, C, H, W, requires_grad=True, device="cuda", dtype=input_type
        ).to(memory_format=torch.channels_last)
        - 2
    )
    epsilon_cpu = torch.full(
        (1, 1, 1, 1),
        epsilon_value,
        requires_grad=False,
        device="cpu",
        dtype=torch.float32,
    )

    Y_expected = torch.nn.functional.layer_norm(
        x_gpu,
        [C, H, W],
        weight=scale_gpu.squeeze(0),
        bias=bias_gpu.squeeze(0),
        eps=epsilon_value,
    )
    mean_expected = x_gpu.to(torch.float32).mean(dim=(1, 2, 3), keepdim=True)
    inv_var_expected = torch.rsqrt(
        torch.var(x_gpu.to(torch.float32), dim=(1, 2, 3), keepdim=True) + epsilon_value
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
        name="X", dim=x_gpu.size(), stride=x_gpu.stride(), data_type=x_gpu.dtype
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
    epsilon = graph.tensor(
        name="epsilon",
        dim=epsilon_cpu.size(),
        stride=epsilon_cpu.stride(),
        is_pass_by_value=True,
        data_type=epsilon_cpu.dtype,
    )

    Y, mean, inv_var = graph.layernorm(
        name="LN",
        norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
        input=X,
        scale=scale,
        bias=bias,
        epsilon=epsilon,
    )

    Y.set_output(True).set_data_type(x_gpu.dtype)
    mean.set_output(True).set_data_type(mean_expected.dtype)
    inv_var.set_output(True).set_data_type(inv_var_expected.dtype)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(cudnn.build_plan_policy.ALL)

    Y_actual = torch.empty_like(x_gpu)
    mean_actual = torch.empty_like(mean_expected)
    inv_var_actual = torch.empty_like(inv_var_expected)

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    graph.execute(
        {
            X: x_gpu.detach(),
            scale: scale_gpu.detach(),
            bias: bias_gpu.detach(),
            epsilon: epsilon_cpu,
            Y: Y_actual,
            mean: mean_actual,
            inv_var: inv_var_actual,
        },
        workspace,
        handle=handle,
    )

    torch.testing.assert_close(Y_expected, Y_actual, atol=atol, rtol=rtol)
    torch.testing.assert_close(mean_expected, mean_actual, atol=atol, rtol=rtol)
    torch.testing.assert_close(inv_var_expected, inv_var_actual, atol=atol, rtol=rtol)

    target = torch.randn_like(Y_expected)
    criterion = torch.nn.MSELoss()
    loss = criterion(Y_expected, target)

    Y_expected.retain_grad()
    x_gpu.retain_grad()
    scale_gpu.retain_grad()
    bias_gpu.retain_grad()

    loss.backward()

    bwd_graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    DY = bwd_graph.tensor(
        name="DY", dim=x_gpu.size(), stride=x_gpu.stride(), data_type=x_gpu.dtype
    )
    X_bwd = bwd_graph.tensor_like(X, name="X")
    scale_bwd = bwd_graph.tensor_like(scale, name="scale")
    mean_bwd = bwd_graph.tensor_like(mean, name="mean")
    inv_var_bwd = bwd_graph.tensor_like(inv_var, name="inv_var")

    DX, Dscale, Dbias = bwd_graph.layernorm_backward(
        name="DLN",
        grad=DY,
        input=X_bwd,
        scale=scale_bwd,
        mean=mean_bwd,
        inv_variance=inv_var_bwd,
    )

    DX.set_output(True).set_data_type(x_gpu.dtype)
    Dscale.set_output(True).set_data_type(x_gpu.dtype)
    Dbias.set_output(True).set_data_type(x_gpu.dtype)

    bwd_graph.validate()
    bwd_graph.build_operation_graph()
    bwd_graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    bwd_graph.check_support()
    bwd_graph.build_plans(cudnn.build_plan_policy.ALL)

    DX_actual = torch.empty_like(x_gpu)
    DScale_actual = torch.empty_like(scale_gpu)
    Dbias_actual = torch.empty_like(bias_gpu)

    workspace = torch.empty(
        bwd_graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    bwd_graph.execute(
        {
            X_bwd: x_gpu.detach(),
            scale_bwd: scale_gpu.detach(),
            DY: Y_expected.grad,
            mean_bwd: mean_actual.detach(),
            inv_var_bwd: inv_var_actual.detach(),
            DX: DX_actual,
            Dscale: DScale_actual,
            Dbias: Dbias_actual,
        },
        workspace,
        handle=handle,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(x_gpu.grad, DX_actual, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(scale_gpu.grad, DScale_actual, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(bias_gpu.grad, Dbias_actual, atol=2e-4, rtol=2e-4)
    cudnn.destroy_handle(handle)


if __name__ == "__main__":
    test_layernorm((1600, torch.bfloat16))
