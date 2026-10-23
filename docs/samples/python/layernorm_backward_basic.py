"""Complete standalone Python sample.
API page: docs/api/python/layernorm_backward.md
API: layernorm_backward
Use case: Basic
Source: adapted from `test/python/test_layernorm.py`.
"""

import cudnn
import torch


def sample():
    shape = (8, 64, 1, 1)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last
    )
    scale = torch.randn((1, 64, 1, 1), device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last
    )
    grad = torch.randn_like(x)
    mean = x.float().mean(dim=(1, 2, 3), keepdim=True)
    inv_var = torch.rsqrt(x.float().var(dim=(1, 2, 3), correction=0, keepdim=True) + 1e-3)
    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    grad_desc = graph.tensor_like(grad)
    x_desc = graph.tensor_like(x)
    scale_desc = graph.tensor_like(scale)
    mean_desc = graph.tensor_like(mean)
    inv_var_desc = graph.tensor_like(inv_var)
    dx_desc, dscale_desc, dbias_desc = graph.layernorm_backward(
        grad=grad_desc,
        input=x_desc,
        scale=scale_desc,
        mean=mean_desc,
        inv_variance=inv_var_desc,
    )
    dx_desc.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    dscale_desc.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    dbias_desc.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    dx = torch.empty_like(x)
    dscale = torch.empty_like(scale)
    dbias = torch.empty_like(scale)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(
        {
            grad_desc: grad,
            x_desc: x,
            scale_desc: scale,
            mean_desc: mean,
            inv_var_desc: inv_var,
            dx_desc: dx,
            dscale_desc: dscale,
            dbias_desc: dbias,
        },
        workspace,
    )
    torch.cuda.synchronize()

    x_ref = x.float().detach().requires_grad_(True)
    scale_ref = scale.float().reshape(64, 1, 1).detach().requires_grad_(True)
    bias_ref = torch.zeros_like(scale_ref, requires_grad=True)
    y_ref = torch.nn.functional.layer_norm(
        x_ref,
        (64, 1, 1),
        scale_ref,
        bias_ref,
        eps=1e-3,
    )
    y_ref.backward(grad.float())
    torch.testing.assert_close(dx.float(), x_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(
        dscale.float(), scale_ref.grad.reshape_as(dscale), rtol=3e-2, atol=3e-2
    )
    torch.testing.assert_close(dbias.float(), bias_ref.grad.reshape_as(dbias), rtol=3e-2, atol=3e-2)
    print(f"layernorm backward matches reference: dx={tuple(dx.shape)}")


if __name__ == "__main__":
    sample()
