"""Complete standalone Python sample.
API page: docs/api/python/layernorm.md
API: layernorm
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
    bias = torch.randn_like(scale)
    epsilon = torch.full((1, 1, 1, 1), 1e-3, dtype=torch.float32)
    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    scale_desc = graph.tensor_like(scale)
    bias_desc = graph.tensor_like(bias)
    epsilon_desc = graph.tensor_like(epsilon)
    y_desc, mean_desc, inv_var_desc = graph.layernorm(
        norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
        input=x_desc,
        scale=scale_desc,
        bias=bias_desc,
        epsilon=epsilon_desc,
    )
    y_desc.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    mean_desc.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    inv_var_desc.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    y = torch.empty_like(x)
    mean = torch.empty((8, 1, 1, 1), device="cuda", dtype=torch.float32)
    inv_var = torch.empty_like(mean)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(
        {
            x_desc: x,
            scale_desc: scale,
            bias_desc: bias,
            epsilon_desc: epsilon,
            y_desc: y,
            mean_desc: mean,
            inv_var_desc: inv_var,
        },
        workspace,
    )
    torch.cuda.synchronize()
    expected = torch.nn.functional.layer_norm(
        x.float(), (64, 1, 1), scale.squeeze(0).float(), bias.squeeze(0).float(), 1e-3
    ).to(torch.bfloat16)
    torch.testing.assert_close(y, expected, atol=0.125, rtol=0.125)
    print(f"layernorm output={tuple(y.shape)}, stats={tuple(mean.shape)}")


if __name__ == "__main__":
    sample()
