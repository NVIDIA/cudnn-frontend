import cudnn
import torch

from test_utils import torch_fork_set_rng


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device="cuda",
    base: int = 10000,
    condense_ratio: int = 1,
):
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    return cos, sin


def apply_rope_ref(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    def fn(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        head_size = x.size(-1)
        x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
        x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
        rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
        roped = (x * cos) + (rotated * sin)
        return roped.type_as(x)

    rope_n_elem = cos.size(-1)
    q_roped = fn(q[..., :rope_n_elem], cos, sin)
    return torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)


@torch_fork_set_rng(seed=0)
def test_apply_rope():

    B, nh, T, hs = 8, 32, 4096, 128
    rope_n_elem = int(0.25 * hs)

    # Reference
    x_gpu = torch.randn(B, nh, T, hs, dtype=torch.float16, device="cuda")

    cos_gpu, sin_gpu = build_rope_cache(
        seq_len=T,
        n_elem=rope_n_elem,
    )

    Y_expected = apply_rope_ref(x_gpu, cos_gpu, sin_gpu)

    # Cudnn code
    x_gpu_3d = x_gpu.reshape(-1, T, hs)
    x1_gpu = x_gpu_3d[..., : rope_n_elem // 2]
    x2_gpu = x_gpu_3d[..., rope_n_elem // 2 : rope_n_elem]

    cos_gpu = cos_gpu.reshape(1, T, rope_n_elem)
    cos1_gpu = cos_gpu[..., : rope_n_elem // 2]
    cos2_gpu = cos_gpu[..., rope_n_elem // 2 :]

    sin_gpu = sin_gpu.reshape(1, T, rope_n_elem)
    sin1_gpu = sin_gpu[..., : rope_n_elem // 2]
    sin2_gpu = sin_gpu[..., rope_n_elem // 2 :]

    handle = cudnn.create_handle()
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=handle, stream=stream)

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )
    x1 = graph.tensor_like(x1_gpu)
    x2 = graph.tensor_like(x2_gpu)
    cos1 = graph.tensor_like(cos1_gpu)
    cos2 = graph.tensor_like(cos2_gpu)
    sin1 = graph.tensor_like(sin1_gpu)
    sin2 = graph.tensor_like(sin2_gpu)

    x1_cos1 = graph.mul(a=x1, b=cos1)
    x2_cos2 = graph.mul(a=x2, b=cos2)

    x2_sin1 = graph.mul(a=x2, b=sin1)
    x1_sin2 = graph.mul(a=x1, b=sin2)

    Y1 = graph.sub(a=x1_cos1, b=x2_sin1)
    Y1.set_output(True).set_data_type(torch.float16)

    Y2 = graph.add(a=x2_cos2, b=x1_sin2)
    Y2.set_output(True).set_data_type(torch.float16)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    graph.execute(
        {
            x1: x1_gpu,
            x2: x2_gpu,
            sin1: sin1_gpu,
            sin2: sin2_gpu,
            cos1: cos1_gpu,
            cos2: cos2_gpu,
            Y1: x1_gpu,
            Y2: x2_gpu,
        },
        workspace,
        handle=handle,
    )

    torch.cuda.synchronize()
    # Compare
    torch.testing.assert_close(Y_expected, x_gpu, atol=1e-2, rtol=1e-2)

    cudnn.destroy_handle(handle)
