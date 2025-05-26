import cudnn
import torch
import pytest

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


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
def create_rope_graph(handle, x1_gpu, x2_gpu, cos1_gpu, cos2_gpu, sin1_gpu, sin2_gpu):
    with cudnn.graph(
        handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    ) as (g, _):
        x1 = g.tensor_like(x1_gpu)
        x2 = g.tensor_like(x2_gpu)
        cos1 = g.tensor_like(cos1_gpu)
        cos2 = g.tensor_like(cos2_gpu)
        sin1 = g.tensor_like(sin1_gpu)
        sin2 = g.tensor_like(sin2_gpu)

        x1_cos1 = g.mul(a=x1, b=cos1)
        x2_cos2 = g.mul(a=x2, b=cos2)
        x2_sin1 = g.mul(a=x2, b=sin1)
        x1_sin2 = g.mul(a=x1, b=sin2)

        Y1 = g.sub(a=x1_cos1, b=x2_sin1)
        Y1.set_output(True).set_data_type(torch.float16)

        Y2 = g.add(a=x2_cos2, b=x1_sin2)
        Y2.set_output(True).set_data_type(torch.float16)

        return g, [x1, x2, sin1, sin2, cos1, cos2, Y1, Y2]


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_apply_rope(cudnn_handle):
    B, nh, T, hs = 8, 32, 4096, 128
    rope_n_elem = int(0.25 * hs)

    # Reference
    x_gpu = torch.randn(B, nh, T, hs, dtype=torch.float16, device="cuda")
    cos_gpu, sin_gpu = build_rope_cache(seq_len=T, n_elem=rope_n_elem)
    Y_expected = apply_rope_ref(x_gpu, cos_gpu, sin_gpu)

    # Prepare inputs
    x_gpu_3d = x_gpu.reshape(-1, T, hs)
    x1_gpu = x_gpu_3d[..., : rope_n_elem // 2]
    x2_gpu = x_gpu_3d[..., rope_n_elem // 2 : rope_n_elem]

    cos_gpu = cos_gpu.reshape(1, T, rope_n_elem)
    cos1_gpu = cos_gpu[..., : rope_n_elem // 2]
    cos2_gpu = cos_gpu[..., rope_n_elem // 2 :]

    sin_gpu = sin_gpu.reshape(1, T, rope_n_elem)
    sin1_gpu = sin_gpu[..., : rope_n_elem // 2]
    sin2_gpu = sin_gpu[..., rope_n_elem // 2 :]

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    g, uids = create_rope_graph(
        cudnn_handle, x1_gpu, x2_gpu, cos1_gpu, cos2_gpu, sin1_gpu, sin2_gpu
    )
    x1_uid, x2_uid, sin1_uid, sin2_uid, cos1_uid, cos2_uid, Y1_uid, Y2_uid = uids

    workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)

    g.execute(
        {
            x1_uid: x1_gpu,
            x2_uid: x2_gpu,
            sin1_uid: sin1_gpu,
            sin2_uid: sin2_gpu,
            cos1_uid: cos1_gpu,
            cos2_uid: cos2_gpu,
            Y1_uid: x1_gpu,
            Y2_uid: x2_gpu,
        },
        workspace,
        handle=cudnn_handle,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(Y_expected, x_gpu, atol=1e-2, rtol=1e-2)
