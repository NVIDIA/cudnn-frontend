"""Complete standalone Python sample.
API page: docs/api/python/rope_backward.md
API: rope_backward
Use case: Basic
Source: adapted from `test/python/test_oss_rope.py`.
"""

import cudnn
import torch


def sample():
    B, H, S, D = 1, 2, 16, 64
    torch.manual_seed(0)
    dy = torch.randn((B, H, S, D), device="cuda", dtype=torch.bfloat16)
    freqs = torch.randn((S, 1, 1, D), device="cuda", dtype=torch.float32)
    with cudnn.Graph(
        io_data_type=torch.bfloat16,
        compute_data_type=torch.float32,
        inputs=[dy, freqs],
        outputs=["dX"],
        handle="auto",
    ) as graph:
        dx_desc = graph.rope_backward(dY=dy, freqs=freqs, name="rope_backward")
        dx_desc.set_output(True).set_name("dX").set_data_type(torch.bfloat16)

    dx = graph(dy, freqs)
    angles = freqs[:, 0, 0, : D // 2].reshape(1, 1, S, D // 2)
    dy1, dy2 = dy.float().chunk(2, dim=-1)
    expected = torch.cat(
        (dy1 * angles.cos() + dy2 * angles.sin(), dy2 * angles.cos() - dy1 * angles.sin()),
        dim=-1,
    ).to(torch.bfloat16)
    torch.testing.assert_close(dx, expected, rtol=3e-2, atol=3e-2)
    print(f"rope backward matches reference: {tuple(dx.shape)}")


if __name__ == "__main__":
    sample()
