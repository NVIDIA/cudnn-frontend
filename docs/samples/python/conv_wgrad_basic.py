"""Complete standalone Python sample.
API page: docs/api/python/conv_wgrad.md
API: conv_wgrad
Use case: Basic
Source: adapted from `test/python/test_conv_fuzzer.py`.
"""

import cudnn
import torch


def sample():
    torch.manual_seed(0)
    x = torch.randn((1, 3, 8, 8), device="cuda", dtype=torch.float32)
    dy = torch.randn((1, 4, 8, 8), device="cuda", dtype=torch.float32)
    with cudnn.Graph(
        io_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=[x, dy],
        outputs=["dW"],
        handle="auto",
    ) as graph:
        dw_desc = graph.conv_wgrad(
            image=x,
            loss=dy,
            padding=[1, 1],
            stride=[1, 1],
            dilation=[1, 1],
            compute_data_type=torch.float32,
            name="conv_wgrad",
        )
        dw_desc.set_output(True).set_name("dW").set_dim((4, 3, 3, 3)).set_stride(
            (27, 9, 3, 1)
        ).set_data_type(torch.float32)

    dw = graph(x, dy)
    expected = torch.nn.grad.conv2d_weight(x, (4, 3, 3, 3), dy, padding=1)
    torch.testing.assert_close(dw, expected, rtol=1e-4, atol=1e-4)
    print(f"conv_wgrad result matches reference: {tuple(dw.shape)}")


if __name__ == "__main__":
    sample()
