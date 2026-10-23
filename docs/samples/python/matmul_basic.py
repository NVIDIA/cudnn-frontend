"""Complete standalone Python sample.
API page: docs/api/python/matmul.md
API: matmul
Use case: Basic
Source: adapted from `samples/llm_coverage/test_swiglu_layer_1+bs+d.py`.
"""

import cudnn
import torch


def sample():
    torch.manual_seed(0)
    x = torch.randn((2, 4, 16), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((1, 16, 8), device="cuda", dtype=torch.bfloat16)
    with cudnn.Graph(
        io_data_type=torch.bfloat16,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=[x, w],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        Y = graph.matmul(A=x, B=w, compute_data_type=torch.float32, name="matmul")
        Y.set_output(True).set_name("Y").set_data_type(torch.bfloat16)
    y = graph(x, w)
    expected = torch.matmul(x.float(), w.float()).to(torch.bfloat16)
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)
    print(f"matmul result matches reference: shape={tuple(y.shape)}")


if __name__ == "__main__":
    sample()
