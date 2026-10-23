"""Complete standalone Python broadcasting sample.
API page: docs/api/python/add.md
API: add
Use case: Broadcasting
Source: adapted from `python/pygraph/pointwise.cpp`.
"""

import cudnn
import torch


def sample():
    a = torch.ones((2, 3, 4), device="cuda", dtype=torch.float32)
    b = torch.tensor([[[10.0], [20.0], [30.0]], [[10.0], [20.0], [30.0]]], device="cuda")
    with cudnn.Graph(
        io_data_type=torch.float32,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=[a, b],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        y_desc = graph.add(a=a, b=b, compute_data_type=torch.float32, name="broadcast_add")
        y_desc.set_output(True).set_name("Y")

    y = graph(a, b)
    torch.testing.assert_close(y, a + b)
    print(f"broadcast add output: shape={tuple(y.shape)}, values={y[0, :, 0].tolist()}")


if __name__ == "__main__":
    sample()
