"""Complete standalone Python sample.
API page: docs/api/python/reduction.md
API: reduction
Use case: Basic
Source: adapted from `test/pycudnnTest/python_graph_defs/basic_tests.py`.
"""

import cudnn
import torch


def sample():
    x = torch.randn((10, 8, 4, 4), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )
    with cudnn.Graph(
        io_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=[x],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        Y = graph.reduction(
            input=x,
            mode=cudnn.reduction_mode.ADD,
            compute_data_type=torch.float32,
            name="reduction",
        )
        Y.set_output(True).set_name("Y").set_dim((10, 1, 4, 4)).set_stride((16, 16, 4, 1))

    y = graph(x)
    torch.testing.assert_close(y, x.sum(dim=1, keepdim=True))
    print(f"reduction output: shape={tuple(y.shape)}, values={y.flatten().tolist()}")


if __name__ == "__main__":
    sample()
