"""Complete standalone Python sample.
API page: docs/api/python/binary_select.md
API: binary_select
Use case: Basic
Source: adapted from `python/pygraph/pointwise.cpp`.
"""

import sys

try:
    import cudnn
    import torch
except Exception as exc:
    print(f"SKIP: could not import cudnn or torch: {exc}")
    raise SystemExit(0)


def run_sample(sample):
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available to PyTorch")
        return
    try:
        sample()
    except Exception as exc:
        if "--strict" in sys.argv:
            raise
        print(f"SKIP: sample could not run on this installation: {exc}")


def sample():
    x0 = torch.randn((2, 3, 8), device="cuda", dtype=torch.float32)
    x1 = torch.zeros_like(x0)
    mask = torch.rand_like(x0) > 0.5
    with cudnn.Graph(
        io_data_type=torch.float32,
        inputs=["X0", "X1", "mask"],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        X0 = graph.tensor(name="X0", dim=x0.shape, stride=x0.stride())
        X1 = graph.tensor(name="X1", dim=x1.shape, stride=x1.stride())
        M = graph.tensor(
            name="mask",
            dim=mask.shape,
            stride=mask.stride(),
            data_type=torch.bool,
        )
        Y = graph.binary_select(input0=X0, input1=X1, mask=M, name="binary_select")
        Y.set_output(True).set_name("Y")
    y = graph(x0, x1, mask)
    print(f"binary_select output: shape={tuple(y.shape)}, dtype={y.dtype}")


if __name__ == "__main__":
    run_sample(sample)
