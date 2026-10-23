"""Complete standalone Python sample.
API page: docs/api/python/sin.md
API: sin
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
    x = torch.randn((2, 3, 4), device="cuda", dtype=torch.float32)
    with cudnn.Graph(
        io_data_type=torch.float32,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=["X"],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        X = graph.tensor(name="X", dim=x.shape, stride=x.stride())
        Y = graph.sin(input=X, name="sin")
        Y.set_output(True).set_name("Y")
    y = graph(x)
    print(f"sin output: shape={tuple(y.shape)}, dtype={y.dtype}")


if __name__ == "__main__":
    run_sample(sample)
