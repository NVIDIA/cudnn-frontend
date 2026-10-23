"""Complete standalone Python sample.
API page: docs/api/python/logical_not.md
API: logical_not
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
    x = torch.rand((2, 3, 8), device="cuda") > 0.5
    with cudnn.Graph(io_data_type=torch.bool, inputs=["X"], outputs=["Y"], handle="auto") as graph:
        X = graph.tensor(name="X", dim=x.shape, stride=x.stride(), data_type=cudnn.data_type.BOOLEAN)
        Y = graph.logical_not(input=X, name="logical_not")
        Y.set_output(True).set_name("Y").set_data_type(cudnn.data_type.BOOLEAN)
    y = graph(x)
    print(f"logical_not output: shape={tuple(y.shape)}, dtype={y.dtype}")


if __name__ == "__main__":
    run_sample(sample)
