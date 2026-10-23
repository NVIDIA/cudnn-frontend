"""Complete standalone Python sample.
API page: docs/api/python/logical_or.md
API: logical_or
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
    a = torch.rand((2, 3, 8), device="cuda") > 0.5
    b = torch.rand((2, 3, 8), device="cuda") > 0.5
    with cudnn.Graph(io_data_type=torch.bool, inputs=["A", "B"], outputs=["Y"], handle="auto") as graph:
        A = graph.tensor(name="A", dim=a.shape, stride=a.stride(), data_type=cudnn.data_type.BOOLEAN)
        B = graph.tensor(name="B", dim=b.shape, stride=b.stride(), data_type=cudnn.data_type.BOOLEAN)
        Y = graph.logical_or(a=A, b=B, compute_data_type=cudnn.data_type.BOOLEAN, name="logical_or")
        Y.set_output(True).set_name("Y").set_data_type(cudnn.data_type.BOOLEAN)
    y = graph(a, b)
    print(f"logical_or output: shape={tuple(y.shape)}, dtype={y.dtype}")


if __name__ == "__main__":
    run_sample(sample)
