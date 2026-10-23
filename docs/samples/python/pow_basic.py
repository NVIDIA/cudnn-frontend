"""Complete standalone Python sample.
API page: docs/api/python/pow.md
API: pow
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
    a = torch.rand((2, 3, 4), device="cuda", dtype=torch.float32) + 0.5
    b = torch.rand((2, 3, 4), device="cuda", dtype=torch.float32) + 0.5
    with cudnn.Graph(
        io_data_type=torch.float32,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=["A", "B"],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        A = graph.tensor(name="A", dim=a.shape, stride=a.stride())
        B = graph.tensor(name="B", dim=b.shape, stride=b.stride())
        Y = graph.pow(input0=A, input1=B, compute_data_type=cudnn.data_type.FLOAT, name="pow")
        Y.set_output(True).set_name("Y")
    y = graph(a, b)
    print(f"pow output: shape={tuple(y.shape)}, dtype={y.dtype}")


if __name__ == "__main__":
    run_sample(sample)
