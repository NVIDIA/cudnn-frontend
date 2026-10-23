"""Complete standalone Python sample.
API page: docs/api/python/build_operation_graph.md
API: build_operation_graph
Use case: Basic
Source: adapted from `test/python/test_cost_model.py`.
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
    handle = cudnn.create_handle()
    try:
        graph = cudnn.pygraph(
            handle=handle,
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        A = graph.tensor(
            name="A",
            dim=(2, 4, 16),
            stride=(64, 16, 1),
            data_type=cudnn.data_type.BFLOAT16,
        )
        B = graph.tensor(
            name="B",
            dim=(2, 16, 8),
            stride=(128, 8, 1),
            data_type=cudnn.data_type.BFLOAT16,
        )
        C = graph.matmul(A=A, B=B, compute_data_type=cudnn.data_type.FLOAT, name="matmul")
        C.set_output(True).set_name("C").set_dim((2, 4, 8)).set_stride((32, 8, 1)).set_data_type(cudnn.data_type.BFLOAT16)
        graph.validate()
        graph.build_operation_graph()
        print("build_operation_graph: ok")
    finally:
        cudnn.destroy_handle(handle)


if __name__ == "__main__":
    run_sample(sample)
