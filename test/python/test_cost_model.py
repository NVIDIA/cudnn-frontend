"""
Pytest coverage for the public cost-model API exposed via cudnn.pygraph.

Mirrors the C++ sample test in samples/cpp/matmul/matmuls.cpp
("Matmul estimate_run_times").

Run with:
    pytest test/python/test_cost_model.py -s -v
"""

import math

import cudnn
import pytest
import torch

from test_utils import torch_fork_set_rng


def _device_cc():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@pytest.mark.skipif(
    cudnn.backend_version() < 92400,
    reason="estimate_run_times requires cuDNN 9.24.0 or later",
)
@pytest.mark.skipif(_device_cc() < 86, reason="requires SM 8.6 or higher")
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_matmul_estimate_run_times(cudnn_handle):
    B, M, N, K = 16, 32, 64, 128

    A_gpu = torch.randn(B, M, K, requires_grad=False, device="cuda", dtype=torch.bfloat16)
    B_gpu = torch.randn(B, K, N, requires_grad=False, device="cuda", dtype=torch.bfloat16)

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    graph = cudnn.pygraph(handle=cudnn_handle)
    A = graph.tensor_like(A_gpu)
    B = graph.tensor_like(B_gpu)
    C = graph.matmul(name="matmul", A=A, B=B, compute_data_type=cudnn.data_type.FLOAT)
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()

    try:
        graph.create_execution_plans([cudnn.heur_mode.A])
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"TEST WAIVED: unsupported graph. {e}")

    times = graph.estimate_run_times()
    assert len(times) == graph.get_execution_plan_count()
    assert len(times) > 0

    finite = [t for t in times if math.isfinite(t)]
    print(f"[cost_model] predicted times (ms): {times}")
    assert len(finite) > 0, "no engine config produced a finite predicted time"
    for t in finite:
        assert t > 0.0, f"predicted time must be positive, got {t}"
