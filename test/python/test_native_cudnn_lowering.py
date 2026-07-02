"""GPU parity: NativeGraph builds natively, lowers to cuDNN, executes correctly.

Covers the native -> _lower_to_cpp -> cuDNN execute path (uid propagation, handle
threading, pointwise dispatch). Skipped without a GPU / cuDNN.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import cudnn
from cudnn.graph_native import NativeGraph

pytestmark = pytest.mark.L0

M, K, N = 64, 32, 48


def _handle():
    return cudnn.create_handle()


def test_native_matmul_lowers_to_cudnn():
    h = _handle()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    g = NativeGraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.HALF)
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1], data_type=cudnn.data_type.HALF)
    C = g.matmul(A, B)
    C.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c}, ws, handle=h)
    torch.cuda.synchronize()

    torch.testing.assert_close(c.float(), a.float() @ b.float(), atol=2e-2, rtol=2e-2)


def test_native_matmul_bias_relu_lowers_to_cudnn():
    h = _handle()
    a = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(1, K, N, device="cuda", dtype=torch.float16)
    bias = torch.randn(1, M, N, device="cuda", dtype=torch.float16)
    c = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    g = NativeGraph(handle=h, io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(dim=[1, M, K], stride=[M * K, K, 1], data_type=cudnn.data_type.HALF)
    B = g.tensor(dim=[1, K, N], stride=[K * N, N, 1], data_type=cudnn.data_type.HALF)
    Bi = g.tensor(dim=[1, M, N], stride=[M * N, N, 1], data_type=cudnn.data_type.HALF)
    Y = g.relu(g.bias(g.matmul(A, B), Bi))
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)

    g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, Bi: bias, Y: c}, ws, handle=h)
    torch.cuda.synchronize()

    torch.testing.assert_close(c.float(), torch.relu(a.float() @ b.float() + bias.float()), atol=2e-2, rtol=2e-2)
