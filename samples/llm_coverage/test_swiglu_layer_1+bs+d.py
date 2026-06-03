"""Test cuDNN SwiGLU support surface and tieout with PyTorch implementation"""

import functools

import cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function


def check_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> tuple[int, int, float]:
    """Similar to torch.testing.assert_close, but counts the percentage of mismatches instead of
    raising an exception.
    """
    # find the positions where the actual and expected values are close
    close_mask = torch.isclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    # compute the percentage of close positions
    num_el = actual.numel()
    close_cnt = close_mask.detach().sum().cpu().item()
    # compute the max diff
    max_diff = (actual - expected).detach().abs().max().cpu().item()
    return close_cnt, num_el, max_diff


def report_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> str:
    """reports the percentage of mismatches like torch.testing.assert_close"""
    close_cnt, num_el, max_diff = check_close(actual, expected, atol, rtol)
    # print the results
    result = f"{100 * close_cnt / num_el:.1f}% close at atol={atol} rtol={rtol}, max diff={max_diff}"
    return result


@functools.lru_cache(maxsize=None)
def get_cudnn_swiglu(x_dim, x_stride, w1_dim, w1_stride, w2_dim, w2_stride, w3_dim, w3_stride, dtype):
    """Compute Y = ((X @ W1) . swish(X @ W2)) @ W3"""
    print(f"graph: x_dim: {x_dim}, x_stride: {x_stride}, dtype: {dtype}")
    print(f"graph: w1_dim: {w1_dim}, w1_stride: {w1_stride}, dtype: {dtype}")
    print(f"graph: w2_dim: {w2_dim}, w2_stride: {w2_stride}, dtype: {dtype}")
    print(f"graph: w3_dim: {w3_dim}, w3_stride: {w3_stride}, dtype: {dtype}")
    with cudnn.Graph(
        io_data_type=dtype,
        compute_data_type=torch.float32,
        intermediate_data_type=torch.float32,
        inputs=["X", "W1", "W2", "W3"],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        X = graph.tensor(name="X", dim=x_dim, stride=x_stride)
        W1 = graph.tensor(name="W1", dim=w1_dim, stride=w1_stride)
        W2 = graph.tensor(name="W2", dim=w2_dim, stride=w2_stride)
        W3 = graph.tensor(name="W3", dim=w3_dim, stride=w3_stride)
        Z1 = graph.matmul(name="mm1", A=X, B=W1)
        Z2 = graph.matmul(name="mm2", A=X, B=W2)
        swish = graph.swish(name="swish", input=Z2)
        Z3 = graph.mul(name="mul", a=Z1, b=swish)
        Y = graph.matmul(name="mm3", A=Z3, B=W3)
        Y.set_output(True).set_name("Y")
    return graph


class CudnnSwigluFunction(Function):
    @staticmethod
    def forward(x, w1, w2, w3):
        """Swiglu function: y = ((x @ w1) . swish(x @ w2)) @ w3
        As in the MLP sublayer of Llama model.

        x is a 3D tensor of shape (batch_size, seq_length, in_features)
        w1 is a 2D tensor of shape (out_features, in_features)
        w2 is a 2D tensor of shape (out_features, in_features)
        w3 is a 2D tensor of shape (in_features, out_features)
        y is a 3D tensor of shape (batch_size, seq_length, in_features)
        """
        b, s, m = x.shape
        n1, wm1 = w1.shape
        n2, wm2 = w2.shape
        wm3, n3 = w3.shape
        assert m == wm1, "x.shape[1] != w1.shape[1]"
        assert m == wm2, "x.shape[1] != w2.shape[1]"
        assert m == wm3, "x.shape[1] != w3.shape[0]"
        assert n1 == n2, "w1.shape[0] != w2.shape[0]"
        assert n1 == n3, "w1.shape[0] != w3.shape[1]"
        assert x.dtype == w1.dtype, "x.dtype != w1.dtype"
        assert x.dtype == w2.dtype, "x.dtype != w2.dtype"
        assert x.dtype == w3.dtype, "x.dtype != w3.dtype"
        print(
            f"CudnnSwiglu: x.shape: {x.shape}, w1.shape: {w1.shape}, w2.shape: {w2.shape}, w3.shape: {w3.shape}, "
            f"x.dtype: {x.dtype}, w1.dtype: {w1.dtype}, w2.dtype: {w2.dtype}, w3.dtype: {w3.dtype}"
        )
        X = x.view(1, b * s, m)
        W1 = w1.T.unsqueeze(0)
        W2 = w2.T.unsqueeze(0)
        W3 = w3.T.unsqueeze(0)
        graph = get_cudnn_swiglu(
            tuple(X.shape),
            tuple(X.stride()),
            tuple(W1.shape),
            tuple(W1.stride()),
            tuple(W2.shape),
            tuple(W2.stride()),
            tuple(W3.shape),
            tuple(W3.stride()),
            x.dtype,
        )
        assert X.device == W1.device, "Both X and W1 should be on the same device"
        assert X.device == W2.device, "Both X and W2 should be on the same device"
        assert X.device == W3.device, "Both X and W3 should be on the same device"
        print("x.view(1, b*s, m):", X.shape, X.stride())
        print("w1.T.unsqueeze(0):", W1.shape, W1.stride())
        print("w2.T.unsqueeze(0):", W2.shape, W2.stride())
        print("w3.T.unsqueeze(0):", W3.shape, W3.stride())
        y = graph(X, W1, W2, W3).view(b, s, m)
        return y

    @staticmethod
    def setup_context(ctx, inputs, _output):
        x, w1, w2, w3 = inputs
        ctx.save_for_backward(x, w1, w2, w3)

    @staticmethod
    def backward(ctx, _grad_output):
        raise NotImplementedError("Not implemented")


class CudnnSwiGLUModule(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        assert bias is False, "Requires bias=False in Llama"
        self.linear1 = nn.Parameter(torch.randn(out_features, in_features))
        self.linear2 = nn.Parameter(torch.randn(out_features, in_features))
        self.linear3 = nn.Parameter(torch.randn(in_features, out_features))

    def __repr__(self):
        out_features, in_features = self.linear1.shape
        bias = False
        return f"SwiGLU(in_features={in_features}, out_features={out_features}, bias={bias})"

    def forward(self, x):
        print(
            f"SwiGLU: x.shape: {x.shape}, linear1.shape: {self.linear1.shape}, linear2.shape: {self.linear2.shape}, "
            f"linear3.shape: {self.linear3.shape}, x.dtype: {x.dtype}, linear1.dtype: {self.linear1.dtype}, "
            f"linear2.dtype: {self.linear2.dtype}, linear3.dtype: {self.linear3.dtype}"
        )
        return CudnnSwigluFunction.apply(x, self.linear1, self.linear2, self.linear3)


class PyTorchSwiGLUModule(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        assert bias is False, "Requires bias=False in Llama"
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.linear3 = nn.Linear(out_features, in_features, bias=False)

    def forward(self, x):
        return self.linear3(self.linear1(x) * F.silu(self.linear2(x)))


def compare_swiglu(dtype):
    device = torch.device("cuda")
    torch.set_default_device(device)
    B, S, D, D2 = 3, 5, 28, 7
    native_x = torch.randn(B, S, D, requires_grad=True, dtype=dtype)
    cudnn_x = native_x.detach().clone().requires_grad_(True)
    cudnn_swiglu = CudnnSwiGLUModule(D, D2, bias=False).to(dtype)
    native_swiglu = PyTorchSwiGLUModule(D, D2, bias=False).to(dtype)
    state_dict_map = {
        "linear1.weight": "linear1",
        "linear2.weight": "linear2",
        "linear3.weight": "linear3",
    }
    cudnn_swiglu.load_state_dict({state_dict_map[k]: v for k, v in native_swiglu.state_dict().items()})

    native_output = native_swiglu(native_x)
    cudnn_output = cudnn_swiglu(cudnn_x)
    tensors = {
        "native_output": native_output.detach().clone(),
        "cudnn_output": cudnn_output.detach().clone(),
    }
    return tensors


def test_swiglu_bfloat16():
    """Test SwiGLU + linear projection with bfloat16 data type for PyTest"""
    import pytest

    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _ = compare_swiglu(torch.bfloat16)


def test_swiglu_float16():
    """Test SwiGLU + linear projection with float16 data type for PyTest"""
    import pytest

    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _ = compare_swiglu(torch.float16)


def test_swiglu_float32():
    """Test SwiGLU + linear projection with float32 data type for PyTest"""
    import pytest

    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _ = compare_swiglu(torch.float32)


if __name__ == "__main__":
    print(f"cuDNN version: {cudnn.backend_version()}")
    print()
    print("=" * 10, "bfloat16 (expected to fail)", "=" * 10)
    try:
        tensors = compare_swiglu(torch.bfloat16)
    except cudnn.cudnnGraphNotSupportedError as e:
        # No valid engine configs for Matmul_Matmul_SWISH_FWD_MUL_Matmul_
        print(e)
    print()

    print("=" * 10, "float16 (expected to fail)", "=" * 10)
    try:
        tensors = compare_swiglu(torch.float16)
    except cudnn.cudnnGraphNotSupportedError as e:
        # No valid engine configs for Matmul_Matmul_SWISH_FWD_MUL_Matmul_
        print(e)
    print()

    print("=" * 10, "float32 (expected to fail)", "=" * 10)
    try:
        tensors = compare_swiglu(torch.float32)
    except cudnn.cudnnGraphNotSupportedError as e:
        # No valid engine configs for Matmul_Matmul_SWISH_FWD_MUL_Matmul_
        print(e)
    print()
