"""Test cuDNN matmul+pointwise swish support surface and tieout with PyTorch implementation"""

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
def get_cudnn_matmul_swish(x_dim, x_stride, w_dim, w_stride, dtype):
    """As the replacement for PyTorch nn.Linear module + F.silu(). To compute Y = swish(X @ W).
    Create a cudnn graph for matmul between a tensor "x" of shape (1, m, n) and a
    tensor "w" of shape (1, n, k) to produce a tensor "y" of shape (1, m, k).
    """
    print(f"graph: x_dim: {x_dim}, x_stride: {x_stride}, dtype: {dtype}")
    print(f"graph: w_dim: {w_dim}, w_stride: {w_stride}")
    bx, m, nx = x_dim
    bw, nw, k = w_dim
    assert bx == bw, "x.shape[0] != w.shape[0]"
    assert bx == 1, "x.shape[0] != 1"
    assert nx == nw, "x.shape[2] != w.shape[1]"
    with cudnn.Graph(
        io_data_type=dtype,
        compute_data_type=torch.float32,
        intermediate_data_type=torch.float32,
        inputs=["X", "W"],
        outputs=["Y", "XW"],
        handle="auto",
    ) as graph:
        X = graph.tensor(name="X", dim=x_dim, stride=x_stride)
        W = graph.tensor(name="W", dim=w_dim, stride=w_stride)
        XW = graph.matmul(name="mm", A=X, B=W)
        Y = graph.swish(name="swish", input=XW)
        XW.set_output(True).set_name("XW")
        Y.set_output(True).set_name("Y")
    return graph


@functools.lru_cache(maxsize=None)
def get_cudnn_matmul_swish_backward(grad_dim, grad_stride, xw_dim, xw_stride, xT_dim, xT_stride, wT_dim, wT_stride, dtype):
    """To compute the backward of Y = swish(X @ W).

    dX = dY @ diag(swish_backward(X @ W)) @ W.T
    dW = X.T @ (dY @ diag(swish_backward(X @ W)))
    """
    print(f"graph: grad_dim: {grad_dim}, grad_stride: {grad_stride}, dtype: {dtype}")
    print(f"graph: xw_dim: {xw_dim}, xw_stride: {xw_stride}")
    print(f"graph: xT_dim: {xT_dim}, xT_stride: {xT_stride}")
    print(f"graph: wT_dim: {wT_dim}, wT_stride: {wT_stride}")
    with cudnn.Graph(
        io_data_type=dtype,
        compute_data_type=torch.float32,
        intermediate_data_type=torch.float32,
        inputs=["grad", "XW", "xT", "wT"],
        outputs=["dX", "dW"],
        handle="auto",
    ) as graph:
        dY = graph.tensor(name="grad", dim=grad_dim, stride=grad_stride)
        XW = graph.tensor(name="XW", dim=xw_dim, stride=xw_stride)
        WT = graph.tensor(name="wT", dim=wT_dim, stride=wT_stride)
        XT = graph.tensor(name="xT", dim=xT_dim, stride=xT_stride)

        swish_bwd = graph.swish_backward(name="swish_bwd", input=XW, loss=dY)
        dYXW = graph.mul(name="mul", a=dY, b=swish_bwd)
        dX = graph.matmul(name="mmdx", A=dYXW, B=WT)
        dW = graph.matmul(name="mmdw", A=XT, B=dYXW)
        dX.set_output(True).set_name("dX")
        dW.set_output(True).set_name("dW")
    return graph


class CudnnMatmulSwish(Function):
    """cuDNN matmul + Swish as a drop-in replacement for PyTorch nn.Linear module + F.silu()."""

    @staticmethod
    def forward(x, w):
        """Matmul + Swish function: y = swish(x @ w)

        x is a 3D tensor of shape (batch_size, seq_length, in_features)
        w is a 2D tensor of shape (out_features, in_features)
        y is a 3D tensor of shape (batch_size, seq_length, out_features)
        """
        b, s, m = x.shape
        n, wm = w.shape
        assert m == wm, "x.shape[1] != w.shape[1]"
        assert x.dtype == w.dtype, "x.dtype != w.dtype"
        print(f"CudnnMatmulSwish: x.shape: {x.shape}, w.shape: {w.shape}, x.dtype: {x.dtype}, w.dtype: {w.dtype}")
        X = x.view(1, b * s, m)
        W = w.T.unsqueeze(0)
        graph = get_cudnn_matmul_swish(
            tuple(X.shape),
            tuple(X.stride()),
            tuple(W.shape),
            tuple(W.stride()),
            x.dtype,
        )
        assert X.device == W.device, "Both X and W should be on the same device"
        print("x.view(1, b*s, m):", X.shape, X.stride())
        print("w.T.unsqueeze(0):", W.shape, W.stride())
        y, xw = graph(X, W)
        y = y.view(b, s, n)
        return y, xw

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, w = inputs
        y, xw = output
        ctx.save_for_backward(x, w, y, xw)

    @staticmethod
    def backward(ctx, dY, _dXW):
        x, w, y, xw = ctx.saved_tensors
        # collect shapes and check for consistency
        xb, xs, xm = x.shape
        wn, wm = w.shape
        yb, ys, yn = y.shape
        xwb, xws, xwn = xw.shape
        assert yn == wn, "y.shape[2] != w.shape[0]"
        assert wm == xm, "x.shape[2] != w.shape[1]"
        assert yb == xb, "y.shape[0] != x.shape[0]"
        assert ys == xs, "y.shape[1] != x.shape[1]"
        assert xwb == 1, "xw.shape[0] != 1"
        assert xws == yb * ys, "xw.shape[1] != y.shape[0] * y.shape[1]"
        assert xwn == yn, "xw.shape[2] != y.shape[2]"
        # transpose dy, x and w
        dY = dY.view(1, yb * ys, yn)
        XT = x.view(1, xb * xs, xm).transpose(1, 2)
        WT = w.T.unsqueeze(0).transpose(1, 2)
        # Compute grads for both dX and dW
        # dX = dY @ diag(swish_backward(XW)) @ W.T
        # dW = X.T @ (dY @ diag(swish_backward(XW)))
        swish_bwd = get_cudnn_matmul_swish_backward(
            tuple(dY.shape),
            tuple(dY.stride()),
            tuple(xw.shape),
            tuple(xw.stride()),
            tuple(XT.shape),
            tuple(XT.stride()),
            tuple(WT.shape),
            tuple(WT.stride()),
            x.dtype,
        )
        dX, dW = swish_bwd(dY, xw, XT, WT)
        dX = dX.view(xb, xs, xm)
        dW = dW.view(wn, wm)
        return dX, dW


class LinearSwish(nn.Module):
    """Drop-in replacement for PyTorch nn.Linear module + F.silu() to use cuDNN matmul+pointwise swish
    As in the example of LlamaModel, no bias term. The input tensor x is in
    shape (batch_size, seq_length, in_features) and the output tensor y should
    be in shape (batch_size, seq_length, out_features)
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        assert bias is False, "Requires bias=False in Llama"
        # PyTorch Linear is y = x @ W.T with x in shape (batch_size, in_features) and y in shape (batch_size, out_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def __repr__(self):
        out_features, in_features = self.weight.shape
        bias = False
        return f"LinearSwish(in_features={in_features}, out_features={out_features}, bias={bias})"

    def forward(self, x):
        print(f"Linear: x.shape: {x.shape}, weight.shape: {self.weight.shape}, x.dtype: {x.dtype}, weight.dtype: {self.weight.dtype}")
        return CudnnMatmulSwish.apply(x, self.weight)


def compare_linear_swish(dtype, backward=False):
    device = torch.device("cuda")
    torch.set_default_device(device)
    B, S, D, D2 = 3, 5, 28, 7
    native_x = torch.randn(B, S, D, requires_grad=True, dtype=dtype)
    cudnn_x = native_x.detach().clone().requires_grad_(True)
    native_linear = nn.Linear(D, D2, bias=False).to(dtype)
    cudnn_linear = LinearSwish(D, D2, bias=False).to(dtype)
    cudnn_linear.load_state_dict(native_linear.state_dict())

    # forward pass
    native_output = F.silu(native_linear(native_x))
    cudnn_output, _ = cudnn_linear(cudnn_x)
    tensors = {
        "native_output": native_output.detach().clone(),
        "cudnn_output": cudnn_output.detach().clone(),
    }

    # backward pass: Graph not supported yet
    # cudnnGraphNotSupportedError: No valid engine configs for SWISH_BWD_MUL_Matmul_Matmul_
    if backward:
        y_gt = torch.randn_like(native_output)
        native_loss = F.mse_loss(native_output, y_gt)
        cudnn_loss = F.mse_loss(cudnn_output, y_gt)
        native_loss.backward()
        cudnn_loss.backward()

        native_x.retain_grad()
        cudnn_x.retain_grad()
        native_linear.weight.retain_grad()
        cudnn_linear.weight.retain_grad()
        tensors.update(
            {
                "native_x_grad": native_x.grad.detach().clone(),
                "cudnn_x_grad": cudnn_x.grad.detach().clone(),
                "native_w_grad": native_linear.weight.grad.detach().clone(),
                "cudnn_w_grad": cudnn_linear.weight.grad.detach().clone(),
            }
        )
    return tensors


def test_linear_swish_bfloat16():
    """Test linear projection+swish with bfloat16 data type for PyTest"""
    tensors = compare_linear_swish(torch.bfloat16, backward=False)
    assert tensors["native_output"].shape == tensors["cudnn_output"].shape
    assert tensors["native_output"].stride() == tensors["cudnn_output"].stride()
    assert tensors["native_output"].dtype == tensors["cudnn_output"].dtype
    close_cnt, num_el, max_diff = check_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3


def test_linear_swish_bfloat16_backward():
    """Test backward pass of linear projection+swish with bfloat16 data type for PyTest"""
    import pytest

    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _ = compare_linear_swish(torch.bfloat16, backward=True)


def test_linear_swish_float16():
    """Test linear projection+swish with float16 data type for PyTest"""
    tensors = compare_linear_swish(torch.float16, backward=False)
    assert tensors["native_output"].shape == tensors["cudnn_output"].shape
    assert tensors["native_output"].stride() == tensors["cudnn_output"].stride()
    assert tensors["native_output"].dtype == tensors["cudnn_output"].dtype
    close_cnt, num_el, max_diff = check_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3


def test_linear_swish_float16_backward():
    """Test backward pass of linear projection+swish with float16 data type for PyTest"""
    import pytest

    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _ = compare_linear_swish(torch.float16, backward=True)


def test_linear_swish_float32():
    """Test linear projection+swish with float32 data type for PyTest"""
    tensors = compare_linear_swish(torch.float32, backward=False)
    assert tensors["native_output"].shape == tensors["cudnn_output"].shape
    assert tensors["native_output"].stride() == tensors["cudnn_output"].stride()
    assert tensors["native_output"].dtype == tensors["cudnn_output"].dtype
    close_cnt, num_el, max_diff = check_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3


def test_linear_swish_float32_backward():
    """Test backward pass of linear projection+swish with float32 data type for PyTest"""
    import pytest

    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _ = compare_linear_swish(torch.float32, backward=True)


if __name__ == "__main__":
    print(f"cuDNN version: {cudnn.backend_version()}")
    print()
    print("=" * 10, "bfloat16", "=" * 10)
    tensors = compare_linear_swish(torch.bfloat16, backward=False)
    print()
    print("native output:", tensors["native_output"].shape, tensors["native_output"].stride())
    print("cudnn output:", tensors["cudnn_output"].shape, tensors["cudnn_output"].stride())
    print("output closeness:", report_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3))
    print()

    print("=" * 10, "float16", "=" * 10)
    tensors = compare_linear_swish(torch.float16, backward=False)
    print()
    print("native output:", tensors["native_output"].shape, tensors["native_output"].stride())
    print("cudnn output:", tensors["cudnn_output"].shape, tensors["cudnn_output"].stride())
    print("output closeness:", report_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3))
    print()

    print("=" * 10, "float32", "=" * 10)
    tensors = compare_linear_swish(torch.float32, backward=False)
    print()
    print("native output:", tensors["native_output"].shape, tensors["native_output"].stride())
    print("cudnn output:", tensors["cudnn_output"].shape, tensors["cudnn_output"].stride())
    print("output closeness:", report_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3))
    print()
