"""Test cuDNN matmul support surface and tieout with PyTorch implementation"""

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
def get_cudnn_matmul(x_dim, x_stride, w_dim, w_stride, dtype):
    """As the replacement for PyTorch nn.Linear module. To compute Y = X @ W.
    Create a cudnn graph for matmul between a tensor "x" of shape (1, m, n) and a
    tensor "w" of shape (1, n, k) to produce a tensor "y" of shape (1, m, k).
    """
    print(f"graph: x_dim: {x_dim}, x_stride: {x_stride}, dtype: {dtype}")
    print(f"graph: w_dim: {w_dim}, w_stride: {w_stride}")
    with cudnn.Graph(
        io_data_type=dtype,
        compute_data_type=dtype,
        inputs=["X", "W"],
        outputs=["Y"],
        handle="auto",
    ) as graph:
        X = graph.tensor(name="X", dim=x_dim, stride=x_stride)
        W = graph.tensor(name="W", dim=w_dim, stride=w_stride)
        Y = graph.matmul(name="mm", A=X, B=W)
        Y.set_output(True).set_name("Y")
    return graph


class CudnnMatmul(Function):
    """cuDNN matmul as a drop-in replacement for PyTorch nn.Linear module"""

    @staticmethod
    def forward(x, w):
        """Matmul function: y = x @ w

        x is a 3D tensor of shape (batch_size, seq_length, in_features)
        w is a 2D tensor of shape (out_features, in_features)
        y is a 3D tensor of shape (batch_size, seq_length, out_features)
        """
        b, s, m = x.shape
        n, wm = w.shape
        assert m == wm, "x.shape[1] != w.shape[1]"
        assert x.dtype == w.dtype, "x.dtype != w.dtype"
        print(f"CudnnMatmul: x.shape: {x.shape}, w.shape: {w.shape}, x.dtype: {x.dtype}, w.dtype: {w.dtype}")
        X = x.view(1, b * s, m)
        W = w.T.unsqueeze(0)
        graph = get_cudnn_matmul(
            tuple(X.shape),
            tuple(X.stride()),
            tuple(W.shape),
            tuple(W.stride()),
            x.dtype,
        )
        assert X.device == W.device, "Both X and W should be on the same device"
        print("x.view(1, b*s, m):", X.shape, X.stride())
        print("w.T.unsqueeze(0):", W.shape, W.stride())
        y = graph(X, W).view(b, s, n)
        return y

    @staticmethod
    def setup_context(ctx, inputs, _output):
        x, w = inputs
        ctx.save_for_backward(x, w)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward for matmul function Y = X @ W:
            dX = dY @ W.T
            dW = X.T @ dY

        grad_output is a 3D tensor of shape (batch_size, seq_length, out_features)
        X is a 3D tensor from ctx of shape (batch_size, seq_length, in_features)
        W is a 2D tensor from ctx of shape (out_features, in_features)
        """
        x, w = ctx.saved_tensors
        dx = dw = None
        # collect shapes and check for consistency
        xb, xs, xm = x.shape
        wn, wm = w.shape
        yb, ys, yn = grad_output.shape
        assert yn == wn, "grad_output.shape[2] != w.shape[0]"
        assert wm == xm, "x.shape[2] != w.shape[1]"
        assert yb == xb, "grad_output.shape[0] != x.shape[0]"
        assert ys == xs, "grad_output.shape[1] != x.shape[1]"
        # optimize for efficiency: compute grad only when needed
        if ctx.needs_input_grad[0]:
            # dx = grad_output @ w
            dY = grad_output.view(1, yb * ys, yn)
            W = w.unsqueeze(0)
            graph = get_cudnn_matmul(
                tuple(dY.shape),
                tuple(dY.stride()),
                tuple(W.shape),
                tuple(W.stride()),
                W.dtype,
            )
            dx = graph(dY, W).view(xb, xs, xm)
        if ctx.needs_input_grad[1]:
            # dw = grad_output.view(yb*ys, yn).T @ x.view(xb*xs, xm)
            dY = grad_output.view(yb * ys, yn).T.unsqueeze(0)
            X = x.view(1, xb * xs, xm)
            graph = get_cudnn_matmul(
                tuple(dY.shape),
                tuple(dY.stride()),
                tuple(X.shape),
                tuple(X.stride()),
                X.dtype,
            )
            dw = graph(dY, X).squeeze(0)
        return dx, dw


class Linear(nn.Module):
    """Drop-in replacement for PyTorch nn.Linear module to use cuDNN matmul
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
        return f"Linear(in_features={in_features}, out_features={out_features}, bias={bias})"

    def forward(self, x):
        print(f"Linear: x.shape: {x.shape}, weight.shape: {self.weight.shape}, x.dtype: {x.dtype}, weight.dtype: {self.weight.dtype}")
        return CudnnMatmul.apply(x, self.weight)


def compare_linear(dtype):
    device = torch.device("cuda")
    torch.set_default_device(device)
    B, S, D, D2 = 3, 5, 28, 7
    native_x = torch.randn(B, S, D, requires_grad=True, dtype=dtype)
    cudnn_x = native_x.detach().clone().requires_grad_(True)
    native_linear = nn.Linear(D, D2, bias=False).to(dtype)
    cudnn_linear = Linear(D, D2, bias=False).to(dtype)
    cudnn_linear.load_state_dict(native_linear.state_dict())

    # forward pass
    native_output = native_linear(native_x)
    cudnn_output = cudnn_linear(cudnn_x)
    tensors = {
        "native_output": native_output.detach().clone(),
        "cudnn_output": cudnn_output.detach().clone(),
    }

    # backward pass
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


def test_linear_bfloat16():
    """Test linear projection with bfloat16 data type for PyTest"""
    tensors = compare_linear(torch.bfloat16)
    assert tensors["native_output"].shape == tensors["cudnn_output"].shape
    assert tensors["native_output"].stride() == tensors["cudnn_output"].stride()
    assert tensors["native_output"].dtype == tensors["cudnn_output"].dtype
    assert tensors["native_x_grad"].shape == tensors["cudnn_x_grad"].shape
    assert tensors["native_x_grad"].stride() == tensors["cudnn_x_grad"].stride()
    assert tensors["native_x_grad"].dtype == tensors["cudnn_x_grad"].dtype
    assert tensors["native_w_grad"].shape == tensors["cudnn_w_grad"].shape
    assert tensors["native_w_grad"].stride() == tensors["cudnn_w_grad"].stride()
    assert tensors["native_w_grad"].dtype == tensors["cudnn_w_grad"].dtype
    close_cnt, num_el, max_diff = check_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_x_grad"], tensors["cudnn_x_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_w_grad"], tensors["cudnn_w_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3


def test_linear_float16():
    """Test linear projection with float16 data type for PyTest"""
    tensors = compare_linear(torch.float16)
    assert tensors["native_output"].shape == tensors["cudnn_output"].shape
    assert tensors["native_output"].stride() == tensors["cudnn_output"].stride()
    assert tensors["native_output"].dtype == tensors["cudnn_output"].dtype
    assert tensors["native_x_grad"].shape == tensors["cudnn_x_grad"].shape
    assert tensors["native_x_grad"].stride() == tensors["cudnn_x_grad"].stride()
    assert tensors["native_x_grad"].dtype == tensors["cudnn_x_grad"].dtype
    assert tensors["native_w_grad"].shape == tensors["cudnn_w_grad"].shape
    assert tensors["native_w_grad"].stride() == tensors["cudnn_w_grad"].stride()
    assert tensors["native_w_grad"].dtype == tensors["cudnn_w_grad"].dtype
    close_cnt, num_el, max_diff = check_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_x_grad"], tensors["cudnn_x_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_w_grad"], tensors["cudnn_w_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3


def test_linear_float32():
    """Test linear projection with float32 data type for PyTest"""
    tensors = compare_linear(torch.float32)
    assert tensors["native_output"].shape == tensors["cudnn_output"].shape
    assert tensors["native_output"].stride() == tensors["cudnn_output"].stride()
    assert tensors["native_output"].dtype == tensors["cudnn_output"].dtype
    assert tensors["native_x_grad"].shape == tensors["cudnn_x_grad"].shape
    assert tensors["native_x_grad"].stride() == tensors["cudnn_x_grad"].stride()
    assert tensors["native_x_grad"].dtype == tensors["cudnn_x_grad"].dtype
    assert tensors["native_w_grad"].shape == tensors["cudnn_w_grad"].shape
    assert tensors["native_w_grad"].stride() == tensors["cudnn_w_grad"].stride()
    assert tensors["native_w_grad"].dtype == tensors["cudnn_w_grad"].dtype
    close_cnt, num_el, max_diff = check_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_x_grad"], tensors["cudnn_x_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_w_grad"], tensors["cudnn_w_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 1e-3


if __name__ == "__main__":
    print(f"cuDNN version: {cudnn.backend_version()}")
    print()
    print("=" * 10, "bfloat16", "=" * 10)
    tensors = compare_linear(torch.bfloat16)
    print()
    print("native output:", tensors["native_output"].shape, tensors["native_output"].stride())
    print("cudnn output:", tensors["cudnn_output"].shape, tensors["cudnn_output"].stride())
    print("output closeness:", report_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3))
    print()
    print(
        "native x grad:",
        tensors["native_x_grad"].shape,
        tensors["native_x_grad"].stride(),
        tensors["native_x_grad"].dtype,
    )
    print("cudnn x grad:", tensors["cudnn_x_grad"].shape, tensors["cudnn_x_grad"].stride(), tensors["cudnn_x_grad"].dtype)
    print("x grad closeness:", report_close(tensors["native_x_grad"], tensors["cudnn_x_grad"], atol=2e-3, rtol=2e-3))
    print()
    print(
        "native w grad:",
        tensors["native_w_grad"].shape,
        tensors["native_w_grad"].stride(),
        tensors["native_w_grad"].dtype,
    )
    print("cudnn w grad:", tensors["cudnn_w_grad"].shape, tensors["cudnn_w_grad"].stride(), tensors["cudnn_w_grad"].dtype)
    print("w grad closeness:", report_close(tensors["native_w_grad"], tensors["cudnn_w_grad"], atol=2e-3, rtol=2e-3))
    print()

    print("=" * 10, "float16", "=" * 10)
    tensors = compare_linear(torch.float16)
    print()
    print("native output:", tensors["native_output"].shape, tensors["native_output"].stride())
    print("cudnn output:", tensors["cudnn_output"].shape, tensors["cudnn_output"].stride())
    print("output closeness:", report_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3))
    print()
    print(
        "native x grad:",
        tensors["native_x_grad"].shape,
        tensors["native_x_grad"].stride(),
        tensors["native_x_grad"].dtype,
    )
    print("cudnn x grad:", tensors["cudnn_x_grad"].shape, tensors["cudnn_x_grad"].stride(), tensors["cudnn_x_grad"].dtype)
    print("x grad closeness:", report_close(tensors["native_x_grad"], tensors["cudnn_x_grad"], atol=2e-3, rtol=2e-3))
    print()
    print(
        "native w grad:",
        tensors["native_w_grad"].shape,
        tensors["native_w_grad"].stride(),
        tensors["native_w_grad"].dtype,
    )
    print("cudnn w grad:", tensors["cudnn_w_grad"].shape, tensors["cudnn_w_grad"].stride(), tensors["cudnn_w_grad"].dtype)
    print("w grad closeness:", report_close(tensors["native_w_grad"], tensors["cudnn_w_grad"], atol=2e-3, rtol=2e-3))
    print()

    print("=" * 10, "float32", "=" * 10)
    tensors = compare_linear(torch.float32)
    print()
    print("native output:", tensors["native_output"].shape, tensors["native_output"].stride())
    print("cudnn output:", tensors["cudnn_output"].shape, tensors["cudnn_output"].stride())
    print("output closeness:", report_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3))
    print()
    print(
        "native x grad:",
        tensors["native_x_grad"].shape,
        tensors["native_x_grad"].stride(),
        tensors["native_x_grad"].dtype,
    )
    print("cudnn x grad:", tensors["cudnn_x_grad"].shape, tensors["cudnn_x_grad"].stride(), tensors["cudnn_x_grad"].dtype)
    print("x grad closeness:", report_close(tensors["native_x_grad"], tensors["cudnn_x_grad"], atol=2e-3, rtol=2e-3))
    print()
    print(
        "native w grad:",
        tensors["native_w_grad"].shape,
        tensors["native_w_grad"].stride(),
        tensors["native_w_grad"].dtype,
    )
    print("cudnn w grad:", tensors["cudnn_w_grad"].shape, tensors["cudnn_w_grad"].stride(), tensors["cudnn_w_grad"].dtype)
    print("w grad closeness:", report_close(tensors["native_w_grad"], tensors["cudnn_w_grad"], atol=2e-3, rtol=2e-3))
    print()
