"""Test cuDNN RMSNorm support surface and tieout with PyTorch implementation"""

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
def get_cudnn_rmsnorm_fwd(batch_size, seq_len, hidden_dim, dtype):
    """As the replacement for PyTorch nn.RMSNorm module.

    Execute cuDNN RMSNorm forward pass with tensor shape (B*S, D)
    """
    print(f"rmsnorm fwd: bs={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}, dtype: {dtype}")
    with cudnn.Graph(
        io_data_type=dtype,
        compute_data_type=torch.float32,
        inputs=["x", "scale", "epsilon"],
        outputs=["out", "invvar"],
        handle="auto",
    ) as graph:
        x_gpu = graph.tensor(name="x", dim=(batch_size * seq_len, hidden_dim), stride=(hidden_dim, 1))
        scale_gpu = graph.tensor(name="scale", dim=(1, hidden_dim), stride=(hidden_dim, 1))
        eps_cpu = graph.tensor(name="epsilon", dim=(1, 1), stride=(1, 1), is_pass_by_value=True)
        out, inv_var = graph.rmsnorm(
            norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
            input=x_gpu,
            scale=scale_gpu,
            epsilon=eps_cpu,
        )
        # set output, inv_var must be float32 tensor
        out.set_output(True).set_name("out")
        inv_var.set_output(True).set_name("invvar")
    return graph


@functools.lru_cache(maxsize=None)
def get_cudnn_rmsnorm_bwd(batch_size, seq_len, hidden_dim, dtype):
    """As the replacement for PyTorch nn.RMSNorm module

    Execute cuDNN RMSNorm backward pass with tensor shape (B*S, D)
    """
    print(f"rmsnorm bwd: bs={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}, dtype: {dtype}")
    with cudnn.Graph(
        io_data_type=dtype,
        compute_data_type=torch.float32,
        inputs=["grad", "x", "invvar", "scale"],
        outputs=["dx", "dscale"],
        handle="auto",
    ) as graph:
        grad_gpu = graph.tensor(name="grad", dim=(batch_size * seq_len, hidden_dim), stride=(hidden_dim, 1))
        x_gpu = graph.tensor(name="x", dim=(batch_size * seq_len, hidden_dim), stride=(hidden_dim, 1))
        invvar_gpu = graph.tensor(name="invvar", dim=(batch_size * seq_len, 1), stride=(1, 1))
        scale_gpu = graph.tensor(name="scale", dim=(1, hidden_dim), stride=(hidden_dim, 1))
        dx, dscale, dbias = graph.rmsnorm_backward(
            grad=grad_gpu,
            input=x_gpu,
            inv_variance=invvar_gpu,
            scale=scale_gpu,
            has_dbias=False,
        )
        # set output, inv_var must be float32 tensor
        dx.set_output(True).set_data_type(dtype).set_name("dx")
        dscale.set_output(True).set_data_type(dtype).set_name("dscale")
        assert dbias is None, "requested has_dbias=False, but dbias is not None"
    return graph


class CudnnRmsNorm(Function):
    """cuDNN rmsnorm as a drop-in replacement for PyTorch nn.RMSNorm module"""

    @staticmethod
    def forward(x, scale, eps):
        """RMS norm function: y = scale * x / sqrt(x^2 + eps)

        x and y are 3D tensors of shape (batch_size, seq_length, hidden_dim)
        scale is a 1D tensor of shape (hidden_dim,)
        eps is a 2D tensor of shape (1,1) holding the epsilon value
        """
        b, s, h = x.shape
        assert scale.shape == (h,), "scale.shape != (hidden_dim,)"
        assert eps.shape == (1, 1), "eps.shape != (1,1)"
        assert x.dtype == scale.dtype, "x.dtype != scale.dtype"
        print(f"CudnnRmsFwd: x: {x.shape}, scale: {scale.shape}, x.dtype: {x.dtype}")
        X = x.view(b * s, h)
        W = scale.unsqueeze(0)
        assert X.device == W.device, "Both X and W should be on the same device"
        graph = get_cudnn_rmsnorm_fwd(b, s, h, x.dtype)
        print("x.view(b*s, h):", X.shape, X.stride())
        print("w.unsqueeze(0):", W.shape, W.stride())
        y, inv_var = graph(X, W, eps)
        return y.view(b, s, h), inv_var

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, scale, _eps = inputs
        _y, inv_var = output
        ctx.save_for_backward(x, scale, inv_var)

    @staticmethod
    def backward(ctx, dy, _dinv_var):
        """Backward for rmsnorm function Y = scale * X / sqrt(X^2 + eps):

        grad_output, X, Y are 3D tensors of shape (batch_size, seq_length, hidden_dim)
        scale is a 1D tensor from ctx of shape (hidden_dim,)
        """
        x, scale, inv_var = ctx.saved_tensors
        dx = dscale = deps = None
        # collect shapes and check for consistency
        xb, xs, xh = x.shape
        wh = scale.shape[0]
        yb, ys, yh = dy.shape
        assert yh == wh, "grad_output.shape[2] != w.shape[0]"
        assert xh == wh, "x.shape[2] != w.shape[0]"
        assert yb == xb, "grad_output.shape[0] != x.shape[0]"
        assert ys == xs, "grad_output.shape[1] != x.shape[1]"
        # cuDNN compute all grads at once
        graph = get_cudnn_rmsnorm_bwd(xb, xs, xh, x.dtype)
        dY = dy.view(yb * ys, yh)
        X = x.view(xb * xs, xh)
        W = scale.unsqueeze(0)
        dx, dscale = graph(dY, X, inv_var, W)
        dx = dx.view(xb, xs, xh)
        dscale = dscale.squeeze(0)
        return dx, dscale, deps


class RMSNorm(nn.Module):
    """Drop-in replacement for PyTorch nn.RMSNorm module to use cuDNN rmsnorm
    The input tensor x is in shape (batch_size, seq_length, hidden_dim) and the
    output tensor y should be in the same shape
    """

    def __init__(
        self,
        normalized_shape,
        eps=None,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert eps is not None, "Requires eps in Llama"
        assert elementwise_affine is True, "Requires elementwise_affine=True in Llama"
        assert isinstance(normalized_shape, int), "normalized_shape must be an integer in Llama"
        # PyTorch Linear is y = x @ W.T with x in shape (batch_size, in_features) and y in shape (batch_size, out_features)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.eps_cpu = torch.full((1, 1), eps, dtype=torch.float32, device="cpu")
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def __repr__(self):
        normalized_shape = tuple(self.weight.shape)
        eps = self.eps
        elementwise_affine = self.elementwise_affine
        return f"RMSNorm(normalized_shape={normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine})"

    def forward(self, x):
        print(f"RMSNorm: x.shape: {x.shape}, weight.shape: {self.weight.shape}, " f"x.dtype: {x.dtype}, weight.dtype: {self.weight.dtype}")
        y, _inv_var = CudnnRmsNorm.apply(x, self.weight, self.eps_cpu)
        return y


def compare_rmsnorm(dtype):
    device = torch.device("cuda")
    torch.set_default_device(device)
    B, S, D = 3, 5, 7
    native_x = torch.randn(B, S, D, requires_grad=True, dtype=dtype)
    cudnn_x = native_x.detach().clone().requires_grad_(True)
    native_rmsnorm = nn.RMSNorm(D, eps=1e-5).to(dtype)
    cudnn_rmsnorm = RMSNorm(D, eps=1e-5).to(dtype)
    cudnn_rmsnorm.load_state_dict(native_rmsnorm.state_dict())

    # forward pass
    native_output = native_rmsnorm(native_x)
    cudnn_output = cudnn_rmsnorm(cudnn_x)
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
    native_rmsnorm.weight.retain_grad()
    cudnn_rmsnorm.weight.retain_grad()
    tensors.update(
        {
            "native_x_grad": native_x.grad.detach().clone(),
            "cudnn_x_grad": cudnn_x.grad.detach().clone(),
            "native_w_grad": native_rmsnorm.weight.grad.detach().clone(),
            "cudnn_w_grad": cudnn_rmsnorm.weight.grad.detach().clone(),
        }
    )
    return tensors


def test_rmsnorm_bfloat16():
    """Test RMSNorm with bfloat16 data type for PyTest"""
    tensors = compare_rmsnorm(torch.bfloat16)
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
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_x_grad"], tensors["cudnn_x_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_w_grad"], tensors["cudnn_w_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3


def test_rmsnorm_float16():
    """Test RMSNorm with float16 data type for PyTest"""
    tensors = compare_rmsnorm(torch.float16)
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
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_x_grad"], tensors["cudnn_x_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_w_grad"], tensors["cudnn_w_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3


def test_rmsnorm_float32():
    """Test RMSNorm with float32 data type for PyTest"""
    tensors = compare_rmsnorm(torch.float32)
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
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_x_grad"], tensors["cudnn_x_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_w_grad"], tensors["cudnn_w_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3


if __name__ == "__main__":
    print(f"cuDNN version: {cudnn.backend_version()}")
    print()

    print("=" * 10, "bfloat16", "=" * 10)
    tensors = compare_rmsnorm(torch.bfloat16)
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
    tensors = compare_rmsnorm(torch.float16)
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
    tensors = compare_rmsnorm(torch.float32)
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
