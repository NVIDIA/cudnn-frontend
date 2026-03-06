"""Test cuDNN Rotary Embedding support surface and tieout with PyTorch implementation"""

import functools

import cudnn
import torch
import torch.nn as nn
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
def get_cudnn_rope(x1_dim, x1_stride, x2_dim, x2_stride, cos_dim, cos_stride, sin_dim, sin_stride, dtype):
    """Create a cudnn graph for RoPE operation."""
    print(f"graph: x1_dim: {x1_dim}, x1_stride: {x1_stride}, dtype: {dtype}")
    print(f"graph: x2_dim: {x2_dim}, x2_stride: {x2_stride}, dtype: {dtype}")
    print(f"graph: cos_dim: {cos_dim}, cos_stride: {cos_stride}, dtype: {dtype}")
    print(f"graph: sin_dim: {sin_dim}, sin_stride: {sin_stride}, dtype: {dtype}")
    with cudnn.Graph(
        io_data_type=dtype,
        compute_data_type=torch.float32,
        intermediate_data_type=torch.float32,
        inputs=["x1", "x2", "cos", "sin"],
        outputs=["y1", "y2"],
        handle="auto",
    ) as graph:
        x1 = graph.tensor(name="x1", dim=x1_dim, stride=x1_stride)
        x2 = graph.tensor(name="x2", dim=x2_dim, stride=x2_stride)
        cos = graph.tensor(name="cos", dim=cos_dim, stride=cos_stride)
        sin = graph.tensor(name="sin", dim=sin_dim, stride=sin_stride)
        x1_cos = graph.mul(a=x1, b=cos)
        x2_sin = graph.mul(a=x2, b=sin)
        x2_cos = graph.mul(a=x2, b=cos)
        x1_sin = graph.mul(a=x1, b=sin)
        y1 = graph.sub(a=x1_cos, b=x2_sin)
        y2 = graph.add(a=x2_cos, b=x1_sin)
        y1.set_output(True).set_data_type(dtype).set_dim(x1_dim).set_stride(x1_stride).set_name("y1")
        y2.set_output(True).set_data_type(dtype).set_dim(x2_dim).set_stride(x2_stride).set_name("y2")
    return graph


class CudnnRopeFunction(Function):
    """cuDNN RoPE function"""

    @staticmethod
    def forward(x, cos, sin):
        """RoPE function: y = x1 * cos - x2 * sin, y2 = x2 * cos + x1 * sin

        x is a 4D tensor of shape (batch_size, seq_length, num_heads, head_dim)
        cos is a 2D tensor of shape (seq_length, head_dim/2)
        sin is a 2D tensor of shape (seq_length, head_dim/2)
        """
        _b, s, _h, d = x.shape
        cos_s, cos_d = cos.shape
        sin_s, sin_d = sin.shape
        print(
            f"CudnnRope: x.shape: {x.shape}, cos.shape: {cos.shape}, sin.shape: {sin.shape}, x.dtype: {x.dtype}, cos.dtype: {cos.dtype}, sin.dtype: {sin.dtype}"
        )
        assert d / 2 == cos_d == sin_d, "Requires: (x.shape[-1] / 2) == cos.shape[1] == sin.shape[1]"
        assert cos_s == sin_s == s, "Requires: cos.shape[0] == sin.shape[0] == seq_length"
        assert x.dtype == cos.dtype == sin.dtype, "Requires: x.dtype == cos.dtype == sin.dtype"
        assert x.device == cos.device == sin.device, "x, cos and sin should be on the same device"
        cos = cos.view(1, s, 1, cos_d)
        sin = sin.view(1, s, 1, sin_d)
        y = torch.empty_like(x)
        x1, x2 = x.chunk(2, dim=-1)
        y1, y2 = y.chunk(2, dim=-1)
        graph = get_cudnn_rope(
            tuple(x1.shape),
            tuple(x1.stride()),
            tuple(x2.shape),
            tuple(x2.stride()),
            tuple(cos.shape),
            tuple(cos.stride()),
            tuple(sin.shape),
            tuple(sin.stride()),
            x.dtype,
        )
        graph(
            {
                "x1": x1,
                "x2": x2,
                "cos": cos,
                "sin": sin,
                "y1": y1,
                "y2": y2,
            }
        )
        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        # backward pass of RoPE is not supported
        pass

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass of cuDNN RoPE is not implemented")


class CudnnRopeModule(nn.Module):
    def __init__(self, cos, sin):
        super().__init__()
        self.cos = cos
        self.sin = sin

    def forward(self, x):
        return CudnnRopeFunction.apply(x, self.cos, self.sin)


def compare_rope(dtype):
    device = torch.device("cuda")
    torch.set_default_device(device)
    B, S, H, D = 3, 5, 8, 14
    # fake sin and cos for RoPE
    cos = torch.randn(S, D // 2, dtype=dtype)
    sin = torch.randn(S, D // 2, dtype=dtype)
    native_x = torch.randn(B, S, H, D, requires_grad=True, dtype=dtype)
    cudnn_x = native_x.detach().clone().requires_grad_(True)

    # PyTorch RoPE implementation
    # rotate half of the hidden dimensions, then mul and add
    x1, x2 = native_x.chunk(2, dim=-1)
    x1_roped = (x1 * cos.view(S, 1, D // 2)) - (x2 * sin.view(S, 1, D // 2))
    x2_roped = (x2 * cos.view(S, 1, D // 2)) + (x1 * sin.view(S, 1, D // 2))
    native_output = torch.cat((x1_roped, x2_roped), dim=-1)

    # cuDNN implementation
    rope_op = CudnnRopeModule(cos, sin)
    cudnn_output = rope_op(cudnn_x)
    tensors = {
        "native_output": native_output.detach().clone(),
        "cudnn_output": cudnn_output.detach().clone(),
    }
    return tensors


def test_rope_bfloat16():
    """Test RoPE with bfloat16 data type for PyTest"""
    import pytest

    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _ = compare_rope(torch.bfloat16)


def test_rope_float16():
    """Test RoPE with float16 data type for PyTest"""
    import pytest

    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _ = compare_rope(torch.float16)


def test_rope_float32():
    """Test RoPE with float32 data type for PyTest"""
    import pytest

    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _ = compare_rope(torch.float32)


if __name__ == "__main__":
    print(f"cuDNN version: {cudnn.backend_version()}")
    print()
    print("=" * 10, "bfloat16 (expected to fail)", "=" * 10)
    try:
        tensors = compare_rope(torch.bfloat16)
    except cudnn.cudnnGraphNotSupportedError as e:
        # No valid engine configs for MUL_MUL_MUL_MUL_SUB_ADD_
        print(e)
    print()

    print("=" * 10, "float16 (expected to fail)", "=" * 10)
    try:
        tensors = compare_rope(torch.float16)
    except cudnn.cudnnGraphNotSupportedError as e:
        # No valid engine configs for MUL_MUL_MUL_MUL_SUB_ADD_
        print(e)
    print()

    print("=" * 10, "float32 (expected to fail)", "=" * 10)
    try:
        tensors = compare_rope(torch.float32)
    except cudnn.cudnnGraphNotSupportedError as e:
        # No valid engine configs for MUL_MUL_MUL_MUL_SUB_ADD_
        print(e)
    print()
