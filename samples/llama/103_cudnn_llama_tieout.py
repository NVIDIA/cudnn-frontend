"""Build a concise PyTorch implementation of the Llama 3.1 8B model with cuDNN optimizations
that can seamlessly accept the LlamaModel model weight from Hugging Face.

This implementation follows the architecture described in the LLaMA paper:
"LLaMA: Open and Efficient Foundation Language Models" (https://arxiv.org/abs/2302.13971)
and "The Llama 3 Herd of Models" (https://arxiv.org/abs/2407.21783)

Key architectural features:
- Pre-normalization using RMSNorm
- SwiGLU activation in the MLP
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention for efficient computation

This implementation uses custom PyTorch functions using cuDNN for the use case of Llama model.
See <https://docs.pytorch.org/docs/stable/autograd.html#function> and
<https://docs.pytorch.org/docs/stable/notes/extending.html> for details in extending PyTorch
using torch.autograd.function.Function.
"""

import dataclasses
import functools
import math
import time
from typing import Any, Optional, Tuple

import cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn import Linear, RMSNorm   # Commented out. Using custom class as drop-in replacement
from torch.autograd.function import Function


# For type annotations
Tensor = torch.Tensor


def report_close(actual: Tensor, expected: Tensor, atol: float, rtol: float) -> str:
    """Similar to torch.testing.assert_close, but reports the percentage of mismatches instead of
    raising an exception.
    """
    # find the positions where the actual and expected values are close
    close_mask = torch.isclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    # compute the percentage of close positions
    num_el = actual.numel()
    close_cnt = close_mask.detach().sum().cpu().item()
    # compute the max diff
    max_diff = (actual - expected).detach().abs().max().cpu().item()
    # print the results
    result = f"{100 * close_cnt / num_el:.1f}% close at atol={atol} rtol={rtol}, max diff={max_diff}"
    return result


@functools.lru_cache(maxsize=None)
def get_cudnn_matmul(
    x_dim: Tuple[int, ...],
    x_stride: Tuple[int, ...],
    w_dim: Tuple[int, ...],
    w_stride: Tuple[int, ...],
    dtype: torch.dtype,
) -> cudnn.Graph:
    """For use in the replacement of PyTorch nn.Linear module. To compute Y = X @ W.
    Create a cuDNN graph for matmul between a tensor "x" of shape (1, m, n) and a
    tensor "w" of shape (1, n, k) to produce a tensor "y" of shape (1, m, k).
    """
    with cudnn.Graph(
        handle="auto",
        io_data_type=dtype,
        compute_data_type=dtype,
        inputs=["X", "W"],
        outputs=["Y"],
    ) as graph:
        X = graph.tensor(name="X", dim=x_dim, stride=x_stride)
        W = graph.tensor(name="W", dim=w_dim, stride=w_stride)
        Y = graph.matmul(name="mm", A=X, B=W)
        Y.set_output(True).set_name("Y")
    return graph


class CudnnMatmul(Function):
    """Custom PyTorch matmul function using cuDNN for the use case of Llama model"""

    @staticmethod
    def forward(x: Tensor, w: Tensor) -> Tensor:
        """Matmul function: Y = X @ W.T

        X is a 3D tensor of shape (batch_size, seq_length, in_features)
        W is a 2D tensor of shape (out_features, in_features)
        Y is a 3D tensor of shape (batch_size, seq_length, out_features)
        """
        b, s, m = x.shape
        n, wm = w.shape
        assert m == wm, "x.shape[1] != w.shape[1]"
        assert x.dtype == w.dtype, "x.dtype != w.dtype"
        X = x.view(1, b * s, m)
        W = w.T.unsqueeze(0)
        graph = get_cudnn_matmul(
            tuple(X.shape),
            tuple(X.stride()),
            tuple(W.shape),
            tuple(W.stride()),
            x.dtype,
        )
        y = graph(X, W).view(b, s, n)
        return y

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Tensor, Tensor], output: Tuple[Tensor, Tensor]) -> None:
        """Save tensors to help computing backward"""
        x, w = inputs
        ctx.save_for_backward(x, w)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward for matmul function Y = X @ W.T:
            dX = dY @ W.T
            dW = X.T @ dY

        grad_output = dY is a 3D tensor of shape (batch_size, seq_length, out_features)
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
        # for compatibility with cuDNN, tensors need to be reshaped such that first dimension is 1
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
    For the use in LlamaModel, the input tensor x is in shape (batch_size, seq_length, in_features)
    output tensor y should be in shape (batch_size, seq_length, out_features)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        assert bias is False, "Requires bias=False in Llama"
        # PyTorch Linear is y = x @ W.T with x in shape (batch_size, in_features) and y in shape (batch_size, out_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        bias = False
        return f"Linear(in_features={in_features}, out_features={out_features}, bias={bias})"

    def forward(self, x: Tensor) -> Tensor:
        return CudnnMatmul.apply(x, self.weight)


@functools.lru_cache(maxsize=None)
def get_cudnn_rmsnorm_fwd(batch_size: int, seq_len: int, hidden_dim: int, dtype: torch.dtype) -> cudnn.Graph:
    """For use in the replacement of PyTorch nn.RMSNorm module. To compute RMS norm forward pass
    with scale and epsilon
    """
    with cudnn.Graph(
        handle="auto",
        io_data_type=dtype,
        compute_data_type=cudnn.data_type.FLOAT,
        inputs=["x", "scale", "epsilon"],
        outputs=["out", "invvar"],
    ) as graph:
        x_gpu = graph.tensor(name="x", dim=(batch_size * seq_len, hidden_dim), stride=(hidden_dim, 1))
        scale_gpu = graph.tensor(name="scale", dim=(1, hidden_dim), stride=(hidden_dim, 1))
        eps_cpu = graph.tensor(
            name="epsilon",
            dim=(1, 1),
            stride=(1, 1),
            data_type=cudnn.data_type.FLOAT,
            is_pass_by_value=True,
        )
        out, inv_var = graph.rmsnorm(
            norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
            input=x_gpu,
            scale=scale_gpu,
            epsilon=eps_cpu,
        )
        # set output, inv_var must be float32 tensor
        out.set_output(True).set_name("out")
        inv_var.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_name("invvar")
    return graph


@functools.lru_cache(maxsize=None)
def get_cudnn_rmsnorm_bwd(batch_size: int, seq_len: int, hidden_dim: int, dtype: torch.dtype) -> cudnn.Graph:
    """For use in the replacement of PyTorch nn.RMSNorm module. To compute RMS norm backward pass
    with scale
    """
    with cudnn.Graph(
        handle="auto",
        io_data_type=dtype,
        compute_data_type=cudnn.data_type.FLOAT,
        inputs=["grad", "x", "invvar", "scale"],
        outputs=["dx", "dscale"],
    ) as graph:
        grad_gpu = graph.tensor(name="grad", dim=(batch_size * seq_len, hidden_dim), stride=(hidden_dim, 1))
        x_gpu = graph.tensor(name="x", dim=(batch_size * seq_len, hidden_dim), stride=(hidden_dim, 1))
        invvar_gpu = graph.tensor(
            name="invvar",
            dim=(batch_size * seq_len, 1),
            stride=(1, 1),
            data_type=cudnn.data_type.FLOAT,
        )
        scale_gpu = graph.tensor(name="scale", dim=(1, hidden_dim), stride=(hidden_dim, 1))
        dx, dscale, dbias = graph.rmsnorm_backward(
            grad=grad_gpu,
            input=x_gpu,
            inv_variance=invvar_gpu,
            scale=scale_gpu,
            has_dbias=False,
        )
        # set outputs
        dx.set_output(True).set_data_type(dtype).set_name("dx")
        dscale.set_output(True).set_data_type(dtype).set_name("dscale")
        assert dbias is None, "requested has_dbias=False, but dbias is not None"
    return graph


class CudnnRmsNorm(Function):
    """Custom PyTorch RMS norm function using cuDNN for the use case of Llama model"""

    @staticmethod
    def forward(x: Tensor, scale: Tensor, eps: Tensor) -> Tuple[Tensor, Tensor]:
        """RMS norm function: y = scale * x / sqrt(x^2 + eps)

        x and y are 3D tensors of shape (batch_size, seq_length, hidden_dim)
        scale is a 1D tensor of shape (hidden_dim,)
        eps is a 2D tensor of shape (1,1) holding the epsilon value
        """
        b, s, h = x.shape
        assert scale.shape == (h,), "scale.shape != (hidden_dim,)"
        assert eps.shape == (1, 1), "eps.shape != (1,1)"
        assert eps.stride() == (1, 1), "eps.stride() != (1, 1)"
        assert x.dtype == scale.dtype, "x.dtype != scale.dtype"
        X = x.view(b * s, h)
        W = scale.unsqueeze(0)
        assert X.device == W.device, "Both X and W should be on the same device"
        assert X.stride() == (h, 1), "X.stride() != (hidden_dim, 1)"
        assert W.stride() == (h, 1), "W.stride() != (hidden_dim, 1)"
        graph = get_cudnn_rmsnorm_fwd(b, s, h, x.dtype)
        y, inv_var = graph(X, W, eps)
        return y.view(b, s, h), inv_var

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Tensor, Tensor, Tensor], output: Tuple[Tensor, Tensor]) -> None:
        """Save tensors to help computing backward"""
        x, scale, _eps = inputs
        _y, inv_var = output
        ctx.save_for_backward(x, scale, inv_var)

    @staticmethod
    def backward(ctx: Any, dy: Tensor, dinv_var: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Backward for RMS norm function y = scale * x / sqrt(x^2 + eps):

        grad_output (dy), x, y are 3D tensors of shape (batch_size, seq_length, hidden_dim)
        scale is a 1D tensor from ctx of shape (hidden_dim,)
        inv_var and dinv_var are 2D tensors of shape (batch_size*seq_length, 1)
        """
        x, scale, inv_var = ctx.saved_tensors
        dx = dscale = deps = None
        # collect shapes and check for consistency
        xb, xs, xh = x.shape
        wh = scale.shape[0]
        yb, ys, yh = dy.shape
        assert xh == wh, "x.shape[2] != w.shape[0]"
        assert dy.shape == (xb, xs, yh), "dy.shape != x.shape"
        assert inv_var.shape == (xb * xs, 1), "inv_var.shape != (batch_size*seq_length, 1)"
        assert inv_var.stride() == (1, 1), "inv_var.stride() != (1, 1)"
        # use cuDNN to compute all grads at once
        # tensors need to be reshaped to 2D to be compatible with cuDNN
        graph = get_cudnn_rmsnorm_bwd(xb, xs, xh, x.dtype)
        dY = dy.view(yb * ys, yh)
        assert dY.stride() == (yh, 1), "dY.stride() != (hidden_dim, 1)"
        X = x.view(xb * xs, xh)
        assert X.stride() == (xh, 1), "X.stride() != (hidden_dim, 1)"
        W = scale.unsqueeze(0)
        assert W.shape == (1, xh), "W.shape != (1, hidden_dim)"
        assert W.stride() == (xh, 1), "W.stride() != (hidden_dim, 1)"
        dx, dscale = graph(dY, X, inv_var, W)
        dx = dx.view(xb, xs, xh)
        dscale = dscale.squeeze(0)
        return dx, dscale, deps


class RMSNorm(nn.Module):
    """Drop-in replacement for PyTorch nn.RMSNorm module to use cuDNN RMS norm
    For the use in LlamaModel, the input and output tensors are in shape (batch_size, seq_length, hidden_dim)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float,
        elementwise_affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        assert eps is not None, "Requires eps in Llama"
        assert elementwise_affine is True, "Requires elementwise_affine=True in Llama"
        assert isinstance(normalized_shape, int), "normalized_shape must be an integer in Llama"
        # PyTorch RMSNorm is y = scale * x / sqrt(x^2 + eps) with x and y in shape (batch_size, seq_length, hidden_dim)
        # the scale tensor is a parameter of this module in shape (hidden_dim,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.eps_cpu = torch.full((1, 1), eps, dtype=torch.float32, device="cpu")
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def __repr__(self) -> str:
        normalized_shape = tuple(self.weight.shape)
        eps = self.eps
        elementwise_affine = self.elementwise_affine
        return f"RMSNorm(normalized_shape={normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine})"

    def forward(self, x: Tensor) -> Tensor:
        """Forward for RMS norm function y = scale * x / sqrt(x^2 + eps)

        While cuDNN computes the inv_var tensor, PyTorch RMSNorm does not. For
        compatibility with PyTorch, we return the output tensor y only.
        """
        y, inv_var = CudnnRmsNorm.apply(x, self.weight, self.eps_cpu)
        return y


@functools.lru_cache(maxsize=None)
def get_cudnn_gqa_fwd(
    batch_size: int, seq_len: int, heads_q: int, heads_kv: int, dim: int, dtype: torch.dtype
) -> cudnn.Graph:
    """For use in the replacement of PyTorch GQA function. To compute GQA forward pass
    with causal mask
    """
    attn_scale = float(dim) ** -0.5
    q_dim = (batch_size, heads_q, seq_len, dim)
    q_stride = (dim * seq_len * heads_q, dim, dim * heads_q, 1)
    kv_dim = (batch_size, heads_kv, seq_len, dim)
    kv_stride = (dim * seq_len * heads_kv, dim, dim * heads_kv, 1)
    with cudnn.Graph(
        handle="auto",
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        inputs=["q", "k", "v"],
        outputs=["out", "stats"],
    ) as graph:
        q_gpu = graph.tensor(name="q", dim=q_dim, stride=q_stride)
        k_gpu = graph.tensor(name="k", dim=kv_dim, stride=kv_stride)
        v_gpu = graph.tensor(name="v", dim=kv_dim, stride=kv_stride)
        out, stats = graph.sdpa(
            q=q_gpu,
            k=k_gpu,
            v=v_gpu,
            attn_scale=attn_scale,
            is_inference=False,
            use_causal_mask=True,
        )
        # set output, inv_var must be float32 tensor
        out.set_output(True).set_dim(q_dim).set_stride(q_stride).set_name("out")
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_name("stats")
    return graph


@functools.lru_cache(maxsize=None)
def get_cudnn_gqa_bwd(
    batch_size: int, seq_len: int, heads_q: int, heads_kv: int, dim: int, dtype: torch.dtype
) -> cudnn.Graph:
    """For use in the replacement of PyTorch GQA function. To compute GQA backward pass
    with causal mask
    """
    attn_scale = float(dim) ** -0.5
    q_dim = (batch_size, heads_q, seq_len, dim)
    q_stride = (dim * seq_len * heads_q, dim, dim * heads_q, 1)
    kv_dim = (batch_size, heads_kv, seq_len, dim)
    kv_stride = (dim * seq_len * heads_kv, dim, dim * heads_kv, 1)
    stats_dim = (batch_size, heads_q, seq_len, 1)
    stats_stride = (heads_q * seq_len, seq_len, 1, 1)
    dO_stride = (dim * seq_len * heads_q, dim, dim * heads_q, 1)
    with cudnn.Graph(
        handle="auto",
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        inputs=["q", "k", "v", "o", "dO", "stats"],
        outputs=["dQ", "dK", "dV"],
    ) as graph:
        q_gpu = graph.tensor(name="q", dim=q_dim, stride=q_stride)
        k_gpu = graph.tensor(name="k", dim=kv_dim, stride=kv_stride)
        v_gpu = graph.tensor(name="v", dim=kv_dim, stride=kv_stride)
        o_gpu = graph.tensor(name="o", dim=q_dim, stride=q_stride)
        dO_gpu = graph.tensor(name="dO", dim=q_dim, stride=dO_stride)
        stats_gpu = graph.tensor(
            name="stats",
            dim=stats_dim,
            stride=stats_stride,
            data_type=cudnn.data_type.FLOAT,
        )
        dQ, dK, dV = graph.sdpa_backward(
            q=q_gpu,
            k=k_gpu,
            v=v_gpu,
            o=o_gpu,
            dO=dO_gpu,
            stats=stats_gpu,
            attn_scale=attn_scale,
            use_causal_mask=True,
        )
        # set output, inv_var must be float32 tensor
        dQ.set_output(True).set_dim(q_dim).set_stride(q_stride).set_name("dQ")
        dK.set_output(True).set_dim(kv_dim).set_stride(kv_stride).set_name("dK")
        dV.set_output(True).set_dim(kv_dim).set_stride(kv_stride).set_name("dV")
    return graph


class CudnnGQA(Function):
    """Custom PyTorch GQA function using cuDNN for the use case of Llama model"""

    @staticmethod
    def forward(q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """GQA function: o = softmax(qk^T/sqrt(d)) @ v

        q, k, v are 4D tensors of shape (batch_size, num_heads, seq_length, head_dim), with different head dimensions
        """
        bq, hq, sq, dq = q.shape
        bk, hk, sk, dk = k.shape
        bv, hv, sv, dv = v.shape
        assert hq % hk == 0, "H_q must be a multiple of H_kv (GQA/MQA constraint)"
        assert hv == hk, "H_v must be equal to H_kv"
        assert dq == dk == dv, "All head dimensions must be equal"
        assert bq == bk == bv, "All batch sizes must be equal"
        assert q.dtype == k.dtype == v.dtype, "All input tensors must have the same dtype"
        assert q.stride() == (sq * dq * hq, dq, dq * hq, 1), "q.stride() != (s*d*h, d, d*h, 1)"
        assert k.stride() == (sk * dk * hk, dk, dk * hk, 1), "k.stride() != (s*d*h, d, d*h, 1)"
        assert v.stride() == (sv * dv * hv, dv, dv * hv, 1), "v.stride() != (s*d*h, d, d*h, 1)"
        graph = get_cudnn_gqa_fwd(bq, sq, hq, hk, dq, q.dtype)
        o, stats = graph(q, k, v)
        return o, stats

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Tensor, Tensor, Tensor], output: Tuple[Tensor, Tensor]) -> None:
        """Save tensors to help computing backward"""
        q, k, v = inputs
        o, stats = output
        ctx.save_for_backward(q, k, v, o, stats)

    @staticmethod
    def backward(ctx: Any, dO: Tensor, dstats: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Backward for GQA function: o = softmax(qk^T/sqrt(d)) @ v

        All tensors are 4D tensors of shape (batch_size, num_heads, seq_length, head_dim)
        """
        q, k, v, o, stats = ctx.saved_tensors
        dq = dk = dv = None
        # collect shapes and check for consistency
        bq, hq, sq, dq = q.shape
        bk, hk, sk, dk = k.shape
        bv, hv, sv, dv = v.shape
        bo, ho, so, do = o.shape
        bs, hs, ss, ds = stats.shape
        bdO, hdO, sdO, ddO = dO.shape
        assert bq == bk == bv == bo == bs == bdO, "All batch sizes must be equal"
        assert sq == so == ss == sdO, "Output and stats sequence lengths must match query sequence length"
        assert hk == hv, "H_kv must be equal to H_kv"
        assert sk == sv, "K and V sequence lengths must match"
        assert hq == ho == hs == hdO, "Output and stats num heads must match num query heads"
        assert ds == 1, "stats.shape[-1] != 1"
        assert dq == dk == dv == do == ddO, "All head dimensions must be equal"
        assert q.stride() == (sq * dq * hq, dq, dq * hq, 1), "q.stride() != (s*d*h, d, d*h, 1)"
        assert k.stride() == (sk * dk * hk, dk, dk * hk, 1), "k.stride() != (s*d*h, d, d*h, 1)"
        assert v.stride() == (sv * dv * hv, dv, dv * hv, 1), "v.stride() != (s*d*h, d, d*h, 1)"
        assert o.stride() == (so * do * ho, do, do * ho, 1), "o.stride() != (s*d*h, d, d*h, 1)"
        assert stats.stride() == (ss * hs, ss, 1, 1), "stats.stride() != (s*h, s, 1, 1)"
        assert dO.stride() == (sdO * ddO * hdO, ddO, ddO * hdO, 1), "dO.stride() != (s*d*h, d, d*h, 1)"
        assert q.dtype == k.dtype == v.dtype == o.dtype == dO.dtype, "All input/output tensors must have the same dtype"
        # cuDNN compute all grads of GQA at once
        graph = get_cudnn_gqa_bwd(bq, sq, hq, hk, dq, q.dtype)
        dQ, dK, dV = graph(q, k, v, o, dO, stats)
        return dQ, dK, dV


@dataclasses.dataclass
class LlamaConfig:
    """Configuration class for LLaMA model hyperparameters.

    This matches the configuration of LLaMA 3.1 8B model with:
    - 32 transformer layers
    - 4096 hidden dimension
    - 32 attention heads
    - Grouped-query attention with 8 key-value heads
    """

    vocab_size: int = 128256  # Size of the tokenizer vocabulary
    max_position_embeddings: int = 131072  # Maximum sequence length
    hidden_size: int = 4096  # Dimension of hidden layers
    intermediate_size: int = 14336  # Dimension of MLP's hidden layer
    num_hidden_layers: int = 32  # Number of transformer layers
    num_attention_heads: int = 32  # Number of attention heads
    num_key_value_heads: int = 8  # Number of key-value heads for GQA
    rms_norm_eps: float = 1e-5  # Epsilon for RMSNorm


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input.

    This is a helper function for rotary position embeddings (RoPE).
    For a tensor of shape (..., d), it returns a tensor where the last
    d/2 dimensions are rotated by swapping and negating.

    Args:
        x: Input tensor of shape (..., d)

    Returns:
        Tensor of same shape with rotated last dimension
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)  # Concatenate with rotation


def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to a tensor.

    RoPE performs rotation in vector space based on position using
    trigonometric functions. This allows the model to learn relative
    positions in a more efficient way than absolute position embeddings.

    Args:
        x: Input tensor of shape (batch_size, seq_length, num_heads, head_dim)
        cos: Cosine position embeddings matching the shape of x
        sin: Sine position embeddings matching the shape of x
    """
    return (x * cos) + (rotate_half(x) * sin)


def get_inv_freq(N: float, dim: int) -> Tensor:
    """Get the inverse frequency for the RoPE with the Llama 3.1 scaling.
    Always computed in float32

    Args:
        N: Base, a large number
        dim: Size of hidden dimension, should be divisible by 2
    """
    N = float(N)
    dim = int(dim)
    # Llama 3.1 RoPE parameters
    factor = 8.0
    low_freq, high_freq = 1.0, 4.0
    context_len = 8192
    # Compute the inverse frequency based on the standard RoPE formula
    inv_freq = 1.0 / (N ** (torch.arange(0, dim, 2).float().to("cuda") / dim))
    # Compute the modified inverse frequency, then derive the smoothed inverse frequency
    wavelen = 2 * math.pi / inv_freq
    max_wavelen = context_len / low_freq
    min_wavelen = context_len / high_freq
    inv_freq = torch.where(wavelen > max_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (context_len / wavelen - low_freq) / (high_freq - low_freq)
    smoothed = (1 - smooth_factor) * inv_freq / factor + smooth_factor * inv_freq
    # Output inverse frequency as a mix of the two
    is_medium_freq = ~(wavelen < min_wavelen) * ~(wavelen > max_wavelen)
    inv_freq_final = torch.where(is_medium_freq, smoothed, inv_freq)
    return inv_freq_final


class RotaryPositionEncoding(nn.Module):
    """Rotary position encoding."""

    def __init__(self, dim: int, max_position_embeddings: int) -> None:
        """Initialize the RotaryPositionEncoding module

        Args:
            dim: The hidden dimension of the input tensor to which RoPE is applied
            max_position_embeddings: The maximum sequence length of the input tensor
        """
        super().__init__()
        # compute a matrix of n\theta_i
        N = 500_000.0
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.inv_freq = get_inv_freq(N, dim)
        position = torch.arange(max_position_embeddings).float().to("cuda")
        inv_freq = torch.cat((self.inv_freq, self.inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        # save cosine and sine matrices as buffers, not parameters
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def __repr__(self) -> str:
        return f"RotaryPositionEncoding(dim={self.dim}, max_position_embeddings={self.max_position_embeddings})"

    def forward(self, x: Tensor) -> Tensor:
        """Apply RoPE to tensor x

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_heads, head_dim)

        Returns:
            Output tensor of shape (batch_size, seq_length, num_heads, head_dim)
        """
        dtype = x.dtype
        seq_len = x.shape[1]
        # transform the cosine and sine matrices to 4D tensor and the same dtype as x
        cos = self.cos.to(dtype)[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin.to(dtype)[:seq_len].view(1, seq_len, 1, -1)
        # apply RoPE to x
        return apply_rotary_pos_emb(x, cos, sin)


class LlamaMLP(nn.Module):
    """MLP layer with SwiGLU activation.

    The architecture follows:
    1. Project input to intermediate size through two parallel layers
    2. Apply SwiGLU activation (multiply gate and up-projected inputs)
    3. Project back to hidden size
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        # Two parallel projections for SwiGLU
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        # Project back to hidden size
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu  # SwiGLU activation function

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU activation: multiply gate and up-projected inputs
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaAttention(nn.Module):
    """Multi-head attention with grouped-query attention and rotary embeddings.

    Grouped-query attention reduces computation by using fewer key-value heads
    than query heads, then sharing the same key-value heads across multiple queries.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads  # GQA: H_kv < H_q

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")

        # Linear layers for Q, K, V projections
        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        rope: Optional[RotaryPositionEncoding] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # Project inputs to Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Apply rotary position embeddings
        if rope is not None:
            query_states = rope(query_states)
            key_states = rope(key_states)

        # Transpose tensors from BSHD to BHSD dimension for scaled_dot_product_attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Use cuDNN's optimized attention implementation with causal mask
        attn_output, _stats = CudnnGQA.apply(query_states, key_states, value_states)

        # Transpose output tensor from BHSD to BSHD dimension, reshape to 3D, and then project output
        attn_output = attn_output.transpose(1, 2).view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayer(nn.Module):
    """Single transformer layer for LLaMA.

    Architecture:
    1. Input -> RMSNorm -> Self-Attention -> Residual
    2. RMSNorm -> MLP -> Residual
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        rope: Optional[RotaryPositionEncoding] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # First residual block: Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(hidden_states=hidden_states, rope=rope)
        hidden_states = attn_outputs + residual

        # Second residual block: MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states


class LlamaModel(nn.Module):
    """The full Llama model."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.rotary_emb = RotaryPositionEncoding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
        )

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Stack of transformer layers
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        output_hidden_states: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]]]:
        # Convert input token IDs to embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Initialize list to collect hidden states if requested
        all_hidden_states = () if output_hidden_states else None

        # Process through all transformer layers, accumulating hidden states as the input to each layer
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(hidden_states, rope=self.rotary_emb)

        # Final layer norm, accumulate as the final hidden state
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return tuple of the output and the list of hidden states from all layers
        return [hidden_states, all_hidden_states]


# Create model with default config
test_config = LlamaConfig()
torch.set_default_device("cuda")
model = LlamaModel(test_config).to(torch.bfloat16)
print(time.time(), "model created")
state_dict = torch.load("llama3.1_8b_weights.bf16.pt", map_location="cuda")
print(time.time(), "state_dict loaded from disk")
model.load_state_dict(state_dict, strict=False)
print(time.time(), "model loaded")
del state_dict
print(model)
print()

# load sample input and output tensors from 101_hf_llama_tieout.py
tensors = torch.load("tensors-bf16-tieout.pt", map_location="cuda")
x, rope_ref, inv_freq_ref, y_ref, hidden_states_ref, target, grad_embed_ref, grad_norm_ref = tensors

# trial run one forward & backward pass
epoch = time.time()
y, hidden_states = model.forward(x, output_hidden_states=True)
print(time.time(), f"forward pass finished in {time.time() - epoch:.5f} sec")
criterion = torch.nn.MSELoss()
assert y.shape == target.shape, f"y.shape={y.shape} not the same as target.shpae={target.shape}"
loss = criterion(y, target)
epoch = time.time()
loss.backward()
print(time.time(), f"backward pass finished in {time.time() - epoch:.5f} sec")
grad_embed = model.embed_tokens.weight.grad
grad_norm = model.norm.weight.grad

# compare results
x_embed = model.embed_tokens(x)
inv_freq = model.rotary_emb.inv_freq.type_as(inv_freq_ref)
x_rope = (
    model.rotary_emb.cos[: x.shape[1]].unsqueeze(0),
    model.rotary_emb.sin[: x.shape[1]].unsqueeze(0),
)

print()
print("Numerical difference compared to reference implementation:")
print("RoPE cosine:", report_close(x_rope[0], rope_ref[0], atol=1e-3, rtol=1e-3))
print("RoPE sine:", report_close(x_rope[1], rope_ref[1], atol=1e-3, rtol=1e-3))
print("inv_freq:", report_close(inv_freq, inv_freq_ref, atol=0, rtol=0))
print()
for i in range(len(hidden_states)):
    print(
        f"output of layer {i}:",
        report_close(hidden_states[i], hidden_states_ref[i], atol=1e-2, rtol=1e-2),
    )
print()
print("final norm grad:", report_close(grad_norm, grad_norm_ref, atol=1e-2, rtol=1e-2))
print("embed grad:", report_close(grad_embed, grad_embed_ref, atol=1e-2, rtol=1e-2))
