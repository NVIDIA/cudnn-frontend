# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Triton caller conversions for the C96 causal example.

Both backend and FROST graphs consume the same compact tensors. These helpers
allocate caller-owned storage; they are not part of graph.execute().
"""

import torch
import triton
import triton.language as tl


@triton.jit
def pack_kernel(
    X,
    HIST,
    OUT,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HD: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    X3: tl.constexpr,
    X4: tl.constexpr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    H2: tl.constexpr,
    H3: tl.constexpr,
    H4: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
):
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s = tl.program_id(0) * BS + tl.arange(0, BS)
    n = tl.program_id(2)
    d, hw = s // (H * W), s % (H * W)
    hh, ww = hw // W, hw % W
    xm = (c[:, None] < C) & (d[None, :] >= HD) & (d[None, :] < HD + D)
    xp = n * X0 + c[:, None] * X1 + (d[None, :] - HD) * X2 + hh[None, :] * X3 + ww[None, :] * X4
    val = tl.load(X + xp, mask=xm, other=0)
    if HD > 0:
        hp = n * H0 + c[:, None] * H1 + d[None, :] * H2 + hh[None, :] * H3 + ww[None, :] * H4
        hv = tl.load(HIST + hp, mask=(c[:, None] < C) & (d[None, :] < HD), other=0)
        val = tl.where(d[None, :] < HD, hv, val)
    dest = n * C * (D + HD) * H * W + s[None, :] * C + c[:, None]
    tl.store(OUT + dest, val, mask=(c[:, None] < C) & (s[None, :] < (D + HD) * H * W))


@triton.jit
def unpack_bias_kernel(Y, BIAS, OUT, C: tl.constexpr, S: tl.constexpr, BC: tl.constexpr, BS: tl.constexpr):
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s = tl.program_id(0) * BS + tl.arange(0, BS)
    n = tl.program_id(2)
    mask = (c[:, None] < C) & (s[None, :] < S)
    val = tl.load(Y + n * C * S + s[None, :] * C + c[:, None], mask=mask, other=0).to(tl.float32)
    bias = tl.load(BIAS + c, mask=c < C, other=0).to(tl.float32)
    tl.store(OUT + n * C * S + c[:, None] * S + s[None, :], val + bias[:, None], mask=mask)


def _validate(tensor):
    if tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16 or tensor.ndim != 5:
        raise ValueError("expected a CUDA BF16 tensor with logical NCDHW dimensions")
    if min(tensor.shape) <= 0 or min(tensor.stride()) <= 0:
        raise ValueError("expected nonempty dimensions and positive strides")
    if tensor.shape[0] > 65535:
        raise ValueError("the example requires a batch size at most 65535")
    if sum((d - 1) * s for d, s in zip(tensor.shape, tensor.stride())) >= 2**31:
        raise ValueError("the example uses 32-bit tensor offsets")


def pack(x, history):
    """Join up to two history frames and transpose C96 features to channels-last."""
    _validate(x)
    n, c, d, h, w = x.shape
    if c != 96:
        raise ValueError("this example requires C96 input")
    hd = 0
    if history is not None:
        _validate(history)
        hd = history.shape[2]
        if history.device != x.device or history.shape != (n, c, hd, h, w) or hd not in (1, 2):
            raise ValueError("history must match the input and contain one or two frames")
    if n * c * (d + hd) * h * w >= 2**31:
        raise ValueError("the example uses 32-bit tensor offsets")
    out = torch.empty((n, c, d + hd, h, w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
    hist = x if history is None else history
    with torch.cuda.device(x.device):
        pack_kernel[(triton.cdiv((d + hd) * h * w, 64), triton.cdiv(c, 64), n)](x, hist, out, c, d, h, w, hd, *x.stride(), *hist.stride(), 64, 64, num_warps=8)
    return out


def unpack_bias(y, bias):
    """Transpose rounded BF16 output to NCDHW and add BF16 channel bias."""
    _validate(y)
    n, c, d, h, w = y.shape
    if c not in (96, 192) or y.stride() != (c * d * h * w, 1, h * w * c, w * c, c):
        raise ValueError("expected compact channels-last K96 or K192 output")
    if bias.device != y.device or bias.dtype != y.dtype or bias.shape != (c,) or not bias.is_contiguous():
        raise ValueError("bias must be a contiguous BF16 vector on the output device")
    out = torch.empty(y.shape, device=y.device, dtype=y.dtype)
    with torch.cuda.device(y.device):
        unpack_bias_kernel[(triton.cdiv(d * h * w, 128), triton.cdiv(c, 32), n)](y, bias, out, c, d * h * w, 32, 128, num_warps=8)
    return out
