# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Fused Triton kernels for the backward of the shared expert's output gate.

The gated forward adds G * S to the routed output, where S is the shared-expert output and G = sigmoid(X @ Wg^T) per
token. Per token row, the backward keeps every BF16 tensor boundary of the eager PyTorch operations it replaces:

    d_shared = BF16(dY * G)                            grad_output * gate
    d_gate   = BF16(sum_h BF16(dY_h * S_h))            (grad_output * shared_output).sum(-1), FP32 accumulator
    d_z      = BF16(BF16(d_gate * BF16(1 - G)) * G)    aten.sigmoid_backward on BF16 (c10::BFloat16 arithmetic)
    d_x      = BF16(d_x + BF16(d_z * Wg))              d_x + d_z @ Wg; the K=1 product is exact in FP32

gate_bwd_pre computes d_shared and d_z in one pass over dY and S; gate_bwd_post_ merges d_z @ Wg into the backward
epilogue's d_x in place. Both match the eager operations bitwise, except that the FP32 summation order of d_gate
differs from the PyTorch reduction. The kernels index their operands as BF16 with the shapes documented below, so
both wrappers raise ValueError for any other dtype, shape or device instead of misreading the memory. CuteMoK's backward
calls the unchecked _launch_* functions: it checks the gate weight itself, and its other operands are valid by
construction.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_rtne(x):
    """Rounds FP32 to the nearest-even BF16 value and returns it as FP32; NaN stays NaN.

    Integer arithmetic keeps the rounding: Triton can fold an x.to(bf16).to(fp32) round trip whose result feeds an
    add or a multiply, which would silently drop the intermediate BF16 rounding. The integer rounding alone would turn
    some NaN encodings into signed zeros or infinities (0x7FFFFFFF -> -0.0), so NaN inputs pass through unchanged.
    """
    b = x.to(tl.uint32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return tl.where(x != x, x, b.to(tl.float32, bitcast=True))


@triton.jit
def _gate_bwd_pre_kernel(dy_ptr, s_ptr, g_ptr, dshared_ptr, dz_ptr, H, BLOCK_H: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    base = row * H
    dy = tl.load(dy_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row).to(tl.float32)
    tl.store(dshared_ptr + base + offs, (dy * g).to(tl.bfloat16), mask=mask)
    d_gate = _bf16_rtne(tl.sum(_bf16_rtne(dy * s), axis=0))
    # aten.sigmoid_backward on BF16 evaluates a * (1 - b) * b in c10::BFloat16 arithmetic, rounding every step.
    d_z = _bf16_rtne(_bf16_rtne(d_gate * _bf16_rtne(1.0 - g)) * g)
    tl.store(dz_ptr + row, d_z.to(tl.bfloat16))


@triton.jit
def _gate_bwd_post_kernel(dx_ptr, dz_ptr, w_ptr, H, BLOCK_H: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    base = row * H
    dx = tl.load(dx_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dz = tl.load(dz_ptr + row).to(tl.float32)
    tl.store(dx_ptr + base + offs, (dx + _bf16_rtne(dz * w)).to(tl.bfloat16), mask=mask)


def _warps(block_h):
    return 4 if block_h <= 2048 else 8


def gate_bwd_pre(
    dy: torch.Tensor,
    shared_output: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the shared-expert output gradient and the gate-logit gradient in one pass over dY and S.

    Inputs:
        dy:            bfloat16 [num_local_tokens, hidden_size]
        shared_output: bfloat16 [num_local_tokens, hidden_size]
        gate:          bfloat16 [num_local_tokens, 1]

    Outputs:
        d_shared: bfloat16 [num_local_tokens, hidden_size]
        d_z:      bfloat16 [num_local_tokens, 1]
    """
    if (
        dy.dim() != 2
        or dy.dtype != torch.bfloat16
        or shared_output.dtype != torch.bfloat16
        or gate.dtype != torch.bfloat16
        or tuple(shared_output.shape) != tuple(dy.shape)
        or tuple(gate.shape) != (dy.shape[0], 1)
        or not dy.is_cuda
        or shared_output.device != dy.device
        or gate.device != dy.device
    ):
        raise ValueError("gate_bwd_pre needs BF16 dy, shared_output [T, H] and gate [T, 1] on one CUDA device")
    return _launch_gate_bwd_pre(dy, shared_output, gate)


def _launch_gate_bwd_pre(
    dy: torch.Tensor,
    shared_output: torch.Tensor,
    gate: torch.Tensor,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """gate_bwd_pre without its argument checks, for operands that are valid by construction (CuteMoK's backward).
    ``out``: contiguous (d_shared, d_z) buffers to store into instead of new tensors (backward_execute's)."""
    dy = dy.contiguous()
    shared_output = shared_output.contiguous()
    gate = gate.contiguous()
    num_local_tokens, hidden_size = dy.shape
    if out is None:
        d_shared = torch.empty_like(dy)
        d_z = torch.empty((num_local_tokens, 1), dtype=torch.bfloat16, device=dy.device)
    else:
        d_shared, d_z = out
    if num_local_tokens:
        block_h = triton.next_power_of_2(hidden_size)
        _gate_bwd_pre_kernel[(num_local_tokens,)](dy, shared_output, gate, d_shared, d_z, hidden_size, BLOCK_H=block_h, num_warps=_warps(block_h))
    return d_shared, d_z


def gate_bwd_post_(d_x: torch.Tensor, d_z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Merges the gate's input gradient d_z @ weight into d_x in place.

    Inputs:
        d_x:    bfloat16 [num_local_tokens, hidden_size], updated in place
        d_z:    bfloat16 [num_local_tokens, 1]
        weight: bfloat16 [1, hidden_size]

    Outputs:
        d_x: the updated input tensor
    """
    if (
        d_x.dim() != 2
        or d_x.dtype != torch.bfloat16
        or d_z.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or tuple(d_z.shape) != (d_x.shape[0], 1)
        or tuple(weight.shape) != (1, d_x.shape[1])
        or not d_x.is_cuda
        or d_z.device != d_x.device
        or weight.device != d_x.device
    ):
        raise ValueError("gate_bwd_post_ needs BF16 d_x [T, H], d_z [T, 1] and weight [1, H] on one CUDA device")
    if not d_x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("d_x and weight must be contiguous")
    return _launch_gate_bwd_post_(d_x, d_z, weight)


def _launch_gate_bwd_post_(d_x: torch.Tensor, d_z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """gate_bwd_post_ without its argument checks, for operands that are valid by construction (CuteMoK's backward)."""
    num_local_tokens, hidden_size = d_x.shape
    if num_local_tokens:
        block_h = triton.next_power_of_2(hidden_size)
        _gate_bwd_post_kernel[(num_local_tokens,)](d_x, d_z.contiguous(), weight, hidden_size, BLOCK_H=block_h, num_warps=_warps(block_h))
    return d_x
