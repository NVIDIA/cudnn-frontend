# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities and parameterization for the fused GEMM + per-head RoPE + MXFP8 projection tests.

The operation has two sibling kernels differing only in GEMM input precision:
  * GemmProjRopeMxfp8Bf16InSm100  -- BF16 inputs.
  * GemmProjRopeMxfp8Mxfp8InSm100 -- MXFP8 inputs (E4M3 codes + E8M0 block scales).
This module provides tensor creation (both precisions, no transformer_engine dependency) and
reference-comparison helpers.
"""

import torch
import pytest

# DSv3 Q up-proj shapes the kernel is specialized for.
NUM_HEADS = 128
QK_ROPE = 64
HEAD_DIM = 192
Q_LORA = 1536
Q_OUT = NUM_HEADS * HEAD_DIM
BLOCK = 32
TILE_M = 128
E8M0_BIAS = 127
FP8_MAX = 448.0


GEMM_PROJ_ROPE_MXFP8_PARAM_MARKS = [
    pytest.mark.parametrize("tokens", [2048, 4096]),
    pytest.mark.parametrize("w_out_in", [False, True]),
]


def with_gemm_proj_rope_mxfp8_params(func):
    """Apply all parameterization marks to a test function."""
    for mark in reversed(GEMM_PROJ_ROPE_MXFP8_PARAM_MARKS):
        func = mark(func)
    return func


def gemm_proj_rope_mxfp8_init(request, tokens, w_out_in):
    """Build test config; skip on unsupported architecture."""
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor
    if compute_capability < 100:
        pytest.skip(f"Environment not supported: requires compute capability >= 100, found {compute_capability}")

    skip_ref = request.config.getoption("--skip-ref", default=False)

    return {
        "tokens": tokens,
        "w_out_in": bool(w_out_in),
        "skip_ref": skip_ref,
    }


# ------------------------------- BF16-input helpers -------------------------------


def allocate_input_tensors(tokens, w_out_in):
    """Allocate bf16 activations, projection weight (in the requested layout), and rope tables."""
    dev = "cuda"
    x = torch.randn(tokens, Q_LORA, dtype=torch.bfloat16, device=dev) * 0.5
    if w_out_in:
        w = torch.randn(Q_OUT, Q_LORA, dtype=torch.bfloat16, device=dev) * 0.02  # [out, in]
    else:
        w = torch.randn(Q_LORA, Q_OUT, dtype=torch.bfloat16, device=dev) * 0.02  # [in, out]
    cos = torch.randn(tokens, QK_ROPE, dtype=torch.bfloat16, device=dev)
    sin = torch.randn(tokens, QK_ROPE, dtype=torch.bfloat16, device=dev)
    return x, w, cos, sin


def allocate_output_tensors(tokens):
    """Allocate the four MXFP8 output tensors (rowwise + columnwise data and E8M0 scales)."""
    dev = "cuda"
    out_fp8_row = torch.empty(tokens, NUM_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev)
    out_scales_row = torch.empty(tokens, NUM_HEADS, HEAD_DIM // BLOCK, dtype=torch.uint8, device=dev)
    out_fp8_col = torch.empty(tokens, NUM_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev)
    out_scales_col = torch.empty(tokens // BLOCK, NUM_HEADS, HEAD_DIM, dtype=torch.uint8, device=dev)
    return out_fp8_row, out_scales_row, out_fp8_col, out_scales_col


# ------------------------------- MXFP8-input helpers -------------------------------


def quant_mxfp8_rowwise(t):
    """Manual MXFP8 rowwise quantization of a 2-D [rows, K] tensor (no transformer_engine dep).

    Matches the E8M0/E4M3 block-scaling used by the reference oracle: one E8M0 scale per 32-wide
    block along K. Returns (code float8_e4m3fn [rows, K], scale uint8 [rows, K//32]).
    """
    rows, k = t.shape
    assert k % BLOCK == 0
    blk = t.float().reshape(rows, k // BLOCK, BLOCK)
    amax = blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    exp = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(-127.0, 127.0)
    code = (blk * torch.pow(2.0, -exp)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).reshape(rows, k)
    scale = (exp.squeeze(-1) + E8M0_BIAS).to(torch.uint8)
    return code, scale


def dequant_mxfp8_rowwise(code, scale):
    """Inverse of quant_mxfp8_rowwise: [rows, K] fp8 code + [rows, K//32] scale -> fp32 [rows, K]."""
    rows, k = code.shape
    inv = torch.pow(2.0, scale.float() - E8M0_BIAS).unsqueeze(-1)
    return (code.float().reshape(rows, k // BLOCK, BLOCK) * inv).reshape(rows, k)


def allocate_mxfp8_input_tensors(tokens):
    """Allocate MXFP8 inputs: fp8 codes + E8M0 scales for x [tokens, Q_LORA] and w [Q_OUT, Q_LORA]
    (weight is TE-native [out, in]); plus bf16 rope tables. Returns the dequantized bf16 x/w too,
    so the reference can be computed from the exact quantized operands the kernel consumes."""
    dev = "cuda"
    x_bf16 = torch.randn(tokens, Q_LORA, dtype=torch.bfloat16, device=dev) * 0.5
    w_bf16 = torch.randn(Q_OUT, Q_LORA, dtype=torch.bfloat16, device=dev) * 0.02  # [out, in]
    x_code, x_scale = quant_mxfp8_rowwise(x_bf16)
    w_code, w_scale = quant_mxfp8_rowwise(w_bf16)
    cos = torch.randn(tokens, QK_ROPE, dtype=torch.bfloat16, device=dev)
    sin = torch.randn(tokens, QK_ROPE, dtype=torch.bfloat16, device=dev)
    x_deq = dequant_mxfp8_rowwise(x_code, x_scale).to(torch.bfloat16)
    w_deq = dequant_mxfp8_rowwise(w_code, w_scale).to(torch.bfloat16)
    return x_code, x_scale, w_code, w_scale, cos, sin, x_deq, w_deq


# ------------------------------- reference comparison -------------------------------


def _deq_row(data, scale):
    t = data.shape[0]
    inv = torch.pow(2.0, scale.float() - E8M0_BIAS).unsqueeze(-1)
    return (data.float().reshape(t, NUM_HEADS, HEAD_DIM // BLOCK, BLOCK) * inv).reshape(t, NUM_HEADS, HEAD_DIM)


def _deq_col(data, scale):
    t = data.shape[0]
    inv = torch.pow(2.0, scale.float() - E8M0_BIAS).reshape(t // BLOCK, 1, NUM_HEADS, HEAD_DIM)
    return (data.float().reshape(t // BLOCK, BLOCK, NUM_HEADS, HEAD_DIM) * inv).reshape(t, NUM_HEADS, HEAD_DIM)


def _matched(got, ref, atol=0.1, rtol=0.1):
    diff = (got.float() - ref.float()).abs()
    return (diff <= atol + rtol * ref.float().abs()).float().mean().item()


def check_ref_gemm_proj_rope_mxfp8(x, w, cos, sin, outputs, w_out_in, skip_ref=False, need=0.95):
    """Compare each of the four public outputs against the PyTorch reference oracle (bf16 GEMM)."""
    if skip_ref:
        print("Skipping reference check")
        return

    from cudnn.gemm.cutedsl.dense.proj_rope_mxfp8 import gemm_proj_rope_mxfp8_reference

    out_fp8_row, out_scales_row, out_fp8_col, out_scales_col = outputs
    ref_qr, ref_sr, ref_qc, ref_sc = gemm_proj_rope_mxfp8_reference(x, w, cos, sin, w_out_in=w_out_in)

    row = _matched(_deq_row(out_fp8_row, out_scales_row), _deq_row(ref_qr, ref_sr))
    col = _matched(_deq_col(out_fp8_col, out_scales_col), _deq_col(ref_qc, ref_sc))
    assert row >= need, f"rowwise MXFP8 mismatch vs reference: matched={row:.4f}"
    assert col >= need, f"columnwise MXFP8 mismatch vs reference: matched={col:.4f}"
