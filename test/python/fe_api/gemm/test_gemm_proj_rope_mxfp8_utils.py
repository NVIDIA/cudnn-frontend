"""
Utilities and parameterization for the fused GEMM + per-head RoPE + MXFP8 projection tests.
Contains test configuration, tensor creation, and reference comparison helpers.
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
    """Compare each of the four public outputs against the PyTorch reference oracle."""
    if skip_ref:
        print("Skipping reference check")
        return

    from cudnn.gemm_proj_rope_mxfp8 import gemm_proj_rope_mxfp8_reference

    out_fp8_row, out_scales_row, out_fp8_col, out_scales_col = outputs
    ref_qr, ref_sr, ref_qc, ref_sc = gemm_proj_rope_mxfp8_reference(x, w, cos, sin, w_out_in=w_out_in)

    row = _matched(_deq_row(out_fp8_row, out_scales_row), _deq_row(ref_qr, ref_sr))
    col = _matched(_deq_col(out_fp8_col, out_scales_col), _deq_col(ref_qc, ref_sc))
    assert row >= need, f"rowwise MXFP8 mismatch vs reference: matched={row:.4f}"
    assert col >= need, f"columnwise MXFP8 mismatch vs reference: matched={col:.4f}"
