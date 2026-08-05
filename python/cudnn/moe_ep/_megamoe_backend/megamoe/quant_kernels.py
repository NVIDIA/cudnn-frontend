# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Triton MXFP8 quantizers for the fp8 backward's hot paths.

Same numerics as pt.quant.quant_mxfp8_tensors (ceil-log2 E8M0 per-32
scales, RN-saturating E4M3 cast), but the E8M0 exponent is computed with
exact bit ops instead of float log2: for amax = m * 2^e (1 <= m < 2),
ceil(log2(amax / 448)) = e - 8 + (m > 1.75). Two kernels:

- ``mxfp8_rowquant(x)``   — quantize (M, K) along K (one fused kernel vs
  ~8 torch launches).
- ``mxfp8_transquant(x)`` — for wgrad operands: quantize (Ktot, N) along
  the TOKEN dim and emit the (N, Ktot) transposed fp8 layout directly,
  replacing ``x.t().contiguous()`` + rowquant.

Both fall back to the torch implementation when triton is unavailable.
"""

from __future__ import annotations

import torch

from pt.quant import MXFP8_BLOCK, quant_mxfp8_tensors

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _e8m0_from_amax(amax):
        """ceil(log2(amax/448)) + 127 as uint8 semantics (0 when amax==0)."""
        bits = amax.to(tl.int32, bitcast=True)
        e = (bits >> 23) - 127
        m_gt = (bits & 0x7FFFFF) > 0x600000  # mantissa > 1.75
        se = e - 8 + tl.where(m_gt, 1, 0)
        se = tl.where(amax > 0, se, -127)
        return tl.minimum(tl.maximum(se + 127, 0), 254)

    @triton.jit
    def _rowquant_kernel(
        x_ptr, q_ptr, s_ptr,
        K, sxm, sxk,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < K
        x = tl.load(
            x_ptr + pid_m * sxm + offs_k * sxk, mask=mask, other=0.0
        ).to(tl.float32)
        xb = tl.reshape(x, (BLOCK_K // 32, 32))
        u8 = _e8m0_from_amax(tl.max(tl.abs(xb), axis=1))
        scale = tl.exp2((u8 - 127).to(tl.float32))
        qv = xb / scale[:, None]
        qv = tl.minimum(tl.maximum(qv, -448.0), 448.0)
        q = tl.reshape(qv, (BLOCK_K,)).to(tl.float8e4nv)
        tl.store(q_ptr + pid_m * K + offs_k, q, mask=mask)
        offs_s = pid_k * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
        tl.store(
            s_ptr + pid_m * (K // 32) + offs_s,
            u8.to(tl.uint8),
            mask=offs_s < K // 32,
        )

    @triton.jit
    def _transquant_kernel(
        x_ptr, q_ptr, s_ptr,
        Kt, N, sxk, sxn,
        BLOCK_N: tl.constexpr,
    ):
        pid_k = tl.program_id(0)  # one 32-token scale block
        pid_n = tl.program_id(1)
        offs_t = pid_k * 32 + tl.arange(0, 32)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        x = tl.load(
            x_ptr + offs_t[:, None] * sxk + offs_n[None, :] * sxn,
            mask=mask_n[None, :], other=0.0,
        ).to(tl.float32)
        u8 = _e8m0_from_amax(tl.max(tl.abs(x), axis=0))  # per output col
        scale = tl.exp2((u8 - 127).to(tl.float32))
        qv = tl.minimum(tl.maximum(x / scale[None, :], -448.0), 448.0)
        q = tl.trans(qv).to(tl.float8e4nv)  # (BLOCK_N, 32)
        tl.store(
            q_ptr + offs_n[:, None] * Kt + offs_t[None, :],
            q, mask=mask_n[:, None],
        )
        tl.store(
            s_ptr + offs_n * (Kt // 32) + pid_k,
            u8.to(tl.uint8), mask=mask_n,
        )


def mxfp8_rowquant(x: torch.Tensor):
    """(M, K) -> (fp8 (M, K), e8m0 (M, K/32)) quantized along K."""
    if not HAVE_TRITON:
        return quant_mxfp8_tensors(x, dim=-1)
    M, K = x.shape
    if K % MXFP8_BLOCK:
        raise ValueError(f"K ({K}) must be a multiple of {MXFP8_BLOCK}")
    q = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((M, K // 32), device=x.device, dtype=torch.uint8)
    if M:
        BLOCK_K = 256 if K % 256 == 0 else 128 if K % 128 == 0 else 32
        grid = (M, triton.cdiv(K, BLOCK_K))
        _rowquant_kernel[grid](
            x, q, s, K, x.stride(0), x.stride(1), BLOCK_K=BLOCK_K
        )
    return q, s.view(torch.float8_e8m0fnu)


def mxfp8_transquant(x: torch.Tensor):
    """(Ktot, N) -> (fp8 (N, Ktot), e8m0 (N, Ktot/32)) quantized along the
    token (Ktot) dim, transposed output — the wgrad operand layout."""
    if not HAVE_TRITON:
        return quant_mxfp8_tensors(x.t().contiguous(), dim=-1)
    Kt, N = x.shape
    if Kt % MXFP8_BLOCK:
        raise ValueError(f"Ktot ({Kt}) must be a multiple of {MXFP8_BLOCK}")
    q = torch.empty((N, Kt), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((N, Kt // 32), device=x.device, dtype=torch.uint8)
    if Kt and N:
        BLOCK_N = 64
        grid = (Kt // 32, triton.cdiv(N, BLOCK_N))
        _transquant_kernel[grid](
            x, q, s, Kt, N, x.stride(0), x.stride(1), BLOCK_N=BLOCK_N
        )
    return q, s.view(torch.float8_e8m0fnu)
