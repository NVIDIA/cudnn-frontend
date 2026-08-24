# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 04: multi-GEMM — dual matmul + SwiGLU (pure cuDNN frontend API).

Two parallel GEMMs share the A operand and feed one fused epilogue:
    out = silu(A @ B0) * (A @ B1) [* scale]
A is loaded once (deduped); two tcgen05 MMAs feed two TMEM accumulators that the
shared epilogue reads. Multi-GEMM lives in the 1-CTA-MMA template (cta_group=1).

Cases: dense BF16, then the block-scale combos NVFP4 (packed FP4 E2M1 +
per-16-block FP8 E4M3 scale) and MXFP8 (FP8 E4M3 + per-32-block FP8 E8M0
scale) — a single block_scale_dequantize(A) is matched into BOTH GEMMs (one
distinct A operand + SFA). Block-scale dual needs cta_n <= 128: dual
accumulators + the SF region must fit 512 TMEM cols.
"""

from __future__ import annotations

import cudnn
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.tile_config import by_name

# E2M1 (FP4) 4-bit code -> value lookup (low nibble first within a byte).
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _to_blocked(x: torch.Tensor) -> torch.Tensor:
    """(rows, cols) SF matrix -> F8_128x4 blocked layout, flat."""
    rows, cols = x.shape
    nrb, ncb = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pad = torch.zeros(nrb * 128, ncb * 4, dtype=x.dtype, device=x.device)
    pad[:rows, :cols] = x
    blocks = pad.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _unpack_fp4(u8: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """(..., Kp) uint8 -> (..., 2*Kp) fp32, low nibble first."""
    lo = lut[(u8 & 0xF).long()]
    hi = lut[(u8 >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def _dense_case(M: int, N: int, K: int) -> None:
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B0 = g.tensor(name="B0", dim=[1, K, N], stride=[K * N, 1, K])
    B1 = g.tensor(name="B1", dim=[1, K, N], stride=[K * N, 1, K])
    scale = g.tensor(name="scale", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)

    C0 = g.matmul(A=A, B=B0, name="mm0")
    C1 = g.matmul(A=A, B=B1, name="mm1")  # shares A
    S0 = g.swish(input=C0, name="silu")
    MU = g.mul(a=S0, b=C1, name="mul")
    DQ = g.mul(a=MU, b=scale, name="dequant")
    DQ.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    compiled = jit_from_cudnn_graph(g, cta_group=1)

    torch.manual_seed(0)
    a = torch.empty(1, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b0 = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b1 = torch.empty(1, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    c = torch.empty(1, M, N, dtype=torch.bfloat16, device="cuda")
    scale_t = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)

    compiled({A: a, B0: b0, B1: b1, scale: scale_t, DQ: c})
    torch.cuda.synchronize()

    mm0 = torch.einsum("bmk,bnk->bmn", a.float(), b0.float())
    mm1 = torch.einsum("bmk,bnk->bmn", a.float(), b1.float())
    ref = (torch.nn.functional.silu(mm0) * mm1 * 0.5).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print(f"[04] PASS  {'dense bf16':10s} M={M} N={N} K={K}")


# combo -> (block_size, data cudnn dtype, SF cudnn dtype)
_BS_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


def _block_scale_case(combo: str, M: int, N: int, K: int) -> None:
    dev = "cuda"
    torch.manual_seed(0)
    block_size, data_dt, sf_dt = _BS_COMBOS[combo]
    sf_k = K // block_size
    rk = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=data_dt)
    SFA = g.tensor(name="SFA", dim=[1, M, sf_k], stride=[M * sf_k, sf_k, 1], data_type=sf_dt, **rk)
    B0 = g.tensor(name="B0", dim=[1, K, N], stride=[K * N, 1, K], data_type=data_dt)
    SFB0 = g.tensor(name="SFB0", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, **rk)
    B1 = g.tensor(name="B1", dim=[1, K, N], stride=[K * N, 1, K], data_type=data_dt)
    SFB1 = g.tensor(name="SFB1", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, **rk)

    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block_size])  # shared by both GEMMs
    B0d = g.block_scale_dequantize(input=B0, descale=SFB0, block_size=[block_size, 1])
    B1d = g.block_scale_dequantize(input=B1, descale=SFB1, block_size=[block_size, 1])
    C0 = g.matmul(A=Ad, B=B0d, name="mm0")
    C1 = g.matmul(A=Ad, B=B1d, name="mm1")
    Y = g.mul(a=g.swish(input=C0, name="silu"), b=C1, name="mul")
    Y.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")
    compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=1)

    if combo == "nvfp4":
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

        def _mk(rows):
            u8 = torch.randint(0, 256, (1, rows, K // 2), dtype=torch.uint8, device=dev)
            return u8.view(torch.float4_e2m1fn_x2), _unpack_fp4(u8, lut).view(rows, K)

        def _mksf(rows):
            return torch.randint(1, 4, (rows, sf_k), device=dev).to(torch.float8_e4m3fn)

    else:

        def _mk(rows):
            data = torch.empty(1, rows, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.float8_e4m3fn, device=dev)
            return data, data.float().view(rows, K)

        def _mksf(rows):
            return torch.randint(125, 129, (rows, sf_k), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)

    a_rt, a_deq = _mk(M)
    b0_rt, b0_deq = _mk(N)
    b1_rt, b1_deq = _mk(N)
    sfa, sfb0, sfb1 = _mksf(M), _mksf(N), _mksf(N)

    c = torch.zeros(1, M, N, dtype=torch.float32, device=dev)
    compiled(
        {
            A: a_rt,
            SFA: _to_blocked(sfa).view(1, M, sf_k),
            B0: b0_rt,
            SFB0: _to_blocked(sfb0).view(1, N, sf_k),
            B1: b1_rt,
            SFB1: _to_blocked(sfb1).view(1, N, sf_k),
            Y: c,
        }
    )
    torch.cuda.synchronize()

    a_s = a_deq * sfa.float().repeat_interleave(block_size, 1)
    b0_s = b0_deq * sfb0.float().repeat_interleave(block_size, 1)
    b1_s = b1_deq * sfb1.float().repeat_interleave(block_size, 1)
    ref = torch.nn.functional.silu(a_s @ b0_s.t()) * (a_s @ b1_s.t())
    # matmul exact, but swish uses cuDNN's fast __expf/__fdividef → ~1e-3 rel.
    torch.testing.assert_close(c[0], ref, rtol=2e-2, atol=2e-1)
    print(f"[04] PASS  {combo:10s} M={M} N={N} K={K} block_size={block_size}")


def main(M: int = 256, N: int = 256, K: int = 128) -> None:
    _dense_case(M, N, K)
    _block_scale_case("nvfp4", M, 128, 512)
    _block_scale_case("mxfp8", M, 128, 512)


if __name__ == "__main__":
    main()
