# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 06: MoE grouped matmul + SwiGLU (multi-GEMM, mode=NONE).

Two parallel grouped matmuls share the token operand AND one
first_token_offset (identical routed-group layout), feeding one fused
pointwise epilogue. Each routed group g (expert g % E) computes
    out[fto[g] : fto[g+1]] =
        silu(token[range] @ w0[g % E].T) * (token[range] @ w1[g % E].T) * scale
The token is loaded once (deduped); the MMA loops over the two GEMMs into
per-GEMM TMEM regions and the shared epilogue merges them.

Cases: dense BF16 (uneven groups incl. an empty one), then the block-scale
combos NVFP4 (packed FP4 E2M1 + per-16-block FP8 E4M3 scale) and MXFP8 (FP8
E4M3 + per-32-block FP8 E8M0 scale) with BxE > E routed groups — the shared
token+SFA dequant is matched into BOTH GEMMs. Block-scale dual needs
cta_tile_n <= 128 (dual accumulators + SF region must fit 512 TMEM cols).

The block-scale cases additionally end the chain in a block_scale_quantize:
the SwiGLU result is re-quantized to FP8 E4M3 with a per-32-block E8M0 scale,
producing TWO outputs (quantized data + scale factors).
"""

from __future__ import annotations

import cudnn
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.tile_config import CATALOG

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


def _moe_config():
    # cta_tile_n=128 fits the block-scale dual TMEM budget too.
    return next(c for c in CATALOG if c.name == "CONFIG_sm100_128x128x128_128x128x32_cluster2x1")


def _swiglu_ref(tok_f32, w0_f32, w1_f32, offsets_list, S, N, num_groups, E, scale):
    """torch group-loop reference: silu(tok@w0.T) * (tok@w1.T) * scale per group."""
    ref = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        mm0 = tok_f32[b:e] @ w0_f32[gi % E].T
        mm1 = tok_f32[b:e] @ w1_f32[gi % E].T
        ref[b:e] = torch.nn.functional.silu(mm0) * mm1 * scale
    return ref


def _block_quant_ref(x, block_size, out_dtype, scale_dtype):
    """Torch reference for the block-quant epilogue: per-block amax scale
    (E8M0 scales round toward +inf) + quantized output."""
    blocks = x.view(1, x.shape[0], x.shape[1] // block_size, block_size)
    output_max = 448.0 if out_dtype is torch.float8_e4m3fn else 57344.0
    scale_f = blocks.abs().amax(dim=-1) / output_max
    if scale_dtype is torch.float8_e8m0fnu:
        safe = torch.where(scale_f > 0, scale_f, 1.0)
        scale_f = torch.where(scale_f > 0, torch.pow(2.0, torch.ceil(torch.log2(safe))), 0.0)
    scale = scale_f.to(scale_dtype)
    inv = torch.where(scale.float() > 0, scale.float().reciprocal(), 0.0)
    q = (blocks * inv.unsqueeze(-1)).clamp(-output_max, output_max)
    q = q.to(out_dtype).view(1, x.shape[0], x.shape[1])
    return q, scale


def _dense_case() -> None:
    # E experts; uneven (incl. empty) token groups summing to S.
    E, N, K = 8, 256, 128
    group_sizes = [64, 0, 200, 128, 100, 12, 196, 68]
    S = sum(group_sizes)
    starts, cur = [], 0
    for gs in group_sizes:
        starts.append(cur)
        cur += gs

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=cudnn.data_type.BFLOAT16)
    # weights [E, K, N] K-major (== (E, N, K) row-major in memory)
    w0 = g.tensor(name="weight0", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    w1 = g.tensor(name="weight1", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    fto = g.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    sf = g.tensor(name="scaleFactor", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)

    c0 = g.moe_grouped_matmul(tok, w0, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe0")
    c1 = g.moe_grouped_matmul(tok, w1, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe1")
    s0 = g.swish(input=c0, name="silu")
    mu = g.mul(a=s0, b=c1, name="mul")
    dq = g.mul(a=mu, b=sf, name="dequant")
    dq.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    compiled = jit_from_cudnn_graph(g, config=_moe_config(), cta_group=2)

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight0 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    weight1 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")
    scale_t = torch.tensor([[[0.5]]], dtype=torch.float32, device="cuda")
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")

    compiled({tok: token, w0: weight0, w1: weight1, fto: offsets, sf: scale_t, dq: output})
    torch.cuda.synchronize()

    ref = _swiglu_ref(token[0].float(), weight0.float(), weight1.float(), starts, S, N, E, E, 0.5)
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=2e-1, rtol=2e-2)
    print(f"[06] PASS  {'dense bf16':10s} E={E} S={S} N={N} K={K}  groups={group_sizes}")


# combo -> (block_size, data cudnn dtype, SF cudnn dtype)
_BS_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


def _block_scale_case(combo: str, S: int = 1024, N: int = 256, K: int = 512, E: int = 2) -> None:
    dev = "cuda"
    torch.manual_seed(0)
    block_size, data_dt, sf_dt = _BS_COMBOS[combo]
    sf_k = K // block_size
    rk = dict(reordering_type=cudnn.tensor_reordering.F8_128x4)
    # 4 routed groups over E=2 experts (BxE > E).
    offsets_list = [0, 256, 384, 512]
    num_groups = len(offsets_list)

    if combo == "nvfp4":
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        tok_u8 = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev)
        w0_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        w1_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        tok_rt, w0_rt, w1_rt = (t.view(torch.float4_e2m1fn_x2) for t in (tok_u8, w0_u8, w1_u8))
        tok_f32 = _unpack_fp4(tok_u8, lut).view(S, K)
        w0_f32 = _unpack_fp4(w0_u8, lut).view(E, N, K)
        w1_f32 = _unpack_fp4(w1_u8, lut).view(E, N, K)
        sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb0_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb1_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        tok_rt = torch.empty(1, S, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.float8_e4m3fn, device=dev)
        w0_rt = torch.empty(E, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.float8_e4m3fn, device=dev)
        w1_rt = torch.empty(E, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.float8_e4m3fn, device=dev)
        tok_f32, w0_f32, w1_f32 = tok_rt.float().view(S, K), w0_rt.float(), w1_rt.float()
        # SF: E8M0 powers of two around 1.0 (biased exponent 125..128 -> 0.25x..2x).
        sfa_log = torch.randint(125, 129, (S, sf_k), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
        sfb0_log = torch.randint(125, 129, (E, N, sf_k), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
        sfb1_log = torch.randint(125, 129, (E, N, sf_k), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=data_dt)
    w0 = g.tensor(name="weight0", dim=[E, K, N], stride=[K * N, 1, K], data_type=data_dt)
    w1 = g.tensor(name="weight1", dim=[E, K, N], stride=[K * N, 1, K], data_type=data_dt)
    SFA = g.tensor(name="SFA", dim=[1, S, sf_k], stride=[S * sf_k, sf_k, 1], data_type=sf_dt, **rk)
    SFB0 = g.tensor(name="SFB0", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, **rk)
    SFB1 = g.tensor(name="SFB1", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, **rk)
    fto = g.tensor(name="first_token_offset", dim=[num_groups, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    sf = g.tensor(name="scaleFactor", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)

    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size])  # shared by both GEMMs
    w0_d = g.block_scale_dequantize(input=w0, descale=SFB0, block_size=[block_size, 1])
    w1_d = g.block_scale_dequantize(input=w1, descale=SFB1, block_size=[block_size, 1])
    c0 = g.moe_grouped_matmul(tok_d, w0_d, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe0")
    c1 = g.moe_grouped_matmul(tok_d, w1_d, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe1")
    s0 = g.swish(input=c0, name="silu")
    mu = g.mul(a=s0, b=c1, name="mul")
    dq = g.mul(a=mu, b=sf, name="dequant")
    # Terminal block-quantize: SwiGLU result -> FP8 E4M3 + per-32-block E8M0 scale.
    qblock = 32
    Q, QS = g.block_scale_quantize(input=dq, block_size=qblock, name="q")
    Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
    QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)

    compiled = jit_from_cudnn_graph(g, config=_moe_config(), cta_group=2)

    # SFA reordered + padded to 128 rows PER GROUP then concatenated; SFB per-expert.
    sfa_parts = []
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        sfa_parts.append(_to_blocked(sfa_log[b:e]))
    sfa_blk = torch.cat(sfa_parts).view(1, -1, 1)
    sfb0_blk = torch.cat([_to_blocked(sfb0_log[e]) for e in range(E)]).view(E, sf_k, N)
    sfb1_blk = torch.cat([_to_blocked(sfb1_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    scale_t = torch.tensor([[[0.5]]], dtype=torch.float32, device=dev)
    q_out = torch.zeros(1, S, N, dtype=torch.float8_e4m3fn, device=dev)
    q_scale = torch.zeros(1, S, N // qblock, dtype=torch.float8_e8m0fnu, device=dev)

    compiled(
        {
            tok: tok_rt,
            SFA: sfa_blk,
            w0: w0_rt,
            SFB0: sfb0_blk,
            w1: w1_rt,
            SFB1: sfb1_blk,
            fto: offsets,
            sf: scale_t,
            Q: q_out,
            QS: q_scale,
        }
    )
    torch.cuda.synchronize()

    tok_deq = tok_f32 * sfa_log.float().repeat_interleave(block_size, 1)
    w0_deq = w0_f32 * sfb0_log.float().repeat_interleave(block_size, 2)
    w1_deq = w1_f32 * sfb1_log.float().repeat_interleave(block_size, 2)
    ref = _swiglu_ref(tok_deq, w0_deq, w1_deq, offsets_list, S, N, num_groups, E, 0.5)
    q_ref, scale_ref = _block_quant_ref(ref, qblock, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q_out.float(), q_ref.float(), atol=0, rtol=0)
    print(f"[06] PASS  {combo:10s} S={S} N={N} K={K} E={E} groups={offsets_list} block_size={block_size} -> fp8+e8m0/{qblock}")


def main() -> None:
    _dense_case()
    _block_scale_case("nvfp4")
    _block_scale_case("mxfp8")


if __name__ == "__main__":
    main()
