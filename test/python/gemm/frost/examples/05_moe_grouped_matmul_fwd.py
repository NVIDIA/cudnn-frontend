# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 05: MoE grouped matmul forward (mode=NONE).

Each routed group g (expert g % E) computes
    out[first_token_offset[g] : first_token_offset[g+1]] =
        token[that range] @ weight[g % E].T
Compared against a torch group-loop reference.

Cases: dense BF16 (uneven groups incl. an empty one), then the block-scale
combos NVFP4 (packed FP4 E2M1 + per-16-block FP8 E4M3 scale) and MXFP8 (FP8
E4M3 + per-32-block FP8 E8M0 scale) with BxE > E routed groups —
block_scale_dequantize(token/weight) feeding moe_grouped_matmul folds into one
fused kernel. SFA is reordered + padded to 128 rows PER GROUP and concatenated;
the scheduler tracks each group's start SF-block.
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
    return next(c for c in CATALOG if c.name == "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma")


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
    # weight [E, K, N] K-major (== (E, N, K) row-major in memory)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=cudnn.data_type.BFLOAT16)
    fto = g.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    out = g.moe_grouped_matmul(
        tok,
        w,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
    )
    out.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    compiled = jit_from_cudnn_graph(g, config=_moe_config())

    torch.manual_seed(0)
    token = torch.randn(1, S, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device="cuda")
    offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")

    compiled({tok: token, w: weight, fto: offsets, out: output})
    torch.cuda.synchronize()

    ref = torch.zeros((S, N), dtype=torch.float32, device="cuda")
    for gi in range(E):
        b = starts[gi]
        e = starts[gi + 1] if gi + 1 < E else S
        if b == e:
            continue
        ref[b:e] = token[0, b:e].float() @ weight[gi].float().T
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    print(f"[05] PASS  {'dense bf16':10s} E={E} S={S} N={N} K={K}  groups={group_sizes}")


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
    # 4 routed groups over E=2 experts (BxE > E). Any group sizes work.
    offsets_list = [0, 256, 384, 512]
    num_groups = len(offsets_list)

    if combo == "nvfp4":
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        tok_u8 = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev)
        w_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        tok_rt, w_rt = tok_u8.view(torch.float4_e2m1fn_x2), w_u8.view(torch.float4_e2m1fn_x2)
        tok_f32 = _unpack_fp4(tok_u8, lut).view(S, K)
        w_f32 = _unpack_fp4(w_u8, lut).view(E, N, K)
        sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        tok_rt = torch.empty(1, S, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.float8_e4m3fn, device=dev)
        w_rt = torch.empty(E, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.float8_e4m3fn, device=dev)
        tok_f32, w_f32 = tok_rt.float().view(S, K), w_rt.float()
        # SF: E8M0 powers of two around 1.0 (biased exponent 125..128 -> 0.25x..2x).
        sfa_log = torch.randint(125, 129, (S, sf_k), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
        sfb_log = torch.randint(125, 129, (E, N, sf_k), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    # token [1,S,K]; weight [E,K,N] K-major (== (E,N,K) row-major memory).
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=data_dt)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K], data_type=data_dt)
    SFA = g.tensor(
        name="SFA",
        dim=[1, S, sf_k],
        stride=[S * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[E, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[num_groups, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size])
    w_d = g.block_scale_dequantize(input=w, descale=SFB, block_size=[block_size, 1])
    out = g.moe_grouped_matmul(
        tok_d,
        w_d,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
    )
    out.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)

    compiled = jit_from_cudnn_graph(g, config=_moe_config())

    # SFA reordered + padded to 128 rows PER GROUP then concatenated; SFB per-expert.
    sfa_parts = []
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        sfa_parts.append(_to_blocked(sfa_log[b:e]))
    sfa_blk = torch.cat(sfa_parts).view(1, -1, 1)
    sfb_blk = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)

    compiled({tok: tok_rt, w: w_rt, SFA: sfa_blk, SFB: sfb_blk, fto: offsets, out: output})
    torch.cuda.synchronize()

    tok_deq = tok_f32 * sfa_log.float().repeat_interleave(block_size, 1)
    w_deq = w_f32 * sfb_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        ref[b:e] = tok_deq[b:e] @ w_deq[gi % E].T
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
    print(f"[05] PASS  {combo:10s} S={S} N={N} K={K} E={E} groups={offsets_list} block_size={block_size}")


def main() -> None:
    _dense_case()
    _block_scale_case("nvfp4")
    _block_scale_case("mxfp8")


if __name__ == "__main__":
    main()
