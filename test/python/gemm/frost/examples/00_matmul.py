# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 00: matmul across input dtypes — FROST engine vs native cuDNN vs torch.

One pure (no-fusion) matmul per dtype family: BF16, FP16, FP8 E4M3 -> FP16
(FP32 accumulate), INT8 -> FP32 (INT32 accumulate; SM 100/110 only), and the
block-scale combos NVFP4 (packed FP4 E2M1 + per-16-block FP8 E4M3 scale) and
MXFP8 (FP8 E4M3 + per-32-block FP8 E8M0 scale) -> FP16, with scale factors in
the F8_128x4 swizzled layout. The tile configs are
dtype-agnostic (K is stored in bytes), so every case rides the same catalog.
FROST is a named engine (``frost_gemm``) in the graph's ranked plan list:
select_plan it for the OSS JIT GEMM, deselect it for native cuDNN — the BF16
case runs both and asserts they match.
"""

from __future__ import annotations

import cudnn
import torch

# label, io dtype, output dtype (None = io), compute dtype, torch in/out, |input| range
_CASES = (
    ("bf16", cudnn.data_type.BFLOAT16, None, cudnn.data_type.FLOAT, torch.bfloat16, torch.bfloat16, 2),
    ("fp16", cudnn.data_type.HALF, None, cudnn.data_type.FLOAT, torch.float16, torch.float16, 2),
    ("fp8_e4m3->fp16", cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, cudnn.data_type.FLOAT, torch.float8_e4m3fn, torch.float16, 3),
    ("int8->fp32", cudnn.data_type.INT8, cudnn.data_type.FLOAT, cudnn.data_type.INT32, torch.int8, torch.float32, 40),
)


def _active_sm() -> int | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _int8_mma_unavailable() -> str | None:
    from cudnn.gemm.frost.kernel_registry import MMA_GPU_ARCH_SPECIAL_CASES

    ranges = MMA_GPU_ARCH_SPECIAL_CASES[("sm100", ("int8", "int8", "int32"))]
    sm = _active_sm()
    if sm is not None and any(lo <= sm < hi for lo, hi in ranges):
        return None
    where = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in ranges)
    return f"int8 MMA exists only on {where}, have " + ("no GPU" if sm is None else f"sm_{sm}")


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


def _build_matmul_graph(M: int, N: int, K: int, io_dt, out_dt, compute_dt):
    g = cudnn.pygraph(
        io_data_type=io_dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=compute_dt,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    if out_dt is not None:
        C.set_data_type(out_dt)
    return g, A, B, C


def _build_plans(g, *, frost: bool) -> None:
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    if frost:
        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        g.select_plan(names.index("frost_gemm"))  # pin the FROST entry
    else:
        g.deselect_engines(["frost_gemm"])
    g.check_support()
    g.build_plans()


def _mkdata(M: int, N: int, K: int, torch_in, rng: int):
    torch.manual_seed(0)
    if torch_in is torch.int8:
        a = torch.randint(-rng, rng, (1, M, K), dtype=torch.int8, device="cuda")
        b = torch.randint(-rng, rng, (1, N, K), dtype=torch.int8, device="cuda")
    else:
        # Small-integer inputs keep the FP32 reference exactly representable.
        a = torch.empty(1, M, K, dtype=torch.int32).random_(-rng, rng).to(dtype=torch_in, device="cuda")
        b = torch.empty(1, N, K, dtype=torch.int32).random_(-rng, rng).to(dtype=torch_in, device="cuda")
    return a, b


def _execute(g, A, B, C, a, b, torch_out, M: int, N: int):
    c = torch.empty(1, M, N, dtype=torch_out, device="cuda")
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a, B: b, C: c}, workspace)
    return c


# combo -> (block_size, data cudnn dtype, SF cudnn dtype)
_BS_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


def _block_scale_case(combo: str, M: int, N: int, K: int) -> None:
    """Block-scale matmul: packed FP4/FP8 inputs + per-K-block scale factors
    (F8_128x4 swizzled layout), dequantized in the MMA."""
    dev = "cuda"
    torch.manual_seed(0)
    block_size, data_dt, sf_dt = _BS_COMBOS[combo]
    sf_k = K // block_size

    if combo == "nvfp4":
        a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
        b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
        a_data, b_data = a_u8.view(torch.float4_e2m1fn_x2), b_u8.view(torch.float4_e2m1fn_x2)
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        a_f32, b_f32 = _unpack_fp4(a_u8, lut).view(M, K), _unpack_fp4(b_u8, lut).view(N, K)
        # SF: small positive E4M3 integers.
        sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        a_data = torch.empty(1, M, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.float8_e4m3fn, device=dev)
        b_data = torch.empty(1, N, K, dtype=torch.int32).random_(-3, 3).to(dtype=torch.float8_e4m3fn, device=dev)
        a_f32, b_f32 = a_data.float().view(M, K), b_data.float().view(N, K)
        # SF: E8M0 powers of two around 1.0 (biased exponent 125..128 -> 0.25x..2x).
        sfa_log = torch.randint(125, 129, (M, sf_k), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
        sfb_log = torch.randint(125, 129, (N, sf_k), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=data_dt)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=data_dt)
    SFA = g.tensor(
        name="SFA",
        dim=[1, M, sf_k],
        stride=[M * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[1, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block_size])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[block_size, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.HALF)
    _build_plans(g, frost=True)

    sfa_blk = _to_blocked(sfa_log).view(1, M, sf_k)
    sfb_blk = _to_blocked(sfb_log).view(1, N, sf_k)
    c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({A: a_data, B: b_data, SFA: sfa_blk, SFB: sfb_blk, C: c}, workspace)
    torch.cuda.synchronize()

    a_deq = a_f32 * sfa_log.float().repeat_interleave(block_size, 1)
    b_deq = b_f32 * sfb_log.float().repeat_interleave(block_size, 1)
    ref = (a_deq @ b_deq.t()).to(torch.float16)
    torch.testing.assert_close(c[0], ref, atol=1e-1, rtol=1e-2)
    print(f"[00] PASS  {combo + '->fp16':16s} M={M} N={N} K={K} block_size={block_size}")


def main(M: int = 256, N: int = 256, K: int = 256) -> None:
    for label, io_dt, out_dt, compute_dt, torch_in, torch_out, rng in _CASES:
        if torch_in is torch.int8:
            why = _int8_mma_unavailable()
            if why is not None:
                print(f"[00] SKIP  {label:16s} {why}")
                continue
        a, b = _mkdata(M, N, K, torch_in, rng)

        g, A, B, C = _build_matmul_graph(M, N, K, io_dt, out_dt, compute_dt)
        _build_plans(g, frost=True)
        c_frost = _execute(g, A, B, C, a, b, torch_out, M, N)
        torch.cuda.synchronize()

        ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
        if torch_out is torch.float32:
            torch.testing.assert_close(c_frost, ref, atol=0, rtol=0)
        else:
            torch.testing.assert_close(c_frost, ref.to(torch_out), atol=1e-1, rtol=1e-2)

        if label == "bf16":
            g_ref, A_r, B_r, C_r = _build_matmul_graph(M, N, K, io_dt, out_dt, compute_dt)
            _build_plans(g_ref, frost=False)
            c_ref = _execute(g_ref, A_r, B_r, C_r, a, b, torch_out, M, N)
            torch.cuda.synchronize()
            torch.testing.assert_close(c_frost, c_ref, atol=1e-1, rtol=1e-2)
            print(f"[00] PASS  {label:16s} M={M} N={N} K={K}  (FROST == cuDNN native == torch)")
        else:
            print(f"[00] PASS  {label:16s} M={M} N={N} K={K}")

    _block_scale_case("nvfp4", M, N, K)
    _block_scale_case("mxfp8", M, N, K)


if __name__ == "__main__":
    main()
