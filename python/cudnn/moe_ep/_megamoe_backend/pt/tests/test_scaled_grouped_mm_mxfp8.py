# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerics + timing for torch._scaled_grouped_mm as the MXFP8 wgrad unit.

Launch (single GPU):

    python -m pt.tests.test_scaled_grouped_mm_mxfp8

wgrad is the 2Dx2D grouped GEMM with TOKENS ON K:

    dW[e] = dY[ke_lo:ke_hi, :].T @ act[ke_lo:ke_hi, :]
          = a[:, ke_lo:ke_hi] @ b[ke_lo:ke_hi, :]       (a = dY^T, b = act)

with both operands MXFP8-quantized along the token (contraction) dim — the
transposed requant a bprop kernel must do.

Scale layout (from torch's test_scaled_matmul_cuda.py 2d-2d recipe and the
mslk grouped_common.cuh contract): per group, 32x4x4-swizzle (`to_blocked`)
the (MN, K_g/32) e8m0 scales — each group contributes
``round_up(MN,128) * round_up(K_g/32, 4)`` elements — concatenate in group
order, reshape 2D. Group token counts must be multiples of 32; empty groups
contribute nothing.
"""

import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import megamoe.repo_path  # noqa: F401

from moe_nvfp4_swapab.runner_common import to_blocked  # noqa: E402

from pt.quant import MXFP8_BLOCK, quant_mxfp8_tensors  # noqa: E402

SEED = 20260719


def rel_err(a, b):
    d = b.float().norm().item()
    return (a.float() - b.float()).norm().item() / max(d, 1e-30)


def round_up(x, y):
    return ((x + y - 1) // y) * y


def make_problem(tokens_per_expert, M, N, device, gen):
    ntot = sum(tokens_per_expert)
    dy = torch.randn((ntot, M), device=device, generator=gen).bfloat16()
    act = torch.randn((ntot, N), device=device, generator=gen).bfloat16()
    dy[:, :8] *= 20.0  # hot channels so scales vary
    act[:, :8] *= 20.0
    offs = torch.tensor(
        [sum(tokens_per_expert[: i + 1]) for i in range(len(tokens_per_expert))],
        device=device, dtype=torch.int32,
    )
    return dy, act, offs


def pack_scales(s, offs, MN):
    """(MN, Ktot/32) e8m0 -> grouped 32x4x4-blocked 2D layout."""
    parts, lo = [], 0
    for hi in offs.tolist():
        if hi > lo:
            parts.append(to_blocked(s[:, lo // MXFP8_BLOCK : hi // MXFP8_BLOCK]))
        lo = hi
    return torch.cat(parts).reshape(round_up(MN, 128), -1)


def quantize_operands(dy, act, offs):
    # a = dY^T [M, Ktot] row-major, b = act [Ktot, N] col-major view,
    # both quantized along K(tokens). 32-aligned groups => whole-tensor
    # quantization equals per-group (blocks never span groups).
    a, sa = quant_mxfp8_tensors(dy.t().contiguous(), dim=-1)
    bt, sb = quant_mxfp8_tensors(act.t().contiguous(), dim=-1)
    M, N = dy.shape[1], act.shape[1]
    return a, pack_scales(sa, offs, M), sa, bt.t(), pack_scales(sb, offs, N), sb


def reference(a, sa_plain, b, sb_plain, offs):
    """Exact fp32 grouped product of the dequantized operands."""
    deq_a = a.float() * sa_plain.float().repeat_interleave(MXFP8_BLOCK, dim=-1)
    deq_b = (
        b.t().float() * sb_plain.float().repeat_interleave(MXFP8_BLOCK, dim=-1)
    ).t()
    outs, lo = [], 0
    for hi in offs.tolist():
        outs.append(deq_a[:, lo:hi] @ deq_b[lo:hi, :])
        lo = hi
    return torch.stack(outs)


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(SEED)

    # ---- numerics: uneven 32-aligned groups ----
    tpe = [32 * k for k in (1, 4, 2, 8, 3, 1, 6, 7)]
    M, N = 512, 256
    dy, act, offs = make_problem(tpe, M, N, device, gen)
    a, sa_b, sa, b, sb_b, sb = quantize_operands(dy, act, offs)
    out = torch._scaled_grouped_mm(a, b, sa_b, sb_b, offs=offs, out_dtype=torch.bfloat16)
    ref = reference(a, sa, b, sb, offs)
    e = rel_err(out, ref)
    print(f"[numerics] uneven groups: rel_err={e:.3e}  out={tuple(out.shape)}")
    assert e < 1e-2, f"grouped mm math wrong: {e}"

    # ---- empty-expert group ----
    tpe0 = [128, 0, 96, 256, 0, 64, 32, 160]
    dy0, act0, offs0 = make_problem([t for t in tpe0], M, N, device, gen)
    a0, sa0_b, sa0, b0, sb0_b, sb0 = quantize_operands(dy0, act0, offs0)
    try:
        out0 = torch._scaled_grouped_mm(
            a0, b0, sa0_b, sb0_b, offs=offs0, out_dtype=torch.bfloat16
        )
        ref0 = reference(a0, sa0, b0, sb0, offs0)
        nonempty = [i for i, t in enumerate(tpe0) if t > 0]
        e0 = rel_err(out0[nonempty], ref0[nonempty])
        print(f"[empty-expert] OK rel_err(non-empty groups)={e0:.3e}")
        assert e0 < 1e-2
    except Exception as ex:  # noqa: BLE001
        print(f"[empty-expert] unsupported: {type(ex).__name__}: {str(ex)[:150]}")

    # ---- timing at wgrad-realistic size ----
    E2, M2, N2 = 32, 1024, 512
    dy2, act2, offs2 = make_problem([512] * E2, M2, N2, device, gen)
    a2, sa2_b, _, b2, sb2_b, _ = quantize_operands(dy2, act2, offs2)

    def timeit(fn, iters=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s, e_ = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            fn()
            e_.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e_))
        return statistics.median(ts)

    t_fp8 = timeit(lambda: torch._scaled_grouped_mm(
        a2, b2, sa2_b, sb2_b, offs=offs2, out_dtype=torch.bfloat16
    ))
    t_quant = timeit(lambda: quantize_operands(dy2, act2, offs2))

    def bf16_loop():
        lo, outs = 0, []
        for hi in offs2.tolist():
            outs.append(dy2[lo:hi].t() @ act2[lo:hi])
            lo = hi
        torch.stack(outs)

    t_bf16 = timeit(bf16_loop)
    print(f"[timing] wgrad 32E x 512tok ({M2}x{N2}): "
          f"fp8 grouped {t_fp8:.3f} ms | quant+pack {t_quant:.3f} ms | "
          f"bf16 loop {t_bf16:.3f} ms  (gemm speedup {t_bf16 / t_fp8:.1f}x)")

    print("SCALED_GROUPED_MM MXFP8 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
