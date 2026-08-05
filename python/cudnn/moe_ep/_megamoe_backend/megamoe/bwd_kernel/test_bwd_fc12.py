# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""M2.A parity: the backward FC12 kernel vs a torch mxfp8 reference chain.

Standalone (no NVSHMEM): feeds pool-order inputs directly —

    activation  = quantized DOUT pool (Mp, H) fp8 + swizzled SFs
    fc1_weight  = W2^T  (E, H, I) view   (gemm1: dA = dout @ W2^T)
    fc1_c       = forward's raw (gate, up) stash (Mp, 2I) bf16 interleaved
    topk_scores = per-pool-row tw        (gemm1 epilogue: dA *= tw)
    fc2_weight  = W13^T (E, 2I, H) view  (gemm2: dx = [dg,du] @ W13^T)

and checks BOTH the 2I-wide fc1_output workspace (DFC1, dequantized) and the
final fc2_output (dXG) against the same chain computed in torch with the
identical quantized operands.

Launch:  python -m megamoe.bwd_kernel.test_bwd_fc12
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

import megamoe.repo_path  # noqa: F401

from megamoe.fp8_bwd import _pack_scales_rowgroups, _round_up
from megamoe.quant_kernels import mxfp8_rowquant
from megamoe.bwd_kernel.weights_bwd import quantize_moe_weights_mxfp8_bwd
from megamoe.weights import interleave_gate_up

HIDDEN = 1024
INTERMEDIATE = 512      # per-branch I; gemm1 N = I, gemm2 K = 2I
NUM_EXPERTS = 4
COUNTS = [300, 128, 0, 517]
SEED = 4242


def dequant(q, sf_plain):
    scale = torch.exp2(sf_plain.view(torch.uint8).float() - 127.0)
    return q.float() * scale.repeat_interleave(32, dim=-1)


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(SEED)
    E, H, I = NUM_EXPERTS, HIDDEN, INTERMEDIATE

    padded = [_round_up(n, 128) for n in COUNTS]
    doffs = [sum(padded[:i]) for i in range(E)]
    Mp = sum(padded)
    valid = torch.cat([
        torch.arange(o, o + n, device=device) for o, n in zip(doffs, COUNTS) if n
    ])

    # ---- synthetic forward state: stash + tw + dout, zero pad rows ----
    def pool_randn(cols, scale):
        t = torch.zeros((Mp, cols), device=device, dtype=torch.bfloat16)
        t[valid] = (torch.randn((valid.numel(), cols), device=device,
                                generator=gen) * scale).bfloat16()
        return t

    x_pool = pool_randn(H, 0.1)
    dout_pool = pool_randn(H, 1.0)
    tw = torch.zeros((Mp,), device=device, dtype=torch.float32)
    tw[valid] = torch.rand((valid.numel(),), device=device, generator=gen) + 0.1
    w13 = (torch.randn((E, 2 * I, H), device=device, generator=gen) * 0.05).bfloat16()
    w2 = (torch.randn((E, H, I), device=device, generator=gen) * 0.05).bfloat16()

    # fc1_c = raw pre-SwiGLU gate+up in kernel interleave, per expert slot
    w13_int = interleave_gate_up(w13)
    fc1_c = torch.zeros((Mp, 2 * I), device=device, dtype=torch.bfloat16)
    for e in range(E):
        rows = slice(doffs[e], doffs[e] + COUNTS[e])
        fc1_c[rows] = (x_pool[rows].float() @ w13_int[e].float().t()).bfloat16()

    # ---- kernel operands ----
    qw = quantize_moe_weights_mxfp8_bwd(w13, w2)
    dout_q, dout_sf = mxfp8_rowquant(dout_pool)          # (Mp,H) fp8 + (Mp,H/32)
    offs_padded = torch.tensor(
        [sum(padded[: i + 1]) for i in range(E)], device=device, dtype=torch.int32
    )
    act_sf_swz = _pack_scales_rowgroups(dout_sf, offs_padded)
    offs_valid = torch.tensor(
        [sum(COUNTS[: i + 1]) for i in range(E)], device=device, dtype=torch.int32
    )

    # ---- torch reference chain (same quantized bytes) ----
    w2t_fq = dequant(
        qw.gemm1_weight.permute(0, 2, 1).reshape(-1, H),
        qw.gemm1_weight_sf_plain.reshape(-1, H // 32),
    ).view(E, I, H)
    w13t_fq = dequant(
        qw.gemm2_weight.permute(0, 2, 1).reshape(-1, 2 * I),
        qw.gemm2_weight_sf_plain.reshape(-1, 2 * I // 32),
    ).view(E, H, 2 * I)
    dout_fq = dequant(dout_q, dout_sf)

    ref_dfc1 = torch.zeros((Mp, 2 * I), device=device, dtype=torch.float32)
    for e in range(E):
        rows = slice(doffs[e], doffs[e] + COUNTS[e])
        dacc = dout_fq[rows] @ w2t_fq[e].t()
        da = dacc * tw[rows, None]
        pair = fc1_c[rows].float().view(-1, I // 32, 2, 32)
        g, u = pair[:, :, 0].reshape(-1, I), pair[:, :, 1].reshape(-1, I)
        s = torch.sigmoid(g)
        dg = da * u * s * (1 + g * (1 - s))
        du = da * F.silu(g)
        out = torch.stack(
            [dg.view(-1, I // 32, 32), du.view(-1, I // 32, 32)], dim=2
        ).reshape(-1, 2 * I)                              # (dg, du) interleave
        ref_dfc1[rows] = out
    # fc2 ref consumes the REQUANTIZED dfc1 (what the kernel's gemm2 reads)
    from pt.quant import fake_quant_mxfp8

    dfc1_fq = fake_quant_mxfp8(ref_dfc1.bfloat16())
    ref_out = torch.zeros((Mp, H), device=device, dtype=torch.float32)
    for e in range(E):
        rows = slice(doffs[e], doffs[e] + COUNTS[e])
        ref_out[rows] = dfc1_fq[rows] @ w13t_fq[e].t()

    # ---- kernel ----
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils

    from common.megamoe_constants import SfPaddingBlock
    from megamoe.bwd_kernel.kernel_bwd_fc12 import Sm100SwigluMxfp8Fc12Kernel

    mma_tiler = (256, 256, 128)
    cluster = (2, 1, 1)
    max_active = utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
    kernel = Sm100SwigluMxfp8Fc12Kernel(
        mma_tiler_mnk=mma_tiler,
        cluster_shape_mnk=cluster,
        use_2cta_instrs=True,
        group_hint=max_active,
        token_padding_block=128,
        sf_padding_block=SfPaddingBlock,
        load_balance_mode="static",
        static_expert_shape=(E, I, H),      # gemm1 N = I
        ab_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=32,
        apply_topk_in_fc1=True,
        generate_c=True,
        use_stg_fc1=True,
    )

    def to_cute(t, align=16):
        ct = cutlass_torch.from_dlpack(t, assumed_align=align)
        return ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))

    fc1_weight_t = qw.gemm1_weight       # (E, H, I) view
    fc2_weight_t = qw.gemm2_weight       # (E, 2I, H) view
    ws_bytes = kernel.get_workspace_size_in_bytes(
        to_cute(dout_q), to_cute(fc1_weight_t)
    )
    workspace = torch.zeros((ws_bytes,), dtype=torch.uint8, device=device)

    # partition (mirror runner_fc12_common._partition_workspace, downproj=2I)
    downproj = 2 * I
    sf_rows_upper = Mp + E * SfPaddingBlock
    sf_block_cols = ((downproj // 32) + 3) // 4 * 4
    counter_slots = (Mp + mma_tiler[1] - 1) // mma_tiler[1] + E
    n0 = Mp * downproj
    n1 = sf_rows_upper * sf_block_cols
    n2 = counter_slots * 4
    fc1_output = workspace[:n0].view(torch.float8_e4m3fn).view(Mp, downproj)
    fc1_output_sf = workspace[n0 : n0 + n1].view(sf_rows_upper, sf_block_cols)
    fc1_done_counter = workspace[n0 + n1 : n0 + n1 + n2].view(torch.int32)

    fc2_output = torch.full((Mp, H), float("nan"), device=device, dtype=torch.bfloat16)
    dfc1_stash = torch.full(
        (Mp, 2 * I), float("nan"), device=device, dtype=torch.bfloat16
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    runtime_kwargs = dict(
        activation=to_cute(dout_q),
        activation_sf=to_cute(act_sf_swz),
        fc1_weight=to_cute(fc1_weight_t),
        fc1_weight_sf=to_cute(qw.gemm1_weight_sf),
        fc1_output=to_cute(fc1_output),
        fc1_output_sf=to_cute(fc1_output_sf),
        fc2_weight=to_cute(fc2_weight_t),
        fc2_weight_sf=to_cute(qw.gemm2_weight_sf),
        fc2_output=to_cute(fc2_output),
        topk_scores=to_cute(tw),
        fc1_done_counter=to_cute(fc1_done_counter),
        offs=to_cute(offs_valid),
        fc1_c=to_cute(fc1_c),
        fc1_c_out=to_cute(dfc1_stash),
        stream=stream,
    )
    compile_kwargs = dict(runtime_kwargs)
    compile_kwargs["max_active_clusters"] = max_active
    compiled = cute.compile(kernel, **compile_kwargs)
    compiled(**runtime_kwargs)
    torch.cuda.synchronize()

    # ---- compare ----
    ok = True
    got_stash = dfc1_stash.float()[valid]
    want_stash = ref_dfc1[valid]
    r0 = ((got_stash - want_stash).norm() / want_stash.norm().clamp_min(1e-12)).item()
    print(f"dfc1 bf16 stash   rel_l2 = {r0:.4e}")
    ok &= r0 < 5e-3  # bf16 rounding only — same operands, no requant

    got_out = fc2_output.float()[valid]
    want_out = ref_out[valid]
    r2 = ((got_out - want_out).norm() / want_out.norm().clamp_min(1e-12)).item()
    print(f"fc2_output (dXG)  rel_l2 = {r2:.4e}")
    ok &= r2 < 2e-2

    print("BWD_FC12 " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
