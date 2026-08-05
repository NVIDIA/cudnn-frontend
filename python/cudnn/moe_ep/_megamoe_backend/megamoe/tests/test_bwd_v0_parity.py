# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Gradient parity for the M1 pool-reuse backward (bwd_v0.pool_backward).

Drives the SAME MegaMoeHybridMxfp8Layer through three backwards — the
autograd fake-quant replay oracle, the manual fp8 backward, and the new
pool-reuse backward — one forward each (fresh pools per generation), same
dout, and compares (dx, dtw, dw13, dw2).

Acceptance is distance to the replay oracle (the STE-exact fake-quant
gradient of the quantized forward). pool and fp8 factor tw differently
(fp8/replay quantize tw*dout; pool applies tw in fp32 after quantizing
dout — the kernel's factorization), so pool-vs-fp8 sits at mxfp8
rounding-noise level (~5% rel L2) BY CONSTRUCTION and is reported as info,
not asserted tightly.

Launch:
    MEGA_NO_DIST=1 python -m megamoe.tests.test_bwd_v0_parity
    torchrun --nproc_per_node=4 --standalone -m megamoe.tests.test_bwd_v0_parity
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dataclasses

import torch

import megamoe.repo_path  # noqa: F401

from megamoe.forward import MegaMoeForwardConfig

TOKENS = 4096
HIDDEN = 1024
INTERMEDIATE = 512
NUM_EXPERTS = 32
TOPK = 4
SEED = 4242

TOL_VS_REPLAY = 0.06   # rel-L2 vs the fake-quant oracle (quant-noise floor)

_failures = []


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def gen_problem(device, rank):
    gen = torch.Generator(device=device).manual_seed(SEED + rank)
    x = (torch.randn((TOKENS, HIDDEN), device=device, generator=gen) / 10.0).bfloat16()
    scores = torch.rand((TOKENS, NUM_EXPERTS), device=device, generator=gen)
    _, ids = scores.topk(TOPK, dim=-1)
    w = torch.rand((TOKENS, TOPK), device=device, generator=gen) + 0.1
    tw = (w / w.sum(-1, keepdim=True)).float()
    dout = torch.randn((TOKENS, HIDDEN), device=device, generator=gen).bfloat16()
    return x, ids.long(), tw, dout


def gen_weights(device):
    gen = torch.Generator(device=device).manual_seed(SEED)  # same on all ranks
    w13 = (torch.randn((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
                       device=device, generator=gen) * 0.05).bfloat16()
    w2 = (torch.randn((NUM_EXPERTS, HIDDEN, INTERMEDIATE),
                      device=device, generator=gen) * 0.05).bfloat16()
    return w13, w2


def main():
    no_dist = os.environ.get("MEGA_NO_DIST", "0") == "1"
    if no_dist:
        rank, world = 0, 1
        torch.cuda.set_device(0)
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                "nccl", init_method="tcp://127.0.0.1:29531", world_size=1, rank=0
            )
    else:
        from src.bootstrap import init_dist_and_nvshmem

        _, rank, world, _ = init_dist_and_nvshmem()
    device = torch.device("cuda", torch.cuda.current_device())

    from pt import EpConfig, QuantConfig
    from megamoe.training import MegaMoeHybridMxfp8Layer

    x, ids, tw, dout = gen_problem(device, rank)
    w13_full, w2_full = gen_weights(device)
    e_loc = NUM_EXPERTS // world
    w13 = w13_full[rank * e_loc : (rank + 1) * e_loc].clone()
    w2 = w2_full[rank * e_loc : (rank + 1) * e_loc].clone()

    ep_cfg = EpConfig(
        num_experts=NUM_EXPERTS, top_k=TOPK, hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE, ep_size=world, ep_rank=rank,
    )
    mm_cfg = MegaMoeForwardConfig(
        max_tokens_per_rank=TOKENS, hidden=HIDDEN, intermediate=INTERMEDIATE,
        num_total_experts=NUM_EXPERTS, num_topk=TOPK,
    )
    mm_cfg = dataclasses.replace(
        mm_cfg, impl=dataclasses.replace(mm_cfg.impl, generate_c=True)
    )
    layer = MegaMoeHybridMxfp8Layer(
        ep_cfg, mm_cfg, w13, w2,
        qcfg=QuantConfig(fprop_fmt="mxfp8", quant_bprop=True),
        bwd_impl="pool",
    )

    grads = {}
    for impl in ("replay", "fp8", "pool", "mega"):
        layer.bwd_impl = impl
        layer.w13.grad = layer.w2.grad = None
        x_l = x.detach().clone().requires_grad_()
        tw_l = tw.detach().clone().requires_grad_()
        out = layer(x_l, ids, tw_l)
        dx, dtw, dw13, dw2 = torch.autograd.grad(
            out, (x_l, tw_l, layer.w13, layer.w2), dout
        )
        grads[impl] = dict(dx=dx, dtw=dtw, dw13=dw13, dw2=dw2)
        torch.cuda.synchronize()

    for name in ("dx", "dtw", "dw13", "dw2"):
        for a, b, tol in (
            ("pool", "replay", TOL_VS_REPLAY),
            ("fp8", "replay", TOL_VS_REPLAY),
            ("mega", "replay", TOL_VS_REPLAY),
            ("mega", "pool", None),   # info: same factorization, kernel GEMMs
            ("pool", "fp8", None),    # info: differs by tw factorization noise
        ):
            r = rel_l2(grads[a][name], grads[b][name])
            if tol is None:
                if rank == 0:
                    print(f"  [INFO] {a} vs {b:<6} {name:<5} rel_l2={r:.4f}")
                continue
            ok = r < tol
            if not ok:
                _failures.append(f"{name}: {a} vs {b}")
            if rank == 0:
                print(f"  [{'PASS' if ok else 'FAIL'}] {a} vs {b:<6} {name:<5} "
                      f"rel_l2={r:.4f} (tol {tol})")

    layer.finalize()
    if rank == 0:
        print("BWD_V0 PARITY " + ("PASS" if not _failures else f"FAIL ({_failures})"))
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
