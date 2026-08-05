# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multi-rank hybrid training validation: kernel fprop + fp8/replay bprop
with REAL all-to-all adjoints.

Launch (from moe_ep_training/):

    torchrun --nproc_per_node=4 --standalone -m megamoe.tests.test_hybrid_training_dist

Every rank regenerates the same global problem from a shared seed (the
repo's parity-test pattern), evaluates the single-process fp4/fp8 reference
on the full batch, and runs the hybrid layer on its token/expert shard:

1. hybrid replay-bwd grads   == ReferenceMoEFp4(mxfp8) shard slices (tight)
2. hybrid fp8-bwd grads      ~  reference (bf16 grouped-GEMM rounding)
3. per-rank fwd / fp8-bwd timing with real NCCL a2a in backward
"""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

import megamoe.repo_path  # noqa: F401

from megamoe.forward import MegaMoeForwardConfig
from megamoe.training import MegaMoeHybridMxfp8Layer

from pt import EpConfig, QuantConfig, ReferenceMoEFp4

TOKENS_PER_RANK = 2048
HIDDEN = 1024
INTERMEDIATE = 512
NUM_EXPERTS = 32
TOPK = 4
SEED = 4242
ITERS = 10


def rel_err(a, b):
    d = b.float().norm().item()
    return (a.float() - b.float()).norm().item() / max(d, 1e-30)


def gen_global(world, device):
    gen = torch.Generator(device=device).manual_seed(SEED)
    total = world * TOKENS_PER_RANK
    x = torch.randn((total, HIDDEN), device=device, generator=gen) / 10.0
    hot = torch.randperm(HIDDEN, generator=gen, device=device)[:10]
    x[:, hot] *= 30.0
    scores = torch.rand((total, NUM_EXPERTS), device=device, generator=gen)
    _, ids = scores.topk(TOPK, dim=-1)
    w = torch.rand((total, TOPK), device=device, generator=gen) + 0.1
    tw = (w / w.sum(-1, keepdim=True)).float()
    w13 = (torch.randn((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
                       device=device, generator=gen) * 0.05).bfloat16()
    w2 = (torch.randn((NUM_EXPERTS, HIDDEN, INTERMEDIATE),
                      device=device, generator=gen) * 0.05).bfloat16()
    gout = torch.randn((total, HIDDEN), device=device, generator=gen).bfloat16()
    return x.bfloat16(), ids.long(), tw, w13, w2, gout


def run_hybrid(layer, x, ids, tw, gout):
    x_h = x.detach().clone().requires_grad_()
    tw_h = tw.detach().clone().requires_grad_()
    out = layer(x_h, ids, tw_h)
    out.backward(gout)
    return out.detach(), x_h.grad, tw_h.grad, layer.w13.grad, layer.w2.grad


def main():
    from src.bootstrap import init_dist_and_nvshmem, finalize_dist_and_nvshmem

    _, rank, world, _ = init_dist_and_nvshmem()
    device = torch.device("cuda", torch.cuda.current_device())

    x_g, ids_g, tw_g, w13_g, w2_g, gout_g = gen_global(world, device)
    qcfg = QuantConfig(fprop_fmt="mxfp8", quant_bprop=True)

    # single-process reference on the global batch (redundant per rank)
    x_r = x_g.detach().clone().requires_grad_()
    tw_r = tw_g.detach().clone().requires_grad_()
    ref = ReferenceMoEFp4(w13_g.detach().clone(), w2_g.detach().clone(), qcfg)
    ref_out = ref(x_r, ids_g, tw_r)
    ref_out.backward(gout_g)

    lo, hi = rank * TOKENS_PER_RANK, (rank + 1) * TOKENS_PER_RANK
    n_local = NUM_EXPERTS // world
    elo, ehi = rank * n_local, (rank + 1) * n_local

    ep_cfg = EpConfig(
        num_experts=NUM_EXPERTS, top_k=TOPK, hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE, ep_size=world, ep_rank=rank,
    )
    mm_cfg = MegaMoeForwardConfig(
        max_tokens_per_rank=TOKENS_PER_RANK, hidden=HIDDEN,
        intermediate=INTERMEDIATE, num_total_experts=NUM_EXPERTS,
        num_topk=TOPK,
    )
    # bwd_impl="mega" reads the forward's fc1_c stash (generate_c); harmless
    # for replay/fp8 (they just ignore the extra stash).
    import dataclasses
    mm_cfg = dataclasses.replace(
        mm_cfg, impl=dataclasses.replace(mm_cfg.impl, generate_c=True)
    )

    names = ("fwd", "dX", "dTW", "dW13", "dW2")
    refs = (ref_out[lo:hi], x_r.grad[lo:hi], tw_r.grad[lo:hi],
            ref.w13.grad[elo:ehi], ref.w2.grad[elo:ehi])

    any_failed = False
    layers = {}
    for impl, tol in (("replay", 1e-3), ("fp8", 0.05), ("mega", 0.06)):
        layer = MegaMoeHybridMxfp8Layer(
            ep_cfg, mm_cfg,
            w13_g[elo:ehi].detach().clone(), w2_g[elo:ehi].detach().clone(),
            qcfg, bwd_impl=impl,
        )
        layers[impl] = layer
        got = run_hybrid(layer, x_g[lo:hi], ids_g[lo:hi], tw_g[lo:hi], gout_g[lo:hi])
        errs = {n: rel_err(g, r) for n, g, r in zip(names, got, refs)}
        line = "  ".join(f"{n}={e:.4f}" for n, e in errs.items())
        print(f"[rank {rank}] bwd_impl={impl:<6} vs reference: {line}")
        # fwd goes through the kernel either way; grads carry the impl tol.
        if errs["fwd"] > 0.05 or any(e > tol for n, e in errs.items() if n != "fwd"):
            any_failed = True
            print(f"[rank {rank}] FAIL bwd_impl={impl} (tol={tol})")

    # ---- timing with real a2a (collective; all ranks iterate together) ----
    def time_layer(layer):
        torch.distributed.barrier()
        fwd_ms, bwd_ms = [], []
        for _ in range(3 + ITERS):
            x_h = x_g[lo:hi].detach().clone().requires_grad_()
            tw_h = tw_g[lo:hi].detach().clone().requires_grad_()
            s, m, e = (torch.cuda.Event(True) for _ in range(3))
            s.record()
            out = layer(x_h, ids_g[lo:hi], tw_h)
            m.record()
            out.backward(gout_g[lo:hi])
            e.record()
            torch.cuda.synchronize()
            fwd_ms.append(s.elapsed_time(m))
            bwd_ms.append(m.elapsed_time(e))
            torch.distributed.barrier()
        return statistics.median(fwd_ms[3:]), statistics.median(bwd_ms[3:])

    for name in ("fp8", "mega"):
        fwd, bwd = time_layer(layers[name])
        print(f"[rank {rank}] T/rank={TOKENS_PER_RANK} world={world}: "
              f"fwd {fwd:.3f} ms  {name}-bwd {bwd:.3f} ms ({bwd/fwd:.1f}x fwd)")

    for layer in layers.values():
        layer.finalize()
    torch.distributed.barrier()
    if rank == 0 and not any_failed:
        print("HYBRID DIST PASS")
    finalize_dist_and_nvshmem()
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
