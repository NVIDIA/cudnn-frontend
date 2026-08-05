# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multirank EP forward+backward parity vs the single-process reference.

Launch:  torchrun --standalone --nproc-per-node=4 parity_ep_vs_reference.py

Every rank deterministically builds the SAME global problem from a shared
seed (CPU generator), so the single-process reference can be evaluated
redundantly on each rank with the full expert set and the full global batch;
the EP layer runs on this rank's token shard with its local expert shard.

Checks per (dtype, routing) case:
- forward:        ep_out            == ref_out[rank shard]
- dgrad:          x_local.grad      == x_ref.grad[rank shard]
- router grad:    topk_w_local.grad == topk_w_ref.grad[rank shard]
- wgrad:          w13/w2 local .grad == full ref .grad[local expert slice]
  (wgrads need no cross-rank reduction: every copy of a token routed to a
  local expert already arrived here in forward)

Row order inside each expert's GEMM is identical between EP and reference
(global tokens are concatenated in rank order, and dispatch preserves
source-rank-then-copy order within each expert), so tolerances can stay
tight even in bf16.
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pt import EpConfig, MoEEpTrainingLayer, ReferenceMoE  # noqa: E402

TOLERANCES = {
    torch.float32: dict(rtol=2e-4, atol=2e-5),
    torch.bfloat16: dict(rtol=3e-2, atol=3e-2),
}
# topk_weights grads are always fp32 (leaf dtype), but they accumulate
# through the activation dtype, so tolerance follows the activation dtype.


def make_global_problem(
    *, seed, world, tokens_per_rank, num_experts, top_k, hidden, intermediate, skew
):
    """Same seed => bit-identical problem on every rank (CPU generator)."""
    gen = torch.Generator().manual_seed(seed)
    total = world * tokens_per_rank
    x = torch.randn(total, hidden, generator=gen) * 0.5
    scores = torch.randn(total, num_experts, generator=gen)
    if skew:
        # Pile most of the traffic onto rank 0's experts to exercise the
        # reversed uneven splits in backward.
        scores[:, 0] += 6.0
        scores[:, 1] += 4.0
    topk_weights, topk_ids = torch.topk(torch.softmax(scores, dim=-1), top_k)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    w13 = torch.randn(num_experts, 2 * intermediate, hidden, generator=gen) * hidden**-0.5
    w2 = torch.randn(num_experts, hidden, intermediate, generator=gen) * intermediate**-0.5
    grad_out = torch.randn(total, hidden, generator=gen)
    return x, topk_ids, topk_weights.float(), w13, w2, grad_out


def check(name, actual, expected, dtype, failures):
    tol = TOLERANCES[dtype]
    a, e = actual.float(), expected.float()
    max_abs = (a - e).abs().max().item() if a.numel() else 0.0
    ok = torch.allclose(a, e, **tol)
    if not ok:
        failures.append(f"{name}: max_abs_diff={max_abs:.3e} (tol={tol})")
    return max_abs


def run_case(*, dtype, skew, rank, world, device):
    tokens_per_rank, num_experts, top_k = 64, 4 * world, 4
    hidden, intermediate = 128, 256

    x_g, ids_g, tw_g, w13_g, w2_g, gout_g = make_global_problem(
        seed=1234 + int(skew),
        world=world,
        tokens_per_rank=tokens_per_rank,
        num_experts=num_experts,
        top_k=top_k,
        hidden=hidden,
        intermediate=intermediate,
        skew=skew,
    )
    x_g = x_g.to(device=device, dtype=dtype)
    ids_g = ids_g.to(device)
    tw_g = tw_g.to(device)  # stays fp32
    w13_g = w13_g.to(device=device, dtype=dtype)
    w2_g = w2_g.to(device=device, dtype=dtype)
    gout_g = gout_g.to(device=device, dtype=dtype)

    # ---- single-process reference on the full global batch ----
    x_ref = x_g.detach().clone().requires_grad_()
    tw_ref = tw_g.detach().clone().requires_grad_()
    ref = ReferenceMoE(w13_g.detach().clone(), w2_g.detach().clone())
    ref_out = ref(x_ref, ids_g, tw_ref)
    ref_out.backward(gout_g)

    # ---- EP layer on this rank's shards ----
    lo, hi = rank * tokens_per_rank, (rank + 1) * tokens_per_rank
    num_local = num_experts // world
    elo, ehi = rank * num_local, (rank + 1) * num_local

    x_loc = x_g[lo:hi].detach().clone().requires_grad_()
    tw_loc = tw_g[lo:hi].detach().clone().requires_grad_()
    cfg = EpConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size=intermediate,
        ep_size=world,
        ep_rank=rank,
    )
    layer = MoEEpTrainingLayer(
        cfg,
        w13_g[elo:ehi].detach().clone(),
        w2_g[elo:ehi].detach().clone(),
    ).to(device)
    ep_out = layer(x_loc, ids_g[lo:hi], tw_loc)
    ep_out.backward(gout_g[lo:hi])

    # ---- compare ----
    failures = []
    check("forward out", ep_out, ref_out[lo:hi], dtype, failures)
    check("dX", x_loc.grad, x_ref.grad[lo:hi], dtype, failures)
    check("d(topk_weights)", tw_loc.grad, tw_ref.grad[lo:hi], dtype, failures)
    check("dW13", layer.w13.grad, ref.w13.grad[elo:ehi], dtype, failures)
    check("dW2", layer.w2.grad, ref.w2.grad[elo:ehi], dtype, failures)
    return failures


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    any_failed = False
    for dtype in (torch.float32, torch.bfloat16):
        for skew in (False, True):
            case = f"dtype={dtype} skew={skew}"
            failures = run_case(
                dtype=dtype, skew=skew, rank=rank, world=world, device=device
            )
            if failures:
                any_failed = True
                print(f"[rank {rank}] FAIL {case}:\n  " + "\n  ".join(failures))
            elif rank == 0:
                print(f"[rank 0] PASS {case}")
            dist.barrier()

    dist.barrier()
    dist.destroy_process_group()
    if any_failed:
        sys.exit(1)
    if rank == 0:
        print("all parity cases passed")


if __name__ == "__main__":
    main()
