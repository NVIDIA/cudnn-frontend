# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multirank fp4 EP forward+backward parity vs the single-process fp4 reference.

Launch:  torchrun --standalone --nproc-per-node=4 parity_ep_vs_reference_fp4.py

Same scaffolding as parity_ep_vs_reference.py, but BOTH sides run the
simulated-NVFP4 expert FFN with the same QuantConfig, so this checks that
the EP wiring (dispatch/combine/grouping) is transparent to quantization:
each expert sees the identical token slab in the identical order on both
sides, hence identical block scales, identical fake-quant values, and
grads matching to bf16 tolerance. Stochastic rounding stays OFF here
(nondeterministic by design); it is covered by test_fp4_qat_numerics.
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pt import (  # noqa: E402
    EpConfig,
    MoEEpTrainingLayerFp4,
    QuantConfig,
    ReferenceMoEFp4,
)
from pt.tests.parity_ep_vs_reference import (  # noqa: E402
    TOLERANCES,
    check,
    make_global_problem,
)

QCFGS = {
    "nvfp4": QuantConfig(),
    "nvfp4+tq": QuantConfig(turboquant=True),
    "nvfp4+qb": QuantConfig(quant_bprop=True),
    "nvfp4+tq+qb": QuantConfig(turboquant=True, quant_bprop=True),
    "mxfp8+tq+qb": QuantConfig(fprop_fmt="mxfp8", turboquant=True, quant_bprop=True),
    "nvfp4/8bw": QuantConfig(bprop_fmt="mxfp8", quant_bprop=True),
}


def run_case(*, qcfg, skew, rank, world, device):
    dtype = torch.bfloat16
    tokens_per_rank, num_experts, top_k = 64, 4 * world, 4
    hidden, intermediate = 128, 256  # hidden == rotation_block

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

    # ---- single-process fp4 reference on the full global batch ----
    x_ref = x_g.detach().clone().requires_grad_()
    tw_ref = tw_g.detach().clone().requires_grad_()
    ref = ReferenceMoEFp4(w13_g.detach().clone(), w2_g.detach().clone(), qcfg).to(device)
    ref_out = ref(x_ref, ids_g, tw_ref)
    ref_out.backward(gout_g)

    # ---- fp4 EP layer on this rank's shards ----
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
    layer = MoEEpTrainingLayerFp4(
        cfg,
        w13_g[elo:ehi].detach().clone(),
        w2_g[elo:ehi].detach().clone(),
        qcfg,
    ).to(device)
    ep_out = layer(x_loc, ids_g[lo:hi], tw_loc)
    ep_out.backward(gout_g[lo:hi])

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
    for name, qcfg in QCFGS.items():
        for skew in (False, True):
            case = f"qcfg={name} skew={skew}"
            failures = run_case(
                qcfg=qcfg, skew=skew, rank=rank, world=world, device=device
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
        print("all fp4 parity cases passed")


if __name__ == "__main__":
    main()
