# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parity test: MegaMoeMxfp8Forward vs compute_megamoe_reference_mxfp8.

Launch (from moe_ep_training/):

    torchrun --nproc_per_node=4 --standalone -m megamoe.tests.test_forward_parity

or single-rank without NVSHMEM:

    MEGA_NO_DIST=1 python -m megamoe.tests.test_forward_parity

Each rank builds real bf16 inputs + local expert weights, runs the hackable
forward, then reads back the DEVICE-quantized activation/sf and the staged
routing tensors, all-gathers the global view, and feeds the repo's own MXFP8
reference (apply_topk_in_fc1=True, deepgemm graph).  Tolerances match the
repo validator (atol=rtol=1e-2): the only unmodeled error sources are the
fc1-out MXFP8 round-trip RTNE vs the host quantizer and fp32 GEMM noise.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

import megamoe.repo_path  # noqa: F401

from common.host_utils import compare_and_report_mismatches
from common.megamoe_constants import Mxfp8BlockSize
from moe_nvfp4_swapab.runner_common import ceil_div
from moe_mxfp8_glu.mega_reference_mxfp8 import compute_megamoe_reference_mxfp8

from megamoe.forward import MegaMoeForwardConfig, MegaMoeMxfp8Forward

_NO_DIST = bool(int(os.environ.get("MEGA_NO_DIST", "0")))

# problem shape (small; multi-rank exercises dispatch/combine across NVLink)
TOKENS = 128
HIDDEN = 2048
INTERMEDIATE = 512          # per-branch I  ->  kernel gate+up width 1024
NUM_TOTAL_EXPERTS = 32
TOPK = 4
KIND = "mxfp8_e4m3"
SEED = 1234


def _all_gather_stack(t: torch.Tensor, world_size: int) -> torch.Tensor:
    """all_gather -> (world_size, *t.shape); fp8/e8m0 go through uint8 views."""
    if world_size == 1:
        return t.unsqueeze(0)
    byte_backed = t.element_size() == 1 and t.dtype != torch.uint8
    src = t.contiguous()
    view = src.view(torch.uint8) if byte_backed else src
    out = [torch.empty_like(view) for _ in range(world_size)]
    torch.distributed.all_gather(out, view)
    stacked = torch.stack(out, dim=0)
    return stacked.view(t.dtype) if byte_backed else stacked


def _make_routing(gen, num_tokens, rank):
    scores = torch.rand(
        (num_tokens, NUM_TOTAL_EXPERTS), device="cuda", generator=gen
    )
    _, topk_idx = scores.topk(TOPK, dim=-1)
    w = torch.rand((num_tokens, TOPK), device="cuda", generator=gen) + 0.1
    topk_weights = w / w.sum(dim=-1, keepdim=True)
    return topk_idx.to(torch.int64), topk_weights.float()


def _run_and_check(fwd, rank, world_size, num_tokens, weights_q, tag):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED + 1000 * rank + num_tokens)
    x = (torch.rand((num_tokens, HIDDEN), device="cuda", generator=gen) - 0.5).bfloat16()
    topk_idx, topk_weights = _make_routing(gen, num_tokens, rank)

    out = fwd(x, topk_idx, topk_weights)
    torch.cuda.synchronize()

    # global view for the reference: device-quantized activation + staged
    # (repacked, padding-masked) routing, straight from the forward's buffers.
    sf_cols = ceil_div(HIDDEN, Mxfp8BlockSize)
    g_act = _all_gather_stack(fwd.my_activation, world_size)
    g_act_sf = _all_gather_stack(fwd.my_activation_sf[:, :sf_cols], world_size)
    g_topk_idx = _all_gather_stack(fwd.my_topk_idx, world_size)
    g_topk_w = _all_gather_stack(fwd.my_topk_weights, world_size)

    # weights: reference wants (R, E, hidden, 2I) hidden-stride-1 fc1 and
    # (R, E, I, hidden) I-stride-1 fc2 -- gather the contiguous storages and
    # permute per rank.
    g_fc1 = _all_gather_stack(weights_q.fc1_weight.permute(0, 2, 1), world_size)
    g_fc1_sf = _all_gather_stack(weights_q.fc1_weight_sf_plain, world_size)
    g_fc2 = _all_gather_stack(weights_q.fc2_weight.permute(0, 2, 1), world_size)
    g_fc2_sf = _all_gather_stack(weights_q.fc2_weight_sf_plain, world_size)

    combine_ref = compute_megamoe_reference_mxfp8(
        input_activation=g_act,
        input_activation_sf=g_act_sf,
        input_topk_idx=g_topk_idx,
        input_topk_weights=g_topk_w,
        fc1_weight=g_fc1.permute(0, 1, 3, 2),  # (R, E, H, 2I), H stride-1
        fc1_weight_sf=g_fc1_sf,
        fc2_weight=g_fc2.permute(0, 1, 3, 2),  # (R, E, I, H), I stride-1
        fc2_weight_sf=g_fc2_sf,
        ab_dtype=fwd.torch_ab_dtype,
        fc2_output_dtype=torch.bfloat16,
        combine_format=fwd._combine_format,
        apply_topk_in_fc1=True,
    )
    # apply_topk_in_fc1: weights already folded in -> plain sum over topk.
    ref_reduced = combine_ref[rank, :num_tokens].float().sum(dim=1)

    compare_and_report_mismatches(
        out.float(),
        ref_reduced,
        name=f"[{tag}] combine_output[rank{rank}] T={num_tokens}",
        atol=1e-2,
        rtol=1e-2,
    )


def main() -> int:
    if _NO_DIST:
        torch.cuda.set_device(0)
        rank, world_size = 0, 1
    else:
        from src.bootstrap import init_dist_and_nvshmem, finalize_dist_and_nvshmem
        _, rank, world_size, _ = init_dist_and_nvshmem()

    E_local = NUM_TOTAL_EXPERTS // world_size
    cfg = MegaMoeForwardConfig(
        max_tokens_per_rank=TOKENS,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        num_total_experts=NUM_TOTAL_EXPERTS,
        num_topk=TOPK,
        kind=KIND,
    )
    # MEGA_DEDUP_DISPATCH=1 exercises the rank-dedup dispatch kernel variant;
    # MEGA_COMBINE_IFR=1 the in-flight combine reduce; MEGA_COMBINE_PRE=1 the
    # same-rank combine pre-reduce (implies the other two).
    if bool(int(os.environ.get("MEGA_COMBINE_PRE", "0"))):
        cfg.impl.dedup_dispatch = True
        cfg.impl.combine_in_flight_reduce = True
        cfg.impl.combine_pre_reduce = True
    if bool(int(os.environ.get("MEGA_COMBINE_IFR", "0"))):
        cfg.impl.combine_in_flight_reduce = True
    if bool(int(os.environ.get("MEGA_DEDUP_DISPATCH", "0"))):
        cfg.impl.dedup_dispatch = True
    if rank == 0:
        print(f"[parity] impl: dedup={cfg.impl.dedup_dispatch} "
              f"ifr={cfg.impl.combine_in_flight_reduce} "
              f"pre={cfg.impl.combine_pre_reduce}")
    fwd = MegaMoeMxfp8Forward(cfg, rank=rank, world_size=world_size)

    wgen = torch.Generator(device="cuda")
    wgen.manual_seed(SEED + 7 * rank)
    scale = 0.05  # keep fc outputs O(1) so the 1e-2 atol is meaningful
    w13 = (torch.randn((E_local, 2 * INTERMEDIATE, HIDDEN), device="cuda",
                       generator=wgen) * scale).bfloat16()
    w2 = (torch.randn((E_local, HIDDEN, INTERMEDIATE), device="cuda",
                      generator=wgen) * scale).bfloat16()
    fwd.load_weights(w13, w2)

    _run_and_check(fwd, rank, world_size, TOKENS, fwd._weights, "full")
    _run_and_check(fwd, rank, world_size, TOKENS // 2, fwd._weights, "padded")

    # weight update in place (no recompile), then re-verify
    fwd.load_weights((w13.float() * 1.5).bfloat16(), (w2.float() * 1.5).bfloat16())
    _run_and_check(fwd, rank, world_size, TOKENS, fwd._weights, "reload")

    if rank == 0:
        print("ALL PARITY CHECKS PASS")

    fwd.finalize()
    if not _NO_DIST:
        finalize_dist_and_nvshmem()
    return 0


if __name__ == "__main__":
    sys.exit(main())
