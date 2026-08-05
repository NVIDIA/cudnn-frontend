# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""generate_c smoke: does the fc1-intermediate epilogue stash work, and what
does it cost fprop? (The stash-vs-recompute data point for training bwd.)

Launch:

    MEGA_NO_DIST=1 python -m megamoe.tests.smoke_generate_c

Runs the same problem through MegaMoeMxfp8Forward with impl.generate_c off
and on, checks output parity between the two, reports whether fc1_c was
populated, and times both (CUDA events, median of ITERS).
"""

import os
import sys

os.environ.setdefault("MEGA_NO_DIST", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dataclasses
import statistics

import torch

import megamoe.repo_path  # noqa: F401

from megamoe.forward import MegaMoeForwardConfig, MegaMoeMxfp8Forward

TOKENS = 4096
HIDDEN = 1024
INTERMEDIATE = 512
NUM_EXPERTS = 32
TOPK = 4
SEED = 4242
ITERS = 20


def gen_problem(device):
    gen = torch.Generator(device=device).manual_seed(SEED)
    x = (torch.randn((TOKENS, HIDDEN), device=device, generator=gen) / 10.0).bfloat16()
    scores = torch.rand((TOKENS, NUM_EXPERTS), device=device, generator=gen)
    _, ids = scores.topk(TOPK, dim=-1)
    w = torch.rand((TOKENS, TOPK), device=device, generator=gen) + 0.1
    tw = (w / w.sum(-1, keepdim=True)).float()
    w13 = (torch.randn((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
                       device=device, generator=gen) * 0.05).bfloat16()
    w2 = (torch.randn((NUM_EXPERTS, HIDDEN, INTERMEDIATE),
                      device=device, generator=gen) * 0.05).bfloat16()
    return x, ids.long(), tw, w13, w2


def bench(fwd, x, ids, tw):
    out = None
    for _ in range(3):
        out = fwd(x, ids, tw)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = fwd(x, ids, tw)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return out.clone(), statistics.median(times)


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    x, ids, tw, w13, w2 = gen_problem(device)

    cfg_off = MegaMoeForwardConfig(
        max_tokens_per_rank=TOKENS, hidden=HIDDEN, intermediate=INTERMEDIATE,
        num_total_experts=NUM_EXPERTS, num_topk=TOPK,
    )
    cfg_on = dataclasses.replace(
        cfg_off, impl=dataclasses.replace(cfg_off.impl, generate_c=True)
    )

    results = {}
    for name, cfg in (("generate_c=off", cfg_off), ("generate_c=on", cfg_on)):
        fwd = MegaMoeMxfp8Forward(cfg, rank=0, world_size=1)
        fwd.load_weights(w13, w2)
        out, ms = bench(fwd, x, ids, tw)
        stash_rows = 0
        if fwd.fc1_c is not None:
            stash_rows = int((fwd.fc1_c.abs().sum(dim=-1) > 0).sum().item())
        results[name] = (out, ms, stash_rows, fwd)
        print(f"{name:<16} median {ms:.3f} ms   fc1_c nonzero rows: {stash_rows}"
              + (f"/{fwd.fc1_c.shape[0]} (shape {tuple(fwd.fc1_c.shape)})"
                 if fwd.fc1_c is not None else ""))

    out_off, ms_off, _, f_off = results["generate_c=off"]
    out_on, ms_on, rows_on, f_on = results["generate_c=on"]
    d = (out_off.float() - out_on.float()).abs().max().item()
    print(f"output parity off-vs-on: max_abs_diff={d:.3e}")
    print(f"stash overhead: {ms_on - ms_off:+.3f} ms ({100*(ms_on/ms_off-1):+.1f}%)")
    ok = d < 1e-2 and rows_on > 0
    f_off.finalize()
    f_on.finalize()
    print("GENERATE_C SMOKE " + ("PASS" if ok else "INCONCLUSIVE"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
