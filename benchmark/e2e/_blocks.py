# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-block timing: measure one block of each unique type, then weight-sum.

A hybrid linear-attention LM is a short list of distinct blocks repeated many times.
Timing one of each and multiplying by its count reproduces the assembled model to within
a fraction of a percent, and does so without ever holding the whole model:

    whole model, 3 GDN + 1 attn + head     334.52 ms   (Qwen3.5-27B dims, seq 16384, B200)
    weighted sum                           332.46 ms   -0.6%

That matters because the configurations worth measuring cannot all be assembled.
Qwen3.5-397B-A17B carries 12 GB of experts per MoE layer; four layers is 48 GB before
activations. It also removes an approximation: FLA's attention derives
``head_dim = hidden_size // num_heads``, so an assembled proxy cannot express the real
``(num_heads, head_dim=256)`` of six of the eight published Qwen3.5 architectures, while a
block instantiated directly can.

A block is timed directly, on a random residual-stream tensor, with a random gradient for the
backward. Nothing is subtracted, so nothing can be lost to the difference of two large numbers.
Whatever is not per-layer -- embedding, final norm, LM head, loss -- is one more thing to time
directly and weight once, at the real vocabulary rather than a scaled-down stand-in.

Do not reconstruct a block from whole-model step times by subtraction. Measuring Qwen3.5-0.8B,
whose linear-attention block is ~9 ms behind a ~98 ms LM head, ``t(2 layers) - t(1 layer)``
returned 0.45 ms: two nearly equal numbers, and the answer was noise. It reported linear
attention as 2.9% of the step rather than 42%, and made full attention look like the largest
component when it is not. Every block here is reachable on its own, so reach it.

What this does NOT model, deliberately:

* cross-layer allocator and L2 pressure. An isolated block keeps its weights hot; in a deep
  model they are long evicted. At 27B dimensions a layer is ~0.64 GB and DRAM-bound either
  way, but a small model (0.8B, ~50 MB/layer) is flattered.
* the optimizer, which is not a block.
* MoE routing imbalance. A synthetic uniform router produces grouped-GEMM shapes that do not
  occur in practice, and imbalance is the whole story for MoE throughput. Feed a measured
  routing distribution or say plainly that the number is a placeholder.

Forward and forward+backward are timed separately so that a training recipe with several
forward passes per backward -- GRPO runs three -- or activation recomputation is expressed
as a *weight* rather than as a different model.
"""

from __future__ import annotations

import collections

import torch


def time_step(step, *, warmup=3, iters=10, device=None):
    """Minimum wall time of ``step`` over ``iters``, after ``warmup``."""
    device = device or torch.cuda.current_device()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    best = float("inf")
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        torch.cuda.synchronize(device)
        best = min(best, s.elapsed_time(e))
    return best


def time_block(module, make_input, *, warmup=3, iters=10):
    """Forward and forward+backward time of one block, in milliseconds.

    ``make_input()`` returns the block's input -- a random residual-stream tensor is enough,
    because none of these kernels is data dependent in cost. The exception is a mixture-of-
    experts block, whose router decides how many tokens each expert receives: there the input
    distribution *is* the measurement, and a uniform random one is not representative.

    Timing a block directly beats reconstructing it from whole-model steps: there is no
    subtraction, so there is nothing for the difference of two large numbers to destroy.
    """
    x = make_input()
    params = [p for p in module.parameters() if p.requires_grad]

    def fwd():
        with torch.no_grad():
            module(x)

    out = module(x)
    grad = torch.randn_like(out) if torch.is_tensor(out) else torch.randn_like(out[0])

    def fwd_bwd():
        for p in params:
            p.grad = None
        if x.grad is not None:
            x.grad = None
        y = module(x)
        (y if torch.is_tensor(y) else y[0]).backward(grad)

    return time_step(fwd, warmup=warmup, iters=iters), time_step(fwd_bwd, warmup=warmup, iters=iters)


def compose(blocks, head_ms=0.0):
    """Weighted sum of measured blocks: ``blocks`` maps name -> (count, ms_each)."""
    total = head_ms + sum(count * ms for count, ms in blocks.values())
    return total, {name: count * ms for name, (count, ms) in blocks.items()}


def report(name, blocks, head_ms, *, reference_ms=None):
    """Print the composed step and each block's share of it."""
    total, contrib = compose(blocks, head_ms)
    print(f"\n{name}: composed step {total:.2f} ms")
    print(f"  {'block':22} {'count':>6} {'ms each':>9} {'ms total':>9} {'share':>7}")
    print("  " + "-" * 58)
    rows = sorted(contrib.items(), key=lambda kv: -kv[1])
    for block, ms_total in rows:
        count, ms_each = blocks[block]
        print(f"  {block:22} {count:6d} {ms_each:9.3f} {ms_total:9.2f} {100 * ms_total / total:6.1f}%")
    print(f"  {'head (not per-layer)':22} {1:6d} {head_ms:9.3f} {head_ms:9.2f} {100 * head_ms / total:6.1f}%")
    if reference_ms is not None:
        err = 100 * (total - reference_ms) / reference_ms
        print(f"\n  cross-check vs an assembled model: {reference_ms:.2f} ms, error {err:+.1f}%")
    return {"total_ms": total, "blocks": dict(blocks), "head_ms": head_ms, "contribution_ms": contrib}
