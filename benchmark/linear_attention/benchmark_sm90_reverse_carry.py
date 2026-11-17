# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the reusable SM90 backward prefix versus its G-only prefix.

Both candidates launch the same ``k_prep`` and ``k_scan_t`` kernels over the
same addresses.  The G-only candidate stops after ``d_initial_state`` is
written; the control additionally launches ``k_bwd`` for token gradients.
This isolates the largest safe saving available before designing a narrower
reverse-only preparation or scan.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from cudnn.linear_attention.hopper import cuda_bwd_host


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def measure(run, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        run()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=positive_int, default=65536)
    parser.add_argument("--heads", type=positive_int, default=32)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("warmups must be nonnegative")

    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(args.seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).bfloat16().to(device)

    q, k = [torch.nn.functional.normalize(randn(args.tokens, args.heads, 128).float(), dim=-1).bfloat16() for _ in range(2)]
    v = randn(args.tokens, args.heads, 128)
    d_o = randn(args.tokens, args.heads, 128) / (args.tokens * 16)
    g = -(0.01 + 0.04 * torch.rand(args.tokens, args.heads, 128, generator=generator)).to(device)
    beta = torch.sigmoid(torch.randn(args.tokens, args.heads, generator=generator)).to(device)
    cu = torch.tensor([0, args.tokens], dtype=torch.int32, device=device)
    state_shape = (1, args.heads, 128, 128)
    initial_state = torch.zeros(state_shape, dtype=torch.float32, device=device)
    d_final_state = torch.zeros_like(initial_state)
    d_initial_full = torch.empty_like(initial_state)
    d_initial_only = torch.empty_like(initial_state)
    dq, dk, dv = (torch.empty_like(q) for _ in range(3))
    dg = torch.empty_like(g)
    dbeta = torch.empty_like(beta)
    workspace_bytes = cuda_bwd_host.workspace_bytes(args.tokens, args.heads, 1)
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    stream = torch.cuda.current_stream().cuda_stream

    common = {
        "device": torch.cuda.current_device(),
        "stream": stream,
        "workspace": workspace.data_ptr(),
        "dO": d_o.data_ptr(),
        "q": q.data_ptr(),
        "k": k.data_ptr(),
        "v": v.data_ptr(),
        "g": g.data_ptr(),
        "beta": beta.data_ptr(),
        "cu_seqlens": cu.data_ptr(),
        "initial_state": initial_state.data_ptr(),
        "d_final_state": d_final_state.data_ptr(),
        "dq": dq.data_ptr(),
        "dk": dk.data_ptr(),
        "dv": dv.data_ptr(),
        "dg": dg.data_ptr(),
        "dbeta": dbeta.data_ptr(),
        "total_tokens": args.tokens,
        "n_heads": args.heads,
        "n_seqs": 1,
        "scale": 128**-0.5,
    }

    def full() -> None:
        cuda_bwd_host.launch(**common, d_initial_state=d_initial_full.data_ptr())

    def carry_only() -> None:
        cuda_bwd_host.launch(
            **common,
            d_initial_state=d_initial_only.data_ptr(),
            gradient_outputs=False,
        )

    full()
    carry_only()
    torch.cuda.synchronize()
    bitwise = torch.equal(d_initial_only.view(torch.uint8), d_initial_full.view(torch.uint8))
    if not bitwise:
        raise AssertionError("stopping before k_bwd changed d_initial_state")
    full_ms = measure(full, args.warmups, args.repeats)
    carry_ms = measure(carry_only, args.warmups, args.repeats)
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "heads": args.heads,
                "workspace_bytes": workspace_bytes,
                "full_ms": full_ms,
                "carry_only_ms": carry_ms,
                "speedup": full_ms / carry_ms,
                "bitwise": bitwise,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
