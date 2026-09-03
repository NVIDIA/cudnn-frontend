# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile exactly one packed HSTU qlen=1 execution.

This utility keeps compilation and warmup outside the CUDA profiler range so
Nsight Compute captures only the launches issued by one selected forward or
backward implementation.
"""

from __future__ import annotations

import argparse
import json

import torch
from benchmark_hstu_qlen1 import _compile_backward, _compile_forward, _make_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("forward", "backward"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--average-kv", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--mask", choices=("causal", "local", "full"), default="causal")
    parser.add_argument("--window-left", type=int, default=255)
    parser.add_argument("--window-right", type=int, default=0)
    parser.add_argument(
        "--forward-impl",
        choices=(
            "auto",
            "dispatch",
            "tc",
            "tc-split2",
            "tc-split4",
            "tc-m64",
            "tc-m64-split2",
            "tc-m64-warp1",
            "tc-m64-16dp",
            "tc-m64-16dp-split2",
            "tc-m64-16dp-split4",
            "tc-m64-16dp-tail",
            "tc-m64-16dp-tail-kv5",
            "tc-m64-16dp-tail-kv5-split2",
            "tc-m64-16dp-tail-kv5-split4",
        ),
        default="dispatch",
    )
    parser.add_argument(
        "--backward-impl",
        choices=(
            "auto",
            "dispatch",
            "direct",
            "direct-split2",
            "direct-split4",
            "direct-split8",
            "direct-split16",
            "direct-split22",
            "direct-split26",
            "direct-split32",
            "direct-split64",
            "direct-pair",
            "direct-pair-split2",
            "direct-pair-split4",
            "direct-pair-split8",
            "direct-pair-split13",
            "direct-pair-split16",
            "tc",
            "tc-small",
            "legacy",
        ),
        default="direct",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-launches", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("profile_hstu_qlen1.py requires a CUDA GPU")
    if args.batch_size <= 0 or args.profile_launches <= 0 or args.warmup < 0:
        raise ValueError("batch-size and profile-launches must be positive, and warmup must be non-negative")

    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    window_size = (-1, 0) if args.mask == "causal" else (args.window_left, args.window_right) if args.mask == "local" else (-1, -1)
    alpha = 0.7
    scaling_seqlen = 2048.0
    tensors = _make_inputs(
        args.batch_size,
        args.heads,
        args.head_dim,
        args.average_kv,
        dtype,
        device,
        seed=20260902,
        random_values=False,
    )
    k_lengths = tensors["k_lengths"]
    assert isinstance(k_lengths, list)

    if args.direction == "forward":
        _, run, compile_seconds = _compile_forward(
            tensors,
            max(k_lengths),
            window_size,
            alpha,
            scaling_seqlen,
            args.forward_impl,
        )
        implementation = args.forward_impl
    else:
        _, run, compile_seconds = _compile_backward(
            tensors,
            max(k_lengths),
            window_size,
            alpha,
            scaling_seqlen,
            args.backward_impl,
        )
        implementation = args.backward_impl

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    metadata = {
        "direction": args.direction,
        "implementation": implementation,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "batch_size": args.batch_size,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "mask": args.mask,
        "window_size": window_size,
        "total_kv": sum(k_lengths),
        "average_kv": sum(k_lengths) / args.batch_size,
        "min_kv": min(k_lengths),
        "max_kv": max(k_lengths),
        "compile_seconds": compile_seconds,
        "profile_launches": args.profile_launches,
    }
    print("PROFILE_READY " + json.dumps(metadata, sort_keys=True), flush=True)

    torch.cuda.nvtx.range_push("hstu_q1_profile")
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.profile_launches):
        run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.nvtx.range_pop()
    print("PROFILE_DONE", flush=True)


if __name__ == "__main__":
    main()
