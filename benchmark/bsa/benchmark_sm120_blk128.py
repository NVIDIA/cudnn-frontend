# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the native SM120 blk128 BF16 block-sparse attention kernel."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cudnn import BSA


def _rounded_topk(num_blocks: int, density: float) -> int:
    """Round density to a nearest block count, clamped to [1, num_blocks]."""
    return max(1, min(num_blocks, math.floor(density * num_blocks + 0.5)))


def _coprime_step(num_blocks: int) -> int:
    """Choose a modular stride that visits every KV block before repeating."""
    step = min(997, num_blocks - 1)
    while step > 1 and math.gcd(step, num_blocks) != 1:
        step -= 1
    return max(1, step)


def _make_block_indices(
    heads: int,
    num_blocks: int,
    topk: int,
    pattern: str,
    device: torch.device,
) -> torch.Tensor:
    """Build unique per-head, per-query KV selections in [1, H, Q, topk]."""
    q_idx = torch.arange(num_blocks, device=device, dtype=torch.int64).view(1, -1, 1)
    h_idx = torch.arange(heads, device=device, dtype=torch.int64).view(-1, 1, 1)
    k_idx = torch.arange(topk, device=device, dtype=torch.int64).view(1, 1, -1)
    if pattern == "strided":
        indices = (q_idx * 131 + h_idx * 577 + k_idx * _coprime_step(num_blocks)) % num_blocks
    elif pattern == "local":
        indices = (q_idx + h_idx * 17 - topk // 2 + k_idx) % num_blocks
    else:
        raise ValueError(f"unsupported pattern: {pattern}")
    return indices.to(torch.int32).unsqueeze(0).contiguous()


def _time_cuda(fn: Callable[[], object], warmup: int, repeats: int) -> tuple[list[float], float]:
    """Return CUDA-event samples and mean wall time in ms after warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeats)]
    wall_start = time.perf_counter()
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3 / repeats
    return [start.elapsed_time(end) for start, end in events], wall_ms


def _validate_samples(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
) -> tuple[float, float]:
    """Check two boundary query/head pairs against FP32 sparse attention."""
    sample_pairs = ((0, 0), (q.shape[1] - 1, q.shape[2] - 1))
    within_block = torch.arange(block_size, device=q.device, dtype=torch.int64)
    max_out_error = 0.0
    max_lse_error = 0.0
    for head, row in sample_pairs:
        q_block = row // block_size
        selected = block_indices[0, head, q_block].to(torch.int64)
        token_indices = (selected[:, None] * block_size + within_block[None, :]).reshape(-1)
        scores = torch.mv(k[0, head, token_indices].float(), q[0, head, row].float()) * (q.shape[-1] ** -0.5)
        ref_lse = torch.logsumexp(scores, dim=0)
        ref_out = torch.mv(v[0, head, token_indices].float().transpose(0, 1), torch.softmax(scores, dim=0))
        out_error = (out[0, head, row].float() - ref_out).abs().max().item()
        lse_error = (lse[0, head, row] - ref_lse).abs().item()
        max_out_error = max(max_out_error, out_error)
        max_lse_error = max(max_lse_error, lse_error)
        torch.testing.assert_close(out[0, head, row].float(), ref_out, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(lse[0, head, row], ref_lse, rtol=2e-3, atol=2e-3)
    return max_out_error, max_lse_error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse workload, timing, and dense-relative acceptance options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=142720)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--densities", type=float, nargs="+", default=(0.15, 0.20))
    parser.add_argument("--patterns", choices=("strided", "local"), nargs="+", default=("strided", "local"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--required-speedup", type=float, default=4.5)
    parser.add_argument("--fail-below-target", action="store_true")
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> int:
    """Compare BF16 blk128 to cuDNN dense SDPA and optionally enforce a target."""
    args = _parse_args(argv)
    block_size = 128
    head_dim = 128
    target_density = 0.20
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this benchmark requires an SM120 GPU")
    if args.sequence < 1 or args.sequence % block_size:
        raise ValueError("sequence must be a positive multiple of 128")
    if args.heads < 1 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("heads/repeats must be positive and warmup must be nonnegative")
    if any(not 0.0 < density <= 1.0 for density in args.densities):
        raise ValueError("densities must be in (0, 1]")

    torch.manual_seed(20260901)
    device = torch.device("cuda")
    shape = (1, args.heads, args.sequence, head_dim)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)

    def dense_call() -> torch.Tensor:
        """Execute dense noncausal attention using the cuDNN SDPA backend."""
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    dense_times, dense_wall_ms = _time_cuda(dense_call, args.warmup, args.repeats)
    dense_ms = statistics.median(dense_times)
    print(f"DENSE median_ms={dense_ms:.4f} wall_ms={dense_wall_ms:.4f}", flush=True)

    num_blocks = args.sequence // block_size
    target_speedups = []
    for pattern in args.patterns:
        for requested_density in args.densities:
            topk = _rounded_topk(num_blocks, requested_density)
            actual_density = topk / num_blocks
            block_indices = _make_block_indices(args.heads, num_blocks, topk, pattern, device)

            def sparse_call():
                """Execute native blk128 attention with the current sparse mask."""
                return BSA.block_sparse_attention_forward(
                    q,
                    k,
                    v,
                    block_indices,
                    block_sparse_num=topk,
                    sparse_block_size=block_size,
                )

            sparse_times, sparse_wall_ms = _time_cuda(sparse_call, args.warmup, args.repeats)
            sparse_ms = statistics.median(sparse_times)
            speedup = dense_ms / sparse_ms
            conversion = speedup * actual_density
            result = sparse_call()
            torch.cuda.synchronize()
            max_out_error, max_lse_error = _validate_samples(
                q,
                k,
                v,
                result["o_tensor"],
                result["lse_tensor"],
                block_indices,
                block_size,
            )
            print(
                f"SPARSE pattern={pattern} requested_density={requested_density:.2f} "
                f"actual_density={actual_density:.9f} topk={topk}/{num_blocks} "
                f"median_ms={sparse_ms:.4f} wall_ms={sparse_wall_ms:.4f} "
                f"speedup={speedup:.4f}x conversion={conversion:.4%} "
                f"max_out_error={max_out_error:.6g} max_lse_error={max_lse_error:.6g}",
                flush=True,
            )
            if math.isclose(requested_density, target_density):
                target_speedups.append(speedup)

    if not target_speedups:
        raise ValueError("densities must include the 0.20 target")
    observed_min = min(target_speedups)
    passed = observed_min >= args.required_speedup
    print(
        f"TARGET density={target_density:.2f} required={args.required_speedup:.4f}x " f"observed_min={observed_min:.4f}x status={'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return 0 if passed or not args.fail_below_target else 1


if __name__ == "__main__":
    raise SystemExit(main())
