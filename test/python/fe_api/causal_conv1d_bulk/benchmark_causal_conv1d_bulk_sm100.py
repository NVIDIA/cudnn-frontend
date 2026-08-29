# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interleaved benchmark for the bulk causal-convolution forward slice.

This script is intentionally not a pytest test.  It runs on every functionally
supported architecture; the operation's support check remains authoritative.

The ``fe_execute`` arm uses caller-preallocated output tensors, whereas FLA's
direct API allocates its outputs internally. CUDA events also include any GPU
idle gap while the host enqueues work between the two events. Therefore this
script reports direct-API event elapsed, not pure per-kernel active cycles;
obtain the latter from same-process CUPTI/Nsight kernel durations. The
wrapper/public arm is the more symmetric allocation-inclusive comparison.
"""

import argparse
import json
import os
import platform
import statistics

import torch
from cudnn._causal_conv1d_bulk_arch import is_functional_arch


def _validate_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    capability = (properties.major, properties.minor)
    if not is_functional_arch(capability):
        raise RuntimeError("Bulk causal conv1d does not support compute capability " f"{capability[0]}.{capability[1]} on {properties.name}")
    return properties, capability


def _slurm_metadata(environ=None):
    environ = os.environ if environ is None else environ
    fields = {
        "job_id": environ.get("SLURM_JOB_ID"),
        "node_name": environ.get("SLURMD_NODENAME"),
    }
    return {name: value for name, value in fields.items() if value}


def _event_us(fn, inner: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(inner):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / inner


def _interleaved_medians(functions, *, warmup: int, rounds: int, inner: int):
    names = list(functions)
    for _ in range(warmup):
        for name in names:
            functions[name]()
    torch.cuda.synchronize()

    samples = {name: [] for name in names}
    for round_index in range(rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            samples[name].append(_event_us(functions[name], inner))
    return {
        name: {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
            "samples_us": values,
        }
        for name, values in samples.items()
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--channels", type=int, default=8192)
    parser.add_argument("--packed-sequences", type=int, default=0)
    parser.add_argument("--state", action="store_true")
    parser.add_argument("--final-state", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--inner", type=int, default=5)
    args = parser.parse_args()

    properties, capability = _validate_environment()

    from cudnn.causal_conv1d_bulk_sm100 import (
        CausalConv1dBulkFwdSm100,
        causal_conv1d_bulk_fwd_wrapper_sm100,
    )
    from fla.modules.conv.causal_conv1d import causal_conv1d as fla_causal_conv1d
    from fla.modules.conv.triton.ops import causal_conv1d_fwd as fla_causal_conv1d_fwd
    from fla.ops.utils import prepare_chunk_indices

    torch.manual_seed(20260828)
    x = torch.randn(1, args.tokens, args.channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.channels, 4, device="cuda", dtype=torch.bfloat16)

    cu_seqlens = None
    chunk_indices = None
    num_sequences = 1
    if args.packed_sequences:
        if args.tokens % args.packed_sequences:
            raise ValueError("--tokens must be divisible by --packed-sequences")
        num_sequences = args.packed_sequences
        length = args.tokens // num_sequences
        cu_seqlens = torch.arange(
            0,
            args.tokens + 1,
            length,
            dtype=torch.int32,
            device="cuda",
        )
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    initial_state = None
    if args.state:
        initial_state = torch.randn(num_sequences, args.channels, 4, device="cuda", dtype=torch.bfloat16)

    output = torch.empty_like(x)
    final_state = torch.empty(num_sequences, args.channels, 4, device="cuda", dtype=torch.bfloat16) if args.final_state else None
    fe_api = CausalConv1dBulkFwdSm100(
        sample_x=x,
        sample_weight=weight,
        sample_output=output,
        sample_cu_seqlens=cu_seqlens,
        sample_initial_state=initial_state,
        sample_final_state=final_state,
    )
    fe_api.check_support()
    fe_api.compile()

    def fe_execute():
        return fe_api.execute(
            x,
            weight,
            output,
            cu_seqlens_tensor=cu_seqlens,
            initial_state_tensor=initial_state,
            final_state_tensor=final_state,
        )

    def fe_wrapper():
        return causal_conv1d_bulk_fwd_wrapper_sm100(
            x,
            weight,
            cu_seqlens_tensor=cu_seqlens,
            initial_state_tensor=initial_state,
            output_final_state=args.final_state,
        )

    def fla_direct():
        return fla_causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=None,
            residual=None,
            initial_state=initial_state,
            output_final_state=args.final_state,
            activation="silu",
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    def fla_public():
        return fla_causal_conv1d(
            x=x,
            weight=weight,
            initial_state=initial_state,
            output_final_state=args.final_state,
            activation="silu",
            backend="triton",
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    # Compile/autotune before checking or timing. This validates the exact
    # external implementation used by the benchmark but is not the independent
    # oracle used by the permanent correctness suite.
    fe_result = fe_wrapper()
    fla_result = fla_public()
    torch.cuda.synchronize()
    torch.testing.assert_close(fe_result["output_tensor"], fla_result[0], atol=3e-2, rtol=3e-2)
    if args.final_state:
        torch.testing.assert_close(fe_result["final_state_tensor"], fla_result[1], atol=0, rtol=0)

    timings = _interleaved_medians(
        {
            "fe_execute": fe_execute,
            "fla_direct": fla_direct,
            "fe_wrapper": fe_wrapper,
            "fla_public": fla_public,
        },
        warmup=args.warmup,
        rounds=args.rounds,
        inner=args.inner,
    )
    timings["ratios"] = {
        "fla_direct_over_fe_execute": timings["fla_direct"]["median_us"] / timings["fe_execute"]["median_us"],
        "fla_public_over_fe_wrapper": timings["fla_public"]["median_us"] / timings["fe_wrapper"]["median_us"],
    }
    timings["shape"] = {
        "tokens": args.tokens,
        "channels": args.channels,
        "packed_sequences": args.packed_sequences,
        "initial_state": args.state,
        "final_state": args.final_state,
    }
    timings["hardware"] = {
        "device": properties.name,
        "compute_capability": list(capability),
        "total_memory_bytes": properties.total_memory,
    }
    timings["software"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    slurm = _slurm_metadata()
    if slurm:
        timings["slurm"] = slurm
    print(json.dumps(timings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
