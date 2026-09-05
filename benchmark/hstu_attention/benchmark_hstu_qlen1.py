# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the packed HSTU qlen=1, variable-KV workload.

The default cases model the target decode-like workload: H=4, D=128, every
sequence has one query, and per-sequence KV lengths vary around 2K tokens.
Both explicit APIs use preallocated public outputs. Backward timing covers all
GPU work issued by production dispatch or an explicitly selected qlen=1 direct
kernel variant.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import cudnn
from cudnn import HSTUBwdSm100, HSTUFwdSm100
from cudnn.hstu_attention import _interface

_KV_LENGTH_PATTERN = (1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072)
_Q1_FWD_TUNING_ALGORITHMS = (
    "tc",
    "tc-split2",
    "tc-split4",
    "tc-m64",
    "tc-m64-split2",
    "tc-m64-split4",
    "tc-m64-n64",
    "tc-m64-n64-split2",
    "tc-m64-n64-split4",
    "tc-m64-warp1",
    "tc-m64-warp1-split2",
    "tc-m64-warp1-split4",
    "tc-m64-warp2",
    "tc-m64-warp2-split2",
    "tc-m64-warp2-split4",
    "tc-m64-warp3",
    "tc-m64-warp3-split2",
    "tc-m64-warp3-split4",
    "tc-m64-inplace",
    "tc-m64-inplace-split2",
    "tc-m64-inplace-split4",
    "tc-m64-16dp",
    "tc-m64-16dp-split2",
    "tc-m64-16dp-split4",
    "tc-m64-16dp-tail",
    "tc-m64-16dp-tail-kv5",
    "tc-m64-16dp-tail-kv5-split2",
    "tc-m64-16dp-tail-kv5-split4",
    "tc-epi1",
    "tc-epi1-split2",
    "tc-epi1-split4",
)


def _q1_fwd_tuning_config(algorithm: str) -> _interface._Q1FwdKernelConfig:
    """Translate a benchmark-only candidate name into kernel knobs."""
    if algorithm not in _Q1_FWD_TUNING_ALGORITHMS:
        raise ValueError(f"Unsupported qlen=1 forward tuning algorithm: {algorithm}")
    split_kv = 2 if algorithm.endswith("-split2") else 4 if algorithm.endswith("-split4") else 1
    silu_warps = 1 if "tc-m64-warp1" in algorithm else 2 if "tc-m64-warp2" in algorithm else 3 if "tc-m64-warp3" in algorithm else 0
    return _interface._Q1FwdKernelConfig(
        block_m=64 if algorithm.startswith("tc-m64") else 128,
        block_n=64 if "m64-n64" in algorithm else 128,
        split_kv=split_kv,
        single_warp_epilogue=algorithm.startswith("tc-epi1"),
        m64_silu_warps=silu_warps,
        m64_inplace_silu="tc-m64-inplace" in algorithm or "tc-m64-16dp" in algorithm,
        m64_16dp_silu="tc-m64-16dp" in algorithm,
        m64_tail_branch="tc-m64-16dp-tail" in algorithm,
        m64_kv_stage=5 if "-kv5" in algorithm else 0,
    )


def _kv_lengths(batch_size: int, average_kv: int) -> list[int]:
    if average_kv <= 0:
        raise ValueError("average_kv must be positive")
    scaled = [max(1, round(length * average_kv / 2048)) for length in _KV_LENGTH_PATTERN]
    # Rotate each repetition so non-multiples of nine do not always inherit the
    # same short prefix of the symmetric pattern.
    lengths = []
    for batch_idx in range(batch_size):
        repetition = batch_idx // len(scaled)
        pattern_idx = (batch_idx + repetition * 4) % len(scaled)
        lengths.append(scaled[pattern_idx])
    return lengths


def _cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def _clamp_window_size(window_size: tuple[int, int], max_k: int) -> tuple[int, int]:
    left, right = window_size
    return (
        left if left < 0 else min(left, max_k),
        right if right < 0 else min(right, max_k),
    )


def _make_inputs(
    batch_size: int,
    heads: int,
    head_dim: int,
    average_kv: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    *,
    random_values: bool = True,
) -> dict[str, torch.Tensor | list[int]]:
    q_lengths = [1] * batch_size
    k_lengths = _kv_lengths(batch_size, average_kv)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def randn(tokens: int) -> torch.Tensor:
        if not random_values:
            # Performance sweeps can exceed billions of elements.  Generating
            # normal random values can then take much longer than all timed
            # kernels combined, while HSTU's control flow is value-independent.
            return torch.full(
                (tokens, heads, head_dim),
                0.125,
                dtype=dtype,
                device=device,
            )
        return (
            torch.randn(
                (tokens, heads, head_dim),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * 0.2
        )

    q = randn(batch_size)
    k = randn(sum(k_lengths))
    v = randn(sum(k_lengths))
    do = randn(batch_size)
    return {
        "q": q,
        "k": k,
        "v": v,
        "do": do,
        "cu_q": _cu_seqlens(q_lengths, device),
        "cu_k": _cu_seqlens(k_lengths, device),
        "k_lengths": k_lengths,
    }


def _reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_lengths: list[int],
    alpha: float,
    scaling_seqlen: float,
    window_size: tuple[int, int] = (-1, 0),
) -> torch.Tensor:
    outputs = []
    k_offset = 0
    for batch_idx, k_length in enumerate(k_lengths):
        q_i = q[batch_idx : batch_idx + 1]
        k_i = k[k_offset : k_offset + k_length]
        v_i = v[k_offset : k_offset + k_length]
        scores = alpha * torch.einsum("qhd,khd->hqk", q_i, k_i)
        weights = F.silu(scores)
        window_left = k_length if window_size[0] < 0 else min(window_size[0], k_length)
        window_right = k_length if window_size[1] < 0 else min(window_size[1], k_length)
        q_row = torch.arange(q_i.shape[0], device=q.device)[:, None] + k_length - q_i.shape[0]
        k_col = torch.arange(k_length, device=q.device)[None, :]
        keep = (k_col >= q_row - window_left) & (k_col <= q_row + window_right)
        weights = torch.where(keep.unsqueeze(0), weights, torch.zeros_like(weights))
        outputs.append(torch.einsum("hqk,khd->qhd", weights, v_i) / scaling_seqlen)
        k_offset += k_length
    return torch.cat(outputs, dim=0)


def _measure_ms(run: Callable[[], None], warmup: int, iterations: int, groups: int) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    samples = []
    for _ in range(groups):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _compile_forward(
    tensors: dict[str, torch.Tensor | list[int]],
    max_k: int,
    window_size: tuple[int, int],
    alpha: float,
    scaling_seqlen: float,
    forward_impl: str = "auto",
) -> tuple[torch.Tensor, Callable[[], None], float]:
    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    cu_q = tensors["cu_q"]
    cu_k = tensors["cu_k"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert isinstance(cu_q, torch.Tensor)
    assert isinstance(cu_k, torch.Tensor)
    window_size = _clamp_window_size(window_size, max_k)
    out = torch.empty_like(q)
    if forward_impl == "auto":
        api = HSTUFwdSm100(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_o=out,
            sample_cu_seqlens_q=cu_q,
            sample_cu_seqlens_k=cu_k,
            max_seqlen_q=1,
            max_seqlen_k=max_k,
            window_size=window_size,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
        )
        api.check_support()
        start = time.perf_counter()
        api.compile()
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - start

        def run() -> None:
            api.execute(q, k, v, out, cu_q, cu_k)

    else:
        internal_forward_impl = "auto" if forward_impl == "dispatch" else forward_impl
        tuning_config = None if internal_forward_impl == "auto" else _q1_fwd_tuning_config(internal_forward_impl)
        call_args = (
            q,
            k,
            v,
            cu_q,
            cu_k,
            1,
            max_k,
            window_size[0],
            window_size[1],
            alpha,
            None,
        )
        start = time.perf_counter()
        _interface.hstu_varlen_fwd_100(
            *call_args,
            scaling_seqlen=scaling_seqlen,
            out=out,
            _compile_only=True,
            _q1_fwd_tuning_config=tuning_config,
        )
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - start

        def run() -> None:
            _interface.hstu_varlen_fwd_100(
                *call_args,
                scaling_seqlen=scaling_seqlen,
                out=out,
                _q1_fwd_tuning_config=tuning_config,
            )

    return out, run, compile_seconds


def _compile_backward(
    tensors: dict[str, torch.Tensor | list[int]],
    max_k: int,
    window_size: tuple[int, int],
    alpha: float,
    scaling_seqlen: float,
    backward_impl: str = "auto",
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], Callable[[], None], float]:
    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    do = tensors["do"]
    cu_q = tensors["cu_q"]
    cu_k = tensors["cu_k"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert isinstance(do, torch.Tensor)
    assert isinstance(cu_q, torch.Tensor)
    assert isinstance(cu_k, torch.Tensor)
    window_size = _clamp_window_size(window_size, max_k)
    dq, dk, dv = (torch.empty_like(tensor) for tensor in (q, k, v))
    if backward_impl == "auto":
        api = HSTUBwdSm100(
            sample_do=do,
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_dq=dq,
            sample_dk=dk,
            sample_dv=dv,
            sample_cu_seqlens_q=cu_q,
            sample_cu_seqlens_k=cu_k,
            max_seqlen_q=1,
            max_seqlen_k=max_k,
            window_size=window_size,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
        )
        api.check_support()
        start = time.perf_counter()
        api.compile()
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - start

        def run() -> None:
            api.execute(do, q, k, v, dq, dk, dv, cu_q, cu_k)

    else:
        internal_backward_impl = "auto" if backward_impl == "dispatch" else backward_impl
        call_args = (
            do,
            q,
            k,
            v,
            cu_q,
            cu_k,
            1,
            max_k,
            dq,
            dk,
            dv,
            window_size[0],
            window_size[1],
            alpha,
            None,
            False,
            scaling_seqlen,
        )
        start = time.perf_counter()
        _interface.hstu_varlen_bwd_100(
            *call_args,
            _compile_only=True,
            _q1_bwd_algorithm=internal_backward_impl,
        )
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - start

        def run() -> None:
            _interface.hstu_varlen_bwd_100(
                *call_args,
                _q1_bwd_algorithm=internal_backward_impl,
            )

    return (dq, dk, dv), run, compile_seconds


def _correctness(
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    window_size: tuple[int, int],
    alpha: float,
    scaling_seqlen: float,
    forward_impl: str,
    backward_impl: str,
    direction: str,
) -> dict[str, float | bool]:
    tensors = _make_inputs(5, heads, head_dim, 2048, dtype, device, seed=20260901)
    tensors["k_lengths"] = [1, 127, 128, 2049, 3072]
    k_lengths = tensors["k_lengths"]
    assert isinstance(k_lengths, list)
    # Rebuild K/V and metadata for the boundary-heavy correctness lengths.
    generator = torch.Generator(device=device)
    generator.manual_seed(20260902)
    tensors["k"] = torch.randn((sum(k_lengths), heads, head_dim), dtype=dtype, device=device, generator=generator) * 0.2
    tensors["v"] = torch.randn((sum(k_lengths), heads, head_dim), dtype=dtype, device=device, generator=generator) * 0.2
    tensors["cu_k"] = _cu_seqlens(k_lengths, device)

    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    do = tensors["do"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert isinstance(do, torch.Tensor)

    # Compile and execute CuTe before the reference path. Early Rubin systems
    # do not yet support PyTorch's device-code reference toolchain, so the
    # correctness oracle intentionally runs on CPU.
    actual_out = None
    actual_grads = None
    if direction in ("forward", "both"):
        actual_out, fwd_run, _ = _compile_forward(tensors, max(k_lengths), window_size, alpha, scaling_seqlen, forward_impl)
        fwd_run()
    if direction in ("backward", "both"):
        actual_grads, bwd_run, _ = _compile_backward(
            tensors,
            max(k_lengths),
            window_size,
            alpha,
            scaling_seqlen,
            backward_impl,
        )
        bwd_run()
    torch.cuda.synchronize()

    q_ref = q.cpu().float().requires_grad_(True)
    k_ref = k.cpu().float().requires_grad_(True)
    v_ref = v.cpu().float().requires_grad_(True)
    expected_out = _reference_forward(q_ref, k_ref, v_ref, k_lengths, alpha, scaling_seqlen, window_size)
    expected_grads = torch.autograd.grad(expected_out, (q_ref, k_ref, v_ref), do.cpu().float())

    result: dict[str, float | bool] = {}
    if actual_out is not None:
        actual_out_cpu = actual_out.cpu().float()
        fwd_error = (actual_out_cpu - expected_out).abs()
        result["forward_ok"] = bool(torch.allclose(actual_out_cpu, expected_out, rtol=3.0e-2, atol=3.0e-2))
        result["forward_max_abs"] = float(fwd_error.max().item())
    if actual_grads is not None:
        actual_grads_cpu = [actual.cpu().float() for actual in actual_grads]
        grad_errors = [(actual - expected).abs() for actual, expected in zip(actual_grads_cpu, expected_grads)]
        result["backward_ok"] = bool(
            all(torch.allclose(actual, expected, rtol=6.0e-2, atol=6.0e-2) for actual, expected in zip(actual_grads_cpu, expected_grads))
        )
        result["dq_max_abs"] = float(grad_errors[0].max().item())
        result["dk_max_abs"] = float(grad_errors[1].max().item())
        result["dv_max_abs"] = float(grad_errors[2].max().item())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(64, 512, 1024))
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--average-kv", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--mask", choices=("causal", "local", "full"), default="causal")
    parser.add_argument("--window-left", type=int, default=255)
    parser.add_argument("--window-right", type=int, default=0)
    parser.add_argument("--direction", choices=("forward", "backward", "both"), default="both")
    parser.add_argument(
        "--forward-impl",
        choices=(
            "auto",
            "dispatch",
            *_Q1_FWD_TUNING_ALGORITHMS,
        ),
        default="auto",
    )
    parser.add_argument(
        "--backward-impl",
        choices=(
            "auto",
            "dispatch",
            *_interface._Q1_BWD_DIRECT_SPLITS,
        ),
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_hstu_qlen1.py requires a CUDA GPU")
    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    window_size = (-1, 0) if args.mask == "causal" else (args.window_left, args.window_right) if args.mask == "local" else (-1, -1)
    alpha = 0.7
    scaling_seqlen = 2048.0
    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cudnn_module": cudnn.__file__,
        "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
        "dtype": args.dtype,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "average_kv_target": args.average_kv,
        "mask": args.mask,
        "window_size": window_size,
        "direction": args.direction,
        "forward_impl": args.forward_impl,
        "backward_impl": args.backward_impl,
    }

    if not args.skip_correctness:
        correctness = _correctness(
            args.heads,
            args.head_dim,
            dtype,
            device,
            window_size,
            alpha,
            scaling_seqlen,
            args.forward_impl,
            args.backward_impl,
            args.direction,
        )
        print("CORRECTNESS " + json.dumps(correctness, sort_keys=True), flush=True)
        if not all(value for key, value in correctness.items() if key.endswith("_ok")):
            raise AssertionError("HSTU qlen=1 correctness check failed")
        results["correctness"] = correctness

    case_results = []
    for case_index, batch_size in enumerate(args.batch_sizes):
        tensors = _make_inputs(batch_size, args.heads, args.head_dim, args.average_kv, dtype, device, seed=1234 + case_index)
        k_lengths = tensors["k_lengths"]
        assert isinstance(k_lengths, list)
        case: dict[str, object] = {
            "batch_size": batch_size,
            "total_q": batch_size,
            "total_kv": sum(k_lengths),
            "average_kv": sum(k_lengths) / batch_size,
            "min_kv": min(k_lengths),
            "max_kv": max(k_lengths),
        }
        if args.direction in ("forward", "both"):
            _, run, compile_seconds = _compile_forward(tensors, max(k_lengths), window_size, alpha, scaling_seqlen, args.forward_impl)
            case["forward"] = {
                "compile_seconds": compile_seconds,
                **_measure_ms(run, args.warmup, args.iterations, args.groups),
            }
        if args.direction in ("backward", "both"):
            if args.backward_impl in ("auto", "dispatch"):
                q1_direct_supported = args.head_dim in (64, 128, 256) and args.mask in ("causal", "local")
                if q1_direct_supported:
                    q1_split_supported = dtype == torch.bfloat16
                    split_kv = _interface._select_q1_bwd_split_kv(
                        "auto",
                        torch.cuda.get_device_capability(device),
                        q1_split_supported,
                        batch_size=batch_size,
                        num_heads=args.heads,
                        total_kv=sum(k_lengths),
                        head_dim=args.head_dim,
                    )
                    selected_backward_impl = "direct-pair" if split_kv == 1 else f"direct-pair-split{split_kv}"
                else:
                    selected_backward_impl = "generic"
                case["selected_backward_impl"] = selected_backward_impl
            _, run, compile_seconds = _compile_backward(
                tensors,
                max(k_lengths),
                window_size,
                alpha,
                scaling_seqlen,
                args.backward_impl,
            )
            case["backward"] = {
                "compile_seconds": compile_seconds,
                **_measure_ms(run, args.warmup, args.iterations, args.groups),
            }
        print("CASE " + json.dumps(case, sort_keys=True), flush=True)
        case_results.append(case)
        del tensors
        torch.cuda.empty_cache()

    results["cases"] = case_results
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"WROTE {args.output}", flush=True)
    else:
        print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
