# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Direct-adapter timing for the MXFP8-QK / BF16-PV experiment.

This does not alter graph routing. It compares the direct-only hybrid kernel
with ordinary MXFP8 QKV and native BF16 using identical logical shapes. The
hybrid and ordinary MXFP8 paths share Q/K quantization; their V inputs differ
by design (BF16 versus columnwise-MXFP8).

Example (GB200):
    python benchmark/attention_training/benchmark_mxfp8_qk_bf16_pv.py
"""

import argparse
import math
from collections.abc import Callable

import torch


def _quantize_mxfp8(
    x: torch.Tensor, *, columnwise: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an MXFP8 tensor and its F8_128x4 SF tensor for direct SDPA."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    b, h, s, d = x.shape
    data_d, _dq_d, sf_d, data_s, _dq_s, sf_s = quantize_to_mxfp8(x.float(), b, h, s, d)
    data, sf = (data_s, sf_s) if columnwise else (data_d, sf_d)
    return data.reshape_as(x), sf


def _time_cuda_events(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    """Return CUDA-event average microseconds per launch after warmup."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iters <= 0:
        raise ValueError("iters must be positive")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _capture_cuda_graph(fn: Callable[[], None]) -> Callable[[], None]:
    """Capture one already-compiled direct API launch and return its replay."""
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph.replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="timed launches averaged for each kernel and execution mode",
    )
    parser.add_argument(
        "--execution", choices=("both", "eager", "graph"), default="both"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        raise RuntimeError("the MXFP8 experiment requires an SM100/SM103 GPU")
    if args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")
    if args.dim != 128:
        raise ValueError("the current hybrid specialization is D128-only")

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    torch.manual_seed(17)
    shape_q = (args.batch, args.q_heads, args.seqlen, args.dim)
    shape_kv = (args.batch, args.kv_heads, args.seqlen, args.dim)
    q_bf16 = (torch.randn(shape_q, device="cuda") * 0.5).to(torch.bfloat16)
    k_bf16 = (torch.randn(shape_kv, device="cuda") * 0.5).to(torch.bfloat16)
    v_bf16 = (torch.randn(shape_kv, device="cuda") * 0.5).to(torch.bfloat16)
    q_mx, sf_q = _quantize_mxfp8(q_bf16, columnwise=False)
    k_mx, sf_k = _quantize_mxfp8(k_bf16, columnwise=False)
    v_mx, sf_v = _quantize_mxfp8(v_bf16, columnwise=True)
    scale = 1.0 / math.sqrt(args.dim)

    def build(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, pv_bf16: bool):
        o = torch.empty(shape_q, device="cuda", dtype=torch.bfloat16)
        api = SdpaFwdDslSm100(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_o=o,
            is_causal=True,
            scale_softmax=scale,
            dtype_o=torch.bfloat16,
            split_kv=1,
            pv_bf16=pv_bf16,
        )
        assert api.check_support()
        api.compile()
        return api, o

    entries: list[tuple[str, Callable[[], None]]] = []
    hybrid, o_hybrid = build(q_mx, k_mx, v_bf16, pv_bf16=True)
    entries.append(
        (
            "Hybrid QK MXFP8 / PV BF16 (no Amax)",
            lambda: hybrid.execute(
                q_tensor=q_mx,
                k_tensor=k_mx,
                v_tensor=v_bf16,
                o_tensor=o_hybrid,
                sf_q=sf_q,
                sf_k=sf_k,
            ),
        )
    )
    baseline_mx, o_mx = build(q_mx, k_mx, v_mx, pv_bf16=False)
    entries.append(
        (
            "QKV MXFP8",
            lambda: baseline_mx.execute(
                q_tensor=q_mx,
                k_tensor=k_mx,
                v_tensor=v_mx,
                o_tensor=o_mx,
                sf_q=sf_q,
                sf_k=sf_k,
                sf_v=sf_v,
            ),
        )
    )
    baseline_bf16, o_bf16 = build(q_bf16, k_bf16, v_bf16, pv_bf16=False)
    entries.append(
        (
            "QKV BF16",
            lambda: baseline_bf16.execute(
                q_tensor=q_bf16, k_tensor=k_bf16, v_tensor=v_bf16, o_tensor=o_bf16
            ),
        )
    )

    print(
        f"shape: B={args.batch} Hq={args.q_heads} Hkv={args.kv_heads} S={args.seqlen} D={args.dim}; causal; "
        f"{args.warmup} warmup launches; average of {args.iters} CUDA-event timed launches per row"
    )
    results: dict[tuple[str, str], float] = {}
    for name, fn in entries:
        if args.execution in ("eager", "both"):
            results[(name, "eager")] = _time_cuda_events(
                fn, warmup=args.warmup, iters=args.iters
            )
        if args.execution in ("graph", "both"):
            try:
                graph_replay = _capture_cuda_graph(fn)
            except RuntimeError as exc:
                raise RuntimeError(f"{name} is not CUDA-graph capturable") from exc
            results[(name, "cuda graph")] = _time_cuda_events(
                graph_replay, warmup=args.warmup, iters=args.iters
            )

    print(f"{'kernel':38s} {'execution':11s} {'avg us/launch':>14s} {'vs BF16':>9s}")
    for name, _fn in entries:
        for execution in ("eager", "cuda graph"):
            average_us = results.get((name, execution))
            if average_us is None:
                continue
            bf16_us = results.get(("QKV BF16", execution))
            relative = "n/a" if bf16_us is None else f"{average_us / bf16_us:.3f}x"
            print(f"{name:38s} {execution:11s} {average_us:14.2f} {relative:>9s}")


if __name__ == "__main__":
    main()
