# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Direct-adapter timing for the MXFP8-QK / BF16-PV experiment.

This does not alter graph routing. It compares the direct-only hybrid kernel
with ordinary MXFP8 QKV and native BF16 using identical logical shapes. The
hybrid and ordinary MXFP8 paths share Q/K quantization; their V inputs differ
by design (BF16 versus columnwise-MXFP8).

Examples (GB200):
    python benchmark/attention_training/benchmark_mxfp8_qk_bf16_pv.py
    python benchmark/attention_training/benchmark_mxfp8_qk_bf16_pv.py --sweep

The full sweep is the Cartesian product B={1,2,4,8,16,32,64,128} and
Sq=Sk={1024,2048,4096,8192,16384}. It intentionally takes a long time:
every row is a 100-launch warmup followed by a 1,000-launch CUDA-event average.
"""

import argparse
import math
from collections.abc import Callable

import torch

SWEEP_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
SWEEP_SEQLENS = (1024, 2048, 4096, 8192, 16384)
HYBRID_NAME = "Hybrid QK MXFP8 / PV BF16 (no Amax)"
MXFP8_NAME = "QKV MXFP8"
BF16_NAME = "QKV BF16"


def _quantize_mxfp8(
    x: torch.Tensor, *, columnwise: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an MXFP8 tensor and its F8_128x4 SF tensor for direct SDPA."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    b, h, s, d = x.shape
    data_d, _dq_d, sf_d, data_s, _dq_s, sf_s = quantize_to_mxfp8(x.float(), b, h, s, d)
    data, sf = (data_s, sf_s) if columnwise else (data_d, sf_d)
    return data.reshape_as(x), sf


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated positive-integer list for sweep sharding."""
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a non-empty list of positive integers"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must not repeat")
    return values


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


def _time_variant(
    api_cls,
    *,
    shape_q: tuple[int, int, int, int],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    pv_bf16: bool,
    sf_q: torch.Tensor | None,
    sf_k: torch.Tensor | None,
    sf_v: torch.Tensor | None,
    warmup: int,
    iters: int,
    execution: str,
) -> dict[str, float]:
    """Compile and time one variant; retain only one output buffer at a time."""
    o = torch.empty(shape_q, device="cuda", dtype=torch.bfloat16)
    api = api_cls(
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

    if pv_bf16:
        launch = lambda: api.execute(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            o_tensor=o,
            sf_q=sf_q,
            sf_k=sf_k,
        )
    elif sf_q is not None:
        launch = lambda: api.execute(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            o_tensor=o,
            sf_q=sf_q,
            sf_k=sf_k,
            sf_v=sf_v,
        )
    else:
        launch = lambda: api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o)

    timings: dict[str, float] = {}
    if execution in ("eager", "both"):
        timings["eager"] = _time_cuda_events(launch, warmup=warmup, iters=iters)
    if execution in ("graph", "both"):
        try:
            graph_replay = _capture_cuda_graph(launch)
        except RuntimeError as exc:
            raise RuntimeError("variant is not CUDA-graph capturable") from exc
        timings["cuda graph"] = _time_cuda_events(
            graph_replay, warmup=warmup, iters=iters
        )

    # The launch closure owns the output buffer, so release it before moving to
    # the next variant. This is essential at the high-B/high-Sq sweep points.
    del launch, api, o
    torch.cuda.synchronize()
    return timings


def _benchmark_shape(
    api_cls,
    *,
    batch: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    dim: int,
    warmup: int,
    iters: int,
    execution: str,
) -> dict[str, dict[str, float]]:
    """Run the three-way comparison at one B, Sq=Sk point."""
    shape_q = (batch, q_heads, seqlen, dim)
    shape_kv = (batch, kv_heads, seqlen, dim)
    q_bf16 = torch.empty(shape_q, device="cuda", dtype=torch.bfloat16).normal_(
        0.0, 0.5
    )
    k_bf16 = torch.empty(shape_kv, device="cuda", dtype=torch.bfloat16).normal_(
        0.0, 0.5
    )
    v_bf16 = torch.empty(shape_kv, device="cuda", dtype=torch.bfloat16).normal_(
        0.0, 0.5
    )
    torch.cuda.empty_cache()
    q_mx, sf_q = _quantize_mxfp8(q_bf16, columnwise=False)
    torch.cuda.empty_cache()
    k_mx, sf_k = _quantize_mxfp8(k_bf16, columnwise=False)
    torch.cuda.empty_cache()
    v_mx, sf_v = _quantize_mxfp8(v_bf16, columnwise=True)
    torch.cuda.empty_cache()
    scale = 1.0 / math.sqrt(dim)

    return {
        HYBRID_NAME: _time_variant(
            api_cls,
            shape_q=shape_q,
            q=q_mx,
            k=k_mx,
            v=v_bf16,
            scale=scale,
            pv_bf16=True,
            sf_q=sf_q,
            sf_k=sf_k,
            sf_v=None,
            warmup=warmup,
            iters=iters,
            execution=execution,
        ),
        MXFP8_NAME: _time_variant(
            api_cls,
            shape_q=shape_q,
            q=q_mx,
            k=k_mx,
            v=v_mx,
            scale=scale,
            pv_bf16=False,
            sf_q=sf_q,
            sf_k=sf_k,
            sf_v=sf_v,
            warmup=warmup,
            iters=iters,
            execution=execution,
        ),
        BF16_NAME: _time_variant(
            api_cls,
            shape_q=shape_q,
            q=q_bf16,
            k=k_bf16,
            v=v_bf16,
            scale=scale,
            pv_bf16=False,
            sf_q=None,
            sf_k=None,
            sf_v=None,
            warmup=warmup,
            iters=iters,
            execution=execution,
        ),
    }


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
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="run the B x Sq Cartesian product selected by --batches and --seqlens",
    )
    parser.add_argument(
        "--batches",
        type=_parse_positive_ints,
        default=SWEEP_BATCHES,
        help="comma-separated B values for --sweep (default: 1,2,4,8,16,32,64,128)",
    )
    parser.add_argument(
        "--seqlens",
        type=_parse_positive_ints,
        default=SWEEP_SEQLENS,
        help="comma-separated Sq=Sk values for --sweep (default: 1024,2048,4096,8192,16384)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        raise RuntimeError("the MXFP8 experiment requires an SM100/SM103 GPU")
    if args.batch <= 0 or args.seqlen <= 0:
        raise ValueError("batch and seqlen must be positive")
    if args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")
    if args.dim != 128:
        raise ValueError("the current hybrid specialization is D128-only")

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    shapes = (
        tuple((batch, seqlen) for batch in args.batches for seqlen in args.seqlens)
        if args.sweep
        else ((args.batch, args.seqlen),)
    )
    torch.manual_seed(17)

    scope = (
        "full B x Sq sweep"
        if args.sweep
        and args.batches == SWEEP_BATCHES
        and args.seqlens == SWEEP_SEQLENS
        else "selected B x Sq sweep" if args.sweep else "single shape"
    )
    print(
        f"{scope}: Hq={args.q_heads} Hkv={args.kv_heads} D={args.dim}; causal; "
        f"{args.warmup} warmup launches; average of {args.iters} CUDA-event timed launches per row"
    )
    print(
        f"{'B':>4s} {'Sq=Sk':>7s} {'kernel':38s} {'execution':11s} "
        f"{'avg us/launch':>14s} {'vs BF16':>9s}"
    )

    for batch, seqlen in shapes:
        try:
            results = _benchmark_shape(
                SdpaFwdDslSm100,
                batch=batch,
                seqlen=seqlen,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                dim=args.dim,
                warmup=args.warmup,
                iters=args.iters,
                execution=args.execution,
            )
        except torch.OutOfMemoryError:
            print(f"{batch:4d} {seqlen:7d} {'OOM':38s} {'-':11s} {'-':>14s} {'-':>9s}")
        else:
            for name in (HYBRID_NAME, MXFP8_NAME, BF16_NAME):
                for execution in ("eager", "cuda graph"):
                    average_us = results[name].get(execution)
                    if average_us is None:
                        continue
                    bf16_us = results[BF16_NAME][execution]
                    relative = f"{average_us / bf16_us:.3f}x"
                    print(
                        f"{batch:4d} {seqlen:7d} {name:38s} {execution:11s} "
                        f"{average_us:14.2f} {relative:>9s}"
                    )
        finally:
            # Drop tensors, API objects, and graph pools before the next point.
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
