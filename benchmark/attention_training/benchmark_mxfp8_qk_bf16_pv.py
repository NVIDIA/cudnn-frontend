# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Direct-adapter timing for the MXFP8-QK / BF16-PV experiment.

This does not alter graph routing. It compares the direct-only hybrid kernel
with ordinary MXFP8 QKV and native BF16 using identical logical shapes. The
hybrid and ordinary MXFP8 paths share Q/K quantization; their V inputs differ
by design (BF16 versus columnwise-MXFP8). Both D128/D128 and the native MLA
D192/D128 flavor can be selected in one sweep.

Examples (GB200):
    python benchmark/attention_training/benchmark_mxfp8_qk_bf16_pv.py
    python benchmark/attention_training/benchmark_mxfp8_qk_bf16_pv.py --sweep
    python benchmark/attention_training/benchmark_mxfp8_qk_bf16_pv.py \
        --sweep --d-shapes d128,d192_d128 --amax-o

The full sweep is the Cartesian product B={1,2,4,8,16,32,64,128} and
Sq=Sk={1024,2048,4096,8192,16384}. It intentionally takes a long time:
every row is a 100-launch warmup followed by a 1,000-launch CUDA-event average.
"""

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path

import torch

SWEEP_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
SWEEP_SEQLENS = (1024, 2048, 4096, 8192, 16384)
HYBRID_NAME = "Hybrid QK MXFP8 / PV BF16"
MXFP8_NAME = "QKV MXFP8"
BF16_NAME = "QKV BF16"
D_SHAPES = {
    "d128": (128, 128),
    "d192_d128": (192, 128),
}
DEFAULT_D_SHAPES = (("d128", 128, 128),)


def _bshd_physical_input(shape: tuple[int, int, int, int]) -> torch.Tensor:
    """Allocate logical BHSD data on compact BSHD physical storage.

    The direct FROST API accepts logical BHSD tensors and derives its
    kernel-facing BSHD view with ``transpose(1, 2)``. Using this layout lets
    that view be contiguous, avoiding adapter-side Q/K/V gathers and O scatter.
    """
    batch, heads, seqlen, dim = shape
    return torch.empty(
        (batch, seqlen, heads, dim), device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)


def _as_bshd_physical(tensor: torch.Tensor) -> torch.Tensor:
    """Repack logical BHSD data so its derived BSHD view is contiguous."""
    return tensor.transpose(1, 2).contiguous().transpose(1, 2)


def _quantize_mxfp8(
    x: torch.Tensor, *, columnwise: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MXFP8 data/SF without materializing a monolithic FP32 Q tensor."""
    quantizer_dir = Path(__file__).parents[2] / "test" / "python" / "sdpa"
    if str(quantizer_dir) not in sys.path:
        sys.path.insert(0, str(quantizer_dir))
    from mxfp8_quant import quantize_to_mxfp8

    def quantize_chunk(x_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_b, h, s, d = x_chunk.shape
        data_d, _dq_d, sf_d, data_s, _dq_s, sf_s = quantize_to_mxfp8(
            x_chunk.float(), chunk_b, h, s, d, with_ref=False
        )
        data, sf = (data_s, sf_s) if columnwise else (data_d, sf_d)
        return data, sf

    batch = x.shape[0]
    chunk_size = 8
    if batch <= chunk_size:
        return quantize_chunk(x)

    data_chunks: list[torch.Tensor] = []
    sf_chunks: list[torch.Tensor] = []
    for x_chunk in x.split(chunk_size, dim=0):
        data, sf = quantize_chunk(x_chunk)
        data_chunks.append(data)
        sf_chunks.append(sf)
        torch.cuda.empty_cache()
    return torch.cat(data_chunks, dim=0), torch.cat(sf_chunks, dim=0)


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


def _parse_d_shapes(value: str) -> tuple[tuple[str, int, int], ...]:
    """Parse named Q/K--V head-dimension flavors for a comparative sweep."""
    labels = tuple(item.strip() for item in value.split(",") if item.strip())
    if not labels:
        raise argparse.ArgumentTypeError("expected at least one D-shape label")
    if len(labels) != len(set(labels)):
        raise argparse.ArgumentTypeError("D-shape labels must not repeat")
    unknown = tuple(label for label in labels if label not in D_SHAPES)
    if unknown:
        choices = ", ".join(D_SHAPES)
        raise argparse.ArgumentTypeError(
            f"unknown D-shape {unknown[0]!r}; choose from {choices}"
        )
    return tuple((label, *D_SHAPES[label]) for label in labels)


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
    shape_o: tuple[int, int, int, int],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    pv_bf16: bool,
    sf_q: torch.Tensor | None,
    sf_k: torch.Tensor | None,
    sf_v: torch.Tensor | None,
    emit_amax_o: bool,
    warmup: int,
    iters: int,
    execution: str,
) -> dict[str, float]:
    """Compile and time one variant; retain only one output buffer at a time."""
    o = _bshd_physical_input(shape_o)
    amax_o = (
        torch.empty(1, device="cuda", dtype=torch.float32) if emit_amax_o else None
    )
    api = api_cls(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_amax_o=amax_o,
        is_causal=True,
        scale_softmax=scale,
        dtype_o=torch.bfloat16,
        split_kv=1,
        pv_bf16=pv_bf16,
    )
    assert api.check_support()
    api.compile()

    execute_kwargs = dict(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o)
    if pv_bf16:
        execute_kwargs.update(sf_q=sf_q, sf_k=sf_k)
    elif sf_q is not None:
        execute_kwargs.update(sf_q=sf_q, sf_k=sf_k, sf_v=sf_v)
    if amax_o is not None:
        execute_kwargs["amax_o"] = amax_o

    def launch() -> None:
        api.execute(**execute_kwargs)

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

    del launch, execute_kwargs, api, amax_o, o
    torch.cuda.synchronize()
    return timings


def _benchmark_shape(
    api_cls,
    *,
    batch: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    d_qk: int,
    d_v: int,
    emit_amax_o: bool,
    warmup: int,
    iters: int,
    execution: str,
    variant: str,
) -> dict[str, dict[str, float]]:
    """Run direct-adapter variants at one B, Sq=Sk, Dqk/Dv point."""
    shape_q = (batch, q_heads, seqlen, d_qk)
    shape_k = (batch, kv_heads, seqlen, d_qk)
    shape_v = (batch, kv_heads, seqlen, d_v)
    shape_o = (batch, q_heads, seqlen, d_v)

    def bf16_input(shape: tuple[int, int, int, int]) -> torch.Tensor:
        return _bshd_physical_input(shape).normal_(0.0, 0.5)

    scale = 1.0 / math.sqrt(d_qk)
    results: dict[str, dict[str, float]] = {}
    run_mxfp8 = variant in ("all", "hybrid", "mxfp8")

    # Retain only the inputs used by the active variant. For high-memory MHA,
    # invoke each variant in a fresh process: CUDA-graph pools are process-
    # scoped and cannot be returned to the allocator between variants.
    if run_mxfp8:
        q_bf16 = bf16_input(shape_q)
        q_mx, sf_q = _quantize_mxfp8(q_bf16, columnwise=False)
        q_mx = _as_bshd_physical(q_mx)
        del q_bf16
        torch.cuda.empty_cache()

        k_bf16 = bf16_input(shape_k)
        k_mx, sf_k = _quantize_mxfp8(k_bf16, columnwise=False)
        k_mx = _as_bshd_physical(k_mx)
        del k_bf16
        torch.cuda.empty_cache()

        if variant in ("all", "hybrid"):
            v_bf16 = bf16_input(shape_v)
            results[HYBRID_NAME] = _time_variant(
                api_cls,
                shape_o=shape_o,
                q=q_mx,
                k=k_mx,
                v=v_bf16,
                scale=scale,
                pv_bf16=True,
                sf_q=sf_q,
                sf_k=sf_k,
                sf_v=None,
                emit_amax_o=emit_amax_o,
                warmup=warmup,
                iters=iters,
                execution=execution,
            )
            del v_bf16
            torch.cuda.empty_cache()

        if variant in ("all", "mxfp8"):
            v_bf16 = bf16_input(shape_v)
            v_mx, sf_v = _quantize_mxfp8(v_bf16, columnwise=True)
            v_mx = _as_bshd_physical(v_mx)
            del v_bf16
            torch.cuda.empty_cache()
            results[MXFP8_NAME] = _time_variant(
                api_cls,
                shape_o=shape_o,
                q=q_mx,
                k=k_mx,
                v=v_mx,
                scale=scale,
                pv_bf16=False,
                sf_q=sf_q,
                sf_k=sf_k,
                sf_v=sf_v,
                emit_amax_o=emit_amax_o,
                warmup=warmup,
                iters=iters,
                execution=execution,
            )
            del v_mx, sf_v
            torch.cuda.empty_cache()

        del q_mx, k_mx, sf_q, sf_k
        torch.cuda.empty_cache()

    if variant in ("all", "bf16"):
        q_bf16 = bf16_input(shape_q)
        k_bf16 = bf16_input(shape_k)
        v_bf16 = bf16_input(shape_v)
        results[BF16_NAME] = _time_variant(
            api_cls,
            shape_o=shape_o,
            q=q_bf16,
            k=k_bf16,
            v=v_bf16,
            scale=scale,
            pv_bf16=False,
            sf_q=None,
            sf_k=None,
            sf_v=None,
            emit_amax_o=False,
            warmup=warmup,
            iters=iters,
            execution=execution,
        )
        del q_bf16, k_bf16, v_bf16
        torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument(
        "--d-shapes",
        type=_parse_d_shapes,
        default=DEFAULT_D_SHAPES,
        help="comma-separated Q/K--V flavors: d128,d192_d128 (default: d128)",
    )
    parser.add_argument(
        "--amax-o",
        action="store_true",
        help="include the optional float32 Amax_O output for MXFP8 and hybrid",
    )
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
        "--variant",
        choices=("all", "hybrid", "mxfp8", "bf16"),
        default="all",
        help="time one variant in a fresh process, or all three (default)",
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
    d_labels = ",".join(label for label, _, _ in args.d_shapes)
    print(
        f"{scope}: Hq={args.q_heads} Hkv={args.kv_heads}; causal; "
        "logical BHSD / physical BSHD (zero-copy FROST adapter); "
        f"D-shapes={d_labels}; Amax_O={args.amax_o}; "
        f"{args.warmup} warmup launches; average of {args.iters} CUDA-event timed launches per row; "
        f"variant={args.variant}"
    )
    print(
        "{:>4s} {:>7s} {:>4s} {:>4s} {:38s} {:11s} {:>14s} {:>9s}".format(
            "B", "Sq=Sk", "Dqk", "Dv", "kernel", "execution", "avg us/launch", "vs BF16"
        )
    )

    for _label, d_qk, d_v in args.d_shapes:
        for batch, seqlen in shapes:
            try:
                results = _benchmark_shape(
                    SdpaFwdDslSm100,
                    batch=batch,
                    seqlen=seqlen,
                    q_heads=args.q_heads,
                    kv_heads=args.kv_heads,
                    d_qk=d_qk,
                    d_v=d_v,
                    emit_amax_o=args.amax_o,
                    warmup=args.warmup,
                    iters=args.iters,
                    execution=args.execution,
                    variant=args.variant,
                )
            except torch.OutOfMemoryError:
                print(
                    "{:4d} {:7d} {:4d} {:4d} {:38s} {:11s} {:>14s} {:>9s}".format(
                        batch, seqlen, d_qk, d_v, "OOM", "-", "-", "-"
                    )
                )
            else:
                bf16_timings = results.get(BF16_NAME, {})
                for name in (HYBRID_NAME, MXFP8_NAME, BF16_NAME):
                    for execution in ("eager", "cuda graph"):
                        average_us = results.get(name, {}).get(execution)
                        if average_us is None:
                            continue
                        bf16_us = bf16_timings.get(execution)
                        relative = (
                            f"{average_us / bf16_us:.3f}x"
                            if bf16_us is not None
                            else "-"
                        )
                        print(
                            f"{batch:4d} {seqlen:7d} {d_qk:4d} {d_v:4d} "
                            f"{name:38s} {execution:11s} {average_us:14.2f} {relative:>9s}"
                        )
            finally:
                # Drop tensors, API objects, and graph pools before the next point.
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
