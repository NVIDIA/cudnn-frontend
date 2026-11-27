# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the batch-invariant RMSNorm forward against other RMSNorm implementations.

    python benchmark/norms/benchmark_batch_invariant_rmsnorm.py --hidden 4096 7168 --rows 1 64 1024 16384
    python benchmark/norms/benchmark_batch_invariant_rmsnorm.py --impls cudnn_bi flashinfer --csv rmsnorm.csv

Implementations:

- ``cudnn_bi``: the graph with the ``BATCH_INVARIANT`` numerical note (the reference of every ratio).
- ``cudnn_frost``: the same graph without the note (cuDNN Frontend's default norm engine).
- ``flashinfer``: ``flashinfer.norm.rmsnorm`` with ``enable_pdl=False``.

All implementations of one hidden size share the weight and the rows; each row count takes the
leading rows. Every call is captured ``--calls`` times into one CUDA graph, which is replayed
``--replays`` times per round; rounds visit the implementations in rotating order and the report
gives the median time per call, the median of the per-round ratios to ``cudnn_bi``, and the
coefficient of variation of each implementation across rounds. Lock the SM and memory clocks before
quoting results. ``cudnn_bi`` is checked for batch invariance: a sample of rows, run as a batch of their own, must
reproduce the bytes they got in the full batch.
"""

import argparse
import csv
import statistics
from functools import partial

import torch

import cudnn

_TORCH = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _graph(x, w, out, eps, note):
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    xt = graph.tensor_like(x)
    wt = graph.tensor_like(w.view(1, -1))
    et = graph.tensor_like(torch.full((1, 1), eps, dtype=torch.float32), name="epsilon")
    y, _ = graph.rmsnorm(norm_forward_phase=cudnn.norm_forward_phase.INFERENCE, input=xt, scale=wt, bias=None, epsilon=et)
    y.set_output(True).set_data_type(out.dtype)
    if note:
        graph.select_numeric_notes([cudnn.numerical_note.BATCH_INVARIANT])
    graph.build([cudnn.heur_mode.A])
    engine = graph.selected_engine.name if graph.selected_engine is not None else "backend"
    if note and engine != "batch_invariant_rmsnorm":
        raise RuntimeError(f"BATCH_INVARIANT selected {engine!r}")
    workspace = torch.empty(graph.get_workspace_size(), device=x.device, dtype=torch.uint8)
    bindings = {xt: x, wt: w.view(1, -1), et: torch.full((1, 1), eps, dtype=torch.float32), y: out}
    return engine, partial(graph.execute, bindings, workspace)


def _flashinfer(x, w, out, eps):
    import flashinfer

    return "flashinfer", partial(flashinfer.norm.rmsnorm, x, w, eps, out=out, enable_pdl=False)


def _capture(call, calls):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            call()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _time(graph, calls, replays):
    """Microseconds per call; the replays run on the current stream, which the events bracket."""
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1e3 / (calls * replays)


def _check_rows(x, w, out, eps):
    """Run a sample of rows as a batch of their own and compare the bytes with the full batch's."""
    rows = torch.linspace(0, x.shape[0] - 1, min(x.shape[0], 16)).long().unique().to(x.device)
    alone, alone_out = x[rows].contiguous(), torch.empty((len(rows), x.shape[1]), dtype=out.dtype, device=out.device)
    _graph(alone, w, alone_out, eps, note=True)[1]()
    torch.cuda.synchronize()
    if not torch.equal(alone_out.view(torch.uint8), out[rows].contiguous().view(torch.uint8)):
        raise RuntimeError("cudnn_bi is not batch invariant: rows run as their own batch differ from the full batch")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, nargs="+", default=[2048, 4096, 5120, 7168, 8192])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 64, 256, 1024, 4096, 16384])
    parser.add_argument("--dtype", choices=sorted(_TORCH), default="bf16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--impls", nargs="+", choices=["cudnn_bi", "cudnn_frost", "flashinfer"], default=["cudnn_bi", "cudnn_frost", "flashinfer"])
    parser.add_argument("--calls", type=int, default=20, help="calls per captured graph")
    parser.add_argument("--replays", type=int, default=20, help="graph replays per round")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--csv")
    args = parser.parse_args()

    makers = {
        "cudnn_bi": lambda *a: _graph(*a, note=True),
        "cudnn_frost": lambda *a: _graph(*a, note=False),
        "flashinfer": _flashinfer,
    }
    order = list(dict.fromkeys(args.impls))
    if order[0] != "cudnn_bi":
        order = ["cudnn_bi"] + [name for name in order if name != "cudnn_bi"]

    dtype = _TORCH[args.dtype]
    records = []
    header = f"{'hidden':>6} {'rows':>6} " + " ".join(f"{name:>22}" for name in order)
    print(f"time per call in us (ratio = time / cudnn_bi time; cv = across {args.rounds} rounds)\n{header}")
    for hidden in args.hidden:
        gen = torch.Generator(device="cuda").manual_seed(hidden)
        x_all = torch.randn((max(args.rows), hidden), device="cuda", generator=gen).mul_(3).to(dtype)
        w = torch.rand((hidden,), device="cuda", generator=gen).add_(0.5).to(dtype)
        for rows in sorted(args.rows):
            x = x_all[:rows]
            calls, engines = {}, {}
            for name in order:
                out = torch.empty_like(x)
                try:
                    engines[name], call = makers[name](x, w, out, args.eps)
                    call()
                    torch.cuda.synchronize()
                    # The graph replays into ``out`` and the plan's buffers: keep them alive with it.
                    calls[name] = (_capture(call, args.calls), call, out)
                except Exception as exc:  # noqa: BLE001 - report and keep measuring the others
                    print(f"  {name} unavailable at hidden={hidden} rows={rows}: {type(exc).__name__}: {exc}")
                    continue
                if name == "cudnn_bi":
                    _check_rows(x, w, out, args.eps)
            if not calls:
                print(f"{hidden:>6} {rows:>6} " + " ".join(f"{'n/a':>22}" for _ in order), flush=True)
                continue
            times = {name: [] for name in calls}
            for r in range(args.rounds):
                names = list(calls)
                names = names[r % len(names) :] + names[: r % len(names)]
                for name in names:
                    times[name].append(_time(calls[name][0], args.calls, args.replays))
            ref = times.get("cudnn_bi")
            cells = []
            for name in order:
                if name not in times:
                    cells.append(f"{'n/a':>22}")
                    continue
                t = times[name]
                med = statistics.median(t)
                cv = statistics.pstdev(t) / statistics.mean(t)
                ratio = statistics.median(a / b for a, b in zip(t, ref)) if ref else float("nan")
                cells.append(f"{med:8.2f} ({ratio:5.2f}x cv{cv:4.0%})")
                records.append(dict(hidden=hidden, rows=rows, dtype=args.dtype, impl=name, engine=engines[name], us=med, ratio_to_cudnn_bi=ratio, cv=cv))
            print(f"{hidden:>6} {rows:>6} " + " ".join(cells), flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["hidden", "rows", "dtype", "impl", "engine", "us", "ratio_to_cudnn_bi", "cv"])
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
