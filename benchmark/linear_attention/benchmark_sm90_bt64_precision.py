# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare SM90 BT64 backward checkouts on identical inputs.

Precision compares every gradient with recurrent FP64 autograd. Performance
measures the complete prepared backward (preparation, state scan and local
backward) with alternating CUDA-graph replays; it excludes forward and Python
plan construction. --baseline must point at an unchanged checkout.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch
import cudnn
from cudnn.linear_attention.hopper import cuda_bwd_host

NAMES = ("q", "k", "v", "g", "beta", "state")
OUTS = ("O", "S", "dQ", "dK", "dV", "dG", "dBeta", "dS0")


def inputs(lengths, heads, kind="random", seed=713):
    rng = torch.Generator().manual_seed(seed)
    t, d = sum(lengths), 128
    q = torch.nn.functional.normalize(torch.randn(t, heads, d, generator=rng), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(t, heads, d, generator=rng), dim=-1).bfloat16()
    v = torch.randn(t, heads, d, generator=rng).bfloat16()
    g = -torch.rand(t, heads, d, generator=rng) * 0.02
    beta = torch.rand(t, heads, generator=rng) * 0.5 + 0.25
    state = torch.randn(len(lengths), heads, d, d, generator=rng) * 0.1
    if kind == "weak":
        g.fill_(-1e-6)
    if kind == "steep":
        g.fill_(-5)
    if kind == "correlated":
        k.zero_()
        k[..., :2] = 0.70703125
        q.copy_(k)
        g.zero_()
        beta.fill_(0.5)
        state.zero_()
    x = dict(zip(NAMES, (q, k, v, g, beta, state)))
    x["dO"] = torch.randn(t, heads, d, generator=rng).bfloat16()
    x["dS"] = torch.randn(len(lengths), heads, d, d, generator=rng) * 0.1
    x["cu"] = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return {n: v.cuda() for n, v in x.items()}


def reference(x):
    leaves = [x[n].double().detach().requires_grad_() for n in NAMES]
    q, k, v, g, beta, states = leaves
    out, final = [], []
    cu = x["cu"].cpu().tolist()
    for n, (lo, hi) in enumerate(zip(cu, cu[1:])):
        s = states[n]
        for t in range(lo, hi):
            s = s * g[t].exp()[:, None, :]
            residual = v[t] - (s * k[t, :, None, :]).sum(-1)
            s = s + (beta[t, :, None] * residual)[:, :, None] * k[t, :, None, :]
            out.append((s * q[t, :, None, :]).sum(-1) / (128**0.5))
        final.append(s)
    o, fs = torch.stack(out), torch.stack(final)
    grads = torch.autograd.grad((o, fs), leaves, (x["dO"].double(), x["dS"].double()))
    return dict(zip(OUTS, (o.detach(), fs.detach(), *(a.detach() for a in grads))))


def prepare(x, module):
    names = ("dQ", "dK", "dV", "dG", "dBeta", "dS0")
    outputs = {n: torch.empty_like(x[k]) for n, k in zip(names, NAMES)}
    T, H = x["q"].shape[:2]
    N = x["state"].shape[0]
    ws = torch.empty(module.workspace_bytes(T, H, N), dtype=torch.uint8, device="cuda")

    def execute():
        module.launch(
            device=0,
            stream=torch.cuda.current_stream().cuda_stream,
            workspace=ws.data_ptr(),
            dO=x["dO"].data_ptr(),
            **{n: x[n].data_ptr() for n in ("q", "k", "v", "g", "beta")},
            cu_seqlens=x["cu"].data_ptr(),
            initial_state=x["state"].data_ptr(),
            d_final_state=x["dS"].data_ptr(),
            **{n: outputs[o].data_ptr() for n, o in zip(("dq", "dk", "dv", "dg", "dbeta", "d_initial_state"), names)},
            total_tokens=T,
            n_heads=H,
            n_seqs=N,
            scale=128**-0.5,
        )

    return outputs, ws, execute


def precision(modules):
    """Four seeds and thirteen layouts; retain all six gradient error metrics."""
    cases = [
        *(("correlated", [t]) for t in (16, 63, 64, 65, 128, 257)),
        ("random", [65]),
        ("weak", [257]),
        ("steep", [65]),
        ("random", [1024]),
        ("random", [0, 1, 15, 16, 17, 63, 64, 65]),
        ("dense_correlated", [257]),
        ("mixed_gate", [257]),
    ]
    rows = []
    for seed in (1, 17, 713, 2026):
        for kind, lengths in cases:
            x = inputs(lengths, 2, kind, seed)
            if kind == "dense_correlated":
                x["k"][:] = x["k"][0].clone()
                x["q"].copy_(x["k"])
                x["g"].zero_()
                x["beta"].fill_(0.5)
                x["state"].zero_()
            elif kind == "mixed_gate":
                x["g"][..., ::8] = -5
                x["g"][..., 1::8] = -1e-6
            ref = reference(x)
            results, errors = {}, {}
            for name, module in modules.items():
                got, workspace, execute = prepare(x, module)
                execute()
                saved = {key: value.clone() for key, value in got.items()}
                execute()
                errors[name] = {}
                for key, value in got.items():
                    assert torch.isfinite(value).all(), (name, kind, seed, key)
                    assert torch.equal(value.view(torch.uint8), saved[key].view(torch.uint8)), (name, kind, seed, key)
                    delta = value.double() - ref[key]
                    errors[name][key] = {
                        "relative_l2": (delta.norm() / ref[key].norm().clamp_min(1e-30)).item(),
                        "max_abs": delta.abs().max().item(),
                    }
                results[name] = saved
            unchanged = {
                key: torch.equal(results["baseline"][key].view(torch.uint8), results["candidate"][key].view(torch.uint8)) for key in results["baseline"]
            }
            row = dict(kind=kind, lengths=lengths, seed=seed, errors=errors, unchanged=unchanged)
            rows.append(row)
            print(kind, lengths, seed, {name: e["dG"]["relative_l2"] for name, e in errors.items()}, flush=True)
    return rows


def performance(modules):
    """Median of nine alternating groups of nine timed graph replays."""
    rows = []
    for lengths, heads in [
        ([0, 1, 15, 16, 17, 65], 2),
        ([1024], 8),
        ([16384], 8),
        ([0, 1, 65, 257, 4097], 8),
        ([131072], 8),
        ([131072], 32),
        ([256] * 20 + [62976] * 2, 8),
    ]:
        x = inputs(lengths, heads)
        owned, graphs, sizes = {}, {}, {}
        try:
            for name, module in modules.items():
                owned[name] = prepare(x, module)
                outputs, workspace, execute = owned[name]
                execute()
                sizes[name] = workspace.numel()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    execute()
                graphs[name] = graph
            for graph in graphs.values():
                for _ in range(9):
                    graph.replay()
            timings = {name: [] for name in modules}
            for group in range(9):
                for name in list(modules)[:: 1 if group % 2 == 0 else -1]:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(9):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end) / 9)
            row = dict(lengths=lengths, heads=heads, ms={n: statistics.median(t) for n, t in timings.items()}, samples_ms=timings, workspace_bytes=sizes)
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            torch.cuda.synchronize()
            for graph in graphs.values():
                graph.reset()
        del owned, graphs, outputs, workspace, execute, graph, x
    return rows


def smoke():
    """Candidate-only raw-launch replay with poisoned scratch, for sanitizers."""
    rows = []
    for lengths, heads in [([0, 1, 15, 16, 17, 63, 64, 65], 2), ([256], 8), ([257], 8), ([0, 0], 2)]:
        x = inputs(lengths, heads)
        outputs, workspace, execute = prepare(x, cuda_bwd_host)
        execute()
        expected = {key: value.clone() for key, value in outputs.items()}
        for _ in range(5):
            workspace.fill_(0xFF)
            for value in outputs.values():
                value.fill_(float("nan"))
            execute()
            for key, value in outputs.items():
                assert torch.isfinite(value).all(), (lengths, key)
                assert torch.equal(value.view(torch.uint8), expected[key].view(torch.uint8)), (lengths, key)
        rows.append(dict(lengths=lengths, heads=heads, replay_equal=True))
        print(lengths, heads, "all six gradients finite and byte-stable", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="required for precision and performance")
    parser.add_argument("--mode", choices=("precision", "performance", "smoke"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "smoke":
        report = dict(source=cudnn.__file__, rows=smoke())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    if args.baseline is None:
        parser.error("--baseline is required for precision and performance")
    path = args.baseline / "python/cudnn/linear_attention/hopper/cuda_bwd_host.py"
    spec = importlib.util.spec_from_file_location("cudnn.linear_attention.hopper._bt64_baseline", path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    modules = dict(baseline=baseline, candidate=cuda_bwd_host)
    hashes = {name: hashlib.sha256((module.KERNEL_DIR / module.KERNEL_BODY).read_bytes()).hexdigest() for name, module in modules.items()}
    if hashes["baseline"] == hashes["candidate"]:
        parser.error("baseline and candidate kernel sources are identical")
    report = dict(source=cudnn.__file__, torch=torch.__version__, cuda=torch.version.cuda, kernel_sha256=hashes)
    report["rows"] = precision(modules) if args.mode == "precision" else performance(modules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
