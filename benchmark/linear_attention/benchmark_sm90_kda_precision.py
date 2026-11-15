# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H20 A/B: public KDA accuracy and prepared-launch CUDA-graph latency.

--baseline points at an unchanged checkout. Both paths use identical inputs.
Precision runs select one host module before constructing any public plans.
Performance alternates implementations on the same tensors, with separate scratch.
--previous-fp32 optionally adds an earlier FP32-P checkout to performance runs
and requires all six gradients to match the current implementation bytewise.
"""

import argparse
import importlib.util
import json
import hashlib
import statistics
from pathlib import Path

import torch
import cudnn
import cudnn.linear_attention.hopper as hopper
from cudnn.linear_attention.hopper import cuda_bwd_host, cuda_host
from cudnn.linear_attention import kimi_delta_attention

NAMES = ("q", "k", "v", "g", "beta", "state")
OUTS = ("O", "S", "dQ", "dK", "dV", "dG", "dBeta", "dS0")


def inputs(lengths, heads, kind="random"):
    rng = torch.Generator().manual_seed(713)
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


def public(x):
    leaves = [x[n].detach().clone().requires_grad_() for n in NAMES]
    o, fs = kimi_delta_attention(*leaves[:5], x["cu"], initial_state=leaves[-1], output_final_state=True, batch_invariant=True, plan_name="kda_hopper_cuda")[:2]
    grads = torch.autograd.grad((o, fs), leaves, (x["dO"], x["dS"]))
    return dict(zip(OUTS, (o.detach(), fs.detach(), *(a.detach() for a in grads))))


def precision():
    rows = []
    cases = [("ragged", [0, 1, 15, 16, 17, 63, 64, 65]), ("weak", [257]), ("steep", [65]), ("correlated", [65]), ("long", [1024])]
    for name, lengths in cases:
        x = inputs(lengths, 2, name)
        ref, got = reference(x), public(x)
        again = public(x)
        errors = {}
        for n in OUTS:
            delta = got[n].double() - ref[n]
            assert torch.isfinite(got[n]).all(), n
            assert torch.equal(got[n].view(torch.uint8), again[n].view(torch.uint8)), n
            errors[n] = dict(
                relative_l2=(delta.norm() / ref[n].norm().clamp_min(1e-30)).item(), max_abs=delta.abs().max().item(), reference_l2=ref[n].norm().item()
            )
        rows.append(dict(case=name, lengths=lengths, heads=2, errors=errors))
        print(name, {n: round(errors[n]["relative_l2"], 6) for n in OUTS}, flush=True)
    return rows


def performance(modules):
    rows = []
    cases = [
        ("1k-h8", [1024], 8),
        ("16k-h8", [16384], 8),
        ("128k-h8", [131072], 8),
        ("128k-h32", [131072], 32),
        ("128k-thd22-h8", [256] * 20 + [62976] * 2, 8),
        ("ragged-h8", [0, 1, 65, 257, 4097], 8),
    ]
    for name, lengths, heads in cases:
        x = inputs(lengths, heads)
        outputs = {n: torch.empty_like(x[key]) for n, key in zip(OUTS, ("v", "state", *NAMES))}
        ptr = {n: v.data_ptr() for n, v in x.items()}
        common = dict(device=0, total_tokens=sum(lengths), n_heads=heads, n_seqs=len(lengths))

        def forward():
            cuda_host.launch(
                **common,
                stream=torch.cuda.current_stream().cuda_stream,
                **{n: ptr[n] for n in ("q", "k", "v", "g", "beta")},
                cu_seqlens=ptr["cu"],
                initial_state=ptr["state"],
                o=outputs["O"].data_ptr(),
                final_state=outputs["S"].data_ptr(),
            )

        graphs, scratch, memory = {}, {}, {}
        previous_gradients = None
        forward()
        for variant, module in modules.items():
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            scratch[variant] = torch.empty(module.workspace_bytes(sum(lengths), heads, len(lengths)), dtype=torch.uint8, device="cuda")
            memory[variant] = dict(
                workspace_bytes=scratch[variant].numel(),
                workspace_peak_allocated_bytes=torch.cuda.max_memory_allocated() - before,
                tensor_bytes=sum(v.numel() * v.element_size() for v in (*x.values(), *outputs.values())),
            )

            def backward(module=module, ws=scratch[variant]):
                module.launch(
                    **common,
                    stream=torch.cuda.current_stream().cuda_stream,
                    workspace=ws.data_ptr(),
                    dO=ptr["dO"],
                    **{n: ptr[n] for n in ("q", "k", "v", "g", "beta")},
                    cu_seqlens=ptr["cu"],
                    initial_state=ptr["state"],
                    d_final_state=ptr["dS"],
                    **{n: outputs[key].data_ptr() for n, key in zip(("dq", "dk", "dv", "dg", "dbeta", "d_initial_state"), OUTS[2:])},
                    scale=128**-0.5,
                )

            backward()
            if variant == "previous_fp32":
                previous_gradients = {n: outputs[n].clone() for n in OUTS[2:]}
            elif variant == "fp32" and previous_gradients is not None:
                for n in OUTS[2:]:
                    assert torch.equal(outputs[n].view(torch.uint8), previous_gradients[n].view(torch.uint8)), (name, n)
            for mode in ("backward", "forward_backward"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    if mode == "forward_backward":
                        forward()
                    backward()
                graphs[(variant, mode)] = graph
        kernel_us = {}
        if name == "16k-h8":
            from torch.profiler import profile, ProfilerActivity

            for variant in modules:
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):
                        graphs[(variant, "backward")].replay()
                    torch.cuda.synchronize()
                kernel_us[variant] = {e.key: e.self_device_time_total / 5 for e in prof.key_averages() if e.self_device_time_total > 0}
        timings = {key: [] for key in graphs}
        for graph in graphs.values():
            for _ in range(10):
                graph.replay()
        for group in range(9):
            keys = list(graphs)
            if group % 2:
                keys.reverse()
            for key in keys:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    graphs[key].replay()
                end.record()
                end.synchronize()
                timings[key].append(start.elapsed_time(end) / 20)
        row = dict(
            case=name,
            lengths=lengths,
            heads=heads,
            memory=memory,
            kernel_us=kernel_us,
            previous_fp32_byte_equal=previous_gradients is not None,
            timing={
                variant: {
                    mode: dict(median_ms=statistics.median(timings[(variant, mode)]), samples_ms=timings[(variant, mode)])
                    for mode in ("backward", "forward_backward")
                }
                for variant in modules
            },
        )
        rows.append(row)
        print(name, {v: {m: round(t["median_ms"], 4) for m, t in a.items()} for v, a in row["timing"].items()}, flush=True)
        for graph in graphs.values():
            graph.reset()
        del graphs, scratch, outputs, x, graph, backward, forward, previous_gradients
        torch.cuda.synchronize()
    return rows


def peak_memory():
    rows = []
    for name, lengths, heads in [("16k-h8", [16384], 8), ("128k-h8", [131072], 8), ("128k-h32", [131072], 32), ("128k-thd22-h8", [256] * 20 + [62976] * 2, 8)]:
        x = inputs(lengths, heads)
        for _ in range(3):
            warm = public(x)
            torch.cuda.synchronize()
            del warm
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        got = public(x)
        torch.cuda.synchronize()
        row = dict(
            case=name,
            inputs_resident_bytes=before,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_additional_bytes=torch.cuda.max_memory_allocated() - before,
            live_after_bytes=torch.cuda.memory_allocated(),
        )
        del got
        del x
        rows.append(row)
        print(name, row, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "fp32", "performance"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--previous-fp32", type=Path, help="previous FP32-P checkout; performance only")
    args = parser.parse_args()
    if args.memory and args.mode == "performance":
        parser.error("--memory requires --mode baseline or fp32")
    if args.previous_fp32 is not None and args.mode != "performance":
        parser.error("--previous-fp32 requires --mode performance")
    torch.set_num_threads(1)
    path = args.baseline / "python/cudnn/linear_attention/hopper/cuda_bwd_host.py"
    spec = importlib.util.spec_from_file_location("cudnn.linear_attention.hopper._baseline_bwd", path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    modules = dict(baseline=baseline)
    if args.previous_fp32 is not None:
        previous_path = args.previous_fp32 / "python/cudnn/linear_attention/hopper/cuda_bwd_host.py"
        previous_spec = importlib.util.spec_from_file_location("cudnn.linear_attention.hopper._previous_fp32_bwd", previous_path)
        previous = importlib.util.module_from_spec(previous_spec)
        previous_spec.loader.exec_module(previous)
        modules["previous_fp32"] = previous
    modules["fp32"] = cuda_bwd_host
    report = dict(source=cudnn.__file__, baseline_host=str(path), torch=torch.__version__, mode=args.mode)
    report["source_hashes"] = {
        variant: {
            part: hashlib.sha256(file.read_bytes()).hexdigest()
            for part, file in {"host": Path(module.__file__), "kernel": module.KERNEL_DIR / module.KERNEL_BODY}.items()
        }
        for variant, module in modules.items()
    }
    if args.mode == "performance":
        report["rows"] = performance(modules)
    else:
        hopper.cuda_bwd_host = baseline if args.mode == "baseline" else cuda_bwd_host
        report["rows"] = peak_memory() if args.memory else precision()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
