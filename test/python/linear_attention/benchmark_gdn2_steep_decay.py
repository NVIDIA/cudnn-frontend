# SPDX-FileCopyrightText: Copyright (c) 2026 Alfred Wong
# SPDX-License-Identifier: Apache-2.0

"""Standalone synthetic GDN-2 steep-decay benchmark and FP64 check.

See benchmark_gdn2_steep_decay.md for public environment setup and ABBA usage.
The scheduler observer reads scratch only during an untimed diagnostic call.
GPU busy time is the union of profiler intervals, not end-to-end throughput.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import cudnn
from cudnn.linear_attention.ops.gdn2 import gated_delta_net_v2
from cudnn.linear_attention.frost import gdn2_engine

NAMES = ("q", "k", "v", "g", "b", "w")


def inputs(tokens, steep):
    generator = torch.Generator().manual_seed(7082)
    shape = (tokens, 1, 64)
    q = F.normalize(torch.randn(shape, generator=generator), dim=-1).bfloat16()
    k = F.normalize(torch.randn(shape, generator=generator), dim=-1).bfloat16()
    v = torch.randn(shape, generator=generator).bfloat16()
    g = torch.full(shape, -0.02)
    g[:, :, :steep] = -6.0
    b = torch.full(shape, 0.5, dtype=torch.bfloat16)
    w = torch.full(shape, 0.5, dtype=torch.bfloat16)
    do = torch.randn(shape, generator=generator).bfloat16()
    return [x.cuda().requires_grad_() for x in (q, k, v, g, b, w)], do.cuda()


def reference(xs, do):
    q, k, v, g, b, w = refs = [x.detach().cpu().double().requires_grad_() for x in xs]
    state = torch.zeros(1, 64, 64, dtype=torch.float64)
    outputs = []
    for t in range(q.shape[0]):
        decayed = state * g[t].exp().unsqueeze(-1)
        erase = torch.einsum("hk,hkv->hv", b[t] * k[t], decayed)
        state = decayed + k[t].unsqueeze(-1) * (w[t] * v[t] - erase).unsqueeze(-2)
        outputs.append(torch.einsum("hk,hkv->hv", q[t], state) * 64**-0.5)
    output = torch.stack(outputs)
    grads = torch.autograd.grad(output, refs, do.cpu().double())
    return [output.detach(), *(x.detach() for x in grads)]


@contextlib.contextmanager
def observe_schedulers(records):
    originals = {}
    for cls in (gdn2_engine.CompiledGdn2, gdn2_engine.CompiledGdn2Bwd):
        original = cls.run
        originals[cls] = original

        def observed(self, views, workspace, stream, _original=original):
            _original(self, views, workspace, stream)
            # Diagnostic only. Read while the caller still owns the workspace.
            regions = dict(zip(self.carve_names, workspace.carve(self.carve)))
            schedulers = {
                name: torch.from_dlpack(value).cpu().tolist() for name, value in regions.items() if name.startswith("scheduler_") and name != "scheduler_all"
            }
            records.append({"plan": type(self).__name__, "chain": self.chain, "schedulers": schedulers})

        cls.run = observed
    try:
        yield
    finally:
        for cls, original in originals.items():
            cls.run = original


def timed(fn, repeats=30, inner=1):
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    return {"median_ms": statistics.median(samples), "samples_ms": samples, "calls_per_sample": inner}


def run_case(tokens, steep, output_dir, label):
    xs, do = inputs(tokens, steep)
    offsets = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)

    def call():
        output, _ = gated_delta_net_v2(
            *xs,
            offsets,
            scale=64**-0.5,
            use_qk_l2norm_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=False,
            gate_domain="log",
            plan_name="gdn2_frost",
        )
        grads = torch.autograd.grad(output, xs, do)
        return [output, *grads]

    for _ in range(5):
        result = call()
    torch.cuda.synchronize()
    records = []
    with observe_schedulers(records):
        result = call()
    bwd = [r for r in records if r["plan"] == "CompiledGdn2Bwd"]
    assert bwd, records
    flag = bwd[-1]["schedulers"]["scheduler_bwd"][1]
    assert (flag > 1) if steep else (flag == 0), records
    ref = reference(xs, do)
    errors = {}
    snapshots = {}
    for name, actual, expected in zip(("output", *NAMES), result, ref):
        actual = actual.detach().cpu().double()
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        relative = ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()
        errors[name] = {"relative_frobenius": relative, "max_abs": (actual - expected).abs().max().item()}
        snapshots[name] = actual
    torch.save(snapshots, output_dir / f"{label}-t{tokens}-c{steep}.pt")
    eager = timed(call)
    busy_samples = []
    events = []
    for profile_round in range(5):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                call()
            torch.cuda.synchronize()
        gpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        assert gpu, "Profiler recorded no GPU activity"
        intervals = sorted((e.time_range.start, e.time_range.end) for e in gpu)
        begin, end = intervals[0]
        busy = 0.0
        for next_begin, next_end in intervals[1:]:
            if next_begin > end:
                busy += end - begin
                begin, end = next_begin, next_end
            else:
                end = max(end, next_end)
        busy += end - begin
        busy_samples.append(busy / 1000 / 10)
        if profile_round == 0:
            grouped = {}
            for event in gpu:
                entry = grouped.setdefault(event.name, {"count": 0, "total_us": 0.0})
                entry["count"] += 1
                entry["total_us"] += event.time_range.end - event.time_range.start
            events = grouped
    replay = {
        "median_ms": statistics.median(busy_samples),
        "samples_ms": busy_samples,
        "calls_per_sample": 10,
        "definition": "Union of GPU event intervals, excluding idle gaps, divided by calls",
    }
    row = {
        "tokens": tokens,
        "heads": 1,
        "dim": 64,
        "steep_channels": steep,
        "scheduler_records": records,
        "errors_fp64": errors,
        "correct_within_one_percent": all(e["relative_frobenius"] <= 0.01 for e in errors.values()),
        "eager": eager,
        "gpu_busy": replay,
        "gpu_events": events,
    }
    print(json.dumps({key: value for key, value in row.items() if key != "gpu_events"}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required; use a GPU supported by gdn2_frost")
    if Path(args.label).name != args.label or args.label in (".", ".."):
        parser.error("label must be a file-name component")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    helper = Path(cudnn.__file__).parent / "linear_attention/frost/common/steep_decay.py"
    record = {
        "label": args.label,
        "seed": 7082,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_file": cudnn.__file__,
        "cudnn_backend": cudnn.backend_version(),
        "dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": [],
    }
    for tokens in (16, 256):
        for steep in (0, 1, 64):
            record["results"].append(run_case(tokens, steep, args.output.parent, args.label))
            args.output.write_text(json.dumps(record, indent=2) + "\n")
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    assert all(r["correct_within_one_percent"] for r in record["results"]), "FP64 tolerance failed; inspect saved errors"


if __name__ == "__main__":
    main()
