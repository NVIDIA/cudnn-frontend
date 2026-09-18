# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the connected BF16 Engram forward and complete backward on SM100.

Includes the native projection, gate, projection/input GEMM gradients, and
product-rule gradients for both normalization weights. Synthetic inputs use
the model's four streams, H5120 and embedding width6144. This floating training
surrogate does not measure lookup, FP8/QAT, collectives, or a whole model step.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import cudnn

from references import reference, source_reduction_consumer, full64_consumer, source_dot_consumer

PHASES = ("inference_forward", "training_forward", "both")
GRADIENTS = ("input", "embedding_contribution", "projection_weight", "q_norm_weight", "k_norm_weight")
REPETITIONS = 3
BLOCKS = 8
SAMPLES = 5


def metrics(actual, expected, limit, *, exact=False):
    a, b = actual.detach().float(), expected.detach().float()
    delta = a - b
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    relative = float(delta.norm() / b.norm().clamp_min(1e-20)) if finite else None
    scaled = float(delta.abs().max() / b.abs().max().clamp_min(1e-20)) if finite else None
    equal = torch.equal(actual, expected)
    finite = finite and math.isfinite(relative) and math.isfinite(scaled)
    relative, scaled = (relative, scaled) if finite else (None, None)
    passed = finite and relative <= limit and scaled <= 2 * limit and (not exact or equal)
    return dict(
        finite=finite, relative_l2=relative, max_scaled=scaled, relative_limit=limit, max_scaled_limit=2 * limit, exact=exact, equal=equal, passed=passed
    )


def route_summary(events, phase, role):
    kernels = [event for event in events if event.get("cat") == "kernel"]
    if not kernels or not all(event.get("dur", 0) > 0 for event in kernels):
        raise AssertionError("empty or invalid GPU trace")
    metadata = {}
    streams = {event.get("args", {}).get("stream") for event in kernels}
    if None in streams or (role == "frost" and len(streams) != 1):
        raise AssertionError("missing stream metadata or Frost launched on multiple streams")
    metadata["streams"] = sorted(streams)
    for key in ("graph id", "correlation"):
        values = {event.get("args", {}).get(key) for event in kernels}
        if len(values) != 1 or None in values or (key == "graph id" and next(iter(values)) <= 0):
            raise AssertionError("GPU kernels must belong to one captured graph launch: " + key)
        metadata[key] = next(iter(values))
    if role == "frost":
        names = ["engram_gate_saved_forward"]
        if phase == "both":
            names += ["engram_gate_saved_moments", "engram_gate_saved_apply", "engram_gate_weight_reduce"]
        for name in names:
            if sum(name in event["name"] for event in kernels) != REPETITIONS:
                raise AssertionError("incorrect captured kernel count: " + name)
    return dict(kernels=len(kernels), **metadata)


class FrostConsumer:
    def __init__(self, sources, mask, upstream):
        self.sources, self.mask, self.upstream = sources, mask, upstream
        x, embedding, weight, qw, _ = sources
        n, _, h = x.shape
        self.kv = torch.empty((n, 5 * h), device=x.device, dtype=x.dtype)
        self.product = torch.empty_like(qw)
        self.saved = torch.empty((n, 4, 4), device=x.device, dtype=torch.float32)
        self.out = torch.empty_like(x)
        self.dx = torch.empty_like(x)
        self.dkv = torch.empty_like(self.kv)
        self.dproduct = torch.empty_like(qw)
        self.dembedding = torch.empty_like(embedding)
        self.dweight = torch.empty_like(weight)
        self.dqw, self.dkw = torch.empty_like(qw), torch.empty_like(qw)
        self.forward = cudnn.EngramGateSavedForward(x, self.kv, self.product, mask, eps=1e-20, backend="frost")
        self.backward = cudnn.EngramGateSavedBackward(x, self.kv, self.product, self.saved, upstream, backend="frost")
        self.forward.compile()
        self.backward.compile()
        self.workspace = self.backward.allocate_workspace()

    def poison(self):
        with torch.no_grad():
            for name in ("kv", "product", "saved", "out", "dx", "dkv", "dproduct", "dembedding", "dweight", "dqw", "dkw"):
                getattr(self, name).fill_(float("nan"))
            self.workspace.fill_(255)

    def run(self, phase):
        x, embedding, weight, qw, kw = self.sources
        with torch.no_grad():
            torch.mm(embedding, weight.T, out=self.kv)
            torch.mul(qw, kw, out=self.product)
            self.forward.execute(x, self.kv, self.product, self.mask, self.out, self.saved)
            if phase == "both":
                self.backward.execute(x, self.kv, self.product, self.saved, self.upstream, self.dx, self.dkv, self.dproduct, self.workspace)
                torch.mul(self.dproduct, kw, out=self.dqw)
                torch.mul(self.dproduct, qw, out=self.dkw)
                torch.mm(self.dkv, weight, out=self.dembedding)
                torch.mm(self.dkv.T, embedding, out=self.dweight)
        return self.out, (self.dx, self.dembedding, self.dweight, self.dqw, self.dkw) if phase == "both" else None


def leaves_of(outputs):
    output, gradients = outputs
    return (output, *(gradients or ()))


def refresh(sources, mask, upstream, seed, generation):
    x, embedding, weight, qw, kw = sources
    torch.manual_seed(seed + generation * 10000)
    with torch.no_grad():
        x.normal_()
        embedding.normal_()
        weight.normal_().div_(math.sqrt(6144))
        qw.uniform_(0.75, 1.25)
        kw.uniform_(0.75, 1.25)
        upstream.normal_()
        mask.copy_(torch.rand(x.shape[0], device=x.device) > 0.13)
        if generation == 2:
            mask.zero_()
        if generation == 3:
            embedding.zero_()
        if generation == 4:
            x.zero_()
            mask.fill_(True)
        if generation == 5:
            x.mul_(1e-4)
            embedding.mul_(1e-4)
            mask.fill_(True)
        if generation == 6:
            mask.fill_(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; traces and failures use the same stem.")
    parser.add_argument("--tokens", type=int, choices=(4096, 8192), default=4096)
    parser.add_argument("--seed", type=int, default=419236)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status="running", timing=None, checks={}, rejected={}, traces={}, scope=__doc__, created_at=datetime.now(timezone.utc).isoformat())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def save():
        temporary = args.output.with_suffix(".writing.json")
        with temporary.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
        temporary.replace(args.output)

    def require(name, condition):
        result["checks"][name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    def compare(name, actual, expected, limit, phase, role, *, exact=False):
        value = metrics(actual, expected, limit, exact=exact)
        result["checks"][name] = value
        if not value["passed"]:
            key = phase + "." + role
            first = key not in result["rejected"]
            result["rejected"].setdefault(key, []).append(name)
            if first:
                failure = args.output.with_name(args.output.stem + "." + key + ".first_failure.pt")
                if failure.exists():
                    raise FileExistsError(failure)
                torch.save(dict(name=name, actual=actual.detach().cpu(), expected=expected.detach().cpu(), seed=args.seed, tokens=args.tokens), failure)
                result.setdefault("failure_artifacts", {})[key] = dict(path=str(failure), sha256=sha(failure))
            if role in ("frost", "torch_eager"):
                raise AssertionError(name + ": " + str(value))

    save()
    try:
        require("sm100", torch.cuda.get_device_capability() == (10, 0))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch._inductor.config.emulate_precision_casts = True
        torch.manual_seed(args.seed)
        n, h = args.tokens, 5120
        x = torch.randn(n, 4, h, device="cuda", dtype=torch.bfloat16)
        embedding = torch.randn(n, 6144, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(5 * h, 6144, device="cuda", dtype=torch.bfloat16) / math.sqrt(6144)
        qw, kw = [torch.rand(4, h, device="cuda", dtype=torch.float32) * 0.5 + 0.75 for _ in range(2)]
        mask = torch.ones(n, device="cuda", dtype=torch.bool)
        upstream = torch.randn_like(x)
        sources = (x, embedding, weight, qw, kw)
        leaves = tuple(t.detach().requires_grad_() for t in sources)
        tuned = dict(emulate_precision_casts=True, max_autotune=True, **{"triton.cudagraphs": False})
        native = {"torch_eager": reference}
        options = {}
        for name, function, config in (
            ("torch_compiled", reference, {"emulate_precision_casts": True}),
            ("torch_compiled_tuned", reference, tuned),
            ("torch_source_reductions", source_reduction_consumer, {"emulate_precision_casts": True}),
            ("torch_full64_dot", full64_consumer, {"emulate_precision_casts": True}),
            ("torch_source_dot", source_dot_consumer, {"emulate_precision_casts": True}),
            ("torch_source_dot_tuned", source_dot_consumer, tuned),
            ("torch_source_reductions_tuned", source_reduction_consumer, tuned),
        ):
            native[name] = torch.compile(function, fullgraph=True, options=config)
            options[name] = config
        frost = FrostConsumer(sources, mask, upstream)
        roles = [*native, "frost"]
        properties = torch.cuda.get_device_properties(0)
        result.update(
            seed=args.seed,
            tokens=n,
            hidden=h,
            streams=4,
            embedding_width=6144,
            dtype="bfloat16",
            norm_weight_dtype="float32",
            norm_epsilon=1e-20,
            gate_clamp=1e-6,
            roles=roles,
            native_options=options,
            phases=PHASES,
            generations=7,
            graph_repetitions=REPETITIONS,
            blocks=BLOCKS,
            samples=SAMPLES,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=str(properties),
            gpu_uuid=str(properties.uuid),
            cudnn_file=str(Path(cudnn.__file__).resolve()),
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            benchmark_sha256=sha(__file__),
            references_sha256=sha(Path(__file__).with_name("references.py")),
            saved_bytes=frost.saved.numel() * frost.saved.element_size(),
            scratch_bytes=frost.workspace.numel(),
        )

        def run(role, phase):
            if role == "frost":
                return frost.run(phase)
            output = native[role](*(sources if phase == "inference_forward" else leaves), mask)
            return output, torch.autograd.grad(output, leaves, upstream) if phase == "both" else None

        graphs, outputs = {}, {}
        for phase in PHASES:
            for role in roles:
                warm = torch.cuda.Stream()
                warm.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm):
                    for _ in range(REPETITIONS):
                        run(role, phase)
                warm.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=warm):
                    for _ in range(REPETITIONS):
                        captured = run(role, phase)
                graphs[phase, role], outputs[phase, role] = graph, captured
                print("Captured", phase, role, flush=True)

        previous = None
        for generation in range(7):
            refresh(sources, mask, upstream, args.seed, generation)
            oracle_leaves = tuple(t.detach().clone().requires_grad_() for t in sources)
            expected = reference(*oracle_leaves, mask)
            expected_grads = torch.autograd.grad(expected, oracle_leaves, upstream)
            expected_kv = torch.nn.functional.linear(embedding, weight)
            frozen = tuple(t.detach().clone() for t in (*sources, mask, upstream))
            if previous is not None:
                require(f"g{generation}.stale_output_negative", not metrics(previous, expected, 0.01)["passed"])
            require(f"g{generation}.nan_negative", not metrics(torch.full_like(expected, float("nan")), expected, 0.01)["passed"])
            require(f"g{generation}.zero_negative", not metrics(torch.zeros_like(expected), expected, 0.01)["passed"])
            for phase in PHASES:
                for role in roles:
                    for mode in ("eager", "graph"):
                        with torch.no_grad():
                            if role == "frost":
                                frost.poison()
                            if mode == "graph":
                                for t in leaves_of(outputs[phase, role]):
                                    t.fill_(float("nan"))
                        if mode == "eager":
                            output, gradients = run(role, phase)
                        else:
                            graphs[phase, role].replay()
                            output, gradients = outputs[phase, role]
                        tag = f"g{generation}.{phase}.{role}.{mode}"
                        compare(tag + ".output", output, expected, 0.01, phase, role)
                        if role == "frost":
                            compare(tag + ".projection", frost.kv, expected_kv, 0.006, phase, role)
                        if phase == "both":
                            for label, actual, target in zip(GRADIENTS, gradients, expected_grads, strict=True):
                                compare(tag + ".gradient." + label, actual, target, 0.03, phase, role)
                        if generation == 2:
                            compare(tag + ".masked_output", output, x, 0.01, phase, role, exact=True)
                            if phase == "both":
                                compare(tag + ".masked_dx", gradients[0], upstream, 0.03, phase, role, exact=True)
                                for label, actual in zip(GRADIENTS[1:], gradients[1:], strict=True):
                                    compare(tag + ".masked_zero." + label, actual, torch.zeros_like(actual), 0.03, phase, role, exact=True)
                        if generation == 3 and phase == "both":
                            for label, actual in zip(GRADIENTS[2:], gradients[2:], strict=True):
                                compare(tag + ".zero_embedding." + label, actual, torch.zeros_like(actual), 0.03, phase, role, exact=True)
            require(f"g{generation}.readonly", all(torch.equal(a, b) for a, b in zip((*sources, mask, upstream), frozen, strict=True)))
            previous = expected.detach().clone()
            del oracle_leaves, expected, expected_grads, expected_kv, frozen
            save()
            print("Validated generation", generation, flush=True)

        eligible = {phase: [role for role in roles if phase + "." + role not in result["rejected"]] for phase in PHASES}
        for phase in PHASES:
            require(phase + ".eligible_native_and_frost", "frost" in eligible[phase] and any(role in native for role in eligible[phase]))
        result["eligible"] = eligible
        save()

        # Validate that the timed graphs contain actual work, with every Frost
        # gate launch on one graph/stream and the expected number of replays.
        for phase in PHASES:
            for role in eligible[phase]:
                path = args.output.with_name(args.output.stem + "." + phase + "." + role + ".trace.json")
                if path.exists():
                    raise FileExistsError(path)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    graphs[phase, role].replay()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(path))
                events = json.loads(path.read_text())["traceEvents"]
                route = route_summary(events, phase, role)
                require(phase + "." + role + ".trace_work", True)
                result["traces"][phase + "." + role] = dict(path=str(path), sha256=sha(path), **route)
        for graph in graphs.values():
            for _ in range(20):
                graph.replay()
        torch.cuda.synchronize()

        timed_inputs = tuple(t.detach().clone() for t in (*sources, mask, upstream))
        timing = {}
        for phase in PHASES:
            timing[phase] = {role: {metric: [] for metric in ("graph_us", "eager_wall_us", "eager_gpu_us")} for role in eligible[phase]}
            for block in range(BLOCKS):
                order = eligible[phase] if block % 2 == 0 else list(reversed(eligible[phase]))
                for role in order:
                    graph_samples, wall_samples, gpu_samples = [], [], []
                    for _ in range(SAMPLES):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        graphs[phase, role].replay()
                        start.record()
                        graphs[phase, role].replay()
                        end.record()
                        end.synchronize()
                        graph_samples.append(start.elapsed_time(end) * 1000 / REPETITIONS)
                        torch.cuda.synchronize()
                        begin = time.perf_counter_ns()
                        start.record()
                        for _ in range(REPETITIONS):
                            run(role, phase)
                        end.record()
                        end.synchronize()
                        wall_samples.append((time.perf_counter_ns() - begin) / (1000 * REPETITIONS))
                        gpu_samples.append(start.elapsed_time(end) * 1000 / REPETITIONS)
                    for key, values in zip(("graph_us", "eager_wall_us", "eager_gpu_us"), (graph_samples, wall_samples, gpu_samples), strict=True):
                        require(f"{phase}.{role}.{block}.{key}.finite_positive", all(math.isfinite(v) and v > 0 for v in values))
                        timing[phase][role][key].append(values)
        # Recheck the measured inputs and outputs before admitting any timing.
        require("after.readonly", all(torch.equal(a, b) for a, b in zip((*sources, mask, upstream), timed_inputs, strict=True)))
        oracle = tuple(t.detach().clone().requires_grad_() for t in sources)
        expected = reference(*oracle, mask)
        expected_grads = torch.autograd.grad(expected, oracle, upstream)
        for phase in PHASES:
            for role in eligible[phase]:
                graphs[phase, role].replay()
                actual = leaves_of(outputs[phase, role])
                targets = (expected, *expected_grads) if phase == "both" else (expected,)
                for index, (a, b) in enumerate(zip(actual, targets, strict=True)):
                    require(f"after.{phase}.{role}.{index}", metrics(a, b, 0.01 if index == 0 else 0.03)["passed"])

        comparisons = {}
        for phase in PHASES:
            comparisons[phase] = {}
            for metric in ("graph_us", "eager_wall_us", "eager_gpu_us"):
                medians = {role: statistics.median(map(statistics.median, timing[phase][role][metric])) for role in eligible[phase]}
                baseline = min((role for role in eligible[phase] if role in native), key=medians.__getitem__)
                ratios = [
                    statistics.median(b) / statistics.median(f) for b, f in zip(timing[phase][baseline][metric], timing[phase]["frost"][metric], strict=True)
                ]
                speedup = statistics.median(ratios)
                comparisons[phase][metric] = dict(
                    baseline=baseline,
                    median_us=medians,
                    paired_speedup=speedup,
                    latency_reduction_percent=100 * (1 - 1 / speedup),
                    faster_blocks=sum(v > 1 for v in ratios),
                    paired_ratios=ratios,
                )
        imported = {}
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename and "/cudnn/engram/" in filename and Path(filename).is_file():
                imported[str(Path(filename).resolve())] = sha(filename)
        require("actual_engram_api_import", any(path.endswith("/engram/api.py") for path in imported))
        result.update(
            status="passed_with_recorded_native_rejections" if result["rejected"] else "passed",
            timing=timing,
            comparisons=comparisons,
            engram_imports=imported,
            independent_repeat_pending=True,
        )
        save()
        print(json.dumps(dict(status=result["status"], comparisons=comparisons, rejected=result["rejected"]), indent=2))
    except BaseException as exc:
        result.update(status="fail", timing=None, error=str(exc), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
