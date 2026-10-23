# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Same-input comparison against the customer's unchanged grouped.py."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relax-source", type=Path, required=True, help="Directory containing unchanged grouped.py and selector.py")
    parser.add_argument("--frontend-source", type=Path, help="Verify that cudnn imports from this checkout; never silently overrides an editable install")
    parser.add_argument("--lengths", default="8192")
    parser.add_argument("--heads", type=int, choices=(12, 24), default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--indexed", action="store_true")
    args = parser.parse_args()
    lengths = tuple(map(int, args.lengths.split(",")))
    if not lengths or min(lengths) < 1 or sum(lengths) > 131072 or args.repeats < 3 or args.warmup < 2:
        parser.error("Require positive packed lengths totaling at most 131072 and at least three samples and two warmups")
    if args.indexed and args.heads != 12:
        parser.error("The indexed path supports 12Q/1KV")
    import torch
    import cudnn
    import importlib.metadata

    if args.frontend_source and Path(cudnn.__file__).resolve() != (args.frontend_source / "python/cudnn/__init__.py").resolve():
        raise RuntimeError(f"Imported {cudnn.__file__}, outside --frontend-source; rebuild/install the intended checkout")
    grouped = load(args.relax_source / "grouped.py", "external_relax_grouped")
    selector = load(args.relax_source / "selector.py", "external_relax_selector")
    reference = load(Path(__file__).with_name("reference.py"), "qsa_independent_reference")
    from benchmark_qsa import tensor_hash, file_hash, save
    from cudnn.ops import qsa_forward, qsa_backward, qsa_indexed_forward, qsa_indexed_backward

    os.environ["RELAX_QWEN38_QSA_GROUPED_BACKWARD_BLOCK_K"] = "32"
    os.environ["RELAX_QWEN38_QSA_GROUPED_BACKWARD_STAGES"] = "3"
    os.environ["RELAX_QWEN38_QSA_GROUPED_BACKWARD_WARPS"] = "4"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(20261016)
    total, hq, hkv = sum(lengths), args.heads, args.heads // 12
    q = torch.randn((total, hq, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total, hkv, 256), device="cuda", dtype=torch.bfloat16)
    v, grad = torch.randn_like(k), torch.randn_like(q)
    iq = torch.randn((total, 4, 128), device="cuda", dtype=torch.bfloat16)
    selected = torch.full((total, 512), -1, device="cuda", dtype=torch.int32)
    first, last, origins, bases = [], [], [], []
    start, base = 0, 0
    for length in lengths:
        starts = list(range(start, start + length, 4))
        first.extend(starts)
        last.extend(min(i + 3, start + length - 1) for i in starts)
        origins.extend([start] * length)
        bases.extend([base] * length)
        keys = torch.randn((length // 4, 128), device="cuda", dtype=torch.bfloat16)
        for a in range(start, start + length, 512):
            b = min(a + 512, start + length)
            positions = torch.arange(a - start, b - start, device="cuda")
            logits = selector.index_scores(iq[a:b], keys).relu().sum(-1) / 128**0.5
            visible = torch.arange(keys.shape[0], device="cuda")[None] < (positions[:, None] + 1) // 4
            values, ids = logits.masked_fill(~visible, -torch.inf).topk(min(512, keys.shape[0]), dim=-1)
            selected[a:b, : ids.shape[1]] = ids.masked_fill(values == -torch.inf, -1).int()
        start += length
        base += len(starts)
    del iq, keys, logits, visible, values, ids
    origins, bases, first, last = [torch.tensor(x, dtype=torch.int32, device="cuda") for x in (origins, bases, first, last)]
    positions = torch.arange(total, device="cuda", dtype=torch.int32) - origins
    offsets = torch.arange(4, device="cuda", dtype=torch.int32)
    complete = origins[:, None, None] + selected[:, :, None] * 4 + offsets
    complete = complete.masked_fill(selected[:, :, None] < 0, -1).flatten(1)
    tail = origins[:, None] + ((positions[:, None] + 1) // 4) * 4 + offsets
    tail = tail.masked_fill(tail > (origins + positions)[:, None], -1)
    indices = torch.cat((complete, tail), dim=1).contiguous()
    del complete, tail
    # The extra sentinel column prevents invalid -1 slots from overwriting block zero.
    sel = torch.zeros((total, base + 1), dtype=torch.uint8, device="cuda")
    global_blocks = torch.where(selected >= 0, selected + bases[:, None], base).long()
    sel.scatter_(1, global_blocks, 1)
    partial = (positions + 1) % 4 != 0
    rows = torch.arange(total, device="cuda")[partial]
    sel[rows, (bases[partial] + positions[partial] // 4).long()] = 1
    sel = sel[:, :base].contiguous()
    del global_blocks
    inputs = dict(
        q=q,
        k=k,
        v=v,
        sel_blocks=sel,
        lo=origins,
        hi=origins + positions,
        blk_base=bases,
        tok_base=origins,
        block_first=first,
        block_last=last,
        scale=1 / 16,
        block_size=4,
    )

    class Context:
        needs_input_grad = (True, True, True, False, False, False, False, False)

        def save_for_backward(self, *values):
            self.saved_tensors = values

    def relax_forward():
        ctx = Context()
        out = grouped._Grouped.forward(ctx, q, k, v, indices, 1 / 16, True, 64, None)
        return out, ctx

    if args.indexed:

        def qsa_forward(**kwargs):
            return qsa_indexed_forward(q, k, v, indices, 1 / 16)

        def qsa_backward(**kwargs):
            return qsa_indexed_backward(kwargs["grad_out"], q, k, v, kwargs["out_fp32"], kwargs["lse"], indices, 1 / 16)

    def native_forward():
        return qsa_forward(**inputs)

    out_relax, ctx = relax_forward()
    native = native_forward()
    back_inputs = dict(inputs, grad_out=grad, out_fp32=native["out_fp32"], lse=native["lse"])
    native_grads = qsa_backward(**back_inputs)
    relax_grads = grouped._Grouped.backward(ctx, grad)[:3]
    report = dict(
        lengths=lengths,
        heads=[hq, hkv],
        gpu=torch.cuda.get_device_properties(0).name,
        sms=torch.cuda.get_device_properties(0).multi_processor_count,
        cc=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        source=cudnn.__file__,
        boundary="Warm full attention calls including allocation, metadata, gradient preparation and casts; selector, route expansion, compilation and TP communication excluded",
        reference_sha256={name: file_hash(args.relax_source / name) for name in ("grouped.py", "selector.py")},
        frontend_sha256={
            str(path.relative_to(Path(cudnn.__file__).parent)): file_hash(path)
            for folder in ("QSA",)
            for path in sorted((Path(cudnn.__file__).parent / folder).rglob("*.py"))
        },
        input_sha256={name: tensor_hash(value) for name, value in dict(q=q, k=k, v=v, grad_out=grad, indices=indices).items()},
        indexed=args.indexed,
        warmup=args.warmup,
        repeats=args.repeats,
        indices_shape=list(indices.shape),
        bitmap_bytes=sel.numel(),
        compact_forward_bytes=total * 513 * 4 if args.indexed else sel.numel() * 4,
        compact_backward_bytes=((total + 1) // 2) * 1026 * 4 if args.indexed else ((total + 1) // 2) * sel.shape[1] * 4 if hkv == 1 else sel.numel() * 4,
        checks={},
        timings={},
    )
    names = ("out", "out_fp32", "lse", "dq", "dk", "dv")
    actual_native = dict(zip(names, (*native, *native_grads)))
    actual_relax = dict(zip(names, (out_relax, ctx.saved_tensors[-2], ctx.saved_tensors[-1].T.contiguous(), *relax_grads)))
    report["checks"]["native_vs_relax"] = reference.compare(actual_relax, actual_native)
    expected_fwd = reference.forward_reference(**inputs)
    for arm, actual in (("native", actual_native), ("relax", actual_relax)):
        # Test the exact saved state consumed by each backward separately.
        expected_bwd = reference.backward_reference(**dict(inputs, grad_out=grad, out_fp32=actual["out_fp32"], lse=actual["lse"]))
        report["checks"][arm + "_vs_fp32"] = reference.compare(dict(zip(names, (*expected_fwd, *expected_bwd))), actual)
    report["passed"] = all(r["passed"] for group in report["checks"].values() for r in group.values())
    save(args.output, report)
    print("All pre-timing output checks:", report["passed"], flush=True)
    if not report["passed"]:
        raise RuntimeError("Correctness failed; performance is not reported")

    def measure(fn):
        for _ in range(args.warmup):
            result = fn()
            del result
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before_allocated = torch.cuda.memory_allocated()
        gpu, wall = [], []
        for _ in range(args.repeats):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tick = time.perf_counter()
            a.record()
            result = fn()
            b.record()
            b.synchronize()
            wall.append((time.perf_counter() - tick) * 1000)
            gpu.append(a.elapsed_time(b))
            del result
        return dict(
            gpu_ms=statistics.median(gpu),
            wall_ms=statistics.median(wall),
            gpu_samples_ms=gpu,
            wall_samples_ms=wall,
            peak_extra_allocated_bytes=torch.cuda.max_memory_allocated() - before_allocated,
        )

    # Reverse order across rounds exposes drift without composing separate stages.
    functions = dict(
        relax_fwd=relax_forward,
        native_fwd=native_forward,
        relax_bwd=lambda: grouped._Grouped.backward(ctx, grad)[:3],
        native_bwd=lambda: qsa_backward(**back_inputs),
    )
    for round_id, order in enumerate((list(functions), list(reversed(functions)))):
        for name in order:
            row = measure(functions[name])
            report["timings"][f"{name}_{round_id}"] = row
            save(args.output, report)
            print(name, round_id, json.dumps(row), flush=True)
    # Recheck after timed kernels have reused all scratch and accumulation paths.
    report["checks"]["native_after_timing"] = reference.compare(actual_native, dict(zip(names, (*native_forward(), *qsa_backward(**back_inputs)))))
    report["passed"] = all(r["passed"] for group in report["checks"].values() for r in group.values())
    save(args.output, report)
    if not report["passed"]:
        raise RuntimeError("Post-timing correctness failed")


if __name__ == "__main__":
    main()
