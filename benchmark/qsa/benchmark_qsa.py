# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the public QSA API with an external, unchanged Miles source file."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

CASES = [
    (129, 3, 12, 1, 1),
    (511, 3, 24, 2, 1),
    (512, 1, 12, 1, 0),
    (4096, 3, 12, 1, 0),
    (4096, 3, 24, 2, 0),
    (16384, 3, 12, 1, 0),
    (16384, 3, 24, 2, 0),
    (32768, 1, 12, 1, 0),
    (32768, 1, 24, 2, 0),
    (131072, 1, 12, 1, 0),
    (131072, 1, 24, 2, 0),
]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    import torch

    digest = hashlib.sha256()
    raw = tensor.contiguous().view(torch.uint8).flatten()
    for chunk in raw.split(64 * 1024 * 1024):
        digest.update(memoryview(chunk.cpu().numpy()))
    return digest.hexdigest()


def load_implementation(args):
    if args.implementation == "native":
        import cudnn

        if args.frontend_source:
            package = Path(args.frontend_source).resolve() / "python" / "cudnn"
            cudnn.__path__.insert(0, str(package))
        from cudnn.ops import qsa_forward, qsa_backward
        import cudnn.QSA.api as api

        root = Path(api.__file__).parent
        hashes = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py")) if not p.name.startswith("._")}
        return qsa_forward, qsa_backward, {"loaded_cudnn_root": str(cudnn.__file__), "api_path": str(api.__file__), "source_sha256": hashes}

    spec = importlib.util.spec_from_file_location("qsa_external_miles", args.miles_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Context:
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

    def forward(q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size=4):
        ctx = Context()
        out = module._QSABlockSparseAttn.forward(ctx, q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size)
        return out, ctx.saved_tensors[-2], ctx.saved_tensors[-1]

    def backward(grad_out, q, k, v, out_fp32, lse, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size=4):
        ctx = SimpleNamespace(
            saved_tensors=(q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, out_fp32, lse),
            scale=scale,
            group=q.shape[1] // k.shape[1],
            block_size=block_size,
        )
        return module._QSABlockSparseAttn.backward(ctx, grad_out)[:3]

    return forward, backward, {"source_path": str(Path(args.miles_source).resolve()), "source_sha256": file_hash(args.miles_source)}


def measure(fn, warmup, repeats, before_each=None):
    import torch

    for _ in range(warmup):
        if before_each:
            before_each()
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    events, walls = [], []
    for _ in range(repeats):
        if before_each:
            before_each()
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - before) * 1000)
        events.append(begin.elapsed_time(end))
    peak = torch.cuda.max_memory_allocated()
    return {
        "synchronized_wall_median_ms": statistics.median(walls),
        "cuda_event_median_ms": statistics.median(events),
        "synchronized_wall_samples_ms": walls,
        "cuda_event_samples_ms": events,
        "allocated_before_bytes": allocated_before,
        "peak_allocated_bytes": peak,
        "peak_additional_bytes": peak - allocated_before,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def native_components(inputs, backward_inputs, args):
    import torch
    import cutlass
    from cuda.bindings import driver as cuda
    from cudnn.QSA import api, _metadata

    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    lo, hi, first, last = (inputs[name] for name in ("lo", "hi", "block_first", "block_last"))
    fprop, arch = api._implementation(q, k, "fprop")
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    compact_args = (inputs["sel_blocks"], lo, hi, first, last)
    ids, counts = api._compact(*compact_args, arch, stream)
    compact = _metadata.compile_kernel(arch)
    compiled_fwd = fprop.compile_kernel(cutlass.BFloat16, q.shape[1], k.shape[1], arch)
    out, fp32 = torch.empty_like(q), torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32)

    def timed(fn, before_each=None):
        return measure(fn, args.warmup, args.repeats, before_each)

    def host_timed(fn):
        samples = []
        for _ in range(args.repeats):
            before = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - before) * 1000)
        return {"host_wall_median_ms": statistics.median(samples), "host_wall_samples_ms": samples}

    result = {
        "boundary": "Diagnostic gather-only launches with warm caches and prebuilt IDs. Public forward may use dense tiles; these components do not describe that route.",
        "compact_with_allocation": timed(lambda: api._compact(*compact_args, arch, stream)),
        "compact_preallocated": timed(lambda: compact(*compact_args, ids, counts, stream)),
        "gather_fprop_preallocated": timed(
            lambda: fprop.launch(compiled_fwd, q, k, v, out, fp32, lse, ids, counts, lo, hi, first, last, inputs["scale"], stream)
        ),
        "warm_fprop_compile_lookup": host_timed(lambda: fprop.compile_kernel(cutlass.BFloat16, q.shape[1], k.shape[1], arch)),
        "warm_compact_compile_lookup": host_timed(lambda: _metadata.compile_kernel(arch)),
    }
    if args.directions != "both":
        return result
    bprop, _ = api._implementation(q, k, "bprop")
    compiled = bprop.compile_kernel(cutlass.BFloat16, q.shape[1], k.shape[1], arch)
    heads = (q.shape[1], k.shape[1])
    backward_ids, backward_counts = api._compact_backward(*compact_args, arch, stream, *heads)
    result["backward_compact_with_allocation"] = timed(lambda: api._compact_backward(*compact_args, arch, stream, *heads))
    if arch in (100, 103) and api.paired_backward(*heads):
        result["warm_backward_compact_compile_lookup"] = host_timed(lambda: _metadata.compile_group_kernel(arch, bprop.NQ, bprop.BLK_BITS))
    else:
        result["warm_backward_compact_compile_lookup"] = host_timed(lambda: _metadata.compile_kernel(arch))
    result["warm_bprop_compile_lookup"] = host_timed(lambda: bprop.compile_kernel(cutlass.BFloat16, q.shape[1], k.shape[1], arch))
    scratch = {
        "delta": torch.empty(q.shape[:2], dtype=torch.float32, device=q.device),
        "dq_acc": torch.empty_like(q, dtype=torch.float32),
        "dk_acc": torch.empty_like(k, dtype=torch.float32),
        "dv_acc": torch.empty_like(v, dtype=torch.float32),
    }
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    grad, saved_out, saved_lse = (backward_inputs[name] for name in ("grad_out", "out_fp32", "lse"))
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    blocks = max(1, min(sms * 8, (q.numel() // 8 + 255) // 256))

    def prep():
        compiled["prep"](saved_out, grad, scratch["delta"], scratch["dq_acc"], scratch["dk_acc"], scratch["dv_acc"], blocks, stream)

    def main():
        compiled["main"](
            q,
            k,
            v,
            grad,
            saved_lse,
            scratch["delta"],
            scratch["dq_acc"],
            scratch["dk_acc"],
            scratch["dv_acc"],
            backward_ids,
            backward_counts,
            lo,
            hi,
            first,
            last,
            inputs["scale"],
            min(backward_counts.shape[0], sms),
            stream,
        )

    result["bprop_prep"] = timed(prep)
    result["bprop_main"] = timed(main, before_each=prep)
    result["bprop_finish"] = timed(lambda: compiled["finish"](scratch["dq_acc"], scratch["dk_acc"], scratch["dv_acc"], dq, dk, dv, blocks, stream))
    result["bprop_preallocated_all_kernels"] = timed(
        lambda: bprop.launch(
            compiled, grad, q, k, v, saved_out, saved_lse, backward_ids, backward_counts, lo, hi, first, last, dq, dk, dv, scratch, inputs["scale"], stream, sms
        )
    )
    import reference

    expected = api.qsa_backward(**backward_inputs)
    result["backward_component_checks"] = reference.compare(dict(zip(("dq", "dk", "dv"), expected)), dict(dq=dq, dk=dk, dv=dv))
    if not all(check["passed"] for check in result["backward_component_checks"].values()):
        raise AssertionError("Preallocated backward components differ from the public call")
    return result


def environment():
    import torch

    properties = torch.cuda.get_device_properties(0)
    versions = {}
    for name in ("nvidia-cutlass-dsl", "triton", "nvidia-cudnn-frontend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,clocks.current.sm,clocks.current.memory,power.limit", "--format=csv,noheader"],
            text=True,
            timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        gpu_info = None
    return {
        "gpu": properties.name,
        "compute_capability": torch.cuda.get_device_capability(),
        "gpu_memory_bytes": properties.total_memory,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "python": sys.version,
        "packages": versions,
        "nvidia_smi": gpu_info,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "assigned_gpu_uuid": os.environ.get("QSA_GPU_UUID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def worker(args):
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    import torch
    import reference

    report = {"status": "running", "implementation": args.implementation, "case": json.loads(args.worker_case), "phase": "import"}
    save(args.output, report)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["environment"] = environment()
        report["reference_precision"] = {
            "dtype": "float32",
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"],
        }
        forward, backward, source = load_implementation(args)
        report["source"] = source
        report["harness_sha256"] = {p.name: file_hash(p) for p in (Path(__file__), Path(reference.__file__))}
        total, batch, hq, hkv, empty = report["case"]
        report["phase"] = "input_generation"
        save(args.output, report)
        before = time.perf_counter()
        inputs = reference.make_inputs(dict(T=total, B=batch, Hq=hq, Hkv=hkv, D=256, Empty=empty), "cuda", args.seed)
        torch.cuda.synchronize()
        report["input_generation_ms"] = (time.perf_counter() - before) * 1000
        report["seed"] = args.seed
        report["input_sha256"] = {name: tensor_hash(value) for name, value in inputs.items() if isinstance(value, torch.Tensor)}
        nb = inputs["sel_blocks"].shape[1]
        report["memory_model"] = {"bitmap_bytes": total * nb, "dense_int32_ids_bytes": total * nb * 4}
        report["miles_sel_max_linear_offset"] = (total - 1) * nb + nb - 1
        report["miles_sel_offset_exceeds_signed_int32"] = total * nb > 2**31
        report["phase"] = "reference_forward"
        save(args.output, report)
        expected = reference.forward_reference(**inputs)
        torch.cuda.synchronize()
        report["phase"] = "forward_correctness"
        save(args.output, report)
        before = time.perf_counter()
        actual = forward(**inputs)
        torch.cuda.synchronize()
        report["first_forward_with_compile_ms"] = (time.perf_counter() - before) * 1000
        checks = reference.compare(dict(zip(("out", "out_fp32", "lse"), expected)), dict(zip(("out", "out_fp32", "lse"), actual)))
        del actual
        backward_inputs = dict(grad_out=inputs["q"] * 0.125, out_fp32=expected[1], lse=expected[2], **inputs)
        report["saved_state_sha256"] = {name: tensor_hash(backward_inputs[name]) for name in ("grad_out", "out_fp32", "lse")}
        if args.directions == "both":
            report["phase"] = "reference_backward"
            save(args.output, report)
            expected_grad = reference.backward_reference(**backward_inputs)
            torch.cuda.synchronize()
            report["phase"] = "backward_correctness"
            save(args.output, report)
            before = time.perf_counter()
            actual_grad = backward(**backward_inputs)
            torch.cuda.synchronize()
            report["first_backward_with_compile_ms"] = (time.perf_counter() - before) * 1000
            checks.update(reference.compare(dict(zip(("dq", "dk", "dv"), expected_grad)), dict(zip(("dq", "dk", "dv"), actual_grad))))
            del expected_grad, actual_grad
        del expected
        report["checks"] = checks
        if not all(item["passed"] for item in checks.values()):
            raise AssertionError("QSA output check failed; timings withheld")
        report["phase"] = "timing"
        save(args.output, report)
        report["forward"] = measure(lambda: forward(**inputs), args.warmup, args.repeats)
        if args.directions == "both":
            report["backward"] = measure(lambda: backward(**backward_inputs), args.warmup, args.repeats)
        if args.components and args.implementation == "native":
            report["components"] = native_components(inputs, backward_inputs, args)
        report.update(status="passed", phase="complete")
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    save(args.output, report)
    print(json.dumps({"status": report["status"], "phase": report["phase"], "output": args.output}), flush=True)
    return int(report["status"] != "passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miles-source", required=True, help="Original block_sparse_attention.py; no edits are applied")
    parser.add_argument("--frontend-source", help="Optional repository root to test over an installed cudnn package")
    parser.add_argument("--output", default="qsa-results.json")
    parser.add_argument("--directions", choices=("both", "forward"), default="both")
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--components", action="store_true", help="Add native compaction and preallocated kernel timings")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=900, help="Seconds per implementation and case")
    parser.add_argument("--implementation", choices=("native", "miles"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1 or args.timeout < 1:
        parser.error("repeats, warmup and timeout must be positive")
    if args.worker_case:
        return worker(args)
    if not Path(args.miles_source).is_file():
        parser.error("--miles-source must name an existing file")
    output = Path(args.output).resolve()
    records = output.parent / (output.stem + "-cases")
    records.mkdir(parents=True, exist_ok=True)
    report = {
        "timing_contract": "Full public calls, including metadata, allocation, delta, casts and helper kernels. Warmup and compile excluded. No CUDA graph replay.",
        "directions": args.directions,
        "seed": args.seed,
        "cases": [],
        "correctness_contract": "All requested outputs checked before timing; both backward implementations use the same reference FP32 output and natural-log LSE.",
        "selector": "Synthetic uniform-score top512 complete blocks plus partial causal tail; balanced packed lengths; block_size4; BF16 D256.",
    }
    for case in CASES:
        if not args.min_tokens <= case[0] <= args.max_tokens:
            continue
        row = {"case": case, "implementations": {}}
        for implementation in ("native", "miles"):
            name = "-".join(map(str, case)) + "-" + implementation
            result_path, log_path = records / (name + ".json"), records / (name + ".log")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-case",
                json.dumps(case),
                "--implementation",
                implementation,
                "--miles-source",
                str(Path(args.miles_source).resolve()),
                "--output",
                str(result_path),
                "--directions",
                args.directions,
                "--seed",
                str(args.seed),
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
            ]
            if args.frontend_source:
                command.extend(("--frontend-source", str(Path(args.frontend_source).resolve())))
            if args.components:
                command.append("--components")
            result_path.unlink(missing_ok=True)
            with log_path.open("w") as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
                    exit_code = process.returncode
                except subprocess.TimeoutExpired:
                    exit_code = "timeout"
            try:
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
            except json.JSONDecodeError as error:
                result = {"status": "failed", "error": f"Invalid worker result JSON: {error}"}
            if exit_code != 0:
                result["status"] = "failed"
            result.update(exit_code=exit_code, log_path=str(log_path))
            row["implementations"][implementation] = result
            print(json.dumps({"case": case, "implementation": implementation, "status": result.get("status"), "phase": result.get("phase")}), flush=True)
        native, miles = row["implementations"]["native"], row["implementations"]["miles"]
        hashes_match = native.get("input_sha256") == miles.get("input_sha256") and native.get("input_sha256") is not None
        states_match = native.get("saved_state_sha256") == miles.get("saved_state_sha256") and native.get("saved_state_sha256") is not None
        row["identical_inputs"] = hashes_match
        row["identical_saved_state"] = states_match
        if native.get("status") == miles.get("status") == "passed" and hashes_match and states_match:
            row["speedup_synchronized_wall"] = {
                direction: miles[direction]["synchronized_wall_median_ms"] / native[direction]["synchronized_wall_median_ms"]
                for direction in ("forward", "backward")
                if direction in native
            }
        report["cases"].append(row)
        save(output, report)
    return int(
        not report["cases"]
        or any(not row["identical_inputs"] or not row["identical_saved_state"] for row in report["cases"])
        or any(x.get("status") != "passed" for row in report["cases"] for x in row["implementations"].values())
    )


if __name__ == "__main__":
    sys.exit(main())
