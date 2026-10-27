# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A/B kernel variants with shared addresses and strict backend data comparisons."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
import cudnn
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import tune
from backend import verify_loaded_library


def main():
    """Validate kernel variants against backend outputs, then measure paired device timings."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True, help="One retained timing JSON")
    p.add_argument("--direction", choices=["forward", "backward"])
    p.add_argument("--variant", action="append", required=True, help="label=kernel.py[:rows]")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--frontend", action="store_true", help="Include the existing BF16 NWH K4 SiLU kernel")
    p.add_argument(
        "--shared-output-timing", action="store_true", help="Time backend and candidates using the same output buffers, with a backend A/A allocation control"
    )
    p.add_argument("--accumulate", action="store_true", help="Add to existing parameter gradients; exclude clearing from both timed paths")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "test/python/causal_conv1d/cutedsl"))
    from se_bitwise import bitwise_difference, parameter_difference, ATOMIC_OUTPUTS

    a = json.loads(args.case.read_text())["args"]
    requested_backend = Path(a["backend_library"]).resolve(strict=True)
    B, D, L = map(int, a["shape"].split(","))
    b2b, nwh = "projection_width" in a, a.get("layout") == "nwc"
    backward = args.direction == "backward" if args.direction else "rows" in a
    if args.accumulate and (not backward or args.frontend):
        raise ValueError("Additive timing requires backward without the overwrite-only frontend comparison")
    K, Km = (a["projection_width"], a["mixer_width"]) if b2b else (a["width"], None)
    silu = not a.get("identity", True)
    dtype = getattr(torch, a["dtype"])
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    generator = torch.Generator().manual_seed(args.seed)

    def rand(shape):
        """Generate seeded inputs on CPU and transfer them to the selected CUDA dtype."""
        return (torch.randn(shape, generator=generator) * (0.25 if b2b else 1.0)).to(device="cuda", dtype=dtype)

    x = rand((B, 3 * D, L) if b2b else ((B, L, D) if nwh else (B, D, L)))
    w = rand((3 * D, K) if b2b else ((K, D) if nwh else (D, K)))
    bias, wm = rand((D,)), rand((D, Km)) if b2b else None
    yshape = (B, D, L) if b2b else x.shape
    dy = rand(yshape)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    typecode = {torch.float32: 0, torch.float64: 1, torch.float16: 2, torch.bfloat16: 9}[dtype]
    saved = torch.empty(yshape, device="cuda", dtype=dtype) if b2b and backward else None
    if saved is not None:
        auxiliary = torch.empty_like(saved)
        cudnn.b2b_causal_conv1d_forward(int(stream), *(t.data_ptr() for t in (x, w, wm, bias, saved, auxiliary)), B, D, L, K, Km, typecode)
    if backward:
        expected = {"dx": torch.empty_like(x)}
        if b2b:
            expected.update(dwp=torch.empty_like(w, dtype=compute), dwm=torch.empty_like(wm, dtype=compute), dskip=torch.empty_like(bias, dtype=compute))
        else:
            expected.update(dw=torch.empty_like(w, dtype=compute), db=torch.empty_like(bias, dtype=compute))
    else:
        expected = {"y": torch.empty(yshape, device="cuda", dtype=dtype)}
        if b2b:
            expected["yg"] = torch.empty_like(expected["y"])
    actual = {k: torch.empty_like(v) for k, v in expected.items()}
    if args.accumulate:
        for name, value in expected.items():
            if name in ATOMIC_OUTPUTS:
                value.zero_()
    dz = torch.empty_like(x, dtype=compute) if backward and not b2b else None
    inputs = [x, w, wm, bias] + ([saved, dy] if backward else []) if b2b else [x, w, bias] + ([dy] if backward else [])
    tensors = inputs + ([dz] if backward and not b2b else []) + list(actual.values())
    binding = ("b2b_causal_conv1d_" if b2b else ("causal_conv1d_nwh_" if nwh else "causal_conv1d_")) + ("backward" if backward else "forward")
    dimensions = [B, D, L, K, Km, typecode] if b2b else [B, D, L, K, typecode]
    if backward:
        dimensions.append(1 if compute == torch.float64 else 0)
    if not b2b:
        dimensions.append(int(silu))
    backend_fn = getattr(cudnn, binding)

    def backend():
        """Run the backend into reference buffers, resetting gradients unless accumulation is requested."""
        if backward and not args.accumulate:
            for name, value in expected.items():
                if name in ATOMIC_OUTPUTS:
                    value.zero_()
        backend_fn(torch.cuda.current_stream().cuda_stream, *(t.data_ptr() for t in inputs + list(expected.values())), *dimensions)

    backend()
    backend_identity = verify_loaded_library(requested_backend, cudnn.backend_version())
    functions = {"backend": backend}
    records = {}
    for index, item in enumerate(args.variant):
        label, path = item.split("=", 1)
        rows = 32 if nwh else 4
        if ":" in path:
            path, row_text = path.rsplit(":", 1)
            rows = int(row_text)
        path = Path(path).resolve()
        spec = importlib.util.spec_from_file_location(f"variant_{index}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        function = getattr(module, ("launch_b2b_" if b2b else "launch_") + ("bwd" if backward else "fwd"))
        if b2b:
            constants = [K, Km]
        elif backward:
            constants = [K, silu, nwh, rows]
        else:
            if not nwh:
                vector, tile = (8 if L % 8 == 0 and dtype != torch.float64 else 1), 8
            elif K >= 64:
                vector, tile = 1, 4
            else:
                vector = 4 if D % 4 == 0 and (D > 768 or B * D * L > 10_000_000) else (2 if D % 2 == 0 else 1)
                if dtype == torch.float64:
                    vector = 1
                tile = 4 if K >= 32 else (16 if dtype == torch.float32 and B * D * L >= 8_000_000 else (32 if B * D * L >= 32_000_000 else 8))
            constants = [K, silu, nwh, tile, vector, nwh and vector > 1 and torch.cuda.get_device_capability()[0] >= 10]
        if args.accumulate:
            constants.append(True)
        start = time.perf_counter()
        compiled = cute.compile(function, *(from_dlpack(t, assumed_align=16) for t in tensors), stream, *constants, options="--enable-tvm-ffi")
        compile_s = time.perf_counter() - start

        def run(compiled=compiled):
            """Launch the bound candidate variant with shared tensors on the current CUDA stream."""
            compiled(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

        checks = []
        for trial in range(3):
            for name, value in actual.items():
                value.fill_(0.25 if args.accumulate and name in ATOMIC_OUTPUTS else float("nan"))
            if args.accumulate:
                for name, value in expected.items():
                    if name in ATOMIC_OUTPUTS:
                        value.fill_(0.25)
            backend()
            run()
            checks.append(
                {
                    name: (parameter_difference(actual[name], value, B * L) if name in ATOMIC_OUTPUTS else bitwise_difference(actual[name], value))
                    for name, value in expected.items()
                }
            )
            assert all(v.get("accepted", v.get("equal", False)) for v in checks[-1].values()), (label, checks[-1])
        functions[label] = run
        records[label] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": rows,
            "constants": constants,
            "compile_seconds": compile_s,
            "checks": checks,
        }
        print(label, "accepted", compile_s, flush=True)
    output_address_control = None
    if args.shared_output_timing:

        def shared_backend():
            """Run the backend into candidate output addresses for the allocation control."""
            if backward and not args.accumulate:
                for name, value in actual.items():
                    if name in ATOMIC_OUTPUTS:
                        value.zero_()
            backend_fn(torch.cuda.current_stream().cuda_stream, *(t.data_ptr() for t in inputs + list(actual.values())), *dimensions)

        output_address_control = tune.measure({"backend_reference_outputs": backend, "backend_candidate_outputs": shared_backend})
        backend = shared_backend
        functions["backend"] = shared_backend
    frontend_checks = None
    if args.frontend:
        if not (nwh and dtype == torch.bfloat16 and K == 4 and silu):
            raise ValueError("Frontend comparison requires BF16 NWH K4 SiLU")
        if backward:
            from cudnn.causal_conv1d_bulk_sm100.backward import CausalConv1dBulkBwdPrototype

            frontend_w = w.T.contiguous()
            frontend_dw = torch.empty_like(frontend_w, dtype=compute)
            frontend = CausalConv1dBulkBwdPrototype(x, frontend_w, dy, sample_bias=bias)
            frontend.check_support()
            frontend.compile()
            frontend_workspace = torch.empty(frontend.dweight_workspace_numel, device="cuda", dtype=torch.float32) if frontend.dweight_workspace_numel else None

            def frontend_run():
                """Execute the existing frontend implementation into the comparison buffers."""
                frontend.execute(x, frontend_w, dy, actual["dx"], frontend_dw, bias=bias, db_accum=actual["db"], dweight_workspace=frontend_workspace)

            frontend_run()
            torch.testing.assert_close(actual["dx"], expected["dx"], atol=0.03, rtol=0.03)
            frontend_checks = {
                "dx": bitwise_difference(actual["dx"], expected["dx"]),
                "dw": parameter_difference(frontend_dw.T, expected["dw"], B * L),
                "db": parameter_difference(actual["db"], expected["db"], B * L),
            }
            assert frontend_checks["dw"]["accepted"] and frontend_checks["db"]["accepted"], frontend_checks
        else:
            from cudnn.causal_conv1d_bulk_sm100 import CausalConv1dBulkFwdSm100

            frontend_w = w.T.contiguous()
            frontend = CausalConv1dBulkFwdSm100(sample_x=x, sample_weight=frontend_w, sample_bias=bias, sample_output=actual["y"])
            frontend.check_support()
            frontend.compile()

            def frontend_run():
                """Execute the existing frontend implementation into the comparison buffers."""
                frontend.execute(x, frontend_w, actual["y"], bias_tensor=bias)

            frontend_run()
            torch.testing.assert_close(actual["y"], expected["y"], atol=0.03, rtol=0.03)
            frontend_checks = {"y": bitwise_difference(actual["y"], expected["y"])}
        functions["frontend"] = frontend_run
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for label, fn in functions.items():
            torch.cuda.nvtx.range_push(label)
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    measurements = tune.measure(functions)
    paired = {label: tune.measure({"backend": backend, label: fn}) for label, fn in functions.items() if label != "backend"}
    frontend_paired = {label: tune.measure({"frontend": functions["frontend"], label: functions[label]}) for label in records} if args.frontend else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "parameter_initialization": (
                    "additive; excluded from both timed paths; correctness starts from 0.25" if args.accumulate else "reset included in both timed paths"
                ),
                "gpu": torch.cuda.get_device_name(),
                "cc": torch.cuda.get_device_capability(),
                "case": str(args.case),
                "args": a,
                "backend_identity": backend_identity,
                "direction": "backward" if backward else "forward",
                "variants": records,
                "timing": measurements,
                "paired": paired,
                "frontend_paired": frontend_paired,
                "frontend_checks": frontend_checks,
                "timing_output_addresses": "shared_backend_and_candidates" if args.shared_output_timing else "shared_candidates_separate_backend",
                "output_address_control": output_address_control,
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print({k: round(v["median_us"], 3) for k, v in measurements.items()}, flush=True)


if __name__ == "__main__":
    main()
