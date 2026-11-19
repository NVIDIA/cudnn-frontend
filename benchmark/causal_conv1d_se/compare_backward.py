# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare preallocated SE and Cake backward on an identical BF16 K4 SiLU workload.

Run three fresh processes per case. All compilation and allocation precede
timing. Dense filter preparation is excluded; gradient conversion is included.
The implementations use different dW/db accumulation methods.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import torch
from cuda.bindings import driver as cuda
from cutlass.base_dsl.version_info import CUDA_VERSION

from cudnn.causal_conv1d_cake.api import CausalConv1dPlan
from cudnn.causal_conv1d_se import packed, dense
from timing import measure
from reference_checks import reference_check, replay_check


def _hashes():
    package = Path(packed.__file__).parent
    files = list(package.glob("*.py")) + list((package.parent / "causal_conv1d_cake").glob("*.py"))
    files += [
        package.parent / "_cake/_vendor/VENDORED_CAUSAL_CONV1D.json",
        Path(__file__),
        Path(__file__).with_name("backward_workloads.json"),
        Path(__file__).with_name("timing.py"),
        Path(__file__).with_name("reference_checks.py"),
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _idle():
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True).splitlines()
    foreign = [row for row in rows if uuid in row and int(row.rsplit(",", 1)[-1]) != os.getpid()]
    if foreign:
        raise RuntimeError(f"foreign work on benchmark GPU: {foreign}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=("packed", "dense"), default="packed")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    case = next(c for c in json.loads(Path(__file__).with_name("backward_workloads.json").read_text()) if c["id"] == args.case)
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)) or (args.mode == "dense" and capability != (10, 3)):
        raise RuntimeError("the comparison requires SM100/103 for packed mode or SM103 for dense mode")
    before = _hashes()
    _idle()
    lengths, channels = case["lengths"], case["channels"]
    tokens = sum(lengths)
    generator = torch.Generator().manual_seed(20261113)

    def rand(shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(device="cuda", dtype=torch.bfloat16)

    x, w, b, dy = rand((tokens, channels)), rand((channels, 4), 0.15), rand((channels,), 0.1), rand((tokens, channels))
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device=x.device, dtype=torch.int32)
    out = {"dx": torch.empty_like(x), "dw": torch.empty_like(w), "db": torch.empty_like(b)}
    plan = CausalConv1dPlan(tokens, channels, len(lengths), device=x.device, weight_dtype=w.dtype, bias_dtype=b.dtype)
    workspace = torch.empty(plan.get_workspace_size(), device=x.device, dtype=torch.uint8)

    def cake():
        plan.backward(dy, x, w, cu, out["dx"], out["dw"], workspace, bias=b, grad_bias=out["db"])

    dz = torch.empty_like(x, dtype=torch.float32)
    if args.mode == "packed":
        route, splits = packed._backward_schedule(x, w, b, None, dy, None, True)
        partials = torch.empty((splits, 5, channels), device=x.device, dtype=torch.float32)
        tensors = (x, w, b, cu, None, dy, None, dz, out["dx"], out["dw"], out["db"], None, partials)
        fn = packed._compiled("backward", tensors, 4, True)

        def se():
            fn(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    else:
        if len(set(lengths)) != 1:
            raise ValueError("dense mode requires equal sequence lengths")
        shape = (len(lengths), lengths[0], channels)
        wk = w.T.contiguous()
        if dense._persistent_shape(shape, (10, 3)):
            route = "persistent-dense"
            splits = max(1, 2 * torch.cuda.get_device_properties(0).multi_processor_count // ((channels + 255) // 256))
            partials = torch.empty((splits, 5, channels), device=x.device, dtype=torch.float32)
            tensors = (x, wk.T, b, cu, None, dy, None, dz, out["dx"], out["dw"], out["db"], None, partials)
            fn = dense._compiled("persistent", tensors)

            def se():
                fn(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

        else:
            route, splits = "dense-tiled", None
            dw = torch.empty_like(wk, dtype=torch.float32)
            db = torch.empty_like(b, dtype=torch.float32)
            tensors = (x.view(shape), wk, b, dy.view(shape), dz.view(shape), out["dx"].view(shape), dw, db)
            fn = dense._compiled("backward", tensors)

            def se():
                fn(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
                out["dw"].copy_(dw.T)
                out["db"].copy_(db)

    cake()
    control = {k: v.clone() for k, v in out.items()}
    se()
    errors = {}
    for name, value in out.items():
        assert torch.isfinite(value).all(), name
        a, ref = value.float(), control[name].float()
        error = ((a - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        assert error <= 0.002, (name, error)
        errors[name] = error
    reference_errors = reference_check(out, x, w, b, dy, lengths)
    replay = replay_check(se, out, route) if args.repeat == 0 else None
    _idle()
    aa = measure({"cake_a": cake, "cake_b": cake}, repeats=20, rounds=21)
    timing = measure({"cake": cake, "se": se}, repeats=20, rounds=21)
    _idle()
    assert before == _hashes(), "sources changed during benchmark"
    props = torch.cuda.get_device_properties(0)
    result = dict(
        case=case,
        mode=args.mode,
        repeat=args.repeat,
        route=route,
        splits=splits,
        timing=timing,
        speedup=timing["cake"]["median_us"] / timing["se"]["median_us"],
        relative_l2=errors,
        reference_errors=reference_errors,
        replay=replay,
        aa_control=aa,
        hashes=before,
        environment=dict(
            gpu=props.name,
            capability=capability,
            compiled_target=os.environ.get("CUTE_DSL_ARCH"),
            uuid=str(props.uuid),
            sms=props.multi_processor_count,
            torch=torch.__version__,
            compiler=str(CUDA_VERSION),
            dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
            allocation=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"case": args.case, "mode": args.mode, "repeat": args.repeat, "speedup": result["speedup"], "relative_l2": errors}), flush=True)


if __name__ == "__main__":
    main()
