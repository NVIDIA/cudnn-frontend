# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare a packed-convolution case with cuhyena and the existing FE implementation.

Run once per case and process. Compilation and allocation precede graph timing.
The caller owns environment setup; imports must resolve to the candidate checkout.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import traceback

import torch
import cudnn
from cuda.bindings import driver as cuda
from cutlass.base_dsl.version_info import CUDA_VERSION
from cudnn.causal_conv1d_se.packed import _compiled
from subquadratic_ops_torch.utils import get_cuda_arch, get_module, get_sm_count

from timing import measure


def hashes(paths):
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def lengths_for(case):
    if "lengths" in case:
        return case["lengths"]
    kind = case["distribution"]
    if kind == "decode":
        return [1] * case["sequences"]
    if kind == "mixed":
        return [2048] * 4 + [1] * 64
    tokens = case["tokens"]
    if kind.startswith("uniform-"):
        length = int(kind.split("-")[1])
        assert tokens % length == 0
        return [length] * (tokens // length)
    if kind == "fewlong":
        assert tokens % 4 == 0
        return [tokens // 4] * 4
    if kind == "lognormal":
        rng = torch.Generator().manual_seed(case.get("seed", 0))
        lengths, total = [], 0
        while total < tokens:
            draw = torch.normal(torch.tensor(6.5), torch.tensor(1.2), generator=rng).item()
            length = max(1, min(tokens - total, int(math.exp(draw))))
            lengths.append(length)
            total += length
        return lengths
    raise ValueError(f"unknown distribution {kind}")


def idle_gpu():
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True).splitlines()
    foreign = [row for row in rows if uuid in row and row.rsplit(",", 1)[-1].strip() != str(os.getpid())]
    if foreign:
        raise RuntimeError(f"foreign GPU work: {foreign}")


def difference(actual, expected, dtype, exact=False):
    # Bound temporary storage for the largest FP32 matrix on 40-48 GB GPUs.
    # Numerical verification is outside timing and may synchronize per chunk.
    actual_flat, expected_flat = actual.reshape(-1), expected.reshape(-1)
    squared_error, squared_reference, maximum = 0.0, 0.0, 0.0
    for first in range(0, actual.numel(), 1 << 20):
        a = actual_flat[first : first + (1 << 20)].double()
        b = expected_flat[first : first + (1 << 20)].double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError("nonfinite numerical result")
        delta = a - b
        squared_error += delta.square().sum().item()
        squared_reference += b.square().sum().item()
        maximum = max(maximum, delta.abs().max().item())
    relative_l2 = math.sqrt(squared_error) / max(math.sqrt(squared_reference), 1e-30)
    limit = {torch.float32: 2e-4, torch.float16: 0.003, torch.bfloat16: 0.02}[dtype]
    if (exact and maximum != 0) or relative_l2 > limit:
        raise AssertionError(f"numerical mismatch: relative_l2={relative_l2}, max_abs={maximum}, limit={limit}")
    return dict(relative_l2=relative_l2, max_abs=maximum, relative_l2_limit=0 if exact else limit)


def sampled_reference(x, w, b, initial, dy, dfinal, lengths, silu, backward):
    """FP64 grouped convolution/autograd on eight channels, across every token."""
    channels = torch.linspace(0, x.shape[1] - 1, min(8, x.shape[1]), device=x.device).long()
    xx, ww, bb = (t.double().detach().requires_grad_() for t in (x[:, channels], w[channels], b[channels]))
    ii = initial[:, :, channels].double().detach().requires_grad_() if initial is not None else None
    ys, fs, offset = [], [], 0
    for seq, length in enumerate(lengths):
        prefix = xx.new_zeros((w.shape[1] - 1, channels.numel())) if ii is None else ii[seq]
        joined = torch.cat((prefix, xx[offset : offset + length]))
        if length:
            y = torch.nn.functional.conv1d(joined.T[None], ww[:, None, :], bb, groups=channels.numel())[0].T
            ys.append(torch.nn.functional.silu(y) if silu else y)
        fs.append(joined[-(w.shape[1] - 1) :])
        offset += length
    y, final = torch.cat(ys), torch.stack(fs)
    if not backward:
        return channels, {"y": y.detach(), **({"final": final.detach()} if dfinal is not None else {})}
    inputs = (xx, ww, bb) + ((ii,) if ii is not None else ())
    outputs, grads = (y,), (dy[:, channels].double(),)
    if dfinal is not None:
        outputs += (final,)
        grads += (dfinal[:, :, channels].double(),)
    values = torch.autograd.grad(outputs, inputs, grads)
    return channels, dict(zip(("dx", "dw", "db", "di"), values))


def frontend_call(case, layout, lengths, x, w, b, cu, initial, dy, dfinal, outputs):
    if case["dtype"] != "bfloat16" or case["width"] != 4 or case["activation"] != "silu":
        return None, {"status": "unsupported", "reason": "existing FE packed path requires BF16 K4 SiLU"}
    if layout != "HK" or not all(length > 0 for length in lengths):
        return None, {"status": "unsupported", "reason": "existing FE requires contiguous [H,K] filter and nonempty sequences"}
    from cudnn.causal_conv1d_bulk_sm100.api import CausalConv1dBulkFwdSm100
    from cudnn.causal_conv1d_bulk_sm100.backward import compile_causal_conv1d_bulk_bwd_prototype

    def full_state(state):
        if state is None:
            return None
        result = state.new_zeros((len(lengths), x.shape[1], 4))
        result[:, :, 1:].copy_(state.permute(0, 2, 1))
        return result

    init = full_state(initial)
    info = {
        "status": "supported",
        "input_state_conversion": "prepared before timing; input state is read-only",
        "output_conversion": "final/dinitial layout and parameter dtype conversions included in timing",
    }
    if case["direction"] == "forward":
        final = None if "final" not in outputs else full_state(outputs["final"])
        api = CausalConv1dBulkFwdSm100(x[None], w, outputs["y"][None], cu, init, final, sample_bias=b)
        api.check_support()
        api.compile()

        def call():
            api.execute(x[None], w, outputs["y"][None], cu, init, final, bias_tensor=b)
            if final is not None:
                outputs["final"].copy_(final[:, :, 1:].permute(0, 2, 1))

    else:
        dfin, di = full_state(dfinal), full_state(outputs.get("di"))
        api = compile_causal_conv1d_bulk_bwd_prototype(x[None], w, dy[None], cu, bias=b, initial_state=init, d_final_state=dfin)
        dw, db = torch.empty_like(w, dtype=torch.float32), torch.empty_like(b, dtype=torch.float32)
        tile_map = torch.empty(api.packed_tile_map_numel, device=x.device, dtype=torch.int32)
        workspace = torch.empty(api.dweight_workspace_numel, device=x.device, dtype=torch.float32) if api.dweight_workspace_numel else None
        info["schedule"] = api.schedule
        info["dweight_mode"] = api.dweight_mode

        def call():
            api.execute(
                x[None],
                w,
                dy[None],
                outputs["dx"][None],
                dw,
                cu_seqlens=cu,
                packed_tile_map=tile_map,
                dweight_workspace=workspace,
                bias=b,
                db_accum=db,
                initial_state=init,
                d_final_state=dfin,
                d_initial_state=di,
            )
            outputs["dw"].copy_(dw)
            outputs["db"].copy_(db)
            if di is not None:
                outputs["di"].copy_(di[:, :, 1:].permute(0, 2, 1))

    return call, info


def run(case, args):
    lengths = lengths_for(case)
    if not sum(lengths):
        raise ValueError("empty packs are correctness cases; this harness times nonempty packs")
    dtype, h, k = getattr(torch, case["dtype"]), case["hidden"], case["width"]
    silu, backward = case["activation"] == "silu", case["direction"] == "backward"
    generator = torch.Generator().manual_seed(args.seed)
    data_hash = hashlib.sha256()

    def rand(shape):
        data = torch.randn(shape, generator=generator)
        data_hash.update(str(tuple(shape)).encode())
        data_hash.update(memoryview(data.numpy()))
        return data.to(device="cuda", dtype=dtype)

    x, weight = rand((sum(lengths), h)), rand((h, k) if args.weight_layout == "HK" else (k, h))
    w = weight if args.weight_layout == "HK" else weight.T
    b = rand((h,)) if args.random_bias else torch.zeros(h, device="cuda", dtype=dtype)
    initial = rand((len(lengths), k - 1, h)) if case["initial"] else None
    dy = rand(x.shape)
    dfinal = rand((len(lengths), k - 1, h)) if case["final"] else None
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    stub = x.new_empty((0, k - 1, h))
    if backward:
        outputs = {"dx": torch.empty_like(x), "dw": torch.empty_like(w), "db": torch.empty_like(b)}
        if initial is not None:
            outputs["di"] = torch.empty_like(initial)
        partial_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        partial_bytes = 8 if partial_dtype == torch.float64 else 4
        splits = max(1, min(64, (16 * 1024 * 1024) // (h * (k + 1) * partial_bytes)))
        scratch = torch.empty(splits, k + 1, h, device="cuda", dtype=partial_dtype)
        tensors = (
            x,
            w,
            b,
            cu,
            initial,
            dy,
            dfinal,
            torch.empty_like(x, dtype=torch.float32),
            outputs["dx"],
            outputs["dw"],
            outputs["db"],
            outputs.get("di"),
            scratch,
        )
    else:
        outputs = {"y": torch.empty_like(x)}
        if case["final"]:
            outputs["final"] = torch.empty_like(dfinal)
        tensors = (x, w, b, cu, initial, outputs["y"], outputs.get("final"))
    candidate_kernel = _compiled(case["direction"], tensors, k, silu)
    module = get_module("causal_conv1d_packed_" + ("bwd" if backward else "fwd"), [dtype])
    arch, sms = get_cuda_arch(0), get_sm_count(0)
    split_s = max(1, min(64, (16 * 1024 * 1024) // (h * k * 4)))
    dw_partial = torch.empty(split_s, h, k, device=x.device, dtype=torch.float32) if backward else None
    db_partial = torch.empty(split_s, h, device=x.device, dtype=torch.float32) if backward else None

    def baseline_for(out):
        if backward:
            dw = out["dw"] if args.weight_layout == "HK" else out["dw"].T

            def call():
                module(
                    out["dx"],
                    dw,
                    out["db"],
                    out.get("di", stub),
                    x,
                    dy,
                    weight,
                    b,
                    initial if initial is not None else stub,
                    dfinal if dfinal is not None else stub,
                    cu,
                    dw_partial,
                    db_partial,
                    k,
                    int(silu),
                    int(initial is not None),
                    int(initial is not None),
                    int(dfinal is not None),
                    sms,
                    arch,
                    torch.cuda.current_stream().cuda_stream,
                )

        else:

            def call():
                module(
                    out["y"],
                    out.get("final", stub),
                    x,
                    weight,
                    b,
                    initial if initial is not None else stub,
                    cu,
                    k,
                    int(silu),
                    int(initial is not None),
                    int(case["final"]),
                    sms,
                    arch,
                    torch.cuda.current_stream().cuda_stream,
                )

        return call

    def candidate():
        candidate_kernel(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    baseline = baseline_for(outputs)
    baseline()
    expected = {key: value.clone() for key, value in outputs.items()}
    channels, reference = sampled_reference(x, w, b, initial, dy, dfinal, lengths, silu, backward)
    oracle_metrics = {}
    for key, value in reference.items():
        sample = expected[key][channels] if key in ("dw", "db") else expected[key][..., channels]
        oracle_metrics[key] = difference(sample, value, dtype, exact=key == "final")
    del reference
    candidate()
    numerics = {key: difference(value, expected[key], dtype, exact=key == "final") for key, value in outputs.items()}
    frontend, fe_info = frontend_call(case, args.weight_layout, lengths, x, w, b, cu, initial, dy, dfinal, outputs)
    if frontend is not None:
        frontend()
        fe_info["numerics"] = {key: difference(value, expected[key], dtype, exact=key == "final") for key, value in outputs.items()}
    alternate = {key: torch.empty_like(value) for key, value in outputs.items()}
    aa_call = baseline_for(alternate)
    aa_call()
    for key, value in alternate.items():
        difference(value, expected[key], dtype, exact=key == "final")
    idle_gpu()
    options = dict(repeats=args.graph_calls, rounds=args.rounds, cold=args.cold)
    aa = measure({"cuhyena": baseline, "cuhyena_other_output": aa_call}, **options)
    pair = measure({"cuhyena": baseline, "candidate": candidate}, **options)
    fe_pair = measure({"frontend": frontend, "candidate": candidate}, **options) if frontend is not None else None
    idle_gpu()
    return dict(
        case=case,
        lengths=lengths,
        layout=args.weight_layout,
        random_bias=args.random_bias,
        generated_cpu_fp32_sha256=data_hash.hexdigest(),
        numerical_reference="FP64 grouped conv/autograd, eight channels, every token",
        cuhyena_reference_metrics=oracle_metrics,
        candidate_metrics=numerics,
        frontend=fe_info,
        aa=aa,
        cuhyena_pair=pair,
        frontend_pair=fe_pair,
        cuhyena_speedup=pair["cuhyena"]["median_us"] / pair["candidate"]["median_us"],
        frontend_speedup=None if fe_pair is None else fe_pair["frontend"]["median_us"] / fe_pair["candidate"]["median_us"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="manifest case ID, or JSON file containing one case")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("packed_workloads.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-layout", choices=("HK", "KH"), default="KH")
    parser.add_argument("--random-bias", action="store_true")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--graph-calls", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--cold", action="store_true", help="one call after cache eviction; separate from repeated-buffer warm timing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    import cudnn.causal_conv1d_se.packed as candidate

    if Path(candidate.__file__).resolve() != root / "python/cudnn/causal_conv1d_se/packed.py":
        raise RuntimeError(f"candidate import resolved to {candidate.__file__}, expected {root}")
    manifest = json.loads(args.manifest.read_text())
    case = json.loads(Path(args.case).read_text()) if Path(args.case).is_file() else next(c for c in manifest["primary_cases"] if c["id"] == args.case)
    paths = [Path(__file__), Path(__file__).with_name("timing.py"), args.manifest]
    paths += list((root / "python/cudnn/causal_conv1d_se").glob("*.py"))
    paths += list((root / "python/cudnn/causal_conv1d_bulk_sm100").glob("*.py"))
    import subquadratic_ops_torch

    reference_package = Path(subquadratic_ops_torch.__file__).resolve().parent
    paths += [reference_package / "causal_conv1d_packed.py", reference_package / "_ext/libsubquadratic_ops.so"]
    source = hashes(paths)
    properties = torch.cuda.get_device_properties(0)
    report = dict(
        status="running",
        source=source,
        args=vars(args).copy(),
        baseline=manifest["reference"],
        frontend_revision=manifest["frontend_baseline"],
        gpu=properties.name,
        gpu_uuid=str(properties.uuid),
        arch=torch.cuda.get_device_capability(),
        sm_count=properties.multi_processor_count,
        compiler=str(CUDA_VERSION),
        dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        torch=torch.__version__,
        backend=cudnn.backend_version(),
        compiled_arch=os.environ.get("CUTE_DSL_ARCH"),
        driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        timing="cache-evicted single graph call" if args.cold else "cache-warm repeated buffers in CUDA graphs",
        qualification="single process; repeat in three predetermined fresh processes before accepting performance",
    )
    try:
        idle_gpu()
        report["result"] = run(case, args)
        if hashes(paths) != source:
            raise RuntimeError("source changed during comparison")
        report["status"] = "pass"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["loaded_libraries"] = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if any(
                    name in line
                    for name in ("libnvrtc", "libcudnn", "libnvJitLink", "libcuda.so", "libnvptxcompiler", "libsubquadratic", "cutlass", "CuteDSL", "CuTeDSL")
                )
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
