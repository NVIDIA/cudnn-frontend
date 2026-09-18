# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete graph-engine SSD passes against unmodified mamba_ssm Triton.

Device-only CUDA-graph timings include dt preprocessing, state propagation,
optional SiLU gate, backward recomputation and every gradient reduction.
No GatedRMSNorm, convolution, projection or model execution is included.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import types

import torch
import cudnn

ops = importlib.import_module("cudnn.linear_attention.ops.mamba2")


def load_mamba(path):
    # Load the original Triton modules without requiring the unrelated Mamba-1
    # selective_scan extension. Module source is neither patched nor copied.
    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton", "mamba_ssm.utils"):
        module = types.ModuleType(name)
        module.__path__ = [str(path / name.replace(".", "/"))]
        sys.modules[name] = module
    return importlib.import_module("mamba_ssm.ops.triton.ssd_combined")


def make_inputs(batch, length, heads, groups, gate):
    torch.manual_seed(42)
    x = torch.randn(batch, length, heads, 64, device="cuda", dtype=torch.bfloat16)
    dt = torch.randn(batch, length, heads, device="cuda", dtype=torch.bfloat16)
    a = -torch.empty(heads, device="cuda").uniform_(1, 16)
    b = torch.randn(batch, length, groups, 64, device="cuda", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn(heads, device="cuda")
    delta = torch.exp(torch.empty(heads, device="cuda").uniform_(-6.9, -2.3))
    bias = delta + torch.log(-torch.expm1(-delta))
    return dict(x=x, dt=dt, A=a, B=b, C=c, D=d, dt_bias=bias, z=torch.randn_like(x) if gate else None)


def prepare(values, bwd, precision):
    graph, ports, handle = ops._get_graph(values, bwd, 32, precision, output_final_state=not bwd)
    outputs = ops._outputs(values, bwd, output_final_state=not bwd, intermediate_dtype=precision)
    all_values = {**values, **outputs}
    pack = {port: all_values[name] for name, port in ports.items()}
    workspace = torch.empty(graph.get_workspace_size(), device=values["x"].device, dtype=torch.uint8)

    def run():
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
        graph.execute(pack, workspace=workspace, handle=handle)
        return outputs

    return run, outputs, graph.get_workspace_size()


def timer(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 150)
    return dict(median_us=statistics.median(samples), samples_us=samples)


def error(actual, expected):
    a, e = actual.double(), expected.double()
    diff = a - e
    rms = (diff.square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-10)).item()
    peak = (diff.abs().max() / e.abs().max().clamp_min(1e-10)).item()
    assert torch.isfinite(a).all() and rms < 0.01 and peak < 0.015, (rms, peak)
    return dict(relative_rms=rms, relative_peak=peak)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mamba-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intermediate-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--groups", type=int, default=1)
    args = parser.parse_args()
    mod = load_mamba(args.mamba_repo)
    git = lambda *a: subprocess.check_output(["git", "-C", str(args.mamba_repo), *a], text=True).strip()
    root = Path(cudnn.__file__).parent
    files = [
        root / "linear_attention/frost/mamba2_engine.py",
        root / "linear_attention/ops/mamba2.py",
        *sorted((root / "linear_attention/frost/kernel").glob("mamba2*.py")),
    ]
    report = dict(
        shape=dict(B=args.batch, L=args.length, H=args.heads, P=64, G=args.groups, N=64, chunk_size=32, dtype="bfloat16"),
        intermediate_dtype=args.intermediate_dtype,
        reuse_forward_states=False,
        environment=dict(
            gpu=torch.cuda.get_device_name(),
            sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=cudnn.backend_version(),
            frontend_path=str(root),
            mamba_commit=git("rev-parse", "HEAD"),
            mamba_dirty=bool(git("status", "--porcelain")),
        ),
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        modes=[],
    )
    for gate in (False, True):
        values = make_inputs(args.batch, args.length, args.heads, args.groups, gate)
        x, dt, A, B, C, D, bias, z = (values[k] for k in ("x", "dt", "A", "B", "C", "D", "dt_bias", "z"))
        dy = torch.randn_like(x)
        fwd, saved, fwd_bytes = prepare(values, False, args.intermediate_dtype)
        fwd()
        backward_values = {**values, "dO": dy, "ungated_out": saved["ungated_out"] if gate else None}
        bwd, grads, bwd_bytes = prepare(backward_values, True, args.intermediate_dtype)

        def pair():
            fwd()
            return bwd()

        def baseline_fwd():
            return mod._mamba_chunk_scan_combined_fwd(x, dt, A, B, C, 32, D=D, z=z, dt_bias=bias, dt_softplus=True)

        reference = baseline_fwd()

        def baseline_bwd(pre_gate=reference[1] if gate else reference[0]):
            return mod._mamba_chunk_scan_combined_bwd(dy, x, dt, A, B, C, pre_gate, 32, D=D, z=z, dt_bias=bias, dt_softplus=True)

        def baseline_pair():
            result = baseline_fwd()
            return baseline_bwd(result[1] if gate else result[0])

        pair()
        ref_grads = baseline_bwd()
        row = dict(
            gate=gate,
            workspace_bytes=dict(forward=fwd_bytes, backward=bwd_bytes),
            output=error(saved["O"], reference[0]),
            final_state=error(saved["final_state"], reference[5]),
            gradients={},
        )
        for name, index in (("dX", 0), ("dDt", 1), ("dA", 2), ("dB", 3), ("dC", 4), ("dD", 5), ("d_dt_bias", 7), ("dZ", 6)):
            if name != "dZ" or gate:
                row["gradients"][name] = error(grads[name], ref_grads[index])
        row["timing"] = {
            name: timer(fn)
            for name, fn in (
                ("triton_forward", baseline_fwd),
                ("native_forward", fwd),
                ("triton_backward", baseline_bwd),
                ("native_backward", bwd),
                ("triton_pair", baseline_pair),
                ("native_pair", pair),
            )
        }
        row["speedup"] = {p: row["timing"]["triton_" + p]["median_us"] / row["timing"]["native_" + p]["median_us"] for p in ("forward", "backward", "pair")}
        report["modes"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(gate=gate, **{k: v["median_us"] for k, v in row["timing"].items()}, speedup=row["speedup"])), flush=True)


if __name__ == "__main__":
    main()
