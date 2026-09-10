# SPDX-License-Identifier: Apache-2.0

"""Compare complete QAT backward backends; never a model E2E claim.

Run against an installed checkout, for example:
python benchmark/nvfp4_attention_qat/benchmark_backends.py --output results.json
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import traceback

import cudnn
import torch
import triton


def make_inputs(heads, sequence):
    from cudnn.sdpa.bwd.qat._nvfp4 import fake_quantize_q

    q, k, v, do = (torch.randn((1, heads, sequence, 128), device="cuda", dtype=torch.bfloat16) for _ in range(4))
    fake = []
    for tensor in (q, k, v):
        quantized = torch.empty_like(tensor)
        fake_quantize_q[(triton.cdiv(sequence, 32), heads)](
            tensor,
            quantized,
            *tensor.stride(),
            *quantized.stride(),
            heads,
            sequence,
            block_m=32,
            head_dim=128,
            num_warps=4,
            num_stages=2,
        )
        fake.append(quantized.float())
    o = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    # Chunk the reference forward to avoid allocating full FP32 S-by-S scores.
    for first in range(0, sequence, 128):
        rows = slice(first, min(first + 128, sequence))
        scores = fake[0][:, :, rows] @ fake[1].transpose(-1, -2) * 128**-0.5
        lse[:, :, rows] = scores.logsumexp(-1)
        o[:, :, rows] = scores.softmax(-1) @ fake[2]
    return q, k, v, o, do, lse


def compare(actual, reference):
    metrics = []
    for a, b in zip(actual, reference):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError("nonfinite gradients")
        error = a.float() - b.float()
        relative_l2 = (error.norm() / b.float().norm().clamp_min(1e-30)).item()
        torch.testing.assert_close(a, b, atol=0.005, rtol=0.005)
        if relative_l2 >= 0.01:
            raise AssertionError(f"relative L2 {relative_l2} >= 0.01")
        metrics.append(dict(max_abs=error.abs().max().item(), relative_l2=relative_l2, exact=torch.equal(a, b)))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--head-chunk", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(status="fail", scope="complete QAT backward component, not model E2E", cases=[])
    try:
        torch.manual_seed(20260910)
        torch.backends.cuda.matmul.allow_tf32 = False
        props = torch.cuda.get_device_properties(0)
        package = Path(cudnn.__file__).parent
        source_files = [Path(__file__)] + list((package / "sdpa/bwd/qat").glob("*.py")) + list((package / "frost/tile_dsl").glob("*.py"))
        report.update(
            device=props.name,
            sm_count=props.multi_processor_count,
            capability=torch.cuda.get_device_capability(),
            torch=torch.__version__,
            triton=triton.__version__,
            cutedsl=importlib.metadata.version("nvidia-cutlass-dsl"),
            cudnn_frontend=cudnn.__version__,
            cuda=torch.version.cuda,
            seed=20260910,
            source_sha256={
                str(p.relative_to(package)) if p.is_relative_to(package) else p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
            },
        )
        for sequence in args.lengths:
            inputs = make_inputs(args.heads, sequence)
            arms = {}
            case = dict(sequence=sequence, heads=args.heads, head_chunk=args.head_chunk, status="checking")
            report["cases"].append(case)
            for backend in ("triton", "cutedsl"):
                op = cudnn.Nvfp4AttentionQatBackward(*inputs, backend=backend, head_chunk=args.head_chunk if backend == "cutedsl" else 0)
                op.check_support()
                op.compile()
                outputs = tuple(torch.empty_like(t) for t in inputs[:3])
                workspace = torch.empty(op.scratch_workspace_bytes(), device=inputs[0].device, dtype=torch.uint8)

                def run(op=op, outputs=outputs, workspace=workspace):
                    op.execute(*inputs, *outputs, workspace)

                for _ in range(3):
                    run()
                arms[backend] = (outputs, workspace, run)
            case["eager_comparisons"] = compare(arms["cutedsl"][0], arms["triton"][0])
            graphs = {}
            for backend, (_, _, run) in arms.items():
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(3):
                        run()
                graphs[backend] = graph
            inputs[4].normal_()
            for outputs, workspace, _ in arms.values():
                workspace.fill_(255)
                for tensor in outputs:
                    tensor.fill_(float("nan"))
            for graph in graphs.values():
                graph.replay()
            case["graph_comparisons"] = compare(arms["cutedsl"][0], arms["triton"][0])
            case["workspace_bytes"] = {key: arm[1].numel() for key, arm in arms.items()}
            if not args.check_only:
                samples = {backend: [] for backend in arms}
                for round_index in range(5):
                    order = ("triton", "cutedsl", "cutedsl", "triton") if round_index % 2 == 0 else ("cutedsl", "triton", "triton", "cutedsl")
                    for backend in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[backend].replay()
                        end.record()
                        end.synchronize()
                        samples[backend].append(start.elapsed_time(end) / 3)
                case["samples_ms"] = samples
                case["median_ms"] = {key: statistics.median(values) for key, values in samples.items()}
                case["speedup"] = case["median_ms"]["triton"] / case["median_ms"]["cutedsl"]
            case["status"] = "pass"
            print(json.dumps(case), flush=True)
            del graphs, arms, inputs
        report["status"] = "pass"
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
