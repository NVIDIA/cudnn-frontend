# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure FE artifact reuse and require exact data outputs after reload."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from cutlass import cute


def main():
    """Check disk and in-process artifact reuse, output correctness, and CUDA-graph replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-disk-hit", action="store_true")
    parser.add_argument("--large-nwh", action="store_true", help="Also exercise the staged large NWH forward/backward specializations")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "test/python/causal_conv1d/cutedsl"))
    from se_bitwise import run_case, bitwise_difference, parameter_difference, ATOMIC_OUTPUTS
    import cuda.bindings.driver as cuda
    from cudnn.frost import compiled_cache

    torch.cuda.init()
    torch.empty(1, device="cuda")
    original = cute.compile
    records = []
    for family in (("ncw", "nwh", "b2b", "nwh_pipeline") if args.large_nwh else ("ncw", "nwh", "b2b")):
        for direction in ("forward", "backward"):
            case = dict(
                name=f"compile_probe_{family}_{direction}",
                family="nwh" if family == "nwh_pipeline" else family,
                direction=direction,
                shape=[1, 12288, 8192] if family == "nwh_pipeline" else ([2, 64, 256] if family == "b2b" else [2, 768, 4096]),
                width=3 if family == "b2b" else 4,
                mixer_width=7 if family == "b2b" else None,
                dtype="bfloat16",
                silu=family != "b2b",
                iterations=3,
                min_device_version=800,
            )
            key = "causal-se-compile-probe-v1:" + hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
            for repeat in range(2):
                compile_info = {}
                launch = {}

                def forbidden(*values, **kwargs):
                    """Fail if a cache-hit path attempts to compile a new kernel."""
                    raise AssertionError("Cache reuse attempted to invoke cute.compile")

                def cached(fn, *values, **kwargs):
                    """Measure a cache lookup or compilation and retain the compiled launch for replay."""
                    before = compiled_cache.stats()
                    cute.compile = forbidden if args.require_disk_hit or repeat else original
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    try:
                        result = compiled_cache.compile_cached(fn, *values, cache_key=key, **kwargs)
                    finally:
                        cute.compile = cached
                    compile_info.update(seconds=time.perf_counter() - start, stats_delta={k: v - before[k] for k, v in compiled_cache.stats().items()})
                    launch["function"] = result

                    def invoke(*runtime):
                        """Record runtime arguments while invoking the cached compiled function."""
                        launch["arguments"] = runtime
                        return result(*runtime)

                    return invoke

                cute.compile = cached
                try:
                    result = run_case(case)
                    assert result["accepted"], result
                finally:
                    cute.compile = original
                output_names = (
                    (["dx", "dwp", "dwm", "dskip"] if family == "b2b" else ["dx", "dw", "db"])
                    if direction == "backward"
                    else (["y", "yg"] if family == "b2b" else ["y"])
                )
                runtime = launch["arguments"]
                outputs = runtime[-1 - len(output_names) : -1]
                snapshots = [t.clone() for t in outputs]
                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    launch["function"](*runtime[:-1], cuda.CUstream(torch.cuda.current_stream().cuda_stream))
                for t in outputs:
                    t.fill_(float("nan"))
                graph.replay()
                graph_checks = {
                    name: (parameter_difference(t, ref, case["shape"][0] * case["shape"][2]) if name in ATOMIC_OUTPUTS else bitwise_difference(t, ref))
                    for name, t, ref in zip(output_names, outputs, snapshots)
                }
                assert all(v.get("accepted", v.get("equal", False)) for v in graph_checks.values()), graph_checks
                compile_info["graph_replay"] = graph_checks
                expected = "memo_hits" if repeat else ("hits" if args.require_disk_hit else "misses")
                assert compile_info["stats_delta"][expected] == 1, compile_info
                record = {
                    "case": case,
                    "repeat": repeat,
                    "mode": expected,
                    **compile_info,
                    "accepted": result["accepted"],
                    "kernel_sha256": result["kernel_sha256"],
                }
                records.append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "gpu": torch.cuda.get_device_name(),
                            "cc": torch.cuda.get_device_capability(),
                            "require_disk_hit": args.require_disk_hit,
                            "records": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"{family} {direction} {expected}: {compile_info['seconds']:.6f} s", flush=True)


if __name__ == "__main__":
    main()
