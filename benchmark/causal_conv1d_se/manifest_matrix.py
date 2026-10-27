# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run every named backend workload and the complete frontend batch sweep."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import torch


def key(case, direction):
    """Build a workload identity from the manifest case and requested direction."""
    return (case["family"], case["dtype"], tuple(case["shape"]), case["width"], case.get("mixer_width"), case["silu"], direction)


def measured_key(data):
    """Recover a workload identity from retained warm-cache benchmark arguments."""
    a = data.get("args", {})
    if "projection_width" in a:
        return ("b2b", a["dtype"], tuple(map(int, a["shape"].split(","))), a["projection_width"], a["mixer_width"], False, "both")
    if "layout" not in a or a.get("cold", False):
        return None
    direction = "backward" if "rows" in a else "forward"
    return (a["layout"], a["dtype"], tuple(map(int, a["shape"].split(","))), a["width"], None, not a["identity"], direction)


def main():
    """Run the manifest grid, reusing only results with matching GPU, backend, source, and schedule."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend-library", required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("workloads.json"))
    parser.add_argument("--reuse", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bench = Path(__file__).parent
    kernels = bench.parents[1] / "python/cudnn/causal_conv1d_se"
    expected = {name: hashlib.sha256((kernels / name).read_bytes()).hexdigest() for name in ("kernel.py", "backward_kernel.py", "b2b_kernel.py")}
    cache = {}
    gpu = torch.cuda.get_device_name()
    cc = list(torch.cuda.get_device_capability())
    for folder in args.reuse:
        for p in folder.glob("*.json"):
            try:
                data = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or data.get("gpu") != gpu or data.get("cc") != cc:
                continue
            if str(Path(data.get("args", {}).get("backend_library", "")).resolve()) != str(Path(args.backend_library).resolve()):
                continue
            mk = measured_key(data)
            if mk is None:
                continue
            settings = data["args"]
            if mk[-1] == "forward" and settings.get("schedules") != "auto":
                continue
            if mk[-1] == "backward" and str(settings.get("rows")) != ("4" if mk[0] == "ncw" else "32"):
                continue
            name = "b2b_kernel.py" if mk[0] == "b2b" else ("backward_kernel.py" if mk[-1] == "backward" else "kernel.py")
            if expected[name] not in [v for k, v in data.get("sha256", {}).items() if Path(k).name == name]:
                continue
            values = data.get("results", data.get("timing", {}))
            if mk[-1] == "both":
                required = ("candidate_fwd", "backend_fwd", "candidate_bwd", "backend_bwd")
            else:
                candidate_names = [k for k in values if k not in ("backend", "frontend")]
                if len(candidate_names) != 1:
                    continue
                required = ("backend", candidate_names[0])
            medians = [values.get(name, {}).get("median_us") for name in required]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in medians):
                continue
            cache[mk] = (p, data)
    cases = json.loads(args.manifest.read_text())["cases"]
    common = {
        "family": "nwc",
        "dtype": "bfloat16",
        "shape": [2, 768, 4096],
        "width": 4,
        "silu": True,
        "directions": ["forward", "backward"],
        "source": "frontend common workload",
    }
    if not any(key(case, "forward") == key(common, "forward") for case in cases):
        cases.append(common)
    jobs = [(case, direction) for case in cases for direction in (["both"] if case["family"] == "b2b" else case["directions"])]
    records = []
    for index, (case, direction) in enumerate(jobs):
        target = args.output / f"{index:03d}.json"
        entry = {"case": case, "direction": direction}
        cached = cache.get(key(case, direction))
        if cached:
            origin, data = cached
            target.write_text(json.dumps(data, indent=2) + "\n")
            entry.update(status="pass", origin=str(origin), reused=True)
        else:
            target.unlink(missing_ok=True)
            shape = ",".join(map(str, case["shape"]))
            if direction == "both":
                command = [
                    sys.executable,
                    str(bench / "probe_b2b.py"),
                    "--shape",
                    shape,
                    "--dtype",
                    case["dtype"],
                    "--projection-width",
                    str(case["width"]),
                    "--mixer-width",
                    str(case["mixer_width"]),
                    "--timing",
                ]
            else:
                script = "tune_bwd.py" if direction == "backward" else "tune.py"
                command = [
                    sys.executable,
                    str(bench / script),
                    "--shape",
                    shape,
                    "--dtype",
                    case["dtype"],
                    "--width",
                    str(case["width"]),
                    "--layout",
                    case["family"],
                    "--output",
                    str(target),
                ]
                command += ["--rows", "4" if case["family"] == "ncw" else "32"] if direction == "backward" else ["--schedules", "auto"]
                if not case["silu"]:
                    command.append("--identity")
                if case["family"] == "nwc" and case["dtype"] == "bfloat16" and case["width"] == 4 and case["silu"]:
                    command.append("--frontend")
            command += ["--backend-library", args.backend_library]
            try:
                proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired as e:
                proc = subprocess.CompletedProcess(command, 124, stdout=str(e.stdout or ""), stderr="Timed out after 900 seconds\n" + str(e.stderr or ""))
            log = proc.stdout + proc.stderr
            (args.output / f"{index:03d}.log").write_text(log)
            entry.update(
                command=command,
                exit_code=proc.returncode,
                status=("pass" if proc.returncode == 0 else ("backend_unsupported" if "returned 3010" in log or "status 3010" in log else "failed")),
            )
            if direction == "both" and proc.returncode == 0:
                data = json.loads(next(line for line in reversed(proc.stdout.splitlines()) if line.startswith("{")))
                target.write_text(json.dumps(data, indent=2) + "\n")
            else:
                data = json.loads(target.read_text()) if target.exists() else None
        if data:
            values = data.get("results", data.get("timing", {}))
            entry["median_us"] = {name: item["median_us"] for name, item in values.items()}
        records.append(entry)
        (args.output / "summary.json").write_text(
            json.dumps(
                {"gpu": gpu, "cc": torch.cuda.get_device_capability(), "manifest": str(args.manifest), "kernel_sha256": expected, "results": records}, indent=2
            )
            + "\n"
        )
        print(
            f'{index+1}/{len(jobs)} {case["family"]} {direction} {case["shape"]} {case["dtype"]} K{case["width"]}: {entry["status"]}'
            + (" reused" if cached else ""),
            flush=True,
        )
    sys.exit(any(r["status"] == "failed" for r in records))


if __name__ == "__main__":
    main()
