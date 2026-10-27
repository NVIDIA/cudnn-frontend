# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare every fixed workload with identical output addresses and exact data bits."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


def main():
    """Run fixed-source comparisons with shared output addresses and retain every job result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend-library", required=True)
    parser.add_argument("--accumulate", action="store_true", help="Run all backward directions without parameter clearing")
    parser.add_argument("--supplement", action="store_true", help="Run the additional original tuning-grid workloads")
    parser.add_argument("--kernel-dir", type=Path, help="Frozen candidate source directory; defaults to this worktree")
    args = parser.parse_args()
    bench = Path(__file__).resolve().parent
    kernel_dir = args.kernel_dir.resolve() if args.kernel_dir else bench.parents[1] / "python/cudnn/causal_conv1d_se"
    names = ("kernel.py", "backward_kernel.py", "b2b_kernel.py")
    hashes = {name: hashlib.sha256((kernel_dir / name).read_bytes()).hexdigest() for name in names}
    if args.supplement:
        cases = json.loads((bench / "supplemental_workloads.json").read_text())["cases"]
    else:
        cases = json.loads((bench / "workloads.json").read_text())["cases"]
        cases.append(
            dict(
                family="nwc",
                dtype="bfloat16",
                shape=[2, 768, 4096],
                width=4,
                silu=True,
                directions=["forward", "backward"],
                source="frontend common workload",
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(case, direction) for case in cases for direction in (["both"] if case["family"] == "b2b" else case["directions"])]
    records = []
    for index, (case, direction) in enumerate(jobs):
        b2b = case["family"] == "b2b"
        for selected in (["forward", "backward"] if b2b else [direction]):
            if args.accumulate and selected != "backward":
                continue
            settings = dict(shape=",".join(map(str, case["shape"])), dtype=case["dtype"], backend_library=args.backend_library)
            if b2b:
                settings.update(projection_width=case["width"], mixer_width=case["mixer_width"])
            else:
                settings.update(layout=case["family"], width=case["width"], identity=not case["silu"])
                settings.update({"rows": "4" if case["family"] == "ncw" else "32"} if selected == "backward" else {"schedules": "auto"})
            stem = f"{index:03d}-{selected}"
            config, target = args.output / (stem + ".case.json"), args.output / (stem + ".json")
            config.write_text(json.dumps({"args": settings}, indent=2) + "\n")
            name = "b2b_kernel.py" if b2b else ("backward_kernel.py" if selected == "backward" else "kernel.py")
            command = [
                sys.executable,
                str(bench / "compare_variants.py"),
                "--case",
                str(config),
                "--direction",
                selected,
                "--variant",
                f"candidate={kernel_dir / name}",
                "--shared-output-timing",
                "--output",
                str(target),
            ]
            if args.accumulate:
                command.append("--accumulate")
            if not args.accumulate and not b2b and case["family"] == "nwc" and case["dtype"] == "bfloat16" and case["width"] == 4 and case["silu"]:
                command.append("--frontend")
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout or ""
                stderr = error.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                result = subprocess.CompletedProcess(command, 124, stdout=stdout, stderr="Timed out after 900 seconds\n" + stderr)
            log = result.stdout + result.stderr
            (args.output / (stem + ".log")).write_text(log)
            unsupported = result.returncode != 124 and ("returned 3010" in log or "status 3010" in log or "SHARED_MEMORY_INSUFFICIENT" in log)
            status = "pass" if result.returncode == 0 else ("backend_unsupported" if unsupported else "failed")
            records.append(dict(case=case, direction=selected, status=status, returncode=result.returncode, evidence=str(target.resolve()), command=command))
            current = {name: hashlib.sha256((kernel_dir / name).read_bytes()).hexdigest() for name in names}
            if current != hashes:
                raise RuntimeError("Candidate source changed during the performance sweep")
            (args.output / "summary.json").write_text(
                json.dumps(
                    dict(
                        gpu=torch.cuda.get_device_name(),
                        cc=torch.cuda.get_device_capability(),
                        kernel_sha256=hashes,
                        suite="supplement" if args.supplement else "main",
                        protocol="paired_shared_outputs_additive_no_clear_v1" if args.accumulate else "paired_shared_output_addresses_v1",
                        results=records,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(stem, case["family"], case["shape"], case["dtype"], status, flush=True)
    raise SystemExit(any(row["status"] == "failed" for row in records))


if __name__ == "__main__":
    main()
