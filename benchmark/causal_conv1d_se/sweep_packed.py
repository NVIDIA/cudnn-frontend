# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run every primary packed workload in fresh processes and retain missing rows.

The parent imports no GPU libraries. Environment setup belongs to the runner;
--runner-json accepts an argv array, never a shell command. Use a new output
folder for each frozen source revision. One repetition is diagnostic only.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def hashes(paths):
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    bench = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=bench / "packed_workloads.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--graph-calls", type=int, default=20)
    parser.add_argument("--weight-layout", choices=("HK", "KH"), default="KH")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--runner-json", help="JSON argv prefix; defaults to this Python and compare_packed.py")
    parser.add_argument("--dry-run", action="store_true", help="Write the complete expected lane without importing GPU libraries")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["primary_cases"]
    if args.arch not in manifest["architectures"]:
        parser.error("architecture is outside the declared matrix")
    if args.repetitions < 1 or args.rounds < 1 or args.graph_calls < 1:
        parser.error("repetitions, rounds, and graph calls must be positive")
    if len(cases) != manifest["primary_case_count"] or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("manifest primary case count or IDs are inconsistent")
    runner = json.loads(args.runner_json) if args.runner_json else [sys.executable, str(bench / "compare_packed.py")]
    if not isinstance(runner, list) or not runner or any(not isinstance(a, str) or not a for a in runner):
        parser.error("runner-json must be a nonempty array of nonempty strings")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty; existing evidence is never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    root = bench.parents[1]
    paths = [Path(__file__), bench / "compare_packed.py", bench / "timing.py", args.manifest]
    paths += sorted((root / "python/cudnn/causal_conv1d_se").glob("*.py"))
    paths += sorted((root / "python/cudnn/causal_conv1d_bulk_sm100").glob("*.py"))
    frozen = hashes(paths)
    records = [dict(case=case, repetition=rep, status="unverified") for rep in range(args.repetitions) for case in cases]
    report = dict(
        requested_arch=args.arch,
        expected_primary_cases=len(cases),
        expected_matrix_case_arch_pairs=len(cases) * len(manifest["architectures"]),
        expected_processes=len(records),
        repetitions=args.repetitions,
        source=frozen,
        qualification="diagnostic only" if args.repetitions < 3 else "three or more fresh processes; inspect every result",
        options=dict(rounds=args.rounds, graph_calls=args.graph_calls, weight_layout=args.weight_layout, seed=args.seed),
        runner=runner,
        records=records,
    )
    summary = args.output / "summary.json"

    def checkpoint():
        report["counts"] = dict(Counter(r["status"] for r in records))
        save(summary, report)

    checkpoint()
    if args.dry_run:
        print(json.dumps(dict(expected_cases=len(cases), expected_processes=len(records), counts=report["counts"])))
        return
    identity = None
    for index, record in enumerate(records):
        if hashes(paths) != frozen:
            report["stopped_reason"] = "source changed; remaining rows are unverified"
            checkpoint()
            raise RuntimeError(report["stopped_reason"])
        case = record["case"]
        stem = f"{index:04d}-r{record['repetition']}-{case['id']}"
        target, log = args.output / (stem + ".json"), args.output / (stem + ".log")
        command = runner + [
            "--case",
            case["id"],
            "--manifest",
            str(args.manifest.resolve()),
            "--output",
            str(target.resolve()),
            "--rounds",
            str(args.rounds),
            "--graph-calls",
            str(args.graph_calls),
            "--weight-layout",
            args.weight_layout,
            "--seed",
            str(args.seed),
        ]
        record.update(status="running", command=command, evidence=str(target.resolve()), log=str(log.resolve()))
        checkpoint()
        with log.open("w") as stream:
            child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        record["returncode"] = child.returncode
        try:
            data = json.loads(target.read_text()) if target.exists() else None
        except (OSError, json.JSONDecodeError) as error:
            data = None
            record["report_error"] = str(error)
        if data is None or child.returncode != 0 or data.get("status") != "pass":
            record.update(status="failed", error=None if data is None else data.get("error"))
            if data and "foreign GPU work" in data.get("error", ""):
                report["stopped_reason"] = "foreign GPU work; remaining rows are unverified"
                checkpoint()
                raise RuntimeError(report["stopped_reason"])
        else:
            actual_arch = data["arch"][0] * 10 + data["arch"][1]
            current_identity = {name: data[name] for name in ("gpu_uuid", "arch", "compiler", "dsl", "torch", "backend", "driver", "compiled_arch", "source")}
            if actual_arch != args.arch or data["result"]["case"] != case or (identity is not None and current_identity != identity):
                record["status"] = "invalid_environment"
                report["stopped_reason"] = "GPU, compiler, source, or case identity changed"
                checkpoint()
                raise RuntimeError(report["stopped_reason"])
            identity = current_identity
            report["identity"] = identity
            result = data["result"]
            cu_ok = result["cuhyena_speedup"] >= manifest["requirements"]["cuhyena_speedup_min"]
            fe_ok = result["frontend_speedup"] is None or result["frontend_speedup"] > manifest["requirements"]["frontend_speedup_min_exclusive"]
            status = (
                ("measured_cuhyena_target_met" if result["frontend_speedup"] is None else "measured_both_targets_met")
                if cu_ok and fe_ok
                else "measured_below_target"
            )
            record.update(
                status=status,
                cuhyena_target_met=cu_ok,
                frontend_target_met=None if result["frontend_speedup"] is None else fe_ok,
                cuhyena_speedup=result["cuhyena_speedup"],
                frontend_speedup=result["frontend_speedup"],
                frontend=result["frontend"],
                generated_cpu_fp32_sha256=result["generated_cpu_fp32_sha256"],
            )
        checkpoint()
        print(f"{index + 1}/{len(records)} {case['id']} repetition={record['repetition']} {record['status']}", flush=True)
    if hashes(paths) != frozen:
        report["stopped_reason"] = "source changed before completion"
        checkpoint()
        raise RuntimeError(report["stopped_reason"])
    report["completed"] = True
    checkpoint()
    raise SystemExit(any(r["status"] == "failed" for r in records))


if __name__ == "__main__":
    main()
