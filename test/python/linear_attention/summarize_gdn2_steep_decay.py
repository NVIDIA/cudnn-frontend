# SPDX-FileCopyrightText: Copyright (c) 2026 Alfred Wong
# SPDX-License-Identifier: Apache-2.0

"""Summarize all ABBA arms; refuse incomplete or mismatched runs."""

import argparse
import json
import math
import statistics
from pathlib import Path


def require(condition, message):
    """Reject invalid benchmark records, including under optimized Python."""
    if not condition:
        raise ValueError(message)


parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
args = parser.parse_args()
labels = ("base-A1", "patch-B1", "patch-B2", "base-A2")
runs = {label: json.loads((args.results / f"{label}.json").read_text()) for label in labels}
require(len({r["script_sha256"] for r in runs.values()}) == 1, "Benchmark script hashes differ")
for field in ("torch", "cuda", "dsl", "cudnn_backend", "gpu", "capability", "seed"):
    require(len({json.dumps(r[field], allow_nan=False) for r in runs.values()}) == 1, f"Environment differs: {field}")
for a, b in (("base-A1", "base-A2"), ("patch-B1", "patch-B2")):
    require(runs[a]["helper_sha256"] == runs[b]["helper_sha256"], f"Helper hashes differ: {a}, {b}")
require(runs["base-A1"]["helper_sha256"] != runs["patch-B1"]["helper_sha256"], "Base and patch helpers are identical")
for label, run in runs.items():
    require(run["label"] == label, f"Unexpected label in {label}")
    require(
        [(r["tokens"], r["steep_channels"]) for r in run["results"]] == [(t, c) for t in (16, 256) for c in (0, 1, 64)],
        f"Incomplete or reordered cases in {label}",
    )
    require(all(r["correct_within_one_percent"] is True for r in run["results"]), f"FP64 check failed in {label}")
    for case in run["results"]:
        errors = case["errors_fp64"]
        require(set(errors) == {"output", "q", "k", "v", "g", "b", "w"}, f"Incomplete FP64 checks in {label}")
        for name, error in errors.items():
            relative = error["relative_frobenius"]
            require(math.isfinite(relative) and 0 <= relative <= 0.01, f"Invalid FP64 relative error for {name} in {label}")
            absolute = error["max_abs"]
            require(math.isfinite(absolute) and absolute >= 0, f"Invalid FP64 absolute error for {name} in {label}")
rows = []
for index in range(6):
    cases = {label: run["results"][index] for label, run in runs.items()}
    first = cases[labels[0]]
    row = {"tokens": first["tokens"], "steep_channels": first["steep_channels"]}
    for mode in ("eager", "gpu_busy"):
        medians = {label: r[mode]["median_ms"] for label, r in cases.items()}
        for label, median in medians.items():
            require(math.isfinite(median) and median > 0, f"{mode} median_ms must be finite and positive in {label}, case {index}")
        base = statistics.mean(medians[label] for label in ("base-A1", "base-A2"))
        patch = statistics.mean(medians[label] for label in ("patch-B1", "patch-B2"))
        speedup = base / patch
        reduction = 100 * (1 - patch / base)
        require(all(math.isfinite(value) for value in (base, patch, speedup, reduction)), f"Non-finite derived {mode} measurement in case {index}")
        row[mode] = {"base_ms": base, "patch_ms": patch, "speedup": speedup, "reduction_percent": reduction, "arm_medians_ms": medians}
    row["max_fp64_relative_error"] = max(e["relative_frobenius"] for r in cases.values() for e in r["errors_fp64"].values())
    row["backward_scheduler_flags"] = {
        label: [s["schedulers"]["scheduler_bwd"][1] for s in r["scheduler_records"] if s["plan"] == "CompiledGdn2Bwd"] for label, r in cases.items()
    }
    for flags in row["backward_scheduler_flags"].values():
        require(flags and all(flag > 1 if first["steep_channels"] else flag == 0 for flag in flags), f"Unexpected backward scheduler flags in case {index}")
    rows.append(row)
summary = {"results": rows, "helper_hashes": {label: r["helper_sha256"] for label, r in runs.items()}, "script_hash": runs[labels[0]]["script_sha256"]}
serialized = json.dumps(summary, indent=2, allow_nan=False)
(args.results / "summary.json").write_text(serialized + "\n")
print(serialized)
