# SPDX-FileCopyrightText: Copyright (c) 2026 Alfred Wong
# SPDX-License-Identifier: Apache-2.0

"""Summarize all ABBA arms; refuse incomplete or mismatched runs."""

import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
args = parser.parse_args()
labels = ("base-A1", "patch-B1", "patch-B2", "base-A2")
runs = {label: json.loads((args.results / f"{label}.json").read_text()) for label in labels}
assert len({r["script_sha256"] for r in runs.values()}) == 1
for field in ("torch", "cuda", "dsl", "cudnn_backend", "gpu", "capability", "seed"):
    assert len({json.dumps(r[field]) for r in runs.values()}) == 1, field
for a, b in (("base-A1", "base-A2"), ("patch-B1", "patch-B2")):
    assert runs[a]["helper_sha256"] == runs[b]["helper_sha256"]
assert runs["base-A1"]["helper_sha256"] != runs["patch-B1"]["helper_sha256"]
for label, run in runs.items():
    assert run["label"] == label
    assert [(r["tokens"], r["steep_channels"]) for r in run["results"]] == [(t, c) for t in (16, 256) for c in (0, 1, 64)]
    assert all(r["correct_within_one_percent"] for r in run["results"])
rows = []
for index in range(6):
    cases = {label: run["results"][index] for label, run in runs.items()}
    first = cases[labels[0]]
    assert all((r["tokens"], r["steep_channels"]) == (first["tokens"], first["steep_channels"]) for r in cases.values())
    row = {"tokens": first["tokens"], "steep_channels": first["steep_channels"]}
    for mode in ("eager", "gpu_busy"):
        medians = {label: r[mode]["median_ms"] for label, r in cases.items()}
        base = statistics.mean(medians[label] for label in ("base-A1", "base-A2"))
        patch = statistics.mean(medians[label] for label in ("patch-B1", "patch-B2"))
        row[mode] = {"base_ms": base, "patch_ms": patch, "speedup": base / patch, "reduction_percent": 100 * (1 - patch / base), "arm_medians_ms": medians}
    row["max_fp64_relative_error"] = max(e["relative_frobenius"] for r in cases.values() for e in r["errors_fp64"].values())
    row["backward_scheduler_flags"] = {
        label: [s["schedulers"]["scheduler_bwd"][1] for s in r["scheduler_records"] if s["plan"] == "CompiledGdn2Bwd"] for label, r in cases.items()
    }
    for flags in row["backward_scheduler_flags"].values():
        assert flags and all(flag > 1 if first["steep_channels"] else flag == 0 for flag in flags)
    rows.append(row)
summary = {"results": rows, "helper_hashes": {label: r["helper_sha256"] for label, r in runs.items()}, "script_hash": runs[labels[0]]["script_sha256"]}
(args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
