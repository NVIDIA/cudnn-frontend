# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export exact-comparison results without suppressing failures or missing cases."""

import argparse
import collections
import csv
import json
from pathlib import Path


def main():
    """Export per-output bitwise results and report missing cases, failures, and repeatability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[2] / "test/python/causal_conv1d/cutedsl/se_levels_cases.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected = {c["name"] for c in manifest["cases"]}
    rows, summaries = [], []
    for folder in args.input:
        results = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
        results = [r for r in results if isinstance(r, dict) and "case" in r and "status" in r]
        found = {r["case"]["name"] for r in results}
        if found - expected or len(results) != len(found):
            raise ValueError(f"Unexpected or duplicate cases in {folder}")
        hashes = {tuple(sorted(r["kernel_sha256"].items())) for r in results if "kernel_sha256" in r}
        if len(hashes) > 1:
            raise ValueError(f"Mixed kernel revisions in {folder}")
        summary = {
            "folder": str(folder),
            "gpu": results[0]["gpu"] if results else None,
            "cc": results[0]["cc"] if results else None,
            "expected": len(expected),
            "missing": sorted(expected - found),
            "statuses": dict(collections.Counter(r["status"] for r in results)),
            "kernel_sha256": dict(next(iter(hashes))) if hashes else None,
            "strict_bitwise_passes": sum(r.get("bitwise_match", False) for r in results),
            "exact_data_passes": sum(r.get("data_bitwise_match", r.get("bitwise_match", False)) for r in results),
            "accepted_atomic_tolerance_cases": sum(r.get("accepted", False) and not r.get("bitwise_match", False) for r in results),
            "contracts": sorted({r["acceptance_contract"] for r in results if "acceptance_contract" in r}),
            "backend_nonrepeatable_cases": [r["case"]["name"] for r in results if r.get("backend_repeatable") is False],
            "candidate_nonrepeatable_cases": [r["case"]["name"] for r in results if r.get("candidate_repeatable") is False],
        }
        summary["directions"] = {
            direction: dict(collections.Counter(r["status"] for r in results if r["case"]["direction"] == direction)) for direction in ("forward", "backward")
        }
        summaries.append(summary)
        for result in results:
            case = result["case"]
            base = {
                "gpu": result["gpu"],
                "cc": ".".join(map(str, result["cc"])),
                "case": case["name"],
                "family": case["family"],
                "direction": case["direction"],
                "dtype": case["dtype"],
                "B": case["shape"][0],
                "D": case["shape"][1],
                "L": case["shape"][2],
                "width": case["width"],
                "silu": case["silu"],
                "status": result["status"],
                "source": str(folder / (case["name"] + ".json")),
            }
            trials = result.get("trials", [])
            if not trials:
                rows.append(
                    dict(
                        base,
                        output="",
                        cross_mismatches="",
                        backend_repeat_mismatches="",
                        candidate_repeat_mismatches="",
                        parameter_failures="",
                        max_abs="",
                        max_tolerance_fraction="",
                        atol="",
                        rtol="",
                    )
                )
                continue
            for name in trials[0]["cross"]:
                counts = {
                    key: max((t[key][name]["mismatched_elements"] for t in trials if key in t), default=0)
                    for key in ("cross", "backend_repeat", "candidate_repeat")
                }
                rows.append(
                    dict(
                        base,
                        output=name,
                        cross_mismatches=counts["cross"],
                        backend_repeat_mismatches=counts["backend_repeat"],
                        candidate_repeat_mismatches=counts["candidate_repeat"],
                        parameter_failures=max((t.get("parameters", {}).get(name, {}).get("failed_elements", 0) for t in trials), default=0),
                        max_abs=max(t["cross"][name]["max_abs"] for t in trials),
                        max_tolerance_fraction=max((t.get("parameters", {}).get(name, {}).get("max_tolerance_fraction", 0) for t in trials), default=0),
                        atol=trials[0].get("parameters", {}).get(name, {}).get("atol", ""),
                        rtol=trials[0].get("parameters", {}).get(name, {}).get("rtol", ""),
                    )
                )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if rows:
        with (args.output / "outputs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for summary in summaries:
        print(summary["gpu"], summary["cc"], summary["statuses"], f"missing={len(summary['missing'])}")
    raise SystemExit(any(s["missing"] or s["statuses"].get("pass", 0) != len(expected) for s in summaries))


if __name__ == "__main__":
    main()
