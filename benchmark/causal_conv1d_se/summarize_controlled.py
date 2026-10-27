# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect a complete fixed-source shared-output sweep, retaining every loss."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from summarize_regressions import interval


def key(row):
    """Build a common workload identity for matching original and repeated measurements."""
    return tuple(str(row[name]) for name in ("cc", "family", "direction", "dtype", "B", "D", "L", "K", "Km", "silu"))


def write_csv(path, rows):
    """Write nonempty records as a CSV using the first record field order."""
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Validate complete controlled sweeps and export timing, compilation, and regression summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--source-hashes", type=Path, required=True)
    parser.add_argument("--original-regressions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accumulate", action="store_true")
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()
    expected = json.loads(args.source_hashes.read_text())
    rows, gpu_rows, compile_rows = [], [], []
    for folder in args.run:
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["kernel_sha256"] == expected
        protocol = "paired_shared_outputs_additive_no_clear_v1" if args.accumulate else "paired_shared_output_addresses_v1"
        assert summary["protocol"] == protocol
        expected_count = (11 if args.accumulate else 32) if args.supplement else (38 if args.accumulate else 77)
        assert len(summary["results"]) == expected_count, (folder, len(summary["results"]))
        assert all(r["status"] in ("pass", "backend_unsupported") for r in summary["results"])
        group = []
        for result in summary["results"]:
            if result["status"] != "pass":
                continue
            path = Path(result["evidence"])
            data = json.loads(path.read_text())
            case = result["case"]
            variant = data["variants"]["candidate"]
            assert variant["sha256"] == expected[Path(variant["path"]).name]
            assert data["timing_output_addresses"] == "shared_backend_and_candidates"
            assert all(v.get("accepted", v.get("equal", False)) for trial in variant["checks"] for v in trial.values())
            pair = data["paired"]["candidate"]
            c, b = pair["candidate"], pair["backend"]
            ratio = c["median_us"] / b["median_us"]
            address = data["output_address_control"]
            address_ratio = address["backend_reference_outputs"]["median_us"] / address["backend_candidate_outputs"]["median_us"]
            row = dict(
                gpu=data["gpu"],
                cc="sm" + "".join(map(str, data["cc"])),
                family=case["family"],
                direction=result["direction"],
                dtype=case["dtype"],
                B=case["shape"][0],
                D=case["shape"][1],
                L=case["shape"][2],
                K=case["width"],
                Km=case.get("mixer_width", ""),
                silu=case.get("silu", False),
                candidate_us=c["median_us"],
                backend_us=b["median_us"],
                backend_speedup=1 / ratio,
                slowdown_percent=100 * (ratio - 1),
                backend_slower_median=ratio > 1,
                ratio_ci95_low="",
                ratio_ci95_high="",
                frontend_candidate_us="",
                frontend_us="",
                frontend_speedup="",
                frontend_slower_median=False,
                backend_output_address_ratio=address_ratio,
                kernel_sha256=variant["sha256"],
                evidence=str(path.resolve()),
            )
            if ratio > 1:
                row["ratio_ci95_low"], row["ratio_ci95_high"] = interval(c["samples_us"], b["samples_us"])
            if data["frontend_paired"]:
                fp = data["frontend_paired"]["candidate"]
                fc, f = fp["candidate"]["median_us"], fp["frontend"]["median_us"]
                row.update(frontend_candidate_us=fc, frontend_us=f, frontend_speedup=f / fc, frontend_slower_median=fc > f)
            group.append(row)
            compile_rows.append(
                dict(
                    gpu=row["gpu"],
                    cc=row["cc"],
                    family=row["family"],
                    direction=row["direction"],
                    dtype=row["dtype"],
                    seconds=variant["compile_seconds"],
                    evidence=row["evidence"],
                )
            )
        rows.extend(group)
        gpu_rows.append(
            dict(
                gpu=summary["gpu"],
                cc="sm" + "".join(map(str, summary["cc"])),
                comparisons=len(group),
                backend_unsupported=sum(r["status"] == "backend_unsupported" for r in summary["results"]),
                backend_slower_medians=sum(r["backend_slower_median"] for r in group),
                backend_speedup_geomean=math.exp(statistics.mean(math.log(r["backend_speedup"]) for r in group)),
                backend_speedup_min=min(r["backend_speedup"] for r in group),
                frontend_comparisons=sum(bool(r["frontend_us"]) for r in group),
                frontend_slower_medians=sum(r["frontend_slower_median"] for r in group),
                frontend_speedup_min=min((r["frontend_speedup"] for r in group if r["frontend_us"]), default=None),
                evidence=str(folder.resolve()),
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "comparisons.csv", rows)
    write_csv(args.output / "compilation.csv", compile_rows)
    (args.output / "summary.json").write_text(json.dumps(gpu_rows, indent=2) + "\n")
    if args.original_regressions:
        lookup = {key(row): row for row in rows}
        original = list(csv.DictReader(args.original_regressions.open()))
        ledger = []
        for row in original:
            if args.accumulate and row["direction"] != "backward":
                continue
            latest = lookup[key(row)]
            row.update(
                latest_candidate_us=latest["candidate_us"],
                latest_backend_us=latest["backend_us"],
                latest_slowdown_percent=latest["slowdown_percent"],
                latest_gpu=latest["gpu"],
                same_gpu_model=row["gpu"] == latest["gpu"],
                latest_evidence=latest["evidence"],
            )
            ledger.append(row)
        write_csv(args.output / "original-regression-ledger.csv", ledger)
    print(json.dumps(gpu_rows, indent=2))


if __name__ == "__main__":
    main()
