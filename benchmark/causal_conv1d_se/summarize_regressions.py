# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Account for every slower baseline case without discarding small regressions."""

import argparse
import collections
import csv
import json
from pathlib import Path
import random
import statistics


def interval(candidate, backend):
    rng = random.Random(20261026)
    n = len(candidate)
    assert n == len(backend)
    samples = []
    for _ in range(5000):
        indices = [rng.randrange(n) for _ in range(n)]
        samples.append(statistics.median(candidate[i] for i in indices) / statistics.median(backend[i] for i in indices))
    samples.sort()
    return samples[125], samples[4874]


def category(row):
    if row["family"] == "b2b":
        return "fused_forward_halo_reuse" if row["direction"] == "forward" else "fused_backward_recomputation"
    if row["dtype"] == "float64":
        return "short_fp64_tile_and_empty_halo"
    if row["family"] == "nwc":
        return "nwh_forward_register_tile" if row["direction"] == "forward" else "nwh_backward_pipeline_and_reduction"
    if row["direction"] == "forward":
        return "ncw_forward_tile"
    if int(row["L"]) % 8:
        return "ncw_backward_cooperative_tail"
    return "ncw_backward_sequence_partition"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--replay", action="append", required=True, help="smNN=directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = dict(item.split("=", 1) for item in args.replay)
    rows = [r for r in csv.DictReader(args.baseline.open()) if float(r["candidate_us"]) > float(r["backend_us"])]
    records, compile_records = [], []
    for row in rows:
        folder = Path(replay[row["cc"]]) / Path(row["evidence"]).stem
        data = json.loads((folder / "timing.json").read_text())
        values = data.get("results", data.get("timing"))
        cname = row["schedule"]
        bname = "backend_" + ("fwd" if row["direction"] == "forward" else "bwd") if row["family"] == "b2b" else "backend"
        c, b = values[cname], values[bname]
        assert row["kernel_sha256"] in data["sha256"].values(), (row, data["sha256"])
        low, high = interval(c["samples_us"], b["samples_us"])
        ratio = c["median_us"] / b["median_us"]
        record = dict(row)
        record.update(
            category=category(row),
            replay_candidate_us=c["median_us"],
            replay_backend_us=b["median_us"],
            replay_slowdown_percent=100 * (ratio - 1),
            ratio_ci95_low=low,
            ratio_ci95_high=high,
            status="slower" if low > 1 else ("faster" if high < 1 else "unresolved_near_parity"),
            replay_evidence=str(folder / "timing.json"),
            replay_gpu=data["gpu"],
        )
        records.append(record)
        compile_data = json.loads((folder / "compilation.json").read_text())
        compile_records.append({"cc": row["cc"], "case": Path(row["evidence"]).stem, **compile_data})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "regressions.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "original_regressions": len(records),
        "replayed": len(records),
        "status": dict(collections.Counter(r["status"] for r in records)),
        "categories": dict(collections.Counter(r["category"] for r in records)),
        "per_architecture": {cc: dict(collections.Counter(r["status"] for r in records if r["cc"] == cc)) for cc in replay},
        "method": "Paired-round bootstrap, 5000 resamples, 95% interval of the median latency ratio. Intervals quantify within-run sampling only; they do not remove between-run or cache-layout effects.",
        "greater_than_five_percent": sum(r["replay_slowdown_percent"] > 5 for r in records),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "compilation.json").write_text(json.dumps(compile_records, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
