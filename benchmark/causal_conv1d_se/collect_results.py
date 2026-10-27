# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export fixed-policy comparisons without selecting the fastest repeated result."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def spread(samples):
    """Return the lower and upper quartile samples from the measured timings."""
    ordered = sorted(samples)
    return ordered[len(ordered) // 4], ordered[(3 * len(ordered)) // 4]


def records(path, data, expected):
    """Yield comparison rows only for matching kernel hashes and the fixed schedule policy."""
    args = data.get("args", {})
    if "gpu" not in data or "shape" not in args:
        return
    b2b = "projection_width" in args
    direction = "backward" if "rows" in args else "forward"
    source = "b2b_kernel.py" if b2b else ("backward_kernel.py" if direction == "backward" else "kernel.py")
    digests = {Path(k).name: v for k, v in data.get("sha256", {}).items()}
    # Earlier B2B results did not record source hashes: retain those files separately.
    if digests.get(source) != expected[source]:
        return
    if not b2b and direction == "forward" and args.get("schedules") != "auto":
        return
    if not b2b and direction == "backward" and str(args.get("rows")) != ("4" if args["layout"] == "ncw" else "32"):
        return
    timing = data.get("timing", {}) if b2b else data.get("results", {})
    if not timing:
        return
    candidates = [k for k in timing if k not in ("backend", "frontend", "backend_fwd", "backend_bwd")]
    if not b2b and len(candidates) != 1:
        return
    for suffix in (["fwd", "bwd"] if b2b else [None]):
        name = f"candidate_{suffix}" if b2b else candidates[0]
        backend_name = f"backend_{suffix}" if b2b else "backend"
        candidate = timing[name]
        backend = timing[backend_name]
        frontend = timing.get("frontend")
        shape = tuple(map(int, args["shape"].split(",")))
        row = {
            "gpu": data["gpu"],
            "cc": "sm" + "".join(map(str, data["cc"])),
            "family": "b2b" if b2b else args["layout"],
            "direction": ("forward" if suffix == "fwd" else "backward") if b2b else direction,
            "dtype": args["dtype"],
            "B": shape[0],
            "D": shape[1],
            "L": shape[2],
            "K": args["projection_width"] if b2b else args["width"],
            "Km": args.get("mixer_width", ""),
            "silu": False if b2b else not args["identity"],
            "cache": "cold" if args.get("cold", False) else "warm",
            "schedule": name,
            "candidate_us": candidate["median_us"],
            "backend_us": backend["median_us"],
            "backend_speedup": backend["median_us"] / candidate["median_us"],
            "frontend_us": frontend["median_us"] if frontend else "",
            "frontend_speedup": frontend["median_us"] / candidate["median_us"] if frontend else "",
            "candidate_p25_us": spread(candidate["samples_us"])[0],
            "candidate_p75_us": spread(candidate["samples_us"])[1],
            "kernel_sha256": expected[source],
            "evidence": str(path.resolve()),
        }
        yield row


def identity(row):
    """Build the workload key used to retain the first supplied matching result."""
    return tuple(row[k] for k in ("gpu", "cc", "family", "direction", "dtype", "B", "D", "L", "K", "Km", "silu", "cache"))


def main():
    """Export matching performance comparisons and preserve failed or unsupported jobs separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Result directory or JSON file; first matching case wins")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kernel_dir = Path(__file__).resolve().parents[2] / "python/cudnn/causal_conv1d_se"
    expected = {name: hashlib.sha256((kernel_dir / name).read_bytes()).hexdigest() for name in ("kernel.py", "backward_kernel.py", "b2b_kernel.py")}
    selected = {}
    failures = []
    for item in args.input:
        paths = [item] if item.is_file() else sorted(item.glob("*.json"))
        for path in paths:
            data = json.loads(path.read_text())
            if path.name == "summary.json":
                entries = data["results"] if isinstance(data, dict) else data
                for index, entry in enumerate(entries):
                    if entry["status"] not in ("pass", 0):
                        failures.append({"summary": str(path.resolve()), "index": index, "record": entry})
                continue
            if not isinstance(data, dict):
                continue
            for row in records(path, data, expected):
                selected.setdefault(identity(row), row)
    rows = list(selected.values())
    args.output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.output / "comparisons.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "exceptions.json").write_text(json.dumps(failures, indent=2) + "\n")
    groups = sorted({(r["cc"], r["gpu"]) for r in rows})
    lines = [
        "Fixed-policy performance evidence. Speedup is baseline time divided by candidate time; values below one favor the baseline.",
        "",
        "The ±5% band below is a reporting band, not a statistical confidence interval or an acceptance waiver. Every comparison checks numerical results before timing. Repeated cases use the first supplied input, never the fastest result. Rows with missing or mismatched kernel hashes are excluded from this table.",
        "",
        "| GPU | Warm comparisons | Candidate >5% faster vs backend | Within ±5% time | Backend regressions >5% time | Frontend comparisons | Frontend regressions >5% time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cc, gpu in groups:
        group = [r for r in rows if (r["cc"], r["gpu"]) == (cc, gpu) and r["cache"] == "warm"]
        win = sum(r["backend_speedup"] > 1 / 0.95 for r in group)
        loss = sum(r["backend_speedup"] < 1 / 1.05 for r in group)
        frontend = [r for r in group if r["frontend_us"] != ""]
        fe_loss = sum(r["frontend_speedup"] < 1 / 1.05 for r in frontend)
        lines.append(f"| {gpu} ({cc}) | {len(group)} | {win} | {len(group)-win-loss} | {loss} | {len(frontend)} | {fe_loss} |")
    lines += [
        "",
        "The CSV contains every selected per-case median, interquartile range, source digest, and path to the raw samples. `exceptions.json` preserves failed jobs and backend rejections. A backend rejection leaves that comparison unmeasured. Cold-cache cases remain separate in the CSV.",
        "",
    ]
    (args.output / "performance.md").write_text("\n".join(lines))
    print(json.dumps({"comparisons": len(rows), "exceptions": len(failures), "kernel_sha256": expected}))


if __name__ == "__main__":
    main()
