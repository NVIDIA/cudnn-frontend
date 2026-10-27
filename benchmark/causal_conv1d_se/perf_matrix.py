# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward workload grid derived from backend levels.layer and frontend benchmarks."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    """Run the forward workload grid with fixed or exploratory schedules and retain each result."""
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend-library", required=True)
    p.add_argument("--fixed", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for layout in ("ncw", "nwc"):
        for shape, k, silu in [
            ("64,768,256", 7, False),
            ("64,768,4096", 4, False),
            ("2,768,4096", 4, True),
            ("2,768,4096", 8, False),
            ("2,1024,8192", 3, True),
            ("2,4096,510", 4, False),
        ]:
            cases.append((layout, "bfloat16", shape, k, silu))
        for dtype in ("float16", "float32"):
            cases.append((layout, dtype, "64,768,256", 7, False))
        for k in (16, 32, 64, 128):
            cases.append((layout, "bfloat16", "2,768,256", k, False))
    cases.append(("ncw", "bfloat16", "2,768,256", 256, False))
    for shape in ("1,768,128", "1,4096,8192", "1,12288,16384", "8,12288,2048"):
        cases.append(("nwc", "bfloat16", shape, 4, True))
    records = []
    for index, (layout, dtype, shape, k, silu) in enumerate(cases):
        length = int(shape.split(",")[2])
        schedules = "2:8,4:8" if layout == "nwc" else ("1:8,8:8" if dtype in ("bfloat16", "float16") and length % 8 == 0 else "1:8")
        if args.fixed:
            schedules = "auto"
        output = args.output / f"{index:03d}.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("tune.py")),
            "--layout",
            layout,
            "--dtype",
            dtype,
            "--shape",
            shape,
            "--width",
            str(k),
            "--schedules",
            schedules,
            "--backend-library",
            args.backend_library,
            "--output",
            str(output),
        ]
        if not silu:
            command.append("--identity")
        if layout == "nwc" and dtype == "bfloat16" and k == 4 and silu:
            command.append("--frontend")
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
            (args.output / f"{index:03d}.log").write_text(proc.stdout + proc.stderr)
            status = proc.returncode
        except subprocess.TimeoutExpired as e:
            (args.output / f"{index:03d}.log").write_text("Timed out after 900 seconds\n" + str(e.stdout) + str(e.stderr))
            status = 124
        entry = {"case": [layout, dtype, shape, k, silu], "command": command, "status": status}
        if output.exists():
            data = json.loads(output.read_text())
            entry["median_us"] = {name: item["median_us"] for name, item in data["results"].items()}
        records.append(entry)
        (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"{index+1}/{len(cases)} " + json.dumps(entry.get("median_us", entry)), flush=True)
    sys.exit(any(r["status"] for r in records))


if __name__ == "__main__":
    main()
