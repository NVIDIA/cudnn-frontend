# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed backward and back-to-back workloads; preserve failures alongside timings."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    """Run the fixed backward and fused workload grid, retaining subprocess logs and timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend-library", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    bench = Path(__file__).parent
    for layout in ("ncw", "nwc"):
        for shape, k, silu in [
            ("64,768,256", 7, False),
            ("2,768,4096", 4, True),
            ("2,1024,8192", 3, True),
            ("2,4096,510", 4, False),
            ("2,768,256", 32, False),
            ("2,768,256", 128, False),
        ]:
            command = [
                sys.executable,
                str(bench / "tune_bwd.py"),
                "--layout",
                layout,
                "--shape",
                shape,
                "--width",
                str(k),
                "--rows",
                "4" if layout == "ncw" else "32",
                "--output",
                str(args.output / f"{len(cases):03d}.json"),
            ]
            if not silu:
                command.append("--identity")
            if layout == "nwc" and k == 4 and silu:
                command.append("--frontend")
            cases.append((f"{layout} {shape} K{k} silu={silu}", command))
    for shape, kp, km in [("2,1024,8192", 3, 3), ("2,768,539", 3, 16), ("2,768,256", 4, 7), ("2,64,256", 32, 256)]:
        command = [sys.executable, str(bench / "probe_b2b.py"), "--shape", shape, "--projection-width", str(kp), "--mixer-width", str(km), "--timing"]
        cases.append((f"b2b {shape} Kp={kp} Km={km}", command))
    records = []
    for index, (label, command) in enumerate(cases):
        command.extend(["--backend-library", args.backend_library])
        path = args.output / f"{index:03d}.json"
        path.unlink(missing_ok=True)
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired as e:
            proc = subprocess.CompletedProcess(command, 124, stdout=str(e.stdout or ""), stderr="Timed out after 900 seconds\n" + str(e.stderr or ""))
        (args.output / f"{index:03d}.log").write_text(proc.stdout + proc.stderr)
        entry = {"case": label, "command": command, "status": proc.returncode}
        path = args.output / f"{index:03d}.json"
        if path.exists():
            data = json.loads(path.read_text())
            entry["median_us"] = {name: item["median_us"] for name, item in data["results"].items()}
        for line in proc.stdout.splitlines():
            if line.startswith("{") and "backend_max_abs" in line:
                data = json.loads(line)
                path.write_text(json.dumps(data, indent=2) + "\n")
                entry["median_us"] = {name: item["median_us"] for name, item in data["timing"].items()}
        records.append(entry)
        (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"{index+1}/{len(cases)} {label} " + json.dumps(entry.get("median_us", entry)), flush=True)
    sys.exit(any(r["status"] for r in records))


if __name__ == "__main__":
    main()
