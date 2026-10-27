# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay every recorded regression, preserving compile costs and raw timings."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def worker(args):
    import torch
    import cutlass
    from cutlass import cute
    import tune
    from backend import Backend

    source = json.loads(args.evidence.read_text())
    config = source["args"]
    args.output.mkdir(parents=True, exist_ok=True)
    compile_records = []
    dispatch_records = []
    original_compile = cute.compile

    def timed_compile(fn, *values, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_compile(fn, *values, **kwargs)
        elapsed = time.perf_counter() - start
        compile_records.append({"function": getattr(fn, "__qualname__", type(fn).__name__), "seconds": elapsed})
        return result

    cute.compile = timed_compile
    for direction in ("forward", "backward"):
        original = getattr(Backend, direction)

        def timed_dispatch(self, *values, _original=original, _direction=direction, **kwargs):
            if any(r["direction"] == _direction for r in dispatch_records):
                return _original(self, *values, **kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = _original(self, *values, **kwargs)
            torch.cuda.synchronize()
            dispatch_records.append({"direction": _direction, "first_dispatch_seconds": time.perf_counter() - start})
            return result

        setattr(Backend, direction, timed_dispatch)

    original_measure = tune.measure

    def measured(functions, **kwargs):
        # This range is also usable with nsys; all compilation precedes it.
        if args.profile:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for name, fn in functions.items():
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(name)
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        return original_measure(functions, **kwargs)

    tune.measure = measured
    b2b = "projection_width" in config
    backward = "rows" in config
    script = "probe_b2b.py" if b2b else ("tune_bwd.py" if backward else "tune.py")
    command = [script]
    config = dict(config)
    if not b2b:
        config["output"] = str(args.output / "timing.json")
        if args.sweep:
            if backward:
                config["rows"] = "1,2,4,8" if config["layout"] == "ncw" else "8,16,32,64,128"
            else:
                config["schedules"] = "1:4,1:8,2:4,2:8,4:4,4:8,4:16,4:32,8:4,8:8,8:16,8:32"
                if config["layout"] == "ncw":
                    config["schedules"] = "1:4,1:8,1:16"
                    if int(config["shape"].split(",")[-1]) % 8 == 0:
                        config["schedules"] += ",8:8,8:16"
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                command.append("--" + key.replace("_", "-"))
        elif value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    sys.argv = command
    try:
        runpy.run_path(str(Path(__file__).with_name(script)), run_name="__main__")
    finally:
        (args.output / "compilation.json").write_text(
            json.dumps(
                {
                    "evidence": str(args.evidence),
                    "command": command,
                    "gpu": torch.cuda.get_device_name(),
                    "cc": torch.cuda.get_device_capability(),
                    "dsl": cutlass.__version__,
                    "compiles": compile_records,
                    "backend_first_dispatch": dispatch_records,
                    "compile_measurement": "wall clock around cute.compile, imports and CUDA context initialized; no persistent FE cache",
                    "backend_measurement": "first dispatch plus synchronization and any NVRTC/module initialization, not pure compiler time",
                },
                indent=2,
            )
            + "\n"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evidence", type=Path)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    if args.evidence:
        worker(args)
        return
    import torch

    cc = "sm%d%d" % torch.cuda.get_device_capability()
    rows = [r for r in csv.DictReader(args.source_csv.open()) if r["cc"] == cc and float(r["candidate_us"]) > float(r["backend_us"])]
    unique = list(dict.fromkeys(r["evidence"] for r in rows))
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, evidence in enumerate(unique):
        folder = args.output / (Path(evidence).stem)
        command = [sys.executable, __file__, "--evidence", evidence, "--output", str(folder)]
        if args.sweep:
            command.append("--sweep")
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "run.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        record = {"evidence": evidence, "returncode": result.returncode, "folder": str(folder)}
        if "projection_width" in json.loads(Path(evidence).read_text())["args"] and result.returncode == 0:
            data = json.loads(next(line for line in reversed((folder / "run.log").read_text().splitlines()) if line.startswith("{")))
            (folder / "timing.json").write_text(json.dumps(data, indent=2) + "\n")
        records.append(record)
        (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"{index+1}/{len(unique)} {cc} {Path(evidence).name}: {result.returncode}", flush=True)
    if any(r["returncode"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
