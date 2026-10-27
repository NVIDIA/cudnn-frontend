# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated correctness cases: preserve each command, result, and failure."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend-library", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    files = list((root / "python/cudnn/causal_conv1d_se").glob("*.py")) + list(Path(__file__).parent.glob("*.py"))
    manifest = {
        "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.executable,
        "backend": args.backend_library,
        "results": [],
    }
    cases = []
    for layout in ("ncw", "nwc"):
        for dtype in ("bfloat16", "float16", "float32", "float64"):
            for backward in (False, True):
                cases.append((layout, dtype, backward, False, "2,7,65", 4))
        for shape, width in [("2,33,257", 3), ("1,3,2", 8), ("2,8,33", 32), ("1,7,129", 128)]:
            cases.append((layout, "float32", False, True, shape, width))
            cases.append((layout, "float32", True, True, shape, width))
    for index, (layout, dtype, backward, identity, shape, width) in enumerate(cases):
        command = [
            sys.executable,
            str(Path(__file__).with_name("probe.py")),
            "--layout",
            layout,
            "--dtype",
            dtype,
            "--shape",
            shape,
            "--width",
            str(width),
            "--backend-library",
            args.backend_library,
        ]
        if backward:
            command.append("--backward")
        if identity:
            command.append("--identity")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as error:
            result = subprocess.CompletedProcess(command, 124, stdout=str(error.stdout or ""), stderr="Timed out after 300 seconds\n" + str(error.stderr or ""))
        (args.output / f"{index:03d}.log").write_text(result.stdout + result.stderr)
        entry = {"command": command, "exit_code": result.returncode, "log": f"{index:03d}.log"}
        for line in result.stdout.splitlines():
            if line.startswith("{"):
                entry["measurement"] = json.loads(line)
        manifest["results"].append(entry)
        (args.output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f'{index+1}/{len(cases)} {layout} {dtype} {"bwd" if backward else "fwd"} K{width}: {result.returncode}', flush=True)
    failures = sum(r["exit_code"] != 0 for r in manifest["results"])
    print(f"{len(cases)-failures} passed, {failures} failed", flush=True)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
