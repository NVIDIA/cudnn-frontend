# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract the six non-FFT causal-conv routines from a pinned backend levels.layer."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROUTINES = {
    "causalconv1d": ("ncw", "forward"),
    "causalconv1dbwd": ("ncw", "backward"),
    "causalconv1dnwh": ("nwh", "forward"),
    "causalconv1dnwhbwd": ("nwh", "backward"),
    "b2bcausalconv1d": ("b2b", "forward"),
    "b2bcausalconv1dbwd": ("b2b", "backward"),
}


def extract(source):
    cases = []
    for line_number, line in enumerate(source.splitlines(), 1):
        match = re.match(r'"([^"]+)"\s*=\s*(.*)', line)
        if not match:
            continue
        fields = dict(re.findall(r"(\w+):([^*\s]*)", match[2]))
        if fields.get("R") not in ROUTINES:
            continue
        family, direction = ROUTINES[fields["R"]]
        dtype = {"s": "float32", "d": "float64", "h": "float16", "g": "bfloat16"}[fields.get("Pin", fields.get("P"))]
        if fields.get("mode", "0") not in ("0", "1"):
            raise ValueError(f"Unknown activation: {line}")
        cases.append(
            dict(
                name=match[1],
                family=family,
                direction=direction,
                shape=[int(fields[x]) for x in ("n", "c", "w")],
                width=int(fields["r"]),
                mixer_width=int(fields["s"]) if family == "b2b" else None,
                dtype=dtype,
                silu=fields.get("mode", "0") == "1",
                iterations=int(fields.get("N", 1)),
                min_device_version=int(fields.get("minDevVer", 0)),
                source_line=line_number,
                definition=line,
            )
        )
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Expected nonempty, uniquely named causal-conv cases")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--ref", required=True, help="Backend revision containing scripts/levels.layer")
    parser.add_argument("--active-levels", type=Path, help="Optionally verify that another checkout's cases are an exact subset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", args.ref], cwd=args.backend_root, text=True).strip()
    source = subprocess.check_output(["git", "show", revision + ":scripts/levels.layer"], cwd=args.backend_root, text=True)
    cases = extract(source)
    result = dict(backend_revision=revision, source="scripts/levels.layer", source_sha256=hashlib.sha256(source.encode()).hexdigest())
    if args.active_levels:
        active = extract(args.active_levels.read_text())
        by_name = {c["name"]: c for c in cases}
        for case in active:
            other = by_name[case["name"]]
            for key in case.keys() - {"definition", "source_line"}:
                if case[key] != other[key]:
                    raise ValueError(f"Active case {case['name']} differs in {key}")
        result["active_checkout_named_cases"] = len(active)
    result["cases"] = cases
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{revision}: {len(cases)} cases")


if __name__ == "__main__":
    main()
