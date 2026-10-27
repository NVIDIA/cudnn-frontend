# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check exact backend data bits on the frontend's common BF16 K4 SiLU workloads."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test/python/causal_conv1d/cutedsl"))
from se_bitwise import run_case


def main():
    """Check candidate/backend equality on the shared frontend BF16 K4 SiLU workload shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    for B, D, L in [(2, 768, 4096), (1, 12288, 16384), (2, 12288, 8192), (4, 12288, 4096), (8, 12288, 2048)]:
        for direction in ("forward", "backward"):
            case = dict(
                name=f"frontend_overlap_{B}_{D}_{L}_{direction}",
                family="nwh",
                direction=direction,
                shape=[B, D, L],
                width=4,
                mixer_width=None,
                dtype="bfloat16",
                silu=True,
                iterations=3,
                min_device_version=800,
            )
            result = run_case(case)
            result["status"] = "pass" if result["accepted"] else "failure"
            (args.output / (case["name"] + ".json")).write_text(json.dumps(result, indent=2) + "\n")
            print(case["name"], result["status"], flush=True)
            if not result["accepted"]:
                failures.append(case["name"])
    if failures:
        raise SystemExit("Failed: " + str(failures))


if __name__ == "__main__":
    main()
