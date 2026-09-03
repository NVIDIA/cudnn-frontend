# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print compact rows from one or more qlen=1 sweep JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--reference")
    args = parser.parse_args()

    for path in args.files:
        payload = json.loads(path.read_text())
        if "windows" in payload:
            print(
                "SWEEP "
                + json.dumps(
                    {
                        "file": str(path),
                        "device": payload["device"],
                        "direction": payload["direction"],
                        "windows": payload["windows"],
                    },
                    sort_keys=True,
                )
            )
            for case in payload["cases"]:
                for window_name, window in case["windows"].items():
                    times = {implementation: result["median_ms"] for implementation, result in window["results"].items() if "median_ms" in result}
                    errors = {implementation: result["error"] for implementation, result in window["results"].items() if "error" in result}
                    print(
                        "ROW "
                        + json.dumps(
                            {
                                "batch": case["batch_size"],
                                "heads": case["heads"],
                                "average_kv_target": case["average_kv_target"],
                                "average_kv": case["average_kv"],
                                "max_kv": case["max_kv"],
                                "window": window_name,
                                "times_ms": times,
                                "errors": errors,
                            },
                            sort_keys=True,
                        )
                    )
            reference = args.reference or payload["implementations"][0]
            for window in payload["windows"]:
                window_name = window["name"]
                for implementation in payload["implementations"]:
                    speedups = []
                    for case in payload["cases"]:
                        results = case["windows"][window_name]["results"]
                        if reference in results and implementation in results and "median_ms" in results[reference] and "median_ms" in results[implementation]:
                            speedups.append(results[reference]["median_ms"] / results[implementation]["median_ms"])
                    if speedups:
                        print(
                            "AGG "
                            + json.dumps(
                                {
                                    "window": window_name,
                                    "implementation": implementation,
                                    "reference": reference,
                                    "cases": len(speedups),
                                    "geomean_speedup": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
                                    "worst_speedup": min(speedups),
                                    "best_speedup": max(speedups),
                                    "wins": sum(value > 1.0 for value in speedups),
                                },
                                sort_keys=True,
                            )
                        )
            continue
        print(
            "SWEEP "
            + json.dumps(
                {
                    "file": str(path),
                    "device": payload["device"],
                    "direction": payload["direction"],
                    "mask": payload["mask"],
                    "window_size": payload["window_size"],
                },
                sort_keys=True,
            )
        )
        for case in payload["cases"]:
            times = {implementation: result.get("median_ms") for implementation, result in case["results"].items() if "median_ms" in result}
            errors = {implementation: result["error"] for implementation, result in case["results"].items() if "error" in result}
            print(
                "ROW "
                + json.dumps(
                    {
                        "batch": case["batch_size"],
                        "heads": case["heads"],
                        "average_kv_target": case["average_kv_target"],
                        "average_kv": case["average_kv"],
                        "max_kv": case["max_kv"],
                        "times_ms": times,
                        "errors": errors,
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
