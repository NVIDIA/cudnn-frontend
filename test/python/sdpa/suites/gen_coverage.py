# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render COVERAGE.md (the human-readable master list) from registry.py.

Usage (from test/python):
    python -m sdpa.suites.gen_coverage
"""

import os
import sys


def render():
    from sdpa.suites.registry import REGISTRY

    lines = [
        "# SDPA suites — coverage master list",
        "",
        "Generated from `registry.py` by `gen_coverage.py` — do not edit by hand.",
        "",
        "Conventions:",
        "",
        "- One suite = one deterministic seed sweep (`num_tests` configs from `rng_seed`);",
        "  every config prints a `test_repro_suite.py::test_repro --repro ...` command.",
        "- 16-bit is one family: f16 suites draw fp16 or bf16 per config, like fp8 draws e4m3/e5m2.",
        "- Sliding-window/causal masks and bias are fuzz axes inside suites, never separate suites.",
        "- THD suites fuzz first-class packed capacities: `total_q`/`total_kv` slack and",
        "  declaring them on the graph (`sdpa(max_total_seq_len_q/kv=...)`).",
        "- Model suites pin head/dim geometry of popular models (full/global attention only)",
        "  and fuzz everything else through the same runner.",
        "",
    ]

    for phase, title in (
        ("context", "Context (prefill forward)"),
        ("generation", "Generation (decode / small-s_q forward)"),
        ("bprop", "Bprop (training fwd+bwd)"),
    ):
        lines += [
            f"## {title}",
            "",
            "| suite | dtype | level | N | fuzzed | pinned | gates / notes |",
            "|---|---|---|---|---|---|---|",
        ]
        for spec in REGISTRY.values():
            if spec.phase != phase:
                continue
            gates = []
            if spec.min_sm:
                gates.append(f"SM>={spec.min_sm[0]}{spec.min_sm[1]}")
            if spec.notes:
                gates.append(spec.notes)
            lines.append(
                f"| {spec.name} | {spec.dtype} | {spec.level} | {spec.num_tests} "
                f"| {', '.join(spec.fuzzed)} | {', '.join(spec.pinned)} | {'; '.join(gates)} |"
            )
        lines.append("")

    total = sum(s.num_tests for s in REGISTRY.values())
    lines += [f"**Total configs: {total} across {len(REGISTRY)} suites.**", ""]
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "..")))
    out = os.path.join(here, "COVERAGE.md")
    with open(out, "w") as f:
        f.write(render())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
