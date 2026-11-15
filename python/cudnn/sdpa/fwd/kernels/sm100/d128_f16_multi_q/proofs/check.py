# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the prefill source/model proof and reject four isolated source mutations.

Requires Lean 4.15.0 and Python's standard library, no CUDA/GPU/Python packages.
Generated proofs and exact Lean diagnostics are retained in --output-dir.
Kernel source is copied for mutations; production files are never modified.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from trace_softmax import KERNEL, MASK, POINTWISE, ROOT, SOURCES, emit

MUTATIONS = (
    (
        "reverse_pair_fold",
        POINTWISE,
        "for i in range(1, half):",
        "for i in range(half - 1, 0, -1):",
        "unmasked",
    ),
    (
        "move_rescale_after_sum",
        KERNEL,
        "total_sum = total_sum * alpha_pair + new_p_sum_pair",
        "total_sum = (total_sum + new_p_sum_pair) * alpha_pair",
        "unmasked",
    ),
    (
        "change_exp2_mode",
        KERNEL,
        "chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)",
        "chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=False)",
        "unmasked",
    ),
    (
        "causal_upper_edge_off_by_one",
        MASK,
        "hi = q_caus_lim + cutlass.Int32(1)",
        "hi = q_caus_lim + cutlass.Int32(2)",
        "causalFirst",
    ),
)


def check(lean, proof, *, expected_success):
    # A useful detector, not a proof that the Python bridge is trustworthy.
    forbidden = re.search(r"\b(sorry|admit|native_decide)\b", proof.read_text())
    if forbidden:
        raise ValueError(f"Unexpected proof escape hatch: {forbidden.group()}")
    start = time.monotonic()
    proc = subprocess.run([lean, str(proof)], capture_output=True, text=True, timeout=180)
    log = proc.stdout + proc.stderr
    log_path = proof.with_suffix(".log")
    log_path.write_text(log)
    axiom_lines = [line for line in log.splitlines() if "axioms" in line]
    used_axioms = set()
    for line in axiom_lines:
        match = re.search(r"depends on axioms: \[(.*)\]", line)
        if match:
            used_axioms.update(x.strip() for x in match[1].split(",") if x.strip())
    axiom_audit_passed = bool(axiom_lines) and used_axioms <= {"propext"}
    if expected_success:
        accepted = proc.returncode == 0 and "error:" not in log and axiom_audit_passed
    else:
        # Do not count an import, parser, or missing-tool error as a successful
        # negative control: the intended source/spec equality must be false.
        accepted = proc.returncode != 0 and "tactic 'decide' proved that the proposition" in log and "is false" in log
    return dict(
        accepted=accepted,
        expected_success=expected_success,
        lean_exit_code=proc.returncode,
        lean_seconds=round(time.monotonic() - start, 3),
        log=str(log_path),
        log_sha256=hashlib.sha256(log.encode()).hexdigest(),
        axioms=axiom_lines,
        axiom_allowlist=["propext"],
        axiom_audit_passed=axiom_audit_passed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-bin", required=True, help="Path to Lean 4.15.0 executable")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    lean = shutil.which(args.lean_bin)
    if lean is None:
        parser.error("Lean executable not found")
    version = subprocess.check_output([lean, "--version"], text=True).strip()
    if not version.startswith("Lean (version 4.15.0,"):
        parser.error(f"This demonstration is pinned to Lean 4.15.0; found {version}")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(lean_version=version, source_root=str(args.source_root.resolve()), trials=[], all_checks_passed=False)

    proof = out / "PrefillSource.lean"
    trace = emit(args.source_root, proof)
    result = check(lean, proof, expected_success=True)
    trace["lean_checked"] = result["accepted"]
    report["trials"].append(dict(name="baseline", trace=trace, check=result))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"baseline: {'PASS' if result['accepted'] else 'FAIL'} ({result['lean_seconds']} seconds)", flush=True)
    if not result["accepted"]:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit("Baseline did not prove; see PrefillSource.log")

    for name, path, before, after, case in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix="frost-prefill-proof-") as directory:
            mutated = Path(directory)
            for source in SOURCES:
                dest = mutated / source
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.source_root / source, dest)
            target = mutated / path
            code = target.read_text()
            helper = {KERNEL: "_softmax_kv_body", POINTWISE: "row_reduction_pair", MASK: "apply_mask_chunk"}[path]
            functions = [node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef) and node.name == helper]
            if len(functions) != 1:
                raise ValueError(f"Mutation {name} expected a unique helper")
            fn = functions[0]
            lines = code.splitlines(keepends=True)
            body = "".join(lines[fn.lineno - 1 : fn.end_lineno])
            if body.count(before) != 1:
                raise ValueError(f"Mutation {name} expected exactly one site in {helper}")
            target.write_text("".join(lines[: fn.lineno - 1]) + body.replace(before, after) + "".join(lines[fn.end_lineno :]))
            proof = out / f"{name}.lean"
            trace = emit(mutated, proof, case_names=(case,))
            result = check(lean, proof, expected_success=False)
            report["trials"].append(dict(name=name, mutation=dict(path=path, helper=helper, before=before, after=after), trace=trace, check=result))
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name}: {'REJECTED AS EXPECTED' if result['accepted'] else 'FAIL'} ({result['lean_seconds']} seconds)", flush=True)

    report["all_checks_passed"] = all(trial["check"]["accepted"] for trial in report["trials"])
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_checks_passed"]:
        raise SystemExit("Proof validation failed; inspect report.json and Lean logs")


if __name__ == "__main__":
    main()
