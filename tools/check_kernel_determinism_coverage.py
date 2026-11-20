# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check registration and PR test updates for frontend-owned GPU kernels (standard library only)."""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "tools/kernel_determinism_manifest.json"
KERNEL_DIRS = {"kernel", "kernels", "kernel_templates"}
# The frontend also routes compilation through cudnn.frost.compiled_cache.
DSL_APIS = {
    "cutlass.cute.kernel",
    "cutlass.cute.jit",
    "cutlass.cute.compile",
    "cuda.tile.kernel",
    "cuda.tile.launch",
    "triton.jit",
    "triton.autotune",
    "triton.heuristics",
    "cudnn.frost.compiled_cache.compile_cached",
}
CUDA_KERNEL = re.compile(r"\b__global__\b|<<<")


def git_paths(root, *args):
    data = subprocess.check_output(["git", *args], cwd=root)
    return {os.fsdecode(path) for path in data.split(b"\0") if path}


def has_dsl_dispatch(source):
    if not any(word in source for word in ("cute", "tile", "triton", "compile_cached")):
        return False
    tree = ast.parse(source)
    aliases = {}
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({item.asname: item.name for item in node.names if item.asname})
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            aliases.update({item.asname or item.name: f"{node.module}.{item.name}" for item in node.names})
        elif isinstance(node, ast.Call):
            calls.append(node.func)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            calls.extend(node.decorator_list)
    for target in calls:
        parts = []
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            name = ".".join([aliases.get(target.id, target.id), *reversed(parts)])
            if name in DSL_APIS:
                return True
    return False


def kernel_source(root, path):
    """Recognize kernel directories, DSL definitions/launches and native CUDA sources."""
    p = Path(path)
    if (
        p.parent.as_posix() in {"python/cudnn/_cake", "python/cudnn/_cake/_vendor"} and p.match("VENDORED_*.json")
    ) or path == "python/cudnn/linear_attention/cake/kernels/SHA256SUMS":
        return True
    if path.startswith(("python/cudnn/_cake/_vendor/", "python/cudnn/moe_ep/_megamoe_backend/cutedsl_src/", "include/cudnn_frontend/thirdparty/")):
        return False
    if path.startswith("python/cudnn/linear_attention/cake/kernels/") and not p.name.startswith("fe_"):
        return False
    if not path.startswith(("python/", "include/")) and not (len(p.parts) >= 4 and p.parts[0] == "aot" and p.parts[2] == "src"):
        return False
    if p.suffix in {".cu", ".cuh"}:
        return True
    if p.suffix == ".py" and path.startswith("python/cudnn/"):
        if p.name in {"api.py", "jax_api.py"} or "ops" in p.parts[2:-1]:
            return True
        if p.name != "__init__.py" and KERNEL_DIRS.intersection(p.parts[2:-1]):
            return True
        return has_dsl_dispatch((root / p).read_text(encoding="utf-8"))
    if p.suffix in {".c", ".cc", ".cpp", ".h", ".hpp"}:
        return bool(CUDA_KERNEL.search((root / p).read_text(encoding="utf-8")))
    return False


def kernel_files(root):
    return {path for path in git_paths(root, "ls-files", "-z", "--", "python", "include", "aot") if (root / path).is_file() and kernel_source(root, path)}


def load_registry(root, tracked):
    """Validate all paths even on PRs that only edit the manifest or remove a test."""
    groups = json.loads((root / MANIFEST).read_text())
    if not isinstance(groups, list):
        raise ValueError("expected a list of kernel groups")
    entries = {}
    for index, group in enumerate(groups, 1):
        tests = group.get("tests", [])
        reason = group.get("exempt_reason", "")
        sources = group.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"group {index} needs a nonempty sources list")
        if not isinstance(tests, list) or not isinstance(reason, str) or bool(tests) == bool(reason.strip()):
            raise ValueError(f"group {index} needs tests or an explicit exempt_reason")
        for path in sources + tests:
            if path not in tracked or not (root / path).is_file():
                raise ValueError(f"group {index}: missing tracked file {path!r}")
        for test in tests:
            p = Path(test)
            in_test_tree = test.startswith("test/") or (len(p.parts) >= 4 and p.parts[0] == "aot" and p.parts[2] == "test")
            if not in_test_tree or p.suffix not in {".py", ".cpp", ".cc", ".cu"}:
                raise ValueError(f"{test}: expected a replay test under test/ or aot/<arch>/test/")
        for path in sources:
            if path in entries:
                raise ValueError(f"{path}: source appears in multiple groups")
            entries[path] = group
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="check registration across the tracked tree")
    mode.add_argument("--base", help="check the committed PR diff from the merge base with this ref")
    parser.add_argument("--allow-test-exemption", action="store_true", help="waive the test update for a non-numerical PR")
    args = parser.parse_args()
    tracked = git_paths(ROOT, "ls-files", "-z")
    try:
        entries = load_registry(ROOT, tracked)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        parser.exit(1, f"{MANIFEST}: {exc}\n")

    if args.all:
        changed = kernel_files(ROOT)
    else:
        changed = git_paths(ROOT, "diff", "-z", "--name-only", "--no-renames", "--diff-filter=AMT", f"{args.base}...HEAD", "--")
    passed = True
    for path in sorted(changed):
        if path not in entries:
            if args.all or kernel_source(ROOT, path):
                print(f"{path}: unregistered kernel source. Add its determinism tests or an explicit exemption to {MANIFEST}.", file=sys.stderr)
                passed = False
            continue
        if args.all:
            continue
        group = entries[path]
        if group.get("exempt_reason"):
            print(f"{path}: registered replay gap: {group['exempt_reason']}")
            continue
        tests = group["tests"]
        if changed.intersection(tests):
            continue
        if args.allow_test_exemption:
            print(f"{path}: test update waived by determinism-exempt")
        else:
            print(f"{path}: update a registered determinism test in this PR: {', '.join(tests)}", file=sys.stderr)
            passed = False
    if passed:
        print("Kernel determinism coverage check passed.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
