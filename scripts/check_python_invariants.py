#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-level invariants of python/cudnn, checked without importing the package.

Usage: ``check_python_invariants.py [FILE ...]`` (pre-commit passes the staged
python files). Exit status 1 when any file violates a rule:

* kernel-name-prefix (python/cudnn/AGENTS.md Rule 6): under ``KERNEL_PREFIX_ROOTS``,
  every ``@cute.kernel`` / ``@cute.experimental.kernel`` definition is followed
  immediately by ``<name>.set_name_prefix("cudnn", ...)``.
* device-0: no ``get_device_properties(0)`` / ``get_device_capability(0)`` in
  python/cudnn; the graph API builds for the device in scope. A line that must
  name device 0 carries ``# allow-device-0`` with the reason next to it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

KERNEL_PREFIX_ROOTS = (
    "python/cudnn/QSA/",
    "python/cudnn/attention_gate/",
    "python/cudnn/attention_prepare/",
    "python/cudnn/conv/",
    "python/cudnn/gemm/",
    "python/cudnn/norm/",
    "python/cudnn/sdpa/",
    "python/cudnn/engram/",
    "python/cudnn/rope/",
    "python/cudnn/row_quantize/",
    "python/cudnn/linear_attention/frost/common/state_snapshot.py",
    "python/cudnn/linear_attention/frost/common/head_wave.py",
)

DEVICE0_ROOT = "python/cudnn/"
DEVICE0_PRAGMA = "# allow-device-0"

_KERNEL_DECORATORS = ("cute.kernel", "cute.experimental.kernel")
_DEVICE_QUERIES = ("get_device_properties", "get_device_capability")


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    return ""


def _is_cute_kernel(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return any(_dotted_name(d.func if isinstance(d, ast.Call) else d) in _KERNEL_DECORATORS for d in node.decorator_list)


def _is_cudnn_name_prefix(statement, kernel_name: str) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    if _dotted_name(call.func) != f"{kernel_name}.set_name_prefix":
        return False
    return len(call.args) == 1 and isinstance(call.args[0], ast.Constant) and call.args[0].value == "cudnn"


def missing_name_prefixes(tree: ast.AST, scope: tuple = ()) -> list:
    """``"<line>:<qualified name>"`` of each kernel not followed by its cuDNN name prefix."""
    missing = []
    for _, value in ast.iter_fields(tree):
        if isinstance(value, list):
            for index, statement in enumerate(value):
                if _is_cute_kernel(statement):
                    following = value[index + 1] if index + 1 < len(value) else None
                    if not _is_cudnn_name_prefix(following, statement.name):
                        missing.append(f"{statement.lineno}:{'.'.join((*scope, statement.name))}")
            children = value
        else:
            children = [value]
        for child in children:
            if isinstance(child, ast.AST):
                child_scope = (*scope, child.name) if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else scope
                missing.extend(missing_name_prefixes(child, child_scope))
    return missing


def device0_queries(tree: ast.AST, lines: list) -> list:
    """Line numbers of device-property queries that name device 0 literally."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, (ast.Attribute, ast.Name)):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        if name not in _DEVICE_QUERIES:
            continue
        args = list(node.args[:1]) + [kw.value for kw in node.keywords if kw.arg == "device"]
        if any(isinstance(a, ast.Constant) and a.value == 0 and not isinstance(a.value, bool) for a in args):
            if DEVICE0_PRAGMA not in lines[node.lineno - 1]:
                hits.append(node.lineno)
    return hits


def check_file(path: str, text: str) -> list:
    rel = Path(path).as_posix()
    if rel.startswith("./"):
        rel = rel[2:]
    in_prefix_root = rel.startswith(KERNEL_PREFIX_ROOTS)
    in_device0_root = rel.startswith(DEVICE0_ROOT)
    if not (in_prefix_root or in_device0_root):
        return []
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        return [f"{rel}:{exc.lineno}: cannot parse: {exc.msg}"]
    errors = []
    if in_prefix_root:
        for offender in missing_name_prefixes(tree):
            line, name = offender.split(":", 1)
            errors.append(f'{rel}:{line}: kernel {name} needs {name.rsplit(".", 1)[-1]}.set_name_prefix("cudnn", ...) right after it (Rule 6)')
    if in_device0_root:
        for line in device0_queries(tree, text.splitlines()):
            errors.append(f"{rel}:{line}: device property query names device 0; use the device in scope (or mark the line {DEVICE0_PRAGMA!r})")
    return errors


def main(argv: list) -> int:
    errors = []
    for path in argv:
        if path.endswith(".py"):
            errors.extend(check_file(path, Path(path).read_text(encoding="utf-8")))
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
