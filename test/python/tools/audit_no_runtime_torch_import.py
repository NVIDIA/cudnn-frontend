# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit: no runtime PyTorch import may exist outside ``cudnn/_deps/torch_dep.py``.

Rejects, anywhere under ``python/cudnn`` except the dependency helper itself:

* ``import torch`` and aliased forms (``import torch as _torch``)
* ``from torch import ...`` and submodule forms (``from torch.nn import ...``)
* dynamic imports with a literal target: ``importlib.import_module("torch")``,
  ``__import__("torch.foo")``
* PyTorch-valued parameter defaults
* unguarded module-scope ``torch.<attr>`` access (attribute, decorator, or
  registration)

Imports that only run under ``if TYPE_CHECKING:`` are permitted. For the custom-op
implementation modules, PyTorch syntax is permitted only beneath the single
top-level ``if torch_dep.is_available():`` guard.

Run: ``python test/python/tools/audit_no_runtime_torch_import.py``
Exits non-zero and prints every violation when the invariant is broken.
"""

from __future__ import annotations

import ast
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PKG_ROOT = os.path.join(REPO_ROOT, "python", "cudnn")

# The one module allowed to import torch at runtime.
ALLOWED = {os.path.join("_deps", "torch_dep.py")}

# Custom-op implementation modules: torch syntax allowed only inside the single
# top-level ``if torch_dep.is_available():`` block.
GUARDED_IMPL = {
    os.path.join("ops", "_causal_conv1d_torch.py"),
    os.path.join("experimental", "ops", "_sdpa_torch.py"),
    os.path.join("experimental", "ops", "_moe_grouped_matmul_torch.py"),
}


def _is_torch_name(name: str) -> bool:
    return name == "torch" or name.startswith("torch.")


class Auditor(ast.NodeVisitor):
    def __init__(self, relpath: str):
        self.rel = relpath
        self.violations: list[tuple[int, str]] = []
        self._type_checking_depth = 0
        self._guard_depth = 0
        self._func_depth = 0

    # -- context tracking ------------------------------------------------
    def visit_If(self, node: ast.If) -> None:
        test = ast.unparse(node.test)
        is_tc = "TYPE_CHECKING" in test
        is_guard = "torch_dep.is_available()" in test

        if is_tc:
            self._type_checking_depth += 1
        if is_guard:
            self._guard_depth += 1
        for child in node.body:
            self.visit(child)
        if is_tc:
            self._type_checking_depth -= 1
        if is_guard:
            self._guard_depth -= 1

        # the else branch gets no guard credit
        for child in node.orelse:
            self.visit(child)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_defaults(node)
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    # -- rules -----------------------------------------------------------
    @property
    def _torch_syntax_ok(self) -> bool:
        if self._type_checking_depth:
            return True
        return self.rel in GUARDED_IMPL and self._guard_depth > 0

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if _is_torch_name(alias.name) and not self._torch_syntax_ok:
                self.violations.append((node.lineno, f"runtime import of {alias.name}"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if _is_torch_name(mod) and not self._torch_syntax_ok:
            self.violations.append((node.lineno, f"runtime import from {mod}"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = ast.unparse(node.func)
        if fn in ("importlib.import_module", "import_module", "__import__") and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _is_torch_name(arg.value):
                if not self._torch_syntax_ok:
                    self.violations.append((node.lineno, f"dynamic import of {arg.value!r}"))
        self.generic_visit(node)

    def _check_defaults(self, node) -> None:
        args = node.args
        pos = args.posonlyargs + args.args
        pairs = list(zip(pos[len(pos) - len(args.defaults) :], args.defaults))
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
        for a, d in pairs:
            src = ast.unparse(d)
            if src.startswith("torch."):
                self.violations.append((d.lineno, f"PyTorch-valued default for parameter {a.arg!r}: {src}"))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # module-scope torch.<attr> access, evaluated at import time
        if self._func_depth == 0 and not self._torch_syntax_ok:
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "torch":
                self.violations.append((node.lineno, f"module-scope PyTorch access: {ast.unparse(node)}"))
        self.generic_visit(node)


def _strip_annotations(tree: ast.AST) -> ast.AST:
    """Remove annotations so annotation-only torch references are not flagged."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.arg, ast.AnnAssign)) and getattr(node, "annotation", None) is not None:
            node.annotation = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            node.returns = None
    return tree


def main() -> int:
    failures: list[str] = []
    scanned = 0
    for root, _dirs, files in os.walk(PKG_ROOT):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, PKG_ROOT)
            if rel in ALLOWED:
                continue
            scanned += 1
            tree = _strip_annotations(ast.parse(open(path).read()))
            auditor = Auditor(rel)
            auditor.visit(tree)
            for lineno, msg in auditor.violations:
                failures.append(f"{os.path.join('python/cudnn', rel)}:{lineno}: {msg}")

    print(f"audit: scanned {scanned} modules under python/cudnn")
    if failures:
        print(f"audit FAILED with {len(failures)} violation(s):")
        for f in failures:
            print("  " + f)
        return 1
    print("audit PASSED: no runtime PyTorch import, PyTorch-valued default, or module-scope PyTorch access")
    return 0


if __name__ == "__main__":
    sys.exit(main())
