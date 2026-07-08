# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Automatically discovered import-boundary contracts for JAX APIs."""

from __future__ import annotations

import ast
from collections import deque
import importlib.util
from pathlib import Path
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"


def _module_name(path):
    relative = path.relative_to(_CUDNN_ROOT)
    parts = relative.parts[:-1] if path.name == "__init__.py" else relative.with_suffix("").parts
    return ".".join(("cudnn", *parts))


def _static_condition(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id == "TYPE_CHECKING":
        return False
    if isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING":
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _static_condition(node.operand)
        return None if value is None else not value
    if (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "__name__"
        and len(node.ops) == len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == "__main__"
    ):
        if isinstance(node.ops[0], ast.Eq):
            return False
        if isinstance(node.ops[0], ast.NotEq):
            return True
    return None


def _catches_import_error(node):
    if isinstance(node, ast.Name):
        return node.id in {"ImportError", "ModuleNotFoundError"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"ImportError", "ModuleNotFoundError"}
    if isinstance(node, ast.Tuple):
        return any(_catches_import_error(item) for item in node.elts)
    return False


def _imports(path, *, all_scopes):
    """Collect imports and whether a direct Torch import is optional."""

    imports = []

    def visit(node, torch_optional=False):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append((node, torch_optional))
            return
        if isinstance(node, ast.If):
            condition = _static_condition(node.test)
            if condition is True:
                branches = node.body
            elif condition is False:
                branches = node.orelse
            else:
                branches = (*node.body, *node.orelse)
            for child in branches:
                visit(child, torch_optional)
            return
        if isinstance(node, ast.Try):
            catches_import = any(
                handler.type is not None and _catches_import_error(handler.type)
                for handler in node.handlers
            )
            for child in node.body:
                visit(child, torch_optional or catches_import)
            for handler in node.handlers:
                for child in handler.body:
                    visit(child, torch_optional)
            for child in (*node.orelse, *node.finalbody):
                visit(child, torch_optional)
            return
        if not all_scopes and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        for child in ast.iter_child_nodes(node):
            visit(child, torch_optional)

    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        visit(node)
    return imports


def _import_targets(path, node, modules):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]

    if node.level:
        current_module = _module_name(path)
        package = current_module if path.name == "__init__.py" else current_module.rpartition(".")[0]
        base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
    else:
        base = node.module or ""

    targets = [base] if base else []
    for alias in node.names:
        candidate = f"{base}.{alias.name}" if base else alias.name
        if alias.name != "*" and candidate in modules:
            targets.append(candidate)
    return targets


def _package_initializers(path):
    initializers = [_CUDNN_ROOT / "__init__.py"]
    current = _CUDNN_ROOT
    for part in path.parent.relative_to(_CUDNN_ROOT).parts:
        current /= part
        initializer = current / "__init__.py"
        if initializer != path and initializer.is_file():
            initializers.append(initializer)
    return initializers


def _jax_import_violations(root, modules):
    # Scan every scope in JAX roots because kernel imports are deferred until
    # tracing. In reached modules, scan only code executed during import so a
    # shared kernel file may still contain function-local Torch wrappers.
    queue = deque([(root, True, (root,))])
    queue.extend((path, False, (root, path)) for path in _package_initializers(root))
    scanned_all_scopes = set()
    scanned_module_scope = set()
    visited = set()
    violations = []

    while queue:
        path, all_scopes, chain = queue.popleft()
        if path in scanned_all_scopes or (not all_scopes and path in scanned_module_scope):
            continue
        (scanned_all_scopes if all_scopes else scanned_module_scope).add(path)
        visited.add(path)

        for node, torch_optional in _imports(path, all_scopes=all_scopes):
            for target in _import_targets(path, node, modules):
                chain_text = " -> ".join(str(item.relative_to(_REPO_ROOT)) for item in chain)
                if target == "torch" or target.startswith("torch."):
                    if not torch_optional:
                        violations.append(f"{chain_text}:{node.lineno} imports {target}")
                    continue

                dependency = modules.get(target)
                if dependency is None:
                    continue
                if dependency.name == "api.py":
                    violations.append(
                        f"{chain_text}:{node.lineno} imports Torch API module "
                        f"{dependency.relative_to(_REPO_ROOT)}"
                    )
                    continue
                queue.extend(
                    (initializer, False, (*chain, initializer))
                    for initializer in _package_initializers(dependency)
                )
                queue.append((dependency, False, (*chain, dependency)))

    return visited, violations


class JaxImportContractTest(unittest.TestCase):
    def test_jax_import_graph_has_no_required_torch_dependency(self):
        modules = {_module_name(path): path for path in _CUDNN_ROOT.rglob("*.py")}
        roots = set(_CUDNN_ROOT.rglob("jax.py"))
        roots.update((_CUDNN_ROOT / "jax").rglob("*.py"))
        self.assertTrue(roots, "No JAX API modules were discovered")

        visited = set()
        violations = []
        for root in sorted(roots):
            root_visited, root_violations = _jax_import_violations(root, modules)
            visited.update(root_visited)
            violations.extend(root_violations)

        self.assertTrue(
            any(path.name == "__init__.py" and path not in roots for path in visited),
            "JAX import graph did not discover package initializers",
        )
        self.assertTrue(
            any(path.name != "__init__.py" and path not in roots for path in visited),
            "JAX import graph did not discover deferred kernel dependencies",
        )
        self.assertEqual(
            violations,
            [],
            "JAX import paths must not require Torch:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
