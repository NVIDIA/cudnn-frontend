# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KERNEL_ROOTS = (
    _REPO_ROOT / "python" / "cudnn" / "gemm",
    _REPO_ROOT / "python" / "cudnn" / "sdpa",
)


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    return ""


def _is_cute_kernel(node: ast.FunctionDef) -> bool:
    return any(_dotted_name(decorator.func if isinstance(decorator, ast.Call) else decorator) == "cute.kernel" for decorator in node.decorator_list)


def _is_cudnn_name_prefix(statement: ast.stmt, kernel_name: str) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False

    call = statement.value
    if _dotted_name(call.func) != f"{kernel_name}.set_name_prefix":
        return False
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant) or call.args[0].value != "cudnn":
        return False

    return True


def _missing_prefixes(node: ast.AST, scope: tuple[str, ...] = ()) -> list[str]:
    missing = []
    for _, value in ast.iter_fields(node):
        children = value if isinstance(value, list) else [value] if isinstance(value, ast.AST) else []
        if isinstance(value, list):
            for index, statement in enumerate(value):
                if isinstance(statement, ast.FunctionDef) and _is_cute_kernel(statement):
                    next_statement = value[index + 1] if index + 1 < len(value) else None
                    if next_statement is None or not _is_cudnn_name_prefix(next_statement, statement.name):
                        missing.append(f"{statement.lineno}:{'.'.join((*scope, statement.name))}")
        for child in children:
            if isinstance(child, ast.AST):
                child_scope = (*scope, child.name) if isinstance(child, (ast.ClassDef, ast.FunctionDef)) else scope
                missing.extend(_missing_prefixes(child, child_scope))

    return missing


@pytest.mark.L0
def test_gemm_and_sdpa_kernels_have_cudnn_name_prefix():
    missing = []
    counts = {}
    for root in _KERNEL_ROOTS:
        kernel_count = 0
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            kernel_count += sum(isinstance(node, ast.FunctionDef) and _is_cute_kernel(node) for node in ast.walk(tree))
            offenders = _missing_prefixes(tree)
            missing.extend(f"{path.relative_to(_REPO_ROOT)}:{offender}" for offender in offenders)
        counts[root.name] = kernel_count

    assert all(counts.values()), f"expected Frost kernels under every scanned root, got {counts}"
    assert not missing, "Frost @cute.kernel definitions missing the required cuDNN name prefix:\n" + "\n".join(missing)


@pytest.mark.L0
def test_prefix_guard_covers_called_decorator_and_nested_kernel():
    tree = ast.parse("def outer():\n    @cute.kernel()\n    def missing():\n        pass\n")
    assert _missing_prefixes(tree) == ["3:outer.missing"]
