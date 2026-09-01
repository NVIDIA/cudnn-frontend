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
    return any(_dotted_name(decorator) == "cute.kernel" for decorator in node.decorator_list)


def _is_cudnn_name_prefix(statement: ast.stmt, kernel_name: str) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False

    call = statement.value
    if _dotted_name(call.func) != f"{kernel_name}.set_name_prefix":
        return False
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant) or call.args[0].value != "cudnn":
        return False

    remove_cutlass_symbol = next((kw.value for kw in call.keywords if kw.arg == "remove_cutlass_symbol"), None)
    return isinstance(remove_cutlass_symbol, ast.Constant) and remove_cutlass_symbol.value is True


def _is_any_cudnn_name_prefix(statement: ast.AST) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    function = statement.value.func
    if not isinstance(function, ast.Attribute) or function.attr != "set_name_prefix":
        return False
    return _is_cudnn_name_prefix(statement, _dotted_name(function.value))


def _missing_prefixes(statements: list[ast.stmt], scope: tuple[str, ...] = ()) -> list[str]:
    missing = []
    for index, statement in enumerate(statements):
        if isinstance(statement, ast.FunctionDef) and _is_cute_kernel(statement):
            next_statement = statements[index + 1] if index + 1 < len(statements) else None
            if next_statement is None or not _is_cudnn_name_prefix(next_statement, statement.name):
                qualified_name = ".".join((*scope, statement.name))
                missing.append(f"{statement.lineno}:{qualified_name}")

        if isinstance(statement, ast.ClassDef):
            missing.extend(_missing_prefixes(statement.body, (*scope, statement.name)))

    return missing


@pytest.mark.L0
def test_gemm_and_sdpa_kernels_have_cudnn_name_prefix():
    missing = []
    counts = {}
    for root in _KERNEL_ROOTS:
        kernel_count = prefix_count = 0
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            kernel_count += sum(isinstance(node, ast.FunctionDef) and _is_cute_kernel(node) for node in ast.walk(tree))
            prefix_count += sum(_is_any_cudnn_name_prefix(node) for node in ast.walk(tree))
            offenders = _missing_prefixes(tree.body)
            missing.extend(f"{path.relative_to(_REPO_ROOT)}:{offender}" for offender in offenders)
        counts[root.name] = (kernel_count, prefix_count)

    assert all(kernel_count for kernel_count, _ in counts.values()), f"expected Frost kernels under every scanned root, got {counts}"
    assert all(kernel_count == prefix_count for kernel_count, prefix_count in counts.values()), f"kernel/prefix count mismatch: {counts}"
    assert not missing, "Frost @cute.kernel definitions missing the required cuDNN name prefix:\n" + "\n".join(missing)
