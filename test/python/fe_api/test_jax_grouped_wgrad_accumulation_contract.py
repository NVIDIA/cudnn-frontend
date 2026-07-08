# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for functional JAX grouped-wgrad accumulation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.L0

_ROOT = Path(__file__).resolve().parents[3]
_WGRAD_PATH = (
    _ROOT / "python" / "cudnn" / "grouped_gemm" / "grouped_gemm_wgrad" / "jax.py"
)


def _definitions() -> dict[str, ast.AST]:
    tree = ast.parse(_WGRAD_PATH.read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }


def _method(node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        child
        for child in node.body
        if isinstance(child, ast.FunctionDef) and child.name == name
    )


def _argument_names(function: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(
        argument.arg for argument in (*function.args.args, *function.args.kwonlyargs)
    )


def test_wgrad_seed_is_part_of_the_class_and_wrapper_signatures():
    definitions = _definitions()
    operation = definitions["GroupedGemmWgradSm100"]
    assert isinstance(operation, ast.ClassDef)
    wrapper = definitions["grouped_gemm_wgrad_wrapper_sm100"]
    assert isinstance(wrapper, ast.FunctionDef)

    assert "sample_wgrad_tensor" in _argument_names(_method(operation, "__init__"))
    assert "wgrad_tensor" in _argument_names(_method(operation, "__call__"))
    assert "wgrad_tensor" in _argument_names(_method(operation, "_call_impl"))
    assert "wgrad_tensor" in _argument_names(wrapper)


def test_wgrad_seed_is_aliased_as_the_custom_call_result():
    implementation = _definitions()["_grouped_gemm_wgrad_impl"]
    assert isinstance(implementation, ast.FunctionDef)
    calls = [
        node
        for node in ast.walk(implementation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "call_cutedsl"
    ]
    assert len(calls) == 1

    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.unparse(keywords["output_seeds"]) == "(wgrad_tensor,)"

    source = ast.get_source_segment(_WGRAD_PATH.read_text(), implementation)
    assert source is not None
    assert "wgrad_tensor is only valid when accumulate_on_output=True" in source
    assert "wgrad_dtype must match wgrad_tensor.dtype" in source
    assert "accumulate_on_output and wgrad_tensor is None" in source


def test_fresh_output_and_seeded_accumulation_are_documented_separately():
    source = _WGRAD_PATH.read_text()

    assert "Omitting the seed preserves the" in source
    assert "simple fresh-output behavior by starting from zero" in source
    assert "Pointer-table outputs are not supported" in source
