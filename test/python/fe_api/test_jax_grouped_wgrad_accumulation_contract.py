# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts for JAX-owned grouped-wgrad outputs."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.L0

_ROOT = Path(__file__).resolve().parents[3]
_WGRAD_PATH = (
    _ROOT / "python" / "cudnn" / "grouped_gemm" / "grouped_gemm_wgrad" / "jax.py"
)
_SHARED_PATH = _ROOT / "python" / "cudnn" / "grouped_gemm" / "_jax_api.py"


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


def test_wgrad_output_is_not_part_of_the_class_or_wrapper_signatures():
    definitions = _definitions()
    operation = definitions["GroupedGemmWgradSm100"]
    assert isinstance(operation, ast.ClassDef)
    wrapper = definitions["grouped_gemm_wgrad_wrapper_sm100"]
    assert isinstance(wrapper, ast.FunctionDef)
    implementation = definitions["_grouped_gemm_wgrad_impl"]
    assert isinstance(implementation, ast.FunctionDef)

    assert "sample_wgrad_tensor" not in _argument_names(
        _method(operation, "__init__")
    )
    assert "wgrad_tensor" not in _argument_names(_method(operation, "__call__"))
    assert "wgrad_tensor" not in _argument_names(_method(operation, "_call_impl"))
    assert "wgrad_tensor" not in _argument_names(implementation)
    assert "wgrad_tensor" not in _argument_names(wrapper)


def test_accumulating_wgrad_is_zero_initialized_by_its_descriptor():
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
    assert "output_seeds" not in keywords

    output_descs = [
        node
        for node in ast.walk(implementation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_buffer_desc"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "wgrad_tensor"
    ]
    assert len(output_descs) == 1
    output_keywords = {
        keyword.arg: keyword.value for keyword in output_descs[0].keywords
    }
    assert ast.unparse(output_keywords["init_value"]) == (
        "0.0 if accumulate_on_output else None"
    )

    source = ast.get_source_segment(_WGRAD_PATH.read_text(), implementation)
    assert source is not None
    assert "default=jnp.bfloat16" in source


def test_grouped_lowering_does_not_accept_preallocated_outputs():
    tree = ast.parse(_SHARED_PATH.read_text())
    call_cutedsl = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "call_cutedsl"
    )

    assert "output_seeds" not in _argument_names(call_cutedsl)
    assert "output_seeds" not in ast.unparse(call_cutedsl)


def test_wgrad_layout_and_alignment_constraints_live_on_descriptors():
    source = _WGRAD_PATH.read_text()
    implementation = _definitions()["_grouped_gemm_wgrad_impl"]
    assert isinstance(implementation, ast.FunctionDef)
    call = next(
        node
        for node in ast.walk(implementation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "call_cutedsl"
    )
    keywords = {keyword.arg for keyword in call.keywords}

    assert "WGRAD_B_STRIDE_ORDER = (0, 1)" in source
    assert "WGRAD_ALIGNMENT = 16" in source
    assert "ptr_assumed_align=GROUPED_WORKSPACE_ALIGNMENT" in source
    assert "input_descs" in keywords
    assert not {"input_specs", "output_specs", "workspace_specs"} & keywords


def test_jax_owned_output_and_zero_initialization_are_documented():
    source = _WGRAD_PATH.read_text()

    assert "The output is inferred and allocated by JAX" in source
    assert "zero-initialized output" in source
    assert "Pointer-table outputs are not supported" in source
