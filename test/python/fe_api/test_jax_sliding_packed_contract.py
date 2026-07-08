# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Executable contracts for packed-THD sliding-window repacking."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.L0

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

_ADAPTER = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "cudnn"
    / "native_sparse_attention"
    / "sliding_window_attention"
    / "jax.py"
)


def _load_static_method(name: str):
    tree = ast.parse(_ADAPTER.read_text(), filename=str(_ADAPTER))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SlidingWindowAttention"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"Any": Any, "jax": jax, "jnp": jnp}
    exec(compile(module, str(_ADAPTER), "exec"), namespace)
    return namespace[name]


def test_packed_padding_and_gather_round_trip_under_jit():
    pad_packed = _load_static_method("_pad_packed")
    unpad_packed = _load_static_method("_unpad_packed")
    values = jnp.arange(5, dtype=jnp.float32).reshape(5, 1, 1)
    cumulative = jnp.array([0, 2, 5], dtype=jnp.int32)

    padded = jax.jit(lambda x, offsets: pad_packed(x, offsets[:-1], 3))(
        values, cumulative
    )
    assert padded.shape == (2, 3, 1, 1)
    assert padded[:, :, 0, 0].tolist() == [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]

    restored = jax.jit(lambda x, offsets: unpad_packed(x, offsets, 5))(
        padded, cumulative
    )
    assert restored.shape == values.shape
    assert restored[:, 0, 0].tolist() == values[:, 0, 0].tolist()
