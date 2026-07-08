# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts for framework-neutral frontend results."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest

_CUDNN_ROOT = Path(__file__).resolve().parents[3] / "python" / "cudnn"
_COMMON_SPEC = importlib.util.spec_from_file_location(
    "_cudnn_result_contract",
    _CUDNN_ROOT / "common" / "result.py",
)
assert _COMMON_SPEC is not None and _COMMON_SPEC.loader is not None
_COMMON_MODULE = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON_MODULE)
TupleDict = _COMMON_MODULE.TupleDict


pytestmark = pytest.mark.L0


def test_tuple_dict_preserves_named_and_positional_access():
    result = TupleDict(output=1, amax=2)

    assert tuple(result) == (1, 2)
    assert result[0] == result["output"] == 1
    assert result[1] == result["amax"] == 2
    with pytest.raises(IndexError, match="index -1 out of range"):
        _ = result[-1]


def test_tuple_dict_is_a_jax_pytree(monkeypatch):
    jax = pytest.importorskip("jax")

    package_name = "_cudnn_result_jax_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_CUDNN_ROOT)]
    common_package = types.ModuleType(f"{package_name}.common")
    common_package.__path__ = [str(_CUDNN_ROOT / "common")]
    jax_package = types.ModuleType(f"{package_name}._jax")
    jax_package.__path__ = [str(_CUDNN_ROOT / "_jax")]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, f"{package_name}.common", common_package)
    monkeypatch.setitem(sys.modules, f"{package_name}._jax", jax_package)
    monkeypatch.setitem(sys.modules, f"{package_name}.common.result", _COMMON_MODULE)

    spec = importlib.util.spec_from_file_location(
        f"{package_name}._jax.result",
        _CUDNN_ROOT / "_jax" / "result.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    JaxTupleDict = module.TupleDict

    result = JaxTupleDict(output=1, amax=2)
    leaves, treedef = jax.tree_util.tree_flatten(result)

    assert leaves == [1, 2]
    assert jax.tree_util.tree_unflatten(treedef, leaves) == result
