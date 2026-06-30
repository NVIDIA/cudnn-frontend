# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free smoke tests for the optional JAX API surface."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import re
import sys
import types
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"
_TEST_PACKAGE = "cudnn_frontend_jax_surface_test"


class JaxApiSurfaceTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    def test_optional_namespace_does_not_import_frameworks(self):
        parent = types.ModuleType(_TEST_PACKAGE)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = _TEST_PACKAGE
        sys.modules[_TEST_PACKAGE] = parent

        optional_roots = ("jax", "cutlass", "torch")
        before = {name for name in sys.modules if name.split(".", 1)[0] in optional_roots}

        jax_namespace = importlib.import_module(f"{_TEST_PACKAGE}.jax")

        after = {name for name in sys.modules if name.split(".", 1)[0] in optional_roots}
        self.assertEqual(after - before, set())
        self.assertTrue(callable(jax_namespace.rmsnorm_rht_amax_sm100))
        self.assertEqual(jax_namespace.RmsNormRhtAmaxResult._fields, ("output", "amax"))
        self.assertEqual(jax_namespace.__all__, ["RmsNormRhtAmaxResult", "rmsnorm_rht_amax_sm100"])

    def test_torch_function_remains_a_lazy_top_level_export(self):
        top_level_tree = ast.parse(
            (_CUDNN_ROOT / "__init__.py").read_text(),
            filename=str(_CUDNN_ROOT / "__init__.py"),
        )
        lazy_imports_node = next(
            node.value
            for node in top_level_tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_LAZY_OPTIONAL_IMPORTS" for target in node.targets)
        )
        lazy_imports = ast.literal_eval(lazy_imports_node)
        self.assertEqual(
            lazy_imports["rmsnorm_rht_amax_sm100"],
            (".rmsnorm_rht_amax", "rmsnorm_rht_amax_sm100"),
        )

    def test_jax_optional_extra_does_not_install_torch(self):
        pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
        optional_dependencies = re.search(
            r"(?ms)^\[project\.optional-dependencies\]\n(?P<body>.*?)(?=^\[|\Z)",
            pyproject,
        )
        self.assertIsNotNone(optional_dependencies)
        body = optional_dependencies.group("body")
        jax_dependencies = re.search(r"(?ms)^jax = \[\n(?P<body>.*?)^\]$", body)
        self.assertIsNotNone(jax_dependencies)
        self.assertNotIn('"torch"', jax_dependencies.group("body"))


if __name__ == "__main__":
    unittest.main()
