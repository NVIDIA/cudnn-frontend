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
from unittest import mock

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

    def _optional_modules(self):
        fake_jnp = types.ModuleType("jax.numpy")
        fake_jax = types.ModuleType("jax")
        fake_jax.__path__ = []
        fake_jax.numpy = fake_jnp

        fake_cutlass_jax = types.ModuleType("cutlass.jax")
        fake_cutlass_jax.TensorSpec = type("TensorSpec", (), {})
        fake_cutlass_jax.jax_to_cutlass_dtype = lambda dtype: dtype
        fake_cutlass = types.ModuleType("cutlass")
        fake_cutlass.__path__ = []
        fake_cutlass.jax = fake_cutlass_jax

        return mock.patch.dict(
            sys.modules,
            {
                "jax": fake_jax,
                "jax.numpy": fake_jnp,
                "cutlass": fake_cutlass,
                "cutlass.jax": fake_cutlass_jax,
            },
        )

    def test_explicit_namespace_loads_jax_without_torch(self):
        parent = types.ModuleType(_TEST_PACKAGE)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = _TEST_PACKAGE
        sys.modules[_TEST_PACKAGE] = parent

        torch_before = {name for name in sys.modules if name.split(".", 1)[0] == "torch"}
        with self._optional_modules():
            jax_namespace = importlib.import_module(f"{_TEST_PACKAGE}.jax")

            self.assertTrue(callable(jax_namespace.rmsnorm_rht_amax_sm100))
            self.assertTrue(callable(jax_namespace.indexer_forward_wrapper))
            self.assertTrue(callable(jax_namespace.indexer_top_k_wrapper))
            self.assertIs(
                jax_namespace.DSA.indexer_forward_wrapper,
                jax_namespace.indexer_forward_wrapper,
            )
            self.assertIs(jax_namespace.DSA.indexer_top_k_wrapper, jax_namespace.indexer_top_k_wrapper)
            self.assertEqual(jax_namespace.RmsNormRhtAmaxResult._fields, ("output", "amax"))
            self.assertEqual(jax_namespace.IndexerForwardResult._fields, ("scores",))
            self.assertEqual(jax_namespace.IndexerTopKResult._fields, ("indices", "values"))
            self.assertEqual(
                jax_namespace.__all__,
                [
                    "DSA",
                    "IndexerForwardResult",
                    "IndexerTopKResult",
                    "RmsNormRhtAmaxResult",
                    "indexer_forward_wrapper",
                    "indexer_top_k_wrapper",
                    "rmsnorm_rht_amax_sm100",
                ],
            )

            torch_after = {name for name in sys.modules if name.split(".", 1)[0] == "torch"}
            self.assertEqual(torch_after - torch_before, set())

    def test_missing_dependency_reports_optional_extra(self):
        missing_modules = {
            "jax": {"jax": None},
            "cutlass": {
                "jax": types.ModuleType("jax"),
                "cutlass": None,
                "cutlass.jax": None,
            },
        }
        for dependency, blocked_modules in missing_modules.items():
            with self.subTest(dependency=dependency):
                package_name = f"{_TEST_PACKAGE}_missing_{dependency}"
                parent = types.ModuleType(package_name)
                parent.__path__ = [str(_CUDNN_ROOT)]
                parent.__package__ = package_name
                sys.modules[package_name] = parent

                try:
                    with (
                        mock.patch.dict(sys.modules, blocked_modules),
                        self.assertRaisesRegex(ImportError, r"nvidia-cudnn-frontend\[jax\]"),
                    ):
                        importlib.import_module(f"{package_name}.jax")
                finally:
                    for module_name in tuple(sys.modules):
                        if module_name == package_name or module_name.startswith(f"{package_name}."):
                            sys.modules.pop(module_name, None)

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
        self.assertEqual(lazy_imports["DSA"], (".deepseek_sparse_attention", "DSA"))

    def test_dsa_operator_names_match_torch_namespace(self):
        dsa_tree = ast.parse(
            (_CUDNN_ROOT / "deepseek_sparse_attention" / "__init__.py").read_text(),
            filename=str(_CUDNN_ROOT / "deepseek_sparse_attention" / "__init__.py"),
        )
        symbols_node = next(
            node.value
            for node in dsa_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_SYMBOLS" for target in node.targets)
        )
        torch_symbols = ast.literal_eval(symbols_node)
        for name in ("indexer_forward_wrapper", "indexer_top_k_wrapper"):
            with self.subTest(name=name):
                self.assertIn(name, torch_symbols)

    def test_shared_dsa_kernel_import_paths_do_not_import_torch(self):
        shared_files = (
            _CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_forward" / "__init__.py",
            _CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_forward" / "indexer_fwd_sm100.py",
            _CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_top_k" / "__init__.py",
            _CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_top_k" / "indexer_top_k_decode_varlen.py",
            _CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_top_k" / "indexer_top_k_varlen_util.py",
        )
        for path in shared_files:
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(), filename=str(path))
                top_level_imports = (node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)))
                imported_roots = {
                    alias.name.split(".", 1)[0]
                    for node in top_level_imports
                    for alias in node.names
                    if isinstance(node, ast.Import)
                }
                imported_roots.update(
                    node.module.split(".", 1)[0]
                    for node in tree.body
                    if isinstance(node, ast.ImportFrom) and node.module
                )
                self.assertNotIn("torch", imported_roots)

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
