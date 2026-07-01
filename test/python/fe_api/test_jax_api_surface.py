# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free smoke tests for the optional JAX API surface."""

from __future__ import annotations

import ast
import importlib
from importlib.machinery import ModuleSpec
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
        fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
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
                "torch": None,
            },
        )

    def test_explicit_namespace_loads_jax_without_torch(self):
        parent = types.ModuleType(_TEST_PACKAGE)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = _TEST_PACKAGE
        sys.modules[_TEST_PACKAGE] = parent

        torch_before = {
            name
            for name, module in sys.modules.items()
            if name.split(".", 1)[0] == "torch" and module is not None
        }
        with self._optional_modules():
            jax_namespace = importlib.import_module(f"{_TEST_PACKAGE}.jax")
            colocated_modules = (
                f"{_TEST_PACKAGE}.rmsnorm_rht_amax.jax",
                f"{_TEST_PACKAGE}.deepseek_sparse_attention.indexer_forward.jax",
                f"{_TEST_PACKAGE}.deepseek_sparse_attention.indexer_top_k.jax",
            )
            for module_name in colocated_modules:
                self.assertNotIn(module_name, sys.modules)

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
            rmsnorm_package = importlib.import_module(f"{_TEST_PACKAGE}.rmsnorm_rht_amax")
            indexer_forward_package = importlib.import_module(
                f"{_TEST_PACKAGE}.deepseek_sparse_attention.indexer_forward"
            )
            indexer_top_k_package = importlib.import_module(
                f"{_TEST_PACKAGE}.deepseek_sparse_attention.indexer_top_k"
            )
            self.assertNotIn("rmsnorm_rht_amax_sm100", vars(rmsnorm_package))
            self.assertNotIn("indexer_forward_wrapper", vars(indexer_forward_package))
            self.assertNotIn("indexer_top_k_wrapper", vars(indexer_top_k_package))
            self.assertIs(
                rmsnorm_package.jax.rmsnorm_rht_amax_sm100,
                jax_namespace.rmsnorm_rht_amax_sm100,
            )
            self.assertIs(
                indexer_forward_package.jax.indexer_forward_wrapper,
                jax_namespace.indexer_forward_wrapper,
            )
            self.assertIs(
                indexer_top_k_package.jax.indexer_top_k_wrapper,
                jax_namespace.indexer_top_k_wrapper,
            )
            for module_name in colocated_modules:
                self.assertIn(module_name, sys.modules)
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

            torch_after = {
                name
                for name, module in sys.modules.items()
                if name.split(".", 1)[0] == "torch" and module is not None
            }
            self.assertEqual(torch_after - torch_before, set())

    def test_missing_jax_reports_optional_extra(self):
        package_name = f"{_TEST_PACKAGE}_missing_jax"
        parent = types.ModuleType(package_name)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = package_name
        sys.modules[package_name] = parent

        try:
            with (
                mock.patch.dict(sys.modules, {"jax": None}),
                self.assertRaisesRegex(ImportError, r"nvidia-cudnn-frontend\[jax\]"),
            ):
                importlib.import_module(f"{package_name}.jax")
        finally:
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(f"{package_name}."):
                    sys.modules.pop(module_name, None)

    def test_jax_namespace_does_not_probe_cutedsl(self):
        package_name = f"{_TEST_PACKAGE}_no_cutlass"
        parent = types.ModuleType(package_name)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = package_name
        sys.modules[package_name] = parent
        fake_jax = types.ModuleType("jax")
        fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)

        try:
            with mock.patch.dict(
                sys.modules,
                {"jax": fake_jax, "cutlass": None, "cutlass.jax": None},
            ):
                namespace = importlib.import_module(f"{package_name}.jax")
                self.assertIn("rmsnorm_rht_amax_sm100", namespace.__all__)
        finally:
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(f"{package_name}."):
                    sys.modules.pop(module_name, None)

    def test_operation_packages_use_explicit_jax_namespaces(self):
        operations = (
            (
                "rmsnorm_rht_amax",
                "rmsnorm_rht_amax_sm100",
                "RmsNormRhtAmaxResult",
            ),
            (
                "deepseek_sparse_attention.indexer_forward",
                "indexer_forward_wrapper",
                "IndexerForwardResult",
            ),
            (
                "deepseek_sparse_attention.indexer_top_k",
                "indexer_top_k_wrapper",
                "IndexerTopKResult",
            ),
        )
        package_name = f"{_TEST_PACKAGE}_explicit_jax"
        parent = types.ModuleType(package_name)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = package_name
        sys.modules[package_name] = parent

        operation_apis = {}
        for operation_path, symbol_name, jax_only_name in operations:
            operation_name = f"{package_name}.{operation_path}"
            torch_function = object()
            jax_function = object()
            jax_only_value = object()
            torch_api = types.ModuleType(f"{operation_name}.api")
            setattr(torch_api, symbol_name, torch_function)
            jax_api = types.ModuleType(f"{operation_name}.jax")
            setattr(jax_api, symbol_name, jax_function)
            setattr(jax_api, jax_only_name, jax_only_value)
            operation_apis[operation_path] = (
                symbol_name,
                jax_only_name,
                torch_function,
                jax_function,
                jax_only_value,
                torch_api,
                jax_api,
            )

        try:
            with mock.patch(
                "importlib.util.find_spec",
                side_effect=AssertionError("operation packages must not probe frameworks"),
            ):
                for operation_path, _, _ in operations:
                    with self.subTest(operation=operation_path):
                        operation_name = f"{package_name}.{operation_path}"
                        operation = importlib.import_module(operation_name)
                        (
                            symbol_name,
                            jax_only_name,
                            torch_function,
                            jax_function,
                            jax_only_value,
                            torch_api,
                            jax_api,
                        ) = operation_apis[operation_path]
                        self.assertNotIn(torch_api.__name__, sys.modules)
                        self.assertNotIn(jax_api.__name__, sys.modules)
                        self.assertEqual(operation.__all__, list(operation._API_EXPORTS))

                        with mock.patch.dict(
                            sys.modules,
                            {
                                torch_api.__name__: torch_api,
                                jax_api.__name__: jax_api,
                            },
                        ):
                            self.assertIs(getattr(operation, symbol_name), torch_function)
                            self.assertIn(symbol_name, vars(operation))
                            self.assertIs(getattr(operation, symbol_name), torch_function)
                            with self.assertRaises(AttributeError):
                                getattr(operation, jax_only_name)

                            self.assertIs(operation.api, torch_api)
                            self.assertIs(operation.jax, jax_api)
                            self.assertIs(getattr(operation.jax, symbol_name), jax_function)
                            self.assertIs(getattr(operation.jax, jax_only_name), jax_only_value)
        finally:
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(f"{package_name}."):
                    sys.modules.pop(module_name, None)

    def test_unqualified_api_does_not_fall_back_to_jax(self):
        package_name = f"{_TEST_PACKAGE}_jax_only"
        operation_name = f"{package_name}.rmsnorm_rht_amax"
        parent = types.ModuleType(package_name)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = package_name
        sys.modules[package_name] = parent

        jax_api = types.ModuleType(f"{operation_name}.jax")
        jax_api.rmsnorm_rht_amax_sm100 = object()

        try:
            with mock.patch.dict(
                sys.modules,
                {
                    f"{operation_name}.api": None,
                    jax_api.__name__: jax_api,
                },
            ):
                operation = importlib.import_module(operation_name)
                self.assertIs(operation.jax, jax_api)
                with self.assertRaises(ModuleNotFoundError):
                    operation.rmsnorm_rht_amax_sm100
        finally:
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(f"{package_name}."):
                    sys.modules.pop(module_name, None)

    def test_jax_implementations_are_colocated_with_torch_apis(self):
        operation_dirs = (
            _CUDNN_ROOT / "rmsnorm_rht_amax",
            _CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_forward",
            _CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_top_k",
        )
        for operation_dir in operation_dirs:
            with self.subTest(operation=operation_dir.name):
                self.assertTrue((operation_dir / "api.py").is_file())
                self.assertTrue((operation_dir / "jax.py").is_file())

        for old_module in ("rmsnorm_rht_amax.py", "indexer_forward.py", "indexer_top_k.py"):
            with self.subTest(old_module=old_module):
                self.assertFalse((_CUDNN_ROOT / "jax" / old_module).exists())

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
