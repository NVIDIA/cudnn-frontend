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


def _identity_jit(fn=None, **_kwargs):
    return (lambda decorated_fn: decorated_fn) if fn is None else fn


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"
_TEST_PACKAGE = "cudnn_frontend_jax_surface_test"


def _colocated_jax_files():
    """Discover operation-local JAX APIs without maintaining an inventory."""

    facade_dir = _CUDNN_ROOT / "jax"
    return tuple(path for path in sorted(_CUDNN_ROOT.rglob("jax.py")) if path.parent != facade_dir)


def _operation_path(jax_file):
    return ".".join(jax_file.parent.relative_to(_CUDNN_ROOT).parts)


def _literal_assignment(path, name):
    tree = ast.parse(path.read_text(), filename=str(path))
    value = next(
        node.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    return ast.literal_eval(value)


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
        fake_cutlass_jax.is_available = lambda: True
        fake_cutlass_jax.TensorSpec = type("TensorSpec", (), {})
        fake_cutlass_jax.jax_to_cutlass_dtype = lambda dtype: dtype
        fake_cutlass = types.ModuleType("cutlass")
        fake_cutlass.__path__ = []
        fake_cutlass.Constexpr = object
        fake_cutlass_cute = types.ModuleType("cutlass.cute")
        fake_cutlass_cute.jit = _identity_jit
        fake_cutlass.cute = fake_cutlass_cute
        fake_cutlass.jax = fake_cutlass_jax

        return mock.patch.dict(
            sys.modules,
            {
                "jax": fake_jax,
                "jax.numpy": fake_jnp,
                "cutlass": fake_cutlass,
                "cutlass.cute": fake_cutlass_cute,
                "cutlass.jax": fake_cutlass_jax,
                "torch": None,
            },
        )

    def test_explicit_namespace_loads_jax_without_torch(self):
        parent = types.ModuleType(_TEST_PACKAGE)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = _TEST_PACKAGE
        sys.modules[_TEST_PACKAGE] = parent

        torch_before = {name for name, module in sys.modules.items() if name.split(".", 1)[0] == "torch" and module is not None}
        with self._optional_modules():
            jax_namespace = importlib.import_module(f"{_TEST_PACKAGE}.jax")
            operation_modules = {}
            for jax_file in _colocated_jax_files():
                operation_path = _operation_path(jax_file)
                module_name = f"{_TEST_PACKAGE}.{operation_path}.jax"
                self.assertIn(module_name, sys.modules)
                operation_modules[operation_path] = sys.modules[module_name]
            self.assertIn(f"{_TEST_PACKAGE}._jax.cutedsl", sys.modules)

            expected_exports = {"DSA", "NSA"}
            expected_dsa_exports = set()
            expected_nsa_exports = set()
            torch_dsa_exports = set(
                _literal_assignment(
                    _CUDNN_ROOT / "deepseek_sparse_attention" / "__init__.py",
                    "_SYMBOLS",
                )
            )
            torch_nsa_exports = set(
                _literal_assignment(
                    _CUDNN_ROOT / "native_sparse_attention" / "__init__.py",
                    "_SYMBOLS",
                )
            )
            for operation_path, operation_module in operation_modules.items():
                expected_exports.update(operation_module.__all__)
                operation_package = importlib.import_module(f"{_TEST_PACKAGE}.{operation_path}")
                torch_exports = set(operation_package._API_EXPORTS)
                shared_exports = torch_exports.intersection(operation_module.__all__)
                for name in shared_exports:
                    self.assertNotIn(name, vars(operation_package))
                    self.assertIs(getattr(operation_package.jax, name), getattr(jax_namespace, name))
                    self.assertTrue(callable(getattr(jax_namespace, name)))
                    if name in torch_dsa_exports:
                        expected_dsa_exports.add(name)
                    if name in torch_nsa_exports:
                        expected_nsa_exports.add(name)

                loaded_descendants = {name for name in sys.modules if name.startswith(f"{_TEST_PACKAGE}.{operation_path}.")}
                self.assertLessEqual(
                    loaded_descendants,
                    {
                        f"{_TEST_PACKAGE}.{operation_path}.config",
                        f"{_TEST_PACKAGE}.{operation_path}.jax",
                    },
                )

            self.assertEqual(set(jax_namespace.__all__), expected_exports)
            self.assertEqual(set(vars(jax_namespace.DSA)), expected_dsa_exports)
            self.assertEqual(set(vars(jax_namespace.NSA)), expected_nsa_exports)
            for name in expected_dsa_exports:
                self.assertIs(getattr(jax_namespace.DSA, name), getattr(jax_namespace, name))
            for name in expected_nsa_exports:
                self.assertIs(getattr(jax_namespace.NSA, name), getattr(jax_namespace, name))

            torch_after = {name for name, module in sys.modules.items() if name.split(".", 1)[0] == "torch" and module is not None}
            self.assertEqual(torch_after - torch_before, set())

    def test_colocated_jax_modules_can_load_before_facade(self):
        package_name = f"{_TEST_PACKAGE}_direct_first"
        parent = types.ModuleType(package_name)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = package_name
        sys.modules[package_name] = parent

        try:
            with self._optional_modules():
                operation_modules = []
                for jax_file in _colocated_jax_files():
                    module = importlib.import_module(f"{package_name}.{_operation_path(jax_file)}.jax")
                    operation_modules.append(module)
                    self.assertNotIn(f"{package_name}.jax", sys.modules)

                facade = importlib.import_module(f"{package_name}.jax")
                compatibility_adapter = importlib.import_module(f"{package_name}.jax.cutedsl")
                internal_adapter = importlib.import_module(f"{package_name}._jax.cutedsl")
                self.assertIs(compatibility_adapter.BufferSpec, internal_adapter.BufferSpec)
                self.assertIs(compatibility_adapter.call_cutedsl, internal_adapter.call_cutedsl)

                for operation_module in operation_modules:
                    for name in operation_module.__all__:
                        self.assertIs(getattr(facade, name), getattr(operation_module, name))
        finally:
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(f"{package_name}."):
                    sys.modules.pop(module_name, None)

    def test_missing_jax_reports_optional_extra(self):
        package_name = f"{_TEST_PACKAGE}_missing_jax"
        parent = types.ModuleType(package_name)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = package_name
        sys.modules[package_name] = parent

        try:
            with mock.patch.dict(sys.modules, {"jax": None, "jax.numpy": None}):
                with self.assertRaisesRegex(
                    ImportError,
                    r"requires the 'jax' module.*nvidia-cudnn-frontend\[jax\]",
                ):
                    importlib.import_module(f"{package_name}.jax")
        finally:
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(f"{package_name}."):
                    sys.modules.pop(module_name, None)

    def test_missing_cutedsl_reports_optional_extra(self):
        package_name = f"{_TEST_PACKAGE}_no_cutlass"
        parent = types.ModuleType(package_name)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = package_name
        sys.modules[package_name] = parent
        fake_jax = types.ModuleType("jax")
        fake_jnp = types.ModuleType("jax.numpy")
        fake_jax.__path__ = []
        fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        fake_jax.numpy = fake_jnp

        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "jax": fake_jax,
                    "jax.numpy": fake_jnp,
                    "cutlass": None,
                    "cutlass.jax": None,
                },
            ):
                with self.assertRaisesRegex(
                    ImportError,
                    r"requires the 'cutlass' module.*nvidia-cudnn-frontend\[jax\]",
                ):
                    importlib.import_module(f"{package_name}.jax")
        finally:
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(f"{package_name}."):
                    sys.modules.pop(module_name, None)

    def test_unavailable_cutlass_jax_reports_minimum_version(self):
        package_name = f"{_TEST_PACKAGE}_unavailable_cutlass_jax"
        parent = types.ModuleType(package_name)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = package_name
        sys.modules[package_name] = parent

        fake_jax = types.ModuleType("jax")
        fake_jax.__version__ = "0.4.0"
        fake_jax.version = types.SimpleNamespace(__version_info__=(0, 4, 0))
        fake_cutlass = types.ModuleType("cutlass")
        fake_cutlass.__path__ = []
        fake_cutlass_jax = types.ModuleType("cutlass.jax")
        fake_cutlass_jax.CUTE_DSL_MIN_SUPPORTED_JAX_VERSION = (0, 5, 0)
        fake_cutlass_jax.is_available = lambda: False
        fake_cutlass.jax = fake_cutlass_jax

        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "jax": fake_jax,
                    "cutlass": fake_cutlass,
                    "cutlass.jax": fake_cutlass_jax,
                },
            ):
                with self.assertRaisesRegex(
                    ImportError,
                    r"CUTLASS JAX support is unavailable with JAX 0\.4\.0; " r"the minimum supported JAX version is 0\.5\.0.*nvidia-cudnn-frontend\[jax\]",
                ):
                    importlib.import_module(f"{package_name}.jax")
        finally:
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(f"{package_name}."):
                    sys.modules.pop(module_name, None)

    def test_operation_packages_use_explicit_jax_namespaces(self):
        operations = []
        for jax_file in _colocated_jax_files():
            operation_path = _operation_path(jax_file)
            torch_exports = set(_literal_assignment(jax_file.parent / "__init__.py", "_API_EXPORTS"))
            jax_exports = set(_literal_assignment(jax_file, "__all__"))
            operations.append(
                (
                    operation_path,
                    sorted(torch_exports.intersection(jax_exports)),
                    sorted(jax_exports.difference(torch_exports)),
                )
            )

        package_name = f"{_TEST_PACKAGE}_explicit_jax"
        parent = types.ModuleType(package_name)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = package_name
        sys.modules[package_name] = parent

        operation_apis = {}
        for operation_path, shared_names, jax_only_names in operations:
            self.assertTrue(shared_names, f"{operation_path} has no aligned Torch/JAX symbol")
            self.assertTrue(jax_only_names, f"{operation_path} has no JAX-specific result type")
            operation_name = f"{package_name}.{operation_path}"
            torch_api = types.ModuleType(f"{operation_name}.api")
            jax_api = types.ModuleType(f"{operation_name}.jax")
            torch_values = {name: object() for name in shared_names}
            jax_values = {name: object() for name in (*shared_names, *jax_only_names)}
            for name, value in torch_values.items():
                setattr(torch_api, name, value)
            for name, value in jax_values.items():
                setattr(jax_api, name, value)
            operation_apis[operation_path] = (
                shared_names,
                jax_only_names,
                torch_values,
                jax_values,
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
                            shared_names,
                            jax_only_names,
                            torch_values,
                            jax_values,
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
                            for name in shared_names:
                                self.assertIs(getattr(operation, name), torch_values[name])
                                self.assertIn(name, vars(operation))
                            for name in jax_only_names:
                                with self.assertRaises(AttributeError):
                                    getattr(operation, name)

                            self.assertIs(operation.api, torch_api)
                            self.assertIs(operation.jax, jax_api)
                            for name, value in jax_values.items():
                                self.assertIs(getattr(operation.jax, name), value)
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
        jax_files = _colocated_jax_files()
        self.assertTrue(jax_files)
        for jax_file in jax_files:
            operation_dir = jax_file.parent
            with self.subTest(operation=operation_dir.name):
                self.assertTrue((operation_dir / "api.py").is_file())
                self.assertTrue((operation_dir / "__init__.py").is_file())

        self.assertEqual(
            {path.name for path in (_CUDNN_ROOT / "jax").glob("*.py")},
            {"__init__.py", "cutedsl.py"},
        )

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

    def test_root_lazily_imports_explicit_jax_namespace(self):
        top_level_tree = ast.parse(
            (_CUDNN_ROOT / "__init__.py").read_text(),
            filename=str(_CUDNN_ROOT / "__init__.py"),
        )
        getattr_node = next(node for node in top_level_tree.body if isinstance(node, ast.FunctionDef) and node.name == "__getattr__")
        jax_branch = next(
            node
            for node in getattr_node.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "name"
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "jax"
        )
        imported_modules = {
            call.args[0].value
            for call in ast.walk(jax_branch)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "import_module"
            and call.args
            and isinstance(call.args[0], ast.Constant)
        }
        self.assertEqual(imported_modules, {".jax"})

    def test_dsa_operator_names_match_torch_namespace(self):
        dsa_tree = ast.parse(
            (_CUDNN_ROOT / "deepseek_sparse_attention" / "__init__.py").read_text(),
            filename=str(_CUDNN_ROOT / "deepseek_sparse_attention" / "__init__.py"),
        )
        symbols_node = next(
            node.value
            for node in dsa_tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_SYMBOLS" for target in node.targets)
        )
        torch_symbols = ast.literal_eval(symbols_node)
        facade_tree = ast.parse(
            (_CUDNN_ROOT / "jax" / "__init__.py").read_text(),
            filename=str(_CUDNN_ROOT / "jax" / "__init__.py"),
        )
        dsa_assignment = next(
            node
            for node in facade_tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "DSA" for target in node.targets)
        )
        jax_dsa_symbols = {keyword.arg for keyword in dsa_assignment.value.keywords}
        self.assertTrue(jax_dsa_symbols)
        self.assertLessEqual(jax_dsa_symbols, set(torch_symbols))

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
