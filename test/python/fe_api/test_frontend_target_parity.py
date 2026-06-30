# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contract tests for Torch-first FE-OSS target support."""

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
_PYTHON_ROOT = _REPO_ROOT / "python"
_CUDNN_ROOT = _PYTHON_ROOT / "cudnn"
_FE_OSS_DOCS_ROOT = _REPO_ROOT / "docs" / "fe-oss-apis"
_TEST_PACKAGE = "cudnn_frontend_parity_contract_test"
_REQUIRED_DEFAULT = "<required>"
_TAB_SET_PATTERN = re.compile(r"(?ms)^(?P<fence>`{4,})\{tab-set\}\n(?P<body>.*?)^(?P=fence)$")
_TAB_ITEM_PATTERN = re.compile(r"(?ms)^(?P<fence>`{3,})\{tab-item\} (?P<label>[^\n]+)\n" r"(?P<body>.*?)^(?P=fence)$")


def _load_frontend_package():
    parent = types.ModuleType(_TEST_PACKAGE)
    parent.__path__ = [str(_CUDNN_ROOT)]
    parent.__package__ = _TEST_PACKAGE
    sys.modules[_TEST_PACKAGE] = parent
    return importlib.import_module(f"{_TEST_PACKAGE}.frontend")


_FRONTEND = _load_frontend_package()
_REGISTRY_MODULE = importlib.import_module(f"{_TEST_PACKAGE}.frontend._registry")


def _binding(
    module,
    symbol,
    *,
    parameter_map=None,
    output_map=None,
    target_only_parameters=(),
):
    return _REGISTRY_MODULE.TargetBinding(
        module=module,
        symbol=symbol,
        parameter_map=parameter_map or {"x": "x"},
        output_map=output_map or {"output": "output"},
        target_only_parameters=target_only_parameters,
    )


def _aligned_bindings(name):
    return {
        _REGISTRY_MODULE.FrontendTarget.TORCH: _binding("example.torch", name),
        _REGISTRY_MODULE.FrontendTarget.JAX: _binding("example.jax", name),
    }


def _base_name(base):
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _decorator_name(decorator):
    parts = []
    while isinstance(decorator, ast.Attribute):
        parts.append(decorator.attr)
        decorator = decorator.value
    if isinstance(decorator, ast.Name):
        parts.append(decorator.id)
    return ".".join(reversed(parts))


def _discover_python_api_anchors():
    anchors = set()
    for path in sorted(_CUDNN_ROOT.rglob("api.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        module = ".".join(path.with_suffix("").relative_to(_PYTHON_ROOT).parts)
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

        def is_api_class(name, seen=()):
            if name in seen:
                return False
            for base in classes[name].bases:
                parent = _base_name(base)
                if parent == "APIBase":
                    return True
                if parent in classes and is_api_class(parent, seen + (name,)):
                    return True
            return False

        for name in classes:
            if not name.startswith("_") and is_api_class(name):
                anchors.add(f"{module}:{name}")

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_") and "wrapper" in node.name:
                anchors.add(f"{module}:{node.name}")
    return anchors


def _discover_cute_kernel_anchors():
    anchors = set()

    class KernelVisitor(ast.NodeVisitor):
        def __init__(self, module):
            self.module = module
            self.scope = []

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node):
            if any(_decorator_name(decorator) == "cute.kernel" for decorator in node.decorator_list):
                qualified_name = ".".join((*self.scope, node.name))
                anchors.add(f"{self.module}:{qualified_name}")
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    for path in sorted(_CUDNN_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        module = ".".join(path.with_suffix("").relative_to(_PYTHON_ROOT).parts)
        KernelVisitor(module).visit(tree)
    return anchors


def _ast_function_parameters(path, function_name):
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name)
    arguments = function.args
    parameters = {}

    positional = (*arguments.posonlyargs, *arguments.args)
    positional_defaults = (_REQUIRED_DEFAULT,) * (len(positional) - len(arguments.defaults)) + tuple(ast.dump(default) for default in arguments.defaults)
    for argument, default in zip(positional, positional_defaults):
        parameters[argument.arg] = ("positional", default)

    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        parameters[argument.arg] = (
            "keyword_only",
            _REQUIRED_DEFAULT if default is None else ast.dump(default),
        )
    return parameters


class FrontendTargetParityTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    def test_registered_operation_keeps_torch_canonical_and_jax_optional(self):
        operations = _FRONTEND.registered_operations()
        self.assertEqual(
            [operation.name for operation in operations],
            ["rmsnorm_rht_amax_sm100"],
        )

        operation = operations[0]
        torch_binding = operation.status(_REGISTRY_MODULE.FrontendTarget.TORCH)
        jax_binding = operation.status(_REGISTRY_MODULE.FrontendTarget.JAX)
        self.assertIsInstance(torch_binding, _REGISTRY_MODULE.TargetBinding)
        self.assertIsInstance(jax_binding, _REGISTRY_MODULE.TargetBinding)
        self.assertEqual(
            torch_binding.qualified_name,
            "cudnn.rmsnorm_rht_amax.api:rmsnorm_rht_amax_sm100",
        )
        self.assertEqual(torch_binding.symbol, operation.name)
        self.assertEqual(jax_binding.symbol, operation.name)
        self.assertEqual(torch_binding.target_only_parameters, ("current_stream",))
        self.assertEqual(
            dict(torch_binding.output_map),
            {"output": "output", "amax": "amax"},
        )
        self.assertEqual(
            dict(jax_binding.output_map),
            {"output": "output", "amax": "amax"},
        )
        self.assertEqual(operation.parity_case, "rmsnorm_rht_amax")
        self.assertEqual(_REGISTRY_MODULE.FRONTEND_OPERATION_REGISTRY.audit(), ())
        self.assertEqual(
            _REGISTRY_MODULE.FRONTEND_OPERATION_REGISTRY.audit(require_jax_complete=True),
            (),
        )

    def test_optional_jax_namespace_does_not_eagerly_import_frameworks(self):
        optional_roots = ("jax", "cutlass", "torch")
        before = {name for name in sys.modules if name.split(".", 1)[0] in optional_roots}

        jax_namespace = importlib.import_module(f"{_TEST_PACKAGE}.jax")

        after = {name for name in sys.modules if name.split(".", 1)[0] in optional_roots}
        self.assertEqual(after - before, set())
        self.assertTrue(callable(jax_namespace.rmsnorm_rht_amax_sm100))
        self.assertEqual(
            jax_namespace.RmsNormRhtAmaxResult._fields,
            ("output", "amax"),
        )
        self.assertNotIn(f"{_TEST_PACKAGE}.rmsnorm_rht_amax.api", sys.modules)

    def test_aligned_torch_name_is_exported_from_the_canonical_namespace(self):
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

        operation = _FRONTEND.registered_operations()[0]
        self.assertEqual(
            lazy_imports[operation.name],
            (".rmsnorm_rht_amax", operation.name),
        )

        package_tree = ast.parse(
            (_CUDNN_ROOT / "rmsnorm_rht_amax" / "__init__.py").read_text(),
            filename=str(_CUDNN_ROOT / "rmsnorm_rht_amax" / "__init__.py"),
        )
        package_exports_node = next(
            node.value
            for node in package_tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        )
        self.assertIn(operation.name, ast.literal_eval(package_exports_node))

    def test_jax_optional_extra_is_framework_named_and_torch_free(self):
        pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
        optional_dependencies = re.search(
            r"(?ms)^\[project\.optional-dependencies\]\n(?P<body>.*?)(?=^\[|\Z)",
            pyproject,
        )
        self.assertIsNotNone(optional_dependencies)
        body = optional_dependencies.group("body")
        self.assertRegex(body, r"(?m)^jax = \[$")
        self.assertNotRegex(body, r"(?m)^cutedsl-jax = \[$")

        jax_dependencies = re.search(
            r"(?ms)^jax = \[\n(?P<body>.*?)^\]$",
            body,
        )
        self.assertIsNotNone(jax_dependencies)
        self.assertNotIn('"torch"', jax_dependencies.group("body"))

    def test_catalog_resolves_target_bindings_lazily(self):
        def torch_impl(
            x,
            weight,
            *,
            eps=1e-5,
            num_threads=None,
            rows_per_cta=None,
            current_stream=None,
        ):
            return x, weight, current_stream

        def jax_impl(
            x,
            weight,
            *,
            eps=1e-5,
            num_threads=None,
            rows_per_cta=None,
        ):
            return x, weight

        modules = {
            "cudnn.rmsnorm_rht_amax.api": types.SimpleNamespace(rmsnorm_rht_amax_sm100=torch_impl),
            "cudnn.jax.rmsnorm_rht_amax": types.SimpleNamespace(rmsnorm_rht_amax_sm100=jax_impl),
        }
        operation = _FRONTEND.registered_operations()[0]
        with mock.patch.object(
            _REGISTRY_MODULE.importlib,
            "import_module",
            side_effect=lambda name: modules[name],
        ):
            self.assertIs(
                operation.resolve(_REGISTRY_MODULE.FrontendTarget.TORCH),
                torch_impl,
            )
            self.assertIs(
                operation.resolve(_REGISTRY_MODULE.FrontendTarget.JAX),
                jax_impl,
            )

    def test_semantic_default_drift_is_rejected(self):
        def drifted_jax_impl(
            x,
            weight,
            *,
            eps=1e-4,
            num_threads=None,
            rows_per_cta=None,
        ):
            return x, weight

        operation = _FRONTEND.registered_operations()[0]
        with mock.patch.object(
            _REGISTRY_MODULE.importlib,
            "import_module",
            return_value=types.SimpleNamespace(rmsnorm_rht_amax_sm100=drifted_jax_impl),
        ):
            with self.assertRaisesRegex(TypeError, "default.*eps.*drifted"):
                operation.resolve(_REGISTRY_MODULE.FrontendTarget.JAX)

    def test_semantic_parameter_kind_drift_is_rejected(self):
        def drifted_jax_impl(
            x,
            weight,
            eps=1e-5,
            num_threads=None,
            rows_per_cta=None,
        ):
            return x, weight

        operation = _FRONTEND.registered_operations()[0]
        with mock.patch.object(
            _REGISTRY_MODULE.importlib,
            "import_module",
            return_value=types.SimpleNamespace(rmsnorm_rht_amax_sm100=drifted_jax_impl),
        ):
            with self.assertRaisesRegex(TypeError, "parameter kind.*eps.*drifted"):
                operation.resolve(_REGISTRY_MODULE.FrontendTarget.JAX)

    def test_operation_ownership_cannot_overlap(self):
        for duplicate_kind in ("api", "kernel"):
            with self.subTest(duplicate_kind=duplicate_kind):
                local_registry = _REGISTRY_MODULE.FrontendOperationRegistry()

                @_REGISTRY_MODULE.frontend_operation(
                    name="first",
                    targets=_aligned_bindings("first"),
                    api_anchors=("example.api:First",),
                    kernel_anchors=("example.kernel:First.kernel",),
                    output_names=("output",),
                    parity_case="first",
                    registry=local_registry,
                )
                def first(x):
                    return x

                second_api = "example.api:First" if duplicate_kind == "api" else "example.api:Second"
                second_kernel = "example.kernel:First.kernel" if duplicate_kind == "kernel" else "example.kernel:Second.kernel"
                with self.assertRaisesRegex(ValueError, "overlapping ownership"):

                    @_REGISTRY_MODULE.frontend_operation(
                        name="second",
                        targets=_aligned_bindings("second"),
                        api_anchors=(second_api,),
                        kernel_anchors=(second_kernel,),
                        output_names=("output",),
                        parity_case="second",
                        registry=local_registry,
                    )
                    def second(x):
                        return x

                self.assertEqual(first("value"), "value")

    def test_normalized_target_names_cannot_be_declared_twice(self):
        local_registry = _REGISTRY_MODULE.FrontendOperationRegistry()

        with self.assertRaisesRegex(ValueError, "same target more than once"):

            @_REGISTRY_MODULE.frontend_operation(
                name="duplicate_jax",
                targets={
                    _REGISTRY_MODULE.FrontendTarget.TORCH: _binding("example.torch", "duplicate_jax"),
                    _REGISTRY_MODULE.FrontendTarget.JAX: _binding("example.jax", "duplicate_jax"),
                    "JAX": _binding("example.other_jax", "duplicate_jax"),
                },
                api_anchors=("example.api:Example",),
                kernel_anchors=("example.kernel:Example.kernel",),
                output_names=("output",),
                parity_case="duplicate_jax",
                registry=local_registry,
            )
            def duplicate_jax(x):
                return x

    def test_missing_jax_target_requires_an_explicit_gap(self):
        local_registry = _REGISTRY_MODULE.FrontendOperationRegistry()

        with self.assertRaisesRegex(ValueError, "missing=.*jax"):

            @_REGISTRY_MODULE.frontend_operation(
                name="missing_jax",
                targets={_REGISTRY_MODULE.FrontendTarget.TORCH: _binding("example", "missing_jax")},
                api_anchors=("example.api:Example",),
                kernel_anchors=("example.kernel:Example.kernel",),
                output_names=("output",),
                parity_case=None,
                registry=local_registry,
            )
            def missing_jax(x):
                return x

    def test_torch_cannot_be_declared_as_optional_gap(self):
        local_registry = _REGISTRY_MODULE.FrontendOperationRegistry()

        with self.assertRaisesRegex(ValueError, "canonical Torch binding"):

            @_REGISTRY_MODULE.frontend_operation(
                name="missing_torch",
                targets={
                    _REGISTRY_MODULE.FrontendTarget.TORCH: _REGISTRY_MODULE.TargetGap(
                        reason="not implemented",
                        tracking_issue="https://example.invalid/issue/1",
                    ),
                    _REGISTRY_MODULE.FrontendTarget.JAX: _binding("example", "missing_torch"),
                },
                api_anchors=("example.api:Example",),
                kernel_anchors=("example.kernel:Example.kernel",),
                output_names=("output",),
                parity_case="missing_torch",
                registry=local_registry,
            )
            def missing_torch(x):
                return x

    def test_declared_jax_gap_is_visible_but_not_a_structural_failure(self):
        local_registry = _REGISTRY_MODULE.FrontendOperationRegistry()

        @_REGISTRY_MODULE.frontend_operation(
            name="known_gap",
            targets={
                _REGISTRY_MODULE.FrontendTarget.TORCH: _binding("example", "known_gap"),
                _REGISTRY_MODULE.FrontendTarget.JAX: _REGISTRY_MODULE.TargetGap(
                    reason="not migrated",
                    tracking_issue="https://example.invalid/issue/1",
                ),
            },
            api_anchors=("example.api:Example",),
            kernel_anchors=("example.kernel:Example.kernel",),
            output_names=("output",),
            parity_case=None,
            registry=local_registry,
        )
        def known_gap(x):
            return x

        self.assertEqual(known_gap("value"), "value")
        self.assertEqual(local_registry.audit(), ())
        issues = local_registry.audit(require_jax_complete=True)
        self.assertEqual(len(issues), 1)
        self.assertIn("known_gap:jax", issues[0])
        with self.assertRaisesRegex(NotImplementedError, "not migrated"):
            local_registry.get("known_gap").resolve(_REGISTRY_MODULE.FrontendTarget.JAX)

    def test_jax_binding_requires_a_parity_case(self):
        local_registry = _REGISTRY_MODULE.FrontendOperationRegistry()

        with self.assertRaisesRegex(ValueError, "no registered parity case"):

            @_REGISTRY_MODULE.frontend_operation(
                name="missing_parity_case",
                targets={
                    _REGISTRY_MODULE.FrontendTarget.TORCH: _binding("example.torch", "missing_parity_case"),
                    _REGISTRY_MODULE.FrontendTarget.JAX: _binding("example.jax", "missing_parity_case"),
                },
                api_anchors=("example.api:Example",),
                kernel_anchors=("example.kernel:Example.kernel",),
                output_names=("output",),
                parity_case=None,
                registry=local_registry,
            )
            def missing_parity_case(x):
                return x

    def test_target_symbols_must_match_the_semantic_operation_name(self):
        local_registry = _REGISTRY_MODULE.FrontendOperationRegistry()

        with self.assertRaisesRegex(ValueError, "binding symbol must match"):

            @_REGISTRY_MODULE.frontend_operation(
                name="aligned_name",
                targets={
                    _REGISTRY_MODULE.FrontendTarget.TORCH: _binding("example.torch", "legacy_wrapper_name"),
                    _REGISTRY_MODULE.FrontendTarget.JAX: _binding("example.jax", "aligned_name"),
                },
                api_anchors=("example.api:Example",),
                kernel_anchors=("example.kernel:Example.kernel",),
                output_names=("output",),
                parity_case="aligned_name",
                registry=local_registry,
            )
            def aligned_name(x):
                return x

    def test_all_discovered_python_apis_are_registered_or_baselined(self):
        discovered = _discover_python_api_anchors()
        registered = {anchor for operation in _FRONTEND.registered_operations() for anchor in operation.api_anchors}
        gaps = set(_FRONTEND.known_jax_gaps())

        self.assertFalse(registered & gaps)
        self.assertEqual(discovered, registered | gaps)

    def test_registered_jax_gaps_cannot_grow_silently(self):
        registered_gap_ids = {
            operation.name
            for operation in _FRONTEND.registered_operations()
            if isinstance(
                operation.status(_REGISTRY_MODULE.FrontendTarget.JAX),
                _REGISTRY_MODULE.TargetGap,
            )
        }

        self.assertEqual(
            registered_gap_ids,
            set(_FRONTEND.known_registered_jax_gap_ids()),
        )

    def test_all_cute_kernels_have_a_semantic_operation_owner(self):
        discovered = _discover_cute_kernel_anchors()
        registered = {anchor for operation in _FRONTEND.registered_operations() for anchor in operation.kernel_anchors}
        gaps = set(_FRONTEND.known_kernel_ownership_gaps())

        self.assertFalse(registered & gaps)
        self.assertEqual(discovered, registered | gaps)

    def test_target_source_signatures_match_semantic_mappings(self):
        operation = _FRONTEND.registered_operations()[0]
        contract_path = _CUDNN_ROOT / "frontend" / "rmsnorm_rht_amax.py"
        contract_parameters = _ast_function_parameters(
            contract_path,
            "_rmsnorm_rht_amax_sm100_contract",
        )
        target_sources = {
            _REGISTRY_MODULE.FrontendTarget.TORCH: (
                _CUDNN_ROOT / "rmsnorm_rht_amax" / "api.py",
                "rmsnorm_rht_amax_sm100",
            ),
            _REGISTRY_MODULE.FrontendTarget.JAX: (
                _CUDNN_ROOT / "jax" / "rmsnorm_rht_amax.py",
                "rmsnorm_rht_amax_sm100",
            ),
        }

        self.assertEqual(
            set(contract_parameters),
            {"x", "weight", "eps", "num_threads", "rows_per_cta"},
        )
        for target, (path, function_name) in target_sources.items():
            with self.subTest(target=target.value):
                binding = operation.status(target)
                self.assertIsInstance(binding, _REGISTRY_MODULE.TargetBinding)
                target_parameters = _ast_function_parameters(path, function_name)
                expected_target_names = set(binding.parameter_map.values()) | set(binding.target_only_parameters)
                self.assertEqual(set(target_parameters), expected_target_names)
                for semantic_name, target_name in binding.parameter_map.items():
                    self.assertEqual(
                        contract_parameters[semantic_name][1],
                        target_parameters[target_name][1],
                    )

        torch_parameters = _ast_function_parameters(*target_sources[_REGISTRY_MODULE.FrontendTarget.TORCH])
        self.assertEqual(
            tuple(torch_parameters),
            (
                "x",
                "weight",
                "eps",
                "num_threads",
                "rows_per_cta",
                "current_stream",
            ),
        )
        for target in _REGISTRY_MODULE.FrontendTarget:
            target_parameters = _ast_function_parameters(*target_sources[target])
            self.assertEqual(target_parameters["x"][0], "positional")
            self.assertEqual(target_parameters["weight"][0], "positional")
            for name in ("eps", "num_threads", "rows_per_cta"):
                self.assertEqual(target_parameters[name][0], "keyword_only")
        self.assertEqual(torch_parameters["current_stream"][0], "keyword_only")

        legacy_parameters = _ast_function_parameters(
            _CUDNN_ROOT / "rmsnorm_rht_amax" / "api.py",
            "rmsnorm_rht_amax_wrapper_sm100",
        )
        self.assertEqual(
            tuple(legacy_parameters),
            ("x_tensor", "w_tensor", "eps", "num_threads", "rows_per_cta", "current_stream"),
        )
        self.assertTrue(all(kind == "positional" for kind, _ in legacy_parameters.values()))

    def test_available_jax_binding_has_target_lifecycle_tests(self):
        test_names_by_path = {}
        for path in (
            _REPO_ROOT / "test" / "python" / "fe_api" / "test_rmsnorm_rht_amax.py",
            _REPO_ROOT / "test" / "python" / "fe_api" / "test_jax_rmsnorm_rht_amax.py",
        ):
            tree = ast.parse(path.read_text(), filename=str(path))
            test_names_by_path[path.name] = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

        for operation in _FRONTEND.registered_operations():
            if not isinstance(
                operation.status(_REGISTRY_MODULE.FrontendTarget.JAX),
                _REGISTRY_MODULE.TargetBinding,
            ):
                continue
            case = operation.parity_case
            self.assertIn(
                f"test_{case}_wrapper",
                test_names_by_path["test_rmsnorm_rht_amax.py"],
            )
            self.assertIn(
                f"test_{case}_aligned_api",
                test_names_by_path["test_rmsnorm_rht_amax.py"],
            )
            self.assertIn(
                f"test_jax_{case}_jit",
                test_names_by_path["test_jax_rmsnorm_rht_amax.py"],
            )

    def test_available_jax_binding_has_torch_default_documentation_tabs(self):
        documentation = {path: path.read_text() for path in sorted(_FE_OSS_DOCS_ROOT.rglob("*.md"))}

        for operation in _FRONTEND.registered_operations():
            if not isinstance(
                operation.status(_REGISTRY_MODULE.FrontendTarget.JAX),
                _REGISTRY_MODULE.TargetBinding,
            ):
                continue

            torch_import = f"from cudnn import {operation.name}"
            jax_import = f"from cudnn.jax import {operation.name}"
            matching_tab_sets = []
            for path, contents in documentation.items():
                for tab_set_match in _TAB_SET_PATTERN.finditer(contents):
                    tab_set_body = tab_set_match.group("body")
                    tab_items = [(match.group("label"), match.group("body")) for match in _TAB_ITEM_PATTERN.finditer(tab_set_body)]
                    by_label = dict(tab_items)
                    if torch_import in by_label.get("PyTorch", "") and jax_import in by_label.get("JAX", ""):
                        matching_tab_sets.append((path, tab_set_body, tab_items))

            self.assertEqual(
                len(matching_tab_sets),
                1,
                f"{operation.name} must have exactly one paired PyTorch/JAX " "documentation tab set",
            )
            path, tab_set_body, tab_items = matching_tab_sets[0]
            self.assertEqual(
                [label for label, _ in tab_items],
                ["PyTorch", "JAX"],
                f"PyTorch must be the first tab in {path}",
            )
            self.assertIn(":sync-group: frontend-framework", tab_set_body)
            self.assertIn(":sync: torch", tab_items[0][1])
            self.assertIn(":selected:", tab_items[0][1])
            self.assertIn(":sync: jax", tab_items[1][1])
            self.assertNotIn(":selected:", tab_items[1][1])


if __name__ == "__main__":
    unittest.main()
