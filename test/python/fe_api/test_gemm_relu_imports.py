# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Import boundaries for dense GEMM + sReLU and dsReLU."""

import ast
import importlib.util
from pathlib import Path
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


_CUDNN_ROOT = Path(__file__).resolve().parents[3] / "python" / "cudnn"


def _load_operation_package(name: str, operation: str):
    parent_name = name.rpartition(".")[0]
    parent = types.ModuleType(parent_name)
    parent.__path__ = [str(_CUDNN_ROOT)]
    parent.__package__ = parent_name
    sys.modules[parent_name] = parent

    operation_root = _CUDNN_ROOT / operation
    spec = importlib.util.spec_from_file_location(
        name,
        operation_root / "__init__.py",
        submodule_search_locations=[str(operation_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {operation} package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _remove_package(name: str) -> None:
    root_name = name.split(".", 1)[0]
    for module_name in tuple(sys.modules):
        if module_name == root_name or module_name.startswith(f"{root_name}."):
            sys.modules.pop(module_name, None)


class GemmReluImportContractTest(unittest.TestCase):
    def test_public_jax_facade_routes_both_operations(self):
        tree = ast.parse((_CUDNN_ROOT / "jax" / "__init__.py").read_text(), filename="cudnn/jax/__init__.py")
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_OPERATION_EXPORTS" for target in node.targets)
        )
        exports = ast.literal_eval(assignment.value)
        self.assertEqual(exports["GemmSreluSm100"], ("..gemm_srelu.jax", "GemmSreluSm100"))
        self.assertEqual(exports["gemm_srelu_wrapper_sm100"], ("..gemm_srelu.jax", "gemm_srelu_wrapper_sm100"))
        self.assertEqual(exports["GemmDsreluSm100"], ("..gemm_dsrelu.jax", "GemmDsreluSm100"))
        self.assertEqual(exports["gemm_dsrelu_wrapper_sm100"], ("..gemm_dsrelu.jax", "gemm_dsrelu_wrapper_sm100"))

    def test_package_imports_are_framework_free_and_lazy(self):
        blocked = {module_name: None for module_name in ("torch", "jax", "cutlass", "cuda")}
        for operation, op_class, api_class, wrapper in (
            ("gemm_srelu", "GemmSreluSm100Op", "GemmSreluSm100", "gemm_srelu_wrapper_sm100"),
            ("gemm_dsrelu", "GemmDsreluSm100Op", "GemmDsreluSm100", "gemm_dsrelu_wrapper_sm100"),
        ):
            name = f"cudnn_{operation}_import_test.{operation}"
            try:
                with self.subTest(operation=operation), mock.patch.dict(sys.modules, blocked):
                    package = _load_operation_package(name, operation)
                    self.assertNotIn(f"{name}.api", sys.modules)
                    self.assertNotIn(f"{name}.op", sys.modules)
                    self.assertTrue({op_class, api_class, wrapper, "api", "op"}.issubset(dir(package)))
            finally:
                _remove_package(name)

    def test_exports_route_to_op_and_torch_modules_independently(self):
        for operation, op_class, api_class, wrapper in (
            ("gemm_srelu", "GemmSreluSm100Op", "GemmSreluSm100", "gemm_srelu_wrapper_sm100"),
            ("gemm_dsrelu", "GemmDsreluSm100Op", "GemmDsreluSm100", "gemm_dsrelu_wrapper_sm100"),
        ):
            name = f"cudnn_{operation}_route_test.{operation}"
            op_sentinel = object()
            api_sentinel = object()
            op_module = types.ModuleType(f"{name}.op")
            setattr(op_module, op_class, op_sentinel)
            api_module = types.ModuleType(f"{name}.api")
            setattr(api_module, api_class, api_sentinel)
            setattr(api_module, wrapper, object())
            try:
                with self.subTest(operation=operation):
                    sys.modules[op_module.__name__] = op_module
                    sys.modules[api_module.__name__] = api_module
                    package = _load_operation_package(name, operation)
                    self.assertIs(getattr(package, op_class), op_sentinel)
                    self.assertIs(getattr(package, api_class), api_sentinel)
            finally:
                _remove_package(name)

    def test_framework_neutral_op_modules_have_no_framework_imports(self):
        paths = [
            _CUDNN_ROOT / "gemm" / "__init__.py",
            _CUDNN_ROOT / "gemm" / "helpers.py",
            _CUDNN_ROOT / "gemm" / "srelu.py",
            _CUDNN_ROOT / "gemm_srelu" / "op.py",
            _CUDNN_ROOT / "gemm_dsrelu" / "op.py",
        ]
        for path in paths:
            imports = [node for node in ast.walk(ast.parse(path.read_text(), filename=str(path))) if isinstance(node, (ast.Import, ast.ImportFrom))]
            imported = {alias.name.split(".", 1)[0] for node in imports if isinstance(node, ast.Import) for alias in node.names}
            imported.update((node.module or "").split(".", 1)[0] for node in imports if isinstance(node, ast.ImportFrom) and node.level == 0)
            with self.subTest(path=path.name):
                self.assertTrue(imported.isdisjoint({"torch", "jax", "cutlass", "cuda"}))


if __name__ == "__main__":
    unittest.main()
