# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Import and Torch-adapter contracts for dense GEMM + amax."""

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
_OPERATION_ROOT = _CUDNN_ROOT / "gemm_amax"


def _load_operation_package(name: str):
    parent_name, separator, _ = name.rpartition(".")
    if not separator:
        raise ValueError("Synthetic operation package must have a parent")
    parent = types.ModuleType(parent_name)
    parent.__path__ = [str(_CUDNN_ROOT)]
    parent.__package__ = parent_name
    sys.modules[parent_name] = parent

    spec = importlib.util.spec_from_file_location(
        name,
        _OPERATION_ROOT / "__init__.py",
        submodule_search_locations=[str(_OPERATION_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load GEMM + amax package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _remove_package(name: str) -> None:
    root_name = name.split(".", 1)[0]
    for module_name in tuple(sys.modules):
        if module_name == root_name or module_name.startswith(f"{root_name}."):
            sys.modules.pop(module_name, None)


class GemmAmaxImportContractTest(unittest.TestCase):
    def test_package_import_loads_no_framework_adapter(self):
        name = "cudnn_gemm_amax_import_test.gemm_amax"
        blocked = {module_name: None for module_name in ("torch", "jax", "cutlass", "cuda")}
        try:
            with mock.patch.dict(sys.modules, blocked):
                package = _load_operation_package(name)
            self.assertNotIn(f"{name}.api", sys.modules)
            self.assertNotIn(f"{name}.op", sys.modules)
            self.assertTrue(
                {
                    "GemmAmaxSm100Op",
                    "GemmAmaxSm100",
                    "gemm_amax_wrapper_sm100",
                    "api",
                    "op",
                }.issubset(dir(package))
            )
        finally:
            _remove_package(name)

    def test_static_adapter_boundaries_and_public_torch_signatures(self):
        def imports_framework(node, framework: str):
            if isinstance(node, ast.Import):
                return any(alias.name == framework or alias.name.startswith(f"{framework}.") for alias in node.names)
            return node.level == 0 and (node.module == framework or (node.module or "").startswith(f"{framework}."))

        op_tree = ast.parse((_OPERATION_ROOT / "op.py").read_text(), filename="op.py")
        api_tree = ast.parse((_OPERATION_ROOT / "api.py").read_text(), filename="api.py")
        op_imports = [node for node in ast.walk(op_tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        api_imports = [node for node in ast.walk(api_tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(imports_framework(node, framework) for node in op_imports for framework in ("torch", "jax", "cutlass", "cuda")))
        self.assertFalse(any(imports_framework(node, "jax") for node in api_imports))

        adapter = next(node for node in api_tree.body if isinstance(node, ast.ClassDef) and node.name == "GemmAmaxSm100")
        constructor = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        self.assertEqual(
            [argument.arg for argument in constructor.args.args],
            [
                "self",
                "sample_a",
                "sample_b",
                "sample_sfa",
                "sample_sfb",
                "sample_c",
                "sample_amax",
                "acc_dtype",
                "mma_tiler_mn",
                "cluster_shape_mn",
                "sf_vec_size",
            ],
        )
        op_constructors = [
            node for node in ast.walk(constructor) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "GemmAmaxSm100Op"
        ]
        self.assertEqual(len(op_constructors), 1)

        execute = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
        self.assertEqual(
            [argument.arg for argument in execute.args.args],
            [
                "self",
                "a_tensor",
                "b_tensor",
                "sfa_tensor",
                "sfb_tensor",
                "c_tensor",
                "amax_tensor",
                "current_stream",
            ],
        )

        wrapper = next(node for node in api_tree.body if isinstance(node, ast.FunctionDef) and node.name == "gemm_amax_wrapper_sm100")
        self.assertEqual(
            [argument.arg for argument in wrapper.args.args],
            [
                "a_tensor",
                "b_tensor",
                "sfa_tensor",
                "sfb_tensor",
                "c_major",
                "c_dtype",
                "acc_dtype",
                "mma_tiler_mn",
                "cluster_shape_mn",
                "sf_vec_size",
                "stream",
            ],
        )
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name.startswith("_check_torch_support") for node in adapter.body))


if __name__ == "__main__":
    unittest.main()
