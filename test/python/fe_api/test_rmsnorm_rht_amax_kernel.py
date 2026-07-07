# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the RMSNorm CuTe kernel."""

import ast
import importlib
from importlib.machinery import ModuleSpec
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
_OPERATION_ROOT = _CUDNN_ROOT / "rmsnorm_rht_amax"
_PACKAGE = "cudnn_rmsnorm_kernel_test"


def _identity_decorator(function=None, **_kwargs):
    if function is None:
        return lambda decorated: decorated
    return function


def _module(name: str, *, package: bool = False, **attributes):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _cutlass_stubs() -> dict[str, types.ModuleType]:
    driver = _module("cuda.bindings.driver", CUstream=object)
    bindings = _module("cuda.bindings", package=True, driver=driver)
    cuda = _module("cuda", package=True, bindings=bindings)

    arch = _module("cutlass.cute.arch", shuffle_sync_bfly=lambda value, **_kwargs: value)
    cute = _module(
        "cutlass.cute",
        package=True,
        arch=arch,
        kernel=_identity_decorator,
        jit=_identity_decorator,
    )
    utils = _module("cutlass.utils")
    llvm = _module(
        "cutlass._mlir.dialects.llvm",
        AsmDialect=types.SimpleNamespace(AD_ATT=object()),
    )
    dialects = _module("cutlass._mlir.dialects", package=True, llvm=llvm)
    mlir = _module("cutlass._mlir", package=True, dialects=dialects)
    cutlass_dsl = _module(
        "cutlass.cutlass_dsl",
        T=types.SimpleNamespace(f32=lambda: None),
        dsl_user_op=_identity_decorator,
    )
    cutlass = _module(
        "cutlass",
        package=True,
        cute=cute,
        utils=utils,
        Float32=object,
        Int32=object,
        BFloat16=object,
    )

    return {
        module.__name__: module
        for module in (
            cuda,
            bindings,
            driver,
            cutlass,
            cute,
            arch,
            utils,
            mlir,
            dialects,
            llvm,
            cutlass_dsl,
        )
    }


class RmsNormRhtAmaxKernelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        sys.modules[_PACKAGE] = root

        operation_name = f"{_PACKAGE}.rmsnorm_rht_amax"
        operation = types.ModuleType(operation_name)
        operation.__path__ = [str(_OPERATION_ROOT)]
        operation.__package__ = operation_name
        operation.__spec__ = ModuleSpec(operation_name, loader=None, is_package=True)
        sys.modules[operation_name] = operation

        try:
            with mock.patch.dict(sys.modules, _cutlass_stubs()):
                cls.kernel_module = importlib.import_module(f"{operation_name}.kernel")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def test_kernel_consumes_only_resolved_configuration(self):
        kernel = self.kernel_module.RMSNormRHTAmaxKernel(
            n=2048,
            num_threads=128,
            eps=1e-4,
            rows_per_cta=2,
        )

        self.assertEqual(kernel.n, 2048)
        self.assertEqual(kernel.num_threads, 128)
        self.assertEqual(kernel.eps, 1e-4)
        self.assertEqual(kernel.rows_per_cta, 2)
        self.assertEqual(kernel.ept, 16)
        self.assertEqual(kernel.vec_size, 8)
        self.assertEqual(kernel.num_vec_blocks, 2)
        self.assertEqual(kernel.warps_per_row, 4)
        self.assertEqual(kernel.tiler_mn, (1, 2048))

    def test_kernel_module_has_no_operation_or_framework_dependency(self):
        tree = ast.parse((_OPERATION_ROOT / "kernel.py").read_text(), filename="kernel.py")
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

        imported_modules = set()
        for node in imports:
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            else:
                if node.module:
                    imported_modules.add(node.module)
                imported_modules.update(alias.name for alias in node.names)

        self.assertNotIn("torch", imported_modules)
        self.assertNotIn("jax", imported_modules)
        self.assertNotIn("_op", imported_modules)
        self.assertNotIn("_tensor_desc", imported_modules)
        self.assertNotIn("data_type", imported_modules)

        kernel_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RMSNormRHTAmaxKernel")
        self.assertFalse(kernel_class.bases)
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name == "check_support" for node in kernel_class.body))
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name.startswith("infer_") for node in kernel_class.body))

    def test_hadamard_block_matches_operation_contract(self):
        tree = ast.parse((_OPERATION_ROOT / "op.py").read_text(), filename="op.py")
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "HAD_BLOCK" for target in node.targets)
        )

        self.assertEqual(ast.literal_eval(assignment.value), self.kernel_module.RMSNormRHTAmaxKernel.HAD_BLOCK)


if __name__ == "__main__":
    unittest.main()
