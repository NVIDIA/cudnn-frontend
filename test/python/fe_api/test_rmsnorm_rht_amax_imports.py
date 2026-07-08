# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Import-boundary contracts for the RMSNorm framework adapters."""

import ast
import importlib
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
_OPERATION_ROOT = _CUDNN_ROOT / "rmsnorm_rht_amax"


def _load_cudnn_submodule(name: str, submodule: str):
    package = types.ModuleType(name)
    package.__path__ = [str(_CUDNN_ROOT)]
    package.__package__ = name
    sys.modules[name] = package
    return importlib.import_module(f"{name}.{submodule}")


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
        raise RuntimeError("Unable to load RMSNorm package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _remove_package(name: str) -> None:
    root_name = name.split(".", 1)[0]
    for module_name in tuple(sys.modules):
        if module_name == root_name or module_name.startswith(f"{root_name}."):
            sys.modules.pop(module_name, None)


class RmsNormImportContractTest(unittest.TestCase):
    def test_jax_facade_reports_missing_jax_dependency(self):
        name = "cudnn_jax_missing_dependency_test"
        try:
            with mock.patch.dict(sys.modules, {"jax": None}):
                with self.assertRaisesRegex(ImportError, r"nvidia-cudnn-frontend\[jax\]"):
                    _load_cudnn_submodule(name, "jax")
        finally:
            _remove_package(name)

    def test_jax_facade_reports_missing_cutlass_dependency(self):
        name = "cudnn_jax_missing_cutlass_test"
        jax = types.ModuleType("jax")
        try:
            with mock.patch.dict(sys.modules, {"jax": jax, "cutlass": None, "cutlass.jax": None}):
                with self.assertRaisesRegex(ImportError, r"nvidia-cudnn-frontend\[jax\]"):
                    _load_cudnn_submodule(name, "jax")
        finally:
            _remove_package(name)

    def test_jax_facade_reports_incompatible_jax(self):
        name = "cudnn_jax_incompatible_test"
        jax = types.ModuleType("jax")
        jax.__version__ = "0.8.0"
        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass_jax = types.ModuleType("cutlass.jax")
        cutlass_jax.is_available = lambda: False
        cutlass_jax.CUTE_DSL_MIN_SUPPORTED_JAX_VERSION = (0, 9, 1)
        cutlass.jax = cutlass_jax
        try:
            with mock.patch.dict(sys.modules, {"jax": jax, "cutlass": cutlass, "cutlass.jax": cutlass_jax}):
                with self.assertRaisesRegex(ImportError, r"JAX 0\.8\.0.*minimum supported JAX version is 0\.9\.1.*nvidia-cudnn-frontend\[jax\]"):
                    _load_cudnn_submodule(name, "jax")
        finally:
            _remove_package(name)

    def test_operation_package_does_not_advertise_jax_submodule(self):
        name = "cudnn_jax_operator_route_test.rmsnorm_rht_amax"
        try:
            package = _load_operation_package(name)
            self.assertNotIn("jax", dir(package))
        finally:
            _remove_package(name)

    def test_package_import_loads_no_framework_adapter(self):
        name = "cudnn_rmsnorm_import_test.rmsnorm_rht_amax"
        blocked = {module_name: None for module_name in ("torch", "jax", "cutlass", "cuda")}
        try:
            with mock.patch.dict(sys.modules, blocked):
                package = _load_operation_package(name)
            self.assertNotIn(f"{name}.api", sys.modules)
            self.assertNotIn(f"{name}.jax", sys.modules)
            self.assertNotIn(f"{name}.kernel", sys.modules)
            self.assertTrue(
                {
                    "RMSNormRHTAmaxKernel",
                    "RmsNormRhtAmaxSm100Op",
                    "RmsNormRhtAmaxSm100",
                    "api",
                    "kernel",
                    "op",
                }.issubset(dir(package))
            )
        finally:
            _remove_package(name)

    def test_unqualified_exports_route_only_to_torch_api(self):
        name = "cudnn_rmsnorm_torch_route_test.rmsnorm_rht_amax"
        sentinel = object()
        api = types.ModuleType(f"{name}.api")
        api.RmsNormRhtAmaxSm100 = sentinel
        try:
            sys.modules[api.__name__] = api
            package = _load_operation_package(name)
            self.assertIs(package.RmsNormRhtAmaxSm100, sentinel)
            self.assertNotIn(f"{name}.jax", sys.modules)
        finally:
            _remove_package(name)

    def test_kernel_exports_do_not_load_an_adapter(self):
        name = "cudnn_rmsnorm_kernel_route_test.rmsnorm_rht_amax"
        sentinel = object()
        kernel = types.ModuleType(f"{name}.kernel")
        kernel.RMSNormRHTAmaxKernel = sentinel
        try:
            sys.modules[kernel.__name__] = kernel
            package = _load_operation_package(name)
            self.assertIs(package.RMSNormRHTAmaxKernel, sentinel)
            self.assertNotIn(f"{name}.api", sys.modules)
            self.assertNotIn(f"{name}.jax", sys.modules)
        finally:
            _remove_package(name)

    def test_operation_exports_do_not_load_a_framework_adapter(self):
        name = "cudnn_rmsnorm_op_route_test.rmsnorm_rht_amax"
        op_sentinel = object()
        best_num_threads_sentinel = object()
        pick_rows_per_cta_sentinel = object()
        op = types.ModuleType(f"{name}.op")
        op.RmsNormRhtAmaxSm100Op = op_sentinel
        op.best_num_threads = best_num_threads_sentinel
        op.pick_rows_per_cta = pick_rows_per_cta_sentinel
        try:
            sys.modules[op.__name__] = op
            package = _load_operation_package(name)
            self.assertIs(package.RmsNormRhtAmaxSm100Op, op_sentinel)
            self.assertIs(package.best_num_threads, best_num_threads_sentinel)
            self.assertIs(package.pick_rows_per_cta, pick_rows_per_cta_sentinel)
            self.assertNotIn(f"{name}.api", sys.modules)
            self.assertNotIn(f"{name}.jax", sys.modules)
            self.assertNotIn(f"{name}.kernel", sys.modules)
        finally:
            _remove_package(name)

    def test_static_import_directions(self):
        def all_imports(filename):
            path = Path(filename)
            if not path.is_absolute():
                path = _OPERATION_ROOT / path
            tree = ast.parse(path.read_text(), filename=str(path))
            return [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

        def imports_framework(node, framework: str):
            if isinstance(node, ast.Import):
                return any(alias.name == framework or alias.name.startswith(f"{framework}.") for alias in node.names)
            return node.level == 0 and (node.module == framework or (node.module or "").startswith(f"{framework}."))

        base_op_imports = all_imports(_CUDNN_ROOT / "_op.py")
        self.assertFalse(any(imports_framework(node, framework) for node in base_op_imports for framework in ("torch", "jax", "cutlass", "cuda")))

        operation_imports = all_imports("op.py")
        self.assertFalse(any(imports_framework(node, framework) for node in operation_imports for framework in ("torch", "jax", "cutlass", "cuda")))

        kernel_imports = all_imports("kernel.py")
        self.assertFalse(any(imports_framework(node, framework) for node in kernel_imports for framework in ("torch", "jax")))

        api_imports = all_imports("api.py")
        self.assertFalse(any(imports_framework(node, "jax") for node in api_imports))

        jax_imports = all_imports("jax.py")
        self.assertFalse(any(imports_framework(node, "torch") for node in jax_imports))

        api_tree = ast.parse((_OPERATION_ROOT / "api.py").read_text(), filename="api.py")
        jax_tree = ast.parse((_OPERATION_ROOT / "jax.py").read_text(), filename="jax.py")
        jax_package_tree = ast.parse((_CUDNN_ROOT / "_jax" / "__init__.py").read_text(), filename="_jax/__init__.py")
        jax_facade_tree = ast.parse((_CUDNN_ROOT / "jax" / "__init__.py").read_text(), filename="jax/__init__.py")
        jax_base_tree = ast.parse((_CUDNN_ROOT / "_jax" / "api_base.py").read_text(), filename="_jax/api_base.py")
        op_tree = ast.parse((_OPERATION_ROOT / "op.py").read_text(), filename="op.py")
        kernel_tree = ast.parse((_OPERATION_ROOT / "kernel.py").read_text(), filename="kernel.py")
        torch_adapter = next(node for node in api_tree.body if isinstance(node, ast.ClassDef) and node.name == "RmsNormRhtAmaxSm100")
        self.assertEqual([base.id for base in torch_adapter.bases if isinstance(base, ast.Name)], ["APIBase"])
        constructor = next(node for node in torch_adapter.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        required_argument_count = len(constructor.args.args) - len(constructor.args.defaults)
        self.assertEqual(
            [argument.arg for argument in constructor.args.args[:required_argument_count]],
            ["self", "sample_x", "sample_w", "sample_o", "sample_amax"],
        )
        self.assertNotIn(
            "_kernel",
            [argument.arg for argument in (*constructor.args.args, *constructor.args.kwonlyargs)],
        )
        op_constructors = [
            node for node in ast.walk(api_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RmsNormRhtAmaxSm100Op"
        ]
        self.assertEqual(len(op_constructors), 1)
        self.assertIn(op_constructors[0], ast.walk(constructor))
        self.assertTrue({"x", "weight", "output", "amax"}.issubset({keyword.arg for keyword in op_constructors[0].keywords}))
        self.assertFalse(
            any(
                isinstance(node, ast.FunctionDef) and any(isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in node.decorator_list)
                for node in torch_adapter.body
            )
        )
        torch_check_support = next(node for node in torch_adapter.body if isinstance(node, ast.FunctionDef) and node.name == "check_support")
        self.assertTrue(any(isinstance(node, ast.Attribute) and node.attr == "check_support" for node in ast.walk(torch_check_support)))
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == "check_output" for node in ast.walk(torch_check_support)))
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr == "_op"
                for node in ast.walk(torch_check_support)
            )
        )

        jax_adapter = next(node for node in jax_tree.body if isinstance(node, ast.ClassDef) and node.name == "RmsNormRhtAmaxSm100")
        self.assertEqual([base.id for base in jax_adapter.bases if isinstance(base, ast.Name)], ["JaxApiBase"])
        self.assertFalse(jax_adapter.decorator_list)
        self.assertTrue(
            any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RmsNormRhtAmaxSm100Op" for node in ast.walk(jax_adapter))
        )
        jax_check_support = next(node for node in jax_adapter.body if isinstance(node, ast.FunctionDef) and node.name == "check_support")
        self.assertTrue(any(isinstance(node, ast.Attribute) and node.attr == "_check_device_compatibility" for node in ast.walk(jax_check_support)))
        jax_launch = next(node for node in jax_adapter.body if isinstance(node, ast.FunctionDef) and node.name == "_launch")
        self.assertTrue(
            any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RMSNormRHTAmaxKernel" for node in ast.walk(jax_launch))
        )

        jax_wrapper = next(node for node in jax_tree.body if isinstance(node, ast.FunctionDef) and node.name == "rmsnorm_rht_amax_sm100")
        jit_decorator = next(
            decorator
            for decorator in jax_wrapper.decorator_list
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "jit"
        )
        static_argnames = next(keyword.value for keyword in jit_decorator.keywords if keyword.arg == "static_argnames")
        self.assertEqual(
            tuple(element.value for element in static_argnames.elts),
            ("eps", "num_threads", "rows_per_cta"),
        )

        torch_wrapper = next(node for node in api_tree.body if isinstance(node, ast.FunctionDef) and node.name == "rmsnorm_rht_amax_wrapper_sm100")
        self.assertFalse(
            any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RMSNormRHTAmaxKernel" for node in ast.walk(torch_wrapper))
        )
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == "_materialize_outputs" for node in ast.walk(torch_wrapper)))
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute)
                and node.attr in {"infer_output_from", "from_tensor", "materialize", "_to_tensor_desc", "_materialize_tensor_desc"}
                for node in ast.walk(torch_wrapper)
            )
        )
        torch_api_construction = next(
            node for node in ast.walk(torch_wrapper) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RmsNormRhtAmaxSm100"
        )
        self.assertTrue({"sample_o", "sample_amax"}.issubset({keyword.arg for keyword in torch_api_construction.keywords}))

        torch_execute = next(node for node in torch_adapter.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
        self.assertEqual(
            [argument.arg for argument in torch_execute.args.args],
            ["self", "x_tensor", "w_tensor", "o_tensor", "amax_tensor", "current_stream"],
        )
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr in {"infer_output", "materialize"} for node in ast.walk(torch_execute)))

        kernel_class = next(node for node in kernel_tree.body if isinstance(node, ast.ClassDef) and node.name == "RMSNormRHTAmaxKernel")
        self.assertFalse(kernel_class.bases)
        launcher = next(node for node in kernel_class.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
        self.assertEqual(
            [argument.arg for argument in launcher.args.args],
            ["self", "x_tensor", "w_tensor", "o_tensor", "amax_tensor", "stream"],
        )

        compile_method = next(node for node in torch_adapter.body if isinstance(node, ast.FunctionDef) and node.name == "compile")
        kernel_constructors = [
            node for node in ast.walk(api_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RMSNormRHTAmaxKernel"
        ]
        self.assertEqual(len(kernel_constructors), 1)
        self.assertIn(kernel_constructors[0], ast.walk(compile_method))
        self.assertTrue(
            {"n", "rows_per_cta"}.issubset(
                {
                    node.attr
                    for node in ast.walk(compile_method)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "self"
                    and node.value.attr == "_op"
                }
            )
        )
        tensor_api = next(node for node in ast.walk(compile_method) if isinstance(node, ast.FunctionDef) and node.name == "tensor_api")
        compiled_call = next(
            node for node in ast.walk(tensor_api) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "compiled_kernel"
        )
        self.assertEqual(
            [argument.id for argument in compiled_call.args if isinstance(argument, ast.Name)],
            ["x_tensor", "w_tensor", "o_tensor", "amax_tensor", "stream"],
        )

        operation_class = next(node for node in op_tree.body if isinstance(node, ast.ClassDef) and node.name == "RmsNormRhtAmaxSm100Op")
        self.assertEqual([base.id for base in operation_class.bases if isinstance(base, ast.Name)], ["Op"])
        self.assertTrue(any(isinstance(node, ast.Attribute) and node.attr == "_call_kernel" for node in ast.walk(jax_adapter)))
        self.assertFalse(
            any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cutlass_call" for node in ast.walk(jax_adapter))
        )
        self.assertIn("_INSTALL_HINT", {node.id for node in ast.walk(jax_facade_tree) if isinstance(node, ast.Name)})
        self.assertTrue(any(isinstance(node, ast.Attribute) and node.attr == "is_available" for node in ast.walk(jax_facade_tree)))
        self.assertTrue(any(isinstance(node, ast.Try) for node in jax_facade_tree.body))
        self.assertTrue(any(isinstance(node, ast.Import) and any(alias.name == "jax" for alias in node.names) for node in jax_tree.body))
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and any(alias.name == "jax" for alias in node.names) for node in jax_tree.body))
        for tree in (jax_package_tree, jax_tree):
            self.assertNotIn("_INSTALL_HINT", {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
            self.assertFalse(any(isinstance(node, ast.Try) for node in tree.body))
            self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == "is_available" for node in ast.walk(tree)))
        jax_base = next(node for node in jax_base_tree.body if isinstance(node, ast.ClassDef) and node.name == "JaxApiBase")
        base_call = next(node for node in jax_base.body if isinstance(node, ast.FunctionDef) and node.name == "_call_kernel")
        cutlass_call = next(
            node for node in ast.walk(base_call) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cutlass_call"
        )
        self.assertTrue(cutlass_call.args)
        self.assertNotIn("_call_cache", {node.attr for node in ast.walk(jax_adapter) if isinstance(node, ast.Attribute)})
        self.assertNotIn("_trace_lock", {node.attr for node in ast.walk(jax_adapter) if isinstance(node, ast.Attribute)})

        root_source = (_CUDNN_ROOT / "__init__.py").read_text()
        self.assertIn("from ._op import Op", root_source)
        self.assertIn("from ._tensor_desc import TensorDesc", root_source)
        self.assertIn('if name == "jax":', root_source)
        self.assertIn('importlib.import_module(".jax", __name__)', root_source)


if __name__ == "__main__":
    unittest.main()
