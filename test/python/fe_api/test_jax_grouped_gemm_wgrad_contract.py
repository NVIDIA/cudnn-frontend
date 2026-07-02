# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for JAX grouped GEMM weight gradients."""

from __future__ import annotations

import ast
from enum import Enum
from importlib import import_module
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


def _identity_jit(fn=None, **_kwargs):
    return (lambda decorated_fn: decorated_fn) if fn is None else fn


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"
_TEST_PACKAGE = "cudnn_frontend_jax_grouped_wgrad_contract_test"


class _DType:
    def __init__(self, name: str, itemsize: int):
        self.name = name
        self.itemsize = itemsize

    def __repr__(self):
        return self.name


class _Array:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _TensorSpec:
    def __init__(
        self,
        *,
        layout=None,
        mode=None,
        static=None,
        ptr_assumed_align=256,
        divisibility=None,
    ):
        self.layout = layout
        self.mode = mode
        self.static = static
        self.ptr_assumed_align = ptr_assumed_align
        self.divisibility = divisibility


class _WeightMode(Enum):
    DENSE = "dense"
    DISCRETE = "discrete"


class _InputOrder(Enum):
    Tensor2D = "tensor2d"
    TensorRagged = "tensor_ragged"


class _HardwareInfo:
    def get_max_active_clusters(self, cluster_size):
        self.cluster_size = cluster_size
        return 12


class _Kernel:
    MMA_TILER_M = (128, 256)
    MMA_TILER_N = (128, 256)
    TWO_CTA_MMA_TILER_M = 256
    FP8_SF_VEC_SIZE = 32
    instances = []

    @classmethod
    def require_mma_tiler(cls, mma_tiler_mn):
        value = tuple(mma_tiler_mn)
        if value[0] not in cls.MMA_TILER_M or value[1] not in cls.MMA_TILER_N:
            raise ValueError("unsupported mma_tiler_mn")
        return value

    @classmethod
    def require_cluster_shape(cls, cluster_shape_mn, *, mma_tiler_mn):
        value = tuple(cluster_shape_mn)
        if mma_tiler_mn[0] == cls.TWO_CTA_MMA_TILER_M and value[0] % 2:
            raise ValueError("cluster_shape_mn[0] must be divisible by 2")
        return value

    @staticmethod
    def require_input_order(input_order):
        if isinstance(input_order, _InputOrder):
            return input_order
        return _InputOrder(input_order)

    @classmethod
    def get_dense_workspace_bytes(cls, expert_cnt, input_order="tensor2d"):
        order = cls.require_input_order(input_order)
        slots = 4 if order is _InputOrder.TensorRagged else 2
        return slots * expert_cnt * 128

    def __init__(self, **configuration):
        self.configuration = configuration
        self.calls = []
        self.instances.append(self)

    def __call__(self, *args):
        self.calls.append(args)


class JaxGroupedGemmWgradContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.float8_e4m3fn = _DType("float8_e4m3fn", 1)
        cls.float8_e5m2 = _DType("float8_e5m2", 1)
        cls.float8_e8m0fnu = _DType("float8_e8m0fnu", 1)
        cls.bfloat16 = _DType("bfloat16", 2)
        cls.float16 = _DType("float16", 2)
        cls.float32 = _DType("float32", 4)
        cls.int32 = _DType("int32", 4)
        cls.uint8 = _DType("uint8", 1)

        cls.fake_jnp = types.ModuleType("jax.numpy")
        for name in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e8m0fnu",
            "bfloat16",
            "float16",
            "float32",
            "int32",
            "uint8",
        ):
            setattr(cls.fake_jnp, name, getattr(cls, name))
        cls.fake_jnp.dtype = lambda value: value

        cls.fake_jax = types.ModuleType("jax")
        cls.fake_jax.__path__ = []
        cls.fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        cls.fake_jax.numpy = cls.fake_jnp
        cls.fake_jax.tree_util = types.SimpleNamespace(
            DictKey=lambda key: key,
            register_pytree_with_keys=lambda *_args: None,
        )
        cls.fake_jax.ShapeDtypeStruct = lambda shape, dtype: (shape, dtype)

        cls.fake_cutlass = types.ModuleType("cutlass")
        cls.fake_cutlass.__path__ = []
        cls.fake_cutlass.Constexpr = object
        cls.fake_cutlass.utils = types.SimpleNamespace(HardwareInfo=_HardwareInfo)
        cls.fake_cutlass_cute = types.ModuleType("cutlass.cute")
        cls.fake_cutlass_cute.jit = _identity_jit
        cls.fake_cutlass.cute = cls.fake_cutlass_cute
        cls.fake_cutlass_jax = types.ModuleType("cutlass.jax")
        cls.fake_cutlass_jax.TensorSpec = _TensorSpec
        cls.fake_cutlass_jax.cutlass_call = None
        cls.fake_cutlass_jax.jax_to_cutlass_dtype = lambda dtype: f"cutlass.{dtype.name}"
        cls.fake_cutlass.jax = cls.fake_cutlass_jax

        cls.kernel_module_name = f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_wgrad." "moe_blockscaled_grouped_gemm_wgrad"
        cls.kernel_module = types.ModuleType(cls.kernel_module_name)
        cls.kernel_module.BlockScaledMoEGroupedGemmWgradKernel = _Kernel

        cls.moe_utils_module_name = f"{_TEST_PACKAGE}.grouped_gemm.moe_utils"
        cls.moe_utils_module = types.ModuleType(cls.moe_utils_module_name)
        cls.moe_utils_module.MoEWeightMode = _WeightMode
        cls.moe_utils_module.WGradInputOrder = _InputOrder

        package_paths = {
            _TEST_PACKAGE: _CUDNN_ROOT,
            f"{_TEST_PACKAGE}.grouped_gemm": _CUDNN_ROOT / "grouped_gemm",
            f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_wgrad": (_CUDNN_ROOT / "grouped_gemm" / "grouped_gemm_wgrad"),
        }
        for package_name, package_path in package_paths.items():
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            package.__package__ = package_name
            sys.modules[package_name] = package

        with cls._optional_modules():
            cls.module = import_module(f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_wgrad.jax")

    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    def setUp(self):
        _Kernel.instances.clear()

    @classmethod
    def _optional_modules(cls, *, include_kernel=False):
        modules = {
            "jax": cls.fake_jax,
            "jax.numpy": cls.fake_jnp,
            "cutlass": cls.fake_cutlass,
            "cutlass.cute": cls.fake_cutlass_cute,
            "cutlass.jax": cls.fake_cutlass_jax,
        }
        if include_kernel:
            modules[cls.kernel_module_name] = cls.kernel_module
            modules[cls.moe_utils_module_name] = cls.moe_utils_module
        return mock.patch.dict(sys.modules, modules)

    @classmethod
    def _inputs(cls):
        return (
            _Array((384, 640), cls.float8_e4m3fn),
            _Array((640, 640), cls.float8_e4m3fn),
            _Array((384, 20), cls.float8_e8m0fnu),
            _Array((640, 20), cls.float8_e8m0fnu),
            _Array((2,), cls.int32),
        )

    @staticmethod
    def _fake_call(captured):
        def call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=tuple(inputs), **options)
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        return call

    def test_kernel_module_is_lazy(self):
        self.assertNotIn(self.kernel_module_name, sys.modules)

    def test_dense_output_and_workspace_are_xla_owned(self):
        captured = {}
        inputs = self._inputs()
        with (
            self._optional_modules(include_kernel=True),
            mock.patch.object(
                self.module,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            result = self.module.grouped_gemm_wgrad_wrapper_sm100(*inputs)

        self.assertEqual(result["wgrad_tensor"].shape, (2, 384, 640))
        self.assertIs(result["wgrad_tensor"].dtype, self.bfloat16)
        self.assertEqual(captured["inputs"], inputs)
        self.assertTrue(captured["use_static_tensors"])
        self.assertIs(captured["launcher"], self.module._launch)
        self.assertEqual(
            captured["static_args"],
            {
                "acc_dtype": self.float32,
                "mma_tiler_mn": (256, 256),
                "cluster_shape_mn": (2, 1),
                "sf_vec_size": 32,
                "accumulate_on_output": False,
                "expert_cnt": 2,
                "input_order": "tensor2d",
                "has_global_scale": False,
                "cluster_overlap_margin": 0,
            },
        )

        (output,) = captured["outputs"]
        self.assertEqual(
            (output.name, output.shape, output.dtype, output.fill_value),
            ("wgrad_tensor", (2, 384, 640), self.bfloat16, None),
        )
        self.assertEqual(output.tensor_spec.layout, (2, 1, 0))
        (workspace,) = captured["workspaces"]
        self.assertEqual(
            (workspace.name, workspace.shape, workspace.dtype),
            ("workspace", (512,), self.uint8),
        )
        self.assertEqual(workspace.tensor_spec.ptr_assumed_align, 128)
        self.assertEqual(
            [spec.layout if spec is not None else None for spec in captured["input_specs"]],
            [(1, 0), (0, 1), (1, 0), (1, 0), None],
        )

    def test_global_scales_remain_runtime_inputs(self):
        captured = {}
        global_a = _Array((2,), self.float32)
        global_b = _Array((2,), self.float32)
        with (
            self._optional_modules(include_kernel=True),
            mock.patch.object(
                self.module,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            self.module.grouped_gemm_wgrad_wrapper_sm100(
                *self._inputs(),
                global_scale_a=global_a,
                global_scale_b=global_b,
            )

        self.assertEqual(captured["inputs"][-2:], (global_a, global_b))
        self.assertEqual(len(captured["input_specs"]), 7)
        self.assertTrue(captured["static_args"]["has_global_scale"])

    def test_accumulating_and_ragged_modes_use_initialized_storage(self):
        captured = {}
        with (
            self._optional_modules(include_kernel=True),
            mock.patch.object(
                self.module,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            self.module.grouped_gemm_wgrad_wrapper_sm100(
                *self._inputs(),
                accumulate_on_output=True,
                input_order="tensor_ragged",
            )

        self.assertEqual(captured["outputs"][0].fill_value, 0.0)
        self.assertEqual(captured["workspaces"][0].shape, (1024,))
        self.assertTrue(captured["static_args"]["accumulate_on_output"])
        self.assertEqual(captured["static_args"]["input_order"], "tensor_ragged")

    def test_launcher_preserves_kernel_abi(self):
        placeholders = [object() for _ in range(10)]
        with self._optional_modules(include_kernel=True):
            self.module._launch(
                *placeholders,
                acc_dtype=self.float32,
                mma_tiler_mn=(256, 256),
                cluster_shape_mn=(2, 1),
                sf_vec_size=32,
                accumulate_on_output=False,
                expert_cnt=2,
                input_order="tensor2d",
                has_global_scale=True,
                cluster_overlap_margin=1,
            )

        kernel = _Kernel.instances[-1]
        self.assertEqual(
            kernel.configuration,
            {
                "sf_vec_size": 32,
                "acc_dtype": "cutlass.float32",
                "use_2cta_instrs": True,
                "mma_tiler_mn": (256, 256),
                "cluster_shape_mn": (2, 1),
                "accumulate_on_output": False,
                "expert_cnt": 2,
                "weight_mode": _WeightMode.DENSE,
                "input_order": _InputOrder.Tensor2D,
            },
        )
        args = kernel.calls[-1]
        self.assertEqual(
            args[:7],
            (
                placeholders[1],
                placeholders[2],
                placeholders[3],
                placeholders[4],
                placeholders[8],
                placeholders[5],
                placeholders[9],
            ),
        )
        self.assertEqual(args[7], 11)
        self.assertIs(args[8], placeholders[0])
        self.assertIs(args[9], placeholders[6])
        self.assertIs(args[10], placeholders[7])
        self.assertIsNone(args[11])

    def test_rejects_non_fp8_bad_scales_and_partial_global_scale(self):
        a, b, sfa, sfb, offsets = self._inputs()
        with (
            self._optional_modules(include_kernel=True),
            self.assertRaisesRegex(ValueError, "a_tensor.dtype"),
        ):
            self.module.grouped_gemm_wgrad_wrapper_sm100(
                _Array(a.shape, self.bfloat16),
                b,
                sfa,
                sfb,
                offsets,
            )

        with (
            self._optional_modules(include_kernel=True),
            self.assertRaisesRegex(ValueError, "sfa_tensor must have shape"),
        ):
            self.module.grouped_gemm_wgrad_wrapper_sm100(
                a,
                b,
                _Array((384, 16), self.float8_e8m0fnu),
                sfb,
                offsets,
            )

        with (
            self._optional_modules(include_kernel=True),
            self.assertRaisesRegex(ValueError, "must be provided together"),
        ):
            self.module.grouped_gemm_wgrad_wrapper_sm100(
                a,
                b,
                sfa,
                sfb,
                offsets,
                global_scale_a=_Array((2,), self.float32),
            )

    def test_kernel_capabilities_and_child_exports_are_literal(self):
        kernel_path = _CUDNN_ROOT / "grouped_gemm" / "grouped_gemm_wgrad" / "moe_blockscaled_grouped_gemm_wgrad.py"
        tree = ast.parse(kernel_path.read_text())
        kernel_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BlockScaledMoEGroupedGemmWgradKernel")
        constants = {
            target.id: ast.literal_eval(node.value)
            for node in kernel_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id in {"MMA_TILER_M", "MMA_TILER_N", "FP8_SF_VEC_SIZE"}
        }
        self.assertEqual(
            constants,
            {
                "MMA_TILER_M": (128, 256),
                "MMA_TILER_N": (128, 256),
                "FP8_SF_VEC_SIZE": 32,
            },
        )
        methods = {node.name for node in kernel_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertTrue({"require_mma_tiler", "require_cluster_shape", "get_dense_workspace_bytes"} <= methods)

        init_path = kernel_path.with_name("__init__.py")
        init_tree = ast.parse(init_path.read_text())
        exports = next(
            node.value
            for node in init_tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_API_EXPORTS" for target in node.targets)
        )
        self.assertEqual(
            ast.literal_eval(exports),
            (
                "GroupedGemmWgradSm100",
                "grouped_gemm_wgrad_wrapper_sm100",
            ),
        )


if __name__ == "__main__":
    unittest.main()
