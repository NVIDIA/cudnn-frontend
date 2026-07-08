# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the discrete grouped GEMM JAX adapters."""

from __future__ import annotations

from enum import Enum, auto
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
_DISCRETE_ROOT = _CUDNN_ROOT / "discrete_grouped_gemm"
_PACKAGE = "cudnn_jax_discrete_grouped_gemm_contract_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    INT32 = auto()
    INT64 = auto()
    UINT8 = auto()
    FP8_E4M3 = auto()
    FP8_E5M2 = auto()
    FP8_E8M0 = auto()
    FP4_E2M1 = auto()


_DTYPE_TO_CUDNN = {
    "float16": _DataType.HALF,
    "bfloat16": _DataType.BFLOAT16,
    "float32": _DataType.FLOAT,
    "int32": _DataType.INT32,
    "int64": _DataType.INT64,
    "uint8": _DataType.UINT8,
    "float4_e2m1fn": _DataType.FP4_E2M1,
    "float8_e4m3fn": _DataType.FP8_E4M3,
    "float8_e5m2": _DataType.FP8_E5M2,
    "float8_e8m0fnu": _DataType.FP8_E8M0,
}


class _Array:
    def __init__(self, shape, dtype, fill_value=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.fill_value = fill_value
        self.iterator = object()


class _TensorSpec:
    def __init__(self, mode=None, ptr_assumed_align=None):
        self.mode = mode
        self.ptr_assumed_align = ptr_assumed_align


class JaxDiscreteGroupedGemmContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        discrete_name = f"{_PACKAGE}.discrete_grouped_gemm"
        discrete = types.ModuleType(discrete_name)
        discrete.__path__ = [str(_DISCRETE_ROOT)]
        discrete.__package__ = discrete_name
        discrete.__spec__ = ModuleSpec(discrete_name, loader=None, is_package=True)
        sys.modules[discrete_name] = discrete

        for child in ("discrete_grouped_gemm_swiglu", "discrete_grouped_gemm_dswiglu"):
            child_name = f"{discrete_name}.{child}"
            child_module = types.ModuleType(child_name)
            child_module.__path__ = [str(_DISCRETE_ROOT / child)]
            child_module.__package__ = child_name
            child_module.__spec__ = ModuleSpec(child_name, loader=None, is_package=True)
            sys.modules[child_name] = child_module

        internal_name = f"{_PACKAGE}._jax"
        internal = types.ModuleType(internal_name)
        internal.__path__ = [str(_CUDNN_ROOT / "_jax")]
        internal.__package__ = internal_name
        internal.__spec__ = ModuleSpec(internal_name, loader=None, is_package=True)
        sys.modules[internal_name] = internal

        tensor_module = importlib.import_module(f"{_PACKAGE}._tensor_desc")
        layout_module = importlib.import_module(f"{internal_name}.layout")
        result_module = importlib.import_module(f"{_PACKAGE}._result")

        class JaxTensorDesc(tensor_module.TensorDesc):
            @property
            def cudnn_dtype(self):
                return _DTYPE_TO_CUDNN.get(self.dtype, _DataType.NOT_SET)

        class JaxApiBase:
            @staticmethod
            def _resolve_compute_capability(_target, _supported, _name):
                return 100

            @staticmethod
            def _to_tensor_desc(value, name, *, mode=None, init_value=None, **_unused):
                public_shape = tuple(value.shape)
                mode = layout_module.normalize_mode(len(public_shape), mode)
                public_order = tuple(reversed(range(len(public_shape))))
                public_stride = layout_module.compact_stride(public_shape, public_order)
                canonical_axis_by_public_axis = layout_module.to_public_axes(
                    tuple(range(len(public_shape))), mode
                )
                return JaxTensorDesc(
                    dtype=value.dtype,
                    shape=layout_module.to_canonical_axes(public_shape, mode),
                    stride=layout_module.to_canonical_axes(public_stride, mode),
                    stride_order=tuple(
                        canonical_axis_by_public_axis[axis] for axis in public_order
                    ),
                    name=name,
                    init_value=init_value,
                )

            @staticmethod
            def _check_tensor_signature(value, expected, *, mode=None):
                actual = layout_module.to_canonical_axes(tuple(value.shape), mode)
                if actual != expected.shape:
                    raise ValueError(
                        f"{expected.name} shape mismatch: expected {expected.shape}, got {actual}"
                    )
                if (
                    _DTYPE_TO_CUDNN.get(value.dtype, _DataType.NOT_SET)
                    != expected.cudnn_dtype
                ):
                    raise ValueError(f"{expected.name} dtype mismatch")

            @staticmethod
            def _to_tensor_spec(_desc, *, mode=None, **_unused):
                return _TensorSpec(mode)

            @staticmethod
            def _materialize_tensor_desc(desc, *, mode=None):
                return _Array(
                    layout_module.to_public_axes(desc.shape, mode), desc.dtype
                )

            @staticmethod
            def _get_max_active_clusters(_cluster_size, *, overlap_margin=0):
                return 8 - overlap_margin

            def _call_kernel(
                self,
                inputs,
                *,
                launch,
                output_descs,
                workspace_descs=(),
                output_spec=(),
                workspace_spec=None,
                **options,
            ):
                outputs = tuple(
                    _Array(
                        layout_module.to_public_axes(desc.shape, spec.mode), desc.dtype
                    )
                    for desc, spec in zip(output_descs, output_spec)
                )
                workspaces = tuple(
                    _Array(desc.shape, desc.dtype) for desc in workspace_descs
                )
                launch("stream", *inputs, *outputs, *workspaces)
                self.captured_kernel_call = (
                    tuple(inputs),
                    tuple(output_descs),
                    tuple(workspace_descs),
                    options,
                )
                self.captured_workspaces = workspaces
                self.captured_workspace_specs = workspace_spec
                return outputs

        internal.JaxApiBase = JaxApiBase
        internal.JaxTensorDesc = JaxTensorDesc
        internal.TupleDict = result_module.TupleDict

        datatypes = types.ModuleType(f"{internal_name}.datatypes")
        datatypes.jax_to_cudnn_dtype = lambda dtype: _DTYPE_TO_CUDNN.get(
            dtype, _DataType.NOT_SET
        )
        datatypes.normalize_jax_dtype = lambda value, default, _name: (
            default if value is None else value
        )
        sys.modules[datatypes.__name__] = datatypes

        fake_jax = types.ModuleType("jax")
        fake_jax.ShapeDtypeStruct = _Array
        fake_jax.jit = lambda function=None, **_kwargs: (
            (lambda target: target) if function is None else function
        )
        fake_jnp = types.ModuleType("jax.numpy")
        for dtype in _DTYPE_TO_CUDNN:
            setattr(fake_jnp, dtype, dtype)
        fake_jnp.dtype = lambda dtype: dtype
        fake_jnp.ones = lambda shape, dtype: _Array(shape, dtype)
        fake_jnp.empty = lambda shape, dtype: _Array(shape, dtype)
        fake_jnp.full = lambda shape, value, dtype: _Array(shape, dtype, value)
        fake_jax.numpy = fake_jnp

        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "jax": fake_jax,
                    "jax.numpy": fake_jnp,
                    "torch": None,
                    "cutlass": None,
                },
            ):
                cls.common = importlib.import_module(f"{discrete_name}._jax_common")
                cls.forward = importlib.import_module(
                    f"{discrete_name}.discrete_grouped_gemm_swiglu.jax"
                )
                cls.backward = importlib.import_module(
                    f"{discrete_name}.discrete_grouped_gemm_dswiglu.jax"
                )
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    @staticmethod
    def _scale_shape(
        rows: int, k: int, experts: int, sf_vec_size: int = 32
    ) -> tuple[int, ...]:
        canonical = (
            32,
            4,
            (rows + 127) // 128,
            4,
            ((k + sf_vec_size - 1) // sf_vec_size + 3) // 4,
            experts,
        )
        return (experts, canonical[2], canonical[4], 32, 4, 4)

    def test_forward_layouts_map_stacked_experts_to_canonical_kernel_axes(self):
        m, n, k, experts = 256, 256, 128, 4
        operation = self.forward.DiscreteGroupedGemmSwigluSm100(
            _Array((1, m, k), "float8_e4m3fn"),
            _Array((experts, n, k), "float8_e4m3fn"),
            _Array(self._scale_shape(m, k, 1), "float8_e8m0fnu"),
            _Array(self._scale_shape(n, k, experts), "float8_e8m0fnu"),
            _Array((experts,), "int32"),
            _Array((experts,), "float32"),
            sample_norm_const=_Array((1,), "float32"),
            sample_prob=_Array((1, 1, m), "float32"),
            sample_bias=_Array((experts, n), "bfloat16"),
        )

        self.assertTrue(operation.check_support())
        self.assertEqual(operation.a_desc.shape, (m, k, 1))
        self.assertEqual(operation.b_desc.shape, (n, k, experts))
        self.assertEqual(operation.b_desc.stride_order, (1, 0, 2))
        self.assertEqual(operation.c_desc.shape, (m, n, 1))
        self.assertEqual(operation.d_desc.shape, (m, n // 2, 1))
        self.assertEqual(operation.d_desc.stride_order, (1, 0, 2))
        self.assertEqual(operation.sfd_row_desc.shape, (32, 4, 2, 4, 1, 1))
        self.assertEqual(operation.amax_desc.shape, (experts, 1))
        self.assertEqual(operation.amax_desc.init_value, float("-inf"))

    def test_backward_infers_initialized_reduction_outputs(self):
        m, n, k, experts = 256, 256, 128, 4
        operation = self.backward.DiscreteGroupedGemmDswigluSm100(
            _Array((1, m, k), "float8_e4m3fn"),
            _Array((experts, n, k), "float8_e4m3fn"),
            _Array((1, m, 2 * n), "bfloat16"),
            _Array(self._scale_shape(m, k, 1), "float8_e8m0fnu"),
            _Array(self._scale_shape(n, k, experts), "float8_e8m0fnu"),
            _Array((experts,), "int32"),
            _Array((experts,), "float32"),
            _Array((experts,), "float32"),
            _Array((1, 1, m), "float32"),
            generate_dbias=True,
        )

        self.assertTrue(operation.check_support())
        self.assertEqual(operation.d_row_desc.shape, (m, 2 * n, 1))
        self.assertEqual(operation.dprob_desc.shape, (m, 1, 1))
        self.assertEqual(operation.dprob_desc.init_value, 0.0)
        self.assertEqual(operation.dbias_desc.shape, (experts, 2 * n, 1))
        self.assertEqual(operation.dbias_desc.init_value, 0.0)
        self.assertEqual(operation.amax_desc.shape, (experts, 2, 1))
        self.assertEqual(operation.amax_desc.init_value, float("-inf"))

    def test_non_k_major_stacked_weights_are_rejected(self):
        m, n, k, experts = 256, 256, 128, 4
        operation = self.forward.DiscreteGroupedGemmSwigluSm100(
            _Array((1, m, k), "float8_e4m3fn"),
            _Array((experts, k, n), "float8_e4m3fn"),
            _Array(self._scale_shape(m, k, 1), "float8_e8m0fnu"),
            _Array(self._scale_shape(n, k, experts), "float8_e8m0fnu"),
            _Array((experts,), "int32"),
            _Array((experts,), "float32"),
            b_layout="LKN",
        )

        with self.assertRaisesRegex(ValueError, "K-major A and B"):
            operation.check_support()

    def test_launcher_passes_live_stacked_operands_before_outputs_and_workspace(self):
        m, n, k, experts = 256, 256, 128, 4
        a = _Array((1, m, k), "float8_e4m3fn")
        b = _Array((experts, n, k), "float8_e4m3fn")
        sfa = _Array(self._scale_shape(m, k, 1), "float8_e8m0fnu")
        sfb = _Array(self._scale_shape(n, k, experts), "float8_e8m0fnu")
        offsets = _Array((experts,), "int32")
        alpha = _Array((experts,), "float32")
        operation = self.forward.DiscreteGroupedGemmSwigluSm100(
            a, b, sfa, sfb, offsets, alpha
        )

        captured = {}

        class FakeKernel:
            def __init__(self, **config):
                captured["config"] = config

            @staticmethod
            def get_workspace_bytes():
                return 512

            def __call__(self, *args):
                captured["args"] = args

        kernel_module_name = f"{_PACKAGE}.discrete_grouped_gemm.discrete_grouped_gemm_swiglu.discrete_B_blockscaled_grouped_gemm_glu_bias"
        kernel_module = types.ModuleType(kernel_module_name)
        kernel_module.BlockScaledDiscreteWeightGroupedGemmBiasKernel = FakeKernel

        cutlass = types.ModuleType("cutlass")
        cutlass.Int32 = lambda value: ("i32", value)
        cutlass.Int64 = lambda value: ("i64", value)
        cutlass.Float32 = lambda value: ("f32", value)
        cute = types.ModuleType("cutlass.cute")
        nvgpu = types.ModuleType("cutlass.cute.nvgpu")
        nvgpu.OperandMajorMode = types.SimpleNamespace(K="K")
        cutlass_jax = types.ModuleType("cutlass.jax")
        cutlass_jax.TensorSpec = _TensorSpec
        cutlass_jax.jax_to_cutlass_dtype = lambda dtype: dtype
        cutlass.cute = cute

        with mock.patch.dict(
            sys.modules,
            {
                "cutlass": cutlass,
                "cutlass.cute": cute,
                "cutlass.cute.nvgpu": nvgpu,
                "cutlass.jax": cutlass_jax,
                kernel_module_name: kernel_module,
            },
        ):
            result = operation(a, b, sfa, sfb, offsets, alpha)

        kernel_args = captured["args"]
        self.assertIs(kernel_args[0], a)
        self.assertIs(kernel_args[1], b.iterator)
        self.assertIs(kernel_args[2], sfb.iterator)
        self.assertEqual(kernel_args[6], "K")
        self.assertIs(kernel_args[7], operation.captured_workspaces[0].iterator)
        self.assertEqual(operation.captured_workspaces[0].shape, (512,))
        self.assertEqual(operation.captured_workspace_specs[0].ptr_assumed_align, 128)
        self.assertEqual(kernel_args[18].shape, (1, 1, m))
        self.assertEqual(kernel_args[18].dtype, "float32")
        self.assertEqual(
            tuple(result.keys()),
            (
                "c_tensor",
                "d_tensor",
                "d_col_tensor",
                "amax_tensor",
                "sfd_row_tensor",
                "sfd_col_tensor",
            ),
        )
        self.assertEqual(kernel_args[20], 8)
        self.assertEqual(kernel_args[21], "stream")
        self.assertTrue(captured["config"]["stacked_expert_inputs"])

    def test_zero_m_materializes_results_without_launching(self):
        m, n, k, experts = 0, 256, 128, 4
        a = _Array((1, m, k), "float8_e4m3fn")
        b = _Array((experts, n, k), "float8_e4m3fn")
        sfa = _Array(self._scale_shape(m, k, 1), "float8_e8m0fnu")
        sfb = _Array(self._scale_shape(n, k, experts), "float8_e8m0fnu")
        offsets = _Array((experts,), "int32")
        alpha = _Array((experts,), "float32")

        forward = self.forward.DiscreteGroupedGemmSwigluSm100(
            a, b, sfa, sfb, offsets, alpha
        )
        forward_result = forward(a, b, sfa, sfb, offsets, alpha)
        self.assertEqual(forward_result["c_tensor"].shape, (1, 0, n))
        self.assertEqual(forward_result["d_tensor"].shape, (1, 0, n // 2))
        self.assertEqual(forward_result["amax_tensor"].fill_value, float("-inf"))

        c = _Array((1, m, 2 * n), "bfloat16")
        beta = _Array((experts,), "float32")
        prob = _Array((1, 1, m), "float32")
        backward = self.backward.DiscreteGroupedGemmDswigluSm100(
            a,
            b,
            c,
            sfa,
            sfb,
            offsets,
            alpha,
            beta,
            prob,
            generate_dbias=True,
        )
        backward_result = backward(a, b, c, sfa, sfb, offsets, alpha, beta, prob)
        self.assertEqual(backward_result["d_row_tensor"].shape, (1, 0, 2 * n))
        self.assertEqual(backward_result["dprob_tensor"].shape, (1, 1, 0))
        self.assertEqual(backward_result["dbias_tensor"].fill_value, 0.0)
        self.assertEqual(backward_result["amax_tensor"].fill_value, float("-inf"))
        self.assertEqual(
            tuple(backward_result.keys()),
            (
                "d_row_tensor",
                "d_col_tensor",
                "dprob_tensor",
                "dbias_tensor",
                "amax_tensor",
                "sfd_row_tensor",
                "sfd_col_tensor",
            ),
        )


if __name__ == "__main__":
    unittest.main()
