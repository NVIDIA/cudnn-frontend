# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the JAX GEMM + amax adapter."""

from __future__ import annotations

from enum import Enum, auto
import importlib
from importlib.machinery import ModuleSpec
import os
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
_PACKAGE = "cudnn_jax_gemm_amax_contract_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    INT8 = auto()
    UINT8 = auto()
    FP4_E2M1 = auto()
    FP8_E4M3 = auto()
    FP8_E5M2 = auto()
    FP8_E8M0 = auto()


_DTYPE_TO_CUDNN = {
    "float16": _DataType.HALF,
    "bfloat16": _DataType.BFLOAT16,
    "float32": _DataType.FLOAT,
    "float4_e2m1fn": _DataType.FP4_E2M1,
    "float8_e4m3fn": _DataType.FP8_E4M3,
    "float8_e5m2": _DataType.FP8_E5M2,
    "float8_e8m0fnu": _DataType.FP8_E8M0,
}
_CUDNN_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CUDNN.items()}


class _Array:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _TensorSpec:
    def __init__(self, *, layout, mode, divisibility=None):
        self.layout = layout
        self.mode = mode
        self.divisibility = divisibility


class JaxGemmAmaxContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        operation_name = f"{_PACKAGE}.gemm_amax"
        operation = types.ModuleType(operation_name)
        operation.__path__ = [str(_OPERATION_ROOT)]
        operation.__package__ = operation_name
        operation.__spec__ = ModuleSpec(operation_name, loader=None, is_package=True)
        sys.modules[operation_name] = operation

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
            def _to_tensor_desc(
                value,
                name,
                *,
                mode=None,
                public_stride_order=None,
                init_value=None,
            ):
                public_shape = tuple(value.shape)
                mode = layout_module.normalize_mode(len(public_shape), mode)
                if public_stride_order is None:
                    public_stride_order = tuple(reversed(range(len(public_shape))))
                public_stride = layout_module.compact_stride(
                    public_shape,
                    public_stride_order,
                )
                canonical_axis_by_public_axis = layout_module.to_public_axes(
                    tuple(range(len(public_shape))),
                    mode,
                )
                return JaxTensorDesc(
                    dtype=value.dtype,
                    shape=layout_module.to_canonical_axes(public_shape, mode),
                    stride=layout_module.to_canonical_axes(public_stride, mode),
                    stride_order=tuple(canonical_axis_by_public_axis[axis] for axis in public_stride_order),
                    name=name,
                    init_value=init_value,
                )

            @staticmethod
            def _resolve_compute_capability(target, supported, operation_name):
                del operation_name
                resolved = 100 if target is None else target
                if resolved not in supported:
                    raise ValueError(f"unsupported target {resolved}")
                return resolved

            @staticmethod
            def _check_tensor_signature(value, expected, *, mode=None):
                actual_shape = layout_module.to_canonical_axes(tuple(value.shape), mode)
                if actual_shape != expected.shape:
                    raise ValueError(f"{expected.name} tensor shape mismatch: expected " f"{expected.shape}, got {actual_shape}")
                actual_dtype = _DTYPE_TO_CUDNN.get(value.dtype, _DataType.NOT_SET)
                if actual_dtype != expected.cudnn_dtype:
                    raise ValueError(f"{expected.name} tensor dtype mismatch: expected " f"{expected.cudnn_dtype}, got {actual_dtype}")

            @staticmethod
            def _to_tensor_spec(desc, *, mode=None, divisibility=None):
                mode = layout_module.normalize_mode(desc.ndim, mode)
                return _TensorSpec(
                    layout=layout_module.to_cutlass_layout(
                        desc.shape,
                        desc.stride,
                        desc.stride_order,
                        mode=mode,
                        name=desc.name,
                    ),
                    mode=mode,
                    divisibility=divisibility,
                )

            def _call_kernel(self, inputs, **options):
                self.captured_call = (tuple(inputs), options)
                return tuple(
                    _Array(
                        layout_module.to_public_axes(desc.shape, spec.mode),
                        _CUDNN_TO_DTYPE[desc.cudnn_dtype],
                    )
                    for desc, spec in zip(
                        options["output_descs"],
                        options["output_spec"],
                    )
                )

            def _get_max_active_clusters(self, cluster_size, *, overlap_margin=0):
                self.captured_active_cluster_queries = getattr(
                    self,
                    "captured_active_cluster_queries",
                    [],
                )
                self.captured_active_cluster_queries.append((cluster_size, overlap_margin))
                return 12 - overlap_margin

        internal.JaxApiBase = JaxApiBase
        internal.JaxTensorDesc = JaxTensorDesc
        internal.TupleDict = result_module.TupleDict

        datatypes = types.ModuleType(f"{internal_name}.datatypes")
        datatypes.jax_to_cudnn_dtype = lambda dtype: _DTYPE_TO_CUDNN.get(dtype, _DataType.NOT_SET)
        datatypes.normalize_jax_dtype = lambda value, default, name: (default if value is None else value)
        datatypes.cudnn_to_jax_dtype = lambda dtype: _CUDNN_TO_DTYPE[dtype]
        sys.modules[datatypes.__name__] = datatypes

        cls.jit_static_argnames = {}
        fake_jax = types.ModuleType("jax")

        def jit(function=None, *, static_argnames=()):
            def decorate(target):
                cls.jit_static_argnames[target.__name__] = tuple(static_argnames)
                return target

            return decorate if function is None else decorate(function)

        fake_jax.jit = jit
        fake_jax.ShapeDtypeStruct = _Array
        fake_jnp = types.ModuleType("jax.numpy")
        fake_jnp.dtype = lambda dtype: dtype
        for dtype_name in _DTYPE_TO_CUDNN:
            setattr(fake_jnp, dtype_name, dtype_name)
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
                cls.module = importlib.import_module(f"{operation_name}.jax")
        except Exception:
            cls.tearDownClass()
            raise

        cls.operation_name = operation_name

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    @staticmethod
    def _samples(
        *,
        a_layout="LMK",
        b_layout="LNK",
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        sf_vec_size=32,
    ):
        m, n, k, batch = 128, 256, 128, 2
        a_shape = (batch, m, k) if a_layout == "LMK" else (batch, k, m)
        b_shape = (batch, n, k) if b_layout == "LNK" else (batch, k, n)
        k_tiles = ((k + sf_vec_size - 1) // sf_vec_size + 3) // 4
        sfa_shape = (batch, 1, k_tiles, 32, 4, 4)
        sfb_shape = (batch, 2, k_tiles, 32, 4, 4)
        return (
            _Array(a_shape, ab_dtype),
            _Array(b_shape, ab_dtype),
            _Array(sfa_shape, sf_dtype),
            _Array(sfb_shape, sf_dtype),
        )

    def test_default_layouts_bind_row_major_scales_and_infer_outputs(self):
        samples = self._samples()
        api = self.module.GemmAmaxSm100(*samples)

        self.assertEqual(api.a_mode, (1, 2, 0))
        self.assertEqual(api.b_mode, (1, 2, 0))
        self.assertEqual(api.output_mode, (1, 2, 0))
        self.assertEqual(api.scale_mode, (3, 4, 1, 5, 2, 0))
        self.assertEqual(api.sfa_desc.shape, (32, 4, 1, 4, 1, 2))
        self.assertEqual(api.sfa_desc.stride_order, (3, 1, 0, 4, 2, 5))
        self.assertEqual(api.amax_desc.init_value, float("-inf"))

        result = api(*samples)
        self.assertEqual(tuple(result.keys()), ("c_tensor", "amax_tensor"))
        self.assertEqual(result["c_tensor"].shape, (2, 128, 256))
        self.assertEqual(result["c_tensor"].dtype, "float32")
        self.assertEqual(result["amax_tensor"].shape, (1, 1, 1))

        inputs, options = api.captured_call
        self.assertEqual(inputs, samples)
        self.assertTrue(callable(options["launch"]))
        self.assertEqual(
            tuple(spec.mode for spec in options["input_spec"]),
            (api.a_mode, api.b_mode, api.scale_mode, api.scale_mode),
        )
        self.assertEqual(options["input_spec"][2].layout, (5, 4, 3, 2, 1, 0))
        self.assertEqual(options["input_spec"][3].layout, (5, 4, 3, 2, 1, 0))
        self.assertEqual(options["output_descs"], (api.c_desc, api.amax_desc))
        self.assertIn("--gpu-arch sm_100a", options["compile_options"])

    def test_alternate_gemm_layouts_and_native_fp4_are_supported(self):
        alternate_samples = self._samples(
            a_layout="LKM",
            b_layout="LKN",
        )
        alternate = self.module.GemmAmaxSm100(
            *alternate_samples,
            a_layout="LKM",
            b_layout="LKN",
            c_layout="LNM",
            c_dtype="bfloat16",
        )
        alternate_result = alternate(*alternate_samples)
        self.assertEqual(alternate.a_desc.stride_order, (0, 1, 2))
        self.assertEqual(alternate.b_desc.stride_order, (0, 1, 2))
        self.assertEqual(alternate_result["c_tensor"].shape, (2, 256, 128))

        fp4_samples = self._samples(
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            sf_vec_size=16,
        )
        fp4 = self.module.GemmAmaxSm100(
            *fp4_samples,
            c_dtype="bfloat16",
            sf_vec_size=16,
        )
        fp4_result = fp4(*fp4_samples)
        self.assertEqual(fp4_result["c_tensor"].shape, (2, 128, 256))
        self.assertEqual(fp4_result["c_tensor"].dtype, "bfloat16")

        for c_dtype in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1fn"):
            with self.subTest(c_dtype=c_dtype):
                quantized_output = self.module.GemmAmaxSm100(
                    *fp4_samples,
                    c_dtype=c_dtype,
                    sf_vec_size=16,
                )(*fp4_samples)
                self.assertEqual(quantized_output["c_tensor"].dtype, c_dtype)

    def test_explicit_outputs_and_runtime_signatures_are_checked(self):
        samples = self._samples()
        api = self.module.GemmAmaxSm100(
            *samples,
            sample_c=_Array((2, 128, 256), "bfloat16"),
            sample_amax=_Array((1, 1, 1), "float32"),
            c_dtype="bfloat16",
        )
        result = api(*samples)
        self.assertEqual(result["c_tensor"].dtype, "bfloat16")

        with self.assertRaisesRegex(
            ValueError,
            "sample_c and sample_amax must be provided together",
        ):
            self.module.GemmAmaxSm100(
                *samples,
                sample_c=_Array((2, 128, 256), "float32"),
            )
        with self.assertRaisesRegex(ValueError, "sample_a tensor shape mismatch"):
            api(_Array((2, 64, 128), samples[0].dtype), *samples[1:])

    def test_wrapper_configuration_is_static(self):
        self.assertEqual(
            self.jit_static_argnames["gemm_amax_wrapper_sm100"],
            (
                "c_layout",
                "c_dtype",
                "acc_dtype",
                "mma_tiler_mn",
                "cluster_shape_mn",
                "sf_vec_size",
                "a_layout",
                "b_layout",
            ),
        )
        samples = self._samples()
        result = self.module.gemm_amax_wrapper_sm100(*samples)
        self.assertEqual(result["c_tensor"].shape, (2, 128, 256))

    def test_inline_launch_constructs_kernel_and_preserves_abi_order(self):
        samples = self._samples()
        with mock.patch.dict(os.environ, {"CUDNNFE_CLUSTER_OVERLAP_MARGIN": "2"}):
            api = self.module.GemmAmaxSm100(*samples, cluster_shape_mn=(2, 2))
        api(*samples)
        self.assertEqual(api.captured_active_cluster_queries, [(4, 2)])
        _, options = api.captured_call
        seen = {}

        class Kernel:
            def __init__(self, **configuration):
                seen["configuration"] = configuration

            def __call__(self, *arguments):
                seen["arguments"] = arguments

        kernel_module = types.ModuleType(f"{self.operation_name}.dense_blockscaled_gemm_persistent_amax")
        kernel_module.Sm100BlockScaledPersistentDenseGemmKernel = Kernel
        stream = object()
        with mock.patch.dict(sys.modules, {kernel_module.__name__: kernel_module}):
            options["launch"](stream, "A", "B", "SFA", "SFB", "C", "Amax")

        self.assertEqual(
            seen["configuration"],
            {
                "sf_vec_size": 32,
                "mma_tiler_mn": (128, 128),
                "cluster_shape_mn": (2, 2),
            },
        )
        self.assertEqual(
            seen["arguments"],
            ("A", "B", "SFA", "SFB", "C", "Amax", 10, stream),
        )

    def test_adapter_import_does_not_load_torch_or_kernel(self):
        self.assertNotIn(f"{self.operation_name}.api", sys.modules)
        self.assertNotIn(
            f"{self.operation_name}.dense_blockscaled_gemm_persistent_amax",
            sys.modules,
        )


if __name__ == "__main__":
    unittest.main()
