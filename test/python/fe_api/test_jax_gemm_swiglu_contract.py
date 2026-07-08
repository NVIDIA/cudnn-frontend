# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the standard JAX GEMM + SwiGLU adapter."""

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
_OPERATION_ROOT = _CUDNN_ROOT / "gemm_swiglu"
_PACKAGE = "cudnn_jax_gemm_swiglu_contract_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    FP8_E4M3 = auto()
    FP8_E5M2 = auto()
    FP8_E8M0 = auto()
    FP4_E2M1 = auto()


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


class JaxGemmSwigluContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        operation_name = f"{_PACKAGE}.gemm_swiglu"
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
            def _to_tensor_desc(value, name, *, mode=None, init_value=None, **_unused):
                public_shape = tuple(value.shape)
                mode = layout_module.normalize_mode(len(public_shape), mode)
                public_order = tuple(reversed(range(len(public_shape))))
                public_stride = layout_module.compact_stride(public_shape, public_order)
                canonical_axis_by_public_axis = layout_module.to_public_axes(tuple(range(len(public_shape))), mode)
                return JaxTensorDesc(
                    dtype=value.dtype,
                    shape=layout_module.to_canonical_axes(public_shape, mode),
                    stride=layout_module.to_canonical_axes(public_stride, mode),
                    stride_order=tuple(canonical_axis_by_public_axis[axis] for axis in public_order),
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
                    raise ValueError(f"{expected.name} tensor shape mismatch: " f"expected {expected.shape}, got {actual_shape}")
                actual_dtype = _DTYPE_TO_CUDNN.get(value.dtype, _DataType.NOT_SET)
                if actual_dtype != expected.cudnn_dtype:
                    raise ValueError(f"{expected.name} tensor dtype mismatch: " f"expected {expected.cudnn_dtype}, got {actual_dtype}")

            @staticmethod
            def _to_tensor_spec(desc, *, mode=None, divisibility=None):
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

            def _call_kernel(self, inputs, *, launch, **options):
                options["launch"] = launch
                self.captured_call = (tuple(inputs), options)
                return tuple(
                    _Array(
                        layout_module.to_public_axes(desc.shape, spec.mode),
                        _CUDNN_TO_DTYPE[desc.cudnn_dtype],
                    )
                    for desc, spec in zip(options["output_descs"], options["output_spec"])
                )

            def _get_max_active_clusters(self, cluster_size, *, overlap_margin=0):
                if not hasattr(self, "captured_active_cluster_queries"):
                    self.captured_active_cluster_queries = []
                self.captured_active_cluster_queries.append((cluster_size, overlap_margin))
                return 12 - overlap_margin

        internal.JaxApiBase = JaxApiBase
        internal.JaxTensorDesc = JaxTensorDesc
        internal.TupleDict = result_module.TupleDict

        datatypes = types.ModuleType(f"{internal_name}.datatypes")
        datatypes.jax_to_cudnn_dtype = lambda dtype: _DTYPE_TO_CUDNN.get(dtype, _DataType.NOT_SET)
        datatypes.normalize_jax_dtype = lambda value, default, name: default if value is None else value
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
        fake_jnp.float16 = "float16"
        fake_jnp.bfloat16 = "bfloat16"
        fake_jnp.float32 = "float32"
        fake_jnp.float4_e2m1fn = "float4_e2m1fn"
        fake_jnp.float8_e4m3fn = "float8_e4m3fn"
        fake_jnp.float8_e5m2 = "float8_e5m2"
        fake_jnp.float8_e8m0fnu = "float8_e8m0fnu"
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

        cls.JaxApiBase = JaxApiBase
        cls.operation_name = operation_name

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def setUp(self) -> None:
        self.JaxApiBase.captured_call = None

    @staticmethod
    def _samples(a_layout="LMK", b_layout="LNK"):
        m, n, k, batch = 128, 192, 64, 3
        a_shape = (batch, m, k) if a_layout == "LMK" else (batch, k, m)
        b_shape = (batch, n, k) if b_layout == "LNK" else (batch, k, n)
        return _Array(a_shape, "bfloat16"), _Array(b_shape, "bfloat16")

    def test_default_layouts_infer_outputs_and_declare_cutlass_call(self):
        sample_a, sample_b = self._samples()
        api = self.module.GemmSwigluSm100(
            sample_a,
            sample_b,
        )

        self.assertEqual(api.a_mode, (1, 2, 0))
        self.assertEqual(api.b_mode, (1, 2, 0))
        self.assertEqual(api.output_mode, (1, 2, 0))
        self.assertEqual(api.a_desc.shape, (128, 64, 3))
        self.assertEqual(api.b_desc.shape, (192, 64, 3))
        self.assertEqual(api.ab12_desc.shape, (128, 192, 3))
        self.assertEqual(api.c_desc.shape, (128, 96, 3))
        self.assertIs(api._op.a, api.a_desc)
        self.assertIs(api._op.b, api.b_desc)
        self.assertIs(api._op.ab12, api.ab12_desc)
        self.assertIs(api._op.c, api.c_desc)

        result = api(sample_a, sample_b)
        self.assertEqual(
            tuple(result.keys()),
            ("ab12_tensor", "c_tensor", "sfc_tensor", "amax_tensor"),
        )
        self.assertEqual(result["ab12_tensor"].shape, (3, 128, 192))
        self.assertEqual(result["ab12_tensor"].dtype, "float32")
        self.assertEqual(result["c_tensor"].shape, (3, 128, 96))
        self.assertEqual(result["c_tensor"].dtype, "float16")
        self.assertIsNone(result["sfc_tensor"])
        self.assertIsNone(result["amax_tensor"])

        inputs, options = api.captured_call
        self.assertEqual(inputs, (sample_a, sample_b))
        self.assertTrue(callable(options["launch"]))
        self.assertEqual(options["output_descs"], (api.ab12_desc, api.c_desc))
        self.assertEqual(
            tuple(spec.mode for spec in options["input_spec"]),
            (api.a_mode, api.b_mode),
        )
        self.assertEqual(
            tuple(spec.mode for spec in options["output_spec"]),
            (api.output_mode, api.output_mode),
        )
        self.assertEqual(
            tuple(spec.layout for spec in options["input_spec"]),
            ((2, 1, 0), (2, 1, 0)),
        )
        self.assertEqual(
            tuple(spec.layout for spec in options["output_spec"]),
            ((2, 1, 0), (2, 1, 0)),
        )
        self.assertIn("--gpu-arch sm_100a", options["compile_options"])

    def test_alternate_layouts_map_public_shapes_to_the_same_canonical_op(self):
        sample_a, sample_b = self._samples(a_layout="LKM", b_layout="LKN")
        api = self.module.GemmSwigluSm100(
            sample_a,
            sample_b,
            a_layout="LKM",
            b_layout="LKN",
            c_layout="LNM",
        )
        result = api(sample_a, sample_b)

        self.assertEqual(api.a_mode, (2, 1, 0))
        self.assertEqual(api.b_mode, (2, 1, 0))
        self.assertEqual(api.output_mode, (2, 1, 0))
        self.assertEqual(api.a_desc.shape, (128, 64, 3))
        self.assertEqual(api.b_desc.shape, (192, 64, 3))
        self.assertEqual(api.a_desc.stride_order, (0, 1, 2))
        self.assertEqual(api.b_desc.stride_order, (0, 1, 2))
        self.assertEqual(api.ab12_desc.stride_order, (0, 1, 2))
        self.assertEqual(api.c_desc.stride_order, (0, 1, 2))
        self.assertEqual(result["ab12_tensor"].shape, (3, 192, 128))
        self.assertEqual(result["c_tensor"].shape, (3, 96, 128))
        self.assertIn("--gpu-arch sm_100a", api.captured_call[1]["compile_options"])

    def test_explicit_output_exemplars_are_checked_and_preserved(self):
        sample_a, sample_b = self._samples()
        sample_ab12 = _Array((3, 128, 192), "bfloat16")
        sample_c = _Array((3, 128, 96), "bfloat16")
        api = self.module.GemmSwigluSm100(
            sample_a,
            sample_b,
            sample_ab12=sample_ab12,
            sample_c=sample_c,
            ab12_dtype="bfloat16",
            c_dtype="bfloat16",
        )
        result = api(sample_a, sample_b)
        self.assertEqual(result["ab12_tensor"].dtype, "bfloat16")
        self.assertEqual(result["c_tensor"].dtype, "bfloat16")

        with self.assertRaisesRegex(ValueError, "sample_ab12 and sample_c must be provided together"):
            self.module.GemmSwigluSm100(
                sample_a,
                sample_b,
                sample_ab12=sample_ab12,
            )
        with self.assertRaisesRegex(ValueError, "c_dtype=float16 does not match the explicit sample dtype"):
            self.module.GemmSwigluSm100(
                sample_a,
                sample_b,
                sample_ab12=sample_ab12,
                sample_c=sample_c,
                c_dtype="float16",
            )

    def test_runtime_inputs_must_match_the_specialized_signature(self):
        sample_a, sample_b = self._samples()
        api = self.module.GemmSwigluSm100(
            sample_a,
            sample_b,
        )
        with self.assertRaisesRegex(ValueError, "sample_a tensor shape mismatch"):
            api(_Array((3, 64, 64), "bfloat16"), sample_b)
        with self.assertRaisesRegex(ValueError, "sample_b tensor dtype mismatch"):
            api(sample_a, _Array(sample_b.shape, "float16"))

    def test_rejects_unreachable_fp8_c_and_block_only_standard_options(self):
        sample_a, sample_b = self._samples()
        sample_ab12 = _Array((3, 128, 192), "bfloat16")
        sample_c_fp8 = _Array((3, 128, 96), "float8_e4m3fn")
        with self.assertRaisesRegex(NotImplementedError, "does not support FP8 C"):
            self.module.GemmSwigluSm100(
                sample_a,
                sample_b,
                sample_ab12=sample_ab12,
                sample_c=sample_c_fp8,
            )

        with self.assertRaisesRegex(NotImplementedError, "does not expose FP8 C or SFC"):
            self.module.GemmSwigluSm100(
                sample_a,
                sample_b,
                sample_sfc=_Array((1,), "float8_e8m0fnu"),
            )

        for option, value in (("sf_vec_size", 32), ("vector_f32", True), ("ab12_stages", 3)):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "only applies to block-scaled"):
                self.module.GemmSwigluSm100(sample_a, sample_b, **{option: value})

    def test_wrapper_marks_configuration_as_static_and_remains_functional(self):
        self.assertIn("gemm_swiglu_wrapper_sm100", self.module.__all__)
        self.assertNotIn("gemm_swiglu_wrapper", self.module.__all__)
        static = self.jit_static_argnames["gemm_swiglu_wrapper_sm100"]
        self.assertEqual(
            static,
            (
                "alpha",
                "c_layout",
                "ab12_dtype",
                "c_dtype",
                "acc_dtype",
                "mma_tiler_mn",
                "cluster_shape_mn",
                "sf_vec_size",
                "vector_f32",
                "ab12_stages",
                "a_layout",
                "b_layout",
            ),
        )

        sample_a, sample_b = self._samples(a_layout="LKM", b_layout="LKN")
        result = self.module.gemm_swiglu_wrapper_sm100(
            sample_a,
            sample_b,
            c_layout="LNM",
            a_layout="LKM",
            b_layout="LKN",
        )
        self.assertEqual(result["ab12_tensor"].shape, (3, 192, 128))
        self.assertEqual(result["c_tensor"].shape, (3, 96, 128))

    def test_block_scaled_mxfp8_uses_packed_scale_modes_and_kernel_abi(self):
        batch, m, n, k = 2, 128, 128, 128
        sample_a = _Array((batch, m, k), "float8_e4m3fn")
        sample_b = _Array((batch, n, k), "float8_e4m3fn")
        scale_shape = (batch, 1, 1, 32, 4, 4)
        sample_sfa = _Array(scale_shape, "float8_e8m0fnu")
        sample_sfb = _Array(scale_shape, "float8_e8m0fnu")
        api = self.module.GemmSwigluSm100(
            sample_a,
            sample_b,
            sample_sfa=sample_sfa,
            sample_sfb=sample_sfb,
            alpha=0.5,
            ab12_dtype="bfloat16",
            c_dtype="bfloat16",
            sf_vec_size=32,
            vector_f32=True,
            ab12_stages=3,
        )

        self.assertTrue(api.is_block_scaled)
        self.assertEqual(api.scale_mode, (3, 4, 1, 5, 2, 0))
        self.assertEqual(api.sfa_desc.shape, (32, 4, 1, 4, 1, batch))
        self.assertEqual(api.sfa_desc.stride_order, (3, 1, 0, 4, 2, 5))
        result = api(sample_a, sample_b, sample_sfa, sample_sfb)
        self.assertEqual(result["ab12_tensor"].shape, (batch, m, n))
        self.assertEqual(result["ab12_tensor"].dtype, "bfloat16")
        self.assertEqual(result["c_tensor"].shape, (batch, m, n // 2))
        self.assertEqual(result["c_tensor"].dtype, "bfloat16")
        self.assertIsNone(result["sfc_tensor"])
        self.assertIsNone(result["amax_tensor"])

        inputs, options = api.captured_call
        self.assertEqual(inputs, (sample_a, sample_b, sample_sfa, sample_sfb))
        self.assertEqual(
            tuple(spec.mode for spec in options["input_spec"]),
            (api.a_mode, api.b_mode, api.scale_mode, api.scale_mode),
        )
        self.assertEqual(tuple(spec.mode for spec in options["output_spec"]), (api.output_mode, api.output_mode))
        self.assertEqual(api.captured_active_cluster_queries, [(1, 0)])

        seen = {}

        class Kernel:
            def __init__(self, **configuration):
                seen["configuration"] = configuration

            def __call__(self, *arguments):
                seen["arguments"] = arguments

        cutlass = types.ModuleType("cutlass")
        cutlass.Float32 = lambda value: ("Float32", value)
        kernel_module = types.ModuleType(f"{self.operation_name}.dense_blockscaled_gemm_persistent_swiglu_interleaved_quant")
        kernel_module.Sm100BlockScaledPersistentDenseGemmKernel = Kernel
        stream = object()
        with mock.patch.dict(
            sys.modules,
            {
                "cutlass": cutlass,
                kernel_module.__name__: kernel_module,
            },
        ):
            options["launch"](stream, "A", "B", "SFA", "SFB", "AB12", "C")

        self.assertEqual(
            seen["configuration"],
            {
                "sf_vec_size": 32,
                "mma_tiler_mn": (128, 128),
                "cluster_shape_mn": (1, 1),
                "vector_f32": True,
                "ab12_stages": 3,
            },
        )
        self.assertEqual(
            seen["arguments"],
            ("A", "B", "SFA", "SFB", "C", "AB12", None, None, None, ("Float32", 0.5), 12, stream),
        )

        wrapped = self.module.gemm_swiglu_wrapper_sm100(
            sample_a,
            sample_b,
            alpha=0.5,
            ab12_dtype="bfloat16",
            c_dtype="bfloat16",
            sfa_tensor=sample_sfa,
            sfb_tensor=sample_sfb,
            sf_vec_size=32,
        )
        self.assertEqual(wrapped["ab12_tensor"].shape, (batch, m, n))
        self.assertEqual(wrapped["c_tensor"].shape, (batch, m, n // 2))

    def test_block_scaled_fp4_infers_initialized_amax(self):
        sample_a = _Array((1, 128, 128), "float4_e2m1fn")
        sample_b = _Array((1, 128, 128), "float4_e2m1fn")
        scale_shape = (1, 1, 2, 32, 4, 4)
        sample_sfa = _Array(scale_shape, "float8_e8m0fnu")
        sample_sfb = _Array(scale_shape, "float8_e8m0fnu")
        api = self.module.GemmSwigluSm100(
            sample_a,
            sample_b,
            sample_sfa=sample_sfa,
            sample_sfb=sample_sfb,
            c_dtype="bfloat16",
            sf_vec_size=16,
        )

        result = api(sample_a, sample_b, sample_sfa, sample_sfb)
        self.assertEqual(result["amax_tensor"].shape, (1,))
        self.assertEqual(result["amax_tensor"].dtype, "float32")
        self.assertEqual(api.amax_desc.init_value, float("-inf"))
        self.assertEqual(api.captured_call[1]["output_descs"], (api.ab12_desc, api.c_desc, api.amax_desc))

        with self.assertRaisesRegex(ValueError, "sample_sfa and sample_sfb are required"):
            self.module.GemmSwigluSm100(sample_a, sample_b, sample_sfa=sample_sfa)

    def test_launch_constructs_the_kernel_internally_and_preserves_abi_order(self):
        _, sample_b = self._samples()
        sample_a = _Array((3, 256, 64), "bfloat16")
        with mock.patch.dict(os.environ, {"CUDNNFE_CLUSTER_OVERLAP_MARGIN": "2"}):
            api = self.module.GemmSwigluSm100(
                sample_a,
                sample_b,
                alpha=0.25,
                mma_tiler_mn=(256, 128),
            )
        self.assertFalse(hasattr(api, "captured_active_cluster_queries"))
        api(sample_a, sample_b)
        self.assertEqual(api.captured_active_cluster_queries, [(4, 2)])
        launch = api.captured_call[1]["launch"]
        seen = {}

        class HardwareInfo:
            def get_max_active_clusters(self, cluster_size):
                raise AssertionError(f"HardwareInfo queried from launch callback for cluster size {cluster_size}")

        class Kernel:
            def __init__(self, **configuration):
                seen["configuration"] = configuration

            def __call__(self, *arguments):
                seen["arguments"] = arguments

        cutlass = types.ModuleType("cutlass")
        cutlass.Float32 = lambda value: ("Float32", value)
        cutlass.utils = types.SimpleNamespace(HardwareInfo=HardwareInfo)
        cutlass_jax = types.ModuleType("cutlass.jax")
        cutlass_jax.jax_to_cutlass_dtype = lambda dtype: ("cutlass_dtype", dtype)
        cutlass.jax = cutlass_jax
        kernel_module = types.ModuleType(f"{self.operation_name}.dense_gemm_persistent_swiglu")
        kernel_module.PersistentDenseGemmKernel = Kernel
        stream = object()

        with mock.patch.dict(
            sys.modules,
            {
                "cutlass": cutlass,
                "cutlass.jax": cutlass_jax,
                kernel_module.__name__: kernel_module,
            },
        ):
            launch(stream, "A", "B", "AB12", "C")

        self.assertEqual(
            seen["configuration"],
            {
                "acc_dtype": ("cutlass_dtype", "float32"),
                "use_2cta_instrs": True,
                "mma_tiler_mn": (256, 128),
                "cluster_shape_mn": (2, 2),
            },
        )
        self.assertEqual(
            seen["arguments"],
            ("A", "B", "AB12", "C", ("Float32", 0.25), 10, stream),
        )

        with self.assertRaises(TypeError):
            launch(stream, "A", "B", "AB12", "C", object())

    def test_adapter_import_does_not_load_torch_or_the_cute_kernel(self):
        self.assertNotIn(f"{self.operation_name}.api", sys.modules)
        self.assertNotIn(f"{self.operation_name}.dense_gemm_persistent_swiglu", sys.modules)


if __name__ == "__main__":
    unittest.main()
