# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts for the JAX sReLU and dsReLU adapters."""

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
_PACKAGE = "cudnn_jax_gemm_relu_contract_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    UINT8 = auto()
    FP4_E2M1 = auto()
    FP8_E4M3 = auto()
    FP8_E5M2 = auto()
    FP8_E8M0 = auto()


_DTYPE_TO_CUDNN = {
    "float4_e2m1fn": _DataType.FP4_E2M1,
    "float8_e4m3fn": _DataType.FP8_E4M3,
    "float8_e5m2": _DataType.FP8_E5M2,
    "float8_e8m0fnu": _DataType.FP8_E8M0,
    "float16": _DataType.HALF,
    "bfloat16": _DataType.BFLOAT16,
    "float32": _DataType.FLOAT,
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


class JaxGemmReluContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        for operation in ("gemm_srelu", "gemm_dsrelu"):
            operation_name = f"{_PACKAGE}.{operation}"
            operation_module = types.ModuleType(operation_name)
            operation_module.__path__ = [str(_CUDNN_ROOT / operation)]
            operation_module.__package__ = operation_name
            operation_module.__spec__ = ModuleSpec(operation_name, loader=None, is_package=True)
            sys.modules[operation_name] = operation_module

        internal_name = f"{_PACKAGE}._jax"
        internal = types.ModuleType(internal_name)
        internal.__path__ = [str(_CUDNN_ROOT / "_jax")]
        internal.__package__ = internal_name
        internal.__spec__ = ModuleSpec(internal_name, loader=None, is_package=True)
        sys.modules[internal_name] = internal

        tensor_module = importlib.import_module(f"{_PACKAGE}.common.tensor_desc")
        layout_module = importlib.import_module(f"{internal_name}.layout")
        result_module = importlib.import_module(f"{_PACKAGE}.common.result")

        class JaxTensorDesc(tensor_module.TensorDesc):
            @classmethod
            def from_shape(cls, shape, dtype, *, name, mode=None, init_value=None):
                public_shape = tuple(shape)
                mode = layout_module.normalize_mode(len(public_shape), mode)
                public_order = tuple(reversed(range(len(public_shape))))
                public_stride = layout_module.compact_stride(public_shape, public_order)
                canonical_axis_by_public_axis = layout_module.to_public_axes(tuple(range(len(public_shape))), mode)
                desc = cls(
                    dtype=dtype,
                    shape=layout_module.to_canonical_axes(public_shape, mode),
                    stride=layout_module.to_canonical_axes(public_stride, mode),
                    stride_order=tuple(canonical_axis_by_public_axis[axis] for axis in public_order),
                    name=name,
                    init_value=init_value,
                )
                object.__setattr__(desc, "mode", mode)
                return desc

            @property
            def cudnn_dtype(self):
                return _DTYPE_TO_CUDNN.get(self.dtype, _DataType.NOT_SET)

            def compact_like(self, *, cudnn_dtype, shape, stride_order=None, name="", init_value=None):
                canonical = tensor_module.make_compact_tensor_desc(
                    dtype=cudnn_dtype,
                    shape=shape,
                    stride_order=stride_order,
                    name=name,
                    init_value=init_value,
                )
                return JaxTensorDesc(
                    dtype=_CUDNN_TO_DTYPE[cudnn_dtype],
                    shape=canonical.shape,
                    stride=canonical.stride,
                    stride_order=canonical.stride_order,
                    name=name,
                    init_value=init_value,
                )

        class JaxApiBase:
            @staticmethod
            def _to_tensor_desc(value, name, *, mode=None, init_value=None, **_unused):
                public_shape = tuple(value.shape)
                mode = layout_module.normalize_mode(len(public_shape), mode)
                public_order = tuple(reversed(range(len(public_shape))))
                public_stride = layout_module.compact_stride(public_shape, public_order)
                canonical_axis_by_public_axis = layout_module.to_public_axes(tuple(range(len(public_shape))), mode)
                desc = JaxTensorDesc(
                    dtype=value.dtype,
                    shape=layout_module.to_canonical_axes(public_shape, mode),
                    stride=layout_module.to_canonical_axes(public_stride, mode),
                    stride_order=tuple(canonical_axis_by_public_axis[axis] for axis in public_order),
                    name=name,
                    init_value=init_value,
                )
                object.__setattr__(desc, "mode", mode)
                return desc

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
                    raise ValueError(f"{expected.name} tensor shape mismatch: expected {expected.shape}, got {actual_shape}")
                actual_dtype = _DTYPE_TO_CUDNN.get(value.dtype, _DataType.NOT_SET)
                if actual_dtype != expected.cudnn_dtype:
                    raise ValueError(f"{expected.name} tensor dtype mismatch: expected {expected.cudnn_dtype}, got {actual_dtype}")

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

            def _call_kernel(self, inputs, **options):
                input_descs = options.get("input_descs")
                if input_descs is not None:
                    for value, desc in zip(inputs, input_descs):
                        self._check_tensor_signature(value, desc, mode=desc.mode)
                    options["input_spec"] = tuple(self._to_tensor_spec(desc, mode=desc.mode) for desc in input_descs)
                if "output_spec" not in options:
                    options["output_spec"] = tuple(
                        self._to_tensor_spec(desc, mode=getattr(desc, "mode", None)) for desc in options["output_descs"]
                    )
                self.captured_call = (tuple(inputs), options)
                return tuple(
                    _Array(
                        layout_module.to_public_axes(desc.shape, spec.mode),
                        _CUDNN_TO_DTYPE[desc.cudnn_dtype],
                    )
                    for desc, spec in zip(options["output_descs"], options["output_spec"])
                )

            def _get_max_active_clusters(self, cluster_size, *, overlap_margin=0):
                self.captured_active_cluster_query = (cluster_size, overlap_margin)
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
                cls.srelu = importlib.import_module(f"{_PACKAGE}.gemm_srelu.jax")
                cls.dsrelu = importlib.import_module(f"{_PACKAGE}.gemm_dsrelu.jax")
        except Exception:
            cls.tearDownClass()
            raise

        cls.operation_names = {
            cls.srelu: f"{_PACKAGE}.gemm_srelu",
            cls.dsrelu: f"{_PACKAGE}.gemm_dsrelu",
        }
        cls.tensor_module = tensor_module

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    @staticmethod
    def _samples(*, dtype="float8_e4m3fn", sf_vec_size=32, batch=2, m=256, n=256, k=512):
        rest_k = ((k + sf_vec_size - 1) // sf_vec_size + 3) // 4
        a = _Array((batch, m, k), dtype)
        b = _Array((batch, n, k), dtype)
        sfa = _Array((batch, (m + 127) // 128, rest_k, 32, 4, 4), "float8_e8m0fnu")
        sfb = _Array((batch, (n + 127) // 128, rest_k, 32, 4, 4), "float8_e8m0fnu")
        prob = _Array((batch, 1, m), "float32")
        c = _Array((batch, m, n), "bfloat16")
        return a, b, c, sfa, sfb, prob

    def test_forward_and_backward_use_row_major_public_auxiliary_layouts(self):
        a, b, c, sfa, sfb, prob = self._samples()
        forward = self.srelu.GemmSreluSm100(a, b, sfa, sfb, prob, sf_vec_size=32)
        backward = self.dsrelu.GemmDsreluSm100(a, b, c, sfa, sfb, prob, sf_vec_size=32)

        forward_result = forward(a, b, sfa, sfb, prob)
        backward_result = backward(a, b, c, sfa, sfb, prob)

        self.assertEqual(forward.sfa_desc.shape, (32, 4, 2, 4, 4, 2))
        self.assertEqual(forward.sfa_desc.stride_order, (3, 1, 0, 4, 2, 5))
        self.assertEqual(forward.prob_desc.shape, (256, 1, 2))
        self.assertEqual(forward.prob_desc.stride_order, (0, 1, 2))
        self.assertIs(forward._op.sfa, forward.sfa_desc)
        self.assertIs(backward._op.dprob, backward.dprob_desc)

        self.assertEqual(forward_result["c_tensor"].shape, (2, 256, 256))
        self.assertEqual(forward_result["d_tensor"].shape, (2, 256, 256))
        self.assertIsNone(forward_result["amax_tensor"])
        self.assertEqual(backward_result["d_tensor"].shape, (2, 256, 256))
        self.assertEqual(backward_result["dprob_tensor"].shape, (2, 1, 256))
        self.assertEqual(backward.dprob_desc.init_value, 0.0)

        forward_inputs, forward_options = forward.captured_call
        self.assertEqual(forward_inputs, (a, b, sfa, sfb, prob))
        self.assertTrue(callable(forward_options["launch"]))
        self.assertEqual(
            tuple(spec.mode for spec in forward_options["input_spec"]),
            (
                forward.a_desc.mode,
                forward.b_desc.mode,
                forward.sfa_desc.mode,
                forward.sfb_desc.mode,
                forward.prob_desc.mode,
            ),
        )
        self.assertEqual(forward_options["input_spec"][2].layout, (5, 4, 3, 2, 1, 0))
        self.assertEqual(forward_options["input_spec"][4].layout, (2, 1, 0))
        self.assertIn("--gpu-arch sm_100a", forward_options["compile_options"])

    def test_native_fp4_returns_initialized_amax(self):
        a, b, c, sfa, sfb, prob = self._samples(dtype="float4_e2m1fn", sf_vec_size=16)
        forward = self.srelu.GemmSreluSm100(a, b, sfa, sfb, prob, sf_vec_size=16)
        backward = self.dsrelu.GemmDsreluSm100(a, b, c, sfa, sfb, prob, sf_vec_size=16)

        forward_result = forward(a, b, sfa, sfb, prob)
        backward_result = backward(a, b, c, sfa, sfb, prob)

        for api, result in ((forward, forward_result), (backward, backward_result)):
            self.assertIsNotNone(api.amax_desc)
            self.assertEqual(api.amax_desc.shape, (1,))
            self.assertEqual(api.amax_desc.init_value, float("-inf"))
            self.assertEqual(result["amax_tensor"].shape, (1,))
            self.assertIs(api.captured_call[1]["output_descs"][-1], api.amax_desc)

    def test_fp8_d_output_is_rejected_until_sfd_generation_is_implemented(self):
        a, b, c, sfa, sfb, prob = self._samples()
        constructors = (
            lambda: self.srelu.GemmSreluSm100(
                a,
                b,
                sfa,
                sfb,
                prob,
                d_dtype="float8_e4m3fn",
                sf_vec_size=32,
            ),
            lambda: self.dsrelu.GemmDsreluSm100(
                a,
                b,
                c,
                sfa,
                sfb,
                prob,
                d_dtype="float8_e4m3fn",
                sf_vec_size=32,
            ),
        )
        for construct in constructors:
            with self.subTest(construct=construct), self.assertRaisesRegex(NotImplementedError, "FP8 D output is unavailable"):
                construct().check_support()

    def test_explicit_output_samples_are_alternative_to_dtype_arguments(self):
        a, b, c, sfa, sfb, prob = self._samples()
        sample_d = _Array(c.shape, "bfloat16")
        sample_dprob = _Array(prob.shape, "float32")

        forward = self.srelu.GemmSreluSm100(
            a,
            b,
            sfa,
            sfb,
            prob,
            sample_c=c,
            sample_d=sample_d,
            sf_vec_size=32,
        )
        backward = self.dsrelu.GemmDsreluSm100(
            a,
            b,
            c,
            sfa,
            sfb,
            prob,
            sample_d=sample_d,
            sample_dprob=sample_dprob,
            sf_vec_size=32,
        )
        self.assertTrue(forward.check_support())
        self.assertTrue(backward.check_support())

        with self.assertRaisesRegex(ValueError, "c_dtype and d_dtype cannot be specified"):
            self.srelu.GemmSreluSm100(
                a,
                b,
                sfa,
                sfb,
                prob,
                sample_c=c,
                sample_d=sample_d,
                c_dtype="bfloat16",
                sf_vec_size=32,
            )
        with self.assertRaisesRegex(ValueError, "d_dtype cannot be specified"):
            self.dsrelu.GemmDsreluSm100(
                a,
                b,
                c,
                sfa,
                sfb,
                prob,
                sample_d=sample_d,
                sample_dprob=sample_dprob,
                d_dtype="bfloat16",
                sf_vec_size=32,
            )

    def test_unit_scale_tiles_and_batch_preserve_packed_layout(self):
        a, b, c, sfa, sfb, prob = self._samples(batch=1, m=128, n=128, k=128)
        forward = self.srelu.GemmSreluSm100(
            a,
            b,
            sfa,
            sfb,
            prob,
            sf_vec_size=32,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )
        backward = self.dsrelu.GemmDsreluSm100(
            a,
            b,
            c,
            sfa,
            sfb,
            prob,
            sf_vec_size=32,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )

        self.assertTrue(forward.check_support())
        self.assertTrue(backward.check_support())
        self.assertEqual(forward.sfa_desc.shape, (32, 4, 1, 4, 1, 1))
        self.assertTrue(forward.sfa_desc.is_compact((3, 1, 0, 4, 2, 5)))

    def test_launch_callbacks_preserve_native_kernel_abis(self):
        a, b, c, sfa, sfb, prob = self._samples(dtype="float4_e2m1fn", sf_vec_size=16)
        with mock.patch.dict(os.environ, {"CUDNNFE_CLUSTER_OVERLAP_MARGIN": "2"}):
            forward = self.srelu.GemmSreluSm100(a, b, sfa, sfb, prob, alpha=0.25)
            backward = self.dsrelu.GemmDsreluSm100(a, b, c, sfa, sfb, prob, alpha=0.5)
        forward(a, b, sfa, sfb, prob)
        backward(a, b, c, sfa, sfb, prob)
        self.assertEqual(forward.captured_active_cluster_query, (2, 2))
        self.assertEqual(backward.captured_active_cluster_query, (2, 2))

        calls = {}

        def kernel_type(label):
            class Kernel:
                def __init__(self, **configuration):
                    calls[f"{label}_configuration"] = configuration

                def __call__(self, *arguments, **keywords):
                    calls[f"{label}_arguments"] = arguments
                    calls[f"{label}_keywords"] = keywords

            return Kernel

        cutlass = types.ModuleType("cutlass")
        cutlass.Float32 = lambda value: ("Float32", value)
        cute = types.ModuleType("cutlass.cute")
        cute.where = lambda condition, yes, no: yes if condition else no
        cute.full_like = lambda value, fill: fill
        cutlass.cute = cute

        forward_kernel_name = f"{self.operation_names[self.srelu]}.dense_blockscaled_gemm_persistent_srelu_quant"
        forward_kernel = types.ModuleType(forward_kernel_name)
        forward_kernel.Sm100BlockScaledPersistentDenseGemmKernel = kernel_type("forward")
        backward_kernel_name = f"{self.operation_names[self.dsrelu]}.dense_blockscaled_gemm_persistent_dsrelu_quant"
        backward_kernel = types.ModuleType(backward_kernel_name)
        backward_kernel.Sm100BlockScaledPersistentDenseGemmKernel = kernel_type("backward")

        with mock.patch.dict(
            sys.modules,
            {
                "cutlass": cutlass,
                "cutlass.cute": cute,
                forward_kernel_name: forward_kernel,
                backward_kernel_name: backward_kernel,
            },
        ):
            forward.captured_call[1]["launch"]("stream", "A", "B", "SFA", "SFB", "PROB", "C", "D", "AMAX")
            backward.captured_call[1]["launch"](
                "stream",
                "A",
                "B",
                "C",
                "SFA",
                "SFB",
                "PROB",
                "D",
                "DPROB",
                "AMAX",
            )

        self.assertEqual(
            calls["forward_arguments"],
            ("A", "B", "SFA", "SFB", "C", "D", "PROB", "AMAX", None, None, ("Float32", 0.25), 10, "stream"),
        )
        self.assertEqual(
            calls["backward_arguments"],
            ("A", "B", "SFA", "SFB", "C", "D", "PROB", "DPROB", "AMAX", None, None, ("Float32", 0.5), 10, "stream"),
        )
        self.assertEqual(set(calls["forward_keywords"]), {"epilogue_op"})
        self.assertEqual(set(calls["backward_keywords"]), {"epilogue_op"})

    def test_wrappers_are_jitted_with_static_configuration(self):
        for wrapper_name in ("gemm_srelu_wrapper_sm100", "gemm_dsrelu_wrapper_sm100"):
            static = self.jit_static_argnames[wrapper_name]
            self.assertIn("mma_tiler_mn", static)
            self.assertIn("cluster_shape_mn", static)
            self.assertIn("sf_vec_size", static)
            self.assertIn("a_layout", static)
            self.assertIn("b_layout", static)

    def test_validation_rejects_bad_scale_shape_and_partial_mma_rows(self):
        a, b, _, sfa, sfb, prob = self._samples()
        bad_sfa = _Array((2, 2, 3, 32, 4, 4), "float8_e8m0fnu")
        with self.assertRaisesRegex(ValueError, "SFA must have shape"):
            self.srelu.GemmSreluSm100(a, b, bad_sfa, sfb, prob, sf_vec_size=32)(a, b, bad_sfa, sfb, prob)

        partial_a, partial_b, _, partial_sfa, partial_sfb, partial_prob = self._samples(m=129)
        with self.assertRaisesRegex(ValueError, "CTA_TILE_M=128"):
            self.srelu.GemmSreluSm100(
                partial_a,
                partial_b,
                partial_sfa,
                partial_sfb,
                partial_prob,
                sf_vec_size=32,
            )(partial_a, partial_b, partial_sfa, partial_sfb, partial_prob)

    def test_shared_op_normalizes_legacy_uint8_storage_to_logical_fp4(self):
        make_desc = self.tensor_module.make_compact_tensor_desc
        a = make_desc(dtype=_DataType.UINT8, shape=(128, 128, 1), stride_order=(1, 0, 2), name="A")
        b = make_desc(dtype=_DataType.UINT8, shape=(128, 128, 1), stride_order=(1, 0, 2), name="B")
        c = make_desc(dtype=_DataType.BFLOAT16, shape=(128, 128, 1), stride_order=(1, 0, 2), name="C")
        d = make_desc(dtype=_DataType.BFLOAT16, shape=(128, 128, 1), stride_order=(1, 0, 2), name="D")
        sfa = make_desc(dtype=_DataType.FP8_E8M0, shape=(32, 4, 1, 4, 2, 1), stride_order=(3, 1, 0, 4, 2, 5), name="SFA")
        sfb = make_desc(dtype=_DataType.FP8_E8M0, shape=(32, 4, 1, 4, 2, 1), stride_order=(3, 1, 0, 4, 2, 5), name="SFB")
        prob = make_desc(dtype=_DataType.FLOAT, shape=(128, 1, 1), stride_order=(0, 1, 2), name="prob")
        op = self.srelu.GemmSreluSm100Op(
            a=a,
            b=b,
            c=c,
            d=d,
            sfa=sfa,
            sfb=sfb,
            prob=prob,
            sf_vec_size=16,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )

        self.assertTrue(op.check_support())
        self.assertEqual(op.ab_dtype, _DataType.FP4_E2M1)

        with self.assertRaisesRegex(ValueError, "at most 4"):
            self.srelu.GemmSreluSm100Op(
                a=a,
                b=b,
                c=c,
                d=d,
                sfa=sfa,
                sfb=sfb,
                prob=prob,
                sf_vec_size=16,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(8, 1),
            ).check_support()

    def test_relu_ops_enforce_forward_and_backward_specific_signatures(self):
        make_desc = self.tensor_module.make_compact_tensor_desc
        a = make_desc(dtype=_DataType.FP4_E2M1, shape=(128, 128, 1), stride_order=(1, 0, 2), name="A")
        b = make_desc(dtype=_DataType.FP4_E2M1, shape=(128, 128, 1), stride_order=(1, 0, 2), name="B")
        c = make_desc(dtype=_DataType.BFLOAT16, shape=(128, 128, 1), stride_order=(1, 0, 2), name="C")
        d = make_desc(dtype=_DataType.BFLOAT16, shape=(128, 128, 1), stride_order=(1, 0, 2), name="D")
        dprob = make_desc(dtype=_DataType.FLOAT, shape=(128, 1, 1), stride_order=(0, 1, 2), name="dprob")
        sfa = make_desc(dtype=_DataType.FP8_E8M0, shape=(32, 4, 1, 4, 2, 1), stride_order=(3, 1, 0, 4, 2, 5), name="SFA")
        sfb = make_desc(dtype=_DataType.FP8_E8M0, shape=(32, 4, 1, 4, 2, 1), stride_order=(3, 1, 0, 4, 2, 5), name="SFB")
        prob = make_desc(dtype=_DataType.FLOAT, shape=(128, 1, 1), stride_order=(0, 1, 2), name="prob")
        arguments = dict(
            a=a,
            b=b,
            c=c,
            d=d,
            sfa=sfa,
            sfb=sfb,
            prob=prob,
            sf_vec_size=16,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )

        with self.assertRaisesRegex(ValueError, "dprob is only part of"):
            self.srelu.GemmSreluSm100Op(**arguments, dprob=dprob).check_support()
        with self.assertRaisesRegex(ValueError, "dprob is required"):
            self.dsrelu.GemmDsreluSm100Op(**arguments).check_support()
        self.assertTrue(self.dsrelu.GemmDsreluSm100Op(**arguments, dprob=dprob).check_support())

        fp8_c = make_desc(dtype=_DataType.FP8_E4M3, shape=(128, 128, 1), stride_order=(1, 0, 2), name="C")
        with self.assertRaisesRegex(ValueError, "dsReLU C must use"):
            self.dsrelu.GemmDsreluSm100Op(**{**arguments, "c": fp8_c}, dprob=dprob).check_support()

    def test_shared_op_rejects_unimplemented_fp8_output(self):
        make_desc = self.tensor_module.make_compact_tensor_desc
        a = make_desc(dtype=_DataType.FP8_E4M3, shape=(128, 128, 1), stride_order=(1, 0, 2), name="A")
        b = make_desc(dtype=_DataType.FP8_E4M3, shape=(128, 128, 1), stride_order=(1, 0, 2), name="B")
        c = make_desc(dtype=_DataType.BFLOAT16, shape=(128, 128, 1), stride_order=(1, 0, 2), name="C")
        d = make_desc(dtype=_DataType.FP8_E4M3, shape=(128, 128, 1), stride_order=(1, 0, 2), name="D")
        sfa = make_desc(dtype=_DataType.FP8_E8M0, shape=(32, 4, 1, 4, 1, 1), stride_order=(3, 1, 0, 4, 2, 5), name="SFA")
        sfb = make_desc(dtype=_DataType.FP8_E8M0, shape=(32, 4, 1, 4, 1, 1), stride_order=(3, 1, 0, 4, 2, 5), name="SFB")
        prob = make_desc(dtype=_DataType.FLOAT, shape=(128, 1, 1), stride_order=(0, 1, 2), name="prob")
        arguments = dict(
            a=a,
            b=b,
            c=c,
            d=d,
            sfa=sfa,
            sfb=sfb,
            prob=prob,
            sf_vec_size=32,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )
        with self.assertRaisesRegex(NotImplementedError, "SFD generation is not implemented"):
            self.srelu.GemmSreluSm100Op(**arguments).check_support()

        sfd = make_desc(dtype=_DataType.FP8_E8M0, shape=(32, 4, 1, 4, 1, 1), stride_order=(3, 1, 0, 4, 2, 5), name="SFD")
        norm_const = make_desc(dtype=_DataType.FLOAT, shape=(1,), name="norm_const")
        with self.assertRaisesRegex(NotImplementedError, "SFD generation is not implemented"):
            self.srelu.GemmSreluSm100Op(
                **arguments,
                sfd=sfd,
                norm_const=norm_const,
            ).check_support()

    def test_shared_op_rejects_scale_formats_that_do_not_match_the_mma(self):
        make_desc = self.tensor_module.make_compact_tensor_desc

        def descriptor(dtype, shape, stride_order, name):
            return make_desc(dtype=dtype, shape=shape, stride_order=stride_order, name=name)

        c = descriptor(_DataType.BFLOAT16, (128, 128, 1), (1, 0, 2), "C")
        d = descriptor(_DataType.BFLOAT16, (128, 128, 1), (1, 0, 2), "D")
        prob = descriptor(_DataType.FLOAT, (128, 1, 1), (0, 1, 2), "prob")

        cases = (
            (_DataType.FP8_E4M3, _DataType.FP8_E4M3, 16, 2, "FP8 inputs require FP8_E8M0"),
            (_DataType.FP4_E2M1, _DataType.FP8_E4M3, 32, 1, "FP4 inputs with sf_vec_size=32 require FP8_E8M0"),
        )
        for ab_dtype, sf_dtype, sf_vec_size, rest_k, message in cases:
            a = descriptor(ab_dtype, (128, 128, 1), (1, 0, 2), "A")
            b = descriptor(ab_dtype, (128, 128, 1), (1, 0, 2), "B")
            sfa = descriptor(sf_dtype, (32, 4, 1, 4, rest_k, 1), (3, 1, 0, 4, 2, 5), "SFA")
            sfb = descriptor(sf_dtype, (32, 4, 1, 4, rest_k, 1), (3, 1, 0, 4, 2, 5), "SFB")
            with self.subTest(ab_dtype=ab_dtype, sf_dtype=sf_dtype, sf_vec_size=sf_vec_size), self.assertRaisesRegex(ValueError, message):
                self.srelu.GemmSreluSm100Op(
                    a=a,
                    b=b,
                    c=c,
                    d=d,
                    sfa=sfa,
                    sfb=sfb,
                    prob=prob,
                    sf_vec_size=sf_vec_size,
                    mma_tiler_mn=(128, 128),
                    cluster_shape_mn=(1, 1),
                ).check_support()

    def test_jax_modules_do_not_load_torch_or_kernels(self):
        self.assertNotIn(f"{_PACKAGE}.gemm_srelu.api", sys.modules)
        self.assertNotIn(f"{_PACKAGE}.gemm_dsrelu.api", sys.modules)
        self.assertNotIn(f"{_PACKAGE}.gemm_srelu.dense_blockscaled_gemm_persistent_srelu_quant", sys.modules)
        self.assertNotIn(f"{_PACKAGE}.gemm_dsrelu.dense_blockscaled_gemm_persistent_dsrelu_quant", sys.modules)


if __name__ == "__main__":
    unittest.main()
