# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX contracts for DSA indexer backward."""

from __future__ import annotations

import ast
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
_DSA_ROOT = _CUDNN_ROOT / "deepseek_sparse_attention"
_OPERATION_ROOT = _DSA_ROOT / "indexer_backward"
_PACKAGE = "cudnn_jax_dsa_indexer_backward_test"


class _DataType(Enum):
    NOT_SET = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    INT32 = auto()


class _DType:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name


class _Array:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    def astype(self, dtype):
        return _Array(self.shape, dtype)


class _TupleDict(dict):
    def __iter__(self):
        return iter(self.values())


class _TensorSpec:
    def __init__(self, *, mode=None, divisibility=None):
        self.mode = mode
        self.divisibility = divisibility


class _Scalar:
    def __init__(self, value):
        self.value = value


class _CuteTensor:
    def __init__(self, shape, stride):
        self.shape = tuple(shape)
        self.stride = tuple(stride)
        self.iterator = self


class _RecordedKernel:
    events = []

    def __init__(self, label, *configuration, **keywords):
        self.label = label
        self.configuration = (configuration, keywords)

    def __call__(self, *arguments):
        self.events.append((self.label, self.configuration, arguments))


class JaxIndexerBackwardContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bfloat16 = _DType("bfloat16")
        cls.float32 = _DType("float32")
        cls.int32 = _DType("int32")
        cls.dtype_to_cudnn = {
            cls.bfloat16: _DataType.BFLOAT16,
            cls.float32: _DataType.FLOAT,
            cls.int32: _DataType.INT32,
        }
        cls.cudnn_to_dtype = {value: key for key, value in cls.dtype_to_cudnn.items()}

        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        dsa_name = f"{_PACKAGE}.deepseek_sparse_attention"
        dsa = types.ModuleType(dsa_name)
        dsa.__path__ = [str(_DSA_ROOT)]
        dsa.__package__ = dsa_name
        dsa.__spec__ = ModuleSpec(dsa_name, loader=None, is_package=True)
        sys.modules[dsa_name] = dsa

        operation_name = f"{dsa_name}.indexer_backward"
        operation = types.ModuleType(operation_name)
        operation.__path__ = [str(_OPERATION_ROOT)]
        operation.__package__ = operation_name
        operation.__spec__ = ModuleSpec(operation_name, loader=None, is_package=True)
        sys.modules[operation_name] = operation

        tensor_module = importlib.import_module(f"{_PACKAGE}.common.tensor_desc")

        class JaxTensorDesc(tensor_module.TensorDesc):
            @property
            def cudnn_dtype(self):
                return cls.dtype_to_cudnn.get(self.dtype, _DataType.NOT_SET)

            def compact_like(
                self,
                *,
                cudnn_dtype,
                shape,
                stride_order=None,
                name="",
                init_value=None,
                mode=None,
            ):
                canonical = tensor_module.make_compact_tensor_desc(
                    dtype=cudnn_dtype,
                    shape=shape,
                    stride_order=stride_order,
                    name=name,
                    init_value=init_value,
                )
                desc = JaxTensorDesc(
                    dtype=cls.cudnn_to_dtype[cudnn_dtype],
                    shape=canonical.shape,
                    stride=canonical.stride,
                    stride_order=canonical.stride_order,
                    name=name,
                    init_value=init_value,
                )
                object.__setattr__(
                    desc,
                    "mode",
                    tuple(range(len(shape))) if mode is None else tuple(mode),
                )
                return desc

            def with_divisibility(self, divisibility):
                desc = JaxTensorDesc(
                    dtype=self.dtype,
                    shape=self.shape,
                    stride=self.stride,
                    stride_order=self.stride_order,
                    name=self.name,
                    init_value=self.init_value,
                )
                object.__setattr__(desc, "mode", self.mode)
                object.__setattr__(desc, "divisibility", tuple(divisibility))
                return desc

        class JaxApiBase:
            captured_call = None

            @staticmethod
            def _to_tensor_desc(value, name, *, mode=None, init_value=None):
                public_shape = tuple(value.shape)
                mode = tuple(range(len(public_shape))) if mode is None else tuple(mode)
                public_stride = [0] * len(public_shape)
                running = 1
                for dimension in reversed(range(len(public_shape))):
                    public_stride[dimension] = running
                    running *= max(public_shape[dimension], 1)
                canonical_axis_by_public = [0] * len(mode)
                for canonical_axis, public_axis in enumerate(mode):
                    canonical_axis_by_public[public_axis] = canonical_axis
                desc = JaxTensorDesc(
                    dtype=value.dtype,
                    shape=tuple(public_shape[axis] for axis in mode),
                    stride=tuple(public_stride[axis] for axis in mode),
                    stride_order=tuple(
                        canonical_axis_by_public[axis]
                        for axis in reversed(range(len(public_shape)))
                    ),
                    name=name,
                    init_value=init_value,
                )
                object.__setattr__(desc, "mode", mode)
                return desc

            @staticmethod
            def _resolve_compute_capability(target, supported, operation_name):
                del supported, operation_name
                if target is None:
                    raise RuntimeError("target required by test")
                return target

            @staticmethod
            def _compute_capability_family(target, supported):
                return max(
                    (value for value in supported if value <= target), default=None
                )

            @staticmethod
            def _check_tensor_signature(value, expected, *, mode=None):
                mode = expected.mode if mode is None else tuple(mode)
                actual_dtype = cls.dtype_to_cudnn.get(value.dtype, _DataType.NOT_SET)
                actual_shape = tuple(value.shape[axis] for axis in mode)
                if (
                    actual_shape != expected.shape
                    or actual_dtype != expected.cudnn_dtype
                ):
                    raise ValueError(f"{expected.name} tensor signature mismatch")

            @staticmethod
            def _to_tensor_spec(desc, *, mode=None, divisibility=None):
                if mode is None:
                    mode = desc.mode
                if divisibility is None:
                    divisibility = getattr(desc, "divisibility", None)
                return _TensorSpec(mode=mode, divisibility=divisibility)

            def _call_kernel(self, inputs, *, launch, **options):
                options["launch"] = launch
                input_descs = options.get("input_descs")
                if input_descs is not None:
                    for value, desc in zip(inputs, input_descs):
                        self._check_tensor_signature(value, desc)
                    derived_specs = tuple(
                        self._to_tensor_spec(desc) for desc in input_descs
                    )
                    supplied_specs = options.get("input_spec")
                    options["input_spec"] = (
                        derived_specs
                        if supplied_specs is None
                        else tuple(
                            derived if supplied is None else supplied
                            for derived, supplied in zip(
                                derived_specs, supplied_specs
                            )
                        )
                    )
                output_descs = options["output_descs"]
                derived_output_specs = tuple(
                    self._to_tensor_spec(desc) for desc in output_descs
                )
                supplied_output_specs = options.get("output_spec")
                options["output_spec"] = (
                    derived_output_specs
                    if supplied_output_specs is None
                    else tuple(
                        derived if supplied is None else supplied
                        for derived, supplied in zip(
                            derived_output_specs, supplied_output_specs
                        )
                    )
                )
                options["workspace_spec"] = tuple(
                    self._to_tensor_spec(desc)
                    for desc in options.get("workspace_descs", ())
                )
                JaxApiBase.captured_call = (inputs, options)
                specs = options.get("output_spec") or (None,) * len(
                    options["output_descs"]
                )
                results = []
                for desc, spec in zip(options["output_descs"], specs):
                    mode = (
                        desc.mode
                        if spec is None or spec.mode is None
                        else tuple(spec.mode)
                    )
                    canonical_axis_by_public = [0] * len(mode)
                    for canonical_axis, public_axis in enumerate(mode):
                        canonical_axis_by_public[public_axis] = canonical_axis
                    public_shape = tuple(
                        desc.shape[canonical_axis_by_public[public_axis]]
                        for public_axis in range(desc.ndim)
                    )
                    results.append(
                        _Array(
                            public_shape,
                            cls.cudnn_to_dtype[desc.cudnn_dtype],
                        )
                    )
                return tuple(results)

        internal = types.ModuleType(f"{_PACKAGE}._jax")
        internal.__path__ = [str(_CUDNN_ROOT / "_jax")]
        internal.__package__ = internal.__name__
        internal.__spec__ = ModuleSpec(internal.__name__, loader=None, is_package=True)
        internal.JaxApiBase = JaxApiBase
        internal.JaxTensorDesc = JaxTensorDesc
        internal.TupleDict = _TupleDict
        sys.modules[internal.__name__] = internal

        fake_jnp = types.ModuleType("jax.numpy")
        fake_jnp.bfloat16 = cls.bfloat16
        fake_jnp.float32 = cls.float32
        fake_jnp.int32 = cls.int32
        fake_jnp.dtype = lambda value: value.dtype if hasattr(value, "dtype") else value
        fake_jnp.asarray = lambda value, dtype: _Array(
            getattr(value, "shape", ()), dtype
        )
        fake_jnp.reshape = lambda value, shape: _Array(shape, value.dtype)

        fake_jax = types.ModuleType("jax")
        fake_jax.__path__ = []
        fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        fake_jax.numpy = fake_jnp
        fake_jax.ShapeDtypeStruct = _Array
        fake_jax.jit = lambda function=None, **_kwargs: (
            (lambda fn: fn) if function is None else function
        )

        try:
            with mock.patch.dict(sys.modules, {"jax": fake_jax, "jax.numpy": fake_jnp}):
                cls.module = importlib.import_module(f"{operation_name}.jax")
        except Exception:
            cls.tearDownClass()
            raise
        cls.JaxApiBase = JaxApiBase
        cls.fake_jax_modules = {"jax": fake_jax, "jax.numpy": fake_jnp}

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def setUp(self):
        self.JaxApiBase.captured_call = None
        _RecordedKernel.events.clear()

    @classmethod
    def _sparse_inputs(cls, *, topk=128):
        score_shape = (2, 64, topk)
        return (
            _Array((2, 64, 64, 128), cls.bfloat16),
            _Array((2, 64, 64), cls.bfloat16),
            _Array((2, 256, 128), cls.bfloat16),
            _Array(score_shape, cls.float32),
            _Array(score_shape, cls.float32),
            _Array(score_shape, cls.int32),
        )

    @classmethod
    def _sparse_thd_inputs(cls, *, topk=128):
        score_shape = (96, topk)
        return (
            _Array((96, 64, 128), cls.bfloat16),
            _Array((96, 64), cls.bfloat16),
            _Array((384, 128), cls.bfloat16),
            _Array(score_shape, cls.float32),
            _Array(score_shape, cls.float32),
            _Array(score_shape, cls.int32),
        )

    @classmethod
    def _sparse_sequence_major_inputs(cls, *, topk=128):
        score_shape = (64, 2, topk)
        return (
            _Array((64, 2, 64, 128), cls.bfloat16),
            _Array((64, 2, 64), cls.bfloat16),
            _Array((256, 2, 128), cls.bfloat16),
            _Array(score_shape, cls.float32),
            _Array(score_shape, cls.float32),
            _Array(score_shape, cls.int32),
        )

    @classmethod
    def _dense_bshd_inputs(cls):
        return (
            _Array((2, 64, 64, 128), cls.bfloat16),
            _Array((2, 64, 64), cls.bfloat16),
            _Array((2, 256, 128), cls.bfloat16),
            _Array((2, 64, 256), cls.float32),
            _Array((2, 64), cls.float32),
            _Array((2, 64, 256), cls.float32),
            _Array((2, 64), cls.float32),
        )

    @classmethod
    def _dense_thd_inputs(cls):
        return (
            _Array((96, 64, 128), cls.bfloat16),
            _Array((96, 64), cls.bfloat16),
            _Array((384, 128), cls.bfloat16),
            _Array((96, 256), cls.float32),
            _Array((96,), cls.float32),
            _Array((96, 256), cls.float32),
            _Array((96,), cls.float32),
        )

    @classmethod
    def _dense_sequence_major_inputs(cls):
        return (
            _Array((64, 2, 64, 128), cls.bfloat16),
            _Array((64, 2, 64), cls.bfloat16),
            _Array((256, 2, 128), cls.bfloat16),
            _Array((64, 2, 256), cls.float32),
            _Array((64, 2), cls.float32),
            _Array((64, 2, 256), cls.float32),
            _Array((64, 2), cls.float32),
        )

    def test_sparse_wrapper_declares_functional_outputs_and_zeroed_dk(self):
        inputs = self._sparse_inputs()
        with mock.patch.dict(sys.modules, self.fake_jax_modules):
            result = self.module.indexer_backward_wrapper(
                *inputs, target_compute_capability=103
            )

        self.assertEqual(
            tuple(result),
            (result["d_index_q"], result["d_weights"], result["d_index_k"]),
        )
        self.assertEqual(result["d_index_q"].shape, inputs[0].shape)
        self.assertEqual(result["d_weights"].shape, inputs[1].shape)
        self.assertEqual(result["d_index_k"].shape, inputs[2].shape)
        kernel_inputs, options = self.JaxApiBase.captured_call
        self.assertEqual(kernel_inputs[:-1], inputs)
        self.assertEqual(
            (kernel_inputs[-1].shape, kernel_inputs[-1].dtype), ((1,), self.float32)
        )
        self.assertEqual(options["output_descs"][2].init_value, 0.0)
        self.assertEqual(options["workspace_descs"][0].shape, inputs[3].shape)
        self.assertIn("--gpu-arch sm_103a", options["compile_options"])
        self.assertEqual(options["input_spec"][0].divisibility, (None, None, None, 128))
        self.assertEqual(options["input_spec"][2].divisibility, (None, None, 128))

    def test_sparse_packed_thd_signature_requires_global_indices(self):
        inputs = self._sparse_thd_inputs()
        with mock.patch.dict(sys.modules, self.fake_jax_modules):
            with self.assertRaisesRegex(
                ValueError, "requires topk_indices_global=True"
            ):
                self.module.indexer_backward_wrapper(
                    *inputs,
                    target_compute_capability=100,
                )
            result = self.module.indexer_backward_wrapper(
                *inputs,
                topk_indices_global=True,
                target_compute_capability=100,
            )

        self.assertEqual(result["d_index_q"].shape, inputs[0].shape)
        self.assertEqual(result["d_weights"].shape, inputs[1].shape)
        self.assertEqual(result["d_index_k"].shape, inputs[2].shape)
        _, options = self.JaxApiBase.captured_call
        self.assertEqual(options["input_spec"][0].divisibility, (None, None, 128))
        self.assertEqual(options["input_spec"][2].divisibility, (None, 128))

    def test_sparse_sequence_major_layouts_preserve_public_shapes(self):
        inputs = self._sparse_sequence_major_inputs()
        with mock.patch.dict(sys.modules, self.fake_jax_modules):
            result = self.module.indexer_backward_wrapper(
                *inputs,
                q_layout="SBHD",
                w_layout="SBH",
                k_layout="SBD",
                score_layout="SBK",
                target_compute_capability=100,
            )

        self.assertEqual(result["d_index_q"].shape, inputs[0].shape)
        self.assertEqual(result["d_weights"].shape, inputs[1].shape)
        self.assertEqual(result["d_index_k"].shape, inputs[2].shape)
        _, options = self.JaxApiBase.captured_call
        self.assertEqual(
            tuple(spec.mode for spec in options["input_spec"][:6]),
            (
                (1, 0, 2, 3),
                (1, 0, 2),
                (1, 0, 2),
                (1, 0, 2),
                (1, 0, 2),
                (1, 0, 2),
            ),
        )
        self.assertEqual(
            tuple(spec.mode for spec in options["output_spec"]),
            ((1, 0, 2, 3), (1, 0, 2), (1, 0, 2)),
        )
        self.assertEqual(options["workspace_spec"][0].mode, (1, 0, 2))

    def test_grad_loss_uses_a_fixed_runtime_abi_descriptor(self):
        inputs = self._sparse_inputs()
        with mock.patch.dict(sys.modules, self.fake_jax_modules):
            api = self.module.IndexerBackward(
                *inputs,
                target_compute_capability=100,
            )
        self.assertEqual(api.grad_loss_desc.shape, (1,))
        self.assertIs(api.grad_loss_desc.dtype, self.float32)

    def test_dense_bshd_declares_functional_outputs_and_runtime_grad_loss(self):
        inputs = self._dense_bshd_inputs()
        grad_loss = _Array((1,), self.float32)
        with mock.patch.dict(sys.modules, self.fake_jax_modules):
            result = self.module.dense_indexer_backward_wrapper(
                *inputs,
                grad_loss=grad_loss,
                target_compute_capability=100,
            )

        self.assertEqual(result["d_index_q"].shape, inputs[0].shape)
        self.assertEqual(result["d_weights"].shape, inputs[1].shape)
        self.assertEqual(result["d_index_k"].shape, inputs[2].shape)
        kernel_inputs, options = self.JaxApiBase.captured_call
        self.assertEqual(
            (kernel_inputs[7].shape, kernel_inputs[7].dtype), ((1,), self.float32)
        )
        self.assertEqual(options["output_descs"][2].init_value, 0.0)
        self.assertEqual(options["workspace_descs"][0].shape, inputs[3].shape)
        self.assertIn("--gpu-arch sm_100a", options["compile_options"])

    def test_dense_thd_signature_and_optional_runtime_operands(self):
        inputs = self._dense_thd_inputs()
        cu_q = _Array((3,), self.int32)
        cu_k = _Array((3,), self.int32)
        offsets = _Array((2,), self.int32)
        with mock.patch.dict(sys.modules, self.fake_jax_modules):
            api = self.module.DenseIndexerBackward(
                *inputs,
                sample_cu_seqlens_q=cu_q,
                sample_cu_seqlens_k=cu_k,
                sample_q_causal_offsets=offsets,
                max_seqlen_q=64,
                max_seqlen_k=256,
                target_compute_capability=90,
            )
            result = api(
                *inputs,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                q_causal_offsets=offsets,
            )

        self.assertTrue(api._op.is_thd)
        self.assertEqual(result["d_index_q"].shape, inputs[0].shape)
        kernel_inputs, options = self.JaxApiBase.captured_call
        self.assertEqual(kernel_inputs[-3:], (cu_q, cu_k, offsets))
        self.assertEqual(options["input_spec"][0].divisibility, (None, None, 128))
        self.assertEqual(options["input_spec"][2].divisibility, (None, 128))
        self.assertIn("--gpu-arch sm_90a", options["compile_options"])

    def test_dense_sequence_major_layouts_preserve_public_shapes(self):
        inputs = self._dense_sequence_major_inputs()
        with mock.patch.dict(sys.modules, self.fake_jax_modules):
            result = self.module.dense_indexer_backward_wrapper(
                *inputs,
                q_layout="SBHD",
                w_layout="SBH",
                k_layout="SBD",
                score_layout="SBK",
                denom_layout="SB",
                target_compute_capability=100,
            )

        self.assertEqual(result["d_index_q"].shape, inputs[0].shape)
        self.assertEqual(result["d_weights"].shape, inputs[1].shape)
        self.assertEqual(result["d_index_k"].shape, inputs[2].shape)
        _, options = self.JaxApiBase.captured_call
        self.assertEqual(
            tuple(spec.mode for spec in options["input_spec"][:7]),
            (
                (1, 0, 2, 3),
                (1, 0, 2),
                (1, 0, 2),
                (1, 0, 2),
                (1, 0),
                (1, 0, 2),
                (1, 0),
            ),
        )
        self.assertEqual(
            tuple(spec.mode for spec in options["output_spec"]),
            ((1, 0, 2, 3), (1, 0, 2), (1, 0, 2)),
        )
        self.assertEqual(options["workspace_spec"][0].mode, (1, 0, 2))

    def _kernel_modules(self):
        package = self.module.__package__

        def kernel(label):
            return lambda *args, **kwargs: _RecordedKernel(label, *args, **kwargs)

        sparse_100 = types.ModuleType(f"{package}.indexer_backward_sm100")
        sparse_100.ScoreGradSm100 = kernel("sparse_score_100")
        sparse_100.IndexerBackwardSm100 = kernel("sparse_gemm_100")
        sparse_90 = types.ModuleType(f"{package}.indexer_backward_sm90")
        sparse_90.ScoreGradSm90 = kernel("sparse_score_90")
        sparse_90.IndexerBackwardSm90 = kernel("sparse_gemm_90")
        dense_100 = types.ModuleType(f"{package}.dense_indexer_backward_sm100")
        dense_100.ScoreGradDense = kernel("dense_score_100")
        dense_100.DenseIndexerBackward2QGemmSm100 = kernel("dense_gemm_100")
        dense_90 = types.ModuleType(f"{package}.dense_indexer_backward_sm90")
        dense_90.ScoreGradDenseSm90 = kernel("dense_score_90")
        return {
            module.__name__: module
            for module in (sparse_100, sparse_90, dense_100, dense_90)
        }

    @staticmethod
    def _fake_cutlass():
        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass.Float32 = _Scalar
        cutlass.Int32 = _Scalar
        return cutlass

    @staticmethod
    def _fake_cute():
        cute = types.ModuleType("cutlass.cute")
        cute.make_layout = lambda shape, stride: types.SimpleNamespace(
            shape=tuple(shape),
            stride=tuple(stride),
        )
        cute.make_tensor = lambda _iterator, layout: _CuteTensor(
            layout.shape, layout.stride
        )
        return cute

    @staticmethod
    def _compact_cute(shape):
        stride = [0] * len(shape)
        running = 1
        for dimension in reversed(range(len(shape))):
            stride[dimension] = running
            running *= shape[dimension]
        return _CuteTensor(shape, stride)

    def test_sparse_sm100_and_sm90_launch_functionally(self):
        inputs = self._sparse_inputs()
        kernel_inputs = tuple(object() for _ in range(7))
        outputs = tuple(object() for _ in range(3))
        workspace = (object(),)
        modules = {
            **self.fake_jax_modules,
            "cutlass": self._fake_cutlass(),
            **self._kernel_modules(),
        }

        for target, labels in (
            (100, ("sparse_score_100", "sparse_gemm_100")),
            (90, ("sparse_score_90", "sparse_gemm_90")),
        ):
            with self.subTest(target=target), mock.patch.dict(sys.modules, modules):
                _RecordedKernel.events.clear()
                api = self.module.IndexerBackward(
                    *inputs, target_compute_capability=target
                )
                api.check_support()
                api._launch_kernel(object(), *kernel_inputs, *outputs, *workspace)
                self.assertEqual(
                    tuple(event[0] for event in _RecordedKernel.events), labels
                )
                score_args = _RecordedKernel.events[0][2]
                self.assertIs(score_args[-2 if target == 100 else -1], workspace[0])
                gemm_args = _RecordedKernel.events[1][2]
                self.assertIs(gemm_args[6], workspace[0])

    def test_sparse_packed_thd_launch_adds_a_synthetic_batch_view(self):
        inputs = self._sparse_thd_inputs()
        cute = self._fake_cute()
        cutlass = self._fake_cutlass()
        cutlass.cute = cute
        modules = {
            **self.fake_jax_modules,
            "cutlass": cutlass,
            "cutlass.cute": cute,
            **self._kernel_modules(),
        }
        kernel_inputs = tuple(self._compact_cute(value.shape) for value in inputs) + (
            self._compact_cute((1,)),
        )
        outputs = (
            self._compact_cute(inputs[0].shape),
            self._compact_cute(inputs[1].shape),
            self._compact_cute(inputs[2].shape),
        )
        workspace = (self._compact_cute(inputs[3].shape),)

        for target, labels in (
            (100, ("sparse_score_100", "sparse_gemm_100")),
            (90, ("sparse_score_90", "sparse_gemm_90")),
        ):
            with self.subTest(target=target), mock.patch.dict(sys.modules, modules):
                _RecordedKernel.events.clear()
                api = self.module.IndexerBackward(
                    *inputs,
                    topk_indices_global=True,
                    target_compute_capability=target,
                )
                api.check_support()
                api._launch_kernel(object(), *kernel_inputs, *outputs, *workspace)

            self.assertEqual(
                tuple(event[0] for event in _RecordedKernel.events), labels
            )
            score_args = _RecordedKernel.events[0][2]
            self.assertEqual(score_args[0].shape, (1, 96, 128))
            self.assertEqual(score_args[1].shape, (1, 96, 128))
            self.assertEqual(score_args[5].shape, (1, 96, 128))
            gemm_args = _RecordedKernel.events[1][2]
            self.assertEqual(gemm_args[0].shape, (1, 96, 64, 128))
            self.assertEqual(gemm_args[1].shape, (1, 96, 64))
            self.assertEqual(gemm_args[2].shape, (1, 384, 128))
            self.assertEqual(gemm_args[7].shape, (1, 96, 128))

    def test_dense_sm100_bshd_and_sm90_thd_launch_functionally(self):
        modules = {
            **self.fake_jax_modules,
            "cutlass": self._fake_cutlass(),
            **self._kernel_modules(),
        }
        outputs = tuple(object() for _ in range(3))
        workspace = (object(),)

        bshd = self._dense_bshd_inputs()
        with mock.patch.dict(sys.modules, modules):
            api = self.module.DenseIndexerBackward(*bshd, target_compute_capability=100)
            api.check_support()
            kernel_inputs = tuple(object() for _ in range(8))
            api._launch_kernel(object(), *kernel_inputs, *outputs, *workspace)
        self.assertEqual(
            tuple(event[0] for event in _RecordedKernel.events),
            ("dense_score_100", "dense_gemm_100"),
        )
        score_args = _RecordedKernel.events[0][2]
        self.assertIs(score_args[11], workspace[0])
        self.assertIs(score_args[12], kernel_inputs[7])

        _RecordedKernel.events.clear()
        thd = self._dense_thd_inputs()
        cu_q = _Array((3,), self.int32)
        cu_k = _Array((3,), self.int32)
        offsets = _Array((2,), self.int32)
        with mock.patch.dict(sys.modules, modules):
            api = self.module.DenseIndexerBackward(
                *thd,
                sample_cu_seqlens_q=cu_q,
                sample_cu_seqlens_k=cu_k,
                sample_q_causal_offsets=offsets,
                max_seqlen_q=64,
                max_seqlen_k=256,
                target_compute_capability=90,
            )
            api.check_support()
            kernel_inputs = tuple(object() for _ in range(11))
            api._launch_kernel(object(), *kernel_inputs, *outputs, *workspace)
        self.assertEqual(
            tuple(event[0] for event in _RecordedKernel.events),
            ("dense_score_90", "sparse_gemm_90"),
        )
        score_args = _RecordedKernel.events[0][2]
        self.assertEqual(score_args[4:7], kernel_inputs[8:11])
        self.assertIs(score_args[11], workspace[0])
        self.assertIs(score_args[12], kernel_inputs[7])
        gemm_args = _RecordedKernel.events[1][2]
        self.assertEqual(gemm_args[10:12], kernel_inputs[8:10])
        self.assertIs(gemm_args[14], kernel_inputs[10])

    def test_jax_reachable_modules_do_not_import_torch_at_module_scope(self):
        for filename in (
            "jax.py",
            "indexer_backward_sm90.py",
            "indexer_backward_sm100.py",
            "dense_indexer_backward_sm90.py",
            "dense_indexer_backward_sm100.py",
        ):
            path = _OPERATION_ROOT / filename
            tree = ast.parse(path.read_text(), filename=str(path))
            imports = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertNotIn("torch", imports, filename)


if __name__ == "__main__":
    unittest.main()
