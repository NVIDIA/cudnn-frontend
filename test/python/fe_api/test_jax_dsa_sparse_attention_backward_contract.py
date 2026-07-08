# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free JAX contracts for DSA sparse-attention backward."""

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
_DSA_ROOT = _CUDNN_ROOT / "deepseek_sparse_attention"
_OPERATION_ROOT = _DSA_ROOT / "sparse_attention_backward"
_PACKAGE = "cudnn_jax_dsa_sparse_attention_backward_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    INT32 = auto()
    UINT8 = auto()


_DTYPE_TO_CUDNN = {
    "float16": _DataType.HALF,
    "bfloat16": _DataType.BFLOAT16,
    "float32": _DataType.FLOAT,
    "int32": _DataType.INT32,
    "uint8": _DataType.UINT8,
}
_CUDNN_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CUDNN.items()}


class _Array:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _TupleDict(dict):
    def __iter__(self):
        return iter(self.values())


class _TensorSpec:
    def __init__(self, *, mode=None, divisibility=None):
        self.mode = mode
        self.divisibility = divisibility


class JaxSparseAttentionBackwardContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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

        operation_name = f"{dsa_name}.sparse_attention_backward"
        operation = types.ModuleType(operation_name)
        operation.__path__ = [str(_OPERATION_ROOT)]
        operation.__package__ = operation_name
        operation.__spec__ = ModuleSpec(operation_name, loader=None, is_package=True)
        sys.modules[operation_name] = operation

        tensor_module = importlib.import_module(f"{_PACKAGE}._tensor_desc")

        class JaxTensorDesc(tensor_module.TensorDesc):
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
            captured_call = None

            @staticmethod
            def _to_tensor_desc(value, name, *, mode=None, init_value=None):
                del mode
                shape = tuple(value.shape)
                stride = []
                running = 1
                for size in reversed(shape):
                    stride.append(running)
                    running *= max(size, 1)
                stride = tuple(reversed(stride))
                return JaxTensorDesc(
                    dtype=value.dtype,
                    shape=shape,
                    stride=stride,
                    stride_order=tuple(reversed(range(len(shape)))),
                    name=name,
                    init_value=init_value,
                )

            @staticmethod
            def _resolve_compute_capability(target, supported, operation_name):
                del supported, operation_name
                if target is None:
                    raise RuntimeError("target required by dependency-free test")
                return target

            @staticmethod
            def _compute_capability_family(target, supported):
                return max((value for value in supported if value <= target), default=None)

            @staticmethod
            def _check_tensor_signature(value, expected, *, mode=None):
                del mode
                if tuple(value.shape) != expected.shape or _DTYPE_TO_CUDNN[value.dtype] != expected.cudnn_dtype:
                    raise ValueError(f"{expected.name} tensor signature mismatch")

            @staticmethod
            def _to_tensor_spec(desc, *, mode=None, divisibility=None):
                del desc
                return _TensorSpec(mode=mode, divisibility=divisibility)

            def _call_kernel(self, inputs, *, launch, **options):
                options["launch"] = launch
                JaxApiBase.captured_call = (inputs, options)
                return tuple(_Array(desc.shape, _CUDNN_TO_DTYPE[desc.cudnn_dtype]) for desc in options["output_descs"])

        internal = types.ModuleType(f"{_PACKAGE}._jax")
        internal.JaxApiBase = JaxApiBase
        internal.JaxTensorDesc = JaxTensorDesc
        internal.TupleDict = _TupleDict
        sys.modules[internal.__name__] = internal

        fake_jax = types.ModuleType("jax")
        fake_jax.jit = lambda function=None, **_kwargs: (lambda fn: fn) if function is None else function
        fake_jax.ShapeDtypeStruct = _Array

        try:
            with mock.patch.dict(sys.modules, {"jax": fake_jax}):
                cls.module = importlib.import_module(f"{operation_name}.jax")
        except Exception:
            cls.tearDownClass()
            raise
        cls.JaxApiBase = JaxApiBase

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    @staticmethod
    def _inputs(*, head_dim=512, dtype="bfloat16", with_length=False):
        m, n, h, topk = 65, 130, 64, 32
        head_dim_v = 512 if head_dim == 576 else head_dim
        values = (
            _Array((m, h, head_dim), dtype),
            _Array((n, head_dim), dtype),
            _Array((m, h, head_dim_v), dtype),
            _Array((m, h, head_dim_v), dtype),
            _Array((m, h), "float32"),
            _Array((h,), "float32"),
            _Array((m, topk), "int32"),
        )
        return (*values, _Array((m,), "int32") if with_length else None)

    def test_sm100_declares_functional_outputs_and_zeroed_workspaces(self):
        inputs = self._inputs(with_length=True)
        api = self.module.SparseAttentionBackward(
            *inputs[:-1],
            sample_topk_length=inputs[-1],
            target_compute_capability=103,
        )
        result = api(*inputs[:-1], inputs[-1])
        self.assertEqual(tuple(result), (result["dq"], result["dkv"], result["d_sink"]))
        self.assertEqual(result["dq"].shape, inputs[0].shape)
        self.assertEqual(result["dkv"].shape, inputs[1].shape)
        self.assertEqual(result["d_sink"].shape, inputs[5].shape)

        _, options = self.JaxApiBase.captured_call
        outputs = options["output_descs"]
        self.assertIsNone(outputs[0].init_value)
        self.assertIsNone(outputs[1].init_value)
        self.assertEqual(outputs[2].init_value, 0.0)
        workspaces = options["workspace_descs"]
        self.assertEqual(
            [(desc.shape, desc.cudnn_dtype, desc.init_value) for desc in workspaces],
            [
                ((1, 64, 72, 8), _DataType.UINT8, 0),
                ((1, 1, 136, 2048), _DataType.UINT8, 0),
            ],
        )
        self.assertIn("--gpu-arch sm_103a", options["compile_options"])

    def test_sm90_declares_pipeline_workspaces_and_optional_length_dummy(self):
        inputs = self._inputs(dtype="float16")
        api = self.module.SparseAttentionBackward(*inputs[:-1], target_compute_capability=90)
        api(*inputs[:-1])
        _, options = self.JaxApiBase.captured_call
        self.assertEqual(
            [(desc.shape, desc.cudnn_dtype, desc.init_value) for desc in options["workspace_descs"]],
            [
                ((1, 128, 64), _DataType.FLOAT, None),
                ((1, 128, 64), _DataType.FLOAT, None),
                ((1, 1, 192 * 512), _DataType.FLOAT, 0.0),
                ((1,), _DataType.INT32, None),
            ],
        )
        self.assertIn("--gpu-arch sm_90a", options["compile_options"])

    def test_target_specific_dtype_and_576_shape_validation(self):
        fp16 = self._inputs(dtype="float16")
        with self.assertRaisesRegex(ValueError, "SM100.*bfloat16"):
            self.module.SparseAttentionBackward(*fp16[:-1], target_compute_capability=100)(*fp16[:-1])

        inputs_576 = self._inputs(head_dim=576)
        result = self.module.SparseAttentionBackward(*inputs_576[:-1], target_compute_capability=100)(*inputs_576[:-1])
        self.assertEqual(result["dq"].shape, (65, 64, 576))
        self.assertEqual(result["dkv"].shape, (130, 576))

    def test_sm100_launcher_passes_max_topk_to_kernel(self):
        inputs = self._inputs()
        api = self.module.SparseAttentionBackward(*inputs[:-1], target_compute_capability=100)
        api.check_support()
        seen = {}

        class Kernel:
            def __init__(self, **configuration):
                seen["configuration"] = configuration

            def __call__(self, *arguments):
                seen["arguments"] = arguments

        kernel_module = types.ModuleType(f"{self.module.__package__}.dsa_bwd_sm100")
        kernel_module.FlashAttentionDSABackwardSm100 = Kernel
        cutlass = types.ModuleType("cutlass")
        cutlass.Int32 = int
        cutlass.Float32 = float
        placeholders = tuple(object() for _ in range(12))
        with mock.patch.dict(sys.modules, {"cutlass": cutlass, kernel_module.__name__: kernel_module}):
            api._launch_sm100(placeholders[:7], placeholders[7:10], placeholders[10:12], object())

        self.assertEqual(seen["configuration"]["max_topk"], 32)
        self.assertEqual(seen["configuration"]["head_dim"], 512)
        self.assertIsNone(seen["arguments"][8])

    def test_sm90_launcher_builds_flat_mqa_views_and_orders_pipeline(self):
        inputs = self._inputs()
        api = self.module.SparseAttentionBackward(*inputs[:-1], target_compute_capability=90)
        api.check_support()
        seen = {}

        class CuteTensor:
            def __init__(self, shape, stride, *, element_type="BFloat16", label=""):
                self.shape = tuple(shape)
                self.stride = tuple(stride)
                self.element_type = element_type
                self.iterator = self
                self.label = label

        def compact(shape, *, element_type="BFloat16", label=""):
            stride = []
            running = 1
            for size in reversed(shape):
                stride.append(running)
                running *= max(size, 1)
            return CuteTensor(shape, tuple(reversed(stride)), element_type=element_type, label=label)

        class Stage:
            def __init__(self, label, *configuration, **keywords):
                seen[f"{label}_configuration"] = (configuration, keywords)
                self.label = label

            def __call__(self, *arguments):
                seen[f"{self.label}_arguments"] = arguments

        kernel_module = types.ModuleType(f"{self.module.__package__}.dsa_bwd_sm90")
        kernel_module._FlashAttentionDSABackwardPreprocessSm90 = lambda *args, **kwargs: Stage("pre", *args, **kwargs)
        kernel_module.FlashAttentionDSABackwardSm90 = lambda *args, **kwargs: Stage("main", *args, **kwargs)
        kernel_module._FlashAttentionDSABackwardPostprocessSm90 = lambda *args, **kwargs: Stage("post", *args, **kwargs)

        cute = types.ModuleType("cutlass.cute")
        cute.make_layout = lambda shape, stride: types.SimpleNamespace(shape=tuple(shape), stride=tuple(stride))
        cute.make_tensor = lambda iterator, layout: CuteTensor(
            layout.shape,
            layout.stride,
            element_type=iterator.element_type,
            label=iterator.label,
        )
        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass.Int32 = int
        cutlass.Float32 = float
        cutlass.cute = cute

        kernel_inputs = (
            compact((65, 64, 512), label="q"),
            compact((130, 512), label="kv"),
            compact((65, 64, 512), label="out"),
            compact((65, 64, 512), label="dout"),
            compact((65, 64), element_type="Float32", label="lse"),
            compact((64,), element_type="Float32", label="sink"),
            compact((65, 32), element_type="Int32", label="topk"),
        )
        kernel_outputs = (
            compact((65, 64, 512), label="dq"),
            compact((130, 512), label="dkv"),
            compact((64,), element_type="Float32", label="dsink"),
        )
        kernel_workspaces = (
            compact((1, 128, 64), element_type="Float32", label="dpsum"),
            compact((1, 128, 64), element_type="Float32", label="lse_log2"),
            compact((1, 1, 192 * 512), element_type="Float32", label="dkv_accum"),
            compact((1,), element_type="Int32", label="dummy_length"),
        )
        with mock.patch.dict(
            sys.modules,
            {
                "cutlass": cutlass,
                "cutlass.cute": cute,
                kernel_module.__name__: kernel_module,
            },
        ):
            api._launch_sm90(kernel_inputs, kernel_outputs, kernel_workspaces, object())

        pre_args = seen["pre_arguments"]
        self.assertEqual(pre_args[0].shape, (1, 65, 64, 512))
        self.assertEqual(pre_args[1].shape, (1, 65, 64, 512))
        main_args = seen["main_arguments"]
        self.assertEqual(main_args[0].shape, (1, 65, 64, 512))
        self.assertEqual(main_args[1].shape, (1, 130, 1, 512))
        self.assertEqual(main_args[7].shape, (1, 65, 32))
        self.assertEqual(main_args[8].shape, (1, 1))
        main_config, main_keywords = seen["main_configuration"]
        self.assertEqual(main_config[:4], ("BFloat16", 512, 512, 64))
        self.assertEqual(main_keywords["max_topk"], 32)
        self.assertFalse(main_keywords["have_topk_length"])
        post_args = seen["post_arguments"]
        self.assertEqual(post_args[1].shape, (1, 130, 1, 512))
        self.assertEqual(post_args[2], 130)

    def test_jax_module_has_no_torch_import(self):
        source = (_OPERATION_ROOT / "jax.py").read_text()
        self.assertNotIn("import torch", source)
        for kernel_name in ("dsa_bwd_sm90.py", "dsa_bwd_sm100.py"):
            self.assertNotIn("import torch", (_OPERATION_ROOT / kernel_name).read_text())


if __name__ == "__main__":
    unittest.main()
