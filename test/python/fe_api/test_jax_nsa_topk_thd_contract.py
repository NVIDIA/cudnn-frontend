# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts for packed JAX NSA top-K reduction."""

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
_NSA_ROOT = _CUDNN_ROOT / "native_sparse_attention"
_TOPK_ROOT = _NSA_ROOT / "top_k"
_PACKAGE = "cudnn_jax_nsa_topk_thd_contract_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    INT32 = auto()
    INT64 = auto()


_DTYPE_TO_CUDNN = {
    "float16": _DataType.HALF,
    "bfloat16": _DataType.BFLOAT16,
    "float32": _DataType.FLOAT,
    "int32": _DataType.INT32,
    "int64": _DataType.INT64,
}
_CUDNN_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CUDNN.items()}


class _Array:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    def __getitem__(self, key):
        if key == (None, Ellipsis):
            return _Array((1, *self.shape), self.dtype)
        raise TypeError(f"unsupported test index {key!r}")


class _TensorSpec:
    def __init__(self, *, layout, mode, divisibility=None):
        self.layout = layout
        self.mode = mode
        self.divisibility = divisibility


class JaxNsaTopkThdContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        nsa_name = f"{_PACKAGE}.native_sparse_attention"
        nsa = types.ModuleType(nsa_name)
        nsa.__path__ = [str(_NSA_ROOT)]
        nsa.__package__ = nsa_name
        nsa.__spec__ = ModuleSpec(nsa_name, loader=None, is_package=True)
        sys.modules[nsa_name] = nsa

        topk_name = f"{nsa_name}.top_k"
        topk = types.ModuleType(topk_name)
        topk.__path__ = [str(_TOPK_ROOT)]
        topk.__package__ = topk_name
        topk.__spec__ = ModuleSpec(topk_name, loader=None, is_package=True)
        sys.modules[topk_name] = topk

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
            def from_shape(
                cls,
                shape,
                dtype,
                *,
                name="",
                mode=None,
                public_stride_order=None,
                init_value=None,
            ):
                return JaxApiBase._to_tensor_desc(
                    _Array(shape, dtype),
                    name,
                    mode=mode,
                    public_stride_order=public_stride_order,
                    init_value=init_value,
                )

            @property
            def cudnn_dtype(self):
                return _DTYPE_TO_CUDNN.get(self.dtype, _DataType.NOT_SET)

        class JaxApiBase:
            @staticmethod
            def _resolve_compute_capability(target, supported, operation_name):
                del operation_name
                resolved = 100 if target is None else target
                if resolved not in supported:
                    raise ValueError(f"unsupported target {resolved}")
                return resolved

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
                    public_shape, tuple(public_stride_order)
                )
                canonical_axis_by_public_axis = layout_module.to_public_axes(
                    tuple(range(len(public_shape))), mode
                )
                desc = JaxTensorDesc(
                    dtype=value.dtype,
                    shape=layout_module.to_canonical_axes(public_shape, mode),
                    stride=layout_module.to_canonical_axes(public_stride, mode),
                    stride_order=tuple(
                        canonical_axis_by_public_axis[axis]
                        for axis in public_stride_order
                    ),
                    name=name,
                    init_value=init_value,
                )
                object.__setattr__(desc, "mode", mode)
                return desc

            @staticmethod
            def _check_tensor_signature(value, expected, *, mode=None):
                if mode is None:
                    mode = expected.mode
                actual_shape = layout_module.to_canonical_axes(tuple(value.shape), mode)
                if actual_shape != expected.shape:
                    raise ValueError(f"{expected.name} shape mismatch")
                if (
                    _DTYPE_TO_CUDNN.get(value.dtype, _DataType.NOT_SET)
                    != expected.cudnn_dtype
                ):
                    raise ValueError(f"{expected.name} dtype mismatch")

            @staticmethod
            def _to_tensor_spec(desc, *, mode=None, divisibility=None):
                if mode is None:
                    mode = desc.mode
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

            def _call_kernel(
                self,
                inputs,
                *,
                launch,
                output_descs,
                input_descs=None,
                input_spec=None,
                output_spec=None,
                **options,
            ):
                if input_descs is not None:
                    for value, desc in zip(inputs, input_descs):
                        self._check_tensor_signature(value, desc)
                if input_spec is None:
                    input_spec = tuple(
                        self._to_tensor_spec(desc) for desc in input_descs or ()
                    )
                if output_spec is None:
                    output_spec = tuple(
                        self._to_tensor_spec(desc) for desc in output_descs
                    )
                self.captured_call = {
                    "inputs": tuple(inputs),
                    "launch": launch,
                    "output_descs": tuple(output_descs),
                    "input_spec": tuple(input_spec),
                    "output_spec": tuple(output_spec),
                    **options,
                }
                return tuple(
                    _Array(
                        layout_module.to_public_axes(desc.shape, spec.mode),
                        _CUDNN_TO_DTYPE[desc.cudnn_dtype],
                    )
                    for desc, spec in zip(output_descs, output_spec)
                )

        internal.JaxApiBase = JaxApiBase
        internal.JaxTensorDesc = JaxTensorDesc
        internal.TupleDict = result_module.TupleDict

        datatypes = types.ModuleType(f"{internal_name}.datatypes")
        datatypes.normalize_jax_dtype = lambda value, default, _name: (
            default if value is None else value
        )
        sys.modules[datatypes.__name__] = datatypes

        fake_jax = types.ModuleType("jax")
        fake_jax.__path__ = []
        fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        fake_jax.ShapeDtypeStruct = _Array
        cls.static_argnames = {}

        def jit(function=None, *, static_argnames=()):
            def decorate(target):
                cls.static_argnames[target.__name__] = tuple(static_argnames)
                return target

            return decorate if function is None else decorate(function)

        fake_jax.jit = jit
        fake_jnp = types.ModuleType("jax.numpy")
        fake_jnp.float16 = "float16"
        fake_jnp.bfloat16 = "bfloat16"
        fake_jnp.float32 = "float32"
        fake_jnp.int32 = "int32"
        fake_jnp.dtype = lambda value: value
        fake_jnp.reshape = lambda value, shape: _Array(shape, value.dtype)
        fake_jnp.transpose = lambda value, axes: _Array(
            tuple(value.shape[axis] for axis in axes), value.dtype
        )
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
                cls.module = importlib.import_module(f"{topk_name}.jax")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    @staticmethod
    def _packed_samples(cum_dtype="int32"):
        return (
            _Array((20, 8, 128), "bfloat16"),
            _Array((18, 2, 128), "bfloat16"),
            _Array((20, 8, 1), "float32"),
            _Array((3,), cum_dtype),
            _Array((3,), cum_dtype),
        )

    def test_packed_thd_uses_explicit_offsets_and_virtual_kernel_views(self):
        q, k, lse, cum_q, cum_k = self._packed_samples()
        api = self.module.TopKReduction(
            q,
            k,
            lse,
            cum_q,
            cum_k,
            max_s_q=12,
            max_s_k=10,
            layout="T,H,D",
            target_compute_capability=100,
        )
        result = api(q, k, lse, cum_q, cum_k)

        self.assertEqual(api.input_layout, "THD")
        self.assertEqual(api.batch, 2)
        self.assertEqual(api.q_kernel_desc.shape, (1, 8, 20, 128))
        self.assertEqual(api.k_kernel_desc.shape, (1, 2, 18, 128))
        self.assertEqual(api.q_kernel_desc.stride_order, (3, 1, 2, 0))
        self.assertEqual(api.lse_kernel_desc.shape, (1, 8, 20))
        self.assertEqual(
            tuple(value.shape for value in api.captured_call["inputs"]),
            ((1, 8, 20, 128), (1, 2, 18, 128), (1, 8, 20), (3,), (3,)),
        )
        self.assertEqual(
            tuple(spec.layout for spec in api.captured_call["input_spec"]),
            ((3, 1, 2, 0), (3, 1, 2, 0), (2, 1, 0), (0,), (0,)),
        )
        self.assertEqual(api.captured_call["launch"].__name__, "_launch_packed_kernel")
        self.assertEqual(result["topk_scores_tensor"].shape, (20, 2, 16))
        self.assertEqual(result["topk_indices_tensor"].shape, (20, 2, 16))
        self.assertEqual(api.scores_desc.init_value, float("-inf"))
        self.assertEqual(api.indices_desc.init_value, -1)

    def test_fixed_bhsd_contract_is_preserved(self):
        q = _Array((2, 8, 20, 128), "float16")
        k = _Array((2, 2, 18, 128), "float16")
        lse = _Array((2, 8, 20), "float32")
        api = self.module.TopKReduction(
            q,
            k,
            lse,
            target_compute_capability=100,
        )
        result = api(q, k, lse)

        self.assertEqual(api.input_layout, "BHSD")
        self.assertEqual(len(api.captured_call["inputs"]), 3)
        self.assertEqual(api.captured_call["launch"].__name__, "_launch_kernel")
        self.assertEqual(api.captured_call["input_spec"][0].layout, (3, 1, 2, 0))
        self.assertEqual(result["topk_scores_tensor"].shape, (2, 2, 20, 16))
        self.assertEqual(result["topk_indices_tensor"].shape, (2, 2, 20, 16))

    def test_packed_metadata_is_rejected_before_lowering(self):
        q, k, lse, cum_q, cum_k = self._packed_samples()
        with self.assertRaisesRegex(ValueError, "max_s_q and max_s_k"):
            self.module.TopKReduction(q, k, lse, cum_q, cum_k)
        with self.assertRaisesRegex(ValueError, "both required"):
            self.module.TopKReduction(
                q,
                k,
                lse,
                cum_q,
                None,
                max_s_q=12,
                max_s_k=10,
            )

        _, _, _, bad_cum_q, bad_cum_k = self._packed_samples("int64")
        with self.assertRaisesRegex(ValueError, "dtype int32"):
            self.module.TopKReduction(
                q,
                k,
                lse,
                bad_cum_q,
                bad_cum_k,
                max_s_q=12,
                max_s_k=10,
            )
        with self.assertRaisesRegex(ValueError, "BHSD layout requires rank-4"):
            self.module.TopKReduction(q, k, lse, layout="BHSD")

    def test_wrapper_marks_packed_problem_extents_and_layout_static(self):
        static = self.static_argnames["topk_reduction_wrapper"]
        self.assertIn("max_s_q", static)
        self.assertIn("max_s_k", static)
        self.assertIn("layout", static)

    def test_packed_launcher_follows_stream_inputs_outputs_order(self):
        path = _TOPK_ROOT / "jax.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        adapter = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TopKReduction"
        )
        launcher = next(
            node
            for node in adapter.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_launch_packed_kernel"
        )
        arguments = tuple(argument.arg for argument in launcher.args.args[1:])
        self.assertEqual(
            arguments,
            (
                "stream",
                "q",
                "k",
                "lse",
                "cum_seqlen_q",
                "cum_seqlen_k",
                "topk_scores",
                "topk_indices",
            ),
        )
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertFalse(
            any(
                (
                    isinstance(node, ast.Import)
                    and any(alias.name == "torch" for alias in node.names)
                )
                or (isinstance(node, ast.ImportFrom) and node.module == "torch")
                for node in imports
            )
        )


if __name__ == "__main__":
    unittest.main()
