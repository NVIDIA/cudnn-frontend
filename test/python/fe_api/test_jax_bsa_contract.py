# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the JAX block-sparse attention adapters."""

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
_BSA_ROOT = _CUDNN_ROOT / "block_sparse_attention"
_PACKAGE = "cudnn_jax_bsa_contract_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    INT32 = auto()


_DTYPE_TO_CUDNN = {
    "float16": _DataType.HALF,
    "bfloat16": _DataType.BFLOAT16,
    "float32": _DataType.FLOAT,
    "int32": _DataType.INT32,
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


class JaxBsaContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        bsa_name = f"{_PACKAGE}.block_sparse_attention"
        bsa = types.ModuleType(bsa_name)
        bsa.__path__ = [str(_BSA_ROOT)]
        bsa.__package__ = bsa_name
        bsa.__spec__ = ModuleSpec(bsa_name, loader=None, is_package=True)
        sys.modules[bsa_name] = bsa

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

            def compact_like(
                self, *, cudnn_dtype, shape, stride_order=None, name="", init_value=None
            ):
                if stride_order is None:
                    stride_order = tuple(reversed(range(len(shape))))
                stride = layout_module.compact_stride(tuple(shape), tuple(stride_order))
                return JaxTensorDesc(
                    dtype=_CUDNN_TO_DTYPE[cudnn_dtype],
                    shape=tuple(shape),
                    stride=stride,
                    stride_order=tuple(stride_order),
                    name=name,
                    init_value=init_value,
                )

        class JaxApiBase:
            @staticmethod
            def _resolve_compute_capability(target, supported, operation_name):
                del operation_name
                resolved = 100 if target is None else target
                if resolved not in supported:
                    raise ValueError(f"unsupported target {resolved}")
                return resolved

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

            def _call_kernel(self, inputs, *, output_descs, output_spec, **options):
                self.captured_call = {
                    "inputs": tuple(inputs),
                    "output_descs": tuple(output_descs),
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
        fake_jnp.float32 = "float32"
        fake_jnp.int32 = "int32"
        fake_jnp.iinfo = lambda _dtype: types.SimpleNamespace(max=2**31 - 1)
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
                cls.module = importlib.import_module(f"{bsa_name}.jax")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    @staticmethod
    def _forward_samples(layout="bhsd"):
        q_shape = (2, 4, 256, 128) if layout == "bhsd" else (2, 256, 4, 128)
        kv_shape = (2, 2, 512, 128) if layout == "bhsd" else (2, 512, 2, 128)
        return (
            _Array(q_shape, "bfloat16"),
            _Array(kv_shape, "bfloat16"),
            _Array(kv_shape, "bfloat16"),
            _Array((2, 2, 4, 8), "int32"),
        )

    def test_forward_infers_public_output_layout_and_cutlass_buffers(self):
        q, k, v, block_index = self._forward_samples("bshd")
        api = self.module.BlockSparseAttentionForward(
            q,
            k,
            v,
            block_index,
            block_sparse_num=8,
            layout="bshd",
            target_compute_capability=100,
        )
        result = api(q, k, v, block_index)

        self.assertEqual(api.data_mode, (0, 2, 1, 3))
        self.assertEqual(api.q_desc.shape, (2, 4, 256, 128))
        self.assertTrue(api._op.pack_gqa_effective)
        self.assertEqual(tuple(result.keys()), ("o_tensor", "lse_tensor"))
        self.assertEqual(result["o_tensor"].shape, (2, 256, 4, 128))
        self.assertEqual(result["lse_tensor"].shape, (2, 4, 256))
        self.assertEqual(api.captured_call["workspace_descs"], ())
        self.assertIn("--gpu-arch sm_100a", api.captured_call["compile_options"])

    def test_backward_declares_csr_and_accumulator_workspaces(self):
        q = _Array((2, 4, 256, 128), "bfloat16")
        k = _Array((2, 4, 512, 128), "bfloat16")
        lse = _Array((2, 4, 256), "float32")
        block_index = _Array((2, 4, 2, 8), "int32")
        api = self.module.BlockSparseAttentionBackward(
            q,
            q,
            k,
            k,
            q,
            lse,
            block_index,
            block_sparse_num=8,
            target_compute_capability=100,
        )
        result = api(q, q, k, k, q, lse, block_index)

        self.assertEqual(tuple(result.keys()), ("dq_tensor", "dk_tensor", "dv_tensor"))
        self.assertEqual(result["dq_tensor"].shape, q.shape)
        self.assertEqual(result["dk_tensor"].shape, k.shape)
        workspace_names = tuple(
            desc.name for desc in api.captured_call["workspace_descs"]
        )
        self.assertIn("counts_workspace", workspace_names)
        self.assertIn("bucket_offsets_workspace", workspace_names)
        self.assertIn("dq_accum_workspace", workspace_names)
        counts = next(
            desc
            for desc in api.captured_call["workspace_descs"]
            if desc.name == "counts_workspace"
        )
        self.assertEqual(counts.init_value, 0)

    def test_invalid_metadata_and_unsupported_paths_fail_before_lowering(self):
        q, k, v, _ = self._forward_samples()
        wrong_index = _Array((2, 4, 3, 8), "int32")
        with self.assertRaisesRegex(ValueError, "block_index shape prefix"):
            self.module.BlockSparseAttentionForward(
                q,
                k,
                v,
                wrong_index,
                block_sparse_num=8,
                pack_gqa=False,
                target_compute_capability=100,
            )

        with self.assertRaisesRegex(ValueError, "sparse_block_size=64"):
            self.module.BlockSparseAttentionForward(
                q,
                k,
                v,
                _Array((2, 4, 4, 8), "int32"),
                block_sparse_num=8,
                sparse_block_size=128,
                target_compute_capability=90,
            )

    def test_split_kv_disables_the_clc_scheduler(self):
        resolve = self.module._resolve_sm100_blk64_use_clc

        self.assertFalse(
            resolve(
                kv_splits=2,
                requested=None,
                batch=1,
                heads=4,
                seqlen_q=256,
                block_sparse_num=8,
                has_variable_block_nums=True,
            )
        )

        q = _Array((2, 4, 256, 128), "bfloat16")
        k = _Array((2, 4, 512, 128), "bfloat16")
        v = _Array((2, 4, 512, 128), "bfloat16")
        block_index = _Array((2, 4, 4, 8), "int32")
        with self.assertRaisesRegex(ValueError, r"\[1, 256\]"):
            self.module.BlockSparseAttentionForward(
                q,
                k,
                v,
                block_index,
                block_sparse_num=8,
                kv_splits=257,
                sparse_block_size=64,
                target_compute_capability=100,
            )

        q, k, v, block_index = self._forward_samples()
        with self.assertRaisesRegex(ValueError, "SM100-family blk64"):
            self.module.BlockSparseAttentionForward(
                q,
                k,
                v,
                block_index,
                block_sparse_num=8,
                use_clc=False,
                sparse_block_size=128,
                target_compute_capability=100,
            )

    def test_backward_auto_bucket_matches_kernel_tuning(self):
        choose = self.module._default_bucket_size

        self.assertEqual(choose(100, 64, 2048, 4), 1152)
        self.assertEqual(choose(100, 64, 3000, 4), 1024)
        self.assertEqual(choose(100, 128, 4096, 1), 256)

    def test_forward_auto_kv_splits_uses_static_metadata(self):
        choose = self.module._sm100_blk64_auto_kv_splits

        self.assertEqual(choose(255), 1)
        self.assertEqual(choose(256), 2)
        self.assertEqual(choose(450), 4)
        self.assertEqual(choose(900), 8)

    def test_function_wrappers_make_configuration_static(self):
        self.assertIn(
            "block_sparse_num", self.static_argnames["block_sparse_attention_forward"]
        )
        self.assertIn("layout", self.static_argnames["block_sparse_attention_forward"])
        self.assertIn(
            "bucket_size_blocks",
            self.static_argnames["block_sparse_attention_backward"],
        )
        self.assertIn(
            "target_compute_capability",
            self.static_argnames["block_sparse_attention_backward"],
        )


if __name__ == "__main__":
    unittest.main()
