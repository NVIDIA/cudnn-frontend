# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contract tests for the JAX NSA wrappers."""

from __future__ import annotations

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


def _identity_jit(fn=None, **_kwargs):
    return (lambda decorated_fn: decorated_fn) if fn is None else fn


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"
_TEST_PACKAGE = "cudnn_frontend_jax_nsa_contract_test"


class _DType:
    def __init__(self, name, itemsize):
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


class JaxNsaContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bfloat16 = _DType("bfloat16", 2)
        cls.float16 = _DType("float16", 2)
        cls.float32 = _DType("float32", 4)
        cls.int32 = _DType("int32", 4)
        cls.int64 = _DType("int64", 8)

        cls.fake_jnp = types.ModuleType("jax.numpy")
        cls.fake_jnp.bfloat16 = cls.bfloat16
        cls.fake_jnp.float16 = cls.float16
        cls.fake_jnp.float32 = cls.float32
        cls.fake_jnp.int32 = cls.int32
        cls.fake_jnp.int64 = cls.int64
        cls.fake_jnp.dtype = lambda value: value
        cls.fake_jnp.reshape = lambda value, shape: _Array(shape, value.dtype)

        cls.fake_jax = types.ModuleType("jax")
        cls.fake_jax.__path__ = []
        cls.fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        cls.fake_jax.numpy = cls.fake_jnp

        cls.fake_cutlass = types.ModuleType("cutlass")
        cls.fake_cutlass.__path__ = []
        cls.fake_cutlass.Constexpr = object
        cls.fake_cutlass_cute = types.ModuleType("cutlass.cute")
        cls.fake_cutlass_cute.jit = _identity_jit
        cls.fake_cutlass.cute = cls.fake_cutlass_cute
        cls.fake_cutlass_jax = types.ModuleType("cutlass.jax")
        cls.fake_cutlass_jax.TensorSpec = _TensorSpec
        cls.fake_cutlass_jax.jax_to_cutlass_dtype = lambda dtype: f"cutlass.{dtype.name}"
        cls.fake_cutlass.jax = cls.fake_cutlass_jax

        parent = types.ModuleType(_TEST_PACKAGE)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = _TEST_PACKAGE
        sys.modules[_TEST_PACKAGE] = parent
        with mock.patch.dict(
            sys.modules,
            {
                "jax": cls.fake_jax,
                "jax.numpy": cls.fake_jnp,
                "cutlass": cls.fake_cutlass,
                "cutlass.cute": cls.fake_cutlass_cute,
                "cutlass.jax": cls.fake_cutlass_jax,
                "torch": None,
            },
        ):
            cls.nsa = importlib.import_module(f"{_TEST_PACKAGE}.native_sparse_attention")
            cls.compression_package = importlib.import_module(f"{_TEST_PACKAGE}.native_sparse_attention.compression")
            cls.selection_package = importlib.import_module(f"{_TEST_PACKAGE}.native_sparse_attention.selection")
            cls.topk_package = importlib.import_module(f"{_TEST_PACKAGE}.native_sparse_attention.top_k")
            cls.compression = importlib.import_module(f"{_TEST_PACKAGE}.native_sparse_attention.compression.jax")
            cls.selection = importlib.import_module(f"{_TEST_PACKAGE}.native_sparse_attention.selection.jax")
            cls.topk = importlib.import_module(f"{_TEST_PACKAGE}.native_sparse_attention.top_k.jax")

    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    def _optional_modules(self):
        return mock.patch.dict(
            sys.modules,
            {
                "jax": self.fake_jax,
                "jax.numpy": self.fake_jnp,
                "cutlass": self.fake_cutlass,
                "cutlass.cute": self.fake_cutlass_cute,
                "cutlass.jax": self.fake_cutlass_jax,
            },
        )

    def test_jax_submodules_do_not_load_torch_apis(self):
        self.assertNotIn(
            f"{_TEST_PACKAGE}.native_sparse_attention.compression.api",
            sys.modules,
        )
        self.assertNotIn(
            f"{_TEST_PACKAGE}.native_sparse_attention.top_k.api",
            sys.modules,
        )
        self.assertNotIn(
            f"{_TEST_PACKAGE}.native_sparse_attention.selection.api",
            sys.modules,
        )
        self.assertNotIn(
            f"{_TEST_PACKAGE}.native_sparse_attention.compression.fmha",
            sys.modules,
        )
        self.assertNotIn(
            f"{_TEST_PACKAGE}.native_sparse_attention.top_k.nsa_top_k_reduction_fwd",
            sys.modules,
        )
        self.assertNotIn(
            f"{_TEST_PACKAGE}.native_sparse_attention.selection.NSA_select_attn_fwd_hmma",
            sys.modules,
        )
        self.assertIs(self.compression_package.jax, self.compression)
        self.assertIs(self.selection_package.jax, self.selection)
        self.assertIs(self.topk_package.jax, self.topk)

    def test_selection_attention_declares_runtime_offsets_and_safe_outputs(self):
        captured = {}

        def fake_call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        q = _Array((16, 4, 128), self.float16)
        k = _Array((16, 1, 128), self.float16)
        v = _Array((16, 1, 64), self.float16)
        block_indices = _Array((16, 1, 8), self.int32)
        block_counts = _Array((16, 1), self.int32)
        cum_q = _Array((3,), self.int32)
        cum_k = _Array((3,), self.int32)
        launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(
                self.selection,
                "_make_launcher",
                return_value=launcher,
            ) as make_launcher,
            mock.patch.object(
                self.selection,
                "call_cutedsl",
                side_effect=fake_call,
            ),
        ):
            result = self.selection.selection_attention_wrapper(
                q,
                k,
                v,
                block_indices,
                block_counts,
                cum_q,
                cum_k,
                max_s_q=8,
                max_s_k=8,
            )

        self.assertEqual((result.o_tensor.shape, result.o_tensor.dtype), ((16, 4, 64), self.float16))
        self.assertEqual((result.l_tensor.shape, result.l_tensor.dtype), ((16, 4, 1), self.float32))
        self.assertEqual((result.m_tensor.shape, result.m_tensor.dtype), ((16, 4, 1), self.float32))
        self.assertIs(captured["launcher"], launcher)
        self.assertEqual(tuple(value.shape for value in captured["inputs"][:3]), ((1, 16, 4, 128), (1, 16, 1, 128), (1, 16, 1, 64)))
        self.assertEqual(captured["inputs"][3:], (block_indices, block_counts, cum_q, cum_k))
        self.assertTrue(captured["use_static_tensors"])

        output, lse_sum, row_max = captured["outputs"]
        self.assertEqual(output.shape, (1, 16, 4, 64))
        self.assertEqual(output.fill_value, 0)
        self.assertEqual(lse_sum.shape, (1, 16, 4))
        self.assertEqual(lse_sum.fill_value, 0.0)
        self.assertEqual(row_max.shape, (1, 16, 4))
        self.assertEqual(row_max.fill_value, float("-inf"))

        config = make_launcher.call_args.kwargs
        self.assertEqual(config["element_dtype"], "cutlass.float16")
        self.assertEqual(config["gqa_group_size"], 4)
        self.assertEqual(config["max_s_q"], 8)
        self.assertAlmostEqual(config["scale_softmax"], 1.0 / (128**0.5))

    def test_selection_launcher_preserves_native_argument_order(self):
        calls = []
        kernel_options = []

        class FakeKernel:
            def __init__(self, **options):
                kernel_options.append(options)

            def __call__(self, *args):
                calls.append(args)

        kernel_name = f"{_TEST_PACKAGE}.native_sparse_attention.selection.NSA_select_attn_fwd_hmma"
        kernel_module = types.ModuleType(kernel_name)
        kernel_module.HopperSelectAttentionFwd = FakeKernel

        def float32(value=None):
            return ("Float32", value)

        with (
            self._optional_modules(),
            mock.patch.dict(sys.modules, {kernel_name: kernel_module}),
            mock.patch.object(self.fake_cutlass, "Float32", float32, create=True),
        ):
            self.selection._make_launcher.cache_clear()
            launcher = self.selection._make_launcher(
                element_dtype="float16",
                head_dim=128,
                value_dim=64,
                gqa_group_size=4,
                block_size=64,
                max_s_q=8,
                scale_softmax=0.125,
            )
            launcher(
                "stream",
                "q",
                "k",
                "v",
                "indices",
                "counts",
                "cum_q",
                "cum_k",
                "out",
                "l",
                "m",
            )
            self.selection._make_launcher.cache_clear()

        self.assertEqual(kernel_options[0]["head_dim"], 128)
        self.assertEqual(kernel_options[0]["GQA_group_size"], 4)
        self.assertEqual(
            calls[0],
            (
                "q",
                "k",
                "v",
                "out",
                "l",
                "m",
                "indices",
                "counts",
                8,
                "cum_q",
                ("Float32", 0.125),
                "stream",
            ),
        )

    def test_selection_validation_fails_before_launch(self):
        q = _Array((16, 4, 128), self.float16)
        k = _Array((16, 1, 128), self.float16)
        v = _Array((16, 1, 128), self.float16)
        indices = _Array((16, 1, 8), self.int32)
        counts = _Array((16, 1), self.int32)
        offsets = _Array((3,), self.int32)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(ValueError, "must be identical"),
            mock.patch.object(self.selection, "_make_launcher") as make_launcher,
        ):
            self.selection.selection_attention_wrapper(
                q,
                k,
                v,
                indices,
                counts,
                offsets,
                offsets,
                max_s_q=8,
                max_s_k=16,
            )
        make_launcher.assert_not_called()

    def test_compression_attention_declares_layouts_and_optional_lse(self):
        captured = {}

        def fake_call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        q = _Array((2, 4, 16, 64), self.float16)
        k = _Array((2, 2, 8, 64), self.float16)
        v = _Array((2, 2, 8, 64), self.float16)
        launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(
                self.compression,
                "_make_launcher",
                return_value=launcher,
            ) as make_launcher,
            mock.patch.object(
                self.compression,
                "call_cutedsl",
                side_effect=fake_call,
            ),
        ):
            result = self.compression.compression_attention_wrapper(
                q,
                k,
                v,
                enable_lse=True,
                o_dtype=self.bfloat16,
                scale_q=2.0,
                scale_k=0.5,
                scale_v=3.0,
                inv_scale_o=0.25,
            )

        self.assertEqual((result.o_tensor.shape, result.o_tensor.dtype), ((2, 4, 16, 64), self.bfloat16))
        self.assertEqual((result.lse_tensor.shape, result.lse_tensor.dtype), ((2, 4, 16), self.float32))
        self.assertIs(captured["launcher"], launcher)
        self.assertEqual(captured["inputs"], (q, k, v))
        self.assertTrue(captured["use_static_tensors"])
        self.assertEqual(len(captured["input_specs"]), 3)
        for spec in captured["input_specs"]:
            self.assertEqual(spec.layout, (3, 1, 2, 0))
            self.assertEqual(spec.mode, (0, 2, 1, 3))

        output, lse = captured["outputs"]
        self.assertEqual(output.name, "o_tensor")
        self.assertIsNone(output.fill_value)
        self.assertEqual(output.tensor_spec.layout, (3, 1, 2, 0))
        self.assertEqual(output.tensor_spec.mode, (0, 2, 1, 3))
        self.assertEqual(lse.name, "lse_tensor")
        self.assertEqual(lse.tensor_spec.layout, (2, 1, 0))
        self.assertEqual(lse.tensor_spec.mode, (0, 2, 1))

        config = make_launcher.call_args.kwargs
        self.assertTrue(config["enable_lse"])
        self.assertEqual(config["scale_softmax"], 0.125)
        self.assertEqual(config["scale_output"], 0.75)

    def test_compression_launcher_preserves_native_argument_order(self):
        calls = []
        kernel_options = []

        class FakeKernel:
            def __init__(self, **options):
                kernel_options.append(options)

            def __call__(self, *args):
                calls.append(args)

        package = f"{_TEST_PACKAGE}.native_sparse_attention.compression"
        kernel_module = types.ModuleType(f"{package}.fmha")
        kernel_module.BlackwellFusedMultiHeadAttentionForward = FakeKernel
        helpers_module = types.ModuleType(f"{package}.fmha_helpers")
        helpers_module.MaskType = types.SimpleNamespace(COMPRESSED_CAUSAL_MASK="compressed")

        def float32(value=None):
            return ("Float32", value)

        def int32(value):
            return ("Int32", value)

        with (
            self._optional_modules(),
            mock.patch.dict(
                sys.modules,
                {
                    kernel_module.__name__: kernel_module,
                    helpers_module.__name__: helpers_module,
                },
            ),
            mock.patch.object(self.fake_cutlass, "Float32", float32, create=True),
            mock.patch.object(self.fake_cutlass, "Int32", int32, create=True),
        ):
            self.compression._make_launcher.cache_clear()
            launcher = self.compression._make_launcher(
                batch=2,
                seqlen_q=16,
                seqlen_k=8,
                num_query_heads=4,
                num_kv_heads=2,
                head_dim=64,
                enable_lse=True,
                is_persistent=False,
                scale_softmax=0.125,
                scale_output=0.75,
            )
            launcher("stream", "q", "k", "v", "out", "lse")
            self.compression._make_launcher.cache_clear()

        self.assertEqual(kernel_options[0]["qk_acc_dtype"], float32)
        self.assertEqual(kernel_options[0]["mask_type"], "compressed")
        args = calls[0]
        self.assertEqual(args[:4], ("q", "k", "v", "out"))
        self.assertEqual(args[4][0], ("Int32", 2))
        self.assertEqual(args[4][1:4], (("Int32", 16), ("Int32", 16), ("Int32", 8)))
        self.assertEqual(args[7], "lse")
        self.assertEqual(args[-3:], (None, ("Int32", 0), "stream"))

    def test_topk_reduction_declares_initialized_outputs(self):
        captured = {}

        def fake_call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        q = _Array((2, 4, 16, 64), self.bfloat16)
        k = _Array((2, 2, 8, 64), self.bfloat16)
        lse = _Array((2, 4, 16), self.float32)
        launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(
                self.topk,
                "_make_launcher",
                return_value=launcher,
            ) as make_launcher,
            mock.patch.object(self.topk, "call_cutedsl", side_effect=fake_call),
        ):
            result = self.topk.topk_reduction_wrapper(q, k, lse)

        self.assertEqual((result.topk_scores_tensor.shape, result.topk_scores_tensor.dtype), ((2, 2, 16, 16), self.float32))
        self.assertEqual((result.topk_indices_tensor.shape, result.topk_indices_tensor.dtype), ((2, 2, 16, 16), self.int32))
        self.assertEqual(captured["inputs"], (q, k, lse))
        self.assertTrue(captured["use_static_tensors"])
        scores, indices = captured["outputs"]
        self.assertEqual(scores.fill_value, float("-inf"))
        self.assertEqual(indices.fill_value, -1)
        for spec in (scores.tensor_spec, indices.tensor_spec):
            self.assertEqual(spec.layout, (3, 1, 2, 0))
            self.assertEqual(spec.mode, (0, 1, 2, 3))
        self.assertEqual(captured["input_specs"][2].layout, (2, 1, 0))
        self.assertEqual(make_launcher.call_args.kwargs["element_dtype"], "cutlass.bfloat16")

    def test_topk_launcher_preserves_native_argument_order(self):
        calls = []

        class FakeKernel:
            def __init__(self, **options):
                self.options = options

            def __call__(self, *args):
                calls.append(args)

        kernel_name = f"{_TEST_PACKAGE}.native_sparse_attention.top_k.nsa_top_k_reduction_fwd"
        kernel_module = types.ModuleType(kernel_name)
        kernel_module.FineGrainedReductionQK = FakeKernel

        def float32(value=None):
            return ("Float32", value)

        def int32(value):
            return ("Int32", value)

        with (
            self._optional_modules(),
            mock.patch.dict(sys.modules, {kernel_name: kernel_module}),
            mock.patch.object(self.fake_cutlass, "Float32", float32, create=True),
            mock.patch.object(self.fake_cutlass, "Int32", int32, create=True),
        ):
            self.topk._make_launcher.cache_clear()
            launcher = self.topk._make_launcher(
                element_dtype="float16",
                batch=2,
                seqlen_q=16,
                seqlen_k=8,
                num_query_heads=4,
                num_kv_heads=2,
                head_dim=64,
                k_value=16,
                selection_block_size=64,
                compress_stride=32,
                is_causal=True,
                scale_softmax=0.125,
            )
            launcher("stream", "q", "k", "lse", "scores", "indices")
            self.topk._make_launcher.cache_clear()

        args = calls[0]
        self.assertEqual(args[1:6], ("q", "k", "lse", "scores", "indices"))
        self.assertEqual(args[0][0], ("Int32", 2))
        self.assertEqual(args[-3:], (None, None, "stream"))

    def test_topk_validation_fails_before_launch(self):
        q = _Array((1, 4, 16, 64), self.float16)
        k = _Array((1, 2, 8, 64), self.float16)
        lse = _Array((1, 4, 16), self.float32)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(ValueError, "positive multiple of 4"),
            mock.patch.object(self.topk, "_make_launcher") as make_launcher,
        ):
            self.topk.topk_reduction_wrapper(q, k, lse, k_value=7)
        make_launcher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
