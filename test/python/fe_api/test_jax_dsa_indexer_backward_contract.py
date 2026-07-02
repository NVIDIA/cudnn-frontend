# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for JAX DSA indexer backward."""

from __future__ import annotations

import ast
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
_TEST_PACKAGE = "cudnn_frontend_jax_dsa_indexer_bwd_contract_test"


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


class _Float32:
    def __init__(self, value=0.0):
        self.value = value


class _ScoreGradKernel:
    instances = []

    def __init__(self, *, topk):
        self.topk = topk
        self.calls = []
        self.instances.append(self)

    def __call__(self, *args):
        self.calls.append(args)


class _BackwardKernel:
    instances = []

    def __init__(
        self,
        *,
        head_dim,
        heads,
        block_I,
        topk,
        topk_indices_global,
    ):
        self.configuration = (
            head_dim,
            heads,
            block_I,
            topk,
            topk_indices_global,
        )
        self.calls = []
        self.instances.append(self)

    def __call__(self, *args):
        self.calls.append(args)


class JaxDsaIndexerBackwardContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bfloat16 = _DType("bfloat16")
        cls.float32 = _DType("float32")
        cls.int32 = _DType("int32")

        cls.fake_jnp = types.ModuleType("jax.numpy")
        cls.fake_jnp.bfloat16 = cls.bfloat16
        cls.fake_jnp.float32 = cls.float32
        cls.fake_jnp.int32 = cls.int32
        cls.fake_jnp.dtype = lambda value: value
        cls.fake_jnp.reshape = lambda value, shape: _Array(shape, value.dtype)
        cls.fake_jnp.asarray = lambda value, dtype: _Array((len(value),), dtype)

        cls.fake_jax = types.ModuleType("jax")
        cls.fake_jax.__path__ = []
        cls.fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        cls.fake_jax.numpy = cls.fake_jnp
        cls.fake_jax.ShapeDtypeStruct = lambda shape, dtype: (shape, dtype)

        cls.fake_cutlass = types.ModuleType("cutlass")
        cls.fake_cutlass.__path__ = []
        cls.fake_cutlass.Constexpr = object
        cls.fake_cutlass.Float32 = _Float32
        cls.fake_cutlass_cute = types.ModuleType("cutlass.cute")
        cls.fake_cutlass_cute.jit = _identity_jit
        cls.fake_cutlass.cute = cls.fake_cutlass_cute
        cls.fake_cutlass_jax = types.ModuleType("cutlass.jax")
        cls.fake_cutlass_jax.TensorSpec = _TensorSpec
        cls.fake_cutlass_jax.cutlass_call = None
        cls.fake_cutlass.jax = cls.fake_cutlass_jax

        cls.kernel_module_name = f"{_TEST_PACKAGE}.deepseek_sparse_attention.indexer_backward." "indexer_backward_sm100"
        cls.kernel_module = types.ModuleType(cls.kernel_module_name)
        cls.kernel_module.ScoreGradSm100 = _ScoreGradKernel
        cls.kernel_module.IndexerBackwardSm100 = _BackwardKernel

        package_paths = {
            _TEST_PACKAGE: _CUDNN_ROOT,
            f"{_TEST_PACKAGE}.deepseek_sparse_attention": (_CUDNN_ROOT / "deepseek_sparse_attention"),
            f"{_TEST_PACKAGE}.deepseek_sparse_attention.indexer_backward": (_CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_backward"),
        }
        for package_name, package_path in package_paths.items():
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            package.__package__ = package_name
            sys.modules[package_name] = package

        with cls._optional_modules():
            cls.module = import_module(f"{_TEST_PACKAGE}.deepseek_sparse_attention.indexer_backward.jax")

    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    def setUp(self):
        self.module._make_launcher.cache_clear()
        _ScoreGradKernel.instances.clear()
        _BackwardKernel.instances.clear()

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
        return mock.patch.dict(sys.modules, modules)

    @classmethod
    def _inputs(cls, *, topk=128):
        q = _Array((2, 64, 64, 128), cls.bfloat16)
        weights = _Array((2, 64, 64), cls.bfloat16)
        k = _Array((2, 256, 128), cls.bfloat16)
        score_shape = (2, 64, topk)
        attn_score = _Array(score_shape, cls.float32)
        index_score = _Array(score_shape, cls.float32)
        topk_indices = _Array(score_shape, cls.int32)
        return q, weights, k, attn_score, index_score, topk_indices

    @staticmethod
    def _fake_call(captured):
        def call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        return call

    def test_kernel_module_is_lazy(self):
        self.assertNotIn(self.kernel_module_name, sys.modules)

    def test_declares_functional_outputs_and_hidden_workspace(self):
        captured = {}
        launcher = object()
        inputs = self._inputs()

        with (
            self._optional_modules(include_kernel=True),
            mock.patch.object(self.module, "_make_launcher", return_value=launcher),
            mock.patch.object(
                self.module,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            result = self.module.indexer_backward_wrapper(*inputs)

        q, weights, k, attn_score, index_score, topk_indices = inputs
        self.assertEqual(result.d_index_q.shape, q.shape)
        self.assertEqual(result.d_weights.shape, weights.shape)
        self.assertEqual(result.d_index_k.shape, k.shape)
        self.assertIs(result.d_index_k.dtype, self.bfloat16)
        self.assertEqual(captured["inputs"][:-1], inputs)
        grad_loss = captured["inputs"][-1]
        self.assertEqual((grad_loss.shape, grad_loss.dtype), ((1,), self.float32))
        self.assertTrue(captured["use_static_tensors"])
        self.assertIs(captured["launcher"], launcher)
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype, spec.fill_value) for spec in captured["outputs"]],
            [
                ("d_index_q", q.shape, self.bfloat16, None),
                ("d_weights", weights.shape, self.bfloat16, None),
                ("d_index_k_accum", k.shape, self.float32, 0.0),
            ],
        )
        (workspace,) = captured["workspaces"]
        self.assertEqual(
            (workspace.name, workspace.shape, workspace.dtype, workspace.fill_value),
            ("grad_signal", attn_score.shape, self.float32, None),
        )
        self.assertEqual(len(captured["input_specs"]), 7)
        self.assertIs(captured["inputs"][3], attn_score)
        self.assertIs(captured["inputs"][4], index_score)
        self.assertIs(captured["inputs"][5], topk_indices)

    def test_grad_loss_array_remains_a_runtime_operand(self):
        captured = {}
        grad_loss = _Array((), self.float32)
        with (
            self._optional_modules(include_kernel=True),
            mock.patch.object(self.module, "_make_launcher", return_value=object()),
            mock.patch.object(
                self.module,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            self.module.indexer_backward_wrapper(*self._inputs(), grad_loss=grad_loss)

        operand = captured["inputs"][-1]
        self.assertEqual((operand.shape, operand.dtype), ((1,), self.float32))

    def test_launcher_orders_stages_without_mutating_score_inputs(self):
        placeholders = [object() for _ in range(12)]
        with self._optional_modules(include_kernel=True):
            launcher = self.module._make_launcher(
                heads=64,
                head_dim=128,
                topk=128,
                block_i=128,
                sm_scale=0.125,
                grad_scale=0.25,
                topk_indices_global=False,
            )
            launcher(*placeholders)

        score_grad = _ScoreGradKernel.instances[-1]
        backward = _BackwardKernel.instances[-1]
        self.assertEqual(score_grad.topk, 128)
        self.assertEqual(backward.configuration, (128, 64, 128, 128, False))

        score_args = score_grad.calls[-1]
        self.assertIs(score_args[0], placeholders[4])
        self.assertIs(score_args[1], placeholders[5])
        self.assertIs(score_args[2], placeholders[7])
        self.assertEqual(score_args[3].value, 0.25)
        self.assertIs(score_args[4], placeholders[0])
        self.assertIs(score_args[5], placeholders[11])
        self.assertIsNone(score_args[6])

        backward_args = backward.calls[-1]
        self.assertEqual(
            backward_args[:8],
            (
                placeholders[1],
                placeholders[2],
                placeholders[3],
                placeholders[8],
                placeholders[9],
                placeholders[10],
                placeholders[11],
                placeholders[6],
            ),
        )
        self.assertEqual(backward_args[8].value, 0.125)
        self.assertIs(backward_args[9], placeholders[0])

    def test_rejects_unsupported_dtype_shape_and_topk(self):
        q, weights, k, attn_score, index_score, topk_indices = self._inputs()
        with self.assertRaisesRegex(ValueError, "index_q.dtype"):
            self.module.indexer_backward_wrapper(
                _Array(q.shape, self.float32),
                weights,
                k,
                attn_score,
                index_score,
                topk_indices,
            )

        with self.assertRaisesRegex(ValueError, "heads=64 and head_dim=128"):
            self.module.indexer_backward_wrapper(
                _Array((2, 64, 32, 128), self.bfloat16),
                _Array((2, 64, 32), self.bfloat16),
                k,
                attn_score,
                index_score,
                topk_indices,
            )

        inputs = self._inputs(topk=64)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            self.module.indexer_backward_wrapper(*inputs)

    def test_child_package_declares_literal_api_surface(self):
        init_path = _CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_backward" / "__init__.py"
        tree = ast.parse(init_path.read_text())
        exports = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_API_EXPORTS" for target in node.targets)
        )
        self.assertEqual(
            ast.literal_eval(exports),
            (
                "DenseIndexerBackward",
                "IndexerBackward",
                "dense_indexer_backward_wrapper",
                "indexer_backward_wrapper",
            ),
        )


if __name__ == "__main__":
    unittest.main()
