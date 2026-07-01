# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contract tests for the JAX DSA wrappers."""

from __future__ import annotations

import importlib
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


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"
_TEST_PACKAGE = "cudnn_frontend_jax_dsa_contract_test"


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

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, key):
        if not (
            isinstance(key, tuple)
            and key[:-1] == (Ellipsis,)
            and isinstance(key[-1], slice)
            and key[-1].start is None
            and key[-1].step is None
        ):
            raise AssertionError(f"Unexpected test slice {key!r}")
        return _Array((*self.shape[:-1], key[-1].stop), self.dtype)


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


class JaxDsaContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = types.ModuleType(_TEST_PACKAGE)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = _TEST_PACKAGE
        sys.modules[_TEST_PACKAGE] = parent
        cls.forward = importlib.import_module(f"{_TEST_PACKAGE}.jax.indexer_forward")
        cls.top_k = importlib.import_module(f"{_TEST_PACKAGE}.jax.indexer_top_k")

        cls.bfloat16 = _DType("bfloat16", 2)
        cls.float16 = _DType("float16", 2)
        cls.float32 = _DType("float32", 4)
        cls.int32 = _DType("int32", 4)

        cls.fake_jnp = types.ModuleType("jax.numpy")
        cls.fake_jnp.bfloat16 = cls.bfloat16
        cls.fake_jnp.float16 = cls.float16
        cls.fake_jnp.float32 = cls.float32
        cls.fake_jnp.int32 = cls.int32
        cls.fake_jnp.dtype = lambda value: value

        cls.fake_jax = types.ModuleType("jax")
        cls.fake_jax.__path__ = []
        cls.fake_jax.numpy = cls.fake_jnp

        cls.fake_cutlass = types.ModuleType("cutlass")
        cls.fake_cutlass.__path__ = []
        cls.fake_cutlass_jax = types.ModuleType("cutlass.jax")
        cls.fake_cutlass_jax.TensorSpec = _TensorSpec
        cls.fake_cutlass_jax.jax_to_cutlass_dtype = lambda dtype: f"cutlass.{dtype.name}"

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
                "cutlass.jax": self.fake_cutlass_jax,
            },
        )

    def test_indexer_forward_declares_initialized_padded_result(self):
        captured = {}

        def fake_call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            result = options["outputs"][0]
            return (_Array(result.shape, result.dtype),)

        q = _Array((1, 4, 32, 128), self.bfloat16)
        k = _Array((1, 5, 1, 128), self.bfloat16)
        w = _Array((1, 4, 32), self.bfloat16)
        launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(
                self.forward,
                "_make_launcher",
                return_value=launcher,
            ),
            mock.patch.object(self.forward, "call_cutedsl", side_effect=fake_call),
        ):
            result = self.forward.indexer_forward_wrapper(q, k, w, ratio=1)

        self.assertEqual(result.scores.shape, (1, 4, 5))
        self.assertIs(captured["launcher"], launcher)
        self.assertEqual(captured["inputs"], (q, k, w))
        self.assertTrue(captured["use_static_tensors"])

        output = captured["outputs"][0]
        self.assertEqual(output.name, "scores")
        self.assertEqual(output.shape, (1, 4, 8))
        self.assertEqual(output.dtype, self.float32)
        self.assertEqual(output.initialization.value, "value")
        self.assertEqual(output.fill_value, float("-inf"))
        self.assertEqual(output.tensor_spec.layout, (2, 1, 0))
        self.assertEqual(output.tensor_spec.mode, (0, 1, 2))
        self.assertEqual(output.tensor_spec.divisibility, (None, None, 4))

    def test_indexer_top_k_declares_hidden_workspace(self):
        captured = {}

        def fake_call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        input_values = _Array((2, 64), self.float32)
        seq_lens = _Array((2,), self.int32)
        launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(
                self.top_k,
                "_make_launcher",
                return_value=launcher,
            ) as make_launcher,
            mock.patch.object(
                self.top_k,
                "call_cutedsl",
                side_effect=fake_call,
            ),
        ):
            result = self.top_k.indexer_top_k_wrapper(
                input_values,
                seq_lens,
                top_k=8,
            )

        self.assertEqual(result.indices.shape, (2, 8))
        self.assertEqual(result.values.shape, (2, 8))
        self.assertIs(captured["launcher"], launcher)
        self.assertEqual(captured["inputs"], (input_values, seq_lens))
        self.assertTrue(captured["use_static_tensors"])
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype) for spec in captured["outputs"]],
            [
                ("indices", (2, 8), self.int32),
                ("values", (2, 8), self.float32),
            ],
        )
        self.assertEqual(len(captured["workspaces"]), 1)
        workspace = captured["workspaces"][0]
        self.assertEqual(workspace.name, "extra_buffer")
        self.assertEqual(workspace.shape, (2, 2, 64))
        self.assertEqual(workspace.dtype, self.int32)
        self.assertEqual(workspace.initialization.value, "uninitialized")
        self.assertEqual(make_launcher.call_args.args[:2], ("cutlass.float32", 32))

    def test_indexer_top_k_rejects_unsupported_result_mode(self):
        input_values = _Array((2, 64), self.float16)
        seq_lens = _Array((2,), self.int32)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(
                NotImplementedError,
                "return_val=True",
            ),
        ):
            self.top_k.indexer_top_k_wrapper(
                input_values,
                seq_lens,
                top_k=8,
                return_val=False,
            )

    def test_indexer_top_k_validates_selected_output_vector_width(self):
        with self.assertRaisesRegex(ValueError, "selected output vector width"):
            self.top_k.require_supported_top_k_output(
                top_k=513,
                num_threads_per_cta=256,
                num_copy_bits=256,
                dtype_bits=32,
            )

        # Small odd top-K values stay on the scalar path and remain valid.
        self.top_k.require_supported_top_k_output(
            top_k=3,
            num_threads_per_cta=256,
            num_copy_bits=256,
            dtype_bits=32,
        )

    def test_wrapper_validation_fails_before_launch(self):
        bad_q = _Array((1, 4, 32, 128), self.float16)
        k = _Array((1, 5, 1, 128), self.bfloat16)
        w = _Array((1, 4, 32), self.bfloat16)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(
                ValueError,
                "dtype bfloat16",
            ),
            mock.patch.object(self.forward, "_make_launcher") as make_forward,
        ):
            self.forward.indexer_forward_wrapper(bad_q, k, w, ratio=1)
        make_forward.assert_not_called()

        input_values = _Array((2, 64), self.float32)
        wrong_batch = _Array((1,), self.int32)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(
                ValueError,
                "num_rows.*must equal",
            ),
            mock.patch.object(self.top_k, "_make_launcher") as make_top_k,
        ):
            self.top_k.indexer_top_k_wrapper(
                input_values,
                wrong_batch,
                top_k=8,
            )
        make_top_k.assert_not_called()


if __name__ == "__main__":
    unittest.main()
