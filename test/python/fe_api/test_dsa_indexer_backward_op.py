# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for DSA indexer-backward metadata."""

from enum import Enum, auto
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import types
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_CUDNN_ROOT = Path(__file__).resolve().parents[3] / "python" / "cudnn"
_OPERATION_ROOT = _CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_backward"
_PACKAGE = "cudnn_dsa_indexer_backward_op_test"


class _DataType(Enum):
    NOT_SET = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    INT32 = auto()


class IndexerBackwardOpContractTest(unittest.TestCase):
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
        dsa.__path__ = [str(_CUDNN_ROOT / "deepseek_sparse_attention")]
        dsa.__package__ = dsa_name
        dsa.__spec__ = ModuleSpec(dsa_name, loader=None, is_package=True)
        sys.modules[dsa_name] = dsa

        operation_name = f"{dsa_name}.indexer_backward"
        operation = types.ModuleType(operation_name)
        operation.__path__ = [str(_OPERATION_ROOT)]
        operation.__package__ = operation_name
        operation.__spec__ = ModuleSpec(operation_name, loader=None, is_package=True)
        sys.modules[operation_name] = operation

        try:
            cls.tensor_module = importlib.import_module(f"{_PACKAGE}._tensor_desc")
            cls.base_module = importlib.import_module(f"{_PACKAGE}._op")
            cls.op_module = importlib.import_module(f"{operation_name}.op")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def _desc(self, shape, dtype=_DataType.BFLOAT16, *, name=""):
        return self.tensor_module.make_compact_tensor_desc(
            dtype=dtype,
            shape=tuple(shape),
            name=name,
        )

    def _sparse(self, **overrides):
        batch, seqlen_q, seqlen_k, heads, dim, topk = 2, 64, 256, 64, 128, 128
        arguments = {
            "index_q": self._desc((batch, seqlen_q, heads, dim), name="index_q"),
            "weights": self._desc((batch, seqlen_q, heads), name="weights"),
            "index_k": self._desc((batch, seqlen_k, dim), name="index_k"),
            "d_index_q": self._desc((batch, seqlen_q, heads, dim), name="d_index_q"),
            "d_weights": self._desc((batch, seqlen_q, heads), name="d_weights"),
            "d_index_k": self._desc((batch, seqlen_k, dim), name="d_index_k"),
            "attn_score": self._desc((batch, seqlen_q, topk), _DataType.FLOAT, name="attn_score"),
            "index_score": self._desc((batch, seqlen_q, topk), _DataType.FLOAT, name="index_score"),
            "topk_indices": self._desc((batch, seqlen_q, topk), _DataType.INT32, name="topk_indices"),
        }
        arguments.update(overrides)
        return self.op_module.IndexerBackwardOp(**arguments)

    def _sparse_thd(self, **overrides):
        total_q, total_k, heads, dim, topk = 96, 384, 64, 128, 128
        arguments = {
            "index_q": self._desc((total_q, heads, dim), name="index_q"),
            "weights": self._desc((total_q, heads), name="weights"),
            "index_k": self._desc((total_k, dim), name="index_k"),
            "d_index_q": self._desc((total_q, heads, dim), name="d_index_q"),
            "d_weights": self._desc((total_q, heads), name="d_weights"),
            "d_index_k": self._desc((total_k, dim), name="d_index_k"),
            "attn_score": self._desc((total_q, topk), _DataType.FLOAT, name="attn_score"),
            "index_score": self._desc((total_q, topk), _DataType.FLOAT, name="index_score"),
            "topk_indices": self._desc((total_q, topk), _DataType.INT32, name="topk_indices"),
            "topk_indices_global": True,
        }
        arguments.update(overrides)
        return self.op_module.IndexerBackwardOp(**arguments)

    def _dense_bshd(self, **overrides):
        batch, seqlen_q, seqlen_k, heads, dim = 2, 64, 256, 64, 128
        arguments = {
            "index_q": self._desc((batch, seqlen_q, heads, dim), name="index_q"),
            "weights": self._desc((batch, seqlen_q, heads), name="weights"),
            "index_k": self._desc((batch, seqlen_k, dim), name="index_k"),
            "d_index_q": self._desc((batch, seqlen_q, heads, dim), name="d_index_q"),
            "d_weights": self._desc((batch, seqlen_q, heads), name="d_weights"),
            "d_index_k": self._desc((batch, seqlen_k, dim), name="d_index_k"),
            "attn_score": self._desc((batch, seqlen_q, seqlen_k), _DataType.FLOAT, name="attn_score"),
            "attn_l1norm": self._desc((batch, seqlen_q), _DataType.FLOAT, name="attn_l1norm"),
            "index_score": self._desc((batch, seqlen_q, seqlen_k), _DataType.FLOAT, name="index_score"),
            "index_lse": self._desc((batch, seqlen_q), _DataType.FLOAT, name="index_lse"),
        }
        arguments.update(overrides)
        return self.op_module.DenseIndexerBackwardOp(**arguments)

    def _dense_thd(self, **overrides):
        batch, total_q, total_k, max_k, heads, dim = 2, 96, 384, 256, 64, 128
        arguments = {
            "index_q": self._desc((total_q, heads, dim), name="index_q"),
            "weights": self._desc((total_q, heads), name="weights"),
            "index_k": self._desc((total_k, dim), name="index_k"),
            "d_index_q": self._desc((total_q, heads, dim), name="d_index_q"),
            "d_weights": self._desc((total_q, heads), name="d_weights"),
            "d_index_k": self._desc((total_k, dim), name="d_index_k"),
            "attn_score": self._desc((total_q, max_k), _DataType.FLOAT, name="attn_score"),
            "attn_l1norm": self._desc((total_q,), _DataType.FLOAT, name="attn_l1norm"),
            "index_score": self._desc((total_q, max_k), _DataType.FLOAT, name="index_score"),
            "index_lse": self._desc((total_q,), _DataType.FLOAT, name="index_lse"),
            "cu_seqlens_q": self._desc((batch + 1,), _DataType.INT32, name="cu_seqlens_q"),
            "cu_seqlens_k": self._desc((batch + 1,), _DataType.INT32, name="cu_seqlens_k"),
            "q_causal_offsets": self._desc((batch,), _DataType.INT32, name="q_causal_offsets"),
            "max_seqlen_q": 64,
            "max_seqlen_k": max_k,
        }
        arguments.update(overrides)
        return self.op_module.DenseIndexerBackwardOp(**arguments)

    def test_sparse_complete_signature(self):
        operation = self._sparse()
        self.assertIsInstance(operation, self.base_module.Op)
        self.assertTrue(operation.check_support())
        self.assertFalse(operation.is_thd)
        self.assertEqual(
            (operation.batch, operation.seqlen_q, operation.seqlen_k, operation.heads, operation.head_dim, operation.topk),
            (2, 64, 256, 64, 128, 128),
        )

    def test_sparse_packed_thd_complete_signature(self):
        operation = self._sparse_thd()
        self.assertTrue(operation.check_support())
        self.assertTrue(operation.is_thd)
        self.assertEqual(
            (operation.batch, operation.seqlen_q, operation.seqlen_k, operation.heads, operation.head_dim, operation.topk),
            (1, 96, 384, 64, 128, 128),
        )

    def test_sparse_packed_thd_requires_global_indices(self):
        with self.assertRaisesRegex(ValueError, "requires topk_indices_global=True"):
            self._sparse_thd(topk_indices_global=False).check_support()

    def test_sparse_rejects_invalid_output_score_and_configuration(self):
        cases = (
            ({"d_index_q": self._desc((2, 64, 32, 128))}, "d_index_q must have shape"),
            ({"index_score": self._desc((2, 64, 64), _DataType.FLOAT)}, "index_score must have shape"),
            ({"topk_indices": self._desc((2, 64, 64), _DataType.INT32)}, r"topk \(64\) must be divisible"),
            ({"block_i": 64}, "block_i must be 128"),
            ({"sm_scale": float("inf")}, "sm_scale must be finite"),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._sparse(**overrides).check_support()

    def test_scale_preserves_values_supported_by_forward(self):
        for scale in (0.0, -0.5):
            with self.subTest(scale=scale):
                operation = self._sparse(sm_scale=scale)
                self.assertTrue(operation.check_support())
                self.assertEqual(operation.sm_scale, scale)

    def test_dense_bshd_complete_signature(self):
        operation = self._dense_bshd(q_causal_offsets=self._desc((2,), _DataType.INT32))
        self.assertTrue(operation.check_support())
        self.assertFalse(operation.is_thd)
        self.assertEqual(
            (operation.batch, operation.normalization_tokens, operation.total_k, operation.max_seqlen_q, operation.max_seqlen_k),
            (2, 128, 512, 64, 256),
        )

    def test_dense_thd_complete_signature(self):
        operation = self._dense_thd()
        self.assertTrue(operation.check_support())
        self.assertTrue(operation.is_thd)
        self.assertEqual(
            (operation.batch, operation.normalization_tokens, operation.total_k, operation.max_seqlen_q, operation.max_seqlen_k),
            (2, 96, 384, 64, 256),
        )

    def test_dense_rejects_partial_or_invalid_thd_signature(self):
        cases = (
            ({"cu_seqlens_k": None}, "must be provided together"),
            ({"max_seqlen_q": None}, "requires max_seqlen_q and max_seqlen_k"),
            ({"q_causal_offsets": self._desc((3,), _DataType.INT32)}, "q_causal_offsets must have shape"),
            ({"attn_score": self._desc((96, 128), _DataType.FLOAT)}, "attn_score must have shape"),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._dense_thd(**overrides).check_support()

    def test_operation_imports_no_framework_or_kernel(self):
        source = (_OPERATION_ROOT / "op.py").read_text()
        for forbidden in ("import torch", "import jax", "import cutlass", "import cuda"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
