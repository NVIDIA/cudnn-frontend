# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for DSA score-recompute operations."""

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
_DSA_ROOT = _CUDNN_ROOT / "deepseek_sparse_attention"
_SCORE_ROOT = _DSA_ROOT / "score_recompute"
_PACKAGE = "cudnn_dsa_score_op_test"


class _DataType(Enum):
    NOT_SET = auto()
    FLOAT = auto()
    BFLOAT16 = auto()
    INT32 = auto()


class DsaScoreRecomputeOpTest(unittest.TestCase):
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

        score_name = f"{dsa_name}.score_recompute"
        score = types.ModuleType(score_name)
        score.__path__ = [str(_SCORE_ROOT)]
        score.__package__ = score_name
        score.__spec__ = ModuleSpec(score_name, loader=None, is_package=True)
        sys.modules[score_name] = score

        try:
            cls.tensor = importlib.import_module(f"{_PACKAGE}._tensor_desc")
            cls.op = importlib.import_module(f"{score_name}.op")
            cls.config = importlib.import_module(f"{score_name}.config")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def desc(self, shape, dtype=_DataType.BFLOAT16, *, stride_order=None, name=""):
        return self.tensor.make_compact_tensor_desc(
            dtype=dtype,
            shape=tuple(shape),
            stride_order=stride_order,
            name=name,
        )

    def sparse(
        self, *, target=100, score_type="indexer", with_length=False, **overrides
    ):
        aux_dtype = _DataType.BFLOAT16 if score_type == "indexer" else _DataType.FLOAT
        arguments = {
            "q": self.desc((2, 4, 32, 128), name="q"),
            "k": self.desc((2, 256, 128), name="k"),
            "per_head": self.desc((2, 4, 32), aux_dtype, name="per_head"),
            "topk_indices": self.desc(
                (2, 4, 128), _DataType.INT32, name="topk_indices"
            ),
            "output": self.desc((2, 4, 128), _DataType.FLOAT, name="output"),
            "topk_length": self.desc((2, 4), _DataType.INT32, name="topk_length")
            if with_length
            else None,
            "score_type": score_type,
            "softmax_scale": 0.125 if score_type == "attention" else 1.0,
            "target_compute_capability": target,
        }
        arguments.update(overrides)
        return self.op.SparseScoreRecomputeOp(**arguments)

    def dense(self, *, target=100, score_type="indexer", thd=False, **overrides):
        aux_dtype = _DataType.BFLOAT16 if score_type == "indexer" else _DataType.FLOAT
        if thd:
            q_shape, k_shape, aux_shape = (7, 32, 128), (11, 1, 128), (7, 32)
            output_shape, denominator_shape = (7, 8), (7,)
            cu_q = self.desc((3,), _DataType.INT32, name="cu_q")
            cu_k = self.desc((3,), _DataType.INT32, name="cu_k")
            max_q, max_k = 4, 8
        else:
            q_shape, k_shape, aux_shape = (2, 4, 32, 128), (2, 8, 1, 128), (2, 4, 32)
            output_shape, denominator_shape = (2, 4, 8), (2, 4)
            cu_q = cu_k = None
            max_q = max_k = None
        arguments = {
            "q": self.desc(q_shape, name="q"),
            "k": self.desc(k_shape, name="k"),
            "per_head": self.desc(aux_shape, aux_dtype, name="per_head"),
            "output": self.desc(output_shape, _DataType.FLOAT, name="output"),
            "denominator": self.desc(
                denominator_shape, _DataType.FLOAT, name="denominator"
            ),
            "score_type": score_type,
            "scale": 0.125 if score_type == "attention" else 1.0,
            "ratio": 2,
            "target_compute_capability": target,
            "is_thd": thd,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max_q,
            "max_seqlen_k": max_k,
        }
        arguments.update(overrides)
        return self.op.DenseScoreRecomputeOp(**arguments)

    def test_sparse_resolves_sm100_and_preserves_explicit_lengths(self):
        operation = self.sparse(with_length=True)
        self.assertTrue(operation.check_support())
        self.assertEqual(operation.qhead_per_kv_head, 32)
        self.assertIsInstance(operation.config, self.config.SparseScoreKernelConfig)
        self.assertEqual(
            (operation.config.m_block_size, operation.config.n_block_size), (32, 128)
        )
        self.assertTrue(operation.config.have_topk_length)

    def test_sparse_resolves_sm90(self):
        operation = self.sparse(target=90)
        self.assertTrue(operation.check_support())
        self.assertIsInstance(operation.config, self.op.SparseScoreSm90Config)
        self.assertEqual(
            (
                operation.config.tile_m,
                operation.config.tile_n,
                operation.config.num_threads,
            ),
            (32, 64, 256),
        )

    def test_dense_bshd_and_thd_resolve_sm100(self):
        bshd = self.dense()
        self.assertTrue(bshd.check_support())
        self.assertEqual((bshd.max_seqlen_q, bshd.max_seqlen_k), (4, 8))
        self.assertIsInstance(bshd.config, self.config.DenseScoreKernelConfig)

        thd = self.dense(thd=True, score_type="attention")
        self.assertTrue(thd.check_support())
        self.assertEqual((thd.max_seqlen_q, thd.max_seqlen_k), (4, 8))

    def test_dense_sm90_thd_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(
            NotImplementedError, "host-side sequence-length reads"
        ):
            self.dense(target=90, thd=True).check_support()

    def test_complete_signatures_are_cross_checked(self):
        with self.assertRaisesRegex(ValueError, "output must have shape"):
            self.sparse(output=self.desc((2, 4, 64), _DataType.FLOAT)).check_support()
        with self.assertRaisesRegex(ValueError, "per-head tensor must have shape"):
            self.dense(per_head=self.desc((2, 4, 16))).check_support()
        with self.assertRaisesRegex(ValueError, "qhead_per_kv_head must equal"):
            self.dense(qhead_per_kv_head=16).check_support()
        with self.assertRaisesRegex(ValueError, "ratio must be >= 1"):
            self.dense(ratio=0).check_support()

    def test_native_abi_feature_axes_must_be_contiguous(self):
        unsupported_layout = self.tensor.TensorDesc(
            dtype=_DataType.BFLOAT16,
            shape=(2, 4, 32, 128),
            stride=(16384, 128, 512, 1),
            stride_order=(3, 1, 2, 0),
        )
        with self.assertRaisesRegex(
            ValueError, "Q must have its final 2 axes contiguous"
        ):
            self.sparse(q=unsupported_layout).check_support()

    def test_sm100_configuration_rejects_incompatible_topk(self):
        with self.assertRaisesRegex(
            ValueError, "multiple of the selected n_block_size"
        ):
            self.sparse(
                topk_indices=self.desc((2, 4, 96), _DataType.INT32),
                output=self.desc((2, 4, 96), _DataType.FLOAT),
            ).check_support()

    def test_rejects_signatures_the_native_kernels_cannot_cover(self):
        with self.assertRaisesRegex(ValueError, "SM90.*multiple of 128"):
            self.sparse(
                target=90,
                topk_indices=self.desc((2, 4, 96), _DataType.INT32),
                output=self.desc((2, 4, 96), _DataType.FLOAT),
            ).check_support()

        with self.assertRaisesRegex(ValueError, "requires MQA with H_kv=1"):
            self.dense(
                q=self.desc((2, 4, 64, 128)),
                k=self.desc((2, 8, 2, 128)),
                per_head=self.desc((2, 4, 64)),
            ).check_support()

    def test_common_modules_do_not_load_frameworks_or_kernels(self):
        self.sparse().check_support()
        prefix = f"{_PACKAGE}.deepseek_sparse_attention.score_recompute"
        self.assertNotIn(f"{prefix}.jax", sys.modules)
        self.assertNotIn(f"{prefix}.api", sys.modules)
        self.assertNotIn(f"{prefix}.sparse_score_recompute_sm100", sys.modules)


if __name__ == "__main__":
    unittest.main()
