# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for DSA sparse-attention backward metadata."""

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
_OPERATION_ROOT = _CUDNN_ROOT / "deepseek_sparse_attention" / "sparse_attention_backward"
_PACKAGE = "cudnn_dsa_sparse_attention_backward_op_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    INT32 = auto()
    UINT8 = auto()


class SparseAttentionBackwardOpContractTest(unittest.TestCase):
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

        operation_name = f"{dsa_name}.sparse_attention_backward"
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

    def _desc(self, shape, dtype=_DataType.BFLOAT16, *, stride=None, name=""):
        shape = tuple(shape)
        if stride is None:
            stride_values = [0] * len(shape)
            running = 1
            for dimension in reversed(range(len(shape))):
                stride_values[dimension] = running
                running *= max(shape[dimension], 1)
            stride = tuple(stride_values)
        else:
            stride = tuple(stride)
        stride_order = tuple(index for index, _ in sorted(enumerate(stride), key=lambda item: (item[1], shape[item[0]])))
        return self.tensor_module.TensorDesc(
            dtype=dtype,
            shape=shape,
            stride=stride,
            stride_order=stride_order,
            name=name,
        )

    def _op(self, *, head_dim=512, dtype=_DataType.BFLOAT16, with_length=True, **overrides):
        m, n, h, topk = 65, 130, 64, 32
        head_dim_v = 512 if head_dim == 576 else head_dim
        arguments = {
            "q": self._desc((m, h, head_dim), dtype, name="q"),
            "kv": self._desc((n, head_dim), dtype, name="kv"),
            "output": self._desc((m, h, head_dim_v), dtype, name="output"),
            "doutput": self._desc((m, h, head_dim_v), dtype, name="doutput"),
            "lse": self._desc((m, h), _DataType.FLOAT, name="lse"),
            "attn_sink": self._desc((h,), _DataType.FLOAT, name="attn_sink"),
            "topk_idxs": self._desc((m, topk), _DataType.INT32, name="topk_idxs"),
            "topk_length": self._desc((m,), _DataType.INT32, name="topk_length") if with_length else None,
            "dq": self._desc((m, h, head_dim), dtype, name="dq"),
            "dkv": self._desc((n, head_dim), dtype, name="dkv"),
            "d_sink": self._desc((h,), _DataType.FLOAT, name="d_sink"),
        }
        arguments.update(overrides)
        return self.op_module.SparseAttentionBackwardOp(**arguments)

    def test_validates_both_supported_head_dimensions(self):
        for head_dim, expected_v in ((512, 512), (576, 512)):
            with self.subTest(head_dim=head_dim):
                operation = self._op(head_dim=head_dim)
                self.assertIsInstance(operation, self.base_module.Op)
                self.assertTrue(operation.check_support())
                self.assertEqual(operation.head_dim_v, expected_v)
                self.assertEqual(operation.softmax_scale, 1.0 / head_dim**0.5)
                self.assertEqual(
                    (operation.total_seqlen_q, operation.total_seqlen_kv, operation.num_heads, operation.max_topk),
                    (65, 130, 64, 32),
                )

    def test_supports_fp16_and_optional_topk_length(self):
        operation = self._op(dtype=_DataType.HALF, with_length=False, softmax_scale=0.125)
        self.assertTrue(operation.check_support())
        self.assertIsNone(operation.topk_length)
        self.assertEqual(operation.softmax_scale, 0.125)

    def test_rejects_incomplete_signatures(self):
        cases = (
            ({"q": self._desc((65, 512))}, "Q must have rank 3"),
            ({"kv": self._desc((130, 256))}, "KV head dimension must match Q"),
            ({"output": self._desc((65, 64, 576))}, "O must have shape"),
            ({"doutput": self._desc((65, 64, 576))}, "dO must have shape"),
            ({"lse": self._desc((65, 32), _DataType.FLOAT)}, "LSE must have shape"),
            ({"attn_sink": self._desc((32,), _DataType.FLOAT)}, "attn_sink must have shape"),
            ({"topk_idxs": self._desc((64, 32), _DataType.INT32)}, "topk_idxs must have shape"),
            ({"topk_length": self._desc((64,), _DataType.INT32)}, "topk_length must have shape"),
            ({"dq": self._desc((64, 64, 512))}, "dQ must have shape"),
            ({"dkv": self._desc((130, 256))}, "dKV must have shape"),
            ({"d_sink": self._desc((32,), _DataType.FLOAT)}, "d_sink must have shape"),
            ({"q": self._desc((65, 64, 512), _DataType.FLOAT)}, "Q must have dtype"),
            ({"topk_idxs": self._desc((65, 32), _DataType.FLOAT)}, "topk_idxs must have dtype int32"),
            (
                {"q": self._desc((65, 64, 256)), "kv": self._desc((130, 256)), "dq": self._desc((65, 64, 256)), "dkv": self._desc((130, 256))},
                "head_dim must be one of",
            ),
            (
                {
                    "q": self._desc((65, 32, 512)),
                    "output": self._desc((65, 32, 512)),
                    "doutput": self._desc((65, 32, 512)),
                    "lse": self._desc((65, 32), _DataType.FLOAT),
                    "attn_sink": self._desc((32,), _DataType.FLOAT),
                    "dq": self._desc((65, 32, 512)),
                    "d_sink": self._desc((32,), _DataType.FLOAT),
                },
                "num_heads must be divisible",
            ),
            ({"q": self._desc((65, 64, 512), stride=(1, 65 * 512, 65))}, "must be row-major contiguous"),
            ({"block_tile": 32}, "block_tile must be 64"),
            ({"softmax_scale": float("inf")}, "softmax_scale must be finite"),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._op(**overrides).check_support()

    def test_explicit_scale_preserves_finite_values(self):
        for scale in (0.0, -0.125):
            with self.subTest(scale=scale):
                operation = self._op(softmax_scale=scale)
                self.assertTrue(operation.check_support())
                self.assertEqual(operation.softmax_scale, scale)

    def test_operation_imports_no_framework_or_kernel(self):
        source = (_OPERATION_ROOT / "op.py").read_text()
        for forbidden in ("import torch", "import jax", "import cutlass", "import cuda"):
            self.assertNotIn(forbidden, source)
        operation = self._op()
        operation.check_support()
        prefix = f"{_PACKAGE}.deepseek_sparse_attention.sparse_attention_backward"
        self.assertNotIn(f"{prefix}.api", sys.modules)
        self.assertNotIn(f"{prefix}.jax", sys.modules)
        self.assertNotIn(f"{prefix}.dsa_bwd_sm90", sys.modules)
        self.assertNotIn(f"{prefix}.dsa_bwd_sm100", sys.modules)


if __name__ == "__main__":
    unittest.main()
