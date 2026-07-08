# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the dense GEMM + SwiGLU operation."""

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
_OPERATION_ROOT = _CUDNN_ROOT / "gemm_swiglu"
_PACKAGE = "cudnn_gemm_swiglu_op_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    FP8_E4M3 = auto()
    FP8_E5M2 = auto()
    FP8_E8M0 = auto()
    FP4_E2M1 = auto()


class GemmSwigluOpContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        operation_name = f"{_PACKAGE}.gemm_swiglu"
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

    def _desc(self, shape, dtype, order, name):
        shape = tuple(shape)
        stride = [0] * len(shape)
        running = 1
        for dimension in order:
            stride[dimension] = running
            running *= max(shape[dimension], 1)
        return self.tensor_module.TensorDesc(
            dtype=dtype,
            shape=shape,
            stride=tuple(stride),
            stride_order=tuple(order),
            name=name,
        )

    def _op(self, **overrides):
        m, n, k, batch = 128, 128, 128, 2
        arguments = {
            "a": self._desc((m, k, batch), _DataType.BFLOAT16, (1, 0, 2), "A"),
            "b": self._desc((n, k, batch), _DataType.BFLOAT16, (1, 0, 2), "B"),
            "ab12": self._desc((m, n, batch), _DataType.FLOAT, (1, 0, 2), "AB12"),
            "c": self._desc((m, n // 2, batch), _DataType.HALF, (1, 0, 2), "C"),
        }
        arguments.update(overrides)
        return self.op_module.GemmSwigluSm100Op(**arguments)

    def test_validates_complete_signature_and_resolves_configuration(self):
        operation = self._op(alpha=0.5)

        self.assertIsInstance(operation, self.base_module.Op)
        self.assertTrue(operation.check_support())
        self.assertEqual((operation.m, operation.n, operation.k, operation.l), (128, 128, 128, 2))
        self.assertEqual(operation.output_n, 64)
        self.assertEqual(
            (operation.a_major, operation.b_major, operation.output_major),
            ("k", "k", "n"),
        )
        self.assertEqual(operation.mma_tiler_mn, (128, 128))
        self.assertEqual(operation.cluster_shape_mn, (1, 1))
        self.assertEqual(operation.alpha, 0.5)
        self.assertNotIn(f"{_PACKAGE}.gemm_swiglu.api", sys.modules)
        self.assertNotIn(f"{_PACKAGE}.gemm_swiglu.dense_gemm_persistent_swiglu", sys.modules)

    def test_accepts_each_supported_compact_major_mode(self):
        m, n, k, batch = 128, 128, 128, 2
        for a_order, b_order, output_order, expected in (
            ((0, 1, 2), (0, 1, 2), (0, 1, 2), ("m", "n", "m")),
            ((1, 0, 2), (1, 0, 2), (1, 0, 2), ("k", "k", "n")),
        ):
            with self.subTest(expected=expected):
                operation = self._op(
                    a=self._desc((m, k, batch), _DataType.BFLOAT16, a_order, "A"),
                    b=self._desc((n, k, batch), _DataType.BFLOAT16, b_order, "B"),
                    ab12=self._desc((m, n, batch), _DataType.FLOAT, output_order, "AB12"),
                    c=self._desc((m, n // 2, batch), _DataType.HALF, output_order, "C"),
                )
                self.assertTrue(operation.check_support())
                self.assertEqual(
                    (operation.a_major, operation.b_major, operation.output_major),
                    expected,
                )

    def test_rejects_invalid_shapes_and_layouts(self):
        cases = (
            (
                {"a": self._desc((128, 128), _DataType.BFLOAT16, (1, 0), "A")},
                "A must have rank 3",
            ),
            (
                {"b": self._desc((128, 64, 2), _DataType.BFLOAT16, (1, 0, 2), "B")},
                "B shape mismatch",
            ),
            (
                {
                    "b": self._desc((127, 128, 2), _DataType.BFLOAT16, (1, 0, 2), "B"),
                    "ab12": self._desc((128, 127, 2), _DataType.FLOAT, (1, 0, 2), "AB12"),
                    "c": self._desc((128, 63, 2), _DataType.HALF, (1, 0, 2), "C"),
                },
                "N must be even",
            ),
            (
                {
                    "b": self._desc((96, 128, 2), _DataType.BFLOAT16, (1, 0, 2), "B"),
                    "ab12": self._desc((128, 96, 2), _DataType.FLOAT, (1, 0, 2), "AB12"),
                    "c": self._desc((128, 48, 2), _DataType.HALF, (1, 0, 2), "C"),
                },
                "N must be divisible by 64",
            ),
            (
                {"c": self._desc((128, 128, 2), _DataType.HALF, (1, 0, 2), "C")},
                "C must have shape",
            ),
            (
                {"c": self._desc((128, 64, 2), _DataType.HALF, (0, 1, 2), "C")},
                "AB12 and C must use the same major mode",
            ),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._op(**overrides).check_support()

    def test_rejects_invalid_standard_dtypes(self):
        cases = (
            (
                {"b": self._desc((128, 128, 2), _DataType.HALF, (1, 0, 2), "B")},
                ValueError,
                "A and B must have the same dtype",
            ),
            (
                {"ab12": self._desc((128, 128, 2), _DataType.FP8_E4M3, (1, 0, 2), "AB12")},
                NotImplementedError,
                "FP8 AB12 output is currently disabled",
            ),
            (
                {
                    "acc_dtype": _DataType.HALF,
                    "ab12": self._desc((128, 128, 2), _DataType.HALF, (1, 0, 2), "AB12"),
                },
                ValueError,
                "unsupported for float16 accumulation",
            ),
            (
                {"c": self._desc((128, 64, 2), _DataType.FLOAT, (1, 0, 2), "C")},
                ValueError,
                "C dtype must be float16 or bfloat16",
            ),
        )
        for overrides, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    self._op(**overrides).check_support()

    def test_validates_alignment_tiler_and_cluster(self):
        with self.assertRaisesRegex(ValueError, "A contiguous extent must be a multiple"):
            self._op(
                a=self._desc((128, 124, 2), _DataType.BFLOAT16, (1, 0, 2), "A"),
                b=self._desc((128, 124, 2), _DataType.BFLOAT16, (1, 0, 2), "B"),
            ).check_support()

        with self.assertRaisesRegex(ValueError, r"mma_tiler_mn\[1\]"):
            self._op(mma_tiler_mn=(128, 32)).check_support()

        with self.assertRaisesRegex(ValueError, "must contain two integers"):
            self._op(mma_tiler_mn=()).check_support()

        with self.assertRaisesRegex(ValueError, r"must be \(1, 1\)"):
            self._op(cluster_shape_mn=(2, 1)).check_support()

        with self.assertRaisesRegex(ValueError, "M must be divisible by 256"):
            self._op(mma_tiler_mn=(256, 128)).check_support()

        operation = self._op(
            a=self._desc((256, 128, 2), _DataType.BFLOAT16, (1, 0, 2), "A"),
            ab12=self._desc((256, 128, 2), _DataType.FLOAT, (1, 0, 2), "AB12"),
            c=self._desc((256, 64, 2), _DataType.HALF, (1, 0, 2), "C"),
            mma_tiler_mn=(256, 128),
        )
        self.assertTrue(operation.check_support())
        self.assertEqual(operation.cluster_shape_mn, (2, 2))

    def test_constructor_requires_canonical_descriptors_and_dtype(self):
        for name in ("a", "b", "ab12", "c"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(TypeError, f"{name} must be a TensorDesc"):
                    self._op(**{name: types.SimpleNamespace(shape=(1,))})

        with self.assertRaisesRegex(TypeError, "acc_dtype must be a cudnn.data_type"):
            self._op(acc_dtype="float32")


if __name__ == "__main__":
    unittest.main()
