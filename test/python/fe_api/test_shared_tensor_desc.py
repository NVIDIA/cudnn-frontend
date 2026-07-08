# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free tests for framework-neutral tensor descriptors."""

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
_PACKAGE = "cudnn_tensor_desc_test"


class _DataType(Enum):
    FLOAT = auto()
    BFLOAT16 = auto()


def _load_tensor_desc_module():
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_CUDNN_ROOT)]
    package.__package__ = _PACKAGE
    package.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
    package.data_type = _DataType
    sys.modules[_PACKAGE] = package
    return importlib.import_module(f"{_PACKAGE}._tensor_desc")


class SharedTensorDescTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tensor_desc = _load_tensor_desc_module()

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def test_compact_descriptor_defaults_to_row_major(self):
        desc = self.tensor_desc.make_compact_tensor_desc(
            dtype=_DataType.BFLOAT16,
            shape=(2, 3, 4),
            name="output",
        )

        self.assertIsInstance(desc, self.tensor_desc.TensorDesc)
        self.assertEqual(desc.dtype, _DataType.BFLOAT16)
        self.assertEqual(desc.cudnn_dtype, _DataType.BFLOAT16)
        self.assertEqual(desc.shape, (2, 3, 4))
        self.assertEqual(desc.stride, (12, 4, 1))
        self.assertEqual(desc.stride_order, (2, 1, 0))
        self.assertEqual(desc.ndim, 3)
        self.assertEqual(desc.name, "output")
        self.assertIsNone(desc.init_value)

    def test_compact_descriptor_preserves_initial_value(self):
        for init_value in (0, False, float("-inf")):
            with self.subTest(init_value=init_value):
                desc = self.tensor_desc.make_compact_tensor_desc(
                    dtype=_DataType.FLOAT,
                    shape=(2, 3),
                    init_value=init_value,
                )

                if init_value is False:
                    self.assertIs(desc.init_value, False)
                else:
                    self.assertEqual(desc.init_value, init_value)

        with self.assertRaisesRegex(TypeError, "init_value must be a real scalar or None"):
            self.tensor_desc.make_compact_tensor_desc(
                dtype=_DataType.FLOAT,
                shape=(2, 3),
                init_value=[],
            )

    def test_compact_like_derives_a_canonical_descriptor(self):
        source = self.tensor_desc.TensorDesc(
            dtype=_DataType.BFLOAT16,
            shape=(7,),
            stride=(1,),
            stride_order=(0,),
            name="input",
        )

        derived = source.compact_like(
            cudnn_dtype=_DataType.FLOAT,
            shape=(2, 3),
            stride_order=(0, 1),
            name="workspace",
            init_value=0,
        )

        self.assertIs(type(derived), self.tensor_desc.TensorDesc)
        self.assertEqual(derived.dtype, _DataType.FLOAT)
        self.assertEqual(derived.shape, (2, 3))
        self.assertEqual(derived.stride, (1, 2))
        self.assertEqual(derived.stride_order, (0, 1))
        self.assertEqual(derived.name, "workspace")
        self.assertEqual(derived.init_value, 0)

    def test_compact_descriptor_supports_an_explicit_dimension_order(self):
        desc = self.tensor_desc.make_compact_tensor_desc(
            dtype=_DataType.FLOAT,
            shape=(7, 3, 5),
            stride_order=(1, 2, 0),
        )

        self.assertEqual(desc.stride, (15, 1, 3))
        self.assertEqual(desc.stride_order, (1, 2, 0))

    def test_compact_descriptor_supports_scalars(self):
        desc = self.tensor_desc.make_compact_tensor_desc(
            dtype=_DataType.FLOAT,
            shape=(),
        )

        self.assertEqual(desc.shape, ())
        self.assertEqual(desc.stride, ())
        self.assertEqual(desc.stride_order, ())
        self.assertEqual(desc.ndim, 0)
        self.assertTrue(desc.is_compact())

    def test_is_compact_uses_canonical_or_explicit_dimension_order(self):
        row_major = self.tensor_desc.make_compact_tensor_desc(
            dtype=_DataType.FLOAT,
            shape=(2, 3, 4),
        )
        self.assertTrue(row_major.is_compact())
        self.assertTrue(row_major.is_compact((2, 1, 0)))
        self.assertFalse(row_major.is_compact((0, 1, 2)))

        size_one = self.tensor_desc.TensorDesc(
            dtype=_DataType.FLOAT,
            shape=(2, 1, 4),
            stride=(4, 17, 1),
            stride_order=(2, 0, 1),
        )
        self.assertTrue(size_one.is_compact((2, 1, 0)))

        with self.assertRaisesRegex(ValueError, "must be a permutation"):
            row_major.is_compact((2, 2, 0))

    def test_compact_descriptor_preserves_order_for_zero_extents(self):
        desc = self.tensor_desc.make_compact_tensor_desc(
            dtype=_DataType.FLOAT,
            shape=(2, 0, 3),
        )

        self.assertEqual(desc.stride, (3, 3, 1))
        self.assertEqual(desc.stride_order, (2, 1, 0))

    def test_compact_descriptor_rejects_noncanonical_dtype(self):
        with self.assertRaisesRegex(TypeError, "dtype must be a cudnn.data_type"):
            self.tensor_desc.make_compact_tensor_desc(dtype="float32", shape=(4,))

    def test_compact_descriptor_rejects_invalid_shapes(self):
        for shape, message in (
            ((-1, 4), "must be non-negative"),
            ((2.5, 4), "must be integers"),
            ((True, 4), "must be integers"),
        ):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    self.tensor_desc.make_compact_tensor_desc(dtype=_DataType.FLOAT, shape=shape)

    def test_compact_descriptor_rejects_invalid_dimension_orders(self):
        for stride_order, message in (
            ((1,), "rank mismatch"),
            ((0, 0), "must be a permutation"),
            ((0, True), "must be integers"),
        ):
            with self.subTest(stride_order=stride_order):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    self.tensor_desc.make_compact_tensor_desc(
                        dtype=_DataType.FLOAT,
                        shape=(2, 3),
                        stride_order=stride_order,
                    )

    def test_descriptor_rejects_stride_order_that_contradicts_strides(self):
        with self.assertRaisesRegex(ValueError, "is inconsistent with stride"):
            self.tensor_desc.TensorDesc(
                dtype=_DataType.FLOAT,
                shape=(2, 3),
                stride=(3, 1),
                stride_order=(0, 1),
            )


if __name__ == "__main__":
    unittest.main()
