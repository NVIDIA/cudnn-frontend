# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts for the dense GEMM + amax operation."""

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
_OPERATION_ROOT = _CUDNN_ROOT / "gemm_amax"
_PACKAGE = "cudnn_gemm_amax_op_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    INT8 = auto()
    UINT8 = auto()
    FP4_E2M1 = auto()
    FP8_E4M3 = auto()
    FP8_E5M2 = auto()
    FP8_E8M0 = auto()


class GemmAmaxOpContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        operation_name = f"{_PACKAGE}.gemm_amax"
        operation = types.ModuleType(operation_name)
        operation.__path__ = [str(_OPERATION_ROOT)]
        operation.__package__ = operation_name
        operation.__spec__ = ModuleSpec(operation_name, loader=None, is_package=True)
        sys.modules[operation_name] = operation

        try:
            cls.tensor_module = importlib.import_module(f"{_PACKAGE}.common.tensor_desc")
            cls.base_module = importlib.import_module(f"{_PACKAGE}.common.op")
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
        m, n, k, batch = 128, 256, 128, 2
        scale_order = (3, 1, 0, 4, 2, 5)
        arguments = {
            "a": self._desc((m, k, batch), _DataType.FP8_E4M3, (1, 0, 2), "A"),
            "b": self._desc((n, k, batch), _DataType.FP8_E4M3, (1, 0, 2), "B"),
            "sfa": self._desc((32, 4, 1, 4, 1, batch), _DataType.FP8_E8M0, scale_order, "SFA"),
            "sfb": self._desc((32, 4, 2, 4, 1, batch), _DataType.FP8_E8M0, scale_order, "SFB"),
            "c": self._desc((m, n, batch), _DataType.FLOAT, (1, 0, 2), "C"),
            "amax": self._desc((1, 1, 1), _DataType.FLOAT, (2, 1, 0), "Amax"),
        }
        arguments.update(overrides)
        return self.op_module.GemmAmaxSm100Op(**arguments)

    def test_validates_complete_signature_and_resolves_configuration(self):
        operation = self._op()

        self.assertIsInstance(operation, self.base_module.Op)
        self.assertTrue(operation.check_support())
        self.assertEqual((operation.m, operation.n, operation.k, operation.l), (128, 256, 128, 2))
        self.assertEqual((operation.a_major, operation.b_major, operation.c_major), ("k", "k", "n"))
        self.assertEqual(operation.ab_dtype, _DataType.FP8_E4M3)
        self.assertEqual(operation.scale_dtype, _DataType.FP8_E8M0)
        self.assertEqual(operation.c_dtype, _DataType.FLOAT)
        self.assertEqual(operation.mma_tiler_mn, (128, 128))
        self.assertEqual(operation.cluster_shape_mn, (1, 1))
        self.assertNotIn(f"{_PACKAGE}.gemm_amax.api", sys.modules)
        self.assertNotIn(
            f"{_PACKAGE}.gemm_amax.dense_blockscaled_gemm_persistent_amax",
            sys.modules,
        )

    def test_accepts_supported_major_modes_and_storage_aliases(self):
        scale_order = (3, 1, 0, 4, 2, 5)
        operation = self._op(
            a=self._desc((128, 128, 2), _DataType.UINT8, (1, 0, 2), "A"),
            b=self._desc((256, 128, 2), _DataType.UINT8, (1, 0, 2), "B"),
            sfa=self._desc((32, 4, 1, 4, 2, 2), _DataType.INT8, scale_order, "SFA"),
            sfb=self._desc((32, 4, 2, 4, 2, 2), _DataType.INT8, scale_order, "SFB"),
            c=self._desc((128, 256, 2), _DataType.UINT8, (1, 0, 2), "C"),
            sf_vec_size=16,
        )
        self.assertTrue(operation.check_support())
        self.assertEqual(operation.ab_dtype, _DataType.FP4_E2M1)
        self.assertEqual(operation.scale_dtype, _DataType.FP8_E8M0)
        self.assertEqual(operation.c_dtype, _DataType.FP4_E2M1)

        alternate = self._op(
            a=self._desc((128, 128, 2), _DataType.FP8_E4M3, (0, 1, 2), "A"),
            b=self._desc((256, 128, 2), _DataType.FP8_E4M3, (0, 1, 2), "B"),
            c=self._desc((128, 256, 2), _DataType.BFLOAT16, (0, 1, 2), "C"),
        )
        self.assertTrue(alternate.check_support())
        self.assertEqual((alternate.a_major, alternate.b_major, alternate.c_major), ("m", "n", "m"))

    def test_scale_layout_accepts_size_one_stride_order_ties(self):
        scale = self.tensor_module.TensorDesc(
            dtype=_DataType.FP8_E8M0,
            shape=(32, 4, 1, 4, 1, 1),
            stride=(16, 4, 512, 1, 512, 512),
            stride_order=(3, 1, 0, 2, 4, 5),
            name="scale",
        )
        operation = self._op(
            a=self._desc((128, 128, 1), _DataType.FP8_E4M3, (1, 0, 2), "A"),
            b=self._desc((128, 128, 1), _DataType.FP8_E4M3, (1, 0, 2), "B"),
            sfa=scale,
            sfb=scale,
            c=self._desc((128, 128, 1), _DataType.FLOAT, (1, 0, 2), "C"),
        )
        self.assertTrue(operation.check_support())

    def test_existing_torch_configuration_families_remain_accepted(self):
        scale_order = (3, 1, 0, 4, 2, 5)
        cases = (
            {
                "a": self._desc((128, 256, 2), _DataType.FP8_E5M2, (0, 1, 2), "A"),
                "b": self._desc((256, 256, 2), _DataType.FP8_E5M2, (0, 1, 2), "B"),
                "sfa": self._desc((32, 4, 1, 4, 2, 2), _DataType.FP8_E8M0, scale_order, "SFA"),
                "sfb": self._desc((32, 4, 2, 4, 2, 2), _DataType.FP8_E8M0, scale_order, "SFB"),
                "c": self._desc((128, 256, 2), _DataType.BFLOAT16, (0, 1, 2), "C"),
                "mma_tiler_mn": (128, 256),
                "cluster_shape_mn": (2, 2),
            },
            {
                "a": self._desc((128, 128, 2), _DataType.FP4_E2M1, (1, 0, 2), "A"),
                "b": self._desc((256, 128, 2), _DataType.FP4_E2M1, (1, 0, 2), "B"),
                "sfa": self._desc((32, 4, 1, 4, 2, 2), _DataType.FP8_E4M3, scale_order, "SFA"),
                "sfb": self._desc((32, 4, 2, 4, 2, 2), _DataType.FP8_E4M3, scale_order, "SFB"),
                "c": self._desc((128, 256, 2), _DataType.FP8_E4M3, (1, 0, 2), "C"),
                "sf_vec_size": 16,
                "cluster_shape_mn": (2, 2),
            },
            {
                "a": self._desc((128, 128, 2), _DataType.FP4_E2M1, (1, 0, 2), "A"),
                "b": self._desc((256, 128, 2), _DataType.FP4_E2M1, (1, 0, 2), "B"),
                "c": self._desc((128, 256, 2), _DataType.FLOAT, (0, 1, 2), "C"),
                "cluster_shape_mn": (4, 4),
            },
        )
        for overrides in cases:
            with self.subTest(overrides=tuple(sorted(overrides))):
                self.assertTrue(self._op(**overrides).check_support())

    def test_rejects_inconsistent_shapes_and_scale_layout(self):
        scale_order = (3, 1, 0, 4, 2, 5)
        cases = (
            (
                {"b": self._desc((256, 64, 2), _DataType.FP8_E4M3, (1, 0, 2), "B")},
                "B shape mismatch",
            ),
            (
                {"c": self._desc((128, 128, 2), _DataType.FLOAT, (1, 0, 2), "C")},
                "C must have shape",
            ),
            (
                {
                    "sfa": self._desc(
                        (32, 4, 1, 4, 2, 2),
                        _DataType.FP8_E8M0,
                        scale_order,
                        "SFA",
                    )
                },
                "SFA must have shape",
            ),
            (
                {
                    "sfa": self._desc(
                        (32, 4, 1, 4, 1, 2),
                        _DataType.FP8_E8M0,
                        (5, 4, 3, 2, 1, 0),
                        "SFA",
                    )
                },
                "packed block-scale layout",
            ),
            (
                {"amax": self._desc((1,), _DataType.FLOAT, (0,), "Amax")},
                "Amax must have shape",
            ),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._op(**overrides).check_support()

    def test_rejects_invalid_dtype_combinations(self):
        cases = (
            (
                {"b": self._desc((256, 128, 2), _DataType.FP8_E5M2, (1, 0, 2), "B")},
                ValueError,
                "A and B must have the same dtype",
            ),
            (
                {"amax": self._desc((1, 1, 1), _DataType.HALF, (2, 1, 0), "Amax")},
                ValueError,
                "Amax must have dtype float32",
            ),
            (
                {"acc_dtype": _DataType.HALF},
                ValueError,
                "Accumulator dtype must be float32",
            ),
            (
                {"c": self._desc((128, 256, 2), _DataType.FP8_E5M2, (1, 0, 2), "C")},
                NotImplementedError,
                "FP8 A and B with FP8 C",
            ),
            (
                {"sf_vec_size": 16},
                ValueError,
                "FP8 A and B do not support sf_vec_size=16",
            ),
        )
        for overrides, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    self._op(**overrides).check_support()

    def test_validates_tiler_cluster_and_alignment(self):
        with self.assertRaisesRegex(ValueError, r"mma_tiler_mn\[1\]"):
            self._op(mma_tiler_mn=(128, 64)).check_support()
        with self.assertRaisesRegex(NotImplementedError, "currently hangs"):
            self._op(mma_tiler_mn=(256, 128)).check_support()
        with self.assertRaisesRegex(ValueError, "entries must be at most 4"):
            self._op(cluster_shape_mn=(8, 1)).check_support()
        with self.assertRaisesRegex(TypeError, "mma_tiler_mn must contain two integers"):
            self._op(mma_tiler_mn=None).check_support()

        scale_order = (3, 1, 0, 4, 2, 5)
        with self.assertRaisesRegex(ValueError, "16-byte alignment"):
            self._op(
                a=self._desc((128, 120, 2), _DataType.FP8_E4M3, (1, 0, 2), "A"),
                b=self._desc((256, 120, 2), _DataType.FP8_E4M3, (1, 0, 2), "B"),
                sfa=self._desc((32, 4, 1, 4, 1, 2), _DataType.FP8_E8M0, scale_order, "SFA"),
                sfb=self._desc((32, 4, 2, 4, 1, 2), _DataType.FP8_E8M0, scale_order, "SFB"),
            ).check_support()

    def test_constructor_requires_descriptors_and_canonical_accumulator(self):
        for name in ("a", "b", "sfa", "sfb", "c", "amax"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(TypeError, f"{name} must be a TensorDesc"):
                    self._op(**{name: types.SimpleNamespace(shape=(1,))})
        with self.assertRaisesRegex(TypeError, "acc_dtype must be a cudnn.data_type"):
            self._op(acc_dtype="float32")


if __name__ == "__main__":
    unittest.main()
