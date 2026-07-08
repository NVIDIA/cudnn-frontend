# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts for the RMSNorm logical operation."""

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
_OPERATION_ROOT = _CUDNN_ROOT / "rmsnorm_rht_amax"
_PACKAGE = "cudnn_rmsnorm_op_test"


class _DataType(Enum):
    NOT_SET = auto()
    FLOAT = auto()
    BFLOAT16 = auto()


class RmsNormRhtAmaxOpContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        operation_name = f"{_PACKAGE}.rmsnorm_rht_amax"
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

    def _desc(self, shape, dtype=_DataType.BFLOAT16, *, stride=None, name=""):
        shape = tuple(shape)
        if stride is None:
            stride_order = tuple(reversed(range(len(shape))))
            stride_values = [0] * len(shape)
            running = 1
            for dimension in stride_order:
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

    def _op(self, **overrides):
        arguments = {
            "x": self._desc((256, 2048), name="x"),
            "weight": self._desc((2048,), name="weight"),
            "output": self._desc((256, 2048), name="output"),
            "amax": self._desc((128,), _DataType.FLOAT, name="amax"),
        }
        arguments.update(overrides)
        return self.op_module.RmsNormRhtAmaxSm100Op(**arguments)

    def test_operation_validates_complete_signature_and_resolves_configuration(self):
        operation = self._op()

        self.assertIsInstance(operation, self.base_module.Op)
        self.assertTrue(operation.check_support())
        self.assertEqual((operation.m, operation.n), (256, 2048))
        self.assertEqual((operation.num_threads, operation.rows_per_cta), (128, 2))
        self.assertFalse(hasattr(operation, "infer_output"))
        self.assertFalse(hasattr(operation, "check_output"))
        self.assertNotIn(f"{_PACKAGE}.rmsnorm_rht_amax.kernel", sys.modules)
        self.assertNotIn(f"{_PACKAGE}.rmsnorm_rht_amax.api", sys.modules)
        self.assertNotIn(f"{_PACKAGE}.rmsnorm_rht_amax.jax", sys.modules)

    def test_positive_strided_amax_is_supported_but_broadcast_amax_is_not(self):
        self.assertTrue(self._op(amax=self._desc((128,), _DataType.FLOAT, stride=(2,))).check_support())

        with self.assertRaisesRegex(ValueError, "Amax stride must be positive"):
            self._op(amax=self._desc((128,), _DataType.FLOAT, stride=(0,))).check_support()

    def test_framework_descriptors_are_compared_through_cudnn_dtype(self):
        tensor_module = self.tensor_module

        class FrameworkTensorDesc(tensor_module.TensorDesc):
            @property
            def cudnn_dtype(self):
                return {
                    "framework.bfloat16": _DataType.BFLOAT16,
                    "framework.float32": _DataType.FLOAT,
                }.get(self.dtype, _DataType.NOT_SET)

        def desc(shape, dtype, *, stride=None):
            canonical = self._desc(shape, stride=stride)
            return FrameworkTensorDesc(
                dtype=dtype,
                shape=canonical.shape,
                stride=canonical.stride,
                stride_order=canonical.stride_order,
            )

        operation = self.op_module.RmsNormRhtAmaxSm100Op(
            x=desc((256, 2048), "framework.bfloat16"),
            weight=desc((2048,), "framework.bfloat16"),
            output=desc((256, 2048), "framework.bfloat16"),
            amax=desc((128,), "framework.float32"),
        )
        self.assertTrue(operation.check_support())

    def test_invalid_complete_signatures_and_configuration(self):
        cases = (
            ({"x": self._desc((2048,))}, "X must have rank 2"),
            ({"weight": self._desc((1, 2048))}, "W must have rank 1"),
            ({"output": self._desc((256 * 2048,))}, "O must have rank 2"),
            ({"amax": self._desc((128, 1), _DataType.FLOAT)}, "Amax must have rank 1"),
            ({"x": self._desc((256, 2048), _DataType.FLOAT)}, "X must have dtype bfloat16"),
            ({"weight": self._desc((2048,), _DataType.FLOAT)}, "W must have dtype bfloat16"),
            ({"output": self._desc((256, 2048), _DataType.FLOAT)}, "O must have dtype bfloat16"),
            ({"amax": self._desc((128,), _DataType.BFLOAT16)}, "Amax must have dtype float32"),
            ({"x": self._desc((0, 2048))}, "M must be positive"),
            (
                {
                    "x": self._desc((256, 2049)),
                    "weight": self._desc((2049,)),
                    "output": self._desc((256, 2049)),
                },
                "N must be divisible by 16",
            ),
            ({"weight": self._desc((1024,))}, "W must have shape"),
            ({"output": self._desc((128, 2048))}, "O must have shape"),
            ({"amax": self._desc((64,), _DataType.FLOAT)}, "Amax must have shape"),
            ({"x": self._desc((256, 2048), stride=(1, 256))}, "X must be row-major contiguous"),
            ({"weight": self._desc((2048,), stride=(2,))}, "W must be contiguous"),
            ({"output": self._desc((256, 2048), stride=(1, 256))}, "O must be row-major contiguous"),
            ({"num_threads": 512}, "EPT=4 must be >= 8 and divisible by 8"),
            ({"num_threads": 2048}, "must not exceed the CUDA block size limit"),
            ({"rows_per_cta": 0}, "rows_per_cta must be positive"),
            ({"rows_per_cta": 3}, "M must be divisible by rows_per_cta"),
        )

        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._op(**overrides).check_support()

    def test_constructor_requires_tensor_descriptors(self):
        for name in ("x", "weight", "output", "amax"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(TypeError, f"{name} must be a TensorDesc"):
                    self._op(**{name: types.SimpleNamespace(shape=(1,))})


if __name__ == "__main__":
    unittest.main()
