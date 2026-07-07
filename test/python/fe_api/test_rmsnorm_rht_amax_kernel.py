# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the RMSNorm operation kernel."""

from enum import Enum, auto
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


_CUDNN_ROOT = Path(__file__).resolve().parents[3] / "python" / "cudnn"
_OPERATION_ROOT = _CUDNN_ROOT / "rmsnorm_rht_amax"
_PACKAGE = "cudnn_rmsnorm_kernel_test"


class _DataType(Enum):
    NOT_SET = auto()
    FLOAT = auto()
    BFLOAT16 = auto()


def _identity_decorator(function=None, **_kwargs):
    if function is None:
        return lambda decorated: decorated
    return function


def _module(name: str, *, package: bool = False, **attributes):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _cutlass_stubs() -> dict[str, types.ModuleType]:
    driver = _module("cuda.bindings.driver", CUstream=object)
    bindings = _module("cuda.bindings", package=True, driver=driver)
    cuda = _module("cuda", package=True, bindings=bindings)

    arch = _module("cutlass.cute.arch", shuffle_sync_bfly=lambda value, **_kwargs: value)
    cute = _module(
        "cutlass.cute",
        package=True,
        arch=arch,
        kernel=_identity_decorator,
        jit=_identity_decorator,
    )
    utils = _module("cutlass.utils")
    llvm = _module(
        "cutlass._mlir.dialects.llvm",
        AsmDialect=types.SimpleNamespace(AD_ATT=object()),
    )
    dialects = _module("cutlass._mlir.dialects", package=True, llvm=llvm)
    mlir = _module("cutlass._mlir", package=True, dialects=dialects)
    cutlass_dsl = _module(
        "cutlass.cutlass_dsl",
        T=types.SimpleNamespace(f32=lambda: None),
        dsl_user_op=_identity_decorator,
    )
    cutlass = _module(
        "cutlass",
        package=True,
        cute=cute,
        utils=utils,
        Float32=object,
        Int32=object,
        BFloat16=object,
    )

    return {
        module.__name__: module
        for module in (
            cuda,
            bindings,
            driver,
            cutlass,
            cute,
            arch,
            utils,
            mlir,
            dialects,
            llvm,
            cutlass_dsl,
        )
    }


def _install_test_package() -> None:
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


def _remove_test_package() -> None:
    for name in tuple(sys.modules):
        if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
            sys.modules.pop(name, None)


class RmsNormRhtAmaxKernelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_test_package()
        try:
            with mock.patch.dict(sys.modules, _cutlass_stubs()):
                cls.tensor_module = importlib.import_module(f"{_PACKAGE}._tensor_desc")
                cls.base_module = importlib.import_module(f"{_PACKAGE}._op_kernel")
                cls.kernel_module = importlib.import_module(f"{_PACKAGE}.rmsnorm_rht_amax.kernel")
        except Exception:
            _remove_test_package()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        _remove_test_package()

    def _desc(self, shape, dtype=_DataType.BFLOAT16, *, name=""):
        shape = tuple(shape)
        stride_order = tuple(reversed(range(len(shape))))
        stride = [0] * len(shape)
        running = 1
        for dimension in stride_order:
            stride[dimension] = running
            running *= shape[dimension]
        return self.tensor_module.TensorDesc(
            dtype=dtype,
            shape=shape,
            stride=tuple(stride),
            stride_order=stride_order,
            name=name,
        )

    def _kernel(self, **overrides):
        arguments = {
            "x": self._desc((256, 2048)),
            "weight": self._desc((2048,)),
        }
        arguments.update(overrides)
        return self.kernel_module.RMSNormRHTAmaxKernel(**arguments)

    def test_kernel_implements_neutral_contract_and_infers_buffers(self):
        kernel = self._kernel()
        self.assertIsInstance(kernel, self.base_module.OpKernel)
        with self.assertRaisesRegex(RuntimeError, r"check_support\(\)"):
            kernel.infer_output()

        self.assertTrue(kernel.check_support())
        self.assertEqual((kernel.m, kernel.n), (256, 2048))
        self.assertEqual((kernel.num_threads, kernel.rows_per_cta), (128, 2))

        output, amax = kernel.infer_output()
        self.assertIs(type(output), self.tensor_module.TensorDesc)
        self.assertEqual(
            (output.name, output.dtype, output.shape, output.stride, output.stride_order, output.init_value),
            ("output", _DataType.BFLOAT16, (256, 2048), (2048, 1), (1, 0), None),
        )
        self.assertEqual(
            (amax.name, amax.dtype, amax.shape, amax.stride, amax.stride_order, amax.init_value),
            ("amax", _DataType.FLOAT, (128,), (1,), (0,), None),
        )
        self.assertEqual(kernel.infer_workspace(), ())

        self.assertNotIn(f"{_PACKAGE}.rmsnorm_rht_amax.api", sys.modules)
        self.assertNotIn(f"{_PACKAGE}.rmsnorm_rht_amax.jax", sys.modules)

    def test_requested_configuration_is_recomputed(self):
        kernel = self._kernel(num_threads=256, rows_per_cta=4)
        kernel.check_support()
        self.assertEqual((kernel.num_threads, kernel.rows_per_cta), (256, 4))

        kernel.requested_num_threads = 128
        kernel.requested_rows_per_cta = 8
        kernel.amax = self._desc((32,), _DataType.FLOAT)
        kernel.check_support()
        self.assertEqual((kernel.num_threads, kernel.rows_per_cta), (128, 8))
        self.assertEqual(kernel.infer_output()[1].shape, (32,))

    def test_framework_descriptor_extension_maps_native_dtype(self):
        class FrameworkTensorDesc(self.tensor_module.TensorDesc):
            @property
            def cudnn_dtype(self):
                return {
                    "framework.bfloat16": _DataType.BFLOAT16,
                    "framework.float32": _DataType.FLOAT,
                }.get(self.dtype, _DataType.NOT_SET)

        def desc(shape, dtype):
            base = self._desc(shape)
            return FrameworkTensorDesc(
                dtype=dtype,
                shape=base.shape,
                stride=base.stride,
                stride_order=base.stride_order,
                name="framework_tensor",
            )

        kernel = self.kernel_module.RMSNormRHTAmaxKernel(
            x=desc((256, 2048), "framework.bfloat16"),
            weight=desc((2048,), "framework.bfloat16"),
            output=desc((256, 2048), "framework.bfloat16"),
            amax=desc((128,), "framework.float32"),
        )
        self.assertTrue(kernel.check_support())

    def test_invalid_logical_signatures_and_configuration(self):
        cases = (
            ({"x": self._desc((2048,))}, "X must have rank 2"),
            ({"weight": self._desc((1, 2048))}, "W must have rank 1"),
            ({"x": self._desc((256, 2048), _DataType.FLOAT)}, "X must have dtype bfloat16"),
            ({"weight": self._desc((2048,), _DataType.FLOAT)}, "W must have dtype bfloat16"),
            ({"x": self._desc((0, 2048))}, "M must be positive"),
            (
                {"x": self._desc((256, 2049)), "weight": self._desc((2049,))},
                "N must be divisible by 16",
            ),
            ({"weight": self._desc((1024,))}, "W must have shape"),
            (
                {
                    "x": self.tensor_module.TensorDesc(
                        dtype=_DataType.BFLOAT16,
                        shape=(256, 2048),
                        stride=(1, 256),
                        stride_order=(0, 1),
                    )
                },
                "X must be row-major contiguous",
            ),
            ({"num_threads": 512}, "EPT=4 must be >= 8 and divisible by 8"),
            ({"num_threads": 2048}, "must not exceed the CUDA block size limit"),
            ({"rows_per_cta": 0}, "rows_per_cta must be positive"),
            ({"rows_per_cta": 3}, "M must be divisible by rows_per_cta"),
            ({"output": self._desc((256, 1024))}, "O must have shape"),
            ({"output": self._desc((256, 2048), _DataType.FLOAT)}, "O must have dtype bfloat16"),
            ({"amax": self._desc((64,), _DataType.FLOAT)}, "Amax must have shape"),
            ({"amax": self._desc((128,))}, "Amax must have dtype float32"),
        )

        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._kernel(**overrides).check_support()

        with self.assertRaisesRegex(TypeError, "x must be a TensorDesc"):
            self.kernel_module.RMSNormRHTAmaxKernel(
                x=types.SimpleNamespace(shape=(256, 2048)),
                weight=self._desc((2048,)),
            )
        with self.assertRaisesRegex(TypeError, "weight must be a TensorDesc"):
            self.kernel_module.RMSNormRHTAmaxKernel(
                x=self._desc((256, 2048)),
                weight=None,
            )


if __name__ == "__main__":
    unittest.main()
