# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for shared GEMM + amax validation."""

from __future__ import annotations

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


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"
_PACKAGE = "cudnn_frontend_gemm_amax_validation_test"


def _canonical_dtype_name(dtype):
    if not isinstance(dtype, type) and hasattr(dtype, "dtype_name"):
        return dtype.dtype_name
    name = getattr(dtype, "name", str(dtype)).rsplit(".", 1)[-1].lower()
    return {"float4_e2m1fn_x2": "float4_e2m1fn"}.get(name, name)


def _load_validation_modules():
    root = types.ModuleType(_PACKAGE)
    root.__path__ = [str(_CUDNN_ROOT)]
    root.__package__ = _PACKAGE
    root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
    sys.modules[_PACKAGE] = root

    fake_api_base = types.ModuleType(f"{_PACKAGE}.api_base")
    fake_api_base.TensorDesc = object
    fake_api_base.canonical_dtype_name = _canonical_dtype_name
    sys.modules[fake_api_base.__name__] = fake_api_base

    operation = types.ModuleType(f"{_PACKAGE}.gemm_amax")
    operation.__path__ = [str(_CUDNN_ROOT / "gemm_amax")]
    operation.__package__ = operation.__name__
    operation.__spec__ = ModuleSpec(operation.__name__, loader=None, is_package=True)
    sys.modules[operation.__name__] = operation

    gemm_validation = importlib.import_module(f"{_PACKAGE}.gemm_validation")
    operation_validation = importlib.import_module(f"{_PACKAGE}.gemm_amax.validation")
    return gemm_validation, operation_validation


_GEMM_VALIDATION, _VALIDATION = _load_validation_modules()


def _compact_stride(shape, order):
    stride = [None] * len(shape)
    running = 1
    for dim in order:
        stride[dim] = running
        running *= shape[dim]
    return tuple(stride)


def _desc(shape, dtype_name, *, order=None, packed_fp4=False, name=""):
    shape = tuple(shape)
    bits = {
        "int8": 8,
        "uint8": 8,
        "float4_e2m1fn": 4,
        "float8_e4m3fn": 8,
        "float8_e5m2": 8,
        "float8_e8m0fnu": 8,
        "bfloat16": 16,
        "float16": 16,
        "float32": 32,
    }[dtype_name]
    return types.SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype_name=dtype_name,
        storage_dtype_name=dtype_name,
        element_bits=bits,
        stride=None if order is None else _compact_stride(shape, order),
        stride_order=order,
        packing="fp4x2" if packed_fp4 else "native",
        interpret_uint8_as_fp4x2=packed_fp4,
        name=name,
    )


def _valid_descriptors(*, ab_dtype="float8_e4m3fn", c_dtype="float32", k=128, sf_vec_size=32):
    m = n = 128
    batch = 1
    scale_a = _GEMM_VALIDATION.block_scale_shape(m, k, batch, sf_vec_size)
    scale_b = _GEMM_VALIDATION.block_scale_shape(n, k, batch, sf_vec_size)
    return {
        "a": _desc((m, k, batch), ab_dtype, order=(1, 0, 2), name="A"),
        "b": _desc((n, k, batch), ab_dtype, order=(1, 0, 2), name="B"),
        "sfa": _desc(scale_a, "float8_e8m0fnu", name="sfa"),
        "sfb": _desc(scale_b, "float8_e8m0fnu", name="sfb"),
        "c": _desc((m, n, batch), c_dtype, order=(1, 0, 2), name="C"),
        "amax": _desc((1, 1, 1), "float32", name="amax"),
    }


def _validate(descs, **kwargs):
    return _VALIDATION.validate_gemm_amax(
        descs["a"],
        descs["b"],
        descs["sfa"],
        descs["sfb"],
        descs["c"],
        descs["amax"],
        acc_dtype=kwargs.pop("acc_dtype", "float32"),
        sf_vec_size=kwargs.pop("sf_vec_size", 32),
        supported_sf_vec_sizes=(16, 32),
        mma_tiler_mn=kwargs.pop("mma_tiler_mn", (128, 128)),
        **kwargs,
    )


class GemmAmaxValidationTest(unittest.TestCase):
    def test_returns_shared_logical_plan(self):
        plan = _validate(_valid_descriptors())

        self.assertEqual((plan.m, plan.n, plan.k, plan.batch), (128, 128, 128, 1))
        self.assertEqual(plan.c_shape, (128, 128, 1))
        self.assertEqual(plan.amax_shape, (1, 1, 1))
        self.assertEqual((plan.a_major, plan.b_major, plan.c_major), ("k", "k", "n"))
        self.assertEqual(plan.ab_dtype_name, "float8_e4m3fn")

    def test_accepts_torch_packed_fp4_as_logical_fp4(self):
        descs = _valid_descriptors(ab_dtype="uint8")
        descs["a"] = _desc((128, 128, 1), "uint8", order=(1, 0, 2), packed_fp4=True, name="A")
        descs["b"] = _desc((128, 128, 1), "uint8", order=(1, 0, 2), packed_fp4=True, name="B")

        plan = _validate(descs)

        self.assertEqual(plan.ab_dtype_name, "float4_e2m1fn")

    def test_rejects_invalid_shapes_dtypes_and_layouts(self):
        cases = []

        descs = _valid_descriptors()
        descs["b"] = _desc((128, 128, 1), "float8_e5m2", order=(1, 0, 2))
        cases.append((descs, {}, "a_tensor and b_tensor dtypes must match"))

        descs = _valid_descriptors()
        descs["sfa"] = _desc((32, 4, 1, 4, 2, 1), "float8_e8m0fnu")
        cases.append((descs, {}, "sfa_tensor must have shape"))

        descs = _valid_descriptors()
        descs["c"] = _desc((128, 64, 1), "float32", order=(1, 0, 2))
        cases.append((descs, {}, "C must have shape"))

        descs = _valid_descriptors()
        descs["a"].stride = (256, 2, 32768)
        cases.append((descs, {}, "A tensor must use one of the compact strides"))

        for descs, kwargs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _validate(descs, **kwargs)

    def test_rejects_kernel_domain_combinations(self):
        descs = _valid_descriptors(c_dtype="float8_e5m2")
        with self.assertRaisesRegex(NotImplementedError, "FP8 A/B with FP8 C"):
            _validate(descs)

        descs = _valid_descriptors(sf_vec_size=16)
        with self.assertRaisesRegex(ValueError, "do not support sf_vec_size=16"):
            _validate(descs, sf_vec_size=16)

        descs = _valid_descriptors(ab_dtype="float4_e2m1fn", k=128)
        with self.assertRaisesRegex(ValueError, "requires K > 128"):
            _validate(descs, mma_tiler_mn=(128, 256))

        descs = _valid_descriptors(ab_dtype="float4_e2m1fn", k=128, sf_vec_size=16)
        descs["sfa"] = _desc(descs["sfa"].shape, "int8")
        descs["sfb"] = _desc(descs["sfb"].shape, "int8")
        with self.assertRaisesRegex(ValueError, "sfa_tensor and sfb_tensor dtype must be one of"):
            _validate(descs, sf_vec_size=16)

    def test_rejects_unaligned_contiguous_extent(self):
        descs = _valid_descriptors(k=120)
        with self.assertRaisesRegex(ValueError, "16-byte aligned"):
            _validate(descs)


if __name__ == "__main__":
    unittest.main()
