# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for shared GEMM + SwiGLU validation."""

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
_PACKAGE = "cudnn_frontend_gemm_swiglu_validation_test"


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

    operation = types.ModuleType(f"{_PACKAGE}.gemm_swiglu")
    operation.__path__ = [str(_CUDNN_ROOT / "gemm_swiglu")]
    operation.__package__ = operation.__name__
    operation.__spec__ = ModuleSpec(operation.__name__, loader=None, is_package=True)
    sys.modules[operation.__name__] = operation

    gemm_validation = importlib.import_module(f"{_PACKAGE}.gemm_validation")
    operation_validation = importlib.import_module(f"{_PACKAGE}.gemm_swiglu.validation")
    return gemm_validation, operation_validation


_GEMM_VALIDATION, _VALIDATION = _load_validation_modules()


def _compact_stride(shape, order):
    stride = [None] * len(shape)
    running = 1
    for dim in order:
        stride[dim] = running
        running *= shape[dim]
    return tuple(stride)


def _desc(shape, dtype_name, *, order=None, packing="native", name=""):
    shape = tuple(shape)
    bits = {
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
        packing=packing,
        name=name,
    )


def _valid_descriptors(
    *,
    ab_dtype="float8_e4m3fn",
    scale_dtype="float8_e8m0fnu",
    ab12_dtype="bfloat16",
    c_dtype="bfloat16",
    k=128,
    sf_vec_size=32,
    packing="native",
    a_order=(1, 0, 2),
    b_order=(1, 0, 2),
    output_order=(1, 0, 2),
    include_amax=False,
):
    m = n = 128
    batch = 1
    output_n = n // 2
    return {
        "a": _desc(
            (m, k, batch),
            ab_dtype,
            order=a_order,
            packing=packing,
            name="A",
        ),
        "b": _desc(
            (n, k, batch),
            ab_dtype,
            order=b_order,
            packing=packing,
            name="B",
        ),
        "ab12": _desc(
            (m, n, batch),
            ab12_dtype,
            order=output_order,
            name="AB12",
        ),
        "c": _desc(
            (m, output_n, batch),
            c_dtype,
            order=output_order,
            name="C",
        ),
        "sfa": _desc(
            _GEMM_VALIDATION.block_scale_shape(m, k, batch, sf_vec_size),
            scale_dtype,
            name="SFA",
        ),
        "sfb": _desc(
            _GEMM_VALIDATION.block_scale_shape(n, k, batch, sf_vec_size),
            scale_dtype,
            name="SFB",
        ),
        "amax": (_desc((1, 1, 1), "float32", name="amax") if include_amax else None),
        "sfc": None,
        "norm_const": None,
    }


def _validate(descs, **kwargs):
    return _VALIDATION.validate_quantized_gemm_swiglu(
        descs["a"],
        descs["b"],
        descs["ab12"],
        descs["c"],
        sfa=descs["sfa"],
        sfb=descs["sfb"],
        amax=descs["amax"],
        sfc=descs["sfc"],
        norm_const=descs["norm_const"],
        acc_dtype=kwargs.pop("acc_dtype", "float32"),
        output_n=64,
        sf_vec_size=kwargs.pop("sf_vec_size", 32),
        supported_sf_vec_sizes=(16, 32),
        mma_tiler_mn=kwargs.pop("mma_tiler_mn", (128, 128)),
        **kwargs,
    )


class GemmSwigluValidationTest(unittest.TestCase):
    def test_returns_mxfp8_plan(self):
        plan = _validate(_valid_descriptors())

        self.assertEqual((plan.m, plan.n, plan.k, plan.batch), (128, 128, 128, 1))
        self.assertEqual(plan.ab12_shape, (128, 128, 1))
        self.assertEqual(plan.c_shape, (128, 64, 1))
        self.assertEqual((plan.a_major, plan.b_major, plan.output_major), ("k", "k", "n"))
        self.assertEqual(plan.ab_dtype_name, "float8_e4m3fn")
        self.assertFalse(plan.generate_amax)
        self.assertFalse(plan.generate_sfc)

    def test_accepts_native_and_torch_packed_fp4(self):
        cases = (
            ("float4_e2m1fn", "native"),
            ("uint8", "fp4x2"),
        )
        for dtype_name, packing in cases:
            with self.subTest(dtype_name=dtype_name, packing=packing):
                descs = _valid_descriptors(
                    ab_dtype=dtype_name,
                    scale_dtype="float8_e4m3fn",
                    sf_vec_size=16,
                    packing=packing,
                    include_amax=True,
                )
                plan = _validate(descs, sf_vec_size=16)

                self.assertEqual(plan.ab_dtype_name, "float4_e2m1fn")
                self.assertTrue(plan.requires_amax)
                self.assertTrue(plan.generate_amax)

    def test_requires_a_complete_scale_pair(self):
        for missing in ("sfa", "sfb"):
            with self.subTest(missing=missing):
                descs = _valid_descriptors()
                descs[missing] = None
                with self.assertRaisesRegex(
                    ValueError,
                    "sfa_tensor and sfb_tensor are required",
                ):
                    _validate(descs)

    def test_rejects_invalid_scale_formats(self):
        cases = (
            (
                _valid_descriptors(sf_vec_size=16),
                16,
                "FP8 A and B require float8_e8m0fnu scale factors and sf_vec_size=32",
            ),
            (
                _valid_descriptors(scale_dtype="float8_e4m3fn"),
                32,
                "FP8 A and B require float8_e8m0fnu scale factors and sf_vec_size=32",
            ),
            (
                _valid_descriptors(
                    ab_dtype="float4_e2m1fn",
                    scale_dtype="float8_e4m3fn",
                    sf_vec_size=32,
                    include_amax=True,
                ),
                32,
                "FP4 A and B do not support float8_e4m3fn scale factors with sf_vec_size=32",
            ),
        )
        for descs, sf_vec_size, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _validate(descs, sf_vec_size=sf_vec_size)

    def test_rejects_raw_uint8_and_non_k_major_fp4(self):
        raw_uint8 = _valid_descriptors(ab_dtype="uint8")
        with self.assertRaisesRegex(
            ValueError,
            "a_tensor and b_tensor dtype must be one of",
        ):
            _validate(raw_uint8)

        wrong_layout = _valid_descriptors(
            ab_dtype="float4_e2m1fn",
            sf_vec_size=16,
            a_order=(0, 1, 2),
            include_amax=True,
        )
        with self.assertRaisesRegex(ValueError, "FP4 requires k-major A and B"):
            _validate(wrong_layout, sf_vec_size=16)

    def test_requires_fp4_amax_and_uses_four_bit_alignment(self):
        missing_amax = _valid_descriptors(
            ab_dtype="float4_e2m1fn",
            sf_vec_size=16,
        )
        with self.assertRaisesRegex(ValueError, "amax_tensor is required"):
            _validate(missing_amax, sf_vec_size=16)

        unaligned = _valid_descriptors(
            ab_dtype="float4_e2m1fn",
            k=16,
            sf_vec_size=16,
            include_amax=True,
        )
        with self.assertRaisesRegex(ValueError, "A's contiguous extent must be 16-byte aligned"):
            _validate(unaligned, sf_vec_size=16)


if __name__ == "__main__":
    unittest.main()
