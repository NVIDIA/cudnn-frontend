# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts for framework-neutral dense GEMM helpers."""

import ast
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
_PACKAGE = "cudnn_dense_gemm_test"


class _DataType(Enum):
    NOT_SET = auto()
    FLOAT = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FP8_E4M3 = auto()
    FP8_E5M2 = auto()
    FP8_E8M0 = auto()
    FP4_E2M1 = auto()


class DenseGemmHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_CUDNN_ROOT)]
        package.__package__ = _PACKAGE
        package.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        package.data_type = _DataType
        sys.modules[_PACKAGE] = package

        cls.tensor_module = importlib.import_module(f"{_PACKAGE}.common.tensor_desc")
        cls.helpers = importlib.import_module(f"{_PACKAGE}.gemm.helpers")

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def desc(self, shape, dtype=_DataType.BFLOAT16, *, stride_order=None, name=""):
        return self.tensor_module.make_compact_tensor_desc(
            dtype=dtype,
            shape=tuple(shape),
            stride_order=stride_order,
            name=name,
        )

    def test_gemm_inputs_resolve_mnkl(self):
        a = self.desc((17, 31, 3), name="A")
        b = self.desc((19, 31, 3), name="B")

        self.assertEqual(self.helpers.require_gemm_inputs(a, b), (17, 19, 31, 3))

    def test_gemm_inputs_reject_invalid_rank_extents_and_matching(self):
        cases = (
            (self.desc((17, 31)), self.desc((19, 31, 3)), "A must have rank 3"),
            (self.desc((17, 31, 3)), self.desc((19, 31)), "B must have rank 3"),
            (self.desc((0, 31, 3)), self.desc((19, 31, 3)), "M must be positive"),
            (self.desc((17, 31, 3)), self.desc((0, 31, 3)), "N must be positive"),
            (self.desc((17, 0, 3)), self.desc((19, 0, 3)), "K must be positive"),
            (self.desc((17, 31, 0)), self.desc((19, 31, 0)), "L must be positive"),
            (self.desc((17, 31, 3)), self.desc((19, 29, 3)), "B shape mismatch"),
            (self.desc((17, 31, 3)), self.desc((19, 31, 4)), "B shape mismatch"),
        )

        for a, b, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.helpers.require_gemm_inputs(a, b)

    def test_exact_output_shape_uses_logical_descriptor_shape(self):
        output = self.desc((17, 19, 3), name="AB12")
        self.helpers.require_tensor_shape(output, (17, 19, 3))

        with self.assertRaisesRegex(ValueError, "AB12 must have shape"):
            self.helpers.require_tensor_shape(output, (17, 18, 3))

    def test_compact_major_recognizes_mode_zero_and_mode_one(self):
        m_major = self.desc((8, 16, 2), stride_order=(0, 1, 2), name="A")
        k_major = self.desc((8, 16, 2), stride_order=(1, 0, 2), name="A")

        self.assertEqual(self.helpers.require_compact_major(m_major, "m", "k"), "m")
        self.assertEqual(self.helpers.require_compact_major(k_major, "m", "k"), "k")

    def test_compact_major_rejects_batch_major_and_noncompact_layouts(self):
        batch_major = self.desc((8, 16, 2), stride_order=(2, 1, 0), name="A")
        noncompact = self.tensor_module.TensorDesc(
            dtype=_DataType.BFLOAT16,
            shape=(8, 16, 2),
            stride=(1, 16, 256),
            stride_order=(0, 1, 2),
            name="A",
        )

        for tensor in (batch_major, noncompact):
            with self.subTest(stride=tensor.stride):
                with self.assertRaisesRegex(ValueError, "compact m-major or k-major"):
                    self.helpers.require_compact_major(tensor, "m", "k")

    def test_data_type_widths_and_16_byte_alignment(self):
        cases = (
            (_DataType.FLOAT, 32, 4),
            (_DataType.HALF, 16, 8),
            (_DataType.BFLOAT16, 16, 8),
            (_DataType.FP8_E4M3, 8, 16),
            (_DataType.FP8_E5M2, 8, 16),
            (_DataType.FP4_E2M1, 4, 32),
        )

        for dtype, bits, extent in cases:
            with self.subTest(dtype=dtype):
                tensor = self.desc((extent, 7, 2), dtype, stride_order=(0, 1, 2), name="A")
                self.assertEqual(self.helpers.data_type_bits(dtype), bits)
                self.helpers.require_16_byte_alignment(tensor)

                misaligned = self.desc((extent - 1, 7, 2), dtype, stride_order=(0, 1, 2), name="A")
                with self.assertRaisesRegex(ValueError, "16-byte alignment"):
                    self.helpers.require_16_byte_alignment(misaligned)

        with self.assertRaisesRegex(ValueError, "Unsupported cuDNN data type"):
            self.helpers.data_type_bits(_DataType.NOT_SET)

    def test_16_byte_alignment_uses_the_selected_major_extent(self):
        tensor = self.desc((7, 8, 2), stride_order=(1, 0, 2), name="A")
        self.helpers.require_16_byte_alignment(tensor)

    def test_mma_tiler_validation_is_configurable(self):
        self.assertEqual(self.helpers.require_mma_tiler((128, 32)), (128, 32))
        self.assertEqual(self.helpers.require_mma_tiler((256, 256)), (256, 256))
        self.assertEqual(
            self.helpers.require_mma_tiler((64, 96), allowed_m=(64,), allowed_n=(96,)),
            (64, 96),
        )

        cases = (
            ((64, 128), "mma_tiler_mn\\[0\\]"),
            ((128, 48), "mma_tiler_mn\\[1\\]"),
            ((128,), "must contain two integers"),
        )
        for tiler, message in cases:
            with self.subTest(tiler=tiler):
                with self.assertRaisesRegex(ValueError, message):
                    self.helpers.require_mma_tiler(tiler)

        with self.assertRaisesRegex(TypeError, "must contain integers"):
            self.helpers.require_mma_tiler((True, 128))

    def test_cluster_shape_validation_handles_cta_groups(self):
        self.assertEqual(self.helpers.require_cluster_shape((1, 1)), (1, 1))
        self.assertEqual(
            self.helpers.require_cluster_shape((2, 4), cta_group_size=2),
            (2, 4),
        )

        cases = (
            ((3, 1), {}, "positive powers of two"),
            ((4, 8), {}, "must not exceed 16"),
            ((1, 2), {"cta_group_size": 2}, "divisible by CTA group size"),
        )
        for shape, kwargs, message in cases:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, message):
                    self.helpers.require_cluster_shape(shape, **kwargs)

    def test_module_has_no_framework_or_kernel_dependencies(self):
        tree = ast.parse((_CUDNN_ROOT / "gemm" / "helpers.py").read_text())
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append("." * node.level + (node.module or ""))

        self.assertFalse(
            any(imported == dependency or imported.startswith(f"{dependency}.") for imported in imports for dependency in ("torch", "jax", "cutlass", "cuda")),
            imports,
        )


if __name__ == "__main__":
    unittest.main()
