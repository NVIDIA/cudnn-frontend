# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for dense JAX GEMM axis bindings."""

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
_PACKAGE = "cudnn_jax_gemm_layout_test"


class JaxGemmLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        sys.modules[_PACKAGE] = root

        internal_name = f"{_PACKAGE}._jax"
        internal = types.ModuleType(internal_name)
        internal.__path__ = [str(_CUDNN_ROOT / "_jax")]
        internal.__package__ = internal_name
        internal.__spec__ = ModuleSpec(internal_name, loader=None, is_package=True)
        sys.modules[internal_name] = internal

        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "jax": None,
                    "jax.numpy": None,
                    "cutlass": None,
                    "cutlass.jax": None,
                    "torch": None,
                },
            ):
                cls.module = importlib.import_module(f"{internal_name}.gemm")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def test_maps_supported_public_layouts_to_canonical_modes(self):
        cases = (
            (self.module.gemm_a_mode, "LMK", (1, 2, 0)),
            (self.module.gemm_a_mode, "LKM", (2, 1, 0)),
            (self.module.gemm_b_mode, "LNK", (1, 2, 0)),
            (self.module.gemm_b_mode, "LKN", (2, 1, 0)),
            (self.module.gemm_output_mode, "LMN", (1, 2, 0)),
            (self.module.gemm_output_mode, "LNM", (2, 1, 0)),
        )

        for helper, layout, expected in cases:
            with self.subTest(layout=layout):
                self.assertEqual(helper(layout), expected)

    def test_rejects_unsupported_axis_orders(self):
        with self.assertRaisesRegex(ValueError, r"a_layout must be one of \('LMK', 'LKM'\)"):
            self.module.gemm_a_mode("MKL")
        with self.assertRaisesRegex(ValueError, r"b_layout must be one of \('LNK', 'LKN'\)"):
            self.module.gemm_b_mode("NKL")
        with self.assertRaisesRegex(ValueError, r"c_layout must be one of \('LMN', 'LNM'\)"):
            self.module.gemm_output_mode("MNL")

        with self.assertRaisesRegex(ValueError, r"d_layout must be one of \('LMN', 'LNM'\)"):
            self.module.gemm_output_mode("MNL", name="d_layout")

    def test_layout_validation_is_case_sensitive(self):
        with self.assertRaisesRegex(ValueError, "got 'lmk'"):
            self.module.gemm_a_mode("lmk")

    def test_layout_validation_requires_a_string(self):
        with self.assertRaisesRegex(TypeError, "a_layout must be a string"):
            self.module.gemm_a_mode(("L", "M", "K"))

    def test_fixed_auxiliary_stride_orders_match_kernel_abis(self):
        self.assertEqual(self.module.ROW_MAJOR_STRIDE_ORDER_3D, (2, 1, 0))
        self.assertEqual(self.module.BLOCK_SCALE_STRIDE_ORDER, (3, 1, 0, 4, 2, 5))
        self.assertEqual(self.module.PROBABILITY_STRIDE_ORDER, (0, 1, 2))

        layout = importlib.import_module(f"{self.module.__package__}.layout")
        block_scale_shape = (32, 4, 2, 4, 3, 5)
        block_scale_stride = layout.compact_stride(block_scale_shape, self.module.BLOCK_SCALE_STRIDE_ORDER)
        self.assertEqual(
            layout.to_cutlass_layout(
                block_scale_shape,
                block_scale_stride,
                self.module.BLOCK_SCALE_STRIDE_ORDER,
            ),
            (2, 1, 4, 0, 3, 5),
        )

        probability_shape = (7, 1, 5)
        probability_stride = layout.compact_stride(probability_shape, self.module.PROBABILITY_STRIDE_ORDER)
        self.assertEqual(
            layout.to_cutlass_layout(
                probability_shape,
                probability_stride,
                self.module.PROBABILITY_STRIDE_ORDER,
            ),
            (0, 1, 2),
        )


if __name__ == "__main__":
    unittest.main()
