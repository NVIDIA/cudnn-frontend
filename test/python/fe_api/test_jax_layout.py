# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tests for CUTLASS JAX layout conversion."""

import importlib.util
from pathlib import Path
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_LAYOUT_FILE = (
    Path(__file__).resolve().parents[3] / "python" / "cudnn" / "_jax" / "layout.py"
)


def _load_layout_module():
    spec = importlib.util.spec_from_file_location("cudnn_jax_layout_test", _LAYOUT_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load CUTLASS JAX layout helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JaxLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = _load_layout_module()

    def test_non_self_inverse_stride_order_maps_to_cutlass_axis_ranks(self):
        shape = (7, 3, 5)
        stride_order = (1, 2, 0)
        stride = self.layout.compact_stride(shape, stride_order)

        self.assertEqual(stride, (15, 1, 3))
        self.assertEqual(
            self.layout.to_cutlass_layout(shape, stride, stride_order),
            (2, 0, 1),
        )

    def test_named_layout_maps_kernel_axes_to_public_axes(self):
        cases = (
            ("LMK", "MKL", (1, 2, 0)),
            ("LKM", "MKL", (2, 1, 0)),
            ("LNK", "NKL", (1, 2, 0)),
            ("LKN", "NKL", (2, 1, 0)),
            ("LMN", "MNL", (1, 2, 0)),
            ("LNM", "MNL", (2, 1, 0)),
        )
        for public_layout, kernel_axes, expected in cases:
            with self.subTest(public_layout=public_layout, kernel_axes=kernel_axes):
                self.assertEqual(
                    self.layout.mode_from_layout(
                        public_layout,
                        kernel_axes=kernel_axes,
                    ),
                    expected,
                )

    def test_named_layout_requires_strings(self):
        with self.assertRaisesRegex(TypeError, "layout must be a string"):
            self.layout.mode_from_layout(("L", "M", "K"), kernel_axes="MKL")
        with self.assertRaisesRegex(TypeError, "kernel_axes must be a string"):
            self.layout.mode_from_layout("LMK", kernel_axes=("M", "K", "L"))

    def test_named_layout_rejects_rank_mismatch(self):
        with self.assertRaisesRegex(ValueError, "layout rank must match"):
            self.layout.mode_from_layout("MK", kernel_axes="MKL")

    def test_named_layout_rejects_duplicate_axes(self):
        with self.assertRaisesRegex(ValueError, "layout axes must be unique"):
            self.layout.mode_from_layout("MML", kernel_axes="MKL")
        with self.assertRaisesRegex(ValueError, "kernel_axes must be unique"):
            self.layout.mode_from_layout("MKL", kernel_axes="MML")

    def test_named_layout_requires_the_exact_axis_set(self):
        with self.assertRaisesRegex(ValueError, "layout must contain exactly"):
            self.layout.mode_from_layout("MKN", kernel_axes="MKL")

    def test_noncompact_stride_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot represent non-compact stride"):
            self.layout.to_cutlass_layout(
                shape=(7, 3, 5),
                stride=(16, 1, 3),
                stride_order=(1, 2, 0),
                name="sample",
            )

    def test_compact_stride_preserves_order_for_zero_extents(self):
        self.assertEqual(
            self.layout.compact_stride((2, 0, 3), (2, 1, 0)),
            (3, 3, 1),
        )

    def test_identity_mode_is_the_default(self):
        self.assertEqual(self.layout.normalize_mode(3), (0, 1, 2))
        self.assertEqual(self.layout.to_canonical_axes((2, 3, 4)), (2, 3, 4))
        self.assertEqual(self.layout.to_public_axes((2, 3, 4)), (2, 3, 4))

    def test_mode_maps_kernel_axes_to_public_axes(self):
        mode = (2, 0, 1)

        self.assertEqual(self.layout.to_canonical_axes((2, 3, 4), mode), (4, 2, 3))
        self.assertEqual(self.layout.to_public_axes((4, 2, 3), mode), (2, 3, 4))

    def test_cutlass_layout_is_indexed_by_public_axes(self):
        self.assertEqual(
            self.layout.to_cutlass_layout(
                shape=(4, 2, 3),
                stride=(1, 12, 4),
                stride_order=(0, 2, 1),
                mode=(2, 0, 1),
            ),
            (2, 1, 0),
        )

    def test_canonical_stride_order_maps_to_named_public_layout(self):
        bhsd_to_bshd = self.layout.mode_from_layout("BHSD", kernel_axes="BSHD")
        bshd_to_bshd = self.layout.mode_from_layout("BSHD", kernel_axes="BSHD")

        self.assertEqual(
            self.layout.stride_order_to_public((3, 2, 1, 0), bhsd_to_bshd),
            (3, 1, 2, 0),
        )
        self.assertEqual(
            self.layout.stride_order_to_public((3, 2, 1, 0), bshd_to_bshd),
            (3, 2, 1, 0),
        )

        bshd_to_bhsd = self.layout.mode_from_layout("BSHD", kernel_axes="BHSD")
        self.assertEqual(
            self.layout.to_public_axes((2, 8, 32, 16), bshd_to_bhsd),
            (2, 32, 8, 16),
        )
        self.assertEqual(
            self.layout.stride_order_to_public((3, 1, 2, 0), bshd_to_bhsd),
            (3, 2, 1, 0),
        )

    def test_canonical_stride_order_requires_a_permutation(self):
        with self.assertRaisesRegex(ValueError, "must be a permutation"):
            self.layout.stride_order_to_public((2, 2, 0))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode must be a permutation"):
            self.layout.normalize_mode(3, (0, 0, 2))


if __name__ == "__main__":
    unittest.main()
