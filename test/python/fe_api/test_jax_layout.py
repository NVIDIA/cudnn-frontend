# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free tests for CUTLASS JAX layout conversion."""

import importlib.util
from pathlib import Path
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_LAYOUT_FILE = Path(__file__).resolve().parents[3] / "python" / "cudnn" / "_jax" / "layout.py"


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

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode must be a permutation"):
            self.layout.normalize_mode(3, (0, 0, 2))


if __name__ == "__main__":
    unittest.main()
