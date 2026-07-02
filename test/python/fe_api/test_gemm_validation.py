# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for shared dense GEMM validation."""

import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_MODULE_PATH = Path(__file__).resolve().parents[3] / "python" / "cudnn" / "gemm_validation.py"
_SPEC = importlib.util.spec_from_file_location("cudnn_gemm_validation_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class GemmValidationTest(unittest.TestCase):
    def test_reconciles_input_shapes(self):
        self.assertEqual(
            _MODULE.require_gemm_shapes((129, 257, 3), (65, 257, 3)),
            (129, 65, 257, 3),
        )

    def test_rejects_invalid_input_shapes(self):
        cases = (
            (((1, 2), (3, 2, 1)), "a_tensor must have rank 3"),
            (((1, 2, 1), (3, 4, 1)), "matching K and L dimensions"),
            (((0, 2, 1), (3, 2, 1)), "M=0"),
        )
        for args, message in cases:
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, message):
                    _MODULE.require_gemm_shapes(*args)

    def test_calculates_and_requires_block_scale_shapes(self):
        expected_a = (32, 4, 2, 4, 3, 2)
        expected_b = (32, 4, 1, 4, 3, 2)
        self.assertEqual(_MODULE.block_scale_shape(129, 257, 2, 32), expected_a)
        _MODULE.require_block_scale_shapes(
            expected_a,
            expected_b,
            m=129,
            n=65,
            k=257,
            batch=2,
            sf_vec_size=32,
        )
        with self.assertRaisesRegex(ValueError, "sfa_tensor must have shape"):
            _MODULE.require_block_scale_shapes(
                (32, 4, 2, 4, 2, 2),
                expected_b,
                m=129,
                n=65,
                k=257,
                batch=2,
                sf_vec_size=32,
            )

    def test_validates_mma_tiler(self):
        self.assertEqual(
            _MODULE.require_mma_tiler(
                (256, 192),
                allowed_m=(128, 256),
                allowed_n=(64, 128, 192, 256),
            ),
            (256, 192),
        )
        with self.assertRaisesRegex(ValueError, r"N in \{64, 128, 192, 256\}"):
            _MODULE.require_mma_tiler(
                (128, 32),
                allowed_m=(128, 256),
                allowed_n=(64, 128, 192, 256),
            )

    def test_validates_cluster_shape(self):
        for cluster_shape in ((1, 1), (2, 4), (4, 4)):
            with self.subTest(cluster_shape=cluster_shape):
                self.assertEqual(
                    _MODULE.require_cluster_shape(
                        cluster_shape,
                        mma_m=128,
                        max_dimension=4,
                    ),
                    cluster_shape,
                )

        for cluster_shape, mma_m, message in (
            ((0, 1), 128, "positive powers of two"),
            ((3, 1), 128, "positive powers of two"),
            ((8, 1), 128, "each at most 4"),
            ((4, 8), 128, "product at most 16"),
            ((1, 2), 256, "divisible by 2"),
        ):
            with self.subTest(cluster_shape=cluster_shape, mma_m=mma_m):
                with self.assertRaisesRegex(ValueError, message):
                    _MODULE.require_cluster_shape(
                        cluster_shape,
                        mma_m=mma_m,
                        max_dimension=4,
                    )

    def test_validates_contiguous_alignment_for_scalar_and_packed_types(self):
        _MODULE.require_contiguous_alignment("fp8", 16, 8)
        _MODULE.require_contiguous_alignment("fp16", 8, 16)
        _MODULE.require_contiguous_alignment("packed_fp4", 32, 4)
        with self.assertRaisesRegex(ValueError, "16-byte aligned"):
            _MODULE.require_contiguous_alignment("fp16", 7, 16)

    def test_requires_complete_mma_rows(self):
        _MODULE.require_full_mma_rows(256, 128)
        _MODULE.require_full_mma_rows(384, 256, cta_group_size=2)
        with self.assertRaisesRegex(ValueError, "TILE_M=256"):
            _MODULE.require_full_mma_rows(
                130,
                256,
                cta_group_size=2,
                reason="the probability load is not predicated",
            )

    def test_validates_swiglu_block_pairs(self):
        self.assertEqual(_MODULE.require_swiglu_n(128), 64)
        with self.assertRaisesRegex(ValueError, "32-column SwiGLU block pairs"):
            _MODULE.require_swiglu_n(96)

    def test_resolves_cluster_overlap_margin(self):
        self.assertEqual(_MODULE.resolve_max_active_clusters(12, 2), 10)
        with self.assertRaisesRegex(ValueError, "CUDNNFE_CLUSTER_OVERLAP_MARGIN"):
            _MODULE.resolve_max_active_clusters(2, 2)


if __name__ == "__main__":
    unittest.main()
