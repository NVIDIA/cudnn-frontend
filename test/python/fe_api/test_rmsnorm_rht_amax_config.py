# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CPU-only tests for shared RMSNorm + RHT launch configuration."""

import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import pytest
except ImportError:
    # Keep this contract test runnable with the standard library alone.
    pass
else:
    pytestmark = pytest.mark.L0


_MODULE_PATH = Path(__file__).resolve().parents[3] / "python" / "cudnn" / "_rmsnorm_rht_amax_config.py"
_SPEC = importlib.util.spec_from_file_location("cudnn_rmsnorm_rht_amax_config", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class RmsNormRhtAmaxLaunchConfigTest(unittest.TestCase):
    def test_package_does_not_eagerly_import_torch_api(self):
        package_dir = _MODULE_PATH.parent / "rmsnorm_rht_amax"
        module_name = "cudnn_rmsnorm_rht_amax_lazy_import_test"
        package_spec = importlib.util.spec_from_file_location(
            module_name,
            package_dir / "__init__.py",
            submodule_search_locations=[str(package_dir)],
        )
        assert package_spec is not None and package_spec.loader is not None
        package = importlib.util.module_from_spec(package_spec)
        sys.modules[module_name] = package
        try:
            package_spec.loader.exec_module(package)
            self.assertNotIn(f"{module_name}.api", sys.modules)
        finally:
            sys.modules.pop(module_name, None)

    def test_resolves_tuned_defaults(self):
        self.assertEqual(
            _MODULE.resolve_launch_config(256, 2048),
            (128, 2),
        )

    def test_preserves_valid_overrides(self):
        self.assertEqual(
            _MODULE.resolve_launch_config(
                256,
                4096,
                num_threads=256,
                rows_per_cta=4,
            ),
            (256, 4),
        )

    def test_rejects_invalid_dimensions_and_launch_parameters(self):
        cases = (
            ({"m": 0, "n": 2048}, "M must be positive"),
            ({"m": 256, "n": 0}, "N must be positive"),
            ({"m": 256, "n": 2047}, "Hadamard block size"),
            (
                {"m": 256, "n": 2048, "num_threads": 512},
                "EPT=4 must be >= 8 and divisible by 8",
            ),
            (
                {"m": 255, "n": 2048, "rows_per_cta": 2},
                "M must be divisible",
            ),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    _MODULE.resolve_launch_config(**kwargs)


if __name__ == "__main__":
    unittest.main()
