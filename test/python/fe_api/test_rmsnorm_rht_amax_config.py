# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CPU-only tests for shared RMSNorm + RHT launch configuration."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest

try:
    import pytest
except ImportError:
    # Keep this contract test runnable with the standard library alone.
    pass
else:
    pytestmark = pytest.mark.L0


_MODULE_PATH = Path(__file__).resolve().parents[3] / "python" / "cudnn" / "rmsnorm_rht_amax" / "config.py"
_SPEC = importlib.util.spec_from_file_location("cudnn_rmsnorm_rht_amax_config", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _desc(shape, dtype_name):
    shape = tuple(shape)
    return types.SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype_name=dtype_name,
    )


class RmsNormRhtAmaxLaunchConfigTest(unittest.TestCase):
    def test_package_does_not_eagerly_import_torch_api(self):
        package_dir = _MODULE_PATH.parent
        root_name = "cudnn_frontend_rmsnorm_lazy_import_test"
        module_name = f"{root_name}.rmsnorm_rht_amax"
        root = types.ModuleType(root_name)
        root.__path__ = [str(package_dir.parent)]
        root.__package__ = root_name
        sys.modules[root_name] = root
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
            for loaded_name in tuple(sys.modules):
                if loaded_name == root_name or loaded_name.startswith(f"{root_name}."):
                    sys.modules.pop(loaded_name, None)

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

    def test_validates_tensor_metadata_and_infers_outputs(self):
        plan = _MODULE.validate_rmsnorm_rht_amax(
            _desc((256, 2048), "bfloat16"),
            _desc((2048,), "bfloat16"),
            output=_desc((256, 2048), "bfloat16"),
            amax=_desc((64,), "float32"),
            num_threads=128,
            rows_per_cta=4,
        )

        self.assertEqual(plan.m, 256)
        self.assertEqual(plan.n, 2048)
        self.assertEqual(plan.num_threads, 128)
        self.assertEqual(plan.rows_per_cta, 4)
        self.assertEqual(plan.output_shape, (256, 2048))
        self.assertEqual(plan.amax_shape, (64,))

    def test_rejects_invalid_tensor_metadata(self):
        valid_x = _desc((256, 2048), "bfloat16")
        valid_weight = _desc((2048,), "bfloat16")
        cases = (
            (
                (_desc((256, 2048, 1), "bfloat16"), valid_weight),
                {},
                "X must have rank 2",
            ),
            (
                (valid_x, _desc((2048, 1), "bfloat16")),
                {},
                "W must have rank 1",
            ),
            (
                (_desc((256, 2048), "float16"), valid_weight),
                {},
                "X must have dtype bfloat16",
            ),
            (
                (valid_x, _desc((2048,), "float32")),
                {},
                "W must have dtype bfloat16",
            ),
            (
                (valid_x, _desc((1024,), "bfloat16")),
                {},
                r"W must have shape \(2048,\)",
            ),
            (
                (valid_x, valid_weight),
                {"output": _desc((256, 1024), "bfloat16")},
                "O must have shape",
            ),
            (
                (valid_x, valid_weight),
                {"output": _desc((256, 2048), "float16")},
                "O must have dtype bfloat16",
            ),
            (
                (valid_x, valid_weight),
                {"amax": _desc((128,), "float32"), "rows_per_cta": 4},
                "Amax must have shape",
            ),
            (
                (valid_x, valid_weight),
                {"amax": _desc((64,), "bfloat16"), "rows_per_cta": 4},
                "Amax must have dtype float32",
            ),
        )

        for args, kwargs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _MODULE.validate_rmsnorm_rht_amax(*args, **kwargs)

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
