# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tests for reusable JAX wrapper utilities."""

from __future__ import annotations

from enum import IntEnum
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
import unittest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "python" / "cudnn" / "jax" / "utils.py"
_OPTIONAL_ROOTS = ("jax", "numpy", "torch", "cutlass")
_OPTIONAL_MODULES_BEFORE = {name for name in sys.modules if name.split(".", 1)[0] in _OPTIONAL_ROOTS}
_SPEC = importlib.util.spec_from_file_location("cudnn_jax_utils_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_OPTIONAL_MODULES_AFTER = {name for name in sys.modules if name.split(".", 1)[0] in _OPTIONAL_ROOTS}

optional_static_int = _MODULE.optional_static_int
require_concrete_dim = _MODULE.require_concrete_dim
require_concrete_dims = _MODULE.require_concrete_dims
require_static_bool = _MODULE.require_static_bool
require_static_float = _MODULE.require_static_float
require_static_int = _MODULE.require_static_int


class _Integer(IntEnum):
    VALUE = 7


class _DynamicValue:
    def __index__(self):
        raise AssertionError("dynamic values must be rejected before conversion")

    def __float__(self):
        raise AssertionError("dynamic values must be rejected before conversion")


class JaxStaticValuesTest(unittest.TestCase):
    def test_module_imports_no_optional_frameworks(self):
        self.assertEqual(_OPTIONAL_MODULES_AFTER - _OPTIONAL_MODULES_BEFORE, set())

    def test_concrete_dim_accepts_and_normalizes_integral_values(self):
        for value in (0, -1, 4, _Integer.VALUE):
            with self.subTest(value=value):
                result = require_concrete_dim(value, name="M")
                self.assertIs(type(result), int)
                self.assertEqual(result, int(value))

    def test_concrete_dim_rejects_nonintegral_and_symbolic_values(self):
        for value in (True, 4.0, "4", None, _DynamicValue()):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(TypeError, "M.*shape-polymorphic"):
                    require_concrete_dim(value, name="M")

    def test_concrete_dims_normalizes_multiple_values(self):
        result = require_concrete_dims((4, _Integer.VALUE), "M", "N")
        self.assertEqual(result, (4, 7))
        self.assertTrue(all(type(value) is int for value in result))

    def test_concrete_dims_checks_name_count_and_preserves_dimension_name(self):
        with self.assertRaisesRegex(ValueError, "Expected 2 dimension names, got 1"):
            require_concrete_dims((4, 7), "M")
        with self.assertRaisesRegex(TypeError, "N.*shape-polymorphic"):
            require_concrete_dims((4, _DynamicValue()), "M", "N")

    def test_static_int_accepts_required_and_optional_integral_values(self):
        self.assertIsNone(optional_static_int(None, name="num_threads"))
        for value in (0, 128, _Integer.VALUE):
            with self.subTest(value=value):
                required = require_static_int(value, name="num_threads")
                optional = optional_static_int(value, name="num_threads")
                self.assertIs(type(required), int)
                self.assertIs(type(optional), int)
                self.assertEqual(required, int(value))
                self.assertEqual(optional, int(value))

    def test_static_int_rejects_nonintegral_and_dynamic_values(self):
        for value in (None, True, 128.0, "128", 128 + 0j, _DynamicValue()):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(TypeError, "num_threads.*Python-static integer"):
                    require_static_int(value, name="num_threads")

    def test_static_float_accepts_and_normalizes_real_scalars(self):
        for value in (0, 0.125, Fraction(1, 8)):
            with self.subTest(value=value):
                result = require_static_float(value, name="eps")
                self.assertIs(type(result), float)
                self.assertEqual(result, float(value))

    def test_static_bool_accepts_only_builtin_bools(self):
        self.assertIs(require_static_bool(True, name="return_val"), True)
        self.assertIs(require_static_bool(False, name="return_val"), False)
        for value in (None, 0, 1, "true", _DynamicValue()):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(TypeError, "return_val.*Python-static bool"):
                    require_static_bool(value, name="return_val")

    def test_static_float_rejects_nonreal_and_dynamic_values(self):
        for value in (None, True, "0.125", b"0.125", 1 + 0j, _DynamicValue()):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(TypeError, "eps.*Python-static real"):
                    require_static_float(value, name="eps")

    def test_static_float_translates_overflow(self):
        with self.assertRaisesRegex(TypeError, "eps.*Python-static real"):
            require_static_float(10**10000, name="eps")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
    def test_numpy_scalars_are_host_static_but_arrays_are_not(self):
        import numpy as np

        for value in (np.int8(1), np.int64(2), np.uint64(3)):
            self.assertIs(type(require_static_int(value, name="value")), int)
        for value in (np.float32(0.5), np.float64(0.25), np.int32(1)):
            self.assertIs(type(require_static_float(value, name="value")), float)
        for value in (np.bool_(True), np.array(1), np.array([1])):
            with self.assertRaises(TypeError):
                require_static_float(value, name="value")

    @unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is not installed")
    def test_jax_tracers_and_symbolic_dimensions_are_rejected(self):
        import jax

        for converter, name in (
            (require_static_int, "num_threads"),
            (require_static_float, "eps"),
            (require_static_bool, "return_val"),
        ):
            with self.subTest(converter=converter.__name__):
                with self.assertRaisesRegex(TypeError, "Python-static"):
                    jax.jit(lambda value: converter(value, name=name))(1)

        symbolic_dim = jax.export.symbolic_shape("m")[0]
        with self.assertRaisesRegex(TypeError, "shape-polymorphic"):
            require_concrete_dim(symbolic_dim, name="M")


if __name__ == "__main__":
    unittest.main()
