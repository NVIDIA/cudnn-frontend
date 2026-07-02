# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for shared JAX validation helpers."""

from __future__ import annotations

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


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"
_TEST_PACKAGE = "cudnn_frontend_jax_validation_test"


class _DType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class _Float16:
    dtype = object()


class _Float32:
    dtype = object()


class _ArrayMetadata:
    def __init__(self, dtype):
        self.dtype = dtype


class JaxValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.float16 = _DType("float16")
        cls.float32 = _DType("float32")
        dtype_map = {
            _Float16: cls.float16,
            _Float32: cls.float32,
            "float16": cls.float16,
            "float32": cls.float32,
        }

        fake_jnp = types.ModuleType("jax.numpy")

        def dtype(value):
            if isinstance(value, _DType):
                return value
            try:
                return dtype_map[value]
            except (KeyError, TypeError):
                raise TypeError(f"Cannot interpret {value!r} as a dtype") from None

        fake_jnp.dtype = dtype
        fake_jax = types.ModuleType("jax")
        fake_jax.__path__ = []
        fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        fake_jax.numpy = fake_jnp

        parent = types.ModuleType(_TEST_PACKAGE)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = _TEST_PACKAGE
        sys.modules[_TEST_PACKAGE] = parent
        with mock.patch.dict(sys.modules, {"jax": fake_jax, "jax.numpy": fake_jnp}):
            cls.validation = importlib.import_module(f"{_TEST_PACKAGE}._jax.validation")

    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    def test_accepts_dtype_like_values_and_returns_normalized_dtype(self):
        self.assertIs(
            self.validation.require_dtype("dtype", "float32", (_Float16, _Float32)),
            self.float32,
        )

    def test_accepts_objects_with_dtype_metadata(self):
        value = _ArrayMetadata(self.float16)
        self.assertIs(
            self.validation.require_dtype("value", value, (_Float16, _Float32)),
            self.float16,
        )
        self.assertIs(self.validation.as_dtype(value), self.float16)
        self.assertIs(self.validation.as_optional_dtype(value), self.float16)
        self.assertIsNone(self.validation.as_optional_dtype(None))

    def test_does_not_unwrap_scalar_dtype_classes(self):
        self.assertIs(
            self.validation.require_dtype("dtype", _Float32, (_Float32,)),
            self.float32,
        )

    def test_applies_default_only_to_none(self):
        self.assertIs(
            self.validation.require_dtype("dtype", None, (_Float32,), default=_Float32),
            self.float32,
        )
        with self.assertRaisesRegex(ValueError, "dtype must not be None"):
            self.validation.require_dtype("dtype", None, (_Float32,))

    def test_rejects_unsupported_and_invalid_dtype_values(self):
        with self.assertRaisesRegex(ValueError, r"dtype must be one of \{float32\}, got float16"):
            self.validation.require_dtype("dtype", _Float16, (_Float32,))
        with self.assertRaisesRegex(TypeError, "Cannot interpret"):
            self.validation.require_dtype("dtype", object(), (_Float32,))


if __name__ == "__main__":
    unittest.main()
