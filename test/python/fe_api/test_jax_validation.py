# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for JAX API-base validation helpers."""

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
    def __init__(self, dtype, *, name=None):
        self.dtype = dtype
        if name is not None:
            self.name = name


class JaxApiBaseValidationTest(unittest.TestCase):
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
        fake_jax.tree_util = types.SimpleNamespace(
            DictKey=lambda key: key,
            register_pytree_with_keys=lambda *_args: None,
        )

        def identity_jit(fn=None, **_kwargs):
            return (lambda decorated_fn: decorated_fn) if fn is None else fn

        fake_cutlass_jax = types.ModuleType("cutlass.jax")
        fake_cutlass_jax.TensorSpec = type("TensorSpec", (), {})
        fake_cute = types.ModuleType("cutlass.cute")
        fake_cute.jit = identity_jit
        fake_cutlass = types.ModuleType("cutlass")
        fake_cutlass.__path__ = []
        fake_cutlass.Constexpr = object
        fake_cutlass.cute = fake_cute
        fake_cutlass.jax = fake_cutlass_jax

        parent = types.ModuleType(_TEST_PACKAGE)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = _TEST_PACKAGE
        sys.modules[_TEST_PACKAGE] = parent
        with mock.patch.dict(
            sys.modules,
            {
                "jax": fake_jax,
                "jax.numpy": fake_jnp,
                "cutlass": fake_cutlass,
                "cutlass.cute": fake_cute,
                "cutlass.jax": fake_cutlass_jax,
            },
        ):
            cls.api_base = importlib.import_module(f"{_TEST_PACKAGE}._jax.api_base")

    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    def test_accepts_dtype_like_values_and_returns_normalized_dtype(self):
        self.assertIs(
            self.api_base.require_dtype("float32", (_Float16, _Float32)),
            self.float32,
        )

    def test_accepts_objects_with_dtype_metadata(self):
        value = _ArrayMetadata(self.float16)
        self.assertIs(
            self.api_base.require_dtype(value, (_Float16, _Float32)),
            self.float16,
        )
        self.assertIs(self.api_base.as_dtype(value), self.float16)
        self.assertIs(self.api_base.as_optional_dtype(value), self.float16)
        self.assertIsNone(self.api_base.as_optional_dtype(None))

    def test_does_not_unwrap_scalar_dtype_classes(self):
        self.assertIs(
            self.api_base.require_dtype(_Float32, (_Float32,)),
            self.float32,
        )

    def test_applies_default_only_to_none(self):
        self.assertIs(
            self.api_base.require_dtype(
                None,
                (_Float32,),
                name="output_dtype",
                default=_Float32,
            ),
            self.float32,
        )
        with self.assertRaisesRegex(ValueError, "output_dtype must not be None"):
            self.api_base.require_dtype(None, (_Float32,), name="output_dtype")

    def test_infers_or_accepts_diagnostic_name(self):
        named_value = _ArrayMetadata(self.float16, name="sample")
        with self.assertRaisesRegex(
            ValueError,
            r"sample\.dtype must be one of \{float32\}, got float16",
        ):
            self.api_base.require_dtype(named_value, (_Float32,))

        with self.assertRaisesRegex(ValueError, r"dtype must be one of \{float32\}, got float16"):
            self.api_base.require_dtype(_Float16, (_Float32,))

        with self.assertRaisesRegex(
            ValueError,
            r"compute_dtype must be one of \{float32\}, got float16",
        ):
            self.api_base.require_dtype(_Float16, (_Float32,), name="compute_dtype")

    def test_rejects_invalid_dtype_values(self):
        with self.assertRaisesRegex(TypeError, "Cannot interpret"):
            self.api_base.require_dtype(object(), (_Float32,))


if __name__ == "__main__":
    unittest.main()
