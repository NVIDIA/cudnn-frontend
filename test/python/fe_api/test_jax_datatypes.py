# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free tests for common JAX/cuDNN dtype conversion."""

from enum import Enum, auto
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
_PACKAGE = "cudnn_jax_datatypes_test"


class _DataType(Enum):
    NOT_SET = auto()
    FLOAT = auto()
    BFLOAT16 = auto()


class JaxDataTypesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        internal_name = f"{_PACKAGE}._jax"
        internal = types.ModuleType(internal_name)
        internal.__path__ = [str(_CUDNN_ROOT / "_jax")]
        internal.__package__ = internal_name
        internal.__spec__ = ModuleSpec(internal_name, loader=None, is_package=True)
        sys.modules[internal_name] = internal

        jax = types.ModuleType("jax")
        jax.__path__ = []
        jnp = types.ModuleType("jax.numpy")
        jnp.bfloat16 = "bfloat16"
        jnp.float32 = "float32"
        jnp.dtype = lambda value: value
        jax.numpy = jnp

        try:
            with mock.patch.dict(sys.modules, {"jax": jax, "jax.numpy": jnp}):
                cls.module = importlib.import_module(f"{internal_name}.datatypes")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def test_supported_types_round_trip(self):
        self.assertEqual(self.module.jax_to_cudnn_dtype("bfloat16"), _DataType.BFLOAT16)
        self.assertEqual(self.module.jax_to_cudnn_dtype("float32"), _DataType.FLOAT)
        self.assertEqual(self.module.cudnn_to_jax_dtype(_DataType.BFLOAT16), "bfloat16")
        self.assertEqual(self.module.cudnn_to_jax_dtype(_DataType.FLOAT), "float32")

    def test_unsupported_types_have_explicit_behavior(self):
        self.assertEqual(self.module.jax_to_cudnn_dtype("unsupported"), _DataType.NOT_SET)
        with self.assertRaisesRegex(ValueError, "Unsupported JAX data type"):
            self.module.cudnn_to_jax_dtype(_DataType.NOT_SET)


if __name__ == "__main__":
    unittest.main()
