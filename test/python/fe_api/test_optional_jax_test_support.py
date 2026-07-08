# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free tests for optional JAX test selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import types
import unittest
from unittest import mock

import pytest

pytestmark = pytest.mark.L0

_CONFTEST_PATH = Path(__file__).with_name("conftest.py")


def _load_support_module():
    spec = importlib.util.spec_from_file_location(
        "cudnn_frontend_jax_test_support",
        _CONFTEST_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {_CONFTEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _missing_module(name):
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


class _Item:
    def __init__(self, filename):
        self.path = Path(filename)
        self.added_markers = []

    def add_marker(self, marker):
        self.added_markers.append(marker)


class JaxTestSupportTest(unittest.TestCase):
    def setUp(self):
        self.support = _load_support_module()

    def test_missing_jax_skips_jax_tests(self):
        with mock.patch.object(
            self.support,
            "import_module",
            side_effect=_missing_module("jax"),
        ):
            reason = self.support._jax_test_skip_reason()

        self.assertIn("require JAX", reason)
        self.assertIn("nvidia-cudnn-frontend[jax]", reason)

    def test_missing_cutlass_jax_skips_jax_tests(self):
        jax = types.SimpleNamespace(__version__="0.9.1")

        def import_module(name):
            if name == "jax":
                return jax
            raise _missing_module("cutlass.jax")

        with mock.patch.object(
            self.support, "import_module", side_effect=import_module
        ):
            reason = self.support._jax_test_skip_reason()

        self.assertIn("require CUTLASS JAX support", reason)
        self.assertIn("nvidia-cudnn-frontend[jax]", reason)

    def test_unsupported_jax_version_skips_jax_tests(self):
        jax = types.SimpleNamespace(__version__="0.8.0")
        cutlass_jax = types.SimpleNamespace(
            CUTE_DSL_MIN_SUPPORTED_JAX_VERSION=(0, 9, 1),
            is_available=lambda: False,
        )

        with mock.patch.object(
            self.support,
            "import_module",
            side_effect=lambda name: {
                "jax": jax,
                "cutlass.jax": cutlass_jax,
            }[name],
        ):
            reason = self.support._jax_test_skip_reason()

        self.assertEqual(
            reason,
            "CUTLASS JAX support is unavailable with JAX 0.8.0; "
            "the minimum supported JAX version is 0.9.1.",
        )

    def test_supported_runtime_does_not_skip(self):
        jax = types.SimpleNamespace(__version__="0.9.1")
        cutlass_jax = types.SimpleNamespace(is_available=lambda: True)

        with mock.patch.object(
            self.support,
            "import_module",
            side_effect=lambda name: {
                "jax": jax,
                "cutlass.jax": cutlass_jax,
            }[name],
        ):
            self.assertIsNone(self.support._jax_test_skip_reason())

    def test_transitive_import_failure_is_not_hidden(self):
        error = _missing_module("ml_dtypes")
        with mock.patch.object(
            self.support,
            "import_module",
            side_effect=error,
        ):
            with self.assertRaises(ModuleNotFoundError) as raised:
                self.support._jax_test_skip_reason()

        self.assertIs(raised.exception, error)

    def test_collection_hook_selects_jax_test_filenames(self):
        runtime_item = _Item("test_jax_gemm.py")
        contract_item = _Item("test_jax_gemm_contract.py")
        non_jax_item = _Item("test_api_base_framework_split.py")
        with mock.patch.object(
            self.support,
            "_jax_test_skip_reason",
            return_value="JAX is unavailable",
        ):
            self.support.pytest_collection_modifyitems(
                None,
                [runtime_item, contract_item, non_jax_item],
            )

        self.assertEqual(len(runtime_item.added_markers), 1)
        self.assertEqual(len(contract_item.added_markers), 1)
        self.assertEqual(non_jax_item.added_markers, [])


if __name__ == "__main__":
    unittest.main()
