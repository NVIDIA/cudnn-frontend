# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts for CUTLASS JAX compilation targets."""

import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import types
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_CUDNN_ROOT = Path(__file__).resolve().parents[3] / "python" / "cudnn"
_PACKAGE = "cudnn_jax_compiler_test"


class JaxCompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        sys.modules[_PACKAGE] = root

        jax_package_name = f"{_PACKAGE}._jax"
        jax_package = types.ModuleType(jax_package_name)
        jax_package.__path__ = [str(_CUDNN_ROOT / "_jax")]
        jax_package.__package__ = jax_package_name
        jax_package.__spec__ = ModuleSpec(jax_package_name, loader=None, is_package=True)
        sys.modules[jax_package_name] = jax_package

        cls.arch = importlib.import_module(f"{_PACKAGE}.common.cute_arch")
        cls.compiler = importlib.import_module(f"{jax_package_name}.compiler")

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def test_explicit_targets_do_not_import_frameworks(self):
        torch_before = sys.modules.get("torch")
        jax_before = sys.modules.get("jax")
        self.assertEqual(self.arch.gpu_arch_flag_for_compute_capability(90), "sm_90a")
        self.assertEqual(self.arch.gpu_arch_flag_for_compute_capability(100), "sm_100a")
        self.assertEqual(self.arch.gpu_arch_flag_for_compute_capability(103), "sm_103a")
        self.assertEqual(self.arch.gpu_arch_flag_for_compute_capability(107), "sm_100f")
        self.assertIs(sys.modules.get("torch"), torch_before)
        self.assertIs(sys.modules.get("jax"), jax_before)

    def test_builds_jax_compile_options_from_explicit_target(self):
        self.assertEqual(
            self.compiler.compile_options_for_target(103, "--opt-level 3"),
            "--gpu-arch sm_103a --opt-level 3",
        )

    def test_rejects_unknown_or_non_integer_targets(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported GPU compute capability SM101"):
            self.compiler.compile_options_for_target(101)
        with self.assertRaisesRegex(TypeError, "must be an int"):
            self.compiler.compile_options_for_target(True)


if __name__ == "__main__":
    unittest.main()
