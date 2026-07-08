# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for DSA compilation targets."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_MODULE_PATH = Path(__file__).resolve().parents[3] / "python" / "cudnn" / "deepseek_sparse_attention" / "utils" / "compiler.py"


class DsaCompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = spec_from_file_location("dsa_compiler_contract", _MODULE_PATH)
        assert spec is not None and spec.loader is not None
        cls.module = module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_explicit_targets_do_not_import_torch(self):
        torch_before = sys.modules.get("torch")
        self.assertEqual(self.module.gpu_arch_flag_for_compute_capability(90), "sm_90a")
        self.assertEqual(self.module.gpu_arch_flag_for_compute_capability(100), "sm_100a")
        self.assertEqual(self.module.gpu_arch_flag_for_compute_capability(103), "sm_103a")
        self.assertEqual(self.module.gpu_arch_flag_for_compute_capability(107), "sm_100f")
        self.assertIs(sys.modules.get("torch"), torch_before)

    def test_builds_jax_compile_options_from_explicit_target(self):
        self.assertEqual(
            self.module.compile_options_for_target(103, "--opt-level 3"),
            "--enable-tvm-ffi --gpu-arch sm_103a --opt-level 3",
        )

    def test_rejects_unknown_or_non_integer_targets(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported GPU compute capability SM101"):
            self.module.compile_options_for_target(101)
        with self.assertRaisesRegex(TypeError, "must be an int"):
            self.module.compile_options_for_target(True)


if __name__ == "__main__":
    unittest.main()
