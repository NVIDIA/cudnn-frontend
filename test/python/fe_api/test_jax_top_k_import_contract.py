# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Import-safety contract for the JAX indexer top-K kernel path."""

import ast
from pathlib import Path
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "cudnn"
    / "deepseek_sparse_attention"
    / "indexer_top_k"
    / "indexer_top_k_decode_varlen.py"
)


class JaxTopKImportContractTest(unittest.TestCase):
    def test_torch_compiler_helper_is_not_imported_at_module_scope(self):
        tree = ast.parse(_MODULE_PATH.read_text(), filename=str(_MODULE_PATH))
        imported_modules = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn(
            "cudnn.deepseek_sparse_attention.utils.compiler",
            imported_modules,
        )


if __name__ == "__main__":
    unittest.main()
