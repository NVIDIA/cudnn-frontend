# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel imports must not depend on the adapter's heuristics layer."""

from pathlib import Path
import subprocess
import sys

import cudnn
import pytest

from frost_test_utils import requires_dsl


@pytest.mark.L0
@requires_dsl
@pytest.mark.parametrize("entry", ["cudnn.sdpa.fwd.kernels.sm100.split_combine", "cudnn.sdpa.fwd.kernels.sm80.prepared_host"])
def test_split_combine_import_without_heuristics(entry):
    # A fresh interpreter catches dependency cycles that the suite's already
    # imported adapters can hide. Use the same checkout as the parent process.
    code = """
import importlib
import importlib.abc
import sys

sys.meta_path[:] = [finder for finder in sys.meta_path if '__editable__' not in getattr(finder, '__module__', '')]
sys.path.insert(0, sys.argv[1])

class NoHeuristics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cudnn.sdpa.fwd.heuristics':
            raise AssertionError('kernel import depends on heuristics')

guard = NoHeuristics()
sys.meta_path.insert(0, guard)
importlib.import_module(sys.argv[2])
assert 'cudnn.sdpa.fwd.heuristics' not in sys.modules
from cudnn.sdpa.fwd.kernels.sm100 import split_combine
assert split_combine.decode_grid_selected(4, 32, 1, 128)
assert not split_combine.decode_grid_selected(1, 129, 1, 128)
sys.meta_path.remove(guard)
from cudnn.sdpa.fwd import heuristics
assert heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS == 128
assert split_combine._DECODE_COMBINE_MAX_LEAN_BLOCKS == heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS
"""
    result = subprocess.run([sys.executable, "-c", code, str(Path(cudnn.__file__).resolve().parent.parent), entry], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
