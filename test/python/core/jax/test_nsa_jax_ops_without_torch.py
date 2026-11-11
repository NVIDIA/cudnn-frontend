# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JAX NSA ops resolve without importing PyTorch."""

import subprocess
import sys

import pytest

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("JAX op import test")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytest.importorskip("jax", minversion="0.9.1")

pytestmark = pytest.mark.L0


def test_nsa_jax_ops_import_without_torch():
    script = """
import sys
class NoTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("torch import attempted: " + fullname)
sys.meta_path.insert(0, NoTorch())
import cudnn.ops
for name in [n for n in cudnn.ops.nsa.__all__ if n.endswith("_jax")]:
    assert callable(getattr(cudnn.ops.nsa, name))
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
