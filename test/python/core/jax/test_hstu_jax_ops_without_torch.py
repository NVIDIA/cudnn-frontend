# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JAX HSTU ops resolve without importing PyTorch."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.L0


def test_hstu_jax_ops_import_without_torch():
    pytest.importorskip("jax", minversion="0.9.1")
    from cudnn.frost.buffers import cutedsl_requirement_error

    requirement_error = cutedsl_requirement_error("JAX op import test")
    if requirement_error:
        pytest.skip(requirement_error)
    script = """
import sys
class NoTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("torch import attempted: " + fullname)
sys.meta_path.insert(0, NoTorch())
import cudnn.ops
for name in ("hstu_attention_forward_jax", "hstu_attention_backward_jax", "hstu_lmsd_forward_jax", "hstu_lmsd_backward_jax"):
    assert callable(getattr(cudnn.ops, name))
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
