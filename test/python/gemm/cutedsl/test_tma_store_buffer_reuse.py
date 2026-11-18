# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
A TMA-store epilogue may rewrite an smem output buffer only after the bulk store reading it has been waited (#2350).

``PipelineTmaStore.producer_commit()`` is ``cp.async.bulk.commit_group`` and ``producer_acquire()`` is
``cp.async.bulk.wait_group.read (num_stages - 1)``.  A store pipeline that is committed but never acquired lets the
epilogue write a later subtile into a buffer that an earlier bulk store may still be reading; only ``producer_tail``
waits, at kernel exit.  The block-scaled MoE GLU epilogues did this for D: on SM107 with ``generate_c=False`` it
silently corrupted D / D_col codes, and elsewhere the C store's acquire masked it.

The source checks need no GPU.  A kernel may acquire right after the commit (the GLU and quant epilogues) or before
the next write into the buffer (the dGLU / dSwiGLU epilogues, the GLU-Hadamard delayed acquire); either placement
passes.  What fails is a committed store that the kernel never acquires.  The device reproducer is
``test_grouped_gemm_glu.py::test_rubin_mxfp8_glu_without_c_matches_with_c``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GEMM_ROOT = _REPO_ROOT / "python" / "cudnn" / "gemm" / "cutedsl"

# The four epilogues fixed for #2350: each commits its D store and must acquire it.
_D_STORE_KERNELS = (
    "grouped/glu/moe_blockscaled_grouped_gemm_glu_rubin.py",
    "grouped/glu/moe_blockscaled_grouped_gemm_glu_bias.py",
    "discrete_grouped/swiglu/discrete_B_blockscaled_grouped_gemm_glu_bias.py",
    "grouped/swiglu/grouped_gemm_swiglu_quant.py",
)


def _tma_store_calls(path):
    """Map each name bound to ``PipelineTmaStore.create(...)`` in ``path`` to the methods called on that name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func).endswith("PipelineTmaStore.create"):
            calls.update({target.id: set() for target in node.targets if isinstance(target, ast.Name)})
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in calls:
            calls[node.func.value.id].add(node.func.attr)
    return calls


@pytest.mark.L0
@pytest.mark.parametrize("kernel", _D_STORE_KERNELS)
def test_block_scaled_glu_d_store_is_acquired(kernel):
    calls = _tma_store_calls(_GEMM_ROOT / kernel)
    assert "producer_commit" in calls.get("d_pipeline", ()), f"{kernel}: no committed d_pipeline TMA store found"
    assert "producer_acquire" in calls["d_pipeline"], f"{kernel}: the D TMA store is committed but never acquired"


@pytest.mark.L0
def test_every_committed_tma_store_is_acquired():
    offenders = []
    for path in sorted(_GEMM_ROOT.rglob("*.py")):
        for name, methods in _tma_store_calls(path).items():
            if "producer_commit" in methods and "producer_acquire" not in methods:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}: {name}")
    assert offenders == [], f"TMA stores committed but never acquired before the buffer is rewritten: {offenders}"
