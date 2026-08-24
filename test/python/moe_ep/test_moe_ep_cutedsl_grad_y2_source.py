# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Source-only contracts for the upstream grad_y2 and dFC2 scale layout."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_CUTEDSL = (
    _ROOT
    / "python/cudnn/moe_ep/_megamoe_backend/cutedsl_src/kernel_src/rubin"
    / "training/mega"
)
_DGLU = _CUTEDSL / "bwd_dglu/dglu_mxfp8_mega_moe_kernel.py"
_DGLU_EPILOGUE = _CUTEDSL / "bwd_dglu/dglu_mxfp8_fc12_epilogue.py"


def _source_and_tree(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


@pytest.mark.L0
def test_dglu_source_exports_upstream_grad_y2_col_quant():
    source, tree = _source_and_tree(_DGLU)

    assert tree is not None
    for contract in (
        "enable_grad_y2_col_quant",
        "num_ctas_grad_y2_col_quant",
        "grad_y2_sizes_region",
        "_snapshot_grad_y2_expert_sizes",
        "grad_y2_col_quant",
        "grad_y2: cute.Tensor",
        "grad_y2_sf: cute.Tensor",
    ):
        assert contract in source
    assert source.index(
        "self._snapshot_grad_y2_expert_sizes(tidx)"
    ) < source.index("self.token_comm.reset_tail()")
    assert source.index("self._topk_reduce(") < source.index(
        "self.grad_y2_col_quant("
    )


@pytest.mark.L0
def test_dfc2_scale_source_uses_upstream_mn_major_atoms():
    source, tree = _source_and_tree(_DGLU_EPILOGUE)

    assert tree is not None
    assert "def _stg_col_sf_atom_value(" in source
    assert "MN-major 128-column × 4-token-block atom" in source
    assert "atom_idx * Int64(512)" in source
    assert "Int64(hidden_lane) * Int64(16)" in source
    assert "Int64(hidden_bank) * Int64(4)" in source
    assert "Int64(token_bank)" in source
    assert source.count("self._stg_col_sf_atom_value(") >= 2
