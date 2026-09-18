# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public M64 records retain native workspace, cache and graph admission contracts."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import cudnn
from cudnn.gemm.frost import moe_pair
from cudnn.gemm.frost.heuristics import analyze_facts, recommend
from cudnn.gemm.frost.knobs import GemmKnobs

pytestmark = pytest.mark.L0


def graph_case(rows, layout):
    e, n, k = 4, 64, 128
    b, f = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    g = cudnn.pygraph(io_data_type=b, intermediate_data_type=f, compute_data_type=f)
    x = g.tensor(name="x", dim=[1, rows, k], stride=[rows * k, k, 1], data_type=b)
    dims = [e, k // 64, 2 * n, 64] if layout else [e, k, 2 * n]
    strides = [2 * n * k, 2 * n * 64, 64, 1] if layout else [2 * n * k, 1, k]
    w = g.tensor(name="w", dim=dims, stride=strides, data_type=b)
    offsets = g.tensor(name="off", dim=[e, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    outputs = []
    for start in (0, n):
        slices = [slice(None), slice(None), slice(start, start + n)] + ([slice(None)] if layout else [])
        weight = g.slice(w, slices).set_stride(strides)
        outputs.append(g.moe_grouped_matmul(x, weight, offsets, mode=cudnn.moe_grouped_matmul_mode.NONE, weight_layout=layout))
    g.mul(g.swish(outputs[0]), outputs[1]).set_dim([1, rows, n]).set_stride([rows * n, n, 1]).set_data_type(b).set_output(True)
    return g


@pytest.mark.parametrize("rows", [9, 64, 512, 513])
@pytest.mark.parametrize("layout", [None, "k_blocked_64_v1"])
def test_tile_record_controls_grid_workspace_and_template(rows, layout, monkeypatch):
    seen = []
    base = moe_pair.KernelParams(132, 44214954, "test helper identity")
    monkeypatch.setattr(moe_pair, "device_params", lambda: base)

    def load(path, params, *, tag):
        seen.append((Path(path).name, params, tag))
        return SimpleNamespace(compile=lambda: object())

    monkeypatch.setattr("cudnn.frost.template_loader.load_template", load)
    g = graph_case(rows, layout)
    for tile in (128, 64):
        knobs = moe_pair.pair_knobs(tile)
        assert GemmKnobs.from_public(knobs.to_public()) == knobs
        plan = moe_pair.build_pair(g, knobs)
        grid = 132 if tile == 128 else 264
        assert plan.workspace_bytes == (grid * 2 + 1) * 128
        assert plan.tile_m == tile and plan.spec.weight_layout == layout
        template, params, tag = seen[-1]
        assert template == ("sm100_moe_swiglu_pair.py" if tile == 128 else "sm100_moe_swiglu_pair_m64.py")
        assert params == replace(base, grid_ctas=grid, weight_layout=layout)
        assert tag == ("moe_swiglu_pair" if tile == 128 else "moe_swiglu_pair_m64")
    moe_pair.build_pair(g, None)
    assert seen[-1] == seen[0] and seen[0] != seen[1]
    proposals = recommend("A", analyze_facts(g), {"frost_moe_swiglu_pair": 20401})
    assert [p.knobs for p in proposals] == [moe_pair.pair_knobs(), moe_pair.pair_knobs(64)]


@pytest.mark.parametrize("rows", [1, 8])
def test_small_rows_keep_existing_enumeration_and_decline_m64(rows, monkeypatch):
    g = graph_case(rows, "k_blocked_64_v1")
    monkeypatch.setattr(moe_pair, "device_params", lambda: None)
    proposals = recommend("A", analyze_facts(g), {"frost_moe_swiglu_pair": 20401})
    assert [p.knobs for p in proposals] == [moe_pair.pair_knobs()]

    def unexpected():
        raise AssertionError("unsupported tile must decline before device probing")

    monkeypatch.setattr(moe_pair, "device_params", unexpected)
    with pytest.raises(NotImplementedError, match="9..513"):
        moe_pair.build_pair(g, moe_pair.pair_knobs(64))


@pytest.mark.parametrize("change", [{"mma_tile_m": 128}, {"cta_tile_n": 16}, {"moe_sched_policy": 0}])
def test_partial_or_unimplemented_m64_records_decline(change, monkeypatch):
    def unexpected():
        raise AssertionError("unsupported tile must decline before device probing")

    monkeypatch.setattr(moe_pair, "device_params", unexpected)
    with pytest.raises(NotImplementedError, match="exact M128N8 or M64N8"):
        moe_pair.build_pair(graph_case(9, None), replace(moe_pair.pair_knobs(64), **change))


def test_m64_uses_existing_public_axes_only():
    old = moe_pair.pair_knobs().to_public()
    new = moe_pair.pair_knobs(64).to_public()
    assert old.keys() == new.keys()
    assert {k for k in old if old[k] != new[k]} == {cudnn.knob_type.TILE_M, cudnn.knob_type.MMA_TILE_M}
    assert new[cudnn.knob_type.TILE_M] == new[cudnn.knob_type.MMA_TILE_M] == 64


@pytest.mark.parametrize("tile", [32, 256, 64.0, True, "64", None])
def test_invalid_mma_m_rejected(tile):
    with pytest.raises(ValueError, match="physical MMA-M"):
        moe_pair.pair_knobs(tile)
