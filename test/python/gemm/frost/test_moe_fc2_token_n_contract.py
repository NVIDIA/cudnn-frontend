# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N16 must reach public knobs, the actual template and immutable cache identity."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import cudnn
from cudnn.gemm.frost.moe_pair import KernelParams, pair_knobs
from cudnn.gemm.frost.moe_fc2_pair import Fc2Knobs, build_fc2
from test_moe_fc2_pair_contract import make_graph

pytestmark = [pytest.mark.L0]


@pytest.mark.parametrize("n", [8, 16])
@pytest.mark.parametrize("stages", [12, 6])
def test_token_tile_roundtrip(n, stages):
    knobs = Fc2Knobs(replace(pair_knobs(), cta_tile_n=n, mma_tile_n=n), stages)
    record = knobs.to_public()
    assert record[cudnn.knob_type.TILE_N] == n
    assert record[cudnn.knob_type.MMA_TILE_N] == n
    assert Fc2Knobs.from_public(record) == knobs
    if n == 8 and stages == 12:
        assert record == pair_knobs().to_public()


@pytest.mark.parametrize(
    "changes",
    [
        dict(cta_tile_n=16),
        dict(mma_tile_n=16),
        dict(cta_tile_n=32, mma_tile_n=32),
        dict(cta_tile_m=64),
        dict(moe_sched_policy=0),
        dict(cta_tile_n=64, mma_tile_n=64),
        dict(cta_tile_n=16, mma_tile_n=8),
    ],
)
def test_invalid_tile_declines_before_device(changes):
    record = replace(pair_knobs(), **changes).to_public()
    with patch("cudnn.gemm.frost.moe_fc2_pair.device_params", side_effect=AssertionError("device probe")):
        with pytest.raises(NotImplementedError, match="exact M128N8"):
            build_fc2(make_graph(), record)


def test_tile_selects_template_and_immutable_key():
    seen = []

    def load(path, params, tag):
        seen.append((path, params, tag))
        return SimpleNamespace(compile=lambda: object())

    with (
        patch("cudnn.gemm.frost.moe_fc2_pair.device_params", return_value=KernelParams(148, 44214954, "cpu-proof")),
        patch("cudnn.frost.template_loader.load_template", side_effect=load),
    ):
        for n, stages in ((8, 12), (16, 12), (8, 6), (16, 6)):
            knobs = Fc2Knobs(replace(pair_knobs(), cta_tile_n=n, mma_tile_n=n), stages)
            compiled = build_fc2(make_graph(), knobs.to_public())
            assert compiled.workspace_bytes == 38016
            path, params, tag = seen[-1]
            assert params.token_n == n and params.ab_stages == stages
            assert Path(path).name == ("sm100_moe_fc2_pair_n16.py" if n == 16 else "sm100_moe_fc2_pair.py")
    assert len({params for _, params, _ in seen}) == 4
