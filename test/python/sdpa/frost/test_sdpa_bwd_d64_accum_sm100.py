# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense d64 accumulation backward, with and without a left window: selection, numerics and replay."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from test_sdpa_bwd_d128_sm100 import _case, _check_case
from test_sdpa_bwd_dsl_sm107 import _reference64

pytestmark = [pytest.mark.L0, requires_dsl]


def test_window_and_head_dim_select_the_accumulation_route():
    """Head dim 64 and a left window at d64 stay on the accumulation route; a d128 window and a deterministic request do not."""
    from cudnn.sdpa.bwd.d128_contract import workspace_layout

    regions, total = workspace_layout(2, 3, 300, 64)
    assert regions[0][1] == (2, 384, 3, 64) and regions[0][2] == (384 * 3 * 64, 3 * 64, 64, 1)
    assert regions[1][0] == 4 * 2 * 384 * 3 * 64 and total == 4 * 2 * 3 * 384 * 65 + 240
    with pytest.raises(ValueError):
        workspace_layout(1, 1, 128, 96)


def test_config_geometry_d64():
    from cudnn.sdpa.bwd.config_sm100_d128_accum import CFG, make_bar_layout, make_config, validate_config

    c64 = make_config(64)
    validate_config(c64, make_bar_layout(c64))
    assert (c64.D, c64.OPERAND_BYTES, c64.DQ_SLICES, c64.DO_SUBTILES, c64.ALIAS_SLICES) == (64, 16384, 2, 1, 2)
    assert c64.SMEM_BYTES < CFG.SMEM_BYTES and make_config(128) == CFG


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "causal,left,sq,sk",
    [
        (False, None, 512, 512),
        (True, None, 512, 512),
        (True, None, 1, 65),
        (False, None, 65, 1),
        (True, 128, 2048, 2048),
        (True, 128, 257, 385),
        (True, 1, 257, 385),
        (True, 129, 300, 300),
        (True, 2, 1, 2),
        (False, 65, 193, 257),
        (False, 300, 257, 513),
    ],
    ids=["dense", "causal", "causal-1x65", "dense-65x1", "w128-2k", "w128-tail", "w1", "w129", "w2-1x2", "dense-w65", "dense-w300"],
)
def test_d64_accumulation(dt, causal, left, sq, sk):
    _case(dt=dt, d=64, causal=causal, left=left, b=2, h=3, sq=sq, sk=sk, accumulation=True, allow_zero=True)


@requires_pre_rubin_blackwell
def test_d128_window_stays_on_recompute():
    _case(dt=torch.bfloat16, d=128, causal=True, left=128, b=1, h=2, sq=257, sk=385, accumulation=None)


@requires_pre_rubin_blackwell
def test_deterministic_d64_stays_on_recompute():
    _case(dt=torch.bfloat16, d=64, causal=True, left=128, b=1, h=2, sq=257, sk=385, deterministic=True, accumulation=None)


@requires_pre_rubin_blackwell
def test_d64_window_rebind_and_replay():
    """Fresh operands through the same plan, then a CUDA Graph capture replayed with changed inputs."""
    case = _case(dt=torch.bfloat16, d=64, causal=True, left=128, b=2, h=3, sq=640, sk=640, accumulation=True)
    for _ in range(2):
        fresh = {name: torch.empty_like(tensor) for name, tensor in case.tensors.items()}
        for name in ("q", "k", "v", "do"):
            fresh[name].normal_()
        o, lse, _, *expected = _reference64(*(fresh[name] for name in ("q", "k", "v", "do")), case.keep)
        fresh["o"].copy_(o)
        fresh["stats"].copy_(lse.unsqueeze(-1))
        for name in ("dq", "dk", "dv"):
            fresh[name].fill_(float("nan"))
        case.tensors, case.expected = fresh, expected
        case.pack = {case.refs[name]: tensor for name, tensor in fresh.items()}
        case.graph.execute(case.pack, case.workspace)
        _check_case(case)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            case.graph.execute(case.pack, case.workspace)
    for name in ("q", "k", "v", "do"):
        case.tensors[name].normal_()
    o, lse, _, *expected = _reference64(*(case.tensors[name] for name in ("q", "k", "v", "do")), case.keep)
    case.tensors["o"].copy_(o)
    case.tensors["stats"].copy_(lse.unsqueeze(-1))
    for name in ("dq", "dk", "dv"):
        case.tensors[name].fill_(float("nan"))
    case.expected = expected
    graph.replay()
    torch.cuda.synchronize()
    _check_case(case)
