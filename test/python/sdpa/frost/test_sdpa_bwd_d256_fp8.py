# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100/SM103 D256 FP8 backward, including independent references and replay."""

import pytest
import torch

from d256_fp8_bwd_helpers import check, execute, graph_plan, make_case, snapshot_inputs, standalone
from determinism_utils import assert_bitwise
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]

INPUTS = [torch.float8_e4m3fn, torch.float8_e5m2]
OUTPUTS = [torch.float8_e4m3fn, torch.float8_e5m2, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("itype", INPUTS, ids=["e4m3-in", "e5m2-in"])
@pytest.mark.parametrize("otype", OUTPUTS, ids=["e4m3-grad", "e5m2-grad", "fp16-grad", "bf16-grad"])
@pytest.mark.parametrize(
    "shape",
    [(17, 33, 2, 1, False, None), (129, 257, 2, 1, True, 127), (257, 129, 1, 1, False, 1)],
    ids=["gqa-q17-k33", "gqa-q129-k257-causal-window", "mha-q257-k129-empty-tail"],
)
def test_standalone_reference_and_repeat(itype, otype, shape):
    sq, sk, h, hk, causal, left = shape
    case = make_case(itype, otype, sq=sq, sk=sk, h=h, hk=hk, causal=causal, left=left)
    before = snapshot_inputs(case)
    run = standalone(case)
    first = execute(run)
    check(case, first)
    assert_bitwise(execute(run), first, "D256 FP8 fresh output poison")
    assert_bitwise(snapshot_inputs(case), before, "D256 FP8 immutable inputs")


@pytest.mark.parametrize("itype", INPUTS, ids=["e4m3-in", "e5m2-in"])
@pytest.mark.parametrize("requested", [(), ("amax_dP",), ("amax_dQ", "amax_dV")], ids=["no-amax", "dP-amax", "dQ-dV-amax"])
def test_optional_amax(itype, requested):
    case = make_case(itype, torch.bfloat16)
    run = standalone(case, requested)
    got = execute(run)
    assert set(got) == {"dQ", "dK", "dV", *requested}
    check(case, got)


def test_input_dtype_specializations_coexist():
    cases = [make_case(t, torch.bfloat16) for t in INPUTS]
    runs = [standalone(c) for c in cases]
    first = [execute(r) for r in runs]
    for c, result in zip(cases, first):
        check(c, result)
    assert not torch.equal(first[0]["dQ"], first[1]["dQ"])
    for i in (1, 0, 1, 0):
        assert_bitwise(execute(runs[i]), first[i], "E4M3/E5M2 specialization remains bound")


@pytest.mark.parametrize("flag", [None, "0", "1"], ids=["unset", "disabled", "enabled"])
@pytest.mark.parametrize("itype", INPUTS, ids=["e4m3-in", "e5m2-in"])
def test_issue2672_default_plan_and_record(monkeypatch, flag, itype):
    if flag is None:
        monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    else:
        monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", flag)
    # The reported B2 H8 S1024 geometry, with a normal public plan walk.
    case = make_case(itype, torch.bfloat16, b=2, h=8, hk=8, sq=1024, sk=1024)
    run = graph_plan(case)
    first = execute(run)
    check(case, first)
    replayed = graph_plan(case, record=run.record)
    assert_bitwise(execute(replayed), first, "explicit public record reproduces the default plan")
