# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D256 FP8 gradient/amax repeatability, rebatching, retained capture and refill stress."""

from copy import copy

import pytest
import torch

from d256_fp8_bwd_helpers import AMAX, GRADS, check, execute, graph_plan, make_case, snapshot_inputs, standalone
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("itype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3-in", "e5m2-in"])
@pytest.mark.parametrize(
    "otype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float16, torch.bfloat16], ids=["e4m3-grad", "e5m2-grad", "fp16-grad", "bf16-grad"]
)
def test_backward_replay_batch_capture_and_golden(itype, otype):
    case = make_case(itype, otype, b=3, h=2, hk=1, sq=17, sk=33, causal=True, left=9, seed=91457)
    before = snapshot_inputs(case)
    baseline_run = graph_plan(case, record=(20637, {}))
    baseline = execute(baseline_run)
    check(case, baseline)
    assert_bitwise(execute(standalone(case)), baseline, "graph and standalone window/dtype semantics")

    def prepare(order):
        selected = copy(case)
        selected.b = len(order)
        selected.inputs = {n: torch.cat([v[i : i + 1] for i in order], 0) for n, v in case.inputs.items()}
        return selected, graph_plan(selected, record=baseline_run.record)

    single = [execute(prepare([i])[1]) for i in range(3)]
    for i, result in enumerate(single):
        assert_bitwise({n: result[n] for n in GRADS}, {n: baseline[n][i : i + 1] for n in GRADS}, "singleton")
    for order in ([0, 1, 2], [2, 0, 1], [1, 0, 1, 2], [0, 2, 1] * 43):
        selected, run = prepare(order)
        selected_before = snapshot_inputs(selected)
        got = assert_replays(lambda: execute(run), repeats=3)
        expected = {n: torch.cat([single[i][n] for i in order], 0) for n in GRADS}
        expected.update({n: torch.stack([single[i][n] for i in order]).amax(0) for n in AMAX})
        assert_bitwise(got, expected, "reordered/duplicated fixed quantized payloads")
        assert_bitwise(snapshot_inputs(selected), selected_before, "immutable graph inputs")
        # A neighbor's upstream gradient changes, while the first sequence stays fixed.
        selected.inputs["dO"][1:].view(torch.uint8).bitwise_xor_(128)
        changed = execute(run)
        assert_bitwise({n: changed[n][:1] for n in GRADS}, {n: got[n][:1] for n in GRADS}, "changed neighbor isolation")
        for n in GRADS:
            torch.testing.assert_close(changed[n][1:].float(), -got[n][1:].float(), rtol=0, atol=0)
        assert_bitwise({n: changed[n] for n in AMAX}, {n: got[n] for n in AMAX}, "sign flip preserves global amax")
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            run.launch()
        selected.inputs["dO"].view(torch.uint8).bitwise_xor_(128)
        eager = execute(run)
        for n in GRADS:
            torch.testing.assert_close(eager[n].float(), -changed[n].float(), rtol=0, atol=0)
        for x in run.outputs.values():
            x.fill_(float("nan"))
        run.workspace.fill_(255)
        captured.replay()
        assert_bitwise(snapshot(run.outputs), eager, "changed-input retained CUDA Graph")
        torch.cuda.set_sync_debug_mode("error")
        try:
            run.launch()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert_bitwise(snapshot_inputs(case), before, "source inputs/scales preserved")
    assert_golden(f"sdpa_sm100_d256_fp8_bwd_{itype}_{otype}", before, baseline)


@pytest.mark.parametrize("itype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3-refill", "e5m2-refill"])
def test_backward_refilled_stage_warm_stress(itype):
    from warm_stress import StressCase, assert_warm_stress

    case = make_case(itype, torch.bfloat16, b=16, h=8, hk=4, sq=257, sk=385, seed=7729)
    before = snapshot_inputs(case)
    run = graph_plan(case, record=(20637, {}))
    check(case, execute(run))
    report = assert_warm_stress(
        StressCase(
            name=f"D256 FP8 backward {itype}",
            launch=run.launch,
            outputs=run.outputs,
            thread_blocks=16 * 4 * 4 * 2,
            refills_per_launch=11,
            reset=lambda: run.workspace.fill_(255),
        )
    )
    assert report.failures == 0
    assert_bitwise(snapshot_inputs(case), before, "warm stress immutable inputs")
