# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native D128 FP8 single-CTA dense splits, exposed by the SM107 CI entry."""

from unittest.mock import patch

import pytest
import torch

import cudnn
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
import test_sdpa_prepared_fp8 as shared
from fp8_test_reference import _bound_handle, _check, _execute, _ref, _ref_kwargs


@pytest.mark.L0
def test_dense_split_domain():
    from cudnn.sdpa.fwd import engines
    from test_sdpa_fwd_dsl_sm107 import _caps, _quant_facts

    caps = _caps("sdpa_fwd_prefill_sm107_fp8")
    facts = _quant_facts(b=1, h_q=64, h_kv=8, s_q=1, s_kv=131072)
    for split in (2, 9, 64):
        for cga in (1, 2):
            knobs = engines.SdpaFwdKnobs(cga=cga, split_kv=split, pack_gqa=True)
            assert engines.mismatch(caps, facts, knobs) is None
    assert 1 not in engines.effective_cgas(caps, _quant_facts(thd=True), split_kv=2)
    mx = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    assert 1 not in engines.effective_cgas(mx, _quant_facts(is_mx=True), split_kv=2)


def _case(*, split=3, pack=True, dtype=torch.float8_e4m3fn, output=torch.bfloat16, b=2, sq=17, skv=1024, d=128, dv=128, mask=None, stats=True, sink=False):
    from cudnn.sdpa.fwd import api_dsl

    sinks = torch.linspace(-3, 3, 4, device="cuda").view(1, 4, 1, 1) if sink else None
    g, vp, _, bufs, tensors = shared._case(
        arch="sm107",
        d=d,
        dv=dv,
        b=b,
        sq=sq,
        skv=skv,
        split_kv=split,
        dtype=dtype,
        output_dtype=output,
        sdpa_kwargs=mask,
        stats=stats,
        sink=sinks,
    )
    eid, knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    knobs.update({cudnn.knob_type.TILE_CGA_M: 1, cudnn.knob_type.PACK_GQA: int(pack), cudnn.knob_type.SPLIT_KV: split, cudnn.knob_type.SCHED_POLICY: 0})
    g.create_execution_plan(eid, knobs)
    g.select_plan(g.get_execution_plan_count() - 1)
    g.check_support()
    original = api_dsl._load_kernel_template
    seen = []

    def record(filename, params, tag, part=None):
        module = original(filename, params, tag, part=part)
        seen.append(filename)
        assert filename == "sm107/prefill_d128_fp8.py"
        assert module.CFG.CTA_MMA == 1 and module.CFG.STAGES_KV == 4
        assert module.CFG.TILE_K_HW_BMM1 == module.CFG.TILE_K_HW_BMM2 == 64
        assert module.CFG.SPLIT_KV == split and bool(module.CFG.PACK_GQA) == pack
        return module

    with patch.object(api_dsl, "_load_kernel_template", record):
        g.build_plans()
    assert seen
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared.spec.split == split and prepared.spec.combine is not None
    workspace = torch.empty(max(1, g.get_workspace_size()), device="cuda", dtype=torch.uint8)
    return g, vp, workspace, bufs, tensors


def _check_case(bufs, mask=None):
    in_key = "e4m3" if bufs["q"].dtype == torch.float8_e4m3fn else "e5m2"
    kwargs = _ref_kwargs(mask or {})
    if "sinks" in bufs:
        kwargs["sinks"] = bufs["sinks"].flatten()
    ref, lse = _ref(
        bufs["q"],
        bufs["k"],
        bufs["v"],
        bufs["descale_q"],
        bufs["descale_k"],
        bufs["descale_v"],
        in_key,
        scale=bufs["q"].shape[-1] ** -0.5,
        return_stats=True,
        **kwargs,
    )
    _check(bufs["o"], ref * bufs["scale_o"], bufs["o"].dtype, in_key, bufs["amax_o"].item(), ref.abs().max().item())
    if "lse" in bufs:
        torch.testing.assert_close(bufs["lse"], lse.float(), atol=0.001, rtol=0.001)


_CASES = [
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, 3, True, 17, 1024, 128, 128, {}, True, False, id="e4m3-bf16-pack-split3-dense"),
    pytest.param(
        torch.float8_e5m2, torch.float16, 2, False, 65, 2048, 128, 128, {"use_causal_mask": True}, True, False, id="e5m2-f16-unpacked-split2-causal-refill"
    ),
    pytest.param(
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        9,
        True,
        1,
        512,
        128,
        128,
        {"use_causal_mask_bottom_right": True},
        False,
        False,
        id="e4m3-e4m3-pack-split9-decode-no-stats",
    ),
    pytest.param(
        torch.float8_e5m2,
        torch.float8_e5m2,
        9,
        False,
        257,
        1024,
        128,
        128,
        {"use_causal_mask": True, "left_bound": 65},
        True,
        False,
        id="e5m2-e5m2-unpacked-split9-swa",
    ),
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, 2, True, 17, 2048, 112, 96, {}, True, True, id="e4m3-bf16-pack-split2-envelope-sink"),
    pytest.param(
        torch.float8_e5m2,
        torch.float16,
        3,
        False,
        257,
        512,
        80,
        112,
        {"use_causal_mask_bottom_right": True},
        True,
        False,
        id="e5m2-f16-unpacked-split3-envelope-tail",
    ),
]


@pytest.mark.L0
@pytest.mark.parametrize("dtype,output,split,pack,sq,skv,d,dv,mask,stats,sink", _CASES)
def test_rebind_and_capture(dtype, output, split, pack, sq, skv, d, dv, mask, stats, sink, cudnn_handle):
    g, vp, ws, bufs, tensors = _case(split=split, pack=pack, dtype=dtype, output=output, sq=sq, skv=skv, d=d, dv=dv, mask=mask, stats=stats, sink=sink)
    with _bound_handle(cudnn_handle) as execute:
        for iteration in range(2):
            if iteration:
                bufs = {name: value.clone() for name, value in bufs.items()}
                vp = {tensors[name]: value for name, value in bufs.items()}
                bufs["descale_v"].fill_(0.3)
                bufs["scale_o"].fill_(1.9)
                ws = torch.empty_like(ws)
            execute(g, vp, ws)
            _check_case(bufs, mask)
            with shared._cuda_graph() as capture:
                with torch.cuda.graph(capture):
                    execute(g, vp, ws)
                bufs["v"].copy_((bufs["v"].float() * 0.5).to(dtype))
                for name in ("o", "lse", "amax_o"):
                    if name in bufs:
                        bufs[name].fill_(float("nan"))
                capture.replay()
                _check_case(bufs, mask)


def _batch_runner(dtype, output, split, pack, execute):
    source_case = _case(dtype=dtype, output=output, split=split, pack=pack, b=3, skv=512)
    source = source_case[3]
    inputs = {name: value.clone() for name, value in source.items() if name not in ("o", "lse", "amax_o")}
    plans = {3: source_case}

    def run(order):
        b = len(order)
        if b not in plans:
            plans[b] = _case(dtype=dtype, output=output, split=split, pack=pack, b=b, skv=512)
        graph, variant, workspace, _, ports = plans[b]
        index = torch.tensor(order, device="cuda")
        bufs = {name: value.clone() for name, value in source.items()}
        for name in ("q", "k", "v", "o"):
            bufs[name] = source[name].index_select(0, index).transpose(1, 2).contiguous().transpose(1, 2)
        bufs["lse"] = source["lse"].index_select(0, index)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        execute(graph, {ports[name]: value for name, value in bufs.items()}, torch.empty_like(workspace))
        _check_case(bufs)
        assert_bitwise({name: source[name] for name in inputs}, inputs, "fixed source payloads")
        return {name: bufs[name] for name in ("o", "lse", "amax_o")}

    return run, inputs, source


_BYTE_CASES = [
    pytest.param(torch.float8_e4m3fn, torch.bfloat16, 3, True, id="e4m3-bf16-pack-split3"),
    pytest.param(torch.float8_e5m2, torch.float16, 2, False, id="e5m2-f16-unpacked-split2"),
]


@pytest.mark.L0
@pytest.mark.parametrize("dtype,output,split,pack", _BYTE_CASES)
def test_replay_and_batch_invariance(dtype, output, split, pack, cudnn_handle):
    with _bound_handle(cudnn_handle) as execute:
        run, inputs, source = _batch_runner(dtype, output, split, pack, execute)
        expected = assert_replays(lambda: run([0]), repeats=4)
        target = {name: expected[name] for name in ("o", "lse")}
        for order in ([1, 0, 2], [2, 1, 0], [0, 1, 0, 2], [1] * 64 + [0] + [2] * 64):
            actual = snapshot(run(order))
            for row, sequence in enumerate(order):
                if sequence == 0:
                    assert_bitwise({name: actual[name][row : row + 1] for name in target}, target, f"batch row {row}")
        source["v"][1].copy_((source["v"][1].float() * 0.5).to(dtype))
        inputs["v"][1].copy_(source["v"][1])
        changed = snapshot(run([1, 0, 2]))
        assert_bitwise({name: changed[name][1:2] for name in target}, target, "changed neighbor")


@pytest.mark.L0
@pytest.mark.parametrize("dtype,output,split,pack", _BYTE_CASES)
def test_golden(dtype, output, split, pack, cudnn_handle):
    with _bound_handle(cudnn_handle) as execute:
        run, inputs, _ = _batch_runner(dtype, output, split, pack, execute)
        outputs = assert_replays(lambda: run([0, 1, 2]), repeats=3)
        assert_golden(f"sdpa-fp8-sm107-d128-cga1-{dtype}-{output}-split{split}-pack{int(pack)}", inputs, outputs)


@pytest.mark.L0
def test_warm_refilled_ring(cudnn_handle):
    from warm_stress import StressCase, assert_warm_stress

    g, vp, ws, bufs, _ = _case(split=2, b=128, skv=2048)
    with _bound_handle(cudnn_handle) as execute:
        execute(g, vp, ws)
        _check_case(bufs)
        case = StressCase(
            name="sm107-fp8-d128-cga1-split2",
            launch=lambda: execute(g, vp, ws),
            outputs={"o": bufs["o"].transpose(1, 2), "lse": bufs["lse"], "amax_o": bufs["amax_o"]},
            thread_blocks=128 * 2 * 2,
            refills_per_launch=4,
        )
        assert_warm_stress(case, warm=512, poisoned=512)
