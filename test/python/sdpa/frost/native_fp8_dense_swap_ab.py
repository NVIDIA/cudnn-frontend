# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense FP8 decode through the public SWAP_AB knob on SM100/103/107.

Forward only: this implementation choice adds no backward path. Every plan
keeps its packing and split count fixed during byte comparisons. AMAX is a
whole-batch reduction; compare it across replay, and O/Stats across batches.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import cudnn
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
import test_sdpa_prepared_fp8 as shared
from fp8_test_reference import _bound_handle, _check, _ref, _ref_kwargs


def _arch():
    return "sm107" if torch.cuda.get_device_capability() == (10, 7) else "sm100"


def _outputs(bufs):
    return {name: bufs[name] for name in ("o", "lse", "amax_o") if name in bufs}


def _poison(bufs):
    for value in _outputs(bufs).values():
        value.fill_(float("nan"))


def _fresh(bufs):
    """Fresh storage with the same physical row pitches, including canaries."""
    fresh = {}
    for name, value in bufs.items():
        if name == "o" and "o_storage" in bufs:
            continue
        fresh[name] = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
        fresh[name].copy_(value)
    if "o_storage" in bufs:
        fresh["o"] = fresh["o_storage"][..., : bufs["o"].shape[-1]].transpose(1, 2)
    return fresh


def _variant(bufs, tensors):
    return {tensor: bufs[name] for name, tensor in tensors.items()}


def _case(
    *,
    dtype=torch.float8_e4m3fn,
    output=torch.bfloat16,
    split=3,
    pack=True,
    b=3,
    sq=2,
    skv=1024,
    group=4,
    d=128,
    dv=128,
    mask=None,
    stats=True,
    amax=True,
    sink=False,
    padded=False,
    lengths=False,
    swap=1,
):
    from cudnn.sdpa.fwd import api_dsl

    heads = (2 * group, 2)
    sinks = torch.linspace(-3, 3, heads[0], device="cuda").view(1, heads[0], 1, 1) if sink else None
    seq_lens = ([sq, max(sq - 1, 0), 0], [skv, 0, skv // 2]) if lengths else None
    assert not lengths or b == 3
    g, _, _, bufs, tensors = shared._case(
        arch=_arch(),
        dtype=dtype,
        output_dtype=output,
        split_kv=split,
        b=b,
        sq=sq,
        skv=skv,
        d=d,
        dv=dv,
        heads=heads,
        sdpa_kwargs=mask,
        stats=stats,
        amax=amax,
        sink=sinks,
        padded=padded,
        output_padding=16 if padded else 0,
        seq_lens=seq_lens,
    )
    # CUDA RNG depends on SM count. Quantize once on CPU for identical source
    # payloads on every GPU and retain those bytes under rebatching/replay.
    rng = torch.Generator().manual_seed(83520)
    for name in ("q", "k", "v"):
        value = bufs[name]
        value.copy_((torch.randn(value.shape, generator=rng) * 0.3).to(dtype))
    eid, knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    knobs.update({cudnn.knob_type.TILE_CGA_M: 1, cudnn.knob_type.PACK_GQA: int(pack), cudnn.knob_type.SPLIT_KV: split, cudnn.knob_type.SCHED_POLICY: 0})
    knobs.pop(cudnn.knob_type.SWAP_AB, None)
    if swap is not None:
        knobs[cudnn.knob_type.SWAP_AB] = swap
    g.create_execution_plan(eid, knobs)
    g.select_plan(g.get_execution_plan_count() - 1)
    g.check_support()
    original = api_dsl._load_kernel_template
    seen = []

    def record(filename, params, tag, part=None):
        module = original(filename, params, tag, part=part)
        seen.append(filename)
        expected = "sm100/decode_d128_fp8.py" if swap == 1 else _arch() + "/prefill_d128_fp8.py"
        assert filename == expected
        assert params.cta_mma == 1 and module.CFG.SPLIT_KV == split
        assert bool(module.CFG.PACK_GQA) == pack
        if swap == 1:
            assert module.CFG.STAGES_KV == 5 and module.COLS == 8
            assert module.CFG.TILE_K_HW == 32
            assert not params.paged_kv and not params.thd_varlen
        return module

    with patch.object(api_dsl, "_load_kernel_template", record):
        g.build_plans()
    assert seen
    actual_eid, actual_knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    assert actual_eid == eid and actual_knobs.get(cudnn.knob_type.SWAP_AB) == swap
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared.spec.split == split
    assert (prepared.spec.combine is not None) == (split > 1)
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return g, _variant(bufs, tensors), ws, bufs, tensors


def _reference(bufs, mask=None, *, log2=False):
    in_key = "e4m3" if bufs["q"].dtype == torch.float8_e4m3fn else "e5m2"
    kwargs = _ref_kwargs(mask or {})
    if "sinks" in bufs:
        kwargs["sinks"] = bufs["sinks"].flatten()
    if "seq_q" in bufs:
        kwargs.update(seq_lens_q=bufs["seq_q"].flatten(), seq_lens_kv=bufs["seq_kv"].flatten())
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
    am = ref.abs().max().item()
    _check(bufs["o"], ref * bufs["scale_o"], bufs["o"].dtype, in_key, bufs["amax_o"].item() if "amax_o" in bufs else am, am)
    assert torch.isfinite(bufs["o"].float()).all()
    if "lse" in bufs:
        torch.testing.assert_close(bufs["lse"], lse.float() * (1.4426950408889634 if log2 else 1), atol=0.001, rtol=0.001)
    if "o_storage" in bufs:
        assert torch.all(bufs["o_storage"][..., bufs["o"].shape[-1] :] == 12), "output row padding overwritten"
    return _outputs(bufs)


_CASES = [
    pytest.param(dict(dtype=torch.float8_e4m3fn, output=torch.bfloat16, split=1, sq=1, group=8), id="e4-bf16-q1-g8-unsplit"),
    pytest.param(dict(dtype=torch.float8_e5m2, output=torch.float16, split=2, mask=dict(use_causal_mask=True)), id="e5-f16-q2-g4-split2-causal"),
    pytest.param(
        dict(output=torch.float8_e4m3fn, split=9, sq=1, group=8, stats=False, mask=dict(use_causal_mask_bottom_right=True)), id="e4-e4-q1-g8-split9-br-no-stats"
    ),
    pytest.param(
        dict(dtype=torch.float8_e5m2, output=torch.float8_e5m2, split=3, pack=False, sq=8, mask=dict(use_causal_mask_bottom_right=True, left_bound=393)),
        id="e5-e5-q8-unpacked-split3-window",
    ),
    pytest.param(dict(split=2, sq=4, group=2, d=112, dv=96, sink=True, padded=True), id="e4-bf16-d112-v96-sink-strides"),
    pytest.param(
        dict(dtype=torch.float8_e5m2, output=torch.float16, split=1, pack=False, sq=5, group=1, d=80, dv=112, padded=True),
        id="e5-f16-d80-v112-unpacked-strides",
    ),
    pytest.param(dict(lengths=True, padded=True, skv=1025, split=1), id="e4-bf16-empty-q-kv-strides"),
    pytest.param(dict(stats=False, amax=False, split=1), id="e4-bf16-unsplit-no-stats-amax"),
    pytest.param(dict(stats=True, amax=False, split=3), id="e4-bf16-split3-no-amax"),
]


@pytest.mark.L0
@pytest.mark.parametrize("options", _CASES)
def test_rebind_capture(options, cudnn_handle):
    g, vp, ws, bufs, tensors = _case(**options)
    with _bound_handle(cudnn_handle) as execute:
        for iteration in range(2):
            if iteration:
                bufs = _fresh(bufs)
                vp = _variant(bufs, tensors)
                bufs["descale_v"].fill_(0.3)
                bufs["scale_o"].fill_(1.9)
                ws = torch.empty_like(ws)
            inputs = snapshot({name: value for name, value in bufs.items() if name not in (*_outputs(bufs), "o_storage")})
            _poison(bufs)
            execute(g, vp, ws)
            baseline = snapshot(_reference(bufs, options.get("mask")))
            with shared._cuda_graph() as capture:
                with torch.cuda.graph(capture):
                    execute(g, vp, ws)
                saved = bufs["v"].clone()
                bufs["v"].copy_((-saved.float()).to(saved.dtype))
                _poison(bufs)
                ws.fill_(255)
                capture.replay()
                changed = snapshot(_reference(bufs, options.get("mask")))
                assert not torch.equal(changed["o"].view(torch.uint8), baseline["o"].view(torch.uint8))
                bufs["v"].copy_(saved)
                _poison(bufs)
                capture.replay()
                assert_bitwise(_reference(bufs, options.get("mask")), baseline, "restored capture")
            assert_bitwise({name: bufs[name] for name in inputs}, inputs, "input mutation")


@pytest.mark.L0
@pytest.mark.parametrize("swap", [None, 0, 1])
def test_public_choice(swap, cudnn_handle):
    g, vp, ws, bufs, _ = _case(swap=swap)
    with _bound_handle(cudnn_handle) as execute:
        execute(g, vp, ws)
    _reference(bufs)


@pytest.mark.L0
@pytest.mark.parametrize("split", [1, 3])
@pytest.mark.parametrize("log2", [False, True], ids=["ln", "log2"])
def test_standalone_current_lengths_and_logbase(split, log2):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    # D128 padding+split remains outside the shared graph/standalone contract.
    # Check current lengths unsplit, and natural-log partials with either final
    # Stats base on unpadded split graphs.
    lengths = split == 1
    _, _, _, bufs, _ = _case(split=split, lengths=lengths, padded=True, skv=1025 if lengths else 1024)
    api = SdpaFwdDslSm100(
        bufs["q"],
        bufs["k"],
        bufs["v"],
        bufs["o"],
        bufs["lse"],
        seq_q_lens_present=lengths,
        seq_kv_lens_present=lengths,
        pertensor_fp8=True,
        dtype_o=bufs["o"].dtype,
        stats_log2=log2,
        cga=1,
        pack_gqa=True,
        split_kv=split,
        swap_ab=1,
    )
    api.check_support()
    api.compile()
    assert api._k_mod.__file__.endswith("sm100/decode_d128_fp8.py")

    for iteration in range(2):
        if iteration:
            bufs = _fresh(bufs)
            bufs["descale_q"].fill_(0.5)
            bufs["descale_k"].fill_(1.25)
            bufs["descale_v"].fill_(0.3)
            bufs["scale_o"].fill_(1.9)
        ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)

        def launch():
            api.execute(
                bufs["q"],
                bufs["k"],
                bufs["v"],
                bufs["o"],
                bufs["lse"],
                seq_q_lens=bufs.get("seq_q"),
                seq_kv_lens=bufs.get("seq_kv"),
                workspace=ws,
                **{name: bufs[name] for name in ("descale_q", "descale_k", "descale_v", "scale_o", "amax_o")},
            )

        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            launch()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        expected = snapshot(_reference(bufs, log2=log2))
        with shared._cuda_graph() as capture:
            with torch.cuda.graph(capture):
                launch()
            saved = snapshot({name: bufs[name] for name in (("seq_q", "seq_kv") if lengths else ("v",))})
            if lengths:
                bufs["seq_q"].fill_(2)
                bufs["seq_kv"].zero_()
            else:
                bufs["v"].copy_((-bufs["v"].float()).to(bufs["v"].dtype))
            _poison(bufs)
            capture.replay()
            _reference(bufs, log2=log2)
            if lengths:
                assert torch.count_nonzero(bufs["o"].float()) == 0 and bufs["amax_o"].item() == 0
                assert torch.isneginf(bufs["lse"]).all()
            for name, value in saved.items():
                bufs[name].copy_(value)
            _poison(bufs)
            capture.replay()
            assert_bitwise(_reference(bufs, log2=log2), expected, "restored lengths")


@pytest.mark.L0
@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)])
def test_domain(cc):
    from cudnn.sdpa.fwd import engines
    from test_sdpa_fwd_dsl_sm107 import _caps, _quant_facts

    caps = _caps("sdpa_fwd_prefill_" + ("sm107" if cc == (10, 7) else "sm100") + "_fp8")
    facts = _quant_facts(device_cc=cc, b=1, h_q=16, h_kv=2, s_q=1, s_kv=1024)
    knobs = engines.SdpaFwdKnobs(cga=1, split_kv=3, pack_gqa=True, swap_ab=1)
    for valid, choice in (
        (facts, knobs),
        # Keep Rubin's independent unsplit working-set limit satisfied while
        # testing the decode grid's batch endpoint, without GPU allocation.
        (replace(facts, b=65535, h_q=8, h_kv=1, s_kv=1), replace(knobs, split_kv=1)),
        (replace(facts, b=21845), knobs),
        (replace(facts, d_qk=80, d_v=112), knobs),
    ):
        assert engines.effective_swap_abs(caps, valid, choice) == frozenset({0, 1})
        reason = engines.mismatch(caps, valid, choice)
        assert reason is None, reason
    assert engines.mismatch(caps, replace(facts, b=21846), knobs) is not None, "retain the common batch-times-split grid limit"
    for change in (
        dict(b=65536),
        dict(b=0),
        dict(s_q=2),
        dict(thd=True),
        dict(has_paged_kv=True),
        dict(d_qk=192),
        dict(device_cc=(12, 0)),
        dict(h_q=32),
        dict(has_epilogue_gate=True),
        dict(has_bias=True),
        dict(has_score_max=True),
        dict(has_score_sum_exp=True),
        dict(has_kv_mask=True),
        dict(o_block_scale=True),
        dict(attn_scale_prefolded=True),
    ):
        assert engines.effective_swap_abs(caps, replace(facts, **change), knobs) == frozenset(), change
    for change in (dict(cga=2), dict(cga=None), dict(tile_n=64)):
        assert engines.effective_swap_abs(caps, facts, replace(knobs, **change)) == frozenset(), change
    assert engines.mismatch(caps, replace(facts, h_q=6), knobs) is not None, "G3 remains outside common packing contract"
    assert engines.mismatch(caps, replace(facts, padded=True), knobs) is not None, "D128 padded splits retain their common decline"


@pytest.mark.L0
def test_invalid_plan_then_recovery(cudnn_handle):
    g, vp, ws, bufs, _ = _case()
    valid_index = g._plan_index
    eid, valid = g.get_engine_and_knobs_at_index(valid_index)
    for value in _outputs(bufs).values():
        value.fill_(11)
    ws.fill_(173)
    before = snapshot(dict(_outputs(bufs), workspace=ws))
    for change, reason in (({cudnn.knob_type.TILE_CGA_M: 2}, "swap_ab"), ({cudnn.knob_type.TILE_N: 64}, "tile_n=64")):
        with pytest.raises((ValueError, NotImplementedError, RuntimeError), match="(?i)" + reason):
            g.create_execution_plan(eid, {**valid, **change})
            g.select_plan(g.get_execution_plan_count() - 1)
            g.check_support()
            g.build_plans()
        assert_bitwise(dict(_outputs(bufs), workspace=ws), before, "rejected plan wrote output/workspace")
    g.select_plan(valid_index)
    with _bound_handle(cudnn_handle) as execute:
        execute(g, vp, ws)
    _reference(bufs)


_BYTE_CASES = [
    pytest.param(torch.float8_e4m3fn, torch.float8_e4m3fn, 1, True, id="e4-e4-packed-unsplit"),
    pytest.param(torch.float8_e5m2, torch.bfloat16, 3, True, id="e5-bf16-packed-split3"),
    pytest.param(torch.float8_e4m3fn, torch.float16, 3, False, id="e4-f16-unpacked-split3"),
    pytest.param(torch.float8_e5m2, torch.float8_e5m2, 1, False, id="e5-e5-unpacked-unsplit"),
]


def _batch_runner(dtype, output, split, pack, execute):
    options = dict(dtype=dtype, output=output, split=split, pack=pack, skv=512)
    source_case = _case(**options)
    source = source_case[3]
    inputs = snapshot({name: value for name, value in source.items() if name not in _outputs(source)})
    plans = {3: source_case}

    def run(order):
        b = len(order)
        if b not in plans:
            plans[b] = _case(b=b, **options)
        g, _, ws, _, ports = plans[b]
        index = torch.tensor(order, device="cuda")
        bufs = {name: value.clone() for name, value in source.items()}
        for name in ("q", "k", "v", "o"):
            bufs[name] = source[name].index_select(0, index).transpose(1, 2).contiguous().transpose(1, 2)
        bufs["lse"] = source["lse"].index_select(0, index)
        _poison(bufs)
        current_inputs = snapshot({name: value for name, value in bufs.items() if name not in _outputs(bufs)})
        execute(g, _variant(bufs, ports), torch.empty_like(ws))
        result = _reference(bufs)
        assert_bitwise({name: bufs[name] for name in current_inputs}, current_inputs, "current bindings mutated")
        assert_bitwise({name: source[name] for name in inputs}, inputs, "fixed source mutated")
        return result

    return run, inputs, source


@pytest.mark.L0
@pytest.mark.parametrize("dtype,output,split,pack", _BYTE_CASES)
def test_replay_batch(dtype, output, split, pack, cudnn_handle):
    with _bound_handle(cudnn_handle) as execute:
        run, inputs, source = _batch_runner(dtype, output, split, pack, execute)
        target = assert_replays(lambda: run([0]), repeats=4)
        concurrent = torch.cuda.Stream()
        noise = torch.zeros(16384, device="cuda")
        concurrent.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(concurrent):
            for _ in range(32):
                noise.add_(1)
        # 514 packed CTAs cross scheduling waves on the target GPUs, while
        # the unsplit fixture stays below Rubin's existing 96 MiB KV limit.
        b = 128
        for order in ([1, 0, 2], [2, 1, 0], [0, 1, 0, 2], [1] * b + [0] + [2] * b):
            actual = snapshot(run(order))
            for row, sequence in enumerate(order):
                if sequence == 0:
                    assert_bitwise(
                        {name: actual[name][row : row + 1] for name in ("o", "lse")}, {name: target[name] for name in ("o", "lse")}, f"batch row {row}"
                    )
        torch.cuda.current_stream().wait_stream(concurrent)
        source["v"][1].copy_((-source["v"][1].float()).to(dtype))
        inputs["v"][1].copy_(source["v"][1])
        changed = snapshot(run([1, 0, 2]))
        assert_bitwise({name: changed[name][1:2] for name in ("o", "lse")}, {name: target[name] for name in ("o", "lse")}, "changed neighbor")


@pytest.mark.L0
@pytest.mark.parametrize("dtype,output,split,pack", _BYTE_CASES)
def test_golden(dtype, output, split, pack, cudnn_handle):
    with _bound_handle(cudnn_handle) as execute:
        run, inputs, _ = _batch_runner(dtype, output, split, pack, execute)
        outputs = assert_replays(lambda: run([0, 1, 2]), repeats=3)
        assert_golden(f"sdpa-fp8-dense-swap-{dtype}-{output}-split{split}-pack{int(pack)}", inputs, outputs)


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"])
def test_warm_refill(dtype, cudnn_handle):
    from warm_stress import StressCase, assert_warm_stress

    b = 128
    with _bound_handle(cudnn_handle) as execute:
        g, vp, ws, bufs, _ = _case(dtype=dtype, b=b, sq=1, group=8, pack=False, split=1, skv=1025)
        execute(g, vp, ws)
        _reference(bufs)
        assert_warm_stress(
            StressCase(
                name=f"dense FP8 eight-column decode {dtype}",
                launch=lambda: execute(g, vp, ws),
                outputs={"o": bufs["o"].transpose(1, 2), "lse": bufs["lse"], "amax_o": bufs["amax_o"]},
                thread_blocks=b * 16,
                refills_per_launch=13,
            ),
            warm=256,
            poisoned=256,
        )
