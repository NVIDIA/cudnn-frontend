# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""FP8 page addressing: fixed-payload repeatability and batch invariance."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
import test_sdpa_fwd_paged_sm100 as paged
import test_sdpa_prepared_fp8_paged as prepared

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]

_CASES = [
    pytest.param(1, False, "e4m3", torch.bfloat16, id="fwd-d512-e4m3-bf16-split1-NHD"),
    pytest.param(4, True, "e4m3", torch.float8_e4m3fn, id="fwd-d512-e4m3-e4m3-split4-HND"),
    pytest.param(1, True, "e5m2", torch.float16, id="fwd-d512-e5m2-f16-split1-HND"),
    pytest.param(4, False, "e5m2", torch.float8_e5m2, id="fwd-d512-e5m2-e5m2-split4-NHD"),
]


def _runner(*, split, hnd, in_key, dtype, softmax_precision=None, d=512, page=16, cga=None):
    # Quantize once for all batch arrangements. No requantization or scale
    # recomputation is allowed when the target sequence moves or is duplicated.
    g, vp, ws, source, tensors = paged._run_graph_fp8(
        5,
        4,
        2,
        d,
        page,
        256 // page,
        [256] * 5,
        hnd,
        s_q=4,
        in_key=in_key,
        out_dt=dtype,
        explicit_split=split,
        explicit_pack=True,
        explicit_cga=cga,
        softmax_precision=softmax_precision,
        override=True,
        return_case=True,
    )
    source["v_table"] = source["k_table"].flip(2).clone()
    before = {name: value.clone() for name, value in source.items() if name not in ("o", "lse", "amax_o")}
    plans = {5: (g, vp, ws, tensors)}

    def run(order):
        b = len(order)
        if b not in plans:
            graph, variant, workspace, _, ports = paged._run_graph_fp8(
                b,
                4,
                2,
                d,
                page,
                256 // page,
                [256] * b,
                hnd,
                s_q=4,
                in_key=in_key,
                out_dt=dtype,
                explicit_split=split,
                explicit_pack=True,
                explicit_cga=cga,
                softmax_precision=softmax_precision,
                override=True,
                return_case=True,
            )
            plans[b] = graph, variant, workspace, ports
        graph, variant, workspace, ports = plans[b]
        owner = graph._compiled_plans[graph._plan_index]._prepared.spec.owner
        index = torch.tensor(order, device="cuda")
        bufs = {name: value.clone() for name, value in source.items()}
        for name in ("q", "o", "lse", "k_table", "v_table", "seq_q", "seq_kv"):
            bufs[name] = source[name].index_select(0, index)
        # Retain the BSHD storage required by whole-head PackGQA.
        bufs["q"] = bufs["q"].transpose(1, 2).contiguous().transpose(1, 2)
        bufs["o"] = bufs["o"].transpose(1, 2).contiguous().transpose(1, 2)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        bufs["amax_o"].fill_(999)
        variant_pack = dict(variant)
        variant_pack.update({ports[name]: value for name, value in bufs.items()})
        inputs = {name: value.clone() for name, value in bufs.items() if name not in ("o", "lse", "amax_o")}
        graph.execute(variant_pack, torch.empty_like(workspace), **prepared._overrides(bufs, ports, bufs))
        prepared._check(bufs)
        assert_bitwise({name: bufs[name] for name in inputs}, inputs, "input bytes")
        assert_bitwise({name: source[name] for name in before}, before, "source bytes")
        assert graph._compiled_plans[graph._plan_index]._prepared.spec.owner is owner
        return {name: bufs[name] for name in ("o", "lse", "amax_o")}

    return run, before


def _check_replay_and_batch_invariance(run):
    # Includes the scalar Amax on repeated calls, with fresh device buffers and
    # queued side-stream work. Amax is a whole-batch reduction, not a per-row result.
    expected = assert_replays(lambda: run([0]), repeats=4)
    target = {name: expected[name] for name in ("o", "lse")}
    # The last arrangement spans multiple resident-cluster waves on SM100/103.
    for order in ([1, 0, 2], [2, 1, 0], [0, 1, 0, 2, 3], [4, 0, 3], [1] * 64 + [0] + [2] * 64):
        actual = snapshot(run(order))
        for row, sequence in enumerate(order):
            if sequence == 0:
                assert_bitwise({name: actual[name][row : row + 1] for name in target}, target, f"batch {order}, row {row}")


@pytest.mark.parametrize("split,hnd,in_key,dtype", _CASES)
def test_fp8_d512_paged_replay_and_batch_invariance(split, hnd, in_key, dtype):
    run, _ = _runner(split=split, hnd=hnd, in_key=in_key, dtype=dtype)
    _check_replay_and_batch_invariance(run)


@pytest.mark.parametrize("split,hnd,in_key,dtype", _CASES)
def test_fp8_d512_paged_golden(split, hnd, in_key, dtype):
    run, inputs = _runner(split=split, hnd=hnd, in_key=in_key, dtype=dtype)
    actual = assert_replays(lambda: run([0, 1, 2]), repeats=3)
    assert_golden(f"sdpa-fp8-d512-paged-{in_key}-{dtype}-split{split}", inputs, actual)


_SMALL_CASES = [
    pytest.param(64, 8, 1, 1, False, "e4m3", torch.bfloat16, id="fwd-d64-page8-cga1-split1-e4m3-bf16-NHD"),
    pytest.param(128, 16, 1, 2, True, "e4m3", torch.float8_e4m3fn, id="fwd-d128-page16-cga1-split2-e4m3-e4m3-HND"),
    pytest.param(128, 8, 2, 1, True, "e5m2", torch.float16, id="fwd-d128-page8-cga2-split1-e5m2-f16-HND"),
    pytest.param(64, 64, 1, 4, False, "e5m2", torch.float8_e5m2, id="fwd-d64-page64-cga1-split4-e5m2-e5m2-NHD"),
    pytest.param(128, 128, 1, 4, False, "e4m3", torch.bfloat16, id="fwd-d128-page128-cga1-split4-e4m3-bf16-NHD"),
]


@pytest.mark.parametrize("d,page,cga,split,hnd,in_key,dtype", _SMALL_CASES)
def test_fp8_small_paged_replay_and_batch_invariance(d, page, cga, split, hnd, in_key, dtype):
    run, _ = _runner(d=d, page=page, cga=cga, split=split, hnd=hnd, in_key=in_key, dtype=dtype)
    _check_replay_and_batch_invariance(run)


@pytest.mark.parametrize("d,page,cga,split,hnd,in_key,dtype", _SMALL_CASES)
def test_fp8_small_paged_golden(d, page, cga, split, hnd, in_key, dtype):
    run, inputs = _runner(d=d, page=page, cga=cga, split=split, hnd=hnd, in_key=in_key, dtype=dtype)
    actual = assert_replays(lambda: run([0, 1, 2]), repeats=3)
    assert_golden(f"sdpa-fp8-d{d}-paged-page{page}-cga{cga}-{in_key}-{dtype}-split{split}", inputs, actual)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
def test_fp8_paged_loader_warm_stress():
    from warm_stress import StressCase, assert_warm_stress

    g, vp, ws, bufs, _ = paged._run_graph_fp8(
        2, 4, 2, 128, 8, 128, [1024, 1024], True, s_q=4, explicit_cga=1, explicit_pack=True, explicit_split=1, return_case=True
    )
    # Eight KV tiles traverse the four-slot ring twice within each work item.
    assert_warm_stress(
        StressCase(
            "SM100 D128 FP8 coalesced page table",
            lambda: g.execute(vp, ws),
            {"o": bufs["o"].transpose(1, 2), "stats": bufs["lse"], "amax_o": bufs["amax_o"]},
            thread_blocks=4,
            refills_per_launch=4,
        ),
        warm=256,
        poisoned=256,
    )
