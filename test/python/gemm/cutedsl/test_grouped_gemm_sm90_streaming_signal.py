# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming input signal of the SM90 BF16 grouped GEMM kernels: plain and GLU forward (cooperative and pingpong) and
dGLU backward (cooperative), cluster (2, 1).

Each signal path must give the same bytes as the stock dynamic scheduler: the tile math is the same, only the order
in which tiles run and the readiness waits differ.

L0 keeps one case per behaviour and kernel; the remaining routings, variants and release orders are L1.
"""

import pytest
import torch

from gemm.cutedsl import _streaming_signal_sm90 as sig
from gemm.cutedsl._streaming_signal import signal_ready
from gemm.cutedsl._streaming_signal_sm90 import require_sm90  # noqa: F401  (autouse fixture)
from gemm.cutedsl._streaming_signal_sm90 import Launcher, assert_same_bytes, capacity_of, levels, make_kernel, make_problem, signal_options


def case_id(kind, schedule, *rest):
    return "-".join([sig.KIND_IDS[kind], schedule, *rest])


MATCH_CASES = [
    (kind, schedule, variant, routing)
    for kind, schedule in sig.KIND_SCHEDULES
    for variant in ("signal", "signal_layout_pdl")
    for routing in sorted(sig.ROUTINGS)
]
MATCH_L0 = {
    ("plain", "cooperative", "signal", "many_experts_64"),
    ("plain", "pingpong", "signal_layout_pdl", "big_expert"),
    ("glu", "cooperative", "signal_layout_pdl", "many_experts_64"),
    ("glu", "pingpong", "signal", "many_experts_64"),
    ("dglu", "cooperative", "signal", "many_experts_64"),
    ("dglu", "cooperative", "signal_layout_pdl", "big_expert"),
}


@pytest.mark.parametrize("kind, schedule, variant, routing", levels(MATCH_CASES, MATCH_L0, case_id))
def test_signal_consumer_matches_dynamic(kind, schedule, variant, routing):
    """Every M tile ready: the signal consumer's outputs are the stock dynamic scheduler's, byte for byte."""
    tokens = sig.ROUTINGS[routing]
    problem = make_problem(kind, tokens, seed=len(routing))
    want = Launcher(make_kernel(kind, len(tokens), schedule), problem)(problem)
    kernel = make_kernel(kind, len(tokens), schedule, **signal_options(kind, capacity_of(problem), variant))
    got = Launcher(kernel, problem)(problem, signal_ready(kernel))
    torch.cuda.synchronize()
    assert_same_bytes(got, want, f"{kind} {schedule} {variant} {routing} vs stock")


PRODUCER_ROUTINGS = ("many_experts_64", "big_expert")
PRODUCER_CASES = [
    (kind, schedule, routing, order) for kind, schedule in sig.KIND_SCHEDULES for routing in PRODUCER_ROUTINGS for order in ("in_order", "reverse", "random")
]
PRODUCER_L0 = {
    ("plain", "pingpong", "many_experts_64", "reverse"),
    ("glu", "cooperative", "big_expert", "random"),
    ("dglu", "cooperative", "many_experts_64", "reverse"),
}


@pytest.mark.parametrize("kind, schedule, routing, order", levels(PRODUCER_CASES, PRODUCER_L0, case_id))
def test_signal_consumer_waits_for_producer(kind, schedule, routing, order):
    """The GEMM starts while every signal[] entry reads not-ready and A holds NaN; a producer on another stream writes A
    one M tile at a time, in ``order``, and then marks it ready.  Reading any tile early would carry NaN into the outputs."""
    tokens = sig.ROUTINGS[routing]
    problem = make_problem(kind, tokens, seed=7)
    want = Launcher(make_kernel(kind, len(tokens), schedule), problem)(problem)
    kernel = make_kernel(kind, len(tokens), schedule, streaming_signal_capacity=capacity_of(problem))
    # A few clusters only, so the producer kernels find free SMs while the GEMM waits.
    launcher = Launcher(kernel, problem, max_active_clusters=4)
    got = sig.run_with_producer(launcher, problem, problem["a"].clone(), [order])[order]
    assert_same_bytes(got, want, f"{kind} {schedule} {routing} {order} release vs stock")


@pytest.mark.parametrize(
    "kind, schedule",
    levels([("plain", "cooperative"), ("glu", "pingpong"), ("plain", "pingpong"), ("glu", "cooperative")], {("glu", "pingpong")}, case_id),
)
def test_device_layout_waits_for_producer(kind, schedule):
    """The GEMM starts while the layout reads -1, every signal[] entry reads not-ready and A holds NaN; the producer writes
    the layout (65 entries), then A one M tile at a time with its signal[] entry."""
    tokens = sig.ROUTINGS["many_experts_64"]
    problem = make_problem(kind, tokens, seed=11)
    want = Launcher(make_kernel(kind, len(tokens), schedule), problem)(problem)
    options = signal_options(kind, capacity_of(problem), "signal_layout_pdl")
    launcher = Launcher(make_kernel(kind, len(tokens), schedule, **options), problem, max_active_clusters=4)
    got = sig.run_with_producer(launcher, problem, problem["a"].clone(), ["in_order"], final_layout=problem["layout"].clone())["in_order"]
    assert_same_bytes(got, want, f"{kind} {schedule} device layout vs stock")


NO_ROWS_CASES = [(kind, device_layout) for kind in ("plain", "glu", "dglu") for device_layout in (False, True) if not (kind == "dglu" and device_layout)]


@pytest.mark.parametrize(
    "kind, device_layout",
    levels(NO_ROWS_CASES, {("plain", True), ("dglu", False)}, lambda kind, device_layout: f"{sig.KIND_IDS[kind]}-device_layout_{device_layout}"),
)
def test_signal_consumer_with_no_rows_exits(kind, device_layout):
    """No expert has a row: every cluster ends on its first claim, with or without the producer-written layout."""
    experts = 4
    problem = make_problem(kind, [0] * experts)
    options = dict(streaming_signal_capacity=1, **({"device_layout": True} if device_layout else {}))
    kernel = make_kernel(kind, experts, **options)
    Launcher(kernel, problem)(problem, signal_ready(kernel))
    torch.cuda.synchronize()


@pytest.mark.L0
@pytest.mark.parametrize("schedule", ["cooperative", "pingpong"])
def test_signal_slot_release_warm_stress(schedule):
    """The claim slot each cluster broadcasts is released only after every value read from it is consumed:
    back-to-back and shared-memory-poisoned launches of a shape with many claims per cluster, every byte against stock."""
    from warm_stress import StressCase, assert_warm_stress

    tokens = sig.ROUTINGS["big_expert"]
    problem = make_problem("plain", tokens, seed=13)
    clusters = 4
    want = Launcher(make_kernel("plain", len(tokens), schedule), problem)(problem)
    kernel = make_kernel("plain", len(tokens), schedule, streaming_signal_capacity=capacity_of(problem))
    launcher = Launcher(kernel, problem, max_active_clusters=clusters)
    buffers = launcher.buffers(problem)
    reset = signal_ready(kernel)
    reset(buffers["workspace"])
    assert sig.valid_rows(problem) == problem["rows"], "every row belongs to an expert, so the reference covers the buffers"
    tile_n = sig.SCHEDULES[schedule][0][1]
    claims = capacity_of(problem) * -(-sig.FFN // tile_n)
    assert_warm_stress(
        StressCase(
            f"SM90 streaming signal scheduler, {schedule}, big expert, {clusters} clusters",
            launch=lambda: launcher.launch(problem, buffers),
            outputs={"c": buffers["c"], "d": buffers["d"]},
            thread_blocks=2 * clusters,
            refills_per_launch=claims // clusters,
            reference={"c": want["c"], "d": want["d"]},
            reset=lambda: reset(buffers["workspace"]),
        ),
        warm=500,
        poisoned=500,
        budget_s=30.0,
    )


@pytest.mark.L0
def test_streaming_options_reject_invalid():
    import cutlass
    from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode

    for kind in ("plain", "glu"):
        with pytest.raises(ValueError, match="device_layout"):
            make_kernel(kind, 4, device_layout=True)
    for kind in sig.KINDS:
        # Without a producer the host launch puts helper_kernel (the claim counter reset) right before the GEMM, which
        # waits for its programmatic primary only at exit: use_pdl is rejected there and kept with streaming_signal_capacity.
        with pytest.raises(ValueError, match="helper_kernel"):
            make_kernel(kind, 4, use_pdl=True)
        make_kernel(kind, 4, streaming_signal_capacity=4, use_pdl=True)
    for kind in sig.KINDS:
        common = dict(acc_dtype=cutlass.Float32, use_2cta_instrs=False, vectorized_f32=False, expert_cnt=4, weight_mode=MoEWeightMode.DENSE)
        common.update(use_dynamic_sched=True, streaming_signal_capacity=4)
        extra = dict(act_func="dswiglu") if kind == "dglu" else dict(generate_c=True, enable_bias=False)
        if kind == "glu":
            extra["act_func"] = "swiglu"
        cls = sig.kernel_class(kind)
        # One signal[] entry is the cluster's two 128-row CTA tiles: other clusters and tile M are rejected.
        for tile, cluster in (((128, 128), (1, 1)), ((128, 128), (1, 2)), ((128, 128), (2, 2))):
            with pytest.raises(ValueError, match="cluster"):
                cls(mma_tiler_mn=tile, cluster_shape_mn=cluster, **common, **extra)
        if kind != "dglu":
            with pytest.raises(ValueError, match="CTA tile M 128"):
                cls(mma_tiler_mn=(64, 128), cluster_shape_mn=(2, 1), **common, **extra)
    # A valid one builds, and the signal workspace follows the claim counter.
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import streaming_signal_workspace_offsets

    kernel = make_kernel("plain", 4, "pingpong", streaming_signal_capacity=8)
    offsets = streaming_signal_workspace_offsets(kernel)
    assert offsets["counter"] == 0 and offsets["signal"] == 4 and offsets["state"] % 128 == 0
    assert make_kernel("plain", 4).get_workspace_bytes() == 4
