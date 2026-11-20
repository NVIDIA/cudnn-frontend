# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming signal options of the BF16 grouped GEMM kernels (plain and GLU).

Each signal path must give the same bits as the stock dynamic scheduler: the tile math is the same, only the order
in which tiles run and the readiness waits differ.

L0 keeps one case per behaviour, chosen to reuse the kernels the determinism suite compiles; the remaining routings,
release orders and chunk shapes are L1.
"""

import pytest
import torch

from gemm.cutedsl._streaming_signal import BIG_EXPERT, MANY_EXPERTS_64, ROWS_ALIGNMENT, compile_once, kernel_args, launch_args
from gemm.cutedsl._streaming_signal import make_kernel as build_kernel
from gemm.cutedsl._streaming_signal import make_problem, new_buffers, signal_ready

EXPERTS = 4
HIDDEN = 512
FFN = 512


@pytest.fixture(autouse=True)
def require_sm10x():
    """Every test here: an SM10x GPU (compute capability 10.x) with tcgen05, and a CuTe DSL that is new enough and
    targets the device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old
    from cudnn.tensor_adapter import has_tcgen05

    cc = torch.cuda.get_device_capability()
    if cc[0] != 10 or not has_tcgen05(cc[0]):
        pytest.skip("an SM10x GPU (compute capability 10.x) is required")
    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip(f"the CuTe DSL is older than the library floor: {cutedsl_state()[1]}")
    reason = cutedsl_arch_requirement_error(cc)
    if reason is not None:
        pytest.skip(reason)


def levels(cases, l0, ids):
    """``cases`` as pytest params: L0 for those in ``l0``, L1 for the rest."""
    return [pytest.param(*case, marks=pytest.mark.L0 if case in l0 else pytest.mark.L1, id=ids(*case)) for case in cases]


def kernel_classes():
    from cudnn.gemm.cutedsl.grouped.unfused.moe_grouped_gemm import MoEGroupedGemmBf16Kernel
    from cudnn.gemm.cutedsl.grouped.glu.moe_grouped_gemm_glu_bias import MoEGroupedGemmGluBiasBf16Kernel

    return MoEGroupedGemmBf16Kernel, MoEGroupedGemmGluBiasBf16Kernel


def make_kernel(glu, one_cta=False, experts=EXPERTS, **options):
    return build_kernel(glu, experts, one_cta, **options)


def run(kernel, problem, glu, prepare_workspace=None, max_active_clusters=None, stream=None, defer_launch=False):
    """Compile ``kernel`` on ``problem`` (once per construction and layout) and launch it, or with ``defer_launch`` leave
    the launch to the caller; returns (C, D, workspace, launch), ``launch`` None once launched.  ``prepare_workspace(ws)``
    runs on the launch stream first."""
    import cuda.bindings.driver as cuda

    buffers = new_buffers(kernel, problem, glu)
    c, d, workspace = buffers
    stream = stream or torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if prepare_workspace is not None:
            prepare_workspace(workspace)
    compiled = compile_once(kernel, problem, glu, buffers, max_active_clusters)
    runtime_args = launch_args(kernel_args(problem, glu, *buffers))
    cu_stream = cuda.CUstream(stream.cuda_stream)

    def launch():
        compiled(*runtime_args, cu_stream)

    if defer_launch:
        return c, d, workspace, launch
    launch()
    return c, d, workspace, None


def plain_reference(problem):
    a, b, alpha, prob = problem["a"].float(), problem["b"].float(), problem["alpha"], problem["prob"][:, 0, 0]
    out = torch.zeros(problem["rows"], problem["n"], device="cuda")
    begin = 0
    for expert, end in enumerate(problem["offsets"].tolist()):
        out[begin:end] = alpha[expert] * (a[begin:end] @ b[expert].T)
        begin = end
    return out, out * prob[:, None]


def valid_rows(problem):
    return int(problem["offsets"][-1].item())


# Rows per expert.  More than 32 experts take the batch opener several lane rounds; an expert of more than 16 M tiles
# (EXPERT_BATCH_MAX_M_TILES) needs several batches.
ROUTINGS = {
    "uneven": [700, 0, 256, 1],
    "one_expert": [1300, 0, 0, 0],
    "many_experts_64": MANY_EXPERTS_64,
    "many_experts_128_small": [(256, 300, 512, 600, 768)[i % 5] for i in range(128)],
    "big_expert": BIG_EXPERT,
    "mostly_empty": [700 if i == 5 else 300 if i == 31 else 0 for i in range(32)],
    "rows_not_aligned": [255, 257, 511, 1025],
}


def experts_of(routing):
    return len(ROUTINGS[routing])


MATCH_CASES = [(glu, variant, routing) for glu in (False, True) for variant in ("signal", "signal_layout_pdl") for routing in sorted(ROUTINGS)]
MATCH_L0 = {(False, "signal", "many_experts_64"), (True, "signal", "big_expert"), (False, "signal_layout_pdl", "big_expert")}


@pytest.mark.parametrize(
    "glu, variant, routing", levels(MATCH_CASES, MATCH_L0, lambda glu, variant, routing: f"{'glu' if glu else 'plain'}-{variant}-{routing}")
)
def test_signal_consumer_matches_dynamic(glu, variant, routing):
    n = 2 * FFN if glu else FFN
    problem = make_problem(ROUTINGS[routing], n, HIDDEN, seed=len(routing))
    capacity = problem["rows"] // ROWS_ALIGNMENT
    options = dict(use_dynamic_sched=True, streaming_signal_capacity=capacity)
    if variant == "signal_layout_pdl":
        options.update(device_layout=True, use_pdl=True)
        trigger = "launch_pdl_dependents_at_all_streaming_signal_slots_ready" if glu else "launch_pdl_dependents_at_entry"
        options[trigger] = True
    experts = experts_of(routing)
    stock_c, stock_d, _, _ = run(make_kernel(glu, experts=experts, use_dynamic_sched=True), problem, glu)
    signal_kernel = make_kernel(glu, experts=experts, **options)
    c, d, _, _ = run(signal_kernel, problem, glu, prepare_workspace=signal_ready(signal_kernel))
    torch.cuda.synchronize()
    rows = valid_rows(problem)
    assert torch.equal(c[:rows], stock_c[:rows])
    assert torch.equal(d[:rows], stock_d[:rows])
    if not glu:
        c_ref, d_ref = plain_reference(problem)
        torch.testing.assert_close(c[:rows].float(), c_ref[:rows], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(d[:rows].float(), d_ref[:rows], atol=2e-2, rtol=2e-2)


NO_ROWS_CASES = [(device_layout, experts) for device_layout in (False, True) for experts in (EXPERTS, 64)]


@pytest.mark.parametrize("device_layout, experts", levels(NO_ROWS_CASES, {(True, EXPERTS)}, lambda device_layout, experts: f"{device_layout}-{experts}"))
def test_signal_consumer_with_no_rows_exits(device_layout, experts):
    problem = make_problem([0] * experts, FFN, HIDDEN)
    kernel = make_kernel(False, experts=experts, use_dynamic_sched=True, streaming_signal_capacity=1, device_layout=device_layout)
    run(kernel, problem, False, prepare_workspace=signal_ready(kernel))
    torch.cuda.synchronize()


PRODUCER_ROUTINGS = {"four_experts": [700, 300, 256, 513], "many_experts_64": ROUTINGS["many_experts_64"], "big_expert": ROUTINGS["big_expert"]}


def release_order(order, capacity):
    """The order in which the producer releases the M tiles."""
    if order == "in_order":
        return list(range(capacity))
    if order == "reverse":
        return list(reversed(range(capacity)))
    return torch.randperm(capacity, generator=torch.Generator().manual_seed(capacity)).tolist()


PRODUCER_CASES = [(glu, routing, order) for glu in (False, True) for routing in sorted(PRODUCER_ROUTINGS) for order in ("in_order", "reverse", "random")]


@pytest.mark.parametrize(
    "glu, routing, order",
    levels(PRODUCER_CASES, {(False, "many_experts_64", "reverse")}, lambda glu, routing, order: f"{'glu' if glu else 'plain'}-{routing}-{order}"),
)
def test_signal_consumer_waits_for_producer(glu, routing, order):
    """The GEMM starts while every signal[] entry reads not-ready and A holds NaN; a later kernel on another stream writes A one
    M tile at a time, in ``order``, and then marks it ready.  Reading any tile early would carry NaN into D."""
    n = 2 * FFN if glu else FFN
    tokens_per_expert = PRODUCER_ROUTINGS[routing]
    experts = len(tokens_per_expert)
    problem = make_problem(tokens_per_expert, n, HIDDEN, seed=7)
    rows = problem["rows"]
    capacity = rows // ROWS_ALIGNMENT
    final_a = problem["a"].clone()
    problem["a"].fill_(float("nan"))
    kernel = make_kernel(glu, experts=experts, use_dynamic_sched=True, streaming_signal_capacity=capacity)
    stock_c, stock_d, _, _ = run(make_kernel(glu, experts=experts, use_dynamic_sched=True), dict(problem, a=final_a), glu)
    torch.cuda.synchronize()

    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace, streaming_signal_workspace_offsets

    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    holder = {}

    def prepare(workspace):
        reset_streaming_signal_workspace(workspace, kernel)
        signal_word = streaming_signal_workspace_offsets(kernel)["signal"] // 4
        holder["signal"] = workspace.view(torch.int32)[signal_word : signal_word + capacity]

    # A few clusters only, so the producer kernels find free SMs while the GEMM waits.
    c, d, workspace, launch = run(kernel, problem, glu, max_active_clusters=4, stream=gemm_stream, defer_launch=True)
    # One launch with every M tile ready loads the GEMM's module: under lazy loading a first launch waits for running work.
    with torch.cuda.stream(gemm_stream):
        reset_streaming_signal_workspace(workspace, kernel, all_complete=True)
    launch()
    torch.cuda.synchronize()
    c.fill_(float("nan"))
    d.fill_(float("nan"))
    gemm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(gemm_stream):
        prepare(workspace)
    producer_stream.wait_stream(gemm_stream)
    # Producer launches go before the GEMM's, so no host launch has to wait for the spinning GEMM.  The sleep makes
    # the GEMM start waiting before the first row lands.
    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(50_000_000)
        for tile in release_order(order, capacity):
            rows_of_tile = slice(tile * ROWS_ALIGNMENT, (tile + 1) * ROWS_ALIGNMENT)
            problem["a"][rows_of_tile].copy_(final_a[rows_of_tile])
            holder["signal"][tile].fill_(tile)
    launch()
    torch.cuda.synchronize()
    valid = valid_rows(problem)
    assert torch.equal(c[:valid], stock_c[:valid])
    assert torch.equal(d[:valid], stock_d[:valid])


@pytest.mark.parametrize("glu", levels([(False,), (True,)], {(True,)}, lambda glu: "glu" if glu else "plain"))
def test_device_layout_waits_for_producer(glu):
    """The GEMM starts while the layout reads -1, every signal[] entry reads not-ready and A holds NaN; a later kernel on
    another stream writes the layout, then A one M tile at a time with its signal[] entry.  The GLU also triggers once every M tile is ready."""
    n = 2 * FFN if glu else FFN
    # Many experts: the layout has 65 entries to wait for, and the stock kernel is the one the other tests built.
    experts = len(MANY_EXPERTS_64)
    problem = make_problem(MANY_EXPERTS_64, n, HIDDEN, seed=11)
    capacity = problem["rows"] // ROWS_ALIGNMENT
    final_a, final_layout = problem["a"].clone(), problem["layout"].clone()
    options = dict(use_dynamic_sched=True, streaming_signal_capacity=capacity, device_layout=True)
    if glu:
        options["launch_pdl_dependents_at_all_streaming_signal_slots_ready"] = True
    kernel = make_kernel(glu, experts=experts, **options)
    stock_c, stock_d, _, _ = run(make_kernel(glu, experts=experts, use_dynamic_sched=True), problem, glu)
    torch.cuda.synchronize()

    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace, streaming_signal_workspace_offsets

    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    c, d, workspace, launch = run(kernel, problem, glu, max_active_clusters=4, stream=gemm_stream, defer_launch=True)
    # One launch with the layout written and every M tile ready loads the module (lazy loading waits for running work).
    with torch.cuda.stream(gemm_stream):
        reset_streaming_signal_workspace(workspace, kernel, all_complete=True)
    launch()
    torch.cuda.synchronize()
    signal_word = streaming_signal_workspace_offsets(kernel)["signal"] // 4
    signal = workspace.view(torch.int32)[signal_word : signal_word + capacity]
    c.fill_(float("nan"))
    d.fill_(float("nan"))
    problem["a"].fill_(float("nan"))
    problem["layout"].fill_(-1)
    gemm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(gemm_stream):
        reset_streaming_signal_workspace(workspace, kernel)
    producer_stream.wait_stream(gemm_stream)
    # Producer launches go before the GEMM's; the sleep makes the GEMM wait on the layout first.
    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(50_000_000)
        problem["layout"].copy_(final_layout)
        for tile in range(capacity):
            rows_of_tile = slice(tile * ROWS_ALIGNMENT, (tile + 1) * ROWS_ALIGNMENT)
            problem["a"][rows_of_tile].copy_(final_a[rows_of_tile])
            signal[tile].fill_(tile)
    launch()
    torch.cuda.synchronize()
    valid = valid_rows(problem)
    assert torch.equal(c[:valid], stock_c[:valid])
    assert torch.equal(d[:valid], stock_d[:valid])


# (D columns N, N chunk C): whole chunks; N < C (one chunk); N == C; a partial last chunk (columns 512..767).
OUTPUT_CHUNKS = {"n1024_c256": (2 * HIDDEN, 256), "n768_c1024": (768, 1024), "n768_c768": (768, 768), "n768_c512": (768, 512)}
# (N and chunk, routing): every chunk shape on the uneven routing; many experts and a big expert on two of them.
OUTPUT_CASES = [(n_and_chunk, "uneven") for n_and_chunk in sorted(OUTPUT_CHUNKS)] + [
    (n_and_chunk, routing) for n_and_chunk in ("n1024_c256", "n768_c512") for routing in ("many_experts_64", "big_expert")
]


NOTIFY_CASES = [(beside, one_cta, chunk, routing) for beside in (1, 1000) for one_cta in (False, True) for chunk, routing in OUTPUT_CASES]
# A partial last chunk at cluster (2, 1), whole chunks at (1, 1), and N < C (one chunk).
NOTIFY_L0 = {(1, False, "n768_c512", "many_experts_64"), (1, True, "n1024_c256", "uneven"), (1000, False, "n768_c1024", "uneven")}


@pytest.mark.parametrize(
    "clusters_beside_consumer, one_cta, n_and_chunk, routing",
    levels(NOTIFY_CASES, NOTIFY_L0, lambda beside, one_cta, chunk, routing: f"{beside}-{'cluster11' if one_cta else 'cluster21'}-{chunk}-{routing}"),
)
def test_output_notify_publishes_every_chunk(clusters_beside_consumer, one_cta, n_and_chunk, routing):
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import (
        StreamingOutputNotifyConfig,
        StreamingOutputNotifyMode,
        streaming_output_notify_capacity,
        streaming_output_notify_offsets,
    )

    n, chunk = OUTPUT_CHUNKS[n_and_chunk]
    experts = experts_of(routing)
    problem = make_problem(ROUTINGS[routing], n, FFN, seed=3)
    capacity = streaming_output_notify_capacity(chunk, n)
    assert capacity == len(range(0, n, chunk))
    notify = StreamingOutputNotifyConfig(StreamingOutputNotifyMode.N_CHUNK, chunk, capacity, clusters_beside_consumer=clusters_beside_consumer)
    stock_c, stock_d, _, _ = run(make_kernel(False, one_cta, experts=experts, use_dynamic_sched=True), problem, False)
    kernel = make_kernel(False, one_cta, experts=experts, use_dynamic_sched=True, streaming_output_notify=notify, launch_pdl_dependents_at_entry=True)
    c, d, workspace, _ = run(kernel, problem, False)
    torch.cuda.synchronize()
    valid = valid_rows(problem)
    assert torch.equal(c[:valid], stock_c[:valid])
    assert torch.equal(d[:valid], stock_d[:valid])
    offset = streaming_output_notify_offsets(kernel)["signal"]
    signal = workspace[offset : offset + 4 * capacity].view(torch.int32)
    assert signal.tolist() == list(range(capacity))


@pytest.mark.L0
def test_signal_slot_release_warm_stress():
    """The claim slot each cluster broadcasts is released only after every lane read it (python/cudnn/AGENTS.md Rule 11):
    back-to-back and shared-memory-poisoned launches of a shape with many claims per cluster, every byte against stock."""
    from warm_stress import StressCase, assert_warm_stress

    routing = ROUTINGS["big_expert"]
    problem = make_problem(routing, FFN, HIDDEN, seed=13)
    clusters = 4
    stock_c, stock_d, _, _ = run(make_kernel(False, experts=len(routing), use_dynamic_sched=True), problem, False)
    kernel = make_kernel(False, experts=len(routing), use_dynamic_sched=True, streaming_signal_capacity=problem["rows"] // ROWS_ALIGNMENT)
    c, d, workspace, launch = run(kernel, problem, False, prepare_workspace=signal_ready(kernel), max_active_clusters=clusters, defer_launch=True)
    reset_workspace = signal_ready(kernel)
    claims = (problem["rows"] // ROWS_ALIGNMENT) * (FFN // 256)
    assert_warm_stress(
        StressCase(
            "streaming signal scheduler, big expert, 4 clusters",
            launch=launch,
            outputs={"c": c, "d": d},
            thread_blocks=2 * clusters,
            refills_per_launch=claims // clusters,
            reference={"c": stock_c, "d": stock_d},
            reset=lambda: reset_workspace(workspace),
        ),
        warm=500,
        poisoned=500,
        budget_s=30.0,
    )


@pytest.mark.L0
def test_streaming_options_reject_invalid():
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import StreamingOutputNotifyConfig, StreamingOutputNotifyMode

    with pytest.raises(ValueError, match="use_dynamic_sched"):
        make_kernel(False, streaming_signal_capacity=4)
    with pytest.raises(ValueError, match="device_layout"):
        make_kernel(False, use_dynamic_sched=True, device_layout=True)
    with pytest.raises(ValueError, match="launch_pdl_dependents_at_all_streaming_signal_slots_ready"):
        make_kernel(True, use_dynamic_sched=True, launch_pdl_dependents_at_all_streaming_signal_slots_ready=True)
    notify = StreamingOutputNotifyConfig(StreamingOutputNotifyMode.N_CHUNK, 256, 4, clusters_beside_consumer=1)
    with pytest.raises(ValueError, match="use_dynamic_sched"):
        make_kernel(False, streaming_output_notify=notify)
    with pytest.raises(AssertionError):
        make_kernel(False, use_dynamic_sched=True, streaming_signal_capacity=4, streaming_output_notify=notify)
    import cutlass

    plain_class, _ = kernel_classes()
    with pytest.raises(ValueError, match="cluster"):
        plain_class(
            cutlass.Float32,
            True,
            (256, 256),
            (2, 2),
            True,
            True,
            False,
            EXPERTS,
            use_dynamic_sched=True,
            streaming_signal_capacity=4,
        )
