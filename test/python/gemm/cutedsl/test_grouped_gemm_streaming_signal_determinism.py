# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise determinism of the streaming signal options of the BF16 grouped GEMM kernels (plain and GLU).

The options (``streaming_signal_capacity``, ``device_layout``, ``streaming_output_notify=N_CHUNK``) change which
cluster runs a tile and when, never the tile math, so C and D must be raw-byte equal across replays, producer release
orders, neighboring experts and the stock dynamic scheduler.

Directions: forward only.  These are forward kernels; backward is unsupported by contract, not untested.

Every category (replay, release order, batch invariance, N_CHUNK) runs at L0 on plain and GLU, both signal layouts, many
experts and an expert of several batches, and ragged and partial tiles; L1 adds the remaining combinations of those.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl._streaming_signal import BIG_EXPERT, MANY_EXPERTS_64, ROWS_ALIGNMENT, compile_once, kernel_args, launch_args
from gemm.cutedsl._streaming_signal import make_kernel as build_kernel
from gemm.cutedsl._streaming_signal import make_problem, new_buffers, padded, signal_ready

HIDDEN = 512
FFN = 512
REPLAYS = 3


@pytest.fixture(autouse=True)
def require_sm10x_cutedsl():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    from cudnn.tensor_adapter import has_tcgen05

    if major != 10 or not has_tcgen05(major):
        pytest.skip("an SM10x GPU (compute capability 10.x) is required")
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("requires nvidia-cutlass-dsl at or above the library floor")
    reason = cutedsl_arch_requirement_error((major, minor))
    if reason:
        pytest.skip(reason)


def make_kernel(glu, experts, **options):
    return build_kernel(glu, experts, use_dynamic_sched=True, **options)


class Launcher:
    """One compile of ``kernel`` for a problem shape, launched any number of times on fresh NaN-filled C, D and
    workspace buffers, so a replay never reads what an earlier launch left behind."""

    def __init__(self, kernel, problem, glu, max_active_clusters=None):
        self.kernel, self.glu = kernel, glu
        self._compiled = compile_once(kernel, problem, glu, self.buffers(problem), max_active_clusters)

    def buffers(self, problem):
        return new_buffers(self.kernel, problem, self.glu)

    def launch(self, problem, buffers, stream=None):
        """Enqueue the GEMM on ``stream`` (default: current) with ``buffers`` = (C, D, workspace)."""
        import cuda.bindings.driver as cuda

        stream = stream or torch.cuda.current_stream()
        self._compiled(*launch_args(kernel_args(problem, self.glu, *buffers)), cuda.CUstream(stream.cuda_stream))

    def __call__(self, problem, prepare_workspace=None):
        """Fresh buffers, ``prepare_workspace(workspace)``, one launch; returns {"c", "d"} over the experts' rows."""
        buffers = self.buffers(problem)
        if prepare_workspace is not None:
            prepare_workspace(buffers[2])
        self.launch(problem, buffers)
        rows = int(problem["offsets"][-1].item())
        return {"c": buffers[0][:rows], "d": buffers[1][:rows]}


def signal_options(glu, capacity, variant):
    options = dict(streaming_signal_capacity=capacity)
    if variant == "signal_layout_pdl":
        options.update(device_layout=True, use_pdl=True, launch_pdl_dependents_at_entry=True)
    return options


# More than 32 experts take the batch opener several lane rounds; an expert of more than 16 M tiles needs several batches.
ROUTINGS = {"many_experts_64": MANY_EXPERTS_64, "big_expert": BIG_EXPERT}


def levels(cases, l0, ids):
    """``cases`` as pytest params: L0 for those in ``l0``, L1 for the rest."""
    return [pytest.param(*case, marks=pytest.mark.L0 if case in l0 else pytest.mark.L1, id=ids(*case)) for case in cases]


def kernel_id(glu):
    return f"fprop-bf16-{'glu' if glu else 'plain'}"


REPLAY_CASES = [(glu, variant, routing) for glu in (False, True) for variant in ("signal", "signal_layout_pdl") for routing in sorted(ROUTINGS)]
# Each kernel, signal layout and routing at L0 at least once (the other four pairings are L1).
REPLAY_L0 = {
    (False, "signal", "many_experts_64"),
    (True, "signal", "big_expert"),
    (False, "signal_layout_pdl", "big_expert"),
    (True, "signal_layout_pdl", "many_experts_64"),
}


@pytest.mark.parametrize("glu, variant, routing", levels(REPLAY_CASES, REPLAY_L0, lambda glu, variant, routing: f"{kernel_id(glu)}-{variant}-{routing}"))
def test_signal_replays_match_dynamic(glu, variant, routing):
    """Replays on fresh buffers beside concurrent work give the same bytes, which equal the stock dynamic scheduler's."""
    n = 2 * FFN if glu else FFN
    tokens = ROUTINGS[routing]
    problem = make_problem(tokens, n, HIDDEN, seed=len(routing))
    capacity = problem["rows"] // ROWS_ALIGNMENT
    stock = Launcher(make_kernel(glu, len(tokens)), problem, glu)
    expected = snapshot(stock(problem))
    signal_kernel = make_kernel(glu, len(tokens), **signal_options(glu, capacity, variant))
    signal = Launcher(signal_kernel, problem, glu)
    replayed = assert_replays(lambda: signal(problem, signal_ready(signal_kernel)), repeats=REPLAYS)
    assert_bitwise(replayed, expected, "signal vs stock")


def release_order(order, capacity):
    if order == "in_order":
        return list(range(capacity))
    if order == "reverse":
        return list(reversed(range(capacity)))
    return torch.randperm(capacity, generator=torch.Generator().manual_seed(capacity)).tolist()


@pytest.mark.L0
@pytest.mark.parametrize("glu", [False, True], ids=["fprop-bf16-plain", "fprop-bf16-glu"])
def test_producer_release_order_invariance(glu):
    """The GEMM starts while every M tile reads not-ready and A holds NaN; a producer on another stream releases the M
    tiles in order, in reverse and in a seeded random order.  Every order gives the stock scheduler's bytes."""
    from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace, streaming_signal_workspace_offsets

    n = 2 * FFN if glu else FFN
    tokens = ROUTINGS["many_experts_64"]
    problem = make_problem(tokens, n, HIDDEN, seed=7)
    capacity = problem["rows"] // ROWS_ALIGNMENT
    final_a = problem["a"].clone()
    expected = snapshot(Launcher(make_kernel(glu, len(tokens)), problem, glu)(problem))
    kernel = make_kernel(glu, len(tokens), streaming_signal_capacity=capacity)
    # A few clusters only, so the producer kernels find free SMs while the GEMM waits.
    launcher = Launcher(kernel, problem, glu, max_active_clusters=4)
    # One launch with every M tile ready loads the module: under lazy loading a first launch waits for running work.
    launcher(problem, signal_ready(kernel))
    torch.cuda.synchronize()
    signal_word = streaming_signal_workspace_offsets(kernel)["signal"] // 4
    gemm_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    results = {}
    for order in ("in_order", "reverse", "random"):
        buffers = launcher.buffers(problem)
        problem["a"].fill_(float("nan"))
        signal = buffers[2].view(torch.int32)[signal_word : signal_word + capacity]
        gemm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gemm_stream):
            reset_streaming_signal_workspace(buffers[2], kernel)
        producer_stream.wait_stream(gemm_stream)
        # Producer launches go before the GEMM's, so no host launch waits for the spinning GEMM.
        with torch.cuda.stream(producer_stream):
            torch.cuda._sleep(20_000_000)
            for tile in release_order(order, capacity):
                rows_of_tile = slice(tile * ROWS_ALIGNMENT, (tile + 1) * ROWS_ALIGNMENT)
                problem["a"][rows_of_tile].copy_(final_a[rows_of_tile])
                signal[tile].fill_(tile)
        launcher.launch(problem, buffers, gemm_stream)
        torch.cuda.synchronize()
        rows = int(problem["offsets"][-1].item())
        results[order] = snapshot({"c": buffers[0][:rows], "d": buffers[1][:rows]})
        assert_bitwise(results[order], expected, f"{order} release vs stock")
    assert_bitwise(results["reverse"], results["in_order"], "reverse vs in-order release")
    assert_bitwise(results["random"], results["in_order"], "random vs in-order release")


# Batch invariance: one target expert keeps its rows, B, alpha and prob while its neighbors change.  Four expert slots
# each (empty slots pad the count), so one compiled kernel serves every packing.
BANK_TOKENS = {"target": 700, "n300": 300, "n513": 513, "r1": 1, "r255": 255, "r257": 257}
PACKINGS = {
    "alone": ["target", None, None, None],
    "neighbors": ["n300", "target", "n513", None],
    "empty_neighbors": [None, "target", None, None],
    "ragged": ["r1", "r255", "target", "r257"],
    "reordered": ["n513", "n300", None, "target"],
    "duplicated": ["target", "n300", "target", None],
}
SLOTS = 4
BATCH_CAPACITY = max(sum(padded(BANK_TOKENS[key]) for key in packing if key) for packing in PACKINGS.values()) // ROWS_ALIGNMENT


def expert_bank(n, k):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    bank = {}
    for key, tokens in BANK_TOKENS.items():
        rows = padded(tokens)
        bank[key] = dict(
            a=torch.randn(rows, k, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25,
            b=torch.randn(n, k, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25,
            alpha=torch.rand(1, generator=generator, device="cuda") + 0.5,
            prob=torch.rand(rows, 1, 1, generator=generator, device="cuda"),
        )
    return bank


def pack(bank, packing, n, k):
    """A problem of SLOTS experts in ``packing`` order over BATCH_CAPACITY M tiles; returns it and the target's row
    ranges."""
    rows = BATCH_CAPACITY * ROWS_ALIGNMENT
    a = torch.zeros(rows, k, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(SLOTS, n, k, device="cuda", dtype=torch.bfloat16)
    alpha = torch.ones(SLOTS, device="cuda")
    prob = torch.ones(rows, 1, 1, device="cuda")
    layout = torch.zeros(SLOTS + 1, device="cuda", dtype=torch.int32)
    begin, target_rows = 0, []
    for slot, key in enumerate(packing):
        if key is not None:
            expert = bank[key]
            end = begin + expert["a"].shape[0]
            a[begin:end], prob[begin:end] = expert["a"], expert["prob"]
            b[slot], alpha[slot] = expert["b"], expert["alpha"][0]
            if key == "target":
                target_rows.append(slice(begin, end))
            begin = end
        layout[slot + 1] = begin
    problem = {"a": a, "b": b, "layout": layout, "offsets": layout[1:], "alpha": alpha, "prob": prob, "rows": rows, "n": n, "k": k}
    return problem, target_rows


# (GLU, option): both signal layouts on both kernels; N_CHUNK output notify on the plain kernel, its tested contract.
BATCH_CASES = [(glu, mode) for glu in (False, True) for mode in ("signal-host-layout", "signal-device-layout")] + [(False, "n-chunk")]


def batch_kernel(glu, mode, n):
    if mode == "n-chunk":
        from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import (
            StreamingOutputNotifyConfig,
            StreamingOutputNotifyMode,
            streaming_output_notify_capacity,
        )

        notify = StreamingOutputNotifyConfig(StreamingOutputNotifyMode.N_CHUNK, 256, streaming_output_notify_capacity(256, n), clusters_beside_consumer=1)
        return make_kernel(glu, SLOTS, streaming_output_notify=notify, launch_pdl_dependents_at_entry=True), None
    kernel = make_kernel(glu, SLOTS, streaming_signal_capacity=BATCH_CAPACITY, device_layout=mode == "signal-device-layout")
    return kernel, signal_ready(kernel)


# Each kernel and each option at L0 once: plain host layout, GLU device layout, plain N_CHUNK.
BATCH_L0 = {(False, "signal-host-layout"), (True, "signal-device-layout"), (False, "n-chunk")}


@pytest.mark.parametrize("glu, mode", levels(BATCH_CASES, BATCH_L0, lambda glu, mode: f"{kernel_id(glu)}-{mode}"))
def test_target_expert_batch_invariance(glu, mode):
    """The target expert's C and D bytes do not depend on what is packed around it: alone, with neighbors, behind empty
    experts, among ragged ones (1, 255, 257 rows), reordered, or duplicated."""
    n = 2 * FFN if glu else FFN
    bank = expert_bank(n, HIDDEN)
    problems = {name: pack(bank, packing, n, HIDDEN) for name, packing in PACKINGS.items()}
    alone, alone_rows = problems["alone"]
    stock = Launcher(make_kernel(glu, SLOTS), alone, glu)
    stock_out = stock(alone)
    expected = snapshot({"c": stock_out["c"][alone_rows[0]], "d": stock_out["d"][alone_rows[0]]})
    kernel, prepare = batch_kernel(glu, mode, n)
    launcher = Launcher(kernel, alone, glu)
    for name, (problem, target_rows) in problems.items():
        out = launcher(problem, prepare)
        for copy, rows in enumerate(target_rows):
            got = snapshot({"c": out["c"][rows], "d": out["d"][rows]})
            assert_bitwise(got, expected, f"target in {name} (copy {copy})")


# A partial last chunk at L0; whole chunks are at L0 in the batch-invariance N_CHUNK case.
@pytest.mark.parametrize(
    "n_and_chunk", [pytest.param((2 * HIDDEN, 256), marks=pytest.mark.L1, id="n1024_c256"), pytest.param((768, 512), marks=pytest.mark.L0, id="n768_c512")]
)
def test_output_notify_replays_match_dynamic(n_and_chunk):
    """N_CHUNK replays give the same bytes, which equal the stock scheduler's whether the extra clusters stop early
    (one cluster beside the consumer) or never stop."""
    from cudnn.gemm.cutedsl.grouped.moe_streaming_output_chunk_tile_scheduler import (
        StreamingOutputNotifyConfig,
        StreamingOutputNotifyMode,
        streaming_output_notify_capacity,
    )

    n, chunk = n_and_chunk
    tokens = ROUTINGS["many_experts_64"]
    problem = make_problem(tokens, n, FFN, seed=3)
    capacity = streaming_output_notify_capacity(chunk, n)
    expected = snapshot(Launcher(make_kernel(False, len(tokens)), problem, False)(problem))
    results = {}
    for clusters_beside_consumer in (1, 1000):
        notify = StreamingOutputNotifyConfig(StreamingOutputNotifyMode.N_CHUNK, chunk, capacity, clusters_beside_consumer=clusters_beside_consumer)
        kernel = make_kernel(False, len(tokens), streaming_output_notify=notify, launch_pdl_dependents_at_entry=True)
        launcher = Launcher(kernel, problem, False)
        results[clusters_beside_consumer] = assert_replays(lambda: launcher(problem), repeats=REPLAYS)
        assert_bitwise(results[clusters_beside_consumer], expected, f"N_CHUNK, {clusters_beside_consumer} cluster(s) beside the consumer, vs stock")
    assert_bitwise(results[1], results[1000], "early-stopping vs non-stopping extra clusters")
