# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise determinism of the streaming input signal of the SM90 BF16 grouped GEMM kernels (plain and GLU forward,
cooperative and pingpong; dGLU backward with ``deterministic=True``, cooperative).

``streaming_signal_capacity`` (with or without ``device_layout`` and PDL) changes which cluster runs
a tile and when, never the tile math, so every output must be raw-byte equal across replays, producer release orders,
neighboring experts and the stock dynamic scheduler.

Directions: forward (plain, GLU) and backward (dGLU: D and the per-N-tile dprob slots).

Every category (replay, release order, batch invariance) runs at L0 on each kernel and schedule at least once; L1 adds the
remaining combinations.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl import _streaming_signal_sm90 as sig
from gemm.cutedsl._streaming_signal import ROWS_ALIGNMENT, padded, signal_ready
from gemm.cutedsl._streaming_signal_sm90 import require_sm90  # noqa: F401  (autouse fixture)
from gemm.cutedsl._streaming_signal_sm90 import Launcher, capacity_of, levels, make_kernel, make_problem, signal_options

REPLAYS = 3


def case_id(kind, schedule, *rest):
    return "-".join([sig.KIND_IDS[kind], schedule, *rest])


REPLAY_CASES = [
    (kind, schedule, variant, routing)
    for kind, schedule in sig.KIND_SCHEDULES
    for variant in ("signal", "signal_layout_pdl")
    for routing in ("big_expert", "many_experts_64")
]
REPLAY_L0 = {
    ("plain", "cooperative", "signal_layout_pdl", "many_experts_64"),
    ("plain", "pingpong", "signal", "big_expert"),
    ("glu", "cooperative", "signal", "big_expert"),
    ("glu", "pingpong", "signal_layout_pdl", "many_experts_64"),
    ("dglu", "cooperative", "signal", "many_experts_64"),
}


@pytest.mark.parametrize("kind, schedule, variant, routing", levels(REPLAY_CASES, REPLAY_L0, case_id))
def test_signal_replays_match_dynamic(kind, schedule, variant, routing):
    """Replays on fresh buffers beside concurrent work give the same bytes, which equal the stock dynamic scheduler's."""
    tokens = sig.ROUTINGS[routing]
    problem = make_problem(kind, tokens, seed=len(routing))
    expected = snapshot(Launcher(make_kernel(kind, len(tokens), schedule), problem)(problem))
    kernel = make_kernel(kind, len(tokens), schedule, **signal_options(kind, capacity_of(problem), variant))
    launcher = Launcher(kernel, problem)
    replayed = assert_replays(lambda: launcher(problem, signal_ready(kernel)), repeats=REPLAYS)
    assert_bitwise(replayed, expected, "signal vs stock")


ORDER_L0 = {("plain", "pingpong"), ("glu", "cooperative"), ("dglu", "cooperative")}


@pytest.mark.parametrize("kind, schedule", levels(sig.KIND_SCHEDULES, ORDER_L0, case_id))
def test_producer_release_order_invariance(kind, schedule):
    """The GEMM starts while every M tile reads not-ready and A holds NaN; a producer on another stream releases the M
    tiles in order, in reverse and in a seeded random order.  Every order gives the stock scheduler's bytes."""
    tokens = sig.ROUTINGS["many_experts_64"]
    problem = make_problem(kind, tokens, seed=7)
    expected = snapshot(Launcher(make_kernel(kind, len(tokens), schedule), problem)(problem))
    kernel = make_kernel(kind, len(tokens), schedule, streaming_signal_capacity=capacity_of(problem))
    # A few clusters only, so the producer kernels find free SMs while the GEMM waits.
    launcher = Launcher(kernel, problem, max_active_clusters=4)
    results = sig.run_with_producer(launcher, problem, problem["a"].clone(), ("in_order", "reverse", "random"))
    for order, got in results.items():
        assert_bitwise(snapshot(got), expected, f"{order} release vs stock")


# Batch invariance: one target expert keeps its rows, weights, scales and probabilities while its neighbors change.  Four
# expert slots each (empty slots pad the count), so one compiled kernel serves every packing.
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


def expert_bank(kind):
    n, k = sig.gemm_n(kind), sig.HIDDEN
    generator = torch.Generator(device="cuda").manual_seed(1234)
    bank = {}
    for key, tokens in BANK_TOKENS.items():
        rows = padded(tokens)
        bank[key] = dict(
            a=torch.randn(rows, k, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25,
            b=torch.randn(n, k, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.25,
            alpha=torch.rand(1, generator=generator, device="cuda") + 0.5,
            beta=torch.rand(1, generator=generator, device="cuda") + 0.5,
            prob=torch.rand(rows, 1, 1, generator=generator, device="cuda"),
            c=(torch.randn(rows, 2 * sig.FFN, generator=generator, device="cuda") * 4).to(torch.bfloat16),
        )
    return bank


def pack(kind, bank, packing):
    """A problem of SLOTS experts in ``packing`` order over BATCH_CAPACITY M tiles, and the target's row ranges."""
    n, k = sig.gemm_n(kind), sig.HIDDEN
    rows = BATCH_CAPACITY * ROWS_ALIGNMENT
    problem = dict(
        a=torch.zeros(rows, k, device="cuda", dtype=torch.bfloat16),
        b=torch.zeros(SLOTS, n, k, device="cuda", dtype=torch.bfloat16),
        alpha=torch.ones(SLOTS, device="cuda"),
        beta=torch.ones(SLOTS, device="cuda"),
        prob=torch.ones(rows, 1, 1, device="cuda"),
        c=torch.zeros(rows, 2 * sig.FFN, device="cuda", dtype=torch.bfloat16),
        layout=torch.zeros(SLOTS + 1, device="cuda", dtype=torch.int32),
        rows=rows,
        n=n,
        k=k,
        kind=kind,
    )
    begin, target_rows = 0, []
    for slot, key in enumerate(packing):
        if key is not None:
            expert = bank[key]
            end = begin + expert["a"].shape[0]
            for name in ("a", "prob", "c"):
                problem[name][begin:end] = expert[name]
            problem["b"][slot] = expert["b"]
            problem["alpha"][slot], problem["beta"][slot] = expert["alpha"][0], expert["beta"][0]
            if key == "target":
                target_rows.append(slice(begin, end))
            begin = end
        problem["layout"][slot + 1] = begin
    problem["offsets"] = problem["layout"][1:]
    return problem, target_rows


BATCH_CASES = [(kind, schedule, mode) for kind, schedule in sig.KIND_SCHEDULES for mode in ("signal", "signal_layout_pdl")]
BATCH_L0 = {("plain", "cooperative", "signal"), ("glu", "pingpong", "signal_layout_pdl"), ("dglu", "cooperative", "signal")}


@pytest.mark.parametrize("kind, schedule, mode", levels(BATCH_CASES, BATCH_L0, case_id))
def test_target_expert_batch_invariance(kind, schedule, mode):
    """The target expert's output bytes do not depend on what is packed around it: alone, with neighbors, behind empty
    experts, among ragged ones (1, 255, 257 rows), reordered, or duplicated."""
    bank = expert_bank(kind)
    problems = {name: pack(kind, bank, packing) for name, packing in PACKINGS.items()}
    alone, alone_rows = problems["alone"]
    stock = Launcher(make_kernel(kind, SLOTS, schedule), alone)(alone)
    expected = snapshot({name: value[alone_rows[0]] for name, value in stock.items()})
    kernel = make_kernel(kind, SLOTS, schedule, **signal_options(kind, BATCH_CAPACITY, mode))
    launcher = Launcher(kernel, alone)
    for name, (problem, target_rows) in problems.items():
        out = launcher(problem, signal_ready(kernel))
        for copy, rows in enumerate(target_rows):
            got = snapshot({key: value[rows] for key, value in out.items()})
            assert_bitwise(got, expected, f"target in {name} (copy {copy})")
