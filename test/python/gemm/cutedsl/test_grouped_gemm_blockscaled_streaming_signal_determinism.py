# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise determinism of the streaming signal options on the SM100 block-scaled grouped GEMM kernels: N_CHUNK output
notify on the MXFP8 quant kernel (FC2), and the input signal on the MXFP8 SwiGLU kernel (FC1; stock SF atoms or
``compact_sfa_sf_warp``) and on the BF16 and MXFP8 dSwiGLU kernels (``deterministic=True``).

The options change which cluster runs a tile and when, never the tile math, so the outputs (FC2: BF16 D; FC1: BF16 C
and D; dGLU: BF16 D and the FP32 dprob slots) must be raw-byte equal across replays, extra clusters that stop early or
never stop, chunk sizes, producer release orders, neighboring experts and the stock dynamic scheduler.

Directions: FC2 and FC1 are forward kernels, whose backward is unsupported by contract, not untested; the dGLU is FC1's
backward.  Inputs are quantized once and reused unchanged across every replay, packing and release order.
"""

import pytest

from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.cutedsl import _streaming_signal_blockscaled as bs
from gemm.cutedsl._streaming_signal_blockscaled import require_sm10x  # noqa: F401  (autouse fixture)

REPLAYS = 3


def run(launcher, problem):
    return launcher(problem)[0]


def case_ids(cases):
    return ["-".join([bs.KIND_IDS[kind], *map(str, rest)]) for kind, *rest in cases]


# ---- N_CHUNK output notify (FC2) ----


@pytest.mark.L0
@pytest.mark.parametrize("one_cta", [False, True], ids=lambda one_cta: f"fprop-mxfp8-{bs.cluster_id(one_cta)}")
@pytest.mark.parametrize("n_and_chunk", [(768, 512), (1024, 256)], ids=["n768_c512", "n1024_c256"])
def test_mxfp8_output_notify_replays_match_dynamic(n_and_chunk, one_cta):
    """Replays on fresh buffers beside concurrent work give the same bytes, which equal the stock dynamic scheduler's,
    whether the extra clusters stop early (one cluster beside the consumer) or never stop."""
    n, chunk = n_and_chunk
    tokens = bs.ROUTINGS["many_experts_64"]
    problem = bs.make_problem("quant", tokens, n, seed=3)
    expected = snapshot(run(bs.Launcher(bs.stock_kernel("quant", len(tokens), one_cta), problem), problem))
    for beside in (1, 1000):
        launcher = bs.Launcher(bs.notify_kernel(len(tokens), n, chunk, one_cta, beside), problem)
        assert_bitwise(assert_replays(lambda: run(launcher, problem), repeats=REPLAYS), expected, f"{beside} cluster(s) beside the consumer vs stock")


@pytest.mark.L0
def test_mxfp8_output_notify_chunk_size_invariance():
    """The chunk size reorders the tiles only: C = 256, 512 and 1024 (one chunk) give the same bytes."""
    tokens = bs.ROUTINGS["big_expert"]
    problem = bs.make_problem("quant", tokens, 1024, seed=9)
    results = {chunk: snapshot(run(bs.Launcher(bs.notify_kernel(len(tokens), 1024, chunk), problem), problem)) for chunk in (256, 512, 1024)}
    assert_bitwise(results[512], results[256], "C = 512 vs 256")
    assert_bitwise(results[1024], results[256], "C = 1024 vs 256")


# ---- The input signal (FC1 GLU, dGLU) ----

# Variants (``variant_options``): the signal, with its trigger (the GLU's device layout and all-slots-ready trigger, the
# dGLU's entry trigger), and with the compact SF warp.
REPLAY_CASES = [("glu", v, r) for v in ("signal", "trigger", "compact_signal", "compact_trigger") for r in ("many_experts_64", "big_expert")]
REPLAY_CASES += [
    (k, v, "many_experts_64")
    for k, v in [("dglu_bf16", "signal"), ("dglu_bf16", "trigger"), ("dglu", "signal"), ("dglu", "trigger"), ("dglu", "compact_signal")]
]


@pytest.mark.L0
@pytest.mark.parametrize("kind, variant, routing", REPLAY_CASES, ids=case_ids(REPLAY_CASES))
def test_signal_replays_match_dynamic(kind, variant, routing):
    """Every M tile ready: replays on fresh buffers beside concurrent work give the same bytes, which equal the stock
    scheduler's."""
    tokens = bs.ROUTINGS[routing]
    problem = bs.make_problem(kind, tokens, seed=len(routing))
    expected = snapshot(run(bs.Launcher(bs.stock_kernel(kind, len(tokens)), problem), problem))
    launcher = bs.Launcher(bs.make_kernel(kind, len(tokens), **bs.variant_options(kind, problem, variant)), problem)
    assert_bitwise(assert_replays(lambda: run(launcher, problem), repeats=REPLAYS), expected, f"{variant} vs stock")


RELEASE_CASES = [("glu", "signal"), ("glu", "trigger"), ("glu", "compact_signal"), ("glu", "compact_trigger")]
RELEASE_CASES += [("dglu_bf16", "signal"), ("dglu", "signal"), ("dglu", "compact_signal")]


@pytest.mark.L0
@pytest.mark.parametrize("kind, variant", RELEASE_CASES, ids=case_ids(RELEASE_CASES))
def test_signal_producer_release_order_invariance(kind, variant):
    """The GEMM starts while every M tile reads not-ready and A (dY) and its scales (SF atoms or compact rows) hold NaN,
    with ``device_layout`` also the expert layout; a producer stream writes each M tile and releases it, in order, in
    reverse and in a seeded random order, over 64 experts.  Every order gives the stock scheduler's bytes."""
    _, expected, results = bs.producer_case(kind, variant, "many_experts_64")
    expected = snapshot(expected)
    for order, got in results.items():
        assert_bitwise(snapshot(got), expected, f"{order} release vs stock")


# ---- Batch invariance and claim-slot warm stress, every kernel ----

BATCH_CASES = [("quant", "cluster21"), ("quant", "cluster11"), ("glu", "signal"), ("glu", "trigger"), ("dglu_bf16", "signal"), ("dglu", "signal")]


@pytest.mark.L0
@pytest.mark.parametrize("kind, variant", BATCH_CASES, ids=case_ids(BATCH_CASES))
def test_target_expert_batch_invariance(kind, variant):
    """The target expert's bytes do not depend on what is packed around it: alone, with neighbors, behind empty
    experts, among ragged ones (1, 255, 257 rows), reordered, or duplicated (N_CHUNK: N = 768, chunk 512)."""
    one_cta = variant == "cluster11"
    bank = bs.make_problem(kind, list(bs.BANK_TOKENS.values()), 768 if kind == "quant" else None, seed=1234)
    problems = {name: bs.pack(bank, packing) for name, packing in bs.PACKINGS.items()}
    alone, alone_rows = problems["alone"]
    if kind == "quant":
        kernel = bs.notify_kernel(bs.SLOTS, 768, 512, one_cta)
    else:
        kernel = bs.make_kernel(kind, bs.SLOTS, **bs.variant_options(kind, alone, variant))
    stock = bs.Launcher(bs.stock_kernel(kind, bs.SLOTS, one_cta), alone)(alone)[0]
    expected = snapshot({name: value[alone_rows[0]] for name, value in stock.items()})
    launcher = bs.Launcher(kernel, alone)
    for name, (problem, target_rows) in problems.items():
        out = run(launcher, problem)
        for copy, rows in enumerate(target_rows):
            assert_bitwise(snapshot({key: value[rows] for key, value in out.items()}), expected, f"target in {name} (copy {copy})")


WARM_CASES = [("quant", "n_chunk"), ("glu", "signal"), ("glu", "compact_signal"), ("dglu_bf16", "signal"), ("dglu", "signal"), ("dglu", "compact_signal")]


@pytest.mark.L0
@pytest.mark.parametrize("kind, variant", WARM_CASES, ids=case_ids(WARM_CASES))
def test_claim_slot_release_warm_stress(kind, variant):
    """The claim slot each cluster broadcasts, and with the SF warp its tile_info slot, are released only after every
    lane read them (python/cudnn/AGENTS.md Rule 11): back-to-back and shared-memory-poisoned launches of a 5000-row
    expert on 4 clusters (many claims per cluster), every byte against stock."""
    from warm_stress import StressCase, assert_warm_stress

    clusters, tokens = 4, bs.ROUTINGS["big_expert"]
    problem = bs.make_problem(kind, tokens, seed=13)
    assert problem["rows"] == sum(bs.padded(t) for t in tokens), "every output row is written, so the sentinel fill is overwritten"
    stock, _ = bs.Launcher(bs.stock_kernel(kind, len(tokens)), problem)(problem)
    if kind == "quant":
        kernel = bs.notify_kernel(len(tokens), problem["n"], 256)
    else:
        kernel = bs.make_kernel(kind, len(tokens), **bs.variant_options(kind, problem, variant))
    launcher = bs.Launcher(kernel, problem, max_active_clusters=clusters)
    buffers = launcher.buffers(problem)
    ready = bs.signal_ready(kernel)

    def launch():
        if kind.startswith("dglu"):
            buffers[1].zero_()  # the dprob slots: every launch starts from the same zeros
        launcher.launch(problem, buffers)

    assert_warm_stress(
        StressCase(
            f"{bs.KIND_IDS[kind]} {variant}, 5000-row expert, {clusters} clusters",
            launch=launch,
            outputs=bs.outputs(kernel, problem, buffers),
            thread_blocks=2 * clusters,
            refills_per_launch=(problem["rows"] // bs.ROWS_ALIGNMENT) * (problem["n"] // 256) // clusters,
            reference=stock,
            reset=(lambda: ready(buffers[-1])) if kernel.test_signal else None,
        ),
        warm=500,
        poisoned=500,
        budget_s=30.0,
    )
