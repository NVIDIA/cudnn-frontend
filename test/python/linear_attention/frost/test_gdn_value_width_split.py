# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The value-width split of the chunked GDN / GDP forward: which plans take it, and that it writes the uncut plan's bytes.

The split runs every (sequence, head) tile as two thread blocks, each owning half of the value columns.  A plan takes it where
its slot budget is exactly two.  A plan that runs ONE sequence uncut whatever its gates (the in-place call,
``overwrite_initial_state``) also takes it wherever two thread blocks per head fit in one wave of SMs, at every length at
which it neither chains nor carries a chain group.  ``batch_invariant`` never splits.

The byte tests call the public ops: the in-place call against its ``batch_invariant`` twin on the same operands, which runs
the uncut plan at these lengths, bit for bit; output and final state also against the fp64 recurrence, except in the
state-pool tests, which compare bytes only.  Equal bytes cannot tell which plan ran and the ops report no plan, so the plan
tests read the plan object the engine builds for a graph node: its thread blocks per tile, and that those of a split fit in
one wave.  They compile and launch nothing.  Expectations are computed from the package's own piece rule and from that fit,
and every length that stands beside one of the rule's thresholds comes from its constants or from the rule itself, so a
change of those constants moves these tests with it.  The L0 shapes below one chunk, one token past a chunk, over several
chunks and of the state pool are counted in the suite's chunk size, as are the edges of one chunk in the length sweeps."""

import pytest
import torch

import cudnn
from cudnn.linear_attention.frost import gdn_engine
from cudnn.linear_attention.frost.common.piece_chain import (
    CHAIN_MAX_PIECES,
    CHAIN_MIN_MEAN_TOKENS,
    DV_SPLIT_TILES,
    LENGTH_RULE_PIECE_TOKENS,
    choose_pieces,
    piece_budget,
    ragged_pieces,
)
from determinism_utils import assert_bitwise, snapshot
from linear_attention import test_la as la
from linear_attention.determinism import check_reference, pack, sequences

SM_COUNTS = (148, 152, 204)  # the sweep; the L0 cells take the SM count of the GPU they run on
KEY_HEADS, VALUE_HEADS = 16, 48  # one tile per (sequence, value head)
WIDTH = 128  # the key width, and the value width the split serves: a narrower value runs one thread block per tile
CHUNK = la.CHUNK["gdn"]
SUB_TOKENS = la.HOUSEHOLDER  # GDP: sub-tokens per token
BELOW_ONE_CHUNK, SEVERAL_CHUNKS = CHUNK - 1, 3 * CHUNK + 1
POOL_TOKENS = 2 * CHUNK + 2
COARSE_CADENCE = 2 * CHAIN_MIN_MEAN_TOKENS  # a sequence of the chain's first length is shorter than one unit of it: no chain
# both sides of every multiple of the length rule's piece, where a plan that does not chain may start to carry a chain group
PIECE_EDGES = [pieces * LENGTH_RULE_PIECE_TOKENS + past for pieces in range(1, CHAIN_MAX_PIECES + 1) for past in (0, 1)]


def takes_the_split(
    num_sm,
    lengths,
    *,
    heads=KEY_HEADS,
    value_heads=VALUE_HEADS,
    value_dim=WIDTH,
    in_place=True,
    batch_invariant=False,
    householders=1,
    cadence=0,
    summary=False,
):
    """Whether the plan of one call runs two thread blocks per tile on ``num_sm`` SMs, by the package's own piece rule.  A plan
    that neither chains nor is ``batch_invariant`` splits, at the split's value width, where its slot budget is the split's
    thread blocks per tile; one in-place sequence that carries no chain group also splits wherever two thread blocks per
    head fit in one wave.  A summary node has no in-place form."""
    heads_out, tokens = max(heads, value_heads), sum(lengths)
    shape = dict(num_seqs=len(lengths), heads_out=heads_out, num_sm=num_sm, total_tokens=tokens, b_t=CHUNK, expand_num=householders)
    pieces, unit_chunks = choose_pieces(**shape, cadence_tokens=cadence, batch_invariant=batch_invariant)
    if pieces or batch_invariant:
        return False
    if value_dim != WIDTH:
        return False
    if piece_budget(**shape, unit_chunks=unit_chunks) == DV_SPLIT_TILES:
        return True
    one_wave = DV_SPLIT_TILES * heads_out <= num_sm
    chain_group = ragged_pieces(heads_out=heads_out, num_sm=num_sm, total_tokens=tokens)
    return in_place and not summary and len(lengths) == 1 and one_wave and not chain_group


def build_plan(
    lengths, *, heads=KEY_HEADS, value_heads=VALUE_HEADS, value_dim=WIDTH, in_place=True, batch_invariant=False, householders=1, cadence=0, summary=False
):
    """The plan object the FROST engine builds for one GDN node (GDP with ``householders`` > 1), or for one GDN summary node."""
    total, heads_out = sum(lengths), max(heads, value_heads)
    bf16, fp32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    graph = cudnn.pygraph()
    ports = dict(
        k=graph.tensor([total * householders, heads, WIDTH], data_type=bf16, name="k"),
        v=graph.tensor([total * householders, value_heads, value_dim], data_type=bf16, name="v"),
        g=graph.tensor([total, heads_out], data_type=fp32, name="g"),
        beta=graph.tensor([total * householders, heads_out], data_type=fp32, name="beta"),
        cu_seqlens=graph.tensor([len(lengths) + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens"),
        initial_state=graph.tensor([len(lengths), heads_out, value_dim, WIDTH], data_type=fp32, name="initial_state"),
    )
    if summary:
        for output in graph.gdn_summary(**ports, output_transition=True, name="gdn_summary"):
            output.set_data_type(fp32)
        return gdn_engine.build_gdn_summary(graph)
    ports["q"] = graph.tensor([total, heads, WIDTH], data_type=bf16, name="q")
    attrs = dict(scale=0.125, output_final_state=True, batch_invariant=batch_invariant, overwrite_initial_state=in_place or None)
    if cadence:
        attrs["checkpoint_every_n_tokens"] = cadence
    if householders > 1:
        outputs = graph.gdp(**ports, **attrs, num_householder=householders, name="gdp")
    else:
        outputs = graph.gdn(**ports, **attrs, name="gdn")
    if cadence:
        outputs[2].set_data_type(bf16)
    return gdn_engine.build_gdn(graph)


def assert_plans_follow_the_rule(monkeypatch, num_sm, calls):
    """The plan of each call, built for a GPU of ``num_sm`` SMs, runs the thread blocks per tile the rule says; a plan on the
    split launches all of them in one wave."""
    monkeypatch.setattr(gdn_engine, "multiprocessor_count", lambda device: num_sm)
    for call in calls:
        plan = build_plan(**call)
        split = takes_the_split(num_sm, **call)
        assert plan.tiles_per_head == (DV_SPLIT_TILES if split else 1), (num_sm, call)
        assert not split or plan.n_tiles <= num_sm, (num_sm, call)


CODE_PATHS = (
    "below_one_chunk",
    "several_chunks",
    "state_rows",
    "expanded_timeline",
    "head_count_boundary",
    "chain_group",
    "batch_invariant",
    "default_call",
    "narrow_value",
    "several_sequences",
    "summary_node",
)


def one_row(tokens):
    """A cadence of one state row, at the end of the sequence: a plan whose cadence spans its sequence does not chain."""
    return -(-tokens // CHUNK) * CHUNK


def code_paths(num_sm):
    """The calls of each code path of the rule on a GPU of ``num_sm`` SMs, as keyword arguments of ``build_plan``."""
    fit = num_sm // DV_SPLIT_TILES  # the largest head count whose two thread blocks per head fit in one wave
    both = ([BELOW_ONE_CHUNK], [SEVERAL_CHUNKS])
    return {
        "below_one_chunk": [dict(lengths=[BELOW_ONE_CHUNK])],
        "several_chunks": [dict(lengths=[SEVERAL_CHUNKS])],
        "state_rows": [dict(lengths=[POOL_TOKENS], cadence=CHUNK), dict(lengths=[CHAIN_MIN_MEAN_TOKENS], cadence=COARSE_CADENCE)],
        "expanded_timeline": [dict(lengths=[CHUNK + 1], householders=SUB_TOKENS)],
        "head_count_boundary": [dict(lengths=lengths, heads=heads, value_heads=heads) for lengths in both for heads in (fit, fit + 1)],
        # one head: the slots of a chain group fit in one wave on any SM count
        "chain_group": [dict(lengths=[tokens], heads=1, value_heads=1, cadence=one_row(tokens)) for tokens in PIECE_EDGES],
        "batch_invariant": [dict(lengths=lengths, batch_invariant=True) for lengths in both],
        "default_call": [dict(lengths=lengths, in_place=False) for lengths in both],
        "narrow_value": [dict(lengths=lengths, value_dim=WIDTH // 2) for lengths in both],
        "several_sequences": [dict(lengths=lengths * 2, heads=KEY_HEADS, value_heads=KEY_HEADS) for lengths in both],
        "summary_node": [dict(lengths=lengths, summary=True) for lengths in both],
    }


@pytest.mark.L0
@pytest.mark.parametrize("path", CODE_PATHS)
def test_plan_follows_the_split_rule(monkeypatch, path):
    """One cell per code path of the rule, on this GPU's SM count.  One sequence in place takes the split below one chunk,
    over several chunks, with state rows, on GDP's expanded timeline, and up to the head count whose two thread blocks per
    head still fit; it keeps its plan past that head count and where it carries a chain group.  ``batch_invariant``, the
    default call, a narrower value, several sequences and the summary node answer as the rule did without ``uncut``."""
    assert_plans_follow_the_rule(monkeypatch, la.sm_count(), code_paths(la.sm_count())[path])


@pytest.mark.L1
@pytest.mark.parametrize("num_sm", SM_COUNTS)
def test_plans_follow_the_split_rule_at_every_length(monkeypatch, num_sm):
    """The sweep on the SM counts of other GPUs: every code path; every length of one sequence up to one chunk past the
    chain's first, for each call form, timeline, value width and head count around the fit, and for two to four sequences;
    the piece edges at the head counts of the byte tests."""
    fit = num_sm // DV_SPLIT_TILES
    forms = [dict(), dict(in_place=False), dict(summary=True), dict(batch_invariant=True), dict(value_dim=WIDTH // 2), dict(householders=SUB_TOKENS)]
    forms += [dict(cadence=CHUNK), dict(cadence=COARSE_CADENCE)]
    forms += [dict(heads=heads, value_heads=heads) for heads in (1, KEY_HEADS, fit, fit + 1)]
    calls = [call for path in code_paths(num_sm).values() for call in path]
    for tokens in range(1, CHAIN_MIN_MEAN_TOKENS + CHUNK + 1):
        calls += [dict(lengths=[tokens], **form) for form in forms]
        calls += [dict(lengths=[tokens] * count, heads=KEY_HEADS, value_heads=heads) for count in (2, 3, 4) for heads in (KEY_HEADS, VALUE_HEADS)]
    calls += [dict(lengths=[tokens], value_heads=heads, cadence=one_row(tokens)) for tokens in PIECE_EDGES for heads in (KEY_HEADS, VALUE_HEADS)]
    assert_plans_follow_the_rule(monkeypatch, num_sm, calls)


@pytest.fixture
def frost_engine():
    """The byte tests run where a FROST engine serves the ops and the split of the 48 value heads fits in one wave; each
    builds its own plans."""
    reason = la.frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    if DV_SPLIT_TILES * VALUE_HEADS > la.sm_count():
        pytest.skip(f"two thread blocks per head ({DV_SPLIT_TILES * VALUE_HEADS}) do not fit in one wave of {la.sm_count()} SMs")
    la.clear_caches()
    try:
        yield
    finally:
        la.clear_caches()


def lengths_where_the_plan_changes(num_sm, householders=1):
    """Lengths of one in-place sequence at 16 key / 48 value heads, all below the chain's first: the edges of one chunk, both
    sides of the range the slot-budget rule serves on ``num_sm`` SMs, the middle and the last."""
    below_chain = range(1, CHAIN_MIN_MEAN_TOKENS)
    budget_two = [tokens for tokens in below_chain if takes_the_split(num_sm, [tokens], in_place=False, householders=householders)]
    edges = [budget_two[0] - 1, budget_two[0], budget_two[-1], budget_two[-1] + 1] if budget_two else []
    lengths = {1, CHUNK - 1, CHUNK, CHUNK + 1, *edges, CHAIN_MIN_MEAN_TOKENS // 2, 3 * CHAIN_MIN_MEAN_TOKENS // 4, CHAIN_MIN_MEAN_TOKENS - 1}
    return sorted(lengths.intersection(below_chain))


def assert_in_place_matches_uncut(variant, tokens, *, heads=KEY_HEADS, value_heads=VALUE_HEADS, gate_domain="log"):
    """One sequence in place: output and final state are bit for bit those of the ``batch_invariant`` twin on the same
    operands, and within the fp64 recurrence's tolerance."""
    inputs = pack(sequences(variant, (tokens,), heads=heads, value_heads=value_heads, key_dim=WIDTH, value_dim=WIDTH))
    gate = inputs["g"].exp() if gate_domain == "linear" else inputs["g"]
    operands = [inputs["q"], inputs["k"], inputs["v"], gate, inputs["beta"], inputs["cu"]]
    op = la.la_ops.gated_delta_net
    if variant == "gdp":
        op, operands = la.la_ops.gated_delta_product, [*operands, SUB_TOKENS]

    def run(**kwargs):
        state = inputs["initial_state"].clone()
        out, final = op(*operands, initial_state=state, overwrite_initial_state=True, gate_domain=gate_domain, plan_name=f"{variant}_frost", **kwargs)
        assert final is state
        return snapshot({"out": out, "final_state": state})

    in_place = run()
    assert_bitwise(in_place, run(batch_invariant=True), f"{variant}, {tokens} tokens, {heads} / {value_heads} heads, {gate_domain} gate")
    check_reference(variant, inputs, in_place, False)


def assert_pool_and_rows_match_uncut(tokens, cadence, rows):
    """One sequence written in place into a paged state pool together with its checkpoint series: output, the pool's row, its
    untouched neighbours and every checkpoint row are bit for bit those of the ``batch_invariant`` twin."""
    inputs = pack(sequences("gdn", (tokens,), heads=KEY_HEADS, value_heads=VALUE_HEADS, key_dim=WIDTH, value_dim=WIDTH))
    indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    kwargs = dict(checkpoint_every_n_tokens=cadence, plan_name="gdn_frost")
    if rows == "fp32":
        kwargs["checkpoint_dtype"] = torch.float32

    def run(**more):
        pool = torch.full((3, VALUE_HEADS, WIDTH, WIDTH), 3.25, device="cuda")
        pool[1] = inputs["initial_state"][0]
        out, final, series = la.la_ops.gated_delta_net(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["cu"],
            initial_state=pool,
            state_indices=indices,
            output_final_state=True,
            **kwargs,
            **more,
        )
        assert final is pool and (pool[0] == 3.25).all() and (pool[2] == 3.25).all()
        return snapshot({"out": out, "pool": pool, "state_checkpoints": series[: -(-tokens // cadence)]})

    assert_bitwise(run(), run(batch_invariant=True), f"{tokens} tokens, a row every {cadence} tokens, {rows} rows")


@pytest.mark.L0
@pytest.mark.parametrize("tokens", [BELOW_ONE_CHUNK, SEVERAL_CHUNKS], ids=["below_one_chunk", "several_chunks"])
def test_split_writes_the_uncut_plan_bytes(frost_engine, tokens):
    """One sequence in place at 16 key / 48 value heads, below one chunk and over several: the split's output and final state
    are bit for bit the uncut plan's and within the fp64 recurrence's tolerance."""
    assert_in_place_matches_uncut("gdn", tokens)


@pytest.mark.L0
def test_split_that_fills_one_wave_writes_the_uncut_plan_bytes(frost_engine):
    """The head count whose two thread blocks per head fill one wave of this GPU."""
    heads = la.sm_count() // DV_SPLIT_TILES
    assert_in_place_matches_uncut("gdn", SEVERAL_CHUNKS, heads=heads, value_heads=heads)


@pytest.mark.L0
def test_split_with_a_state_pool_and_state_rows(frost_engine):
    """The paged state pool written in place together with the checkpoint series, a row at every chunk."""
    assert_pool_and_rows_match_uncut(POOL_TOKENS, CHUNK, "io")


@pytest.mark.L0
def test_split_writes_the_uncut_plan_bytes_on_the_expanded_timeline(frost_engine):
    """GDP, three sub-tokens per token, one sequence in place."""
    assert_in_place_matches_uncut("gdp", CHUNK + 1)


@pytest.mark.L1
@pytest.mark.parametrize("gate_domain", ["log", "linear"])
def test_split_writes_the_uncut_plan_bytes_around_every_plan_change(frost_engine, gate_domain):
    """The length sweep in both gate domains: on either side of every length at which the plan changes on this GPU."""
    for tokens in lengths_where_the_plan_changes(la.sm_count()):
        assert_in_place_matches_uncut("gdn", tokens, gate_domain=gate_domain)


@pytest.mark.L1
@pytest.mark.parametrize("rows", ["io", "fp32"])
def test_split_with_a_state_pool_and_state_rows_at_every_cadence(frost_engine, rows):
    """Both row dtypes, a row at every chunk and a cadence that keeps a sequence of the chain's first length from chaining."""
    for tokens, cadence in ((POOL_TOKENS, CHUNK), (CHAIN_MIN_MEAN_TOKENS // 2 + 1, CHUNK), (CHAIN_MIN_MEAN_TOKENS, COARSE_CADENCE)):
        assert_pool_and_rows_match_uncut(tokens, cadence, rows)


@pytest.mark.L1
def test_split_writes_the_uncut_plan_bytes_on_the_expanded_timeline_around_every_plan_change(frost_engine):
    """GDP on either side of every length at which its plan changes on this GPU."""
    for tokens in lengths_where_the_plan_changes(la.sm_count(), householders=SUB_TOKENS):
        assert_in_place_matches_uncut("gdp", tokens)
