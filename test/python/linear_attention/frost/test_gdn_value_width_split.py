# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The value-width split of the chunked GDN / GDP forward: which plans take it, and that it writes the uncut plan's bytes.

The split runs every (sequence, head) tile as two thread blocks, each owning half of the 128 value columns.  A plan takes it
where its slot budget is exactly two.  A plan that runs ONE sequence uncut, without the split-K table (the in-place call,
``overwrite_initial_state``) and without a chain group for the device to take, also takes it at every length below the
chain's, wherever two thread blocks per head fit in one wave of SMs.  ``batch_invariant`` never splits.  The plan tests build
plan objects only: nothing compiles and nothing is launched.  The byte tests run where a FROST engine serves the op."""

import pytest
import torch

import cudnn
from cudnn.linear_attention.frost import gdn_engine
from cudnn.linear_attention.frost.common.piece_chain import DV_SPLIT_TILES, choose_pieces, is_dv_split
from determinism_utils import assert_bitwise, snapshot
from linear_attention import test_la as la
from linear_attention.determinism import check_reference, pack, sequences

pytestmark = [pytest.mark.L0]

SM_COUNTS = (148, 152, 204)
KEY_HEADS, VALUE_HEADS = 16, 48  # one tile per (sequence, value head): 48 tiles for one sequence
CHUNK = 64
CHAIN_TOKENS = 2048  # one sequence chains from here on, at three or more SMs per tile
COARSE_CADENCE = 4096  # a checkpoint cadence of 64 chunks: one sequence then chains from 12 x 4,096 tokens on only
GROUP_TOKENS = 2 * 8192  # above it a plan without pieces carries the chain group the device may take
BUDGET_TWO = (449, 512, 704)  # one sequence at 48 heads: 8 to 11 chunks hold exactly two slots of four chunks
NEW_LENGTHS = (1, 63, 64, 65, 448, 705, 1024, 1536, 2047)  # the lengths on either side of them, up to the chain's


def rule(tokens, *, num_sm, num_seqs=1, heads_out=VALUE_HEADS, dim_v=128, cadence=0, **uncut):
    """``(pieces, split)`` of a plan without ``batch_invariant``: the chain's slot count and, for a plan that does not chain,
    the split rule's answer."""
    shape = dict(num_seqs=num_seqs, heads_out=heads_out, num_sm=num_sm, total_tokens=tokens, b_t=CHUNK, expand_num=1)
    pieces, unit_chunks = choose_pieces(**shape, cadence_tokens=cadence, batch_invariant=False)
    return pieces, pieces == 0 and is_dv_split(**shape, dim_v=dim_v, unit_chunks=unit_chunks, **uncut)


@pytest.mark.parametrize("num_sm", SM_COUNTS)
def test_one_uncut_sequence_splits_at_every_length_below_the_chain(num_sm):
    """One sequence of a plan that runs it uncut: the split at every length up to the chain's first, at any head count whose two
    thread blocks per head fit in one wave, and only at a value width of 128."""
    for tokens in range(1, CHAIN_TOKENS):
        assert rule(tokens, num_sm=num_sm, uncut=True) == (0, True), tokens
        assert rule(tokens, num_sm=num_sm, dim_v=64, uncut=True) == (0, False), tokens
    assert rule(CHAIN_TOKENS, num_sm=num_sm, uncut=True) == (num_sm // VALUE_HEADS, False)
    for tokens in (1, 448, 705, CHAIN_TOKENS, 4 * CHAIN_TOKENS):
        for heads_out in (1, KEY_HEADS, num_sm // DV_SPLIT_TILES):
            pieces, split = rule(tokens, num_sm=num_sm, heads_out=heads_out, uncut=True)
            assert split == (pieces == 0), (tokens, heads_out)
        assert rule(tokens, num_sm=num_sm, heads_out=num_sm // DV_SPLIT_TILES + 1, uncut=True) == (0, False), tokens
    # a coarse checkpoint cadence keeps the plan from chaining: the split up to the length where the plan carries a chain group
    for tokens, split in ((CHAIN_TOKENS, True), (GROUP_TOKENS, True), (GROUP_TOKENS + 1, False), (3 * 8192, False)):
        assert rule(tokens, num_sm=num_sm, cadence=COARSE_CADENCE, uncut=True) == (0, split), tokens


@pytest.mark.parametrize("num_sm", SM_COUNTS)
def test_several_uncut_sequences_keep_the_slot_budget_rule(num_sm):
    """Two or more sequences: an uncut plan answers as a plan with the split-K table does, also where two thread blocks per tile
    would fit in one wave."""
    for num_seqs in (2, 3, 4, 16):
        for heads_out in (1, KEY_HEADS, VALUE_HEADS):
            for tokens in range(num_seqs, num_seqs * CHAIN_TOKENS + 1, 61):
                shape = dict(num_sm=num_sm, num_seqs=num_seqs, heads_out=heads_out)
                assert rule(tokens, **shape, uncut=True) == rule(tokens, **shape), (num_seqs, heads_out, tokens)


@pytest.mark.parametrize("num_sm", SM_COUNTS)
def test_a_plan_with_the_split_k_table_splits_at_a_slot_budget_of_two(num_sm):
    """A plan that may cut a sequence where its gates saturate keeps its rule: the split where the slot budget is exactly two."""
    for tokens in range(1, CHAIN_TOKENS):
        assert rule(tokens, num_sm=num_sm) == (0, 449 <= tokens <= 704), tokens
    assert rule(CHAIN_TOKENS, num_sm=num_sm) == (num_sm // VALUE_HEADS, False)
    # four sequences of 1,024 tokens at 16 heads: 64 tiles, a budget of two where the SMs hold two per tile and of three where three
    assert rule(4 * 1024, num_sm=num_sm, num_seqs=4, heads_out=KEY_HEADS) == (0, num_sm // (4 * KEY_HEADS) == DV_SPLIT_TILES)
    # one sequence on more than a third of the SMs never chains: two slots from 449 tokens on, at any length
    for tokens, split in ((448, False), (449, True), (4 * CHAIN_TOKENS, True)):
        assert rule(tokens, num_sm=num_sm, heads_out=num_sm // DV_SPLIT_TILES) == (0, split), tokens


def gdn_plan(
    monkeypatch, num_sm, lengths, *, heads=KEY_HEADS, value_heads=VALUE_HEADS, value_dim=128, in_place=True, batch_invariant=False, householders=1, cadence=0
):
    """The plan object the FROST engine builds for one GDN node (GDP with ``householders`` > 1) on a GPU of ``num_sm`` SMs."""
    if not torch.cuda.is_available():
        pytest.skip("a plan object is built for a CUDA device")
    monkeypatch.setattr(gdn_engine, "multiprocessor_count", lambda device: num_sm)
    total, heads_out = sum(lengths), max(heads, value_heads)
    bf16, fp32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    graph = cudnn.pygraph()
    ports = dict(
        q=graph.tensor([total, heads, 128], data_type=bf16, name="q"),
        k=graph.tensor([total * householders, heads, 128], data_type=bf16, name="k"),
        v=graph.tensor([total * householders, value_heads, value_dim], data_type=bf16, name="v"),
        g=graph.tensor([total, heads_out], data_type=fp32, name="g"),
        beta=graph.tensor([total * householders, heads_out], data_type=fp32, name="beta"),
        cu_seqlens=graph.tensor([len(lengths) + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens"),
        initial_state=graph.tensor([len(lengths), heads_out, value_dim, 128], data_type=fp32, name="initial_state"),
    )
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


def scheme(plan):
    """``"chain"``, ``"split"`` (the value-width split), ``"table"`` (the split-K table) or ``"uncut"``."""
    return "chain" if plan.chain else "split" if plan.dv_split else "table" if plan.split else "uncut"


@pytest.mark.parametrize("num_sm", SM_COUNTS)
def test_one_sequence_in_place_takes_the_split(monkeypatch, num_sm):
    """One sequence run in place at 16 key / 48 value heads: two items per head, each a thread block of its own, at every length
    below the chain's; GDP's expanded timeline and any head count that fits follow the same rule."""
    for tokens in (*NEW_LENGTHS, *BUDGET_TWO):
        plan = gdn_plan(monkeypatch, num_sm, [tokens])
        assert scheme(plan) == "split", tokens
        assert (plan.tiles_per_head, plan.n_tiles, plan.work_item_rows, plan.tail) == (
            DV_SPLIT_TILES,
            DV_SPLIT_TILES * VALUE_HEADS,
            DV_SPLIT_TILES * VALUE_HEADS,
            None,
        )
    for tokens in (1, 448, 705):
        assert scheme(gdn_plan(monkeypatch, num_sm, [tokens], householders=3)) == "split", tokens
        half = num_sm // DV_SPLIT_TILES
        assert scheme(gdn_plan(monkeypatch, num_sm, [tokens], heads=half, value_heads=half)) == "split", tokens
        assert scheme(gdn_plan(monkeypatch, num_sm, [tokens], heads=half + 1, value_heads=half + 1)) == "uncut", tokens
        assert scheme(gdn_plan(monkeypatch, num_sm, [tokens], heads=KEY_HEADS, value_heads=KEY_HEADS)) == "split", tokens
    for tokens in (CHAIN_TOKENS, GROUP_TOKENS):
        assert scheme(gdn_plan(monkeypatch, num_sm, [tokens], cadence=COARSE_CADENCE)) == "split", tokens


@pytest.mark.parametrize("num_sm", SM_COUNTS)
def test_plans_that_keep_their_scheme(monkeypatch, num_sm):
    """``batch_invariant`` never splits; a plan with the split-K table splits where its slot budget is exactly two and keeps the
    table elsewhere; the chain starts at 2,048 tokens; a value width of 64, a plan with the chain group and several sequences
    keep the uncut plan."""
    for tokens in (*NEW_LENGTHS, *BUDGET_TWO):
        assert scheme(gdn_plan(monkeypatch, num_sm, [tokens], batch_invariant=True)) == "uncut", tokens
        assert scheme(gdn_plan(monkeypatch, num_sm, [tokens], value_dim=64)) == "uncut", tokens
        assert scheme(gdn_plan(monkeypatch, num_sm, [tokens], in_place=False)) == ("split" if tokens in BUDGET_TWO else "table"), tokens
    for in_place in (False, True):
        assert not gdn_plan(monkeypatch, num_sm, [CHAIN_TOKENS - 1], in_place=in_place).chain
        chain = gdn_plan(monkeypatch, num_sm, [CHAIN_TOKENS], in_place=in_place)
        assert (scheme(chain), chain.pieces) == ("chain", num_sm // VALUE_HEADS)
    # one sequence that does not chain (a coarse cadence) and is long enough for the chain group: the device may still take the group
    grouped = gdn_plan(monkeypatch, num_sm, [GROUP_TOKENS + 1], cadence=COARSE_CADENCE)
    assert (scheme(grouped), grouped.tiles_per_head, grouped.ragged_pieces) == ("uncut", 1, num_sm // VALUE_HEADS)
    # several sequences in place: the SMs filled (four at 48 heads), and two thread blocks per tile fitting (two at 16 heads)
    for lengths, heads in (([1024] * 4, VALUE_HEADS), ([1024] * 3, VALUE_HEADS), ([1024, 1024], KEY_HEADS), ([300, 300], KEY_HEADS)):
        plan = gdn_plan(monkeypatch, num_sm, lengths, value_heads=heads)
        assert (scheme(plan), plan.tiles_per_head) == ("uncut", 1), (lengths, heads)


@pytest.fixture
def plans(monkeypatch):
    """Every forward plan object the GDN / GDP engine builds during the test, in build order."""
    reason = la.frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    if DV_SPLIT_TILES * VALUE_HEADS > la.sm_count():
        pytest.skip(f"two thread blocks per head ({DV_SPLIT_TILES * VALUE_HEADS}) do not fit in one wave of {la.sm_count()} SMs")
    built, construct = [], gdn_engine.CompiledGdn.__init__

    def recording(self, *args, **kwargs):
        construct(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(gdn_engine.CompiledGdn, "__init__", recording)
    la.clear_caches()
    try:
        yield built
    finally:
        la.clear_caches()


def assert_split_and_uncut_twin(plans):
    """The two plans of a byte test: the in-place call's value-width split and its ``batch_invariant`` twin's uncut plan."""
    split, uncut = plans
    assert (scheme(split), split.tiles_per_head) == ("split", DV_SPLIT_TILES)
    assert (scheme(uncut), uncut.tiles_per_head) == ("uncut", 1)


@pytest.mark.parametrize("gate_domain", ["log", "linear"])
@pytest.mark.parametrize("tokens", [1, 63, 64, 65, 448, 449, 704, 705, 1024, 1536, 2047])
def test_split_writes_the_uncut_plan_bytes(plans, tokens, gate_domain):
    """One sequence in place at 16 key / 48 value heads, in both gate domains: output and final state of the split are bit for
    bit the uncut plan's (the ``batch_invariant`` twin on the same operands), and within the fp64 recurrence's tolerance."""
    inputs = pack(sequences("gdn", (tokens,), heads=KEY_HEADS, value_heads=VALUE_HEADS, key_dim=128, value_dim=128))
    gate = inputs["g"].exp() if gate_domain == "linear" else inputs["g"]

    def run(**kwargs):
        state = inputs["initial_state"].clone()
        out, final = la.la_ops.gated_delta_net(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            gate,
            inputs["beta"],
            inputs["cu"],
            initial_state=state,
            overwrite_initial_state=True,
            gate_domain=gate_domain,
            plan_name="gdn_frost",
            **kwargs,
        )
        assert final is state
        return snapshot({"out": out, "final_state": state})

    split, uncut = run(), run(batch_invariant=True)
    assert_split_and_uncut_twin(plans)
    assert_bitwise(split, uncut, f"{tokens} tokens, {gate_domain} gate")
    check_reference("gdn", inputs, split, False)


@pytest.mark.parametrize("rows", ["io", "fp32"])
@pytest.mark.parametrize("tokens,cadence", [(130, CHUNK), (705, CHUNK), (CHAIN_TOKENS, COARSE_CADENCE)])
def test_split_with_a_state_pool_and_state_rows(plans, tokens, cadence, rows):
    """The paged state pool written in place together with the checkpoint series, in both row dtypes: output, the pool's row,
    its untouched neighbours and every checkpoint row are bit for bit the uncut plan's; the coarse cadence keeps 2,048 tokens
    from chaining."""
    inputs = pack(sequences("gdn", (tokens,), heads=KEY_HEADS, value_heads=VALUE_HEADS, key_dim=128, value_dim=128))
    indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    kwargs = dict(checkpoint_every_n_tokens=cadence, plan_name="gdn_frost")
    if rows == "fp32":
        kwargs["checkpoint_dtype"] = torch.float32

    def run(**more):
        pool = torch.full((3, VALUE_HEADS, 128, 128), 3.25, device="cuda")
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

    split, uncut = run(), run(batch_invariant=True)
    assert_split_and_uncut_twin(plans)
    assert_bitwise(split, uncut, f"{tokens} tokens, {rows} rows")


@pytest.mark.parametrize("tokens", [65, 705])
def test_split_writes_the_uncut_plan_bytes_on_the_expanded_timeline(plans, tokens):
    """GDP, three sub-tokens per token, one sequence in place: output and final state bit for bit the uncut plan's, within the
    fp64 recurrence's tolerance."""
    inputs = pack(sequences("gdp", (tokens,), heads=KEY_HEADS, value_heads=VALUE_HEADS, key_dim=128, value_dim=128))

    def run(**kwargs):
        state = inputs["initial_state"].clone()
        out, final = la.la_ops.gated_delta_product(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["cu"],
            3,
            initial_state=state,
            overwrite_initial_state=True,
            plan_name="gdp_frost",
            **kwargs,
        )
        assert final is state
        return snapshot({"out": out, "final_state": state})

    split, uncut = run(), run(batch_invariant=True)
    assert_split_and_uncut_twin(plans)
    assert_bitwise(split, uncut, f"{tokens} tokens")
    check_reference("gdp", inputs, split, False)
