# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the chunked GDN forward chains, and that both sides of that length compute the same function.

The shared piece rule (``choose_pieces``) cuts long sequences on few tiles into a chain of pieces.  A GDN forward that
overwrites its initial state keeps the chain only where ``chain_pays`` against the plan it would run instead: the value-width
split for one sequence, the uncut plan otherwise, and the leftover tiles' own round for the chain tail of a batch above one
wave.  Such a plan that does not chain carries the length-rule chain group for the device to take only where some batch of
its shape can gain from it (``chain_group_pays``), and a chain whose plan below would carry that group stays.  The default
call, whose plan below the chain is the split-K table, ``batch_invariant``, a context port, GDP and the summary node keep the
shared rule.

The rule tests state properties of ``chain_pays``.  Equal function cannot tell which plan ran and the ops report no plan, so
the plan tests read the plan object the engine builds for a graph node; they compile and launch nothing.  Every length and
expectation comes from the package's own rules, so a change of their constants moves these tests with it.  The byte tests
call the public op: below the chain a sequence has the bytes of its ``batch_invariant`` twin where that twin runs uncut, and
on both sides of the chain's first length the output and final state match the fp64 recurrence; a batch that no longer
carries the chain group has the bytes it has with the group carried."""

import pytest
import torch

from cudnn.linear_attention.frost import gdn_engine
from cudnn.linear_attention.frost.common import piece_chain
from cudnn.linear_attention.frost.common.piece_chain import (
    CHAIN_MAX_PIECES,
    CHAIN_MIN_MEAN_TOKENS,
    CHAIN_MIN_PIECES,
    LENGTH_RULE_PIECE_TOKENS,
    chain_group_pays,
    chain_pays,
    choose_pieces,
    ragged_pieces,
)
from cudnn.linear_attention.frost.common.tail import choose_tail
from determinism_utils import assert_bitwise, snapshot
from linear_attention import test_la as la
from linear_attention.determinism import check_reference, pack, sequences
from test_gdn_value_width_split import CHUNK, KEY_HEADS, SM_COUNTS, SUB_TOKENS, VALUE_HEADS, WIDTH, build_plan, one_row, plan_scheme

SLOTS = range(CHAIN_MIN_PIECES, CHAIN_MAX_PIECES + 1)
LENGTHS = range(CHUNK, 64 * CHAIN_MIN_MEAN_TOKENS + 1, CHUNK)  # whole chunks, far past every length at which a chain starts
NEVER = float("inf")


def first_paying(pieces, **rival):
    """The shortest sequence of whole chunks at which a chain of ``pieces`` slots pays against ``rival``."""
    return next((tokens for tokens in LENGTHS if chain_pays(pieces=pieces, num_seqs=1, total_tokens=tokens, **rival)), NEVER)


@pytest.mark.L0
@pytest.mark.parametrize("rival", [dict(), dict(split=True), dict(tail=True)], ids=["uncut", "value_width_split", "leftover_round"])
def test_chain_pays_from_one_length_on(rival):
    """Against each rival plan: a chain that pays at one length pays at every longer one, more slots never need a longer
    sequence, and the rule reads the mean length of the batch."""
    firsts = [first_paying(pieces, **rival) for pieces in SLOTS]
    for pieces, first in zip(SLOTS, firsts):
        assert all(chain_pays(pieces=pieces, num_seqs=1, total_tokens=tokens, **rival) == (tokens >= first) for tokens in LENGTHS), pieces
        for count in (2, 5):
            assert all(chain_pays(pieces=pieces, num_seqs=count, total_tokens=count * tokens, **rival) == (tokens >= first) for tokens in LENGTHS), (
                pieces,
                count,
            )
    assert firsts == sorted(firsts, reverse=True)
    assert firsts[-1] < NEVER


@pytest.mark.L0
def test_faster_rival_delays_the_chain():
    """The value-width split walks a sequence faster than the uncut plan: against it a chain of the same slots starts no
    earlier."""
    for pieces in SLOTS:
        assert first_paying(pieces, split=True) >= first_paying(pieces), pieces


@pytest.mark.L0
def test_chain_group_pays_for_long_batches_of_few_sequences():
    """At every slot count, head count and SM count: a batch that carries the chain group carries it with fewer sequences and
    on more SMs, and with more tokens for as long as the length rule cuts them into as many slots."""
    step, counts = LENGTH_RULE_PIECE_TOKENS // 4, (1, 2, 8, 32, 128)
    for pieces in SLOTS:
        ends = [slots * LENGTH_RULE_PIECE_TOKENS for slots in range(pieces)] + [64 * LENGTH_RULE_PIECE_TOKENS]
        for heads in (1, KEY_HEADS, VALUE_HEADS):
            for below, end in zip(ends, ends[1:]):
                shape = dict(pieces=pieces, heads_out=heads)
                carried = {
                    (num_sm, count): [
                        chain_group_pays(**shape, num_seqs=count, num_sm=num_sm, total_tokens=total) for total in range(below + step, end + 1, step)
                    ]
                    for num_sm in SM_COUNTS
                    for count in counts
                }
                for key, row in carried.items():
                    assert row == sorted(row), (pieces, heads, key)
                for num_sm in SM_COUNTS:
                    for fewer, more in zip(counts, counts[1:]):
                        assert all(a or not b for a, b in zip(carried[num_sm, fewer], carried[num_sm, more])), (pieces, heads, num_sm)
                for count in counts:
                    for fewer, more in zip(sorted(SM_COUNTS), sorted(SM_COUNTS)[1:]):
                        assert all(b or not a for a, b in zip(carried[fewer, count], carried[more, count])), (pieces, heads, count)


def first_chain_length(num_sm, *, count=1, **call):
    """The shortest length in whole chunks at which a batch of ``count`` equal sequences chains on ``num_sm`` SMs."""
    return next((tokens for tokens in LENGTHS if plan_scheme(num_sm, [tokens] * count, **call)[0]), None)


def around_the_first_chain_length(num_sm, **call):
    """The shared rule's first chain length, the last chunk below the forward's own first chain length and that length."""
    first = first_chain_length(num_sm, **call)
    count = call.pop("count", 1)
    lengths = {CHAIN_MIN_MEAN_TOKENS} if first is None else {CHAIN_MIN_MEAN_TOKENS, first - CHUNK, first}
    return [dict(lengths=[tokens] * count, **call) for tokens in sorted(lengths) if tokens >= CHUNK]


def tail_scheme(num_sm, lengths, *, heads, value_heads, **call):
    """The tail of the plan of one call by the package's own rules: ``choose_tail``'s, a chain tail only where it pays."""
    pieces, split = plan_scheme(num_sm, lengths, heads=heads, value_heads=value_heads, **call)
    if pieces or split or call.get("batch_invariant"):
        return None
    shape = dict(num_seqs=len(lengths), heads_out=max(heads, value_heads), num_sm=num_sm, total_tokens=sum(lengths))
    tail = choose_tail(**shape, b_t=CHUNK, cadence_tokens=call.get("cadence", 0), expand_num=1, dim_v=call.get("value_dim", WIDTH))
    by_cost = tail is not None and not tail.dv_split and call.get("in_place", True) and not call.get("context")
    if by_cost and not chain_pays(pieces=tail.pieces, num_seqs=len(lengths), total_tokens=sum(lengths), tail=True):
        return None
    return tail


def group_scheme(num_sm, lengths, *, heads, value_heads, **call):
    """The slots of the chain group the plan of one call carries, by the package's own rules: ``ragged_pieces``' for a plan
    that neither chains nor is ``batch_invariant``, and for an in-place GDN forward without a context port only where the
    group pays."""
    if plan_scheme(num_sm, lengths, heads=heads, value_heads=value_heads, **call)[0] or call.get("batch_invariant"):
        return 0
    shape = dict(heads_out=max(heads, value_heads), total_tokens=sum(lengths))
    slots = ragged_pieces(**shape, num_sm=num_sm)
    by_cost = call.get("in_place", True) and not call.get("context")
    if slots and by_cost and not chain_group_pays(pieces=slots, num_seqs=len(lengths), num_sm=num_sm, **shape):
        return 0
    return slots


def first_tail_length(num_sm, **call):
    """The shortest length in whole chunks at which ``tail_scheme`` hands the leftover tiles of four sequences to a chain."""
    for tokens in LENGTHS:
        tail = tail_scheme(num_sm, [tokens] * 4, **call)
        if tail is not None and not tail.dv_split:
            return tokens
    return None


def assert_plans_follow_the_rules(monkeypatch, num_sm, calls):
    """The plan of each call, built for a GPU of ``num_sm`` SMs, chains with the slots the rules say and carries their tail."""
    monkeypatch.setattr(gdn_engine, "multiprocessor_count", lambda device: num_sm)
    for call in calls:
        plan = build_plan(**call)
        pieces, split = plan_scheme(num_sm, **call)
        assert (plan.pieces, plan.chain, plan.dv_split) == (pieces, pieces > 0, split), (num_sm, call)
        if not call.get("summary") and call.get("householders", 1) == 1:
            heads = dict(heads=call.get("heads", KEY_HEADS), value_heads=call.get("value_heads", VALUE_HEADS))
            assert plan.tail == tail_scheme(num_sm, call["lengths"], **heads, **tail_forms(call)), (num_sm, call)
            assert plan.ragged_pieces == group_scheme(num_sm, call["lengths"], **heads, **tail_forms(call)), (num_sm, call)


def tail_forms(call):
    return {name: value for name, value in call.items() if name not in ("lengths", "heads", "value_heads")}


CODE_PATHS = (
    "in_place",
    "default_call",
    "narrow_value",
    "several_sequences",
    "chain_tail",
    "chain_group",
    "chain_beside_group",
    "batch_invariant",
    "context_port",
    "expanded_timeline",
    "summary_node",
)


def group_calls(num_sm, **heads):
    """Dense batches of sequences half the shared rule's first chain length, on both sides of the batch size from which the
    shared rule carries the chain group and of the one from which the forward carries it; and two long sequences of few
    heads, a batch that gains from the group."""
    tokens, sizes = CHAIN_MIN_MEAN_TOKENS // 2, range(1, 64 * CHAIN_MAX_PIECES)
    shared = [count for count in sizes if ragged_pieces(heads_out=max(heads.values()), num_sm=num_sm, total_tokens=count * tokens)]
    carried = [count for count in sizes if group_scheme(num_sm, [tokens] * count, **heads)]
    edges = {count for found in (shared[:1], carried[:1]) for first in found for count in (first - 1, first) if count > 0}
    long = CHAIN_MIN_PIECES * LENGTH_RULE_PIECE_TOKENS // 2 + CHUNK
    return [dict(lengths=[tokens] * count, **heads) for count in sorted(edges)] + [dict(lengths=[long] * 2, heads=1, value_heads=1, cadence=one_row(long))]


def calls_beside_the_group(num_sm):
    """Sequences of the shared rule's first chain length on one head: the largest batch whose plan below the chain carries no
    chain group, and one sequence more."""
    tokens = CHAIN_MIN_MEAN_TOKENS
    first = next(count for count in range(1, num_sm) if ragged_pieces(heads_out=1, num_sm=num_sm, total_tokens=count * tokens))
    return [dict(lengths=[tokens] * count, heads=1, value_heads=1) for count in (first - 1, first)]


def code_paths(num_sm):
    """The calls of each code path of the forward's chain decision on a GPU of ``num_sm`` SMs, as keyword arguments of
    ``build_plan``.  At one tile in three SMs the slot budget of one sequence is the chain's minimum and two thread blocks per
    head still fit in one wave; four sequences of that head count are one wave of tiles and a leftover third."""
    third = dict(heads=num_sm // CHAIN_MIN_PIECES, value_heads=num_sm // CHAIN_MIN_PIECES)
    tail_first = first_tail_length(num_sm, **third)
    tail_lengths = {CHAIN_MIN_MEAN_TOKENS} if tail_first is None else {CHAIN_MIN_MEAN_TOKENS, tail_first - CHUNK, tail_first}
    in_place = around_the_first_chain_length(num_sm, **third)
    default_call = around_the_first_chain_length(num_sm, in_place=False, **third)
    return {
        "in_place": in_place,
        "default_call": default_call,
        "narrow_value": around_the_first_chain_length(num_sm, value_dim=WIDTH // 2, **third),
        "several_sequences": around_the_first_chain_length(num_sm, count=2, heads=KEY_HEADS, value_heads=KEY_HEADS),
        "chain_tail": [dict(lengths=[tokens] * 4, **third) for tokens in sorted(tail_lengths)],
        "chain_group": group_calls(num_sm, **third),
        "chain_beside_group": calls_beside_the_group(num_sm),
        "batch_invariant": [dict(call, batch_invariant=True) for call in in_place],
        "context_port": [dict(call, context=True) for call in in_place],
        "expanded_timeline": [dict(call, householders=SUB_TOKENS) for call in in_place],
        "summary_node": [dict(call, summary=True) for call in in_place],
    }


@pytest.mark.L0
@pytest.mark.parametrize("path", CODE_PATHS)
def test_plan_follows_the_chain_rule(monkeypatch, path):
    """One cell per code path, on this GPU's SM count, at the shared rule's first chain length and on both sides of the
    forward's own: one sequence in place, a value width without the split, two sequences, and the chain tail of four; batches
    on both sides of the size from which a plan carries the chain group, without a chain and below one; the default call,
    ``batch_invariant``, a context port, GDP and the summary node answer the shared rule."""
    assert_plans_follow_the_rules(monkeypatch, la.sm_count(), code_paths(la.sm_count())[path])


@pytest.mark.L0
@pytest.mark.parametrize("fixed_tokens", [0, LENGTHS[-1]], ids=["chain_free", "chain_never_pays"])
def test_tuning_moves_only_the_cost_ruled_forward(monkeypatch, fixed_tokens):
    """With the chain's fixed part at nothing and beyond every length: the default call, ``batch_invariant``, a context port,
    GDP and the summary node keep the slots of the shared rule; the cost-ruled forward follows the rule's verdict."""
    monkeypatch.setattr(piece_chain, "CHAIN_FIXED_TOKENS", fixed_tokens)
    monkeypatch.setattr(piece_chain, "CHAIN_TAIL_FIXED_TOKENS", fixed_tokens)
    monkeypatch.setattr(piece_chain, "CHAIN_GROUP_TILE_TOKENS", fixed_tokens)
    num_sm = la.sm_count()
    monkeypatch.setattr(gdn_engine, "multiprocessor_count", lambda device: num_sm)
    tokens, heads = 4 * CHAIN_MIN_MEAN_TOKENS, num_sm // CHAIN_MIN_PIECES
    third = dict(heads=heads, value_heads=heads)
    shape = dict(num_seqs=1, heads_out=heads, num_sm=num_sm, total_tokens=tokens, b_t=CHUNK, cadence_tokens=0)
    shared = choose_pieces(**shape, batch_invariant=False, expand_num=1)[0]
    assert shared
    for form in (dict(in_place=False), dict(context=True), dict(summary=True), dict(householders=SUB_TOKENS)):
        slots = choose_pieces(**shape, batch_invariant=False, expand_num=form.get("householders", 1))[0]
        assert build_plan([tokens], **third, **form).pieces == slots, form
    assert build_plan([tokens], **third, batch_invariant=True).pieces == choose_pieces(**shape, batch_invariant=True, expand_num=1)[0]
    assert build_plan([tokens], **third).pieces == (shared if fixed_tokens == 0 else 0)


@pytest.mark.L1
@pytest.mark.parametrize("num_sm", SM_COUNTS)
def test_plans_follow_the_chain_rule_at_every_length(monkeypatch, num_sm):
    """The sweep on the SM counts of other GPUs: every code path; every whole chunk from the shared rule's first chain length
    to past the forward's own, for each call form and head count, for two to seven sequences and for the chain tail."""
    third = dict(heads=num_sm // CHAIN_MIN_PIECES, value_heads=num_sm // CHAIN_MIN_PIECES)
    forms = [dict(), dict(in_place=False), dict(value_dim=WIDTH // 2), dict(batch_invariant=True), dict(context=True)]
    forms += [dict(heads=heads, value_heads=heads) for heads in (1, KEY_HEADS, 2 * KEY_HEADS)]
    calls = [call for path in code_paths(num_sm).values() for call in path]
    starts = [first_chain_length(num_sm, **form) for form in forms] + [first_tail_length(num_sm, **third)]
    last = max(start for start in starts if start is not None) + 2 * CHUNK
    for tokens in range(CHAIN_MIN_MEAN_TOKENS - CHUNK, last + 1, CHUNK):
        calls += [dict(lengths=[tokens], **form) for form in forms]
        calls += [dict(lengths=[tokens] * count, heads=KEY_HEADS, value_heads=heads) for count in (2, 3, 4, 7) for heads in (KEY_HEADS, VALUE_HEADS)]
        calls += [dict(lengths=[tokens] * 4, **third)]
    for count in range(1, 3 * num_sm // KEY_HEADS):
        calls += [
            dict(lengths=[CHAIN_MIN_MEAN_TOKENS // 2] * count, heads=KEY_HEADS, value_heads=heads, **form)
            for heads in (KEY_HEADS, VALUE_HEADS)
            for form in (dict(), dict(in_place=False))
        ]
    assert_plans_follow_the_rules(monkeypatch, num_sm, calls)


@pytest.fixture
def frost_engine():
    """The byte tests run where a FROST engine serves the op; each builds its own plans."""
    reason = la.frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    la.clear_caches()
    try:
        yield
    finally:
        la.clear_caches()


def run_forward(inputs, *, in_place, **kwargs):
    """Output and final state of one GDN forward on ``inputs``; the in-place call hands back the state buffer it was given."""
    state = inputs["initial_state"].clone()
    operands = [inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], inputs["cu"]]
    out, final = la.la_ops.gated_delta_net(
        *operands, initial_state=state, output_final_state=True, overwrite_initial_state=in_place, plan_name="gdn_frost", **kwargs
    )
    assert (final is state) == in_place
    return snapshot({"out": out, "final_state": final})


def assert_forward_matches(tokens, *, in_place):
    """One sequence of ``tokens`` tokens at 16 key / 48 value heads: output and final state within the fp64 recurrence's
    tolerance, and bit for bit those of the ``batch_invariant`` twin where neither plan chains."""
    inputs = pack(sequences("gdn", (tokens,), heads=KEY_HEADS, value_heads=VALUE_HEADS, key_dim=WIDTH, value_dim=WIDTH))
    result = run_forward(inputs, in_place=in_place)
    check_reference("gdn", inputs, result, False)
    num_sm = la.sm_count()
    chains = plan_scheme(num_sm, [tokens], in_place=in_place)[0] or plan_scheme(num_sm, [tokens], in_place=in_place, batch_invariant=True)[0]
    if in_place and not chains:
        assert_bitwise(result, run_forward(inputs, in_place=True, batch_invariant=True), f"{tokens} tokens in place against the batch_invariant twin")


@pytest.mark.L0
def test_forward_below_the_chain_writes_the_uncut_plan_bytes(frost_engine):
    """One sequence in place at the shared rule's first chain length, where this GPU's forward does not chain: the bytes of
    the uncut plan, within the fp64 recurrence's tolerance."""
    if plan_scheme(la.sm_count(), [CHAIN_MIN_MEAN_TOKENS])[0]:
        pytest.skip(f"the forward chains at {CHAIN_MIN_MEAN_TOKENS} tokens on {la.sm_count()} SMs")
    assert_forward_matches(CHAIN_MIN_MEAN_TOKENS, in_place=True)


@pytest.mark.L0
def test_batch_invariant_bytes_do_not_follow_the_batch_across_the_chain_rule(frost_engine):
    """``batch_invariant``: a sequence of the shared rule's first chain length has the same bytes alone and beside a short
    neighbour, in either order."""
    samples = sequences("gdn", (CHAIN_MIN_MEAN_TOKENS, CHUNK + 1))

    def run(selected):
        inputs = pack(selected)
        return run_forward(inputs, in_place=False, batch_invariant=True), inputs["cu"]

    alone, _ = run(samples[:1])
    for order in ((0, 1), (1, 0)):
        batch, cu = run([samples[index] for index in order])
        at = order.index(0)
        part = {"out": batch["out"][int(cu[at]) : int(cu[at + 1])], "final_state": batch["final_state"][at : at + 1]}
        assert_bitwise(part, alone, f"batch order {order}")


@pytest.mark.L0
def test_batch_without_the_chain_group_writes_the_bytes_it_writes_with_it(frost_engine, monkeypatch):
    """A dense batch above the tokens from which the shared rule carries the chain group, at a size at which this GPU's
    forward does not: output and final state match the fp64 recurrence and are bit for bit those of the same call with the
    group carried, whose five launches find no work."""
    num_sm, tokens = la.sm_count(), CHAIN_MIN_MEAN_TOKENS // 2
    heads = dict(heads=KEY_HEADS, value_heads=VALUE_HEADS)
    count = next((n for n in range(2, 16 * CHAIN_MIN_PIECES) if ragged_pieces(heads_out=VALUE_HEADS, num_sm=num_sm, total_tokens=n * tokens)), None)
    if count is None or group_scheme(num_sm, [tokens] * count, **heads):
        pytest.skip(f"a batch of {tokens}-token sequences carries the chain group at every size on {num_sm} SMs")
    inputs = pack(sequences("gdn", (tokens,) * count, heads=KEY_HEADS, value_heads=VALUE_HEADS, key_dim=WIDTH, value_dim=WIDTH))
    without = run_forward(inputs, in_place=True)
    check_reference("gdn", inputs, without, False)
    la.clear_caches()
    monkeypatch.setattr(piece_chain, "CHAIN_GROUP_TILE_TOKENS", 0)
    assert group_scheme(num_sm, [tokens] * count, **heads)
    assert_bitwise(without, run_forward(inputs, in_place=True), f"{count} x {tokens} tokens, the chain group carried and not carried")


@pytest.mark.L1
@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "default_call"])
def test_both_sides_of_the_first_chain_length_match_the_reference(frost_engine, in_place):
    """One sequence at 16 key / 48 value heads on the last chunk below this GPU's first chain length and on that length."""
    first = first_chain_length(la.sm_count(), in_place=in_place)
    if first is None:
        pytest.skip(f"one sequence of {VALUE_HEADS} value heads does not chain on {la.sm_count()} SMs")
    for tokens in (first - CHUNK, first):
        assert_forward_matches(tokens, in_place=in_place)
