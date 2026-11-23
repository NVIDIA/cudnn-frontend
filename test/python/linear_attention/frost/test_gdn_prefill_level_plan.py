# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The levelled plan of the chunked GDN prefill kernel (``LEVEL_PLAN``).

The plan lays the (sequence, head) tiles end to end, in groups of the value heads of one key head, shares them out to the
thread blocks and runs a tile that crosses a share border as two work items, the second started from the state the first
stored.  It may not change a word of the output or of a state, a tile's second item may never be handed out in front of
the first, and the plan is taken only where the engine and the device can both tell that it applies."""

import random
import sys

import pytest
import torch

from cudnn import cpp_launcher
from cudnn.linear_attention.frost import gdn_engine
from cudnn.linear_attention.frost.common import split_k
from cudnn.linear_attention.frost.common import tail as tail_rule
from cudnn.linear_attention.ops import gdn as gdn_ops
from linear_attention.test_la import (
    FWD_TOL,
    STATE_TOL,
    Backend,
    assert_bitwise,
    assert_rms_close,
    clear_caches,
    frost_la_unavailable,
    make_case,
    op_args,
    random_state,
    reference,
    run_fwd,
    state_pool,
)

gdn_prefill_f16 = pytest.importorskip("cudnn.linear_attention.frost.kernel.gdn_prefill_f16")
gdn_warmup_forward_f16 = pytest.importorskip("cudnn.linear_attention.frost.kernel.gdn_warmup_forward_f16")

SEED_COPY_FORM = ("STATE_IO_COALESCED", "TILE_HANDOVER_EARLY", "STATE_SEED_COPY")
B_T = gdn_prefill_f16.CFG.B_T
UNWRITTEN = -(1 << 30)  # what a device case leaves in the prologue's tables in front of a call: no row, order entry or count reads so
# Shapes beside the measured cells: (tiles, chunks, SMs, heads per sequence, group).
SWEEP = [
    (n_seqs * heads, chunks, sms, heads, group)
    for sms in (2, 3, 7, 50, 148, 152, 170)
    for heads, group in ((1, 1), (3, 3), (4, 2), (48, 1), (48, 3), (48, 6))
    for n_seqs in (1, 2, 3, 4, 5, 7, 11, 16, 23)
    for chunks in (2, 3, 4, 5, 16, 17, 64)
]


def measured_shapes():
    """``(tiles, chunks, SMs, heads per sequence, group)`` of every measured cell."""
    cells = gdn_prefill_f16.LEVEL_MEASURED_CELLS
    return [(cell.sequences * cell.heads_out, cell.chunks, cell.sms, cell.heads_out, cell.heads_out // cell.heads_key) for cell in cells]


def swept_shapes():
    """The shapes of the sweep the rule takes."""
    return [shape for shape in SWEEP if split_k.level_plan_shape(*shape)]


# What a table case runs over: every measured cell, or the sweep around them.
TABLE_SHAPES = [pytest.param(measured_shapes, marks=pytest.mark.L0, id="measured_cells"), pytest.param(swept_shapes, marks=pytest.mark.L1, id="sweep")]


def line_layout(n_tiles, chunks, num_sms, group):
    """The levelled table from its definition, by laying the groups on a line of chunk slots and cutting it: ``(parts,
    order)`` with ``parts[row] = (tile, chunk_start, chunk_end)``.  A group's slots run from its last chunk to its first, so
    the slots at and above a border are the group's first chunks."""
    groups, shares = n_tiles // group, num_sms // group
    line = [g for g in range(groups) for _ in range(chunks)]
    q, rem = divmod(len(line), shares)
    opening, closing = [], []
    for b in range(shares):
        x = b * q + min(b, rem)
        first = sum(1 for slot in line[x : x + chunks] if slot == line[x])
        if x and line[x - 1] == line[x] and first % 2:
            x, first = x + 1, first - 1
        cut = x > 0 and line[x - 1] == line[x] if first else False
        g, first = (line[x], first) if first else (line[x], chunks)
        opening.append((g, first, cut, b))
        if cut:
            closing.append((chunks - first, b, g))
    rows = [(g, 0, first, b if cut else 0) for g, first, cut, b in opening]
    rows += [(g, 0, chunks, 0) for g in range(groups) if g not in {g for g, _, _, _ in opening}]
    rows += [(g, chunks - length, chunks, -b) for length, b, g in sorted(closing, key=lambda part: (-part[0], part[1]))]
    parts = [(g * group + s, start, end) for g, start, end, _ in rows for s in range(group)]
    order = [(1 if b > 0 else -1) * (abs(b) * group + s) if b else 0 for _, _, _, b in rows for s in range(group)]
    return parts, order


def order_problems(parts, order, n_tiles, chunks, num_sms, group):
    """What a levelled table must keep for the main kernel's order, as a list of what it breaks."""
    problems = []
    cover = {}
    for tile, start, end in parts:
        cover.setdefault(tile, []).append((start, end))
    for tile in range(n_tiles):
        pieces = sorted(cover.get(tile, []))
        if pieces != [(0, chunks)] and not (len(pieces) == 2 and pieces[0][0] == 0 < pieces[0][1] == pieces[1][0] < pieces[1][1] == chunks):
            problems.append(f"tile {tile} is not covered once, whole or in two parts that meet: {pieces}")
    for word in {abs(entry) for entry in order if entry}:
        setting = [row for row, entry in enumerate(order) if entry == word]
        waiting = [row for row, entry in enumerate(order) if entry == -word]
        if len(setting) != 1 or len(waiting) != 1:
            problems.append(f"level word {word}: {len(setting)} setting rows and {len(waiting)} waiting rows")
            continue
        (tile, start, end), held = parts[setting[0]], parts[waiting[0]]
        if not setting[0] < waiting[0]:
            problems.append(f"level word {word}: its waiting row is handed out in front of its setting row")
        if not 0 < word < num_sms:
            problems.append(f"level word {word} is outside the {num_sms} level words")
        if held != (tile, end, chunks) or start != 0:
            problems.append(f"level word {word}: its rows are not the two parts of one tile")
        if end - start < 1 or (end - start) % 2:
            problems.append(f"level word {word}: the setting row has {end - start} chunks")
    for row in range(0, len(parts), group):
        siblings = parts[row : row + group]
        if (
            [tile for tile, _, _ in siblings] != list(range(siblings[0][0], siblings[0][0] + group))
            or siblings[0][0] % group
            or len({part[1:] for part in siblings}) != 1
        ):
            problems.append(f"rows {row} .. {row + group - 1} are not one group cut at one chunk")
    if len(parts) > n_tiles + num_sms - 1:
        problems.append("more rows than the work-item region holds")
    return problems


@pytest.fixture
def level_form(monkeypatch):
    """The kernel's build switches of a level plan."""
    for switch in SEED_COPY_FORM + ("LEVEL_PLAN",):
        monkeypatch.setattr(gdn_prefill_f16, switch, True)


def facts_of(cell, **overrides):
    """The engine's facts of a batch that is the line ``cell``."""
    facts = gdn_engine.LevelFacts(
        io_name=cell.io,
        dim_k=cell.d_k,
        dim_v=cell.d_v,
        heads_out=cell.heads_out,
        heads_key=cell.heads_key,
        num_seqs=cell.sequences,
        rows=cell.sequences * cell.chunks * B_T,
        b_t=B_T,
        num_sm=cell.sms,
        compute_capability=cell.compute_capability,
    )
    return facts._replace(**overrides)


def cell_facts(**overrides):
    """The engine's facts of the first measured cell."""
    return facts_of(gdn_prefill_f16.LEVEL_MEASURED_CELLS[0], **overrides)


def line_of(facts):
    """The line the admission looks up for the engine's facts."""
    chunks = -(-(facts.rows // facts.num_seqs) // B_T)
    return gdn_prefill_f16.LevelCell(
        tuple(facts.compute_capability), facts.num_sm, facts.io_name, facts.dim_k, facts.dim_v, facts.heads_key, facts.heads_out, facts.num_seqs, chunks
    )


def engine_tail(cell, chunks=None):
    """The tail the engine's own rule gives the cell's sequences and heads (``chunks``: at another length), or None."""
    tokens = cell.sequences * (cell.chunks if chunks is None else chunks) * B_T
    return tail_rule.choose_tail(
        num_seqs=cell.sequences, heads_out=cell.heads_out, num_sm=cell.sms, total_tokens=tokens, b_t=B_T, cadence_tokens=0, expand_num=1, dim_v=cell.d_v
    )


def tail_cells():
    """``(cell, chunk counts beside it)`` of every measured cell the engine's own plan has a tail at: the counts one
    chunk longer or shorter that keep a tail and that no line names."""
    cells = gdn_prefill_f16.LEVEL_MEASURED_CELLS

    def beside(cell):
        return [chunks for chunks in (cell.chunks + 1, cell.chunks - 1) if engine_tail(cell, chunks) and cell._replace(chunks=chunks) not in cells]

    return [(cell, beside(cell)) for cell in cells if engine_tail(cell)]


# ---------------------------------------------------------------------------------------------------------------------------
# the table and its shape rule, on the host
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shapes", TABLE_SHAPES)
def test_levelled_table_is_the_line_of_groups_cut_at_even_borders(shapes):
    """Both tables -- the work items and the order entries -- of every measured cell, and of every shape of a sweep the
    rule takes, are the groups laid end to end and cut at the share borders, a border moved up where it would leave a
    first part of an odd number of chunks."""
    shapes = shapes()
    assert shapes
    for n_tiles, chunks, num_sms, _, group in shapes:
        assert split_k.level_table(n_tiles, chunks, num_sms, group) == line_layout(n_tiles, chunks, num_sms, group), (n_tiles, chunks, num_sms, group)


@pytest.mark.parametrize("shapes", TABLE_SHAPES)
def test_levelled_table_keeps_the_order(shapes):
    """One setting row above one waiting row per level word, the word inside the level words, every setting row with a
    chunk to store a state behind, the siblings of a group in adjacent rows: for every measured cell, and for every
    shape of the sweep the rule takes, by this module's own check and by the kernel's gate."""
    for n_tiles, chunks, num_sms, _, group in shapes():
        parts, order = split_k.level_table(n_tiles, chunks, num_sms, group)
        assert order_problems(parts, order, n_tiles, chunks, num_sms, group) == [], (n_tiles, chunks, num_sms, group)
        assert split_k.level_table_gate(parts, order, n_tiles, chunks, num_sms)
    for n_tiles, chunks, num_sms, _, group in measured_shapes():
        assert any(entry > 0 for entry in split_k.level_table(n_tiles, chunks, num_sms, group)[1]), "a measured cell cuts a tile"


def waiterless(parts, order, word, setting, waiting):
    return parts, [0 if entry == word else entry for entry in order]


def waiter_first(parts, order, word, setting, waiting):
    swap = {setting: waiting, waiting: setting}
    return [parts[swap.get(row, row)] for row in range(len(parts))], [order[swap.get(row, row)] for row in range(len(order))]


def chunkless_setter(parts, order, word, setting, waiting):
    edited = list(parts)
    edited[setting], edited[waiting] = (parts[setting][0], 0, 0), (parts[waiting][0], 0, parts[waiting][2])
    return edited, order


def word_outside(parts, order, word, setting, waiting):
    return parts, [entry + (1 if entry > 0 else -1) * 100000 if abs(entry) == word else entry for entry in order]


def word_set_twice(parts, order, word, setting, waiting):
    return parts, [word if row == order.index(0) else entry for row, entry in enumerate(order)]


@pytest.mark.L0
@pytest.mark.parametrize("edit", [waiterless, waiter_first, chunkless_setter, word_outside, word_set_twice], ids=lambda edit: edit.__name__)
def test_table_gate_refuses_a_table_that_breaks_the_order(edit):
    """The gate takes a measured cell's table and refuses it with one row edited: a waiting row whose word no row sets, a
    waiting row handed out in front of its setting row, a setting row without a chunk (it would never set), a word outside
    the level words, a word two rows set."""
    n_tiles, chunks, num_sms, _, group = measured_shapes()[0]
    parts, order = split_k.level_table(n_tiles, chunks, num_sms, group)
    assert split_k.level_table_gate(parts, order, n_tiles, chunks, num_sms)
    word = next(entry for entry in order if entry > 0)
    edited = edit(parts, order, word, order.index(word), order.index(-word))
    assert edited != (parts, order)
    assert not split_k.level_table_gate(*edited, n_tiles, chunks, num_sms)
    assert order_problems(*edited, n_tiles, chunks, num_sms, group)


# (tiles, chunks, SMs, heads per sequence, group): a shape the rule takes, then its neighbour that misses one condition.
SHAPE_REFUSALS = {
    "heads_not_a_multiple_of_the_group": ((768, 16, 152, 48, 3), (768, 16, 152, 48, 5)),
    "tiles_not_whole_sequences": ((768, 16, 152, 48, 3), (771, 16, 152, 48, 3)),
    "a_share_not_longer_than_a_chain": ((192, 16, 152, 48, 3), (153, 16, 152, 3, 3)),
    "one_chunk_chains": ((768, 2, 152, 48, 3), (768, 1, 152, 48, 3)),
    "no_partial_round": ((303, 16, 152, 3, 3), (300, 16, 152, 3, 3)),
    "fewer_blocks_than_a_group": ((9, 16, 6, 3, 3), (9, 16, 2, 3, 3)),
    "more_blocks_than_prologue_threads": ((4095, 16, split_k.ORDER_THREADS, 1, 1), (4095, 16, split_k.ORDER_THREADS + 1, 1, 1)),
    "more_tiles_than_the_uncut_writer_sorts": ((split_k.ORDER_CAPACITY, 16, 152, 1, 1), (split_k.ORDER_CAPACITY + 1, 16, 152, 1, 1)),
}


@pytest.mark.L0
@pytest.mark.parametrize("refusal", list(SHAPE_REFUSALS))
def test_shape_rule_refuses(refusal):
    """Each condition of the levelled table's layout: a shape that meets it is taken, its neighbour that misses it is not."""
    taken, refused = SHAPE_REFUSALS[refusal]
    assert split_k.level_plan_shape(*taken)
    assert not split_k.level_plan_shape(*refused)


@pytest.mark.L0
def test_shape_rule_asks_the_table_gate(monkeypatch):
    """The rule takes a shape only with a table that keeps the order: a measured cell whose table lost one setting row
    -- its waiting row would wait for ever -- is refused."""
    shape = measured_shapes()[0]
    assert split_k.level_plan_shape(*shape)
    level_table = split_k.level_table

    def without_one_setting_row(*args):
        parts, order = level_table(*args)
        word = next(entry for entry in order if entry > 0)
        return parts, [0 if entry == word else entry for entry in order]

    monkeypatch.setattr(split_k, "level_table", without_one_setting_row)
    assert not split_k.level_plan_shape(*shape)


# ---------------------------------------------------------------------------------------------------------------------------
# the admission, on the host
# ---------------------------------------------------------------------------------------------------------------------------


# The batches the admission is asked for around the cells: (SM counts beside the cells' own, sequence counts, chunk counts).
ADMISSION_BATCHES = [
    pytest.param(((148,), (3, 4, 5, 15, 16), (15, 16, 17)), marks=pytest.mark.L0, id="neighbours"),
    pytest.param(((144, 148, 150, 160, 170), range(1, 33), (1, 2, 4, 8, 15, 16, 17, 32, 64)), marks=pytest.mark.L1, id="sweep"),
]


@pytest.mark.parametrize("batches", ADMISSION_BATCHES)
def test_admission_takes_every_measured_cell_and_no_other_shape(level_form, batches):
    """Every measured cell is a shape the table can be laid out for and is admitted; of the sequence counts, lengths and
    SM counts around the cells exactly the cells are admitted."""
    other_sms, seq_counts, chunk_counts = batches
    cells = set(gdn_prefill_f16.LEVEL_MEASURED_CELLS)
    for shape, cell in zip(measured_shapes(), gdn_prefill_f16.LEVEL_MEASURED_CELLS):
        assert split_k.level_plan_shape(*shape)
        assert gdn_engine.level_plan_refusal(gdn_prefill_f16, facts_of(cell)) is None
    base = cell_facts()
    for sms in sorted({cell.sms for cell in cells} | set(other_sms)):
        for num_seqs in seq_counts:
            for chunks in chunk_counts:
                facts = base._replace(num_sm=sms, num_seqs=num_seqs, rows=num_seqs * chunks * B_T)
                assert (gdn_engine.level_plan_refusal(gdn_prefill_f16, facts) is None) == (line_of(facts) in cells), line_of(facts)


ADMISSION_REFUSALS = {
    "another_sm_count": (dict(num_sm=148), "cell"),
    "another_part_of_the_same_sm_count": (dict(compute_capability=(10, 3)), "cell"),
    "another_operand_type": (dict(io_name="float16"), "cell"),
    "the_same_tiles_as_other_sequences_and_heads": (dict(heads_out=96, heads_key=32, num_seqs=8, rows=8 * 16 * B_T), "cell"),
    "a_batch_no_line_names": (dict(num_seqs=15, rows=15 * 16 * B_T), "cell"),
    "another_key_width": (dict(dim_k=64), "cell"),
    "one_value_head_per_key_head": (dict(heads_key=48), "cell"),
    "value_heads_not_in_whole_groups": (dict(heads_key=5), "heads"),
    "a_total_that_does_not_divide": (dict(rows=16 * 16 * B_T + 1), "lengths"),
    "no_sequence": (dict(num_seqs=0, rows=0), "lengths"),
    "batch_invariant": (dict(batch_invariant=True), "batch_invariant"),
    "checkpoints": (dict(checkpoints=True), "checkpoints"),
    "sub_token_expansion": (dict(expand_num=2), "expand_num"),
    "ragged_pieces": (dict(ragged_pieces=True), "tail"),
    "a_chain_plan": (dict(scheme="chain"), "scheme"),
    "a_value_width_split": (dict(scheme="dv_split"), "scheme"),
    "another_node": (dict(gdn=False), "scheme"),
    "no_state_in_place": (dict(state_in_place=False), "state"),
    "a_folded_seed": (dict(folded_seed=True), "state"),
    "a_value_width_without_the_seed_copy": (dict(dim_v=64), "level_plan_form"),
}


@pytest.mark.L0
@pytest.mark.parametrize("refusal", list(ADMISSION_REFUSALS))
def test_admission_refuses(level_form, refusal):
    """A measured cell is admitted; the same cell with one fact changed keeps the uncut plan, for that fact's reason."""
    change, reason = ADMISSION_REFUSALS[refusal]
    assert gdn_engine.level_plan_refusal(gdn_prefill_f16, cell_facts()) is None
    assert gdn_engine.level_plan_refusal(gdn_prefill_f16, cell_facts(**change)) == reason


@pytest.mark.L0
def test_a_measured_cell_ranks_above_the_tail(level_form):
    """Where the engine's own plan runs a batch's last partial wave as a tail, a measured cell takes the levelled table
    in the tail's place.  The same sequences one chunk longer or shorter, a batch no line names, keep the tail, and
    ragged pieces refuse a cell as they refuse any batch."""
    cells = tail_cells()
    assert cells and all(beside for _, beside in cells)
    for cell, beside in cells:
        assert gdn_engine.level_plan_refusal(gdn_prefill_f16, facts_of(cell, tail=True)) is None
        assert gdn_engine.level_plan_refusal(gdn_prefill_f16, facts_of(cell, tail=True, ragged_pieces=True)) == "tail"
        for chunks in beside:
            assert gdn_engine.level_plan_refusal(gdn_prefill_f16, facts_of(cell, tail=True, rows=cell.sequences * chunks * B_T)) == "tail"


@pytest.mark.L0
@pytest.mark.parametrize("off", SEED_COPY_FORM + ("LEVEL_PLAN",))
def test_admission_needs_the_switch_and_the_seed_copy_form(monkeypatch, level_form, off):
    """With the level switch or one switch of the seed-copy form off no plan is levelled."""
    assert gdn_engine.level_plan_refusal(gdn_prefill_f16, cell_facts()) is None
    monkeypatch.setattr(gdn_prefill_f16, off, False)
    assert gdn_engine.level_plan_refusal(gdn_prefill_f16, cell_facts()) == "level_plan_form"


@pytest.mark.L0
def test_level_form_is_the_build_configs_seed_copy(monkeypatch, level_form):
    """The geometry of a level plan has one source: the form is taken exactly where ``build_cfg`` takes the seed copy."""
    import cutlass

    assert gdn_prefill_f16.level_plan_form(cutlass.BFloat16, 128, 128)
    real_build_cfg = gdn_prefill_f16.build_cfg
    asked = []

    def build_cfg(*args, **kwargs):
        cfg = real_build_cfg(*args, **kwargs)
        asked.append((kwargs["d_k"], kwargs["d_v"]))
        cfg.state_seed_copy = False
        return cfg

    monkeypatch.setattr(gdn_prefill_f16, "build_cfg", build_cfg)
    assert not gdn_prefill_f16.level_plan_form(cutlass.BFloat16, 128, 128)
    assert asked == [(128, 128)]


@pytest.mark.L0
def test_test_knobs_are_inert_without_the_test_switch(monkeypatch, level_form):
    """A build without ``LEVEL_TEST`` orders its cut tiles and takes neither a delay, a count nor a batch no line names,
    whatever the test knobs say; with the switch they act, and a forced admission still asks the shape rule."""
    import cutlass

    def build(level_plan=True):
        return gdn_prefill_f16.build_cfg(
            cutlass.BFloat16,
            cutlass.Float32,
            max_active_clusters=152,
            use_initial_state=True,
            store_final_state=True,
            d_k=128,
            d_v=128,
            level_plan=level_plan,
            level_group=3,
        )

    unlisted = cell_facts(num_seqs=15, rows=15 * 16 * B_T)
    served_key = gdn_prefill_f16.level_build_key(True, 3)
    served = build()
    assert (served.level_wait, served.level_stress, served.level_reload_count, served.level_group) == (1, (), False, 3)
    monkeypatch.setattr(gdn_prefill_f16, "LEVEL_UNORDERED", True)
    monkeypatch.setattr(gdn_prefill_f16, "LEVEL_STRESS", (1, 2, 3, 4))
    monkeypatch.setattr(gdn_prefill_f16, "LEVEL_FORCE", True)
    monkeypatch.setattr(gdn_prefill_f16, "LEVEL_RELOAD_COUNT", True)
    assert gdn_prefill_f16.level_build_key(True, 3) == served_key
    assert build() == served
    assert gdn_engine.level_plan_refusal(gdn_prefill_f16, unlisted) == "cell"
    monkeypatch.setattr(gdn_prefill_f16, "LEVEL_TEST", True)
    tested = build()
    assert (tested.level_wait, tested.level_stress) == (0, (1, 2, 3, 4))
    assert gdn_prefill_f16.level_build_key(True, 3) != served_key
    assert gdn_engine.level_plan_refusal(gdn_prefill_f16, unlisted) is None
    # the forced admission lifts the measured cells alone: a batch the table cannot be laid out for is still refused
    assert gdn_engine.level_plan_refusal(gdn_prefill_f16, cell_facts(num_seqs=1, rows=16 * B_T)) == "shape"
    assert build(level_plan=False).level_wait == 0
    monkeypatch.setattr(gdn_prefill_f16, "LEVEL_UNORDERED", False)
    monkeypatch.setattr(gdn_prefill_f16, "LEVEL_STRESS", ())
    assert build().level_reload_count and build().level_wait == 1


@pytest.mark.L0
def test_a_build_that_holds_a_row_sets_its_word(level_form):
    """The load warp of a level build holds a tile's second item for a level word; a compute group that would not compile
    the set of that word -- no early hand-over, no stored final state -- does not build."""
    import cutlass

    def build(**kwargs):
        return gdn_prefill_f16.build_cfg(cutlass.BFloat16, cutlass.Float32, max_active_clusters=152, use_initial_state=True, d_k=128, d_v=128, **kwargs)

    level = build(store_final_state=True, level_plan=True, level_group=3)
    assert gdn_prefill_f16.level_orders(level) and gdn_prefill_f16.level_sets(level, True)
    with pytest.raises(ValueError, match="early_handover"):
        gdn_prefill_f16.level_sets(level, False)
    uncut = build(store_final_state=True)
    assert not gdn_prefill_f16.level_orders(uncut) and not gdn_prefill_f16.level_sets(uncut, False)
    with pytest.raises(ValueError, match="level_plan"):
        build(store_final_state=False, level_plan=True, level_group=3)


# ---------------------------------------------------------------------------------------------------------------------------
# the hand-out of the rows: no thread block waits for ever
# ---------------------------------------------------------------------------------------------------------------------------


def hand_out_ends(order, n_blocks, resident, rng, late_ticket=True):
    """Whether a hand-out of the table's rows to ``n_blocks`` thread blocks, at most ``resident`` of them running, ends
    with every row run.  Per block three roles as the main kernel has them.  The load warp publishes a ticket once its
    consumers took the one two back, holds a row with a negative order entry until its level word is set, then issues the
    item's loads.  Compute group 1 runs the item and stores its state; an item whose entry is positive then sets its word
    and takes its next ticket behind the set (``late_ticket``), every other item takes that ticket in front of its store.
    The other consumers start an item behind the store of the one in front and take their ticket at the item's end.
    Every precondition is monotone, so a hand-out that stops stops at any pace; ``rng`` picks which blocks step."""
    rows = len(order)
    ticket, words = [0], set()
    drawn = [[] for _ in range(n_blocks)]
    load = [[0, "publish"] for _ in range(n_blocks)]  # the role's item and stage
    group = [[0, "ticket"] for _ in range(n_blocks)]
    others = [[0, "ticket"] for _ in range(n_blocks)]
    released, loaded = [0] * n_blocks, [0] * n_blocks
    group_tickets, group_ended, group_stored = [0] * n_blocks, [0] * n_blocks, [0] * n_blocks
    other_tickets, other_items = [0] * n_blocks, [0] * n_blocks

    def step_load(b):
        item, stage = load[b]
        if stage == "publish" and (item < 2 or (group_tickets[b] >= item - 1 and other_tickets[b] >= item - 1)):
            drawn[b].append(ticket[0])
            ticket[0] += 1
            load[b][1] = "release"
            return True
        if stage == "release":
            row = drawn[b][item]
            if row >= rows or order[row] >= 0 or -order[row] in words:
                released[b] = item + 1
                load[b][1] = "done" if row >= rows else "loads"
                return True
        if stage == "loads" and other_items[b] >= item:
            loaded[b] = item + 1
            load[b] = [item + 1, "publish"]
            return True
        return False

    def step_group(b):
        item, stage = group[b]
        if stage == "ticket":
            if released[b] >= 1:
                group_tickets[b] = 1
                group[b][1] = "item" if drawn[b][0] < rows else "done"
                return True
            return False
        if stage == "done":
            return False
        entry = order[drawn[b][item]]
        hands_over = entry > 0 and late_ticket
        if stage == "item" and loaded[b] > item:
            group_ended[b] = item + 1
            group[b][1] = "store" if hands_over else "early"
            return True
        if stage in ("early", "late") and released[b] >= item + 2:
            group_tickets[b] = item + 2
            group[b][1] = "store" if stage == "early" else "next"
            return True
        if stage == "store":
            group_stored[b] = item + 1
            if entry > 0:
                words.add(entry)
            group[b][1] = "late" if hands_over else "next"
            return True
        if stage == "next":
            group[b] = [item + 1, "item" if drawn[b][item + 1] < rows else "done"]
            return True
        return False

    def step_others(b):
        item, stage = others[b]
        if stage == "ticket" and released[b] >= 1:
            other_tickets[b] = 1
            others[b][1] = "item" if drawn[b][0] < rows else "done"
            return True
        if stage == "item" and loaded[b] > item and group_ended[b] > item and group_stored[b] >= item:
            other_items[b] = item + 1
            others[b][1] = "end"
            return True
        if stage == "end" and released[b] >= item + 2:
            other_tickets[b] = item + 2
            others[b] = [item + 1, "item" if drawn[b][item + 1] < rows else "done"]
            return True
        return False

    def step(b):
        roles = [step_load, step_group, step_others]
        rng.shuffle(roles)
        return any(role(b) for role in roles)

    running, started = [], 0
    while True:
        while len(running) < resident and started < n_blocks:
            running.append(started)
            started += 1
        if not running:
            return True
        moved = [step(b) for b in running if rng.random() < 0.6]
        if not any(moved) and not any([step(b) for b in running]):
            return False
        running = [b for b in running if not (load[b][1] == group[b][1] == others[b][1] == "done")]


# The hand-outs a case runs: (the measured cells it takes, seeds per hand-out).
HAND_OUTS = [
    pytest.param((lambda: [min(measured_shapes(), key=lambda shape: shape[0] * shape[1])], 1), marks=pytest.mark.L0, id="smallest_cell"),
    pytest.param((measured_shapes, 3), marks=pytest.mark.L1, id="every_cell"),
]


@pytest.mark.parametrize("hand_outs", HAND_OUTS)
def test_no_hand_out_of_a_levelled_table_waits_for_ever(hand_outs):
    """Every hand-out of a measured cell's rows ends, with every block running and with one, two or half of them: a
    setting row stands above its waiting row and an item that hands its state over takes its next ticket behind its
    set.  The same hand-outs with that ticket taken in front of the store do stop: a block's load warp holds the very
    publication its compute group waits for in front of the set."""
    cells, seeds = hand_outs
    stopped_without_the_late_ticket = 0
    for n_tiles, chunks, num_sms, _, group in cells() + [(9, 16, 6, 3, 3), (3, 16, 2, 1, 1), (192, 16, 152, 48, 1)]:
        _, order = split_k.level_table(n_tiles, chunks, num_sms, group)
        for resident in sorted({1, 2, max(1, num_sms // 2), num_sms}):
            for seed in range(seeds):
                assert hand_out_ends(order, num_sms, resident, random.Random(seed)), (n_tiles, chunks, num_sms, group, resident, seed)
                stopped_without_the_late_ticket += not hand_out_ends(order, num_sms, resident, random.Random(seed), late_ticket=False)
    assert stopped_without_the_late_ticket > 0


# ---------------------------------------------------------------------------------------------------------------------------
# on the device
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def frost(monkeypatch):
    """``(backend, built, regions)``: the FROST backend, the prefill configs the test's calls build and, per call, what
    the prologue left in the regions the forward host was handed (the item count, the work items, the order entries, the
    scheduler words).  The count, the work items and the order entries read ``UNWRITTEN`` in front of every call, so a
    region shows what that call's prologue wrote.  The host cache starts empty and every call runs the op, not a
    recording of an earlier one."""
    reason = frost_la_unavailable()
    if reason:
        pytest.skip(reason)
    built, regions = [], []
    real_build_cfg = gdn_prefill_f16.build_cfg
    real_run = gdn_warmup_forward_f16.run_warmup_forward

    def build_cfg(*args, **kwargs):
        cfg = real_build_cfg(*args, **kwargs)
        built.append(cfg)
        return cfg

    def run_warmup_forward(compiled, facts, **buffers):
        for name in ("work_count", "work_items", "level_order"):
            if buffers.get(name) is not None:
                torch.from_dlpack(buffers[name]).fill_(UNWRITTEN)
        torch.cuda.synchronize()
        real_run(compiled, facts, **buffers)
        torch.cuda.synchronize()

        def words(name):
            return None if buffers.get(name) is None else torch.from_dlpack(buffers[name]).clone().cpu()

        regions.append({name: words(name) for name in ("work_count", "work_items", "level_order", "scheduler")})

    monkeypatch.setattr(gdn_prefill_f16, "build_cfg", build_cfg)
    monkeypatch.setattr(gdn_warmup_forward_f16, "run_warmup_forward", run_warmup_forward)
    monkeypatch.setattr(gdn_warmup_forward_f16, "warmup_forward_cache", {})
    monkeypatch.setattr(cpp_launcher, "HOT_CALLS", sys.maxsize)
    clear_caches()
    yield Backend("frost"), built, regions
    clear_caches()


def set_level(monkeypatch, on):
    """The seed-copy form with the level plan on or off."""
    for switch in SEED_COPY_FORM:
        monkeypatch.setattr(gdn_prefill_f16, switch, True)
    monkeypatch.setattr(gdn_prefill_f16, "LEVEL_PLAN", on)
    clear_caches()


def smallest_cell(cells=None):
    """The measured cell with the fewest chunks to run (of ``cells``)."""
    return min(cells or gdn_prefill_f16.LEVEL_MEASURED_CELLS, key=lambda cell: cell.sequences * cell.chunks)


def cell_case(cell=None, seq_lens=None):
    """``(case, cell)``: a batch of a measured cell, the smallest unless one is named (``seq_lens``: other lengths of
    its sequences)."""
    cell = cell or smallest_cell()
    seq_lens = [cell.chunks * B_T] * cell.sequences if seq_lens is None else seq_lens
    return make_case("gdn", getattr(torch, cell.io), seq_lens=seq_lens, H=cell.heads_key, HV=cell.heads_out, K=cell.d_k, V=cell.d_v, lo=0.9), cell


def plan_for(monkeypatch, cell):
    """The engine plans for the cell's part, whatever part the device is."""
    monkeypatch.setattr(gdn_engine, "multiprocessor_count", lambda device: cell.sms)
    monkeypatch.setattr(gdn_engine, "compute_capability", lambda device: cell.compute_capability)


def pooled_forward(backend, case, packed, indices=None, cu=None):
    """The op on a fresh pool holding ``packed`` (``cu``: other cumulative lengths over the case's tokens): ``(o, pool, indices)``."""
    pool, table, _ = state_pool(case, packed)
    indices = table if indices is None else indices
    if indices is not table:
        pool[indices.long()] = packed
    o, final_state = run_fwd(backend, case, cu=cu, initial_state=pool, state_indices=indices, output_final_state=True)
    assert final_state is pool
    torch.cuda.synchronize()
    return o, pool, indices


def device_table(regions, case):
    """The last call's ``(item count, parts, order entries, level words)`` as the prologue wrote them: every row and
    order entry below the count, and nothing at or above it."""
    last = regions[-1]
    count = int(last["work_count"][0])
    assert 0 <= count <= last["work_items"].shape[0]
    for name in ("work_items", "level_order"):
        if last[name] is not None:
            assert not bool((last[name][:count] == UNWRITTEN).any()), f"{name}: a row below the item count {count} was not written"
            assert bool((last[name][count:] == UNWRITTEN).all()), f"{name}: a row at or above the item count {count} was written"
    rows = last["work_items"][:count].tolist()
    parts = [(row[0] * case.HO + row[1], row[2], row[3]) for row in rows]
    order = None if last["level_order"] is None else last["level_order"][:count].tolist()
    return count, parts, order, last["scheduler"][split_k.LEVEL_WORD_BASE :].tolist()


@pytest.mark.L0
def test_levelled_plan_keeps_the_uncut_plans_bytes(monkeypatch, frost):
    """A measured cell planned for its SM count: the prologue writes the levelled table -- both tables as the host form
    has them -- every level word a row sets reads the lanes of compute group 1 behind the launch and no other word is
    touched, and output and pool are bitwise the uncut plan's and agree with the fp64 reference."""
    backend, built, regions = frost
    case, cell = cell_case()
    group = case.HO // case.HK
    plan_for(monkeypatch, cell)
    packed = random_state(case)
    set_level(monkeypatch, False)
    o_off, pool_off, indices = pooled_forward(backend, case, packed)
    assert not any(cfg.level_plan for cfg in built)
    count_off, _, order_off, _ = device_table(regions, case)
    assert count_off == case.N * case.HO and order_off is None
    set_level(monkeypatch, True)
    o_on, pool_on, _ = pooled_forward(backend, case, packed)
    assert any(cfg.level_plan and cfg.level_wait == 1 and cfg.level_group == group for cfg in built)
    count, parts, order, words = device_table(regions, case)
    host_parts, host_order = split_k.level_table(case.N * case.HO, cell.chunks, cell.sms, group)
    assert (count, parts, order) == (len(host_parts), host_parts, host_order)
    lanes = gdn_prefill_f16.level_set_lanes(next(cfg for cfg in built if cfg.level_plan))
    assert words == [lanes if word in {entry for entry in host_order if entry > 0} else 0 for word in range(cell.sms)]
    assert_bitwise("o[level plan]", o_on, o_off)
    assert_bitwise("pool[level plan]", pool_on, pool_off)
    o_ref, fs_ref = reference(case, initial_state=packed)
    assert_rms_close("o", o_on, o_ref, FWD_TOL[case.dtype])
    assert_rms_close("final_state", pool_on[indices.long()], fs_ref, STATE_TOL[case.dtype])


@pytest.mark.L0
def test_levelled_plan_of_partial_last_chunks_keeps_the_uncut_plans_bytes(monkeypatch, frost):
    """A measured cell's sequences one token short of their last chunk: the same chunk count, so the same line and the
    same levelled table, and every tile -- a cut tile's second item included -- ends in a chunk that is not full.
    Output and pool are bitwise the uncut plan's."""
    backend, built, regions = frost
    cell = smallest_cell()
    case, _ = cell_case(cell, seq_lens=[cell.chunks * B_T - 1] * cell.sequences)
    plan_for(monkeypatch, cell)
    packed = random_state(case)
    set_level(monkeypatch, False)
    o_off, pool_off, _ = pooled_forward(backend, case, packed)
    set_level(monkeypatch, True)
    o_on, pool_on, _ = pooled_forward(backend, case, packed)
    assert any(cfg.level_plan for cfg in built)
    count, parts, order, _ = device_table(regions, case)
    host_parts, host_order = split_k.level_table(case.N * case.HO, cell.chunks, cell.sms, case.HO // case.HK)
    assert (count, parts, order) == (len(host_parts), host_parts, host_order)
    assert_bitwise("o[partial last chunks]", o_on, o_off)
    assert_bitwise("pool[partial last chunks]", pool_on, pool_off)


@pytest.mark.L0
def test_levelled_plan_is_taken_on_a_measured_grid_only(monkeypatch, frost):
    """The same batch planned for the device itself: the plan is levelled exactly where the device is a measured cell's
    part -- its compute capability and its SM count -- and output and pool are the uncut plan's either way."""
    backend, built, regions = frost
    case, cell = cell_case()
    device = torch.cuda.current_device()
    part = cell._replace(compute_capability=tuple(gdn_engine.compute_capability(device)), sms=gdn_engine.multiprocessor_count(device))
    measured = part in gdn_prefill_f16.LEVEL_MEASURED_CELLS
    packed = random_state(case)
    set_level(monkeypatch, False)
    o_off, pool_off, _ = pooled_forward(backend, case, packed)
    set_level(monkeypatch, True)
    o_on, pool_on, _ = pooled_forward(backend, case, packed)
    assert any(cfg.level_plan for cfg in built) == measured
    assert (device_table(regions, case)[0] > case.N * case.HO) == measured
    assert_bitwise("o[device grid]", o_on, o_off)
    assert_bitwise("pool[device grid]", pool_on, pool_off)


# what the op is asked for beside the state pool, each a fact the admission refuses a level plan for
REFUSED_PLANS = {"batch_invariant": dict(batch_invariant=True), "checkpoints": dict(checkpoint_every_n_tokens=4 * B_T)}


@pytest.mark.L0
@pytest.mark.parametrize("plan", list(REFUSED_PLANS))
def test_a_plan_the_admission_refuses_builds_no_level_plan(monkeypatch, frost, plan):
    """A measured cell planned for its SM count, asked for with one thing the admission refuses -- a schedule that may
    not depend on the batch, a checkpoint cadence: the engine tells the admission so, no level build is made and the
    prologue writes the uncut table."""
    backend, built, regions = frost
    case, cell = cell_case()
    plan_for(monkeypatch, cell)
    pool, indices, _ = state_pool(case, random_state(case))
    set_level(monkeypatch, True)
    run_fwd(backend, case, initial_state=pool, state_indices=indices, output_final_state=True, **REFUSED_PLANS[plan])
    torch.cuda.synchronize()
    assert built and not any(cfg.level_plan for cfg in built)
    count, _, order, _ = device_table(regions, case)
    assert count == case.N * case.HO and order is None


@pytest.mark.L0
def test_levelled_plan_takes_the_tails_place_at_a_measured_cell(monkeypatch, frost):
    """A measured cell whose tiles leave a partial last wave, planned for its part.  With the switches as they are
    committed, and with the seed-copy form alone, the engine runs that wave as a tail: the main launch's table stops at
    the full waves.  With the level switch on the whole batch runs levelled -- both tables as the host form has them
    -- and output and pool are bitwise those of the plan without a tail."""
    backend, built, regions = frost
    cell = smallest_cell([cell for cell, _ in tail_cells()])
    case, _ = cell_case(cell)
    plan_for(monkeypatch, cell)
    n_tiles = case.N * case.HO
    full_waves = n_tiles // cell.sms * cell.sms
    packed = random_state(case)
    pooled_forward(backend, case, packed)
    assert device_table(regions, case)[0] == full_waves
    waves = tail_rule.TAIL_MAX_WAVES
    monkeypatch.setattr(tail_rule, "TAIL_MAX_WAVES", 0)
    set_level(monkeypatch, False)
    o_uncut, pool_uncut, _ = pooled_forward(backend, case, packed)
    assert device_table(regions, case)[0] == n_tiles
    monkeypatch.setattr(tail_rule, "TAIL_MAX_WAVES", waves)
    set_level(monkeypatch, False)
    pooled_forward(backend, case, packed)
    assert device_table(regions, case)[0] == full_waves
    assert not any(cfg.level_plan for cfg in built)
    set_level(monkeypatch, True)
    o_on, pool_on, _ = pooled_forward(backend, case, packed)
    assert any(cfg.level_plan for cfg in built)
    count, parts, order, _ = device_table(regions, case)
    host_parts, host_order = split_k.level_table(n_tiles, cell.chunks, cell.sms, case.HO // case.HK)
    assert (count, parts, order) == (len(host_parts), host_parts, host_order)
    assert_bitwise("o[level plan in the tail's place]", o_on, o_uncut)
    assert_bitwise("pool[level plan in the tail's place]", pool_on, pool_uncut)


@pytest.mark.L0
def test_a_batch_beside_a_measured_cell_keeps_its_tail(monkeypatch, frost):
    """The sequences of a measured cell with a tail, one chunk longer or shorter -- a batch no line names: the level
    switch changes nothing.  The engine builds no level plan, runs the tail and returns the same bytes."""
    backend, built, regions = frost
    cell, beside = min(tail_cells(), key=lambda entry: entry[0].sequences * entry[0].chunks)
    case, _ = cell_case(cell, seq_lens=[beside[0] * B_T] * cell.sequences)
    plan_for(monkeypatch, cell)
    full_waves = case.N * case.HO // cell.sms * cell.sms
    packed = random_state(case)
    set_level(monkeypatch, False)
    o_off, pool_off, _ = pooled_forward(backend, case, packed)
    assert device_table(regions, case)[0] == full_waves
    set_level(monkeypatch, True)
    o_on, pool_on, _ = pooled_forward(backend, case, packed)
    assert not any(cfg.level_plan for cfg in built)
    assert device_table(regions, case)[0] == full_waves
    assert_bitwise("o[beside a cell]", o_on, o_off)
    assert_bitwise("pool[beside a cell]", pool_on, pool_off)


def unequal(lens):
    return [lens[0] - B_T, lens[1] + B_T] + lens[2:]


def first_empty(lens):
    return [0, lens[0] + lens[1]] + lens[2:]


def last_empty(lens):
    return lens[:-2] + [lens[-2] + lens[-1], 0]


@pytest.mark.L0
@pytest.mark.parametrize("lengths", [unequal, first_empty, last_empty], ids=lambda lengths: lengths.__name__)
def test_other_lengths_take_the_uncut_table(monkeypatch, frost, lengths):
    """A batch with a measured cell's total but sequences of other chunk counts -- two lengths, an empty first sequence,
    an empty last one -- builds the level plan (the shapes cannot tell) and its prologue writes the uncut table: the
    chains as items, no order entry, no level word, and the uncut plan's bytes.  The levelled table would cut every chain
    where the chunk count the token rows give says."""
    backend, built, regions = frost
    even, cell = cell_case()
    case, _ = cell_case(seq_lens=lengths([int(n) for n in (even.cu[1:] - even.cu[:-1]).tolist()]))
    assert int(case.cu[-1]) == int(even.cu[-1])
    plan_for(monkeypatch, cell)
    packed = random_state(case)
    set_level(monkeypatch, False)
    o_off, pool_off, _ = pooled_forward(backend, case, packed)
    _, parts_off, _, _ = device_table(regions, case)
    set_level(monkeypatch, True)
    o_on, pool_on, _ = pooled_forward(backend, case, packed)
    assert any(cfg.level_plan for cfg in built)
    count, parts, order, words = device_table(regions, case)
    assert (count, parts) == (case.N * case.HO, parts_off)
    assert not any(order) and not any(words)
    assert_bitwise(f"o[{lengths.__name__}]", o_on, o_off)
    assert_bitwise(f"pool[{lengths.__name__}]", pool_on, pool_off)


@pytest.mark.L0
def test_lengths_short_of_the_token_rows_take_the_uncut_table(monkeypatch, frost):
    """Cumulative lengths that stop half way through the token rows: the engine admits the cell the rows give, the
    sequences have another chunk count -- a shape no measured cell names -- and the prologue writes the uncut table.
    The output rows the lengths name and the pool are the uncut plan's."""
    backend, built, regions = frost
    case, cell = cell_case()
    plan_for(monkeypatch, cell)
    packed = random_state(case)
    cu = (case.cu // 2).contiguous()
    rows = int(cu[-1])
    set_level(monkeypatch, False)
    o_off, pool_off, _ = pooled_forward(backend, case, packed, cu=cu)
    _, parts_off, _, _ = device_table(regions, case)
    set_level(monkeypatch, True)
    o_on, pool_on, _ = pooled_forward(backend, case, packed, cu=cu)
    assert any(cfg.level_plan for cfg in built)
    count, parts, order, words = device_table(regions, case)
    assert (count, parts) == (case.N * case.HO, parts_off)
    assert not any(order) and not any(words)
    assert_bitwise("o[short lengths]", o_on[:rows], o_off[:rows])
    assert_bitwise("pool[short lengths]", pool_on, pool_off)


@pytest.mark.L0
def test_a_pool_row_named_twice_takes_the_uncut_table(monkeypatch, frost):
    """Two sequences that name one pool row store the same row: a cut tile's second item could start from the other
    sequence's state.  The prologue writes the uncut table for such a batch, with no order entry of the levelled call in
    front of it left among its rows, and the levelled one again for the same batch with every row named once."""
    backend, built, regions = frost
    case, cell = cell_case()
    plan_for(monkeypatch, cell)
    packed = random_state(case)
    set_level(monkeypatch, True)
    _, _, indices = pooled_forward(backend, case, packed)
    assert any(cfg.level_plan for cfg in built)
    levelled = device_table(regions, case)
    assert levelled[0] > case.N * case.HO and any(levelled[2])
    twice = indices.clone()
    twice[-1] = twice[0]
    pooled_forward(backend, case, packed, twice)
    count, parts, order, words = device_table(regions, case)
    assert count == case.N * case.HO and sorted(parts) == [(tile, 0, cell.chunks) for tile in range(count)]
    assert not any(order) and not any(words)
    pooled_forward(backend, case, packed, indices)
    assert device_table(regions, case) == levelled


@pytest.mark.L0
def test_two_state_buffers_run_the_uncut_build(monkeypatch, frost):
    """The final state bound to another buffer than the seed: a cut tile's second item would start from the seed buffer,
    which its first item did not write.  The engine runs the uncut build then, and the level build when one buffer is
    bound for both; either gives the uncut plan's bytes."""
    backend, built, regions = frost
    case, cell = cell_case()
    plan_for(monkeypatch, cell)
    packed = random_state(case)
    scale = case.K**-0.5

    def forward(two_buffers):
        seed = packed.clone()
        final = torch.zeros_like(seed) if two_buffers else seed
        builds = len(built)
        o = gdn_ops.run_gdn_fwd(*op_args(case), scale, initial_state=seed, output_final_state=True, final_state_out=final, plan_name=backend.plan("gdn"))[0]
        torch.cuda.synchronize()
        assert_bitwise("seed", seed, packed if two_buffers else final)
        return o, final, [cfg.level_plan for cfg in built[builds:]]

    set_level(monkeypatch, False)
    o_off, final_off, _ = forward(True)
    set_level(monkeypatch, True)
    o_two, final_two, level_two = forward(True)
    assert not any(level_two) and device_table(regions, case)[0] == case.N * case.HO
    o_one, final_one, level_one = forward(False)
    assert any(level_one) and device_table(regions, case)[0] > case.N * case.HO
    for name, o, final in (("two buffers", o_two, final_two), ("one buffer", o_one, final_one)):
        assert_bitwise(f"o[{name}]", o, o_off)
        assert_bitwise(f"final_state[{name}]", final, final_off)
