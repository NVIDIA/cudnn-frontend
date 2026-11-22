# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the row argmax decode step (``cudnn.RowArgmaxDecode``), raw bytes throughout: bitwise replay into fresh
buffers under side-stream contention at 1, 17, 300 and 1024 rows; a fresh entry on fresh identical operands; a launch on a side
stream; batch invariance of every row -- alone (a one-row call) against the same row inside the first 2 .. 1024 rows of ONE
operand set (on both sides of the part's first four scheduling-wave edges), inside the reversed batch, after an
insertion, a removal and a permutation of its neighbours, duplicated inside one call, and with every other row redrawn; the
inputs unmodified by a launch; a change of ONE element moving the result of its row and of no other; the row ceiling call; the
row ceiling and the empty call as typed declines; and the goldens (four small cases, recorded with
``CUDNN_DETERMINISM_RECORD=<new dir>`` only after the replay and the reference pass).

DIRECTIONS.  The step is an INFERENCE step: forward only.  It has no backward entry, no gradient and no state, so this module
has no backward cell -- that direction does not exist by contract; it is not a coverage gap of an existing path.  The result is
the one vector a launch writes, ``out``.  The entry owns no scratch.  An empty call (zero rows) is not served: it is a typed
decline, pinned below, so there is no empty-sequence cell.

WHAT IS HELD FIXED.  A row's result is a function of the row's values and of the table: the op is exact, there is no reduction
whose order could matter, no split, no pick and no knob, so there is nothing to pin across batch layouts besides the operands.
The cells build the operands ONCE and slice / permute them (the row under test is never redrawn) and keep the one table.  Every
output is a fresh caller-owned vector filled with a sentinel, never the entry-owned one, so a reused buffer cannot make a
comparison pass.  The independent reference (``row_argmax_decode_utils.assert_rule``: every byte) is a separate check and gates
the goldens."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from row_argmax_decode_utils import (  # noqa: E402
    I64,
    RESULT,
    TABLES,
    WIDTH,
    _device_or_skip,
    assert_rule,
    make_entry,
    make_rows,
    make_table,
    rule_rows,
)

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot  # noqa: E402

from cudnn.row_argmax.frost.kernel import row_argmax_decode_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

MAX_ROWS = kcfg.MAX_ROWS
SENTINEL = -0x5A5A5A5A
#: the row counts a batch-invariance cell runs as ONE call besides the row ceiling and the part's scheduling-wave edges
INSIDE = (2, 3, 8, 17, 64)


def run(k, x, ids):
    """One launch into a FRESH caller-owned vector filled with the sentinel.  Returns ``{"out"}``."""
    table = "none" if ids is None else str(ids.dtype).split(".")[-1]
    out = torch.full((x.shape[0],), SENTINEL, dtype=RESULT[table], device=x.device)
    k(x, ids, out)
    return dict(out=out)


def rows_of(k, x, ids):
    """The result of one launch, snapshotted before the next one."""
    return snapshot(run(k, x, ids))


def take(x, order):
    """The rows ``order`` of ONE operand set as a new contiguous call (the values are moved, never redrawn)."""
    return x[torch.tensor(order, dtype=torch.long, device=x.device)].contiguous()


def pick(results, order):
    """The rows ``order`` of per-row results."""
    index = torch.tensor(order, dtype=torch.long, device=results["out"].device)
    return {name: t[index] for name, t in results.items()}


def wave_edges(dev, below=MAX_ROWS):
    """The row counts below ``below`` at which a scheduling wave of this part can end: every multiple of its SM count.  A wave runs
    one thread block per SM or two -- the compiled kernel's registers decide, not the config (on compute capability 10.0 the
    driver reports two) -- and an edge of either is such a multiple."""
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return list(range(sms, below, sms))


@pytest.mark.parametrize("table, rows", (("none", 1), ("int32", 17), ("int64", 300), ("int64", MAX_ROWS)), ids=str)
def test_replay(table, rows):
    """Eight executions into fresh vectors under side-stream contention: every byte of the result matches the first execution;
    then the independent reference."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_rows(rows, dev, seed=200 + rows, kind="special")
    ids = make_table(table, dev, seed=rows)
    expected = assert_replays(lambda: run(k, x, ids), repeats=8)
    assert_rule(expected["out"], x, ids, f"replay rows {rows}, table {table}")


def test_replay_on_a_fresh_entry_and_fresh_operands():
    """Two entries, two identical operand sets drawn afresh: byte-identical (the entry, the operand storage and the output all
    fresh)."""
    dev = _device_or_skip()

    def once():
        return snapshot(run(make_entry(dev), make_rows(24, dev, seed=9, kind="ties"), make_table("int64", dev, seed=9)))

    assert_bitwise(once(), once(), "fresh entry")


def test_launch_on_a_side_stream_writes_the_same_result():
    """The entry launches on torch's CURRENT stream of its device: the same call under a side stream (ordered after the
    operands' producer stream) writes the result of the default-stream call."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_rows(17, dev, seed=61, kind="special")
    ids = make_table("int32", dev, seed=61)
    want = rows_of(k, x, ids)
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))  # the operands were produced on the current stream
    with torch.cuda.stream(side):
        got = rows_of(k, x, ids)
    side.synchronize()
    assert_bitwise(got, want, "side stream vs the current stream")


@pytest.mark.parametrize("table", ("int64", "none"))
def test_batch_invariance(table):
    """ONE operand set of 1024 rows.  Every row alone (a one-row call) writes the result it writes inside the 1024-row call,
    inside every shorter call that holds it (the first 2 .. 64 rows and the rows around this part's first four scheduling-wave
    edges) and inside the reversed batch.  What changes between "alone" and "inside" is the grid, the row's thread block and
    the scheduling wave it runs in."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_rows(MAX_ROWS, dev, seed=300, kind="special")
    ids = make_table(table, dev, seed=300)
    full = assert_replays(lambda: run(k, x, ids), repeats=3)
    assert_rule(full["out"], x, ids, f"batch invariance: the {MAX_ROWS}-row call")
    singles = [rows_of(k, x[r : r + 1], ids) for r in range(MAX_ROWS)]
    alone = dict(out=torch.cat([one["out"] for one in singles]))
    assert_bitwise(alone, full, f"every row alone vs inside {MAX_ROWS} rows")
    around = {n for edge in wave_edges(dev)[:4] for n in (edge - 1, edge, edge + 1) if 1 < n < MAX_ROWS}
    for inside in sorted({*INSIDE, *around}):
        batch = rows_of(k, x[:inside], ids)
        assert_bitwise(batch, dict(out=alone["out"][:inside]), f"the first {inside} rows as one call vs each row alone")
    reverse = list(range(MAX_ROWS - 1, -1, -1))
    flipped = rows_of(k, take(x, reverse), ids)
    assert_bitwise(dict(out=flipped["out"].flip(0)), alone, "the reversed batch")


def test_insertion_removal_and_reordering():
    """A 130-row call of ONE operand set, then the same rows with a row inserted near the head, with three rows removed and
    permuted: every row keeps the result it has in the first call, and the inserted row writes the result it writes alone."""
    dev = _device_or_skip()
    n = 130
    k = make_entry(dev)
    x = make_rows(n + 1, dev, seed=700, kind="special")  # row n is the row to insert
    ids = make_table("int64", dev, seed=700)
    base = rows_of(k, x[:n], ids)
    new_row = rows_of(k, x[n:], ids)
    every = dict(out=torch.cat([base["out"], new_row["out"]]))  # the result of operand row r at index r
    orders = {
        "a row inserted at position 5": list(range(5)) + [n] + list(range(5, n)),
        "rows 3, 11 and the last removed": [r for r in range(n) if r not in (3, 11, n - 1)],
        "a permutation": torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(7)).tolist(),
    }
    for what, order in orders.items():
        got = rows_of(k, take(x, order), ids)
        assert_bitwise(got, pick(every, order), what)


def test_duplicated_row_and_neighbour_isolation():
    """A row twice in one call: both copies equal its alone result.  The same row among REDRAWN neighbours -- other values
    around it, another position, a taller call: at position 5 of 64 rows and, in a call 104 rows taller than the part's second
    scheduling-wave edge (twice its SM count), at the first position behind that edge -- keeps its result."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_rows(8, dev, seed=400, kind="ties")
    ids = make_table("int64", dev, seed=400)
    alone = rows_of(k, x[2:3], ids)
    got = rows_of(k, take(x, [0, 1, 2, 2, 3, 4, 5, 6, 7]), ids)
    assert_bitwise(pick(got, [2]), alone, "a duplicated row, first copy")
    assert_bitwise(pick(got, [3]), alone, "a duplicated row, second copy")
    edge = wave_edges(dev)[1]
    for position, others in ((5, 63), (edge, edge + 103)):
        other = make_rows(others, dev, seed=401 + others)
        mixed = torch.cat([other[:position], x[2:3], other[position:]]).contiguous()
        got = rows_of(k, mixed, ids)
        assert_bitwise(pick(got, [position]), alone, f"the row at position {position} among {others} redrawn neighbours")


def test_inputs_are_unmodified():
    """``x`` and ``ids`` hold the bytes they held before the launches -- every table kind, caller-owned and entry-owned
    outputs, at 1, 17 and 1024 rows."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_rows(MAX_ROWS, dev, seed=88, kind="special")
    tables = {table: make_table(table, dev, seed=88) for table in TABLES}
    before = snapshot(dict(x=x.view(torch.int16), **{table: ids for table, ids in tables.items() if ids is not None}))
    launches = 0
    for table, ids in tables.items():
        for rows in (1, 17, MAX_ROWS):
            run(k, x[:rows], ids)
            k(x[:rows], ids)
            launches += 2
    torch.cuda.synchronize()
    after = snapshot(dict(x=x.view(torch.int16), **{table: ids for table, ids in tables.items() if ids is not None}))
    assert_bitwise(after, before, f"the inputs after {launches} launches")


def test_one_element_moves_one_row():
    """Locality: ONE element raised above its row's maximum moves the result of ITS row -- to that element's column -- and the
    result of no other row; the first and the last column, the first row, a row in the middle and the last row."""
    dev = _device_or_skip()
    rows = 9
    k = make_entry(dev)
    x = make_rows(rows, dev, seed=555)
    base = rows_of(k, x, None)
    for row in (0, 4, rows - 1):
        for column in (0, 31999, WIDTH - 1):
            changed = x.clone()
            changed[row, column] = 3.0e38  # above every value of the draw
            got = rows_of(k, changed, None)
            moved = (got["out"] != base["out"]).nonzero().reshape(-1).tolist()
            assert int(got["out"][row]) == column, f"element ({row}, {column}): its row's result is {int(got['out'][row])}"
            assert moved in ([row], []), f"element ({row}, {column}): the results of rows {moved} moved"


def test_the_row_ceiling_call():
    """The 1024-row call, eight executions under side-stream contention, and rows alone against the same rows inside it: the
    rows on both sides of every scheduling-wave edge of this part below the ceiling (every multiple of its SM count: an edge
    whether one thread block occupies an SM or two share it), and rows 0, 511, 512 and 1023 on every part.  How many rows were
    compared is one line of the run's "measured" section, never an assertion: the kernel holds no state between thread blocks."""
    dev = _device_or_skip()
    import frost_routing

    k = make_entry(dev)
    x = make_rows(MAX_ROWS, dev, seed=900, kind="ties")
    ids = make_table("int64", dev, seed=900)
    full = assert_replays(lambda: run(k, x, ids), repeats=8)
    assert_rule(full["out"], x, ids, "the row ceiling call")
    edges = {r for edge in wave_edges(dev) for r in (edge - 1, edge, edge + 1) if 0 <= r < MAX_ROWS}
    rows = sorted(edges | {0, 511, 512, MAX_ROWS - 1})
    alone = dict(out=torch.cat([rows_of(k, x[r : r + 1], ids)["out"] for r in rows]))
    assert_bitwise(alone, pick(full, rows), f"rows {rows} alone vs inside {MAX_ROWS} rows")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    frost_routing.measured(
        "row argmax, the row ceiling call",
        f"rows compared alone against the {MAX_ROWS}-row call: {len(rows)} (rows 0, 511, 512, {MAX_ROWS - 1} and both sides of every multiple of {sms} SMs)",
    )


def test_row_ceiling_and_the_empty_call_are_typed_declines():
    """The row ceiling and the empty call decline by type -- never a launch, never another provider."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_rows(1, dev, seed=25).expand(MAX_ROWS + 1, WIDTH)
    ids = make_table("int64", dev, seed=25)
    snapshot(run(k, x[:MAX_ROWS], ids))
    sentinel = torch.full((MAX_ROWS + 1,), SENTINEL, dtype=I64, device=dev)
    with pytest.raises(NotImplementedError, match="rows"):
        k(x, ids, sentinel)
    with pytest.raises(ValueError, match=r"\bx\b"):
        k(x[:0], ids, sentinel)
    torch.cuda.synchronize()
    assert bool((sentinel == SENTINEL).all()), "a declined call launches nothing"


def _golden_operands(case, rows, dev):
    if case == "rule":
        _, x, _ = rule_rows()
        assert x.shape[0] == rows
        return x.to(dev)
    return make_rows(rows, dev, seed=500 + rows, kind=case)


@pytest.mark.parametrize(
    "case, rows, table",
    [
        pytest.param("random", 1, "none", id="random-rows1-none"),
        pytest.param("special", 8, "int32", id="special-rows8-int32"),
        pytest.param("rule", 27, "int64", id="rule-rows27-int64"),
        pytest.param("ties", 300, "int64", id="ties-rows300-int64"),
    ],
)
def test_golden(case, rows, table):
    """Four small cases -- one random row without a table, eight rows with NaN, infinities and zeros of both signs through an
    int32 table, the rows that state the rule one sentence at a time and 300 rows that hold their maximum thousands of times
    (more than twice the SM count of a 148-SM part) through an int64 table: three executions, the independent reference, then
    the saved hashes of the inputs and of the result.  Recorded into a NEW directory with ``CUDNN_DETERMINISM_RECORD`` only
    after the replay and reference cells pass.  The op is exact: these hashes are the same on every part, and a change of one
    is a change of the rule."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = _golden_operands(case, rows, dev)
    ids = make_table(table, dev, seed=rows)
    label = f"golden {case} rows {rows} table {table}"
    actual = assert_replays(lambda: run(k, x, ids), repeats=3)
    assert_rule(actual["out"], x, ids, label)
    inputs = dict(x=x.view(torch.int16)) if ids is None else dict(x=x.view(torch.int16), ids=ids)
    assert_golden(f"row-argmax-decode-w{WIDTH}-{case}-rows{rows}-{table}", inputs, actual)
