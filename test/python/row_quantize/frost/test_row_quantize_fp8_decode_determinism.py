# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the FP8 row quantization decode step (``cudnn.RowQuantizeFp8Decode``), raw bytes throughout: bitwise replay into
fresh buffers under side-stream contention at 1, 17, 297 and 1024 rows; a fresh entry on fresh identical operands; a launch on a
side stream; batch invariance of every row -- alone (a one-row call) against the same row inside the first 2 .. 1024 rows of ONE
operand set, inside the reversed batch, after an insertion, a removal and a permutation of its neighbours, duplicated inside one
call, and with every other row redrawn or replaced by rows of huge values; the inputs unmodified by a launch; a change of ONE
element moving the bytes of its own row and nothing else (its own code byte alone while the row maximum stands); the row ceiling
call across scheduling waves; the row ceiling and the empty call as typed declines; and the goldens (two small cases, recorded with
``CUDNN_DETERMINISM_RECORD=<new dir>`` only after the replay and the reference pass).

DIRECTIONS.  The step is an INFERENCE step: forward only.  It has no backward entry, no gradient and no state, so this module
has no backward cell -- that direction does not exist by contract; it is not a coverage gap of an existing path.  The results are
the two buffers a launch writes: ``codes`` and ``scales``.  The entry owns no scratch.  An empty call (zero rows) is not served:
it is a typed decline, pinned below, so there is no empty-sequence cell.

WHAT IS HELD FIXED.  A row's bytes are a function of its values: the one reduction (the row maximum) never leaves the row's
thread block and is exact in any order, and there is no split, no pick and no knob that selects another association, so there
is nothing to pin across batch layouts besides the operands.  The cells build the operands ONCE and slice / permute them (the
row under test is never redrawn).  Every output is a fresh caller-owned, sentinel-filled buffer, never the entry-owned one, so a
reused buffer cannot make a comparison pass.  The independent reference (``row_quantize_fp8_decode_utils.assert_reference``:
every code byte and every scale bit) is a separate check and gates the goldens."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from row_quantize_fp8_decode_utils import (  # noqa: E402
    BF16,
    E4M3,
    FP32,
    LAUNCH_ARM_IDS,
    LAUNCH_ARMS,
    U8,
    WIDTH,
    _device_or_skip,
    assert_reference,
    make_entry,
    make_operands,
)

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot  # noqa: E402

from cudnn.row_quantize.frost.kernel import row_quantize_fp8_decode_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

MAX_ROWS = kcfg.MAX_ROWS
#: the row counts a batch-invariance cell runs as ONE call besides the row ceiling
INSIDE = (2, 3, 8, 16, 17, 64, 129, 160)


def run(k, x):
    """One launch into FRESH caller-owned buffers, sentinel-filled.  Returns ``{"codes", "scales"}``."""
    rows, dev = x.shape[0], x.device
    outs = dict(codes=torch.full((rows, WIDTH), 0xA5, dtype=U8, device=dev).view(E4M3), scales=torch.full((rows, 1), -7.0, dtype=FP32, device=dev))
    k(x, **outs)
    return outs


def as_bytes(results):
    """A launch's results with the codes as their bytes (uint8): what the cells concatenate, index and compare."""
    return dict(results, codes=results["codes"].view(U8))


def rows_of(k, x):
    """The bytes of one launch, snapshotted before the next one."""
    return as_bytes(snapshot(run(k, x)))


def take(x, order):
    """The rows ``order`` of ONE operand set as a new contiguous call (the values are moved, never redrawn)."""
    return x[torch.tensor(order, dtype=torch.long, device=x.device)].contiguous()


def pick(results, order):
    """The rows ``order`` of per-row results."""
    index = torch.tensor(order, dtype=torch.long, device=next(iter(results.values())).device)
    return {name: t[index] for name, t in results.items()}


def wave_edges(dev, below=MAX_ROWS):
    """The row counts below ``below`` at which a scheduling wave of this part can end: every multiple of its SM count.  A wave runs
    one thread block per SM or two -- the compiled kernel's registers decide, not the config -- and an edge of either is such a
    multiple."""
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return list(range(sms, below, sms))


@pytest.mark.parametrize("rows", (1, 17, 297, MAX_ROWS), ids=lambda rows: f"rows{rows}")
def test_replay(rows):
    """Eight executions into fresh buffers under side-stream contention: every byte of the codes and every bit of the scales
    matches the first execution; then the independent reference."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(rows, dev, seed=200 + rows, kind="structured")
    expected = assert_replays(lambda: run(k, x), repeats=8)
    assert_reference(expected["codes"], expected["scales"], x, label=f"replay rows {rows}")


def test_replay_on_a_fresh_entry_and_fresh_operands():
    """Two entries, two identical operand sets drawn afresh: byte-identical (the entry, the operand storage and every output
    all fresh)."""
    dev = _device_or_skip()

    def once():
        return snapshot(run(make_entry(dev), make_operands(24, dev, seed=9, kind="structured")))

    assert_bitwise(once(), once(), "fresh entry")


@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
def test_launch_on_a_side_stream_writes_the_same_bytes(arm, arm_id):
    """The entry launches on torch's CURRENT stream of its device: the same call under a side stream (ordered after the
    operands' producer stream) writes the bytes of the default-stream call, under every launch arm."""
    dev = _device_or_skip()
    k = make_entry(dev, arm)
    x = make_operands(17, dev, seed=61, kind="structured")
    want = snapshot(run(k, x))
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))  # the operands were produced on the current stream
    with torch.cuda.stream(side):
        got = snapshot(run(k, x))
    side.synchronize()
    assert_bitwise(got, want, f"side stream vs the current stream, {arm_id}")


def test_batch_invariance():
    """ONE operand set of 1024 rows.  Every row alone (a one-row call) writes the bytes it writes inside the 1024-row call, inside
    every shorter call that holds it (the first 2 .. 160 rows and the rows around this part's first two scheduling-wave edges) and inside
    the reversed batch: its codes and its scale.  What changes between "alone" and "inside" is the grid, the row's thread block
    and the scheduling wave it runs in."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(MAX_ROWS, dev, seed=300, kind="structured")
    full = as_bytes(assert_replays(lambda: run(k, x), repeats=3))
    assert_reference(full["codes"], full["scales"], x, label=f"batch invariance: the {MAX_ROWS}-row call")
    singles = [rows_of(k, x[r : r + 1]) for r in range(MAX_ROWS)]
    alone = {name: torch.cat([one[name] for one in singles]) for name in full}
    assert_bitwise(alone, full, f"every row alone vs inside {MAX_ROWS} rows")
    around = {n for edge in wave_edges(dev)[:2] for n in (edge - 1, edge, edge + 1) if 1 < n < MAX_ROWS}
    for inside in sorted({*INSIDE, *around}):
        batch = rows_of(k, x[:inside])
        assert_bitwise(batch, {name: t[:inside] for name, t in alone.items()}, f"the first {inside} rows as one call vs each row alone")
    reverse = list(range(MAX_ROWS - 1, -1, -1))
    flipped = rows_of(k, take(x, reverse))
    assert_bitwise({name: t.flip(0) for name, t in flipped.items()}, alone, "the reversed batch")


def test_insertion_removal_and_reordering():
    """A 40-row call of ONE operand set, then the same rows with a row inserted near the head, with three rows removed and
    permuted: every row keeps the bytes it has in the first call, and the inserted row writes the bytes it writes alone."""
    dev = _device_or_skip()
    n = 40
    k = make_entry(dev)
    x = make_operands(n + 1, dev, seed=700, kind="structured")  # row n is the row to insert
    base = rows_of(k, x[:n])
    new_row = rows_of(k, x[n:])
    every = {name: torch.cat([base[name], new_row[name]]) for name in base}  # the bytes of operand row r at index r
    orders = {
        "a row inserted at position 5": list(range(5)) + [n] + list(range(5, n)),
        "rows 3, 11 and the last removed": [r for r in range(n) if r not in (3, 11, n - 1)],
        "a permutation": torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(7)).tolist(),
    }
    for what, order in orders.items():
        got = rows_of(k, take(x, order))
        assert_bitwise(got, pick(every, order), what)


def test_duplicated_row_and_neighbour_isolation():
    """A row twice in one call: both copies equal its alone bytes.  The same row among REDRAWN neighbours -- other values around
    it, another position, a taller call -- and between two rows of the largest finite magnitude keeps its bytes: a row's maximum
    is its own."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(8, dev, seed=400)
    alone = rows_of(k, x[2:3])
    got = rows_of(k, take(x, [0, 1, 2, 2, 3, 4, 5, 6, 7]))
    assert_bitwise(pick(got, [2]), alone, "a duplicated row, first copy")
    assert_bitwise(pick(got, [3]), alone, "a duplicated row, second copy")
    for position, others in ((5, 63), (299, 600)):
        other = make_operands(others, dev, seed=401 + others)
        mixed = torch.cat([other[:position], x[2:3], other[position:]]).contiguous()
        got = rows_of(k, mixed)
        assert_bitwise(pick(got, [position]), alone, f"the row at position {position} among {others} redrawn neighbours")
    huge = torch.full((1, WIDTH), 3.0e38, dtype=BF16, device=dev)
    got = rows_of(k, torch.cat([huge, x[2:3], -huge]).contiguous())
    assert_bitwise(pick(got, [1]), alone, "the row between two rows of huge values")


def test_inputs_are_unmodified():
    """``x`` holds the bytes it held before the launches -- every launch arm, caller-owned and entry-owned outputs, at 1, 17 and
    1024 rows."""
    dev = _device_or_skip()
    x = make_operands(MAX_ROWS, dev, seed=88, kind="structured")
    before = snapshot(dict(x=x))
    launches = 0
    for arm in LAUNCH_ARMS:
        k = make_entry(dev, arm)
        for rows in (1, 17, MAX_ROWS):
            run(k, x[:rows])
            k(x[:rows])
            launches += 2
    torch.cuda.synchronize()
    assert_bitwise(snapshot(dict(x=x)), before, f"the inputs after {launches} launches")


def test_one_element_moves_only_its_own_row():
    """Locality.  A change of ONE element that leaves its row's maximum standing moves that element's code byte and NO other
    byte of the call.  A change that makes the element its row's new maximum moves that row's scale and codes and no byte of
    another row.  For elements of the first and the last vector of each of the three passes, in the first row, a row in the
    middle and the last row."""
    dev = _device_or_skip()
    rows = 9
    k = make_entry(dev)
    gen = torch.Generator(device="cpu").manual_seed(555)
    x = (torch.randn(rows, WIDTH, generator=gen) * 0.05).to(BF16)
    x[:, 3] = 4.0  # every row's maximum, away from the columns changed below
    x = x.to(dev)
    base = rows_of(k, x)
    for row in (0, 4, rows - 1):
        for column in (0, 8191, 8192, 16383, 16384, WIDTH - 1):
            where = f"element ({row}, {column})"
            changed = x.clone()
            changed[row, column] = 2.0  # below the row's maximum: another code for this element alone
            got = rows_of(k, changed)
            codes_differ = (got["codes"] != base["codes"]).cpu()
            assert bool(codes_differ[row, column]) and int(codes_differ.sum()) == 1, f"{where} below the maximum: {int(codes_differ.sum())} code bytes moved"
            assert torch.equal(got["scales"], base["scales"]), f"{where} below the maximum: a scale moved"
            changed[row, column] = -64.0  # the row's new maximum: its scale and its codes move
            got = rows_of(k, changed)
            codes_differ = (got["codes"] != base["codes"]).cpu()
            scales_differ = (got["scales"] != base["scales"]).cpu().reshape(-1)
            others = [r for r in range(rows) if r != row]
            assert not bool(codes_differ[others].any()) and not bool(scales_differ[others].any()), f"{where} as the new maximum: a byte of another row moved"
            assert (
                bool(scales_differ[row]) and bool(codes_differ[row, column]) and int(codes_differ[row].sum()) > 1
            ), f"{where} as the new maximum: the row did not move"


def test_the_row_ceiling_call():
    """The 1024-row call, eight executions under side-stream contention, and rows alone against the same rows inside it: the
    rows on both sides of every scheduling-wave edge of this part below the ceiling (every multiple of its SM count: an edge
    whether one thread block occupies an SM or two share it), and rows 0, 511, 512 and 1023 on every part.  How many rows were
    compared is one line of the run's "measured" section, never an assertion: the kernel holds no state between thread blocks."""
    dev = _device_or_skip()
    import frost_routing

    k = make_entry(dev)
    x = make_operands(MAX_ROWS, dev, seed=900)
    full = as_bytes(assert_replays(lambda: run(k, x), repeats=8))
    edges = {r for edge in wave_edges(dev) for r in (edge - 1, edge, edge + 1) if 0 <= r < MAX_ROWS}
    rows = sorted(edges | {0, 511, 512, MAX_ROWS - 1})
    singles = [rows_of(k, x[r : r + 1]) for r in rows]
    alone = {name: torch.cat([one[name] for one in singles]) for name in full}
    assert_bitwise(alone, pick(full, rows), f"rows {rows} alone vs inside {MAX_ROWS} rows")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    frost_routing.measured(
        "FP8 row quantization, the row ceiling call",
        f"rows compared alone against the {MAX_ROWS}-row call: {len(rows)} (rows 0, 511, 512, {MAX_ROWS - 1} and both sides of every multiple of {sms} SMs)",
    )


def test_row_ceiling_is_the_typed_decline():
    """The row ceiling and the empty call decline by type -- never a launch, never another provider."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(MAX_ROWS + 1, dev, seed=25)
    snapshot(run(k, x[:MAX_ROWS]))
    sentinel = dict(
        codes=torch.full((MAX_ROWS + 1, WIDTH), 0x77, dtype=U8, device=dev).view(E4M3), scales=torch.full((MAX_ROWS + 1, 1), -3.0, dtype=FP32, device=dev)
    )
    with pytest.raises(NotImplementedError, match=f"at most {MAX_ROWS} rows"):
        k(x, **sentinel)
    with pytest.raises(ValueError, match="at least one row"):
        k(x[:0], **sentinel)
    torch.cuda.synchronize()
    assert bool((sentinel["codes"].view(U8) == 0x77).all()) and bool((sentinel["scales"] == -3.0).all()), "a declined call launches nothing"


@pytest.mark.parametrize("case, rows", [pytest.param("random", 1, id="random-rows1"), pytest.param("structured", 8, id="structured-rows8")])
def test_golden(case, rows):
    """Two small cases -- one random row, and eight structured rows (a zero row, a row of negative zeros, a row under the floor
    scale, rows whose one dominant element sits in the last column, the first column and at the edge of the passes, a row of two
    values): three executions, the independent reference, the launch arms on the same operands (each writes the default's bytes,
    so the one golden holds for every arm), then the saved hashes of the codes and the scales.  Recorded into a NEW directory
    with ``CUDNN_DETERMINISM_RECORD`` only after the replay and reference cells pass; a change of these hashes is a change of the
    rounding chain and needs a stated numerical reason.  Every step of the chain is exactly specified: the bytes do not depend
    on the part."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(rows, dev, seed=500 + rows, kind=case)
    label = f"golden {case} rows {rows}"
    actual = assert_replays(lambda: run(k, x), repeats=3)
    assert_reference(actual["codes"], actual["scales"], x, label=label)
    for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
        other = snapshot(run(make_entry(dev, arm), x))
        assert_bitwise(other, actual, f"{label}: the {arm_id} arm")
    assert_golden(f"row-quantize-fp8-decode-w{WIDTH}-{case}-rows{rows}", dict(x=x), actual)
