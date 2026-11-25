# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the FP4 row quantization decode step (``cudnn.RowQuantizeFp4Decode``), raw bytes throughout: bitwise replay into
fresh buffers under side-stream contention at 1, 17, 160 and 1024 rows; a fresh entry on fresh identical operands; a launch on a side
stream; batch invariance of every row -- alone (a one-row call) against the same row inside the first 2 .. 1024 rows of ONE operand
set, inside the reversed batch, after an insertion, a removal and a permutation of its neighbours (each moves rows across the
32-row lines and the 128-row band edge of the scale layout), duplicated inside one call, and with every other row redrawn; the
inputs unmodified by a launch; a change of ONE element moving the bytes of its block and nothing else; the row ceiling call; the
row ceiling and the empty call as typed declines; and the goldens (four small cases, recorded with
``CUDNN_DETERMINISM_RECORD=<new dir>`` only after the replay and the reference pass).

DIRECTIONS.  The step is an INFERENCE step: forward only.  It has no backward entry, no gradient and no state, so this module has no
backward cell -- that direction does not exist by contract; it is not a coverage gap of an existing path.  The results are the two
buffers a launch writes: ``codes`` and ``scales``.  The entry owns no scratch.  An empty call (zero rows) is not served: it is a
typed decline, pinned below, so there is no empty-sequence cell.

WHAT IS HELD FIXED.  A block's bytes are a function of its sixteen values and of the global scale: there is no reduction across
threads, no split, no pick and no knob that selects another association, so there is nothing to pin across batch layouts besides
the operands.  The cells build the operands ONCE and slice / permute them (the row under test is never redrawn) and keep the one
scale tensor.  Every output is a fresh caller-owned buffer -- the codes filled with a sentinel, the scale buffer zero-filled as its
owner's contract asks (the kernel writes the live rows' bytes only) -- never the entry-owned one, so a reused buffer cannot make a
comparison pass.  A block scale's POSITION in the 128 x 4 scale layout moves with its row index while its value does not: batch
layouts are compared on the scale bytes in logical (row, block) order, read through the layout written out in the test helpers,
never by offset; the replay and golden cells compare the whole scale buffer.  The independent references
(``row_quantize_fp4_decode_utils.assert_reference``: the block scales bitwise, the two-code bracket) are a separate check and gate
the goldens."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from row_quantize_fp4_decode_utils import (  # noqa: E402
    BF16,
    BLOCK,
    BLOCKS,
    CONSTRUCTED_SCALE_BITS,
    E4M3,
    LAUNCH_ARM_IDS,
    LAUNCH_ARMS,
    ODD_SCALE_BITS,
    U8,
    WIDTH,
    _device_or_skip,
    assert_reference,
    code_domain_operand,
    make_entry,
    make_operands,
    scale_byte_index,
    scale_id,
    scale_rows,
    scale_tensor,
)

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot  # noqa: E402

from cudnn.row_quantize.frost.kernel import row_quantize_fp4_decode_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

MAX_ROWS = kcfg.MAX_ROWS
#: the row counts a batch-invariance cell runs as ONE call besides the row ceiling: 31 .. 33, 63 .. 65 and 95 .. 97 are the lines of
#: one scale atom's quarters, 127 .. 129 the edge of a scale band
INSIDE = (2, 3, 8, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 160)
QUARTER_SCALE_BITS = 0x3E800000  # 0.25: the power-of-two scale of the golden cases

_INDEX = {}


def run(k, x, scale):
    """One launch into FRESH caller-owned buffers: the codes filled with the sentinel ``0xA5``, the scale buffer zero-filled (its
    owner's contract: the kernel writes the live rows' bytes only).  Returns ``{"codes", "scales"}``."""
    rows, dev = x.shape[0], x.device
    outs = dict(
        codes=torch.full((rows, WIDTH // 2), 0xA5, dtype=U8, device=dev),
        scales=torch.zeros(scale_rows(rows), BLOCKS, dtype=E4M3, device=dev),
    )
    k(x, scale, **outs)
    return outs


def logical(got):
    """A launch's results with the scale buffer replaced by the live rows' bytes in LOGICAL (row, block) order (device uint8
    ``[rows, width / 16]``): two batch layouts are compared on the bytes a row owns, wherever the layout puts them."""
    rows, dev = got["codes"].shape[0], got["codes"].device
    if dev not in _INDEX:
        _INDEX[dev] = scale_byte_index(MAX_ROWS).to(dev)
    return dict(got, scales=got["scales"].view(U8).reshape(-1)[_INDEX[dev][:rows]])


def rows_of(k, x, scale):
    """The bytes of one launch, snapshotted before the next one, with the scale bytes per row (:func:`logical`)."""
    return logical(snapshot(run(k, x, scale)))


def take(x, order):
    """The rows ``order`` of ONE operand set as a new contiguous call (the values are moved, never redrawn)."""
    return x[torch.tensor(order, dtype=torch.long, device=x.device)].contiguous()


def pick(results, order):
    """The rows ``order`` of per-row results."""
    index = torch.tensor(order, dtype=torch.long, device=next(iter(results.values())).device)
    return {name: t[index] for name, t in results.items()}


def check_reference(got, x, scale, label):
    """The independent references on one launch's bytes."""
    assert_reference(got["codes"], got["scales"], x, scale, label=label)


@pytest.mark.parametrize("rows", (1, 17, 160, MAX_ROWS), ids=lambda rows: f"rows{rows}")
def test_replay(rows):
    """Eight executions into fresh buffers under side-stream contention: every byte of the codes and of the WHOLE scale buffer
    matches the first execution; then the independent references."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(rows, dev, seed=200 + rows)
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    expected = assert_replays(lambda: run(k, x, scale), repeats=8)
    check_reference(expected, x, scale, f"replay rows {rows}")


def test_replay_on_a_fresh_entry_and_fresh_operands():
    """Two entries, two identical operand sets drawn afresh: byte-identical (the entry, the operand storage and every output all
    fresh)."""
    dev = _device_or_skip()

    def once():
        x = make_operands(24, dev, seed=9, kind="structured")
        return snapshot(run(make_entry(dev), x, scale_tensor(CONSTRUCTED_SCALE_BITS[0], dev)))

    assert_bitwise(once(), once(), "fresh entry")


@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
def test_launch_on_a_side_stream_writes_the_same_bytes(arm, arm_id):
    """The entry launches on torch's CURRENT stream of its device: the same call under a side stream (ordered after the operands'
    producer stream) writes the bytes of the default-stream call, under every launch arm."""
    dev = _device_or_skip()
    k = make_entry(dev, arm)
    x = make_operands(17, dev, seed=61, kind="structured")
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    want = snapshot(run(k, x, scale))
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))  # the operands were produced on the current stream
    with torch.cuda.stream(side):
        got = snapshot(run(k, x, scale))
    side.synchronize()
    assert_bitwise(got, want, f"side stream vs the current stream, {arm_id}")


@pytest.mark.parametrize("bits", (ODD_SCALE_BITS, CONSTRUCTED_SCALE_BITS[1]), ids=scale_id)
def test_batch_invariance(bits):
    """ONE operand set of 1024 rows.  Every row alone (a one-row call) writes the bytes it writes inside the 1024-row call, inside
    every shorter call that holds it (the first 2 .. 160 rows, on both sides of the 32-row lines of a scale atom and of the 128-row
    band edge) and inside the reversed batch: its codes and its 320 block scales.  What changes between "alone" and "inside" is the
    grid, the row's thread block, the scheduling wave it runs in and the position of its scale bytes."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(MAX_ROWS, dev, seed=300, kind="structured")
    scale = scale_tensor(bits, dev)
    replayed = assert_replays(lambda: run(k, x, scale), repeats=3)
    check_reference(replayed, x, scale, f"batch invariance: the {MAX_ROWS}-row call")
    full = logical(replayed)
    singles = [rows_of(k, x[r : r + 1], scale) for r in range(MAX_ROWS)]
    alone = {name: torch.cat([one[name] for one in singles]) for name in full}
    assert_bitwise(alone, full, f"every row alone vs inside {MAX_ROWS} rows")
    for inside in INSIDE:
        batch = rows_of(k, x[:inside], scale)
        assert_bitwise(batch, {name: t[:inside] for name, t in alone.items()}, f"the first {inside} rows as one call vs each row alone")
    reverse = list(range(MAX_ROWS - 1, -1, -1))
    flipped = rows_of(k, take(x, reverse), scale)
    assert_bitwise({name: t.flip(0) for name, t in flipped.items()}, alone, "the reversed batch")


def test_insertion_removal_and_reordering():
    """A 130-row call of ONE operand set (two scale bands), then the same rows with a row inserted near the head (rows 5 .. 129 move
    down by one: row 127 enters the second band; rows 31, 63 and 95 each move to the next quarter of their atoms), with three rows
    removed (rows 128 and 129 move up into the first band) and permuted: every row keeps the bytes it has in the first call, and the
    inserted row writes the bytes it writes alone."""
    dev = _device_or_skip()
    n = 130
    k = make_entry(dev)
    x = make_operands(n + 1, dev, seed=700, kind="structured")  # row n is the row to insert
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    base = rows_of(k, x[:n], scale)
    new_row = rows_of(k, x[n:], scale)
    every = {name: torch.cat([base[name], new_row[name]]) for name in base}  # the bytes of operand row r at index r
    orders = {
        "a row inserted at position 5": list(range(5)) + [n] + list(range(5, n)),
        "rows 3, 11 and the last removed": [r for r in range(n) if r not in (3, 11, n - 1)],
        "a permutation": torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(7)).tolist(),
    }
    for what, order in orders.items():
        got = rows_of(k, take(x, order), scale)
        assert_bitwise(got, pick(every, order), what)


def test_duplicated_row_and_neighbour_isolation():
    """A row twice in one call: both copies equal its alone bytes.  The same row among REDRAWN neighbours -- other values around it,
    another position, a taller call: at position 5 of 64 rows and at position 128 of 160 rows (the first row of the second scale
    band) -- keeps its bytes."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(8, dev, seed=400, kind="structured")
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    alone = rows_of(k, x[2:3], scale)
    got = rows_of(k, take(x, [0, 1, 2, 2, 3, 4, 5, 6, 7]), scale)
    assert_bitwise(pick(got, [2]), alone, "a duplicated row, first copy")
    assert_bitwise(pick(got, [3]), alone, "a duplicated row, second copy")
    for position, others in ((5, 63), (128, 159)):
        other = make_operands(others, dev, seed=401 + others)
        mixed = torch.cat([other[:position], x[2:3], other[position:]]).contiguous()
        got = rows_of(k, mixed, scale)
        assert_bitwise(pick(got, [position]), alone, f"the row at position {position} among {others} redrawn neighbours")


def test_inputs_are_unmodified():
    """``x`` and ``scale`` hold the bytes they held before the launches -- every launch arm, caller-owned and entry-owned outputs,
    at 1, 17 and 1024 rows.  ``scale`` is read in front of the dependent-launch wait and ``x`` behind it: both are inputs only."""
    dev = _device_or_skip()
    x = make_operands(MAX_ROWS, dev, seed=88, kind="structured")
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    before = snapshot(dict(x=x, scale=scale))
    launches = 0
    for arm in LAUNCH_ARMS:
        k = make_entry(dev, arm)
        for rows in (1, 17, MAX_ROWS):
            run(k, x[:rows], scale)
            k(x[:rows], scale)
            launches += 2
    torch.cuda.synchronize()
    assert_bitwise(snapshot(dict(x=x, scale=scale)), before, f"the inputs after {launches} launches")


def test_one_element_moves_one_block():
    """Locality: a change of ONE element moves the eight code bytes and the one scale byte of ITS block and no other byte of the
    call -- for the first and the last element of a block, in the first and the last block of a row, in the first row, a row in the
    middle and the last row.  A code is a function of its block alone.  The operand's values are small (no block scale
    saturates) and the changed element becomes its block's new maximum, so the block's scale byte and its codes do move."""
    dev = _device_or_skip()
    rows = 9
    k = make_entry(dev)
    gen = torch.Generator(device="cpu").manual_seed(555)
    x = (torch.randn(rows, WIDTH, generator=gen) * 0.05).to(BF16).to(dev)
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    base = rows_of(k, x, scale)
    assert int(base["scales"].max()) < 0x7E
    for row in (0, 4, rows - 1):
        for block in (0, 137, BLOCKS - 1):
            for element in (0, BLOCK - 1):
                changed = x.clone()
                column = block * BLOCK + element
                changed[row, column] = 20.0  # the block's new maximum, far above the others: its scale and its codes move
                got = rows_of(k, changed, scale)
                codes_differ = (got["codes"] != base["codes"]).cpu()
                scales_differ = (got["scales"] != base["scales"]).cpu()
                outside_codes = codes_differ.clone()
                outside_codes[row, block * (BLOCK // 2) : (block + 1) * (BLOCK // 2)] = False
                outside_scales = scales_differ.clone()
                outside_scales[row, block] = False
                where = f"element ({row}, {column})"
                assert not bool(outside_codes.any()) and not bool(outside_scales.any()), f"{where}: a byte outside the element's block moved"
                own_codes = codes_differ[row, block * (BLOCK // 2) : (block + 1) * (BLOCK // 2)]
                assert bool(scales_differ[row, block]) and bool(own_codes.any()), f"{where}: the block's own bytes did not move"


def test_the_row_ceiling_call():
    """The 1024-row call, eight executions under side-stream contention, and rows alone against the same rows inside it: the rows on
    both sides of every scheduling-wave edge below the ceiling where this part has one (one wave holds ``CTAS_PER_SM`` thread
    blocks per SM: 888 rows on a 148-SM part, 912 on 152 SMs, 1,272 on 212 SMs -- there no served call crosses a wave), and rows 0,
    511, 512 and 1023 on every part.  Whether the call crossed a wave on this part is one line of the run's "measured" section,
    never an assertion: the kernel holds no state between thread blocks, so nothing depends on it."""
    dev = _device_or_skip()
    import frost_routing

    k = make_entry(dev)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    wave = k.cfg.CTAS_PER_SM * sms
    x = make_operands(MAX_ROWS, dev, seed=900)
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    full = logical(assert_replays(lambda: run(k, x, scale), repeats=8))
    edges = {r for edge in range(wave, MAX_ROWS, wave) for r in (edge - 1, edge, edge + 1) if 0 <= r < MAX_ROWS}
    rows = sorted(edges | {0, 511, 512, MAX_ROWS - 1})
    singles = [rows_of(k, x[r : r + 1], scale) for r in rows]
    alone = {name: torch.cat([one[name] for one in singles]) for name in full}
    assert_bitwise(alone, pick(full, rows), f"rows {rows} alone vs inside {MAX_ROWS} rows ({wave} thread blocks per wave)")
    crossed = "yes" if MAX_ROWS > wave else "no"
    frost_routing.measured(
        "FP4 row quantization, the row ceiling call",
        f"crossed a scheduling wave: {crossed} (wave = {wave} thread blocks on {sms} SMs; rows compared alone: {len(rows)})",
    )


def test_row_ceiling_is_the_typed_decline():
    """The row ceiling and the empty call decline by type -- never a launch, never another provider."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = make_operands(MAX_ROWS + 1, dev, seed=25)
    scale = scale_tensor(ODD_SCALE_BITS, dev)
    snapshot(run(k, x[:MAX_ROWS], scale))
    sentinel = dict(
        codes=torch.full((MAX_ROWS + 1, WIDTH // 2), 0x77, dtype=U8, device=dev), scales=torch.full((1152, BLOCKS), 0x77, dtype=U8, device=dev).view(E4M3)
    )
    with pytest.raises(NotImplementedError, match=f"at most {MAX_ROWS} rows per call; got {MAX_ROWS + 1}"):
        k(x, scale, **sentinel)
    with pytest.raises(ValueError, match="at least one row"):
        k(x[:0], scale, **sentinel)
    torch.cuda.synchronize()
    assert bool((sentinel["codes"] == 0x77).all()) and bool((sentinel["scales"].view(U8) == 0x77).all()), "a declined call launches nothing"


def _golden_operands(case, rows, dev):
    if case == "code-domain":
        x = code_domain_operand(ODD_SCALE_BITS).to(dev)
        assert x.shape[0] == rows
        return x
    return make_operands(rows, dev, seed=500 + rows, kind=case)


@pytest.mark.parametrize(
    "case, rows, bits",
    [
        pytest.param("random", 1, ODD_SCALE_BITS, id="random-rows1"),
        pytest.param("random", 8, QUARTER_SCALE_BITS, id="random-rows8"),
        pytest.param("structured", 130, CONSTRUCTED_SCALE_BITS[0], id="structured-rows130"),
        pytest.param("code-domain", 854, ODD_SCALE_BITS, id="code-domain-rows854"),
    ],
)
def test_golden(case, rows, bits):
    """Four small cases -- one row and eight rows of random operands under a non-power-of-two and a power-of-two global scale, 130
    structured rows (two scale bands) under a scale that separates the associations of the block scale, and the ENUMERATED code
    domain of the non-power-of-two scale (every bf16 element under every scale byte, both signs: the inputs that see the last bit
    of the approximate reciprocals, which random operands meet only by luck): three executions, the independent references, the
    launch arms on the same operands (each writes the default's bytes, so the one golden holds for every arm), then the saved
    hashes of the codes and of the whole scale buffer.  Recorded into a NEW directory with ``CUDNN_DETERMINISM_RECORD`` only after
    the replay and reference cells pass; a change of these hashes on one runtime is a change of the rounding chain and needs a
    stated numerical reason.  The codes follow the part's approximate reciprocal: a golden holds on its own runtime and is not a
    claim across GPU architectures."""
    dev = _device_or_skip()
    k = make_entry(dev)
    x = _golden_operands(case, rows, dev)
    scale = scale_tensor(bits, dev)
    label = f"golden {case} rows {rows}"
    actual = assert_replays(lambda: run(k, x, scale), repeats=3)
    check_reference(actual, x, scale, label)
    for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
        other = snapshot(run(make_entry(dev, arm), x, scale))
        assert_bitwise(other, actual, f"{label}: the {arm_id} arm")
    assert_golden(f"row-quantize-fp4-decode-w{WIDTH}-{case}-rows{rows}", dict(x=x, scale=scale), actual)
