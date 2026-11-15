# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the add + RMSNorm + quantize decode step (``cudnn.AddNormQuantDecode``), raw bytes throughout: bitwise replay into
fresh buffers under side-stream contention at 1, 17, 160 and 1024 rows (1024 rows are more thread blocks than one scheduling wave
holds); a fresh entry on fresh identical operands; a launch on a side stream; batch invariance of every row -- alone (a one-row call)
against the same row inside the first 2 .. 1024 rows of ONE operand set, inside the reversed batch, after an insertion, a removal and
a permutation of its neighbours (each moves rows across the edge of a 128-row scale tile), duplicated inside one call, and with every
other row redrawn; the inputs unmodified by a launch and the residual equal to its documented update; the row ceiling and the empty
call as typed declines; and the goldens (six small cases, recorded with ``CUDNN_DETERMINISM_RECORD=<new dir>`` only after the replay
and the reference pass).  Run for one form per numerical path and per residual mode (``DETERMINISM_FORMS``): the add with bf16 and
static E4M3 codes, with bf16 and NVFP4 codes, with bf16 alone; the residual store with E4M3 codes; no residual with E4M3 codes and
with bf16 and NVFP4 codes.

DIRECTIONS.  The step is an INFERENCE step: forward only.  It has no backward entry, no gradient and no state, so this module has no
backward cell -- that direction does not exist by contract; it is not a coverage gap of an existing path.  The results are every
buffer a launch writes: the residual (in / out under ``add``, out under ``store_residual``), ``out``, ``codes`` and the NVFP4
``scales``.  The entry owns no scratch.  An empty call (zero rows) is not served: it is a typed decline, pinned below, so there is no
empty-sequence cell.

WHAT IS HELD FIXED.  The row's sum of squares is ONE fixed tree whose shape depends on ``hidden`` alone -- no split, no pick and no
knob selects another association -- so there is nothing to pin across batch layouts besides the operands.  The cells build the
operands ONCE and slice / permute them (the row under test is never redrawn) and keep the one weight and the one scale tensor.  The
residual is in / out: every execution runs on a fresh copy of the residual's values before the call.  Every output is a fresh
caller-owned buffer filled with a sentinel (a NaN pattern where the dtype has one), never the entry-owned one, so a reused buffer
cannot make a comparison pass.  An NVFP4 block scale's POSITION in the 128 x 4 scale layout moves with its row index while its value
does not: batch layouts are compared on the scale bytes in logical (row, block) order, read through the layout rule
(``kernel.add_norm_quant_decode_config.scale_byte_offset``), never by offset; the replay and golden cells compare the whole scale
buffer.  The independent references (``add_norm_quant_utils.assert_reference``: the fp64 bracket of the normalized value, the
residual, the E4M3 codes, the NVFP4 block scales and code bracket) are a separate check and gate the goldens."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from add_norm_quant_utils import (  # noqa: E402
    BF16,
    DETERMINISM_FORMS,
    E4M3,
    FORMS,
    HIDDEN,
    LAUNCH_ARM_IDS,
    LAUNCH_ARMS,
    U8,
    _device_or_skip,
    assert_nvfp4_codes,
    assert_reference,
    assert_y,
    exact_operands,
    exact_signs,
    exact_targets,
    make_entry,
    make_operands,
    residual_reference,
    scale_byte_index,
    scale_bytes_of_rows,
    scale_tensor,
    twin_form,
)

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot  # noqa: E402

from cudnn.norm.frost.kernel import add_norm_quant_decode_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

MAX_ROWS = kcfg.MAX_ROWS
#: the row counts a batch-invariance cell runs as ONE call besides the row ceiling: 127 / 128 / 129 are the edge of a scale tile
INSIDE = (2, 3, 8, 16, 17, 24, 32, 64, 127, 128, 129, 160)
SCALE_VALUES = {"e4m3_static": 0.02, "nvfp4": 37.0}

_INDEX = {}


def _scale(k, value=None):
    """The scale tensor of an entry's form on its device (``None`` for a form without codes)."""
    quant = k.cfg.QUANT
    return None if quant is None else scale_tensor(SCALE_VALUES[quant] if value is None else value, k.device)


def run(k, x, res0, weight, scale):
    """One launch into FRESH buffers: a fresh copy of the residual (it is in / out) and fresh caller-owned outputs filled with a
    sentinel -- the bf16 and E4M3 NaN patterns, ``0xA5`` for NVFP4 code bytes -- except the scale buffer, which its owner zero-fills
    by contract (the kernel writes the live rows' bytes only).  Returns ``{name: tensor}``: the residual after the call (where the
    form has one) and every output the form produces."""
    cfg, dev, rows = k.cfg, x.device, x.shape[0]
    uses_residual = cfg.ADD or cfg.STORE_RESIDUAL
    residual = res0.clone() if uses_residual else None
    outs = {}
    if cfg.BF16_OUT:
        outs["out"] = torch.full((rows, cfg.HIDDEN), 0x7FC0, dtype=torch.int16, device=dev).view(BF16)
    if cfg.QUANT == "e4m3_static":
        outs["codes"] = torch.full((rows, cfg.HIDDEN), 0x7F, dtype=U8, device=dev).view(E4M3)
    if cfg.QUANT == "nvfp4":
        outs["codes"] = torch.full((rows, cfg.HIDDEN // 2), 0xA5, dtype=U8, device=dev)
        outs["scales"] = torch.zeros(kcfg.scale_rows(rows), cfg.HIDDEN // kcfg.SF_BLOCK, dtype=E4M3, device=dev)
    k(x, residual, weight, scale, **outs)
    return dict(outs, **({"residual": residual} if uses_residual else {}))


def logical(got):
    """A launch's results with the scale buffer replaced by the live rows' bytes in LOGICAL (row, block) order (device uint8
    ``[rows, hidden / 16]``): two batch layouts are compared on the bytes a row owns, wherever the layout rule puts them."""
    if "scales" not in got:
        return got
    rows, dev = got["codes"].shape[0], got["codes"].device
    if dev not in _INDEX:
        _INDEX[dev] = scale_byte_index(MAX_ROWS).to(dev)
    return dict(got, scales=got["scales"].view(U8).reshape(-1)[_INDEX[dev][:rows]])


def rows_of(k, x, res0, weight, scale):
    """The bytes of one launch, snapshotted before the next one, with the scale bytes per row (:func:`logical`)."""
    return logical(snapshot(run(k, x, res0, weight, scale)))


def take(x, res0, order):
    """The rows ``order`` of ONE operand set as a new contiguous call (the values are moved, never redrawn)."""
    index = torch.tensor(order, dtype=torch.long, device=x.device)
    return x[index].contiguous(), res0[index].contiguous()


def pick(results, order):
    """The rows ``order`` of per-row results."""
    index = torch.tensor(order, dtype=torch.long, device=next(iter(results.values())).device)
    return {name: t[index] for name, t in results.items()}


def check_reference(form, got, x, res0, weight, scale, label, share_bound=False):
    """The independent references on one launch's bytes.  A form that does not write the bf16 value is checked against the output
    of the form that does, on the same operands (that output is checked against the fp64 bracket first)."""
    add, _, bf16, _ = FORMS[form]
    y = None
    if not bf16:
        y = snapshot(run(make_entry(x.device, twin_form(form)), x, res0, weight, scale))["out"]
        assert_y(y, x, res0, weight, add, share_bound=share_bound, label=f"{label} (the form that also writes bf16)")
    assert_reference(form, got, x, res0, weight, scale, y=y, share_bound=share_bound, label=label)


@pytest.mark.parametrize("rows", (1, 17, 160, MAX_ROWS), ids=lambda rows: f"rows{rows}")
@pytest.mark.parametrize("form", DETERMINISM_FORMS)
def test_replay(form, rows):
    """Eight executions into fresh buffers under side-stream contention: every byte of the residual and of every output (the whole
    scale buffer included) matches the first execution; then the independent references.  At 1024 rows the grid is more thread
    blocks than one scheduling wave holds."""
    dev = _device_or_skip()
    k = make_entry(dev, form)
    x, res0, w = make_operands(rows, HIDDEN, dev, seed=200 + rows)
    scale = _scale(k)
    expected = assert_replays(lambda: run(k, x, res0, w, scale), repeats=8)
    check_reference(form, expected, x, res0, w, scale, f"replay {form} rows {rows}", share_bound=True)


@pytest.mark.parametrize("form", DETERMINISM_FORMS)
def test_replay_on_a_fresh_entry_and_fresh_operands(form):
    """Two entries, two identical operand sets drawn afresh: byte-identical (the entry, the operand storage and every output all
    fresh)."""
    dev = _device_or_skip()

    def once():
        x, res0, w = make_operands(24, HIDDEN, dev, seed=9, kind="structured")
        k = make_entry(dev, form)
        return snapshot(run(k, x, res0, w, _scale(k)))

    assert_bitwise(once(), once(), f"fresh entry, {form}")


@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
def test_launch_on_a_side_stream_writes_the_same_bytes(arm, arm_id):
    """The entry launches on torch's CURRENT stream of its device: the same call under a side stream (ordered after the operands'
    producer stream) writes the bytes of the default-stream call, under every launch arm."""
    dev = _device_or_skip()
    form = "add-bf16-nvfp4"
    k = make_entry(dev, form, arm)
    x, res0, w = make_operands(17, HIDDEN, dev, seed=61, kind="structured")
    scale = _scale(k)
    want = snapshot(run(k, x, res0, w, scale))
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))  # the operands were produced on the current stream
    with torch.cuda.stream(side):
        got = snapshot(run(k, x, res0, w, scale))
    side.synchronize()
    assert_bitwise(got, want, f"side stream vs the current stream, {arm_id}")


@pytest.mark.parametrize("form", DETERMINISM_FORMS)
def test_batch_invariance(form):
    """ONE operand set of 1024 rows.  Every row alone (a one-row call) writes the bytes it writes inside the 1024-row call, inside
    every shorter call that holds it (the first 2 .. 160 rows, on both sides of the 128-row scale-tile edge) and inside the reversed
    batch: the residual, ``out``, ``codes`` and the row's block scales.  The sum tree is a function of the width alone, so there is
    no split or association to hold fixed across the calls; what changes between "alone" and "inside" is the grid, the row's thread
    block, the scheduling wave it runs in and the position of its scale bytes."""
    dev = _device_or_skip()
    k = make_entry(dev, form)
    x, res0, w = make_operands(MAX_ROWS, HIDDEN, dev, seed=300, kind="structured")
    scale = _scale(k)
    replayed = assert_replays(lambda: run(k, x, res0, w, scale), repeats=3)
    check_reference(form, replayed, x, res0, w, scale, f"batch invariance {form}: the {MAX_ROWS}-row call")
    full = logical(replayed)
    singles = [rows_of(k, x[r : r + 1], res0[r : r + 1], w, scale) for r in range(MAX_ROWS)]
    alone = {name: torch.cat([one[name] for one in singles]) for name in full}
    assert_bitwise(alone, full, f"{form}: every row alone vs inside {MAX_ROWS} rows")
    for inside in INSIDE:
        batch = rows_of(k, x[:inside], res0[:inside], w, scale)
        assert_bitwise(batch, {name: t[:inside] for name, t in alone.items()}, f"{form}: the first {inside} rows as one call vs each row alone")
    reverse = list(range(MAX_ROWS - 1, -1, -1))
    flipped = rows_of(k, *take(x, res0, reverse), w, scale)
    assert_bitwise({name: t.flip(0) for name, t in flipped.items()}, alone, f"{form}: the reversed batch")


@pytest.mark.parametrize("form", DETERMINISM_FORMS)
def test_insertion_removal_and_reordering(form):
    """A 130-row call of ONE operand set (two scale tiles), then the same rows with a row inserted near the head (rows 5 .. 129 move
    down by one: row 127 enters the second scale tile), with three rows removed (rows 128 and 129 move up into the first tile) and
    permuted: every row keeps the bytes it has in the first call, and the inserted row writes the bytes it writes alone."""
    dev = _device_or_skip()
    n = 130
    k = make_entry(dev, form)
    x, res0, w = make_operands(n + 1, HIDDEN, dev, seed=700, kind="structured")  # row n is the row to insert
    scale = _scale(k)
    base = rows_of(k, x[:n], res0[:n], w, scale)
    new_row = rows_of(k, x[n:], res0[n:], w, scale)
    every = {name: torch.cat([base[name], new_row[name]]) for name in base}  # the bytes of operand row r at index r
    orders = {
        "a row inserted at position 5": list(range(5)) + [n] + list(range(5, n)),
        "rows 3, 11 and the last removed": [r for r in range(n) if r not in (3, 11, n - 1)],
        "a permutation": torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(7)).tolist(),
    }
    for what, order in orders.items():
        got = rows_of(k, *take(x, res0, order), w, scale)
        assert_bitwise(got, pick(every, order), f"{form}: {what}")


@pytest.mark.parametrize("form", DETERMINISM_FORMS)
def test_duplicated_row_and_neighbour_isolation(form):
    """A row twice in one call: both copies equal its alone bytes.  The same row among REDRAWN neighbours -- other values around it,
    another position, a taller call: at position 5 of 64 rows and at position 128 of 160 rows (the first row of the second scale
    tile) -- keeps its bytes."""
    dev = _device_or_skip()
    k = make_entry(dev, form)
    x, res0, w = make_operands(8, HIDDEN, dev, seed=400, kind="structured")
    scale = _scale(k)
    alone = rows_of(k, x[2:3], res0[2:3], w, scale)
    got = rows_of(k, *take(x, res0, [0, 1, 2, 2, 3, 4, 5, 6, 7]), w, scale)
    assert_bitwise(pick(got, [2]), alone, f"{form}: a duplicated row, first copy")
    assert_bitwise(pick(got, [3]), alone, f"{form}: a duplicated row, second copy")
    for position, others in ((5, 63), (128, 159)):
        other_x, other_res, _ = make_operands(others, HIDDEN, dev, seed=401 + others)
        mixed_x = torch.cat([other_x[:position], x[2:3], other_x[position:]]).contiguous()
        mixed_res = torch.cat([other_res[:position], res0[2:3], other_res[position:]]).contiguous()
        got = rows_of(k, mixed_x, mixed_res, w, scale)
        assert_bitwise(pick(got, [position]), alone, f"{form}: the row at position {position} among {others} redrawn neighbours")


def test_inputs_are_unmodified():
    """``x``, ``weight`` and ``scale`` hold the bytes they held before the launches -- every determinism form under every launch arm,
    caller-owned and entry-owned outputs, at 1, 17 and 1024 rows -- and the residual holds its documented update: bitwise
    ``bf16(x + residual)`` under ``add``, bitwise ``x`` under ``store_residual``."""
    dev = _device_or_skip()
    x, res0, w = make_operands(MAX_ROWS, HIDDEN, dev, seed=88, kind="structured")
    scales = {quant: scale_tensor(value, dev) for quant, value in SCALE_VALUES.items()}
    before = snapshot(dict(x=x, weight=w, **scales))
    launches = 0
    for form in DETERMINISM_FORMS:
        add, store, _, quant = FORMS[form]
        for arm in LAUNCH_ARMS:
            k = make_entry(dev, form, arm)
            for rows in (1, 17, MAX_ROWS):
                got = run(k, x[:rows], res0[:rows], w, scales.get(quant))
                residual = res0[:rows].clone() if (add or store) else None
                k(x[:rows], residual, w, scales.get(quant))
                launches += 2
                if residual is not None:
                    want = {"residual": residual_reference(x[:rows], res0[:rows], add).to(dev)}
                    assert_bitwise({"residual": got["residual"]}, want, f"{form} rows {rows}: the residual, caller-owned outputs")
                    assert_bitwise({"residual": residual}, want, f"{form} rows {rows}: the residual, entry-owned outputs")
    torch.cuda.synchronize()
    assert_bitwise(snapshot(dict(x=x, weight=w, **scales)), before, f"the inputs after {launches} launches")


@pytest.mark.parametrize("form", ("add-bf16-e4m3", "add-bf16-nvfp4"))
def test_crosses_a_scheduling_wave(form):
    """The row ceiling is more thread blocks than one scheduling wave holds on this part (``CTAS_PER_SM`` blocks of ``hidden / 8``
    threads per SM): asserted first, because a part on which it did not hold would not run this cell's premise.  Then eight
    executions of the 1024-row call under side-stream contention, and the rows on both sides of every wave edge alone against the
    same rows inside the call."""
    dev = _device_or_skip()
    k = make_entry(dev, form)
    wave = k.cfg.CTAS_PER_SM * torch.cuda.get_device_properties(dev).multi_processor_count
    assert MAX_ROWS > wave, f"{MAX_ROWS} rows do not cross a scheduling wave of {wave} thread blocks on this part: the cell's premise does not hold"
    x, res0, w = make_operands(MAX_ROWS, HIDDEN, dev, seed=900)
    scale = _scale(k)
    full = logical(assert_replays(lambda: run(k, x, res0, w, scale), repeats=8))
    edges = sorted({r for edge in range(0, MAX_ROWS + 1, wave) for r in (edge - 1, edge, edge + 1) if 0 <= r < MAX_ROWS} | {MAX_ROWS - 1})
    singles = [rows_of(k, x[r : r + 1], res0[r : r + 1], w, scale) for r in edges]
    alone = {name: torch.cat([one[name] for one in singles]) for name in full}
    assert_bitwise(alone, pick(full, edges), f"{form}: the rows at the wave edges ({wave} thread blocks per wave) alone vs inside {MAX_ROWS} rows")


def test_row_ceiling_is_the_typed_decline():
    """The row ceiling and the empty call decline by type -- never a launch, never another provider."""
    dev = _device_or_skip()
    k = make_entry(dev, "add-bf16-e4m3")
    x, res0, w = make_operands(MAX_ROWS + 1, HIDDEN, dev, seed=25)
    scale = _scale(k)
    snapshot(run(k, x[:MAX_ROWS], res0[:MAX_ROWS], w, scale))
    residual = res0.clone()
    with pytest.raises(NotImplementedError, match=f"at most {MAX_ROWS} rows per call; got {MAX_ROWS + 1}"):
        k(x, residual, w, scale)
    with pytest.raises(ValueError, match="at least one row"):
        k(x[:0], residual[:0], w, scale)
    torch.cuda.synchronize()
    assert_bitwise({"residual": residual}, {"residual": res0}, "a declined call launches nothing: the residual is untouched")


def _golden_operands(case, rows, dev):
    """The operands of a golden case, ``(x, residual, weight, y)``: a host draw (``y`` is ``None``), or the exact construction whose
    block scales 1.5 and 1.75 put fourteen elements of two blocks on exact E2M1 ties (``y`` is the output it places)."""
    if case == "ties":
        return exact_operands(exact_targets(), exact_signs(rows), dev)
    x, res0, w = make_operands(rows, HIDDEN, dev, seed=500 + rows, kind=case)
    return x, res0, w, None


@pytest.mark.parametrize(
    "form, tag, rows, case, scale_value",
    [
        pytest.param("add-bf16-e4m3", "add-bf16-e4m3", 3, "random", 0.02, id="add-bf16-e4m3-rows3"),
        pytest.param("store-e4m3", "store-e4m3", 3, "random", 0.02, id="store-e4m3-rows3"),
        pytest.param("none-e4m3", "none-e4m3", 3, "random", 1.0, id="none-e4m3-rows3"),
        pytest.param("add-nvfp4", "add-nvfp4", 3, "random", 37.0, id="add-nvfp4-rows3"),
        pytest.param("add-nvfp4", "add-nvfp4", 130, "structured", 0.25, id="add-nvfp4-rows130"),
        pytest.param("add-bf16-nvfp4", "add-bf16-nvfp4-ties", 2, "ties", 1.0, id="add-bf16-nvfp4-ties-rows2"),
    ],
)
def test_golden(form, tag, rows, case, scale_value):
    """Six small cases -- the three residual modes with static E4M3 codes, the NVFP4 codes in one and in two scale tiles, and the
    exact-tie construction under block scales that are not powers of two: three executions, the independent references, the launch
    arms on the same operands (each writes the default's bytes, so the one golden holds for every arm), then the saved hashes of the
    residual and of every output (the scale buffer whole, in its layout).  Recorded into a NEW directory with
    ``CUDNN_DETERMINISM_RECORD`` only after the replay and reference cells pass; a change of these hashes on one runtime is a change
    of the sum tree or of the rounding chain and needs a stated numerical reason.  The tie case's codes follow the part's approximate
    reciprocal: its golden holds on its own runtime and is not a claim across GPU architectures."""
    dev = _device_or_skip()
    k = make_entry(dev, form)
    x, res0, w, y = _golden_operands(case, rows, dev)
    scale = _scale(k, scale_value)
    label = f"golden {tag} rows {rows}"
    actual = assert_replays(lambda: run(k, x, res0, w, scale), repeats=3)
    check_reference(form, actual, x, res0, w, scale, label, share_bound=case == "random")
    if case == "ties":
        assert_bitwise({"out": actual["out"]}, {"out": y}, f"{label}: the exact construction places out on the targets")
        ties = assert_nvfp4_codes(actual["codes"], y, scale_bytes_of_rows(actual["scales"], rows), scale, label)
        assert ties >= rows * 2 * 14, f"{label}: {ties} elements sit on an E2M1 tie; the construction places {rows * 2 * 14}"
    for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
        other = snapshot(run(make_entry(dev, form, arm), x, res0, w, scale))
        assert_bitwise(other, actual, f"{label}: the {arm_id} arm")
    inputs = dict(x=x, weight=w)
    if k.cfg.ADD or k.cfg.STORE_RESIDUAL:
        inputs["residual"] = res0
    if scale is not None:
        inputs["scale"] = scale
    assert_golden(f"add-norm-quant-decode-h{HIDDEN}-{tag}-rows{rows}", inputs, actual)
