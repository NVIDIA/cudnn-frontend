# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the FP8 split-K projection for decode rows (``cudnn.Fp8ProjectionDecode``), forward direction, raw bytes only.

DIRECTIONS.  The entry is an inference step: it has ONE direction (fprop: E4M3 weight and activation, one fp32 scale, bf16 ``out``) and
ONE result (``out``; no scratch, no state, no saved statistic).  There is NO backward entry point -- no gradient kernel and no autograd
function -- so this module holds no backward cell; that direction is unsupported by contract, not skipped.

WHAT IS PINNED, per served byte class ``(geometry, split_k)`` -- ``(4608, 8192)`` at split 4 and at split 2, ``(8192, 2048)`` at split
2 -- and for EVERY served row count 1 .. 32 (the kernel is specialised on the row count, so each count is its own rendering and gets its
own cells; nothing here is a sampled set):

* replay: eight executions on fresh output buffers under side-stream contention, then the independent fp64 reference;
* two fresh entries on two operand sets drawn afresh (the compile, the buffers and the operands all new);
* batch invariance: every row of an R-row call, R = 2 .. 32, writes the bytes it writes ALONE (the one-row rendering: the 8-row token
  tile, accumulator column 0) -- which crosses the three token tiles and every column position --, in the reversed batch and in a
  shuffled batch;
* insertion and removal: one row inserted at the head, in the middle or at the tail of a batch (8 -> 9 and 16 -> 17 rows cross a token
  tile, 31 -> 32 reaches the ceiling) leaves every other row's bytes as they were, and the inserted row writes its alone bytes; read
  the other way round it is the removal of that row;
* a duplicated row (both copies write the alone bytes) and neighbours redrawn around a fixed row at each token tile;
* the measured pick is byte-neutral: every form the pick returns on other parts' readings (fewer SMs, less shared memory per SM: other
  ring depths, three to nine CTAs per SM) and every band of the pick's own table applied to every row count -- a form that takes two
  waves among them -- writes the bytes of this part's pick;
* the operands hold their bytes after the launches; 32 -> 33 rows is the typed decline, never another provider;
* goldens: one small case per byte class and token tile (rows 3 / 12 / 20), compared only after the replay and the reference passed.
  Record a missing set into a NEW directory with ``CUDNN_DETERMINISM_RECORD``; a golden belongs to the one runtime that recorded it.

Every comparison of results is raw bytes (``determinism_utils``); the fp64 reference with its per-element bound is a separate
correctness check (``fp8_projection_decode_utils.error_over_bound``).  The operands are drawn by a CPU generator, so the bytes of an
operand set do not depend on the device that runs the test.  The remaining performance knobs (weight tile, tensor-memory columns,
a wider token tile, the weight loader's wait, the tensor-memory release, the dependent-launch attribute and release) are bitwise with
the default in ``test_fp8_projection_decode.py``, which also holds the CUDA-graph replay and the warm-launch stress."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from fp8_projection_decode_utils import CLASSES, cid, device_or_skip, error_over_bound, make_operands, new_out  # noqa: E402

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot  # noqa: E402

from cudnn.gemm.frost.projection_decode import fp8_projection_decode as entry  # noqa: E402
from cudnn.gemm.frost.projection_decode.kernel import fp8_projection_decode_config as kcfg  # noqa: E402

# ``no_workspace_shim``: this directory's conftest wraps the GEMM engine's MoE plans for every test unless a test opts out, which would
# import the engine's compilers (and the DSL) into tests of an entry that never touches the engine.
pytestmark = [pytest.mark.L0, pytest.mark.no_workspace_shim]

Form = kcfg.Form
MAX_ROWS = kcfg.MAX_ROWS
ALL_ROWS = tuple(range(1, MAX_ROWS + 1))
GOLDEN_ROWS = (3, 12, 20)  # one row count per token tile (8 / 16 / 32)

_ENTRIES = {}
_OPS = {}
_ALONE = {}


def did(numerics):
    """Parameter id: direction, operand dtype, result dtype, byte class."""
    return f"fprop-e4m3-bf16-{cid(numerics)}"


def rid(rows):
    return f"rows{rows}"


def _entry(dev, numerics, form=None):
    key = (str(dev), numerics, form)
    if key not in _ENTRIES:
        _ENTRIES[key] = entry.Fp8ProjectionDecode(dev, numerics.GEOM, split_k=numerics.SPLIT_K, form=form)
    return _ENTRIES[key]


def _ops(dev, geometry, seed=0, kind="wide"):
    """A 32-row operand set of a geometry, built once per (seed, kind) and sliced by the cells: every finite E4M3 code by default (large
    partial sums and cancellation, so that another association would show)."""
    key = (str(dev), geometry, seed, kind)
    if key not in _OPS:
        _OPS[key] = make_operands(geometry, MAX_ROWS, dev, seed, kind)
    return _OPS[key]


def run(k, ops, activation=None):
    """One launch into a FRESH output buffer (pre-filled with NaN: an element the kernel did not write fails the snapshot); the result
    compared as bytes is ``out`` [rows, OUT_FEATURES] bf16 -- the entry's only result."""
    x = ops["activation"] if activation is None else activation
    out = new_out(k.geometry, x.shape[0], x.device)
    k(ops["weight"], x, ops["scale"], out)
    return {"out": out}


def rows_of(snap, index):
    """The rows ``index`` (a slice or an index tensor) of a snapshot, as a snapshot."""
    return {name: value[index] for name, value in snap.items()}


def _alone(dev, numerics):
    """Every row of the fixed operand set launched ALONE (one row: the 8-row token tile, accumulator column 0), built once per class."""
    key = (str(dev), numerics)
    if key not in _ALONE:
        k, ops = _entry(dev, numerics), _ops(dev, numerics.GEOM)
        parts = [snapshot(run(k, ops, ops["activation"][t : t + 1]))["out"] for t in range(MAX_ROWS)]
        _ALONE[key] = {"out": torch.cat(parts, dim=0)}
    return _ALONE[key]


def _in_bound(out, ops, activation, numerics, label):
    ratio, bad = error_over_bound(out, dict(ops, activation=activation), numerics.SPLIT_K)
    assert bad == 0, f"{label}: {bad} elements are non-finite where the reference is finite (or the reverse)"
    assert ratio <= 1.0, f"{label}: max |out - ref| / bound = {ratio:.4f} > 1"


@pytest.mark.parametrize("rows", ALL_ROWS, ids=rid)
@pytest.mark.parametrize("numerics", CLASSES, ids=did)
def test_replay(numerics, rows):
    """Eight executions on fresh output buffers under side-stream contention: every byte of ``out`` matches the first execution; then
    the fp64 reference under its bound.  One cell per served row count (each is its own rendering)."""
    dev = device_or_skip()
    ops = _ops(dev, numerics.GEOM)
    k = _entry(dev, numerics)
    x = ops["activation"][:rows]
    expected = assert_replays(lambda: run(k, ops, x), repeats=8)
    assert expected["out"].shape == (rows, numerics.GEOM.OUT_FEATURES) and expected["out"].dtype == torch.bfloat16
    _in_bound(expected["out"], ops, x, numerics, f"{did(numerics)} rows {rows}")


@pytest.mark.parametrize("rows", GOLDEN_ROWS, ids=rid)
@pytest.mark.parametrize("numerics", CLASSES, ids=did)
def test_replay_on_a_fresh_entry_and_fresh_operands(numerics, rows):
    """Two entries, two identical operand sets drawn afresh: byte-identical (the compile, the output buffers and the operands all
    fresh), at one row count per token tile."""
    dev = device_or_skip()
    results = []
    for _ in range(2):
        k = entry.Fp8ProjectionDecode(dev, numerics.GEOM, split_k=numerics.SPLIT_K)
        results.append(snapshot(run(k, make_operands(numerics.GEOM, rows, dev, seed=9, kind="wide"))))
    assert_bitwise(results[1], results[0], f"{did(numerics)} rows {rows}: a fresh entry on fresh operands")


@pytest.mark.parametrize("rows", ALL_ROWS[1:], ids=rid)
@pytest.mark.parametrize("numerics", CLASSES, ids=did)
def test_batch_invariance(numerics, rows):
    """Every row of an R-row call writes the bytes it writes alone, for EVERY served R = 2 .. 32 (so across the 8 / 16 / 32 token tiles
    and every accumulator column), in the reversed batch and in a shuffled batch.  The operands are built once and sliced."""
    dev = device_or_skip()
    ops = _ops(dev, numerics.GEOM)
    k = _entry(dev, numerics)
    alone = _alone(dev, numerics)
    x = ops["activation"][:rows]
    batch = assert_replays(lambda: run(k, ops, x), repeats=3)
    assert_bitwise(batch, rows_of(alone, slice(0, rows)), f"{did(numerics)}: {rows} rows against each row alone")
    reverse = torch.arange(rows - 1, -1, -1, device=dev)
    shuffle = torch.randperm(rows, generator=torch.Generator(device="cpu").manual_seed(1000 + rows)).to(dev)
    for name, perm in (("reversed", reverse), ("shuffled", shuffle)):
        got = snapshot(run(k, ops, x[perm].contiguous()))
        assert_bitwise(got, rows_of(alone, perm), f"{did(numerics)}: the {name} batch of {rows} rows")


@pytest.mark.parametrize("rows", (8, 16, 31), ids=rid)
@pytest.mark.parametrize("numerics", CLASSES, ids=did)
def test_inserting_or_removing_a_row_keeps_the_other_rows_bytes(numerics, rows):
    """A batch of ``rows`` rows and the same rows with ONE new row inserted at the head, in the middle or at the tail (``rows + 1``
    rows: 8 -> 9 and 16 -> 17 change the token tile, 31 -> 32 reaches the ceiling): the original rows keep their bytes at their new
    positions and the inserted row writes the bytes it writes alone.  Read from the larger batch it is the removal of that row."""
    dev = device_or_skip()
    ops, other = _ops(dev, numerics.GEOM), _ops(dev, numerics.GEOM, seed=77)
    k = _entry(dev, numerics)
    x, new_row = ops["activation"][:rows], other["activation"][5:6]
    base = snapshot(run(k, ops, x))
    new_alone = snapshot(run(k, ops, new_row))
    for position in (0, rows // 2, rows):
        grown = snapshot(run(k, ops, torch.cat([x[:position], new_row, x[position:]], dim=0).contiguous()))
        keep = torch.tensor([i for i in range(rows + 1) if i != position], device=dev)
        assert_bitwise(rows_of(grown, keep), base, f"{did(numerics)}: {rows} rows with one inserted at {position}, the original rows")
        assert_bitwise(rows_of(grown, slice(position, position + 1)), new_alone, f"{did(numerics)}: the row inserted at {position}")


@pytest.mark.parametrize("numerics", CLASSES, ids=did)
def test_duplicated_row_and_redrawn_neighbours(numerics):
    """A row duplicated inside a batch: both copies write its alone bytes.  The other rows redrawn around it (another operand draw),
    with the row at the head, in the middle and at the tail of an 8-, a 16- and a 32-row batch: its bytes do not move."""
    dev = device_or_skip()
    ops, other = _ops(dev, numerics.GEOM), _ops(dev, numerics.GEOM, seed=78)
    k = _entry(dev, numerics)
    x, y = ops["activation"], other["activation"]
    alone = rows_of(_alone(dev, numerics), slice(2, 3))
    duplicated = snapshot(run(k, ops, torch.cat([x[:3], x[2:3], x[3:6]], dim=0).contiguous()))  # row 2 twice: positions 2 and 3
    assert_bitwise(rows_of(duplicated, slice(2, 3)), alone, f"{did(numerics)}: a duplicated row, first copy")
    assert_bitwise(rows_of(duplicated, slice(3, 4)), alone, f"{did(numerics)}: a duplicated row, second copy")
    for size in (8, 16, 32):
        for position in (0, size // 2, size - 1):
            mixed = torch.cat([y[:position], x[2:3], y[position : size - 1]], dim=0).contiguous()
            assert mixed.shape[0] == size
            got = snapshot(run(k, ops, mixed))
            assert_bitwise(rows_of(got, slice(position, position + 1)), alone, f"{did(numerics)}: neighbours redrawn, row at {position} of {size}")


# The readings the pick is evaluated on besides this part's own: (multiprocessors, shared memory per SM, per-CTA opt-in).  They are MODELS
# fed to the pick function -- the launches run on this part; what a modelled part would place is not claimed here.
PICK_MODELS = {
    "148 SMs": (148, kcfg.SM100_SMEM_PER_SM_BYTES, kcfg.SM100_SMEM_OPTIN_BYTES),
    "212 SMs": (212, kcfg.SM100_SMEM_PER_SM_BYTES, kcfg.SM100_SMEM_OPTIN_BYTES),
    "64 SMs": (64, kcfg.SM100_SMEM_PER_SM_BYTES, kcfg.SM100_SMEM_OPTIN_BYTES),
    "100 KB of shared memory per SM": (kcfg.SM100_SM_COUNT, 100 * 1024, 99 * 1024),
}


def pick_arms(numerics, rows, default):
    """The forms the pick function can return for ``rows`` other than this part's ``default``: the pick on each modelled part and every
    band of the pick's own table (the band a row count is not in included).  ``{form: [where it comes from, ...]}``."""
    arms = {}
    for name, (sm_count, smem_per_sm, smem_optin) in PICK_MODELS.items():
        arms.setdefault(kcfg.pick(numerics, rows, sm_count, smem_per_sm, smem_optin), []).append(name)
    for lo, hi, mma_m, stages in kcfg.PICKS[numerics]:
        arms.setdefault(Form(MMA_M=mma_m, STAGES=stages, TMEM_COLS=kcfg.tmem_cols_for(kcfg.token_tile_for(rows))), []).append(f"the band of rows {lo}-{hi}")
    arms.pop(default, None)
    return arms


def test_pick_arms_cover_other_ring_depths_at_every_class():
    """Host: the arms of :func:`test_pick_is_byte_neutral` are not vacuous -- at every served class the modelled parts make the pick
    return at least two ring depths other than the default's, a pick never names a split, and every arm is a form the reference part
    validates at the row count it is used at."""
    part = kcfg.part_readings(kcfg.SM100_SMEM_PER_SM_BYTES, kcfg.SM100_SMEM_OPTIN_BYTES, kcfg.SM100_SM_COUNT)
    for numerics in CLASSES:
        depths = set()
        for rows in ALL_ROWS:
            default = kcfg.pick(numerics, rows, kcfg.SM100_SM_COUNT, kcfg.SM100_SMEM_PER_SM_BYTES, kcfg.SM100_SMEM_OPTIN_BYTES)
            for form in pick_arms(numerics, rows, default):
                assert isinstance(form, Form) and not hasattr(form, "SPLIT_K")
                kcfg.validate_cfg(kcfg.resolve_form(numerics, rows, form, part))
                depths.add(form.STAGES)
        assert len(depths) >= 2, (cid(numerics), sorted(depths))


@pytest.mark.parametrize("numerics", CLASSES, ids=did)
def test_pick_is_byte_neutral(numerics):
    """The pick function reads the row count and the part's readings and returns a performance form; no form it can return changes a
    byte.  For EVERY served row count: this part's pick against the pick on each modelled part (fewer SMs or less shared memory lower
    the ring depth: three to nine CTAs per SM) and against every band of the pick's table (a deep ring at the wider token tiles takes
    one CTA per SM and two waves), each built as the entry's ``form`` on this part."""
    dev = device_or_skip()
    ops = _ops(dev, numerics.GEOM)
    default = _entry(dev, numerics)
    part = default.readings
    compared, forms, waves = 0, set(), set()
    for rows in ALL_ROWS:
        x = ops["activation"][:rows]
        want = snapshot(run(default, ops, x))
        own = kcfg.pick(numerics, rows, part["SM_COUNT"], part["SMEM_PER_SM_BYTES"], part["SMEM_OPTIN_BYTES"])  # what the default entry runs
        assert (default.cfg_for(rows).MMA_M, default.cfg_for(rows).STAGES) == (own.MMA_M, own.STAGES)
        for form, sources in pick_arms(numerics, rows, own).items():
            arm = _entry(dev, numerics, form=form)
            cfg = arm.cfg_for(rows)
            assert cfg.NUMERICS == numerics and (cfg.MMA_M, cfg.STAGES) == (form.MMA_M, form.STAGES)
            assert_bitwise(snapshot(run(arm, ops, x)), want, f"{did(numerics)} rows {rows}: the pick of {' / '.join(sources)} ({cfg.describe()})")
            compared += 1
            forms.add((form.MMA_M, form.STAGES))
            waves.add(cfg.WAVES)
    print(f"{did(numerics)}: {compared} (rows, form) cells bitwise with this part's pick; ring depths {sorted(s for _, s in forms)}; waves {sorted(waves)}")
    assert compared >= MAX_ROWS and len(forms) >= 2, (compared, sorted(forms))


@pytest.mark.parametrize("numerics", CLASSES, ids=did)
def test_inputs_are_not_modified(numerics):
    """Every operand holds the bytes it held before the launches (one launch per token tile and the row-alone form)."""
    dev = device_or_skip()
    ops = make_operands(numerics.GEOM, MAX_ROWS, dev, seed=600, kind="wide")
    k = _entry(dev, numerics)
    before = snapshot(ops)
    for rows in (1, 8, 9, 16, 17, 32):
        run(k, ops, ops["activation"][:rows])
    torch.cuda.synchronize()
    assert_bitwise(snapshot(ops), before, f"{did(numerics)}: the operands after six launches")


def test_thirty_two_to_thirty_three_rows_is_the_typed_decline():
    """The row ceiling is a ``NotImplementedError`` at the launch and at the plan-time compile -- never another kernel or provider."""
    dev = device_or_skip()
    numerics = CLASSES[2]
    k = _entry(dev, numerics)
    ops = make_operands(numerics.GEOM, MAX_ROWS + 1, dev, seed=33)
    snapshot(run(k, ops, ops["activation"][:MAX_ROWS]))
    with pytest.raises(NotImplementedError, match="at most 32 rows"):
        run(k, ops)
    with pytest.raises(NotImplementedError, match="at most 32 rows"):
        k.compile(MAX_ROWS + 1)


@pytest.mark.parametrize("rows", GOLDEN_ROWS, ids=rid)
@pytest.mark.parametrize("numerics", CLASSES, ids=did)
def test_golden(numerics, rows):
    """One small case per byte class and token tile: three executions on a fresh entry, the independent reference, then the saved
    hashes of the operands and of ``out`` (recorded into a NEW directory with ``CUDNN_DETERMINISM_RECORD`` only after the replay and
    reference cells pass; a golden set belongs to the one runtime that recorded it, and a change of the split, of the K tile or of the
    cast is a new set with a stated reason)."""
    dev = device_or_skip()
    ops = make_operands(numerics.GEOM, rows, dev, seed=500 + rows, kind="wide")
    k = entry.Fp8ProjectionDecode(dev, numerics.GEOM, split_k=numerics.SPLIT_K)
    actual = assert_replays(lambda: run(k, ops), repeats=3)
    _in_bound(actual["out"], ops, ops["activation"], numerics, f"{did(numerics)} golden rows {rows}")
    cfg = k.cfg_for(rows)
    assert cfg.NUMERICS == numerics and cfg.TOKEN_TILE == kcfg.token_tile_for(rows) and cfg.ROWS == rows
    g = numerics.GEOM
    assert_golden(f"fp8-projection-decode-n{g.OUT_FEATURES}-k{g.IN_FEATURES}-split{numerics.SPLIT_K}-tile{cfg.TOKEN_TILE}", ops, actual)
