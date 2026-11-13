# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the gate / up product of an FP4 MLP for decode rows (``cudnn.SwigluFp4Decode``), forward direction, raw bytes only.

DIRECTIONS.  The entry is an inference step: it has ONE direction (fprop: FP4 E2M1 weight and activation with E4M3 block scales, two
fp32 scalars) and TWO results -- the packed FP4 codes ``out`` and the block scales ``out_scale`` -- with no scratch, no state and no
saved statistic.  There is NO backward entry point -- no gradient kernel and no autograd function -- so this module holds no backward
cell; that direction is unsupported by contract, not skipped.

WHAT IS COMPARED.  ``out`` ``[rows, I / 2]`` (every byte), the scale bytes of the live rows gathered from their positions in the
128 x 4 scale-atom order (``scale`` ``[rows, I / 16]``) and, where one call is compared with itself, the whole caller-owned scale band
(``scale_band``: the live bytes at their positions and the sentinel everywhere else, so a byte the call does not own is part of the
comparison).

WHAT IS PINNED, at both served geometries -- ``(5120, 17408)`` and ``(5120, 128)``:

* replay: eight executions into fresh buffers under side-stream contention at one row count per token tile, at the ceiling and under
  the launch attribute; then the independent fp64 reference under its bound;
* replay where a launch crosses scheduling boundaries: one thread block that runs every work tile, eight thread blocks (the staging
  ring is refilled inside the launch) and more thread blocks than the part has SMs (two waves);
* two fresh entries on two operand sets drawn afresh; the same call on a side stream;
* batch invariance: every row of an R-row call, for EVERY R = 2 .. 64, writes the bytes it writes ALONE (one row: the 8-token tile,
  token position 0) -- which crosses the four token tiles, every token position, every epilogue warp and both epilogue sub-tiles --,
  in the reversed batch and in a shuffled batch;
* insertion and removal: one row inserted at the head, in the middle or at the tail of a batch (8 -> 9, 16 -> 17 and 32 -> 33 rows
  cross a token tile, 63 -> 64 reaches the ceiling) leaves every other row's bytes as they were and writes its alone bytes; read the
  other way round it is the removal of that row; a permutation of the grown batch keeps every row's bytes;
* a duplicated row (both copies write the alone bytes); neighbours redrawn around a fixed row at every token tile -- with ordinary
  rows, with raw random codes and scales, and with rows whose scale bytes are E4M3 NaN codes;
* the token-tile pick is byte-neutral: at EVERY row count 1 .. 64 a wider token tile than the rows need writes the default's bytes;
* the operands hold their bytes after the launches; 64 -> 65 rows is the typed decline, never another kernel or provider;
* goldens: one small case per token tile at the default geometry and one at the small geometry, compared only after the replay and
  the reference passed.  Record a missing set into a NEW directory with ``CUDNN_DETERMINISM_RECORD``; a golden belongs to the one
  runtime that recorded it.

Every comparison of results is raw bytes (``determinism_utils``); the fp64 reference with its derived bound is a separate
correctness check (``swiglu_fp4_decode_utils.check_against_reference``).  The operands are drawn by a CPU generator, so the bytes of
an operand set do not depend on the device that runs the test.  The remaining performance knobs (the pre-wait stages, the L2 policy
of the weight stream, every grid cap, the launch attribute) are bitwise with the default in ``test_swiglu_fp4_decode.py``, which also
holds the CUDA-graph replay, the dependent launch that waits behind the entry and the warm-launch stress with byte references."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

import swiglu_fp4_decode_utils as U  # noqa: E402  (the operand builders, the reference and the device gate)

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot  # noqa: E402

from cudnn.gemm.frost.swiglu_decode import swiglu_fp4_decode as entry  # noqa: E402
from cudnn.gemm.frost.swiglu_decode.kernel import swiglu_fp4_decode_config as kcfg  # noqa: E402

# ``no_workspace_shim``: this directory's conftest wraps the GEMM engine's MoE plans for every test unless a test opts out, which would
# import the engine's compilers (and the DSL) into tests of an entry that never touches the engine.
pytestmark = [pytest.mark.L0, pytest.mark.no_workspace_shim]

Cfg = kcfg.Cfg
MAX_ROWS = kcfg.MAX_ROWS
GEOMETRIES = (U.G_DEFAULT, U.G_SMALL)
TILE_ROWS = (1, 9, 17, 33)  # the first row count of each token tile (8 / 16 / 32 / 64)
GOLDEN_CASES = ((U.G_DEFAULT, 3), (U.G_DEFAULT, 12), (U.G_DEFAULT, 20), (U.G_DEFAULT, 40), (U.G_SMALL, 20))  # one per token tile, + the small geometry
REFERENCE_TILES = {U.G_DEFAULT: (0, 135, 271), U.G_SMALL: (0, 1)}  # the weight tiles the reference is computed for (channels are independent)
SENTINEL = 0xA5  # what a fresh output buffer holds before the launch
OUT_GLOBAL_SCALE = 3.0  # not a power of two: both approximate reciprocals of the block scale are inexact

_ENTRIES = {}
_WEIGHTS = {}
_ACTS = {}
_ALONE = {}


def did(geometry):
    """Parameter id: direction, operand and result format, geometry."""
    return f"fprop-fp4-e4m3-h{geometry.HIDDEN}-i{geometry.INTERMEDIATE}"


def rid(rows):
    return f"rows{rows}"


def _entry(dev, geometry, **kw):
    key = (str(dev), geometry, tuple(sorted(kw.items())))
    if key not in _ENTRIES:
        _ENTRIES[key] = entry.SwigluFp4Decode(dev, geometry, **kw)
    return _ENTRIES[key]


def _weight(dev, geometry, seed=1):
    """``(weight in natural row order, its interleaved pair)``, built once per (geometry, seed)."""
    key = (str(dev), geometry, seed)
    if key not in _WEIGHTS:
        w = U.make_weight(geometry, dev, seed=seed, kind="random")
        _WEIGHTS[key] = (w, U.interleaved(w))
    return _WEIGHTS[key]


def _ops(dev, geometry, seed=0, kind="mixed", rows=MAX_ROWS):
    """``(launch operands, activation)`` of a ``rows``-row operand set, built once per key and sliced by the cells.  ``mixed``: ordinary
    rows alternating with an all-zero row, a row with one large value per block, a row of tiny values, a saturating row and a row
    with all-zero blocks -- so a batch always holds rows that take the zero rule and rows that saturate beside ordinary ones."""
    key = (str(dev), geometry, seed, kind, rows)
    if key not in _ACTS:
        weight, pair = _weight(dev, geometry)
        act = U.make_activation(geometry, rows, dev, seed=seed, kind=kind)
        _ACTS[key] = (U.launch_operands(weight, act, out_global_scale=OUT_GLOBAL_SCALE, pair=pair), act)
    return _ACTS[key]


def run(k, ops):
    """One launch into FRESH output buffers prefilled with a sentinel byte; the results compared as bytes are ``out`` [rows, I / 2],
    the live rows' scale bytes ``scale`` [rows, I / 16] and the whole scale band ``scale_band`` (sentinels outside the live rows)."""
    rows = ops["activation"].shape[0]
    out, band = U.run(k, ops, fill=SENTINEL)
    return {"out": out, "scale": U.live_scales(band, rows, k.geometry), "scale_band": band}


def run_rows(k, ops, act, index):
    """One launch of the rows ``index`` (in that order) of an operand set; ``(results, the sub-batch's operands, its activation)``."""
    sub_ops, sub_act = U.sub_batch(ops, act, index)
    return run(k, sub_ops), sub_ops, sub_act


def rows_of(snap, index):
    """The rows ``index`` (a slice or an index tensor) of the per-row results of a snapshot."""
    return {name: snap[name][index] for name in ("out", "scale")}


def per_row(snap):
    return {name: snap[name] for name in ("out", "scale")}


def _alone(dev, geometry):
    """Every row of the fixed operand set launched ALONE (one row: the 8-token tile, token position 0), built once per geometry."""
    key = (str(dev), geometry)
    if key not in _ALONE:
        k = _entry(dev, geometry)
        ops, act = _ops(dev, geometry)
        parts = [per_row(snapshot(run_rows(k, ops, act, [t])[0])) for t in range(MAX_ROWS)]
        _ALONE[key] = {name: torch.cat([p[name] for p in parts], dim=0) for name in ("out", "scale")}
    return _ALONE[key]


def _assert_untouched(snap, rows, geometry, label):
    """The scale bytes outside the live rows keep the sentinel, and the launch wrote something."""
    band = snap["scale_band"]
    live = U.live_mask(rows, geometry, band.device)
    assert int((band[~live] != SENTINEL).sum()) == 0, f"{label}: a scale byte outside the live rows was written"
    assert int((snap["out"] != SENTINEL).sum()) > 0, f"{label}: the launch wrote no code byte"


def _in_bound(snap, sub_ops, sub_act, geometry, label):
    """The independent reference: every block scale inside its admissible interval, every code inside its bound."""
    weight = _weight(sub_ops["weight"].device, geometry)[0]
    chk = U.check_against_reference(
        snap["out"], snap["scale_band"], weight, sub_act, float(sub_ops["alpha"]), float(sub_ops["out_global_scale"]), REFERENCE_TILES[geometry]
    )
    assert chk["scale_outside"] == 0, f"{label}: {chk['scale_outside']} of {chk['blocks']} block scales outside the admissible interval ({chk})"
    assert chk["code_ratio"] <= 1.0, f"{label}: a code is {chk['code_ratio']:.4f} of its bound away from the reference ({chk})"
    return chk


@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
@pytest.mark.parametrize("rows", TILE_ROWS + (MAX_ROWS,), ids=rid)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=did)
def test_replay(geometry, rows, pdl):
    """Eight executions into fresh output buffers under side-stream contention: every byte of ``out`` and of the scale band matches
    the first execution; then the fp64 reference under its bound.  One row count per token tile (each is its own compiled kernel) and
    the ceiling, without and with the launch attribute."""
    dev = U.device_or_skip()
    ops, act = _ops(dev, geometry)
    k = _entry(dev, geometry, pdl=pdl)
    sub_ops, sub_act = U.sub_batch(ops, act, list(range(rows)))
    expected = assert_replays(lambda: run(k, sub_ops), repeats=8)
    assert expected["out"].shape == (rows, geometry.INTERMEDIATE // 2) and expected["out"].dtype == torch.uint8
    assert expected["scale"].shape == (rows, geometry.INTERMEDIATE // kcfg.SF_VEC)
    _assert_untouched(expected, rows, geometry, f"{did(geometry)} rows {rows}")
    chk = _in_bound(expected, sub_ops, sub_act, geometry, f"{did(geometry)} rows {rows}")
    if rows >= 6:
        assert chk["zero_blocks"] > 0, "the operand set reaches the zero rule"


SCHEDULING_ARMS = ("one-block", "eight-blocks", "two-waves")


@pytest.mark.parametrize("arm", SCHEDULING_ARMS)
@pytest.mark.parametrize("rows", (20, MAX_ROWS), ids=rid)
def test_replay_across_scheduling_boundaries(rows, arm):
    """Replay where one launch crosses scheduling boundaries, at the default geometry (272 work tiles): ONE thread block runs every
    work tile; eight thread blocks run 34 each, so the staging ring is refilled inside the launch; and a grid cap of twice the part's
    SM count gives one thread block per work tile -- more than the part has SMs, a second wave.  Eight executions each, then the bytes
    of the default entry (the grid cap is a performance field)."""
    dev = U.device_or_skip()
    geometry = U.G_DEFAULT
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    cap = {"one-block": 1, "eight-blocks": 8, "two-waves": 2 * sms}[arm]
    k = _entry(dev, geometry, cfg=Cfg(GEOM=geometry, MAX_CLUSTERS=cap))
    blocks = k.launch_shape_for(rows).grid[2]
    assert blocks == min(cap, geometry.WEIGHT_TILES) and (arm != "two-waves" or blocks > sms), (blocks, sms)
    ops, act = _ops(dev, geometry)
    sub_ops, _ = U.sub_batch(ops, act, list(range(rows)))
    got = assert_replays(lambda: run(k, sub_ops), repeats=8)
    want = snapshot(run(_entry(dev, geometry), sub_ops))
    assert_bitwise(got, want, f"{arm} ({blocks} thread blocks) rows {rows} against the default entry")


@pytest.mark.parametrize("rows", (3, 12, 20, 40), ids=rid)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=did)
def test_replay_on_a_fresh_entry_and_fresh_operands(geometry, rows):
    """Two entries, two identical operand sets drawn afresh: byte-identical (the compile, the output buffers and every operand
    tensor fresh), at one row count per token tile."""
    dev = U.device_or_skip()
    results = []
    for _ in range(2):
        k = entry.SwigluFp4Decode(dev, geometry)
        weight = U.make_weight(geometry, dev, seed=9, kind="random")
        act = U.make_activation(geometry, rows, dev, seed=9, kind="mixed")
        results.append(snapshot(run(k, U.launch_operands(weight, act, out_global_scale=OUT_GLOBAL_SCALE))))
        del weight, act
    assert_bitwise(results[1], results[0], f"{did(geometry)} rows {rows}: a fresh entry on fresh operands")


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=did)
def test_launch_on_a_side_stream_writes_the_same_bytes(geometry):
    """The entry launches on torch's current stream of its device: the same call under a side stream (ordered behind the stream that
    produced the operands) writes the bytes of the call on the default stream, at each token tile."""
    dev = U.device_or_skip()
    ops, act = _ops(dev, geometry)
    k = _entry(dev, geometry)
    side = torch.cuda.Stream(device=dev)
    for rows in TILE_ROWS + (MAX_ROWS,):
        sub_ops, _ = U.sub_batch(ops, act, list(range(rows)))
        want = snapshot(run(k, sub_ops))
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            got = run(k, sub_ops)
        torch.cuda.current_stream(dev).wait_stream(side)
        assert_bitwise(snapshot(got), want, f"{did(geometry)} rows {rows}: the launch on a side stream")


@pytest.mark.parametrize("rows", tuple(range(2, MAX_ROWS + 1)), ids=rid)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=did)
def test_batch_invariance(geometry, rows):
    """Every row of an R-row call writes the bytes it writes alone, for EVERY served R = 2 .. 64 (so across the 8 / 16 / 32 / 64
    token tiles, every token position, the four epilogue warps and both epilogue sub-tiles of the 64-token tile), in the reversed
    batch and in a shuffled batch.  The operands are built once and sliced; the R-row call is replayed three times."""
    dev = U.device_or_skip()
    ops, act = _ops(dev, geometry)
    k = _entry(dev, geometry)
    alone = _alone(dev, geometry)
    first = list(range(rows))
    batch = assert_replays(lambda: run_rows(k, ops, act, first)[0], repeats=3)
    _assert_untouched(batch, rows, geometry, f"{did(geometry)} rows {rows}")
    assert_bitwise(per_row(batch), rows_of(alone, slice(0, rows)), f"{did(geometry)}: {rows} rows against each row alone")
    reverse = torch.arange(rows - 1, -1, -1, device=dev)
    shuffle = torch.randperm(rows, generator=torch.Generator(device="cpu").manual_seed(1000 + rows)).to(dev)
    for name, perm in (("reversed", reverse), ("shuffled", shuffle)):
        got = snapshot(run_rows(k, ops, act, perm)[0])
        assert_bitwise(per_row(got), rows_of(alone, perm), f"{did(geometry)}: the {name} batch of {rows} rows")


@pytest.mark.parametrize("rows", (8, 16, 32, 63), ids=rid)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=did)
def test_inserting_or_removing_a_row_keeps_the_other_rows_bytes(geometry, rows):
    """A batch of ``rows`` rows and the same rows with ONE new row inserted at the head, in the middle or at the tail (``rows + 1``
    rows: 8 -> 9, 16 -> 17 and 32 -> 33 change the token tile, 63 -> 64 reaches the ceiling): the original rows keep their bytes at
    their new positions and the inserted row writes the bytes it writes alone.  Read from the larger batch it is the removal of that
    row.  A permutation of the grown batch keeps every row's bytes."""
    dev = U.device_or_skip()
    ops, act = _ops(dev, geometry)
    _, other = _ops(dev, geometry, seed=77)
    k = _entry(dev, geometry)
    # one activation that holds the batch's rows and, behind them, the new row (an ordinary row of another draw)
    pool = dict(
        codes=torch.cat([act["codes"][:rows], other["codes"][6:7]]),
        scales=torch.cat([act["scales"][:rows], other["scales"][6:7]]),
        global_scale=act["global_scale"],
    )
    base = per_row(snapshot(run_rows(k, ops, pool, list(range(rows)))[0]))
    new_alone = per_row(snapshot(run_rows(k, ops, pool, [rows])[0]))
    for position in (0, rows // 2, rows):
        order = list(range(position)) + [rows] + list(range(position, rows))  # the new row at ``position``
        grown = snapshot(run_rows(k, ops, pool, order)[0])
        keep = torch.tensor([i for i in range(rows + 1) if i != position], device=dev)
        assert_bitwise(rows_of(grown, keep), base, f"{did(geometry)}: {rows} rows with one inserted at {position}, the original rows")
        assert_bitwise(rows_of(grown, slice(position, position + 1)), new_alone, f"{did(geometry)}: the row inserted at {position}")
    perm = torch.randperm(rows + 1, generator=torch.Generator(device="cpu").manual_seed(2000 + rows))
    permuted = snapshot(run_rows(k, ops, pool, perm.to(dev))[0])
    grown_in_order = {name: torch.cat([base[name], new_alone[name]]) for name in base}
    assert_bitwise(per_row(permuted), rows_of(grown_in_order, perm.to(dev)), f"{did(geometry)}: a permutation of the {rows + 1} rows")


def _nan_scaled(act):
    """An activation whose scale bytes are E4M3 NaN codes in every third block (out of the entry's contract for a live row)."""
    scales = act["scales"].clone()
    scales[:, ::3] = U.E4M3_NAN
    return dict(codes=act["codes"], scales=scales, global_scale=act["global_scale"])


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=did)
def test_duplicated_row_and_neighbour_isolation(geometry):
    """A row duplicated inside a batch: both copies write its alone bytes.  The other rows redrawn around it -- another ordinary
    draw, raw random codes with random scales, and rows whose scale bytes hold NaN codes -- with the row at the head, in the middle
    and at the tail of an 8-, a 16-, a 32- and a 64-row batch: its bytes do not move."""
    dev = U.device_or_skip()
    ops, act = _ops(dev, geometry)
    k = _entry(dev, geometry)
    alone = rows_of(_alone(dev, geometry), slice(2, 3))
    duplicated = snapshot(run_rows(k, ops, act, [0, 1, 2, 2, 3, 4, 5])[0])  # row 2 twice: positions 2 and 3
    assert_bitwise(rows_of(duplicated, slice(2, 3)), alone, f"{did(geometry)}: a duplicated row, first copy")
    assert_bitwise(rows_of(duplicated, slice(3, 4)), alone, f"{did(geometry)}: a duplicated row, second copy")
    neighbours = {
        "another draw": _ops(dev, geometry, seed=78)[1],
        "raw codes and scales": _ops(dev, geometry, seed=79, kind="raw")[1],
        "NaN scale codes": _nan_scaled(_ops(dev, geometry, seed=80, kind="random")[1]),
    }
    for name, other in neighbours.items():
        # the fixed row behind the 64 neighbours: index 64 of the pool
        pool = dict(
            codes=torch.cat([other["codes"], act["codes"][2:3]]), scales=torch.cat([other["scales"], act["scales"][2:3]]), global_scale=act["global_scale"]
        )
        for size in kcfg.TOKEN_TILES:
            for position in (0, size // 2, size - 1):
                order = list(range(position)) + [MAX_ROWS] + list(range(position, size - 1))
                assert len(order) == size
                got = snapshot(run_rows(k, ops, pool, order)[0])
                assert_bitwise(rows_of(got, slice(position, position + 1)), alone, f"{did(geometry)}: {name} around the row at {position} of {size}")


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=did)
def test_token_tile_pick_is_byte_neutral(geometry):
    """The token tile is a performance pick by the row count (the narrowest of 8 / 16 / 32 / 64 that holds the rows).  At EVERY row
    count 1 .. 64 an entry whose token tile starts at 16, at 32 or at 64 writes the default entry's bytes -- so every row count is
    compared on every token tile that holds it."""
    dev = U.device_or_skip()
    ops, act = _ops(dev, geometry)
    default = _entry(dev, geometry)
    wider = {floor: _entry(dev, geometry, cfg=Cfg(GEOM=geometry, TOKEN_TILE=floor)) for floor in kcfg.TOKEN_TILES[1:]}
    compared, tiles = 0, set()
    for rows in range(1, MAX_ROWS + 1):
        sub_ops, _ = U.sub_batch(ops, act, list(range(rows)))
        want = snapshot(run(default, sub_ops))
        own = default.cfg_for(rows).TOKEN_TILE
        assert own == kcfg.token_tile_for(rows)
        for floor, arm in wider.items():
            tile = arm.cfg_for(rows).TOKEN_TILE
            assert tile == max(floor, own)
            if tile == own:
                continue  # the same form as the default's
            assert_bitwise(snapshot(run(arm, sub_ops)), want, f"{did(geometry)} rows {rows}: token tile {tile} against {own}")
            compared += 1
            tiles.add((own, tile))
    print(f"{did(geometry)}: {compared} (rows, wider token tile) cells bitwise with the narrowest tile; pairs {sorted(tiles)}")
    assert compared == 8 * 3 + 8 * 2 + 16 * 1 and len(tiles) == 6, (compared, sorted(tiles))


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=did)
def test_inputs_are_unmodified(geometry):
    """Every operand holds the bytes it held before the launches: the row-alone form, both sides of each token-tile boundary, the
    launch attribute and a grid cap that refills the staging ring."""
    dev = U.device_or_skip()
    weight = U.make_weight(geometry, dev, seed=6, kind="random")
    act = U.make_activation(geometry, MAX_ROWS, dev, seed=600, kind="mixed")
    ops = U.launch_operands(weight, act, out_global_scale=OUT_GLOBAL_SCALE)
    entries = (_entry(dev, geometry), _entry(dev, geometry, pdl=True), _entry(dev, geometry, cfg=Cfg(GEOM=geometry, MAX_CLUSTERS=8)))
    before, subs = snapshot(ops), {}
    for rows in (1, 8, 9, 16, 17, 32, 33, MAX_ROWS):
        subs[rows] = U.sub_batch(ops, act, list(range(rows)))[0]
        kept = snapshot({name: subs[rows][name] for name in ("activation", "activation_scale")})
        for k in entries:
            run(k, subs[rows])
        torch.cuda.synchronize()
        assert_bitwise(snapshot({name: subs[rows][name] for name in ("activation", "activation_scale")}), kept, f"{did(geometry)}: the {rows}-row activation")
    assert_bitwise(snapshot(ops), before, f"{did(geometry)}: the operands after {len(entries) * len(subs)} launches")


def test_sixty_four_to_sixty_five_rows_is_the_typed_decline():
    """The row ceiling is a ``NotImplementedError`` at the launch and at the plan-time compile -- never another kernel or provider --
    and the declined call writes nothing."""
    dev = U.device_or_skip()
    geometry = U.G_SMALL
    k = _entry(dev, geometry)
    ops, act = _ops(dev, geometry, seed=33, rows=MAX_ROWS + 1)
    snapshot(run_rows(k, ops, act, list(range(MAX_ROWS)))[0])
    out = torch.full((MAX_ROWS + 1, geometry.INTERMEDIATE // 2), SENTINEL, dtype=torch.uint8, device=dev)
    band = U.new_outputs(geometry, MAX_ROWS, dev, SENTINEL)[1]
    with pytest.raises(NotImplementedError, match="at most 64 rows per call"):
        k(ops["weight"], ops["weight_scale"], ops["activation"], ops["activation_scale"], ops["alpha"], ops["out_global_scale"], out, band)
    with pytest.raises(NotImplementedError, match="at most 64 rows per call"):
        k.compile(MAX_ROWS + 1)
    torch.cuda.synchronize()
    assert int((out != SENTINEL).sum()) == 0 and int((band != SENTINEL).sum()) == 0, "the declined call wrote an output byte"


@pytest.mark.parametrize("geometry, rows", GOLDEN_CASES, ids=[f"{did(g)}-{rid(r)}" for g, r in GOLDEN_CASES])
def test_golden(geometry, rows):
    """One small case per token tile at the default geometry and one at the small geometry: three executions on a fresh entry, the
    independent reference, then the saved hashes of the six operands and of the results (recorded into a NEW directory with
    ``CUDNN_DETERMINISM_RECORD`` only after the replay and reference cells pass; a golden set belongs to the one runtime that
    recorded it, and a change of the rounding chain is a new set with a stated reason).  Every case draws its own activation."""
    dev = U.device_or_skip()
    weight, pair = _weight(dev, geometry)
    act = U.make_activation(geometry, rows, dev, seed=500 + rows, kind="mixed")
    ops = U.launch_operands(weight, act, out_global_scale=OUT_GLOBAL_SCALE, pair=pair)
    k = entry.SwigluFp4Decode(dev, geometry)
    actual = assert_replays(lambda: run(k, ops), repeats=3)
    _assert_untouched(actual, rows, geometry, f"{did(geometry)} golden rows {rows}")
    _in_bound(actual, ops, act, geometry, f"{did(geometry)} golden rows {rows}")
    cfg = k.cfg_for(rows)
    assert cfg.GEOM == geometry and cfg.TOKEN_TILE == kcfg.token_tile_for(rows)
    assert_golden(f"swiglu-fp4-decode-h{geometry.HIDDEN}-i{geometry.INTERMEDIATE}-tile{cfg.TOKEN_TILE}", ops, actual)
