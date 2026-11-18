# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the attention gate + quantize decode step (``cudnn.AttentionGateQuantDecode``), raw bytes throughout: bitwise replay
into fresh output buffers under side-stream contention, up to the widest served slab (``160 x 32768`` = 2,560 CTAs: several waves on any
SM100-class part); a fresh entry on fresh identical operands; a launch on a side stream; batch invariance of every row -- alone (a one-row
call) against the same row inside the first 2 .. 160 rows of ONE operand set, inside the reversed batch, after an insertion, a removal
and a permutation of its neighbours, duplicated inside one call, and with every other row redrawn -- at widths whose rows are whole CTAs
(2048, 6144) and at widths whose rows share CTAs (1536) and warps (1032), under both served thread counts; the pick function (a pick
table that depends on the row count moves no byte); the inputs unmodified by a launch; the row ceiling and the empty call as typed
declines; and the goldens (two small cases, recorded with ``CUDNN_DETERMINISM_RECORD=<new dir>`` only after the replay and the reference
pass).

DIRECTIONS.  The step is an INFERENCE step: forward only.  It has no backward entry, no gradient and no state, so this module has no
backward cell -- that direction does not exist by contract; it is not a coverage gap of an existing path.  The one result is ``out`` (the
E4M3 codes); the entry owns no scratch and mutates no operand.  An empty call (zero rows) is not served: it is a typed decline, pinned
below, so there is no empty-sequence cell.

WHAT IS HELD FIXED.  The kernel has no reduction, hence no split, no association and no order to pin: a row's bytes are a function of
its own ``attn`` / ``gate`` values and of the scale.  The cells still build the operands ONCE and slice / permute them (the row under
test is never redrawn), keep the one scale tensor, and compare ``out`` as bytes only.  The independent reference
(``gate_quant_utils.assert_reference``: the fp64 bracket, the true-nearest margin rule, the torch model of the convention) is a separate
check and gates the goldens."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from gate_quant_utils import _device_or_skip, assert_reference, make_operands, scale_tensor  # noqa: E402

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot  # noqa: E402

from cudnn.attention_gate.frost import gate_quant_decode as entry  # noqa: E402
from cudnn.attention_gate.frost.kernel import gate_quant_decode_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

U8 = torch.uint8
E4M3 = torch.float8_e4m3fn
MAX_ROWS = kcfg.MAX_ROWS
INSIDE = (2, 3, 8, 16, 17, 24, 32, 64, MAX_ROWS)
# a row is 1 / 3 whole CTAs of 256 threads at widths 2048 / 6144 (2 / 6 of 128); at 1536 rows share CTAs under both thread counts, and
# at 1032 (129 threads per row) they share warps too -- there "alone" and "inside a batch" differ in what a CTA and a warp hold
WIDTHS = (2048, 6144, 1032, 1536)

_width_id = lambda width: f"w{width}"  # noqa: E731
_threads_id = lambda threads: f"threads{threads}"  # noqa: E731


def run(k, attn, gate, scale):
    """One launch into a FRESH output buffer (never the entry-owned one, so a reused buffer cannot make a comparison pass)."""
    out = torch.empty(attn.shape[0], attn.shape[1], dtype=E4M3, device=attn.device)
    k(attn, gate, scale, out)
    return {"out": out}


def codes(k, attn, gate, scale):
    """The bytes of one launch, snapshotted before the next one: ``{"out": uint8 [rows, width]}``."""
    return {"out": snapshot(run(k, attn, gate, scale))["out"].view(U8)}


def take(attn, gate, order):
    """The rows ``order`` of ONE operand set as a new contiguous call (the values are moved, never redrawn)."""
    index = torch.tensor(order, dtype=torch.long, device=attn.device)
    return attn[index].contiguous(), gate[index].contiguous()


@pytest.mark.parametrize(
    "rows, width",
    [pytest.param(rows, width, id=f"rows{rows}-w{width}") for width in (2048, 6144) for rows in (1, 17, MAX_ROWS)]
    + [pytest.param(MAX_ROWS, kcfg.MAX_WIDTH, id=f"rows{MAX_ROWS}-w{kcfg.MAX_WIDTH}-multiwave")],
)
def test_replay(rows, width):
    """Eight executions into fresh output buffers under side-stream contention: every byte matches the first execution; then the
    independent reference.  The last cell is the widest served slab: 2,560 CTAs of 256 threads, more than one wave on any SM100-class
    part, so the grid crosses a scheduling boundary."""
    dev = _device_or_skip()
    attn, gate = make_operands(rows, width, dev, seed=200 + rows, kind="adversarial")
    scale = scale_tensor(0.01, dev)
    k = entry.AttentionGateQuantDecode(dev, width)
    expected = assert_replays(lambda: run(k, attn, gate, scale), repeats=8)
    assert_reference(expected["out"], attn, gate, scale, f"replay rows {rows} width {width}")


@pytest.mark.parametrize("width", (2048, 1032), ids=_width_id)
def test_replay_on_a_fresh_entry_and_fresh_operands(width):
    """Two entries, two identical operand sets drawn afresh: byte-identical (the compile, the operand storage and the output all fresh)."""
    dev = _device_or_skip()

    def once():
        attn, gate = make_operands(24, width, dev, seed=9, kind="adversarial")
        return snapshot(run(entry.AttentionGateQuantDecode(dev, width), attn, gate, scale_tensor(0.02, dev)))

    assert_bitwise(once(), once(), f"fresh entry, width {width}")


def test_launch_on_a_side_stream_writes_the_same_bytes():
    """The entry launches on torch's CURRENT stream of its device: the same call under a side stream (ordered after the operands'
    producer stream) writes the bytes of the default-stream call."""
    dev = _device_or_skip()
    attn, gate = make_operands(17, 2048, dev, seed=61, kind="adversarial")
    scale = scale_tensor(0.01, dev)
    k = entry.AttentionGateQuantDecode(dev, 2048)
    want = snapshot(run(k, attn, gate, scale))
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))  # the operands were produced on the current stream
    with torch.cuda.stream(side):
        got = snapshot(run(k, attn, gate, scale))
    side.synchronize()
    assert_bitwise(got, want, "side stream vs the current stream")


@pytest.mark.parametrize("threads", kcfg.SERVED_THREADS, ids=_threads_id)
@pytest.mark.parametrize("width", WIDTHS, ids=_width_id)
def test_batch_invariance(width, threads):
    """ONE operand set of 160 rows.  Every row alone (a one-row call) writes the bytes it writes inside the 160-row call, inside every
    shorter call that holds it (the first 2, 3, 8, 16, 17, 24, 32, 64 rows) and inside the reversed batch -- under the thread count of
    the cell.  There is no split or association to hold fixed across the calls: the kernel has no reduction; what changes between "alone"
    and "inside" is the grid, the row's CTA and (at widths 1032 / 1536) the rows that share its CTA or warp."""
    dev = _device_or_skip()
    attn, gate = make_operands(MAX_ROWS, width, dev, seed=300 + width, kind="adversarial")
    scale = scale_tensor(0.01, dev)
    k = entry.AttentionGateQuantDecode(dev, width, cfg=kcfg.Cfg(THREADS=threads))
    label = f"width {width} THREADS {threads}"
    full = {"out": assert_replays(lambda: run(k, attn, gate, scale), repeats=3)["out"].view(U8)}
    alone = {"out": torch.cat([codes(k, attn[r : r + 1], gate[r : r + 1], scale)["out"] for r in range(MAX_ROWS)])}
    assert_bitwise(alone, full, f"{label}: every row alone vs inside {MAX_ROWS} rows")
    for inside in INSIDE[:-1]:
        batch = codes(k, attn[:inside], gate[:inside], scale)
        assert_bitwise(batch, {"out": alone["out"][:inside]}, f"{label}: the first {inside} rows as one call vs each row alone")
    reverse = list(range(MAX_ROWS - 1, -1, -1))
    flipped = codes(k, *take(attn, gate, reverse), scale)
    assert_bitwise({"out": flipped["out"].flip(0)}, alone, f"{label}: the reversed batch")


@pytest.mark.parametrize("threads", kcfg.SERVED_THREADS, ids=_threads_id)
@pytest.mark.parametrize("width", (2048, 1032), ids=_width_id)
def test_insertion_removal_and_reordering(width, threads):
    """A 24-row call of ONE operand set, then the same rows with a row inserted in the middle, with three rows removed, and permuted:
    every row keeps the bytes it has in the first call (the inserted row: the bytes it writes alone).  At width 1032 a moved row starts
    at another thread and another CTA boundary."""
    dev = _device_or_skip()
    n = 24
    attn, gate = make_operands(n + 1, width, dev, seed=700 + width, kind="adversarial")  # row n is the row to insert
    scale = scale_tensor(1.0, dev)
    k = entry.AttentionGateQuantDecode(dev, width, cfg=kcfg.Cfg(THREADS=threads))
    base = codes(k, attn[:n], gate[:n], scale)["out"]
    new_row = codes(k, attn[n:], gate[n:], scale)["out"]
    every = torch.cat([base, new_row])  # the bytes of operand row r at index r
    orders = {
        "a row inserted at position 5": list(range(5)) + [n] + list(range(5, n)),
        "rows 3, 11 and the last removed": [r for r in range(n) if r not in (3, 11, n - 1)],
        "a permutation": torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(7)).tolist(),
    }
    for what, order in orders.items():
        got = codes(k, *take(attn, gate, order), scale)
        want = every[torch.tensor(order, dtype=torch.long, device=dev)]
        assert_bitwise(got, {"out": want}, f"width {width} THREADS {threads}: {what}")


@pytest.mark.parametrize("threads", kcfg.SERVED_THREADS, ids=_threads_id)
@pytest.mark.parametrize("width", WIDTHS, ids=_width_id)
def test_duplicated_row_and_neighbour_isolation(width, threads):
    """A row twice in one call: both copies equal its alone bytes.  The same row among 63 REDRAWN neighbours (other values around it,
    another position, a taller call): its bytes are unchanged."""
    dev = _device_or_skip()
    attn, gate = make_operands(8, width, dev, seed=400 + width, kind="adversarial")
    scale = scale_tensor(0.01, dev)
    k = entry.AttentionGateQuantDecode(dev, width, cfg=kcfg.Cfg(THREADS=threads))
    label = f"width {width} THREADS {threads}"
    alone = codes(k, attn[2:3], gate[2:3], scale)
    got = codes(k, *take(attn, gate, [0, 1, 2, 2, 3, 4, 5, 6, 7]), scale)["out"]
    assert_bitwise({"out": got[2:3]}, alone, f"{label}: a duplicated row, first copy")
    assert_bitwise({"out": got[3:4]}, alone, f"{label}: a duplicated row, second copy")
    other_attn, other_gate = make_operands(63, width, dev, seed=401 + width)
    mixed_attn = torch.cat([other_attn[:5], attn[2:3], other_attn[5:]]).contiguous()
    mixed_gate = torch.cat([other_gate[:5], gate[2:3], other_gate[5:]]).contiguous()
    got = codes(k, mixed_attn, mixed_gate, scale)["out"]
    assert_bitwise({"out": got[5:6]}, alone, f"{label}: the row among redrawn neighbours")


def test_pick_function_moves_no_byte(monkeypatch):
    """The pick (``default_cfg``: threads per CTA from the element count and the part) is performance-only.  (1) Every thread count the
    table may return writes the default entry's bytes.  (2) With a pick table that DEPENDS ON THE ROW COUNT (128 threads up to 8 rows,
    256 above), the entry resolves two configs, ``compile()`` builds both (a warm call at any row count never compiles), and every row
    alone (the 128-thread config) writes the bytes it writes inside 2 .. 160 rows on either side of the pick boundary -- so a row added
    to the table later cannot make a result depend on the batch."""
    dev = _device_or_skip()
    width = 2048
    attn, gate = make_operands(MAX_ROWS, width, dev, seed=77, kind="adversarial")
    scale = scale_tensor(0.01, dev)
    default = entry.AttentionGateQuantDecode(dev, width)
    assert {default.cfg_for(rows).THREADS for rows in range(1, MAX_ROWS + 1)} <= set(kcfg.SERVED_THREADS)
    want = codes(default, attn, gate, scale)
    for threads in kcfg.SERVED_THREADS:
        arm = entry.AttentionGateQuantDecode(dev, width, cfg=kcfg.Cfg(THREADS=threads))
        assert_bitwise(codes(arm, attn, gate, scale), want, f"THREADS {threads} vs the default pick")
    monkeypatch.setattr(kcfg, "THREADS_PICKS", ((0, 999, 8 * width, 128), (0, 999, kcfg.ELEMENT_LIMIT, 256)))
    k = entry.AttentionGateQuantDecode(dev, width)
    assert [k.cfg_for(rows).THREADS for rows in (1, 8, 9, MAX_ROWS)] == [128, 128, 256, 256] and len(k.cfgs) == 2
    k.compile(attn[:1], gate[:1], scale)
    assert len(k.compiled) == 2, "compile() builds every config the pick table can return"
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("a warm call compiled")

    monkeypatch.setattr(cute, "compile", forbidden)
    for inside in (1, 2, 8, 9, 17, MAX_ROWS):
        assert_bitwise(codes(k, attn[:inside], gate[:inside], scale), {"out": want["out"][:inside]}, f"row-dependent pick, {inside} rows")
    alone = {"out": torch.cat([codes(k, attn[r : r + 1], gate[r : r + 1], scale)["out"] for r in range(MAX_ROWS)])}
    assert_bitwise(alone, want, "row-dependent pick: every row alone (128 threads) vs inside 160 rows (256 threads)")


def test_inputs_are_unmodified():
    """``attn``, ``gate`` and ``scale`` hold the bytes they held before the launches (caller-owned and entry-owned output, the release
    arm included: the kernel reads its inputs after the release)."""
    dev = _device_or_skip()
    attn, gate = make_operands(MAX_ROWS, 2048, dev, seed=88, kind="adversarial")
    scale = scale_tensor(0.02, dev)
    before = snapshot({"attn": attn, "gate": gate, "scale": scale})
    for k in (entry.AttentionGateQuantDecode(dev, 2048), entry.AttentionGateQuantDecode(dev, 2048, pdl_release=True)):
        for rows in (1, 17, MAX_ROWS):
            run(k, attn[:rows], gate[:rows], scale)
            k(attn[:rows], gate[:rows], scale)
    torch.cuda.synchronize()
    assert_bitwise(snapshot({"attn": attn, "gate": gate, "scale": scale}), before, "the inputs after twelve launches")


def test_one_sixty_to_one_sixty_one_and_the_empty_call_are_typed_declines():
    """The row ceiling and the empty call decline by type -- never a launch, never another provider."""
    dev = _device_or_skip()
    k = entry.AttentionGateQuantDecode(dev, 2048)
    attn, gate = make_operands(MAX_ROWS + 1, 2048, dev, seed=25)
    scale = scale_tensor(0.01, dev)
    snapshot(run(k, attn[:MAX_ROWS], gate[:MAX_ROWS], scale))
    with pytest.raises(NotImplementedError, match="at most 160 rows.*got 161"):
        run(k, attn, gate, scale)
    with pytest.raises(ValueError, match="at least one row"):
        run(k, attn[:0], gate[:0], scale)


@pytest.mark.parametrize(
    "width, rows, scale_value",
    [pytest.param(2048, 3, 0.01, id="w2048-rows3"), pytest.param(6144, 19, 1.0, id="w6144-rows19")],
)
def test_golden(width, rows, scale_value):
    """Two small cases: three executions, the independent reference, the knob arms on the same operands (each writes the default's
    bytes, so the one golden holds for every served config), then the saved hashes.  Recorded into a NEW directory with
    ``CUDNN_DETERMINISM_RECORD`` only after the replay and reference cells pass; a change of these hashes on one runtime is a change of
    the rounding chain and needs a stated numerical reason."""
    dev = _device_or_skip()
    attn, gate = make_operands(rows, width, dev, seed=500 + rows, kind="adversarial")
    scale = scale_tensor(scale_value, dev)
    k = entry.AttentionGateQuantDecode(dev, width)
    actual = assert_replays(lambda: run(k, attn, gate, scale), repeats=3)
    assert_reference(actual["out"], attn, gate, scale, f"golden width {width} rows {rows}")
    for arm in (kcfg.Cfg(THREADS=128), kcfg.Cfg(PDL_RELEASE=True), kcfg.Cfg(THREADS=128, PDL_RELEASE=True)):
        other = snapshot(run(entry.AttentionGateQuantDecode(dev, width, cfg=arm), attn, gate, scale))
        assert_bitwise(other, actual, f"golden width {width} rows {rows}: {arm}")
    assert_golden(f"attention-gate-quant-decode-w{width}-rows{rows}", {"attn": attn, "gate": gate, "scale": scale}, actual)
