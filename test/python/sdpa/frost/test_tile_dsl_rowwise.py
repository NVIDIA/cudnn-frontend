# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.frost.tile_dsl.rowwise`` -- the reduction of one value per thread over a thread block (``block_reduce``) -- and the
four one-instruction helpers of ``cudnn.frost.tile_dsl.pointwise`` (``div_full_f32``, ``rsqrt_approx_ftz_f32``,
``ex2_approx_f32``, ``min_xorsign_abs_f32``).

A block sum is an ORDER, a maximum an exact function, an approximate instruction a bound: each cell compares raw bits with a
reference that is named here, or a bound that is the instruction's own specification.

* host, no kernel: the trace-time refusals of the API; the slot arithmetic; the target test of the warp maximum over the DSL's
  architecture list; the fp32 replay of the sum trees, with three row classes that tell the two orders apart LEVEL BY LEVEL (so
  the device cell that compares with the replay is a proof of an order, not of a sum); the judge of a bound that is specified
  from the correctly rounded result.
* host rendering (a file probe in a fresh process; ``sm_100a``, and ``sm_107a`` / ``sm_120a`` where the compiler flavour serves
  them): every primitive alone -- the shuffle offsets in program order per level, one barrier, the slot stores and loads, the
  maximum's two arms by target, one line per one-instruction helper and none of the neighbour it is not.
* device (the SM100 line; ``tile_dsl_rowwise_probe.py``), bit-exact unless a bound is named:
  - the block sum against the replay of ITS tree in every lane, per order and warp count, with the wrong orders and a swapped
    ``(warp, lane)`` call shown NOT to reproduce it;
  - the block maximum against ``torch.amax(abs)``, and the maximum of SIGNED values against ``torch.amax`` at warp counts with
    and without padding lanes; the maximum with index against the lowest index of the maximum;
  - the shuffle arm of the two maxima against the single-instruction arm;
  - ``min_xorsign_abs_f32`` against its definition; the three approximate instructions inside the bound of their
    specification (``ex2.approx.f32`` from the correctly rounded result, as its specification states it), each beside the
    instruction it is not.
"""

import importlib
import os
import subprocess
import sys

import frost_routing
import pytest
import torch

from frost_test_utils import arch_known_to_the_dsl, nvdisasm_candidates, requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

_PROBE = "tile_dsl_rowwise_probe"
_PROBE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PROBE + ".py")
WARP = 32
ASCENDING, DESCENDING = (1, 2, 4, 8, 16), (16, 8, 4, 2, 1)
E4M3_MAX = 448.0
FP32_MIN_NORMAL = 2.0**-126
RSQRT_RELATIVE_BOUND = 2.0**-22.9  # the specification of rsqrt.approx.f32 over the positive finite range


def _probe():
    """The probe module, imported lazily (its DSL imports must not run at collection on a box without the DSL)."""
    return importlib.import_module(_PROBE)


def _rowwise():
    return importlib.import_module("cudnn.frost.tile_dsl.rowwise")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _all_bf16() -> torch.Tensor:
    """All 65,536 bf16 bit patterns widened exactly to fp32, in pattern order (the sign in the upper half)."""
    return (torch.arange(65536, dtype=torch.int32) << 16).view(torch.float32)


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _wide(gen, shape, *, signed=True) -> torch.Tensor:
    """fp32 values over 25 binades: the square of a normal draw times ``2^k``, ``k`` in -12 .. 12, with a random sign."""
    v = torch.randn(shape, generator=gen) ** 2 * torch.exp2(torch.randint(-12, 13, shape, generator=gen).float())
    if signed:
        v = torch.where(torch.rand(shape, generator=gen) < 0.5, v, -v)
    return v.to(torch.float32)


# ---------------------------------------------------------------------------
# The fp32 replay of the sum trees (the reference of the device sum cell)
# ---------------------------------------------------------------------------


def _butterfly(v: torch.Tensor, offsets) -> torch.Tensor:
    """``v = v + v[lane ^ offset]`` over the last (32-lane) axis for each offset in order: IEEE fp32 additions on the host."""
    lanes = torch.arange(WARP)
    for off in offsets:
        v = v + v[..., lanes ^ off]
    return v


def _same_in_every_lane(v: torch.Tensor) -> None:
    assert bool((_bits(v) == _bits(v[..., :1].expand_as(v))).all()), "a full butterfly leaves one value in every lane"


def replay_sum(x: torch.Tensor, level1: str, level2: str) -> torch.Tensor:
    """The block total of ``x`` (``[rows, warps, 32]`` fp32, on the host) under a tree named by its two levels.

    Level 1 (inside each warp): ``"ascending"`` = offsets 1, 2, 4, 8, 16; ``"descending"`` = 16, 8, 4, 2, 1.  Level 2 (over the
    warp sums): ``"ascending"`` = lanes below ``warps`` hold their warp's sum, the others ``+0``, offsets 1 .. 16;
    ``"descending"`` = every lane holds the sum of warp ``lane mod warps``, offsets ``warps / 2 .. 1`` (a power-of-two warp count);
    ``"descending_padded"`` = the padding of ``"ascending"`` with the offsets 16 .. 1 -- not an order of the primitive, the WRONG
    second level a cell holds against ``"ascending"`` at any warp count."""
    rows, warps, _ = x.shape
    sums = _butterfly(x, ASCENDING if level1 == "ascending" else DESCENDING)
    _same_in_every_lane(sums)
    sums = sums[..., 0]
    if level2 == "descending":
        assert warps & (warps - 1) == 0
        total = _butterfly(sums[:, torch.arange(WARP) & (warps - 1)], [warps >> (k + 1) for k in range(warps.bit_length() - 1)])
    else:
        padded = torch.zeros(rows, WARP, dtype=torch.float32)
        padded[:, :warps] = sums
        total = _butterfly(padded, ASCENDING if level2 == "ascending" else DESCENDING)
    _same_in_every_lane(total)
    return total[:, 0].contiguous()


def sum_rows(gen, warps: int, rows: int) -> dict:
    """The three row classes of the sum cells, ``[rows, warps, 32]`` each.  ``one_warp``: all mass in ONE warp, so the second
    level adds exact zeros and the first level alone decides the bits.  ``one_lane_per_warp``: ONE non-zero lane per warp, so the
    first level is exact and the second alone decides.  ``full``: every lane, 25 binades, both signs."""
    one_warp = torch.zeros(rows, warps, WARP, dtype=torch.float32)
    one_warp[torch.arange(rows), torch.randint(0, warps, (rows,), generator=gen)] = _wide(gen, (rows, WARP))
    one_lane = torch.zeros(rows, warps, WARP, dtype=torch.float32)
    lane = torch.randint(0, WARP, (rows, warps), generator=gen)
    one_lane.scatter_(2, lane.unsqueeze(-1), _wide(gen, (rows, warps, 1)))
    return {"one_warp": one_warp, "one_lane_per_warp": one_lane, "full": _wide(gen, (rows, warps, WARP))}


def wrong_trees(order: str, warps: int) -> dict:
    """``{row class: [(level1, level2), ...]}``: the trees each class must tell apart from ``(order, order)``.  The second level
    has ONE association below four warps (no slot, or a single addition), so no class can see it there."""
    other = "descending" if order == "ascending" else "ascending"
    other2 = "descending_padded" if order == "ascending" else "ascending"
    trees = {"one_warp": [(other, order)], "one_lane_per_warp": [], "full": [(other, order)]}
    if warps >= 4:
        trees["one_lane_per_warp"] = [(order, other2)]
        trees["full"] += [(order, other2), (other, other2)]
    return trees


SUM_CELLS = [("ascending", (1, 4, 8, 20, 24, 32))]
SUM_ROWS = 384


# ---------------------------------------------------------------------------
# Host cells
# ---------------------------------------------------------------------------

_REFUSED = [
    (dict(kind="total", order="ascending"), "kind must be one of"),
    (dict(kind="sum", order="up"), "a sum needs order"),
    (dict(kind="sum", order=None), "a sum needs order"),
    (dict(kind="max", order="ascending"), "exact in any order"),
    (dict(kind="max_index", order="ascending", index_limit=128), "exact in any order"),
    (dict(kind="max_index", order=None), "needs index_limit"),
    (dict(kind="max_index", order=None, index_limit=True), "needs index_limit"),
    (dict(kind="max_index", order=None, index_limit=128.0), "needs index_limit"),
    (dict(kind="max_index", order=None, index_limit=0), "needs index_limit"),
    (dict(kind="max_index", order=None, index_limit=2**31), "needs index_limit"),
    (dict(kind="sum", order="ascending", index_limit=128), "index_limit belongs to"),
    (dict(kind="max", order=None, index_limit=128), "index_limit belongs to"),
]
_REFUSED_WARPS = [0, 33, -4, True, 4.0, "4", None]


def test_api_refuses_a_missing_or_wrong_order():
    """``kind`` and ``order`` are keyword-only and have no default: a call without either is a ``TypeError`` before anything is
    traced.  Every other malformed call is a ``ValueError`` raised in plain Python before any operation is emitted (the operands
    here are ``None``: nothing reads them): an unknown kind or order, an order handed to a maximum, a warp count that is not an
    int in 1 .. 32, a missing or malformed ``index_limit``, an ``index_limit`` handed to another kind, a slot tensor of another
    type or statically too small."""
    rw = _rowwise()
    with pytest.raises(TypeError):
        rw.block_reduce(None, None, None, None, 4)
    with pytest.raises(TypeError):
        rw.block_reduce(None, None, None, None, 4, kind="sum")
    with pytest.raises(TypeError):
        rw.block_reduce(None, None, None, None, 4, order="ascending")
    with pytest.raises(TypeError):
        rw.block_reduce(None, None, None, None, 4, "sum", "ascending")
    for kwargs, text in _REFUSED:
        kwargs = dict(kwargs)
        warps = kwargs.pop("warps", 4)
        with pytest.raises(ValueError, match=text):
            rw.block_reduce(None, None, None, None, warps, **kwargs)
    for warps in _REFUSED_WARPS:
        with pytest.raises(ValueError, match="warps must be an int in 1 .. 32"):
            rw.block_reduce(None, None, None, None, warps, kind="sum", order="ascending")
    import cutlass

    with pytest.raises(ValueError, match="must be an fp32 shared-memory tensor"):
        rw._check_slots(cutlass.Int32, 8, "sum", 4)
    with pytest.raises(ValueError, match="needs 8 slot words, the tensor holds 4"):
        rw._check_slots(cutlass.Float32, 4, "max_index", 4)
    with pytest.raises(ValueError, match="needs 20 slot words, the tensor holds 16"):
        rw._check_slots(cutlass.Float32, 16, "sum", 20)
    rw._check_slots(cutlass.Float32, 8, "max_index", 4)
    rw._check_slots(cutlass.Float32, 32, "sum", 4)  # a larger array is the caller's choice


def test_slot_words():
    """One word per warp for a sum and a maximum, two for a maximum with its index; plain arithmetic with its own refusals."""
    rw = _rowwise()
    assert rw.WARP == WARP and rw.BUTTERFLY_STEPS == 5 and rw.KINDS == ("sum", "max", "max_index")
    assert set(order for order, _ in SUM_CELLS) == set(rw.SUM_ORDERS)
    for warps in (1, 4, 20, 32):
        assert rw.slot_words("sum", warps) == warps
        assert rw.slot_words("max", warps) == warps
        assert rw.slot_words("max_index", warps) == 2 * warps
    with pytest.raises(ValueError, match="kind must be one of"):
        rw.slot_words("amax", 4)
    for warps in _REFUSED_WARPS:
        with pytest.raises(ValueError, match="warps must be an int in 1 .. 32"):
            rw.slot_words("sum", warps)


def test_target_predicate():
    """The single-instruction fp32 warp maximum is taken for compute capability 10.x and for nothing else the installed DSL
    names: the 11.0 and 12.x targets do not assemble it, and an older or unlisted target takes the arm that assembles
    everywhere."""
    from cutlass.base_dsl.enums import Arch

    rw = _rowwise()
    names = {arch.name for arch in Arch}
    for name in ("sm_100a", "sm_103a", "sm_107a"):
        if name in names:
            assert rw._arch_has_redux_f32(Arch.from_string(name)), name
    for name in ("sm_110a", "sm_120a", "sm_121a"):
        if name in names:
            assert not rw._arch_has_redux_f32(Arch.from_string(name)), name
    below = [arch for arch in Arch if int(arch.major) < 10]
    assert below, "the installed DSL names no architecture below compute capability 10"
    newest_below = max(below, key=lambda arch: (int(arch.major), int(arch.minor)))
    assert not rw._arch_has_redux_f32(newest_below), newest_below
    assert {int(arch.major) for arch in Arch if rw._arch_has_redux_f32(arch)} == {10}
    assert "sm_100a" in names


@pytest.mark.parametrize("order,warp_counts", SUM_CELLS, ids=[order for order, _ in SUM_CELLS])
def test_sum_orders_are_observable(order, warp_counts):
    """What the device sum cell relies on, shown on the host replay alone: at every warp count the three row classes tell the
    order's tree apart from each wrong tree of ``wrong_trees`` -- ``one_warp`` through the first level alone (its second level
    adds exact zeros: swapping the SECOND level changes no row), ``one_lane_per_warp`` through the second alone (swapping the
    FIRST changes no row), ``full`` through both and through each mixed tree.  Plus the signed-zero edge of each order."""
    gen = torch.Generator(device="cpu").manual_seed(31)
    for warps in warp_counts:
        classes = sum_rows(gen, warps, SUM_ROWS)
        trees = wrong_trees(order, warps)
        for name, x in classes.items():
            want = replay_sum(x, order, order)
            for tree in trees[name]:
                differ = int((_bits(want) != _bits(replay_sum(x, *tree))).sum())
                assert differ > 0, f"{order} at {warps} warps: class {name} cannot tell the tree {tree} apart"
        other = "descending" if order == "ascending" else "ascending"
        if warps >= 4:
            blind = "descending_padded" if order == "ascending" else "ascending"
            assert bool((_bits(replay_sum(classes["one_warp"], order, order)) == _bits(replay_sum(classes["one_warp"], order, blind))).all())
        assert bool((_bits(replay_sum(classes["one_lane_per_warp"], order, order)) == _bits(replay_sum(classes["one_lane_per_warp"], other, order))).all())
    negative_zero = torch.full((1, 4, WARP), -0.0)
    assert int(_bits(replay_sum(negative_zero, "ascending", "ascending"))[0]) == 0  # the padding lanes add +0
    assert int(_bits(replay_sum(torch.full((1, 32, WARP), -0.0), "ascending", "ascending"))[0]) == -(1 << 31)  # no padding lane at 32 warps
    assert int(_bits(replay_sum(negative_zero, "descending", "descending"))[0]) == -(1 << 31)  # a plain sum


def _ulp_at_larger(got: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """The fp32 unit in the last place at the LARGER of the two magnitudes (fp64), never below the subnormal spacing.  EXACT: the
    binade comes from ``frexp`` and the power of two is built from its bits, so a result that sits exactly on a bound reads
    exactly the bound on every host (no ``log2`` / ``exp2`` of the math library in between)."""
    magnitude = torch.maximum(got.double().abs(), exact.abs()).clamp_min(FP32_MIN_NORMAL)
    _, exponent = torch.frexp(magnitude)  # magnitude = m * 2^exponent with m in [0.5, 1): the binade is exponent - 1
    return ((exponent.to(torch.int64) - 24 + 1023) << 52).view(torch.float64)  # 2^(exponent - 1 - 23) as a float64 bit pattern


def _units_from_rounded(got: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """The distance of the fp32 results ``got`` from the CORRECTLY ROUNDED fp32 value of ``exact`` (fp64), in fp32 units in the
    last place at the larger of the two magnitudes.  The judge of a bound that the specification states from the correctly
    rounded result: that result lies up to half a unit from the exact value, so the distance from the EXACT value may reach
    half a unit more than the bound."""
    rounded = exact.to(torch.float32).double()
    return (got.double() - rounded).abs() / _ulp_at_larger(got, rounded)


def test_bound_from_the_correctly_rounded_result_is_judged_from_it():
    """``_units_from_rounded``, the judge of the ``ex2.approx.f32`` cell, on constructed results (no kernel).  For 4,096 exponents
    whose ``2^x`` is at least three fp32 values away from a power of two:

    * a result TWO fp32 values away from the correctly rounded ``2^x``, on the side opposite to the exact value, reads exactly 2
      -- inside the specification ("2 units from the correctly rounded result") -- while its distance from the EXACT value is
      above 2 on every one of them (up to 2.5): a judge that measured from the exact value would refuse results the
      specification allows;
    * a result THREE fp32 values away reads 3 and is refused;
    * the host's fp64 -> fp32 conversion keeps a subnormal (the judge of the underflow range relies on it)."""
    gen = torch.Generator(device="cpu").manual_seed(39)
    x = torch.randint(-100, 100, (4096,), generator=gen).float() + torch.empty(4096).uniform_(0.2, 0.8, generator=gen)
    exact = torch.exp2(x.double())
    rounded = exact.to(torch.float32)
    assert bool(((rounded.double() - exact).abs() <= 0.5 * _ulp_at_larger(rounded, exact)).all())
    away = torch.where(exact >= rounded.double(), torch.tensor(0.0), torch.tensor(float("inf")))  # step to the side the exact value is NOT on
    two = torch.nextafter(torch.nextafter(rounded, away), away)
    three = torch.nextafter(two, away)
    assert bool((_units_from_rounded(rounded, exact) == 0).all())
    assert bool((_units_from_rounded(two, exact) == 2.0).all())
    assert bool((_units_from_rounded(three, exact) == 3.0).all())
    from_exact = (two.double() - exact).abs() / _ulp_at_larger(two, exact)
    assert bool((from_exact > 2.0).all()) and bool((from_exact <= 2.5).all()) and float(from_exact.max()) > 2.4
    assert float(torch.tensor([2.0**-140], dtype=torch.float64).to(torch.float32)[0]) == 2.0**-140


# ---------------------------------------------------------------------------
# Host rendering: every primitive alone
# ---------------------------------------------------------------------------

_UNSERVED = "Unsupported GPU architecture"  # the DSL's message when its compiler flavour has no such target
_ASC10 = "1,2,4,8,16,1,2,4,8,16"


def _sum_pin(offsets: str, guard: int) -> dict:
    count = offsets.count(",") + 1
    return {"shfl.sync.bfly": count, "add.f32": count, "bar.sync": 1, "st.shared": 1, "ld.shared": 1, "offsets": offsets, "load_guard": guard}


def _max_pin(redux: bool, index: bool) -> dict:
    pin = {"bar.sync": 1, "st.shared": 2 if index else 1, "ld.shared": 2 if index else 1, "load_guard": 2, "0fFF800000": 1}
    if redux:
        pin.update({"redux.sync.max.f32": 2, "offsets": "-"})
    else:
        pin.update({"shfl.sync.bfly": 10, "max.f32": 10, "offsets": _ASC10})
    if index:
        pin["redux.sync.min.s32"] = 2
    return pin


def expected_pins(arch: str) -> dict:
    """``{rendering tag: the non-zero counts}``; every count not named is zero.  ``redux`` = the target assembles the
    single-instruction fp32 warp maximum (compute capability 10.x)."""
    redux = arch in ("sm_100a", "sm_103a", "sm_107a")
    want = {}
    want.update(
        {
            "sum_ascending_w4": _sum_pin(_ASC10, 2),
            "sum_ascending_w20": _sum_pin(_ASC10, 2),
            "sum_ascending_w32": _sum_pin(_ASC10, 0),  # every lane is below 32 warps: the load needs no guard
            "max_w4": _max_pin(redux, False),
            "max_w4_shuffle": _max_pin(False, False),
            "max_index_w4": _max_pin(redux, True),
            "max_index_w4_shuffle": _max_pin(False, True),
            "div_full": {"div.full.f32": 1},
            "rsqrt": {"rsqrt.approx.ftz.f32": 1},
            "ex2": {"ex2.approx.f32": 1},
            "min_xorsign": {"min.xorsign.abs.f32": 1},
        }
    )
    return want


def _render(tmp_path, arch):
    dump = tmp_path / f"dump_{arch}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx,cubin")
    return subprocess.run([sys.executable, _PROBE_PATH, arch, str(dump), *nvdisasm_candidates()], capture_output=True, text=True, timeout=900, env=env)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_107a", "sm_120a"])
def test_rendering_of_every_primitive(tmp_path, arch):
    """Each primitive rendered alone for ``arch`` (no device of that class needed).  ASSERTED per rendering, from the PTX:

    * a sum: the shuffle offsets IN PROGRAM ORDER -- 1, 2, 4, 8, 16 twice for ``"ascending"`` -- one addition per shuffle, one
      barrier, one slot store, one slot load; the ``"ascending"`` load sits behind a lane guard below 32 warps;
    * a maximum: on a compute-capability-10.x target two single-instruction warp maxima and no shuffle; on any other target,
      and on EVERY target when the shuffle arm is forced, ten shuffles with ten ``max.f32`` and no single-instruction maximum;
      with an index two integer warp minima and two slot words; in every maximum the constant ``-inf`` (``0fFF800000``) once
      -- the identity the lanes past the warp count hold -- and in no other rendering;
    * each one-instruction helper: exactly one statement of its instruction and none of the instruction it is not;

    Every count that is not named is ZERO.  With a disassembler that decodes the target also: one barrier instruction per
    reduction, no spill, and the hardware function behind each approximate instruction.  A target the installed compiler
    flavour does not serve is SKIPPED; the disassembly half is skipped where no disassembler decodes the cubin."""
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch}")
    probe = _probe()
    proc = _render(tmp_path, arch)
    if proc.returncode != 0 and arch != "sm_100a" and _UNSERVED in (proc.stdout + proc.stderr):
        pytest.skip(f"the installed compiler flavour of the DSL does not serve {arch}")
    assert proc.returncode == 0, f"{arch} trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    tags = [tag for tag, _, _, _ in probe.RENDERS]
    for tag in tags:
        assert f"COMPILED {tag} {arch}" in lines, proc.stdout[-2000:]
    pin = {tag: {} for tag in tags}
    for ln in lines:
        if ln.startswith("PIN "):
            _, tag, key, value = ln.split()
            pin[tag][key] = value if key == "offsets" else int(value)
    want = expected_pins(arch)
    assert set(want) == set(tags), sorted(set(want) ^ set(tags))
    zero = {key: 0 for key in next(iter(pin.values()))}
    zero.update(offsets="-", load_guard=-1)
    for tag in tags:
        assert pin[tag] == dict(zero, **want[tag]), f"{arch} {tag}: the rendered PTX is not the documented instruction set: {pin[tag]}"
    if not nvdisasm_candidates():
        pytest.skip("compiled + PTX pinned; no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIPSASS") for ln in lines):
        pytest.skip(f"compiled + PTX pinned; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIPSASS', 'REJECT'))][:4]}")
    sass = {tag: {} for tag in tags}
    for ln in lines:
        if ln.startswith("SASS "):
            _, tag, key, count = ln.split()
            sass[tag][key] = int(count)
    print(f"\n{arch} SASS: {sass}")
    for tag in tags:
        assert sass[tag]["spill"] == 0, (tag, sass[tag])
        assert sass[tag]["bar_sync"] == want[tag].get("bar.sync", 0), (tag, sass[tag])
    assert sass["div_full"]["mufu_rcp"] == 1 and sass["rsqrt"]["mufu_rsq"] == 1 and sass["ex2"]["mufu_ex2"] == 1, sass
    assert sass["min_xorsign"]["fmnmx_xorsign"] == 1, sass["min_xorsign"]


# ---------------------------------------------------------------------------
# Device cells (the SM100 line): the reductions
# ---------------------------------------------------------------------------


def _first_bad_row(bad: torch.Tensor) -> str:
    row = int(bad.any(dim=1).nonzero()[0])
    return f"first row {row}, {int(bad[row].sum())} of {bad.shape[1]} lanes"


@requires_blackwell
@pytest.mark.parametrize("order,warp_counts", SUM_CELLS, ids=[order for order, _ in SUM_CELLS])
def test_block_sum_is_the_documented_tree(order, warp_counts):
    """``block_reduce(kind="sum", order=...)`` at every warp count of the cell, on the three row classes of ``sum_rows``:

    * the total EVERY thread of the block holds equals, bit for bit, the host replay of the order's own tree;
    * per class the replay of each wrong tree of ``wrong_trees`` differs on at least one row -- so a body that is right at one
      level and wrong at the other fails ``one_warp`` or ``one_lane_per_warp``, and a mixed tree fails ``full``;
    * a row of ``-0`` in every lane returns what the replay returns (``+0`` behind the padding lanes of ``"ascending"`` below 32
      warps, ``-0`` otherwise);
    * NEGATIVE CONTROL at four warps: the probe that hands the primitive ``(warp, lane)`` in place of ``(lane, warp)``
      reproduces the tree on NO row of the ``full`` class."""
    probe = _probe()
    dev = _device()
    gen = torch.Generator(device="cpu").manual_seed(32)
    for warps in warp_counts:
        classes = sum_rows(gen, warps, SUM_ROWS)
        classes["negative_zero"] = torch.full((2, warps, WARP), -0.0)
        trees = wrong_trees(order, warps)
        for name, x in classes.items():
            want = replay_sum(x, order, order)
            got = probe.run_sum(x.reshape(x.shape[0], warps * WARP).contiguous().to(dev), warps, order).cpu()
            bad = _bits(got) != _bits(want)[:, None]
            assert not bool(
                bad.any()
            ), f"{order} at {warps} warps, class {name}: {int(bad.any(dim=1).sum())} rows are not the documented tree ({_first_bad_row(bad)})"
            for tree in trees.get(name, []):
                differ = int((_bits(want) != _bits(replay_sum(x, *tree))).sum())
                assert differ > 0, f"{order} at {warps} warps: class {name} cannot tell the tree {tree} apart"
    full = sum_rows(gen, 4, SUM_ROWS)["full"]
    want = replay_sum(full, order, order)
    swapped = probe.run_sum(full.reshape(SUM_ROWS, 4 * WARP).contiguous().to(dev), 4, order, swapped=True).cpu()
    reproduced = int((_bits(swapped) == _bits(want)[:, None]).all(dim=1).sum())
    assert (
        reproduced == 0
    ), f"{order}: the swapped (warp, lane) call reproduced the tree on {reproduced} of {SUM_ROWS} rows; the cell cannot see the argument order"
    assert bool(torch.isfinite(swapped).all())


def max_rows(gen, warps: int) -> torch.Tensor:
    """Rows of ``32 * warps`` fp32 for the maximum cells: random rows over 25 binades with both signs; the position sweep (row
    ``r`` holds its largest magnitude, negative, at thread ``r``); a zero row, a row of ``-0``, mixed zeros; a row of fp32
    subnormals; rows with ``+inf`` and with ``-inf``; a row of equal values."""
    threads = warps * WARP
    sweep = torch.full((threads, threads), 0.25)
    sweep[torch.arange(threads), torch.arange(threads)] = -7.0
    zeros = torch.zeros(3, threads)
    zeros[1] = -0.0
    zeros[2, ::2] = -0.0
    subnormal = (torch.randperm(threads, generator=gen).to(torch.int32) + 1).view(torch.float32).reshape(1, threads)
    infinite = _wide(gen, (2, threads))
    infinite[0, threads // 3] = float("inf")
    infinite[1, threads - 1] = -float("inf")
    return torch.cat([_wide(gen, (256, threads)), sweep, zeros, subnormal, infinite, torch.full((1, threads), -1.5)]).contiguous()


@requires_blackwell
def test_block_max_is_the_row_maximum():
    """``block_reduce(kind="max", order=None)`` of ``|x|`` at 4, 24 and 32 warps: every thread of the block holds
    ``torch.amax(abs(x))`` of its row bit for bit -- finite values, signed zeros, fp32 subnormals, infinities, a zero row -- and
    the position sweep puts the maximum under every lane of every warp once."""
    probe = _probe()
    dev = _device()
    gen = torch.Generator(device="cpu").manual_seed(33)
    for warps in (4, 24, 32):
        x = max_rows(gen, warps)
        want = x.abs().amax(dim=1)
        got = probe.run_max(x.to(dev), warps).cpu()
        bad = _bits(got) != _bits(want)[:, None]
        assert not bool(bad.any()), f"{warps} warps: {int(bad.any(dim=1).sum())} rows are not the row maximum ({_first_bad_row(bad)})"


def signed_max_rows(gen, warps: int):
    """``(x, mixed_zero)``: rows of ``32 * warps`` SIGNED fp32 for the maximum without ``abs`` -- random rows of both signs; rows
    whose every value is NEGATIVE (normal values, and one row of negative subnormals); the position sweep with a negative
    maximum (row ``r`` holds ``-0.25`` at thread ``r`` among ``-3``); a row of ``-inf``; a row with one ``+inf``; a row of equal
    negative values; a row of ``-0``; and one row whose maximum is a zero held by zeros of both signs (``mixed_zero`` marks it:
    the sign of that zero is not compared)."""
    threads = warps * WARP
    negative = -_wide(gen, (64, threads), signed=False) - 1.0
    subnormal = -((torch.randperm(threads, generator=gen).to(torch.int32) + 1).view(torch.float32)).reshape(1, threads)
    sweep = torch.full((threads, threads), -3.0)
    sweep[torch.arange(threads), torch.arange(threads)] = -0.25
    low = torch.full((1, threads), -float("inf"))
    high = -_wide(gen, (1, threads), signed=False) - 1.0
    high[0, threads // 3] = float("inf")
    equal = torch.full((1, threads), -1.5)
    negative_zero = torch.full((1, threads), -0.0)
    mixed = -_wide(gen, (1, threads), signed=False) - 1.0
    mixed[0, 5::7] = -0.0
    mixed[0, 9::11] = 0.0
    x = torch.cat([_wide(gen, (128, threads)), negative, subnormal, sweep, low, high, equal, negative_zero, mixed]).contiguous()
    mark = torch.zeros(x.shape[0], dtype=torch.bool)
    mark[-1] = True
    return x, mark


@requires_blackwell
def test_block_max_of_signed_values_is_the_row_maximum_at_every_warp_count():
    """``block_reduce(kind="max", order=None)`` of SIGNED values (no ``abs``) at 4, 20 and 32 warps, through both arms of the warp
    maximum: every thread of the block holds ``torch.amax(x)`` of its row bit for bit.  The lanes past the warp count hold the
    identity of a maximum, ``-inf``, so a row whose every value is negative returns its (negative) maximum at 4 and at 20 warps
    exactly as it does at 32, where there is no padding lane -- an identity of ``+0`` would return ``+0`` for such a row below
    32 warps and pass at 32.  Rows: random signed values, all-negative rows (normal and subnormal), a negative maximum under
    every lane of every warp, ``-inf`` everywhere, one ``+inf``, a row of ``-0``.  A zero maximum held by zeros of both signs is
    compared as a number (its sign is no contract of either arm)."""
    probe = _probe()
    dev = _device()
    gen = torch.Generator(device="cpu").manual_seed(40)
    for warps in (4, 20, 32):
        x, mixed_zero = signed_max_rows(gen, warps)
        want = x.amax(dim=1)
        assert int((want < 0).sum()) > 64 + warps * WARP, "the cell needs rows whose maximum is negative"
        for arm, shuffle in (("the target's arm", False), ("the shuffle arm", True)):
            got = probe.run_max(x.to(dev), warps, shuffle=shuffle, signed=True).cpu()
            bad = (_bits(got) != _bits(want)[:, None]) & ~mixed_zero[:, None]
            assert not bool(
                bad.any()
            ), f"{warps} warps, {arm}: {int(bad.any(dim=1).sum())} rows of signed values are not the row maximum ({_first_bad_row(bad)})"
            assert bool((got[mixed_zero] == 0).all()), f"{warps} warps, {arm}: a zero maximum came back non-zero"


def max_index_rows(gen, warps: int):
    """``(x, zero_top)``: rows for the maximum-with-index cells -- random rows; rows drawn from four values (ties across lanes
    and across warps); the position sweep; a row of ``-inf``; a row of equal values; a row whose maximum is its LAST thread; and
    rows whose maximum is a zero held by zeros of both signs (``zero_top`` marks them: the sign of that zero is not compared)."""
    threads = warps * WARP
    ties = torch.tensor([-1.0, 0.5, 2.0, 2.0])[torch.randint(0, 4, (64, threads), generator=gen)]
    sweep = torch.full((threads, threads), -3.0)
    sweep[torch.arange(threads), torch.arange(threads)] = 5.0
    low = torch.full((1, threads), -float("inf"))
    equal = torch.full((1, threads), 0.75)
    last = -_wide(gen, (1, threads), signed=False) - 1.0
    last[0, threads - 1] = 1.0
    zero_top = -_wide(gen, (2, threads), signed=False) - 1.0
    zero_top[0, 5::7] = -0.0
    zero_top[0, 9::11] = 0.0
    zero_top[1, 3::5] = 0.0
    zero_top[1, 4::9] = -0.0
    x = torch.cat([_wide(gen, (128, threads)), ties, sweep, low, equal, last, zero_top]).contiguous()
    mark = torch.zeros(x.shape[0], dtype=torch.bool)
    mark[-2:] = True
    return x, mark


def max_index_twin(x: torch.Tensor, index: torch.Tensor, index_limit: int):
    """``(top, first)`` on the host: the row maximum and the lowest index among the positions that hold it."""
    top = x.amax(dim=1)
    first = torch.where(x == top[:, None], index.to(torch.int64), torch.tensor(index_limit, dtype=torch.int64)).amin(dim=1)
    return top, first.to(torch.int32)


def _check_max_index(tag, x, index, zero_top, got_top, got_first, index_limit):
    want_top, want_first = max_index_twin(x, index, index_limit)
    bad = got_first != want_first[:, None]
    assert not bool(bad.any()), f"{tag}: {int(bad.any(dim=1).sum())} rows do not return the lowest index of the maximum ({_first_bad_row(bad)})"
    assert int(got_first.max()) < index_limit, f"{tag}: the identity index won a row"
    bad = (_bits(got_top) != _bits(want_top)[:, None]) & ~zero_top[:, None]
    assert not bool(bad.any()), f"{tag}: {int(bad.any(dim=1).sum())} rows do not return the row maximum ({_first_bad_row(bad)})"
    assert bool((got_top[zero_top] == 0).all()), f"{tag}: a zero maximum came back non-zero"


@requires_blackwell
def test_block_max_index_is_the_lowest_index_of_the_maximum():
    """``block_reduce(kind="max_index", order=None, index_limit=...)`` at 4, 24 and 32 warps, in every thread of the block:

    * with the thread number as the index and with a random PERMUTATION of it per row: the row maximum bit for bit and the
      lowest index among the positions that hold it -- random rows, ties across lanes and across warps, the position sweep, a
      row of ``-inf`` (its lowest index), a maximum in the last thread; the identity ``index_limit`` never wins;
    * a maximum that is a zero held by zeros of both signs: the lowest index of either, the value compared as a number;
    * at 4 warps with ``index_limit = 2**31 - 1`` and indices whose 32 bits are fp32 NaN and infinity patterns: the index
      travels through the fp32 slot as its bits."""
    probe = _probe()
    dev = _device()
    gen = torch.Generator(device="cpu").manual_seed(34)
    for warps in (4, 24, 32):
        threads = warps * WARP
        x, zero_top = max_index_rows(gen, warps)
        natural = torch.arange(threads, dtype=torch.int32).repeat(x.shape[0], 1).contiguous()
        permuted = torch.stack([torch.randperm(threads, generator=gen) for _ in range(x.shape[0])]).to(torch.int32).contiguous()
        for name, index in (("thread index", natural), ("permuted index", permuted)):
            top, first = probe.run_max_index(x.to(dev), index.to(dev), warps, threads)
            _check_max_index(f"{warps} warps, {name}", x, index, zero_top, top.cpu(), first.cpu(), threads)
    x, zero_top = max_index_rows(gen, 4)
    limit = 2**31 - 1
    pattern = torch.stack([torch.randperm(4 * WARP, generator=gen) for _ in range(x.shape[0])]).to(torch.int32)
    pattern = (pattern + torch.where(torch.rand(x.shape[0], 1, generator=gen) < 0.5, 0x7F800000, 0x7FC00000).to(torch.int32)).contiguous()
    top, first = probe.run_max_index(x.to(dev), pattern.to(dev), 4, limit)
    _check_max_index("4 warps, NaN-pattern indices", x, pattern, zero_top, top.cpu(), first.cpu(), limit)


@requires_blackwell
def test_shuffle_arm_is_the_redux_arm():
    """The probe compiled twice per warp count, once with the primitive's target test answering False: ``max`` and
    ``max_index`` through five butterfly steps of ``max.f32`` equal the single-instruction arm bit for bit on the rows of the
    two cells above (the value of a zero maximum with zeros of both signs is compared as a number: its sign is no contract of
    either arm).  Device evidence for the arm every target outside compute capability 10.x takes."""
    probe = _probe()
    dev = _device()
    gen = torch.Generator(device="cpu").manual_seed(35)
    for warps in (4, 24, 32):
        threads = warps * WARP
        x = max_rows(gen, warps).to(dev)
        redux = probe.run_max(x, warps).cpu()
        shuffle = probe.run_max(x, warps, shuffle=True).cpu()
        bad = _bits(redux) != _bits(shuffle)
        assert not bool(bad.any()), f"max at {warps} warps: the shuffle arm differs from the single-instruction arm ({_first_bad_row(bad)})"
        xi, zero_top = max_index_rows(gen, warps)
        index = torch.stack([torch.randperm(threads, generator=gen) for _ in range(xi.shape[0])]).to(torch.int32).contiguous()
        top_r, first_r = (t.cpu() for t in probe.run_max_index(xi.to(dev), index.to(dev), warps, threads))
        top_s, first_s = (t.cpu() for t in probe.run_max_index(xi.to(dev), index.to(dev), warps, threads, shuffle=True))
        assert torch.equal(first_r, first_s), f"max_index at {warps} warps: the two arms return different indices"
        bad = (_bits(top_r) != _bits(top_s)) & ~zero_top[:, None]
        assert not bool(bad.any()), f"max_index at {warps} warps: the two arms return different maxima ({_first_bad_row(bad)})"
        assert bool((top_s[zero_top] == 0).all()) and bool((top_r[zero_top] == 0).all())
        _check_max_index(f"{warps} warps, shuffle arm", xi, index, zero_top, top_s, first_s, threads)


# ---------------------------------------------------------------------------
# Device cells: the four one-instruction helpers
# ---------------------------------------------------------------------------


def _pointwise(a: torch.Tensor, b: torch.Tensor) -> dict:
    """The probe's rows for two host fp32 tensors, LAUNCHED TWICE: the two launches must agree bit for bit (replay)."""
    probe = _probe()
    dev = _device()
    first = {k: v.cpu() for k, v in probe.run_pointwise(a.contiguous().to(dev), b.contiguous().to(dev)).items()}
    second = {k: v.cpu() for k, v in probe.run_pointwise(a.contiguous().to(dev), b.contiguous().to(dev)).items()}
    for name in probe.POINTWISE_ROWS:
        assert torch.equal(_bits(first[name]), _bits(second[name])), f"{name}: two launches on the same operands differ"
    return first


@requires_blackwell
def test_one_instruction_helpers():
    """The four helpers on every bf16 operand (and dense fp32 draws for the approximate ones), each launched twice with equal
    bits:

    * ``min_xorsign_abs_f32(a, +-448)``: bitwise its definition -- the magnitude ``min(|a|, 448)`` with the sign
      ``sign(a) XOR sign(bound)`` -- on every bf16 value that is not a NaN: zeros keep their sign, infinities become the bound;
    * ``div_full_f32``: within 2 units in the last place of the exact quotient (normal operands, normal quotient) for every
      bf16 numerator over a set of divisors, the reciprocal ``1 / b`` of every bf16 divisor, and random pairs; it is NOT the
      correctly rounded division (some quotient differs from ``div.rn.f32`` of the same operands);
    * ``rsqrt_approx_ftz_f32``: relative error at most ``2^-22.9`` on every positive normal operand; ``.ftz`` -- a positive
      subnormal operand gives ``+inf`` (the form without ``.ftz`` returns a finite value there) -- and ``+0 -> +inf``,
      ``-0 -> -inf``, ``+inf -> +0``, a negative normal operand a NaN;
    * ``ex2_approx_f32``: within 2 units in the last place of the CORRECTLY ROUNDED ``2^x`` where the result is normal -- the
      specification's own reference; the exact value lies up to half a unit from it, so the distance from the exact value is
      recorded and not judged; a zero gives exactly 1; overflow gives ``+inf``; NO flush -- where ``2^x`` is subnormal the
      result is a non-zero subnormal within 2 subnormal spacings of the correctly rounded one while the ``.ftz`` form returns
      zero."""
    gen = torch.Generator(device="cpu").manual_seed(36)
    bf16 = _all_bf16()
    not_nan = ~torch.isnan(bf16)
    finite = torch.isfinite(bf16)

    # min.xorsign.abs against its definition
    for bound in (E4M3_MAX, -E4M3_MAX):
        got = _pointwise(bf16, torch.full_like(bf16, bound))["min_xorsign"]
        sign = (_bits(bf16) ^ _bits(torch.tensor([bound], dtype=torch.float32))) & -(1 << 31)
        want = _bits(torch.minimum(bf16.abs(), torch.tensor(E4M3_MAX))) | sign
        bad = (_bits(got) != want) & not_nan
        assert not bool(
            bad.any()
        ), f"min_xorsign_abs_f32(a, {bound}): {int(bad.sum())} values differ from the definition (first a bits {hex(int(_bits(bf16)[bad][0]) & 0xFFFFFFFF)})"
        assert int(not_nan.sum()) == 65536 - 254

    # div.full: every bf16 numerator over a set of divisors, 1 / every bf16 divisor, random pairs
    divisors = torch.tensor([256.0, 5120.0, 6144.0, 448.0, 3.0, 7.0, 1.0e-3, 1.5e7, -448.0], dtype=torch.float32)
    drawn_a = torch.exp2(torch.empty(1 << 17).uniform_(-60.0, 60.0, generator=gen)).to(torch.float32)
    drawn_b = torch.exp2(torch.empty(1 << 17).uniform_(-60.0, 60.0, generator=gen)).to(torch.float32)
    drawn_a = torch.where(torch.rand(1 << 17, generator=gen) < 0.5, drawn_a, -drawn_a)
    a = torch.cat([bf16.repeat(divisors.numel()), torch.ones_like(bf16), drawn_a])
    b = torch.cat([divisors.repeat_interleave(bf16.numel()), bf16, drawn_b])
    rows = _pointwise(a, b)
    exact = a.double() / b.double()
    judged = torch.isfinite(a) & torch.isfinite(b) & (a.abs() >= FP32_MIN_NORMAL) & (b.abs() >= FP32_MIN_NORMAL)
    judged &= torch.isfinite(exact) & (exact.abs() >= FP32_MIN_NORMAL) & (exact.abs() < 2.0**127)
    assert int(judged.sum()) > 500000, int(judged.sum())
    units = (rows["div_full"].double() - exact).abs() / _ulp_at_larger(rows["div_full"], exact)
    bad = judged & ~(units <= 2.0)
    worst = float(units[judged].max())
    assert not bool(bad.any()), f"div_full_f32: {int(bad.sum())} quotients are more than 2 units from the exact quotient (worst {worst:.3f})"
    not_ieee = int((judged & (_bits(rows["div_full"]) != _bits(rows["div_rn"]))).sum())
    frost_routing.measured(
        "tile_dsl div_full_f32", f"{int(judged.sum())} judged quotients, worst {worst:.3f} units in the last place, {not_ieee} differ from div.rn.f32"
    )
    assert not_ieee > 0, "bitwise the correctly rounded quotient on every operand: not the approximate instruction this helper names"

    # rsqrt.approx.ftz and ex2.approx on every bf16 value and a dense draw
    dense_pos = torch.exp2(torch.empty(1 << 17).uniform_(-126.0, 127.0, generator=gen)).to(torch.float32)
    dense_exp = torch.empty(1 << 17).uniform_(-151.0, 129.0, generator=gen).to(torch.float32)
    x = torch.cat([bf16, dense_pos, dense_exp])
    rows = _pointwise(x, torch.ones_like(x))
    got, plain = rows["rsqrt_ftz"], rows["rsqrt_plain"]
    normal = torch.isfinite(x) & (x >= FP32_MIN_NORMAL)
    exact = x.double().rsqrt()
    relative = (got.double() - exact).abs() / exact
    bad = normal & ~(relative <= RSQRT_RELATIVE_BOUND)
    worst = float(relative[normal].max())
    assert int(normal.sum()) > 150000
    assert not bool(
        bad.any()
    ), f"rsqrt_approx_ftz_f32: {int(bad.sum())} results exceed the relative bound 2^-22.9 (worst 2^{torch.log2(torch.tensor(worst)).item():.2f})"
    same_as_plain = int((normal & (_bits(got) == _bits(plain))).sum())
    frost_routing.measured(
        "tile_dsl rsqrt_approx_ftz_f32",
        f"{int(normal.sum())} positive normal operands, worst relative error 2^{torch.log2(torch.tensor(worst)).item():.2f}, {same_as_plain} bitwise the form without .ftz",
    )
    assert same_as_plain == int(normal.sum()), "the .ftz form and the plain form differ on a normal operand"
    subnormal = (x > 0) & (x < FP32_MIN_NORMAL)
    assert int(subnormal.sum()) >= 127 and bool(torch.isposinf(got[subnormal]).all()), "a positive subnormal operand must flush to +0 and return +inf"
    assert bool(torch.isfinite(plain[subnormal]).all()) and bool(
        (plain[subnormal] > 1.0e18).all()
    ), "the form without .ftz returns a finite value on a subnormal operand"
    zero_bits = _bits(x)
    assert bool(torch.isposinf(got[zero_bits == 0]).all()) and bool(torch.isneginf(got[zero_bits == -(1 << 31)]).all())
    assert bool((_bits(got)[torch.isposinf(x)] == 0).all())
    negative = torch.isfinite(x) & (x <= -FP32_MIN_NORMAL)
    assert int(negative.sum()) > 30000 and bool(torch.isnan(got[negative]).all())

    got, flushed = rows["ex2"], rows["ex2_ftz"]
    exact = torch.exp2(x.double())
    in_range = torch.isfinite(x) & (x >= -126.0) & (x < 128.0)
    units = _units_from_rounded(got, exact)  # the specification: 2 units from the CORRECTLY ROUNDED result
    bad = in_range & ~(units <= 2.0)
    worst = float(units[in_range].max())
    worst_from_exact = float(((got.double() - exact).abs() / _ulp_at_larger(got, exact))[in_range].max())
    assert int(in_range.sum()) > 150000
    assert not bool(
        bad.any()
    ), f"ex2_approx_f32: {int(bad.sum())} results are more than 2 units from the correctly rounded 2^x (worst {worst:.3f}; {worst_from_exact:.3f} from the exact value)"
    assert bool((_bits(got)[x == 0] == _bits(torch.tensor([1.0]))[0]).all()) and int((x == 0).sum()) >= 2
    assert bool(torch.isposinf(got[torch.isfinite(x) & (x >= 128.0)]).all())
    assert bool((_bits(got)[torch.isneginf(x)] == 0).all()) and bool(torch.isposinf(got[torch.isposinf(x)]).all())
    underflow = torch.isfinite(x) & (x < -126.5) & (x > -148.5)
    spacing = 2.0**-149
    assert int(underflow.sum()) > 5000
    assert bool((got[underflow] > 0).all()) and bool((got[underflow] < FP32_MIN_NORMAL).all()), "ex2_approx_f32 flushed a subnormal result"
    assert bool(((got.double() - exact.to(torch.float32).double()).abs()[underflow] <= 2.0 * spacing).all())
    assert bool((flushed[underflow] == 0).all()), "the .ftz form is expected to return zero where 2^x is subnormal"
    same_as_flushed = int((in_range & (_bits(got) == _bits(flushed))).sum())
    frost_routing.measured(
        "tile_dsl ex2_approx_f32",
        f"{int(in_range.sum())} operands with a normal result, worst {worst:.3f} units in the last place from the correctly rounded 2^x ({worst_from_exact:.3f} from the exact value), {same_as_flushed} bitwise the .ftz form, {int(underflow.sum())} subnormal results not flushed",
    )
    assert bool((got[torch.isfinite(x) & (x < -151.0)] <= 2.0 * spacing).all()) and bool((got[torch.isfinite(x) & (x < -151.0)] >= 0).all())
