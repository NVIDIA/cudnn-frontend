# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The shared split combine's LEAN GROUP WALK (``sm100/split_combine._combine_lean_walk_kernel``): the module's former lean
kernel -- one warp per (q_row, head, batch), four splits loaded up front, then sixteen per memory round trip, the row accumulated
in split order from +0 after a full max pass -- restored as the module's fifth form (``DECODE_GRID_LEAN_WALK``, appended to the
selector) for the lean-grid-OVERFLOW launches of the cc 10.7 line, where the decode-grid kernel is never eligible and where,
measured on the Rubin performance class, the walk beats the module's wide lean group at split 2 (B 32 / B 128 x 24 heads: 3.3-3.4
/ 5.9 us against 5.9 / 12.4) and its serial walk at S_q 4 (4.0 / 8.9 against 5.3-5.5 / 11.7), while the small grids inside the
bound keep the module's forms (faster there).

What this module pins.  HOST: (1) the selector and the walk's group constants are APPENDED (the three existing values keep their
numbers; the walk's group of 16 is the bitwise window shared with the wide lean group and the decode-grid chain); (2) the rule
``heuristics.decode_combine_lean_walk_selected`` is the lean-grid-overflow rule on the cc 10.7 line only (``device_cc`` with
``decode_combine_grid_selected``'s semantics), partitions the grid axis with the decode-grid rule (one past its bound), is bounded to
the walk's up-front group (split <= 4: the measured regime; a deeper split keeps the module's own forms until a measurement widens
it), and names the six measured cells; (3) the adapter's plan-time pin (``prepared._split_combine_decode_grid``) answers
``DECODE_GRID_LEAN_WALK`` for the six measured cells on cc 10.7, keeps AUTO on the eight decode-grid cells, and never names the walk
off cc 10.7; (4)
``compile_ptr`` accepts the value, refuses the next one, hands it to the half / gated / ragged / sink entries as their trailing
form constexpr and drops it for the quantized and packed entries (their form is ``()``).  DEVICE (any SM100-line part; the kernel
has no arch-specific lowering): the walk's O / LSE BITWISE the module's serial walk at every split count (S_q >= 3) and its wide
lean group within 16 splits (S_q <= 2) -- the same weights and products added in split order from +0 -- against the float64
reference past 16 splits at S_q <= 2 (the decode form's other association) at the existing suite's tolerances; the kernel NAME the
profiler sees; the gated, ragged and programmatic-dependent-launch entries; the quantized and sink entries unchanged under the
walk's selector value.  GRAPH (cc 10.7): the Rubin d256 decode tile's split plan at the measured B 32 geometry launches the walk,
the within-bound cells their decode-grid / own forms as before."""

import math
from types import SimpleNamespace

import pytest

import torch

from cudnn.frost.compiled_cache import positional_entry

from frost_test_utils import requires_blackwell, requires_dsl, requires_rubin
from test_sdpa_fwd_decode_d256_sm107 import _run_graph, requires_sm107_dsl
from test_sdpa_split_combine_decode_grid import _DECODE_GRID_KERNEL, _check_reference, _combine_kernels_of, _launch, _own_form, _partials, _same

pytestmark = [pytest.mark.L0]
_gpu = [requires_dsl, requires_blackwell]

_WALK_KERNEL = "cudnn_kernel__combine_lean_walk_kernel"
# The six cells of the measured regression (B, H_q, S_q, split): the serving step at B 32 / B 128, 24 or 32 Q heads, S_q 1 and 4,
# whose lean combine grid (768 .. 3072 blocks) is past the decode-grid kernel's bound; and the eight decode-grid cells (the B 1 class
# and B 4 x 128K) whose plan keeps the decode-grid kernel.
_MEASURED_OVERFLOW_CELLS = [(32, 24, 1, 2), (128, 24, 1, 2), (32, 32, 1, 2), (32, 24, 4, 3), (128, 24, 4, 2), (32, 32, 4, 3)]
_DECODE_GRID_CELLS = [(1, 24, 1, 32), (1, 24, 1, 64), (4, 24, 1, 16), (1, 24, 4, 64), (1, 32, 4, 64), (1, 32, 1, 32), (1, 32, 1, 64), (1, 24, 1, 16)]
_OTHER_LINES = [(10, 0), (10, 3), (12, 0), (12, 1), (9, 0), (8, 0)]


def _walk(name):
    return name.startswith(_WALK_KERNEL)


def _bitwise_window(sq, splits):
    """Where the walk's arithmetic IS the module's own form's: the serial walk (S_q >= 3) at every split count, the wide lean
    group (S_q <= 2) up to its 16-slot group; past it the decode form associates the sum over the block's warps."""
    return sq >= 3 or splits <= 16


def _forms(comb, **kw):
    """The walk entry beside the module's own three-way rule (OFF) and the grid rule (AUTO) of one specialization."""
    return {
        name: positional_entry(comb.compile_ptr(**kw, decode_grid=code))
        for name, code in (("walk", comb.DECODE_GRID_LEAN_WALK), ("off", comb.DECODE_GRID_OFF), ("auto", comb.DECODE_GRID_AUTO))
    }


# --- (1) the constants ---------------------------------------------------------------------------------------------


@requires_dsl
def test_selector_and_group_constants_are_appended():
    from cudnn.sdpa.fwd import heuristics as heur
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    assert (comb.DECODE_GRID_AUTO, comb.DECODE_GRID_ON, comb.DECODE_GRID_OFF, comb.DECODE_GRID_LEAN_WALK) == (0, 1, 2, 3)
    assert comb.LEAN_WALK_G_S == 4, "the walk's up-front group (the former lean kernel's G_S)"
    assert comb.LEAN_WALK_G_T == comb.DECODE_SPLITS_PER_CHAIN == comb.G_S == 16, "the group per round trip is the forms' shared bitwise window"
    assert heur.DECODE_COMBINE_LEAN_WALK_MIN_LEAN_BLOCKS == heur.DECODE_COMBINE_MAX_LEAN_BLOCKS + 1 == 129, "one past the decode-grid bound: a partition"
    assert (
        heur.DECODE_COMBINE_LEAN_WALK_MAX_SPLIT == comb.LEAN_WALK_G_S == 4
    ), "the walk's split bound is its up-front group: the measured one-round-trip regime"
    assert comb._combine_lean_walk_kernel is not comb._combine_kernel


# --- (2) the rule ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_cc", [None, (10, 7), *_OTHER_LINES], ids=lambda cc: "default" if cc is None else f"cc{cc[0]}_{cc[1]}")
def test_lean_walk_rule_is_the_overflow_launches_on_the_cc_10_7_line_only(device_cc):
    """``decode_combine_lean_walk_selected``: True exactly where the lean grid is past the decode-grid kernel's bound AND the
    split is at most the walk's up-front group (4: the measured regime), on the cc 10.7 line (and the model's default line,
    ``device_cc=None``); False past that split on every line (the module's own forms, as before the restore) and False on every
    other device, where the module's own forms serve every launch (the Blackwell line measured on B200 / B300, and any line the
    rule was never run on)."""
    from cudnn.sdpa.fwd import heuristics as heur

    rule_line = device_cc is None or tuple(device_cc) == (10, 7)
    walk = lambda b, h, s_q, split: heur.decode_combine_lean_walk_selected(b, h, s_q, split, device_cc=device_cc)
    for b, h, s_q, split in _MEASURED_OVERFLOW_CELLS:
        assert heur.decode_combine_lean_blocks(b, h, s_q) > heur.DECODE_COMBINE_MAX_LEAN_BLOCKS
        assert walk(b, h, s_q, split) is rule_line, (b, h, s_q, split)
        assert not heur.decode_combine_grid_selected(b, h, s_q, split, device_cc=device_cc)
    # the smallest overflow grid (129 lean blocks) and a prefill grid: the walk up to split 4 on the rule line, never past it
    for grid in ((129, 1, 1), (43, 3, 1), (6, 24, 1), (8, 32, 1), (2, 4, 1024), (1, 8, 512)):
        assert heur.decode_combine_lean_blocks(*grid) >= heur.DECODE_COMBINE_LEAN_WALK_MIN_LEAN_BLOCKS
        assert all(walk(*grid, s) is rule_line for s in (2, 3, 4)), grid
        assert not any(walk(*grid, s) for s in (5, 8, 16, 32, 64, 128)), grid
    # inside the bound -- the eight decode-grid cells, the bound itself (B 4 x 32 heads = 128 blocks), the crossover grid -- never
    for b, h, s_q, split in _DECODE_GRID_CELLS:
        assert not walk(b, h, s_q, split), (b, h, s_q, split)
    for grid in ((4, 32, 1), (4, 24, 4), (1, 24, 1), (2, 32, 4), (128, 1, 1)):
        assert heur.decode_combine_lean_blocks(*grid) <= heur.DECODE_COMBINE_MAX_LEAN_BLOCKS
        assert not any(walk(*grid, s) for s in (2, 4, 8, 16, 32, 64, 128)), grid


def test_lean_walk_rule_partitions_the_grid_axis_with_the_decode_grid_rule():
    """At no (B, H_q, S_q, split) are both rules True; past the bound the decode-grid rule is False and the walk rule True up to
    the walk's split bound (cc 10.7) and False past it; inside the bound the walk rule is False.  So the plan-time pin has one
    answer per launch and the decode-grid kernel is never eligible where the walk is."""
    from cudnn.sdpa.fwd import heuristics as heur

    for b in (1, 2, 4, 5, 6, 8, 16, 32, 128):
        for h in (1, 8, 24, 32):
            for s_q in (1, 2, 3, 4, 8, 64):
                over = heur.decode_combine_lean_blocks(b, h, s_q) > heur.DECODE_COMBINE_MAX_LEAN_BLOCKS
                for split in (2, 3, 4, 8, 16, 32, 64, 128):
                    grid = heur.decode_combine_grid_selected(b, h, s_q, split)
                    walk = heur.decode_combine_lean_walk_selected(b, h, s_q, split)
                    assert not (grid and walk), (b, h, s_q, split)
                    assert walk is (over and split <= heur.DECODE_COMBINE_LEAN_WALK_MAX_SPLIT), (b, h, s_q, split)
                    if over:
                        assert not grid and not heur.decode_combine_force_lean(b, h, s_q, split)


# --- (3) the plan-time pin -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "rule_line"),
    [
        pytest.param(SimpleNamespace(_device_cc=(10, 7)), True, id="cc10_7"),
        pytest.param(SimpleNamespace(_device_cc=(10, 0)), False, id="cc10_0"),
        pytest.param(SimpleNamespace(_device_cc=(10, 3)), False, id="cc10_3"),
        pytest.param(SimpleNamespace(_device_cc=None, compute_capability=(12, 0)), False, id="cc12_0"),
        pytest.param(SimpleNamespace(_device_cc=(9, 0)), False, id="cc9_0"),
    ],
)
@requires_dsl
def test_prepared_form_pin_names_the_walk_for_the_measured_cells_on_cc_10_7_only(adapter, rule_line):
    """``prepared._split_combine_decode_grid`` -- the form the adapter compiles a split plan's combine entry with -- answers
    ``DECODE_GRID_LEAN_WALK`` for the six measured cells on a cc 10.7 plan (before the decode-grid / force-OFF logic, which never
    fires past the bound anyway), ``DECODE_GRID_AUTO`` for the eight decode-grid cells (unchanged), and off cc 10.7 what it answered
    before: AUTO past the bound (the module's own forms by construction), OFF inside it.  RED before the restore: the cc 10.7
    overflow cells read AUTO."""
    from cudnn.sdpa.fwd import prepared as prep
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    form = lambda b, h, s_q, split: prep._split_combine_decode_grid(adapter, b, h, s_q, split)
    for b, h, s_q, split in _MEASURED_OVERFLOW_CELLS:
        assert form(b, h, s_q, split) == (comb.DECODE_GRID_LEAN_WALK if rule_line else comb.DECODE_GRID_AUTO), (b, h, s_q, split)
    for b, h, s_q, split in _DECODE_GRID_CELLS:
        assert form(b, h, s_q, split) == (comb.DECODE_GRID_AUTO if rule_line else comb.DECODE_GRID_OFF), (b, h, s_q, split)
    # inside the bound below the decode-grid rule's split floor the pin is OFF on every line (unchanged)
    assert form(1, 24, 1, 4) == comb.DECODE_GRID_OFF and form(4, 24, 4, 16) == comb.DECODE_GRID_OFF
    # a prefill grid: the walk on cc 10.7, AUTO (= the module's own forms past the bound) elsewhere
    assert form(2, 4, 1024, 4) == (comb.DECODE_GRID_LEAN_WALK if rule_line else comb.DECODE_GRID_AUTO)
    # past the walk's split bound an overflow grid keeps AUTO (= the module's own forms there, as before the restore) on every line
    assert form(2, 4, 1024, 5) == comb.DECODE_GRID_AUTO and form(32, 24, 1, 16) == comb.DECODE_GRID_AUTO and form(8, 32, 1, 32) == comb.DECODE_GRID_AUTO


# --- (4) compile_ptr: the value, the entries that carry it, the entries that drop it --------------------------------


@requires_dsl
def test_compile_ptr_accepts_the_walk_value_and_refuses_the_next():
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    with pytest.raises(ValueError, match="DECODE_GRID_LEAN_WALK"):
        comb.compile_ptr(dtype_o="bf16", has_lse=True, decode_grid=comb.DECODE_GRID_LEAN_WALK + 1)
    with pytest.raises(ValueError, match="decode_grid"):
        comb.compile_ptr(dtype_o="bf16", has_lse=True, decode_grid=-1)


@requires_dsl
@pytest.mark.parametrize(
    ("kw", "carries"),
    [
        pytest.param(dict(dtype_o="bf16", has_lse=True), True, id="dense"),
        pytest.param(dict(dtype_o="bf16", has_lse=True, use_pdl=True), True, id="dense_pdl"),
        pytest.param(dict(dtype_o="bf16", has_lse=True, gate=True, dtype_gate="bf16"), True, id="gated"),
        pytest.param(dict(dtype_o="bf16", has_lse=True, ragged=True), True, id="ragged"),
        pytest.param(dict(dtype_o="bf16", has_lse=True, has_sink=True), True, id="sink"),
        pytest.param(dict(dtype_o="bf16", has_lse=True, has_sink=True, has_q_lens=True), True, id="sink_q_lens"),
        pytest.param(dict(dtype_o="e4m3", has_lse=True, quantized=True, has_amax=True, has_scale_o=True), False, id="quantized_fp8"),
        pytest.param(dict(dtype_o="bf16", has_lse=True, quantized=True, has_amax=True, has_scale_o_input=False), False, id="quantized_mxfp8_half"),
        pytest.param(dict(dtype_o="bf16", has_lse=True, packed=True), False, id="packed"),
    ],
)
def test_compile_ptr_hands_the_walk_value_to_the_half_entries_and_drops_it_for_the_quantized_and_packed_ones(monkeypatch, kw, carries):
    """The form selector is the trailing constexpr of the half / gated / ragged / sink entries' positional frame (``carries``) and
    absent from the quantized and packed entries' (their form is ``()``: develop's three-way rule under every value, so the walk's
    amax / scale arms are reached by a direct launch only).  Pinned without a GPU: the compile is stubbed and its positional
    arguments read (``compile_ptr``'s memo bypassed)."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    seen = []
    monkeypatch.setattr(comb, "_compile_cached", lambda entry, *args, **kwargs: seen.append((entry, args, kwargs)) or "stub")
    assert comb.compile_ptr.__wrapped__(**kw, decode_grid=comb.DECODE_GRID_LEAN_WALK) == "stub"
    entry, args, _ = seen[-1]
    assert (args[-1] == comb.DECODE_GRID_LEAN_WALK) is carries, (entry.__name__ if hasattr(entry, "__name__") else entry, args[-3:])
    if not carries:
        assert comb.DECODE_GRID_LEAN_WALK not in [a for a in args if isinstance(a, int) and not isinstance(a, bool)]


# --- (5) the numerics: bitwise the module's own forms inside their windows, the reference beyond -------------------


_SHAPES = [
    pytest.param((32, 24, 1, 256), id="overflow_b32_h24_sq1_d256", marks=_gpu),  # 768 lean blocks: the measured B 32 geometry
    pytest.param((8, 24, 4, 256), id="overflow_b8_h24_sq4_d256", marks=_gpu),  # 192 lean blocks, S_q 4: the serial-walk window
    pytest.param((8, 32, 1, 256), id="overflow_b8_h32_sq1_d256", marks=_gpu),  # 256 lean blocks
    pytest.param((2, 3, 5, 160), id="b2_h3_sq5_d160", marks=_gpu),  # the suite's odd shape: a 128-column chunk plus a 32-column tail
    pytest.param((1, 24, 1, 256), id="inside_bound_b1_h24_sq1_d256", marks=_gpu),  # the entry obeys its selector whatever the grid
    pytest.param((1, 2, 1, 8), id="b1_h2_sq1_d8", marks=_gpu),
]


@pytest.mark.parametrize("splits", [2, 3, 4, 5, 8, 16, 17, 20, 32, 33, 64, 65, 130], ids=lambda s: f"s{s}")
@pytest.mark.parametrize("shape", _SHAPES)
def test_lean_walk_matches_the_reference_and_is_bitwise_the_own_forms_inside_their_windows(shape, splits):
    """The walk entry against the float64 reference (O, LSE, dead splits, dead rows), two launches bitwise, and BITWISE the OFF
    entry's output wherever the OFF form adds the same weights and products in split order from +0: the serial walk at S_q >= 3
    (every split count: its two legs are one sequential chain) and the wide lean group at S_q <= 2 up to 16 splits.  Past 16
    splits at S_q <= 2 the OFF form is the decode form (warp-strided partial sums merged through SMEM): the reference bound only,
    at the suite's tolerances, nothing widened."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = shape
    fns = _forms(comb, dtype_o="bf16", has_lse=True, dtype_partial="f32")
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    outs = {}
    for name in ("walk", "off"):
        first = _launch(fns[name], op, lp, b, h, sq, d, splits)
        second = _launch(fns[name], op, lp, b, h, sq, d, splits)
        assert _same(first, second), f"{name}: two launches differ (a fixed-order reduction is deterministic)"
        _check_reference(*first, ref_o, ref_lse, sq)
        outs[name] = first
    if _bitwise_window(sq, splits):
        assert _same(outs["walk"], outs["off"]), f"S_q {sq}, {splits} splits: the walk must be bitwise the module's own form here"


@pytest.mark.parametrize("stats", ["none", "ln", "log2"])
@pytest.mark.parametrize("dtype", [pytest.param(torch.float16, id="f16", marks=_gpu), pytest.param(torch.bfloat16, id="bf16", marks=_gpu)])
def test_lean_walk_stats_bases_and_output_dtypes(dtype, stats):
    """Both half outputs, the final LSE in both bases and without Stats, at 8 and 32 splits on an S_q 4 overflow grid: bitwise the
    serial walk at both."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = 8, 24, 4, 256
    tag = "f16" if dtype == torch.float16 else "bf16"
    fns = _forms(comb, dtype_o=tag, has_lse=stats != "none", stats_log2=stats == "log2", dtype_partial="f32")
    for splits in (8, 32):
        op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
        outs = {name: _launch(fns[name], op, lp, b, h, sq, d, splits, dtype=dtype, stats=stats != "none") for name in ("walk", "off")}
        for name in outs:
            _check_reference(*outs[name], ref_o, ref_lse, sq, stats_log2=stats == "log2")
        assert _same(outs["walk"], outs["off"])


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_gpu)])
def test_lean_walk_strided_outputs_touch_only_their_elements(_m):
    """The final O / Stats keep the caller's strides (Rule S5) through the walk's packed and element store paths."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 2, 3, 5, 160, 20
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    head, token, batch = d * 2 + 8, h * (d * 2 + 8) + 16, sq * (h * (d * 2 + 8) + 16) + 32
    ostride, lstride = (batch, token, head, 2), (h * (sq * 2 + 3) + 5, sq * 2 + 3, 2)

    def storage(shape, strides, dtype):
        span = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
        store = torch.full((span + 32,), -31.0, device="cuda", dtype=torch.float32).to(dtype)
        view = store.as_strided(shape, strides, storage_offset=16)
        offsets = torch.zeros(shape, dtype=torch.int64)
        for axis, (n, s) in enumerate(zip(shape, strides)):
            dims = [1] * len(shape)
            dims[axis] = n
            offsets += torch.arange(n).view(dims) * s
        used = torch.zeros(store.numel(), dtype=torch.bool)
        used[offsets.flatten() + 16] = True
        return view, store, used

    fn = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True, decode_grid=comb.DECODE_GRID_LEAN_WALK))
    o, ostore, oused = storage((b, sq, h, d), ostride, torch.bfloat16)
    lse, lstore, lused = storage((b, h, sq), lstride, torch.float32)
    fn(op.data_ptr(), lp.data_ptr(), o.data_ptr(), lse.data_ptr(), (b, h, sq, d), splits, ostride, lstride, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    _check_reference(o, lse, ref_o, ref_lse, sq)
    assert torch.all(ostore.float().cpu()[~oused] == -31.0) and torch.all(lstore.cpu()[~lused] == -31.0)


# --- (6) the kernel name: what the profiler sees (the perf runners key the form by it) ------------------------------


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_gpu)])
def test_lean_walk_entry_launches_the_walk_kernel_and_falls_back_on_an_unaligned_head_dim(_m):
    """The LEAN_WALK entry runs ``_combine_lean_walk_kernel`` on an overflow grid AND on a small one (the entry obeys its
    selector; the grid rule is the adapter's), never the decode-grid kernel; a head dim off the 4-column lane (d 66) takes the
    module's own forms (the walk needs 16-byte rows); the AUTO entry on the overflow grid still runs the module's own forms."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    fns = _forms(comb, dtype_o="bf16", has_lse=True)

    def names(fn, b, h, sq, d, splits):
        op, lp, _, _ = _partials(b, h, sq, d, splits)
        _launch(fn, op, lp, b, h, sq, d, splits)  # warm (the first launch of a shape may carry module loads)
        return _combine_kernels_of(lambda: _launch(fn, op, lp, b, h, sq, d, splits))

    for shape in ((32, 24, 1, 256, 2), (1, 24, 1, 256, 16), (8, 24, 4, 256, 3)):
        got = names(fns["walk"], *shape)
        assert any(_walk(n) for n in got) and not any(_own_form(n) or n.startswith(_DECODE_GRID_KERNEL) for n in got), (shape, got)
    got = names(fns["walk"], 32, 24, 1, 66, 2)
    assert any(_own_form(n) for n in got) and not any(_walk(n) for n in got), got
    got = names(fns["auto"], 32, 24, 1, 256, 2)
    assert any(_own_form(n) for n in got) and not any(_walk(n) or n.startswith(_DECODE_GRID_KERNEL) for n in got), got


# --- (7) the gated, ragged and programmatic-dependent-launch entries ---------------------------------------------


@pytest.mark.parametrize("splits", [4, 16, 64], ids=lambda s: f"s{s}")
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16", marks=_gpu), pytest.param(torch.float16, id="f16", marks=_gpu)])
def test_lean_walk_gated_entry(dtype, splits):
    """The gate in the combine on the walk: ``O = merged_fp32 * sigmoid(G)`` rounded once, the dead row exactly 0 under a NaN gate
    (the SELECT after the fma), the LSE gate-free, and bitwise the serial walk's gated output (S_q 4)."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = 8, 24, 4, 256
    tag = "f16" if dtype == torch.float16 else "bf16"
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    gen = torch.Generator().manual_seed(7 + splits)
    gate = (torch.randn(b, sq, h, d, generator=gen) * 3.0).to(dtype)
    gate[:, 0] = torch.nan  # row 0 is the all-dead row: a NaN gate there must never reach the output
    gate = gate.cuda()
    ref = ref_o * torch.sigmoid(gate.float().cpu().double()).masked_fill(torch.isnan(gate.float().cpu()), 0.0)
    fns = _forms(comb, dtype_o=tag, has_lse=True, dtype_partial="f32", gate=True, dtype_gate=tag)
    outs = {}
    for name in ("walk", "off"):
        o, lse = _launch(fns[name], op, lp, b, h, sq, d, splits, dtype=dtype, extra=(gate.data_ptr(), gate.stride()))
        assert not torch.isnan(o.float()).any()
        torch.testing.assert_close(o.cpu().float(), ref.float(), atol=0.004, rtol=0.004)
        assert torch.count_nonzero(o[:, 0]).item() == 0, "the dead row is exactly 0 whatever G holds"
        live = torch.isfinite(ref_lse)
        torch.testing.assert_close(lse.cpu().double()[live], ref_lse[live], atol=2e-5, rtol=2e-5)
        outs[name] = (o, lse)
    assert _same(outs["walk"], outs["off"])


@pytest.mark.parametrize("splits", [4, 32], ids=lambda s: f"s{s}")
@pytest.mark.parametrize("i64", [pytest.param(False, id="i32", marks=_gpu), pytest.param(True, id="i64", marks=_gpu)])
def test_lean_walk_ragged_entry(i64, splits):
    """The ragged-Q decode leg's entry on the walk: three sequences of 2 / 0 / 1 query rows land on the packed O / Stats at their
    offsets, nothing past a sequence's length or the capacity, the values the reference's, bitwise the OFF form inside its window
    (S_q 2: the wide lean group up to 16 splits)."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = 3, 4, 2, 256
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    off_t = torch.int64 if i64 else torch.int32
    q_lens = [2, 0, 1]
    tok_off = [0, 2, 2, 3]
    rq = torch.tensor([t * h * d for t in tok_off], dtype=off_t, device="cuda")
    ro = rq.clone()
    rl = torch.tensor([t * h for t in tok_off], dtype=off_t, device="cuda")
    cap = tok_off[-1]
    fns = _forms(comb, dtype_o="bf16", has_lse=True, dtype_partial="f32", ragged=True, ragged_i64=i64)
    outs = {}
    for name in ("walk", "off"):
        o = torch.full((1, cap + 2, h, d), -31.0, device="cuda", dtype=torch.bfloat16)
        lse = torch.full((1, h, cap + 2), -31.0, device="cuda", dtype=torch.float32)
        fns[name](
            op.data_ptr(),
            lp.data_ptr(),
            o.data_ptr(),
            lse.data_ptr(),
            (b, h, sq, d),
            splits,
            o.stride(),
            lse.stride(),
            rq.data_ptr(),
            ro.data_ptr(),
            rl.data_ptr(),
            (h * d, h * d, h),
            (cap, cap),
            torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        got_o = o[0, :cap].cpu().float()
        got_l = lse[0, :, :cap].cpu().double()
        want_o = torch.cat([ref_o[bi, : q_lens[bi]] for bi in range(b)], dim=0).float()
        want_l = torch.cat([ref_lse[bi, :, : q_lens[bi]] for bi in range(b)], dim=1)
        torch.testing.assert_close(got_o, want_o, atol=0.004, rtol=0.004)
        live = torch.isfinite(want_l)
        torch.testing.assert_close(got_l[live], want_l[live], atol=2e-5, rtol=2e-5)
        assert torch.all(torch.isneginf(got_l[~live]))
        assert torch.all(o[0, cap:].float() == -31.0) and torch.all(lse[0, :, cap:] == -31.0), "nothing past the capacity"
        outs[name] = (o, lse)
    if _bitwise_window(sq, splits):
        assert _same(outs["walk"], outs["off"])


@pytest.mark.parametrize("splits", [2, 20], ids=lambda s: f"s{s}")
@pytest.mark.parametrize("_m", [pytest.param(None, marks=_gpu)])
def test_lean_walk_pdl_twin_is_bitwise_the_plain(_m, splits):
    """``use_pdl=True`` on the walk entry (the griddepcontrol pair before the first global read, the launch attribute): the same
    math, bitwise the plain walk entry, at the measured B 32 geometry."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = 32, 24, 1, 256
    plain = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True, decode_grid=comb.DECODE_GRID_LEAN_WALK))
    pdl = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True, decode_grid=comb.DECODE_GRID_LEAN_WALK, use_pdl=True))
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    a, c = _launch(plain, op, lp, b, h, sq, d, splits), _launch(pdl, op, lp, b, h, sq, d, splits)
    _check_reference(*a, ref_o, ref_lse, sq)
    assert _same(a, c)
    got = _combine_kernels_of(lambda: _launch(pdl, op, lp, b, h, sq, d, splits))
    assert any(_walk(n) for n in got), got


# --- (8) the entries the selector does not move ---------------------------------------------------------------------


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_gpu)])
def test_quantized_entries_keep_the_three_way_rule_under_the_walk_value(_m):
    """The quantized entry (``_host_ptr_quantized``: amax + output scale, the fp8 split's combine) names no form: under
    ``DECODE_GRID_LEAN_WALK`` it compiles the same kernels as without a value, runs the module's own form (the kernel name) and
    produces bitwise the same O / amax / LSE -- the walk's amax / scale arms are reachable by a direct launch only."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 32, 3, 1, 160, 3  # 96 lean blocks x 3 heads = inside the bound; the entry has no grid rule either way
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    kw = dict(dtype_o="e4m3", has_lse=True, has_amax=True, dtype_partial="f32", has_scale_o=True, stats_log2=True, quantized=True, has_scale_o_input=True)
    outs = {}
    for name, extra in (("walk", dict(decode_grid=comb.DECODE_GRID_LEAN_WALK)), ("default", {})):
        fn = positional_entry(comb.compile_ptr(**kw, **extra))
        o = torch.empty((b, sq, h, d), device="cuda", dtype=torch.float8_e4m3fn)
        lse = torch.full((b, h, sq), torch.nan, device="cuda")
        amax = torch.zeros(1, dtype=torch.float32, device="cuda")
        scale = torch.tensor([1.75], dtype=torch.float32, device="cuda")

        def run():
            fn(
                op.data_ptr(),
                lp.data_ptr(),
                o.data_ptr(),
                lse.data_ptr(),
                (b, h, sq, d),
                splits,
                o.stride(),
                lse.stride(),
                amax.data_ptr(),
                scale.data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
            torch.cuda.synchronize()

        run()
        amax.zero_()
        got = _combine_kernels_of(run)
        assert any(_own_form(n) for n in got) and not any(_walk(n) or n.startswith(_DECODE_GRID_KERNEL) for n in got), (name, got)
        torch.testing.assert_close(o.cpu().float(), (ref_o * 1.75).float().to(torch.float8_e4m3fn).float(), atol=0.03125, rtol=0.01)
        torch.testing.assert_close(amax.cpu(), ref_o.abs().amax().float().view(1), atol=2e-6, rtol=2e-6)
        live = torch.isfinite(ref_lse)
        torch.testing.assert_close(lse.cpu().double()[live], (ref_lse * math.log2(math.e))[live], atol=2e-5, rtol=2e-5)
        outs[name] = (o.view(torch.uint8).clone(), lse.clone(), amax.clone())
    assert _same(outs["walk"], outs["default"])


@pytest.mark.parametrize("_m", [pytest.param(None, marks=_gpu)])
def test_sink_entry_keeps_the_three_way_rule_under_the_walk_value(_m):
    """The sink entry folds the sink in the module's own forms under every selector value: ``DECODE_GRID_LEAN_WALK`` on it runs
    the same kernel as OFF (the name) and lands bitwise on the same O / LSE, the keyless row on ``O := 0 / LSE := sink``."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 32, 3, 5, 128, 3
    op, lp, _, _ = _partials(b, h, sq, d, splits)
    sinks = torch.tensor([-120.0, 0.7, 4.0], device="cuda")
    outs = {}
    for name, code in (("walk", comb.DECODE_GRID_LEAN_WALK), ("off", comb.DECODE_GRID_OFF)):
        fn = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True, has_sink=True, decode_grid=code))
        _launch(fn, op, lp, b, h, sq, d, splits, extra=(sinks.data_ptr(),))
        got = _combine_kernels_of(lambda: _launch(fn, op, lp, b, h, sq, d, splits, extra=(sinks.data_ptr(),)))
        assert any(_own_form(n) for n in got) and not any(_walk(n) or n.startswith(_DECODE_GRID_KERNEL) for n in got), (name, got)
        o, lse = _launch(fn, op, lp, b, h, sq, d, splits, extra=(sinks.data_ptr(),))
        assert torch.count_nonzero(o[:, 0]).item() == 0
        torch.testing.assert_close(lse[:, :, 0].cpu(), sinks.cpu().view(1, h).expand(b, h), atol=1e-6, rtol=0)
        outs[name] = (o, lse)
    assert _same(outs["walk"], outs["off"])


# --- (9) the graph path on cc 10.7: the measured geometry launches the walk ------------------------------------------


@pytest.mark.parametrize(
    ("B", "s_q", "kv", "split", "expect"),
    [
        pytest.param(32, 1, 4096, 2, "walk", id="b32_sq1_kv4k_split2_walk"),  # the measured cell: 768 lean blocks
        pytest.param(32, 4, 4096, 2, "walk", id="b32_sq4_kv4k_split2_walk"),  # the measured S_q 4 geometry (its kv shortened)
        pytest.param(6, 1, 4096, 2, "walk", id="b6_sq1_kv4k_split2_walk"),  # the smallest overflow grid at 24 heads: 144 blocks
        pytest.param(4, 1, 4096, 2, "own", id="b4_sq1_kv4k_split2_own_forms"),  # 96 blocks, split 2: the module's own forms, as before
        pytest.param(32, 1, 4096, 8, "own", id="b32_sq1_kv4k_split8_own_forms"),  # 768 blocks past the walk's split bound: the module's own forms, as before
        pytest.param(1, 1, 4096, 16, "decode_grid", id="b1_sq1_kv4k_split16_decode_grid"),  # the decode-grid cell, as before
    ],
)
@pytest.mark.parametrize("_m", [pytest.param(None, marks=[requires_dsl, requires_rubin, requires_sm107_dsl])])
def test_decode_graph_overflow_split_launches_the_lean_walk(B, s_q, kv, split, expect, monkeypatch, _m):
    """The Rubin d256 decode tile's split plan (24/2, page 16, HND pools, bf16, the split PINNED through the model's entry point)
    at the measured B 32 geometry launches ``_combine_lean_walk_kernel`` (the profiler's kernel names around the graph's execute);
    the within-bound cells keep their decode-grid / own forms.  The recombined O / Stats are checked against the suite's fp32
    reference by ``_run_graph`` itself."""
    from cudnn.sdpa.fwd import heuristics

    assert heuristics.decode_combine_lean_walk_selected(B, 24, s_q, split) is (expect == "walk")
    monkeypatch.setattr(heuristics, "choose_decode_tile_split_kv", lambda **kw: split)
    names = _combine_kernels_of(
        lambda: _run_graph(B=B, H=24, KH=2, s_q=s_q, lens=[kv] * B, page=16, hnd=True, dtype=torch.bfloat16, split_kv=split, causal_br=s_q > 1, seed=split)
    )
    ran = dict(walk=any(_walk(n) for n in names), decode_grid=any(n.startswith(_DECODE_GRID_KERNEL) for n in names), own=any(_own_form(n) for n in names))
    assert ran == {k: k == expect for k in ran}, (names, expect)
