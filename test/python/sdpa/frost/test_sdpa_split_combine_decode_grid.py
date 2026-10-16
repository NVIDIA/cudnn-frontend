# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The shared split combine's DECODE-GRID form (``sm100/split_combine._combine_decode_grid_kernel``): one block per (row,
head, batch, 64-column chunk) with the split axis spread over the block's eight 16-lane chains and merged once, in a fixed
order, through SMEM -- against the module's own forms (``_combine_kernel``'s lean group and serial walk, and its decode form
``_combine_decode_kernel`` past 16 splits at S_q <= 2) it replaces on decode-shaped grids.  ``DECODE_GRID_OFF`` is the
module's own three-way rule: at <= 16 splits that is the lean group (``lean_wide``) or the serial walk, whose arithmetic
is the decode-grid kernel's own sequence, so the BITWISE pins below hold against it; past 16 splits at S_q <= 2 it is the
decode form, another fp32 association, so there the reference bound is the only claim (and what the cells assert).

What this module pins: (1) the SELECTION -- the host twin ``decode_grid_selected`` reads the lean kernel's grid against
``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS`` (a decode step's tens of blocks take the decode-grid kernel, a prefill's
thousands keep the module's own forms), the ``decode_grid`` keyword of ``compile_ptr`` is validated (its default is
``DECODE_GRID_OFF``: an entry that names no form is the module's own rule on every device, and only the cc 10.7 adapter
names AUTO), and a decode-shaped launch of the AUTO entry runs the decode-grid kernel while a prefill-shaped launch runs
the module's lean / serial kernel (the kernel NAMES, from the profiler); (2) the NUMERICS on decode cells (B 1 / B 4, 24 Q heads, S_q 1 / 4, d 256,
plus the suite's d 160 shape and the small head dims) at splits 2 .. 130: the float64 reference at the suite's
tolerances, dead splits and dead rows, the LSE in both bases, two launches bitwise (every reduction is in a fixed order),
and -- where the summation order is UNCHANGED, ``n_splits <= 16`` (only chain 0 holds splits) -- the decode-grid output
BITWISE the OFF form's on the same inputs; past 16 splits the sum is associated differently and only the reference bound
holds; (3) the gate-in-combine entry on the decode grid (the fp32 merged value times sigmoid(G) before the one rounding,
the dead-row SELECT under a NaN gate, bitwise the OFF form at <= 16 splits); (4) the ragged-Q placement (packed O / Stats
at the per-sequence offsets, a zero-length sequence, the capacity bound, int32 and int64 offsets).  Runs on any SM100-line
part (the kernel has no arch-specific lowering; the gate entry's packed f32x2 arithmetic is Blackwell-only like the fused
kernels'); the graph-level cells are the Rubin adapter's, which pins the form per plan (the Blackwell adapter pins OFF)."""

import math

import pytest

import torch

from cudnn.frost.compiled_cache import positional_entry

from frost_test_utils import requires_blackwell, requires_dsl, requires_rubin
from test_sdpa_fwd_decode_d256_sm107 import _run_graph, requires_sm107_dsl

# No module-level level mark: the direct cells are L0, the full graph-level split ladder below is L1 (its core is L0).
_gpu = [requires_dsl, requires_blackwell]
L0 = pytest.mark.L0
L1 = pytest.mark.L1


def _partials(b, h, sq, d, splits, seed=20261015):
    """Split-major partials with the suite's dead patterns: every split dead on row 0 (multi-row shapes), a dead split
    beside live ones, the LAST split dead on head 0 (the clamped tail slots re-read it); dead payloads are NaN."""
    generator = torch.Generator().manual_seed(seed + splits)
    o = torch.randn(splits, b, sq, h, d, generator=generator) * 0.4
    lse = torch.randn(splits, b, h, sq, generator=generator) * 2.0 + 0.7
    if sq > 1:
        lse[:, :, :, 0] = -torch.inf
    lse[1, :, :, min(1, sq - 1)] = -torch.inf
    lse[splits - 1, :, 0, :] = -torch.inf
    dead = torch.isneginf(lse).permute(0, 1, 3, 2).unsqueeze(-1)
    o.masked_fill_(dead, torch.nan)
    scores = lse.double()
    maximum = scores.amax(0)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = (scores - safe_maximum).exp()
    denominator = weights.sum(0)
    safe_o = o.double().masked_fill(dead, 0)
    ref_o = (weights.permute(0, 1, 3, 2).unsqueeze(-1) * safe_o).sum(0) / denominator.clamp_min(1).permute(0, 2, 1).unsqueeze(-1)
    ref_lse = safe_maximum + denominator.log()
    return o.reshape(splits * b, sq, h, d).cuda(), lse.reshape(splits * b, h, sq).cuda(), ref_o, ref_lse


def _entries(comb, **kw):
    """The three forms of one specialization: AUTO (the adapters' entry), ON and OFF (the forced kernels)."""
    return {
        name: positional_entry(comb.compile_ptr(**kw, decode_grid=code))
        for name, code in (("auto", comb.DECODE_GRID_AUTO), ("on", comb.DECODE_GRID_ON), ("off", comb.DECODE_GRID_OFF))
    }


def _launch(fn, op, lp, b, h, sq, d, splits, *, dtype=torch.bfloat16, stats=True, extra=()):
    o = torch.full((b, sq, h, d), -31.0, device="cuda", dtype=dtype)
    lse = torch.full((b, h, sq), -31.0, device="cuda", dtype=torch.float32) if stats else None
    fn(
        op.data_ptr(),
        lp.data_ptr(),
        o.data_ptr(),
        lse.data_ptr() if lse is not None else None,
        (b, h, sq, d),
        splits,
        o.stride(),
        lse.stride() if lse is not None else (h * sq, sq, 1),
        *extra,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return o, lse


def _check_reference(o, lse, ref_o, ref_lse, sq, *, stats_log2=False):
    torch.testing.assert_close(o.cpu().float(), ref_o.float(), atol=0.004, rtol=0.004)
    if sq > 1:
        assert torch.count_nonzero(o[:, 0]).item() == 0, "the all-dead row is exactly 0"
    if lse is not None:
        live = torch.isfinite(ref_lse)
        expect = ref_lse * (math.log2(math.e) if stats_log2 else 1.0)
        torch.testing.assert_close(lse.cpu().double()[live], expect[live], atol=2e-5, rtol=2e-5)
        assert torch.all(torch.isneginf(lse.cpu()[~live])), "a dead row's LSE is -inf"


def _same(a, b):
    return all((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y)) for x, y in zip(a, b))


# The kernel names the profiler reports: the decode-grid kernel, and the module's own forms (``_combine_kernel`` -- the lean
# group and the serial walk -- and its decode form ``_combine_decode_kernel``, a different prefix from the decode-grid kernel's).
_DECODE_GRID_KERNEL = "cudnn_kernel__combine_decode_grid_kernel"
_OWN_FORM_KERNELS = ("cudnn_kernel__combine_kernel", "cudnn_kernel__combine_decode_kernel")


def _own_form(name):
    return name.startswith(_OWN_FORM_KERNELS) and not name.startswith(_DECODE_GRID_KERNEL)


# --- (1) the selection ---------------------------------------------------------------------------------------------


@L0
@requires_dsl
def test_selection_rule_reads_the_lean_grid_against_the_heuristics_constant():
    """The host twin of the launch rule: the lean kernel's grid (``ceil(S_q / 4) x H_q x B`` blocks) at most the
    heuristics constant, and ``d_v % 4 == 0`` (the lean path's own geometry); the constant is the heuristics module's."""
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    assert comb._DECODE_COMBINE_MAX_LEAN_BLOCKS is heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS
    assert heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS == 128  # fewer blocks than SMs on every SM100-line part (148 .. 212)
    assert comb.lean_grid_blocks(1, 24, 1) == 24 and comb.lean_grid_blocks(4, 24, 4) == 96 and comb.lean_grid_blocks(2, 4, 1024) == 2048
    # the serving decode cells: B 1 and B 4 at 24 Q heads, S_q 1 and 4
    assert comb.decode_grid_selected(1, 24, 1, 256) and comb.decode_grid_selected(4, 24, 1, 256)
    assert comb.decode_grid_selected(1, 24, 4, 256) and comb.decode_grid_selected(4, 24, 4, 256)
    assert comb.decode_grid_selected(4, 32, 1, 256)  # exactly the bound: 128 blocks
    # a large-batch decode and every prefill keep the lean kernel by construction
    assert not comb.decode_grid_selected(8, 32, 1, 256)  # 256 blocks
    assert not comb.decode_grid_selected(32, 24, 1, 256)
    assert not comb.decode_grid_selected(2, 4, 1024, 256)
    assert not comb.decode_grid_selected(1, 8, 512, 128)
    # the lean path's geometry: a head dim off the 4-column lane takes the element path of the lean kernel
    assert not comb.decode_grid_selected(1, 2, 1, 5) and not comb.decode_grid_selected(1, 2, 1, 66)
    assert comb.decode_grid_selected(1, 2, 1, 8) and comb.decode_grid_selected(2, 3, 5, 160)


@L0
@requires_dsl
def test_decode_grid_geometry_constants():
    """The chain geometry is derived: 16 lanes x 4 columns = the 64-column chunk, eight chains per 128-thread block, the
    per-chain slot count is 16, the lean wide group's width (``G_S``; the former group walk's ``G_T``), so <= 16 splits sit in ONE chain: the bitwise window, and the
    SMEM exchange holds a max + a sum per chain plus four columns per lane."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    assert comb.DECODE_LANES_PER_CHAIN * 4 == comb.DECODE_CHUNK_COLS == 64
    assert comb.DECODE_CHAINS * comb.DECODE_LANES_PER_CHAIN == comb.DECODE_THREADS == 128
    assert comb.DECODE_SPLITS_PER_CHAIN == comb.G_S == 16
    assert comb.DECODE_SPLITS_PER_ROUND == 128
    assert comb.DECODE_RED_WORDS == 2 * comb.DECODE_CHAINS + comb.DECODE_CHAINS * comb.DECODE_LANES_PER_CHAIN * 4
    assert len({comb.DECODE_GRID_AUTO, comb.DECODE_GRID_ON, comb.DECODE_GRID_OFF}) == 3


@L0
@requires_dsl
def test_compile_ptr_validates_the_decode_grid_keyword():
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    with pytest.raises(ValueError, match="decode_grid"):
        comb.compile_ptr(dtype_o="bf16", has_lse=True, decode_grid=7)


@L0
@requires_dsl
def test_compile_ptr_names_no_form_by_default_and_declines_pdl_on_a_sink_entry():
    """An entry that names no ``decode_grid`` is the module's own three-way rule (``DECODE_GRID_OFF``) on every device: the
    decode-grid kernel is reached only where a caller names AUTO / ON -- the cc 10.7 adapter per plan, this module's cells --
    so the combine module's own pointer-level tests run the forms their ids name on every line.  And ``use_pdl=True`` on a
    sink entry is a typed decline (the sink entries carry no programmatic-dependent-launch form; the adapter refuses the
    same combination at check_support)."""
    import inspect

    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    assert inspect.signature(comb.compile_ptr).parameters["decode_grid"].default == comb.DECODE_GRID_OFF
    with pytest.raises(ValueError, match="sink entries"):
        comb.compile_ptr(dtype_o="bf16", has_lse=True, has_sink=True, use_pdl=True)


@L0
@pytest.mark.parametrize("_m", [pytest.param(None, marks=_gpu)])
def test_decode_shaped_launch_runs_the_decode_grid_kernel_and_a_prefill_shaped_one_the_lean_kernel(_m):
    """The AUTO entry (the cc 10.7 adapter's) picks per launch from the runtime geometry: the kernel names from the profiler."""
    from torch.profiler import ProfilerActivity, profile

    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    fn = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True, decode_grid=comb.DECODE_GRID_AUTO))

    def names(b, h, sq, d, splits):
        op, lp, _, _ = _partials(b, h, sq, d, splits)
        _launch(fn, op, lp, b, h, sq, d, splits)  # warm (the first launch of a shape may carry module loads)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _launch(fn, op, lp, b, h, sq, d, splits)
        return {e.name for e in prof.events() if e.device_type.name == "CUDA" and e.name.startswith("cudnn_kernel_")}

    decode = names(1, 24, 1, 256, 16)
    prefill = names(2, 4, 1024, 256, 4)
    assert any(n.startswith(_DECODE_GRID_KERNEL) for n in decode), decode
    assert not any(_own_form(n) for n in decode), decode
    assert any(_own_form(n) for n in prefill), prefill
    assert not any(n.startswith(_DECODE_GRID_KERNEL) for n in prefill), prefill


# --- (2) the numerics ----------------------------------------------------------------------------------------------


@L0
@pytest.mark.parametrize("splits", [2, 4, 5, 8, 16, 17, 20, 32, 33, 64, 65, 130], ids=lambda s: f"s{s}")
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 24, 1, 256), id="decode_b1_h24_sq1_d256", marks=_gpu),
        pytest.param((4, 24, 1, 256), id="decode_b4_h24_sq1_d256", marks=_gpu),
        pytest.param((1, 24, 4, 256), id="mtp_b1_h24_sq4_d256", marks=_gpu),
        pytest.param((4, 24, 4, 256), id="mtp_b4_h24_sq4_d256", marks=_gpu),
        pytest.param((2, 3, 5, 160), id="b2_h3_sq5_d160", marks=_gpu),
        pytest.param((1, 2, 1, 128), id="b1_h2_sq1_d128", marks=_gpu),
        pytest.param((1, 2, 1, 8), id="b1_h2_sq1_d8", marks=_gpu),
    ],
)
def test_decode_grid_matches_the_reference_and_is_bitwise_the_lean_kernel_up_to_sixteen_splits(shape, splits):
    """Every form against the float64 reference (O, LSE, dead rows), two launches bitwise, AUTO == ON on these decode-shaped
    grids, and ON == OFF bitwise for ``splits <= 16`` -- the one-chain window where the decode-grid kernel's arithmetic IS
    the lean kernel's sequence.  Past 16 splits the two associate the sum differently (the chains' partial sums merged with
    ``exp(m_c - m)`` factors): no bitwise claim, the reference bound only."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = shape
    assert comb.decode_grid_selected(b, h, sq, d)
    fns = _entries(comb, dtype_o="bf16", has_lse=True, dtype_partial="f32")
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    outs = {}
    for name, fn in fns.items():
        first = _launch(fn, op, lp, b, h, sq, d, splits)
        second = _launch(fn, op, lp, b, h, sq, d, splits)
        assert _same(first, second), f"{name}: two launches differ (a fixed-order reduction is deterministic)"
        _check_reference(*first, ref_o, ref_lse, sq)
        outs[name] = first
    assert _same(outs["auto"], outs["on"]), "a decode-shaped grid takes the decode-grid kernel under AUTO"
    if splits <= 16:
        assert _same(outs["on"], outs["off"]), f"splits {splits} <= 16: the decode-grid output must be bitwise the lean kernel's"


@L0
@pytest.mark.parametrize("stats", ["none", "ln", "log2"])
@pytest.mark.parametrize("dtype", [pytest.param(torch.float16, id="f16", marks=_gpu), pytest.param(torch.bfloat16, id="bf16", marks=_gpu)])
def test_decode_grid_stats_bases_and_output_dtypes(dtype, stats):
    """The final LSE in both bases (the partials stay natural; the merge needs them) and the two half outputs, with and
    without Stats, at 8 splits (bitwise the lean kernel) and 32 (the reference)."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = 1, 24, 4, 256
    tag = "f16" if dtype == torch.float16 else "bf16"
    fns = _entries(comb, dtype_o=tag, has_lse=stats != "none", stats_log2=stats == "log2", dtype_partial="f32")
    for splits in (8, 32):
        op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
        outs = {name: _launch(fn, op, lp, b, h, sq, d, splits, dtype=dtype, stats=stats != "none") for name, fn in fns.items()}
        for name in outs:
            _check_reference(*outs[name], ref_o, ref_lse, sq, stats_log2=stats == "log2")
        assert _same(outs["auto"], outs["on"])
        if splits <= 16:
            assert _same(outs["on"], outs["off"])


@L0
@pytest.mark.parametrize("_m", [pytest.param(None, marks=_gpu)])
def test_decode_grid_strided_outputs_touch_only_their_elements(_m):
    """The final O / Stats keep the caller's strides (Rule S5): padded storage around a strided output holds its canary
    after the decode-grid kernel stored the row (the element-store path: a row whose storage is not 8-byte aligned or whose
    D stride is not 1 takes it, the packed path otherwise)."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d, splits = 2, 3, 5, 160, 8
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

    fn = positional_entry(comb.compile_ptr(dtype_o="bf16", has_lse=True, decode_grid=comb.DECODE_GRID_ON))
    o, ostore, oused = storage((b, sq, h, d), ostride, torch.bfloat16)
    lse, lstore, lused = storage((b, h, sq), lstride, torch.float32)
    fn(op.data_ptr(), lp.data_ptr(), o.data_ptr(), lse.data_ptr(), (b, h, sq, d), splits, ostride, lstride, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    _check_reference(o, lse, ref_o, ref_lse, sq)
    assert torch.all(ostore.float().cpu()[~oused] == -31.0) and torch.all(lstore.cpu()[~lused] == -31.0)


# --- (3) the gate in the combine -----------------------------------------------------------------------------------


@L0
@pytest.mark.parametrize("splits", [4, 16, 64], ids=lambda s: f"s{s}")
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16", marks=_gpu), pytest.param(torch.float16, id="f16", marks=_gpu)])
def test_gate_in_combine_on_the_decode_grid(dtype, splits):
    """The gated entry on the decode grid: ``O = merged_fp32 * sigmoid(G)`` rounded once (the reference gates its float64
    merge; the approximate tanh and the single rounding sit inside the suite tolerance), the dead row exactly 0 under a NaN
    gate (the SELECT after the fma, never a multiply by zero), and bitwise the lean kernel's gated output at <= 16 splits."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = 1, 24, 4, 256
    tag = "f16" if dtype == torch.float16 else "bf16"
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    gen = torch.Generator().manual_seed(7 + splits)
    gate = (torch.randn(b, sq, h, d, generator=gen) * 3.0).to(dtype)
    gate[:, 0] = torch.nan  # row 0 is the all-dead row: a NaN gate there must never reach the output
    gate = gate.cuda()
    ref = ref_o * torch.sigmoid(gate.float().cpu().double()).masked_fill(torch.isnan(gate.float().cpu()), 0.0)
    fns = _entries(comb, dtype_o=tag, has_lse=True, dtype_partial="f32", gate=True, dtype_gate=tag)
    outs = {}
    for name, fn in fns.items():
        o, lse = _launch(fn, op, lp, b, h, sq, d, splits, dtype=dtype, extra=(gate.data_ptr(), gate.stride()))
        assert not torch.isnan(o.float()).any()
        torch.testing.assert_close(o.cpu().float(), ref.float(), atol=0.004, rtol=0.004)
        assert torch.count_nonzero(o[:, 0]).item() == 0, "the dead row is exactly 0 whatever G holds"
        live = torch.isfinite(ref_lse)
        torch.testing.assert_close(lse.cpu().double()[live], ref_lse[live], atol=2e-5, rtol=2e-5)  # the LSE never sees the gate
        outs[name] = (o, lse)
    assert _same(outs["auto"], outs["on"])
    if splits <= 16:
        assert _same(outs["on"], outs["off"])


# --- (4) the ragged-Q placement ------------------------------------------------------------------------------------


@L0
@pytest.mark.parametrize("splits", [4, 32], ids=lambda s: f"s{s}")
@pytest.mark.parametrize("i64", [pytest.param(False, id="i32", marks=_gpu), pytest.param(True, id="i64", marks=_gpu)])
def test_ragged_placement_on_the_decode_grid(i64, splits):
    """The ragged-Q decode leg's entry: three sequences of 2 / 0 / 1 query rows (``S_q`` max 2) land on the packed O / Stats
    at their offsets (tokens 0, 1 and 2; the zero-length sequence writes nothing), the rows past a sequence's length are not
    written, a canary past the capacity is untouched, the values match the reference, and ON == OFF bitwise at <= 16 splits."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    b, h, sq, d = 3, 4, 2, 256
    op, lp, ref_o, ref_lse = _partials(b, h, sq, d, splits)
    off_t = torch.int64 if i64 else torch.int32
    q_lens = [2, 0, 1]
    tok_off = [0, 2, 2, 3]  # exclusive prefix sums
    # Q / O offsets in ELEMENTS of a [T, H, D] packed buffer (div = H * D), Stats offsets in elements of [H, T] per
    # sequence (div = H: the head-major packed Stats), all int32 / int64.
    rq = torch.tensor([t * h * d for t in tok_off], dtype=off_t, device="cuda")
    ro = rq.clone()
    rl = torch.tensor([t * h for t in tok_off], dtype=off_t, device="cuda")
    cap = tok_off[-1]
    fns = _entries(comb, dtype_o="bf16", has_lse=True, dtype_partial="f32", ragged=True, ragged_i64=i64)
    outs = {}
    for name, fn in fns.items():
        o = torch.full((1, cap + 2, h, d), -31.0, device="cuda", dtype=torch.bfloat16)  # two canary tokens past the capacity
        lse = torch.full((1, h, cap + 2), -31.0, device="cuda", dtype=torch.float32)
        fn(
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
        got_o = o[0, :cap].cpu().float()  # [T, H, D]
        got_l = lse[0, :, :cap].cpu().double()  # [H, T]
        want_o = torch.cat([ref_o[bi, : q_lens[bi]] for bi in range(b)], dim=0).float()
        want_l = torch.cat([ref_lse[bi, :, : q_lens[bi]] for bi in range(b)], dim=1)
        torch.testing.assert_close(got_o, want_o, atol=0.004, rtol=0.004)
        live = torch.isfinite(want_l)
        torch.testing.assert_close(got_l[live], want_l[live], atol=2e-5, rtol=2e-5)
        assert torch.all(torch.isneginf(got_l[~live]))
        assert torch.all(o[0, cap:].float() == -31.0) and torch.all(lse[0, :, cap:] == -31.0), "nothing past the capacity"
        outs[name] = (o, lse)
    assert _same(outs["auto"], outs["on"])
    if splits <= 16:
        assert _same(outs["on"], outs["off"])


# --- (5) the graph path: the serving decode cells at every split of the route's ladder ---------------------------


def _graph_ladder_cells(full):
    """(B, S_q, kv, split) over B {1, 4} x S_q {1, 4} x kv {4K, 32K, 128K} x splits {4 .. 64}; a split past the candidate
    list's bound (``kv_tiles // 2``: 16 at kv 4K) is not a plan of the graph and is left out.  ``full`` = the whole matrix
    (L1); otherwise the L0 core: the three B 1 x S_q 1 cells at their route's split, B 4 x kv 128K at 16, an MTP cell."""
    if not full:
        return [(1, 1, 4096, 16), (1, 1, 32768, 32), (1, 1, 131072, 64), (4, 1, 131072, 16), (1, 4, 32768, 16)]
    cells = []
    for B in (1, 4):
        for s_q in (1, 4):
            for kv in (4096, 32768, 131072):
                for split in (4, 8, 16, 32, 64):
                    if split <= (kv // 128) // 2:
                        cells.append((B, s_q, kv, split))
    return cells


def _graph_ladder_id(c):
    return f"b{c[0]}_sq{c[1]}_kv{c[2] // 1024}k_split{c[3]}"


def _run_graph_ladder_cell(B, s_q, kv, split, monkeypatch):
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb

    assert comb.decode_grid_selected(B, 24, s_q, 256), "every cell of the ladder is inside the combine module's decode-grid bound"
    # The kernel the graph's launch takes is the split-aware rule's (heuristics.decode_combine_grid_selected, which the adapter
    # pins the compiled entry to): the decode-grid kernel on the B 1 cells and on B 4 x S_q 1 from split 8, the lean kernel on
    # B 4 x S_q 4 below split 32 and at split 4 everywhere -- both forms are equivalent to the reference and to each other.
    assert split <= (kv // 128) // 2, "a split the candidate list offers (kv_tiles // 2)"
    # The ranked list of a decode graph holds the model's choice and its runner-up only, so the ladder's split is PINNED through the
    # model's own entry point: the decode split chooser answers ``split`` for this graph (the lever a serving stack pins through the
    # block's ``decode_split_kv``), and ``_run_graph`` selects the plan that carries it.
    monkeypatch.setattr(heuristics, "choose_decode_tile_split_kv", lambda **kw: split)
    # The serving form: 24/2, page 16, HND pools, bf16, Stats on; the multi-token step carries the bottom-right diagonal.
    plan = _run_graph(B=B, H=24, KH=2, s_q=s_q, lens=[kv] * B, page=16, hnd=True, dtype=torch.bfloat16, split_kv=split, causal_br=s_q > 1, seed=split)
    assert plan.knobs.split_kv == split, plan.knobs


def _ladder_params(full):
    return [pytest.param(c, id=_graph_ladder_id(c), marks=[requires_dsl, requires_rubin, requires_sm107_dsl]) for c in _graph_ladder_cells(full)]


@L0
@pytest.mark.parametrize("cell", _ladder_params(False))
def test_decode_graph_pinned_split_ladder_core(cell, monkeypatch):
    """The Rubin d256 decode tile + the decode-grid combine on the serving cells, the split PINNED (the route's own choice per
    cell): the recombined O / Stats against the fp32 reference at the decode suite's tolerances."""
    _run_graph_ladder_cell(*cell, monkeypatch)


@L1
@pytest.mark.parametrize("cell", _ladder_params(True))
def test_decode_graph_pinned_split_ladder_full(cell, monkeypatch):
    """The whole ladder: splits 4 / 8 / 16 / 32 / 64 (where the graph offers them) on B 1 and B 4, S_q 1 and 4, kv 4K / 32K /
    128K -- every combine launch inside the module's decode-grid bound, the kernel per cell the split-aware rule's."""
    _run_graph_ladder_cell(*cell, monkeypatch)


def _combine_kernels_of(fn):
    """The combine kernel names (``cudnn_kernel__combine*``) the CUDA profiler sees while ``fn`` runs."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
    return {e.name for e in prof.events() if e.device_type.name == "CUDA" and e.name.startswith("cudnn_kernel__combine")}


@L0
@pytest.mark.parametrize(
    ("B", "s_q", "split", "decode_grid"),
    [
        pytest.param(1, 1, 16, True, id="b1_sq1_split16_decode_grid"),
        pytest.param(1, 1, 4, False, id="b1_sq1_split4_lean"),
        pytest.param(4, 1, 16, True, id="b4_sq1_split16_decode_grid"),
        pytest.param(4, 4, 16, False, id="b4_sq4_split16_lean"),
        pytest.param(4, 4, 32, True, id="b4_sq4_split32_decode_grid"),
    ],
)
@pytest.mark.parametrize("_m", [pytest.param(None, marks=[requires_dsl, requires_rubin, requires_sm107_dsl])])
def test_decode_graph_combine_kernel_follows_the_split_aware_rule(B, s_q, split, decode_grid, monkeypatch, _m):
    """The adapter pins the combine entry's kernel FORM to the split-aware rule (``heuristics.decode_combine_grid_selected``):
    a graph-level launch at the B 4 x S_q 4 crossover takes the lean kernel at split 16 and the decode-grid kernel at 32,
    a small grid the decode-grid kernel from split 8 and the lean kernel at 4 -- read off the CUDA profiler's kernel names
    around the graph's execute (the kernel module's own entry would take the decode-grid kernel on every one of these)."""
    from cudnn.sdpa.fwd import heuristics

    assert heuristics.decode_combine_grid_selected(B, 24, s_q, split) is decode_grid
    monkeypatch.setattr(heuristics, "choose_decode_tile_split_kv", lambda **kw: split)
    names = _combine_kernels_of(
        lambda: _run_graph(B=B, H=24, KH=2, s_q=s_q, lens=[8192] * B, page=16, hnd=True, dtype=torch.bfloat16, split_kv=split, causal_br=s_q > 1, seed=split)
    )
    ran_decode = any(n.startswith(_DECODE_GRID_KERNEL) for n in names)
    ran_lean = any(_own_form(n) for n in names)
    assert (ran_decode, ran_lean) == (decode_grid, not decode_grid), (names, decode_grid)
