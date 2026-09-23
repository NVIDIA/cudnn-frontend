# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The "bits" per-cell mask form (`tile_dsl.mask.apply_mask_chunk_bits`) masks EXACTLY the cells
`apply_mask_chunk` masks -- pinned on a host emulation of both formulas.

`apply_mask_chunk` compares every cell against every active bound (padded: `kv >= seq_kv_len`;
causal: `kv > q_caus_lim`; SWA: `kv < q_minus_w`) and ORs the terms.  `apply_mask_chunk_bits`
maps the same terms onto one band `[lo, hi)` per lane, builds a 32-column KEEP word per word
with two saturating shifts (PTX `shr.u32` / `shl.b32` clamp a shift amount >= 32 to a zero
result) and selects on one bit per cell.  The two must agree on every cell for every flag
combination, every sentinel and every payload -- including the cases a random sweep rarely
hits: a band edge exactly on a word boundary, an edge left / right of the whole chunk (the
saturated shift), a fully-masked and a fully-unmasked row, NaN / +-inf / +-0 / the sentinel
itself as payload.  Bitwise (`view(int32)`) so a NaN payload is compared as a bit pattern.

Device-independent: the emulation below mirrors `tile_dsl/mask.py` line by line in torch int64
arithmetic and wraps the intermediates the device computes in Int32 (`lo - kv_col_base`, the
per-word shift count), so the reason for `MASK_BOUND_LIMIT` is visible here too.  The GPU half --
the real kernels, `MASK_FORM` cells vs bits -- was pinned by dumping O / LSE from develop and
from the branch on a Rubin GPU (82 kernel x mask x shape cases bitwise, 2026-09-22); the
lowering itself is pinned by `test_sm107_masked_softmax_sass_is_register_to_predicate`.
"""

import itertools

import pytest
import torch

from cudnn.frost.tile_dsl.constants import MASK_CAUSAL, MASK_PADDED, MASK_SWA
from cudnn.frost.tile_dsl.mask import _NEG_INF_BITS, MASK_BOUND_LIMIT, MASK_FORM_BITS, MASK_FORM_CELLS, MASK_FORMS, MASK_WORD_COLS, apply_mask_chunk_bits

pytestmark = [pytest.mark.L0]

_ALL_ONES = (1 << 32) - 1
_SENTINELS = [_NEG_INF_BITS, float("-inf")]


def _i32(x):
    """Two's-complement wrap to Int32: the width the device computes the band bounds and shift counts in."""
    return ((x + (1 << 31)) & _ALL_ONES) - (1 << 31)


def _f32(value):
    """`cutlass.Float32(value)` on the host: round-to-nearest fp32.  The legacy sentinel -3.4028235e38 sits a hair
    past -FLT_MAX in decimal, which the DSL rounds to -FLT_MAX (the SASS immediate reads -3.40282346638528859812e+38)
    and torch's Python-float conversion refuses; go through fp64 so both agree."""
    return torch.tensor(value, dtype=torch.float64).to(torch.float32)


def _fill(S, value):
    return _f32(value).expand_as(S)


def _band(q_abs, seq_kv_len, window_left, mask_flags, bottom_right, causal_diag, window_right):
    """`apply_mask_chunk_bits`'s flag -> band mapping, on int64 tensors.  Returns (lo, hi), either None."""
    lo = hi = None
    diag = causal_diag if bottom_right else torch.zeros_like(q_abs)
    if mask_flags & MASK_SWA:
        lo = _i32(q_abs + diag - window_left)
    if mask_flags & MASK_CAUSAL:
        hi = _i32(q_abs + diag + window_right + 1)
    if mask_flags & MASK_PADDED:
        hi = seq_kv_len.clone() if hi is None else torch.minimum(hi, seq_kv_len)
    return lo, hi


def _shift_sat(x, n, left):
    """PTX `shl.b32` / `shr.u32` on a 32-bit value: shift amounts >= 32 yield 0; `n` is already clamped at 0."""
    n = n.clamp(min=0)
    y = torch.where(left, (x << n.clamp(max=32)) & _ALL_ONES, x >> n.clamp(max=32))
    return torch.where(n >= 32, torch.zeros_like(y), y)


def _keep_words(lo, hi, kv_col_base, n_cols):
    """`band_mask_words`: one keep word per 32 columns, bit i of word s <=> column 32 s + i kept."""
    n_words = (n_cols + MASK_WORD_COLS - 1) // MASK_WORD_COLS
    ones = torch.full_like(kv_col_base, _ALL_ONES)
    words = []
    for s in range(n_words):
        word = None
        if hi is not None:
            n_masked = _i32(MASK_WORD_COLS * (s + 1) - _i32(hi - kv_col_base))
            word = _shift_sat(ones, n_masked, left=torch.zeros_like(n_masked, dtype=torch.bool))
        if lo is not None:
            n_masked = _i32(_i32(lo - kv_col_base) - MASK_WORD_COLS * s)
            above = _shift_sat(ones, n_masked, left=torch.ones_like(n_masked, dtype=torch.bool))
            word = above if word is None else (word & above)
        words.append(word)
    return words


def mask_bits(S, q_abs, kv_col_base, seq_kv_len, window_left, mask_flags, bottom_right, causal_diag, mask_value, window_right):
    """`apply_mask_chunk_bits`, emulated: S [rows, n_cols] fp32, bounds [rows] int64."""
    n_cols = S.shape[1]
    lo, hi = _band(q_abs, seq_kv_len, window_left, mask_flags, bottom_right, causal_diag, window_right)
    words = _keep_words(lo, hi, kv_col_base, n_cols)
    keep = torch.zeros(S.shape, dtype=torch.bool)
    for s, word in enumerate(words):
        for i in range(MASK_WORD_COLS):
            c = MASK_WORD_COLS * s + i
            if c >= n_cols:
                break
            keep[:, c] = ((word >> i) & 1) != 0
    return torch.where(keep, S, _fill(S, mask_value))


def mask_cells(S, q_abs, kv_col_base, seq_kv_len, window_left, mask_flags, bottom_right, causal_diag, mask_value, window_right):
    """`apply_mask_chunk`, emulated: the per-cell reference."""
    n_cols = S.shape[1]
    kv = kv_col_base[:, None] + torch.arange(n_cols, dtype=torch.int64)[None, :]
    diag = causal_diag if bottom_right else torch.zeros_like(q_abs)
    masked = torch.zeros(S.shape, dtype=torch.bool)
    if mask_flags & MASK_PADDED:
        masked |= kv >= seq_kv_len[:, None]
    if mask_flags & MASK_CAUSAL:
        q_caus_lim = _i32(q_abs + diag + window_right)
        masked |= kv > q_caus_lim[:, None]
    if mask_flags & MASK_SWA:
        q_minus_w = _i32(q_abs + diag - window_left)
        masked |= kv < q_minus_w[:, None]
    return torch.where(masked, _fill(S, mask_value), S)


def _payload(rows, n_cols, gen):
    """Random fp32 scores with every special value a real S chunk can carry."""
    S = torch.randn(rows, n_cols, generator=gen) * 30.0
    specials = torch.tensor([float("-inf"), float("inf"), float("nan"), -3.4028234663852886e38, 0.0, -0.0, 3.4028234663852886e38, 1e-45])
    pick = torch.rand(rows, n_cols, generator=gen) < 0.15
    idx = torch.randint(0, len(specials), (rows, n_cols), generator=gen)
    S = torch.where(pick, specials[idx], S)
    # NaN with a distinct payload: the select must pass the BITS through, not canonicalize them
    nan_payload = torch.tensor([0x7FC0BEEF], dtype=torch.int32).view(torch.float32)
    S[0, 3] = nan_payload[0]
    return S


def _assert_same(a, b, what):
    ai, bi = a.view(torch.int32), b.view(torch.int32)
    bad = (ai != bi).nonzero()
    assert bad.numel() == 0, f"{what}: {bad.shape[0]} cell(s) differ, first {bad[:8].tolist()}: cells={a[tuple(bad[0])].item()} bits={b[tuple(bad[0])].item()}"


_FLAG_SETS = [f for f in range(1, 8)]  # every non-empty subset of {PADDED, CAUSAL, SWA}
_FLAG_IDS = {1: "padded", 2: "causal", 3: "causal_padded", 4: "swa", 5: "swa_padded", 6: "causal_swa", 7: "causal_swa_padded"}


@pytest.mark.parametrize("n_cols", [64, 128], ids=["N64", "N128"])
@pytest.mark.parametrize("bottom_right", [0, 1], ids=["top_left", "bottom_right"])
@pytest.mark.parametrize("mask_flags", _FLAG_SETS, ids=[_FLAG_IDS[f] for f in _FLAG_SETS])
def test_bits_form_masks_the_same_cells_random(mask_flags, bottom_right, n_cols):
    """Random q rows, chunk bases (aligned and not, up to 2^22), sequence lengths, diagonals, windows
    and right bands, both sentinels: bitwise equal to the per-cell reference on every cell."""
    gen = torch.Generator().manual_seed(0x5EED + mask_flags * 16 + bottom_right * 4 + n_cols)
    rows = 512
    for mask_value, window_left, window_right in itertools.product(_SENTINELS, [0, 1, 5, 64, 640, 4096], [0, 3, 127, 129]):
        S = _payload(rows, n_cols, gen)
        q_abs = torch.randint(0, 1 << 14, (rows,), generator=gen)
        aligned = torch.randint(0, (1 << 22) // n_cols, (rows,), generator=gen) * n_cols
        kv_col_base = torch.where(torch.rand(rows, generator=gen) < 0.7, aligned, torch.randint(0, 1 << 22, (rows,), generator=gen))
        seq_kv_len = torch.randint(0, 1 << 14, (rows,), generator=gen)
        causal_diag = torch.randint(-(1 << 12), 1 << 12, (rows,), generator=gen)
        # keep most rows NEAR their band edges so the chunk straddles an edge (the interesting case)
        near = torch.rand(rows, generator=gen) < 0.8
        kv_col_base = torch.where(
            near, (q_abs + (causal_diag if bottom_right else 0) - torch.randint(-n_cols, 2 * n_cols, (rows,), generator=gen)).clamp(min=0), kv_col_base
        )
        args = (q_abs, kv_col_base, seq_kv_len, window_left, mask_flags, bottom_right, causal_diag, mask_value, window_right)
        _assert_same(
            mask_cells(S, *args), mask_bits(S, *args), f"flags={mask_flags} br={bottom_right} N={n_cols} W={window_left} R={window_right} v={mask_value}"
        )


@pytest.mark.parametrize("n_cols", [64, 128], ids=["N64", "N128"])
@pytest.mark.parametrize("mask_flags", _FLAG_SETS, ids=[_FLAG_IDS[f] for f in _FLAG_SETS])
def test_bits_form_masks_the_same_cells_at_word_edges(mask_flags, n_cols):
    """Exhaustive over every RELATIVE band-edge position in [-40, n_cols + 40] for the upper edge (causal /
    padded) and the lower edge (SWA) -- so every shift amount 0..32 and both saturated sides are hit for every
    word -- plus the padded edge on the same lattice.  Includes hi <= lo (fully masked), hi <= 0 and lo >= n_cols
    (fully masked through the saturated shift) and lo <= 0 with hi >= n_cols (fully unmasked)."""
    gen = torch.Generator().manual_seed(0xB175 + mask_flags + n_cols)
    edges = torch.arange(-40, n_cols + 41)
    hi_rel, lo_rel, pad_rel = torch.meshgrid(edges, edges, torch.arange(-8, n_cols + 9, 7), indexing="ij")
    hi_rel, lo_rel, pad_rel = hi_rel.flatten(), lo_rel.flatten(), pad_rel.flatten()
    rows = hi_rel.numel()
    kv_col_base = torch.randint(0, 1 << 20, (rows,), generator=gen) * 0 + 4096  # a fixed base: the edges are RELATIVE
    # the causal limit is q_caus_lim = q_abs + window_right, and the band's exclusive upper edge is q_caus_lim + 1
    window_right = 3
    q_abs = kv_col_base + hi_rel - 1 - window_right
    seq_kv_len = kv_col_base + pad_rel
    # Two windows: a narrow one (every lower-edge position lands INSIDE the chunk under bottom_right, where
    # causal_diag is chosen so lo = kv_col_base + lo_rel) and one wider than the chunk (so a fully-unmasked
    # row exists: a 17-column window under SWA can never keep all 64 / 128 columns).
    saw_fully_masked = saw_untouched = False
    for window_left in (17, n_cols + 96):
        # lo = q_abs (+ causal_diag) - window_left; under bottom_right the diagonal puts the SWA edge on lo_rel
        causal_diag = (kv_col_base + lo_rel + window_left) - q_abs
        for mask_value in _SENTINELS:
            for bottom_right in (0, 1):
                S = _payload(rows, n_cols, gen)
                args = (q_abs, kv_col_base, seq_kv_len, window_left, mask_flags, bottom_right, causal_diag, mask_value, window_right)
                ref = mask_cells(S, *args)
                got = mask_bits(S, *args)
                _assert_same(ref, got, f"edges flags={mask_flags} N={n_cols} br={bottom_right} W={window_left} v={mask_value}")
                # the sweep really reaches both degenerate rows somewhere (a narrow SWA window never leaves a row
                # untouched, a wide one never masks a whole row -- hence the two windows above)
                saw_fully_masked |= bool((ref.view(torch.int32) == _f32(mask_value).view(torch.int32)).all(dim=1).any())
                saw_untouched |= bool((ref.view(torch.int32) == S.view(torch.int32)).all(dim=1).any())
    assert saw_fully_masked, "no fully-masked row anywhere in the sweep"
    assert saw_untouched, "no fully-unmasked row anywhere in the sweep"


def test_bits_form_domain_guard():
    """The band arithmetic is Int32.  A window bound at or past `MASK_BOUND_LIMIT` can wrap `lo - kv_col_base`, and the
    bits form then masks EVERYTHING where the per-cell form masks nothing.  Three pins: the two forms agree at the largest
    in-domain window with every index below 2**28 (the documented domain); the Int32-faithful emulation shows the
    divergence just past it (why the guard exists, not a claim about the device); and `apply_mask_chunk_bits` refuses such
    a window at trace time, before it touches a register (Python ints, so the check costs no instruction)."""
    gen = torch.Generator().manual_seed(0xD0A1)
    rows, n_cols = 256, 128
    q_abs = torch.randint(0, 1 << 28, (rows,), generator=gen)
    kv_col_base = torch.randint(0, (1 << 28) // n_cols, (rows,), generator=gen) * n_cols
    seq_kv_len = torch.randint(0, 1 << 28, (rows,), generator=gen)
    causal_diag = torch.randint(-(1 << 27), 1 << 27, (rows,), generator=gen)
    for mask_flags in (MASK_SWA, MASK_CAUSAL | MASK_SWA, MASK_CAUSAL | MASK_SWA | MASK_PADDED):
        for bottom_right in (0, 1):
            S = _payload(rows, n_cols, gen)
            args = (q_abs, kv_col_base, seq_kv_len, MASK_BOUND_LIMIT - 1, mask_flags, bottom_right, causal_diag, _NEG_INF_BITS, MASK_BOUND_LIMIT - 1)
            _assert_same(mask_cells(S, *args), mask_bits(S, *args), f"largest in-domain window flags={mask_flags} br={bottom_right}")
    # Just past the domain: q row 0 with the widest Int32 window keeps every column under the per-cell form (nothing is
    # 2**31 - 1 columns back), while `lo - kv_col_base` wraps positive for any chunk past column 0 and the bits form masks
    # the whole chunk.  This is the emulated hazard the guard closes.
    S = _payload(rows, n_cols, gen)
    zero = torch.zeros(rows, dtype=torch.int64)
    base = torch.full((rows,), n_cols, dtype=torch.int64)
    args = (zero, base, zero, (1 << 31) - 1, MASK_SWA, 0, zero, _NEG_INF_BITS, 0)
    _assert_same(mask_cells(S, *args), S, "per-cell form with the widest window leaves the chunk untouched")
    _assert_same(mask_bits(S, *args), _fill(S, _NEG_INF_BITS), "bits form with the widest window masks the whole chunk")
    with pytest.raises(ValueError, match="window_left must be <"):
        apply_mask_chunk_bits(None, None, None, None, MASK_BOUND_LIMIT, MASK_SWA)
    with pytest.raises(ValueError, match="window_right must be <"):
        apply_mask_chunk_bits(None, None, None, None, 0, MASK_CAUSAL, window_right=MASK_BOUND_LIMIT)


def test_mask_form_vocabulary():
    """The kernels' `MASK_FORM` constant takes one of exactly these two spellings; a third would silently fall
    through `apply_mask_chunk_form` to its ValueError at trace time."""
    assert MASK_FORMS == (MASK_FORM_CELLS, MASK_FORM_BITS) == ("cells", "bits")
    assert MASK_WORD_COLS == 32
