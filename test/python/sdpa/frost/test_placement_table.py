# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated placement table: well-formed entries, the runtime class key the generator also uses, and the boundary
between table-decided and hand-rule graphs."""

import math

import pytest

import cudnn
from cudnn.sdpa import placement_table as pt
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts


def _facts(**over):
    base = dict(
        b=8,
        h_q=64,
        h_kv=1,
        s_q=1,
        s_kv=8192,
        d_qk=576,
        d_v=512,
        dtype=cudnn.data_type.BFLOAT16,
        causal=False,
        bottom_right=False,
        window_left=None,
        has_sink=False,
        has_paged_kv=True,
        thd=False,
        max_total_seq_len_q=None,
        device_cc=(10, 0),
        has_bias=False,
        has_dropout=False,
        right_band_widening=False,
    )
    base.update(over)
    return SdpaGraphFacts(**base)


@pytest.mark.L0
def test_entries_are_well_formed_and_name_known_rows():
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS as BWD
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS as FWD

    rows = {s.name for s in FWD} | {s.name for s in BWD}
    for key, entry in pt.TABLE.items():
        row, cc, dtype, heads, phase, layout, mask, sink, group, kv_off_tile = key
        assert row in rows, key
        assert isinstance(cc, tuple) and len(cc) == 2 and dtype in ("bf16", "f16") and len(heads) == 2
        assert phase in ("q1", "mtp", "verify", "prefill", "prefill_short_square", "prefill_short_q")
        assert layout.removesuffix("_bounded") in ("dense", "paged", "thd", "thd_ragged")
        assert mask in ("none", "window", "causal_br", "causal_tl") and isinstance(sink, bool) and group in ("mha", "gqa", "gqa_unpacked", "gqa_split")
        assert isinstance(kv_off_tile, bool)
        assert entry.direction in ("always", "never", "ge", "le") and entry.study
        assert (entry.log2_work is None) == (entry.direction in ("always", "never"))
        assert entry.feature in pt.FEATURES


@pytest.mark.L0
def test_class_key_ignores_bottom_right_causal_for_one_query_row():
    """A single bottom-right row sees every key (no mask); a single top-left row sees only key 0 -- a different graph."""
    assert pt.class_key("r", _facts(causal=True, bottom_right=True)) == pt.class_key("r", _facts(causal=False))
    assert pt.class_key("r", _facts(causal=True, bottom_right=False))[6] == "causal_tl"
    assert pt.class_key("r", _facts(s_q=2, causal=True, bottom_right=True))[6] == "causal_br"


@pytest.mark.L0
@pytest.mark.parametrize("feature", [dict(has_bias=True), dict(has_dropout=True), dict(right_band_widening=True), dict(is_fp8=True)])
def test_unmeasured_features_never_reach_the_table(feature):
    row = "sdpa_fwd_prefill_sm100"
    assert pt.lookup(row, _facts()) is not None, "the D576 decode entry should answer the plain graph"
    assert pt.lookup(row, _facts(**feature)) is None


@pytest.mark.L0
def test_bounded_declaration_is_its_own_class():
    """A bounded declaration (``shape_overrides``) serves every live shape up to the declared one, so an exact graph's entry
    never answers it: its class differs only in the layout, and it has no entry until a study measures it."""
    row = "sdpa_fwd_prefill_sm100"
    exact, bounded = pt.class_key(row, _facts()), pt.class_key(row, _facts(shape_overrides=True))
    assert bounded[5] == exact[5] + "_bounded" and bounded[:5] + bounded[6:] == exact[:5] + exact[6:]
    assert pt.lookup(row, _facts()) is not None and pt.lookup(row, _facts(shape_overrides=True)) is None


@pytest.mark.L0
def test_threshold_entries_split_on_log2_work():
    row = "sdpa_fwd_prefill_sm100"
    entry = pt.TABLE[pt.class_key(row, _facts())]
    assert entry.direction == "ge"
    low, high = _facts(h_q=16, b=1, s_kv=2048), _facts(h_q=128, b=128, s_kv=131072)
    assert pt.log2_work(low) < entry.log2_work < pt.log2_work(high)
    assert pt.lookup(row, low) is False and pt.lookup(row, high) is True
    assert pt.lookup(row, _facts(device_cc=(10, 1))) is None  # another part: no study, hand rules decide


@pytest.mark.L0
def test_cost_proxy_is_byte_bound_for_decode_and_flop_bound_for_prefill():
    """Decode cost follows KV bytes (independent of h_q at a fixed group of KV heads); prefill follows FLOPs."""
    dec = dict(d_qk=256, d_v=256, s_q=1, h_kv=1, b=16, s_kv=32768)
    assert pt.log2_work(_facts(h_q=4, **dec)) == pytest.approx(pt.log2_work(_facts(h_q=16, **dec)), abs=0.01)
    pre = dict(d_qk=256, d_v=256, s_kv=4096, h_kv=2, b=1, h_q=32)
    assert pt.log2_work(_facts(s_q=4096, **pre)) == pytest.approx(pt.log2_work(_facts(s_q=2048, **pre)) + 1)


@pytest.mark.L0
def test_kv_off_the_tile_is_its_own_class_with_its_own_entry():
    """S_kv off the 128-key tile is a separate class for dense and paged KV (an off-tile dense d256 decode was 11-13x slower on
    FROST; off-tile paged d256 decode wins at lower cost than on-tile), and every measured on-tile class has a measured
    off-tile twin, so an off-tile graph is never answered by an on-tile study nor dropped to the backend (review on #1957: an
    off-tile paged D576 decode, B1 H_q16 S_kv 49151, lost its FROST split route)."""
    row = "sdpa_fwd_prefill_sm100"
    for kw in (dict(), dict(has_paged_kv=False, d_qk=256, d_v=256, h_q=16)):
        assert pt.class_key(row, _facts(s_kv=8191, **kw))[-1] is True
        assert pt.class_key(row, _facts(s_kv=8192, **kw))[-1] is False
    for key in pt.TABLE:
        if key[5].endswith("_bounded"):
            continue  # FlashInfer declares capacities on the tile (128, powers of two, 65536); an off-tile one keeps the hand rules
        assert key[:-1] + (not key[-1],) in pt.TABLE, key
    assert pt.lookup(row, _facts(s_kv=49151, h_q=16, b=1)) is True


@pytest.mark.L0
def test_off_tile_dense_d256_decode_reads_parallel_tiles():
    """Off the KV tile FROST cannot split, so dense d256 decode follows the parallel tiles (b * h_kv), not the cost: the
    generated entry thresholds on "tiles" and is indifferent to S_kv."""
    row = "sdpa_fwd_prefill_sm100"
    d256 = dict(d_qk=256, d_v=256, has_paged_kv=False, h_q=16, h_kv=1)
    entry = pt.TABLE[pt.class_key(row, _facts(s_kv=8191, b=1, **d256))]
    assert entry.feature == "tiles"
    for s_kv in (2049, 32769, 131071):
        assert pt.lookup(row, _facts(s_kv=s_kv, b=16, **d256)) is False
        assert pt.lookup(row, _facts(s_kv=s_kv, b=64, **d256)) is True
        assert pt.lookup(row, _facts(s_kv=s_kv, b=40, **d256)) is True  # 33-63 tiles measured on 9.28 and 9.26 (review on #1957)


@pytest.mark.L0
def test_off_tile_dense_d576_decode_reads_query_rows():
    """Off the KV tile the unsplit tile runs every head of a request, so dense D576 decode follows the query rows b * h_q,
    not the cost: 256 rows lose (1.13-1.25 on the training grid) and 384 win (0.86-0.90), for any b / h_q split."""
    row = "sdpa_fwd_prefill_sm100"
    dense = dict(has_paged_kv=False, d_qk=576, d_v=512, h_kv=1, s_kv=32765)
    assert pt.TABLE[pt.class_key(row, _facts(b=1, h_q=8, **dense))].feature == "rows"
    for b, h_q in ((8, 32), (2, 128), (32, 8)):
        assert pt.lookup(row, _facts(b=b, h_q=h_q, **dense)) is False
    for b, h_q in ((4, 96), (3, 128), (48, 8)):
        assert pt.lookup(row, _facts(b=b, h_q=h_q, **dense)) is True


@pytest.mark.L0
def test_decode_the_row_cannot_pack_is_its_own_group():
    """A decode-shaped GQA graph the FROST row cannot pack re-reads K/V per query head -- another kernel regime (paged d512
    at G = 12-96: 5-21x slower than packed neighbours). The row's own pack decision splits the class, so the packed study
    never answers for it; a flavor that packs a partial group (D576 at G = 96 packs 32) keeps the packed class."""
    row = "sdpa_fwd_prefill_sm100"
    d512 = dict(d_qk=512, d_v=512, h_kv=1, b=8, s_kv=65536, page_size=64, padded=True)
    assert pt.class_key(row, _facts(h_q=96, **d512))[8] == "gqa_unpacked"
    assert pt.lookup(row, _facts(h_q=96, **d512)) is None  # no study measured it: the hand rule decides
    assert pt.class_key(row, _facts(h_q=64, **d512))[8] == "gqa"
    assert pt.lookup(row, _facts(h_q=64, **d512)) is True
    assert pt.class_key(row, _facts(h_q=96, page_size=64, padded=True))[8] == "gqa"  # D576 packs 32 of 96
    assert pt.class_key(row, _facts(h_q=96, s_q=1024, **d512))[8] == "gqa"  # prefill: s_q fills the tile, no packing regime


@pytest.mark.L0
def test_cc103_borrows_only_the_studies_its_transfer_check_named(monkeypatch):
    """Where cc 10.3 has no entry of its own, it answers with a cc 10.0 entry only when that entry's study is in
    PART_TRANSFERS[(10, 3)] (study sm103_transfer_20261031) -- never a later cc 10.0 study by default."""
    row = "sdpa_fwd_prefill_sm100"
    base, studies = pt.PART_TRANSFERS[(10, 3)]
    assert base == (10, 0)
    cases = (
        dict(),  # paged D576 decode
        dict(d_qk=512, d_v=512, b=8, s_kv=65536, page_size=64, padded=True),  # paged d512 decode
        dict(d_qk=576, d_v=512, has_paged_kv=False, b=1, h_q=96, s_kv=131072),  # dense D576 decode
        dict(d_qk=256, d_v=256, h_q=32, h_kv=2, s_q=2048, s_kv=65536, causal=True, bottom_right=True),  # paged d256 context
    )
    for kw in cases:
        on_100, on_103 = pt.lookup(row, _facts(**kw)), pt.lookup(row, _facts(device_cc=(10, 3), **kw))
        assert on_100 is not None and on_103 == on_100, kw
    key = pt.class_key(row, _facts())
    monkeypatch.setitem(pt.TABLE, key, pt.Entry("always", None, "a_later_cc100_study"))
    assert pt.lookup(row, _facts()) is True
    assert pt.lookup(row, _facts(device_cc=(10, 3))) is None


@pytest.mark.L0
@pytest.mark.parametrize("s", [1, 512])
def test_a_square_entry_reaches_the_runtime_from_either_corner(monkeypatch, s):
    """The generator classifies through class_key; an entry it emits for a top-left square must be the one place() honors."""
    from cudnn.sdpa.fwd import placement
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")
    tl, br = (_facts(s_q=s, s_kv=s, h_q=16, h_kv=2, d_qk=128, d_v=128, causal=True, bottom_right=a) for a in (False, True))
    key = pt.class_key(spec.name, tl)
    assert key == pt.class_key(spec.name, br) and ("none" if s == 1 else "causal_br") in key  # one row sees its only key
    for verdict, expected in (("never", placement.TRAIL), ("always", placement.LEAD)):
        monkeypatch.setattr(pt, "TABLE", {key: pt.Entry(verdict, None, "synthetic")})
        assert placement.place(spec, tl) == placement.place(spec, br) == expected


@pytest.mark.L0
def test_cc103_d256_backward_thd_entries_are_its_own():
    """cc 10.3's own d256 backward entries answer with no transfer: THD on the 128-key tile (study sm103_d256_bwd_v2_20261031,
    THD half) and off it (study sm103_d256_bwd_thd_offtile_20261031). The same graph on cc 10.0 has no entry, so the
    hand-written box decides there. The mask-free MHA classes split on the cost proxy, at their own threshold off the tile.
    Dense graphs have no entry: they lost on small-work holdout cases."""
    row = "sdpa_bwd_sm100_d256"

    def bwd(b, h_q, h_kv, s, causal=True, **over):
        base = dict(b=b, h_q=h_q, h_kv=h_kv, s_q=s, s_kv=s, d_qk=256, d_v=256, causal=causal, has_paged_kv=False, device_cc=(10, 3))
        base.update(thd=True, max_total_seq_len_q=b * s)
        base.update(over)
        return _facts(**base)

    assert pt.lookup(row, bwd(2, 16, 2, 4096)) is True  # THD causal GQA: always
    assert pt.lookup(row, bwd(2, 16, 2, 4096, max_total_seq_len_q=6 * 1024)) is True  # ragged causal GQA: always
    assert pt.lookup(row, bwd(2, 16, 2, 4096, device_cc=(10, 0))) is None
    small, large = bwd(1, 8, 8, 256, causal=False), bwd(1, 16, 16, 1024, causal=False)
    assert pt.log2_work(small) < 28.5 <= pt.log2_work(large)
    assert (pt.lookup(row, small), pt.lookup(row, large)) == (False, True)
    assert pt.lookup(row, bwd(2, 16, 2, 4096, thd=False, max_total_seq_len_q=None)) is None  # dense: no entry
    assert pt.lookup(row, bwd(2, 16, 2, 4096, dtype=cudnn.data_type.HALF)) is None  # f16: not measured
    assert pt.lookup(row, bwd(2, 16, 2, 4000)) is True  # off the 128-key tile: THD causal GQA, always
    assert pt.lookup(row, bwd(2, 16, 2, 4000, max_total_seq_len_q=6 * 1000)) is True  # ragged, off the tile
    assert pt.lookup(row, bwd(2, 16, 2, 4000, device_cc=(10, 0))) is None
    small, large = bwd(1, 8, 8, 1000, causal=False), bwd(1, 16, 16, 3000, causal=False)
    assert pt.log2_work(small) < 30.374 <= pt.log2_work(large)
    assert (pt.lookup(row, small), pt.lookup(row, large)) == (False, True)
    assert pt.lookup(row, bwd(2, 16, 2, 4000, thd=False, max_total_seq_len_q=None)) is None  # dense off the tile: no entry


@pytest.mark.L0
def test_cc103_d128_forward_dense_chunk_gqa_entries_are_its_own():
    """cc 10.3's own d128 forward entries for dense bottom-right causal GQA chunks (study sm103_d128_fwd_chunk_20261031, GQA
    half): they lead on and off the 128-key tile, past the hand rules' chunk-size and cache bounds, with no transfer. The same
    graphs on cc 10.0, MHA and f16 have no entry; top-left chunks belong to a separate study."""
    row = "sdpa_fwd_prefill_sm100"

    def chunk(b, h_q, h_kv, s_q, s_kv, **over):
        base = dict(b=b, h_q=h_q, h_kv=h_kv, s_q=s_q, s_kv=s_kv, d_qk=128, d_v=128, causal=True, bottom_right=True, has_paged_kv=False, device_cc=(10, 3))
        base.update(over)
        return _facts(**base)

    assert pt.lookup(row, chunk(4, 64, 8, 128, 2048)) is True  # 256 Q tiles from a 2k cache: past both hand-rule bounds
    assert pt.lookup(row, chunk(2, 16, 2, 512, 10000)) is True  # off the tile: its own class and entry
    assert pt.lookup(row, chunk(4, 64, 8, 128, 2048, device_cc=(10, 0))) is None
    assert pt.lookup(row, chunk(4, 32, 32, 128, 2048)) is None  # MHA: no entry
    assert pt.lookup(row, chunk(4, 64, 8, 128, 2048, dtype=cudnn.data_type.HALF)) is None  # f16: not measured
    # The top-left counterpart has its own measured entry after #2120.
    top_left = chunk(4, 64, 8, 128, 2048, bottom_right=False)
    key = pt.class_key(row, top_left)
    assert key[6] == "causal_tl"
    assert pt.TABLE[key].study == "sm103_d128_fwd_dense_20261031"


@pytest.mark.L0
def test_cc103_d128_forward_dense_causal_gqa_entries_are_its_own():
    """cc 10.3's own d128 forward entries (study sm103_d128_fwd_dense_20261031, causal GQA half): dense unequal-length top-left causal GQA graphs lead
    on and off the 128-key tile, with no transfer. The same graphs on cc 10.0, mask-free graphs and MHA have no entry."""
    row = "sdpa_fwd_prefill_sm100"

    def fwd(b, h_q, h_kv, s, causal=True, **over):
        base = dict(b=b, h_q=h_q, h_kv=h_kv, s_q=s // 2, s_kv=s, d_qk=128, d_v=128, causal=causal, has_paged_kv=False, device_cc=(10, 3))
        base.update(over)
        return _facts(**base)

    assert pt.lookup(row, fwd(2, 32, 8, 4096)) is True  # on the tile
    assert pt.lookup(row, fwd(4, 16, 2, 3000)) is True  # off the tile: its own class and entry
    assert pt.lookup(row, fwd(2, 32, 8, 4096, device_cc=(10, 0))) is None
    assert pt.lookup(row, fwd(2, 32, 8, 4096, causal=False)) is None  # mask-free: lost on its holdout
    assert pt.lookup(row, fwd(2, 16, 16, 4096)) is None  # MHA: no entry
    assert pt.lookup(row, fwd(2, 32, 8, 4096, dtype=cudnn.data_type.HALF)) is None  # f16: not measured
    # Squares share the bottom-right class; the top-left study must not claim them.
    square = fwd(2, 32, 8, 4096, s_q=4096)
    key = pt.class_key(row, square)
    assert key[6] == "causal_br"
    assert pt.TABLE[key].study == "sm103_d128_fwd_chunk_20261031"
    for s_q in (5, 64, 8192):
        facts = fwd(1, 16, 2, 2048, s_q=s_q)
        assert pt.lookup(row, facts) is True
        assert pt.TABLE[pt.class_key(row, facts)].study == "sm103_d128_fwd_dense_20261031"


@pytest.mark.L0
def test_b300_dense_prefill_entries_lead_the_sm100_row():
    """bf16 dense prefill-phase d192 and windowed d64 lead on cc 10.3 from their own study; f16, other head dims, a paged cache
    and cc 10.0 are not answered by these entries."""
    from cudnn.sdpa.fwd import placement
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    row = "sdpa_fwd_prefill_sm100"
    spec = next(s for s in ENGINE_SPECS if s.name == row)

    def prefill(**over):
        base = {"b": 16, "h_q": 32, "h_kv": 4, "s_q": 65, "s_kv": 3072, "d_qk": 192, "d_v": 192, "dtype": cudnn.data_type.BFLOAT16}
        base.update(causal=True, bottom_right=True, has_paged_kv=False, device_cc=(10, 3))
        base.update(over)
        return _facts(**base)

    half = cudnn.data_type.HALF
    d64 = {"d_qk": 64, "d_v": 64, "window_left": 1024}
    assert pt.TABLE[pt.class_key(row, prefill())].study == "sm103_dense_prefill_bf16_20261107"
    led = (
        prefill(),
        prefill(causal=False, bottom_right=False),
        prefill(h_q=8, h_kv=8, window_left=1024),
        prefill(s_kv=3073),
        prefill(h_q=8, h_kv=8, **d64),
        prefill(h_q=8, h_kv=8, s_kv=3073, **d64),
    )
    for facts in led:
        assert pt.lookup(row, facts) is True
        assert placement.place(spec, facts) == placement.LEAD

    assert pt.class_key(row, prefill(d_qk=64, d_v=64)) not in pt.TABLE  # d64 without a window
    assert pt.class_key(row, prefill(dtype=half)) not in pt.TABLE  # f16 keeps the hand rules
    assert pt.class_key(row, prefill(dtype=half, **d64)) not in pt.TABLE
    assert pt.class_key(row, prefill(dtype=half, d_qk=128, d_v=128)) not in pt.TABLE
    assert pt.class_key(row, prefill(has_paged_kv=True, page_size=64)) not in pt.TABLE
    assert pt.class_key(row, prefill(device_cc=(10, 0))) not in pt.TABLE


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [127, 128, 129, 45248, 45312, 45313])
def test_b300_d192_window_gqa_keeps_default_order(s_kv, monkeypatch):
    """Withdraw both tile classes after the q19/kv45248 holdout regression.

    Include both sides of the first tile boundary and long on/off-tile caches;
    windowed MHA and non-windowed GQA remain covered by the preceding study test.
    """
    from cudnn.sdpa.fwd import placement
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    row = "sdpa_fwd_prefill_sm100"
    spec = next(s for s in ENGINE_SPECS if s.name == row)
    facts = _facts(
        b=24,
        h_q=56,
        h_kv=8,
        s_q=19,
        s_kv=s_kv,
        d_qk=192,
        d_v=192,
        dtype=cudnn.data_type.BFLOAT16,
        causal=True,
        bottom_right=True,
        window_left=96,
        has_paged_kv=False,
        device_cc=(10, 3),
    )
    assert pt.lookup(row, facts) is None
    assert placement.place(spec, facts) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize("window_left", [None, 127], ids=["causal", "window"])
def test_sm107_dense_d64_sink_prefill_follows_its_study(window_left):
    """Study sm107_d64_sink_prefill_20261112 (Rubin): dense d64 prefill with a sink leads off the 128-key tile at any length and on the
    tile up to 2^8.5 parallel tiles; without a sink, f16, or another part the hand rules keep deciding."""
    row = "sdpa_fwd_prefill_sm107"

    def f(**over):
        base = dict(b=1, h_q=64, h_kv=8, s_q=128, s_kv=131072, d_qk=64, d_v=64, causal=True, bottom_right=True, has_sink=True, window_left=window_left)
        base.update(has_paged_kv=False, padded=False, page_size=0, device_cc=(10, 7), device_sm_count=208)
        base.update(over)
        return _facts(**base)

    for s_kv in (131072, 131071):
        assert pt.lookup(row, f(s_kv=s_kv)) is True
        assert pt.matched_entry(row, f(s_kv=s_kv)).study == "sm107_d64_sink_prefill_20261112"
    assert pt.lookup(row, f(b=4, s_q=8192, s_kv=131072)) is False  # 4 x 8 x 64 tiles: past the on-tile threshold
    assert pt.lookup(row, f(b=4, s_q=8192, s_kv=131071)) is True
    assert getattr(pt.matched_entry(row, f(has_sink=False)), "study", None) != "sm107_d64_sink_prefill_20261112"
    assert pt.lookup(row, f(dtype=cudnn.data_type.HALF)) is None
    assert pt.lookup(row, f(device_cc=(10, 3))) is None


@pytest.mark.L0
@pytest.mark.parametrize("row, cc", [("sdpa_fwd_prefill_sm80", (8, 0)), ("sdpa_fwd_prefill_sm90", (9, 0))], ids=["a100", "h200"])
def test_sm80_sm90_decode_shaped_gqa_follows_the_decode_studies(row, cc):
    """Studies sm80_decode_20261112 / sm90_decode_20261112: GQA decode-shaped rows -- s_q 2-4 (mtp) and 5-16 (verify, a row-local
    phase on SM80 / SM90) -- are their own classes and lead; MHA and s_q > 16 stay in the prefill class, and other rows keep theirs.
    d128 single-row dense decode is non-monotone: no entry."""
    study = "sm80_decode_20261112" if cc == (8, 0) else "sm90_decode_20261112"

    def f(**over):
        base = dict(b=8, h_q=32, h_kv=2, s_q=4, s_kv=32768, d_qk=256, d_v=256, causal=True, bottom_right=True)
        base.update(has_paged_kv=False, padded=False, page_size=0, device_cc=cc)
        base.update(over)
        return _facts(**base)

    for s_q in (2, 4, 8, 16):
        assert pt.lookup(row, f(s_q=s_q)) is True, s_q
        assert pt.matched_entry(row, f(s_q=s_q)).study == study
    assert pt.class_key(row, f(s_q=8))[4] == "verify"
    assert pt.class_key(row, f(s_q=17))[4] == "prefill"
    assert pt.class_key(row, f(s_q=8, h_q=2))[4] == "prefill"
    assert pt.class_key("sdpa_fwd_prefill_sm100", f(s_q=8, device_cc=(10, 0)))[4] == "prefill"
    assert pt.lookup(row, f(s_q=1, d_qk=128, d_v=128, h_q=64, h_kv=8, causal=False, bottom_right=False)) is None


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [5, 8, 16])
def test_sm80_verify_class_without_an_entry_keeps_its_prefill_entry(s_q):
    """A verify class no study answers (here SM80 d64) keeps the prefill entry it had before the verify phase existed."""
    row = "sdpa_fwd_prefill_sm80"
    cases = [
        dict(b=64, h_q=32, h_kv=4, s_kv=8193, causal=True, bottom_right=True),
        dict(b=8, h_q=32, h_kv=4, s_kv=8192, causal=True, bottom_right=True, window_left=127, has_sink=True),
    ]
    for over in cases:
        facts = _facts(s_q=s_q, d_qk=64, d_v=64, has_paged_kv=False, device_cc=(8, 0), **over)
        assert pt.class_key(row, facts)[4] == "verify" and pt.class_key(row, facts) not in pt.TABLE
        assert pt.matched_entry(row, facts) is pt.TABLE[pt.class_key(row, facts, verify=False)]
        assert pt.lookup(row, facts) is True


@pytest.mark.L0
def test_sm120_decode_entries_follow_their_study():
    """Study sm120_decode_20261112 (RTX PRO 6000): paged d256 and D576 single-row decode keep the backend, dense D576 and long paged
    d512 single-row decode lead; dense d512 s_q 1 has no entry (the hand rule decides, #2497); sink-only d64 MTP has no entry."""
    row = "sdpa_fwd_prefill_sm120"

    def f(**over):
        base = dict(b=8, h_q=32, h_kv=2, s_q=1, s_kv=32768, d_qk=256, d_v=256, page_size=16, padded=True, device_cc=(12, 0), device_sm_count=188)
        base.update(over)
        return _facts(**base)

    for s_kv in (32768, 32767):
        assert pt.lookup(row, f(s_kv=s_kv)) is False
        assert pt.matched_entry(row, f(s_kv=s_kv)).study == "sm120_decode_20261112"
    assert pt.lookup(row, f(h_q=96, h_kv=1, d_qk=576, d_v=512, page_size=32)) is False
    assert pt.lookup(row, f(h_q=96, h_kv=1, d_qk=576, d_v=512, has_paged_kv=False, padded=False, page_size=0)) is True
    assert pt.lookup(row, f(b=64, h_q=64, h_kv=1, d_qk=512, d_v=512, page_size=64, s_kv=131072)) is True  # 2^12 query rows past 2^11
    assert pt.lookup(row, f(b=8, h_q=64, h_kv=1, d_qk=512, d_v=512, page_size=64, s_kv=131072)) is False
    assert pt.lookup(row, f(h_q=128, h_kv=1, d_qk=512, d_v=512, has_paged_kv=False, padded=False, page_size=0)) is None
    sink = dict(h_q=64, h_kv=8, d_qk=64, d_v=64, s_q=2, has_sink=True, causal=True, bottom_right=True)
    assert pt.lookup(row, f(**sink)) is None
    assert pt.matched_entry(row, f(window_left=127, **sink)).study == "sm120_decode_20261112"


@pytest.mark.L0
def test_sm80_f16_dense_prefill_entries_are_their_own_study():
    """The A100 f16 dense prefill classes answer from their own study (sm80_f16_dense_prefill_20261106), not the bf16 one, and
    another Ampere part has no entry: the hand rules decide there."""
    row = "sdpa_fwd_prefill_sm80"

    def prefill(s_q, s_kv, d, h_q=16, h_kv=16, **over):
        base = dict(b=2, h_q=h_q, h_kv=h_kv, s_q=s_q, s_kv=s_kv, d_qk=d, d_v=d, dtype=cudnn.data_type.HALF, causal=True, bottom_right=True)
        base.update(has_paged_kv=False, device_cc=(8, 0))
        base.update(over)
        return _facts(**base)

    causal_d64 = prefill(1024, 1024, 64)
    assert pt.TABLE[pt.class_key(row, causal_d64)].study == "sm80_f16_dense_prefill_20261106"
    assert pt.lookup(row, causal_d64) is True  # d64 bottom-right causal MHA: FROST leads from 2^7.86 query tokens per request
    long_d128 = prefill(4096, 4096, 128, causal=False, bottom_right=False)
    assert pt.lookup(row, long_d128) is False  # d128 mask-free MHA: FROST leads only up to a cost of 2^23.02
    assert pt.lookup(row, prefill(1024, 1024, 64, causal=False, bottom_right=False)) is None  # a class the hand rules keep
    assert pt.lookup(row, prefill(16, 128, 128, causal=False, bottom_right=False)) is True  # small mask-free d128 MHA: FROST leads
    bf16 = prefill(1024, 1024, 64, dtype=cudnn.data_type.BFLOAT16)
    assert pt.TABLE[pt.class_key(row, bf16)].study.startswith("sm80_dense_prefill_")  # the bf16 study keeps its classes
    assert pt.lookup(row, prefill(1024, 1024, 64, device_cc=(8, 6))) is None


@pytest.mark.L0
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "mask_free"])
@pytest.mark.parametrize("s_kv", [2048, 2049], ids=["on_tile", "off_tile"])
def test_sm80_d192_gqa_prefill_leads_at_every_query_length(causal, s_kv):
    """Study sm80_d192_gqa_r5_20261112 replaces the q >= 271 threshold of the D192/V128 GQA classes: with packed GQA and split-KV
    FROST leads from 16 query tokens up (the old b1 64/8 q192 loss is gone). MHA and another Ampere part keep their own answers."""
    row = "sdpa_fwd_prefill_sm80"

    def prefill(s_q, **over):
        base = dict(b=1, h_q=64, h_kv=8, s_q=s_q, s_kv=s_kv, d_qk=192, d_v=128, causal=causal, bottom_right=causal)
        base.update(has_paged_kv=False, device_cc=(8, 0))
        base.update(over)
        return _facts(**base)

    for s_q in (16, 192, 1024):
        assert pt.lookup(row, prefill(s_q)) is True, s_q
        assert pt.matched_entry(row, prefill(s_q)).study == "sm80_d192_gqa_r5_20261112"
    assert getattr(pt.matched_entry(row, prefill(192, h_q=8)), "study", None) != "sm80_d192_gqa_r5_20261112"
    assert pt.lookup(row, prefill(192, device_cc=(8, 6))) is None


@pytest.mark.L0
def test_a100_sxm_dense_prefill_entries_lead_only_their_classes():
    """MHA d64 windowed context leads on cc 8.0 from their own study; a class whose fit kept a threshold, whose tile
    twin did not fit `always`, or that failed qualification has no entry, and another Ampere part keeps the hand rules."""
    row = "sdpa_fwd_prefill_sm80"

    def prefill(d, dtype=cudnn.data_type.HALF, **over):
        base = dict(
            b=2,
            h_q=16,
            h_kv=16,
            s_q=512,
            s_kv=4096,
            d_qk=d,
            d_v=d,
            dtype=dtype,
            causal=False,
            bottom_right=False,
            has_paged_kv=False,
            device_cc=(8, 0),
        )
        base.update(over)
        return _facts(**base)

    window = dict(causal=True, bottom_right=True, window_left=256)
    assert pt.lookup(row, prefill(256)) is None  # MHA failed native qualification; keep the prior rule
    assert pt.lookup(row, prefill(256, h_q=32, h_kv=8)) is None  # GQA also failed qualification
    assert pt.lookup(row, prefill(256, h_q=32, h_kv=8, s_kv=4100)) is None  # keep the tile twin on the prior rule too
    assert pt.TABLE[pt.class_key(row, prefill(64, **window))].study == "sm80_dense_prefill_f16_20261107"
    assert pt.lookup(row, prefill(64, cudnn.data_type.BFLOAT16, **window)) is True
    assert pt.TABLE[pt.class_key(row, prefill(64, cudnn.data_type.BFLOAT16, **window))].study == "sm80_dense_prefill_bf16_20261107"

    assert pt.lookup(row, prefill(64, h_q=32, h_kv=8, **window)) is None  # f16 d64 windowed GQA: its on-tile class kept a threshold
    assert pt.lookup(row, prefill(64, cudnn.data_type.BFLOAT16, h_q=32, h_kv=8, **window)) is None  # bf16 d64 windowed GQA: failed qualification
    assert pt.lookup(row, prefill(256, causal=True, bottom_right=True)) is None  # f16 d256 causal: the off-tile class kept a threshold
    assert pt.lookup(row, prefill(256, device_cc=(8, 6))) is None


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(32, 8), (32, 32)], ids=["gqa", "mha"])
def test_sm107_paged_prefill_entries_lead_their_classes(heads):
    """Study sm107_paged_prefill_20261109 (Rubin, FlashInfer's paged path): D192/V128 leads on the 128-key tile from 512 KV
    tokens and always off it; D256 leads everywhere. D128 belongs to study sm107_thd_prefill_20261111; f16, a window, a
    mask-free D192 graph and another part keep the hand rules."""
    row = "sdpa_fwd_prefill_sm107"
    h_q, h_kv = heads

    def paged(d=(192, 128), **over):
        base = dict(b=4, h_q=h_q, h_kv=h_kv, s_q=128, s_kv=512, d_qk=d[0], d_v=d[1], causal=True, bottom_right=True)
        base.update(thd=True, max_total_seq_len_q=4 * 128, device_cc=(10, 7))
        base.update(over)
        return _facts(**base)

    assert pt.matched_entry(row, paged()).study == "sm107_paged_prefill_20261109"
    assert pt.lookup(row, paged(s_kv=384)) is False  # on the tile below 512 KV the backend keeps the lead
    assert pt.lookup(row, paged(s_kv=512)) is True
    assert pt.lookup(row, paged(s_kv=385)) is True  # off the tile: always
    assert pt.matched_entry(row, paged(s_kv=385)).study == "sm107_paged_prefill_20261109"
    for s_kv in (256, 257, 65536):
        assert pt.lookup(row, paged(d=(256, 256), s_kv=s_kv)) is True
        assert pt.matched_entry(row, paged(d=(256, 256), s_kv=s_kv)).study == "sm107_paged_prefill_20261109"
    assert pt.matched_entry(row, paged(d=(128, 128))).study == "sm107_thd_prefill_20261111"
    assert pt.lookup(row, paged(dtype=cudnn.data_type.HALF)) is None
    assert pt.lookup(row, paged(window_left=128)) is None
    assert pt.lookup(row, paged(causal=False, bottom_right=False)) is None
    assert pt.lookup(row, paged(device_cc=(10, 3))) is None  # no transfer names this study


_SM107_PAGED_MTP_SHORT_KV = [
    # (head pair, (h_q, h_kv), mask, on-tile (KV, verdict) pairs, off-tile (KV, verdict) pairs) at b 1, s_q 2: where the thresholds
    # of study sm107_paged_mtp_20261110 fall; one KV tile is the backend's, and d512 / D576 lead at any length.
    ((64, 64), (32, 4), {}, ((128, False), (256, True)), ((49, False), (59, True))),
    ((64, 64), (64, 8), dict(has_sink=True), ((128, False), (256, True)), ((104, False), (140, True))),
    ((64, 64), (64, 8), dict(has_sink=True, window_left=128), ((128, True),), ((24, True),)),
    ((128, 128), (16, 2), {}, ((128, False), (8192, True)), ((24, False), (8191, True))),
    ((256, 256), (16, 1), {}, ((128, False), (256, True)), ((125, False), (133, True))),
    ((512, 512), (16, 2), {}, ((128, True),), ((24, True),)),
    ((512, 512), (128, 1), {}, ((128, True),), ((24, True),)),
    ((576, 512), (96, 1), {}, ((128, True),), ((24, True),)),
]


@pytest.mark.L0
@pytest.mark.parametrize(
    "d, heads, mask, on_tile, off_tile", _SM107_PAGED_MTP_SHORT_KV, ids=["d64", "d64_sink", "d64_sink_window", "d128", "d256", "d512_gqa", "d512", "d576"]
)
def test_sm107_paged_mtp_entries_lead_their_classes(d, heads, mask, on_tile, off_tile):
    """Study sm107_paged_mtp_20261110 (Rubin, the inference benchmark's paged generation graph): the bf16 MTP rows (s_q 2-4 under the
    bottom-right diagonal) lead from long caches down to the short-KV threshold of each class; d512 / D576 and the windowed d64 sink
    rows lead at any length. Without the diagonal, f16, and another part keep the hand rules."""
    row = "sdpa_fwd_prefill_sm107"

    def mtp(**over):
        base = dict(b=1, h_q=heads[0], h_kv=heads[1], s_q=2, s_kv=8192, d_qk=d[0], d_v=d[1], causal=True, bottom_right=True)
        base.update(device_cc=(10, 7), page_size=16, padded=True, **mask)
        base.update(over)
        return _facts(**base)

    for s_kv in (8192, 8191, 1048576):
        assert pt.lookup(row, mtp(b=4, s_q=3, s_kv=s_kv)) is True
    for s_kv, verdict in on_tile + off_tile:
        assert pt.lookup(row, mtp(s_kv=s_kv)) is verdict, s_kv
        assert pt.matched_entry(row, mtp(s_kv=s_kv)).study == "sm107_paged_mtp_20261110"
    assert pt.lookup(row, mtp(causal=False, bottom_right=False, window_left=None)) is None
    assert pt.lookup(row, mtp(dtype=cudnn.data_type.HALF)) is None
    assert pt.lookup(row, mtp(device_cc=(10, 3))) is None  # no transfer names this study


@pytest.mark.L0
@pytest.mark.parametrize("d", [(512, 512), (576, 512)], ids=["d512", "d576"])
@pytest.mark.parametrize("h_kv", [1, 2])
def test_sm107_paged_mla_single_row_decode_follows_the_q1_study(d, h_kv):
    """Study sm107_decode_q1_20261112 leads paged d512 / D576 s_q == 1 at every length on and off the tile, h_kv 1 and 2 (FROST
    0.12-0.83 of the backend). f16 and another part keep the hand rules."""
    row = "sdpa_fwd_prefill_sm107"
    for s_kv in (131072, 131071, 1024, 1023):
        facts = _facts(b=128, h_q=16 * h_kv, h_kv=h_kv, s_q=1, s_kv=s_kv, d_qk=d[0], d_v=d[1], device_cc=(10, 7), page_size=64, padded=True)
        assert pt.lookup(row, facts) is True
        assert pt.matched_entry(row, facts).study == "sm107_decode_q1_20261112"
    facts = dict(b=8, h_q=64, h_kv=1, s_q=1, s_kv=8192, d_qk=d[0], d_v=d[1], page_size=64, padded=True)
    assert pt.lookup(row, _facts(device_cc=(10, 7), dtype=cudnn.data_type.HALF, **facts)) is None
    assert pt.lookup(row, _facts(device_cc=(10, 3), **facts)) is None


@pytest.mark.L0
def test_sm107_dense_d256_single_row_decode_follows_the_q1_study():
    """Study sm107_decode_q1_20261112: dense d256 s_q == 1 leads past a cost threshold on the KV tile and a tile-count threshold off
    it; small launches keep the backend. d128 single-row decode has no entry (its holdout moved no cell)."""
    row = "sdpa_fwd_prefill_sm107"

    def q1(**over):
        base = dict(b=32, h_q=32, h_kv=2, s_q=1, s_kv=131072, d_qk=256, d_v=256, device_cc=(10, 7), has_paged_kv=False, padded=False, page_size=0)
        base.update(over)
        return _facts(**base)

    for s_kv in (131072, 131071):
        assert pt.lookup(row, q1(s_kv=s_kv)) is True
        assert pt.matched_entry(row, q1(s_kv=s_kv)).study == "sm107_decode_q1_20261112"
    assert pt.lookup(row, q1(b=1, s_kv=2048)) is False
    assert pt.lookup(row, q1(b=1, h_q=8, h_kv=1, s_kv=2049)) is False
    assert getattr(pt.matched_entry(row, q1(d_qk=128, d_v=128, h_q=64, h_kv=8)), "study", None) != "sm107_decode_q1_20261112"


def _fi_ragged(d, h_q, h_kv, b=4096, s_q=128, s_kv=65536, total=None, causal=True, **over):
    """FlashInfer's ragged cudnn declaration on cc 10.7: the broad envelope (b 4096, S_q 128 or 65536) or a power-of-two
    bounded class (b <= 4, with its packed-Q capacity), always with execute-time shape overrides."""
    base = dict(b=b, h_q=h_q, h_kv=h_kv, s_q=s_q, s_kv=s_kv, d_qk=d[0], d_v=d[1], causal=causal, bottom_right=causal)
    base.update(thd=True, padded=True, has_paged_kv=False, max_total_seq_len_q=total, shape_overrides=True, device_cc=(10, 7))
    base.update(over)
    return _facts(**base)


@pytest.mark.L0
def test_sm107_thd_prefill_bounded_declarations_follow_the_study():
    """Study sm107_thd_prefill_20261111 answers FlashInfer's ragged declarations by what they declare. The short-row envelope
    (S_q 128) leads at d64 / D192 causal / d256, where the backend's short-row engine ran 1.5-6.5x behind FROST; at d128 it
    keeps the backend from 64 query heads (FROST runs that declaration unpacked) and leads on the small bounded classes. The long
    envelope leads at d64; at d128 the holdout dropped the entry, so the hand rules decide. An exact graph of the same shape
    is another class."""
    row = "sdpa_fwd_prefill_sm107"
    study = "sm107_thd_prefill_20261111"
    for d in ((64, 64), (192, 128), (256, 256)):
        assert pt.lookup(row, _fi_ragged(d, 32, 8)) is True, d
        assert pt.matched_entry(row, _fi_ragged(d, 32, 8)).study == study
    assert pt.class_key(row, _fi_ragged((256, 256), 32, 8))[4:6] == ("prefill_short_q", "thd_bounded")
    assert pt.lookup(row, _fi_ragged((128, 128), 32, 8)) is True
    assert pt.lookup(row, _fi_ragged((128, 128), 64, 8)) is False
    assert pt.lookup(row, _fi_ragged((128, 128), 32, 8, b=2, s_kv=4096, total=256)) is True
    assert pt.lookup(row, _fi_ragged((64, 64), 32, 8, s_q=65536)) is True
    assert pt.lookup(row, _fi_ragged((128, 128), 32, 8, s_q=65536)) is None
    exact = _fi_ragged((256, 256), 32, 8, b=4, s_q=96, s_kv=4000, total=384, shape_overrides=False)
    assert pt.class_key(row, exact)[5] == "thd" and pt.lookup(row, exact) is None
    assert pt.lookup(row, _fi_ragged((256, 256), 32, 8, dtype=cudnn.data_type.HALF)) is None


@pytest.mark.L0
def test_sm107_thd_prefill_paged_entries_follow_the_study():
    """The same study's exact paged classes: d256 mask-free leads everywhere (0.38-0.80 of the default on the holdout), d128
    mask-free GQA squares past the threshold keep the backend, and d128 mask-free MHA keeps the hand rules (the holdout dropped
    both tile twins)."""
    row = "sdpa_fwd_prefill_sm107"

    def paged(d, h_q, h_kv, b=4, s_q=512, s_kv=4096, causal=False, **over):
        base = dict(b=b, h_q=h_q, h_kv=h_kv, s_q=s_q, s_kv=s_kv, d_qk=d[0], d_v=d[1], causal=causal, bottom_right=causal)
        base.update(thd=True, padded=True, page_size=16, max_total_seq_len_q=b * s_q, device_cc=(10, 7))
        base.update(over)
        return _facts(**base)

    for s_kv in (4096, 4100):
        assert pt.lookup(row, paged((256, 256), 32, 8, s_kv=s_kv)) is True
        assert pt.matched_entry(row, paged((256, 256), 32, 8, s_kv=s_kv)).study == "sm107_thd_prefill_20261111"
    assert pt.lookup(row, paged((128, 128), 32, 8, b=16, s_q=2688, s_kv=2688)) is False
    assert pt.lookup(row, paged((128, 128), 32, 32)) is None
    assert pt.lookup(row, paged((128, 128), 32, 32, s_kv=4100)) is None
    assert pt.lookup(row, paged((128, 128), 32, 8, causal=True)) is True


def _sm100_paged(d, h_q, h_kv, b, s_q, s_kv, **over):
    """FlashInfer's exact paged prefill declaration on a 148-SM cc 10.0 part (head-major page pool, page 16)."""
    from types import SimpleNamespace

    base = dict(b=b, h_q=h_q, h_kv=h_kv, s_q=s_q, s_kv=s_kv, d_qk=d, d_v=d, causal=True, bottom_right=True, thd=True, padded=True)
    base.update(page_size=16, max_total_seq_len_q=b * s_q, device_sm_count=148, k_t=SimpleNamespace(get_stride=lambda: (16 * d * h_kv, 16 * d, d, 1)))
    base.update(over)
    return _facts(**base)


@pytest.mark.L0
def test_sm100_paged_packed_split_classes_follow_their_study():
    """The SM100 row's packed split is its own group (``gqa_split``) and, at D64, one Q tile per request its own phase
    (``prefill_short_q``): study sm100_paged_d64_d256_prefill_20261111 leads the split and short-Q D64 graphs, keeps the backend
    for a long-Q D64 graph over a long cache, and leaves D256 unsplit graphs to sm100_d256_paged_context_20261030. cc 10.3 does
    not borrow the new study (its B300 holdout did not qualify the transfer): the split still leads there through the row's
    hand rule, and the unsplit D64 graphs keep the backend."""
    from cudnn.sdpa.fwd import placement
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    row = "sdpa_fwd_prefill_sm100"
    study = "sm100_paged_d64_d256_prefill_20261111"
    spec = next(s for s in ENGINE_SPECS if s.name == row)
    split64 = _sm100_paged(64, 32, 8, 1, 64, 32768)
    assert pt.class_key(row, split64)[4] == "prefill_short_q" and pt.class_key(row, split64)[8] == "gqa_split"
    split256 = _sm100_paged(256, 16, 8, 1, 128, 32768)
    assert pt.class_key(row, split256)[4] == "prefill" and pt.class_key(row, split256)[8] == "gqa_split"
    short_unsplit = _sm100_paged(64, 16, 8, 16, 64, 16384)
    for facts in (split64, split256, short_unsplit):
        assert pt.lookup(row, facts) is True
        assert pt.matched_entry(row, facts).study == study
    assert pt.lookup(row, _sm100_paged(64, 32, 8, 16, 512, 32768)) is False
    unsplit256 = _sm100_paged(256, 32, 8, 16, 512, 8192)
    assert pt.class_key(row, unsplit256)[8] == "gqa"
    assert pt.matched_entry(row, unsplit256).study == "sm100_d256_paged_context_20261030"
    exact_d128 = _sm100_paged(128, 32, 8, 1, 64, 32768)
    assert pt.class_key(row, exact_d128)[4] == "prefill" and pt.class_key(row, exact_d128)[8] == "gqa"
    b300 = dict(device_cc=(10, 3))
    for d, h_q, s_q in ((64, 32, 64), (256, 16, 128)):
        facts = _sm100_paged(d, h_q, 8, 1, s_q, 32768, **b300)
        assert pt.lookup(row, facts) is None and placement.place(spec, facts) == placement.LEAD
    assert pt.lookup(row, _sm100_paged(64, 16, 8, 16, 64, 16384, **b300)) is None
    assert pt.matched_entry(row, _sm100_paged(256, 32, 8, 16, 512, 8192, **b300)).study == "sm100_d256_paged_context_20261030"
