# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated placement table: well-formed entries, the runtime class key the generator also uses, and the boundary
between table-decided and hand-rule graphs."""

import math

import dataclasses

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

    # Retired SM90 records keep their measured study key through a private
    # policy domain; no public engine registration is implied by that key.
    from cudnn.sdpa.fwd.engines import sm100_legacy_policy_spec

    rows = {s.name for s in FWD} | {s.name for s in BWD} | {sm100_legacy_policy_spec("sdpa_fwd_prefill_sm90").name}
    for key, entry in pt.TABLE.items():
        row, cc, dtype, heads, phase, layout, mask, sink, group, kv_off_tile = key
        assert row in rows, key
        assert isinstance(cc, tuple) and len(cc) == 2 and dtype in ("bf16", "f16") and len(heads) == 2
        assert phase.removesuffix("_det") in ("q1", "mtp", "verify", "prefill", "prefill_short_square", "prefill_short_q")
        assert not phase.endswith("_det") or row in pt.DETERMINISTIC_KEYED_ROWS, key
        assert layout.removesuffix("_bounded") in ("dense", "paged", "thd", "thd_ragged")
        assert (
            mask in ("none", "window", "window_narrow", "causal_br", "causal_tl")
            and isinstance(sink, bool)
            and group in ("mha", "gqa", "gqa_unpacked", "gqa_split", "gqa_body", "gqa_partial")
        )
        assert mask != "window_narrow" or (row, phase) == ("sdpa_bwd_sm80", "prefill_det"), key  # a row-local key
        assert group != "gqa_body" or (row, layout, phase in ("mtp", "prefill")) == ("sdpa_fwd_prefill_sm120", "paged", True), key
        assert group != "gqa_partial" or (row, phase) in (("sdpa_fwd_prefill_sm107", "verify"), ("sdpa_fwd_prefill_sm90", "q1")), key
        assert isinstance(kv_off_tile, bool)
        assert entry.direction in ("always", "never", "ge", "le") and entry.study
        assert (entry.log2_work is None) == (entry.direction in ("always", "never"))
        assert entry.feature in pt.FEATURES
        assert entry.also is None or (
            entry.direction in ("ge", "le") and entry.also[0] in ("ge", "le") and isinstance(entry.also[1], float) and entry.also[2] in pt.FEATURES
        ), key


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
    FROST; off-tile paged d256 decode wins at lower cost than on-tile). Measured classes that cover both tile regimes
    need paired entries (review on #1957: an off-tile paged D576 decode, B1 H_q16 S_kv 49151, lost its FROST split route)."""
    row = "sdpa_fwd_prefill_sm100"
    for kw in (dict(), dict(has_paged_kv=False, d_qk=256, d_v=256, h_q=16)):
        assert pt.class_key(row, _facts(s_kv=8191, **kw))[-1] is True
        assert pt.class_key(row, _facts(s_kv=8192, **kw))[-1] is False
    for key in pt.TABLE:
        if key[5].endswith("_bounded"):
            continue  # FlashInfer declares capacities on the tile (128, powers of two, 65536); an off-tile one keeps the hand rules
        if key[0] == "sdpa_bwd_sm89_f16_d64":
            # The dedicated Ada body requires a whole KV tile.
            assert key[-1] is False, key
            continue
        twin = key[:-1] + (not key[-1],)
        assert twin in pt.TABLE or twin in pt.HAND_RULE_SIDES, key
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
        # FlashInfer's paged d64 short-Q prefill (llama3.2-1b, unsplit), study b300_d64_transfer_20261117
        dict(d_qk=64, d_v=64, h_q=32, h_kv=8, b=4, s_q=128, s_kv=8192, causal=True, bottom_right=True, thd=True, page_size=16, padded=True),
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
    # a paged cache is another study's class (paged THD d192 x 192 context, sm100_prefill_d192_gaps_20261118)
    assert pt.TABLE[pt.class_key(row, prefill(has_paged_kv=True, page_size=64))].study == "sm100_prefill_d192_gaps_20261118"
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
@pytest.mark.parametrize("s_kv", [131072, 131071], ids=["on_tile", "off_tile"])
def test_sm80_decode_gaps_follow_their_study(s_kv):
    """Study sm80_decode_gaps_20261117 (A100): MiMo-V2 SWA verify (window + sink) and gemma4 SWA paged d256 MTP / verify lead;
    ar_dit d128 MHA MTP leads on few tiles; dense d256 single-row decode failed its holdout here and belongs to a later study."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_decode_gaps_20261117"

    def f(**over):
        base = dict(b=1, h_q=64, h_kv=8, s_q=8, s_kv=s_kv, d_qk=192, d_v=128, causal=True, bottom_right=True, window_left=127, has_sink=True)
        base.update(device_cc=(8, 0), page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for over in (dict(), dict(has_paged_kv=False, page_size=0, padded=False), dict(b=64, h_q=16, h_kv=1)):  # MiMo-V2 SWA verify
        assert pt.lookup(row, f(**over)) is True, over
        assert pt.matched_entry(row, f(**over)).study == study, over
    gemma = dict(h_q=16, h_kv=8, d_qk=256, d_v=256, window_left=1023, has_sink=False)
    for s_q in (4, 8):  # gemma4 SWA MTP / verify
        assert pt.lookup(row, f(**dict(gemma, s_q=s_q))) is True, s_q
        assert pt.matched_entry(row, f(**dict(gemma, s_q=s_q))).study == study
    ar_dit = dict(h_q=9, h_kv=9, s_q=2, d_qk=128, d_v=128, window_left=None, has_sink=False, has_paged_kv=False, page_size=0, padded=False)
    assert pt.lookup(row, f(**ar_dit)) is True
    assert pt.lookup(row, f(**dict(ar_dit, b=64, h_q=32, h_kv=32))) is False  # 2^11 tiles
    qwen = dict(h_q=32, h_kv=2, s_q=1, d_qk=256, d_v=256, causal=False, window_left=None, has_sink=False, has_paged_kv=False, page_size=0, padded=False)
    other = pt.matched_entry(row, f(**qwen))
    assert other is None or other.study != study  # dense d256 single-row decode: failed its holdout


@pytest.mark.L0
@pytest.mark.parametrize("off", [0, 1], ids=["on_tile", "off_tile"])
def test_sm80_dense_d256_single_row_decode_follows_its_study(off):
    """sm80_dense_d256_q1_v3_pcie_20261121: SM80 dense d256 GQA single-row decode is owned by its study on both tile sides, FROST's lead
    never shrinks as the cache or the batch grows, and a one-KV-head b 1 decode over a 1k-key cache keeps the backend while a b 128
    decode over 128k keys leads."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_dense_d256_q1_v3_pcie_20261121"

    def f(b, h_q, h_kv, s_kv):
        dense = dict(has_paged_kv=False, page_size=0, padded=False)
        return _facts(b=b, h_q=h_q, h_kv=h_kv, s_q=1, s_kv=s_kv + off, d_qk=256, d_v=256, causal=False, device_cc=(8, 0), **dense)

    heads = ((32, 2), (8, 1), (16, 1), (4, 1), (21, 3), (40, 5), (36, 6))
    for shape in [(b, h_q, h_kv, s_kv) for b in (1, 3, 8, 128) for h_q, h_kv in heads for s_kv in (1024, 8192, 131072)]:
        assert pt.matched_entry(row, f(*shape)).study == study, shape
    for h_q, h_kv in heads:
        for b in (1, 3, 8):
            leads = [pt.lookup(row, f(b, h_q, h_kv, s_kv)) for s_kv in (1024, 2048, 4096, 16384, 65536, 131072)]
            assert leads == sorted(leads), (b, h_q, h_kv, leads)
        for s_kv in (1024, 2048, 8192):
            leads = [pt.lookup(row, f(b, h_q, h_kv, s_kv)) for b in (1, 2, 3, 6, 16, 128)]
            assert leads == sorted(leads), (s_kv, h_q, h_kv, leads)
    assert pt.lookup(row, f(1, 8, 1, 1024)) is False
    assert pt.lookup(row, f(128, 32, 2, 131072)) is True


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131071], ids=["on_tile", "off_tile"])
def test_sm80_paged_d256_single_row_decode_follows_its_study(s_kv):
    """sm80_paged_d256_q1_v3_pcie_20261118 (A100 80GB PCIe): after the paged d256 split rule, single-row decode on the tile leads from
    cost 2^24.511 (the qwen3.5 / qwen3.6 b 1 single-KV-head rows over 131k keys included); off the tile the gaps study's floor stays."""
    row = "sdpa_fwd_prefill_sm80"

    def f(**over):
        base = dict(b=1, h_q=8, h_kv=1, s_q=1, s_kv=s_kv, d_qk=256, d_v=256, causal=True, bottom_right=True, device_cc=(8, 0))
        base.update(page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    if s_kv % 128 == 0:
        assert pt.matched_entry(row, f()).study == "sm80_paged_d256_q1_v3_pcie_20261118"
        assert pt.lookup(row, f()) is True
        assert pt.lookup(row, f(s_kv=1024)) is False  # cost 2^24
    else:
        assert pt.matched_entry(row, f()).study == "sm80_decode_gaps_20261117"


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131071], ids=["on_tile", "off_tile"])
def test_sm80_paged_wide_head_verify_follows_its_study(s_kv):
    """Study sm80_paged_wide_verify_20261117 (A100): paged d192 x V128 speculative decode at s_q 2-16 leads (FROST 0.13-0.20 of the
    backend). Its d512 / D576 verify twins are retired (a short-cache supplement lost 1.10-1.16x); windowed graphs and f16 are other
    studies' or the hand rules'."""
    row = "sdpa_fwd_prefill_sm80"

    def spec(**over):
        base = dict(b=1, h_q=16, h_kv=1, s_q=8, s_kv=s_kv, d_qk=576, d_v=512, causal=True, bottom_right=True)
        base.update(device_cc=(8, 0), page_size=64, padded=True)
        base.update(over)
        return _facts(**base)

    for over in (dict(), dict(h_q=128), dict(d_qk=512, d_v=512, h_q=16, h_kv=2, s_q=5)):  # re-studied over KV 128-128k
        assert pt.matched_entry(row, spec(**over)).study == "sm80_paged_wide_verify_v2_20261117", over
        assert pt.lookup(row, spec(**over)) is True, over
    assert pt.lookup(row, spec(d_qk=512, d_v=512, h_q=8, h_kv=1, s_q=5, b=1, s_kv=128 + (s_kv % 2))) is False  # below the cost floor
    for over in (
        dict(d_qk=192, d_v=128, h_q=64, h_kv=4, s_q=2),
        dict(d_qk=192, d_v=128, h_q=64, h_kv=4, s_q=16, b=64),
    ):
        assert pt.class_key(row, spec(**over))[4] in ("mtp", "verify"), over
        assert pt.lookup(row, spec(**over)) is True, over
        assert pt.matched_entry(row, spec(**over)).study == "sm80_paged_wide_verify_20261117", over
    other = pt.matched_entry(row, spec(d_qk=512, d_v=512, s_q=4))  # d512 MTP: not this study's class
    assert other is None or other.study != "sm80_paged_wide_verify_20261117"
    assert pt.lookup(row, spec(dtype=cudnn.data_type.HALF)) is None


@pytest.mark.L0
@pytest.mark.parametrize("window_left", [None, 128], ids=["causal", "window"])
@pytest.mark.parametrize("s_kv", [131072, 131071], ids=["on_tile", "off_tile"])
def test_sm80_paged_d64_sink_mtp_follows_its_study(window_left, s_kv):
    """Study sm80_paged_mtp_sink_20261117 (A100, gpt-oss's paged generation graph): d64 sink MTP rows lead once enough tiles are in
    flight and keep the backend at b 1; f16 and another part keep the hand rules (no sink: sm80_gen_gaps_20261117)."""
    row = "sdpa_fwd_prefill_sm80"

    def mtp(**over):
        base = dict(b=128, h_q=64, h_kv=8, s_q=2, s_kv=s_kv, d_qk=64, d_v=64, causal=True, bottom_right=True, has_sink=True)
        base.update(window_left=window_left, device_cc=(8, 0), page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for b in (8, 128):
        assert pt.lookup(row, mtp(b=b)) is True, b
        assert pt.matched_entry(row, mtp(b=b)).study == "sm80_paged_mtp_sink_20261117"
    assert pt.lookup(row, mtp(b=1, h_q=8, h_kv=1)) is False
    assert pt.matched_entry(row, mtp(b=1, h_q=8, h_kv=1)).study == "sm80_paged_mtp_sink_20261117"
    assert pt.matched_entry(row, mtp(has_sink=False)).study == "sm80_gen_gaps_20261117"  # no sink: the generation-gaps study
    assert pt.lookup(row, mtp(dtype=cudnn.data_type.HALF)) is None
    assert pt.lookup(row, mtp(device_cc=(8, 6))) is None


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [8192, 65537], ids=["on_tile", "off_tile"])
def test_sm80_thd_q_paged_verify_keys_the_unpacked_regime(s_kv):
    """A100 THD Q cannot pack or split on the k_prefetch / d256 engines: paged THD-Q verify keys ``gqa_unpacked`` (llama3.1 /
    qwen3.5 context re-time: FROST 1.25-3.3x behind). Study sm80_thd_verify_unpacked_20261117: d128 / d256 keep the backend (no
    entry), d192 x 128 leads on few rows. d64 leads past a floor, re-fit on the nightly's board (sm80_thd_verify_d64_pcie_20261118);
    its sink twin failed off the tile there. Padded Q keeps its packed class's entry; the wide engine's THD-Q verify keeps its
    entries (sm80_d576_prefill_20261117 measured FROST at 0.34-0.55)."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_thd_verify_unpacked_20261117"
    on_tile = s_kv % 128 == 0

    def f(**over):
        base = dict(b=16, h_q=64, h_kv=8, s_q=16, s_kv=s_kv, d_qk=128, d_v=128, causal=True, bottom_right=True)
        base.update(device_cc=(8, 0), page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    def thd(**over):
        return f(**dict(over, thd=True, max_total_seq_len_q=over.get("b", 16) * over.get("s_q", 16)))

    for over in (dict(), dict(h_q=32, h_kv=2, d_qk=256, d_v=256), dict(d_qk=192, d_v=128, s_q=5), dict(d_qk=64, d_v=64)):
        assert pt.class_key(row, thd(**over))[4:9:4] == ("verify", "gqa_unpacked"), over
        assert pt.class_key(row, f(**over))[8] == "gqa", over
    for over in (dict(), dict(h_q=32, h_kv=2, d_qk=256, d_v=256), dict(d_qk=192, d_v=128, s_q=5)):
        assert pt.lookup(row, f(**over)) is True, over
    for over in (dict(), dict(h_q=32, h_kv=2, d_qk=256, d_v=256)):
        assert pt.matched_entry(row, thd(**over)) is None, over
    d192 = dict(d_qk=192, d_v=128, s_q=5)
    assert pt.matched_entry(row, thd(**d192)).study == study
    assert pt.lookup(row, thd(**d192, b=1, h_q=12, h_kv=2)) is True  # few rows
    assert pt.lookup(row, thd(**d192)) is False
    d64 = dict(d_qk=64, d_v=64)
    assert pt.matched_entry(row, thd(**d64)).study == "sm80_thd_verify_d64_pcie_20261118"
    assert pt.lookup(row, thd(**d64)) is True
    small = dict(d64, b=1, h_q=8, h_kv=1) if on_tile else dict(d64, b=1, h_q=8, h_kv=1, s_kv=1025)
    assert pt.lookup(row, thd(**small)) is False  # below the floor
    assert pt.lookup(row, thd(**d64, b=4, h_q=32, h_kv=8)) is True  # llama3.2-1b at b 4: FROST 0.79-0.88 on the nightly's board
    assert pt.lookup(row, thd(**d64, has_sink=True)) is None  # the sink twin failed off the tile (1.150): neither side ships
    wide = dict(h_q=64, h_kv=1, d_qk=576, d_v=512, page_size=64, thd=True, max_total_seq_len_q=256)
    assert pt.class_key(row, f(**wide))[8] == "gqa"
    assert pt.matched_entry(row, f(**wide)).study == "sm80_paged_wide_verify_v2_20261117"


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [65536, 65537], ids=["on_tile", "off_tile"])
def test_sm80_generation_gaps_follow_their_study(s_kv):
    """Study sm80_gen_gaps_20261117 (A100, padded Q): paged d64 (muse_spark), windowed d64 / d128 and d192 x 192 (step5) MTP and verify
    lead (the backend ran them 1.2-5.1x behind); windowed d64 single-row decode follows sm80_paged_window_decode_pcie_20261118; THD-Q
    verify keeps its own class."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_gen_gaps_20261117"

    def f(**over):
        base = dict(b=4, h_q=128, h_kv=16, s_q=2, s_kv=s_kv, d_qk=64, d_v=64, causal=True, bottom_right=True, device_cc=(8, 0))
        base.update(page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    leads = (
        dict(),
        dict(s_q=8),
        dict(window_left=2047),
        dict(d_qk=192, d_v=192, h_q=64, h_kv=4),
        dict(d_qk=192, d_v=192, h_q=64, h_kv=4, s_q=8),
        dict(d_qk=128, d_v=128, h_q=80, h_kv=8, s_q=8, window_left=1023),
    )
    for over in leads:
        on_tile_d64_sink = over == dict(has_sink=True) and s_kv % 128 == 0
        expected = "sm80_paged_d64_ontile_pcie_20261118" if on_tile_d64_sink else study
        assert pt.matched_entry(row, f(**over)).study == expected, over
        assert pt.lookup(row, f(**over)) is True, over
    assert pt.matched_entry(row, f(s_q=1, window_left=2047)).study == "sm80_paged_window_decode_pcie_20261118"
    thd = f(s_q=8, thd=True, max_total_seq_len_q=32)
    assert pt.class_key(row, thd)[8] == "gqa_unpacked" and pt.matched_entry(row, thd).study != study


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm80_paged_window_decode_follows_the_pcie_study(s_kv):
    """sm80_paged_window_decode_pcie_20261118 (A100 80GB PCIe, padded Q, no sink): d128 windowed MTP leads at every S_q on the tile
    (step37 / ripple S_q 4 ran the backend at 1.3-3.9x); single-row windowed decode leads up to 2^5.5 parallel tiles for d128 and
    2^7.5 tiles (on the tile) / 2^10.661 query rows (off it) for d64. Larger batches keep the backend."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_paged_window_decode_pcie_20261118"
    on_tile = s_kv % 128 == 0

    def f(**over):
        base = dict(b=1, h_q=64, h_kv=8, s_q=1, s_kv=s_kv, d_qk=128, d_v=128, causal=True, bottom_right=True, window_left=511)
        base.update(device_cc=(8, 0), page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    assert pt.matched_entry(row, f()).study == study
    assert pt.lookup(row, f()) is True  # 2^3 tiles
    assert pt.lookup(row, f(b=128, h_q=16, h_kv=2)) is False  # 2^8 tiles
    if on_tile:
        assert pt.lookup(row, f(s_q=4, b=128, h_q=16, h_kv=2)) is True
        assert pt.matched_entry(row, f(s_q=4)).study == study
    else:
        assert pt.matched_entry(row, f(s_q=4)).study == "sm80_gen_gaps_20261117"
    d64 = dict(d_qk=64, d_v=64, h_q=16, h_kv=2, window_left=2047)
    assert pt.lookup(row, f(**d64)) is True
    assert pt.lookup(row, f(**dict(d64, b=128, h_q=128, h_kv=16))) is False


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm80_prefill_gaps_follow_their_study(s_kv):
    """Study sm80_prefill_gaps_20261117 (A100): paged THD-Q prefill (gpt-oss d64 with a sink, causal and windowed; dsv4 d512) and dense
    dsv4 d512 lead past a floor; small chunks keep the backend. Paged d128 has no entry; paged d192 x 128 and d256 were retired after
    their edge supplement (few query tokens over a long cache lost 1.30-1.47x) and re-studied as sm80_paged_d256_d192_v3_20261117."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_prefill_gaps_20261117"

    def f(**over):
        base = dict(b=1, h_q=64, h_kv=8, s_q=512, s_kv=s_kv, d_qk=64, d_v=64, causal=True, bottom_right=True, device_cc=(8, 0))
        base.update(page_size=16, padded=True, thd=True, max_total_seq_len_q=512)
        base.update(over)
        return _facts(**base)

    leads = (
        dict(has_sink=True),
        dict(has_sink=True, window_left=127),
        dict(d_qk=512, d_v=512, h_q=64, h_kv=1, page_size=64),
        dict(d_qk=512, d_v=512, h_q=64, h_kv=1, has_paged_kv=False, thd=False, max_total_seq_len_q=None, page_size=0, padded=False),
    )
    for over in leads:
        on_tile_d64_sink = over == dict(has_sink=True) and s_kv % 128 == 0
        expected = "sm80_paged_d64_ontile_pcie_20261118" if on_tile_d64_sink else study
        expected = "sm80_paged_d64_swa_pcie_20261118" if "window_left" in over else expected
        assert pt.matched_entry(row, f(**over)).study == expected, over
        assert pt.lookup(row, f(**over)) is True, over
    small = dict(s_q=32, s_kv=1024 + s_kv % 2, max_total_seq_len_q=32)
    assert pt.lookup(row, f(**small, h_q=8, h_kv=2)) is False
    assert pt.matched_entry(row, f(d_qk=128, d_v=128)) is None
    for over in (dict(d_qk=256, d_v=256, h_q=32, h_kv=2), dict(d_qk=192, d_v=128, h_q=64, h_kv=4)):
        on_tile_owner = {256: "sm80_paged_d256_v7_pcie_20261118", 192: "sm80_paged_d192x128_pcie_20261121"}[over["d_qk"]]
        expected = on_tile_owner if s_kv % 128 == 0 else "sm80_paged_d256_d192_v3_20261117"
        assert pt.matched_entry(row, f(**over)).study == expected, over


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm80_paged_d256_window_prefill_follows_its_study(s_kv):
    """sm80_paged_d256_swa_v4_pcie_20261121: SM80 paged THD-Q d256 windowed prefill is owned by its study on both tile sides, FROST's
    lead never shrinks as the grid widens, and the narrowest grid keeps the backend while the widest leads."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_paged_d256_swa_v4_pcie_20261121"

    def f(b, h_q, h_kv, s_q):
        base = dict(b=b, h_q=h_q, h_kv=h_kv, s_q=s_q, s_kv=s_kv, d_qk=256, d_v=256, causal=True, bottom_right=True, device_cc=(8, 0))
        base.update(page_size=16, padded=True, thd=True, max_total_seq_len_q=b * s_q, window_left=1023)
        return _facts(**base)

    for shape in [(b, h_q, h_kv, s_q) for b in (1, 4) for h_q, h_kv in ((8, 4), (16, 8), (32, 16)) for s_q in (512, 2048, 8192)]:
        assert pt.matched_entry(row, f(*shape)).study == study, shape
    for s_q in (512, 2048, 8192):
        leads = [pt.lookup(row, f(b, 2 * h_kv, h_kv, s_q)) for b, h_kv in ((1, 1), (1, 4), (1, 16), (4, 16), (16, 16))]
        assert leads == sorted(leads), (s_q, leads)
    assert pt.lookup(row, f(1, 2, 1, 512)) is False
    assert pt.lookup(row, f(16, 32, 16, 8192)) is True


@pytest.mark.L0
@pytest.mark.parametrize("off_tile", [False, True], ids=["on_tile", "off_tile"])
def test_sm80_training_thd_and_d512_window_prefill_follow_their_study(off_tile):
    """sm80_train_fwd_gaps_pcie_20261123 (A100 80GB PCIe): the training harness's equal-length THD d256, ragged THD MHA and dense d512
    sliding-window prefill are owned by the study on both tile sides, and FROST's lead never shrinks as the grid widens. Equal-length THD
    d128 and ragged THD d128 GQA keep no entry; ragged THD d256 GQA keeps its earlier study."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_train_fwd_gaps_pcie_20261123"

    def thd(b, h_q, h_kv, s, d=256, ragged=False):
        s += off_tile
        base = dict(b=b, h_q=h_q, h_kv=h_kv, s_q=s, s_kv=s, d_qk=d, d_v=d, causal=True, device_cc=(8, 0), has_paged_kv=False, thd=True)
        return _facts(**base, max_total_seq_len_q=b * s // 2 if ragged else b * s)

    def window(b, h_q, h_kv, s, left=128):
        s += off_tile
        base = dict(b=b, h_q=h_q, h_kv=h_kv, s_q=s, s_kv=s, d_qk=512, d_v=512, causal=True, device_cc=(8, 0), has_paged_kv=False)
        return _facts(**base, window_left=left - 1)

    for facts in (thd(64, 32, 8, 1024), thd(16, 16, 16, 2048), thd(16, 16, 16, 2048, ragged=True), thd(16, 16, 16, 2048, d=128, ragged=True)):
        assert pt.matched_entry(row, facts).study == study, facts
    for facts in (window(2, 128, 1, 8192), window(2, 64, 1, 2048), window(4, 16, 16, 4096, left=1024)):
        assert pt.matched_entry(row, facts).study == study, facts
        assert pt.lookup(row, facts) is True, facts
    for make in (lambda b, s: thd(b, 4 * b, b, s), lambda b, s: thd(b, 2 * b, 2 * b, s), lambda b, s: window(b, 8 * b, b, s)):
        for s in (128, 1024, 8192):
            leads = [pt.lookup(row, make(b, s)) for b in (1, 2, 4, 8, 16)]
            assert leads == sorted(leads), (s, leads)
    assert pt.lookup(row, thd(1, 4, 1, 128)) is False
    assert pt.lookup(row, thd(64, 32, 8, 1024)) is True
    assert pt.matched_entry(row, thd(16, 16, 4, 2048, d=128)) is None
    assert pt.matched_entry(row, thd(16, 16, 4, 2048, d=128, ragged=True)) is None
    assert pt.matched_entry(row, thd(16, 16, 4, 2048, ragged=True)).study == "sm80_thd_d256_v2_20261117"


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [65536, 65537], ids=["on_tile", "off_tile"])
@pytest.mark.parametrize("sink", [False, True], ids=["no_sink", "sink"])
def test_sm80_paged_d64_window_prefill_follows_the_pcie_study(sink, s_kv):
    """sm80_paged_d64_swa_pcie_20261118 (A100 80GB PCIe): paged THD-Q d64 sliding-window prefill leads from 256 query tokens per
    request, with or without a sink: gpt-oss (128-key window, sink) and muse_spark (2048-key window) chunks of 512-1024 tokens ran
    the backend at 1.2-1.95x FROST. Short chunks keep the backend; off the tile without a sink the floor is 2^6.5 parallel tiles."""
    row = "sdpa_fwd_prefill_sm80"

    def f(**over):
        base = dict(b=1, h_q=8, h_kv=1, s_q=512, s_kv=s_kv, d_qk=64, d_v=64, causal=True, bottom_right=True, device_cc=(8, 0))
        base.update(page_size=16, padded=True, thd=True, max_total_seq_len_q=512, has_sink=sink, window_left=127 if sink else 2047)
        base.update(over)
        return _facts(**base)

    assert pt.matched_entry(row, f()).study == "sm80_paged_d64_swa_pcie_20261118"
    if sink or s_kv % 128 == 0:
        for over in (dict(), dict(s_q=1024, max_total_seq_len_q=1024), dict(h_q=128, h_kv=16)):
            assert pt.lookup(row, f(**over)) is True, over
        assert pt.lookup(row, f(s_q=128, max_total_seq_len_q=128)) is False
    else:  # no sink, off the tile: parallel tiles b * h_kv * ceil(s_q * G / 128) from 2^6.5
        assert pt.lookup(row, f(h_q=128, h_kv=16)) is True  # 2^9
        assert pt.lookup(row, f(s_q=128, max_total_seq_len_q=128, b=8, h_q=64, h_kv=8)) is True  # 2^9
        assert pt.lookup(row, f()) is False  # 2^5


@pytest.mark.L0
@pytest.mark.parametrize("sink", [False, True], ids=["no_sink", "sink"])
def test_sm80_paged_d64_on_tile_floors_follow_the_pcie_study(sink):
    """sm80_paged_d64_ontile_pcie_20261118 (A100 80GB PCIe): paged THD-Q d64 causal prefill on the KV tile leads from cost 2^29.292
    without a sink and 2^30.044 with one (the 2^34.161 / 2^33.792 floors were fit on a 275 W board), so llama3.2 and gpt-oss chunks
    at cost 2^31-2^34 (FROST 0.51-0.92 of the backend) lead; small chunks keep the backend. Off the tile the earlier entries stay."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_paged_d64_ontile_pcie_20261118"

    def f(**over):
        base = dict(b=4, h_q=32, h_kv=8, s_q=128, s_kv=8192, d_qk=64, d_v=64, causal=True, bottom_right=True, device_cc=(8, 0))
        base.update(page_size=16, padded=True, thd=True, max_total_seq_len_q=4 * 128, has_sink=sink)
        base.update(over)
        return _facts(**base)

    moved = f()
    assert 31.0 <= pt.FEATURES["cost"](moved) < 33.792
    assert pt.matched_entry(row, moved).study == study
    assert pt.lookup(row, moved) is True
    assert pt.lookup(row, f(b=1, h_q=8, h_kv=2, s_q=32, s_kv=1024, max_total_seq_len_q=32)) is False
    assert pt.matched_entry(row, f(s_kv=8193)).study == "sm80_prefill_gaps_20261117"


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm80_paged_d192x192_prefill_follows_its_study(s_kv):
    """sm80_paged_d192x192_v2_20261117 (A100): paged THD-Q d192 x 192 causal prefill leads past a cost floor with enough query tokens
    (step5 64/4 at S_q 512-1024 over 128k ran the backend at 1.21-1.42x); short-cache squares and the 512-key window keep the backend."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_paged_d192x192_v2_20261117"

    def f(**over):
        base = dict(b=1, h_q=64, h_kv=4, s_q=1024, s_kv=s_kv, d_qk=192, d_v=192, causal=True, bottom_right=True, device_cc=(8, 0))
        base.update(page_size=16, padded=True, thd=True, max_total_seq_len_q=1024)
        base.update(over)
        return _facts(**base)

    assert pt.matched_entry(row, f()).study == study
    assert pt.lookup(row, f()) is True
    square = f(h_q=8, h_kv=4, s_q=8192, s_kv=8192 + s_kv % 2, max_total_seq_len_q=8192)
    assert pt.lookup(row, square) is False
    assert pt.lookup(row, f(s_q=17, max_total_seq_len_q=17)) is False
    assert pt.matched_entry(row, f(window_left=511)) is None


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
@pytest.mark.parametrize("d", [(256, 256), (192, 128)], ids=["d256", "d192x128"])
def test_sm80_paged_d256_d192_prefill_follow_v3(d, s_kv):
    """sm80_paged_d256_d192_v3_20261117 (A100): paged THD-Q d256 and d192 x 128 causal prefill lead past a floor only with enough query
    tokens, so few-query chunks over a long cache with many heads (the edge that retired the earlier floors, 1.30-1.47x) keep the
    backend."""
    row = "sdpa_fwd_prefill_sm80"

    def f(**over):
        base = dict(b=4, h_q=32, h_kv=4, s_q=2048, s_kv=s_kv, d_qk=d[0], d_v=d[1], causal=True, bottom_right=True, device_cc=(8, 0))
        base.update(page_size=16, padded=True, thd=True, max_total_seq_len_q=4 * 2048)
        base.update(over)
        return _facts(**base)

    on_tile_owner = {(256, 256): "sm80_paged_d256_v7_pcie_20261118", (192, 128): "sm80_paged_d192x128_pcie_20261121"}[d]
    assert pt.matched_entry(row, f()).study == (on_tile_owner if s_kv % 128 == 0 else "sm80_paged_d256_d192_v3_20261117")
    assert pt.lookup(row, f()) is True
    edge = f(b=32, h_q=64, h_kv=8, s_q=32, max_total_seq_len_q=32 * 32)
    assert pt.lookup(row, edge) is False


@pytest.mark.L0
def test_sm80_paged_d256_on_tile_floor_follows_v7():
    """sm80_paged_d256_v7_pcie_20261118 (A100 80GB PCIe): on the KV tile paged THD-Q d256 causal prefill leads from cost 2^35.357, so
    qwen3.5's b 1 long-cache chunks at cost 2^36.0 (backend 1.14-1.47x) lead; off the tile v3's 2^36.454 floor stays, which keeps the
    single-KV-head chunk a lower floor lost on (FROST 1.33x) with the backend."""
    row = "sdpa_fwd_prefill_sm80"

    def f(**over):
        base = dict(b=1, h_q=16, h_kv=1, s_q=512, s_kv=131072, d_qk=256, d_v=256, causal=True, bottom_right=True, device_cc=(8, 0))
        base.update(page_size=16, padded=True, thd=True, max_total_seq_len_q=512)
        base.update(over)
        return _facts(**base)

    assert pt.FEATURES["cost"](f()) == 36.0
    assert pt.matched_entry(row, f()).study == "sm80_paged_d256_v7_pcie_20261118"
    assert pt.lookup(row, f()) is True
    off = f(h_q=10, s_kv=131073)
    assert pt.matched_entry(row, off).study == "sm80_paged_d256_d192_v3_20261117"
    assert pt.lookup(row, off) is False


@pytest.mark.L0
@pytest.mark.parametrize("off_tile", [0, 1], ids=["on_tile", "off_tile"])
def test_sm80_d512_decode_follows_its_study(off_tile):
    """A100 d512 decode (dense and paged, single-row and MTP, 1-8 KV heads) belongs to study sm80_d512_decode_v3_pcie_20261121 on and
    off the KV tile. Within each class the lead is monotone in the entry's feature, and `cost` grows with h_kv."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_d512_decode_v3_pcie_20261121"

    def f(b, h_q, h_kv, s_q, s_kv, paged):
        causal = s_q > 1
        return _facts(
            b=b,
            h_q=h_q,
            h_kv=h_kv,
            s_q=s_q,
            s_kv=s_kv + off_tile,
            d_qk=512,
            d_v=512,
            causal=causal,
            bottom_right=causal,
            has_paged_kv=paged,
            page_size=64 if paged else 0,
            padded=paged,
            device_cc=(8, 0),
        )

    classes = {}
    grid = [
        (b, h_q, h_kv, s_q, s_kv, paged)
        for paged in (False, True)
        for s_q in (1, 2, 4)
        for h_q, h_kv in ((16, 1), (128, 1), (16, 2), (32, 4), (32, 8))
        for b in (1, 4, 64)
        for s_kv in (128, 1024, 8192, 131072)
    ]
    for b, h_q, h_kv, s_q, s_kv, paged in grid:
        facts = f(b, h_q, h_kv, s_q, s_kv, paged)
        key = pt.class_key(row, facts)
        entry = pt.matched_entry(row, facts)
        assert key[-1] == bool(off_tile) and entry.study == study, (b, h_q, h_kv, s_q, s_kv, paged)
        classes.setdefault(key, []).append((pt.FEATURES[entry.feature](facts), pt.lookup(row, facts)))
    assert len(classes) == 4  # dense / paged x single-row / MTP
    for key, points in classes.items():
        leads = [lead for _, lead in sorted(points)]
        assert leads == sorted(leads), key  # the backend below the floor, FROST from it
    assert pt.FEATURES["cost"](f(1, 32, 8, 2, 4096, True)) > pt.FEATURES["cost"](f(1, 32, 1, 2, 4096, True))


@pytest.mark.L0
def test_sm107_dense_d128_noncausal_mha_follows_its_study():
    """sm107_dense_d128_nc_mha_v2_20261117 (Rubin): dense d128 MHA non-causal leads up to 2^14.246 query rows on the KV tile, which hands
    the hand rule's 256-tile S_q 256 chunks (FROST 1.36-1.53x) back to the backend; off the tile it leads cost <= 2^32.953 over at least
    2^10.501 keys."""
    row = "sdpa_fwd_prefill_sm107"
    study = "sm107_dense_d128_nc_mha_v2_20261117"

    def f(**over):
        base = dict(b=1, h_q=16, h_kv=16, s_q=512, s_kv=4096, d_qk=128, d_v=128, causal=False, bottom_right=False, device_cc=(10, 7))
        base.update(has_paged_kv=False)
        base.update(over)
        return _facts(**base)

    assert pt.matched_entry(row, f()).study == study
    assert pt.lookup(row, f()) is True
    assert pt.lookup(row, f(b=8, s_q=256, s_kv=32768)) is False
    off = f(h_q=8, h_kv=8, s_q=128, s_kv=16385)
    assert pt.FEATURES["cost"](off) <= 32.953
    assert pt.matched_entry(row, off).study == study and pt.lookup(row, off) is True
    assert pt.lookup(row, f(h_q=8, h_kv=8, s_q=128, s_kv=1025)) is False


@pytest.mark.L0
def test_two_condition_entry_leads_only_where_both_hold():
    """``Entry.also``: on the KV tile sm80_dense_d128_gqa_v4_split_20261117 leads A100 dense d128 GQA causal prefill only with few
    launch tiles AND a long cache; off the tile one condition on query rows answers."""
    row = "sdpa_fwd_prefill_sm80"

    def f(**over):
        base = dict(b=1, h_q=8, h_kv=1, s_q=256, s_kv=131072, d_qk=128, d_v=128, causal=True, bottom_right=True)
        base.update(device_cc=(8, 0), has_paged_kv=False)
        base.update(over)
        return _facts(**base)

    entry = pt.matched_entry(row, f())
    assert entry.study == "sm80_dense_d128_gqa_v4_split_20261117" and entry.feature == "tiles" and entry.also == ("ge", 10.5, "kv")
    assert pt.lookup(row, f()) is True
    assert pt.lookup(row, f(s_kv=1024)) is False  # few tiles, short cache
    assert pt.lookup(row, f(b=8, h_q=64, h_kv=8, s_q=2048)) is False  # many tiles, long cache
    off = pt.matched_entry(row, f(s_kv=131073))
    assert off.feature == "rows" and off.also is None
    assert pt.lookup(row, f(s_kv=1025)) is True  # few rows lead off the tile whatever the cache


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm80_dense_chunk_follows_its_study(s_kv):
    """Study sm80_dense_chunk_20261117 (A100): dense d256 GQA causal prefill leads past a cost floor, so qwen3.5's b 1 long-KV chunks
    (FROST 2.3-3.0x faster) lead and small squares keep the backend; d64 MHA causal leads past a floor off the tile. d128 GQA answers
    from its own re-study."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_dense_chunk_20261117"

    def f(**over):
        base = dict(b=1, h_q=4, h_kv=1, s_q=512, s_kv=s_kv, d_qk=256, d_v=256, causal=True, bottom_right=True)
        base.update(device_cc=(8, 0), has_paged_kv=False)
        base.update(over)
        return _facts(**base)

    assert pt.matched_entry(row, f()).study == study
    assert pt.lookup(row, f()) is True
    assert pt.lookup(row, f(s_q=128, s_kv=128)) is False  # a small square
    mha64 = dict(d_qk=64, d_v=64, h_q=16, h_kv=16)
    assert pt.matched_entry(row, f(**mha64)).study == study
    assert pt.lookup(row, f(**mha64)) is True
    assert pt.matched_entry(row, f(d_qk=128, d_v=128, h_q=8, h_kv=1)).study == "sm80_dense_d128_gqa_v4_split_20261117"  # its own re-study


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
    """Study sm120_decode_20261112 (RTX PRO 6000): paged d256 single-row decode keeps the backend, dense D576 single-row decode
    leads; d512 s_q 1 and paged D576 s_q 1 have no entry (hand rules decide: #2497, sm120_paged_d512_q1_v2_20261117 and its D576
    transfer check); sink-only d64 MTP has no entry."""
    row = "sdpa_fwd_prefill_sm120"

    def f(**over):
        base = dict(b=8, h_q=32, h_kv=2, s_q=1, s_kv=32768, d_qk=256, d_v=256, page_size=16, padded=True, device_cc=(12, 0), device_sm_count=188)
        base.update(over)
        return _facts(**base)

    for s_kv in (32768, 32767):
        assert pt.lookup(row, f(s_kv=s_kv)) is False
        assert pt.matched_entry(row, f(s_kv=s_kv)).study == "sm120_decode_20261112"
    assert pt.lookup(row, f(h_q=96, h_kv=1, d_qk=576, d_v=512, page_size=32)) is None
    assert pt.lookup(row, f(h_q=96, h_kv=1, d_qk=576, d_v=512, has_paged_kv=False, padded=False, page_size=0)) is True
    assert pt.lookup(row, f(b=64, h_q=64, h_kv=1, d_qk=512, d_v=512, page_size=64, s_kv=131072)) is None
    assert pt.lookup(row, f(b=8, h_q=64, h_kv=1, d_qk=512, d_v=512, page_size=64, s_kv=131071)) is None
    assert pt.lookup(row, f(h_q=128, h_kv=1, d_qk=512, d_v=512, has_paged_kv=False, padded=False, page_size=0)) is None
    sink = dict(h_q=64, h_kv=8, d_qk=64, d_v=64, s_q=2, has_sink=True, causal=True, bottom_right=True)
    assert pt.lookup(row, f(**sink)) is None
    assert pt.matched_entry(row, f(window_left=127, **sink)).study == "sm120_decode_20261112"


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [32768, 32769], ids=["on_tile", "off_tile"])
def test_sm120_paged_d128_body_follows_its_study(s_kv):
    """Paged d128 at s_q 2-16 past 32 packed rows runs the prefill body, not the decode tile: its own group (`gqa_body`) and study
    (sm120_paged_d128_body_20261117). It leads from enough query rows in flight and trails one request; decode-tile graphs, THD Q,
    s_q past 16 and other rows keep `gqa`."""
    row = "sdpa_fwd_prefill_sm120"

    def paged(**over):
        base = dict(b=16, h_q=64, h_kv=4, s_q=4, s_kv=s_kv, d_qk=128, d_v=128, causal=True, bottom_right=True)
        base.update(device_cc=(12, 0), page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    off = s_kv % 128 != 0
    for s_q, phase in ((4, "mtp"), (8, "prefill"), (16, "prefill")):
        assert pt.class_key(row, paged(s_q=s_q)) == (row, (12, 0), "bf16", (128, 128), phase, "paged", "causal_br", False, "gqa_body", off)
        assert pt.class_key(row, paged(s_q=s_q, window_left=1023))[6:9] == ("window", False, "gqa_body")
        assert pt.matched_entry(row, paged(s_q=s_q)).study == "sm120_paged_d128_body_20261117"
        assert pt.lookup(row, paged(s_q=s_q)) is True  # 16 requests x 64 heads
        assert pt.lookup(row, paged(b=1, h_q=16, h_kv=1, s_q=s_q)) is False  # one request, 16 heads
    assert pt.class_key(row, paged(h_q=32, h_kv=4, s_q=4))[8] == "gqa"  # 4 x group 8 = 32 rows: the decode tile
    assert pt.class_key(row, paged(s_q=32))[8] == "gqa"
    for other in (paged(thd=True, max_total_seq_len_q=64), paged(d_qk=64, d_v=64, h_q=128), paged(has_paged_kv=False, page_size=0)):
        assert pt.class_key(row, other)[8] != "gqa_body"
    assert pt.class_key("sdpa_fwd_prefill_sm100", paged(device_cc=(10, 0)))[8] != "gqa_body"


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131071], ids=["on_tile", "off_tile"])
def test_sm120_paged_d128_mtp_follows_its_study(s_kv):
    """Study sm120_paged_d128_mtp_v6_20261117 retired the d128 paged MTP entries: the GQA group decides there, which the class key
    cannot carry (fwd/placement._place_sm120_f16; test_sdpa_fwd_placement.test_sm120_paged_d128_mtp_follows_the_group)."""
    row = "sdpa_fwd_prefill_sm120"

    def mtp(**over):
        base = dict(b=1, h_q=32, h_kv=4, s_q=2, s_kv=s_kv, d_qk=128, d_v=128, causal=True, bottom_right=True)
        base.update(device_cc=(12, 0), page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for b, h_q, h_kv in ((1, 32, 4), (1, 64, 8), (16, 32, 4), (128, 64, 8), (32, 8, 1)):
        assert pt.class_key(row, mtp(b=b, h_q=h_q, h_kv=h_kv))[4:9] == ("mtp", "paged", "causal_br", False, "gqa")
        assert pt.lookup(row, mtp(b=b, h_q=h_q, h_kv=h_kv)) is None, (b, h_q, h_kv)


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131071], ids=["on_tile", "off_tile"])
def test_sm120_paged_window_decode_follows_its_study(s_kv):
    """Study sm120_paged_window_decode_20261117: paged sliding-window decode at s_q 1 leads on few KV-head units at d64 / d128 and
    always at d256 (gemma4 31B SWA at TP4 and muse_spark SWA ran the backend at 2.3-4.4x); an unwindowed graph is another class."""
    row = "sdpa_fwd_prefill_sm120"

    def dec(**over):
        base = dict(b=1, h_q=8, h_kv=4, s_q=1, s_kv=s_kv, d_qk=256, d_v=256, causal=True, bottom_right=True, window_left=1023)
        base.update(device_cc=(12, 0), page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for over in (dict(), dict(b=64), dict(h_q=16, h_kv=2, d_qk=64, d_v=64, window_left=2047), dict(h_q=32, h_kv=4, d_qk=128, d_v=128)):
        assert pt.lookup(row, dec(**over)) is True, over
        assert pt.matched_entry(row, dec(**over)).study == "sm120_paged_window_decode_20261117"
    assert pt.lookup(row, dec(b=128, h_q=64, h_kv=8, d_qk=128, d_v=128)) is False  # 1024 KV-head units
    unwindowed = pt.matched_entry(row, dec(window_left=None))
    assert unwindowed is None or unwindowed.study != "sm120_paged_window_decode_20261117"  # another class: not this study's
    # d192 x V128: its own study, groups 2-16 (sm120_paged_window_decode_d192_v2_20261117)
    wide_k = dict(h_q=64, h_kv=4, d_qk=192, d_v=128, window_left=511)
    assert pt.lookup(row, dec(**wide_k)) is True
    assert pt.matched_entry(row, dec(**wide_k)).study == "sm120_paged_window_decode_d192_v2_20261117"
    assert pt.lookup(row, dec(**dict(wide_k, b=32, h_q=96, h_kv=8))) is False  # 256 KV-head units, group 12


@pytest.mark.L0
@pytest.mark.parametrize("s_q, s_kv", [(8192, 8192), (1024, 131072), (1024, 131071)], ids=["square", "chunk", "chunk_off_tile"])
def test_sm120_windowed_d256_d512_prefill_leads(s_q, s_kv):
    """Study sm120_window_prefill_20261117: dense windowed d512 prefill leads on SM120, squares included (the hand rule trails every
    windowed square); d256 follows its re-study (lead from KV 2^7.5); d128, the d64 sink class, f16 and another part keep the hand rules."""
    row = "sdpa_fwd_prefill_sm120"

    def win(**over):
        base = dict(b=2, h_q=128, h_kv=1, s_q=s_q, s_kv=s_kv, d_qk=512, d_v=512, causal=True, bottom_right=True, window_left=128)
        base.update(has_paged_kv=False, device_cc=(12, 0))
        base.update(over)
        return _facts(**base)

    for over in (dict(), dict(h_q=64)):
        assert pt.lookup(row, win(**over)) is True, over
        assert pt.matched_entry(row, win(**over)).study == "sm120_window_prefill_20261117"
    d256 = dict(h_q=16, h_kv=8, d_qk=256, d_v=256, window_left=1024)
    assert pt.matched_entry(row, win(**d256)).study == "sm120_window_d256_v2_20261117"  # re-studied over S_q 5-2048
    assert pt.lookup(row, win(**d256, s_q=33, s_kv=33, b=16)) is False  # the square that lost 1.125 in edge_supp_20261117
    assert pt.lookup(row, win(h_q=32, h_kv=8, d_qk=128, d_v=128, window_left=1024)) is None
    assert pt.lookup(row, win(h_q=64, h_kv=8, d_qk=64, d_v=64, has_sink=True)) is None
    assert pt.lookup(row, win(dtype=cudnn.data_type.HALF)) is None
    assert pt.lookup(row, win(device_cc=(12, 1))) is None


@pytest.mark.L0
@pytest.mark.parametrize("s", [8192, 3000], ids=["on_tile", "off_tile"])
def test_sm120_deterministic_backward_follows_its_study(s):
    """Study sm120_det_bwd_20261117: deterministic SM120 backward graphs are classes of their own (phase prefill_det) and lead where
    the holdout exercised them (gpt-oss d64 SWA ran engine 7 at 9.8x); non-deterministic graphs keep their study, the unexercised
    pairs keep the hand rule, and on other rows a deterministic graph stays unmeasured."""
    row = "sdpa_bwd_sm120"

    def bwd(**over):
        base = dict(b=2, h_q=64, h_kv=64, s_q=s, s_kv=s, d_qk=64, d_v=64, causal=True, bottom_right=False, window_left=128)
        base.update(has_paged_kv=False, device_cc=(12, 0), deterministic=True)
        base.update(over)
        return _facts(**base)

    assert pt.class_key(row, bwd())[4] == "prefill_det"
    assert pt.lookup(row, bwd()) is True
    assert pt.matched_entry(row, bwd()).study == "sm120_det_bwd_20261117"
    assert pt.matched_entry(row, bwd(deterministic=False)).study == "sm120_bwd_bf16_20261107"
    d128 = dict(d_qk=128, d_v=128, h_q=32, h_kv=8)
    assert pt.lookup(row, bwd(s_q=1024, s_kv=1024, **d128)) is False  # the d128 window classes lead from S 2048
    assert pt.lookup(row, bwd(s_q=2048, s_kv=2048, **d128)) is True
    assert pt.lookup(row, bwd(d_qk=128, d_v=128, window_left=None)) is None  # d128 causal MHA: no holdout cell
    assert pt.class_key("sdpa_bwd_sm89", bwd(device_cc=(8, 9))) is None  # an unkeyed row: deterministic stays unmeasured


@pytest.mark.L0
@pytest.mark.parametrize("s", [8192, 5000], ids=["on_tile", "off_tile"])
def test_sm80_deterministic_backward_follows_its_study(s):
    """Study sm80_det_bwd_20261117: on A100 deterministic backward is keyed like SM120's; windowed d128 and windowed d64 GQA lead from
    S 4096, unwindowed classes keep the hand rule (measured `never`). d64 windowed MHA (gpt-oss) answers from its own study
    (sm80_det_bwd_d64mha_v3_20261117) from S ~3.5k on the tile, ~3.9k off it. Windows of up to ~384 keys key apart on this row
    (sm80_det_bwd_narrow_window_20261117) and lead from S ~900, except d128 MHA, which keeps its `window` rule."""
    row = "sdpa_bwd_sm80"

    def bwd(**over):
        base = dict(b=2, h_q=32, h_kv=8, s_q=s, s_kv=s, d_qk=128, d_v=128, causal=True, bottom_right=False, window_left=1023)
        base.update(has_paged_kv=False, device_cc=(8, 0), deterministic=True)
        base.update(over)
        return _facts(**base)

    assert pt.class_key(row, bwd())[4] == "prefill_det"
    for over in (dict(), dict(d_qk=64, d_v=64)):
        assert pt.lookup(row, bwd(**over)) is True, over
        assert pt.matched_entry(row, bwd(**over)).study == "sm80_det_bwd_20261117"
    assert pt.lookup(row, bwd(s_q=2048, s_kv=2048)) is False
    assert pt.lookup(row, bwd(window_left=None)) is None
    d64_mha = dict(h_q=64, h_kv=64, d_qk=64, d_v=64)
    assert pt.lookup(row, bwd(**d64_mha)) is True
    assert pt.matched_entry(row, bwd(**d64_mha)).study == "sm80_det_bwd_d64mha_v3_20261117"
    assert pt.lookup(row, bwd(s_q=3000, s_kv=3000, **d64_mha)) is False
    narrow = bwd(s_q=2048, s_kv=2048, window_left=127, **d64_mha)  # gpt-oss: engine 7 at 2.61x before the narrow class
    assert pt.class_key(row, narrow)[6] == "window_narrow"
    assert pt.lookup(row, narrow) is True
    assert pt.matched_entry(row, narrow).study == "sm80_det_bwd_narrow_window_20261117"
    assert pt.lookup(row, bwd(s_q=512, s_kv=512, window_left=127, **d64_mha)) is False
    assert pt.matched_entry(row, bwd(window_left=127, h_q=64, h_kv=64)).study == "sm80_det_bwd_20261117"  # d128 MHA keeps its rule
    assert pt.class_key("sdpa_bwd_sm120", bwd(window_left=127, device_cc=(12, 0)))[6] == "window"  # row-local


@pytest.mark.L0
@pytest.mark.parametrize("s", [8192, 8100], ids=["on_tile", "off_tile"])
def test_sm80_thd_d256_prefill_follows_its_v2_study(s):
    """Study sm80_thd_d256_v2_20261117 (A100) replaced sm80_thd_prefill_20261117, whose `always` entries lost below S_q 500: ragged
    d256 THD causal prefill leads past a floor (S_q 2^7.5 on the tile, 2^9.09 tiles off it). Equal lengths are keyed apart and belong to
    sm80_train_fwd_gaps_pcie_20261123, which leads this wide grid."""
    from cudnn.sdpa.fwd import placement
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    row = "sdpa_fwd_prefill_sm80"

    def thd(total_fraction, **over):
        base = dict(b=64, h_q=8, h_kv=2, s_q=s, s_kv=s, d_qk=256, d_v=256, causal=True, bottom_right=False, has_paged_kv=False, thd=True)
        base.update(max_total_seq_len_q=int(base["b"] * s * total_fraction), device_cc=(8, 0), device_sm_count=108)
        base.update(over)
        return _facts(**base)

    assert pt.matched_entry(row, thd(0.75)).study == "sm80_thd_d256_v2_20261117"
    assert pt.lookup(row, thd(0.75)) is True
    assert pt.matched_entry(row, thd(1.0)).study == "sm80_train_fwd_gaps_pcie_20261123"
    assert pt.lookup(row, thd(1.0)) is True
    d256 = [spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_sm80_d256_kv_prefetch"]
    assert d256 and placement.place(d256[0], thd(0.75)) == placement.LEAD
    assert placement.place(d256[0], thd(1.0)) == placement.LEAD


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm80_d576_mla_prefill_follows_its_study(s_kv):
    """Study sm80_d576_prefill_20261117 (A100): absorbed-MLA (576, 512) dense prefill and verify lead (kimi_k3's prefill ran the
    backend at up to 2.61x); paged prefill leads past a tile floor (on the tile from sm80_d576_paged_v3_20261117, which leads
    kimi_k3-tp8's long-KV chunks); paged verify keeps the wide-verify study's entries."""
    row = "sdpa_fwd_prefill_sm80"
    study = "sm80_d576_prefill_20261117"

    def f(paged, **over):
        base = dict(b=1, h_q=96, h_kv=1, s_q=1024, s_kv=s_kv, d_qk=576, d_v=512, causal=True, bottom_right=True, device_cc=(8, 0))
        base.update(has_paged_kv=paged, padded=paged, page_size=64 if paged else 0, thd=paged, max_total_seq_len_q=1024 if paged else None)
        base.update(over)
        return _facts(**base)

    for paged in (False, True):
        assert pt.class_key(row, f(paged))[4] == "prefill"
        assert pt.matched_entry(row, f(paged)).study == (study if not paged or s_kv % 2 else "sm80_d576_paged_v3_20261117")
        assert pt.lookup(row, f(paged)) is True
    assert pt.lookup(row, f(True, h_q=12, s_q=512, max_total_seq_len_q=512)) is True  # kimi_k3-tp8 (2^5.6 tiles)
    assert pt.lookup(row, f(False, s_q=8)) is True  # dense verify
    assert pt.matched_entry(row, f(False, s_q=8)).study == study
    assert pt.matched_entry(row, f(True, s_q=8, max_total_seq_len_q=8)).study == "sm80_paged_wide_verify_v2_20261117"
    assert pt.lookup(row, f(True, b=1, h_q=8, s_q=32, s_kv=32 + (s_kv % 2), max_total_seq_len_q=32)) is False  # few tiles


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
    assert pt.TABLE[pt.class_key(row, bf16)].study == "sm80_dense_chunk_20261117"  # a bf16 study keeps the bf16 class
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
@pytest.mark.parametrize(
    "d, heads, s_qs",
    [((576, 512), (64, 1), (5, 8, 16, 1024)), ((512, 512), (32, 4), (5, 8, 16, 1024)), ((192, 192), (64, 4), (2, 4, 5, 8, 1024))],
    ids=["d576", "d512", "d192"],
)
def test_sm107_paged_wide_head_verify_entries_lead(d, heads, s_qs):
    """Study sm107_paged_wide_gaps_20261118 (Rubin, the inference benchmark's paged generation graph): paged D576 MLA, d512 and d192x192
    GQA under the bottom-right diagonal lead at every measured S_q on and off the tile. Verify rows (S_q 5-16) key `prefill` on SM107.
    Without the diagonal, f16, and another part keep the hand rules."""
    row = "sdpa_fwd_prefill_sm107"

    def paged(**over):
        base = dict(b=16, h_q=heads[0], h_kv=heads[1], s_q=8, s_kv=131072, d_qk=d[0], d_v=d[1], causal=True, bottom_right=True)
        base.update(device_cc=(10, 7), has_paged_kv=True, page_size=64, padded=True)
        base.update(over)
        return _facts(**base)

    for s_q in s_qs:
        for s_kv in (8192, 131072, 131073):
            assert pt.lookup(row, paged(s_q=s_q, s_kv=s_kv)) is True, (s_q, s_kv)
            assert pt.matched_entry(row, paged(s_q=s_q, s_kv=s_kv)).study == "sm107_paged_wide_gaps_20261118"
    assert pt.lookup(row, paged(causal=False, bottom_right=False)) is None
    assert pt.lookup(row, paged(dtype=cudnn.data_type.HALF)) is None
    assert pt.lookup(row, paged(device_cc=(10, 3))) is None  # no transfer names this study


@pytest.mark.L0
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3)])
def test_sm100_paged_d192_prefill_entries_follow_the_study(device_cc):
    """Study sm100_prefill_d192_gaps_20261118 (B200, B300 transfer): paged THD d192 x 192 context (step5) leads with the full cache and
    with a window, on and off the tile, on cc 10.0 and 10.3. The mimo d192 x 128 classes failed their holdout there; paged d192 x 128
    now has its own study (sm100_d192x128_paged_v2_20261118)."""
    row = "sdpa_fwd_prefill_sm100"

    def paged(**over):
        base = dict(b=1, h_q=64, h_kv=4, s_q=8192, s_kv=8192, d_qk=192, d_v=192, causal=True, bottom_right=True, thd=True)
        base.update(device_cc=device_cc, has_paged_kv=True, page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for over in (dict(), dict(s_q=512, s_kv=131072), dict(s_q=512, s_kv=131073), dict(window_left=511), dict(window_left=511, s_kv=131073, s_q=2048)):
        assert pt.lookup(row, paged(**over)) is True, over
        assert pt.matched_entry(row, paged(**over)).study == "sm100_prefill_d192_gaps_20261118"
    assert pt.matched_entry(row, paged(d_v=128)).study == "sm100_d192x128_paged_v2_20261118"  # mimo d192 x 128: its own v2 study
    assert pt.lookup(row, paged(dtype=cudnn.data_type.HALF)) is None


@pytest.mark.L0
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3)])
def test_sm100_windowed_d128_prefill_keys_unpacked_groups_apart(device_cc):
    """Study sm100_d128_window_prefill_unpacked_20261118: SM100 / SM103 windowed paged d128 prefill whose GQA group the row does not
    pack whole keys `gqa_unpacked` and leads on the tile only up to 2k keys, off the tile as before. Group 8 packs whole and keeps the
    swa_prefill_20261031 entries; dense caches keep their key."""
    row = "sdpa_fwd_prefill_sm100"

    def paged(**over):
        base = dict(b=1, h_q=72, h_kv=8, s_q=1024, s_kv=131072, d_qk=128, d_v=128, causal=True, bottom_right=True, thd=True, window_left=511)
        base.update(device_cc=device_cc, has_paged_kv=True, page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for h_q, h_kv in ((72, 8), (80, 8), (10, 1), (48, 8), (24, 2)):  # groups 9 / 10 / 10 / 6 / 12
        facts = paged(h_q=h_q, h_kv=h_kv)
        assert pt.class_key(row, facts)[8] == "gqa_unpacked", (h_q, h_kv)
        assert pt.lookup(row, facts) is False, (h_q, h_kv)
        assert pt.matched_entry(row, facts).study == "sm100_d128_window_prefill_unpacked_20261118"
        assert pt.lookup(row, paged(h_q=h_q, h_kv=h_kv, s_q=2048, s_kv=2048)) is True, (h_q, h_kv)
        assert pt.lookup(row, paged(h_q=h_q, h_kv=h_kv, s_kv=131073)) is True, (h_q, h_kv)
    packed = paged(h_q=64, h_kv=8)
    assert pt.class_key(row, packed)[8] == "gqa"
    assert pt.lookup(row, packed) is True
    assert pt.matched_entry(row, packed).study.endswith("_swa_prefill_20261031")
    for verify in (paged(h_q=80, h_kv=8, s_q=8, thd=False), paged(h_q=72, h_kv=8, s_q=255)):  # below the measured S_q: as before
        assert pt.class_key(row, verify)[8] == "gqa"
        assert pt.matched_entry(row, verify).study.endswith("_swa_prefill_20261031")
    assert pt.class_key(row, paged(has_paged_kv=False, padded=False, page_size=0))[8] == "gqa"


@pytest.mark.L0
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3)])
def test_sm100_dense_d192x128_window_sink_prefill_entries_follow_the_study(device_cc):
    """Study sm100_d192x128_swa_sink_pack_20261118: mimo swa dense d192 x 128 context (128-key window, sink) keys `gqa` when the row
    packs the group whole and `gqa_unpacked` otherwise; on the tile each leads past its tiles floor (the b 1 2048-square 16-head rows,
    1.27-1.31 on GB200, stay with the backend), off the tile both lead. Paged caches keep their key."""
    row = "sdpa_fwd_prefill_sm100"

    def dense(**over):
        base = dict(b=1, h_q=64, h_kv=8, s_q=2048, s_kv=2048, d_qk=192, d_v=128, causal=True, bottom_right=True, window_left=127, has_sink=True)
        base.update(device_cc=device_cc, has_paged_kv=False)
        base.update(over)
        return _facts(**base)

    for over, key, lead in (
        (dict(), "gqa", True),
        (dict(h_q=128), "gqa", True),
        (dict(h_q=16, h_kv=1), "gqa", False),
        (dict(h_q=16, h_kv=2), "gqa", False),
        (dict(h_q=16, h_kv=1, s_q=8192, s_kv=8192), "gqa", True),
        (dict(h_q=16, h_kv=1, s_kv=2049), "gqa", True),
        (dict(h_q=48), "gqa_unpacked", False),
        (dict(h_q=96, s_q=16384, s_kv=16384, b=2), "gqa_unpacked", True),
        (dict(h_q=48, s_kv=2049), "gqa_unpacked", True),
    ):
        facts = dense(**over)
        assert pt.class_key(row, facts)[8] == key, over
        assert pt.lookup(row, facts) is lead, over
        assert pt.matched_entry(row, facts).study == "sm100_d192x128_swa_sink_pack_20261118"
    assert pt.class_key(row, dense(has_paged_kv=True, page_size=16, padded=True, thd=True, h_q=48))[8] == "gqa"
    for over in (dict(h_q=16, h_kv=1, s_q=8, s_kv=131072, b=128), dict(h_q=48, s_q=255)):  # below the measured S_q: the hand rules
        assert pt.class_key(row, dense(**over))[8] == "gqa_short", over
        assert pt.lookup(row, dense(**over)) is None, over


@pytest.mark.L0
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3)])
def test_sm100_paged_d192x128_context_entries_follow_the_study(device_cc):
    """Study sm100_d192x128_paged_v2_20261118: mimo full paged d192 x 128 context (THD Q) leads on the tile from S_q ~887 at enough
    work, off the tile from S_q ~362; graphs below S_q 256 (spec-decode verify included) key `gqa_short` and keep the hand rules."""
    row = "sdpa_fwd_prefill_sm100"

    def paged(**over):
        base = dict(b=1, h_q=64, h_kv=4, s_q=8192, s_kv=8192, d_qk=192, d_v=128, causal=True, bottom_right=True, thd=True)
        base.update(device_cc=device_cc, has_paged_kv=True, page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for over, lead in (
        (dict(), True),
        (dict(s_q=1024, s_kv=131072), True),
        (dict(h_q=128, h_kv=8, s_q=32768, s_kv=32768), True),
        (dict(s_q=512, s_kv=131072), False),
        (dict(s_q=768, s_kv=768), False),
        (dict(s_q=512, s_kv=131073), True),
    ):
        facts = paged(**over)
        assert pt.class_key(row, facts)[8] == "gqa", over
        assert pt.lookup(row, facts) is lead, over
        assert pt.matched_entry(row, facts).study == "sm100_d192x128_paged_v2_20261118"
    for short in (paged(s_q=8, s_kv=131072, thd=False), paged(s_q=255, s_kv=16384)):
        assert pt.class_key(row, short)[8] == "gqa_short"
        assert pt.lookup(row, short) is None


@pytest.mark.L0
def test_sm90_d192x128_verify_entries_follow_the_study():
    """Study sm90_d192x128_verify_20261118 (H100): mimo d192 x 128 spec-decode verify. Dense full cache leads on the KV tile past a cost
    floor and always off it; paged leads with the full cache and with a window + sink on and off the tile; dense window + sink failed
    its holdout here (the v2 study answers it)."""
    row = "sdpa_fwd_prefill_sm90"

    def g(**over):
        base = dict(b=128, h_q=64, h_kv=4, s_q=8, s_kv=131072, d_qk=192, d_v=128, causal=True, bottom_right=True, device_cc=(9, 0), has_paged_kv=False)
        base.update(over)
        return _facts(**base)

    study = "sm90_d192x128_verify_20261118"
    assert pt.class_key(row, g())[4] == "verify"
    assert pt.lookup(row, g()) is True and pt.matched_entry(row, g()).study == study
    assert pt.lookup(row, g(b=1, h_q=16, h_kv=1, s_q=5, s_kv=2048)) is False  # below the on-tile dense cost floor
    assert pt.lookup(row, g(b=1, h_q=16, h_kv=1, s_q=5, s_kv=2049)) is True  # off the tile: always
    paged = dict(has_paged_kv=True, page_size=16, padded=True)
    for kv in (131072, 131073):
        assert pt.lookup(row, g(**paged, b=1, h_q=16, h_kv=1, s_kv=kv)) is True
        assert pt.matched_entry(row, g(**paged, window_left=127, has_sink=True, b=1, h_q=16, h_kv=2, s_kv=kv)).study == study
    dense_sink = pt.matched_entry(row, g(window_left=127, has_sink=True, h_q=64, h_kv=8))
    assert dense_sink is None or dense_sink.study != study


@pytest.mark.L0
def test_sm90_d192x128_dense_sink_verify_v2_entries_follow_the_study():
    """Study sm90_d192x128_dense_sink_verify_v2_20261118 (H100): mimo dense d192 x 128 verify with a window and a sink leads from 2^4.16
    parallel tiles on the KV tile and 2^4.40 off it."""
    row = "sdpa_fwd_prefill_sm90"

    def g(**over):
        base = dict(b=128, h_q=128, h_kv=8, s_q=8, s_kv=131072, d_qk=192, d_v=128, causal=True, bottom_right=True, device_cc=(9, 0))
        base.update(window_left=127, has_sink=True, has_paged_kv=False)
        base.update(over)
        return _facts(**base)

    study = "sm90_d192x128_dense_sink_verify_v2_20261118"
    for kv in (131072, 131073):
        assert pt.class_key(row, g(s_kv=kv))[4:7] == ("verify", "dense", "window")
        assert pt.lookup(row, g(s_kv=kv)) is True and pt.matched_entry(row, g(s_kv=kv)).study == study
        assert pt.lookup(row, g(s_kv=kv, b=1)) is False  # 8 parallel tiles: under both floors
    assert pt.lookup(row, g(b=1, h_q=16, h_kv=1, s_q=16, s_kv=16384)) is False
    assert pt.lookup(row, g(dtype=cudnn.data_type.HALF)) is None


@pytest.mark.L0
def test_sm90_d64_sink_mtp_entries_follow_the_study():
    """Study sm90_d64_sink_mtp_20261118 (H100): gpt-oss paged d64 MTP with a sink, which SM90 runs unpacked, leads with a window always
    and with the full cache up to 2^4.5 parallel tiles, on and off the KV tile."""
    row = "sdpa_fwd_prefill_sm90"

    def mtp(**over):
        base = dict(b=1, h_q=64, h_kv=8, s_q=3, s_kv=131072, d_qk=64, d_v=64, causal=True, bottom_right=True, has_sink=True, device_cc=(9, 0))
        base.update(has_paged_kv=True, page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    study = "sm90_d64_sink_mtp_20261118"
    for kv in (131072, 131073):
        assert pt.class_key(row, mtp(s_kv=kv))[8] == "gqa_unpacked"
        assert pt.lookup(row, mtp(s_kv=kv)) is True and pt.matched_entry(row, mtp(s_kv=kv)).study == study
        assert pt.lookup(row, mtp(s_kv=kv, b=128, h_q=8, h_kv=1)) is False  # 128 parallel tiles: past the full-cache floor
        assert pt.lookup(row, mtp(s_kv=kv, b=128, h_q=8, h_kv=1, window_left=127)) is True


@pytest.mark.L0
def test_sm107_paged_window_entries_follow_the_study():
    """Study sm107_window_gaps_20261118 (Rubin, the inference benchmark's windowed paged generation graph): windowed paged d192 x 192 and
    d256 lead at S_q 1 through prefill, d64 up to S_q 45, d128 MTP (whole-pack groups) on caches up to 32k. The unpacked d128 MTP
    classes have no entry, as do f16 and another part; windowed d128 verify answers from sm107_d128_window_mtp_verify_20261118."""
    row = "sdpa_fwd_prefill_sm107"

    def windowed(**over):
        base = dict(b=16, h_q=64, h_kv=4, s_q=4, s_kv=131072, d_qk=192, d_v=192, causal=True, bottom_right=True, window_left=511)
        base.update(device_cc=(10, 7), has_paged_kv=True, page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for s_q in (1, 4, 8, 1024):
        assert pt.lookup(row, windowed(s_q=s_q)) is True, s_q
        assert pt.lookup(row, windowed(s_q=s_q, d_qk=256, d_v=256, h_q=16, h_kv=8, window_left=1023)) is True, s_q
        assert pt.matched_entry(row, windowed(s_q=s_q)).study == "sm107_window_gaps_20261118"
    d64 = dict(d_qk=64, d_v=64, h_q=128, h_kv=16, window_left=2047)
    assert pt.lookup(row, windowed(**d64, s_q=4)) is True
    assert pt.lookup(row, windowed(**d64, s_q=16)) is True  # `le q 5.5`: log2 S_q, so up to S_q 45
    assert pt.lookup(row, windowed(**d64, s_q=128)) is False  # longer prefill keeps the backend
    d128 = dict(d_qk=128, d_v=128, h_q=64, h_kv=8)
    assert pt.lookup(row, windowed(**d128, s_kv=16384)) is None  # on the tile: the SM107 hand rule (sm107_d128_window_mtp_v4_20261118)
    assert pt.lookup(row, windowed(**d128, s_kv=131072)) is None
    assert pt.matched_entry(row, windowed(**d128, s_q=8)).study == "sm107_d128_window_mtp_verify_20261118"  # windowed d128 verify
    assert pt.lookup(row, windowed(**{**d128, "h_q": 80})) is None  # group 10 runs unpacked: no entry
    assert pt.lookup(row, windowed(dtype=cudnn.data_type.HALF)) is None
    assert pt.lookup(row, windowed(device_cc=(10, 3))) is None


@pytest.mark.L0
def test_sm107_padded_verify_keys_apart_and_d192x128_sink_window_entries_follow_the_study():
    """Study sm107_d192x128_sink_verify_20261118 (Rubin): padded-Q GQA verify (S_q 5-16) keys `verify` and falls back to the prefill
    entry while its own class has none; THD Q keeps `prefill`. Mimo d192 x 128 verify with a window and a sink leads from 2^2 parallel
    tiles where the row packs the group whole; a group packed in part keys `gqa_partial` and leads up to 2^6.5 tiles."""
    row = "sdpa_fwd_prefill_sm107"

    def v(**over):
        base = dict(b=128, h_q=128, h_kv=8, s_q=8, s_kv=131072, d_qk=192, d_v=128, causal=True, bottom_right=True, device_cc=(10, 7))
        base.update(has_paged_kv=True, page_size=16, padded=True, window_left=127, has_sink=True)
        base.update(over)
        return _facts(**base)

    study = "sm107_d192x128_sink_verify_20261118"
    for kv in (131072, 131073):
        assert pt.class_key(row, v(s_kv=kv))[4:9] == ("verify", "paged", "window", True, "gqa")
        assert pt.lookup(row, v(s_kv=kv)) is True and pt.matched_entry(row, v(s_kv=kv)).study == study
        assert pt.lookup(row, v(s_kv=kv, b=1, h_q=16, h_kv=1)) is False  # one parallel tile: under the floor
        assert pt.class_key(row, v(s_kv=kv, h_q=48))[8] == "gqa_partial"  # group 6 packs 2
        assert pt.lookup(row, v(s_kv=kv, h_q=48)) is False  # 1024 parallel tiles: past the partial-pack ceiling
        assert pt.lookup(row, v(s_kv=kv, h_q=48, b=4)) is True
    assert pt.class_key(row, v(thd=True, max_total_seq_len_q=128 * 8))[4] == "prefill"
    # a verify graph whose own class has no entry answers from its prefill entry, packed whole or in part
    for h_q, group in ((64, "gqa"), (48, "gqa_partial")):
        d128 = v(d_qk=128, d_v=128, h_q=h_q, h_kv=8, window_left=None, has_sink=False)
        assert pt.class_key(row, d128)[4] == "verify" and pt.class_key(row, d128)[8] == group
        fallback = pt._class_entry(pt.class_key(row, d128, verify=False))
        assert fallback is not None and pt.matched_entry(row, d128) is fallback


@pytest.mark.L0
def test_sm107_d128_window_verify_entries_follow_the_study():
    """Study sm107_d128_window_mtp_verify_20261118 (Rubin): windowed paged d128 verify the row packs whole leads from 2^10.8 query rows
    on the KV tile and 2^5.3 parallel tiles off it; partial packs keep the hand rule, and MTP keeps the 91st commit's entries."""
    row = "sdpa_fwd_prefill_sm107"

    def v(**over):
        base = dict(b=128, h_q=64, h_kv=8, s_q=8, s_kv=131072, d_qk=128, d_v=128, causal=True, bottom_right=True, device_cc=(10, 7))
        base.update(has_paged_kv=True, page_size=16, padded=True, window_left=511)
        base.update(over)
        return _facts(**base)

    study = "sm107_d128_window_mtp_verify_20261118"
    assert pt.class_key(row, v())[4:9] == ("verify", "paged", "window", False, "gqa")
    for kv in (131072, 131073):
        assert pt.lookup(row, v(s_kv=kv)) is True and pt.matched_entry(row, v(s_kv=kv)).study == study  # step37_swa b 128
        assert pt.lookup(row, v(s_kv=kv, b=1)) is False  # 2^9 query rows / 2^3 tiles: under both floors
    assert pt.lookup(row, v(b=4)) is True  # 2^11 query rows
    assert pt.lookup(row, v(s_kv=131073, b=8)) is True  # 2^6 tiles
    assert pt.class_key(row, v(h_q=80))[8] == "gqa_partial" and pt.matched_entry(row, v(h_q=80)) is None
    assert pt.matched_entry(row, v(s_q=4)) is None  # MTP on the tile: the SM107 hand rule (sm107_d128_window_mtp_v4_20261118)
    assert pt.matched_entry(row, v(s_q=4, s_kv=131073)).study == "sm107_window_gaps_20261118"


@pytest.mark.L0
def test_sm107_paged_d128_unpacked_mtp_entries_follow_the_study():
    """Study sm107_d128_paged_mtp_unpacked_20261118 (Rubin): paged d128 MTP over the full cache at GQA groups the row cannot pack whole
    leads below cost 2^36.7 on the KV tile and below 2^9.25 query rows off it."""
    row = "sdpa_fwd_prefill_sm107"

    def mtp(**over):
        base = dict(b=1, h_q=48, h_kv=8, s_q=4, s_kv=131072, d_qk=128, d_v=128, causal=True, bottom_right=True, device_cc=(10, 7))
        base.update(has_paged_kv=True, page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    study = "sm107_d128_paged_mtp_unpacked_20261118"
    assert pt.class_key(row, mtp())[8] == "gqa_unpacked"
    assert pt.lookup(row, mtp()) is True and pt.matched_entry(row, mtp()).study == study
    assert pt.lookup(row, mtp(b=128)) is True  # cost 2^36.58
    assert pt.lookup(row, mtp(b=128, h_q=80)) is False  # cost 2^37.32: the backend
    assert pt.lookup(row, mtp(s_kv=131073)) is True and pt.matched_entry(row, mtp(s_kv=131073)).study == study
    assert pt.lookup(row, mtp(s_kv=131073, b=16)) is False  # 3072 query rows
    assert pt.matched_entry(row, mtp(h_q=64)).study == "sm107_paged_mtp_20261110"  # group 8 packs whole
    assert pt.lookup(row, mtp(dtype=cudnn.data_type.HALF)) is None


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
def test_sm107_dense_d256_single_row_decode_entries_are_retired():
    """Study sm107_decode_q1_20261112's dense d256 s_q == 1 entries are retired: the class follows FROST's planned CTA count, which
    the SM107 hand rule reads (sm107_dense_d256_q1_cta_v2_20261118). Its paged d512 / D576 entries stay."""
    row = "sdpa_fwd_prefill_sm107"

    def q1(**over):
        base = dict(b=32, h_q=32, h_kv=2, s_q=1, s_kv=131072, d_qk=256, d_v=256, device_cc=(10, 7), has_paged_kv=False, padded=False, page_size=0)
        base.update(over)
        return _facts(**base)

    for s_kv in (131072, 131071, 2048, 2049):
        assert pt.class_key(row, q1(s_kv=s_kv))[2:9] == ("bf16", (256, 256), "q1", "dense", "none", False, "gqa")
        assert pt.lookup(row, q1(s_kv=s_kv)) is None
    assert pt.matched_entry(row, q1(d_qk=512, d_v=512, h_q=64, h_kv=1, has_paged_kv=True, padded=True, page_size=64)).study == "sm107_decode_q1_20261112"


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm107_dense_d256_prefill_entries_are_retired(s_kv):
    """Rubin dense d256 prefill: sm107_dense_chunk_20261117 (small squares, 1.58x) and its v2 (decode-shaped short-cache chunks,
    5.9-7.8x) are both retired; the hand rule decides (it leads decode-shaped chunks and chunks within 128 launch tiles)."""
    from cudnn.sdpa.fwd import placement
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    row = "sdpa_fwd_prefill_sm107"
    base = dict(b=16, h_q=40, h_kv=8, s_q=8, s_kv=2048 + (s_kv % 2), d_qk=256, d_v=256, causal=True, bottom_right=True, device_cc=(10, 7))
    base.update(has_paged_kv=False, padded=False, page_size=0, device_sm_count=208)
    assert pt.matched_entry(row, _facts(**base)) is None
    spec = next(s for s in ENGINE_SPECS if s.name == row)
    assert placement.place(spec, _facts(**base)) == placement.LEAD  # decode-shaped: the hand rule leads (backend 5.9-7.8x)


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_d576_mla_prefill_follows_its_study(s_kv):
    """Study sm100_d576_prefill_20261117: absorbed-MLA (576, 512) dense prefill leads on B200, B300 (its transfer check) and Rubin;
    FROST ran 0.09-0.20 of the default backend. The paged twin is retired after an edge supplement (a small square at 1.157)."""
    study = "sm100_d576_prefill_20261117"

    def chunk(cc, paged, **over):
        base = dict(b=1, h_q=96, h_kv=1, s_q=1024, s_kv=s_kv, d_qk=576, d_v=512, causal=True, bottom_right=True, device_cc=cc)
        base.update(has_paged_kv=paged, padded=paged, page_size=32 if paged else 0, thd=paged, max_total_seq_len_q=1024 if paged else None)
        base.update(over)
        return _facts(**base)

    for row, cc, layouts in (("sdpa_fwd_prefill_sm100", (10, 0), (False,)), ("sdpa_fwd_prefill_sm107", (10, 7), (False,))):
        for paged in layouts:
            for over in (dict(), dict(b=16, h_q=24, s_q=8)):
                # Rubin's padded-Q GQA verify keys `verify` and answers from the prefill entry while its own class has none
                verify = row == "sdpa_fwd_prefill_sm107" and over
                assert pt.class_key(row, chunk(cc, paged, **over))[4] == ("verify" if verify else "prefill")
                assert pt.lookup(row, chunk(cc, paged, **over)) is True, (row, paged, over)
                assert pt.matched_entry(row, chunk(cc, paged, **over)).study == study
    assert getattr(pt.matched_entry("sdpa_fwd_prefill_sm107", chunk((10, 7), True)), "study", None) != study
    assert pt.lookup("sdpa_fwd_prefill_sm100", chunk((10, 3), False)) is True
    assert pt.matched_entry("sdpa_fwd_prefill_sm100", chunk((10, 3), False)).study == study
    # the paged twin was retired (edge_supp_20261117: 1.157 on a small square); its re-study answers cc 10.0 and, by transfer, cc 10.3
    for cc in ((10, 0), (10, 3)):
        assert pt.matched_entry("sdpa_fwd_prefill_sm100", chunk(cc, True)).study == "sm100_d576_paged_v2_20261117"
        assert pt.lookup("sdpa_fwd_prefill_sm100", chunk(cc, True)) is True
    if s_kv % 128:  # off the tile: a tiny square sits below the cost floor
        assert pt.lookup("sdpa_fwd_prefill_sm100", chunk((10, 0), True, b=1, h_q=8, s_q=33, s_kv=33, max_total_seq_len_q=33)) is False


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm107_paged_d64_sink_prefill_follows_its_v2_study(s_kv):
    """Study sm107_paged_d64_sink_v2_20261117 replaced sm107_paged_d64_sink_prefill_20261117 (whose rules lost up to 1.65x on small
    squares): Rubin paged gpt-oss d64 + sink prefill leads from KV 2^13.5 on the tile and 2^8.5 off it."""
    row = "sdpa_fwd_prefill_sm107"
    base = dict(b=1, h_q=64, h_kv=8, s_q=1024, s_kv=s_kv, d_qk=64, d_v=64, causal=True, bottom_right=True, device_cc=(10, 7))
    base.update(has_paged_kv=True, padded=True, page_size=16, thd=True, has_sink=True, max_total_seq_len_q=1024)
    f = lambda **over: _facts(**{**base, **over})
    assert pt.matched_entry(row, f()).study == "sm107_paged_d64_sink_v2_20261117"
    assert pt.lookup(row, f()) is True  # gpt-oss b 1 q 1024 over a 128k cache
    small = 32 + (s_kv % 2)  # a small square (a first chunk)
    assert pt.lookup(row, f(b=8, s_q=small, s_kv=small, max_total_seq_len_q=8 * small)) is False


@pytest.mark.L0
def test_sm120_window_sink_d192_prefill_entries_are_retired():
    """Study sm120_window_sink_prefill_20261117 trained at S_q >= 256; an edge supplement at S_q 5-192 lost up to 2.94x against the
    hand rule's lead, so its entries are retired."""
    row = "sdpa_fwd_prefill_sm120"
    base = dict(b=1, h_q=16, h_kv=1, s_q=1024, s_kv=131072, d_qk=192, d_v=128, causal=True, bottom_right=True, device_cc=(12, 0))
    base.update(has_paged_kv=False, padded=False, page_size=0, window_left=127, has_sink=True)
    for s_kv in (131072, 131073):
        assert getattr(pt.matched_entry(row, _facts(**{**base, "s_kv": s_kv})), "study", None) != "sm120_window_sink_prefill_20261117"


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
    for a long-Q D64 graph over a long cache, and leaves D256 unsplit graphs to sm100_d256_paged_context_20261030. cc 10.3 borrows
    the study after its own B300 transfer check (b300_d64_transfer_20261117), so the unsplit short-Q D64 graphs lead there too."""
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
        assert pt.lookup(row, facts) is True and placement.place(spec, facts) == placement.LEAD
    assert pt.lookup(row, _sm100_paged(64, 16, 8, 16, 64, 16384, **b300)) is True
    assert pt.matched_entry(row, _sm100_paged(64, 16, 8, 16, 64, 16384, **b300)).study == study
    assert pt.matched_entry(row, _sm100_paged(256, 32, 8, 16, 512, 8192, **b300)).study == "sm100_d256_paged_context_20261030"


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131071], ids=["on_tile", "off_tile"])
def test_sm90_decode_gaps_follow_their_study(s_kv):
    """Study sm90_decode_gaps_20261117 (H100): paged d512 verify and paged d256 sliding-window decode / MTP / verify lead (FROST
    0.05 / 0.28-0.46 of the backend)."""
    row = "sdpa_fwd_prefill_sm90"
    study = "sm90_decode_gaps_20261117"

    def f(**over):
        base = dict(b=1, h_q=8, h_kv=1, s_q=8, s_kv=s_kv, d_qk=512, d_v=512, causal=True, bottom_right=True)
        base.update(device_cc=(9, 0), page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for over in (dict(), dict(b=64, h_q=32, h_kv=4, s_q=16)):  # paged d512 verify
        assert pt.class_key(row, f(**over))[4] == "verify"
        assert pt.lookup(row, f(**over)) is True, over
        assert pt.matched_entry(row, f(**over)).study == study, over
    swa = dict(h_q=16, h_kv=8, d_qk=256, d_v=256, window_left=1023)
    assert pt.lookup(row, f(**dict(swa, s_q=8))) is True  # gemma4 SWA verify
    assert pt.matched_entry(row, f(**dict(swa, s_q=8))).study == study
    for s_q in (1, 4):  # single-row / MTP: gemma4 SWA's group 2, and groups the row cannot pack (3 here)
        assert pt.class_key(row, f(**dict(swa, s_q=s_q, h_q=12, h_kv=4)))[8] == "gqa_unpacked"
        assert pt.lookup(row, f(**dict(swa, s_q=s_q, h_q=12, h_kv=4))) is True, s_q
        assert pt.lookup(row, f(**dict(swa, s_q=s_q))) is True, s_q


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [32768, 32769], ids=["on_tile", "off_tile"])
def test_sm90_dense_d256_single_row_decode_keys_partial_packs_apart(s_kv):
    """Study sm90_dense_d256_q1_fullpack_h100_20261118 (H100): whole-group packs lead once the decode carries enough work (FROST 0.81-0.83
    of the backend on the holdout); a group the row packs only in part keys `gqa_partial` (1.35-1.40x on H100) and keeps the backend,
    as does an unpacked one. The partial key is scoped to the measured classes: dense and paged d256 single-row decode on SM90."""
    row = "sdpa_fwd_prefill_sm90"

    def q1(**over):
        base = dict(b=128, h_q=32, h_kv=2, s_q=1, s_kv=s_kv, d_qk=256, d_v=256, causal=False, bottom_right=True)
        base.update(device_cc=(9, 0), has_paged_kv=False, page_size=0, padded=False)
        base.update(over)
        return _facts(**base)

    assert pt.class_key(row, q1())[8] == "gqa"
    assert pt.lookup(row, q1()) is True  # qwen3.5 32/2: h200 run-564 ran it on the backend at 1.14-1.25x
    assert pt.matched_entry(row, q1()).study == "sm90_dense_d256_q1_fullpack_h100_20261118"
    assert pt.lookup(row, q1(h_q=8, h_kv=1, b=32)) is True  # qwen3.5 TP4 8/1
    assert pt.lookup(row, q1(b=1, s_kv=s_kv // 16)) is False  # little work: the backend
    assert pt.class_key(row, q1(h_q=24, h_kv=4))[8] == "gqa_partial"  # G6 packs 2
    assert pt.lookup(row, q1(h_q=24, h_kv=4)) is None
    assert pt.class_key(row, q1(h_q=12, h_kv=4))[8] == "gqa_unpacked"  # G3
    paged = q1(h_q=24, h_kv=4, has_paged_kv=True, page_size=16, padded=True)
    assert pt.class_key(row, paged)[8] == "gqa_partial"  # paged too (sm90_paged_decode_gaps_20261118)
    assert pt.lookup(row, paged) is False and pt.matched_entry(row, paged).study == "sm90_paged_decode_gaps_20261118"


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm90_paged_decode_gap_entries_follow_the_study(s_kv):
    """Study sm90_paged_decode_gaps_20261118 (H100 NVL): the h200 nightly's paged generation models. Windowed d128 MTP leads always and
    verify from 2^9.2 query rows; d192 x 192 MTP / verify leads, full cache or windowed (windowed MTP on the KV tile from 2^1.5 tiles)."""
    row = "sdpa_fwd_prefill_sm90"

    def g(**over):
        base = dict(b=128, h_q=64, h_kv=8, s_q=4, s_kv=s_kv, d_qk=128, d_v=128, causal=True, bottom_right=True, window_left=511)
        base.update(device_cc=(9, 0), has_paged_kv=True, page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    study = "sm90_paged_decode_gaps_20261118"
    for s_q in (4, 8):
        owner = "sm90_paged_window_mtp_gaps_20261121" if s_q == 8 and s_kv % 128 else study  # off-tile windowed verify is the window study's
        assert pt.lookup(row, g(s_q=s_q)) is True and pt.matched_entry(row, g(s_q=s_q)).study == owner  # step37_swa b 128
    assert pt.lookup(row, g(b=1)) is True  # MTP: always
    assert pt.lookup(row, g(s_q=8, b=1, h_q=16, h_kv=2)) is False  # verify: 2^7 query rows, under the floor
    step5 = dict(d_qk=192, d_v=192, h_q=64, h_kv=4)
    for s_q in (4, 8):
        assert pt.lookup(row, g(**step5, s_q=s_q, window_left=None)) is True  # step5_full
        assert pt.lookup(row, g(**step5, s_q=s_q)) is True  # step5_swa
    assert pt.lookup(row, g(**{**step5, "b": 1, "h_q": 16, "h_kv": 1})) is (s_kv % 128 != 0)  # windowed MTP: one tile leads only off the KV tile


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm90_paged_window_and_unpacked_mtp_follow_their_study(s_kv):
    """Study sm90_paged_window_mtp_gaps_20261121 owns paged d128 MTP whose group the decode tile does not divide (gqa_unpacked),
    windowed d128 single-row decode, windowed d128 verify off the KV tile, and windowed d64 MTP and verify. Windowed d192 x 192 and
    d64 single-row decode are not its."""
    row = "sdpa_fwd_prefill_sm90"
    study = "sm90_paged_window_mtp_gaps_20261121"
    on_tile = s_kv % 128 == 0

    def g(**over):
        base = dict(b=1, h_q=40, h_kv=8, s_q=4, s_kv=s_kv, d_qk=128, d_v=128, causal=True, bottom_right=True, window_left=None)
        base.update(device_cc=(9, 0), has_paged_kv=True, page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for b in (1, 64):
        mtp = g(b=b)  # group 5
        assert pt.class_key(row, mtp)[8] == "gqa_unpacked" and pt.matched_entry(row, mtp).study == study
    for b, h_q, h_kv in ((1, 10, 1), (128, 16, 2)):
        assert pt.matched_entry(row, g(b=b, h_q=h_q, h_kv=h_kv, s_q=1, window_left=1023)).study == study
    for b, h_q, h_kv in ((1, 10, 1), (4, 64, 8)):
        verify = g(b=b, h_q=h_q, h_kv=h_kv, s_q=8, window_left=1023)
        assert pt.matched_entry(row, verify).study == ("sm90_paged_decode_gaps_20261118" if on_tile else study)
    for s_q, h_q, h_kv in ((4, 16, 2), (8, 16, 2), (4, 128, 16), (8, 128, 16)):
        d64 = g(b=128, h_q=h_q, h_kv=h_kv, s_q=s_q, d_qk=64, d_v=64, window_left=2047)
        assert pt.matched_entry(row, d64).study == study
    for single_row in (g(s_q=1, d_qk=192, d_v=192, h_q=64, h_kv=4, window_left=511), g(s_q=1, d_qk=64, d_v=64, h_q=16, h_kv=2, window_left=2047)):
        entry = pt.matched_entry(row, single_row)
        assert entry is None or entry.study != study


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm90_paged_window_d192_decode_keys_by_pack_group(s_kv):
    """Windowed paged d192 x 192 single-row decode keys by the row's pack group: a whole pack keys gqa, a partial pack (the packed
    heads divide the tile but not the group) gqa_partial, and both belong to study sm90_paged_window_q1_v2_20261121. An unpacked
    (odd) group, d192 without a window, dense d192 and d256 keep their keys and owners."""
    row = "sdpa_fwd_prefill_sm90"
    study = "sm90_paged_window_q1_v2_20261121"

    def g(b, h_q, h_kv, **over):
        base = dict(b=b, h_q=h_q, h_kv=h_kv, s_q=1, s_kv=s_kv, d_qk=192, d_v=192, causal=True, bottom_right=True, window_left=511)
        base.update(device_cc=(9, 0), has_paged_kv=True, page_size=16, padded=True)
        base.update(over)
        return _facts(**base)

    for b, h_q, h_kv in ((1, 64, 4), (128, 64, 4), (128, 16, 8), (64, 128, 4)):  # groups 16 / 16 / 2 / 32
        whole = g(b, h_q, h_kv)
        assert pt.class_key(row, whole)[8] == "gqa" and pt.matched_entry(row, whole).study == study
    for b, h_q, h_kv in ((1, 48, 4), (64, 48, 4), (8, 96, 8), (16, 24, 2)):  # groups 12 / 12 / 12 / 12
        partial = g(b, h_q, h_kv)
        assert pt.class_key(row, partial)[8] == "gqa_partial" and pt.matched_entry(row, partial).study == study
    assert pt.class_key(row, g(1, 40, 8))[8] == "gqa_unpacked" and pt.matched_entry(row, g(1, 40, 8)) is None
    for other in (g(8, 48, 4, causal=False, bottom_right=False, window_left=None), g(8, 48, 4, has_paged_kv=False, page_size=0, padded=False)):
        assert pt.class_key(row, other)[8] == "gqa" and pt.matched_entry(row, other) is None
    d256 = g(8, 48, 4, d_qk=256, d_v=256, causal=False, bottom_right=False, window_left=None)
    assert pt.class_key(row, d256)[8] == "gqa_partial" and pt.matched_entry(row, d256).study == "sm90_paged_decode_gaps_20261118"


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [62208, 62209], ids=["on_tile", "off_tile"])
def test_sm90_mha_mtp_follows_its_study(s_kv):
    """Study sm90_mha_mtp_v2_20261117 (H100): dense d128 MHA MTP leads while few head-batch units cover a long cache (FROST splits
    KV; the h200 auto-regressive DiT 9/9 ran 6.4x on the backend) and keeps the backend from about 2^6 tiles."""
    row = "sdpa_fwd_prefill_sm90"

    def mtp(**over):
        base = dict(b=1, h_q=9, h_kv=9, s_q=2, s_kv=s_kv, d_qk=128, d_v=128, causal=True, bottom_right=True)
        base.update(has_paged_kv=False, device_cc=(9, 0))
        base.update(over)
        return _facts(**base)

    for over in (dict(), dict(s_q=4), dict(b=4, h_q=12, h_kv=12)):
        assert pt.lookup(row, mtp(**over)) is True, over
        assert pt.matched_entry(row, mtp(**over)).study == "sm90_mha_mtp_v2_20261117"
    assert pt.lookup(row, mtp(b=16, h_q=32, h_kv=32)) is False
    assert pt.matched_entry(row, mtp(h_q=32, h_kv=8)).study == "sm90_d128_h100_refit_20261114"  # GQA keeps its study


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [32768, 32767], ids=["on_tile", "off_tile"])
def test_sm90_dense_d256_chunks_follow_their_study(s_kv):
    """Study sm90_dense_d256_chunk_v2_20261117 (H100): dense bottom-right d256 chunks lead while few query tiles are in flight
    (qwen3.5 TP4 / TP8 ran the backend at 1.8-4.1x) and keep the backend past about 2^6 tiles. A causal square reads as bottom-right,
    so it shares the class (sm90_dense_d256_square_xfer_20261117 checked it)."""
    row = "sdpa_fwd_prefill_sm90"

    def chunk(**over):
        base = dict(b=1, h_q=8, h_kv=1, s_q=512, s_kv=s_kv, d_qk=256, d_v=256, causal=True, bottom_right=True)
        base.update(has_paged_kv=False, device_cc=(9, 0))
        base.update(over)
        return _facts(**base)

    for over in (dict(), dict(h_q=4), dict(h_q=4, h_kv=2, s_q=1024)):
        assert pt.lookup(row, chunk(**over)) is True, over
        assert pt.matched_entry(row, chunk(**over)).study == "sm90_dense_d256_chunk_v2_20261117"
    assert pt.lookup(row, chunk(b=4, h_q=32, h_kv=2, s_q=2048)) is False  # 32 / 2 at q 2048: 1.8-1.9x on FROST
    square = chunk(s_q=s_kv)
    assert pt.matched_entry(row, square).study == "sm90_dense_d256_chunk_v2_20261117"
    assert pt.lookup(row, square) is False  # 2048 query tiles at 32k
    assert pt.lookup(row, chunk(s_q=512, s_kv=512)) is True  # 32 tiles


_SM90_D128_STUDY = "sm90_d128_h100_refit_20261114"
_SM90_D128_ENTRIES = [
    pytest.param(
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "gqa", False),
        "always",
        None,
        "cost",
        id="mtp-gqa-causal_br-on",
    ),
    pytest.param(
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "gqa", True),
        "always",
        None,
        "cost",
        id="mtp-gqa-causal_br-off",
    ),
    pytest.param(
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "none", False, "gqa", False), "always", None, "cost", id="mtp-gqa-none-on"
    ),
    pytest.param(
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "none", False, "gqa", True), "always", None, "cost", id="mtp-gqa-none-off"
    ),
    pytest.param(
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "none", False, "gqa_unpacked", False),
        "never",
        None,
        "cost",
        id="mtp-gqa_unpacked-none-on",
    ),
    pytest.param(
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "none", False, "gqa_unpacked", True),
        "le",
        23.977,
        "cost",
        id="mtp-gqa_unpacked-none-off",
    ),
    pytest.param(
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "causal_tl", False, "gqa", False),
        "ge",
        8.293,
        "kv",
        id="prefill-gqa-causal_tl-on",
    ),
    pytest.param(
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "causal_tl", False, "gqa", True),
        "always",
        None,
        "cost",
        id="prefill-gqa-causal_tl-off",
    ),
]


def _sm90_d128_candidates(key):
    """Graphs of the SM90 exact dense d128 class ``key``: the class lattice the study timed, searched in a fixed order."""
    phase, layout, mask, has_sink, group, off = key[4], key[5], key[6], key[7], key[8], key[9]
    heads = [(16, 16), (32, 32), (24, 24)] if group == "mha" else [(32, 8), (64, 8), (32, 4), (16, 2), (32, 1), (40, 8), (24, 8), (56, 8), (28, 4), (12, 4)]
    s_qs = {"prefill": [32, 64, 128, 256, 512, 1024, 2048, 4096], "mtp": [2, 3, 4], "verify": [5, 8, 12, 16]}[phase]
    for s_q in s_qs:
        for ratio in (1, 2, 4, 8, 32):
            s_kv = s_q * ratio if s_q * ratio >= 128 else 128 * ratio
            s_kv = s_kv + 64 if off else s_kv
            if mask == "causal_tl" and s_kv == s_q:
                continue
            for b in (1, 4, 16, 64):
                for h_q, h_kv in heads:
                    yield _facts(
                        b=b,
                        h_q=h_q,
                        h_kv=h_kv,
                        s_q=s_q,
                        s_kv=s_kv,
                        d_qk=128,
                        d_v=128,
                        dtype=cudnn.data_type.BFLOAT16,
                        causal=mask != "none",
                        bottom_right=mask == "causal_br",
                        has_sink=has_sink,
                        has_paged_kv=False,
                        padded=False,
                        page_size=0,
                        device_cc=(9, 0),
                        device_sm_count=132,
                    )


@pytest.mark.L0
@pytest.mark.parametrize("key, direction, threshold, feature", _SM90_D128_ENTRIES)
def test_sm90_d128_classes_follow_their_study(key, direction, threshold, feature):
    """Study sm90_d128_h100_refit_20261114 (Rule S12): each SM90 exact dense d128 class the D128 template serves answers from its own entry, on both
    sides of a threshold; the on- and off-tile classes are each their own and a long off-tile length stays in its class."""
    row = "sdpa_fwd_prefill_sm90"
    found = [f for f in _sm90_d128_candidates(key) if pt.class_key(row, f) == key]
    assert found, key
    for f in found[:8]:
        entry = pt.matched_entry(row, f)
        assert entry is not None and entry.study == _SM90_D128_STUDY, (key, entry)
    if direction in ("always", "never"):
        assert all(pt.lookup(row, f) is (direction == "always") for f in found)
    else:
        values = sorted(((pt.FEATURES[feature](f), f) for f in found), key=lambda vf: vf[0])
        below = [f for v, f in values if (v < threshold if direction == "ge" else v > threshold)]
        above = [f for v, f in values if (v >= threshold if direction == "ge" else v <= threshold)]
        assert below and above, (key, threshold, [v for v, _ in values][:3], [v for v, _ in values][-3:])
        assert pt.lookup(row, below[-1]) is False and pt.lookup(row, above[0]) is True
    # the twin class across the first placement tile, and a long off-tile control in the off-tile class
    f = found[0]
    twin = dataclasses.replace(f, s_kv=f.s_kv - 64 if key[9] else f.s_kv + 64)
    assert pt.class_key(row, twin)[9] is (not key[9])
    if key[9]:
        long = dataclasses.replace(f, s_kv=65535)
        assert pt.class_key(row, long) == key and pt.matched_entry(row, long).study == _SM90_D128_STUDY


@pytest.mark.L0
def test_sm90_d128_square_causal_reads_bottom_right():
    """A square causal mask reads causal_br from either corner; top-left is its own class only with S_q != S_kv."""
    row = "sdpa_fwd_prefill_sm90"
    base = dict(
        b=4,
        h_q=32,
        h_kv=8,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        causal=True,
        has_paged_kv=False,
        padded=False,
        page_size=0,
        device_cc=(9, 0),
        device_sm_count=132,
    )
    assert pt.class_key(row, _facts(s_q=1024, s_kv=1024, bottom_right=False, **base))[6] == "causal_br"
    assert pt.class_key(row, _facts(s_q=1024, s_kv=1024, bottom_right=True, **base))[6] == "causal_br"
    assert pt.class_key(row, _facts(s_q=512, s_kv=1024, bottom_right=False, **base))[6] == "causal_tl"


_SM90_D128_HAND_RULE_CLASSES = frozenset(
    {
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "gqa_unpacked", True),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "gqa_unpacked", False),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "none", False, "mha", True),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "none", False, "mha", False),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", True),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", False),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "gqa", True),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "gqa", False),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "mha", True),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "mha", False),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "causal_tl", False, "mha", True),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "causal_tl", False, "mha", False),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "mha", True),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "mha", False),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "verify", "dense", "causal_br", False, "gqa", True),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "verify", "dense", "causal_br", False, "gqa", False),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "verify", "dense", "none", False, "gqa", True),
        ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "verify", "dense", "none", False, "gqa", False),
    }
)


@pytest.mark.L0
def test_sm90_d128_hand_rule_answers_exactly_its_classes(monkeypatch):
    """Study sm90_d128_h100_refit_20261114: _place_sm90_d128 answers exactly the exact dense d128 classes whose generated entries did not qualify on
    the fresh holdout (Rule S12's allowed case, its reason in the rule); every other class the study fitted is answered by
    its table entry. Default placement: the FROST flag unset in the body."""
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    from cudnn.sdpa.fwd import placement
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_sm90_f16_three_wg")
    row = "sdpa_fwd_prefill_sm90"
    assert placement.SM90_D128_HAND_RULE_CLASSES == _SM90_D128_HAND_RULE_CLASSES
    for key in sorted(_SM90_D128_HAND_RULE_CLASSES, key=repr):
        found = [f for f in _sm90_d128_candidates(key) if pt.class_key(row, f) == key]
        assert found, key
        for f in found[:4]:
            assert pt.lookup(row, f) is None, key
            assert placement.place(spec, f) == placement._place_sm90_d128(f), key
    for param in _SM90_D128_ENTRIES:
        key = param.values[0]
        assert key not in _SM90_D128_HAND_RULE_CLASSES
        f = next(f for f in _sm90_d128_candidates(key) if pt.class_key(row, f) == key)
        assert placement.place(spec, f) == (placement.LEAD if pt.lookup(row, f) else placement.TRAIL), key
