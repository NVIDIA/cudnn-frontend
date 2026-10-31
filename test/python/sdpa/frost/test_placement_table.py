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
        assert phase in ("q1", "mtp", "prefill") and layout in ("dense", "paged", "thd", "thd_ragged")
        assert mask in ("none", "window", "causal_br", "causal_tl") and isinstance(sink, bool) and group in ("mha", "gqa", "gqa_unpacked")
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
@pytest.mark.parametrize(
    "feature", [dict(has_bias=True), dict(has_dropout=True), dict(right_band_widening=True), dict(is_fp8=True), dict(shape_overrides=True)]
)
def test_unmeasured_features_never_reach_the_table(feature):
    row = "sdpa_fwd_prefill_sm100"
    assert pt.lookup(row, _facts()) is not None, "the D576 decode entry should answer the plain graph"
    assert pt.lookup(row, _facts(**feature)) is None


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
    graphs on cc 10.0, MHA, f16 and top-left chunks have no entry."""
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
    assert pt.lookup(row, chunk(4, 64, 8, 128, 2048, bottom_right=False)) is None  # top-left chunk: not this class
