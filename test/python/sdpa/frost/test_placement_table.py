# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated placement table: well-formed entries, the runtime class key the generator also uses, and the boundary
between table-decided and hand-rule graphs."""

import math
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa import placement_table as pt


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
    return SimpleNamespace(**base)


@pytest.mark.L0
def test_entries_are_well_formed_and_name_known_rows():
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS as BWD
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS as FWD

    rows = {s.name for s in FWD} | {s.name for s in BWD}
    for key, entry in pt.TABLE.items():
        row, cc, dtype, heads, phase, layout, mask, sink, group, split_blocked = key
        assert row in rows, key
        assert isinstance(cc, tuple) and len(cc) == 2 and dtype in ("bf16", "f16") and len(heads) == 2
        assert phase in ("q1", "mtp", "prefill") and layout in ("dense", "paged", "thd", "thd_ragged")
        assert mask in ("none", "window", "causal_br", "causal_tl") and isinstance(sink, bool) and group in ("mha", "gqa")
        assert isinstance(split_blocked, bool) and not (split_blocked and layout == "paged")
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
    assert pt.lookup(row, _facts(device_cc=(10, 3))) is None  # another part: no study, hand rules decide


@pytest.mark.L0
def test_cost_proxy_is_byte_bound_for_decode_and_flop_bound_for_prefill():
    """Decode cost follows KV bytes (independent of h_q at a fixed group of KV heads); prefill follows FLOPs."""
    dec = dict(d_qk=256, d_v=256, s_q=1, h_kv=1, b=16, s_kv=32768)
    assert pt.log2_work(_facts(h_q=4, **dec)) == pytest.approx(pt.log2_work(_facts(h_q=16, **dec)), abs=0.01)
    pre = dict(d_qk=256, d_v=256, s_kv=4096, h_kv=2, b=1, h_q=32)
    assert pt.log2_work(_facts(s_q=4096, **pre)) == pytest.approx(pt.log2_work(_facts(s_q=2048, **pre)) + 1)


@pytest.mark.L0
def test_dense_kv_off_the_tile_is_its_own_class_paged_is_not():
    """A dense S_kv off the 128-key tile cannot split KV (an off-tile dense d256 decode was 11-13x slower on FROST, which
    an on-tile grid cannot see), so it is a separate class; paged KV splits at any logical maximum and keeps its class
    (review on #1957: an off-tile paged D576 decode lost its FROST split route to the backend)."""
    row = "sdpa_fwd_prefill_sm100"
    assert pt.class_key(row, _facts(s_kv=8191))[-1] is False  # paged
    assert pt.lookup(row, _facts(s_kv=49151, h_q=16, b=1)) == pt.lookup(row, _facts(s_kv=49152, h_q=16, b=1))
    dense = dict(has_paged_kv=False, d_qk=256, d_v=256, h_q=16)
    assert pt.class_key(row, _facts(s_kv=8191, **dense))[-1] is True
    assert pt.class_key(row, _facts(s_kv=8192, **dense))[-1] is False


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
