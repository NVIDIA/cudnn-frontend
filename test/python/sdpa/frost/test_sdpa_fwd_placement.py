# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDPA-forward placement tree (``sdpa/fwd/placement.py``): CPU-only.

Three things are pinned. (1) The table's verdict on one representative graph per
shard, so a threshold cannot move without a test naming the shard. (2) The hook:
``propose()`` emits ``recommend()``'s proposals plus the ``BACKEND`` marker at the
place the table says, ours-first under ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES``,
nothing when nothing is eligible. (3) The data: every decisive cell of the newest
committed benchmark CSV per (config, arch) -- FROST/backend kernel-time ratio
outside +-10 % -- must agree with the table, so the thresholds cannot drift from
the measurements they cite (``benchmark/attention_inference/results``).
"""

import csv
import glob
import os

import pytest

import cudnn
from cudnn.engines import manifest
from cudnn.engines.heuristics import BACKEND, is_backend_block
from cudnn.sdpa.fwd import placement
from cudnn.sdpa.fwd.engines import ENGINE_SPECS
from cudnn.sdpa.fwd.heuristics import propose, recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

_FAMILY = next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_fwd")
_SM100 = "sdpa_fwd_prefill_sm100"
_SM120 = "sdpa_fwd_prefill_sm120"
_OFFERED = {name: _FAMILY.engine_id + slot.slot for name, slot in _FAMILY.slots.items()}
_SPEC = {s.name: s for s in ENGINE_SPECS}


def _facts(**over):
    base = dict(
        b=1,
        h_q=8,
        h_kv=1,
        s_q=1,
        s_kv=8192,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        causal=True,
        bottom_right=True,
        device_cc=(10, 0),
        device_sm_count=148,
    )
    base.update(over)
    return SdpaGraphFacts(**base)


@pytest.fixture(autouse=True)
def _no_opt_in_flag(monkeypatch):
    """The suite's autouse opt-in would make every verdict LEAD; this file tests the table."""
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)


# --- (1) the table ---------------------------------------------------------------

_SM100_CASES = [
    # decode-shaped: the backend has no decode-class engine above s_q == 1
    ("mtp_d128_paged", dict(s_q=4, h_q=64, h_kv=8, b=8, has_paged_kv=True, page_size=16, padded=True), placement.LEAD),
    ("mtp_d256_rows_above_decode_tile", dict(s_q=4, h_q=32, h_kv=2, d_qk=256, d_v=256, b=1), placement.LEAD),
    ("verify_depth_16_d64_sink", dict(s_q=16, h_q=64, h_kv=8, d_qk=64, d_v=64, has_sink=True), placement.LEAD),
    ("verify_depth_17_is_prefill_over_2k", dict(s_q=17, h_q=8, h_kv=1, s_kv=2048), placement.TRAIL),
    ("verify_depth_17_is_prefill_over_32k", dict(s_q=17, h_q=8, h_kv=1, s_kv=32768), placement.LEAD),
    # s_q == 1
    ("decode_d128_b128_kv128k", dict(s_q=1, h_q=64, h_kv=8, b=128, s_kv=131072, has_paged_kv=True, page_size=16, padded=True), placement.TRAIL),
    ("decode_d128_b1", dict(s_q=1, h_q=8, h_kv=1, b=1, s_kv=131072), placement.TRAIL),
    ("decode_d256_units_64", dict(s_q=1, h_q=32, h_kv=2, d_qk=256, d_v=256, b=32, s_kv=2048), placement.LEAD),
    ("decode_d256_units_16_kv8k", dict(s_q=1, h_q=32, h_kv=2, d_qk=256, d_v=256, b=8, s_kv=8192), placement.LEAD),
    ("decode_d256_units_16_kv2k", dict(s_q=1, h_q=32, h_kv=2, d_qk=256, d_v=256, b=8, s_kv=2048), placement.TRAIL),
    ("decode_d256_units_8_kv32k", dict(s_q=1, h_q=8, h_kv=1, d_qk=256, d_v=256, b=8, s_kv=32768), placement.LEAD),
    ("decode_d256_units_8_kv8k", dict(s_q=1, h_q=8, h_kv=1, d_qk=256, d_v=256, b=8, s_kv=8192), placement.TRAIL),
    ("decode_d256_b1", dict(s_q=1, h_q=32, h_kv=2, d_qk=256, d_v=256, b=1, s_kv=131072), placement.TRAIL),
    ("decode_d512_b1_32_heads", dict(s_q=1, h_q=32, h_kv=1, d_qk=512, d_v=512, b=1, s_kv=131072), placement.LEAD),
    ("decode_d512_b1_16_heads", dict(s_q=1, h_q=16, h_kv=1, d_qk=512, d_v=512, b=1, s_kv=131072), placement.TRAIL),
    ("decode_d512_b128", dict(s_q=1, h_q=8, h_kv=1, d_qk=512, d_v=512, b=128, s_kv=131072), placement.LEAD),
    # prefill
    ("chunked_512_over_128k_8_heads", dict(s_q=512, h_q=8, h_kv=1, s_kv=131072), placement.LEAD),
    ("chunked_1024_over_64k_16_heads", dict(s_q=1024, h_q=16, h_kv=2, s_kv=65536), placement.LEAD),
    ("chunked_512_over_128k_32_heads_128_tiles", dict(s_q=512, h_q=32, h_kv=4, s_kv=131072), placement.LEAD),
    ("chunked_512_over_128k_64_heads_256_tiles", dict(s_q=512, h_q=64, h_kv=8, s_kv=131072), placement.TRAIL),
    ("chunked_1024_over_128k_32_heads_256_tiles", dict(s_q=1024, h_q=32, h_kv=4, s_kv=131072), placement.TRAIL),
    ("chunked_2048_over_128k_8_heads_128_tiles", dict(s_q=2048, h_q=8, h_kv=1, s_kv=131072), placement.LEAD),
    ("chunked_512_over_32k_8_heads_32_tiles", dict(s_q=512, h_q=8, h_kv=1, s_kv=32768), placement.LEAD),
    ("chunked_256_over_8k_32_heads_64_tiles", dict(s_q=256, h_q=32, h_kv=4, s_kv=8192), placement.LEAD),
    ("chunked_256_over_8k_64_heads_128_tiles", dict(s_q=256, h_q=64, h_kv=8, s_kv=8192), placement.TRAIL),
    ("chunked_512_over_4k_8_heads", dict(s_q=512, h_q=8, h_kv=1, s_kv=4096), placement.TRAIL),
    ("dense_square_2k", dict(s_q=2048, h_q=8, h_kv=1, s_kv=2048), placement.TRAIL),
    ("dense_square_32k", dict(s_q=32768, h_q=8, h_kv=1, s_kv=32768), placement.TRAIL),
    ("d512_prefill_square", dict(s_q=8192, h_q=64, h_kv=1, s_kv=8192, d_qk=512, d_v=512), placement.LEAD),
    ("d512_prefill_chunked", dict(s_q=1024, h_q=128, h_kv=1, s_kv=131072, d_qk=512, d_v=512), placement.LEAD),
    ("sliding_window_prefill", dict(s_q=512, h_q=8, h_kv=1, s_kv=131072, window_left=127), placement.TRAIL),
    ("d64_envelope_prefill", dict(s_q=512, h_q=8, h_kv=1, s_kv=131072, d_qk=64, d_v=64), placement.TRAIL),
    ("thd_prefill", dict(s_q=1024, h_q=8, h_kv=1, s_kv=131072, thd=True, padded=True), placement.TRAIL),
    ("paged_prefill", dict(s_q=512, h_q=8, h_kv=1, s_kv=131072, has_paged_kv=True, page_size=16, padded=True), placement.TRAIL),
]


@pytest.mark.L0
@pytest.mark.parametrize("case,over,verdict", _SM100_CASES, ids=[c[0] for c in _SM100_CASES])
def test_sm100_f16_placement_table(case, over, verdict):
    assert placement.place(_SPEC[_SM100], _facts(**over)) == verdict


_SM120_CASES = [
    ("decode_b128", dict(s_q=1, h_q=64, h_kv=8, b=128, s_kv=131072, has_paged_kv=True, page_size=16, padded=True), placement.LEAD),
    ("decode_b1_units_1", dict(s_q=1, h_q=8, h_kv=1, b=1, s_kv=131072), placement.TRAIL),
    ("decode_b1_units_8", dict(s_q=1, h_q=64, h_kv=8, b=1, s_kv=131072), placement.LEAD),
    ("decode_d512_group_128", dict(s_q=1, h_q=128, h_kv=1, b=128, s_kv=131072, d_qk=512, d_v=512), placement.TRAIL),
    ("decode_d512_group_64", dict(s_q=1, h_q=64, h_kv=1, b=128, s_kv=131072, d_qk=512, d_v=512), placement.LEAD),
    ("decode_d512_b1_32_heads", dict(s_q=1, h_q=32, h_kv=1, b=1, s_kv=131072, d_qk=512, d_v=512), placement.LEAD),
    ("decode_d512_b1_16_heads", dict(s_q=1, h_q=16, h_kv=1, b=1, s_kv=131072, d_qk=512, d_v=512), placement.TRAIL),
    ("mtp", dict(s_q=4, h_q=32, h_kv=2, b=1, d_qk=256, d_v=256), placement.LEAD),
    ("square", dict(s_q=2048, h_q=8, h_kv=1, s_kv=2048), placement.LEAD),
    ("chunked", dict(s_q=512, h_q=64, h_kv=8, s_kv=131072), placement.LEAD),
    ("swa_square", dict(s_q=2048, h_q=8, h_kv=1, s_kv=2048, window_left=127, d_qk=64, d_v=64), placement.TRAIL),
    ("swa_chunked", dict(s_q=512, h_q=8, h_kv=1, s_kv=131072, window_left=127, d_qk=64, d_v=64), placement.LEAD),
    ("thd", dict(s_q=1024, h_q=8, h_kv=1, s_kv=131072, thd=True, padded=True), placement.LEAD),
]


@pytest.mark.L0
@pytest.mark.parametrize("case,over,verdict", _SM120_CASES, ids=[c[0] for c in _SM120_CASES])
def test_sm120_f16_placement_table(case, over, verdict):
    assert placement.place(_SPEC[_SM120], _facts(device_cc=(12, 0), device_sm_count=188, **over)) == verdict


@pytest.mark.L0
def test_unmeasured_rows_keep_the_historical_order():
    for name in ("sdpa_fwd_prefill_sm107", "sdpa_fwd_prefill_sm80", "sdpa_fwd_prefill_sm100_fp8"):
        assert placement.place(_SPEC[name], _facts(s_q=1, b=1)) == placement.LEAD, name


@pytest.mark.L0
def test_only_the_f16_sm100_and_sm120_slots_are_default_candidates():
    assert set(_FAMILY.offered_ids()) == {_SM100, _SM120}


# --- (2) the hook ------------------------------------------------------------------


def _own(plans):
    return [p for p in plans if not is_backend_block(p)]


@pytest.mark.L0
def test_propose_places_the_marker_per_shard():
    offered = {_SM100: _OFFERED[_SM100]}
    trail = propose("A", _facts(s_q=1, h_q=64, h_kv=8, b=1, s_kv=131072), offered)
    lead = propose("A", _facts(s_q=4, h_q=64, h_kv=8, b=8, has_paged_kv=True, page_size=16, padded=True), offered)
    assert trail and is_backend_block(trail[0]) and not any(is_backend_block(p) for p in trail[1:])
    assert lead and is_backend_block(lead[-1]) and not any(is_backend_block(p) for p in lead[:-1])
    for plans, facts in (
        (trail, _facts(s_q=1, h_q=64, h_kv=8, b=1, s_kv=131072)),
        (lead, _facts(s_q=4, h_q=64, h_kv=8, b=8, has_paged_kv=True, page_size=16, padded=True)),
    ):
        assert [(p.engine_id, p.knobs) for p in _own(plans)] == [(p.engine_id, p.knobs) for p in recommend("A", facts, offered)], "propose adds only the marker"
    assert sum(is_backend_block(p) for p in trail) == 1 and sum(is_backend_block(p) for p in lead) == 1


@pytest.mark.L0
def test_propose_leads_everywhere_under_the_opt_in_flag(monkeypatch):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    plans = propose("A", _facts(s_q=1, h_q=64, h_kv=8, b=1, s_kv=131072), {_SM100: _OFFERED[_SM100]})
    assert plans and is_backend_block(plans[-1]) and not is_backend_block(plans[0])


@pytest.mark.L0
def test_propose_is_empty_when_nothing_is_eligible():
    assert propose("A", _facts(device_cc=(9, 0), device_sm_count=132), {_SM100: _OFFERED[_SM100]}) == []
    assert propose("A", _facts(), {}) == []


@pytest.mark.L0
def test_propose_fallback_kind_carries_the_same_verdict():
    offered = {_SM100: _OFFERED[_SM100]}
    a = propose("A", _facts(s_q=1, h_q=64, h_kv=8, b=1, s_kv=131072), offered)
    fb = propose("FALLBACK", _facts(s_q=1, h_q=64, h_kv=8, b=1, s_kv=131072), offered)
    assert is_backend_block(a[0]) and is_backend_block(fb[0])


@pytest.mark.L0
def test_marker_never_reaches_a_ranked_list():
    from cudnn.engines.base import PlanConfig
    from cudnn.engines.heuristics import _assemble

    backend = [PlanConfig(-1, None), PlanConfig(7, {"k": 1}, cpp_index=0, mode=cudnn.heur_mode.A)]
    for hook in (lambda kind: [BACKEND, PlanConfig(20511, "x")], lambda kind: [PlanConfig(20511, "x"), BACKEND], lambda kind: [BACKEND]):
        for modes in ([cudnn.heur_mode.A], [cudnn.heur_mode.OPENSOURCE], [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]):
            assert not any(is_backend_block(p) for p in _assemble(modes, hook, backend))


# --- (3) the data --------------------------------------------------------------------

_RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "benchmark", "attention_inference", "results")
_ARCH = {"b200": ((10, 0), 148, _SM100), "rtxpro6000": ((12, 0), 188, _SM120)}
_CONFIGS = ("llama3.1", "qwen35", "gpt_oss", "deepseek_v4", "decode_sweep", "prefill_sweep")
_TOL = 0.10  # |ratio - 1| below this: parity, either order is right
_DTYPE = {"bfloat16": cudnn.data_type.BFLOAT16, "float16": cudnn.data_type.HALF}


def _row_facts(r, cc, sms):
    gen, s_q = r["phase"] == "generation", int(r["q_tokens"])
    window = r.get("sliding_window_size") or ""
    return _facts(
        b=int(r["batch_size"]),
        h_q=int(r["num_q_heads"]),
        h_kv=int(r["num_kv_heads"]),
        s_q=s_q,
        s_kv=int(r["kv_len"]),
        d_qk=int(r["head_dim_qk"]),
        d_v=int(r["head_dim_vo"]),
        dtype=_DTYPE[r["data_type"]],
        causal=(not gen) or s_q > 1,
        bottom_right=gen and s_q > 1,
        window_left=int(float(window)) - 1 if window else None,
        has_sink=r["config_name"] == "gpt_oss",
        has_paged_kv=gen,
        padded=gen,
        page_size=int(r["page_size"]) if gen and r.get("page_size") else 0,
        device_cc=cc,
        device_sm_count=sms,
    )


def _decisive_cells(path):
    cells = {}
    for r in csv.DictReader(open(path)):
        if r.get("success") != "True" or not r.get("time_ms") or r["kv_cache_dtype"] != r["data_type"]:
            continue  # fp8-KV rows: the fp8 rows are still opt-in and not what the table places
        key = tuple(r[c] for c in ("model_name", "phase", "batch_size", "q_tokens", "kv_len", "sliding_window_size"))
        cells.setdefault(key, {})[r["backend"]] = r
    for key, d in cells.items():
        if {"cudnn", "cudnn_oss"} <= d.keys() and d["cudnn_oss"]["backend_detail"].startswith("cudnn_oss plan=sdpa_fwd_prefill_"):
            ratio = float(d["cudnn_oss"]["time_ms"]) / float(d["cudnn"]["time_ms"])
            if abs(ratio - 1) >= _TOL:
                yield key, d["cudnn_oss"], ratio


@pytest.mark.L0
@pytest.mark.parametrize("arch", sorted(_ARCH))
@pytest.mark.parametrize("config", _CONFIGS)
def test_table_agrees_with_the_newest_committed_csv(config, arch):
    files = sorted(glob.glob(os.path.join(_RESULTS, config, arch, "*.csv")))
    if not files:
        pytest.skip(f"no {arch} CSV committed for {config}")
    cc, sms, engine = _ARCH[arch]
    spec, wrong, decisive = _SPEC[engine], [], 0
    for key, row, ratio in _decisive_cells(files[-1]):
        verdict = placement.place(spec, _row_facts(row, cc, sms))
        decisive += 1
        if (verdict == placement.LEAD) != (ratio < 1):
            wrong.append(f"{key}: FROST/backend = {ratio:.2f} but the table says {verdict}")
    assert decisive, "no decisive cell in the CSV"
    assert not wrong, f"{len(wrong)} of {decisive} decisive cells disagree with the table:\n" + "\n".join(wrong)
