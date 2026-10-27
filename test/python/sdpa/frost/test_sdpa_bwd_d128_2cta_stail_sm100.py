# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L0 tests of the d128 grouped THD cta_group::2 pair body (``bprop_d128_accum_grouped_2cta_stail_f16``: the dQ half-row image in S^T's dead
tail, FAv4's rotated leader order, the dedicated sDQ ring, 35 barriers).

Rows: the THD matrix (uniform pairs, GQA 8, a varlen ramp, non-causal, the suite's degenerate THD row, a half-dead pair, pair tail 300),
all-empty, the P14 rows (n_q == 1 with more pairs than SMs; a half-dead last pair; a zero-trip MIX of N = 1 / 2 / 3 / 4 / 6 / 8 and dead
blocks), the launch census (the pair MAIN kernel), the typed field tests (the probe domain, the STAMPS lever, the mask form), the host renders
(census + md5 record; every probe's purity delta), the deterministic-hang topologies as SOURCE assertions (RED 3: dp_empty after the dQ
readout; RED 4: the dS gates on ready_state; RED 5: an unguarded lookahead wait; D4: the ring release cadence), the pair-skew detectors
(the four base levers + SOFTMAX_S_READ / EPILOGUE_DQ_READOUT, in the fresh-process watchdog form; the fp32_reduce arm on the two seam levers),
the probes' launch rows, the exchange_off-under-leader-skew child (the mechanism's repro pair), the STAMPS dump accounting row, and the
mask-form rows (the typed domain and the kernel codes, the EXACTNESS of `tile_may_mask` against the per-cell mask on host ints, the three
renderings' census, the fsel / predicated BITWISE twins on dK / dV (dQ bitwise on the order-free cells, by the fp64 gate beyond)).
Budget: the skew rows and the probe-skew child name ``skew_module`` (every row is a fresh child = cold import + JIT).
"""

import os
import pathlib
import re
import subprocess
import sys

import pytest
import torch

from d128_2cta_test_utils import _CELLS, _CENSUS as _CENSUS_BASE, _CHILD_TIMEOUT_S, _EXPECTED_BASE, _GLOBAL_PREFIX, _LEVER_LINE, _SKV, _WATCHDOG_S
from d128_2cta_test_utils import SITES_BASE, _adapter_class, _require_2cta, _setmaxnreg_of, make_adapter_2cta, render_2cta
from d128_grouped_test_utils import adapter_arguments, check_case, make_case, make_workspace, poison
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

_MASK_FORMS = ("fsel", "predicated", "diag_mask_off")  # None = the default "fsel" (the keep-word mask selected on every tile)
# The MASK FORM census (the mask-form lever; rendered beside the module census): the per-cell selects of the keep-word mask (arith.select -> selp.f32: the body's
# CFG.MASK_SELECTS_PER_LANE_PER_TILE = 64 plus two others), the keep-word upper-edge shifts (keep_below_word -> shr.u32: CFG.MASK_WORDS_PER_LANE_PER_TILE
# = 2 of the body's), the branches (the predicated form adds ONE CTA-uniform `if mask_tile:` per q quarter = CFG.QUARTERS_PER_GROUP).
_MASK_CENSUS = {"selp_f32": r"selp\.f32", "shr_u32": r"shr\.u32", "bra": r"\bbra(\.uni)?\s"}
# dQ is a sum of per-pair fp32 reduce-adds onto zero: with at most TWO contributions per element the result is order-free (a + b == b + a), with
# three or more it is not (fp32 addition is not associative) -- the cells where the dQ twin is a BITWISE claim (G09: name the cells).
_DQ_ORDER_FREE_PAIRS = 2
# The dQ staging census (rendered beside the module census): the sDQ ring's fp32 slice STS (the fp32 16-B rows lower to `st.shared.v2.b64`, the
# only ones of the body: 3 _dq_round sites x DQ_BOX // 4 = 24), their proxy fences, the bulk commit groups (4 per-use + 1 item-end).
_ROUTE_CENSUS = {
    "sts_dq_v2_b64": r"st\.shared\.v2\.b64",
    "fence_proxy_async": r"fence\.proxy\.async",
    "bulk_commit_group": r"cp\.async\.bulk\.commit_group",
}
_BODY_MODULE = "cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_grouped_2cta_stail_f16"
_BODY_REL = pathlib.Path("python/cudnn/sdpa/bwd/kernels/sm100/bprop_d128_accum_grouped_2cta_stail_f16.py")
_TREE = pathlib.Path(__file__).resolve().parents[4]  # test/python/sdpa/frost/<file> -> the checkout
_RECORD = pathlib.Path(__file__).resolve().parent / "renderings" / "md5_d128_accum_grouped_2cta_stail_default_sm_100a.txt"
# The probes' purity census: the stats / delta LDS.128 (`ld.shared.v4.b32`: stats_one_chunk removes exactly CFG.DIAG_STATS_LDS_REMOVED_PER_LANE
# static sites), the expect_tx arms (exchange_off keeps the count), the %clock64 reads (0 unless `debug_clk`), and the three counts every probe
# must leave untouched: `bar.warp.sync`, `tcgen05.fence::after_thread_sync`, `mapa` (exchange_off loses the ship's own two).
_PROBE_CENSUS = {
    "lds_v4_b32": r"ld\.shared\.v4\.b32",
    "expect_tx": r"mbarrier\.arrive\.expect_tx",
    "clock64": r"%clock64",
    "bar_warp_sync": r"bar\.warp\.sync",
    "tcgen05_fence_after": r"tcgen05\.fence::after_thread_sync",
    "mapa": r"mapa\.",
}
_CENSUS = dict(_CENSUS_BASE, entry_stail=r"\.entry\s+cudnn_kernel_bprop_d128_accum_grouped_2cta_stail_kernel", **_PROBE_CENSUS)
# The body's census against the base counts (kernel header section 5 / 11): dQ is issued at TWO sites (in-tile dQ(t-1) + the tail dQ(hi-1):
# 16 + 16 k-steps -> mma 56 + 16 = 72), the tail's dq_done commit (+1), the arrive sites 41 - 4 (a dq_empty bootstrap + two per-slot sites of a
# 4-slice ring -> 2 slots x 2 rounds) + 1 (dq_full per round, 3 sites) + 1 (s_read) + 1 (s_tail_empty) = 40, the wait sites 63 - 2 (dq_empty:
# 3 rounds x 2 slots vs 4 + 4 terminal) - 1 (the q1 drain, one stage) + 1 (s_read) + 4 (s_tail_empty: prologue, lookahead, tail, drain) + 3 (the
# tail's dS gates) = 68.  The entry is this body's kernel.
_EXPECTED = dict(
    _EXPECTED_BASE,
    mma_cta_group_2=72,
    commit_multicast=13,
    mbarrier_arrive=40,
    try_wait=68,
    entry=0,
    entry_stail=1,
    lds_v4_b32=38,  # 32 stats / delta chunks (CFG.DIAG_STATS_LDS_PER_LANE) + 6 other 16-B shared loads
    expect_tx=11,  # the 8 tensor rows (do1 x 2) + ds_recv_full + the scheduler's local and remote arm
    clock64=0,
    bar_warp_sync=67,
    tcgen05_fence_after=32,
    mapa=17,
)
_PROBES = ("dq_reduce_add_off", "exchange_off", "stats_one_chunk", "sdq_sts_off")  # CFG.DIAG_2CTA_NAMES_ON_MAP (asserted in the config-truth row)


def _probe_census_delta(probe, c, default):
    """What each probe removes from the DEFAULT rendering's census and nothing else (the arrive / try_wait / bar.warp.sync / tcgen05.fence counts
    are identical by construction: a probe moves no barrier row); the counts come from the config, never a literal."""
    return {
        "dq_reduce_add_off": {"reduce_add": 0},  # dq_reduce_add_off: the four reduce-add ISSUES (the per-round dq_empty arrives stay)
        "exchange_off": {
            "dsmem_ship": 0,
            "mapa": default["mapa"] - 2,
        },  # exchange_off: the ONE bulk copy + its two mapa (dst pointer, peer mbar); the arm's tx operand
        "stats_one_chunk": {"lds_v4_b32": default["lds_v4_b32"] - c.DIAG_STATS_LDS_REMOVED_PER_LANE},  # stats_one_chunk: 28 of the 32 stats LDS.128
        "sdq_sts_off": {
            "sts_dq_v2_b64": 0,
            "fence_proxy_async": default["fence_proxy_async"] - c.DQ_ROUNDS - 1,
        },  # sdq_sts_off: the 3 _dq_round STS sites + fences
    }[probe]


def make_adapter_2cta_stail(case, *, compile_plan=True, declaration=None, **fields):
    """The suite's adapter construction on the 2-CTA subclass (the pair body at any S) plus any typed field set on the instance before
    ``check_support()`` (``diag_2cta`` / ``mask_form`` / ``warp_roles`` / ``dkv_accum`` / ``debug_clk``)."""
    return make_adapter_2cta(case, compile_plan=compile_plan, declaration=declaration, **fields)


def _run_case(case, repeats=2, **fields):
    api = make_adapter_2cta_stail(case, **fields)
    workspace = make_workspace(api.scratch_workspace_bytes())
    for _ in range(repeats):
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
    return api, workspace


def _body_source():
    path = _TREE / _BODY_REL
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    return path.read_text()


def _body_module():
    """The second body's module (host import: its constants and the plain-Python `tile_may_mask`; no device)."""
    import importlib

    _body_source()
    return importlib.import_module(_BODY_MODULE)


def _function_code(source, name, next_name):
    """The CODE lines of ``def <name>`` up to ``def <next_name>`` with the docstring and the comment lines stripped (the punch lists repeat the
    site names in prose; a source assertion reads the statements only)."""
    start = source.index(f"def {name}(")
    end = source.index(f"def {next_name}(", start)
    body = source[start:end]
    doc = re.search(r'"""[\s\S]*?"""', body)
    if doc:
        body = body[: doc.start()] + body[doc.end() :]
    return [ln for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith("@")]


# ------------------------------------------------------------------------------------------------------------ host (no GPU)
def test_2cta_stail_config_is_the_truth():
    """Every number the second body's header quotes is a derived, asserted field of ``config_sm100_d128_accum_2cta_stail.CFG`` (G10 config-as-truth):
    the slab chain (Q1 one stage, the dedicated sDQ ring at 212992, STATS / DELTA / scaffold unchanged), the TMEM map (dQ at S^T's dead tail),
    the ring depths, the 35-row BAR_LAYOUT (kinds, counts, END 230960 with the fp32_reduce row), the three alias rows, the probe domain.
    """
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta as first
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod

    c = cfgmod.CFG
    assert c.TMEM_MAP == "s_tail"
    assert (c.Q1_STAGES, c.DQ_SLOTS, c.DQ_ROUNDS, c.DQ_SLICES) == (1, 2, 2, 4) and c.DQ_SLOTS * c.DQ_ROUNDS == c.DQ_SLICES
    assert (c.Q1, c.Q2, c.DO1, c.DO2, c.K1, c.K2, c.V, c.DS, c.XCHG, c.DQ, c.STATS, c.DELTA, c.SLABS_END) == (
        0,
        16384,
        32768,
        49152,
        65536,
        98304,
        131072,
        163840,
        196608,
        212992,
        229376,
        229888,
        230400,
    )
    assert (c.DQ_BYTES, c.SMEM_BYTES, c.BARS, c.TMEM_HANDLE) == (16384, first.CFG.SMEM_BYTES, 230400, 231344)
    assert [c.dq_slot(s) for s in range(c.DQ_SLOTS)] == [212992, 221184] and [c.dq_use(u) for u in range(c.DQ_SLICES)] == [(0, 0), (1, 2), (0, 1), (1, 3)]
    assert (c.S_COL, c.P_COL, c.DQ_COL, c.DQ_COLS, c.DP_COL, c.DS_COL, c.TMEM_DEAD_COL, c.DV_COL, c.DK_COL, c.TMEM_COLS) == (
        0,
        0,
        64,
        64,
        128,
        128,
        192,
        256,
        384,
        512,
    )
    assert first.CFG.DQ_COL == 192 and c.DQ_COL == first.CFG.S_COL + first.CFG.PACK_COLS  # the image sits in S^T's dead tail, not dP^T's dead half
    # 35 rows / 40 slots: + dvk_stage_empty (THREAD 32, the fp32 dK / dV rounds' WAR release; init only under dkv_fp32)
    assert len(cfgmod.BAR_LAYOUT) == 35 and sum(r[2] for r in cfgmod.BAR_LAYOUT) == 40 and cfgmod.BAR_LAYOUT_END == 230960
    assert cfgmod.BAR_LAYOUT[-1][0] == "dvk_stage_empty" and cfgmod.BAR_LAYOUT[-1][3] == cfgmod.CFG.STORE_LANES
    counts = {row[0]: row[3] for row in cfgmod.BAR_LAYOUT}
    assert counts["s_read"] == c.PAIR_SOFTMAX_LANES == 512 and counts["s_tail_empty"] == c.PAIR_EPILOGUE_LANES == 256
    assert counts["dp_empty"] == 256 and counts["dq_empty"] == c.STORE_LANES == 32 and counts["dq_full"] == c.DQ_SLICE_LANES == 64 and counts["dv_full"] == 160
    assert cfgmod.BAR_KIND_COUNTS == {"tma_leader": 7, "mma": 9, "thread": 9, "leader": 6, "peer": 2, "tma_dsmem": 1, "tma": 1}
    assert [a[0] for a in cfgmod.ALIASES] == ["dk_store", "dv_store", "ds_recv(rank)"]  # the dq_staging alias is deleted
    assert c.DIAG_2CTA_NAMES_ON_MAP == _PROBES == c.DIAG_2CTA_NAMES == first.CFG.DIAG_2CTA_NAMES
    # the probes' constants and the STAMPS record (config = truth; the kernel reads DBG_CLK, never a word index)
    assert (c.DIAG_EXCHANGE_OFF_TX, c.DIAG_STATS_CHUNK, c.DIAG_STATS_CHUNKS_PER_QUARTER, c.DIAG_STATS_LDS_PER_LANE, c.DIAG_STATS_LDS_REMOVED_PER_LANE) == (
        0,
        0,
        8,
        32,
        28,
    )
    d = cfgmod.DBG_CLK
    assert (d.SMEM, d.WORDS, d.BYTES, d.LEADER_BLOCKED_STRIDE, len(d.WORD), d.BLOCKED_THRESH, d.HIST_EDGES) == (
        230960,
        96,
        384,
        17,
        82,
        128,
        (100, 400, 1600),
    )  # the record = the 384-B spare left by the fp32_reduce row
    assert d.SMEM == cfgmod.BAR_LAYOUT_END and d.SMEM + d.BYTES <= c.TMEM_HANDLE and c.DEBUG_CLK_DEFAULT is False
    assert (d.TAG_BACK_IN_TILE, d.TAG_BACK_TAIL, d.TAG_BACK_RELAY) == (2, 1, 1) and set(d.ROLE_WAITS) == {
        "leader",
        "softmax",
        "epilogue",
        "follower",
        "loader",
        "store",
    }
    assert (
        cfgmod.dbg_clk_records(2, 8, 512) == 2 * 2 * 8 * 2
        and cfgmod.dbg_clk_bytes(2, 8, 512) == 64 * cfgmod.DBG_CLK.BYTES
        and cfgmod.dbg_clk_region_offset(1000) == 1024
    )
    cfgmod.validate_config()
    first.validate_config()


def test_2cta_stail_fields_are_typed():
    """The typed fields decline BY NAME: ``diag_2cta`` resolves the four probes of the code space and nothing else (two probes of an earlier pair
    body are outside the domain); ``debug_clk`` is a bool (False by default); the adapter readers prefix the engine name; the record defaults are
    None / False; the pruned perf knobs of the measurement measurement series are not fields of the adapter (one body, one order, one load form)."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta as base
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import debug_clk, diag_2cta

    assert base.resolve_diag_2cta(None) is None and [base.resolve_diag_2cta(p) for p in _PROBES] == list(_PROBES)
    assert [base.diag_2cta_code(p) for p in _PROBES] == [1, 2, 3, 4] and base.diag_2cta_code(None) == 0
    for bad in ("dp_empty_wait_off", "ds_gates_off", "bogus", "", 1):
        with pytest.raises(ValueError, match="diag_2cta must be None or one of"):
            base.resolve_diag_2cta(bad)
    assert diag_2cta(SimpleNamespace(_NAME="x")) is None and diag_2cta(SimpleNamespace(_NAME="x", diag_2cta="exchange_off")) == "exchange_off"
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: diag_2cta must be None or one of"):
        diag_2cta(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", diag_2cta="dp_empty_wait_off"))
    # the STAMPS lever: a typed bool, False by default; the reader prefixes the engine name
    assert cfgmod.resolve_debug_clk(None) is False and cfgmod.resolve_debug_clk(True) is True and cfgmod.resolve_debug_clk(False) is False
    assert debug_clk(SimpleNamespace(_NAME="x")) is False and debug_clk(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", debug_clk=True)) is True
    for bad_flag in (1, "True", 0, ""):
        with pytest.raises(ValueError, match="must be a bool"):
            cfgmod.resolve_debug_clk(bad_flag)
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: debug_clk must be a bool"):
        debug_clk(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", debug_clk=1))
    assert SdpaBwdDslSm100D128.debug_clk is False and SdpaBwdDslSm100D128.diag_2cta is None  # the record defaults
    for pruned in ("tmem_map", "pair_mma_order", "dq_route_pair", "tma_load_form", "store_wait_form", "reg_split", "datapath_2cta"):
        assert not hasattr(SdpaBwdDslSm100D128, pruned), pruned
    for pruned in ("PAIR_MMA_ORDERS", "DQ_ROUTES_PAIR", "TMA_LOAD_FORMS", "STORE_WAIT_FORMS", "REG_SPLITS", "REG_SPLIT_NAMES"):
        assert not hasattr(cfgmod.CFG, pruned) and not hasattr(base.CFG, pruned), pruned


def test_2cta_stail_mask_form_is_typed():
    """``mask_form`` is a typed field: ``None`` = "fsel" (the shipped rendering); "predicated" and the DIAGNOSTIC "diag_mask_off" are the other two
    values; anything else declines BY NAME; the kernel's constants are the config's code space (config = truth); the adapter reader prefixes the
    engine name; the record default is None."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import mask_form

    c = cfgmod.CFG
    assert c.MASK_FORMS == _MASK_FORMS and c.MASK_FORM_DEFAULT == "fsel" and c.MASK_FORM_DIAGNOSTIC == "diag_mask_off"
    assert cfgmod.MASK_FORM_CODES == {"fsel": 0, "predicated": 1, "diag_mask_off": 2} and tuple(cfgmod.MASK_FORM_CODES) == c.MASK_FORMS
    assert (c.MASK_SELECTS_PER_LANE_PER_TILE, c.MASK_WORDS_PER_LANE_PER_TILE, c.MASK_WORD_COLS) == (64, 2, 32)
    assert cfgmod.resolve_mask_form(None) == "fsel"
    for form, code in (("fsel", 0), ("predicated", 1), ("diag_mask_off", 2)):
        assert cfgmod.resolve_mask_form(form) == form and cfgmod.mask_form_code(form) == code
    assert cfgmod.mask_form_code(None) == 0
    for bad_form in ("bogus", "", "FSEL", 1):
        with pytest.raises(ValueError, match="mask_form must be one of"):
            cfgmod.resolve_mask_form(bad_form)
        with pytest.raises(ValueError, match="mask_form must be one of"):
            cfgmod.mask_form_code(bad_form)
    assert mask_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", mask_form="predicated")) == "predicated"
    assert mask_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", mask_form="diag_mask_off")) == "diag_mask_off"
    assert mask_form(SimpleNamespace(_NAME="x")) == "fsel"
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: mask_form must be one of"):
        mask_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", mask_form="pred"))
    assert SdpaBwdDslSm100D128.mask_form is None  # the record default: today's rendering
    body = _body_module()
    assert (body.MASK_FORM_FSEL, body.MASK_FORM_PREDICATED, body.MASK_FORM_DIAG_OFF) == tuple(cfgmod.MASK_FORM_CODES[f] for f in _MASK_FORMS)


def test_2cta_stail_tile_mask_predicate_is_exact_against_the_per_cell_mask():
    """Lever the mask-form lever's interior-tile skip is bitwise ONLY if `tile_may_mask` is never FALSE on a tile that holds a masked cell (kernel header section 7).
    The kernel's own plain-Python helper runs here on host ints against the per-cell mask _p_values applies -- keep column c of the lane with key k iff
    c >= k under causal and c < (seq_q if k < seq_k else 0) -- over EVERY cell of every (pair, rank, q tile) the roles visit (lo = 2 pair under causal,
    hi = ceil(seq_q / N)) on a grid of sequence lengths covering the THD edges (empty, shorter than one tile, partial tiles on both axes, exact
    multiples, a dead second half, a dead pair past seq_k): sound (FALSE -> no masked cell) AND tight (TRUE -> a masked cell exists), both causal values."""
    body = _body_module()
    c = body.CFG
    m_rows, n_cols, pair_rows = c.M, c.N, c.PAIR_KV_ROWS
    lengths = (0, 1, 3, 65, 127, 128, 129, 200, 255, 256, 257, 300, 383, 384, 385, 511, 512, 513, 700, 1000, 1024)
    rows = torch.arange(m_rows).view(-1, 1)
    cols = torch.arange(n_cols).view(1, -1)
    checked = {True: 0, False: 0}
    for causal in (True, False):
        for seq_q in lengths:
            for seq_k in lengths:
                hi_tile = (seq_q + n_cols - 1) // n_cols
                for pair in range((seq_k + pair_rows - 1) // pair_rows + 1):  # + one dead pair past seq_k
                    for rank in range(c.CGA_M):
                        key = pair * pair_rows + rank * m_rows + rows
                        hi = torch.where(key < seq_k, torch.tensor(seq_q), torch.tensor(0))
                        for qtile in range(pair * c.Q_TILES_PER_PAIR if causal else 0, hi_tile):
                            col = qtile * n_cols + cols
                            masked = col >= hi
                            if causal:
                                masked = masked | (col < key)
                            truth = bool(masked.any())
                            assert body.tile_may_mask(pair, rank, qtile, seq_q, seq_k, causal) == truth, (causal, seq_q, seq_k, pair, rank, qtile, truth)
                            checked[truth] += 1
    assert checked[True] > 100 and checked[False] > 100, checked  # both verdicts exercised many times


def _render(tmp_path, form_kw, extra_census=None):
    """Host trace-compile of the 2-CTA chain for sm_100a in a fresh process with THIS module's census (plus ``extra_census`` rows) -> (PTX md5,
    census, setmaxnreg pairs)."""
    return render_2cta(tmp_path, form_kw, dict(_CENSUS, **(extra_census or {})))


def _record_md5(path, body_token):
    dsl, md5 = None, None
    for ln in path.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(rf"{body_token} sm_100a default rc=0 ptx_md5=([0-9a-f]{{32}})", ln)
        if m:
            md5 = m.group(1)
    return dsl, md5


def test_2cta_stail_host_renders_for_sm_100a_and_matches_its_record(tmp_path):
    """Host trace-compile of the chain with every field unset (no device): the body's PTX census is the header's (sections 5 / 11), no local
    memory, no nanosleep, the 144 / 144 / 80 register split; the md5 matches the committed record at this geometry (skips, never widens, on
    another DSL build)."""
    from cudnn.frost.buffers import cutedsl_state

    md5, census, regs = _render(tmp_path, {})
    print(f"\n2-CTA s_tail sm_100a: PTX md5 {md5} census {census} setmaxnreg {regs}")
    assert census == _EXPECTED, census
    assert sorted(regs) == _setmaxnreg_of(), regs
    assert _RECORD.is_file(), _RECORD
    dsl, want = _record_md5(_RECORD, "d128_accum_grouped_2cta_stail")
    assert dsl and want, (dsl, want)
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: PTX text differs by DSL build, re-render the record")
    assert md5 == want, f"PTX md5 {md5} != the record's {want} -- the s_tail rendering changed"


@pytest.mark.parametrize("probe", _PROBES)
def test_2cta_stail_host_renders_probes_for_sm_100a(tmp_path, probe):
    """``diag_2cta = <probe>`` removes exactly its own ISSUE (dq_reduce_add_off: the four reduce-adds), TRANSFER (exchange_off: the one bulk copy +
    its two mapa, the arm keeping its count with a 0-byte operand) or BYTES (stats_one_chunk: 28 of the 32 stats LDS.128; sdq_sts_off: the 24
    sDQ slice STS + their 3 proxy fences) and nothing else -- every arrive and wait site, every bar.warp.sync and tcgen05.fence stays (the barrier
    table is identical: the purity census); the probe renders a different md5 from the default."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta as base
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod

    md5_d, census_d, _ = _render(tmp_path, {}, _ROUTE_CENSUS)
    assert census_d == dict(_EXPECTED, sts_dq_v2_b64=24, fence_proxy_async=15, bulk_commit_group=5), census_d
    md5, census, _ = _render(tmp_path, {"diag_2cta": probe}, _ROUTE_CENSUS)
    print(f"\n2-CTA s_tail sm_100a PTX census ({probe}): md5 {md5} {census}")
    assert census == dict(census_d, **_probe_census_delta(probe, cfgmod.CFG, census_d)), (probe, census)
    assert md5 != md5_d, f"{probe} must render different PTX from the default"
    with pytest.raises(ValueError, match="diag_2cta must be None or one of"):  # a probe of an earlier pair body is outside the domain: never rendered
        base.resolve_diag_2cta("ds_gates_off")


def test_2cta_stail_probes_resolve_the_same_way_under_fp32_dkv_accum(tmp_path):
    """fp32_reduce x the probes: the fp32 dK / dV staging of ``dkv_accum = "fp32_reduce"`` (header section 11) lives in the dS slab halves (own +
    recv) and XCHG, slots that rest on the three dS gates -- a probe that skipped those gates is outside the domain under BOTH field values
    (asserted here at compile_host, the chain's entry, exactly as the probe arms reach it).  ``exchange_off`` x ``fp32_reduce`` renders like every
    other probe: OUTPUT WRONG BY DESIGN (dQ: the recv half is never written), a different md5 from the fp32 default under the SAME mbarrier
    arrive / wait census with exchange_off's own transfer delta only (the arm's bytes go to 0, the one bulk copy + its two mapa gone; no barrier
    row moves; nothing asynchronous writes the recv-half slots under the probe).  The combination is reachable through the typed fields only;
    the ``--validate`` refusal is per probe name."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_2cta_host import compile_host

    for accum in (None, "fp32_reduce"):
        with pytest.raises(ValueError, match="diag_2cta must be None or one of"):
            compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, diag_2cta="ds_gates_off", dkv_accum=accum)
    md5_fp32, census_fp32, _ = _render(tmp_path, {"dkv_accum": "fp32_reduce"}, _ROUTE_CENSUS)
    md5, census, _ = _render(tmp_path, {"dkv_accum": "fp32_reduce", "diag_2cta": "exchange_off"}, _ROUTE_CENSUS)
    print(f"\n2-CTA s_tail sm_100a PTX census (fp32_reduce x exchange_off): md5 {md5} {census}")
    assert md5 != md5_fp32, "exchange_off x fp32_reduce must render different PTX from the fp32 default"
    assert census == dict(census_fp32, **_probe_census_delta("exchange_off", cfgmod.CFG, census_fp32)), (census_fp32, census)


def test_2cta_stail_fields_under_16w_fa4_decline_or_resite_by_name(tmp_path):
    """The alternate role shape x the other fields: under ``warp_roles = "16w_fa4"`` the STAMPS lever ``debug_clk`` and ``dkv_accum = "fp32_reduce"``
    DECLINE BY NAME at compile_host (the record layout names the 16w roles' waits; the fp32 staging rounds live in the 16w epilogue loop), and the
    fp32 row ``dvk_stage_empty`` is UNSERVED there (absent from ``BAR_LAYOUT_XFA4`` / ``BarsXFA4``); ``exchange_off`` is sited at the softmax's arm +
    ship under 16w_fa4 and renders with its own transfer delta only (the one bulk copy + its two mapa gone, the arm's bytes 0; every arrive / wait
    stays), a different md5 from the 16w_fa4 default; both fields pass under the default shape (16w)."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_2cta_host import compile_host

    with pytest.raises(ValueError, match="debug_clk is not defined under warp_roles '16w_fa4'"):
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, warp_roles="16w_fa4", debug_clk=True)
    with pytest.raises(ValueError, match="dkv_accum 'fp32_reduce' is not built under warp_roles '16w_fa4'"):
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, warp_roles="16w_fa4", dkv_accum="fp32_reduce")
    assert cfgmod.validate_fields_for_roles("16w", None, debug_clk=True, dkv_accum="fp32_reduce") == "16w"
    assert cfgmod.UNSERVED_BARRIERS_XFA4 == ("dvk_stage_empty",) and "dvk_stage_empty" not in {row[0] for row in cfgmod.BAR_LAYOUT_XFA4}
    assert "dvk_stage_empty" in {row[0] for row in cfgmod.BAR_LAYOUT} and "dvk_stage_empty" not in _body_module().BarsXFA4._fields
    md5_fa4, census_fa4, _ = _render(tmp_path, {"warp_roles": "16w_fa4"}, _ROUTE_CENSUS)
    md5, census, _ = _render(tmp_path, {"warp_roles": "16w_fa4", "diag_2cta": "exchange_off"}, _ROUTE_CENSUS)
    print(f"\n2-CTA s_tail sm_100a PTX census (16w_fa4 x exchange_off): md5 {md5} {census}")
    assert md5 != md5_fa4, "exchange_off x 16w_fa4 must render different PTX from the 16w_fa4 default"
    assert census == dict(census_fa4, **_probe_census_delta("exchange_off", cfgmod.CFG, census_fa4)), (census_fa4, census)


def test_2cta_stail_host_renders_debug_clk_for_sm_100a(tmp_path):
    """``debug_clk = True`` (the STAMPS lever, header section 11) renders the %clock64 brackets (and the two %globaltimer body stamps) on top of the
    SAME barrier census: 40 arrive / 68 try_wait / 11 expect_tx / 67 bar.warp.sync / 32 tcgen05.fence sites, the ship, the reduce-adds, the STS,
    no local memory, no nanosleep, the default register split -- under a different md5; the default (False) rendering is the record's."""
    md5_on, census_on, regs_on = _render(tmp_path, {"debug_clk": True})
    print(f"\n2-CTA s_tail debug_clk sm_100a: PTX md5 {md5_on} census {census_on} setmaxnreg {regs_on}")
    # the lever adds the clock reads and the record dump's SMEM reads (the 100-word copy at exit vectorizes into 16-B loads); nothing else moves
    assert census_on["clock64"] > 0 and census_on["lds_v4_b32"] >= _EXPECTED["lds_v4_b32"], census_on
    assert census_on == dict(_EXPECTED, clock64=census_on["clock64"], lds_v4_b32=census_on["lds_v4_b32"]), census_on
    assert sorted(regs_on) == _setmaxnreg_of(), regs_on
    md5_off, census_off, _ = _render(tmp_path, {"debug_clk": False})
    assert census_off == _EXPECTED and md5_off != md5_on, "debug_clk=False is the default rendering; True must differ (the clock reads)"
    dsl, want = _record_md5(_RECORD, "d128_accum_grouped_2cta_stail")
    assert md5_off == want or dsl, "debug_clk=False renders the record's md5 (a DSL build difference is a skip, not a widen)"


# ------------------------------------------------------------------------------------------------------------ host: the audit's deterministic-hang topologies as SOURCE assertions
def test_2cta_stail_host_renders_mask_forms_for_sm_100a(tmp_path):
    """Lever the mask-form lever's three mask forms on this body (header section 11): "fsel" named explicitly IS the default rendering (the record's md5); "predicated"
    renders a DIFFERENT PTX under the SAME mbarrier census (no wait, arrive or fence moves), the same setmaxnreg split, the SAME number of per-cell
    selects and keep-word shifts (the mask body is kept, inside the branch) and exactly CFG.QUARTERS_PER_GROUP more branches (one CTA-uniform
    `if mask_tile:` per q quarter); "diag_mask_off" drops exactly CFG.MASK_SELECTS_PER_LANE_PER_TILE selects and CFG.MASK_WORDS_PER_LANE_PER_TILE
    upper-edge shifts (the whole mask body) under the same census; the three md5s differ."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG

    md5_d, census_d, regs_d = _render(tmp_path, {}, _MASK_CENSUS)
    md5_f, census_f, regs_f = _render(tmp_path, {"mask_form": "fsel"}, _MASK_CENSUS)
    md5_p, census_p, regs_p = _render(tmp_path, {"mask_form": "predicated"}, _MASK_CENSUS)
    md5_o, census_o, regs_o = _render(tmp_path, {"mask_form": "diag_mask_off"}, _MASK_CENSUS)
    print(
        f"\n2-CTA s_tail mask forms sm_100a: default {md5_d} fsel {md5_f} predicated {md5_p} diag_mask_off {md5_o}; census {census_d} | {census_p} | {census_o}"
    )
    assert md5_f == md5_d and census_f == census_d and regs_f == regs_d, "fsel named explicitly is the default rendering"
    dsl, want = _record_md5(_RECORD, "d128_accum_grouped_2cta_stail")
    have = __import__("cutlass").__version__ if hasattr(__import__("cutlass"), "__version__") else None
    if dsl is not None and have is not None and dsl.endswith(have):
        assert md5_d == want, f"the default (fsel) rendering {md5_d} != the record's {want}: the mask-form chain must leave the default byte-identical"
    assert len({md5_d, md5_p, md5_o}) == 3, "the three mask forms must render three different PTX texts"
    for census in (census_d, census_p, census_o):
        assert {k: v for k, v in census.items() if k not in _MASK_CENSUS} == _EXPECTED, census  # no barrier, TMA or TMEM op moves under any form
    assert sorted(regs_d) == sorted(regs_p) == sorted(regs_o) == _setmaxnreg_of(), (regs_d, regs_p, regs_o)
    assert census_p["selp_f32"] == census_d["selp_f32"] and census_p["shr_u32"] == census_d["shr_u32"], (census_d, census_p)
    assert census_p["bra"] - census_d["bra"] == CFG.QUARTERS_PER_GROUP, (census_d["bra"], census_p["bra"])
    assert census_d["selp_f32"] - census_o["selp_f32"] == CFG.MASK_SELECTS_PER_LANE_PER_TILE, (census_d, census_o)
    assert census_d["shr_u32"] - census_o["shr_u32"] == CFG.MASK_WORDS_PER_LANE_PER_TILE, (census_d, census_o)


def test_2cta_stail_dp_empty_is_arrived_at_the_ds_readout_before_the_recv_arm():
    """RED 3: dp_empty arrived after the dQ readout under the rotated order is a deterministic HANG (dP^T(t+1) waits a readout of a dQ issued
    after it).  The epilogue's ONE dp_empty arrive precedes the ds_recv_full arm and the dQ readout."""
    code = _function_code(_body_source(), "_epilogue_loop", "_mma_loop_leader")
    arrive = [i for i, ln in enumerate(code) if "_arrive_leader(bars.dp_empty)" in ln]
    recv = [i for i, ln in enumerate(code) if "bars.ds_recv_full.arrive(" in ln]
    dq_ld = [i for i, ln in enumerate(code) if "_tmem_load(tmem_base + CFG.DQ_COL" in ln]
    assert len(arrive) == 1 and len(recv) == 1 and len(dq_ld) == 1, (arrive, recv, dq_ld)
    assert arrive[0] < recv[0] < dq_ld[0], "dp_empty must be arrived right after the dS^T readout, before the XCHG arm, never after the dQ readout"
    s_tail = [i for i, ln in enumerate(code) if "_arrive_leader(bars.s_tail_empty)" in ln]
    assert len(s_tail) == 1 and dq_ld[0] < s_tail[0], "s_tail_empty is arrived right after the dQ readout"


def test_2cta_stail_ds_gates_ride_the_xchg_state():
    """RED 4: the three dS gates waited on ready_state (the P2 split missing) hang at tile lo+1 of the first N >= 2 item.  Every ds_smem /
    ds_recv_full / ds_pair_ready wait of the leader names ``xchg_state.phase`` (one helper ``_dq_issue``, called from the in-tile slot helper
    ``_dq_slot`` and at the tail), never ready_state.  The in-tile slot is ONE helper at ONE site (before dV(t)); the s_read wait lives inside it,
    every tile, and nowhere else in the leader."""
    source = _body_source()
    gates = re.findall(r"_wait_tcgen\(bars\.(ds_smem|ds_recv_full|ds_pair_ready), (\w+)\.phase, dbg, _W\[\"wait:leader:\w+\"\]\)", source)
    assert sorted(g[0] for g in gates) == ["ds_pair_ready", "ds_recv_full", "ds_smem"], gates
    assert {g[1] for g in gates} == {"xchg_state"}, gates
    issue = _function_code(source, "_dq_issue", "_dq_slot")
    # the three gates are UNCONDITIONAL (no const_expr guard: they are the ds_recv_full arm's only back-edge); the trace-time branches in the
    # helper are the STAMPS handshake, the test-only skew lever and the dq_war_done commit of the alternate role shape; none guards a gate
    assert [ln.strip() for ln in issue if ln.strip().startswith("if ")] == [
        "if cutlass.const_expr(dbg is not None):",
        "if cutlass.const_expr(DEBUG_SOFTMAX_DQ_WAR_DELAY_US > 0):",
        "if cutlass.const_expr(fa4_roles):",
    ], issue
    gate_lines = [ln for ln in issue if "_wait_tcgen(bars.ds_" in ln]
    assert len(gate_lines) == 3 and len({len(ln) - len(ln.lstrip()) for ln in gate_lines}) == 1 and "DIAG" not in "".join(issue), issue
    leader = _function_code(source, "_mma_loop_leader", "_mma_loop_follower")
    slots = [i for i, ln in enumerate(leader) if "_dq_slot(" in ln and "xchg_state" in ln]
    issues = [ln for ln in leader if "_dq_issue(" in ln and "xchg_state" in ln]
    assert len(slots) == 1 and len(issues) == 1, (slots, issues)  # the in-tile dQ(t-1) slot + the tail dQ(hi-1)
    dv = [i for i, ln in enumerate(leader) if "_ts(grad, tmem_base + CFG.P_COL" in ln]
    assert len(dv) == 1 and slots[0] < dv[0], "the dQ slot sits before dV(t)"
    assert not any("_wait_tcgen(bars.s_read" in ln for ln in leader), "s_read is waited inside the ONE slot helper only"
    slot = _function_code(source, "_dq_slot", "_dq_round")
    assert (
        sum('_wait_tcgen(bars.s_read, ready_state.phase, dbg, _W["wait:leader:s_read"])' in ln for ln in slot) == 1
        and sum("_dq_issue(" in ln for ln in slot) == 1
    ), slot
    assert any(ln.strip() == "if qtile > lo:" for ln in slot), "the in-tile issue is guarded (no dQ at tile lo)"
    assert not any("ready_state" in ln and ("ds_smem" in ln or "ds_recv_full" in ln or "ds_pair_ready" in ln) for ln in leader)


def test_2cta_stail_s_tail_empty_waits_sit_at_the_three_guarded_sites():
    """RED 5 (D1): table 3's form (a wait before EVERY S^T) parks tile lo's lookahead on the readout of a dQ not yet issued.  The body's four
    s_tail_empty waits are: the prologue S^T(lo) (unguarded inside `if lo < hi`), the lookahead under `if qtile > lo`, the tail under
    `if hi - 1 > lo`, and the post-loop drain; the lookahead's and the tail's advance sit with their waits."""
    source = _body_source()
    wait_s_tail = r'_wait_tcgen\(bars\.s_tail_empty, s_tail_state\.phase, dbg, _W\["wait:leader:s_tail_empty"\]\)'
    waits = [m.start() for m in re.finditer(r"^\s+" + wait_s_tail, source, re.M)]
    assert len(waits) == 4, len(waits)
    assert re.search(r"if qtile > lo:\n\s+" + wait_s_tail + r"\n\s+s_tail_state = advance\(s_tail_state, 1\)", source)
    assert re.search(r"if hi - 1 > lo:\n\s+" + wait_s_tail + r"\n\s+s_tail_state = advance\(s_tail_state, 1\)", source)
    leader = _function_code(source, "_mma_loop_leader", "_mma_loop_follower")
    prologue = [i for i, ln in enumerate(leader) if "_wait_tcgen(bars.s_tail_empty" in ln]
    s_lo = [i for i, ln in enumerate(leader) if "mma_ss(score, k1d, q1d, s_acc)" in ln]
    assert prologue and s_lo and prologue[0] < s_lo[0], "the prologue S^T(lo) waits s_tail_empty (the previous item's tail readout)"
    assert leader[-1].strip() == '_wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])', leader[-1]  # the P15 drain: last
    softmax = _function_code(source, "_softmax_loop", "_epilogue_loop")
    s_read = [ln for ln in softmax if "_arrive_leader(bars.s_read)" in ln]
    s_done = [ln for ln in softmax if "_wait_tcgen(bars.s_done" in ln]
    assert len(s_read) == 1 and len(s_done) == 1 and len(s_read[0]) - len(s_read[0].lstrip()) == len(s_done[0]) - len(s_done[0].lstrip())  # D2: EVERY tile


def test_2cta_stail_dq_ring_release_cadence_is_the_d4_pair():
    """D4: the store's closed per-round release (tma_store_wait(1) -> slot 0, (0) -> slot 1) AND the epilogue's both-slots wait before each round
    are ONE decision; the streaming release with the both-slots wait form deadlocks at round B.  No dq_empty bootstrap, no terminal observation (D3)."""
    source = _body_source()
    store = _function_code(source, "_store_loop", "_scheduler_loop")
    release = [i for i, ln in enumerate(store) if "bars.dq_empty[s].arrive()" in ln]
    waits = [i for i, ln in enumerate(store) if "tma_store_wait(CFG.DQ_SLOTS - 1 - s)" in ln]
    assert len(release) == 1 and len(waits) == 1 and waits[0] < release[0], (waits, release)
    assert sum("dq_empty" in ln and ".arrive()" in ln for ln in store) == 1, "no bootstrap arrive"
    assert any("for r in cutlass.range_constexpr(CFG.DQ_ROUNDS)" in ln for ln in store)
    rnd = _function_code(source, "_dq_round", "_softmax_loop")
    assert any("for s in cutlass.range_constexpr(CFG.DQ_SLOTS)" in ln for ln in rnd) and any("_wait(bars.dq_empty[s], empty_state.phase" in ln for ln in rnd)
    epi = _function_code(source, "_epilogue_loop", "_mma_loop_leader")
    assert not any("dq_empty" in ln for ln in epi), "the epilogue waits dq_empty only inside _dq_round (both slots, per round)"
    assert sum("_dq_round(" in ln for ln in epi) == 3  # round B of dQ(t-1), round A of dQ(t), the item-end round B of dQ(hi-1)
    assert not any("empty_state = advance" in ln for ln in epi), "no item-end flip: 2N rounds == 2N releases per slot"


def test_2cta_stail_exchange_off_keeps_every_wait_and_fakes_the_bytes():
    """SOURCE assertion for `exchange_off` (the safe exchange ceiling): in the epilogue the ONE ds_recv_full arm keeps its elected lane and arms
    the config-selected byte count (CFG.DIAG_EXCHANGE_OFF_TX = 0 under the probe, XCHG_TX otherwise); the ONLY statement under the probe's guard
    is the `_ship_xchg` call (the transfer); the XCHG STS, its proxy fence, the CFG.XCHG_BARRIER named barrier, the own-half STS and every wait /
    arrive of the epilogue are unguarded; the three dS gates (the arm's back-edge) are unconditional in `_dq_issue` (asserted with RED 4 above);
    the relay in the follower is unconditional."""
    src = _body_source()
    epi = _function_code(src, "_epilogue_loop", "_mma_loop_leader")
    arms = [ln for ln in epi if "bars.ds_recv_full.arrive(" in ln]
    assert arms == [ln for ln in arms if "n_bytes=xchg_tx, pred=ship_pred" in ln] and len(arms) == 1, arms
    assert any(
        ln.strip() == "xchg_tx = CFG.DIAG_EXCHANGE_OFF_TX if cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF) else CFG.XCHG_TX" for ln in epi
    ), "the arm's byte count is the config-selected constant (0 under exchange_off)"
    guard = [i for i, ln in enumerate(epi) if ln.split("#")[0].strip() == "if cutlass.const_expr(not exchange_off):"]
    assert len(guard) == 1 and "_ship_xchg(slab, bars, rank, ship_pred)" in epi[guard[0] + 1], epi[guard[0] : guard[0] + 2]
    depth = len(epi[guard[0]]) - len(epi[guard[0]].lstrip())
    assert len(epi[guard[0] + 2]) - len(epi[guard[0] + 2].lstrip()) <= depth, "only the ship sits under the exchange_off guard"
    for token in (
        "_store_words(slab, CFG.XCHG",
        "named_barrier_fence(CFG.XCHG_BARRIER",
        "_store_words(slab, own_base",
        "bars.ds_smem.arrive()",
        "_arrive_leader(bars.dp_empty)",
    ):
        lines = [ln for ln in epi if token in ln]
        assert len(lines) == 1 and len(lines[0]) - len(lines[0].lstrip()) == depth, (token, lines)  # at the loop body's depth: never guarded
    assert not any("exchange_off" in ln and ("_wait" in ln or ".arrive(" in ln) for ln in epi), "exchange_off guards no wait and no arrive"
    relay = _function_code(src, "_mma_loop_follower", "_reduce_loop")
    assert sum('_wait(bars.ds_recv_full, relay_state.phase, dbg, _W["wait:follower:ds_recv_full"])' in ln for ln in relay) == 1
    assert not any("diag" in ln.lower() for ln in relay), "no probe reaches the relay"


def test_2cta_stail_skew_levers_follow_the_house_form():
    """The body exposes the four base levers plus SOFTMAX_S_READ / EPILOGUE_DQ_READOUT / SOFTMAX_DQ_WAR, each in the house form (module global +
    const_expr nanosleep loop, default 0 = no traced code: the default render's census pins nanosleep == 0)."""
    source = _body_source()
    exposed = set(_LEVER_LINE.findall(source))
    known = {row["lever"] for row in SITES.values()}
    assert exposed == known, (sorted(exposed), sorted(known))
    for lever in sorted(exposed):
        name = f"DEBUG_{lever}_DELAY_US"
        assert re.search(rf"cutlass\.const_expr\({name} > 0\)", source), name
        assert re.search(rf"cutlass\.range\({name}, unroll=1\)", source), name
        assert source.count(f'globals().get("{_GLOBAL_PREFIX}{lever}_DELAY_US", 0)') == 1, name
        assert f"{_GLOBAL_PREFIX}{lever}" not in os.environ, name
    assert source.count('nvvm.inline_ptx("nanosleep.u32 1000;"') == len(exposed)


# ------------------------------------------------------------------------------------------------------------ GPU (the fp64 gate)
def test_2cta_stail_route_declines_dense_and_mha_typed():
    """The pair body serves grouped THD only: dense and MHA requests decline BY NAME before any compile."""
    _require_2cta()
    dense = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=False)
    with pytest.raises(NotImplementedError, match="requires THD"):  # the typed decline of the direct entry (cta_pair 'on'): by name, before any compile
        make_adapter_2cta_stail(dense, compile_plan=False).check_support()
    from d128_thd_test_utils import declaration as mha_declaration
    from d128_thd_test_utils import make_case as make_mha_case

    mha = make_mha_case((129, 3), (129, 3), heads=2)
    with pytest.raises(NotImplementedError, match="grouped heads"):
        make_adapter_2cta_stail(mha, compile_plan=False, declaration=mha_declaration).check_support()


@pytest.mark.parametrize(
    "lq,lk,hq,hkv,causal,layout,offset_dtype,units",
    [
        pytest.param((256, 256), (256, 256), 4, 2, True, "TH1", torch.int32, "tokens", id="uniform-256-causal"),
        pytest.param((1024,), (1024,), 8, 1, True, "HN", torch.int32, "tokens", id="uniform-1024-gqa8-causal"),
        pytest.param((384, 384), (384, 384), 4, 2, False, "TH1", torch.int64, "elements", id="uniform-384-noncausal"),
        pytest.param((256, 512, 768, 1024), (256, 512, 768, 1024), 4, 2, True, "TH1", torch.int32, "tokens", id="ramp-1024-causal"),
        pytest.param((129, 0, 127, 1), (65, 128, 0, 3), 4, 2, True, "TH1", torch.int32, "tokens", id="suite-degenerate-causal"),
        pytest.param((129, 0, 127, 1), (65, 128, 0, 3), 6, 2, False, "HN", torch.int64, "elements", id="suite-degenerate-noncausal"),
        pytest.param((65, 0, 3), (17, 129, 0), 6, 2, True, "HN", torch.int64, "elements", id="short-pair-half-dead"),
        pytest.param((300, 300), (300, 300), 3, 1, True, "TH1", torch.int64, "elements", id="pair-tail-300"),
    ],
)
def test_2cta_stail_thd_matrix(lq, lk, hq, hkv, causal, layout, offset_dtype, units):
    _require_2cta()
    case = make_case(lq, lk, hq=hq, hkv=hkv, thd=True, dtype=torch.bfloat16, causal=causal, stats_layout=layout, offset_dtype=offset_dtype, units=units)
    _run_case(case)


def test_2cta_stail_thd_all_empty():
    """Every sequence empty: exact zeros, no map dereference (the dead-pair protocol on BOTH CTAs; no s_read / s_tail_empty traffic at all)."""
    _require_2cta()
    case = make_case((0, 0), (0, 0), hq=3, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout="HN", envelope=(128, 128))
    _run_case(case)


def test_2cta_stail_persistent_reuse_more_pairs_than_sms():
    """P14 row: n_q == 1 with more pairs than SMs (a cluster claims a second CLC item; the N == 1 tail path: the tail dQ(lo) without an s_tail_empty
    wait, two rounds per item) plus a half-dead last pair (seq_k % 256 == 128) and zero-trip blocks past seq_q."""
    _require_2cta()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    hq, b = 8, 2
    skv = 256 * ((sms // (hq * b)) + 2) + 128
    case = make_case((128, 64), (skv, skv - 256), hq=hq, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _run_case(case)


def test_2cta_stail_zero_trip_mix():
    """The the barrier audit N-walk as one case: seq_q < seq_k under causal makes the kv blocks past the last q tile zero-trip, so one launch runs
    items with N = 3, 1 and 0 (seq_q 300 / seq_k 2000), N = 1 and 0 (128 / 700), N = 8, 6, 4, 2 (1000 / 1000): every s_tail_empty wait-site guard
    (prologue always, lookahead iff qtile > lo, tail iff hi - 1 > lo) and the pending-completion carry across zero-trip items are exercised."""
    _require_2cta()
    case = make_case((300, 128, 1000), (2000, 700, 1000), hq=4, hkv=2, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _run_case(case)


def test_2cta_stail_launch_count_and_cluster_facts():
    """The chain is five launches with the pair MAIN kernel; the header's facts come from the body's CFG."""
    _require_2cta()
    import importlib

    body = importlib.import_module(_BODY_MODULE)
    assert (body.CFG.CGA_M, body.CFG.CGA_N, body.CFG.PAIR_KV_ROWS, body.CFG.DQ_COL, body.CFG.Q1_STAGES) == (2, 1, 256, 64, 1)
    case = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True)
    api, workspace = _run_case(case, repeats=1)
    names = []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            names.append(event.name)
    kernels = [n for n in names if n.startswith("cudnn_kernel_")]
    assert len(kernels) == 5, kernels
    assert any("bprop_d128_accum_grouped_2cta_stail_kernel" in n for n in kernels), kernels
    assert not any("bprop_d128_accum_grouped_2cta_kernel" in n for n in kernels), kernels
    assert any("setup_thd_d128_2cta_kernel" in n for n in kernels), kernels


def _bitwise_twin(case, build_a, build_b, what):
    grads = {}
    for tag, build in (("a", build_a), ("b", build_b)):
        api = build(case)
        workspace = make_workspace(api.scratch_workspace_bytes())
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
        grads[tag] = {role: case.tensors[role][0].detach().clone() for role in ("dk", "dv")}
    for role in ("dk", "dv"):
        assert torch.equal(grads["a"][role].view(torch.int16), grads["b"][role].view(torch.int16)), f"{role}: {what} differ bitwise"


_MASK_TWIN_CASES = [
    pytest.param(dict(lq=(256, 512), lk=(256, 512), hq=4, hkv=2, causal=True), id="uniform-4-tiles-causal"),  # the diagonal straddle + interior tiles
    pytest.param(
        dict(lq=(300, 300), lk=(300, 300), hq=3, hkv=1, causal=True), id="pair-tail-300"
    ),  # S % 256 = 44: a partial last q tile, a half-dead second CTA
    pytest.param(dict(lq=(384, 384), lk=(384, 384), hq=4, hkv=2, causal=False), id="uniform-384-noncausal"),  # the THD edges without the causal term
    pytest.param(dict(lq=(65, 0, 3, 129, 200), lk=(65, 0, 3, 129, 200), hq=4, hkv=2, causal=True), id="ramp-short-sequences"),  # < one tile, empty, 3 rows
    pytest.param(dict(lq=(0, 0), lk=(0, 0), hq=3, hkv=1, causal=True, stats_layout="HN", envelope=(128, 128)), id="all-empty"),
    pytest.param(dict(lq=(300, 128, 1000), lk=(2000, 700, 1000), hq=4, hkv=2, causal=True), id="zero-trip-mix"),  # 8 pairs: dQ by the fp64 gate only
]


@pytest.mark.parametrize("spec", _MASK_TWIN_CASES)
def test_2cta_stail_mask_forms_agree_bitwise(spec):
    """The mask-form twin (a claim BY CONSTRUCTION: a masked P / dS cell is exactly +0.0 under fsel and under predicated, and an interior
    tile has no masked cell -- `tile_may_mask` is exact): dK / dV BITWISE between mask_form = fsel and predicated on every row, dQ BITWISE on the rows
    whose sequences have at most _DQ_ORDER_FREE_PAIRS kv pairs (the per-pair fp32 reduce-adds onto zero are then order-free) and by the fp64 gate
    beyond that (the zero-trip mix: 8 pairs); each form passes the suite's fp64 gate first.  The rows exercise the causal diagonal straddle (4 tiles
    per pair), the THD edges (S % 256 = 44; a ramp with a sequence shorter than one tile, an empty one and a 3-row one; all-empty), the non-causal
    edges and the N-walk with dead blocks."""
    _require_2cta()
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG

    spec = dict(spec)
    lq, lk, causal = spec.pop("lq"), spec.pop("lk"), spec.pop("causal")
    case = make_case(lq, lk, thd=True, dtype=torch.bfloat16, causal=causal, stats_layout=spec.pop("stats_layout", "TH1"), **spec)
    dq_bitwise = max(lk) <= _DQ_ORDER_FREE_PAIRS * CFG.PAIR_KV_ROWS
    grads = {}
    for form in ("fsel", "predicated"):
        api = make_adapter_2cta_stail(case, mask_form=form)
        workspace = make_workspace(api.scratch_workspace_bytes())
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
        grads[form] = {role: case.tensors[role][0].detach().clone() for role in ("dq", "dk", "dv")}
    for role in ("dk", "dv") + (("dq",) if dq_bitwise else ()):
        assert torch.equal(grads["fsel"][role].view(torch.int16), grads["predicated"][role].view(torch.int16)), f"{role}: fsel vs predicated differ bitwise"


# ------------------------------------------------------------------------------------------------------------ GPU: the the later probes + the STAMPS lever
def _expected_tile_visits(case, hq):
    """Tile visits per launch over BOTH CTAs of every pair (header section 7): per sequence, pair p (256 kv rows) walks q tiles [lo, hi) with lo =
    2 p under causal (0 otherwise) and hi = ceil(seq_q / 128); a pair past seq_k is dead (0); x the q heads x the 2 CTAs."""
    total = 0
    for sq, sk in zip(case.lens_q, case.lens_kv, strict=True):
        for p in range((sk + 255) // 256):
            lo = 2 * p if case.causal else 0
            hi = (sq + 127) // 128
            total += max(hi - lo, 0)
    return total * hq * 2


def test_2cta_stail_debug_clk_dump_accounts_the_waits():
    """The STAMPS lever (`debug_clk = True`, header section 11): dK / dV BITWISE vs the default (the lever changes no arithmetic) and the fp64 gate
    under both; the dump's record layout (config DBG_CLK): every record carries the magic and its rank; each role accumulates EXACTLY the barriers
    it waits on -- the leader's words non-zero on rank 0 and zero on rank 1, the relay's the reverse, the shared roles' on both -- the tile
    counts agree across the roles and the pair and sum to the case's tile visits, the ship -> recv handshake accounts every dQ issue / relay,
    the ship histogram accounts every tile, the body clock and its %globaltimer span are live (the implied SM clock is a plausible GHz).
    Only the CTAs that RAN wrote a record (the CLC scheduler cancels the rest of the declared grid; the region is not zeroed): a record is valid
    iff its magic word matches, and a never-launched cluster leaves BOTH of its CTAs' records unwritten."""
    _require_2cta()
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import debug_clk_records

    d = cfgmod.DBG_CLK
    W = d.WORD
    hq, hkv = 4, 2
    case = make_case((256, 512), (256, 512), hq=hq, hkv=hkv, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _bitwise_twin(case, lambda c: make_adapter_2cta_stail(c), lambda c: make_adapter_2cta_stail(c, debug_clk=True), "the stamps lever (off vs on)")
    plain = make_adapter_2cta_stail(case)
    api = make_adapter_2cta_stail(case, debug_clk=True)
    n_rec = cfgmod.dbg_clk_records(case.batch, hq, api.s_k_max)
    assert api.scratch_workspace_bytes() >= plain.scratch_workspace_bytes() + n_rec * d.BYTES, "the dump region is carved only under the lever"
    assert debug_clk_records(plain, make_workspace(plain.scratch_workspace_bytes()).tensor) is None
    workspace = make_workspace(api.scratch_workspace_bytes())
    poison(case, workspace)
    api.execute(**adapter_arguments(case), workspace=workspace.tensor)
    torch.cuda.synchronize()
    check_case(case, workspace)
    rec_all = debug_clk_records(api, workspace.tensor).cpu().to(torch.int64) & 0xFFFFFFFF  # u32 sums
    assert tuple(rec_all.shape) == (n_rec, d.WORDS), rec_all.shape
    nx = 2 * max((api.s_k_max + 255) // 256, 1)  # grid x = pairs x CGA_M: record r -> x = r % nx, rank = x % 2
    valid = rec_all[:, W["magic"]] == d.MAGIC  # the CTAs that ran (the CLC cancels the rest of the declared grid)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert 2 <= int(valid.sum()) <= min(n_rec, sms), (int(valid.sum()), n_rec, sms)
    pair_valid = valid.view(-1, nx // 2, 2)
    assert bool((pair_valid[..., 0] == pair_valid[..., 1]).all()), "a cluster launches or is cancelled as a PAIR"
    rec = rec_all[valid]
    rank = (torch.arange(n_rec) % nx % 2)[valid]
    col = lambda name: rec[:, W[name]]  # noqa: E731
    assert bool((col("rank") == rank).all()), (col("rank"), rank)
    assert bool((col("items") >= 1).all()), "every CTA runs at least its initial CLC item"
    tiles = col("tiles")
    assert int(tiles.sum()) == _expected_tile_visits(case, hq), (int(tiles.sum()), _expected_tile_visits(case, hq))
    assert bool((col("tiles_leader") == torch.where(rank == 0, tiles, 0)).all()) and bool((col("tiles_follower") == torch.where(rank == 1, tiles, 0)).all())
    assert bool((tiles[rank == 0] == tiles[rank == 1]).all()), "both CTAs of a pair walk the same tiles (the valid records alternate rank 0 / 1)"
    live = tiles > 0
    for role, bars in d.ROLE_WAITS.items():
        words = torch.stack([col(f"wait:{role}:{b}") for b in bars], dim=1)  # (records, barriers)
        on = {"leader": (rank == 0) & live, "follower": (rank == 1) & live}.get(role, live)
        off = {"leader": rank == 1, "follower": rank == 0}.get(role)
        assert bool((words[on] > 0).all()), (role, words[on])  # every barrier the role waits on, every live CTA that runs it
        if off is not None:
            assert bool((words[off] == 0).all()), (role, "a role that does not run on this rank must accumulate nothing")
    blocked = torch.stack([col(f"blocked:leader:{b}") for b in d.ROLE_WAITS["leader"]], dim=1)
    assert bool((blocked[rank == 1] == 0).all()) and bool((blocked >= 0).all())
    for side, tcount in (("leader", col("tiles_leader")), ("follower", col("tiles_follower"))):
        assert bool((col(f"recv_{side}_matched") + col(f"recv_{side}_missed") == tcount).all()), side  # one handshake per dQ issue / per relay
    hist = sum(col(f"seg_ship_hist{i}") for i in range(4))
    assert bool((hist == tiles).all()), "the ship histogram accounts every tile of the epilogue lead"
    assert bool((col("seg_ship_max")[live] > 0).all()) and bool((col("seg_ship_sum")[live] > 0).all()) and bool((col("seg_dq_done_sum")[live] > 0).all())
    ghz = col("body_clk").double() / col("body_ns").double().clamp(min=1)
    assert bool((col("body_ns") > 0).all()) and bool(((ghz > 0.2) & (ghz < 4.0)).all()), ghz  # clk / ns = the SM clock the body ran at


@pytest.mark.parametrize("probe", _PROBES)
def test_2cta_stail_probes_launch_and_return(probe):
    """Each DIAGNOSTIC probe of this map builds, launches and returns (no fault, no hang: the probes delete bytes / a transfer, never a wait); the
    output is WRONG BY DESIGN and is NOT checked here -- `--validate` refuses them by name in the driver.  exchange_off is the SAFE form (every wait kept)."""
    _require_2cta()
    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    api = make_adapter_2cta_stail(case, diag_2cta=probe)
    workspace = make_workspace(api.scratch_workspace_bytes())
    poison(case, workspace)
    for _ in range(2):
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()


_CHILD_PROBE_SKEW = """
import os, sys, importlib.util
from pathlib import Path
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
sys.path.insert(0, {testdir!r})
name = {body!r}
spec = importlib.util.find_spec(name)
mod = importlib.util.module_from_spec(spec)
setattr(mod, {global_name!r}, {delay})
sys.modules[name] = mod
spec.loader.exec_module(mod)
assert getattr(mod, {attr!r}) == {delay}, getattr(mod, {attr!r}, None)
import torch
import cudnn
assert Path(cudnn.__file__).resolve() == Path({expected_fe!r}).resolve(), cudnn.__file__
from d128_grouped_test_utils import adapter_arguments, make_case, make_workspace, poison
from frost_test_utils import process_watchdog
from test_sdpa_bwd_d128_2cta_stail_sm100 import make_adapter_2cta_stail
case = make_case((1024, 1024), (1024, 1024), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
api = make_adapter_2cta_stail(case, diag_2cta={probe!r})
workspace = make_workspace(api.scratch_workspace_bytes())
args = adapter_arguments(case)
with process_watchdog({watchdog}, "2-CTA s_tail probe {probe} under leader_ship skew"):
    for _ in range(2):
        poison(case, workspace)
        api.execute(**args, workspace=workspace.tensor)
        torch.cuda.synchronize()
print("PROBE_SKEW_OK {probe} {delay}", flush=True)
"""


def test_2cta_stail_exchange_off_is_clean_under_the_leader_ship_skew(tmp_path):
    """The mechanism's repro pair: a probe that skipped the three dS gates + `LEADER_SHIP` 300 us at s=1024 FAULTED (an over-arrival) at tile lo+1
    of every item -- the leader's late ship let the follower's ds_recv_full arm(t+1) land on the tx-pending phase t.  `exchange_off` under the
    SAME lever must run clean: no copy exists, every phase completes at its own arm (output wrong by design, not checked).  A fresh process arms
    the lever before the body executes; exit 70 / a timeout = HANG, a CUDA error = the fault class."""
    _require_2cta()
    _adapter_class()
    import cudnn

    lever, delay = "LEADER_SHIP", 300
    assert lever in set(_LEVER_LINE.findall(_body_source()))
    script = _CHILD_PROBE_SKEW.format(
        testdir=str(pathlib.Path(__file__).resolve().parent),
        body=_BODY_MODULE,
        global_name=f"{_GLOBAL_PREFIX}{lever}_DELAY_US",
        attr=f"DEBUG_{lever}_DELAY_US",
        delay=delay,
        expected_fe=cudnn.__file__,
        probe="exchange_off",
        watchdog=_WATCHDOG_S,
    )
    (tmp_path / "probe_skew_exchange_off.py").write_text(script)
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path / "probe_skew_exchange_off.py")], capture_output=True, text=True, timeout=_CHILD_TIMEOUT_S, env=dict(os.environ)
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"[exchange_off + leader_ship {delay} us] HANG: the child did not return within {_CHILD_TIMEOUT_S} s -> barrier AUDIT, never a bigger budget"
        )
    (tmp_path / "probe_skew_exchange_off.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode == 70:
        pytest.fail(
            f"[exchange_off + leader_ship] HANG: process_watchdog killed the child after {_WATCHDOG_S:.0f} s\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    assert proc.returncode == 0 and f"PROBE_SKEW_OK exchange_off {delay}" in proc.stdout, f"rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"


# ------------------------------------------------------------------------------------------------------------ GPU: pair-skew detectors (fresh process, watchdog)
SITES = dict(
    SITES_BASE,
    softmax_s_read=dict(
        lever="SOFTMAX_S_READ",
        role="softmax warps 0-7 of the FOLLOWER (rank 1)",
        site="_softmax_loop: between _wait_tcgen(bars.s_done, ...) and the S^T ld pair (_s_values), EVERY tile",
        delay_us=300,
        isolates="s_read (the leader's slot-2 wait <- 512 lanes after the S^T ld + wait(LOAD)): with the follower softmax asleep, dQ(t-1) is held by "
        "s_read ALONE (the dS gates of t-1 and the leader softmax's arrive are long done)",
        red="the s_read wait removed before dQ(t-1) (or the arrive moved after the exp body): dQ(t-1) lands in cols 64..127 before the follower reads "
        "S^T(t) -> the follower's P^T(t) q columns 64..127 = exp2 of dQ values -> dV / dK RED on the follower's kv rows (g03 Rule 6 'second half')",
        green="fp64 gate on dQ / dK / dV, eager x2 + capture / replay x4, no hang",
        cell="skew",
    ),
    epilogue_dq_readout=dict(
        lever="EPILOGUE_DQ_READOUT",
        role="epilogue warps 8-11 of the FOLLOWER (rank 1)",
        site="_epilogue_loop: between _wait_tcgen(bars.dq_done, ...) and the dQ tcgen05.ld (64 cols @ DQ_COL)",
        delay_us=300,
        isolates="s_tail_empty (the leader's wait before S^T(t+1) / the tail dQ / the next prologue <- 256 lanes after the dQ ld): with the follower "
        "epilogue asleep, S^T(t+1) is held by s_tail_empty ALONE (q1_full landed long ago, the leader's readout is done)",
        red="the wait removed at the lookahead: S^T(t+1) overwrites cols 64..127 before the follower reads dQ(t-1) -> its 64 dQ rows of tile t-1 = "
        "S^T values (dQ RED, dK / dV green); at the tail: dQ rows of tile hi-2 wrong; at the prologue: dQ rows of tile hi-1 of the previous item wrong",
        green="fp64 gate on dQ / dK / dV, eager x2 + capture / replay x4, no hang",
        cell="skew",
    ),
    softmax_dq_war=dict(
        lever="SOFTMAX_DQ_WAR",
        role="MMA warp 12 of the LEADER (rank 0): _dq_issue, both sites (in-tile dQ(t-1) and the tail dQ(hi-1))",
        site="_dq_issue: between the three dS-gate waits and the dQ MMA issue (the gates are consumed, the issue -- hence dq_done -- is held)",
        delay_us=300,
        isolates="dq_done's waiters as the ONLY gate of the sDS writers: under warp_roles=16w_fa4 both softmaxes' dS(t) STS wait their `dq_war` "
        "observation of dq_done(t-1) ALONE (the alternate role shape X7; the barrier audit x-fa4 table section 12 item 12); under 16w the epilogue's dq_done(t-1) "
        "readout wait before its dS(t) STS (the same WAR by program order)",
        red="the softmax `dq_war` wait removed (16w_fa4): the dS(t) STS lands under dQ(t-1)'s async A read of sDS own + recv -> dQ rows wrong, dK / dV green",
        green="fp64 gate on dQ / dK / dV, eager x2 + capture / replay x4, no hang",
        cell="skew",
    ),
)

_CHILD = """
import os, sys, importlib.util
from pathlib import Path
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
sys.path.insert(0, {testdir!r})
name = {body!r}
spec = importlib.util.find_spec(name)
mod = importlib.util.module_from_spec(spec)
setattr(mod, {global_name!r}, {delay})
sys.modules[name] = mod
spec.loader.exec_module(mod)
assert getattr(mod, {attr!r}) == {delay}, getattr(mod, {attr!r}, None)
import torch
import cudnn
assert Path(cudnn.__file__).resolve() == Path({expected_fe!r}).resolve(), cudnn.__file__
from d128_grouped_test_utils import adapter_arguments, check_case, make_case, make_workspace, poison
from frost_test_utils import process_watchdog
from test_sdpa_bwd_d128_2cta_stail_sm100 import make_adapter_2cta_stail
sms = torch.cuda.get_device_properties(0).multi_processor_count
skv = {skv}
print("DATASET sms=%d skv=%d (inputs: the suite's CPU generator, seed 149)" % (sms, skv), flush=True)
case = {cell}
api = make_adapter_2cta_stail(case, **{fields!r})  # the typed op-level field(s) through the suite's factory, as the driver sets them
assert api._prepared.artifact.dkv_accum == {dkv!r}, api._prepared.artifact.dkv_accum  # the configuration under test ran (G13 section 6)
print("DKV_ACCUM %s" % api._prepared.artifact.dkv_accum, flush=True)
workspace = make_workspace(api.scratch_workspace_bytes())
args = adapter_arguments(case)
with process_watchdog({watchdog}, "2-CTA s_tail pair-skew {site} dkv_accum={dkv}"):
    for _ in range(2):
        poison(case, workspace)
        api.execute(**args, workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        api.execute(**args, workspace=workspace.tensor)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(4):
        poison(case, workspace)
        graph.replay()
        torch.cuda.synchronize()
        check_case(case, workspace)
    graph.reset()
print("SKEW_OK {site} {delay} dkv_accum={dkv}", flush=True)
"""

# ``dkv_accum = "fp32_reduce"`` (the alias seams of the barrier audit): the fp32 dV / dK slices are staged in sK2's halves, BOTH dS halves (own +
# RECV = where the PEER's ship lands) and XCHG, drained through the row dvk_stage_empty and the dvk_empty / k2_empty chain.  Exactly two levers
# reach those seams, so exactly two ids get an fp32 twin row (the other levers' seams are the same bytes under both forms: no row, the module
# stays small).  The default-form ids keep their names.  Budget: ``skew_module`` (every row is a fresh child = cold import + JIT, ~160-240 s each
# + ~240 s parent preamble; <= 4 rows per -k half).
_DKV_ACCUM_SEAM_ROWS = {
    "leader_ship": "recv-half seam (sDS h0 / h1): round 1 stages fp32 dV slices 2 / 3 (round 2: dK slice 3) in the dS slab whose RECV half "
    "is the LEADER's late ship; the landing guarantee is ds_recv_full(hi-1) -> dQ(hi-1) -> dq_done(hi-1), the generic STS after our dq_done wait",
    "follower_epilogue_item_end": "store-drain seam (lever 3): the FOLLOWER's late item-end readout holds round 1 -> dvk_stage_empty (THREAD 32 / 32, "
    "LOCAL) -> round 2 -> the item-end drain -> dvk_empty (pair-wide 64) / k2_empty: the leader's next dS write / ship waits on BOTH CTAs' drains",
}
_SKEW_ROWS = [pytest.param(site, None, id=site) for site in SITES] + [
    pytest.param(site, "fp32_reduce", id=f"{site}-fp32_reduce") for site in _DKV_ACCUM_SEAM_ROWS
]


@pytest.mark.parametrize("site, dkv_accum", _SKEW_ROWS)
def test_sdpa_bwd_d128_2cta_stail_survives_pair_skew(site, dkv_accum, tmp_path):
    """One lever per id on the SECOND body (``SITES[site]``: role, site, delay, what it isolates, the RED form, the expected PASS): a fresh process
    arms ``FROST_D128_2CTA_DEBUG_<SITE>_DELAY_US`` before the body executes, runs the skew / p14 cell (more pairs than SMs) under ``process_watchdog``
    and asserts the fp64 gate eagerly and under capture / replay.  Exit 70 or a subprocess timeout = HANG -> the barrier AUDIT, never a bigger
    budget; a gate miss = the RED form's wrong rows, named by ``check_case``.  ``dkv_accum = "fp32_reduce"`` (the ``-fp32_reduce`` ids, the two
    ``_DKV_ACCUM_SEAM_ROWS`` levers) sets the typed op-level field in the child exactly as the driver does and pins ``artifact.dkv_accum``; the
    pass criterion is the SAME gate (the fp32 form is tighter on every validated cell) -- never a widened tolerance."""
    row = SITES[site]
    fields = {} if dkv_accum is None else {"dkv_accum": dkv_accum}
    form = "bf16_partials" if dkv_accum is None else dkv_accum
    seam = _DKV_ACCUM_SEAM_ROWS.get(site, "(the field does not move this lever's bytes)") if dkv_accum else ""
    assert row["lever"] in set(_LEVER_LINE.findall(_body_source())), f"the second body must expose {row['lever']}"
    _require_2cta()
    _adapter_class()
    import cudnn

    script = _CHILD.format(
        testdir=str(pathlib.Path(__file__).resolve().parent),
        body=_BODY_MODULE,
        global_name=f"{_GLOBAL_PREFIX}{row['lever']}_DELAY_US",
        attr=f"DEBUG_{row['lever']}_DELAY_US",
        delay=row["delay_us"],
        expected_fe=cudnn.__file__,
        skv=_SKV,
        cell=_CELLS[row["cell"]],
        watchdog=_WATCHDOG_S,
        site=site,
        fields=fields,
        dkv=form,
    )
    tag = f"{site}_{form}"
    (tmp_path / f"skew_stail_{tag}.py").write_text(script)
    print(f"\n[{site}] {row['role']} sleeps {row['delay_us']} us at {row['site']}; isolates {row['isolates']}; RED = {row['red']}; GREEN = {row['green']}")
    if dkv_accum:
        print(f"[{site}] dkv_accum={form}: {seam}")
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path / f"skew_stail_{tag}.py")], capture_output=True, text=True, timeout=_CHILD_TIMEOUT_S, env=dict(os.environ)
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        (tmp_path / f"skew_stail_{tag}.log").write_text(out)
        pytest.fail(f"[{site}] HANG: the child did not return within {_CHILD_TIMEOUT_S} s (RED form: {row['red']}) -> barrier AUDIT, never a bigger budget")
    (tmp_path / f"skew_stail_{tag}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode == 70:
        pytest.fail(
            f"[{site}] HANG: process_watchdog killed the child after {_WATCHDOG_S:.0f} s (RED form: {row['red']})\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    assert (
        proc.returncode == 0 and f"SKEW_OK {site} {row['delay_us']} dkv_accum={form}" in proc.stdout
    ), f"rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
