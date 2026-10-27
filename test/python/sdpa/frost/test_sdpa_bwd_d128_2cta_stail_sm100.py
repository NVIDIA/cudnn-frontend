# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 grouped THD d128 backward on the cta_group::2 datapath, ARM (ii): the SECOND pair body ``bprop_d128_accum_grouped_2cta_stail_f16.py``
(typed adapter field ``tmem_map = "s_tail"``: the dQ half-row image in S^T's dead tail, FAv4's rotated leader order ``fa4`` -- or ``fa4b`` =
dQ(t-1) issued after dV(t), the exchange-cover order of code-writer step 6 on the SAME barrier table (``pair_mma_order``) -- the dedicated sDQ
ring, 34 barriers).  The first body (``tmem_map`` unset = "dp_tail") is UNTOUCHED and stays pinned by ``test_sdpa_bwd_d128_2cta_sm100.py``.

RED-first: the GPU rows below were committed with the body's first host trace-compile, before it ran; every one is the grouped suite's
independent FP64 gate (``check_case``: per-sequence atol / rtol + cosine, storage canaries, exact zeros for empty attention), dQ / dK / dV apart.

Rows: the first body's matrix (uniform pairs, GQA 8, a varlen ramp, non-causal, the suite's degenerate THD row, a half-dead pair, pair tail 300),
all-empty, the P14 rows of the barrier-inspector's ARM (ii) table (n_q == 1 with more pairs than SMs; a half-dead last pair; a zero-trip MIX of
N = 1 / 2 / 3 / 4 / 6 / 8 and dead blocks), the launch census (the stail MAIN kernel), the TMEM maps' BITWISE dK / dV twin (the same
accumulation chain per GEMM on both bodies), the load-form / knob bitwise twins on this body, the typed field tests (tmem_map, the order
and probe domains PER MAP), the host renders (census + md5 record; the first body's record unchanged through this chain), the
deterministic-hang topologies of the inspector's section 8 as SOURCE assertions (RED 3: dp_empty at the first body's site; RED 4: the dS
gates on ready_state; RED 5: the unguarded lookahead wait; D4: the ring release cadence), and the pair-skew detectors (the four carried
levers + the two NEW ones, SOFTMAX_S_READ / EPILOGUE_DQ_READOUT, in the skew module's fresh-process watchdog form; the C2b arm
``dkv_accum = "fp32_reduce"`` as a twin row on the two levers that reach its staging seams, ``_DKV_ACCUM_SEAM_ROWS``).

Day 3 (code-writer 2026-10-27 on optimizer 07:45 rank 0 + barrier-inspector 08:01): the three s_tail-only probes `exchange_off` (D8s, the
safe exchange ceiling: the ship skipped, the arm with 0 tx bytes, every wait kept), `stats_one_chunk` (D10) and `sdq_sts_off` (D9) with their
PTX census deltas from the config; `ds_gates_off` (D8) RETIRED on this map (declined by name: it faulted); the SOURCE assertion that D8s
keeps every wait and arrive and only fakes the bytes; the D8s-under-leader-skew child (the mechanism's repro pair: D8 faulted there, D8s must
stay clean); the STAMPS lever `debug_clk` (typed bool; the render differs by the %clock64 reads only, the default stays byte-identical) and
its GPU accounting row (every role accumulates exactly the barriers it waits on; dK / dV bitwise vs the default; the dump's record layout).

Lever L5 rows (code-writer D, 2026-10-27; the typed field ``mask_form``): the per-map typed domain and the kernel codes, the EXACTNESS of the
interior-tile predicate `tile_may_mask` against the per-cell mask on host ints (the bitwise claim's construction), the three forms' host renders
(fsel = the default rendering byte-identical; predicated / diag_mask_off under the SAME mbarrier census and setmaxnreg split with the designed selp /
keep-word / branch deltas), and the GPU twin: dQ / dK / dV BITWISE between fsel and predicated on the cells that exercise the diagonal, the THD
edges (S % 256 != 0, a ramp with a sequence shorter than one tile and an empty one, all-empty, the zero-trip mix) and the uniform case.
"""

import os
import pathlib
import re
import subprocess
import sys

import pytest
import torch

from d128_grouped_test_utils import adapter_arguments, check_case, make_case, make_workspace, poison
from frost_test_utils import requires_dsl
from test_sdpa_bwd_d128_2cta_sm100 import _CENSUS as _CENSUS_FIRST_BODY
from test_sdpa_bwd_d128_2cta_sm100 import _EXPECTED as _EXPECTED_FIRST_BODY
from test_sdpa_bwd_d128_2cta_sm100 import (
    _HOST_MODULE,
    _IMPLEMENTATION,
    _RENDER,
    _adapter_class,
    _parse_record,
    _require_2cta,
    _setmaxnreg_of,
    make_adapter_2cta,
)
from test_sdpa_bwd_d128_2cta_skew_sm100 import _CELLS, _CHILD_TIMEOUT_S, _GLOBAL_PREFIX, _LEVER_LINE, _SKV, _WATCHDOG_S
from test_sdpa_bwd_d128_2cta_skew_sm100 import SITES as _SITES_FIRST_BODY

pytestmark = [pytest.mark.L0, requires_dsl]

_TMEM_MAP = "s_tail"
_ORDERS = [pytest.param(None, id="fa4"), pytest.param("fa4b", id="fa4b")]  # the map's two leader orders: None = the default (fa4) | fa4b (step 6)
_RED_ROUTES = ["red_global", "red_global_pair"]  # the register-direct dQ routes (step 8); None = the default "tma" (the ring STS + reduce-add)
_MASK_FORMS = ("fsel", "predicated", "diag_mask_off")  # lever L5: None = the default "fsel" (the keep-word mask selected on every tile)
# The MASK FORM census (L5; rendered beside the module census): the per-cell selects of the keep-word mask (arith.select -> selp.f32: the body's
# CFG.MASK_SELECTS_PER_LANE_PER_TILE = 64 plus two others), the keep-word upper-edge shifts (keep_below_word -> shr.u32: CFG.MASK_WORDS_PER_LANE_PER_TILE
# = 2 of the body's), the branches (the predicated form adds ONE CTA-uniform `if mask_tile:` per q quarter = CFG.QUARTERS_PER_GROUP).
_MASK_CENSUS = {"selp_f32": r"selp\.f32", "shr_u32": r"shr\.u32", "bra": r"\bbra(\.uni)?\s"}
# dQ is a sum of per-pair fp32 reduce-adds onto zero: with at most TWO contributions per element the result is order-free (a + b == b + a), with
# three or more it is not (fp32 addition is not associative) -- the cells where the dQ twin is a BITWISE claim (G09: name the cells).
_DQ_ORDER_FREE_PAIRS = 2
# The dQ ROUTE census (step 8; rendered beside the module census): the register-direct reds, the sDQ ring's fp32 slice STS (the fp32 16-B rows
# lower to `st.shared.v2.b64`, the only ones of the body: 3 _dq_round sites x DQ_BOX // 4 = 24 under tma, 0 under a red route) and their proxy
# fences (15 -> 12), the lane-pair shuffles of the sector-paired form, the bulk commit groups (4 per-use + 1 item-end under tma, the item-end one
# alone under a red route).
_ROUTE_CENSUS = {
    "red_global_add_v4": r"red\.global\.add\.v4\.f32",
    "sts_dq_v2_b64": r"st\.shared\.v2\.b64",
    "fence_proxy_async": r"fence\.proxy\.async",
    "shfl_bfly": r"shfl\.sync\.bfly",
    "bulk_commit_group": r"cp\.async\.bulk\.commit_group",
}
_BODY_MODULE = "cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_grouped_2cta_stail_f16"
_BODY_REL = pathlib.Path("python/cudnn/sdpa/bwd/kernels/sm100/bprop_d128_accum_grouped_2cta_stail_f16.py")
_TREE = pathlib.Path(__file__).resolve().parents[4]  # test/python/sdpa/frost/<file> -> the checkout
_RECORD = pathlib.Path(__file__).resolve().parent / "renderings" / "md5_d128_accum_grouped_2cta_stail_default_sm_100a.txt"
_RECORD_FIRST_BODY = pathlib.Path(__file__).resolve().parent / "renderings" / "md5_d128_accum_grouped_2cta_default_sm_100a.txt"
# Day-3 census rows (the probes' purity, the inspector's 08:01 method): the stats / delta LDS.128 (`ld.shared.v4.b32`: D10 removes exactly
# CFG.DIAG_STATS_LDS_REMOVED_PER_LANE static sites), the expect_tx arms (D8s keeps the count), the %clock64 reads (0 unless `debug_clk`), and the
# three counts every probe must leave untouched: `bar.warp.sync`, `tcgen05.fence::after_thread_sync`, `mapa` (D8s loses the ship's own two).
_DAY3_CENSUS = {
    "lds_v4_b32": r"ld\.shared\.v4\.b32",
    "expect_tx": r"mbarrier\.arrive\.expect_tx",
    "clock64": r"%clock64",
    "bar_warp_sync": r"bar\.warp\.sync",
    "tcgen05_fence_after": r"tcgen05\.fence::after_thread_sync",
    "mapa": r"mapa\.",
}
_CENSUS = dict(_CENSUS_FIRST_BODY, entry_stail=r"\.entry\s+cudnn_kernel_bprop_d128_accum_grouped_2cta_stail_kernel", **_DAY3_CENSUS)
_DAY3_EXPECTED_FIRST_BODY = dict(lds_v4_b32=38, expect_tx=11, clock64=0, bar_warp_sync=62, tcgen05_fence_after=24, mapa=15)  # the dp_tail body (unchanged)
# The s_tail body's census against the first body's (kernel header section 5 / 11): dQ is issued at TWO sites (in-tile dQ(t-1) + the tail
# dQ(hi-1): 16 + 16 k-steps -> mma 56 + 16 = 72), the tail's dq_done commit (+1), the arrive sites 41 - 4 (the dq_empty bootstrap + the two
# per-slot sites gone with the 4-slice ring -> 2 slots x 2 rounds) + 1 (dq_full per round, 3 sites) + 1 (s_read) + 1 (s_tail_empty) = 40, the
# wait sites 63 - 2 (dq_empty: 3 rounds x 2 slots vs 4 + 4 terminal) - 1 (the q1 drain, one stage) + 1 (s_read) + 4 (s_tail_empty: prologue,
# lookahead, tail, drain) + 3 (the tail's dS gates) = 68.  The entry is the stail kernel; the first body's entry is absent.
_EXPECTED = dict(
    _EXPECTED_FIRST_BODY,
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
        "dq_reduce_add_off": {"reduce_add": 0},  # D2-pair: the four reduce-add ISSUES (the per-round dq_empty arrives stay)
        "exchange_off": {"dsmem_ship": 0, "mapa": default["mapa"] - 2},  # D8s: the ONE bulk copy + its two mapa (dst pointer, peer mbar); the arm's tx operand
        "stats_one_chunk": {"lds_v4_b32": default["lds_v4_b32"] - c.DIAG_STATS_LDS_REMOVED_PER_LANE},  # D10: 28 of the 32 stats LDS.128
        "sdq_sts_off": {"sts_dq_v2_b64": 0, "fence_proxy_async": default["fence_proxy_async"] - c.DQ_ROUNDS - 1},  # D9: the 3 _dq_round STS sites + fences
    }[probe]


def make_adapter_2cta_stail(case, *, compile_plan=True, declaration=None, **fields):
    """The suite's adapter construction on the 2-CTA subclass with the SECOND body selected (``tmem_map = "s_tail"`` on the instance before
    ``check_support()``), plus any other typed field (``tma_load_form`` / ``store_wait_form`` / ``reg_split`` / ``diag_2cta`` / ``pair_mma_order`` /
    ``dq_route_pair`` / ``mask_form``)."""
    api = make_adapter_2cta(case, compile_plan=False, declaration=declaration)
    api.tmem_map = _TMEM_MAP
    for name, value in fields.items():
        setattr(api, name, value)
    if not compile_plan:
        return api
    assert api.check_support()
    api.compile()
    assert api._prepared.native is not None
    assert api._prepared.artifact.implementation == _IMPLEMENTATION
    return api


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
        pytest.skip(f"{path} is not in this checkout: the second pair body belongs to task/frost-sdpa-bwd-d128-gqa-thd-2cta-b200-arm2")
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
    the ring depths, the 35-row BAR_LAYOUT (kinds, counts, END 230960 since the C2b row), the three alias rows, the ONE order and the probe domain on the map.
    """
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta as first
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod

    c = cfgmod.CFG
    assert c.TMEM_MAP == "s_tail" and cfgmod.TMEM_MAPS == ("dp_tail", "s_tail") and cfgmod.TMEM_MAP_DEFAULT == "dp_tail"
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
    assert first.CFG.DQ_COL == 192 and c.DQ_COL == first.CFG.S_COL + first.CFG.PACK_COLS  # the image moved from dP^T's dead half to S^T's dead tail
    # 35 rows / 40 slots since C2b (code-writer C, 2026-10-27): + dvk_stage_empty (THREAD 32, the fp32 dK / dV rounds' WAR release; init only under dkv_fp32)
    assert len(cfgmod.BAR_LAYOUT) == 35 and sum(r[2] for r in cfgmod.BAR_LAYOUT) == 40 and cfgmod.BAR_LAYOUT_END == 230960
    assert cfgmod.BAR_LAYOUT[-1][0] == "dvk_stage_empty" and cfgmod.BAR_LAYOUT[-1][3] == cfgmod.CFG.STORE_LANES
    counts = {row[0]: row[3] for row in cfgmod.BAR_LAYOUT}
    assert counts["s_read"] == c.PAIR_SOFTMAX_LANES == 512 and counts["s_tail_empty"] == c.PAIR_EPILOGUE_LANES == 256
    assert counts["dp_empty"] == 256 and counts["dq_empty"] == c.STORE_LANES == 32 and counts["dq_full"] == c.DQ_SLICE_LANES == 64 and counts["dv_full"] == 160
    assert cfgmod.BAR_KIND_COUNTS == {"tma_leader": 7, "mma": 9, "thread": 9, "leader": 6, "peer": 2, "tma_dsmem": 1, "tma": 1}
    assert [a[0] for a in cfgmod.ALIASES] == ["dk_store", "dv_store", "ds_recv(rank)"]  # the dq_staging alias is deleted
    assert c.PAIR_MMA_ORDERS == ("fa4", "fa4b") and c.PAIR_MMA_ORDER_DEFAULT == "fa4" and c.DIAG_2CTA_NAMES_ON_MAP == _PROBES
    assert "ds_gates_off" not in c.DIAG_2CTA_NAMES_ON_MAP and "ds_gates_off" in first.CFG.DIAG_2CTA_NAMES_ON_MAP  # D8 RETIRED here (it faults), kept on dp_tail
    # the day-3 probes' constants and the STAMPS record (config = truth; the kernel reads DBG_CLK, never a word index)
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
    )  # the record = the 384-B spare left by the C2b row
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
    assert first.CFG.PAIR_MMA_ORDERS == ("dk_first", "s_first") and first.CFG.PAIR_MMA_ORDER_DEFAULT == "s_first"  # the first body's order field is untouched
    # the dQ route (step 8): the domain, the default, the kernel code space and the red route's counts derive from the 64-column half-row image
    assert c.DQ_ROUTES_PAIR == ("tma", "red_global", "red_global_pair") and c.DQ_ROUTE_PAIR_DEFAULT == "tma"
    assert cfgmod.DQ_ROUTE_PAIR_CODES == {"tma": 0, "red_global": 1, "red_global_pair": 2}
    assert (c.DQ_RED_ROW_WORDS, c.DQ_RED_V4_PER_LANE, c.DQ_RED_SECTORS_PER_LANE, c.DQ_RED_V4_PER_TILE_PER_CTA) == (64, 16, 8, 2048)
    assert c.DQ_RED_ROW_WORDS == c.DQ_COLS == c.D_HALF and c.DQ_RED_V4_PER_TILE_PER_CTA == c.EPILOGUE_LANES * c.DQ_RED_V4_PER_LANE
    cfgmod.validate_config()
    first.validate_config()


def test_2cta_stail_tmem_map_and_per_map_domains_are_typed():
    """``tmem_map`` is a typed field (``None`` = "dp_tail" = the first body; "s_tail" = this body; anything else declines BY NAME); the order and the
    probe domains are resolved PER MAP: ``fa4`` / ``fa4b`` exist on s_tail only (s_first / dk_first decline there BY NAME: their s_read / s_tail_empty
    site sets differ), ``dp_empty_wait_off`` (D7) declines on s_tail BY NAME (dp_empty is arrived at the dS^T readout there); the adapter
    readers prefix the engine name; the record default is None."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import diag_2cta, dq_route_pair, pair_mma_order, tmem_map

    assert cfgmod.resolve_tmem_map(None) == "dp_tail" and cfgmod.resolve_tmem_map("s_tail") == "s_tail"
    assert cfgmod.tmem_map_s_tail(None) is False and cfgmod.tmem_map_s_tail("s_tail") is True
    for bad in ("bogus", True, 1, "", "stail", "S_TAIL", "dp"):
        with pytest.raises(ValueError, match="tmem_map must be one of"):
            cfgmod.resolve_tmem_map(bad)
    assert cfgmod.resolve_pair_mma_order_for_map(None, "s_tail") == "fa4" and cfgmod.resolve_pair_mma_order_for_map("fa4", "s_tail") == "fa4"
    assert cfgmod.resolve_pair_mma_order_for_map("fa4b", "s_tail") == "fa4b"  # the exchange-cover order (step 6); the default stays fa4 until measured
    assert cfgmod.dq_after_dv_for_map(None, "s_tail") is False and cfgmod.dq_after_dv_for_map("fa4", "s_tail") is False
    assert cfgmod.dq_after_dv_for_map("fa4b", "s_tail") is True and cfgmod.dq_after_dv_for_map(None, None) is False
    assert cfgmod.dq_after_dv_for_map("dk_first", "dp_tail") is False  # the first body's orders never read it
    assert cfgmod.resolve_pair_mma_order_for_map(None, None) == "s_first" and cfgmod.resolve_pair_mma_order_for_map("dk_first", "dp_tail") == "dk_first"
    bad = (("s_first", "s_tail"), ("dk_first", "s_tail"), ("fa4", "dp_tail"), ("fa4", None), ("fa4b", "dp_tail"), ("fa4b", None), ("bogus", "s_tail"))
    for order, tmap in bad:
        with pytest.raises(ValueError, match="pair_mma_order must be one of"):
            cfgmod.resolve_pair_mma_order_for_map(order, tmap)
        with pytest.raises(ValueError, match="pair_mma_order must be one of"):
            cfgmod.dq_after_dv_for_map(order, tmap)
    assert cfgmod.resolve_diag_2cta_for_map(None, "s_tail") is None and cfgmod.resolve_diag_2cta_for_map("dq_reduce_add_off", "s_tail") == "dq_reduce_add_off"
    for probe in _PROBES:  # the day-3 probes resolve on s_tail and decline BY NAME on dp_tail (the first body has no code for them)
        assert cfgmod.resolve_diag_2cta_for_map(probe, "s_tail") == probe
        if probe not in cfgmod.CFG_DP_TAIL.DIAG_2CTA_NAMES_ON_MAP:
            with pytest.raises(ValueError, match=f"diag_2cta '{probe}' is not defined on tmem_map 'dp_tail'"):
                cfgmod.resolve_diag_2cta_for_map(probe, "dp_tail")
    assert cfgmod.resolve_diag_2cta_for_map("dp_empty_wait_off", "dp_tail") == "dp_empty_wait_off"
    assert cfgmod.resolve_diag_2cta_for_map("ds_gates_off", "dp_tail") == "ds_gates_off"  # kept on the first body (its rendering pins)
    with pytest.raises(ValueError, match="is not defined on tmem_map 's_tail'"):
        cfgmod.resolve_diag_2cta_for_map("dp_empty_wait_off", "s_tail")
    with pytest.raises(ValueError, match="RETIRED on the s_tail map.*only back-edge.*exchange_off"):  # D8: the inspector's 08:01 reading, by name
        cfgmod.resolve_diag_2cta_for_map("ds_gates_off", "s_tail")
    with pytest.raises(ValueError, match="diag_2cta must be None or one of"):
        cfgmod.resolve_diag_2cta_for_map("bogus", "s_tail")
    # the STAMPS lever: a typed bool, s_tail only, False by default; the reader prefixes the engine name
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import debug_clk

    assert cfgmod.resolve_debug_clk_for_map(None, "s_tail") is False and cfgmod.resolve_debug_clk_for_map(True, "s_tail") is True
    assert cfgmod.resolve_debug_clk_for_map(False, None) is False and debug_clk(SimpleNamespace(_NAME="x")) is False
    assert debug_clk(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail", debug_clk=True)) is True
    for bad_flag, bad_map, why in (
        (True, "dp_tail", "s_tail' only"),
        (True, None, "s_tail' only"),
        (1, "s_tail", "must be a bool"),
        ("True", "s_tail", "must be a bool"),
    ):
        with pytest.raises(ValueError, match=why):
            cfgmod.resolve_debug_clk_for_map(bad_flag, bad_map)
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: debug_clk is defined on tmem_map 's_tail' only"):
        debug_clk(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", debug_clk=True))
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: debug_clk must be a bool"):
        debug_clk(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail", debug_clk=1))
    assert SdpaBwdDslSm100D128.debug_clk is False  # the record default: no stamps
    api = SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail")
    assert tmem_map(api) == "s_tail" and pair_mma_order(api) == "fa4" and diag_2cta(api) is None
    assert pair_mma_order(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail", pair_mma_order="fa4b")) == "fa4b"
    assert tmem_map(SimpleNamespace(_NAME="x")) == "dp_tail" and pair_mma_order(SimpleNamespace(_NAME="x")) == "s_first"
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: tmem_map must be one of"):
        tmem_map(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="stail"))
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: pair_mma_order must be one of"):
        pair_mma_order(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail", pair_mma_order="s_first"))
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: pair_mma_order must be one of"):
        pair_mma_order(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", pair_mma_order="fa4"))
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: diag_2cta 'dp_empty_wait_off' is not defined"):
        diag_2cta(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail", diag_2cta="dp_empty_wait_off"))
    assert SdpaBwdDslSm100D128.tmem_map is None  # the record default: the first body
    # the dQ route (step 8): "tma" on both maps; the red forms on s_tail only (declined BY NAME on dp_tail); the reader prefixes the engine name
    assert cfgmod.resolve_dq_route_pair_for_map(None, "s_tail") == "tma" and cfgmod.resolve_dq_route_pair_for_map(None, None) == "tma"
    for route, code in (("tma", 0), ("red_global", 1), ("red_global_pair", 2)):
        assert cfgmod.resolve_dq_route_pair_for_map(route, "s_tail") == route and cfgmod.dq_route_pair_code(route, "s_tail") == code
    assert cfgmod.dq_route_pair_code(None, None) == 0 and cfgmod.dq_route_pair_code("tma", "dp_tail") == 0
    for bad_route, bad_map in (
        ("red_global", "dp_tail"),
        ("red_global", None),
        ("red_global_pair", "dp_tail"),
        ("bogus", "s_tail"),
        ("", "s_tail"),
        ("RED_GLOBAL", "s_tail"),
    ):
        with pytest.raises(ValueError, match="dq_route_pair must be one of"):
            cfgmod.resolve_dq_route_pair_for_map(bad_route, bad_map)
        with pytest.raises(ValueError, match="dq_route_pair must be one of"):
            cfgmod.dq_route_pair_code(bad_route, bad_map)
    assert dq_route_pair(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail", dq_route_pair="red_global")) == "red_global"
    assert dq_route_pair(SimpleNamespace(_NAME="x")) == "tma" and dq_route_pair(SimpleNamespace(_NAME="x", tmem_map="s_tail")) == "tma"
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: dq_route_pair must be one of"):
        dq_route_pair(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", dq_route_pair="red_global"))  # the first body has no red route
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: dq_route_pair must be one of"):
        dq_route_pair(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail", dq_route_pair="red"))
    assert SdpaBwdDslSm100D128.dq_route_pair is None  # the record default: the TMA path
    # D2-pair under a red route has no reduce-add to skip: compile_host declines by name before any trace (host only: the key is never built)
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_2cta_host import compile_host

    with pytest.raises(ValueError, match="dq_reduce_add_off.*is not defined under dq_route_pair 'red_global'"):
        compile_host(
            (2, 8, 2, True),
            tuple((i, 1, 1, "int32") for i in range(9)),
            tuple("int32" for _ in range(9)),
            None,
            True,
            100,
            tmem_map="s_tail",
            diag_2cta="dq_reduce_add_off",
            dq_route_pair="red_global",
        )
    with pytest.raises(ValueError, match="sdq_sts_off.*is not defined under dq_route_pair 'red_global_pair'"):  # D9: no sDQ slice under a red route
        compile_host(
            (2, 8, 2, True),
            tuple((i, 1, 1, "int32") for i in range(9)),
            tuple("int32" for _ in range(9)),
            None,
            True,
            100,
            tmem_map="s_tail",
            diag_2cta="sdq_sts_off",
            dq_route_pair="red_global_pair",
        )
    with pytest.raises(ValueError, match="debug_clk is defined on tmem_map 's_tail' only"):  # the lever never reaches the first body
        compile_host((2, 8, 2, True), tuple((i, 1, 1, "int32") for i in range(9)), tuple("int32" for _ in range(9)), None, True, 100, debug_clk=True)


def test_2cta_stail_mask_form_is_typed_per_map():
    """``mask_form`` (lever L5) is a typed field resolved PER MAP: ``None`` = "fsel" on both maps (today's rendering); "predicated" and the
    DIAGNOSTIC "diag_mask_off" exist on s_tail only (declined BY NAME on dp_tail: the first body keeps its own per-cell select form); the kernel's
    constants are the config's code space (config = truth); the adapter reader prefixes the engine name; the record default is None."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import mask_form

    c = cfgmod.CFG
    assert c.MASK_FORMS == _MASK_FORMS and c.MASK_FORM_DEFAULT == "fsel" and c.MASK_FORM_DIAGNOSTIC == "diag_mask_off"
    assert cfgmod.MASK_FORM_CODES == {"fsel": 0, "predicated": 1, "diag_mask_off": 2} and tuple(cfgmod.MASK_FORM_CODES) == c.MASK_FORMS
    assert (c.MASK_SELECTS_PER_LANE_PER_TILE, c.MASK_WORDS_PER_LANE_PER_TILE, c.MASK_WORD_COLS) == (64, 2, 32)
    assert cfgmod.resolve_mask_form_for_map(None, "s_tail") == "fsel" and cfgmod.resolve_mask_form_for_map(None, None) == "fsel"
    for form, code in (("fsel", 0), ("predicated", 1), ("diag_mask_off", 2)):
        assert cfgmod.resolve_mask_form_for_map(form, "s_tail") == form and cfgmod.mask_form_code(form, "s_tail") == code
    assert cfgmod.mask_form_code(None, None) == 0 and cfgmod.mask_form_code("fsel", "dp_tail") == 0
    for bad_form, bad_map in (
        ("predicated", "dp_tail"),
        ("predicated", None),
        ("diag_mask_off", "dp_tail"),
        ("bogus", "s_tail"),
        ("", "s_tail"),
        ("FSEL", "s_tail"),
    ):
        with pytest.raises(ValueError, match="mask_form must be one of"):
            cfgmod.resolve_mask_form_for_map(bad_form, bad_map)
        with pytest.raises(ValueError, match="mask_form must be one of"):
            cfgmod.mask_form_code(bad_form, bad_map)
    assert mask_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail", mask_form="predicated")) == "predicated"
    assert mask_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail", mask_form="diag_mask_off")) == "diag_mask_off"
    assert mask_form(SimpleNamespace(_NAME="x")) == "fsel" and mask_form(SimpleNamespace(_NAME="x", tmem_map="s_tail")) == "fsel"
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: mask_form must be one of"):
        mask_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", mask_form="predicated"))  # the first body has ONE form
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: mask_form must be one of"):
        mask_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="s_tail", mask_form="pred"))
    assert SdpaBwdDslSm100D128.mask_form is None  # the record default: today's rendering
    body = _body_module()
    assert (body.MASK_FORM_FSEL, body.MASK_FORM_PREDICATED, body.MASK_FORM_DIAG_OFF) == tuple(cfgmod.MASK_FORM_CODES[f] for f in _MASK_FORMS)


def test_2cta_stail_tile_mask_predicate_is_exact_against_the_per_cell_mask():
    """Lever L5's interior-tile skip is bitwise ONLY if `tile_may_mask` is never FALSE on a tile that holds a masked cell (kernel header section 7).
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
    _adapter_class()  # the tree under test must be the one this venv imports
    script = _RENDER.format(host=_HOST_MODULE, census=repr(dict(_CENSUS, **(extra_census or {}))), form_kw=repr(form_kw))
    proc = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=600, env=dict(os.environ))
    assert proc.returncode == 0 and "RENDER_OK" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]
    md5 = re.search(r"PTX_MD5 ([0-9a-f]{32})", proc.stdout).group(1)
    census = {m.group(1): int(m.group(2)) for m in re.finditer(r"CENSUS (\S+) (\d+)", proc.stdout)}
    regs = [(m.group(1), int(m.group(2))) for m in re.finditer(r"SETMAXNREG (inc|dec) (\d+)", proc.stdout)]
    return md5, census, regs


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
    """Host trace-compile of the chain with ``tmem_map = "s_tail"`` (no device): the second body's PTX census is the design (header section 5 / 11),
    no local memory, no nanosleep, the default register split; the md5 matches the committed record at this geometry (skips, never widens, on
    another DSL build)."""
    from cudnn.frost.buffers import cutedsl_state

    md5, census, regs = _render(tmp_path, {"tmem_map": _TMEM_MAP})
    print(f"\n2-CTA s_tail sm_100a: PTX md5 {md5} census {census} setmaxnreg {regs}")
    assert census == _EXPECTED, census
    assert sorted(regs) == _setmaxnreg_of("144/144/80"), regs
    assert _RECORD.is_file(), _RECORD
    dsl, want = _record_md5(_RECORD, "d128_accum_grouped_2cta_stail")
    assert dsl and want, (dsl, want)
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: PTX text differs by DSL build, re-render the record")
    assert md5 == want, f"PTX md5 {md5} != the record's {want} -- the s_tail rendering changed"


def test_2cta_first_body_default_rendering_is_unchanged_by_the_stail_chain(tmp_path):
    """The chain edits that carry ``tmem_map`` (helpers dispatch, prepared host key + Constexpr, the adapter field) leave the FIRST body's
    DEFAULT rendering byte-identical: the first body's record md5 (RULE-GAPS 241) renders through the edited chain with every field unset."""
    from cudnn.frost.buffers import cutedsl_state

    md5, census, _ = _render(tmp_path, {})
    assert census == dict(_EXPECTED_FIRST_BODY, entry_stail=0, **_DAY3_EXPECTED_FIRST_BODY), census
    dsl, want = _parse_record(_RECORD_FIRST_BODY)
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if have != dsl:
        pytest.skip(f"the first body's record was rendered with {dsl}; installed {have}")
    assert md5 == want, f"the FIRST body's default rendering moved: {md5} != {want} (the arm-(ii) chain must leave tmem_map=dp_tail byte-identical)"


@pytest.mark.parametrize("probe", _PROBES)
def test_2cta_stail_host_renders_probes_for_sm_100a(tmp_path, probe):
    """``diag_2cta = <probe>`` on the s_tail body removes exactly its own ISSUE (D2-pair: the four reduce-adds), TRANSFER (D8s: the one bulk copy +
    its two mapa, the arm keeping its count with a 0-byte operand) or BYTES (D10: 28 of the 32 stats LDS.128; D9: the 24 sDQ slice STS + their 3
    proxy fences) and nothing else -- every arrive and wait site, every bar.warp.sync and tcgen05.fence stays (the barrier table is identical,
    barrier-inspector 2026-10-27 08:01 purity census); the probe renders a different md5 from the default."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod

    md5_d, census_d, _ = _render(tmp_path, {"tmem_map": _TMEM_MAP}, _ROUTE_CENSUS)
    assert census_d == dict(_EXPECTED, red_global_add_v4=0, sts_dq_v2_b64=24, fence_proxy_async=15, shfl_bfly=4, bulk_commit_group=5), census_d
    md5, census, _ = _render(tmp_path, {"tmem_map": _TMEM_MAP, "diag_2cta": probe}, _ROUTE_CENSUS)
    print(f"\n2-CTA s_tail sm_100a PTX census ({probe}): md5 {md5} {census}")
    assert census == dict(census_d, **_probe_census_delta(probe, cfgmod.CFG, census_d)), (probe, census)
    assert md5 != md5_d, f"{probe} must render different PTX from the default"
    with pytest.raises(ValueError, match="RETIRED on the s_tail map"):  # the retired D8 never renders on this body (compile_host resolves per map first)
        cfgmod.resolve_diag_2cta_for_map("ds_gates_off", "s_tail")


def test_2cta_stail_probes_resolve_the_same_way_under_fp32_dkv_accum(tmp_path):
    """C2b x the day-3 probes (the stack fold 2026-10-27; the barrier-inspector's C2b AUDIT item): the fp32 dK / dV staging of ``dkv_accum =
    "fp32_reduce"`` (header section 11) lives in the dS slab halves (own + recv) and XCHG, slots that rest on the three dS gates the RETIRED D8
    removed -- so ``ds_gates_off`` must stay DECLINED BY NAME on s_tail under BOTH field values (the resolver is per map, before the field is
    read; asserted here at compile_host, the chain's entry, exactly as the probe arms reach it).  ``exchange_off`` x ``fp32_reduce`` renders like
    every other probe: OUTPUT WRONG BY DESIGN (dQ: the recv half is never written), a different md5 from the fp32 default under the SAME
    mbarrier arrive / wait census with D8s's own transfer delta only (the arm's bytes go to 0, the one bulk copy + its two mapa gone; no
    barrier row moves; nothing asynchronous writes the recv-half slots under the probe).  The driver has no combined arm (the probe arms are
    bf16_partials), so the combination is reachable through the typed fields only; the ``--validate`` refusal is per probe name."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_2cta_host import compile_host

    for accum in (None, "fp32_reduce"):
        with pytest.raises(ValueError, match="ds_gates_off.*RETIRED on the s_tail map"):
            compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, tmem_map=_TMEM_MAP, diag_2cta="ds_gates_off", dkv_accum=accum)
    md5_fp32, census_fp32, _ = _render(tmp_path, {"tmem_map": _TMEM_MAP, "dkv_accum": "fp32_reduce"}, _ROUTE_CENSUS)
    md5, census, _ = _render(tmp_path, {"tmem_map": _TMEM_MAP, "dkv_accum": "fp32_reduce", "diag_2cta": "exchange_off"}, _ROUTE_CENSUS)
    print(f"\n2-CTA s_tail sm_100a PTX census (fp32_reduce x exchange_off): md5 {md5} {census}")
    assert md5 != md5_fp32, "exchange_off x fp32_reduce must render different PTX from the fp32 default"
    assert census == dict(census_fp32, **_probe_census_delta("exchange_off", cfgmod.CFG, census_fp32)), (census_fp32, census)


def test_2cta_stail_fields_under_16w_fa4_decline_or_resite_by_name(tmp_path):
    """X-FA4 x the other day-3 fields (the stack fold 2026-10-27): under ``warp_roles = "16w_fa4"`` the STAMPS lever ``debug_clk`` and the C2b
    ``dkv_accum = "fp32_reduce"`` DECLINE BY NAME at compile_host (the record layout names the 16w roles' waits; the fp32 staging rounds live in
    the 16w epilogue loop -- a re-site is the barrier-inspector's seam first), and the C2b row ``dvk_stage_empty`` is UNSERVED there (absent from
    ``BAR_LAYOUT_XFA4`` / ``BarsXFA4``); ``exchange_off`` (D8s) is RE-SITED to the softmax's arm + ship: under 16w_fa4 it renders with D8s's own
    transfer delta only (the one bulk copy + its two mapa gone, the arm's bytes 0; every arrive / wait stays), a different md5 from the 16w_fa4
    default; both pass under the default shape (16w) as before."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_2cta_host import compile_host

    with pytest.raises(ValueError, match="debug_clk is not defined under warp_roles '16w_fa4'"):
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, tmem_map=_TMEM_MAP, warp_roles="16w_fa4", debug_clk=True)
    with pytest.raises(ValueError, match="dkv_accum 'fp32_reduce' is not built under warp_roles '16w_fa4'"):
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, tmem_map=_TMEM_MAP, warp_roles="16w_fa4", dkv_accum="fp32_reduce")
    assert cfgmod.validate_fields_for_roles("16w", None, None, debug_clk=True, dkv_accum="fp32_reduce") == "16w"
    assert cfgmod.UNSERVED_BARRIERS_XFA4 == ("dvk_stage_empty",) and "dvk_stage_empty" not in {row[0] for row in cfgmod.BAR_LAYOUT_XFA4}
    assert "dvk_stage_empty" in {row[0] for row in cfgmod.BAR_LAYOUT} and "dvk_stage_empty" not in _body_module().BarsXFA4._fields
    md5_fa4, census_fa4, _ = _render(tmp_path, {"tmem_map": _TMEM_MAP, "warp_roles": "16w_fa4"}, _ROUTE_CENSUS)
    md5, census, _ = _render(tmp_path, {"tmem_map": _TMEM_MAP, "warp_roles": "16w_fa4", "diag_2cta": "exchange_off"}, _ROUTE_CENSUS)
    print(f"\n2-CTA s_tail sm_100a PTX census (16w_fa4 x exchange_off): md5 {md5} {census}")
    assert md5 != md5_fa4, "exchange_off x 16w_fa4 must render different PTX from the 16w_fa4 default"
    assert census == dict(census_fa4, **_probe_census_delta("exchange_off", cfgmod.CFG, census_fa4)), (census_fa4, census)


def test_2cta_stail_host_renders_debug_clk_for_sm_100a(tmp_path):
    """``debug_clk = True`` (the STAMPS lever, header section 11) renders the %clock64 brackets (and the two %globaltimer body stamps) on top of the
    SAME barrier census: 40 arrive / 68 try_wait / 11 expect_tx / 67 bar.warp.sync / 32 tcgen05.fence sites, the ship, the reduce-adds, the STS,
    no local memory, no nanosleep, the default register split -- under a different md5; the default (False) rendering is the record's."""
    md5_on, census_on, regs_on = _render(tmp_path, {"tmem_map": _TMEM_MAP, "debug_clk": True})
    print(f"\n2-CTA s_tail debug_clk sm_100a: PTX md5 {md5_on} census {census_on} setmaxnreg {regs_on}")
    # the lever adds the clock reads and the record dump's SMEM reads (the 100-word copy at exit vectorizes into 16-B loads); nothing else moves
    assert census_on["clock64"] > 0 and census_on["lds_v4_b32"] >= _EXPECTED["lds_v4_b32"], census_on
    assert census_on == dict(_EXPECTED, clock64=census_on["clock64"], lds_v4_b32=census_on["lds_v4_b32"]), census_on
    assert sorted(regs_on) == _setmaxnreg_of("144/144/80"), regs_on
    md5_off, census_off, _ = _render(tmp_path, {"tmem_map": _TMEM_MAP, "debug_clk": False})
    assert census_off == _EXPECTED and md5_off != md5_on, "debug_clk=False is the default rendering; True must differ (the clock reads)"
    dsl, want = _record_md5(_RECORD, "d128_accum_grouped_2cta_stail")
    assert md5_off == want or dsl, "debug_clk=False renders the record's md5 (a DSL build difference is a skip, not a widen)"


def test_2cta_stail_host_renders_knobs_and_plain_tma_for_sm_100a(tmp_path):
    """The perf-only knobs and the plain TMA form carry onto the s_tail body: ``store_wait_form = "two_instr"`` and ``tma_load_form = "plain"`` keep
    the census row for row (the plain form swaps the twelve loads' modifier), ``reg_split = "136/136/104"`` moves only the setmaxnreg immediates."""
    _md5, census, regs = _render(tmp_path, {"tmem_map": _TMEM_MAP, "store_wait_form": "two_instr"})
    assert census == _EXPECTED and sorted(regs) == _setmaxnreg_of("144/144/80"), (census, regs)
    _md5, census, regs = _render(tmp_path, {"tmem_map": _TMEM_MAP, "reg_split": "136/136/104"})
    assert census == _EXPECTED and sorted(regs) == _setmaxnreg_of("136/136/104"), (census, regs)
    _md5, census, _regs = _render(tmp_path, {"tmem_map": _TMEM_MAP, "tma_load_form": "plain"})
    assert census == dict(_EXPECTED, tma_cta_group_2=0, tma_plain_cta_group_2=_EXPECTED["tma_cta_group_2"]), census


def test_2cta_stail_host_renders_fa4b_for_sm_100a(tmp_path):
    """``pair_mma_order = "fa4b"`` (the leader issues dQ(t-1) AFTER dV(t): header section 5 MMA ISSUE ORDER fa4b, code-writer step 6) renders the
    SAME census as fa4 -- 72 pair MMAs, 13 commits, 40 arrive / 68 try_wait sites, no local memory, no nanosleep, the default register split:
    every wait keeps its object and its count, only the leader's in-tile position of the dQ slot moves -- under a DIFFERENT PTX md5; the fa4
    rendering is the default's (the record pin is the row above)."""
    md5_b, census_b, regs_b = _render(tmp_path, {"tmem_map": _TMEM_MAP, "pair_mma_order": "fa4b"})
    print(f"\n2-CTA s_tail fa4b sm_100a: PTX md5 {md5_b} census {census_b} setmaxnreg {regs_b}")
    assert census_b == _EXPECTED and sorted(regs_b) == _setmaxnreg_of("144/144/80"), (census_b, regs_b)
    md5_a, census_a, _ = _render(tmp_path, {"tmem_map": _TMEM_MAP, "pair_mma_order": "fa4"})
    md5_d, census_d, _ = _render(tmp_path, {"tmem_map": _TMEM_MAP})
    assert census_a == _EXPECTED and census_d == _EXPECTED and md5_a == md5_d, "fa4 named explicitly is the default rendering"
    assert md5_a != md5_b, "fa4 and fa4b must render different PTX (the dQ slot moves) under the same census"


def test_2cta_stail_host_renders_dq_routes_for_sm_100a(tmp_path):
    """``dq_route_pair = "red_global" | "red_global_pair"`` (header section 11, code-writer step 8): the fp32 dQ partial leaves through
    ``red.global.add.v4.f32`` straight from the dQ tcgen05.ld registers -- CFG.DQ_RED_V4_PER_LANE = 16 reds at the one epilogue site, the four
    ``cp.reduce.async.bulk`` and the sDQ ring's 24 fp32 slice STS (``st.shared.v2.b64``) + their 3 proxy fences GONE, the per-use bulk commit groups
    gone (the item-end one stays) -- under
    the SAME mbarrier census (40 arrive / 68 try_wait: the ring protocol is unchanged), no local memory, no nanosleep, the default register
    split; the paired form adds exactly its 8 sectors x 4 lane-pair shuffles; the default (tma) rendering is byte-identical to the record."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod

    c = cfgmod.CFG
    md5_t, census_t, regs_t = _render(tmp_path, {"tmem_map": _TMEM_MAP}, _ROUTE_CENSUS)
    sts_tma = 3 * (c.DQ_BOX // 4)  # the three _dq_round sites (round B in-tile, round A, the item-end round B) x 8 fp32 16-B rows each
    fences_tma = census_t["fence_proxy_async"]  # the body's proxy-fence sites (15 at step 5: 3 of them are the _dq_round fences)
    want_t = dict(
        _EXPECTED, red_global_add_v4=0, sts_dq_v2_b64=sts_tma, fence_proxy_async=fences_tma, shfl_bfly=census_t["shfl_bfly"], bulk_commit_group=c.DQ_SLICES + 1
    )
    assert census_t == want_t and fences_tma >= 3, census_t
    dsl, want = _record_md5(_RECORD, "d128_accum_grouped_2cta_stail")
    assert md5_t == want or dsl, "the default (tma) rendering must stay the record's (a DSL build difference is a skip, not a widen)"
    seen = {}
    for route in _RED_ROUTES:
        md5_r, census_r, regs_r = _render(tmp_path, {"tmem_map": _TMEM_MAP, "dq_route_pair": route}, _ROUTE_CENSUS)
        print(f"\n2-CTA s_tail {route} sm_100a: PTX md5 {md5_r} census {census_r} setmaxnreg {regs_r}")
        shfl = census_t["shfl_bfly"] + (c.DQ_RED_SECTORS_PER_LANE * 4 if route == "red_global_pair" else 0)
        want_r = dict(
            _EXPECTED,
            reduce_add=0,
            red_global_add_v4=c.DQ_RED_V4_PER_LANE,
            sts_dq_v2_b64=0,
            fence_proxy_async=fences_tma - 3,
            shfl_bfly=shfl,
            bulk_commit_group=1,
        )
        assert census_r == want_r, (route, census_r)
        assert sorted(regs_r) == _setmaxnreg_of("144/144/80"), regs_r
        assert md5_r != md5_t, f"{route} must render different PTX from the default (the reds replace the STS + reduce-adds)"
        seen[route] = md5_r
    assert seen["red_global"] != seen["red_global_pair"], "the two red forms differ (the lane-pair shuffles)"


# ------------------------------------------------------------------------------------------------------------ host: the inspector's deterministic-hang topologies as SOURCE assertions
def test_2cta_stail_host_renders_mask_forms_for_sm_100a(tmp_path):
    """Lever L5's three mask forms on this body (header section 11): "fsel" named explicitly IS the default rendering (the record's md5); "predicated"
    renders a DIFFERENT PTX under the SAME mbarrier census (no wait, arrive or fence moves), the same setmaxnreg split, the SAME number of per-cell
    selects and keep-word shifts (the mask body is kept, inside the branch) and exactly CFG.QUARTERS_PER_GROUP more branches (one CTA-uniform
    `if mask_tile:` per q quarter); "diag_mask_off" drops exactly CFG.MASK_SELECTS_PER_LANE_PER_TILE selects and CFG.MASK_WORDS_PER_LANE_PER_TILE
    upper-edge shifts (the whole mask body) under the same census; the three md5s differ."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG

    md5_d, census_d, regs_d = _render(tmp_path, {"tmem_map": _TMEM_MAP}, _MASK_CENSUS)
    md5_f, census_f, regs_f = _render(tmp_path, {"tmem_map": _TMEM_MAP, "mask_form": "fsel"}, _MASK_CENSUS)
    md5_p, census_p, regs_p = _render(tmp_path, {"tmem_map": _TMEM_MAP, "mask_form": "predicated"}, _MASK_CENSUS)
    md5_o, census_o, regs_o = _render(tmp_path, {"tmem_map": _TMEM_MAP, "mask_form": "diag_mask_off"}, _MASK_CENSUS)
    print(
        f"\n2-CTA s_tail mask forms sm_100a: default {md5_d} fsel {md5_f} predicated {md5_p} diag_mask_off {md5_o}; census {census_d} | {census_p} | {census_o}"
    )
    assert md5_f == md5_d and census_f == census_d and regs_f == regs_d, "fsel named explicitly is the default rendering"
    dsl, want = _record_md5(_RECORD, "d128_accum_grouped_2cta_stail")
    have = __import__("cutlass").__version__ if hasattr(__import__("cutlass"), "__version__") else None
    if dsl is not None and have is not None and dsl.endswith(have):
        assert md5_d == want, f"the default (fsel) rendering {md5_d} != the record's {want}: the L5 chain must leave the default byte-identical"
    assert len({md5_d, md5_p, md5_o}) == 3, "the three mask forms must render three different PTX texts"
    for census in (census_d, census_p, census_o):
        assert {k: v for k, v in census.items() if k not in _MASK_CENSUS} == _EXPECTED, census  # no barrier, TMA or TMEM op moves under any form
    assert sorted(regs_d) == sorted(regs_p) == sorted(regs_o) == _setmaxnreg_of("144/144/80"), (regs_d, regs_p, regs_o)
    assert census_p["selp_f32"] == census_d["selp_f32"] and census_p["shr_u32"] == census_d["shr_u32"], (census_d, census_p)
    assert census_p["bra"] - census_d["bra"] == CFG.QUARTERS_PER_GROUP, (census_d["bra"], census_p["bra"])
    assert census_d["selp_f32"] - census_o["selp_f32"] == CFG.MASK_SELECTS_PER_LANE_PER_TILE, (census_d, census_o)
    assert census_d["shr_u32"] - census_o["shr_u32"] == CFG.MASK_WORDS_PER_LANE_PER_TILE, (census_d, census_o)


def test_2cta_stail_dp_empty_is_arrived_at_the_ds_readout_before_the_recv_arm():
    """RED 3 (barrier-inspector ARM (ii) section 8): dp_empty left at the first body's site (after the dQ readout) under the rotated order is a
    deterministic HANG (dP^T(t+1) waits a readout of a dQ issued after it).  The epilogue's ONE dp_empty arrive precedes the ds_recv_full arm
    and the dQ readout."""
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
    ``_dq_slot`` and at the tail), never ready_state.  The in-tile slot is ONE helper at TWO ``const_expr(dq_after_dv)`` sites (fa4: before dV(t);
    fa4b: after it) so the two orders cannot drift; the s_read wait lives inside it, every tile, and nowhere else in the leader."""
    source = _body_source()
    gates = re.findall(r"_wait_tcgen\(bars\.(ds_smem|ds_recv_full|ds_pair_ready), (\w+)\.phase, dbg, _W\[\"wait:leader:\w+\"\]\)", source)
    assert sorted(g[0] for g in gates) == ["ds_pair_ready", "ds_recv_full", "ds_smem"], gates
    assert {g[1] for g in gates} == {"xchg_state"}, gates
    issue = _function_code(source, "_dq_issue", "_dq_slot")
    # the three gates are UNCONDITIONAL (no const_expr guard: they are the ds_recv_full arm's only back-edge, barrier-inspector 08:01); the only
    # trace-time branch in the helper is the STAMPS handshake, which touches no barrier
    # (the stack fold: writer B's test-only skew lever 7 and the section-15 dq_war_done commit are the two other trace-time branches; neither guards a gate)
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
    assert len(slots) == 2 and len(issues) == 1, (slots, issues)  # the in-tile dQ(t-1) slot at the two order sites + the tail dQ(hi-1)
    assert leader[slots[0] - 1].strip() == "if cutlass.const_expr(not dq_after_dv):", leader[slots[0] - 1]
    assert leader[slots[1] - 1].strip() == "if cutlass.const_expr(dq_after_dv):", leader[slots[1] - 1]
    dv = [i for i, ln in enumerate(leader) if "_ts(grad, tmem_base + CFG.P_COL" in ln]
    assert len(dv) == 1 and slots[0] < dv[0] < slots[1], "fa4: the dQ slot before dV(t); fa4b: after it"
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


def test_2cta_stail_red_route_follows_the_dq_readout_and_keeps_the_ring_protocol():
    """SOURCE assertion (step 8, barrier-inspector AUDIT 04:17 rows dq_empty / dq_full / s_tail_empty): under a red route the reds are issued AFTER
    the dQ tcgen05.ld + wait(LOAD) + the s_tail_empty arrive (the image is in registers), inside the epilogue's tile loop, through the ONE helper
    `_dq_red_out`; `_dq_round` keeps BOTH dq_empty waits and the dq_full arrive under every route and skips only the STS + proxy fence under the red
    one; the store warp keeps its dq_full waits, its per-round `tma_store_wait` and both dq_empty arrives and skips only the reduce-add issue + its
    commit.  The arrive / wait census of the two routes is identical (the render row above pins it in PTX)."""
    src = _body_source()
    epi = _function_code(src, "_epilogue_loop", "_mma_loop_leader")
    ld = next(i for i, ln in enumerate(epi) if "dq = _tmem_load(tmem_base + CFG.DQ_COL" in ln)
    arrive = next(i for i, ln in enumerate(epi) if "_arrive_leader(bars.s_tail_empty)" in ln)
    red = next(i for i, ln in enumerate(epi) if "_dq_red_out(" in ln)
    round_a = next(i for i, ln in enumerate(epi) if "_dq_half(dq, 0), empty_state, route_red" in ln)
    assert ld < arrive < red < round_a, (ld, arrive, red, round_a)
    assert any("nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)" in ln for ln in epi[ld:arrive]), "wait(LOAD) between the dQ ld and the arrive"
    assert epi[red - 2].strip().startswith("if cutlass.const_expr(route_red)") or "const_expr(route_red)" in " ".join(epi[red - 4 : red]), epi[red - 4 : red]
    assert sum("_dq_round(" in ln for ln in epi) == 3 and all("route_red" in ln for ln in epi if "_dq_round(" in ln), "every round carries the route"
    rnd = _function_code(src, "_dq_round", "_dq_red_out")
    waits = [ln for ln in rnd if '_wait(bars.dq_empty[s], empty_state.phase, dbg, _W["wait:epilogue:dq_empty"])' in ln]
    assert len(waits) == 1 and any("range_constexpr(CFG.DQ_SLOTS)" in ln for ln in rnd), "both slots waited per round under every route"
    i_if = next(i for i, ln in enumerate(rnd) if "if cutlass.const_expr(not route_red and not sts_off)" in ln)  # the red route OR D9 (sdq_sts_off)
    assert "_store_dq_slice(" in rnd[i_if + 1] and "fence_proxy" in rnd[i_if + 2], "only the STS + its proxy fence are route- / probe-gated"
    assert any("bars.dq_full[dhalf].arrive()" in ln for ln in rnd[i_if + 3 :]), "the dq_full arrive is unconditional"
    red_fn = _function_code(src, "_dq_red_out", "_softmax_loop")
    assert any("if live:" in ln for ln in red_fn) and any(
        "live_even, live_odd" in ln for ln in red_fn
    ), "OOB rows skipped by a branch (never a predicate operand)"
    store = _function_code(src, "_store_loop", "_scheduler_loop")
    gate = next(i for i, ln in enumerate(store) if "if cutlass.const_expr(dq_route_code == DQ_ROUTE_TMA):  # a red route" in ln)
    assert any("tma_reduce_add_subtile(" in ln for ln in store[gate : gate + 6]) and any("tma_store_commit()" in ln for ln in store[gate : gate + 7])
    after = store[gate + 7 : gate + 15]  # the per-slot release loop (with the STAMPS bracket of the store drain inside it)
    assert any("tma_store_wait(CFG.DQ_SLOTS - 1 - s)" in ln for ln in after) and any("bars.dq_empty[s].arrive()" in ln for ln in after), after
    assert sum("_wait_store(bars.dq_full[slot]" in ln for ln in store) == 1, "the dq_full wait stays, unconditional"


def test_2cta_stail_exchange_off_keeps_every_wait_and_fakes_the_bytes():
    """SOURCE assertion for D8s `exchange_off` (barrier-inspector 2026-10-27 08:01, the safe exchange ceiling): in the epilogue the ONE ds_recv_full
    arm keeps its elected lane and arms the config-selected byte count (CFG.DIAG_EXCHANGE_OFF_TX = 0 under the probe, XCHG_TX otherwise); the ONLY
    statement under the probe's guard is the `_ship_xchg` call (the transfer); the XCHG STS, its proxy fence, the CFG.XCHG_BARRIER named barrier,
    the own-half STS and every wait / arrive of the epilogue are unguarded; the three dS gates (the arm's back-edge) are unconditional in
    `_dq_issue` (asserted with RED 4 above); the relay in the follower is unconditional."""
    src = _body_source()
    epi = _function_code(src, "_epilogue_loop", "_mma_loop_leader")
    arms = [ln for ln in epi if "bars.ds_recv_full.arrive(" in ln]
    assert arms == [ln for ln in arms if "n_bytes=xchg_tx, pred=ship_pred" in ln] and len(arms) == 1, arms
    assert any(
        ln.strip() == "xchg_tx = CFG.DIAG_EXCHANGE_OFF_TX if cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF) else CFG.XCHG_TX" for ln in epi
    ), "the arm's byte count is the config-selected constant (0 under D8s)"
    guard = [i for i, ln in enumerate(epi) if ln.split("#")[0].strip() == "if cutlass.const_expr(not exchange_off):"]
    assert len(guard) == 1 and "_ship_xchg(slab, bars, rank, ship_pred)" in epi[guard[0] + 1], epi[guard[0] : guard[0] + 2]
    depth = len(epi[guard[0]]) - len(epi[guard[0]].lstrip())
    assert len(epi[guard[0] + 2]) - len(epi[guard[0] + 2].lstrip()) <= depth, "only the ship sits under the D8s guard"
    for token in (
        "_store_words(slab, CFG.XCHG",
        "named_barrier_fence(CFG.XCHG_BARRIER",
        "_store_words(slab, own_base",
        "bars.ds_smem.arrive()",
        "_arrive_leader(bars.dp_empty)",
    ):
        lines = [ln for ln in epi if token in ln]
        assert len(lines) == 1 and len(lines[0]) - len(lines[0].lstrip()) == depth, (token, lines)  # at the loop body's depth: never guarded
    assert not any("exchange_off" in ln and ("_wait" in ln or ".arrive(" in ln) for ln in epi), "D8s guards no wait and no arrive"
    relay = _function_code(src, "_mma_loop_follower", "_reduce_loop")  # the stack: writer B's _reduce_loop sits before _load_loop
    assert sum('_wait(bars.ds_recv_full, relay_state.phase, dbg, _W["wait:follower:ds_recv_full"])' in ln for ln in relay) == 1
    assert not any("diag" in ln.lower() for ln in relay), "no probe reaches the relay"


def test_2cta_stail_skew_levers_follow_the_house_form():
    """The second body exposes the four carried levers plus SOFTMAX_S_READ / EPILOGUE_DQ_READOUT, each in the house form (module global + const_expr
    nanosleep loop, default 0 = no traced code: the default render's census pins nanosleep == 0)."""
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
    """The second body serves grouped THD only: dense and MHA requests decline BY NAME before any compile (the first body's declines, same route)."""
    _require_2cta()
    dense = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=False)
    with pytest.raises(NotImplementedError, match="2-CTA datapath requires THD"):
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


@pytest.mark.parametrize("order", _ORDERS)
def test_2cta_stail_persistent_reuse_more_pairs_than_sms(order):
    """P14 row: n_q == 1 with more pairs than SMs (a cluster claims a second CLC item; the N == 1 tail path: the tail dQ(lo) without an s_tail_empty
    wait, two rounds per item) plus a half-dead last pair (seq_k % 256 == 128) and zero-trip blocks past seq_q."""
    _require_2cta()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    hq, b = 8, 2
    skv = 256 * ((sms // (hq * b)) + 2) + 128
    case = make_case((128, 64), (skv, skv - 256), hq=hq, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _run_case(case, pair_mma_order=order)  # under both leader orders (fa4 | fa4b): the same flips on every path


@pytest.mark.parametrize("order", _ORDERS)
def test_2cta_stail_zero_trip_mix(order):
    """The barrier-inspector's N-walk as one case: seq_q < seq_k under causal makes the kv blocks past the last q tile zero-trip, so one launch runs
    items with N = 3, 1 and 0 (seq_q 300 / seq_k 2000), N = 1 and 0 (128 / 700), N = 8, 6, 4, 2 (1000 / 1000): every s_tail_empty wait-site guard
    (prologue always, lookahead iff qtile > lo, tail iff hi - 1 > lo) and the pending-completion carry across zero-trip items are exercised."""
    _require_2cta()
    case = make_case((300, 128, 1000), (2000, 700, 1000), hq=4, hkv=2, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _run_case(case, pair_mma_order=order)  # under both leader orders (fa4 | fa4b): the same flips on every path


def test_2cta_stail_launch_count_and_cluster_facts():
    """The chain is five launches with the stail MAIN kernel (the first body's kernel absent); the header's facts come from the s_tail CFG."""
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


def test_2cta_tmem_maps_agree_bitwise_on_dk_dv():
    """The two pair bodies are the SAME function on dK / dV: every accumulation chain per GEMM (dV += P^T dO over the tiles in ascending t, dK
    likewise, the same TS k-order) is unchanged by the TMEM re-plan and the rotated issue order, so the deterministic native partials + the
    fp32 group fold agree BITWISE between tmem_map = "dp_tail" and "s_tail" (G09 twin-diff); dQ (reduce-add order across pairs) passes the
    fp64 gate under each.  Four tiles per pair so the lookahead, the in-tile dQ and the tail are all exercised."""
    _require_2cta()
    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _bitwise_twin(case, lambda c: make_adapter_2cta(c), lambda c: make_adapter_2cta_stail(c), "the two TMEM maps (dp_tail vs s_tail)")


def test_2cta_stail_mma_orders_agree_bitwise_on_dk_dv():
    """fa4 vs fa4b are the SAME function on dK / dV by construction (one accumulation chain per GEMM; only the leader's in-tile position of the
    dQ(t-1) issue moves, header section 5), pinned BITWISE (G09 twin-diff); dQ by the fp64 gate under each.  Four tiles per pair so the
    lookahead, the in-tile dQ slot and the tail are all exercised under both orders."""
    _require_2cta()
    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    fa4, fa4b = (lambda c: make_adapter_2cta_stail(c, pair_mma_order="fa4")), (lambda c: make_adapter_2cta_stail(c, pair_mma_order="fa4b"))
    _bitwise_twin(case, fa4, fa4b, "the two leader orders (fa4 vs fa4b)")


def _dq_route_twin(case, route):
    """tma vs a red route on ONE dataset: dK / dV BITWISE (they never touch the route), dQ a TOLERANCE twin at the suite's own gate (``_TOL`` of the
    dtype: both routes sum the fp32 partials in a non-deterministic order, so dQ is never a bitwise claim) after EACH route passed the fp64 gate."""
    from test_sdpa_bwd_dsl_sm107 import _TOL

    grads = {}
    for tag, fields in (("tma", {}), (route, {"dq_route_pair": route})):
        api = make_adapter_2cta_stail(case, **fields)
        workspace = make_workspace(api.scratch_workspace_bytes())
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
        grads[tag] = {role: case.tensors[role][0].detach().clone() for role in ("dq", "dk", "dv")}
    for role in ("dk", "dv"):
        assert torch.equal(grads["tma"][role].view(torch.int16), grads[route][role].view(torch.int16)), f"{role}: tma vs {route} differ bitwise"
    for i, (start, length) in enumerate(zip(case.origins["dq"], case.lens_q, strict=True)):
        a, b = grads["tma"]["dq"][start : start + length].float(), grads[route]["dq"][start : start + length].float()
        torch.testing.assert_close(b, a, **_TOL[case.dtype], msg=lambda m: f"dq/seq{i}: tma vs {route}: {m}")


@pytest.mark.parametrize("route", _RED_ROUTES)
def test_2cta_stail_dq_routes_agree_on_dk_dv_bitwise_and_dq_at_the_gate(route):
    """``dq_route_pair`` is perf-only on dK / dV (bitwise vs tma: the route never touches their chains) and a tolerance twin on dQ (header section 11,
    step 8); four tiles per pair so every dQ round (in-tile B, A, the item-end B) and the clip-free full tiles run under both routes."""
    _require_2cta()
    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _dq_route_twin(case, route)


@pytest.mark.parametrize("route", _RED_ROUTES)
@pytest.mark.parametrize(
    "lq,lk,hq,hkv,causal,layout,offset_dtype,units",
    [
        pytest.param((256, 512, 768, 1024), (256, 512, 768, 1024), 4, 2, True, "TH1", torch.int32, "tokens", id="ramp-1024-causal"),
        pytest.param((129, 0, 127, 1), (65, 128, 0, 3), 4, 2, True, "TH1", torch.int32, "tokens", id="suite-degenerate-causal"),
        pytest.param((300, 300), (300, 300), 3, 1, True, "TH1", torch.int64, "elements", id="pair-tail-300"),
        pytest.param((300, 128, 1000), (2000, 700, 1000), 4, 2, True, "TH1", torch.int32, "tokens", id="zero-trip-mix"),
    ],
)
def test_2cta_stail_dq_route_matrix(route, lq, lk, hq, hkv, causal, layout, offset_dtype, units):
    """The red routes on the rows that exercise their one runtime guard (`row < seq_q`: S % 256 = 44 / 300 / 129 / 127 / 1 tails, empty sequences,
    the odd-length tail of the paired form) and the N-walk (zero-trip items between live ones); every row = the fp64 gate on dQ / dK / dV."""
    _require_2cta()
    case = make_case(lq, lk, hq=hq, hkv=hkv, thd=True, dtype=torch.bfloat16, causal=causal, stats_layout=layout, offset_dtype=offset_dtype, units=units)
    _run_case(case, dq_route_pair=route)


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
    """Lever L5's twin (G09 twin-diff, a claim BY CONSTRUCTION: a masked P / dS cell is exactly +0.0 under fsel and under predicated, and an interior
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


def test_2cta_stail_tma_load_forms_agree_bitwise_on_dk_dv():
    _require_2cta()
    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _bitwise_twin(
        case, lambda c: make_adapter_2cta_stail(c, tma_load_form="multicast"), lambda c: make_adapter_2cta_stail(c, tma_load_form="plain"), "the TMA load forms"
    )


@pytest.mark.parametrize("field,values", [("store_wait_form", ("retry", "two_instr")), ("reg_split", ("144/144/80", "136/136/104"))])
def test_2cta_stail_knobs_agree_bitwise_on_dk_dv(field, values):
    _require_2cta()
    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _bitwise_twin(
        case, lambda c: make_adapter_2cta_stail(c, **{field: values[0]}), lambda c: make_adapter_2cta_stail(c, **{field: values[1]}), f"the {field} values"
    )


# ------------------------------------------------------------------------------------------------------------ GPU: the day-3 probes + the STAMPS lever
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
    output is WRONG BY DESIGN and is NOT checked here -- `--validate` refuses them by name in the driver.  D8s is the SAFE form (every wait kept)."""
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
    """The mechanism's repro pair (barrier-inspector 2026-10-27 08:01): `ds_gates_off` + `LEADER_SHIP` 300 us at s=1024 FAULTED (CUDA 719) at tile
    lo+1 of every item -- the leader's late ship let the follower's ds_recv_full arm(t+1) land on the tx-pending phase t.  `exchange_off` (D8s)
    under the SAME lever must run clean: no copy exists, every phase completes at its own arm (output wrong by design, not checked).  A fresh
    process arms the lever before the body executes (the skew module's form); exit 70 / a timeout = HANG, a CUDA error = the fault class."""
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
    _SITES_FIRST_BODY,
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
        "observation of dq_done(t-1) ALONE (X-FA4 X7; barrier-inspector x-fa4 table section 12 item 12); under 16w the epilogue's dq_done(t-1) "
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

# C2b (``dkv_accum = "fp32_reduce"``, e3c9f741; the barrier-inspector's AUDIT 2026-10-27 10:30 table B "alias seams"): the fp32 dV / dK slices are
# staged in sK2's halves, BOTH dS halves (own + RECV = where the PEER's ship lands) and XCHG, drained through the NEW row dvk_stage_empty and the
# dvk_empty / k2_empty chain.  Exactly two levers reach those seams, so exactly two ids get an fp32 twin row (the other four levers' seams are
# the same bytes under both forms: no row, the module stays small).  The default-form ids keep their names.  Budget: ``skew_module``
# (config/budgets.yaml; every row is a fresh child = cold import + JIT, ~160-240 s each + ~240 s parent preamble; <= 4 rows per -k half).
_DKV_ACCUM_SEAM_ROWS = {
    "leader_ship": "recv-half seam (table B: sDS h0 / h1): round 1 stages fp32 dV slices 2 / 3 (round 2: dK slice 3) in the dS slab whose RECV half "
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
    pass criterion is the SAME gate (the fp32 form is tighter on every cell: writer C's validate table) -- never a widened tolerance."""
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
