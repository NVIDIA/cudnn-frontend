# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L0 tests of the d128 grouped THD cta_group::2 pair body (section 8 of ``kernels/sm100/bprop_d128_f16.py``: the dQ half-row image in S^T's dead
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
The TMEM MAP rows (the typed, PART-KEYED field ``tmem_map``; the kernel header sections 9 / 11 / 13): the typed domain and the per-part defaults (s_tail
on every part; dq_576 selectable by name on cc 10.7 -- a neutral structural map, no perf claim), the declines by name (dq_576 off cc 10.7 = the
engine's NotImplementedError; dq_576 x 16w_fa4; the S-tail WAIT probes x dq_576), the sm_107a renderings (the s_tail default against the sm_100a record:
target-independent per-kernel md5s; dq_576 NAMED against its own record), the census delta of the map (exactly the two deleted rows' arrive / wait /
fence sites + the exclusive alloc), the RED-first SOURCE
assertion of the independent image's reuse edge (the epilogue's readout trio BEFORE the next tile's staging), the dK / dV twin of the two maps where a
part serves both, the STAMPS dump under the map (the four S-tail words dead).
The SMEM MAP rows (the typed, PART-KEYED field ``smem_map``; the kernel header sections 5 / 8 / 11; arm R2.c): the typed domain and the per-part
defaults (sm100 on every part; rubin_b2 selectable by name on cc 10.7 together with tmem_map dq_576 -- the 327 KiB oversized carveout: sQ1 x2, sDO1 x2,
the TWO-STAGE sDS + XCHG staging ring with the lagging epilogue order and the LEADER row dq_read_done), the declines by name (rubin_b2 off cc 10.7 = the
engine's NotImplementedError; rubin_b2 x s_tail / 16w_fa4 / ldg), the sm_107a rendering (dq_576 x rubin_b2 NAMED against its own record; dq_576 x
sm100 = the dq_576 record), the census delta of the map (the row's arrive / wait sites, the extra rounds' sites, the grown slots' inits), the RED-first
SOURCE assertion of the row (the arrive after each readout trio, the wait before every dQ issue, the pre-armed state and its drain), the dK / dV twin of
the two SMEM maps where a part serves both (the P14 killers: zero-trip mix, all-empty, n_q == 1 with more pairs than SMs), the STAMPS dump and the
skew children under the map.
Budget: the skew rows and the probe-skew child name ``skew_module`` (every row is a fresh child = cold import + JIT).
"""

import os
import pathlib
import re
import subprocess
import sys

import pytest
import torch

from d128_2cta_test_utils import CELLS, CENSUS as CENSUS_BASE, CHILD_TIMEOUT_S, EXPECTED_BASE, GLOBAL_PREFIX, LEVER_LINE, SKV, WATCHDOG_S
from d128_2cta_test_utils import CHILD_EXIT_HANG, SITES_BASE, _adapter_class, _require_2cta, _setmaxnreg_of, make_adapter_2cta, render_2cta
from d128_2cta_test_utils import render_2cta_kernels, skip_if_child_declined
from d128_grouped_test_utils import adapter_arguments, check_case, make_case, make_workspace, poison
from frost_test_utils import D128_THD_SHARED_KERNELS, assert_kernel_record, read_kernel_record, requires_dsl

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
_BODY_MODULE = "cudnn.sdpa.bwd.kernels.sm100.bprop_d128_f16"  # the d128 family module; the pair body is its section 8
_BODY_REL = pathlib.Path("python/cudnn/sdpa/bwd/kernels/sm100/bprop_d128_f16.py")
_BODY_SECTION = ("# section 8:", "# section 9:")  # the pair body's banners in the family module: the source assertions read that section only
_TREE = pathlib.Path(__file__).resolve().parents[4]  # test/python/sdpa/frost/<file> -> the checkout
_RECORD = pathlib.Path(__file__).resolve().parent / "renderings" / "md5_d128_accum_grouped_2cta_stail_default_sm_100a.txt"
_RECORD_DEV = _RECORD.with_name("md5_d128_accum_grouped_2cta_stail_default_sm_100a_dsl_4_9_dev_20261028.txt")
_RECORD_TOKEN = "d128_accum_grouped_2cta_stail"
_RECORD_TOKEN_LDG = "d128_accum_grouped_2cta_stail_ldg"  # the stats_form = "ldg" rendering's rows (the previous default), beside the default's in the same file
_RECORD_TOKEN_DENSE = "d128_accum_grouped_2cta_stail_dense"  # the dense compact-BSHD twin's rendering (layout = "dense"): its own rows in the same file
_MAIN_KERNEL = "bprop_d128_accum_grouped_2cta_stail_kernel"
# The TMEM MAP (PART-KEYED): tmem_map "dq_576" NAMED on cc 10.7 (the independent dQ image at [512, 576) of the 576-column carve; the s_read /
# s_tail_empty rows deleted; selectable by name, not the default) has its own record at its own target; the default (s_tail) on sm_107a renders the
# sm_100a record's rows.
_RECORD_DQ576 = pathlib.Path(__file__).resolve().parent / "renderings" / "md5_d128_accum_grouped_2cta_stail_dq576_sm_107a.txt"
_RECORD_TOKEN_DQ576 = "d128_accum_grouped_2cta_stail_dq576"
_RECORD_TOKEN_DQ576_LDG = "d128_accum_grouped_2cta_stail_dq576_ldg"  # tmem_map dq_576 x stats_form ldg, both NAMED: its own rows in the same file
_RECORD_ARCH_DQ576 = "sm_107a"
# The SMEM MAP (PART-KEYED): smem_map "rubin_b2" NAMED together with tmem_map "dq_576" on cc 10.7 (the 327 KiB oversized carveout: sQ1 x2, sDO1 x2, the
# two-stage sDS + XCHG staging ring with the lagging epilogue order and the dq_read_done row; the kernel header sections 5 / 8 / 11; selectable by name,
# not the default) has its own record at its own target; the dq_576 x sm100 rendering keeps the dq_576 record's rows.
_RECORD_B2 = pathlib.Path(__file__).resolve().parent / "renderings" / "md5_d128_accum_grouped_2cta_stail_dq576_b2_sm_107a.txt"
_RECORD_TOKEN_B2 = "d128_accum_grouped_2cta_stail_dq576_b2"
_SMEM_MAPS = ("sm100", "rubin_b2", "sm100_oversized")  # CFG.SMEM_MAPS (asserted in the typed row); None = the PART's default (sm100 everywhere)
_RING_TWIN_MAPS = ("sm100", "rubin_b2")  # the two-stage ring's twin pair (the DIAGNOSTIC sm100_oversized map has its own byte-identity row below)
_OVS_FIELDS = {"smem_map": "sm100_oversized"}  # the DIAGNOSTIC map: the sm100 map declared in the oversized carveout (cc 10.7 by name, never a default)
_B2_FIELDS = {"tmem_map": "dq_576", "smem_map": "rubin_b2"}  # the B2 rendering's typed fields (BOTH named: neither is a part default today)
_TMEM_MAPS = ("s_tail", "dq_576")  # CFG.TMEM_MAPS (asserted in the typed row); None = the PART's default
# The kernels the pair chain OWNS under the DEFAULT form (tma_ring: the delta-and-stats launch; one record row each; the group fold
# dkv_reduce_thd_origins_kernel is shared with the one-CTA chain: rendered, never pinned here -- frost_test_utils.D128_THD_SHARED_KERNELS);
# under "ldg" the delta launch is the compact one.
_OWN_KERNELS = ("setup_thd_d128_2cta_kernel", "delta_zero_stats_thd_d128_2cta_kernel", "bprop_d128_accum_grouped_2cta_stail_kernel", "cast_dq_thd_d128_kernel")
_OWN_KERNELS_LDG = ("setup_thd_d128_2cta_kernel", "delta_zero_thd_d128_kernel", "bprop_d128_accum_grouped_2cta_stail_kernel", "cast_dq_thd_d128_kernel")
# The probes' purity census: the stats / delta LDS.128 (`ld.shared.v4.b32`: stats_one_chunk removes exactly CFG.DIAG_STATS_LDS_REMOVED_PER_LANE
# static sites), the expect_tx arms (exchange_off keeps the count), the %clock64 reads (0 unless `debug_clk`), and the three counts an issue /
# transfer / bytes probe must leave untouched: `bar.warp.sync`, `tcgen05.fence::after_thread_sync`, `mapa` (exchange_off loses the ship's own
# two).  The WAIT probes (s_tail_wait_off / s_read_wait_off / both) lower try_wait, bar.warp.sync and tcgen05.fence::after by exactly their
# static site counts (CFG.DIAG_S_TAIL_WAIT_SITES / DIAG_S_READ_WAIT_SITES) and move no arrive.
_PROBE_CENSUS = {
    "lds_v4_b32": r"ld\.shared\.v4\.b32",
    "expect_tx": r"mbarrier\.arrive\.expect_tx",
    "clock64": r"%clock64",
    "bar_warp_sync": r"bar\.warp\.sync",
    "tcgen05_fence_after": r"tcgen05\.fence::after_thread_sync",
    "mapa": r"mapa\.",
}
_CENSUS = dict(CENSUS_BASE, entry_stail=r"\.entry\s+cudnn_kernel_bprop_d128_accum_grouped_2cta_stail_kernel", **_PROBE_CENSUS)
# The body's census against the base counts (kernel header section 5 / 11): dQ is issued at TWO sites (in-tile dQ(t-1) + the tail dQ(hi-1):
# 16 + 16 k-steps -> mma 56 + 16 = 72), the tail's dq_done commit (+1), the arrive sites 41 - 4 (a dq_empty bootstrap + two per-slot sites of a
# 4-slice ring -> 2 slots x 2 rounds) + 1 (dq_full per round, 3 sites) + 1 (s_read) + 1 (s_tail_empty) = 40, the wait sites 63 - 2 (dq_empty:
# 3 rounds x 2 slots vs 4 + 4 terminal) - 1 (the q1 drain, one stage) + 1 (s_read) + 4 (s_tail_empty: prologue, lookahead, tail, drain) + 3 (the
# tail's dS gates) = 68 -- the "ldg" form's counts (_EXPECTED_LDG, the merged rendering).  The DEFAULT form "tma_ring" (the day-4 decision) adds the
# stats ring's rows: the ring's expect_tx arm + the softmax's plain stage release (arrive 42), the softmax's stats_full wait + the loader's
# stats_empty wait (try_wait 70, their two `_wait` reconvergences: bar.warp.sync 69), the arm (expect_tx 12); every other row is the ldg form's
# (no proxy fence, no tcgen05 fence, no LDS.128 moved).  The entry is this body's kernel.
_EXPECTED = dict(
    EXPECTED_BASE,
    mma_cta_group_2=72,
    commit_multicast=13,
    mbarrier_arrive=42,
    try_wait=70,
    entry=0,
    entry_stail=1,
    lds_v4_b32=38,  # 32 stats / delta chunks (CFG.DIAG_STATS_LDS_PER_LANE) + 6 other 16-B shared loads
    expect_tx=12,  # the 8 tensor rows (do1 x 2) + ds_recv_full + the scheduler's local and remote arm + the stats ring's arm
    clock64=0,
    bar_warp_sync=69,
    tcgen05_fence_after=32,
    mapa=17,
)
_EXPECTED_LDG = dict(_EXPECTED, mbarrier_arrive=40, try_wait=68, expect_tx=11, bar_warp_sync=67)  # stats_form = "ldg": the merged rendering's census
# The TMEM map's census rows beyond the module census: the mbarrier inits (the two deleted rows = -2) and the `tcgen05.fence::before_thread_sync`
# sites (the softmax's fence before its s_read arrive goes with the arrive: -1; the epilogue's readout fence STAYS = the reuse edge's local half).
_TMEM_MAP_CENSUS = {"mbarrier_init": r"mbarrier\.init", "tcgen05_fence_before": r"tcgen05\.fence::before_thread_sync"}
# The SMEM map's census rows beyond the TMEM map's: the dQ readout's `tcgen05.wait::ld` sites, the relaxed cluster arrives (every `_arrive_leader` site
# = the LEADER rows' arrives: the mapa count is not a site count, two sites may share one hoisted mapa) and the dQ route rows (the extra `_dq_round` sites).
_SMEM_MAP_CENSUS = dict(_TMEM_MAP_CENSUS, tcgen05_wait_ld=r"tcgen05\.wait::ld", arrive_relaxed_cluster=r"mbarrier\.arrive\.relaxed\.cluster", **_ROUTE_CENSUS)


def _smem_map_census_delta(c, cfgmod, base):
    """What smem_map = rubin_b2 adds to the dq_576 x sm100 rendering's census and nothing else (the kernel header sections 5 / 8 / 11): the row
    dq_read_done -- its two `_arrive_leader` sites (the in-loop and the item-end readout: mbarrier.arrive +2, the relaxed cluster arrive form +2; mapa
    +1, the two sites sharing one hoisted mapa of the constant leader address) and its three `_wait_tcgen` sites (the two dQ issue sites + the P15 drain:
    try_wait / bar.warp.sync / fence::after +3); the lagging order's second readout site (the item end: tcgen05.wait::ld +1, fence::before +1) and its
    one more dq_done `_wait_tcgen` site (+1 on the three wait rows); the two extra `_dq_round` sites (five instead of three: dq_full arrive +2, their two
    dq_empty waits each, the slice STS and the proxy fence); the loader's two whole-ring drains at depth 2 (q1_empty, do1_empty: +1 wait site each); the
    grown slots' inits (+8) and the new row's (+1).  The counts come from the config and the site counts, never a literal."""
    b2 = cfgmod.CFG_RING_DQ576_B2
    rounds_extra = 2  # round B of dQ(t-2) in-loop + rounds A / B of dQ(hi-1) at the item end, against today's in-loop B and item-end B: 5 - 3 sites
    readout_sites_extra = 1  # the readout trio at the item end AND in-loop (under `qtile > lo`), against the single in-loop site
    dq_read_done_waits = 2 + 1  # `_dq_issue` inlined at the in-tile and the tail site + the P15 drain
    dq_read_done_arrives = 2
    drains_extra = (b2.Q1_STAGES - 1) + (b2.DO1_STAGES - 1)  # the loader's whole-ring drains (range_constexpr over the depth)
    extra_slots = sum(row[2] for row in cfgmod.BAR_LAYOUT_RING_DQ576_B2) - sum(row[2] for row in cfgmod.BAR_LAYOUT_RING_DQ576)  # 8 grown + 1 new
    waits = dq_read_done_waits + readout_sites_extra + rounds_extra * c.DQ_SLOTS + drains_extra
    assert (rounds_extra, readout_sites_extra, drains_extra, extra_slots, waits) == (2, 1, 2, 9, 10)
    return {
        "mbarrier_arrive": base["mbarrier_arrive"] + dq_read_done_arrives + rounds_extra,
        "arrive_relaxed_cluster": base["arrive_relaxed_cluster"] + dq_read_done_arrives,
        "mapa": base["mapa"] + 1,
        "try_wait": base["try_wait"] + waits,
        "bar_warp_sync": base["bar_warp_sync"] + waits,
        "tcgen05_fence_after": base["tcgen05_fence_after"] + dq_read_done_waits + readout_sites_extra,
        "tcgen05_fence_before": base["tcgen05_fence_before"] + readout_sites_extra,
        "tcgen05_wait_ld": base["tcgen05_wait_ld"] + readout_sites_extra,
        "mbarrier_init": base["mbarrier_init"] + extra_slots,
        "fence_proxy_async": base["fence_proxy_async"] + rounds_extra,
        "sts_dq_v2_b64": base["sts_dq_v2_b64"] + rounds_extra * (c.DQ_BOX // 4),  # DQ_BOX // 4 = 8 16-B vectors per slice row = 8 st.shared.v2.b64 per site
    }


def _tmem_map_census_delta(c, cfgmod, default):
    """What tmem_map = dq_576 removes from the s_tail rendering's census and nothing else (the kernel header section 11): the two deleted rows'
    arrive sites (one each: the softmax's s_read, the epilogue's s_tail_empty -- `_arrive_leader` = a mapa'd relaxed cluster arrive, so mapa -2 too),
    their inits (-2), the leader's five wait sites (each one try_wait + one bar.warp.sync + one tcgen05.fence::after: CFG.DIAG_S_TAIL_WAIT_SITES +
    CFG.DIAG_S_READ_WAIT_SITES), the softmax's one before-fence, and the alloc spelled `tcgen05.alloc.exclusive.cta_group::2` (the 576-column carve);
    the counts come from the config, never a literal."""
    deleted = len(cfgmod.DELETED_BARRIERS_DQ576)
    waits = c.DIAG_S_TAIL_WAIT_SITES + c.DIAG_S_READ_WAIT_SITES
    return {
        "mbarrier_arrive": default["mbarrier_arrive"] - deleted,
        "mapa": default["mapa"] - deleted,
        "mbarrier_init": default["mbarrier_init"] - deleted,  # one slot each
        "try_wait": default["try_wait"] - waits,
        "bar_warp_sync": default["bar_warp_sync"] - waits,
        "tcgen05_fence_after": default["tcgen05_fence_after"] - waits,
        "tcgen05_fence_before": default["tcgen05_fence_before"] - 1,
        "tmem_alloc_cta2": 0,
        "tmem_alloc_exclusive_cta2": 1,
    }


_PROBES = (
    "dq_reduce_add_off",
    "exchange_off",
    "stats_one_chunk",
    "sdq_sts_off",
    "s_tail_wait_off",
    "s_read_wait_off",
    "s_tail_and_s_read_wait_off",
)  # CFG.DIAG_2CTA_NAMES_ON_MAP (asserted in the config-truth row)
# The STATS FORM (lever S1, header section 11): "tma_ring" = THE default since the day-4 decision (the loader-fed two-stage SMEM stats ring: ONE 1 KiB
# bulk copy per q tile from the chain's padded pre-scaled stats region) | "ldg" = the previous default, selectable (the softmax's per-tile LDG fetch of
# LSE / delta under the `bar.sync 1,256` pair; the form the 16w_fa4 role shape resolves to when the field is unset).  The census
# rows the form moves (the S1 skeleton 14.6): the named-barrier pair, the two stats STS, the plain bulk copy (a NEW opcode class in this body: today's
# only cp.async.bulk is the DSMEM ship), the two ring rows' inits, the one global load the loader adds (rowoff[b]) against the six the softmax drops.
_STATS_FORMS = ("ldg", "tma_ring")
_STATS_CENSUS = {
    "bar_sync_1_256": r"bar\.sync\s+1,\s*256",
    "st_shared_b32": r"st\.shared\.b32",
    # Non-multicast copies use either destination spelling, depending on the DSL build.
    "bulk_g2s_plain": r"cp\.async\.bulk\.shared::(?:cta|cluster)\.global\.mbarrier",
    "mbarrier_init": r"mbarrier\.init",
    "ld_global": r"ld\.global",
    "fence_proxy_async": r"fence\.proxy\.async",
}


def _probe_census_delta(probe, c, default):
    """What each probe removes from the DEFAULT rendering's census and nothing else (the arrive count is identical by construction: a probe
    moves no barrier row; the issue / transfer / bytes probes keep every wait, the WAIT probes drop exactly their own `_wait_tcgen` sites =
    one try_wait + one bar.warp.sync + one tcgen05.fence::after each); the counts come from the config, never a literal."""

    def waits_off(sites):
        return {
            "try_wait": default["try_wait"] - sites,
            "bar_warp_sync": default["bar_warp_sync"] - sites,
            "tcgen05_fence_after": default["tcgen05_fence_after"] - sites,
        }

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
        "s_tail_wait_off": waits_off(c.DIAG_S_TAIL_WAIT_SITES),  # the leader's 4 s_tail_empty waits (prologue, lookahead, tail, drain)
        "s_read_wait_off": waits_off(c.DIAG_S_READ_WAIT_SITES),  # the leader's 1 s_read wait (slot 2, every tile)
        "s_tail_and_s_read_wait_off": waits_off(c.DIAG_S_TAIL_WAIT_SITES + c.DIAG_S_READ_WAIT_SITES),  # both
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
    """The pair body's section of the family module (its banner to the next): the source pins count sites in THIS body only."""
    path = _TREE / _BODY_REL
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    text = path.read_text()
    start, end = text.index(_BODY_SECTION[0]), text.index(_BODY_SECTION[1])
    assert start < end, _BODY_SECTION
    return text[start:end]


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
    # the TMEM MAP (arm R1, PART-KEYED): the dq_576 records = the same geometry with the image past dK of the 576-column carve, exclusive; SMEM untouched
    g, gr = cfgmod.CFG_DQ576, cfgmod.CFG_RING_DQ576
    assert (c.TMEM_MAP, g.TMEM_MAP, gr.TMEM_MAP, c.TMEM_MAPS, cfgmod.TMEM_MAP_CODES) == ("s_tail", "dq_576", "dq_576", _TMEM_MAPS, {"s_tail": 0, "dq_576": 1})
    assert (g.DQ_COL, g.DQ_COLS, g.TMEM_USED, g.TMEM_COLS, g.TMEM_IS_EXCLUSIVE, cfgmod.TMEM_COLS_SM107) == (512, 64, 576, 576, True, 576)
    assert g.DQ_COL == g.DK_COL + g.DK_COLS and g.DQ_COL + g.DQ_COLS == g.TMEM_COLS and (c.TMEM_IS_EXCLUSIVE, c.TMEM_COLS, c.TMEM_USED) == (False, 512, 512)
    assert (g.SMEM_BYTES, g.BARS, g.TMEM_HANDLE, g.TMEM_DEAD_COL, g.DQ_LANE_SPLIT) == (c.SMEM_BYTES, c.BARS, c.TMEM_HANDLE, 192, 64)
    assert cfgmod.TMEM_MAP_PARTS == {"s_tail": ((10, 0), (10, 3), (10, 7)), "dq_576": ((10, 7),)} and cfgmod.TMEM_MAP_DEFAULT_BY_PART == {
        (10, 0): "s_tail",
        (10, 3): "s_tail",
        (10, 7): "s_tail",  # dq_576 is selectable by name only (the probe verdict: a neutral structural map, kernel header section 13)
    }
    assert cfgmod.DELETED_BARRIERS_DQ576 == ("s_read", "s_tail_empty") and c.DIAG_2CTA_S_TAIL_PROBES == _PROBES[4:]
    dq_tables = (cfgmod.BAR_LAYOUT_DQ576, cfgmod.BAR_LAYOUT_RING_DQ576)
    assert [(len(t), sum(row[2] for row in t)) for t in dq_tables] == [(33, 38), (35, 42)]
    assert (cfgmod.BAR_LAYOUT_DQ576_END, cfgmod.BAR_LAYOUT_RING_DQ576_END) == (230928, 231984) and dq_tables[0][0][1] == c.BARS and dq_tables[1][0][1] == 231424
    assert [row[0] for row in dq_tables[0]] == [row[0] for row in cfgmod.BAR_LAYOUT if row[0] not in ("s_read", "s_tail_empty")]
    assert cfgmod.bar_kind_counts_for_roles("16w", "tma_ring", "dq_576") == {
        "tma_leader": 7,
        "mma": 9,
        "thread": 10,
        "leader": 4,
        "peer": 2,
        "tma_dsmem": 1,
        "tma": 2,
    }
    assert (cfgmod.DBG_CLK_DQ576.SMEM, cfgmod.DBG_CLK_DQ576.WORDS, cfgmod.DBG_CLK_RING_DQ576.SMEM, cfgmod.DBG_CLK_RING_DQ576.WORDS) == (230928, 104, 231984, 96)
    assert cfgmod.DBG_CLK_RING_DQ576.WORD == cfgmod.DBG_CLK.WORD and cfgmod.dbg_clk_dead_words("dq_576") == (
        "wait:leader:s_read",
        "blocked:leader:s_read",
        "wait:leader:s_tail_empty",
        "blocked:leader:s_tail_empty",
        "wait:leader:dq_read_done",  # the rubin_b2 row: dead on the sm100 SMEM map
        "blocked:leader:dq_read_done",
    )
    # the SMEM MAP (arm R2.c, PART-KEYED; kernel header sections 5 / 8 / 11): the B2 record's slab chain on the oversized carveout, every tcgen05 root
    # below the version-0 line, the DERIVED 36-row table (51 slots, the 32-B do1_full row, dq_read_done after dp_empty), the STAMPS record's spare
    b2 = cfgmod.CFG_RING_DQ576_B2
    assert (b2.SMEM_MAP, b2.Q1_STAGES, b2.DO1_STAGES, b2.DS_STAGES, b2.XCHG_STAGES, b2.TMEM_MAP, b2.TMEM_COLS) == ("rubin_b2", 2, 2, 2, 2, "dq_576", 576)
    assert (c.SMEM_MAP, c.SMEM_MAPS, cfgmod.SMEM_MAP_CODES, c.DO1_STAGES, c.DS_STAGES, c.XCHG_STAGES) == (
        "sm100",
        _SMEM_MAPS,
        {"sm100": 0, "rubin_b2": 1, "sm100_oversized": 2},
        1,
        1,
        1,
    )
    assert (b2.Q1, b2.Q2, b2.DO1, b2.DO2, b2.K1, b2.K2, b2.V, b2.DS, b2.XCHG, b2.DQ, b2.STATS, b2.SLABS_END) == (
        0,
        32768,
        49152,
        81920,
        98304,
        131072,
        163840,
        196608,
        262144,
        294912,
        311296,
        313344,
    )
    assert (b2.SMEM_BYTES, b2.BARS, b2.TMEM_HANDLE, b2.SCHED_PAYLOAD, b2.SMEM_CAP, c.SMEM_OPT_IN_CAP, c.SMEM_OVERSIZED_CAP) == (
        314368,
        313344,
        314288,
        314304,
        334848,
        232448,
        334848,
    )
    assert c.SMEM_OPT_IN_CAP < b2.SMEM_BYTES <= b2.SMEM_CAP and b2.SMEM_BYTES - gr.SMEM_BYTES == 81920  # the four second stages: +16 +16 +32 +16 KiB
    assert (
        max(b2.desc_roots()) == 229376 < b2.TCGEN05_V0_ADDR_LIMIT == 262144 == b2.XCHG and b2.desc_roots()[-1] + b2.DS_BYTES == 262144
    )  # the last root ends AT the line
    assert [b2.ds_stage(s) for s in range(2)] == [196608, 229376] and [b2.xchg_stage(s) for s in range(2)] == [262144, 278528] and b2.dq_slot(1) == 303104
    assert {b2.dk_store(0, 0), b2.dk_store(0, 1)} == {b2.ds_own_stage(0, 0), b2.xchg_stage(0)}  # the dK staging alias over the STAGE-0 bytes (section 8)
    t = cfgmod.BAR_LAYOUT_RING_DQ576_B2
    assert (len(t), sum(r[2] for r in t), cfgmod.BAR_LAYOUT_RING_DQ576_B2_END) == (36, 51, 313936) and t[0][1] == b2.BARS
    names_b2 = [r[0] for r in t]
    assert names_b2.index("dq_read_done") == names_b2.index("dp_empty") + 1 and cfgmod.ADDED_BARRIERS_B2 == ("dq_read_done",)
    assert {r[0]: r[2] for r in t} == dict(
        {r[0]: r[2] for r in cfgmod.BAR_LAYOUT_RING_DQ576},
        q1_full=2,
        q1_empty=2,
        do1_full=4,
        do1_empty=2,
        ds_smem=2,
        ds_recv_full=2,
        ds_pair_ready=2,
        dq_read_done=1,
    )
    assert cfgmod.DQ_READ_DONE_ROW == ("dq_read_done", 1, c.PAIR_EPILOGUE_LANES, "leader", "LEADER") and c.PAIR_EPILOGUE_LANES == 256
    assert cfgmod.bar_kind_counts_for_roles("16w", "tma_ring", "dq_576", "rubin_b2") == {
        "tma_leader": 7,
        "mma": 9,
        "thread": 10,
        "leader": 5,
        "peer": 2,
        "tma_dsmem": 1,
        "tma": 2,
    }
    assert {r[0]: r[1] for r in t}["do1_empty"] - {r[0]: r[1] for r in t}["do1_full"] == 32  # the cumulative walk: a 4-slot row takes 32 B
    assert (
        cfgmod.layout_for_geometry(c) == cfgmod.BAR_LAYOUT
        and cfgmod.layout_for_geometry(gr) == cfgmod.BAR_LAYOUT_RING_DQ576
        and cfgmod.layout_for_geometry(b2) == t
    )
    assert (cfgmod.DBG_CLK_RING_DQ576_B2.SMEM, cfgmod.DBG_CLK_RING_DQ576_B2.WORDS) == (313936, 88) and cfgmod.DBG_CLK_RING_DQ576_B2.WORD == cfgmod.DBG_CLK.WORD
    assert cfgmod.dbg_clk_dead_words("dq_576", "rubin_b2") == cfgmod.dbg_clk_dead_words("dq_576")[:4] and cfgmod.dbg_clk_dead_words("s_tail") == (
        "wait:leader:dq_read_done",
        "blocked:leader:dq_read_done",
    )
    assert "dq_read_done" not in {r[0] for r in cfgmod.BAR_LAYOUT} | {r[0] for r in cfgmod.BAR_LAYOUT_RING_DQ576} | {r[0] for r in cfgmod.BAR_LAYOUT_XFA4}
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
        18,
        86,
        128,
        (100, 400, 1600),
    )  # the record = the 384-B spare left by the fp32_reduce row (derived); 82 ldg words + the leader's dq_read_done pair (rubin_b2) + the stats ring's 2
    assert d.SMEM == cfgmod.BAR_LAYOUT_END and d.SMEM + d.BYTES <= c.TMEM_HANDLE and c.DEBUG_CLK_DEFAULT is False
    # --- lever S1 (stats_form; header sections 5 / 8 / 11): the two geometries, the ring, the re-based ring barrier table, the record per form ---
    from cudnn.sdpa.bwd import d128_grouped_2cta_contract as contract

    r = cfgmod.CFG_RING
    assert (gr.DQ_COL, gr.TMEM_COLS, gr.SMEM_BYTES, gr.BARS, gr.TMEM_HANDLE, gr.STATS_STAGES) == (512, 576, r.SMEM_BYTES, r.BARS, r.TMEM_HANDLE, 2)
    assert cfgmod.cfg_for_part((10, 0)) is r is cfgmod.cfg_for_part((10, 3)) is cfgmod.cfg_for_part((10, 7)) is cfgmod.cfg_for_part((10, 7), tmem_map="s_tail")
    assert cfgmod.cfg_for_part((10, 7), tmem_map="dq_576") is gr and cfgmod.cfg_for_part((10, 7), "ldg", tmem_map="dq_576") is g  # NAMED on the 576-column part
    assert cfgmod.cfg_for_part((10, 7), "ldg") is c is cfgmod.cfg_for_part((10, 7), roles="16w_fa4")  # unset: s_tail everywhere (16w_fa4: ldg too)
    assert (c.STATS_FORMS, c.STATS_FORM_DEFAULT, cfgmod.STATS_FORM_CODES) == (("ldg", "tma_ring"), "tma_ring", {"ldg": 0, "tma_ring": 1})
    assert c.STATS_FORM_FA4 == "ldg" and c.STATS_FORM_DEFAULT_BY_ROLES == {"16w": "tma_ring", "16w_fa4": "ldg"}  # the unset field per role shape
    assert (c.STATS_STAGES, r.STATS_STAGES, c.STATS_STAGE_BYTES, c.STATS_TX, c.STATS_DELTA_OFF, c.STATS_RING) == (1, 2, 1024, 1024, 512, 229376)
    assert (c.SLABS_END, c.SMEM_BYTES, c.BARS, c.TMEM_HANDLE, c.SCHED_PAYLOAD) == (230400, 231424, 230400, 231344, 231360)  # ldg = the shipped scaffold
    assert (r.SLABS_END, r.SMEM_BYTES, r.BARS, r.TMEM_HANDLE, r.SCHED_PAYLOAD) == (231424, 232448, 231424, 232368, 232384)  # tma_ring: +1024, == the cap
    assert r.SMEM_BYTES == c.SMEM_OPT_IN_CAP == 232448 and [r.stats_stage(st) for st in range(2)] == [229376, 230400] == [r.stats_lse(st) for st in range(2)]
    assert [r.stats_delta(st) for st in range(2)] == [229888, 230912] and (c.stats_lse(0), c.stats_delta(0)) == (c.STATS, c.DELTA)
    assert (c.STATS_KINDS, c.STATS_RECORD_BYTES, c.STATS_PAD_GRAN) == (2, 1024, 128) and c.STATS_RECORD_BYTES == c.STATS_STAGE_BYTES
    assert cfgmod.cfg_for_stats_form(None) is r is cfgmod.cfg_for_stats_form("tma_ring") and cfgmod.cfg_for_stats_form("ldg") is c is cfgmod.cfg_for_stats_form(
        None, "16w_fa4"
    )
    ring = cfgmod.BAR_LAYOUT_RING
    assert len(ring) == 37 and sum(row[2] for row in ring) == 44 and cfgmod.BAR_LAYOUT_RING_END == 232016 and ring[0][1] == r.BARS == 231424
    assert ring[:35] == tuple((n, off + 1024, sl, cnt, k, sc) for n, off, sl, cnt, k, sc in cfgmod.BAR_LAYOUT)  # the audited rows re-based, nothing else
    assert ring[35] == ("stats_full", 231984, 2, c.ONE_LANE, "tma", "LOCAL") and ring[36] == ("stats_empty", 232000, 2, c.SOFTMAX_LANES, "thread", "LOCAL")
    assert cfgmod.BAR_KIND_COUNTS_RING == {"tma_leader": 7, "mma": 9, "thread": 10, "leader": 6, "peer": 2, "tma_dsmem": 1, "tma": 2}
    assert (
        cfgmod.bar_layout_for_roles("16w", "tma_ring") is ring is cfgmod.bar_layout_for_roles("16w")
        and cfgmod.bar_layout_for_roles("16w", "ldg") is cfgmod.BAR_LAYOUT
    )
    assert cfgmod.bar_layout_end_for_roles("16w", "tma_ring") == 232016 and cfgmod.bar_layout_end_for_roles("16w_fa4", "tma_ring") == 231984
    dr = cfgmod.DBG_CLK_RING
    assert (dr.SMEM, dr.WORDS, dr.BYTES) == (232016, 88, 352) and dr.SMEM + dr.BYTES == r.TMEM_HANDLE and dr.WORD == d.WORD  # derived: the re-based spare
    assert [d.WORD[f"wait:{role}:{b}"] for role, b in cfgmod.DBG_CLK_RING_WAITS] == [84, 85] and cfgmod.dbg_clk_for_stats_form("tma_ring") is dr
    assert cfgmod.dbg_clk_bytes(1, 1, 256, "ldg") == 2 * 384 and cfgmod.dbg_clk_bytes(1, 1, 256) == cfgmod.dbg_clk_bytes(1, 1, 256, "tma_ring") == 2 * 352
    # the padded pre-scaled stats region (the contract): one (head, q tile) record = one ring stage image; PAD_TOTAL bounds every sequence's block
    assert contract.ABI_VERSION == 2 and contract.STATS_KINDS == 2
    assert contract.stats2_pad_total(4, 16384) == 16384 + 512 and contract.stats2_shape_strides(32, 16896) == ((32, 132, 2, 128), (132 * 256, 256, 128, 1))
    for b, lens in ((3, (129, 0, 127)), (5, (300, 128, 1000, 0, 257)), (4, (1, 1, 1, 1))):
        rounded_q = (sum(lens) + 127) // 128 * 128
        assert sum((n + 127) // 128 * 128 for n in lens) <= contract.stats2_pad_total(b, rounded_q), (b, lens)
    ldg_off, ring_off = contract.thd_workspace_offsets(2, 8, 1000, 1000), contract.thd_workspace_offsets(2, 8, 1000, 1000, True)
    assert ldg_off[-1] is None and ring_off[-1] is not None and ldg_off[:3] == ring_off[:3] and ring_off[3] > ldg_off[3]  # the stats region replaces delta
    assert contract.thd_workspace_layout(2, 8, 1000, 1000)[0][1][1] == (1, 8, 1024) and contract.thd_workspace_layout(2, 8, 1000, 1000, True)[0][1][1] == (
        8,
        10,
        2,
        128,
    )
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
        and cfgmod.dbg_clk_bytes(2, 8, 512) == 64 * cfgmod.DBG_CLK_RING.BYTES
        and cfgmod.dbg_clk_bytes(2, 8, 512, "ldg") == 64 * cfgmod.DBG_CLK.BYTES
        and cfgmod.dbg_clk_region_offset(1000) == 1024
    )
    cfgmod.validate_config()
    first.validate_config()


def test_2cta_stail_fields_are_typed():
    """The typed fields decline BY NAME: ``diag_2cta`` resolves the seven probes of the code space and nothing else (two probes of an earlier pair
    body are outside the domain); ``debug_clk`` is a bool (False by default); the adapter readers prefix the engine name; the record defaults are
    None / False; the pruned perf knobs of the measurement measurement series are not fields of the adapter (one body, one order, one load form)."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta as base
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import debug_clk, diag_2cta

    assert base.resolve_diag_2cta(None) is None and [base.resolve_diag_2cta(p) for p in _PROBES] == list(_PROBES)
    assert [base.diag_2cta_code(p) for p in _PROBES] == list(range(1, len(_PROBES) + 1)) and base.diag_2cta_code(None) == 0
    assert _PROBES[:4] == ("dq_reduce_add_off", "exchange_off", "stats_one_chunk", "sdq_sts_off"), "the three wait probes joined AFTER the first four"
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
    assert SdpaBwdDslSm100D128.tmem_map is None  # the TMEM map: unset = the PART's default (a typed field since arm R1; the typed row below)
    for pruned in ("pair_mma_order", "dq_route_pair", "tma_load_form", "store_wait_form", "reg_split", "datapath_2cta"):
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


def test_2cta_stail_stats_form_is_typed():
    """``stats_form`` is a typed field (lever S1): ``None`` = the role shape's form -- "tma_ring" (THE default since the day-4 decision) under the
    default roles, "ldg" under "16w_fa4" (the only form sited there); "ldg" is the other value (the previous default); anything else declines BY
    NAME; the kernel's constants are the config's code space (config = truth); the adapter reader prefixes the engine name; the record default is
    None; the combination rules: "tma_ring" NAMED x ``warp_roles = "16w_fa4"`` declines BY NAME at compile_host (the first build sites the ring in
    the 16w bundle; the unset field is served there as ldg), "tma_ring" x ``debug_clk`` is SERVED (the re-based record carries the two ring wait
    words), "tma_ring" x every probe / mask form passes."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_host import compile_host_grouped_2cta as compile_host
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import stats_form

    c = cfgmod.CFG
    assert c.STATS_FORMS == _STATS_FORMS and c.STATS_FORM_DEFAULT == "tma_ring" and tuple(cfgmod.STATS_FORM_CODES) == _STATS_FORMS
    assert cfgmod.resolve_stats_form(None) == "tma_ring" == cfgmod.resolve_stats_form(None, "16w") and cfgmod.resolve_stats_form(None, "16w_fa4") == "ldg"
    for form, code in (("ldg", 0), ("tma_ring", 1)):
        assert cfgmod.resolve_stats_form(form) == form and cfgmod.stats_form_code(form) == code
    assert cfgmod.stats_form_code(None) == 1 and cfgmod.stats_form_code(None, "16w_fa4") == 0
    with pytest.raises(ValueError, match="warp_roles must be one of"):
        cfgmod.resolve_stats_form(None, "bogus")
    for bad_form in ("bogus", "", "TMA_RING", "tma", 1):
        with pytest.raises(ValueError, match="stats_form must be one of"):
            cfgmod.resolve_stats_form(bad_form)
        with pytest.raises(ValueError, match="stats_form must be one of"):
            cfgmod.stats_form_code(bad_form)
    assert stats_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", stats_form="ldg")) == "ldg" and stats_form(SimpleNamespace(_NAME="x")) == "tma_ring"
    assert stats_form(SimpleNamespace(_NAME="x", warp_roles="16w_fa4")) == "ldg" and stats_form(SimpleNamespace(_NAME="x", warp_roles="16w")) == "tma_ring"
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: stats_form must be one of"):
        stats_form(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", stats_form="ring"))
    assert SdpaBwdDslSm100D128.stats_form is None  # the record default: unset = the role shape's form (tma_ring under 16w)
    body = _body_module()
    assert (
        (body.STATS_FORM_LDG, body.STATS_FORM_TMA_RING) == (0, 1)
        and body.STATS_RECORD_ELEMS == 256
        and tuple(body.BarsRing._fields[-2:]) == ("stats_full", "stats_empty")
    )
    assert (
        cfgmod.validate_fields_for_roles("16w", None, stats_form="tma_ring") == "16w"
        and cfgmod.validate_fields_for_roles("16w", None, debug_clk=True, stats_form="tma_ring") == "16w"
    )
    for probe in _PROBES:
        assert cfgmod.validate_fields_for_roles("16w", probe, stats_form="tma_ring") == "16w"
    assert cfgmod.validate_fields_for_roles("16w_fa4") == "16w_fa4" and cfgmod.validate_fields_for_roles("16w_fa4", None, stats_form="ldg") == "16w_fa4"
    with pytest.raises(ValueError, match="stats_form 'tma_ring' is not built under warp_roles '16w_fa4'"):
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, warp_roles="16w_fa4", stats_form="tma_ring")
    with pytest.raises(ValueError, match="stats_form must be one of"):
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, stats_form="ring")


def test_2cta_stail_tmem_map_is_typed_and_part_keyed(monkeypatch):
    """``tmem_map`` is a typed, PART-KEYED field (the kernel header sections 9 / 11 / 13): ``None`` = the part's default = "s_tail" on every admitted
    part (the only map the 512-column carve serves; on cc 10.7 "dq_576" is SELECTABLE BY NAME and not the default: its ceiling probe read inside noise,
    a neutral structural map), "s_tail" under "16w_fa4" too; a name outside the domain declines BY NAME (ValueError); a known map NAMED on a part
    that does not serve it is the engine's typed decline (the config's ``TmemMapNotServed`` -> the adapter reader's NotImplementedError); an
    unadmitted part has no default; the cross-field rules decline by name at the config bundle and at ``compile_host`` before any trace ("dq_576" x
    "16w_fa4", the three S-tail WAIT probes x "dq_576"); the kernel's constants are the config's code space; the record default is None."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd import prepared_sm100_d128_grouped_2cta as prepared
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_host import compile_host_grouped_2cta as compile_host

    c = cfgmod.CFG
    assert c.TMEM_MAPS == _TMEM_MAPS and tuple(cfgmod.TMEM_MAP_CODES) == _TMEM_MAPS and c.TMEM_MAP == "s_tail" and cfgmod.CFG_DQ576.TMEM_MAP == "dq_576"
    for cc in ((10, 0), (10, 3)):
        assert cfgmod.resolve_tmem_map(None, cc) == "s_tail" == cfgmod.resolve_tmem_map("s_tail", cc) and cfgmod.tmem_map_code(None, cc) == 0
        with pytest.raises(cfgmod.TmemMapNotServed, match="tmem_map 'dq_576' is not served on cc"):
            cfgmod.resolve_tmem_map("dq_576", cc)
    assert cfgmod.resolve_tmem_map(None, (10, 7)) == "s_tail" and cfgmod.tmem_map_code(None, (10, 7)) == 0  # the part's default: s_tail (no perf claim)
    assert (
        cfgmod.resolve_tmem_map("dq_576", (10, 7)) == "dq_576" == cfgmod.resolve_tmem_map("dq_576", (10, 7), "16w")
        and cfgmod.tmem_map_code("dq_576", (10, 7)) == 1
    )
    assert (
        cfgmod.resolve_tmem_map("s_tail", (10, 7)) == "s_tail" == cfgmod.resolve_tmem_map(None, (10, 7), "16w_fa4")
        and cfgmod.tmem_map_code("s_tail", (10, 7)) == 0
    )
    for bad_name in ("bogus", "", "DQ_576", 1):
        with pytest.raises(ValueError, match="tmem_map must be one of") as info:
            cfgmod.resolve_tmem_map(bad_name, (10, 7))
        assert not isinstance(info.value, cfgmod.TmemMapNotServed)  # a name outside the domain is a bug, never a decline
    for bad_cc in (None, (9, 0), (12, 0), (10, 9)):
        with pytest.raises(ValueError, match="no TMEM map is defined for cc"):
            cfgmod.resolve_tmem_map(None, bad_cc)
    # the adapter reader: the part from Q's device (patched here: the host has none), the prefix, the typed decline
    monkeypatch.setattr(prepared, "_device_cc", lambda api: (10, 7))
    assert prepared.tmem_map(SimpleNamespace(_NAME="x")) == "s_tail" and prepared.tmem_map(SimpleNamespace(_NAME="x", tmem_map="dq_576")) == "dq_576"
    assert prepared.tmem_map(SimpleNamespace(_NAME="x", tmem_map="s_tail")) == "s_tail"
    assert prepared.tmem_map(SimpleNamespace(_NAME="x", warp_roles="16w_fa4")) == "s_tail"
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: tmem_map must be one of"):
        prepared.tmem_map(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="dq576"))
    monkeypatch.setattr(prepared, "_device_cc", lambda api: (10, 0))
    assert prepared.tmem_map(SimpleNamespace(_NAME="x")) == "s_tail"
    with pytest.raises(NotImplementedError, match="sdpa_bwd_sm100_d128: 2-CTA datapath tmem_map 'dq_576' is not served on cc \\(10, 0\\)"):
        prepared.tmem_map(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", tmem_map="dq_576"))
    assert SdpaBwdDslSm100D128.tmem_map is None  # the record default: the part's map
    # the cross-field rules at the config bundle and at compile_host (sm = the part): before any trace
    assert (
        cfgmod.validate_fields_for_roles("16w", None, tmem_map="dq_576") == "16w" == cfgmod.validate_fields_for_roles("16w", "exchange_off", tmem_map="dq_576")
    )
    with pytest.raises(ValueError, match="tmem_map 'dq_576' is not built under warp_roles '16w_fa4'"):
        cfgmod.validate_fields_for_roles("16w_fa4", None, tmem_map="dq_576")
    for probe in c.DIAG_2CTA_S_TAIL_PROBES:
        with pytest.raises(ValueError, match=f"diag_2cta '{probe}' is not defined under tmem_map 'dq_576'"):
            cfgmod.validate_fields_for_roles("16w", probe, tmem_map="dq_576")
        assert cfgmod.validate_fields_for_roles("16w", probe, tmem_map="s_tail") == "16w"
    args = ((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True)
    with pytest.raises(ValueError, match="tmem_map 'dq_576' is not served on cc \\(10, 0\\)"):
        compile_host(*args, 100, tmem_map="dq_576")
    with pytest.raises(ValueError, match="tmem_map 'dq_576' is not served on cc \\(10, 3\\)"):
        compile_host(*args, 103, tmem_map="dq_576")
    with pytest.raises(ValueError, match="tmem_map 'dq_576' is not built under warp_roles '16w_fa4'"):
        compile_host(*args, 107, tmem_map="dq_576", warp_roles="16w_fa4")
    for probe in c.DIAG_2CTA_S_TAIL_PROBES:
        with pytest.raises(ValueError, match=f"diag_2cta '{probe}' is not defined under tmem_map 'dq_576'"):
            compile_host(*args, 107, tmem_map="dq_576", diag_2cta=probe)  # NAMED: the cc 10.7 default map is s_tail, where the probes are served
    for probe in c.DIAG_2CTA_S_TAIL_PROBES:
        assert cfgmod.validate_fields_for_roles("16w", probe, tmem_map=cfgmod.resolve_tmem_map(None, (10, 7))) == "16w"  # the default serves them
    with pytest.raises(ValueError, match="tmem_map must be one of"):
        compile_host(*args, 107, tmem_map="bogus")
    body = _body_module()
    assert (body.TMEM_MAP_S_TAIL, body.TMEM_MAP_DQ576) == (0, 1) and body.DQ_COL_BY_MAP == {0: c.DQ_COL, 1: cfgmod.CFG_DQ576.DQ_COL} == {0: 64, 1: 512}
    assert body._dq_col(0) == 64 and body._dq_col(1) == 512 and body._geometry(True, True) is cfgmod.CFG_RING_DQ576 and body._geometry(False, False) is c
    assert tuple(body.BarsDQ576._fields) == tuple(r[0] for r in cfgmod.BAR_LAYOUT_DQ576) and tuple(body.BarsRingDQ576._fields) == tuple(
        r[0] for r in cfgmod.BAR_LAYOUT_RING_DQ576
    )
    assert "s_read" not in body.BarsDQ576._fields and "s_tail_empty" not in body.BarsRingDQ576._fields and body.DVK_STAGE_EMPTY_ROW == 34
    assert body._dvk_stage_empty_row(body.BarsDQ576) == body._dvk_stage_empty_row(body.BarsRingDQ576) == 32 and body._dvk_stage_empty_row(body.BarsXFA4) == -1


def test_2cta_stail_smem_map_is_typed_and_part_keyed(monkeypatch):
    """``smem_map`` is a typed, PART-KEYED field (the kernel header sections 5 / 8 / 11; arm R2.c): ``None`` = the part's default = "sm100" on every
    admitted part (the opt-in slab chain); "rubin_b2" (the 327 KiB oversized carveout) is SELECTABLE BY NAME on cc 10.7 together with ``tmem_map =
    "dq_576"`` under the default role shape and stats form and not a default anywhere (its A/B/A row flips the defaults); a name outside the domain
    declines BY NAME (ValueError); a known map NAMED on a part that does not serve it is the engine's typed decline (the config's ``SmemMapNotServed`` ->
    the adapter reader's NotImplementedError); an unadmitted part has no default; the cross-field rules decline by name at the config bundle and at
    ``compile_host`` before any trace ("rubin_b2" x "s_tail", x "16w_fa4", x "ldg"); the kernel's constants, bundles and trace-time declines are the
    config's code space; the record default is None."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd import prepared_sm100_d128_grouped_2cta as prepared
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_host import compile_host_grouped_2cta as compile_host

    c = cfgmod.CFG
    assert (
        c.SMEM_MAPS == _SMEM_MAPS and tuple(cfgmod.SMEM_MAP_CODES) == _SMEM_MAPS and c.SMEM_MAP == "sm100" and cfgmod.CFG_RING_DQ576_B2.SMEM_MAP == "rubin_b2"
    )
    assert cfgmod.SMEM_MAP_PARTS == {"sm100": ((10, 0), (10, 3), (10, 7)), "rubin_b2": ((10, 7),), "sm100_oversized": ((10, 7),)}
    assert cfgmod.SMEM_MAP_DIAGNOSTIC == ("sm100_oversized",) and cfgmod.smem_map_code("sm100_oversized", (10, 7)) == 2  # the DIAGNOSTIC map, by name
    assert (cfgmod.CFG_RING_DQ576_OVS.SMEM_BYTES, cfgmod.CFG_RING_OVS.SMEM_BYTES, cfgmod.SMEM_OVERSIZED_PROBE_BYTES) == (262144, 262144, 262144)
    for cc in ((10, 0), (10, 3)):
        with pytest.raises(cfgmod.SmemMapNotServed, match="smem_map 'sm100_oversized' is not served on cc"):
            cfgmod.resolve_smem_map("sm100_oversized", cc)
    assert cfgmod.SMEM_MAP_DEFAULT_BY_PART == {(10, 0): "sm100", (10, 3): "sm100", (10, 7): "sm100"}  # rubin_b2 by name only (its A/B/A row pending)
    for cc in ((10, 0), (10, 3)):
        assert cfgmod.resolve_smem_map(None, cc) == "sm100" == cfgmod.resolve_smem_map("sm100", cc) and cfgmod.smem_map_code(None, cc) == 0
        with pytest.raises(cfgmod.SmemMapNotServed, match="smem_map 'rubin_b2' is not served on cc"):
            cfgmod.resolve_smem_map("rubin_b2", cc)
    assert cfgmod.resolve_smem_map(None, (10, 7)) == "sm100" and cfgmod.smem_map_code(None, (10, 7)) == 0  # the part's default: sm100 (no default flip)
    assert (
        cfgmod.resolve_smem_map("rubin_b2", (10, 7)) == "rubin_b2" == cfgmod.resolve_smem_map("rubin_b2", (10, 7), "16w", "tma_ring", "dq_576")
        and cfgmod.smem_map_code("rubin_b2", (10, 7)) == 1
    )
    for bad_name in ("bogus", "", "RUBIN_B2", "b2", 1):
        with pytest.raises(ValueError, match="smem_map must be one of") as info:
            cfgmod.resolve_smem_map(bad_name, (10, 7))
        assert not isinstance(info.value, cfgmod.SmemMapNotServed)  # a name outside the domain is a bug, never a decline
    for bad_cc in (None, (9, 0), (12, 0), (10, 9)):
        with pytest.raises(ValueError, match="no SMEM map is defined for cc"):
            cfgmod.resolve_smem_map(None, bad_cc)
    # the adapter reader: the part from Q's device (patched here: the host has none), the prefix, the typed decline
    monkeypatch.setattr(prepared, "_device_cc", lambda api: (10, 7))
    assert prepared.smem_map(SimpleNamespace(_NAME="x")) == "sm100" and prepared.smem_map(SimpleNamespace(_NAME="x", **_B2_FIELDS)) == "rubin_b2"
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: smem_map must be one of"):
        prepared.smem_map(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", smem_map="b2"))
    monkeypatch.setattr(prepared, "_device_cc", lambda api: (10, 0))
    assert prepared.smem_map(SimpleNamespace(_NAME="x")) == "sm100"
    with pytest.raises(NotImplementedError, match="sdpa_bwd_sm100_d128: 2-CTA datapath smem_map 'rubin_b2' is not served on cc \\(10, 0\\)"):
        prepared.smem_map(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", smem_map="rubin_b2"))
    assert SdpaBwdDslSm100D128.smem_map is None  # the record default: the part's map
    # the cross-field rules at the config bundle and at compile_host (sm = the part): before any trace
    assert (
        cfgmod.validate_fields_for_roles("16w", None, tmem_map="dq_576", smem_map="rubin_b2")
        == "16w"
        == cfgmod.validate_fields_for_roles("16w", None, smem_map="sm100")
    )
    with pytest.raises(ValueError, match="smem_map 'rubin_b2' is not built under tmem_map 's_tail'"):
        cfgmod.validate_fields_for_roles("16w", None, smem_map="rubin_b2")
    with pytest.raises(ValueError, match="smem_map 'rubin_b2' is not built under warp_roles '16w_fa4'"):
        cfgmod.validate_fields_for_roles("16w_fa4", None, tmem_map="dq_576", smem_map="rubin_b2")
    with pytest.raises(ValueError, match="smem_map 'rubin_b2' is not built under stats_form 'ldg'"):
        cfgmod.validate_fields_for_roles("16w", None, stats_form="ldg", tmem_map="dq_576", smem_map="rubin_b2")
    args = ((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True)
    with pytest.raises(ValueError, match="smem_map 'rubin_b2' is not served on cc \\(10, 0\\)"):
        compile_host(*args, 100, smem_map="rubin_b2")
    with pytest.raises(ValueError, match="smem_map 'rubin_b2' is not served on cc \\(10, 3\\)"):
        compile_host(*args, 103, smem_map="rubin_b2")
    with pytest.raises(ValueError, match="smem_map 'rubin_b2' is not built under tmem_map 's_tail'"):
        compile_host(*args, 107, smem_map="rubin_b2")  # the cc 10.7 default TMEM map is s_tail: both maps must be NAMED
    with pytest.raises(ValueError, match="smem_map 'rubin_b2' is not built under warp_roles '16w_fa4'"):
        compile_host(*args, 107, warp_roles="16w_fa4", **_B2_FIELDS)
    with pytest.raises(ValueError, match="smem_map 'rubin_b2' is not built under stats_form 'ldg'"):
        compile_host(*args, 107, stats_form="ldg", **_B2_FIELDS)
    with pytest.raises(ValueError, match="smem_map must be one of"):
        compile_host(*args, 107, smem_map="bogus")
    body = _body_module()
    assert (body.SMEM_MAP_SM100, body.SMEM_MAP_RUBIN_B2) == (0, 1) and body._geometry(True, True, True) is cfgmod.CFG_RING_DQ576_B2
    assert body._geometry(True, True) is cfgmod.CFG_RING_DQ576 and body._geometry(False, False) is c
    assert tuple(body.BarsRingDQ576B2._fields) == tuple(r[0] for r in cfgmod.BAR_LAYOUT_RING_DQ576_B2) and "dq_read_done" in body.BarsRingDQ576B2._fields
    assert "dq_read_done" not in body.BarsRingDQ576._fields and "dq_read_done" not in body.Bars._fields and "dq_read_done" not in body.BarsXFA4._fields
    assert body._dvk_stage_empty_row(body.BarsRingDQ576B2) == body._dvk_stage_empty_row(body.BarsRingDQ576) + 1
    for bad_geo in ((False, True, True), (True, False, True)):  # ldg x rubin_b2, s_tail x rubin_b2: no geometry record
        with pytest.raises(ValueError, match="no geometry for"):
            body._geometry(*bad_geo)
    kernel = _function_code(_body_source(), "bprop_d128_accum_grouped_2cta_stail_kernel", "main_thd_2cta_stail_host")
    for guard in ("b2 and not dq576", "b2 and fa4_roles", "b2 and not tma_ring"):  # the trace-time declines by name
        assert any(f"if cutlass.const_expr({guard}):" in ln for ln in kernel), guard


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


def _render_kernels(tmp_path, form_kw, extra_census=None):
    """``_render`` plus the per-kernel md5s of the dumped module -> (PTX md5, census, setmaxnreg pairs, {kernel: md5})."""
    return render_2cta_kernels(tmp_path, form_kw, dict(_CENSUS, **(extra_census or {})))


def _record_kernels(path, token, arch="sm_100a"):
    """(dsl line, {kernel: md5}) of a per-kernel record's ``default`` variant for ``arch`` (frost_test_utils.read_kernel_record): the
    installed DSL build's record where one CARRIES the token (the release record, then the dev-build record beside it); an unrecorded build,
    or a token the matching build's record lacks, still reaches the existing skip guard."""
    import cudnn.frost.buffers as buffers

    _installed, version = buffers.cutedsl_state()
    have = " ".join(version) if version else None
    paths = (path, _RECORD_DEV) if path == _RECORD else (path,)
    records = [read_kernel_record(candidate, token, arch) for candidate in paths]
    dsl, want = next((record for record in records if record[0] == have and record[1].get("default")), records[0])
    return dsl, want.get("default", {})


def _assert_record(kernels, path, token, what, arch="sm_100a"):
    """The skip-then-assert pin: a SKIP on another DSL build, else every kernel the chain owns renders the record's md5 and every rendered kernel
    outside the shared set is recorded (the fold is excluded by name) -- never a widened or a conditional pin."""
    dsl, want = _record_kernels(path, token, arch)
    assert dsl and want, (dsl, want, path)
    _skip_unless_record_dsl(dsl)
    assert_kernel_record(kernels, want, token=token, variant="default", excluded=D128_THD_SHARED_KERNELS, what=what)


def test_2cta_stail_record_selector_falls_back_to_the_release_record_for_tokens_the_dev_record_lacks(monkeypatch):
    """RED-first (PR 2086 review): on an install whose DSL build string equals the dev-build record's, the selector must take the dev record
    only for the tokens that record carries (the default); a token recorded in the release file alone (the dense twin) falls back to the
    release record, so the skip guard -- never an empty record -- decides; no record file is read as the "installed" one by position."""
    import cudnn.frost.buffers as buffers

    dev_dsl = read_kernel_record(_RECORD_DEV, _RECORD_TOKEN)[0]
    assert dev_dsl and dev_dsl != read_kernel_record(_RECORD, _RECORD_TOKEN)[0], dev_dsl
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, tuple(dev_dsl.split(" ", 1))))
    dsl_default, want_default = _record_kernels(_RECORD, _RECORD_TOKEN)
    assert dsl_default == dev_dsl and want_default, (dsl_default, want_default)
    dsl_dense, want_dense = _record_kernels(_RECORD, _RECORD_TOKEN_DENSE)
    assert want_dense and dsl_dense != dev_dsl, (dsl_dense, want_dense)  # the release record: the guard skips it by name on this build


def _skip_unless_record_dsl(dsl):
    """A record's PTX md5 holds for the DSL build that rendered it (its ``dsl=`` line): another installed build is a SKIP before the md5
    assertion -- never a widened or a conditional pin (PR 1943 review)."""
    import cudnn.frost.buffers as buffers

    _installed, version = buffers.cutedsl_state()
    have = " ".join(version) if version else None
    if have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: PTX text differs by DSL build, re-render the record")


def test_2cta_stail_host_renders_for_sm_100a_and_matches_its_record(tmp_path):
    """Host trace-compile of the chain with every field unset (no device): the body's PTX census is the header's (sections 5 / 11), no local
    memory, no nanosleep, the 144 / 144 / 80 register split; the md5 matches the committed record at this geometry (skips, never widens, on
    another DSL build)."""
    md5, census, regs, kernels = _render_kernels(tmp_path, {})
    print(f"\n2-CTA s_tail sm_100a: PTX md5 {md5} census {census} setmaxnreg {regs} kernels {kernels}")
    assert census == _EXPECTED, census
    assert sorted(regs) == _setmaxnreg_of(), regs
    assert set(kernels) == set(_OWN_KERNELS) | set(D128_THD_SHARED_KERNELS), kernels  # the chain's five launches: four own + the shared fold
    assert _RECORD.is_file(), _RECORD
    _assert_record(kernels, _RECORD, _RECORD_TOKEN, "the s_tail default rendering")


def test_2cta_stail_host_renders_dense_form_for_sm_100a_and_matches_its_record(tmp_path):
    """RED-first (no ``layout`` form existed): the dense compact-BSHD twin (``layout = "dense"``: the setup kernel's dense arm synthesizing the
    metadata from the two declared extents, the head-major Stats binding with the synthesized stats origin, no length / offset operands)
    trace-compiles on the host for sm_100a (no device).  The MAIN kernel's `.entry` region hashes to the DEFAULT record's main-kernel md5 -- the
    twin is a host lowering, the pair body is byte-identical by construction and this row is the proof -- with the main kernel's census and the
    register split the default form's; the three helper kernels the arm re-renders (setup / delta / cast: their origin-spec and layout Constexprs
    differ) match their own committed rows (token ``_dense``, same file; skips, never widens, on another DSL build)."""
    from cudnn.sdpa.bwd.d128_grouped_2cta_contract import DENSE_ORIGIN_DTYPES, DENSE_ORIGIN_SPECS

    md5, census, regs, kernels = render_2cta_kernels(
        tmp_path, {"layout": "dense"}, _CENSUS, config=(2, 8, 2, False), specs=DENSE_ORIGIN_SPECS, dtypes=DENSE_ORIGIN_DTYPES
    )
    print(f"\n2-CTA s_tail sm_100a dense twin: PTX md5 {md5} census {census} setmaxnreg {regs} kernels {kernels}")
    assert census == _EXPECTED, census  # the module census is the main kernel's: the three re-rendered helpers carry none of its rows
    assert sorted(regs) == _setmaxnreg_of(), regs
    assert set(kernels) == set(_OWN_KERNELS) | set(D128_THD_SHARED_KERNELS), kernels  # the same five launches as the packed chain
    _, default = _record_kernels(_RECORD, _RECORD_TOKEN)
    assert kernels[_MAIN_KERNEL] == default[_MAIN_KERNEL], (
        f"the dense form re-rendered the MAIN kernel ({kernels[_MAIN_KERNEL]} vs the default record's {default[_MAIN_KERNEL]}): "
        "the twin is a host lowering, the pair body must be byte-identical"
    )
    _assert_record(kernels, _RECORD, _RECORD_TOKEN_DENSE, "the dense twin's rendering")


def test_2cta_stail_host_renders_for_sm_103a_and_matches_the_record(tmp_path):
    """The GB300 target (cc 10.3 -> the prepared module's ``sm`` = 103 -> ``--gpu-arch sm_103a`` in the host, the same selection as the one-CTA
    body; the sm103 CI lane): the chain trace-compiles for sm_103a on the host (no device) with the SAME census and register split as the sm_100a
    rendering, and every kernel's `.entry` region hashes to the committed (sm_100a) record -- the body has no per-part branch, so only the module's
    `.target` line differs between the two targets (the module md5 does, the per-kernel rows do not).  Like the one-CTA body's records, one record
    file serves; skips by name on a DSL without the target or another DSL build."""
    from frost_test_utils import arch_known_to_the_dsl

    if not arch_known_to_the_dsl("sm_103a"):
        pytest.skip("this cutlass-dsl has no sm_103a target")
    md5, census, regs, kernels = render_2cta_kernels(tmp_path, {}, _CENSUS, arch="sm_103a")
    print(f"\n2-CTA s_tail sm_103a: PTX md5 {md5} census {census} setmaxnreg {regs} kernels {kernels}")
    assert census == _EXPECTED, census
    assert sorted(regs) == _setmaxnreg_of(), regs
    assert set(kernels) == set(_OWN_KERNELS) | set(D128_THD_SHARED_KERNELS), kernels
    _assert_record(kernels, _RECORD, _RECORD_TOKEN, "the s_tail default rendering for sm_103a (per-kernel `.entry` md5s are target-independent)")


def test_2cta_stail_host_renders_for_sm_107a_and_matches_the_record(tmp_path):
    """The Rubin target (cc 10.7 -> the prepared module's ``sm`` = 107 -> ``--gpu-arch sm_107a`` in the host; the cc 10.7 admission of the d128
    family, 2026-11-03) with every field unset: ``tmem_map`` resolves to the part's default "s_tail" (dq_576 is selectable by name only), so the
    chain trace-compiles for sm_107a on the host (no device) with the SAME census and register split as the sm_100a rendering, and every kernel's
    `.entry` region hashes to the committed (sm_100a) record -- the s_tail body has no per-part branch, so only the module's `.target` line differs
    (as for sm_103a); "s_tail" NAMED renders the same module.  Skips BY NAME on a DSL without the sm_107a target (the public 4.7.x wheels) or on
    another DSL build -- never a silent pass."""
    from frost_test_utils import arch_known_to_the_dsl

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a target")
    md5, census, regs, kernels = render_2cta_kernels(tmp_path, {}, _CENSUS, arch="sm_107a")
    print(f"\n2-CTA s_tail sm_107a (the default): PTX md5 {md5} census {census} setmaxnreg {regs} kernels {kernels}")
    assert census == _EXPECTED, census
    assert sorted(regs) == _setmaxnreg_of(), regs
    assert set(kernels) == set(_OWN_KERNELS) | set(D128_THD_SHARED_KERNELS), kernels
    _assert_record(kernels, _RECORD, _RECORD_TOKEN, "the s_tail default rendering for sm_107a (per-kernel `.entry` md5s are target-independent)")
    md5_n, census_n, _, kernels_n = render_2cta_kernels(tmp_path, {"tmem_map": "s_tail"}, _CENSUS, arch="sm_107a")
    assert md5_n == md5 and census_n == census and kernels_n == kernels, "s_tail named explicitly IS the cc 10.7 default rendering"


def test_2cta_stail_host_renders_dq576_named_for_sm_107a_and_matches_its_record(tmp_path):
    """``tmem_map = "dq_576"`` NAMED on cc 10.7 (PART-KEYED: the independent dQ image at [512, 576) of the 576-column carve, the kernel header
    sections 9 / 11; a neutral structural map, selectable by name, never the default): the chain trace-compiles for sm_107a on the host (no device)
    with the s_tail census MINUS exactly the two deleted rows' sites (``_tmem_map_census_delta``: arrive / mapa / init -2, the leader's 5 wait sites
    in try_wait / bar.warp.sync / fence::after, the softmax's one before-fence) and the alloc spelled `tcgen05.alloc.exclusive.cta_group::2` (the
    512 form gone); the same register split; the record is its OWN per-kernel file at its OWN target (``_RECORD_DQ576``); the three kernels that do
    not read the map (setup, delta-and-stats, cast) hash to the s_tail record's rows.  Skips BY NAME on a DSL without the sm_107a target or on
    another DSL build."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from frost_test_utils import arch_known_to_the_dsl

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a target")
    census_rows = dict(_CENSUS, **_TMEM_MAP_CENSUS)
    md5_s, census_s, regs_s, kernels_s = render_2cta_kernels(tmp_path, {}, census_rows, arch="sm_107a")  # the default = s_tail
    md5, census, regs, kernels = render_2cta_kernels(tmp_path, {"tmem_map": "dq_576"}, census_rows, arch="sm_107a")
    print(f"\n2-CTA dq_576 sm_107a (named): PTX md5 {md5} census {census} setmaxnreg {regs} kernels {kernels}")
    assert {k: v for k, v in census_s.items() if k not in _TMEM_MAP_CENSUS} == _EXPECTED, census_s
    assert census == dict(census_s, **_tmem_map_census_delta(cfgmod.CFG, cfgmod, census_s)), {
        k: (census_s.get(k), census.get(k)) for k in census if census_s.get(k) != census.get(k)
    }
    assert sorted(regs) == _setmaxnreg_of() and regs == regs_s, (regs, regs_s)
    assert set(kernels) == set(_OWN_KERNELS) | set(D128_THD_SHARED_KERNELS), kernels
    assert md5 != md5_s and kernels[_OWN_KERNELS[2]] != kernels_s[_OWN_KERNELS[2]], "the map renders a different MAIN kernel"
    assert all(kernels[k] == kernels_s[k] for k in _OWN_KERNELS if k != _OWN_KERNELS[2]), (kernels, kernels_s)  # only the MAIN kernel reads the map
    assert _RECORD_DQ576.is_file(), _RECORD_DQ576
    _assert_record(kernels, _RECORD_DQ576, _RECORD_TOKEN_DQ576, "the dq_576 rendering named for sm_107a", arch=_RECORD_ARCH_DQ576)


def test_2cta_stail_host_renders_dq576_ldg_for_sm_107a_and_matches_its_record(tmp_path):
    """``tmem_map = "dq_576"`` x ``stats_form = "ldg"``, both NAMED on cc 10.7 (the lever cross the dq_576 record left unpinned -- the barrier AUDIT's
    u2): the chain trace-compiles for sm_107a on the host (no device) and its census is EXACTLY the s_tail x ldg rendering's census minus the map's
    delta (``_tmem_map_census_delta``: the two levers commute -- the ring rows the ldg form removes and the two rows the map deletes are disjoint
    sites), the same register split, the ldg chain's five launches (the compact delta launch); the MAIN kernel differs from both the dq_576 default's
    and the s_tail x ldg one's; the three kernels that do not read the map (setup without the row-offset pass, compact delta, cast) hash to the ldg
    record's rows (sm_100a: per-kernel `.entry` md5s are target-independent); the record is its OWN token in the dq_576 file at its own target.
    Skips BY NAME on a DSL without the sm_107a target or on another DSL build."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from frost_test_utils import arch_known_to_the_dsl

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a target")
    rows = {**_CENSUS, **_TMEM_MAP_CENSUS, **_STATS_CENSUS}  # mbarrier_init is in both lever tables (the same regex)
    md5_l, census_l, regs_l, kernels_l = render_2cta_kernels(tmp_path, {"stats_form": "ldg"}, rows, arch="sm_107a")  # s_tail x ldg
    md5_d, census_d, regs_d, kernels_d = render_2cta_kernels(tmp_path, {"tmem_map": "dq_576"}, rows, arch="sm_107a")  # dq_576 x tma_ring (the record)
    md5, census, regs, kernels = render_2cta_kernels(tmp_path, {"tmem_map": "dq_576", "stats_form": "ldg"}, rows, arch="sm_107a")
    print(f"\n2-CTA dq_576 x ldg sm_107a (named): PTX md5 {md5} census {census} setmaxnreg {regs} kernels {kernels}")
    assert {k: v for k, v in census_l.items() if k not in _TMEM_MAP_CENSUS and k not in _STATS_CENSUS} == _EXPECTED_LDG, census_l
    expected = dict(census_l, **_tmem_map_census_delta(cfgmod.CFG, cfgmod, census_l))
    assert census == expected, {k: (census.get(k), expected.get(k)) for k in set(census) | set(expected) if census.get(k) != expected.get(k)}
    assert (census["bar_sync_1_256"], census["st_shared_b32"], census["bulk_g2s_plain"]) == (4, 4, 0), census  # the ldg form's rows under the map
    assert sorted(regs) == _setmaxnreg_of() and regs == regs_l == regs_d, (regs, regs_l, regs_d)
    assert set(kernels) == set(_OWN_KERNELS_LDG) | set(D128_THD_SHARED_KERNELS), kernels
    main = _OWN_KERNELS[2]
    assert md5 not in (md5_l, md5_d) and kernels[main] not in (kernels_l[main], kernels_d[main]), "dq_576 x ldg renders its own MAIN kernel"
    _, ldg_record = _record_kernels(_RECORD, _RECORD_TOKEN_LDG)
    assert all(kernels[k] == kernels_l[k] == ldg_record[k] for k in _OWN_KERNELS_LDG if k != main), (kernels, kernels_l, ldg_record)
    assert _RECORD_DQ576.is_file(), _RECORD_DQ576
    _assert_record(kernels, _RECORD_DQ576, _RECORD_TOKEN_DQ576_LDG, "the dq_576 x ldg rendering named for sm_107a", arch=_RECORD_ARCH_DQ576)


def test_2cta_stail_host_renders_probes_under_dq576_for_sm_107a(tmp_path):
    """The four probes the dq_576 map SERVES (``CFG.DIAG_2CTA_NAMES_ON_MAP`` minus ``DIAG_2CTA_S_TAIL_PROBES``: the issue / transfer / bytes probes)
    render under ``tmem_map = "dq_576"`` on sm_107a with EXACTLY their own purity delta against the dq_576 default rendering (``_probe_census_delta``:
    the same arithmetic as on the s_tail map -- a probe moves no barrier row on either map; the dq_576 default itself is the s_tail census minus the
    map's delta, the route rows untouched), each a different md5 from that default; the three S-tail WAIT probes decline BY NAME at ``compile_host``
    under the map (their wait sites are the deleted rows) -- never a rendering (the barrier AUDIT's u4).  Skips BY NAME on a DSL without the sm_107a
    target."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_host import compile_host_grouped_2cta as compile_host
    from frost_test_utils import arch_known_to_the_dsl

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a target")
    c = cfgmod.CFG
    served = tuple(p for p in _PROBES if p not in c.DIAG_2CTA_S_TAIL_PROBES)
    assert served == tuple(p for p in c.DIAG_2CTA_NAMES_ON_MAP if p not in c.DIAG_2CTA_S_TAIL_PROBES) and len(served) == 4, served
    rows = {**_CENSUS, **_TMEM_MAP_CENSUS, **_ROUTE_CENSUS}
    md5_d, census_d, _, _ = render_2cta_kernels(tmp_path, {"tmem_map": "dq_576"}, rows, arch="sm_107a")
    delta = _tmem_map_census_delta(c, cfgmod, dict(_EXPECTED, mbarrier_init=0, tcgen05_fence_before=0))  # the module-census rows of the map's delta
    expected_d = dict(_EXPECTED, **{k: v for k, v in delta.items() if k in _EXPECTED}, sts_dq_v2_b64=24, fence_proxy_async=15, bulk_commit_group=5)
    assert {k: v for k, v in census_d.items() if k not in _TMEM_MAP_CENSUS} == expected_d, census_d
    args = ((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True)
    for probe in c.DIAG_2CTA_S_TAIL_PROBES:
        with pytest.raises(ValueError, match=f"diag_2cta '{probe}' is not defined under tmem_map 'dq_576'"):
            compile_host(*args, 107, tmem_map="dq_576", diag_2cta=probe)
    for probe in served:
        md5, census, _, _ = render_2cta_kernels(tmp_path, {"tmem_map": "dq_576", "diag_2cta": probe}, rows, arch="sm_107a")
        print(f"\n2-CTA dq_576 sm_107a PTX census ({probe}): md5 {md5} {census}")
        expected = dict(census_d, **_probe_census_delta(probe, c, census_d))
        assert census == expected, (probe, {k: (census.get(k), expected.get(k)) for k in set(census) | set(expected) if census.get(k) != expected.get(k)})
        assert md5 != md5_d, f"{probe} must render different PTX from the dq_576 default"


def test_2cta_stail_host_renders_sm100_oversized_byte_identical_to_sm100(tmp_path):
    """The DIAGNOSTIC map ``smem_map = "sm100_oversized"`` renders the sm100 map's kernels BYTE-IDENTICAL on both TMEM maps (every per-kernel md5 equal,
    the census equal, the register split equal): the dynamic-SMEM declaration (`.extern .shared ... []`) carries no size in PTX -- the padded size
    reaches the launcher alone (``geo.SMEM_BYTES`` 262144 of the two padded records).  The real maps' records hold (dq_576: the R1 record; s_tail named
    on sm_107a: the sm_100a default record).  Skips BY NAME on a DSL without the sm_107a target or on another DSL build."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from frost_test_utils import arch_known_to_the_dsl

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a target")
    assert "slab = cutlass.Array(cutlass.Uint8, geo.SMEM_BYTES, alignment=1024, space=cutlass.AddressSpace.smem)" in _body_source()  # the one size read
    census_rows = dict(_CENSUS, **_SMEM_MAP_CENSUS)
    for tmem, record, token, arch in (("dq_576", _RECORD_DQ576, _RECORD_TOKEN_DQ576, _RECORD_ARCH_DQ576), ("s_tail", _RECORD, _RECORD_TOKEN, "sm_100a")):
        md5_s, census_s, regs_s, kernels_s = render_2cta_kernels(tmp_path, {"tmem_map": tmem}, census_rows, arch="sm_107a")
        md5_o, census_o, regs_o, kernels_o = render_2cta_kernels(tmp_path, dict(_OVS_FIELDS, tmem_map=tmem), census_rows, arch="sm_107a")
        print(f"\n2-CTA {tmem} x sm100_oversized sm_107a: PTX md5 {md5_o} (sm100 {md5_s}) kernels {kernels_o}")
        assert kernels_o == kernels_s and census_o == census_s and regs_o == regs_s, (kernels_o, kernels_s)
        _assert_record(kernels_o, record, token, f"the {tmem} x sm100_oversized rendering (= the sm100 record)", arch=arch)
        ovs = cfgmod.cfg_for_geometry("tma_ring", tmem, "sm100_oversized")
        assert ovs.SMEM_BYTES == 262144 and ovs.SCHED_PAYLOAD == cfgmod.cfg_for_geometry("tma_ring", tmem, "sm100").SCHED_PAYLOAD


def test_2cta_stail_host_renders_rubin_b2_named_for_sm_107a_and_matches_its_record(tmp_path):
    """``smem_map = "rubin_b2"`` NAMED together with ``tmem_map = "dq_576"`` on cc 10.7 (PART-KEYED; the 327 KiB oversized carveout: sQ1 x2, sDO1 x2,
    the two-stage sDS + XCHG staging ring with the lagging epilogue order and the dq_read_done row -- the kernel header sections 5 / 8 / 11; selectable
    by name, never a default today): the chain trace-compiles for sm_107a on the host (no device) with the dq_576 x sm100 census PLUS exactly the
    rows ``_smem_map_census_delta`` names (the dq_read_done row's two arrive sites and three wait sites, the item-end readout site, the two extra dQ
    rounds, the two deeper drains, the nine extra inits) and nothing else (the MMA / commit / TMA / ship / reduce-add / expect_tx / LDS rows, the
    exclusive alloc, .local 0, nanosleep 0 and the register split unchanged); the record is its OWN per-kernel file at its OWN target (``_RECORD_B2``);
    the three kernels that read neither map hash to the dq_576 record's rows; the dq_576 x sm100 rendering (the SMEM map unset or NAMED "sm100")
    stays the dq_576 record's.  Skips BY NAME on a DSL without the sm_107a target or on another DSL build."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from frost_test_utils import arch_known_to_the_dsl

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a target")
    census_rows = dict(_CENSUS, **_SMEM_MAP_CENSUS)
    md5_d, census_d, regs_d, kernels_d = render_2cta_kernels(tmp_path, {"tmem_map": "dq_576"}, census_rows, arch="sm_107a")  # dq_576 x sm100
    md5, census, regs, kernels = render_2cta_kernels(tmp_path, _B2_FIELDS, census_rows, arch="sm_107a")
    print(f"\n2-CTA dq_576 x rubin_b2 sm_107a (named): PTX md5 {md5} census {census} setmaxnreg {regs} kernels {kernels}")
    assert census == dict(census_d, **_smem_map_census_delta(cfgmod.CFG, cfgmod, census_d)), {
        k: (census_d.get(k), census.get(k)) for k in census if census_d.get(k) != census.get(k)
    }
    assert (census["tmem_alloc_exclusive_cta2"], census["tmem_alloc_cta2"], census["local_mem"], census["nanosleep"]) == (1, 0, 0, 0), census
    assert sorted(regs) == _setmaxnreg_of() and regs == regs_d, (regs, regs_d)
    assert set(kernels) == set(_OWN_KERNELS) | set(D128_THD_SHARED_KERNELS), kernels
    assert md5 != md5_d and kernels[_OWN_KERNELS[2]] != kernels_d[_OWN_KERNELS[2]], "the SMEM map renders a different MAIN kernel"
    assert all(kernels[k] == kernels_d[k] for k in _OWN_KERNELS if k != _OWN_KERNELS[2]), (kernels, kernels_d)  # only the MAIN kernel reads the maps
    _assert_record(kernels_d, _RECORD_DQ576, _RECORD_TOKEN_DQ576, "the dq_576 x sm100 rendering (arm R1's record holds)", arch=_RECORD_ARCH_DQ576)
    assert _RECORD_B2.is_file(), _RECORD_B2
    _assert_record(kernels, _RECORD_B2, _RECORD_TOKEN_B2, "the dq_576 x rubin_b2 rendering named for sm_107a", arch="sm_107a")
    md5_n, census_n, _, kernels_n = render_2cta_kernels(tmp_path, {"tmem_map": "dq_576", "smem_map": "sm100"}, census_rows, arch="sm_107a")
    assert md5_n == md5_d and census_n == census_d and kernels_n == kernels_d, "sm100 named explicitly IS the dq_576 rendering"


@pytest.mark.parametrize("probe", _PROBES)
def test_2cta_stail_host_renders_probes_for_sm_100a(tmp_path, probe):
    """``diag_2cta = <probe>`` removes exactly its own ISSUE (dq_reduce_add_off: the four reduce-adds), TRANSFER (exchange_off: the one bulk copy +
    its two mapa, the arm keeping its count with a 0-byte operand), BYTES (stats_one_chunk: 28 of the 32 stats LDS.128; sdq_sts_off: the 24
    sDQ slice STS + their 3 proxy fences) or WAIT sites (s_tail_wait_off: the leader's 4 s_tail_empty `_wait_tcgen`; s_read_wait_off: its 1 s_read
    `_wait_tcgen`; the combined name: 5 -- each site one try_wait + one bar.warp.sync + one tcgen05.fence::after) and nothing else -- every
    arrive site stays (the barrier table is identical: the purity census); the probe renders a different md5 from the default."""
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
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_host import compile_host_grouped_2cta as compile_host

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
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_host import compile_host_grouped_2cta as compile_host

    with pytest.raises(ValueError, match="debug_clk is not defined under warp_roles '16w_fa4'"):
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, warp_roles="16w_fa4", debug_clk=True)
    with pytest.raises(ValueError, match="dkv_accum 'fp32_reduce' is not built under warp_roles '16w_fa4'"):
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, warp_roles="16w_fa4", dkv_accum="fp32_reduce")
    with pytest.raises(ValueError, match="stats_form 'tma_ring' is not built under warp_roles '16w_fa4'"):  # lever S1: the ring is the 16w bundle's
        compile_host((2, 8, 2, True), (None,) * 9, (None,) * 9, None, True, 100, warp_roles="16w_fa4", stats_form="tma_ring")
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
    SAME barrier census: 42 arrive / 70 try_wait / 12 expect_tx / 69 bar.warp.sync / 32 tcgen05.fence sites, the ship, the reduce-adds, the STS,
    no local memory, no nanosleep, the default register split -- under a different md5; the default (False) rendering is the record's."""
    md5_on, census_on, regs_on = _render(tmp_path, {"debug_clk": True})
    print(f"\n2-CTA s_tail debug_clk sm_100a: PTX md5 {md5_on} census {census_on} setmaxnreg {regs_on}")
    # the lever adds the clock reads and the record dump's SMEM reads (the 100-word copy at exit vectorizes into 16-B loads); nothing else moves
    assert census_on["clock64"] > 0 and census_on["lds_v4_b32"] >= _EXPECTED["lds_v4_b32"], census_on
    assert census_on == dict(_EXPECTED, clock64=census_on["clock64"], lds_v4_b32=census_on["lds_v4_b32"]), census_on
    assert sorted(regs_on) == _setmaxnreg_of(), regs_on
    md5_off, census_off, _, kernels_off = _render_kernels(tmp_path, {"debug_clk": False})
    assert census_off == _EXPECTED and md5_off != md5_on, "debug_clk=False is the default rendering; True must differ (the clock reads)"
    _assert_record(kernels_off, _RECORD, _RECORD_TOKEN, "debug_clk=False (the default rendering)")


# ------------------------------------------------------------------------------------------------------------ host: the audit's deterministic-hang topologies as SOURCE assertions
def test_2cta_stail_host_renders_mask_forms_for_sm_100a(tmp_path):
    """Lever the mask-form lever's three mask forms on this body (header section 11): "fsel" named explicitly IS the default rendering (the record's md5); "predicated"
    renders a DIFFERENT PTX under the SAME mbarrier census (no wait, arrive or fence moves), the same setmaxnreg split, the SAME number of per-cell
    selects and keep-word shifts (the mask body is kept, inside the branch) and exactly CFG.QUARTERS_PER_GROUP more branches (one CTA-uniform
    `if mask_tile:` per q quarter); "diag_mask_off" drops exactly CFG.MASK_SELECTS_PER_LANE_PER_TILE selects and CFG.MASK_WORDS_PER_LANE_PER_TILE
    upper-edge shifts (the whole mask body) under the same census; the three md5s differ."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG

    md5_d, census_d, regs_d, kernels_d = _render_kernels(tmp_path, {}, _MASK_CENSUS)
    md5_f, census_f, regs_f = _render(tmp_path, {"mask_form": "fsel"}, _MASK_CENSUS)
    md5_p, census_p, regs_p = _render(tmp_path, {"mask_form": "predicated"}, _MASK_CENSUS)
    md5_o, census_o, regs_o = _render(tmp_path, {"mask_form": "diag_mask_off"}, _MASK_CENSUS)
    print(
        f"\n2-CTA s_tail mask forms sm_100a: default {md5_d} fsel {md5_f} predicated {md5_p} diag_mask_off {md5_o}; census {census_d} | {census_p} | {census_o}"
    )
    assert md5_f == md5_d and census_f == census_d and regs_f == regs_d, "fsel named explicitly is the default rendering"
    assert len({md5_d, md5_p, md5_o}) == 3, "the three mask forms must render three different PTX texts"
    for census in (census_d, census_p, census_o):
        assert {k: v for k, v in census.items() if k not in _MASK_CENSUS} == _EXPECTED, census  # no barrier, TMA or TMEM op moves under any form
    assert sorted(regs_d) == sorted(regs_p) == sorted(regs_o) == _setmaxnreg_of(), (regs_d, regs_p, regs_o)
    assert census_p["selp_f32"] == census_d["selp_f32"] and census_p["shr_u32"] == census_d["shr_u32"], (census_d, census_p)
    assert census_p["bra"] - census_d["bra"] == CFG.QUARTERS_PER_GROUP, (census_d["bra"], census_p["bra"])
    assert census_d["selp_f32"] - census_o["selp_f32"] == CFG.MASK_SELECTS_PER_LANE_PER_TILE, (census_d, census_o)
    assert census_d["shr_u32"] - census_o["shr_u32"] == CFG.MASK_WORDS_PER_LANE_PER_TILE, (census_d, census_o)
    _assert_record(kernels_d, _RECORD, _RECORD_TOKEN, "the default (fsel) rendering: the mask-form chain must leave the default byte-identical")


def test_2cta_stail_host_renders_stats_forms_for_sm_100a(tmp_path):
    """Lever S1's two stats forms (header section 11): "tma_ring" named explicitly IS the default rendering (the record's per-kernel md5s, the default
    census: the ring's 43 inits, one plain `cp.async.bulk` global -> shared, no `bar.sync 1,256`, no stats STS); "ldg" (the previous default) renders a
    DIFFERENT PTX whose census differs EXACTLY by the rows the skeleton names (resources/s1-barrier-table.md 14.6): `bar.sync 1,256` 0 -> 4 and the two
    stats STS 0 -> 4 (the `_stats` block back), the plain bulk copy 1 -> 0, the two ring rows' inits (-4), the ring's arm (expect_tx -1) and its release
    (-1 plain arrive = arrive 42 -> 40), the two ring waits (try_wait 70 -> 68, bar.warp.sync 69 -> 67); the chain-level global-load count is equal
    (the softmax's six loads against the loader's one and the setup / delta launches' five); every other row of the module census identical (no proxy
    fence, no tcgen05 fence, no LDS.128 moved, the 144 / 144 / 80 split, no local memory, no nanosleep); each form matches its committed per-kernel
    record rows (the ldg rows under their own token); the default x ``debug_clk`` renders with the ring's barrier census + the clock reads only."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG_RING, STATS_RING_ROWS

    md5_d, census_d, regs_d, kernels_d = _render_kernels(tmp_path, {}, _STATS_CENSUS)
    md5_r, census_r, regs_r = _render(tmp_path, {"stats_form": "tma_ring"}, _STATS_CENSUS)
    md5_l, census_l, regs_l, kernels_l = _render_kernels(tmp_path, {"stats_form": "ldg"}, _STATS_CENSUS)
    print(f"\n2-CTA s_tail stats forms sm_100a: default {md5_d} tma_ring {md5_r} ldg {md5_l}; census {census_d} | {census_l}")
    assert md5_r == md5_d and census_r == census_d and regs_r == regs_d, "tma_ring named explicitly is the default rendering"
    assert {k: v for k, v in census_d.items() if k not in _STATS_CENSUS} == _EXPECTED, census_d
    assert (census_d["bar_sync_1_256"], census_d["st_shared_b32"], census_d["bulk_g2s_plain"], census_d["mbarrier_init"]) == (0, 0, 1, 43), census_d
    assert set(kernels_d) == set(_OWN_KERNELS) | set(D128_THD_SHARED_KERNELS), kernels_d  # the ring chain's five launches
    _assert_record(kernels_d, _RECORD, _RECORD_TOKEN, "the default (tma_ring) rendering")
    expected_ldg = dict(
        _EXPECTED_LDG,
        bar_sync_1_256=4,  # the named-barrier pair of `_stats` at its two sites (the item prologue, the per-tile lookahead)
        st_shared_b32=4,  # the two stats STS per site
        bulk_g2s_plain=0,  # the ring's copy is the ONLY plain global -> shared bulk copy of this body
        mbarrier_init=census_d["mbarrier_init"] - len(STATS_RING_ROWS) * CFG_RING.STATS_STAGES,  # stats_full[2] + stats_empty[2] = 4 inits fewer
        # the chain-level count (the module census spans the five launches): the MAIN kernel's two `_stats` sites load six words (origin word,
        # LSE, delta x 2) where the ring's loader loads one (rowoff[b] per item) = +5; the setup kernel's row-offset loop (-2) and the pre-scaling
        # delta launch's LSE / row-offset loads (-3) take the same five away: net 0 (per-kernel split: the host census of the writer's section)
        ld_global=census_d["ld_global"],
        fence_proxy_async=census_d["fence_proxy_async"],  # neither form needs a proxy fence for the stats (async write -> mbarrier -> generic read)
    )
    assert census_l == expected_ldg, {
        k: (census_l.get(k), expected_ldg.get(k)) for k in set(census_l) | set(expected_ldg) if census_l.get(k) != expected_ldg.get(k)
    }
    assert regs_l == regs_d, "the 144 / 144 / 80 split is the same under both forms"
    assert md5_l != md5_d, "ldg must render different PTX from tma_ring"
    # the ldg chain's five launches: the compact delta launch; the main kernel and the setup kernel (no padded row-offset pass) render differently
    # from the default's, the dQ cast is the same kernel; the shared fold is rendered and excluded from the pin
    assert set(kernels_l) == set(_OWN_KERNELS_LDG) | set(D128_THD_SHARED_KERNELS), kernels_l
    assert kernels_l[_OWN_KERNELS[2]] != kernels_d[_OWN_KERNELS[2]], (kernels_d, kernels_l)
    # Pin the cast independently in each form: its entry-region hash also includes the following kernel's .globl annotation,
    # which can differ between forms even when the cast instructions are identical.
    _assert_record(kernels_l, _RECORD, _RECORD_TOKEN_LDG, "the ldg rendering")
    md5_c, census_c, regs_c = _render(tmp_path, {"debug_clk": True}, _STATS_CENSUS)
    assert census_c["clock64"] > 0 and md5_c != md5_d and regs_c == regs_d
    # the STAMPS lever adds its clock reads, its record's word stores (st.shared.b32: the 88-word record, zeroed and accumulated) and its record
    # LDS; every barrier / TMA / copy row is the ring's (the two ring wait words bracket the two ring waits: no barrier moves)
    moved_by_clk = ("clock64", "lds_v4_b32", "st_shared_b32")
    assert census_c["st_shared_b32"] > 0, "the STAMPS record's stores (the stats STS stay gone: the default render above pins 0)"
    assert {k: v for k, v in census_c.items() if k not in moved_by_clk} == {k: v for k, v in census_d.items() if k not in moved_by_clk}


def test_2cta_stail_stats_ring_sites_follow_the_skeleton():
    """SOURCE assertions on lever S1's sites (resources/s1-barrier-table.md sections 1 / 7.1 / 14 -- the deterministic-hang topologies the skeleton
    ruled out become rows): every stats-ring statement sits under `if cutlass.const_expr(tma_ring):` (the ldg rendering has no stats_full / stats_empty /
    bulk copy), every `_stats` call under `not tma_ring` (the tma_ring rendering has no `bar.sync 1,256`); the softmax waits stats_full AFTER the
    s_read arrive and BEFORE _p_values, releases stats_empty right after _ds_values (its last stats read) and before the STORE wait, with a plain
    `_wait` (never `_wait_tcgen`, no proxy fence); the loader waits stats_empty, arms stats_full with `pred=own_pred` (`elect_sync()`, never
    `is_leader`) and issues ONE `bulk_copy` onto its OWN mbar right after Q1(t) and before the dO1 wait; both ring states are declared BEFORE the
    persistent loop and advanced once per tile inside it (never reset at the item top: the odd-K hang of the skeleton's 14.4)."""
    src = _body_source()
    softmax = _function_code(src, "_softmax_loop", "_epilogue_loop")
    loader = _function_code(src, "_load_loop", "_store_loop")

    def guarded(lines, needle, guard):
        hits = [i for i, ln in enumerate(lines) if needle in ln]
        assert hits, needle
        for i in hits:
            indent = len(lines[i]) - len(lines[i].lstrip())
            j = i - 1
            while j >= 0 and (len(lines[j]) - len(lines[j].lstrip()) >= indent or not lines[j].strip().startswith("if cutlass.const_expr(")):
                j -= 1
            assert j >= 0 and guard in lines[j], (needle, lines[i].strip(), lines[j].strip() if j >= 0 else None)
        return hits

    first = lambda lines, needle: next(i for i, ln in enumerate(lines) if needle in ln)  # noqa: E731
    # the softmax: declared before the loop; the wait after s_read and before _p_values; the release after _ds_values and before the STORE wait
    while_at = first(softmax, "while valid != 0:")
    assert first(softmax, "stats_state = PipelineState.start()") < while_at and sum("stats_state = PipelineState.start(" in ln for ln in softmax) == 1
    guarded(softmax, "stats_state = PipelineState.start()", "tma_ring")
    wait_at = guarded(softmax, "_wait(bars.stats_full[stats_state.idx], stats_state.phase", "tma_ring")[0]
    assert first(softmax, "_arrive_leader(bars.s_read)") < wait_at < first(softmax, "p = _p_values(slab, stats_base,")
    assert not any("_wait_tcgen(bars.stats_full" in ln for ln in softmax) and not any("fence_proxy" in ln and "stats" in ln for ln in softmax)
    rel_at = guarded(softmax, "bars.stats_empty[stats_state.idx].arrive()", "tma_ring")[0]
    ds_at = first(softmax, "words = _ds_values(slab, stats_base + CFG.STATS_DELTA_OFF,")
    store_waits = [i for i, ln in enumerate(softmax) if "tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)" in ln]
    assert ds_at < rel_at < min(i for i in store_waits if i > ds_at)
    adv_at = guarded(softmax, "stats_state = advance(stats_state, geo.STATS_STAGES)", "tma_ring")[0]  # the geometry's depth (2 under every tma_ring record)
    assert rel_at < adv_at and adv_at < first(softmax, "tile_state = advance(tile_state, 1)") and adv_at > while_at
    for needle in ("_stats(slab, lse, delta, row, group, lo * CFG.N,", "_stats(slab, lse, delta, row, group, (qtile + 1) * CFG.N,"):
        guarded(softmax, needle, "not tma_ring")
    assert not any("named_barrier_fence(CFG.SOFTMAX_BARRIER" in ln for ln in softmax)  # the pair lives inside `_stats` only (retired with it)
    # the loader: declared before the loop; the own-CTA arm; wait -> arm -> ONE bulk copy after Q1(t) and before the dO1 wait; one advance per tile
    lwhile = first(loader, "while valid != 0:")
    assert first(loader, "stats_ld = PipelineState.start(phase=1)") < lwhile and sum("stats_ld = PipelineState.start(" in ln for ln in loader) == 1
    guarded(loader, "stats_ld = PipelineState.start(phase=1)", "tma_ring")
    guarded(loader, "own_pred = nvvm.elect_sync()", "tma_ring")
    lwait = guarded(loader, "_wait(bars.stats_empty[stats_ld.idx], stats_ld.phase", "tma_ring")[0]
    arm = guarded(loader, "bars.stats_full[stats_ld.idx].arrive(n_bytes=CFG.STATS_TX, pred=own_pred)", "tma_ring")[0]
    copy = guarded(loader, "bulk_copy(", "tma_ring")[0]
    q1_load = [i for i, ln in enumerate(loader) if "tma_slice_runtime_desc(q1_desc, 0, q_row + q_half, head, 0)" in ln][0]
    assert (
        q1_load < lwait < arm < copy < first(loader, "_wait(_slot(bars.do1_empty, do1_state, geo.DO1_STAGES), do1_state.phase")
    )  # the ring slot (depth 1 or 2)
    depth, end = 0, copy  # the copy call's extent (black spreads its arguments over lines): the mbar operand is OUR stats_full copy
    while True:
        depth += loader[end].count("(") - loader[end].count(")")
        if depth <= 0:
            break
        end += 1
    assert sum("bulk_copy(" in ln for ln in loader) == 1 and any("bars.stats_full[stats_ld.idx].smem_ptr" in ln for ln in loader[copy : end + 1])
    assert not any("is_leader" in ln and "stats_full" in ln for ln in loader) and not any("tma_tensormap_acquire" in ln and "stats" in ln for ln in loader)
    ladv = guarded(loader, "stats_ld = advance(stats_ld, geo.STATS_STAGES)", "tma_ring")[0]
    assert copy < ladv and ladv < first(loader, "pair, head, batch, valid, sched_state = _next_tile(")


def test_2cta_stail_dp_empty_is_arrived_at_the_ds_readout_before_the_recv_arm():
    """RED 3: dp_empty arrived after the dQ readout under the rotated order is a deterministic HANG (dP^T(t+1) waits a readout of a dQ issued
    after it).  The epilogue's ONE dp_empty arrive precedes the ds_recv_full arm and the dQ readout."""
    full = _function_code(_body_source(), "_epilogue_loop", "_mma_loop_leader")
    code = _sm100_arm(full)  # the sm100 SMEM maps' view (one readout site); the rubin_b2 view has two readout sites, both after the arm (below)
    arrive = [i for i, ln in enumerate(code) if "_arrive_leader(bars.dp_empty)" in ln]
    recv = [i for i, ln in enumerate(code) if "recv_bar.arrive(" in ln]  # the ds_recv_full arm (the slot of the stage under rubin_b2, the bar under sm100)
    dq_ld = [i for i, ln in enumerate(code) if "_tmem_load(tmem_base + dq_col, CFG.DQ_COLS)" in ln]  # the map-keyed column base (header section 9)
    assert len(arrive) == 1 and len(recv) == 1 and len(dq_ld) == 1, (arrive, recv, dq_ld)
    assert arrive[0] < recv[0] < dq_ld[0], "dp_empty must be arrived right after the dS^T readout, before the XCHG arm, never after the dQ readout"
    s_tail = [i for i, ln in enumerate(code) if "_arrive_leader(bars.s_tail_empty)" in ln]
    assert len(s_tail) == 1 and dq_ld[0] < s_tail[0], "s_tail_empty is arrived right after the dQ readout"
    b2 = _b2_arm(full)
    dq_ld_b2 = [i for i, ln in enumerate(b2) if "_tmem_load(tmem_base + dq_col, CFG.DQ_COLS)" in ln]
    arrive_b2, recv_b2 = [i for i, ln in enumerate(b2) if "_arrive_leader(bars.dp_empty)" in ln], [i for i, ln in enumerate(b2) if "recv_bar.arrive(" in ln]
    assert len(arrive_b2) == 1 and len(recv_b2) == 1 and len(dq_ld_b2) == 2 and arrive_b2[0] < recv_b2[0] < dq_ld_b2[0] < dq_ld_b2[1], (
        arrive_b2,
        recv_b2,
        dq_ld_b2,
    )


def test_2cta_stail_ds_gates_ride_the_xchg_state():
    """RED 4: the three dS gates waited on ready_state (the P2 split missing) hang at tile lo+1 of the first N >= 2 item.  Every ds_smem /
    ds_recv_full / ds_pair_ready wait of the leader names ``xchg_state.phase`` (one helper ``_dq_issue``, called from the in-tile slot helper
    ``_dq_slot`` and at the tail), never ready_state.  The in-tile slot is ONE helper at ONE site (before dV(t)); the s_read wait lives inside it,
    every tile, and nowhere else in the leader."""
    source = _body_source()
    # the slot of each gate is the state's (`_slot(bar, state, geo.DS_STAGES)`: the bar under one stage, bar[state.idx] under the two-stage ring)
    gates = re.findall(
        r"_wait_tcgen\(_slot\(bars\.(ds_smem|ds_recv_full|ds_pair_ready), (\w+), geo\.DS_STAGES\), \2\.phase, dbg, _W\[\"wait:leader:\w+\"\]\)", source
    )
    assert sorted(g[0] for g in gates) == ["ds_pair_ready", "ds_recv_full", "ds_smem"], gates
    assert {g[1] for g in gates} == {"xchg_state"}, gates
    issue = _function_code(source, "_dq_issue", "_dq_slot")
    # the three gates are UNCONDITIONAL (no const_expr guard: they are the ds_recv_full arm's only back-edge); the trace-time branches in the
    # helper are the STAMPS handshake, the test-only skew lever and the dq_war_done commit of the alternate role shape; none guards a gate
    assert [ln.strip() for ln in issue if ln.strip().startswith("if ")] == [
        "if cutlass.const_expr(dbg is not None):",
        "if cutlass.const_expr(b2):",  # the FOURTH gate of smem_map rubin_b2 (dq_read_done) + the per-issue dS stage advance: after the three, never guarding them
        "if cutlass.const_expr(DEBUG_SOFTMAX_DQ_WAR_DELAY_US > 0):",
        "if cutlass.const_expr(fa4_roles):",
    ], issue
    gate_lines = [ln for ln in issue if "_wait_tcgen(_slot(bars.ds_" in ln]
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
    `if hi - 1 > lo`, and the post-loop drain; the lookahead's and the tail's advance sit with their waits.  Each of the four sits under the ONE
    DIAGNOSTIC guard `if cutlass.const_expr(not s_tail_off):` (s_tail_wait_off, header section 11) and its advance stays OUTSIDE the guard (the
    phase flips unobserved under the probe; CFG.DIAG_S_TAIL_WAIT_SITES == 4); the s_read wait of `_dq_slot` sits under `not s_read_off` the same way."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod

    source = _body_source()
    wait_s_tail = r'_wait_tcgen\(bars\.s_tail_empty, s_tail_state\.phase, dbg, _W\["wait:leader:s_tail_empty"\]\)'
    guard = r"if cutlass\.const_expr\(not s_tail_off\):\n\s+"
    waits = [m.start() for m in re.finditer(r"^\s+" + wait_s_tail, source, re.M)]
    assert len(waits) == 4 == cfgmod.CFG.DIAG_S_TAIL_WAIT_SITES, len(waits)
    assert len(re.findall(guard + wait_s_tail, source)) == 4, "every s_tail_empty wait sits under the s_tail_wait_off guard"
    assert re.search(r"if qtile > lo:\n\s+" + guard + wait_s_tail + r"\n\s+s_tail_state = advance\(s_tail_state, 1\)", source)
    assert re.search(r"if hi - 1 > lo:\n\s+" + guard + wait_s_tail + r"\n\s+s_tail_state = advance\(s_tail_state, 1\)", source)
    assert re.search(guard + wait_s_tail + r"\n\s+s_tail_state = advance", source), "the advance follows the guarded wait (dedented, unconditional)"
    s_read_wait = r'_wait_tcgen\(bars\.s_read, ready_state\.phase, dbg, _W\["wait:leader:s_read"\]\)'
    assert len(re.findall(r"if cutlass\.const_expr\(not s_read_off\):\n\s+" + s_read_wait, source)) == 1 == cfgmod.CFG.DIAG_S_READ_WAIT_SITES
    assert len(re.findall(r"^\s+" + s_read_wait, source, re.M)) == 1
    leader = _function_code(source, "_mma_loop_leader", "_mma_loop_follower")
    prologue = [i for i, ln in enumerate(leader) if "_wait_tcgen(bars.s_tail_empty" in ln]
    s_lo = [i for i, ln in enumerate(leader) if "mma_ss(score, k1d, q1d, s_acc)" in ln]
    assert prologue and s_lo and prologue[0] < s_lo[0], "the prologue S^T(lo) waits s_tail_empty (the previous item's tail readout)"
    assert leader[-1].strip() == '_wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])', leader[-1]  # the P15 drain: last
    assert leader[-2].strip() == "if cutlass.const_expr(not s_tail_off):", leader[-2]  # the drain's probe guard (the wait is still the last statement)
    # the guards derive from the threaded code, never a literal; no arrive and no advance sits under a wait-probe guard
    assert any(
        ln.strip().startswith("s_tail_off = cutlass.const_expr(") and "DIAG_S_TAIL_WAIT_OFF" in ln and "DIAG_S_TAIL_AND_S_READ_WAIT_OFF" in ln for ln in leader
    )
    assert any(
        ln.strip().startswith("s_read_off = cutlass.const_expr(") and "DIAG_S_READ_WAIT_OFF" in ln and "DIAG_S_TAIL_AND_S_READ_WAIT_OFF" in ln for ln in leader
    )
    for i, ln in enumerate(leader):
        if ln.strip() in ("if cutlass.const_expr(not s_tail_off):", "if cutlass.const_expr(not s_read_off):"):
            depth = len(ln) - len(ln.lstrip())
            assert "_wait_tcgen(" in leader[i + 1] and len(leader[i + 1]) - len(leader[i + 1].lstrip()) > depth, leader[i : i + 2]
            if i + 2 < len(leader):  # the P15 drain's guard is the role's second-to-last statement
                assert len(leader[i + 2]) - len(leader[i + 2].lstrip()) <= depth, "only the wait sits under a wait-probe guard"
    softmax = _function_code(source, "_softmax_loop", "_epilogue_loop")
    s_read = [i for i, ln in enumerate(softmax) if "_arrive_leader(bars.s_read)" in ln]
    s_done = [i for i, ln in enumerate(softmax) if "_wait_tcgen(bars.s_done" in ln]
    assert len(s_read) == 1 and len(s_done) == 1
    # D2: EVERY tile -- the arrive carries no RUNTIME guard: its only guard is the trace-time map guard (two statements under it: the fence, the
    # arrive), whose indentation is the s_done wait's; the fence + arrive go together under tmem_map dq_576 (the row does not exist there)
    guard_at = s_read[0] - 2
    assert softmax[guard_at].strip() == "if cutlass.const_expr(not dq576):" and "tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)" in softmax[guard_at + 1]
    depth = lambda ln: len(ln) - len(ln.lstrip())  # noqa: E731
    assert depth(softmax[guard_at]) == depth(softmax[s_done[0]]) and depth(softmax[s_read[0]]) == depth(softmax[guard_at + 1]) == depth(softmax[guard_at]) + 4
    assert depth(softmax[s_read[0] + 1]) <= depth(softmax[guard_at]), "only the fence and the arrive sit under the s_tail map's guard"
    # tmem_map dq_576 (the kernel header sections 5 / 9): EVERY s_tail_empty / s_read / s_tail_state statement of the leader sits under
    # `if cutlass.const_expr(not dq576):` (the probe guard may sit between); the state's dq_576 arm is the None the lookahead threads through
    leader_lines = leader

    def enclosing_guards(i):
        guards, indent = [], depth(leader_lines[i])
        for j in range(i - 1, -1, -1):
            if depth(leader_lines[j]) < indent:
                indent = depth(leader_lines[j])
                if leader_lines[j].strip().startswith("if cutlass.const_expr("):
                    guards.append(leader_lines[j].strip())
        return guards

    s_tail_sites = [i for i, ln in enumerate(leader_lines) if any(tok in ln for tok in ("bars.s_tail_empty", "bars.s_read", "s_tail_state"))]
    assert (
        len(s_tail_sites) >= 8
    ), s_tail_sites  # the state pair, the prologue wait + advance, the slot-2 call, the lookahead call, the tail wait + advance, the drain
    for i in s_tail_sites:
        stmt = leader_lines[i].strip()
        if stmt in ("s_tail_state = PipelineState.start(phase=1)  # pre-armed: the kernel's first S^T(lo) finds the S tail free",) or stmt.startswith(
            "s_tail_state = None"
        ):
            continue  # the definition pair: the s_tail arm under `not dq576`, the dq_576 arm (None) under its else
        if "_dq_slot(" in stmt or "_s_lookahead(" in stmt:
            continue  # the two helper calls thread the flags / the state through (the helpers guard their own S-tail sites)
        assert "if cutlass.const_expr(not dq576):" in enclosing_guards(i), (stmt, enclosing_guards(i))
    assert any(
        ln.strip() == "s_tail_state = None  # dq_576: no S-tail row, no state; `_s_lookahead` threads the None through unchanged (no site reads it)"
        for ln in leader_lines
    )
    helper = _function_code(source, "_s_lookahead", "_dq_issue") + _function_code(source, "_dq_slot", "_dq_round")
    for i, ln in enumerate(helper):
        if "bars.s_tail_empty" in ln or "bars.s_read" in ln or "s_tail_state = advance" in ln:
            guards, indent = [], depth(ln)
            for j in range(i - 1, -1, -1):
                if depth(helper[j]) < indent:
                    indent = depth(helper[j])
                    if helper[j].strip().startswith("if cutlass.const_expr("):
                        guards.append(helper[j].strip())
            assert "if cutlass.const_expr(not dq576):" in guards, (ln.strip(), guards)
    assert not any("s_tail_empty" in ln or "bars.s_read" in ln for ln in _function_code(source, "_dq_issue", "_dq_slot")), "the three gates know no S-tail row"


def test_2cta_stail_geometry_keyed_bodies_take_every_depth_and_offset_from_geo():
    """RED-first source pin of the rubin_b2 bring-up HANG (runs/r2c/l0_map: exit 124 after STAGE=kernel on the FIRST rubin_b2 cell; the three
    rubin_b2 skew children killed by the watchdog): a role body keyed by the geometry record (``geo``) takes EVERY ring depth and slab offset from
    ``geo``, never from the module's base ``CFG`` -- the two agree on every sm100 geometry (the byte-identical sm100 renderings cannot see the slip)
    and disagree under rubin_b2 (``CFG.Q1_STAGES`` = 1 against ``geo.Q1_STAGES`` = 2: the leader walked the sQ1 ring 1-deep against the loader's
    2-deep walk -- q1_empty[1] never committed, q1_full[0]'s third completion never armed: both CTAs spin).  The differing-field set is computed
    from the config module (config as truth); NO ``CFG.<field>`` of that set may appear in a role body -- the arms rubin_b2 declines by name (stats_form
    ldg's `_stats` slab, the 16w_fa4 softmax arm and reduce body) read the record too (the record IS the base config there: byte-identical renderings);
    comments and docstrings are prose."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfg_mod

    base, b2 = cfg_mod.CFG, cfg_mod.CFG_RING_DQ576_B2
    differing = sorted(n for n in dir(base) if not n.startswith("_") and not callable(getattr(base, n)) and getattr(base, n) != getattr(b2, n))
    assert {"Q1_STAGES", "DO1_STAGES", "DS_STAGES", "XCHG_STAGES", "DS", "XCHG", "DQ", "SMEM_BYTES", "BARS"} <= set(differing), differing
    pattern = re.compile(r"\bCFG\.(" + "|".join(differing) + r")\b")
    bodies = ("_softmax_loop", "_epilogue_loop", "_mma_loop_leader", "_mma_loop_follower", "_reduce_loop", "_load_loop", "_store_loop")
    nexts = ("_epilogue_loop", "_mma_loop_leader", "_mma_loop_follower", "_reduce_loop", "_load_loop", "_store_loop", "_scheduler_loop")
    src = _body_source()
    offenders = []
    for name, nxt in zip(bodies, nexts):
        in_doc = False
        for ln in _function_code(src, name, nxt):
            quotes = ln.count('"""')
            if in_doc:
                in_doc = quotes % 2 == 0
                continue
            if quotes % 2 == 1:
                in_doc = True
                continue
            if quotes >= 2 and ln.strip().startswith('"""'):
                continue  # a one-line docstring
            code = ln.split("#", 1)[0]
            m = pattern.search(code)
            if m:
                offenders.append((name, code.strip(), m.group(1)))
    assert offenders == [], offenders


def _const_expr_arm(code, keep_first):
    """The epilogue's statement lines as ONE SMEM map traces them: every `if cutlass.const_expr(not b2):` keeps its first arm (``keep_first``: the sm100
    maps) or its `else:` arm (the rubin_b2 map), dedented; the kernel header section 5 is one such view per map."""
    depth = lambda ln: len(ln) - len(ln.lstrip())  # noqa: E731
    out, i = [], 0
    while i < len(code):
        ln = code[i]
        if ln.strip() == "if cutlass.const_expr(not b2):":
            d = depth(ln)
            i += 1
            first = []
            while i < len(code) and depth(code[i]) > d:
                first.append(code[i][4:])
                i += 1
            assert i < len(code) and code[i].strip() == "else:" and depth(code[i]) == d, (ln, code[i] if i < len(code) else None)
            i += 1
            second = []
            while i < len(code) and depth(code[i]) > d:
                second.append(code[i][4:])
                i += 1
            out.extend(first if keep_first else second)
            continue
        out.append(ln)
        i += 1
    return out


def _sm100_arm(code):
    """The epilogue as the sm100 SMEM map traces it (the audited order: the readout of dQ(t) BEFORE the staging of dS(t+1))."""
    return _const_expr_arm(code, keep_first=True)


def _b2_arm(code):
    """The epilogue as the rubin_b2 SMEM map traces it (the LAGGING order: the staging of dS(t) BEFORE the readout of dQ(t-1), the dq_read_done arrives)."""
    return _const_expr_arm(code, keep_first=False)


def _assert_epilogue_readout_precedes_next_staging(code):
    """The static detector of the independent dQ image's REUSE edge on the sm100 SMEM map (tmem_map dq_576; the kernel header section 5, the barrier
    skeleton R1 1.3.1): in the epilogue's per-tile body the own-half STS + ``smem_bar.arrive()`` (the gate of the leader's dQ issue) come FIRST and the
    dq_done wait + the dQ readout trio (the LDTM at the map-keyed column base, tcgen05_wait(LOAD), tcgen05_fence(BEFORE_THREAD_SYNC)) come AFTER them at
    the loop body's depth -- so the readout of dQ(t) precedes the staging (hence the gate) of dQ(t+1) in the SAME warps' program order; the loaded
    registers are consumed by the rounds of the same iteration (no lagging readout); no second staging follows the readout inside the body.  ``code``
    = the statement lines of ``_epilogue_loop`` as the sm100 map traces them (``_sm100_arm(_function_code(..))``) -- or a deliberately reordered copy
    (the RED half of the test)."""
    idx = lambda needle: [i for i, ln in enumerate(code) if needle in ln]  # noqa: E731
    depth = lambda i: len(code[i]) - len(code[i].lstrip())  # noqa: E731
    sts, arrive, ld = idx("_store_words(slab, own_sts, lane,"), idx("smem_bar.arrive()"), idx("dq = _tmem_load(tmem_base + dq_col, CFG.DQ_COLS)")
    assert len(sts) == 1 and len(arrive) == 1 and len(ld) == 1, (sts, arrive, ld)
    waits = [i for i in idx("_wait_tcgen(bars.dq_done, dq_state.phase") if i < ld[0]]
    assert len(waits) == 1, waits  # the in-loop dq_done(t) wait (the zero-trip observation sits after the loop)
    load_wait = next(i for i in idx("nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)") if i > ld[0])
    fence = next(i for i in idx("nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)") if i > load_wait)
    for_at = idx("for qtile in cutlass.range(lo, hi, 1, unroll=1):")[0]
    end_at = next(i for i in idx("if lo < hi:") if i > for_at)  # the item end
    assert for_at < sts[0] < arrive[0] < waits[0] < ld[0] < load_wait < fence < end_at, (for_at, sts, arrive, waits, ld, load_wait, fence, end_at)
    assert depth(arrive[0]) == depth(waits[0]) == depth(ld[0]) == depth(load_wait) == depth(fence) == depth(for_at) + 4, "the readout carries no runtime guard"
    assert not any("_store_words(slab, own_sts" in code[i] or "smem_bar.arrive()" in code[i] for i in range(fence + 1, end_at)), "no staging after the readout"
    round_a = [i for i in idx("_dq_half(dq, 0)") if fence < i < end_at]
    assert len(round_a) == 1, "the loaded registers feed this iteration's round A (no lagging readout)"


def test_2cta_stail_dq576_reuse_edge_rides_the_epilogue_program_order():
    """RED-first SOURCE assertion of the independent dQ image's reuse edge on the sm100 SMEM map (tmem_map dq_576 x smem_map sm100; the kernel header
    section 5, P14-B): under that map no barrier row frees the dQ image -- dQ(t+1)'s write is gated by the dS gates of tile t+1 (ds_smem /
    ds_pair_ready), which the epilogues arrive AFTER reading dQ(t) out ONLY because the dq_done(t) wait + the readout trio sit BEFORE the next tile's
    own-half STS + ``smem_bar.arrive()`` in the loop body.  RED: a deliberately reordered copy of the epilogue (the readout block moved ahead of the
    staging = the lagging-readout form the two-stage ring takes, which under rubin_b2 is why the dq_read_done row exists; the gate arrive moved
    behind the readout) FAILS the detector; GREEN: the sm100 arm of the body passes.  The detector reads the statements only."""
    code = _sm100_arm(_function_code(_body_source(), "_epilogue_loop", "_mma_loop_leader"))
    _assert_epilogue_readout_precedes_next_staging(code)  # GREEN
    idx = lambda needle, lines: [i for i, ln in enumerate(lines) if needle in ln]  # noqa: E731
    wait = [i for i in idx("_wait_tcgen(bars.dq_done, dq_state.phase", code) if i < idx("dq = _tmem_load(tmem_base + dq_col", code)[0]][0]
    load_wait = next(i for i in idx("nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)", code) if i > wait)
    fence = next(i for i in idx("nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)", code) if i > load_wait)
    block = code[wait : fence + 1]  # the dq_done wait, the skew lever, the LDTM, wait(LOAD), the fence
    ds_tmem_wait = idx("_wait_tcgen(bars.ds_tmem, ds_state.phase", code)[0]
    # RED 1: the readout block ahead of the dS staging (reads dQ(t-1) out while dS(t) is staged after it: the lagging-readout form)
    red1 = code[:ds_tmem_wait] + block + code[ds_tmem_wait:wait] + code[fence + 1 :]
    with pytest.raises(AssertionError):
        _assert_epilogue_readout_precedes_next_staging(red1)
    # RED 2: the gate arrive moved behind the readout (ds_smem(t) would publish the STS before the dQ(t) readout -> dQ(t+1) may overwrite an unread image)
    arrive = idx("smem_bar.arrive()", code)[0]
    red2 = code[:arrive] + code[arrive + 1 : fence + 1] + [code[arrive]] + code[fence + 1 :]
    with pytest.raises(AssertionError):
        _assert_epilogue_readout_precedes_next_staging(red2)
    # RED 3: the readout under a runtime guard (a conditional readout = a tile whose image is never read: the next write finds it unread)
    red3 = list(code)
    for i in range(wait, fence + 1):
        red3[i] = "    " + red3[i]
    red3.insert(wait, " " * (len(code[wait]) - len(code[wait].lstrip())) + "if qtile > lo:")
    with pytest.raises(AssertionError):
        _assert_epilogue_readout_precedes_next_staging(red3)
    # RED 4: the rubin_b2 arm itself (the lagging order) FAILS this detector -- that is why the row dq_read_done exists under that map (the next test)
    with pytest.raises(AssertionError):
        _assert_epilogue_readout_precedes_next_staging(_b2_arm(_function_code(_body_source(), "_epilogue_loop", "_mma_loop_leader")))


_READOUT_TRIO = (
    "dq = _tmem_load(tmem_base + dq_col, CFG.DQ_COLS)",
    "nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)",
    "nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)",
)
_DQ_READ_DONE_WAIT = '_wait_tcgen(bars.dq_read_done, dq_read_state.phase, dbg, _W["wait:leader:dq_read_done"])'


def _assert_dq_read_done_row(epi_b2, issue, leader):
    """The static detector of the two-stage ring's reuse edge (smem_map rubin_b2 ON dq_576; the kernel header section 5, the barrier skeleton R2.c 2.5
    b / 1.7): the epilogue's rubin_b2 arm arrives ``dq_read_done`` bare at EXACTLY two sites, each right after the readout trio, the first in the loop
    body AFTER the staging's ``smem_bar.arrive()`` under the item's ``qtile > lo`` guard, the second at the item end under ``lo < hi`` (both the
    loop-structure predicate of _item_bounds, pair-uniform); `_dq_issue` waits the row (``_wait_tcgen``: the after-thread-sync fence) on
    ``dq_read_state`` after the three dS gates and before the dQ MMA, under its b2 guard, and advances the state right after; the leader pre-arms the
    state (st(1)) under the b2 guard and drains the row exactly once after its persistent loop (P15).  ``epi_b2`` / ``issue`` / ``leader`` = the
    statement lines of the three functions (the epilogue's as the rubin_b2 map traces them) -- or deliberately altered copies (the RED half)."""
    idx = lambda needle, lines: [i for i, ln in enumerate(lines) if needle in ln]  # noqa: E731
    depth = lambda lines, i: len(lines[i]) - len(lines[i].lstrip())  # noqa: E731

    def guards(lines, i):
        d, out = depth(lines, i), []
        for j in range(i - 1, -1, -1):
            if depth(lines, j) < d and lines[j].rstrip().endswith(":"):
                out.append(lines[j].strip())
                d = depth(lines, j)
        return out

    arrives = idx("_arrive_leader(bars.dq_read_done)", epi_b2)
    assert len(arrives) == 2, arrives
    for a in arrives:
        trio = tuple(ln.strip().split("  #")[0] for ln in epi_b2[a - 3 : a])
        assert trio == _READOUT_TRIO, trio  # the arrive right after the readout trio (the local half of the edge: G03 lifecycle rule 6)
    for_at = idx("for qtile in cutlass.range(lo, hi, 1, unroll=1):", epi_b2)[0]
    staging = idx("smem_bar.arrive()", epi_b2)
    assert len(staging) == 1 and for_at < staging[0] < arrives[0], (for_at, staging, arrives)  # the lagging order: staged, THEN read out
    end_at = next(i for i in idx("if lo < hi:", epi_b2) if i > for_at)
    assert arrives[0] < end_at < arrives[1], (arrives, end_at)  # one in-loop site, one item-end site
    assert guards(epi_b2, arrives[0])[:2] == ["if qtile > lo:", "for qtile in cutlass.range(lo, hi, 1, unroll=1):"], guards(epi_b2, arrives[0])
    assert guards(epi_b2, arrives[1])[0] == "if lo < hi:", guards(epi_b2, arrives[1])
    waits = idx(_DQ_READ_DONE_WAIT, issue)
    mma = idx("mma_ss(dqdesc, dsd_s, k2d, dq_acc)", issue)
    gates = idx("_slot(bars.ds_pair_ready, xchg_state, geo.DS_STAGES)", issue)
    assert len(waits) == 1 and len(mma) == 1 and len(gates) == 1 and gates[0] < waits[0] < mma[0], (gates, waits, mma)
    assert issue[waits[0] - 1].strip() == "if cutlass.const_expr(b2):" and issue[waits[0] + 1].strip() == "dq_read_state = advance(dq_read_state, 1)", issue[
        waits[0] - 1 : waits[0] + 2
    ]
    pre_arm = idx("dq_read_state = PipelineState.start(phase=1)", leader)
    assert len(pre_arm) == 1 and leader[pre_arm[0] - 1].strip() == "if cutlass.const_expr(b2):", pre_arm
    drains = idx(_DQ_READ_DONE_WAIT, leader)
    loop_end = max(idx("_next_tile(sched, bars, sched_state, dbg,", leader))
    assert len(drains) == 1 and drains[0] > loop_end and drains[0] > idx("_wait_tcgen(bars.dvk_empty, out_empty.phase", leader)[-1], (drains, loop_end)
    assert leader[drains[0] - 1].strip() == "if cutlass.const_expr(b2):", leader[drains[0] - 1]


def test_2cta_stail_rubin_b2_reuse_edge_is_the_dq_read_done_row():
    """RED-first SOURCE assertion of the two-stage ring's reuse edge (smem_map rubin_b2 ON dq_576; the kernel header section 5, the barrier skeleton
    R2.c 2.5 b / 1.7): under the LAGGING order the epilogue arrives ds_smem(t) BEFORE it reads dQ(t-1) out, so the dq_576 map's program-order carrier
    is void and the LEADER row ``dq_read_done`` (PAIR_EPILOGUE_LANES 256, after dp_empty; in the B2 table only) carries the edge -- the epilogue
    arrives it after each readout trio (two sites), `_dq_issue` waits it before every dQ issue on the pre-armed state, the leader drains it once.
    RED: the arrive removed, the arrive moved ahead of its trio, the wait removed, or the drain removed FAILS the detector; GREEN: the body passes."""
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod

    c = cfgmod.CFG
    assert cfgmod.DQ_READ_DONE_ROW == ("dq_read_done", 1, c.PAIR_EPILOGUE_LANES, "leader", "LEADER") and cfgmod.ADDED_BARRIERS_B2 == ("dq_read_done",)
    names_b2 = [r[0] for r in cfgmod.BAR_LAYOUT_RING_DQ576_B2]
    assert names_b2.index("dq_read_done") == names_b2.index("dp_empty") + 1
    for table in (cfgmod.BAR_LAYOUT, cfgmod.BAR_LAYOUT_RING, cfgmod.BAR_LAYOUT_DQ576, cfgmod.BAR_LAYOUT_RING_DQ576, cfgmod.BAR_LAYOUT_XFA4):
        assert "dq_read_done" not in {r[0] for r in table}  # the row exists under the two-stage ring only
    src = _body_source()
    epi = _b2_arm(_function_code(src, "_epilogue_loop", "_mma_loop_leader"))
    issue = _function_code(src, "_dq_issue", "_dq_slot")
    leader = _function_code(src, "_mma_loop_leader", "_mma_loop_follower")
    _assert_dq_read_done_row(epi, issue, leader)  # GREEN
    idx = lambda needle, lines: [i for i, ln in enumerate(lines) if needle in ln]  # noqa: E731
    arrives = idx("_arrive_leader(bars.dq_read_done)", epi)
    # RED 1: the in-loop arrive removed (the leader's next issue would wait a flip nobody produces: a hang, or with the wait gone too, an unread image overwritten)
    red1 = epi[: arrives[0]] + epi[arrives[0] + 1 :]
    with pytest.raises(AssertionError):
        _assert_dq_read_done_row(red1, issue, leader)
    # RED 2: the arrive ahead of its readout trio (the image published as read out before the LDTM completed: the next dQ may overwrite it under the read)
    red2 = list(epi)
    red2[arrives[0] - 3], red2[arrives[0]] = red2[arrives[0]], red2[arrives[0] - 3]
    with pytest.raises(AssertionError):
        _assert_dq_read_done_row(red2, issue, leader)
    # RED 3: the wait removed from _dq_issue (the gate gone: the overwrite races the readout)
    waits = idx(_DQ_READ_DONE_WAIT, issue)
    red3 = issue[: waits[0]] + issue[waits[0] + 1 :]
    with pytest.raises(AssertionError):
        _assert_dq_read_done_row(epi, red3, leader)
    # RED 4: the drain removed (the follower's last relaxed arrive would land on an exited leader: P15)
    drains = idx(_DQ_READ_DONE_WAIT, leader)
    red4 = leader[: drains[0]] + leader[drains[0] + 1 :]
    with pytest.raises(AssertionError):
        _assert_dq_read_done_row(epi, issue, red4)


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
    assert sum("_dq_round(" in ln for ln in _sm100_arm(epi)) == 3  # round B of dQ(t-1), round A of dQ(t), the item-end round B of dQ(hi-1)
    # smem_map rubin_b2 (the lagging order): round B of dQ(t-2), round A of dQ(t-1), the item end's round B of dQ(hi-2) + rounds A, B of dQ(hi-1)
    assert sum("_dq_round(" in ln for ln in _b2_arm(epi)) == 5
    assert not any("empty_state = advance" in ln for ln in epi), "no item-end flip: 2N rounds == 2N releases per slot"


def test_2cta_stail_exchange_off_keeps_every_wait_and_fakes_the_bytes():
    """SOURCE assertion for `exchange_off` (the safe exchange ceiling): in the epilogue the ONE ds_recv_full arm keeps its elected lane and arms
    the config-selected byte count (CFG.DIAG_EXCHANGE_OFF_TX = 0 under the probe, XCHG_TX otherwise); the ONLY statement under the probe's guard
    is the `_ship_xchg` call (the transfer); the XCHG STS, its proxy fence, the CFG.XCHG_BARRIER named barrier, the own-half STS and every wait /
    arrive of the epilogue are unguarded; the three dS gates (the arm's back-edge) are unconditional in `_dq_issue` (asserted with RED 4 above);
    the relay in the follower is unconditional."""
    src = _body_source()
    epi = _function_code(src, "_epilogue_loop", "_mma_loop_leader")
    arms = [ln for ln in epi if "recv_bar.arrive(" in ln]  # the ds_recv_full arm: the bar (sm100) or the slot of the stage (rubin_b2), ONE site
    assert arms == [ln for ln in arms if "n_bytes=xchg_tx, pred=ship_pred" in ln] and len(arms) == 1, arms
    assert any(
        ln.strip() == "xchg_tx = CFG.DIAG_EXCHANGE_OFF_TX if cutlass.const_expr(diag_2cta_code == DIAG_EXCHANGE_OFF) else CFG.XCHG_TX" for ln in epi
    ), "the arm's byte count is the config-selected constant (0 under exchange_off)"
    guard = [i for i, ln in enumerate(epi) if ln.split("#")[0].strip() == "if cutlass.const_expr(not exchange_off):"]
    assert len(guard) == 1 and "_ship_xchg(slab, bars, rank, ship_pred, stage, geo, b2)" in epi[guard[0] + 1], epi[guard[0] : guard[0] + 2]
    depth = len(epi[guard[0]]) - len(epi[guard[0]].lstrip())
    assert len(epi[guard[0] + 2]) - len(epi[guard[0] + 2].lstrip()) <= depth, "only the ship sits under the exchange_off guard"
    for token in (
        "_store_words(slab, xchg_sts",
        "named_barrier_fence(CFG.XCHG_BARRIER",
        "_store_words(slab, own_sts",
        "smem_bar.arrive()",
        "_arrive_leader(bars.dp_empty)",
    ):
        lines = [ln for ln in epi if token in ln]
        assert len(lines) == 1 and len(lines[0]) - len(lines[0].lstrip()) == depth, (token, lines)  # at the loop body's depth: never guarded
    assert not any("exchange_off" in ln and ("_wait" in ln or ".arrive(" in ln) for ln in epi), "exchange_off guards no wait and no arrive"
    relay = _function_code(src, "_mma_loop_follower", "_reduce_loop")
    assert (
        sum('_wait(_slot(bars.ds_recv_full, relay_state, geo.DS_STAGES), relay_state.phase, dbg, _W["wait:follower:ds_recv_full"])' in ln for ln in relay) == 1
    )
    assert not any("diag" in ln.lower() for ln in relay), "no probe reaches the relay"


def test_2cta_stail_skew_levers_follow_the_house_form():
    """The body exposes the four base levers plus SOFTMAX_S_READ / EPILOGUE_DQ_READOUT / SOFTMAX_DQ_WAR and the rubin_b2 ring's LEADER_ARM_DELAY /
    LEADER_SLOT1_DELAY, each in the house form (module global + const_expr nanosleep loop, default 0 = no traced code: the default render's census
    pins nanosleep == 0)."""
    source = _body_source()
    exposed = set(LEVER_LINE.findall(source))
    known = {row["lever"] for row in SITES.values()}
    assert exposed == known, (sorted(exposed), sorted(known))
    for lever in sorted(exposed):
        name = f"DEBUG_{lever}_DELAY_US"
        assert re.search(rf"cutlass\.const_expr\({name} > 0\)", source), name
        assert re.search(rf"cutlass\.range\({name}, unroll=1\)", source), name
        assert source.count(f'globals().get("{GLOBAL_PREFIX}{lever}_DELAY_US", 0)') == 1, name
        assert f"{GLOBAL_PREFIX}{lever}" not in os.environ, name
    assert source.count('nvvm.inline_ptx("nanosleep.u32 1000;"') == len(exposed)


# ------------------------------------------------------------------------------------------------------------ GPU (the fp64 gate)
def test_2cta_stail_route_accepts_dense_and_declines_mha_typed():
    """RED-first (the pair body served grouped THD only; a dense declaration declined "requires THD"): the direct entry (``cta_pair = "on"``) ACCEPTS
    a dense compact-BSHD BF16 GQA declaration -- the resolution names the pair body, the plan the dense implementation token, the workspace is the
    THD chain's at the packed-view capacities B*S -- and an MHA request still declines BY NAME before any compile."""
    _require_2cta()
    from cudnn.sdpa.bwd import d128_grouped_2cta_contract as c

    dense = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=False)
    api = make_adapter_2cta_stail(dense, compile_plan=False)
    assert api.check_support() is True and api._cta_pair.body == c.PAIR_BODY and api._implementation == c.DENSE, api._cta_pair
    assert api.scratch_workspace_bytes() == c.thd_workspace_layout(2, 4, 2 * 256, 2 * 256, True)[1]
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
    assert (body.CFG_DQ576.DQ_COL, body.CFG_DQ576.TMEM_COLS, body.CFG_DQ576.TMEM_IS_EXCLUSIVE) == (512, 576, True)  # the cc 10.7 map (header section 9)
    assert (body.CFG_RING_DQ576_B2.SMEM_BYTES, body.CFG_RING_DQ576_B2.DS_STAGES, body.CFG_RING_DQ576_B2.SMEM_CAP) == (314368, 2, 334848)  # header section 8
    case = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True)
    api, workspace = _run_case(case, repeats=1)
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import SMEM_MAP_DEFAULT_BY_PART, TMEM_MAP_DEFAULT_BY_PART
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import smem_map, tmem_map

    assert tmem_map(api) == TMEM_MAP_DEFAULT_BY_PART[torch.cuda.get_device_capability()]  # the map this device rendered = its part's default
    assert smem_map(api) == SMEM_MAP_DEFAULT_BY_PART[torch.cuda.get_device_capability()] == "sm100"  # the SMEM map unset: the opt-in chain everywhere
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


_STATS_TWIN_CASES = [
    pytest.param(dict(lq=(256, 256), lk=(256, 256), hq=4, hkv=2, causal=True), id="uniform-256-causal"),
    pytest.param(dict(lq=(1024,), lk=(1024,), hq=8, hkv=1, causal=True, stats_layout="HN"), id="uniform-1024-gqa8-causal-HN"),
    pytest.param(dict(lq=(384, 384), lk=(384, 384), hq=4, hkv=2, causal=False), id="uniform-384-noncausal"),  # S % 128 = 0, S % 256 != 0
    pytest.param(dict(lq=(256, 512, 768, 1024), lk=(256, 512, 768, 1024), hq=4, hkv=2, causal=True), id="ramp-1024-causal"),  # dead pairs = zero-trip items
    pytest.param(dict(lq=(129, 0, 127, 1), lk=(65, 128, 0, 3), hq=4, hkv=2, causal=True), id="suite-degenerate-causal"),  # len % 128 == 1, empty, 1 row
    pytest.param(dict(lq=(129, 0, 127, 1), lk=(65, 128, 0, 3), hq=6, hkv=2, causal=False, stats_layout="HN"), id="suite-degenerate-noncausal-HN"),
    pytest.param(dict(lq=(65, 0, 3), lk=(17, 129, 0), hq=6, hkv=2, causal=True, stats_layout="HN"), id="short-pair-half-dead-HN"),
    pytest.param(dict(lq=(300, 300), lk=(300, 300), hq=3, hkv=1, causal=True), id="pair-tail-300"),  # S % 128 = 44: a partial last q tile, pad rows
    pytest.param(dict(lq=(0, 0), lk=(0, 0), hq=3, hkv=1, causal=True, stats_layout="HN", envelope=(128, 128)), id="all-empty"),
    pytest.param(dict(lq=(300, 128, 1000), lk=(2000, 700, 1000), hq=4, hkv=2, causal=True), id="zero-trip-mix"),  # N = 3 / 1 / 0 items; 8 pairs
    pytest.param(dict(lq=(128, 64), lk=("skv", "skv - 256"), hq=8, hkv=1, causal=True), id="n_q-1-pairs-gt-sms"),  # n_q == 1, two items per CTA
]


@pytest.mark.parametrize("spec", _STATS_TWIN_CASES)
def test_2cta_stail_stats_forms_agree_bitwise(spec):
    """Lever S1's twin (G09: a claim BY CONSTRUCTION -- the tma_ring stat words are the SAME two fp32 multiplies on the same operands, performed
    once per q row in the delta launch instead of per (row, pair) in the softmax; a pad row's 0.0 is the ldg form's `stat = 0`): dK / dV BITWISE
    between stats_form = ldg and tma_ring on every row, dQ BITWISE where the sequences have at most _DQ_ORDER_FREE_PAIRS kv pairs, by the fp64 gate
    beyond; each form passes the suite's fp64 gate first.  The rows: the 8-cell THD matrix (incl. S % 256 != 0 at 384, S % 128 != 0 at 300 = pad rows
    in the padded stats region, a sequence of 129 rows = len % 128 == 1, empty sequences, a 1-row sequence, dead pairs = zero-trip items) in BOTH Stats
    layouts (token-major TH1 and head-major HN: the delta launch reads either), the all-empty THD case, the zero-trip mix, and n_q == 1 with more
    pairs than SMs (two items per CTA: the k-indexed ring across items, never a per-item stage reset)."""
    _require_2cta()
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG

    spec = dict(spec)
    lq, lk, causal = spec.pop("lq"), spec.pop("lk"), spec.pop("causal")
    if lk and lk[0] == "skv":
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        skv = 256 * ((sms // (spec["hq"] * 2)) + 2) + 128  # pairs x hq x b > sms, the last pair half-dead (the P14 row's cell)
        lk = (skv, skv - 256)
    case = make_case(lq, lk, thd=True, dtype=torch.bfloat16, causal=causal, stats_layout=spec.pop("stats_layout", "TH1"), **spec)
    dq_bitwise = max(lk) <= _DQ_ORDER_FREE_PAIRS * CFG.PAIR_KV_ROWS
    grads = {}
    for form in _STATS_FORMS:
        api = make_adapter_2cta_stail(case, stats_form=form)
        workspace = make_workspace(api.scratch_workspace_bytes())
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
        grads[form] = {role: case.tensors[role][0].detach().clone() for role in ("dq", "dk", "dv")}
    for role in ("dk", "dv") + (("dq",) if dq_bitwise else ()):
        assert torch.equal(grads["ldg"][role].view(torch.int16), grads["tma_ring"][role].view(torch.int16)), f"{role}: ldg vs tma_ring differ bitwise"


def test_2cta_stail_tma_ring_launch_count_and_stats_launch():
    """Under stats_form = tma_ring the chain is still five launches: the nine-map setup (now also writing the padded row offsets), the pre-scaling
    padded delta launch ``delta_zero_stats_thd_d128_2cta_kernel`` (replacing ``delta_zero_thd_d128_kernel``), the pair MAIN kernel, the cast, the fold;
    the workspace carries the padded stats region instead of the compact delta (the contract's layout) and is larger than the ldg form's."""
    _require_2cta()
    case = make_case((256, 300), (256, 300), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True)
    plain = make_adapter_2cta_stail(case, stats_form="ldg")
    api, workspace = _run_case(case, repeats=1, stats_form="tma_ring")
    assert api.scratch_workspace_bytes() > plain.scratch_workspace_bytes(), "the padded pre-scaled stats region is larger than the compact delta"
    names = []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            names.append(event.name)
    kernels = [n for n in names if n.startswith("cudnn_kernel_")]
    assert len(kernels) == 5, kernels
    assert any("delta_zero_stats_thd_d128_2cta_kernel" in n for n in kernels) and not any("delta_zero_thd_d128_kernel" in n for n in kernels), kernels
    assert any("bprop_d128_accum_grouped_2cta_stail_kernel" in n for n in kernels) and any("setup_thd_d128_2cta_kernel" in n for n in kernels), kernels


_TMEM_MAP_TWIN_CASES = [
    pytest.param(dict(lq=(256, 256), lk=(256, 256), hq=4, hkv=2, causal=True), id="uniform-256-causal"),
    pytest.param(dict(lq=(256, 512, 768, 1024), lk=(256, 512, 768, 1024), hq=4, hkv=2, causal=True), id="ramp-1024-causal"),  # N up to 8, dead pairs
    pytest.param(dict(lq=(300, 300), lk=(300, 300), hq=3, hkv=1, causal=True), id="pair-tail-300"),  # a partial last q tile, a half-dead second CTA
    pytest.param(dict(lq=(300, 128, 1000), lk=(2000, 700, 1000), hq=4, hkv=2, causal=True), id="zero-trip-mix"),  # N = 3 / 1 / 0 items: the N-walk
    pytest.param(dict(lq=(0, 0), lk=(0, 0), hq=3, hkv=1, causal=True, stats_layout="HN", envelope=(128, 128)), id="all-empty"),
    pytest.param(dict(lq=(128, 64), lk=("skv", "skv - 256"), hq=8, hkv=1, causal=True), id="n_q-1-pairs-gt-sms"),  # n_q == 1: the tail dQ(lo) alone
]


def _require_tmem_map(name):
    """The GPU rows of a TMEM map run on the parts that SERVE it (``TMEM_MAP_PARTS``: the library's own set, never a copy) and skip elsewhere."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import TMEM_MAP_PARTS

    _require_2cta()
    cc = torch.cuda.get_device_capability()
    if cc not in TMEM_MAP_PARTS[name]:
        pytest.skip(f"tmem_map {name!r} is served on cc {TMEM_MAP_PARTS[name]}; got {cc}")


@pytest.mark.parametrize("spec", _TMEM_MAP_TWIN_CASES)
def test_2cta_stail_tmem_maps_agree_bitwise(spec):
    """The TMEM map's twin (G09: a claim BY CONSTRUCTION -- the map moves the dQ accumulator's columns and deletes two barrier rows; no GEMM, k-order
    or rounding changes): on a part that serves both maps (cc 10.7) dK / dV are BITWISE between ``tmem_map = "s_tail"`` and ``"dq_576"`` on every
    row, dQ BITWISE where the sequences have at most _DQ_ORDER_FREE_PAIRS kv pairs and by the fp64 gate beyond; each map passes the suite's fp64 gate
    first.  The rows: the uniform cell, the ramp with dead pairs (N up to 8), the partial last q tile with a half-dead CTA, the zero-trip mix (N = 3 /
    1 / 0: the independent image's N-walk), the all-empty case, and n_q == 1 with more pairs than SMs (the tail dQ(lo) alone, two items per CTA)."""
    _require_tmem_map("dq_576")
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG

    spec = dict(spec)
    lq, lk, causal = spec.pop("lq"), spec.pop("lk"), spec.pop("causal")
    if lk and lk[0] == "skv":
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        skv = 256 * ((sms // (spec["hq"] * 2)) + 2) + 128  # pairs x hq x b > sms, the last pair half-dead (the P14 row's cell)
        lk = (skv, skv - 256)
    case = make_case(lq, lk, thd=True, dtype=torch.bfloat16, causal=causal, stats_layout=spec.pop("stats_layout", "TH1"), **spec)
    dq_bitwise = max(lk) <= _DQ_ORDER_FREE_PAIRS * CFG.PAIR_KV_ROWS
    grads = {}
    for name in _TMEM_MAPS:
        api = make_adapter_2cta_stail(case, tmem_map=name)
        workspace = make_workspace(api.scratch_workspace_bytes())
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
        grads[name] = {role: case.tensors[role][0].detach().clone() for role in ("dq", "dk", "dv")}
    for role in ("dk", "dv") + (("dq",) if dq_bitwise else ()):
        assert torch.equal(grads["s_tail"][role].view(torch.int16), grads["dq_576"][role].view(torch.int16)), f"{role}: s_tail vs dq_576 differ bitwise"


def test_2cta_stail_tmem_maps_agree_under_fp32_dkv_accum():
    """``dkv_accum = "fp32_reduce"`` x the TMEM map (the barrier AUDIT's u1): under dq_576 the fp32 staging row ``dvk_stage_empty`` moves from slot 34
    to 32 (``_dvk_stage_empty_row(BarsDQ576)``: the two deleted rows sat ahead of it) and the body's init loop guards that slot BY INDEX -- a wrong
    index leaves ``dvk_stage_empty`` or the slot it displaces uninitialised: a hang or silent wrong dK / dV that no s_tail cell and no bf16-partials
    dq_576 cell can see.  On the part that serves the map (cc 10.7; skips by name elsewhere) one THD GQA cell (two sequences, up to two kv pairs,
    causal: the item-end fp32 rounds across items) runs under fp32_reduce on EACH map through the suite's fp64 gate (two executes on a poisoned
    workspace, the fp32 form pinned on the artifact), and dK / dV are BITWISE between the maps (the same fp32 reduce-adds -- Hq / Hkv = 2 per element
    into a zero-filled image, order-free -- the map changes no arithmetic: a claim by construction); dQ bitwise too at <= _DQ_ORDER_FREE_PAIRS kv pairs."""
    _require_tmem_map("dq_576")
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG

    hq, hkv = 4, 2
    case = make_case((256, 512), (256, 512), hq=hq, hkv=hkv, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    assert hq // hkv <= 2 and max(case.lens_kv) <= _DQ_ORDER_FREE_PAIRS * CFG.PAIR_KV_ROWS  # the bitwise claims below are by construction: name the cell
    grads = {}
    for name in _TMEM_MAPS:
        api, _ = _run_case(case, dkv_accum="fp32_reduce", tmem_map=name)
        assert api._prepared.artifact.dkv_accum == "fp32_reduce"
        grads[name] = {role: case.tensors[role][0].detach().clone() for role in ("dq", "dk", "dv")}
    for role in ("dk", "dv", "dq"):
        assert torch.equal(
            grads["s_tail"][role].view(torch.int16), grads["dq_576"][role].view(torch.int16)
        ), f"{role}: fp32_reduce x s_tail vs x dq_576 differ bitwise"


def test_2cta_stail_dq576_declines_by_name_on_a_512_column_part():
    """``tmem_map = "dq_576"`` NAMED on a part without the 576-column carve (cc 10.0 / 10.3) is the engine's typed decline at ``check_support``
    (NotImplementedError naming the field, the map and the part) -- never a compile, never a silent s_tail; on cc 10.7 the row is the twin above."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import TMEM_MAP_PARTS

    _require_2cta()
    cc = torch.cuda.get_device_capability()
    if cc in TMEM_MAP_PARTS["dq_576"]:
        pytest.skip(f"cc {cc} serves tmem_map 'dq_576' (the bitwise twin covers it)")
    case = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True)
    with pytest.raises(NotImplementedError, match="2-CTA datapath tmem_map 'dq_576' is not served on cc"):
        make_adapter_2cta_stail(case, compile_plan=False, tmem_map="dq_576").check_support()
    assert make_adapter_2cta_stail(case, compile_plan=False, tmem_map="s_tail").check_support() is True


def _require_smem_map(name):
    """The GPU rows of an SMEM map run on the parts that SERVE it (``SMEM_MAP_PARTS``: the library's own set, never a copy) and skip elsewhere."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import SMEM_MAP_PARTS

    _require_2cta()
    cc = torch.cuda.get_device_capability()
    if cc not in SMEM_MAP_PARTS[name]:
        pytest.skip(f"smem_map {name!r} is served on cc {SMEM_MAP_PARTS[name]}; got {cc}")


@pytest.mark.parametrize("spec", _TMEM_MAP_TWIN_CASES)
def test_2cta_stail_smem_maps_agree_bitwise(spec):
    """The SMEM map's twin (G09: a claim BY CONSTRUCTION -- the map re-stages the same bytes through deeper rings and re-orders the epilogue's waits;
    no GEMM, k-order or rounding changes): on a part that serves rubin_b2 (cc 10.7) dK / dV are BITWISE between ``smem_map = "sm100"`` and
    ``"rubin_b2"`` ON ``tmem_map = "dq_576"`` on every row, dQ BITWISE where the sequences have at most _DQ_ORDER_FREE_PAIRS kv pairs and by the fp64
    gate beyond; each map passes the suite's fp64 gate first.  The rows are the TMEM map's (the uniform cell, the ramp with dead pairs, the partial
    last q tile with a half-dead CTA, the zero-trip mix = the N-walk of the dq_read_done row and the two-slot rings, the all-empty case, and n_q == 1
    with more pairs than SMs = the tail dQ(lo) alone, two items per CTA: the P14 killers of the kernel header section 12).  The first launch of the
    314368-B slab IS the oversized-SMEM launch proof (a launcher that did not set the mode fails the launch)."""
    _require_smem_map("rubin_b2")
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG

    spec = dict(spec)
    lq, lk, causal = spec.pop("lq"), spec.pop("lk"), spec.pop("causal")
    if lk and lk[0] == "skv":
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        skv = 256 * ((sms // (spec["hq"] * 2)) + 2) + 128  # pairs x hq x b > sms, the last pair half-dead (the P14 row's cell)
        lk = (skv, skv - 256)
    case = make_case(lq, lk, thd=True, dtype=torch.bfloat16, causal=causal, stats_layout=spec.pop("stats_layout", "TH1"), **spec)
    dq_bitwise = max(lk) <= _DQ_ORDER_FREE_PAIRS * CFG.PAIR_KV_ROWS
    grads = {}
    for name in _RING_TWIN_MAPS:
        api = make_adapter_2cta_stail(case, tmem_map="dq_576", smem_map=name)
        workspace = make_workspace(api.scratch_workspace_bytes())
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
        grads[name] = {role: case.tensors[role][0].detach().clone() for role in ("dq", "dk", "dv")}
    for role in ("dk", "dv") + (("dq",) if dq_bitwise else ()):
        assert torch.equal(grads["sm100"][role].view(torch.int16), grads["rubin_b2"][role].view(torch.int16)), f"{role}: sm100 vs rubin_b2 differ bitwise"


@pytest.mark.parametrize("tmem", ["dq_576", "s_tail"])
def test_2cta_stail_sm100_oversized_agrees_bitwise_with_sm100(tmem):
    """The DIAGNOSTIC map ``smem_map = "sm100_oversized"`` (the kernel header section 11; cc 10.7 by name, never a default): the sm100 map ITSELF
    declared with SMEM_BYTES 262144 (above the 232448-B opt-in cap), so the DSL launcher takes the oversized carveout for the shipping map -- the probe
    of the carveout's own cost (the SKU card's L1 shrink).  On the smallest L0 cell dQ / dK / dV are BITWISE against ``"sm100"`` on the same TMEM map
    (the same kernel bytes; dQ bitwise at <= _DQ_ORDER_FREE_PAIRS kv pairs) and each passes the fp64 gate; the launch under the map IS the proof that
    the 262144-B declaration launched in the carveout (the opt-in cap cannot hold it).  The declared size is read back from the geometry record."""
    _require_smem_map("sm100_oversized")
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG, cfg_for_part
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import smem_map

    cc = torch.cuda.get_device_capability()
    geo = cfg_for_part(cc, "tma_ring", "16w", tmem, "sm100_oversized")
    assert geo.SMEM_MAP == "sm100_oversized" and geo.SMEM_BYTES == 262144 > CFG.SMEM_OPT_IN_CAP and geo.SMEM_PAD_BYTES == 29696
    print(
        f"\nsm100_oversized x {tmem} on cc {cc}: the launch declares {geo.SMEM_BYTES} B of dynamic SMEM (opt-in cap {CFG.SMEM_OPT_IN_CAP}, pad {geo.SMEM_PAD_BYTES})"
    )
    case = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    grads = {}
    for name in ("sm100", "sm100_oversized"):
        api = make_adapter_2cta_stail(case, tmem_map=tmem, smem_map=name)
        assert smem_map(api) == name
        workspace = make_workspace(api.scratch_workspace_bytes())
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
        grads[name] = {role: case.tensors[role][0].detach().clone() for role in ("dq", "dk", "dv")}
    for role in ("dq", "dk", "dv"):  # 256 kv rows = one pair: dQ bitwise too
        assert torch.equal(
            grads["sm100"][role].view(torch.int16), grads["sm100_oversized"][role].view(torch.int16)
        ), f"{role}: sm100 vs sm100_oversized differ bitwise"


def test_2cta_stail_rubin_b2_declines_by_name_on_a_sm100_part():
    """``smem_map = "rubin_b2"`` NAMED on a part without the oversized carveout (cc 10.0 / 10.3) is the engine's typed decline at ``check_support``
    (NotImplementedError by name, through the prepared module's reader); "sm100" named is served.  On a part that serves rubin_b2 the row skips (the
    bitwise twin covers it)."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import SMEM_MAP_PARTS

    _require_2cta()
    cc = torch.cuda.get_device_capability()
    if cc in SMEM_MAP_PARTS["rubin_b2"]:
        pytest.skip(f"cc {cc} serves smem_map 'rubin_b2' (the bitwise twin covers it)")
    case = make_case((256, 256), (256, 256), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True)
    with pytest.raises(NotImplementedError, match="2-CTA datapath smem_map 'rubin_b2' is not served on cc"):
        make_adapter_2cta_stail(case, compile_plan=False, smem_map="rubin_b2").check_support()
    assert make_adapter_2cta_stail(case, compile_plan=False, smem_map="sm100").check_support() is True


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


@pytest.mark.parametrize(
    "maps",
    [
        pytest.param({}, id="default-map"),
        pytest.param({"tmem_map": "dq_576"}, id="dq_576"),
        pytest.param(dict(_B2_FIELDS), id="dq_576-rubin_b2"),  # + the leader's dq_read_done words live (the record of the re-based bar block's spare)
    ],
)
def test_2cta_stail_debug_clk_dump_accounts_the_waits(maps):
    """The STAMPS lever (`debug_clk = True`, header section 11) on the part's default map and, where served, on ``tmem_map = "dq_576"`` (its record
    is the shorter bar block's spare with the same word map; the four S-tail words are dead): dK / dV BITWISE vs the default (the lever changes no arithmetic) and the fp64 gate
    under both; the dump's record layout (config DBG_CLK): every record carries the magic and its rank; each role accumulates EXACTLY the barriers
    it waits on -- the leader's words non-zero on rank 0 and zero on rank 1, the relay's the reverse, the shared roles' on both -- the tile
    counts agree across the roles and the pair and sum to the case's tile visits, the ship -> recv handshake accounts every dQ issue / relay,
    the ship histogram accounts every tile, the body clock and its %globaltimer span are live (the implied SM clock is a plausible GHz).
    Only the CTAs that RAN wrote a record (the CLC scheduler cancels the rest of the declared grid; the region is not zeroed): a record is valid
    iff its magic word matches, and a never-launched cluster leaves BOTH of its CTAs' records unwritten."""
    if not maps:
        _require_2cta()
    elif "smem_map" in maps:
        _require_smem_map(maps["smem_map"])
    else:
        _require_tmem_map(maps["tmem_map"])
    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import debug_clk_record, debug_clk_records

    on_map = dict(maps)
    hq, hkv = 4, 2
    case = make_case((256, 512), (256, 512), hq=hq, hkv=hkv, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _bitwise_twin(
        case, lambda c: make_adapter_2cta_stail(c, **on_map), lambda c: make_adapter_2cta_stail(c, debug_clk=True, **on_map), "the stamps lever (off vs on)"
    )
    plain = make_adapter_2cta_stail(case, **on_map)
    api = make_adapter_2cta_stail(case, debug_clk=True, **on_map)
    d = debug_clk_record(api)  # the record the carve and the dump use: derived from the layout the body renders (never a literal here)
    W = d.WORD
    n_rec = cfgmod.dbg_clk_records(case.batch, hq, api.s_k_max)
    assert (
        d.BYTES == 4 * d.WORDS and api.scratch_workspace_bytes() >= plain.scratch_workspace_bytes() + n_rec * d.BYTES
    ), "the dump region is carved only under the lever, sized exactly n_rec x record"
    assert api.scratch_workspace_bytes() < plain.scratch_workspace_bytes() + n_rec * d.BYTES + 2 * d.RECORD_ALIGN, "no second record size hides in the carve"
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
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import smem_map, tmem_map

    # tmem_map dq_576: the four S-tail words have no site and read 0; smem_map sm100: the dq_read_done pair has no site (the same word map everywhere)
    dead = set(cfgmod.dbg_clk_dead_words(tmem_map(api), smem_map(api)))
    for role, bars in d.ROLE_WAITS.items():
        words = torch.stack([col(f"wait:{role}:{b}") for b in bars if f"wait:{role}:{b}" not in dead], dim=1)  # (records, barriers)
        on = {"leader": (rank == 0) & live, "follower": (rank == 1) & live}.get(role, live)
        off = {"leader": rank == 1, "follower": rank == 0}.get(role)
        assert bool((words[on] > 0).all()), (role, words[on])  # every barrier the role waits on, every live CTA that runs it
        if off is not None:
            assert bool((words[off] == 0).all()), (role, "a role that does not run on this rank must accumulate nothing")
    for name in sorted(dead):
        assert bool((col(name) == 0).all()), (name, "a word without a site under this map must read 0")
    blocked = torch.stack([col(f"blocked:leader:{b}") for b in d.ROLE_WAITS["leader"] if f"blocked:leader:{b}" not in dead], dim=1)
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
    """Each DIAGNOSTIC probe of this map builds, launches and returns (no fault, no hang: the issue / transfer / bytes probes delete bytes or a
    transfer and keep every wait; the WAIT probes skip LEADER-waited relaxed thread-count rows with no tx bytes, whose arrives keep landing on
    phases nobody observes -- no over-arrival fault, no waiter left behind); the output is WRONG BY DESIGN and is NOT checked here --
    `--validate` refuses them by name in the driver.  exchange_off is the SAFE form (every wait kept).  The three S-tail WAIT probes are built on
    the map that defines them (``tmem_map = "s_tail"``, NAMED: under dq_576 they decline by name), so the row RUNS on every admitted part whatever
    its default; the other probes run on the part's default map."""
    from cudnn.sdpa.bwd.config_sm100_d128_accum_2cta_stail import CFG

    _require_2cta()
    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    fields = {"tmem_map": "s_tail"} if probe in CFG.DIAG_2CTA_S_TAIL_PROBES else {}
    api = make_adapter_2cta_stail(case, diag_2cta=probe, **fields)
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
from d128_2cta_test_utils import child_adapter
case = make_case((1024, 1024), (1024, 1024), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
api = child_adapter(make_adapter_2cta_stail, case, diag_2cta={probe!r})  # exit CHILD_EXIT_SKIP on a typed decline (the parent skips)
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
    the lever before the body executes; exit 70 / a timeout = HANG, a CUDA error = the fault class, exit 77 = the pair body declined on this
    device (the row SKIPS with the decline)."""
    _require_2cta()
    _adapter_class()
    import cudnn

    lever, delay = "LEADER_SHIP", 300
    assert lever in set(LEVER_LINE.findall(_body_source()))
    script = _CHILD_PROBE_SKEW.format(
        testdir=str(pathlib.Path(__file__).resolve().parent),
        body=_BODY_MODULE,
        global_name=f"{GLOBAL_PREFIX}{lever}_DELAY_US",
        attr=f"DEBUG_{lever}_DELAY_US",
        delay=delay,
        expected_fe=cudnn.__file__,
        probe="exchange_off",
        watchdog=WATCHDOG_S,
    )
    (tmp_path / "probe_skew_exchange_off.py").write_text(script)
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path / "probe_skew_exchange_off.py")], capture_output=True, text=True, timeout=CHILD_TIMEOUT_S, env=dict(os.environ)
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"[exchange_off + leader_ship {delay} us] HANG: the child did not return within {CHILD_TIMEOUT_S} s -> barrier AUDIT, never a bigger budget"
        )
    (tmp_path / "probe_skew_exchange_off.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    skip_if_child_declined(proc, "exchange_off + leader_ship")
    if proc.returncode == CHILD_EXIT_HANG:
        pytest.fail(
            f"[exchange_off + leader_ship] HANG: process_watchdog killed the child after {WATCHDOG_S:.0f} s\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
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
        "s_read ALONE (the dS gates of t-1 and the leader softmax's arrive are long done); under tmem_map dq_576 (selectable on cc 10.7) the row does not "
        "exist and the lever isolates nothing: a held S^T read delays p_ready only (SKEW_OK = the expected PASS)",
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
        "epilogue asleep, S^T(t+1) is held by s_tail_empty ALONE (q1_full landed long ago, the leader's readout is done); under tmem_map dq_576 "
        "(selectable on cc 10.7) the row does not exist and the lever is the TIMING probe of the independent image's reuse edge: the held readout delays "
        "this warp's next ds_smem.arrive() and hence dQ(t+1) (the carrier of the kernel header section 5) -- SKEW_OK, no RED exists by construction; "
        "under smem_map rubin_b2 (ON dq_576, the lagging order) the held readout holds the LEADER row dq_read_done ALONE: the leader's next dQ issue "
        "(the fourth gate inside _dq_issue) waits that readout, nothing else does -- SKEW_OK is the expected PASS",
        red="the wait removed at the lookahead: S^T(t+1) overwrites cols 64..127 before the follower reads dQ(t-1) -> its 64 dQ rows of tile t-1 = "
        "S^T values (dQ RED, dK / dV green); at the tail: dQ rows of tile hi-2 wrong; at the prologue: dQ rows of tile hi-1 of the previous item wrong; "
        "under rubin_b2: the dq_read_done wait inside _dq_issue removed -> dQ(t) overwrites the independent image before the follower reads dQ(t-1) -> its "
        "64 dQ rows of tile t-1 = dQ(t) values (dQ RED, dK / dV green); the static source pin test_2cta_stail_rubin_b2_reuse_edge_is_the_dq_read_done_row is the detector",
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
    # smem_map rubin_b2 (the barrier skeleton R2.c section 2.8; the AUDIT R2.c follow-up): the correctness instruments of the staging ring's two new edges.
    # The plain ids run them on the part's default map too (a pure delay there); the ``-rubin_b2`` ids are the rows the skeleton names.
    leader_arm=dict(
        lever="LEADER_ARM",  # the house-form id (the global is DEBUG_LEADER_ARM_DELAY_US; the skeleton 2.8 writes LEADER_ARM_DELAY)
        role="epilogue warps 8-11 of the LEADER (rank 0)",
        site="_epilogue_loop: right before ds_recv_full[s].arrive(n_bytes=XCHG_TX) (slot s = sds_state.idx under rubin_b2; the single row under sm100), EVERY tile",
        delay_us=300,
        isolates="the bytes-first order on slot s: the follower's ship (its XCHG stage -> our sDS stage s + complete_tx on our ds_recv_full[s]) lands BEFORE "
        "our expect_tx arm -> the transiently negative tx-count path of that mbarrier phase, which still completes at the arm (one arrival + the bytes "
        "already counted); under sm100 the same order on the single row",
        red="a wrong mbar / stage mapping inside _ship_xchg (the peer's complete_tx on another slot, or another stage's bytes): the armed phase never "
        "balances (HANG) or a pending phase over-arrives (the tx fault); SKEW_OK is the expected PASS -- the static ring pins are the RED detectors",
        green="fp64 gate on dQ / dK / dV, eager x2 + capture / replay x4, no hang",
        cell="skew",
    ),
    leader_slot1=dict(
        lever="LEADER_SLOT1",  # the house-form id (the global is DEBUG_LEADER_SLOT1_DELAY_US; the skeleton 2.8 writes LEADER_SLOT1_DELAY)
        role="MMA warp 12 of the LEADER (rank 0)",
        site="_mma_loop_leader: slot 1 of EVERY tile, before its first wait (leader-only by construction)",
        delay_us=300,
        isolates="under rubin_b2 the depth-2 ABA of the ds_smem / ds_recv_full / ds_pair_ready rings: with the leader held at slot 1 of tile t, both "
        "epilogues stage dS(t) into slot t % 2 and the follower relays ds_pair_ready[t % 2] BEFORE the leader consumed flip t-1 of the other slot at "
        "slot 2 -- the two slots' phases must stay distinct (SKEW_OK); a 1-slot form would HANG (the skeleton's static-RED row 2.6); under sm100 the "
        "single-slot rows are gated by dq_done (dS(t) is staged only after dq_done(t-1)): a pure delay",
        red="one state or one slot shared by the two stages (xchg_state / relay_state / sds_state at different cadences): the leader's gate of slot s "
        "passes on the OTHER tile's flip -> dQ(t-1) built on dS(t), or a HANG; SKEW_OK is the expected PASS",
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
from d128_2cta_test_utils import child_adapter
sms = torch.cuda.get_device_properties(0).multi_processor_count
skv = {skv}
print("DATASET sms=%d skv=%d (inputs: the suite's CPU generator, seed 149)" % (sms, skv), flush=True)
case = {cell}
api = child_adapter(make_adapter_2cta_stail, case, **{fields!r})  # the typed op-level field(s) through the suite's factory, as the driver sets them; exit 77 on a typed decline
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
# ``stats_form = "tma_ring"`` (lever S1; the S1 skeleton section 10): the levers that reach the ring's two new waits -- the LOADER lever (the loader's
# item-top sleep delays stats(t) behind the softmax's wait: a bounded stats_full stall, never a hang), the FOLLOWER softmax's S^T-read lever (the softmax
# behind the loader: the loader's stats_empty credit wait is the one that blocks -- bounded by the ring's depth) and the leader's ship lever (the
# exchange chain late while the ring keeps running).  Each a fresh child; the pass criterion is the SAME fp64 gate + capture / replay.
_STATS_RING_SKEW_ROWS = ("loader_zero_trip", "softmax_s_read", "leader_ship")
# ``tmem_map = "dq_576"`` (the independent dQ image; the kernel header section 5): the two levers that touch the deleted rows' former seams become
# the TIMING probes of the new reuse edge (a held readout / a held own-half STS delays ds_smem(t+1), hence dQ(t+1)): SKEW_OK is the expected PASS, no
# RED exists by construction (the static order pin is the RED detector).  The child names the map; a part that does not serve it exits 77 = skip.
_TMEM_MAP_SKEW_ROWS = ("epilogue_dq_readout", "follower_own_half")
# ``smem_map = "rubin_b2"`` ON dq_576 (the two-stage ring; the kernel header section 5, the barrier skeleton R2.c 2.8): the leader's held ship isolates
# ds_recv_full[s] and the per-stage ship routing (the peer's slot s), the follower's held own-half STS isolates ds_pair_ready[s] through the two-slot relay,
# and the held readout is the TIMING probe of the dq_read_done row (the leader's next issue waits the readout ALONE): SKEW_OK is the expected PASS; the
# RED detector of the row is the static source pin.  The child names both maps; a part without the carveout exits 77 = skip.
_SMEM_MAP_SKEW_ROWS = ("leader_ship", "follower_own_half", "epilogue_dq_readout", "leader_arm", "leader_slot1")
_SKEW_ROWS = (
    [pytest.param(site, None, None, None, None, id=site) for site in SITES]
    + [pytest.param(site, "fp32_reduce", None, None, None, id=f"{site}-fp32_reduce") for site in _DKV_ACCUM_SEAM_ROWS]
    + [pytest.param(site, None, "tma_ring", None, None, id=f"{site}-tma_ring") for site in _STATS_RING_SKEW_ROWS]
    + [pytest.param(site, None, None, "dq_576", None, id=f"{site}-dq_576") for site in _TMEM_MAP_SKEW_ROWS]
    + [pytest.param(site, None, None, "dq_576", "rubin_b2", id=f"{site}-rubin_b2") for site in _SMEM_MAP_SKEW_ROWS]
)


@pytest.mark.parametrize("site, dkv_accum, stats_form, tmem_map, smem_map", _SKEW_ROWS)
def test_sdpa_bwd_d128_2cta_stail_survives_pair_skew(site, dkv_accum, stats_form, tmem_map, smem_map, tmp_path):
    """One lever per id on the SECOND body (``SITES[site]``: role, site, delay, what it isolates, the RED form, the expected PASS): a fresh process
    arms ``FROST_D128_2CTA_DEBUG_<SITE>_DELAY_US`` before the body executes, runs the skew / p14 cell (more pairs than SMs) under ``process_watchdog``
    and asserts the fp64 gate eagerly and under capture / replay.  Exit 70 or a subprocess timeout = HANG -> the barrier AUDIT, never a bigger
    budget; exit 77 = the pair body declined on this device -> the row SKIPS with the decline (the sm80 / sm120 lanes); a gate miss = the RED
    form's wrong rows, named by ``check_case``.  ``dkv_accum = "fp32_reduce"`` (the ``-fp32_reduce`` ids, the two
    ``_DKV_ACCUM_SEAM_ROWS`` levers) sets the typed op-level field in the child exactly as the driver does and pins ``artifact.dkv_accum``; the
    pass criterion is the SAME gate (the fp32 form is tighter on every validated cell) -- never a widened tolerance."""
    row = SITES[site]
    fields = {} if dkv_accum is None else {"dkv_accum": dkv_accum}
    if stats_form is not None:
        fields["stats_form"] = stats_form  # lever S1: the typed stats form in the child, as the driver sets it
    if tmem_map is not None:
        fields["tmem_map"] = tmem_map  # the TMEM map in the child (the ``-dq_576`` ids): a part without the carve declines -> exit 77 -> skip
    if smem_map is not None:
        fields["smem_map"] = smem_map  # the SMEM map in the child (the ``-rubin_b2`` ids, ON dq_576): a part without the carveout declines -> exit 77 -> skip
    form = "bf16_partials" if dkv_accum is None else dkv_accum
    seam = _DKV_ACCUM_SEAM_ROWS.get(site, "(the field does not move this lever's bytes)") if dkv_accum else ""
    assert row["lever"] in set(LEVER_LINE.findall(_body_source())), f"the second body must expose {row['lever']}"
    _require_2cta()
    _adapter_class()
    import cudnn

    script = _CHILD.format(
        testdir=str(pathlib.Path(__file__).resolve().parent),
        body=_BODY_MODULE,
        global_name=f"{GLOBAL_PREFIX}{row['lever']}_DELAY_US",
        attr=f"DEBUG_{row['lever']}_DELAY_US",
        delay=row["delay_us"],
        expected_fe=cudnn.__file__,
        skv=SKV,
        cell=CELLS[row["cell"]],
        watchdog=WATCHDOG_S,
        site=site,
        fields=fields,
        dkv=form,
    )
    tag = (
        f"{site}_{form}"
        + ("" if stats_form is None else f"_{stats_form}")
        + ("" if tmem_map is None else f"_{tmem_map}")
        + ("" if smem_map is None else f"_{smem_map}")
    )
    (tmp_path / f"skew_stail_{tag}.py").write_text(script)
    print(f"\n[{site}] {row['role']} sleeps {row['delay_us']} us at {row['site']}; isolates {row['isolates']}; RED = {row['red']}; GREEN = {row['green']}")
    if dkv_accum:
        print(f"[{site}] dkv_accum={form}: {seam}")
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path / f"skew_stail_{tag}.py")], capture_output=True, text=True, timeout=CHILD_TIMEOUT_S, env=dict(os.environ)
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        (tmp_path / f"skew_stail_{tag}.log").write_text(out)
        pytest.fail(f"[{site}] HANG: the child did not return within {CHILD_TIMEOUT_S} s (RED form: {row['red']}) -> barrier AUDIT, never a bigger budget")
    (tmp_path / f"skew_stail_{tag}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    skip_if_child_declined(proc, site)
    if proc.returncode == CHILD_EXIT_HANG:
        pytest.fail(
            f"[{site}] HANG: process_watchdog killed the child after {WATCHDOG_S:.0f} s (RED form: {row['red']})\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    assert (
        proc.returncode == 0 and f"SKEW_OK {site} {row['delay_us']} dkv_accum={form}" in proc.stdout
    ), f"rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
