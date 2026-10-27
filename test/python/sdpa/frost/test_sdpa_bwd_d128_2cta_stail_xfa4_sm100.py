# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 grouped THD d128 backward on the cta_group::2 datapath: the S-tail pair body under FAv4's ROLE SHAPE -- the typed adapter field
``warp_roles = "16w_fa4"`` ("X-FA4", day-3 rank 1; barrier-inspector SKELETON resources/x-fa4-barrier-table.md 2026-10-27 08:45 UTC, resource-manager
resources/xfa4-budgets.md 08:15 UTC).  The SOFTMAX writes sDS own + XCHG straight from its 32 packed dS^T words (today's LDTM image word for word),
fences ONCE, arrives ``ds_smem`` (256), passes the 256-lane XCHG named barrier and its lead warp's elected lane arms ``ds_recv_full`` + ships; a NEW
softmax ``dq_done`` WAR wait (``dq_war`` st(1)) precedes the first dS STS of every tile; warps 8-11 become the REDUCE group (``_reduce_loop``:
dq_done -> x64 dQ ld -> s_tail_empty -> rounds A, B; the item-end dV / dK staging kept); ``ds_tmem`` AND ``dp_empty`` are DELETED (``BAR_LAYOUT_XFA4``
32 names / 37 slots / END 230912); s_read kept; the dQ ring keeps form A; TMEM / SMEM / the 144-144-80 split UNCHANGED.  The default (``warp_roles``
unset = "16w") stays pinned byte-identical by ``test_sdpa_bwd_d128_2cta_stail_sm100.py`` (645f8a24 @ 2/8/2) and re-asserted here.

RED-first: the GPU rows below were committed with the body's first host trace-compile, before it ran; every one is the grouped suite's independent
FP64 gate (``check_case``: per-sequence atol / rtol + cosine, storage canaries, exact zeros for empty attention), dQ / dK / dV apart.

Rows: host -- config = truth for the role shape (the typed domain PER MAP, the two declines by name (red dQ routes, ``ds_gates_off``), the 32-row
table pasted == derived, the role-derived lane counts), the host renders (16w_fa4 census vs 16w with EVERY delta a table row; the md5 record; the
default unchanged), the inspector's section-12 punch list as SOURCE assertions (the softmax site ORDER, the reduce group's order, ds_tmem / dp_empty
absent from the 16w_fa4 paths, the arm's back-edge in program order, ONE proxy fence and no tcgen05 fence before ds_smem, the leader's two deletions,
the one-site skew levers); GPU -- the 8-cell matrix, all-empty, the P14 rows (n_q == 1 with more pairs than SMs, a half-dead last pair, the zero-trip
MIX) under both leader orders, the launch census, the dK / dV BITWISE twin 16w vs 16w_fa4 (the same accumulation chain per GEMM; dQ by the gate),
the fa4 / fa4b twin under 16w_fa4, and the seven pair-skew detectors run with ``warp_roles="16w_fa4"`` (incl. ``leader_ship`` at 300 us = the D8
mechanism's deterministic repro, which MUST stay clean under the correct protocol, and the NEW ``softmax_dq_war``).  Budget: the skew rows are the
``skew_module`` class (one cold child per row).
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
from d128_2cta_test_utils import _CELLS, _CHILD_TIMEOUT_S, _GLOBAL_PREFIX, _LEVER_LINE, _SKV, _WATCHDOG_S, _adapter_class, _require_2cta, _setmaxnreg_of
from test_sdpa_bwd_d128_2cta_stail_sm100 import _BODY_MODULE, _CHILD, _EXPECTED, _RECORD, SITES
from test_sdpa_bwd_d128_2cta_stail_sm100 import _bitwise_twin, _body_source, _function_code, _record_md5, _render, make_adapter_2cta_stail

pytestmark = [pytest.mark.L0, requires_dsl]

_ROLES = "16w_fa4"
_RECORD_XFA4 = pathlib.Path(__file__).resolve().parent / "renderings" / "md5_d128_accum_grouped_2cta_stail_xfa4_sm_100a.txt"
# The role shape's census beside the module census (resources/render_2cta_stail_x86.py keys): the dS^T readout LDTMs (4 x16 gone), the proxy fences
# (the epilogue's two -> the softmax's one; one _dq_round site fewer), the sDQ slice STS (3 -> 2 round sites x 8), the XCHG named barrier's thread
# count (128 -> 256), the dp_empty relaxed.cluster arrive, the plain arrives (ds_tmem + the third dq_full site), the leader's `.release` /
# `.relaxed` cluster arrives, the softmax's dS STS.128 (st.shared.v4.b32: the epilogue's 16 per lane -> the softmax's 8).
_XFA4_CENSUS = {
    "tcgen05_ld": r"tcgen05\.ld",
    "fence_proxy_async": r"fence\.proxy\.async",
    "sts_dq_v2_b64": r"st\.shared\.v2\.b64",
    "bar_sync_xchg_256": r"bar\.sync 2, 256",
    "bar_sync_xchg_128": r"bar\.sync 2, 128",
    "relaxed_cluster_arrive": r"mbarrier\.arrive\.relaxed\.cluster",
    "plain_arrive": r"mbarrier\.arrive\.shared\.b64",
    "expect_tx": r"mbarrier\.arrive\.expect_tx",
    "dsmem_ship": r"cp\.async\.bulk\.shared::cluster\.shared::cta",
}
# 16w (the audited S-tail body; AUDIT 04:17: arrive 40 / try_wait 68) and 16w_fa4 (the inspector's section 12 item 11; the writer's record 2026-10-27):
#   arrive 40 -> 37 = -1 ds_tmem (plain) -1 dp_empty (relaxed.cluster) -1 the item-end `_dq_round` site (dq_full plain: 3 -> 2 sites);
#   try_wait 68 -> 65 = -1 ds_tmem wait -2 dp_empty (the slot-1 wait + the P15 drain) +2 `dq_war` (the per-tile wait + the zero-trip wait)
#                       -2 dq_empty (3 -> 2 `_dq_round` sites x 2 slots);  commit 13, mma 72, the ship 1, expect_tx 11 (the arm MOVED), `.local` 0 unchanged.
_EXPECTED_16W_EXTRA = dict(
    tcgen05_ld=13,
    fence_proxy_async=15,
    sts_dq_v2_b64=24,
    bar_sync_xchg_256=0,
    bar_sync_xchg_128=1,
    relaxed_cluster_arrive=7,
    plain_arrive=15,
    expect_tx=11,
    dsmem_ship=1,
)
# Section 15 (the zero-trip correction): `dq_war_done` committed at the two dQ issue sites (one helper: + 2 `tcgen05.commit`); the softmax's zero-trip
# wait gone (-1 try_wait) and its post-loop drain added (+1): try_wait unchanged at 65; arrive unchanged at 37.
_EXPECTED_XFA4 = dict(
    _EXPECTED,
    commit_multicast=_EXPECTED["commit_multicast"] + 2,
    mbarrier_arrive=_EXPECTED["mbarrier_arrive"] - 3,
    try_wait=_EXPECTED["try_wait"] - 3,
    tcgen05_ld=_EXPECTED_16W_EXTRA["tcgen05_ld"] - 4,
    fence_proxy_async=_EXPECTED_16W_EXTRA["fence_proxy_async"] - 2,
    sts_dq_v2_b64=_EXPECTED_16W_EXTRA["sts_dq_v2_b64"] - 8,
    bar_sync_xchg_256=1,
    bar_sync_xchg_128=0,
    relaxed_cluster_arrive=_EXPECTED_16W_EXTRA["relaxed_cluster_arrive"] - 1,
    plain_arrive=_EXPECTED_16W_EXTRA["plain_arrive"] - 2,
    expect_tx=11,
    dsmem_ship=1,
    # the stack fold: the stail module's census (writer A) also counts these; the 16w_fa4 deltas are writer B's measured ones (three `_wait_tcgen`
    # sites gone: -3 tcgen05 fences / -3 warp syncs net; the epilogue's XCHG address mapa gone with its ship site: 17 -> 16)
    bar_warp_sync=_EXPECTED["bar_warp_sync"] - 3,
    tcgen05_fence_after=_EXPECTED["tcgen05_fence_after"] - 3,
    mapa=_EXPECTED["mapa"] - 1,
)
# the stack fold: the stail module's template carries writer C's `**{fields!r}` / `dkv_accum={dkv}` keys (426a5442); the role shape is set beside them
_CHILD_XFA4 = _CHILD.replace(
    "api = make_adapter_2cta_stail(case, **{fields!r})", f"api = make_adapter_2cta_stail(case, warp_roles={_ROLES!r}, **{{fields!r}})"
).replace('"2-CTA s_tail pair-skew {site} dkv_accum={dkv}"', '"2-CTA s_tail X-FA4 pair-skew {site} dkv_accum={dkv}"')
assert "warp_roles=" in _CHILD_XFA4 and "X-FA4" in _CHILD_XFA4 and "{fields!r}" in _CHILD_XFA4 and "dkv_accum={dkv}" in _CHILD_XFA4
from cudnn.sdpa.bwd.d128_grouped_contract import DKV_ACCUM_DEFAULT as _DKV_ACCUM_DEFAULT  # noqa: E402  (the artifact's recorded form under the default)


def _run_case(case, repeats=2, **fields):
    api = make_adapter_2cta_stail(case, warp_roles=_ROLES, **fields)
    workspace = make_workspace(api.scratch_workspace_bytes())
    for _ in range(repeats):
        poison(case, workspace)
        api.execute(**adapter_arguments(case), workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
    return api, workspace


# ------------------------------------------------------------------------------------------------------------ host (no GPU)
def test_xfa4_role_shape_config_is_the_truth():
    """Every number the X-FA4 header block quotes is a derived, asserted field of ``config_sm100_d128_accum_2cta_stail`` (G10 config-as-truth): the
    typed domain, the 32-row ``BAR_LAYOUT_XFA4`` (pasted from the inspector's section 8 AND re-derived from the audited 34-row table: the two deletions +
    ds_smem counted from the softmax, re-packed, END 230912), the kind counts, the role-derived lane counts (XCHG named barrier 256, ds_smem 256), the
    lead warp, the dS STS image counts, and the bundles the body builds from them (``Bars`` 34 fields, ``BarsXFA4`` 32)."""
    import importlib

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod

    c = cfgmod.CFG
    assert c.WARP_ROLES == ("16w", "16w_fa4") and c.WARP_ROLES_DEFAULT == "16w" and cfgmod.WARP_ROLES_CODES == {"16w": 0, "16w_fa4": 1}
    assert cfgmod.DELETED_BARRIERS_XFA4 == ("ds_tmem", "dp_empty") and cfgmod.ADDED_BARRIERS_XFA4 == ("dq_war_done",)
    assert len(cfgmod.BAR_LAYOUT_XFA4) == 33 and sum(r[2] for r in cfgmod.BAR_LAYOUT_XFA4) == 38 and cfgmod.BAR_LAYOUT_XFA4_END == 230928 <= c.TMEM_HANDLE
    names = [r[0] for r in cfgmod.BAR_LAYOUT_XFA4]
    assert "ds_tmem" not in names and "dp_empty" not in names
    # section 15: appended after tmem_dealloc; the stack fold: writer C's dvk_stage_empty is UNSERVED under 16w_fa4 (fp32_reduce declines by name)
    assert names == [r[0] for r in cfgmod.BAR_LAYOUT if r[0] not in cfgmod.DELETED_BARRIERS_XFA4 and r[0] not in cfgmod.UNSERVED_BARRIERS_XFA4] + [
        "dq_war_done"
    ]
    assert [r[1] for r in cfgmod.BAR_LAYOUT_XFA4] == [c.BARS + 16 * i for i in range(33)], "re-packed in declaration order, 16 B per row"
    counts = {r[0]: r[3] for r in cfgmod.BAR_LAYOUT_XFA4}
    counts_16w = {r[0]: r[3] for r in cfgmod.BAR_LAYOUT}
    assert counts["ds_smem"] == c.SOFTMAX_LANES == 256 and counts_16w["ds_smem"] == c.EPILOGUE_LANES == 128  # the ONE count that changes
    assert counts["dq_war_done"] == c.ONE_LANE == 1 and cfgmod.BAR_LAYOUT_XFA4[-1][4:] == ("mma", "LOCAL")  # the softmax's WAR signal: MMA_COMMIT, LOCAL
    assert all(counts[n] == counts_16w[n] for n in counts if n not in ("ds_smem", "dq_war_done")), "every other row keeps its count"
    assert cfgmod.BAR_KIND_COUNTS_XFA4 == {"tma_leader": 7, "mma": 10, "thread": 7, "leader": 5, "peer": 2, "tma_dsmem": 1, "tma": 1}
    assert cfgmod.BAR_KIND_COUNTS == {
        "tma_leader": 7,
        "mma": 9,
        "thread": 9,  # the stack: + writer C's dvk_stage_empty (THREAD 32) on the 16w table
        "leader": 6,
        "peer": 2,
        "tma_dsmem": 1,
        "tma": 1,
    }  # 16w: + ds_tmem (thread) + dp_empty (leader)
    assert (c.xchg_barrier_lanes("16w"), c.xchg_barrier_lanes("16w_fa4")) == (128, 256) and (c.ds_smem_lanes("16w"), c.ds_smem_lanes("16w_fa4")) == (128, 256)
    assert (c.SOFTMAX_LEAD_WARP, c.REDUCE_LEAD_WARP, c.REDUCE_WARPS, c.REDUCE_LANES, c.PAIR_REDUCE_LANES) == (0, 8, 4, 128, 256)
    assert (c.DS_STS_VECS_PER_QUARTER, c.DS_STS_PER_LANE_PER_TILE) == (4, 8) and c.M * c.SOFTMAX_GROUPS * c.DS_STS_PER_LANE_PER_TILE * 16 == 32768
    # every probe of the code space serves both role shapes
    assert all(cfgmod.validate_fields_for_roles("16w_fa4", probe) == "16w_fa4" for probe in c.DIAG_2CTA_NAMES_ON_MAP)
    assert (cfgmod.bar_layout_for_roles("16w") is cfgmod.BAR_LAYOUT) and (cfgmod.bar_layout_for_roles("16w_fa4") is cfgmod.BAR_LAYOUT_XFA4)
    assert (c.SOFTMAX_REGS, c.EPILOGUE_REGS, c.SERVICE_REGS) == (144, 144, 80) and (c.SMEM_BYTES, c.TMEM_COLS) == (231424, 512)  # UNCHANGED
    cfgmod.validate_config()
    body = importlib.import_module(_BODY_MODULE)
    assert tuple(body.Bars._fields) == tuple(r[0] for r in cfgmod.BAR_LAYOUT) and len(body.Bars._fields) == 35  # the audited 34 rows + the fp32 row
    assert tuple(body.BarsXFA4._fields) == tuple(names) and len(body.BarsXFA4._fields) == 33  # the 16w_fa4 bundle = 32 kept rows + dq_war_done
    assert (body.WARP_ROLES_16W, body.WARP_ROLES_16W_FA4, body.SOFTMAX_LEAD_WARP) == (0, 1, 0)
    assert (body.XCHG_BARRIER_LANES_16W, body.XCHG_BARRIER_LANES_16W_FA4) == (128, 256)


def test_xfa4_warp_roles_is_typed_and_declines_by_name():
    """``warp_roles`` is a typed field: ``None`` -> "16w" (the shipped rendering); "16w_fa4" is the alternate shape; any other name declines BY NAME;
    under "16w_fa4" the STAMPS lever and ``dkv_accum = "fp32_reduce"`` are not built (both decline BY NAME at the config bundle check and at
    ``compile_host`` before any trace); the adapter reader prefixes the engine name; the record default is None."""
    from types import SimpleNamespace

    from cudnn.sdpa.bwd import config_sm100_d128_accum_2cta_stail as cfgmod
    from cudnn.sdpa.bwd.api_dsl_sm100_d128 import SdpaBwdDslSm100D128
    from cudnn.sdpa.bwd.kernels.sm100.prepared_d128_grouped_2cta_host import compile_host
    from cudnn.sdpa.bwd.prepared_sm100_d128_grouped_2cta import warp_roles

    assert cfgmod.resolve_warp_roles(None) == "16w" == cfgmod.resolve_warp_roles("16w")
    assert cfgmod.resolve_warp_roles("16w_fa4") == "16w_fa4" and cfgmod.warp_roles_code("16w_fa4") == 1
    assert cfgmod.warp_roles_code(None) == 0 == cfgmod.warp_roles_code("16w")
    for bad_roles in ("bogus", "", 1, "16W", "20w"):
        with pytest.raises(ValueError, match="warp_roles must be one of"):
            cfgmod.resolve_warp_roles(bad_roles)
        with pytest.raises(ValueError, match="warp_roles must be one of"):
            cfgmod.warp_roles_code(bad_roles)
    assert cfgmod.validate_fields_for_roles("16w_fa4", "dq_reduce_add_off") == "16w_fa4" and cfgmod.validate_fields_for_roles("16w", "exchange_off") == "16w"
    with pytest.raises(ValueError, match="debug_clk is not defined under warp_roles '16w_fa4'"):
        cfgmod.validate_fields_for_roles("16w_fa4", None, debug_clk=True)
    with pytest.raises(ValueError, match="is not built under warp_roles '16w_fa4'"):
        cfgmod.validate_fields_for_roles("16w_fa4", None, dkv_accum="fp32_reduce")
    with pytest.raises(ValueError, match="diag_2cta must be None or one of"):
        cfgmod.validate_fields_for_roles("16w_fa4", "ds_gates_off")
    assert warp_roles(SimpleNamespace(_NAME="x")) == "16w" and warp_roles(SimpleNamespace(_NAME="x", warp_roles="16w_fa4")) == "16w_fa4"
    with pytest.raises(ValueError, match="sdpa_bwd_sm100_d128: warp_roles must be one of"):
        warp_roles(SimpleNamespace(_NAME="sdpa_bwd_sm100_d128", warp_roles="16w-fa4"))
    assert SdpaBwdDslSm100D128.warp_roles is None  # the record default: the shipped rendering
    specs, dtypes = tuple((i, 1, 1, "int32") for i in range(9)), tuple("int32" for _ in range(9))
    with pytest.raises(ValueError, match="debug_clk is not defined under warp_roles '16w_fa4'"):
        compile_host((2, 8, 2, True), specs, dtypes, None, True, 100, warp_roles="16w_fa4", debug_clk=True)
    with pytest.raises(ValueError, match="dkv_accum 'fp32_reduce' is not built under warp_roles '16w_fa4'"):
        compile_host((2, 8, 2, True), specs, dtypes, None, True, 100, warp_roles="16w_fa4", dkv_accum="fp32_reduce")
    with pytest.raises(ValueError, match="warp_roles must be one of"):
        compile_host((2, 8, 2, True), specs, dtypes, None, True, 100, warp_roles="20w")


def test_xfa4_host_renders_for_sm_100a_and_matches_its_record(tmp_path):
    """Host trace-compile of the chain with ``warp_roles = "16w_fa4"`` (no device): the census is the inspector's section-12 item 11 with EVERY delta a
    table row (arrive 40 -> 37, try_wait 68 -> 65, `bar.sync 2, 256` and no `2, 128`, 4 LDTM fewer, 2 proxy fences fewer, 8 slice STS fewer, one
    relaxed.cluster arrive fewer, the ship and the expect_tx arm count unchanged), no local memory, no nanosleep, the default register split; the md5
    matches the committed record at this geometry (skips, never widens, on a DSL build difference)."""
    md5, census, regs = _render(tmp_path, {"warp_roles": _ROLES}, _XFA4_CENSUS)
    print(f"\n2-CTA s_tail 16w_fa4 sm_100a: PTX md5 {md5} census {census} setmaxnreg {regs}")
    assert census == _EXPECTED_XFA4, {k: (census.get(k), v) for k, v in _EXPECTED_XFA4.items() if census.get(k) != v}
    assert sorted(regs) == _setmaxnreg_of(), regs  # the split is UNCHANGED
    dsl, want = _record_md5(_RECORD_XFA4, "d128_accum_grouped_2cta_stail_xfa4")
    assert want is not None, _RECORD_XFA4
    import importlib.metadata as md

    have = f"nvidia-cutlass-dsl {md.version('nvidia-cutlass-dsl')}"
    if dsl != have:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: PTX text differs by DSL build, re-render the record")
    assert md5 == want, f"PTX md5 {md5} != the record's {want} -- the 16w_fa4 rendering changed"


def test_xfa4_default_rendering_is_unchanged_and_16w_named_is_the_default(tmp_path):
    """The field ABSENT and ``warp_roles = "16w"`` render the SAME PTX = the S-tail record (645f8a24 @ 2/8/2: the pin of the sibling module), with the
    16w census (arrive 40 / try_wait 68 / `bar.sync 2, 128` / 13 LDTM / 15 proxy fences / 24 slice STS); nothing of X-FA4 reaches the default."""
    md5_d, census_d, regs_d = _render(tmp_path, {}, _XFA4_CENSUS)
    md5_n, census_n, _ = _render(tmp_path, {"warp_roles": "16w"}, _XFA4_CENSUS)
    assert md5_d == md5_n, "warp_roles named '16w' must be the default rendering"
    assert census_d == dict(_EXPECTED, **_EXPECTED_16W_EXTRA) == census_n, census_d
    assert sorted(regs_d) == _setmaxnreg_of()
    dsl, want = _record_md5(_RECORD, "d128_accum_grouped_2cta_stail")
    import importlib.metadata as md

    if dsl == f"nvidia-cutlass-dsl {md.version('nvidia-cutlass-dsl')}":
        assert md5_d == want, f"the default S-tail rendering moved: {md5_d} != {want}"


# ------------------------------------------------------------------------------------------------------------ host: the inspector's section-12 punch list as SOURCE assertions
def _stmt(ln):
    """The statement of a code line without its trailing comment (the bodies carry `# RESOURCE-NOTE` / `# skew lever` comments inline)."""
    return ln.strip().split("  #")[0].strip()


def _xfa4_block(code, start_token, end_token):
    """The statements of ONE const_expr branch: from the line holding ``start_token`` (exclusive) to the first line holding ``end_token`` (exclusive)."""
    start = next(i for i, ln in enumerate(code) if start_token in ln)
    end = next(i for i, ln in enumerate(code) if end_token in ln and i > start)
    return code[start + 1 : end]


def test_xfa4_softmax_site_order_is_the_inspectors_stub_order():
    """Punch list item 2: per tile under ``fa4_roles`` -- ... STTM -> wait(STORE) -> fence -> ds_ready -> `_wait(dq_done, dq_war)` -> advance -> 8 STS
    (two `_store_words_quarter` calls = 4 own + 4 XCHG) -> ONE `fence.proxy.async` -> `ds_smem.arrive()` -> `bar.sync 2, 256` -> the elected arm ->
    the ship; `ds_tmem.arrive()` only under `not fa4_roles`; `dq_war = PipelineState.start(phase=1)`; `ship_pred` computed ONCE per item from
    `(warp == SOFTMAX_LEAD_WARP) & elect_sync()`; the zero-trip wait + advance at the item end under `lo >= hi`; NO tcgen05 fence between the STS and
    the ds_smem arrive (item 9); the arm's back-edge = the WAR wait precedes it in program order (item 8)."""
    src = _body_source()
    sm = _function_code(src, "_softmax_loop", "_epilogue_loop")
    assert sum("_arrive_leader(bars.ds_ready)" in ln for ln in sm) == 1 and sum("_wait_tcgen(bars.dp_done" in ln for ln in sm) == 1
    assert any(_stmt(ln) == "dq_war = PipelineState.start(phase=1)" for ln in sm), "the WAR state is PRE-ARMED (P5)"
    assert sum("ship_pred = is_lead_warp & nvvm.elect_sync()" in ln for ln in sm) == 1, "ONE elect per item (P16)"
    assert any("is_lead_warp = warp == cutlass.Int32(SOFTMAX_LEAD_WARP)" in ln for ln in sm)
    i_ready = next(i for i, ln in enumerate(sm) if "_arrive_leader(bars.ds_ready)" in ln)
    assert sm[i_ready + 1].strip() == "if cutlass.const_expr(not fa4_roles):" and "bars.ds_tmem.arrive()" in sm[i_ready + 2], sm[i_ready + 1 : i_ready + 3]
    assert sm[i_ready + 3].strip() == "else:"
    fa4 = _xfa4_block(sm, "bars.ds_tmem.arrive()", "if qtile + 1 < hi:")
    fa4 = fa4[1:]  # drop the `else:` line
    order = [
        "_wait(bars.dq_war_done, dq_war.phase, None, 0)",  # the stack: the STAMPS arity, dbg=None (debug_clk declines under 16w_fa4)
        "dq_war = advance(dq_war, 1)",
        "_skew_follower_own_half(rank)",
        "xor_t = opaque_i32(xor_lane, anchor=qtile)",
        "_store_words_quarter(slab, lo_row, words, 0, xor_t)",
        "_store_words_quarter(slab, hi_row, words, CFG.PACK_QUARTER_COLS, xor_t)",
        'nvvm.fence_proxy("async.shared", space="cta")',
        "bars.ds_smem.arrive()",
        "named_barrier_fence(CFG.XCHG_BARRIER, XCHG_BARRIER_LANES_16W_FA4)",
        "_skew_leader_ship(rank)",
        "bars.ds_recv_full.arrive(n_bytes=xchg_tx, pred=ship_pred)",  # the stack: D8s re-sited -- the config-selected byte count (0 under the probe)
        "if cutlass.const_expr(not exchange_off):",  # the stack: the ship is the ONLY statement under the D8s guard (as at the 16w site)
        "_ship_xchg(slab, bars, rank, ship_pred)",
    ]
    assert [_stmt(ln) for ln in fa4] == order, fa4
    assert not any("tcgen05_fence" in ln for ln in fa4), "no tcgen05 fence between the STS and the ds_smem arrive (the words come from registers)"
    assert sum("fence_proxy" in ln for ln in fa4) == 1, "ONE proxy fence for the 8 STS"
    assert any("_wait(bars.dq_war_done, dq_war.phase" in ln for ln in fa4) and "_wait_tcgen(bars.dq_" not in " ".join(fa4), "plain _wait at the WAR site"
    # section 15: the softmax waits dq_war_done ONLY (never dq_done); exactly one per-tile wait + one post-loop drain; no zero-trip wait / advance
    assert not any("bars.dq_done" in ln for ln in sm), "the softmax never touches dq_done (section 15: dq_done is the reduce group's alone)"
    waits = [i for i, ln in enumerate(sm) if "_wait(bars.dq_war_done, dq_war.phase" in ln]
    assert len(waits) == 2, waits
    i_next = next(i for i, ln in enumerate(sm) if "_next_tile(sched, bars, sched_state" in ln)
    assert waits[0] < i_next < waits[1], "the per-tile wait inside the loop; the drain AFTER the persistent loop"
    assert _stmt(sm[waits[1] - 1]) == "if cutlass.const_expr(fa4_roles):" and waits[1] == len(sm) - 1, sm[
        waits[1] - 2 :
    ]  # the drain is the role's last statement
    assert sum("dq_war = advance(dq_war, 1)" in ln for ln in sm) == 1, "ONE advance (per live tile); none on the zero-trip path, none after the drain"
    assert not any("if lo >= hi:" in ln for ln in sm), "no zero-trip wait site in the softmax"
    # every dS STS of the softmax runs AFTER its WAR wait in program order (the arm's back-edge, section 5.3)
    assert waits[0] < sm.index(next(ln for ln in sm if "_store_words_quarter(" in ln))
    # section 15.5 item 1: dq_war_done is committed inside _dq_issue only, immediately BEFORE dq_done, under fa4_roles; the zero-trip site commits dq_done only
    issue = _function_code(src, "_dq_issue", "_dq_slot")
    i_c = next(i for i, ln in enumerate(issue) if "_commit_pair(bars.dq_war_done, elect_p)" in ln)
    assert _stmt(issue[i_c - 1]) == "if cutlass.const_expr(fa4_roles):" and "_commit_pair(bars.dq_done, elect_p)" in issue[i_c + 1], issue[i_c - 1 : i_c + 2]
    assert "mma_ss(dqdesc, dsd, k2d, dq_acc)" in issue[i_c - 2], issue[i_c - 2]
    assert (
        len(re.findall(r"^\s+_commit_pair\(bars\.dq_war_done, elect_p\)", src, re.M)) == 1
    ), "ONE commit site in code (the helper at two call sites; the header quotes it)"
    leader = _function_code(src, "_mma_loop_leader", "_mma_loop_follower")
    assert not any("dq_war_done" in ln for ln in leader), "the leader body names dq_war_done only through _dq_issue (the zero-trip site commits dq_done only)"
    zt = [ln for ln in leader if "_commit_pair(bars.dq_done, elect_p)" in ln]
    assert len(zt) == 1, zt  # the zero-trip empty commit (the dQ sites commit through _dq_issue)
    assert len(re.findall(r"^\s+_wait\(bars\.dq_war_done, dq_war\.phase, None, 0\)", src, re.M)) == 2, "two waits: the per-tile wait + the drain"
    # the quarter -> slab map of resource-manager table 1: lo = quarter g (q half 0) -> own on rank 0 / XCHG on rank 1; hi = quarter g+2 -> the other
    assert any("lo_base = cutlass.Int32(arith.select(is_leader_cta.ir_value(), own_base.ir_value(), xchg_base.ir_value()))" in ln for ln in sm)
    assert any("hi_base = cutlass.Int32(arith.select(is_leader_cta.ir_value(), xchg_base.ir_value(), own_base.ir_value()))" in ln for ln in sm)
    helper = _function_code(src, "_store_words_quarter", "_store_dq_slice")
    assert any("for v in cutlass.range_constexpr(CFG.DS_STS_VECS_PER_QUARTER):" in ln for ln in helper)
    assert any("off = row_base + (xor_lane ^ v) * 16" in ln for ln in helper), "the (row % 8) ^ vec XOR of _store_words"


def test_xfa4_reduce_group_order_and_no_ds_work():
    """Punch list item 5: the reduce group = dq_done wait -> the x64 dQ ld -> wait(LOAD) -> fence -> s_tail_empty BEFORE any `_dq_round`; both rounds
    inside the tile (no carried half, no item-end round); the zero-trip dq_done observation; `_native_out` + dv_full / dk_full at the item end; NO dS
    code (no ds_tmem, no XCHG STS, no arm, no ship, no dp_empty) in the role."""
    src = _body_source()
    rd = _function_code(src, "_reduce_loop", "_load_loop")
    wait = next(i for i, ln in enumerate(rd) if "_wait_tcgen(bars.dq_done, dq_state.phase, None, 0)" in ln)
    ld = next(i for i, ln in enumerate(rd) if "dq = _tmem_load(tmem_base + CFG.DQ_COL, CFG.DQ_COLS)" in ln)
    arrive = next(i for i, ln in enumerate(rd) if "_arrive_leader(bars.s_tail_empty)" in ln)
    rounds = [i for i, ln in enumerate(rd) if "_dq_round(" in ln]
    assert wait < ld < arrive < rounds[0] < rounds[1] and len(rounds) == 2, (wait, ld, arrive, rounds)
    assert any("nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)" in ln for ln in rd[ld:arrive]) and any("tcgen05_fence" in ln for ln in rd[ld:arrive])
    assert (
        "_dq_half(dq, 0)" in rd[rounds[0]] and "_dq_half(dq, CFG.DQ_BOX)" in rd[rounds[1]]
    ), "round A = cols [0, 32), round B = cols [32, 64) of the x64 readout"
    assert all(", False, sts_off, None)" in rd[i] for i in rounds), "the tma route only (route_red False), the sdq_sts_off probe's flag, no stamps (dbg None)"
    assert sum("_wait_tcgen(bars.dq_done, dq_state.phase, None, 0)" in ln for ln in rd) == 2, "the per-tile wait + the zero-trip observation"
    assert any(_stmt(ln) == "if lo >= hi:" for ln in rd) and any(_stmt(ln) == "dq_state = advance(dq_state, 1)" for ln in rd)
    assert not any("carried" in ln for ln in rd) and not any(
        "empty_state = advance" in ln for ln in rd
    ), "no carried half; the rounds flip empty_state inside _dq_round"
    for token in ("ds_tmem", "dp_empty", "ds_recv_full", "_ship_xchg", "XCHG_BARRIER", "_store_words(", "x0 = _tmem_load"):
        assert not any(token in ln for ln in rd), token
    assert sum("_native_out(" in ln for ln in rd) == 2 and any("bars.dv_full.arrive()" in ln for ln in rd) and any("bars.dk_full.arrive()" in ln for ln in rd)
    assert any("empty_state = PipelineState.start(phase=1)" in ln for ln in rd), "the ring's pre-armed empty state (D3, no bootstrap)"


def test_xfa4_dp_empty_and_ds_tmem_absent_from_the_16w_fa4_paths():
    """Punch list item 3 / 4: every ``dp_empty`` statement of the leader (the state, the slot-1 wait, the advance, the P15 drain) sits under
    `if cutlass.const_expr(not fa4_roles):`; the leader's `_dq_issue` / `_dq_slot` / `_s_lookahead` / tail / prologue sites are untouched (the sibling
    module's RED 4 / RED 5 rows); the P15 drains end with s_tail_empty; the bundle is built per role shape; the dispatch runs `_reduce_loop` under
    `fa4_roles` and `_epilogue_loop` otherwise; the kernel's trace-time backstops are `const_expr` raises."""
    src = _body_source()
    leader = _function_code(src, "_mma_loop_leader", "_mma_loop_follower")
    dpe = [i for i, ln in enumerate(leader) if "dp_empty" in ln]
    assert len(dpe) == 4, dpe  # the state, the slot-1 wait, the advance, the drain
    for i in dpe:
        assert leader[i - 1].strip() == "if cutlass.const_expr(not fa4_roles):", (leader[i - 1], leader[i])
    assert leader[-1].strip() == '_wait_tcgen(bars.s_tail_empty, s_tail_state.phase, dbg, _W["wait:leader:s_tail_empty"])' and "dvk_empty" in leader[-2]
    kernel = _function_code(src, "bprop_d128_accum_grouped_2cta_stail_kernel", "main_thd_2cta_stail_host")
    assert any("bars = _bars(slab, fa4_roles)" in ln for ln in kernel) and any("fa4_roles = warp_roles_code == WARP_ROLES_16W_FA4" in ln for ln in kernel)
    i_red = next(i for i, ln in enumerate(kernel) if "_reduce_loop(" in ln)
    assert kernel[i_red - 1].strip() == "if cutlass.const_expr(fa4_roles):" and kernel[i_red + 1].strip() == "else:" and "_epilogue_loop(" in kernel[i_red + 2]
    for guard in ("warp_roles_code not in (WARP_ROLES_16W, WARP_ROLES_16W_FA4)", "fa4_roles and debug_clk", "fa4_roles and dkv_fp32"):
        assert any(f"if cutlass.const_expr({guard}):" in ln for ln in kernel), guard
    # the bundles: one 34-field class + one 32-field NamedTuple built from BAR_LAYOUT_XFA4; the init loop runs over whichever bundle it was handed
    assert 'BarsXFA4 = NamedTuple("BarsXFA4", [(row[0], MBarrier) for row in BAR_LAYOUT_XFA4])' in src
    assert "for i in cutlass.range_constexpr(len(bars)):" in src
    bars_fn = _function_code(src, "_bars", "_wait")
    assert any('bar_layout_for_roles("16w_fa4" if fa4_roles else "16w")' in ln for ln in bars_fn) and any(
        "bundle = BarsXFA4 if fa4_roles else Bars" in ln for ln in bars_fn
    )
    # the follower relay, the loader, the store warp and the scheduler read no role shape (item 6)
    for fn, nxt in (("_mma_loop_follower", "_reduce_loop"), ("_load_loop", "_store_loop"), ("_store_loop", "_scheduler_loop")):
        code = _function_code(src, fn, nxt)
        assert not any("fa4_roles" in ln for ln in code), fn


def test_xfa4_skew_levers_keep_one_site_each_and_the_new_dq_war_lever_exists():
    """Punch list item 12: the moved levers (LEADER_SHIP, FOLLOWER_OWN_HALF, EPILOGUE_DQ_READOUT, FOLLOWER_EPILOGUE_ITEM_END) are ONE-site helpers called
    from both role shapes' bodies; the NEW SOFTMAX_DQ_WAR lever sits in `_dq_issue` between the three gate waits and the dQ issue (leader-only by
    construction); every lever keeps the house form (the sibling module's row) and is a SITES row here."""
    src = _body_source()
    exposed = set(_LEVER_LINE.findall(src))
    assert "SOFTMAX_DQ_WAR" in exposed and exposed == {row["lever"] for row in SITES.values()}, (sorted(exposed), sorted(r["lever"] for r in SITES.values()))
    for helper, lever in (
        ("_skew_leader_ship", "LEADER_SHIP"),
        ("_skew_follower_own_half", "FOLLOWER_OWN_HALF"),
        ("_skew_follower_dq_readout", "EPILOGUE_DQ_READOUT"),
        ("_skew_follower_item_end", "FOLLOWER_EPILOGUE_ITEM_END"),
    ):
        assert src.count(f"cutlass.const_expr(DEBUG_{lever}_DELAY_US > 0)") == 1, lever
        assert len(re.findall(rf"^\s+{helper}\(rank\)", src, re.M)) == 2, helper  # the 16w site (epilogue) + the 16w_fa4 site (softmax / reduce group)
    issue = _function_code(src, "_dq_issue", "_dq_slot")
    gates = [i for i, ln in enumerate(issue) if "_wait_tcgen(bars." in ln]
    lever = next(i for i, ln in enumerate(issue) if "DEBUG_SOFTMAX_DQ_WAR_DELAY_US > 0" in ln)
    mma = next(i for i, ln in enumerate(issue) if "mma_ss(dqdesc, dsd, k2d, dq_acc)" in ln)
    assert len(gates) == 3 and gates[-1] < lever < mma, (gates, lever, mma)
    assert src.count('nvvm.inline_ptx("nanosleep.u32 1000;"') == len(exposed)


# ------------------------------------------------------------------------------------------------------------ GPU (the fp64 gate)
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
def test_xfa4_thd_matrix(lq, lk, hq, hkv, causal, layout, offset_dtype, units):
    """The sibling module's 8-cell matrix under ``warp_roles = "16w_fa4"`` (S % 256 != 0, a half-dead pair, the degenerate THD rows, non-causal, a ramp)."""
    _require_2cta()
    case = make_case(lq, lk, hq=hq, hkv=hkv, thd=True, dtype=torch.bfloat16, causal=causal, stats_layout=layout, offset_dtype=offset_dtype, units=units)
    _run_case(case)


def test_xfa4_thd_all_empty():
    """Every sequence empty: exact zeros; the dead-pair protocol on BOTH CTAs (no dS STS, no ship, no `dq_war` traffic beyond the zero-trip observation)."""
    _require_2cta()
    case = make_case((0, 0), (0, 0), hq=3, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout="HN", envelope=(128, 128))
    _run_case(case)


def test_xfa4_persistent_reuse_more_pairs_than_sms():
    """P14 row: n_q == 1 with more pairs than SMs (a cluster claims a second CLC item: the softmax's `dq_war` consumes the previous item's tail flip at
    tile lo'; the N == 1 tail path), a half-dead last pair, zero-trip blocks past seq_q."""
    _require_2cta()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    hq, b = 8, 2
    skv = 256 * ((sms // (hq * b)) + 2) + 128
    case = make_case((128, 64), (skv, skv - 256), hq=hq, hkv=1, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _run_case(case)


def test_xfa4_zero_trip_mix():
    """The N-walk as one case (N = 3, 1, 0 / 1, 0 / 8, 6, 4, 2 with dead blocks): every s_tail_empty guard, the softmax's `dq_war_done` observation
    across empty items, the carry of the pending completions across zero-trip items."""
    _require_2cta()
    case = make_case((300, 128, 1000), (2000, 700, 1000), hq=4, hkv=2, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _run_case(case)


def test_xfa4_launch_count_and_cluster_facts():
    """The chain is five launches with the stail MAIN kernel under 16w_fa4 (the same entry name: the role shape is a Constexpr, keyed in the plan cache)."""
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
    assert any("bprop_d128_accum_grouped_2cta_stail_kernel" in n for n in kernels) and not any(
        "bprop_d128_accum_grouped_2cta_kernel" in n for n in kernels
    ), kernels


def test_xfa4_role_shapes_agree_bitwise_on_dk_dv():
    """16w and 16w_fa4 are the SAME function on dK / dV by construction: the softmax's arithmetic, the packed words and every accumulation chain per GEMM
    (dV += P^T dO, dK += dS^T Q over the tiles in ascending t, the same TS k-order) are untouched -- only WHO stores the identical dS^T words to the
    identical sDS bytes changes; pinned BITWISE (G09 twin-diff); dQ (the ring's fp32 reduce order, unchanged) passes the fp64 gate under each.  Four
    tiles per pair so the lookahead, the in-tile dQ and the tail are all exercised."""
    _require_2cta()
    case = make_case((256, 512), (256, 512), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout="TH1")
    _bitwise_twin(case, lambda c: make_adapter_2cta_stail(c), lambda c: make_adapter_2cta_stail(c, warp_roles=_ROLES), "the two role shapes (16w vs 16w_fa4)")


@pytest.mark.parametrize("site", list(SITES))
def test_sdpa_bwd_d128_2cta_stail_xfa4_survives_pair_skew(site, tmp_path):
    """One lever per id on the S-tail body under ``warp_roles = "16w_fa4"`` (``SITES`` of the sibling module incl. the NEW ``softmax_dq_war``): a fresh
    process arms ``FROST_D128_2CTA_DEBUG_<SITE>_DELAY_US`` before the body executes, runs the skew / p14 cell (more pairs than SMs) under
    ``process_watchdog`` and asserts the fp64 gate eagerly and under capture / replay.  ``leader_ship`` at 300 us = the barrier-inspector's deterministic
    repro of the D8 TX-PENDING over-arrival (08:01): under the correct protocol the arm's back-edge (the softmax's `dq_war` wait) holds on both CTAs and
    the run stays clean (SKEW_OK); a 719 = STOP, the inspector reads it.  Exit 70 or a subprocess timeout = HANG -> the barrier AUDIT, never a bigger
    budget; a gate miss = the RED form's wrong rows, named by ``check_case``.  Budget class: ``skew_module``."""
    row = SITES[site]
    assert row["lever"] in set(_LEVER_LINE.findall(_body_source())), f"the body must expose {row['lever']}"
    _require_2cta()
    _adapter_class()
    import cudnn

    script = _CHILD_XFA4.format(
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
        fields={},  # the stail template's C2b keys: the default form under 16w_fa4 (fp32_reduce declines there by name)
        dkv=_DKV_ACCUM_DEFAULT,
    )
    (tmp_path / f"skew_xfa4_{site}.py").write_text(script)
    print(f"\n[{site}] 16w_fa4: {row['role']} sleeps {row['delay_us']} us at {row['site']}; isolates {row['isolates']}; RED = {row['red']}")
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path / f"skew_xfa4_{site}.py")], capture_output=True, text=True, timeout=_CHILD_TIMEOUT_S, env=dict(os.environ)
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        (tmp_path / f"skew_xfa4_{site}.log").write_text(out)
        pytest.fail(f"[{site}] HANG: the child did not return within {_CHILD_TIMEOUT_S} s (RED form: {row['red']}) -> barrier AUDIT, never a bigger budget")
    (tmp_path / f"skew_xfa4_{site}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode == 70:
        pytest.fail(
            f"[{site}] HANG: process_watchdog killed the child after {_WATCHDOG_S:.0f} s (RED form: {row['red']})\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    assert proc.returncode == 0 and f"SKEW_OK {site} {row['delay_us']}" in proc.stdout, f"rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"


# ------------------------------------------------------------------------------------------------------------ GPU: the zero-trip HANG repro, fresh-process rows
# An earlier build of the 16w_fa4 form HUNG (a 25-minute pytest timeout inside torch.cuda.synchronize) on the first matrix cell with DEAD PAIRS:
# the softmax's `dq_war` WAR wait is PRE-ARMED (its first wait passes only while NO dq_done flip has
# completed) and the table asked it to observe the zero-trip EMPTY dq_done commit too -- but that commit has no softmax edge in its producer
# chain (dvk_empty <- the store warps <- the reduce group), so on a zero-trip item it can land BEFORE the softmax's wait: the softmax spins on
# a completed phase forever and the CTA never reaches the final barrier_cta_sync (a zero-trip item as a cluster's FIRST item); two consecutive
# zero-trip items after a live one on one cluster = two flips before one observation = a LAP (a deadlock at the next live item's first STS).
# RED-then-green: on the pre-fix tree these rows FAIL by the watchdog (exit 70 = HANG, the process killed to free the GPU); with the dedicated
# `dq_war_done` row they pass.  Each row is ONE fresh child process (the kernel wedges its context); the watchdog wraps the executes only (compile is outside it),
# with the SHORT kernel-stage budget by name -- never the 25-minute pytest timeout that found the hang.
_HANG_WATCHDOG_S = 120.0  # config/budgets.yaml stages.validate_multihead (a 1024-class validate cell: run + the CPU reference; compile outside)
_HANG_CASES = {
    "zero_trip_first_item_ramp": dict(
        case="make_case((256, 512, 768, 1024), (256, 512, 768, 1024), hq=4, hkv=2, thd=True, dtype=torch.bfloat16, causal=True, stats_layout='TH1')",
        what="the matrix cell that HUNG: 64 items on 76 cluster slots (one item per cluster), 12 dead pairs (the 256-row sequence inside the 1024-row "
        "envelope) = 12 clusters whose ONLY item is zero-trip -> 24 first-item races (both CTAs' softmaxes) between the leader's empty dq_done commit "
        "and the softmax's pre-armed dq_war wait",
    ),
    "zero_trip_first_then_full_item": dict(
        case="make_case((256, 2048), (256, 2048), hq=8, hkv=1, thd=True, dtype=torch.bfloat16, causal=True, stats_layout='TH1')",
        what="128 items > 76 slots: batch 0 (256 rows in a 2048-row envelope) hands 56 clusters a zero-trip FIRST item (pairs 1-7 of 8 heads), which "
        "then steal batch 1's live items in CLC order -> a full item AFTER a zero-trip one on the same cluster (the softmax's dq_war must still be in "
        "step) + 112 first-item races",
    ),
    "consecutive_zero_trip_items_lap": dict(
        case="make_case((2048, 256, 256), (2048, 256, 256), hq=8, hkv=1, thd=True, dtype=torch.bfloat16, causal=True, stats_layout='TH1')",
        what="192 items: after their live batch-0 items the clusters steal RUNS of dead pairs (batches 1-2: 112 zero-trip items contiguous in grid "
        "order) -> two consecutive zero-trip items after a live one on one cluster = the LAP hazard (two empty commits before one softmax observation) "
        "+ 10 first-item races (batch 1's dead pairs among the first 76 items)",
    ),
}
_HANG_CHILD = """
import os, sys
from pathlib import Path
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
sys.path.insert(0, {testdir!r})
import torch
import cudnn
assert Path(cudnn.__file__).resolve() == Path({expected_fe!r}).resolve(), cudnn.__file__
from d128_grouped_test_utils import adapter_arguments, check_case, make_case, make_workspace, poison
from frost_test_utils import process_watchdog
from test_sdpa_bwd_d128_2cta_stail_sm100 import make_adapter_2cta_stail
sms = torch.cuda.get_device_properties(0).multi_processor_count
print("DATASET sms=%d (inputs: the suite's CPU generator, seed 149)" % sms, flush=True)
case = {case}
api = make_adapter_2cta_stail(case, warp_roles={roles!r})  # compile (outside the watchdog: a slow compiler is not a hang)
workspace = make_workspace(api.scratch_workspace_bytes())
args = adapter_arguments(case)
print("STAGE=kernel {name}", flush=True)
with process_watchdog({watchdog}, "2-CTA s_tail 16w_fa4 zero-trip repro {name}"):
    for _ in range(2):
        poison(case, workspace)
        api.execute(**args, workspace=workspace.tensor)
        torch.cuda.synchronize()
        check_case(case, workspace)
print("REPRO_OK {name}", flush=True)
"""


@pytest.mark.parametrize("name", list(_HANG_CASES))
def test_xfa4_zero_trip_items_do_not_hang(name, tmp_path):
    """RULE-GAPS 649 (RED-first): the zero-trip repro rows of the 16w_fa4 body in a FRESH child process with the short kernel-stage watchdog
    (`_HANG_WATCHDOG_S` = budgets `validate_multihead`): a zero-trip item as a cluster's ONLY / FIRST item, a full item after a zero-trip one, and
    two consecutive zero-trip items after a live one (the lap).  Exit 70 = the watchdog fired inside the executes = HANG (RED on a90d6521: the
    softmax's pre-armed dq_war wait vs the leader's empty dq_done commit); rc 0 + REPRO_OK = green; any other rc = the child's own error."""
    row = _HANG_CASES[name]
    _require_2cta()
    _adapter_class()
    import cudnn

    script = _HANG_CHILD.format(
        testdir=str(pathlib.Path(__file__).resolve().parent), expected_fe=cudnn.__file__, case=row["case"], roles=_ROLES, watchdog=_HANG_WATCHDOG_S, name=name
    )
    (tmp_path / f"hang_xfa4_{name}.py").write_text(script)
    print(f"\n[{name}] {row['what']}")
    timeout_s = 900  # import (~100-130 s cold) + JIT + the fp64 reference + the watchdog; a TimeoutExpired here is read as a HANG too
    try:
        proc = subprocess.run([sys.executable, str(tmp_path / f"hang_xfa4_{name}.py")], capture_output=True, text=True, timeout=timeout_s, env=dict(os.environ))
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        (tmp_path / f"hang_xfa4_{name}.log").write_text(out)
        pytest.fail(f"[{name}] HANG: the child did not return within {timeout_s} s -> barrier AUDIT (RULE-GAPS 649), never a bigger budget")
    (tmp_path / f"hang_xfa4_{name}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode == 70:
        pytest.fail(
            f"[{name}] HANG: process_watchdog killed the child after {_HANG_WATCHDOG_S:.0f} s inside the executes (RED: the softmax's pre-armed dq_war "
            f"wait vs the zero-trip empty dq_done commit, RULE-GAPS 649)\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
        )
    assert proc.returncode == 0 and f"REPRO_OK {name}" in proc.stdout, f"rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
