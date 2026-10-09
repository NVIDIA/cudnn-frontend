# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.sdpa.bwd.config_sm100``: the 2x2-datapath stage-2 config (``CfgBwdD512x2``), host-only.

The layout facts the fused kernel ``kernels/sm100/bprop_d512_f16_2x2.py`` and the adapter rely on are pinned as
VALUES, not just as "does not raise": the SMEM tally and slab offsets per arm (SM100 4-stage / SM107 8-stage ring),
every tcgen05 descriptor root and the version-0 window rule, the 256-column TMEM carve, the expect_tx byte counts,
the mbarrier arrival counts (P3) and the cluster q span.  The 4x1 role-split record is pinned UNCHANGED: the base
``TemplateParams`` gained no field and ``make_cfg_d512`` renders the numbers it always did (its PTX md5 pin lives in
``test_sdpa_bwd_dsl_sm100.py``).

Device-independent: nothing here compiles or launches.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path

import pytest

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16
from cudnn.sdpa.bwd import config_sm100 as cfgmod
from cudnn.sdpa.bwd.config_sm100 import (
    SM107_USABLE_DYN_SMEM_2X2,
    SMEM_SCAFFOLD_BYTES_2X2,
    TCGEN05_V0_ADDR_LIMIT_2X2,
    CfgBwdD512,
    CfgBwdD512x2,
    TemplateParams,
    TemplateParams2x2,
    acc_cols_2x2,
    cast_bytes_2x2,
    cast_subtiles_2x2,
    desc_roots_2x2,
    desc_version_2x2,
    make_cfg_d512,
    make_cfg_d512_2x2,
    op_tx_bytes_2x2,
    operand_bytes_2x2,
    ring_tx_bytes_2x2,
    smem_bytes_2x2,
    smem_layout_2x2,
    tmem_cols_2x2,
)

pytestmark = [pytest.mark.L0]

_KERNELS = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "sdpa" / "bwd" / "kernels" / "sm100"
_KERNEL_2X2 = _KERNELS / "bprop_d512_f16_2x2.py"

_SM107 = dict(stages_kv=8, cast_stages=2, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2)


def _sm100():
    return make_cfg_d512_2x2(TemplateParams2x2())


def _sm107():
    return make_cfg_d512_2x2(TemplateParams2x2(**_SM107))


# --------------------------------------------------------------------------- the two arms' layouts, as values


def test_sm100_arm_smem_layout_is_the_design_table():
    cfg = _sm100()
    assert (cfg.STAGES_KV, cfg.CAST_STAGES, cfg.D_CHUNK, cfg.N_CHUNKS) == (4, 1, 64, 8)
    table = [(s.name, s.nbytes, s.offset) for s in smem_layout_2x2(cfg)]
    assert table == [
        ("sQ", 65536, 0),
        ("sdO", 65536, 65536),
        ("sRingK", 4 * 8192, 131072),
        ("sRingV", 4 * 8192, 163840),
        ("sCastS", 16384, 196608),
        ("sCastDS", 16384, 212992),
    ]
    assert smem_bytes_2x2(cfg) == 229376 == 224 * 1024
    assert smem_bytes_2x2(cfg) + SMEM_SCAFFOLD_BYTES_2X2 <= cfg.SMEM_CAP_BYTES == 227 * 1024
    assert operand_bytes_2x2(cfg) == (65536, 65536, 8192, 8192)
    assert cast_bytes_2x2(cfg) == 16384 and cast_subtiles_2x2(cfg) == 2


def test_sm107_arm_smem_layout_is_the_design_table():
    cfg = _sm107()
    assert (cfg.STAGES_KV, cfg.CAST_STAGES) == (8, 2)
    table = [(s.name, s.nbytes, s.offset) for s in smem_layout_2x2(cfg)]
    assert table == [
        ("sQ", 65536, 0),
        ("sdO", 65536, 65536),
        ("sRingK", 8 * 8192, 131072),
        ("sRingV", 8 * 8192, 196608),
        ("sCastS", 2 * 16384, 262144),
        ("sCastDS", 2 * 16384, 294912),
    ]
    assert smem_bytes_2x2(cfg) == 327680 == 320 * 1024
    assert smem_bytes_2x2(cfg) + SMEM_SCAFFOLD_BYTES_2X2 <= cfg.SMEM_CAP_BYTES == 325 * 1024


def test_descriptor_roots_and_version_per_arm():
    """Every tcgen05 descriptor root (8 Q chunks, 8 dO chunks, one per ring stage of K and of V) and the version-0 rule:
    SM100's last root is sRingV stage 3 at 188416; SM107's is sRingV stage 7 at 253952 whose LAST BYTE is 262143 --
    version 0 with zero margin, and only because the cast slabs (TMA-store sources, no descriptor) are declared last."""
    sm100, sm107 = _sm100(), _sm107()
    roots100 = dict(desc_roots_2x2(sm100))
    assert len(roots100) == 8 + 8 + 4 + 4
    assert roots100["sQ.chunk0"] == 0 and roots100["sQ.chunk7"] == 7 * 8192 and roots100["sdO.chunk0"] == 65536
    assert roots100["sRingK.stage3"] == 131072 + 3 * 8192 and roots100["sRingV.stage3"] == 188416
    assert max(roots100.values()) == 188416 and desc_version_2x2(sm100) == 0
    roots107 = dict(desc_roots_2x2(sm107))
    assert len(roots107) == 8 + 8 + 8 + 8
    assert max(roots107.values()) == roots107["sRingV.stage7"] == 253952
    assert 253952 + 8192 == TCGEN05_V0_ADDR_LIMIT_2X2 == 256 * 1024 and desc_version_2x2(sm107) == 0
    # A 9th Rubin stage grows sRingK by 8 KiB and puts sRingV stage 7 AT the window (262144) -> version 1 for every
    # SmemTile (the flip, not a wrap).
    nine = make_cfg_d512_2x2(TemplateParams2x2(stages_kv=9, cast_stages=1, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2))
    roots9 = dict(desc_roots_2x2(nine))
    assert roots9["sRingV.stage7"] == TCGEN05_V0_ADDR_LIMIT_2X2 and roots9["sRingV.stage8"] == 270336 and desc_version_2x2(nine) == 1


def test_tmem_carve_and_transaction_bytes():
    for cfg in (_sm100(), _sm107()):
        assert acc_cols_2x2(cfg) == 64 and tmem_cols_2x2(cfg) == 256 <= cfg.TMEM_CAP_COLS == 512
        assert op_tx_bytes_2x2(cfg) == 2 * (65536 + 65536) == 262144
        assert ring_tx_bytes_2x2(cfg) == 2 * (8192 + 8192) == 32768


def test_barrier_counts_and_cluster_span():
    cfg = _sm100()
    assert (cfg.CGA_M, cfg.CGA_N, cfg.CTA_MMA, cfg.TILE_M, cfg.TILE_N) == (4, 1, 2, 64, 128)
    assert cfg.CLUSTER_Q_ROWS == 256 == cfg.CGA_M * cfg.TILE_M and cfg.Q_BLOCKS_PER_CLUSTER == 4
    assert cfg.KV_SHARE == 2 and cfg.RING_EMPTY_ARRIVERS == 2 == cfg.CGA_M // cfg.CTA_MMA
    assert cfg.TMEM_DEALLOC_ARRIVERS == 2 == cfg.CTA_MMA
    assert (cfg.DEBUG_WAIT_MS, cfg.DEBUG_DUMP_ADDR) == (0, 0)
    # The pair-local arm: one reader per slot -> one commit per chunk; the multicast masks collapse to self / pair.
    local = make_cfg_d512_2x2(TemplateParams2x2(kv_share=1))
    assert local.KV_SHARE == 1 and local.RING_EMPTY_ARRIVERS == 1
    with pytest.raises(ValueError, match=re.escape("KV_SHARE must be 1")):
        make_cfg_d512_2x2(TemplateParams2x2(kv_share=3))
    with pytest.raises(ValueError, match=re.escape("must be set together")):
        make_cfg_d512_2x2(TemplateParams2x2(debug_wait_ms=1000))
    assert cfg.ACC_EMPTY_ARRIVERS == 256 == cfg.COMPUTE_LANES * cfg.CTA_MMA and cfg.COMPUTE_LANES == 128
    assert cfg.READ_TILE_ARRIVERS == 28 == (cfg.SOFTMAX_WG_WARPS + 3) * cfg.CGA_M * cfg.CGA_N
    assert (cfg.ONE_LANE, cfg.ONE_WARP, cfg.STAGES_ACC, cfg.WS_BLOCK_ROWS) == (1, 32, 2, 128)
    assert cfg.MMA_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512 and cfg.SOFTMAX_REGS == 240


# --------------------------------------------------------------------------- the 4x1 record is untouched


def test_base_record_and_4x1_config_are_unchanged():
    """The base ``TemplateParams`` carries none of the 2x2 levers (so every 4x1 rendering's digest and cache entry is what it
    was), ``TemplateParams2x2`` is a SUBCLASS with defaulted extras, and ``make_cfg_d512`` still renders the role-split
    geometry: TILE_M 128, two sub-groups, 28 scheduler arrivers, no cluster-span field."""
    base_fields = set(TemplateParams.__dataclass_fields__)
    levers = {
        "stages_kv",
        "cast_stages",
        "d_chunk",
        "smem_cap_bytes",
        "stages_acc",
        "kv_share",
        "debug_wait_ms",
        "debug_dump_addr",
        "debug_heartbeat",
        "wait_form",
        "debug_clk",
    }
    assert not base_fields & levers
    assert issubclass(TemplateParams2x2, TemplateParams)
    assert set(TemplateParams2x2.__dataclass_fields__) == base_fields | levers
    # The 4x1's own attribution record is a sibling subclass: the base record still carries neither lever.
    assert issubclass(cfgmod.TemplateParamsDbg, TemplateParams) and not issubclass(cfgmod.TemplateParamsDbg, TemplateParams2x2)
    assert set(cfgmod.TemplateParamsDbg.__dataclass_fields__) == base_fields | {"debug_clk", "debug_dump_addr"}
    assert cfgmod.TemplateParamsDbg() != TemplateParams()  # a different record -> a different template digest
    assert make_cfg_d512_2x2(TemplateParams2x2(stages_acc=4)).STAGES_ACC == 4 and tmem_cols_2x2(make_cfg_d512_2x2(TemplateParams2x2(stages_acc=4))) == 512
    assert TemplateParams2x2() != TemplateParams()  # a different record -> a different template digest
    cfg = make_cfg_d512(TemplateParams())
    assert isinstance(cfg, CfgBwdD512) and not hasattr(cfg, "CLUSTER_Q_ROWS") and not hasattr(cfg, "Q_BLOCKS_PER_CLUSTER")
    assert (cfg.TILE_M, cfg.TILE_N, cfg.CGA_M, cfg.CTA_MMA, cfg.STAGES_KV, cfg.XFER_HALVES, cfg.READ_TILE_ARRIVERS) == (128, 128, 4, 2, 2, 2, 28)
    assert cfg.TILE_M * cfg.CTA_MMA == 256 == CfgBwdD512x2().CLUSTER_Q_ROWS


def test_make_bwd_decode_reads_the_span_through_getattr_defaults():
    """``_common.make_bwd_decode`` serves both geometries: the span and the block count are ``getattr`` reads whose
    defaults are the 4x1 arithmetic (``TILE_M * CTA_MMA``, ``CTA_MMA``) -- pinned on the SOURCE, since the decode
    closures are ``@cute.jit`` bodies that need a trace context to run."""
    from cudnn.sdpa.bwd.kernels.sm100 import _common

    src = inspect.getsource(_common.make_bwd_decode)
    assert 'getattr(CFG, "CLUSTER_Q_ROWS", CFG.TILE_M * CFG.CTA_MMA)' in src
    assert 'getattr(CFG, "Q_BLOCKS_PER_CLUSTER", CFG.CTA_MMA)' in src
    assert "cutlass.Int32(_QB) + row_key" in src and "CFG.CTA_MMA) + cta_in_pair" not in src


# --------------------------------------------------------------------------- the validator's predicates


@pytest.mark.parametrize(
    "params,match",
    [
        (TemplateParams2x2(stages_kv=5), "over the 227 KiB per-CTA cap"),
        (TemplateParams2x2(cast_stages=2), "over the 227 KiB per-CTA cap"),
        (TemplateParams2x2(stages_kv=1), "STAGES_KV must be >= 2"),
        (TemplateParams2x2(cast_stages=0), "CAST_STAGES must be >= 1"),
        (TemplateParams2x2(d_chunk=48), "d_chunk must be a positive divisor of 512"),
        (TemplateParams2x2(d_chunk=256), "over the 227 KiB per-CTA cap"),
        (TemplateParams2x2(dtype_qkv=0), "dtype_qkv must be DTYPE_BF16"),
        (TemplateParams2x2(bottom_right=True), "bottom_right alignment requires a causal upper bound"),
        (TemplateParams2x2(stages_kv=11, cast_stages=1, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2), "over the 325 KiB per-CTA cap"),
        (TemplateParams2x2(debug_clk=1), "debug_heartbeat / debug_clk need debug_dump_addr"),
        (TemplateParams2x2(debug_clk=1, debug_dump_addr=4096, debug_heartbeat=1), "debug_clk writes a different dump record"),
        (TemplateParams2x2(debug_clk=1, debug_dump_addr=4096, debug_wait_ms=5), "debug_clk writes a different dump record"),
        (TemplateParams2x2(debug_clk=2, debug_dump_addr=4096), "debug_heartbeat / debug_clk need debug_dump_addr"),
    ],
    ids=["sm100_5_stages", "sm100_2_cast", "1_stage", "0_cast", "chunk_48", "chunk_256", "fp8", "br_without_causal", "cc107_11_stages"]
    + ["clk_without_dump", "clk_with_heartbeat", "clk_with_bounded", "clk_not_a_flag"],
)
def test_validator_rejects(params, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        make_cfg_d512_2x2(params)


def test_debug_clk_lever_is_append_only_default_off_and_declared_after_the_slabs():
    """The stage-2 attribution lever (``TemplateParams2x2.debug_clk``): the LAST field of the record (append-only, so every
    positional caller and every recorded ``repr`` of a default record is unchanged), default 0 -> ``CFG.DEBUG_CLK == 0`` and
    the kernel folds every ``_DBG_CLK`` block away (the PTX md5 pin in test_sdpa_bwd_dsl_sm100.py is the tripwire); armed
    with a dump address it reaches the config.  Source pins: the SMEM accumulator array is declared AFTER the last slab
    (``sCastDS_raw``) so no tcgen05 descriptor root moves under a debug build, the record is 32 x Int64 per warp, every
    ``_dbg_exit`` call hands over the tile counts the decoder normalizes by, and the lever never touches a wait's FORM
    (``_wait_plain`` is what the instrumented branch calls)."""
    fields = [f.name for f in dataclasses.fields(TemplateParams2x2)]
    assert fields[-2:] == ["wait_form", "debug_clk"], fields
    assert TemplateParams2x2().debug_clk == 0
    assert _sm100().DEBUG_CLK == 0
    assert make_cfg_d512_2x2(TemplateParams2x2(debug_clk=1, debug_dump_addr=4096)).DEBUG_CLK == 1
    # A base TemplateParams renders the all-defaults twin: the lever reads through getattr like the others.
    assert make_cfg_d512_2x2(TemplateParams()).DEBUG_CLK == 0
    src = _KERNEL_2X2.read_text()
    assert src.count("_DBG_CLK: bool = bool(CFG.DEBUG_CLK) and _DBG_DUMP_ADDR != 0") == 1
    assert "DBG_CLK_WORDS = 32" in src
    assert src.index("sCastDS_raw = cutlass.Array(") < src.index("sClk = cutlass.Array(cutlass.Int64, CFG.TOTAL_WARPS * DBG_CLK_WORDS")
    assert len(re.findall(r"^\s+_dbg_exit\(dbg, tile_no, ", src, flags=re.M)) == 5  # the five warp bodies' exits
    assert src.count("_dbg_exit(dbg)") == 0
    clk_branch = src[src.index("elif cutlass.const_expr(_DBG_CLK):") :]
    clk_branch = clk_branch[: clk_branch.index("else:")]
    assert "_wait_plain(mb, phase, poll)" in clk_branch and "clock64()" in clk_branch


def test_4x1_debug_clk_lever_is_a_sibling_record_default_off_and_wraps_every_wait():
    """The 4x1 role split's attribution lever (``TemplateParamsDbg``): a base ``TemplateParams`` renders ``DEBUG_CLK 0``
    (the shipped kernel, PTX md5 pinned in test_sdpa_bwd_dsl_sm100.py), the lever reaches the config only from the
    sibling record with a dump address, and the validator refuses one without the other.  Source pins: every mbarrier
    wait of the kernel goes through the two bracketing wrappers (the ONLY ``.wait(`` left is the wrapper's own), the
    named barrier 8 through its wrapper, the SMEM accumulators are declared AFTER the last slab, and all five warp bodies
    hand their tile counts to ``_dbg_exit``."""
    cfg = make_cfg_d512(TemplateParams())
    assert (cfg.DEBUG_CLK, cfg.DEBUG_DUMP_ADDR) == (0, 0)
    assert make_cfg_d512(cfgmod.TemplateParamsDbg()).DEBUG_CLK == 0
    assert make_cfg_d512(cfgmod.TemplateParamsDbg(debug_clk=1, debug_dump_addr=4096)).DEBUG_CLK == 1
    with pytest.raises(ValueError, match=re.escape("debug_clk (0 / 1) and debug_dump_addr must be set together")):
        make_cfg_d512(cfgmod.TemplateParamsDbg(debug_clk=1))
    with pytest.raises(ValueError, match=re.escape("debug_clk (0 / 1) and debug_dump_addr must be set together")):
        make_cfg_d512(cfgmod.TemplateParamsDbg(debug_dump_addr=4096))
    src = (_KERNELS / "bprop_d512_f16.py").read_text()
    # No barrier of the kernel is waited directly any more: the only `.wait(` calls are _wait_c's two arms.
    assert re.findall(r"^\s+(?:bars|sched)\.[\w\[\]\.\s]*\.wait\(", src, flags=re.M) == []
    assert src.count("        mb.wait(phase)\n") == 2
    assert src.count("        wait(ptr, phase)\n") == 2  # _wait_cp's two arms
    assert len(re.findall(r"(?<![_\w])wait\(sched\.", src)) == 0
    assert src.count("barrier_id=8") == 2  # the wrapper's two arms are the only named-barrier-8 syncs
    assert len(re.findall(r"^\s+_bar8_c\(dbg\)$", src, flags=re.M)) == 2  # both call sites wrapped
    assert src.index("sCast_raw = cutlass.Array(") < src.index("sClk = cutlass.Array(cutlass.Int64, CFG.TOTAL_WARPS * DBG_CLK_WORDS")
    assert len(re.findall(r"^\s+_dbg_exit\(dbg, tile_no, ", src, flags=re.M)) == 5  # the five warp bodies' exits
    assert "DBG_CLK_WORDS = 32" in src


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("TILE_M", 128, "TILE_M must be 64"),
        ("TILE_N", 64, "TILE_N must be 128"),
        ("CLUSTER_Q_ROWS", 128, "CLUSTER_Q_ROWS must be CGA_M"),
        ("Q_BLOCKS_PER_CLUSTER", 2, "Q_BLOCKS_PER_CLUSTER must be CGA_M"),
        ("RING_EMPTY_ARRIVERS", 1, "RING_EMPTY_ARRIVERS must be KV_SHARE"),
        ("KV_SHARE", 4, "KV_SHARE must be 1 (pair-local K / V) or CGA_M // CTA_MMA"),
        ("TMEM_DEALLOC_ARRIVERS", 1, "TMEM_DEALLOC_ARRIVERS must be CTA_MMA"),
        ("DEBUG_WAIT_MS", 5, "debug_wait_ms and debug_dump_addr must be set together"),
        ("WAIT_FORM", 5, "WAIT_FORM must be 0 (shipped: poll on the cross-pair ring barriers"),
        ("ACC_EMPTY_ARRIVERS", 128, "ACC_EMPTY_ARRIVERS must be COMPUTE_LANES"),
        ("READ_TILE_ARRIVERS", 25, "READ_TILE_ARRIVERS"),
        ("STAGES_ACC", 3, "STAGES_ACC must be 2 or 4"),
        ("WS_BLOCK_ROWS", 96, "WS_BLOCK_ROWS must be a multiple of TILE_M"),
        ("SOFTMAX_WG_WARPS", 8, "exactly four compute warps"),
        ("TILE_K_HW_BMM1", 32, "TILE_K_HW must be 16"),
    ],
)
def test_validator_pins_the_geometry_fields(field, value, match):
    cfg = dataclasses.replace(_sm100(), **{field: value})
    with pytest.raises(ValueError, match=re.escape(match)):
        cfgmod._validate_cfg_d512_2x2(cfg)


@pytest.mark.parametrize("dtype,bpe", [(DTYPE_BF16, 2), (DTYPE_FP16, 2)])
def test_accepts_both_io_dtypes(dtype, bpe):
    cfg = make_cfg_d512_2x2(TemplateParams2x2(dtype_qkv=dtype))
    assert cfg.DTYPE_QKV == dtype and cfg.BPE == bpe


def test_stages_acc_4_fills_tmem():
    """The free TMEM lever: STAGES_ACC 4 -> 512 columns, still a power of two and within the cap."""
    cfg = dataclasses.replace(_sm100(), STAGES_ACC=4)
    cfgmod._validate_cfg_d512_2x2(cfg)
    assert tmem_cols_2x2(cfg) == 512


# --------------------------------------------------------------------------- the kernel source's pins


def test_kernel_source_pins():
    """Source-level rules of the sibling kernel: every SmemTile takes ``desc_version=DESC_VERSION`` (bound once from the
    config), every ``@cute.kernel`` is followed by the cuDNN symbol prefix (Rule 6), the four THD descriptor slots keep
    the 4x1 sibling's order and sequence axis (``prepared_host`` sizes the scratch once), and no DSMEM / UTCCP / alias
    seam / named barrier 8 survived the fusion."""
    src = _KERNEL_2X2.read_text()
    tiles = re.findall(r"= SmemTile\((.*?)\n    \)", src, flags=re.S)
    assert len(tiles) == 6, len(tiles)
    assert all("desc_version=DESC_VERSION" in t for t in tiles)
    assert src.count("DESC_VERSION: int = desc_version_2x2(CFG)") == 1
    assert src.count("@cute.kernel") == 2 == src.count('.set_name_prefix("cudnn", remove_cutlass_symbol=True)')
    assert "Q_SLOT, DO_SLOT, K_SLOT, V_SLOT = 0, 1, 2, 3" in src and "_THD_SEQ_ORD = 2" in src
    for gone in ("cp_async_bulk_shared_cluster_shared_cta", "tcgen05_cp", "mb_op_utccp_done", "mb_s_xfer", "barrier_id=8", "cross_sg_peer", "mma_ts("):
        assert gone not in src, gone
    # The init-count ledger: every MBarrier init is a named CFG constant.
    inits = re.findall(r"init_count=CFG\.(\w+)", src)
    assert sorted(inits) == sorted(
        ["ONE_LANE", "ONE_LANE", "CGA_M", "RING_EMPTY_ARRIVERS", "ONE_LANE", "ACC_EMPTY_ARRIVERS", "COMPUTE_LANES", "ONE_WARP", "TMEM_DEALLOC_ARRIVERS"]
    )
    # The cross-pair waits remain polled; the observer ACK protocol also orders phase reuse. The waits are never
    # parked in NANOSLEEP.SYNCS) under KV_SHARE 2, at every wait site -- the kv-loop ring waits and the end-of-kernel drain.
    polled = re.findall(r"_wait_b\(\s*bars\.mb_tma_ring_(empty|full)\[[^\]]+\]\.smem_ptr,(?:[^()]|\([^()]*\))*?poll=_KV_SHARED,", src, flags=re.S)
    assert sorted(polled) == ["empty", "empty", "full"], polled
    assert src.count("poll=_KV_SHARED,") == 3  # the three call sites (comments spell it without the trailing comma)
    # The symmetric TMEM dealloc gate: own + peer compute lead warp arrive, both MMA arms wait.
    assert "bars.mb_tmem_dealloc.arrive()" in src and "bars.mb_tmem_dealloc.arrive_on_peer(cta_id_x ^ cutlass.Int32(1))" in src


def test_twin_is_default_off_and_names_the_sibling_file():
    from cudnn.sdpa.bwd import api_dsl

    assert api_dsl.STAGE2_2X2 is False
    assert api_dsl._SM100_STAGE2_FILE_2X2 == "sm100/bprop_d512_f16_2x2.py"
    assert Path(api_dsl._sm100_kernel_path(api_dsl._SM100_STAGE2_FILE_2X2)).is_file()
    assert api_dsl._SM100_STAGE2_FILE == "sm100/bprop_d512_f16.py"
