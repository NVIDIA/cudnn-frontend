# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The streamed bias ring's MULTICAST form on the Rubin (SM107) f16/bf16 small head-dim tiles -- ``TemplateParams.bias_mode = 3``
(``config_sm107.BIAS_MODE_STREAMED_MULTICAST``) on ``sm107/prefill_d128_f16.py``.

The (n, n+1) PAIR item's streamed ring (``bias_mode = 2``: one ``128 x TILE_N`` bias tile per kv tile through a 2- / 3-stage SMEM ring)
launched as a CLUSTER of two INDEPENDENT CTAs along the grid's chunk axis (cluster (1, 1, 2); CTA_MMA / CGA_M / CGA_N stay 1 -- no
collective MMA, no shared K / V ring).  The two CTAs decode the SAME (q tile, head) and ADJACENT chunks (c, c + 1), so they read the
SAME bias tile sequence: the rank-0 CTA issues every tile's boxes ONCE as a cta_group::1 ``.multicast::cluster`` TMA load whose bytes
land at the same SMEM offset in both CTAs and complete each CTA's OWN ``mb_bias_ring_full`` (every CTA arms its own), and every stage is
released to BOTH CTAs' ``mb_bias_ring_empty`` (init 2 x the softmax warps = 16: one elected local arrive + one elected ``.release.cta``
cluster arrive per softmax warp -- a READ-then-credit across CTAs takes the release form, never the relaxed one: the credit must be
ordered after the thread's generic loads of the stage it frees; same SASS on sm_107a, no drain).  A PADDED chunk (an odd chunk count) or a
SHORTER tail chunk runs the ring-only GHOST handshake for every fill
the leader issues beyond its own steps, so both rings stay in lockstep.  The L2 -> SMEM bias feed of the pair item is halved; O / Stats
are BITWISE the cga1 ring's (same tiles, same order, same FFMA).  Served with GLOBAL lengths only (per-row lengths would make the two
chunks' kv-tile sequences differ) under the NATURAL scheduler; a REQUEST-ONLY form (the adapter's ``bias_mode=3`` / the test seam):
``sm107_bias_mode_pick`` never picks it by itself, so no Capabilities row moves.

Three tiers: (1) CPU contract -- the pick returns the form on request only and declines it off the pair / off cga1, the config
factory's couplings (each raises with its message), the appended Cfg field, the ring's 16-count bars, the kernel's one const_expr axis,
the engine row's typed declines, the adapter twin; (2) host rendering pins (sm_107a trace-compile) -- the mode-2 renderings stay
byte-identical to their committed record (the multicast seams fold OUT of the cga1 ring), the multicast rendering keeps the register
split, no cluster-scope drain, spills no more than the mode-2 rendering, carries the predicated multicast issue (UTMALDG + one ghost
site), and its PTX carries the cta_group::1 ``.multicast::cluster`` form and NO ``cta_group::2`` (the form whose bytes land on the pair
leader only); (3) GPU cells on a Rubin device -- the fp64 oracle over the ring-wrap axis (n_kv 1 / 2 / 3 / 4 / 8), an even and an odd
chunk count (the padded peer), a shorter tail chunk paired with a full one, causal, a kv_mask, fp16, D 64, 2 stages; the BITWISE pins
against the cga1 ring and against the resident block; the typed declines on real graphs.
"""

import dataclasses
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from frost_test_utils import assert_no_new_spills, nvdisasm_candidates, requires_dsl, requires_rubin, run_sass_probe, sass_probe_source
import test_sdpa_bias_sm107 as _s13
import test_sdpa_bias_streamed_sm107 as _s15

pytestmark = [pytest.mark.L0, requires_dsl]

_ROW = "sdpa_fwd_prefill_sm107"
_KERNEL = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "sdpa" / "fwd" / "kernels" / "sm107" / "prefill_d128_f16.py"
_RENDERINGS = Path(__file__).resolve().parent / "renderings"
_MCAST = 3


@pytest.fixture(autouse=True)
def _sm107_target_for_metadata(monkeypatch):
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


@pytest.fixture
def _pins(monkeypatch):
    """Pin the bias FORM, the ring depth and the chunk length a knob-less plan gets (the config module's private, process-local test
    seams); ``None`` = the pick.  ``mode=3`` is the A/B's multicast arm."""
    from cudnn.sdpa.fwd import config_sm107 as c

    def _set(mode=None, stages=None, n_chunk=None):
        monkeypatch.setattr(c, "_BIAS_MODE_OVERRIDE_FOR_SWEEPS", mode)
        monkeypatch.setattr(c, "_BIAS_STAGES_OVERRIDE_FOR_SWEEPS", stages)
        monkeypatch.setattr(c, "_N_CHUNK_OVERRIDE_FOR_SWEEPS", n_chunk)

    return _set


@pytest.fixture
def _s13_rubin(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))


# --------------------------------------------------------------------------- tier 1: CPU contract


def test_the_pick_returns_the_multicast_form_on_request_only():
    """Never by itself (no Capabilities row moves); on request / the seam only where the pair item's cga1 ring fits."""
    from cudnn.sdpa.fwd import config_sm107 as c

    assert c.BIAS_MODE_STREAMED_MULTICAST == _MCAST and _MCAST in c.BIAS_MODES and c.BIAS_STREAMED_MODES == frozenset({2, 3}) and c.BIAS_MCAST_CTAS == 2
    for d, s_kv in ((32, 384), (32, 896), (32, 1024), (32, 4096), (64, 512), (64, 1024)):
        assert c.sm107_bias_mode_pick(d, 1, s_kv, pair=True) in (1, 2), (d, s_kv)
        assert c.sm107_bias_mode_pick(d, 1, s_kv, pair=True, request=_MCAST) == _MCAST, (d, s_kv)
        assert c.sm107_bias_stages_pick(d, 1, pair=True) == 3 and c.sm107_bias_stages_pick(d, 1, pair=True, request=2) == 2
    assert c.sm107_bias_mode_pick(32, 1, 1024, pair=False, request=_MCAST) == 0, "the one-batch-entry item's two CTAs read DIFFERENT bias rows"
    assert c.sm107_bias_mode_pick(64, 2, 512, pair=False, request=_MCAST) == 0, "cga2 is the collective geometry"
    assert c.sm107_bias_mode_pick(64, 1, 512, pair=False, request=_MCAST) == 0, "no ring at the 64-wide tile's cga1 one-batch-entry item"


def test_the_seam_pins_the_multicast_form(_pins):
    from cudnn.sdpa.fwd import config_sm107 as c

    _pins(mode=_MCAST)
    assert c.sm107_bias_mode_pick(32, 1, 384, pair=True) == _MCAST, "where the resident block would fit too (the A/B's cells)"
    assert c.sm107_bias_mode_pick(32, 1, 1024, pair=True) == _MCAST
    assert c.sm107_bias_mode_pick(32, 1, 1024, pair=False) == 0, "...never off the pair item"
    assert c.sm107_bias_mode_pick(32, 1, 1024, pair=True, request=2) == 2, "an explicit request outranks the seam"
    _pins()
    assert c.sm107_bias_mode_pick(32, 1, 1024, pair=True) == 2


def test_config_factory_derives_the_multicast_cfg_and_raises_on_each_coupling():
    """Every multicast field follows the form; each coupling moved alone raises with the failure it would produce (frost-kernels s5
    "bind the couplings"; an unreachable count is a hang at every shape)."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT
    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    for d, kw, stages in ((32, dict(n_chunk=4), 3), (32, dict(n_chunk=4, bias_stages=2), 2), (64, dict(n_chunk=2), 3), (32, dict(n_chunk=4, tile_n=64), 3)):
        cfg, _ = {32: c.make_cfg_d32, 64: c.make_cfg_d64}[d](TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=_MCAST, **kw))
        assert (cfg.BIAS_MODE, cfg.BIAS_MCAST_CTAS, cfg.STAGES_BIAS, cfg.BIAS_RESIDENT_TILES, cfg.BIAS_BLOCK_ROWS, cfg.BIAS_EMPTY_ARRIVERS) == (
            _MCAST,
            2,
            stages,
            0,
            128,
            8,
        ), (d, kw, cfg)
        assert (cfg.CTA_MMA, cfg.CGA_M, cfg.CGA_N) == (1, 1, 1)
        assert c.bias_smem_tiles(cfg) == stages and c.bias_resident_block_bytes(cfg) == stages * 128 * cfg.TILE_N * 2
        # the SAME SMEM as the cga1 ring of the same arm
        ring, _ = {32: c.make_cfg_d32, 64: c.make_cfg_d64}[d](TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=2, **kw))
        assert c.bias_resident_block_bytes(ring) == c.bias_resident_block_bytes(cfg) and ring.STAGES_BIAS == cfg.STAGES_BIAS
        for field, value, word in (
            ("BIAS_MCAST_CTAS", 1, "BIAS_MCAST_CTAS must be 2"),
            ("BIAS_MCAST_CTAS", 4, "BIAS_MCAST_CTAS must be 2"),
            ("N_CHUNK", 0, "PAIR item"),
            ("CTA_MMA", 2, "INDEPENDENT CTAs"),
            ("SCHEDULER_POLICY", SCHED_LPT, "NATURAL"),
            ("SEQ_KV_LENS_PRESENT", 1, "global lengths only"),
            ("SEQ_Q_LENS_PRESENT", 1, "global lengths only"),
            ("STAGES_BIAS", 4, "stages"),
            ("BIAS_RESIDENT_TILES", 1, "no resident block"),
        ):
            with pytest.raises(ValueError, match=re.escape(word)):
                c._validate_bias(dataclasses.replace(cfg, **{field: value}), "probe")
    # the other forms carry no cluster extent, and a cluster extent on them raises
    for mode in (0, 1, 2):
        other, _ = c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=mode, n_chunk=4 if mode else 0))
        assert other.BIAS_MCAST_CTAS == 1
        with pytest.raises(ValueError, match="BIAS_MCAST_CTAS"):
            c._validate_bias(dataclasses.replace(other, BIAS_MCAST_CTAS=2), "probe")
    with pytest.raises(ValueError, match="PAIR item"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=_MCAST))
    with pytest.raises(ValueError, match="global lengths only"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=_MCAST, n_chunk=4, seq_kv_lens_present=True))
    with pytest.raises(ValueError):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=2, bias_mode=_MCAST, n_chunk=4))
    with pytest.raises(ValueError, match="NATURAL"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=_MCAST, n_chunk=4, sched_policy=SCHED_LPT))
    with pytest.raises(ValueError, match="bias_mode=2 / 3"):
        c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=1, n_chunk=4, bias_stages=2))


def test_the_ring_bars_count_both_ctas_releases():
    """``mb_bias_ring_empty`` init = 8 own + 8 peer = 16 on the multicast ring: the kernel passes BIAS_EMPTY_ARRIVERS x BIAS_MCAST_CTAS (8 x 2;
    8 x 1 on the cga1 ring) through ``make_classic_bars(bias_ring_empty_arrivers=)``, whose SMEM allocations exist only inside a kernel --
    the arithmetic is pinned on the Cfg, the kwarg's validation (a ring is required) raises before any allocation, and the kernel's call is
    pinned by source."""
    import inspect

    from cudnn.sdpa.fwd import config_sm107 as c
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.kernels._common_blackwell import make_classic_bars

    cfg, _ = c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=_MCAST, n_chunk=4))
    ring, _ = c.make_cfg_d32(TemplateParams(dtype_qkv=2, cta_mma=1, bias_mode=2, n_chunk=4))
    assert cfg.BIAS_EMPTY_ARRIVERS * cfg.BIAS_MCAST_CTAS == 16 and ring.BIAS_EMPTY_ARRIVERS * ring.BIAS_MCAST_CTAS == 8
    with pytest.raises(ValueError, match="needs a ring"):
        make_classic_bars(ring, bias_ring_empty_arrivers=16)
    with pytest.raises(ValueError, match="needs a ring"):
        make_classic_bars(ring, bias_ring_stages=3, bias_ring_empty_arrivers=0)
    src = inspect.getsource(make_classic_bars)
    assert "ring_empty_count = CFG.BIAS_EMPTY_ARRIVERS if bias_ring_empty_arrivers is None else int(bias_ring_empty_arrivers)" in src
    assert "init_count=ring_empty_count, producer=Producer.THREAD" in src


def test_cfg_is_appended_and_the_kernel_carries_the_multicast_ring_as_one_const_expr_axis():
    """The Cfg gains ONE appended field; the kernel folds every multicast seam on ``_BIAS_MCAST`` (the ring seams stay on
    ``_BIAS_STREAMED``, which now names both forms): the rank read, the chunk remap at every decode (5 initial + 5 payload), the fill
    helper's predicated issue, the peer release beside the own one (the kv body and the ghost loop), the two ghost loops, the TMA-LDG
    drain, the cluster fence, the launch's cluster and the padded chunk count, the 16-count bars."""
    from cudnn.sdpa.fwd import config_sm107 as c

    fields = [f.name for f in dataclasses.fields(c._CfgSm107)]
    # the tail of the appended fields: STAGES_BIAS (the ring depth), BIAS_MAX_FORM (the row-max form) and S_SLOTS (the double-buffered S),
    # both appended by the base this arm was rebased onto, then this arm's BIAS_MCAST_CTAS -- each a later append, never an insertion
    assert (
        fields[-4:] == ["STAGES_BIAS", "BIAS_MAX_FORM", "S_SLOTS", "BIAS_MCAST_CTAS"] and c._CfgSm107.BIAS_MCAST_CTAS == 1
    ), "append-only; 1 renders every existing record byte-identically"
    src = _KERNEL.read_text()
    assert "_BIAS_STREAMED = CFG.BIAS_MODE in BIAS_STREAMED_MODES" in src and "_BIAS_MCAST = CFG.BIAS_MODE == BIAS_MODE_STREAMED_MULTICAST" in src
    assert "CGA_SIZE = CFG.CGA_M * CFG.CGA_N * CFG.BIAS_MCAST_CTAS" in src
    assert "+ (CGA_SIZE // CFG.CTA_MMA)" in src, "the MMA warps that credit the scheduler: one per CTA_MMA CTAs of the cluster"
    assert "bias_ring_empty_arrivers=CFG.BIAS_EMPTY_ARRIVERS * CFG.BIAS_MCAST_CTAS if _BIAS_STREAMED else None" in src
    assert "if cutlass.const_expr(CFG.CTA_MMA == 2 or _BIAS_MCAST):\n        cga_arrive()\n        cga_wait()" in src, "P4: the cluster fence after the inits"
    assert "bias_mcast_rank = cute.arch.block_idx_in_cluster() if cutlass.const_expr(_BIAS_MCAST) else None" in src
    assert (
        src.count("batch_idx = _mcast_chunk(batch_idx, bias_mcast_rank) if cutlass.const_expr(_BIAS_MCAST) else batch_idx") == 10
    ), "5 roles x (initial + payload)"
    assert src.count("bias_ring_state = _bias_ring_fill(") == 5, "the four kv-tile paths + the TMA-LDG ghost loop"
    assert src.count("ghost_tiles = _mcast_ghost_tiles(chunk_idx, bias_mcast_rank, n_rows_total, item_b)") == 2, "the TMA-LDG and the softmax ghost loops"
    assert (
        src.count("bars.mb_bias_ring_empty[bias_ring_state.idx].arrive_on_peer(bias_mcast_rank ^ cutlass.Int32(1), pred=nvvm.elect_sync(), release_cta=True)")
        == 2
    ), "the kv body's and the ghost loop's peer credit, both in the .release.cta form (a read-then-credit across CTAs)"
    assert "arrive_on_peer(bias_mcast_rank ^ cutlass.Int32(1), pred=nvvm.elect_sync())" not in src, "a relaxed peer credit on the ring"
    assert src.count("bars.mb_bias_ring_empty[bias_ring_state.idx].arrive()") == 2, "the kv body's own release + the ghost loop's"
    assert "issue_bias_ring_tile_multicast(" in src and "bias_mcast_rank == cutlass.Int32(0)" in src
    assert "cluster=(CFG.CTA_MMA, 1, CFG.BIAS_MCAST_CTAS)" in src
    assert "sched_batch = (-(-sched_batch // CFG.BIAS_MCAST_CTAS)) * CFG.BIAS_MCAST_CTAS if cutlass.const_expr(_BIAS_MCAST) else sched_batch" in src
    assert "is_cga_first_cta = (bias_mcast_rank == cutlass.Int32(0)) if cutlass.const_expr(_BIAS_MCAST) else (cta_id_x == cutlass.Int32(0))" in src
    assert (
        "    if cutlass.const_expr(_BIAS_MCAST):\n        for _bs in cutlass.range_constexpr(CFG.STAGES_BIAS):\n            bars.mb_bias_ring_empty[bias_ring_state.idx].wait(bias_ring_state.phase)"
        in src
    ), "P15: the drain"
    # the two tile_dsl halves: the cta_group::1 multicast load and the per-CTA arm + predicated issue
    from cudnn.frost.tile_dsl import bias as tb, tma as tt

    assert hasattr(tt, "tma_load_tile_multicast") and hasattr(tb, "issue_bias_ring_tile_multicast")


# (name, facts kwargs, knobs kwargs, accept?) with the seam at 3: the pair cells the multicast ring serves, and the typed declines.
_CELLS = [
    ("pair d32 S1024", dict(s_q=1024, s_kv=1024, **_s13._bias_kw(s_q=1024, s_kv=1024)), None, True),
    ("pair d32 S384 (the resident block would fit too)", dict(s_q=384, s_kv=384, **_s13._bias_kw(s_q=384, s_kv=384)), None, True),
    (
        "pair d32 S2048 causal with a kv_mask",
        dict(s_q=2048, s_kv=2048, causal=True, **_s13._bias_kw(s_q=2048, s_kv=2048), **_s13._mask_kw(s_kv=2048)),
        None,
        True,
    ),
    ("pair d64 S1024", dict(d_qk=64, d_v=64, s_q=1024, s_kv=1024, **_s13._bias_kw(s_q=1024, s_kv=1024)), None, True),
    ("pair d32 S1024 with per-batch lengths: the cga1 ring only", dict(s_q=1024, s_kv=1024, padded=True, **_s13._bias_kw(s_q=1024, s_kv=1024)), None, False),
    ("per-batch bias d32 S1024: no pair, no form on the seam", dict(s_q=1024, s_kv=1024, **_s13._bias_kw(b=2, s_q=1024, s_kv=1024)), None, False),
    ("one flattened row d32 S1024: no pair", dict(b=1, s_q=1024, s_kv=1024, **_s13._bias_kw(s_q=1024, s_kv=1024)), None, False),
]


@pytest.mark.parametrize("name, fkw, kkw, accept", _CELLS, ids=[c[0] for c in _CELLS])
def test_row_serves_the_multicast_ring_on_the_seam_and_declines_typed(name, fkw, kkw, accept, _pins):
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    _pins(mode=_MCAST)
    caps = _s13._caps()
    f = _s13._facts(**_s13._resolve_dtypes(fkw))
    knobs = SdpaFwdKnobs(**kkw) if kkw else None
    why = engines.mismatch(caps, f, knobs)
    assert (why is None) == accept, (name, why)
    if not accept:
        assert ("multicast bias ring" in why) == ("per-batch lengths" in name), (name, why)
    _pins()
    assert engines.mismatch(caps, f, knobs) is None or "lengths" not in name, "without the seam the cga1 ring serves the padded cell"


# (adapter kwargs, accept?, expected (bias_mode, bias_stages, cta_mma, n_chunk >= 2?)): the adapter honours bias_mode=3 on the pair at
# cga1 with global lengths and declines it typed elsewhere (never degraded).
_ADAPTER_CELLS = [
    # S_q % 256 == 0 at d32 picks the one-batch-entry item knob-less (config_sm107.sm107_n_chunk_pick's measured guard): the pair, and so the
    # multicast form, is reached there through an explicit n_chunk; at S 384 / 896 and at d64 the pick is the pair already.
    (dict(s_q=1024, s_kv=1024, bias="h", bias_mode=3, n_chunk=4), True, (3, 3, 1, True)),
    (dict(s_q=1024, s_kv=1024, bias="h", bias_mode=3, n_chunk=4, bias_stages=2), True, (3, 2, 1, True)),
    (dict(s_q=896, s_kv=896, bias="h", bias_mode=3), True, (3, 3, 1, True)),
    (dict(s_q=384, s_kv=384, bias="h", bias_mode=3), True, (3, 3, 1, True)),
    (dict(d=64, s_q=1024, s_kv=1024, bias="h", bias_mode=3), True, (3, 3, 1, True)),
    (dict(s_q=896, s_kv=896, bias="h"), True, (3, 3, 1, True)),  # knob-less: the multicast ring, the pair ring's MEASURED default at D 32
    (dict(s_q=896, s_kv=896, bias="h", seq_kv_lens_present=True), True, (2, 3, 1, True)),  # knob-less + per-batch lengths: the cga1 ring
    (dict(d=64, s_q=1024, s_kv=1024, bias="h"), True, (2, 3, 1, True)),  # knob-less at D 64: the cga1 ring (the multicast ring loses there)
    (dict(s_q=1024, s_kv=1024, bias="h", bias_mode=3), False, None),  # S_q % 256 == 0 knob-less = the one-batch-entry item: no pair, so no form 3
    (dict(s_q=1024, s_kv=1024, bias="b", bias_mode=3, n_chunk=4), False, None),  # a per-batch bias: no pair item
    (dict(s_q=1024, s_kv=1024, bias="h", b=1, bias_mode=3), False, None),  # one flattened row: no pair item
    (dict(s_q=1024, s_kv=1024, bias="h", bias_mode=3, n_chunk=4, cga=2), False, None),  # cga2 is the collective geometry
    (dict(s_q=1024, s_kv=1024, bias="h", bias_mode=3, n_chunk=4, seq_kv_lens_present=True), False, None),  # per-batch lengths break the lockstep
    (dict(s_q=1024, s_kv=1024, bias="h", bias_mode=3, n_chunk=0), False, None),  # the one-batch-entry item requested
    (dict(s_q=1024, s_kv=1024, bias="h", bias_mode=4), False, None),
]


@pytest.mark.parametrize(
    "kw, accept, want",
    _ADAPTER_CELLS,
    ids=[
        "d32_S1024_n4_mcast",
        "d32_S1024_n4_mcast_2_stages",
        "d32_S896_mcast",
        "d32_S384_mcast_where_the_block_fits",
        "d64_S1024_mcast",
        "d32_S896_knobless_is_the_multicast_ring",
        "d32_S896_knobless_per_batch_lengths_is_the_cga1_ring",
        "d64_S1024_knobless_is_the_cga1_ring",
        "d32_S1024_knobless_one_batch_entry_declines",
        "per_batch_bias_declines",
        "one_row_declines",
        "cga2_declines",
        "per_batch_lengths_decline",
        "one_batch_entry_item_declines",
        "mode_4_declines",
    ],
)
def test_adapter_check_support_is_the_rows_twin(_s13_rubin, kw, accept, want):
    kw = dict(kw)
    d = kw.pop("d", 32)
    form = kw.pop("bias")
    b = kw.pop("b", 4)
    h = 4
    lens = kw.pop("seq_kv_lens_present", False)
    extra = dict(seq_kv_lens_present=True) if lens else {}
    api = _s13._api(d, b=b, h=h, bias=((b if form == "b" else 1), h, kw["s_q"], kw["s_kv"]), **kw, **extra)
    if not accept:
        with pytest.raises((NotImplementedError, ValueError)):
            api.check_support()
        return
    api.check_support()
    params = api.template_params()
    assert (params.bias_mode, params.bias_stages, params.cta_mma, params.n_chunk >= 2) == want, params


def test_the_multicast_ring_is_the_pair_rings_measured_default_at_d32():
    """The pick (ONE arithmetic for the row, the heuristics and the adapter) returns the multicast ring where the PAIR item would run its
    cga1 ring at D 32 -- the pair's resident block does not hold S_kv -- and the caller vouches for GLOBAL lengths and the NATURAL
    scheduler (the two facts the cluster's shared kv-tile sequence needs); every other input keeps its pick: the block where it fits, the
    cga1 ring at D 64, under per-batch lengths, under any other scheduler, on the one-batch-entry item, and for every caller that passes
    neither fact (the default).  MEASURED on the 212-SM Rubin perf-node class (lock 2376 per job, 100 x 3 graph replay, control pair in
    process): +3.38 % over the knob-less pick at D 32 S 896 B 1 (the pair's n8 ring), +3.16 / +3.29 % over the n4 / n8 ring; the same
    arm-vs-arm delta on the pair ring at S 640 / 768 / 1024 (+2.0..+3.1 %, B 1 and B 2); -6.3..-10.1 % at D 64; -4.7 / -1.9 % against the
    resident block at S 384 / 640."""
    from cudnn.sdpa.fwd import config_sm107 as c

    assert c.SM107_PAIR_RING_MULTICAST_D_FLAVORS == frozenset({32})
    both = dict(global_lengths=True, natural=True)
    assert c.sm107_bias_mode_pick(32, 1, 896, pair=True, **both) == c.BIAS_MODE_STREAMED_MULTICAST
    assert c.sm107_bias_mode_pick(32, 1, 1152, pair=True, **both) == c.BIAS_MODE_STREAMED_MULTICAST
    assert c.sm107_bias_mode_pick(32, 1, 896, pair=True) == c.BIAS_MODE_STREAMED, "a caller that vouches for neither fact keeps the cga1 ring"
    assert c.sm107_bias_mode_pick(32, 1, 896, pair=True, global_lengths=True) == c.BIAS_MODE_STREAMED, "the NATURAL scheduler is required"
    assert c.sm107_bias_mode_pick(32, 1, 896, pair=True, natural=True) == c.BIAS_MODE_STREAMED, "global lengths are required"
    assert c.sm107_bias_mode_pick(32, 1, 768, pair=True, **both) == c.BIAS_MODE_RESIDENT, "the block where it holds S_kv (-1.9 % at S 640)"
    assert c.sm107_bias_mode_pick(32, 1, 384, pair=True, **both) == c.BIAS_MODE_RESIDENT
    assert c.sm107_bias_mode_pick(64, 1, 1024, pair=True, **both) == c.BIAS_MODE_STREAMED, "D 64 keeps the cga1 ring (-6.3..-10.1 %)"
    assert c.sm107_bias_mode_pick(32, 1, 896, pair=False, **both) == c.BIAS_MODE_STREAMED, "the one-batch-entry item's own ring"
    assert c.sm107_bias_mode_pick(32, 2, 896, pair=False, **both) in (c.BIAS_MODE_RESIDENT, c.BIAS_MODE_STREAMED), "cga2 is the collective geometry"
    assert c.sm107_bias_mode_pick(32, 1, 896, pair=True, request=2, **both) == c.BIAS_MODE_STREAMED, "a request is honoured"
    assert (
        c.sm107_bias_mode_pick(32, 1, 896, pair=True, request=3) == c.BIAS_MODE_STREAMED_MULTICAST
    ), "...and a request needs neither fact (the adapter declines them)"


# --------------------------------------------------------------------------- tier 2: host rendering pins (sm_107a trace-compile)


def _record():
    f = _RENDERINGS / "md5_fwd_d128_file_bias_streamed_sm_107a.txt"
    assert f.is_file(), f
    dsl, want = None, {}
    for ln in f.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(r"fwd_d128_file sm_107a (\S+) rc=0 cubin_md5=([0-9a-f]{32})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


_PROBE_BODY = (
    _s15._PROBE_BODY
    + 'print("BIAS_MCAST_CTAS", mod.CFG.BIAS_MCAST_CTAS)\nprint("BIAS_BOXES_PER_TILE", mod.CFG.BIAS_BOXES_PER_TILE)\nprint("CGA_SIZE", mod.CGA_SIZE)\n'
)
_COUNTS = {**_s15._COUNTS, "MEMBAR_GPU": ("MEMBAR.ALL.GPU",)}
# the streamed arms of test_sdpa_bias_streamed_sm107.py's rendering pins -> their TemplateParams (the record's arm names)
_STREAMED_RECORD_ARMS = {
    "d32_pair_3": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 2, "n_chunk": 4},
    "d32_pair_2": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 2, "n_chunk": 4, "bias_stages": 2},
    "d32_pair_mask_padded_causal": {
        "__d": 32,
        "cta_mma": 1,
        "dtype_qkv": 2,
        "bias_mode": 2,
        "n_chunk": 4,
        "kv_mask_form": 1,
        "seq_kv_lens_present": True,
        "window_right": 0,
    },
    "d32_one_batch_entry_2": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 2},
    "d32_one_batch_entry_3_requested": {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 2, "bias_stages": 3},
    "d64_pair_3": {"__d": 64, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 2, "n_chunk": 4},
    "d64_one_batch_entry_cga2_2": {"__d": 64, "cta_mma": 2, "dtype_qkv": 2, "bias_mode": 2},
}


def _probe(tmp_path, params, tag):
    return run_sass_probe(tmp_path, probe_src=sass_probe_source(_PROBE_BODY, _COUNTS), arch="sm_107a", params=params, tag=tag)


def test_streamed_record_is_committed_and_complete():
    dsl, want = _record()
    assert dsl and dsl.startswith("nvidia-cutlass-dsl "), dsl
    assert set(want) == set(_STREAMED_RECORD_ARMS), (sorted(want), sorted(_STREAMED_RECORD_ARMS))


@pytest.mark.parametrize("arm", list(_STREAMED_RECORD_ARMS))
def test_streamed_renderings_are_byte_identical_to_their_record(tmp_path, arm):
    """The multicast seams fold OUT of the cga1 ring: every bias_mode = 2 rendering traces byte-identically to the tree BEFORE the
    multicast form landed (every seam is a const_expr on BIAS_MODE == 3 or a Python-int factor equal to 1; host trace-compile)."""
    from cudnn.frost.buffers import cutedsl_state

    dsl, want = _record()
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: the cubin is a function of the DSL build, re-render the record")
    probe = _probe(tmp_path, _STREAMED_RECORD_ARMS[arm], f"streamed_{arm}")
    assert (
        probe.expect["BIAS_MODE"] == 2
        and probe.expect["BIAS_MCAST_CTAS"] == 1
        and probe.expect["CGA_SIZE"] == probe.expect["BIAS_MCAST_CTAS"] * _STREAMED_RECORD_ARMS[arm]["cta_mma"]
    )
    assert probe.cubin_md5 == want[arm], f"{arm}: cubin md5 {probe.cubin_md5} != the streamed record's {want[arm]} -- the cga1 ring's rendering changed"


@pytest.mark.parametrize(
    "d, extra, stages",
    [
        (32, {"n_chunk": 4}, 3),
        (32, {"n_chunk": 4, "bias_stages": 2}, 2),
        (32, {"n_chunk": 4, "kv_mask_form": 1, "window_right": 0}, 3),
        (32, {"n_chunk": 4, "tile_n": 64}, 3),
        (64, {"n_chunk": 4}, 3),
    ],
    ids=["d32_pair_3", "d32_pair_2", "d32_pair_mask_causal", "d32_pair_t64", "d64_pair_3"],
)
def test_multicast_renderings_keep_the_split_no_drain_and_the_predicated_issue(tmp_path, d, extra, stages):
    """Each multicast rendering traced for sm_107a keeps the per-role register split (USETMAXREG > 0), no cluster-scope drain (no CGAERRBAR,
    no MEMBAR.ALL.GPU: the peer release is the .release.cta cluster arrive, never .release.cluster), spills no more than the cga1 ring's
    rendering of the same arm, carries
    the packed bias add and the software row max, the bit-word form under a mask, a cluster of 2 and the ring's depth -- and ONE more
    bias-issue site than the cga1 ring (the ghost fill), each site predicated by rank, not duplicated: UTMALDG grows by exactly the boxes
    of one tile."""
    base = _probe(tmp_path, {"__d": d, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": 2, **extra}, f"ring_{d}_{'_'.join(sorted(extra))}")
    probe = _probe(tmp_path, {"__d": d, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": _MCAST, **extra}, f"mcast_{d}_{'_'.join(sorted(extra))}")
    assert probe.expect["BIAS_MODE"] == _MCAST and probe.expect["STAGES_BIAS"] == stages and probe.expect["BIAS_RESIDENT_TILES"] == 0
    assert probe.expect["BIAS_MCAST_CTAS"] == 2 and probe.expect["CGA_SIZE"] == 2 and probe.expect["BIAS_BLOCK_ROWS"] == 128
    assert probe.cubin_md5 != base.cubin_md5
    assert probe.stats["USETMAXREG"] > 0 and probe.stats["CGAERRBAR"] == 0 and probe.stats["MEMBAR_GPU"] == 0, probe.stats
    assert_no_new_spills(probe.stats, base.stats, tag=f"d{d} multicast {sorted(extra)} vs the cga1 ring: ")
    assert probe.stats["FMNMX3"] > 0 and probe.stats["FFMA2"] >= 64, probe.stats
    assert probe.stats["UTMALDG"] == base.stats["UTMALDG"] + probe.expect["BIAS_BOXES_PER_TILE"], (probe.stats, base.stats)
    if extra.get("kv_mask_form"):
        assert probe.stats["R2P"] > 0, probe.stats


_PTX_PROBE = r"""
import glob, json, os, re, sys
dump, params_json = sys.argv[1], sys.argv[2]
os.environ["CUTE_DSL_DUMP_DIR"] = dump
os.environ["CUTE_DSL_KEEP"] = "ptx"
os.environ["CUTE_DSL_ARCH"] = "sm_107a"
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
from cudnn.sdpa.fwd.config_sm100 import TemplateParams
params_kw = json.loads(params_json)
d = params_kw.pop("__d")
mod = _load_sm100_kernel_module((d, d), TemplateParams(**params_kw), rubin=True)
print("BIAS_BOXES_PER_TILE", mod.CFG.BIAS_BOXES_PER_TILE)
mod.compile(d_qk=d, d_v=d, has_lse=True, lse_kind="dense", has_bias=bool(mod.CFG.BIAS_MODE), has_kv_mask=bool(mod.CFG.KV_MASK_FORM))
ptxs = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
if not ptxs:
    print("FAIL no ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
ptx = open(ptxs[-1]).read()
pats = {
    "MCAST": r"cp\.async\.bulk\.tensor\.\dd\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes\.multicast::cluster",
    "CTA_GROUP2": r"cta_group::2",
    "LOCAL_TMA": r"cp\.async\.bulk\.tensor\.\dd\.shared::cta\.global\.tile\.mbarrier::complete_tx::bytes",
    "CTARANK": r"%cluster_ctarank",
    "RELAXED_CLUSTER_ARRIVE": r"mbarrier\.arrive\.relaxed\.cluster\.shared::cluster\.b64",
    "RELEASE_CTA_CLUSTER_ARRIVE": r"mbarrier\.arrive\.release\.cta\.shared::cluster\.b64",
    "RELEASE_CLUSTER": r"mbarrier\.arrive\.release\.cluster",
    "CLUSTER_BARRIER": r"barrier\.cluster\.(arrive|wait)",
}
for k, p in pats.items():
    print("PTX", k, len(re.findall(p, ptx)))
"""


@pytest.mark.parametrize("mode", [2, _MCAST], ids=["cga1_ring", "multicast"])
def test_ptx_carries_the_cta_group_1_multicast_form_on_the_multicast_rendering_only(tmp_path, mode):
    """The byte-routing pin that no numerics test shows: the multicast rendering's bias loads are the cta_group::1
    ``.multicast::cluster`` form (the bytes and the complete_tx land on EVERY destination CTA's own barrier), never a ``cta_group::2``
    token (the form whose bytes land on the pair leader only -- a peer waiting its own barrier under it hangs); exactly the fill
    sites x the boxes of a tile (3 sites: the prologue, the mainloop, the ghost loop); the cluster rank is read; the peer release is
    the .release.cta cluster arrive (a READ-then-credit across CTAs: zero relaxed cluster arrives, zero .release.cluster); a cluster
    barrier pair exists.  The cga1 ring's rendering carries none of them."""
    from frost_test_utils import arch_known_to_the_dsl

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a")
    params = {"__d": 32, "cta_mma": 1, "dtype_qkv": 2, "bias_mode": mode, "n_chunk": 4}
    dump = tmp_path / f"ptx_{mode}"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _PTX_PROBE, str(dump), json.dumps(params)], capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    got = {ln.split()[1]: int(ln.split()[2]) for ln in proc.stdout.splitlines() if ln.startswith("PTX ")}
    boxes = next(int(ln.split()[1]) for ln in proc.stdout.splitlines() if ln.startswith("BIAS_BOXES_PER_TILE "))
    assert got["CTA_GROUP2"] == 0, got
    if mode == _MCAST:
        assert got["MCAST"] == 3 * boxes and got["CTARANK"] == 1 and got["CLUSTER_BARRIER"] == 2, got
        # The ring's peer credit is the .release.cta cluster arrive (a READ-then-credit across CTAs: ordered after the thread's generic
        # loads of the stage it frees), NEVER the relaxed one, and never .release.cluster (a GPU-scope drain per arrive).  The count:
        # the scheduler credit's 6 sites (read_tile_id_arrive at CGA_SIZE 2 emits the SAME instruction; the pre-fix rendering carried
        # exactly those 6 beside 4 relaxed peer credits) + the kv body's and the ghost loop's peer credit x the two softmax warp-group
        # instantiations = 10.
        assert got["RELAXED_CLUSTER_ARRIVE"] == 0 and got["RELEASE_CTA_CLUSTER_ARRIVE"] == 6 + 4 and got["RELEASE_CLUSTER"] == 0, got
    else:
        assert got["MCAST"] == 0 and got["CTARANK"] == 0 and got["CLUSTER_BARRIER"] == 0, got
        assert got["RELAXED_CLUSTER_ARRIVE"] == 0 and got["RELEASE_CTA_CLUSTER_ARRIVE"] == 0 and got["RELEASE_CLUSTER"] == 0, got
    assert got["LOCAL_TMA"] > 0, got


# --------------------------------------------------------------------------- tier 3: GPU cells (a Rubin device)

# The multicast ring's degenerate-input matrix.  The pair item on a batch-broadcast bias; ``b`` = the flattened rows, ``n_chunk`` the
# chunk, so the chunk count is ceil(b / n_chunk): EVEN (every cluster has two live chunks) or ODD (a PADDED peer runs the ghost
# handshake); a chunk count whose last chunk is SHORTER than n_chunk pairs a full chunk with a short one (ghost steps on a LIVE peer).
# The ring-wrap axis at the 128-wide tile: S_kv 128 / 256 / 384 / 512 / 1024 = n_kv 1 / 2 / 3 / 4 / 8 (the heuristics' width pick for a
# bias alone at one / two tiles is 64, so those cells wrap the 16 KiB stage ring instead -- both widths covered).  ``mode`` 3 = the seam.
_CASES = {
    "nkv1_S128_b8_even": dict(b=8, h=2, h_kv=2, s_q=128, s_kv=128, n_chunk=4, mode=3),
    "nkv2_S256_b8_even": dict(b=8, h=2, h_kv=2, s_q=256, s_kv=256, n_chunk=4, mode=3),
    "nkv3_S384_b8_even": dict(b=8, h=2, h_kv=2, s_q=384, s_kv=384, n_chunk=4, mode=3),
    "nkv4_S512_b8_even": dict(b=8, h=2, h_kv=2, s_q=512, s_kv=512, n_chunk=4, mode=3),
    "nkv8_S1024_b8_even": dict(b=8, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=4, mode=3),
    "nkv8_S1024_b4_odd_padded_peer": dict(b=4, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=4, mode=3),
    "nkv8_S1024_b12_odd_padded_peer": dict(b=12, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=4, mode=3),
    "nkv8_S1024_b6_short_live_peer": dict(b=6, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=4, mode=3),
    "nkv8_S1024_b10_short_tail_and_padded": dict(b=10, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=4, mode=3),
    "nkv8_S1024_b8_causal": dict(b=8, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=4, mode=3, causal=True),
    "nkv8_S1024_b8_2_stages": dict(b=8, h=2, h_kv=2, s_q=1024, s_kv=1024, n_chunk=4, mode=3, stages=2),
    "nkv16_S2048_b6_causal_n2": dict(b=6, h=2, h_kv=2, s_q=2048, s_kv=2048, n_chunk=2, mode=3, causal=True),
    "head_broadcast_bias_S1024_b8": dict(b=8, h=4, h_kv=4, s_q=1024, s_kv=1024, bias="1", n_chunk=4, mode=3),
    "d64_nkv8_S1024_b8": dict(b=8, h=2, h_kv=2, s_q=1024, s_kv=1024, d=64, n_chunk=4, mode=3),
    "d64_nkv4_S512_b6_short_live_peer_causal": dict(b=6, h=2, h_kv=2, s_q=512, s_kv=512, d=64, n_chunk=4, mode=3, causal=True),
    # the D 64 twin of the resident-block comparison: the pair item's 128-row block holds S_kv <= 384 at D 64 (sm107_bias_resident_s_kv_max),
    # so the "where it fits" cell sits at S 384; the S 512 cell above runs the ring forms only
    "d64_nkv3_S384_b6_short_live_peer_causal": dict(b=6, h=2, h_kv=2, s_q=384, s_kv=384, d=64, n_chunk=4, mode=3, causal=True),
    "pick_n_chunk_S896_b8": dict(b=8, h=2, h_kv=2, s_q=896, s_kv=896, mode=3),  # the heuristics' chunk pick (the pair at S_q % 256 != 0), the seam's form
}


# Cells the multicast form DECLINES by its own contract (global lengths only): the harness serves a dense S_kv % 128 != 0 tail through the
# padding path (per-batch lengths synthesised at build, test_sdpa_bias_sm107._build), which the form's typed decline refuses -- the cga1 ring
# (bias_mode 2) serves the same cell.  Not in _CASES: a decline is pinned as a decline, never run as an oracle cell.
_DECLINED_CASES = {
    "nkv8_S1000_tail_tile_b8": dict(b=8, h=2, h_kv=2, s_q=1000, s_kv=1000, n_chunk=4, mode=3),
}


def _case_inputs(case, dtype, *, kv_mask_p=None, table=None):
    return _s13._case_inputs(case, dtype, kv_mask_p=kv_mask_p, table=_CASES if table is None else table)


def _run(q, k, v, kw, c, _pins, *, mode=None, stages=None, n_chunk=None):
    _pins(
        mode=c.get("mode") if mode is None else mode,
        stages=c.get("stages") if stages is None else stages,
        n_chunk=c.get("n_chunk") if n_chunk is None else n_chunk,
    )
    return _s13._run(q, k, v, **kw)


@requires_rubin
@pytest.mark.parametrize("case", list(_CASES))
def test_multicast_ring_matches_the_fp64_oracle(case, _pins):
    """The multicast ring over its degenerate-input matrix (bf16): O and Stats against the fp64 oracle, keyless rows exactly 0 / -inf,
    two launches bitwise; every cell runs the pair item at cga1 (the seam pins the form)."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    out = _run(q, k, v, kw, c, _pins)
    assert out["knobs"].cga == 1, (case, out["knobs"])
    _s13._check(out, ref)


@requires_rubin
@pytest.mark.parametrize("case", ["nkv8_S1024_b8_even", "nkv8_S1024_b4_odd_padded_peer", "d64_nkv8_S1024_b8"])
def test_multicast_ring_matches_the_fp64_oracle_fp16(case, _pins):
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.float16)
    out = _run(q, k, v, kw, c, _pins)
    _s13._check(out, ref)


@requires_rubin
@pytest.mark.parametrize("case", ["nkv8_S1024_b8_even", "nkv8_S1024_b12_odd_padded_peer", "nkv8_S1024_b8_causal", "d64_nkv8_S1024_b8"])
def test_multicast_ring_composes_with_the_kv_mask(case, _pins):
    """The bit-word kv_mask moves no kv-tile bound (it masks columns), so the two CTAs' sequences stay identical."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16, kv_mask_p=0.3)
    out = _run(q, k, v, kw, c, _pins)
    _s13._check(out, ref)


@requires_rubin
@pytest.mark.parametrize("case", list(_CASES))
def test_multicast_is_bitwise_the_cga1_ring(case, _pins):
    """Same tiles in the same order, the bias through the same FFMA from the same 128-byte swizzled row slice: O / Stats are BITWISE the
    cga1 ring's -- only WHICH CTA issues the TMA and the barrier counts differ."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    mcast = _run(q, k, v, kw, c, _pins, mode=_MCAST)
    _s13._check(mcast, ref)
    ring = _run(q, k, v, kw, c, _pins, mode=2)
    assert torch.equal(mcast["o"], ring["o"]) and torch.equal(mcast["stats"], ring["stats"]), f"{case}: the multicast ring differs from the cga1 ring"


@requires_rubin
@pytest.mark.parametrize("case", ["nkv3_S384_b8_even", "nkv4_S512_b8_even", "d64_nkv3_S384_b6_short_live_peer_causal"])
def test_multicast_is_bitwise_the_resident_block_where_it_fits(case, _pins):
    """Where the pair item's resident block holds S_kv (D 32: <= 768, D 64: <= 384 at the 128-wide tile), the multicast ring's O / Stats are
    BITWISE the block's -- the same bias words through the same FFMA; only the staging differs."""
    c = _CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16)
    mcast = _run(q, k, v, kw, c, _pins, mode=_MCAST)
    _s13._check(mcast, ref)
    resident = _run(q, k, v, kw, c, _pins, mode=1)
    assert torch.equal(mcast["o"], resident["o"]) and torch.equal(mcast["stats"], resident["stats"]), f"{case}: the multicast ring differs from the block"


@requires_rubin
@pytest.mark.parametrize("case", list(_DECLINED_CASES))
def test_multicast_declines_the_padded_dense_tail_the_cga1_ring_serves(case, _pins):
    """A dense S_kv % 128 != 0 tail reaches the standalone adapter with per-batch lengths (the harness pads it), which the multicast form
    declines TYPED (global lengths only: one kv-tile sequence for both CTAs of the cluster); the cga1 ring (bias_mode 2) serves the very
    same inputs and matches the fp64 oracle."""
    c = _DECLINED_CASES[case]
    q, k, v, kw, ref = _case_inputs(case, torch.bfloat16, table=_DECLINED_CASES)
    with pytest.raises((NotImplementedError, RuntimeError), match=r"global lengths|per-batch lengths|no unsplit"):
        _run(q, k, v, kw, c, _pins, mode=_MCAST)
    ring = _run(q, k, v, kw, c, _pins, mode=2)
    _s13._check(ring, ref)


@requires_rubin
def test_served_and_declined_on_real_graphs(_pins):
    """With the seams at form 3 / chunk 4: a batch-broadcast bias on 8 rows at S_kv 1024 is SERVED by the FROST row (the multicast ring,
    cga1); the same graph with per-batch KV lengths is a typed decline of the row (the cga1 ring serves it once the seam is off); a
    per-batch bias (no pair item) is declined on the seam."""
    import cudnn

    torch.manual_seed(7)

    def plans(b, s, bias_b, *, kv_lens=None):
        q, k, v = (_s13._bhsd(b, 2, s, 32, torch.bfloat16) for _ in range(3))
        bias = torch.randn(bias_b, 2, s, s, device="cuda").to(torch.bfloat16)
        lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda").view(b, 1, 1, 1) if kv_lens else None
        g, *_ = _s13._build(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(32),
            bias=bias,
            seq_len_kv=lens,
            seq_len_q=(torch.full((b, 1, 1, 1), s, dtype=torch.int32, device="cuda") if lens is not None else None),
        )
        try:
            g.create_execution_plans([cudnn.heur_mode.A])
            names, ours = _s13._frost_plans(g)
            return names, ours, [g.plans[i].knobs.cga for i in ours]
        except cudnn.cudnnGraphNotSupportedError:
            return [], [], []

    _pins(mode=_MCAST, n_chunk=4)
    names, ours, cgas = plans(8, 1024, 1)
    assert ours and set(cgas) == {1}, ("served by the multicast ring at cga1", names, cgas)
    names, ours, _ = plans(8, 1024, 1, kv_lens=[1024, 500, 1024, 0, 1024, 1024, 700, 1024])
    assert not ours, ("per-batch lengths: the multicast ring declines typed", names)
    names, ours, _ = plans(8, 1024, 8)
    assert not ours, ("a per-batch bias has no pair item: the seam's form does not fit", names)
    _pins()
    names, ours, _ = plans(8, 1024, 1, kv_lens=[1024, 500, 1024, 0, 1024, 1024, 700, 1024])
    assert ours, ("the cga1 ring serves the padded graph once the seam is off", names)


import math  # noqa: E402  (used by the real-graph cell above)
