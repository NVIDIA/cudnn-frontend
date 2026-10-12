# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm100_d256``: the SM100 / SM103 (cc 10.0-10.6) d=256 bf16 / fp16 SDPA backward on the 2x2 tcgen05 datapath.

The Rubin half chain (``api_dsl_sm107``) over the shared 2x2-datapath body ``kernels/bprop_d256_2x2_f16.py`` at
``datapath_2x2_profile = 1``.  Every capability the row claims gets an ACCEPT test through the graph API with the engine
PINNED (the bf16 d256 graph has a native backend competitor, cuDNN engine 5 on B200 / 9.26) against the sm107 suite's fp64
oracle, every decline a REJECT probe on a REAL graph with a faked cc 10.0 device.  The 2x2-specific detectors: the
poisoned-workspace cases (the kv WRITE-PAIR invariant: a 128-row block derives its q range from its 256-row pair), the
two-launch bitwise + race test (the first ``Producer.LEADER_RELEASE`` consumer in the tree), the host-only mbarrier
ledger pin (every ``MBarrier`` in ``_make_bars`` against the config ledger), and the sm_100a SASS pins (0 / 0 spills at
176 / 152, the 40 UTCHMMA / 3 LDTM census, no GPU-scope drain, every TMEM load before the arrive that frees its slot).

Profile 2 (the Rubin interleaved twin) is trace-compiled here for Rule S6 as far as the installed DSL allows: its
descriptor version 1 needs the ``tcgen05_mma_smem_desc_v2`` intrinsic (DSL >= 4.8.0); on a 4.7.0 DSL the case SKIPS with
that reason rather than failing.  On the Rubin board (internal DSL with sm_107a) the trace, the sm_107a SASS rows for
both profiles and the twin's GPU matrix (``test_sdpa_bwd_dsl_sm107.py -k twox2``) all ran green on 2026-10-01.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import cudnn
from frost_test_utils import _SM, arch_known_to_the_dsl, assert_no_new_spills, nvdisasm_candidates, requires_dsl, requires_pre_rubin_blackwell, select_engine
from test_sdpa_bwd_dsl_sm107 import (
    _MASK_POISON_CASES,
    _TOL,
    _adapter,
    _bshd_empty,
    _build_graph,
    _causal_keep,
    _check,
    _code_lines,
    _code_only,
    _half_bwd_graph,
    _mask_case_kw,
    _padded_keep,
    _reference64,
)

pytestmark = [pytest.mark.L0, requires_dsl]

_ENGINE = "sdpa_bwd_sm100_d256"
_FAMILY_NAME = "frost_sdpa_bwd"
_SLOT = 7  # engines/manifest.py: EngineSlot(7, opt_in=True) -> FROST_SDPA_BWD_ID_BASE + 7 (slot 6 is sdpa_bwd_sm107_mxfp8)
_ENGINE_ID = 20_607
_D = 256
_SM100_CC = (10, 0)
_SM_RANGE = (100, 106)  # the SM100 MXFP8 row's range; 107+ is the native Rubin row's
_DTYPES = (torch.bfloat16, torch.float16)
_DTYPE_IDS = ("bf16", "fp16")
_KERNEL_FILE = "bprop_d256_2x2_f16.py"
_TAG = "sdpa_bwd_sm100_d256_main"


def _kernel_path():
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path

    return _sm100_kernel_path(_KERNEL_FILE)


def _kernel_source():
    with open(_kernel_path(), encoding="utf-8") as fh:
        return fh.read()


def _load_kernel(profile=1, **params):
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams

    params.setdefault("dtype_qkv", DTYPE_BF16)
    return load_template(_kernel_path(), TemplateParams(datapath_2x2_profile=profile, **params), tag=f"{_TAG}_p{profile}")


def _spec(name=_ENGINE):
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == name), None)
    assert spec is not None, f"{name} is not in cudnn.sdpa.bwd.engines.ENGINE_SPECS"
    return spec


# =========================================================================== registration / capabilities (host)


def test_engine_is_registered_and_opt_in():
    from cudnn.engines.engine_ids import FROST_SDPA_BWD_ID_BASE
    from cudnn.engines.manifest import MANIFEST

    _spec()
    fam = next(f for f in MANIFEST if f.name == _FAMILY_NAME)
    assert _ENGINE in fam.slots, f"{_ENGINE} has no manifest slot (append slot {_SLOT}; never reuse a retired slot)"
    slot = fam.slots[_ENGINE]
    assert slot.slot == _SLOT and slot.opt_in, "new engines stay opt-in until they earn arch coverage + benchmarks"
    assert FROST_SDPA_BWD_ID_BASE + slot.slot == _ENGINE_ID
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    names = [s.name for s in ENGINE_SPECS]
    # Append-only: this row ranks after every row that existed when it landed (the position is the rank in the ranked plan
    # list, never the id); rows appended later (the cc 10.7 d512 row) rank after it.  A membership + order pin, not a
    # ``[-1]`` pin, so the next appended row does not break this one.
    assert names.index(_ENGINE) > names.index("sdpa_bwd_sm107_fp8"), names


def test_capabilities_match_what_is_implemented():
    """The row claims exactly what the accept tests run; every deferral is pinned False with its reject test."""
    c = _spec().capabilities
    assert (c.sm_lo, c.sm_hi) == _SM_RANGE, "SM100 / SM103 (cc 10.0-10.6); the native Rubin row owns 107+"
    assert c.d == frozenset({_D}) and not c.d_envelope and not c.dqk_ge_dv, "exact d_qk == d_v == 256 (the body hardcodes the 256-wide tiles)"
    assert c.dtypes == frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})
    assert not c.is_fp8 and not c.is_mxfp8 and c.out_dtypes == frozenset()
    assert c.causal and c.bottom_right and c.swa and c.gqa
    assert not c.right_band_widening and not c.thd and not c.thd_declared_totals and not c.cu_seq_len
    assert not c.bias and not c.dbias and not c.decode
    assert c.layouts == frozenset({"bshd"})
    assert not c.tile_ms and not c.tile_ns, "fixed geometry: no tile axis, the heuristics list {} as the complete record"
    for deferred in ("padded", "sink", "dsink", "deterministic"):
        assert not getattr(c, deferred), f"{deferred} is deferred: claim it together with its accept test here and the tracker line"
    for never in (
        "dropout",
        "score_mod",
        "paged_kv",
        "alibi",
        "block_mask",
        "rng_dump",
        "score_max",
        "score_sum_exp",
        "dynamic_scale",
        "unfuse_fma",
        "seq_q_trim",
    ):
        assert not getattr(c, never), never


def test_adapter_is_the_rubin_chain_over_the_2x2_body():
    """The adapter class, its kernel file / tag / profile, its explicit (256, 256) stage-3 tile and its 256-row trim granularity."""
    from cudnn.sdpa.bwd import config_d256_2x2 as c2
    from cudnn.sdpa.bwd.api_dsl_sm100_d256 import SdpaBwdDslSm100D256
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107, _stage3_cgrp_tile_mn
    from cudnn.sdpa.bwd.engines import _adapter as adapter_of

    assert adapter_of("SdpaBwdDslSm100D256") is SdpaBwdDslSm100D256 and issubclass(SdpaBwdDslSm100D256, SdpaBwdDslSm107)
    api = _adapter(SdpaBwdDslSm100D256)
    assert api._NAME == _ENGINE and api._kernel_file() == _KERNEL_FILE and api._template_tag() == _TAG
    assert api._datapath_2x2_profile() == c2.PROFILE_SM100 == 1
    assert api._template_params().datapath_2x2_profile == 1
    for sm in (100, 103, 106):
        assert api._stage3_tile_mn(sm) == (256, 256), "the d256 tile is passed explicitly on the SM100 line"
        assert _stage3_cgrp_tile_mn(sm, 256) == (512, 512), "and _stage3_cgrp_tile_mn itself stays the Rubin-line rule"
    mod = _load_kernel(1)
    assert (
        api._stage3_gran(mod) == c2.kv_pad_rows_2x2(mod.CFG) == 256 == mod._KV_WRITE_ROWS
    ), "the stage-3 granularity is the kv WRITE PAIR, not the 128-row block"
    assert mod._KV_BLOCK_ROWS == 128 and mod._Q_WRITE_TILES == 2 and mod.DESC_VERSION == 0
    p_dk, p_dq = api._stage3_records(mod, (256, 256))
    assert p_dk.cgrp_tile_mn == p_dq.cgrp_tile_mn == (256, 256) and p_dk.causal_gran == p_dq.causal_gran == 256


# =========================================================================== graph builders / probes (host, fake cc 10.0)


def _decline_reason(monkeypatch, engine=_ENGINE, cc=_SM100_CC, **kw):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    spec = _spec(engine)
    try:
        g, _t, _outs = _half_bwd_graph(**kw)
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as e:
        return f"frontend refused the graph: {e}"
    facts = ga.analyze(g)
    if facts is None:
        return "analyzer did not recognise the graph"
    return mismatch(spec.capabilities, facts)


def _eligible(monkeypatch, cc=_SM100_CC, **kw):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd import engines as bwd_engines

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    g, _t, _outs = _half_bwd_graph(**kw)
    return {s.name for s in bwd_engines.ENGINE_SPECS if bwd_engines.analyze_for(s, g, None)[1] is None}


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(dt=torch.float16),
        dict(scale=None),
        dict(use_causal_mask=True),
        dict(use_causal_mask_bottom_right=True),
        dict(use_causal_mask=True, diagonal_band_left_bound=256),
        dict(hq=8, hkv=2),
        dict(hq=16, hkv=1),
        dict(sq=500, skv=500),
        dict(sq=768, skv=1280),
        dict(sq=500, skv=1024, use_causal_mask_bottom_right=True),
    ],
    ids=["dense-bf16", "dense-fp16", "default-scale", "causal", "bottom-right", "swa", "gqa-r4", "mqa-r16", "non-tile-S", "768x1280", "bottom-right-ragged-sq"],
)
def test_served_graph_passes_the_row_probe(monkeypatch, kw):
    assert _decline_reason(monkeypatch, **kw) is None


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)], ids=["sm100", "sm103"])
def test_d256_half_graph_is_served_by_exactly_this_row_on_the_sm100_line(monkeypatch, cc):
    """No other python row serves the bf16 d256 backward on cc 10.0 / 10.3 (the d512 row's envelope floor is exclusive at
    256, the MXFP8 row is E4M3-only, the Rubin rows start at 107): the plan list offers this row and the backend's."""
    assert _eligible(monkeypatch, cc=cc) == {_ENGINE}
    assert _eligible(monkeypatch, cc=cc, use_causal_mask=True, hq=8, hkv=2) == {_ENGINE}


@pytest.mark.parametrize("d", [128, 264, 512])
def test_reject_other_head_dims(monkeypatch, d):
    assert _decline_reason(monkeypatch, d=d) is not None


def test_reject_rectangular_head_dims(monkeypatch):
    assert _decline_reason(monkeypatch, d=256, d_v=128) is not None


def test_reject_gqa_ratio_not_integer(monkeypatch):
    assert _decline_reason(monkeypatch, hq=6, hkv=4) is not None


def test_reject_bias(monkeypatch):
    assert _decline_reason(monkeypatch, bias=True) is not None


def test_reject_right_band_widening(monkeypatch):
    """``diagonal_band_right_bound`` alone (passing use_causal_mask with it forces the bound back to 0)."""
    assert _decline_reason(monkeypatch, diagonal_band_right_bound=64) is not None


def test_reject_sink(monkeypatch):
    assert _decline_reason(monkeypatch, sink=True) is not None


def test_reject_thd(monkeypatch):
    assert _decline_reason(monkeypatch, thd=True) is not None


def test_reject_decode_shaped(monkeypatch):
    assert _decline_reason(monkeypatch, sq=1) is not None


def test_reject_non_bshd_layout(monkeypatch):
    from test_sdpa_bwd_dsl_sm107 import _bhsd_stride

    assert _decline_reason(monkeypatch, stride_fn=_bhsd_stride) is not None


def test_padding_mask_follows_the_padded_claim(monkeypatch):
    claimed = _spec().capabilities.padded
    assert (_decline_reason(monkeypatch, padded=True) is None) == claimed


def test_deterministic_follows_the_claim(monkeypatch):
    claimed = _spec().capabilities.deterministic
    assert (_decline_reason(monkeypatch, use_deterministic_algorithm=True) is None) == claimed


@pytest.mark.parametrize("cc", [(10, 7), (11, 0), (12, 0), (8, 0), (9, 0)], ids=["sm107", "sm110", "sm120", "sm80", "sm90"])
def test_reject_other_arch_lines(monkeypatch, cc):
    reason = _decline_reason(monkeypatch, cc=cc)
    assert reason is not None and "SM100-106" in reason, reason


def test_reject_quantized_graphs(monkeypatch):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: _SM100_CC)
    facts = ga.SdpaGraphFacts(
        is_backward=True, b=1, h_q=2, h_kv=2, s_q=256, s_kv=256, d_qk=256, d_v=256, dtype=cudnn.data_type.FP8_E4M3, is_fp8=True, device_cc=_SM100_CC
    )
    assert "serves only" in (mismatch(_spec().capabilities, facts) or "")


# =========================================================================== GPU accept: the chain against the fp64 oracle, engine PINNED


def _run(
    b=2,
    hq=2,
    hkv=None,
    sq=512,
    skv=512,
    dt=torch.bfloat16,
    keep=None,
    omit_scale=False,
    seed=0,
    poison=None,
    runs=1,
    seq_lens=None,
    ws_poison=None,
    attn_scale=None,
    **sdpa_kwargs,
):
    """The sm107 suite's driver with THIS row pinned: build, pin, execute ``runs`` times, hand back every run's (dQ, dK, dV)
    plus the fp64 oracle.  Inputs unit-normal on a CPU generator; ``poison`` pre-fills the outputs, ``ws_poison`` (a byte)
    the WORKSPACE before every run; ``attn_scale`` None = 1/sqrt(d) on graph and oracle alike.  ``omit_scale`` leaves
    attn_scale off the graph (no scaling, 1.0) and pre-scales Q by 1/sqrt(d) so the logits keep their usual range."""
    from test_sdpa_bwd_dsl_sm107 import _Run

    hkv = hq if hkv is None else hkv
    group = hq // hkv
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(bb, s, h):
        return torch.randn(bb, s, h, _D, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, do = draw(b, sq, hq), draw(b, sq, hq)
    k, v = draw(b, skv, hkv), draw(b, skv, hkv)
    if omit_scale:
        q.mul_(_D**-0.5)
    if seq_lens is not None:
        keep = _padded_keep(sq, skv, *seq_lens) if keep is None else (keep & _padded_keep(sq, skv, *seq_lens))
    o64, lse64, all_masked, dq_r, dk_r, dv_r = _reference64(q, k, v, do, keep, group, scale=1.0 if omit_scale else attn_scale)
    o = _bshd_empty(b, sq, hq, _D, dt)
    o.copy_(o64.to(dt))
    lse = lse64.float()
    if all_masked is not None:
        lse = lse.masked_fill(all_masked, 0.0)
    if seq_lens is not None:
        sdpa_kwargs["padded"] = True
    graph_scale = None if omit_scale else ("default" if attn_scale is None else attn_scale)
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, dt=dt, scale=graph_scale, **sdpa_kwargs)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    assert g._compiled_plans[g._plan_index]._prepared is not None, "the row lowers through the prepared-launch contract"
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    dq, dk, dv = _bshd_empty(b, sq, hq, _D, dt), _bshd_empty(b, skv, hkv, _D, dt), _bshd_empty(b, skv, hkv, _D, dt)
    pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: lse.unsqueeze(-1).contiguous(), dq_t: dq, dk_t: dk, dv_t: dv}
    if seq_lens is not None:
        pack[t["seq_len_q"]] = torch.tensor(seq_lens[0], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
        pack[t["seq_len_kv"]] = torch.tensor(seq_lens[1], dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    outs = []
    for _ in range(runs):
        if poison is not None:
            for x in (dq, dk, dv):
                x.fill_(poison)
        if ws_poison is not None:
            ws.fill_(ws_poison)
        g.execute(pack, ws)
        torch.cuda.synchronize()
        outs.append(tuple(x.clone() for x in (dq, dk, dv)))
    return _Run(outs, (dq_r, dk_r, dv_r), dt)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_dense(dt):
    _run(dt=dt).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_causal_dtypes(dt):
    _run(dt=dt, keep=_causal_keep(512, 512), use_causal_mask=True).check()


@requires_pre_rubin_blackwell
def test_causal_bottom_right():
    _run(keep=_causal_keep(512, 512, bottom_right=True), use_causal_mask_bottom_right=True).check()


@requires_pre_rubin_blackwell
def test_causal_bottom_right_rectangular():
    _run(sq=512, skv=1024, keep=_causal_keep(512, 1024, bottom_right=True), use_causal_mask_bottom_right=True).check()


@requires_pre_rubin_blackwell
def test_causal_bottom_right_ragged_s_q():
    """The diagonal is S_kv - S_q in REAL rows (1024 - 500); the body takes sq_real / SQ_REAL like the 4x1 one."""
    _run(b=1, hq=2, sq=500, skv=1024, keep=_causal_keep(500, 1024, bottom_right=True), use_causal_mask_bottom_right=True).check()


@requires_pre_rubin_blackwell
def test_sliding_window():
    _run(keep=_causal_keep(512, 512, left=256), use_causal_mask=True, diagonal_band_left_bound=256).check()


@requires_pre_rubin_blackwell
def test_sliding_window_640_plus_bottom_right():
    _run(sq=512, skv=1024, keep=_causal_keep(512, 1024, bottom_right=True, left=640), use_causal_mask_bottom_right=True, diagonal_band_left_bound=640).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("hq,hkv", [(4, 2), (8, 1), (16, 1), (8, 4), (32, 2)], ids=["r2", "r8-mqa", "r16-mqa", "r2-h8", "r16-h32"])
def test_gqa(hq, hkv):
    _run(hq=hq, hkv=hkv, sq=256, skv=256).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("sq,skv", [(768, 1280), (500, 500), (300, 200), (257, 129), (384, 640)])
def test_non_tile_multiple_seqlens(sq, skv):
    _run(sq=sq, skv=skv).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("sq,skv", [(500, 500), (1000, 1000)])
def test_non_tile_multiple_causal(sq, skv):
    _run(sq=sq, skv=skv, keep=_causal_keep(sq, skv), use_causal_mask=True).check()


@requires_pre_rubin_blackwell
def test_default_attn_scale():
    _run(omit_scale=True).check()


@requires_pre_rubin_blackwell
def test_explicit_zero_attn_scale_is_preserved():
    run = _run(b=1, hq=1, sq=128, skv=256, attn_scale=0.0, poison=float("nan")).check()
    for name, got in zip(("dQ", "dK"), run.outs[0][:2]):
        assert (got.float() == 0).all(), f"{name}: attn_scale = 0.0 must give an EXACT zero, got max |{name}| = {got.float().abs().max().item():.4f}"


@requires_pre_rubin_blackwell
def test_workspace_is_build_time_honest():
    g, _t, _outs = _build_graph(b=2, hq=2, sq=256, skv=256)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == g.get_workspace_size() > 0


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("n_q_tiles", [1, 2, 3, 4, 5])
def test_q_tiles_per_kv_block(n_q_tiles):
    """1..5 q tiles per kv block: the 2-deep S / dP / P parity rings wrap at 2, the 1-deep Q / dO / dO_dv rings at every tile."""
    _run(b=1, hq=1, sq=128 * n_q_tiles, skv=256).check()


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("n_kv_pairs", [1, 3])
def test_kv_blocks_with_several_tiles_in_flight(n_kv_pairs):
    """kv write pairs = 1 and 3 (2 / 6 kv blocks) with B*H > 1: several persistent tiles per CTA (P14)."""
    _run(b=2, hq=2, sq=256, skv=256 * n_kv_pairs).check()


@requires_pre_rubin_blackwell
def test_causal_tail_kv_block_writes_zeros_not_residue():
    """Top-left causal with S_kv > S_q: the kv blocks past S_q run the forced fully-masked tile and must STORE zeros."""
    sq, skv = 256, 768
    run = _run(b=2, hq=2, sq=sq, skv=skv, keep=_causal_keep(sq, skv), use_causal_mask=True, poison=float("nan")).check()
    _dq, dk, dv = run.outs[0]
    for name, got in (("dK", dk), ("dV", dv)):
        tail = got[:, :, sq:, :].float()
        assert torch.isfinite(tail).all() and (tail == 0).all(), f"{name}: unattended kv rows must be EXACTLY zero"


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("case", list(_MASK_POISON_CASES), ids=list(_MASK_POISON_CASES))
def test_masked_stage3_reads_only_what_stage2_wrote(case):
    """The 2x2-specific detector of the kv WRITE-PAIR invariant: a 128-row kv block derives its q range from the 256-row pair
    it belongs to, so the stage-3 GEMMs' two-sided K-trim reads only tiles stage 2 wrote under every mask arm the row serves
    (whole workspace poisoned 0xFF = NaN before the execute; ``_check`` asserts finite first).  ``bottom-right-ragged-sq``
    (shift 524, not a multiple of 256) and the windows are the cases where a block deriving its own 128-row bounds would read
    an unwritten tile."""
    shape, kw, keep = _mask_case_kw(_MASK_POISON_CASES[case])
    _run(keep=keep, poison=float("nan"), ws_poison=0xFF, **shape, **kw).check()


@requires_pre_rubin_blackwell
def test_poisoned_asymmetric_columns_detect_a_lane_half_mix_up():
    """Asymmetric data: Q and dO scaled per q column, K / V per kv row, so a q-column-half (lane half) or kv-row (lane quadrant)
    mix-up in the 2x2 image changes the gradients (symmetric unit-normal data hides a swapped half)."""
    b, hq, sq, skv, dt = 1, 1, 256, 256, torch.bfloat16
    gen = torch.Generator(device="cpu").manual_seed(7)
    qcol = (1.0 + 0.5 * torch.arange(sq) / sq).view(1, sq, 1, 1)
    kvrow = (1.0 + torch.arange(skv) / skv).view(1, skv, 1, 1)
    q = (torch.randn(b, sq, hq, _D, generator=gen) * qcol).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)
    do = (torch.randn(b, sq, hq, _D, generator=gen) * qcol.flip(1)).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)
    k = (torch.randn(b, skv, hq, _D, generator=gen) * kvrow).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)
    v = (torch.randn(b, skv, hq, _D, generator=gen) * kvrow.flip(1)).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)
    o64, lse64, _m, dq_r, dk_r, dv_r = _reference64(q, k, v, do, None, 1)
    o = _bshd_empty(b, sq, hq, _D, dt)
    o.copy_(o64.to(dt))
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b=b, hq=hq, hkv=hq, sq=sq, skv=skv, dt=dt, scale="default")
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8).fill_(0xFF)
    dq, dk, dv = (
        _bshd_empty(b, sq, hq, _D, dt, fill=float("nan")),
        _bshd_empty(b, skv, hq, _D, dt, fill=float("nan")),
        _bshd_empty(b, skv, hq, _D, dt, fill=float("nan")),
    )
    g.execute({t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: lse64.float().unsqueeze(-1).contiguous(), dq_t: dq, dk_t: dk, dv_t: dv}, ws)
    torch.cuda.synchronize()
    for name, got, ref in zip(("dQ", "dK", "dV"), (dq, dk, dv), (dq_r, dk_r, dv_r)):
        _check(name, got, ref, dt)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_two_launches_are_bitwise_and_race_free(dt):
    """Launch 1 vs 2 is the two-launch race trick (a first-launch race on the lane-written sP slab published by the
    .release.cta leader arrive -- the first Producer.LEADER_RELEASE consumer in the tree -- would show here), launch 2 vs 3
    the determinism the row must show before claiming use_deterministic_algorithm.  Raw bits (int16 views)."""
    run = _run(b=2, hq=4, hkv=2, sq=512, skv=768, dt=dt, keep=_causal_keep(512, 768), use_causal_mask=True, runs=3, poison=float("nan")).check()
    for which, (a, b_) in (("launch 2 vs 1 (race)", (run.outs[1], run.outs[0])), ("launch 3 vs 2 (determinism)", (run.outs[2], run.outs[1]))):
        for name, x, y in zip(("dQ", "dK", "dV"), a, b_):
            n_diff = (x.view(torch.int16) != y.view(torch.int16)).sum().item()
            assert n_diff == 0, f"{name} {which}: {n_diff} elements differ"


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("b", [257])
def test_ragged_kv_batches_past_the_fill_block_read_their_own_kv_length(b):
    _run(b=b, hq=1, hkv=1, sq=128, skv=129, ws_poison=0, poison=float("nan")).check()


# =========================================================================== prepared launch: rebind / stream / replay / fresh process


def _prepared_case(dt=torch.bfloat16, causal=True, b=2, hq=4, hkv=2, sq=512, skv=512, seed=0):
    from types import SimpleNamespace

    group = hq // hkv
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(bb, s_, h):
        return torch.randn(bb, s_, h, _D, generator=gen).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, do, k, v = draw(b, sq, hq), draw(b, sq, hq), draw(b, skv, hkv), draw(b, skv, hkv)
    keep = _causal_keep(sq, skv) if causal else None
    o64, lse64, all_masked, dq_r, dk_r, dv_r = _reference64(q, k, v, do, keep, group)
    o = _bshd_empty(b, sq, hq, _D, dt)
    o.copy_(o64.to(dt))
    lse = lse64.float()
    if all_masked is not None:
        lse = lse.masked_fill(all_masked, 0.0)
    kw = dict(use_causal_mask=True) if causal else {}
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, dt=dt, scale="default", **kw)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8).fill_(0xBD)
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=lse.unsqueeze(-1).contiguous())
    tensors.update(
        dq=_bshd_empty(b, sq, hq, _D, dt, fill=float("nan")),
        dk=_bshd_empty(b, skv, hkv, _D, dt, fill=float("nan")),
        dv=_bshd_empty(b, skv, hkv, _D, dt, fill=float("nan")),
    )
    refs = dict(q=t["q"], k=t["k"], v=t["v"], o=t["o"], do=t["do"], stats=t["stats"], dq=dq_t, dk=dk_t, dv=dv_t)
    pack = {refs[name]: value for name, value in tensors.items()}
    g.execute(pack, ws)
    torch.cuda.synchronize()
    case = SimpleNamespace(
        graph=g,
        refs=refs,
        tensors=tensors,
        pack=pack,
        workspace=ws,
        keep=keep,
        group=group,
        dt=dt,
        expected=(dq_r, dk_r, dv_r),
        b=b,
        hq=hq,
        hkv=hkv,
        sq=sq,
        skv=skv,
        causal=causal,
    )
    assert g._compiled_plans[g._plan_index]._prepared is not None, "the row must lower through the prepared-launch contract"
    _check_prepared(case)
    return case


def _check_prepared(case, tensors=None, expected=None):
    tensors = case.tensors if tensors is None else tensors
    expected = case.expected if expected is None else expected
    for name, key, want in zip(("dQ", "dK", "dV"), ("dq", "dk", "dv"), expected):
        _check(name, tensors[key], want, case.dt)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("causal", [False, True])
def test_prepared_rebind_stream_and_replay(dt, causal):
    """The plan's prepared launch rebinds fresh buffers, follows the HANDLE's stream (not the ambient one) and captures
    into a CUDA graph whose replay recomputes new inputs -- the contract every prepared backward shares (the sm107 pin,
    verbatim: the capture runs on the handle's stream, the launch is issued from ANOTHER ambient stream)."""
    case = _prepared_case(dt=dt, causal=causal)
    tensors = {name: value.clone() for name, value in case.tensors.items()}
    pack = {case.refs[name]: value for name, value in tensors.items()}
    workspace = torch.empty_like(case.workspace).fill_(0xBD)
    stream, other = torch.cuda.Stream(), torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    capture = torch.cuda.CUDAGraph()

    def refresh():
        tensors["q"].mul_(0.75)
        tensors["do"].mul_(1.25)
        o64, lse64, all_masked, dq, dk, dv = _reference64(tensors["q"], tensors["k"], tensors["v"], tensors["do"], case.keep, case.group)
        tensors["o"].copy_(o64.to(dt))
        lse = lse64.float()
        if all_masked is not None:
            lse = lse.masked_fill(all_masked, 0.0)
        tensors["stats"].copy_(lse.unsqueeze(-1))
        for name in ("dq", "dk", "dv"):
            tensors[name].fill_(float("nan"))
        workspace.fill_(0xBD)
        return dq, dk, dv

    try:
        expected = refresh()
        stream.wait_stream(torch.cuda.current_stream())
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            case.graph.execute(pack, workspace, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        _check_prepared(case, tensors, expected)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            with torch.cuda.stream(other):
                case.graph.execute(pack, workspace, handle=handle)
        expected = refresh()
        capture.replay()
        torch.cuda.synchronize()
        _check_prepared(case, tensors, expected)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_prepared_backward_artifact_reloads_in_fresh_process(dtype, tmp_path):
    from prepared_bwd_cache_utils import check_backward_artifact_reload

    check_backward_artifact_reload("sm100_d256", "dense", dtype, tmp_path)


# =========================================================================== adapter host pins


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(dt=torch.float16),
        dict(hq=8, hkv=2, sq=256, skv=256),
        dict(sq=500, skv=500, is_causal=True, window_size_left=256),
        dict(b=1, hq=2, sq=500, skv=1024, is_causal=True, causal_bottom_right=True),
        dict(sq=257, skv=129),
    ],
    ids=["dense", "fp16", "gqa", "swa-padded", "br-ragged-sq", "257x129"],
)
def test_adapter_backstop_admits_the_served_matrix_and_sizes_its_workspace_at_build(kw):
    from cudnn.sdpa.bwd.api_dsl_sm100_d256 import SdpaBwdDslSm100D256
    from cudnn.sdpa.fwd.api_dsl import ws_align

    api = _adapter(SdpaBwdDslSm100D256, **kw)
    assert api.check_support()
    plan = api._scratch_plan()
    names = [n for n, _n, _d in plan]
    assert names[:4] == ["delta", "ds_ws", "seq_kv", "desc_words"]
    assert api.scratch_workspace_bytes() == sum(ws_align(n * d.itemsize) for _, n, d in plan) > 0
    assert ("q_pad" in names) == (api.s_q_max % 128 != 0) and ("k_pad" in names) == (
        api.s_k_max % 256 != 0
    ), "S_q padded to the q tile, S_kv to the kv WRITE PAIR"


@pytest.mark.parametrize(
    "kw, needle",
    [
        (dict(dt=torch.float32), "dtype"),
        (dict(hq=6, hkv=4), "multiple of h_kv"),
        (dict(sq=1), "decode"),
        (dict(is_causal=True, window_size_right=16), "right-band"),
        (dict(window_size_left=0), "window_left > 0"),
        (dict(deterministic=True), "deterministic"),
        (dict(seq_kv_lens_present=True), "padding masks"),
        (dict(thd=True), "THD"),
        (dict(d=512), "d_qk = d_v = 256"),
    ],
    ids=["fp32", "gqa-ratio", "decode", "right-band", "swa-zero", "deterministic", "padded", "thd", "d512"],
)
def test_adapter_backstop_refuses_what_the_row_declines(kw, needle):
    from cudnn.sdpa.bwd.api_dsl_sm100_d256 import SdpaBwdDslSm100D256

    with pytest.raises(ValueError, match=needle):
        _adapter(SdpaBwdDslSm100D256, **kw).check_support()


def test_prepared_host_admits_the_sm100_line_for_the_half_chain(monkeypatch):
    """``compile_host_f16`` compiles for sm_100a / sm_103a (the 2x2 body) and still for the Rubin line; the fp8 host keeps
    the Rubin-only range; the artifact symbol is the row's."""
    import cutlass
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host

    calls = []
    monkeypatch.setattr(prepared_host, "compile_cached", lambda *a, **k: calls.append((k["options"], k["symbol"])) or object())
    cfg = (1, 2, 2, 256, 256, 256, 256, 256, 1, 2, False, 2, 2, 1)
    for sm in (100, 103, 107, 110):
        prepared_host.compile_host_f16(None, None, None, cfg, (), (), cutlass.BFloat16, sm, "probe", symbol=f"frost_{_ENGINE}_prepared")
    assert calls == [(f"--enable-tvm-ffi --gpu-arch sm_{sm}a", f"frost_{_ENGINE}_prepared") for sm in (100, 103, 107, 110)]
    with pytest.raises(ValueError, match="got SM90"):
        prepared_host.compile_host_f16(None, None, None, cfg, (), (), cutlass.BFloat16, 90, "probe")
    with pytest.raises(ValueError, match="SM107-SM119"):
        prepared_host.compile_host_fp8(None, None, None, cfg, (), (), cutlass.Float8E4M3FN, (True, True, True, True), 100, "probe", sm_count=148)
    assert prepared_host.compile_host_f16.__defaults__[-1] == "frost_sdpa_bwd_sm107_prepared", "the Rubin row's symbol is the default"


# =========================================================================== static pins on the 2x2 body


@pytest.mark.parametrize("profile", [1, 2])
def test_descriptor_version_is_derived_per_profile(profile):
    from cudnn.sdpa.bwd import config_d256_2x2 as c2

    mod = _load_kernel(profile)
    assert mod.DESC_VERSION == c2.desc_version_2x2(mod.CFG) == (0 if profile == 1 else 1)
    assert mod._KV_BLOCK_ROWS == 128 * profile and mod._KV_WRITE_ROWS == 256 and mod._Q_WRITE_TILES == 2
    assert mod.LAYOUT.TOTAL_COLS == 512 and mod.L_CNT == (512 if profile == 1 else 256)


def test_every_smem_tile_takes_the_module_desc_version():
    code = _code_lines(_kernel_source())
    n_tiles = len(re.findall(r"\bSmemTile\(", code))
    assert n_tiles == 8, "sQ, sdO, sdO_dv, sK, sV, sP, sdS, sdV"
    assert code.count("desc_version=DESC_VERSION") == n_tiles
    assert not re.search(r"desc_version=[01]\b", code)
    assert len(re.findall(r"^DESC_VERSION\b.*=", code, re.M)) == 1


def test_ring_waits_take_the_module_spin_constant():
    from test_sdpa_bwd_dsl_sm107 import _wait_sites

    mod = _load_kernel(1)
    assert isinstance(mod.SPIN_RING_WAITS, bool)
    code = _code_lines(_kernel_source())
    assert len(re.findall(r"^SPIN_RING_WAITS: bool = (?:True|False)$", code, re.M)) == 1
    assert not re.search(r"spin=(?:True|False)\b", code)
    sites = _wait_sites(code)
    spun = [t for t, args in sites if "spin=SPIN_RING_WAITS" in args]
    assert spun and code.count("spin=") == len(spun)
    assert not [t for t in spun if t.startswith("mb_tmem_dealloc") or t.startswith("sched.")], "whole-tile idle waits keep the sleeping form"


def test_no_cluster_scope_release_arrive_and_no_ptxas_knobs():
    code = _code_lines(_kernel_source())
    bad = [ln for ln in code.splitlines() if "MemScope.CLUSTER" in ln and "relaxed" not in ln and "cga_arrive" not in ln]
    assert not bad
    knobs = ["".join(p) for p in (("uu", "mn"), ("Fence", "Code"), ("cf", "ence"))]
    assert not re.findall("|".join(knobs), _kernel_source())


def test_idesc_k_dim_and_q_loop_bounds_come_from_the_shared_primitives():
    code = _code_only(_kernel_source())
    assert sorted(set(re.findall(r"\bk_dim\s*=\s*([^,)\s]+)", code))) == ["CFG.IDESC_K_DIM"]
    assert "compute_q_loop_bounds(" in code and "band_mask_words(" in code and "apply_mask_words(" in code
    assert (
        "mma_ts(" not in code and "tcgen05_cp" not in code and "tcgen05_st(" not in code
    ), "every operand is an SMEM SS operand: no TS MMA, no UTCCP, no TMEM P store"
    assert "m_dim=CFG.MMA_M" in code and "m_dim=CFG.TILE_M * CFG.CTA_MMA" not in code, "every idesc carries the 2x2 collective M = 128"
    assert code.count("elect_once=True") == 5, "the five mma_ss sites (prologue S, NATURAL S, dP, lookahead S, BMM2) elect once per call"
    assert "Producer.LEADER_RELEASE" in code, "mb_p_full publishes the lane-written sP slab with the .release.cta leader arrive"


def test_mbarrier_ledger_matches_the_config():
    """Host-only ledger pin (P3): every MBarrier(...) in _make_bars allocates, stages and initialises exactly what the config
    ledger derives, for both profiles; LEADER / LEADER_RELEASE producers carry the LEADER scope."""
    from cudnn.sdpa.bwd import config_d256_2x2 as c2

    src = _kernel_source()
    bars_src = src[src.index("def _make_bars") : src.index("# Tile decode")]
    pat = re.compile(r"(mb_\w+)=MBarrier\(_alloc\(([^)]+)\),\s*stages=([^,]+),\s*init_count=([^,]+),\s*producer=Producer\.(\w+)(?:,\s*scope=Scope\.(\w+))?\)")
    rows = pat.findall(bars_src)
    assert len(rows) == 25
    for profile in (1, 2):
        mod = _load_kernel(profile)
        want_stages, want_counts = c2.mbar_stage_counts_2x2(mod.CFG), c2.mbar_init_counts_2x2(mod.CFG)
        env = {
            k: getattr(mod, k)
            for k in ("ONE_LANE", "ONE_WARP", "SOFTMAX_LANES_ALL_WG", "SOFT_X_CTA_MMA", "L_CNT", "MMA_COMMIT_ARRIVES", "N_S_BARS", "N_P_BARS")
        }
        env["cfg"] = mod.CFG
        seen = set()
        for name, alloc, stages, count, prod, scope in rows:
            seen.add(name)
            assert eval(stages, {}, env) == eval(alloc, {}, env) == want_stages[name], (profile, name)
            assert eval(count, {}, env) == want_counts[name], (profile, name)
            assert (prod in ("LEADER", "LEADER_RELEASE")) == (scope == "LEADER"), (profile, name)
        assert seen == set(want_stages)
        assert (mod.qTmaTransactionBytes, mod.kTmaTransactionBytes) == (65536, 65536 * profile)
        assert mod.READ_TILE_ARRIVERS_TOT == 21


def test_kernel_is_named_by_geometry():
    src = _kernel_source()
    words = ["".join(p) for p in (("vibe", "tile"), ("qw", "en"), ("dk", "g"), ("tile_", "ct", "m"), ("ct", "m"))]
    assert not re.findall(r"(?i)\b(" + "|".join(words) + r")\b", src)


# =========================================================================== SASS pins: trace-compile per arch / profile on ANY Blackwell box

_SASS_PROBE = textwrap.dedent(r"""
    import glob, os, re, subprocess, sys
    dump, arch, profile, mask, cands = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "cubin"
    os.environ["CUTE_DSL_ARCH"] = arch          # unconditional: an inherited value would pin the wrong target's SASS
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams
    MASKS = {"dense": {}, "causal": dict(window_right=0), "causal_swa": dict(window_right=0, window_left=640)}
    params = TemplateParams(dtype_qkv=DTYPE_BF16, datapath_2x2_profile=profile, **MASKS[mask])
    mod = load_template(_sm100_kernel_path("bprop_d256_2x2_f16.py"), params, tag=f"sass_2x2_{arch}_p{profile}_" + mask)
    print("SASS SOFTMAX_REGS", mod.CFG.SOFTMAX_REGS)
    print("SASS OTHER_REGS", mod.CFG.OTHER_REGS)
    slot_cols = mod.LAYOUT.S_COLS  # one S / dP accumulator slot = 64 TMEM columns (the 2x2 D atom); dV is a separate base
    print("SASS SLOT_COLS", slot_cols)
    mod.compile(b=1, qh=8, kh=8, sq=1024, skv=1024)
    cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
    if not cubins:
        print("FAIL no cubin dumped into", dump, os.listdir(dump)); sys.exit(3)
    nvd = None
    for c in cands:
        try:
            proc = subprocess.run([c, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", c, "->", repr(exc)); continue
        if proc.returncode == 0 and proc.stdout.strip():
            nvd = c; print("NVDISASM", c); break
        print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if nvd is None:
        print("SKIP no nvdisasm candidate decodes the cubin"); sys.exit(0)
    sass = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, check=True).stdout.splitlines()
    def cnt(*subs):
        return sum(1 for ln in sass if all(sb in ln for sb in subs))
    for k, subs in (("USETMAXREG", ("USETMAXREG",)), ("STL", ("STL",)), ("LDL", ("LDL",)), ("MEMBAR_GPU", ("MEMBAR.ALL.GPU",)), ("CGAERRBAR", ("CGAERRBAR",)),
                    ("UTCHMMA", ("UTCHMMA",)), ("LDTM", ("LDTM",)), ("STTM", ("STTM",)), ("UTCCP", ("UTCCP",)), ("R2P", (" R2P",)), ("MUFU_EX2", ("MUFU.EX2",))):
        print("SASS", k, cnt(*subs))
    print("SASS LINES", len(sass))
    alloc = [int(ln.split(",")[-1].strip(" ;"), 16) for ln in sass if "USETMAXREG.TRY_ALLOC" in ln]
    dealloc = [int(ln.split()[-2], 16) for ln in sass if "USETMAXREG.DEALLOC" in ln]
    print("SASS USETMAXREG_ALLOC", alloc[0] if alloc else -1)
    print("SASS USETMAXREG_DEALLOC", dealloc[0] if dealloc else -1)
    INS = re.compile(r"^\s+/\*([0-9a-f]+)\*/\s+(?:(@!?U?P[0-9T]+)\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;")
    LABEL = re.compile(r"^(\.L_x_\d+):")
    ins, labels, pending = [], {}, []
    for ln in sass:
        m = LABEL.match(ln)
        if m:
            pending.append(m.group(1)); continue
        m = INS.match(ln)
        if not m:
            continue
        for lab in pending:
            labels[lab] = len(ins)
        pending = []
        ins.append((m.group(3), m.group(4)))
    bodies = []
    for i, (op, args) in enumerate(ins):
        if not op.startswith("BRA"):
            continue
        m = re.search(r"(\.L_x_\d+)", args)
        if m and m.group(1) in labels and labels[m.group(1)] <= i:
            t = labels[m.group(1)]
            body = ins[t : i + 1]
            if len(body) >= 40 and sum(1 for o, _ in body if o.startswith("MUFU.EX2")) >= 16:
                bodies.append((t, i))
    inner = [(t, e) for t, e in bodies if not any((t2, e2) != (t, e) and t <= e2 <= e for t2, e2 in bodies)]
    n_groups = n_bad = n_ldtm = 0
    for t, e in inner:
        slots = {}
        arrives = []
        for j in range(t, e + 1):
            op, args = ins[j]
            if op.startswith("LDTM"):
                n_ldtm += 1
                m = re.search(r"tmem\[(U?R\d+)(?:\+(0x[0-9a-fA-F]+))?\]", args)
                if m:
                    slots.setdefault((m.group(1), int(m.group(2) or "0", 16) // slot_cols), []).append(j)
            elif "ARRIVE" in op:
                arrives.append(j)
        for key, idx in slots.items():
            if len(idx) < 2:
                continue
            n_groups += 1
            lo, hi = min(idx), max(idx)
            if [a for a in arrives if lo < a < hi]:
                n_bad += 1
    print("LDTM_ORDER_BODIES", len(inner))
    print("LDTM_ORDER_LDTMS", n_ldtm)
    print("LDTM_ORDER_GROUPS", n_groups)
    print("LDTM_ORDER_VIOLATIONS", n_bad)
    """)
_NATURAL_PROBE = textwrap.dedent(r"""
    import sys
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams
    MASKS = {"dense": {}, "causal": dict(window_right=0)}
    for mask, kw in MASKS.items():
        mod = load_template(_sm100_kernel_path("bprop_d256_2x2_f16.py"), TemplateParams(dtype_qkv=DTYPE_BF16, datapath_2x2_profile=1, **kw), tag="natural_2x2_" + mask)
        assert mod.MMA_LOOKAHEAD is False, "profile 1 ships the NATURAL MMA order (config_d256_2x2.MMA_LOOKAHEAD = 0)"
        mod.compile(b=1, qh=2, kh=1, sq=256, skv=512)
        print("NATURAL_OK", mask)
        mod.MMA_LOOKAHEAD = True
        mod.compile(b=1, qh=2, kh=1, sq=256, skv=512)
        print("LOOKAHEAD_OK", mask)
    """)


@requires_pre_rubin_blackwell
def test_mma_order_arms_both_compile():
    """Both MMA-order arms trace-compile for dense and causal, and profile 1's default is the NATURAL order (S(i), dP(i),
    BMM2(i) per q tile; ``config_d256_2x2.MMA_LOOKAHEAD = 0``) -- the B200 A/B of 2026-10-01 put stage 2 at 3781 us
    NATURAL vs 4525 us lookahead dense (2020 vs 1974 us causal).  Regression pin for the DSL's branch-join typing: every
    name the NATURAL block assigns inside the q loop (``desc_Q``, ``s_bar``) must already exist with the same type on the
    other path, or the DSL raises TYPE_UNSTABLE_JOIN (hit 2026-10-01 on the first stage-2 arm A/B).  A fresh process,
    because the arm is a module constant flipped on the loaded template (the ``SPIN_RING_WAITS`` idiom) and must not
    leak into the other tests' cached module.  Correctness of both arms: the graph-API suite here (default) and the
    lane's direct-launch bring-up (both arms vs the fp64 reference)."""
    env = dict(os.environ, CUDNN_FRONTEND_DISABLE_COMPILED_CACHE="1")
    proc = subprocess.run([sys.executable, "-c", _NATURAL_PROBE], capture_output=True, text=True, timeout=1500, env=env)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert proc.stdout.count("NATURAL_OK") == 2 and proc.stdout.count("LOOKAHEAD_OK") == 2, proc.stdout


# (arch, profile, mask) rows.  sm_100a / sm_103a at profile 1 = the ``sdpa_bwd_sm100_d256`` row (the heuristics list it on cc
# 10.0 and 10.3, so both codegen targets are pinned); sm_107a at profile 1 = the Rubin twin's A/B arm (the SM100 body as-is);
# sm_107a at profile 2 = the Rubin interleaved twin.  A row SKIPS where the DSL does not know its arch (4.7.0: no sm_107a).
_ARM_NUMERICS_PROBE = textwrap.dedent(r"""
    import math, sys
    import torch
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams
    lookahead = bool(int(sys.argv[1]))
    b, qh, kh, sq, skv, d = 2, 4, 2, 512, 768, 256
    dt = torch.bfloat16
    mod = load_template(_sm100_kernel_path("bprop_d256_2x2_f16.py"), TemplateParams(dtype_qkv=DTYPE_BF16, datapath_2x2_profile=1, window_right=0), tag="arm_numerics")
    mod.MMA_LOOKAHEAD = lookahead
    fn = mod.compile(b=b, qh=qh, kh=kh, sq=sq, skv=skv)
    gen = torch.Generator(device="cpu").manual_seed(3)
    draw = lambda *shape: torch.randn(*shape, generator=gen).to(device="cuda", dtype=dt)
    q, do, k, v = draw(b, sq, qh, d), draw(b, sq, qh, d), draw(b, skv, kh, d), draw(b, skv, kh, d)
    group, scale = qh // kh, 1.0 / math.sqrt(d)
    q64, do64 = q.double().permute(0, 2, 1, 3), do.double().permute(0, 2, 1, 3)
    k64 = k.double().permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    v64 = v.double().permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    keep = torch.arange(skv, device="cuda").view(1, -1) <= torch.arange(sq, device="cuda").view(-1, 1)
    s_ = ((q64 @ k64.transpose(-1, -2)) * scale).masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(s_, dim=-1)
    p = torch.exp(s_ - lse.unsqueeze(-1)).nan_to_num_(0.0)
    delta = ((p @ v64) * do64).sum(-1)
    ds_ref = scale * (do64 @ v64.transpose(-1, -2) - delta.unsqueeze(-1)) * p
    dv_ref = p.transpose(-1, -2) @ do64
    dv = torch.full((b, skv, qh, d), float("nan"), device="cuda", dtype=dt)
    ds = torch.full((b, qh, skv, sq), float("nan"), device="cuda", dtype=dt)
    seq_kv = torch.full((b,), skv, dtype=torch.int32, device="cuda")
    fn(q, do, k, v, dv, ds, lse.float().contiguous(), delta.float().contiguous(), seq_kv, (b, qh, kh, sq, skv, qh, b, sq, skv), scale, 0, 0, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    # written dS tiles: causal top-left, per 256-row kv pair the q tiles [pair-rounded floor(kb / 128), n_q)
    n_q = sq // 128
    written = torch.zeros(skv, sq, dtype=torch.bool, device="cuda")
    for kb in range(0, skv, 256):
        lo = min(((kb // 128) // 2) * 2, n_q - 1)
        written[kb : kb + 256, lo * 128 :] = True
    ds_bhqk = ds.permute(0, 1, 3, 2)
    w = written.t().unsqueeze(0).unsqueeze(0).expand_as(ds_bhqk)
    assert torch.isnan(ds_bhqk.float()[~w]).all(), "stage 2 wrote outside its pair-rounded q range"
    for name, got, ref in (("dV", dv.permute(0, 2, 1, 3).float(), dv_ref.float()), ("dS", ds_bhqk.masked_fill(~w, 0.0).float(), ds_ref.masked_fill(~w, 0.0).float())):
        assert torch.isfinite(got).all(), name + ": non-finite"
        cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
        assert cos > 0.9999, f"{name}: cos {cos}"
        torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2, msg=lambda m: f"{name} vs fp64 (lookahead={lookahead}): {m}")
        print(f"ARM_OK {name} lookahead={lookahead} cos={cos:.6f} max|diff|={(got - ref).abs().max().item():.3e}")
    """)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("lookahead", [0, 1], ids=["natural", "lookahead"])
def test_mma_order_arms_match_the_fp64_reference(lookahead):
    """The NUMERICS of both MMA-order arms, live: a fresh-process direct launch of profile 1 with ``MMA_LOOKAHEAD`` set on the
    loaded module (the SPIN_RING_WAITS idiom), causal GQA 4/2 at 512 x 768 (3 kv pairs x 4 q tiles, a tail pair), dV and the
    written dS tiles against the fp64 reference at the sm107 suite's bf16 tolerance (atol = rtol = 5e-2, cos > 0.9999), the
    unwritten dS tiles still NaN-poisoned.  The lookahead arm does not ship on profile 1 (the A/B arm; profile 2's default),
    so without this pin an edit to its branches could break it unnoticed (review 2026-10-01)."""
    env = dict(os.environ, CUDNN_FRONTEND_DISABLE_COMPILED_CACHE="1")
    proc = subprocess.run([sys.executable, "-c", _ARM_NUMERICS_PROBE, str(lookahead)], capture_output=True, text=True, timeout=1500, env=env)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert proc.stdout.count("ARM_OK") == 2, proc.stdout


_SASS_ROWS = [
    pytest.param("sm_100a", 1, "dense", id="sm100a-p1-dense"),
    pytest.param("sm_100a", 1, "causal", id="sm100a-p1-causal"),
    pytest.param("sm_100a", 1, "causal_swa", id="sm100a-p1-causal-swa"),
    pytest.param("sm_103a", 1, "dense", id="sm103a-p1-dense"),
    pytest.param("sm_103a", 1, "causal", id="sm103a-p1-causal"),
    pytest.param("sm_107a", 1, "dense", id="sm107a-p1-dense"),
    pytest.param("sm_107a", 1, "causal", id="sm107a-p1-causal"),
    pytest.param("sm_107a", 2, "dense", id="sm107a-p2-dense"),
    pytest.param("sm_107a", 2, "causal", id="sm107a-p2-causal"),
    pytest.param("sm_107a", 2, "causal_swa", id="sm107a-p2-causal-swa"),
]
# Spill BOUNDS (frost_test_utils.assert_no_new_spills: measured + SPILL_TOLERANCE), MEASURED 2026-10-01 at B=1 H=8 S=1024:
#   sm_100a / sm_103a profile 1 (176 / 152): 0 / 0 on nvidia-cutlass-dsl 4.7.0 (CUDA 13.3 ptxas) AND on the Rubin board's
#     internal DSL 0.3.0 + internal toolkit ptxas (sm_100a re-measured there: 0 / 0).
#   sm_107a profile 1 (176 / 152): 18-19 STL / 43-44 LDL on the board's toolchain -- the sm_107a codegen of the 32-column
#     lanes; every other split tried is worse (224 / 56 -> 74 / 81, 208 / 88 -> 57 / 64).  The A/B arm, not a row.
#   sm_107a profile 2 (224 / 56): 0 / 0 (91 / 129 at 176 / 152, 113 / 135 at 208 / 88 -- the board register-split sweep).
_SPILL_PINS = {
    ("sm_100a", 1): {"STL": 0, "LDL": 0},
    ("sm_103a", 1): {"STL": 0, "LDL": 0},
    ("sm_107a", 1): {"STL": 19, "LDL": 44},
    ("sm_107a", 2): {"STL": 0, "LDL": 0},
}
# Instruction census per profile: UTCHMMA per q tile and the TMEM loads.  Profile 1 (NATURAL order): 40 = S 16 + dP 16 + BMM2 8,
# 3 LDTM (S x32, dP x32, dV x64).  Profile 2 (lookahead, two sub-blocks): 112 = 2 x (16 S prologue + 40), 4 LDTM (measured
# on the board 2026-10-01: the two sub-blocks' S / dP loads; the dV drain is shared).
_CENSUS_PINS = {1: dict(UTCHMMA=40, LDTM=3), 2: dict(UTCHMMA=112, LDTM=4)}
_SASS_CACHE = {}


def _sass_probe(tmp_path, arch, profile, mask):
    key = (arch, profile, mask)
    if key in _SASS_CACHE:
        return _SASS_CACHE[key]
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch}")
    cands = nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"{arch}_bwd_2x2_p{profile}_{mask}"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _SASS_PROBE, str(dump), arch, str(profile), mask, *cands], capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"{arch} trace-compile of the 2x2 profile-{profile} {mask} backward failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].lstrip("-").isdigit()}
    order = {ln.split()[0]: int(ln.split()[1]) for ln in out if ln.startswith("LDTM_ORDER_") and len(ln.split()) == 2}
    print(f"\n2x2 bwd p{profile} {mask} {arch} SASS: {stats}; {order}")
    _SASS_CACHE[key] = (stats, order)
    return stats, order


@pytest.mark.parametrize("arch, profile, mask", _SASS_ROWS)
def test_register_split_spills_and_drains_sass_pins(tmp_path, arch, profile, mask):
    """USETMAXREG > 0 and its operands ARE the config's split (176 / 152 on profile 1, 224 / 56 on profile 2 -- ptxas C7508
    would drop it silently), no new stack spills against the per-(arch, profile) bound, no GPU-scope drain on a per-tile path,
    and the per-profile instruction census (UTCHMMA per q tile, LDTM; no UTCCP / STTM)."""
    stats, _order = _sass_probe(tmp_path, arch, profile, mask)
    assert stats["USETMAXREG"] > 0, "no USETMAXREG: ptxas dropped the register split (C7508)"
    assert (stats["USETMAXREG_ALLOC"], stats["USETMAXREG_DEALLOC"]) == (stats["SOFTMAX_REGS"], stats["OTHER_REGS"]), stats
    assert_no_new_spills(stats, _SPILL_PINS[(arch, profile)], tag=f"2x2 {arch} p{profile} {mask}: ")
    assert stats["MEMBAR_GPU"] == 0 and stats["CGAERRBAR"] == 0, "a cluster-scope RELEASE arrive is on a per-tile path (GPU-scope drain)"
    census = _CENSUS_PINS[profile]
    assert stats["UTCHMMA"] == census["UTCHMMA"] and stats["LDTM"] == census["LDTM"] and stats["UTCCP"] == 0 and stats["STTM"] == 0, stats


@pytest.mark.parametrize("arch, profile, mask", _SASS_ROWS)
def test_every_tmem_load_precedes_the_arrive_that_frees_its_slot(tmp_path, arch, profile, mask):
    """No ARRIVE is scheduled between two LDTMs of ONE accumulator slot inside the softmax body (frost-kernels.md s3).  A
    slot here is 64 TMEM columns (``LAYOUT.S_COLS``: the 2x2 D atom puts a 64 x 128 fp32 tile in 64 columns), and the
    detector keys by (base register, offset // 64): the sm107 suite's 128-column window is the 4x1 body's slot width and,
    on profile 2, merged a sub-block's adjacent S and dP slots (``tmem[UR+0]`` then ``tmem[UR+0x40]``, with the
    ``s_acc_empty`` and ``p_full`` arrives correctly between them) into a false violation (board, 2026-10-01)."""
    _stats, order = _sass_probe(tmp_path, arch, profile, mask)
    assert order.get("LDTM_ORDER_BODIES", 0) > 0 and order.get("LDTM_ORDER_LDTMS", 0) > 0, "the detector found no softmax body / LDTM -- it pins nothing"
    assert order["LDTM_ORDER_VIOLATIONS"] == 0


@pytest.mark.parametrize("arch, profile, mask", [r for r in _SASS_ROWS if r.values[2] != "dense"])
def test_masked_arm_lowers_to_the_bit_word_form(tmp_path, arch, profile, mask):
    stats, _order = _sass_probe(tmp_path, arch, profile, mask)
    assert stats["R2P"] > 0, f"{arch} p{profile} {mask}: no R2P in the masked build -- the mask arm is the per-cell compare + select form"


_RUBIN_PROFILE_PROBE = textwrap.dedent(r"""
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams

    mod = load_template(
        _sm100_kernel_path("bprop_d256_2x2_f16.py"),
        TemplateParams(dtype_qkv=DTYPE_BF16, datapath_2x2_profile=2),
        tag="rubin_profile_trace",
    )
    assert mod.DESC_VERSION == 1 and mod.CFG.KV_SUBBLOCKS == 2 and mod.COLS_PER_LANE == 64 and mod.L_CNT == 256
    try:
        mod.compile(b=1, qh=2, kh=2, sq=256, skv=256)
    except AttributeError as e:
        if "tcgen05_mma_smem_desc_v2" not in str(e):
            raise
        print(f"SKIP profile 2 needs the version-1 tcgen05 SMEM descriptor intrinsic (nvidia-cutlass-dsl >= 4.8.0): {e}")
""")


def test_rubin_profile_traces_or_names_the_missing_dsl_intrinsic():
    """Rule S6: profile 2 (the Rubin interleaved twin) traces from this box.  Its descriptor version 1 lowers through the
    DSL's ``_tcgen05_mma_smem_desc_v2`` intrinsic (>= 4.8.0); a DSL without it fails the trace with that name, which this
    test turns into a SKIP naming the floor rather than a false red (a compile failure for any OTHER reason is a FAIL).
    PASSED on a cc 10.7 board 2026-10-01 (internal DSL 0.3.0: the trace at the device's sm_107a),
    where the twin's GPU matrix (``test_sdpa_bwd_dsl_sm107.py -k twox2``) and the sm_107a SASS rows above also ran -- the
    SKIP here is a 4.7.0-CI accommodation, not an untested path."""
    # Select before DSL import, independently of the runner GPU or an inherited target.
    arch = "sm_107a" if arch_known_to_the_dsl("sm_107a") else "sm_100a"
    env = dict(os.environ, CUTE_DSL_ARCH=arch, CUDNN_FRONTEND_DISABLE_COMPILED_CACHE="1")
    proc = subprocess.run([sys.executable, "-c", _RUBIN_PROFILE_PROBE], capture_output=True, text=True, timeout=1500, env=env)
    assert proc.returncode == 0, f"{arch} profile-2 trace failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    for line in proc.stdout.splitlines():
        if line.startswith("SKIP "):
            pytest.skip(line.removeprefix("SKIP "))
