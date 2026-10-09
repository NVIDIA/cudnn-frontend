# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm107_d512``: the cc 10.7 (the Rubin-line, 107 <= SM <= 119) d in (256, 512] bf16 / fp16 SDPA backward on the
2x2 tcgen05 datapath.

The SM100 large-head-dim chain (``api_dsl.SdpaBwdDslSm100``) with stage 2 ALWAYS the cc 10.7 sibling of the 2x2 twin,
``kernels/sm107/bprop_d512_f16_2x2.py`` at the 8-stage ring arm (``api_dsl_sm107_d512``).  Every capability the row claims
gets an ACCEPT test through the graph API with the engine PINNED against the sm107 suite's fp64 oracle; every decline a
REJECT probe on a REAL graph with a faked cc 10.7 device (host, runs anywhere).  Row-specific detectors: the fork is PINNED
to the SM100 body -- its code diff is an allowlist of the deltas its docstring names, its module refuses the SM100 ring arm
and a 9th stage at import (RED-first: both raise), its rendering is the committed PTX md5 record
(``renderings/md5_stage2_2x2_sm107a.txt``) and BITWISE the SM100 body at Rubin params on the board (``torch.equal`` on int16
views of dQ / dK / dV and of the S / dS workspace -- the deltas are wait-form and constants, so a mismatch is a real body
divergence); the sm_107a SASS pins (0 / 0 spills, the 64 UTCHMMA / 32 UTMALDG census, no GPU-scope drain, no DSMEM copy);
forced head chunking; two-launch bitwise; the Rule S6 sm_100a lowering of the fork from any box.

Host pins run everywhere; GPU cases carry ``requires_rubin`` (the row's own range) and the wedge-prone arms run under
``process_watchdog``.  The contention case (a second context of the same chain time-slicing the GPU; the stage-3 GEMMs' hinted
``try_wait`` cross-pair waits are inside it) is L1: it takes minutes and wedges a context on failure.
"""

from __future__ import annotations

import difflib
import math
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import cudnn
from frost_test_utils import _SM, arch_known_to_the_dsl, assert_no_new_spills, process_watchdog, requires_dsl, requires_rubin, select_engine
from test_sdpa_bwd_dsl_sm100 import _STAGE2_2X2_SASS_COUNTS, _STAGE2_PTX_PROBE, _parse_md5_record, _stage2_sass_probe
from test_sdpa_bwd_dsl_sm107 import (
    _TOL,
    _Run,
    _bhsd_stride,
    _bshd_empty,
    _bshd_stride,
    _build_graph,
    _causal_keep,
    _check,
    _code_lines,
    _code_only,
    _half_bwd_graph,
    _reference64,
)

pytestmark = [pytest.mark.L0, requires_dsl]

_ENGINE = "sdpa_bwd_sm107_d512"
_FAMILY_NAME = "frost_sdpa_bwd"
_SLOT = 8  # engines/manifest.py: EngineSlot(8, opt_in=True) -> FROST_SDPA_BWD_ID_BASE + 8
_ENGINE_ID = 20_608
_D = 512
_RUBIN_CC = (10, 7)
_SM_RANGE = (107, 119)  # the row mirrors the other Rubin rows (engines._RUBIN)
_DTYPES = (torch.bfloat16, torch.float16)
_DTYPE_IDS = ("bf16", "fp16")
_FORK_FILE = "sm107/bprop_d512_f16_2x2.py"
_SM100_FILE = "sm100/bprop_d512_f16_2x2.py"
_TAG = "sdpa_bwd_sm107_stage2_2x2"
_RUBIN_ARM = dict(stages_kv=8, cast_stages=2)  # + smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2, the kernel's RUBIN_ARM
_MD5_RECORD = Path(__file__).resolve().parent / "renderings" / "md5_stage2_2x2_sm107a.txt"
_WATCHDOG_S = 900.0


def _kernel_path(fname=_FORK_FILE):
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path

    return _sm100_kernel_path(fname)


def _source(fname=_FORK_FILE):
    with open(_kernel_path(fname), encoding="utf-8") as fh:
        return fh.read()


def _rubin_arm():
    from cudnn.sdpa.bwd.config_sm100 import SM107_USABLE_DYN_SMEM_2X2

    return dict(_RUBIN_ARM, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2)


def _load_fork(**params):
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.config_sm100 import TemplateParams2x2

    params.setdefault("dtype_qkv", DTYPE_BF16)
    return load_template(_kernel_path(), TemplateParams2x2(**params), tag=f"{_TAG}_test")


def _spec(name=_ENGINE):
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == name), None)
    assert spec is not None, f"{name} is not in cudnn.sdpa.bwd.engines.ENGINE_SPECS"
    return spec


# =========================================================================== registration / capabilities (host)


def test_engine_is_registered_and_opt_in():
    from cudnn.engines.engine_ids import FROST_SDPA_BWD_ID_BASE
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    _spec()
    fam = next(f for f in MANIFEST if f.name == _FAMILY_NAME)
    assert _ENGINE in fam.slots, f"{_ENGINE} has no manifest slot (append slot {_SLOT}; never reuse a retired slot)"
    slot = fam.slots[_ENGINE]
    assert slot.slot == _SLOT and slot.opt_in, "new engines stay opt-in until they earn arch coverage + benchmarks"
    assert FROST_SDPA_BWD_ID_BASE + slot.slot == _ENGINE_ID
    names = [s.name for s in ENGINE_SPECS]
    # Append-only: this row ranks after every row that existed when it landed (a membership + order pin; the position is the
    # rank in the ranked plan list, never the id, and the NEXT appended row must not break this test).
    assert names.index(_ENGINE) > names.index("sdpa_bwd_sm100_d256"), names


def test_capabilities_match_what_is_implemented():
    """The row claims exactly what the accept tests run; every deferral is pinned False with its reject test."""
    c = _spec().capabilities
    assert (c.sm_lo, c.sm_hi) == _SM_RANGE, "the Rubin-line (107 <= SM <= 119), like the d256 rows; the SM100 d512 row stops at 103"
    assert c.d == frozenset({_D}) and c.d_envelope and c.d_pad_multiple == 8 and c.d_envelope_floor == 256 and not c.dqk_ge_dv
    assert c.dtypes == frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})
    assert not c.is_fp8 and not c.is_mxfp8 and c.out_dtypes == frozenset()
    assert c.causal and c.bottom_right and c.swa and c.gqa and c.right_band_widening
    assert not c.thd and not c.thd_declared_totals and not c.cu_seq_len, "THD starts declined: the packed path has not run on cc 10.7"
    assert not c.bias and not c.dbias and not c.decode
    assert c.layouts == frozenset({"bshd"}), "dense_flex staging is the SM100 row's claim, unvalidated on cc 10.7"
    assert not c.tile_ms and not c.tile_ns, "fixed geometry: no tile axis, the heuristics list {} as the complete record"
    for deferred in ("padded", "sink", "dsink", "deterministic"):
        assert not getattr(c, deferred), f"{deferred} is deferred: claim it together with its accept test here and the tracker line"
    sm100 = _spec("sdpa_bwd_sm100").capabilities
    assert (sm100.sm_lo, sm100.sm_hi) == (100, 103), "the SM100 row is untouched: this row is its own EngineSpec, not a widened sm_hi"


def test_adapter_is_the_sm100_chain_over_the_cc107_twin(monkeypatch):
    """The adapter class, its name / tag, and the two overrides: the stage-2 FILE is the cc 10.7 sibling and the RECORD is the
    8-stage ring arm, whatever ``api_dsl.STAGE2_2X2`` says -- while the SM100 row keeps its own selection."""
    from cudnn.sdpa.bwd import api_dsl
    from cudnn.sdpa.bwd.api_dsl_sm107_d512 import SdpaBwdDslSm107D512
    from cudnn.sdpa.bwd.config_sm100 import SM107_USABLE_DYN_SMEM_2X2, TemplateParams, TemplateParams2x2
    from cudnn.sdpa.bwd.engines import _adapter as adapter_of

    assert adapter_of("SdpaBwdDslSm107D512") is SdpaBwdDslSm107D512 and issubclass(SdpaBwdDslSm107D512, api_dsl.SdpaBwdDslSm100)
    assert SdpaBwdDslSm107D512._NAME == _ENGINE and SdpaBwdDslSm107D512._STAGE2_TAG == _TAG
    assert api_dsl.SdpaBwdDslSm100._NAME == "sdpa_bwd_sm100" and api_dsl.SdpaBwdDslSm100._STAGE2_TAG == "sdpa_bwd_sm100_stage2"
    fields = dict(dtype_qkv=2, window_right=None, window_left=None, bottom_right=False, thd_varlen=False)
    for twin in (False, True):
        monkeypatch.setattr(api_dsl, "STAGE2_2X2", twin)
        assert SdpaBwdDslSm107D512._stage2_file(None) == _FORK_FILE
        rec = SdpaBwdDslSm107D512._stage2_record(None, fields)
        assert isinstance(rec, TemplateParams2x2) and (rec.stages_kv, rec.cast_stages, rec.smem_cap_bytes) == (8, 2, SM107_USABLE_DYN_SMEM_2X2)
        # The SM100 row: the 4x1 role split by default, the SM100-arm twin when the module constant is flipped.
        assert api_dsl.SdpaBwdDslSm100._stage2_file(None) == (_SM100_FILE if twin else "sm100/bprop_d512_f16.py")
        if not twin:
            base = api_dsl.SdpaBwdDslSm100._stage2_record(None, fields)
            assert type(base) is TemplateParams, "the 4x1 record is the BASE TemplateParams (its PTX md5 is pinned)"
    assert api_dsl._rubin_line(107) and api_dsl._rubin_line(119) and not api_dsl._rubin_line(106) and not api_dsl._rubin_line(120)
    assert Path(_kernel_path()).is_file() and Path(_kernel_path(_SM100_FILE)).is_file()


# =========================================================================== graph builders / probes (host, fake cc 10.7)


def _without_the_dsl_arch_gate(monkeypatch):
    """``mismatch()`` declines every Rubin-line row through Rule 7 (``cutedsl_arch_requirement_error``) before any feature gate
    on a box whose DSL lacks sm_107a -- the public 4.7.0 on every SM100 CI lane.  These probes ask what the FEATURE gate says on a
    faked cc 10.7 device, so the arch gate is lifted here; Rule 7 keeps its own test (``test_rule7_dsl_without_sm107a_..``)."""
    from cudnn.sdpa.bwd import engines as bwd_engines

    monkeypatch.setattr(bwd_engines, "cutedsl_arch_requirement_error", lambda cc: None)


def _decline_reason(monkeypatch, engine=_ENGINE, cc=_RUBIN_CC, **kw):
    """Why ``engine`` declines the graph, or None if it would serve it: the row's own ``mismatch()`` against the analyzer's facts
    of a real graph on a FAKED device, so the feature gate -- not the arch gate -- answers."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    _without_the_dsl_arch_gate(monkeypatch)
    spec = _spec(engine)
    kw.setdefault("d", _D)
    try:
        g, _t, _outs = _half_bwd_graph(**kw)
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as e:
        return f"frontend refused the graph: {e}"
    facts = ga.analyze(g)
    if facts is None:
        return "analyzer did not recognise the graph"
    return mismatch(spec.capabilities, facts)


def _eligible(monkeypatch, cc=_RUBIN_CC, **kw):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd import engines as bwd_engines

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    _without_the_dsl_arch_gate(monkeypatch)
    kw.setdefault("d", _D)
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
        dict(diagonal_band_right_bound=64),
        dict(hq=8, hkv=2),
        dict(hq=16, hkv=1),
        dict(sq=500, skv=500),
        dict(sq=768, skv=1280),
        dict(sq=500, skv=1024, use_causal_mask_bottom_right=True),
        dict(d=264),
        dict(d=504),
        dict(b=3, hq=4),
    ],
    ids=[
        "dense-bf16",
        "dense-fp16",
        "default-scale",
        "causal",
        "bottom-right",
        "swa",
        "right-band",
        "gqa-r4",
        "mqa-r16",
        "non-tile-S",
        "768x1280",
        "br-ragged-sq",
        "d264",
        "d504",
        "b3",
    ],
)
def test_served_graph_passes_the_row_probe(monkeypatch, kw):
    assert _decline_reason(monkeypatch, **kw) is None


@pytest.mark.parametrize("cc", [(10, 7), (10, 8), (11, 0), (11, 9)], ids=["sm107", "sm108", "sm110", "sm119"])
def test_d512_half_graph_is_served_by_exactly_this_row_on_the_rubin_line(monkeypatch, cc):
    """No other python row serves the half d512 backward on the Rubin line (the SM100 d512 row stops at 103, the d256 rows are
    exact-256): the plan list offers this row alone (and no backend plan exists for it on cc 10.7)."""
    assert _eligible(monkeypatch, cc=cc) == {_ENGINE}
    assert _eligible(monkeypatch, cc=cc, use_causal_mask=True, hq=8, hkv=2) == {_ENGINE}
    assert _eligible(monkeypatch, cc=cc, d=264) == {_ENGINE}


def test_d256_half_graph_stays_the_d256_rows(monkeypatch):
    """The envelope floor is exclusive at 256: the Rubin d256 half row keeps that graph, this row declines it."""
    assert _eligible(monkeypatch, d=256) == {"sdpa_bwd_sm107"}
    assert _decline_reason(monkeypatch, d=256) is not None


@pytest.mark.parametrize("d", [128, 256])
def test_reject_head_dim_at_or_below_256(monkeypatch, d):
    assert _decline_reason(monkeypatch, d=d) is not None


def test_reject_head_dim_not_multiple_of_8(monkeypatch):
    assert _decline_reason(monkeypatch, d=260) is not None


def test_reject_rectangular_head_dims(monkeypatch):
    assert _decline_reason(monkeypatch, d=512, d_v=384) is not None


def test_reject_gqa_ratio_not_integer(monkeypatch):
    assert _decline_reason(monkeypatch, hq=6, hkv=4) is not None


def test_reject_bias(monkeypatch):
    assert _decline_reason(monkeypatch, bias=True) is not None


def test_reject_sink(monkeypatch):
    assert _decline_reason(monkeypatch, sink=True) is not None


def test_reject_thd(monkeypatch):
    """THD is the SM100 row's claim; this row starts without it (flip with a board-run accept + the tracker line)."""
    reason = _decline_reason(monkeypatch, thd=True)
    assert reason is not None and "THD" in reason, reason


def test_reject_decode_shaped(monkeypatch):
    assert _decline_reason(monkeypatch, sq=1) is not None


def test_reject_non_bshd_layout(monkeypatch):
    """``layouts={"bshd"}``: a BHSD-contiguous dO (what ``torch.randn(o.shape)`` hands over) is declined, not staged."""
    assert _decline_reason(monkeypatch, stride_fn=_bhsd_stride) is not None


def test_padding_mask_follows_the_padded_claim(monkeypatch):
    claimed = _spec().capabilities.padded
    assert (_decline_reason(monkeypatch, padded=True) is None) == claimed


def test_deterministic_follows_the_claim(monkeypatch):
    claimed = _spec().capabilities.deterministic
    assert (_decline_reason(monkeypatch, use_deterministic_algorithm=True) is None) == claimed


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 6), (12, 0), (8, 0), (9, 0)], ids=["sm100", "sm103", "sm106", "sm120", "sm80", "sm90"])
def test_reject_other_arch_lines(monkeypatch, cc):
    reason = _decline_reason(monkeypatch, cc=cc)
    assert reason is not None and "SM107-119" in reason, reason


def test_reject_quantized_graphs(monkeypatch):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    _without_the_dsl_arch_gate(monkeypatch)
    facts = ga.SdpaGraphFacts(
        is_backward=True, b=1, h_q=2, h_kv=2, s_q=256, s_kv=256, d_qk=512, d_v=512, dtype=cudnn.data_type.FP8_E4M3, is_fp8=True, device_cc=_RUBIN_CC
    )
    assert "serves only" in (mismatch(_spec().capabilities, facts) or "")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="constructs the adapter on a CUDA device (any arch; the cc is faked)")
def test_rule7_dsl_without_sm107a_declines_by_version_before_compile(monkeypatch):
    """Rule 7: a DSL that does not know ``sm_107a`` must decline with the installed version at ``check_support``, never fail
    inside the DSL at compile.  The device cc is faked to 10.7 and the DSL's sm_107a knowledge to False; the adapter is
    constructed directly (the probe path is ``mismatch``; this is the adapter's own backstop)."""
    import cudnn.frost.buffers as buffers
    import cudnn.frost.device as device
    from cudnn.sdpa.bwd.api_dsl_sm107_d512 import SdpaBwdDslSm107D512

    monkeypatch.setattr(device, "compute_capability", lambda dev: (10, 7))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    b, h, s, d = 1, 2, 256, _D
    dt = torch.bfloat16
    q, k, v, o, do = (_bshd_empty(b, s, h, d, dt) for _ in range(5))
    stats = torch.zeros(b, h, s, 1, device="cuda", dtype=torch.float32)
    dq, dk, dv = (_bshd_empty(b, s, h, d, dt) for _ in range(3))
    api = SdpaBwdDslSm107D512(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_do=do, sample_stats=stats, sample_dq=dq, sample_dk=dk, sample_dv=dv)
    with pytest.raises(NotImplementedError, match="sm_107a"):
        api.check_support()
    assert api._compiled is None, "declined before any compile"


# =========================================================================== the fork: structural pins (host)


def test_fork_module_pins_its_arm_at_import():
    """The loaded fork reports the arm it is: DESC_VERSION 0 hard-coded AND equal to the Cfg derivation, 8 / 2 stages,
    320 KiB of slabs under the 325 KiB line, the SM100 body's poll shape, SPIN_RING_WAITS False, the 256-row cluster span."""
    from cudnn.sdpa.bwd.config_sm100 import SM107_USABLE_DYN_SMEM_2X2, desc_roots_2x2, desc_version_2x2, smem_bytes_2x2, tmem_cols_2x2

    mod = _load_fork(**_rubin_arm())
    assert mod.DESC_VERSION == 0 == desc_version_2x2(mod.CFG)
    assert (mod.CFG.STAGES_KV, mod.CFG.CAST_STAGES, mod.CFG.D_CHUNK, mod.CFG.N_CHUNKS) == (8, 2, 64, 8)
    assert smem_bytes_2x2(mod.CFG) == 320 * 1024 and mod.CFG.SMEM_CAP_BYTES == SM107_USABLE_DYN_SMEM_2X2 == 325 * 1024
    assert max(off for _, off in desc_roots_2x2(mod.CFG)) == 253952, "the zero-margin root (sRingV stage 7)"
    assert tmem_cols_2x2(mod.CFG) == 256, "256 TMEM columns: no is_exclusive needed on the 576-column line"
    assert (mod.POLL_TIGHT_ITERS, mod.POLL_SLEEP_NS) == (128, 128), "the backward's poll shape (the forwards ship 32 / 128)"
    assert mod.SPIN_RING_WAITS is False
    assert mod.CLUSTER_Q_ROWS == 256 and mod.CFG.KV_SHARE == 2 and mod.CFG.RING_EMPTY_ARRIVERS == 2
    assert mod.RUBIN_ARM == _rubin_arm()


def test_fork_refuses_the_sm100_arm_and_a_ninth_stage():
    """RED-first detectors the fork carries at import: the SM100 ring levers (4 / 1 / 227 KiB) must not render this file, and
    a 9th stage -- which puts sRingV stage 7 AT the version-0 window and flips the derived version to 1 -- must trip the
    hard-coded DESC_VERSION's agreement check before it can silently widen."""
    from cudnn.sdpa.bwd.config_sm100 import SM107_USABLE_DYN_SMEM_2X2

    # The SM100 arm (4 / 1 / 227 KiB): its last root ends 64 KiB short of the window, so the zero-margin pin trips first.
    with pytest.raises(ValueError, match="version-0 window"):
        _load_fork()  # TemplateParams2x2() defaults = the SM100 arm
    # The right rings but one cast stage: the roots are the arm's, the arm pin is what refuses it.
    with pytest.raises(ValueError, match="cc 10.7 arm"):
        _load_fork(stages_kv=8, cast_stages=1, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2)
    # A 9th stage: the derived version flips to 1 and disagrees with the hard-coded 0 before anything else is checked.
    with pytest.raises(ValueError, match="disagrees with the module constant"):
        _load_fork(stages_kv=9, cast_stages=1, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2)


def test_fork_source_pins():
    """Source-level rules of the sibling: DESC_VERSION bound ONCE as the literal 0 and wired into every SmemTile (no re-literalled
    ``desc_version=``); exactly one SPIN_RING_WAITS definition, threaded into exactly one wait site (``_wait_plain``'s pair-local
    branch) and never onto the two polled barriers (the three ``poll=_KV_SHARED`` sites stay); the mbarrier init ledger of the
    SM100 body (every init a named CFG constant); Rule 6 on both kernels; plain launches, no exclusive TMEM, no row reduction."""
    src = _source()
    code = _code_lines(src)
    tiles = re.findall(r"= SmemTile\((.*?)\n    \)", src, flags=re.S)
    assert len(tiles) == 6 and all("desc_version=DESC_VERSION" in t for t in tiles), len(tiles)
    assert len(re.findall(r"^DESC_VERSION: int = 0$", code, re.M)) == 1, "DESC_VERSION is the literal 0, bound once"
    assert "desc_version_2x2(CFG) == DESC_VERSION" in code and not re.search(r"desc_version=[01]\b", code)
    assert "_LAST_DESC_ROOT + vBytesPerStage == TCGEN05_V0_ADDR_LIMIT_2X2" in code, "the zero-margin fact is _require'd"
    assert len(re.findall(r"^SPIN_RING_WAITS: bool = (?:True|False)$", code, re.M)) == 1
    assert code.count("spin=SPIN_RING_WAITS") == 1, "one threaded site: _wait_plain's pair-local branch"
    assert re.search(r"if cutlass\.const_expr\(poll\):\n\s+_poll_wait\(mb, phase\)\n\s+else:\n\s+wait\(mb, phase, spin=SPIN_RING_WAITS\)", code)
    code_only = _code_only(src)  # strings blanked too: the docstring quotes ``wait(spin=True)`` when it explains the hang
    spin_literals = re.findall(r"wait\(mb, phase, spin=True\)", code_only)
    assert len(spin_literals) == 1 and code_only.count("spin=True") == 1, "the only spin literal is the WAIT_FORM 1 diagnostic arm"
    polled = re.findall(r"_wait_b\(\s*bars\.mb_tma_ring_(empty|full)\[[^\]]+\]\.smem_ptr,(?:[^()]|\([^()]*\))*?poll=_KV_SHARED,", src, flags=re.S)
    assert sorted(polled) == ["empty", "empty", "full"] and src.count("poll=_KV_SHARED,") == 3, polled
    inits = re.findall(r"init_count=CFG\.(\w+)", src)
    assert sorted(inits) == sorted(
        ["ONE_LANE", "ONE_LANE", "CGA_M", "RING_EMPTY_ARRIVERS", "ONE_LANE", "ACC_EMPTY_ARRIVERS", "COMPUTE_LANES", "ONE_WARP", "TMEM_DEALLOC_ARRIVERS"]
    )
    assert src.count("@cute.kernel") == 2 == src.count('.set_name_prefix("cudnn", remove_cutlass_symbol=True)')
    assert _code_only(src).count(".launch(") == 2, "plain launches (the THD clamp kernel + the main kernel); the DSL launcher sets the oversized-SMEM attribute"
    for gone in ("is_exclusive", "tmem_load_max_reduction_tile", "ld.red", "cp_async_bulk_shared_cluster_shared_cta", "tcgen05_cp", "mma_ts("):
        assert gone not in _code_only(src), gone  # the prose names what is absent; the code must not
    assert "bars.mb_tmem_dealloc.arrive()" in src and "bars.mb_tmem_dealloc.arrive_on_peer(cta_id_x ^ cutlass.Int32(1))" in src
    assert "RUBIN_ARM = dict(stages_kv=8, cast_stages=2, smem_cap_bytes=SM107_USABLE_DYN_SMEM_2X2)" in code
    assert 'raise ValueError(f"bprop_d512_f16_2x2_sm107: {msg}")' in code


# The code lines the fork may ADD / REMOVE relative to the SM100 body (strings and comments blanked first, so the docstring
# and the _require messages do not count).  Any other difference between the two files fails: a fix landed in one sibling
# only, or an unlisted body delta.
# The condition lines of the fork's three MULTI-LINE `_require(` blocks (opener / condition / blanked message / `)`): the
# generic opener, message and paren patterns below are anchored to these in the test, so a fourth three-line
# `_require(<anything>)` cannot ride in on them.
_FORK_REQUIRE_CONDITIONS = (
    r"^    _LAST_DESC_ROOT \+ vBytesPerStage == TCGEN05_V0_ADDR_LIMIT_2X2,$",
    r"^    CFG\.STAGES_KV == RUBIN_ARM\[",
    r"^    CFG\.SMEM_CAP_BYTES == SM107_USABLE_DYN_SMEM_2X2 and smem_bytes_2x2\(CFG\) == 320 \* 1024,$",
)
_FORK_ADDED = (
    r"^RUBIN_ARM = dict\(",
    r"^PARAMS: TemplateParams = globals\(\)\.get\(\s*, TemplateParams2x2\(\*\*RUBIN_ARM\)\)$",  # the string literal is blanked
    r"^    SM107_USABLE_DYN_SMEM_2X2,$",
    r"^    TCGEN05_V0_ADDR_LIMIT_2X2,$",
    r"^    desc_roots_2x2,$",
    r"^SPIN_RING_WAITS: bool = False$",
    r"^            wait\(mb, phase, spin=SPIN_RING_WAITS\)$",
    r"^DESC_VERSION: int = 0$",
    r"^_require\(desc_version_2x2\(CFG\) == DESC_VERSION, ",
    r"^_LAST_DESC_ROOT = max\(",
    r"^_require\($",
    *_FORK_REQUIRE_CONDITIONS,
    r"^\s*,$",  # the blanked f-string message line of a multi-line _require
    r"^\)$",
)
_FORK_REMOVED = (
    r"^PARAMS: TemplateParams = globals\(\)\.get\(\s*, TemplateParams2x2\(\)\)$",
    r"^            wait\(mb, phase\)$",
    r"^DESC_VERSION: int = desc_version_2x2\(CFG\)$",
)
# The fork's whole delta, counted: 22 added / 3 removed code lines (3 imports, RUBIN_ARM, PARAMS, SPIN_RING_WAITS + its one
# site, DESC_VERSION + its one-line _require, _LAST_DESC_ROOT, three 4-line _require blocks).  A new delta changes the count.
_FORK_DELTA_LINES = (22, 3)


def test_fork_code_diff_is_only_the_listed_deltas():
    """``diff sm100/ sm107/`` is the review surface: with strings and comments blanked, every added or removed CODE line of the
    fork matches one allowlisted delta pattern.  A change to either sibling that is not mirrored (or a new body delta that is
    not listed here and in the fork's docstring) fails this test."""
    a = [ln.rstrip() for ln in _code_only(_source(_SM100_FILE)).splitlines()]
    b = [ln.rstrip() for ln in _code_only(_source(_FORK_FILE)).splitlines()]
    added, removed = [], []
    for ln in difflib.unified_diff(a, b, lineterm="", n=0):
        if ln.startswith(("+++", "---", "@@")) or not ln[1:].strip():
            continue
        (added if ln[0] == "+" else removed).append(ln[1:])
    bad_add = [ln for ln in added if not any(re.match(p, ln) for p in _FORK_ADDED)]
    bad_rm = [ln for ln in removed if not any(re.match(p, ln) for p in _FORK_REMOVED)]
    assert not bad_add and not bad_rm, f"unlisted deltas between the SM100 body and the fork:\n+ {bad_add}\n- {bad_rm}"
    assert added and removed, "the fork must differ from the SM100 body in exactly the listed lines (it does not differ at all?)"
    # Anchor the generic patterns: every `_require(` opener is followed by one of the listed condition lines, a blanked
    # message line and the closing paren, and the three generic patterns match exactly as many lines as there are openers.
    openers = [i for i, ln in enumerate(added) if re.match(r"^_require\($", ln)]
    for i in openers:
        block = added[i : i + 4]
        assert len(block) == 4 and any(re.match(p, block[1]) for p in _FORK_REQUIRE_CONDITIONS), f"unanchored multi-line _require: {block}"
        assert re.match(r"^\s*,$", block[2]) and block[3] == ")", f"unanchored multi-line _require: {block}"
    assert len(openers) == len(_FORK_REQUIRE_CONDITIONS) == sum(bool(re.match(r"^\s*,$", ln)) for ln in added) == sum(ln == ")" for ln in added), openers
    assert (len(added), len(removed)) == _FORK_DELTA_LINES, (
        f"the fork's delta is exactly {_FORK_DELTA_LINES[0]} added / {_FORK_DELTA_LINES[1]} removed code lines, got {len(added)} / {len(removed)}: "
        "a new delta must be listed in the fork's docstring, in _FORK_ADDED / _FORK_REMOVED and in _FORK_DELTA_LINES"
    )


# =========================================================================== renderings: PTX md5 record, SASS pins, Rule S6 (host trace-compiles)

_S2_MASKS = {"dense_bf16": {}, "causal_bf16": dict(window_right=0), "thd_bf16": dict(thd_varlen=True)}


def _ptx_probe(tmp_path, arch, record, timeout=1500):
    dump = tmp_path / f"{arch}_fork_{record}"
    dump.mkdir()
    script = dump / "ptx_probe.py"
    script.write_text(_STAGE2_PTX_PROBE)
    kw = dict(dtype_qkv=2, kernel_file=_FORK_FILE, twin=True, **_rubin_arm(), **_S2_MASKS[record])
    import json

    proc = subprocess.run([sys.executable, str(script), str(dump), arch, json.dumps(kw)], capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, f"{arch} trace-compile of the fork ({record}) failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return dict(ln.split(maxsplit=1) for ln in proc.stdout.splitlines() if ln.startswith(("PTX_MD5", "CLUSTER_Q_ROWS", "DESC_VERSION", "N_CHUNKS")))


def test_fork_md5_record_is_committed_and_complete():
    assert _MD5_RECORD.is_file(), _MD5_RECORD
    dsl, want = _parse_md5_record_2x2(_MD5_RECORD)
    assert dsl and dsl.startswith("nvidia-cutlass-dsl"), dsl
    assert set(want) == set(_S2_MASKS), (sorted(want), sorted(_S2_MASKS))


def _parse_md5_record_2x2(f):
    dsl, want = None, {}
    for ln in f.read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[len("dsl=") :].strip()
        m = re.match(r"stage2_2x2 sm_107a (\S+) rc=0 ptx_md5=([0-9a-f]{32})", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


@pytest.mark.parametrize("record", list(_S2_MASKS))
def test_fork_rendering_ptx_md5_is_recorded(tmp_path, record):
    """The fork's sm_107a rendering matches the committed record, refreshed deliberately for protocol changes.  Compares only when the installed DSL build is the
    recorded one (PTX text is a function of it) and knows sm_107a; skips otherwise (Rule 7)."""
    from cudnn.frost.buffers import cutedsl_state

    dsl, want = _parse_md5_record_2x2(_MD5_RECORD)
    _installed, version = cutedsl_state()
    have = " ".join(version) if version else None
    if dsl is not None and have != dsl:
        pytest.skip(f"the md5 record was rendered with {dsl}; installed {have}: PTX text differs by DSL build, re-render the record on the board")
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a")
    out = _ptx_probe(tmp_path, "sm_107a", record)
    assert out["DESC_VERSION"] == "0" and out["CLUSTER_Q_ROWS"] == "256" and out["N_CHUNKS"] == "8", out
    got = out["PTX_MD5"].strip()
    print(f"\ncc 10.7 stage-2 2x2 fork {record}: PTX md5 {got} (recorded {want[record]})")
    assert got == want[record], f"{record}: PTX md5 {got} != the recorded {want[record]} -- the fork's rendering changed; re-record deliberately"


@pytest.mark.parametrize("record", ["dense_bf16", "causal_bf16"])
def test_fork_trace_compiles_for_sm_100a(tmp_path, record):
    """Rule S6: the other arch line's lowering of the fork, from any box whose DSL knows sm_100a (the public 4.7.0 does).  The
    320 KiB arm does not FIT an SM100 CTA (227 KiB) -- dynamic SMEM is a launch attribute, not a PTX fact -- so this is a
    lowering smoke only; the arm's own pins (``_require``) hold at import either way."""
    if not arch_known_to_the_dsl("sm_100a"):
        pytest.skip("this cutlass-dsl has no sm_100a")
    out = _ptx_probe(tmp_path, "sm_100a", record)
    assert out["DESC_VERSION"] == "0" and out["CLUSTER_Q_ROWS"] == "256" and out["N_CHUNKS"] == "8", out


# sm_107a SASS of the fork (MEASURED on the board's toolchain, internal DSL 0.3.0+20260728 + the cuda-39029786 nvdisasm,
# 2026-10-01): STL 0 / LDL 0 (the 8-warp body: ptxas C7508 drops the register split, USETMAXREG 0 -- recorded, not required),
# UTCHMMA 64 (8 chunks x 4 k-steps x 2 BMMs), UTMALDG 32, LDTM 2 on the dense body and 4 on the causal one: two tcgen05.ld
# per traced compute body, and the internal toolchain emits TWO masked-body copies for the three-range split where the public
# 4.7.0 + CUDA 13.3 ptxas keeps three (6; the SM100 twin's pin on that toolchain).  The pin accepts either toolchain's count.
_FORK_SPILL_PINS = {"STL": 0, "LDL": 0}


@pytest.mark.parametrize("arm", ["dense", "causal"])
def test_fork_sass_pins_sm_107a(tmp_path, arm):
    params = dict(dtype_qkv=2, kernel_file=_FORK_FILE, twin=True, **_rubin_arm(), **({} if arm == "dense" else dict(window_right=0)))
    st, expect = _stage2_sass_probe(tmp_path, "sm_107a", params, tag=f"fork_stage2_2x2_{arm}")
    assert expect["DESC_VERSION"] == 0 and expect["CLUSTER_Q_ROWS"] == 256 and expect["N_CHUNKS"] == 8
    assert st["MEMBAR_GPU"] == 0 and st["CGAERRBAR"] == 0, st
    assert st["UBLKCP"] == 0, f"no DSMEM bulk copy: {st}"
    assert st["UTCHMMA"] == 64 and st["UTMALDG"] == 32, st
    assert st["LDTM"] == 2 if arm == "dense" else st["LDTM"] in (4, 6), f"two tcgen05.ld per traced compute body: {st}"
    assert_no_new_spills(st, _FORK_SPILL_PINS, tag=f"sm_107a {arm}: ")


# =========================================================================== GPU accept: the chain against the fp64 oracle, engine PINNED


def _run(b=2, hq=2, hkv=None, sq=512, skv=512, d=_D, dt=torch.bfloat16, keep=None, omit_scale=False, seed=0, poison=None, runs=1, **sdpa_kwargs):
    """The sm107 suite's driver with THIS row pinned: build, pin, execute ``runs`` times, hand back every run's (dQ, dK, dV)
    plus the fp64 oracle.  Inputs unit-normal on a CPU generator (the dataset does not depend on the GPU's SM count)."""
    hkv = hq if hkv is None else hkv
    group = hq // hkv
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(bb, s, h):
        return (torch.randn(bb, s, h, d, generator=gen) * 0.5).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, do = draw(b, sq, hq), draw(b, sq, hq)
    k, v = draw(b, skv, hkv), draw(b, skv, hkv)
    if omit_scale:  # attn_scale off the graph = no scaling (1.0); pre-scaled Q keeps the logits in their usual range
        q.mul_(d**-0.5)
    o64, lse64, all_masked, dq_r, dk_r, dv_r = _reference64(q, k, v, do, keep, group, scale=1.0 if omit_scale else None)
    o = _bshd_empty(b, sq, hq, d, dt)
    o.copy_(o64.to(dt))
    lse = lse64.float()
    if all_masked is not None:
        lse = lse.masked_fill(all_masked, 0.0)
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, d=d, dt=dt, scale=None if omit_scale else "default", **sdpa_kwargs)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    assert g._compiled_plans[g._plan_index]._prepared is not None, "the row lowers through the prepared-launch contract"
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    dq, dk, dv = _bshd_empty(b, sq, hq, d, dt), _bshd_empty(b, skv, hkv, d, dt), _bshd_empty(b, skv, hkv, d, dt)
    pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: lse.unsqueeze(-1).contiguous(), dq_t: dq, dk_t: dk, dv_t: dv}
    outs = []
    for _ in range(runs):
        if poison is not None:
            for x in (dq, dk, dv):
                x.fill_(poison)
        g.execute(pack, ws)
        torch.cuda.synchronize()
        outs.append(tuple(x.clone() for x in (dq, dk, dv)))
    run = _Run(outs, (dq_r, dk_r, dv_r), dt)
    # For the launch-count pins: one more execute of the SAME plan over the same pack; ``outputs`` are the live dQ / dK / dV.
    run.execute = lambda: g.execute(pack, ws)
    run.outputs = (dq, dk, dv)
    return run


@pytest.fixture
def watchdog(request):
    with process_watchdog(_WATCHDOG_S, f"the cc 10.7 d512 chain in {request.node.nodeid}"):
        yield


@requires_rubin
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_dense(dt, watchdog):
    _run(dt=dt).check()


@requires_rubin
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_causal_dtypes(dt, watchdog):
    _run(dt=dt, keep=_causal_keep(512, 512), use_causal_mask=True).check()


@requires_rubin
def test_causal_bottom_right(watchdog):
    _run(keep=_causal_keep(512, 512, bottom_right=True), use_causal_mask_bottom_right=True).check()


@requires_rubin
def test_causal_bottom_right_rectangular(watchdog):
    """S_kv > S_q shifts the diagonal, which the stage-3 K-trim has to follow."""
    _run(sq=512, skv=1024, keep=_causal_keep(512, 1024, bottom_right=True), use_causal_mask_bottom_right=True).check()


@requires_rubin
def test_sliding_window(watchdog):
    _run(keep=_causal_keep(512, 512, left=256), use_causal_mask=True, diagonal_band_left_bound=256).check()


@requires_rubin
def test_right_band_widening(watchdog):
    """``diagonal_band_right_bound`` alone (with use_causal_mask the bound is forced back to 0)."""
    _run(keep=_causal_keep(512, 512, right=64), diagonal_band_right_bound=64).check()


@requires_rubin
@pytest.mark.parametrize("hq,hkv", [(8, 8), (8, 4), (8, 2), (8, 1), (6, 3)], ids=["mha8", "r2", "r4", "mqa-r8", "r2-h6"])
def test_gqa_mqa(hq, hkv, watchdog):
    _run(hq=hq, hkv=hkv, sq=256, skv=256).check()


@requires_rubin
@pytest.mark.parametrize("d", [264, 320, 384, 504, _D])
def test_head_dim_band(d, watchdog):
    """d in (256, 512], any multiple of 8 (264 and 504 narrow the stage-3 epilogue store vector from 32 B to 16 B)."""
    _run(sq=256, skv=256, d=d).check()


@requires_rubin
@pytest.mark.parametrize("sq,skv", [(500, 500), (300, 200), (257, 129), (384, 640)])
def test_non_tile_multiple_seqlens(sq, skv, watchdog):
    _run(sq=sq, skv=skv).check()


@requires_rubin
@pytest.mark.parametrize("sq,skv", [(500, 500), (1000, 1000)])
def test_non_tile_multiple_causal(sq, skv, watchdog):
    _run(sq=sq, skv=skv, keep=_causal_keep(sq, skv), use_causal_mask=True).check()


@requires_rubin
def test_default_attn_scale(watchdog):
    _run(omit_scale=True).check()


@requires_rubin
def test_batch_gt_one(watchdog):
    _run(b=3, hq=4, sq=256, skv=384).check()


@requires_rubin
@pytest.mark.parametrize("hq", [1, 4])
def test_short_trip_single_kv_tile_per_cluster(hq, watchdog):
    """One kv tile per cluster (S_kv = 128, acc_total = 1 < STAGES_ACC): the geometry whose TMEM release the min(total, stages)
    drain never gated (review P1; the SM100 twin pins the host model).  Eight launches against the fp32 reference."""
    _run(b=1, hq=hq, sq=256, skv=128, runs=8).check()


@requires_rubin
def test_forced_head_chunking_is_the_same_result(watchdog):
    """The head-chunk loop (several stage-2 / stage-3 launches with ``head_base``) forced at ``chunk = group``: the result must
    equal the single-chunk one bitwise (same kernels, same data per head) and the oracle."""
    from cudnn.sdpa.bwd import api_dsl

    kw = dict(hq=8, hkv=2, sq=256, skv=256, keep=_causal_keep(256, 256), use_causal_mask=True, poison=float("nan"))
    one = _run(**kw)
    with patch.object(api_dsl, "_sm100_head_chunk", side_effect=lambda *a, group=1, **k: group):
        chunked = _run(**kw)
    chunked.check()
    for name, x, y in zip(("dQ", "dK", "dV"), one.outs[0], chunked.outs[0]):
        assert torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16)), f"{name}: head chunking changed the result"


@requires_rubin
@pytest.mark.parametrize("dt", _DTYPES, ids=_DTYPE_IDS)
def test_two_launches_are_bitwise(dt, watchdog):
    run = _run(dt=dt, keep=_causal_keep(512, 512), use_causal_mask=True, poison=float("nan"), runs=2)
    run.check()
    for name, x, y in zip(("dQ", "dK", "dV"), run.outs[0], run.outs[1]):
        assert torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16)), f"{name}: not deterministic across launches"


@requires_rubin
def test_workspace_is_build_time_honest():
    g, _t, _o = _build_graph(b=2, hq=2, sq=256, skv=256, d=_D)
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == g.get_workspace_size() > 0


# =========================================================================== the bitwise twin: the fork vs the SM100 body at Rubin params (board)


def _direct_capture(Adapter, tensors, *, b, hq, hkv, sq, skv, dt, akw):
    """Construct ``Adapter`` directly (no engine row admits the SM100 body on cc 10.7), compile, execute once into NaN-poisoned
    outputs and a byte-poisoned workspace; return int16 views of dQ / dK / dV and the S / dS workspace regions + the stage-2
    file that served."""
    from cudnn.sdpa.bwd import api_dsl

    served = []
    original = api_dsl.load_template

    def spy(path, params, tag="template"):
        if "stage2" in tag:
            served.append(path.rsplit("/", 1)[-1])
        return original(path, params, tag)

    with patch.object(api_dsl, "load_template", spy):
        d = _D
        dq, dk, dv = _bshd_empty(b, sq, hq, d, dt, float("nan")), _bshd_empty(b, skv, hkv, d, dt, float("nan")), _bshd_empty(b, skv, hkv, d, dt, float("nan"))
        api = Adapter(
            sample_q=tensors["q"],
            sample_k=tensors["k"],
            sample_v=tensors["v"],
            sample_o=tensors["o"],
            sample_do=tensors["do"],
            sample_stats=tensors["stats"],
            sample_dq=dq,
            sample_dk=dk,
            sample_dv=dv,
            scale_softmax=1.0 / math.sqrt(d),
            **akw,
        )
        api.check_support()
        api.compile()
        ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8).fill_(0xBD)
        api.execute(
            q_tensor=tensors["q"],
            k_tensor=tensors["k"],
            v_tensor=tensors["v"],
            o_tensor=tensors["o"],
            do_tensor=tensors["do"],
            stats_tensor=tensors["stats"],
            dq_tensor=dq,
            dk_tensor=dk,
            dv_tensor=dv,
            scale_softmax=1.0 / math.sqrt(d),
            workspace=ws,
        )
        torch.cuda.synchronize()
    align = lambda n: (n + 127) // 128 * 128
    delta = align(b * hq * (-(-sq // 128) * 128) * 4)
    region = align(b * api._qh_chunk * api._sq_pad * api._skv_pad * 2)
    s_ws, ds_ws = ws[delta : delta + region].clone(), ws[delta + region : delta + 2 * region].clone()
    return [x.contiguous().view(torch.int16).clone() for x in (dq, dk, dv)] + [s_ws, ds_ws], served


@requires_rubin
@pytest.mark.parametrize(
    "case",
    [
        dict(b=1, hq=8, hkv=8, sq=1024, skv=1024),
        dict(b=1, hq=8, hkv=8, sq=2048, skv=2048, is_causal=True),
        dict(b=1, hq=8, hkv=8, sq=1024, skv=1024, is_causal=True, window_size_left=255),
        dict(b=2, hq=4, hkv=4, sq=768, skv=1280, is_causal=True, causal_bottom_right=True),
        dict(b=1, hq=8, hkv=2, sq=1024, skv=1024),
        dict(b=1, hq=4, hkv=4, sq=1024, skv=1024, dt=torch.float16),
    ],
    ids=["dense", "causal_2k", "swa", "br_rect_b2", "gqa", "dense_fp16"],
)
def test_fork_is_bitwise_the_sm100_body_at_rubin_params(monkeypatch, case, watchdog):
    """The fork (through its adapter) and the SM100 2x2 body at the same ring arm (``SdpaBwdDslSm100`` constructed directly
    with ``STAGE2_2X2`` flipped; the base adapter detects the Rubin-line device and builds 8 / 2 / 325 KiB) produce BITWISE the
    same S / dS workspace and dQ / dK / dV: the deltas are wait-form and constants (the rendering is even PTX-identical), so a
    mismatch is a real body divergence, never accumulation noise -- ``torch.equal`` on int16 views, not a tolerance."""
    from cudnn.sdpa.bwd import api_dsl
    from cudnn.sdpa.bwd.api_dsl_sm107_d512 import SdpaBwdDslSm107D512

    case = dict(case)
    dt = case.pop("dt", torch.bfloat16)
    b, hq, hkv, sq, skv = (case.pop(k) for k in ("b", "hq", "hkv", "sq", "skv"))
    d = _D
    gen = torch.Generator(device="cpu").manual_seed(1811)

    def draw(bb, s, h):
        return (torch.randn(bb, s, h, d, generator=gen) * 0.5).to(device="cuda", dtype=dt).permute(0, 2, 1, 3)

    q, do, k, v = draw(b, sq, hq), draw(b, sq, hq), draw(b, skv, hkv), draw(b, skv, hkv)
    keep = None
    if case.get("is_causal"):
        left = case.get("window_size_left")
        keep = _causal_keep(sq, skv, bottom_right=bool(case.get("causal_bottom_right")), left=None if left is None else left + 1)
    o64, lse64, all_masked, _, _, _ = _reference64(q, k, v, do, keep, hq // hkv)
    o = _bshd_empty(b, sq, hq, d, dt)
    o.copy_(o64.to(dt))
    lse = lse64.float()
    if all_masked is not None:
        lse = lse.masked_fill(all_masked, 0.0)
    tensors = dict(q=q, k=k, v=v, o=o, do=do, stats=lse.unsqueeze(-1).contiguous())
    geom = dict(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv, dt=dt, akw=case)
    monkeypatch.setattr(api_dsl, "STAGE2_2X2", True)
    base, served_base = _direct_capture(api_dsl.SdpaBwdDslSm100, tensors, **geom)
    monkeypatch.setattr(api_dsl, "STAGE2_2X2", False)
    fork, served_fork = _direct_capture(SdpaBwdDslSm107D512, tensors, **geom)
    assert served_base == ["bprop_d512_f16_2x2.py"] and served_fork == ["bprop_d512_f16_2x2.py"], (served_base, served_fork)
    for name, x, y in zip(("dQ", "dK", "dV", "S_ws", "dS_ws"), base, fork):
        assert torch.isfinite(y.view(torch.bfloat16 if dt == torch.bfloat16 else torch.float16).float()).all() or name.endswith("_ws"), name
        assert torch.equal(x, y), f"{name}: the fork is NOT bitwise the SM100 body at Rubin params ({(x != y).sum().item()} of {x.numel()} differ)"


# =========================================================================== stage 3 under GQA: the dQ single launch this row INHERITS (api_dsl.DQ_SINGLE_LAUNCH)

_MM_TAGS = ("sdpa_bwd_sm100_mm_lo", "sdpa_bwd_sm100_mm_hi")
_STAGE3_GEMM_KERNEL = "bprop_matmul"  # the stage-3 GEMM kernel's name fragment (``_bprop_matmul_bh_sm100_kernel``)


def test_row_inherits_the_dq_single_launch_lever():
    """``api_dsl.DQ_SINGLE_LAUNCH`` (True: under GQA the dQ GEMM is ONE launch per head chunk -- the stage-3 template's
    ``b_head_group = group`` -- instead of one launch per group member) is read by ``SdpaBwdDslSm100.compile``, which this row
    does NOT override: the lever is this row's default too, through the SM100 host's ``_dq_launches`` arithmetic, with
    ``Params.dq_b_head_group`` the 14th (last, appended) element of the compiled host's config tuple.  Rule 9: one engine's
    evidence is not the other's, so the row carries its own pins (this section) instead of inheriting the SM100 file's."""
    import dataclasses

    from cudnn.sdpa.bwd import api_dsl
    from cudnn.sdpa.bwd.api_dsl_sm107_d512 import SdpaBwdDslSm107D512
    from cudnn.sdpa.bwd.kernels.sm100.prepared_host import Params, _dq_launches

    assert "compile" not in vars(SdpaBwdDslSm107D512) and SdpaBwdDslSm107D512.compile is api_dsl.SdpaBwdDslSm100.compile
    assert api_dsl.DQ_SINGLE_LAUNCH is True, "one dQ launch per chunk is what ships; the pins below flip it OFF for the twin"
    fields = dataclasses.fields(Params)
    assert len(fields) == 14 and fields[-1].name == "dq_b_head_group" and fields[-1].default == 1
    assert (_dq_launches(1, 1), _dq_launches(8, 8), _dq_launches(8, 1), _dq_launches(16, 16), _dq_launches(16, 1)) == (1, 1, 8, 1, 16)
    for group, bhg in ((8, 2), (16, 4), (2, 4), (1, 2)):
        with pytest.raises(ValueError, match="b_head_group"):
            _dq_launches(group, bhg)


def _spy_dq_record(monkeypatch, *, hq, hkv, single, sq=256, skv=256, dt=torch.bfloat16):
    """``compile()`` of THIS row's adapter on ANY CUDA device (the cc faked to 10.7, the DSL's sm_107a knowledge faked True),
    with the stage-3 records spied off ``load_template`` and the host trace-compile STUBBED at ``compile_host`` (the sm_107a
    lowering needs the board's DSL; the records and the ``Params`` are host facts, which is what this pins).  Returns
    ``(records, api, (params, sm, symbol))``: the two stage-3 ``MatmulTemplateParams``, the adapter and what ``compile_host``
    was handed."""
    import cudnn.frost.buffers as buffers
    import cudnn.frost.device as device
    from cudnn.sdpa.bwd import api_dsl, prepared_sm100
    from cudnn.sdpa.bwd.api_dsl_sm107_d512 import SdpaBwdDslSm107D512
    from cudnn.sdpa.bwd.kernels.sm100 import prepared_host

    monkeypatch.setattr(device, "compute_capability", lambda dev: _RUBIN_CC)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(api_dsl, "DQ_SINGLE_LAUNCH", single)
    records, hosts = {}, []
    original = api_dsl.load_template

    def spy(path, params, tag="template"):
        if tag in _MM_TAGS:
            records[tag] = params
        return original(path, params, tag)

    def stub_compile_host(stage2, mm_lo, mm_hi, params, geometry, regions, dtype, sm, cache_key, symbol="frost_sdpa_bwd_sm100_prepared"):
        hosts.append((params, sm, symbol))
        return object()

    monkeypatch.setattr(api_dsl, "load_template", spy)
    monkeypatch.setattr(prepared_host, "compile_host", stub_compile_host)
    monkeypatch.setattr(prepared_sm100, "positional_entry", lambda entry: (lambda *args: None))
    b, d = 1, _D
    q, do, o, dq = (_bshd_empty(b, sq, hq, d, dt) for _ in range(4))
    k, v, dk, dv = (_bshd_empty(b, skv, hkv, d, dt) for _ in range(4))
    stats = torch.zeros(b, hq, sq, 1, device="cuda", dtype=torch.float32)
    api = SdpaBwdDslSm107D512(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_do=do, sample_stats=stats, sample_dq=dq, sample_dk=dk, sample_dv=dv)
    api.check_support()
    api.compile()
    assert records.keys() == set(_MM_TAGS), sorted(records)
    assert len(hosts) == 1, hosts
    return records, api, hosts[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="constructs the adapter on a CUDA device (any arch; the cc is faked)")
@pytest.mark.parametrize("hq,hkv", [(16, 2), (8, 1), (8, 8)], ids=["gqa16-2", "mqa8-1", "mha8"])
@pytest.mark.parametrize("single", (True, False), ids=("one-launch", "per-member"))
def test_dq_record_carries_the_group_under_gqa(monkeypatch, hq, hkv, single):
    """The record spy: under GQA the dQ rendering (``mm_hi``) this row compiles carries ``b_head_group == group`` with the lever
    on and 1 with it off; dV / dK (``mm_lo``) stay 1; MHA renders 1 either way.  The adapter copies the record's value
    (``_dq_b_head_group``, ONE source of truth) and the compiled host's ``Params`` carry it as their 14th element -- the value
    ``prepared_host.host`` hands ``_dq_launches`` at trace time, so a record / host drift raises instead of pairing a Q head
    with the wrong K head.  RED when ``compile()`` forces the record to 1: the ``one-launch`` GQA / MQA cases fail on the
    record, the adapter's copy and the Params alike."""
    import dataclasses

    from cudnn.sdpa.bwd.kernels.sm100.prepared_host import _dq_launches

    records, api, (params, sm, symbol) = _spy_dq_record(monkeypatch, hq=hq, hkv=hkv, single=single)
    group = hq // hkv
    want = group if (single and group > 1) else 1
    assert records["sdpa_bwd_sm100_mm_hi"].b_head_group == want, records["sdpa_bwd_sm100_mm_hi"]
    assert records["sdpa_bwd_sm100_mm_lo"].b_head_group == 1, records["sdpa_bwd_sm100_mm_lo"]
    assert (api._gqa_group, api._dq_b_head_group) == (group, want)
    config = dataclasses.astuple(params)
    assert params.dq_b_head_group == want and len(config) == 14 and config[-1] == want, params
    assert (params.heads, params.kv_heads, params.chunk % group) == (hq, hkv, 0), params
    assert sm == 107 and symbol == f"frost_{_ENGINE}_prepared", (sm, symbol)
    assert _dq_launches(group, want) == (1 if want == group else group)


def _captured_kernel_launches(fn):
    """The kernel launches of ONE call of ``fn``, counted from a CUDA-graph CAPTURE of it.  CUPTI is dead on the cc 10.7
    board (``CUPTI_ERROR_INVALID_DEVICE``: ``torch.profiler`` records nothing, not even a torch matmul), so the SM100 file's
    profiler count cannot run here.  The plan launches on torch's current stream (``sdpa/_plan.py``: no handle stream -> the
    caller's current stream, precisely so a capture is not left empty), which inside ``torch.cuda.graph`` IS the capture
    stream: every launch of the execute becomes one kernel node and nothing runs (the outputs stay poisoned -- the caller
    checks that).  Returns ``(kernel nodes, all nodes, names)``; a name is None when the driver cannot resolve it (the count,
    not the names, is the pin)."""
    from cuda.bindings import driver
    from cuda.bindings import runtime as cudart

    cg = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(cg, stream=torch.cuda.Stream()):
        fn()
    graph = cudart.cudaGraph_t(cg.raw_cuda_graph())
    err, _nodes, n = cudart.cudaGraphGetNodes(graph, 0)
    assert err == cudart.cudaError_t.cudaSuccess, err
    err, nodes, n = cudart.cudaGraphGetNodes(graph, n)
    assert err == cudart.cudaError_t.cudaSuccess, err
    kernels, names = 0, []
    for node in nodes:
        err, kind = cudart.cudaGraphNodeGetType(node)
        assert err == cudart.cudaError_t.cudaSuccess, err
        name = None
        if kind == cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            kernels += 1
            try:
                err, params = driver.cuGraphKernelNodeGetParams(driver.CUgraphNode(int(node)))
                func = params.func
                if int(func) == 0 and int(getattr(params, "kern", 0)) != 0:
                    _err, func = driver.cuKernelGetFunction(params.kern)
                _err, raw = driver.cuFuncGetName(func)
                name = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
            except Exception:  # noqa: BLE001 -- naming is diagnostic; the count is the pin
                name = None
        names.append(name)
    return kernels, len(nodes), names


def _dq_arm(monkeypatch, single, *, hq, hkv, sq, skv, chunks=False, **kw):
    """One arm of the single-launch pin on THIS row: ``api_dsl.DQ_SINGLE_LAUNCH = single`` (read at compile() time), the row
    built + pinned + executed through ``_run`` into NaN-poisoned outputs, and what the plan did: the stage-3 records (spied off
    ``load_template``), the row's adapter (captured off its inherited ``compile``), the host trace's ``_dq_launches`` calls (the
    compiled-plan cache is switched off for the arm so ``compile_host`` traces ``prepared_host.host`` in-process -- a cache HIT
    would skip the trace the spy watches; the chunk loop is one ``scf.for`` body, so the spy sees the arithmetic once per trace,
    not once per launch) and the RUNTIME kernel launches of one execute (``_captured_kernel_launches``).  ``chunks`` forces the
    head chunk down to the GQA group, so the chain runs ``hq // group`` head chunks (``head_base > 0`` on every launch form)."""
    from contextlib import nullcontext

    from cudnn.sdpa.bwd import api_dsl
    from cudnn.sdpa.bwd.kernels.sm100 import prepared_host

    monkeypatch.setattr(api_dsl, "DQ_SINGLE_LAUNCH", single)
    monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "1")
    # This probe observes host tracing, so bypass both disk reuse and the in-process memo.
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    records, apis, dq_calls = {}, [], []
    original_load = api_dsl.load_template

    def load_spy(path, params, tag="template"):
        if tag in _MM_TAGS:
            records[tag] = params
        return original_load(path, params, tag)

    original_compile = api_dsl.SdpaBwdDslSm100.compile

    def compile_spy(adapter):
        apis.append(adapter)
        return original_compile(adapter)

    original_dq = prepared_host._dq_launches

    def dq_spy(group, bhg):
        n = original_dq(group, bhg)
        dq_calls.append((group, bhg, n))
        return n

    monkeypatch.setattr(api_dsl, "load_template", load_spy)
    monkeypatch.setattr(api_dsl.SdpaBwdDslSm100, "compile", compile_spy)
    monkeypatch.setattr(prepared_host, "_dq_launches", dq_spy)
    guard = patch.object(api_dsl, "_sm100_head_chunk", side_effect=lambda *a, group=1, **k: group) if chunks else nullcontext()
    with guard:
        run = _run(b=1, hq=hq, hkv=hkv, sq=sq, skv=skv, poison=float("nan"), **kw)
    assert len(apis) == 1 and type(apis[0]).__name__ == "SdpaBwdDslSm107D512", [type(a).__name__ for a in apis]
    api = apis[0]
    for x in run.outputs:
        x.fill_(float("nan"))
    kernels, nodes, names = _captured_kernel_launches(run.execute)
    torch.cuda.synchronize()
    assert all(torch.isnan(x.float()).all().item() for x in run.outputs), "the capture must not execute: the count is of the launches, not a run"
    facts = dict(
        lo_bhg=records["sdpa_bwd_sm100_mm_lo"].b_head_group,
        hi_bhg=records["sdpa_bwd_sm100_mm_hi"].b_head_group,
        dq_bhg=api._dq_b_head_group,
        chunk=api._qh_chunk,
        chunks=hq // api._qh_chunk,
        dq_calls=list(dq_calls),  # a snapshot: the next arm's spy wraps this one and would append to the same list
        kernel_launches=kernels,
        graph_nodes=nodes,
        gemm_launches_by_name=sum(1 for n in names if n and _STAGE3_GEMM_KERNEL in n) if names and all(names) else None,
        kernel_names=names,
    )
    return run, facts


@requires_rubin
@pytest.mark.parametrize(
    "case",
    [
        dict(hq=16, hkv=2, sq=1024, skv=1024),
        dict(hq=16, hkv=2, sq=1024, skv=1024, use_causal_mask=True, chunks=True),
        dict(hq=128, hkv=8, sq=2048, skv=2048),
        dict(hq=8, hkv=8, sq=512, skv=512),
    ],
    ids=["gqa16-2_1k", "gqa16-2_1k_causal_chunked", "gqa128-8_2k", "mha8"],
)
def test_stage3_dq_single_launch_per_chunk_is_bitwise_the_per_member_launches(monkeypatch, case, watchdog):
    """Under GQA this row's shipped dQ GEMM is ONE launch per head chunk: the (512, 512) dQ rendering indexes B = K by
    ``h // group`` (``MatmulTemplateParams.b_head_group = group``) over the whole dS and dQ chunk, where the chain used to run
    one launch per group MEMBER over every ``group``-th Q head (at H_q / H_kv = 16 and S = 8K, 16 launches of 16 clusters on
    the 208-SM board per chunk).  Both forms pair every Q head with the same K head and walk the same k tiles per output tile
    into an fp32 accumulator, so dQ must be the SAME BITS -- and dK / dV, which the change never touches.
    ``api_dsl.DQ_SINGLE_LAUNCH = False`` is the twin (``b_head_group = 1``, the per-member loop); both arms are held to the
    fp64 oracle with NaN-poisoned outputs.  The LAUNCH COUNT is pinned twice, from a CUDA-graph capture of one execute (the
    runtime count; CUPTI is dead on the board) -- dot + stage 2 per chunk + the stage-3 GEMMs per chunk (+ the causal
    zero-fill, + the GQA dK / dV fold): ``3 * chunks`` GEMMs on the shipped arm, ``(2 + group) * chunks`` on the twin -- and
    from the host trace's ``_dq_launches`` call (1 vs ``group`` launches per chunk).  MHA renders, traces and launches
    identically either way (``b_head_group`` stays 1).  The b_head_group arm of the (512, 512) rendering is new on sm_107a:
    this is its board evidence (the SM100 file's pin never runs here)."""
    case = dict(case)
    chunks = case.pop("chunks", False)
    hq, hkv, sq, skv = (case.pop(k) for k in ("hq", "hkv", "sq", "skv"))
    group = hq // hkv
    causal = bool(case.get("use_causal_mask"))
    keep = _causal_keep(sq, skv) if causal else None
    single, f_single = _dq_arm(monkeypatch, True, hq=hq, hkv=hkv, sq=sq, skv=skv, chunks=chunks, keep=keep, **case)
    members, f_members = _dq_arm(monkeypatch, False, hq=hq, hkv=hkv, sq=sq, skv=skv, chunks=chunks, keep=keep, **case)
    n_chunks = f_single["chunks"]
    assert f_single["chunk"] % group == 0 and (not chunks or f_single["chunk"] == group) and f_members["chunks"] == n_chunks, (f_single, f_members)
    # the records and the adapter's copy: dQ takes the group on the shipped arm and 1 on the twin; dV / dK (B per Q head) stay 1
    assert (f_single["lo_bhg"], f_single["hi_bhg"], f_single["dq_bhg"]) == (1, group, group), f_single
    assert (f_members["lo_bhg"], f_members["hi_bhg"], f_members["dq_bhg"]) == (1, 1, 1), f_members
    # the host trace: `_dq_launches(group, b_head_group)` exactly once per trace -> 1 dQ launch per chunk on the shipped arm, `group` on the twin
    assert f_single["dq_calls"] == [(group, group, 1)], f_single
    assert f_members["dq_calls"] == [(group, 1, group)], f_members
    # the runtime launches of one execute: every node a kernel; the chain's fixed launches + the stage-3 GEMMs per chunk
    base = 1 + n_chunks + (1 if causal else 0) + (1 if group > 1 else 0)  # dot, stage 2 per chunk, causal zero-fill, GQA dK/dV fold
    assert f_single["kernel_launches"] == f_single["graph_nodes"] == base + 3 * n_chunks, (
        f"single-launch arm: {f_single['kernel_launches']} kernel launches, expected {base} + 3 * {n_chunks} chunks = {base + 3 * n_chunks} "
        f"(the per-member loop launches {base} + (2 + {group}) * {n_chunks} = {base + (2 + group) * n_chunks}); facts {f_single}"
    )
    assert (
        f_members["kernel_launches"] == f_members["graph_nodes"] == base + (2 + group) * n_chunks
    ), f"per-member arm: {f_members['kernel_launches']} kernel launches, expected {base} + (2 + {group}) * {n_chunks} = {base + (2 + group) * n_chunks}; facts {f_members}"
    if f_single["gemm_launches_by_name"] is not None:
        assert (f_single["gemm_launches_by_name"], f_members["gemm_launches_by_name"]) == (3 * n_chunks, (2 + group) * n_chunks), (f_single, f_members)
    print(
        f"\n{_ENGINE} dQ launches hq={hq} hkv={hkv} s={sq} chunks={n_chunks}: single {f_single['kernel_launches']} kernels, per-member {f_members['kernel_launches']}; "
        f"GEMMs by name {f_single['gemm_launches_by_name']} / {f_members['gemm_launches_by_name']}; kernels {f_single['kernel_names']}"
    )
    # the oracle on both arms, then the bits
    single.check()
    members.check()
    for name, x, y in zip(("dQ", "dK", "dV"), single.outs[0], members.outs[0]):
        xi, yi = x.contiguous().view(torch.int16), y.contiguous().view(torch.int16)
        n_diff = (xi != yi).sum().item()
        assert n_diff == 0, (
            f"{name}: the single dQ launch vs the per-member launches differ in {n_diff} of {xi.numel()} int16 words "
            f"(max|diff|={(x.float() - y.float()).abs().max().item():.3e})"
        )


# =========================================================================== GPU time-slicing: the chain (stage 2 + the stage-3 GEMMs) under a second context (board, L1)

_CONTENTION_CHILD = r"""
import math, os, sys, time
role, n, budget_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
import torch
import cudnn
import cudnn.sdpa  # noqa: F401
b, hq, s, d = 1, 128, 8192, 512
torch.manual_seed(0)
def bshd(fill=True):
    t = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16) if fill else torch.zeros(b, s, hq, d, device="cuda", dtype=torch.bfloat16)
    return (t.mul_(0.1) if fill else t).permute(0, 2, 1, 3)
q, k, v, o, do = bshd(), bshd(), bshd(), bshd(), bshd()
dq, dk, dv = bshd(False), bshd(False), bshd(False)
stats = torch.full((b, hq, s, 1), math.log(s), device="cuda", dtype=torch.float32)
g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
sh, st_ = [b, hq, s, d], [s * hq * d, d, hq * d, 1]
t = {name: g.tensor(name=name, dim=sh, stride=st_) for name in ("q", "k", "v", "o", "do")}
t["stats"] = g.tensor(name="stats", dim=[b, hq, s, 1], stride=[hq * s, s, 1, 1], data_type=cudnn.data_type.FLOAT)
causal = role == "twin"
tdq, tdk, tdv = g.sdpa_backward(name="bwd", q=t["q"], k=t["k"], v=t["v"], o=t["o"], dO=t["do"], stats=t["stats"], attn_scale=1.0 / math.sqrt(d), use_causal_mask=causal)
for out in (tdq, tdk, tdv):
    out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_stride(st_)
g.validate(); g.build_operation_graph(); g.create_execution_plans([cudnn.heur_mode.A])
idx = next(i for i in range(g.get_execution_plan_count()) if "sdpa_bwd_sm107_d512" in g.get_plan_name_at_index(i))
g.select_plan(idx); g.check_support(); g.build_plans()
ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
feed = {t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: stats, tdq: dq, tdk: dk, tdv: dv}
hist = []  # per-launch seconds: a starved launch (seconds, then the wall) reads differently from a wedged one (ms, ms, never)
for i in range(n):
    t0 = time.time()
    g.execute(feed, ws)
    ev = torch.cuda.Event(); ev.record()
    while not ev.query():
        time.sleep(0.02)
        if time.time() - t0 > budget_s:
            import subprocess
            try:
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
            except Exception as e:  # a missing or stuck nvidia-smi must not turn the 45 s hang exit into a 40 min one
                apps = f"<nvidia-smi unavailable: {e!r}>"
            print(f"[{role}] HANG: launch {i + 1} exceeded {budget_s:.0f} s; history (s): " + " ".join(f"{h:.2f}" for h in hist[-30:]), flush=True)
            print(f"[{role}] other compute processes on the device at the hang: {apps.strip().splitlines()}", flush=True)
            os._exit(3)
    hist.append(time.time() - t0)
    if i == 0:
        print(f"[{role}] ready", flush=True)
torch.cuda.synchronize()
print(f"[{role}] done {n} launches", flush=True)
"""


@requires_rubin
@pytest.mark.L1
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_chain_survives_gpu_time_slicing(tmp_path):
    """The 2x2 lessons' detector on this line: a second CUDA context running the same chain (dense, B=1 H=128 S=8192) makes the
    GPU time-slice; the row's chain (causal: stage 2 with its polled cross-pair ring barriers AND the stage-3 (512, 512) GEMMs
    with their hinted ``try_wait`` cross-pair waits -- the form that hung stage 2 before the poll) must complete 100 launches,
    each returning inside 45 s.  A hang exits the child with 3 (the stuck context dies with it), so the suite never wedges."""
    import os
    import time

    script = tmp_path / "contention_child.py"
    script.write_text(_CONTENTION_CHILD)
    env = dict(os.environ, CUDNN_FRONTEND_ENABLE_FROST_ENGINES="1")
    load_log = tmp_path / "load.log"
    with open(load_log, "w") as f:
        load = subprocess.Popen([sys.executable, str(script), "load", "100000", "600"], stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        t0 = time.time()
        while "[load] ready" not in load_log.read_text():
            if load.poll() is not None:
                pytest.fail(f"the load child exited early:\n{load_log.read_text()[-3000:]}")
            if time.time() - t0 > 900:
                pytest.fail(f"the load child did not start launching within 900 s:\n{load_log.read_text()[-3000:]}")
            time.sleep(1.0)
        twin = subprocess.run([sys.executable, str(script), "twin", "100", "45"], capture_output=True, text=True, timeout=2400, env=env)
    finally:
        load.kill()
        load.wait()
    (tmp_path / "twin.log").write_text(twin.stdout + "\n--- stderr ---\n" + twin.stderr)
    assert twin.returncode == 0 and "[twin] done 100 launches" in twin.stdout, f"rc={twin.returncode}\n{twin.stdout[-3000:]}\n{twin.stderr[-3000:]}"


@requires_rubin
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_chain_waits_for_delayed_empty_observer(tmp_path):
    """The eight-stage ring also orders a passive observer before slot reuse across KV tiles."""
    from frost_test_utils import run_d512_delayed_observer

    run_d512_delayed_observer(tmp_path, _ENGINE)
