# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM107 (Rubin) routing of the f16/bf16 forward SDPA kernels.

The adapter routes cc10.7 graphs to the SM107 sibling modules, which bake the
Rubin geometry the Blackwell modules cannot express — notably the **version-1
tcgen05 SMEM descriptor** every operand tile needs once the per-CTA SMEM budget
passes 256 KiB (a version-0 descriptor's ``start_address`` is 14 bits, so a
buffer at 256 KiB wraps to offset 0 and the MMA silently multiplies whatever
lives at the bottom of SMEM).

Routing tests run before compilation on any device. Native split-KV execution
tests require SM107; other end-to-end numerics ride the shared SM10x suites.
"""

import dataclasses
import re
import subprocess
import textwrap
import sys
import shutil
import os

import pytest

from frost_test_utils import requires_dsl

from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
from cudnn.sdpa.fwd.config_sm100 import TemplateParams

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.fixture(autouse=True)
def _sm107_target_for_metadata(monkeypatch):
    """Model SM107 eligibility on other GPUs without requiring their DSL to target Rubin."""
    import torch
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


_E4M3, _BF16_OUT = 0, 2
# (192, 128) is in the sweep for all three dtype families as of 2026-09-09 --
# it is the flavor whose wider K moves SMEM offsets, so it is exactly the one
# a DESC_VERSION check must cover, not skip.
_FLAVORS = [(128, 128), (192, 128), (256, 256), (512, 512)]


def _load(flavor, rubin, *, fp8=False, pertensor=False, **params):
    return _load_sm100_kernel_module(flavor, TemplateParams(**params), fp8=fp8, pertensor=pertensor, rubin=rubin)


def _code_lines(src):
    """Source with whole-line comments dropped -- the prose in these modules
    quotes ``desc_version=1`` when explaining the hazard."""
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_f16_routes_to_the_sm107_sibling(flavor):
    """Every f16 flavor has a Rubin sibling, and Blackwell keeps the SM100 one."""
    assert "sm107" in _load(flavor, rubin=True).__name__
    assert "sm100" in _load(flavor, rubin=False).__name__


# Which flavors' MMA-operand SMEM crosses 256 KiB -- the boundary above which a
# version-0 tcgen05 descriptor cannot address the buffer.  d512 alone does: its
# Q(u)O and K(u)V slabs are 128 KiB each, putting the P transfer ring at exactly
# 262144 (and, on the MXFP8 sibling, the scale-factor tiles near 292 KiB).  Keep
# d128/d256 on version 0 so they stay byte-identical to the shipped
# sm107/prefill_d128_fp8.py sibling.
_NEEDS_DESC_V1 = {(512, 512)}

# (kind, loader kwargs) for every SM107 dtype family, so the check below covers
# the QUANTIZED kernels too -- the d512 MXFP8 sibling is where a missing
# version-1 descriptor actually shipped (LSE = +inf, O = NaN on 100% of cells).
# dtype_qkv=_E4M3 is load-bearing on the quantized arms: the FP8/MXFP8 SM107
# bodies pin idesc k_dim=1 and raise unless TILE_K_HW=64, which only the FP8
# dtypes select (config_sm107.tile_k_hw).
_DTYPE_FAMILIES = [
    ("f16", {}),
    ("fp8", {"fp8": True, "pertensor": True, "dtype_qkv": _E4M3, "dtype_o": _BF16_OUT}),
    ("mxfp8", {"fp8": True, "pertensor": False, "dtype_qkv": _E4M3, "dtype_o": _BF16_OUT}),
]


def test_sm107_every_prefill_kernel_takes_a_guarded_running_max_step():
    """Source pin (no GPU) over EVERY cc 10.7 prefill kernel: the online-softmax running-max step is either the shared
    finite-sentinel helper ``running_max_step_finite_sentinel`` (the kernels that mask with the finite sentinel; it keeps a
    KV tile that is fully masked AHEAD of the row's first live key out of the running state -- total_max kept, alpha = 1, a
    shift of 0 so P = 0 -- where the inlined chain took the sentinel as the running max: scaled by scale_log2 > 1 it
    overflowed to -inf and the shift read -inf - (-inf) = NaN into P, below that P = 1 per masked column wiped only by
    alpha = 0 at the next live tile) or ``row_max_for_exp2`` (the -inf-masked 2x2 twin).  Every helper call passes the RAW
    tile max first and folds its guard on the module's mask flags, and no inlined ``is_first`` / ``exp_input`` chain remains
    beside it.  A kernel that matches neither is the unguarded form this pin exists to refuse."""
    import glob
    import os

    from cudnn.sdpa.fwd.kernels import sm107 as pkg

    files = sorted(glob.glob(os.path.join(os.path.dirname(pkg.__file__), "prefill_*.py")))
    assert len(files) >= 13, files
    shared_call = re.compile(
        r"running_max_step_finite_sentinel\(\s*(\w+),\s*current_max,\s*total_max,\s*(NEG_INF(?:_F32)?),\s*RESCALE_THRESHOLD(?:_F32)?,\s*masked=CFG\.MASK_FLAGS != MASK_NONE\s*\)"
    )
    seen_shared = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            code = _code_lines(fh.read())
        name = os.path.basename(f)
        if "row_max_for_exp2(" in code:
            assert "running_max_step_finite_sentinel" not in code, f"{name}: both running-max forms in one body"
            continue
        calls = shared_call.findall(code)
        assert (
            calls
        ), f"{name}: the running-max step is neither running_max_step_finite_sentinel (RAW max first, masked=CFG.MASK_FLAGS != MASK_NONE) nor row_max_for_exp2"
        assert code.count("running_max_step_finite_sentinel(") == len(calls), f"{name}: a helper call with a different argument shape"
        assert {c[0] for c in calls} & {"raw_max", "current_max_unscaled", "current_max_raw"} == {
            c[0] for c in calls
        }, f"{name}: the first operand must be the RAW tile max: {calls}"
        for frag in ("is_first = total_max ==", "update_cond = is_first", "exp_input = ", "new_total_max = total_max"):
            assert frag not in code, f"{name}: the inlined running-max chain {frag!r} remains beside the shared helper"
        seen_shared += 1
    assert (
        seen_shared == 12
    ), f"the twelve finite-sentinel prefill bodies (d128 / d192x128 / d256 / d512 x f16 / fp8 / mxfp8) take the helper; saw {seen_shared}"


@pytest.mark.parametrize("kind,load_kw", _DTYPE_FAMILIES, ids=[k for k, _ in _DTYPE_FAMILIES])
@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_descriptor_version_matches_the_smem_budget(flavor, kind, load_kw):
    """A version-0 descriptor's ``start_address`` is 14 bits = a 256 KiB
    window; Rubin raises the per-CTA cap to 327 KiB.  An operand buffer at or
    above 256 KiB therefore wraps to offset 0 and the MMA multiplies whatever
    is at the bottom of SMEM -- O comes out EXACTLY zero (or, when the wrapped
    tile is a scale factor, LSE = +inf and O = NaN), no crash, both operands
    provably correct in SMEM, every race probe clean.

    Asserted against the module's own ``DESC_VERSION`` constant, which every
    ``SmemTile`` in that module is constructed with -- so this pins the value
    the kernel actually uses, not the presence of a substring that a comment
    could also supply.

    This is the counter-assertion for the flavors below the boundary: if a tile
    config ever pushes one of them over 256 KiB, move it into _NEEDS_DESC_V1
    and flip its DESC_VERSION -- do not delete the check."""
    mod = _load(flavor, rubin=True, **load_kw)
    want = 1 if flavor in _NEEDS_DESC_V1 else 0
    assert mod.DESC_VERSION == want, f"{mod.__name__}: DESC_VERSION={mod.DESC_VERSION}, expected {want}"


@pytest.mark.parametrize("kind,load_kw", _DTYPE_FAMILIES, ids=[k for k, _ in _DTYPE_FAMILIES])
@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_every_smem_tile_takes_the_module_desc_version(flavor, kind, load_kw):
    """DESC_VERSION is only meaningful if EVERY ``SmemTile`` is built with it.
    The d512 MXFP8 kernel shipped NaN on 100% of cells because its four
    scale-factor tiles were declared under a comment claiming "every operand
    tile here carries desc_version=1" -- without the kwarg.  One re-literalled
    call site is all it takes, so count them."""
    import re

    mod = _load(flavor, rubin=True, **load_kw)
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    n_tiles = len(re.findall(r"\bSmemTile\($", code, re.M))
    assert n_tiles > 0, mod.__name__
    n_wired = code.count("desc_version=DESC_VERSION")
    assert n_wired == n_tiles, f"{mod.__name__}: {n_tiles} SmemTile(s) but {n_wired} wired to DESC_VERSION"
    assert not re.search(r"desc_version=[01]\b", code), f"{mod.__name__}: a re-literalled desc_version bypasses DESC_VERSION"


# ------------------------------------------------------------------ ring-wait retry form: ONE module constant per kernel
# ``tile_dsl.barrier.wait(spin=True)`` is the hint-less uniform spin (USYNCS.PHASECHK / BRA.U on sm_107a) in place of the DSL's
# time_limit form (per-lane SYNCS.PHASECHK + NANOSLEEP.SYNCS).  Whether the per-KV-iteration RING waits of a kernel take it is a
# MEASURED per-kernel fact, not a shape-derived one -- the same form is +4.9 % on d128 bf16 and -5.8 % on d128 per-tensor fp8 --
# so, like DESC_VERSION, each sm107 prefill kernel spells the decision ONCE as ``SPIN_RING_WAITS`` and every ring site passes
# ``spin=SPIN_RING_WAITS``; the waits a warp parks in for a whole tile (the scheduler payload wait of every role, mb_tmem_dealloc,
# the TMA-STG's mb_o_full / mb_tma_o_full / mb_tmastg_go) and the end-of-kernel drains keep the default.  The site counts below are
# the RING / IDLE halves of the wait-form study's 521-site classification (370 ring + 151 idle, of which 4 sit in
# tile_dsl/scheduler.py and stay on the default); pinning them is what keeps a re-classified, re-literalled or newly added site from
# drifting in silently.  Flipping a kernel's constant is the whole experiment; the SASS twin of this check is
# test_sm107_ring_wait_form_sass_pins.
_SPIN_RING_WAITS = {  # (kind, flavor): (SPIN_RING_WAITS, ring wait sites, idle wait sites)
    ("f16", (128, 128)): (True, 32, 11),  # +1 ring wait: the FP32 split-partial epilogue's mb_o_empty wait
    ("fp8", (128, 128)): (False, 44, 12),  # +1 ring wait: the block-scaled O epilogue's first mb_o_empty wait (sf_o)
    ("mxfp8", (128, 128)): (True, 36, 13),  # +1 ring wait: the block-scaled O epilogue's first mb_o_empty wait (sf_o)
    ("f16", (192, 128)): (True, 32, 11),  # +1 ring wait: the FP32 split-partial epilogue's mb_o_empty wait
    ("fp8", (192, 128)): (True, 43, 12),  # +1 ring wait: the FP32 split-partial epilogue's mb_o_empty wait
    ("mxfp8", (192, 128)): (True, 35, 13),
    ("f16", (256, 256)): (False, 29, 11),
    ("fp8", (256, 256)): (False, 29, 11),
    ("mxfp8", (256, 256)): (False, 29, 11),
    ("f16", (512, 512)): (True, 23, 14),  # +1 ring wait: the in-loop mb_bmm2_done wait is spelled once per CORR_READY_BEFORE_DONE arm (ONE traced site)
    ("fp8", (512, 512)): (True, 22, 14),
    ("mxfp8", (512, 512)): (False, 22, 14),
    # The 2x2-datapath sibling (sm107/prefill_d512_f16_2x2.py, TemplateParams.mma_2x2): 16 LEVER ring sites = q_empty (TMA-LDG) +
    # bmm2_ready + empty_mainloop + q_full + p_full 2 (MMA) + bmm1_done + stat_empty 2 (softmax) + stat_full 3 + bmm2_done 4 (three
    # in-loop spellings, one per hand-off arm, + the final) (correction); 12 idle = 5 scheduler payload waits + 2 tmem_dealloc +
    # 4 mb_o_full (TMA-STG arms) + the q_empty drain.  The 10 CROSS-PAIR sites (k/v_empty per iteration + their drains, k_full 2 +
    # v_full, the three o_empty waits) are POLLED through _poll_wait (_D512_2X2_CROSS_PAIR_WAITS below), never waited.
    ("f16_2x2", (512, 512)): (True, 16, 12),
}
# Cross-pair-released barriers of the 2x2 module (mirrors the fix-lane poll-wait rule, bprop fix @ bd6bed12c): a barrier whose
# completing event is issued from OUTSIDE the pair -- under KV_SHARE=2 k/v_empty (both pair leaders' tcgen05.commit multicast 0xF)
# and k/v_full (the twin's TMA complete_tx), under the pair-wide O u V gate o_empty (the twin's remote arrive) -- is POLLED with the
# non-blocking mbarrier.test_wait.parity loop (the module's _poll_wait), never parked: under GPU time-slicing tile_dsl wait() hung
# at 2/300 and the hint-less spin at 74/200 on such a barrier, the poll ran 200/200 + 300/300 (d512 bwd lane, 2026-10-01).
# POLL_CROSS_PAIR_WAITS is a correctness constant pinned True by the module itself, never the SPIN_RING_WAITS perf lever:
# (poll-site count, targets).
_D512_2X2_CROSS_PAIR_WAITS = (10, {"mb_k_empty", "mb_v_empty", "mb_k_full", "mb_v_full", "mb_o_empty"})
_IDLE_WAIT_TARGETS = ("mb_tmem_dealloc", "mb_o_full", "mb_tma_o_full", "mb_tmastg_go")


def _wait_sites(code):
    """Every mbarrier wait call site of a kernel source: (target, balanced argument text).  Line-anchored like the study's
    classifier (``<bars.>mb_X[...].wait(`` or the bare ``wait(sched.mb_...`` payload wait), so docstring prose that spells
    ``wait(...)`` is not a site; the argument text is paren-matched because black wraps some calls over several lines."""
    import re

    out = []
    for m in re.finditer(r"^\s*(?:(?:bars\.)?(mb_\w+)(?:\[[^\]]*\])*\.wait\(|wait\((sched\.mb_\w+))", code, re.M):
        target = m.group(1) or m.group(2)
        i, depth = m.end(), 1
        while depth:
            depth += (code[i] == "(") - (code[i] == ")")
            i += 1
        out.append((target, code[m.end() : i - 1]))
    return out


@pytest.mark.parametrize("kind,load_kw", _DTYPE_FAMILIES, ids=[k for k, _ in _DTYPE_FAMILIES])
@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_ring_waits_take_the_module_spin_constant(flavor, kind, load_kw):
    """SPIN_RING_WAITS holds the measured per-kernel value (the module's own constant, not a source grep), it is defined exactly
    once, no wait site carries a ``spin=True`` / ``spin=False`` literal, every RING site (and only those: the pinned count) passes
    ``spin=SPIN_RING_WAITS``, and no whole-tile idle target does.  A kernel that gains from the spin but regresses to the sleeping
    form at one site, or a loser that leaks the spin onto one ring, changes the count here before it costs 2-6 % on the node."""
    import re

    want, n_ring, n_idle = _SPIN_RING_WAITS[(kind, flavor)]
    mod = _load(flavor, rubin=True, **load_kw)
    assert isinstance(mod.SPIN_RING_WAITS, bool) and mod.SPIN_RING_WAITS is want, f"{mod.__name__}: SPIN_RING_WAITS={mod.SPIN_RING_WAITS}, expected {want}"
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    assert len(re.findall(r"^SPIN_RING_WAITS: bool = (?:True|False)$", code, re.M)) == 1, f"{mod.__name__}: exactly one SPIN_RING_WAITS definition"
    assert not re.search(r"spin=(?:True|False)\b", code), f"{mod.__name__}: a spin= literal at a call site bypasses SPIN_RING_WAITS"
    sites = _wait_sites(code)
    spun = [t for t, args in sites if "spin=SPIN_RING_WAITS" in args]
    assert len(spun) == n_ring, f"{mod.__name__}: {len(spun)} ring waits pass spin=SPIN_RING_WAITS, the classification says {n_ring}"
    assert len(sites) == n_ring + n_idle, f"{mod.__name__}: {len(sites)} wait sites, expected {n_ring} ring + {n_idle} idle"
    leaked = [t for t in spun if t.startswith(_IDLE_WAIT_TARGETS) or t.startswith("sched.")]
    assert not leaked, f"{mod.__name__}: whole-tile idle waits must keep the sleeping form: {leaked}"
    assert code.count("spin=") == n_ring, f"{mod.__name__}: a spin= outside a .wait( call"


# The per-cell softmax mask is ONE tile_dsl op, `tile_dsl.mask.apply_mask_chunk`: a keep-word per 32 columns from two
# saturating shifts, then a register-to-predicate `R2P` + one `FSEL` per cell (1.4-1.6 instructions per cell, independent
# of the number of active terms).  It replaced a per-cell compare + select (3-7 instructions per cell, 51-72 % of a masked
# softmax tile's instructions serialized ahead of the exp burst; sm_107a listings, 2026-09-22: masked body -208 (1 term)
# / -466 (2 terms) / -903 (the mxfp8 d512 SWA build, whose 128 live i1 values had spilled into GPR bits through
# predicate-to-register moves and LOP3) instructions per KV tile per lane) -- first behind a per-kernel `MASK_FORM`
# constant (#1192 / #1197), then collapsed into the op itself.  Same masked set, same sentinel -> O / LSE bitwise
# identical.  What is left to pin: every masked site calls the op DIRECTLY, and no per-kernel form vocabulary comes back.


@pytest.mark.parametrize("kind,load_kw", _DTYPE_FAMILIES, ids=[k for k, _ in _DTYPE_FAMILIES])
@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_every_mask_site_calls_apply_mask_chunk(flavor, kind, load_kw):
    """Every masked call site of every sm107 prefill kernel is a direct `apply_mask_chunk(` call -- no dispatcher, no
    `form=` kwarg, no module `MASK_FORM` constant (the vocabulary the collapse removed).  A reintroduced per-kernel
    selector would let one arm (the d256 and mxfp8 kernels have 2-4 masked arms each) drift to a slower lowering with
    bitwise-identical output, which no numerics test sees; the lowering itself is held by
    test_sm107_masked_softmax_sass_is_register_to_predicate."""
    import re

    mod = _load(flavor, rubin=True, **load_kw)
    assert not hasattr(mod, "MASK_FORM"), f"{mod.__name__}: a MASK_FORM constant is back"
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    n_sites = len(re.findall(r"\bapply_mask_chunk\(", code))
    assert n_sites > 0, f"{mod.__name__}: no masked call site found"
    for spelling in (r"\bapply_mask_chunk_form\b", r"\bapply_mask_chunk_bits\b", r"\bMASK_FORM", r"(?<!\w)form="):
        assert not re.search(spelling, code), f"{mod.__name__}: {spelling!r} -- the per-kernel mask-form selector was collapsed into apply_mask_chunk"


# ------------------------------------------------------------------ the O store / epilogue levers of the d512 kernels: module constants
# The sm107 d512 kernels (the cga4x1 role-split, one TMA-STG warp per CTA) carry two measured perf levers, each spelled ONCE as a
# module constant next to CFG and folded at every site with ``cutlass.const_expr`` -- the SPIN_RING_WAITS discipline:
#   O_STORE_STREAM  -- the sg1 TMA-STG warp issues chunk c's TMA-O subtile (``tile_dsl.tma.tma_store_subtile``) right behind its
#                      ``mb_tma_o_full[c]`` wait instead of queueing the whole tile after the last chunk.  The store's cost is
#                      the SM's in-order TMA ENGINE occupancy ahead of the next work item's V loads, not HBM bytes: +16.3 / +16.5 /
#                      +5.9 / +0.5 % of time at S_KV = 512 / 1024 / 2048 / 8192 on the d512 mxfp8 kernel (212-SM Rubin, B=1 H_Q=64
#                      H_KV=1 S_Q=16K bf16-O dense, A/B/A x3, controls <= 0.03 %), O / LSE / Amax_O bitwise identical.
#   O_EPI_PIPELINE  -- the sg1 epilogue issues the tcgen05.ld batch of chunk c+1 before chunk c's ALU / STS / fence / arrive (the
#                      fence + arrive pin a tcgen05.ld; the classic order exposes one TMEM latency per chunk) and skips the
#                      per-element dead-row select behind a warp-uniform vote; +1.7 pt @1024, -1.1 pt @8192 on top of the stream.
# Both arms of both levers trace, and both constants False is cubin-md5-identical to develop (f16 / fp8 / mxfp8, sm_107a).  The
# per-block body is ONE shared helper (``_common_blackwell.o_epilogue_convert_store``) and the subtile store ONE library op -- the
# kernels carry no in-file copy of either (twice hand-rolled belongs in the library).  The SASS twin of this check is
# test_sm107_d512_o_store_path_sass_pins.
_O_STORE_LEVERS = ("O_STORE_STREAM", "O_EPI_PIPELINE")
_D512 = (512, 512)


@pytest.mark.parametrize("kind,load_kw", _DTYPE_FAMILIES, ids=[k for k, _ in _DTYPE_FAMILIES])
def test_sm107_d512_o_store_levers_are_module_constants(kind, load_kw):
    """Every sm107 d512 kernel holds both levers as bool module constants (the module's own values, not a source grep), defines
    each exactly once, folds each at its sites with ``const_expr`` (no literal, both arms present), streams the O store through
    ``tile_dsl.tma.tma_store_subtile`` and converts through ``_common_blackwell.o_epilogue_convert_store`` -- with no in-file
    copy of either (the experiment's ``_tma_store_subtile`` / ``_o_epi_convert_store``) and no bare bulk-tensor store op."""
    import re

    mod = _load(_D512, rubin=True, **load_kw)
    for name in _O_STORE_LEVERS:
        val = getattr(mod, name)
        assert isinstance(val, bool) and val is True, f"{mod.__name__}: {name}={val!r}, the shipped value is True"
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    for name in _O_STORE_LEVERS:
        assert len(re.findall(rf"^{name}: bool = (?:True|False)$", code, re.M)) == 1, f"{mod.__name__}: exactly one {name} definition"
        assert f"const_expr({name})" in code, f"{mod.__name__}: {name} is not folded at a site"
    assert "const_expr(not O_STORE_STREAM)" in code, f"{mod.__name__}: the whole-tile store arm is gone -- the lever is no longer an A/B"
    assert re.search(r"\btma_store_tile\(", code) and re.search(r"\btma_store_subtile\(", code), f"{mod.__name__}: both store forms must trace"
    assert "    tma_store_subtile," in code, f"{mod.__name__}: tma_store_subtile must come from cudnn.frost.tile_dsl.tma"
    assert (
        "o_epilogue_convert_store(" in code and "o_epilogue_convert_store" in code.split("def _compute_warp_group")[0]
    ), f"{mod.__name__}: the per-block convert is the shared _common_blackwell helper"
    for spelling in (
        "def _tma_store_subtile",
        "def _o_epi_convert_store",
        "nvvm.cp_async_bulk_tensor_global_shared_cta(",
        "_O_STORE_STREAM",
        "_O_EPI_LD_GROUP",
        "abs_max_tree",
    ):
        assert spelling not in code, f"{mod.__name__}: {spelling!r} -- an in-file copy / experiment knob is back"


# ------------------------------------------------------------------ the d512 2x2-DATAPATH sibling (TemplateParams.mma_2x2)
# sm107/prefill_d512_f16_2x2.py: the SM100 2x2 body (one pipeline per CTA on the cta_group::2 M=128 atom, 64 Q rows per
# CTA, a 4-CTA cluster of twin pairs sharing K/V by multicast) with the Rubin deltas -- 3-deep K/V sub-chunk rings,
# DESC_VERSION=1 derived from the layout (the P ring starts at exactly 262144 B), the 320 KiB usable budget, the
# ld.red.max dense softmax arm, SPIN_RING_WAITS, and the PAIR-WIDE O u V alias gate (mb_o_empty init ONE_WARP x KV_SHARE:
# every twin's TMA-STG warp arrives on its own copy and its twin's, fix-lane FATAL-1).  The loader's rubin arm routes an
# mma_2x2=True record to it; the role-split module (and every pin above) is untouched by the field.  These tests apply the
# SAME structural checks as the role-split rows above to the 2x2 module (same regexes, same classification), keyed by the
# ("f16_2x2", (512, 512)) rows of the tables; the GPU half lives in test_sdpa_fwd_d512_2x2_sm107.py.
_D512_2X2_KW = {"mma_2x2": True}
_D512_2X2_FILE = "sm107/prefill_d512_f16_2x2.py"
_D512_2X2_SMEM_TOTAL = 297880  # sQ 64 KiB | sK 3 x 32 | sV 3 x 32 u sO | sP 2 x 16 | xchg 1.5 KiB | 39 bars + sched | 1008 B pad


def _load_2x2(**params):
    return _load(_D512, rubin=True, **_D512_2X2_KW, **params)


def _source_lines(mod):
    with open(mod.__file__, encoding="utf-8") as fh:
        return _code_lines(fh.read())


def test_sm107_d512_2x2_routes_to_the_rubin_sibling():
    """mma_2x2=True on the Rubin line loads the sm107 2x2 sibling; the role-split module keeps its file, and the SM100
    line keeps its own 2x2 sibling (two records -> two module cache keys, both coexist in one process)."""
    mod = _load_2x2()
    # The loader's tag is "sdpa_fwd_sm107_f16_d512_2x2" (the role-split tag + "_2x2"); the template cache appends a per-record suffix.
    assert mod.__file__.endswith(_D512_2X2_FILE) and "sdpa_fwd_sm107_f16_d512_2x2" in mod.__name__, mod.__name__
    assert _load(_D512, rubin=True).__file__.endswith("sm107/prefill_d512_f16.py")
    assert _load(_D512, rubin=False, **_D512_2X2_KW).__file__.endswith("sm100/prefill_d512_f16_2x2.py")
    assert mod.CFG.CGA_M == 4 and mod.KV_SHARE == 2 and mod.CFG.TILE_M == 64


def test_sm107_d512_2x2_config_pins_and_ledger():
    """The Rubin 2x2 Cfg: the SM100 record with exactly the five arch deltas (3/3 rings, STAGES_KV 3, DESC_VERSION 1,
    the 320 KiB budget line, the pair-wide O-empty gate), every arrival count re-derived, and the validator raising on
    each Rubin fact when it no longer holds."""
    from dataclasses import fields, replace

    from cudnn.sdpa.fwd.config_sm100 import CfgD512X2, d512_2x2_p_ring_start_bytes, d512_2x2_smem_bytes
    from cudnn.sdpa.fwd.config_sm100 import make_cfg_d512 as make_cfg_d512_sm100
    from cudnn.sdpa.fwd.config_sm107 import SMEM_USABLE_BYTES, TCGEN05_V0_ADDR_LIMIT, _validate_cfg_d512_2x2_sm107, make_cfg_d512, make_cfg_d512_2x2

    cfg, tma = make_cfg_d512(TemplateParams(mma_2x2=True))
    assert isinstance(cfg, CfgD512X2) and (cfg, tma) == make_cfg_d512_2x2(TemplateParams(mma_2x2=True))
    assert (cfg.STAGES_K_SUB, cfg.STAGES_V_SUB, cfg.STAGES_KV, cfg.XFER_STAGES, cfg.BMM1_LOOKAHEAD) == (3, 3, 3, 2, 1)
    assert cfg.DESC_VERSION == 1 and d512_2x2_p_ring_start_bytes(cfg) == TCGEN05_V0_ADDR_LIMIT == 262144
    assert cfg.SMEM_CAP_BYTES == SMEM_USABLE_BYTES == 320 * 1024
    smem = d512_2x2_smem_bytes(cfg)
    assert smem["data"] == 65536 + 3 * 32768 + 3 * 32768 + 2 * 16384 and smem["n_bars"] == 39
    assert smem["total"] == _D512_2X2_SMEM_TOTAL <= SMEM_USABLE_BYTES
    assert (cfg.TILE_K_HW_BMM1, cfg.TILE_K_HW_BMM2, cfg.TMEM_COLS) == (16, 16, 512)
    assert (cfg.READ_TILE_ARRIVERS, cfg.KV_EMPTY_ARRIVERS, cfg.O_CHUNK_ARRIVERS, cfg.PAIR_LANES) == (42, 2, 64, 256)
    assert cfg.O_EMPTY_ARRIVERS == cfg.ONE_WARP * cfg.KV_SHARE == 64, "the O u V alias gate is pair-wide: both twins' TMA-STG warps arrive"
    assert (tma.QK_ITERS, tma.VO_ITERS, tma.QK_GRANU_ELEMS) == (8, 8, 64)
    # Exactly the arch deltas vs the SM100 record, nothing else (the pair-wide O u V gate is shared: O_EMPTY_ARRIVERS is 64 on
    # both arch records, so it is NOT a delta).
    sm100, _ = make_cfg_d512_sm100(TemplateParams(mma_2x2=True))
    assert sm100.O_EMPTY_ARRIVERS == cfg.O_EMPTY_ARRIVERS == 64
    deltas = {f.name for f in fields(CfgD512X2) if getattr(sm100, f.name) != getattr(cfg, f.name)}
    assert deltas == {"STAGES_K_SUB", "STAGES_V_SUB", "STAGES_KV", "DESC_VERSION", "SMEM_CAP_BYTES"}, deltas
    # The bring-up arm: one pair, own-bit loads, own-warp gate.
    c2, _ = make_cfg_d512_2x2(TemplateParams(mma_2x2=True), cga_m=2)
    assert (c2.READ_TILE_ARRIVERS, c2.KV_EMPTY_ARRIVERS, c2.KV_SHARE, c2.ROWS_PER_CLUSTER, c2.O_EMPTY_ARRIVERS) == (21, 1, 1, 128, 32)
    for bad, pattern in (
        (dict(DESC_VERSION=0), "DESC_VERSION"),
        (dict(STAGES_K_SUB=2, STAGES_V_SUB=2, STAGES_KV=2), "3-deep"),
        (dict(SMEM_CAP_BYTES=327 * 1024), "usable"),
        (dict(O_EMPTY_ARRIVERS=32), r"ONE_WARP \* KV_SHARE|PAIR-WIDE"),  # the shared geometry check (strict, both archs) raises first
        (dict(TMEM_COLS=576), "512 TMEM"),
    ):
        with pytest.raises(ValueError, match=pattern):
            _validate_cfg_d512_2x2_sm107(replace(cfg, **bad), "test")
    with pytest.raises(ValueError, match="BF16/FP16"):
        make_cfg_d512(TemplateParams(mma_2x2=True, dtype_qkv=0, dtype_o=2))
    with pytest.raises(ValueError, match="qh_per_kh"):
        make_cfg_d512(TemplateParams(mma_2x2=True, pack_gqa=True, qh_per_kh=128))
    # The default arm is untouched by the dispatch (dataclass-equal with and without the field).
    assert make_cfg_d512(TemplateParams()) == make_cfg_d512(TemplateParams(mma_2x2=False))


def test_sm107_d512_2x2_descriptor_version_matches_the_smem_budget():
    """Same assertion as test_sm107_descriptor_version_matches_the_smem_budget, on the 2x2 module: (512, 512) is in
    _NEEDS_DESC_V1 and the module's own constant (what every SmemTile reads) agrees with the Cfg's layout-derived value."""
    mod = _load_2x2()
    want = 1 if _D512 in _NEEDS_DESC_V1 else 0
    assert mod.DESC_VERSION == want == mod.CFG.DESC_VERSION, f"{mod.__name__}: DESC_VERSION={mod.DESC_VERSION}, expected {want}"


def test_sm107_d512_2x2_every_smem_tile_takes_the_module_desc_version():
    """Same count as test_sm107_every_smem_tile_takes_the_module_desc_version: SmemTile( sites == desc_version=DESC_VERSION
    sites (5: sQ, sK ring, sV ring, sO staging, sP ring), no re-literalled version."""
    import re

    mod = _load_2x2()
    code = _source_lines(mod)
    n_tiles = len(re.findall(r"\bSmemTile\($", code, re.M))
    assert n_tiles == 5, f"{mod.__name__}: {n_tiles} SmemTile(s), the 2x2 layout has sQ / sK / sV / sO / sP"
    n_wired = code.count("desc_version=DESC_VERSION")
    assert n_wired == n_tiles, f"{mod.__name__}: {n_tiles} SmemTile(s) but {n_wired} wired to DESC_VERSION"
    assert not re.search(r"desc_version=[01]\b", code), f"{mod.__name__}: a re-literalled desc_version bypasses DESC_VERSION"
    assert not re.search(r"desc_version=CFG\.DESC_VERSION", code), f"{mod.__name__}: the tiles read the module constant, not the Cfg field"


def test_sm107_d512_2x2_ring_waits_take_the_module_spin_constant():
    """Same classification as test_sm107_ring_waits_take_the_module_spin_constant, on the 2x2 module (the
    ("f16_2x2", (512, 512)) row): every per-iteration ring wait passes spin=SPIN_RING_WAITS, the whole-tile idle waits
    (scheduler payload, mb_tmem_dealloc, the TMA-STG's mb_o_full) and the end-of-kernel drains (k/v/q_empty and, under
    the pair-wide gate, the final mb_o_empty phase) keep the sleeping form."""
    import re

    want, n_ring, n_idle = _SPIN_RING_WAITS[("f16_2x2", _D512)]
    n_poll, poll_targets = _D512_2X2_CROSS_PAIR_WAITS
    mod = _load_2x2()
    assert isinstance(mod.SPIN_RING_WAITS, bool) and mod.SPIN_RING_WAITS is want
    assert mod.POLL_CROSS_PAIR_WAITS is True, "cross-pair-released barriers are polled (a correctness constant, not a lever)"
    code = _source_lines(mod)
    assert len(re.findall(r"^SPIN_RING_WAITS: bool = (?:True|False)$", code, re.M)) == 1
    assert len(re.findall(r"^POLL_CROSS_PAIR_WAITS: bool = True$", code, re.M)) == 1, "the cross-pair constant is spelled True exactly once"
    assert not re.search(r"spin=(?:True|False)\b", code)
    sites = _wait_sites(code)  # the .wait( sites: lever ring + idle (the poll sites are _poll_wait( calls, counted below)
    spun = [t for t, args in sites if "spin=SPIN_RING_WAITS" in args]
    assert len(spun) == n_ring, f"{mod.__name__}: {len(spun)} ring waits pass spin=SPIN_RING_WAITS, the classification says {n_ring}"
    assert not (set(spun) & poll_targets), f"{mod.__name__}: a cross-pair-released barrier is waited through the perf lever: {set(spun) & poll_targets}"
    assert len(sites) == n_ring + n_idle, f"{mod.__name__}: {len(sites)} .wait( sites, expected {n_ring} ring + {n_idle} idle"
    leaked = [t for t in spun if t.startswith(_IDLE_WAIT_TARGETS) or t.startswith("sched.")]
    assert not leaked, f"{mod.__name__}: whole-tile idle waits must keep the sleeping form: {leaked}"
    assert code.count("spin=") == n_ring
    # The sleeping sites: the scheduler payload waits, tmem_dealloc, the TMA-STG's mb_o_full and the q_empty drain (same-pair).
    idle = [t for t, args in sites if "spin=" not in args]
    assert sorted(set(idle)) == ["mb_o_full", "mb_q_empty", "mb_tmem_dealloc", "sched.mb_scheduler"], idle
    # The poll sites: every cross-pair-released barrier, each spelled _poll_wait(bars.<mb>[...].smem_ptr, <phase>), and no
    # cross-pair barrier left on a .wait( of any form.
    polls = re.findall(r"^\s*_poll_wait\(bars\.(mb_\w+)", code, re.M)
    assert len(polls) == n_poll and set(polls) == poll_targets, f"{mod.__name__}: poll sites {polls}, expected {n_poll} on {sorted(poll_targets)}"
    assert not (set(t for t, _ in sites) & poll_targets), f"{mod.__name__}: a cross-pair-released barrier still has a .wait( site"
    # The poll body is the SHARED tile_dsl wait_poll in this module's shape (32 / 128, the forwards' shared default; the tight
    # loop measured within 0.15 % of it on the board), not a module-local test_wait loop.
    from cudnn.frost.tile_dsl.barrier import poll_ptx

    assert code.count("def _poll_wait(") == 1 and "wait_poll(mb, phase, tight_iters=POLL_TIGHT_ITERS, sleep_ns=POLL_SLEEP_NS)" in code
    assert "mbarrier.test_wait.parity.acquire" not in code, "no module-local test_wait inline PTX: the shared helper is the one spelling"
    from cudnn.frost.tile_dsl.barrier import POLL_SLEEP_NS, POLL_TIGHT_ITERS

    assert (mod.POLL_TIGHT_ITERS, mod.POLL_SLEEP_NS) == (POLL_TIGHT_ITERS, POLL_SLEEP_NS) == (32, 128), "both forwards ship the shared default shape"
    ptx = poll_ptx(mod.POLL_TIGHT_ITERS, mod.POLL_SLEEP_NS)
    assert "mbarrier.test_wait.parity.acquire.cta" in ptx and "nanosleep.u32 128" in ptx and "try_wait" not in ptx


def test_sm107_d512_2x2_every_mask_site_calls_apply_mask_chunk():
    """The masked softmax arm is ONE direct apply_mask_chunk( call (the 2x2 body has one masked arm over this lane's 64
    columns); the dense arm is the Rubin ld.red.max fused load (tmem_load_max_reduction_tile) -- a HALF-row max under the
    2x2 atom, so the exchange with lane r + 64 follows it unconditionally (the named barrier 8 is spelled once per
    iteration, outside both arms)."""
    import re

    mod = _load_2x2()
    assert not hasattr(mod, "MASK_FORM")
    code = _source_lines(mod)
    assert len(re.findall(r"\bapply_mask_chunk\(", code)) == 1
    for spelling in (r"\bapply_mask_chunk_form\b", r"\bapply_mask_chunk_bits\b", r"\bMASK_FORM", r"(?<!\w)form="):
        assert not re.search(spelling, code), spelling
    # Call sites (the module docstring quotes both helper names, so count the assignments, not the names).
    assert (
        code.count("= tmem_load_max_reduction_tile(") == 1 and "num_elems=SOFTMAX_COLS" in code
    ), "the dense arm is the fused ld.red.max over this lane's half row"
    assert code.count("= row_max_reduction(") == 1, "the masked arm keeps the software half-row max"
    # Both arms feed the SAME exchange: exactly one STS / barrier / LDS triple per iteration, after the arms join.
    assert code.count("sXchgMax.subview(xchg_mine).store(half_max)") == 1
    assert code.count("barrier_cta_sync(barrier_id=8, thread_count=CFG.SOFTMAX_LANES)") == 3  # per-iteration max + two tile-end sum barriers


def test_sm107_d512_2x2_levers_are_module_constants():
    """The three measured levers plus SPIN_RING_WAITS are bool module constants of the 2x2 module, each defined exactly
    once and folded at its sites with const_expr (both arms of the O store trace: tma_store_subtile streamed and
    tma_store_tile whole-tile; both correction hand-off arms; both epilogue forms), through the library ops -- no in-file
    copies or experiment knobs."""
    import re

    mod = _load_2x2()
    for name in _O_STORE_LEVERS + ("CORR_READY_BEFORE_DONE", "SPIN_RING_WAITS"):
        val = getattr(mod, name)
        assert isinstance(val, bool) and val is True, f"{mod.__name__}: {name}={val!r}, the shipped value is True"
    code = _source_lines(mod)
    for name in _O_STORE_LEVERS + ("CORR_READY_BEFORE_DONE",):
        assert len(re.findall(rf"^{name}: bool = (?:True|False)$", code, re.M)) == 1, f"{mod.__name__}: exactly one {name} definition"
        assert re.search(rf"const_expr\({name}\b", code), f"{mod.__name__}: {name} is not folded at a site"
    assert re.search(r"\btma_store_tile\(", code) and re.search(r"\btma_store_subtile\(", code), "both store forms must trace"
    assert "    tma_store_subtile," in code, "tma_store_subtile must come from cudnn.frost.tile_dsl.tma"
    assert "const_expr(O_EPI_PIPELINE and b + 1 < N_O_EPI_BLOCKS)" in code, "the pipelined epilogue issues batch b+1's tcgen05.ld before processing b"
    for spelling in (
        "def _tma_store_subtile",
        "def _o_epi_convert_store",
        "nvvm.cp_async_bulk_tensor_global_shared_cta(",
        "_O_STORE_STREAM",
        "_O_EPI_LD_GROUP",
    ):
        assert spelling not in code, spelling
    assert mod._O_SUBTILES_PER_CHUNK == 1 and mod.TMA_O_ITERS_HOST == 8


# Arrive SITES of the 2x2 module per barrier (the per-phase SUMS are the Cfg constants): the SM100 body's ledger plus the
# pair-wide O-empty gate's second arrive (arrive_on_peer on the twin).
_D512_2X2_ARRIVE_SITE_PINS = {
    "mb_p_full[": 1,  # one per-lane release arrive per softmax iteration
    "mb_stat_full[": 2,  # per-iteration alpha + tile-end stats
    "mb_stat_empty[": 3,  # correction: kv_left consume + per-iteration + tile-end
    "mb_bmm2_ready[": 8,  # correction: 2 (kv_left) + 2 (fast arm) + 2 (slow arm) + 2 (lever-off arm)
    "mb_o_full[": 2,  # epilogue: staged + fp32-partials arms (64 lanes of one half each)
    "mb_o_empty.arrive": 1,  # TMA-STG warp, all lanes, own copy ...
    "mb_o_empty.arrive_on_peer": 1,  # ... + the twin's copy (KV_SHARE=2) = ONE_WARP x KV_SHARE per CTA
    "mb_empty_mainloop.arrive": 1,
    "mb_tmem_dealloc.arrive": 1,
    "mb_tmem_dealloc.arrive_on_peer": 1,
    "mb_q_empty.arrive": 1,
    "mb_bmm1_done[": 1,
    "mb_bmm2_done[": 2,  # per-iteration (last N-block) + empty-tile commit
}


def test_sm107_d512_2x2_source_arrive_sites_match_the_ledger():
    """The arrive-site counts of the 2x2 module against the ledger (a new site on a per-lane barrier changes its init
    count), the pair-wide O-empty gate's two arrives on the twin's rank, the proxy fence on every lane right before the P
    release arrive, and the kernel-end drain of the final mb_o_empty phase."""
    mod = _load_2x2()
    with open(mod.__file__, encoding="utf-8") as fh:
        src = fh.read()
    for key, n in _D512_2X2_ARRIVE_SITE_PINS.items():
        if key.endswith("["):
            got = sum(1 for ln in src.splitlines() if key in ln and "].arrive(" in ln)
        else:
            got = src.count(key + "(")
        assert got == n, f"{key}: {got} arrive sites, ledger says {n}"
    assert "bars.mb_o_empty.arrive_on_peer(cta_id_x ^ cutlass.Int32(2))" in src, "the twin is cluster rank cta_id_x ^ 2"
    # The three mb_o_empty waits (TMA-LDG per tile + the kernel-end drain of the final phase, correction per tile) are POLLED:
    # the barrier is released by the twin's remote arrive (the poll-wait rule), so no .wait( site may remain on it.
    assert src.count("_poll_wait(bars.mb_o_empty.smem_ptr, ") == 3 and "bars.mb_o_empty.wait(" not in src, "mb_o_empty: 3 polled waits, no parked wait"
    lines = src.splitlines()
    idx = [i for i, ln in enumerate(lines) if "mb_p_full[" in ln and "].arrive(" in ln]
    assert len(idx) == 1 and 'fence_proxy("async.shared", space="cta")' in lines[idx[0] - 1]
    assert src.count("make_sdpa_helpers(") == 1 and "kv_shared_cluster=True" in src
    assert 'set_name_prefix("cudnn", remove_cutlass_symbol=True)' in src
    assert "is_exclusive=" not in src, "512 TMEM columns: the 576-col is_exclusive=True allocation is not needed (388 used)"


def test_sm107_d512_2x2_prefolded_scale_arm_is_a_const_expr_elision():
    """The pre-folded softmax scale on the 2x2 module (TemplateParams.softmax_scale_prefolded, graph.sdpa attn_scale_prefolded;
    the twin's own validator flag on the cc 10.0 record check): a PARAMS-derived int defined once and folded with const_expr at
    exactly the two lever sites of _softmax_kv_iter -- the raw row max and the FADD2 shift -- the scaled chain's FMUL / FFMA2
    spellings kept as the default arm, scale_softmax_log2 left in the kernel and host signatures (a dead runtime argument under
    the fold; the GPU oracle launches it with garbage).  The f16x2 exponent arm (softmax_f16) is declined for this half-input
    body by the config backstop and the module guard, and the SM100 sibling keeps declining the fold."""
    mod = _load_2x2()
    assert mod.SCALE_PREFOLDED == 0 and _load_2x2(softmax_scale_prefolded=True).SCALE_PREFOLDED == 1
    code = _source_lines(mod)
    definition = "SCALE_PREFOLDED = int(PARAMS.softmax_scale_prefolded)"
    assert code.count(definition) == 1, "one PARAMS-derived definition"
    body = code.split(definition, 1)[1]
    assert body.count("SCALE_PREFOLDED") == body.count("cutlass.const_expr(SCALE_PREFOLDED)") == 2, "exactly the two folded sites, no runtime read"
    assert body.count("        current_max = current_max_raw\n") == 1 and body.count("current_max = current_max_raw * scale_log2") == 1
    assert body.count("cute.math.exp2(reg_S - total_max_safe, fastmath=True)") == 1
    assert body.count("cute.math.exp2(reg_S * scale_log2 - total_max_safe, fastmath=True)") == 1
    assert "    scale_log2: cutlass.Float32,\n" in code and "    scale_softmax_log2: cutlass.Float32,\n" in code, "the scale stays in the ABI"
    assert code.count("_require(not PARAMS.softmax_f16,") == 1 and "softmax_f16 as _softmax_f16" not in code, "no f16x2 exponent arm on the half body"
    with pytest.raises(ValueError, match="softmax_f16"):
        _load_2x2(softmax_f16=True)
    with pytest.raises(ValueError, match="softmax_scale_prefolded"):
        _load(_D512, rubin=False, **_D512_2X2_KW, softmax_scale_prefolded=True)


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_f16_thd_specialization_matches_the_ported_flavors(flavor):
    """A THD config must TRACE for a ported f16 flavor and be REFUSED for the
    rest -- the config guard is what stops an unported body pairing its 3B+2
    metadata with the shared 4B+4 decode (a wrong-sequence read, not a raise).

    Also pins that the dense specialization stays THD-free everywhere, which is
    what keeps `get_workspace_size() == 0` for a dense stats-less graph."""
    from cudnn.sdpa.fwd.config_sm107 import SM107_F16_THD_SHAPES

    assert _load(flavor, rubin=True, seq_kv_lens_present=True).CFG.THD_VARLEN == 0

    if flavor in SM107_F16_THD_SHAPES:
        assert _load(flavor, rubin=True, seq_kv_lens_present=True, thd_varlen=1).CFG.THD_VARLEN == 1
    else:
        with pytest.raises(ValueError, match="THD/varlen"):
            _load(flavor, rubin=True, seq_kv_lens_present=True, thd_varlen=1)


def test_rubin_f16_never_picks_a_flavor_it_has_no_kernel_for():
    """The adapter must narrow the f16 candidate pool BY ARCH LINE, or a graph
    picks a flavor with no Rubin module and dies with a KeyError inside module
    loading instead of riding the next covering envelope.

    PARTIALLY INVERTED 2026-09-04: d192xd128 now HAS a Rubin sibling
    (``sm107/prefill_d192_d128_f16.py`` -- the same body as d128 with
    ``make_cfg_d192``), so a d=192 f16 graph lowers onto the NATIVE kernel
    instead of riding the d256 envelope.

    FULLY INVERTED 2026-09-09: the quantized lines gained their d192 siblings
    too, so every Rubin pool now covers all four flavors.  The pool-NARROWING
    invariant is what this test really pins and it is unchanged -- it is what
    keeps a graph off a flavor with no Rubin module, and it must survive every
    future arch line that ships a partial flavor set.
    """
    from cudnn.sdpa.fwd.api_dsl import (
        _SM100_FLAVORS,
        _SM107_FP8_KERNEL_FILES,
        _SM107_KERNEL_FILES,
        _pick_flavor,
    )

    rubin_pool = tuple(f for f in _SM100_FLAVORS if f in _SM107_KERNEL_FILES)
    assert _pick_flavor(192, 128, rubin_pool) == (192, 128)
    # Blackwell keeps its native d192xd128 kernel.
    assert _pick_flavor(192, 128, None) == (192, 128)
    # INVERTED: the quantized Rubin lines gained their d192 siblings, so a
    # d=192 FP8 graph lands on the NATIVE kernel instead of an envelope.
    rubin_fp8_pool = tuple(f for f in _SM100_FLAVORS if f in _SM107_FP8_KERNEL_FILES)
    assert (192, 128) in rubin_fp8_pool
    assert _pick_flavor(192, 128, rubin_fp8_pool) == (192, 128)
    # The narrowing itself still has teeth: every Rubin pool is a SUBSET of the
    # Blackwell flavor list, and _pick_flavor is only ever handed the pool whose
    # kernel files exist.  A pool built from a map that lacks a flavor must not
    # offer it -- this is the guard that survives the next partial arch line.
    assert set(rubin_pool) <= set(_SM100_FLAVORS) and set(rubin_fp8_pool) <= set(_SM100_FLAVORS)
    assert _pick_flavor(192, 128, tuple(f for f in rubin_fp8_pool if f != (192, 128))) != (192, 128)
    # Every flavor Rubin can pick has a module ON DISK -- the map is the only
    # thing standing between _pick_flavor and a KeyError/ImportError deep in
    # module loading, so check the file, not the map key it came from.
    import pathlib

    import cudnn.sdpa.fwd.kernels as _k

    kdir = pathlib.Path(_k.__file__).parent
    for flavor in rubin_pool:
        assert (kdir / _SM107_KERNEL_FILES[flavor]).is_file(), (flavor, _SM107_KERNEL_FILES[flavor])


# --------------------------------------------------------------------------
# Capability row: what sdpa_fwd_prefill_sm107 accepts, and what it declines.
# Every accept below corresponds to a case MEASURED on Rubin against an
# explicit fp32 reference; every decline corresponds to machinery the kernels
# genuinely lack.  Declines are asserted, never skipped -- and the ones marked
# INVERTS-WHEN are counter-assertions to flip, not delete, when the feature
# lands.
# --------------------------------------------------------------------------


def _f16_facts(**kw):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga

    base = dict(
        b=2,
        h_q=8,
        h_kv=8,
        s_q=512,
        s_kv=512,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.HALF,
        dtype_o=cudnn.data_type.HALF,
        device_cc=(10, 7),
    )
    base.update(kw)
    return ga.SdpaGraphFacts(**base)


def _caps(name):
    from cudnn.sdpa.fwd import engines

    return {s.name: s.capabilities for s in engines.ENGINE_SPECS}[name]


def test_sm107_f16_row_owns_the_rubin_lane():
    """One row per ARCH LINE: sm100 stops at cc 10.6, sm107 starts at 10.7."""
    from cudnn.sdpa.fwd import engines

    sm107, sm100 = _caps("sdpa_fwd_prefill_sm107"), _caps("sdpa_fwd_prefill_sm100")
    assert engines.mismatch(sm107, _f16_facts()) is None
    assert "SM107-119" in engines.mismatch(sm107, _f16_facts(device_cc=(10, 0)))
    assert "SM100-106" in engines.mismatch(sm100, _f16_facts(device_cc=(10, 7)))


@pytest.mark.parametrize("d", [(128, 128), (256, 256), (512, 512)])
def test_sm107_f16_accepts_every_native_shape(d):
    from cudnn.sdpa.fwd import engines

    assert engines.mismatch(_caps("sdpa_fwd_prefill_sm107"), _f16_facts(d_qk=d[0], d_v=d[1])) is None


def test_sm107_f16_accepts_both_dtypes():
    """A frozenset field is one claim PER MEMBER -- exercise bf16, not just the
    fp16 the suite happens to default to."""
    import cudnn
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    for dt in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
        assert engines.mismatch(caps, _f16_facts(dtype=dt, dtype_o=dt)) is None


@pytest.mark.parametrize(
    "feature",
    [
        dict(causal=True),
        dict(causal=True, bottom_right=True),
        dict(right_band_widening=True),
        dict(window_left=128, causal=True),
        dict(padded=True),
        dict(has_sink=True),
        dict(h_kv=2),  # GQA 4:1
    ],
)
def test_sm107_f16_accepts_the_measured_feature_set(feature):
    """Each of these was validated on Rubin (cos = 1.000000)."""
    from cudnn.sdpa.fwd import engines

    assert engines.mismatch(_caps("sdpa_fwd_prefill_sm107"), _f16_facts(**feature)) is None


def test_sm107_f16_serves_d192_on_its_native_kernel():
    """d192xd128 is a NATIVE f16 Rubin flavor (``sm107/prefill_d192_d128_f16.py``
    -- the d128 body with ``make_cfg_d192``), not an envelope ride.

    Both halves of that claim are pinned, because they can drift apart
    silently: the row must LIST the shape (an envelope acceptance would pass
    this assertion too, which is why the d_shapes membership is asserted
    separately), and the adapter must PICK the native kernel for it
    (test_rubin_f16_never_picks_a_flavor_it_has_no_kernel_for).  The FP8 and
    MXFP8 Rubin lines genuinely lack the sibling -- see
    test_sdpa_fp8_sm107.py, where d192 is DECLINED rather than padded."""
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert (192, 128) in caps.d_shapes
    assert engines.mismatch(caps, _f16_facts(d_qk=192, d_v=128)) is None


def test_sm107_f16_thd_is_served_on_every_flavor():
    """FULLY INVERTED 2026-09-09: every f16 flavor now serves THD.

    The row must cover exactly `d_shapes` -- no more (a shape with no kernel
    would KeyError inside module loading) and no less (a served shape left out
    is a capability the engine silently declines).  Asserted on real facts
    objects, not the dataclass field, so it walks the path mismatch() takes.

    This keeps `thd_d_shapes` as a named SET rather than collapsing to `True`,
    because that is what makes the next partial arch line expressible without
    reintroducing a boolean that cannot describe it."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.config_sm107 import SM107_F16_THD_SHAPES

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert caps.thd is True
    assert caps.thd_d_shapes is SM107_F16_THD_SHAPES
    assert SM107_F16_THD_SHAPES == caps.d_shapes, "every served f16 shape must serve THD, and no unserved one may"
    for d_qk, d_v in sorted(caps.d_shapes):
        assert engines.mismatch(caps, _f16_facts(thd=True, padded=True, d_qk=d_qk, d_v=d_v)) is None, (d_qk, d_v)


def test_sm107_f16_split_coverage_and_pack_gqa_gate():
    """Dense and packed split admission retains the unwired feature boundaries."""
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert caps.split_kv_supported is True
    for d_qk, d_v in _FLAVORS:
        why = engines.mismatch(caps, _f16_facts(d_qk=d_qk, d_v=d_v), engines.SdpaFwdKnobs(split_kv=2))
        assert (why is None) == (d_v == 128), (d_qk, d_v, why)
    for feature in (dict(thd=True, padded=True), dict(padded=True), dict(has_sink=True)):
        assert engines.mismatch(caps, _f16_facts(**feature), engines.SdpaFwdKnobs(split_kv=2)) is not None
    # THD nonpaged d128 unsplit has no packed leg (dense d128 packs on the shared SM100 body since issue #1472, below).
    assert engines.mismatch(caps, _f16_facts(thd=True, padded=True), engines.SdpaFwdKnobs(pack_gqa=True)) is not None

    for d_qk, d_v, paged in ((128, 128, True), (192, 128, False)):
        facts = _f16_facts(d_qk=d_qk, d_v=d_v, thd=True, padded=True, has_paged_kv=paged, page_size=16 if paged else 0)
        knobs = engines.SdpaFwdKnobs(cga=1, split_kv=2, pack_gqa=False)
        assert engines.mismatch(caps, facts, knobs) is None
        assert (engines.mismatch(caps, dataclasses.replace(facts, has_sink=True), knobs) is None) == paged
        bounded = dataclasses.replace(facts, shape_overrides=True, max_total_seq_len_q=facts.b * facts.s_q)
        assert engines.mismatch(caps, bounded, knobs) is None
        assert engines.mismatch(caps, dataclasses.replace(bounded, max_total_seq_len_q=None), knobs) is not None
    for d in (128, 256):
        facts = _f16_facts(d_qk=d, d_v=d, thd=True, padded=True, has_paged_kv=True, page_size=16)
        knobs = engines.SdpaFwdKnobs(cga=2, split_kv=1, pack_gqa=False)
        assert engines.mismatch(caps, facts, knobs) is None
        assert engines.mismatch(caps, dataclasses.replace(facts, thd=False), knobs) is not None
        # cc 10.7 paged THD + attention sink: the sink composes with the paged THD leg -- unsplit, packed or not,
        # at every cluster width the leg admits -- while dense paged queries and sink x split-KV keep their declines.
        sink = dataclasses.replace(facts, has_sink=True)
        gqa_sink = dataclasses.replace(sink, h_q=16, h_kv=2)
        assert engines.mismatch(caps, sink, knobs) is None, d
        assert engines.mismatch(caps, gqa_sink, dataclasses.replace(knobs, pack_gqa=True)) is None, d
        if d == 128:
            for packed in (False, True):  # the two-slab cga1 prefill body (supports_paged_prefill_cga1)
                assert engines.mismatch(caps, gqa_sink, engines.SdpaFwdKnobs(cga=1, split_kv=1, pack_gqa=packed)) is None
        assert "THD queries" in engines.mismatch(caps, dataclasses.replace(sink, thd=False), knobs)
        assert engines.mismatch(caps, sink, dataclasses.replace(knobs, split_kv=2)) is not None


def test_sm107_dense_d128_shared_legs_admission():
    """Issue #1472: the cc 10.7 half row admits the shared SM100 d128 bodies on DENSE d128 half graphs -- PackGQA (the
    shared prefill body at cga2) and cga1 (the shared decode tile) -- and keeps every other leg where it was: the
    pre-folded scale stays on the Rubin body (its arm is not in the shared bodies), dense d256 PackGQA stays declined,
    THD nonpaged keeps the cga2 prefill pipeline, and the standalone cga domain mirrors the row (keep the three in
    lockstep).  An MHA PackGQA pin is the bit-exact unpacked fold (PACK_G = 1), honorable as on the SM100 row; the
    heuristics never propose it."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.api_dsl import supported_cgas_for

    caps = _caps("sdpa_fwd_prefill_sm107")
    pack, cga1 = engines.SdpaFwdKnobs(pack_gqa=True), engines.SdpaFwdKnobs(cga=1)
    assert engines.mismatch(caps, _f16_facts(h_kv=2), pack) is None
    assert engines.mismatch(caps, _f16_facts(h_kv=2, d_qk=64, d_v=64), pack) is None, "the d64 envelope rides the same bodies"
    assert engines.mismatch(caps, _f16_facts(), pack) is None, "MHA: PACK_G = 1, the bit-exact unpacked fold (SM100 parity)"
    assert "pre-folded" in engines.mismatch(caps, _f16_facts(h_kv=2, attn_scale_prefolded=True), pack)
    assert engines.mismatch(caps, _f16_facts(h_kv=2, d_qk=256, d_v=256), pack) is not None
    assert engines.mismatch(caps, _f16_facts(h_kv=2, s_q=4), cga1) is None
    assert engines.mismatch(caps, _f16_facts(s_q=4), cga1) is None, "MHA rides the decode tile unpacked"
    assert "outside this engine's domain" in engines.mismatch(caps, _f16_facts(h_kv=2, thd=True, padded=True), cga1)
    assert "outside this engine's domain" in engines.mismatch(caps, _f16_facts(h_kv=2, attn_scale_prefolded=True), cga1)
    assert supported_cgas_for((128, 128), fp8=False, device_cc=(10, 7)) == (1, 2)


@pytest.mark.parametrize("dtype_name", ["HALF", "BFLOAT16"])
@pytest.mark.parametrize("group", [2, 4, 8, 16])
def test_sm107_paged_d256_pack_gqa_support_contract(dtype_name, group):
    """Packing is explicit and confined to the qualified paged half THD path."""
    import cudnn
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    dt = getattr(cudnn.data_type, dtype_name)
    facts = _f16_facts(
        h_q=group * 2,
        h_kv=2,
        d_qk=256,
        d_v=256,
        dtype=dt,
        dtype_o=dt,
        thd=True,
        padded=True,
        has_paged_kv=True,
        page_size=16,
    )
    knobs = engines.SdpaFwdKnobs(cga=2, split_kv=1, pack_gqa=True)
    assert engines.mismatch(caps, facts, knobs) is None
    bounded = dataclasses.replace(facts, wants_stats=True, shape_overrides=True, max_total_seq_len_q=facts.b * facts.s_q)
    assert engines.mismatch(caps, bounded, knobs) is None
    # cc 10.7 paged THD + sink: the sink composes with paged D256 PackGQA (unsplit); sink x split stays declined.
    assert engines.mismatch(caps, dataclasses.replace(facts, has_sink=True), knobs) is None
    assert engines.mismatch(caps, dataclasses.replace(facts, has_sink=True), dataclasses.replace(knobs, split_kv=2)) is not None
    for changed in (
        dict(thd=False),
        dict(has_paged_kv=False),
        dict(device_cc=(10, 0)),
        dict(device_cc=(10, 8)),
        dict(h_q=6),
    ):
        assert engines.mismatch(caps, dataclasses.replace(facts, **changed), knobs) is not None, changed
    for changed in (dict(cga=1), dict(split_kv=2)):
        assert engines.mismatch(caps, facts, dataclasses.replace(knobs, **changed)) is not None, changed
    blackwell = dataclasses.replace(facts, device_cc=(10, 0))
    assert engines.mismatch(_caps("sdpa_fwd_prefill_sm100"), blackwell, knobs) is not None


def test_sm107_fp8_pack_gqa_is_d128_only():
    """The Rubin per-tensor FP8 row packs GQA on the d128 flavor only (`pack_gqa_d_shapes = {(128, 128)}`: the
    d192x128 / d256 / d512 siblings carry no PackGQA path) while it serves those flavors UNPACKED.  This is the
    typed decline behind `_skip_pack_gqa_wide_on_rubin` in test_sdpa_fwd_fp8_sm100.py -- the one Rubin marker that
    survived retiring the d128-only-era skips -- so it is asserted on real facts here, where no GPU is needed: a
    packed d192 / d256 / d512 graph is ineligible with a reason naming the knob, the same graph unpacked is eligible,
    and packed d128 is eligible.  When a wider kernel gains PackGQA: widen the row, INVERT that shape's packed
    assertion and drop the marker (test/AGENTS.md: invert the counter assertion, do not delete it)."""
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107_fp8")
    assert caps.pack_gqa_d_shapes == frozenset({(128, 128)})
    for d_qk, d_v in ((192, 128), (256, 256), (512, 512)):
        facts = _f16_facts(**_fp8_ungated_kw(h_kv=2, d_qk=d_qk, d_v=d_v))
        assert engines.mismatch(caps, facts) is None, (d_qk, d_v, "the unpacked graph must be served")
        why = engines.mismatch(caps, facts, engines.SdpaFwdKnobs(pack_gqa=True))
        assert why is not None and "pack_gqa" in why, (d_qk, d_v, why)
    packed_d128 = _f16_facts(**_fp8_ungated_kw(h_kv=2, d_qk=128, d_v=128))
    assert engines.mismatch(caps, packed_d128, engines.SdpaFwdKnobs(pack_gqa=True)) is None


def test_sm107_fp8_paged_sink_declines():
    """The paged + sink lift is the HALF row's: the Rubin per-tensor FP8 row has no paged capability, so a paged
    THD + sink FP8 graph on cc 10.7 keeps its typed decline (the fp8 row's `paged_kv=not rubin_row`)."""
    from cudnn.sdpa.fwd import engines

    why = engines.mismatch(_caps("sdpa_fwd_prefill_sm107_fp8"), _quant_facts(has_paged_kv=True, page_size=16, padded=True, thd=True, has_sink=True))
    assert why is not None and "paged" in why, why


@pytest.mark.parametrize("family", ["fp8", "mxfp8"])
def test_sm107_quantized_strided_stats_is_ported_for_all_flavors(family, monkeypatch):
    """Invert the previous D256/D512 gap detector; runtime Stats strides are
    accepted by each prepared entry, with native numerics in the FP8 suite."""
    from cudnn.sdpa.fwd.kernels import _fp8_host, _mxfp8_host

    host_module = _fp8_host if family == "fp8" else _mxfp8_host
    load_kw = _DTYPE_FAMILIES[1 if family == "fp8" else 2][1]
    caps = _caps("sdpa_fwd_prefill_sm107_" + family)
    assert caps.stats is True and caps.d_shapes == frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})
    captured = []
    sentinel = object()

    def compile_host(*args, **kwargs):
        captured.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(host_module, "compile_host", compile_host)
    for flavor in _FLAVORS:
        mod = _load(flavor, rubin=True, **load_kw)
        assert not hasattr(mod, "compile"), "quantized attention must use the pointer compiler"
        with open(mod.__file__, encoding="utf-8") as fh:
            assert "strided Stats not ported" not in _code_lines(fh.read())
        # Bypass the memoized result so the real entry forwards its metadata.
        assert mod.compile_prepared.__wrapped__(d_qk=flavor[0], d_v=flavor[1], has_lse=True) is sentinel
    assert len(captured) == len(_FLAVORS)


def test_sm107_rows_carry_the_padded_stats_trim():
    """Every Rubin template carries the per-batch seq_len_q trim (padded q
    rows write LSE=-inf, O=0), so the rows serve dense padded Stats and the
    Capabilities record has no dense_seq_q_trim field left to declare."""
    from cudnn.sdpa.fwd import engines

    for row in ("sdpa_fwd_prefill_sm107", "sdpa_fwd_prefill_sm107_fp8", "sdpa_fwd_prefill_sm107_mxfp8"):
        assert _caps(row).padded_stats is True, row
    assert not hasattr(engines.Capabilities, "dense_seq_q_trim")


def test_sm107_rows_serve_natural_scheduling_only():
    """The ROW-WIDE floor of every Rubin row stays NATURAL-only: every LPT
    variant is claimed per flavor (`sched_policies_by_d_shape`) where it is
    validated.  SCHED_LPT_L2 needs `qh_per_kh` / `seqlen_kv` at every decode
    call site, which only the d128 / d192x128 FP8 and MXFP8 kernels pass, so it
    may appear ONLY on those flavors -- INVERTED 2026-09-14 (it used to appear
    on none, and the MXFP8 row claimed nothing beyond NATURAL).

    (This test used to assert LPT was "not honored" by the ported kernels; the
    cause was the dropped `lpt_q_tiles_in_cga_units` argument, fixed in #1001.
    The per-flavor claims are pinned by the *_advertises_lpt_* tests.)"""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

    lpt_l2_flavors = {
        "sdpa_fwd_prefill_sm107": set(),
        "sdpa_fwd_prefill_sm107_fp8": {(128, 128), (192, 128)},
        "sdpa_fwd_prefill_sm107_mxfp8": {(128, 128), (192, 128)},
    }
    for row, l2_shapes in lpt_l2_flavors.items():
        caps = _caps(row)
        assert caps.sched_policies == frozenset({SCHED_NATURAL}), row
        assert SCHED_LPT not in caps.sched_policies, row
        assert SCHED_LPT_L2 not in caps.sched_policies, row
        claimed_l2 = {shape for shape, dom in caps.sched_policies_by_d_shape if SCHED_LPT_L2 in dom}
        assert claimed_l2 == l2_shapes, (row, claimed_l2)
    assert dict(_caps("sdpa_fwd_prefill_sm107_mxfp8").sched_policies_by_d_shape) == {
        (128, 128): frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
        (192, 128): frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
    }, "MXFP8 claims LPT and LPT_L2 exactly where its kernels thread the decode inputs"


def test_sched_points_falls_back_along_the_preference_order():
    """A row whose heuristic primary is outside its DOMAIN must fall back along
    the same preference order, not drop to NATURAL -- and must not list a policy
    twice.  The old form returned [NATURAL, LPT, NATURAL] for a causal Rubin FP8
    graph: NATURAL first (losing the LPT balancing) and a duplicate autotune
    slot.  Rows whose primary IS in domain are unaffected."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines, heuristics

    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    rubin_fp8 = caps["sdpa_fwd_prefill_sm107_fp8"]
    facts = _f16_facts()
    facts = facts.__class__(**{**facts.__dict__, "causal": True, "is_fp8": True, "device_cc": (10, 7)})
    points = heuristics._sched_points(rubin_fp8, facts)
    assert len(points) == len(set(points)), points
    assert SCHED_LPT_L2 not in rubin_fp8.sched_policies
    # d128 FP8 claims {NATURAL, LPT, LPT_L2} (2026-09-14).  These facts have
    # h_q == h_kv (no GQA) and a tiny grid, so the Rubin rule leads with LPT
    # and keeps both other policies as autotune runners.
    assert points == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL], points

    # The per-flavor claim must REACH the ranking: at (256, 256) the FP8 row's
    # effective domain is {NATURAL, LPT}, the causal primary (LPT_L2) is out of
    # domain, so the fallback walks the preference order -- LPT first, NATURAL
    # as the autotune alternative.  Before the heuristics read the flavor's
    # domain this returned [NATURAL] and LPT was never proposed.
    facts256 = facts.__class__(**{**facts.__dict__, "d_qk": 256, "d_v": 256})
    assert heuristics._sched_points(rubin_fp8, facts256) == [SCHED_LPT, SCHED_NATURAL]

    # (192, 128) FP8 claims LPT_L2 too.  Without GQA the Rubin rule still leads
    # with LPT on this few-wave grid; give the KV heads Q heads to share and
    # the causal primary becomes LPT_L2, which IS in domain now, so the ranking
    # leads with it and keeps both fallbacks as autotune runners.  A mask-free
    # graph gains nothing from either remap: NATURAL alone.
    facts192 = facts.__class__(**{**facts.__dict__, "d_qk": 192, "d_v": 128})
    assert heuristics._sched_points(rubin_fp8, facts192) == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL]
    gqa192 = facts.__class__(**{**facts192.__dict__, "h_kv": 2})
    assert heuristics._sched_points(rubin_fp8, gqa192) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL]
    dense192 = facts.__class__(**{**facts192.__dict__, "causal": False})
    assert heuristics._sched_points(rubin_fp8, dense192) == [SCHED_NATURAL]

    # ...and the multi-element case, which is the branch the single-element
    # shortcut above skips: with a causal primary (LPT_L2, a GQA graph) OUT of
    # a {NATURAL, LPT} domain, the fallback must walk the preference order
    # rather than dropping straight to NATURAL -- the bug this function was
    # fixed for.  The per-shape claims are cleared so the row-wide domain is
    # the one in force.
    widened = dataclasses.replace(rubin_fp8, sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT}), sched_policies_by_d_shape=())
    widened_points = heuristics._sched_points(widened, facts.__class__(**{**facts.__dict__, "h_kv": 2}))
    assert widened_points[0] == SCHED_LPT, widened_points
    assert set(widened_points) == {SCHED_NATURAL, SCHED_LPT}, widened_points
    assert len(widened_points) == len(set(widened_points)), widened_points


def test_sm107_rows_claim_optional_stats():
    """INVERTED 2026-09-08: every Rubin kernel now const_expr's its LSE store
    out on a None ``lse_tensor`` and ``compile(has_lse=False)`` binds no dummy
    buffer, so a stats-less graph reports ``get_workspace_size() == 0``.  The
    FP8 row already claimed this from the shared spec -- which was a LIE for the
    d256/d512 FP8 kernels it was widened to in the session-5 work, and is now
    true."""
    for row in ("sdpa_fwd_prefill_sm107", "sdpa_fwd_prefill_sm107_fp8", "sdpa_fwd_prefill_sm107_mxfp8"):
        assert _caps(row).lse_optional is True, row


def test_rubin_mxfp8_forward_row_exists():
    """The Rubin MXFP8 row (slot 16) and Blackwell both have a d512 kernel.
    INVERTED 2026-09-09: d192xd128 gained a
    Rubin MXFP8 sibling, at cga2 ONLY -- at cga1 that flavor's scale-factor
    tiles start past the 256 KiB version-0 tcgen05 descriptor window.  SM100's
    row stays capped at cc 10.6.
    INVERTED 2026-10-08: the row serves THD at d256 ONLY -- that body rides the
    FROST THD contract at cga1 with the packed per-sequence-tile-padded
    scale-factor layout, so ``thd_d_shapes`` is the config's
    ``SM107_MXFP8_THD_SHAPES`` (one constant with the standalone wrapper's
    Rubin THD gate and the config backstop); the d128 / d192xd128 / d512 MXFP8
    bodies keep the pre-upstream THD arm and stay declined through it.
    Split-KV and PackGQA stay declined row-wide."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.config_sm107 import SM107_MXFP8_THD_SHAPES

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    assert caps.is_mxfp8 is True
    assert caps.d_shapes == frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})
    assert (512, 512) in _caps("sdpa_fwd_prefill_sm100_mxfp8").d_shapes
    assert (192, 128) in caps.d_shapes
    # (192, 128) takes NO cgas_by_d_shape entry, so it inherits the row default
    # cgas={2}.  That is a DESCRIPTOR constraint, not a tuning choice: at cga1
    # the four SF tiles land at 256-278 KiB, past the version-0 tcgen05
    # descriptor window, and the UTCCP would read Q data as scale factors
    # (LSE=+inf, O=NaN on every cell).  SM100 serves the same shape at {1, 2}.
    assert caps.cgas == frozenset({2})
    assert all(shape != (192, 128) for shape, _ in caps.cgas_by_d_shape)
    assert _caps("sdpa_fwd_prefill_sm100_mxfp8").cgas_by_d_shape != caps.cgas_by_d_shape
    # Exact-native only: the SF tensors are not zero-padded.
    assert caps.d_pad_multiple == 0
    # THD at d256 only, on the named SET (never a bare True), with both length forms and
    # the per-batch padded Stats layout like every other Rubin THD row.  The mismatch()
    # walk over the row's d_shapes (admit exactly these, decline the rest typed) is
    # test_sdpa_fp8_sm107.py::test_sm107_mxfp8_thd_shapes_match_the_row.
    assert caps.thd is True
    assert caps.thd_d_shapes is SM107_MXFP8_THD_SHAPES
    assert caps.thd_d_shapes == frozenset({(256, 256)}) and caps.thd_d_shapes < caps.d_shapes
    assert caps.thd_padded_stats is True and caps.cu_seq_len is True
    # The machinery the ported kernels lack stays declined: split-KV and PackGQA.
    assert caps.split_kv_supported is False
    assert caps.pack_gqas == frozenset({False})
    assert _caps("sdpa_fwd_prefill_sm100_mxfp8").sm_hi == 106


def test_rubin_mxfp8_row_serves_paged_pools_on_d128_d256():
    """The cc 10.7 MXFP8 row serves the SM100 paged MXFP8 pool contract (#1214) on d128 / d256 with DENSE queries:
    F8_128x4 descale pools paging with K/V, page_size % 128, the sink / causal / bottom-right / SWA / padding masks,
    Stats and the f16x2 exponent arm composed.  Declined, each by its own typed reason: page 64, THD queries over
    pools (stage 2), the d192x128 / d512 pools (stage 3), a block-scaled O over pools, the pre-folded scale over paged
    KV, pools without a padding mask, and split_kv > 1 (also with a sink).  The half row keeps its THD requirement
    and the per-tensor FP8 row serves no paged KV."""
    import cudnn
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    assert caps.paged_kv is True and caps.paged_d_shapes == frozenset({(128, 128), (256, 256)})
    assert caps.sink is True and caps.thd is True and caps.thd_d_shapes == frozenset({(256, 256)})  # THD at d256 only (#1488); pools serve dense queries

    def paged(**kw):
        return _quant_facts(**{"is_mx": True, "has_paged_kv": True, "page_size": 128, "padded": True, **kw})

    for d in (128, 256):
        for extra in (
            {},
            dict(has_sink=True),
            dict(causal=True, bottom_right=True, right_bound=0),
            dict(page_size=256),
            dict(dtype=cudnn.data_type.FP8_E5M2),
            dict(window_left=128, causal=True, right_bound=0),
            dict(s_q=1),
            dict(wants_stats=True),
            dict(softmax_precision=cudnn.data_type.HALF),
            dict(has_sink=True, causal=True, bottom_right=True, right_bound=0, wants_stats=True, s_q=4),
        ):
            assert engines.mismatch(caps, paged(d_qk=d, d_v=d, **extra)) is None, (d, extra)
    # The explicit knobs of the served plans: cga2 at d128, cga1 at d256, NATURAL, 128x128, unpacked, unsplit.
    assert engines.mismatch(caps, paged(), SdpaFwdKnobs(sched_policy=0, tile_m=128, tile_n=128, cga=2, pack_gqa=False, split_kv=1)) is None
    assert engines.mismatch(caps, paged(d_qk=256, d_v=256), SdpaFwdKnobs(sched_policy=0, tile_m=128, tile_n=128, cga=1, pack_gqa=False, split_kv=1)) is None
    for kw, needle in (
        (dict(page_size=64), "multiple of 128"),
        (dict(thd=True), "THD"),
        (dict(d_qk=192, d_v=128), "d128, d256 kernel flavors only"),
        (dict(d_qk=512, d_v=512), "d128, d256 kernel flavors only"),
        (dict(o_block_scale=32, dtype_o=cudnn.data_type.FP8_E4M3), "block-scaled O"),
        (dict(attn_scale_prefolded=True), "paged-KV kernel bodies"),
        (dict(padded=False), "use_padding_mask"),
    ):
        reason = engines.mismatch(caps, paged(**kw))
        assert reason is not None and needle in reason, (kw, reason)
    assert engines.mismatch(caps, paged(has_sink=True), SdpaFwdKnobs(split_kv=2)) is not None
    # The half row: dense paged queries stay declined on cc 10.7 (THD queries are its paged form).
    half = engines.mismatch(_caps("sdpa_fwd_prefill_sm107"), _f16_facts(has_paged_kv=True, page_size=16, padded=True, thd=False))
    assert half is not None and "THD queries" in half, half
    assert not _caps("sdpa_fwd_prefill_sm107_fp8").paged_kv


def test_sm107_quantized_rows_serve_d192_on_their_native_kernels():
    """ACCEPT side of the 2026-09-09 d192 addition, one case PER DTYPE MEMBER of
    each row's frozenset -- a row listing two dtypes is making two claims, and a
    suite that exercises only E4M3 leaves the other as an untested assertion
    (contract rule 12).

    The shape must land on the NATIVE (192, 128) kernel, not an envelope: the
    d256 flavor's floor (255) would otherwise swallow it onto a padded path
    nobody validated."""
    import cudnn
    from cudnn.sdpa.fwd import engines

    for row, is_mx in (("sdpa_fwd_prefill_sm107_fp8", False), ("sdpa_fwd_prefill_sm107_mxfp8", True)):
        caps = _caps(row)
        assert (192, 128) in caps.d_shapes, row
        for dt in sorted(caps.dtypes, key=lambda d: int(d)):
            facts = _quant_facts(d_qk=192, d_v=128, dtype=dt, is_mx=is_mx)
            assert engines.mismatch(caps, facts) is None, (row, dt)
            assert engines._selected_d_shape(caps, facts) == (192, 128), (row, dt)


def test_sm107_d192_mxfp8_is_cga2_only_because_of_the_descriptor_window():
    """REJECT side, and the reason is a DESCRIPTOR constraint rather than taste.

    At cga1 the K/V rings are not halved, so this flavor's four scale-factor
    tiles start at 256-278 KiB -- past the 256 KiB version-0 tcgen05 descriptor
    window, where ``start_address`` wraps to 0 and the UTCCP copies Q DATA bytes
    into the SF TMEM columns (LSE=+inf, O=NaN on 100% of cells, at every shape).

    Two independent guards, and this test pins BOTH: the row keeps (192, 128) on
    the default ``cgas={2}`` so a graph cannot request cga1, and the kernel body
    raises at import if it is ever handed one anyway.  INVERTS-WHEN DESC_VERSION
    is derived from the layout and the version-1 SF path is validated on Rubin
    -- which is not free: desc_version=1 on the d128/d256 MXFP8 tiles turned 21
    green tests red (2026-09-08)."""
    import pytest as _pytest
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    facts = _quant_facts(d_qk=192, d_v=128, is_mx=True)
    # Guard 1 -- knob domain. cga2 is honored, cga1 makes the plan ineligible.
    assert engines.mismatch(caps, facts, SdpaFwdKnobs(cga=2)) is None
    assert engines.mismatch(caps, facts, SdpaFwdKnobs(cga=1)) is not None
    # SM100 serves the same shape at BOTH widths -- so this is Rubin-specific,
    # which is what makes it worth pinning rather than assuming.
    assert 1 in engines.effective_cgas(_caps("sdpa_fwd_prefill_sm100_mxfp8"), facts, 1)

    # Guard 2 -- the kernel body itself, so the constraint survives a row edit.
    with _pytest.raises(ValueError, match="CTA_MMA must be 2"):
        _load((192, 128), rubin=True, fp8=True, pertensor=False, dtype_qkv=_E4M3, dtype_o=_BF16_OUT, cta_mma=1)


def _quant_facts(**kw):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga

    is_mx = kw.pop("is_mx", False)
    base = dict(
        b=2,
        h_q=4,
        h_kv=4,
        s_q=384,
        s_kv=384,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(10, 7),
    )
    base.update(kw)
    base["is_mxfp8" if is_mx else "is_fp8"] = True
    return ga.SdpaGraphFacts(**base)


def test_sm107_ones_tile_stays_inside_the_v0_descriptor_window():
    """The row-sum "ones" tile is allocated LAST and read as an MMA operand, so
    its OFFSET -- not its size -- has to stay under 256 KiB while DESC_VERSION
    is 0.  At cga1 the K/V rings are not halved, and with a HALF-PRECISION O
    the d192 tile lands at 272 KiB: its version-0 descriptor wraps to offset 0
    and the Sigma MMA multiplies sQ instead of all-ones, so `total_sum` becomes
    a function of Q -- wrong LSE and wrong O on every row, no crash.

    Caught in review, not by a test, because the kernel docstring ARGUED cga1
    was safe from a hand-summed figure that omitted two buffers.  The guard is
    therefore derived from the allocator's own constants; this pins that it
    fires exactly on the unsafe config and on nothing else, so the safe ones
    cannot be walled off by a future over-broad tightening."""
    import pytest as _pytest

    _E4M3_OUT, _BF16 = 0, 2
    # (id, flavor, cta_mma, dtype_o, per-tensor?).  The MXFP8 rows matter on
    # their own: those kernels carry FOUR scale-factor slabs the per-tensor
    # sibling does not, so their ones tile sits ~7 KiB higher and a guard that
    # forgot the SF slabs would under-report the offset rather than fire.
    safe = [
        ("d128", (128, 128), 1, _E4M3_OUT, True),
        ("d128", (128, 128), 1, _BF16, True),
        ("d128", (128, 128), 2, _BF16, True),
        ("d192", (192, 128), 1, _E4M3_OUT, True),
        ("d192", (192, 128), 2, _BF16, True),
        ("d128-mx", (128, 128), 1, _E4M3_OUT, False),
        ("d128-mx", (128, 128), 1, _BF16, False),
        ("d128-mx", (128, 128), 2, _E4M3_OUT, False),
        ("d128-mx", (128, 128), 2, _BF16, False),
        ("d192-mx", (192, 128), 2, _E4M3_OUT, False),
        ("d192-mx", (192, 128), 2, _BF16, False),
    ]
    for _id, flavor, cta, dto, pertensor in safe:
        mod = _load(flavor, rubin=True, fp8=True, pertensor=pertensor, dtype_qkv=_E4M3, dtype_o=dto, cta_mma=cta)
        assert mod._ONES_SMEM_OFFSET < 256 * 1024, (_id, cta, dto, mod._ONES_SMEM_OFFSET)

    # d192 MXFP8 x cga1 never reaches the ones guard: its own CTA_MMA raise
    # fires first (the four SF tiles cross the window before the ones tile
    # does).  Pin WHICH guard speaks, so a future reordering cannot silently
    # swap a precise diagnostic for a vaguer one.
    with _pytest.raises(ValueError, match="CTA_MMA must be 2"):
        _load((192, 128), rubin=True, fp8=True, pertensor=False, dtype_qkv=_E4M3, dtype_o=_BF16, cta_mma=1)

    # The one unsafe combination: d192 x cga1 x half-precision O.
    with _pytest.raises(ValueError, match="version-0 tcgen05 descriptor window"):
        _load((192, 128), rubin=True, fp8=True, pertensor=True, dtype_qkv=_E4M3, dtype_o=_BF16, cta_mma=1)


def test_sm107_mxfp8_tmem_map_fills_the_rubin_allocation_exactly():
    """The row-sum (Sigma) columns took the 32 TMEM columns the scale-factor
    tiles used to leave free: at d192 the map ends EXACTLY at the 576-column
    Rubin allocation, at d128 at 564.  The import-time raise catches an
    overflow; this pins the layout so a re-literalled offset that still fits
    cannot drift unnoticed."""
    for flavor, end in (((192, 128), 576), ((128, 128), 564)):
        mod = _load(flavor, rubin=True, fp8=True, pertensor=False, dtype_qkv=_E4M3, dtype_o=_BF16_OUT, cta_mma=2)
        assert mod.LAYOUT.TOTAL_COLS == 576, flavor
        assert mod.LAYOUT.SF_V_OFF + mod.SF_TMEM_COLS_V == end, (flavor, mod.LAYOUT.SF_V_OFF, mod.SF_TMEM_COLS_V)

    # ...and the STANDALONE wrapper must decline it too, rather than letting a
    # bare ValueError escape from compile() (contract rule 8b' -- the kernel
    # raise and the wrapper gate are two enforcement points for one fact).
    # Assert the wrapper's OWN decision function, so this holds from any host:
    # the Rubin arm is unreachable on a non-cc-10.7 box, which is exactly the
    # kind of branch that rots untested.
    from cudnn.sdpa.fwd.api_dsl import supported_cgas_for

    assert supported_cgas_for((192, 128), fp8=True, device_cc=(10, 7)) == (2,)
    # ...and check_support must ACT on that, not merely compute it.  The helper
    # was extracted in review and the `_value_error_if` that consumed it got
    # dropped in the same edit, so `supported_cgas` was computed and discarded --
    # an explicit cga=1 then cleared support validation and died inside
    # compile(), which is exactly the rule-8b' failure the helper exists to
    # prevent.  Pin the CONSUMPTION, since the value alone was already correct.
    import inspect

    from cudnn.sdpa.fwd import api_dsl as _api_dsl

    _src = inspect.getsource(_api_dsl.SdpaFwdDslSm100.check_support)
    assert "supported_cgas_for(" in _src, "check_support no longer derives the CGA domain"
    assert "self.cga not in supported_cgas" in _src, "check_support computes supported_cgas but never validates self.cga against it"

    # ...and only there: Blackwell keeps both widths, and the f16 Rubin path is
    # not narrowed (it has no quantized SF or ones tile to push over the line).
    assert supported_cgas_for((192, 128), fp8=True, device_cc=(10, 0)) == (1, 2)
    assert supported_cgas_for((192, 128), fp8=False, device_cc=(10, 7)) == (1, 2)
    assert supported_cgas_for((128, 128), fp8=True, device_cc=(10, 7)) == (2,)
    assert supported_cgas_for((256, 256), fp8=True, device_cc=(10, 7)) == (1,)


# --- LPT on Rubin, advertised PER D-SHAPE -----------------------------------
#
# `sched_policies` is row-wide but LPT support is per-KERNEL: the SM107 port
# dropped `lpt_q_tiles_in_cga_units=True` on 9 of 11 flavors, and without it the
# LPT decode claims no tile and the kernel writes nothing. The argument is
# restored everywhere; only the flavors validated under LPT are ADVERTISED.


@pytest.mark.L0
def test_sm107_advertises_lpt_only_for_the_validated_d_shape():
    """D128/D256 serve LPT; the row-wide default stays NATURAL-only.

    An accept AND a reject, because a capability that is only ever exercised on
    its accepting side is an untested assertion (engine contract, Rule 9).
    """
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert caps.sched_policies == frozenset({SCHED_NATURAL}), "the row-wide floor must stay NATURAL"

    for d in (128, 256):
        assert SCHED_LPT in engines.effective_sched_policies(caps, _f16_facts(d_qk=d, d_v=d))
    for dq, dv in ((192, 128), (512, 512)):
        assert SCHED_LPT not in engines.effective_sched_policies(caps, _f16_facts(d_qk=dq, d_v=dv))

    # LPT_L2 is a separate, still-open gap: its decode needs qh_per_kh and
    # seqlen_kv, which the SM107 call sites do not pass. No flavor claims it.
    for shape in ((128, 128), (256, 256), (512, 512)):
        assert SCHED_LPT_L2 not in engines.effective_sched_policies(caps, _f16_facts(d_qk=shape[0], d_v=shape[1]))


def test_sm107_fp8_advertises_lpt_only_for_the_validated_d_shape():
    """The FP8 twin of the f16 test above: (256, 256), (192, 128) and -- since
    2026-09-14 -- (128, 128) serve LPT (validated through the standalone
    adapter, bit-identical to NATURAL -- see the row's comment); d512 does
    not; the row-wide floor stays NATURAL.  LPT_L2 needs qh_per_kh / seqlen_kv
    at every decode call site: the d128 and d192x128 kernels thread them and
    both claim it; d256 / d512 pass neither argument."""
    import cudnn
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107_fp8")
    assert caps.sched_policies == frozenset({SCHED_NATURAL}), "the row-wide floor must stay NATURAL"
    fp8 = dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_fp8=True)
    for shape in ((256, 256), (192, 128), (128, 128)):
        assert SCHED_LPT in engines.effective_sched_policies(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8)), shape
    assert SCHED_LPT not in engines.effective_sched_policies(caps, _f16_facts(d_qk=512, d_v=512, **fp8))
    for shape in ((192, 128), (128, 128)):
        assert SCHED_LPT_L2 in engines.effective_sched_policies(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8)), shape
    for shape in ((256, 256), (512, 512)):
        assert SCHED_LPT_L2 not in engines.effective_sched_policies(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8)), shape


@pytest.mark.L0
def test_sm107_fp8_lpt_knob_is_honored_or_ineligible_per_d_shape():
    """Requesting LPT on the FP8 row: ACCEPTED at d256, d192xd128 and d128,
    DECLINED (typed, naming the knob) at d512; LPT_L2: ACCEPTED at d192xd128
    and d128 only -- never silently downgraded (engine contract, Rule 4)."""
    import cudnn
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps("sdpa_fwd_prefill_sm107_fp8")
    fp8 = dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_fp8=True)
    for shape in ((256, 256), (192, 128), (128, 128)):
        assert engines.mismatch(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8), SdpaFwdKnobs(sched_policy=SCHED_LPT)) is None, shape
    why = engines.mismatch(caps, _f16_facts(d_qk=512, d_v=512, **fp8), SdpaFwdKnobs(sched_policy=SCHED_LPT))
    assert why is not None and "sched_policy" in why
    for shape in ((192, 128), (128, 128)):
        assert engines.mismatch(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8), SdpaFwdKnobs(sched_policy=SCHED_LPT_L2)) is None, shape
    for shape in ((256, 256), (512, 512)):
        why = engines.mismatch(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8), SdpaFwdKnobs(sched_policy=SCHED_LPT_L2))
        assert why is not None and "sched_policy" in why, shape


@pytest.mark.L0
def test_sm107_lpt_knob_is_honored_or_ineligible_per_d_shape():
    """Explicit LPT is honored on qualified flavors, never silently downgraded."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps("sdpa_fwd_prefill_sm107")
    for d in (128, 256):
        assert engines.mismatch(caps, _f16_facts(d_qk=d, d_v=d), SdpaFwdKnobs(sched_policy=SCHED_LPT)) is None
    for dq, dv in ((192, 128), (512, 512)):
        why = engines.mismatch(caps, _f16_facts(d_qk=dq, d_v=dv), SdpaFwdKnobs(sched_policy=SCHED_LPT))
        assert why is not None and "sched_policy" in why


@pytest.mark.L0
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_sm107_d256_desc_version_follows_the_stages_kv_layout(depth):
    """STAGES_KV moves the last V stage past the 14-bit tcgen05 address window
    at depth 4, so DESC_VERSION must be DERIVED, not a literal.

    This is the guard for a silent wrong answer: at depth 4 with a version-0
    descriptor the last two V stages alias the bottom of SMEM and the result is
    wrong from the KV iteration that first touches one (measured cos 0.68 / 0.50
    / 0.46 at n_kv 3/4/8).
    """
    from dataclasses import replace

    import cudnn.sdpa.fwd.kernels.sm107.prefill_d256_f16 as kern

    cfg = replace(kern.CFG, STAGES_KV=depth)
    want = 1 if depth >= 4 else 0
    assert int(kern._needs_desc_v1(cfg)) == want, f"STAGES_KV={depth} must select desc_version {want}"


@pytest.mark.L0
@pytest.mark.parametrize("bad_depth", [0, 1, 5])
def test_sm107_d256_rejects_an_out_of_domain_stages_kv(bad_depth):
    """An out-of-domain STAGES_KV must RAISE, never be silently defaulted.

    `make_cfg_d256` reads `stages_kv` as an OPTIONAL attribute, because the
    shared forward `TemplateParams` has no such field yet -- nothing on the
    shipped path sets it, so this is a latent path, not a live one. It stops
    being latent the day the knob is declared, and the failure it would have
    then is the quiet kind: a truthiness test maps an explicit 0 onto the
    default 2, so the engine runs a depth the caller did not ask for and the
    2..4 check below never sees it. That is knob SUBSTITUTION, which the
    engine contract forbids -- a knob is honored or the engine is ineligible.

    Pinning 0 specifically: 1 and 5 fail under any spelling, but 0 is the only
    value a truth test swallows, so it is the one that regresses silently.
    """
    from dataclasses import dataclass

    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.config_sm107 import make_cfg_d256

    @dataclass(frozen=True)
    class _ParamsWithStagesKv(TemplateParams):
        stages_kv: int = None

    with pytest.raises(ValueError, match="STAGES_KV"):
        make_cfg_d256(_ParamsWithStagesKv(stages_kv=bad_depth))


@pytest.mark.L0
def test_sm107_d256_stages_kv_defaults_when_the_attribute_is_absent():
    """The accept side of the test above: a plain TemplateParams (no
    `stages_kv` attribute at all) still gets the depth-2 default, so the fix
    for the explicit-zero case did not break the only path that ships."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.config_sm107 import make_cfg_d256

    cfg, _ = make_cfg_d256(TemplateParams())
    assert cfg.STAGES_KV == 2


@pytest.mark.L0
def test_thd_stats_padded_is_appended_to_the_public_signature():
    """The adapters' constructors are public and not keyword-only; an old
    positional call (..., thd, max_total_seq_len_q, max_total_seq_len_kv) must
    keep binding the same way, so every new parameter is APPENDED after the
    legacy prefix, in the order it landed.

    Tail history: ``thd_stats_padded`` closed the legacy prefix; #931 / #983
    appended ``sample_amax_o``, ``pv_bf16`` and ``stats_log2``; the fused
    epilogue gate then appended ``sample_gate`` and ``has_amax_o`` after those,
    and the SM100 adapter's ``execute`` gained a trailing ``gate`` after
    ``block_table_v``; the block-scaled O then appended ``sample_sf_o`` to the
    constructor and ``sf_o`` to ``execute``.  The pin moves with the tail: every
    addition must sit after the prefix in landing order, so a positional caller
    of any earlier signature still binds where it always did."""
    import inspect

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDsl, SdpaFwdDslSm100

    params = list(inspect.signature(SdpaFwdDsl.__init__).parameters)
    legacy_tail = [
        "paged_table_stride",
        "paged_table_v_stride",
        "thd_stats_padded",
    ]
    assert params[params.index("paged_table_stride") : params.index("thd_stats_padded") + 1] == legacy_tail
    extension_start = params.index("sample_amax_o")
    assert extension_start == params.index("thd_stats_padded") + 1, params[extension_start - 1 : extension_start + 1]
    # Append-only: the ragged-Q decode leg's two plan-time facts follow the block-scale tail.
    assert params[extension_start:] == [
        "sample_amax_o",
        "pv_bf16",
        "stats_log2",
        "sample_gate",
        "has_amax_o",
        "sample_sf_o",
        "sample_scale_o",
        "ragged_divisors",
        "ragged_offsets_int64",
    ], params[extension_start:]
    assert inspect.signature(SdpaFwdDsl.__init__).parameters["stats_log2"].default is False
    assert params.index("thd") + 1 == params.index("max_total_seq_len_q")
    # Both gate parameters default OFF, so every pre-gate call site is untouched.
    sig = inspect.signature(SdpaFwdDsl.__init__).parameters
    assert sig["sample_gate"].default is None and sig["has_amax_o"].default is True
    exec_params = list(inspect.signature(SdpaFwdDslSm100.execute).parameters)
    # Append-only: the ragged-offset operands of the ragged-Q decode leg follow sf_o.
    assert exec_params[-6:] == ["block_table_v", "gate", "sf_o", "ragged_q", "ragged_o", "ragged_lse"], exec_params[-7:]
    assert inspect.signature(SdpaFwdDslSm100.execute).parameters["gate"].default is None
    assert inspect.signature(SdpaFwdDslSm100.execute).parameters["sf_o"].default is None
    assert all(inspect.signature(SdpaFwdDslSm100.execute).parameters[n].default is None for n in ("ragged_q", "ragged_o", "ragged_lse"))


# --- MXFP8 scheduler-policy claims (2026-09-14) -------------------------------
# The d128 and d192x128 MXFP8 kernels thread qh_per_kh / seqlen_kv into every
# decode call site (the LPT_L2 cost-model inputs the shared decode raises
# without), so they honour LPT AND LPT_L2; d256 / d512 pass neither and stay
# NATURAL.  Pure tests first (row + ranking), then the Rubin e2e that is the
# evidence for the claim.


def _mxfp8_facts(**kw):
    import cudnn

    return _f16_facts(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_mxfp8=True, **kw)


@pytest.mark.L0
def test_sm107_mxfp8_advertises_lpt_and_lpt_l2_per_d_shape():
    """INVERTED 2026-09-14 (the row was NATURAL-only): d128 and d192x128 claim
    LPT and LPT_L2, d256 and d512 do not, the row-wide floor stays NATURAL --
    and the claim REACHES the ranking: a causal GQA graph leads with LPT_L2, a
    causal graph without GQA on a few-wave grid leads with LPT, dense stays
    NATURAL.  An accept AND a reject per shape (engine contract, Rule 9)."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines, heuristics

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    assert caps.sched_policies == frozenset({SCHED_NATURAL}), "the row-wide floor must stay NATURAL"
    for shape in ((128, 128), (192, 128)):
        dom = engines.effective_sched_policies(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1]))
        assert dom == frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}), (shape, dom)
        assert heuristics._sched_points(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1], causal=True, h_kv=2)) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL], shape
        assert heuristics._sched_points(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1], causal=True)) == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL], shape
        assert heuristics._sched_points(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1])) == [SCHED_NATURAL], shape
    for shape in ((256, 256), (512, 512)):
        dom = engines.effective_sched_policies(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1]))
        assert dom == frozenset({SCHED_NATURAL}), (shape, dom)
        assert heuristics._sched_points(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1], causal=True)) == [SCHED_NATURAL], shape


@pytest.mark.L0
def test_sm107_mxfp8_sched_knob_is_honored_or_ineligible_per_d_shape():
    """Requesting LPT or LPT_L2 on the MXFP8 row: ACCEPTED at d128 and
    d192x128, DECLINED (typed, naming the knob) at d256 and d512 -- never
    silently downgraded (engine contract, Rule 4)."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    for pol in (SCHED_LPT, SCHED_LPT_L2):
        for shape in ((128, 128), (192, 128)):
            assert engines.mismatch(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1]), SdpaFwdKnobs(sched_policy=pol)) is None, (pol, shape)
        for shape in ((256, 256), (512, 512)):
            why = engines.mismatch(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1]), SdpaFwdKnobs(sched_policy=pol))
            assert why is not None and "sched_policy" in why, (pol, shape)


@pytest.mark.parametrize("d_qk, d_v", [(128, 128), (192, 128)])
@pytest.mark.parametrize("causal, b, hq, hkv, s", [(True, 2, 16, 4, 2048), (False, 1, 8, 2, 1024)])
def test_mxfp8_sched_policies_are_bit_identical_to_natural(d_qk, d_v, causal, b, hq, hkv, s):
    """Rubin e2e behind the MXFP8 (128, 128) and (192, 128) LPT / LPT_L2
    claims: the same block-scaled problem under EVERY policy the row claims
    for the flavor.  The scheduler only reorders whole (batch, head, q-tile)
    work items -- each tile's KV loop is unchanged -- so O and LSE must be
    BIT-IDENTICAL to NATURAL, every cell written (sentinel-filled O, NaN-filled
    LSE), and all within the fp32 oracle bound.  The causal case is a
    multi-wave grid (256 work items over ~106 clusters) so the persistent
    loop walks the remapped order for several rounds."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the sm107 MXFP8 kernels serve cc10.7 only")
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    policies = sorted(engines.effective_sched_policies(caps, _mxfp8_facts(d_qk=d_qk, d_v=d_v, causal=causal)))
    assert SCHED_NATURAL in policies and len(policies) == 3, policies

    torch.manual_seed(0)
    dev = "cuda"
    qf = torch.randn(b, hq, s, d_qk, device=dev) * 0.5
    kf = torch.randn(b, hkv, s, d_qk, device=dev) * 0.5
    vf = torch.randn(b, hkv, s, d_v, device=dev) * 0.5

    def mx(x, h, d, columnwise):
        data_d, _, swz_d, data_s, _, swz_s = quantize_to_mxfp8(x.contiguous(), b, h, s, d, 32, torch.float8_e4m3fn, with_ref=False)
        data, swz = (data_s, swz_s) if columnwise else (data_d, swz_d)
        return data.permute(0, 2, 1, 3).contiguous().transpose(1, 2), swz.contiguous()  # BHSD view over BSHD storage

    q8, sfq = mx(qf, hq, d_qk, False)
    k8, sfk = mx(kf, hkv, d_qk, False)
    v8, sfv = mx(vf, hkv, d_v, True)
    outs, lses = {}, {}
    for pol in policies:
        out = torch.full((b, s, hq, d_v), 1.5e30, device=dev, dtype=torch.bfloat16).transpose(1, 2)  # sentinel: an unclaimed tile stays visible
        lse = torch.full((b, hq, s), float("nan"), device=dev, dtype=torch.float32)
        api = SdpaFwdDslSm100(
            q8, k8, v8, out, lse, scale_softmax=d_qk**-0.5, is_causal=causal, pertensor_fp8=False, dtype_o=torch.bfloat16, sched_policy=pol, cga=2
        )
        assert api.check_support()
        api.compile()
        ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
        api.execute(q8, k8, v8, out, lse_tensor=lse, sf_q=sfq, sf_k=sfk, sf_v=sfv, workspace=ws)
        torch.cuda.synchronize()
        outs[pol], lses[pol] = out.clone(), lse.clone()
    rep = hq // hkv
    logits = qf @ kf.repeat_interleave(rep, 1).transpose(-1, -2) * d_qk**-0.5
    if causal:
        logits = logits.masked_fill(~torch.tril(torch.ones(s, s, dtype=torch.bool, device=dev)), float("-inf"))
    ref = torch.softmax(logits, dim=-1) @ vf.repeat_interleave(rep, 1)
    scale = ref.abs().max().item()
    for pol in policies:
        assert torch.isfinite(outs[pol]).all(), f"policy {pol}: non-finite / unwritten O cells"
        assert torch.isfinite(lses[pol]).all(), f"policy {pol}: unwritten LSE rows"
        err = (outs[pol].float() - ref).abs().max().item()
        assert err <= 0.1 * scale, f"policy {pol}: max err {err} vs fp32 reference (scale {scale})"
    for pol in policies:
        if pol != SCHED_NATURAL:
            assert torch.equal(
                outs[pol], outs[SCHED_NATURAL]
            ), f"policy {pol}: O must be bit-identical to NATURAL -- a different tile walk, the same per-tile math"
            assert torch.equal(lses[pol], lses[SCHED_NATURAL]), f"policy {pol}: LSE must be bit-identical to NATURAL"


@pytest.mark.L0
def test_sm107_causal_ranking_picks_the_policy_by_gqa_and_wave_count():
    """The Rubin causal rule (heuristics._sched_points, 2026-09-14), pinned on
    the two charted layouts.  Perf node, kernel-level, vs NATURAL:
    DSv3 d192x128 H128 (no GQA) -- LPT_L2 -9.7 % at S=2K, LPT +16 % at S=4K,
    LPT -6.5 / -13 / -7.9 % at S=8K/16K/32K, LPT_L2 within 1 % there; Llama
    d128 H64/8 (GQA) -- LPT_L2 +4.2..+7.6 % at every S.  So: without GQA
    LPT_L2 has nothing to group and is never proposed first; LPT leads while
    the grid is at most 24 waves of 2-CTA clusters and NATURAL beyond; with
    GQA the L2-budget rule (LPT_L2) stands.  SM100 keeps its own rule."""
    import cudnn
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import heuristics

    fp8 = dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_fp8=True, causal=True, b=1)
    rubin = _caps("sdpa_fwd_prefill_sm107_fp8")
    dsv3 = dict(h_q=128, h_kv=128, d_qk=192, d_v=128, **fp8)
    # 1 x 128 heads x 8 q-clusters / 106 clusters = 9.7 waves -> LPT; 19 waves at S=4K -> LPT; 39 waves at S=8K -> NATURAL.
    assert heuristics._sched_points(rubin, _f16_facts(s_q=2048, s_kv=2048, **dsv3)) == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL]
    assert heuristics._sched_points(rubin, _f16_facts(s_q=4096, s_kv=4096, **dsv3)) == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL]
    for s in (8192, 16384, 32768):
        assert heuristics._sched_points(rubin, _f16_facts(s_q=s, s_kv=s, **dsv3)) == [SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2], s
    llama = dict(h_q=64, h_kv=8, d_qk=128, d_v=128, **fp8)
    for s in (2048, 8192, 32768):
        assert heuristics._sched_points(rubin, _f16_facts(s_q=s, s_kv=s, **llama)) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL], s
    # The SM100 row is untouched by the Rubin rule: the L2-budget primary stays.
    sm100 = _caps("sdpa_fwd_prefill_sm100_fp8")
    assert heuristics._sched_points(sm100, _f16_facts(s_q=2048, s_kv=2048, device_cc=(10, 0), **dsv3)) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL]
    # SM120 (sm_lo=120) is NOT the Rubin line even though 120 >= 107: a causal no-GQA
    # GeForce-Blackwell graph keeps the SM100/SM120 L2-budget primary too (PR #1059 review).
    sm120 = _caps("sdpa_fwd_prefill_sm120")
    f16 = dict(dtype=cudnn.data_type.HALF, dtype_o=cudnn.data_type.HALF, causal=True, b=1, h_q=32, h_kv=32, d_qk=128, d_v=128)
    assert heuristics._sched_points(sm120, _f16_facts(s_q=2048, s_kv=2048, device_cc=(12, 0), **f16)) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL]


@pytest.mark.L0
@pytest.mark.parametrize("d_qk, d_v", [(128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_mxfp8_stats_is_the_exact_softmax_lse(d_qk, d_v, causal):
    """Rubin e2e for the MXFP8 d128 / d192x128 / d256 kernels (row-sum-in-MMA since
    #1059): the PUBLISHED Stats is the fp32 log-sum-exp of the block-scaled
    problem the kernel saw, not the log of the quantized-P sum that
    normalizes O -- cuDNN's mxfp8 backward recomputes P = exp(S - Stats), and
    the fp8 twin lost a dK row to exactly that (test_mhas_v2 fp8_bwd_ragged
    test31).  (1) LSE within 1e-4 of the exact value from the DEQUANTIZED
    inputs; (2) O bit-identical with and without Stats.  The (256, 256) row is
    the forward the d=256 MXFP8 BACKWARD (`sdpa_bwd_sm107_mxfp8`) recomputes P
    from (the exact-LSE contract pinned in ``test_sdpa_bwd_mxfp8_sm107.py``)."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the sm107 MXFP8 kernels serve cc10.7 only")
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    torch.manual_seed(0)
    b, hq, hkv, s = 2, 8, 2, 1024
    dev = "cuda"
    qf = torch.randn(b, hq, s, d_qk, device=dev) * 0.5
    kf = torch.randn(b, hkv, s, d_qk, device=dev) * 0.5
    vf = torch.randn(b, hkv, s, d_v, device=dev) * 0.5

    def mx(x, h, d, columnwise):
        data_d, sf_d, swz_d, data_s, sf_s, swz_s = quantize_to_mxfp8(x.contiguous(), b, h, s, d, 32, torch.float8_e4m3fn, with_ref=True)
        data, sf, swz = (data_s, sf_s, swz_s) if columnwise else (data_d, sf_d, swz_d)
        # sf_*_ref are the per-element fp32 DEQUANT SCALES [b, h, s, d]; the value the kernel sees is data * scale.
        # fp64 so the reference logits below are exact: the DLFW containers run fp32 matmul in TF32, which a
        # causal row with one valid column turns into a 1e-4-class LSE error (test_fp8_stats_is_the_exact_softmax_lse).
        deq = data.double().reshape(b, h, s, d) * sf.double().reshape(b, h, s, d)
        return data.permute(0, 2, 1, 3).contiguous().transpose(1, 2), deq, swz.contiguous()

    q8, q_deq, sfq = mx(qf, hq, d_qk, False)
    k8, k_deq, sfk = mx(kf, hkv, d_qk, False)
    v8, _, sfv = mx(vf, hkv, d_v, True)
    outs = {}
    lse = torch.full((b, hq, s), float("nan"), device=dev, dtype=torch.float32)
    for with_stats in (True, False):
        out = torch.full((b, s, hq, d_v), 1.5e30, device=dev, dtype=torch.bfloat16).transpose(1, 2)
        api = SdpaFwdDslSm100(
            q8,
            k8,
            v8,
            out,
            lse if with_stats else None,
            scale_softmax=d_qk**-0.5,
            is_causal=causal,
            pertensor_fp8=False,
            dtype_o=torch.bfloat16,
            cga=1 if d_qk == 256 else 2,
        )
        assert api.check_support()
        api.compile()
        ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
        api.execute(q8, k8, v8, out, lse_tensor=lse if with_stats else None, sf_q=sfq, sf_k=sfk, sf_v=sfv, workspace=ws)
        torch.cuda.synchronize()
        outs[with_stats] = out.clone()
    assert torch.equal(outs[True], outs[False]), "O must not depend on whether Stats is requested (the MMA row-sum normalizes O in both specializations)"
    rep = hq // hkv
    logits = q_deq @ k_deq.repeat_interleave(rep, 1).transpose(-1, -2) * d_qk**-0.5
    if causal:
        logits = logits.masked_fill(~torch.tril(torch.ones(s, s, dtype=torch.bool, device=dev)), float("-inf"))
    lse_ref = torch.logsumexp(logits, dim=-1).float()
    assert torch.isfinite(lse).all(), "unwritten LSE rows"
    err = (lse - lse_ref).abs()
    assert (
        err.max().item() <= 1e-4
    ), f"Stats is not the exact log-sum-exp: max |dLSE| {err.max().item():.3e}, rms {err.pow(2).mean().sqrt().item():.3e} (quantized-sum LSE reads ~1e-3..1e-2)"


@pytest.mark.parametrize("kind,load_kw", _DTYPE_FAMILIES, ids=[k for k, _ in _DTYPE_FAMILIES])
@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_stats_log2_specializes_every_dtype_and_flavor(flavor, kind, load_kw):
    natural = _load(flavor, rubin=True, stats_log2=False, **load_kw)
    log2 = _load(flavor, rubin=True, stats_log2=True, **load_kw)
    assert natural is not log2
    assert natural.CFG.STATS_LOG2 == 0
    assert log2.CFG.STATS_LOG2 == 1


# ============================================================================
# Fused epilogue gate -- O := O * sigmoid(G) (PR-A, 2026-09-15)
#
# The Rubin d256 f16/bf16 and per-tensor FP8 kernels carry the gate behind
# ``TemplateParams.epilogue_gate`` (a module-cache key) / ``CFG.EPILOGUE_GATE``
# (const_expr seams).  Four enforcement points share ONE constant,
# ``config_sm107.SM107_EPILOGUE_GATE_SHAPES``: the two engine rows
# (``epilogue_gate_d_shapes``), the standalone adapter's rule-8b twin in
# ``SdpaFwdDslSm100.check_support``, the config validator
# (``_EPILOGUE_GATE_FLAVORS``) and the gated-attention block's geometry pin.
# The tests below walk them in that order -- rows, config/template, adapter --
# and end with the Rubin e2e that is the evidence behind every claim.
# ============================================================================

# Three rows, one per dtype family of the d256 flavor: f16/bf16, per-tensor FP8
# (PR-A) and block-scale MXFP8 (PR-B slice S7).  Order is load-bearing for the
# `f16, fp8 = ...` destructurings below; the MXFP8 row is indexed explicitly.
_GATE_ROWS = ("sdpa_fwd_prefill_sm107", "sdpa_fwd_prefill_sm107_fp8", "sdpa_fwd_prefill_sm107_mxfp8")
_D256 = (256, 256)
_GATE_SEAMS = (
    "cfg",
    "kernel_params",
    "smem",
    "bars",
    "init",
    "dispatch_corr",
    "dispatch_tmaldg",
    "tmaldg_state",
    "tmaldg_issue",
    "corr_state",
    "corr_prologue",
    "corr_chunk",
    "corr_release",
    "host_desc",
    "host_launch",
    "compile",
)
# The fork-era module knobs the production kernels must NOT carry (engine
# contract rule 5: no module-level knobs; the block drives the adapter).
_RETIRED_GATE_KNOBS = ("GATE_SOURCE", "GATE_ISSUE", "GATE_MATH", "AMAX_O", "KEEP_SASS")
# cta_mma=1 is what the adapter pins for FP8 d256 (supported_cgas_for((256, 256), fp8=True) == (1,)).
_FP8_LOAD_KW = dict(fp8=True, pertensor=True, dtype_qkv=_E4M3, dtype_o=_BF16_OUT, cta_mma=1)
# ...and for the block-scale MXFP8 d256 flavor (same cga1 pin; bf16 O is the unfused block's mode, e4m3 O the fused one).
_MXFP8_LOAD_KW = dict(fp8=True, pertensor=False, dtype_qkv=_E4M3, dtype_o=_BF16_OUT, cta_mma=1)
_QUANT_LOAD_KWS = (_FP8_LOAD_KW, _MXFP8_LOAD_KW)


def _gate_facts(**kw):
    """f16 facts for a GATED (256, 256) graph; the gate dtype defaults to Q's."""
    import cudnn

    kw.setdefault("d_qk", 256)
    kw.setdefault("d_v", 256)
    kw.setdefault("has_epilogue_gate", True)
    kw.setdefault("epilogue_gate_dtype", kw.get("dtype", cudnn.data_type.HALF))
    return _f16_facts(**kw)


def _fp8_gate_facts(**kw):
    """Per-tensor FP8 facts for a GATED (256, 256) graph with a bf16 G."""
    import cudnn

    kw.setdefault("dtype", cudnn.data_type.FP8_E4M3)
    kw.setdefault("dtype_o", cudnn.data_type.BFLOAT16)
    kw.setdefault("epilogue_gate_dtype", cudnn.data_type.BFLOAT16)
    return _gate_facts(is_fp8=True, **kw)


def _fp8_ungated_kw(**kw):
    """The UNGATED per-tensor FP8 (256, 256) facts kwargs -- the control next to ``_fp8_gate_facts``."""
    import cudnn

    base = dict(is_fp8=True, d_qk=256, d_v=256, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16)
    base.update(kw)
    return base


def _gate_kernel_modules():
    """(f16, fp8) d256 Rubin modules loaded with the gate ON."""
    return _load(_D256, rubin=True, epilogue_gate=True), _load(_D256, rubin=True, epilogue_gate=True, **_FP8_LOAD_KW)


def _mxfp8_gate_kernel_module(**kw):
    """The block-scale MXFP8 d256 Rubin module loaded with the gate ON (kw overrides the load kwargs)."""
    return _load(_D256, rubin=True, epilogue_gate=True, **{**_MXFP8_LOAD_KW, **kw})


def _all_gate_kernel_modules():
    """(f16, fp8, mxfp8) -- every d256 body that carries the seams."""
    return (*_gate_kernel_modules(), _mxfp8_gate_kernel_module())


def _mxfp8_gate_facts(**kw):
    """MXFP8 facts for a GATED (256, 256) graph with a bf16 G (the only gate dtype the row lists)."""
    import cudnn

    kw.setdefault("d_qk", 256)
    kw.setdefault("d_v", 256)
    kw.setdefault("has_epilogue_gate", True)
    kw.setdefault("epilogue_gate_dtype", cudnn.data_type.BFLOAT16)
    kw.setdefault("dtype", cudnn.data_type.FP8_E4M3)
    kw.setdefault("dtype_o", cudnn.data_type.BFLOAT16)
    return _f16_facts(is_mxfp8=True, **kw)


# --- Rows / mismatch -----------------------------------------------------------


def test_sm107_gate_rows_claim_exactly_d256():
    """All three Rubin d256 rows (f16/bf16, per-tensor FP8, block-scale MXFP8)
    claim the gate at EXACTLY (256, 256) -- and nothing else does.  The exact-dims rule is deliberate (plan S7 Q6): a d=200 graph rides the
    (256, 256) envelope for plain attention, but the gate tile would multiply the
    zero-padded columns too and nobody has validated a padded G, so the envelope
    flavor is declined with the gate's own reason rather than silently served."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.config_sm107 import SM107_EPILOGUE_GATE_SHAPES

    assert SM107_EPILOGUE_GATE_SHAPES == frozenset({_D256})
    for row in _GATE_ROWS:
        caps = _caps(row)
        assert caps.epilogue_gate is True, row
        assert caps.epilogue_gate_d_shapes == frozenset({_D256}), row
        assert caps.epilogue_gate_d_shapes is SM107_EPILOGUE_GATE_SHAPES, f"{row}: the row must consume the shared constant, not a copy"
    for spec in engines.ENGINE_SPECS:
        if spec.name not in _GATE_ROWS:
            assert spec.capabilities.epilogue_gate is False, spec.name
            assert spec.capabilities.epilogue_gate_d_shapes is None, spec.name
    # The gate is a FEATURE of the Rubin rows, never a row of its own -- a claim
    # unrelated rows landing cannot falsify (a bare spec count could not say it).
    assert {s.name for s in engines.ENGINE_SPECS if s.capabilities.epilogue_gate} == set(_GATE_ROWS)

    f16, fp8, mxfp8 = _caps(_GATE_ROWS[0]), _caps(_GATE_ROWS[1]), _caps(_GATE_ROWS[2])
    assert engines.mismatch(f16, _gate_facts()) is None
    assert engines.mismatch(fp8, _fp8_gate_facts()) is None
    assert engines.mismatch(mxfp8, _mxfp8_gate_facts()) is None, "PR-B: the MXFP8 row serves the tail at (256, 256) with a bf16 G"
    # Every other flavor -- and the d=200 envelope ride -- declines by the gate's reason.
    for d_qk, d_v in ((128, 128), (192, 128), (512, 512), (200, 200)):
        why = engines.mismatch(f16, _gate_facts(d_qk=d_qk, d_v=d_v))
        assert why is not None and "epilogue gate" in why, (d_qk, d_v, why)
    for d_qk, d_v in ((128, 128), (192, 128), (512, 512)):
        why = engines.mismatch(fp8, _fp8_gate_facts(d_qk=d_qk, d_v=d_v))
        assert why is not None and "epilogue gate" in why, (d_qk, d_v, why)
        why = engines.mismatch(mxfp8, _mxfp8_gate_facts(d_qk=d_qk, d_v=d_v))
        assert why is not None and "epilogue gate" in why, (d_qk, d_v, why)
    # The MXFP8 row's UNGATED (256, 256) graph is untouched by the new fields.
    assert engines.mismatch(mxfp8, _mxfp8_facts(d_qk=256, d_v=256)) is None
    # An UNGATED graph is untouched by the new fields on every row.
    assert engines.mismatch(f16, _f16_facts(d_qk=256, d_v=256)) is None
    assert engines.mismatch(f16, _f16_facts(d_qk=128, d_v=128)) is None


def test_sm107_gate_accepts_every_dtype_member():
    """A frozenset field is one claim PER MEMBER (contract rule 9).  f16 row:
    HALF and BFLOAT16, gate dtype == Q's.  FP8 row: every input dtype x every O
    dtype the row lists, with the bf16 gate the kernel stages
    (``GATE_STORAGE_DTYPE = cutlass.BFloat16``); a HALF gate on that row is an
    asserted decline naming the axis."""
    import cudnn
    from cudnn.sdpa.fwd import engines

    f16 = _caps(_GATE_ROWS[0])
    assert f16.epilogue_gate_dtypes is None, "None = G must equal Q's dtype"
    for dt in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
        assert engines.mismatch(f16, _gate_facts(dtype=dt, dtype_o=dt, epilogue_gate_dtype=dt)) is None, dt
    # ...and a gate of the OTHER half dtype declines, naming the gate dtype.
    why = engines.mismatch(f16, _gate_facts(dtype=cudnn.data_type.HALF, dtype_o=cudnn.data_type.HALF, epilogue_gate_dtype=cudnn.data_type.BFLOAT16))
    assert why is not None and "gate dtype" in why, why

    fp8 = _caps(_GATE_ROWS[1])
    assert fp8.epilogue_gate_dtypes == frozenset({cudnn.data_type.BFLOAT16})
    assert fp8.dtypes == frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2})
    assert fp8.out_dtypes == frozenset(
        {cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1}
    )
    # FP4_E2M1 is listed for the block-scaled O epilogue (sf_o), which is its own
    # O store: it is never gated, so the gate matrix runs over the other members.
    for dt in sorted(fp8.dtypes, key=int):
        for dto in sorted(fp8.out_dtypes - {cudnn.data_type.FP4_E2M1}, key=int):
            assert engines.mismatch(fp8, _fp8_gate_facts(dtype=dt, dtype_o=dto)) is None, (dt, dto)
    why = engines.mismatch(fp8, _fp8_gate_facts(epilogue_gate_dtype=cudnn.data_type.HALF))
    assert why is not None and "gate dtype" in why, why
    # ...and the two epilogues decline each other, each naming the block-scaled O.
    why = engines.mismatch(fp8, _fp8_gate_facts(dtype_o=cudnn.data_type.FP4_E2M1))
    assert why is not None and "block-scaled" in why, why
    why = engines.mismatch(fp8, _fp8_gate_facts(dtype_o=cudnn.data_type.FP8_E4M3, o_block_scale=32))
    assert why is not None and "epilogue gate" in why, why

    # The MXFP8 row (PR-B): the same per-member claims -- every input dtype x
    # every O dtype with the bf16 G its kernel stages (GATE_STORAGE_DTYPE); a
    # HALF gate is an asserted decline naming the axis.  (INVERTED 2026-09-15:
    # this row declined the tail outright under PR-A.)
    mxfp8 = _caps(_GATE_ROWS[2])
    assert mxfp8.epilogue_gate_dtypes == frozenset({cudnn.data_type.BFLOAT16})
    assert mxfp8.dtypes == frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2})
    assert mxfp8.out_dtypes == frozenset(
        {cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1}
    )
    # FP4_E2M1 is listed for the block-scaled O epilogue (sf_o), its own O store:
    # never gated, so the gate matrix runs over the other members.
    for dt in sorted(mxfp8.dtypes, key=int):
        for dto in sorted(mxfp8.out_dtypes - {cudnn.data_type.FP4_E2M1}, key=int):
            assert engines.mismatch(mxfp8, _mxfp8_gate_facts(dtype=dt, dtype_o=dto)) is None, (dt, dto)
    why = engines.mismatch(mxfp8, _mxfp8_gate_facts(epilogue_gate_dtype=cudnn.data_type.HALF))
    assert why is not None and "gate dtype" in why, why
    # ...and the two epilogues decline each other on this row too.
    why = engines.mismatch(mxfp8, _mxfp8_gate_facts(dtype_o=cudnn.data_type.FP4_E2M1))
    assert why is not None and "block-scaled" in why, why
    why = engines.mismatch(mxfp8, _mxfp8_gate_facts(dtype_o=cudnn.data_type.FP8_E4M3, o_block_scale=32))
    assert why is not None and "epilogue gate" in why, why
    why = engines.mismatch(mxfp8, _mxfp8_gate_facts(epilogue_gate_dtype=cudnn.data_type.FP8_E4M3))
    assert why is not None and "gate dtype" in why, why


def test_sm107_gate_declines_the_interactions():
    """Gate x THD is REACHABLE (the f16 row serves THD at d256), gate x paged
    is REACHABLE (paged is wired on d256), so both must be declined by the gate
    block itself -- and the knob interactions (split, PackGQA) by their knob
    blocks. A flavor may serve both features on different paths; test the
    requested combination instead of assuming their flavor sets are disjoint."""
    import cudnn
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    f16, fp8 = _caps(_GATE_ROWS[0]), _caps(_GATE_ROWS[1])
    assert engines.mismatch(f16, _f16_facts(d_qk=256, d_v=256, thd=True, padded=True)) is None, "THD at d256 is served -- the interaction is live"
    why = engines.mismatch(f16, _gate_facts(thd=True, padded=True))
    assert why is not None and "dense-only" in why, why
    paged = dict(has_paged_kv=True, padded=True, page_size=128)
    # The half row serves ungated paged THD; the gate block must decline
    # the interaction before paged layout admission. FP8 remains nonpaged.
    assert f16.paged_kv and not fp8.paged_kv
    assert engines.mismatch(f16, _f16_facts(d_qk=256, d_v=256, thd=True, **paged)) is None
    why = engines.mismatch(f16, _gate_facts(**paged))
    assert why is not None and "paged" in why and "gate" in why, why
    # Knob interactions: a split or packed plan can never carry the gate.
    why = engines.mismatch(fp8, _fp8_gate_facts(), SdpaFwdKnobs(split_kv=2))
    assert why is not None and "gate" in why, why
    # PackGQA on the REAL fp8 row is declined by its flavor table first (pack_gqa_d_shapes = {(128, 128)}); the
    # gate clause is what returns on a row that WOULD pack a d256 graph, so pin it on that synthetic row.
    why = engines.mismatch(fp8, _fp8_gate_facts(h_kv=2), SdpaFwdKnobs(pack_gqa=True))
    assert why is not None, why
    packs_d256 = dataclasses.replace(fp8, pack_gqas=frozenset({False, True}), pack_gqa_d_shapes=None)
    assert engines.mismatch(packs_d256, _f16_facts(**_fp8_ungated_kw(h_kv=2)), SdpaFwdKnobs(pack_gqa=True)) is None, "the control must pack"
    why = engines.mismatch(packs_d256, _fp8_gate_facts(h_kv=2), SdpaFwdKnobs(pack_gqa=True))
    assert why is not None and "gate" in why, why
    for row, gate_facts in zip(_GATE_ROWS, (_gate_facts, _fp8_gate_facts, _mxfp8_gate_facts), strict=True):
        caps = _caps(row)
        facts = gate_facts(h_kv=2)
        assert engines.mismatch(caps, facts) is None, row
        for knobs in (SdpaFwdKnobs(split_kv=2), SdpaFwdKnobs(pack_gqa=True)):
            assert engines.mismatch(caps, facts, knobs) is not None, (row, knobs)
    # The ONE exception, half row only: a DECODE-shaped gated graph (S_q x G <= 16 packed rows) splits on the
    # d256 decode tile, whose combine applies the gate -- packed or not, paged or dense; its UNSPLIT form keeps
    # the prefill kernel (dense only, unpacked).  test_sdpa_fwd_decode_d256_sm107 carries the full contract.
    decode = _gate_facts(
        h_kv=2, s_q=1, s_kv=4096, dtype=cudnn.data_type.BFLOAT16, epilogue_gate_dtype=cudnn.data_type.BFLOAT16
    )  # dense UNPADDED: the one dense form a split rides
    assert engines.mismatch(f16, decode, SdpaFwdKnobs(split_kv=2, pack_gqa=True)) is None
    assert engines.mismatch(f16, decode, SdpaFwdKnobs(split_kv=2)) is None
    assert engines.mismatch(f16, decode, SdpaFwdKnobs(split_kv=1)) is None  # the prefill kernel's fused epilogue
    why = engines.mismatch(f16, decode, SdpaFwdKnobs(split_kv=1, pack_gqa=True))
    assert why is not None and "decode tile" in why, why
    decode_paged = _gate_facts(h_kv=2, s_q=1, s_kv=4096, dtype=cudnn.data_type.BFLOAT16, epilogue_gate_dtype=cudnn.data_type.BFLOAT16, **paged)
    assert engines.mismatch(f16, decode_paged) is None and engines.mismatch(f16, decode_paged, SdpaFwdKnobs(split_kv=2, pack_gqa=True)) is None
    why = engines.mismatch(f16, decode_paged, SdpaFwdKnobs(split_kv=1))
    assert why is not None and "paged" in why and "gate" in why, why
    assert engines.mismatch(fp8, _fp8_gate_facts(h_kv=2, s_q=1, s_kv=4096), SdpaFwdKnobs(split_kv=2)) is not None, "the quantized rows keep the gate unsplit"
    # A broadcast G / an undeclared O_v are DECLINES (legal graphs for the backend), never facts.invalid.
    why = engines.mismatch(f16, _gate_facts(epilogue_gate_shape_ok=False))
    assert why is not None and "shape" in why, why
    # A G the kernel cannot TMA-load zero-copy (head-major / unaligned strides) is the row's twin of the
    # adapter's "TMA-expressible" ValueError (rule 8b: both halves, or the row admits what the adapter rejects).
    why = engines.mismatch(f16, _gate_facts(epilogue_gate_layout_ok=False))
    assert why is not None and "zero-copy" in why, why
    why = engines.mismatch(f16, _gate_facts(sdpa_o_virtual_declared=False))
    assert why is not None and "set_dim" in why, why
    for ok in (None, cudnn.data_type.FLOAT, cudnn.data_type.HALF):
        assert engines.mismatch(f16, _gate_facts(sdpa_o_virtual_dtype=ok)) is None, ok
    why = engines.mismatch(f16, _gate_facts(sdpa_o_virtual_dtype=cudnn.data_type.BFLOAT16))
    assert why is not None and "virtual O" in why, why


def test_epilogue_gate_is_not_a_knob():
    """The gate is a graph FACT (the tail is there or it is not), never a tuning
    axis: the knob vocabulary has no such field and the heuristics cannot
    propose it (contract rule 4)."""
    from cudnn.sdpa.fwd import engines, heuristics

    assert "epilogue_gate" not in engines.SdpaFwdKnobs.__dataclass_fields__
    assert not hasattr(heuristics, "_epilogue_gate_points")
    # ...but it IS a template-params field (the module-cache key) with the gate OFF by default.
    assert TemplateParams.__dataclass_fields__["epilogue_gate"].default is False


def test_sm107_gate_row_matches_the_adapter_twin():
    """Rule 8b': a row decline has a STANDALONE-WRAPPER twin, and the two must
    read ONE constant.  Iterate ROWS (not one named row -- the d_envelope_floors
    lesson), check identity with the shared object, and pin that
    ``check_support`` CONSUMES it rather than a private literal."""
    import inspect

    from cudnn.sdpa.fwd import api_dsl, engines
    from cudnn.sdpa.fwd.config_sm107 import SM107_EPILOGUE_GATE_SHAPES

    gate_rows = [s for s in engines.ENGINE_SPECS if s.capabilities.epilogue_gate]
    assert {s.name for s in gate_rows} == set(_GATE_ROWS)
    for spec in gate_rows:
        assert spec.capabilities.epilogue_gate_d_shapes is SM107_EPILOGUE_GATE_SHAPES, spec.name
        assert spec.capabilities.epilogue_gate_d_shapes <= spec.capabilities.d_shapes, spec.name
    assert api_dsl._SM107_EPILOGUE_GATE_SHAPES is SM107_EPILOGUE_GATE_SHAPES
    src = inspect.getsource(api_dsl.SdpaFwdDslSm100.check_support)
    assert "_SM107_EPILOGUE_GATE_SHAPES" in src, "check_support no longer consumes the shared gate-shape constant"
    assert "gate_desc" in src, "check_support does not look at the gate descriptor at all"
    # The config validator is the third consumer, keyed by flavor string.
    from cudnn.sdpa.fwd import config_sm107

    assert config_sm107._EPILOGUE_GATE_FLAVORS == frozenset({"sm107 d256", "sm107 d256 mxfp8"})
    assert "SM107_EPILOGUE_GATE_SHAPES" in config_sm107.__all__


# --- Config / template -----------------------------------------------------------


def test_sm107_gate_template_loads_for_d256_only():
    """The config backstop behind the rows: a gated (256, 256) module loads on
    BOTH dtype families with ``CFG.EPILOGUE_GATE == 1``; every other flavor
    raises at config build (a body without the seams would trace an ungated
    epilogue and silently return O instead of O * sigmoid(G))."""
    f16, fp8 = _gate_kernel_modules()
    assert f16.CFG.EPILOGUE_GATE == 1 and fp8.CFG.EPILOGUE_GATE == 1
    assert f16.CFG.GATE_BPE == 2 and fp8.CFG.GATE_BPE == 2, "the gate is 2 B/elem on both kernels (Q dtype / bf16)"
    assert "sm107" in f16.__name__ and "sm107" in fp8.__name__
    for flavor in ((128, 128), (192, 128), (512, 512)):
        with pytest.raises(ValueError, match="epilogue_gate"):
            _load(flavor, rubin=True, epilogue_gate=True)
        with pytest.raises(ValueError, match="epilogue_gate"):
            _load(flavor, rubin=True, epilogue_gate=True, **_FP8_LOAD_KW)
    # MXFP8 d256 (PR-B): the block-scale body carries the seams too -- it loads
    # gated, on both O dtypes; every OTHER mxfp8 flavor is refused by the config.
    for dto in (_E4M3, _BF16_OUT):
        mx = _mxfp8_gate_kernel_module(dtype_o=dto)
        assert mx.CFG.EPILOGUE_GATE == 1 and mx.CFG.GATE_BPE == 2 and "mxfp8" in mx.__name__, mx.__name__
    for flavor in ((128, 128), (192, 128), (512, 512)):
        with pytest.raises(ValueError, match="epilogue_gate"):
            _load(flavor, rubin=True, epilogue_gate=True, **_MXFP8_LOAD_KW)
    # The SM100 line has no gated body at all.
    with pytest.raises(ValueError, match="epilogue_gate"):
        _load(_D256, rubin=False, epilogue_gate=True)
    # The feature interactions are refused by the config too (backstop of the rows / adapter twin).
    for bad in (
        dict(thd_varlen=True, seq_kv_lens_present=True),
        dict(split_kv=2),
        dict(pack_gqa=True, qh_per_kh=4),
        dict(paged_kv=True, page_size=128, seq_kv_lens_present=True),
    ):
        # split_kv > 1 is refused by the (earlier) SplitHelpers guard on this flavor; the rest by the gate rule.
        with pytest.raises(ValueError, match="epilogue_gate|split_kv > 1 is not wired"):
            _load(_D256, rubin=True, epilogue_gate=True, **bad)
        with pytest.raises(ValueError, match="epilogue_gate|split_kv > 1 is not wired"):
            _load(_D256, rubin=True, epilogue_gate=True, **{**_MXFP8_LOAD_KW, **bad})


def test_sm107_gate_off_is_the_default_module():
    """Gate on/off are TWO coexisting specializations of one template, and
    gate-off is the module every pre-gate caller already loads: same
    ``TemplateParams``, same module object, same source digest."""
    assert TemplateParams() == TemplateParams(epilogue_gate=False)
    for kw in ({}, *_QUANT_LOAD_KWS):
        off = _load(_D256, rubin=True, **kw)
        assert off is _load(_D256, rubin=True, epilogue_gate=False, **kw)
        assert off.CFG.EPILOGUE_GATE == 0
        on = _load(_D256, rubin=True, epilogue_gate=True, **kw)
        assert on is not off
        assert (
            on.FROST_SOURCE_DIGEST != off.FROST_SOURCE_DIGEST
        ), "the digest must key the gate so the compiled cache cannot hand a gated artifact to an ungated caller"
        assert on.__file__ == off.__file__, "one template file, two specializations"


def test_sm107_gate_smem_tally():
    """The gate adds ONE 64 KiB staging tile (TILES_Q x TILE_M x TILE_O x 2 B,
    NOT divided by CTA_MMA -- each CTA of a cga2 pair gates its own full-width
    128 q rows).  Pure arithmetic on the shared tally, then the validator: depth
    2 fits at every configuration the block and engine path run, depth 3 does
    not, and the raise NAMES the gate term so a reader sees why a depth that
    fits ungated no longer does."""
    from dataclasses import dataclass

    from cudnn.sdpa.fwd.config_sm107 import SMEM_USABLE_BYTES, make_cfg_d256, sdpa_smem_bytes

    KIB = 1024
    fixed = 2 * KIB  # _SMEM_FIXED_OVERHEAD: barriers + scheduler ring + tmem ptr

    def tally(*, cta_mma, stages_kv, bpe_qkv, bpe_o, gate_bpe):
        return sdpa_smem_bytes(128, 128, 256, 256, 1, stages_kv, cta_mma, bpe_qkv, bpe_o, qo_alias=True, gate_bpe=gate_bpe) + fixed

    # Gate-off numbers are today's (194 / 162 KiB); the gate term is +64 KiB.
    assert tally(cta_mma=2, stages_kv=2, bpe_qkv=2, bpe_o=2, gate_bpe=0) == 194 * KIB
    assert tally(cta_mma=1, stages_kv=2, bpe_qkv=1, bpe_o=1, gate_bpe=0) == 162 * KIB
    assert tally(cta_mma=2, stages_kv=2, bpe_qkv=2, bpe_o=2, gate_bpe=2) == 258 * KIB  # d256 f16 cga2 (block + engine path)
    assert tally(cta_mma=1, stages_kv=2, bpe_qkv=1, bpe_o=1, gate_bpe=2) == 226 * KIB  # d256 fp8 e4m3-O (block's fully-fused mode)
    assert tally(cta_mma=1, stages_kv=2, bpe_qkv=1, bpe_o=2, gate_bpe=2) == 258 * KIB  # d256 fp8 bf16-O (unfused fp8)
    assert tally(cta_mma=2, stages_kv=3, bpe_qkv=2, bpe_o=2, gate_bpe=2) == 322 * KIB > SMEM_USABLE_BYTES
    assert tally(cta_mma=1, stages_kv=3, bpe_qkv=1, bpe_o=2, gate_bpe=2) == 322 * KIB > SMEM_USABLE_BYTES
    # ...and the same shapes carry the 4-CTA-wide gate term a d128 cga2 flavor would need (TILES_Q=2): not claimed.
    assert sdpa_smem_bytes(128, 128, 128, 128, 2, 4, 2, 2, 2, qo_alias=False, gate_bpe=2) + fixed == 322 * KIB

    @dataclass(frozen=True)
    class _ParamsWithStagesKv(TemplateParams):
        stages_kv: int = None

    cfg, _ = make_cfg_d256(_ParamsWithStagesKv(stages_kv=2, epilogue_gate=True))
    assert cfg.EPILOGUE_GATE == 1 and cfg.STAGES_KV == 2
    cfg, _ = make_cfg_d256(_ParamsWithStagesKv(stages_kv=3, epilogue_gate=False))
    assert cfg.STAGES_KV == 3, "depth 3 still fits UNGATED (290 KiB)"
    with pytest.raises(ValueError, match="GATE") as ei:
        make_cfg_d256(_ParamsWithStagesKv(stages_kv=3, epilogue_gate=True))
    assert "322 KiB" in str(ei.value) and "EPILOGUE_GATE=1" in str(ei.value), str(ei.value)
    # FP8 twins: e4m3-O and bf16-O at depth 2 fit; bf16-O at depth 3 does not.
    for dto in (_E4M3, _BF16_OUT):
        cfg, _ = make_cfg_d256(_ParamsWithStagesKv(stages_kv=2, epilogue_gate=True, dtype_qkv=_E4M3, dtype_o=dto, cta_mma=1))
        assert cfg.EPILOGUE_GATE == 1
    with pytest.raises(ValueError, match="GATE"):
        make_cfg_d256(_ParamsWithStagesKv(stages_kv=3, epilogue_gate=True, dtype_qkv=_E4M3, dtype_o=_BF16_OUT, cta_mma=1))


def test_sm107_gate_desc_version_unchanged():
    """The gate tile is allocated AFTER every MMA operand (sQO, sK, sV) and is
    never an MMA / UTCCP operand itself, so it cannot move an operand across
    the 256 KiB version-0 tcgen05 descriptor window: DESC_VERSION and its
    STAGES_KV derivation are identical with the gate on and off, on both
    kernels.  (The FP8 sibling now derives it too -- its literal 0 was one
    depth away from the same wrap.)"""
    from dataclasses import replace

    for kw in ({}, _FP8_LOAD_KW):
        off = _load(_D256, rubin=True, **kw)
        on = _load(_D256, rubin=True, epilogue_gate=True, **kw)
        assert on.DESC_VERSION == off.DESC_VERSION == 0, (kw, on.DESC_VERSION, off.DESC_VERSION)
        assert callable(getattr(on, "_needs_desc_v1", None)), f"{on.__name__}: DESC_VERSION must be DERIVED, not a literal"
        for depth in (2, 3, 4):
            want = int(off._needs_desc_v1(replace(off.CFG, STAGES_KV=depth)))
            assert int(on._needs_desc_v1(replace(on.CFG, STAGES_KV=depth))) == want, (kw, depth)
            # ...and both agree with the layout: the LAST V stage's START (Q(u)O byte-max + K ring + the
            # preceding V stages) crossing the 14-bit window.  The gate tile is not in this sum.
            c = on.CFG
            qo = max(c.TILE_M * c.TILE_K * c.BPE, c.TILE_M * c.TILE_O * c.BPE_O)
            k_stage = (c.TILE_N * c.TILE_K // c.CTA_MMA) * c.BPE
            v_stage = (c.TILE_O * c.TILE_N // c.CTA_MMA) * c.BPE
            assert want == int(qo + depth * k_stage + (depth - 1) * v_stage >= 256 * 1024), (kw, depth, want)
        # The f16 cga2 layout crosses at depth 4 exactly as the existing STAGES_KV test pins; fp8 bf16-O cga1 too.
        assert int(on._needs_desc_v1(replace(on.CFG, STAGES_KV=4))) == 1, kw


def test_sm107_gate_kernel_signatures_are_append_only():
    """The f16 kernel is an explicit pointer/int host: the gate is a MODULE specialization
    (TemplateParams.epilogue_gate -> CFG.EPILOGUE_GATE), its slot (``gate_ptr`` + runtime
    ``gate_strides``) is always declared and compile() keys only what specializes the trace.
    Both quantized families use prepared pointer hosts with runtime gate strides.
    Their internal tensor host signatures remain append-only: ``gate_tensor``
    immediately follows ``stream``; prepared flags may follow.  Ahead of ``stream``
    the slot ORDER is the pointer host's ABI: ``_mxfp8_host._launch`` passes
    everything up to ``seq_q_lens_addr`` positionally and, under ``thd_slots``,
    the three THD length slots right after it (``stream=`` and the flags go by
    keyword), so those slots are pinned by position on both Rubin d256
    quantized kernels and on the SM100 MXFP8 twin the same host drives."""
    import inspect

    f16, fp8, mxfp8 = _all_gate_kernel_modules()
    assert f16.EXPLICIT_ABI is True
    f16_c = inspect.signature(f16.compile).parameters
    assert "gate_stride" not in f16_c and "lse_stride" not in f16_c and "b" not in f16_c, list(f16_c)
    f16_host = inspect.signature(f16._host).parameters
    assert "gate_ptr" in f16_host and "gate_strides" in f16_host, list(f16_host)[-6:]
    assert list(f16_host)[-1] == "stream"

    fp8_c = inspect.signature(fp8.compile_prepared).parameters
    assert not {"b", "qh", "lse_stride", "gate_stride"} & fp8_c.keys()
    assert fp8_c["has_amax"].default is True
    fp8_host = inspect.signature(fp8._host_prepared).parameters
    assert {"gate_ptr", "gate_strides"} <= fp8_host.keys()
    assert list(fp8_host)[-1] == "stream"
    mx_c = inspect.signature(mxfp8.compile_prepared).parameters
    assert not {"b", "qh", "lse_stride", "gate_stride"} & mx_c.keys()
    assert mx_c["has_amax"].default is True
    from cudnn.sdpa.fwd.kernels import _mxfp8_host

    assert {"gate_ptr", "gate_strides"} <= inspect.signature(_mxfp8_host.host).parameters.keys()
    for mod in (fp8, mxfp8):
        host = list(inspect.signature(mod._host).parameters)
        stream_slot = host.index("stream")
        assert host[stream_slot : stream_slot + 2] == ["stream", "gate_tensor"], (mod.__name__, host[stream_slot:])
        assert host[stream_slot + 2 :] in ([], ["prepared"]), (mod.__name__, host[stream_slot:])
        assert inspect.signature(mod._host).parameters["gate_tensor"].default is None
    for mod in (f16, fp8, mxfp8):
        assert "gate_tensor" in inspect.signature(mod._kernel).parameters and "tma_gate_desc" in inspect.signature(mod._kernel).parameters
    # The THD length slots sit RIGHT AFTER seq_q_lens_addr and BEFORE stream: the positional tail the shared
    # pointer hosts pass (_fp8_host._launch always, _mxfp8_host._launch under thd_slots), so a parameter inserted
    # ahead of them would swallow the THD lengths.  The MXFP8 kernel reads its SF tile extents off the bound SF
    # tensors -- the total_*_sf_tiles host parameters its pre-upstream THD arm kept in exactly these slots are
    # gone (no caller ever passed them; the pointer hosts are the only callers).
    thd_slots = ["thd_q_lens_tensor", "thd_kv_lens_tensor", "thd_lens_form"]
    for mod in (fp8, mxfp8):
        host = list(inspect.signature(mod._host).parameters)
        at = host.index("seq_q_lens_addr")
        assert host[at + 1 : at + 4] == thd_slots and host.index("thd_lens_form") < host.index("stream"), (mod.__name__, host[at:])
    mx_host = list(inspect.signature(mxfp8._host).parameters)
    assert not {"total_q_sf_tiles", "total_kv_sf_tiles"} & set(mx_host), mx_host
    sm100_mx_host = list(inspect.signature(_load(_D256, rubin=False, **_MXFP8_LOAD_KW)._host).parameters)
    at = sm100_mx_host.index("seq_q_lens_addr")
    assert sm100_mx_host[at + 1 : at + 4] == thd_slots, sm100_mx_host[at:]

    # Ungated f16: the gate slot is folded out (the fake is None iff CFG.EPILOGUE_GATE == 0).
    off = _load(_D256, rubin=True)
    assert off.CFG.EPILOGUE_GATE == 0 and "gate_ptr" in inspect.signature(off._host).parameters


def test_sm107_gate_seams_are_named_once():
    """Every splice in all three kernels is marked ``# == EPILOGUE_FUSION_SEAM(<name>) ==``
    -- the 16 names exactly once each, so a reviewer can walk the fusion in
    pipeline order with one grep and a dropped / duplicated seam is visible.
    And none of the fork's module knobs survived the port (contract rule 5)."""
    import re
    from pathlib import Path

    for mod in _all_gate_kernel_modules():
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        seams = re.findall(r"# == EPILOGUE_FUSION_SEAM\((\w+)\) ==", src)
        counts = {name: seams.count(name) for name in set(seams)}
        assert set(seams) == set(_GATE_SEAMS), f"{mod.__name__}: seams {sorted(set(seams) ^ set(_GATE_SEAMS))} missing or unknown"
        assert all(n == 1 for n in counts.values()), f"{mod.__name__}: duplicated seams {[k for k, n in counts.items() if n != 1]}"
        code = _code_lines(src)
        for knob in _RETIRED_GATE_KNOBS:
            assert not re.search(rf"^{knob}\b\s*[:=]", code, re.M), f"{mod.__name__}: fork-era module knob {knob} survived"
            assert not re.search(rf"\b{knob}\b", code), f"{mod.__name__}: fork-era knob {knob} is still referenced"
        if not hasattr(mod, "compile"):
            from cudnn.sdpa.fwd.kernels import _fp8_host, _mxfp8_host

            host_module = _mxfp8_host if "mxfp8" in mod.__file__ else _fp8_host
            code = Path(host_module.__file__).read_text()
        assert 'options="--enable-tvm-ffi"' in code, f"{mod.__name__}: compile options must stay the production ones"


# --- Adapter accept / reject (CPU; the device capability is monkeypatched) ------


def _desc(shape, dtype, name, stride=None):
    """A BSHD-physical (BHSD-logical) TensorDesc with no storage behind it."""
    from cudnn.api_base import TensorDesc

    b, h, s, d = shape
    stride = tuple(stride) if stride is not None else (s * h * d, d, h * d, 1)
    return TensorDesc(
        dtype=dtype, shape=tuple(shape), stride=stride, stride_order=TensorDesc._compute_stride_order(tuple(shape), stride), device="cuda", name=name
    )


def _fake_cc(monkeypatch, cc):
    """check_support reads the LIVE device; pin it so the Rubin arm runs on any host."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cc)


def _gate_api(
    *,
    b=1,
    h=8,
    h_kv=None,
    s=512,
    d=256,
    d_v=None,
    dtype=None,
    fp8=False,
    pertensor=True,
    gate_dtype=None,
    gate_shape=None,
    gate_stride=None,
    with_gate=True,
    dtype_o=None,
    **kw,
):
    """A d256-shaped SdpaFwdDslSm100 built from descriptors only.  ``dtype_o``
    overrides the quantized paths' default bf16 O (e.g. an e4m3 O)."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    h_kv = h if h_kv is None else h_kv
    d_v = d if d_v is None else d_v
    dtype = dtype or torch.float16
    qkv_dtype = torch.float8_e4m3fn if fp8 else dtype
    o_dtype = (dtype_o or torch.bfloat16) if fp8 else dtype
    gate_dtype = gate_dtype or (torch.bfloat16 if fp8 else dtype)
    q = _desc((b, h, s, d), qkv_dtype, "q")
    k = _desc((b, h_kv, s, d), qkv_dtype, "k")
    v = _desc((b, h_kv, s, d_v), qkv_dtype, "v")
    o = _desc((b, h, s, d_v), o_dtype, "o")
    if with_gate:
        kw["sample_gate"] = _desc(gate_shape or (b, h, s, d_v), gate_dtype, "gate", stride=gate_stride)
    if fp8:
        kw.update(pertensor_fp8=pertensor, dtype_o=o_dtype)
    return SdpaFwdDslSm100(q, k, v, o, None, **kw)


def _stub_compiled(monkeypatch, api):
    """Let execute() reach its presence contracts without a kernel or a stream."""
    api._compiled_kernel = object()
    monkeypatch.setattr(api, "_get_default_stream", lambda stream: stream)


def test_gate_check_support_declines_typed(monkeypatch):
    """Rule 8b twins of the rows' gate claims, on the STANDALONE adapter -- each
    a typed decline (ValueError for a malformed request, NotImplementedError for
    "not mine"), never a bare escape from compile().  The accept side first, so
    a decline cannot pass by the whole path being dead."""
    import torch

    _fake_cc(monkeypatch, (10, 7))
    for dt in (torch.float16, torch.bfloat16):
        api = _gate_api(dtype=dt)
        assert api.check_support()
        assert api.gate_desc is not None and api.gate_desc.dtype == dt
        from cudnn.sdpa.fwd.config_sm100 import dense_bind_strides

        assert dense_bind_strides(tuple(api.gate_desc.shape), tuple(api.gate_desc.stride), dt.itemsize) is not None
        assert api.template_params().epilogue_gate is True
    assert _gate_api(with_gate=False).template_params().epilogue_gate is False
    assert _gate_api(fp8=True).check_support(), "bf16 G on the per-tensor FP8 path"
    assert _gate_api(fp8=True, has_amax_o=False).check_support(), "the Amax_O fold-out is a quantized-path option"
    # MXFP8 (PR-B; INVERTED from PR-A's "not wired" decline): admitted on the SAME terms as FP8 -- Rubin, exactly
    # (256, 256), a bf16 G -- with a bf16 or an (unscaled) e4m3 O, and the Amax_O fold-out.
    import torch as _torch

    for dto in (_torch.bfloat16, _torch.float8_e4m3fn):
        mx = _gate_api(fp8=True, pertensor=False, dtype_o=dto)
        assert mx.check_support(), dto
        assert mx.template_params().epilogue_gate is True and mx.template_params().dtype_o == (2 if dto is _torch.bfloat16 else 0)
    assert _gate_api(fp8=True, pertensor=False, has_amax_o=False).check_support()
    with pytest.raises(ValueError, match="GATE"):
        _gate_api(fp8=True, pertensor=False, gate_dtype=torch.float16).check_support()
    for d, d_v in ((128, 128), (192, 128), (512, 512)):
        with pytest.raises(NotImplementedError, match="256"):
            _gate_api(fp8=True, pertensor=False, d=d, d_v=d_v).check_support()

    # Malformed requests -> ValueError.
    with pytest.raises(ValueError, match="GATE"):
        _gate_api(gate_shape=(1, 8, 256, 256)).check_support()
    with pytest.raises(ValueError, match="GATE"):
        _gate_api(gate_dtype=torch.bfloat16).check_support()
    with pytest.raises(ValueError, match="GATE"):
        _gate_api(fp8=True, gate_dtype=torch.float16).check_support()
    # A G whose head stride is not a 16-byte multiple is not TMA-expressible; the adapter never hides that behind a copy.
    with pytest.raises(ValueError, match="TMA-expressible"):
        _gate_api(gate_stride=(512 * 8 * 260, 260, 8 * 260, 1)).check_support()
    with pytest.raises(ValueError, match="has_amax_o"):
        _gate_api(with_gate=False, has_amax_o=False).check_support()

    # "Not mine" -> NotImplementedError, naming the axis.
    for d, d_v in ((128, 128), (192, 128), (512, 512)):
        with pytest.raises(NotImplementedError, match="256"):
            _gate_api(d=d, d_v=d_v).check_support()
    with pytest.raises(NotImplementedError, match="dense-only"):
        _gate_api(thd=True).check_support()
    # split_kv=2: the PRE-EXISTING SM107 split guard ("split_kv > 1 on cc10.7 is wired only for per-tensor FP8
    # d128") fires first, gate on or off -- the gate's own split clause is unreachable on Rubin, because the
    # only split-wired flavor (fp8 d128) is declined by the gate's head-dim guard before it.  Pinned on the
    # shared "split_kv" token so either ordering is a typed decline.
    with pytest.raises(NotImplementedError, match="split_kv"):
        _gate_api(split_kv=2).check_support()
    with pytest.raises(NotImplementedError, match="PackGQA"):
        _gate_api(h_kv=2, pack_gqa=True).check_support()
    # Paged KV: K/V are page pools; the d256 f16 flavor serves paged, the gate does not ride it.
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    pool = _desc((16, 8, 128, 256), torch.float16, "k")
    with pytest.raises(NotImplementedError, match="paged"):
        SdpaFwdDslSm100(
            _desc((1, 8, 512, 256), torch.float16, "q"),
            pool,
            _desc((16, 8, 128, 256), torch.float16, "v"),
            _desc((1, 8, 512, 256), torch.float16, "o"),
            None,
            seq_kv_lens_present=True,
            paged_page_size=128,
            paged_max_seq_len_kv=512,
            sample_gate=_desc((1, 8, 512, 256), torch.float16, "gate"),
        ).check_support()

    # Execute presence contracts, both directions, plus the Amax_O fold-out.
    t = torch.empty(1)
    api = _gate_api()
    assert api.check_support()
    _stub_compiled(monkeypatch, api)
    with pytest.raises(ValueError, match="gate"):
        api.execute(t, t, t, t)
    plain = _gate_api(with_gate=False)
    assert plain.check_support()
    _stub_compiled(monkeypatch, plain)
    with pytest.raises(ValueError, match="sample_gate"):
        plain.execute(t, t, t, t, gate=t)
    no_amax = _gate_api(fp8=True, with_gate=False, has_amax_o=False)
    assert no_amax.check_support()
    _stub_compiled(monkeypatch, no_amax)
    with pytest.raises(ValueError, match="has_amax_o"):
        no_amax.execute(t, t, t, t, amax_o=t)


def test_gate_check_support_declines_other_arch_lines(monkeypatch):
    """The gate is served by the Rubin d256 kernels ONLY: the SM100 arm of the
    same adapter and the SM120 / SM80 adapters all decline it typed."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120, SdpaFwdDslSm80

    _fake_cc(monkeypatch, (10, 0))
    with pytest.raises(NotImplementedError, match="Rubin"):
        _gate_api().check_support()
    assert _gate_api(with_gate=False).check_support(), "the ungated d256 graph is served on SM100"
    for cls in (SdpaFwdDslSm120, SdpaFwdDslSm80):
        q = _desc((1, 8, 512, 128), torch.float16, "q")
        api = cls(q, q, q, _desc((1, 8, 512, 128), torch.float16, "o"), None, sample_gate=_desc((1, 8, 512, 128), torch.float16, "gate"))
        with pytest.raises(NotImplementedError, match="SM107"):
            api.check_support()


# --- Rubin e2e, standalone adapter ----------------------------------------------
#
# Sentinel-filled O (an unwritten tile stays visible), NaN-filled LSE, and the
# two-launch trick (a first-launch race shows as a second-launch delta) on every
# run.  Tolerances are the shared SM10x suites' (O atol 5e-2 / rtol 3e-2, LSE
# atol 2e-2); bitwise claims are ``torch.equal``.

_GATE_O_TOL = dict(atol=5e-2, rtol=3e-2)
_GATE_LSE_TOL = dict(atol=2e-2, rtol=2e-2)
# Amax_O is an in-kernel atomicMax over the fp32 pre-cast values: the MXFP8 suite's
# bound (test_sdpa_fwd_mxfp8_sm100 asserts |amax - ref| <= 0.03 on every flavor).
_MX_AMAX_ATOL = 0.03


def _rubin_only():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the fused epilogue gate is served by the sm107 d256 kernels (cc10.7) only")


def _gate_problem(b, h, h_kv, s, d, dtype, *, seed=0, generator=None):
    """BSHD-physical / BHSD-logical Q, K, V and a gate G of O's shape."""
    import torch

    if generator is None:
        torch.manual_seed(seed)
    dev = "cuda"
    q = (torch.randn(b, s, h, d, device=dev, generator=generator) * 0.5).to(dtype).transpose(1, 2)
    k = (torch.randn(b, s, h_kv, d, device=dev, generator=generator) * 0.5).to(dtype).transpose(1, 2)
    v = (torch.randn(b, s, h_kv, d, device=dev, generator=generator) * 0.5).to(dtype).transpose(1, 2)
    gate = (torch.randn(b, s, h, d, device=dev, generator=generator) * 2.0).to(dtype).transpose(1, 2)
    return q, k, v, gate


def _gate_reference(q, k, v, gate, *, causal, scale, seq_kv_lens=None):
    """fp32 softmax(QK^T) V * sigmoid(G) (``gate=None``: the UNGATED O, the Amax_O reference)
    and the fp64 natural-log LSE (-inf on an empty row)."""
    import torch

    rep = q.shape[1] // k.shape[1]
    kf, vf = k.float().repeat_interleave(rep, 1), v.float().repeat_interleave(rep, 1)
    logits = q.float() @ kf.transpose(-1, -2) * scale
    s_q, s_kv = q.shape[2], k.shape[2]
    if causal:
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        logits = logits.masked_fill(j > i + (s_kv - s_q), float("-inf"))
    if seq_kv_lens is not None:
        cols = torch.arange(s_kv, device=q.device).view(1, 1, 1, s_kv)
        logits = logits.masked_fill(cols >= seq_kv_lens.view(-1, 1, 1, 1), float("-inf"))
    lse = torch.logsumexp(logits.double(), dim=-1).float()
    p = torch.softmax(logits, dim=-1).nan_to_num(0.0)  # an empty row is all -inf: O = 0
    o = p @ vf
    return (o * torch.sigmoid(gate.float()) if gate is not None else o), lse


def _run_gated(q, k, v, gate, *, causal, cga=None, sched_policy=None, seq_kv_lens=None, with_lse=True, gate_on=True):
    """Build, compile and launch TWICE; return (api, O, LSE) with the sentinel / two-launch checks done."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d_v = q.shape[0], q.shape[1], q.shape[2], v.shape[3]
    dtype = q.dtype
    out = torch.full((b, s, h, d_v), 1.5e30, device=q.device, dtype=torch.float32).to(dtype).transpose(1, 2)  # sentinel (inf in fp16)
    sentinel = out[0, 0, 0, 0].item()
    lse = torch.full((b, h, s), float("nan"), device=q.device, dtype=torch.float32) if with_lse else None
    api = SdpaFwdDslSm100(
        q,
        k,
        v,
        out,
        lse,
        is_causal=causal,
        scale_softmax=q.shape[3] ** -0.5,
        seq_kv_lens_present=seq_kv_lens is not None,
        cga=cga,
        sched_policy=sched_policy,
        **({"sample_gate": gate} if gate_on else {}),
    )
    assert api.check_support()
    api.compile()
    kw = dict(lse_tensor=lse, seq_kv_lens=seq_kv_lens)
    if gate_on:
        kw["gate"] = gate
    api.execute(q, k, v, out, **kw)
    torch.cuda.synchronize()
    first_o, first_lse = out.clone(), (lse.clone() if lse is not None else None)
    api.execute(q, k, v, out, **kw)
    torch.cuda.synchronize()
    assert torch.equal(out, first_o), "two-launch delta on O: a first-launch race (missing fence_proxy?)"
    assert lse is None or torch.equal(lse, first_lse), "two-launch delta on LSE"
    assert torch.isfinite(out.float()).all() and not (out == sentinel).any(), "unwritten (sentinel) or non-finite O cells"
    return api, out, lse


@pytest.mark.parametrize(
    "dtype, causal, s, cga",
    [
        ("fp16", False, 512, None),
        ("fp16", True, 1000, None),
        ("bf16", False, 512, None),
        ("bf16", True, 1000, None),
    ],
    ids=["fp16-dense-512", "fp16-causal-1000", "bf16-dense-512", "bf16-causal-1000"],
)
def test_sm107_gate_matches_the_fp32_oracle(dtype, causal, s, cga):
    """Rubin e2e for the f16/bf16 row's gate claim: O == softmax(QK^T)V *
    sigmoid(G) against the fp32 oracle at the shared tolerance; LSE is the
    plain (ungated) fp64 log-sum-exp.  S=1000 causal covers a KV tail.  Always
    the default cga (2): the f16 d256 flavor serves cga2 ONLY on Rubin -- see
    test_sm107_gate_f16_d256_declines_an_explicit_cga1."""
    import torch

    _rubin_only()
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    b, h, h_kv, d = 2, 8, 2, 256
    q, k, v, gate = _gate_problem(b, h, h_kv, s, d, dt)
    _, out, lse = _run_gated(q, k, v, gate, causal=causal, cga=cga)
    ref_o, ref_lse = _gate_reference(q, k, v, gate, causal=causal, scale=d**-0.5)
    torch.testing.assert_close(out.float(), ref_o, **_GATE_O_TOL)
    torch.testing.assert_close(lse, ref_lse, **_GATE_LSE_TOL)


def test_sm107_gate_f16_d256_declines_an_explicit_cga1():
    """The f16 d256 flavor is cga2-only on Rubin (``supported_cgas_for((256, 256),
    fp8=False, (10, 7)) == (2,)``: at cga1 the Q-union-O alias plus the K/V rings
    at STAGES_KV=2 exceed the 320 KiB carveout), with or without the gate.  The
    PR-A plan's Q18 assumed (1, 2) and asked for a cga=1 oracle case; the adapter
    declines it before any kernel loads (measured on the dev node 2026-09-15:
    "only supports cga in (2,)").  Pin the decline with the gate ON so a widening
    of the flavor's cga domain has to come back here and add the oracle case."""
    import torch

    _rubin_only()
    b, h, h_kv, s, d = 2, 8, 2, 1000, 256
    q, k, v, gate = _gate_problem(b, h, h_kv, s, d, torch.bfloat16)
    with pytest.raises(ValueError, match="cga"):
        _run_gated(q, k, v, gate, causal=True, cga=1)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_sm107_gate_dead_padded_entry_is_exactly_zero(dtype):
    """sdpa-invariants S1/S2: a batch entry with seq_kv_len == 0 runs ZERO KV
    iterations, so its epilogue must SELECT zero, never multiply the (possibly
    NaN) accumulator residue -- and the gate fma must sit BEFORE that select,
    so a +-1e4 gate on the dead entry cannot leak through: O exactly 0, LSE
    exactly -inf.  The live entry stays on the oracle."""
    import torch

    _rubin_only()
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    b, h, h_kv, s, d = 2, 8, 2, 512, 256
    q, k, v, gate = _gate_problem(b, h, h_kv, s, d, dt)
    sign = torch.where(torch.arange(s * h * d, device="cuda") % 2 == 0, 1.0, -1.0).view(s, h, d)
    gate.transpose(1, 2)[1] = (sign * 1e4).to(dt)
    lens = torch.tensor([s, 0], dtype=torch.int32, device="cuda")
    _, out, lse = _run_gated(q, k, v, gate, causal=False, seq_kv_lens=lens)
    assert (out[1] == 0).all(), "the dead entry must be EXACTLY zero (select, not residue * 0)"
    assert torch.isneginf(lse[1]).all(), "an empty row's LSE is -inf (no floor may leak: -69.08 = log 1e-30)"
    ref_o, ref_lse = _gate_reference(q, k, v, gate, causal=False, scale=d**-0.5, seq_kv_lens=lens)
    torch.testing.assert_close(out[0].float(), ref_o[0], **_GATE_O_TOL)
    torch.testing.assert_close(lse[0], ref_lse[0], **_GATE_LSE_TOL)


def test_sm107_gate_reads_a_strided_slab_bitwise():
    """G sliced out of a fused projection slab (token stride N, not h*d) is
    read ZERO-COPY through the declared BSHD stride and yields the BITWISE
    same O as a compact G with the same values -- first G alone, then Q/K/V/G
    all strided, which is what the gated attention block runs."""
    import torch

    _rubin_only()
    b, h, s, d = 2, 8, 512, 256
    dt = torch.bfloat16
    q, k, v, gate = _gate_problem(b, h, h, s, d, dt)
    api_c, out_c, lse_c = _run_gated(q, k, v, gate, causal=True)
    from cuda.bindings import driver
    from cudnn.sdpa.fwd.prepared import facts_of_tensor
    from sdpa_binding_reference import bind_dense

    def check_binding(api, values, output, stats, expected):
        spec = api._dense_spec
        facts = {name: facts_of_tensor(t) for name, t in zip(("q", "k", "v", "gate", "o", "lse"), (*values, output, stats))}
        stream = torch.cuda.current_stream().cuda_stream
        frame = bind_dense(spec, facts, driver.CUstream(stream), stream)
        for role, strides in expected.items():
            assert frame[spec.index[role + "_ptr"]] == facts[role].ptr
            assert frame[spec.index[role + "_strides"]] == strides

    check_binding(api_c, (q, k, v, gate), out_c, lse_c, {"gate": (s * h * d, h * d, d)})

    n = 4 * h * d  # a [B, S, N] slab: q | k | v | gate column blocks
    slab = torch.zeros(b, s, n, device="cuda", dtype=dt)
    for i, t in enumerate((q, k, v, gate)):
        slab[:, :, i * h * d : (i + 1) * h * d] = t.transpose(1, 2).reshape(b, s, h * d)
    sliced = [slab[:, :, i * h * d : (i + 1) * h * d].view(b, s, h, d).transpose(1, 2) for i in range(4)]
    assert torch.equal(sliced[3], gate) and not sliced[3].is_contiguous()

    # G strided alone.
    api_g, out_g, lse_g = _run_gated(q, k, v, sliced[3], causal=True)
    check_binding(api_g, (q, k, v, sliced[3]), out_g, lse_g, {"gate": (s * n, n, d)})
    assert torch.equal(out_g, out_c) and torch.equal(lse_g, lse_c), "a strided G must read bitwise as the compact one"
    # Everything strided (the block's layout).
    api_s, out_s, lse_s = _run_gated(*sliced, causal=True)
    check_binding(api_s, sliced, out_s, lse_s, dict.fromkeys(("q", "k", "v", "gate"), (s * n, n, d)))
    assert torch.equal(out_s, out_c) and torch.equal(lse_s, lse_c), "strided Q/K/V/G must read bitwise as compact"


def test_sm107_gate_lse_is_bitwise_independent_of_the_gate():
    """The gate multiplies the fp32 pre-cast accumulator in the epilogue and
    nothing upstream of it: the published LSE is BITWISE the ungated kernel's,
    and O is the ungated O times sigmoid(G) to rounding."""
    import torch

    _rubin_only()
    b, h, h_kv, s, d = 2, 8, 2, 1000, 256
    q, k, v, gate = _gate_problem(b, h, h_kv, s, d, torch.bfloat16)
    _, out_on, lse_on = _run_gated(q, k, v, gate, causal=True, gate_on=True)
    _, out_off, lse_off = _run_gated(q, k, v, gate, causal=True, gate_on=False)
    assert torch.equal(lse_on, lse_off), "LSE must not depend on the gate"
    torch.testing.assert_close(out_on.float(), out_off.float() * torch.sigmoid(gate.float()), **_GATE_O_TOL)
    assert not torch.equal(out_on, out_off), "the gate must actually apply (a +-2 sigma G is far from sigmoid == 1)"


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
def test_sm107_d128_lpt_dense_capture_matches_natural(dtype, causal):
    """D128's dense scheduler writes every row and replays changed inputs."""
    import torch

    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_NATURAL

    _rubin_only()
    dt = torch.float16 if dtype == "fp16" else torch.bfloat16
    q, k, v, _ = _gate_problem(2, 16, 4, 1025, 128, dt, generator=torch.Generator(device="cuda").manual_seed(0))
    seq_kv_lens = torch.tensor([1025, 769], device=q.device, dtype=torch.int32)
    captures, outputs = [], []
    for policy in (SCHED_NATURAL, SCHED_LPT):
        api, out, lse = _run_gated(q, k, v, None, causal=causal, sched_policy=policy, seq_kv_lens=seq_kv_lens, gate_on=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            api.execute(q, k, v, out, lse_tensor=lse, seq_kv_lens=seq_kv_lens)
        captures.append(graph)
        outputs.append((api, out, lse))
    try:
        for _ in range(2):
            v.mul_(-0.5)
            ref_o, ref_lse = _gate_reference(q, k, v, None, causal=causal, scale=128**-0.5, seq_kv_lens=seq_kv_lens)
            for graph, (_, out, lse) in zip(captures, outputs):
                out.fill_(float("nan"))
                lse.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out.float(), ref_o, **_GATE_O_TOL)
                torch.testing.assert_close(lse, ref_lse, **_GATE_LSE_TOL)
            torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=0, rtol=0)
            torch.testing.assert_close(outputs[0][2], outputs[1][2], atol=0, rtol=0)
    finally:
        for graph in captures:
            graph.reset()


def test_sm107_gate_lpt_is_bitwise_natural():
    """The f16 row claims LPT at (256, 256); the scheduler only reorders whole
    work items, so O and LSE under SCHED_LPT with the gate ON are BITWISE the
    NATURAL run's (a multi-wave causal grid so the persistent loop walks the
    remapped order for several rounds)."""
    import torch

    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines

    _rubin_only()
    caps = _caps(_GATE_ROWS[0])
    assert SCHED_LPT in engines.effective_sched_policies(caps, _gate_facts(causal=True))
    b, h, h_kv, s, d = 2, 16, 4, 1024, 256
    q, k, v, gate = _gate_problem(b, h, h_kv, s, d, torch.bfloat16)
    _, out_nat, lse_nat = _run_gated(q, k, v, gate, causal=True, sched_policy=SCHED_NATURAL)
    _, out_lpt, lse_lpt = _run_gated(q, k, v, gate, causal=True, sched_policy=SCHED_LPT)
    assert torch.equal(out_lpt, out_nat), "LPT must be bit-identical to NATURAL -- a different tile walk, the same per-tile math"
    assert torch.equal(lse_lpt, lse_nat)
    ref_o, _ = _gate_reference(q, k, v, gate, causal=True, scale=d**-0.5)
    torch.testing.assert_close(out_nat.float(), ref_o, **_GATE_O_TOL)


# ============================================================================
# Fused epilogue gate on the MXFP8 d256 kernel (PR-B slice S7, 2026-09-15)
#
# The block-scale ``sm107/prefill_d256_mxfp8.py`` carries the SAME shared hook
# and the same 16 seams as its f16 / per-tensor FP8 siblings (the tests above
# now iterate all three modules).  What is specific to this body, and what the
# tests below pin:
#   * the gate ``SmemTile`` is declared AFTER the four scale-factor slabs -- the
#     module pins DESC_VERSION=0, so every UTCCP-fed SF slab must START under the
#     256 KiB version-0 window, and a gate declared before them with a bf16 O at
#     STAGES_KV=2 would put sQ_SF at exactly 262144 (PR-B D7);
#   * the SF-root guard: ``config_sm107.d256_mxfp8_last_sf_tile_start`` (raises at
#     make_cfg) and the kernel's own ``_last_sf_tile_start`` backstop agree, and
#     a synthetic STAGES_KV=3 + bf16-O config is refused with the window's name;
#   * ``compile(has_amax=False)`` folds the amax out exactly as on the FP8 body;
#   * an e4m3 O is UNSCALED (the kernel has no per-tensor scale_o; PR-B D8).
# The Rubin e2e at the end is the evidence behind the row's claim.
# ============================================================================


def test_sm107_mxfp8_gate_row_reaches_the_lowering_with_the_ctor_kwargs():
    """The MXFP8 row's claim is not decorative: the same ``_epilogue_gate_ctor_kwargs``
    that hands the FP8 row its ``sample_gate`` / ``has_amax_o`` does so for MXFP8
    facts, and the row-level knob interactions decline typed.  The d256 MXFP8
    flavor stays NATURAL-only under the gate (no LPT claim rides in on PR-B)."""
    import inspect

    import cudnn
    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
    from cudnn.sdpa.fwd import api_dsl, engines, heuristics
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps(_GATE_ROWS[2])
    assert caps.is_mxfp8 and caps.epilogue_gate is True
    assert caps.epilogue_gate_d_shapes <= caps.d_shapes
    ctor = frozenset(inspect.signature(api_dsl.SdpaFwdDsl.__init__).parameters)
    exe = frozenset(inspect.signature(api_dsl.SdpaFwdDslSm100.execute).parameters)
    assert "sample_gate" in ctor and "has_amax_o" in ctor and "gate" in exe, "the adapter carries the PR-A constructor / execute surface"
    facts = _mxfp8_gate_facts()
    # The Amax_O fold-out rides the same helper for MXFP8 facts (the gate half needs a real IR tensor, so it is
    # exercised by the graph-path e2e, test_mxfp8_gate_tail_graph_api).
    assert engines._epilogue_gate_ctor_kwargs(_mxfp8_facts(d_qk=256, d_v=256), ctor, exe, "x") == {"has_amax_o": False}
    assert engines._epilogue_gate_ctor_kwargs(_mxfp8_facts(d_qk=256, d_v=256, amax_o_t=object()), ctor, exe, "x") == {"has_amax_o": True}
    assert heuristics._sched_points(caps, _mxfp8_gate_facts(causal=True)) == [SCHED_NATURAL]
    assert engines.effective_sched_policies(caps, facts) == frozenset({SCHED_NATURAL})
    why = engines.mismatch(caps, facts, SdpaFwdKnobs(split_kv=2))
    assert why is not None and "split_kv" in why, why
    why = engines.mismatch(caps, _mxfp8_gate_facts(h_kv=2), SdpaFwdKnobs(pack_gqa=True))
    assert why is not None, why
    why = engines.mismatch(caps, _mxfp8_gate_facts(epilogue_gate_shape_ok=False))
    assert why is not None and "shape" in why, why
    why = engines.mismatch(caps, _mxfp8_gate_facts(epilogue_gate_layout_ok=False))
    assert why is not None and "zero-copy" in why, why
    # No row outside the three d256 ones claims MXFP8 + gate.
    for spec in engines.ENGINE_SPECS:
        if spec.capabilities.is_mxfp8 and spec.name != _GATE_ROWS[2]:
            assert spec.capabilities.epilogue_gate is False, spec.name
    _ = cudnn  # imported for symmetry with the sibling tests


def test_sm107_mxfp8_gate_template_geometry_and_desc_version():
    """The gated MXFP8 module: gate ON is a separate specialization of the same
    file, GATE geometry derives from GATE_BPE (4 bf16 subtiles of 64, 64 KiB per
    tile, NO CTA_MMA factor), and -- unlike its siblings -- DESC_VERSION stays the
    PINNED 0 with and without the gate (version 1 is not a transparent widening
    on the MXFP8 tiles; the gate tile is never a descriptor-fed operand)."""
    import cutlass

    for dto in (_E4M3, _BF16_OUT):
        off = _load(_D256, rubin=True, **{**_MXFP8_LOAD_KW, "dtype_o": dto})
        on = _mxfp8_gate_kernel_module(dtype_o=dto)
        assert on is not off and on.__file__ == off.__file__
        assert off.CFG.EPILOGUE_GATE == 0 and on.CFG.EPILOGUE_GATE == 1
        assert on.DESC_VERSION == off.DESC_VERSION == 0, "the MXFP8 body pins version 0 (mma-tma-matrix.md S6)"
        assert not hasattr(on, "_needs_desc_v1"), "no derived version bit on this body -- the SF-root guard is the tripwire instead"
        assert on.GATE_STORAGE_DTYPE is cutlass.BFloat16 and on.CFG.GATE_BPE == 2
        assert on._GG.tma_iters == 4 and on._GG.tma_granu_elems == 64 and on._GG.d_block == 64 and on._GG.granu_local == 128 * 64
        assert on.gateBufferElems == 128 * 256 and on.gateTmaTransactionBytes == 64 * 1024, "one Q tile, bf16, no CTA_MMA factor"
        assert on.SMEM_LAYOUT_GATE == 2 and on._GATE_SMEM_SWIZZLE == cutlass.Swizzle(3, 4, 3)
        # The gate's walk is NOT O's: with an e4m3 O the O subtile is 128 wide.
        o_d_block = on.CFG.TILE_O // ((on.CFG.TILE_O * on.CFG.BPE_O) // on.CFG.O_SWZ_BYTES)
        assert (o_d_block == 128) == (dto == _E4M3) and on._GG.d_block == 64
        assert on.FROST_SOURCE_DIGEST != off.FROST_SOURCE_DIGEST


def test_sm107_mxfp8_gate_smem_tally_and_sf_root_guard():
    """PR-B D7 in numbers.  The tally gains the four SF slabs (6 KiB at depth 2,
    named in the validator's message) on top of the gate's 64 KiB; the two
    layouts the block runs fit (e4m3-O 232 KiB, bf16-O 264 KiB).  The SF-ROOT
    guard: a bf16 O at STAGES_KV=3 (or an e4m3 O at 4) puts the scale factors at
    or past 262144 -- refused at make_cfg with the window's name, BEFORE the
    SMEM check speaks -- and the kernel's own ``_last_sf_tile_start`` backstop
    agrees with the config twin at every depth / O width.  The gate does not
    move the SF slabs (declared after them)."""
    from dataclasses import dataclass, replace

    from cudnn.sdpa.fwd.config_sm107 import (
        SMEM_USABLE_BYTES,
        TCGEN05_V0_ADDR_LIMIT,
        d256_mxfp8_last_sf_tile_start,
        d256_mxfp8_sf_smem_bytes,
        make_cfg_d256_mxfp8,
        sdpa_smem_bytes,
    )

    KIB = 1024
    fixed = 2 * KIB
    assert TCGEN05_V0_ADDR_LIMIT == 262144
    assert d256_mxfp8_sf_smem_bytes(128, 128, 256, 256, 2) == 6 * KIB  # Q 1 | K 2x1 | P 1 (512 B padded) | V 2x1
    assert d256_mxfp8_sf_smem_bytes(128, 128, 256, 256, 3) == 8 * KIB

    def tally(*, stages_kv, bpe_o, gate_bpe):
        sf = d256_mxfp8_sf_smem_bytes(128, 128, 256, 256, stages_kv)
        return sdpa_smem_bytes(128, 128, 256, 256, 1, stages_kv, 1, 1, bpe_o, qo_alias=True, gate_bpe=gate_bpe, sf_bytes=sf) + fixed

    assert tally(stages_kv=2, bpe_o=1, gate_bpe=0) == 168 * KIB  # today's e4m3-O
    assert tally(stages_kv=2, bpe_o=2, gate_bpe=0) == 200 * KIB  # today's bf16-O
    assert tally(stages_kv=2, bpe_o=1, gate_bpe=2) == 232 * KIB  # fully fused block (e4m3 O)
    assert tally(stages_kv=2, bpe_o=2, gate_bpe=2) == 264 * KIB  # bf16 O + gate
    assert tally(stages_kv=3, bpe_o=2, gate_bpe=2) == 330 * KIB > SMEM_USABLE_BYTES

    @dataclass(frozen=True)
    class _ParamsWithStagesKv(TemplateParams):
        stages_kv: int = None

    def cfg_of(depth, dto, gate=True):
        return make_cfg_d256_mxfp8(_ParamsWithStagesKv(stages_kv=depth, epilogue_gate=gate, dtype_qkv=_E4M3, dtype_o=dto, cta_mma=1))[0]

    # Accepted: the block's two modes at depth 2; e4m3-O at depth 3 (SF root 231 KiB, 298 KiB total).
    for depth, dto in ((2, _E4M3), (2, _BF16_OUT), (3, _E4M3)):
        cfg = cfg_of(depth, dto)
        assert cfg.EPILOGUE_GATE == 1 and cfg.STAGES_KV == depth
        assert d256_mxfp8_last_sf_tile_start(cfg) < TCGEN05_V0_ADDR_LIMIT, (depth, dto)
    assert d256_mxfp8_last_sf_tile_start(cfg_of(2, _E4M3)) == 165 * KIB and d256_mxfp8_last_sf_tile_start(cfg_of(2, _BF16_OUT)) == 197 * KIB
    # Refused BY THE WINDOW GUARD (its diagnostic, not the SMEM one): bf16 O at depth 3 (sQ_SF at exactly 256 KiB),
    # e4m3 O at depth 4 -- both latent until PR-B (depth 3 bf16-O fit the SMEM check at 266 KiB ungated).
    for depth, dto, gate in ((3, _BF16_OUT, True), (3, _BF16_OUT, False), (4, _E4M3, True), (4, _E4M3, False)):
        with pytest.raises(ValueError, match="version-0 tcgen05 descriptor window") as ei:
            cfg_of(depth, dto, gate)
        assert "262144" in str(ei.value) and f"STAGES_KV={depth}" in str(ei.value), str(ei.value)
    # The out-of-domain depth message names BOTH bounds for this family.
    with pytest.raises(ValueError, match="2..4") as ei:
        cfg_of(5, _E4M3)
    assert "version-0 tcgen05 descriptor window" in str(ei.value) and "DESC_VERSION=0" in str(ei.value), str(ei.value)

    # Kernel-side backstop == config twin, on the loaded modules, at every depth / O width, gate on and off.
    for dto in (_E4M3, _BF16_OUT):
        off = _load(_D256, rubin=True, **{**_MXFP8_LOAD_KW, "dtype_o": dto})
        on = _mxfp8_gate_kernel_module(dtype_o=dto)
        assert on._LAST_SF_TILE_START == off._LAST_SF_TILE_START == d256_mxfp8_last_sf_tile_start(on.CFG) < 262144, dto
        for depth in (2, 3, 4):
            for bpe_o in (1, 2):
                c = replace(on.CFG, STAGES_KV=depth, BPE_O=bpe_o)
                assert on._last_sf_tile_start(c) == d256_mxfp8_last_sf_tile_start(c), (dto, depth, bpe_o)
        assert on._last_sf_tile_start(replace(on.CFG, STAGES_KV=3, BPE_O=2)) >= 262144, "the kernel backstop must fire where the config does"
        assert on._last_sf_tile_start(replace(on.CFG, STAGES_KV=4, BPE_O=1)) >= 262144
        # D7 in the SOURCE: the gate slab is declared after the last SF slab AND after the last SF SmemTile.
        with open(on.__file__, encoding="utf-8") as fh:
            src = fh.read()
        assert src.index("sV_SF_raw = cutlass.Array(") < src.index("sV_SF = SmemTile(") < src.index("sGate_raw = (") < src.index("bars = make_d256_bars(")


# --- Rubin e2e, MXFP8 standalone adapter -------------------------------------
#
# Inputs from the shared torch MXFP8 quantizer (rowwise Q/K, columnwise V, the
# F8_128x4 SF the kernel consumes); the oracle runs on the DEQUANTIZED operands.
# O tolerance = the MXFP8 suite's (5e-2 abs for a half O; for an e4m3 O
# max(5e-2, 3 x the e4m3 rounding floor of the reference)); LSE = the gate
# suite's; bitwise claims are ``torch.equal``.  Sentinel-filled O (bf16 1.5e30;
# e4m3 byte 0x7F = NaN, which a satfinite cast never produces), NaN-filled LSE
# and the two-launch trick on every run.  sched_policy is passed EXPLICITLY
# (NATURAL, the only policy the row claims at d256) and cga is left to the
# adapter (it pins cta_mma=1 for the quantized d256 flavors).


def _mx_bshd(x_bhsd_f32, b, h, s, d, *, columnwise, fp8_dtype=None):
    """quantize_to_mxfp8 -> (fp8 data as a BHSD view over BSHD storage, fp32 dequantized [b,h,s,d], F8_128x4 SF).
    ``fp8_dtype``: e4m3 (default) or e5m2 -- the row claims BOTH input members."""
    import torch
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    fp8_dtype = fp8_dtype or torch.float8_e4m3fn
    data_d, sf_d, swz_d, data_s, sf_s, swz_s = quantize_to_mxfp8(x_bhsd_f32.contiguous(), b, h, s, d, 32, fp8_dtype, with_ref=True)
    data, sf, swz = (data_s, sf_s, swz_s) if columnwise else (data_d, sf_d, swz_d)
    deq = data.float().reshape(b, h, s, d) * sf.float().reshape(b, h, s, d)
    return data.permute(0, 2, 1, 3).contiguous().transpose(1, 2), deq, swz.contiguous()


def _mxfp8_gate_problem(b, h, h_kv, s, d, *, seed=0, in_dtype=None):
    """((q8, k8, v8, sf_q, sf_k, sf_v), (q_deq, k_deq, v_deq), gate) -- gate bf16 of O's shape, +-2 sigma.
    ``in_dtype``: the fp8 input member (e4m3 default / e5m2); the oracle sees the DEQUANTIZED operands either way."""
    import torch

    torch.manual_seed(seed)
    dev = "cuda"
    qf = torch.randn(b, h, s, d, device=dev) * 0.5
    kf = torch.randn(b, h_kv, s, d, device=dev) * 0.5
    vf = torch.randn(b, h_kv, s, d, device=dev) * 0.5
    q8, q_deq, sfq = _mx_bshd(qf, b, h, s, d, columnwise=False, fp8_dtype=in_dtype)
    k8, k_deq, sfk = _mx_bshd(kf, b, h_kv, s, d, columnwise=False, fp8_dtype=in_dtype)
    v8, v_deq, sfv = _mx_bshd(vf, b, h_kv, s, d, columnwise=True, fp8_dtype=in_dtype)
    gate = (torch.randn(b, s, h, d, device=dev) * 2.0).to(torch.bfloat16).transpose(1, 2)
    return (q8, k8, v8, sfq, sfk, sfv), (q_deq, k_deq, v_deq), gate


# Sentinels per O dtype: fp8 byte 0x7F is NaN in BOTH e4m3 and e5m2 and a
# satfinite cast never produces it; fp16 cannot hold 1.5e30 (overflows to inf,
# which the finiteness check would then blame on the kernel), so it takes 6.0e4
# (exactly representable, far above any |O| these inputs reach); bf16 keeps 1.5e30.
_MX_FP8_DTYPES = ("float8_e4m3fn", "float8_e5m2")


def _mx_sentinel_value(dtype):
    import torch

    return 6.0e4 if dtype == torch.float16 else 1.5e30


def _mx_sentinel_fill(o):
    import torch

    if o.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        o.view(torch.uint8).fill_(0x7F)
    else:
        o.fill_(_mx_sentinel_value(o.dtype))


def _mx_sentinel_survivors(o):
    import torch

    if o.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return int((o.view(torch.uint8) == 0x7F).sum().item())
    return int((o.float() == _mx_sentinel_value(o.dtype)).sum().item())


def _mx_check_o(out, ref, *, in_key="e4m3"):
    """The MXFP8 suite's bound (``test_sdpa_fwd_mxfp8_sm100._check`` at d_qk=256): a half
    O within 5e-2 abs for e4m3 inputs / 8e-2 for the noisier e5m2 inputs (2-bit
    mantissa, d_qk > 128); an fp8 O within max(that, 3 x the reference's own
    rounding floor in the OUTPUT fp8 dtype)."""
    import torch

    diff = (out.float() - ref).abs().max().item()
    tol = 8e-2 if in_key == "e5m2" else 5e-2
    if out.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tol = max(tol, 3.0 * (ref - ref.to(out.dtype).float()).abs().max().item())
    assert diff <= tol, f"max|O-ref| = {diff:.4f} exceeds {tol:.4f}"


def _run_gated_mxfp8(ops, gate, *, causal, out_dtype, seq_kv_lens=None, with_lse=True, gate_on=True, has_amax_o=True, amax=None):
    """Build, compile and launch TWICE through the standalone adapter; return (api, O, LSE)."""
    import torch

    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    q8, k8, v8, sfq, sfk, sfv = ops
    b, h, s, d = q8.shape
    d_v = v8.shape[3]
    out = torch.empty(b, s, h, d_v, device=q8.device, dtype=out_dtype).transpose(1, 2)
    _mx_sentinel_fill(out)
    lse = torch.full((b, h, s), float("nan"), device=q8.device, dtype=torch.float32) if with_lse else None
    api = SdpaFwdDslSm100(
        q8,
        k8,
        v8,
        out,
        lse,
        is_causal=causal,
        scale_softmax=d**-0.5,
        seq_kv_lens_present=seq_kv_lens is not None,
        pertensor_fp8=False,
        dtype_o=out_dtype,
        sched_policy=SCHED_NATURAL,
        has_amax_o=has_amax_o,
        **({"sample_gate": gate} if gate_on else {}),
    )
    assert api.check_support()
    assert api.template_params().epilogue_gate is gate_on
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device=q8.device, dtype=torch.uint8)
    kw = dict(lse_tensor=lse, seq_kv_lens=seq_kv_lens, sf_q=sfq, sf_k=sfk, sf_v=sfv, workspace=workspace)
    if gate_on:
        kw["gate"] = gate
    if amax is not None:
        kw["amax_o"] = amax
    api.execute(q8, k8, v8, out, **kw)
    torch.cuda.synchronize()
    first_o, first_lse = out.clone(), (lse.clone() if lse is not None else None)
    api.execute(q8, k8, v8, out, **kw)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.uint8), first_o.view(torch.uint8)), "two-launch delta on O: a first-launch race (missing fence_proxy?)"
    assert lse is None or torch.equal(lse, first_lse), "two-launch delta on LSE"
    assert _mx_sentinel_survivors(out) == 0, "unwritten (sentinel) O cells"
    assert torch.isfinite(out.float()).all(), "non-finite O cells"
    return api, out, lse


# Every frozenset member the MXFP8 row claims WITH the gate gets one launch
# (engine-contract rule 9: a frozenset field is that many separate claims):
# inputs {e4m3, e5m2} x outputs {fp16, bf16, e4m3, e5m2}.  e4m3-in carries the
# shape/mask coverage (dense 512 + causal 1000 at bf16 and e4m3 O); the other
# members get one shape each -- the gate path is dtype-orthogonal (it multiplies
# the fp32 accumulator before the single output cast), so one launch per member
# is what turns the claim into evidence, not a sweep.
_MX_IN = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}
_MX_OUT = {"fp16": "float16", "bf16": "bfloat16", "e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}


@pytest.mark.parametrize(
    "in_key, out_key, causal, s",
    [
        ("e4m3", "bf16", False, 512),
        ("e4m3", "bf16", True, 1000),
        ("e4m3", "e4m3", False, 512),
        ("e4m3", "e4m3", True, 1000),
        ("e4m3", "fp16", False, 512),
        ("e4m3", "e5m2", False, 512),
        ("e5m2", "bf16", True, 1000),
        ("e5m2", "fp16", False, 512),
        ("e5m2", "e4m3", False, 512),
        ("e5m2", "e5m2", True, 1000),
    ],
    ids=[
        "e4m3-bf16-dense-512",
        "e4m3-bf16-causal-1000",
        "e4m3-e4m3-dense-512",
        "e4m3-e4m3-causal-1000",
        "e4m3-fp16-dense-512",
        "e4m3-e5m2-dense-512",
        "e5m2-bf16-causal-1000",
        "e5m2-fp16-dense-512",
        "e5m2-e4m3-dense-512",
        "e5m2-e5m2-causal-1000",
    ],
)
def test_sm107_mxfp8_gate_matches_the_dequant_oracle(in_key, out_key, causal, s):
    """Rubin e2e for the MXFP8 row's gate claim, one launch per claimed dtype
    member: O == softmax(QK^T)V * sigmoid(G) on the DEQUANTIZED operands within
    the MXFP8 suite's bound (half O, or an UNSCALED fp8 O -- PR-B D8: the kernel
    has no per-tensor scale_o); LSE is the plain (ungated) fp64 log-sum-exp.
    S=1000 causal covers a KV tail."""
    import torch

    _rubin_only()
    in_dt = getattr(torch, _MX_IN[in_key])
    dt = getattr(torch, _MX_OUT[out_key])
    b, h, h_kv, d = 2, 8, 2, 256
    ops, (q_deq, k_deq, v_deq), gate = _mxfp8_gate_problem(b, h, h_kv, s, d, in_dtype=in_dt)
    assert ops[0].dtype == in_dt
    _, out, lse = _run_gated_mxfp8(ops, gate, causal=causal, out_dtype=dt)
    assert out.dtype == dt
    ref_o, ref_lse = _gate_reference(q_deq, k_deq, v_deq, gate, causal=causal, scale=d**-0.5)
    _mx_check_o(out, ref_o, in_key=in_key)
    torch.testing.assert_close(lse, ref_lse, **_GATE_LSE_TOL)


# The one-time MXFP8 gate-off comparison against the pre-gate tensor entry
# is recorded in SUPPORT_MATRIX_TRACKER.md footnote viii (2026-09-15).
# That entry is retired; the live gate-off/gate-on and Amax checks below use
# the supported prepared path and remain the ongoing regressions.


def test_sm107_mxfp8_gate_lse_is_bitwise_independent_of_the_gate():
    """The gate multiplies the fp32 pre-cast accumulator in the epilogue and
    nothing upstream of it: the published LSE is BITWISE the ungated kernel's,
    and O is the ungated O times sigmoid(G) to rounding."""
    import torch

    _rubin_only()
    b, h, h_kv, s, d = 2, 8, 2, 1000, 256
    ops, _, gate = _mxfp8_gate_problem(b, h, h_kv, s, d)
    _, out_on, lse_on = _run_gated_mxfp8(ops, gate, causal=True, out_dtype=torch.bfloat16, gate_on=True)
    _, out_off, lse_off = _run_gated_mxfp8(ops, gate, causal=True, out_dtype=torch.bfloat16, gate_on=False)
    assert torch.equal(lse_on, lse_off), "LSE must not depend on the gate"
    torch.testing.assert_close(out_on.float(), out_off.float() * torch.sigmoid(gate.float()), **_GATE_O_TOL)
    assert not torch.equal(out_on, out_off), "the gate must actually apply (a +-2 sigma G is far from sigmoid == 1)"


@pytest.mark.parametrize("out_key", ["bf16", "e4m3"])
def test_sm107_mxfp8_gate_amax_is_a_compile_time_fact(out_key):
    """``has_amax_o=False`` folds the Amax_O atomic out of the gated MXFP8 kernel
    (the adapter records ``_amax_folded_out``, binds None in the slot and never
    resets a buffer), O stays BITWISE identical, and an amax_o handed to that
    specialization is a typed ValueError.  With the amax requested it is the
    amax of the UNGATED, dead-row-selected fp32 pre-cast O -- the sdpa NODE's
    output, which on the graph PRECEDES the sigmoid/mul tail -- in the O's own
    units (no per-tensor scale_o on this path), while the quantized O itself
    IS the gated value.  A random G cannot tell the two apart (the top |O|
    cells carry sigmoid(G) ~ 1), so the discrimination is STRUCTURAL: G = -1e4
    (sigmoid == 0) zeroes O yet must leave Amax_O at the un-gated value (a
    gated-value reduction reports exactly 0); G = +1e4 (sigmoid == 1)
    reproduces it; and all three gated runs equal the UNGATED specialization's
    Amax_O bit-for-bit (the kernel folds |h| = |u/2| and doubles once per tile,
    exact in fp32) -- the per-tensor FP8 sibling's contract
    (test_sdpa_fp8_sm107::test_fp8_d256_amax_is_the_pre_gate_value)."""
    import torch

    _rubin_only()
    dt = {"bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn}[out_key]
    b, h, h_kv, s, d = 2, 8, 2, 512, 256
    ops, (q_deq, k_deq, v_deq), gate = _mxfp8_gate_problem(b, h, h_kv, s, d)
    ungated_ref, _ = _gate_reference(q_deq, k_deq, v_deq, None, causal=False, scale=d**-0.5)
    ungated_amax = ungated_ref.abs().max().item()
    assert ungated_amax > 2 * _MX_AMAX_ATOL, f"un-gated amax {ungated_amax:.4f} too small to be told from zero -- reshape the probe"
    amax = torch.full((1,), -1.0, device="cuda", dtype=torch.float32)
    api_a, out_a, lse_a = _run_gated_mxfp8(ops, gate, causal=False, out_dtype=dt, has_amax_o=True, amax=amax)
    assert getattr(api_a, "_amax_folded_out", None) is False and amax.item() > 0.0
    assert abs(amax.item() - ungated_amax) <= _MX_AMAX_ATOL, f"Amax_O {amax.item():.4f} vs the UN-gated reference {ungated_amax:.4f}"
    api_n, out_n, lse_n = _run_gated_mxfp8(ops, gate, causal=False, out_dtype=dt, has_amax_o=False)
    assert api_n._amax_folded_out is True, "the MXFP8 d256 kernel carries has_amax, so the fold-out must be live"
    assert torch.equal(out_n.view(torch.uint8), out_a.view(torch.uint8)), "has_amax_o=False must fold only the Amax_O write out"
    assert torch.equal(lse_n, lse_a)
    with pytest.raises(ValueError, match="has_amax_o"):
        api_n.execute(*ops[:3], out_n, lse_tensor=lse_n, sf_q=ops[3], sf_k=ops[4], sf_v=ops[5], gate=gate, amax_o=amax)
    # Structural amax discrimination.  sigmoid(G) == 0: every gated O cell is 0, yet Amax_O is the
    # UN-gated amax (a gated-value reduction would report exactly 0).
    amax_neg = torch.full((1,), -1.0, device="cuda", dtype=torch.float32)
    _, out_neg, _ = _run_gated_mxfp8(ops, torch.full_like(gate, -1e4), causal=False, out_dtype=dt, has_amax_o=True, amax=amax_neg)
    assert (out_neg.float() == 0).all(), "O itself must be the gated (zero) value"
    assert (
        abs(amax_neg.item() - ungated_amax) <= _MX_AMAX_ATOL
    ), f"Amax_O {amax_neg.item():.4f} with sigmoid(G) == 0 vs un-gated ref {ungated_amax:.4f}: the kernel reduced the GATED O"
    # sigmoid(G) == 1: the gated value IS the un-gated one, and so is Amax_O.
    amax_pos = torch.full((1,), -1.0, device="cuda", dtype=torch.float32)
    _, out_pos, _ = _run_gated_mxfp8(ops, torch.full_like(gate, 1e4), causal=False, out_dtype=dt, has_amax_o=True, amax=amax_pos)
    assert abs(amax_pos.item() - ungated_amax) <= _MX_AMAX_ATOL, f"Amax_O {amax_pos.item():.4f} with sigmoid(G) == 1 vs un-gated ref {ungated_amax:.4f}"
    _mx_check_o(out_pos, ungated_ref)
    # G-independence and agreement with the UNGATED specialization are bit-exact, not tolerance-bound.
    amax_off = torch.full((1,), -1.0, device="cuda", dtype=torch.float32)
    _, out_off, _ = _run_gated_mxfp8(ops, gate, causal=False, out_dtype=dt, gate_on=False, has_amax_o=True, amax=amax_off)
    _mx_check_o(out_off, ungated_ref)
    assert amax_neg.item() == amax_pos.item() == amax.item() == amax_off.item(), (
        f"Amax_O must be G-independent and equal the ungated kernel's: -1e4 {amax_neg.item():.6f}, +1e4 {amax_pos.item():.6f}, "
        f"random {amax.item():.6f}, ungated {amax_off.item():.6f}"
    )


@pytest.mark.parametrize("out_key", ["bf16", "e4m3"])
def test_sm107_mxfp8_gate_dead_padded_entry_is_exactly_zero(out_key):
    """sdpa-invariants S1/S2 on the MXFP8 body: a batch entry with seq_kv_len ==
    0 runs ZERO KV iterations, so its epilogue must SELECT zero, never multiply
    the (possibly NaN) accumulator residue -- and the gate fma sits BEFORE that
    select, so a +-1e4 gate on the dead entry cannot leak through: O exactly 0,
    LSE exactly -inf, Amax_O untouched by the dead rows (the gated arm's |h|
    fold sees the dead rows' NaN-able residue, and its once-per-tile select at
    corr_release must zero it before the atomicMax).  The live entry stays on
    the oracle, and Amax_O is the live entry's UNGATED amax (the sdpa node's
    output, independent of G).  (The empty-mainloop arm also issues the gate
    load -- P14 -- which is what this shape exercises.)"""
    import torch

    _rubin_only()
    dt = {"bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn}[out_key]
    b, h, h_kv, s, d = 2, 8, 2, 512, 256
    ops, (q_deq, k_deq, v_deq), gate = _mxfp8_gate_problem(b, h, h_kv, s, d)
    sign = torch.where(torch.arange(s * h * d, device="cuda") % 2 == 0, 1.0, -1.0).view(s, h, d)
    gate.transpose(1, 2)[1] = (sign * 1e4).to(torch.bfloat16)
    lens = torch.tensor([s, 0], dtype=torch.int32, device="cuda")
    amax = torch.full((1,), -1.0, device="cuda", dtype=torch.float32)
    _, out, lse = _run_gated_mxfp8(ops, gate, causal=False, out_dtype=dt, seq_kv_lens=lens, amax=amax)
    assert (out[1].float() == 0).all(), "the dead entry must be EXACTLY zero (select, not residue * 0)"
    assert torch.isneginf(lse[1]).all(), "an empty row's LSE is -inf (no floor may leak: -69.08 = log 1e-30)"
    ref_o, ref_lse = _gate_reference(q_deq, k_deq, v_deq, gate, causal=False, scale=d**-0.5, seq_kv_lens=lens)
    _mx_check_o(out[0], ref_o[0])
    torch.testing.assert_close(lse[0], ref_lse[0], **_GATE_LSE_TOL)
    ungated_ref, _ = _gate_reference(q_deq, k_deq, v_deq, None, causal=False, scale=d**-0.5, seq_kv_lens=lens)
    assert amax.item() > 0.0 and torch.isfinite(amax).all(), "Amax_O must be written, and finite (the dead rows' NaN residue must be selected out)"
    assert (
        abs(amax.item() - ungated_ref[0].abs().max().item()) <= _MX_AMAX_ATOL
    ), f"Amax_O {amax.item():.4f} is the live entry's UNGATED amax {ungated_ref[0].abs().max().item():.4f}; the dead entry contributes nothing"


# ============================================================================ Rubin SASS pins: trace-compile for sm_107a on ANY box
# The two perf regressions PR #1129 closed were invisible to every numerics test (O / LSE / Amax_O bit-identical before and after):
# an Amax_O fold that lowered to thousands of FSETP + FSEL instead of FMNMX3, and a cluster-scope release arrive in the shared CLC
# scheduler that put a GPU-scope drain (MEMBAR.ALL.GPU + CGAERRBAR) before every credit.  The only tripwire for either is the SASS,
# so this pin compiles each kernel for Rubin here (`CUTE_DSL_ARCH=sm_107a` needs no matching device) and counts the instructions.
# SKIPS when the DSL predates sm_107a or no nvdisasm on $CUDA_PATH/bin or $PATH decodes the cubin (the DSL's own wheel nvdisasm
# may not); a compile failure is a FAIL.
_SM107_SASS_PROBE = textwrap.dedent("""
    import glob, os, re, subprocess, sys
    dump, quant, d, dtype_o, mask, cands = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6:]
    # mask specialization: the TemplateParams fields the engine sets for a graph's mask (dense = none)
    MASKS = {"dense": {}, "causal_swa640": dict(window_right=0, window_left=640), "causal_padded": dict(window_right=0, seq_kv_lens_present=True)}
    os.environ["CUTE_DSL_DUMP_DIR"] = dump          # read once, at the first cutlass import
    os.environ["CUTE_DSL_KEEP"] = "cubin"            # keep the cubin, disassemble it ourselves
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"       # unconditional: an inherited value would pin the wrong target's SASS
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"  # a compiled-plan cache HIT skips ptxas and dumps no cubin
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module, supported_cgas_for
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    # The PRODUCTION CTA geometry, from the adapter itself: cga2 for d128 / d512, cga1 for d256 (review on #1129: a d256 fp8 row
    # compiled at cga2 was a configuration the adapter never builds).  One width per Rubin quantized flavor today; if that ever
    # widens, the unpack fails here and the row has to say which geometry it pins.
    (cta_mma,) = supported_cgas_for((d, d), fp8=True, device_cc=(10, 7), pertensor=(quant == "fp8"))
    params = TemplateParams(dtype_qkv=0, dtype_o=dtype_o, cta_mma=cta_mma, **MASKS[mask])  # E4M3 in; the row picks the O dtype
    mod = _load_sm100_kernel_module((d, d), params, fp8=True, pertensor=(quant == "fp8"), rubin=True)
    # By keyword: the MXFP8 kernels' compile() carries total_{q,kv}_sf_tiles between skv and d_qk.  Dense, B=1 H=128 S=8192,
    # Stats + Amax_O (emit_amax_o defaults True) -- the sweep's shape, and the arm that carries the fold.
    mod.compile_prepared(d_qk=d, d_v=d, has_lse=True)
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
    print("SASS FSETP", cnt("FSETP"))
    print("SASS FMNMX3", cnt("FMNMX3"))
    print("SASS MEMBAR_GPU", cnt("MEMBAR.ALL.GPU"))
    print("SASS CGAERRBAR", cnt("CGAERRBAR"))
    print("SASS STL", cnt("STL"))
    print("SASS LDL", cnt("LDL"))
    print("SASS R2P", cnt(" R2P "))
    # The predicate-to-general-register move (the reverse of R2P): P, a digit, R.  Counted through a pattern so the opcode is
    # never spelled in this source.
    print("SASS PRED2GPR", sum(1 for ln in sass if re.search(r" P\\dR ", ln)))
    print("SASS ISETP", cnt(" ISETP"))
    print("SASS LINES", len(sass))
    """)

# One row per pinned kernel: (quantization path, d, TemplateParams.dtype_o).  The fold commit touched all eight sm107 fp8 / mxfp8
# kernels; these four are the two the regression was found on, the d128 fp8 kernel (FSETP 790 -> 26 on the fold) and the mxfp8
# d512 kernel (the largest measured win, +17.5 % dense / +29 % causal at S=8K).  The fp8 rows write E4M3 O; the mxfp8 row writes
# BF16 O, the dtype MXFP8 graphs produce -- the fold sits on the fp32 accumulator before the cast, and the sm_107a counts are the
# same with E4M3 O (2026-09-18: FSETP 9 / FMNMX3 256 / no drain / no spill either way).  Four rows, one sm_107a trace-compile
# each (~20-60 s); the other four kernels share the fold's spelling and the scheduler, and add no new class of failure.
# 4th field: the row's FSETP ceiling = the count measured on the fixed kernel (2026-09-18, sm_107a, PRODUCTION geometry) plus
# _FSETP_SLACK.  One fold site regressing to compare+select adds at least 3 FSETP per O element per lane (384 on a 128-element
# chunk), so a slack of 16 still catches a single site while tolerating unrelated drift; the aggregate `< 100` alone would not
# (review on #1129).  5th field: the row's spill ceiling.  d256 fp8 at its production cga1 carries 3 STL / 3 LDL on the PR BASE
# d34a6909 already (the cga2 build this row compiled before had none), so 3 is the pre-existing count, not a budget for new spills.
_FSETP_SLACK = 16
_SM107_SASS_PIN_ROWS = [
    pytest.param("fp8", 512, _E4M3, 14 + _FSETP_SLACK, 0, id="fp8-d512"),
    pytest.param("fp8", 256, _E4M3, 13 + _FSETP_SLACK, 3, id="fp8-d256"),
    pytest.param("fp8", 128, _E4M3, 26 + _FSETP_SLACK, 0, id="fp8-d128"),
    pytest.param("mxfp8", 512, _BF16_OUT, 9 + _FSETP_SLACK, 0, id="mxfp8-d512"),
]


def _nvdisasm_candidates():
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm"))
    on_path = shutil.which("nvdisasm")
    if on_path:
        cands.append(on_path)
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


def _sm107a_known_to_the_dsl() -> bool:
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string("sm_107a")
        return True
    except Exception:
        return False


def _sm107a_sass_counts(dump, quant, d, dtype_o, mask, cands):
    """Trace-compile one sm107 quantized kernel specialization for sm_107a in a subprocess and return its opcode counts
    (skips the test when no nvdisasm candidate decodes the cubin; a compile failure is a FAIL)."""
    argv = [sys.executable, "-c", _SM107_SASS_PROBE, str(dump), quant, str(d), str(dtype_o), mask, *cands]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_107a trace-compile of the {quant} d={d} {mask} kernel failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    return {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].isdigit()}


# The masked softmax arm's SASS pin (sm_107a listings of both forms, 2026-09-22).  In the bit-word form (`tile_dsl.mask.apply_mask_chunk`)
# every masked KV-tile body carries 4 R2P per 32-column word and ~0.04 ISETP per cell; the per-cell compare + select form it replaced
# carried 0 R2P and 1 ISETP per cell per mask term (611-764 ISETP whole-kernel on these two builds vs 107-108 now), and the mxfp8 d512
# causal+SWA build ran out of predicate registers (152 predicate-to-register moves, REG 254 -> 165).  Rows: (quant, d, dtype_o, mask
# spec, ISETP ceiling, predicate-to-register-move ceiling, spill ceiling); the
# ceilings are the measured bit-word counts (2026-09-22, sm_107a, production geometry) plus slack -- one masked arm falling back
# to per-cell compares adds >= 128 ISETP, so a slack of 32 still catches a single arm.
_SM107_MASK_SASS_ROWS = [
    pytest.param("mxfp8", 512, _BF16_OUT, "causal_swa640", 108 + 32, 2 + 6, 0, id="mxfp8-d512-causal_swa640"),
    pytest.param("fp8", 128, _E4M3, "causal_padded", 108 + 32, 0 + 6, 9, id="fp8-d128-causal_padded"),
]


@pytest.mark.parametrize("quant, d, dtype_o, mask, isetp_max, pred_spill_max, spill_max", _SM107_MASK_SASS_ROWS)
def test_sm107_masked_softmax_sass_is_register_to_predicate(tmp_path, quant, d, dtype_o, mask, isetp_max, pred_spill_max, spill_max):
    """The masked softmax arm masks through R2P + FSEL (the bit-word form of apply_mask_chunk), not one ISETP + FSEL per cell per term: R2P > 0,
    whole-kernel ISETP within the measured ceiling, no predicate-register spill storm (predicate-to-register moves) and no new stack spills
    (the d128 causal builds carry 5 STL / 9 LDL per TILE on develop already -- that is the row's pre-existing count)."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"sm107a_{quant}_d{d}_{mask}"
    dump.mkdir()
    stats = _sm107a_sass_counts(dump, quant, d, dtype_o, mask, cands)
    print(f"\nsm107 {quant} d={d} {mask} sm_107a SASS: {stats}")
    assert stats["R2P"] > 0, "no R2P: the masked arm is back to per-cell compare + select (tile_dsl.mask.apply_mask_chunk regressed)"
    assert stats["ISETP"] <= isetp_max, f"{stats['ISETP']} ISETP > {isetp_max}: a masked arm is comparing per cell again"
    assert (
        stats["PRED2GPR"] <= pred_spill_max
    ), f"{stats['PRED2GPR']} predicate-to-register moves > {pred_spill_max}: predicate registers are spilling into GPRs again"
    assert (
        stats["STL"] <= spill_max and stats["LDL"] <= spill_max
    ), f"the {quant} d={d} {mask} kernel spills ({stats['STL']} STL / {stats['LDL']} LDL, ceiling {spill_max})"


@pytest.mark.parametrize("quant, d, dtype_o, fsetp_max, spill_max", _SM107_SASS_PIN_ROWS)
def test_sm107_fp8_epilogue_and_scheduler_sass_pins(tmp_path, quant, d, dtype_o, fsetp_max, spill_max):
    """The Amax_O fold is FMNMX3 (not a compare+select chain) and the CLC scheduler's credit arrives carry no GPU-scope
    drain -- the two silent perf regressions of PR #1129, pinned on the sm_107a SASS of the fp8 d=512 / d=256 / d=128
    and the mxfp8 d=512 kernels, each compiled at the CTA geometry the adapter serves (cga2 for d128 / d512, cga1 for
    d256).  Bounds: FSETP <= measured + 16 per row (the regression read 1548 / 779 / 790, the fixed kernels 9-26; one
    regressed fold site adds >= 384); FMNMX3 > 0 (it read 0 before); MEMBAR.ALL.GPU == CGAERRBAR == 0 (13-23 before);
    spills <= the row's pre-existing count (0 everywhere but d256 fp8 at cga1, which carries 3 on the PR base)."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"sm107a_{quant}_d{d}"
    dump.mkdir()
    stats = _sm107a_sass_counts(dump, quant, d, dtype_o, "dense", cands)
    print(f"\nsm107 {quant} d={d} dtype_o={dtype_o} sm_107a SASS: {stats}")
    assert stats["FMNMX3"] > 0, "the Amax_O fold must reach FMNMX3 (fmax_f32), not a compare+select chain"
    assert stats["FSETP"] <= fsetp_max, f"{stats['FSETP']} FSETP > {fsetp_max}: an Amax_O fold site is lowering to compare+select again"
    assert stats["MEMBAR_GPU"] == 0 and stats["CGAERRBAR"] == 0, "a cluster-scope RELEASE arrive is back on a per-tile path (GPU-scope drain)"
    assert (
        stats["STL"] <= spill_max and stats["LDL"] <= spill_max
    ), f"the {quant} d={d} kernel spills ({stats['STL']} STL / {stats['LDL']} LDL, ceiling {spill_max})"


# ============================================================================ Rubin SASS pins: the d512 O store path
# The two d512 levers above are invisible to every numerics test (O / LSE / Amax_O bit-identical either way); their SASS is the
# tripwire.  O_STORE_STREAM: each `UTMASTG` sits right behind ITS chunk's `PHASECHK` wait, so the longest run of UTMASTG with no
# PHASECHK between them is _O_SUBTILES_PER_CHUNK (1 at the 128 B O swizzle); the whole-tile form reads a run of TMA_O_ITERS_HOST
# (8 at a half-precision O, 4 at fp8 O -- the develop counts).  O_EPI_PIPELINE: the dead-row fast path is the kernel's only
# `VOTE.ANY` (develop: 0; the correction loop's vote is VOTE.ALL).  Both: ONE bulk group per tile (1 UTMACMDFLUSH + 1 DEPBAR), no
# GPU-scope drain, no spill, and REG within the measured ceiling -- the fp8-O batch sizing is what the REG pin holds: a 2-block
# tcgen05.ld batch at an FP8 O is 256 live registers and read REG 255 / STACK 512 / 192 STL / 332 LDL before the batch was sized by
# registers (frost-tile-dsl.md; both O dtypes of the mxfp8 kernel are pinned for that reason).  Measured 2026-09-28 on the branch
# (sm_107a, cutlass-dsl 4.8.0 + CUDA 13.5 ptxas, dense B=1 H=128 S=8192 LSE on, production cga2): REG f16 205 / fp8 156 / mxfp8 224 /
# mxfp8-fp8out 158 (develop 203 / 158 / 222 / -), STL = LDL = 0 on every row; a REG slack of 16 tolerates ptxas drift and still
# catches the 255 of a wrong batch.
_SM107_O_STORE_PROBE = textwrap.dedent("""
    import glob, os, re, subprocess, sys
    dump, quant, d, dtype_o, cands = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "cubin"
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module, supported_cgas_for
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    fp8 = quant not in ("f16", "f16_2x2")
    (cta_mma,) = supported_cgas_for((d, d), fp8=fp8, device_cc=(10, 7), pertensor=(quant == "fp8"))
    # E4M3 in on the quantized rows, BF16 in on the f16 rows; the row picks the O dtype.  "f16_2x2" is the d512
    # 2x2-datapath sibling (TemplateParams.mma_2x2); the role-split rows pass the field's default.
    params = TemplateParams(dtype_qkv=(0 if fp8 else 2), dtype_o=dtype_o, cta_mma=cta_mma, mma_2x2=(quant == "f16_2x2"))
    mod = _load_sm100_kernel_module((d, d), params, fp8=fp8, pertensor=(quant == "fp8"), rubin=True)
    print("IS_2X2", int(mod.__file__.endswith("_2x2.py")))
    print("EXPECT_UTMASTG", mod.TMA_O_ITERS_HOST)
    print("EXPECT_UTMASTG_MAX_RUN", mod._O_SUBTILES_PER_CHUNK if mod.O_STORE_STREAM else mod.TMA_O_ITERS_HOST)
    print("EXPECT_VOTE_ANY", 1 if mod.O_EPI_PIPELINE else 0)
    entry = getattr(mod, "compile_prepared", None) or mod.compile
    entry(d_qk=d, d_v=d, has_lse=True)
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
    for key in ("UTMASTG", "UTMACMDFLUSH", "DEPBAR", "STL", "LDL", "CGAERRBAR"):
        print("SASS", key, cnt(key))
    print("SASS MEMBAR_GPU", cnt("MEMBAR.ALL.GPU"))
    print("SASS VOTE_ANY", cnt("VOTE.ANY"))
    print("SASS VOTE_ALL", cnt("VOTE.ALL"))
    print("SASS UTCHMMA", cnt("UTCHMMA"))
    # The fused TMEM load + row-max reduction (tcgen05.ld.red): the Rubin dense softmax arm.
    print("SASS LDTM_RED", sum(1 for ln in sass if "LDTM" in ln and (".RED" in ln or "STAT" in ln or "MAX" in ln)))
    run = best = 0
    for ln in sass:
        if "UTMASTG" in ln:
            run += 1; best = max(best, run)
        elif "PHASECHK" in ln:
            run = 0
    print("SASS UTMASTG_MAX_RUN", best)
    # REG of the main kernel = the largest REG of any function in the cubin (the amax reset / unscale helpers are tiny), read by the
    # cuobjdump that ships beside the nvdisasm that decoded the cubin; -1 when there is none (the REG bound is then not checked).
    cuobj = os.path.join(os.path.dirname(nvd), "cuobjdump")
    reg = -1
    if os.path.isfile(cuobj):
        ru = subprocess.run([cuobj, "--dump-resource-usage", cubins[-1]], capture_output=True, text=True).stdout
        regs = [int(m) for m in re.findall(r"REG:(\\d+)", ru)]
        reg = max(regs) if regs else -1
    print("SASS REG", reg)
    print("SASS LINES", len(sass))
    """)

_REG_SLACK = 16
_SM107_O_STORE_SASS_ROWS = [
    # (quant, TemplateParams.dtype_o, REG measured on the branch)
    pytest.param("f16", _BF16_OUT, 205, id="f16-d512"),
    pytest.param("fp8", _E4M3, 156, id="fp8-d512"),
    pytest.param("mxfp8", _BF16_OUT, 224, id="mxfp8-d512"),
    pytest.param("mxfp8", _E4M3, 158, id="mxfp8-d512-fp8out"),
]


@pytest.mark.parametrize("quant, dtype_o, reg_measured", _SM107_O_STORE_SASS_ROWS)
def test_sm107_d512_o_store_path_sass_pins(tmp_path, quant, dtype_o, reg_measured):
    """The streamed O store (one UTMASTG per chunk wait, TMA_O_ITERS_HOST stores, one bulk group per tile), the pipelined epilogue
    (one VOTE.ANY), no GPU-scope drain, no spill, REG within the measured ceiling -- on every sm107 d512 kernel, both O dtypes of the
    mxfp8 one.  Expectations come from the module itself (its constants and TMA geometry), so a flipped lever re-pins its own row."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"sm107a_ostore_{quant}_o{dtype_o}"
    dump.mkdir()
    argv = [sys.executable, "-c", _SM107_O_STORE_PROBE, str(dump), quant, "512", str(dtype_o), *cands]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_107a trace-compile of the {quant} d=512 dtype_o={dtype_o} kernel failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].lstrip("-").isdigit()}
    expect = {ln.split()[0]: int(ln.split()[1]) for ln in out if ln.startswith("EXPECT_") and len(ln.split()) == 2}
    print(f"\nsm107 {quant} d=512 dtype_o={dtype_o} sm_107a SASS: {stats}; module says {expect}")
    assert stats["UTMASTG"] == expect["EXPECT_UTMASTG"], f"{stats['UTMASTG']} UTMASTG, the module's TMA-O geometry says {expect['EXPECT_UTMASTG']}"
    assert stats["UTMASTG_MAX_RUN"] == expect["EXPECT_UTMASTG_MAX_RUN"], (
        f"longest UTMASTG run without a PHASECHK between = {stats['UTMASTG_MAX_RUN']}, O_STORE_STREAM says {expect['EXPECT_UTMASTG_MAX_RUN']}: "
        "a store is no longer issued behind its own chunk wait (or the whole-tile arm is being traced)"
    )
    assert (
        stats["UTMACMDFLUSH"] == 1 and stats["DEPBAR"] == 1
    ), f"{stats['UTMACMDFLUSH']} commit / {stats['DEPBAR']} wait_group: the store must stay ONE bulk group per tile"
    assert (
        stats["VOTE_ANY"] == expect["EXPECT_VOTE_ANY"]
    ), f"{stats['VOTE_ANY']} VOTE.ANY, O_EPI_PIPELINE says {expect['EXPECT_VOTE_ANY']}: the dead-row fast path (which pins the pipelined loads) is gone"
    assert stats["MEMBAR_GPU"] == 0 and stats["CGAERRBAR"] == 0, "a cluster-scope RELEASE arrive is back on a per-tile path (GPU-scope drain)"
    assert (
        stats["STL"] == 0 and stats["LDL"] == 0
    ), f"the {quant} d=512 dtype_o={dtype_o} kernel spills ({stats['STL']} STL / {stats['LDL']} LDL): a tcgen05.ld batch wider than 64 fp32 per lane?"
    if stats["REG"] >= 0:
        assert (
            stats["REG"] <= reg_measured + _REG_SLACK
        ), f"REG {stats['REG']} > {reg_measured} + {_REG_SLACK}: the pipelined epilogue holds more than two 64-register load batches"


# ============================================================================ Rubin SASS pins: the d512 2x2-datapath sibling
# The same probe on sm107/prefill_d512_f16_2x2.py (quant "f16_2x2" -> TemplateParams.mma_2x2).  Structure pinned exactly: 8 UTMASTG
# (the streamed O store's eight 8 KiB subtiles, one behind each mb_o_full wait -> longest run 1), ONE bulk group per tile, 96 UTCHMMA
# (prologue BMM1 32 + loop BMM1 32 + loop BMM2 16 + tail BMM2 16 collective MMAs), no GPU-scope drain / CGAERRBAR on any per-tile
# path, exactly one VOTE.ALL (the correction's alpha == 1 ballot) and NO VOTE.ANY (the 2x2 epilogue has no dead-row vote: its
# dead rows leave through beta = 0 on an O accumulator that P = 0 left exactly zero), one LDTM.RED (the dense arm's ld.red.max).
# MEASURED 2026-10-01 on a cc 10.7 board (sm_107a, internal CUDA 13.6-era toolkit cuda-39029786, cutlass-dsl 0.3.0+20260728, bf16
# dense LSE on, CGA_M=4): REG 168 / STACK 0, STL 0 / LDL 0.  History worth keeping: with the leader MMA warp's k/v_full waits on
# tile_dsl wait() (try_wait.parity + time_limit) the same module read STL 17 / LDL 19 -- all in the 40-register MMA warp, 3 STL +
# 6 LDL of 64-bit descriptor pairs per KV iteration around the UTCHMMA issues -- a ptxas scheduling effect, not pressure (the SM100
# body on the same toolkit was 0 / 0; SPIN_RING_WAITS off 28 / 38, 2-deep rings 140 / 144, 56 / 64 single-warp registers 21 / 50 and
# 23 / 51); the non-blocking mbarrier.test_wait.parity poll those waits now take (the cross-pair rule above) removed every spill.
_SM107_2X2_SASS_ROWS = [
    # (quant, TemplateParams.dtype_o, REG measured, STL measured, LDL measured)
    pytest.param("f16_2x2", _BF16_OUT, 168, 0, 0, id="f16_2x2-d512"),
]


@pytest.mark.parametrize("quant, dtype_o, reg_measured, stl_measured, ldl_measured", _SM107_2X2_SASS_ROWS)
def test_sm107_d512_2x2_sass_pins(tmp_path, quant, dtype_o, reg_measured, stl_measured, ldl_measured):
    """The sm_107a SASS of the d512 2x2 sibling: the streamed O store and bulk-group structure of the role-split pin, the MMA
    stream (96 UTCHMMA), the correction ballot (one VOTE.ALL, no VOTE.ANY), the dense arm's LDTM.RED, no GPU-scope drain, and the
    measured spill / REG ceilings (see the section comment for why the spill pin is a bound here)."""
    from frost_test_utils import SPILL_TOLERANCE

    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"sm107a_2x2_{quant}_o{dtype_o}"
    dump.mkdir()
    argv = [sys.executable, "-c", _SM107_O_STORE_PROBE, str(dump), quant, "512", str(dtype_o), *cands]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_107a trace-compile of the {quant} d=512 dtype_o={dtype_o} kernel failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].lstrip("-").isdigit()}
    expect = {ln.split()[0]: int(ln.split()[1]) for ln in out if (ln.startswith("EXPECT_") or ln.startswith("IS_2X2")) and len(ln.split()) == 2}
    print(f"\nsm107 {quant} d=512 dtype_o={dtype_o} sm_107a SASS: {stats}; module says {expect}")
    assert expect["IS_2X2"] == 1, "the f16_2x2 row must load sm107/prefill_d512_f16_2x2.py"
    assert stats["UTMASTG"] == expect["EXPECT_UTMASTG"] == 8
    assert stats["UTMASTG_MAX_RUN"] == expect["EXPECT_UTMASTG_MAX_RUN"] == 1, "every O subtile store sits behind its own mb_o_full wait"
    assert stats["UTMACMDFLUSH"] == 1 and stats["DEPBAR"] == 1, "ONE bulk group per tile"
    assert stats["UTCHMMA"] == 96, f"{stats['UTCHMMA']} UTCHMMA: the 2x2 MMA stream is 32 + 32 + 16 + 16 collective MMAs"
    assert stats["VOTE_ALL"] == 1 and stats["VOTE_ANY"] == 0, "one correction ballot; the 2x2 epilogue carries no dead-row vote"
    assert stats["LDTM_RED"] >= 1, "the dense softmax arm must reach tcgen05.ld.red.max (LDTM.RED)"
    assert stats["MEMBAR_GPU"] == 0 and stats["CGAERRBAR"] == 0, "a cluster-scope RELEASE arrive is back on a per-tile path (GPU-scope drain)"
    assert stats["STL"] <= stl_measured + SPILL_TOLERANCE and stats["LDL"] <= ldl_measured + SPILL_TOLERANCE, (
        f"the 2x2 kernel spills more than measured ({stats['STL']} STL / {stats['LDL']} LDL vs {stl_measured} / {ldl_measured} + {SPILL_TOLERANCE}); "
        "see the section comment: the MMA warp's measured spill form, not a budget for new ones"
    )
    if stats["REG"] >= 0:
        assert stats["REG"] <= reg_measured + _REG_SLACK, f"REG {stats['REG']} > {reg_measured} + {_REG_SLACK}"


# ============================================================================ Rubin SASS pins: the ring-wait retry form
# The hint-less ``wait(spin=True)`` is visible only in the SASS: on sm_107a the DSL's time_limit form costs 2 divergent
# SYNCS.PHASECHK + 1 NANOSLEEP per wait instantiation, the spin form 2 uniform USYNCS.PHASECHK and no sleep.  One opted-in kernel
# (d192x128 mxfp8, SPIN_RING_WAITS=True: the largest measured win, +9.4 % at S=32K) and one opted-out kernel (d128 per-tensor fp8,
# SPIN_RING_WAITS=False: -5.8 % at S=2K when its ring waits spin) are compiled at the production geometry (cga2, dense S=8K, LSE on)
# and their wait opcodes pinned to the counts measured on the shipped tree (2026-09-19; the develop counts before the opt-in are
# in the comments).  The 8 divergent SYNCS.PHASECHK + 6 NANOSLEEP that every kernel keeps are the DSL's tcgen05.alloc lock
# waits in tile_dsl/tmem.py, not ring waits; the rest of the residue on the opted-in row is its idle sites (2 per instantiation).
# A leak of the spin onto the opted-out kernel moves its divergent count by -2 per ring instantiation (-84 here); one ring site of
# the opted-in kernel falling back to the sleeping form moves its count by +2 per instantiation of that site.  Slack 4 tolerates
# unrelated ptxas drift and still catches a single site.
_SM107_WAIT_FORM_PROBE = textwrap.dedent("""
    import glob, inspect, os, subprocess, sys
    dump, quant, d_qk, d_v, dtype_o, cands = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "cubin"
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module, supported_cgas_for
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    (cta_mma,) = supported_cgas_for((d_qk, d_v), fp8=True, device_cc=(10, 7), pertensor=(quant == "fp8"))
    params = TemplateParams(dtype_qkv=0, dtype_o=dtype_o, cta_mma=cta_mma)
    mod = _load_sm100_kernel_module((d_qk, d_v), params, fp8=True, pertensor=(quant == "fp8"), rubin=True)
    kw = dict(b=1, qh=128, kh=128, sq=8192, skv=8192, d_qk=d_qk, d_v=d_v, has_lse=True)
    entry = mod.compile_prepared
    kw = {k: v for k, v in kw.items() if k in inspect.signature(entry).parameters}
    entry(**kw)
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
    print("SASS SYNCS_PHASECHK", sum(1 for ln in sass if "SYNCS.PHASECHK" in ln and "USYNCS.PHASECHK" not in ln))
    print("SASS USYNCS_PHASECHK", sum(1 for ln in sass if "USYNCS.PHASECHK" in ln))
    print("SASS NANOSLEEP", sum(1 for ln in sass if "NANOSLEEP" in ln))
    print("SASS LINES", len(sass))
    """)

# Per-counter bounds, each BELOW the change one ring-wait instantiation makes when it falls back from the spin to the sleeping
# form (measured on the d192x128 mxfp8 kernel, 42 instantiations: SYNCS.PHASECHK 142 -> 58 = -2 per site, USYNCS.PHASECHK
# 75 -> 117 = +1 per site, NANOSLEEP 73 -> 31 = -1 per site), so a single site regressing fails all three assertions. The counts
# are deterministic for a given DSL + ptxas (the same cubin md5 across compiles); a toolchain change that moves them shows up
# as a pin failure with the new values printed, and is re-pinned deliberately, never by widening these.
_WAIT_FORM_SLACK = {"SYNCS_PHASECHK": 1, "USYNCS_PHASECHK": 0, "NANOSLEEP": 0}
_SM107_WAIT_FORM_PIN_ROWS = [
    # (quant, d_qk, d_v, dtype_o, SPIN_RING_WAITS, divergent SYNCS.PHASECHK, uniform USYNCS.PHASECHK, NANOSLEEP)
    # develop (sleeping form everywhere): 142 / 75 / 73; 42 ring-wait instantiations flipped -> 58 / 117 / 31
    pytest.param("mxfp8", 192, 128, _BF16_OUT, True, 58, 117, 31, id="mxfp8-d192x128-spin"),
    # == develop's 160 / 84 / 82: the opt-out must leave every wait opcode where it was
    pytest.param("fp8", 128, 128, _E4M3, False, 160, 84, 82, id="fp8-d128-sleep"),
]


@pytest.mark.parametrize("quant, d_qk, d_v, dtype_o, spin, n_syncs, n_usyncs, n_sleep", _SM107_WAIT_FORM_PIN_ROWS)
def test_sm107_ring_wait_form_sass_pins(tmp_path, quant, d_qk, d_v, dtype_o, spin, n_syncs, n_usyncs, n_sleep):
    """The ring waits of an opted-in kernel lower to the uniform USYNCS.PHASECHK spin (its divergent SYNCS.PHASECHK count drops to
    the idle-site residue and NANOSLEEP to the tmem.py lock waits), and an opted-out kernel keeps develop's counts exactly.
    Cross-checked against the module constant so the row cannot pin a value the kernel does not hold."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    mod = _load((d_qk, d_v), rubin=True, fp8=True, pertensor=(quant == "fp8"), dtype_qkv=_E4M3, dtype_o=dtype_o, cta_mma=2)
    assert mod.SPIN_RING_WAITS is spin, f"{mod.__name__}: SPIN_RING_WAITS={mod.SPIN_RING_WAITS}, this row pins the {spin} form"
    dump = tmp_path / f"sm107a_waitform_{quant}_d{d_qk}x{d_v}"
    dump.mkdir()
    argv = [sys.executable, "-c", _SM107_WAIT_FORM_PROBE, str(dump), quant, str(d_qk), str(d_v), str(dtype_o), *cands]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_107a trace-compile of the {quant} d={d_qk}x{d_v} kernel failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].isdigit()}
    print(f"\nsm107 {quant} d={d_qk}x{d_v} SPIN_RING_WAITS={spin} sm_107a SASS: {stats}")
    for name, want in (("SYNCS_PHASECHK", n_syncs), ("USYNCS_PHASECHK", n_usyncs), ("NANOSLEEP", n_sleep)):
        slack = _WAIT_FORM_SLACK[name]
        assert abs(stats[name] - want) <= slack, f"{name} = {stats[name]}, pinned {want} +- {slack} for SPIN_RING_WAITS={spin}"


# The CI arch targets select this module explicitly; imported tests need L0.
import test_sdpa_staged_forward_half as _staged_half_checks


@pytest.mark.L0
@requires_dsl
class TestStagedHalf:
    test_current_storage = staticmethod(_staged_half_checks.test_half_staged_has_no_execute_allocations_and_replays_current_storage)
    test_partial_staging = staticmethod(_staged_half_checks.test_half_staged_preserves_native_operands_and_split_output)
    test_wrapper_workspace = staticmethod(_staged_half_checks.test_sm100_wrapper_supplies_current_workspace)
    test_compiled_budget = staticmethod(_staged_half_checks.test_half_compiled_workspace_query_uses_prepared_budget)
    test_invalid_bindings = staticmethod(_staged_half_checks.test_half_staged_rejects_invalid_bindings_before_copy)
    test_physical_wide_stride = staticmethod(_staged_half_checks.test_half_staged_physical_wide_stride)
    test_artifact_reload = staticmethod(_staged_half_checks.test_half_staged_artifact_reloads_without_jit)
    test_launch_stream = staticmethod(_staged_half_checks.test_half_staged_copies_follow_launch_stream)


# ============================================================================
# Runtime regression for the mixed-warp correction hand-off (PR #1288, Codex P2)
#
# The O-store stream re-orders the d512 epilogue: BMM2-ready arrives early
# and the correction warps rescale (alpha != 1) or pass through (alpha == 1)
# PER ROW GROUP.  The source / SASS pins above protect the lowering; this is
# the numerical evidence that an early arrival stays correct when some warps
# rescale and others do not.  Geometry from the review's independent probe:
#   * Q[..., 0] repeats 32-row groups of 0 / +1 / -1 (one softmax warp's rows
#     each; 96 does not divide 128, so every warp position sees every group
#     over the tiles); every other Q component is 0.
#   * K[..., 0] = 12 * floor(key / 128): a staircase, so the +1 rows' row-max
#     grows at EVERY KV tile (a rescale per tile), the 0 rows never rescale,
#     the -1 rows never rescale after the first tile.
#   * per-batch lengths Q [1024, 641, 0] / KV [512, 385, 128]: a full, a
#     mid-tile-trimmed and a QUERYLESS batch; bottom-right anchors the
#     diagonal at (seq_len_q[b], seq_len_kv[b]), so the first 512 / 256 rows
#     of batches 0 / 1 are KEYLESS under the masked arm.
#   * scale 0.5; dense (padding only) and bottom-right causal + left window 129.
# Reference: fp64, composing the SAME padding / diagonal / window as the
# kernel (sdpa-invariants § 8); dead rows (row >= seq_len_q[b]) and keyless
# rows are O = 0 / LSE = -inf (§ 1, § 4).  Replay: compile once, then three
# re-executions with a CHANGED V and NaN / +inf-poisoned outputs -- each
# retained O / LSE equals the recomputed reference, LSE is bitwise independent
# of V, and two executions on identical inputs are bitwise equal.
# The scale = 1 masked variant is the LEADING-DEAD-TILE geometry: the live rows
# whose 130-key window excludes KV tile 0 (bottom-right diag >= 257) took the
# scaled mask sentinel, -inf at scale 1, as their running max and published
# LSE = log(1e-30) and O = NaN.  The softmax's running-max step now selects a
# tile that is fully masked ahead of the row's first live key out of the state
# (alpha = 1, P = 0), so those cells run at scale 1 on whichever d512 kernel the
# call-time twin switch selects; test_d512_half_masked_leading_tile_keeps_rows_
# with_later_keys_finite pins the same geometry to the ROLE-SPLIT kernel.
# ============================================================================

_HANDOFF_GEOMETRY = dict(b=3, h_q=8, h_kv=4, s_q=1024, s_kv=512, d=512)
_HANDOFF_Q_LENS = (1024, 641, 0)
_HANDOFF_KV_LENS = (512, 385, 128)
_HANDOFF_ROW_GROUP = 32  # rows per 0 / +1 / -1 group = one softmax warp's rows of a 128-row CTA tile
_HANDOFF_STAIR_STEP, _HANDOFF_STAIR_KEYS = 12.0, 128  # K[..., 0] = 12 * floor(key / 128)
_HANDOFF_WINDOW_LEFT = 129
_HANDOFF_REPLAYS = 3


def _handoff_problem(dtype, *, seed=0):
    """BSHD-physical / BHSD-logical Q, K, V of the review's correction-storm
    geometry, plus the per-batch (seq_q_lens, seq_kv_lens) int32 vectors."""
    import torch

    g = _HANDOFF_GEOMETRY
    dev = "cuda"
    torch.manual_seed(seed)
    rows = torch.arange(g["s_q"], device=dev)
    sign = torch.tensor([0.0, 1.0, -1.0], device=dev)[(rows // _HANDOFF_ROW_GROUP) % 3]
    q = torch.zeros(g["b"], g["s_q"], g["h_q"], g["d"], device=dev, dtype=dtype)
    q[..., 0] = sign.view(1, g["s_q"], 1).to(dtype)
    keys = torch.arange(g["s_kv"], device=dev)
    stair = _HANDOFF_STAIR_STEP * torch.div(keys, _HANDOFF_STAIR_KEYS, rounding_mode="floor").float()
    k = torch.zeros(g["b"], g["s_kv"], g["h_kv"], g["d"], device=dev, dtype=dtype)
    k[..., 0] = stair.view(1, g["s_kv"], 1).to(dtype)
    v = _handoff_fresh_v(dtype, seed=seed)
    q_lens = torch.tensor(_HANDOFF_Q_LENS, dtype=torch.int32, device=dev)
    kv_lens = torch.tensor(_HANDOFF_KV_LENS, dtype=torch.int32, device=dev)
    return q.transpose(1, 2), k.transpose(1, 2), v, q_lens, kv_lens


def _handoff_fresh_v(dtype, *, seed):
    """A new random V (BSHD-physical) -- the operand the replays change."""
    import torch

    g = _HANDOFF_GEOMETRY
    gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
    return (torch.randn(g["b"], g["s_kv"], g["h_kv"], g["d"], device="cuda", generator=gen) * 0.5).to(dtype).transpose(1, 2)


def _handoff_reference(q, k, v, *, scale, causal_br, window_left, q_lens, kv_lens):
    """fp64 softmax(QK^T) V and the natural-log LSE under the SAME conditions
    the kernel runs: per-batch padding on both sides, the bottom-right
    diagonal anchored at (seq_len_q[b], seq_len_kv[b]) and the left window
    riding it (``window_left`` keys below the diagonal are kept, as in
    ``_ref_sdpa_full``'s ``swa_window``).  Dead (row >= seq_len_q[b]) and
    keyless rows come out as O = 0 / LSE = -inf."""
    import torch

    b, h_q, s_q, _ = q.shape
    h_kv, s_kv = k.shape[1], k.shape[2]
    rep = h_q // h_kv
    kd, vd = k.double().repeat_interleave(rep, 1), v.double().repeat_interleave(rep, 1)
    scores = q.double() @ kd.transpose(-1, -2) * scale
    i = torch.arange(s_q, device=q.device).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=q.device).view(1, 1, 1, s_kv)
    ql, kl = q_lens.to(torch.int64).view(b, 1, 1, 1), kv_lens.to(torch.int64).view(b, 1, 1, 1)
    masked = (i >= ql) | (j >= kl)
    if causal_br:
        diag = i + (kl - ql)
        masked = masked | (j > diag) | (j < diag - window_left)
    scores = scores.masked_fill(masked, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # a row with no live key is -inf
    o = torch.softmax(scores, dim=-1).nan_to_num(0.0) @ vd  # ... and O = 0 there
    return o.float(), lse.float()


def _handoff_execute(api, q, k, v, o, lse, q_lens, kv_lens):
    """Poison the outputs (an unwritten O cell stays NaN, an unwritten LSE +inf,
    both distinct from the legitimate 0 / -inf of a dead row), launch, sync,
    and hand back COPIES -- the retained outputs of this replay."""
    import torch

    o.fill_(float("nan"))
    lse.fill_(float("inf"))
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, seq_q_lens=q_lens, seq_kv_lens=kv_lens)
    torch.cuda.synchronize()
    return o.clone(), lse.clone()


def _handoff_check(o, lse, ref_o, ref_lse, q_lens, *, tag):
    """The retained O / LSE of one replay against the fp64 reference: every
    cell written, dead rows exactly 0 / -inf, the rest at the suite's
    tolerances (the same atol / rtol every dense f16 case in this tree uses)."""
    import torch

    b, h_q, s_q, _ = o.shape
    rows = torch.arange(s_q, device=o.device).view(1, 1, s_q, 1)
    dead = rows >= q_lens.to(torch.int64).view(b, 1, 1, 1)
    bad = ~torch.isfinite(o.float())
    assert not bad.any(), f"{tag}: {int(bad.sum())} non-finite / unwritten O cells, first at (b, h, row, col) = {tuple(bad.nonzero()[0].tolist())}"
    bad = torch.isnan(lse) | torch.isposinf(lse)
    assert not bad.any(), f"{tag}: {int(bad.sum())} NaN / unwritten LSE cells, first at (b, h, row) = {tuple(bad.nonzero()[0].tolist())}"
    assert (o[dead.expand_as(o)] == 0).all(), f"{tag}: rows at / past seq_len_q[b] must be EXACTLY 0 (a select, not residue * 0)"
    assert torch.isneginf(lse[dead.squeeze(-1).expand_as(lse)]).all(), f"{tag}: rows at / past seq_len_q[b] publish LSE = -inf"
    torch.testing.assert_close(o.float(), ref_o, **_GATE_O_TOL, msg=lambda m: f"{tag}: O vs the fp64 reference -- {m}")
    torch.testing.assert_close(lse, ref_lse, **_GATE_LSE_TOL, msg=lambda m: f"{tag}: LSE vs the fp64 reference -- {m}")


_HANDOFF_MASKS = {"dense": dict(causal_br=False, window_left=None), "causal_br_swa129": dict(causal_br=True, window_left=_HANDOFF_WINDOW_LEFT)}


_HANDOFF_CASES = [
    pytest.param("bf16", "dense", 0.5, id="bf16-dense-scale0.5"),
    pytest.param("bf16", "causal_br_swa129", 0.5, id="bf16-causal_br_swa129-scale0.5"),
    pytest.param("fp16", "dense", 0.5, id="fp16-dense-scale0.5"),
    pytest.param("fp16", "causal_br_swa129", 0.5, id="fp16-causal_br_swa129-scale0.5"),
    # scale 1: the 130-key window of the rows past bottom-right diag 257 excludes KV tile 0 -- a fully-masked tile AHEAD of the
    # row's first live key.  The scaled chain took the sentinel * log2 e = -inf as the running max and read -inf - (-inf) = NaN
    # into P (3064 NaN rows, LSE = log(1e-30)); the running-max step now keeps such a tile out of the state (alpha = 1, P = 0).
    # (These cells lower onto the 2x2 twin while api_dsl.D512_2X2 is on; the role-split pin is the dedicated cell below.)
    pytest.param("bf16", "causal_br_swa129", 1.0, id="bf16-causal_br_swa129-scale1"),
    pytest.param("fp16", "causal_br_swa129", 1.0, id="fp16-causal_br_swa129-scale1"),
]


@pytest.mark.parametrize("dtype, mask, scale", _HANDOFF_CASES)
def test_sm107_d512_correction_handoff_under_mixed_rescale(dtype, mask, scale):
    """Rubin e2e for the pipelined O store's correction hand-off (see the
    section comment): the +1 / 0 / -1 row groups make some correction warps
    rescale at every KV tile while their neighbours pass through, a queryless
    batch and (masked arm) keyless rows exercise the dead-row selects, and the
    outputs are retained across three replays with a changed V.  The reference
    composes the same padding, diagonal and window as the graph.  The scale-1
    masked cells are the leading-dead-tile regression: rows whose window excludes
    KV tile 0 keep their later keys and must come out finite and at the reference."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the sm107 d512 f16 / bf16 kernel runs on cc10.7 only")
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    arm = _HANDOFF_MASKS[mask]
    q, k, v, q_lens, kv_lens = _handoff_problem(dt)
    g = _HANDOFF_GEOMETRY
    o = torch.empty(g["b"], g["s_q"], g["h_q"], g["d"], device="cuda", dtype=dt).transpose(1, 2)
    lse = torch.empty(g["b"], g["h_q"], g["s_q"], device="cuda", dtype=torch.float32)
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse,
        is_causal=arm["causal_br"],
        causal_bottom_right=arm["causal_br"],
        window_size_left=arm["window_left"],
        scale_softmax=scale,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
    )
    assert api.check_support()
    api.compile()  # the one capture; everything below is a replay
    ref_kw = dict(scale=scale, causal_br=arm["causal_br"], window_left=arm["window_left"], q_lens=q_lens, kv_lens=kv_lens)

    o0, lse0 = _handoff_execute(api, q, k, v, o, lse, q_lens, kv_lens)
    ref_o, ref_lse = _handoff_reference(q, k, v, **ref_kw)
    _handoff_check(o0, lse0, ref_o, ref_lse, q_lens, tag="launch 0")
    for replay in range(1, _HANDOFF_REPLAYS + 1):
        v.copy_(_handoff_fresh_v(dt, seed=replay))  # a changed V, same storage (what a captured graph sees)
        o_k, lse_k = _handoff_execute(api, q, k, v, o, lse, q_lens, kv_lens)
        ref_o, _ = _handoff_reference(q, k, v, **ref_kw)
        _handoff_check(o_k, lse_k, ref_o, ref_lse, q_lens, tag=f"replay {replay}")
        assert torch.equal(lse_k, lse0), f"replay {replay}: LSE must be bitwise independent of V"
    o_again, lse_again = _handoff_execute(api, q, k, v, o, lse, q_lens, kv_lens)
    assert torch.equal(o_again, o_k) and torch.equal(lse_again, lse_k), "two replays on identical inputs must be bitwise equal (a race otherwise)"


@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (64, 64), (184, 120)])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("stats,stats_log2", [(False, False), (True, False), (True, True)])
def test_sm107_half_split_prepared_rebind_capture(d, dv, dtype_name, stats, stats_log2, monkeypatch, cudnn_handle):
    """Prepared split writes current O/Stats through caller-owned workspace under replay."""
    import math

    import torch
    import cudnn
    import cutlass.cute as cute
    from cudnn.sdpa.fwd import prepared
    from test_sdpa_prepared_thd import _dense_graph

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 required")
    dtype = getattr(torch, dtype_name)
    fe_dtype = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    b, h, hk, sq, sk = 2, 8, 2, 17, 128  # three of the four splits are empty
    pitch = dv + 8
    stride = (sq * h * pitch, pitch, h * pitch, 1)
    g, t = _dense_graph(
        b,
        h,
        hk,
        sq,
        sk,
        d,
        d_v=dv,
        causal=False,
        split_kv=4,
        stats=stats,
        stats_log2=stats_log2,
        o_stride=stride,
        dtype=fe_dtype,
        arch="sm107",
    )
    plan = g._compiled_plans[g._plan_index]
    assert isinstance(plan._prepared, prepared.PreparedDenseLaunch)
    assert plan._prepared.spec.combine is not None
    workspaces = [torch.full((g.get_workspace_size(),), 255, device="cuda", dtype=torch.uint8) for _ in range(2)]
    stream = torch.cuda.Stream()
    old_stream = cudnn.get_stream(cudnn_handle)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared split must not fall back to the adapter or compile during execute")

    monkeypatch.setattr(plan._compiled, "execute_resolved", forbidden)
    monkeypatch.setattr(cute, "compile", forbidden)
    torch.manual_seed(107192)
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        for ws in workspaces:
            q, k, v = [
                (torch.randn(b, s, heads, width, device="cuda") * 0.5).to(dtype).transpose(1, 2) for s, heads, width in ((sq, h, d), (sk, hk, d), (sk, hk, dv))
            ]
            backing = torch.full((b, sq, h, pitch), 42.0, device="cuda", dtype=dtype)
            out = backing[..., :dv].transpose(1, 2)
            lse_backing = torch.full((b, h, sq, 2), 12345.0, device="cuda")
            lse = lse_backing[..., :1]
            pack = {t[n]: x for n, x in zip(("q", "k", "v", "o"), (q, k, v, out))}
            if stats:
                pack[t["stats"]] = lse

            def check():
                scores = q.double() @ k.double().repeat_interleave(h // hk, 1).transpose(-1, -2) / math.sqrt(d)
                ref = scores.softmax(-1) @ v.double().repeat_interleave(h // hk, 1)
                torch.testing.assert_close(out.double(), ref, atol=5e-3, rtol=3e-2)
                if stats:
                    ref_lse = scores.logsumexp(-1) * (math.log2(math.e) if stats_log2 else 1.0)
                    torch.testing.assert_close(lse.squeeze(-1).double(), ref_lse, atol=1e-4, rtol=1e-4)
                assert (backing[..., dv:] == 42).all()
                assert (lse_backing[..., 1] == 12345).all()

            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            mode = torch.cuda.get_sync_debug_mode()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            torch.cuda.set_sync_debug_mode("error")
            try:
                g.execute(pack, ws, handle=cudnn_handle)
            finally:
                torch.cuda.set_sync_debug_mode(mode)
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            stream.synchronize()
            check()
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                g.execute(pack, ws, handle=cudnn_handle)
            q.mul_(0.5)
            v.mul_(1.5)
            ws.fill_(255)
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                capture.replay()
            stream.synchronize()
            check()
            capture.reset()
    finally:
        cudnn.set_stream(cudnn_handle, old_stream)


@pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("mask,sq,skv", [("causal", 17, 385), ("bottom_right", 129, 2049), ("window", 513, 1025)])
def test_sm107_half_split_masked_tails(d, dtype_name, mask, sq, skv):
    """Split bounds include mask position, tail tiles and completely dead partitions."""
    import math

    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 required")
    torch.manual_seed(107192)
    dtype = getattr(torch, dtype_name)
    b, h, hk, dv = 2, 16, 4, 128
    q, k, v = [
        (torch.randn(b, s, heads, width, device="cuda") * 0.5).to(dtype).transpose(1, 2) for s, heads, width in ((sq, h, d), (skv, hk, d), (skv, hk, dv))
    ]
    out = torch.full((b, sq, h, dv), float("nan"), device="cuda", dtype=dtype).transpose(1, 2)
    lse = torch.full((b, h, sq), float("nan"), device="cuda")
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=out,
        sample_lse=lse,
        pack_gqa=False,
        split_kv=8,
        cga=2,
        is_causal=True,
        causal_bottom_right=mask != "causal",
        window_size_left=255 if mask == "window" else None,
        scale_softmax=d**-0.5,
        stats_log2=True,
    )
    assert api.check_support()
    api.compile()
    ws = torch.full((api.scratch_workspace_bytes(),), 255, device="cuda", dtype=torch.uint8)
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=out, lse_tensor=lse, workspace=ws)
    scores = q.double() @ k.double().repeat_interleave(h // hk, 1).transpose(-1, -2) / math.sqrt(d)
    rows = torch.arange(sq, device="cuda")[:, None] + (skv - sq if mask != "causal" else 0)
    cols = torch.arange(skv, device="cuda")[None, :]
    masked = cols > rows
    if mask == "window":
        masked |= cols < rows - 255
    scores.masked_fill_(masked, -float("inf"))
    ref = scores.softmax(-1) @ v.double().repeat_interleave(h // hk, 1)
    torch.testing.assert_close(out.double(), ref, atol=5e-3, rtol=3e-2)
    torch.testing.assert_close(lse.double(), scores.logsumexp(-1) * math.log2(math.e), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("stats_log2", [False, True])
def test_sm107_half_split_direct_template_stats_base(d, stats_log2):
    """A directly compiled split template always writes natural-log partial Stats."""
    import math

    import torch
    import cuda.bindings.driver as cuda
    from cudnn.frost.compiled_cache import positional_entry
    from cudnn.sdpa.fwd.config_sm100 import DTYPE_FP16

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 required")
    b, h, sq, skv, dv, splits = 1, 2, 17, 512, 128, 4
    q = torch.full((b, sq, h, d), 0.5, device="cuda", dtype=torch.float16)
    chunks = torch.arange(skv, device="cuda") // 128 + 1
    k = (chunks.view(1, skv, 1, 1) * 0.5).expand(b, skv, 1, d).half().contiguous()
    v = (chunks.view(1, skv, 1, 1) * 0.125).expand(b, skv, 1, dv).half().contiguous()
    partial_o = torch.full((splits * b, sq, h, dv), float("nan"), device="cuda")
    partial_lse = torch.full((splits * b, h, sq), float("nan"), device="cuda")
    module = _load((d, dv), rubin=True, dtype_qkv=DTYPE_FP16, dtype_o=DTYPE_FP16, split_kv=splits, stats_log2=stats_log2)
    raw = positional_entry(module.compile(d_qk=d, d_v=dv))
    assert raw is not None
    # This bypasses the adapter, which clears stats_log2 when producing partials.
    raw(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        partial_o.data_ptr(),
        partial_lse.data_ptr(),
        0,
        0,
        0,
        (b, h, 1, sq, skv, 0),
        tuple(q.stride()[:3]),
        tuple(k.stride()[:3]),
        tuple(v.stride()[:3]),
        tuple(partial_o.stride()[:3]),
        tuple(partial_lse.stride()),
        0,
        math.log2(math.e) / math.sqrt(d),
        0,
        0,
        None,
        None,
        None,
        partial_o.data_ptr(),
        None,
        None,
        (0, 0),
        0,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    scores = torch.einsum("bqhd,bknd->bhqk", q.double(), k.double()) / math.sqrt(d)
    values = v.double().transpose(1, 2).expand(b, h, skv, dv)
    for split in range(splits):
        lo, hi = split * 128, (split + 1) * 128
        s = scores[..., lo:hi]
        torch.testing.assert_close(partial_lse[split : split + 1].double(), s.logsumexp(-1), atol=2e-4, rtol=2e-5)
        torch.testing.assert_close(partial_o[split : split + 1].double(), (s.softmax(-1) @ values[..., lo:hi, :]).transpose(1, 2), atol=3e-3, rtol=3e-3)


def _mxfp8_prefold_inputs(b, hq, hkv, s, d_qk, d_v, prefold_scale, *, fp8_dtype=None, s_kv=None):
    """Random inputs quantized to MXFP8 the way the engine consumes them, plus the DEQUANTIZED fp32 copies the
    oracle must see.  ``prefold_scale`` multiplies Q BEFORE quantization (the softmax_scale_prefolded contract).
    ``fp8_dtype`` picks the FP8 member (e4m3 default / e5m2); ``s_kv`` (default ``s``) is the K / V length."""
    import torch
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    dev = "cuda"
    fp8_dtype = fp8_dtype or torch.float8_e4m3fn
    s_kv = s if s_kv is None else s_kv
    qf = torch.randn(b, hq, s, d_qk, device=dev) * 0.5
    kf = torch.randn(b, hkv, s_kv, d_qk, device=dev) * 0.5
    vf = torch.randn(b, hkv, s_kv, d_v, device=dev) * 0.5

    def mx(x, h, n, d, columnwise):
        # quantize_to_mxfp8 returns the FP8 data and the per-element DEQUANT SCALE (dq); dequantized = data * dq.
        data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(x.contiguous(), b, h, n, d, 32, fp8_dtype, with_ref=True)
        data, dq, swz = (data_s, dq_s, swz_s) if columnwise else (data_d, dq_d, swz_d)
        dequant = data.double() * dq.double().reshape(b, h, n, d)  # float64 oracle operand
        return data.permute(0, 2, 1, 3).contiguous().transpose(1, 2), swz.contiguous(), dequant  # BHSD view over BSHD storage

    q8, sfq, dq = mx(qf * prefold_scale, hq, s, d_qk, False)
    k8, sfk, dk = mx(kf, hkv, s_kv, d_qk, False)
    v8, sfv, dv = mx(vf, hkv, s_kv, d_v, True)
    return (q8, sfq, dq), (k8, sfk, dk), (v8, sfv, dv)


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False])
@pytest.mark.parametrize("causal, b, hq, hkv, s", [(False, 1, 8, 2, 1024), (True, 2, 16, 4, 2048)])
@pytest.mark.parametrize("precision, prefolded", [("half", False), ("half", True), ("float", True)])
def test_mxfp8_half_softmax_and_prefolded_scale_match_the_oracle(precision, prefolded, causal, b, hq, hkv, s, with_stats):
    """cc10.7 e2e for the two softmax levers of the d128 MXFP8 kernel: softmax_precision=HALF (MUFU EX2.F16x2
    + f16x2 -> FP8 cast, ported from the per-tensor FP8 sibling) and softmax_scale_prefolded (Q carries
    attn_scale * log2(e); the kernel skips the per-score FFMA2 and, with HALF, fuses shift + convert into one
    FHADD2).  The stats-less leg is the one that traces the fused arm (the Stats specialization keeps the
    shifted f32 scores for the exact LSE denominator).  Each variant must stay within the oracle bound of the
    DEQUANTIZED inputs it actually saw (float64 oracle), write every O cell and LSE row, and publish an LSE
    within 1e-4 (natural log) of the oracle's (the exact f32 denominator measures ~1e-6; an f16 pair-sum would read 5e-4..8e-4) -- the prefolded contract keeps the Stats in the same domain.
    """
    import math

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 MXFP8 kernels serve cc10.7 only")
    from cudnn import data_type as cudnn_dtype
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    d_qk = d_v = 128
    attn_scale = d_qk**-0.5
    torch.manual_seed(0)
    dev = "cuda"
    (q8, sfq, dq), (k8, sfk, dk), (v8, sfv, dv) = _mxfp8_prefold_inputs(b, hq, hkv, s, d_qk, d_v, attn_scale * math.log2(math.e) if prefolded else 1.0)
    out = torch.full((b, s, hq, d_v), float("nan"), device=dev, dtype=torch.bfloat16).transpose(1, 2)  # sentinel: an unclaimed tile stays visible
    lse = torch.full((b, hq, s), float("nan"), device=dev, dtype=torch.float32)
    api = SdpaFwdDslSm100(
        q8,
        k8,
        v8,
        out,
        lse if with_stats else None,
        scale_softmax=None if prefolded else attn_scale,
        is_causal=causal,
        pertensor_fp8=False,
        dtype_o=torch.bfloat16,
        cga=2,
        softmax_precision=cudnn_dtype.HALF if precision == "half" else None,
        softmax_scale_prefolded=prefolded,
    )
    assert api.check_support()
    api.compile()
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    api.execute(q8, k8, v8, out, lse_tensor=lse if with_stats else None, sf_q=sfq, sf_k=sfk, sf_v=sfv, workspace=ws)
    torch.cuda.synchronize()
    if precision == "half" and prefolded and not with_stats:
        from cutlass._mlir.dialects import nvvm as nvvm_ops

        # the stats-less HALF + prefolded build is the fused FHADD2 arm whenever the DSL exposes the op
        from cudnn.frost.tile_dsl import softmax_f16 as _sf16

        assert bool(api._k_mod._FUSED_SHIFT_CVT) == _sf16.FUSED_SHIFT_CVT_AVAILABLE  # the op AND its result-type-first builder form

    # Oracle (float64) on the dequantized inputs: a prefolded Q already carries attn_scale*log2(e), so its logits are
    # exp2-domain -> softmax_e(ln2 * S); otherwise softmax_e(attn_scale * S).
    rep = hq // hkv
    logits = (dq @ dk.repeat_interleave(rep, 1).transpose(-1, -2)) * (math.log(2.0) if prefolded else attn_scale)
    if causal:
        logits = logits.masked_fill(~torch.tril(torch.ones(s, s, dtype=torch.bool, device=dev)), float("-inf"))
    ref = torch.softmax(logits, dim=-1) @ dv.repeat_interleave(rep, 1)
    assert torch.isfinite(out).all(), "non-finite / unwritten O cells"
    scale = ref.abs().max().item()
    err = (out.double() - ref).abs().max().item()
    assert err <= 0.1 * scale, f"max err {err} vs oracle (scale {scale})"
    if with_stats:
        ref_lse = torch.logsumexp(logits, dim=-1)
        assert torch.isfinite(lse).all(), "unwritten LSE rows"
        lse_err = (lse.double() - ref_lse).abs().max().item()
        assert lse_err <= 1e-4, f"LSE max err {lse_err} vs oracle (natural log)"


def _mxfp8_lever_reference(dq, dk, dv, hq, hkv, logit_scale, masked=None):
    """float64 softmax(logit_scale * Q K^T) V and the natural-log LSE on the DEQUANTIZED operands the kernel saw;
    ``masked`` ([s_q, s_kv] bool) drops keys, and a row left without any comes out as O = 0 / LSE = -inf (the
    kernels' keyless-row convention)."""
    import torch

    rep = hq // hkv
    logits = (dq @ dk.repeat_interleave(rep, 1).transpose(-1, -2)) * logit_scale
    if masked is not None:
        logits = logits.masked_fill(masked, float("-inf"))
    ref_o = torch.softmax(logits, dim=-1).nan_to_num(0.0) @ dv.repeat_interleave(rep, 1)
    return ref_o, torch.logsumexp(logits, dim=-1)


def _run_mxfp8_d192x128_levers(q8, k8, v8, sfq, sfk, sfv, *, precision, prefolded, with_stats, attn_scale, **mask_kw):
    """Build, compile and launch the (192, 128) MXFP8 flavor with the requested softmax levers (cga2: the only CGA the
    flavor serves); O / LSE are NaN-poisoned first so an unwritten cell stays visible.  Pins the module constants the
    build must carry before launching: the d192x128 kernel served it, SOFTMAX_F16 / SCALE_PREFOLDED follow the request,
    and _FUSED_SHIFT_CVT is set exactly on the HALF + prefolded build when the helper module reports the fused op (a
    DSL without it falls back to the unfused f16 arm, visibly).  Returns (api, out, lse)."""
    import torch

    from cudnn import data_type as cudnn_dtype
    from cudnn.frost.tile_dsl import softmax_f16 as _sf16
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, hq, s_q, _ = q8.shape
    d_v = v8.shape[-1]
    dev = q8.device
    out = torch.full((b, s_q, hq, d_v), float("nan"), device=dev, dtype=torch.bfloat16).transpose(1, 2)
    lse = torch.full((b, hq, s_q), float("nan"), device=dev, dtype=torch.float32)
    api = SdpaFwdDslSm100(
        q8,
        k8,
        v8,
        out,
        lse if with_stats else None,
        scale_softmax=None if prefolded else attn_scale,
        pertensor_fp8=False,
        dtype_o=torch.bfloat16,
        cga=2,
        softmax_precision=cudnn_dtype.HALF if precision == "half" else None,
        softmax_scale_prefolded=prefolded,
        **mask_kw,
    )
    assert api.check_support()
    api.compile()
    k_mod = api._k_mod
    assert api.kernel_template == "prefill_d192_d128_mxfp8" and k_mod.CFG.TILE_K == 192 and k_mod.CFG.TILE_O == 128, api.kernel_template
    assert bool(k_mod.SOFTMAX_F16) == (precision == "half") and bool(k_mod.SCALE_PREFOLDED) == prefolded
    assert bool(k_mod._FUSED_SHIFT_CVT) == (precision == "half" and prefolded and _sf16.FUSED_SHIFT_CVT_AVAILABLE)
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    api.execute(q8, k8, v8, out, lse_tensor=lse if with_stats else None, sf_q=sfq, sf_k=sfk, sf_v=sfv, workspace=ws)
    torch.cuda.synchronize()
    return api, out, lse


# (precision, prefolded) arms of the (192, 128) MXFP8 port: HALF alone (the unfused f16x2 exponent), HALF + fold (the
# fused FHADD2 arm on the stats-less build, the unfused one with Stats), FLOAT + fold (raw max, plain subtract).
_D192_MXFP8_LEVER_ARMS = [("half", False), ("half", True), ("float", True)]
_D192_MXFP8_LEVER_SHAPES = [
    pytest.param("float8_e4m3fn", False, 1, 8, 2, 1024, id="e4m3-dense"),
    pytest.param("float8_e4m3fn", True, 2, 16, 4, 2048, id="e4m3-causal"),
    pytest.param("float8_e5m2", True, 1, 8, 2, 1024, id="e5m2-causal"),  # the e5m2 pair tag of the f16x2 -> FP8 cast
]


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False])
@pytest.mark.parametrize("in_dtype, causal, b, hq, hkv, s", _D192_MXFP8_LEVER_SHAPES)
@pytest.mark.parametrize("precision, prefolded", _D192_MXFP8_LEVER_ARMS)
def test_mxfp8_d192x128_half_softmax_and_prefolded_scale_match_the_oracle(precision, prefolded, in_dtype, causal, b, hq, hkv, s, with_stats):
    """The (192, 128) twin of test_mxfp8_half_softmax_and_prefolded_scale_match_the_oracle: the d192x128 MXFP8 kernel
    carries the same two levers (its softmax body is the d128 sibling's), so every arm -- HALF, HALF + prefolded (the
    fused FHADD2 when stats-less), FLOAT + prefolded -- with and without Stats, dense and causal, must land within the
    oracle bound of the DEQUANTIZED inputs (float64), write every O cell / LSE row, and keep the LSE within 1e-4 natural
    on the exact-sum (Stats) legs.  e5m2 inputs exercise the e5m2 pair tag of the f16x2 -> FP8 cast (a wrong tag or a
    swapped half-word is a wrong O, not a crash).  The prefold multiplies the f32 Q by attn_scale * log2(e) BEFORE
    block quantization and the oracle then uses ln 2 as the logit scale."""
    import math

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 MXFP8 kernels serve cc10.7 only")

    d_qk, d_v = 192, 128
    attn_scale = d_qk**-0.5
    torch.manual_seed(0)
    (q8, sfq, dq), (k8, sfk, dk), (v8, sfv, dv) = _mxfp8_prefold_inputs(
        b, hq, hkv, s, d_qk, d_v, attn_scale * math.log2(math.e) if prefolded else 1.0, fp8_dtype=getattr(torch, in_dtype)
    )
    _, out, lse = _run_mxfp8_d192x128_levers(
        q8, k8, v8, sfq, sfk, sfv, precision=precision, prefolded=prefolded, with_stats=with_stats, attn_scale=attn_scale, is_causal=causal
    )
    masked = ~torch.tril(torch.ones(s, s, dtype=torch.bool, device=out.device)) if causal else None
    ref, ref_lse = _mxfp8_lever_reference(dq, dk, dv, hq, hkv, math.log(2.0) if prefolded else attn_scale, masked)
    assert torch.isfinite(out).all(), "non-finite / unwritten O cells"
    scale = ref.abs().max().item()
    err = (out.double() - ref).abs().max().item()
    assert err <= 0.1 * scale, f"max err {err} vs oracle (scale {scale})"
    if with_stats:
        assert torch.isfinite(lse).all(), "unwritten LSE rows"
        lse_err = (lse.double() - ref_lse).abs().max().item()
        assert lse_err <= 1e-4, f"LSE max err {lse_err} vs oracle (natural log)"


# Keyless rows INSIDE a live CGA tile (CGA_TILE_M = 512 rows under cga2; s_q = 1024, s_kv = 768).  The kv-loop bounds
# are per CGA tile, so these rows run the softmax body on KV tiles that are fully masked for them while their
# neighbours are live:
#   top-left causal + left window 192: rows >= 960 have their band past the last key (767); CGA 1 (rows 512-1023)
#     visits KV tiles 2..5, all four fully masked for those 64 rows;
#   bottom-right causal with s_kv < s_q: rows < 256 sit above the diagonal; CGA 0 visits KV tiles 0-1, both fully
#     masked for those 256 rows.
_D192_MXFP8_KEYLESS_GEOMETRIES = [
    pytest.param(False, 192, 64, id="topleft-swa192"),
    pytest.param(True, None, 256, id="bottomright"),
]
_D192_MXFP8_KEYLESS_ARMS = [("half", True, False), ("half", True, True), ("float", True, True)]


@pytest.mark.L0
@pytest.mark.parametrize("causal_br, window_left, n_keyless", _D192_MXFP8_KEYLESS_GEOMETRIES)
@pytest.mark.parametrize("precision, prefolded, with_stats", _D192_MXFP8_KEYLESS_ARMS)
def test_mxfp8_d192x128_prefolded_scale_keeps_the_keyless_row_select(precision, prefolded, with_stats, causal_br, window_left, n_keyless):
    """Under the pre-folded scale a fully-masked KV tile leaves the raw row max exactly at the finite mask sentinel
    (== NEG_INF); the running-max step selects such a tile out of the row's state on every consecutive keyless tile
    (total_max kept at the sentinel, alpha = 1, a shift of 0 -> P = 0, also through the fused FHADD2 arm), so a keyless
    row ends at (NEG_INF, 0).  Those rows must be overridden by the correction warp's keyless geometry select -- O exactly
    0, LSE exactly -inf -- with every live row of the same CGA tile still at the oracle (no NaN residue).  Both keyless shapes the dense d192x128 kernel can form, on the fused build, the
    Stats-unfused HALF build and the FLOAT fold build."""
    import math

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 MXFP8 kernels serve cc10.7 only")

    b, hq, hkv, s_q, s_kv, d_qk, d_v = 1, 4, 2, 1024, 768, 192, 128
    attn_scale = d_qk**-0.5
    torch.manual_seed(0)
    (q8, sfq, dq), (k8, sfk, dk), (v8, sfv, dv) = _mxfp8_prefold_inputs(b, hq, hkv, s_q, d_qk, d_v, attn_scale * math.log2(math.e), s_kv=s_kv)
    _, out, lse = _run_mxfp8_d192x128_levers(
        q8,
        k8,
        v8,
        sfq,
        sfk,
        sfv,
        precision=precision,
        prefolded=prefolded,
        with_stats=with_stats,
        attn_scale=attn_scale,
        is_causal=True,
        causal_bottom_right=causal_br,
        window_size_left=window_left,
    )
    i = torch.arange(s_q, device=out.device).view(s_q, 1)
    j = torch.arange(s_kv, device=out.device).view(1, s_kv)
    diag = i + (s_kv - s_q) if causal_br else i
    masked = j > diag
    if window_left is not None:
        masked = masked | (j < diag - window_left)
    keyless = masked.all(dim=1)
    assert int(keyless.sum()) == n_keyless, "the geometry must form exactly the keyless rows this case is about"
    live = ~keyless
    ref, ref_lse = _mxfp8_lever_reference(dq, dk, dv, hq, hkv, math.log(2.0), masked)
    assert torch.isfinite(out).all(), "non-finite / unwritten O cells"
    assert (out[:, :, keyless] == 0).all(), "keyless rows must be EXACTLY 0 (a select, not residue * 0)"
    scale = ref[:, :, live].abs().max().item()
    err = (out[:, :, live].double() - ref[:, :, live]).abs().max().item()
    assert err <= 0.1 * scale, f"max err {err} vs oracle on the live rows (scale {scale})"
    if with_stats:
        assert torch.isneginf(lse[:, :, keyless]).all(), "keyless rows publish LSE = -inf"
        assert torch.isfinite(lse[:, :, live]).all(), "unwritten LSE rows"
        lse_err = (lse[:, :, live].double() - ref_lse[:, :, live]).abs().max().item()
        assert lse_err <= 1e-4, f"LSE max err {lse_err} vs oracle (natural log)"


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False], ids=["stats", "nostats"])
@pytest.mark.parametrize(
    "precision, prefolded", [("float", False), ("half", False), ("half", True), ("float", True)], ids=["float", "half", "half+fold", "float+fold"]
)
def test_mxfp8_d192x128_masked_leading_tile_keeps_rows_with_later_keys_finite(precision, prefolded, with_stats):
    """A row whose FIRST KV tile is fully masked while a LATER tile holds its keys, on the d192x128 MXFP8 kernel (the d128
    MXFP8 body): top-left causal with a 34-key band at S = 256 (rows 161..255: no key in tile 0, 34 keys in tile 1) at
    attn_scale 1, on both chains and both exponent arms.  The scaled chain clamps the tile max to the finite sentinel and the
    fold keeps the raw one, so both took the sentinel as the running max and published P = 1 per masked column (also through
    the fused FHADD2 arm), wiped only by alpha = 0 at the next live tile; the running-max step now selects the dead tile out of
    the state (total_max kept, alpha = 1, P = 0).  All-ones operands: a flat softmax (O = V exactly) on every live row, the
    float64 oracle of the dequantized inputs each chain saw (Stats within 1e-4 natural)."""
    import math
    from unittest.mock import patch

    import torch

    _requires_cc107()
    b, hq, hkv, s, d_qk, d_v = 1, 8, 2, 256, 192, 128
    attn_scale = 1.0
    with patch.object(torch, "randn", side_effect=lambda *a, **k: torch.ones(*a, **k)):
        (q8, sfq, dq), (k8, sfk, dk), (v8, sfv, dv) = _mxfp8_prefold_inputs(b, hq, hkv, s, d_qk, d_v, attn_scale * math.log2(math.e) if prefolded else 1.0)
    _, out, lse = _run_mxfp8_d192x128_levers(
        q8, k8, v8, sfq, sfk, sfv, precision=precision, prefolded=prefolded, with_stats=with_stats, attn_scale=attn_scale, is_causal=True, window_size_left=33
    )
    masked = _dense_mask(s, s, causal=True, window_left=33)
    ref, ref_lse = _mxfp8_lever_reference(dq, dk, dv, hq, hkv, math.log(2.0) if prefolded else attn_scale, masked)
    assert torch.isfinite(ref_lse).all(), "geometry: every row keeps 34 keys"
    assert torch.isfinite(out).all(), f"{int((~torch.isfinite(out)).sum())} non-finite / unwritten O cells"
    scale = ref.abs().max().item()
    err = (out.double() - ref).abs().max().item()
    assert err <= 0.1 * scale, f"O max err {err} vs oracle (scale {scale})"
    if with_stats:
        assert torch.isfinite(lse).all(), f"{int((~torch.isfinite(lse)).sum())} non-finite / unwritten LSE rows"
        lse_err = (lse.double() - ref_lse).abs().max().item()
        assert lse_err <= 1e-4, f"LSE max err {lse_err} vs oracle (natural log)"


@pytest.mark.L0
def test_softmax_lever_config_backstops_follow_the_flavor_tables():
    """Config-level backstop (no GPU): every cc10.7 flavor accepts softmax_f16 / softmax_scale_prefolded exactly when
    config_sm107's flavor tables say its kernel body carries the arm.  Reaching a kernel without the arm would trace
    the default chain under the request -- a wrong P scaling or a silent no-op, with no crash."""
    from cudnn.sdpa.fwd import config_sm107 as c

    e4m3, bf16, fp16 = 0, 2, 3
    factories = [
        ("sm107 d128", c.make_cfg_d128, {}),
        ("sm107 d192xd128", c.make_cfg_d192, {}),
        ("sm107 d256", c.make_cfg_d256, dict(cta_mma=2)),
        ("sm107 d512", c.make_cfg_d512, dict(cta_mma=2)),
        ("sm107 d128 mxfp8", c.make_cfg_d128_mxfp8, {}),
        ("sm107 d192xd128 mxfp8", c.make_cfg_d192_mxfp8, {}),
        ("sm107 d256 mxfp8", c.make_cfg_d256_mxfp8, dict(cta_mma=1)),
        ("sm107 d512 mxfp8", c.make_cfg_d512_mxfp8, dict(cta_mma=2)),
    ]
    accepted = set()
    for flavor, make, extra in factories:
        mxfp8 = flavor.endswith(" mxfp8")
        for dtype in ((e4m3,) if mxfp8 else (e4m3, bf16, fp16)):
            quantized = dtype == e4m3
            for f16, prefolded in ((True, False), (False, True), (True, True)):
                wired_f16 = quantized and flavor in c.SM107_SOFTMAX_F16_FLAVORS
                wired_fold = flavor in c.SM107_SCALE_PREFOLDED_FLAVORS and (mxfp8 or not quantized)
                params = c.TemplateParams(dtype_qkv=dtype, dtype_o=bf16 if quantized else dtype, softmax_f16=f16, softmax_scale_prefolded=prefolded, **extra)
                if (not f16 or wired_f16) and (not prefolded or wired_fold):
                    make(params)
                    accepted.add((flavor, dtype, f16, prefolded))
                else:
                    with pytest.raises(ValueError, match="softmax_f16|softmax_scale_prefolded"):
                        make(params)
    # Per-tensor FP8 never takes the fold (the kernel folds descale_q * descale_k into the softmax scale); half
    # inputs never take the f16 exponent; the MXFP8 flavors take both; the half flavors take the fold.
    assert ("sm107 d128", e4m3, False, True) not in accepted and ("sm107 d128", fp16, True, False) not in accepted
    assert all((f, e4m3, True, True) in accepted for f in c.SM107_SOFTMAX_F16_FLAVORS if f.endswith(" mxfp8"))
    assert all((f, bf16, False, True) in accepted for f in c.SM107_SCALE_PREFOLDED_FLAVORS if not f.endswith(" mxfp8"))
    # The 2x2 twin: the fold is its own arm; the f16 exponent never (half inputs).
    c.make_cfg_d512_2x2(c.TemplateParams(dtype_qkv=bf16, dtype_o=bf16, cta_mma=2, mma_2x2=True, softmax_scale_prefolded=True))
    with pytest.raises(ValueError, match="softmax_f16"):
        c.make_cfg_d512_2x2(c.TemplateParams(dtype_qkv=bf16, dtype_o=bf16, cta_mma=2, mma_2x2=True, softmax_f16=True))
    # The cc 10.0 / 10.3 line still serves neither (its kernels apply the scale in-kernel).
    from cudnn.sdpa.fwd import config_sm100 as c100

    with pytest.raises(ValueError, match="softmax_scale_prefolded"):
        c100.make_cfg_d128(c100.TemplateParams(dtype_qkv=e4m3, dtype_o=bf16, softmax_scale_prefolded=True))


@pytest.mark.parametrize(
    "flavor, extra, fold_sites, fused_sites, raw_max, raw_shift, scaled_max, scaled_shift",
    [
        (
            (192, 128),
            {},
            2,
            1,
            "current_max = raw_max\n",
            "reg_S_a = reg_S_a - new_total_max",
            "current_max = cute.math.max(raw_max * scale_log2, NEG_INF)",
            "reg_S_a = reg_S_a * scale_log2 - new_total_max",
        ),
        (
            (512, 512),
            dict(cta_mma=2),
            3,
            2,
            "current_max = current_max_raw\n",
            "reg_S_tile.vec - new_total_max",
            "current_max = cute.math.max(current_max_raw * scale_log2, NEG_INF_F32)",
            "reg_S_tile.vec * scale_log2 - new_total_max",
        ),
    ],
    ids=["d192x128", "d512"],
)
def test_mxfp8_fold_and_fused_arms_are_wired_in_the_body(flavor, extra, fold_sites, fused_sites, raw_max, raw_shift, scaled_max, scaled_shift):
    """Source pin (no GPU) for the d192x128 and d512 MXFP8 kernels: the module constants alone prove nothing -- the
    adapter pins scale_log2 to exactly 1.0 under the fold, so a body left on the scaled chain passes every oracle.
    The BODY must consult SCALE_PREFOLDED at the max site and the shift site (raw max, plain subtract; the scaled
    forms stay on the else arms), and the fused shift+convert must be guarded by ``_FUSED_SHIFT_CVT and not
    has_lse`` (the Stats build keeps the shifted f32 scores for the exact LSE denominator)."""
    mod = _load(flavor, rubin=True, fp8=True, pertensor=False, dtype_qkv=0, dtype_o=2, softmax_f16=True, softmax_scale_prefolded=True, **extra)
    from cudnn.frost.tile_dsl import softmax_f16 as _sf16

    assert mod.SOFTMAX_F16 == 1 and mod.SCALE_PREFOLDED == 1 and mod._FUSED_SHIFT_CVT is _sf16.FUSED_SHIFT_CVT_AVAILABLE
    plain = _load(flavor, rubin=True, fp8=True, pertensor=False, dtype_qkv=0, dtype_o=2, **extra)
    assert plain.SOFTMAX_F16 == 0 and plain.SCALE_PREFOLDED == 0 and plain._FUSED_SHIFT_CVT is False
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    assert code.count("if cutlass.const_expr(SCALE_PREFOLDED):") == fold_sites, "max site + shift site(s) must branch on the fold"
    assert code.count("if cutlass.const_expr(_FUSED_SHIFT_CVT and not has_lse):") == fused_sites, "the fused arm is stats-less only"
    assert raw_max in code and raw_shift in code, "the fold arm: raw running max, plain subtract"
    assert scaled_max in code and scaled_shift in code, "the scaled chain stays on the else arms"
    # the shift operand is the exp2 shift the shared running-max step returns (0 on a tile that is dead ahead of the row's
    # first live key), never the running max itself
    assert "- total_max" not in code.split("def _softmax_kv_body(" if flavor == (192, 128) else "def _sg0_softmax_kv_iter(", 1)[1].split("\ndef ", 1)[0]
    assert re.search(
        r"total_max, alpha, new_total_max = running_max_step_finite_sentinel\(\s*\w+, current_max, total_max, NEG_INF(_F32)?, RESCALE_THRESHOLD(_F32)?, masked=CFG\.MASK_FLAGS != MASK_NONE\s*\)",
        code,
    ), "the running-max step is the shared finite-sentinel helper, called with the RAW tile max"
    assert "fused_shift_f16_exp_chunk" in code or "fused_m=" in code, "the fused arm calls the shared helper"


def test_prefolded_scale_declines_the_single_cta_half_legs_on_the_adapter(monkeypatch):
    """Adapter twin of engines.mismatch's rule (CPU-side, the device pinned to cc 10.7): the half d192x128 THD single-Q
    leg and the d128 packed-split leg load the shared single-CTA body (``_load_sm100_kernel_module``), which applies
    the scale in-kernel, so ``softmax_scale_prefolded`` must DECLINE there (NotImplementedError: the plan walk moves
    on), while the cga2 prefill body of the same flavor accepts it.  Accept side first, so the decline cannot pass by
    the path being dead."""
    import torch

    _fake_cc(monkeypatch, (10, 7))
    common = dict(dtype=torch.float16, with_gate=False, thd=True, seq_kv_lens_present=True, softmax_scale_prefolded=True, scale_softmax=None)
    for (d, d_v), kw in (((192, 128), dict(cga=2)), ((128, 128), dict(cga=2))):
        api = _gate_api(d=d, d_v=d_v, **common, **kw)
        assert api.check_support(), (d, d_v, kw)
        assert api.template_params().softmax_scale_prefolded is True
    for (d, d_v), kw in (((192, 128), dict(cga=1, split_kv=1)), ((128, 128), dict(cga=1, split_kv=2))):
        with pytest.raises(NotImplementedError, match="single-CTA half THD|paged-KV"):
            _gate_api(d=d, d_v=d_v, **common, **kw).check_support()
    # The same legs WITHOUT the fold are served (the decline is about the arm, not the leg).
    for (d, d_v), kw in (((192, 128), dict(cga=1, split_kv=1)), ((128, 128), dict(cga=1, split_kv=2))):
        plain = dict(common, softmax_scale_prefolded=False)
        assert _gate_api(d=d, d_v=d_v, **plain, **kw).check_support(), (d, d_v, kw)


@requires_dsl
def test_rubin_dense_d128_legs_load_the_shared_sm100_bodies():
    """The loader's two dense cc 10.7 arms (issue #1472): dense d128 half at cta_mma=1 -> the shared DECODE tile
    (sm100/decode_d128_f16.py, packed or not), dense packed d128 half at cga2 -> the shared SM100 prefill body, dense
    unpacked cga2 -> the Rubin sibling, and the paged THD cga1 unsplit leg -> the shared prefill body (the two-slab paged
    prefill, never the tile)."""
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16

    def where(**params):
        mod = _load_sm100_kernel_module((128, 128), TemplateParams(dtype_qkv=DTYPE_BF16, **params), rubin=True)
        return os.path.basename(os.path.dirname(mod.__file__)), os.path.basename(mod.__file__)

    assert where(cta_mma=1) == ("sm100", "decode_d128_f16.py")
    assert where(cta_mma=1, pack_gqa=True, qh_per_kh=8) == ("sm100", "decode_d128_f16.py")
    assert where(cta_mma=2, pack_gqa=True, qh_per_kh=8) == ("sm100", "prefill_d128_f16.py")
    assert where(cta_mma=2) == ("sm107", "prefill_d128_f16.py")
    assert where(cta_mma=1, thd_varlen=True, paged_kv=True, page_size=16, seq_kv_lens_present=True, split_kv=1) == ("sm100", "prefill_d128_f16.py")


def test_rubin_shared_dense_legs_decline_the_prefolded_scale_on_the_adapter(monkeypatch):
    """Adapter twins of the dense cc 10.7 legs (CPU-side, the device pinned to cc 10.7): dense d128 half at cga=1 and
    PackGQA at cga2 are ADMITTED (template_params carries the width / packing verbatim) and both DECLINE the pre-folded
    scale (the shared bodies apply the scale in-kernel: a typed NotImplementedError, so the plan walk moves to the Rubin
    cga2 body, which keeps the fold), while THD (ragged) at cga=1 keeps the existing decode-tile decline."""
    import torch

    _fake_cc(monkeypatch, (10, 7))
    common = dict(d=128, d_v=128, dtype=torch.bfloat16, with_gate=False, h=8, h_kv=2, s=128)
    api = _gate_api(**common, cga=1)
    assert api.check_support() and api.template_params().cta_mma == 1
    api = _gate_api(**common, cga=2, pack_gqa=True)
    assert api.check_support() and api.template_params().pack_gqa is True and api.template_params().cta_mma == 2
    for kw in (dict(cga=1), dict(cga=2, pack_gqa=True)):
        with pytest.raises(NotImplementedError, match="apply the scale in-kernel"):
            _gate_api(**common, softmax_scale_prefolded=True, scale_softmax=None, **kw).check_support()
    assert _gate_api(**common, cga=2, softmax_scale_prefolded=True, scale_softmax=None).check_support(), "the Rubin cga2 body keeps the fold"
    with pytest.raises(NotImplementedError, match="decode tile"):
        _gate_api(**common, thd=True, seq_kv_lens_present=True, cga=1, split_kv=1).check_support()


def test_softmax_arms_tag_reads_the_module_constants_and_the_stats_gate():
    """api_dsl.softmax_arms_of is the detector the cc 10.7 test_mhas_v2 sweeps assert against (frost_routing.LAST_ARMS):
    it must name the exponent arm, the fold, and the fused shift+convert ONLY on a stats-less build (the Stats
    specialization keeps the shifted f32 scores for the exact LSE denominator and never traces the fused arm)."""
    from types import SimpleNamespace

    from cudnn.sdpa.fwd.api_dsl import softmax_arms_of

    def mod(**consts):
        return SimpleNamespace(**consts)

    assert softmax_arms_of(mod(), has_lse=False) == "f32"  # a kernel without the constants traces the f32 chain
    assert softmax_arms_of(mod(SOFTMAX_F16=0, SCALE_PREFOLDED=0, _FUSED_SHIFT_CVT=False), has_lse=True) == "f32"
    assert softmax_arms_of(mod(SOFTMAX_F16=1, SCALE_PREFOLDED=0, _FUSED_SHIFT_CVT=False), has_lse=False) == "f16"
    assert softmax_arms_of(mod(SOFTMAX_F16=0, SCALE_PREFOLDED=1, _FUSED_SHIFT_CVT=False), has_lse=False) == "f32+fold"
    assert softmax_arms_of(mod(SOFTMAX_F16=1, SCALE_PREFOLDED=1, _FUSED_SHIFT_CVT=False), has_lse=False) == "f16+fold"
    assert softmax_arms_of(mod(SOFTMAX_F16=1, SCALE_PREFOLDED=1, _FUSED_SHIFT_CVT=True), has_lse=False) == "f16+fold+fused"
    assert softmax_arms_of(mod(SOFTMAX_F16=1, SCALE_PREFOLDED=1, _FUSED_SHIFT_CVT=True), has_lse=True) == "f16+fold"


@pytest.mark.L0
def test_softmax_scale_prefolded_api_rejections():
    """The adapter refuses the combinations the contract forbids: a user-given scale_softmax with the pre-folded flag
    (construction and execute time), and the flag on the per-tensor FP8 path (the arm lives in the MXFP8 kernel only).
    cc10.7 only -- the API resolves the device at construction."""
    import math

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 MXFP8 kernels serve cc10.7 only")
    from cudnn import data_type as cudnn_dtype
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, hq, hkv, s, d = 1, 8, 2, 1024, 128
    (q8, sfq, _), (k8, sfk, _), (v8, sfv, _) = _mxfp8_prefold_inputs(b, hq, hkv, s, d, d, d**-0.5 * math.log2(math.e))
    out = torch.empty((b, s, hq, d), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    kw = dict(is_causal=False, pertensor_fp8=False, dtype_o=torch.bfloat16, cga=2, softmax_precision=cudnn_dtype.HALF)
    with pytest.raises(ValueError, match="softmax_scale_prefolded"):
        SdpaFwdDslSm100(q8, k8, v8, out, None, scale_softmax=d**-0.5, softmax_scale_prefolded=True, **kw).check_support()
    with pytest.raises(ValueError, match="softmax_scale_prefolded"):
        SdpaFwdDslSm100(
            q8.to(torch.float8_e4m3fn), k8, v8, out, None, scale_softmax=None, softmax_scale_prefolded=True, **{**kw, "pertensor_fp8": True}
        ).check_support()
    api = SdpaFwdDslSm100(q8, k8, v8, out, None, scale_softmax=None, softmax_scale_prefolded=True, **kw)
    assert api.check_support()
    api.compile()
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="execute-time scale_softmax"):
        api.execute(q8, k8, v8, out, sf_q=sfq, sf_k=sfk, sf_v=sfv, workspace=ws, scale_softmax=0.5)


# ============================================================================
# softmax_precision=HALF on the per-tensor FP8 d192x128 kernel (cc10.7)
#
# ``sm107/prefill_d192_d128_fp8.py`` carries the f16x2 exponent arm through the shared
# ``cudnn.frost.tile_dsl.softmax_f16`` helpers (``f16_exp_chunk`` stats-less, ``f16_exp_chunk_sum``
# with Stats); the engine row, the adapter gate and the config backstop all admit HALF at (192, 128).
# The pre-folded scale is declined on per-tensor FP8 by contract (the kernel folds
# descale_q * descale_k into the softmax scale in-kernel), so the fused shift+convert arm never
# traces here: ``_FUSED_SHIFT_CVT`` is pinned False.  Float64 oracle on the DEQUANTIZED inputs the
# kernel actually saw; a swapped FP8 half-word, a chunk whose P words were not stored, or a
# mis-biased shift would each push O or the LSE far past these bounds.
# ============================================================================


def _fp8_pertensor_inputs(b, hq, hkv, s_q, s_kv, d_qk, d_v, fp8_dtype):
    """Random inputs quantized per-tensor to ``fp8_dtype`` with amax-derived descales (the way the FP8
    op's callers quantize), as BHSD views over BSHD storage, plus the DEQUANTIZED float64 copies the
    oracle must see.  Returns ``((q8, descale_q, q_f64), (k8, ...), (v8, ...))``."""
    import torch

    dev = "cuda"
    fmax = torch.finfo(fp8_dtype).max

    def quant(x):
        dsc = (x.abs().amax().clamp_min(1e-8) / fmax).item()
        data = (x / dsc).clamp(-fmax, fmax).to(fp8_dtype)
        data = data.transpose(1, 2).contiguous().transpose(1, 2)  # BHSD view over BSHD storage
        return data, torch.full((1,), dsc, device=dev, dtype=torch.float32), data.double() * dsc

    q = quant(torch.randn(b, hq, s_q, d_qk, device=dev) * 0.5)
    k = quant(torch.randn(b, hkv, s_kv, d_qk, device=dev) * 0.5)
    v = quant(torch.randn(b, hkv, s_kv, d_v, device=dev) * 0.5)
    return q, k, v


def _fp8_d192_oracle(q_f64, k_f64, v_f64, *, attn_scale, causal, bottom_right=False, window_left=None):
    """Float64 softmax(Q K^T * attn_scale) V and its natural-log LSE under a top-left or bottom-right
    causal mask (plus an optional left window of ``window_left`` past keys riding the diagonal); a row without a
    live key comes out as O = 0 / LSE = -inf (the kernel's contract)."""
    import torch

    hq, hkv = q_f64.shape[1], k_f64.shape[1]
    s_q, s_kv = q_f64.shape[2], k_f64.shape[2]
    rep = hq // hkv
    logits = (q_f64 @ k_f64.repeat_interleave(rep, 1).transpose(-1, -2)) * attn_scale
    if causal:
        i = torch.arange(s_q, device=logits.device).view(s_q, 1)
        j = torch.arange(s_kv, device=logits.device).view(1, s_kv)
        diag = i + (s_kv - s_q if bottom_right else 0)
        logits = logits.masked_fill(j > diag, float("-inf"))
        if window_left is not None:
            logits = logits.masked_fill(j < diag - window_left, float("-inf"))
    ref_lse = torch.logsumexp(logits, dim=-1)
    ref_o = torch.softmax(logits, dim=-1).nan_to_num(0.0) @ v_f64.repeat_interleave(rep, 1)
    return ref_o, ref_lse


def _run_fp8_d192_softmax_arm(q, k, v, *, precision, with_stats, split_kv, causal, bottom_right, attn_scale, dtype_o, window_left=None):
    """Build + run the per-tensor FP8 (192, 128) kernel on cc10.7 with the requested softmax arm; returns
    ``(api, out, lse)`` with the outputs NaN-poisoned beforehand so an unwritten cell stays visible."""
    import torch
    from cudnn import data_type as cudnn_dtype
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    (q8, dq, _), (k8, dk, _), (v8, dv, _) = q, k, v
    b, hq, s_q, _ = q8.shape
    d_v = v8.shape[-1]
    dev = q8.device
    out = torch.full((b, s_q, hq, d_v), float("nan"), device=dev, dtype=dtype_o).transpose(1, 2)
    lse = torch.full((b, hq, s_q), float("nan"), device=dev, dtype=torch.float32)
    api = SdpaFwdDslSm100(
        q8,
        k8,
        v8,
        out,
        lse if with_stats else None,
        scale_softmax=attn_scale,
        is_causal=causal,
        causal_bottom_right=bottom_right,
        window_size_left=window_left,
        pertensor_fp8=True,
        dtype_o=dtype_o,
        cga=2,
        split_kv=split_kv,
        softmax_precision=cudnn_dtype.HALF if precision == "half" else cudnn_dtype.FLOAT,
    )
    assert api.check_support()
    api.compile()
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    api.execute(q8, k8, v8, out, lse_tensor=lse if with_stats else None, descale_q=dq, descale_k=dk, descale_v=dv, workspace=ws)
    torch.cuda.synchronize()
    # The arm the build traced, and the kernel that traced it.
    assert "d192_d128_fp8" in api.kernel_template, api.kernel_template
    assert api._k_mod.SOFTMAX_F16 == (1 if precision == "half" else 0)
    assert api._k_mod._FUSED_SHIFT_CVT is False, "per-tensor FP8 never traces the fused shift+convert arm (the pre-folded scale is declined)"
    from cudnn.frost.tile_dsl import softmax_f16

    assert api._k_mod._softmax_f16 is softmax_f16, "the kernel must run the SHARED f16x2 arms, not a local copy"
    if split_kv > 1:
        assert api._fp32_partial_split(), "the cc10.7 split must take the fp32-partial path (exact combine)"
    return api, out, lse


_FP8_FORMATS = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}


@pytest.mark.L0
@pytest.mark.parametrize("split_kv", [1, 2], ids=["unsplit", "split2"])
@pytest.mark.parametrize("with_stats", [True, False], ids=["stats", "nostats"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("fp8", sorted(_FP8_FORMATS))
def test_fp8_d192_half_softmax_matches_the_oracle(fp8, causal, with_stats, split_kv):
    """cc10.7 e2e for softmax_precision=HALF on the per-tensor FP8 d192x128 kernel: MUFU EX2.F16x2 on packed
    pairs and a direct f16x2 -> FP8 cast of P, in BOTH FP8 formats the kernel takes (the P pair format follows
    the input format), with and without Stats (the Stats build keeps the EXACT f32 denominator, so its LSE
    must sit within 1e-4 natural-log of the oracle), dense and causal (the masked 3-segment loop), unsplit
    and split in two (the fp32-partial combine must stay exact).  Every O cell and LSE row must be written and
    stay within the oracle bound of the DEQUANTIZED inputs the kernel actually saw."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 FP8 kernels serve cc10.7 only")

    d_qk, d_v = 192, 128
    b, hq, hkv, s = (2, 8, 2, 2048) if causal else (1, 8, 2, 1024)
    attn_scale = d_qk**-0.5
    torch.manual_seed(0)
    q, k, v = _fp8_pertensor_inputs(b, hq, hkv, s, s, d_qk, d_v, getattr(torch, _FP8_FORMATS[fp8]))
    _, out, lse = _run_fp8_d192_softmax_arm(
        q, k, v, precision="half", with_stats=with_stats, split_kv=split_kv, causal=causal, bottom_right=False, attn_scale=attn_scale, dtype_o=torch.bfloat16
    )
    ref_o, ref_lse = _fp8_d192_oracle(q[2], k[2], v[2], attn_scale=attn_scale, causal=causal)
    assert torch.isfinite(out).all(), "non-finite / unwritten O cells"
    scale = ref_o.abs().max().item()
    err = (out.double() - ref_o).abs().max().item()
    assert err <= 0.1 * scale, f"max err {err} vs oracle (scale {scale})"
    if with_stats:
        assert torch.isfinite(lse).all(), "unwritten LSE rows"
        lse_err = (lse.double() - ref_lse).abs().max().item()
        assert lse_err <= 1e-4, f"LSE max err {lse_err} vs oracle (natural log)"


@pytest.mark.L0
@pytest.mark.parametrize(
    "precision, with_stats, split_kv",
    [("half", True, 1), ("half", False, 1), ("half", True, 2), ("half", False, 2), ("float", True, 1), ("float", False, 2)],
    ids=["half-stats-unsplit", "half-nostats-unsplit", "half-stats-split2", "half-nostats-split2", "float-stats-unsplit", "float-nostats-split2"],
)
def test_fp8_d192_softmax_arms_keep_keyless_rows_dead(precision, with_stats, split_kv):
    """Keyless rows under both softmax arms: unpadded bottom-right causal with S_q > S_kv puts the first
    S_q - S_kv rows above the diagonal, 156 of them here -- a whole q tile plus a partial one, so some share
    their KV loop with live rows and see fully-masked iterations only.  The kernel's contract is EXACTLY
    O = 0 / LSE = -inf for them (the fully-masked iterations contribute P = 0 on either arm, so the
    denominator ends at 0 and the epilogue's dead-row select fires); every live row stays within the oracle
    bound.  With split_kv=2 the second split is dead for most live rows (keys past column 143 are reachable
    from rows >= 300 only), exercising the combine's per-row dead-split handling under HALF as well.  The
    FLOAT legs are the control: the keyless semantics are arm-independent."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 FP8 kernels serve cc10.7 only")

    d_qk, d_v = 192, 128
    b, hq, hkv, s_q, s_kv = 2, 8, 2, 456, 300
    keyless = s_q - s_kv  # rows [0, 156) have no key under the bottom-right diagonal
    attn_scale = d_qk**-0.5
    torch.manual_seed(1)
    q, k, v = _fp8_pertensor_inputs(b, hq, hkv, s_q, s_kv, d_qk, d_v, torch.float8_e4m3fn)
    _, out, lse = _run_fp8_d192_softmax_arm(
        q, k, v, precision=precision, with_stats=with_stats, split_kv=split_kv, causal=True, bottom_right=True, attn_scale=attn_scale, dtype_o=torch.bfloat16
    )
    ref_o, ref_lse = _fp8_d192_oracle(q[2], k[2], v[2], attn_scale=attn_scale, causal=True, bottom_right=True)
    assert torch.isneginf(ref_lse[..., :keyless]).all() and torch.isfinite(ref_lse[..., keyless:]).all()  # the geometry is what the docstring says
    assert (out[..., :keyless, :] == 0).all(), "keyless rows must come out exactly O = 0"
    assert torch.isfinite(out).all(), "non-finite / unwritten O cells"
    live = out[..., keyless:, :].double()
    scale = ref_o.abs().max().item()
    err = (live - ref_o[..., keyless:, :]).abs().max().item()
    assert err <= 0.1 * scale, f"live rows: max err {err} vs oracle (scale {scale})"
    if with_stats:
        assert torch.isneginf(lse[..., :keyless]).all(), "keyless rows must publish LSE = -inf"
        assert torch.isfinite(lse[..., keyless:]).all(), "unwritten / non-finite live LSE rows"
        lse_err = (lse[..., keyless:].double() - ref_lse[..., keyless:]).abs().max().item()
        assert lse_err <= 1e-4, f"live rows: LSE max err {lse_err} vs oracle (natural log)"


@pytest.mark.L0
@pytest.mark.parametrize("precision", ["float", "half"])
def test_fp8_d192_masked_leading_tile_keeps_rows_with_later_keys_finite(precision):
    """A row whose FIRST KV tile is fully masked while a LATER tile holds its keys, on the per-tensor FP8 d192x128 kernel:
    top-left causal with a 34-key band at S = 256 (rows 161..255: no key in tile 0, 34 keys in tile 1) under UNIT descales and
    attn_scale 1, so the folded scale_log2 = log2 e > 1 overflows the scaled mask sentinel to -inf -- the running max of that
    tile, and -inf - (-inf) = NaN in P and in the Sigma denominator (with the quantizer's amax / 448 descales the same tile
    published P = 1 instead, wiped by the next live tile's alpha = 0).  Values are drawn inside the fp8 range (std 1.5) so unit
    descales are the kernel's real operating point.  Every row has keys: O and LSE finite and at the float64 oracle on both
    softmax arms (the Stats leg's exact f32 denominator keeps the LSE within 1e-4)."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 FP8 kernels serve cc10.7 only")
    d_qk, d_v, b, hq, hkv, s = 192, 128, 1, 8, 2, 256
    gen = torch.Generator(device="cuda").manual_seed(0)

    def draw(h, d, std):
        x8 = (torch.randn(b, s, h, d, device="cuda", generator=gen) * std).to(torch.float8_e4m3fn).transpose(1, 2)
        return x8, torch.ones(1, device="cuda", dtype=torch.float32), x8.double()

    q, k, v = draw(hq, d_qk, 1.5), draw(hkv, d_qk, 1.5), draw(hkv, d_v, 1.0)
    _, out, lse = _run_fp8_d192_softmax_arm(
        q, k, v, precision=precision, with_stats=True, split_kv=1, causal=True, bottom_right=False, attn_scale=1.0, dtype_o=torch.bfloat16, window_left=33
    )
    ref_o, ref_lse = _fp8_d192_oracle(q[2], k[2], v[2], attn_scale=1.0, causal=True, window_left=33)
    assert torch.isfinite(ref_lse).all(), "geometry: every row keeps 34 keys"
    assert torch.isfinite(out).all(), f"{int((~torch.isfinite(out)).sum())} non-finite O cells"
    assert torch.isfinite(lse).all(), f"{int((~torch.isfinite(lse)).sum())} non-finite LSE rows"
    scale = ref_o.abs().max().item()
    err = (out.double() - ref_o).abs().max().item()
    assert err <= 0.1 * scale, f"O max err {err} vs oracle (scale {scale})"
    lse_err = (lse.double() - ref_lse).abs().max().item()
    assert lse_err <= 1e-4, f"LSE max err {lse_err} vs oracle (natural log)"


# ------------------------------------------------------------------ the d512 MXFP8 softmax levers (the role-split kernel, SMEM P pack)
# sm107/prefill_d512_mxfp8.py packs P per 16-B vector straight into the SMEM transfer ring (no TMEM P), normalizes O with a
# REGISTER row-sum (no ones-MMA) and traces its softmax body twice (dense / masked arm).  The port keeps the bitcast + 16-B
# store + DSMEM ship untouched and switches only the exponent arm: the f32 chain (default), HALF (MUFU EX2.F16x2 and
# f16x2 -> FP8 words per 16-elem unit; stats-less row-sum = the HADD2 pair tree over the stored P words, Stats = the exact
# f32 sum), the pre-folded scale (raw max, S - m) and the fused FHADD2 arm (HALF + fold, stats-less).  A swapped FP8 word,
# a dropped P vector or a sum over the wrong words is a silent wrong-O, so every arm runs against the float64 oracle below.
_D512_MXFP8_LEVER_KW = dict(fp8=True, pertensor=False, dtype_o=_BF16_OUT, cta_mma=2)
_D512_LEVER_ARMS = [("half", False), ("half", True), ("float", True)]
_D512_LEVER_ARM_IDS = ["half", "half+fold", "float+fold"]


@pytest.mark.parametrize("dtype_qkv", [_E4M3, 1], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("f16, prefolded", [(False, False), (True, False), (False, True), (True, True)], ids=["float", "half", "float+fold", "half+fold"])
def test_d512_mxfp8_softmax_lever_constants_and_call_sites(dtype_qkv, f16, prefolded):
    """GPU-less pin of the d512 MXFP8 port: the module constants follow TemplateParams, ``_FUSED_SHIFT_CVT`` is the
    HALF + fold conjunction with the helper module's DSL probe (the body additionally folds it on ``not has_lse``),
    the P pair tag follows the FP8 input member, the arms are the shared helpers (no copied bodies), and the softmax
    body's ``has_lse`` Constexpr reaches all four kv-loop call sites -- a missed site would trace the stats-less f16
    pair sum (or the fused arm) under Stats."""
    import re

    from cudnn.frost.tile_dsl import softmax_f16 as sf

    mod = _load(_D512, rubin=True, dtype_qkv=dtype_qkv, softmax_f16=f16, softmax_scale_prefolded=prefolded, **_D512_MXFP8_LEVER_KW)
    assert mod.__file__.endswith("sm107/prefill_d512_mxfp8.py")
    assert (mod.SOFTMAX_F16, mod.SCALE_PREFOLDED) == (int(f16), int(prefolded))
    assert mod._FUSED_SHIFT_CVT == (f16 and prefolded and sf.FUSED_SHIFT_CVT_AVAILABLE)
    assert mod._FP8_TAG_P == ("e5m2" if dtype_qkv == 1 else "e4m3")
    src = _source_lines(mod)
    body = src[src.index("def _sg0_softmax_kv_iter(") :]
    assert "has_lse: cutlass.Constexpr[bool]" in body[: body.index("):")]
    # the four kv-loop call sites (dense loop; LEFT-masked / center / RIGHT-masked segments) thread the Stats fact
    calls = list(re.finditer(r"= _sg0_softmax_kv_iter\(\s*(True|False),\s*([^,]+),\s*_kv,", body))
    assert len(calls) == 4, [c.group(0) for c in calls]
    assert {c.group(2) for c in calls} == {"lse_tensor is not None"}
    assert {c.group(1) for c in calls} == {"True", "False"}
    # the arms are the shared helpers, consumed per 16-elem unit; the exact-sum Stats leg and the f16 pair sum both exist
    for marker in (
        "_softmax_f16.f16_exp_values(vals, _FP8_TAG_P, P_ELEMS_PER_VEC, fused_m=new_total_max)",
        "_softmax_f16.f16_exp_values(vals, _FP8_TAG_P, P_ELEMS_PER_VEC)",
        "_softmax_f16.f16_pairs_sum_pair(p_pairs)",
        "row_reduction_pair(cute.math.exp2(reg_S_shifted, fastmath=True))",
        "cutlass.const_expr(_FUSED_SHIFT_CVT and not has_lse)",
    ):
        assert marker in body, marker
    assert "sub_packed_f16x2_f32x2_f32x2" not in src and "ex2.approx.f16x2" not in src, "helper bodies must not be copied into the kernel"


def _d512_mxfp8_lever_problem(b, hq, hkv, s_q, s_kv, prefold_scale, fp8_dtype):
    """MXFP8 d512 operands for the lever oracle: Q is multiplied by ``prefold_scale`` BEFORE quantization (the
    softmax_scale_prefolded contract), either FP8 input member, s_q / s_kv independent (keyless-row geometries).
    Returns ((q8, sf_q, dq), (k8, sf_k, dk), (v8, sf_v, dv)) with the float64 DEQUANTIZED copies the oracle sees."""
    import torch

    d = 512
    dev = "cuda"
    torch.manual_seed(0)
    qf = torch.randn(b, hq, s_q, d, device=dev) * 0.5
    kf = torch.randn(b, hkv, s_kv, d, device=dev) * 0.5
    vf = torch.randn(b, hkv, s_kv, d, device=dev) * 0.5
    q8, dq, sfq = _mx_bshd(qf * prefold_scale, b, hq, s_q, d, columnwise=False, fp8_dtype=fp8_dtype)
    k8, dk, sfk = _mx_bshd(kf, b, hkv, s_kv, d, columnwise=False, fp8_dtype=fp8_dtype)
    v8, dv, sfv = _mx_bshd(vf, b, hkv, s_kv, d, columnwise=True, fp8_dtype=fp8_dtype)
    return (q8, sfq, dq.double()), (k8, sfk, dk.double()), (v8, sfv, dv.double())


# mask spec -> (is_causal, causal_bottom_right, window_size_left)
_D512_LEVER_MASKS = {
    "dense": (False, False, None),
    "causal": (True, False, None),
    "causal_br": (True, True, None),
    "causal_br_swa200": (True, True, 200),
    "causal_swa33": (True, False, 33),  # top-left + a 34-key band: rows >= 161 of a 256-row problem see KV tile 0 fully masked
}


def _d512_mxfp8_lever_case(precision, prefolded, with_stats, mask, fp8_dtype, b, hq, hkv, s_q, s_kv, attn_scale=None):
    """One (arm, Stats, mask, FP8 member) build of the d512 MXFP8 kernel against the float64 oracle of the
    DEQUANTIZED inputs: every O cell written and within 0.1 * max|ref|, every LSE row written and within 1e-4
    (natural log -- every Stats leg of this kernel uses the exact f32 sum), keyless rows O = 0 / LSE = -inf.
    ``attn_scale`` defaults to d ** -0.5.  Returns the kernel module the adapter compiled."""
    import math

    import torch

    from cudnn import data_type as cudnn_dtype
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    d = 512
    attn_scale = d**-0.5 if attn_scale is None else attn_scale
    dev = "cuda"
    causal, bottom_right, window_left = _D512_LEVER_MASKS[mask]
    prefold_scale = attn_scale * math.log2(math.e) if prefolded else 1.0
    (q8, sfq, dq), (k8, sfk, dk), (v8, sfv, dv) = _d512_mxfp8_lever_problem(b, hq, hkv, s_q, s_kv, prefold_scale, fp8_dtype)
    out = torch.full((b, s_q, hq, d), float("nan"), device=dev, dtype=torch.bfloat16).transpose(1, 2)  # sentinel: an unclaimed tile stays visible
    lse = torch.full((b, hq, s_q), float("nan"), device=dev, dtype=torch.float32)
    api = SdpaFwdDslSm100(
        q8,
        k8,
        v8,
        out,
        lse if with_stats else None,
        scale_softmax=None if prefolded else attn_scale,
        is_causal=causal,
        causal_bottom_right=bottom_right,
        window_size_left=window_left,
        pertensor_fp8=False,
        dtype_o=torch.bfloat16,
        cga=2,
        softmax_precision=cudnn_dtype.HALF if precision == "half" else None,
        softmax_scale_prefolded=prefolded,
    )
    assert api.check_support()
    api.compile()
    mod = api._k_mod
    assert mod.__file__.endswith("sm107/prefill_d512_mxfp8.py"), mod.__file__
    assert (mod.SOFTMAX_F16, mod.SCALE_PREFOLDED) == (int(precision == "half"), int(prefolded))
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    api.execute(q8, k8, v8, out, lse_tensor=lse if with_stats else None, sf_q=sfq, sf_k=sfk, sf_v=sfv, workspace=ws)
    torch.cuda.synchronize()

    # Oracle (float64) on the dequantized inputs: a prefolded Q already carries attn_scale*log2(e), so its logits are
    # exp2-domain -> softmax_e(ln2 * S); otherwise softmax_e(attn_scale * S).  Bottom-right anchors the diagonal (and
    # the left band) at s_kv - s_q; rows whose band holds no key are keyless.
    rep = hq // hkv
    logits = (dq @ dk.repeat_interleave(rep, 1).transpose(-1, -2)) * (math.log(2.0) if prefolded else attn_scale)
    if causal:
        qi = torch.arange(s_q, device=dev)[:, None] + (s_kv - s_q if bottom_right else 0)
        kj = torch.arange(s_kv, device=dev)[None, :]
        keep = kj <= qi
        if window_left is not None:
            keep &= kj >= qi - window_left
        logits = logits.masked_fill(~keep, float("-inf"))
    keyless = torch.isinf(logits).all(-1)  # [b, hq, s_q]
    ref = (torch.softmax(logits, dim=-1) @ dv.repeat_interleave(rep, 1)).masked_fill(keyless[..., None], 0.0)
    assert torch.isfinite(out).all(), "non-finite / unwritten O cells"
    scale = ref.abs().max().item()
    err = (out.double() - ref).abs().max().item()
    assert err <= 0.1 * scale, f"max err {err} vs oracle (scale {scale})"
    if keyless.any():
        assert (out.double()[keyless] == 0).all(), "keyless rows must publish O = 0"
    if with_stats:
        ref_lse = torch.logsumexp(logits, dim=-1)  # -inf on keyless rows
        assert not torch.isnan(lse).any(), "unwritten LSE rows"
        assert torch.equal(torch.isinf(lse), keyless), "keyless rows publish LSE = -inf, live rows a finite LSE"
        lse_err = (lse.double() - ref_lse)[~keyless].abs().max().item()
        assert lse_err <= 1e-4, f"LSE max err {lse_err} vs oracle (natural log)"
    return mod


def _d512_lever_board_only():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 d512 MXFP8 kernel serves cc10.7 only")


def _assert_d512_fused_gate(mod, precision, prefolded):
    from cudnn.frost.tile_dsl import softmax_f16 as sf

    # the HALF + prefolded build is the fused FHADD2 arm (on its stats-less trace) whenever the DSL exposes the op
    assert mod._FUSED_SHIFT_CVT == (precision == "half" and prefolded and sf.FUSED_SHIFT_CVT_AVAILABLE)


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False], ids=["stats", "nostats"])
@pytest.mark.parametrize("mask, b, hq, hkv, s", [("dense", 1, 4, 1, 1024), ("causal", 1, 8, 2, 2048)], ids=["dense", "causal"])
@pytest.mark.parametrize("precision, prefolded", _D512_LEVER_ARMS, ids=_D512_LEVER_ARM_IDS)
def test_d512_mxfp8_softmax_levers_match_the_oracle(precision, prefolded, mask, b, hq, hkv, s, with_stats):
    """cc10.7 e2e for the softmax levers of the d512 MXFP8 kernel, every arm x Stats x {dense, causal} on e4m3 inputs:
    HALF (f16x2 exponent, f16 pair-tree row-sum without Stats / exact f32 sum with Stats), HALF + fold (the fused
    FHADD2 arm on the stats-less trace) and FLOAT + fold (raw max, S - m).  O within the oracle bound of the
    dequantized inputs, every cell and LSE row written, the LSE within 1e-4 natural on every Stats leg."""
    import torch

    _d512_lever_board_only()
    mod = _d512_mxfp8_lever_case(precision, prefolded, with_stats, mask, torch.float8_e4m3fn, b, hq, hkv, s, s)
    _assert_d512_fused_gate(mod, precision, prefolded)


@pytest.mark.L0
@pytest.mark.parametrize(
    "precision, prefolded, with_stats",
    [("half", False, False), ("half", False, True), ("half", True, False), ("half", True, True), ("float", True, False)],
    ids=["half-nostats", "half-stats", "half+fold-nostats", "half+fold-stats", "float+fold-nostats"],
)
def test_d512_mxfp8_softmax_levers_match_the_oracle_e5m2(precision, prefolded, with_stats):
    """The e5m2 input member: P casts to e5m2 pairs (cvt ... e5m2x2.f16x2 on the HALF arms, the e5m2 tag of the
    shared helpers) and the BMM2 reads e5m2 P -- a wrong tag or a swapped word is a wrong O here as well."""
    import torch

    _d512_lever_board_only()
    mod = _d512_mxfp8_lever_case(precision, prefolded, with_stats, "dense", torch.float8_e5m2, 1, 4, 1, 1024, 1024)
    assert mod._FP8_TAG_P == "e5m2"
    _assert_d512_fused_gate(mod, precision, prefolded)


@pytest.mark.L0
@pytest.mark.parametrize(
    "mask, precision, prefolded, with_stats",
    [
        ("causal_br_swa200", "half", True, False),
        ("causal_br_swa200", "half", True, True),
        ("causal_br_swa200", "float", True, False),
        ("causal_br_swa200", "half", False, False),
        ("causal_br", "half", True, True),
    ],
    ids=["swa-half+fold-nostats", "swa-half+fold-stats", "swa-float+fold-nostats", "swa-half-nostats", "br-half+fold-stats"],
)
def test_d512_mxfp8_softmax_levers_keyless_rows(mask, precision, prefolded, with_stats):
    """Keyless rows and fully-masked tiles under the levers.  Bottom-right causal with s_kv = s_q - 64: rows 0..63 hold
    no key INSIDE a live Q tile, so their tiles run the mask sentinel through the arm (the running-max step selects every
    one of them out of the row's state: alpha = 1, P = 0, the row ends at (sentinel, 0)) and the epilogue's geometry
    select must publish O = 0 / LSE = -inf regardless.  The 200-wide left band additionally gives live rows a
    fully-masked FIRST tile (selected out the same way; the first live tile starts the online softmax) and fully-masked
    trailing tiles (P = 0 under a finite running max) on every arm; the ragged s_kv tail is covered by the diagonal.
    Live rows match the oracle."""
    import torch

    _d512_lever_board_only()
    mod = _d512_mxfp8_lever_case(precision, prefolded, with_stats, mask, torch.float8_e4m3fn, 1, 4, 1, 1024, 960)
    _assert_d512_fused_gate(mod, precision, prefolded)


@pytest.mark.L0
@pytest.mark.parametrize(
    "precision, prefolded, with_stats",
    [("float", False, True), ("half", False, False), ("half", True, False), ("float", True, True)],
    ids=["float-stats", "half-nostats", "half+fold-nostats", "float+fold-stats"],
)
def test_d512_mxfp8_masked_leading_tile_keeps_rows_with_later_keys_finite(precision, prefolded, with_stats):
    """A row whose FIRST KV tile is fully masked while a LATER tile holds its keys, on the d512 MXFP8 role-split kernel:
    top-left causal with a 34-key band at S = 256 (rows 161..255: no key in tile 0, their 34 keys in tile 1) at attn_scale 1,
    on every softmax arm -- the f32 chain, HALF (f16x2 exponent), HALF + fold (the fused shift + convert when stats-less) and
    FLOAT + fold.  The scaled chain clamps the tile max to the finite sentinel and the fold keeps the raw one, so both took the
    sentinel as the running max and published P = 1 per masked column, mass that only alpha = 0 at the next live tile wiped;
    the running-max step now selects the dead tile out of the state (total_max kept, alpha = 1, P = 0).  The all-ones draw
    makes every live row's softmax flat (O = V exactly), so the cells are deterministic; the module's own oracle judges them."""
    from unittest.mock import patch

    import torch

    _d512_lever_board_only()
    with patch.object(torch, "randn", side_effect=lambda *a, **k: torch.ones(*a, **k)):
        mod = _d512_mxfp8_lever_case(precision, prefolded, with_stats, "causal_swa33", torch.float8_e4m3fn, 1, 4, 1, 256, 256, attn_scale=1.0)
    _assert_d512_fused_gate(mod, precision, prefolded)


# ------------------------------------------------------------------------------------- d256 per-tensor FP8: softmax_precision=HALF
# The d256 per-tensor FP8 kernel (sm107/prefill_d256_fp8.py) normalizes O with a REGISTER row-sum -- there is no ones-MMA
# Sigma column -- so its HALF arm has two shapes: a stats-less build sums the f16x2 P words it stores (an HADD2 pair tree,
# softmax_f16.f16_exp_chunk_f16sum) and a Stats build keeps the exact f32 sum of a second f32 exp2 for the published LSE
# (softmax_f16.f16_exp_chunk_sum).  The pre-folded scale is not served on per-tensor FP8 (the kernel folds descale_q *
# descale_k into the softmax scale in-kernel), so _FUSED_SHIFT_CVT is False on every build.  The four inlined softmax
# segments (dense HW-max loop, masked prologue, unmasked interior, masked tail) share ONE P tail, _softmax_p_tail.
_D256_FP8_HALF_MASKS = {
    # name: (causal, window_left).  dense walks the HW-max loop only; causal adds the unmasked interior and the masked tail;
    # the left window adds the masked prologue -- together every inlined segment reaches the shared tail.
    "dense": (False, None),
    "causal": (True, None),
    "causal_swa": (True, 300),
}


def _pertensor_fp8_problem(b, hq, hkv, s_q, s_kv, d, fp8_dtype, *, seed=0):
    """Random Q/K/V quantized per-tensor (amax / FP8 max -- the quantizer of every per-tensor FP8 test in this tree),
    BSHD-physical as the adapter consumes them, their 1-element fp32 descales, and the DEQUANTIZED float64 operands the
    oracle must see.  Q and K are drawn wide (std 1.5 -> logits std ~2.25 at d=256) so the softmax is peaked: under a flat
    softmax O is the mean of V whatever P is, and a permuted or mis-cast P would pass."""
    import torch

    dev = "cuda"
    fmax = torch.finfo(fp8_dtype).max
    gen = torch.Generator(device=dev).manual_seed(seed)

    def quant(x):
        dsc = (x.abs().amax().clamp_min(1e-8) / fmax).item()
        x8 = (x / dsc).clamp(-fmax, fmax).to(fp8_dtype)
        return x8.transpose(1, 2), torch.full((1,), dsc, device=dev, dtype=torch.float32), x8.double().transpose(1, 2) * dsc

    q = quant(torch.randn(b, s_q, hq, d, device=dev, generator=gen) * 1.5)
    k = quant(torch.randn(b, s_kv, hkv, d, device=dev, generator=gen) * 1.5)
    v = quant(torch.randn(b, s_kv, hkv, d, device=dev, generator=gen))
    return q, k, v


def _pertensor_fp8_oracle(qd, kd, vd, *, scale, causal, bottom_right, window_left):
    """float64 softmax(QK^T * scale) V on the dequantized operands under the kernel's mask (top-left or bottom-right
    causal, an optional left window W = keep k in [q - W, q]).  Returns (O, natural-log LSE, keyless-row mask over s_q);
    a keyless row carries O = 0 and LSE = -inf -- the kernel's empty-row contract."""
    import torch

    b, hq, s_q, _ = qd.shape
    s_kv = kd.shape[2]
    rep = hq // kd.shape[1]
    logits = (qd @ kd.repeat_interleave(rep, 1).transpose(-1, -2)) * scale
    qi = torch.arange(s_q, device=qd.device).view(-1, 1)
    kj = torch.arange(s_kv, device=qd.device).view(1, -1)
    diag = (s_kv - s_q) if bottom_right else 0
    keep = torch.ones(s_q, s_kv, dtype=torch.bool, device=qd.device)
    if causal:
        keep &= kj <= qi + diag
    if window_left is not None:
        keep &= kj >= qi + diag - window_left
    logits = logits.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    keyless = ~keep.any(dim=-1)
    o = torch.softmax(logits, dim=-1).nan_to_num(0.0) @ vd.repeat_interleave(rep, 1)
    return o, lse, keyless


def _run_d256_fp8(q8, k8, v8, descales, *, precision, with_stats, causal, bottom_right=False, window_left=None, cga=None, scale_softmax=None):
    """Build, compile and launch the d256 per-tensor FP8 adapter (bf16 O); returns (api, O [b, hq, s_q, d], LSE or None).
    O and LSE start as NaN sentinels so an unwritten cell stays visible.  ``scale_softmax`` defaults to d ** -0.5."""
    import torch
    from cudnn import data_type as cudnn_dtype
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, hq, s_q, d = q8.shape
    dev = q8.device
    out = torch.full((b, s_q, hq, d), float("nan"), device=dev, dtype=torch.bfloat16).transpose(1, 2)
    lse = torch.full((b, hq, s_q), float("nan"), device=dev, dtype=torch.float32) if with_stats else None
    api = SdpaFwdDslSm100(
        q8,
        k8,
        v8,
        out,
        lse,
        is_causal=causal,
        causal_bottom_right=bottom_right,
        window_size_left=window_left,
        scale_softmax=d**-0.5 if scale_softmax is None else scale_softmax,
        pertensor_fp8=True,
        dtype_o=torch.bfloat16,
        cga=cga,
        softmax_precision={"half": cudnn_dtype.HALF, "float": cudnn_dtype.FLOAT}[precision],
    )
    assert api.check_support()
    api.compile()
    dq, dk, dv = descales
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    api.execute(q8, k8, v8, out, lse_tensor=lse, descale_q=dq, descale_k=dk, descale_v=dv, workspace=ws)
    torch.cuda.synchronize()
    return api, out, lse


def _check_d256_fp8(out, lse, ref_o, ref_lse, keyless, *, with_stats, tag):
    """The oracle bound of the quantized family (O within 0.1 * max|ref|; LSE within 1e-4 natural on the exact-sum Stats
    build), every live cell written, and the empty-row contract (O = 0, LSE = -inf) on keyless rows."""
    import torch

    live = ~keyless
    assert torch.isfinite(out[:, :, live]).all(), f"{tag}: non-finite / unwritten O cells"
    scale = ref_o.abs().max().item()
    err = (out[:, :, live].double() - ref_o[:, :, live]).abs().max().item()
    assert err <= 0.1 * scale, f"{tag}: max |O - ref| {err:.4f} vs 0.1 * {scale:.4f}"
    if keyless.any():
        assert (out[:, :, keyless].float() == 0).all(), f"{tag}: a keyless row must write O = 0"
    if with_stats:
        assert torch.isfinite(lse[:, :, live]).all(), f"{tag}: unwritten LSE rows"
        lse_err = (lse[:, :, live].double() - ref_lse[:, :, live]).abs().max().item()
        assert lse_err <= 5e-4, f"{tag}: LSE max err {lse_err:.2e} vs oracle (natural log)"
        if keyless.any():
            assert (lse[:, :, keyless] == float("-inf")).all(), f"{tag}: a keyless row must publish LSE = -inf"
    return err, scale


def _d256_fp8_half_only():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 per-tensor FP8 d256 kernel serves cc10.7 only")


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False])
@pytest.mark.parametrize("fp8", ["e4m3", "e5m2"])
@pytest.mark.parametrize("mask", list(_D256_FP8_HALF_MASKS))
def test_d256_fp8_half_softmax_matches_the_oracle(mask, fp8, with_stats):
    """cc10.7 e2e for softmax_precision=HALF on the d256 per-tensor FP8 kernel: both FP8 formats (the P cast speaks the
    input's pair format), with and without Stats (the f16 pair-tree denominator vs the exact f32 sum), across the mask
    shapes that walk every inlined softmax segment.  The float64 oracle sees the dequantized inputs the kernel saw; a wrong
    P byte order, a dropped chunk sum or a mis-tagged cast lands far outside the 0.1 * max|ref| bound on this peaked
    softmax, and the Stats build's LSE must stay within 5e-4 (natural log) of the oracle's."""
    import torch

    _d256_fp8_half_only()
    fp8_dtype = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fp8]
    causal, window_left = _D256_FP8_HALF_MASKS[mask]
    b, hq, hkv, s, d = 1, 4, 2, 1024, 256
    (q8, dq, qd), (k8, dk, kd), (v8, dv, vd) = _pertensor_fp8_problem(b, hq, hkv, s, s, d, fp8_dtype)
    api, out, lse = _run_d256_fp8(q8, k8, v8, (dq, dk, dv), precision="half", with_stats=with_stats, causal=causal, window_left=window_left)
    assert api._k_mod.SOFTMAX_F16 == 1 and api._k_mod._FP8_TAG_P == fp8, "the HALF request must reach the d256 FP8 module with the input's pair tag"
    assert api._k_mod._FUSED_SHIFT_CVT is False, "per-tensor FP8 never traces the fused shift+convert (the scale fold stays in-kernel)"
    ref_o, ref_lse, keyless = _pertensor_fp8_oracle(qd, kd, vd, scale=d**-0.5, causal=causal, bottom_right=False, window_left=window_left)
    assert not keyless.any()
    _check_d256_fp8(out, lse, ref_o, ref_lse, keyless, with_stats=with_stats, tag=f"{mask}/{fp8}/stats={with_stats}")


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False])
def test_d256_fp8_half_softmax_arm_is_live_and_close_to_float(with_stats):
    """The HALF build is a different kernel from the FLOAT build (MUFU EX2.F16x2 + the f16x2 -> FP8 cast; stats-less, also
    the f16 pair-tree denominator), so its O must DIFFER from FLOAT's somewhere -- a HALF request that silently traced the
    f32 chain would clear every oracle bound -- while staying within half the oracle bound of it (same quantized-P contract).
    With Stats both builds form the LSE from the same exact f32 sum of the same shifted scores, so the two LSEs agree to
    fp32 rounding."""
    import torch

    _d256_fp8_half_only()
    b, hq, hkv, s, d = 1, 4, 2, 1024, 256
    (q8, dq, qd), (k8, dk, kd), (v8, dv, vd) = _pertensor_fp8_problem(b, hq, hkv, s, s, d, torch.float8_e4m3fn, seed=1)
    runs = {p: _run_d256_fp8(q8, k8, v8, (dq, dk, dv), precision=p, with_stats=with_stats, causal=False) for p in ("float", "half")}
    assert runs["float"][0]._k_mod.SOFTMAX_F16 == 0 and runs["half"][0]._k_mod.SOFTMAX_F16 == 1
    ref_o, ref_lse, keyless = _pertensor_fp8_oracle(qd, kd, vd, scale=d**-0.5, causal=False, bottom_right=False, window_left=None)
    for p, (_, out, lse) in runs.items():
        _check_d256_fp8(out, lse, ref_o, ref_lse, keyless, with_stats=with_stats, tag=f"{p}/stats={with_stats}")
    out_f, out_h = runs["float"][1], runs["half"][1]
    assert not torch.equal(out_h, out_f), "HALF O is bitwise the FLOAT O: the f16x2 exponent arm did not trace"
    xerr = (out_h.float() - out_f.float()).abs().max().item()
    assert xerr <= 0.05 * ref_o.abs().max().item(), f"HALF-vs-FLOAT O divergence {xerr:.4f}"
    if with_stats:
        lse_f, lse_h = runs["float"][2], runs["half"][2]
        assert (lse_h - lse_f).abs().max().item() <= 2e-5, "the Stats build's LSE must come from the exact f32 sum on both arms"


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False])
@pytest.mark.parametrize("geometry", ["bottom_right_short_kv", "swa_past_last_key"])
def test_d256_fp8_half_softmax_keyless_rows(geometry, with_stats):
    """Rows with no live key under the HALF arm: bottom-right causal with s_kv < s_q (whole Q tiles above the diagonal
    plus a tile that is keyless in its first rows only), and a causal left window on s_q > s_kv (rows past the last key +
    window).  A fully masked tile leaves every score at the finite sentinel, P = 1 across the tile and the f16 pair tree
    sums 128 ones per step; the correction's empty-row select must still publish O = 0 / LSE = -inf there, and the live
    rows (including the rows of a partially keyless tile) must meet the oracle."""
    import torch

    _d256_fp8_half_only()
    if geometry == "bottom_right_short_kv":
        s_q, s_kv, causal, bottom_right, window_left = 512, 320, True, True, None
    else:
        s_q, s_kv, causal, bottom_right, window_left = 640, 256, True, False, 100
    b, hq, hkv, d = 1, 4, 2, 256
    (q8, dq, qd), (k8, dk, kd), (v8, dv, vd) = _pertensor_fp8_problem(b, hq, hkv, s_q, s_kv, d, torch.float8_e4m3fn, seed=2)
    api, out, lse = _run_d256_fp8(
        q8, k8, v8, (dq, dk, dv), precision="half", with_stats=with_stats, causal=causal, bottom_right=bottom_right, window_left=window_left
    )
    assert api._k_mod.SOFTMAX_F16 == 1
    ref_o, ref_lse, keyless = _pertensor_fp8_oracle(qd, kd, vd, scale=d**-0.5, causal=causal, bottom_right=bottom_right, window_left=window_left)
    n_keyless = int(keyless.sum())
    assert 0 < n_keyless < s_q and n_keyless % 128 != 0, f"{geometry}: the probe must mix keyless and live rows inside one tile ({n_keyless} keyless)"
    _check_d256_fp8(out, lse, ref_o, ref_lse, keyless, with_stats=with_stats, tag=f"{geometry}/stats={with_stats}")


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False])
def test_d256_fp8_half_softmax_two_cta_build(with_stats, monkeypatch):
    """CTA_MMA = 2 (V split along d_v across the CTA pair; the P-ready arrive carries the pair's leader, the softmax body is
    unchanged).  The adapter pins cga1 for FP8 d256 (supported_cgas_for), so the pair build is reached by widening that pin
    for this test only; FLOAT runs first as the control for the pair geometry itself, then HALF against the same oracle."""
    import torch
    from cudnn.sdpa.fwd import api_dsl

    _d256_fp8_half_only()
    orig = api_dsl.supported_cgas_for
    monkeypatch.setattr(api_dsl, "supported_cgas_for", lambda flavor, **kw: (1, 2) if flavor == _D256 else orig(flavor, **kw))
    b, hq, hkv, s, d = 1, 4, 2, 1024, 256
    (q8, dq, qd), (k8, dk, kd), (v8, dv, vd) = _pertensor_fp8_problem(b, hq, hkv, s, s, d, torch.float8_e4m3fn, seed=3)
    ref_o, ref_lse, keyless = _pertensor_fp8_oracle(qd, kd, vd, scale=d**-0.5, causal=True, bottom_right=False, window_left=None)
    for precision in ("float", "half"):
        api, out, lse = _run_d256_fp8(q8, k8, v8, (dq, dk, dv), precision=precision, with_stats=with_stats, causal=True, cga=2)
        assert api._k_mod.CFG.CTA_MMA == 2 and api._k_mod.SOFTMAX_F16 == (precision == "half")
        _check_d256_fp8(out, lse, ref_o, ref_lse, keyless, with_stats=with_stats, tag=f"cga2/{precision}/stats={with_stats}")


@pytest.mark.L0
def test_d256_fp8_softmax_tail_is_one_helper_at_four_sites():
    """Structural pin (no GPU) for the d256 per-tensor FP8 kernel's HALF port.  The four inlined softmax segments call ONE
    module-level P tail (`_softmax_p_tail`) and build no P store of their own, so the arm selection cannot fork between the
    masked and unmasked segments; the tail dispatches SOFTMAX_F16 -> f16_exp_chunk_f16sum (stats-less) / f16_exp_chunk_sum
    (Stats) from the shared softmax_f16 module (no helper body is copied into the kernel), stores the f16 arms' words through
    an Int32 TMEM pointer, and keeps the f32 chain otherwise; has_lse reaches the softmax warp group from lse_tensor; the
    module takes the f16 exponent for both FP8 formats, never the fold (config backstop), and reports _FUSED_SHIFT_CVT False."""
    import re

    mod = _load(_D256, rubin=True, **_FP8_LOAD_KW)
    assert mod.SOFTMAX_F16 == 0 and mod._FUSED_SHIFT_CVT is False and mod._FP8_TAG_P == "e4m3" and not hasattr(mod, "SCALE_PREFOLDED")
    assert _load(_D256, rubin=True, **{**_FP8_LOAD_KW, "softmax_f16": True}).SOFTMAX_F16 == 1
    assert _load(_D256, rubin=True, **{**_FP8_LOAD_KW, "dtype_qkv": 1, "softmax_f16": True})._FP8_TAG_P == "e5m2"
    with pytest.raises(ValueError, match="softmax_scale_prefolded"):
        _load(_D256, rubin=True, **{**_FP8_LOAD_KW, "softmax_scale_prefolded": True})
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    assert "from cudnn.frost.tile_dsl import softmax_f16 as _softmax_f16" in code
    for copied in ("def f16_exp_chunk", "def fused_shift_f16", "def f16_pairs_sum_pair", "def add_f16x2", "ex2_f16x2(", "f16x2x2_to_fp8_word("):
        assert copied not in code, f"{copied!r}: helper bodies live in tile_dsl.softmax_f16, the kernel only calls them"
    tail = code.split("def _softmax_p_tail(")[1].split("def _softmax_warp_group(")[0]
    wg = code.split("def _softmax_warp_group(")[1].split("def _correction_warp_group(")[0]
    assert len(re.findall(r"^\s+total_sum = _softmax_p_tail\(has_lse, reg_S, new_total_max, alpha, scale_log2, total_sum, ", wg, re.M)) == 4
    assert "make_tmem_ptr(p_addr" not in wg and "exp2(chunk_S" not in wg and ".to(STORAGE_DTYPE)" not in wg and "row_reduction_pair(" not in wg
    assert tail.count("_softmax_f16.f16_exp_chunk_f16sum(chunk_S, _FP8_TAG_P, CHUNK)") == 1
    assert tail.count("_softmax_f16.f16_exp_chunk_sum(chunk_S, _FP8_TAG_P, CHUNK)") == 1
    assert "if cutlass.const_expr(SOFTMAX_F16):" in tail and "if cutlass.const_expr(has_lse):" in tail
    assert "make_tmem_ptr(p_addr, cutlass.Int32), p_words" in tail, "the f16 arms store packed FP8 words through an Int32 TMEM pointer"
    assert tail.count("cutlass.Float32), chunk_P_") == 2, "the f32 chain keeps its two Float32-pointer P stores"
    assert "reg_S = reg_S * scale_log2 - new_total_max" in tail and "reg_S - new_total_max\n" not in tail, "no pre-folded shift: the scale fold stays"
    assert "has_lse: cutlass.Constexpr[bool]," in wg.split(")")[0] and "has_lse=lse_tensor is not None," in code


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False], ids=["stats", "nostats"])
def test_d256_fp8_masked_leading_tile_keeps_rows_with_later_keys_finite(with_stats):
    """A row whose FIRST KV tile is fully masked while a LATER tile holds its keys: top-left causal with left bound 34 at S = 256 --
    rows 161..255 have no key in tile 0 (keys 0..127) and their 34 keys in tile 1 -- under UNIT descales and attn_scale 1.  The
    per-tensor kernel folds descale_q * descale_k into scale_log2, so this is the configuration (a producer whose values already sit
    in the fp8 range) whose scaled mask sentinel, taken as the running max, overflows to -inf and reads -inf - (-inf) = NaN into P;
    with the quantizer's amax / 448 descales the same tile publishes P = 1 instead, wiped by the next live tile's alpha = 0.  Measured
    before the fix: O finite garbage of 1e32..1e34 (the NaN row-sum floored to 1e-30).  Every row has keys, so O and LSE are finite
    and inside the module's d256 fp8 oracle bounds; the fix keeps a tile that is dead ahead of the first live key out of the running
    state (total_max kept, alpha = 1, P = 0)."""
    import torch

    _d256_fp8_half_only()
    # Values drawn INSIDE the fp8 range and bound with unit descales (not _pertensor_fp8_problem's amax / 448 quantizer, whose
    # 448-scale codes under unit descales put the logits near 5e7, past fp32's resolution of the LSE): logits std ~36 at d = 256.
    gen = torch.Generator(device="cuda").manual_seed(0)
    q8, k8 = ((torch.randn(1, 256, h, 256, device="cuda", generator=gen) * 1.5).to(torch.float8_e4m3fn).transpose(1, 2) for h in (8, 2))
    v8 = torch.randn(1, 256, 2, 256, device="cuda", generator=gen).to(torch.float8_e4m3fn).transpose(1, 2)
    unit = torch.ones(1, device="cuda", dtype=torch.float32)
    _, out, lse = _run_d256_fp8(q8, k8, v8, (unit, unit, unit), precision="float", with_stats=with_stats, causal=True, window_left=33, scale_softmax=1.0)
    ref_o, ref_lse, keyless = _pertensor_fp8_oracle(q8.double(), k8.double(), v8.double(), scale=1.0, causal=True, bottom_right=False, window_left=33)
    assert not keyless.any(), "geometry: every row keeps 34 keys"
    assert torch.isfinite(out.float()).all(), f"{int((~torch.isfinite(out.float())).sum())} non-finite O cells"
    _check_d256_fp8(out, lse, ref_o, ref_lse, keyless, with_stats=with_stats, tag="fp8 d256 masked leading tile")


# --- the d256 f16/bf16 kernel: pre-folded softmax scale ----------------------------------------------------------------
#
# kernels/sm107/prefill_d256_f16.py gains ONE lever: softmax_scale_prefolded (the graph's attn_scale_prefolded).  Under it
# every one of the four inlined softmax copies (dense / left-masked / unmasked interior / right-masked) takes the RAW row
# max and shifts with reg_S - m, and the runtime scale_log2 is a dead operand.  The f16x2 exponent (softmax_precision=HALF)
# stays declined on half inputs by the config backstop, so the kernel has no fused arm.  Numerically the fold is neutral:
# the oracle sees the SAME pre-scaled half Q the kernel does, with ln 2 as the logit scale.

_D256_FOLD_D = 256
_D256_FOLD_MAX_BLOCK = re.compile(
    r"if cutlass\.const_expr\(SCALE_PREFOLDED\):\s*\n\s*current_max = current_max_unscaled\s*(?:#[^\n]*)?\n\s*else:\s*\n\s*current_max = current_max_unscaled \* scale_log2\s*\n"
)
_D256_FOLD_SHIFT_BLOCK = re.compile(
    r"if cutlass\.const_expr\(SCALE_PREFOLDED\):\s*\n\s*reg_S = reg_S - new_total_max\s*(?:#[^\n]*)?\n\s*else:\s*\n\s*reg_S = reg_S \* scale_log2 - new_total_max\s*\n"
)


def test_d256_half_fold_arm_is_wired_in_every_softmax_copy():
    """Source pin (no GPU) for kernels/sm107/prefill_d256_f16.py: the softmax body is inlined four times, and the pre-folded
    scale must reach the max site AND the shift site of every copy -- a copy left on the scaled chain is a silent perf no-op
    on the tiles it serves (interior tiles vs the band edges vs the diagonal), and one that pairs a raw max with a scaled
    shift (or the reverse) is a wrong O that only a mask-specific run would see.  Each copy must hold exactly one
    ``if const_expr(SCALE_PREFOLDED)`` block of each kind and no other use of scale_log2.  Also the lever constants:
    SCALE_PREFOLDED follows the template flag, the f16 exponent is never built (half inputs), the fused-arm constant reads
    False on every build."""
    mod = _load((256, 256), rubin=True, softmax_scale_prefolded=True)
    assert mod.SCALE_PREFOLDED == 1 and mod.SOFTMAX_F16 == 0 and mod._FUSED_SHIFT_CVT is False
    plain = _load((256, 256), rubin=True)
    assert plain.SCALE_PREFOLDED == 0 and plain.SOFTMAX_F16 == 0 and plain._FUSED_SHIFT_CVT is False
    for dtype in (2, 3):  # bf16, fp16: the f16 exponent is declined on BOTH half inputs before the body loads
        with pytest.raises(ValueError, match="softmax_f16"):
            _load((256, 256), rubin=True, dtype_qkv=dtype, dtype_o=dtype, softmax_f16=True)
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    softmax = code.split("def _softmax_warp_group(")[1].split("\ndef ")[0]
    copies = softmax.split("for kv_loop in cutlass.range(")[1:]
    assert len(copies) == 4, "the d256 softmax body is inlined once per KV segment (dense, left-masked, unmasked, right-masked)"
    for n, body in enumerate(copies):
        assert len(_D256_FOLD_MAX_BLOCK.findall(body)) == 1, f"copy {n}: the max site must carry the raw-max arm under SCALE_PREFOLDED"
        assert len(_D256_FOLD_SHIFT_BLOCK.findall(body)) == 1, f"copy {n}: the shift site must carry the reg_S - m arm under SCALE_PREFOLDED"
        assert body.count("scale_log2") == 2, f"copy {n}: scale_log2 may appear only on the two scaled (else) arms"
        assert body.count("if cutlass.const_expr(SCALE_PREFOLDED):") == 2, f"copy {n}: exactly the max block and the shift block"


def _d256_fold_board_only():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 d256 f16/bf16 kernel serves cc10.7 only")


def _d256_fold_operands(b, hq, hkv, s_q, s_kv, dtype, *, prefold_scale, seed=0):
    """Random BSHD-physical operands for the d256 half kernel as (unfolded Q, folded Q, K, V): the folded Q is the SAME f32
    draw multiplied by ``prefold_scale`` BEFORE the half cast (the softmax_scale_prefolded contract), so each kernel's
    oracle sees exactly the half tensor that kernel consumed."""
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    qf = torch.randn(b, s_q, hq, _D256_FOLD_D, device="cuda", generator=gen) * 0.5
    kf = torch.randn(b, s_kv, hkv, _D256_FOLD_D, device="cuda", generator=gen) * 0.5
    vf = torch.randn(b, s_kv, hkv, _D256_FOLD_D, device="cuda", generator=gen) * 0.5
    return qf.to(dtype).transpose(1, 2), (qf * prefold_scale).to(dtype).transpose(1, 2), kf.to(dtype).transpose(1, 2), vf.to(dtype).transpose(1, 2)


def _d256_fold_oracle(q, k, v, *, scale, causal=False, bottom_right=False, window_left=None, seq_kv_lens=None, sinks=None):
    """float64 softmax(scale * QK^T) V, the natural-log LSE and the per-row ``has a live key`` mask under the kernel's mask /
    padding / sink semantics (BHSD; GQA by expansion; the bottom-right diagonal anchored at the per-batch (S_q, len_kv)
    corner; ``window_left`` keys below the diagonal kept; the sink joins as one extra column).  A keyless row is O = 0 and
    LSE = -inf, or LSE = sink with one."""
    import torch

    b, hq, s_q, _ = q.shape
    hkv, s_kv = k.shape[1], k.shape[2]
    rep = hq // hkv
    dev = q.device
    kd, vd = k.double().repeat_interleave(rep, 1), v.double().repeat_interleave(rep, 1)
    scores = (q.double() @ kd.transpose(-1, -2)) * scale
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    kv_lens = seq_kv_lens.to(torch.int64).view(b, 1, 1, 1) if seq_kv_lens is not None else torch.full((b, 1, 1, 1), s_kv, dtype=torch.int64, device=dev)
    diag = i + (kv_lens - s_q) if bottom_right else i
    masked = j >= kv_lens
    if causal:
        masked = masked | (j > diag)
    if window_left is not None:
        masked = masked | (j < diag - window_left)
    masked = masked.expand(b, 1, s_q, s_kv)
    scores = scores.masked_fill(masked, float("-inf"))
    if sinks is not None:
        full = torch.cat([scores, sinks.double().view(1, hq, 1, 1).expand(b, hq, s_q, 1)], dim=-1)
        o = torch.softmax(full, dim=-1)[..., :s_kv] @ vd
    else:
        full = scores
        o = torch.softmax(scores, dim=-1).nan_to_num(0.0) @ vd
    live = (~masked).any(-1).expand(b, hq, s_q)
    return o, torch.logsumexp(full, dim=-1), live


def _d256_fold_same(a, b):
    """Bitwise agreement that treats a NaN pair as equal (NaN-filled storage past a packed total stays NaN on both launches)."""
    import torch

    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


def _d256_fold_launch(q, k, v, *, with_stats, prefolded, attn_scale, lse=None, lse_exec=None, api_kw=None, exec_kw=None):
    """Build, compile and launch the d256 half kernel through the standalone adapter; returns (api, O, LSE).  O and LSE
    start NaN-filled so an unwritten cell stays visible.  ``lse`` is the DECLARED Stats (the sample the adapter reads strides
    from) and ``lse_exec`` the tensor bound at execute when the two differ (THD binds the flat packed (T, H) storage).
    Under the fold the launch is REPEATED with the adapter's resolved scale poisoned (scale_softmax_log2 = 3 on the wire):
    the kernel must never read it, so both launches have to agree bit for bit -- the check that fails when any of the four
    softmax copies still multiplies by scale_log2 (the adapter itself pins the wire value to 1.0, under which the scaled
    chain is numerically indistinguishable)."""
    import math

    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, hq, s_q, d_v = q.shape[0], q.shape[1], q.shape[2], v.shape[3]
    out = torch.full((b, s_q, hq, d_v), float("nan"), device=q.device, dtype=q.dtype).transpose(1, 2)
    if with_stats and lse is None:
        lse = torch.full((b, hq, s_q), float("nan"), device=q.device, dtype=torch.float32)
    api = SdpaFwdDslSm100(
        q,
        k,
        v,
        out,
        lse if with_stats else None,
        scale_softmax=None if prefolded else attn_scale,
        softmax_scale_prefolded=prefolded,
        **(api_kw or {}),
    )
    assert api.check_support()
    api.compile()
    assert api._k_mod.__file__.replace("\\", "/").endswith("kernels/sm107/prefill_d256_f16.py"), api._k_mod.__file__
    lse_bind = lse_exec if lse_exec is not None else lse
    kw = dict(lse_tensor=lse_bind if with_stats else None, **(exec_kw or {}))
    ws_bytes = api.scratch_workspace_bytes()
    if ws_bytes:
        kw["workspace"] = torch.empty(ws_bytes, device=q.device, dtype=torch.uint8)
    api.execute(q, k, v, out, **kw)
    torch.cuda.synchronize()
    if prefolded:
        mod = api._k_mod
        assert mod.SCALE_PREFOLDED == 1 and mod.SOFTMAX_F16 == 0 and mod._FUSED_SHIFT_CVT is False
        o_first, lse_first = out.clone(), (lse_bind.clone() if with_stats else None)
        out.fill_(float("nan"))
        if with_stats:
            lse_bind.fill_(float("nan"))
        api.scale_softmax = 3.0 / math.log2(math.e)  # a scaled chain would now compute exp2(3 * S - m)
        api.execute(q, k, v, out, **kw)
        torch.cuda.synchronize()
        assert _d256_fold_same(out, o_first), "the pre-folded build read scale_log2: a softmax copy still multiplies by it (O)"
        assert not with_stats or _d256_fold_same(lse_bind, lse_first), "the pre-folded build read scale_log2: a softmax copy still multiplies by it (LSE)"
    return api, out, lse_bind


# (adapter kwargs, oracle kwargs): dense runs the MASK_NONE copy; causal the unmasked-interior and right-masked copies; the
# 300-key left window (not a multiple of the 128-key tile) adds the left-masked copy and rows whose first tile is fully
# masked ahead of their live tiles (the is_first re-fire on the raw-max sentinel).
_D256_FOLD_MASKS = {
    "dense": ({}, {}),
    "causal": (dict(is_causal=True), dict(causal=True)),
    "swa300": (dict(is_causal=True, window_size_left=300), dict(causal=True, window_left=300)),
}


@pytest.mark.parametrize("with_stats", [True, False], ids=["stats", "nostats"])
@pytest.mark.parametrize("mask", list(_D256_FOLD_MASKS))
@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
def test_d256_half_prefolded_scale_matches_the_oracle(dtype_name, mask, with_stats):
    """cc10.7 e2e for the pre-folded softmax scale on the d256 f16/bf16 kernel (FLOAT exponent): Q carries attn_scale *
    log2(e), the kernel takes the raw row max and the FADD2 shift in all four softmax copies.  Each (dtype, mask, Stats)
    specialization must (1) stay within the float64 oracle of the pre-scaled Q it actually saw (ln 2 as the logit scale):
    O within 0.1 * max|ref| AND within 2x the error the UNFOLDED kernel makes on the same draw, LSE within 5e-4 natural;
    (2) write every O cell and LSE row; (3) launch bit-identically with the dead runtime scale poisoned
    (_d256_fold_launch)."""
    import math

    import torch

    _d256_fold_board_only()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    b, hq, hkv, s = 2, 8, 2, 2048
    attn_scale = _D256_FOLD_D**-0.5
    api_kw, ref_kw = _D256_FOLD_MASKS[mask]
    q0, q1, k, v = _d256_fold_operands(b, hq, hkv, s, s, dt, prefold_scale=attn_scale * math.log2(math.e))
    _, o_fold, lse_fold = _d256_fold_launch(q1, k, v, with_stats=with_stats, prefolded=True, attn_scale=attn_scale, api_kw=api_kw)
    _, o_base, lse_base = _d256_fold_launch(q0, k, v, with_stats=with_stats, prefolded=False, attn_scale=attn_scale, api_kw=api_kw)
    ref_fold, ref_fold_lse, _ = _d256_fold_oracle(q1, k, v, scale=math.log(2.0), **ref_kw)
    ref_base, ref_base_lse, _ = _d256_fold_oracle(q0, k, v, scale=attn_scale, **ref_kw)
    assert torch.isfinite(o_fold.float()).all(), "unwritten / non-finite O cells under the fold"
    err_fold = (o_fold.double() - ref_fold).abs().max().item()
    err_base = (o_base.double() - ref_base).abs().max().item()
    amax = ref_fold.abs().max().item()
    assert err_fold <= 0.1 * amax, f"O max err {err_fold} vs the oracle (max|ref| {amax})"
    assert err_fold <= 2.0 * err_base, f"the fold's O error {err_fold} exceeds 2x the unfolded kernel's {err_base}"
    if with_stats:
        assert torch.isfinite(lse_fold).all(), "unwritten LSE rows under the fold"
        lse_err = (lse_fold.double() - ref_fold_lse).abs().max().item()
        assert lse_err <= 5e-4, f"LSE max err {lse_err} vs the oracle (natural log)"
        assert (lse_base.double() - ref_base_lse).abs().max().item() <= 5e-4, "the unfolded control drifted from its own oracle"


@pytest.mark.L0
@pytest.mark.parametrize("prefolded", [False, True], ids=["scaled", "prefolded"])
@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
def test_d256_half_masked_leading_tile_keeps_rows_with_later_keys_finite(dtype_name, prefolded):
    """A row whose FIRST KV tile is fully masked while a LATER tile holds its keys: top-left causal with left bound 34 at S = 256 --
    rows 161..255 have no key in tile 0 (keys 0..127) and their 34 keys in tile 1 -- at attn_scale 1 on both chains.  The scaled
    chain took the finite mask sentinel times scale_log2 > 1 (= -inf) as the running max and read -inf - (-inf) = NaN into P
    (194,560 NaN O elements measured); the pre-folded chain kept the raw sentinel and published P = 1 per masked column.  Both now
    select a tile that is dead ahead of the first live key out of the running state (total_max kept, alpha = 1, P = 0): O and LSE
    finite and at the float64 oracle of the half operands each chain saw."""
    import math

    import torch

    _d256_fold_board_only()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    attn_scale = 1.0
    q0, q1, k, v = _d256_fold_operands(1, 8, 2, 256, 256, dt, prefold_scale=attn_scale * math.log2(math.e))
    q = q1 if prefolded else q0
    _, out, lse = _d256_fold_launch(q, k, v, with_stats=True, prefolded=prefolded, attn_scale=attn_scale, api_kw=dict(is_causal=True, window_size_left=33))
    ref_o, ref_lse, live = _d256_fold_oracle(q, k, v, scale=math.log(2.0) if prefolded else attn_scale, causal=True, window_left=33)
    assert live.all(), "geometry: every row keeps 34 keys"
    assert torch.isfinite(out.float()).all(), f"{int((~torch.isfinite(out.float())).sum())} non-finite O cells"
    assert torch.isfinite(lse).all(), f"{int((~torch.isfinite(lse)).sum())} non-finite LSE rows"
    err, amax = (out.double() - ref_o).abs().max().item(), ref_o.abs().max().item()
    assert err <= 0.1 * amax, f"O max err {err} vs the oracle (max|ref| {amax})"
    lse_err = (lse.double() - ref_lse).abs().max().item()
    assert lse_err <= 5e-4, f"LSE max err {lse_err:.2e} vs the oracle (natural log)"


@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
def test_d256_half_prefolded_scale_thd_matches_the_oracle(dtype_name):
    """THD (packed varlen) under the fold on the d256 half kernel, causal with token-major Stats: four sequences -- a full
    one, a zero-length one (no tokens, no rows), one with Q tokens but ZERO keys (every row keyless: O exactly 0 and LSE
    exactly -inf through the empty-range select, never the sentinel max), and a ragged 257-key one (a partial tail tile)
    -- each against its own float64 oracle of the pre-scaled Q (ln 2 scale), the storage past the packed totals untouched,
    and the dead-scale replay bit-identical."""
    import math

    import torch

    _d256_fold_board_only()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    b, hq, hkv, s = 4, 8, 2, 512  # the packed storage holds b * s tokens
    lens_q, lens_kv = [512, 0, 300, 257], [512, 0, 0, 257]
    attn_scale = _D256_FOLD_D**-0.5
    _, q1, k, v = _d256_fold_operands(b, hq, hkv, s, s, dt, prefold_scale=attn_scale * math.log2(math.e), seed=1)
    t_cap = b * s
    lse_buf = torch.full((t_cap * hq,), float("nan"), device="cuda", dtype=torch.float32)
    lse = lse_buf.as_strided((b, hq, s), (s * hq, 1, hq))  # the token-major declaration; execute binds the flat packed (T, H) storage
    q_lens = torch.tensor(lens_q, dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor(lens_kv, dtype=torch.int32, device="cuda")
    _, out, _ = _d256_fold_launch(
        q1,
        k,
        v,
        with_stats=True,
        prefolded=True,
        attn_scale=attn_scale,
        lse=lse,
        lse_exec=lse_buf.view(t_cap, hq),
        api_kw=dict(is_causal=True, thd=True),
        exec_kw=dict(seq_q_lens=q_lens, seq_kv_lens=kv_lens),
    )

    def tokens(t):  # token-major view of the BSHD storage behind a BHSD sample
        return t.transpose(1, 2).reshape(t_cap, t.shape[1], t.shape[3])

    q_tok, k_tok, v_tok, o_tok = (tokens(t) for t in (q1, k, v, out))
    lse_tok = lse_buf.view(t_cap, hq)
    t_q = sum(lens_q)
    assert torch.isfinite(o_tok[:t_q].float()).all(), "unwritten / non-finite O rows inside the packed total"
    assert torch.isnan(o_tok[t_q:].float()).all() and torch.isnan(lse_tok[t_q:]).all(), "wrote beyond the packed totals"
    assert not torch.isnan(lse_tok[:t_q]).any(), "unwritten LSE rows inside the packed total"
    cu_q = [0] + [sum(lens_q[: i + 1]) for i in range(b)]
    cu_k = [0] + [sum(lens_kv[: i + 1]) for i in range(b)]
    for i, (nq, nkv) in enumerate(zip(lens_q, lens_kv)):
        if nq == 0:
            continue
        o_i = o_tok[cu_q[i] : cu_q[i + 1]].permute(1, 0, 2).unsqueeze(0)
        lse_i = lse_tok[cu_q[i] : cu_q[i + 1]].t().unsqueeze(0)
        if nkv == 0:
            assert (o_i == 0).all() and torch.isneginf(lse_i).all(), f"sequence {i} has no key: O exactly 0, LSE exactly -inf"
            continue
        qb = q_tok[cu_q[i] : cu_q[i + 1]].permute(1, 0, 2).unsqueeze(0)
        kb = k_tok[cu_k[i] : cu_k[i + 1]].permute(1, 0, 2).unsqueeze(0)
        vb = v_tok[cu_k[i] : cu_k[i + 1]].permute(1, 0, 2).unsqueeze(0)
        ref_o, ref_lse, _ = _d256_fold_oracle(qb, kb, vb, scale=math.log(2.0), causal=True)
        err, amax = (o_i.double() - ref_o).abs().max().item(), ref_o.abs().max().item()
        assert err <= 0.1 * amax, f"sequence {i}: O max err {err} vs the oracle (max|ref| {amax})"
        torch.testing.assert_close(o_i.float(), ref_o.float(), **_GATE_O_TOL)
        lse_err = (lse_i.double() - ref_lse).abs().max().item()
        assert lse_err <= 5e-4, f"sequence {i}: LSE max err {lse_err} vs the oracle (natural log)"


# Keyless-row geometries (bottom of the brief's sentinel note): under the fold a fully-masked tile leaves the raw max at
# the finite sentinel, so the correction's _kv_empty select must be what publishes the keyless rows.
#   swa64_padded[_sink]: top-left causal + a 64-key left window + per-batch KV lengths (1000, 0, 129) on S_q = 256.  Batch 1
#     has no key at all (the KV range collapses); in batch 2 the rows past 128 + 64 have their whole window beyond the last
#     key while their CTA still walks two KV tiles -- consecutive fully-masked tiles, the is_first re-fire.  With a sink
#     those rows hold the sink's mass alone (LSE = sink exactly, O = 0).
#   br_short_kv: bottom-right causal with S_kv = 400 < S_q = 640 -- rows 0..239 sit above the diagonal: a whole keyless Q
#     tile (empty KV range) and, in the next tile, keyless rows beside live ones that see one fully-masked tile.
_D256_FOLD_KEYLESS = {
    "swa64_padded": dict(
        sink=False, b=3, s_q=256, s_kv=1024, kv_lens=(1000, 0, 129), api=dict(is_causal=True, window_size_left=64), ref=dict(causal=True, window_left=64)
    ),
    "swa64_padded_sink": dict(
        sink=True, b=3, s_q=256, s_kv=1024, kv_lens=(1000, 0, 129), api=dict(is_causal=True, window_size_left=64), ref=dict(causal=True, window_left=64)
    ),
    "br_short_kv": dict(
        sink=False, b=2, s_q=640, s_kv=400, kv_lens=None, api=dict(is_causal=True, causal_bottom_right=True), ref=dict(causal=True, bottom_right=True)
    ),
}


@pytest.mark.parametrize("geometry", list(_D256_FOLD_KEYLESS))
@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
def test_d256_half_prefolded_scale_keyless_rows(dtype_name, geometry):
    """Keyless rows under the fold on the d256 half kernel (with Stats): every row without a live key publishes O exactly 0
    and LSE exactly -inf -- or exactly the sink logit when the graph carries one -- while the live rows stay on the
    float64 oracle of the pre-scaled Q (O within 0.1 * max|ref| and the suite's shared tolerance, LSE within 5e-4 natural);
    the dead-scale replay is bit-identical."""
    import math

    import torch

    _d256_fold_board_only()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    g = _D256_FOLD_KEYLESS[geometry]
    hq, hkv = 8, 2
    attn_scale = _D256_FOLD_D**-0.5
    _, q1, k, v = _d256_fold_operands(g["b"], hq, hkv, g["s_q"], g["s_kv"], dt, prefold_scale=attn_scale * math.log2(math.e), seed=2)
    api_kw, ref_kw, exec_kw = dict(g["api"]), dict(g["ref"]), {}
    if g["kv_lens"] is not None:
        kv_lens = torch.tensor(g["kv_lens"], dtype=torch.int32, device="cuda")
        api_kw["seq_kv_lens_present"], exec_kw["seq_kv_lens"], ref_kw["seq_kv_lens"] = True, kv_lens, kv_lens
    if g["sink"]:
        sink = torch.randn(1, hq, 1, 1, dtype=torch.float32, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
        api_kw["has_sink"], exec_kw["sinks"], ref_kw["sinks"] = True, sink, sink.flatten()
    _, out, lse = _d256_fold_launch(q1, k, v, with_stats=True, prefolded=True, attn_scale=attn_scale, api_kw=api_kw, exec_kw=exec_kw)
    ref_o, ref_lse, live = _d256_fold_oracle(q1, k, v, scale=math.log(2.0), **ref_kw)
    assert (~live).any(), "the geometry must produce keyless rows"
    assert torch.isfinite(out.float()).all() and not torch.isnan(lse).any(), "unwritten / non-finite cells"
    dead_o = out[~live]
    assert (dead_o == 0).all(), "a keyless row writes O exactly 0 (a select, not residue * 0)"
    if g["sink"]:
        assert torch.equal(lse[~live], sink.view(1, hq, 1).expand_as(lse)[~live]), "a keyless row with a sink publishes exactly the sink logit"
    else:
        assert torch.isneginf(lse[~live]).all(), "a keyless row publishes LSE = -inf"
    err, amax = (out.double() - ref_o).abs().max().item(), ref_o.abs().max().item()
    assert err <= 0.1 * amax, f"O max err {err} vs the oracle (max|ref| {amax})"
    torch.testing.assert_close(out.float(), ref_o.float(), **_GATE_O_TOL)
    lse_err = (lse.double() - ref_lse)[live].abs().max().item()
    assert lse_err <= 5e-4, f"live-row LSE max err {lse_err} vs the oracle (natural log)"


# --- d256 MXFP8 softmax levers: HALF exponent, pre-folded scale, fused shift+convert ---------------------------------------
# The d256 MXFP8 forward normalizes O with a REGISTER row-sum (no ones-MMA Sigma), so its lever arms differ from the d128
# kernel's in the stats-less HALF case: O's denominator is the f16 pair-tree sum of the stored P words.  Every arm below is
# exercised end to end against a float64 oracle of the dequantized inputs; the masked draws (causal, SWA, bottom-right keyless
# rows, padded tail / zero-length KV) run the three masked loop segments as well as the dense one, all of which end in the
# kernel's one shared tail.

_D256_LEVER_ARMS = [("half", False), ("half", True), ("float", True)]
_D256_LEVER_ARM_IDS = ["half", "half-prefolded", "float-prefolded"]
_D256_LEVER_MASKS = {
    "dense": dict(is_causal=False),  # the HW-max segment only
    "causal": dict(is_causal=True),  # + the unmasked interior and the right-masked diagonal tiles
    "swa": dict(is_causal=True, window_size_left=200),  # + the left-masked band tiles (W = 200 spans two 128-wide tiles)
}


def _d256_mxfp8_lever_inputs(b, hq, hkv, s_q, s_kv, d, prefold_scale, fp8):
    """Random inputs quantized to MXFP8 (``fp8`` = E4M3 or E5M2) the way the engine consumes them, plus the DEQUANTIZED float64
    operands the oracle must see.  ``prefold_scale`` multiplies Q BEFORE quantization (the softmax_scale_prefolded contract)."""
    import torch
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    dev = "cuda"
    qf = torch.randn(b, hq, s_q, d, device=dev) * 0.5
    kf = torch.randn(b, hkv, s_kv, d, device=dev) * 0.5
    vf = torch.randn(b, hkv, s_kv, d, device=dev) * 0.5

    def mx(x, h, s, columnwise):
        # quantize_to_mxfp8 returns the FP8 data and the per-element DEQUANT SCALE (dq); dequantized = data * dq.
        data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(x.contiguous(), b, h, s, d, 32, fp8, with_ref=True)
        data, dq, swz = (data_s, dq_s, swz_s) if columnwise else (data_d, dq_d, swz_d)
        dequant = data.double() * dq.double().reshape(b, h, s, d)  # float64 oracle operand
        return data.permute(0, 2, 1, 3).contiguous().transpose(1, 2), swz.contiguous(), dequant  # BHSD view over BSHD storage

    q8, sfq, dq = mx(qf * prefold_scale, hq, s_q, False)
    k8, sfk, dk = mx(kf, hkv, s_kv, False)
    v8, sfv, dv = mx(vf, hkv, s_kv, True)
    return (q8, sfq, dq), (k8, sfk, dk), (v8, sfv, dv)


def _d256_mxfp8_lever_run(precision, prefolded, with_stats, mask_kw, fp8, *, b, hq, hkv, s_q, s_kv, causal_bottom_right=False, seq_kv_lens=None, seed=0):
    """Compile and run the d256 MXFP8 kernel through the adapter under the requested lever arms, then build the float64 oracle.

    Returns ``(out, lse, ref_o, ref_lse, keyless, api)``.  ``keyless`` is the [b, s_q] mask of rows with no live key (the
    kernel's contract there is O = 0 and LSE = -inf; the oracle's O is forced to 0 on those rows, its LSE is -inf).  ``ref_lse``
    is the natural-log LSE of the dequantized problem."""
    import math

    import torch

    from cudnn import data_type as cudnn_dtype
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    d = 256
    attn_scale = d**-0.5
    torch.manual_seed(seed)
    dev = "cuda"
    (q8, sfq, dq), (k8, sfk, dk), (v8, sfv, dv) = _d256_mxfp8_lever_inputs(b, hq, hkv, s_q, s_kv, d, attn_scale * math.log2(math.e) if prefolded else 1.0, fp8)
    out = torch.full((b, s_q, hq, d), float("nan"), device=dev, dtype=torch.bfloat16).transpose(1, 2)  # sentinel: an unclaimed tile stays visible
    lse = torch.full((b, hq, s_q), float("nan"), device=dev, dtype=torch.float32)
    kv_lens = None if seq_kv_lens is None else torch.tensor(seq_kv_lens, dtype=torch.int32, device=dev)
    api = SdpaFwdDslSm100(
        q8,
        k8,
        v8,
        out,
        lse if with_stats else None,
        scale_softmax=None if prefolded else attn_scale,
        causal_bottom_right=causal_bottom_right,
        seq_kv_lens_present=kv_lens is not None,
        pertensor_fp8=False,
        dtype_o=torch.bfloat16,
        cga=1,
        softmax_precision=cudnn_dtype.HALF if precision == "half" else None,
        softmax_scale_prefolded=prefolded,
        **mask_kw,
    )
    assert api.check_support()
    api.compile()
    mod = api._k_mod
    assert mod.__file__.endswith("sm107/prefill_d256_mxfp8.py"), mod.__file__
    assert (mod.SOFTMAX_F16, mod.SCALE_PREFOLDED) == (int(precision == "half"), int(prefolded)), "the adapter must compile the requested arms"
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    api.execute(q8, k8, v8, out, lse_tensor=lse if with_stats else None, seq_kv_lens=kv_lens, sf_q=sfq, sf_k=sfk, sf_v=sfv, workspace=ws)
    torch.cuda.synchronize()

    # Oracle (float64) on the dequantized inputs: a prefolded Q already carries attn_scale*log2(e), so its logits are
    # exp2-domain -> softmax_e(ln2 * S); otherwise softmax_e(attn_scale * S).
    rep = hq // hkv
    logits = (dq @ dk.repeat_interleave(rep, 1).transpose(-1, -2)) * (math.log(2.0) if prefolded else attn_scale)
    rows = torch.arange(s_q, device=dev)[:, None]
    cols = torch.arange(s_kv, device=dev)[None, :]
    masked = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=dev)
    if mask_kw.get("is_causal"):
        diag = rows + ((s_kv - s_q) if causal_bottom_right else 0)
        masked |= cols > diag
        if mask_kw.get("window_size_left") is not None:
            masked |= cols < diag - mask_kw["window_size_left"]
    if kv_lens is not None:
        masked |= cols >= kv_lens.view(b, 1, 1, 1)
    logits = logits.masked_fill(masked, float("-inf"))
    keyless = masked.all(-1)[:, 0]  # [b, s_q]
    ref_lse = torch.logsumexp(logits, dim=-1)  # -inf on keyless rows
    ref_o = torch.softmax(logits, dim=-1).nan_to_num(0.0) @ dv.repeat_interleave(rep, 1)  # keyless rows -> 0 (the kernel's contract)
    return out, lse, ref_o, ref_lse, keyless, api


def _d256_mxfp8_lever_check(out, lse, ref_o, ref_lse, keyless, with_stats, *, o_rel_tol=0.1, lse_tol=1e-4):
    """O within ``o_rel_tol`` * max|ref| and every cell written; keyless rows exactly O = 0 / LSE = -inf; on a Stats build the
    live rows' LSE within ``lse_tol`` (natural log) of the oracle.  Every Stats leg of this kernel keeps the exact f32 sum and
    measures ~1e-6 here; the f16 pair-tree sum fed to the LSE instead reads 5e-4..8e-4, so 1e-4 (the bound of
    test_mxfp8_stats_is_the_exact_softmax_lse) tells the two denominators apart on every draw while leaving a 100x margin."""
    import torch

    assert torch.isfinite(out).all(), "non-finite / unwritten O cells"
    scale = ref_o.abs().max().item()
    err = (out.double() - ref_o).abs().max().item()
    assert err <= o_rel_tol * scale, f"max err {err} vs oracle (scale {scale})"
    rows_o = out.permute(0, 2, 1, 3)  # [b, s_q, hq, d]
    if keyless.any():
        assert (rows_o[keyless] == 0).all(), "a keyless row must come back as O = 0 (the correction's _kv_empty select)"
    if with_stats:
        rows_lse = lse.permute(0, 2, 1)  # [b, s_q, hq]
        if keyless.any():
            assert torch.isneginf(rows_lse[keyless]).all(), "a keyless row's LSE must be -inf"
        live = ~keyless
        assert torch.isfinite(rows_lse[live]).all(), "unwritten LSE rows"
        lse_err = (lse.double() - ref_lse).abs().permute(0, 2, 1)[live].max().item()
        assert lse_err <= lse_tol, f"LSE max err {lse_err} vs oracle (natural log)"


def _d256_fused_expected(precision, prefolded):
    from cudnn.frost.tile_dsl import softmax_f16

    return precision == "half" and prefolded and softmax_f16.FUSED_SHIFT_CVT_AVAILABLE


def _requires_cc107():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 MXFP8 kernels serve cc10.7 only")


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False], ids=["stats", "nostats"])
@pytest.mark.parametrize("mask", list(_D256_LEVER_MASKS))
@pytest.mark.parametrize("precision, prefolded", _D256_LEVER_ARMS, ids=_D256_LEVER_ARM_IDS)
def test_d256_mxfp8_softmax_levers_match_the_oracle(precision, prefolded, mask, with_stats):
    """cc10.7 e2e for the softmax levers of the d256 MXFP8 kernel: softmax_precision=HALF (MUFU EX2.F16x2 + f16x2 -> FP8 cast;
    without Stats O is normalized by the f16 pair-tree sum of the stored P words, with Stats by the exact f32 sum) and
    softmax_scale_prefolded (raw tile max, plain shift; with HALF and no Stats the fused FHADD2 shift+convert arm).  Dense runs
    the HW-max loop segment; causal adds the unmasked interior and the right-masked diagonal tiles; SWA adds the left-masked band
    tiles -- the four segments that end in the one shared tail.  O within 0.1 * max|ref| of the float64 oracle on the DEQUANTIZED
    inputs, every O cell and LSE row written, LSE within 1e-4 (natural log): the pre-folded contract leaves the Stats domain
    unchanged and every Stats leg of this kernel keeps the exact f32 denominator."""
    import torch

    _requires_cc107()
    out, lse, ref_o, ref_lse, keyless, api = _d256_mxfp8_lever_run(
        precision, prefolded, with_stats, _D256_LEVER_MASKS[mask], torch.float8_e4m3fn, b=1, hq=8, hkv=2, s_q=1024, s_kv=1024
    )
    assert not keyless.any()
    # the stats-less HALF + prefolded build is the fused FHADD2 arm whenever the DSL exposes the op (the module constant is
    # independent of Stats; the Stats specialization keeps the shifted f32 scores and never takes the arm)
    assert bool(api._k_mod._FUSED_SHIFT_CVT) is _d256_fused_expected(precision, prefolded)
    _d256_mxfp8_lever_check(out, lse, ref_o, ref_lse, keyless, with_stats)


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False], ids=["stats", "nostats"])
@pytest.mark.parametrize("prefolded", [False, True], ids=["half", "half-prefolded"])
def test_d256_mxfp8_half_softmax_e5m2_matches_the_oracle(prefolded, with_stats):
    """E5M2 inputs under HALF: P casts f16x2 -> E5M2 pairs (``fp8_pair_tag`` = e5m2) in the unfused and the fused arm, with the
    f16 pair-tree denominator (no Stats) and the exact one (Stats).  Causal, so the masked segments run as well."""
    import torch

    _requires_cc107()
    out, lse, ref_o, ref_lse, keyless, api = _d256_mxfp8_lever_run(
        "half", prefolded, with_stats, _D256_LEVER_MASKS["causal"], torch.float8_e5m2, b=1, hq=8, hkv=2, s_q=1024, s_kv=1024
    )
    assert api._k_mod._FP8_TAG_P == "e5m2" and bool(api._k_mod._FUSED_SHIFT_CVT) is _d256_fused_expected("half", prefolded)
    _d256_mxfp8_lever_check(out, lse, ref_o, ref_lse, keyless, with_stats)


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False], ids=["stats", "nostats"])
@pytest.mark.parametrize("precision, prefolded", _D256_LEVER_ARMS, ids=_D256_LEVER_ARM_IDS)
def test_d256_mxfp8_softmax_levers_keyless_rows(precision, prefolded, with_stats):
    """Keyless rows under the levers: bottom-right causal with s_kv < s_q leaves the first s_q - s_kv query rows with no live key.
    Under the pre-folded scale a fully-masked tile leaves the raw tile max at the finite mask sentinel, so is_first re-fires on
    consecutive keyless tiles (alpha = 0 where the scaled chain ran alpha = 1); the correction's _kv_empty select must still
    write O = 0 and LSE = -inf there while the live rows match the oracle.  s_q = 1024, s_kv = 640: three fully keyless Q tiles,
    then live tiles whose diagonal sits 384 keys back."""
    import torch

    _requires_cc107()
    out, lse, ref_o, ref_lse, keyless, api = _d256_mxfp8_lever_run(
        precision, prefolded, with_stats, _D256_LEVER_MASKS["causal"], torch.float8_e4m3fn, b=1, hq=4, hkv=2, s_q=1024, s_kv=640, causal_bottom_right=True
    )
    assert keyless[0, :384].all() and not keyless[0, 384:].any(), "geometry: rows 0..383 are keyless"
    assert bool(api._k_mod._FUSED_SHIFT_CVT) is _d256_fused_expected(precision, prefolded)
    _d256_mxfp8_lever_check(out, lse, ref_o, ref_lse, keyless, with_stats)


@pytest.mark.L0
@pytest.mark.parametrize(
    "precision, prefolded, with_stats",
    [("half", False, False), ("half", True, False), ("float", True, False), ("half", True, True)],
    ids=["half-nostats", "half-prefolded-nostats", "float-prefolded-nostats", "half-prefolded-stats"],
)
def test_d256_mxfp8_softmax_levers_padded_kv_and_zero_length_sequence(precision, prefolded, with_stats):
    """Padding mask under the levers: seq_kv_lens = [1024, 320, 0] -- a full batch, a batch whose third KV tile is half padded
    (the padded slow path through apply_mask_chunk, then the shared tail) and a zero-length KV batch (empty mainloop: the tail
    never runs and the epilogue's select owns O = 0 / LSE = -inf).  Dense (no causal) so the padded arm is the only mask."""
    import torch

    _requires_cc107()
    out, lse, ref_o, ref_lse, keyless, api = _d256_mxfp8_lever_run(
        precision, prefolded, with_stats, _D256_LEVER_MASKS["dense"], torch.float8_e4m3fn, b=3, hq=4, hkv=2, s_q=512, s_kv=1024, seq_kv_lens=[1024, 320, 0]
    )
    assert keyless[2].all() and not keyless[:2].any(), "geometry: only the zero-length batch is keyless"
    _d256_mxfp8_lever_check(out, lse, ref_o, ref_lse, keyless, with_stats)


@pytest.mark.L0
def test_d256_mxfp8_softmax_tail_is_shared_by_every_segment():
    """Source and module pins for the d256 MXFP8 forward's lever port (no GPU).  Its four kv-loop segments (dense / left-masked /
    unmasked interior / right-masked) end in ONE module-level ``@cute.jit`` tail, ``_softmax_tail``, called with the same
    argument list from each; the exponent, scale, FP8 cast, P publish and row-sum chains live only in that helper, so a segment
    cannot drift to a stale copy of an arm.  The helper carries every arm (fused / f16-sum / exact-sum / f32), the kernel threads
    ``has_lse`` into the softmax warp group, and the module constants follow TemplateParams for both FP8 formats."""
    import re

    from cudnn.frost.tile_dsl import softmax_f16

    mod = _load(_D256, rubin=True, **_MXFP8_LOAD_KW)
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    helper = code[code.index("def _softmax_tail(") : code.index("def _softmax_warp_group(")]
    body = code[code.index("def _softmax_warp_group(") : code.index("def _correction_warp_group(")]
    calls = re.findall(r"total_max, total_sum = _softmax_tail\((.*?)\n\s*\)", body, re.S)
    assert len(calls) == 4, f"{len(calls)} _softmax_tail call sites in _softmax_warp_group, the four loop segments need one each"
    assert len({re.sub(r"\s+", "", c) for c in calls}) == 1, "every segment passes the tail the same argument list"
    assert re.sub(r"\s+", "", calls[0]).startswith("has_lse,reg_S_a,reg_S_b,max_a,max_b,scale_log2,total_max,total_sum,")
    for chain in (
        "cute.math.exp2(",
        "* scale_log2",
        ".to(STORAGE_DTYPE)",
        "row_reduction_pair_64(",
        "make_tmem_ptr(p_addr",
        "mb_bmm2_ready[",
        "RESCALE_THRESHOLD",
    ):
        assert chain not in body, f"{chain!r} outside _softmax_tail: a segment carries its own copy of the chain"
        assert chain in helper, f"{chain!r} missing from _softmax_tail"
    for arm in (
        "_FUSED_SHIFT_CVT and not has_lse",
        "_softmax_f16.fused_shift_f16_exp_chunk_f16sum(",
        "_softmax_f16.f16_exp_chunk_f16sum(",
        "_softmax_f16.f16_exp_chunk_sum(",
        "make_tmem_ptr(p_addr_a, cutlass.Int32)",
        "make_tmem_ptr(p_addr_b, cutlass.Int32)",
        "make_tmem_ptr(p_addr_a, cutlass.Float32)",
        "make_tmem_ptr(p_addr_b, cutlass.Float32)",
        "reg_S_a = reg_S_a - new_total_max",
        "reg_S_a = reg_S_a * scale_log2 - new_total_max",
        "raw_max = cute.math.max(max_a, max_b)\n",
        "current_max = raw_max\n",
        "current_max = raw_max * scale_log2\n",
    ):
        assert arm in helper, f"_softmax_tail lacks the arm {arm!r}"
    assert (
        "total_max,alpha,new_total_max=running_max_step_finite_sentinel(raw_max,current_max,total_max,NEG_INF,RESCALE_THRESHOLD,masked=CFG.MASK_FLAGS!=MASK_NONE)"
        in re.sub(r"\s+", "", helper)
    ), "the running-max step (the leading-dead-tile guard) is the shared finite-sentinel helper, called with the RAW tile max"
    assert "fused_shift_f16_exp_chunk(" not in helper and "f16_exp_chunk(" not in helper.replace(
        "f16_exp_chunk_", ""
    ), "the ones-MMA (sum-less) helpers do not serve a register-sum kernel"
    assert (
        "has_lse: cutlass.Constexpr[bool]" in body and "has_lse=lse_tensor is not None" in code
    ), "has_lse is threaded from the kernel into the softmax warp group"
    assert "_softmax_f16.FUSED_SHIFT_CVT_AVAILABLE" in code and "_softmax_f16.fp8_pair_tag(CFG.DTYPE_QKV)" in code
    for dtype, tag in ((_E4M3, "e4m3"), (1, "e5m2")):
        for f16 in (False, True):
            for fold in (False, True):
                m = _load(_D256, rubin=True, **{**_MXFP8_LOAD_KW, "dtype_qkv": dtype}, softmax_f16=f16, softmax_scale_prefolded=fold)
                assert (m.SOFTMAX_F16, m.SCALE_PREFOLDED, m._FP8_TAG_P) == (int(f16), int(fold), tag), (dtype, f16, fold)
                assert m._FUSED_SHIFT_CVT is (f16 and fold and softmax_f16.FUSED_SHIFT_CVT_AVAILABLE), (dtype, f16, fold)


# PTX arm markers of the d256 MXFP8 lever port.  The levers are invisible to the oracle in two respects by design -- the fold
# runs numerically identical FADD2-for-FFMA2 (its scale is pinned to 1.0) and the fused shift+convert differs from the unfused
# HALF arm by one f16 ulp of exp input -- so the PTX of the dense build (ONE kv-loop segment, 128 scores = 64 pairs per thread
# per kv tile) is the tripwire: which exponent runs, which cast feeds the FP8 words, whether the shift fused, and which row-sum
# the build traces.  Counts measured 2026-10-06 (cutlass-dsl 4.8.0, sm_107a); the one f32 ex2 of every HALF build is alpha.
_D256_PTX_PROBE = textwrap.dedent("""
    import glob, os, re, sys
    dump, f16, fold, has_lse, dtype_qkv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx"
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    params = TemplateParams(dtype_qkv=dtype_qkv, dtype_o=2, cta_mma=1, softmax_f16=bool(f16), softmax_scale_prefolded=bool(fold))
    mod = _load_sm100_kernel_module((256, 256), params, fp8=True, pertensor=False, rubin=True)
    print("MODULE FUSED", int(mod._FUSED_SHIFT_CVT))
    mod.compile_prepared(has_lse=bool(has_lse))
    ptx = sorted(glob.glob(os.path.join(dump, "**", "*.ptx"), recursive=True), key=os.path.getmtime)
    if not ptx:
        print("FAIL no ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    text = open(ptx[-1], encoding="utf-8").read()
    for name, pat in (
        ("EX2_F32", r"ex2\\.approx(?:\\.ftz)?\\.f32\\b"),
        ("EX2_F16X2", r"ex2\\.approx(?:\\.ftz)?\\.f16x2\\b"),
        ("CVT_F16X2_F32", r"cvt\\.rn\\.f16x2\\.f32\\b"),
        ("FUSED_SUB_F16X2_F32X2", r"sub\\.rz(?:\\.ftz)?\\.f16x2\\.f32x2"),
        ("CVT_E4M3X2_F32", r"cvt\\.rn\\.satfinite\\.e4m3x2\\.f32\\b"),
        ("CVT_E4M3X2_F16X2", r"cvt\\.rn\\.satfinite\\.e4m3x2\\.f16x2\\b"),
        ("CVT_E5M2X2_F16X2", r"cvt\\.rn\\.satfinite\\.e5m2x2\\.f16x2\\b"),
        ("ADD_F16X2", r"\\badd(?:\\.rn)?\\.f16x2\\b"),
    ):
        print("PTX", name, len(re.findall(pat, text)))
    """)
_D256_PTX_ROWS = [
    # (id, softmax_f16, prefolded, has_lse, dtype_qkv, expected counts)
    pytest.param(
        0, 0, 0, _E4M3, dict(EX2_F32=129, EX2_F16X2=0, CVT_F16X2_F32=0, FUSED_SUB_F16X2_F32X2=0, CVT_E4M3X2_F32=64, CVT_E4M3X2_F16X2=0, ADD_F16X2=0), id="float"
    ),
    pytest.param(
        1,
        0,
        0,
        _E4M3,
        dict(EX2_F32=1, EX2_F16X2=64, CVT_F16X2_F32=64, FUSED_SUB_F16X2_F32X2=0, CVT_E4M3X2_F32=0, CVT_E4M3X2_F16X2=64, ADD_F16X2=56),
        id="half-nostats",
    ),
    pytest.param(
        1,
        1,
        0,
        _E4M3,
        dict(EX2_F32=1, EX2_F16X2=64, CVT_F16X2_F32=0, FUSED_SUB_F16X2_F32X2=64, CVT_E4M3X2_F32=0, CVT_E4M3X2_F16X2=64, ADD_F16X2=56),
        id="half-prefolded-nostats",
    ),
    pytest.param(
        1,
        0,
        1,
        _E4M3,
        dict(EX2_F32=129, EX2_F16X2=64, CVT_F16X2_F32=64, FUSED_SUB_F16X2_F32X2=0, CVT_E4M3X2_F32=0, CVT_E4M3X2_F16X2=64, ADD_F16X2=0),
        id="half-stats",
    ),
    pytest.param(
        1,
        1,
        1,
        _E4M3,
        dict(EX2_F32=129, EX2_F16X2=64, CVT_F16X2_F32=64, FUSED_SUB_F16X2_F32X2=0, CVT_E4M3X2_F32=0, CVT_E4M3X2_F16X2=64, ADD_F16X2=0),
        id="half-prefolded-stats",
    ),
    pytest.param(
        1,
        1,
        0,
        1,
        dict(EX2_F32=1, EX2_F16X2=64, CVT_F16X2_F32=0, FUSED_SUB_F16X2_F32X2=64, CVT_E4M3X2_F16X2=0, CVT_E5M2X2_F16X2=64, ADD_F16X2=56),
        id="half-prefolded-nostats-e5m2",
    ),
]


@pytest.mark.parametrize("f16, fold, has_lse, dtype_qkv, expected", _D256_PTX_ROWS)
def test_d256_mxfp8_softmax_lever_arms_reach_the_ptx(tmp_path, f16, fold, has_lse, dtype_qkv, expected):
    """Each lever build of the d256 MXFP8 kernel traces the arm it claims (dense build, sm_107a PTX): FLOAT runs 128 f32 ex2 + alpha
    and casts FP8 from f32; HALF runs 64 ex2.f16x2 and casts the FP8 words from f16x2 (e4m3 or e5m2 by TemplateParams.dtype_qkv);
    the stats-less HALF arm sums its P words with the 56-add f16x2 pair tree, while the Stats build keeps the exact f32 sum (the
    128 f32 ex2 come back, the tree goes); HALF + pre-folded without Stats fuses shift and convert (64 packed f16x2 <- f32x2
    subtracts, zero f32 -> f16x2 converts) and with Stats never does.  Compile-only, any device whose DSL knows sm_107a."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    from cudnn.frost.tile_dsl import softmax_f16

    if fold and f16 and not has_lse and not softmax_f16.FUSED_SHIFT_CVT_AVAILABLE:
        pytest.skip("this cutlass-dsl has no packed f16x2 <- f32x2 subtract; the fused arm is not traced")
    dump = tmp_path / f"d256_mxfp8_ptx_f16{f16}_fold{fold}_lse{has_lse}_dt{dtype_qkv}"
    dump.mkdir()
    argv = [sys.executable, "-c", _D256_PTX_PROBE, str(dump), str(f16), str(fold), str(has_lse), str(dtype_qkv)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_107a trace-compile of the d256 MXFP8 build failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    counts = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("PTX ") and len(ln.split()) == 3}
    fused = [int(ln.split()[2]) for ln in out if ln.startswith("MODULE FUSED")]
    print(f"\nd256 mxfp8 f16={f16} fold={fold} has_lse={has_lse} dtype_qkv={dtype_qkv} sm_107a PTX: {counts}")
    assert fused == [int(bool(f16 and fold))], f"_FUSED_SHIFT_CVT {fused} for softmax_f16={f16} softmax_scale_prefolded={fold}"
    got = {k: counts[k] for k in expected}
    assert got == expected, f"PTX arm markers {got} != expected {expected}"


# ---------------------------------------------------------------------------------------------------------------------
# The pre-folded softmax scale on the half-input d128 / d192x128 twins (byte-identical softmax bodies).
#
# softmax_scale_prefolded (graph: attn_scale_prefolded): Q arrives multiplied by attn_scale * log2(e), the kernel takes the
# RAW row max and shifts with `S - m`, and `scale_log2` is a dead runtime argument.  The lever is numerically neutral by
# contract, so each leg below is held against a float64 oracle of the half inputs it actually saw (the prefolded Q rounds
# differently from the unfolded one, so the two legs are two problems, each with its own oracle): O within 0.1 * max|ref|
# AND within 2x the unfolded leg's error (plus one output ulp at max|ref|, so a lone rounding flip of the half O store
# cannot fail it), LSE within 5e-4 (natural log).  Both twins, both half dtypes, dense / causal / causal+SWA, Stats on /
# off, split_kv 1 / 2; then THD (a sequence with more queries than keys and a zero-length KV sequence) and dense
# bottom-right rows above the diagonal, with and without a sink -- rows with no key at all, which the correction's
# geometry select (not the max, which the fold leaves at the finite sentinel) must zero.
# ---------------------------------------------------------------------------------------------------------------------

_HALF_PREFOLD_TWINS = [(128, 128), (192, 128)]
# mask -> (adapter / oracle mask kwargs, geometry)
_HALF_PREFOLD_MASKS = {
    "dense": (dict(), dict(b=1, hq=8, hkv=2, s=1024)),
    "causal": (dict(causal=True), dict(b=2, hq=8, hkv=4, s=1024)),
    "swa": (dict(causal=True, window_left=255), dict(b=2, hq=8, hkv=4, s=1024)),
}


def _half_prefold_twin_file(d_qk):
    return "sm107/prefill_d128_f16.py" if d_qk == 128 else "sm107/prefill_d192_d128_f16.py"


def _half_prefold_tensors(b, hq, hkv, s_q, s_kv, d_qk, d_v, dtype, prefold_scale, *, seed):
    """Half Q / K / V in BSHD storage (BHSD views) the way the engine consumes them.  ``prefold_scale`` multiplies the f32 Q
    BEFORE the half cast (the softmax_scale_prefolded contract; 1.0 on the unfolded leg)."""
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, device="cuda", generator=g) * 0.5

    q = (rnd(b, s_q, hq, d_qk) * prefold_scale).to(dtype).transpose(1, 2)
    k = rnd(b, s_kv, hkv, d_qk).to(dtype).transpose(1, 2)
    v = rnd(b, s_kv, hkv, d_v).to(dtype).transpose(1, 2)
    return q, k, v


def _dense_mask(s_q, s_kv, *, causal=False, bottom_right=False, window_left=None):
    """The (s_q, s_kv) bool mask of the dense band the adapter kwargs describe, or None for no mask."""
    import torch

    if not causal and window_left is None:
        return None
    i = torch.arange(s_q, device="cuda")[:, None] + ((s_kv - s_q) if bottom_right else 0)
    j = torch.arange(s_kv, device="cuda")[None, :]
    masked = torch.zeros(s_q, s_kv, dtype=torch.bool, device="cuda")
    if causal:
        masked |= j > i
    if window_left is not None:
        masked |= j < i - window_left
    return masked


def _half_oracle(q, k, v, *, logit_scale, masked=None, sinks=None):
    """float64 softmax(logit_scale * Q K^T) V on the half inputs the kernel saw (BHSD, GQA expanded) -> (O, LSE natural).
    ``masked`` broadcasts over (..., s_q, s_kv).  A keyless row gives O = 0 and LSE = -inf, or LSE = the head's sink logit
    when a sink joins the row as one extra column (the sink's mass is the row's whole denominator then)."""
    import torch

    rep = q.shape[1] // k.shape[1]
    logits = (q.double() @ k.double().repeat_interleave(rep, 1).transpose(-1, -2)) * logit_scale
    if masked is not None:
        logits = logits.masked_fill(masked, float("-inf"))
    full = logits if sinks is None else torch.cat([logits, sinks.double().view(1, -1, 1, 1).expand(*logits.shape[:3], 1)], dim=-1)
    probs = torch.softmax(full, dim=-1).nan_to_num(0.0)[..., : logits.shape[-1]]
    return probs @ v.double().repeat_interleave(rep, 1), torch.logsumexp(full, dim=-1)


def _half_prefold_check(out, lse, ref_o, ref_lse, *, tag):
    """One leg's oracle bounds: every O cell written and finite, O within 0.1 * max|ref|, LSE rows written where the
    oracle's are finite and within 5e-4 natural, -inf exactly where the oracle's are.  Returns (O max abs err, max|ref|)."""
    import torch

    assert torch.isfinite(out).all(), f"{tag}: non-finite / unwritten O cells"
    scale = ref_o.abs().max().item()
    err = (out.double() - ref_o).abs().max().item()
    assert err <= 0.1 * scale, f"{tag}: O max err {err} vs oracle (scale {scale})"
    if lse is not None:
        live = torch.isfinite(ref_lse)
        assert torch.isfinite(lse[live]).all(), f"{tag}: unwritten / non-finite LSE rows"
        assert torch.equal(torch.isneginf(lse), torch.isneginf(ref_lse)), f"{tag}: LSE = -inf exactly on the keyless rows, nowhere else"
        lse_err = (lse.double()[live] - ref_lse[live]).abs().max().item() if bool(live.any()) else 0.0
        assert lse_err <= 5e-4, f"{tag}: LSE max err {lse_err} vs oracle (natural log)"
    return err, scale


def _fold_within_unfolded(err_fold, err_unfolded, scale, dtype, *, tag):
    """The fold is numerically neutral: its O error stays within 2x the unfolded leg's, plus one output ulp at max|ref|."""
    import torch

    ulp = torch.finfo(dtype).eps * scale
    assert err_fold <= 2.0 * err_unfolded + ulp, f"{tag}: fold err {err_fold} vs unfolded {err_unfolded} (+ one ulp {ulp})"


def _half_prefold_api(
    d_qk,
    d_v,
    dtype,
    *,
    prefolded,
    with_stats,
    split_kv,
    b,
    hq,
    hkv,
    s_q,
    s_kv,
    causal=False,
    bottom_right=False,
    window_left=None,
    sink=False,
    seed=0,
    attn_scale=None,
):
    """The adapter of one leg (folded or unfolded) of a dense problem on the cc 10.7 half twin, plus its tensors.
    ``attn_scale`` defaults to d_qk ** -0.5; the leading-dead-tile cells pass 1.0 (scale_log2 > 1)."""
    import math

    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    attn_scale = d_qk**-0.5 if attn_scale is None else attn_scale
    q, k, v = _half_prefold_tensors(b, hq, hkv, s_q, s_kv, d_qk, d_v, dtype, attn_scale * math.log2(math.e) if prefolded else 1.0, seed=seed)
    out = torch.full((b, s_q, hq, d_v), float("nan"), device="cuda", dtype=dtype).transpose(1, 2)  # sentinel: an unclaimed tile stays visible
    lse = torch.full((b, hq, s_q), float("nan"), device="cuda", dtype=torch.float32) if with_stats else None
    sinks = torch.linspace(-2.0, 6.0, hq, device="cuda", dtype=torch.float32) if sink else None
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=out,
        sample_lse=lse,
        scale_softmax=None if prefolded else attn_scale,
        is_causal=causal,
        causal_bottom_right=bottom_right,
        window_size_left=window_left,
        has_sink=sink,
        cga=2,
        split_kv=split_kv,
        pack_gqa=False,
        softmax_scale_prefolded=prefolded,
    )
    return api, dict(q=q, k=k, v=v, out=out, lse=lse, sinks=sinks, attn_scale=attn_scale)


def _half_prefold_leg(
    d_qk, d_v, dtype, *, prefolded, with_stats, split_kv, b, hq, hkv, s_q, s_kv, causal=False, bottom_right=False, window_left=None, sink=False, attn_scale=None
):
    """Run one leg and hold it to its oracle; returns (O max abs err, max|ref|, out, lse, ref_lse)."""
    import math

    import torch

    api, t = _half_prefold_api(
        d_qk,
        d_v,
        dtype,
        prefolded=prefolded,
        with_stats=with_stats,
        split_kv=split_kv,
        b=b,
        hq=hq,
        hkv=hkv,
        s_q=s_q,
        s_kv=s_kv,
        causal=causal,
        bottom_right=bottom_right,
        window_left=window_left,
        sink=sink,
        attn_scale=attn_scale,
    )
    assert api.check_support()
    api.compile()
    # the build that ran is the twin's, specialized by the fold flag (a compile-time PARAMS fact, not a runtime scale)
    assert api._k_mod.__file__.endswith(_half_prefold_twin_file(d_qk)), api._k_mod.__file__
    assert int(api._k_mod.SCALE_PREFOLDED) == int(prefolded)
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    api.execute(q_tensor=t["q"], k_tensor=t["k"], v_tensor=t["v"], o_tensor=t["out"], lse_tensor=t["lse"], sinks=t["sinks"], workspace=ws)
    torch.cuda.synchronize()
    masked = _dense_mask(s_q, s_kv, causal=causal, bottom_right=bottom_right, window_left=window_left)
    ref_o, ref_lse = _half_oracle(t["q"], t["k"], t["v"], logit_scale=math.log(2.0) if prefolded else t["attn_scale"], masked=masked, sinks=t["sinks"])
    tag = f"{'fold' if prefolded else 'unfolded'} d{d_qk}x{d_v} {str(dtype).rsplit('.', 1)[-1]}"
    err, scale = _half_prefold_check(t["out"], t["lse"], ref_o, ref_lse, tag=tag)
    return err, scale, t["out"], t["lse"], ref_lse


@pytest.mark.parametrize("split_kv", [1, 2], ids=["split1", "split2"])
@pytest.mark.parametrize("with_stats", [True, False], ids=["stats", "nostats"])
@pytest.mark.parametrize("mask", list(_HALF_PREFOLD_MASKS))
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("d_qk, d_v", _HALF_PREFOLD_TWINS)
def test_half_prefolded_scale_matches_the_oracle(d_qk, d_v, dtype_name, mask, with_stats, split_kv):
    """cc10.7 e2e for softmax_scale_prefolded on the half d128 / d192x128 twins (FLOAT softmax; HALF stays declined on half
    inputs): Q carries attn_scale * log2(e), the kernel takes the raw row max and shifts with S - m.  The folded leg is held
    to the float64 oracle of the pre-scaled Q with ln 2 as the logit scale (O within 0.1 * max|ref|, LSE within 5e-4
    natural) and to the unfolded leg of the same problem (O error within 2x + one output ulp): with Stats, without Stats
    (the register row-sum still normalizes O), and through the split-KV combine (fp32 partials + natural partial LSE,
    scale-free), on the dense, causal and causal + sliding-window bands."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 half kernels serve cc10.7 only")
    dtype = getattr(torch, dtype_name)
    mask_kw, geom = _HALF_PREFOLD_MASKS[mask]
    common = dict(with_stats=with_stats, split_kv=split_kv, b=geom["b"], hq=geom["hq"], hkv=geom["hkv"], s_q=geom["s"], s_kv=geom["s"], **mask_kw)
    err_fold, scale, *_ = _half_prefold_leg(d_qk, d_v, dtype, prefolded=True, **common)
    err_unfolded, *_ = _half_prefold_leg(d_qk, d_v, dtype, prefolded=False, **common)
    _fold_within_unfolded(err_fold, err_unfolded, scale, dtype, tag=f"d{d_qk}x{d_v} {dtype_name} {mask} stats={with_stats} split={split_kv}")


@pytest.mark.L0
@pytest.mark.parametrize("prefolded", [False, True], ids=["scaled", "prefolded"])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("d_qk, d_v", _HALF_PREFOLD_TWINS)
def test_half_masked_leading_tile_keeps_rows_with_later_keys_finite(d_qk, d_v, dtype_name, prefolded):
    """A row whose FIRST KV tile is fully masked while a LATER tile holds its keys, on the half d128 / d192x128 twins:
    top-left causal with a 34-key band (window_size_left 33) at S = 256 -- rows 161..255 have no key in tile 0 (keys 0..127)
    and their 34 keys in tile 1 -- at attn_scale 1.  The scaled chain took the finite mask sentinel times scale_log2 > 1
    (= -inf) as the running max and read -inf - (-inf) = NaN into P (NaN O on every such row); the pre-folded chain kept the
    raw sentinel and published P = 1 per masked column, wiped only by alpha = 0 at the next live tile.  Both chains now select
    a tile that is dead ahead of the first live key out of the running state (total_max kept, alpha = 1, P = 0): O and LSE
    finite and at the float64 oracle of the half operands each chain saw (every row has keys, so the oracle is finite)."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 half kernels serve cc10.7 only")
    dtype = getattr(torch, dtype_name)
    err, scale, out, lse, ref_lse = _half_prefold_leg(
        d_qk, d_v, dtype, prefolded=prefolded, with_stats=True, split_kv=1, b=1, hq=8, hkv=2, s_q=256, s_kv=256, causal=True, window_left=33, attn_scale=1.0
    )
    assert torch.isfinite(ref_lse).all(), "geometry: every row keeps 34 keys"
    assert torch.isfinite(lse).all() and torch.isfinite(out).all()


@pytest.mark.L0
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("d_qk, d_v", _HALF_PREFOLD_TWINS)
def test_half_masked_leading_tile_bottom_right_padded_keeps_rows_with_later_keys_finite(d_qk, d_v, dtype_name):
    """The padded bottom-right twin of the geometry on the half d128 / d192x128 kernels: per-batch lengths Q 65 / KV 193 on
    a 256 x 256 problem, bottom-right causal (the diagonal 128 keys back) with a 34-key band at attn_scale 1 -- rows 33..64 see
    KV tile 0 fully masked and their keys in tile 1, rows 0..32 keep keys in tile 0, rows 65..255 are padded (O = 0, LSE =
    -inf through the dead-row select).  The fp64 reference composes the same padding, diagonal and window."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 half kernels serve cc10.7 only")
    dtype = getattr(torch, dtype_name)
    b, hq, hkv, s = 1, 8, 2, 256
    q, k, v = _half_prefold_tensors(b, hq, hkv, s, s, d_qk, d_v, dtype, 1.0, seed=0)
    q_lens = torch.tensor([65], dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor([193], dtype=torch.int32, device="cuda")
    o = torch.empty(b, s, hq, d_v, device="cuda", dtype=dtype).transpose(1, 2)
    lse = torch.empty(b, hq, s, device="cuda", dtype=torch.float32)
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse,
        is_causal=True,
        causal_bottom_right=True,
        window_size_left=33,
        scale_softmax=1.0,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
        cga=2,
    )
    assert api.check_support()
    api.compile()
    assert api._k_mod.__file__.endswith(_half_prefold_twin_file(d_qk)), api._k_mod.__file__
    out, lse_out = _handoff_execute(api, q, k, v, o, lse, q_lens, kv_lens)
    ref_o, ref_lse = _handoff_reference(q, k, v, scale=1.0, causal_br=True, window_left=33, q_lens=q_lens, kv_lens=kv_lens)
    assert torch.isfinite(ref_lse[..., :65]).all() and torch.isneginf(ref_lse[..., 65:]).all(), "geometry: rows 0..64 have keys, the rest are padded"
    _handoff_check(out, lse_out, ref_o, ref_lse, q_lens, tag=f"d{d_qk}x{d_v} {dtype_name} bottom-right padded leading tile")


@pytest.mark.parametrize("sink", [False, True], ids=["nosink", "sink"])
@pytest.mark.parametrize("d_qk, d_v", _HALF_PREFOLD_TWINS)
def test_half_prefolded_scale_zeroes_keyless_bottom_right_rows(d_qk, d_v, sink):
    """Dense bottom-right causal with s_kv < s_q: the first s_q - s_kv rows of every head have no key at all.  Under the fold
    such a row's raw max stays at the finite mask sentinel (the scaled path leaves it at sentinel * scale) and its softmax
    sum is N, not 0, so it is the correction's geometry select -- not the max -- that must zero O and publish LSE = -inf,
    or the head's sink logit with a sink (the sink's mass is the row's whole denominator).  Checked exactly on both legs;
    the live rows below the diagonal are held to the oracle and to each other as in the dense matrix."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 half kernels serve cc10.7 only")
    dtype = torch.float16
    b, hq, hkv, s_q, s_kv = 2, 8, 4, 256, 128
    dead = s_q - s_kv
    common = dict(with_stats=True, split_kv=1, b=b, hq=hq, hkv=hkv, s_q=s_q, s_kv=s_kv, causal=True, bottom_right=True, sink=sink)
    errs = {}
    for prefolded in (True, False):
        err, scale, out, lse, ref_lse = _half_prefold_leg(d_qk, d_v, dtype, prefolded=prefolded, **common)
        errs[prefolded] = (err, scale)
        assert (out[:, :, :dead] == 0).all(), f"prefolded={prefolded}: keyless rows must store O = 0 exactly"
        if sink:
            sinks = torch.linspace(-2.0, 6.0, hq, device="cuda", dtype=torch.float32)
            assert torch.equal(lse[:, :, :dead], sinks.view(1, hq, 1).expand(b, hq, dead)), f"prefolded={prefolded}: keyless rows publish the sink logit"
        else:
            assert torch.isneginf(lse[:, :, :dead]).all(), f"prefolded={prefolded}: keyless rows publish LSE = -inf"
        assert torch.isfinite(lse[:, :, dead:]).all()
    _fold_within_unfolded(errs[True][0], errs[False][0], errs[True][1], dtype, tag=f"d{d_qk}x{d_v} bottom-right s_kv<s_q sink={sink}")


@pytest.mark.parametrize("d_qk, d_v", _HALF_PREFOLD_TWINS)
def test_half_prefolded_scale_thd_matches_the_oracle(d_qk, d_v):
    """THD (packed varlen, per-sequence lengths, bottom-right causal) under the fold on both twins, fp16, with Stats: the
    same softmax body behind the varlen scheduler and the token-major LSE store.  One sequence has more queries than
    keys (its leading rows have no key) and one has a zero-length KV (every row keyless): those rows must store O = 0 and
    LSE = -inf exactly, the rest match the oracle and the unfolded leg."""
    import math

    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 half kernels serve cc10.7 only")
    dtype = torch.float16
    b, h, hk = 3, 8, 4
    q_lengths, k_lengths = [257, 161, 33], [385, 65, 0]
    sq, sk = max(q_lengths), max(k_lengths)
    tq, tk = sum(q_lengths), sum(k_lengths)
    attn_scale = d_qk**-0.5

    def view(raw, heads, length, width):
        return raw.as_strided((b, heads, length, width), (length * heads * width, width, heads * width, 1))

    def leg(prefolded):
        g = torch.Generator(device="cuda").manual_seed(7)
        pre = attn_scale * math.log2(math.e) if prefolded else 1.0
        q = torch.full((b * sq, h, d_qk), float("nan"), device="cuda", dtype=dtype)
        k = torch.full((b * sk, hk, d_qk), float("nan"), device="cuda", dtype=dtype)
        v = torch.full((b * sk, hk, d_v), float("nan"), device="cuda", dtype=dtype)
        q[:tq] = (torch.randn(tq, h, d_qk, device="cuda", generator=g) * 0.5 * pre).to(dtype)
        k[:tk] = (torch.randn(tk, hk, d_qk, device="cuda", generator=g) * 0.5).to(dtype)
        v[:tk] = (torch.randn(tk, hk, d_v, device="cuda", generator=g) * 0.5).to(dtype)
        out = torch.full((b * sq, h, d_v), float("nan"), device="cuda", dtype=dtype)
        stats = torch.full((b * sq, h), float("nan"), device="cuda", dtype=torch.float32)
        views = (view(q, h, sq, d_qk), view(k, hk, sk, d_qk), view(v, hk, sk, d_v), view(out, h, sq, d_v), stats.as_strided((b, h, sq), (sq * h, 1, h)))
        api = SdpaFwdDslSm100(
            sample_q=views[0],
            sample_k=views[1],
            sample_v=views[2],
            sample_o=views[3],
            sample_lse=views[4],
            scale_softmax=None if prefolded else attn_scale,
            thd=True,
            is_causal=True,
            causal_bottom_right=True,
            seq_kv_lens_present=True,
            cga=2,
            split_kv=1,
            pack_gqa=False,
            softmax_scale_prefolded=prefolded,
        )
        assert api.check_support()
        api.compile()
        assert api._k_mod.__file__.endswith(_half_prefold_twin_file(d_qk)) and api._k_mod.CFG.THD_VARLEN
        assert int(api._k_mod.SCALE_PREFOLDED) == int(prefolded)
        ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
        # the token-major Stats are bound PACKED (T, H) at execute; the strided (B, H, S) view only declared the layout
        api.execute(
            *views[:4],
            stats,
            seq_q_lens=torch.tensor(q_lengths, dtype=torch.int32, device="cuda"),
            seq_kv_lens=torch.tensor(k_lengths, dtype=torch.int32, device="cuda"),
            workspace=ws,
        )
        torch.cuda.synchronize()
        logit_scale = math.log(2.0) if prefolded else attn_scale
        err = scale = 0.0
        qb = kb = 0
        for nq, nk in zip(q_lengths, k_lengths):
            o_seq, lse_seq = out[qb : qb + nq].transpose(0, 1)[None], stats[qb : qb + nq].T[None]  # (1, h, nq, d_v) / (1, h, nq)
            if nk == 0:
                assert (o_seq == 0).all() and torch.isneginf(lse_seq).all(), f"prefolded={prefolded}: a zero-length KV sequence is O = 0 / LSE = -inf"
            else:
                qs, ks, vs = q[qb : qb + nq].transpose(0, 1)[None], k[kb : kb + nk].transpose(0, 1)[None], v[kb : kb + nk].transpose(0, 1)[None]
                ref_o, ref_lse = _half_oracle(qs, ks, vs, logit_scale=logit_scale, masked=_dense_mask(nq, nk, causal=True, bottom_right=True))
                e, s = _half_prefold_check(o_seq, lse_seq, ref_o, ref_lse, tag=f"{'fold' if prefolded else 'unfolded'} THD seq (nq={nq}, nk={nk})")
                if nq > nk:
                    assert (o_seq[:, :, : nq - nk] == 0).all(), f"prefolded={prefolded}: rows above the bottom-right diagonal store O = 0 exactly"
                err, scale = max(err, e), max(scale, s)
            qb += nq
            kb += nk
        return err, scale

    err_fold, scale = leg(True)
    err_unfolded, _ = leg(False)
    _fold_within_unfolded(err_fold, err_unfolded, scale, dtype, tag=f"d{d_qk}x{d_v} THD")


# The fold's arm marker.  Under the fold the adapter pins scale_log2 to exactly 1.0, so a body that IGNORED
# SCALE_PREFOLDED (kept the multiply by the scale) would pass every oracle above bit-for-bit -- the instruction mix is
# the only witness that the folded arm is the one traced.  The DSL spells the per-score shift `S * scale - m` as a
# packed f32 multiply plus a packed f32 subtract per pair (ptxas fuses them into FFMA2 later); under the fold the
# multiply is gone and the subtract stays (FADD2 after ptxas).  Pinned on the DSL's own PTX (CUTE_DSL_KEEP=ptx, which
# needs no disassembler for the target), one sm_107a trace-compile per fold state in a subprocess with the compiled-plan
# cache off, as the quantized SASS pins above do.  Measured on the causal fp16 Stats build of both twins: 522 -> 262 f32
# multiplies (4 body instantiations x 64 packed shift pairs plus their 4 scalar max scalings leave; the correction's alpha
# and 1/sum scalings stay), 264 subtracts either way.
_HALF_PREFOLD_PTX_PROBE = textwrap.dedent("""
    import glob, os, sys
    dump, d_qk, d_v, prefolded = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
    os.environ["CUTE_DSL_DUMP_DIR"] = dump          # read once, at the first cutlass import
    os.environ["CUTE_DSL_KEEP"] = "ptx"              # the DSL's own PTX -- no disassembler for the target needed
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"       # unconditional: an inherited value would pin the wrong target
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"  # a compiled-plan cache HIT skips the compile and dumps nothing
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, DTYPE_FP16
    # the causal fp16 Stats build at the production cga2 geometry; only the fold flag differs between the two probes
    params = TemplateParams(dtype_qkv=DTYPE_FP16, dtype_o=DTYPE_FP16, cta_mma=2, window_right=0, softmax_scale_prefolded=prefolded)
    mod = _load_sm100_kernel_module((d_qk, d_v), params, fp8=False, pertensor=False, rubin=True)
    assert int(mod.SCALE_PREFOLDED) == int(prefolded), mod.SCALE_PREFOLDED
    mod.compile(d_qk=d_qk, d_v=d_v, has_lse=True)
    ptxs = sorted(glob.glob(os.path.join(dump, "**", "*.ptx"), recursive=True), key=os.path.getmtime)
    if not ptxs:
        print("FAIL no ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    with open(ptxs[-1]) as f:
        ops = [ln.split()[0] for ln in f if ln.strip()]  # the opcode token of every statement (predicated lines start with @)
    def cnt(prefix):
        # scalar and packed f32 forms alike (mul.f32, mul.rn.f32, mul.f32x2, mul.rn.f32x2, ...)
        return sum(1 for op in ops if op.startswith(prefix + ".") and "f32" in op)
    print("PTX MUL", cnt("mul"))
    print("PTX SUB", cnt("sub"))
    print("PTX FMA", cnt("fma"))
    print("PTX STATEMENTS", len(ops))
    """)


@pytest.mark.parametrize("d_qk, d_v", _HALF_PREFOLD_TWINS)
def test_half_prefolded_scale_drops_the_shift_multiply(tmp_path, d_qk, d_v):
    """The folded build of each twin drops the per-score multiply by the scale: at least one 128-score body's worth (64
    pairs) of f32 multiplies leave the causal fp16 Stats build and the shift survives as f32 subtracts.  A twin whose body
    ignored SCALE_PREFOLDED shows identical counts -- the oracle tests cannot see that, since the pinned scale_log2 of 1.0
    makes the two bodies agree bit-for-bit."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    counts = {}
    for prefolded in (False, True):
        dump = tmp_path / f"half_prefold_d{d_qk}x{d_v}_{int(prefolded)}"
        dump.mkdir()
        argv = [sys.executable, "-c", _HALF_PREFOLD_PTX_PROBE, str(dump), str(d_qk), str(d_v), str(int(prefolded))]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=1500)
        assert (
            proc.returncode == 0
        ), f"sm_107a trace-compile of the half d{d_qk}x{d_v} kernel (prefolded={prefolded}) failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
        counts[prefolded] = {
            ln.split()[1]: int(ln.split()[2]) for ln in proc.stdout.splitlines() if ln.startswith("PTX ") and len(ln.split()) == 3 and ln.split()[2].isdigit()
        }
    scaled, folded = counts[False], counts[True]
    print(f"\nhalf d{d_qk}x{d_v} sm_107a PTX: scaled {scaled} / folded {folded}")
    assert folded["MUL"] <= scaled["MUL"] - 64, f"the fold left the per-score multiply in place ({scaled['MUL']} -> {folded['MUL']} f32 multiplies)"
    assert folded["SUB"] >= 64 and folded["FMA"] <= scaled["FMA"], f"the shift must survive as subtracts, not fused multiply-adds ({scaled} -> {folded})"


# ============================================================================ cc 10.7 d512 half: the pre-folded softmax scale
# sdpa(attn_scale_prefolded=True) on the ROLE-SPLIT d512 f16 / bf16 kernel (sm107/prefill_d512_f16.py, SCALE_PREFOLDED): the
# caller multiplied Q by attn_scale * log2(e); the kernel takes the RAW tile max and shifts with `reg_S - m` (an FADD2 for the
# scaled chain's FFMA2).  Numerically neutral by construction (the adapter pins scale_softmax_log2 = 1.0), so the cells below
# pin (a) that the fold build ROUTES onto the role-split body and traces (module constants, the call-time twin pinned off),
# (b) O / LSE against a float64 oracle of the pre-scaled half Q with ln 2 as the logit scale on every leg this body owns --
# dense / causal x Stats, THD with a zero-length sequence, sink + Stats, keyless and dead rows under bottom-right causal +
# SWA + padded Q / KV at scale 0.5 AND 1 (the scale-1 geometry the scaled chain NaNs: its sentinel * log2 e overflows,
# the fold never multiplies the sentinel) -- and (c) the sm_107a SASS: the fold build carries fewer FFMA-class and more
# FADD-class instructions than the same build without it, the only tripwire for the arm itself.  The HALF exponent
# (softmax_f16) stays declined on half inputs: P is already stored in the input half format, so the module raises on the
# record and its _FUSED_SHIFT_CVT is False.
_D512_HALF_FOLD_KERNEL = "sm107/prefill_d512_f16.py"
_D512_HALF_FOLD_DTYPES = [pytest.param("bfloat16", id="bf16"), pytest.param("float16", id="fp16")]


@pytest.fixture
def _d512_role_split(monkeypatch):
    """Pin the call-time d512 twin OFF so every half d512 plan built in the test lowers onto the role-split kernel."""
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(api_dsl, "D512_2X2", False)
    yield


def _d512_half_fold_skip():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cc10.7 d512 f16 / bf16 kernel runs on cc10.7 only")


def _prefold_q(q, factor):
    """Multiply a half Q (any strides) by ``factor`` in f32 and round back into the SAME layout -- the attn_scale_prefolded
    contract (the fold happens before the cast, so the kernel reads the quantized folded Q and the oracle sees that Q)."""
    import torch

    out = torch.empty_like(q)
    out.copy_((q.float() * factor).to(q.dtype))
    return out


def _d512_half_fold_assert_module(api, *, prefolded):
    mod = api._k_mod
    assert mod.__file__.endswith(_D512_HALF_FOLD_KERNEL), mod.__file__
    assert mod.SCALE_PREFOLDED == int(prefolded), (mod.SCALE_PREFOLDED, prefolded)
    assert mod._FUSED_SHIFT_CVT is False and mod.CFG.DTYPE_QKV in (2, 3)


def _d512_half_fold_run(q, k, v, *, causal, with_stats, prefolded, attn_scale):
    """One role-split d512 launch through the standalone adapter: the fold build (scale_softmax unset, the flag) or the
    scaled chain (scale_softmax=attn_scale).  Outputs are NaN-poisoned so an unwritten cell stays visible."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, hq, s = q.shape[0], q.shape[1], q.shape[2]
    out = torch.full((b, s, hq, v.shape[-1]), float("nan"), device="cuda", dtype=q.dtype).transpose(1, 2)
    lse = torch.full((b, hq, s), float("nan"), device="cuda", dtype=torch.float32)
    api = SdpaFwdDslSm100(
        q,
        k,
        v,
        out,
        lse if with_stats else None,
        is_causal=causal,
        scale_softmax=None if prefolded else attn_scale,
        softmax_scale_prefolded=prefolded,
    )
    assert api.check_support()
    api.compile()
    _d512_half_fold_assert_module(api, prefolded=prefolded)
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=out, lse_tensor=lse if with_stats else None, workspace=ws)
    torch.cuda.synchronize()
    return out, lse


def _d512_half_fold_oracle(q, k, v, *, logit_scale, causal):
    """float64 softmax(logit_scale * Q K^T) V and its natural-log LSE on the half inputs the kernel actually read (BHSD)."""
    import torch

    rep = q.shape[1] // k.shape[1]
    logits = (q.double() @ k.double().repeat_interleave(rep, 1).transpose(-1, -2)) * logit_scale
    if causal:
        s_q, s_kv = logits.shape[-2:]
        logits = logits.masked_fill(~torch.tril(torch.ones(s_q, s_kv, dtype=torch.bool, device=logits.device)), float("-inf"))
    return torch.softmax(logits, dim=-1) @ v.double().repeat_interleave(rep, 1), torch.logsumexp(logits, dim=-1)


@pytest.mark.L0
@pytest.mark.parametrize("with_stats", [True, False], ids=["stats", "nostats"])
@pytest.mark.parametrize("causal, b, hq, hkv, s", [(False, 1, 8, 2, 1024), (True, 1, 8, 4, 2048)], ids=["dense", "causal"])
@pytest.mark.parametrize("dtype_name", _D512_HALF_FOLD_DTYPES)
def test_d512_half_prefolded_scale_matches_the_oracle(_d512_role_split, dtype_name, causal, b, hq, hkv, s, with_stats):
    """cc10.7 e2e for softmax_scale_prefolded on the role-split d512 f16 / bf16 kernel: Q carries attn_scale * log2(e)
    (multiplied in f32 BEFORE the half cast), scale_softmax stays unset, the kernel takes the raw max and shifts without a
    multiply.  The fold build and the scaled chain each run against the float64 oracle of the half inputs they actually read
    (ln 2 resp. attn_scale as the logit scale): every O cell and LSE row written, O within 0.1 * max |ref| and within 2x the
    scaled chain's own error (the fold is numerically neutral), LSE within 5e-4 (natural log) -- the published Stats keep
    their domain under the fold.  The module constants pin the route (role-split file, SCALE_PREFOLDED, no fused arm)."""
    import math

    import torch

    _d512_half_fold_skip()
    dt = getattr(torch, dtype_name)
    d = 512
    attn_scale = d**-0.5
    torch.manual_seed(0)
    qf = torch.randn(b, s, hq, d, device="cuda") * 0.5
    k = (torch.randn(b, s, hkv, d, device="cuda") * 0.5).to(dt).transpose(1, 2)
    v = (torch.randn(b, s, hkv, d, device="cuda") * 0.5).to(dt).transpose(1, 2)
    q_scaled = qf.to(dt).transpose(1, 2)  # the scaled chain's Q
    q_folded = (qf * (attn_scale * math.log2(math.e))).to(dt).transpose(1, 2)  # the fold's Q: attn_scale * log2 e before the cast
    o_base, lse_base = _d512_half_fold_run(q_scaled, k, v, causal=causal, with_stats=with_stats, prefolded=False, attn_scale=attn_scale)
    o_fold, lse_fold = _d512_half_fold_run(q_folded, k, v, causal=causal, with_stats=with_stats, prefolded=True, attn_scale=attn_scale)
    ref_base, ref_lse_base = _d512_half_fold_oracle(q_scaled, k, v, logit_scale=attn_scale, causal=causal)
    ref_fold, ref_lse_fold = _d512_half_fold_oracle(q_folded, k, v, logit_scale=math.log(2.0), causal=causal)
    assert torch.isfinite(o_fold).all(), "non-finite / unwritten O cells on the fold build"
    amax = ref_fold.abs().max().item()
    err_fold = (o_fold.double() - ref_fold).abs().max().item()
    err_base = (o_base.double() - ref_base).abs().max().item()
    assert err_fold <= 0.1 * amax, f"fold: O max err {err_fold} vs oracle (max |ref| {amax})"
    # Neutral fold: its error against ITS oracle stays within 2x the scaled chain's against its own.  Floor = one ulp of the
    # output format at max |ref| (both outputs round to the same half format; the two chains see differently rounded Q).
    floor = torch.finfo(dt).eps * amax
    assert err_fold <= max(2.0 * err_base, floor), f"fold O err {err_fold} > 2 x the scaled chain's {err_base} (floor {floor})"
    if with_stats:
        assert torch.isfinite(lse_fold).all(), "unwritten LSE rows on the fold build"
        lse_err = (lse_fold.double() - ref_lse_fold).abs().max().item()
        assert lse_err <= 5e-4, f"fold: LSE max err {lse_err} vs oracle (natural log)"
        assert (lse_base.double() - ref_lse_base).abs().max().item() <= 5e-4, "scaled chain: LSE off its oracle"


@pytest.mark.L0
@pytest.mark.parametrize("scale", [0.5, 1.0], ids=["scale0.5", "scale1"])
@pytest.mark.parametrize("mask", sorted(_HANDOFF_MASKS))
@pytest.mark.parametrize("dtype_name", _D512_HALF_FOLD_DTYPES)
def test_d512_half_prefolded_scale_keyless_and_dead_rows(_d512_role_split, dtype_name, mask, scale):
    """The correction-storm geometry of the d512 correction hand-off test above under the fold: per-batch Q /
    KV padding with a QUERYLESS batch (dead rows: O exactly 0, LSE -inf) and, in the masked arm, bottom-right causal + left
    window 129 (the first 512 / 256 rows of batches 0 / 1 are KEYLESS; live rows whose window excludes KV tile 0 see a
    fully-masked FIRST tile, which the running-max step selects out of the row's state: alpha = 1, P = 0, the first live
    tile then starts the online softmax).  Q carries scale * log2 e; the fp64 reference composes the same padding /
    diagonal / window with ln 2.  Scale 1 is the geometry on which the SCALED chain used to NaN (its sentinel * log2 e
    overflows to -inf) while the fold published P = 1 on the dead tile and wiped it with alpha = 0 at the next one -- both
    chains take the same select now (the hand-off cells above hold the scaled chain at scale 1).  Two launches on identical
    inputs must be bitwise equal."""
    import math

    import torch

    _d512_half_fold_skip()
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    dt = getattr(torch, dtype_name)
    arm = _HANDOFF_MASKS[mask]
    q, k, v, q_lens, kv_lens = _handoff_problem(dt)
    q = _prefold_q(q, scale * math.log2(math.e))
    g = _HANDOFF_GEOMETRY
    o = torch.empty(g["b"], g["s_q"], g["h_q"], g["d"], device="cuda", dtype=dt).transpose(1, 2)
    lse = torch.empty(g["b"], g["h_q"], g["s_q"], device="cuda", dtype=torch.float32)
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse,
        is_causal=arm["causal_br"],
        causal_bottom_right=arm["causal_br"],
        window_size_left=arm["window_left"],
        scale_softmax=None,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
        softmax_scale_prefolded=True,
    )
    assert api.check_support()
    api.compile()
    _d512_half_fold_assert_module(api, prefolded=True)
    o0, lse0 = _handoff_execute(api, q, k, v, o, lse, q_lens, kv_lens)
    ref_o, ref_lse = _handoff_reference(
        q, k, v, scale=math.log(2.0), causal_br=arm["causal_br"], window_left=arm["window_left"], q_lens=q_lens, kv_lens=kv_lens
    )
    _handoff_check(o0, lse0, ref_o, ref_lse, q_lens, tag=f"fold {mask} scale {scale}")
    o1, lse1 = _handoff_execute(api, q, k, v, o, lse, q_lens, kv_lens)
    assert torch.equal(o1, o0) and torch.equal(lse1, lse0), "two launches on identical inputs must be bitwise equal (a race otherwise)"


@pytest.mark.L0
@pytest.mark.parametrize("dtype_name", _D512_HALF_FOLD_DTYPES)
def test_d512_half_masked_leading_tile_keeps_rows_with_later_keys_finite(_d512_role_split, dtype_name):
    """The correction hand-off geometry at scale 1 pinned to the ROLE-SPLIT d512 half kernel (the call-time 2x2 twin off):
    bottom-right causal + a 129-key window on per-batch padded Q / KV lengths -- the live rows past bottom-right diag 257
    (130 keys each, their window excluding KV tile 0) see a fully-masked FIRST tile and their keys in later tiles.  This
    kernel's scaled chain took the sentinel * log2 e = -inf as the running max and published O = NaN with LSE = log(1e-30)
    on 3064 rows; the running-max step now keeps such a tile out of the state (alpha = 1, P = 0): every retained O / LSE at
    the fp64 reference (dead and keyless rows exactly 0 / -inf), two launches on identical inputs bitwise equal."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    _d512_half_fold_skip()
    dt = getattr(torch, dtype_name)
    arm = _HANDOFF_MASKS["causal_br_swa129"]
    q, k, v, q_lens, kv_lens = _handoff_problem(dt)
    g = _HANDOFF_GEOMETRY
    o = torch.empty(g["b"], g["s_q"], g["h_q"], g["d"], device="cuda", dtype=dt).transpose(1, 2)
    lse = torch.empty(g["b"], g["h_q"], g["s_q"], device="cuda", dtype=torch.float32)
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse,
        is_causal=True,
        causal_bottom_right=True,
        window_size_left=arm["window_left"],
        scale_softmax=1.0,
        seq_kv_lens_present=True,
        seq_q_lens_present=True,
    )
    assert api.check_support()
    api.compile()
    _d512_half_fold_assert_module(api, prefolded=False)
    o0, lse0 = _handoff_execute(api, q, k, v, o, lse, q_lens, kv_lens)
    ref_o, ref_lse = _handoff_reference(q, k, v, scale=1.0, causal_br=True, window_left=arm["window_left"], q_lens=q_lens, kv_lens=kv_lens)
    _handoff_check(o0, lse0, ref_o, ref_lse, q_lens, tag=f"role-split {dtype_name} scale 1 leading tile")
    o1, lse1 = _handoff_execute(api, q, k, v, o, lse, q_lens, kv_lens)
    assert torch.equal(o1, o0) and torch.equal(lse1, lse0), "two launches on identical inputs must be bitwise equal (a race otherwise)"


@pytest.mark.L0
@pytest.mark.parametrize("dtype_name", _D512_HALF_FOLD_DTYPES)
def test_d512_half_prefolded_scale_thd(_d512_role_split, dtype_name):
    """Packed THD (three sequences, the middle one ZERO-LENGTH, per-sequence causal) through the graph API with
    sdpa(attn_scale_prefolded=True) and attn_scale unset, on the role-split kernel's persistent THD scheduler, Stats in the
    token-major ragged layout.  Per sequence, O and LSE against the float64 oracle of the folded half Q with ln 2; the
    sentinel outside the packed region (the empty sequence owns no row) comes back untouched."""
    import math

    import torch

    _d512_half_fold_skip()
    import test_sdpa_fwd_d512_2x2_sm100 as _t2x2
    import test_sdpa_fwd_dsl_sm100 as _dsl

    dt = getattr(torch, dtype_name)
    H, d = 4, 512
    seq_lens = [333, 0, 150]
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + n)
    T = cu[-1]
    attn_scale = d**-0.5
    torch.manual_seed(11)
    q_pk = (torch.randn(T, H, d, device="cuda") * 0.5 * (attn_scale * math.log2(math.e))).to(dt)
    k_pk = (torch.randn(T, H, d, device="cuda") * 0.5).to(dt)
    v_pk = (torch.randn(T, H, d, device="cuda") * 0.5).to(dt)
    served = []
    o_stor, stats_stor, _ = _dsl._run_dsl_thd_graph(
        q_pk,
        k_pk,
        v_pk,
        cu,
        cu,
        seq_lens,
        seq_lens,
        scale=None,
        dtype=dt,
        H_q=H,
        H_kv=H,
        d=d,
        mask="causal",
        check_stats=True,
        on_graph=lambda graph: served.append(_t2x2._served_template(graph)),
        sdpa_kwargs=dict(attn_scale_prefolded=True),
    )
    assert served == [_t2x2._ROLE_SPLIT_TEMPLATE], served
    o_pk = o_stor[: T * H * d].view(T, H, d)
    lse_pk = stats_stor[: T * H].view(T, H)
    for bi, n in enumerate(seq_lens):
        if n == 0:
            continue
        qs, ks, vs = (t[cu[bi] : cu[bi + 1]].permute(1, 0, 2).unsqueeze(0) for t in (q_pk, k_pk, v_pk))  # (1, H, n, d)
        ref_o, ref_lse = _d512_half_fold_oracle(qs, ks, vs, logit_scale=math.log(2.0), causal=True)
        o_seq = o_pk[cu[bi] : cu[bi + 1]].permute(1, 0, 2).double()
        assert torch.isfinite(o_seq).all(), f"sequence {bi}: non-finite / unwritten O cells"
        amax = ref_o.abs().max().item()
        err = (o_seq - ref_o[0]).abs().max().item()
        assert err <= 0.1 * amax, f"sequence {bi}: O max err {err} vs oracle (max |ref| {amax})"
        lse_err = (lse_pk[cu[bi] : cu[bi + 1]].t().double() - ref_lse[0]).abs().max().item()
        assert lse_err <= 5e-4, f"sequence {bi}: LSE max err {lse_err} vs oracle (natural log)"
    assert (o_stor[T * H * d :] == _dsl._THD_SENTINEL).all() and (stats_stor[T * H :] == _dsl._THD_SENTINEL).all()


@pytest.mark.L0
def test_d512_half_prefolded_scale_sink_and_stats(_d512_role_split):
    """Causal + sink + Stats through the graph API with sdpa(attn_scale_prefolded=True) on the role-split kernel: the sink
    logit is a natural-domain constant the epilogue folds from the log2-domain (ell, max) pair -- unchanged by the fold --
    so O and LSE match the reference that joins the sink as one extra column of the ln 2 scaled folded logits."""
    import math

    import torch

    _d512_half_fold_skip()
    import test_sdpa_fwd_d512_2x2_sm100 as _t2x2
    import test_sdpa_fwd_dsl_sm100 as _dsl

    dt = torch.bfloat16
    b, h, s, d = 1, 4, 384, 512
    attn_scale = 1.0 / math.sqrt(d)
    torch.manual_seed(6)
    q, k, v = (_dsl._bhsd(b, h, s, d, dt) for _ in range(3))
    q = _prefold_q(q, attn_scale * math.log2(math.e))
    sink = torch.randn(1, h, 1, 1, device="cuda", dtype=torch.float32)
    o, stats = _t2x2._run_graph(
        q,
        k,
        v,
        scale=None,
        dtype=dt,
        sdpa_kwargs=dict(use_causal_mask=True, attn_scale_prefolded=True),
        sink=sink,
        return_stats=True,
        expect_template=_t2x2._ROLE_SPLIT_TEMPLATE,
    )
    o_ref, lse_ref = _dsl._ref_sdpa_full(q, k, v, scale=math.log(2.0), is_causal=True, sinks=sink.flatten(), return_stats=True)
    torch.testing.assert_close(o, o_ref, **_t2x2._TOL)
    torch.testing.assert_close(stats.squeeze(-1), lse_ref, **_t2x2._TOL)


# The fold's only instruction-level signature: the per-score shift drops the scale multiply while the exponent count is
# untouched (no f16 arm exists here).  At the PTX level the scaled chain is `mul.f32x2` + `sub.f32x2` per pair (64 of each per
# traced shift site; the dense specialization traces one site) and the fold keeps the `sub.f32x2` only; ptxas may or may not
# contract the base's pair into FFMA2, so the SASS pin reads the MULTIPLY class (FFMA, FFMA2, FMUL, FMUL2) and lets the FADD
# class rise or stay.  Both probes trace-compile for sm_107a on any box (the shared probe's env); the SASS one needs an
# nvdisasm that decodes the arch (a 13.3 toolkit does not -> skip), the PTX one needs only the DSL.
_D512_HALF_FOLD_SASS_SPECS = {"dense": {}, "causal": {"window_right": 0}}
_D512_HALF_FOLD_PROBE_BODY = """
    params = TemplateParams(dtype_qkv=2, dtype_o=2, cta_mma=2, **params_kw)
    mod = _load_sm100_kernel_module((512, 512), params, fp8=False, pertensor=False, rubin=True)
    assert mod.__file__.endswith("sm107/prefill_d512_f16.py"), mod.__file__
    print("SCALE_PREFOLDED", int(mod.SCALE_PREFOLDED))
    print("FUSED_SHIFT_CVT", int(mod._FUSED_SHIFT_CVT))
    mod.compile(d_qk=512, d_v=512, has_lse=True, lse_kind="dense")
    """
# PTX histogram probe: the shared SASS probe's environment (dump dir, sm_107a, no compiled-plan cache) with the PTX kept
# instead of the cubin; prints one `PTX <key> <count>` line per opcode spelling (a count spec = substrings a line must all contain).
_D512_HALF_FOLD_PTX_COUNTS = {
    "MUL_F32X2": ("mul.f32x2",),
    "MUL_RN_F32X2": ("mul.rn.f32x2",),
    "SUB_F32X2": ("sub.f32x2",),
    "FMA_F32X2": ("fma.rn.f32x2",),
    "MUL_F32": ("mul.f32 ",),
    "EX2": ("ex2.approx",),
}
_D512_HALF_FOLD_PTX_PROBE = textwrap.dedent("""
    import glob, json, os, sys
    dump, params_json = sys.argv[1], sys.argv[2]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx"
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    params_kw = json.loads(params_json)
    %(body)s
    ptxs = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    if not ptxs:
        print("FAIL no ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    lines = open(ptxs[-1]).read().splitlines()
    for key, subs in json.loads(%(counts)r).items():
        print("PTX", key, sum(1 for ln in lines if all(sb in ln for sb in subs)))
    """) % {"body": textwrap.dedent(_D512_HALF_FOLD_PROBE_BODY).strip("\n"), "counts": __import__("json").dumps(_D512_HALF_FOLD_PTX_COUNTS)}


def _d512_half_fold_ptx_counts(tmp_path, params: dict, tag: str) -> dict:
    """Trace-compile the role-split d512 bf16 kernel for sm_107a in a fresh interpreter with the PTX kept and return the
    opcode histogram plus the module constants it printed (skips when the DSL has no sm_107a; a non-zero exit fails)."""
    import json

    from frost_test_utils import arch_known_to_the_dsl

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a")
    dump = tmp_path / f"ptx_{tag}"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _D512_HALF_FOLD_PTX_PROBE, str(dump), json.dumps(params)], capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_107a trace-compile of {tag} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = {}
    for ln in proc.stdout.splitlines():
        parts = ln.split()
        if len(parts) == 3 and parts[0] == "PTX" and parts[2].isdigit():
            out[parts[1]] = int(parts[2])
        elif len(parts) == 2 and parts[0] in ("SCALE_PREFOLDED", "FUSED_SHIFT_CVT"):
            out[parts[0]] = int(parts[1])
    print(f"\n{tag} sm_107a PTX: {out}")
    return out


@pytest.mark.parametrize("spec", sorted(_D512_HALF_FOLD_SASS_SPECS))
def test_d512_half_prefolded_scale_ptx_drops_the_per_score_multiply(tmp_path, spec):
    """The sm_107a PTX of the role-split d512 bf16 kernel with softmax_scale_prefolded carries 64 fewer `mul.f32x2` per traced
    softmax body (the dense specialization traces one body, the causal one two: the unmasked and the masked arm) and one fewer
    scalar `mul.f32` per body (the tile max) than the same specialization without it, the SAME `sub.f32x2` (the shift stays),
    the same `mul.rn.f32x2` (the correction / epilogue scalings) and the same exponent count; neither build fuses the shift
    into an fma.  The module constants pin the flag and the absence of a fused arm.  Runs wherever the DSL knows sm_107a
    (no disassembler needed)."""
    mask = _D512_HALF_FOLD_SASS_SPECS[spec]
    base = _d512_half_fold_ptx_counts(tmp_path, mask, f"d512_half_{spec}_scaled")
    fold = _d512_half_fold_ptx_counts(tmp_path, {**mask, "softmax_scale_prefolded": True}, f"d512_half_{spec}_fold")
    assert (base["SCALE_PREFOLDED"], fold["SCALE_PREFOLDED"]) == (0, 1)
    assert base["FUSED_SHIFT_CVT"] == fold["FUSED_SHIFT_CVT"] == 0, "no f16 exponent arm exists in the half d512 body"
    dropped = base["MUL_F32X2"] - fold["MUL_F32X2"]
    assert (
        dropped >= 64 and dropped % 64 == 0
    ), f"{spec}: mul.f32x2 {base['MUL_F32X2']} -> {fold['MUL_F32X2']}: not 64 per traced body (the fold build still multiplies per score)"
    bodies = dropped // 64
    assert (
        base["MUL_F32"] - fold["MUL_F32"] == bodies
    ), f"{spec}: scalar mul.f32 {base['MUL_F32']} -> {fold['MUL_F32']} vs {bodies} traced bodies: a raw-max site still scales (or a non-softmax multiply moved)"
    for key in ("SUB_F32X2", "MUL_RN_F32X2", "EX2"):
        assert fold[key] == base[key], f"{spec}: {key} moved ({base[key]} -> {fold[key]}); only the per-score multiply may change"
    assert base["FMA_F32X2"] == fold["FMA_F32X2"] == 0, f"{spec}: the shift must not lower to an fma ({base['FMA_F32X2']} / {fold['FMA_F32X2']})"


@pytest.mark.parametrize("spec", sorted(_D512_HALF_FOLD_SASS_SPECS))
def test_d512_half_prefolded_scale_sass_drops_the_per_score_multiply(tmp_path, spec):
    """The sm_107a SASS of the role-split d512 bf16 kernel with softmax_scale_prefolded carries at least 32 fewer MULTIPLY-class
    instructions (FFMA, FFMA2, FMUL, FMUL2 -- 64 packed pairs per traced shift site, whether or not ptxas contracted the scaled
    chain's mul + sub) than the same specialization without it, no fewer FADD-class ones (FADD, FADD2), the same MUFU.EX2
    count and no new spills.  Skips where no nvdisasm decodes the arch (the shared probe's rule)."""
    from frost_test_utils import SASS_OPCODE_COUNTS, SPILL_TOLERANCE, run_sass_probe, sass_probe_source

    counts = {
        **SASS_OPCODE_COUNTS,
        "FFMA": ("regex:", r" FFMA(\.\S+)? "),
        "FADD": ("regex:", r" FADD(\.\S+)? "),
        "FMUL": ("regex:", r" FMUL(\.\S+)? "),
        "FMUL2": ("regex:", r" FMUL2(\.\S+)? "),
    }
    probe = sass_probe_source(_D512_HALF_FOLD_PROBE_BODY, counts=counts)
    mask = _D512_HALF_FOLD_SASS_SPECS[spec]
    base = run_sass_probe(tmp_path, probe_src=probe, arch="sm_107a", params=mask, tag=f"d512_half_{spec}_scaled")
    fold = run_sass_probe(tmp_path, probe_src=probe, arch="sm_107a", params={**mask, "softmax_scale_prefolded": True}, tag=f"d512_half_{spec}_fold")
    assert (base.expect["SCALE_PREFOLDED"], fold.expect["SCALE_PREFOLDED"]) == (0, 1)
    assert base.expect["FUSED_SHIFT_CVT"] == fold.expect["FUSED_SHIFT_CVT"] == 0, "no f16 exponent arm exists in the half d512 body"
    mul_base, mul_fold = (sum(p.stats[k] for k in ("FFMA", "FFMA2", "FMUL", "FMUL2")) for p in (base, fold))
    add_base, add_fold = (p.stats["FADD"] + p.stats["FADD2"] for p in (base, fold))
    assert mul_fold <= mul_base - 32, f"{spec}: multiply-class {mul_base} -> {mul_fold}: the fold build still multiplies per score"
    assert add_fold >= add_base, f"{spec}: FADD-class {add_base} -> {add_fold}: the fold build lost its `reg_S - m` shift"
    assert fold.stats["MUFU_EX2"] == base.stats["MUFU_EX2"], f"{spec}: the exponent count moved ({base.stats['MUFU_EX2']} -> {fold.stats['MUFU_EX2']})"
    for key in ("STL", "LDL"):
        assert fold.stats[key] <= base.stats[key] + SPILL_TOLERANCE, f"{spec}: the fold build adds spills ({key} {base.stats[key]} -> {fold.stats[key]})"


# ============================================================================ d512 per-tensor FP8: the HALF softmax arm
# The role-split d512 per-tensor FP8 kernel (sm107/prefill_d512_fp8.py) honors softmax_precision=HALF: MUFU EX2.F16x2 on
# packed pairs and an f16x2 -> FP8 cast per 16-byte P vector through the SHARED helpers (tile_dsl/softmax_f16.py); the
# SMEM pack and the DSMEM ship are untouched.  The kernel has no ones-MMA row-sum, so the stats-less build normalizes O
# by the f16 pair-tree sum of the P words it stores while a build with Stats keeps the exact f32 denominator (honored,
# not faster).  The pre-folded softmax scale is NOT served here (descale_q * descale_k is folded into the softmax scale
# in-kernel), so the arms are HALF x {Stats, no Stats} x {dense (ld.red max), masked (software max)} x {E4M3, E5M2}, plus
# keyless rows under the masked arm.  Oracle: float64 on the DEQUANTIZED inputs the kernel actually saw.
_D512_FP8_HALF_FMT = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}
_D512_FP8_HALF_D = 512
_D512_FP8_KERNEL_FILE = os.path.join("sm107", "prefill_d512_fp8.py")


@pytest.mark.L0
@pytest.mark.parametrize("dtype_qkv, tag", [(0, "e4m3"), (1, "e5m2")], ids=["e4m3", "e5m2"])
def test_sm107_d512_fp8_half_softmax_module_contract(dtype_qkv, tag):
    """No GPU: the d512 per-tensor FP8 module takes softmax_f16 through the SHARED helpers (no pasted arm bodies), tags P
    with the input's FP8 pair format, threads has_lse into the sg0 softmax body at all four call sites (the Stats leg keeps
    the exact f32 sum), carries no fused shift+convert arm (`_FUSED_SHIFT_CVT` is False: the per-tensor descale fold keeps
    the FFMA2 shift), declines the pre-folded scale, and traces the f32 chain by default."""
    import inspect
    import re

    kw = dict(fp8=True, pertensor=True, dtype_qkv=dtype_qkv, dtype_o=_BF16_OUT, cta_mma=2)
    mod = _load(_D512, rubin=True, softmax_f16=True, **kw)
    assert mod.__file__.endswith(_D512_FP8_KERNEL_FILE), mod.__file__
    assert mod.SOFTMAX_F16 == 1 and mod._FP8_TAG_P == tag and mod._FUSED_SHIFT_CVT is False
    code = _source_lines(mod)
    assert "_softmax_f16.f16_exp_values(" in code and "_softmax_f16.f16_pairs_sum_pair(" in code, "the arm must come from tile_dsl.softmax_f16"
    for pasted in ("def f16_exp_values", "def f16_pairs_sum_pair", "def add_f16x2", "ex2.approx", "cvt.rn.f16x2.f32"):
        assert pasted not in code, f"{pasted!r}: a helper body was pasted into the kernel instead of imported"
    body = _code_lines(inspect.getsource(mod._sg0_softmax_kv_iter))
    assert "has_lse: cutlass.Constexpr[bool]" in body, "has_lse is not a compile-time parameter of the sg0 softmax body"
    assert "if cutlass.const_expr(SOFTMAX_F16):" in body and "if cutlass.const_expr(has_lse):" in body
    sites = re.findall(r"_sg0_softmax_kv_iter\(\s*(True|False),\s*lse_tensor is not None,\s*_kv,", _code_lines(inspect.getsource(mod._compute_warp_group)))
    assert sorted(sites) == ["False", "False", "True", "True"], f"has_lse must reach every sg0 softmax call site; found {sites}"
    with pytest.raises(ValueError, match="softmax_scale_prefolded"):
        _load(_D512, rubin=True, softmax_scale_prefolded=True, **kw)
    assert _load(_D512, rubin=True, **kw).SOFTMAX_F16 == 0


def _fp8_pertensor_problem(b, hq, hkv, s_q, s_kv, d, fp8_dtype, *, std=1.0, seed=0):
    """Random Q / K / V quantized PER TENSOR the way the engine consumes them (amax / fmax descale; BSHD-physical storage
    under a BHSD-logical view), their (1,) fp32 descale tensors, and the float64 DEQUANTIZED copies the oracle must see.
    ``std`` = 1 gives unit-variance logits at d = 512 (attn_scale = d**-0.5): a softmax peaky enough that a P word in the
    wrong byte order or a missed P pair moves O well past the oracle bound (a near-uniform softmax hides a 4-way permutation)."""
    import torch

    dev = "cuda"
    torch.manual_seed(seed)
    fmax = torch.finfo(fp8_dtype).max

    def quant(x_bshd):
        dsc = (x_bshd.abs().amax().clamp_min(1e-8) / fmax).item()
        data = (x_bshd / dsc).clamp(-fmax, fmax).to(fp8_dtype)
        return data.transpose(1, 2), torch.full((1,), dsc, device=dev, dtype=torch.float32), data.double().transpose(1, 2) * dsc

    q8, dq, qd = quant(torch.randn(b, s_q, hq, d, device=dev) * std)
    k8, dk, kd = quant(torch.randn(b, s_kv, hkv, d, device=dev) * std)
    v8, dv, vd = quant(torch.randn(b, s_kv, hkv, d, device=dev) * std)
    return (q8, dq, qd), (k8, dk, kd), (v8, dv, vd)


def _fp8_oracle(qd, kd, vd, *, attn_scale, causal=False, causal_br=False, kv_lens=None, window_left=None):
    """float64 softmax(attn_scale * QK^T) V and the natural-log LSE on the dequantized operands, composing the kernel's
    mask: per-batch KV padding, top-left causal, or bottom-right causal anchored at (s_q, kv_len[b]), plus an optional left
    window of ``window_left`` past keys riding the diagonal.  A row with no live key comes out as O = 0 / LSE = -inf."""
    import torch

    b, hq, s_q, _ = qd.shape
    hkv, s_kv = kd.shape[1], kd.shape[2]
    rep = hq // hkv
    logits = (qd @ kd.repeat_interleave(rep, 1).transpose(-1, -2)) * attn_scale
    i = torch.arange(s_q, device=qd.device).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=qd.device).view(1, 1, 1, s_kv)
    kl = (kv_lens.to(torch.int64) if kv_lens is not None else torch.full((b,), s_kv, dtype=torch.int64, device=qd.device)).view(b, 1, 1, 1)
    masked = j >= kl
    if causal:
        diag = i + (kl - s_q) if causal_br else i
        masked = masked | (j > diag)
        if window_left is not None:
            masked = masked | (j < diag - window_left)
    logits = logits.masked_fill(masked, float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    o = torch.softmax(logits, dim=-1).nan_to_num(0.0) @ vd.repeat_interleave(rep, 1)
    return o, lse


def _run_d512_fp8(q8, k8, v8, descales, *, with_stats, precision, attn_scale, dtype_o, causal=False, causal_br=False, kv_lens=None, window_left=None):
    """Build, compile and launch the d512 per-tensor FP8 kernel TWICE (NaN-poisoned outputs; a two-launch delta is a
    first-launch race); assert the build is the d512 per-tensor module with the requested arm; return (api, O, LSE)."""
    import torch
    from cudnn import data_type as cudnn_dtype
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, hq, s_q = q8.shape[0], q8.shape[1], q8.shape[2]
    dev = q8.device
    out = torch.full((b, s_q, hq, v8.shape[3]), float("nan"), device=dev, dtype=dtype_o).transpose(1, 2)
    lse = torch.full((b, hq, s_q), float("nan"), device=dev, dtype=torch.float32)
    api = SdpaFwdDslSm100(
        q8,
        k8,
        v8,
        out,
        lse if with_stats else None,
        scale_softmax=attn_scale,
        is_causal=causal,
        causal_bottom_right=causal_br,
        window_size_left=window_left,
        seq_kv_lens_present=kv_lens is not None,
        pertensor_fp8=True,
        dtype_o=dtype_o,
        cga=2,
        softmax_precision=precision,
    )
    assert api.check_support()
    api.compile()
    assert api._k_mod.__file__.endswith(_D512_FP8_KERNEL_FILE), api._k_mod.__file__
    assert api._k_mod.SOFTMAX_F16 == int(precision == cudnn_dtype.HALF), "the build does not carry the requested softmax arm"
    assert api._k_mod._FUSED_SHIFT_CVT is False, "per-tensor FP8 has no pre-folded scale, hence no fused shift+convert arm"
    dq, dk, dv = descales
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    kw = dict(lse_tensor=lse if with_stats else None, descale_q=dq, descale_k=dk, descale_v=dv, workspace=ws)
    if kv_lens is not None:
        kw["seq_kv_lens"] = kv_lens
    api.execute(q8, k8, v8, out, **kw)
    torch.cuda.synchronize()
    first_o, first_lse = out.clone(), lse.clone()
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    api.execute(q8, k8, v8, out, **kw)
    torch.cuda.synchronize()
    assert torch.equal(out, first_o) and (not with_stats or torch.equal(lse, first_lse)), "two-launch delta: a first-launch race"
    return api, out, lse


_D512_FP8_HALF_CASES = [
    # (FP8 format, causal, b, hq, hkv, s): both formats on the dense arm and on the masked arm -- the f16x2 -> FP8 cast is
    # per format, the two arms share the exponent / pack / row-sum tail.
    pytest.param("e4m3", False, 1, 8, 2, 1024, id="e4m3-dense-s1024"),
    pytest.param("e4m3", True, 2, 8, 2, 2048, id="e4m3-causal-s2048"),
    pytest.param("e5m2", False, 1, 8, 2, 1024, id="e5m2-dense-s1024"),
    pytest.param("e5m2", True, 1, 8, 2, 1024, id="e5m2-causal-s1024"),
]


@pytest.mark.L0
@pytest.mark.parametrize("fmt, causal, b, hq, hkv, s", _D512_FP8_HALF_CASES)
def test_d512_fp8_half_softmax_matches_the_oracle(fmt, causal, b, hq, hkv, s):
    """cc10.7 e2e for softmax_precision=HALF on the d512 per-tensor FP8 kernel, both Stats legs of each case: every O cell
    and LSE row written; O within 0.1 * max|ref| of the float64 oracle on the DEQUANTIZED inputs (the arm casts the f16x2 P
    straight to FP8 pairs in fp32_to_fp8_pack's byte order -- a swapped half-word or a wrong exponent bias is a wrong O
    here); the Stats leg's LSE within 1e-4 (natural log) of the oracle (it keeps the exact f32 denominator); and the two
    legs' O within 0.5 % of max|ref| of each other -- the stats-less leg's f16 pair-tree denominator against the exact one
    (a dropped P word reads >= 1.6 %), in an fp16 O so the output rounding (2^-11) stays out of that margin.  Red->green on
    the board: a reversed FP8 word order fails the oracle bound by 10x, a dropped P word fails it (and the gap bound), the
    f16 sum leaking into the Stats leg fails the LSE bound (it reads 3.6e-4 rms / 9e-4 max; the exact chain ~1e-5)."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the d512 per-tensor FP8 kernel serves cc10.7 only")
    from cudnn import data_type as cudnn_dtype

    d = _D512_FP8_HALF_D
    attn_scale = d**-0.5
    (q8, dq, qd), (k8, dk, kd), (v8, dv, vd) = _fp8_pertensor_problem(b, hq, hkv, s, s, d, getattr(torch, _D512_FP8_HALF_FMT[fmt]))
    ref, ref_lse = _fp8_oracle(qd, kd, vd, attn_scale=attn_scale, causal=causal)
    scale = ref.abs().max().item()
    outs = {}
    for with_stats in (True, False):
        _, out, lse = _run_d512_fp8(
            q8, k8, v8, (dq, dk, dv), with_stats=with_stats, precision=cudnn_dtype.HALF, attn_scale=attn_scale, dtype_o=torch.float16, causal=causal
        )
        assert torch.isfinite(out).all(), f"stats={with_stats}: non-finite / unwritten O cells"
        err = (out.double() - ref).abs().max().item()
        assert err <= 0.1 * scale, f"stats={with_stats}: max err {err} vs oracle (scale {scale})"
        print(f"\n{fmt} causal={causal} s={s} stats={with_stats}: O max err {err:.3e} (scale {scale:.3e})")
        if with_stats:
            assert torch.isfinite(lse).all(), "unwritten LSE rows"
            lse_err = (lse.double() - ref_lse).abs().max().item()
            print(f"LSE max err {lse_err:.3e} (natural log)")
            assert lse_err <= 1e-4, f"LSE max err {lse_err} vs oracle (natural log): the Stats leg must keep the exact f32 denominator"
        outs[with_stats] = out
    gap = (outs[True].double() - outs[False].double()).abs().max().item()
    print(f"Stats / no-Stats O gap {gap:.3e} (scale {scale:.3e})")
    assert gap <= 5e-3 * scale, f"Stats / no-Stats O gap {gap} (scale {scale}): the f16 pair-tree denominator disagrees with the exact f32 sum"


@pytest.mark.L0
def test_d512_fp8_half_softmax_keyless_rows():
    """Keyless rows under the HALF arm: bottom-right causal with per-batch KV lengths (512, 256, 0) at s_q = 1024, s_kv = 512,
    so the first s_q - kv_len[b] rows of every head (all of batch 2) have no live key.  A fully-masked tile leaves the raw
    max at the finite sentinel and the running-max step selects it out of the row's state (alpha = 1, P = 0 on the f16x2
    arm exactly as on the f32 chain), and the epilogue's empty-row select publishes O = 0 / LSE = -inf there while the live
    rows (some with a single live key) match the oracle on both Stats legs, in the production bf16 O."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the d512 per-tensor FP8 kernel serves cc10.7 only")
    from cudnn import data_type as cudnn_dtype

    b, hq, hkv, s_q, s_kv, d = 3, 8, 2, 1024, 512, _D512_FP8_HALF_D
    attn_scale = d**-0.5
    kv_lens = torch.tensor([512, 256, 0], dtype=torch.int32, device="cuda")
    (q8, dq, qd), (k8, dk, kd), (v8, dv, vd) = _fp8_pertensor_problem(b, hq, hkv, s_q, s_kv, d, torch.float8_e4m3fn, seed=1)
    ref, ref_lse = _fp8_oracle(qd, kd, vd, attn_scale=attn_scale, causal=True, causal_br=True, kv_lens=kv_lens)
    keyless = torch.isneginf(ref_lse)  # [b, hq, s_q]
    assert int(keyless.sum()) == hq * ((s_q - 512) + (s_q - 256) + s_q), "geometry: keyless rows per batch = s_q - kv_len[b]"
    live = ~keyless
    scale = ref.abs().max().item()
    for with_stats in (True, False):
        _, out, lse = _run_d512_fp8(
            q8,
            k8,
            v8,
            (dq, dk, dv),
            with_stats=with_stats,
            precision=cudnn_dtype.HALF,
            attn_scale=attn_scale,
            dtype_o=torch.bfloat16,
            causal=True,
            causal_br=True,
            kv_lens=kv_lens,
        )
        assert torch.isfinite(out).all(), f"stats={with_stats}: non-finite / unwritten O cells"
        assert (out[keyless] == 0).all(), f"stats={with_stats}: keyless rows must publish O = 0 exactly (the empty-row select)"
        err = (out.double() - ref).abs()[live].max().item()
        assert err <= 0.1 * scale, f"stats={with_stats}: live rows max err {err} vs oracle (scale {scale})"
        print(f"\nkeyless stats={with_stats}: live rows O max err {err:.3e} (scale {scale:.3e})")
        if with_stats:
            assert torch.isneginf(lse[keyless]).all(), "keyless rows must publish LSE = -inf"
            assert torch.isfinite(lse[live]).all(), "unwritten / non-finite LSE on live rows"
            lse_err = (lse.double() - ref_lse).abs()[live].max().item()
            print(f"live rows LSE max err {lse_err:.3e} (natural log)")
            assert lse_err <= 1e-4, f"live rows LSE max err {lse_err} vs oracle (natural log): the Stats leg must keep the exact f32 denominator"


@pytest.mark.L0
@pytest.mark.parametrize("precision_name", ["FLOAT", "HALF"])
def test_d512_fp8_masked_leading_tile_keeps_rows_with_later_keys_finite(precision_name):
    """A row whose FIRST KV tile is fully masked while a LATER tile holds its keys, on the d512 per-tensor FP8 role-split
    kernel: top-left causal with a 34-key band at S = 256 (rows 161..255: no key in tile 0, 34 keys in tile 1) under UNIT
    descales and attn_scale 1 -- the folded scale_log2 = log2 e > 1 overflowed the scaled mask sentinel to -inf, the running
    max of that tile, and -inf - (-inf) = NaN went into P and the row-sum (with the quantizer's descales the tile published
    P = 1 instead, wiped by alpha = 0 at the next live tile).  Values drawn inside the fp8 range; both softmax arms, both
    Stats legs through the module's two-launch runner: O and LSE finite and at the float64 oracle."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the d512 per-tensor FP8 kernel serves cc10.7 only")
    from cudnn import data_type as cudnn_dtype

    b, hq, hkv, s, d = 1, 8, 2, 256, _D512_FP8_HALF_D
    gen = torch.Generator(device="cuda").manual_seed(0)

    def draw(h, std):
        x8 = (torch.randn(b, s, h, d, device="cuda", generator=gen) * std).to(torch.float8_e4m3fn).transpose(1, 2)
        return x8, x8.double()

    (q8, qd), (k8, kd), (v8, vd) = draw(hq, 1.0), draw(hkv, 1.0), draw(hkv, 1.0)
    unit = torch.ones(1, device="cuda", dtype=torch.float32)
    ref, ref_lse = _fp8_oracle(qd, kd, vd, attn_scale=1.0, causal=True, window_left=33)
    assert torch.isfinite(ref_lse).all(), "geometry: every row keeps 34 keys"
    scale = ref.abs().max().item()
    for with_stats in (True, False):
        _, out, lse = _run_d512_fp8(
            q8,
            k8,
            v8,
            (unit, unit, unit),
            with_stats=with_stats,
            precision=getattr(cudnn_dtype, precision_name),
            attn_scale=1.0,
            dtype_o=torch.bfloat16,
            causal=True,
            window_left=33,
        )
        assert torch.isfinite(out).all(), f"stats={with_stats}: {int((~torch.isfinite(out)).sum())} non-finite O cells"
        err = (out.double() - ref).abs().max().item()
        assert err <= 0.1 * scale, f"stats={with_stats}: O max err {err} vs oracle (scale {scale})"
        if with_stats:
            assert torch.isfinite(lse).all(), f"{int((~torch.isfinite(lse)).sum())} non-finite LSE rows"
            lse_err = (lse.double() - ref_lse).abs().max().item()
            assert lse_err <= 1e-4, f"LSE max err {lse_err} vs oracle (natural log)"


@pytest.mark.L0
def test_d512_fp8_half_softmax_tracks_the_float_chain():
    """The HALF arm against the FLOAT build of the same d512 problem (dense E4M3, stats-less: the f16 pair-tree denominator
    leg; production bf16 O): both within the oracle bound, and HALF within 10 % of max|ref| of FLOAT.  The two chains feed
    the SAME fp8 cast with P values ~0.5 % apart (the f16 rounding of the exp argument, up to 2^-7 at |arg| in [8, 16), plus
    MUFU EX2.F16x2's 2^-9.9), so a few percent of the P cells land on a different e4m3 step (6-12 % of P each): measured
    5.1 % of max|ref| at the worst of the 4M cells of this unit-variance-logit problem.  A byte-order or bias defect in the
    arm reads as an O(1) divergence from the f32 chain, far past 10 %."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the d512 per-tensor FP8 kernel serves cc10.7 only")
    from cudnn import data_type as cudnn_dtype

    b, hq, hkv, s, d = 1, 8, 2, 1024, _D512_FP8_HALF_D
    attn_scale = d**-0.5
    (q8, dq, qd), (k8, dk, kd), (v8, dv, vd) = _fp8_pertensor_problem(b, hq, hkv, s, s, d, torch.float8_e4m3fn, seed=2)
    ref, _ = _fp8_oracle(qd, kd, vd, attn_scale=attn_scale)
    scale = ref.abs().max().item()
    outs = {}
    for precision in (cudnn_dtype.FLOAT, cudnn_dtype.HALF):
        _, out, _ = _run_d512_fp8(q8, k8, v8, (dq, dk, dv), with_stats=False, precision=precision, attn_scale=attn_scale, dtype_o=torch.bfloat16)
        err = (out.double() - ref).abs().max().item()
        assert err <= 0.1 * scale, f"{precision}: max err {err} vs oracle (scale {scale})"
        outs[precision] = out.double()
    xerr = (outs[cudnn_dtype.HALF] - outs[cudnn_dtype.FLOAT]).abs().max().item()
    print(f"\nHALF-vs-FLOAT O divergence {xerr:.3e} (scale {scale:.3e})")
    assert xerr <= 0.1 * scale, f"HALF-vs-FLOAT softmax divergence {xerr} (scale {scale})"


# ---------------------------------------------------------------------------- SASS pins: the d512 per-tensor FP8 HALF softmax arm
# Trace-compiled for sm_107a on any box (the probe recipe of the pins above).  MEASURED on a cc 10.7 board (sm_107a, dense E4M3 in /
# BF16 out, production cga2, B=1 H=128 S=8192 geometry of compile_prepared), stats-less build: 64 MUFU.EX2.F16x2 (one per packed
# pair of the 128-score row) and ONE f32 MUFU.EX2 (alpha); 64 F2FP f32x2 -> f16x2 packs and 64 F2FP f16x2 -> E4M3 casts (the fp8
# words come from the f16 pairs, never from f32: 0 F2FP.*.E4M3.F32); 0 FADD2 (the f32 register row-sum is gone -- the HADD2 pair
# tree over the stored P words replaces it); 65 FFMA2 (the per-score shift STAYS: per-tensor FP8 serves no pre-folded scale);
# REG 224 (the f32 chain: 224 / 226), STL = LDL = 0.  Stats build: the same f16 P path plus the exact f32 denominator (129 f32
# MUFU.EX2 and 63 FADD2 come back); the 128 scores stay live across that second exponent, which reads REG 255 with 8 STL / 8 LDL
# -- the measured, documented cost of an arm the contract honors but does not promise to be faster (its ceiling is pinned so a
# regression past it is seen; chunking the exponent did not help: 8 per 64 scores, 26 per 16).
_SM107_D512_FP8_HALF_SASS_PROBE = textwrap.dedent("""
    import glob, os, re, subprocess, sys
    dump, has_lse, cands = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "cubin"
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module, supported_cgas_for
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    (cta_mma,) = supported_cgas_for((512, 512), fp8=True, device_cc=(10, 7), pertensor=True)
    params = TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=cta_mma, softmax_f16=True)
    mod = _load_sm100_kernel_module((512, 512), params, fp8=True, pertensor=True, rubin=True)
    assert mod.SOFTMAX_F16 == 1 and mod.__file__.endswith(os.path.join("sm107", "prefill_d512_fp8.py")), mod.__file__
    mod.compile_prepared(d_qk=512, d_v=512, has_lse=bool(has_lse))
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
    def cnt(pred):
        return sum(1 for ln in sass if pred(ln))
    print("SASS MUFU_EX2_F16X2", cnt(lambda l: "MUFU.EX2.F16x2" in l))
    print("SASS MUFU_EX2_F32", cnt(lambda l: "MUFU.EX2" in l and "F16" not in l))
    print("SASS F2FP_F16_PACK", cnt(lambda l: "F2FP.F16.F32" in l))
    print("SASS F2FP_FP8_FROM_F16", cnt(lambda l: "F2FP.SATFINITE.E4M3.F16" in l))
    print("SASS F2FP_FP8_FROM_F32", cnt(lambda l: "F2FP.SATFINITE.E4M3.F32" in l))
    print("SASS HADD2_TREE", cnt(lambda l: "HADD2" in l and "HADD2.F32" not in l))
    print("SASS FADD2", cnt(lambda l: " FADD2" in l))
    print("SASS FFMA2", cnt(lambda l: " FFMA2" in l))
    print("SASS STL", cnt(lambda l: "STL" in l))
    print("SASS LDL", cnt(lambda l: "LDL" in l))
    cuobj = os.path.join(os.path.dirname(nvd), "cuobjdump")
    reg = -1
    if os.path.isfile(cuobj):
        ru = subprocess.run([cuobj, "--dump-resource-usage", cubins[-1]], capture_output=True, text=True).stdout
        regs = [int(m) for m in re.findall(r"REG:(\\d+)", ru)]
        reg = max(regs) if regs else -1
    print("SASS REG", reg)
    print("SASS LINES", len(sass))
    """)

_SM107_D512_FP8_HALF_SASS_ROWS = [
    # (has_lse, f32 MUFU.EX2 floor, FADD2 count, REG measured, spill ceiling)
    pytest.param(False, 0, 0, 224, 0, id="half-nostats"),
    pytest.param(True, 128, 63, 255, 8, id="half-stats"),
]


@pytest.mark.parametrize("has_lse, f32_ex2_min, fadd2, reg_measured, spill_max", _SM107_D512_FP8_HALF_SASS_ROWS)
def test_sm107_d512_fp8_half_softmax_sass_pins(tmp_path, has_lse, f32_ex2_min, fadd2, reg_measured, spill_max):
    """The HALF arm of the d512 per-tensor FP8 kernel in SASS: exactly 64 MUFU.EX2.F16x2 per row (one per packed pair), the
    FP8 P words cast from the f16 pairs (64 F2FP f32->f16x2 packs, 64 F2FP f16x2->E4M3, none from f32), the HADD2 pair tree
    present and the f32 FADD2 row-sum gone on the stats-less build (back, with the second f32 exponent, on the Stats build),
    the per-score FFMA2 shift kept (no pre-folded arm on per-tensor FP8), no spill, REG within the measured ceiling."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"sm107a_d512_fp8_half_lse{int(has_lse)}"
    dump.mkdir()
    argv = [sys.executable, "-c", _SM107_D512_FP8_HALF_SASS_PROBE, str(dump), str(int(has_lse)), *cands]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_107a trace-compile of the d512 fp8 HALF has_lse={has_lse} build failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].lstrip("-").isdigit()}
    print(f"\nd512 fp8 HALF has_lse={has_lse} sm_107a SASS: {stats}")
    assert stats["MUFU_EX2_F16X2"] == 64, f"{stats['MUFU_EX2_F16X2']} MUFU.EX2.F16x2: the exponent is not one MUFU per packed pair of the 128-score row"
    assert stats["MUFU_EX2_F32"] >= f32_ex2_min, f"{stats['MUFU_EX2_F32']} f32 MUFU.EX2: the Stats build must keep the exact f32 exponent for its denominator"
    if not has_lse:
        assert (
            stats["MUFU_EX2_F32"] <= 2
        ), f"{stats['MUFU_EX2_F32']} f32 MUFU.EX2 on the stats-less build (alpha only expected): a second exponent is being traced"
    assert stats["F2FP_F16_PACK"] == 64 and stats["F2FP_FP8_FROM_F16"] == 64, f"P must be packed f32x2 -> f16x2 (64) and cast f16x2 -> E4M3 (64): {stats}"
    assert stats["F2FP_FP8_FROM_F32"] == 0, f"{stats['F2FP_FP8_FROM_F32']} f32 -> E4M3 casts: an FP8 word is still cast from f32"
    assert (
        stats["FADD2"] == fadd2
    ), f"{stats['FADD2']} FADD2 (expected {fadd2}): the f32 register row-sum is {'back on the stats-less build' if not has_lse else 'missing on the Stats build'}"
    if not has_lse:
        assert stats["HADD2_TREE"] > 0, "no HADD2: the stats-less denominator is not the f16 pair tree over the stored P words"
    assert stats["FFMA2"] >= 64, f"{stats['FFMA2']} FFMA2: the per-score shift must stay a multiply-add on per-tensor FP8 (the descale fold rides in the scale)"
    assert (
        stats["STL"] <= spill_max and stats["LDL"] <= spill_max
    ), f"the d512 fp8 HALF has_lse={has_lse} build spills ({stats['STL']} STL / {stats['LDL']} LDL, ceiling {spill_max})"
    if stats["REG"] >= 0:
        assert stats["REG"] <= reg_measured + _REG_SLACK, f"REG {stats['REG']} > {reg_measured} + {_REG_SLACK}"


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("stats", ["none", "ln", "log2"])
def test_packed_sink_combine_counts_virtual_key_once(dtype_name, stats):
    """An empty real-key row still has its sink; dead partials never read poison."""
    import math
    import torch
    from cudnn.frost.compiled_cache import positional_entry
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as comb
    from test_sdpa_split_combine_sm100 import _partials, _output, _strides

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Sink-aware packed split is qualified on SM107")
    dtype = getattr(torch, dtype_name)
    owner = comb.compile_ptr(
        dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=stats != "none", stats_log2=stats == "log2", packed=True, has_sink=True
    )
    fn = positional_entry(owner)
    b, h, sq, d = 1, 3, 7, 160
    ostride, lstride = _strides(b, h, sq, d, "int64_singleton")
    for splits in (2, 32, 33):
        op, lp, _, _ = _partials(b, h, sq, d, splits)
        o, ostorage, used = _output((b, sq, h, d), ostride, dtype)
        lse, lstorage, lused = _output((b, h, sq), lstride, torch.float32) if stats != "none" else (None, None, None)
        total = torch.tensor([sq], device="cuda", dtype=torch.int32)
        sinks = torch.tensor([-torch.inf, 3, 1000], device="cuda")

        def run():
            fn(
                op.data_ptr(),
                lp.data_ptr(),
                o.data_ptr(),
                lse.data_ptr() if lse is not None else None,
                (b, h, sq, d),
                splits,
                ostride,
                lstride,
                total.data_ptr(),
                torch.cuda.current_stream().cuda_stream,
                sinks.data_ptr(),
            )

        run()
        captured = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(captured):
                run()
            for live, changed_sink in ((sq, 3), (3, -120), (0, 10)):
                total.fill_(live)
                sinks[1] = changed_sink
                # Padded partial capacity is intentionally unreadable data.
                op[:, live:].fill_(torch.nan)
                lp[:, :, live:].fill_(torch.nan)
                ostorage.fill_(-31)
                if lstorage is not None:
                    lstorage.fill_(-31)
                captured.replay()
                scores = lp[:, :, :live].cpu().double()
                virtual = sinks.cpu().double()[None, :, None].expand(1, h, live)
                weights = torch.cat((scores, virtual), 0).softmax(0).nan_to_num()
                values = op[:, :live].cpu().double().nan_to_num()
                ref_o = (weights[:-1].permute(0, 2, 1)[..., None] * values).sum(0)
                torch.testing.assert_close(o[0, :live].cpu().double(), ref_o, atol=0.004, rtol=0.004)
                assert torch.all(o[:, live:] == -31)
                assert torch.all(ostorage.cpu()[~used] == -31)
                if lse is not None:
                    ref_s = torch.cat((scores, virtual), 0).logsumexp(0) * (math.log2(math.e) if stats == "log2" else 1)
                    torch.testing.assert_close(lse[0, :, :live].cpu().double(), ref_s, atol=2e-5, rtol=2e-5)
                    assert torch.all(lse[:, :, live:] == -31)
                    assert torch.all(lstorage.cpu()[~lused] == -31)
        finally:
            captured.reset()


def test_d256_pack_gqa_divisibility_rule_is_exempted_on_the_rubin_decode_tile_only(monkeypatch):
    """The prefill tiles' PackGQA divisibility rule (``h_q / h_kv`` must divide -- or, under partial
    PackGQA, share a factor with -- the kernel ``tile_m``) is exempted for a graph the d256 DECODE
    tile packs whole, and that exemption is the cc 10.7 route's only: ``engines.mismatch`` gates its
    twin on the Rubin row (``_decode_packs_whole_group``), so the adapter does the same.  An odd group
    (15/5 = 3 heads per KV head at ``S_q = 1``: 3 packed rows, inside both lines' 16-row tile) is the
    typed ``ValueError`` it always was on the SM100 line and the served decode-tile form on cc 10.7."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    def api():
        return SdpaFwdDslSm100(
            _desc((1, 15, 1, 256), torch.bfloat16, "q"),
            _desc((1, 5, 256, 256), torch.bfloat16, "k"),
            _desc((1, 5, 256, 256), torch.bfloat16, "v"),
            _desc((1, 15, 1, 256), torch.bfloat16, "o"),
            None,
            pack_gqa=True,
        )

    _fake_cc(monkeypatch, (10, 0))
    with pytest.raises(ValueError, match="the kernel tile_m"):
        api().check_support()
    _fake_cc(monkeypatch, (10, 7))
    a = api()
    assert a.check_support()
    assert a._decode_q_tile() == 16 and a._decode_q_tile_for(1, 15, 5) == 16
