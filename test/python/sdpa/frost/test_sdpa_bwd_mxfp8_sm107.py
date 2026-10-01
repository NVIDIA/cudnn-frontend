# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Rubin (SM 10.7) MXFP8 d=256 SDPA backward: the KERNEL BODY ``sdpa/bwd/kernels/sm107/bprop_d256_mxfp8.py`` (static +
sm_107a SASS pins) and the ENGINE ROW ``sdpa_bwd_sm107_mxfp8`` (registration, the row's ``mismatch()`` on real graphs, the Rubin
accept matrix against ``sdpa.mxfp8_ref.compute_ref_backward``, ``quantize_ds=False`` for the bf16-dS chain and ``True`` for the
block-scaled one).

Two dS policies run through ONE adapter constant (``api_dsl_sm107.MXFP8_DS_SF_POLICY``, the fp8 suite's ``FP8_DS_DTYPE`` pattern; the
``ds_policy`` fixture monkeypatches it): **P-b** (``DS_SF_P_B`` = ``config_sm107.DS_SF_POLICY_DEFAULT``, what ships -- the kernel
writes two 1x32-scaled e4m3 dS payloads + their E8M0 atoms and the stage-3 GEMMs render the block-scale arm over them: no dequant
pass, Rubin-line only) and **P-c** (``DS_SF_P_C`` -- bf16 dS into the bf16 stage-3 renderings over dequantized bf16 q_T / k_T, the
oracle twin, selectable).

The ROW section (bottom of this module): REJECT tests run everywhere through the row's own ``mismatch()`` on REAL
``sdpa_mxfp8_backward`` graphs with the analyzer's cc faked to 10.7 (plus the adapter backstops); ACCEPT tests (``requires_rubin``)
PIN the engine (``select_engine``, the bracket-form rule) and run the graph against the MXFP8 backward oracle fed the SAME Stats,
masks and scale (sdpa-invariants s8).  Tolerances, ONE table (never widened -- report the magnitude):

    dV       the fp8 suite's recipe: ``assert_close_fp8_grad`` atol 0.08 / rtol 0.2 + the midpoint-flip budget (the e4m3 P feeds BMM2)
    dK, dQ   P-c: the bf16 ROW's recipe: ``torch.testing.assert_close`` atol 5e-2 / rtol 5e-2, NO flip budget (P-c forms dS from the
             fp32 P and the stage-3 GEMMs run over exactly dequantized bf16 operands: no fp8 rounding reaches dK / dQ) -- pinned equal
             to ``test_sdpa_bwd_dsl_sm107._TOL[bf16]``.  P-b: the fp8 suite's recipe (``assert_close_fp8_grad`` atol 0.08 / rtol 0.2 +
             the midpoint-flip budget): the kernel and the oracle each round dS to e4m3 per 32-block from fp32 values that differ by
             ~1e-6 relative, so an e4m3 midpoint flip of ONE dS cell moves one dK / dQ row by one e4m3 step x the block's 2^(e-127)
             x |Q| / |K| (the fp8 row's dS-flip class) -- a NEW cell's recipe, never a widening of the P-c cells'.

Calibrated ONCE on the bf16-dS bring-up ladder (dV max|diff| <= 3.9e-3, dK / dQ <= 1.6e-2, 0 outside on every cell, Rubin, 2026-09-30).  The
P-b chain is what ships (``config_sm107.DS_SF_POLICY_DEFAULT``), so the default accept cells run the oracle at ``quantize_ds=True``
(1x32 both ways = exactly P-b's convention) under the fp8 recipe on every gradient; the P-c cells (the ``ds_policy`` fixture's other
arm, or an explicit ``MXFP8_DS_SF_POLICY = DS_SF_P_C``) keep the fp32-dS oracle (``quantize_ds=False``) and the bf16 recipe on dK / dQ.
Under the default no dequant pass is launched (the launch census below).

Host-only pins of the body, runnable on any box with a cutlass-dsl that knows ``sm_107a`` (>= 4.8.0) and an nvdisasm that decodes it:

* STATIC (source + the loaded module against ``config_sm107``): the module binds the MXFP8 family, traces P-c and P-b and refuses
  P-a (the 32x32 tile scale); the TMEM map is the fp8 body's (the 2-slot P ring) and every scale-factor atom aliases the dead P
  slot at a slot-relative offset the body derives; the SMEM slabs are declared in the config's order (the desc-root tally describes
  THIS layout; the two P-b-only slabs are ``const_expr`` ternaries); ``Bars`` is the config's inventory (+ ``mb_p_sf_consumed``, the
  ONE new commit ring: one commit after every P.dO, one wait before every P store, the drain) and every LEADER-scope arrive is the
  relaxed form (no release, no cluster-scope arrive); the P path is the 16-pack scaled cvt with the CONSTANT byte ``CFG.P_SF_BYTE``
  -> the ``mb_p_sf_consumed`` wait -> ``tcgen05_st`` -> ``tcgen05_wait(STORE)`` -> the relaxed arrive; every MMA is block-scaled,
  BMM1 = ``mma_ss``, BMM2 = the block-scale ``mma_ts`` over the TMEM P slot; every SF operand has an elect-gated UTCCP site into the
  alias slot; every SF load rides its operand's ``_full`` mbarrier with the grown tx; the P-b dS quantizers spell the E8M0 rule through
  the shared helpers and publish the slot's four buffers behind ONE proxy fence and the same arrive; no per-tensor scale, amax or
  atomic survives; the ``q < seqlen_q_real`` band is a separate ``CFG.MASK_Q_PAD`` arm; the MMA warp's S issue order is a
  compile-time property of the mask arm (``S_LOOKAHEAD``, derived from the mask flags, never a literal: the loop's Q.K block is
  spelled once per ``const_expr`` arm -- identical text, the measured positions -- and exactly one traces per arm).
* SASS (one sm_107a trace-compile per row, ~5 s each): the fused P quantizer lowers to ``F2FP...SCALE_BY_C`` (8 per 16-pack, 0
  ``FMUL`` for it) and the FMUL arm to exactly one ``FMUL`` more per P element; the UTCCP count is the static atom count (15 = 4 in
  the prologue + 11 per q iteration); the block-scale MMA count is the k-step count; ONE ``STTM`` (the P store) and 11 commits; the
  register split reaches the binary (``USETMAXREG``), no new spills, no GPU-scope drain; the masked and the MASK_Q_PAD arms are the
  bit-word form (``R2P``); the tcgen05 stream order follows the mask arm (dense: the loop's K/Q copies + S MMA ahead of V/dO + dP;
  masked: dP first -- the lookahead).

The device cases (the bring-up ladder, the >= 12-fresh-process hang count, the oracle ``mxfp8_ref.compute_ref_backward``
flip-budget gate) are the ``requires_rubin`` ACCEPT tests of the ROW section below.  The
family-parametrized pins the fp8 body shares (desc_version wiring, ring-wait classification, knob / naming / k_dim pins, the
LDTM-order and spill SASS pins) run in ``test_sdpa_bwd_dsl_sm107.py`` with ``mxfp8`` added to its family table.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap

import inspect
import math

import pytest
import torch

import cudnn
from frost_test_utils import arch_known_to_the_dsl, assert_no_new_spills, nvdisasm_candidates, requires_dsl, requires_rubin, select_engine

try:  # module-level on purpose: a @cute.jit driver defined inside a test resolves `cute` / `cuda_driver` in the MODULE's globals
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute
except ImportError:  # requires_dsl skips every test; keep collection alive without the DSL
    cute = cuda_driver = None

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.fixture(autouse=True)
def _mock_target_for_cross_arch_contracts(monkeypatch):
    # The REJECT tests fake the analyzer's cc to 10.7 on non-Rubin hosts; bwd mismatch() carries the fwd rows' sm_107a DSL gate
    # (AGENTS.md Rule 7), so match the fake device with a fake compiler target -- the fwd suites' pattern.  Real Rubin runs use
    # the real build; the Rule-7 reject test below overrides this with False on purpose.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != _RUBIN_CC:
        from cudnn.frost import buffers
        from cudnn.sdpa.bwd import prepared_sm107

        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
        # The shipped block-scaled dS chain (P-b) declines typed off the Rubin line at check_support (the stage-3 arm's 576-column
        # exclusive TMEM), so the host-side plan / reject tests -- real adapters on whatever GPU the box has -- see the prepared
        # plan's device query answer SM107 too; the test that exercises that decline re-patches the query itself (_p_b(sm=)).
        monkeypatch.setattr(prepared_sm107, "_sm", lambda api: 107)


_KERNEL_FILE = "sm107/bprop_d256_mxfp8.py"
_FP8_FILE = "sm107/bprop_d256_fp8.py"
_TAG = "sdpa_bwd_sm107_main_mxfp8"
_ENGINE = "sdpa_bwd_sm107_mxfp8"
_FAMILY_NAME = "frost_sdpa_bwd"
_SLOT = 6  # engines/manifest.py: EngineSlot(6, opt_in=True) -> FROST_SDPA_BWD_ID_BASE + 6
_ENGINE_ID = 20_606
_D = 256
_RUBIN_CC = (10, 7)
_SM_RANGE = (107, 119)
# dV: the fp8 suite's tolerance recipe (test_sdpa_bwd_fp8_sm107.py::_FP8_GRAD_TOL): within atol 0.08 / rtol 0.2 of the oracle under
# assert_close_fp8_grad's midpoint-flip budget.  dK / dQ: the bf16 row's recipe (test_sdpa_bwd_dsl_sm107.py::_TOL[bf16]), no flip
# budget -- P-c's dK / dQ see no fp8 rounding.  ONE place each; never widened (report the magnitude instead).
_GRAD_TOL = dict(atol=0.08, rtol=0.2)
_BF16_GRAD_TOL = dict(atol=5e-2, rtol=5e-2)
_SF_MULT = 2.0**12  # scales a payload so every 32-block's amax exceeds 448: E8M0 bytes >= 128 (legal MXFP8) -- the sign-extension class


def _kernel_path(rel=_KERNEL_FILE):
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path

    return _sm100_kernel_path(rel)


def _kernel_source(rel=_KERNEL_FILE):
    path = _kernel_path(rel)
    assert os.path.isfile(path), f"kernels/{rel} is missing"
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _code_only(src):
    """Source with every string literal and comment blanked (spans replaced by spaces, newlines kept) -- the pins below look for
    code tokens the docstrings and error messages also spell (``tcgen05_st``, ``amax``, ``k_dim=1``)."""
    import io
    import tokenize

    blank = {tokenize.STRING, tokenize.COMMENT} | {
        getattr(tokenize, name) for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END") if hasattr(tokenize, name)
    }
    lines = src.splitlines(keepends=True)
    offs, acc = [], 0
    for ln in lines:
        offs.append(acc)
        acc += len(ln)
    out = list(src)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in blank:
            a = offs[tok.start[0] - 1] + tok.start[1]
            b = offs[tok.end[0] - 1] + tok.end[1]
            for i in range(a, b):
                if out[i] != "\n":
                    out[i] = " "
    return "".join(out)


def _load(**params):
    """Template-load the body the way the adapter does (``load_template`` + the config module's ``TemplateParams``)."""
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3
    from cudnn.sdpa.bwd.config_sm107 import TemplateParams

    params.setdefault("dtype_qkv", DTYPE_E4M3)
    return load_template(_kernel_path(), TemplateParams(**params), tag=_TAG)


def _def_body(code, name):
    """The code of ``def <name>(`` up to the next top-level ``def`` / ``class`` / decorator."""
    m = re.search(rf"^(?:@cute\.\w+\n)?def {name}\(", code, re.M)
    assert m, f"no def {name}( in the kernel"
    n = re.search(r"^(?:@cute\.\w+\n)?(?:def|class) ", code[m.end() :], re.M)
    return code[m.start() : m.end() + (n.start() if n else len(code))]


def _const_expr_block(code, guard):
    """The lines under ``if cutlass.const_expr(<guard>):`` (deeper-indented than the guard line, blank lines included), as text."""
    lines = code.splitlines(True)
    heads = [i for i, ln in enumerate(lines) if ln.strip() == f"if cutlass.const_expr({guard}):"]
    assert len(heads) == 1, f"expected ONE `if cutlass.const_expr({guard}):` guard, found {len(heads)}"
    i = heads[0]
    depth = len(lines[i]) - len(lines[i].lstrip())
    j = i + 1
    while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > depth):
        j += 1
    return "".join(lines[i + 1 : j])


def _traced_mma_body(mma, s_lookahead):
    """The MMA warp's code as ONE arm traces it: the ``const_expr`` block of the OTHER S issue order removed (``S_LOOKAHEAD`` folds one
    of the two spelled Q.K blocks out)."""
    drop = "not S_LOOKAHEAD" if s_lookahead else "S_LOOKAHEAD"
    block = _const_expr_block(mma, drop)
    head = f"if cutlass.const_expr({drop}):"
    i = mma.index(head)
    j = mma.index(block, i) + len(block)
    return mma[:i] + mma[j:]


# =========================================================================== static pins: module binding and config agreement


def test_module_binds_the_mxfp8_family_and_traces_p_b():
    """The bare record traces the shipped P-b chain (e4m3 dS x 2 + atoms, 2-deep dS ring, bf16 dV, the FMUL arm, no q band) on the
    MXFP8 family -- the config's DS_SF_POLICY_DEFAULT, not a literal of this body."""
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from cudnn.sdpa.bwd.config_sm107 import DS_SF_P_B, DS_SF_POLICY_DEFAULT, FAMILY_MXFP8, _FLAVOR

    mod = _load()
    cfg = mod.CFG
    assert cfg.IS_MXFP8 == 1 and cfg.IS_FP8 == 1, "the MXFP8 body is the fp8-CLASS pipeline with block scaling"
    assert cfg.DS_SF_POLICY == DS_SF_POLICY_DEFAULT == DS_SF_P_B and cfg.DTYPE_DS == DTYPE_E4M3 and cfg.XFER_STAGES == 2 and cfg.DTYPE_O == DTYPE_BF16
    assert mod._IS_P_B and (cfg.DS_PAYLOADS, cfg.DS_SF_ATOMS) == (2, 2)
    assert cfg.SCALED_FP8_PACK == 0 and cfg.MASK_Q_PAD == 0
    assert cfg.P_SF_BYTE == 119 and cfg.STAGES_TMEM_P == 2 and not hasattr(cfg, "STAGES_SMEM_P")
    assert _FLAVOR[FAMILY_MXFP8] in ("sm107 bwd d256 mxfp8",)
    assert mod.DESC_VERSION == 0 and isinstance(mod.SPIN_RING_WAITS, bool)


def test_module_refuses_p_a_and_traces_p_b_at_load():
    """P-a (the 32x32 tile scale: optional validation only) is a typed refusal at load; P-b (exact 1x32 block-scaled e4m3 dS BOTH
    ways) traces: e4m3 dS, two payload rings + two staged atoms per 2-deep ring stage, and the body constants its byte arithmetic
    rests on.  The explicit P-c record still traces with every P-b constant folded to its neutral value."""
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3
    from cudnn.sdpa.bwd.config_sm107 import DS_SF_P_A, DS_SF_P_B, DS_SF_P_C, SF_ATOM_BYTES

    with pytest.raises(NotImplementedError, match="P-a"):
        _load(ds_sf_policy=DS_SF_P_A)
    mod = _load(ds_sf_policy=DS_SF_P_B)
    cfg = mod.CFG
    assert cfg.DS_SF_POLICY == DS_SF_P_B and cfg.DTYPE_DS == DTYPE_E4M3 and cfg.BPE_DS == 1
    assert (cfg.XFER_STAGES, cfg.DS_PAYLOADS, cfg.DS_SF_ATOMS) == (2, 2, 2)
    assert mod._IS_P_B and mod.DS_STORAGE_DTYPE.__name__ == "Float8E4M3FN"
    assert (mod._DS_BLOCKS_PER_WG, mod._DS_PACK, mod._DS_SF_STAGE_BYTES, mod._DS_SF_RING_BYTES) == (2, 16, 2 * SF_ATOM_BYTES, 4 * SF_ATOM_BYTES)
    assert (mod._DS_SF_ATOM_DK_OFF, mod._DS_SF_ATOM_DQ_OFF) == (0, SF_ATOM_BYTES)
    assert (mod.P_D_BLOCK, mod.P_TMA_ITERS) == (cfg.TILE_N, 1), "both warpgroups' halves in ONE 128-B row: the fp8 body's e4m3 dS geometry"
    assert mod._DS_KV_RING_ELEMS == cfg.XFER_STAGES * mod.dSBufferElems and mod.DESC_VERSION == 0
    pc = _load(ds_sf_policy=DS_SF_P_C)
    assert not pc._IS_P_B and (pc._DS_BLOCKS_PER_WG, pc._DS_SF_STAGE_BYTES, pc._DS_KV_RING_ELEMS) == (0, 0, 0)


def test_tmem_map_is_the_fp8_ring_and_every_sf_atom_aliases_the_dead_p_slot():
    """S | dP | dV | P slot 0 [512, 544) | P slot 1 [544, 576), RSVD 0 -- the fp8 body's map.  The SF offsets are SLOT-RELATIVE: the
    BMM1 band K 0 | V 8 | Q 16 | dO 24 (= 32, the whole slot) and the BMM2 band P 0 | dOT 4 (= 12); the body derives every SF base
    as P_OFF + (p ^ 1) * P_COLS + SF_*_OFF (``_sf_alias_views``) and stores P with tcgen05_st + wait(STORE) (delta 1)."""
    mod = _load()
    lay = mod.LAYOUT
    assert (lay.P_OFF, lay.P_COLS, mod.CFG.STAGES_TMEM_P, lay.TOTAL_COLS) == (512, 32, 2, 576) and lay.P_OFF + 2 * lay.P_COLS == lay.TOTAL_COLS
    assert (lay.RSVD_OFF, lay.RSVD_COLS) == (576, 0), "no dedicated SF columns"
    assert (lay.SF_K_OFF, lay.SF_V_OFF, lay.SF_Q_OFF, lay.SF_dO_OFF, lay.SF_P_OFF, lay.SF_dOT_OFF) == (0, 8, 16, 24, 0, 4)
    assert (lay.SF_BMM1_COLS, lay.SF_BMM2_COLS) == (32, 12) and lay.SF_BMM1_COLS <= lay.P_COLS and lay.SF_BMM2_COLS <= lay.P_COLS
    code = _code_only(_kernel_source())
    views = _def_body(code, "_sf_alias_views")
    for name in ("SF_K", "SF_V", "SF_Q", "SF_dO", "SF_P", "SF_dOT"):
        assert f"LAYOUT.{name}_OFF" in views, f"the alias helper never subviews LAYOUT.{name}_OFF -- an SF operand has no TMEM base"
    assert "(p_slot ^ cutlass.Int32(1)) * cutlass.Int32(LAYOUT.P_COLS)" in views, "the SF slot is the OTHER P slot (p ^ 1), scaled by P_COLS"
    mma = _def_body(code, "_mma_warp")
    assert "LAYOUT.SF_" not in mma, "every SF base comes from _sf_alias_views (no absolute SF column survives)"
    assert mma.count("_sf_alias_views(tmem_P, ") == 2, "one alias view per tile prologue + one per q iteration"
    assert "tmem_raw.subview(cutlass.Int32(LAYOUT.P_OFF))" in mma, "the TMEM P ring is addressed through the layout"
    assert "tcgen05_st" in code and "Tcgen05Wait.STORE" in code, "the TMEM P store (tcgen05_st + wait(STORE)) is the fp8 body's (delta 1)"


@pytest.mark.parametrize("policy", ["P-c", "P-b"])
def test_smem_slabs_are_declared_in_the_config_layout_order(policy):
    """The desc-root tally (``config_sm107.smem_layout``) describes the kernel only if the kernel declares its 1024-B slabs in the
    SAME order: the six SF slabs BEFORE the dS slabs (their roots stay under the 256 KiB version-0 window); no SMEM P ring (it is the
    TMEM ring).  The two P-b-only slabs (sdS_SF between sdOT_SF and sdS, sdS_kv after sdS) are ``const_expr(_IS_P_B)`` ternaries -- in
    the SOURCE under both policies, in the LAYOUT only under P-b -- so the order is compared over the policy's slabs and the source's
    extras must be exactly them."""
    from cudnn.sdpa.bwd import config_sm107 as cfgmod
    from cudnn.sdpa.bwd.config_sm107 import DS_SF_P_B, DS_SF_P_C

    mod = _load(ds_sf_policy=DS_SF_P_B if policy == "P-b" else DS_SF_P_C)
    want = [re.split(r"[\[(]", slab.name)[0] for slab in cfgmod.smem_layout(mod.CFG)]
    body = _def_body(_code_only(_kernel_source()), "_kernel")
    got = re.findall(r"^\s+(\w+)_raw = cutlass\.Array\([^\n]*alignment=1024[^\n]*space=cutlass\.AddressSpace\.smem", body, re.M)
    assert [g for g in got if g in want] == want, f"kernel slab order {got} != config layout {want}"
    assert set(got) - set(want) == (set() if policy == "P-b" else {"sdS_SF", "sdS_kv"}), (got, want)
    for name in ("sdS_SF", "sdS_kv"):
        decl = [ln for ln in body.splitlines() if re.match(rf"^\s+{name}_raw = cutlass\.Array\(", ln)]
        assert len(decl) == 1 and decl[0].rstrip().endswith("if cutlass.const_expr(_IS_P_B) else None"), f"{name}_raw is a P-b-only ternary: {decl}"
    assert cfgmod.desc_version(mod.CFG) == 0 == mod.DESC_VERSION
    roots = dict(cfgmod.desc_roots(mod.CFG))
    assert max(roots.values()) == roots["sdOT_SF[2]"] == 226304 < cfgmod.TCGEN05_V0_ADDR_LIMIT
    assert not any(k.startswith("sP[") for k in roots), "the P ring is in TMEM: no sP root"


def test_bars_are_the_config_inventory_and_every_leader_arrive_is_relaxed():
    """``Bars`` == ``mbar_stage_counts(CFG)`` (25 rings: the fp8 body's 24 + ``mb_p_sf_consumed``, no p_empty); ``mb_p_ready`` is built
    on ``STAGES_TMEM_P`` with the relaxed ``Producer.LEADER`` (TMEM data is ordered by tcgen05_wait(STORE): no release producer is
    left on this body), ``mb_p_sf_consumed`` is a 1-stage ``MMA_COMMIT`` ring at ``MMA_COMMIT_ARRIVES``, and no cluster-scope release /
    hand-rolled arrive appears (rules/frost-tile-dsl.md s3)."""
    from cudnn.sdpa.bwd import config_sm107 as cfgmod

    mod = _load()
    counts = cfgmod.mbar_stage_counts(mod.CFG)
    assert tuple(mod.Bars._fields) == tuple(counts), "Bars must be the config's inventory, in order"
    assert counts["mb_p_ready"] == mod.CFG.STAGES_TMEM_P == 2 and counts["mb_p_sf_consumed"] == 1 and len(counts) == 25
    code = _code_only(_kernel_source())
    bars_src = _def_body(code, "_make_bars")
    assert "Producer.LEADER_RELEASE" not in code, "no lane-written SMEM operand crosses the pair: no release arrive on this body"
    for name in ("mb_s_acc_empty", "mb_dp_empty", "mb_dv_acc_empty", "mb_p_ready"):
        seg = bars_src.split(f"{name}=")[1].split("\n")[0]
        assert "producer=Producer.LEADER," in seg and "scope=Scope.LEADER" in seg, f"{name} must be the relaxed LEADER form: {seg}"
    assert "stages=CFG.STAGES_TMEM_P" in bars_src.split("mb_p_ready=")[1].split("mb_p_sf_consumed=")[0]
    seg = bars_src.split("mb_p_sf_consumed=")[1].split("\n")[0]
    assert "stages=1" in seg and "init_count=MMA_COMMIT_ARRIVES" in seg and "producer=Producer.MMA_COMMIT" in seg, seg
    assert "arrive_on_leader" not in code and "mbarrier_arrive(" not in code, "no hand-rolled cluster arrive in the body"
    assert "MemScope.CLUSTER" not in code
    # The init loop covers the TMEM ring depth and the new ring is initialised (an uninitialised mbar faults its first wait).
    assert re.search(r"range_constexpr\(CFG\.STAGES_TMEM_P\):\s+bars\.mb_p_ready\[p\]\.init\(\)\s+bars\.mb_p_sf_consumed\.init\(\)", code)
    assert "STAGES_SMEM_P" not in code


def test_the_p_sf_consumed_ring_is_one_commit_per_p_do_and_one_wait_per_p_store_plus_the_drain():
    """The ONE new ring.  Producer: the MMA warp commits it (multicast, pred=elect_p) exactly once per q iteration, right after the
    BMM2 ``mma_ts`` (the commit tracks P.dO[i] and the SF copies issued before it).  Consumer: every softmax lane waits it right
    before its ``tcgen05_st`` of P (after the exp2 / pack work, so the wait hides), from a PRE-ARMED state (the kernel's first store
    has no prior P.dO), and once more after the persistent loop (the P15 drain of the last async multicast commit) BEFORE the
    ``mb_tmem_dealloc`` arrives keep both CTAs resident."""
    code = _code_only(_kernel_source())
    mma = _def_body(code, "_mma_warp")
    assert mma.count("bars.mb_p_sf_consumed.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)") == 1, "exactly one commit site"
    i_loop, i_bmm2 = mma.index("while is_valid_tile"), mma.index("mma_ts(")
    i_dodv_empty = mma.index("bars.mb_dodv_empty[dodv_full_state.idx].arrive(")
    i_commit = mma.index("bars.mb_p_sf_consumed.arrive(")
    assert i_loop < i_bmm2 < i_dodv_empty < i_commit, "the commit follows P.dO[i] inside the persistent loop"
    soft = _def_body(code, "_softmax_warp_group")
    assert soft.count("bars.mb_p_sf_consumed.wait(") == 2, "one ring wait before the P store + one drain after the loop"
    i_pack = soft.index("fp32_to_fp8_pack_scaled(")
    i_wait = soft.index("bars.mb_p_sf_consumed.wait(p_sf_state.phase, spin=SPIN_RING_WAITS)")
    i_st = soft.index("nvvm.tcgen05_st(")
    assert i_pack < i_wait < i_st, "pack -> wait mb_p_sf_consumed -> tcgen05_st: the wait guards the store and sits after the compute"
    assert "p_sf_state = PipelineState.start(phase=1)" in soft, "pre-armed: the kernel's first P store has no prior P.dO"
    assert soft.count("p_sf_state = advance(p_sf_state, 1)") == 1
    i_drain = soft.index("bars.mb_p_sf_consumed.wait(p_sf_state.phase)\n")
    i_dealloc = soft.index("bars.mb_tmem_dealloc.arrive()")
    assert soft.rindex("while is_valid_tile") < i_wait < i_drain < i_dealloc, "the P15 drain sits after the persistent loop, before the dealloc arrives"


def test_p_path_is_the_16_pack_scaled_cvt_with_the_constant_byte_into_the_tmem_p_slot():
    """Delta 1 + 6: ``fp32_to_fp8_pack_scaled(<16 values>, CFG.P_SF_BYTE, dtype=STORAGE_DTYPE, fused=bool(CFG.SCALED_FP8_PACK))``
    -- the byte a PYTHON INT (the cvt.u8.u32 immediate is the ONE constant form that assembles; a register or a kernel parameter
    ICEs ptxas C7907) -- the four words per pack go to TMEM as Int32 bit patterns (``tcgen05_st 32x32b``, 16 words = this wg's 16
    columns at P_OFF + p_slot * P_COLS + p_col_off) after the ``mb_p_sf_consumed`` wait, then ``tcgen05_wait(STORE)`` then the relaxed
    arrive on ``mb_p_ready`` with the ``STAGES_TMEM_P`` advance -- the fp8 body's pair; no SMEM P store, no proxy fence for P."""
    code = _code_only(_kernel_source())
    soft = _def_body(code, "_softmax_warp_group")
    m = re.search(r"fp32_to_fp8_pack_scaled\(\s*_vals,\s*CFG\.P_SF_BYTE,\s*dtype=STORAGE_DTYPE,\s*fused=bool\(CFG\.SCALED_FP8_PACK\)\)", soft)
    assert m, "the P quantizer must be fp32_to_fp8_pack_scaled(<16-pack>, CFG.P_SF_BYTE, dtype=STORAGE_DTYPE, fused=bool(CFG.SCALED_FP8_PACK))"
    assert "fp32_to_fp8x2_scaled" not in code, "the pair twin takes the FMUL arm for a constant byte -- never the P path"
    assert re.search(r"chunk_P_words = cutlass\.Vector\.from_elements\(tuple\(p_words\), cutlass\.Int32\)", soft), "Int32 bit patterns into TMEM"
    i_pack = soft.index("fp32_to_fp8_pack_scaled(")
    i_wait = soft.index("bars.mb_p_sf_consumed.wait(", i_pack)
    i_st = soft.index("nvvm.tcgen05_st(", i_wait)
    i_wst = soft.index("nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)", i_st)
    i_arrive = soft.index("bars.mb_p_ready[p_slot].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)", i_wst)
    assert i_pack < i_wait < i_st < i_wst < i_arrive, "pack -> wait p_sf_consumed -> tcgen05_st -> wait(STORE) -> relaxed arrive, in that order"
    assert "tmem_base + cutlass.Int32(LAYOUT.P_OFF) + p_slot * cutlass.Int32(LAYOUT.P_COLS) + p_col_off" in soft
    assert "p_col_off = wg_id * cutlass.Int32(_SMX_CHUNK * CFG.BPE // 4)" in soft
    assert "advance(p_ready_state, CFG.STAGES_TMEM_P)" in soft and "STAGES_SMEM_P" not in soft
    assert (
        soft.count("store_swizzled(") == 4 and soft.count("nvvm.fence_proxy(") == 2
    ), "the dS stores (P-c bf16 | P-b ds_dk + ds_dq) and the dV store, the dS-slot and dV fences only: no SMEM P store"
    # The exp2 argument is UNFOLDED (lse * log2e, no shift): the stats warp multiplies by log2e only.
    stats = _def_body(code, "_scheduler_stats_warp")
    assert re.search(r"lse_s = lse_tensor\[[^\]]+\] \* cutlass\.Float32\(_LOG2E\)\s*$", stats, re.M), "lse_s = lse * log2e, nothing folded in"
    assert "lse_log2_shift" not in code


def test_p_b_quantizers_use_the_shared_helpers_and_publish_four_buffers_behind_one_fence():
    """P-b's two dS quantizers spell the E8M0 rule ONCE, through the shared helpers: ``abs_max_tree`` (in-lane, the q-blocks of
    a lane's kv row) and ``warp_abs_max_f32`` (one redux per q column = the warp's 32-kv block); ``e8m0_pair`` for the dk bytes
    and ``e8m0_pair_u`` (the same rule on two warp-uniform amaxes, one packed multiply per column pair) for the dq bytes -- both the
    oracle's ``e8m0_ceil(amax * fp32(1/448))``; ``fp32_to_fp8_pack_scaled`` with the per-lane DATA byte for ds_dk and
    ``fp32_to_fp8x4_scaled_pairs`` (one lone scaled cvt per element with ITS column's byte) for ds_dq -- never a re-literaled
    constant, never the raw DSL redux, never the pair twin or the plain pack.  The ``fp32(1/448)`` of the dq pairs is hoisted ONCE
    per kernel (``opaque_e4m3_max_rcp_in_lane``, before the persistent loop).  The atom bytes come from ``sf_atom_byte`` and lane
    l's pair from the select tree over the SAME pair words the payload was scaled by.  Per iteration: ONE ``mb_ds_smem_empty``
    wait, then ds_dk, ds_dq, the dk u16 and the two dq u8 stores, then ONE ``fence_proxy`` and the SAME ``mb_ds_smem_full``
    arrive (the barrier table's counts are untouched)."""
    code = _code_only(_kernel_source())
    soft = _def_body(code, "_softmax_warp_group")
    assert soft.count("e8m0_pair(") == 1 and soft.count("e8m0_pair_u(") == 2 and "e8m0_from_amax(" not in code and "e8m0_rcp(" not in code
    assert soft.count("abs_max_tree(") == 1 and soft.count("warp_abs_max_f32(") == 1 and "warp_redux_sync" not in code
    assert "fp32_to_fp8x2_scaled" not in code and "fp32_to_fp8x2(" not in code and "fp32_to_fp8_pack(" not in code
    assert re.search(r"fp32_to_fp8_pack_scaled\(_vals, dk_bytes\[_b\], dtype=DS_STORAGE_DTYPE, fused=bool\(CFG\.SCALED_FP8_PACK\)\)", soft)
    assert soft.count("fp32_to_fp8x4_scaled_pairs(") == 1 and re.search(
        r"fp32_to_fp8x4_scaled_pairs\(\s*\[chunk_dS\[_c0 \+ _i\] for _i in range\(4\)\], _p01, _p23, dtype=DS_STORAGE_DTYPE, fused=bool\(CFG\.SCALED_FP8_PACK\)",
        soft,
    )
    assert re.search(r"e8m0_pair_u\(col_max\[_q \* 4\], col_max\[_q \* 4 \+ 1\], dq_rcp_max\)", soft) and re.search(
        r"e8m0_pair_u\(col_max\[_q \* 4 \+ 2\], col_max\[_q \* 4 \+ 3\], dq_rcp_max\)", soft
    )
    assert soft.count("opaque_e4m3_max_rcp_in_lane(") == 1 and soft.index("opaque_e4m3_max_rcp_in_lane(") < soft.index(
        "while is_valid_tile"
    ), "hoisted once per kernel"
    assert not re.search(r"\b448\b|0x3B124925|<<\s*23|\b254\b", code), "a re-literaled E8M0 rule (448, 1/448, 254 - e, << 23: the shared helpers own it)"
    assert soft.count("sf_atom_byte(") == 2 and "_select_by_lane(dq_pairs, lane_bits)" in soft
    assert soft.count("bars.mb_ds_smem_empty[ds_slot].wait(") == 1, "ONE ring wait site for both policies (the ring-wait census)"
    # The payload words cross the ring wait as Int32 WORDS and are bitcast to the storage dtype in the store's own block (a 64-element
    # fp8 vector live across the wait is legalized element-wise: ~420 instructions per iteration, MEASURED on 987eb02b).
    i_wait = soft.index("bars.mb_ds_smem_empty[ds_slot].wait(")
    assert soft.index("chunk_dS_dk_words = cutlass.Vector.from_elements(tuple(dk_words), cutlass.Int32)") < i_wait
    assert soft.index("chunk_dS_dq_words = cutlass.Vector.from_elements(tuple(dq_words), cutlass.Int32)") < i_wait
    assert i_wait < soft.index("chunk_dS_dk = chunk_dS_dk_words.bitcast(DS_STORAGE_DTYPE)") < soft.index("store_swizzled(chunk_dS_dk")
    assert i_wait < soft.index("chunk_dS_dq = chunk_dS_dq_words.bitcast(DS_STORAGE_DTYPE)") < soft.index("store_swizzled(chunk_dS_dq")
    order = [
        "bars.mb_ds_smem_empty[ds_slot].wait(",
        "store_swizzled(chunk_dS_dk",
        "store_swizzled(chunk_dS_dq",
        "sdS_SF16_raw.subview(",
        "sdS_SF_raw.subview(sf_stage_off + dq_sf_off)",
        "sdS_SF_raw.subview(sf_stage_off + dq_sf_off + cutlass.Int32(SF_ATOM_LINE_BYTES))",
    ]
    pos = [soft.index(tok) for tok in order]
    assert pos == sorted(pos), order
    i_fence = soft.index("nvvm.fence_proxy(", pos[-1])
    i_arrive = soft.index("bars.mb_ds_smem_full[ds_slot].arrive()", i_fence)
    assert soft.count("nvvm.fence_proxy(", pos[0], i_arrive) == 1 and soft.count("bars.mb_ds_smem_full[ds_slot].arrive()") == 1
    assert "chunk_dS.to(DS_STORAGE_DTYPE)" in soft, "the P-c bf16 arm is untouched"
    # the P-b slabs and descriptors: TMASTG stores four buffers per slot; the host builds the two payload boxes and two atom descriptors
    stg = _def_body(code, "_tmastg_warp")
    assert stg.count("tma_store_tile(") == 5 and stg.count("coord_0=cutlass.Int32(0)") == 2 and ".shifted(_DS_SF_ATOM_DQ_OFF)" in stg
    assert stg.count("tma_store_commit()") == 2 and stg.count("tma_store_wait(0)") == 2, "one bulk group per slot, one per dV tile"
    assert "kv_sf_tile, ds_bh, coord_0" in stg and "q_sf_tile, ds_bh, coord_0" in stg
    host = _def_body(code, "_host")
    assert host.count("build_ds_sf_atom_desc(") == 2
    assert "inner_tiles=n_q_sf_tiles, outer_tiles=n_kv_sf_tiles" in host and "inner_tiles=n_kv_sf_tiles, outer_tiles=n_q_sf_tiles" in host
    assert "ds_payload0 = ds_dk_tensor if cutlass.const_expr(_IS_P_B) else ds_tensor" in host


def test_no_per_tensor_scale_amax_or_atomic_survives():
    """The row produces no amax and reads no descale / scale scalar: the block scale factors
    dequantize inside every MMA.  Compiled OUT, not gated (a None-specialized amax path reading absent tensors is dead code).  The
    P-b block amaxes (``abs_max_tree`` in-lane, ``warp_abs_max_f32`` across the warp) are per-BLOCK scale inputs, not a per-tensor
    amax: no ``amax`` identifier, no atomic, no scalar."""
    code = _code_only(_kernel_source())
    for tok in ("atomicrmw", "descale", "scale_s", "scale_dv", "scale_dp", "amax", "opaque_f32_zero", "fmax_f32", "OUT_IS_FP8", "DS_IS_FP8"):
        assert tok not in code, f"{tok!r} survived the port"
    sig = _def_body(code, "_kernel").split(") -> None:")[0]
    tensors = re.findall(r"(\w+): cute\.Tensor", sig)
    assert tensors == ["lse_tensor", "do_dot_tensor"], f"the kernel's only GMEM vectors are lse and delta; got {tensors}"
    host = _def_body(code, "_host").split(") -> None:")[0]
    # The Launch ABI, append-only: the four P-b operands follow every pre-existing positional (scalars included) and default to None.
    assert re.findall(r"(\w+)_tensor: (?:Optional\[)?cute\.Tensor", host) == [
        "q",
        "k",
        "v",
        "do",
        "do_T",
        "dv",
        "ds",
        "lse",
        "do_dot",
        "sf_q",
        "sf_k",
        "sf_v",
        "sf_do",
        "sf_do_T",
        "ds_dk",
        "ds_dq",
        "sf_ds_dk",
        "sf_ds_dq",
    ]
    assert "seqlen_q_real: cutlass.Int32" in host and "seqlen_kv_real: cutlass.Int32" in host
    assert "ds_tensor: Optional[cute.Tensor]" in host, "ds_ws is None under P-b"
    tail = host.split("seqlen_q_real: cutlass.Int32")[1]
    assert re.findall(r"(\w+)_tensor: Optional\[cute\.Tensor\] = None", tail) == ["ds_dk", "ds_dq", "sf_ds_dk", "sf_ds_dq"], tail
    assert tail.index("sf_ds_dq_tensor") < tail.index("stream:")


def _balanced_call(src, name, start=0):
    """The text of the first ``name(...)`` call at or after ``start``, parens balanced (a one-level regex misses ``a(b(c))``)."""
    i = src.index(name + "(", start)
    j, depth = i + len(name) + 1, 1
    while depth:
        depth += (src[j] == "(") - (src[j] == ")")
        j += 1
    return src[i:j]


def _all_calls(src, name):
    out, pos = [], 0
    while True:
        try:
            c = _balanced_call(src, name, pos)
        except ValueError:
            return out
        out.append(c)
        pos = src.index(c, pos) + len(c)


def test_every_mma_is_block_scaled_and_bmm2_is_the_block_scale_mma_ts_over_the_tmem_p_slot():
    """Delta 4 + 1: three ``MmaDesc`` all ``is_block_scale=True`` on ``Tcgen05MxInstrDesc`` (k_dim from the config); four ``mma_ss``
    sites in source (the Q.K prologue, the loop's Q.K spelled once per S-issue-order arm, dO.V) -- three per traced arm -- and ONE
    block-scale ``mma_ts`` (P.dO), each with ``tmem_sf_a`` / ``tmem_sf_b`` from the alias views; BMM2's A operand is the TMEM P slot
    ``tmem_P.subview(p_slot * P_COLS)`` and it accumulates from q_lo."""
    code = _code_only(_kernel_source())
    mma = _def_body(code, "_mma_warp")
    assert (
        mma.count("MmaDesc(")
        == 3
        == mma.count("is_block_scale=True")
        == mma.count("sf_blocks_per_step=CFG.SF_BLOCKS_PER_STEP")
        == mma.count("scale_vec_size=SCALE_VEC_SIZE")
    )
    assert mma.count("Tcgen05MxInstrDesc.build(") == 3 and "Tcgen05InstrDesc.build(" not in code
    assert sorted(set(re.findall(r"\bk_dim\s*=\s*([^,)\s]+)", code))) == ["CFG.IDESC_K_DIM"]
    ss = _all_calls(mma, "mma_ss")
    ts = _all_calls(mma, "mma_ts")
    assert len(ss) == 4 and len(ts) == 1, (len(ss), len(ts))
    for s_lookahead in (False, True):
        traced = _traced_mma_body(mma, s_lookahead)
        assert len(_all_calls(traced, "mma_ss")) == 3 and len(_all_calls(traced, "mma_ts")) == 1, "three mma_ss + one mma_ts per traced arm"
    assert all("tmem_sf_a=" in c and "tmem_sf_b=" in c for c in ss + ts), ss + ts
    assert "bmm2_dv_desc" in ts[0] and "tmem_P.subview(p_slot * cutlass.Int32(LAYOUT.P_COLS))" in ts[0] and "accumulate=(q_iter > q_lo)" in ts[0], ts
    assert "tmem_sf_a=tmem_SF_P" in ts[0] and "tmem_sf_b=tmem_SF_dOT" in ts[0]
    assert not any("bmm2_dv_desc" in c for c in ss), "BMM2 is the mma_ts (A in TMEM), never an mma_ss"
    assert all("tmem_sf_a=tmem_SF_K" in c and "tmem_sf_b=tmem_SF_Q" in c for c in ss if "bmm1_s_desc" in c)
    assert all("tmem_sf_a=tmem_SF_V" in c and "tmem_sf_b=tmem_SF_dO" in c for c in ss if "bmm1_dp_desc" in c)
    assert "a_major=0" in mma and "b_major=1" in mma, "the BMM2 idesc names P K-major and dO_T transposed"


def test_every_sf_operand_has_an_elect_gated_utccp_site_right_before_its_mma():
    """Delta 3: eight ``_utccp_sf_atoms`` sites per traced arm -- the prologue's K + Q, then per q iteration V + dO (before dO.V), K + Q
    (before the loop's Q.K: ahead of dO.V on the dense arm, after it on the masked arms) and P + dOT (before P.dO); ten in source, the
    loop's K + Q spelled once per arm -- each under ``if nvvm.elect_sync():`` (the macro does not elect: an un-gated call is 32
    redundant copies), every target an alias view, and the static atom count they copy (15) is what the SASS pin counts."""
    mod = _load()
    code = _code_only(_kernel_source())
    mma = _def_body(code, "_mma_warp")
    sites = [m.start() for m in re.finditer(r"_utccp_sf_atoms\(tmem_SF_(\w+), ", mma)]
    names = re.findall(r"_utccp_sf_atoms\(tmem_SF_(\w+), ", mma)
    assert names == ["K", "Q", "K", "Q", "V", "dO", "K", "Q", "P", "dOT"], names
    assert re.findall(r"_utccp_sf_atoms\(tmem_SF_(\w+), ", _traced_mma_body(mma, False)) == ["K", "Q", "K", "Q", "V", "dO", "P", "dOT"]
    assert re.findall(r"_utccp_sf_atoms\(tmem_SF_(\w+), ", _traced_mma_body(mma, True)) == ["K", "Q", "V", "dO", "K", "Q", "P", "dOT"]
    for i in sites:
        prev = mma[:i].rstrip().splitlines()
        gate = [ln for ln in prev[-3:] if "if nvvm.elect_sync():" in ln]
        assert gate, f"UTCCP site at {i} is not elect-gated: {prev[-3:]}"
    prologue = mod._SF_ATOMS_K + mod._SF_ATOMS_Q
    per_iter = mod._SF_ATOMS_V + mod._SF_ATOMS_dO + mod._SF_ATOMS_K + mod._SF_ATOMS_Q + mod._SF_ATOMS_P + mod._SF_ATOMS_dOT
    assert (prologue, per_iter, prologue + per_iter) == (4, 11, 15) and (mod._SF_ATOMS_K, mod._SF_ATOMS_Q, mod._SF_ATOMS_P, mod._SF_ATOMS_dOT) == (2, 2, 1, 2)
    # Every copy (the P-SF one included) sits inside the persistent loop -- the alias slot alternates per q iteration -- and the
    # constant P atom's descriptor is hoisted once; the macro itself is defined once and elects nothing.
    i_loop = mma.index("while is_valid_tile")
    assert all(i > i_loop for i in sites) and mma.index("desc_P_SF = sP_SF[0].desc()") < i_loop
    assert mma.count("_utccp_sf_atoms(tmem_SF_P, desc_P_SF, _SF_ATOMS_P)") == 1
    macro = _def_body(code, "_utccp_sf_atoms")
    assert "elect_sync" not in macro and "tcgen05_cp(" in macro and "Tcgen05CpShape.SHAPE_32X128B" in macro and "Tcgen05CpMulticast.WARPX4" in macro


def test_s_issue_order_is_a_compile_time_property_of_the_mask_arm():
    """``S_LOOKAHEAD`` is derived from the mask flags (the predicate that folds the mask IR), never a literal: False on the dense arm
    (no mask at all), True on every masked arm (causal / SWA / kv-padded and the q-pad band alone).  In the MMA warp the loop's Q.K
    block is spelled once per arm under mutually exclusive ``cutlass.const_expr`` guards -- IDENTICAL text under its runtime guard
    (the same two waits, two copies, MMA, two commits, two advances: ``q_iter > q_lo`` at the iteration top, after the mb_dp_empty
    wait, for the dense arm; ``q_iter + 1 < q_hi`` after dO.V for the masked arms, N - 1 instances per tile either way) -- so each
    traced arm carries exactly ONE loop Q.K issue site plus the prologue, the same waits, arrives and 11 commits.  Measured on Rubin:
    the dense row is faster with Q.K at the top (+1.7..1.8 % main kernel / +0.9 % row at 8K on this body, +4.0 % / +2.2 % on the
    predecessor body), the causal row with the lookahead."""
    code = _code_only(_kernel_source())
    defs = re.findall(r"^S_LOOKAHEAD: bool = (.+)$", code, re.M)
    assert len(defs) == 1, defs
    assert "CFG.MASK_FLAGS" in defs[0] and "MASK_NONE" in defs[0] and "CFG.MASK_Q_PAD" in defs[0], f"S_LOOKAHEAD must derive from the mask flags: {defs}"
    assert not re.fullmatch(r"(?:bool\()?(?:True|False)\)?", defs[0].strip()), f"S_LOOKAHEAD is a literal: {defs}"
    assert _load().S_LOOKAHEAD is False, "the bare record (dense, no q band) takes the top-of-iteration order"
    assert _load(window_right=0).S_LOOKAHEAD is True, "causal keeps the lookahead"
    assert _load(mask_q_pad=True).S_LOOKAHEAD is True, "the q-pad band alone counts as masked"
    mma = _def_body(code, "_mma_warp")
    assert code.count("const_expr(not S_LOOKAHEAD)") == 1 == code.count("const_expr(S_LOOKAHEAD)"), "one guard per arm, both in the MMA warp"
    assert mma.count("const_expr(not S_LOOKAHEAD)") == 1 == mma.count("const_expr(S_LOOKAHEAD)")
    i_loop = mma.index("for q_iter in cutlass.range(q_lo, q_hi")
    i_dp_empty = mma.index("bars.mb_dp_empty[dp_empty_state.idx].wait(")
    i_top, i_look = mma.index("if cutlass.const_expr(not S_LOOKAHEAD):"), mma.index("if cutlass.const_expr(S_LOOKAHEAD):")
    i_dov, i_pdo = mma.index("mma_ss(bmm1_dp_desc"), mma.index("mma_ts(")
    assert i_loop < i_dp_empty < i_top < i_dov < i_look < i_pdo, "dense: dp_empty wait -> Q.K[i] -> dO.V[i]; masked: dO.V[i] -> Q.K[i+1] -> P.dO[i]"
    top, look = _const_expr_block(mma, "not S_LOOKAHEAD"), _const_expr_block(mma, "S_LOOKAHEAD")
    assert top.lstrip().startswith("if q_iter > q_lo:\n") and look.lstrip().startswith("if (q_iter + cutlass.Int32(1)) < q_hi:\n"), (top[:60], look[:60])
    body_top = textwrap.dedent(top.split("\n", 1)[1]).rstrip()
    body_look = textwrap.dedent(look.split("\n", 1)[1]).rstrip()
    assert body_top == body_look and "mma_ss(bmm1_s_desc" in body_top, "the two arms must spell the SAME Q.K block (moved verbatim)"
    for s_lookahead in (False, True):
        traced = _traced_mma_body(mma, s_lookahead)
        assert traced.count("mma_ss(bmm1_s_desc") == 2, "exactly one loop Q.K issue site per traced arm, plus the prologue"
        assert traced.count("mma_ss(bmm1_dp_desc") == 1 and traced.count("mma_ts(") == 1
        assert traced.count("bars.mb_s_acc_full.arrive(") == 2 and traced.count("bars.mb_q_empty[q_full_state.idx].arrive(") == 2
        assert traced.count("bars.mb_s_acc_empty[s_acc_empty_state.idx].wait(") == 3, "prologue + loop + the P15 drain"
        assert traced.count("bars.mb_q_full[q_full_state.idx].wait(") == 2
        assert traced.count("s_acc_empty_state = advance(") == 2 and traced.count("q_full_state = advance(") == 2
        assert traced.count(".arrive(mcast_mask=mcast_mask") == 11, "the same 11 commits per traced arm"
        i_qk_loop = traced.index("mma_ss(bmm1_s_desc", traced.index("for q_iter in cutlass.range(q_lo, q_hi"))
        assert (i_qk_loop > traced.index("mma_ss(bmm1_dp_desc")) == s_lookahead


def test_sf_loads_ride_the_full_bars_with_the_grown_tx():
    """Delta 2: five payload + five SF ``tma_load_tile`` sites; every ``_full`` arrive arms payload + SF bytes (34816 / 67584,
    the config's pinned tx) -- a bare payload count completes the phase before the scale factors land (stale SF, wrong dV / dS,
    no crash); K / V SF whole-slab self-multicast, Q / dO / dO_T SF peer-split with ``sf_mcast_mask``."""
    mod = _load()
    assert (mod.qFullTxBytes, mod.dOFullTxBytes, mod.dOTFullTxBytes, mod.kFullTxBytes, mod.vFullTxBytes) == (34816, 34816, 34816, 67584, 67584)
    code = _code_only(_kernel_source())
    ldg = _def_body(code, "_tmaldg_warp")
    assert ldg.count("tma_load_tile(") == 10
    for bare in ("n_bytes=qTmaTransactionBytes", "n_bytes=dOTmaTransactionBytes", "n_bytes=kTmaTransactionBytes", "n_bytes=vTmaTransactionBytes"):
        assert bare not in ldg, f"{bare}: a _full arrive arms the payload bytes only"
    for grown in ("n_bytes=qFullTxBytes", "n_bytes=dOFullTxBytes", "n_bytes=dOTFullTxBytes", "n_bytes=kFullTxBytes", "n_bytes=vFullTxBytes"):
        assert ldg.count(grown) == 1, grown
    assert ldg.count("mcast_mask=sf_mcast_mask") == 3 and ldg.count("mcast_mask=tma_mcast_mask") == 7
    assert "sK_SF[k_empty_state.idx]," in ldg and ".shifted(q_sf_peer_off)" in ldg and ".shifted(do_sf_peer_off)" in ldg and ".shifted(doT_sf_peer_off)" in ldg
    host = _def_body(code, "_host")
    assert host.count("build_rowwise_sf_desc(") == 4 and host.count("build_columnwise_sf_desc(") == 1
    # The K / V (and Q / dO) SF descriptor tile counts are the SF TENSOR's padded extents (SKV / SQ in 128-row atoms), NEVER the real
    # length's: `build_rowwise_sf_desc` derives the head / batch strides from `num_tiles`, so a real-length count misreads every head > 0
    # whenever S_kv % 256 in (0, 128] (MEASURED 2026-09-30 on Rubin, S_kv 800 / H_kv 2: dV cos 0.93; an earlier revision of this pin asserted the opposite).
    kv_line = host.split("kv_sf_tiles =")[1].split("\n")[0]
    sq_line = host.split("sq_sf_tiles =")[1].split("\n")[0]
    assert (
        "num_tiles=kv_sf_tiles" in host and "SKV" in kv_line and "seqlen_kv_real" not in kv_line
    ), f"K / V SF tiles must be the tensor's padded SKV / 128: {kv_line!r}"
    assert (
        "num_tiles=sq_sf_tiles" in host and "SQ" in sq_line and "seqlen_q_real" not in sq_line
    ), f"Q / dO / dO_T SF tiles must be the tensor's padded SQ / 128: {sq_line!r}"


def test_the_q_pad_band_is_a_separate_const_expr_arm_fed_by_seqlen_q_real():
    """``hi = min(hi, seqlen_q_real)`` under ``cutlass.const_expr(CFG.MASK_Q_PAD)`` -- a SEPARATE arm, not a
    MASK_FLAGS bit (a MASK_FLAGS-keyed dispatch folds it out), reachable at MASK_FLAGS == NONE."""
    code = _code_only(_kernel_source())
    mask = _def_body(code, "_mask_p_chunk")
    assert "if cutlass.const_expr(CFG.MASK_Q_PAD):" in mask
    band = mask.split("if cutlass.const_expr(CFG.MASK_Q_PAD):")[1].split("\n")[1]
    assert "hi = seqlen_q_real if hi is None else cute.math.min(hi, seqlen_q_real)" in band, band
    assert "CFG.MASK_FLAGS == MASK_NONE and not CFG.MASK_Q_PAD" in mask
    soft = _def_body(code, "_softmax_warp_group")
    assert "if cutlass.const_expr(CFG.MASK_FLAGS != MASK_NONE or CFG.MASK_Q_PAD):" in soft
    assert "seqlen_q_real=seqlen_q_real" in soft
    assert "band_mask_words(" in mask and "apply_mask_words(" in mask


def test_p_sf_constant_fill_is_proxy_fenced_before_the_init_sync():
    """The constant SF_P atom is a GENERIC store the async-proxy UTCCP reads: its ``fence_proxy`` must precede
    ``fence_mbarrier_init`` / ``barrier_cta_sync`` / ``cga_arrive`` (the forward's order; neither publishes to the async proxy)."""
    body = _def_body(_code_only(_kernel_source()), "_kernel")
    i_fill = body.index("sP_SF_raw.subview(_off).store(cutlass.Int8(CFG.P_SF_BYTE))")
    i_fence = body.index("nvvm.fence_proxy(", i_fill)
    i_init = body.index("nvvm.fence_mbarrier_init()")
    i_cga = body.index("cga_arrive()")
    assert i_fill < i_fence < i_init < i_cga


# =========================================================================== SASS pins: trace-compile the body for sm_107a on ANY box
_SASS_PROBE = textwrap.dedent(r"""
    import glob, os, re, subprocess, sys
    dump, arm, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "cubin,ptx"
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3
    from cudnn.sdpa.bwd.api_dsl import _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm107 import DS_SF_P_B, DS_SF_P_C, TemplateParams
    # Every arm names its dS policy: the P-c arms are the bf16-dS twin, the pb_* arms the shipped block-scaled chain (= the bare record).
    ARMS = {
        "dense_fused": dict(ds_sf_policy=DS_SF_P_C, scaled_fp8_pack=True),
        "dense_fmul": dict(ds_sf_policy=DS_SF_P_C, scaled_fp8_pack=False),
        "causal_fused": dict(ds_sf_policy=DS_SF_P_C, scaled_fp8_pack=True, window_right=0),
        "qpad_fused": dict(ds_sf_policy=DS_SF_P_C, scaled_fp8_pack=True, mask_q_pad=True),
        "pb_dense_fused": dict(ds_sf_policy=DS_SF_P_B, scaled_fp8_pack=True),
        "pb_dense_fmul": dict(ds_sf_policy=DS_SF_P_B, scaled_fp8_pack=False),
        "pb_causal_fused": dict(ds_sf_policy=DS_SF_P_B, scaled_fp8_pack=True, window_right=0),
    }
    mod = load_template(_sm100_kernel_path("sm107/bprop_d256_mxfp8.py"), TemplateParams(dtype_qkv=DTYPE_E4M3, **ARMS[arm]), tag="sdpa_bwd_sm107_main_mxfp8")
    p_packs = (mod._SMX_CHUNK // mod._P_PACK) * 8  # scaled cvts per lane per iteration on the P path (8 per 16-pack)
    dk_packs = (mod._SMX_CHUNK // mod._DS_PACK) * 8 * int(mod._IS_P_B)  # the same 16-pack shape for the ds_dk payload
    dq_cvts = mod._SMX_CHUNK * int(mod._IS_P_B)  # ds_dq: one LONE scaled cvt per element (its column's byte), fp32_to_fp8x4_scaled_pairs
    dq_plain = (mod._SMX_CHUNK // 2) * int(mod._IS_P_B)  # the FMUL arm of the same: one plain pair cvt per two elements
    print("EXPECT_SCALE_BY_C", (p_packs + dk_packs + dq_cvts) if mod.CFG.SCALED_FP8_PACK else 0)  # P + ds_dk + ds_dq take the scaled cvt
    print("EXPECT_PLAIN_E4M3", 0 if mod.CFG.SCALED_FP8_PACK else p_packs + dk_packs + dq_plain)  # the FMUL arm: plain cvts everywhere
    print("EXPECT_CREDUX", mod._SMX_CHUNK * int(mod._IS_P_B))  # one redux.sync.max.abs.f32 per q column per lane per iteration
    print("EXPECT_E8M0_CVT", (1 + mod._SMX_CHUNK // 2) * int(mod._IS_P_B))  # e8m0_pair: 1 for the two dk blocks + e8m0_pair_u: one per dq column pair
    print("EXPECT_UTMASTG", mod.P_TMA_ITERS * mod.CFG.DS_PAYLOADS + mod.CFG.DS_SF_ATOMS + mod.TMA_DV_ITERS)  # payload subtiles + atoms + dV
    print("EXPECT_FMUL_DELTA", mod._SMX_CHUNK * (1 + 2 * int(mod._IS_P_B)))  # fmul arm minus fused arm: one FMUL per P, ds_dk AND ds_dq element
    print("EXPECT_P_ELEMS", mod._SMX_CHUNK)
    print("EXPECT_UTCCP", (mod._SF_ATOMS_K + mod._SF_ATOMS_Q) + (mod._SF_ATOMS_V + mod._SF_ATOMS_dO + mod._SF_ATOMS_K + mod._SF_ATOMS_Q + mod._SF_ATOMS_P + mod._SF_ATOMS_dOT))
    print("EXPECT_S_LOOKAHEAD", int(mod.S_LOOKAHEAD))  # the parent derives EXPECT_COMMITS from the arm this constant traces
    _k_s, _k_dp, _k_dv = mod.CFG.TILE_K // mod.CFG.TILE_K_HW_BMM1, mod.CFG.TILE_O // mod.CFG.TILE_K_HW_BMM1, mod.CFG.TILE_N // mod.CFG.TILE_K_HW_BMM2
    print("EXPECT_MMA", _k_s + _k_dp + _k_s + _k_dv)
    # The tcgen05 stream the MMA warp issues, in program order: the prologue's K/Q copies + the S MMA, then per q iteration the loop's
    # Q.K (K/Q copies + S) and dO.V (V/dO copies + dP) in the order S_LOOKAHEAD selects, then the P/dO_T copies + the dV MMA.
    _loop = [f"KQ:{_k_s}", f"VdO:{_k_dp}"] if not mod.S_LOOKAHEAD else [f"VdO:{_k_dp}", f"KQ:{_k_s}"]
    print("EXPECT_STREAM", " ".join([f"KQ:{_k_s}"] + _loop + [f"PdOT:{_k_dv}"]))
    from cudnn.sdpa.bwd.config_sm107 import SF_TMEM_COLS_PER_ATOM
    _q_cols = {mod.LAYOUT.SF_Q_OFF + a * SF_TMEM_COLS_PER_ATOM for a in range(mod._SF_ATOMS_Q)}
    _do_cols = {mod.LAYOUT.SF_dO_OFF + a * SF_TMEM_COLS_PER_ATOM for a in range(mod._SF_ATOMS_dO)}
    mod.compile(b=1, qh=2, kh=2, sq=1024, skv=1024)
    cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
    if not cubins:
        print("FAIL no cubin dumped into", dump, os.listdir(dump)); sys.exit(3)
    # The BODY's slab placement, measured on the PTX: every slab base of the config's SMEM table (and every descriptor root) must
    # appear as an immediate in the dumped kernel.ptx -- the config test pins the MODEL, this pins what the body compiled to.
    from cudnn.sdpa.bwd.config_sm107 import smem_layout
    ptxs = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    ptx = open(ptxs[-1]).read() if ptxs else ""
    for slab in smem_layout(mod.CFG):
        # Slab BASES only: a ring stage's root is base + idx * stride at run time (no literal), the base is a compile-time immediate.
        n = len(re.findall(r"(?<![0-9])%d(?![0-9])" % slab.offset, ptx)) if slab.offset else -1
        print("PTX_OFFSET", slab.name.replace(" ", "_"), slab.offset, n)
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
    print("SASS USETMAXREG", cnt("USETMAXREG"))
    print("SASS STL", cnt(" STL"))
    print("SASS LDL", cnt(" LDL"))
    print("SASS MEMBAR_GPU", cnt("MEMBAR.ALL.GPU"))
    print("SASS CGAERRBAR", cnt("CGAERRBAR"))
    print("SASS R2P", cnt(" R2P"))
    print("SASS UTCCP", cnt("UTCCP"))
    print("SASS UTCBAR", cnt("UTCBAR"))
    print("SASS MMA", cnt("UTCQMMA") + cnt("UTCHMMA") + cnt("UTCBMMA"))
    print("SASS SCALE_BY_C", cnt("F2FP", "SCALE_BY_C"))
    print("SASS PLAIN_E4M3", sum(1 for ln in sass if "F2FP" in ln and "E4M3" in ln and "SCALE_BY_C" not in ln))
    print("SASS E8M0_CVT", cnt("F2FP", ".E8.F32"))
    print("SASS CREDUX", cnt("CREDUX.MAXABS.F32"))
    print("SASS MOV_R_UR", sum(1 for ln in sass if re.search(r"\bMOV R\d+, UR\d+", ln)))
    print("SASS UTMASTG", cnt("UTMASTG"))
    print("SASS STS_U16", cnt("STS.U16"))
    print("SASS STS_U8", cnt("STS.U8"))
    print("SASS SEL", sum(1 for ln in sass if re.search(r"\bSEL\b", ln)))
    print("SASS FMUL", sum(1 for ln in sass if re.search(r"\bFMUL\b", ln)))
    print("SASS MUFU_EX2", cnt("MUFU.EX2"))
    print("SASS LDTM", cnt("LDTM"))
    print("SASS STTM", cnt("STTM"))
    print("SASS ATOMG", cnt("ATOMG") + cnt("REDG"))
    print("SASS LINES", len(sass))
    # UTCCP groups (consecutive copies, classified by the slot-relative TMEM column they fill: the Q atoms' columns name the K/Q group,
    # the dO atoms' the V/dO group, the rest is P/dO_T) each followed by the UTCQMMA k-steps that read them, in SASS order.
    groups = []
    for ln in sass:
        if "UTCCP" in ln:
            m = re.search(r"tmem\[UR\d+(?:\+0x([0-9a-f]+))?\]", ln)
            col = (int(m.group(1), 16) if m and m.group(1) else 0) % mod.LAYOUT.P_COLS
            if groups and groups[-1][1] == 0:
                groups[-1][0].append(col)
            else:
                groups.append([[col], 0])
        elif "UTCQMMA" in ln and groups:
            groups[-1][1] += 1
    def kind(cols):
        return "KQ" if _q_cols & set(cols) else ("VdO" if _do_cols & set(cols) else "PdOT")
    print("SASS_STREAM", " ".join(f"{kind(c)}:{n}" for c, n in groups))
    """)
_ARMS = ("dense_fused", "dense_fmul", "causal_fused", "qpad_fused")
_ARMS_PB = ("pb_dense_fused", "pb_dense_fmul", "pb_causal_fused")
# MEASURED 2026-09-30 (cutlass-dsl 4.8.0, the internal ptxas / nvdisasm 13.6, host trace-compile): REG 168, 0 / 0 on every P-c arm AND
# every P-b arm (the 232-register softmax budget absorbs the two quantizers: 64 CREDUX + 33 E8M0 cvts + 96 packs per lane per iteration).
_SPILL_PINS = {arm: {"STL": 0, "LDL": 0} for arm in _ARMS + _ARMS_PB}
_SASS_CACHE = {}


def _sass(tmp_path, arm):
    if arm in _SASS_CACHE:
        return _SASS_CACHE[arm]
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    cands = nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    _kernel_source()
    dump = tmp_path / f"sm107a_bwd_mxfp8_{arm}"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _SASS_PROBE, str(dump), arm, *cands], capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"sm_107a trace-compile of the MXFP8 backward ({arm}) failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ") and len(ln.split()) == 3 and ln.split()[2].lstrip("-").isdigit()}
    expect = {ln.split()[0]: int(ln.split()[1]) for ln in out if ln.startswith("EXPECT_") and len(ln.split()) == 2}
    # The commit count of the arm the module traces: the MMA warp spells the loop's Q.K block once per S issue order, one folds out.
    s_lookahead = bool(expect.pop("EXPECT_S_LOOKAHEAD"))
    expect["EXPECT_COMMITS"] = _traced_mma_body(_def_body(_code_only(_kernel_source()), "_mma_warp"), s_lookahead).count(".arrive(mcast_mask=mcast_mask")
    expect["S_LOOKAHEAD"] = int(s_lookahead)
    expect["STREAM_EXPECT"] = next(ln.split(" ", 1)[1] for ln in out if ln.startswith("EXPECT_STREAM "))
    expect["STREAM_SASS"] = next((ln.split(" ", 1)[1] for ln in out if ln.startswith("SASS_STREAM ")), "")
    # PTX_OFFSET <label> <offset> <count>: how many times the slab base / descriptor root appears as a PTX immediate (-1 = offset 0, unpinnable)
    expect["PTX_OFFSETS"] = {(ln.split()[1], int(ln.split()[2])): int(ln.split()[3]) for ln in out if ln.startswith("PTX_OFFSET ") and len(ln.split()) == 4}
    print(f"\nsm107 bwd mxfp8 {arm} sm_107a SASS: {stats}; module says { {k: v for k, v in expect.items() if k != 'PTX_OFFSETS'} }")
    print(f"{arm}: tcgen05 stream {expect['STREAM_SASS']} (S_LOOKAHEAD={expect['S_LOOKAHEAD']}, expected {expect['STREAM_EXPECT']})")
    _SASS_CACHE[arm] = (stats, expect)
    return stats, expect


@pytest.mark.parametrize("arm", _ARMS)
def test_sm107a_register_split_spills_drains_and_no_global_atomics(tmp_path, arm):
    """The 232 / 40 split reached the binary (USETMAXREG > 0, not dropped by C7508), no new stack spills in the 40-register roles
    (the MMA warp carries the K / V / P SF descriptors across the q loop, the per-iteration Q / dO / dO_T ones and the 32-bit alias
    slot base), no GPU-scope drain (every cross-CTA arrive is relaxed or a commit), no global atomic (the row has no amax), and
    exactly ONE TMEM store (the P ring's ``tcgen05_st``, the fp8 body's)."""
    stats, _ = _sass(tmp_path, arm)
    assert stats["USETMAXREG"] > 0, "no USETMAXREG: ptxas dropped the register split (C7508)"
    assert_no_new_spills(stats, _SPILL_PINS[arm], tag=f"{arm}: ")
    assert stats["MEMBAR_GPU"] == 0 and stats["CGAERRBAR"] == 0, "a cluster-scope RELEASE arrive is on a per-tile path (GPU-scope drain)"
    assert stats["ATOMG"] == 0 and stats["STTM"] == 1, "no global atomic (no amax) and ONE TMEM store (the P ring is in TMEM)"


@pytest.mark.parametrize("arm", _ARMS)
def test_sm107a_utccp_and_block_scale_mma_counts_are_the_static_ones(tmp_path, arm):
    """15 ``UTCCP`` = the eight elect-gated atom sites (prologue K 2 + Q 2; per iteration V 2 + dO 2 + K 2 + Q 2 + P 1 + dO_T 2);
    14 block-scale MMAs = the k-steps (Q.K 4 + 4, dO.V 4, P.dO 2 -- the two with a TMEM A operand); 11 ``tcgen05.commit`` (the fp8
    body's 10 + ``mb_p_sf_consumed``); every ``tcgen05.ld`` of S / dP / dV is there (LDTM >= 4)."""
    stats, expect = _sass(tmp_path, arm)
    assert stats["UTCCP"] == expect["EXPECT_UTCCP"] == 15, stats
    assert stats["MMA"] == expect["EXPECT_MMA"] == 14, stats
    assert stats["UTCBAR"] == expect["EXPECT_COMMITS"] == 11, stats
    assert stats["LDTM"] >= 4 and stats["MUFU_EX2"] == expect["EXPECT_P_ELEMS"], stats


@pytest.mark.parametrize("arm", _ARMS + _ARMS_PB)
def test_sm107a_tcgen05_stream_order_follows_the_mask_arm(tmp_path, arm):
    """The MMA warp's tcgen05 stream in program order -- UTCCP groups named by the slot-relative TMEM columns they fill, each followed by
    the k-steps of the MMA that reads them: the prologue's K/Q + S first in every arm, then per q iteration K/Q + S AHEAD of V/dO + dP
    on the dense arms (``S_LOOKAHEAD`` False) and V/dO + dP ahead of K/Q + S on the masked and q-pad arms (True, the lookahead), the
    P/dO_T copies + the dV MMA last.  The counts (15 UTCCP, 14 MMAs) are the same under both orders; only the order moves."""
    _, expect = _sass(tmp_path, arm)
    assert expect["STREAM_SASS"], f"{arm}: no UTCCP / UTCQMMA stream parsed from the SASS"
    assert expect["STREAM_SASS"] == expect["STREAM_EXPECT"], (arm, expect["S_LOOKAHEAD"], expect["STREAM_SASS"], expect["STREAM_EXPECT"])
    assert expect["S_LOOKAHEAD"] == int(arm.split("_")[-2] not in ("dense",)), f"{arm}: the dense arms take the top order, the rest the lookahead"


def test_sm107a_fused_p_quantizer_is_the_scaled_cvt_and_the_fmul_arm_pays_one_fmul_per_element(tmp_path):
    """With ``scaled_fp8_pack`` the P quantizer is 8 ``F2FP...SCALE_BY_C`` per 16-pack (32 per lane
    per q iteration) and NO multiply; the portable arm has 0 scaled cvts and exactly one ``FMUL`` more per P element (the
    ``x * 2^8`` by an opaque 256.0) -- the saving the fused form buys on Rubin."""
    fused, ef = _sass(tmp_path, "dense_fused")
    fmul, em = _sass(tmp_path, "dense_fmul")
    assert fused["SCALE_BY_C"] == ef["EXPECT_SCALE_BY_C"] == 32, fused
    assert fmul["SCALE_BY_C"] == 0 == em["EXPECT_SCALE_BY_C"], fmul
    assert fmul["FMUL"] - fused["FMUL"] == ef["EXPECT_P_ELEMS"] == 64, (fused["FMUL"], fmul["FMUL"])


@pytest.mark.parametrize("arm", ("causal_fused", "qpad_fused"))
def test_sm107a_masked_and_q_pad_arms_lower_to_the_bit_word_form(tmp_path, arm):
    """``R2P > 0`` = the bit-word mask (rules/frost-tile-dsl.md s10d); the MASK_Q_PAD arm must reach it at MASK_FLAGS == NONE."""
    stats, _ = _sass(tmp_path, arm)
    assert stats["R2P"] > 0, f"{arm}: no R2P -- the mask arm is the per-cell compare + select form, or the q band folded out"


def test_sm107a_dense_arm_carries_no_mask_ir(tmp_path):
    """The dense specialization without the q band emits no mask IR at all (a leaked band would show as R2P)."""
    stats, _ = _sass(tmp_path, "dense_fused")
    assert stats["R2P"] == 0, stats


@pytest.mark.parametrize("arm", _ARMS_PB)
def test_sm107a_p_b_arms_keep_the_split_spill_free_drain_free_and_the_mma_side_untouched(tmp_path, arm):
    """The P-b quantizers live in the 232-register softmax warps: the split still reaches the binary, no new stack spills, no GPU-scope
    drain, no atomic, exactly ONE TMEM store (the P ring's ``tcgen05_st``); the MMA side (15 UTCCP into the alias slot, 14 block-scale
    MMAs, 11 commits incl. ``mb_p_sf_consumed``, 4 LDTM) is the P-c body's -- P-b touches only the dsoftmax tail and the TMASTG."""
    stats, expect = _sass(tmp_path, arm)
    assert stats["USETMAXREG"] > 0, "no USETMAXREG: ptxas dropped the register split (C7508)"
    assert_no_new_spills(stats, _SPILL_PINS[arm], tag=f"{arm}: ")
    assert stats["MEMBAR_GPU"] == 0 and stats["CGAERRBAR"] == 0, "a cluster-scope RELEASE arrive is on a per-tile path (GPU-scope drain)"
    assert stats["ATOMG"] == 0 and stats["STTM"] == 1
    assert stats["UTCCP"] == expect["EXPECT_UTCCP"] == 15 and stats["MMA"] == expect["EXPECT_MMA"] == 14 and stats["LDTM"] >= 4
    assert stats["UTCBAR"] == expect["EXPECT_COMMITS"] == 11, stats
    assert stats["MUFU_EX2"] == expect["EXPECT_P_ELEMS"] == 64


@pytest.mark.parametrize("arm", _ARMS_PB)
def test_sm107a_p_b_census_one_redux_per_column_two_quantizers_four_stores(tmp_path, arm):
    """The static census of the P-b dsoftmax tail, per lane per q iteration: 64 ``CREDUX.MAXABS.F32`` (one ``redux.sync.max.abs.f32``
    per q column = the warp's 32-kv block), 33 ``F2FP...E8.F32...RP`` (the ``e8m0_pair`` cvt for a lane's two dk blocks + one
    ``e8m0_pair_u`` per dq column pair), the scaled cvt on P, ds_dk AND ds_dq under ``scaled_fp8_pack`` (32 + 32 + 64 lone cvts =
    128 ``SCALE_BY_C``, no plain e4m3 cvt at all; the FMUL arm: 0 scaled, 96 plain = P + dk + the dq pair cvts), 8 ``UTMASTG``
    (ds_dk + ds_dq + the two atoms + 4 dV subtiles; the P-c body has 6), a lane's ``STS.U16`` (its two dk bytes) and two ``STS.U8``
    (its dq columns 2l, 2l+1; the constant P-SF fill adds its own), the pair select tree (``SEL``).  MOV R,UR is REPORTED, not
    pinned: a CREDUX result lands in a uniform register and sm_107a ptxas moves it into the vector file before ANY ALU op reads it
    (13 spellings tried, every one 64 moves -- the floor; the P-c body carries 141 of its own)."""
    stats, expect = _sass(tmp_path, arm)
    assert stats["CREDUX"] == expect["EXPECT_CREDUX"] == 64, stats
    assert stats["E8M0_CVT"] == expect["EXPECT_E8M0_CVT"] == 33, stats
    assert stats["SCALE_BY_C"] == expect["EXPECT_SCALE_BY_C"] == (128 if arm != "pb_dense_fmul" else 0), stats
    assert stats["PLAIN_E4M3"] == expect["EXPECT_PLAIN_E4M3"] == (0 if arm != "pb_dense_fmul" else 96), stats
    assert stats["UTMASTG"] == expect["EXPECT_UTMASTG"] == 8, stats
    assert stats["STS_U16"] == 1 and stats["STS_U8"] >= 2 and stats["SEL"] >= 1, stats
    print(f"\n{arm}: MOV R,UR = {stats['MOV_R_UR']} (P-c dense_fused carries 141; +1 per CREDUX, the uniform-register floor), SASS lines {stats['LINES']}")


def test_sm107a_p_b_fmul_arm_pays_one_fmul_per_p_ds_dk_and_ds_dq_element(tmp_path):
    """The portable arm (``scaled_fp8_pack=False``) has 0 scaled cvts and exactly one scalar ``FMUL`` more per P element, per ds_dk
    element AND per ds_dq element (the ``x * 2^(127 - e)`` the fused cvt folds: 3 x 64 = 192); the dq SCALE multiplies are packed
    ``FMUL2`` on both arms (one per column pair, not counted here).  The redux count does not move."""
    fused, ef = _sass(tmp_path, "pb_dense_fused")
    fmul, em = _sass(tmp_path, "pb_dense_fmul")
    assert fused["SCALE_BY_C"] == 128 and fmul["SCALE_BY_C"] == 0
    assert fmul["FMUL"] - fused["FMUL"] == ef["EXPECT_FMUL_DELTA"] == em["EXPECT_FMUL_DELTA"] == 192, (fused["FMUL"], fmul["FMUL"])
    assert fused["CREDUX"] == fmul["CREDUX"] == 64


def test_sm107a_p_b_dense_arm_carries_no_mask_ir_beyond_the_lane_bit_predicates(tmp_path):
    """The dense P-b arm emits no mask words: the only ``R2P`` is the one that turns a lane id's five bits into the pair select
    tree's predicates (``_select_by_lane``); the causal P-b arm adds the bit-word mask's own."""
    dense, _ = _sass(tmp_path, "pb_dense_fused")
    causal, _ = _sass(tmp_path, "pb_causal_fused")
    assert dense["R2P"] <= 1, dense
    assert causal["R2P"] > dense["R2P"], (dense["R2P"], causal["R2P"])


@pytest.mark.parametrize("arm, n_slabs", [("dense_fused", 12), ("pb_dense_fused", 14)], ids=["P-c", "P-b"])
def test_sm107a_ptx_carries_the_config_slab_offsets(tmp_path, arm, n_slabs):
    """The BODY lands its slabs where ``config_sm107.smem_layout(FAMILY_MXFP8)`` says (the docstring's SMEM table): every non-zero slab
    BASE of the config's table appears as an immediate in the sm_107a PTX of the dense arm (a ring stage's descriptor root is base +
    idx * stride at run time -- arithmetic, not a literal -- so the bases are what the PTX can pin).  The config
    test pins the model; this pins the compiled body against it -- a slab declared out of order (or a root past the 256 KiB line under a
    version-0 descriptor) moves an immediate and fails here before it silently copies DATA into the SF columns.  MEASURED 2026-09-30: the
    same pin read its immediates off the SMEM-P-ring predecessor's PTX (147456, 212992, 215040, ..., 227328, 260096); with the
    P ring back in TMEM the P-c table has 12 slabs and sdS sits at 227328, the P-b table 14 (sdS_SF 227328 | sdS 229376 | sdS_kv 262144)
    (host trace-compile: REG 168, UTCCP 15, 14 block-scale MMAs, STTM 1, 11 commits, 0 CGAERRBAR / MEMBAR.ALL.GPU on every arm)."""
    _stats, expect = _sass(tmp_path, arm)
    offsets = expect["PTX_OFFSETS"]
    assert offsets, "the probe printed no PTX_OFFSET rows (no .ptx dumped?) -- it is pinning nothing"
    pinned = {k: n for k, n in offsets.items() if k[1] > 0}
    assert (
        len(offsets) == n_slabs and len(pinned) == n_slabs - 1
    ), f"expected the {n_slabs - 1} non-zero slab bases of the {n_slabs}-slab table (sQ sits at 0); got {sorted(pinned)}"
    if arm == "pb_dense_fused":
        assert {k[0] for k in offsets} >= {"sdS_SF", "sdS", "sdS_kv"} and ("sdS_SF", 222 * 1024) in offsets and ("sdS_kv", 256 * 1024) in offsets
    missing = sorted(k for k, n in pinned.items() if n <= 0)
    assert not missing, f"slab bases of the config table absent from the PTX immediates (a slab moved, or a base is no longer a literal): {missing}"


_DEQUANT_PTX_PROBE = textwrap.dedent(r"""
    import glob, os, re, sys
    dump = sys.argv[1]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "ptx"
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import cutlass, cutlass.cute as cute
    from cuda.bindings import driver
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import dequant_mxfp8_to_bf16_host
    fp8, half, u8 = cutlass.Float8E4M3FN, cutlass.BFloat16, cutlass.Uint8
    def ptr(dt):
        return cute.runtime.make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)
    B, S, H, D = 1, 256, 2, 256
    @cute.jit
    def host(src_p: cute.Pointer, sf_p: cute.Pointer, dst_p: cute.Pointer, columnwise: cutlass.Constexpr, stream: driver.CUstream):
        shape = (B, S, H, D)
        strides = (S * H * D, H * D, D, 1)
        src = cute.make_tensor(src_p, cute.make_layout(shape, stride=strides))
        dst = cute.make_tensor(dst_p, cute.make_layout(shape, stride=strides))
        n = B * H * S * D // 32
        sf = cute.make_tensor(sf_p, cute.make_layout((n,), stride=(1,)))
        dequant_mxfp8_to_bf16_host(src, sf, dst, columnwise, stream)
    for columnwise in (True, False):
        cute.compile(host, ptr(fp8), ptr(u8), ptr(half), columnwise, cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False), options="--gpu-arch sm_107a")
    ptxs = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    if not ptxs:
        print("FAIL no ptx dumped into", dump, os.listdir(dump)); sys.exit(3)
    for f in ptxs:
        txt = open(f).read()
        for m in re.finditer(r"\.entry\s+(\w+)", txt):
            name = m.group(1)
            body = txt[m.end():]
            nxt = body.find(".entry")
            body = body if nxt < 0 else body[:nxt]
            if "dequant" not in name:
                continue
            # .s8 sign-extends into the 32-bit register; .u8 / .b8 zero-extend (PTX ISA ld: 8-bit values are extended to the register
            # width, sign- or zero- per the type).  A masked Int8 load lowers to .b8 (the `and` folds into the load).
            s8 = len(re.findall(r"ld\.global\.(?:nc\.)?s8\b", body))
            zx = len(re.findall(r"ld\.global\.(?:nc\.)?[ub]8\b", body))
            and255 = len(re.findall(r"and\.b32\s+%r\d+,\s+%r\d+,\s+255;", body))
            print("KERNEL", name, "LD_S8", s8, "LD_ZX8", zx, "AND255", and255)
    """)


def test_sm107a_dequant_ptx_loads_the_sf_byte_unsigned(tmp_path):
    """The PTX form of the review blocker (2026-09-30: ``ld.global.s8`` straight into ``shl.b32 23``, no mask): in the sm_107a PTX
    of BOTH dequant specializations every byte load is zero-extending (``ld.global.b8`` / ``.u8`` -- the ``& 0xFF`` folds into the
    load) or an ``s8`` load is followed by an ``and.b32 255``; an ``s8`` load with no mask is the sign-extension bug (byte >= 128: right
    magnitude, flipped sign).  MEASURED after the fix: ``ld.global.b8`` into ``shl.b32 23``, 0 ``s8``."""
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    dump = tmp_path / "sm107a_dequant_ptx"
    dump.mkdir()
    # A FILE, not `python -c`: the DSL needs inspect.getsource of the @cute.jit host (UNSUP_NO_SOURCE otherwise).
    probe = tmp_path / "dequant_ptx_probe.py"
    probe.write_text(_DEQUANT_PTX_PROBE, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(probe), str(dump)], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"sm_107a trace-compile of the dequant host failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    rows = [ln.split() for ln in proc.stdout.splitlines() if ln.startswith("KERNEL ")]
    assert len(rows) >= 2, f"expected the columnwise and the rowwise dequant kernels in the PTX; got {proc.stdout[-2000:]}"
    for row in rows:
        stats = {row[i]: int(row[i + 1]) for i in range(2, len(row), 2)}
        print(f"\n{row[1]}: {stats}")
        assert stats["LD_S8"] + stats["LD_ZX8"] > 0, f"{row[1]}: no byte load found -- the probe is pinning nothing"
        assert stats["LD_S8"] == 0 or stats["AND255"] >= stats["LD_S8"], f"{row[1]}: a sign-extended SF byte load reaches the shift unmasked: {stats}"


# =========================================================================== the ENGINE ROW -- registration / capabilities (host)
_E4M3 = cudnn.data_type.FP8_E4M3
_E5M2 = cudnn.data_type.FP8_E5M2
_BF16 = cudnn.data_type.BFLOAT16
_T_E4M3 = torch.float8_e4m3fn


def _spec(name=_ENGINE):
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == name), None)
    assert spec is not None, f"{name} is not in cudnn.sdpa.bwd.engines.ENGINE_SPECS"
    return spec


def _mxfp8_graph(
    b=1,
    hq=2,
    hkv=None,
    sq=256,
    skv=None,
    *,
    out_dt=torch.bfloat16,
    scale="default",
    causal=False,
    bottom_right=False,
    left_bound=None,
    declare_amax=False,
    fp8=_E4M3,
    with_sink=False,
    padded=False,
    deterministic=False,
    bhsd=False,
):
    """An ``sdpa_mxfp8_backward`` graph over BSHD-physical tensors (the SM100 suite's builder: 17 inputs incl. the seven F8_128x4 SF
    tensors, dQ / dK / dV in ``out_dt``, the amax ports virtual unless ``declare_amax``).  Returns ``(graph, tensors, outputs)``."""
    from test_sdpa_bwd_mxfp8_sm100 import _build_graph, _bshd_stride

    hkv = hq if hkv is None else hkv
    skv = sq if skv is None else skv
    kw = {}
    if causal and not bottom_right:
        kw["use_causal_mask"] = True
    if bottom_right:
        kw["use_causal_mask_bottom_right"] = True
    if left_bound is not None:
        kw["left_bound"] = left_bound
    if deterministic:
        kw["use_deterministic_algorithm"] = True
    if padded:
        kw["use_padding_mask"] = True

    def _bhsd(shape):
        bb, h, s, d = shape
        return [h * s * d, s * d, d, 1]

    return _build_graph(
        b,
        hq,
        hkv,
        sq,
        skv,
        _D,
        out_dt=out_dt,
        scale=(1.0 / math.sqrt(_D) if scale == "default" else scale),
        fp8=fp8,
        stride_fn=_bhsd if bhsd else _bshd_stride,
        declare_amax=declare_amax,
        with_sink=with_sink,
        seq_len_dims=((b, 1, 1, 1) if padded else None),
        **kw,
    )


def _decline_reason(monkeypatch, engine=_ENGINE, cc=_RUBIN_CC, **kw):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    spec = _spec(engine)
    try:
        g, _t, _outs = _mxfp8_graph(**kw)
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as e:
        return f"frontend refused the graph: {e}"
    facts = ga.analyze(g)
    if facts is None:
        return "analyzer did not recognise the graph"
    return mismatch(spec.capabilities, facts)


def test_row_is_registered_and_opt_in():
    from cudnn.engines.engine_ids import FROST_SDPA_BWD_ID_BASE
    from cudnn.engines.manifest import MANIFEST

    _spec()
    fam = next(f for f in MANIFEST if f.name == _FAMILY_NAME)
    assert _ENGINE in fam.slots, f"{_ENGINE} has no manifest slot (append slot {_SLOT}; never reuse or renumber)"
    slot = fam.slots[_ENGINE]
    assert slot.slot == _SLOT and slot.opt_in, "new engines stay opt-in until they earn arch coverage + benchmarks"
    assert FROST_SDPA_BWD_ID_BASE + slot.slot == _ENGINE_ID
    # The slot is appended: the sibling rows keep theirs.
    assert fam.slots["sdpa_bwd_sm107"].slot == 4 and fam.slots["sdpa_bwd_sm107_fp8"].slot == 5


def test_row_capabilities_match_what_is_implemented():
    """The row claims exactly the contract the adapter implements: E4M3 payloads, bf16 gradients (fp16 = a follow-up), no amax
    (a requesting graph is declined, typed), the fp8 row's mask set incl. bottom-right at S_q % 128 == 0, GQA; the v1 deferrals False."""
    c = _spec().capabilities
    assert (c.sm_lo, c.sm_hi) == _SM_RANGE
    assert c.d == frozenset({_D}) and not c.d_envelope and not c.dqk_ge_dv
    assert c.is_mxfp8 and not c.is_fp8, "block-scale MXFP8 (sdpa_mxfp8_backward), not per-tensor"
    assert c.dtypes == frozenset({_E4M3}), "E4M3 only: the body has no E5M2 arm"
    assert c.out_dtypes == frozenset({_BF16}), "bf16 gradients only: the P-c chain's bf16 stage-3 renderings store their io dtype (fp16 is a follow-up)"
    assert not c.amax_dgrad, "no amax in the MXFP8 row: a requesting graph is declined, typed"
    assert c.causal and c.bottom_right and c.swa and c.gqa
    assert c.bottom_right_s_q_multiple == 128, "bottom-right is claimed for S_q % 128 == 0 only (the body derives the diagonal from its padded S_q)"
    assert not c.right_band_widening and not c.thd and not c.thd_declared_totals and not c.cu_seq_len
    assert not c.bias and not c.dbias and not c.decode
    assert c.layouts == frozenset({"bshd"})
    assert not c.tile_ms and not c.tile_ns, "the sm107 rows have no tile axis ({} is the complete record)"
    for deferred in ("padded", "sink", "dsink", "deterministic"):
        assert not getattr(c, deferred), f"{deferred} is deferred: claim it together with its accept test here and the tracker line"


def test_row_ships_the_p_b_chain_and_the_sf_pad_staging():
    """The shipped dS policy is P-b (``config_sm107.DS_SF_POLICY_DEFAULT``, flipped once the block-scaled chain beat the bf16-dS
    twin on Rubin) and the adapter's own constant ``MXFP8_DS_SF_POLICY`` follows it (P-c stays a built, selectable arm); the prepared
    hosts default to the same constant; every sm107 row -- this one under both dS policies included -- chunks its stage-2 workspace
    against the ONE 8 GiB budget (``_SM107_WS_BUDGET_BYTES``; the block-scaled chain's own constant is gone); the SF pad staging ON."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107
    from cudnn.sdpa.bwd import config_sm107 as cfg
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host

    assert cfg.DS_SF_POLICY_DEFAULT == cfg.DS_SF_P_B
    assert sm107.MXFP8_DS_SF_POLICY == cfg.DS_SF_POLICY_DEFAULT == cfg.DS_SF_P_B, "the adapter reads the config's default policy"
    assert inspect.signature(prepared_host.compile_host_mxfp8).parameters["ds_sf_policy"].default == cfg.DS_SF_POLICY_DEFAULT
    assert sm107._SM107_WS_BUDGET_BYTES == 8 << 30 and not hasattr(sm107, "_SM107_MXFP8_BLOCK_SCALED_WS_BUDGET_BYTES"), "one budget, one name"
    assert prepared_host.MXFP8_STAGE_SF_PADS is True
    assert sm107._SM107_KERNEL_FILES[cfg.FAMILY_MXFP8] == _KERNEL_FILE and sm107._SM107_TEMPLATE_TAGS[cfg.FAMILY_MXFP8] == _TAG
    assert sm107.SdpaBwdDslSm107Mxfp8._FAMILY == cfg.FAMILY_MXFP8 and sm107.SdpaBwdDslSm107Mxfp8._NAME == _ENGINE


def test_sf_pad_staging_geometry_derives_from_the_config():
    """Derive, never re-literal (review 2026-09-30): the prepared host's F8_128x4 atom geometry IS ``config_sm107``'s, and its
    atoms-per-tile count is derived from D -- the columnwise D-plane count ``D // 128`` and the rowwise 4-group d-chunk count
    ``(D // 32) // 4`` coincide (an atom is 128 x 128 elements either way); this pin replaces a module-level assert."""
    from cudnn.sdpa.bwd import config_sm107 as cfg
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host as ph

    assert ph.SF_ATOM_BYTES is cfg.SF_ATOM_BYTES and ph.SF_ATOM_ROWS is cfg.SF_ATOM_ROWS and ph.MX_BLOCK is cfg.MX_BLOCK
    assert not hasattr(ph, "_SF_ATOM_BYTES") and not hasattr(ph, "_SF_ATOM_ROWS"), "the re-literaled atom constants are gone"
    assert ph._SF_ATOMS_PER_TILE == ph._D // cfg.SF_ATOM_ROWS == (ph._D // cfg.MX_BLOCK) // 4 == 2
    code = _code_only(open(ph.__file__, encoding="utf-8").read())
    for name in ("_pad_sf_atoms", "_pad_sf", "_dequant_mxfp8_to_bf16"):
        body = _def_body(code, name)
        assert "* 2 *" not in body and "// 128" not in body and "* 1024" not in body and "* 512" not in body, f"{name}: a geometry literal survived"


def test_dk_dq_recipe_is_the_bf16_rows():
    from test_sdpa_bwd_dsl_sm107 import _TOL

    assert _BF16_GRAD_TOL == _TOL[torch.bfloat16], "dK / dQ are gated with the bf16 row's recipe, kept equal by this pin"


def test_dequant_host_reads_the_e8m0_byte_unsigned():
    """Source pin of the review blocker (2026-09-30): every SF byte that reaches ``_e8m0_scale`` passes through ``_sf_byte``, which
    masks the Int8 ``raw_ptr`` load to 0..255; no bare ``.load().to(cutlass.Int32)`` of an SF byte survives in the dequant body."""
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host as ph

    code = _code_only(open(ph.__file__, encoding="utf-8").read())
    deq = _def_body(code, "_dequant_mxfp8_to_bf16")
    assert deq.count("_e8m0_scale(_sf_byte(") == 2 and "_e8m0_scale((sf_ptr" not in deq, "an SF byte reaches _e8m0_scale without the unsigned mask"
    byte = _def_body(code, "_sf_byte")
    assert "& cutlass.Int32(0xFF)" in byte or "& cutlass.Int32(255)" in byte


# =========================================================================== REJECT -- the row's mismatch() on REAL graphs (host, fake cc 10.7)


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(causal=True),
        dict(causal=True, bottom_right=True),
        dict(causal=True, bottom_right=True, sq=512, skv=1024),
        dict(causal=True, bottom_right=True, sq=512, skv=1000),
        dict(causal=True, left_bound=64),
        dict(left_bound=64),
        dict(hq=8, hkv=1),
        dict(hq=8, hkv=2),
        dict(sq=129, skv=300),
        dict(causal=True, sq=500, skv=500),
        dict(scale=None),
        dict(b=3, sq=128, skv=256),
    ],
    ids=[
        "dense",
        "causal",
        "bottom-right",
        "bottom-right-rect",
        "bottom-right-ragged-kv",
        "swa",
        "swa-no-causal",
        "mqa",
        "gqa-r4",
        "non-tile-S",
        "causal-non-tile-S",
        "default-scale",
        "b3-nq1",
    ],
)
def test_served_graph_passes_the_row_probe(monkeypatch, kw):
    assert _decline_reason(monkeypatch, **kw) is None


def test_reject_amax_outputs(monkeypatch):
    """The backend's canonical MXFP8 backward graph declares amax_dQ / dK / dV as real outputs; this row produces none -- a
    typed decline naming them, never garbage in the amax ports."""
    reason = _decline_reason(monkeypatch, declare_amax=True)
    assert reason is not None and "amax" in reason, reason


def test_reject_fp16_gradients(monkeypatch):
    reason = _decline_reason(monkeypatch, out_dt=torch.float16)
    assert reason is not None and "HALF" in reason and "BFLOAT16" in reason, reason


def test_reject_e5m2_payloads(monkeypatch):
    reason = _decline_reason(monkeypatch, fp8=_E5M2)
    assert reason is not None and "dtype" in reason, reason


@pytest.mark.parametrize("sq,skv", [(500, 1024), (300, 1000), (129, 256)], ids=["500x1024", "300x1000", "129x256"])
def test_reject_bottom_right_with_ragged_s_q(monkeypatch, sq, skv):
    reason = _decline_reason(monkeypatch, causal=True, bottom_right=True, sq=sq, skv=skv)
    assert reason is not None and "S_q % 128" in reason, reason
    # Top-left causal at the same ragged S_q is served (the diagonal is 0; the q pad rows read P = 0 through the +inf LSE + the q band).
    assert _decline_reason(monkeypatch, causal=True, sq=sq, skv=skv) is None


def test_reject_padding_mask(monkeypatch):
    reason = _decline_reason(monkeypatch, padded=True)
    assert reason is not None and "padding" in reason, reason


def test_reject_sink(monkeypatch):
    reason = _decline_reason(monkeypatch, with_sink=True)
    assert reason is not None and "sink" in reason.lower(), reason


def test_reject_non_bshd_layout(monkeypatch):
    reason = _decline_reason(monkeypatch, bhsd=True)
    assert reason is not None and "BSHD" in reason, reason


def test_deterministic_declines_with_the_corrected_reason(monkeypatch):
    """``use_deterministic_algorithm`` is declined (the claim waits on the two-run sweep) -- and the SHARED reason no longer blames
    fp32 atomics, which none of the sm107 chains have (the shared reason text was corrected when this row landed)."""
    reason = _decline_reason(monkeypatch, deterministic=True)
    if _spec().capabilities.deterministic:
        assert reason is None, reason
    else:
        assert reason is not None and "two-run bitwise" in reason and "atomics" not in reason, reason


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (12, 0), (9, 0)], ids=["sm100", "sm103", "sm120", "sm90"])
def test_reject_other_arch_lines(monkeypatch, cc):
    reason = _decline_reason(monkeypatch, cc=cc)
    assert reason is not None and f"SM{_SM_RANGE[0]}-{_SM_RANGE[1]}" in reason, reason


@pytest.mark.parametrize("dsl_version", ["4.7.0", "0.3.0+internal"])
def test_reject_missing_sm107a_dsl_target_before_compile(monkeypatch, dsl_version):
    """AGENTS.md Rule 7 (review 2026-09-30): a cc 10.7 box on a DSL that does not know ``sm_107a`` (4.7.x is above FROST's floor
    yet below Rubin's need) gets a TYPED decline at eligibility from every sm107 backward row -- never ``KeyError: 'sm_107a'``
    out of ``Arch.from_string`` inside the template compile after the plan is ranked (fwd/engines.py has had this gate; the
    bwd copy lacked it).  The SM100 MXFP8 row on a cc 10.0 device is untouched."""
    from cudnn.frost import buffers

    dist = "nvidia-cutlass-dsl" if dsl_version == "4.7.0" else "nvidia-cutlass-dsl-internal"
    monkeypatch.setattr(buffers, "_DSL_STATE", (True, (dist, dsl_version)))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    for engine in (_ENGINE, "sdpa_bwd_sm107", "sdpa_bwd_sm107_fp8"):
        reason = _decline_reason(monkeypatch, engine=engine)
        assert reason is not None and "sm_107a" in reason and dsl_version in reason, (engine, reason)
    assert _decline_reason(monkeypatch, engine="sdpa_bwd_sm100_mxfp8", cc=(10, 0)) is None


def _facts_of_served_graph(monkeypatch):
    from cudnn.sdpa import graph_analyzer as ga

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    g, _t, _outs = _mxfp8_graph()
    facts = ga.analyze(g)
    assert facts is not None and facts.is_mxfp8
    return facts


@pytest.mark.parametrize(
    "field,needle",
    [
        ("uniform_dtype", "share Q's dtype"),
        ("uniform_out_dtype", "must match"),
        ("bshd_layout", "BSHD-physical"),
        ("thd", None),
        ("has_bias", "bias"),
        ("has_dbias", "dBias"),
        ("right_band_widening", "right-band"),
    ],
    ids=["mixed-payload-dtypes", "mixed-half-dtypes", "dense_flex-strides", "thd", "bias", "dbias", "right-band-widening"],
)
def test_reject_the_rest_of_the_design_list(monkeypatch, field, needle):
    """The host REJECT list beyond what the graph builder can spell: the analyzer's frozen facts of a SERVED graph
    with ONE fact flipped (``dataclasses.replace``) must decline, typed, through the row's own ``mismatch()``."""
    import dataclasses

    from cudnn.sdpa.bwd.engines import mismatch

    facts = _facts_of_served_graph(monkeypatch)
    caps = _spec().capabilities
    assert mismatch(caps, facts) is None
    flipped = dataclasses.replace(facts, **{field: not getattr(facts, field)})
    reason = mismatch(caps, flipped)
    assert reason is not None, field
    if needle is not None:
        assert needle in reason, (field, reason)


def test_reject_tile_knob_requests(monkeypatch):
    """The sm107 rows are knobless: a ``tile_m`` / ``tile_n`` request is outside the (empty) domain -> ineligible, never degraded."""
    from cudnn.sdpa.bwd.engines import SdpaBwdKnobs, mismatch

    caps = _spec().capabilities
    facts = _facts_of_served_graph(monkeypatch)
    for req in (SdpaBwdKnobs(tile_m=64), SdpaBwdKnobs(tile_n=128)):
        reason = mismatch(caps, facts, req)
        assert reason is not None and "outside this engine's domain" in reason, reason


def test_every_other_row_declines_this_graph(monkeypatch):
    """On a cc 10.7 device the d256 ``sdpa_mxfp8_backward`` graph is served by THIS row alone: the SM100 MXFP8 row (sm 100-106),
    the half / per-tensor sm107 rows ("serves only") and every other arch's row decline it typed (engine-contract 8b')."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    g, _t, _outs = _mxfp8_graph()
    facts = ga.analyze(g)
    assert facts is not None and facts.is_mxfp8
    verdicts = {s.name: mismatch(s.capabilities, facts) for s in ENGINE_SPECS}
    assert verdicts.pop(_ENGINE) is None
    assert all(v is not None for v in verdicts.values()), verdicts
    assert "SM100-106" in verdicts["sdpa_bwd_sm100_mxfp8"] or "requires" in verdicts["sdpa_bwd_sm100_mxfp8"], verdicts["sdpa_bwd_sm100_mxfp8"]


def test_reject_foreign_quantization_families(monkeypatch):
    """This row declines the half ``sdpa_backward`` and the per-tensor ``sdpa_fp8_backward`` graphs (the family gate)."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph
    from test_sdpa_bwd_fp8_sm107 import _build_graph as _fp8_graph

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    caps = _spec().capabilities
    g, _t, _outs = _half_bwd_graph()
    facts = ga.analyze(g)
    assert facts is not None and not facts.is_mxfp8
    assert "serves only" in (mismatch(caps, facts) or "")
    facts = ga.analyze(_fp8_graph())
    assert facts is not None and facts.is_fp8
    assert "serves only" in (mismatch(caps, facts) or "")


# --------------------------------------------------------------------------- the adapter backstops (host, no GPU)


def _ceil128(n):
    return -(-n // 128) * 128


def _mxfp8_adapter(b=1, hq=2, hkv=None, sq=256, skv=None, *, out_dt=torch.bfloat16, sf_over=None, p_scale_log2=8, **kw):
    """``SdpaBwdDslSm107Mxfp8`` from TensorDescs (no device memory): BSHD-physical payloads, the F8_128x4 SF dims of the graph
    (``sf_over`` replaces one SF tensor's shape), ``o_f16``-side dtype ``out_dt``."""
    from cudnn.api_base import TensorDesc
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Mxfp8

    hkv = hq if hkv is None else hkv
    skv = sq if skv is None else skv
    dev = torch.device("cuda", 0)

    def bshd(h, s, dt, name):
        shape, stride = (b, h, s, _D), (s * h * _D, _D, h * _D, 1)
        return TensorDesc(dtype=dt, shape=shape, stride=stride, stride_order=TensorDesc._compute_stride_order(shape, stride), device=dev, name=name)

    def dense(shape, dt, name):
        stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        return TensorDesc(dtype=dt, shape=shape, stride=stride, stride_order=TensorDesc._compute_stride_order(shape, stride), device=dev, name=name)

    row_q, col_q = (b, hq, _ceil128(sq), 8), (b, hq, _ceil128(sq) // 32, 256)
    row_k, col_k = (b, hkv, _ceil128(skv), 8), (b, hkv, _ceil128(skv) // 32, 256)
    sf_shapes = dict(sf_q=row_q, sf_q_T=col_q, sf_k=row_k, sf_k_T=col_k, sf_v=row_k, sf_do=row_q, sf_do_T=col_q)
    sf_shapes.update(sf_over or {})
    return SdpaBwdDslSm107Mxfp8(
        bshd(hq, sq, _T_E4M3, "q"),
        bshd(hkv, skv, _T_E4M3, "k"),
        bshd(hkv, skv, _T_E4M3, "v"),
        bshd(hq, sq, out_dt, "o"),
        bshd(hq, sq, _T_E4M3, "dO"),
        dense((b, hq, sq, 1), torch.float32, "stats"),
        bshd(hq, sq, out_dt, "dQ"),
        bshd(hkv, skv, out_dt, "dK"),
        bshd(hkv, skv, out_dt, "dV"),
        sample_q_T=bshd(hq, sq, _T_E4M3, "q_T"),
        sample_k_T=bshd(hkv, skv, _T_E4M3, "k_T"),
        sample_do_T=bshd(hq, sq, _T_E4M3, "dO_T"),
        sample_do_f16=bshd(hq, sq, out_dt, "dO_f16"),
        **{f"sample_{n}": dense(sf_shapes[n], torch.int8, n) for n in sf_shapes},
        p_scale_log2=p_scale_log2,
        scale_softmax=1.0 / math.sqrt(_D),
        **kw,
    )


def test_adapter_backstop_pins_p_scale_log2_to_8():
    with pytest.raises(ValueError, match="pinned to 8"):
        _mxfp8_adapter(p_scale_log2=7).check_support()
    assert _mxfp8_adapter().check_support()


def test_adapter_backstop_checks_sf_byte_counts_only_and_the_rowwise_sf_v_shape():
    """Any dims with the right F8_128x4 byte total pass (the C++ node rewrites two SF strides; only bytes are trusted) -- except
    ``sf_v``, whose ROWWISE shape is asserted (a columnwise binding has the same byte count and would be a silently wrong dV)."""
    with pytest.raises(ValueError, match="F8_128x4"):
        _mxfp8_adapter(sf_over=dict(sf_q=(1, 2, 128, 8))).check_support()  # half the rows of S_q = 256
    with pytest.raises(ValueError, match="ROWWISE"):
        _mxfp8_adapter(sf_over=dict(sf_v=(1, 2, 8, 256))).check_support()  # the columnwise shape, same bytes
    assert _mxfp8_adapter(sf_over=dict(sf_q=(1, 2 * 256 * 8))).check_support(), "sf_q as a flat byte blob of the right count is fine"


def test_adapter_backstop_declines_fp16_and_ragged_bottom_right():
    with pytest.raises(ValueError, match="fp16 arm"):
        _mxfp8_adapter(out_dt=torch.float16).check_support()
    with pytest.raises(ValueError, match="S_q % 128"):
        _mxfp8_adapter(sq=500, skv=1024, is_causal=True, causal_bottom_right=True).check_support()
    assert _mxfp8_adapter(sq=512, skv=1000, is_causal=True, causal_bottom_right=True).check_support()


def test_workspace_plan_carries_the_sf_pad_slabs_exactly_when_padded(monkeypatch):
    """The bf16-dS chain's scratch plan (the artifact carves it identically): the dequantized q_T / k_T slabs always; the dO_T staging
    + three Q-side SF pad slabs iff S_q % 128 != 0; the two kv-side SF pad slabs (at the kernel's 256-row extent) iff S_kv % 256 != 0.
    (The shipped block-scaled chain's plan is pinned by ``test_p_b_adapter_carves_the_payload_and_atom_workspaces_per_the_contract``.)"""
    from cudnn.sdpa.fwd.api_dsl import ws_align

    _p_c(monkeypatch)
    dense = _mxfp8_adapter(sq=256, skv=512)
    dense.check_support()
    names = [n for n, _shape, _dt in dense._scratch_shapes()]
    assert "q_T_bf16" in names and "k_T_bf16" in names
    assert not any(n in names for n in ("do_T_pad", "sf_q_pad", "sf_do_pad", "sf_doT_pad", "sf_k_pad", "sf_v_pad")), names
    ragged = _mxfp8_adapter(sq=129, skv=800, hq=4, hkv=2)
    ragged.check_support()
    plan = {n: (shape, dt) for n, shape, dt in ragged._scratch_shapes()}
    for n in ("do_T_pad", "sf_q_pad", "sf_do_pad", "sf_doT_pad", "sf_k_pad", "sf_v_pad", "q_T_bf16", "k_T_bf16", "dv_part", "dk_part", "dk_fold", "dv_fold"):
        assert n in plan, (n, sorted(plan))
    assert plan["sf_q_pad"] == ((1, 4, 256, 8), torch.uint8) and plan["sf_doT_pad"] == ((1, 4, 8, 256), torch.uint8)
    assert plan["sf_k_pad"] == ((1, 2, 1024, 8), torch.uint8), "the kv slab grows to the kernel's 256-row pad (800 -> 1024), not the atoms' 896"
    assert plan["q_T_bf16"] == ((1, 129, 4, _D), torch.bfloat16) and plan["k_T_bf16"] == ((1, 800, 2, _D), torch.bfloat16)
    assert ragged.scratch_workspace_bytes() == sum(ws_align(math.prod(shape) * dt.itemsize) for shape, dt in plan.values())
    kv_only = _mxfp8_adapter(sq=256, skv=1000)
    kv_only.check_support()
    names = [n for n, _shape, _dt in kv_only._scratch_shapes()]
    assert "sf_k_pad" in names and "sf_v_pad" in names and "sf_q_pad" not in names and "do_T_pad" not in names


# --------------------------------------------------------------------------- the block-scaled dS chain (P-b) at the adapter


def _p_b(monkeypatch, sm=107):
    """Build adapters under P-b (``MXFP8_DS_SF_POLICY`` monkeypatched -- the default, named explicitly so the test does not depend on
    it) with the prepared plan's device query faked to SM ``sm``."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107, config_sm107 as cfg, prepared_sm107

    monkeypatch.setattr(sm107, "MXFP8_DS_SF_POLICY", cfg.DS_SF_P_B)
    monkeypatch.setattr(prepared_sm107, "_sm", lambda api: sm)


def _p_c(monkeypatch):
    """Build adapters under the bf16-dS twin P-c (``MXFP8_DS_SF_POLICY`` monkeypatched off the shipped default)."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107, config_sm107 as cfg

    monkeypatch.setattr(sm107, "MXFP8_DS_SF_POLICY", cfg.DS_SF_P_C)


def test_both_policies_chunk_against_the_one_8_gib_budget_and_the_8k_h128_chunk_is_32_heads(monkeypatch):
    """ONE stage-2 dS workspace budget for every sm107 row and dS policy (``_SM107_WS_BUDGET_BYTES``, 8 GiB; the block-scaled chain
    no longer carries its own).  The block-scaled default carries 2 + 2/32 bytes per dS element, so at 8K H=128 it chunks 32 heads
    = 4 launches (a 4 GiB budget halved that to 16 heads = 8 launches -- MEASURED on Rubin, cc 10.7, 212 SMs @ 2376 MHz: 1.4 %
    dense / 9.3 % causal of the whole row); the bf16-dS twin at 2 B per element gets 64 heads = 2 launches from the same 8 GiB.
    The smallest legal chunk (nothing fits: the adapter returns it and ``scratch_workspace_bytes`` still reports the whole carve --
    there is no adapter-side device-memory refusal, the caller allocates what the plan reports, or bounds the plan with the graph's
    ``deselect_workspace_greater_than`` for a typed decline) is pinned next to it."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107, config_sm107 as cfg
    from cudnn.sdpa.fwd.api_dsl import ws_align

    pb = _mxfp8_adapter(hq=128, hkv=128, sq=8192, skv=8192)
    assert pb._ds_policy == cfg.DS_SF_P_B and pb._ws_budget_bytes() == sm107._SM107_WS_BUDGET_BYTES == 8 << 30
    assert (pb._b_chunk, pb._qh_chunk) == (1, 32), "the block-scaled chain at 8K H=128: 32-head chunks (4 launches)"
    assert pb._ds_chunk_bytes_per_elem() == 2 + 2 / 32
    assert sm107._sm107_chunks(1, 128, 1, 8192, 8192, pb._ds_chunk_bytes_per_elem(), budget=4 << 30, batch_chunking=False) == (1, 16), "what 4 GiB gave"
    assert sm107._sm107_chunks(1, 128, 1, 16384, 16384, pb._ds_chunk_bytes_per_elem(), batch_chunking=False) == (1, 8), "16K: still chunked (16 launches)"
    assert sm107._sm107_chunks(1, 128, 1, 8192, 8192, pb._ds_chunk_bytes_per_elem(), budget=1, batch_chunking=False) == (1, 1), "the smallest legal chunk"
    _p_c(monkeypatch)
    pc = _mxfp8_adapter(hq=128, hkv=128, sq=8192, skv=8192)
    assert pc._ds_policy == cfg.DS_SF_P_C and pc._ws_budget_bytes() == sm107._SM107_WS_BUDGET_BYTES == 8 << 30
    assert (pc._b_chunk, pc._qh_chunk) == (1, 64) and pc._ds_chunk_bytes_per_elem() == 2, "the bf16-dS twin at 8K H=128: 64-head chunks (2 launches)"
    for api in (pb, pc):
        api.check_support()
        assert api.scratch_workspace_bytes() == sum(ws_align(math.prod(shape) * dt.itemsize) for _n, shape, dt in api._scratch_shapes())


def test_p_b_adapter_carves_the_payload_and_atom_workspaces_per_the_contract(monkeypatch):
    """Under P-b the plan replaces the two dequantized bf16 q_T / k_T slabs by the kernel's second e4m3 payload and the two E8M0 atom
    tensors at the launch's chunk geometry -- ``ds_dq [B, H_chunk, S_kv_pad, S_q_pad]``, ``sf_ds_dk [B, H_chunk, S_kv/128, S_q/128, 512]``,
    ``sf_ds_dq [B, H_chunk, S_q/128, S_kv/128, 512]`` (``config_sm107.sf_workspace_bytes`` each) -- plus the columnwise q_T / k_T SF
    pads the block-scale GEMMs' SFB reads exactly when the shape is ragged (at the SF tensor's own ceil128 extent, not the kernel's
    256-row pad); ``ds_ws`` itself becomes the e4m3 ``ds_dk``; the chunking budgets 2 + 2/32 bytes per dS element."""
    from cudnn.sdpa.bwd import config_sm107 as cfg
    from cudnn.sdpa.fwd.api_dsl import ws_align

    _p_b(monkeypatch)
    dense = _mxfp8_adapter(sq=256, skv=512)
    dense.check_support()
    plan = {n: (tuple(shape), dt) for n, shape, dt in dense._scratch_shapes()}
    assert dense._ds_dtype == torch.float8_e4m3fn and dense._bpe_ds == 1 and dense._ds_block_scaled
    assert plan["ds_ws"] == ((1, 2, 512, 256), torch.float8_e4m3fn), "payload 0 = ds_dk in the shared region"
    assert plan["ds_dq"] == ((1, 2, 512, 256), torch.float8_e4m3fn)
    assert plan["sf_ds_dk"] == ((1, 2, 4, 2, 512), torch.uint8) and plan["sf_ds_dq"] == ((1, 2, 2, 4, 512), torch.uint8)
    c = dense._ds_cfg
    assert math.prod(plan["sf_ds_dk"][0]) == cfg.sf_workspace_bytes(c, 1, 2, 256, 512) == math.prod(plan["ds_ws"][0]) // cfg.MX_BLOCK
    assert not any(n in plan for n in ("q_T_bf16", "k_T_bf16", "sf_qT_pad", "sf_kT_pad")), sorted(plan)
    assert dense._ds_chunk_bytes_per_elem() == c.DS_PAYLOADS * c.BPE_DS + c.DS_SF_ATOMS / c.SF_BLOCK == 2 + 2 / 32
    assert dense.scratch_workspace_bytes() == sum(ws_align(math.prod(shape) * dt.itemsize) for shape, dt in plan.values())
    assert dense._template_params().ds_sf_policy == cfg.DS_SF_P_B
    ragged = _mxfp8_adapter(sq=129, skv=800, hq=4, hkv=2)
    ragged.check_support()
    plan = {n: (tuple(shape), dt) for n, shape, dt in ragged._scratch_shapes()}
    assert plan["sf_qT_pad"] == ((1, 4, 8, 256), torch.uint8), "the columnwise q_T SF at the q pad (S_q 129 -> 256)"
    assert plan["sf_kT_pad"] == ((1, 2, 8, 896), torch.uint8), "the columnwise k_T SF at ITS ceil128 extent (800 -> 896), not the kernel's 1024"
    assert plan["ds_dq"] == ((1, 4, 1024, 256), torch.float8_e4m3fn) and plan["sf_ds_dk"] == ((1, 4, 8, 2, 512), torch.uint8)
    for n in ("do_T_pad", "sf_q_pad", "sf_do_pad", "sf_doT_pad", "sf_k_pad", "sf_v_pad", "dv_part", "dk_part", "dk_fold", "dv_fold"):
        assert n in plan, n
    assert "q_T_bf16" not in plan and "k_T_bf16" not in plan


def test_p_b_adapter_renders_the_block_scale_stage3_records(monkeypatch):
    """P-b's dK / dQ renderings are the block-scale arm (``MatmulTemplateParams.block_scale``) over e4m3 dS: dK K-major, dQ M-major, both
    EPI_NONE with the inherited bf16 output, the K-trim modes / shift / window the bf16 renderings carry; P-c's records are untouched."""
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from cudnn.sdpa.bwd import config_sm107 as cfg
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, EPI_NONE, validate_matmul_params

    class _Mod:
        pass

    def records(api, **kw):
        mod = _Mod()
        mod.CFG = cfg.make_cfg_d256_bwd(api._template_params(), cfg.FAMILY_MXFP8)
        return api._stage3_records(mod, (256, 256))

    _p_c(monkeypatch)
    pc = _mxfp8_adapter(sq=512, skv=1024, is_causal=True, window_size_left=640)
    pc.check_support()
    dk_c, dq_c = records(pc)
    assert (dk_c.dtype_qkv, dq_c.dtype_qkv) == (DTYPE_BF16, DTYPE_BF16) and not dk_c.block_scale and not dq_c.block_scale
    _p_b(monkeypatch)
    pb = _mxfp8_adapter(sq=512, skv=1024, is_causal=True, window_size_left=640)
    pb.check_support()
    dk, dq = records(pb)
    for rec in (dk, dq):
        validate_matmul_params(rec)
        assert rec.block_scale and rec.dtype_qkv == DTYPE_E4M3 and rec.epi_mode == EPI_NONE and rec.dtype_out == -1
        assert rec.cgrp_tile_mn == (256, 256) and rec.causal_window == 640 and rec.causal_gran == 256
    assert (dk.a_is_m_major, dq.a_is_m_major) == (False, True)
    assert (dk.causal_mode, dq.causal_mode) == (dk_c.causal_mode, dq_c.causal_mode) == (CAUSAL_K_LO, CAUSAL_K_HI)


def test_p_b_policy_declines_typed_off_the_rubin_line(monkeypatch):
    """The block-scale GEMM arm needs the Rubin line's 576-column exclusive TMEM: a P-b plan on any other part declines typed at
    ``check_support`` (before anything is compiled); P-a is refused at construction; P-c's backstop is unchanged."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107, config_sm107 as cfg

    for sm in (80, 100, 103, 120):
        _p_b(monkeypatch, sm=sm)
        with pytest.raises(ValueError, match=rf"block-scaled dS policy.*Rubin line.*SM{sm}"):
            _mxfp8_adapter().check_support()
    _p_b(monkeypatch, sm=107)
    assert _mxfp8_adapter().check_support()
    monkeypatch.setattr(sm107, "MXFP8_DS_SF_POLICY", cfg.DS_SF_P_A)
    with pytest.raises(NotImplementedError, match="P-a tile-scale policy"):
        _mxfp8_adapter()


def test_prepared_host_binds_the_appended_ds_operands_before_the_stream_under_both_policies():
    """The kernel's Launch ABI appends ds_dk / ds_dq / sf_ds_dk / sf_ds_dq after seqlen_q_real with ``stream`` last; ``host_mxfp8``
    must hand the four (None under P-c) BEFORE the stream on BOTH arms -- a caller that keeps the pre-arm positional shape binds the
    stream to ``ds_dk`` and launches with none (``cuda.launch_cfg.create`` operand None; MEASURED on a Rubin node, every device case of
    this module red).  Also the region slots: the five appended P-b regions follow the P-c ones and the host's table matches."""
    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host as ph

    code = _code_only(open(ph.__file__, encoding="utf-8").read())
    host = _def_body(code, "host_mxfp8")
    calls = re.findall(r"\bmain\((.*?)\n\s*\)", host, re.S)
    assert len(calls) == 2, f"host_mxfp8 launches the kernel once per policy arm; got {len(calls)} call(s)"
    for call in calls:
        args = [a.strip() for a in call.strip().strip(",").split(",\n")]
        assert len(args) == 25, f"the kernel takes 24 positionals + the stream; got {len(args)}: {args}"
        assert args[-1] == "stream" and args[19] == "sq", args
    pb_call = next(c for c in calls if "ds_dq_full" in c)
    pc_call = next(c for c in calls if "ds_dq_full" not in c)
    assert [a.strip() for a in pb_call.strip().strip(",").split(",\n")][20:24] == ["ds_full", "ds_dq_full", "sf_ds_dk", "sf_ds_dq"]
    assert [a.strip() for a in pc_call.strip().strip(",").split(",\n")][20:24] == ["None", "None", "None", "None"]
    assert [a.strip() for a in pb_call.strip().strip(",").split(",\n")][6] == "None", "P-b binds None for the bf16 ds_ws operand"
    slots = prepared_sm107._REGION_SLOTS_MXFP8
    assert len(slots) == ph.N_REGIONS_MXFP8 == 26
    assert slots[ph.R_MX_DS_DQ :] == ("ds_dq", "sf_ds_dk", "sf_ds_dq", "sf_qT_pad", "sf_kT_pad") and slots[: ph.R_MX_DS_DQ] == slots[:21]
    assert "_stage3_block_scale(" in host and host.count("dequant_mxfp8_to_bf16_host(") == 2, "P-c keeps its two dequant passes; P-b has none"


# =========================================================================== ACCEPT -- the MXFP8 backward oracle (Rubin)


def _to_bshd(t_bhsd):
    """[B,H,S,D] (any storage) -> a [B,H,S,D] view over BSHD-physical memory."""
    return t_bhsd.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


class _Quant:
    """Rowwise + columnwise MXFP8 of a [B, S_real, H, D] tensor, padded to the F8_128x4 extent as the PRODUCER hands it to the graph
    (the bring-up driver's class): ``pay_d`` / ``pay_s`` e4m3 payloads [B, S_real, H, D] BSHD; ``sf_d`` [B, H, S_pad, 8] / ``sf_s``
    [B, H, S_pad / 32, 256] uint8 F8_128x4 bytes (the graph's dims); ``ref_d`` / ``ref_s`` the oracle's [B, H, S_real, D] e4m3 views
    and ``sfref_d`` / ``sfref_s`` its per-element fp32 dequant scales.  ``poison_pads`` fills the pad rows / groups with 0xFF (E8M0
    NaN): the producer-defined bytes the adapter's staging must zero."""

    def __init__(self, t_bshd, s_real, poison_pads=False):
        from sdpa.mxfp8_quant import e8m0_to_float, quantize_mxfp8_2d, swizzle_sf_columnwise, swizzle_sf_rowwise

        bb, s_pad, hh, d = t_bshd.shape[0], _ceil128(s_real), t_bshd.shape[2], t_bshd.shape[3]
        assert t_bshd.shape[1] == s_real
        l = bb * hh
        x = t_bshd.permute(0, 2, 1, 3).float().reshape(l, s_real, d)
        if s_pad != s_real:
            x = torch.nn.functional.pad(x, (0, 0, 0, s_pad - s_real))
        row_data, row_e, col_data, col_e = quantize_mxfp8_2d(x.reshape(l * s_pad, d), _T_E4M3)
        if poison_pads and s_pad != s_real:
            row_e.view(l, s_pad, d // 32)[:, s_real:, :] = 0xFF
            col_e.view(l, s_pad // 32, d)[:, -(-s_real // 32) :, :] = 0xFF
        self.s_real, self.s_pad = s_real, s_pad
        self.pay_d = row_data.reshape(bb, hh, s_pad, d)[:, :, :s_real].permute(0, 2, 1, 3).contiguous()
        self.pay_s = col_data.reshape(bb, hh, s_pad, d)[:, :, :s_real].permute(0, 2, 1, 3).contiguous()
        self.sf_d = swizzle_sf_rowwise(row_e).contiguous().reshape(bb, hh, s_pad, d // 32)
        self.sf_s = swizzle_sf_columnwise(col_e).contiguous().reshape(bb, hh, s_pad // 32, d)
        self.ref_d = row_data.reshape(bb, hh, s_pad, d)[:, :, :s_real].contiguous()
        self.ref_s = col_data.reshape(bb, hh, s_pad, d)[:, :, :s_real].contiguous()
        self.sfref_d = torch.repeat_interleave(e8m0_to_float(row_e).reshape(l, s_pad, d // 32), 32, dim=2)[:, :s_real].contiguous()
        self.sfref_s = torch.repeat_interleave(e8m0_to_float(col_e).reshape(l, s_pad // 32, d), 32, dim=1)[:, :s_real].contiguous()

    def deq_d(self):
        return self.ref_d.float() * self.sfref_d.view(self.ref_d.shape)

    def deq_s(self):
        return self.ref_s.float() * self.sfref_s.view(self.ref_s.shape)


def _report(tag, got, want, atol, rtol):
    diff = (got.float() - want.float()).abs()
    bad = int((diff > atol + rtol * want.float().abs()).sum())
    a, r = got.float().flatten().double(), want.float().flatten().double()
    cos = float(a @ r / (a.norm() * r.norm() + 1e-30))
    print(
        f"\n{tag}: max|diff| = {float(diff.max()):.4g} (max|ref| = {float(want.float().abs().max()):.4g}), cos = {cos:.7f}, outside atol={atol}/rtol={rtol}: {bad} of {got.numel()}"
    )
    return float(diff.max()), cos, bad


class _MxRun:
    pass


_POLICY_IDS = {}


def _policy_params():
    from cudnn.sdpa.bwd import config_sm107 as cfg

    _POLICY_IDS.update({cfg.DS_SF_P_C: "P-c", cfg.DS_SF_P_B: "P-b"})
    return [pytest.param(cfg.DS_SF_P_C, id="P-c"), pytest.param(cfg.DS_SF_P_B, id="P-b")]


@pytest.fixture(params=_policy_params())
def ds_policy(request, monkeypatch):
    """The dS policy the NEXT adapter reads (``api_dsl_sm107.MXFP8_DS_SF_POLICY``): P-b (the block-scaled chain, what ships) or P-c
    (bf16 dS, the oracle twin) -- the fp8 suite's ``ds_knob`` pattern.  Every accept cell that takes it runs BOTH arms; a cell
    without it runs the shipped default."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107

    monkeypatch.setattr(sm107, "MXFP8_DS_SF_POLICY", request.param)
    return request.param


def _active_policy():
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107

    return sm107.MXFP8_DS_SF_POLICY


def _run_mxfp8(
    b=1,
    hq=2,
    hkv=None,
    sq=512,
    skv=None,
    causal=False,
    bottom_right=False,
    window=None,
    seed=0,
    runs=1,
    poison_sf_pads=False,
    check=True,
    poison=float("nan"),
    attn_scale=None,
    q_mult=1.0,
    k_mult=1.0,
    repeat_outputs=0,
    frost_forward=False,
):
    """Quantize bf16-rounded unit-normal operands (rowwise + columnwise, the producer's F8_128x4 SF), compute the forward in torch over
    the DEQUANTIZED operands under the graph's mask (natural-log LSE = the graph's Stats, O -> o_f16), build the graph, PIN the engine,
    execute ``runs`` times over ``poison``-filled outputs, and compare dQ / dK / dV with ``compute_ref_backward(..., stats=LSE,
    quantize_ds=False)`` -- the fp32-dS oracle of the P-c twin -- dV under the fp8 recipe, dK / dQ under the bf16 row's (module
    docstring).  ``window`` is the GRAPH's ``left_bound`` (the analyzer derives the kernel's offset W - 1; the bring-up driver's lesson,
    sdpa-invariants s8).  ``q_mult`` / ``k_mult`` scale a payload by a power of two AFTER the bf16 rounding (exact) so its 32-blocks carry
    E8M0 bytes >= 128; ``attn_scale`` (None = 1 / sqrt(D)) is the graph's and the oracle's -- ``_SF_MULT`` with ``attn_scale`` divided by
    it keeps the softmax and the scaled operand's gradient at the unit-normal magnitudes.  ``repeat_outputs`` > 0 re-executes that many
    times over alternately NaN- and 123.0-poisoned outputs, eager and through a captured CUDA graph, asserting bitwise equality with
    the first run (the SM100 suite's pattern).  ``frost_forward`` takes Stats and o_f16 from the FROST d256 MXFP8 FORWARD on the same
    quantized operands instead of torch (pinning the forward's LSE within 1e-4 of the exact one first): the exact-LSE e2e contract."""
    from sdpa.fp8 import assert_close_fp8_grad
    from sdpa.mxfp8_ref import compute_ref_backward
    from cudnn.sdpa.bwd import config_sm107 as cfg

    policy = _active_policy()
    block_scaled = policy == cfg.DS_SF_P_B
    hkv = hq if hkv is None else hkv
    skv = sq if skv is None else skv
    dev, bf16 = "cuda", torch.bfloat16
    grp = hq // hkv
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(s, h, mult=1.0):
        return torch.randn(b, s, h, _D, generator=gen).to(bf16).float().to(dev) * mult

    qQ, qK, qV = (
        _Quant(draw(sq, hq, q_mult), sq, poison_sf_pads),
        _Quant(draw(skv, hkv, k_mult), skv, poison_sf_pads),
        _Quant(draw(skv, hkv), skv, poison_sf_pads),
    )
    do32 = draw(sq, hq)
    qdO = _Quant(do32, sq, poison_sf_pads)
    scale = 1.0 / math.sqrt(_D) if attn_scale is None else float(attn_scale)
    q_deq, k_deq, v_deq = qQ.deq_d(), qK.deq_d(), qV.deq_d()
    s_raw = torch.einsum("bhqd,bhkd->bhqk", q_deq, k_deq.repeat_interleave(grp, dim=1))
    rel = torch.arange(skv, device=dev).view(1, skv) - torch.arange(sq, device=dev).view(sq, 1)
    diag = (skv - sq) if bottom_right else 0
    masked = torch.zeros(sq, skv, dtype=torch.bool, device=dev)
    right_bound = left_bound = diag_align = None
    if causal:
        right_bound = 0
        masked |= rel > diag
        diag_align = cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT
    if window is not None:
        left_bound = window
        masked |= rel <= diag - window
    s_scaled = (s_raw * scale).masked_fill(masked, float("-inf"))
    lse = torch.logsumexp(s_scaled, dim=-1).contiguous()  # [B, H, S_q], natural log
    assert bool(torch.isfinite(lse).all()), "a fully masked q row: pick S_q <= S_kv for bottom-right"
    p_fwd = torch.exp(s_scaled - lse.unsqueeze(-1))
    o_f16 = torch.einsum("bhqk,bhkd->bhqd", p_fwd, v_deq.repeat_interleave(grp, dim=1)).to(bf16)  # [B, H, S_q, D]
    dO_f16 = do32.permute(0, 2, 1, 3).contiguous().to(bf16)
    lse_exact = lse
    if frost_forward:
        # The FROST d256 MXFP8 forward (sm107/prefill_d256_mxfp8.py through the SM100-named adapter, as test_sdpa_fwd_dsl_sm107 drives
        # it) on the SAME rowwise q / k and columnwise v payloads: its Stats must be the exact fp32 log-sum-exp (sdpa-invariants s4,
        # within 1e-4 of the dequantized-input value) and its O the o_f16 this row folds delta from.
        from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

        assert window is None and not bottom_right, "the e2e cell composes dense / top-left causal only"
        q8 = qQ.pay_d.permute(0, 2, 1, 3)  # [B, H, S, D] views over BSHD memory
        k8, v8 = qK.pay_d.permute(0, 2, 1, 3), qV.pay_s.permute(0, 2, 1, 3)
        out = torch.full((b, sq, hq, _D), 1.5e30, device=dev, dtype=bf16).transpose(1, 2)
        lse_frost = torch.full((b, hq, sq), float("nan"), device=dev, dtype=torch.float32)
        api = SdpaFwdDslSm100(q8, k8, v8, out, lse_frost, scale_softmax=scale, is_causal=causal, pertensor_fp8=False, dtype_o=bf16)  # d256: cga 1 only
        assert api.check_support(), "the FROST d256 MXFP8 forward declined the e2e cell's operands"
        api.compile()
        ws_fwd = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)  # the prepared d256 launch carves 128 B
        api.execute(
            q8,
            k8,
            v8,
            out,
            lse_tensor=lse_frost,
            sf_q=qQ.sf_d.reshape(-1, _D // 32),
            sf_k=qK.sf_d.reshape(-1, _D // 32),
            sf_v=qV.sf_s.reshape(-1, _D),
            workspace=ws_fwd,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(lse_frost).all() and not (out.float() == 1.5e30).any(), "the forward left Stats / O rows unwritten"
        d_lse = (lse_frost - lse).abs().max().item()
        print(f"\nFROST forward Stats vs the exact log-sum-exp: max |dLSE| = {d_lse:.3e}")
        assert d_lse <= 1e-4, f"the forward's Stats is not the exact log-sum-exp (max |dLSE| {d_lse:.3e}; a quantized-P row-sum reads 1e-3..1e-2)"
        lse = lse_frost.contiguous()
        o_f16 = out.contiguous()  # [B, H, S_q, D]

    g, t, (dq_t, dk_t, dv_t, *_amax) = _mxfp8_graph(b, hq, hkv, sq, skv, causal=causal, bottom_right=bottom_right, left_bound=window, scale=scale)
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    outs_t = dict(
        dQ=torch.empty(b, sq, hq, _D, device=dev, dtype=bf16),
        dK=torch.empty(b, skv, hkv, _D, device=dev, dtype=bf16),
        dV=torch.empty(b, skv, hkv, _D, device=dev, dtype=bf16),
    )
    pack = {
        t["q"]: qQ.pay_d.permute(0, 2, 1, 3),
        t["q_T"]: qQ.pay_s.permute(0, 2, 1, 3),
        t["k"]: qK.pay_d.permute(0, 2, 1, 3),
        t["k_T"]: qK.pay_s.permute(0, 2, 1, 3),
        t["v"]: qV.pay_d.permute(0, 2, 1, 3),
        t["o_f16"]: _to_bshd(o_f16),
        t["dO_f16"]: _to_bshd(dO_f16),
        t["dO"]: qdO.pay_d.permute(0, 2, 1, 3),
        t["dO_T"]: qdO.pay_s.permute(0, 2, 1, 3),
        t["stats"]: lse.contiguous(),
        t["sf_q"]: qQ.sf_d,
        t["sf_q_T"]: qQ.sf_s,
        t["sf_k"]: qK.sf_d,
        t["sf_k_T"]: qK.sf_s,
        t["sf_v"]: qV.sf_d,
        t["sf_dO"]: qdO.sf_d,
        t["sf_dO_T"]: qdO.sf_s,
        dq_t: outs_t["dQ"].permute(0, 2, 1, 3),
        dk_t: outs_t["dK"].permute(0, 2, 1, 3),
        dv_t: outs_t["dV"].permute(0, 2, 1, 3),
    }
    outs = []
    for _ in range(runs):
        for x in outs_t.values():
            x.fill_(poison)
        g.execute(pack, ws)
        torch.cuda.synchronize()
        outs.append({n: x.clone() for n, x in outs_t.items()})
    if repeat_outputs:
        # Eager AND captured execution over both a non-finite and a finite previous content of the outputs: a store skipped for
        # a fully masked kv tile, or a per-execute host state, would show as a bit difference against run 0.
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay, stream=torch.cuda.current_stream()):
            g.execute(pack, ws)
        for i in range(repeat_outputs):
            for x in outs_t.values():
                x.fill_(float("nan") if i % 4 < 2 else 123.0)
            if i % 2:
                replay.replay()
            else:
                g.execute(pack, ws)
            torch.cuda.synchronize()
            for name, x in outs_t.items():
                same = torch.equal(x.view(torch.int16), outs[0][name].view(torch.int16))
                assert same, f"{name}: {'replay' if i % 2 else 'eager'} run {i} over {'NaN' if i % 4 < 2 else '123.0'}-poisoned outputs differs from run 0"
    run = _MxRun()
    run.graph, run.pack, run.workspace, run.outs, run.lse, run.scale = g, pack, ws, outs, lse, scale
    run.shape = (b, hq, hkv, sq, skv)
    run.policy = policy
    run.quant = dict(q=qQ, k=qK, v=qV, dO=qdO)
    run.o_f16, run.dO_f16, run.masked, run.s_raw, run.lse_exact = o_f16, dO_f16, masked, s_raw, lse_exact
    if not check:
        return run
    # The oracle composes the chain's own dS rounding (sdpa-invariants s8): fp32 dS for the bf16-dS chain, 1x32 e4m3 both ways for
    # the block-scaled one (exactly P-b's convention).
    dQ_ref, dK_ref, dV_ref, _dsink = compute_ref_backward(
        qQ.ref_d, qQ.ref_s, qK.ref_d, qK.ref_s, qV.ref_d, o_f16, dO_f16, qdO.ref_d, qdO.ref_s, scale,
        qQ.sfref_d, qQ.sfref_s, qK.sfref_d, qK.sfref_s, qV.sfref_d, qdO.sfref_d, qdO.sfref_s,
        torch_itype=_T_E4M3, torch_otype=bf16, left_bound=left_bound, right_bound=right_bound, diag_align=diag_align, stats=lse, quantize_ds=block_scaled,
    )  # fmt: skip
    refs = dict(dQ=dQ_ref, dK=dK_ref, dV=dV_ref)
    keys = dict(dQ=skv, dK=sq, dV=sq)
    run.stats = {}
    oracle = "e4m3-dS (1x32 both ways) oracle" if block_scaled else "fp32-dS oracle"
    for name in ("dV", "dK", "dQ"):
        got = outs[0][name].permute(0, 2, 1, 3).float()  # [B, H, S, D]
        assert torch.isfinite(got).all(), f"{name}: non-finite output ({int(torch.isnan(got).sum())} NaN cells)"
        fp8_recipe = name == "dV" or block_scaled
        tol = _GRAD_TOL if fp8_recipe else _BF16_GRAD_TOL
        run.stats[name] = _report(f"{name} vs the {oracle} ({'fp8' if fp8_recipe else 'bf16'} recipe)", got, refs[name], tol["atol"], tol["rtol"])
        if fp8_recipe:
            # dV: the e4m3 P feeds BMM2 on both sides.  P-b's dK / dQ: both sides round dS to e4m3 per 32-block from fp32 values ~1e-6
            # apart, so a midpoint flip of one dS cell moves one gradient row (the fp8 row's dS-flip class) -- the same budget.
            assert_close_fp8_grad(got, refs[name].float(), tol["atol"], tol["rtol"], tag=name, keys=keys[name], budget=1e-5)
        else:
            torch.testing.assert_close(got, refs[name].float(), **tol, msg=lambda m, n=name: f"{n} vs the fp32-dS oracle under the bf16 row's recipe: {m}")
    return run


@requires_rubin
def test_dense(ds_policy):
    _run_mxfp8()


@requires_rubin
def test_causal(ds_policy):
    _run_mxfp8(sq=1024, skv=1024, causal=True)


@requires_rubin
def test_causal_bottom_right_rectangular():
    _run_mxfp8(sq=512, skv=1024, causal=True, bottom_right=True)


@requires_rubin
def test_sliding_window():
    """W = 640 is the graph's ``left_bound``; the adapter hands the kernel the analyzer's offset (W - 1)."""
    _run_mxfp8(sq=1024, skv=1024, causal=True, window=640)


@requires_rubin
def test_gqa(ds_policy):
    _run_mxfp8(hq=8, hkv=2, sq=512, skv=512)


@requires_rubin
def test_sliding_window_non_multiple():
    """W = 600: a window that is not a multiple of the kv block (the K-trim's roundings and the band's edge inside a tile)."""
    _run_mxfp8(sq=1024, skv=1024, causal=True, window=600)


@requires_rubin
def test_sliding_window_without_causal():
    """The ``left_bound``-only arm the row claims (``swa`` without ``causal``): the right side of every row attends."""
    _run_mxfp8(sq=1024, skv=1024, window=200)


@requires_rubin
def test_mqa():
    _run_mxfp8(hq=8, hkv=1, sq=512, skv=512)


@requires_rubin
def test_causal_bottom_right_ragged_kv():
    """Bottom-right at ``S_q % 128 == 0`` with a ragged S_kv (the served half of the bottom-right rule): the diagonal S_kv - S_q is not
    a tile multiple and the kv pad rows are masked AND staged."""
    _run_mxfp8(sq=512, skv=1000, causal=True, bottom_right=True)


def _assert_tail_rows_are_exact_zeros(run, sq):
    for name in ("dK", "dV"):
        tail = run.outs[0][name][:, sq:].float()
        assert torch.isfinite(tail).all() and (tail == 0).all(), f"{name}: kv rows >= S_q attend no q -- they must be EXACTLY zero, not poison or residue"


@requires_rubin
def test_causal_tail_kv_block_writes_zeros_not_residue():
    """Top-left causal with S_kv > S_q (sdpa-invariants s1, the empty range): the kv blocks past S_q attend no q, so the body runs the
    forced fully-masked tile (``_q_loop_bounds`` clamps N to 1; P = 0 through the SELECT; dV overwritten by ``accumulate=(q_iter >
    q_lo)``; dS = 0), and dK / dV rows >= S_q must come out EXACTLY zero over NaN-poisoned outputs -- on THIS body the forced tile
    also routes the five SF loads + UTCCPs (review 2026-09-30: proven on the fp8 body only until now)."""
    sq, skv = 256, 768
    run = _run_mxfp8(b=2, hq=2, sq=sq, skv=skv, causal=True)
    _assert_tail_rows_are_exact_zeros(run, sq)


@requires_rubin
def test_sliding_window_tail_kv_block_writes_zeros_not_residue():
    """The same empty range under a top-left window (the fp8 suite's ``swa200-tl-rect-kv``): blocks past S_q run the forced tile."""
    sq, skv = 512, 1024
    run = _run_mxfp8(sq=sq, skv=skv, causal=True, window=200)
    _assert_tail_rows_are_exact_zeros(run, sq)


@requires_rubin
@pytest.mark.parametrize("n_q_tiles", [3, 8], ids=["nq3-ring-depth", "nq8"])
def test_q_tiles_per_kv_block_with_several_tiles_in_flight(n_q_tiles, ds_policy):
    """q tiles per kv block = 3 (the Q / dO ring depth) and 8, at B*H = 4 (several tiles in flight per CTA pair): the rings wrap
    under phase drift (sdpa-invariants s6 / s9) and two launches are bitwise (the two-launch trick at n_q_tiles >= 8, B*H >= 2).
    Both dS policies: P-b's 2-deep slot carries FOUR buffers behind the same two barriers."""
    run = _run_mxfp8(b=2, hq=2, sq=128 * n_q_tiles, skv=256, runs=2)
    for name in ("dQ", "dK", "dV"):
        assert torch.equal(run.outs[0][name].view(torch.int16), run.outs[1][name].view(torch.int16)), f"{name}: two launches differ"


@requires_rubin
def test_cuda_graph_replay_is_bitwise_over_poisoned_outputs(ds_policy):
    """Eager and captured (``torch.cuda.CUDAGraph``) executions over NaN- and 123.0-poisoned outputs are bitwise run 0 (the SM100
    suite's ``repeat_outputs`` pattern), on a top-left causal shape with S_kv > S_q so the forced tile's zero stores are replayed too."""
    sq, skv = 256, 768
    stream = torch.cuda.Stream()  # capture needs a non-default stream (the SM100 test's wrapper); replay is stream-agnostic
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run = _run_mxfp8(b=1, hq=2, sq=sq, skv=skv, causal=True, repeat_outputs=6)
    stream.synchronize()
    _assert_tail_rows_are_exact_zeros(run, sq)


@requires_rubin
def test_head_chunked_launch(monkeypatch, ds_policy):
    """H_chunk < H: the stage-2 kernel runs once per head chunk with ``head_base`` walking the chunks (the dS workspace, the Q-side
    SF atoms and the columnwise dO_T SF descriptor's collapsed (batch, head) index are all chunk-relative).  The budget is forced
    down so a 4-head GQA graph launches in chunks of one group (the smallest legal chunk); a wrong ``full_head`` / ``doT_sf_bh``
    index would read another head's scale factors."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107

    orig, seen = sm107._sm107_chunks, []

    def forced(*a, **k):
        k["budget"] = 1  # the adapter passes the shared budget by keyword; override it
        r = orig(*a, **k)
        seen.append(r)
        return r

    monkeypatch.setattr(sm107, "_sm107_chunks", forced)
    _run_mxfp8(hq=4, hkv=2, sq=512, skv=512, causal=True)
    assert seen and seen[-1] == (1, 2), f"the forced chunking did not reach the adapter: {seen}"


@requires_rubin
@pytest.mark.parametrize("which", ["q", "k"])
def test_sf_bytes_at_or_above_128_dequantize_with_the_right_sign(which):
    """The review blocker (2026-09-30): the host-side dequant of the columnwise q_T / k_T sign-extended the E8M0 byte, so every
    32-block with amax > 448 (byte >= 128, legal MXFP8) dequantized with a FLIPPED sign -> dK = dS . Q_T (or dQ = dS^T . K_T) sign-
    flipped, finite, plausible.  Unit-normal data never produces such a byte, so Q (or K) is scaled by 2^12 after the bf16 rounding
    (bytes 128..133 in its SF) and ``attn_scale`` divided by 2^12 keeps the softmax and the scaled operand's gradient at unit-normal
    magnitudes -- the bf16 recipe then reads a sign flip as max|diff| ~ 2 max|ref|.  (The OTHER gradient scales down by 2^12 and
    passes trivially; the two cells cover both stage-3 operands.)"""
    run = _run_mxfp8(
        sq=512, skv=512, attn_scale=(1.0 / math.sqrt(_D)) / _SF_MULT, q_mult=_SF_MULT if which == "q" else 1.0, k_mult=_SF_MULT if which == "k" else 1.0
    )
    q = run.quant[which]
    assert int(q.sf_s.max()) >= 128, f"the scaled {which} carries no E8M0 byte >= 128 (max {int(q.sf_s.max())}) -- the cell tests nothing"
    grad = "dK" if which == "q" else "dQ"
    _md, cos, _bad = run.stats[grad]
    assert cos > 0.999, f"{grad}: cos {cos:.6f} -- a sign flip reads as cos ~ -1"


@requires_rubin
@pytest.mark.parametrize("columnwise", [True, False], ids=["columnwise", "rowwise"])
def test_dequant_host_decodes_every_e8m0_byte_unsigned(columnwise):
    """``dequant_mxfp8_to_bf16_host`` over ALL 256 E8M0 byte values (two heads, two tiles: the head / tile strides too) against the
    oracle's ``e8m0_to_float``: the payload is +-448 (the e4m3 max; ``448 x 2^(e-127)`` is exact in bf16 for every e, overflowing to
    +-inf on BOTH sides for e >= 247 with the SIGN preserved), so every byte is checkable bitwise; byte 255 is NaN on both sides.
    A sign-extended byte >= 128 would land the wrong sign on half the range."""
    from cutlass.cute.runtime import from_dlpack
    from sdpa.mxfp8_quant import e8m0_to_float, swizzle_sf_columnwise, swizzle_sf_rowwise

    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import dequant_mxfp8_to_bf16_host

    b, h, s, d = 1, 2, 256, _D
    l, dev = b * h, "cuda"
    sign = torch.where(torch.arange(d, device=dev) % 2 == 0, 1.0, -1.0)
    pay = (torch.full((b, s, h, d), 448.0, device=dev) * sign).to(_T_E4M3)  # [B, S, H, D] BSHD
    if columnwise:
        col_e = (torch.arange(l * (s // 32) * d, device=dev) % 256).to(torch.uint8).view(l * (s // 32), d)  # logical [S/32, D] per head, stacked
        sf = swizzle_sf_columnwise(col_e).contiguous()
        scale = torch.repeat_interleave(e8m0_to_float(col_e).view(l, s // 32, d), 32, dim=1)  # [L, S, D]
    else:
        row_e = (torch.arange(l * s * (d // 32), device=dev) % 256).to(torch.uint8).view(l * s, d // 32)  # logical [S, D/32] per head, stacked
        sf = swizzle_sf_rowwise(row_e).contiguous()
        scale = torch.repeat_interleave(e8m0_to_float(row_e).view(l, s, d // 32), 32, dim=2)  # [L, S, D]
    want = (pay.float().permute(0, 2, 1, 3).reshape(l, s, d) * scale).to(torch.bfloat16)
    dst = torch.full((b, s, h, d), float("nan"), device=dev, dtype=torch.bfloat16)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    @cute.jit
    def _drv(src_t: cute.Tensor, sf_t: cute.Tensor, dst_t: cute.Tensor, stream_: cuda_driver.CUstream):
        dequant_mxfp8_to_bf16_host(src_t, sf_t, dst_t, columnwise, stream_)

    args = (from_dlpack(pay, assumed_align=16), from_dlpack(sf, assumed_align=16), from_dlpack(dst, assumed_align=16), stream)
    cute.compile(_drv, *args)(*args)
    torch.cuda.synchronize()
    got = dst.permute(0, 2, 1, 3).reshape(l, s, d)
    nan_got, nan_want = torch.isnan(got), torch.isnan(want)
    assert torch.equal(nan_got, nan_want), f"NaN positions differ ({int(nan_got.sum())} vs {int(nan_want.sum())}; byte 255 alone is NaN)"
    ok = nan_want | (got.view(torch.int16) == want.view(torch.int16))
    bad = int((~ok).sum())
    if bad:
        idx = (~ok).nonzero()[:8].tolist()
        detail = [(tuple(i), got[tuple(i)].item(), want[tuple(i)].item()) for i in idx]
        raise AssertionError(f"{bad} of {ok.numel()} dequantized cells differ from the oracle (first: {detail}); a sign flip on bytes >= 128 reads as -want")
    assert (
        int(nan_want.sum()) == (l * (s // 32) * d) // 256 * 32 if columnwise else (l * s * (d // 32)) // 256 * 32
    ), "byte 255 must appear (the probe covers 0..255)"


@requires_rubin
def test_frost_forward_stats_feed_this_row():
    """The exact-LSE e2e contract (sdpa-invariants s4): the FROST d256 MXFP8 FORWARD's Stats and O on the same
    quantized operands, into this row -- the forward's LSE within 1e-4 of the exact log-sum-exp (a Sigma / quantized-P row-sum
    would read 1e-3..1e-2 and cost a dK row per fp8 rounding flip downstream), then dQ / dK / dV within the recipe of the oracle
    fed the SAME Stats."""
    _run_mxfp8(b=1, hq=2, sq=512, skv=512, frost_forward=True)


@requires_rubin
@pytest.mark.parametrize("skv", [800, 1000], ids=["kv800-pad-class-A", "kv1000-pad-class-B"])
def test_kv_tail_both_pad_classes(skv, ds_policy):
    """The kernel pads S_kv to 256, the SF atoms to 128: ``S_kv % 256 in (0, 128]`` (800 -> a whole zero tile appended) and
    ``(128, 256)`` (1000 -> pad rows inside the last atom) are DIFFERENT staging cells (stage 2's head-stride bug hid in the first)."""
    _run_mxfp8(hq=4, hkv=2, sq=512, skv=skv)


@requires_rubin
@pytest.mark.parametrize("sq", [129, 160], ids=["q129", "q160"])
def test_q_tail(sq, ds_policy):
    """A ragged S_q: the +inf-LSE pad rows, the q < seqlen_q_real band (MASK_Q_PAD) and the zero-filled Q-side SF pads."""
    _run_mxfp8(sq=sq, skv=256)


@requires_rubin
def test_batch_three_n_q_tiles_one():
    """B*H > 1 with one q tile per kv block: the persistent scheduler's phase-drift cell (mbarrier-patterns P14)."""
    _run_mxfp8(b=3, hq=2, sq=128, skv=256)


@requires_rubin
def test_two_launches_are_bitwise(ds_policy):
    run = _run_mxfp8(runs=2)
    for name in ("dQ", "dK", "dV"):
        a, c = run.outs[0][name].view(torch.int16), run.outs[1][name].view(torch.int16)
        assert torch.equal(a, c), f"{name}: two launches differ (max |d| = {(run.outs[0][name].float() - run.outs[1][name].float()).abs().max().item():g})"


@requires_rubin
@pytest.mark.parametrize("sq,skv", [(160, 800), (129, 257), (129, 384)], ids=["q160-kv800", "q129-kv257", "q129-kv384"])
def test_poisoned_sf_pads_are_zeroed_by_the_staging(sq, skv, ds_policy):
    """GREEN half: the producer's SF pad rows / groups past S_q and S_kv hold 0xFF (E8M0 NaN) and the
    outputs are still finite and within budget -- the adapter's zero-filled staging is what the kernel reads.  The S_q 129
    x S_kv 257 / 384 cells (a one-row q tail; a kv tail one row past a tile and exactly half a kv block) join the S_q 160 x S_kv 800 cell.
    Under P-b the block-scale GEMMs read the columnwise q_T / k_T scale factors as whole atoms too (no real-extent dequant pass in
    between), so their pad groups are re-staged as well -- the poisoned ``sf_q_T`` / ``sf_k_T`` of these cells prove it."""
    _run_mxfp8(hq=4, hkv=2, sq=sq, skv=skv, poison_sf_pads=True)


@requires_rubin
def test_poisoned_sf_pads_are_red_without_the_staging(monkeypatch):
    """RED half: the same shape with the staging switched OFF (``prepared_host.MXFP8_STAGE_SF_PADS``, a plan-time constant folded
    into the artifact) reads the poisoned bytes -> NaN in the gradients (MEASURED 2026-09-30 through the kernel alone: dV NaN on
    every cell at S_q 160).  A test that cannot fail for the right reason proves nothing; this is the failing half."""
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host

    monkeypatch.setattr(prepared_host, "MXFP8_STAGE_SF_PADS", False)
    run = _run_mxfp8(hq=4, hkv=2, sq=160, skv=800, poison_sf_pads=True, check=False)
    finite = {n: bool(torch.isfinite(x.float()).all()) for n, x in run.outs[0].items()}
    assert not all(finite.values()), f"the RED twin came back finite -- the staging switch is not reaching the artifact: {finite}"


@requires_rubin
def test_ds_is_computed_from_the_fp32_p_not_the_e4m3_p(monkeypatch):
    """The exact-LSE / no-double-rounding pin (sdpa-invariants s4) on the bf16-dS twin P-c, selected explicitly
    (the shipped P-b writes e4m3 payloads, pinned against the fp32-P dS by
    ``test_p_b_ds_payloads_and_atoms_dequantize_to_the_oracles_quantized_ds``): the kernel's bf16 dS workspace (read back from the
    graph's workspace at the plan's region offset) is the fp32 ``P (dP - delta) * scale`` rounded ONCE to bf16 -- within 2^-7
    relative + one bf16 ulp of max|dS| on every cell -- AND closer to that than to the dS a kernel would form from the e4m3 P (the
    positive control: the two twins differ on hundreds of cells at this shape, the bf16-dS bring-up measured 220 at S 512)."""
    from cudnn.sdpa.fwd.api_dsl import ws_align

    _p_c(monkeypatch)
    b, hq, hkv, sq, skv = 1, 2, 2, 512, 512
    run = _run_mxfp8(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv)
    # The dS region's offset from the plan's OWN carve (``prepared_sm107._regions`` walks ``_scratch_shapes()`` in order with
    # ``ws_align``; the standalone adapter of the same shape lays out the same regions) -- never the arithmetic of one layout.
    api = _mxfp8_adapter(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv)
    api.check_support()
    off, ds_shape = 0, None
    for name, shape, dt in api._scratch_shapes():
        if name == "ds_ws":
            ds_shape = tuple(int(x) for x in shape)
            break
        off += ws_align(math.prod(shape) * dt.itemsize)
    assert ds_shape == (b, hq, skv, sq), f"the dS region is not [B, H_chunk = H, S_kv_pad, S_q_pad] on this dense shape: {ds_shape}"
    ds = run.workspace[off : off + b * hq * skv * sq * 2].view(torch.bfloat16).view(b, hq, skv, sq).float().transpose(-1, -2)  # [B, H, S_q, S_kv]
    assert torch.isfinite(ds).all(), "dS workspace holds NaN (unwritten cells on a dense shape)"
    log2e = math.log2(math.e)
    p_ref = torch.pow(2.0, run.s_raw * (run.scale * log2e) - (run.lse * log2e).unsqueeze(-1)).masked_fill(run.masked, 0.0)
    p_e4m3 = (p_ref * 256.0).to(_T_E4M3).float() * (1.0 / 256.0)
    grp = hq // hkv
    dP = torch.einsum("bhqd,bhkd->bhqk", run.quant["dO"].deq_d(), run.quant["v"].deq_d().repeat_interleave(grp, dim=1))
    delta = (run.o_f16.float() * run.dO_f16.float()).sum(-1)
    ds_fp32p = p_ref * (dP - delta.unsqueeze(-1)) * run.scale
    ds_e4m3p = p_e4m3 * (dP - delta.unsqueeze(-1)) * run.scale
    ulp = 2.0**-8 * float(ds_fp32p.abs().max())
    md_a, _c, out_a = _report("dS vs the fp32-P value", ds, ds_fp32p, ulp, 2.0**-7)
    md_b, _c, out_b = _report("dS vs the e4m3-P twin (must be FARTHER)", ds, ds_e4m3p, ulp, 2.0**-7)
    assert out_a == 0, f"{out_a} dS cells outside one bf16 rounding of the fp32-P value (max |diff| {md_a:g})"
    assert out_b > out_a and md_b > md_a, f"the e4m3-P twin is not farther ({out_b} outside, max {md_b:g}) -- P is being quantized before dS"


def _workspace_region(api, name):
    """``(offset, shape)`` of one scratch region from the plan's OWN carve (``prepared_sm107._regions`` walks ``_scratch_shapes()`` in order
    with ``ws_align``) -- never the arithmetic of one layout."""
    from cudnn.sdpa.fwd.api_dsl import ws_align

    off = 0
    for n, shape, dt in api._scratch_shapes():
        if n == name:
            return off, tuple(int(x) for x in shape)
        off += ws_align(math.prod(shape) * dt.itemsize)
    raise KeyError(name)


def _atom_bytes_to_scales(atoms, rows, cols):
    """F8_128x4 atoms ``[..., R/128, C/4, 512]`` (uint8) -> the logical scale matrix ``[..., R, C]``: the quantizer's byte rule inverted
    (``test/python/sdpa/mxfp8_quant.py::_swizzle_128x4``: scale (r, c) at byte (r % 32) * 16 + (r // 32) * 4 + c % 4 of atom (r // 128, c // 4))."""
    *lead, rt, ct, _ = atoms.shape
    assert (rt * 128, ct * 4) == (rows, cols), (atoms.shape, rows, cols)
    n = len(lead)
    v = atoms.reshape(*lead, rt, ct, 32, 4, 4)  # (rt, ct, rr = r % 32, rg = r // 32, cc)
    v = v.permute(*range(n), n + 0, n + 3, n + 2, n + 1, n + 4)  # (rt, rg, rr, ct, cc)
    return v.contiguous().reshape(*lead, rows, cols)


@requires_rubin
def test_p_b_ds_payloads_and_atoms_dequantize_to_the_oracles_quantized_ds(monkeypatch):
    """The P-b workspace contract, read back from the graph's workspace at the plan's region offsets on a dense (1, 2, 512, 512) cell:
    ``ds_dk`` [B, H, S_kv, S_q] e4m3 scaled per 32-q block with the ``sf_ds_dk`` atoms (kv tile, q tile; rows kv, columns q-blocks) and
    ``ds_dq`` scaled per 32-kv block with ``sf_ds_dq`` (q tile, kv tile; rows q, columns kv-blocks) -- dequantized, each must equal the
    oracle's own 1x32 quantization of the fp32 dS along the same axis (``quantize_to_mxfp8`` on ``P (dP - delta) * scale``: the ``_s``
    triple = along q = ds_dk, the ``_d`` triple = along kv = ds_dq).  Two mismatch shapes are told apart and reported: a WHOLE 32-block
    off (its E8M0 byte differs where the block's amax is not within 2^-16 of a power-of-two boundary, or >= 8 of its cells differ) is
    a scale-rule / layout mismatch and FAILS; an ISOLATED cell is an e4m3 midpoint flip (the kernel's fp32 dS and torch's differ by
    ~1e-6 relative) and is budgeted at 1e-4 of the cells, each within one e4m3 step at the block's scale.  The E8M0 rule itself is the
    oracle's ``e8m0_ceil`` (its scales are read back as exponent fields), never the kernel's helper."""
    from cudnn.sdpa.bwd import config_sm107 as cfg
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    _p_b(monkeypatch, sm=107)
    from cudnn.sdpa.bwd import prepared_sm107

    monkeypatch.undo()  # keep the policy, drop the faked device query: the real Rubin device answers below
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107

    monkeypatch.setattr(sm107, "MXFP8_DS_SF_POLICY", cfg.DS_SF_P_B)
    b, hq, hkv, sq, skv = 1, 2, 2, 512, 512
    run = _run_mxfp8(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv)
    api = _mxfp8_adapter(b=b, hq=hq, hkv=hkv, sq=sq, skv=skv)
    api.check_support()
    assert api._ds_block_scaled
    ws = run.workspace

    def region(name, dtype):
        off, shape = _workspace_region(api, name)
        n = math.prod(shape)
        return ws[off : off + n * dtype.itemsize].view(dtype).view(*shape)

    ds_dk = region("ds_ws", _T_E4M3).float()  # [B, H, S_kv, S_q]
    ds_dq = region("ds_dq", _T_E4M3).float()
    e_dk = _atom_bytes_to_scales(region("sf_ds_dk", torch.uint8), skv, sq // 32)  # [B, H, S_kv, S_q/32]
    e_dq = _atom_bytes_to_scales(region("sf_ds_dq", torch.uint8), sq, skv // 32)  # [B, H, S_q, S_kv/32]
    assert torch.isfinite(ds_dk).all() and torch.isfinite(ds_dq).all(), "unwritten / non-finite payload cells on a dense shape"
    assert int(e_dk.max()) <= 254 and int(e_dq.max()) <= 254, "an E8M0 NaN byte (255) in the dS atoms"
    # the oracle's fp32 dS = P (dP - delta) * scale from the fp32 P (the module's dS pin), then ITS 1x32 quantizer both ways
    log2e = math.log2(math.e)
    p_ref = torch.pow(2.0, run.s_raw * (run.scale * log2e) - (run.lse * log2e).unsqueeze(-1)).masked_fill(run.masked, 0.0)
    grp = hq // hkv
    dP = torch.einsum("bhqd,bhkd->bhqk", run.quant["dO"].deq_d(), run.quant["v"].deq_d().repeat_interleave(grp, dim=1))
    delta = (run.o_f16.float() * run.dO_f16.float()).sum(-1)
    ds_ref = p_ref * (dP - delta.unsqueeze(-1)) * run.scale  # [B, H, S_q, S_kv]
    q_d, sf_d, _sw_d, q_s, sf_s, _sw_s = quantize_to_mxfp8(ds_ref, b, hq, sq, skv, block_size=32, fp8_dtype=_T_E4M3)
    ref_dq = (q_d.float() * sf_d).view(b, hq, sq, skv)  # scaled along kv (the last dim): ds_dq's convention
    ref_dk = (q_s.float() * sf_s).view(b, hq, sq, skv)  # scaled along q: ds_dk's convention
    e_ref_dq = ((sf_d.view(b, hq, sq, skv)[..., ::32].contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)  # [B, H, S_q, S_kv/32]
    e_ref_dk = ((sf_s.view(b, hq, sq, skv)[:, :, ::32, :].contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)  # [B, H, S_q/32, S_kv]
    inv448 = torch.tensor(1.0 / 448.0, dtype=torch.float32, device="cuda")

    def near_boundary(amax):
        """A block whose amax * fp32(1/448) sits within 2^-16 relative of a power of two: the two sides' ~1e-6 fp32 disagreement can
        legitimately move e8m0_ceil by one there."""
        x = amax.float() * inv448
        lg = torch.log2(x.clamp_min(2.0**-126))
        return (lg - lg.round()).abs() <= 2.0**-16

    def check(tag, kern_deq, ref_deq, e_kern, e_ref, amax_blocks, block_axis):
        diff = (kern_deq - ref_deq).abs()
        bad = diff > 0
        n_bad = int(bad.sum())
        e_bad = e_kern != e_ref
        # whole-block verdicts: a byte mismatch away from a boundary, or >= 8 differing cells in one block
        cells_per_block = (
            bad.unflatten(block_axis, (-1, 32)).sum(block_axis + 1) if block_axis == bad.dim() - 1 else bad.unflatten(block_axis, (-1, 32)).sum(block_axis + 1)
        )
        byte_off = e_bad & ~near_boundary(amax_blocks)
        blocks_off = int((byte_off | (cells_per_block >= 8)).sum())
        step = (
            (ref_deq.abs() * 2.0**-3).clamp_min(2.0**-9) * torch.pow(2.0, (e_ref.float() - 127.0)).repeat_interleave(32, dim=block_axis).clamp_min(1.0) * 1.001
        )
        flips_too_big = int((bad & (diff > step + 2.0**-9)).sum())
        print(
            f"\n{tag}: {n_bad} of {bad.numel()} dequantized cells differ from the oracle's 1x32 quantization; E8M0 bytes differing {int(e_bad.sum())} of "
            f"{e_bad.numel()} ({int((e_bad & near_boundary(amax_blocks)).sum())} at a power-of-two boundary); whole blocks off {blocks_off}; max |diff| {float(diff.max()):.4g}"
        )
        assert blocks_off == 0, f"{tag}: {blocks_off} whole 32-blocks off -- a scale-rule or atom-layout mismatch, not a rounding flip"
        assert flips_too_big == 0, f"{tag}: {flips_too_big} differing cells exceed one e4m3 step at the block's scale (not a midpoint flip)"
        assert n_bad <= max(1, int(bad.numel() * 1e-4)), f"{tag}: {n_bad} isolated flips exceed the 1e-4 budget"

    # ds_dq: kernel [B, H, S_kv, S_q] -> [B, H, S_q, S_kv]; blocks along kv (the last dim); amax per (q row, kv block)
    kern_dq = ds_dq.transpose(-1, -2) * torch.pow(2.0, e_dq.float() - 127.0).repeat_interleave(32, dim=-1)
    amax_dq = ds_ref.unflatten(-1, (-1, 32)).abs().amax(-1)
    check("ds_dq (per 32-kv block)", kern_dq, ref_dq, e_dq, e_ref_dq, amax_dq, block_axis=3)
    # ds_dk: kernel [B, H, S_kv, S_q] -> [B, H, S_q, S_kv]; blocks along q (dim 2); the kernel's scales [B, H, S_kv, S_q/32] -> [B, H, S_q/32, S_kv]
    e_dk_t = e_dk.transpose(-1, -2)
    kern_dk = ds_dk.transpose(-1, -2) * torch.pow(2.0, e_dk_t.float() - 127.0).repeat_interleave(32, dim=2)
    amax_dk = ds_ref.unflatten(2, (-1, 32)).abs().amax(3)
    check("ds_dk (per 32-q block)", kern_dk, ref_dk, e_dk_t, e_ref_dk, amax_dk, block_axis=2)


@requires_rubin
def test_p_b_runs_no_dequant_pass_and_p_c_runs_two(monkeypatch):
    """The launch census of one execute (torch.profiler / CUPTI): the bf16-dS chain runs the two SF-aware dequant kernels (q_T, k_T)
    ahead of its bf16 GEMMs; the block-scaled chain runs NONE -- its GEMMs read the e4m3 payloads and scale factors directly.  Stage-3
    launches: dK once per chunk on both; dQ once per GQA group member on the block-scale arm (its B scale-factor descriptor is
    indexed per A / C head) and, under ``DQ_SINGLE_LAUNCH``, once per chunk on the bf16 twin's plain rendering."""
    from torch.profiler import ProfilerActivity, profile

    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107, config_sm107 as cfg

    counts = {}
    for policy in (cfg.DS_SF_P_C, cfg.DS_SF_P_B):
        monkeypatch.setattr(sm107, "MXFP8_DS_SF_POLICY", policy)
        run = _run_mxfp8(hq=4, hkv=2, sq=512, skv=512, check=False)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            run.graph.execute(run.pack, run.workspace)
            torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages() if getattr(e, "device_time_total", 0) > 0]
        rows = {e.key: e.count for e in prof.key_averages() if getattr(e, "device_time_total", 0) > 0}
        counts[policy] = dict(
            dequant=sum(c for k, c in rows.items() if "dequant_mxfp8_to_bf16" in k),
            gemm=sum(c for k, c in rows.items() if "bprop_matmul_bh_sm100_kernel" in k),
            main=sum(
                c for k, c in rows.items() if "__kernel_TensorMap" in k
            ),  # the main kernel under the cuDNN name prefix (cudnn_kernel__kernel_TensorMap...)
        )
        print(f"\npolicy {policy}: {counts[policy]} from {names}")
    group = 4 // 2  # the cell's H_q / H_kv
    assert counts[cfg.DS_SF_P_C]["dequant"] == 2, counts
    assert counts[cfg.DS_SF_P_B]["dequant"] == 0 and cfg.DS_SF_POLICY_DEFAULT == cfg.DS_SF_P_B, "the shipped default launches no dequant pass"
    assert counts[cfg.DS_SF_P_B]["gemm"] == 1 + group, counts  # dK + one dQ launch per GQA group member (the block-scale arm)
    assert counts[cfg.DS_SF_P_C]["gemm"] == 1 + (1 if sm107.DQ_SINGLE_LAUNCH else group), counts  # the plain rendering's single dQ launch
    assert counts[cfg.DS_SF_P_B]["main"] == counts[cfg.DS_SF_P_C]["main"] >= 1, counts


@requires_rubin
def test_graph_routes_to_the_row_when_opted_in_and_declines_typed_otherwise(monkeypatch):
    """The d256 ``sdpa_mxfp8_backward`` graph on cc 10.7 lists THIS row (bracket-form name) and no other; with FROST opted out the
    graph is admitted at validate / build and ``create_execution_plans`` raises the typed ``cudnnGraphNotSupportedError`` -- cuDNN
    9.27 has no MXFP8 d=256 backward kernel on smVersion 1070 (verified 2026-09-29), so the row is the sole provider."""
    g, _t, _outs = _mxfp8_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    assert any(n == _ENGINE or n.startswith(_ENGINE + "[") for n in names), names
    assert not any(n.startswith("sdpa_bwd_sm100_mxfp8") for n in names), names
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    g2, _t, _outs = _mxfp8_graph()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g2.create_execution_plans([cudnn.heur_mode.A])


@requires_rubin
def test_unserved_amax_graph_declines_typed_end_to_end():
    """The backend-shaped graph (amax_dQ / dK / dV real) has no provider on Rubin: a typed ``cudnnGraphNotSupportedError`` at
    plan creation, never a bare error and never garbage amax."""
    g, _t, _outs = _mxfp8_graph(declare_amax=True)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g.create_execution_plans([cudnn.heur_mode.A])
