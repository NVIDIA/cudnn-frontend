# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-arch, per-kernel gate of the sm100 exp2 MUFU / FMA split (``TemplateParams.exp2_fma_split``).

The split (the ``_E2E_*`` block of ``sm100/prefill_d128_mxfp8.py``, ``prefill_d128_fp8.py`` and
``prefill_d192_d128_f16.py``) trades MUFU.EX2 pipe-time for FMA pipe-time, so its sign follows the part's
MUFU rate: MEASURED 16 elements/clk/SM on cc 10.0 (B200) and 32 on cc 10.7 (Rubin, -9..-10 %); cc 10.3
(GB300) DOCUMENTS the same doubled exp2 rate.  And it is claimed per KERNEL: on B200 it MEASURED a win on the
d128 MXFP8 (+7.8 %), d128 per-tensor FP8 (+4.5 %) and d192x128 bf16 (+1.9 %) chart layers and a LOSS or a
marginal result on d128 bf16 (causal -1.9 / -2.4 %), d192x128 FP8 (causal -1.6 %) and d192x128 MXFP8
(-3.3..-4.0 %).  The sm100 engine rows serve cc 10.0 AND 10.3 with one kernel file, so the adapter sets the
field from the BUILD device and the (quantization kind, flavor) of the build -- ON only where both were
measured -- and each kernel folds it at trace time.  Host-only: no GPU, no compile.  The per-arch SASS pins
(MUFU.EX2 194 on sm_100a with the gate on, develop's 258 on sm_103a with it off) live in the per-kernel
suites (``test_sdpa_fwd_mxfp8_sm100.py``, ``test_sdpa_fwd_fp8_sm100.py``, ``test_sdpa_fwd_dsl_sm100.py``).
"""

import pytest

from frost_test_utils import requires_dsl

from cudnn.sdpa.fwd.api_dsl import _EXP2_FMA_SPLIT_CC, _EXP2_FMA_SPLIT_KERNELS, _exp2_fma_split_for, _quant_kind
from cudnn.sdpa.fwd.config_sm100 import TemplateParams

pytestmark = [pytest.mark.L0]

# Every cc the SM100-family adapter admits (api_dsl.check_support: cc 10.0 / 10.3 Blackwell, 10.7 Rubin), plus
# the SM120 line and a hypothetical later sm10x part -- only the MEASURED cc may switch the split on.
_CCS = [(10, 0), (10, 3), (10, 7), (11, 0), (12, 0)]
_KINDS = ["mxfp8", "fp8", "f16"]
_FLAVORS = [(128, 128), (192, 128), (256, 256), (512, 512)]
# The three kernels that carry the split AND measured a win on B200 (2026-09-22, A/B/A x3, CUPTI medians).
_ON = {("mxfp8", (128, 128)), ("fp8", (128, 128)), ("f16", (192, 128))}
# Measured on B200 and deliberately OFF: d128 bf16 (dense +3.9 % but causal -1.9 / -2.4 %), d192x128 fp8 (dense
# +1 %, causal -1.6 %), d192x128 mxfp8 (-3.3..-4.0 % dense; it already carries its own exp2 emulation).
_MEASURED_OFF = {("f16", (128, 128)), ("fp8", (192, 128)), ("mxfp8", (192, 128))}


def test_quant_kind_is_the_kernel_file_spelling():
    """``_quant_kind`` maps the adapter's (fp8, pertensor) pair onto the tag ``_load_sm100_kernel_module`` keys
    the kernel file by -- the gate and the loader must agree on which kernel a build is."""
    assert _quant_kind(fp8=True, pertensor=True) == "fp8"
    assert _quant_kind(fp8=True, pertensor=False) == "mxfp8"
    assert _quant_kind(fp8=False, pertensor=False) == "f16"
    assert _quant_kind(fp8=False, pertensor=True) == "f16", "pertensor is meaningless without fp8 and must not change the kind"


def test_exp2_fma_split_is_claimed_only_on_the_measured_cc():
    """Positive gate: exactly cc 10.0 (B200, where every win was measured) turns the split on for the kernels that
    carry it; cc 10.3 (GB300, 2x MUFU.EX2 rate DOCUMENTED) and cc 10.7 (Rubin, 2x MEASURED, split -9..-10 %) get the
    develop all-MUFU kernel, and so does any cc nobody measured."""
    assert _EXP2_FMA_SPLIT_CC == frozenset({(10, 0)}), "widening the cc set is a per-cc A/B, never a default"
    for cc in _CCS:
        for kind, flavor in _ON:
            assert _exp2_fma_split_for(cc, kind=kind, flavor=flavor) is (cc == (10, 0)), (cc, kind, flavor)


def test_exp2_fma_split_is_claimed_only_by_the_kernels_that_measured_a_win():
    """The full (cc, kind, flavor) truth table: ON is exactly cc 10.0 x the three measured kernels.  The three
    kernels that MEASURED a loss (or a marginal win) on B200 stay off even on cc 10.0, as does every flavor that was
    never measured -- a sibling port has to claim the split explicitly, per kernel and per arch."""
    assert _EXP2_FMA_SPLIT_KERNELS == frozenset(_ON), "widening the kernel set is a per-kernel A/B on the chart layer, never a default"
    assert not (_MEASURED_OFF & _ON)
    for cc in _CCS:
        for kind in _KINDS:
            for flavor in _FLAVORS:
                want = cc == (10, 0) and (kind, flavor) in _ON
                assert _exp2_fma_split_for(cc, kind=kind, flavor=flavor) is want, (cc, kind, flavor)
    for kind, flavor in _MEASURED_OFF:
        assert _exp2_fma_split_for((10, 0), kind=kind, flavor=flavor) is False, (kind, flavor)
    # A flavor is passed as whatever sequence the adapter holds; the lookup normalises it.
    assert _exp2_fma_split_for((10, 0), kind="fp8", flavor=[128, 128]) is True


def test_template_params_default_is_the_all_mufu_kernel():
    """A template loaded without the adapter (``python kernels/sm100/prefill_d128_mxfp8.py``, a probe that builds
    ``TemplateParams()`` itself) runs the develop exp2 path: the split is opt-in per build, never a default."""
    assert TemplateParams().exp2_fma_split is False
    assert TemplateParams(exp2_fma_split=True).exp2_fma_split is True
    # The field is part of the frozen record, hence of the template-module cache key: the two specializations are
    # two distinct modules, never one module reused for the other cc.
    assert TemplateParams(exp2_fma_split=True) != TemplateParams(exp2_fma_split=False)
    assert hash(TemplateParams(exp2_fma_split=True)) != hash(TemplateParams(exp2_fma_split=False))


# (kind, flavor, loader kwargs, TemplateParams kwargs of the production build the SASS pins compile)
_GATED_KERNELS = [
    pytest.param("mxfp8", (128, 128), dict(fp8=True, pertensor=False), dict(dtype_qkv=0, dtype_o=2, qh_per_kh=3), id="d128_mxfp8"),
    pytest.param("fp8", (128, 128), dict(fp8=True, pertensor=True), dict(dtype_qkv=0, dtype_o=0, qh_per_kh=8, emit_amax_o=True), id="d128_fp8"),
    pytest.param("f16", (192, 128), dict(fp8=False, pertensor=False), dict(dtype_qkv=2, dtype_o=2, qh_per_kh=1), id="d192x128_bf16"),
]


@requires_dsl
@pytest.mark.parametrize("kind,flavor,load_kw,params_kw", _GATED_KERNELS)
@pytest.mark.parametrize("enabled", [True, False], ids=["cc100_on", "cc103_off"])
def test_gated_kernel_module_folds_the_split_on_the_param(kind, flavor, load_kw, params_kw, enabled):
    """Loading each gated kernel template with the field set / cleared yields the split / develop constants: with the
    gate on 32 of 128 columns (pairs 6, 7 of every 8 in each 64-wide chunk = columns 12..15 of every 16) are routed to
    ``exp2_emul_pair``; off, ``_E2E_PAIRS`` are empty and ``_E2E_EMULATED_COLS`` is 0, which is what each SASS pin
    derives its MUFU.EX2 expectation from (2 x 97 on, 2 x 129 off).  Trace-time only: the module executes on any
    host, no device and no compile."""
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module

    params = TemplateParams(cta_mma=2, fused_ldtm_stat=(kind == "mxfp8" and not enabled), exp2_fma_split=enabled, **params_kw)
    mod = _load_sm100_kernel_module(flavor, params, rubin=False, **load_kw)
    assert mod._E2E_ENABLED is enabled
    assert mod.CFG.TILE_N == 128 and len(mod._E2E_PAIRS) == 2, "the pattern below assumes two 64-wide chunks of a 128-wide row"
    if enabled:
        assert mod._E2E_EMULATED_COLS == 32
        assert all(sorted(p) == [6, 7, 14, 15, 22, 23, 30, 31] for p in mod._E2E_PAIRS), mod._E2E_PAIRS
    else:
        assert mod._E2E_EMULATED_COLS == 0
        assert all(len(p) == 0 for p in mod._E2E_PAIRS), mod._E2E_PAIRS
    assert 2 * sum(len(p) for p in mod._E2E_PAIRS) == mod._E2E_EMULATED_COLS


@requires_dsl
@pytest.mark.parametrize("enabled", [True, False], ids=["cc100_on", "cc103_off"])
def test_d192_f16_keeps_its_causal_tail_emulation_under_the_gate(enabled):
    """The d192x128 f16 kernel's dense top-left causal arm carried a tail exp2 emulation BEFORE the split
    (``_exp2_mixed_late``: the last 34 columns of chunk 1 on ``ex2_emulation_2``).  The measured form of the split
    REPLACES it on every mask arm with the gate on; with the gate off the develop causal arm -- tail emulation
    included -- must still be what the module traces (its own arch gate is a separate follow-up)."""
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module

    params = TemplateParams(dtype_qkv=2, dtype_o=2, cta_mma=2, qh_per_kh=1, window_right=0, sched_policy=2, exp2_fma_split=enabled)
    mod = _load_sm100_kernel_module((192, 128), params, fp8=False, pertensor=False, rubin=False)
    assert mod._E2E_ENABLED is enabled
    assert mod._DENSE_CAUSAL_PUBLIC_EXP2_MIX is True, "this record is the dense top-left causal MHA arm"
    assert callable(mod._exp2_mixed_late) and callable(mod.ex2_emulation_2), "the pre-existing tail emulation must stay importable for the gate-off arm"
    assert mod._REUSE_BMM2_ISSUE_ELECTION is True, "the causal arm's BMM2 issue tweaks are independent of the split"
