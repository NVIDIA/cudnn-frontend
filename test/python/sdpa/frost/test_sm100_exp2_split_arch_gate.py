# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-arch gate of the sm100 d128 MXFP8 exp2 MUFU / FMA split (``TemplateParams.exp2_fma_split``).

The split (``sm100/prefill_d128_mxfp8.py``, the ``_E2E_*`` block) trades MUFU.EX2 pipe-time for FMA
pipe-time, so its sign follows the part's MUFU rate: MEASURED 16 elements/clk/SM on cc 10.0 (B200,
+10.9 %) and 32 on cc 10.7 (Rubin, -9..-10 %); cc 10.3 (GB300) DOCUMENTS the same doubled exp2 rate.
The sm100 engine rows serve cc 10.0 AND 10.3 with one kernel file, so the adapter sets the field from
the BUILD device -- ON only on the cc the split was measured on -- and the kernel folds it at trace
time.  Host-only: no GPU, no compile.  The per-arch SASS pins (MUFU.EX2 194 on sm_100a with the gate
on, develop's 258 on sm_103a with it off) live in ``test_sdpa_fwd_mxfp8_sm100.py``.
"""

import pytest

from frost_test_utils import requires_dsl

from cudnn.sdpa.fwd.api_dsl import _EXP2_FMA_SPLIT_CC, _EXP2_FMA_SPLIT_FLAVORS, _exp2_fma_split_for
from cudnn.sdpa.fwd.config_sm100 import TemplateParams

pytestmark = [pytest.mark.L0]

# Every cc the SM100-family adapter admits (api_dsl.check_support: cc 10.0 / 10.3 Blackwell, 10.7 Rubin), plus
# the SM120 line and a hypothetical later sm10x part -- only the MEASURED cc may switch the split on.
_CCS = [(10, 0), (10, 3), (10, 7), (11, 0), (12, 0)]
_MXFP8_FLAVORS = [(128, 128), (192, 128), (256, 256), (512, 512)]


def test_exp2_fma_split_is_claimed_only_on_the_measured_cc():
    """Positive gate: exactly cc 10.0 (B200, where the +10.9 % was measured) turns the split on for the d128 MXFP8
    kernel; cc 10.3 (GB300, 2x MUFU.EX2 rate DOCUMENTED) and cc 10.7 (Rubin, 2x MEASURED, split -9..-10 %) get the
    develop all-MUFU kernel, and so does any cc nobody measured."""
    assert _EXP2_FMA_SPLIT_CC == frozenset({(10, 0)}), "widening the cc set is a per-cc A/B, never a default"
    for cc in _CCS:
        assert _exp2_fma_split_for(cc, mxfp8=True, flavor=(128, 128)) is (cc == (10, 0)), cc


def test_exp2_fma_split_is_claimed_only_by_the_kernel_that_carries_it():
    """Only the d128 MXFP8 kernel reads the field: the other MXFP8 flavors and every per-tensor FP8 / f16 build get
    False even on cc 10.0, so a sibling port has to claim the split explicitly (per kernel, per arch)."""
    assert _EXP2_FMA_SPLIT_FLAVORS == frozenset({(128, 128)})
    for flavor in _MXFP8_FLAVORS:
        assert _exp2_fma_split_for((10, 0), mxfp8=True, flavor=flavor) is (flavor == (128, 128)), flavor
        assert _exp2_fma_split_for((10, 0), mxfp8=False, flavor=flavor) is False, flavor


def test_template_params_default_is_the_all_mufu_kernel():
    """A template loaded without the adapter (``python kernels/sm100/prefill_d128_mxfp8.py``, a probe that builds
    ``TemplateParams()`` itself) runs the develop exp2 path: the split is opt-in per build, never a default."""
    assert TemplateParams().exp2_fma_split is False
    assert TemplateParams(exp2_fma_split=True).exp2_fma_split is True
    # The field is part of the frozen record, hence of the template-module cache key: the two specializations are
    # two distinct modules, never one module reused for the other cc.
    assert TemplateParams(exp2_fma_split=True) != TemplateParams(exp2_fma_split=False)
    assert hash(TemplateParams(exp2_fma_split=True)) != hash(TemplateParams(exp2_fma_split=False))


@requires_dsl
@pytest.mark.parametrize("enabled", [True, False], ids=["cc100_on", "cc103_off"])
def test_d128_mxfp8_module_folds_the_split_on_the_param(enabled):
    """Loading the kernel template with the field set / cleared yields the split / develop constants: with the gate
    on 32 of 128 columns (8 pairs per 64-wide chunk) are routed to ``exp2_emul_pair``; off, ``_E2E_PAIRS`` are empty
    and ``_E2E_EMULATED_COLS`` is 0, which is what the SASS pin derives its MUFU.EX2 expectation (258) from.  Trace-
    time only: the module executes on any host, no device and no compile."""
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module

    params = TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=2, qh_per_kh=3, fused_ldtm_stat=not enabled, exp2_fma_split=enabled)
    mod = _load_sm100_kernel_module((128, 128), params, fp8=True, pertensor=False, rubin=False)
    assert mod._E2E_ENABLED is enabled
    assert mod.CFG.TILE_N == 128 and len(mod._E2E_PAIRS) == 2, "the pattern below assumes two 64-wide chunks of a 128-wide row"
    if enabled:
        assert mod._E2E_EMULATED_COLS == 32
        # columns 12..15 of every 16 -> pairs 6, 7 of every 8 pairs, in both chunks
        assert all(sorted(p) == [6, 7, 14, 15, 22, 23, 30, 31] for p in mod._E2E_PAIRS), mod._E2E_PAIRS
    else:
        assert mod._E2E_EMULATED_COLS == 0
        assert all(len(p) == 0 for p in mod._E2E_PAIRS), mod._E2E_PAIRS
    assert 2 * sum(len(p) for p in mod._E2E_PAIRS) == mod._E2E_EMULATED_COLS
