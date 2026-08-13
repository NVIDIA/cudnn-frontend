# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM107 (Rubin) routing of the per-tensor FP8 d128 SDPA kernel.

The adapter routes cc10.7 per-tensor-FP8 graphs to the SM107 sibling module
(``prefill_d128_fp8_sm107.py``), which bakes the Rubin dense-FP8 K=64 MMA
geometry; Blackwell keeps the untouched SM100 module. These tests pin the
routing and both modules' derived constants — device-independent (everything
here happens before any compile). End-to-end coverage rides the existing
``test_sdpa_fwd_fp8_sm100.py`` suite, which exercises whichever SM10x part
is present (Rubin included) through the same adapter.
"""

import pytest

from frost_test_utils import requires_dsl

from cudnn.sdpa.fwd.config_sm100 import TemplateParams

pytestmark = [pytest.mark.L0, requires_dsl]

_E4M3, _BF16_OUT = 0, 2


def _load(rubin):
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module

    return _load_sm100_kernel_module(
        (128, 128),
        TemplateParams(dtype_qkv=_E4M3, dtype_o=_BF16_OUT),
        fp8=True,
        pertensor=True,
        rubin=rubin,
    )


def test_sm107_module_bakes_rubin_geometry():
    mod = _load(rubin=True)
    assert "sm107" in mod.__name__
    # Dense-FP8 K=64 steps + the 9-stage KV ring (GR100 SMEM).
    assert (mod.CFG.TILE_K_HW_BMM1, mod.CFG.TILE_K_HW_BMM2) == (64, 64)
    assert mod.CFG.STAGES_KV == 9
    assert mod.NUM_KPHASES_PV == 2  # TILE_N / 64 — in lockstep with the idesc


def test_sm100_module_unchanged():
    mod = _load(rubin=False)
    assert "sm100" in mod.__name__
    assert (mod.CFG.TILE_K_HW_BMM1, mod.CFG.TILE_K_HW_BMM2) == (32, 32)
    assert mod.CFG.STAGES_KV == 4
    assert mod.NUM_KPHASES_PV == 4
