# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Masked dense D192 head envelopes use the native per-tensor FP8 path."""

import math

import pytest
import torch

from frost_test_utils import requires_blackwell, requires_dsl
from test_utils import torch_fork_set_rng
import test_sdpa_fwd_fp8_sm100 as shared

pytestmark = [pytest.mark.L0, requires_blackwell, requires_dsl]


@pytest.mark.parametrize(
    "d,dv,dtype,mask,pack",
    [
        (144, 128, "e4m3", "causal", False),
        (144, 128, "e5m2", "causal_br", True),
        (160, 112, "e4m3", "swa", True),
        (160, 112, "e5m2", "causal", False),
        (176, 64, "e4m3", "causal_br", False),
        (176, 64, "e5m2", "swa", True),
        (192, 16, "e4m3", "causal", True),
        (192, 16, "e5m2", "causal_br", False),
    ],
)
@torch_fork_set_rng(seed=827)
def test_fp8_d192_head_envelope_masks(d, dv, dtype, mask, pack):
    result = shared._run(
        2,
        8,
        2,
        129,
        256,
        dtype,
        torch.bfloat16,
        scale=1 / math.sqrt(d),
        sdpa_kwargs=shared._MASKS[mask],
        d_qk=d,
        d_v=dv,
        pack_gqa=pack,
        seq_lens_kv=[256, 217],
        return_lse=True,
        poison_tmem_before_execute=True,
    )
    shared._check(result.output, result.reference, torch.bfloat16, dtype, result.amax, result.reference_amax)
    torch.testing.assert_close(result.stats, result.reference_stats, atol=shared._half_atol(dtype), rtol=3e-2)
