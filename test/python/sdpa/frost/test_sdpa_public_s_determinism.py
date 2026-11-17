# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded E4M3/BF16 public-S forward determinism; backward is not qualified here."""

import pytest

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell, requires_blackwell_geforce
from sdpa_public_s_determinism_cases import cases, check_public_s_batch_and_golden

pytestmark = [pytest.mark.L0, requires_dsl]


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("case", cases("sm100"))
@pytest.mark.parametrize("source", ["buffer", "omitted"], ids=["device-pair", "default-pair"])
def test_sm100_public_s_batch_and_golden(case, source):
    check_public_s_batch_and_golden(case, source)


@requires_blackwell_geforce
@pytest.mark.parametrize("case", cases("sm120"))
@pytest.mark.parametrize("source", ["buffer", "omitted"], ids=["device-pair", "default-pair"])
def test_sm120_public_s_batch_and_golden(case, source):
    check_public_s_batch_and_golden(case, source)


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("source", ["buffer", "omitted"], ids=["device-pair", "default-pair"])
def test_sm100_public_s_cga1_thd_requires_supported_geometry(source):
    case = ("sm100", "d128_fp8_multi_q", 128, 128, 1, 128, False, True)
    with pytest.raises(NotImplementedError, match="other THD graphs run the cga2 prefill tile"):
        check_public_s_batch_and_golden(case, source)
