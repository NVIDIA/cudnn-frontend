# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pins for ``sdpa.fp8.assert_amax_o``: the quantized harnesses' check of the graph's ``Amax_O`` output.

GPU-free (CPU tensors).  Each case is one way an engine can get Amax_O wrong -- never written, left at a reset value,
taken from the wrong quantity (a dead row, a partial), or simply off -- and the one legitimate deviation (a P code
flip on the dominant key of the largest element) that must stay accepted.
"""

import math

import pytest
import torch

from sdpa.fp8 import BACKEND_AMAX_O_ISSUE, assert_amax_o, p_code_step

pytestmark = pytest.mark.L0

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2


def _amax(value):
    return torch.tensor([[[[value]]]], dtype=torch.float32)


def _o(*values, dtype=torch.float16):
    return torch.tensor(values, dtype=torch.float32).to(dtype)


def test_p_code_step_is_one_code_of_the_input_format():
    assert p_code_step(E4M3) == 0.125 and p_code_step(E5M2) == 0.25


def test_exact_agreement_passes_with_and_without_the_stored_o():
    assert_amax_o(_amax(1.5), 1.5, torch_itype=E4M3, torch_otype=torch.float16)
    assert_amax_o(_amax(1.5), 1.5, torch_itype=E4M3, torch_otype=torch.float16, o_gpu=_o(0.25, -1.5, 0.75))


def test_never_written_output_fails():
    with pytest.raises(AssertionError, match="never written"):
        assert_amax_o(_amax(float("nan")), 1.5, torch_itype=E4M3, torch_otype=torch.float16)


def test_reset_value_against_a_nonzero_reference_fails():
    with pytest.raises(AssertionError, match="exceeds one P code step"):
        assert_amax_o(_amax(0.0), 1.0273, torch_itype=E4M3, torch_otype=torch.float16, o_gpu=_o(1.0273))


@pytest.mark.parametrize("itype", [E4M3, E5M2], ids=["e4m3", "e5m2"])
def test_one_p_code_flip_on_the_largest_element_is_accepted_and_more_is_not(itype):
    step = p_code_step(itype)
    ref = 1.6
    # the stored O follows the kernel's amax: the deviation is between kernel and reference, not kernel and itself
    assert_amax_o(_amax(ref * (1 + 0.9 * step)), ref, torch_itype=itype, torch_otype=torch.float16, o_gpu=_o(ref * (1 + 0.9 * step)))
    assert_amax_o(_amax(ref * (1 - 0.9 * step)), ref, torch_itype=itype, torch_otype=torch.float16, o_gpu=_o(ref * (1 - 0.9 * step)))
    with pytest.raises(AssertionError, match="exceeds one P code step"):
        assert_amax_o(_amax(ref * (1 + 1.1 * step)), ref, torch_itype=itype, torch_otype=torch.float16, o_gpu=_o(ref * (1 + 1.1 * step)))


def test_the_reference_bound_is_relative_so_a_low_amplitude_error_still_fails():
    # a doubled or zeroed O at amplitude 0.12 (the old 5 %-of-max(amax, 1) floor accepted both)
    with pytest.raises(AssertionError):
        assert_amax_o(_amax(0.24), 0.12, torch_itype=E5M2, torch_otype=torch.float16)
    with pytest.raises(AssertionError):
        assert_amax_o(_amax(0.0), 0.12, torch_itype=E5M2, torch_otype=torch.float16)


def test_dead_rows_in_the_amax_fail_against_an_all_zero_output():
    # the backend's paged FP8 forward with seq_len_q = [0]: O is all zero, Amax_O came back 0.137
    with pytest.raises(AssertionError, match="exceeds one P code step"):
        assert_amax_o(_amax(0.1367), 0.0, torch_itype=E4M3, torch_otype=E5M2, o_gpu=_o(0.0, 0.0, dtype=E5M2))


def test_amax_above_the_stored_o_fails_even_inside_the_reference_bound():
    # +5 % on fp16 O: inside one e4m3 code step of the reference, far outside fp16 rounding -> not the stored values
    with pytest.raises(AssertionError, match="stored O reaches only"):
        assert_amax_o(_amax(1.05), 1.0, torch_itype=E4M3, torch_otype=torch.float16, o_gpu=_o(1.0, -0.5))


def test_stored_o_above_the_amax_fails():
    with pytest.raises(AssertionError, match="but the kernel reported Amax_O"):
        assert_amax_o(_amax(0.9), 1.0, torch_itype=E4M3, torch_otype=torch.float16, o_gpu=_o(1.0))


def test_stored_o_agrees_within_its_own_dtype_rounding():
    # an fp8 O rounds the max element by up to half a code spacing (6.25 % e4m3): Amax_O is the pre-cast value
    pre_cast = 1.03
    stored = _o(pre_cast, dtype=E4M3)  # 1.0 in e4m3
    assert_amax_o(_amax(pre_cast), pre_cast, torch_itype=E4M3, torch_otype=E4M3, o_gpu=stored)
    # the same 3 % gap on an fp16 O is 60 fp16 code spacings: not rounding
    with pytest.raises(AssertionError, match="stored O reaches only"):
        assert_amax_o(_amax(pre_cast), pre_cast, torch_itype=E4M3, torch_otype=torch.float16, o_gpu=_o(1.0))


def test_saturated_output_drops_only_the_upper_bound():
    # per-tensor FP8: the scale was built from the reference amax, a kernel max above it clamps at the top code
    ref, kernel = 1.0, 1.08
    stored = _o(ref)  # clamped: the stored max reads the reference amax, below the kernel's
    with pytest.raises(AssertionError, match="stored O reaches only"):
        assert_amax_o(_amax(kernel), ref, torch_itype=E4M3, torch_otype=torch.float16, o_gpu=stored)
    assert_amax_o(_amax(kernel), ref, torch_itype=E4M3, torch_otype=torch.float16, o_gpu=stored, saturated=True)
    # ... never the lower bound: a stored O above the reported amax is wrong, saturated or not
    with pytest.raises(AssertionError, match="but the kernel reported Amax_O"):
        assert_amax_o(_amax(0.9), ref, torch_itype=E4M3, torch_otype=torch.float16, o_gpu=stored, saturated=True)


def test_known_issue_turns_a_failure_into_an_xfail_and_leaves_a_pass_alone():
    with pytest.raises(pytest.xfail.Exception, match="cuDNN backend Amax_O"):
        assert_amax_o(_amax(0.0), 1.0, torch_itype=E4M3, torch_otype=torch.float16, known_issue=BACKEND_AMAX_O_ISSUE)
    assert_amax_o(_amax(1.0), 1.0, torch_itype=E4M3, torch_otype=torch.float16, known_issue=BACKEND_AMAX_O_ISSUE)


def test_zero_output_with_zero_reference_passes():
    assert_amax_o(_amax(0.0), 0.0, torch_itype=E5M2, torch_otype=torch.bfloat16, o_gpu=_o(0.0, 0.0, dtype=torch.bfloat16))
    assert math.isfinite(p_code_step(E5M2))
