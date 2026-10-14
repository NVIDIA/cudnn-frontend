# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 MXFP8 conversion plans stage compact Q/K/V/O copies around the prepared MXFP8 plan.

Runs the shared MXFP8 staged suite (``test_sdpa_staged_forward_mxfp8``) on the SM120 row -- head dims up to 256 and
both in (256, 512] on the d512 template, including the zero-padded envelope: the plan binds current storage and
scale tensors through the pointer host,
keeps its workspace window, rejects invalid scale tensors before any copy, and addresses wide physical strides.
"""

import pytest
import torch

import test_sdpa_staged_forward_mxfp8 as shared
from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]

# Q/K and V scale tiles of different sizes, both ways round, on both templates.
_ASYMMETRIC_DIMS = [(192, 128), (96, 160), (384, 320)]


@pytest.mark.parametrize("d,dv", _ASYMMETRIC_DIMS)
def test_sm120_mxfp8_staged_uses_pointer_host_and_current_scales(d, dv, monkeypatch):
    shared.test_mxfp8_staged_uses_pointer_host_and_current_scales(d, dv, monkeypatch)


@pytest.mark.parametrize("d", [128, 512])
@pytest.mark.parametrize("split", [1, 4])
def test_sm120_mxfp8_staged_fp8_output_without_optional_outputs(d, split):
    shared.test_mxfp8_staged_fp8_output_without_optional_outputs(d, d, split)


@pytest.mark.parametrize("role", ["sf_q", "sf_k", "sf_v"])
@pytest.mark.parametrize("bad", ["missing", "short"])
def test_sm120_mxfp8_staged_sf_rejects_before_copy(role, bad, monkeypatch):
    shared.test_mxfp8_staged_sf_rejects_before_copy(role, bad, monkeypatch)


@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d", [128, 512])
@pytest.mark.parametrize("role,product", [("q", False), ("o", True)])
def test_sm120_mxfp8_staged_physical_wide_batch_stride(d, role, product):
    # Return the allocator's cached blocks from earlier cases: only memory in use should skip.
    torch.cuda.empty_cache()
    shared.test_mxfp8_staged_physical_wide_batch_stride(d, d, role, product)
