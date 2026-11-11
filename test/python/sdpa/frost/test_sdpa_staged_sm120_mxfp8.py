# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 MXFP8 binds aligned declared Q/K/V/O layouts without layout copies.

The shared native stride suite covers asymmetric head dimensions, current scale
buffers, caller workspace, invalid bindings and wide physical addresses.
"""

import pytest
import torch

import native_forward_mxfp8_cases as shared
from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]

# Q/K and V scale tiles of different sizes, both ways round, on both templates.
_ASYMMETRIC_DIMS = [(192, 128), (96, 160), (384, 320)]


@pytest.mark.parametrize("d,dv", _ASYMMETRIC_DIMS)
def test_sm120_mxfp8_staged_uses_pointer_host_and_current_scales(d, dv, monkeypatch):
    shared.test_mxfp8_native_uses_pointer_host_and_current_scales(d, dv, monkeypatch)


@pytest.mark.parametrize("d", [128, 512])
@pytest.mark.parametrize("split", [1, 4])
def test_sm120_mxfp8_staged_fp8_output_without_optional_outputs(d, split):
    shared.test_mxfp8_native_fp8_output_without_optional_outputs(d, d, split)


@pytest.mark.parametrize("role", ["sf_q", "sf_k", "sf_v"])
@pytest.mark.parametrize("bad", ["missing", "short"])
def test_sm120_mxfp8_staged_sf_rejects_before_copy(role, bad, monkeypatch):
    shared.test_mxfp8_native_sf_rejects_before_launch(role, bad, monkeypatch)


@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d", [128, 512])
@pytest.mark.parametrize("role,product", [("q", False), ("o", True)])
def test_sm120_mxfp8_staged_physical_wide_batch_stride(d, role, product):
    # Return the allocator's cached blocks from earlier cases: only memory in use should skip.
    torch.cuda.empty_cache()
    shared.test_mxfp8_native_physical_wide_batch_stride(d, d, role, product)
