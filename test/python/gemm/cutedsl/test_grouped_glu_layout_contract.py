# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLU framework rejection must precede tensor inspection and CUDA setup."""

import pytest

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("entry_point", ["class", "wrapper"])
def test_gate_then_up_rejects_non_torch_before_tensor_inspection(monkeypatch, entry_point):
    from cudnn.gemm.cutedsl.grouped.glu import api

    # An opaque object supplies no tensor metadata. Framework detection alone
    # must produce the documented error without importing JAX or allocating GPU
    # storage. Default interleaved class construction remains supported.
    sample = object()
    monkeypatch.setattr(api, "detect_framework", lambda _: "jax")
    samples = dict(
        sample_a=sample,
        sample_c=sample,
        sample_d=sample,
        sample_sfa=None,
        sample_padded_offsets=sample,
        sample_alpha=sample,
        sample_d_col=None,
    )
    with pytest.raises(NotImplementedError, match="gate_then_up GLU currently supports torch inputs only"):
        if entry_point == "class":
            api.GroupedGemmGluSm100(**samples, glu_layout="gate_then_up")
        else:
            api.grouped_gemm_glu_wrapper_sm100(
                a_tensor=sample,
                sfa_tensor=None,
                padded_offsets=sample,
                alpha_tensor=sample,
                glu_layout="gate_then_up",
            )
    api.GroupedGemmGluSm100(**samples)
