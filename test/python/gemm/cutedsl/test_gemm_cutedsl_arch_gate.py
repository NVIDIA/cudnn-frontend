# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CuTe DSL GEMM APIs run SM100-family tcgen05 kernels, so their architecture gate declines cc 12.x as well as pre-SM100 parts.

cc 12.x reports a major above 10, so a plain ``compute_capability < 100`` gate admitted it: ``check_support()`` returned True and the
kernel then failed to build (ptxas rejects ``tcgen05.mma`` / ``tcgen05.cp`` / ``tcgen05.commit`` for ``sm_120a``)."""

import re
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.L0

_CUTEDSL_GEMM = Path(__file__).resolve().parents[4] / "python/cudnn/gemm/cutedsl"


@pytest.mark.parametrize("major,expected", [(8, False), (9, False), (10, True), (11, True), (12, False)])
def test_has_tcgen05(major, expected):
    from cudnn.tensor_adapter import has_tcgen05

    assert has_tcgen05(major) is expected


def test_no_cutedsl_gemm_gate_admits_cc12_by_threshold():
    """Every gate goes through has_tcgen05; a bare threshold on the capability would admit cc 12.x again."""
    bare = re.compile(r"(compute_capability|capability) *< *100|\bmajor *< *10\b")
    offenders = [
        f"{p.relative_to(_CUTEDSL_GEMM)}:{n}"
        for p in sorted(_CUTEDSL_GEMM.rglob("*.py"))
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if bare.search(line)
    ]
    assert not offenders, offenders
    gated = [p for p in _CUTEDSL_GEMM.rglob("*.py") if "has_tcgen05(major)" in p.read_text()]
    assert len(gated) >= 24, sorted(str(p.relative_to(_CUTEDSL_GEMM)) for p in gated)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gemm_amax_check_support_declines_cc12(monkeypatch):
    """A cc 12.0 device is declined by check_support() with the gate's RuntimeError, before any kernel is built."""
    try:
        import cudnn
        from gemm.cutedsl.test_gemm_amax_utils import allocate_input_tensors, allocate_output_tensors
    except ImportError as error:
        pytest.skip(f"cudnn optional dependencies not installed: {error}")
    api = sys.modules[cudnn.GemmAmaxSm100.__module__]
    monkeypatch.setattr(api, "get_compute_capability", lambda *args, **kwargs: (12, 0))
    a, _, b, _, sfa, _, sfb, _ = allocate_input_tensors(512, 256, 256, 1, torch.float8_e5m2, torch.float8_e8m0fnu, 32, "k", "k")
    c, amax = allocate_output_tensors(512, 256, 1, torch.float32, "n")
    gemm = cudnn.GemmAmaxSm100(
        sample_a=a,
        sample_b=b,
        sample_sfa=sfa,
        sample_sfb=sfb,
        sample_c=c,
        sample_amax=amax,
        acc_dtype=torch.float32,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sf_vec_size=32,
    )
    with pytest.raises(RuntimeError, match=r"not cc 12\.x"):
        gemm.check_support()
