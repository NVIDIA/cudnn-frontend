# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 3 / Rule 8 detectors for the BF16 grouped-GEMM API: execute() never blocks the
host, and the opt-in device-value validation refuses to run under stream capture."""

from __future__ import annotations

import pytest
import torch

from test_grouped_gemm_bf16_utils import (
    assert_grouped_gemm_close,
    grouped_gemm_bf16_reference,
    make_grouped_gemm_bf16_problem,
)


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("SM100 is required")


def _outputs(problem):
    m = problem["a"].shape[0]
    n = problem["n"]
    # (M, N, 1) with stride (N, 1, M*N): the kernel's N-major output layout.
    c = torch.empty((1, m, n), dtype=torch.bfloat16, device="cuda").permute(1, 2, 0)
    d = torch.empty((1, m, n), dtype=torch.bfloat16, device="cuda").permute(1, 2, 0)
    return c, d


def _build_api(problem, discrete):
    from cudnn.gemm.cutedsl.grouped.unfused.api import GroupedGemmSm100

    c, d = _outputs(problem)
    kwargs = dict(
        sample_a=problem["a"],
        sample_c=c,
        sample_d=d,
        sample_padded_offsets=problem["offsets"],
        sample_alpha=problem["alpha"],
        sample_prob=problem["prob"],
        generate_c=True,
    )
    if discrete:
        kwargs.update(num_experts=problem["experts"], b_shape=(problem["n"], problem["k"]), b_dtype=torch.bfloat16)
    else:
        kwargs.update(sample_b=problem["b"])
    api = GroupedGemmSm100(**kwargs)
    assert api.check_support() is True
    api.compile()
    return api


def _run(api, problem, c, d, offsets, *, discrete, b_ptrs=None):
    kwargs = dict(a_tensor=problem["a"], c_tensor=c, d_tensor=d, padded_offsets=offsets, alpha_tensor=problem["alpha"], prob_tensor=problem["prob"])
    if discrete:
        kwargs["b_ptrs"] = problem["b_ptrs"] if b_ptrs is None else b_ptrs
    else:
        kwargs["b_tensor"] = problem["b"]
    api.execute(**kwargs)


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_execute_never_synchronizes(discrete):
    problem = make_grouped_gemm_bf16_problem(discrete=discrete)
    api = _build_api(problem, discrete)
    c, d = _outputs(problem)
    _run(api, problem, c, d, problem["offsets"], discrete=discrete)
    torch.cuda.synchronize()
    expected_c, expected_d = grouped_gemm_bf16_reference(problem)
    assert_grouped_gemm_close(c, expected_c)
    assert_grouped_gemm_close(d, expected_d)

    # Fresh offsets / pointer tables per execute: a tensor the plan has never seen is
    # exactly what an identity-keyed validation memo misses on, which is how the old
    # per-tensor D2H read hid from every warm test.
    runs = []
    for _ in range(3):
        out_c, out_d = _outputs(problem)
        offsets = problem["offsets"].clone()
        b_ptrs = problem["b_ptrs"].clone() if discrete else None
        runs.append((out_c, out_d, offsets, b_ptrs))
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        for out_c, out_d, offsets, b_ptrs in runs:
            _run(api, problem, out_c, out_d, offsets, discrete=discrete, b_ptrs=b_ptrs)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    for out_c, out_d, _, _ in runs:
        torch.testing.assert_close(out_c, c, rtol=0, atol=0)
        torch.testing.assert_close(out_d, d, rtol=0, atol=0)


@pytest.mark.L0
def test_debug_validation_env_var(monkeypatch):
    from cudnn.gemm.cutedsl.grouped import backend_utils

    problem = make_grouped_gemm_bf16_problem(discrete=True)
    api = _build_api(problem, discrete=True)
    c, d = _outputs(problem)
    m = problem["a"].shape[0]
    good = problem["offsets"]

    monkeypatch.setattr(backend_utils, "DEBUG_VALIDATE_DEVICE_VALUES", True)
    _run(api, problem, c, d, good, discrete=True)  # well-formed values pass the blocking checks

    decreasing = good.flip(0)
    with pytest.raises(ValueError, match="non-decreasing"):
        _run(api, problem, c, d, decreasing, discrete=True)
    unaligned = good.clone()
    unaligned[0] += 8
    with pytest.raises(ValueError, match="256-aligned"):
        _run(api, problem, c, d, unaligned, discrete=True)
    too_long = good.clone()
    too_long[-1] = m + 256
    with pytest.raises(ValueError, match="last value"):
        _run(api, problem, c, d, too_long, discrete=True)
    null_entry = problem["b_ptrs"].clone()
    null_entry[0] = 0
    with pytest.raises(ValueError, match="non-null and 16-byte aligned"):
        _run(api, problem, c, d, good, discrete=True, b_ptrs=null_entry)

    # R6: with the flag on, a captured stream is refused BEFORE any D2H read -- our
    # RuntimeError naming the env var, not torch's "operation not permitted when
    # stream is capturing".
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match=backend_utils.DEBUG_VALIDATE_DEVICE_VALUES_ENV):
        with torch.cuda.graph(graph):
            _run(api, problem, c, d, good, discrete=True)
