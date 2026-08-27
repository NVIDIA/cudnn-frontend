# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The wrapper memo must never answer for operands it did not see.

The memo skips the wrapper's derivation on a repeat call. These tests pin the
property that makes that safe: the key is the operands' metadata, so anything that
changes what the derivation would produce takes a different key. In particular a
freshly allocated tensor is not trusted on account of its address -- CPython recycles
those immediately, so an identity-keyed memo would serve a stale entry to a tensor of
a different shape.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("SM100 is required")


N_OUT, K, EXPERTS = 2048, 2048, 8


def _weights():
    b = torch.randn(EXPERTS, N_OUT, K, dtype=torch.bfloat16, device="cuda")
    b_ptrs = torch.tensor([b[i].data_ptr() for i in range(EXPERTS)], dtype=torch.int64, device="cuda")
    return b, b_ptrs


def _operands(m):
    return dict(
        a_tensor=torch.randn(m, K, 1, dtype=torch.bfloat16, device="cuda"),
        padded_offsets=torch.arange(m // EXPERTS, m + 1, m // EXPERTS, dtype=torch.int32, device="cuda"),
        alpha_tensor=torch.randn(EXPERTS, dtype=torch.float32, device="cuda"),
        prob_tensor=torch.linspace(0.25, 0.875, m, dtype=torch.float32, device="cuda").reshape(m, 1, 1),
    )


def _call(m, b_ptrs, a=None, operands=None):
    from cudnn import grouped_gemm_wrapper_sm100

    operands = dict(operands if operands is not None else _operands(m))
    if a is not None:
        operands["a_tensor"] = a
    return grouped_gemm_wrapper_sm100(
        **operands,
        b_ptrs=b_ptrs,
        n=N_OUT,
        b_dtype=torch.bfloat16,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
    )


@pytest.mark.L0
def test_memo_hit_matches_cold_path():
    """A memo hit produces the same bytes as the same call with the memo cleared."""
    from cudnn.gemm.cutedsl.grouped.unfused.api import _wrapper_memo

    b, b_ptrs = _weights()
    operands = _operands(2048)

    _wrapper_memo.clear()
    cold = _call(2048, b_ptrs, operands=operands)["d_tensor"].clone()
    assert _wrapper_memo, "the call should have populated the memo"
    warm = _call(2048, b_ptrs, operands=operands)["d_tensor"]

    torch.testing.assert_close(warm, cold, rtol=0, atol=0)


@pytest.mark.L0
def test_memo_misses_when_m_changes():
    """Alternating token counts with freshly built operands each step.

    Every operand is a new object per call, so their addresses are recycled; only the
    metadata in the key distinguishes the two shapes.
    """
    b, b_ptrs = _weights()
    for m in (2048, 2048, 4096, 2048, 4096, 4096, 2048, 4096):
        assert tuple(_call(m, b_ptrs)["d_tensor"].shape) == (m, N_OUT, 1)


@pytest.mark.L0
def test_memo_does_not_swallow_a_dtype_mismatch():
    """A fresh, wrongly-typed a_tensor is still rejected after the memo is warm."""
    b, b_ptrs = _weights()
    _call(2048, b_ptrs)

    bad = torch.empty(2048, K, 1, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError):
        _call(2048, b_ptrs, a=bad)


@pytest.mark.L0
def test_memo_misses_on_a_transposed_operand():
    """Same shape and dtype, different strides -- a different key, and still rejected."""
    b, b_ptrs = _weights()
    _call(2048, b_ptrs)

    transposed = torch.randn(K, 2048, 1, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    assert tuple(transposed.shape) == (2048, K, 1)
    with pytest.raises(ValueError):
        _call(2048, b_ptrs, a=transposed)


@pytest.mark.L0
def test_glu_memo_hit_matches_cold_path():
    from cudnn import grouped_gemm_glu_wrapper_sm100
    from cudnn.gemm.cutedsl.grouped.glu.api import _glu_wrapper_memo

    m = 2048
    b, b_ptrs = _weights()
    kwargs = dict(
        a_tensor=torch.randn(m, K, 1, dtype=torch.bfloat16, device="cuda"),
        sfa_tensor=None,
        padded_offsets=torch.arange(m // EXPERTS, m + 1, m // EXPERTS, dtype=torch.int32, device="cuda"),
        alpha_tensor=torch.randn(EXPERTS, dtype=torch.float32, device="cuda"),
        prob_tensor=torch.linspace(0.25, 0.875, m, dtype=torch.float32, device="cuda").reshape(m, 1, 1),
        b_ptrs=b_ptrs,
        n=N_OUT,
        b_dtype=torch.bfloat16,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
    )

    _glu_wrapper_memo.clear()
    cold = grouped_gemm_glu_wrapper_sm100(**kwargs)["d_tensor"].clone()
    assert _glu_wrapper_memo, "the call should have populated the memo"
    warm = grouped_gemm_glu_wrapper_sm100(**kwargs)["d_tensor"]

    torch.testing.assert_close(warm, cold, rtol=0, atol=0)


@pytest.mark.L0
def test_memo_key_covers_every_wrapper_parameter():
    """Every parameter that can change the result must appear in that wrapper's memo key.

    A parameter added to a wrapper but not to its key makes two different calls collide,
    and the failure is silent: the second gets the first's compiled op and output spec.
    This caught `sf_fp8_dtype_override` going missing from the GLU key during a rebase.

    Auto-discovers every CuTeDSL wrapper that has a memo, so an op gains coverage the
    moment one is added to it and no edit here is needed.
    """
    import importlib
    import inspect
    import pkgutil
    import re

    import cudnn.gemm.cutedsl as cutedsl

    # linear_offset is not part of any op cache key either; the wrappers resolve and
    # forward the caller's value on every call, hit or miss. current_stream is per-call
    # state, not a cache dimension.
    NOT_CACHE_DIMENSIONS = {"current_stream", "linear_offset"}

    checked = []
    for module_info in pkgutil.walk_packages(cutedsl.__path__, cutedsl.__name__ + "."):
        if not module_info.name.endswith(".api"):
            continue
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        for name, func in vars(module).items():
            if not (name.endswith("_wrapper_sm100") and inspect.isfunction(func)):
                continue
            try:
                source = inspect.getsource(func)
            except OSError:
                continue
            if "memo_key = (" not in source:
                continue  # no memo on this op yet
            start = source.index("memo_key = (") + len("memo_key = (")
            key_source = source[start : source.index("\n    )", start)]
            params = set(inspect.signature(func).parameters) - NOT_CACHE_DIMENSIONS
            missing = sorted(p for p in params if not re.search(rf"\b{p}\b", key_source))
            assert not missing, f"{name} parameters missing from its memo key: {missing}"
            checked.append(name)

    assert checked, "no memoized wrappers discovered -- the discovery logic is broken"
