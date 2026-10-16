# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle of cudnn.torch.install_sdpa_overrides() and its handle.

Kept apart from test_torch_provider.py, whose autouse fixture holds the
overrides installed for every test there.
"""

import pytest
import torch

import cudnn  # noqa: F401

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention import varlen as _varlen  # noqa: F401
except ImportError:
    pytest.skip("torch.nn.attention.varlen required (torch >= 2.13)", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

import cudnn.torch as provider  # noqa: E402
from cudnn.torch import sdpa_provider  # noqa: E402


def _python_fwd_calls_for_one_sdpa():
    q = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16)
    before = provider.calls["fwd"]
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        F.scaled_dot_product_attention(q, q, q, is_causal=True)
    return provider.calls["fwd"] - before


@pytest.mark.L0
def test_remove_restores_builtin_kernels():
    if _python_fwd_calls_for_one_sdpa() != 0:
        pytest.skip("overrides already installed in this process")
    overrides = provider.install_sdpa_overrides()
    try:
        assert _python_fwd_calls_for_one_sdpa() == 1
    finally:
        overrides.remove()
    assert _python_fwd_calls_for_one_sdpa() == 0
    overrides.remove()  # a second remove is a no-op


@pytest.mark.L0
def test_registry_left_alone_when_torch_has_the_switch(monkeypatch):
    attention = pytest.importorskip("torch.nn.attention")
    if not hasattr(attention, "register_flash_attention_impl"):
        pytest.skip("torch.nn.attention flash-impl registry required")
    registered = []
    monkeypatch.setattr(attention, "register_flash_attention_impl", lambda name, **kw: registered.append(name))

    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp_python", lambda enabled: None, raising=False)
    sdpa_provider._register_with_torch()
    assert registered == []

    monkeypatch.delattr(torch.backends.cuda, "enable_cudnn_sdp_python")
    sdpa_provider._register_with_torch()
    assert registered == ["CUDNN"]
