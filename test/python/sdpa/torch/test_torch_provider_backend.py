# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The varlen overrides honor the backend varlen_attn selected.

Later torch versions pass that choice to the ``torch_attn`` varlen ops as a
trailing ``backend`` argument. The provider serves a call only if it selected
cuDNN and the cuDNN python path supports it; everything else must reach torch's
own op body: flash when flash was selected, torch's C++ cuDNN path otherwise.
"""

import logging

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.varlen import varlen_attn, varlen_attn_out
except ImportError:
    pytest.skip("torch.nn.attention.varlen with varlen_attn_out required", allow_module_level=True)

from cudnn.torch import sdpa_provider  # noqa: E402

if not sdpa_provider._has_backend_arg(torch.ops.torch_attn._varlen_attn.default):
    pytest.skip("this torch's varlen ops take no backend argument", allow_module_level=True)

FLASH = [SDPBackend.FLASH_ATTENTION]
CUDNN = [SDPBackend.CUDNN_ATTENTION]


@pytest.fixture
def overrides():
    handle = sdpa_provider.install_sdpa_overrides()
    yield handle
    handle.remove()


def _inputs(lens=(128, 256), H=4, D=64):
    torch.manual_seed(0)
    lens_t = torch.tensor(lens, device="cuda")
    cu = torch.nn.functional.pad(lens_t.cumsum(0), (1, 0)).to(torch.int32)
    T, mx = int(cu[-1]), int(lens_t.max())
    q, k, v = (torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True) for _ in range(3))
    return q, k, v, cu, mx


def _calls_since(before):
    return {key: n - before[key] for key, n in sdpa_provider.calls.items() if n != before[key]}


def _attn(backends, **kwargs):
    q, k, v, cu, mx = _inputs()
    before = dict(sdpa_provider.calls)
    with sdpa_kernel(backends):
        out = varlen_attn(q, k, v, cu, cu, mx, mx, window_size=(-1, 0), **kwargs)
    out.backward(torch.ones_like(out))
    return out, _calls_since(before)


@pytest.mark.L0
def test_selected_cudnn_is_served(overrides):
    _, served = _attn(CUDNN)
    assert served == {"fwd": 1, "bwd": 1}


@pytest.mark.L0
def test_selected_flash_is_passed_to_torch(overrides):
    out, served = _attn(FLASH)
    assert served == {"fwd_torch": 1, "bwd_torch": 1}
    overrides.remove()
    expected, _ = _attn(FLASH)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.parametrize("backends, expected", [(CUDNN, {"fwd": 1}), (FLASH, {"fwd_torch": 1})], ids=["cudnn", "flash"])
def test_out_variant_honors_the_selection(overrides, backends, expected):
    q, k, v, cu, mx = _inputs()
    with torch.no_grad():
        out = torch.empty_like(q)
        before = dict(sdpa_provider.calls)
        with sdpa_kernel(backends):
            varlen_attn_out(out, q, k, v, cu, cu, mx, mx, window_size=(-1, 0))
        assert _calls_since(before) == expected
        with sdpa_kernel(backends):
            ref = varlen_attn(q, k, v, cu, cu, mx, mx, window_size=(-1, 0))
    torch.testing.assert_close(out, ref, rtol=0, atol=2e-2)


@pytest.mark.L0
def test_selected_cudnn_config_the_python_path_declines_runs_torchs_cudnn(overrides, caplog):
    # The cuDNN python path does not serve seqused_k; torch's C++ cuDNN path
    # does. The call must stay on cuDNN rather than drop to flash.
    q, k, v, cu, mx = _inputs()
    seqused_k = (cu[1:] - cu[:-1]).to(torch.int32)
    before = dict(sdpa_provider.calls)
    with torch.no_grad(), sdpa_kernel(CUDNN), caplog.at_level(logging.INFO, logger="torch.nn.attention.varlen"):
        varlen_attn(q, k, v, cu, cu, mx, mx, window_size=(-1, 0), seqused_k=seqused_k)
    assert _calls_since(before) == {"fwd_torch": 1}
    assert "Using cuDNN backend for varlen_attn" in caplog.text
