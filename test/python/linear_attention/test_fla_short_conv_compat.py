# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported-GPU route and parity gates for the FLA short-convolution shim."""

from __future__ import annotations

from importlib import metadata

import pytest
import torch
from cudnn._causal_conv1d_arch import (
    is_supported_causal_conv1d_update_compute_capability,
)

fla_ops = pytest.importorskip("fla.modules.conv.triton.ops")
fla_short_conv = pytest.importorskip("fla.modules.conv.short_conv")

try:
    _FLA_VERSION = metadata.version("flash-linear-attention")
except metadata.PackageNotFoundError:
    _FLA_VERSION = None

from cudnn.fla import (
    accelerate_fla,
    is_accelerated,
    restore_fla,
    short_conv_last_path,
)

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and is_supported_causal_conv1d_update_compute_capability(torch.cuda.get_device_capability())),
        reason="the native decode short-convolution update requires a functionally supported GPU architecture",
    ),
    pytest.mark.skipif(
        _FLA_VERSION != "0.5.2",
        reason="the production short-convolution shim intentionally supports FLA 0.5.2 exactly",
    ),
]

_REAL_UPDATE = fla_ops.causal_conv1d_update
_ATOL = 3e-2
_RTOL = 3e-2


@pytest.fixture(autouse=True)
def _restore_short_conv_patch():
    restore_fla(targets="short_conv")
    yield
    restore_fla(targets="short_conv")
    assert fla_ops.causal_conv1d_update is _REAL_UPDATE


def _layout(base: torch.Tensor, name: str) -> torch.Tensor:
    if name == "ND":
        return base
    if name == "N1D":
        return base.unsqueeze(1)
    if name == "1ND":
        return base.unsqueeze(0)
    raise AssertionError(name)


@pytest.mark.parametrize("layout", ["ND", "N1D", "1ND"])
@torch.no_grad()
def test_real_fla_entry_point_routes_native_with_parity(layout):
    torch.manual_seed(20260828)
    n_rows, n_channels = 3, 257
    x = _layout(
        torch.randn(n_rows, n_channels, device="cuda", dtype=torch.bfloat16) * 0.25,
        layout,
    )
    weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16) * 0.25
    initial_state = torch.randn(n_rows, n_channels, 4, device="cuda", dtype=torch.bfloat16) * 0.25
    fla_state = initial_state.clone()
    native_state = initial_state.clone()

    expected, expected_cache = _REAL_UPDATE(
        x,
        fla_state,
        residual=None,
        weight=weight,
        bias=None,
        activation="silu",
    )
    assert expected_cache is fla_state

    accelerate_fla(verbose=False, targets="shortconv")
    actual, returned_cache = fla_ops.causal_conv1d_update(
        x,
        native_state,
        residual=None,
        weight=weight,
        bias=None,
        activation="silu",
    )

    assert is_accelerated("short_conv")
    assert short_conv_last_path() == "native"
    assert actual.shape == x.shape
    assert returned_cache is native_state
    torch.testing.assert_close(actual, expected, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(
        native_state.view(torch.int16),
        fla_state.view(torch.int16),
        rtol=0,
        atol=0,
    )


@torch.no_grad()
def test_real_short_convolution_module_step_consumes_patched_entry_point():
    torch.manual_seed(20260829)
    n_rows, n_channels = 3, 257
    module = fla_short_conv.ShortConvolution(
        hidden_size=n_channels,
        kernel_size=4,
        bias=False,
        activation="silu",
        backend="triton",
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = torch.randn(n_rows, 1, n_channels, device="cuda", dtype=torch.bfloat16) * 0.25
    initial_state = torch.randn(n_rows, n_channels, 4, device="cuda", dtype=torch.bfloat16) * 0.25
    fla_state = initial_state.clone()
    native_state = initial_state.clone()

    expected, expected_cache = module(
        x,
        cache=fla_state,
        output_final_state=True,
    )
    assert expected_cache is fla_state

    accelerate_fla(verbose=False, targets="short_conv")
    actual, returned_cache = module(
        x,
        cache=native_state,
        output_final_state=True,
    )

    assert short_conv_last_path() == "native"
    assert returned_cache is native_state
    torch.testing.assert_close(actual, expected, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(
        native_state.view(torch.int16),
        fla_state.view(torch.int16),
        rtol=0,
        atol=0,
    )
