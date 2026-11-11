# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape-only workspace sizing must cover the strided-affine fallback."""

import pytest
import torch

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")
from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop  # noqa: E402
from cudnn.norm.bprop.api import bprop_workspace_bytes  # noqa: E402
from cudnn.norm.utils import Workspace  # noqa: E402


class _StrictWorkspace(Workspace):
    def take(self, nelem, dtype):
        result = super().take(nelem, dtype)
        assert result is not None, f"reported workspace cannot hold {nelem} elements of {dtype}"
        return result


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["layer", "rms"])
def test_frost_norm_layer_strided_workspace_covers_fallback(kind):
    torch.manual_seed(7108)
    rows, width = 4096, 4096
    x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    gamma = torch.randn(1, device="cuda", dtype=x.dtype).expand(width)
    variant = NormVariant.LAYER_NORM if kind == "layer" else NormVariant.RMS_NORM
    mean = torch.zeros(rows, device="cuda") if kind == "layer" else None
    rstd = torch.ones(rows, device="cuda")
    has_beta = kind == "layer"
    size = bprop_workspace_bytes(variant, x.shape, "bf16", normalized_shape=(width,), has_beta=has_beta)
    ws = _StrictWorkspace(torch.empty(size, device="cuda", dtype=torch.uint8))
    out = (torch.empty_like(x), torch.empty(width, device="cuda"), torch.empty(width, device="cuda") if has_beta else None)
    dx, _, _ = norm_bprop(variant, dy, x, gamma, mean, rstd, normalized_shape=(width,), has_beta=has_beta, out=out, ws=ws)
    xf, d = x[:8].float(), dy[:8].float() * gamma.float()
    ref = d - (d.mean(-1, keepdim=True) if has_beta else 0) - xf * (d * xf).mean(-1, keepdim=True)
    torch.testing.assert_close(dx[:8].float(), ref, rtol=0.02, atol=0.02)
    assert 0 < ws.peak <= size
