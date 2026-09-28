# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for every staging/vectorization path of the row-wise kernels.

``norm_fprop``/``norm_bprop`` auto-pick TMA bulk staging + scalar stores, so this
test forces each ``Cfg`` (STAGE_NONE / STAGE_CPASYNC / STAGE_BULK, vec on/off) on
the LayerNorm and GroupNorm kernels directly and checks fprop + bprop vs PyTorch
autograd. cp.async requires ``M % (bt*V) == 0``; the harness picks a valid ``bt``.

    pytest test/python/norm/cutedsl/test_frost_norm_staging.py
"""

import os
import sys
import types


import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")

from cudnn.norm.config_sm100 import (
    STAGE_BULK,
    STAGE_CPASYNC,
    STAGE_NONE,
    Cfg,
    NormVariant as NV,
    TemplateParams,
    rowwise_spec,
    vector_width,
)
from cudnn.norm.dtypes import DTYPE_BYTES, torch_dtype_to_str
from cudnn.norm.fprop.kernels import groupnorm_sm100 as gf, layernorm_sm100 as lf
from cudnn.norm.bprop.kernels import groupnorm_sm100 as gb, layernorm_sm100 as lb

_OK = True


def _chk(name, got, ref, tol):
    global _OK
    amax = (got.float() - ref.float()).abs().max().item()
    scale = max(1.0, ref.float().abs().max().item())
    ok = amax <= tol * scale
    _OK &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<30} maxabs={amax:.3e}")


def _cpasync_bt(M, V):
    """Largest multiple-of-32 bt (<=256) with bt*V | M, or None."""
    for bt in (256, 128, 64, 32):
        if M % (bt * V) == 0:
            return bt
    return None


def _modes(M, V, eb):
    out = [
        ("bulk+scal", Cfg(256, V, STAGE_BULK, False, eb)),
        ("bulk+vec", Cfg(256, V, STAGE_BULK, True, eb)),
        ("none+scal", Cfg(256, V, STAGE_NONE, False, eb)),
        ("none+vec", Cfg(256, V, STAGE_NONE, True, eb)),
    ]
    cbt = _cpasync_bt(M, V)
    if cbt is not None:
        out.append(("cpasync+vec", Cfg(cbt, V, STAGE_CPASYNC, True, eb)))
        out.append(("cpasync+scal", Cfg(cbt, V, STAGE_CPASYNC, False, eb)))
    return out


def _ln(dt):
    D = 512
    eb = DTYPE_BYTES[torch_dtype_to_str(dt)]
    V = vector_width(eb)
    tol = 6e-2 if dt != torch.float32 else 3e-4
    torch.manual_seed(0)
    x = torch.randn(6, D, device="cuda", dtype=dt)
    g = torch.randn(D, device="cuda", dtype=dt)
    b = torch.randn(D, device="cuda", dtype=dt)
    spec = rowwise_spec(NV.LAYER_NORM, x.shape, normalized_shape=[D])
    p = TemplateParams(variant=NV.LAYER_NORM, io_dtype=torch_dtype_to_str(dt))
    xr = x.float().detach().requires_grad_(True)
    gr = g.float().detach().requires_grad_(True)
    br = b.float().detach().requires_grad_(True)
    yref = F.layer_norm(xr, (D,), gr, br, 1e-5)
    dy = torch.randn_like(yref)
    dxr, dgr, dbr = torch.autograd.grad(yref, [xr, gr, br], grad_outputs=dy)
    for name, cfg in _modes(spec.M, V, eb):
        y, mean, rstd = lf.forward(spec, x.reshape(spec.R, spec.M), g, b, eps=1e-5, cfg=cfg, params=p)
        dx, dgamma, dbeta = lb.backward(spec, dy.to(dt).reshape(spec.R, spec.M), x.reshape(spec.R, spec.M), g, mean, rstd, has_beta=True, cfg=cfg, params=p)
        _chk(f"LN {name} y", y.reshape(x.shape), yref, tol)
        _chk(f"LN {name} dx", dx.reshape(x.shape), dxr, tol)
        _chk(f"LN {name} dgamma", dgamma, dgr, tol)


def _gn(dt):
    N, C, G, HW = 4, 8, 4, 64  # M = (C/G)*HW = 128
    eb = DTYPE_BYTES[torch_dtype_to_str(dt)]
    V = vector_width(eb)
    tol = 8e-2 if dt != torch.float32 else 3e-4
    torch.manual_seed(0)
    x = torch.randn(N, C, HW, device="cuda", dtype=dt)
    g = torch.randn(C, device="cuda", dtype=dt)
    b = torch.randn(C, device="cuda", dtype=dt)
    spec = rowwise_spec(NV.GROUP_NORM, x.shape, num_groups=G)
    p = TemplateParams(variant=NV.GROUP_NORM, io_dtype=torch_dtype_to_str(dt))
    xr = x.float().detach().requires_grad_(True)
    gr = g.float().detach().requires_grad_(True)
    br = b.float().detach().requires_grad_(True)
    yref = F.group_norm(xr, G, gr, br, 1e-5)
    dy = torch.randn_like(yref)
    dxr, dgr, dbr = torch.autograd.grad(yref, [xr, gr, br], grad_outputs=dy)
    for name, cfg in _modes(spec.M, V, eb):
        y, mean, rstd = gf.forward(spec, x.reshape(spec.R, spec.M), g, b, eps=1e-5, cfg=cfg, params=p)
        dx, dgamma, dbeta = gb.backward(spec, dy.to(dt).reshape(spec.R, spec.M), x.reshape(spec.R, spec.M), g, mean, rstd, has_beta=True, cfg=cfg, params=p)
        _chk(f"GN {name} y", y.reshape(x.shape), yref, tol)
        _chk(f"GN {name} dx", dx.reshape(x.shape), dxr, tol)
        _chk(f"GN {name} dgamma", dgamma, dgr, tol)


def main():
    assert torch.cuda.is_available(), "CUDA required"
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        print(f"\n=== dtype={dt} ===")
        _ln(dt)
        _gn(dt)
    print("\n" + ("ALL PASS" if _OK else "SOME FAILED"))
    sys.exit(0 if _OK else 1)


if __name__ == "__main__":
    main()


def test_frost_norm_staging():
    """pytest entry point; this module also runs standalone via ``__main__``."""
    try:
        main()
    except SystemExit as exc:
        assert not exc.code, "checks failed -- see captured stdout"
