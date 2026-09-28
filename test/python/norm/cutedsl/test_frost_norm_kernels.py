# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness tests for the sm_100 CUTLASS-primitive norm kernels.

Covers all five variants (LayerNorm, RMSNorm, GroupNorm, InstanceNorm,
BatchNorm), fprop + bprop, for fp32 / fp16 / bf16 I/O, validated against PyTorch
autograd. Requires a Blackwell (sm_100) GPU and the internal CUTLASS DSL
(``nvidia-cutlass-dsl-internal``).

Run standalone (no built cuDNN extension required):

    pytest test/python/norm/cutedsl/test_frost_norm_kernels.py

A lightweight stub ``cudnn`` package (with a dummy ``pygraph`` so the shared
``cudnn.frost`` lifecycle patch installs) is registered so the pure-Python
``cudnn.norm`` subtree — and ``cudnn.frost.tile_dsl`` — import without the
compiled cuDNN frontend.
"""

import os
import sys
import types


import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")

from cudnn.norm import NormVariant, norm_bprop, norm_fprop

TOL = {
    torch.float32: dict(fwd=3e-4, bwd=2e-4),
    torch.float16: dict(fwd=6e-2, bwd=4e-2),
    torch.bfloat16: dict(fwd=8e-2, bwd=7e-2),
}
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _check(name, got, ref, dtype, kind):
    if got is None and ref is None:
        return True
    got = got.float()
    ref = ref.float()
    amax = (got - ref).abs().max().item()
    scale = max(1.0, ref.abs().max().item())
    ok = amax <= TOL[dtype][kind] * scale
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<22} {str(dtype):>16} maxabs={amax:.3e}")
    return ok


def _ref_grads(fn, x, g, b, dy):
    xr = x.float().detach().requires_grad_(True)
    gr = g.float().detach().requires_grad_(True) if g is not None else None
    br = b.float().detach().requires_grad_(True) if b is not None else None
    y = fn(xr, gr, br)
    ins = [t for t in (xr, gr, br) if t is not None]
    grads = torch.autograd.grad(y, ins, grad_outputs=dy.float())
    it = iter(grads)
    dx = next(it)
    dg = next(it) if gr is not None else None
    db = next(it) if br is not None else None
    return y.detach(), dx, dg, db


def _check_variant(variant, xshape, ref_fn, *, glen, has_beta=True, fwd=None, bwd=None):
    fwd = fwd or {}
    bwd = bwd or {}
    all_ok = True
    print(f"\n=== {variant.value} shape={xshape} glen={glen} has_beta={has_beta} ===")
    for dtype in DTYPES:
        torch.manual_seed(0)
        x = torch.randn(*xshape, device="cuda", dtype=dtype)
        g = torch.randn(glen, device="cuda", dtype=dtype)
        b = torch.randn(glen, device="cuda", dtype=dtype) if has_beta else None

        y, mean, rstd = norm_fprop(variant, x, g, b, **fwd)
        dy = torch.randn_like(y)
        dx, dgamma, dbeta = norm_bprop(variant, dy, x, g, mean, rstd, has_beta=has_beta, **bwd)

        y_ref, dx_ref, dg_ref, db_ref = _ref_grads(ref_fn, x, g, b, dy)
        all_ok &= _check("fwd y", y, y_ref, dtype, "fwd")
        all_ok &= _check("bwd dx", dx, dx_ref, dtype, "bwd")
        all_ok &= _check("bwd dgamma", dgamma, dg_ref, dtype, "bwd")
        if has_beta:
            all_ok &= _check("bwd dbeta", dbeta, db_ref, dtype, "bwd")
    return all_ok


def main():
    assert torch.cuda.is_available(), "CUDA required"
    ok = True
    D = 256

    ok &= _check_variant(
        NormVariant.LAYER_NORM,
        (8, D),
        lambda x, g, b: F.layer_norm(x, (D,), g, b, 1e-5),
        glen=D,
        fwd=dict(normalized_shape=[D], eps=1e-5),
        bwd=dict(normalized_shape=[D]),
    )
    ok &= _check_variant(
        NormVariant.RMS_NORM,
        (8, D),
        lambda x, g, b: F.rms_norm(x, (D,), g, eps=1e-5),
        glen=D,
        has_beta=False,
        fwd=dict(normalized_shape=[D], eps=1e-5),
        bwd=dict(normalized_shape=[D]),
    )
    ok &= _check_variant(
        NormVariant.GROUP_NORM,
        (4, 8, 32),
        lambda x, g, b: F.group_norm(x, 4, g, b, 1e-5),
        glen=8,
        fwd=dict(num_groups=4, eps=1e-5),
        bwd=dict(num_groups=4),
    )
    ok &= _check_variant(
        NormVariant.INSTANCE_NORM,
        (4, 8, 32),
        lambda x, g, b: F.instance_norm(x, weight=g, bias=b, use_input_stats=True, eps=1e-5),
        glen=8,
        fwd=dict(eps=1e-5),
    )
    ok &= _check_variant(
        NormVariant.BATCH_NORM,
        (16, 8, 4),
        lambda x, g, b: F.batch_norm(x, None, None, g, b, True, 0.1, 1e-5),
        glen=8,
        fwd=dict(training=True, eps=1e-5),
    )

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


def test_frost_norm_kernels():
    """pytest entry point; this module also runs standalone via ``__main__``."""
    try:
        main()
    except SystemExit as exc:
        assert not exc.code, "checks failed -- see captured stdout"
