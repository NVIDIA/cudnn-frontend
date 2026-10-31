# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness tests for the sm_100 CUTLASS-primitive norm kernels.

Covers all five variants (LayerNorm, RMSNorm, GroupNorm, InstanceNorm,
BatchNorm), fprop + bprop, for fp32 / fp16 / bf16 I/O, validated against PyTorch
autograd. Requires a Blackwell (sm_100) GPU and the internal CUTLASS DSL
(``nvidia-cutlass-dsl-internal``).

Unsupported DSL builds skip before kernel imports.
"""

import os
import sys
import types


import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cudnn.norm", reason="frost norm engines require a built cudnn frontend")

from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_bprop, norm_fprop

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _check(name, got, ref, dtype, kind):
    if got is None and ref is None:
        return True
    amax = (got.float() - ref.float()).abs().max().item()
    try:
        torch.testing.assert_close(got, ref, check_dtype=False)
        ok = True
    except AssertionError as exc:
        print(f"  {name}: {exc}")
        ok = False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<22} {str(dtype):>16} maxabs={amax:.3e}")
    return ok


def _ref_grads(fn, x, g, b):
    xr = x.float().detach().requires_grad_(True)
    gr = g.float().detach().requires_grad_(True) if g is not None else None
    br = b.float().detach().requires_grad_(True) if b is not None else None
    y = fn(xr, gr, br)
    # Follow the norm notebooks and compare both backward paths with the same DY.
    loss = F.mse_loss(y, torch.randn_like(x).float(), reduction="mean")
    dy = torch.autograd.grad(loss, y, retain_graph=True)[0].to(x.dtype)
    ins = [t for t in (xr, gr, br) if t is not None]
    grads = torch.autograd.grad(y, ins, grad_outputs=dy.float())
    it = iter(grads)
    dx = next(it)
    dg = next(it) if gr is not None else None
    db = next(it) if br is not None else None
    return y.detach(), dy, dx, dg, db


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
        y_ref, dy, dx_ref, dg_ref, db_ref = _ref_grads(ref_fn, x, g, b)
        dx, dgamma, dbeta = norm_bprop(variant, dy, x, g, mean, rstd, has_beta=has_beta, **bwd)

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


@pytest.mark.L0
def test_frost_norm_kernels():
    """pytest entry point; this module also runs standalone via ``__main__``."""
    try:
        main()
    except SystemExit as exc:
        assert not exc.code, "checks failed -- see captured stdout"
