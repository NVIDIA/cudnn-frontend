"""Numerical correctness tests for the FROST norm kernels vs PyTorch.

Covers all five variants (LayerNorm, RMSNorm, GroupNorm, BatchNorm,
InstanceNorm), fprop + bprop, for fp32 / fp16 / bf16 I/O.

Run standalone (no built cuDNN extension required):

    python cudnn/norm/tests/test_norms.py

The test registers a lightweight stub ``cudnn`` package so the pure-Python
``cudnn.norm`` subtree imports without pulling in the compiled frontend.
"""

import os
import sys
import types

# --- make cudnn.norm importable without the compiled cuDNN extension ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_CUDNN_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))  # .../python/cudnn
if "cudnn" not in sys.modules:
    stub = types.ModuleType("cudnn")
    stub.__path__ = [_CUDNN_DIR]
    sys.modules["cudnn"] = stub

import torch
import torch.nn.functional as F

from cudnn.norm.frost import NormVariant, norm_backward, norm_forward

IMPL = os.environ.get("IMPL", "scalar")

TOL = {
    torch.float32: dict(atol=2e-4, rtol=2e-4),
    torch.float16: dict(atol=3e-2, rtol=3e-2),
    torch.bfloat16: dict(atol=6e-2, rtol=6e-2),
}

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _check(name, got, ref, dtype, extra=""):
    got = got.float()
    ref = ref.float()
    tol = TOL[dtype]
    diff = (got - ref).abs()
    denom = ref.abs().clamp_min(1e-3)
    rel = (diff / denom).max().item()
    amax = diff.max().item()
    ok = amax <= tol["atol"] + tol["rtol"] * ref.abs().max().item()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:<10} maxabs={amax:.3e} maxrel={rel:.3e} {extra}")
    return ok


def _run_ref(fn, x, gamma, beta):
    """Autograd reference: returns (y, dx, dgamma, dbeta) for upstream grad=ones."""
    xr = x.float().detach().requires_grad_(True)
    gr = gamma.float().detach().requires_grad_(True) if gamma is not None else None
    br = beta.float().detach().requires_grad_(True) if beta is not None else None
    y = fn(xr, gr, br)
    dy = torch.ones_like(y)
    grads = torch.autograd.grad(y, [t for t in (xr, gr, br) if t is not None], grad_outputs=dy)
    it = iter(grads)
    dx = next(it)
    dg = next(it) if gr is not None else None
    db = next(it) if br is not None else None
    return y.detach(), dx.detach(), (dg.detach() if dg is not None else None), (db.detach() if db is not None else None), dy


def test_variant(variant, x_shape, ref_fn, *, gamma_len, has_beta=True, fwd_kwargs=None, bwd_kwargs=None):
    fwd_kwargs = fwd_kwargs or {}
    bwd_kwargs = bwd_kwargs or {}
    all_ok = True
    print(f"\n=== {variant.value} shape={x_shape} gamma_len={gamma_len} has_beta={has_beta} ===")
    for dtype in DTYPES:
        torch.manual_seed(0)
        x = torch.randn(*x_shape, device="cuda", dtype=dtype)
        gamma = torch.randn(gamma_len, device="cuda", dtype=dtype)
        beta = torch.randn(gamma_len, device="cuda", dtype=dtype) if has_beta else None

        y_ref, dx_ref, dg_ref, db_ref, dy_ref = _run_ref(ref_fn, x, gamma, beta)

        y, mean, rstd = norm_forward(variant, x, gamma, beta, impl=IMPL, **fwd_kwargs)
        ok = _check("fwd y", y, y_ref, dtype)
        all_ok &= ok

        dy = dy_ref.to(dtype)
        dx, dgamma, dbeta = norm_backward(
            variant, dy, x, gamma, mean, rstd, has_beta=has_beta, impl=IMPL, **bwd_kwargs
        )
        all_ok &= _check("bwd dx", dx, dx_ref, dtype)
        all_ok &= _check("bwd dgamma", dgamma, dg_ref, dtype)
        if has_beta:
            all_ok &= _check("bwd dbeta", dbeta, db_ref, dtype)
    return all_ok


def main():
    assert torch.cuda.is_available(), "CUDA required"
    ok = True

    # LayerNorm: normalize over last dim
    D = 256
    ok &= test_variant(
        NormVariant.LAYER_NORM, (8, D), gamma_len=D,
        ref_fn=lambda x, g, b: F.layer_norm(x, (D,), g, b, eps=1e-5),
        fwd_kwargs=dict(normalized_shape=[D], eps=1e-5),
        bwd_kwargs=dict(normalized_shape=[D]),
    )

    # RMSNorm: no bias (matches F.rms_norm)
    ok &= test_variant(
        NormVariant.RMS_NORM, (8, D), gamma_len=D, has_beta=False,
        ref_fn=lambda x, g, b: F.rms_norm(x, (D,), g, eps=1e-5),
        fwd_kwargs=dict(normalized_shape=[D], eps=1e-5),
        bwd_kwargs=dict(normalized_shape=[D]),
    )

    # GroupNorm: N=4, C=8, spatial=32, groups=4
    ok &= test_variant(
        NormVariant.GROUP_NORM, (4, 8, 32), gamma_len=8,
        ref_fn=lambda x, g, b: F.group_norm(x, 4, g, b, eps=1e-5),
        fwd_kwargs=dict(num_groups=4, eps=1e-5),
        bwd_kwargs=dict(num_groups=4),
    )

    # InstanceNorm: N=4, C=8, spatial=32
    ok &= test_variant(
        NormVariant.INSTANCE_NORM, (4, 8, 32), gamma_len=8,
        ref_fn=lambda x, g, b: F.instance_norm(x, weight=g, bias=b, use_input_stats=True, eps=1e-5),
        fwd_kwargs=dict(eps=1e-5),
    )

    # BatchNorm: N=16, C=8, spatial=4, training (no running stats -> pure batch)
    def bn_ref(x, g, b):
        return F.batch_norm(x, None, None, g, b, training=True, momentum=0.1, eps=1e-5)

    ok &= test_variant(
        NormVariant.BATCH_NORM, (16, 8, 4), gamma_len=8,
        ref_fn=bn_ref,
        fwd_kwargs=dict(training=True, eps=1e-5),
    )

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
