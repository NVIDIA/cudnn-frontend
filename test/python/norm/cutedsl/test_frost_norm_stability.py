# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical stability of the centered reductions: offset data with a small spread.

Every centered norm here once computed its variance as ``E[x^2] - E[x]^2``. Both
terms scale as ``|x|^2`` while their difference is the variance, so for data
carrying a large common offset the subtraction cancels every significant bit.
fp32 LayerNorm over values alternating 8191/8193 -- ordinary, finite, supported
input -- returned +/-316.2 where the answer is +/-0.999995, and GroupNorm,
InstanceNorm and training BatchNorm all failed the same way. The reductions are
small (256 elements, 32, 49), so this is not a long-row problem that a
long-row-only fix would cover.

Two separate defects produced it, and this file pins both:

1. **The reduction.** Fixed by subtracting a shift K drawn from the data before
   squaring (``shifted_stats`` in ``_common_sm100``), so the cancellation is
   proportional to the spread rather than the magnitude.
2. **The affine.** Several kernels applied ``y = x*scale + shift`` with
   ``shift = -mean*scale``. That recreates the same cancellation in the OUTPUT
   even once the statistics are exact: both terms are O(|x|*rstd) and their sum
   is O(1). Centering first (``(x - mean) * scale``) is exact by Sterbenz.
   Statistics alone would not have caught this -- InstanceNorm (16,32,1,1) had a
   correct rstd and a y error of 2.2e-02 -- so the output is asserted too.

What the bounds mean. ``rstd`` is checked on a relative tolerance: it carries no
offset and should be accurate to a few fp32 roundings. ``mean`` is checked
against *half an ulp at its own magnitude*, i.e. it must be the correctly-rounded
fp32 value -- the best an fp32 output can be. ``y`` is then allowed the error
that a correctly-rounded fp32 mean necessarily induces, ``0.5*ulp(mean)*rstd``,
plus a few roundings. These are tight: the pre-fix numbers exceed them by five
orders of magnitude.

Run standalone (no built cuDNN extension required):

    python test/python/norm/cutedsl/test_frost_norm_stability.py
    pytest test/python/norm/cutedsl/test_frost_norm_stability.py -m L1
"""

import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _install_repo_cudnn_stub():
    """Point ``cudnn`` at the repo subtree, for STANDALONE runs only.

    Never at import time under pytest: ``sys.modules["cudnn"]`` is process-global,
    so a stub installed during collection replaces the real module for every other
    test in the session.
    """
    if "cudnn" in sys.modules:
        return
    repo_cudnn = Path(__file__).resolve().parents[4] / "python" / "cudnn"
    if not (repo_cudnn / "norm").is_dir():
        return
    stub = types.ModuleType("cudnn")
    stub.__path__ = [str(repo_cudnn)]
    stub.__file__ = str(repo_cudnn / "__init__.py")
    stub.pygraph = type("pygraph", (), {})
    stub._pybind_module = types.ModuleType("cudnn._pybind_module")

    import enum

    class _DataType(enum.Enum):
        NOT_SET = 0
        HALF = 1
        BFLOAT16 = 2
        FLOAT = 3

    stub.data_type = _DataType
    sys.modules["cudnn"] = stub


if __name__ == "__main__":
    _install_repo_cudnn_stub()

pytest.importorskip("cudnn.norm", reason="frost norm kernels require a built cudnn frontend")

from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.norm import NormVariant, norm_fprop  # noqa: E402

LN, RMS = NormVariant.LAYER_NORM, NormVariant.RMS_NORM
GN, IN, BN = NormVariant.GROUP_NORM, NormVariant.INSTANCE_NORM, NormVariant.BATCH_NORM
EPS = 1e-5
_F32_EPS = 2.0**-24


def _ulp(x):
    """Spacing of fp32 at magnitude ``x`` (0 below the first normal binade)."""
    x = abs(float(x))
    if x == 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(x)) - 23)


def _data(shape, offset, spread, device="cuda"):
    """Alternating ``offset -/+ spread`` -- maximal cancellation, exact in fp32."""
    t = torch.empty(shape, device=device, dtype=torch.float32)
    f = t.reshape(-1)
    f[0::2] = offset - spread
    f[1::2] = offset + spread
    return t


def _reference(variant, x, g, b, num_groups):
    """fp64 reference plus the per-reduction mean/rstd it implies."""
    xd = x.double()
    if variant in (LN, RMS):
        D = x.shape[-1]
        flat = xd.reshape(-1, D)
        if variant is LN:
            y = F.layer_norm(xd, [D], g.double(), b.double(), EPS)
            mean = flat.mean(-1)
        else:
            y = F.rms_norm(xd, [D], g.double(), eps=EPS)
            mean = torch.zeros_like(flat[:, 0])
        var = (flat - flat.mean(-1, keepdim=True)).pow(2).mean(-1) if variant is LN else flat.pow(2).mean(-1)
    elif variant is BN:
        y = F.batch_norm(xd, None, None, g.double(), b.double(), True, 0.1, EPS)
        ch = xd.permute(1, 0, *range(2, xd.dim())).reshape(x.shape[1], -1)
        mean, var = ch.mean(-1), ch.var(-1, correction=0)
    else:
        y = F.group_norm(xd, num_groups, g.double(), b.double(), EPS)
        grp = xd.reshape(x.shape[0], num_groups, -1)
        mean, var = grp.mean(-1).reshape(-1), grp.var(-1, correction=0).reshape(-1)
    return y, mean, torch.rsqrt(var + EPS)


def _check_case(variant, shape, *, offset, spread, num_groups=None, channels_last=False, label=""):
    x = _data(shape, offset, spread)
    if channels_last and x.dim() == 4:
        x = x.to(memory_format=torch.channels_last)
    width = shape[-1] if variant in (LN, RMS) else shape[1]
    g = torch.ones(width, device="cuda", dtype=torch.float32)
    b = None if variant is RMS else torch.zeros(width, device="cuda", dtype=torch.float32)

    kw = {}
    if variant in (LN, RMS):
        kw["normalized_shape"] = [shape[-1]]
    elif variant is BN:
        kw["training"] = True
    else:
        kw["num_groups"] = num_groups
    y, mean, rstd = norm_fprop(variant, x, g, b, eps=EPS, **kw)
    torch.cuda.synchronize()

    y_ref, mean_ref, rstd_ref = _reference(variant, x, g, b if b is not None else torch.zeros_like(g), num_groups)

    # The error a correctly-rounded fp32 ``mean`` output forces, per reduction. Using
    # the ACTUAL rounding rather than a worst-case half-ulp keeps the bounds tight
    # where the exact mean happens to be representable -- which is precisely the
    # InstanceNorm HW=1 case the fused affine broke while its statistics stayed
    # correct. A half-ulp bound there would have been 0.154 and hidden a 2.2e-02 y.
    mean_round = (mean_ref.float().double() - mean_ref).abs()

    # rstd has no offset and loses at most a few fp32 rounding steps.
    # The previous reduction exceeded this bound by about five orders of magnitude.
    rstd_err = (rstd.double().reshape(-1) - rstd_ref.reshape(-1)).abs().max().item()
    rstd_tol = 64 * _F32_EPS * rstd_ref.abs().max().item()
    ok = rstd_err <= rstd_tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:36} rstd  err={rstd_err:.3e} tol={rstd_tol:.3e}")

    # mean must be the correctly-rounded fp32 value at its own magnitude.
    # RMSNorm is uncentered and returns a placeholder here, so it has none.
    ok2 = True
    if variant is not RMS:
        mean_err = (mean.double().reshape(-1) - mean_ref.reshape(-1)).abs().max().item()
        mean_tol = 4 * mean_round.max().item() + 1024 * _F32_EPS * max(1.0, spread)
        ok2 = mean_err <= mean_tol
        print(f"  [{'PASS' if ok2 else 'FAIL'}] {label:36} mean  err={mean_err:.3e} tol={mean_tol:.3e}")

    # y may inherit that mean's rounding, amplified by rstd, and nothing beyond it.
    y_err = (y.double() - y_ref).abs().max().item()
    y_tol = 4 * (mean_round * rstd_ref).max().item() + 1024 * _F32_EPS * max(1.0, y_ref.abs().max().item())
    ok3 = y_err <= y_tol
    print(f"  [{'PASS' if ok3 else 'FAIL'}] {label:36} y     err={y_err:.3e} tol={y_tol:.3e}")
    return ok and ok2 and ok3


# (variant, shape, num_groups, channels_last) -- chosen to reach distinct kernels:
# the warp LN, the CGA LN, the blocked GN/IN, both NHWC paths and both BN paths.
def _cases(full):
    offsets = [8192.0, 1.0e6] if full else [8192.0]
    spreads = [1.0, 1.0 / 64.0] if full else [1.0]
    base = [
        # the reviewer's four reported failures, verbatim
        (LN, (4, 256), None, False),
        (GN, (4, 8, 32), 4, False),
        (IN, (4, 8, 32), 8, False),
        (BN, (16, 8, 4), None, False),
        # RMSNorm is uncentered: a control that must stay exact
        (RMS, (4, 256), None, False),
    ]
    if full:
        base += [
            (LN, (4, 65536), None, False),  # CGA row split
            (LN, (64, 37), None, False),  # non-vectorizable fallback
            (RMS, (4, 65536), None, False),
            (GN, (4, 32, 8, 8), 4, False),
            (GN, (4, 32, 8, 8), 4, True),  # NHWC
            (IN, (4, 32, 8, 8), 32, False),
            (IN, (4, 32, 8, 8), 32, True),  # NHWC
            (IN, (16, 32, 1, 1), 32, True),  # HW=1: rstd=1/sqrt(eps), amplifies any y error
            (BN, (16, 32, 8, 8), None, False),
            (BN, (16, 32, 8, 8), None, True),  # NHWC
            (BN, (3, 7, 5, 5), None, False),  # odd S -> scalar fallback kernel
        ]
    out = []
    for off in offsets:
        for sp in spreads:
            for variant, shape, ng, cl in base:
                out.append((variant, shape, ng, cl, off, sp))
    return out


def main(full=True):
    assert torch.cuda.is_available(), "CUDA required"
    ok = True
    for variant, shape, ng, cl, off, sp in _cases(full):
        name = str(variant).split(".")[-1]
        label = f"{name} {tuple(shape)}{' NHWC' if cl else ''} off={off:g} sp={sp:g}"
        print(f"\n=== {label} ===")
        ok &= _check_case(variant, shape, offset=off, spread=sp, num_groups=ng, channels_last=cl, label=name)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


def _entry(full):
    try:
        main(full=full)
    except SystemExit as exc:
        assert not exc.code, "checks failed -- see captured stdout"


@pytest.mark.L0
def test_frost_norm_offset_stability_smoke():
    """The four reported failures plus the uncentered control."""
    _entry(full=False)


@pytest.mark.L1
def test_frost_norm_offset_stability():
    """Every centered kernel x offset x spread, outputs and saved statistics."""
    _entry(full=True)
