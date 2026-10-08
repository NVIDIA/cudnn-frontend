# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channels-last (NHWC) correctness tests for the sm_100 norm kernels.

Covers the three layout-sensitive variants -- BatchNorm, InstanceNorm, GroupNorm --
fprop + bprop, for fp32 / fp16 / bf16, on channels-last input. LayerNorm and RMSNorm
are deliberately absent: they reduce the last LOGICAL dim, which is strided under
channels-last, so there is no native NHWC map for them and no meaningful layout axis
-- they are covered by ``test_frost_norm_kernels``.

Three things are asserted per case, and the last two are the point of this file:

1. **Numerics** against PyTorch autograd.
2. **The layout survives.** ``y`` and ``dx`` must come back channels-last. A norm that
   transposes on the way in and out is still numerically correct, so a numerics-only
   test passes while every downstream op silently pays a transpose -- which is exactly
   how InstanceNorm and GroupNorm shipped a channels-last forward that did not exist.
3. **The native kernel actually ran.** Even a layout-preserving wrapper around a
   transposing kernel would pass (1) and (2). Dispatch is instrumented so the test
   fails if a case silently falls back off its NHWC kernel -- the failure mode when a
   geometry helper rejects a shape, which is how ``cpg > 128`` GroupNorm sat on the
   transposing path.

The chain is the realistic one: ``dy = torch.randn_like(y)`` inherits ``y``'s layout
rather than being forced channels-last by hand. Forcing it hides the case where fprop
loses the layout, because the backward then gets a channels-last ``dy`` it would never
see in a real model.

Split by cost, because the repo gates on ``-m L0`` and a comparable L0 norm test
runs in 3-5s: the L0 entry is a one-shape-per-variant bf16 smoke run (~5s) so the
default gate still catches a layout or routing regression, and the full matrix --
every shape, every dtype, plus the guard that NCHW input stays off these kernels --
is L1.

Run standalone (no built cuDNN extension required; runs the full matrix):

    python test/python/norm/cutedsl/test_frost_norm_nhwc.py
    pytest test/python/norm/cutedsl/test_frost_norm_nhwc.py -m L1
"""

import functools
import importlib
import pkgutil
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _install_repo_cudnn_stub():
    """Point ``cudnn`` at the repo's pure-Python subtree, for STANDALONE runs only.

    A prebuilt ``cudnn`` wheel has no ``norm`` subtree, so running this file directly
    would otherwise need the extension rebuilt. Registering a stub rooted at the repo's
    ``python/cudnn`` lets the pure-Python kernels import without it.

    Deliberately NOT done at import time under pytest: ``sys.modules["cudnn"]`` is
    process-global, so a stub installed during collection replaces the real module for
    every other test in the session -- ``norm/graph/test_batchnorm.py`` and friends
    need the genuine frontend and fail to import against a stub. Under pytest this
    file uses the same plain importorskip as its siblings.
    """
    if "cudnn" in sys.modules:
        return
    repo_cudnn = Path(__file__).resolve().parents[4] / "python" / "cudnn"
    if not (repo_cudnn / "norm").is_dir():
        return
    stub = types.ModuleType("cudnn")
    stub.__path__ = [str(repo_cudnn)]
    stub.__file__ = str(repo_cudnn / "__init__.py")
    stub.pygraph = type("pygraph", (), {})  # the cudnn.frost lifecycle patch needs this

    import enum

    class _DataType(enum.Enum):
        NOT_SET = 0
        HALF = 1
        BFLOAT16 = 2
        FLOAT = 3

    stub.data_type = _DataType  # the graph analyzer imports this at module scope
    sys.modules["cudnn"] = stub


if __name__ == "__main__":
    # Must precede the cudnn.norm import below, hence up here rather than in the
    # usual __main__ block at the bottom.
    _install_repo_cudnn_stub()

pytest.importorskip("cudnn.norm", reason="frost norm kernels require a built cudnn frontend")

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402

TOL = {
    torch.float32: dict(fwd=3e-4, bwd=2e-4),
    torch.float16: dict(fwd=6e-2, bwd=4e-2),
    torch.bfloat16: dict(fwd=8e-2, bwd=7e-2),
}
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# (label, variant, N, C, H, W, groups, expected fprop kernel, expected bprop kernel).
# The GroupNorm entries are chosen for the channel-tile geometry, not for coverage of
# C alone: cpg=2 exercises the lane-spans-several-groups map (a lane holds V/cpg whole
# groups), cpg=256 and cpg=1024 exercise the tile-grows-to-hold-one-group map that a
# 128-channel cap used to reject outright. 7x7 runs with one CTA per image (no grid
# barrier); 56x56 runs cooperative.
CASES = [
    ("BN 64x56x56", NormVariant.BATCH_NORM, 8, 64, 56, 56, None, "batchnorm_nhwc_sm100", "batchnorm_nhwc_sm100"),
    ("BN 256x7x7", NormVariant.BATCH_NORM, 8, 256, 7, 7, None, "batchnorm_nhwc_sm100", "batchnorm_nhwc_sm100"),
    ("IN 64x56x56", NormVariant.INSTANCE_NORM, 8, 64, 56, 56, None, "instancenorm_nhwc_sm100", "instancenorm_nhwc_sm100"),
    ("IN 256x7x7", NormVariant.INSTANCE_NORM, 8, 256, 7, 7, None, "instancenorm_nhwc_sm100", "instancenorm_nhwc_sm100"),
    ("GN cpg=2 64x56x56", NormVariant.GROUP_NORM, 8, 64, 56, 56, 32, "groupnorm_nhwc_sm100", "groupnorm_nhwc_sm100"),
    ("GN cpg=8 256x14x14", NormVariant.GROUP_NORM, 8, 256, 14, 14, 32, "groupnorm_nhwc_sm100", "groupnorm_nhwc_sm100"),
    ("GN cpg=256 512x7x7", NormVariant.GROUP_NORM, 8, 512, 7, 7, 2, "groupnorm_nhwc_sm100", "groupnorm_nhwc_sm100"),
    ("GN cpg=1024 2048x7x7", NormVariant.GROUP_NORM, 4, 2048, 7, 7, 2, "groupnorm_nhwc_sm100", "groupnorm_nhwc_sm100"),
]


@contextmanager
def _record_dispatch():
    """Record which kernel module each ``norm_fprop``/``norm_bprop`` call lands on."""
    import cudnn.norm.bprop.kernels as bpk
    import cudnn.norm.fprop.kernels as fpk

    seen = {"fwd": [], "bwd": []}
    patched = []
    for pkg, slot, entry in ((fpk, "fwd", "forward"), (bpk, "bwd", "backward")):
        for info in pkgutil.iter_modules(pkg.__path__):
            mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
            orig = getattr(mod, entry, None)
            if orig is None:
                continue

            def wrap(orig, slot=slot, name=info.name):
                @functools.wraps(orig)
                def inner(*a, **k):
                    seen[slot].append(name)
                    return orig(*a, **k)

                return inner

            setattr(mod, entry, wrap(orig))
            patched.append((mod, entry, orig))
    try:
        yield seen
    finally:
        for mod, entry, orig in patched:
            setattr(mod, entry, orig)


def _check(name, got, ref, dtype, kind):
    got = got.float()
    ref = ref.float()
    amax = (got - ref).abs().max().item()
    scale = max(1.0, ref.abs().max().item())
    ok = amax <= TOL[dtype][kind] * scale
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<12} {str(dtype):>16} maxabs={amax:.3e}")
    return ok


def _check_flag(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<12} {detail}")
    return ok


def _reference(variant, x, g, b, groups):
    """BatchNorm/InstanceNorm/GroupNorm in fp32, built so autograd gives the grads."""
    if variant is NormVariant.BATCH_NORM:
        return F.batch_norm(x, None, None, g, b, True, 0.1, 1e-5)
    ng = x.shape[1] if variant is NormVariant.INSTANCE_NORM else groups
    return F.group_norm(x, ng, g, b, 1e-5)


def _check_case(label, variant, N, C, H, W, groups, want_fwd, want_bwd, dtypes=DTYPES):
    all_ok = True
    print(f"\n=== {label} NHWC N={N} C={C} {H}x{W} groups={groups} ===")
    for dtype in dtypes:
        torch.manual_seed(0)
        x = torch.randn(N, C, H, W, device="cuda", dtype=dtype).to(memory_format=torch.channels_last)
        g = torch.randn(C, device="cuda", dtype=dtype)
        b = torch.randn(C, device="cuda", dtype=dtype)

        fwd = dict(eps=1e-5)
        bwd = {}
        if variant is NormVariant.BATCH_NORM:
            fwd["training"] = True
        elif variant is NormVariant.INSTANCE_NORM:
            fwd["num_groups"] = C
            bwd["num_groups"] = C
        else:
            fwd["num_groups"] = groups
            bwd["num_groups"] = groups

        with _record_dispatch() as seen:
            y, mean, rstd = norm_fprop(variant, x, g, b, **fwd)
            # dy inherits y's layout -- the chain a real model produces.
            dy = torch.randn_like(y)
            dx, dgamma, dbeta = norm_bprop(variant, dy, x, g, mean, rstd, has_beta=True, **bwd)
            torch.cuda.synchronize()

        xr = x.float().detach().requires_grad_(True)
        gr = g.float().detach().requires_grad_(True)
        br = b.float().detach().requires_grad_(True)
        y_ref = _reference(variant, xr, gr, br, groups)
        dx_ref, dg_ref, db_ref = torch.autograd.grad(y_ref, [xr, gr, br], grad_outputs=dy.float())

        all_ok &= _check("fwd y", y, y_ref.detach(), dtype, "fwd")
        all_ok &= _check("bwd dx", dx, dx_ref, dtype, "bwd")
        all_ok &= _check("bwd dgamma", dgamma, dg_ref, dtype, "bwd")
        all_ok &= _check("bwd dbeta", dbeta, db_ref, dtype, "bwd")

        cl = torch.channels_last
        all_ok &= _check_flag("y layout", y.is_contiguous(memory_format=cl), "channels_last preserved")
        all_ok &= _check_flag("dy layout", dy.is_contiguous(memory_format=cl), "inherited from y")
        all_ok &= _check_flag("dx layout", dx.is_contiguous(memory_format=cl), "channels_last preserved")
        all_ok &= _check_flag("fwd kernel", want_fwd in seen["fwd"], f"{seen['fwd']} want {want_fwd}")
        all_ok &= _check_flag("bwd kernel", want_bwd in seen["bwd"], f"{seen['bwd']} want {want_bwd}")
    return all_ok


def _check_nchw_unaffected():
    """A contiguous NCHW input must NOT be pulled onto the channels-last kernels."""
    print("\n=== NCHW inputs stay off the NHWC kernels ===")
    all_ok = True
    for label, variant, groups in (
        ("BatchNorm", NormVariant.BATCH_NORM, None),
        ("InstanceNorm", NormVariant.INSTANCE_NORM, None),
        ("GroupNorm", NormVariant.GROUP_NORM, 32),
    ):
        torch.manual_seed(0)
        N, C, H, W = 8, 128, 14, 14
        x = torch.randn(N, C, H, W, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(C, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(C, device="cuda", dtype=torch.bfloat16)
        fwd = dict(eps=1e-5)
        bwd = {}
        if variant is NormVariant.BATCH_NORM:
            fwd["training"] = True
        elif variant is NormVariant.INSTANCE_NORM:
            fwd["num_groups"] = C
            bwd["num_groups"] = C
        else:
            fwd["num_groups"] = groups
            bwd["num_groups"] = groups
        with _record_dispatch() as seen:
            y, mean, rstd = norm_fprop(variant, x, g, b, **fwd)
            dy = torch.randn_like(y)
            norm_bprop(variant, dy, x, g, mean, rstd, has_beta=True, **bwd)
            torch.cuda.synchronize()
        used = seen["fwd"] + seen["bwd"]
        nhwc = [k for k in used if k.endswith("_nhwc_sm100")]
        all_ok &= _check_flag(f"{label} NCHW", not nhwc, f"used {used}")
    return all_ok


# One cheap (7x7) case per variant for the L0 gate. The GroupNorm pick is the
# cpg=256 tile, since an over-tight channel-tile cap silently dropping that shape
# back onto the transposing path is the regression most worth catching early.
SMOKE = ["BN 256x7x7", "IN 256x7x7", "GN cpg=256 512x7x7"]


def _run(cases, dtypes):
    assert torch.cuda.is_available(), "CUDA required"
    ok = True
    for case in cases:
        ok &= _check_case(*case, dtypes=dtypes)
    return ok


def main(full=True):
    if full:
        ok = _run(CASES, DTYPES) and _check_nchw_unaffected()
    else:
        ok = _run([c for c in CASES if c[0] in SMOKE], [torch.bfloat16])
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
def test_frost_norm_nhwc_smoke():
    """One shape per variant, bf16: layout and dispatch regressions, cheaply."""
    _entry(full=False)


@pytest.mark.L1
def test_frost_norm_nhwc():
    """Full matrix: every shape x fp32/fp16/bf16, plus the NCHW routing guard."""
    _entry(full=True)
