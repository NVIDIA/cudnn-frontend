# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layout correctness for the sm_100 norm kernels: all five variants, fprop + bprop.

Covers LayerNorm, RMSNorm, GroupNorm, InstanceNorm and BatchNorm for fp32 / fp16 /
bf16, on contiguous input; and the three layout-sensitive variants (BN, IN, GN) on
channels-last input as well.

LayerNorm and RMSNorm have no channels-last entry on purpose. They reduce the last
LOGICAL dim, which is strided under channels-last, so there is no coalesced native
map for them -- a channels-last LN *must* transpose. Rather than enshrine that as an
expected-dirty case, the layout axis simply does not apply to them.

Four things are asserted per case. The last three are the point of the file:

1. **Numerics** against PyTorch autograd.
2. **No hidden transpose.** ``Tensor.contiguous`` and ``Tensor.reshape`` are patched to
   record every call that lands in NEW storage, and the count must be zero. This is
   stronger than checking the output layout: ``x.reshape(R, M)`` silently materialises
   a transposed copy for a channels-last tensor, which is a full extra read+write of
   the tensor hiding inside what reads like a view.
3. **The layout survives** -- the output's memory format matches the input's.
4. **The expected kernel ran.** Channels-last cases pin the native kernel by name;
   contiguous cases assert the negative, that no ``*_nhwc_sm100`` kernel was used, so
   routing is pinned from both sides without hardcoding heuristic-dependent names.

(2) and (3) are not redundant. A kernel that transposes in and transposes back out
passes (3) and fails (2); one that forgets to transpose back fails (3). And numerics
alone catches neither -- the result stays perfectly correct either way, which is
exactly how InstanceNorm and GroupNorm shipped a channels-last forward that did not
exist.

The chain is the realistic one: ``dy = torch.randn_like(y)`` inherits ``y``'s layout
rather than being forced channels-last by hand. Forcing it hides the case where fprop
loses the layout, because the backward then sees a ``dy`` no real model would hand it.

Split by cost, because the repo gates on ``-m L0`` and a comparable L0 norm test runs
in 3-5s: the L0 entry is one bf16 case per variant per applicable layout, and the full
matrix (every shape, every dtype) is L1.

Run standalone (no built cuDNN extension required; runs the full matrix):

    python test/python/norm/cutedsl/test_frost_norm_layout.py
    pytest test/python/norm/cutedsl/test_frost_norm_layout.py -m L1
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

LN, RMS = NormVariant.LAYER_NORM, NormVariant.RMS_NORM
GN, IN, BN = NormVariant.GROUP_NORM, NormVariant.INSTANCE_NORM, NormVariant.BATCH_NORM


def _case(label, variant, shape, *, cl=False, ns=None, groups=None, has_beta=True,
          want=None, want_fwd=None, want_bwd=None, smoke=False):
    return dict(label=label, variant=variant, shape=shape, cl=cl, ns=ns,
                groups=groups, has_beta=has_beta, smoke=smoke,
                want_fwd=want_fwd or want, want_bwd=want_bwd or want)


# Contiguous: every variant, so all five flavors are covered fprop + bprop.
CONTIG = [
    _case("LN 2D", LN, (256, 1024), ns=[1024], smoke=True),
    _case("RMS 2D", RMS, (256, 1024), ns=[1024], has_beta=False, smoke=True),
    _case("LN 4D ns=[H,W]", LN, (8, 128, 14, 14), ns=[14, 14]),
    # A row too long for one CTA to stage. It must route to the CGA split, which
    # reduces across a cluster through distributed shared memory -- NOT fail to
    # launch (it used to), and not silently drop to the streaming fallback (0.17 of
    # achievable against the split's 0.68). The backward has no CGA path yet.
    _case("LN long row CGA", LN, (64, 65536), ns=[65536], want_fwd="layernorm_cga_sm100"),
    _case("RMS long row CGA", RMS, (64, 65536), ns=[65536], has_beta=False,
          want_fwd="layernorm_cga_sm100"),
    _case("BN", BN, (8, 128, 14, 14), smoke=True),
    _case("IN", IN, (8, 128, 14, 14)),
    _case("GN g=32", GN, (8, 128, 14, 14), groups=32),
    # Too few (sample, group) rows to fill the machine. The forward was one CTA per
    # row (4-16 CTAs on 148 SMs, 0.01-0.02 of achievable); the backward was worse --
    # groupnorm_fast declines on these and the fallback was the original
    # one-atomic-per-element kernel at 0.00. Both must take the cluster split.
    _case("GN few rows", GN, (2, 256, 56, 56), groups=2,
          want_fwd="groupnorm_cga_sm100", want_bwd="groupnorm_cga_sm100"),
    _case("IN few rows", IN, (2, 8, 112, 112),
          want_fwd="groupnorm_cga_sm100", want_bwd="groupnorm_cga_sm100"),
]

# Channels-last: the three layout-sensitive variants. The GroupNorm entries are picked
# for channel-tile geometry rather than for coverage of C -- cpg=2 exercises the
# lane-spans-several-groups map, cpg=256 and cpg=1024 the tile-grows-to-hold-one-group
# map that a 128-channel cap used to reject outright. 7x7 runs one CTA per image (no
# grid barrier); 56x56 runs cooperative.
CL = [
    _case("BN 64x56x56", BN, (8, 64, 56, 56), cl=True, want="batchnorm_nhwc_sm100"),
    _case("BN 256x7x7", BN, (8, 256, 7, 7), cl=True, want="batchnorm_nhwc_sm100", smoke=True),
    _case("IN 64x56x56", IN, (8, 64, 56, 56), cl=True, want="instancenorm_nhwc_sm100"),
    _case("IN 256x7x7", IN, (8, 256, 7, 7), cl=True, want="instancenorm_nhwc_sm100", smoke=True),
    _case("GN cpg=2 64x56x56", GN, (8, 64, 56, 56), cl=True, groups=32, want="groupnorm_nhwc_sm100"),
    _case("GN cpg=8 256x14x14", GN, (8, 256, 14, 14), cl=True, groups=32, want="groupnorm_nhwc_sm100"),
    _case("GN cpg=256 512x7x7", GN, (8, 512, 7, 7), cl=True, groups=2, want="groupnorm_nhwc_sm100", smoke=True),
    _case("GN cpg=1024 2048x7x7", GN, (4, 2048, 7, 7), cl=True, groups=2, want="groupnorm_nhwc_sm100"),
]
CASES = CONTIG + CL


@contextmanager
def _no_hidden_copies():
    """Record every ``contiguous``/``reshape`` call that lands in NEW storage.

    Both are no-ops on a tensor whose layout already suits the request and silently
    full copies otherwise, which is what makes an accidental transpose so easy to miss:
    ``x.reshape(R, M)`` on a channels-last tensor reads like a view and is not one.
    Comparing ``data_ptr()`` separates the two.
    """
    events = []
    orig_contig, orig_reshape = torch.Tensor.contiguous, torch.Tensor.reshape

    def contiguous(self, *a, **k):
        out = orig_contig(self, *a, **k)
        if out.data_ptr() != self.data_ptr():
            events.append(f"contiguous{tuple(self.shape)} strides={self.stride()}")
        return out

    def reshape(self, *a, **k):
        out = orig_reshape(self, *a, **k)
        if out.data_ptr() != self.data_ptr():
            events.append(f"reshape{tuple(self.shape)}->{tuple(a)} strides={self.stride()}")
        return out

    torch.Tensor.contiguous, torch.Tensor.reshape = contiguous, reshape
    try:
        yield events
    finally:
        torch.Tensor.contiguous, torch.Tensor.reshape = orig_contig, orig_reshape


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
    if got is None and ref is None:
        return True
    amax = (got.float() - ref.float()).abs().max().item()
    scale = max(1.0, ref.float().abs().max().item())
    ok = amax <= TOL[dtype][kind] * scale
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<12} {str(dtype):>16} maxabs={amax:.3e}")
    return ok


def _flag(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<12} {detail}")
    return ok


def _reference(c, x, g, b):
    """The variant's fp32 reference, built so autograd supplies the gradients."""
    v, ns = c["variant"], c["ns"]
    if v is BN:
        return F.batch_norm(x, None, None, g, b, True, 0.1, 1e-5)
    if v is IN:
        return F.group_norm(x, x.shape[1], g, b, 1e-5)
    if v is GN:
        return F.group_norm(x, c["groups"], g, b, 1e-5)
    gs = g.reshape(ns)
    if v is LN:
        return F.layer_norm(x, ns, gs, b.reshape(ns), 1e-5)
    return F.rms_norm(x, ns, gs, eps=1e-5)


def _kwargs(c):
    fwd, bwd = dict(eps=1e-5), {}
    v = c["variant"]
    if v is BN:
        fwd["training"] = True
    elif v is IN:
        fwd["num_groups"] = bwd["num_groups"] = c["shape"][1]
    elif v is GN:
        fwd["num_groups"] = bwd["num_groups"] = c["groups"]
    else:
        fwd["normalized_shape"] = bwd["normalized_shape"] = c["ns"]
    return fwd, bwd


def _check_case(c, dtypes=DTYPES):
    ok = True
    fmt = torch.channels_last if c["cl"] else torch.contiguous_format
    print(f"\n=== {c['label']} {'channels_last' if c['cl'] else 'contiguous'} {c['shape']} ===")
    for dtype in dtypes:
        torch.manual_seed(0)
        x = torch.randn(*c["shape"], device="cuda", dtype=dtype)
        if c["cl"]:
            x = x.to(memory_format=torch.channels_last)
        glen = 1
        for d in (c["ns"] if c["ns"] else [c["shape"][1]]):
            glen *= d
        g = torch.randn(glen, device="cuda", dtype=dtype)
        b = torch.randn(glen, device="cuda", dtype=dtype) if c["has_beta"] else None
        fwd, bwd = _kwargs(c)

        with _no_hidden_copies() as copies, _record_dispatch() as seen:
            y, mean, rstd = norm_fprop(c["variant"], x, g, b, **fwd)
            # dy inherits y's layout -- the chain a real model produces.
            dy = torch.randn_like(y)
            dx, dgamma, dbeta = norm_bprop(
                c["variant"], dy, x, g, mean, rstd, has_beta=c["has_beta"], **bwd
            )
            torch.cuda.synchronize()

        xr = x.float().detach().requires_grad_(True)
        gr = g.float().detach().requires_grad_(True)
        br = b.float().detach().requires_grad_(True) if b is not None else None
        y_ref = _reference(c, xr, gr, br)
        ins = [t for t in (xr, gr, br) if t is not None]
        grads = torch.autograd.grad(y_ref, ins, grad_outputs=dy.float())
        dx_ref, dg_ref = grads[0], grads[1]
        db_ref = grads[2] if br is not None else None

        ok &= _check("fwd y", y, y_ref.detach(), dtype, "fwd")
        ok &= _check("bwd dx", dx, dx_ref, dtype, "bwd")
        ok &= _check("bwd dgamma", dgamma, dg_ref, dtype, "bwd")
        ok &= _check("bwd dbeta", dbeta, db_ref, dtype, "bwd")

        ok &= _flag("no transpose", not copies, "; ".join(copies) or "0 materialising copies")
        ok &= _flag("y layout", y.is_contiguous(memory_format=fmt), "matches input")
        ok &= _flag("dy layout", dy.is_contiguous(memory_format=fmt), "inherited from y")
        ok &= _flag("dx layout", dx.is_contiguous(memory_format=fmt), "matches input")

        used = seen["fwd"] + seen["bwd"]
        for slot, want in (("fwd", c["want_fwd"]), ("bwd", c["want_bwd"])):
            if want:
                ok &= _flag(f"{slot} kernel", want in seen[slot], f"{seen[slot]} want {want}")
        if not (c["want_fwd"] or c["want_bwd"]):
            nhwc = [k for k in used if k.endswith("_nhwc_sm100")]
            ok &= _flag("kernel", not nhwc, f"contiguous input stayed off NHWC kernels: {used}")
    return ok


def main(full=True):
    assert torch.cuda.is_available(), "CUDA required"
    cases = CASES if full else [c for c in CASES if c["smoke"]]
    dtypes = DTYPES if full else [torch.bfloat16]
    ok = True
    for c in cases:
        ok &= _check_case(c, dtypes=dtypes)
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
def test_frost_norm_layout_smoke():
    """One bf16 case per variant per applicable layout: layout and dispatch, cheaply."""
    _entry(full=False)


@pytest.mark.L1
def test_frost_norm_layout():
    """Full matrix: every case x fp32/fp16/bf16, both layouts, fprop + bprop."""
    _entry(full=True)
