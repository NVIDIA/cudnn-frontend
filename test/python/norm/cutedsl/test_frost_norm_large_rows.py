# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Long-row LayerNorm / RMSNorm: the CGA split, fprop + bprop.

Rows past ~16K elements leave the one-CTA-per-row design and are split across a
cluster, each CTA reducing its own slice and sharing partials through distributed
shared memory. That path has failure modes the ordinary shapes cannot reach, so this
file covers the (R, D) corners specifically:

* **D beyond what a single CTA can stage.** D=65536 bf16 needs 262208 bytes against a
  232448 limit; before the shared-memory guard this did not fail a check, it failed
  the *launch*.
* **Fewer rows than clusters**, and R=1. The grid is ``nclusters * CGA`` CTAs
  grid-striding over rows, so a cluster may get one row or none. Every CTA in a
  cluster must still reach the same cluster barriers the same number of times --
  divergence there hangs rather than fails.
* **R not a multiple of anything** (31), since the row loop is a grid stride.
* **Both sides of the routing threshold.** The forward takes the split from D=16384;
  the backward only from D=32768, or from 16384 when there are too few rows to fill
  the machine otherwise. Below that the existing pipelined backward is simply better
  -- routing D=16384 R=2048 to the split measured 0.23 against 0.60 -- so one case
  asserts the split is NOT taken, which is the half of a threshold that silently
  rots.

Each case checks numerics against PyTorch autograd for dx, dgamma and dbeta, and
asserts which kernel actually ran on each side.

Run standalone (no built cuDNN extension required):

    python test/python/norm/cutedsl/test_frost_norm_large_rows.py
    pytest test/python/norm/cutedsl/test_frost_norm_large_rows.py -m L1
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

    Never at import time under pytest: ``sys.modules["cudnn"]`` is process-global, so
    a stub installed during collection replaces the real module for every other test
    in the session.
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
    # cudnn.frost.buffers imports this at module scope and only calls into it
    # from a method, so a placeholder is enough to reach the DSL version gate.
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

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402

LN, RMS = NormVariant.LAYER_NORM, NormVariant.RMS_NORM
CGA_F = "layernorm_cga_sm100"
# Relative tolerances: the reductions here are over 16K-128K elements, so the fp32
# reference and the kernel accumulate in different orders over a long chain.
TOL = {torch.float32: 1e-5, torch.float16: 6e-2, torch.bfloat16: 1.5e-1}
SMALL = [torch.float32, torch.float16, torch.bfloat16]
BIG = [torch.float16, torch.bfloat16]  # fp32 at these sizes is several GB with the ref


def _case(R, D, *, dtypes, cga_bwd=True, smoke=False):
    return dict(R=R, D=D, dtypes=dtypes, cga_bwd=cga_bwd, smoke=smoke)


CASES = [
    _case(1, 65536, dtypes=SMALL),  # a single row
    _case(31, 16384, dtypes=SMALL, smoke=True),  # fewer rows than clusters, odd R
    _case(64, 32768, dtypes=SMALL),
    _case(512, 32768, dtypes=BIG, smoke=True),
    _case(512, 65536, dtypes=BIG),
    _case(128, 131072, dtypes=BIG),  # the longest row supported
    # Below the backward's threshold with enough rows to fill the machine: the
    # pipelined backward must keep this one (0.60 of achievable against the split's
    # 0.23). Asserts the threshold from the side that otherwise rots unnoticed.
    _case(2048, 16384, dtypes=BIG, cga_bwd=False, smoke=True),
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


def _check(name, got, ref, dtype):
    if got is None and ref is None:
        return True
    amax = (got.float() - ref.float()).abs().max().item()
    scale = max(1.0, ref.float().abs().max().item())
    ok = amax <= TOL[dtype] * scale
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<10} {str(dtype):>16} rel={amax / scale:.2e}")
    return ok


def _flag(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<10} {detail}")
    return ok


def _check_case(c, dtypes=None):
    ok = True
    R, D = c["R"], c["D"]
    print(f"\n=== R={R} D={D} (bwd split expected: {c['cga_bwd']}) ===")
    for variant, vn, has_beta in ((LN, "LN", True), (RMS, "RMS", False)):
        for dtype in dtypes or c["dtypes"]:
            torch.manual_seed(0)
            x = torch.randn(R, D, device="cuda", dtype=dtype)
            g = torch.randn(D, device="cuda", dtype=dtype)
            b = torch.randn(D, device="cuda", dtype=dtype) if has_beta else None
            kw = dict(normalized_shape=[D])

            with _record_dispatch() as seen:
                y, mean, rstd = norm_fprop(variant, x, g, b, eps=1e-5, **kw)
                dy = torch.randn_like(y)
                dx, dgamma, dbeta = norm_bprop(variant, dy, x, g, mean, rstd, has_beta=has_beta, **kw)
                torch.cuda.synchronize()

            xr = x.float().detach().requires_grad_(True)
            gr = g.float().detach().requires_grad_(True)
            br = b.float().detach().requires_grad_(True) if b is not None else None
            y_ref = F.layer_norm(xr, [D], gr, br, 1e-5) if has_beta else F.rms_norm(xr, [D], gr, eps=1e-5)
            ins = [t for t in (xr, gr, br) if t is not None]
            grads = torch.autograd.grad(y_ref, ins, grad_outputs=dy.float())

            print(f"  -- {vn} {str(dtype).replace('torch.', '')}")
            ok &= _check("fwd y", y, y_ref.detach(), dtype)
            ok &= _check("bwd dx", dx, grads[0], dtype)
            ok &= _check("bwd dgamma", dgamma, grads[1], dtype)
            if has_beta:
                ok &= _check("bwd dbeta", dbeta, grads[2], dtype)
            ok &= _flag("fwd kernel", CGA_F in seen["fwd"], f"{seen['fwd']} want {CGA_F}")
            ok &= _flag("bwd kernel", (CGA_F in seen["bwd"]) == c["cga_bwd"], f"{seen['bwd']} split={c['cga_bwd']}")
            del x, g, b, y, dy, dx, dgamma, dbeta, xr, gr, br, y_ref, grads
            torch.cuda.empty_cache()
    return ok


def main(full=True):
    assert torch.cuda.is_available(), "CUDA required"
    cases = CASES if full else [c for c in CASES if c["smoke"]]
    ok = True
    for c in cases:
        ok &= _check_case(c, dtypes=None if full else [torch.bfloat16])
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
def test_frost_norm_large_rows_smoke():
    """The (R, D) corners that reach the cluster path, bf16."""
    _entry(full=False)


@pytest.mark.L1
def test_frost_norm_large_rows():
    """Full long-row matrix: every (R, D) corner x dtype, fprop + bprop."""
    _entry(full=True)
