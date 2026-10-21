# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channels-last shapes the native NHWC kernels decline.

BN/IN/GN keep an NHWC input un-``contiguous()``-ed so the channels-last kernels
can run on it. That is only sound while one of them actually does. The tile
geometry has real constraints -- C must divide into vectors and channel tiles --
so plenty of ordinary channel counts have no NHWC tile at all, and those fall
through to the contiguous kernels.

The fall-through was silently wrong, and specifically for BatchNorm, because the
(N, C, H, W) -> (N, C, S) view it takes EXISTS in channels-last memory: merging H
and W is legal there, so ``reshape`` returns a strided view rather than copying,
and the contiguous kernel then addresses it as if it were packed. fp32 [4,24,8,8]
came back as 6144 untouched NaNs. GroupNorm and InstanceNorm took the same route
but survived it, because their rowwise view is not expressible and reshape was
forced to copy -- the same bug, hidden by an accident of layout.

So these cases check the whole contract on an output buffer poisoned with NaN:
every element written, and the values right. A C that the NHWC tile *does*
accept is interleaved as a control, since a fix that simply stopped using the
native kernels would also pass a correctness-only test.

The large-M case is here for a different reason. Past M = 131072 the NHWC
BatchNorm switches to a shared-memory pixel cache (KS > 0), and that loop's
index collided with the shifted-statistics list added for the variance fix --
so the whole route failed to compile while every tested shape stayed below the
threshold. It is a cheap case to keep.

Run standalone (no built cuDNN extension required):

    python test/python/norm/cutedsl/test_frost_norm_nhwc_fallback.py
"""

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _install_repo_cudnn_stub():
    """Point ``cudnn`` at the repo subtree, for STANDALONE runs only."""
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

BN, IN, GN = NormVariant.BATCH_NORM, NormVariant.INSTANCE_NORM, NormVariant.GROUP_NORM
_OK = True


def _expect(cond, msg):
    global _OK
    _OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")


def _reference(variant, x, g, b, num_groups):
    xd = x.double()
    if variant is BN:
        return F.batch_norm(xd, None, None, g.double(), b.double(), True, 0.1, 1e-5)
    return F.group_norm(xd, num_groups, g.double(), b.double(), 1e-5)


def _case(variant, shape, num_groups, dtype):
    """Run channels-last, into a NaN-poisoned channels-last out= buffer."""
    n, c, h, w = shape
    name = str(variant).split(".")[-1]
    label = f"{name} {shape} {str(dtype).replace('torch.', '')} NHWC"
    torch.manual_seed(0)
    x = torch.randn(n, c, h, w, device="cuda", dtype=dtype).to(memory_format=torch.channels_last)
    # parameters share the input dtype: these kernels READ gamma/beta at the I/O
    # dtype, and norm_fprop now rejects a mismatch rather than reinterpreting it
    g = torch.ones(c, device="cuda", dtype=dtype)
    b = torch.zeros(c, device="cuda", dtype=dtype)
    y = torch.full((n, c, h, w), float("nan"), device="cuda", dtype=dtype).to(memory_format=torch.channels_last)

    kw = dict(training=True) if variant is BN else dict(num_groups=num_groups)
    out = norm_fprop(variant, x, g, b, eps=1e-5, out=(y, None, None), **kw)[0]
    torch.cuda.synchronize()

    # Both the caller's buffer and the returned tensor must be complete.
    unwritten = int(torch.isnan(y).sum())
    _expect(unwritten == 0, f"{label}: out= buffer fully written ({y.numel() - unwritten}/{y.numel()})")
    _expect(int(torch.isnan(out).sum()) == 0, f"{label}: returned tensor is finite")

    ref = _reference(variant, x, g, b, num_groups)
    tol = 1e-5 if dtype is torch.float32 else 2e-2
    err = (y.double() - ref).abs().max().item()
    _expect(err <= tol, f"{label}: out= buffer matches PyTorch (max {err:.2e})")
    # and the input must come back untouched -- a transposing fallback that
    # wrote through an alias would show up here
    _expect(x.is_contiguous(memory_format=torch.channels_last), f"{label}: input still channels-last")


# C values chosen around the NHWC tile constraints: 24 and 40 have no tile for
# fp32, 32 and 64 do. The accepted ones are controls.
_SHAPES = [
    (BN, (4, 24, 8, 8), None),
    (BN, (4, 40, 8, 8), None),
    (BN, (4, 32, 8, 8), None),
    (BN, (4, 64, 8, 8), None),
    (IN, (4, 24, 8, 8), 24),
    (IN, (4, 32, 8, 8), 32),
    (GN, (4, 24, 8, 8), 6),
    (GN, (4, 32, 8, 8), 8),
]


def main():
    assert torch.cuda.is_available(), "CUDA required"
    for dtype in (torch.float32, torch.bfloat16):
        for variant, shape, ng in _SHAPES:
            print(f"\n=== {str(variant).split('.')[-1]} {shape} {dtype} ===")
            _case(variant, shape, ng, dtype)

    # Past M = 131072 the NHWC BatchNorm takes its shared-memory cache route.
    print("\n=== BATCH_NORM large M (shared-memory cache route) ===")
    _case(BN, (33, 32, 64, 64), None, torch.float32)

    print("\n" + ("ALL PASS" if _OK else "SOME FAILED"))
    sys.exit(0 if _OK else 1)


if __name__ == "__main__":
    main()


@pytest.mark.L0
def test_frost_norm_nhwc_fallback():
    """Channels-last shapes with and without a native NHWC tile."""
    try:
        main()
    except SystemExit as exc:
        assert not exc.code, "checks failed -- see captured stdout"
