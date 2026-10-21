# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LN/RMS backward at the narrow vector widths, where the partial flush skipped.

The pipelined backward accumulates dgamma/dbeta in registers and flushes them to
a ``[ctas, C]`` partial buffer, which a second kernel reduces. That flush used
wide stores sized as ``V // 4`` segments -- correct for the wide vectors, and
silently a no-op for the narrow ones: ``_bwd_pipe_cfg`` hands back V=2 for fp32
at C=64 and V=1 at C=32 or C=96, where ``V // 4`` is zero, the unrolled loop has
no body, and nothing is stored at all. The finalize then summed whatever the
allocator had last left in the buffer.

What made it survive is the shape of the damage: ``dx`` is computed on a
different path and stayed correct, so anything checking the input gradient --
which is most things -- passed. Only dgamma and dbeta were garbage, and only at
channel counts no test used. It reached a torch-level op before anything caught
it, as a 100%-mismatched dscale.

So the sweep below is over C rather than over anything semantic: C is what picks
V, and V is what was wrong. Both gradients are checked at every width, in both
dtypes, and dx is checked too so a regression that trades one for the other is
still visible.

Run standalone (no built cuDNN extension required):

    python test/python/norm/cutedsl/test_frost_norm_small_c_backward.py
"""

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _install_repo_cudnn_stub():
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

from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402

# C picks the vector width: fp32 gives V = 1, 2, 1, 4, 2, 4 across these, and bf16
# reaches 8 at the top. The V<4 entries are the ones that stored nothing.
_CHANNELS = [32, 64, 96, 128, 192, 256]
_ROWS = 8
_OK = True


def _expect(cond, msg):
    global _OK
    _OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")


def _vector_width(C, dtype):
    """The V the backward will actually use, for the report line."""
    try:
        from cudnn.norm.bprop.kernels.layernorm_sm100 import _bwd_pipe_cfg

        cfg = _bwd_pipe_cfg(C, 4 if dtype is torch.float32 else 2)
        return cfg[3] if cfg else "-"
    except Exception:  # noqa: BLE001 - the width is reporting detail, not the assertion
        return "?"


def _case(variant, C, dtype):
    has_beta = variant is NormVariant.LAYER_NORM
    tol = 1e-4 if dtype is torch.float32 else 3e-2
    torch.manual_seed(763)
    x = torch.randn(_ROWS, C, device="cuda", dtype=dtype)
    g = torch.randn(C, device="cuda", dtype=dtype)
    b = torch.randn(C, device="cuda", dtype=dtype) if has_beta else None
    dy = torch.randn_like(x)

    _, mean, rstd = norm_fprop(variant, x, g, b, eps=1e-5, normalized_shape=[C])
    dx, dgamma, dbeta = norm_bprop(variant, dy, x, g, mean, rstd, has_beta=has_beta, normalized_shape=[C])
    torch.cuda.synchronize()

    xr = x.double().detach().requires_grad_(True)
    gr = g.double().detach().requires_grad_(True)
    br = b.double().detach().requires_grad_(True) if has_beta else None
    yr = F.layer_norm(xr, [C], gr, br, 1e-5) if has_beta else F.rms_norm(xr, [C], gr, eps=1e-5)
    targets = [t for t in (xr, gr, br) if t is not None]
    ref = torch.autograd.grad(yr, targets, dy.double())

    name = str(variant).split(".")[-1]
    label = f"{name} C={C} {str(dtype).replace('torch.', '')} (V={_vector_width(C, dtype)})"
    pairs = [("dx", dx, ref[0]), ("dgamma", dgamma, ref[1])]
    if has_beta:
        pairs.append(("dbeta", dbeta, ref[2]))
    for what, got, want in pairs:
        scale = max(1.0, want.abs().max().item())
        err = (got.double() - want).abs().max().item() / scale
        _expect(err <= tol, f"{label}: {what} rel {err:.2e}")


def main():
    assert torch.cuda.is_available(), "CUDA required"
    for variant in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM):
        for dtype in (torch.float32, torch.bfloat16):
            for C in _CHANNELS:
                _case(variant, C, dtype)
    print("\n" + ("ALL PASS" if _OK else "SOME FAILED"))
    sys.exit(0 if _OK else 1)


if __name__ == "__main__":
    main()


@pytest.mark.L0
def test_frost_norm_small_c_backward():
    """dgamma/dbeta at every vector width the backward can pick."""
    try:
        main()
    except SystemExit as exc:
        assert not exc.code, "checks failed -- see captured stdout"
