# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The persistent InstanceNorm backward must keep each warp on one channel.

The warp-per-row backward gives every warp a FIXED channel and grid-strides over
the rows that share it. That only holds while the total warp count is a multiple
of the groups-per-sample: stride a non-multiple and every row past the first wave
lands on a different channel than the partials were accumulated for.

The count also has to be a whole number of CTAs, and satisfying the two in
sequence does not satisfy both -- rounding up to gps and then up to WPB breaks
the gps divisibility again. gps=5 on 68 SMs produced 2184 warps (2184 % 5 == 4).
dgamma was off by 179 and dx by 8.1, silently.

Two things make it easy to miss, and the test is shaped around both:

* It needs enough rows to exceed the persistent cap. Below that the grid is one
  wave, every warp sees exactly one row, and the mapping cannot drift -- the same
  channel count passes at N=600 and fails at N=1000.
* The cap is ``4 * SM_count * WPB``, so whether a given shape trips it depends on
  the GPU. The reported failure was on 68 SMs; on 148 the same shape is fine.
  So the SM count is simulated rather than taken from the device, and the sweep
  covers counts this machine does not have.

Run standalone (no built cuDNN extension required):

    python test/python/norm/cutedsl/test_frost_norm_persistent_map.py
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

# Channel counts that do NOT divide the warps-per-block (8), so the two roundings
# can disagree; 4 and 8 are the controls that divide it and were always fine.
_CHANNELS = [3, 5, 7, 4, 8]
# SM counts to simulate. 68 is where this was reported; the rest bracket it.
_SM_COUNTS = [68, 132, 148, 228]
_N = 2000  # enough rows to pass the persistent cap at every count above
_S = 64
_OK = True


def _expect(cond, msg):
    global _OK
    _OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")


def _case(C, nsm, monkey):
    """InstanceNorm backward with the SM count forced to ``nsm``."""
    import cudnn.norm.bprop.kernels.instancenorm_warp_sm100 as warp_mod

    monkey(warp_mod, nsm)
    torch.manual_seed(0)
    x = torch.randn(_N, C, _S, device="cuda", dtype=torch.float32)
    g = torch.randn(C, device="cuda", dtype=torch.float32)
    b = torch.randn(C, device="cuda", dtype=torch.float32)
    dy = torch.randn_like(x)

    _, mean, rstd = norm_fprop(NormVariant.INSTANCE_NORM, x, g, b, eps=1e-5, num_groups=C)
    dx, dgamma, dbeta = norm_bprop(NormVariant.INSTANCE_NORM, dy, x, g, mean, rstd, has_beta=True, num_groups=C)
    torch.cuda.synchronize()

    xr = x.double().detach().requires_grad_(True)
    gr = g.double().detach().requires_grad_(True)
    br = b.double().detach().requires_grad_(True)
    yr = F.group_norm(xr, C, gr, br, 1e-5)
    ref = torch.autograd.grad(yr, (xr, gr, br), dy.double())

    for what, got, want in (("dx", dx, ref[0]), ("dgamma", dgamma, ref[1]), ("dbeta", dbeta, ref[2])):
        scale = max(1.0, want.abs().max().item())
        err = (got.double() - want).abs().max().item() / scale
        _expect(err <= 1e-4, f"C={C} SMs={nsm}: {what} rel {err:.2e}")


def main():
    assert torch.cuda.is_available(), "CUDA required"
    import cudnn.norm.bprop.kernels.instancenorm_warp_sm100 as warp_mod

    saved = warp_mod._SM_COUNT if hasattr(warp_mod, "_SM_COUNT") else None

    def force(mod, n):
        mod._SM_COUNT = n  # _nsm() caches into this

    try:
        for nsm in _SM_COUNTS:
            print(f"\n=== simulated SM count {nsm} ===")
            for C in _CHANNELS:
                _case(C, nsm, force)
    finally:
        warp_mod._SM_COUNT = saved

    print("\n" + ("ALL PASS" if _OK else "SOME FAILED"))
    sys.exit(0 if _OK else 1)


if __name__ == "__main__":
    main()


@pytest.mark.L0
def test_frost_norm_persistent_instancenorm_map():
    """Channel mapping holds past the first wave, at every simulated SM count."""
    try:
        main()
    except SystemExit as exc:
        assert not exc.code, "checks failed -- see captured stdout"
