# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gamma/beta/DY are READ at the I/O dtype, so a mismatch must be refused.

These kernels do not convert their parameters -- they read the buffer as the
input's dtype. An fp32 gamma against a bf16 input is therefore reinterpreted
bit-for-bit, and the result is garbage rather than merely imprecise.

What made it worth a guard rather than a docstring is the compile cache. Its
key carries the INPUT dtype, not the parameter's, so the two calls below share
a cache entry: compile a shape once with matching operands and a later
mismatched call for the same shape is served straight out of cache. That is how
it actually surfaced -- a GroupNorm case in one test file compiled
``(bf16, M=64, ...)`` with bf16 parameters, and an InstanceNorm case in another
file reused it with fp32 parameters and returned a 3.1 absolute error against
an fp64 reference. Neither test was wrong on its own; only the order was.

The graph engines already decline mixed precision at check time
(``uniform_dtype``), so this closes the same hole on the torch API.
"""

import sys
import types
from pathlib import Path

import pytest
import torch


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


@pytest.mark.L0
@pytest.mark.norm_rejects_before_launch
def test_forward_refuses_mismatched_parameters():
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    g = torch.ones(64, device="cuda", dtype=torch.float32)  # wrong dtype on purpose
    b = torch.zeros(64, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="read .* at the input dtype"):
        norm_fprop(NormVariant.LAYER_NORM, x, g, b, eps=1e-5, normalized_shape=[64])


@pytest.mark.L0
def test_backward_refuses_mismatched_parameters():
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    g = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(64, device="cuda", dtype=torch.bfloat16)
    _, mean, rstd = norm_fprop(NormVariant.LAYER_NORM, x, g, b, eps=1e-5, normalized_shape=[64])
    dy = torch.randn(4, 64, device="cuda", dtype=torch.float32)  # wrong dtype on purpose
    with pytest.raises(ValueError, match="read .* at the input dtype"):
        norm_bprop(NormVariant.LAYER_NORM, dy, x, g, mean, rstd, normalized_shape=[64])


@pytest.mark.L0
def test_cache_entry_is_not_shared_across_parameter_dtypes():
    """The ordering that produced it: same shape, matching operands first."""
    import torch.nn.functional as F

    # compile (bf16, M=64, ...) with bf16 parameters
    x1 = torch.randn(4, 8, 32, device="cuda", dtype=torch.bfloat16)
    g1 = torch.ones(8, device="cuda", dtype=torch.bfloat16)
    b1 = torch.zeros_like(g1)
    norm_fprop(NormVariant.GROUP_NORM, x1, g1, b1, eps=1e-5, num_groups=4)
    torch.cuda.synchronize()

    # the same cache entry, now asked for with fp32 parameters
    x2 = torch.randn(4, 24, 8, 8, device="cuda", dtype=torch.bfloat16)
    g2 = torch.ones(24, device="cuda", dtype=torch.float32)
    b2 = torch.zeros_like(g2)
    with pytest.raises(ValueError, match="read .* at the input dtype"):
        norm_fprop(NormVariant.INSTANCE_NORM, x2, g2, b2, eps=1e-5, num_groups=24)

    # and the matching-dtype call is still correct
    y, _, _ = norm_fprop(NormVariant.INSTANCE_NORM, x2, g2.bfloat16(), b2.bfloat16(), eps=1e-5, num_groups=24)
    torch.cuda.synchronize()
    ref = F.group_norm(x2.double(), 24, g2.double(), b2.double(), 1e-5)
    err = (y.double() - ref).abs().max().item()
    torch.testing.assert_close(
        y, ref, check_dtype=False, msg=lambda detail: f"matching-dtype InstanceNorm wrong after the shared compile (max {err:.2e}): {detail}"
    )
