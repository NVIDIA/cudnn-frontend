# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# _adversarial_bf16 adapts adv_bf16 from FlashInfer's tests/gemm/test_batch_invariant_mm_bf16.py
# (e2d163fb3d46) to the device selected by the cuDNN test fixture.

from types import SimpleNamespace

import pytest
import torch

from determinism_utils import assert_bitwise
from .matmul import build_matmul

try:
    import flashinfer
except ModuleNotFoundError as exc:
    if exc.name != "flashinfer":
        raise
    pytest.skip("FlashInfer is required for the rollout integration tests", allow_module_level=True)

from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.gemm.batch_invariant.api import mm_bf16_batch_invariant

pytestmark = pytest.mark.L0


def _adversarial_bf16(rows, cols, seed, device):
    """BF16 values with cancellation, a wide exponent range, outliers, signed zeros and subnormals."""
    g = torch.Generator().manual_seed(seed)
    e = (torch.rand(rows, cols, generator=g) * 2 - 1) * 20
    x = (torch.rand(rows, cols, generator=g) * 2 - 1) * torch.exp2(e)
    k = torch.arange(0, cols - 1, 8)
    x[:, k + 1] = -x[:, k] * (1 + 2.0**-7)
    x[torch.randint(0, 64, (rows, cols), generator=g) == 0] *= 2.0**12
    z = torch.randint(0, 97, (rows, cols), generator=g)
    x[z == 0] = 0.0
    x[z == 1] = -0.0
    x[z == 2] = 2.0**-130
    return x.to(torch.bfloat16).to(device)


def _weight(n, k, layout, seed, device):
    b = _adversarial_bf16(k, n, seed, device)
    return b.T.contiguous().T if layout == "k" else b


def _rollout(a, b, bias=None):
    return flashinfer.mm_bf16(a, b, bias=bias, backend="auto", batch_invariant=True)


@pytest.mark.parametrize("n,k", [(2048, 4096), (128, 8192)])
@pytest.mark.parametrize("layout", ["k", "n"])
@pytest.mark.parametrize("biased", [False, True])
def test_rollout_matches_direct_cudnn_training_forward(gemm_device, n, k, layout, biased):
    row = 129
    x = _adversarial_bf16(257, k, 170, gemm_device)
    w = _weight(n, k, layout, 171, gemm_device)
    bias = _adversarial_bf16(1, n, 172, gemm_device)[0] if biased else None
    legacy = mm_bf16_batch_invariant(x[row : row + 1], w, bias=bias)
    rollout = _rollout(x[row : row + 1], w, bias)
    training = build_matmul(x.unsqueeze(0), w.unsqueeze(0), None if bias is None else bias.view(1, 1, n))()["out"]
    assert_bitwise({"out": rollout}, {"out": legacy}, "rollout vs original FlashInfer GEMM")
    assert_bitwise({"out": rollout}, {"out": training[:, row, :]}, "rollout vs cuDNN training row")


@pytest.mark.parametrize("n,k,layout,biased", [(4096, 4096, "n", False), (64, 8192, "k", False), (4096, 4096, "k", True), (128, 8192, "n", True)])
def test_batch_invariant_plan_owns_selection_inside_autotune(gemm_device, n, k, layout, biased, monkeypatch):
    a = _adversarial_bf16(300, k, 91, gemm_device)
    b = _weight(n, k, layout, 92, gemm_device)
    bias = _adversarial_bf16(1, n, 93, gemm_device)[0] if biased else None
    sizes = (1, 16, 300)
    legacy = [mm_bf16_batch_invariant(a[:m], b, bias=bias) for m in sizes]

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("cuDNN batch-invariant GEMM consulted the FlashInfer autotuner")

    monkeypatch.setattr(AutoTuner, "choose_one", unexpected_lookup)
    with autotune(True):
        tuned = [_rollout(a[:m], b, bias) for m in sizes]
    post = [_rollout(a[:m], b, bias) for m in sizes]
    for m, expected, during, after in zip(sizes, legacy, tuned, post, strict=True):
        assert_bitwise({"out": during}, {"out": expected}, f"during autotune, M={m}")
        assert_bitwise({"out": after}, {"out": expected}, f"after autotune, M={m}")
    assert_bitwise({"out": post[1]}, {"out": post[2][:16]}, "batch invariance after autotune")


def test_missing_batch_invariant_support_raises(monkeypatch):
    from flashinfer import batch_invariant as bi
    from flashinfer.gemm import gemm_base
    from flashinfer.gemm.batch_invariant import api

    monkeypatch.setattr(api, "check_device", lambda *args: None)
    monkeypatch.setattr(gemm_base, "CUDNN_AVAILABLE", True)
    monkeypatch.setattr(gemm_base, "cudnn", SimpleNamespace(), raising=False)
    a = torch.ones(8, 8, dtype=torch.bfloat16)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="BATCH_INVARIANT numerical note"):
        _rollout(a, a)


def test_expanded_activation_matches_single_row(gemm_device):
    m, n, k = 16, 4096, 4096
    row = _adversarial_bf16(1, k, 111, gemm_device)
    b = _adversarial_bf16(k, n, 112, gemm_device)
    expanded = row.expand(m, k)
    assert expanded.stride(0) == 0
    assert_bitwise({"out": _rollout(expanded, b)}, {"out": _rollout(row, b).expand(m, n)}, "broadcast activation rows")
