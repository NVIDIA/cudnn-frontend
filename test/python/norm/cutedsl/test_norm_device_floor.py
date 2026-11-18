# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The torch-tensor norm API refuses a pre-Hopper GPU before it builds a kernel.

The kernels lower through bulk async copies, mbarrier transaction counts and
thread-block clusters. On an SM8x part most shapes used to fail inside NVVM
("unsupported operation") while a few small ones happened to compile, so the
same call worked or crashed depending on the row width. ``norm_fprop`` and
``norm_bprop`` now raise one ``NotImplementedError`` that names the device and
points at the graph API, which the cuDNN backend serves on those GPUs.

The checks fake the device capability, so they run on any CUDA GPU; on an SM90+
part the faked call would otherwise build and launch normally.
"""

import pytest
import torch

pytest.importorskip("cudnn.norm", reason="frost norm kernels require a built cudnn frontend")

from cudnn.norm._lazy import dsl_unavailable

requirement_error = dsl_unavailable("FROST norm tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

import cudnn.norm.utils as norm_utils  # noqa: E402
from cudnn.norm import NormVariant, norm_bprop, norm_fprop  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture
def fake_sm89(monkeypatch):
    monkeypatch.setattr(norm_utils, "_DEVICE_FLOOR_ERROR", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 9))


@pytest.mark.L0
@pytest.mark.norm_rejects_before_launch
@pytest.mark.parametrize(
    "variant,shape,kwargs",
    [
        (NormVariant.LAYER_NORM, (8, 512), {"normalized_shape": [512]}),
        (NormVariant.LAYER_NORM, (8, 8192), {"normalized_shape": [8192]}),
        (NormVariant.RMS_NORM, (8, 1024), {"normalized_shape": [1024]}),
        (NormVariant.GROUP_NORM, (4, 64, 8, 8), {"num_groups": 8}),
        (NormVariant.INSTANCE_NORM, (4, 64, 8, 8), {}),
        (NormVariant.BATCH_NORM, (4, 64, 8, 8), {"training": True}),
    ],
    ids=["ln-d512", "ln-d8192", "rms", "gn", "in", "bn"],
)
def test_forward_refuses_a_pre_hopper_device(fake_sm89, variant, shape, kwargs):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    channels = shape[-1] if variant in (NormVariant.LAYER_NORM, NormVariant.RMS_NORM) else shape[1]
    gamma = torch.ones(channels, device="cuda", dtype=torch.bfloat16)
    beta = None if variant == NormVariant.RMS_NORM else torch.zeros_like(gamma)
    with pytest.raises(NotImplementedError, match=r"SM90 or newer GPU .* found SM89"):
        norm_fprop(variant, x, gamma, beta, **kwargs)


@pytest.mark.L0
@pytest.mark.norm_rejects_before_launch
def test_backward_refuses_a_pre_hopper_device(fake_sm89):
    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    mean = torch.zeros(8, device="cuda", dtype=torch.float32)
    rstd = torch.ones(8, device="cuda", dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="graph API"):
        norm_bprop(NormVariant.LAYER_NORM, torch.randn_like(x), x, gamma, mean, rstd, normalized_shape=[512])


@pytest.mark.L0
@pytest.mark.norm_rejects_before_launch
def test_an_argument_error_still_wins_over_the_device_floor(fake_sm89):
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones(64, device="cuda", dtype=torch.float32)  # wrong dtype on purpose
    with pytest.raises(ValueError, match="read .* at the input dtype"):
        norm_fprop(NormVariant.LAYER_NORM, x, gamma, torch.zeros_like(gamma), normalized_shape=[64])


@pytest.mark.L0
@pytest.mark.norm_rejects_before_launch
def test_the_floor_matches_this_device():
    major, _ = torch.cuda.get_device_capability()
    reason = norm_utils.unsupported_device_reason(torch.device("cuda"))
    assert (reason is None) == (major >= 9), reason
