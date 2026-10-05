# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared data copies around the existing standalone SM80 RoPE lowering."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_sdpa_bwd_staged_sm80 import test_rope_staged_replay as _backward_replay

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
def test_rope_backward_uses_prepared_data_copies(dtype, d, monkeypatch):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    original = SdpaBwdDslSm80.execute

    def execute(self, *args, **kwargs):
        def forbidden(*args, **kwargs):
            pytest.fail("RoPE execution rebuilt tensor staging copies")

        with monkeypatch.context() as guard:
            for name in ("copy_", "zero_", "as_strided"):
                guard.setattr(torch.Tensor, name, forbidden)
            return original(self, *args, **kwargs)

    monkeypatch.setattr(SdpaBwdDslSm80, "execute", execute)
    # This independent autograd reference also changes the angle table and Q
    # after capture, updates forward O/Stats, and checks all replayed gradients.
    _backward_replay(dtype, d)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [128, 256])
def test_rope_forward_uses_prepared_data_copies(dtype, d, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80
    from test_sdpa_sm80_thd_forward_prepared import test_dense_staged_pointer_launch_and_rope_replay

    original = SdpaFwdDslSm80.execute

    def execute(self, *args, **kwargs):
        def forbidden(*args, **kwargs):
            pytest.fail("RoPE execution rebuilt tensor staging copies")

        with monkeypatch.context() as guard:
            for name in ("copy_", "zero_", "as_strided"):
                guard.setattr(torch.Tensor, name, forbidden)
            return original(self, *args, **kwargs)

    monkeypatch.setattr(SdpaFwdDslSm80, "execute", execute)
    test_dense_staged_pointer_launch_and_rope_replay(d, d, True, dtype, monkeypatch)


def _forward_case(dtype=torch.bfloat16, *, staged=True, wide=None):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80

    torch.manual_seed(1823)
    d, dv, sq, sk = 128, 96 if staged else 128, 17, 33
    values = {}
    owners = []
    for name, heads, seq, dim in (("q", 4, sq, d), ("k", 2, sk, d), ("v", 2, sk, dv), ("o", 4, sq, dv)):
        value = torch.randn(2, seq, heads, dim + int(staged), device="cuda", dtype=dtype)[..., :dim].transpose(1, 2).mul_(0.2)
        if name == wide:
            # Exercise a real allocation and live addresses across 2**32;
            # metadata-only large-stride tests cannot detect pointer truncation.
            shape = tuple(value.shape)
            strides = (2**32 + 16, *value.stride()[1:])
            span = 1 + sum((n - 1) * st for n, st in zip(shape, strides))
            if torch.cuda.mem_get_info()[0] < span * value.element_size() + 2**30:
                pytest.skip("wide-stride control needs about 9 GiB free")
            owner = torch.empty(span, device="cuda", dtype=dtype)
            value = owner.as_strided(shape, strides).copy_(value)
            owners.append(owner)
        values[name] = value
    values["stats"] = torch.empty((2, 4, sq + 3), device="cuda")[..., :sq]
    values["bias"] = torch.randn(1, 4, sq, sk, device="cuda").mul_(0.1)
    values["sink"] = torch.tensor([-0.4, 0.2, 1.0, -0.8], device="cuda")
    # Noncontiguous double angles exercise the existing float32 table lowering.
    values["freqs"] = torch.randn(sk, d, device="cuda", dtype=torch.float64)[:, ::2].mul_(0.1)
    api = SdpaFwdDslSm80(
        *(values[n] for n in ("q", "k", "v", "o", "stats")),
        rope_max_s=sk,
        has_sink=True,
        bias_present=True,
        bias_fp32=True,
        is_causal=True,
        scale_softmax=0.13,
    )
    assert api.check_support()
    api.compile()
    size = api.scratch_workspace_bytes()
    workspace = torch.empty(size, device="cuda", dtype=torch.uint8) if size else None
    return api, values, workspace, owners


def _forward_run(api, values, workspace, stream=None):
    api.execute(
        *(values[n] for n in ("q", "k", "v", "o", "stats")),
        sinks=values["sink"],
        bias_tensor=values["bias"],
        rope_freqs=values["freqs"],
        workspace=workspace,
        current_stream=stream,
    )


def _forward_check(values):
    q, k, v = (values[n].double() for n in ("q", "k", "v"))
    angles = values["freqs"].float().to(q.device).double()

    def rotate(t):
        a, b = t.chunk(2, -1)
        cs, sn = angles[: t.shape[2]].cos()[None, None], angles[: t.shape[2]].sin()[None, None]
        return torch.cat((a * cs - b * sn, b * cs + a * sn), -1).to(values["q"].dtype).double()

    scores = rotate(q) @ rotate(k).repeat_interleave(2, 1).transpose(-1, -2) * 0.13 + values["bias"].double()
    scores.masked_fill_(torch.arange(k.shape[2], device=q.device)[None, :] > torch.arange(q.shape[2], device=q.device)[:, None], -torch.inf)
    lse = torch.cat((scores, values["sink"].double()[None, :, None, None].expand(2, 4, q.shape[2], 1)), -1).logsumexp(-1)
    ref = (scores - lse[..., None]).exp() @ v.repeat_interleave(2, 1)
    torch.testing.assert_close(values["o"].double(), ref, atol=0.003, rtol=0.025)
    torch.testing.assert_close(values["stats"].double(), lse, atol=0.002, rtol=0.002)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_rope_forward_features_stream_and_replay(dtype, staged, explicit):
    from contextlib import nullcontext
    from cuda.bindings import driver

    api, values, workspace, owners = _forward_case(dtype, staged=staged)
    target, ambient = torch.cuda.Stream(), torch.cuda.current_stream()
    target.wait_stream(ambient)

    def run():
        with nullcontext() if explicit else torch.cuda.stream(target):
            mode = torch.cuda.get_sync_debug_mode()
            try:
                torch.cuda.set_sync_debug_mode("error")
                _forward_run(api, values, workspace, driver.CUstream(target.cuda_stream) if explicit else None)
            finally:
                torch.cuda.set_sync_debug_mode(mode)

    run()
    ambient.wait_stream(target)
    _forward_check(values)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture, stream=target):
            run()
        values["q"].mul_(0.75)
        values["freqs"].add_(0.2)
        values["sink"].sub_(0.3)
        values["o"].fill_(float("nan"))
        capture.replay()
        _forward_check(values)
        assert torch.cuda.current_stream() == ambient
    finally:
        capture.reset()


def test_rope_forward_cpu_angles():
    api, values, workspace, owners = _forward_case()
    values["freqs"] = values["freqs"].cpu()
    _forward_run(api, values, workspace)
    _forward_check(values)


@pytest.mark.parametrize("role", ["q", "o"])
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.gpu_exclusive
def test_rope_forward_real_wide_strides(role, staged):
    api, values, workspace, owners = _forward_case(wide=role, staged=staged)
    _forward_run(api, values, workspace)
    _forward_check(values)
    assert owners


@pytest.mark.parametrize("explicit", [False, True])
def test_rope_backward_launch_stream(explicit, monkeypatch):
    from contextlib import nullcontext
    from cuda.bindings import driver
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    original = SdpaBwdDslSm80.execute
    target, ambient = torch.cuda.Stream(), torch.cuda.current_stream()

    def execute(self, *args, **kwargs):
        # The helper owns reference generation and changed-input replay. Order
        # each eager producer before the explicit consumer; capture is already
        # on a side stream and tests that stream's ordinary current-stream path.
        if torch.cuda.is_current_stream_capturing():
            return original(self, *args, **kwargs)
        target.wait_stream(ambient)
        with nullcontext() if explicit else torch.cuda.stream(target):
            kwargs["current_stream"] = driver.CUstream(target.cuda_stream) if explicit else None
            mode = torch.cuda.get_sync_debug_mode()
            try:
                torch.cuda.set_sync_debug_mode("error")
                result = original(self, *args, **kwargs)
            finally:
                torch.cuda.set_sync_debug_mode(mode)
        ambient.wait_stream(target)
        assert torch.cuda.current_stream() == ambient
        return result

    monkeypatch.setattr(SdpaBwdDslSm80, "execute", execute)
    _backward_replay(torch.bfloat16, 128)
