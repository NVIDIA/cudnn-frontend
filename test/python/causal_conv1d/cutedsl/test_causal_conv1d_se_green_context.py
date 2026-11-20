# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SE scheduling, autograd, and graph replay on Rubin green-context streams."""

import subprocess
import sys
import pytest
import torch
from determinism_utils import assert_bitwise, snapshot
from test_causal_conv1d_packed import _dsl
from test_causal_conv1d_se_persistent import _inputs, _public, _reference_check

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA green-context setup failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


@pytest.fixture
def green_pair():
    _, _, cuda = _dsl()
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Rubin locality-domain coverage requires SM107")
    from cudnn.causal_conv1d_se.runtime import require_arch

    try:
        require_arch((10, 7))
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    if _check(cuda.cuDriverGetVersion()) < 13050 or not hasattr(cuda, "cuDevSmResourceSplit"):
        pytest.skip("locality partitions with supplemental SMs require CUDA 13.5 driver and bindings")
    anchor = torch.empty(1, device="cuda")
    device = _check(cuda.cuDeviceGet(torch.cuda.current_device()))
    count = _check(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, device))
    if count != 2:
        pytest.skip(f"test requires two locality domains; GPU exposes {count}")
    kind = cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    whole = _check(cuda.cuDeviceGetDevResource(device, kind))
    params = []
    for index in range(2):
        group = cuda.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
        # LOCALITY_DOMAIN_ID | SUPPLEMENTAL_SMS. Some bindings omit the latter name.
        group.flags = 6
        group.localityDomainId = index
        group.smCount = 0
        params.append(group)
    parts, _ = _check(cuda.cuDevSmResourceSplit(2, whole, 0, params))
    contexts, streams = [], []
    try:
        for part in parts:
            desc = _check(cuda.cuDevResourceGenerateDesc([part], 1))
            context = _check(cuda.cuGreenCtxCreate(desc, device, 0))
            contexts.append(context)
            streams.append(_check(cuda.cuGreenCtxStreamCreate(context, int(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING), 0)))
        yield streams, [int(p.sm.smCount) for p in parts], int(whole.sm.smCount)
    finally:
        torch.cuda.synchronize()
        # Expandable allocator segments retain the external stream. Release
        # cached allocations while that stream and its green context are live.
        torch.cuda.empty_cache()
        for stream in streams:
            _check(cuda.cuStreamDestroy(stream))
        for context in contexts:
            _check(cuda.cuGreenCtxDestroy(context))
        del anchor


@pytest.mark.parametrize("partition", [0, 1])
def test_green_schedule_and_changed_offset_capture(green_pair, partition):
    from cudnn._torch_stream import as_torch_stream, stream_context
    from cudnn.causal_conv1d_se import packed
    from cudnn.causal_conv1d_se.runtime import stream_multiprocessors

    streams, counts, whole = green_pair
    stream = streams[partition]
    args = _inputs([1, 8191], 768)
    torch.cuda.synchronize()
    with stream_context(stream, args[0].device):
        assert stream_multiprocessors(args[0].device) == counts[partition]
        route, splits = packed._backward_schedule(args[0], args[1], args[2], None, args[4], None, True)
        assert route == "persistent"
        assert splits == 2 * counts[partition] // 3
        _public(args)
    assert stream_multiprocessors(args[0].device) == whole
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=as_torch_stream(stream, args[0].device)):
            mode = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                captured = _public(args)
            finally:
                torch.cuda.set_sync_debug_mode(mode)
        torch.cuda.synchronize()
        with torch.no_grad():
            args[0].neg_()
            args[4].mul_(0.5)
            args[3].copy_(torch.tensor([0, 4097, 8192], device=args[0].device, dtype=torch.int32))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        actual = snapshot(captured)
        with stream_context(stream, args[0].device):
            expected = _public(args)
        torch.cuda.synchronize()
        assert_bitwise(actual, expected, "green-context changed-offset graph replay")
        _reference_check(actual, args, [4097, 4095])
    finally:
        graph.reset()


def test_cold_autograd_worker_binds_its_execution_context():
    _dsl()
    from cudnn.causal_conv1d_se.runtime import require_arch

    try:
        require_arch(torch.cuda.get_device_capability())
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    code = """
import torch
from cudnn.ops import causal_conv1d
x = torch.randn(64, 24, device="cuda", dtype=torch.bfloat16).requires_grad_()
w = torch.randn(24, 4, device="cuda", dtype=x.dtype).requires_grad_()
b = torch.zeros(24, device="cuda", dtype=x.dtype, requires_grad=True)
cu = torch.tensor([0, 1, 64], device="cuda", dtype=torch.int32)
y = causal_conv1d(x.T[None], w, b, activation="silu", cu_seqlens=cu, backend="se")
grads = torch.autograd.grad(y, (x,w,b), torch.ones_like(y))
torch.cuda.synchronize()
assert all(torch.isfinite(v).all() for v in (y, *grads))
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
