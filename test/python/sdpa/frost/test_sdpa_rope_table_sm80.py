# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared standalone RoPE tables retain Torch's FP32 trigonometry."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


def _reference(frequencies, width, device):
    angles = frequencies.to(dtype=torch.float32, device=device).reshape(frequencies.shape[0], -1)[:, :width]
    return torch.stack((angles.cos(), angles.sin()), -1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["compact", "strided", "reshape", "cpu"])
def test_rope_table_conversion_and_current_strides(dtype, layout, monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.rope_table_sm80 import compile_plan, prepare

    torch.manual_seed(1828)
    rows, width = 17, 64
    device = torch.device("cuda", torch.cuda.current_device())
    plan = compile_plan(rows, width, device)
    values = torch.randn(rows, width + 8, device="cpu" if layout == "cpu" else device, dtype=dtype)
    if layout == "strided":
        values = torch.randn(rows * 2, (width + 8) * 3, device=device, dtype=dtype)[::2, ::3]
    elif layout == "reshape":
        values = torch.randn(rows, 9, 8, device=device, dtype=dtype).transpose(1, 2)

    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("RoPE execution invoked JIT"))
    for frequencies in (values, values.clone().mul_(2)):
        result = prepare(plan, frequencies, device, torch.cuda.current_stream().cuda_stream)
        torch.testing.assert_close(result, _reference(frequencies, width, device), atol=0, rtol=0)


def test_rope_table_large_angles_and_special_values():
    from cudnn.sdpa.rope_table_sm80 import compile_plan, prepare

    torch.manual_seed(928)
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.randn(1024, 256, device=device)[:, ::2]
    values.mul_(torch.pow(10.0, torch.empty_like(values).uniform_(-40, 38)))
    values[0, :8] = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), torch.pi, 1e20, -1e20], device=device)
    plan = compile_plan(1024, 128, device)
    result = prepare(plan, values, device, torch.cuda.current_stream().cuda_stream)
    reference = _reference(values, 128, device)
    torch.testing.assert_close(result, reference, atol=0, rtol=0, equal_nan=True)
    assert torch.equal(result[0, :2].view(torch.int32), reference[0, :2].view(torch.int32))


@pytest.mark.parametrize("shape,detail", [((16, 64), "rows "), ((17, 63), "last dim ")])
def test_rope_table_rejects_invalid_shape_before_allocation(shape, detail, monkeypatch):
    from cudnn.sdpa.rope_table_sm80 import compile_plan, prepare

    values = torch.zeros(shape, device="cuda")
    plan = compile_plan(17, 64, values.device)
    monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("invalid angle table allocated output"))
    with pytest.raises(ValueError, match=detail):
        prepare(plan, values, values.device, torch.cuda.current_stream().cuda_stream)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_rope_table_restores_caller_device(backward, explicit, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80
    from test_sdpa_bwd_staged_sm80 import test_rope_staged_replay
    from test_sdpa_rope_staged_copy_sm80 import _forward_case, _forward_run, _forward_check

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    operand_device = torch.cuda.current_device()
    caller_device = (operand_device + 1) % torch.cuda.device_count()
    adapter = SdpaBwdDslSm80 if backward else SdpaFwdDslSm80
    original = adapter.execute
    target = torch.cuda.current_stream(operand_device)

    def execute(self, *args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            return original(self, *args, **kwargs)
        kwargs["current_stream"] = target.cuda_stream if explicit else None
        with torch.cuda.device(caller_device):
            result = original(self, *args, **kwargs)
            assert torch.cuda.current_device() == caller_device
        return result

    monkeypatch.setattr(adapter, "execute", execute)
    if backward:
        test_rope_staged_replay(torch.bfloat16, 128)
    else:
        api, values, workspace, owners = _forward_case()
        _forward_run(api, values, workspace)
        _forward_check(values)


def test_rope_table_artifact_reloads_in_fresh_process(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import cudnn

    child = r"""
import json, sys
from pathlib import Path
import torch, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache
from cudnn.sdpa.rope_table_sm80 import compile_plan, prepare
package, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve()
if reload == "1":
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh-process RoPE artifact reload invoked JIT")
    cute.compile = forbidden
device = torch.device("cuda", torch.cuda.current_device())
plan = compile_plan(17, 64, device)
torch.manual_seed(829)
values = torch.randn(17, 192, device=device)[:, ::3]
result = prepare(plan, values, device, torch.cuda.current_stream().cuda_stream)
torch.testing.assert_close(result, torch.stack((values.cos(), values.sin()), -1), atol=0, rtol=0)
if reload == "1":
    assert hasattr(plan.artifact, "_compiled_cache_raw")
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        process = subprocess.run([sys.executable, "-c", child, cudnn.__file__, str(reload)], env=env, capture_output=True, text=True, timeout=180)
        assert process.returncode == 0, process.stdout[-2000:] + process.stderr[-5000:]
        results.append(json.loads(process.stdout.strip().splitlines()[-1]))
    assert results[0]["misses"] == 1 and results[0]["hits"] == 0
    assert results[1]["misses"] == 0 and results[1]["hits"] == 1


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_execution_uses_prepared_table(backward, dtype, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80
    from test_sdpa_bwd_staged_sm80 import test_rope_staged_replay
    from test_sdpa_sm80_thd_forward_prepared import test_dense_staged_pointer_launch_and_rope_replay

    # Intercept only the actual execution; the independent references retain
    # their own Torch sin/cos. This was RED on both original adapters.
    for adapter in (SdpaFwdDslSm80, SdpaBwdDslSm80):
        original = adapter.execute

        def execute(self, *args, _original=original, **kwargs):
            with monkeypatch.context() as guard:
                for name in ("cos", "sin"):
                    guard.setattr(torch.Tensor, name, lambda *a, **k: pytest.fail("RoPE execution rebuilt Torch trigonometry"))
                guard.setattr(torch, "stack", lambda *a, **k: pytest.fail("RoPE execution rebuilt Torch interleave"))
                return _original(self, *args, **kwargs)

        monkeypatch.setattr(adapter, "execute", execute)
    if backward:
        test_rope_staged_replay(dtype, 128)
    else:
        test_dense_staged_pointer_launch_and_rope_replay(128, 128, True, dtype, monkeypatch)


@pytest.mark.parametrize("overflow", ["stride", "product"])
@pytest.mark.gpu_exclusive
def test_rope_table_physical_int64_stride_and_replay(overflow):
    import gc
    from cudnn.sdpa.rope_table_sm80 import compile_plan, prepare
    from cudnn.sdpa.fwd.kernels.sm80.rope_table import compile_table

    rows, width = (2, 64) if overflow == "stride" else (4, 64)
    stride = 2**32 + 128 if overflow == "stride" else 2**31 - 64
    prefix = 0 if overflow == "stride" else 2**31
    span = prefix + (rows - 1) * stride + width
    if torch.cuda.mem_get_info()[0] < span * 4 + 2**30:
        pytest.skip("insufficient free memory for the physical angle-stride control")
    try:
        owner = torch.empty(span, device="cuda", dtype=torch.float32)
    except torch.OutOfMemoryError:
        pytest.skip("insufficient memory for the wide-stride angle allocation")
    # Keep signed-Int32 wraparound addresses inside allocated guard storage,
    # so a deliberately narrowed address calculation fails numerically.
    for row in range(rows):
        offset = row * stride
        wrapped = (offset + 2**31) % 2**32 - 2**31
        if wrapped != offset:
            owner[prefix + wrapped : prefix + wrapped + width].fill_(float("nan"))
    values = owner.as_strided((rows, width), (stride, 1), storage_offset=prefix)
    torch.manual_seed(301)
    values.copy_(torch.randn(rows, width, device="cuda"))
    device = values.device
    plan = compile_plan(rows, width, device)

    def run():
        return prepare(plan, values, device, torch.cuda.current_stream().cuda_stream)

    torch.testing.assert_close(run(), _reference(values, width, device), atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            result = run()
        compile_table.cache_clear()
        del plan
        gc.collect()
        values.add_(0.125)
        result.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(result, _reference(values, width, device), atol=0, rtol=0)
    finally:
        graph.reset()
