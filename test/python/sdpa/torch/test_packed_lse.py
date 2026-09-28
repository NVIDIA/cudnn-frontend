# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed Stats preparation: strides, runtime rebinding and torch tracing."""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old
from cudnn.sdpa.fwd.torch_op import thd_lse_to_padded

pytestmark = pytest.mark.L0


def _require_prepared():
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("prepared Stats requires CuTe DSL >= 4.7")
    if cutedsl_arch_requirement_error(torch.cuda.get_device_capability()):
        pytest.skip("installed DSL cannot target this GPU")


def _reference(lse, values, maximum):
    source = lse.detach().cpu()
    result = torch.zeros((len(values) - 1, source.shape[1], maximum, 1), dtype=torch.float32)
    for i, (start, end) in enumerate(zip(values, values[1:])):
        result[i, :, : end - start, 0] = source[start:end].T
    return result


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["token", "head", "gapped"])
def test_packed_lse_runtime_geometry_and_replay(dtype, layout, monkeypatch):
    _require_prepared()
    import cutlass.cute as cute

    for count, heads, maximum in ((7, 3, 5), (263, 5, 137)):
        if count == 263:
            monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("runtime geometry recompiled"))
        source = torch.arange(count * heads, device="cuda", dtype=torch.float32).reshape(count, heads)
        if layout == "head":
            source = source.T.contiguous().T
        elif layout == "gapped":
            owner = torch.empty((count, heads, 3), device="cuda")
            owner[:, :, 1].copy_(source)
            source = owner[:, :, 1]
        values = [0, 0, count // 2, count]
        prefix = torch.tensor(values, dtype=dtype, device="cuda").repeat_interleave(2)[::2]
        result = thd_lse_to_padded(source, prefix, maximum)
        torch.testing.assert_close(result.cpu(), _reference(source, values, maximum), atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                captured = thd_lse_to_padded(source, prefix, maximum)
            values[1], values[2] = 1, maximum
            prefix.copy_(torch.tensor(values, dtype=dtype, device="cuda"))
            source.add_(11)
            captured.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(captured.cpu(), _reference(source, values, maximum), atol=0, rtol=0)
        finally:
            graph.reset()


@pytest.mark.parametrize("shape,values,maximum", [((0, 3), [0, 0, 0], 5), ((0, 3), [0], 5), ((7, 0), [0, 3, 7], 4), ((0, 3), [0, 0], 0)])
def test_packed_lse_empty_extents(shape, values, maximum):
    _require_prepared()
    source = torch.empty(shape, dtype=torch.float32, device="cuda")
    prefix = torch.tensor(values, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(thd_lse_to_padded(source, prefix, maximum).cpu(), _reference(source, values, maximum), atol=0, rtol=0)


def test_packed_lse_does_not_dispatch_torch_indexing(monkeypatch):
    _require_prepared()
    source = torch.arange(21, device="cuda", dtype=torch.float32).reshape(7, 3)
    prefix = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    thd_lse_to_padded(source, prefix, 4)
    with monkeypatch.context() as guard:
        guard.setattr(torch, "searchsorted", lambda *a, **k: pytest.fail("repad dispatched torch searchsorted"))
        guard.setattr(torch, "zeros", lambda *a, **k: pytest.fail("repad dispatched torch zero initialization"))
        result = thd_lse_to_padded(source, prefix, 4)
    torch.testing.assert_close(result.cpu(), _reference(source, [0, 3, 7], 4), atol=0, rtol=0)


def test_packed_lse_fallback_and_differentiable_helper(monkeypatch):
    from cudnn.sdpa import packed_lse

    source = torch.arange(21, device="cuda", dtype=torch.float32).reshape(7, 3)
    prefix = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    with monkeypatch.context() as guard:
        guard.setattr(packed_lse, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.6.2")))
        guard.setattr(packed_lse, "_plan", lambda *a, **k: pytest.fail("old DSL entered prepared code"))
        torch.testing.assert_close(thd_lse_to_padded(source, prefix, 4).cpu(), _reference(source, [0, 3, 7], 4), atol=0, rtol=0)
    source.requires_grad_(True)
    thd_lse_to_padded(source, prefix, 4).sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source), atol=0, rtol=0)


def test_packed_lse_aot_dynamic_and_opcheck():
    _require_prepared()
    from cudnn.sdpa.packed_lse import _compiled_repad

    compiled = torch.compile(thd_lse_to_padded, backend="aot_eager", fullgraph=True, dynamic=True)
    for count, heads, maximum in ((7, 3, 4), (17, 5, 9)):
        source = torch.arange(count * heads, device="cuda", dtype=torch.float32).reshape(count, heads)
        values = [0, count // 2, count]
        prefix = torch.tensor(values, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(compiled(source, prefix, maximum).cpu(), _reference(source, values, maximum), atol=0, rtol=0)
    torch.library.opcheck(_compiled_repad, (source, prefix, maximum))


@pytest.mark.parametrize("device", [0, 1])
def test_packed_lse_operand_device_and_stream(device):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    original = torch.cuda.current_device()
    try:
        with torch.cuda.device(device):
            _require_prepared()
            source = torch.arange(21, device="cuda", dtype=torch.float32).reshape(7, 3)
            prefix = torch.tensor([0, 3, 7], dtype=torch.int64, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                source.add_(2)
                torch.cuda.set_device(1 - device)
                result = thd_lse_to_padded(source, prefix, 4)
                assert torch.cuda.current_device() == 1 - device
            stream.synchronize()
            torch.testing.assert_close(result.cpu(), _reference(source, [0, 3, 7], 4), atol=0, rtol=0)
    finally:
        torch.cuda.set_device(original)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("axis", ["token", "head", "prefix"])
@pytest.mark.parametrize("product", [False, True])
def test_packed_lse_physical_wide_stride(axis, product):
    _require_prepared()
    stride, extent = (2**30 + 8, 5) if product else (2**32 + 16, 2)
    elements = (extent - 1) * stride + 64
    if torch.cuda.mem_get_info()[0] < elements * 4 + 2**30:
        pytest.skip("physical wide stride requires about 17 GiB free")
    try:
        owner = torch.empty(elements, device="cuda", dtype=torch.int32 if axis == "prefix" else torch.float32)
    except torch.OutOfMemoryError:
        pytest.skip("insufficient memory for physical wide stride")
    owner[:32].fill_(0)
    owner[-16:].fill_(43)
    for i in range(extent):
        owner[(i * stride) % 2**32 : (i * stride) % 2**32 + 2].fill_(0)
    if axis == "prefix":
        values = list(range(extent))
        prefix = owner.as_strided((extent,), (stride,))
        prefix.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        source = torch.arange((extent - 1) * 2, device="cuda", dtype=torch.float32).reshape(extent - 1, 2)
    else:
        shape = (extent, 2) if axis == "token" else (2, extent)
        source = owner.as_strided(shape, (stride, 1) if axis == "token" else (1, stride))
        source.copy_(torch.arange(extent * 2, device="cuda", dtype=torch.float32).reshape(shape))
        values = [0, shape[0]]
        prefix = torch.tensor(values, device="cuda", dtype=torch.int32)
    result = thd_lse_to_padded(source, prefix, extent + 1)
    torch.testing.assert_close(result.cpu(), _reference(source, values, extent + 1), atol=0, rtol=0)
    assert (owner[-16:] == 43).all()


def test_packed_lse_artifact_reload(tmp_path):
    _require_prepared()
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import cudnn

    child = r"""
import json, sys
from pathlib import Path
import torch, cudnn, pytest
import cutlass.cute as cute
from cudnn.frost import compiled_cache
folder, package, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve()
sys.path.insert(0, folder)
from test_packed_lse import test_packed_lse_runtime_geometry_and_replay
with pytest.MonkeyPatch.context() as patch:
    if reload == "1":
        patch.setattr(cute, "compile", lambda *a, **k: pytest.fail("artifact reload invoked JIT"))
    test_packed_lse_runtime_geometry_and_replay(torch.int32, "head", patch)
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, str(Path(__file__).parent), cudnn.__file__, str(reload)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert results[0]["misses"] > 0 and results[0]["hits"] == 0
    assert results[1]["misses"] == 0 and results[1]["hits"] > 0


def test_packed_lse_heavily_padded_replay():
    _require_prepared()
    lengths = [2048] + [1] * 63
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    source = torch.arange(values[-1] * 32, dtype=torch.float32, device="cuda").reshape(-1, 32)
    prefix = torch.tensor(values, dtype=torch.int32, device="cuda")
    thd_lse_to_padded(source, prefix, 2048)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            result = thd_lse_to_padded(source, prefix, 2048)
        source.add_(11)
        result.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(result.cpu(), _reference(source, values, 2048), atol=0, rtol=0)
    finally:
        graph.reset()


@pytest.mark.parametrize("axis", ["batch", "head"])
def test_packed_lse_large_grid_dimension_falls_back(axis, monkeypatch):
    from cudnn.sdpa import packed_lse

    source = torch.empty((0, 65536 if axis == "head" else 1), dtype=torch.float32, device="cuda")
    prefix = torch.zeros(65537 if axis == "batch" else 2, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(packed_lse, "_plan", lambda *a, **k: pytest.fail("oversized CUDA grid entered the compiled path"))
    result = thd_lse_to_padded(source, prefix, 1)
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("helper", ["stats", "lengths"])
def test_prepared_metadata_explicit_operand_target(helper, monkeypatch, tmp_path):
    _require_prepared()
    from cudnn.sdpa import packed_lse, varlen_metadata
    from cudnn.sdpa.fwd.kernels import packed_lse as stats_kernel, varlen_metadata as lengths_kernel

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    expected = f"--gpu-arch sm_{major}{minor}"
    module = stats_kernel if helper == "stats" else lengths_kernel
    original = module.compile_cached
    options = []

    def compile_for_operand(*args, **kwargs):
        assert expected in kwargs.get("options", ""), "compiler target did not name the operand GPU"
        options.append(kwargs["options"])
        return original(*args, **kwargs)

    packed_lse._plan.cache_clear()
    varlen_metadata._plan.cache_clear()
    stats_kernel.compile_packed_lse.cache_clear()
    lengths_kernel.compile_metadata.cache_clear()
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setattr(module, "compile_cached", compile_for_operand)
    prefix = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    if helper == "stats":
        source = torch.arange(21, device="cuda", dtype=torch.float32).reshape(7, 3)
        torch.testing.assert_close(thd_lse_to_padded(source, prefix, 4).cpu(), _reference(source, [0, 3, 7], 4), atol=0, rtol=0)
    else:
        result = varlen_metadata.prepare_varlen_metadata(prefix, prefix, (2**32 + 16,), (32,))
        torch.testing.assert_close(result[2].flatten(), prefix.long() * (2**32 + 16), atol=0, rtol=0)
    assert options


def test_packed_lse_missing_execution_entry_falls_back(monkeypatch):
    _require_prepared()
    from cudnn.frost import compiled_cache
    from cudnn.sdpa import packed_lse

    packed_lse._plan.cache_clear()
    calls = []
    original = packed_lse._torch_repad

    def fallback(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(compiled_cache, "positional_entry", lambda artifact: None)
    monkeypatch.setattr(packed_lse, "_torch_repad", fallback)
    prefix = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    lse = torch.arange(21, dtype=torch.float32, device="cuda").reshape(7, 3)
    graph = torch.cuda.CUDAGraph()
    try:
        actual = packed_lse.prepare_padded_lse(lse, prefix, 4)
        torch.testing.assert_close(actual, original(lse, prefix, 4), atol=0, rtol=0)
        monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("unsupported entry allocated unused prepared output"))
        with torch.cuda.graph(graph):
            captured = packed_lse.prepare_padded_lse(lse, prefix, 4)
        lse.add_(3)
        captured.fill_(-19)
        graph.replay()
        torch.testing.assert_close(captured, original(lse, prefix, 4), atol=0, rtol=0)
        assert len(calls) == 2
        assert packed_lse._plan.cache_info().misses == 1
        assert packed_lse._plan.cache_info().hits == 1
    finally:
        graph.reset()
        packed_lse._plan.cache_clear()
