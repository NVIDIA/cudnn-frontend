# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared forward metadata must match the graph's compact declarations."""

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import torch_op
from sdpa.torch.test_torch_ops import ref_attention, TOL

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("provider", ["backend", "frost"])
@pytest.mark.parametrize("role", ["lengths", "sinks"])
@pytest.mark.parametrize("strided", [False, True])
def test_forward_auxiliary_storage_matches_graph(provider, role, strided, monkeypatch):
    if provider == "frost":
        _require_prepared()
        if torch.cuda.get_device_capability() not in ((8, 0), (10, 0), (10, 3), (10, 7), (12, 0)):
            pytest.skip("pinned forward metadata route requires an available dense FROST engine")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    original = cudnn.pygraph.create_execution_plans
    selected = []

    def create_plans(graph, *args, **kwargs):
        original(graph, *args, **kwargs)
        candidates = [i for i, cfg in enumerate(graph.plans) if (graph._engine_for(cfg) is None) == (provider == "backend")]
        assert candidates, [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
        graph.select_plan(candidates[0])
        selected.append(graph)

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plans", create_plans)
    with torch_op._graph_cache_lock:
        torch_op._graph_cache.clear()
    torch.manual_seed(723)
    q, k, v = [torch.randn((2, 128, 4, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2) * 0.2 for _ in range(3)]
    lq, lkv = (96, 64), (64, 128)
    seq_q = torch.tensor(lq, dtype=torch.int32, device="cuda")
    seq_kv = torch.tensor(lkv, dtype=torch.int32, device="cuda")
    sinks = torch.tensor([-0.7, 0.4, 1.1, -0.2], device="cuda")
    if strided:
        if role == "lengths":
            seq_q = torch.tensor([96, 32, 64, 32], dtype=torch.int32, device="cuda")[::2]
            seq_kv = torch.tensor([64, 32, 128, 32], dtype=torch.int32, device="cuda")[::2]
        else:
            sinks = torch.tensor([-0.7, 10.0, 0.4, 10.0, 1.1, 10.0, -0.2, 10.0], device="cuda")[::2]
    kwargs = dict(seq_len_q=seq_q, seq_len_kv=seq_kv) if role == "lengths" else dict(sinks=sinks)

    def run():
        return torch.ops.cudnn.sdpa_fwd(q, k, v, 128**-0.5, return_lse=True, **kwargs)

    output, stats = run()
    assert len(selected) == 1
    assert (selected[0].selected_engine is None) == (provider == "backend")

    def check(output, stats):
        for batch in range(2):
            nq, nk = (lq[batch], lkv[batch]) if role == "lengths" else (128, 128)
            reference, reference_stats = ref_attention(
                q[batch : batch + 1, :, :nq],
                k[batch : batch + 1, :, :nk],
                v[batch : batch + 1, :, :nk],
                128**-0.5,
                sinks=sinks if role == "sinks" else None,
                return_lse=True,
            )
            assert (output[batch, :, :nq].float() - reference[0]).abs().max() < TOL
            assert (stats[batch, :, :nq, 0] - reference_stats[0, :, :, 0]).abs().max() < TOL

    check(output, stats)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run()
        lq, lkv = (64, 96), (128, 64)
        seq_q.copy_(torch.tensor(lq, device="cuda"))
        seq_kv.copy_(torch.tensor(lkv, device="cuda"))
        sinks.add_(0.5)
        q.mul_(0.75)
        graph.replay()
        check(*captured)
    finally:
        graph.reset()


def _require_prepared():
    from sdpa.torch.test_varlen_metadata import _require_prepared as require

    require()


def _check_columns(outputs, inputs, batch, heads):
    for output, tensor, dtype, shape in zip(outputs, inputs, (torch.int32, torch.int32, torch.float32), ((batch, 1, 1, 1), (batch, 1, 1, 1), (1, heads, 1, 1))):
        if tensor is None:
            assert output is None
        else:
            assert output.shape == shape and output.dtype == dtype
            assert output.is_contiguous() and output.data_ptr() % 16 == 0
            torch.testing.assert_close(output.flatten(), tensor.flatten().to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset", [0, 1])
def test_forward_metadata_compact_sink_needs_no_compiler(dtype, offset, monkeypatch):
    from cudnn.sdpa import forward_metadata as metadata

    monkeypatch.setattr(metadata, "cutedsl_state", lambda: pytest.fail("single compact sink queried compiler eligibility"))
    values = (None, None, torch.arange(8 + offset, device="cuda", dtype=dtype)[offset:])
    output = metadata.prepare_forward_metadata(*values, 2, 8)
    _check_columns(output, values, 2, 8)
    assert output[2].data_ptr() != values[2].data_ptr()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = metadata.prepare_forward_metadata(*values, 2, 8)
        values[2].add_(0.5)
        captured[2].fill_(float("nan"))
        graph.replay()
        _check_columns(captured, values, 2, 8)
    finally:
        graph.reset()


def test_forward_metadata_native_views_need_no_compiler(monkeypatch):
    from cudnn.sdpa import forward_metadata as metadata

    values = (
        torch.arange(3, device="cuda", dtype=torch.int32),
        torch.arange(3, device="cuda", dtype=torch.int32),
        torch.arange(5, device="cuda", dtype=torch.float32),
    )
    monkeypatch.setattr(metadata, "cutedsl_state", lambda: pytest.fail("native views queried the compiler"))
    result = metadata.prepare_forward_metadata(*values, 3, 5)
    _check_columns(result, values, 3, 5)
    assert all(a.data_ptr() == b.data_ptr() for a, b in zip(result, values))


@pytest.mark.parametrize("sink_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["strided", "broadcast", "offset"])
def test_forward_metadata_rebind_geometry_and_replay(sink_dtype, layout, monkeypatch):
    _require_prepared()
    from cudnn.sdpa.forward_metadata import prepare_forward_metadata
    import cutlass.cute as cute

    for batch, heads in ((3, 5), (259, 257)):
        if batch == 259:
            monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("runtime geometry recompiled"))
        values = []
        for count, dtype in ((batch, torch.int64), (batch, torch.int64), (heads, sink_dtype)):
            owner = torch.arange(2 * count + 1, device="cuda").to(dtype)
            value = owner[1 : 2 * count + 1 : 2] if layout == "strided" else owner[1 : count + 1]
            if layout == "broadcast":
                value = owner[1:2].expand(count)
            values.append(value)
        result = prepare_forward_metadata(*values, batch, heads)
        _check_columns(result, values, batch, heads)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.graph(graph, stream=stream):
                captured = prepare_forward_metadata(*values, batch, heads)
            for value in values:
                value[:1].add_(2) if layout == "broadcast" else value.add_(2)
            for output in captured:
                output.fill_(-19)
            graph.replay()
            _check_columns(captured, values, batch, heads)
        finally:
            graph.reset()


@pytest.mark.parametrize("mask", [1, 2, 4, 3, 5, 6, 7])
def test_forward_metadata_optional_roles(mask):
    _require_prepared()
    from cudnn.sdpa.forward_metadata import prepare_forward_metadata

    values = tuple(
        torch.arange(7, device="cuda", dtype=dtype)[::2] if mask & (1 << role) else None
        for role, dtype in enumerate((torch.int64, torch.int32, torch.bfloat16))
    )
    _check_columns(prepare_forward_metadata(*values, 4, 4), values, 4, 4)


@pytest.mark.parametrize("old_dsl", [False, True])
def test_forward_metadata_fallback_compacts_and_aligns(old_dsl, monkeypatch):
    from cudnn.sdpa import forward_metadata as metadata

    if old_dsl:
        monkeypatch.setattr(metadata, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.6.2")))
        dtypes = (torch.int32, torch.int64, torch.float32)
    else:
        dtypes = (torch.float32, torch.int16, torch.float64)
    values = tuple(torch.arange(9, device="cuda", dtype=dtype)[1:9:2] for dtype in dtypes)
    monkeypatch.setattr(metadata, "_plan", lambda *a, **k: pytest.fail("unsupported conversion compiled"))
    _check_columns(metadata.prepare_forward_metadata(*values, 4, 4), values, 4, 4)


def test_forward_metadata_empty_and_wrapping_casts():
    _require_prepared()
    from cudnn.sdpa.forward_metadata import prepare_forward_metadata

    values = (torch.empty(0, dtype=torch.int64, device="cuda"), None, torch.empty(0, dtype=torch.float16, device="cuda"))
    _check_columns(prepare_forward_metadata(*values, 0, 0), values, 0, 0)
    values = (torch.tensor([2**32 + 17, -(2**32) - 23], device="cuda"), None, None)
    _check_columns(prepare_forward_metadata(*values, 2, 0), values, 2, 0)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("role", [0, 1, 2])
@pytest.mark.parametrize("product", [False, True])
def test_forward_metadata_physical_stride(role, product):
    _require_prepared()
    import gc
    from cudnn.sdpa.forward_metadata import prepare_forward_metadata

    stride, count = (2**30 + 8, 5) if product else (2**32 + 16, 2)
    elements = (count - 1) * stride + 1
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < (elements + 32) * 4 + 2**30:
        pytest.skip("wide metadata control needs about 17 GiB free")
    try:
        owner = torch.empty(elements + 32, dtype=torch.float32 if role == 2 else torch.int32, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("insufficient memory for physical metadata storage")
    source = owner[16:-16].as_strided((count,), (stride,))
    owner[:16].fill_(41)
    owner[-16:].fill_(43)
    for i in range(count):
        owner[16 + (i * stride) % 2**32] = 17
    values = [None, None, None]
    values[role] = source
    source.copy_(torch.arange(count, device="cuda") * 3)
    _check_columns(prepare_forward_metadata(*values, count, count), values, count, count)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = prepare_forward_metadata(*values, count, count)
        source.add_(2)
        captured[role].fill_(-19)
        graph.replay()
        _check_columns(captured, values, count, count)
        assert (owner[:16] == 41).all() and (owner[-16:] == 43).all()
    finally:
        graph.reset()


def test_forward_metadata_explicit_operand_target(monkeypatch):
    _require_prepared()
    from cudnn.sdpa import forward_metadata as metadata
    from cudnn.sdpa.fwd.kernels import forward_metadata as kernel

    metadata._plan.cache_clear()
    kernel.compile_metadata.cache_clear()
    original = kernel.compile_cached
    observed = []

    def compile_cached(*args, **kwargs):
        observed.append(kwargs["options"])
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "compile_cached", compile_cached)
    values = (torch.arange(3, dtype=torch.int64, device="cuda"), None, None)
    _check_columns(metadata.prepare_forward_metadata(*values, 3, 0), values, 3, 0)
    major, minor = torch.cuda.get_device_capability(values[0].device)
    assert observed == [f"--enable-tvm-ffi --gpu-arch sm_{major}{minor}"]


@pytest.mark.parametrize("device", [0, 1])
def test_forward_metadata_operand_device_and_stream(device):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    from cudnn.sdpa.forward_metadata import prepare_forward_metadata

    original = torch.cuda.current_device()
    try:
        with torch.cuda.device(device):
            _require_prepared()
            values = tuple(torch.arange(7, device="cuda", dtype=dtype)[::2] for dtype in (torch.int64, torch.int32, torch.bfloat16))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for value in values:
                    value.add_(2)
                torch.cuda.set_device(1 - device)
                output = prepare_forward_metadata(*values, 4, 4)
                assert torch.cuda.current_device() == 1 - device
            stream.synchronize()
            _check_columns(output, values, 4, 4)
    finally:
        torch.cuda.set_device(original)


def test_forward_auxiliary_metadata_opcheck():
    torch.manual_seed(723)
    q, k, v = [torch.randn((2, 128, 4, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2) * 0.2 for _ in range(3)]
    lengths = torch.tensor([64, 32, 96, 32], dtype=torch.int64, device="cuda")[::2]
    sinks = torch.tensor([-0.7, 10.0, 0.4, 10.0, 1.1, 10.0, -0.2, 10.0], dtype=torch.bfloat16, device="cuda")[::2]
    torch.library.opcheck(torch.ops.cudnn.sdpa_fwd, (q, k, v, 128**-0.5), dict(seq_len_q=lengths, seq_len_kv=lengths, sinks=sinks, return_lse=True))


def test_forward_metadata_artifact_reload(tmp_path):
    _require_prepared()
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    child = r"""
import json, sys
from pathlib import Path
import torch, cudnn, pytest
import cutlass.cute as cute
from cudnn.frost import compiled_cache
folder, package, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve()
sys.path[:0] = [folder, str(Path(folder).parents[1])]
from test_aux_metadata import test_forward_metadata_rebind_geometry_and_replay
with pytest.MonkeyPatch.context() as patch:
    if reload == "1":
        patch.setattr(cute, "compile", lambda *a, **k: pytest.fail("artifact reload invoked JIT"))
    test_forward_metadata_rebind_geometry_and_replay(torch.bfloat16, "strided", patch)
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, str(Path(__file__).resolve().parent), cudnn.__file__, str(reload)],
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


def test_forward_metadata_missing_execution_entry_falls_back(monkeypatch):
    _require_prepared()
    from cudnn.frost import compiled_cache
    from cudnn.sdpa import forward_metadata as metadata

    metadata._plan.cache_clear()
    calls = []
    original = metadata._torch_column

    def fallback(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(compiled_cache, "positional_entry", lambda artifact: None)
    monkeypatch.setattr(metadata, "_torch_column", fallback)
    values = tuple(torch.arange(7, device="cuda", dtype=dtype)[::2] for dtype in (torch.int64, torch.int32, torch.bfloat16))
    graph = torch.cuda.CUDAGraph()
    try:
        _check_columns(metadata.prepare_forward_metadata(*values, 4, 4), values, 4, 4)
        monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("unsupported entry allocated unused prepared output"))
        with torch.cuda.graph(graph):
            captured = metadata.prepare_forward_metadata(*values, 4, 4)
        for value in values:
            value.add_(2)
        for output in captured:
            output.fill_(-19)
        graph.replay()
        _check_columns(captured, values, 4, 4)
        assert len(calls) == 6
        assert metadata._plan.cache_info().misses == 1
        assert metadata._plan.cache_info().hits == 1
    finally:
        graph.reset()
        metadata._plan.cache_clear()
