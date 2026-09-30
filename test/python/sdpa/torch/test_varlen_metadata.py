# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared SDPA wrapper metadata: integer widths, streams, rebinding and routes."""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old

pytestmark = pytest.mark.L0


def _require_prepared():
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("prepared metadata requires CuTe DSL >= 4.7")
    if cutedsl_arch_requirement_error(torch.cuda.get_device_capability()):
        pytest.skip("installed DSL cannot target this GPU")


def _expected(q, kv, q_strides, kv_strides):
    values = [[b - a for a, b in zip(prefix, prefix[1:])] for prefix in (q, kv)]
    values += [[v * stride for v in q] for stride in q_strides]
    values += [[v * stride for v in kv] for stride in kv_strides]
    return values


def _check(outputs, expected):
    assert len(outputs) == len(expected)
    for i, (output, values) in enumerate(zip(outputs, expected)):
        assert output.dtype == (torch.int32 if i < 2 else torch.int64)
        assert output.shape == (len(values), 1, 1, 1)
        assert output.is_contiguous() and output.data_ptr() % 16 == 0
        assert output.flatten().tolist() == values


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_metadata_rebind_capture_wide_offsets(dtype, backward, strided, monkeypatch):
    _require_prepared()
    from cudnn.sdpa.varlen_metadata import prepare_varlen_metadata
    import cutlass.cute as cute

    q_strides = (2**32 + 16, 32, 2**30) if backward else (2**32 + 16, 32)
    kv_strides = (80, 2**31 + 8, 128, 256) if backward else (80, 2**31 + 8)
    for n in (3, 257):
        if n == 257:
            monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("changed batch count recompiled"))
        q_values = [i * 3 for i in range(n + 1)]
        kv_values = [i * 5 for i in range(n + 1)]
        owners = [torch.tensor(values, dtype=dtype, device="cuda").repeat_interleave(2 if strided else 1) for values in (q_values, kv_values)]
        q, kv = (value[::2] if strided else value for value in owners)
        outputs = prepare_varlen_metadata(q, kv, q_strides, kv_strides)
        _check(outputs, _expected(q_values, kv_values, q_strides, kv_strides))
        with monkeypatch.context() as guard:
            guard.setattr(cute, "compile", lambda *a, **k: pytest.fail("runtime metadata geometry recompiled"))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, stream=stream):
                    captured = prepare_varlen_metadata(q, kv, q_strides, kv_strides)
                q.add_(2)
                kv.add_(4)
                for value in captured:
                    value.fill_(-1)
                graph.replay()
                _check(captured, _expected([v + 2 for v in q_values], [v + 4 for v in kv_values], q_strides, kv_strides))
            finally:
                graph.reset()


@pytest.mark.parametrize("backward", [False, True])
def test_torch_ops_do_not_rebuild_prefix_casts(backward, monkeypatch):
    _require_prepared()
    from cudnn.sdpa.fwd import torch_op

    q = torch.empty((7, 4, 32), dtype=torch.bfloat16, device="cuda")
    k = torch.empty((9, 2, 32), dtype=q.dtype, device=q.device)
    v = torch.empty_like(k)
    cq = torch.tensor([0, 3, 7], dtype=torch.int32, device=q.device)
    ck = torch.tensor([0, 5, 9], dtype=torch.int32, device=q.device)
    observed = []

    class Graph:
        def execute(self, variant, workspace, *, handle):
            observed.append(variant)

    monkeypatch.setattr(torch_op, "_cached_graph", lambda *a, **k: (Graph(), 0))
    monkeypatch.setattr(torch.Tensor, "to", lambda *a, **k: pytest.fail("torch op rebuilt a prefix conversion"))
    kwargs = dict(cu_seqlens_q=cq, cu_seqlens_kv=ck, max_seqlen_q=4, max_seqlen_kv=5)
    if backward:
        lse = torch.empty((2, 4, 4, 1), dtype=torch.float32, device=q.device)
        torch_op._sdpa_bwd_impl(torch.empty_like(q), q, k, v, torch.empty_like(q), lse, 32**-0.5, **kwargs)
    else:
        torch_op._sdpa_fwd_impl(q, k, v, 32**-0.5, **kwargs)
    assert len(observed) == 1
    uid = torch_op._UIDs
    expected = {
        uid.SEQ_LEN_Q: [3, 4],
        uid.SEQ_LEN_KV: [5, 4],
        uid.RAGGED_Q: [0, 3 * 128, 7 * 128],
        uid.RAGGED_O: [0, 3 * 128, 7 * 128],
        uid.RAGGED_KV: [0, 5 * 64, 9 * 64],
        uid.RAGGED_V: [0, 5 * 64, 9 * 64],
    }
    if backward:
        expected.update({uid.RAGGED_DQ: expected[uid.RAGGED_Q], uid.RAGGED_DK: expected[uid.RAGGED_KV], uid.RAGGED_DV: expected[uid.RAGGED_V]})
    else:
        expected[uid.RAGGED_STATS] = [0, 3 * 4, 7 * 4]
    for role, values in expected.items():
        assert observed[0][int(role)].flatten().tolist() == values


def test_old_dsl_keeps_backend_metadata_path(monkeypatch):
    from cudnn.sdpa import varlen_metadata

    q = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    kv = torch.tensor([0, 5, 9], dtype=torch.int32, device="cuda")
    monkeypatch.setattr(varlen_metadata, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.6.2")))
    monkeypatch.setattr(varlen_metadata, "_plan", lambda *a, **k: pytest.fail("old DSL imported the prepared kernel"))
    values = varlen_metadata.prepare_varlen_metadata(q, kv, (2**32 + 16,), (2**31 + 8,))
    _check(values, _expected([0, 3, 7], [0, 5, 9], (2**32 + 16,), (2**31 + 8,)))


def test_metadata_artifact_reloads_without_jit(tmp_path):
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
from test_varlen_metadata import test_metadata_rebind_capture_wide_offsets
with pytest.MonkeyPatch.context() as patch:
    if reload == "1":
        patch.setattr(cute, "compile", lambda *a, **k: pytest.fail("artifact reload invoked JIT"))
    test_metadata_rebind_capture_wide_offsets(torch.int32, True, True, patch)
print(json.dumps(compiled_cache.stats()))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, str(Path(__file__).parent), cudnn.__file__, str(reload)],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert results[0]["misses"] > 0 and results[0]["hits"] == 0
    assert results[1]["misses"] == 0 and results[1]["hits"] > 0


@pytest.mark.parametrize("device", [0, 1])
def test_metadata_uses_operand_device_and_stream(device):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    from cudnn.sdpa.varlen_metadata import prepare_varlen_metadata

    original = torch.cuda.current_device()
    try:
        with torch.cuda.device(device):
            _require_prepared()
            q = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
            kv = torch.tensor([0, 5, 9], dtype=torch.int64, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                q.add_(2)
                kv.add_(4)
                torch.cuda.set_device(1 - device)
                outputs = prepare_varlen_metadata(q, kv, (2**32 + 16, 32), (2**31 + 8, 80))
                assert torch.cuda.current_device() == 1 - device
            stream.synchronize()
            _check(outputs, _expected([2, 5, 9], [4, 9, 13], (2**32 + 16, 32), (2**31 + 8, 80)))
    finally:
        torch.cuda.set_device(original)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("product", [False, True])
def test_metadata_physical_prefix_stride(product):
    _require_prepared()
    from cudnn.sdpa.varlen_metadata import prepare_varlen_metadata

    stride, n = (2**30 + 8, 4) if product else (2**32 + 16, 1)
    elements = n * stride + 1
    if torch.cuda.mem_get_info()[0] < elements * 4 + 2**30:
        pytest.skip("physical wide prefix control needs about 17 GiB free")
    try:
        owner = torch.empty(elements + 32, dtype=torch.int32, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("insufficient memory for wide prefix storage")
    q = torch.as_strided(owner[16:-16], (n + 1,), (stride,))
    owner[:16].fill_(41)
    owner[-16:].fill_(43)
    # Poison every wrapped Int32/UInt32 location within the allocation.
    for i in range(n + 1):
        owner[16 + ((i * stride) % 2**32)] = 17
    q.copy_(torch.arange(n + 1, device="cuda", dtype=torch.int32) * 3)
    kv_values = [i * 5 for i in range(n + 1)]
    kv = torch.tensor(kv_values, dtype=torch.int32, device="cuda")
    expected = _expected([i * 3 for i in range(n + 1)], kv_values, (2**32 + 16,), (32,))
    outputs = prepare_varlen_metadata(q, kv, (2**32 + 16,), (32,))
    _check(outputs, expected)
    assert (owner[:16] == 41).all() and (owner[-16:] == 43).all()


@pytest.mark.parametrize("provider", ["backend", "frost"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_shared_metadata_serves_both_forward_providers(provider, return_lse, monkeypatch):
    _require_prepared()
    import cudnn
    from cudnn.sdpa.fwd import torch_op
    from sdpa.torch.test_torch_ops import TestSdpaVarlen, TOL

    if provider == "frost" and torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("pinned THD FROST forward requires SM100/SM103; SM80 only serves the standalone THD wrapper")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    original_plans = cudnn.pygraph.create_execution_plans
    selected = []

    def create_plans(graph, *args, **kwargs):
        original_plans(graph, *args, **kwargs)
        candidates = [
            i
            for i, cfg in enumerate(graph.plans)
            if (graph._engine_for(cfg) is None if provider == "backend" else graph.get_plan_name_at_index(i).startswith("sdpa_fwd_prefill_sm100"))
        ]
        assert candidates, [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
        graph.select_plan(candidates[0])
        selected.append(graph)

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plans", create_plans)
    with torch_op._graph_cache_lock:
        torch_op._graph_cache.clear()
    torch.manual_seed(723)
    q = torch.randn((160, 4, 128), dtype=torch.bfloat16, device="cuda") * 0.2
    k = torch.randn((160, 2, 128), dtype=q.dtype, device=q.device) * 0.2
    v = torch.randn_like(k)
    owner = torch.tensor([0, 0, 64, 0, 160, 0], dtype=torch.int32, device="cuda")
    cu = owner[::2]

    def run():
        return torch.ops.cudnn.sdpa_fwd(
            q,
            k,
            v,
            128**-0.5,
            is_causal=True,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=96,
            max_seqlen_kv=96,
            return_lse=return_lse,
        )

    def check(result):
        reference, lse, *_ = TestSdpaVarlen()._ref(q, k, v, cu, True)
        assert (result[0].float() - reference).abs().max() < TOL
        if return_lse:
            assert (result[1][:, :, 0] - lse).abs().max() < TOL

    result = run()
    assert len(selected) == 1
    assert (selected[0].selected_engine is None) == (provider == "backend")
    check(result)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run()
        cu[1] = 96
        q.mul_(0.75)
        graph.replay()
        check(captured)
    finally:
        graph.reset()


def test_varlen_metadata_missing_execution_entry_falls_back(monkeypatch):
    _require_prepared()
    from cudnn.frost import compiled_cache
    from cudnn.sdpa import varlen_metadata

    varlen_metadata._plan.cache_clear()
    calls = []
    original = varlen_metadata._torch_metadata

    def fallback(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(compiled_cache, "positional_entry", lambda artifact: None)
    monkeypatch.setattr(varlen_metadata, "_torch_metadata", fallback)
    q = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    kv = torch.tensor([0, 5, 9], dtype=torch.int64, device="cuda")
    q_strides, kv_strides = (2**32 + 16, 32), (80,)
    graph = torch.cuda.CUDAGraph()
    try:
        actual = varlen_metadata.prepare_varlen_metadata(q, kv, q_strides, kv_strides)
        _check(actual, _expected([0, 3, 7], [0, 5, 9], q_strides, kv_strides))
        with torch.cuda.graph(graph):
            captured = varlen_metadata.prepare_varlen_metadata(q, kv, q_strides, kv_strides)
        q.add_(2)
        kv.add_(4)
        for output in captured:
            output.fill_(-19)
        graph.replay()
        _check(captured, _expected([2, 5, 9], [4, 9, 13], q_strides, kv_strides))
        assert len(calls) == 2
        assert varlen_metadata._plan.cache_info().misses == 1
        assert varlen_metadata._plan.cache_info().hits == 1
    finally:
        graph.reset()
        varlen_metadata._plan.cache_clear()
