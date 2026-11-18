# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SM80 forward bindings and wrapper caches retain caller layouts."""

from types import SimpleNamespace

import pytest
import torch

from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80
from frost_test_utils import _SM, requires_dsl
from test_sdpa_prepared_sm80 import _check
from test_sdpa_bwd_staged_sm80 import _assert_kernel_count

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


def _case(d=96, dv=96, dtype=torch.bfloat16, pad=8, features=False, batch=2):
    torch.manual_seed(391)
    tensors = {}
    for name, heads, seq, dim in (("q", 4, 17, d), ("k", 2, 33, d), ("v", 2, 33, dv), ("o", 4, 17, dv)):
        raw = torch.randn(batch, seq, heads, dim + pad, device="cuda", dtype=dtype).mul_(0.2)
        tensors[name] = raw[..., :dim].transpose(1, 2)
    tensors["stats"] = torch.empty((batch, 4, 20), device="cuda")[..., :17]
    if features:
        tensors.update(
            seq_q=torch.tensor([14, 0], device="cuda", dtype=torch.int32),
            seq_kv=torch.tensor([26, 0], device="cuda", dtype=torch.int32),
            sink=torch.randn(4, device="cuda"),
            bias=torch.randn(1, 4, 17, 33, device="cuda").mul_(0.1),
        )
    api = SdpaFwdDslSm80(
        **{"sample_" + name: tensors[name] for name in ("q", "k", "v", "o")},
        sample_lse=tensors["stats"],
        has_sink=features,
        seq_q_lens_present=features,
        seq_kv_lens_present=features,
        bias_present=features,
        bias_fp32=features,
    )
    assert api.check_support()
    return api, SimpleNamespace(bufs=tensors, features=features, causal=False, scale=d**-0.5)


def _execute(api, case, workspace, stream=None):
    tensors = case.bufs
    api.execute(
        **{name + "_tensor": tensors[name] for name in ("q", "k", "v", "o")},
        lse_tensor=tensors["stats"],
        sinks=tensors.get("sink"),
        seq_q_lens=tensors.get("seq_q"),
        seq_kv_lens=tensors.get("seq_kv"),
        bias_tensor=tensors.get("bias"),
        workspace=workspace,
        current_stream=stream,
    )


@pytest.mark.parametrize(
    "d,dv,pad,native",
    [
        (96, 96, 0, True),
        (48, 32, 8, True),
        (64, 96, 8, True),
        (192, 112, 8, True),
        (224, 208, 8, True),
        (192, 96, 1, False),
        (128, 128, 1, False),
        (63, 47, 1, False),
        (193, 193, 1, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("features", [False, True])
def test_native_replays_or_declines_copy_dependent_layout(d, dv, pad, native, dtype, features, monkeypatch):
    if not native:
        with pytest.raises(NotImplementedError, match="SM80.*(stride|multiple)"):
            _case(d, dv, dtype, pad, features)
        return
    api, case = _case(d, dv, dtype, pad, features)
    required = api.scratch_workspace_bytes()
    api.compile()
    assert api._sm80_spec.native is not None
    assert required == 0
    workspace = torch.empty(required, device="cuda", dtype=torch.uint8)
    _execute(api, case, workspace)
    _check(case)

    def run():
        with monkeypatch.context() as patch:

            def forbidden(*args, **kwargs):
                pytest.fail("execute rebuilt torch staging")

            for name in ("copy_", "contiguous", "repeat_interleave", "zero_"):
                patch.setattr(torch.Tensor, name, forbidden)
            patch.setattr(torch, "empty", forbidden)
            patch.setattr(torch, "mul", forbidden)
            _execute(api, case, workspace)

    run()
    _check(case)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(graph):
            run()
        if native:
            _assert_kernel_count(graph, 1)
        case.bufs["v"].mul_(0.5)
        case.bufs["o"].fill_(float("nan"))
        graph.replay()
        _check(case)
    finally:
        graph.reset()


def test_wrapper_cache_distinguishes_current_input_strides():
    from cudnn.sdpa.fwd.api_dsl import _sm80_wrapper_cache, sdpa_fwd_wrapper_sm80

    _sm80_wrapper_cache.clear()
    for pad in (0, 8, 16, 0):
        _, case = _case(128, 128, pad=pad)
        out = sdpa_fwd_wrapper_sm80(*(case.bufs[name] for name in ("q", "k", "v")))
        case.bufs.update(o=out["o_tensor"], stats=out["lse_tensor"])
        _check(case)
    assert len(_sm80_wrapper_cache) == 3, "each layout needs its own plan; repeating one must reuse it"


def test_wrapper_scratch_survives_explicit_stream_consumer(monkeypatch):
    from cuda.bindings import driver
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(api_dsl, "_sm80_wrapper_cache", {})
    _, case = _case(128, 128, pad=8)
    args = tuple(case.bufs[name] for name in ("q", "k", "v"))
    # Warm the real wrapper before replacing only its scratch consumer. Never
    # run attention with intentionally recycled storage in the negative control.
    warm = api_dsl.sdpa_fwd_wrapper_sm80(*args)
    case.bufs.update(o=warm["o_tensor"], stats=warm["lse_tensor"])
    _check(case)
    plan = next(iter(api_dsl._sm80_wrapper_cache.values()))
    assert plan.scratch_workspace_bytes() == 0
    # The real native warmup needs no copy scratch. Retain the wrapper's
    # independent lifetime detector by substituting only its scratch consumer.
    size = 4096
    monkeypatch.setattr(plan, "scratch_workspace_bytes", lambda: size)
    producer, target = torch.cuda.current_stream(), torch.cuda.Stream()
    done = torch.cuda.Event()
    replacements, pointers, allocation_streams, outputs = [], [], [], [warm]
    # Pre-fill the ambient allocator pool so churn never needs cudaMalloc,
    # whose synchronization would destroy the intended overlap.
    reserve = [torch.full((size,), 17, dtype=torch.uint8, device=args[0].device) for _ in range(32)]
    torch.cuda.synchronize()
    del reserve

    def consume(**kwargs):
        workspace = kwargs["workspace"]
        assert workspace.numel() == size
        assert int(kwargs["current_stream"]) == target.cuda_stream
        pointers.append(workspace.data_ptr())
        allocation_streams.append(torch.cuda.current_stream(workspace.device).cuda_stream)
        with torch.cuda.stream(target):
            torch.cuda._sleep(600000000)
        result = driver.cuMemsetD8Async(workspace.data_ptr(), 165, size, driver.CUstream(target.cuda_stream))
        assert result[0] == driver.CUresult.CUDA_SUCCESS
        done.record(target)

    monkeypatch.setattr(plan, "execute", consume)
    try:
        outputs.append(api_dsl.sdpa_fwd_wrapper_sm80(*args, current_stream=driver.CUstream(target.cuda_stream)))
        assert torch.cuda.current_stream() == producer
        assert not done.query(), "delayed workspace consumer must still be pending"
        # Keep all replacements alive through the bounded byte write. The old
        # wrapper releases ambient-stream scratch into this same allocator pool.
        for _ in range(32):
            replacements.append(torch.full((size,), 17, dtype=torch.uint8, device=args[0].device))
        producer.synchronize()
        assert not done.query(), "allocator churn must overlap the delayed consumer"
        done.synchronize()
        reused = [value for value in replacements if value.data_ptr() == pointers[0]]
        print("wrapper scratch reused on ambient stream:", len(reused))
        for value in replacements:
            torch.testing.assert_close(value, torch.full_like(value, 17), rtol=0, atol=0)
        assert allocation_streams == [target.cuda_stream]
    finally:
        # All owners remain live even if the RED control fails an assertion.
        torch.cuda.synchronize()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("d,dv,pad", [(128, 128, 0), (128, 128, 8), (96, 80, 16)])
def test_wrapper_stream_outputs_and_replay(explicit, d, dv, pad):
    from contextlib import nullcontext
    from cuda.bindings import driver
    from cudnn.sdpa.fwd.api_dsl import sdpa_fwd_wrapper_sm80

    _, case = _case(d, dv, pad=pad, features=True)
    target, ambient = torch.cuda.Stream(), torch.cuda.current_stream()
    target.wait_stream(ambient)

    def run():
        tensors = case.bufs
        with nullcontext() if explicit else torch.cuda.stream(target):
            return sdpa_fwd_wrapper_sm80(
                *(tensors[name] for name in ("q", "k", "v")),
                seq_len_q=tensors["seq_q"],
                seq_kv_lens=tensors["seq_kv"],
                bias_tensor=tensors["bias"],
                sinks=tensors["sink"],
                current_stream=driver.CUstream(target.cuda_stream) if explicit else None,
            )

    out = run()
    ambient.wait_stream(target)
    case.bufs.update(o=out["o_tensor"], stats=out["lse_tensor"])
    _check(case)
    graph = torch.cuda.CUDAGraph()
    mode = torch.cuda.get_sync_debug_mode()
    try:
        with torch.cuda.graph(graph, stream=target):
            torch.cuda.set_sync_debug_mode("error")
            try:
                captured = run()
            finally:
                torch.cuda.set_sync_debug_mode(mode)
        case.bufs["v"].mul_(0.5)
        captured["o_tensor"].fill_(float("nan"))
        graph.replay()
        case.bufs.update(o=captured["o_tensor"], stats=captured["lse_tensor"])
        _check(case)
        assert torch.cuda.current_stream() == ambient
    finally:
        torch.cuda.set_sync_debug_mode(mode)
        graph.reset()


@pytest.mark.parametrize("explicit", [False, True])
def test_native_follows_current_or_explicit_stream(explicit, monkeypatch):
    from contextlib import nullcontext
    from cuda.bindings import driver
    from cudnn.sdpa.fwd import prepared_sm80

    api, case = _case(features=True)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    target = torch.cuda.Stream()
    target.wait_stream(torch.cuda.current_stream())
    seen, original = [], prepared_sm80.execute_tensors

    def copy(entry, frame, stream, **kwargs):
        assert stream == target.cuda_stream
        seen.append(stream)
        original(entry, frame, stream, **kwargs)

    monkeypatch.setattr(prepared_sm80, "execute_tensors", copy)
    mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with nullcontext() if explicit else torch.cuda.stream(target):
            _execute(api, case, workspace, driver.CUstream(target.cuda_stream) if explicit else None)
    finally:
        torch.cuda.set_sync_debug_mode(mode)
    torch.cuda.current_stream().wait_stream(target)
    assert len(seen) == 1
    _check(case)


def test_unsupported_layout_declines_before_workspace_or_copy(monkeypatch):
    from cudnn.sdpa.fwd import prepared_sm80

    monkeypatch.setattr(prepared_sm80, "build_spec", lambda *a, **k: pytest.fail("unsupported layout reached native compilation"))
    with pytest.raises(NotImplementedError, match="q.*stride"):
        _case(pad=1)


@pytest.mark.parametrize("role", ["q", "v", "o"])
@pytest.mark.parametrize("product", [False, True])
def test_native_physical_wide_addresses(role, product):
    api, case = _case(batch=3 if product else 2)
    old = case.bufs[role]
    batch_stride = 2**31 - 65536 if product else 2**32 + 65536
    required = ((old.shape[0] - 1) * batch_stride + old[0].numel() * 2) * old.element_size()
    if torch.cuda.mem_get_info()[0] < required + 2**30:
        pytest.skip("wide physical stride storage unavailable")
    try:
        wide = torch.empty_strided(old.shape, (batch_stride, *old.stride()[1:]), dtype=old.dtype, device="cuda")
    except torch.OutOfMemoryError:
        pytest.skip("wide physical stride allocation unavailable")
    wide.copy_(old)
    case.bufs[role] = wide
    api = SdpaFwdDslSm80(**{"sample_" + n: case.bufs[n] for n in ("q", "k", "v", "o")}, sample_lse=case.bufs["stats"])
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    _execute(api, case, workspace)
    _check(case)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _execute(api, case, workspace)
        case.bufs["v"].mul_(0.6)
        case.bufs["o"].fill_(float("nan"))
        graph.replay()
        _check(case)
    finally:
        graph.reset()


@pytest.mark.parametrize("d,dv", [(48, 32), (224, 208)])
def test_native_artifacts_reload_without_jit(d, dv, tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import cudnn

    child = r"""
import hashlib, json, sys
from pathlib import Path
import torch, cudnn
import cutlass.cute as cute
from cudnn.frost import compiled_cache
folder, package, d, dv, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve(), cudnn.__file__
sys.path[:0] = [folder, str(Path(folder).parents[1])]
from test_sdpa_fwd_native_layouts_sm80 import _case, _execute, _check
if reload == "1":
    def forbidden(*args, **kwargs):
        raise AssertionError("reloaded SM80 copy chain invoked JIT")
    cute.compile = forbidden
api, case = _case(int(d), int(dv), features=True)
api.compile()
spec = api._sm80_spec
assert spec is not None
if reload == "1":
    assert hasattr(spec.artifact, "_compiled_cache_raw")
workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
_execute(api, case, workspace)
_check(case)
graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        _execute(api, case, workspace)
    case.bufs["v"].mul_(0.5)
    case.bufs["o"].fill_(float("nan"))
    graph.replay()
    _check(case)
finally:
    graph.reset()
digest = [hashlib.sha256(case.bufs[n].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for n in ("o", "stats")]
print(json.dumps(dict(digest=digest, stats=compiled_cache.stats())))
"""
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    results = []
    for reload in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", child, str(Path(__file__).parent), cudnn.__file__, str(d), str(dv), str(reload)],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    first, second = results
    assert first["stats"]["misses"] > 0 and first["stats"]["hits"] == 0, first
    assert second["stats"]["misses"] == 0 and second["stats"]["hits"] > 0, second
    assert first["digest"] == second["digest"]


@pytest.mark.parametrize("padded", [False, True])
def test_compiled_workspace_query_reuses_plan_budget(padded, monkeypatch):
    from cudnn.sdpa.fwd import prepared_sm80

    api, case = _case(96, 96) if padded else _case(128, 128, pad=0)
    expected = api.scratch_workspace_bytes()
    api.compile()

    def forbidden(*args, **kwargs):
        pytest.fail("compiled workspace query rebuilt layout metadata")

    monkeypatch.setattr(prepared_sm80, "native_layouts", forbidden)
    assert api.scratch_workspace_bytes() == expected
