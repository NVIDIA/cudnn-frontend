# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100/SM107 half conversion plans use declared, caller-owned scratch."""

import math

import pytest
import torch

from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
from frost_test_utils import requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell]


def _require_half_arch():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100/SM103/SM107 half conversion path")


def _case(d=128, dv=128, dtype=torch.bfloat16, converted=("q", "k", "v", "o"), split=1, b=2):
    _require_half_arch()
    if torch.cuda.get_device_capability() == (10, 7) and split > 1:
        pytest.skip("SM107 half split-KV remains unsupported")
    torch.manual_seed(751)
    tensors, storage = {}, {}
    for role, heads, seq, dim in (("q", 4, 17, d), ("k", 2, 128, d), ("v", 2, 128, dv), ("o", 4, 17, dv)):
        raw = (torch.randn(b, seq, heads, dim + (role in converted), device="cuda") * 0.2).to(dtype)
        if role == "o":
            raw.fill_(11)
        storage[role] = raw
        tensors[role] = raw[..., :dim].transpose(1, 2)
    tensors["lse"] = torch.empty((b, 4, 17), device="cuda")
    api = SdpaFwdDslSm100(**{"sample_" + n: t for n, t in tensors.items()}, split_kv=split)
    assert api.check_support()
    return api, tensors, storage


def _execute(api, tensors, workspace, stream=None):
    api.execute(**{n + "_tensor": t for n, t in tensors.items()}, workspace=workspace, current_stream=stream)


def _check(tensors):
    q, k, v = (tensors[n].float() for n in ("q", "k", "v"))
    scores = q @ k.repeat_interleave(2, 1).transpose(-1, -2) / math.sqrt(q.shape[-1])
    expected = scores.softmax(-1) @ v.repeat_interleave(2, 1)
    torch.testing.assert_close(tensors["o"].float(), expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(tensors["lse"], scores.logsumexp(-1), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_staged_has_no_execute_allocations_and_replays_current_storage(d, dv, dtype, monkeypatch):
    api, tensors, storage = _case(d, dv, dtype)
    required = api.scratch_workspace_bytes()
    api.compile()
    assert required > 0 and api.scratch_workspace_bytes() == required
    workspace = torch.empty(required, dtype=torch.uint8, device="cuda")
    for _ in range(2):
        _, tensors, storage = _case(d, dv, dtype)

        def run():
            with monkeypatch.context() as patch:
                patch.setattr(torch, "empty", lambda *a, **k: pytest.fail("execute allocated scratch"))
                patch.setattr(torch.Tensor, "contiguous", lambda *a, **k: pytest.fail("execute used an allocating repack"))
                _execute(api, tensors, workspace)

        run()
        _check(tensors)
        assert torch.all(storage["o"][..., -1] == 11)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                run()
            tensors["v"].mul_(0.5)
            tensors["o"].fill_(float("nan"))
            graph.replay()
            _check(tensors)
            assert torch.all(storage["o"][..., -1] == 11)
        finally:
            graph.reset()


@pytest.mark.parametrize("converted", [("q",), ("o",), ("q", "k", "v", "o")])
@pytest.mark.parametrize("split", [1, 2])
def test_half_staged_preserves_native_operands_and_split_output(converted, split):
    api, tensors, storage = _case(converted=converted, split=split)
    required = api.scratch_workspace_bytes()
    api.compile()
    staged = api._staged_spec
    expected = set(converted) - ({"o"} if split > 1 else set())
    assert ({r[0] for r in staged.regions} if staged is not None else set()) == expected
    workspace = torch.empty(required, dtype=torch.uint8, device="cuda") if required else None
    _execute(api, tensors, workspace)
    _check(tensors)
    if "o" in converted:
        assert torch.all(storage["o"][..., -1] == 11)


@pytest.mark.parametrize("layout", ["bhsd", "padded", "compact"])
@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("side_stream", [False, True])
def test_sm100_wrapper_supplies_current_workspace(layout, d, side_stream, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import sdpa_fwd_wrapper_dsl_sm100

    _, tensors, _ = _case(d, d, converted=() if layout == "compact" else ("q", "k", "v", "o"))
    if layout == "bhsd":
        tensors.update({n: tensors[n].contiguous() for n in ("q", "k", "v")})
    launch_stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    launch_stream.wait_stream(torch.cuda.current_stream())
    original = SdpaFwdDslSm100.execute
    seen = []

    def execute(api, **kwargs):
        workspace = kwargs.get("workspace")
        required = api.scratch_workspace_bytes()
        if required:
            assert workspace is not None and workspace.device == tensors["q"].device
            assert workspace.dtype == torch.uint8 and workspace.numel() == required
        else:
            assert workspace is None
        seen.append((api, workspace))
        return original(api, **kwargs)

    monkeypatch.setattr(SdpaFwdDslSm100, "execute", execute)
    with torch.cuda.stream(launch_stream):
        for factor in (1.0, 0.5):
            tensors["v"].mul_(factor)
            result = sdpa_fwd_wrapper_dsl_sm100(*(tensors[n] for n in ("q", "k", "v")))
            tensors.update(o=result["o_tensor"], lse=result["lse_tensor"])
            _check(tensors)
    torch.cuda.current_stream().wait_stream(launch_stream)
    assert len(seen) == 2 and seen[0][0] is seen[1][0]
    if seen[0][1] is not None:
        assert seen[0][1].data_ptr() != seen[1][1].data_ptr()


@pytest.mark.parametrize("staged", [False, True])
def test_half_compiled_workspace_query_uses_prepared_budget(staged, monkeypatch):
    api, _, _ = _case(converted=("q", "k", "v", "o") if staged else ())
    required = api.scratch_workspace_bytes()
    api.compile()
    for name in ("_can_prepare_dense_layout", "_can_prepare_fp8", "_can_prepare_mxfp8"):
        monkeypatch.setattr(api, name, lambda: pytest.fail("compiled workspace query repeated admission"))
    assert api.scratch_workspace_bytes() == required


@pytest.mark.parametrize("bad", ["q_dtype", "q_shape", "q_cpu", "o_overlap", "lse_dtype", "workspace_missing", "workspace_short", "workspace_alias"])
def test_half_staged_rejects_invalid_bindings_before_copy(bad, monkeypatch):
    from cudnn.sdpa.fwd import prepared_staged_forward

    api, tensors, _ = _case()
    api.compile()
    required = api.scratch_workspace_bytes()
    workspace = torch.empty(required, device="cuda", dtype=torch.uint8)
    if bad == "q_dtype":
        tensors["q"] = tensors["q"].to(torch.float16)
    elif bad == "q_shape":
        tensors["q"] = tensors["q"][:, :, :-1]
    elif bad == "q_cpu":
        tensors["q"] = tensors["q"].cpu()
    elif bad == "o_overlap":
        t = tensors["o"]
        tensors["o"] = t.as_strided(t.shape, (0, *t.stride()[1:]))
    elif bad == "lse_dtype":
        tensors["lse"] = tensors["lse"].half()
    elif bad == "workspace_missing":
        workspace = None
    elif bad == "workspace_short":
        workspace = workspace[:-1]
    elif bad == "workspace_alias":
        tensors["q"] = workspace[: tensors["q"].numel() * 2].view(torch.bfloat16).view(tensors["q"].shape)
    monkeypatch.setattr(prepared_staged_forward, "_copy", lambda *a, **k: pytest.fail("invalid binding reached staging"))
    monkeypatch.setattr(api._staged_spec.core, "fn", lambda *a, **k: pytest.fail("invalid binding reached attention"))
    with pytest.raises(ValueError):
        _execute(api, tensors, workspace)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("role,product", [("q", False), ("o", True)])
def test_half_staged_physical_wide_stride(d, role, product):
    api, tensors, _ = _case(d, d, b=4 if product else 2)
    api.compile()
    original = tensors[role]
    batch_stride = 2**31 - 65536 if product else 2**32 + 65536
    required = ((original.shape[0] - 1) * batch_stride + original[0].numel() * 2) * original.element_size()
    if torch.cuda.mem_get_info()[0] < required + 2**30:
        pytest.skip("wide physical stride storage unavailable")
    try:
        widened = torch.empty_strided(original.shape, (batch_stride, original.stride(1), original.stride(2), 1), device="cuda", dtype=original.dtype)
    except torch.OutOfMemoryError:
        pytest.skip("wide physical stride allocation unavailable")
    widened.copy_(original)
    tensors[role] = widened
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    _execute(api, tensors, workspace)
    _check(tensors)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _execute(api, tensors, workspace)
        tensors["v"].mul_(0.5)
        graph.replay()
        _check(tensors)
    finally:
        graph.reset()


@pytest.mark.parametrize("d,dv", [(128, 128), (192, 128), (256, 256), (512, 512)])
def test_half_staged_artifact_reloads_without_jit(d, dv, tmp_path):
    _require_half_arch()  # Module pytestmarks do not follow re-exported test methods.
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
from test_sdpa_staged_forward_half import _case, _execute, _check
if reload == "1":
    def forbidden(*args, **kwargs):
        raise AssertionError("reloaded staged artifact invoked JIT")
    cute.compile = forbidden
api, tensors, storage = _case(int(d), int(dv))
workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
api.compile()
owner = api._staged_spec.core.owner
if reload == "1":
    assert hasattr(owner, "_compiled_cache_raw")
    assert all(entry is None or hasattr(entry[0], "_compiled_cache_raw") for entry in api._staged_spec.copies)
_execute(api, tensors, workspace)
_check(tensors)
graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        _execute(api, tensors, workspace)
    tensors["o"].fill_(float("nan"))
    graph.replay()
    _check(tensors)
finally:
    graph.reset()
digest = [hashlib.sha256(tensors[n].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for n in ("o", "lse")]
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


@pytest.mark.parametrize("split", [1, 2])
@pytest.mark.parametrize("explicit", [False, True])
def test_half_staged_copies_follow_launch_stream(split, explicit, monkeypatch):
    from cuda.bindings import driver
    from cudnn.sdpa.fwd import prepared_staged_forward

    api, tensors, storage = _case(split=split)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    target = torch.cuda.Stream(device=api.q_desc.device)
    target.wait_stream(torch.cuda.current_stream())
    original = prepared_staged_forward._copy
    streams = []

    def copy(entry, frame, stream):
        assert stream == target.cuda_stream
        streams.append(stream)
        return original(entry, frame, stream)

    monkeypatch.setattr(prepared_staged_forward, "_copy", copy)
    raw = driver.CUstream(target.cuda_stream) if explicit else None
    # Explicit launch keeps a different torch current stream, so an accidental
    # ambient-stream copy is observable even on a single GPU.
    from contextlib import nullcontext

    context = nullcontext() if explicit else torch.cuda.stream(target)
    old_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with context:
            _execute(api, tensors, workspace, raw)
    finally:
        torch.cuda.set_sync_debug_mode(old_mode)
    torch.cuda.current_stream().wait_stream(target)
    assert len(streams) == (1 if split > 1 else 2)
    _check(tensors)
    assert torch.all(storage["o"][..., -1] == 11)
