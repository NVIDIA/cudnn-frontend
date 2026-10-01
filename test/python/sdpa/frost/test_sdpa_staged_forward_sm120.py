# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Existing SM120 conversion layouts use pointer hosts and caller scratch."""

import math

import pytest
import torch

from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120
from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


def _case(d=128, *, fp8=False, split=1, features=False, b=2):
    torch.manual_seed(4181)
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    h, hk, sq, sk = 4, 2, 17, 113
    tensors, storage = {}, {}
    for name, heads, seq in (("q", h, sq), ("k", hk, sk), ("v", hk, sk), ("o", h, sq)):
        ty = torch.bfloat16 if name == "o" else dtype
        raw = torch.randn(b, seq, heads, d + 1, device="cuda").mul_(0.2).to(ty)
        if name == "o":
            raw.fill_(11)
        storage[name] = raw
        tensors[name] = raw[..., :d].transpose(1, 2)
    storage["lse"] = torch.full((b, h, sq * 2), float("nan"), device="cuda")
    tensors["lse"] = storage["lse"][..., ::2]
    api = SdpaFwdDslSm120(
        **{"sample_" + n: tensors[n] for n in ("q", "k", "v", "o", "lse")},
        pertensor_fp8=fp8,
        split_kv=split,
        is_causal=features,
        has_sink=features,
        seq_q_lens_present=features,
        seq_kv_lens_present=features,
    )
    assert api.check_support()
    assert not api._can_prepare_layout()
    if features:
        tensors["sinks"] = torch.tensor([0.4, -0.1, 0.7, 0.2], device="cuda")
        tensors["seq_q_lens"] = torch.tensor([sq, sq - 3], device="cuda", dtype=torch.int32)
        tensors["seq_kv_lens"] = torch.tensor([sk, sk - 5], device="cuda", dtype=torch.int32)
    if fp8:
        for n, value in (("descale_q", 0.8), ("descale_k", 0.9), ("descale_v", 0.7), ("scale_o", 1.3), ("amax_o", 999)):
            tensors[n] = torch.full((1,), value, device="cuda", dtype=torch.float32)
    return api, tensors, storage


def _forbid_staging(monkeypatch):
    from cudnn.sdpa.fwd import prepared_staged_forward

    monkeypatch.setattr(prepared_staged_forward, "_copy", lambda *a, **k: pytest.fail("invalid operand reached a prepared staging copy"))


def _execute(api, tensors, workspace, stream=None):
    api.execute(
        **{n + "_tensor": tensors[n] for n in ("q", "k", "v", "o", "lse")},
        **{n: t for n, t in tensors.items() if n not in ("q", "k", "v", "o", "lse")},
        workspace=workspace,
        current_stream=stream,
    )


def _check(tensors, storage, features=False):
    q, k, v = (tensors[n].double() for n in ("q", "k", "v"))
    if "descale_q" in tensors:
        q = q * tensors["descale_q"].double()
        k = k * tensors["descale_k"].double()
        v = v * tensors["descale_v"].double() * tensors["scale_o"].double()
    scores = q @ k.repeat_interleave(2, 1).transpose(-1, -2) / math.sqrt(q.shape[-1])
    if features:
        qi = torch.arange(q.shape[2], device="cuda")[None, None, :, None]
        ki = torch.arange(k.shape[2], device="cuda")[None, None, None, :]
        valid = (ki <= qi) & (ki < tensors["seq_kv_lens"][:, None, None, None])
        scores.masked_fill_(~valid, float("-inf"))
        weights = torch.cat((scores, tensors["sinks"][None, :, None, None].expand(*scores.shape[:-1], 1)), -1).softmax(-1)[..., :-1]
        lse = torch.logsumexp(torch.cat((scores, tensors["sinks"][None, :, None, None].expand(*scores.shape[:-1], 1)), -1), -1)
    else:
        weights, lse = scores.softmax(-1), scores.logsumexp(-1)
    if features and "descale_q" in tensors:
        # All live causal keys fit one tile here. The existing SM120 FP8
        # kernel casts exp(logit-row_max) * 16 to E4M3 before PV, while the
        # denominator (including Sink) remains unrounded. Model that declared
        # intermediate precision instead of assigning half tolerances to FP8 P.
        row_max = scores.amax(-1, keepdim=True)
        unnormalized = (scores - row_max).exp()
        denominator = unnormalized.sum(-1, keepdim=True) + (tensors["sinks"][None, :, None, None] - row_max).exp()
        weights = (unnormalized * 16).to(torch.float8_e4m3fn).double() / (16 * denominator)
    ref = weights @ v.repeat_interleave(2, 1)
    for batch in range(q.shape[0]):
        n = int(tensors["seq_q_lens"][batch]) if features else q.shape[2]
        torch.testing.assert_close(tensors["o"][batch, :, :n].double(), ref[batch, :, :n], atol=3e-3, rtol=4e-2)
        torch.testing.assert_close(tensors["lse"][batch, :, :n].double(), lse[batch, :, :n], atol=3e-4, rtol=3e-4)
    assert torch.all(storage["o"][..., -1] == 11)
    assert torch.isnan(storage["lse"][..., 1::2]).all()


@pytest.mark.parametrize("d", [128, 256, 384])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("split,features", [(1, False), (1, True), (4, False)])
def test_staged_forward_uses_pointer_host(d, fp8, split, features, monkeypatch):
    import cutlass.cute as cute

    api, tensors, storage = _case(d, fp8=fp8, split=split, features=features)
    monkeypatch.setattr(cute.runtime, "make_fake_tensor", lambda *a, **k: pytest.fail("staged forward constructed a tensor fake"))
    monkeypatch.setattr(cute.runtime, "make_fake_compact_tensor", lambda *a, **k: pytest.fail("staged forward constructed a compact tensor fake"))
    required = api.scratch_workspace_bytes()
    backing = torch.full((required + 257,), 177, device="cuda", dtype=torch.uint8)
    workspace = backing[128 : 128 + required + 1]
    api.compile()
    assert api.scratch_workspace_bytes() == required
    assert api._staged_spec is not None and api._dense_spec is None
    _execute(api, tensors, workspace)
    _check(tensors, storage, features)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _execute(api, tensors, workspace)
        tensors["v"].copy_(tensors["v"].float().mul_(0.5).to(tensors["v"].dtype))
        tensors["o"].fill_(float("nan"))
        graph.replay()
        _check(tensors, storage, features)
    finally:
        graph.reset()
    assert torch.all(backing[:128] == 177)
    assert torch.all(backing[128 + required :] == 177)


_FAMILIES = [(128, False), (256, False), (384, False), (128, True), (384, True)]


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("workspace_dtype", [torch.float16, torch.int32])
def test_staged_typed_workspace_uses_current_view_origin(fp8, workspace_dtype):
    api, tensors, storage = _case(fp8=fp8)
    api.compile()
    required = api.scratch_workspace_bytes()
    for offset in (128, 256):
        backing = torch.full((required + offset + 128,), 177, device="cuda", dtype=torch.uint8)
        workspace = backing[offset : offset + required].view(workspace_dtype)
        tensors["v"].copy_(tensors["v"].float().mul_(0.5).to(tensors["v"].dtype))
        tensors["o"].fill_(float("nan"))
        _execute(api, tensors, workspace)
        _check(tensors, storage)
        assert torch.all(backing[:offset] == 177)
        assert torch.all(backing[offset + required :] == 177)


@pytest.mark.parametrize("d,fp8", _FAMILIES)
def test_staged_first_capture_rebinds_current_storage_and_stream(d, fp8, monkeypatch):
    from cuda.bindings import driver as cuda
    import cutlass.cute as cute

    api, initial, _ = _case(d, fp8=fp8)
    api.compile()
    owner = api._staged_spec.core.owner
    _, current, storage = _case(d, fp8=fp8)
    current["v"].copy_(current["v"].float().mul_(0.5).to(current["v"].dtype))
    assert all(current[n].data_ptr() != initial[n].data_ptr() for n in initial)
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("execute recompiled"))
    workspace = torch.empty(api.scratch_workspace_bytes() + 1, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            _execute(api, current, workspace, cuda.CUstream(stream.cuda_stream))
        graph.replay()
        _check(current, storage)
        current["v"].copy_(current["v"].float().mul_(0.5).to(current["v"].dtype))
        current["o"].fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        old_mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            _execute(api, current, workspace, cuda.CUstream(stream.cuda_stream))
        finally:
            torch.cuda.set_sync_debug_mode(old_mode)
        torch.cuda.current_stream().wait_stream(stream)
        _check(current, storage)
        graph.replay()
        _check(current, storage)
        assert api._staged_spec.core.owner is owner
    finally:
        graph.reset()


@pytest.mark.parametrize("fp8", [False, True])
def test_staged_default_stream_is_resolved_for_q_device(fp8, monkeypatch):
    from cudnn import _torch_stream

    api, tensors, storage = _case(fp8=fp8)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    target = torch.cuda.Stream(device=api.q_desc.device)
    target.wait_stream(torch.cuda.current_stream())
    original = _torch_stream._raw_current_stream
    devices = []

    def resolve(torch_module, device):
        assert device == api.q_desc.device
        devices.append(device)
        return original(torch_module, device)

    monkeypatch.setattr(_torch_stream, "_raw_current_stream", resolve)
    monkeypatch.setattr(api, "_get_default_stream", lambda *_: pytest.fail("default stream resolved from ambient device"))
    with torch.cuda.stream(target):
        _execute(api, tensors, workspace)
    torch.cuda.current_stream().wait_stream(target)
    assert devices
    _check(tensors, storage)


@pytest.mark.parametrize("fp8", [False, True])
def test_staged_default_stream_with_another_device_current(fp8, monkeypatch):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    device = torch.cuda.current_device()
    api, tensors, storage = _case(fp8=fp8)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    target = torch.cuda.Stream(device=device)
    target.wait_stream(torch.cuda.current_stream(device))
    from cudnn import _torch_stream

    original = _torch_stream._raw_current_stream
    resolved = []

    def resolve(torch_module, selected_device):
        raw = original(torch_module, selected_device)
        assert selected_device == api.q_desc.device
        assert raw == target.cuda_stream
        resolved.append(raw)
        return raw

    monkeypatch.setattr(_torch_stream, "_raw_current_stream", resolve)
    other_device = (device + 1) % torch.cuda.device_count()
    with torch.cuda.stream(target):
        with torch.cuda.device(other_device):
            _execute(api, tensors, workspace)
            assert torch.cuda.current_device() == other_device
    torch.cuda.current_stream(device).wait_stream(target)
    assert resolved
    _check(tensors, storage)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "bad",
    [
        "q_shape",
        "k_dtype",
        "o_device",
        "o_overlap",
        "lse_dtype",
        "sinks_dtype",
        "seq_q_lens_dtype",
        "seq_kv_lens_short",
        "workspace_device",
        "workspace_short",
        "workspace_stride",
        "workspace_alias",
    ],
)
def test_staged_invalid_operand_is_rejected_before_any_copy(fp8, bad, monkeypatch):
    api, tensors, _ = _case(fp8=fp8, features=True)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    if bad == "q_shape":
        tensors["q"] = tensors["q"][..., :-1]
    elif bad == "k_dtype":
        tensors["k"] = tensors["k"].float()
    elif bad == "o_device":
        tensors["o"] = torch.empty(tensors["o"].shape, dtype=tensors["o"].dtype)
    elif bad == "o_overlap":
        t = tensors["o"]
        tensors["o"] = t.as_strided(t.shape, (0, *t.stride()[1:]))
    elif bad in ("lse_dtype", "sinks_dtype", "seq_q_lens_dtype"):
        role = bad.removesuffix("_dtype")
        tensors[role] = tensors[role].double()
    elif bad == "seq_kv_lens_short":
        tensors["seq_kv_lens"] = tensors["seq_kv_lens"][:1]
    elif bad == "workspace_device":
        workspace = torch.empty(workspace.shape, dtype=torch.uint8)
    elif bad == "workspace_short":
        workspace = workspace[:-1]
    elif bad == "workspace_stride":
        workspace = torch.empty(workspace.numel() * 2, device="cuda", dtype=torch.uint8)[::2]
    else:
        tensors["sinks"] = workspace[:16].view(torch.float32)
    _forbid_staging(monkeypatch)
    monkeypatch.setattr(torch.Tensor, "copy_", lambda *a, **k: pytest.fail("invalid operand reached a staging copy"))
    monkeypatch.setattr(api._staged_spec.core, "fn", lambda *a, **k: pytest.fail("invalid operand reached attention"))
    with pytest.raises(ValueError):
        _execute(api, tensors, workspace)


@pytest.mark.parametrize("role", ["q", "o"])
def test_staged_amax_alias_checks_original_operand(role, monkeypatch):
    api, tensors, _ = _case(fp8=True)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    # Reinterpret a small contiguous island without touching device contents.
    tensors["amax_o"] = tensors[role][0, 0, 0, :4].view(torch.float32)[:1]
    _forbid_staging(monkeypatch)
    monkeypatch.setattr(torch.Tensor, "copy_", lambda *a, **k: pytest.fail("aliased Amax reached staging"))
    with pytest.raises(ValueError, match="amax_o overlaps"):
        _execute(api, tensors, workspace)


@pytest.mark.parametrize("block", [16, 32])
@pytest.mark.parametrize("role", ["q", "o"])
def test_staged_sf_alias_checks_original_operand(block, role, monkeypatch):
    from test_sdpa_prepared_block_output import _fp8_case

    g, vp, ws, _, sf, _, ts = _fp8_case(block, stats=True, staged=True)
    source = vp[ts[role]].view(torch.uint8)
    vp[ts["sf_o"]] = source.as_strided((sf.numel(),), (1,)).view(sf.shape)
    _forbid_staging(monkeypatch)
    monkeypatch.setattr(torch.Tensor, "copy_", lambda *a, **k: pytest.fail("aliased SF output reached staging"))
    with pytest.raises(ValueError, match="sf_o overlaps"):
        g.execute(vp, ws)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d,fp8", _FAMILIES)
@pytest.mark.parametrize("role,product", [("q", False), ("o", True)])
def test_staged_physical_wide_batch_stride(d, fp8, role, product):
    api, tensors, storage = _case(d, fp8=fp8, b=4 if product else 2)
    api.compile()
    original = tensors[role]
    # The last batch's offset exceeds 2**32 even though a product-case
    # batch stride fits int32. Both cases perform real addressed reads/writes.
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
    _check(tensors, storage)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _execute(api, tensors, workspace)
        tensors["v"].copy_(tensors["v"].float().mul_(0.5).to(tensors["v"].dtype))
        tensors["o"].fill_(float("nan"))
        graph.replay()
        _check(tensors, storage)
    finally:
        graph.reset()


@pytest.mark.parametrize("d,fp8", _FAMILIES)
def test_staged_artifact_reloads_without_jit(d, fp8, tmp_path):
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
folder, package, d, fp8, reload = sys.argv[1:]
assert Path(cudnn.__file__).resolve() == Path(package).resolve(), cudnn.__file__
sys.path[:0] = [folder, str(Path(folder).parents[1])]
from test_sdpa_staged_forward_sm120 import _case, _execute, _check
if reload == "1":
    def forbidden(*args, **kwargs):
        raise AssertionError("reloaded staged artifact invoked JIT")
    cute.compile = forbidden
api, tensors, storage = _case(int(d), fp8=bool(int(fp8)))
workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
api.compile()
owner = api._staged_spec.core.owner
if reload == "1":
    assert hasattr(owner, "_compiled_cache_raw")
    assert all(entry is None or hasattr(entry[0], "_compiled_cache_raw") for entry in api._staged_spec.copies)
_execute(api, tensors, workspace)
_check(tensors, storage)
graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        _execute(api, tensors, workspace)
    tensors["o"].fill_(float("nan"))
    graph.replay()
    _check(tensors, storage)
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
            [sys.executable, "-c", child, str(Path(__file__).parent), cudnn.__file__, str(d), str(int(fp8)), str(reload)],
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


@pytest.mark.parametrize("block", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_staged_block_output_matches_native_graph(block, dtype):
    from test_sdpa_prepared_block_output import _fp8_case

    reference = _fp8_case(block, dtype=dtype, stats=True)
    staged = _fp8_case(block, dtype=dtype, stats=True, staged=True)
    ref_g, ref_vp, ref_ws, ref_o, ref_sf, ref_amax, ref_ts = reference
    g, vp, ws, output, sf, amax, ts = staged
    assert g.get_workspace_size() > ref_g.get_workspace_size()

    def check():
        torch.testing.assert_close(output, ref_o, rtol=0, atol=0)
        torch.testing.assert_close(sf, ref_sf, rtol=0, atol=0)
        torch.testing.assert_close(amax, ref_amax, rtol=0, atol=0)
        torch.testing.assert_close(vp[ts["lse"]], ref_vp[ref_ts["lse"]], rtol=0, atol=0)
        for role in ("q", "k", "v", "o"):
            t = vp[ts[role]]
            padding = t.as_strided((*t.shape[:-1], 1), t.stride(), storage_offset=t.storage_offset() + t.shape[-1])
            assert torch.all(padding == 12)

    ref_g.execute(ref_vp, ref_ws)
    g.execute(vp, ws)
    check()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            g.execute(vp, ws)
        for pack, tensors in ((vp, ts), (ref_vp, ref_ts)):
            pack[tensors["descale_v"]].fill_(0.3)
        output.fill_(0xAA)
        sf.fill_(0xAA)
        amax.fill_(float("nan"))
        ref_g.execute(ref_vp, ref_ws)
        graph.replay()
        check()
    finally:
        graph.reset()


@pytest.mark.parametrize("layout", ["bhsd", "padded", "compact"])
@pytest.mark.parametrize("d", [128, 384])
@pytest.mark.parametrize("side_stream", [False, True])
def test_sm120_wrapper_supplies_conversion_workspace(layout, d, side_stream, monkeypatch):
    from cudnn.sdpa.fwd.api_dsl import sdpa_fwd_wrapper_dsl_sm120

    _, tensors, _ = _case(d)
    if layout == "bhsd":
        tensors.update({n: tensors[n].contiguous() for n in ("q", "k", "v")})
    elif layout == "compact":
        tensors.update({n: tensors[n].transpose(1, 2).contiguous().transpose(1, 2) for n in ("q", "k", "v")})
    launch_stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    launch_stream.wait_stream(torch.cuda.current_stream())
    original_execute = SdpaFwdDslSm120.execute
    seen = []

    def execute(api, **kwargs):
        workspace = kwargs.get("workspace")
        required = api.scratch_workspace_bytes()
        if layout == "compact":
            assert required == 0 and workspace is None
        else:
            assert required > 0 and workspace is not None
            assert workspace.device == tensors["q"].device
            assert workspace.dtype == torch.uint8 and workspace.numel() == required
        seen.append(api)
        return original_execute(api, **kwargs)

    monkeypatch.setattr(SdpaFwdDslSm120, "execute", execute)
    with torch.cuda.stream(launch_stream):
        # The cached plan must also supply fresh caller-owned scratch on reuse.
        for factor in (1.0, 0.5):
            tensors["v"].mul_(factor)
            result = sdpa_fwd_wrapper_dsl_sm120(*(tensors[n] for n in ("q", "k", "v")))
            q, k, v = (tensors[n].double() for n in ("q", "k", "v"))
            scores = q @ k.repeat_interleave(2, 1).transpose(-1, -2) / math.sqrt(d)
            ref = scores.softmax(-1) @ v.repeat_interleave(2, 1)
            torch.testing.assert_close(result["o_tensor"].double(), ref, atol=3e-3, rtol=4e-2)
            torch.testing.assert_close(result["lse_tensor"].double(), scores.logsumexp(-1), atol=3e-4, rtol=3e-4)
    torch.cuda.current_stream().wait_stream(launch_stream)
    assert len(seen) == 2 and seen[0] is seen[1]


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("staged", [False, True])
def test_compiled_workspace_query_uses_prepared_budget(fp8, staged, monkeypatch):
    api, tensors, _ = _case(fp8=fp8)
    if not staged:
        for name in ("q", "k", "v", "o"):
            tensors[name] = tensors[name].transpose(1, 2).contiguous().transpose(1, 2)
        api = SdpaFwdDslSm120(**{"sample_" + n: tensors[n] for n in ("q", "k", "v", "o", "lse")}, pertensor_fp8=fp8)
        assert api.check_support()
    required = api.scratch_workspace_bytes()
    api.compile()
    monkeypatch.setattr(api, "_can_prepare_layout", lambda: pytest.fail("compiled workspace query rebuilt layout"))
    monkeypatch.setattr(api, "_can_prepare_fp8", lambda: pytest.fail("compiled workspace query repeated admission"))
    assert api.scratch_workspace_bytes() == required


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("broadcast_q", [False, True])
def test_staged_copy_preserves_runtime_element_stride_and_broadcast(fp8, broadcast_q):
    api, tensors, storage = _case(fp8=fp8)
    api.compile()
    b, h, sq, d = tensors["o"].shape
    output_storage = torch.full((b, sq, h, d * 2), 11, device="cuda", dtype=tensors["o"].dtype)
    tensors["o"] = output_storage[..., ::2].transpose(1, 2)
    if broadcast_q:
        tensors["q"] = tensors["q"][:, :, :1].expand_as(tensors["q"])
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    _execute(api, tensors, workspace)
    _check(tensors, storage)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _execute(api, tensors, workspace)
        tensors["v"].copy_(tensors["v"].float().mul_(0.5).to(tensors["v"].dtype))
        graph.replay()
        _check(tensors, storage)
        assert torch.all(output_storage[..., 1::2] == 11)
    finally:
        graph.reset()


@pytest.mark.parametrize("fp8", [False, True])
def test_staged_explicit_stream_with_another_device_current(fp8, monkeypatch):
    from cuda.bindings import driver
    from cudnn.sdpa.fwd import prepared_staged_forward

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    device = torch.cuda.current_device()
    other_device = (device + 1) % torch.cuda.device_count()
    api, tensors, storage = _case(fp8=fp8)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    target = torch.cuda.Stream(device=device)
    target.wait_stream(torch.cuda.current_stream(device))
    original = prepared_staged_forward._copy
    streams = []

    def copy(entry, frame, stream):
        assert stream == target.cuda_stream
        streams.append(stream)
        return original(entry, frame, stream)

    monkeypatch.setattr(prepared_staged_forward, "_copy", copy)
    with torch.cuda.device(other_device):
        _execute(api, tensors, workspace, driver.CUstream(target.cuda_stream))
        assert torch.cuda.current_device() == other_device
    torch.cuda.current_stream(device).wait_stream(target)
    assert len(streams) == 2
    _check(tensors, storage)
