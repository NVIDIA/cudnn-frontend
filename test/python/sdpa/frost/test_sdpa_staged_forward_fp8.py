# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense FP8 conversion plans bind current storage around the prepared host."""

import math

import pytest
import torch

from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
from frost_test_utils import requires_dsl, requires_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell]

_FLAVORS = [(128, 128), (192, 128), (256, 256), (512, 512)]


def _case(d=128, dv=128, *, split=1, dtype=torch.float8_e4m3fn, output_dtype=torch.bfloat16, b=2, strided_stats=False, features=False):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100/SM103/SM107 FP8 conversion path")
    torch.manual_seed(417)
    h, hk, sq, sk = 4, 2, 17, 128
    tensors, storage = {}, {}
    for role, heads, seq, dim in (("q", h, sq, d), ("k", hk, sk, d), ("v", hk, sk, dv), ("o", h, sq, dv)):
        ty = output_dtype if role == "o" else dtype
        raw = (torch.randn(b, seq, heads, dim + 1, device="cuda") * 0.2).to(ty)
        if role == "o":
            raw.fill_(11)
        storage[role] = raw
        tensors[role] = raw[..., :dim].transpose(1, 2)
    step = 2 if strided_stats else 1
    storage["lse"] = torch.full((b * h * sq * step + 64,), float("nan"), device="cuda")
    tensors["lse"] = storage["lse"][32:-32:step].view(b, h, sq)
    api = SdpaFwdDslSm100(
        **{"sample_" + n: tensors[n] for n in ("q", "k", "v", "o", "lse")},
        pertensor_fp8=True,
        split_kv=split,
        has_sink=features,
        seq_q_lens_present=features,
        seq_kv_lens_present=features,
    )
    assert api.check_support()
    assert not api._can_prepare_fp8()
    for role, value in (("descale_q", 0.8), ("descale_k", 0.9), ("descale_v", 0.7), ("scale_o", 1.3), ("amax_o", 999)):
        tensors[role] = torch.full((1,), value, device="cuda", dtype=torch.float32)
    if features:
        tensors["sinks"] = torch.zeros(h, device="cuda")
        tensors["seq_q_lens"] = torch.full((b,), sq, dtype=torch.int32, device="cuda")
        tensors["seq_kv_lens"] = torch.full((b,), sk, dtype=torch.int32, device="cuda")
    return api, tensors, storage


def _execute(api, tensors, workspace, stream=None):
    api.execute(
        **{n + "_tensor": tensors[n] for n in ("q", "k", "v", "o", "lse")},
        **{n: t for n, t in tensors.items() if n not in ("q", "k", "v", "o", "lse")},
        workspace=workspace,
        current_stream=stream,
    )


def _check(tensors, storage):
    q, k, v = (tensors[n].double() * tensors["descale_" + n].double() for n in ("q", "k", "v"))
    scores = q @ k.repeat_interleave(2, 1).transpose(-1, -2) / math.sqrt(q.shape[-1])
    ref = scores.softmax(-1) @ v.repeat_interleave(2, 1)
    scaled = ref * tensors["scale_o"].double()
    torch.testing.assert_close(tensors["o"].double(), scaled, atol=3e-3, rtol=4e-2)
    # Match the established FP8 Stats budget. D256's approximate exponentials
    # differ from this reference by the same amount in the old tensor entry;
    # the staged and legacy O/Stats/Amax are bitwise equal in the baseline probe.
    torch.testing.assert_close(tensors["lse"].double(), scores.logsumexp(-1), atol=0.01, rtol=0.01)
    torch.testing.assert_close(tensors["amax_o"].double().reshape(()), ref.abs().amax(), atol=3e-3, rtol=4e-2)
    assert torch.all(storage["o"][..., -1] == 11)
    assert torch.isnan(storage["lse"][:32]).all()
    assert torch.isnan(storage["lse"][-32:]).all()


@pytest.mark.parametrize("d,dv", _FLAVORS)
def test_fp8_staged_uses_pointer_host_and_current_storage(d, dv, monkeypatch):
    import cutlass.cute as cute

    api, tensors, storage = _case(d, dv)
    monkeypatch.setattr(cute.runtime, "make_fake_tensor", lambda *a, **k: pytest.fail("conversion constructed a tensor fake"))
    monkeypatch.setattr(cute.runtime, "make_fake_compact_tensor", lambda *a, **k: pytest.fail("conversion constructed a compact tensor fake"))
    required = api.scratch_workspace_bytes()
    api.compile()
    assert api._staged_spec is not None
    assert api.scratch_workspace_bytes() == required
    for offset in (128, 256):
        _, tensors, storage = _case(d, dv)
        backing = torch.full((required + offset + 129,), 177, device="cuda", dtype=torch.uint8)
        workspace = backing[offset : offset + required + 1]
        _execute(api, tensors, workspace)
        _check(tensors, storage)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                _execute(api, tensors, workspace)
            tensors["v"].copy_(tensors["v"].float().mul_(0.5).to(tensors["v"].dtype))
            tensors["descale_q"].fill_(1.1)
            tensors["o"].fill_(float("nan"))
            graph.replay()
            _check(tensors, storage)
        finally:
            graph.reset()
        assert torch.all(backing[:offset] == 177)
        assert torch.all(backing[offset + required :] == 177)


@pytest.mark.parametrize("d", [128, 512])
def test_staged_default_stream_is_resolved_for_q_device(d, monkeypatch):
    from cudnn import _torch_stream

    api, tensors, storage = _case(d, d)
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


@pytest.mark.parametrize("d", [128, 512])
def test_staged_default_stream_with_another_device_current(d, monkeypatch):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    device = torch.cuda.current_device()
    api, tensors, storage = _case(d, d)
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


@pytest.mark.parametrize("d,dv", _FLAVORS)
def test_fp8_staged_strided_stats(d, dv):
    api, tensors, storage = _case(d, dv, strided_stats=True)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    _execute(api, tensors, workspace)
    _check(tensors, storage)
    assert torch.isnan(storage["lse"][33:-32:2]).all()


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d,dv", _FLAVORS)
@pytest.mark.parametrize("role,product", [("q", False), ("o", True)])
def test_staged_physical_wide_batch_stride(d, dv, role, product):
    api, tensors, storage = _case(d, dv, b=4 if product else 2)
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


@pytest.mark.parametrize("d,dv", _FLAVORS)
def test_staged_artifact_reloads_without_jit(d, dv, tmp_path):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100/SM103/SM107 FP8 conversion path")
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
from test_sdpa_staged_forward_fp8 import _case, _execute, _check
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


@pytest.mark.parametrize(
    "bad",
    [
        "q_shape",
        "k_dtype",
        "o_device",
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
def test_staged_invalid_operand_is_rejected_before_any_copy(bad, monkeypatch):
    api, tensors, _ = _case(features=True)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    if bad == "q_shape":
        tensors["q"] = tensors["q"][..., :-1]
    elif bad == "k_dtype":
        tensors["k"] = tensors["k"].float()
    elif bad == "o_device":
        tensors["o"] = torch.empty(tensors["o"].shape, dtype=tensors["o"].dtype)
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
    from cudnn.sdpa.fwd import prepared_staged_forward

    monkeypatch.setattr(prepared_staged_forward, "_copy", lambda *a, **k: pytest.fail("invalid operand reached a prepared staging copy"))
    monkeypatch.setattr(torch.Tensor, "copy_", lambda *a, **k: pytest.fail("invalid operand reached a staging copy"))
    monkeypatch.setattr(api._staged_spec.core, "fn", lambda *a, **k: pytest.fail("invalid operand reached attention"))
    with pytest.raises(ValueError):
        _execute(api, tensors, workspace)


@pytest.mark.parametrize("role", ["q", "o"])
def test_staged_amax_alias_checks_original_operand(role, monkeypatch):
    api, tensors, _ = _case()
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    # Reinterpret a small contiguous island without touching device contents.
    tensors["amax_o"] = tensors[role][0, 0, 0, :4].view(torch.float32)[:1]
    from cudnn.sdpa.fwd import prepared_staged_forward

    monkeypatch.setattr(prepared_staged_forward, "_copy", lambda *a, **k: pytest.fail("invalid operand reached a prepared staging copy"))
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
    from cudnn.sdpa.fwd import prepared_staged_forward

    monkeypatch.setattr(prepared_staged_forward, "_copy", lambda *a, **k: pytest.fail("invalid operand reached a prepared staging copy"))
    monkeypatch.setattr(torch.Tensor, "copy_", lambda *a, **k: pytest.fail("aliased SF output reached staging"))
    with pytest.raises(ValueError, match="sf_o overlaps"):
        g.execute(vp, ws)


@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("split", [1, 4])
def test_staged_paged_qo_preserves_kv_pool_views(hnd, split, monkeypatch):
    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("SM107 per-tensor FP8 does not serve paged KV")
    import cutlass.cute.runtime as runtime

    _, tensors, storage = _case()
    for role in ("k", "v"):
        dense = tensors[role]
        pool = dense.reshape(2, 2, 8, 16, 128).permute(0, 2, 1, 3, 4).reshape(16, 2, 16, 128).contiguous()
        if role == "v":
            pool = pool.flip(0)
        tensors[role] = pool if hnd else pool.transpose(1, 2).contiguous().transpose(1, 2)
    table = torch.arange(16, dtype=torch.int32, device="cuda").reshape(2, 8)
    tensors["block_table"] = table
    tensors["block_table_v"] = 15 - table
    tensors["seq_kv_lens"] = torch.full((2,), 128, dtype=torch.int32, device="cuda")
    api = SdpaFwdDslSm100(
        **{"sample_" + name: tensors[name] for name in ("q", "k", "v", "o", "lse")},
        pertensor_fp8=True,
        split_kv=split,
        seq_kv_lens_present=True,
        paged_page_size=16,
        paged_max_seq_len_kv=128,
    )
    assert api.check_support()
    for name in ("make_fake_tensor", "make_fake_compact_tensor"):
        monkeypatch.setattr(runtime, name, lambda *a, **k: pytest.fail("paged conversion constructed a tensor fake"))
    api.compile()
    assert {r[0] for r in api._staged_spec.regions} == {"q", "o"}
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")

    def check():
        dense = dict(tensors)
        for role, table_name in (("k", "block_table"), ("v", "block_table_v")):
            dense[role] = tensors[role][tensors[table_name].long()].permute(0, 2, 1, 3, 4).reshape(2, 2, 128, 128)
        _check(dense, storage)

    _execute(api, tensors, workspace)
    check()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _execute(api, tensors, workspace)
        tensors["block_table_v"].copy_(tensors["block_table_v"].flip(0))
        tensors["descale_v"].fill_(0.4)
        tensors["o"].fill_(float("nan"))
        graph.replay()
        check()
    finally:
        graph.reset()


@pytest.mark.parametrize("converted", ["q", "k", "v", "o"])
def test_sm107_d256_staging_preserves_each_native_operand(converted, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 D256 mixed-layout contract")
    from cudnn.sdpa.fwd import prepared_staged_forward

    _, tensors, storage = _case(256, 256)
    for role in ("q", "k", "v", "o"):
        if role == converted:
            continue
        t = tensors[role]
        b, h, s, d = t.shape
        raw = torch.full((b, s, h, 2 * d), 11, dtype=t.dtype, device=t.device)
        view = raw[..., :d].transpose(1, 2)
        view.copy_(t)
        tensors[role], storage[role] = view, raw
    api = SdpaFwdDslSm100(**{"sample_" + n: tensors[n] for n in ("q", "k", "v", "o", "lse")}, pertensor_fp8=True)
    assert api.check_support()
    compact, operands, regions, required = prepared_staged_forward._layout(api)
    assert [r[0] for r in regions] == [converted], "native operands must not acquire conversion copies"
    api.compile()
    assert [r[0] for r in api._staged_spec.regions] == [converted]
    assert api.scratch_workspace_bytes() == required
    workspace = torch.empty(required, device="cuda", dtype=torch.uint8)
    from cudnn.sdpa.fwd import prepared

    execute = prepared.execute_quantized
    seen = []

    def launch(spec, facts, *args, **kwargs):
        for role in ("q", "k", "v", "o"):
            assert (facts[role].ptr == tensors[role].data_ptr()) == (role != converted)
        seen.append(True)
        return execute(spec, facts, *args, **kwargs)

    monkeypatch.setattr(prepared, "execute_quantized", launch)
    _execute(api, tensors, workspace)
    _check(tensors, storage)
    for role in ("q", "k", "v", "o"):
        old = tensors[role]
        raw = torch.full_like(storage[role], 11)
        view = raw[..., :256].transpose(1, 2)
        view.copy_(old)
        tensors[role], storage[role] = view, raw
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
    assert len(seen) >= 3
