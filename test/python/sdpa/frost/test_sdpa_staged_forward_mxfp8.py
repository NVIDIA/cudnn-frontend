# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 conversion plans preserve opaque scale buffers and current bindings."""

import pytest
import torch

from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
from frost_test_utils import requires_blackwell, requires_dsl
from sdpa.frost.test_sdpa_prepared_mxfp8 import _case as _native_case, _change_scales, _check

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell]
_FLAVORS = [(128, 128), (192, 128), (256, 256), (512, 512)]


def _case(d, dv, *, b=2, output_dtype=torch.bfloat16, split=1, stats=True, amax=True):
    cc = torch.cuda.get_device_capability()
    if cc[0] != 10:
        pytest.skip("SM100/SM103/SM107 MXFP8 conversion path")
    if cc == (10, 7) and split > 1:
        pytest.skip("SM107 MXFP8 does not serve split KV")
    _, _, _, buffers, _ = _native_case(
        d=d,
        dv=dv,
        b=b,
        output_dtype=output_dtype,
        split_kv=split,
        stats=stats,
        amax=amax,
        arch="sm107" if cc == (10, 7) else "sm100",
        explicit_plan=True,
    )
    storage = {}
    for role in ("q", "k", "v", "o"):
        t = buffers[role]
        b, h, s, dim = t.shape
        raw = torch.full((b, s, h, dim + 1), 11, device=t.device, dtype=t.dtype)
        view = raw[..., :dim].transpose(1, 2)
        if role != "o":
            view.copy_(t)
        storage[role], buffers[role] = raw, view
    api = SdpaFwdDslSm100(
        **{"sample_" + role: buffers.get(role) for role in ("q", "k", "v", "o", "lse")},
        sample_amax_o=buffers.get("amax_o"),
        has_amax_o=amax,
        split_kv=split,
        pertensor_fp8=False,
    )
    assert api.check_support()
    assert not api._can_prepare_mxfp8()
    return api, buffers, storage


def _execute(api, buffers, workspace, stream=None):
    api.execute(
        **{role + "_tensor": buffers.get(role) for role in ("q", "k", "v", "o", "lse")},
        **{role: buffers.get(role) for role in ("sf_q", "sf_k", "sf_v", "amax_o")},
        workspace=workspace,
        current_stream=stream,
    )


@pytest.mark.parametrize("d,dv", _FLAVORS)
def test_mxfp8_staged_uses_pointer_host_and_current_scales(d, dv, monkeypatch):
    import cutlass.cute as cute

    api, buffers, storage = _case(d, dv)
    monkeypatch.setattr(cute.runtime, "make_fake_tensor", lambda *a, **k: pytest.fail("MXFP8 conversion constructed a tensor fake"))
    monkeypatch.setattr(cute.runtime, "make_fake_compact_tensor", lambda *a, **k: pytest.fail("MXFP8 conversion constructed a compact tensor fake"))
    required = api.scratch_workspace_bytes()
    api.compile()
    assert api._staged_spec is not None
    assert api.scratch_workspace_bytes() == required
    for offset in (128, 256):
        _, buffers, storage = _case(d, dv)
        backing = torch.full((required + offset + 129,), 177, device="cuda", dtype=torch.uint8)
        workspace = backing[offset : offset + required + 1]
        _execute(api, buffers, workspace)
        _check(buffers, thd=False)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                _execute(api, buffers, workspace)
            _change_scales(buffers)
            buffers["o"].fill_(float("nan"))
            graph.replay()
            _check(buffers, thd=False)
        finally:
            graph.reset()
        assert torch.all(storage["o"][..., -1] == 11)
        assert torch.all(backing[:offset] == 177)
        assert torch.all(backing[offset + required :] == 177)


@pytest.mark.parametrize("d,dv", _FLAVORS)
@pytest.mark.parametrize("split", [1, 4])
def test_mxfp8_staged_fp8_output_without_optional_outputs(d, dv, split):
    from cuda.bindings import driver

    api, buffers, _ = _case(d, dv, output_dtype=torch.float8_e4m3fn, split=split, stats=False, amax=False)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    _execute(api, buffers, workspace, driver.CUstream(stream.cuda_stream))
    torch.cuda.current_stream().wait_stream(stream)
    _check(buffers, thd=False)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            _execute(api, buffers, workspace, driver.CUstream(stream.cuda_stream))
        _change_scales(buffers)
        buffers["o"].fill_(float("nan"))
        graph.replay()
        _check(buffers, thd=False)
    finally:
        graph.reset()


@pytest.mark.parametrize("role", ["sf_q", "sf_k", "sf_v"])
@pytest.mark.parametrize("bad", ["missing", "short", "gapped", "device", "workspace_alias"])
def test_mxfp8_staged_sf_rejects_before_copy(role, bad, monkeypatch):
    api, buffers, _ = _case(128, 128)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    original = buffers[role]
    if bad == "missing":
        buffers[role] = None
    elif bad == "short":
        buffers[role] = original.reshape(-1)[:-16]
    elif bad == "gapped":
        buffers[role] = original[..., ::2]
    elif bad == "device":
        buffers[role] = torch.empty(original.shape, dtype=original.dtype)
    else:
        buffers[role] = workspace[: original.numel()].view(original.dtype).view(original.shape)
    from cudnn.sdpa.fwd import prepared_staged_forward

    monkeypatch.setattr(prepared_staged_forward, "_copy", lambda *a, **k: pytest.fail("invalid operand reached a prepared staging copy"))
    monkeypatch.setattr(torch.Tensor, "copy_", lambda *a, **k: pytest.fail("invalid SF reached a staging copy"))
    with pytest.raises(ValueError):
        _execute(api, buffers, workspace)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d,dv", _FLAVORS)
@pytest.mark.parametrize("role,product", [("q", False), ("o", True)])
def test_mxfp8_staged_physical_wide_batch_stride(d, dv, role, product):
    api, buffers, _ = _case(d, dv, b=4 if product else 2)
    api.compile()
    original = buffers[role]
    batch_stride = 2**31 - 65536 if product else 2**32 + 65536
    required = ((original.shape[0] - 1) * batch_stride + original[0].numel() * 2) * original.element_size()
    if torch.cuda.mem_get_info()[0] < required + 2**30:
        pytest.skip("wide physical stride storage unavailable")
    try:
        wide = torch.empty_strided(original.shape, (batch_stride, original.stride(1), original.stride(2), 1), device="cuda", dtype=original.dtype)
    except torch.OutOfMemoryError:
        pytest.skip("wide physical stride allocation unavailable")
    wide.copy_(original)
    buffers[role] = wide
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    _execute(api, buffers, workspace)
    _check(buffers, thd=False)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _execute(api, buffers, workspace)
        _change_scales(buffers)
        buffers["o"].fill_(float("nan"))
        graph.replay()
        _check(buffers, thd=False)
    finally:
        graph.reset()


@pytest.mark.parametrize("d", [128, 192])
@pytest.mark.parametrize("amax", [False, True])
def test_staged_pv_bf16_keeps_v_width_and_omits_sf_v(d, amax):
    from sdpa.frost.test_sdpa_prepared_pv_bf16 import _case as pv_case, _check as pv_check

    _, buffers, _, scales = pv_case(d=d, amax=amax, has_amax_o=amax)
    for role in ("q", "k", "v", "o"):
        tensor = buffers[role + "_tensor"]
        b, h, seq, dim = tensor.shape
        storage = torch.empty((b, seq, h, dim + 1), device="cuda", dtype=tensor.dtype)
        view = storage[..., :dim].transpose(1, 2)
        view.copy_(tensor)
        buffers[role + "_tensor"] = view
    api = SdpaFwdDslSm100(
        **{"sample_" + role: buffers[role + "_tensor"] for role in ("q", "k", "v", "o", "lse")},
        sample_amax_o=buffers["amax_o"],
        has_amax_o=amax,
        pv_bf16=True,
        is_causal=True,
    )
    assert api.check_support()
    api.compile()
    assert len(api._staged_spec.core.quant.sf_sizes) == 2
    assert api._staged_spec.core.quant.has_amax == amax
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    api.execute(**buffers, workspace=workspace)
    pv_check(buffers, scales)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            api.execute(**buffers, workspace=workspace)
        buffers["v_tensor"].mul_(0.5)
        buffers["o_tensor"].fill_(float("nan"))
        graph.replay()
        pv_check(buffers, scales)
    finally:
        graph.reset()


@pytest.mark.parametrize("d,dv", _FLAVORS)
def test_mxfp8_staged_artifact_reloads_without_jit(d, dv, tmp_path):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100/SM103/SM107 MXFP8 conversion path")
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
from test_sdpa_staged_forward_mxfp8 import _case, _execute, _check
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
_check(tensors, thd=False)
graph = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(graph):
        _execute(api, tensors, workspace)
    tensors["o"].fill_(float("nan"))
    graph.replay()
    _check(tensors, thd=False)
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
def test_mxfp8_staged_block_output_matches_native_graph(block, dtype):
    from test_sdpa_prepared_block_output import _fp8_case

    reference = _fp8_case(block, mxfp8=True, dtype=dtype, stats=True)
    staged = _fp8_case(block, mxfp8=True, dtype=dtype, stats=True, staged=True)
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
            pack[tensors["sf_v"]].fill_(126)
        output.fill_(0xAA)
        sf.fill_(0xAA)
        amax.fill_(float("nan"))
        ref_g.execute(ref_vp, ref_ws)
        graph.replay()
        check()
    finally:
        graph.reset()


@pytest.mark.parametrize("rubin,d", [(False, 256), (False, 512), (True, 256)])
def test_mxfp8_compile_cli_uses_prepared_entry(rubin, d, monkeypatch):
    import sys
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    mod = _load_sm100_kernel_module((d, d), TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=1), fp8=True, pertensor=False, rubin=rubin)
    calls = []

    def compile_prepared():
        calls.append(True)
        return "compiled pointer entry"

    monkeypatch.setattr(mod, "compile_prepared", compile_prepared)
    monkeypatch.setattr(sys, "argv", ["compile-probe", "--b", "2", "--validate"])
    assert mod._main() == 0
    assert calls == [True]


@pytest.mark.parametrize("converted", ["q", "k", "v", "o"])
def test_sm107_d256_mxfp8_staging_preserves_each_native_operand(converted, monkeypatch):
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
    api = SdpaFwdDslSm100(
        **{"sample_" + n: tensors[n] for n in ("q", "k", "v", "o", "lse")}, pertensor_fp8=False, sample_amax_o=tensors["amax_o"], has_amax_o=True
    )
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
    _check(tensors, thd=False)
    for role in ("q", "k", "v", "o"):
        old = tensors[role]
        raw = torch.full_like(storage[role], 11)
        view = raw[..., :256].transpose(1, 2)
        view.copy_(old)
        tensors[role], storage[role] = view, raw
    _execute(api, tensors, workspace)
    _check(tensors, thd=False)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            _execute(api, tensors, workspace)
        _change_scales(tensors)
        tensors["o"].fill_(float("nan"))
        graph.replay()
        _check(tensors, thd=False)
    finally:
        graph.reset()
    assert len(seen) >= 3
