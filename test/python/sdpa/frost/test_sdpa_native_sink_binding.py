# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native sink validation precedes all launches and observes every call's storage."""

import pytest

import cudnn
from cudnn.sdpa.fwd import prepared as prep

pytestmark = [pytest.mark.L0]


def _fixture(thd, d=128, stats=True):
    if thd:
        import test_sdpa_native_thd_binding as fixture

        spec, facts, frames = fixture._fixture(layout="NH" if stats else None)
        binder = cudnn._pybind_module._SdpaThdBinder
        bind = lambda values: spec.native.bind(prep._native_pack_from_facts(values), prep._NATIVE_THD_INDICES, 0x30000, 17)
        execute = lambda values: spec.native.execute(prep._native_pack_from_facts(values), prep._NATIVE_THD_INDICES, 0x30000, 17)
        reference = lambda values: prep._bind_thd_python(spec, values, 0x30000, 17, 17)
    else:
        import test_sdpa_native_dense_binding as fixture

        spec, facts, frames = fixture._fixture(paged=True, d=d, lse=stats)
        binder = cudnn._pybind_module._SdpaDenseBinder
        bind = lambda values: fixture._native(spec, values)
        execute = lambda values: spec.native.execute(fixture._pack(values), prep._NATIVE_DENSE_INDICES, 17)
        reference = lambda values: prep.bind_dense(spec, values, 17, 17)
    spec.has_sink = True
    spec.native = binder(spec)
    facts["sinks"] = prep.BufferFacts(0x70000, "float32", (2, 0), 8, (1, 8, 1, 1), (8, 1, 1, 1))
    return spec, facts, frames, bind, execute, reference


@pytest.mark.parametrize("thd,d", [(False, 64), (False, 128), (False, 256), (True, 128)])
@pytest.mark.parametrize("change", ["missing", "dtype", "device", "host", "short", "misaligned", "size", "strides"])
def test_sink_rejects_bad_storage_after_warmup(thd, d, change):
    spec, facts, frames, bind, execute, reference = _fixture(thd, d)
    assert list(bind(facts)) == reference(facts)
    sink = facts["sinks"]
    updates = dict(
        dtype={"dtype": "bfloat16"},
        device={"device": (2, 1)},
        host={"device": (1, 0)},
        short={"span": 7},
        misaligned={"ptr": sink.ptr + 1},
        size={"shape": (9,), "strides": (1,), "span": 9},
        strides={"strides": (16, 2, 1, 1), "span": 15},
    )
    bad = dict(facts)
    if change == "missing":
        del bad["sinks"]
    else:
        bad["sinks"] = sink._replace(**updates[change])
    with pytest.raises(ValueError, match="sink"):
        reference(bad)
    with pytest.raises(ValueError, match="sink"):
        execute(bad)
    assert frames == []
    fresh = dict(facts, sinks=sink._replace(ptr=sink.ptr + 0x100000))
    assert list(bind(fresh)) == reference(fresh)
    execute(fresh)
    assert frames[-1][spec.index["sinks_ptr"]] == fresh["sinks"].ptr


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("shape,strides", [((8,), (1,)), ((1, 8, 1, 1), (2**34, 1, 2**35, 2**36)), ((2, 4), (4, 1))])
def test_sink_contiguous_shapes_and_dead_strides(thd, stats, shape, strides):
    spec, facts, _, bind, _, reference = _fixture(thd, stats=stats)
    facts["sinks"] = facts["sinks"]._replace(shape=shape, strides=strides)
    frame = bind(facts)
    assert list(frame) == reference(facts)
    assert frame[spec.index["sinks_ptr"]] == facts["sinks"].ptr


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("d,sq", [(64, 1), (128, 1), (256, 4)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_sink_native_graph_rebind_and_replay(paged, d, sq, dtype, monkeypatch, request):
    from test_sdpa_native_dense_binding import test_native_dense_graph_fresh_bindings_and_changed_replay as check

    check(paged, d, sq, dtype, monkeypatch, request, has_sink=True)


@pytest.mark.parametrize("d,sq", [(64, 1), (128, 1), (256, 4)])
def test_sink_native_standalone_rebind_and_replay(d, sq, monkeypatch):
    from test_sdpa_native_dense_binding import test_native_decode_standalone_rebinds_scale_and_capture as check

    check(d, sq, monkeypatch, has_sink=True)


@pytest.mark.parametrize("d", [128, 256])
def test_thd_sink_native_graph_rebind_and_replay(d, monkeypatch):
    import torch
    from frost_test_utils import _dsl_installed
    from test_sdpa_prepared_thd import _thd_graph, _buffers, _pack

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7), (12, 0)) or not _dsl_installed():
        pytest.skip("requires a supported Blackwell/Rubin half THD template")
    b, sq, sk, h, hk = 2, 4, 128, 8, 2
    arch = {(10, 7): "sm107", (12, 0): "sm120"}.get(torch.cuda.get_device_capability(), "sm100")
    graph, desc = _thd_graph(b, sq, sk, h, hk, d, causal=False, has_sink=True, arch=arch)
    launch = graph._compiled_plans[graph._plan_index]._prepared
    assert isinstance(launch, prep.PreparedThdLaunch) and launch.spec.native is not None
    monkeypatch.setattr(prep, "facts_of_roles", lambda *a: pytest.fail("native graph rebuilt Python facts"))
    monkeypatch.setattr(prep, "_bind_thd_python", lambda *a: pytest.fail("native graph used Python binding"))
    buf = _buffers(b, sq, sk, h, hk, d)
    sinks = torch.linspace(-2, 6, h, device="cuda", dtype=torch.float32).view(1, h, 1, 1)
    ws = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def call(ordered=False):
        pack = _pack(desc, buf)
        pack[desc["sink"]] = sinks
        if ordered:
            graph.execute(tuple(pack.values()), ws, tensor_uids=tuple(t.get_uid() for t in pack))
        else:
            graph.execute(pack, ws)

    def check():
        torch.cuda.synchronize()
        for i in range(b):
            q = buf["q"][i * sq : (i + 1) * sq].double().transpose(0, 1)
            k = buf["k"][i * sk : (i + 1) * sk].double().transpose(0, 1).repeat_interleave(h // hk, 0)
            v = buf["v"][i * sk : (i + 1) * sk].double().transpose(0, 1).repeat_interleave(h // hk, 0)
            scores = q @ k.transpose(-1, -2) * d**-0.5
            scores = torch.cat((scores, sinks.double().view(h, 1, 1).expand(h, sq, 1)), -1)
            expected = scores.softmax(-1)[..., :sk] @ v
            torch.testing.assert_close(buf["o"][i * sq : (i + 1) * sq].double(), expected.transpose(0, 1), atol=0.008, rtol=0.015)
            torch.testing.assert_close(buf["lse"][i * sq : (i + 1) * sq].double(), scores.logsumexp(-1).T, atol=0.003, rtol=0.001)

    call()
    check()
    old = buf, sinks, ws
    buf = {name: value.clone() for name, value in buf.items()}
    sinks, ws = sinks.clone().add_(1), torch.empty_like(ws)
    call(True)
    check()
    good = sinks
    sinks = good.view(-1)[:1]
    ws.fill_(37)
    buf["o"].fill_(123)
    with pytest.raises(ValueError):
        call(True)
    torch.cuda.synchronize()
    assert torch.all(buf["o"] == 123).item()
    assert torch.all(ws == 37).item(), "sink rejection must precede the setup kernel"
    sinks = good
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            call(True)
        stream.synchronize()
        with torch.cuda.graph(capture, stream=stream):
            call(True)
        sinks.add_(2)
        buf["q"].add_(0.2)
        buf["o"].fill_(float("nan"))
        buf["lse"].fill_(float("nan"))
        capture.replay()
        check()
    finally:
        capture.reset()
    del old


@pytest.mark.parametrize("d,dv,cga", [(128, 128, 2), (256, 256, 2), (192, 128, 2), (512, 512, 2)])
def test_thd_sink_native_standalone_rebind_and_replay(d, dv, cga, monkeypatch):
    import torch
    from frost_test_utils import _dsl_installed

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)) or not _dsl_installed():
        pytest.skip("requires a supported Blackwell/Rubin half THD template")
    from test_sdpa_thd_tensormap_acquire import test_thd_tensormaps_rebind_and_replay as check

    monkeypatch.setattr(prep, "_bind_thd_python", lambda *a: pytest.fail("native standalone used Python binding"))
    check(d, dv, cga, torch.bfloat16, has_sink=True)
