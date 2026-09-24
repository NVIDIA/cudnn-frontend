# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ahead-of-time plans: ``graph.serialize()`` of a graph whose selected plan is a CuTeDSL
engine, executed after ``deserialize()`` through the C++ AOT engine (no Python engine, no JIT).

Every round trip is compared BITWISE against the Python engine executing the same graph on the
same inputs: the artifact is the same kernels with the same arguments, so anything short of
identical output is a binding bug, not a tolerance question.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import textwrap

import pytest
import torch

import cudnn

pytestmark = [pytest.mark.L0]

DEV = torch.device("cuda")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _is_sm100_class():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


requires_frost_sdpa = pytest.mark.skipif(not _is_sm100_class(), reason="the exportable FROST SDPA plans are SM100/SM107 f16/bf16")


def _select_frost(g):
    from cudnn.sdpa.fwd.engines import engine_name

    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    want = engine_name()
    g.select_plan(next(i for i, n in enumerate(names) if n == want or n.startswith(want + "[")))


def _plan(g):
    return g._compiled_plans[g._plan_index]


def dense_graph(b, h, hk, s_q, s_kv, d, *, causal=True, split_kv=1):
    """Dense bf16 SDPA declared in BSHD storage (the layout the prepared launch binds zero-copy)."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def st(hh, s):
        return [s * hh * d, d, hh * d, 1]

    q = g.tensor(dim=[b, h, s_q, d], stride=st(h, s_q), data_type=cudnn.data_type.BFLOAT16, name="q")
    k = g.tensor(dim=[b, hk, s_kv, d], stride=st(hk, s_kv), data_type=cudnn.data_type.BFLOAT16, name="k")
    v = g.tensor(dim=[b, hk, s_kv, d], stride=st(hk, s_kv), data_type=cudnn.data_type.BFLOAT16, name="v")
    o, lse = g.sdpa(name="sdpa", q=q, k=k, v=v, generate_stats=True, attn_scale=1.0 / math.sqrt(d), use_causal_mask=causal)
    o.set_output(True).set_dim([b, h, s_q, d]).set_stride(st(h, s_q))
    lse.set_output(True).set_dim([b, h, s_q, 1]).set_stride([h * s_q, s_q, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    if split_kv == 1:
        _select_frost(g)
    else:
        from dataclasses import replace

        from cudnn.sdpa.fwd.engines import engine_name

        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        want = engine_name()
        idx = next(i for i, n in enumerate(names) if n == want or n.startswith(want + "["))
        chosen = g.plans[idx]
        g.create_execution_plan(chosen.engine_id, replace(chosen.knobs, split_kv=split_kv))
        g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    tensors = dict(q=q, k=k, v=v, o=o, lse=lse)

    torch.manual_seed(0)

    def mk(hh, s):
        return torch.randn(b, s, hh, d, device=DEV, dtype=torch.bfloat16).transpose(1, 2)

    bufs = dict(
        q=mk(h, s_q),
        k=mk(hk, s_kv),
        v=mk(hk, s_kv),
        o=torch.empty(b, s_q, h, d, device=DEV, dtype=torch.bfloat16).transpose(1, 2),
        lse=torch.empty(b, h, s_q, 1, device=DEV, dtype=torch.float32),
    )
    return g, tensors, bufs, ("o", "lse")


def thd_graph(b, ql, kl, hq, hk, d, *, padded_stats=False):
    """Ragged (THD) bf16 SDPA the way FlashInfer declares it; Stats packed token-major, or padded (B, H, S)."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    i32 = cudnn.data_type.INT32
    q = g.tensor(dim=[b, hq, ql, d], stride=[ql * hq * d, d, hq * d, 1], data_type=cudnn.data_type.BFLOAT16, name="q")
    k = g.tensor(dim=[b, hk, kl, d], stride=[kl * hk * d, d, hk * d, 1], data_type=cudnn.data_type.BFLOAT16, name="k")
    v = g.tensor(dim=[b, hk, kl, d], stride=[kl * hk * d, d, hk * d, 1], data_type=cudnn.data_type.BFLOAT16, name="v")
    cu_q = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="cu_q")
    cu_kv = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="cu_kv")
    off_q = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="off_q")
    off_kv = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="off_kv")
    q.set_ragged_offset(off_q)
    k.set_ragged_offset(off_kv)
    v.set_ragged_offset(off_kv)
    o, lse = g.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=True,
        attn_scale=1.0 / math.sqrt(d),
        use_causal_mask=True,
        use_padding_mask=True,
        cu_seq_len_q=cu_q,
        cu_seq_len_kv=cu_kv,
        max_total_seq_len_q=b * ql,
        max_total_seq_len_kv=b * kl,
    )
    o.set_output(True).set_dim([b, hq, ql, d]).set_stride([ql * hq * d, d, hq * d, 1]).set_ragged_offset(off_q)
    tensors = dict(q=q, k=k, v=v, o=o, lse=lse, cu_q=cu_q, cu_kv=cu_kv, off_q=off_q, off_kv=off_kv)
    if padded_stats:
        lse.set_output(True).set_dim([b, hq, ql, 1]).set_stride([hq * ql, ql, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
    else:
        off_lse = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="off_lse")
        lse.set_output(True).set_dim([b, hq, ql, 1]).set_stride([ql * hq, 1, hq, 1]).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(off_lse)
        tensors["off_lse"] = off_lse
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_frost(g)
    g.check_support()
    g.build_plans()

    torch.manual_seed(0)
    # Sequences shorter than the declared maxima, so padded Stats rows past each length are live.
    lens_q = torch.tensor([max(1, ql - 3 * i) for i in range(b)], dtype=torch.int32)
    lens_kv = torch.tensor([max(1, kl - 5 * i) for i in range(b)], dtype=torch.int32)
    cq = torch.cat([torch.zeros(1, dtype=torch.int32), lens_q.cumsum(0).to(torch.int32)]).to(DEV)
    ckv = torch.cat([torch.zeros(1, dtype=torch.int32), lens_kv.cumsum(0).to(torch.int32)]).to(DEV)
    bufs = dict(
        q=torch.randn(b * ql, hq, d, device=DEV, dtype=torch.bfloat16),
        k=torch.randn(b * kl, hk, d, device=DEV, dtype=torch.bfloat16),
        v=torch.randn(b * kl, hk, d, device=DEV, dtype=torch.bfloat16),
        o=torch.zeros(b * ql, hq, d, device=DEV, dtype=torch.bfloat16),
        lse=torch.zeros(b, hq, ql, 1, device=DEV, dtype=torch.float32) if padded_stats else torch.zeros(b * ql, hq, device=DEV, dtype=torch.float32),
        cu_q=cq,
        cu_kv=ckv,
        off_q=(cq * hq * d).to(torch.int32),
        off_kv=(ckv * hk * d).to(torch.int32),
    )
    if not padded_stats:
        bufs["off_lse"] = (cq * hq).to(torch.int32)
    return g, tensors, bufs, ("o", "lse")


def paged_decode_graph(b=4, h=8, hk=2, d=128, page=16, max_pages=8):
    """Decode (S_q = 1) over a paged NHD KV cache, per-batch KV lengths, Stats."""
    torch.manual_seed(0)
    n_pages = b * max_pages
    k_pool = torch.randn(n_pages, page, hk, d, device=DEV, dtype=torch.bfloat16)
    v_pool = torch.randn(n_pages, page, hk, d, device=DEV, dtype=torch.bfloat16)
    table = torch.randperm(n_pages, device=DEV).to(torch.int32).view(b, 1, max_pages, 1)
    bufs = dict(
        q=torch.randn(b, 1, h, d, device=DEV, dtype=torch.bfloat16).transpose(1, 2),
        k=k_pool.permute(0, 2, 1, 3),
        v=v_pool.permute(0, 2, 1, 3),
        k_table=table,
        v_table=table,
        seq_q=torch.ones(b, 1, 1, 1, device=DEV, dtype=torch.int32),
        seq_kv=torch.tensor([max_pages * page - 7 * i for i in range(b)], device=DEV, dtype=torch.int32).view(b, 1, 1, 1),
        o=torch.empty(b, 1, h, d, device=DEV, dtype=torch.bfloat16).transpose(1, 2),
        lse=torch.empty(b, h, 1, 1, device=DEV, dtype=torch.float32),
    )
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    t = {name: g.tensor_like(bufs[name]) for name in ("q", "k", "v", "k_table", "v_table", "seq_q", "seq_kv")}
    o, lse = g.sdpa(
        name="sdpa",
        q=t["q"],
        k=t["k"],
        v=t["v"],
        generate_stats=True,
        attn_scale=1.0 / math.sqrt(d),
        use_padding_mask=True,
        seq_len_q=t["seq_q"],
        seq_len_kv=t["seq_kv"],
        paged_attention_k_table=t["k_table"],
        paged_attention_v_table=t["v_table"],
        paged_attention_max_seq_len_kv=max_pages * page,
    )
    o.set_output(True).set_dim(bufs["o"].shape).set_stride(bufs["o"].stride())
    lse.set_output(True).set_dim(bufs["lse"].shape).set_stride(bufs["lse"].stride()).set_data_type(cudnn.data_type.FLOAT)
    t.update(o=o, lse=lse)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_frost(g)
    g.check_support()
    g.build_plans()
    return g, t, bufs, ("o", "lse")


def _pack(tensors, bufs):
    return {tensors[name].get_uid(): bufs[name] for name in tensors}


def _workspace(g):
    return torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)


def _poison(bufs, outputs):
    for name in outputs:
        bufs[name].fill_(float("nan") if bufs[name].is_floating_point() else 0)


def _run(g, tensors, bufs, outputs, ws=None):
    _poison(bufs, outputs)
    g.execute(_pack(tensors, bufs), _workspace(g) if ws is None else ws)
    torch.cuda.synchronize()
    return {name: bufs[name].clone() for name in outputs}


def _loaded(blob):
    g = cudnn.pygraph()
    g.deserialize(blob)
    return g


def _assert_bitwise(got, want):
    # Rows a ragged plan never writes keep the NaN poison in both runs.
    for name in want:
        torch.testing.assert_close(got[name], want[name], rtol=0, atol=0, equal_nan=True, msg=f"{name} differs")


CASES = {
    "dense": lambda: dense_graph(2, 8, 2, 256, 512, 128),
    "dense_decode": lambda: dense_graph(4, 8, 2, 1, 1024, 128, causal=False),
    "dense_split_kv": lambda: dense_graph(2, 8, 2, 64, 1024, 128, causal=False, split_kv=2),
    "thd": lambda: thd_graph(4, 64, 128, 8, 2, 128),
    "thd_padded_stats": lambda: thd_graph(4, 64, 128, 8, 2, 128, padded_stats=True),
    "paged_decode": lambda: paged_decode_graph(),
}


@requires_frost_sdpa
@pytest.mark.parametrize("case", sorted(CASES))
def test_round_trip_is_bitwise(case):
    g, tensors, bufs, outputs = CASES[case]()
    assert g.selected_engine is not None  # the plan being exported is a Python engine's
    want = _run(g, tensors, bufs, outputs)
    blob = g.serialize()
    loaded = _loaded(blob)
    assert loaded.selected_engine is None  # runs on the C++ AOT engine, not a Python engine
    assert loaded.get_workspace_size() == g.get_workspace_size()
    _assert_bitwise(_run(loaded, tensors, bufs, outputs), want)
    # A deserialized AOT graph writes its plan back, and that blob loads and runs the same.
    _assert_bitwise(_run(_loaded(loaded.serialize()), tensors, bufs, outputs), want)


@requires_frost_sdpa
def test_split_and_padded_stats_export_their_whole_sequence():
    """The split plan is two kernels (main + combine); the padded-Stats THD plan seeds Stats with a fill."""
    from cudnn.engines import aot

    g, *_ = CASES["dense_split_kv"]()
    steps = _payload_steps(g)
    assert [s["op"] for s in steps] == ["call", "call"] and steps[0]["module"] != steps[1]["module"]
    g, *_ = CASES["thd_padded_stats"]()
    ops = [s["op"] for s in _payload_steps(g)]
    assert ops[-1] == "call" and ops[:-1] and all(op.startswith("fill32") for op in ops[:-1])
    assert aot.FORMAT_VERSION == 1


def _export(g):
    """``(payload, modules)`` that serialize() hands to C++, captured on the way."""
    captured = {}
    lowered = g._lowered_graph
    real = lowered._serialize_aot

    class Spy:
        def __getattr__(self, name):
            return getattr(lowered, name)

        def _serialize_aot(self, payload, modules, uids):
            captured["payload"], captured["modules"] = json.loads(payload), list(modules)
            return real(payload, modules, uids)

    g._lowered_graph = Spy()
    try:
        g.serialize()
    finally:
        g._lowered_graph = lowered
    return captured["payload"], captured["modules"]


def _payload_steps(g):
    return _export(g)[0]["steps"]


@requires_frost_sdpa
def test_modules_are_shared_across_shapes():
    """The kernels are shape-generic: graphs of two shapes carry the same kernel module (which the
    loader maps once per process) and differ only in the bound arguments."""
    a, *_ = dense_graph(2, 8, 2, 256, 512, 128)
    b, *_ = dense_graph(1, 8, 2, 128, 256, 128)
    (pa, ma), (pb, mb) = _export(a), _export(b)
    assert ma == mb
    assert pa["steps"] != pb["steps"]


@requires_frost_sdpa
def test_overrides_are_refused():
    g, tensors, bufs, outputs = CASES["dense"]()
    loaded = _loaded(g.serialize())
    uid = tensors["q"].get_uid()
    with pytest.raises(Exception, match="overrides"):
        loaded.execute(_pack(tensors, bufs), _workspace(g), override_uids=[uid], override_shapes=[[1, 8, 128, 128]], override_strides=[[0, 128, 1024, 1]])


@requires_frost_sdpa
def test_wrong_device_is_refused_at_load(monkeypatch):
    """An artifact for another SKU is an error at deserialize, before any kernel initializes."""
    from cudnn.frost import device as _device

    g, *_ = CASES["dense"]()
    real = _device.multiprocessor_count(0)
    monkeypatch.setattr(_device, "multiprocessor_count", lambda dev: real + 4)
    blob = g.serialize()
    monkeypatch.undo()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="SMs"):
        _loaded(blob)


@requires_frost_sdpa
def test_plan_that_cannot_export_says_why():
    """A FROST plan on the tensor ABI (BHSD storage, which the prepared launch declines) is not
    exportable: serialize() raises naming the plan, and never writes some other plan instead."""
    b, h, s, d = 1, 4, 128, 128
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    st = [h * s * d, s * d, d, 1]  # BHSD storage
    q, k, v = (g.tensor(dim=[b, h, s, d], stride=st, data_type=cudnn.data_type.BFLOAT16, name=n) for n in "qkv")
    o, _ = g.sdpa(name="sdpa", q=q, k=k, v=v, generate_stats=False, attn_scale=1.0 / math.sqrt(d), use_causal_mask=True)
    o.set_output(True).set_dim([b, h, s, d]).set_stride(st)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    # Build a backend plan first: it becomes the C++ graph's candidate, which is what serialize()
    # used to write whatever plan was selected.
    backend = next(i for i in range(len(g.plans)) if g.plans[i].engine_id < 20000 and g.plans[i].engine_id >= 0)
    g.build_plan_at_index(backend)
    _select_frost(g)
    g.check_support()
    g.build_plans()
    assert _plan(g)._prepared is None
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="ahead of time"):
        g.serialize()


@requires_frost_sdpa
def test_execute_allocates_nothing_and_never_synchronizes():
    """Rule 8 for the deserialized plan (recipe R9): no sync, no allocation, three executes."""
    g, tensors, bufs, outputs = CASES["thd_padded_stats"]()
    loaded = _loaded(g.serialize())
    ws = _workspace(g)
    pack = _pack(tensors, bufs)
    loaded.execute(pack, ws)  # first call initializes the kernel module
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            loaded.execute(pack, ws)
    finally:
        torch.cuda.set_sync_debug_mode(mode)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before


@requires_frost_sdpa
def test_first_execute_under_cuda_graph_capture():
    """A lazily initialized kernel's first launch may happen inside a capture (a serving stack's
    first decode step), and the captured graph replays the kernel with fresh inputs."""
    g, tensors, bufs, outputs = CASES["dense_decode"]()
    blob = g.serialize()
    loaded = _loaded(blob)
    ws = _workspace(g)
    stream = torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    stream.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        loaded.execute(_pack(tensors, bufs), ws, handle=handle)
    bufs["q"].mul_(0.5)
    want = _run(g, tensors, bufs, outputs)
    _poison(bufs, outputs)
    capture.replay()
    torch.cuda.synchronize()
    _assert_bitwise({n: bufs[n] for n in outputs}, want)


@requires_frost_sdpa
def test_fresh_process_never_imports_the_compiler(tmp_path):
    """The point of the artifact: a process that loads and runs it imports neither cutlass nor
    any kernel package (the engine registry ``import cudnn`` loads is not one), and needs nothing
    on LD_LIBRARY_PATH."""
    g, tensors, bufs, outputs = CASES["thd"]()
    want = _run(g, tensors, bufs, outputs)
    blob_path = tmp_path / "plan.bin"
    blob_path.write_bytes(bytes(g.serialize()))
    torch.save({"bufs": bufs, "want": want, "uids": {n: t.get_uid() for n, t in tensors.items()}}, tmp_path / "io.pt")
    script = textwrap.dedent(f"""
        import sys, torch, cudnn
        io = torch.load({str(tmp_path / 'io.pt')!r})
        bufs, want, uids = io["bufs"], io["want"], io["uids"]
        g = cudnn.pygraph()
        g.deserialize(open({str(blob_path)!r}, "rb").read())
        for n in want:
            bufs[n].fill_(float("nan"))
        ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
        g.execute({{uids[n]: b for n, b in bufs.items()}}, ws)
        torch.cuda.synchronize()
        for n in want:
            torch.testing.assert_close(bufs[n], want[n], rtol=0, atol=0, equal_nan=True)
        kernels = ("cudnn.sdpa", "cudnn.gemm", "cudnn.conv", "cudnn.linear_attention", "cudnn.frost.compiled_cache", "cudnn.engines.aot")
        loaded = sorted(m for m in sys.modules if m.split(".")[0] == "cutlass" or m.startswith(kernels))
        assert not loaded, loaded
        print("OK")
        """)
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, env=env)
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout + proc.stderr


def _cxx():
    return os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")


@requires_frost_sdpa
@pytest.mark.skipif(_cxx() is None, reason="needs a C++ compiler")
def test_cpp_consumer(tmp_path):
    """Pure C++: samples/aot/run_plan.cpp loads the blob with cudnn_frontend::graph::Graph and executes it."""
    g, tensors, bufs, outputs = CASES["dense"]()
    want = _run(g, tensors, bufs, outputs)
    (tmp_path / "plan.bin").write_bytes(bytes(g.serialize()))
    # The consumer reads raw little-endian buffers keyed by uid and writes the outputs back the same way.
    for name, t in tensors.items():
        buf = bufs[name]
        span = 1 + sum((s - 1) * st for s, st in zip(buf.shape, buf.stride()))
        raw = torch.as_strided(buf, (span,), (1,)) if name not in outputs else torch.zeros(span, dtype=buf.dtype, device=DEV)
        raw.contiguous().view(torch.uint8).cpu().numpy().tofile(tmp_path / f"{t.get_uid()}.bin")

    exe = tmp_path / "run_plan"
    env = _build_cpp(os.path.join(REPO, "samples", "aot", "run_plan.cpp"), exe)
    run = subprocess.run([str(exe), str(tmp_path / "plan.bin"), str(tmp_path)], capture_output=True, text=True, timeout=300, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    for name in outputs:
        buf = bufs[name]
        span = 1 + sum((s - 1) * st for s, st in zip(buf.shape, buf.stride()))
        raw = torch.from_numpy(__import__("numpy").fromfile(tmp_path / f"{tensors[name].get_uid()}.bin", dtype="uint8")).to(DEV)
        got = torch.as_strided(raw.view(buf.dtype)[:span], buf.shape, buf.stride())
        torch.testing.assert_close(got, want[name], rtol=0, atol=0, equal_nan=True)


_CONCURRENT_CPP = r"""
#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <thread>
#include <unordered_map>
#include <vector>

void *cudnn_frontend::cudnn_dlhandle = nullptr;

int main(int argc, char **argv) {
    (void)argc;
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cudaSetDevice(0);
    cudnn_frontend::graph::Graph graph;
    auto status = graph.deserialize(nullptr, blob);
    if (status.is_bad()) { std::fprintf(stderr, "%s\n", status.get_message().c_str()); return 1; }
    int64_t ws_size = 0;
    if (graph.get_workspace_size(ws_size).is_bad()) return 1;
    auto const uids = graph.get_variant_pack_uids_sorted();
    constexpr int kThreads = 8;
    std::vector<std::unordered_map<int64_t, void *>> packs(kThreads);
    std::vector<void *> workspaces(kThreads);
    for (int t = 0; t < kThreads; t++) {
        for (int64_t uid : uids) {
            void *p = nullptr;
            cudaMalloc(&p, 64 << 20);
            cudaMemset(p, 0, 64 << 20);
            packs[t][uid] = p;
        }
        cudaMalloc(&workspaces[t], ws_size > 0 ? ws_size : 1);
    }
    cudaDeviceSynchronize();
    std::atomic<int> ready{0};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; t++) {
        threads.emplace_back([&, t] {
            cudaSetDevice(0);
            ready.fetch_add(1);
            while (ready.load() < kThreads) {
            }
            if (graph.execute(nullptr, packs[t], workspaces[t]).is_bad()) failures.fetch_add(1);
        });
    }
    for (auto &th : threads) th.join();
    cudaDeviceSynchronize();
    std::printf("%s\n", failures.load() == 0 ? "OK" : "FAILED");
    return failures.load();
}
"""


def _cudnn_include():
    if os.environ.get("CUDNN_PATH"):
        return os.path.join(os.environ["CUDNN_PATH"], "include")
    try:
        import nvidia.cudnn

        return os.path.join(list(nvidia.cudnn.__path__)[0], "include")
    except ImportError:
        return "/usr/include"


def _build_cpp(source_path, exe):
    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    if not os.path.isdir(os.path.join(cuda_home, "include")):
        pytest.skip(f"no CUDA toolkit headers under {cuda_home}")
    cmd = [
        _cxx(),
        "-std=gnu++17",  # what CMake's cxx_std_17 selects, with the repo's own warning set (CMakeLists.txt)
        "-O1",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-Wno-error=attributes",
        "-Wno-attributes",
        "-Wno-error=unused-function",
        "-Wno-unused-function",
        "-fno-rtti",
        "-DNV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING",
        f"-I{os.path.join(REPO, 'include')}",
        f"-I{os.path.join(cuda_home, 'include')}",
        f"-I{_cudnn_include()}",
        str(source_path),
        "-o",
        str(exe),
        f"-L{os.path.join(cuda_home, 'lib64')}",
        "-lcudart",
        "-ldl",
        "-pthread",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    # No Python in that process: the runtime libraries come from the loader's search path.
    import tvm_ffi

    from cudnn.engines import aot

    runtime_dirs = sorted({os.path.dirname(p) for p in aot.runtime_libraries()} | {os.path.dirname(tvm_ffi.libinfo.find_libtvm_ffi())})
    return dict(os.environ, LD_LIBRARY_PATH=":".join(runtime_dirs + [os.path.join(cuda_home, "lib64"), os.environ.get("LD_LIBRARY_PATH", "")]))


@requires_frost_sdpa
@pytest.mark.skipif(_cxx() is None, reason="needs a C++ compiler")
def test_cpp_concurrent_first_execute(tmp_path):
    """Eight C++ threads make the FIRST call of the plan's kernel at once. The CuTeDSL runtime's
    one-time module init (nvidia-cutlass-dsl 4.7) deadlocks on that; the AOT engine serializes
    first calls. (Python callers cannot race here: execute holds the GIL.)"""
    g, *_ = CASES["dense"]()
    (tmp_path / "plan.bin").write_bytes(bytes(g.serialize()))
    (tmp_path / "concurrent.cpp").write_text(_CONCURRENT_CPP)
    env = _build_cpp(tmp_path / "concurrent.cpp", tmp_path / "concurrent")
    try:
        run = subprocess.run([str(tmp_path / "concurrent"), str(tmp_path / "plan.bin")], capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("concurrent first execute hung (the CuTeDSL init lock deadlocked)")
    assert run.returncode == 0 and "OK" in run.stdout, run.stdout + run.stderr
