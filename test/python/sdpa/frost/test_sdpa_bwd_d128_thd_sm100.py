# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native d128 THD MHA: ragged storage, CPU FP64 numerics and prepared replay.

Graph tests strictly pin engine 20609. The standalone smoke exercises the same
native adapter before graph capability admission; no test patches capabilities.
These tests make no GQA/MQA support or performance claim.
"""

import pytest
import torch

import cudnn
from d128_thd_test_utils import (
    GRADS,
    ROLES,
    adapter_arguments,
    check_case,
    copy_case_storage,
    make_adapter,
    make_case,
    make_graph,
    make_workspace,
    poison,
    variant_pack,
)
from frost_test_utils import requires_dsl
from test_sdpa_bwd_d128_accum_sm100 import _require_accumulation

pytestmark = [pytest.mark.L0, requires_dsl]


def _graph_case(case):
    graph, refs = make_graph(case)
    return graph, refs, variant_pack(case, refs), make_workspace(graph.get_workspace_size())


def _stable_kv(case, previous=None):
    # MHA has one writer for each dK/dV output. Atomic dQ is deliberately not a
    # bitwise target, but nondeterminism must not spread to these outputs.
    current = {role: case.tensors[role].cpu().view(torch.int16) for role in ("dk", "dv")}
    if previous is not None:
        for role in current:
            assert torch.equal(current[role], previous[role]), f"{role}: repeated inputs changed native output bits"
    return current


@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
def test_thd_mha_standalone_smoke(dtype):
    """Small first dispatch; independent of the graph capability row."""
    _require_accumulation()
    case = make_case((3, 1), (5, 2), dtype=dtype, gaps=False, offset_dtype=torch.int32)
    api = make_adapter(case)
    assert api._prepared.artifact.implementation == "accumulation_thd"
    workspace = make_workspace(api.scratch_workspace_bytes())
    poison(case, workspace)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    api.execute(**adapter_arguments(case), workspace=workspace.tensor, current_stream=stream.cuda_stream)
    torch.cuda.current_stream().wait_stream(stream)
    check_case(case, workspace)


@pytest.mark.parametrize(
    "origin_ports,dtype,causal",
    [
        ((), torch.bfloat16, False),
        (("q", "stats", "dq"), torch.float16, True),
        (("q", "v", "stats", "dk"), torch.bfloat16, True),
    ],
)
def test_thd_mha_standalone_compact_and_mixed_origins(origin_ports, dtype, causal):
    # The public graph contract identifies nonpaged THD through Q/K/V ragged
    # declarations. Standalone also admits compact and optional-port origins.
    _require_accumulation()
    case = make_case((65, 0, 3), (17, 129, 0), origin_ports=origin_ports, dtype=dtype, causal=causal)
    assert set(case.offsets) == set(origin_ports)
    api = make_adapter(case)
    workspace = make_workspace(api.scratch_workspace_bytes())
    poison(case, workspace)
    api.execute(**adapter_arguments(case), workspace=workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize(
    "lens_q,lens_kv,heads,dtype,causal",
    [
        ((1, 63, 64, 65, 127, 128, 129, 255, 256, 257), (257, 256, 255, 129, 128, 127, 65, 64, 63, 1), 3, torch.bfloat16, False),
        ((1, 63, 64, 65, 127, 128, 129, 255, 256, 257), (257, 256, 255, 129, 128, 127, 65, 64, 63, 1), 3, torch.float16, True),
        ((129, 0, 65, 1), (3, 257, 0, 129), 2, torch.bfloat16, True),
        ((129, 0, 65, 1), (3, 257, 0, 129), 2, torch.float16, False),
        ((0, 0), (1, 129), 2, torch.bfloat16, False),
        ((1, 129), (0, 0), 2, torch.float16, True),
        ((0, 0), (0, 0), 2, torch.bfloat16, True),
        ((0, 0), (0, 0), 2, torch.float16, False),
    ],
)
def test_thd_mha_matrix(lens_q, lens_kv, heads, dtype, causal):
    """Empty sequences and both tile-boundary directions across both dtypes."""
    _require_accumulation()
    # The shared graph contract rejects declarations with S_q = S_kv = 1.
    # Actual empty sequences are tested under a valid nonempty envelope.
    envelope = (128, 128) if not any(lens_q) and not any(lens_kv) else None
    case = make_case(lens_q, lens_kv, heads=heads, dtype=dtype, causal=causal, envelope=envelope)
    graph, _, pack, workspace = _graph_case(case)
    stable = None
    for _ in range(2):
        poison(case, workspace)
        graph.execute(pack, workspace.tensor)
        check_case(case, workspace)
        stable = _stable_kv(case, stable)


@pytest.mark.parametrize("dtype,causal", [(torch.bfloat16, False), (torch.float16, True)])
def test_thd_mha_persistent_reuse(dtype, causal):
    """More KV owners than B200 SMs, with odd/even Q and KV tile tails."""
    _require_accumulation()
    case = make_case((385, 257, 129), (513, 385, 257), heads=16, dtype=dtype, causal=causal)
    graph, _, pack, workspace = _graph_case(case)
    stable = None
    for _ in range(3):
        poison(case, workspace)
        graph.execute(pack, workspace.tensor)
        check_case(case, workspace)
        stable = _stable_kv(case, stable)


@pytest.mark.parametrize(
    "layout,offset_dtype,units,dtype,causal",
    [
        ("TH1", torch.int32, "tokens", torch.bfloat16, False),
        ("TH1", torch.int64, "elements", torch.float16, True),
        ("HN", torch.int32, "elements", torch.bfloat16, True),
        ("HN", torch.int64, "tokens", torch.float16, False),
    ],
)
def test_thd_mha_independent_origins_and_stats(layout, offset_dtype, units, dtype, causal):
    _require_accumulation()
    case = make_case((65, 0, 3), (17, 129, 0), dtype=dtype, causal=causal, stats_layout=layout, offset_dtype=offset_dtype, units=units, layout_shift=19)
    assert len({case.origins[role] for role in ROLES}) == len(ROLES)
    assert all(case.origins[role][0] != 0 for role in ROLES)
    graph, _, pack, workspace = _graph_case(case)
    poison(case, workspace)
    graph.execute(pack, workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize("prefix_q,prefix_kv", [(True, False), (False, True)], ids=["prefix-length", "length-prefix"])
def test_thd_mha_standalone_length_forms(prefix_q, prefix_kv):
    _require_accumulation()
    case = make_case((65, 0, 3), (17, 129, 0), stats_layout="HN")
    api = make_adapter(case)
    workspace = make_workspace(api.scratch_workspace_bytes())
    args = adapter_arguments(case)
    # Prefix bases normalize LENGTHS only. Explicit ragged origins remain
    # independently nonzero; confusing the two changes every sequence address.
    for role, lengths, prefix in (("seq_q_lens", case.lens_q, prefix_q), ("seq_kv_lens", case.lens_kv, prefix_kv)):
        if prefix:
            values = [37]
            for n in lengths:
                values.append(values[-1] + n)
            args[role] = torch.tensor(values, dtype=torch.int32, device="cuda")
    poison(case, workspace)
    api.execute(**args, workspace=workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize(
    "layout,dtype,causal",
    [
        ("TH1", torch.bfloat16, False),
        ("HN", torch.float16, True),
    ],
)
def test_thd_mha_capture_rebind_lengths_origins_and_storage(layout, dtype, causal):
    _require_accumulation()
    options = dict(dtype=dtype, causal=causal, stats_layout=layout, envelope=(257, 257), capacities=(1024, 1024))
    old = make_case((129, 0, 65), (3, 257, 0), **options)
    graph, refs, pack, workspace = _graph_case(old)
    fresh = make_case((1, 127, 0), (129, 0, 65), seed=151, layout_shift=7, **options)
    changed = make_case((0, 193, 1), (17, 0, 129), seed=153, layout_shift=13, **options)
    fresh_workspace = make_workspace(graph.get_workspace_size())
    stream, ambient = torch.cuda.Stream(), torch.cuda.Stream()
    handle = cudnn.create_handle()
    capture = torch.cuda.CUDAGraph()
    cudnn.set_stream(handle, stream.cuda_stream)
    try:
        # Work follows the explicit handle stream even when another stream is
        # current. All input producers precede it by an explicit dependency.
        stable = None
        for _ in range(10):
            poison(old, workspace)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ambient):
                graph.execute(pack, workspace.tensor, handle=handle)
            torch.cuda.current_stream().wait_stream(stream)
            check_case(old, workspace)
            stable = _stable_kv(old, stable)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            with torch.cuda.stream(ambient):
                graph.execute(pack, workspace.tensor, handle=handle)

        # Rebinding every pointer on the same plan must not retarget an earlier
        # capture. This changes lengths, per-port origins and all data storage.
        poison(fresh, fresh_workspace)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(ambient):
            graph.execute(variant_pack(fresh, refs), fresh_workspace.tensor, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        check_case(fresh, fresh_workspace)
        snapshots = {role: fresh.tensors[role].clone() for role in GRADS}
        copy_case_storage(old, changed)
        stable = None
        for _ in range(3):
            poison(old, workspace)
            capture.replay()
            check_case(old, workspace)
            stable = _stable_kv(old, stable)
        for role in GRADS:
            torch.testing.assert_close(fresh.tensors[role], snapshots[role], rtol=0, atol=0)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("dtype,causal", [(torch.bfloat16, False)])
def test_thd_mha_four_launches_no_allocation_or_host_reads(dtype, causal, monkeypatch):
    _require_accumulation()
    import cutlass.cute as cute
    from cudnn.sdpa.bwd import prepared

    case = make_case((129, 0, 65), (3, 257, 0), dtype=dtype, causal=causal, stats_layout="HN")
    graph, _, pack, workspace = _graph_case(case)
    graph.execute(pack, workspace.tensor)
    check_case(case, workspace)
    torch.cuda.synchronize()
    probe = case.lengths["seq_q"]
    sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            probe[0].item()  # Prove the sync detector is armed.
        before = torch.cuda.memory_stats()["allocation.all.allocated"]

        def forbidden(*args, **kwargs):
            raise AssertionError("prepared THD execution used Python tensor plumbing, allocation, host read or compilation")

        with monkeypatch.context() as patch:
            patch.setattr(cute, "compile", forbidden)
            patch.setattr(prepared, "facts_of_roles", forbidden)
            for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "full", "tensor"):
                patch.setattr(torch, name, forbidden)
            for name in ("view", "reshape", "as_strided", "permute", "transpose", "cpu", "item", "tolist"):
                patch.setattr(torch.Tensor, name, forbidden)
            for _ in range(3):
                graph.execute(pack, workspace.tensor)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode(sync_mode)
    torch.cuda.synchronize()
    # The CUDA activity census also catches native D2H transfers missed by
    # Python hooks. CPU synchronization emitted by profiler shutdown is outside
    # the guarded execute region, so do not misclassify it as a kernel read.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.execute(pack, workspace.tensor)
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert names, "CUDA profiler activity is required for the launch/D2H gate"
    assert not any("dtoh" in name.lower() or "device to host" in name.lower() for name in names), names
    assert len(names) == 4, names
    assert "bprop_d128" in names[2], names
    check_case(case, workspace)


@pytest.mark.parametrize("layout", ["HN"])
def test_thd_mha_compile_key_excludes_runtime_metadata(layout, tmp_path, monkeypatch):
    _require_accumulation()
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache

    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    monkeypatch.delenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", raising=False)
    compiled_cache.reset_stats()
    options = dict(stats_layout=layout, envelope=(257, 257), capacities=(1024, 1024))
    first = make_case((129, 0, 65), (3, 257, 0), **options)
    graph, _, pack, workspace = _graph_case(first)
    graph.execute(pack, workspace.tensor)
    check_case(first, workspace)
    initial = compiled_cache.stats()
    assert initial["misses"] > 0, initial

    def forbidden(*args, **kwargs):
        raise AssertionError("changed length values, origins or pointers entered the compilation key")

    monkeypatch.setattr(cute, "compile", forbidden)
    # All seven partial-bound runtime scalars change: compact/physical Q/KV
    # capacities, both sequence envelopes and (for HN) the declared head pitch.
    # No dtype, B/H, mask, layout or offset-contract compile fact changes.
    second = make_case((257, 0, 65), (513, 0, 129), seed=159, layout_shift=11, stats_layout=layout, envelope=(385, 513), capacities=(1536, 2048))
    graph2, _, pack2, workspace2 = _graph_case(second)
    graph2.execute(pack2, workspace2.tensor)
    check_case(second, workspace2)
    final = compiled_cache.stats()
    assert final["misses"] == initial["misses"] and final["hits"] > initial["hits"], (initial, final)
