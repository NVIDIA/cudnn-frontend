# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 grouped d128 backward: native partial storage and FP32 group reduction.

Graph cases pin engine 20609. Regular numerics use the unchanged independent
FP64 gates; the exact-P cancellation test separately pins documented native
partial rounding. GPU KV carriers contain Hkv heads, never replicated Hq data.
"""

import pytest
import torch

import cudnn
from d128_grouped_test_utils import (
    GRADS,
    adapter_arguments,
    cancellation_sequences,
    check_case,
    copy_case_storage,
    make_adapter,
    make_case,
    make_graph,
    make_workspace,
    poison,
    reference_sequences,
    variant_pack,
)
from frost_test_utils import requires_dsl
from test_sdpa_bwd_d128_accum_sm100 import _require_accumulation
from test_sdpa_bwd_dsl_sm107 import _reference64

pytestmark = [pytest.mark.L0, requires_dsl]
_DTYPES = [torch.bfloat16]


def _graph_case(case):
    graph, refs = make_graph(case)
    return graph, refs, variant_pack(case, refs), make_workspace(graph.get_workspace_size())


def _stable_kv(case, previous=None):
    current = {role: case.tensors[role].cpu().view(torch.int16) for role in ("dk", "dv")}
    if previous is not None:
        for role in current:
            assert torch.equal(current[role], previous[role]), f"{role}: fixed-order group reduction changed output bits"
    return current


@pytest.mark.parametrize(
    "hq,hkv,dtype,causal,thd",
    [
        (4, 2, torch.bfloat16, False, False),
        (6, 2, torch.bfloat16, True, False),
        (3, 1, torch.bfloat16, False, False),
        (8, 1, torch.bfloat16, True, False),
        (4, 2, torch.bfloat16, True, True),
        (6, 2, torch.bfloat16, False, True),
        (3, 1, torch.bfloat16, True, True),
        (8, 1, torch.bfloat16, False, True),
    ],
)
def test_grouped_matrix(hq, hkv, dtype, causal, thd):
    _require_accumulation()
    # Pair each head ratio with a different storage contract, rather than
    # repeating the entire MHA storage Cartesian matrix for every ratio.
    layout, offset_dtype, units = {
        (4, 2): ("TH1", torch.int32, "tokens"),
        (6, 2): ("HN", torch.int64, "elements"),
        (3, 1): ("TH1", torch.int64, "elements"),
        (8, 1): ("HN", torch.int32, "tokens"),
    }[hq, hkv]
    lq, lk = ((129, 0, 127, 1), (65, 128, 0, 3)) if thd else ((128, 128), (256, 256))
    case = make_case(lq, lk, hq=hq, hkv=hkv, thd=thd, dtype=dtype, causal=causal, stats_layout=layout, offset_dtype=offset_dtype, units=units)
    graph, _, pack, workspace = _graph_case(case)
    stable = None
    for _ in range(2):
        poison(case, workspace)
        graph.execute(pack, workspace.tensor)
        check_case(case, workspace)
        stable = _stable_kv(case, stable)


@pytest.mark.parametrize("dtype,causal", [(torch.bfloat16, True)])
def test_grouped_thd_all_empty(dtype, causal):
    _require_accumulation()
    case = make_case((0, 0), (0, 0), hq=3, hkv=1, dtype=dtype, causal=causal, stats_layout="HN", envelope=(128, 128))
    graph, _, pack, workspace = _graph_case(case)
    poison(case, workspace)
    graph.execute(pack, workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize("ports", [pytest.param((), id="compact"), pytest.param(("q", "v", "stats", "dk"), id="mixed")])
@pytest.mark.parametrize("dtype", _DTYPES, ids=["bf16"])
def test_grouped_thd_optional_origins_and_length_forms(ports, dtype):
    _require_accumulation()
    case = make_case((65, 0, 3), (17, 129, 0), hq=6, hkv=2, dtype=dtype, causal=True, stats_layout="HN", origin_ports=ports)
    api = make_adapter(case)
    workspace = make_workspace(api.scratch_workspace_bytes())
    # Both B and B+1 forms, independently per side, reuse the same compiled
    # plan. Prefix bases normalize lengths, never explicit ragged origins.
    for prefix_q, prefix_kv in ((False, False), (True, False), (False, True), (True, True)):
        args = adapter_arguments(case)
        for role, lengths, prefix in (("seq_q_lens", case.lens_q, prefix_q), ("seq_kv_lens", case.lens_kv, prefix_kv)):
            if prefix:
                values = [37]
                for n in lengths:
                    values.append(values[-1] + n)
                args[role] = torch.tensor(values, dtype=torch.int32, device="cuda")
        poison(case, workspace)
        api.execute(**args, workspace=workspace.tensor)
        check_case(case, workspace)


@pytest.mark.parametrize("dtype,causal", [(torch.bfloat16, False)])
def test_grouped_native_partials_fp32_fold_cancellation(dtype, causal):
    _require_accumulation()
    sequences = cancellation_sequences(dtype, causal)
    case = make_case((2,), (1,), hq=3, hkv=1, dtype=dtype, causal=causal, sequences=sequences)
    graph, _, pack, workspace = _graph_case(case)
    poison(case, workspace)
    graph.execute(pack, workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize("dtype", _DTYPES, ids=["bf16"])
def test_grouped_autograd_oracle_matches_analytic_fp64(dtype):
    # This CPU calculation is independent of CUDA implementation and storage.
    for hq, hkv in ((4, 2), (6, 2), (3, 1), (8, 1)):
        for causal in (False, True):
            for seq in reference_sequences((1, 3, 0), (5, 1, 2), hq, hkv, dtype, causal):
                nq, nk = seq["q"].shape[0], seq["k"].shape[0]
                if not nq or not nk:
                    assert all(torch.count_nonzero(seq[r]) == 0 for r in GRADS)
                    continue
                operands = [seq[r].transpose(0, 1).unsqueeze(0) for r in ("q", "k", "v", "do")]
                keep = (torch.arange(nk)[None, :] <= torch.arange(nq)[:, None])[None, None] if causal else None
                ref = _reference64(*operands, keep=keep, group=hq // hkv)
                for role, expected in zip(GRADS, ref[3:], strict=True):
                    torch.testing.assert_close(seq[role], expected[0].transpose(0, 1), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "thd,dtype,causal",
    [
        (False, torch.bfloat16, False),
        (True, torch.bfloat16, True),
    ],
)
def test_grouped_launches_no_allocation_or_host_reads(thd, dtype, causal, monkeypatch):
    _require_accumulation()
    import cutlass.cute as cute
    from cudnn.sdpa.bwd import prepared

    lq, lk = ((129, 0, 65), (3, 257, 0)) if thd else ((128, 128), (256, 256))
    case = make_case(lq, lk, hq=4, hkv=2, thd=thd, dtype=dtype, causal=causal)
    graph, _, pack, workspace = _graph_case(case)
    graph.execute(pack, workspace.tensor)
    check_case(case, workspace)
    torch.cuda.synchronize()
    probe = case.lengths["seq_q"] if thd else case.tensors["q"].flatten()
    sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            probe[0].item()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]

        def forbidden(*args, **kwargs):
            raise AssertionError("prepared grouped execution used Python tensor plumbing, allocation, host read or compilation")

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
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.execute(pack, workspace.tensor)
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert names, "CUDA profiler activity is required for the launch/D2H gate"
    assert not any("dtoh" in name.lower() or "device to host" in name.lower() for name in names), names
    assert len(names) == (5 if thd else 4), names
    assert any("bprop_d128" in name for name in names), names
    check_case(case, workspace)


@pytest.mark.parametrize("dtype,layout", [(torch.bfloat16, "TH1"), (torch.bfloat16, "HN")], ids=["bf16-TH1", "bf16-HN"])
def test_grouped_capture_rebind_lengths_origins_and_storage(dtype, layout):
    _require_accumulation()
    options = dict(hq=6, hkv=2, dtype=dtype, causal=True, stats_layout=layout, envelope=(257, 257), capacities=(1024, 1024))
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
        stable = None
        for _ in range(3):
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


def test_grouped_compile_key_excludes_runtime_metadata(tmp_path, monkeypatch):
    _require_accumulation()
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache

    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    monkeypatch.delenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", raising=False)
    compiled_cache.reset_stats()
    first = make_case((129, 0, 65), (3, 257, 0), hq=3, hkv=1, stats_layout="HN", envelope=(257, 257), capacities=(1024, 1024))
    graph, _, pack, workspace = _graph_case(first)
    graph.execute(pack, workspace.tensor)
    check_case(first, workspace)
    initial = compiled_cache.stats()
    assert initial["misses"] > 0, initial

    def forbidden(*args, **kwargs):
        raise AssertionError("changed capacities, envelope, HN pitch or device metadata entered the compilation key")

    monkeypatch.setattr(cute, "compile", forbidden)
    second = make_case((257, 0, 65), (513, 0, 129), hq=3, hkv=1, seed=159, layout_shift=11, stats_layout="HN", envelope=(385, 513), capacities=(1536, 2048))
    graph2, _, pack2, workspace2 = _graph_case(second)
    graph2.execute(pack2, workspace2.tensor)
    check_case(second, workspace2)
    final = compiled_cache.stats()
    assert final["misses"] == initial["misses"] and final["hits"] > initial["hits"], (initial, final)


@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
def test_grouped_undersized_workspace_rejected_before_launch(thd):
    _require_accumulation()
    case = make_case((3,), (5,)) if thd else make_case((128,), (256,), thd=False)
    api = make_adapter(case)
    workspace = make_workspace(api.scratch_workspace_bytes())
    args = adapter_arguments(case)
    poison(case, workspace)
    before = {role: case.raw[role].clone() for role in GRADS}
    raw_workspace = workspace.raw.clone()
    with pytest.raises((ValueError, RuntimeError), match="workspace|Workspace"):
        api.execute(**args, workspace=workspace.tensor[:-1])
    for role in GRADS:
        torch.testing.assert_close(case.raw[role], before[role], rtol=0, atol=0, equal_nan=True)
    assert torch.equal(workspace.raw, raw_workspace)
    api.execute(**args, workspace=workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize("dtype,causal", [(torch.bfloat16, True)])
def test_grouped_thd_persistent_reuse(dtype, causal):
    """More Q-head/KV-tile owners than B200 SMs, including a zero-Q sequence."""
    _require_accumulation()
    case = make_case((257, 0, 129), (1025, 769, 513), hq=8, hkv=1, dtype=dtype, causal=causal, stats_layout="HN")
    graph, _, pack, workspace = _graph_case(case)
    stable = None
    for _ in range(3):
        poison(case, workspace)
        graph.execute(pack, workspace.tensor)
        check_case(case, workspace)
        stable = _stable_kv(case, stable)


@pytest.mark.parametrize("thd", [False, True], ids=["dense", "thd"])
def test_grouped_bf16_large_opposing_partials_keep_exact_zero(thd):
    """Native partials exceed FP16 range while the shared gradient is zero."""
    _require_accumulation()
    sq, skv, magnitude = (128, 1, 1024) if thd else (512, 128, 32768)
    inputs = dict(
        q=torch.zeros(sq, 2, 128, dtype=torch.bfloat16),
        k=torch.zeros(skv, 1, 128, dtype=torch.bfloat16),
        v=torch.zeros(skv, 1, 128, dtype=torch.bfloat16),
        do=torch.zeros(sq, 2, 128, dtype=torch.bfloat16),
    )
    inputs["do"][:, 0] = magnitude
    inputs["do"][:, 1] = -magnitude
    sequences = reference_sequences((sq,), (skv,), 2, 1, torch.bfloat16, False, inputs=[inputs])
    assert all(torch.count_nonzero(sequences[0][role]) == 0 for role in GRADS)
    assert magnitude * sq // skv == 131072
    case = make_case((sq,), (skv,), hq=2, hkv=1, thd=thd, dtype=torch.bfloat16, sequences=sequences)
    graph, _, pack, workspace = _graph_case(case)
    poison(case, workspace)
    graph.execute(pack, workspace.tensor)
    check_case(case, workspace)
    for role in GRADS:
        start, length = case.origins[role][0], sq if role == "dq" else skv
        actual = case.tensors[role][0, start : start + length].cpu()
        assert torch.equal(actual, torch.zeros_like(actual)), role
