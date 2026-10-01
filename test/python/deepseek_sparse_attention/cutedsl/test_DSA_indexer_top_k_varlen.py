# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared indices-only Top-K contracts; performance belongs in offline screening."""

import gc
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import weakref

import pytest
import torch


@pytest.fixture
def api():
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip(f"requires CuTe DSL >=4.7: {version}")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 3), (10, 7)):
        pytest.skip("requires SM103 or SM107")
    if capability == (10, 7) and (error := cutedsl_arch_requirement_error(capability)):
        pytest.skip(error)
    module = importlib.import_module("cudnn.deepseek_sparse_attention.indexer_top_k.varlen_api")
    try:
        module._require_dsl(capability)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return module


def inputs(rows=8, cols=8192, k=512, nn=1, cr=1, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(20260930)
    scores = torch.randn((rows, cols), generator=generator, dtype=torch.bfloat16, device=device)
    lengths = torch.full((rows // nn,), cols * cr + nn - 1, dtype=torch.int32, device=device)
    out = torch.empty((rows, k), dtype=torch.int32, device=device)
    return scores, lengths, out


def assert_topk(scores, lengths, out, k, nn=1, cr=1):
    # Independent numerical-multiset oracle permits unordered output and ties.
    s, lens, result = scores.cpu().float(), lengths.cpu().tolist(), out.cpu().to(torch.int64)
    for r in range(s.shape[0]):
        count = min(s.shape[1], max(0, (int(lens[r // nn]) - nn + 1 + r % nn) // cr))
        take = min(k, count)
        selected = result[r, :take]
        assert (result[r, take:] == -1).all()
        assert ((selected >= 0) & (selected < count)).all()
        assert selected.unique().numel() == take
        actual = s[r, selected].sort(descending=True).values
        expected = s[r, :count].topk(take).values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.L0
def test_public_wrapper_and_metadata_only_plan(api):
    import cudnn
    from cudnn.api_base import TensorDesc

    s, lengths, out = inputs()
    descriptors = tuple(TensorDesc(t.dtype, tuple(t.shape), tuple(t.stride()), tuple(reversed(range(t.ndim))), t.device) for t in (s, lengths))
    plan = cudnn.IndexerTopKVarlen(*descriptors, 512)
    assert plan.check_support() and plan.scratch_workspace_bytes() == 0
    plan.compile()
    assert plan.execute(s, lengths, out) is None
    assert_topk(s, lengths, out, 512)
    assert cudnn.DSA.IndexerTopKVarlen is cudnn.IndexerTopKVarlen is api.IndexerTopKVarlen
    wrapped = cudnn.indexer_top_k_varlen_wrapper(s, lengths, 512)
    (unpacked,) = wrapped
    assert unpacked is wrapped["indices"]
    assert_topk(s, lengths, unpacked, 512)


@pytest.mark.L1
@pytest.mark.parametrize("cols,k,nn,cr", [(2053, 512, 2, 2), (8192, 1024, 2, 2), (32768, 2048, 1, 1), (131072, 512, 1, 1), (262144, 2048, 1, 1)])
def test_all_k_tail_and_int64_lengths(api, cols, k, nn, cr):
    s, lengths, out = inputs(24, cols, k, nn, cr)
    boundary = [-(2**31), -1, 0, 1, k * cr, k * cr + nn - 1, (k + 1) * cr, cols * cr, 2**31 - 1]
    lengths.copy_(torch.tensor((boundary * 3)[: lengths.numel()], dtype=torch.int32, device=s.device))
    plan = api.IndexerTopKVarlen(s, lengths, k, nn, cr)
    plan.compile()
    s[:, -1] = 64
    plan.execute(s, lengths, out)
    assert_topk(s, lengths, out, k, nn, cr)


@pytest.mark.L0
def test_fresh_bindings_graph_changes_and_uniform_fallback(api):
    s, lengths, out = inputs()
    plan = api.IndexerTopKVarlen(s, lengths, 512)
    plan.compile()
    s2, l2, o2 = inputs()
    assert s2.data_ptr() != s.data_ptr()
    plan.execute(s2, l2, o2)
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=capture_stream):
        plan.execute(s2, l2, o2)
        gc.collect()  # the plan owns no CUDA allocations/finalizers
    torch.cuda.current_stream().wait_stream(capture_stream)
    for phase in ("equal", "changed", "negative_inf", "signed_zero", "empty", "short", "int_min", "restored"):
        if phase == "equal":
            s2.fill_(1)
            l2.fill_(8192)
        elif phase == "changed":
            s2.copy_(s.flip(1))
            s2[:, -512:] = torch.inf
            l2.fill_(8192)
        elif phase == "negative_inf":
            s2.fill_(-torch.inf)
        elif phase == "signed_zero":
            s2.zero_()
            s2[:, 1::2] = -0.0
        elif phase == "empty":
            l2.zero_()
        elif phase == "short":
            l2.fill_(511)
        elif phase == "int_min":
            l2.fill_(-(2**31))
        else:
            s2.copy_(s)
            l2.fill_(2**31 - 1)
        before_s, before_l = s2.clone(), l2.clone()
        o2.fill_(-77)
        graph.replay()
        assert_topk(s2, l2, o2, 512)
        assert torch.equal(s2.view(torch.int16), before_s.view(torch.int16))
        assert torch.equal(l2, before_l)


@pytest.mark.L1
@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("nn,cr", [(1, 1), (2, 2)])
def test_nan_storage_outside_effective_prefix_is_ignored(api, k, nn, cr):
    s, lengths, out = inputs(12, 8192, k, nn, cr)
    finite_scores = s.clone()
    plan = api.IndexerTopKVarlen(s, lengths, k, nn, cr)
    plan.compile()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=capture_stream):
        plan.execute(s, lengths, out)
    torch.cuda.current_stream().wait_stream(capture_stream)

    # Poison physically present storage, including the whole row when L=0.
    # Change lengths across replay, including short/long and NN/CR boundaries.
    for phase in range(3):
        counts = [0, 1, k - 1, k, k + 1, 8187]
        counts = counts[phase:] + counts[:phase]
        raw_lengths = [(counts[i % len(counts)] * cr + nn - 1) for i in range(12 // nn)]
        lengths.copy_(torch.tensor(raw_lengths, dtype=torch.int32, device=s.device))
        s.copy_(finite_scores.roll(phase, dims=1))
        for row in range(12):
            count = min(8192, max(0, (raw_lengths[row // nn] - nn + 1 + row % nn) // cr))
            s[row, count:] = torch.nan
        original_bits, original_lengths = s.view(torch.int16).clone(), lengths.clone()
        for execute in (lambda: plan.execute(s, lengths, out), graph.replay):
            out.fill_(-77)
            execute()
            assert_topk(s, lengths, out, k, nn, cr)
            assert torch.equal(s.view(torch.int16), original_bits)
            assert torch.equal(lengths, original_lengths)


@pytest.mark.L0
@pytest.mark.parametrize("operand", [0, 1, 2])
def test_lazy_flags_and_malformed_bindings_reject_before_launch(api, operand, monkeypatch):
    tensors = list(inputs())
    plan = api.IndexerTopKVarlen(*tensors[:2], 512)
    plan.compile()
    monkeypatch.setattr(plan, "_entry", lambda *args: pytest.fail("invalid metadata reached launch"))
    negative = list(tensors)
    negative[operand] = torch._neg_view(negative[operand])
    with pytest.raises(ValueError, match="negative or conjugate"):
        plan.execute(*negative)
    wrong_type = list(tensors)
    wrong_type[operand] = wrong_type[operand].to(torch.float32)
    with pytest.raises(ValueError):
        plan.execute(*wrong_type)
    malformed = list(tensors)
    original = tensors[operand]
    storage = torch.empty(original.numel() + 1, dtype=original.dtype, device=original.device)
    malformed[operand] = storage[1:].view(original.shape)
    with pytest.raises(ValueError, match="aligned"):
        plan.execute(*malformed)
    strided = list(tensors)
    wide = torch.empty((*original.shape[:-1], original.shape[-1] * 2), dtype=original.dtype, device=original.device)
    strided[operand] = wide[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        plan.execute(*strided)
    changed = list(tensors)
    changed[operand] = original[..., :-1]
    with pytest.raises(ValueError):
        plan.execute(*changed)
    if operand == 0:
        wrong_rows = list(tensors)
        wrong_rows[0] = tensors[0][:4]
        wrong_rows[1] = tensors[1][:4]
        wrong_rows[2] = tensors[2][:4]
        with pytest.raises(ValueError):
            plan.execute(*wrong_rows)


@pytest.mark.L0
def test_output_alias_and_storage_observation_guard(api, monkeypatch):
    s, lengths, out = inputs()
    plan = api.IndexerTopKVarlen(s, lengths, 512)
    plan.compile()
    monkeypatch.setattr(plan, "_entry", lambda *args: pytest.fail("alias reached kernel"))
    alias = s.view(torch.int32).view(-1)[: out.numel()].view(out.shape)
    with pytest.raises(ValueError, match="overlap"):
        plan.execute(s, lengths, alias)
    base = torch.empty(out.numel() + lengths.numel(), dtype=torch.int32, device=s.device)
    aliased_lengths = base[: lengths.numel()]
    aliased_out = base[: out.numel()].view(out.shape)
    with pytest.raises(ValueError, match="overlap"):
        plan.execute(s, aliased_lengths, aliased_out)

    # A physically bounded host stand-in must also be rejected before any GPU
    # launch. Ordinary Torch cannot construct an out-of-storage as_strided view.
    class Storage:
        def nbytes(self):
            return 15

        def data_ptr(self):
            return 4096

    class Tensor:
        is_cuda = True

        def is_neg(self):
            return False

        def is_conj(self):
            return False

        def data_ptr(self):
            return 4096

        def numel(self):
            return 8

        def element_size(self):
            return 4

        def storage_offset(self):
            return 0

        def untyped_storage(self):
            return Storage()

    with pytest.raises(ValueError, match="observed storage"):
        api._span(Tensor(), "bounded probe")


@pytest.mark.L0
@pytest.mark.parametrize("bad", [(True, 1, 1), (256, 1, 1), (512, 0, 1), (512, 3, 1), (512, 1, 0), (512, 1, 2**31)])
def test_scalar_contract(api, bad):
    s, lengths, _ = inputs()
    with pytest.raises((TypeError, ValueError)):
        api.IndexerTopKVarlen(s, lengths, *bad)


@pytest.mark.L0
def test_plan_owns_no_tensors_and_build_execute_allocate_nothing(api, monkeypatch):
    s, lengths, out = inputs()
    plan = api.IndexerTopKVarlen(s, lengths, 512)
    refs = weakref.ref(s), weakref.ref(lengths)
    del s, lengths
    gc.collect()
    assert all(ref() is None for ref in refs)
    fresh_s, fresh_l, _ = inputs()
    torch.cuda.synchronize()  # outside the no-sync assertion
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    previous_mode = torch.cuda.get_sync_debug_mode()
    with monkeypatch.context() as guard:
        for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "full"):
            guard.setattr(torch, name, lambda *a, **k: pytest.fail("prepared path allocated a Torch tensor"))
        torch.cuda.set_sync_debug_mode("error")
        try:
            plan.compile()
            for _ in range(3):
                plan.execute(fresh_s, fresh_l, out)
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    assert_topk(fresh_s, fresh_l, out, 512)


@pytest.mark.L0
def test_artifact_reuses_different_declared_rows_with_jit_forbidden(api, monkeypatch):
    import cutlass.cute as cute
    from cudnn.deepseek_sparse_attention.indexer_top_k import varlen_kernel

    s, lengths, out = inputs()
    first = api.IndexerTopKVarlen(s, lengths, 512)
    first.compile()
    old_artifact = first._compiled_kernel
    api._wrapper_plans.clear()
    s2, l2, o2 = inputs(32)
    second = api.IndexerTopKVarlen(s2, l2, 512)
    with monkeypatch.context() as guard:
        guard.setattr(cute, "compile", lambda *a, **k: pytest.fail("row extent entered the artifact key"))
        second.compile()
        assert second._compiled_kernel is old_artifact
        second.execute(s2, l2, o2)
    assert_topk(s2, l2, o2, 512)
    assert varlen_kernel.compile_topk.cache_info().hits > 0


@pytest.mark.L0
def test_version_target_and_missing_entry_fail_before_execute(api, monkeypatch):
    import cutlass.cute as cute
    from cudnn.deepseek_sparse_attention.indexer_top_k import varlen_kernel
    from cudnn.frost import buffers

    with monkeypatch.context() as guard:
        # Both admission and the shared diagnostic must observe the same wheel.
        guard.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
        with pytest.raises(RuntimeError, match="4.6.2"):
            api._require_dsl((10, 3))
        guard.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.0")))
        with pytest.raises(RuntimeError, match="4.8.0.*4.7.0"):
            api._require_dsl((10, 7))
    s, lengths, _ = inputs()
    plan = api.IndexerTopKVarlen(s, lengths, 512)
    with monkeypatch.context() as guard:
        guard.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
        with pytest.raises(NotImplementedError, match="SM103 and SM107"):
            plan.check_support()
    with monkeypatch.context() as guard:
        guard.setattr(api, "positional_entry", lambda artifact: None)
        with pytest.raises(NotImplementedError, match="no executable entry"):
            plan.compile()
    assert plan._compiled_kernel is None
    with monkeypatch.context() as guard:
        guard.setattr(varlen_kernel, "compile_topk", lambda *a: (_ for _ in ()).throw(RuntimeError("ordinary compiler failure")))
        with pytest.raises(RuntimeError, match="ordinary compiler failure"):
            plan.compile()


@pytest.mark.L0
def test_explicit_stream_and_default_sentinels(api, monkeypatch):
    s, lengths, out = inputs()
    plan = api.IndexerTopKVarlen(s, lengths, 512)
    plan.compile()
    ambient, launch = torch.cuda.Stream(), torch.cuda.Stream()
    launch.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(ambient):
        plan.execute(s, lengths, out, current_stream=launch)
    launch.synchronize()
    assert_topk(s, lengths, out, 512)
    with monkeypatch.context() as guard:
        guard.setattr(torch.cuda, "ExternalStream", lambda *a, **k: pytest.fail("default sentinel was passed to ExternalStream"))
        for sentinel in (0, 1, 2):
            with torch.cuda.stream(ambient):
                plan.execute(s, lengths, out, current_stream=sentinel)
            torch.cuda.default_stream().synchronize()
            assert_topk(s, lengths, out, 512)


@pytest.mark.L1
def test_two_independent_graphs_and_pending_inputs(api):
    s1, l1, o1 = inputs()
    s2, l2, o2 = inputs()
    plan = api.IndexerTopKVarlen(s1, l1, 512)
    plan.compile()
    streams = torch.cuda.Stream(), torch.cuda.Stream()
    graphs = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    for stream, graph, tensors in zip(streams, graphs, ((s1, l1, o1), (s2, l2, o2))):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            plan.execute(*tensors)
    for phase in range(5):
        s1.fill_(phase)
        s2.copy_(s1.flip(1))
        s2[:, -512:] = 64
        l1.fill_(0 if phase == 1 else (511 if phase == 2 else 8192))
        l2.fill_(2**31 - 1 if phase == 3 else 8192)
        for stream, graph in zip(streams, graphs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(8):
                    graph.replay()
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        assert_topk(s1, l1, o1, 512)
        assert_topk(s2, l2, o2, 512)


@pytest.mark.L1
def test_operand_device_controls_compile_and_launch(api):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    operand = torch.device("cuda", 1)
    if torch.cuda.get_device_capability(operand) not in ((10, 3), (10, 7)):
        pytest.skip("second GPU is outside supported targets")
    api._require_dsl(tuple(torch.cuda.get_device_capability(operand)))
    s, lengths, out = inputs(device=operand)
    original = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        plan = api.IndexerTopKVarlen(s, lengths, 512)
        plan.compile()
        side = torch.cuda.Stream(device=operand)
        side.wait_stream(torch.cuda.current_stream(operand))
        plan.execute(s, lengths, out, current_stream=side)
        side.synchronize()
        assert torch.cuda.current_device() == 0
        assert_topk(s, lengths, out, 512)
    finally:
        torch.cuda.set_device(original)


@pytest.mark.L1
def test_fresh_process_artifact_reload_and_replay(api, tmp_path):
    import cudnn

    # Child verifies the exact selected source tree, including editable finders;
    # no PYTHONPATH assumption can silently choose another worktree.
    source = str(Path(cudnn.__file__).resolve().parent)
    script = textwrap.dedent(r"""
        import importlib, os, pkgutil, sys
        from pathlib import Path
        expected = Path(os.environ["TOPK_TEST_SOURCE"])
        for item in pkgutil.iter_modules():
            if item.name.startswith("__editable___nvidia_cudnn_frontend_") and item.name.endswith("_finder"):
                importlib.import_module(item.name).MAPPING["cudnn"] = str(expected)
        sys.path.insert(0, str(expected.parent))
        import cudnn, torch
        assert Path(cudnn.__file__).resolve().parent == expected
        import cutlass.cute as cute
        from cudnn.frost import compiled_cache
        from cudnn.deepseek_sparse_attention.indexer_top_k.varlen_api import IndexerTopKVarlen
        assert Path(sys.modules[IndexerTopKVarlen.__module__].__file__).is_relative_to(expected)
        compiled_cache.reset_stats()
        if os.environ["TOPK_TEST_RELOAD"] == "1":
            def fail(*args, **kwargs): raise AssertionError("fresh-process artifact reload tried JIT")
            cute.compile = fail
        s = torch.arange(8192, device="cuda", dtype=torch.float32).to(torch.bfloat16).repeat(8, 1)
        lens = torch.full((8,), 8192, device="cuda", dtype=torch.int32)
        out = torch.empty((8,512), device="cuda", dtype=torch.int32)
        plan = IndexerTopKVarlen(s,lens,512); plan.compile()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream): plan.execute(s,lens,out)
        torch.cuda.current_stream().wait_stream(stream)
        for count in (8192,511,0,8192):
            lens.fill_(count); s.neg_(); out.fill_(-77); graph.replay()
            cpu = out.cpu().long(); scores=s.cpu().float()
            for r in range(8):
                take=min(512,count); chosen=cpu[r,:take]
                assert chosen.unique().numel()==take and ((chosen>=0)&(chosen<count)).all()
                assert (cpu[r,take:]==-1).all()
                assert torch.equal(scores[r,chosen].sort().values,scores[r,:count].topk(take).values.sort().values)
        stats=compiled_cache.stats()
        if os.environ["TOPK_TEST_RELOAD"] == "1":
            assert stats["hits"] > 0, stats
        else:
            assert stats["misses"] > 0, stats
        assert hasattr(plan._compiled_kernel, "_compiled_cache_entry"), "artifact was not persisted/reloaded"
    """)
    env = dict(os.environ, TOPK_TEST_SOURCE=source, CUDNN_FRONTEND_COMPILED_CACHE=str(tmp_path / "cache"))
    env.pop("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", None)
    for reload in ("0", "1"):
        subprocess.run([sys.executable, "-c", script], env=dict(env, TOPK_TEST_RELOAD=reload), cwd=tmp_path, check=True, timeout=180)


@pytest.mark.L1
def test_shared_candidate_overflow_and_short_prefix_transitions(api):
    s, lengths, out = inputs(8, 32768)
    s.fill_(1)  # Each thread sees 32 tied entries, greater than CAP=8.
    plan = api.IndexerTopKVarlen(s, lengths, 512)
    plan.compile()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        plan.execute(s, lengths, out)
    torch.cuda.current_stream().wait_stream(stream)
    for value, count in ((1.0, 32768), (-torch.inf, 32768), (torch.inf, 32768), (0.0, 513), (-0.0, 0), (1.0, 32768)):
        s.fill_(value)
        lengths.fill_(count)
        out.fill_(-77)
        graph.replay()
        assert_topk(s, lengths, out, 512)


@pytest.mark.L0
def test_compile_capture_miss_is_explicit(api, monkeypatch):
    s, lengths, out = inputs()
    plan = api.IndexerTopKVarlen(s, lengths, 512)
    plan.check_support()
    with monkeypatch.context() as guard:
        guard.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="before CUDA graph capture"):
            plan(s, lengths, out)
    assert plan._compiled_kernel is None


@pytest.mark.L0
def test_target_compilation_uses_only_metadata_and_explicit_operand_arch(api, monkeypatch):
    from cudnn.deepseek_sparse_attention.indexer_top_k import varlen_kernel

    s, lengths, _ = inputs()
    plan = api.IndexerTopKVarlen(s, lengths, 512)
    calls = []
    sentinel = object()

    def compiler(*args):
        calls.append((args, torch.cuda.current_device()))
        return sentinel

    with monkeypatch.context() as guard:
        guard.setattr(varlen_kernel, "compile_topk", compiler)
        guard.setattr(api, "positional_entry", lambda artifact: lambda *a: None)
        plan.compile()
    capability = torch.cuda.get_device_capability(s.device)
    assert calls == [((8192, 512, 1, 1, s.device.index, f"sm_{capability[0]}{capability[1]}a"), s.device.index)]
    assert not any(isinstance(value, torch.Tensor) for value in calls[0][0])


@pytest.mark.L0
def test_runtime_records_each_allocation_on_actual_launch_stream(api, monkeypatch):
    s, lengths, out = inputs()
    plan = api.IndexerTopKVarlen(s, lengths, 512)
    plan.compile()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    recorded = []
    original = torch.Tensor.record_stream

    def record(tensor, launch_stream):
        recorded.append((tensor.data_ptr(), launch_stream.cuda_stream))
        return original(tensor, launch_stream)

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "record_stream", record)
        plan.execute(s, lengths, out, current_stream=stream)
    assert recorded == [(t.data_ptr(), stream.cuda_stream) for t in (s, lengths, out)]
    stream.synchronize()
    assert_topk(s, lengths, out, 512)


@pytest.mark.L1
def test_wrapper_allocates_on_explicit_consumer_stream(api, monkeypatch):
    s, lengths, _ = inputs()
    api.indexer_top_k_varlen_wrapper(s, lengths, 512)  # Prewarm both caches.
    ambient, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    records = []
    original = torch.empty

    def allocate(*args, **kwargs):
        records.append((torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream))
        return original(*args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(torch, "empty", allocate)
        with torch.cuda.stream(ambient):
            result = api.indexer_top_k_varlen_wrapper(s, lengths, 512, stream=consumer)
    assert records == [(s.device.index, consumer.cuda_stream)]
    consumer.synchronize()
    assert_topk(s, lengths, result["indices"], 512)


@pytest.mark.L0
def test_preserved_device_source_identity_and_prefix():
    import ast
    import hashlib
    import json
    import cudnn

    folder = Path(cudnn.__file__).resolve().parent / "deepseek_sparse_attention/indexer_top_k"
    provenance = json.loads((folder / "varlen_provenance.json").read_text())
    source = (folder / "varlen_kernel.py").read_text()
    device = source[source.index("THREADS = ") : source.index("# fmt: on")].rstrip()
    assert hashlib.sha256(device.encode()).hexdigest() == provenance["preserved_device_block_sha256"]
    tree = ast.parse(source)
    index = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.FunctionDef) and node.name == "_topk_kernel")
    assert ast.unparse(tree.body[index + 1]) == "_topk_kernel.set_name_prefix('cudnn', remove_cutlass_symbol=True)"
