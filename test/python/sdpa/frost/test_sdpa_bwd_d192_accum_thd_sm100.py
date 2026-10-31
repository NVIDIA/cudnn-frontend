# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native D192/V128 packed accumulation through graph and standalone APIs."""

import pytest
import torch

from d128_thd_test_utils import (
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
ENGINE = "sdpa_bwd_sm100_d192"


def _case(q=(129, 0, 65), kv=(3, 257, 0), **options):
    return make_case(q, kv, d_qk=192, d_v=128, **options)


def _adapter(case):
    from cudnn.sdpa.bwd.api_dsl_sm100_d192 import SdpaBwdDslSm100D192

    return make_adapter(case, SdpaBwdDslSm100D192)


def _graph(case):
    return make_graph(case, ENGINE, 20612)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["TH1", "HN"])
def test_graph_independent_origins_tails_and_empty(dtype, causal, layout):
    _require_accumulation()
    case = _case(dtype=dtype, causal=causal, stats_layout=layout, units="elements")
    graph, refs = _graph(case)
    workspace = make_workspace(graph.get_workspace_size())
    poison(case, workspace)
    graph.execute(variant_pack(case, refs), workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize("prefix_q,prefix_kv", [(False, False), (False, True), (True, False), (True, True)])
def test_standalone_compact_and_prefix_forms(prefix_q, prefix_kv):
    _require_accumulation()
    case = _case(origin_ports=(), dtype=torch.float16, causal=True, stats_layout="HN", gaps=False)
    api = _adapter(case)
    workspace = make_workspace(api.scratch_workspace_bytes())
    args = adapter_arguments(case)
    for name, lens, prefix in (("seq_q_lens", case.lens_q, prefix_q), ("seq_kv_lens", case.lens_kv, prefix_kv)):
        if prefix:
            values = [37]
            for length in lens:
                values.append(values[-1] + length)
            args[name] = torch.tensor(values, dtype=torch.int32, device="cuda")
    poison(case, workspace)
    api.execute(**args, workspace=workspace.tensor)
    check_case(case, workspace)


@pytest.mark.parametrize("dtype,causal,layout", [(torch.bfloat16, False, "TH1"), (torch.float16, True, "HN")])
@pytest.mark.parametrize("standalone", [False, True], ids=["graph", "standalone"])
def test_rebind_origins_changed_lengths_and_empty_replay(dtype, causal, layout, standalone):
    _require_accumulation()
    options = dict(dtype=dtype, causal=causal, stats_layout=layout, envelope=(257, 257), capacities=(1024, 1024))
    old = _case(**options)
    fresh = _case((1, 127, 0), (129, 0, 65), seed=157, layout_shift=7, **options)
    empty = _case((0, 0, 0), (0, 0, 0), seed=163, layout_shift=11, **options)
    if standalone:
        api = _adapter(old)
        workspace = make_workspace(api.scratch_workspace_bytes())

        def execute(case):
            api.execute(**adapter_arguments(case), workspace=workspace.tensor)

    else:
        graph, refs = _graph(old)
        workspace = make_workspace(graph.get_workspace_size())

        def execute(case):
            graph.execute(variant_pack(case, refs), workspace.tensor)

    poison(old, workspace)
    execute(old)
    check_case(old, workspace)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture, stream=stream):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                execute(old)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.current_stream().wait_stream(stream)
        copy_case_storage(old, fresh)
        poison(old, workspace)
        capture.replay()
        check_case(old, workspace)
        poison(fresh, workspace)
        execute(fresh)
        check_case(fresh, workspace)
        # Rebinding must not alter the old captured pointers or maps.
        poison(old, workspace)
        capture.replay()
        check_case(old, workspace)
        copy_case_storage(old, empty)
        poison(old, workspace)
        capture.replay()
        check_case(old, workspace)
    finally:
        capture.reset()


def test_short_workspace_and_output_alias_reject_before_launch():
    _require_accumulation()
    case = _case(dtype=torch.bfloat16)
    api = _adapter(case)
    workspace = make_workspace(api.scratch_workspace_bytes())
    args = adapter_arguments(case)
    poison(case, workspace)
    before = workspace.raw.clone()
    with pytest.raises(ValueError, match="workspace"):
        api.execute(**args, workspace=workspace.tensor[:-1])
    torch.testing.assert_close(workspace.raw, before, rtol=0, atol=0)
    # dQ and Q have the same width; aliasing is invalid despite valid geometry.
    with pytest.raises(ValueError, match="overlap"):
        api.execute(**(args | {"dq_tensor": case.tensors["q"]}), workspace=workspace.tensor)
    torch.testing.assert_close(workspace.raw, before, rtol=0, atol=0)
    api.execute(**args, workspace=workspace.tensor)
    check_case(case, workspace)


def test_replan_reloads_artifact_with_changed_capacities(monkeypatch, tmp_path):
    """Packed bounds and Stats pitch change without tracing a new kernel."""
    _require_accumulation()
    import cutlass.cute as cute
    from cudnn.frost import compiled_cache

    monkeypatch.setenv("CUDNN_FRONTEND_DISABLE_COMPILED_CACHE", "0")
    monkeypatch.setenv("CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO", "0")
    monkeypatch.setattr(compiled_cache, "_dir_override", tmp_path / "compiled")
    small = _case(stats_layout="HN", envelope=(257, 257), capacities=(1024, 1024))
    large = _case((385, 0, 65), (3, 513, 0), stats_layout="HN", envelope=(513, 513), capacities=(2048, 2048), seed=173)
    first = _adapter(small)

    def no_compile(*args, **kwargs):
        raise AssertionError("runtime capacities or Stats pitch caused recompilation")

    monkeypatch.setattr(cute, "compile", no_compile)
    hits = compiled_cache.stats()["hits"]
    second, third = _adapter(large), _adapter(small)
    assert compiled_cache.stats()["hits"] >= hits + 2
    for api, case in ((first, small), (second, large), (third, small)):
        workspace = make_workspace(api.scratch_workspace_bytes())
        args = adapter_arguments(case)
        poison(case, workspace)
        api.execute(**args, workspace=workspace.tensor)
        check_case(case, workspace)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        capture = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(capture, stream=stream):
                api.execute(**args, workspace=workspace.tensor)
            torch.cuda.current_stream().wait_stream(stream)
            poison(case, workspace)
            capture.replay()
            check_case(case, workspace)
        finally:
            capture.reset()
